//! B-slice macroblock decode (CAVLC) — ITU-T H.264 §7.3.5 / §8.4.
//!
//! This mirrors [`crate::p_mb`] for the B-slice case:
//!
//! * B_Direct_16x16 and B_Skip — MVs / ref-idx derived via spatial direct
//!   prediction (§8.4.1.2.2). Temporal direct (§8.4.1.2.3) is not yet
//!   implemented and is rejected by the slice header handler.
//! * B_L0_16x16 / B_L1_16x16 / B_Bi_16x16 — whole-MB single- or
//!   bi-directional prediction with explicit MVDs (ref_idx_lN is coded
//!   per list).
//! * B_L0_L0_16x8 / B_L1_L1_16x8 / B_L0_L1_16x8 / B_L1_L0_16x8 /
//!   B_L0_Bi_16x8 / B_L1_Bi_16x8 / B_Bi_L0_16x8 / B_Bi_L1_16x8 /
//!   B_Bi_Bi_16x8 and the 8x16 transposes thereof.
//! * B_8x8 — per-sub-MB `sub_mb_type` (Table 7-17 B entries) including
//!   B_Direct_8x8, B_{L0,L1,Bi}_8x8, B_{L0,L1,Bi}_8x4, B_{L0,L1,Bi}_4x8,
//!   B_{L0,L1,Bi}_4x4.
//! * Intra-in-B macroblocks dispatch back into the I-slice decode helper.
//!
//! Bi-prediction samples default to `(pred_L0 + pred_L1 + 1) >> 1`
//! (§8.4.2.3.1). When the slice header carries an explicit `pred_weight_table`
//! (`weighted_bipred_idc == 1`) the L0 and L1 predictions are combined using
//! per-reference weights + offsets per §8.4.2.3.2. Implicit bipred
//! (`weighted_bipred_idc == 2`) is out of scope and surfaces as
//! `Error::Unsupported` from the slice-header handler.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::cavlc::{decode_residual_block, BlockKind};
use crate::mb::LUMA_BLOCK_RASTER;
use crate::mb_type::{
    decode_b_slice_mb_type, decode_b_sub_mb_type, decode_cbp_inter, BMbType, BPartition,
    BSubPartition, PredDir,
};
use crate::motion::{
    apply_chroma_weight, apply_luma_weight, average_bipred, chroma_mc, luma_mc, predict_mv_list,
    weighted_bipred_plane,
};
use crate::picture::{MbInfo, Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::{ChromaWeight, LumaWeight, PredWeightTable, SliceHeader};
use crate::sps::Sps;
use crate::transform::{chroma_qp, dequantize_4x4, idct_4x4, inv_hadamard_2x2_chroma_dc};

/// Decode a single coded B-slice macroblock at `(mb_x, mb_y)`. `B_Skip`
/// macroblocks are emitted separately by [`decode_b_skip_mb`] from the
/// outer loop (CAVLC signals skips via `mb_skip_run` preceding the
/// macroblock).
pub fn decode_b_slice_mb(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    prev_qp: &mut i32,
) -> Result<()> {
    let mb_type = br.read_ue()?;
    let bmb = decode_b_slice_mb_type(mb_type)
        .ok_or_else(|| Error::invalid(format!("h264 b-slice: bad B mb_type {mb_type}")))?;
    match bmb {
        BMbType::IntraInB(imb) => {
            // Intra MB inside a B-slice — same syntax + reconstruct as I.
            crate::mb::decode_intra_mb_given_imb(br, sps, pps, sh, mb_x, mb_y, pic, prev_qp, imb)
        }
        BMbType::Inter { partition } => decode_b_inter_mb(
            br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, ref_list1, prev_qp, partition,
        ),
    }
}

/// B_Skip derivation — §8.4.1.2.  Same syntax + motion derivation as
/// `B_Direct_16x16` with `cbp_luma = cbp_chroma = 0`, no QP delta.
pub fn decode_b_skip_mb(
    sh: &SliceHeader,
    sps: &Sps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    prev_qp: i32,
) -> Result<()> {
    if !sh.direct_spatial_mv_pred_flag {
        return Err(Error::unsupported(
            "h264 b-slice: temporal direct MV prediction (§8.4.1.2.3) not implemented",
        ));
    }
    // Initialise MB info for the direct MB.
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        *info = MbInfo {
            qp_y: prev_qp,
            coded: true,
            intra: false,
            skipped: true,
            mb_type_p: None,
            p_partition: None,
            ..Default::default()
        };
    }
    // Derive + compensate four 8×8 direct sub-MBs.
    direct_16x16_spatial_compensate(sh, sps, mb_x, mb_y, pic, ref_list0, ref_list1)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Main inter-MB decode path (non-intra, non-skip).
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_b_inter_mb(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    prev_qp: &mut i32,
    partition: BPartition,
) -> Result<()> {
    // Pre-zero the MB's MV state so stale L1 data from the previous MB
    // doesn't leak into neighbours (deblocking / neighbour lookups).
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        *info = MbInfo {
            qp_y: *prev_qp,
            coded: true,
            intra: false,
            skipped: false,
            ..Default::default()
        };
    }

    match partition {
        BPartition::Direct16x16 => {
            direct_16x16_spatial_compensate(sh, sps, mb_x, mb_y, pic, ref_list0, ref_list1)?;
        }
        BPartition::L0_16x16 => {
            decode_16x16_single_dir(br, sh, mb_x, mb_y, pic, ref_list0, ref_list1, PredDir::L0)?;
        }
        BPartition::L1_16x16 => {
            decode_16x16_single_dir(br, sh, mb_x, mb_y, pic, ref_list0, ref_list1, PredDir::L1)?;
        }
        BPartition::Bi_16x16 => {
            decode_16x16_single_dir(
                br,
                sh,
                mb_x,
                mb_y,
                pic,
                ref_list0,
                ref_list1,
                PredDir::BiPred,
            )?;
        }
        BPartition::TwoPart16x8 { dirs } => {
            let rects = [(0usize, 0usize, 2usize, 4usize), (2, 0, 2, 4)];
            decode_two_partition_b(br, sh, mb_x, mb_y, pic, ref_list0, ref_list1, &rects, &dirs)?;
        }
        BPartition::TwoPart8x16 { dirs } => {
            let rects = [(0usize, 0usize, 4usize, 2usize), (0, 2, 4, 2)];
            decode_two_partition_b(br, sh, mb_x, mb_y, pic, ref_list0, ref_list1, &rects, &dirs)?;
        }
        BPartition::B8x8 => {
            decode_b8x8(br, sh, sps, mb_x, mb_y, pic, ref_list0, ref_list1)?;
        }
    }

    // CBP + residual (identical layout to P-slice inter).
    let cbp_raw = br.read_ue()?;
    let (cbp_luma, cbp_chroma) = decode_cbp_inter(cbp_raw)
        .ok_or_else(|| Error::invalid(format!("h264 b-slice: bad inter CBP {cbp_raw}")))?;
    let needs_qp = cbp_luma != 0 || cbp_chroma != 0;
    if needs_qp {
        let dqp = br.read_se()?;
        *prev_qp = ((*prev_qp + dqp + 52) % 52).clamp(0, 51);
    }
    let qp_y = *prev_qp;
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        info.qp_y = qp_y;
        info.intra4x4_pred_mode = [INTRA_DC_FAKE; 16];
    }
    decode_inter_residual_luma(br, mb_x, mb_y, pic, cbp_luma, qp_y)?;
    decode_inter_residual_chroma(br, pps, mb_x, mb_y, pic, cbp_chroma, qp_y)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// 16×16 single-direction partition (L0 / L1 / Bi).
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_16x16_single_dir(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    dir: PredDir,
) -> Result<()> {
    let num_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let num_l1 = sh.num_ref_idx_l1_active_minus1 + 1;
    let (ref_l0, ref_l1) = match dir {
        PredDir::L0 => (Some(read_ref_idx(br, num_l0)?), None),
        PredDir::L1 => (None, Some(read_ref_idx(br, num_l1)?)),
        PredDir::BiPred => (
            Some(read_ref_idx(br, num_l0)?),
            Some(read_ref_idx(br, num_l1)?),
        ),
    };
    let mvd_l0 = if ref_l0.is_some() {
        Some((br.read_se()? as i32, br.read_se()? as i32))
    } else {
        None
    };
    let mvd_l1 = if ref_l1.is_some() {
        Some((br.read_se()? as i32, br.read_se()? as i32))
    } else {
        None
    };
    compensate_partition_b(
        sh, pic, ref_list0, ref_list1, mb_x, mb_y, 0, 0, 4, 4, dir, ref_l0, ref_l1, mvd_l0, mvd_l1,
    )
}

// ---------------------------------------------------------------------------
// 16×8 / 8×16 pair — two partitions each with its own direction.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_two_partition_b(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    rects: &[(usize, usize, usize, usize); 2],
    dirs: &[PredDir; 2],
) -> Result<()> {
    let num_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let num_l1 = sh.num_ref_idx_l1_active_minus1 + 1;

    // Spec order: first all ref_idx_l0, then all ref_idx_l1, then all mvd_l0,
    // then all mvd_l1 — but only for partitions whose direction includes the
    // corresponding list.
    let mut ref_l0 = [None; 2];
    let mut ref_l1 = [None; 2];
    let mut mvd_l0 = [None; 2];
    let mut mvd_l1 = [None; 2];

    for p in 0..2 {
        if uses_l0(dirs[p]) {
            ref_l0[p] = Some(read_ref_idx(br, num_l0)?);
        }
    }
    for p in 0..2 {
        if uses_l1(dirs[p]) {
            ref_l1[p] = Some(read_ref_idx(br, num_l1)?);
        }
    }
    for p in 0..2 {
        if uses_l0(dirs[p]) {
            mvd_l0[p] = Some((br.read_se()? as i32, br.read_se()? as i32));
        }
    }
    for p in 0..2 {
        if uses_l1(dirs[p]) {
            mvd_l1[p] = Some((br.read_se()? as i32, br.read_se()? as i32));
        }
    }

    for p in 0..2 {
        let (r0, c0, ph, pw) = rects[p];
        compensate_partition_b(
            sh, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, ph, pw, dirs[p], ref_l0[p],
            ref_l1[p], mvd_l0[p], mvd_l1[p],
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// B_8x8 — four 8×8 sub-MBs, each with its own sub_mb_type.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_b8x8(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    sps: &Sps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
) -> Result<()> {
    const SUB_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 2), (2, 0), (2, 2)];
    let num_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let num_l1 = sh.num_ref_idx_l1_active_minus1 + 1;

    let mut subs = [BSubPartition::Direct8x8; 4];
    for s in 0..4 {
        let v = br.read_ue()?;
        subs[s] = decode_b_sub_mb_type(v)
            .ok_or_else(|| Error::invalid(format!("h264 b-slice: bad sub_mb_type {v}")))?;
    }
    // Per §7.3.5.2: ref_idx_l0 for non-direct sub-MBs using L0/BiPred, then
    // ref_idx_l1 for non-direct using L1/BiPred, then mvd_l0, then mvd_l1.
    let mut ref_l0 = [None; 4];
    let mut ref_l1 = [None; 4];
    for s in 0..4 {
        if let Some(dir) = subs[s].pred_dir() {
            if uses_l0(dir) {
                ref_l0[s] = Some(read_ref_idx(br, num_l0)?);
            }
        }
    }
    for s in 0..4 {
        if let Some(dir) = subs[s].pred_dir() {
            if uses_l1(dir) {
                ref_l1[s] = Some(read_ref_idx(br, num_l1)?);
            }
        }
    }

    // Read all mvds up front, partition-by-partition, so neighbour MVs from
    // earlier sub-MBs are in place before later ones predict.
    #[derive(Clone, Copy, Default)]
    struct SubMvd {
        mvd_l0: Option<(i32, i32)>,
        mvd_l1: Option<(i32, i32)>,
    }
    let mut mvds: Vec<Vec<SubMvd>> = Vec::with_capacity(4);

    for s in 0..4 {
        let sp = subs[s];
        let n = sp.num_sub_partitions();
        let mut entry = Vec::with_capacity(n);
        let dir_opt = sp.pred_dir();
        for _ in 0..n {
            if let Some(dir) = dir_opt {
                let mvd_l0 = if uses_l0(dir) {
                    Some((br.read_se()? as i32, br.read_se()? as i32))
                } else {
                    None
                };
                let mvd_l1 = if uses_l1(dir) {
                    Some((br.read_se()? as i32, br.read_se()? as i32))
                } else {
                    None
                };
                entry.push(SubMvd { mvd_l0, mvd_l1 });
            } else {
                // B_Direct_8x8 — no MVD in bitstream.
                entry.push(SubMvd::default());
            }
        }
        mvds.push(entry);
    }

    // Per 8×8 sub-MB, walk its sub-partitions and do MC.
    for s in 0..4 {
        let sp = subs[s];
        let (sr0, sc0) = SUB_OFFSETS[s];
        if matches!(sp, BSubPartition::Direct8x8) {
            direct_8x8_spatial_compensate(
                sh, sps, mb_x, mb_y, sr0, sc0, pic, ref_list0, ref_list1,
            )?;
            continue;
        }
        let dir = sp.pred_dir().expect("non-direct");
        for sub_idx in 0..sp.num_sub_partitions() {
            let (dr, dc, sh_h, sh_w) = sp.sub_rect(sub_idx);
            let r0 = sr0 + dr;
            let c0 = sc0 + dc;
            let m = mvds[s][sub_idx];
            compensate_partition_b(
                sh, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, sh_h, sh_w, dir, ref_l0[s],
                ref_l1[s], m.mvd_l0, m.mvd_l1,
            )?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Partition-level motion compensation (L0, L1, or Bi).
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn compensate_partition_b(
    sh: &SliceHeader,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    ph: usize,
    pw: usize,
    dir: PredDir,
    ref_idx_l0: Option<i8>,
    ref_idx_l1: Option<i8>,
    mvd_l0: Option<(i32, i32)>,
    mvd_l1: Option<(i32, i32)>,
) -> Result<()> {
    // Derive per-list MVs by adding MVDs to list-specific median predictors.
    let mv_l0 = if let (Some(r), Some(d)) = (ref_idx_l0, mvd_l0) {
        let pmv = predict_mv_list(pic, mb_x, mb_y, r0, c0, ph, pw, r, 0);
        Some((
            pmv.0.wrapping_add(d.0 as i16),
            pmv.1.wrapping_add(d.1 as i16),
        ))
    } else {
        None
    };
    let mv_l1 = if let (Some(r), Some(d)) = (ref_idx_l1, mvd_l1) {
        let pmv = predict_mv_list(pic, mb_x, mb_y, r0, c0, ph, pw, r, 1);
        Some((
            pmv.0.wrapping_add(d.0 as i16),
            pmv.1.wrapping_add(d.1 as i16),
        ))
    } else {
        None
    };

    // Record MV state for subsequent neighbour lookups + deblocking.
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        for rr in r0..r0 + ph {
            for cc in c0..c0 + pw {
                let idx = rr * 4 + cc;
                if let (Some(r), Some(mv)) = (ref_idx_l0, mv_l0) {
                    info.ref_idx_l0[idx] = r;
                    info.mv_l0[idx] = mv;
                }
                if let (Some(r), Some(mv)) = (ref_idx_l1, mv_l1) {
                    info.ref_idx_l1[idx] = r;
                    info.mv_l1[idx] = mv;
                }
            }
        }
    }

    apply_motion_compensation(
        sh, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, ph, pw, dir, ref_idx_l0, ref_idx_l1,
        mv_l0, mv_l1,
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_motion_compensation(
    sh: &SliceHeader,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    ph: usize,
    pw: usize,
    dir: PredDir,
    ref_idx_l0: Option<i8>,
    ref_idx_l1: Option<i8>,
    mv_l0: Option<(i16, i16)>,
    mv_l1: Option<(i16, i16)>,
) -> Result<()> {
    match dir {
        PredDir::L0 => {
            let (r, mv) = (
                ref_idx_l0.ok_or_else(|| Error::invalid("b: L0 missing ref"))?,
                mv_l0.unwrap(),
            );
            let reference = lookup_ref(ref_list0, r, "L0")?;
            let (lw, cw) = list_weight(sh, 0, r);
            mc_luma_partition(
                pic,
                reference,
                mb_x,
                mb_y,
                r0,
                c0,
                ph,
                pw,
                mv.0 as i32,
                mv.1 as i32,
                lw,
            );
            mc_chroma_partition(
                pic,
                reference,
                mb_x,
                mb_y,
                r0,
                c0,
                ph,
                pw,
                mv.0 as i32,
                mv.1 as i32,
                cw,
            );
        }
        PredDir::L1 => {
            let (r, mv) = (
                ref_idx_l1.ok_or_else(|| Error::invalid("b: L1 missing ref"))?,
                mv_l1.unwrap(),
            );
            let reference = lookup_ref(ref_list1, r, "L1")?;
            let (lw, cw) = list_weight(sh, 1, r);
            mc_luma_partition(
                pic,
                reference,
                mb_x,
                mb_y,
                r0,
                c0,
                ph,
                pw,
                mv.0 as i32,
                mv.1 as i32,
                lw,
            );
            mc_chroma_partition(
                pic,
                reference,
                mb_x,
                mb_y,
                r0,
                c0,
                ph,
                pw,
                mv.0 as i32,
                mv.1 as i32,
                cw,
            );
        }
        PredDir::BiPred => {
            let r0_idx = ref_idx_l0.unwrap();
            let r1_idx = ref_idx_l1.unwrap();
            let mv0 = mv_l0.unwrap();
            let mv1 = mv_l1.unwrap();
            let ref0 = lookup_ref(ref_list0, r0_idx, "L0")?;
            let ref1 = lookup_ref(ref_list1, r1_idx, "L1")?;
            let wt = sh.pred_weight_table.as_ref();
            let explicit = wt.is_some();
            mc_bipred_partition(
                pic,
                ref0,
                ref1,
                mb_x,
                mb_y,
                r0,
                c0,
                ph,
                pw,
                mv0.0 as i32,
                mv0.1 as i32,
                mv1.0 as i32,
                mv1.1 as i32,
                r0_idx,
                r1_idx,
                wt,
                explicit,
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// MC helpers — write directly into `pic` planes.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn mc_luma_partition(
    pic: &mut Picture,
    reference: &Picture,
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    h: usize,
    w: usize,
    mv_x_q: i32,
    mv_y_q: i32,
    luma_weight: Option<&LumaWeight>,
) {
    let pw = w * 4;
    let ph = h * 4;
    let mut tmp = vec![0u8; pw * ph];
    let base_x = (mb_x as i32) * 16 + (c0 as i32) * 4;
    let base_y = (mb_y as i32) * 16 + (r0 as i32) * 4;
    luma_mc(&mut tmp, reference, base_x, base_y, mv_x_q, mv_y_q, pw, ph);
    if let Some(lw) = luma_weight {
        apply_luma_weight(&mut tmp, lw);
    }
    let lstride = pic.luma_stride();
    let off = pic.luma_off(mb_x, mb_y) + r0 * 4 * lstride + c0 * 4;
    for rr in 0..ph {
        for cc in 0..pw {
            pic.y[off + rr * lstride + cc] = tmp[rr * pw + cc];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mc_chroma_partition(
    pic: &mut Picture,
    reference: &Picture,
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    h: usize,
    w: usize,
    mv_x_q: i32,
    mv_y_q: i32,
    chroma_weight: Option<&ChromaWeight>,
) {
    let pw = w * 2;
    let ph = h * 2;
    let mut tmp_cb = vec![0u8; pw * ph];
    let mut tmp_cr = vec![0u8; pw * ph];
    let base_x = (mb_x as i32) * 8 + (c0 as i32) * 2;
    let base_y = (mb_y as i32) * 8 + (r0 as i32) * 2;
    let cstride = reference.chroma_stride();
    let cw = (reference.width / 2) as i32;
    let ch = (reference.height / 2) as i32;
    chroma_mc(
        &mut tmp_cb,
        &reference.cb,
        cstride,
        cw,
        ch,
        base_x,
        base_y,
        mv_x_q,
        mv_y_q,
        pw,
        ph,
    );
    chroma_mc(
        &mut tmp_cr,
        &reference.cr,
        cstride,
        cw,
        ch,
        base_x,
        base_y,
        mv_x_q,
        mv_y_q,
        pw,
        ph,
    );
    if let Some(cw_entry) = chroma_weight {
        apply_chroma_weight(&mut tmp_cb, cw_entry, 0);
        apply_chroma_weight(&mut tmp_cr, cw_entry, 1);
    }
    let stride = pic.chroma_stride();
    let off = pic.chroma_off(mb_x, mb_y) + r0 * 2 * stride + c0 * 2;
    for rr in 0..ph {
        for cc in 0..pw {
            pic.cb[off + rr * stride + cc] = tmp_cb[rr * pw + cc];
            pic.cr[off + rr * stride + cc] = tmp_cr[rr * pw + cc];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mc_bipred_partition(
    pic: &mut Picture,
    ref0: &Picture,
    ref1: &Picture,
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    h: usize,
    w: usize,
    mv0_x_q: i32,
    mv0_y_q: i32,
    mv1_x_q: i32,
    mv1_y_q: i32,
    ref_idx_l0: i8,
    ref_idx_l1: i8,
    pwt: Option<&PredWeightTable>,
    explicit: bool,
) {
    let lpw = w * 4;
    let lph = h * 4;
    let mut tmp_l0 = vec![0u8; lpw * lph];
    let mut tmp_l1 = vec![0u8; lpw * lph];
    let base_lx = (mb_x as i32) * 16 + (c0 as i32) * 4;
    let base_ly = (mb_y as i32) * 16 + (r0 as i32) * 4;
    luma_mc(
        &mut tmp_l0,
        ref0,
        base_lx,
        base_ly,
        mv0_x_q,
        mv0_y_q,
        lpw,
        lph,
    );
    luma_mc(
        &mut tmp_l1,
        ref1,
        base_lx,
        base_ly,
        mv1_x_q,
        mv1_y_q,
        lpw,
        lph,
    );

    let lstride = pic.luma_stride();
    let loff = pic.luma_off(mb_x, mb_y) + r0 * 4 * lstride + c0 * 4;
    let mut luma_out = vec![0u8; lpw * lph];
    if explicit {
        let pwt = pwt.unwrap();
        let idx0 = ref_idx_l0.max(0) as usize;
        let idx1 = ref_idx_l1.max(0) as usize;
        let lw0 = pwt.luma_l0.get(idx0).copied().unwrap_or_default();
        let lw1 = pwt.luma_l1.get(idx1).copied().unwrap_or_default();
        weighted_bipred_plane(
            &mut luma_out,
            &tmp_l0,
            &tmp_l1,
            lw0.weight,
            lw0.offset,
            lw1.weight,
            lw1.offset,
            pwt.luma_log2_weight_denom,
        );
    } else {
        average_bipred(&mut luma_out, &tmp_l0, &tmp_l1);
    }
    for rr in 0..lph {
        for cc in 0..lpw {
            pic.y[loff + rr * lstride + cc] = luma_out[rr * lpw + cc];
        }
    }

    // Chroma
    let cpw = w * 2;
    let cph = h * 2;
    let mut tmp_cb0 = vec![0u8; cpw * cph];
    let mut tmp_cb1 = vec![0u8; cpw * cph];
    let mut tmp_cr0 = vec![0u8; cpw * cph];
    let mut tmp_cr1 = vec![0u8; cpw * cph];
    let base_cx = (mb_x as i32) * 8 + (c0 as i32) * 2;
    let base_cy = (mb_y as i32) * 8 + (r0 as i32) * 2;
    let cstride0 = ref0.chroma_stride();
    let cstride1 = ref1.chroma_stride();
    let cw0 = (ref0.width / 2) as i32;
    let ch0 = (ref0.height / 2) as i32;
    let cw1 = (ref1.width / 2) as i32;
    let ch1 = (ref1.height / 2) as i32;
    chroma_mc(
        &mut tmp_cb0,
        &ref0.cb,
        cstride0,
        cw0,
        ch0,
        base_cx,
        base_cy,
        mv0_x_q,
        mv0_y_q,
        cpw,
        cph,
    );
    chroma_mc(
        &mut tmp_cr0,
        &ref0.cr,
        cstride0,
        cw0,
        ch0,
        base_cx,
        base_cy,
        mv0_x_q,
        mv0_y_q,
        cpw,
        cph,
    );
    chroma_mc(
        &mut tmp_cb1,
        &ref1.cb,
        cstride1,
        cw1,
        ch1,
        base_cx,
        base_cy,
        mv1_x_q,
        mv1_y_q,
        cpw,
        cph,
    );
    chroma_mc(
        &mut tmp_cr1,
        &ref1.cr,
        cstride1,
        cw1,
        ch1,
        base_cx,
        base_cy,
        mv1_x_q,
        mv1_y_q,
        cpw,
        cph,
    );

    let mut cb_out = vec![0u8; cpw * cph];
    let mut cr_out = vec![0u8; cpw * cph];
    if explicit {
        let pwt = pwt.unwrap();
        let idx0 = ref_idx_l0.max(0) as usize;
        let idx1 = ref_idx_l1.max(0) as usize;
        let cw_l0 = pwt.chroma_l0.get(idx0).copied().unwrap_or_default();
        let cw_l1 = pwt.chroma_l1.get(idx1).copied().unwrap_or_default();
        weighted_bipred_plane(
            &mut cb_out,
            &tmp_cb0,
            &tmp_cb1,
            cw_l0.weight[0],
            cw_l0.offset[0],
            cw_l1.weight[0],
            cw_l1.offset[0],
            pwt.chroma_log2_weight_denom,
        );
        weighted_bipred_plane(
            &mut cr_out,
            &tmp_cr0,
            &tmp_cr1,
            cw_l0.weight[1],
            cw_l0.offset[1],
            cw_l1.weight[1],
            cw_l1.offset[1],
            pwt.chroma_log2_weight_denom,
        );
    } else {
        average_bipred(&mut cb_out, &tmp_cb0, &tmp_cb1);
        average_bipred(&mut cr_out, &tmp_cr0, &tmp_cr1);
    }
    let stride = pic.chroma_stride();
    let coff = pic.chroma_off(mb_x, mb_y) + r0 * 2 * stride + c0 * 2;
    for rr in 0..cph {
        for cc in 0..cpw {
            pic.cb[coff + rr * stride + cc] = cb_out[rr * cpw + cc];
            pic.cr[coff + rr * stride + cc] = cr_out[rr * cpw + cc];
        }
    }
}

// ---------------------------------------------------------------------------
// Spatial direct MV prediction — §8.4.1.2.2.
// ---------------------------------------------------------------------------

/// Derive the spatially-predicted `(ref_idx_l0, ref_idx_l1)` for a direct
/// macroblock from the A (left), B (top), C (top-right, fallback D) 16×16
/// neighbours. The rule (§8.4.1.2.1) is: pick the minimum non-negative
/// ref_idx per list across the three neighbours; `-1` means "list not used".
fn direct_spatial_ref_idx(pic: &Picture, mb_x: u32, mb_y: u32) -> (i8, i8) {
    // Neighbour 4×4 blocks at mb-boundary.
    let neighbours_l0 = [
        crate::motion::neighbour_mv_list(pic, mb_x as i32, mb_y as i32, 0, -1, 0),
        crate::motion::neighbour_mv_list(pic, mb_x as i32, mb_y as i32, -1, 0, 0),
        crate::motion::neighbour_mv_list(pic, mb_x as i32, mb_y as i32, -1, 4, 0)
            .or_else(|| crate::motion::neighbour_mv_list(pic, mb_x as i32, mb_y as i32, -1, -1, 0)),
    ];
    let neighbours_l1 = [
        crate::motion::neighbour_mv_list(pic, mb_x as i32, mb_y as i32, 0, -1, 1),
        crate::motion::neighbour_mv_list(pic, mb_x as i32, mb_y as i32, -1, 0, 1),
        crate::motion::neighbour_mv_list(pic, mb_x as i32, mb_y as i32, -1, 4, 1)
            .or_else(|| crate::motion::neighbour_mv_list(pic, mb_x as i32, mb_y as i32, -1, -1, 1)),
    ];

    fn min_ref(items: &[Option<((i16, i16), i8)>]) -> i8 {
        let mut m: Option<i8> = None;
        for it in items {
            if let Some((_, r)) = it {
                if *r >= 0 {
                    m = Some(m.map_or(*r, |cur| cur.min(*r)));
                }
            }
        }
        m.unwrap_or(-1)
    }
    let r0 = min_ref(&neighbours_l0);
    let r1 = min_ref(&neighbours_l1);
    // Spec: if both are -1 (no inter neighbour), default to 0.
    if r0 < 0 && r1 < 0 {
        (0, 0)
    } else {
        (r0, r1)
    }
}

/// Spatial-direct MV derivation for a whole 16×16 MB — §8.4.1.2.2.
///
/// 1. Derive `(ref_idx_l0, ref_idx_l1)` from the neighbour list-N minimums.
/// 2. For each list with a non-negative ref_idx, take the median of the
///    neighbour MVs that reference that ref_idx; use (0,0) if none do.
/// 3. Apply MC from both lists (or one if the other ref_idx is -1).
///
/// This function writes reconstructed samples for the entire 16×16 MB.
fn direct_16x16_spatial_compensate(
    sh: &SliceHeader,
    _sps: &Sps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
) -> Result<()> {
    // Simplified direct prediction: derive a single MV pair for the whole
    // 16×16 and apply it. This is correct for the common case where the
    // colocated block is non-zero MV — the full spec also handles a
    // "zero MV collocated" shortcut which we approximate by always using
    // the median-derived MVs (a conservative choice that matches x264's
    // encoder output for well-behaved clips).
    let (ri0, ri1) = direct_spatial_ref_idx(pic, mb_x, mb_y);
    let (mv0, mv1) = direct_spatial_mvs(pic, mb_x, mb_y, ri0, ri1);
    let dir = pick_bipred_dir(ri0, ri1);
    for rr in 0..4usize {
        for cc in 0..4usize {
            let info = pic.mb_info_mut(mb_x, mb_y);
            info.ref_idx_l0[rr * 4 + cc] = ri0;
            info.ref_idx_l1[rr * 4 + cc] = ri1;
            info.mv_l0[rr * 4 + cc] = mv0;
            info.mv_l1[rr * 4 + cc] = mv1;
        }
    }
    apply_motion_compensation(
        sh,
        pic,
        ref_list0,
        ref_list1,
        mb_x,
        mb_y,
        0,
        0,
        4,
        4,
        dir,
        if ri0 >= 0 { Some(ri0) } else { None },
        if ri1 >= 0 { Some(ri1) } else { None },
        if ri0 >= 0 { Some(mv0) } else { None },
        if ri1 >= 0 { Some(mv1) } else { None },
    )
}

/// Spatial-direct MV derivation for a single 8×8 B_Direct_8x8 sub-MB.
/// Same as [`direct_16x16_spatial_compensate`] but scoped to 8×8.
#[allow(clippy::too_many_arguments)]
fn direct_8x8_spatial_compensate(
    sh: &SliceHeader,
    _sps: &Sps,
    mb_x: u32,
    mb_y: u32,
    sr0: usize,
    sc0: usize,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
) -> Result<()> {
    let (ri0, ri1) = direct_spatial_ref_idx(pic, mb_x, mb_y);
    let (mv0, mv1) = direct_spatial_mvs(pic, mb_x, mb_y, ri0, ri1);
    let dir = pick_bipred_dir(ri0, ri1);
    for rr in sr0..sr0 + 2 {
        for cc in sc0..sc0 + 2 {
            let info = pic.mb_info_mut(mb_x, mb_y);
            info.ref_idx_l0[rr * 4 + cc] = ri0;
            info.ref_idx_l1[rr * 4 + cc] = ri1;
            info.mv_l0[rr * 4 + cc] = mv0;
            info.mv_l1[rr * 4 + cc] = mv1;
        }
    }
    apply_motion_compensation(
        sh,
        pic,
        ref_list0,
        ref_list1,
        mb_x,
        mb_y,
        sr0,
        sc0,
        2,
        2,
        dir,
        if ri0 >= 0 { Some(ri0) } else { None },
        if ri1 >= 0 { Some(ri1) } else { None },
        if ri0 >= 0 { Some(mv0) } else { None },
        if ri1 >= 0 { Some(mv1) } else { None },
    )
}

fn direct_spatial_mvs(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    ref_idx_l0: i8,
    ref_idx_l1: i8,
) -> ((i16, i16), (i16, i16)) {
    let mv0 = if ref_idx_l0 >= 0 {
        predict_mv_list(pic, mb_x, mb_y, 0, 0, 4, 4, ref_idx_l0, 0)
    } else {
        (0, 0)
    };
    let mv1 = if ref_idx_l1 >= 0 {
        predict_mv_list(pic, mb_x, mb_y, 0, 0, 4, 4, ref_idx_l1, 1)
    } else {
        (0, 0)
    };
    (mv0, mv1)
}

fn pick_bipred_dir(r0: i8, r1: i8) -> PredDir {
    match (r0 >= 0, r1 >= 0) {
        (true, true) => PredDir::BiPred,
        (true, false) => PredDir::L0,
        (false, true) => PredDir::L1,
        (false, false) => PredDir::L0, // degenerate — treated as (0,0)/ref0
    }
}

// ---------------------------------------------------------------------------
// Residual decode — identical to P-slice (luma 4×4 + chroma DC + AC).
// ---------------------------------------------------------------------------

fn decode_inter_residual_luma(
    br: &mut BitReader<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    cbp_luma: u8,
    qp_y: i32,
) -> Result<()> {
    let lstride = pic.luma_stride();
    let lo_mb = pic.luma_off(mb_x, mb_y);
    for blk in 0..16usize {
        let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
        let eight_idx = (br_row / 2) * 2 + (br_col / 2);
        if (cbp_luma >> eight_idx) & 1 == 0 {
            continue;
        }
        let nc = predict_inter_nc_luma(pic, mb_x, mb_y, br_row, br_col);
        let blk = decode_residual_block(br, nc, BlockKind::Luma4x4)?;
        let mut residual = blk.coeffs;
        let total_coeff = blk.total_coeff;
        dequantize_4x4(&mut residual, qp_y);
        idct_4x4(&mut residual);
        let lo = lo_mb + br_row * 4 * lstride + br_col * 4;
        for r in 0..4 {
            for c in 0..4 {
                let base = pic.y[lo + r * lstride + c] as i32;
                pic.y[lo + r * lstride + c] = (base + residual[r * 4 + c]).clamp(0, 255) as u8;
            }
        }
        pic.mb_info_mut(mb_x, mb_y).luma_nc[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}

fn decode_inter_residual_chroma(
    br: &mut BitReader<'_>,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    cbp_chroma: u8,
    qp_y: i32,
) -> Result<()> {
    let qpc = chroma_qp(qp_y, pps.chroma_qp_index_offset);
    let mut dc_cb = [0i32; 4];
    let mut dc_cr = [0i32; 4];
    if cbp_chroma >= 1 {
        let blk = decode_residual_block(br, 0, BlockKind::ChromaDc2x2)?;
        dc_cb[..4].copy_from_slice(&blk.coeffs[..4]);
        let blk = decode_residual_block(br, 0, BlockKind::ChromaDc2x2)?;
        dc_cr[..4].copy_from_slice(&blk.coeffs[..4]);
        inv_hadamard_2x2_chroma_dc(&mut dc_cb, qpc);
        inv_hadamard_2x2_chroma_dc(&mut dc_cr, qpc);
    }
    let cstride = pic.chroma_stride();
    let co = pic.chroma_off(mb_x, mb_y);
    for plane_kind in [true, false] {
        let dc = if plane_kind { &dc_cb } else { &dc_cr };
        let mut nc_arr = [0u8; 4];
        for blk_idx in 0..4u8 {
            let br_row = (blk_idx >> 1) as usize;
            let br_col = (blk_idx & 1) as usize;
            let mut res = [0i32; 16];
            let mut total_coeff = 0u32;
            if cbp_chroma == 2 {
                let nc = predict_inter_nc_chroma(pic, mb_x, mb_y, plane_kind, br_row, br_col);
                let ac = decode_residual_block(br, nc, BlockKind::ChromaAc)?;
                total_coeff = ac.total_coeff;
                res = ac.coeffs;
                dequantize_4x4(&mut res, qpc);
            }
            res[0] = dc[(br_row << 1) | br_col];
            idct_4x4(&mut res);
            let off_in_mb = br_row * 4 * cstride + br_col * 4;
            let plane = if plane_kind { &mut pic.cb } else { &mut pic.cr };
            for r in 0..4 {
                for c in 0..4 {
                    let base = plane[co + off_in_mb + r * cstride + c] as i32;
                    let v = (base + res[r * 4 + c]).clamp(0, 255) as u8;
                    plane[co + off_in_mb + r * cstride + c] = v;
                }
            }
            nc_arr[(br_row << 1) | br_col] = total_coeff as u8;
            // Publish the in-progress nc totals back to `MbInfo` after every
            // block so the next block's `predict_inter_nc_chroma` sees the
            // up-to-date counts (§9.2.1.1 neighbour prediction).
            let info = pic.mb_info_mut(mb_x, mb_y);
            if plane_kind {
                info.cb_nc = nc_arr;
            } else {
                info.cr_nc = nc_arr;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Misc helpers.
// ---------------------------------------------------------------------------

fn uses_l0(dir: PredDir) -> bool {
    matches!(dir, PredDir::L0 | PredDir::BiPred)
}
fn uses_l1(dir: PredDir) -> bool {
    matches!(dir, PredDir::L1 | PredDir::BiPred)
}

fn read_ref_idx(br: &mut BitReader<'_>, num_ref: u32) -> Result<i8> {
    if num_ref <= 1 {
        return Ok(0);
    }
    if num_ref == 2 {
        let bit = br.read_u1()?;
        return Ok(if bit == 1 { 0 } else { 1 });
    }
    Ok(br.read_ue()? as i8)
}

fn lookup_ref<'a>(list: &[&'a Picture], ref_idx: i8, tag: &str) -> Result<&'a Picture> {
    if ref_idx < 0 {
        return Err(Error::invalid(format!(
            "h264 b-slice: negative ref_idx in {tag}"
        )));
    }
    let idx = ref_idx as usize;
    list.get(idx).copied().ok_or_else(|| {
        Error::invalid(format!(
            "h264 b-slice: ref_idx {ref_idx} out of RefPicList{tag} range (len={})",
            list.len()
        ))
    })
}

fn list_weight<'a>(
    sh: &'a SliceHeader,
    list: u8,
    ref_idx: i8,
) -> (Option<&'a LumaWeight>, Option<&'a ChromaWeight>) {
    let Some(tbl) = sh.pred_weight_table.as_ref() else {
        return (None, None);
    };
    let idx = ref_idx.max(0) as usize;
    if list == 0 {
        (tbl.luma_l0.get(idx), tbl.chroma_l0.get(idx))
    } else {
        (tbl.luma_l1.get(idx), tbl.chroma_l1.get(idx))
    }
}

fn predict_inter_nc_luma(pic: &Picture, mb_x: u32, mb_y: u32, br_row: usize, br_col: usize) -> i32 {
    let info_here = pic.mb_info_at(mb_x, mb_y);
    let left = if br_col > 0 {
        Some(info_here.luma_nc[br_row * 4 + br_col - 1])
    } else if mb_x > 0 {
        let info = pic.mb_info_at(mb_x - 1, mb_y);
        if info.coded {
            Some(info.luma_nc[br_row * 4 + 3])
        } else {
            None
        }
    } else {
        None
    };
    let top = if br_row > 0 {
        Some(info_here.luma_nc[(br_row - 1) * 4 + br_col])
    } else if mb_y > 0 {
        let info = pic.mb_info_at(mb_x, mb_y - 1);
        if info.coded {
            Some(info.luma_nc[12 + br_col])
        } else {
            None
        }
    } else {
        None
    };
    match (left, top) {
        (Some(l), Some(t)) => (l as i32 + t as i32 + 1) >> 1,
        (Some(l), None) => l as i32,
        (None, Some(t)) => t as i32,
        (None, None) => 0,
    }
}

fn predict_inter_nc_chroma(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    cb: bool,
    br_row: usize,
    br_col: usize,
) -> i32 {
    let pick = |info: &MbInfo, sub: usize| -> u8 {
        if cb {
            info.cb_nc[sub]
        } else {
            info.cr_nc[sub]
        }
    };
    let info_here = pic.mb_info_at(mb_x, mb_y);
    let left = if br_col > 0 {
        Some(pick(info_here, br_row * 2 + br_col - 1))
    } else if mb_x > 0 {
        let info = pic.mb_info_at(mb_x - 1, mb_y);
        if info.coded {
            Some(pick(info, br_row * 2 + 1))
        } else {
            None
        }
    } else {
        None
    };
    let top = if br_row > 0 {
        Some(pick(info_here, (br_row - 1) * 2 + br_col))
    } else if mb_y > 0 {
        let info = pic.mb_info_at(mb_x, mb_y - 1);
        if info.coded {
            Some(pick(info, 2 + br_col))
        } else {
            None
        }
    } else {
        None
    };
    match (left, top) {
        (Some(l), Some(t)) => (l as i32 + t as i32 + 1) >> 1,
        (Some(l), None) => l as i32,
        (None, Some(t)) => t as i32,
        (None, None) => 0,
    }
}
