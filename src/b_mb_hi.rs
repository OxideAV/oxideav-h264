//! 10-bit CAVLC B-slice macroblock decode — ITU-T H.264 §7.3.5 / §8.4 on
//! u16 planes.
//!
//! Mirrors [`crate::b_mb`] but writes `Picture::y16 / cb16 / cr16` and
//! routes motion compensation through the u16 helpers in [`crate::motion`]
//! (`luma_mc_hi`, `chroma_mc_hi`, `apply_luma_weight_hi`,
//! `apply_chroma_weight_hi`). Bipred blends use the u16 variants in this
//! module — `(predL0 + predL1 + 1) >> 1` clipped to `(1 << BitDepth) - 1`
//! for the default case, with §8.4.2.3.2 explicit / §8.4.2.3.3 implicit
//! variants carrying the same Clip1 bound.
//!
//! Residual dequant uses the `*_scaled_ext` i64-widened helpers so the
//! extended `QpY + QpBdOffsetY` range (up to 63 at 10-bit) doesn't
//! overflow; `mb_qp_delta` wraps modulo `52 + QpBdOffsetY`; intra-in-B
//! dispatches back into [`crate::mb_hi::decode_intra_mb_given_imb_hi`].
//!
//! Scope: 4:2:0 CAVLC at 10-bit. All B_Direct / B_L0 / B_L1 / B_Bi shapes
//! for 16×16, 16×8, 8×16, B_8×8 and B_Skip. The 8×8 transform at 10-bit
//! for B slices is scoped out (matches the P 10-bit path).

use oxideav_core::{Error, Result};

use crate::b_mb::{
    direct_16x16_spatial_mvs_pub, direct_8x8_spatial_mvs_pub, direct_temporal_mvs_pub, BSliceCtx,
    BipredMode,
};
use crate::bitreader::BitReader;
use crate::cavlc::{decode_residual_block, BlockKind};
use crate::mb::LUMA_BLOCK_RASTER;
use crate::mb_type::{
    decode_b_slice_mb_type, decode_b_sub_mb_type, decode_cbp_inter, BMbType, BPartition,
    BSubPartition, PredDir,
};
use crate::motion::{
    apply_chroma_weight_hi, apply_luma_weight_hi, chroma_mc_hi, luma_mc_hi,
    predict_mv_list,
};
use crate::picture::{MbInfo, Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::{ChromaWeight, LumaWeight, PredWeightTable, SliceHeader};
use crate::sps::Sps;
use crate::transform::{
    chroma_qp_hi, dequantize_4x4_scaled_ext, idct_4x4, inv_hadamard_2x2_chroma_dc_scaled_ext,
};

/// Decode one coded B-slice macroblock at `(mb_x, mb_y)` on the 10-bit path.
/// Mirrors [`crate::b_mb::decode_b_slice_mb`].
#[allow(clippy::too_many_arguments)]
pub fn decode_b_slice_mb_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    ctx: &BSliceCtx<'_>,
    prev_qp: &mut i32,
) -> Result<()> {
    let mb_type = br.read_ue()?;
    let bmb = decode_b_slice_mb_type(mb_type)
        .ok_or_else(|| Error::invalid(format!("h264 10bit b-slice: bad B mb_type {mb_type}")))?;
    match bmb {
        BMbType::IntraInB(imb) => {
            // Reuse the 10-bit I-slice helper for intra-in-B. Mirrors the 8-bit
            // `decode_intra_mb_given_imb` dispatch the CAVLC B-slice uses.
            crate::mb_hi::decode_intra_mb_given_imb_hi(
                br, sps, pps, sh, mb_x, mb_y, pic, prev_qp, imb,
            )
        }
        BMbType::Inter { partition } => decode_b_inter_mb_hi(
            br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, ref_list1, ctx, prev_qp, partition,
        ),
    }
}

/// B_Skip derivation (10-bit) — same structure as the 8-bit path: direct
/// MVs, no residual, no QP delta; chroma + luma MC through u16 kernels.
#[allow(clippy::too_many_arguments)]
pub fn decode_b_skip_mb_hi(
    sh: &SliceHeader,
    sps: &Sps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    ctx: &BSliceCtx<'_>,
    prev_qp: i32,
) -> Result<()> {
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
    let _ = sps;
    direct_16x16_compensate_hi(sh, ctx, mb_x, mb_y, pic, ref_list0, ref_list1)
}

// ---------------------------------------------------------------------------
// Coded B-inter decode path.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_b_inter_mb_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    ctx: &BSliceCtx<'_>,
    prev_qp: &mut i32,
    partition: BPartition,
) -> Result<()> {
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
            direct_16x16_compensate_hi(sh, ctx, mb_x, mb_y, pic, ref_list0, ref_list1)?;
        }
        BPartition::L0_16x16 => {
            decode_16x16_single_dir(
                br, sh, ctx, mb_x, mb_y, pic, ref_list0, ref_list1, PredDir::L0,
            )?;
        }
        BPartition::L1_16x16 => {
            decode_16x16_single_dir(
                br, sh, ctx, mb_x, mb_y, pic, ref_list0, ref_list1, PredDir::L1,
            )?;
        }
        BPartition::Bi_16x16 => {
            decode_16x16_single_dir(
                br, sh, ctx, mb_x, mb_y, pic, ref_list0, ref_list1, PredDir::BiPred,
            )?;
        }
        BPartition::TwoPart16x8 { dirs } => {
            let rects = [(0usize, 0usize, 2usize, 4usize), (2, 0, 2, 4)];
            decode_two_partition_b(
                br, sh, ctx, mb_x, mb_y, pic, ref_list0, ref_list1, &rects, &dirs,
            )?;
        }
        BPartition::TwoPart8x16 { dirs } => {
            let rects = [(0usize, 0usize, 4usize, 2usize), (0, 2, 4, 2)];
            decode_two_partition_b(
                br, sh, ctx, mb_x, mb_y, pic, ref_list0, ref_list1, &rects, &dirs,
            )?;
        }
        BPartition::B8x8 => {
            decode_b8x8(br, sh, sps, mb_x, mb_y, pic, ref_list0, ref_list1, ctx)?;
        }
    }

    // CBP + residual — same layout as the 8-bit B path but on u16 samples.
    let cbp_raw = br.read_ue()?;
    let (cbp_luma, cbp_chroma) = decode_cbp_inter(cbp_raw)
        .ok_or_else(|| Error::invalid(format!("h264 10bit b-slice: bad inter CBP {cbp_raw}")))?;
    let needs_qp = cbp_luma != 0 || cbp_chroma != 0;
    if needs_qp {
        let dqp = br.read_se()?;
        apply_qp_delta_hi(prev_qp, dqp, sps);
    }
    let qp_y = *prev_qp;
    let qp_y_prime = qp_y + 6 * sps.bit_depth_luma_minus8 as i32;
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        info.qp_y = qp_y;
        info.intra4x4_pred_mode = [INTRA_DC_FAKE; 16];
        info.cbp_luma = cbp_luma;
        info.cbp_chroma = cbp_chroma;
    }
    decode_inter_residual_luma_hi(br, mb_x, mb_y, pic, cbp_luma, qp_y_prime)?;
    decode_inter_residual_chroma_hi(br, sps, pps, mb_x, mb_y, pic, cbp_chroma, qp_y)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// 16×16 single-direction partition (L0 / L1 / Bi).
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_16x16_single_dir(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    ctx: &BSliceCtx<'_>,
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
    compensate_partition_b_hi(
        sh, ctx, pic, ref_list0, ref_list1, mb_x, mb_y, 0, 0, 4, 4, dir, ref_l0, ref_l1, mvd_l0,
        mvd_l1,
    )
}

// ---------------------------------------------------------------------------
// 16×8 / 8×16 pair — two partitions each with its own direction.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_two_partition_b(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    ctx: &BSliceCtx<'_>,
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
        compensate_partition_b_hi(
            sh, ctx, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, ph, pw, dirs[p], ref_l0[p],
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
    _sps: &Sps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    ctx: &BSliceCtx<'_>,
) -> Result<()> {
    const SUB_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 2), (2, 0), (2, 2)];
    let num_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let num_l1 = sh.num_ref_idx_l1_active_minus1 + 1;

    let mut subs = [BSubPartition::Direct8x8; 4];
    for s in 0..4 {
        let v = br.read_ue()?;
        subs[s] = decode_b_sub_mb_type(v)
            .ok_or_else(|| Error::invalid(format!("h264 10bit b-slice: bad sub_mb_type {v}")))?;
    }
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
                entry.push(SubMvd::default());
            }
        }
        mvds.push(entry);
    }

    for s in 0..4 {
        let sp = subs[s];
        let (sr0, sc0) = SUB_OFFSETS[s];
        if matches!(sp, BSubPartition::Direct8x8) {
            direct_8x8_compensate_hi(
                sh, ctx, mb_x, mb_y, sr0, sc0, pic, ref_list0, ref_list1,
            )?;
            continue;
        }
        let dir = sp.pred_dir().expect("non-direct");
        for sub_idx in 0..sp.num_sub_partitions() {
            let (dr, dc, sh_h, sh_w) = sp.sub_rect(sub_idx);
            let r0 = sr0 + dr;
            let c0 = sc0 + dc;
            let m = mvds[s][sub_idx];
            compensate_partition_b_hi(
                sh, ctx, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, sh_h, sh_w, dir, ref_l0[s],
                ref_l1[s], m.mvd_l0, m.mvd_l1,
            )?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Partition-level MV derivation + MC, 10-bit.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn compensate_partition_b_hi(
    sh: &SliceHeader,
    ctx: &BSliceCtx<'_>,
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
    // MV predictor derivation is bit-depth agnostic.
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

    apply_mc_hi(
        sh, ctx, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, ph, pw, dir, ref_idx_l0,
        ref_idx_l1, mv_l0, mv_l1,
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_mc_hi(
    sh: &SliceHeader,
    ctx: &BSliceCtx<'_>,
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
    // At 10-bit the direct_*_compensate_pub helpers from the 8-bit module
    // can't be called — they write into `pic.y/cb/cr`. We only ever enter
    // `apply_mc_hi` from single/two/sub-partition coded paths, so dispatch
    // the three cases directly.
    match dir {
        PredDir::L0 => {
            let r = ref_idx_l0.ok_or_else(|| Error::invalid("b-hi: L0 missing ref"))?;
            let mv = mv_l0.unwrap();
            let reference = lookup_ref(ref_list0, r, "L0")?;
            let (lw, cw) = list_weight(sh, 0, r);
            mc_luma_partition_hi(
                pic, reference, mb_x, mb_y, r0, c0, ph, pw, mv.0 as i32, mv.1 as i32, lw,
            );
            mc_chroma_partition_hi(
                pic, reference, mb_x, mb_y, r0, c0, ph, pw, mv.0 as i32, mv.1 as i32, cw,
            );
        }
        PredDir::L1 => {
            let r = ref_idx_l1.ok_or_else(|| Error::invalid("b-hi: L1 missing ref"))?;
            let mv = mv_l1.unwrap();
            let reference = lookup_ref(ref_list1, r, "L1")?;
            let (lw, cw) = list_weight(sh, 1, r);
            mc_luma_partition_hi(
                pic, reference, mb_x, mb_y, r0, c0, ph, pw, mv.0 as i32, mv.1 as i32, lw,
            );
            mc_chroma_partition_hi(
                pic, reference, mb_x, mb_y, r0, c0, ph, pw, mv.0 as i32, mv.1 as i32, cw,
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
            let mode = resolve_bipred_mode_hi(ctx);
            mc_bipred_partition_hi(
                pic, ref0, ref1, mb_x, mb_y, r0, c0, ph, pw, mv0.0 as i32, mv0.1 as i32,
                mv1.0 as i32, mv1.1 as i32, r0_idx, r1_idx, wt, mode, ctx,
            );
        }
    }
    Ok(())
}

/// Route `apply_mc_hi` as a public helper when we need to compensate from
/// already-derived MVs (not currently reused outside this module, but
/// matches the 8-bit `apply_motion_compensation_b` export shape).
#[allow(clippy::too_many_arguments)]
pub fn apply_motion_compensation_b_hi(
    sh: &SliceHeader,
    ctx: &BSliceCtx<'_>,
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
    apply_mc_hi(
        sh, ctx, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, ph, pw, dir, ref_idx_l0,
        ref_idx_l1, mv_l0, mv_l1,
    )
}

// ---------------------------------------------------------------------------
// Direct-mode compensate on u16 planes (spatial + temporal).
// ---------------------------------------------------------------------------

/// 16×16 direct MV derivation + u16 MC, dispatched on
/// `sh.direct_spatial_mv_pred_flag`. Mirrors
/// [`crate::b_mb::direct_16x16_compensate_pub`] but writes `pic.y16 / cb16 /
/// cr16` instead of the 8-bit planes.
fn direct_16x16_compensate_hi(
    sh: &SliceHeader,
    ctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
) -> Result<()> {
    if sh.direct_spatial_mv_pred_flag {
        let d = direct_16x16_spatial_mvs_pub(pic, mb_x, mb_y);
        apply_mc_hi(
            sh, ctx, pic, ref_list0, ref_list1, mb_x, mb_y, 0, 0, 4, 4, d.dir,
            direct_ref_opt(d.ref_idx_l0), direct_ref_opt(d.ref_idx_l1), direct_mv_opt(d.ref_idx_l0, d.mv_l0),
            direct_mv_opt(d.ref_idx_l1, d.mv_l1),
        )
    } else {
        let grid = direct_temporal_mvs_pub(mb_x, mb_y, 0, 0, 4, 4, pic, ref_list1, ctx)?;
        // Per-4×4 MC so each sub-block's potentially-different MV applies.
        for (rr_idx, row) in grid.iter().enumerate() {
            for (cc_idx, d) in row.iter().enumerate() {
                apply_mc_hi(
                    sh, ctx, pic, ref_list0, ref_list1, mb_x, mb_y, rr_idx, cc_idx, 1, 1, d.dir,
                    direct_ref_opt(d.ref_idx_l0), direct_ref_opt(d.ref_idx_l1),
                    direct_mv_opt(d.ref_idx_l0, d.mv_l0), direct_mv_opt(d.ref_idx_l1, d.mv_l1),
                )?;
            }
        }
        Ok(())
    }
}

/// 8×8 direct compensate on u16 planes — spatial covers the full 8×8 in
/// one MC call; temporal walks the 2×2 4×4-grid so each 4×4 colocated
/// result lands independently.
#[allow(clippy::too_many_arguments)]
fn direct_8x8_compensate_hi(
    sh: &SliceHeader,
    ctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    sr0: usize,
    sc0: usize,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
) -> Result<()> {
    if sh.direct_spatial_mv_pred_flag {
        let d = direct_8x8_spatial_mvs_pub(pic, mb_x, mb_y, sr0, sc0);
        apply_mc_hi(
            sh, ctx, pic, ref_list0, ref_list1, mb_x, mb_y, sr0, sc0, 2, 2, d.dir,
            direct_ref_opt(d.ref_idx_l0), direct_ref_opt(d.ref_idx_l1),
            direct_mv_opt(d.ref_idx_l0, d.mv_l0), direct_mv_opt(d.ref_idx_l1, d.mv_l1),
        )
    } else {
        let grid = direct_temporal_mvs_pub(mb_x, mb_y, sr0, sc0, 2, 2, pic, ref_list1, ctx)?;
        for (idx_r, row) in grid.iter().enumerate() {
            for (idx_c, d) in row.iter().enumerate() {
                apply_mc_hi(
                    sh, ctx, pic, ref_list0, ref_list1, mb_x, mb_y, sr0 + idx_r, sc0 + idx_c, 1, 1,
                    d.dir, direct_ref_opt(d.ref_idx_l0), direct_ref_opt(d.ref_idx_l1),
                    direct_mv_opt(d.ref_idx_l0, d.mv_l0), direct_mv_opt(d.ref_idx_l1, d.mv_l1),
                )?;
            }
        }
        Ok(())
    }
}

fn direct_ref_opt(r: i8) -> Option<i8> {
    if r >= 0 { Some(r) } else { None }
}
fn direct_mv_opt(r: i8, mv: (i16, i16)) -> Option<(i16, i16)> {
    if r >= 0 { Some(mv) } else { None }
}

// ---------------------------------------------------------------------------
// u16 MC kernels — variants of the 8-bit helpers in `b_mb.rs`.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn mc_luma_partition_hi(
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
    let mut tmp = vec![0u16; pw * ph];
    let base_x = (mb_x as i32) * 16 + (c0 as i32) * 4;
    let base_y = (mb_y as i32) * 16 + (r0 as i32) * 4;
    luma_mc_hi(&mut tmp, reference, base_x, base_y, mv_x_q, mv_y_q, pw, ph);
    if let Some(lw) = luma_weight {
        apply_luma_weight_hi(&mut tmp, lw, pic.bit_depth_y);
    }
    let lstride = pic.luma_stride();
    let off = pic.luma_off(mb_x, mb_y) + r0 * 4 * lstride + c0 * 4;
    for rr in 0..ph {
        for cc in 0..pw {
            pic.y16[off + rr * lstride + cc] = tmp[rr * pw + cc];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mc_chroma_partition_hi(
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
    let mut tmp_cb = vec![0u16; pw * ph];
    let mut tmp_cr = vec![0u16; pw * ph];
    let base_x = (mb_x as i32) * 8 + (c0 as i32) * 2;
    let base_y = (mb_y as i32) * 8 + (r0 as i32) * 2;
    let cstride = reference.chroma_stride();
    let cw_plane = (reference.width / 2) as i32;
    let ch_plane = (reference.height / 2) as i32;
    chroma_mc_hi(
        &mut tmp_cb, &reference.cb16, cstride, cw_plane, ch_plane, base_x, base_y, mv_x_q, mv_y_q,
        pw, ph, reference.bit_depth_c,
    );
    chroma_mc_hi(
        &mut tmp_cr, &reference.cr16, cstride, cw_plane, ch_plane, base_x, base_y, mv_x_q, mv_y_q,
        pw, ph, reference.bit_depth_c,
    );
    if let Some(cw_entry) = chroma_weight {
        apply_chroma_weight_hi(&mut tmp_cb, cw_entry, 0, pic.bit_depth_c);
        apply_chroma_weight_hi(&mut tmp_cr, cw_entry, 1, pic.bit_depth_c);
    }
    let stride = pic.chroma_stride();
    let off = pic.chroma_off(mb_x, mb_y) + r0 * 2 * stride + c0 * 2;
    for rr in 0..ph {
        for cc in 0..pw {
            pic.cb16[off + rr * stride + cc] = tmp_cb[rr * pw + cc];
            pic.cr16[off + rr * stride + cc] = tmp_cr[rr * pw + cc];
        }
    }
}

/// Bi-prediction luma + chroma MC, 10-bit. Blends L0 and L1 predictions
/// per §8.4.2.3. Default (unweighted) case: `(predL0 + predL1 + 1) >> 1`,
/// clipped to `(1 << BitDepth) - 1`. Explicit weighted / implicit weighted
/// branches use the same formula as the 8-bit path with the Clip1 bound
/// lifted.
#[allow(clippy::too_many_arguments)]
fn mc_bipred_partition_hi(
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
    mode: BipredMode,
    ctx: &BSliceCtx<'_>,
) {
    let lpw = w * 4;
    let lph = h * 4;
    let mut tmp_l0 = vec![0u16; lpw * lph];
    let mut tmp_l1 = vec![0u16; lpw * lph];
    let base_lx = (mb_x as i32) * 16 + (c0 as i32) * 4;
    let base_ly = (mb_y as i32) * 16 + (r0 as i32) * 4;
    luma_mc_hi(&mut tmp_l0, ref0, base_lx, base_ly, mv0_x_q, mv0_y_q, lpw, lph);
    luma_mc_hi(&mut tmp_l1, ref1, base_lx, base_ly, mv1_x_q, mv1_y_q, lpw, lph);

    let lstride = pic.luma_stride();
    let loff = pic.luma_off(mb_x, mb_y) + r0 * 4 * lstride + c0 * 4;
    let bit_depth_y = pic.bit_depth_y;
    let mut luma_out = vec![0u16; lpw * lph];
    match mode {
        BipredMode::Explicit => {
            let pwt = pwt.unwrap();
            let idx0 = ref_idx_l0.max(0) as usize;
            let idx1 = ref_idx_l1.max(0) as usize;
            let lw0 = pwt.luma_l0.get(idx0).copied().unwrap_or_default();
            let lw1 = pwt.luma_l1.get(idx1).copied().unwrap_or_default();
            weighted_bipred_plane_hi(
                &mut luma_out, &tmp_l0, &tmp_l1, lw0.weight, lw0.offset, lw1.weight, lw1.offset,
                pwt.luma_log2_weight_denom, bit_depth_y,
            );
        }
        BipredMode::Implicit => {
            let weights =
                crate::motion::implicit_bipred_weights(
                    ctx.curr_poc,
                    ctx.list0_pocs.get(ref_idx_l0.max(0) as usize).copied().unwrap_or(0),
                    ctx.list1_pocs.get(ref_idx_l1.max(0) as usize).copied().unwrap_or(0),
                    !ctx.list0_short_term
                        .get(ref_idx_l0.max(0) as usize)
                        .copied()
                        .unwrap_or(true),
                    !ctx.list1_short_term
                        .get(ref_idx_l1.max(0) as usize)
                        .copied()
                        .unwrap_or(true),
                );
            match weights {
                crate::motion::ImplicitBiWeights::Weighted { w0, w1 } => {
                    implicit_weighted_bipred_plane_hi(
                        &mut luma_out, &tmp_l0, &tmp_l1, w0, w1, bit_depth_y,
                    );
                }
                crate::motion::ImplicitBiWeights::Default => {
                    average_bipred_hi(&mut luma_out, &tmp_l0, &tmp_l1, bit_depth_y);
                }
            }
        }
        BipredMode::Default => {
            average_bipred_hi(&mut luma_out, &tmp_l0, &tmp_l1, bit_depth_y);
        }
    }
    for rr in 0..lph {
        for cc in 0..lpw {
            pic.y16[loff + rr * lstride + cc] = luma_out[rr * lpw + cc];
        }
    }

    // Chroma
    let cpw = w * 2;
    let cph = h * 2;
    let mut tmp_cb0 = vec![0u16; cpw * cph];
    let mut tmp_cb1 = vec![0u16; cpw * cph];
    let mut tmp_cr0 = vec![0u16; cpw * cph];
    let mut tmp_cr1 = vec![0u16; cpw * cph];
    let base_cx = (mb_x as i32) * 8 + (c0 as i32) * 2;
    let base_cy = (mb_y as i32) * 8 + (r0 as i32) * 2;
    let cstride0 = ref0.chroma_stride();
    let cstride1 = ref1.chroma_stride();
    let cw0 = (ref0.width / 2) as i32;
    let ch0 = (ref0.height / 2) as i32;
    let cw1 = (ref1.width / 2) as i32;
    let ch1 = (ref1.height / 2) as i32;
    chroma_mc_hi(
        &mut tmp_cb0, &ref0.cb16, cstride0, cw0, ch0, base_cx, base_cy, mv0_x_q, mv0_y_q, cpw, cph,
        ref0.bit_depth_c,
    );
    chroma_mc_hi(
        &mut tmp_cr0, &ref0.cr16, cstride0, cw0, ch0, base_cx, base_cy, mv0_x_q, mv0_y_q, cpw, cph,
        ref0.bit_depth_c,
    );
    chroma_mc_hi(
        &mut tmp_cb1, &ref1.cb16, cstride1, cw1, ch1, base_cx, base_cy, mv1_x_q, mv1_y_q, cpw, cph,
        ref1.bit_depth_c,
    );
    chroma_mc_hi(
        &mut tmp_cr1, &ref1.cr16, cstride1, cw1, ch1, base_cx, base_cy, mv1_x_q, mv1_y_q, cpw, cph,
        ref1.bit_depth_c,
    );

    let bit_depth_c = pic.bit_depth_c;
    let mut cb_out = vec![0u16; cpw * cph];
    let mut cr_out = vec![0u16; cpw * cph];
    match mode {
        BipredMode::Explicit => {
            let pwt = pwt.unwrap();
            let idx0 = ref_idx_l0.max(0) as usize;
            let idx1 = ref_idx_l1.max(0) as usize;
            let cw_l0 = pwt.chroma_l0.get(idx0).copied().unwrap_or_default();
            let cw_l1 = pwt.chroma_l1.get(idx1).copied().unwrap_or_default();
            weighted_bipred_plane_hi(
                &mut cb_out, &tmp_cb0, &tmp_cb1, cw_l0.weight[0], cw_l0.offset[0], cw_l1.weight[0],
                cw_l1.offset[0], pwt.chroma_log2_weight_denom, bit_depth_c,
            );
            weighted_bipred_plane_hi(
                &mut cr_out, &tmp_cr0, &tmp_cr1, cw_l0.weight[1], cw_l0.offset[1], cw_l1.weight[1],
                cw_l1.offset[1], pwt.chroma_log2_weight_denom, bit_depth_c,
            );
        }
        BipredMode::Implicit => {
            let weights =
                crate::motion::implicit_bipred_weights(
                    ctx.curr_poc,
                    ctx.list0_pocs.get(ref_idx_l0.max(0) as usize).copied().unwrap_or(0),
                    ctx.list1_pocs.get(ref_idx_l1.max(0) as usize).copied().unwrap_or(0),
                    !ctx.list0_short_term
                        .get(ref_idx_l0.max(0) as usize)
                        .copied()
                        .unwrap_or(true),
                    !ctx.list1_short_term
                        .get(ref_idx_l1.max(0) as usize)
                        .copied()
                        .unwrap_or(true),
                );
            match weights {
                crate::motion::ImplicitBiWeights::Weighted { w0, w1 } => {
                    implicit_weighted_bipred_plane_hi(
                        &mut cb_out, &tmp_cb0, &tmp_cb1, w0, w1, bit_depth_c,
                    );
                    implicit_weighted_bipred_plane_hi(
                        &mut cr_out, &tmp_cr0, &tmp_cr1, w0, w1, bit_depth_c,
                    );
                }
                crate::motion::ImplicitBiWeights::Default => {
                    average_bipred_hi(&mut cb_out, &tmp_cb0, &tmp_cb1, bit_depth_c);
                    average_bipred_hi(&mut cr_out, &tmp_cr0, &tmp_cr1, bit_depth_c);
                }
            }
        }
        BipredMode::Default => {
            average_bipred_hi(&mut cb_out, &tmp_cb0, &tmp_cb1, bit_depth_c);
            average_bipred_hi(&mut cr_out, &tmp_cr0, &tmp_cr1, bit_depth_c);
        }
    }
    let stride = pic.chroma_stride();
    let coff = pic.chroma_off(mb_x, mb_y) + r0 * 2 * stride + c0 * 2;
    for rr in 0..cph {
        for cc in 0..cpw {
            pic.cb16[coff + rr * stride + cc] = cb_out[rr * cpw + cc];
            pic.cr16[coff + rr * stride + cc] = cr_out[rr * cpw + cc];
        }
    }
}

/// u16 variant of [`crate::motion::average_bipred`] — §8.4.2.3.1 default
/// bipred average with bit-depth-scaled Clip1.
fn average_bipred_hi(dst: &mut [u16], l0: &[u16], l1: &[u16], bit_depth: u8) {
    debug_assert_eq!(dst.len(), l0.len());
    debug_assert_eq!(dst.len(), l1.len());
    let max_sample = (1i32 << bit_depth) - 1;
    for i in 0..dst.len() {
        let v = (l0[i] as i32 + l1[i] as i32 + 1) >> 1;
        dst[i] = v.clamp(0, max_sample) as u16;
    }
}

/// u16 variant of [`crate::motion::weighted_bipred_plane`] — §8.4.2.3.2
/// explicit weighted bipred with bit-depth-scaled Clip1.
#[allow(clippy::too_many_arguments)]
fn weighted_bipred_plane_hi(
    dst: &mut [u16],
    l0: &[u16],
    l1: &[u16],
    w0: i32,
    o0: i32,
    w1: i32,
    o1: i32,
    log2_denom: u32,
    bit_depth: u8,
) {
    debug_assert_eq!(dst.len(), l0.len());
    debug_assert_eq!(dst.len(), l1.len());
    let max_sample = (1i32 << bit_depth) - 1;
    let combined_offset = (o0 + o1 + 1) >> 1;
    if log2_denom == 0 {
        for i in 0..dst.len() {
            let a = l0[i] as i32;
            let b = l1[i] as i32;
            let v = a * w0 + b * w1 + combined_offset;
            dst[i] = v.clamp(0, max_sample) as u16;
        }
        return;
    }
    let round = 1i32 << log2_denom;
    let shift = log2_denom + 1;
    for i in 0..dst.len() {
        let a = l0[i] as i32;
        let b = l1[i] as i32;
        let v = ((a * w0 + b * w1 + round) >> shift) + combined_offset;
        dst[i] = v.clamp(0, max_sample) as u16;
    }
}

/// u16 variant of [`crate::motion::implicit_weighted_bipred_plane`] —
/// §8.4.2.3.3 implicit weighted bipred with bit-depth-scaled Clip1.
fn implicit_weighted_bipred_plane_hi(
    dst: &mut [u16],
    l0: &[u16],
    l1: &[u16],
    w0: i32,
    w1: i32,
    bit_depth: u8,
) {
    debug_assert_eq!(dst.len(), l0.len());
    debug_assert_eq!(dst.len(), l1.len());
    let max_sample = (1i32 << bit_depth) - 1;
    for i in 0..dst.len() {
        let a = l0[i] as i32;
        let b = l1[i] as i32;
        let v = (a * w0 + b * w1 + 32) >> 6;
        dst[i] = v.clamp(0, max_sample) as u16;
    }
}

fn resolve_bipred_mode_hi(ctx: &BSliceCtx<'_>) -> BipredMode {
    // The implicit-weights *per-ref* resolution happens inside the MC
    // kernel (it needs the per-partition ref_idx); we just propagate the
    // slice-level mode here.
    ctx.bipred_mode
}

// ---------------------------------------------------------------------------
// Residual decode — u16 samples, i64-widened dequant.
// ---------------------------------------------------------------------------

fn apply_qp_delta_hi(prev_qp: &mut i32, dqp: i32, sps: &Sps) {
    let qp_bd_offset_y = 6 * sps.bit_depth_luma_minus8 as i32;
    let modulus = 52 + qp_bd_offset_y;
    let mut q = (*prev_qp + dqp + 2 * modulus) % modulus;
    if q > 51 {
        q -= modulus;
    }
    *prev_qp = q;
}

fn decode_inter_residual_luma_hi(
    br: &mut BitReader<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    cbp_luma: u8,
    qp_y_prime: i32,
) -> Result<()> {
    let bit_depth = pic.bit_depth_y;
    let max_sample: i32 = (1i32 << bit_depth) - 1;
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
        // §7.4.2.2 — Inter-Y 4×4 slot 3.
        let scale = *pic.scaling_lists.matrix_4x4(3);
        dequantize_4x4_scaled_ext(&mut residual, qp_y_prime, &scale);
        idct_4x4(&mut residual);
        let lo = lo_mb + br_row * 4 * lstride + br_col * 4;
        for r in 0..4 {
            for c in 0..4 {
                let base = pic.y16[lo + r * lstride + c] as i32;
                let v = (base + residual[r * 4 + c]).clamp(0, max_sample);
                pic.y16[lo + r * lstride + c] = v as u16;
            }
        }
        pic.mb_info_mut(mb_x, mb_y).luma_nc[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}

fn decode_inter_residual_chroma_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    cbp_chroma: u8,
    qp_y: i32,
) -> Result<()> {
    let bit_depth_c = pic.bit_depth_c;
    let max_sample: i32 = (1i32 << bit_depth_c) - 1;
    let qp_bd_offset_c = 6 * sps.bit_depth_chroma_minus8 as i32;
    let qpi = (qp_y + pps.chroma_qp_index_offset).clamp(-qp_bd_offset_c, 51 + qp_bd_offset_c);
    let qpc = chroma_qp_hi(qpi);
    let qpc_prime = qpc + qp_bd_offset_c;

    let mut dc_cb = [0i32; 4];
    let mut dc_cr = [0i32; 4];
    if cbp_chroma >= 1 {
        let blk = decode_residual_block(br, 0, BlockKind::ChromaDc2x2)?;
        dc_cb[..4].copy_from_slice(&blk.coeffs[..4]);
        let blk = decode_residual_block(br, 0, BlockKind::ChromaDc2x2)?;
        dc_cr[..4].copy_from_slice(&blk.coeffs[..4]);
        // B-slice chroma is inter → slots 4 (Cb) / 5 (Cr).
        let w_cb = pic.scaling_lists.matrix_4x4(4)[0];
        let w_cr = pic.scaling_lists.matrix_4x4(5)[0];
        inv_hadamard_2x2_chroma_dc_scaled_ext(&mut dc_cb, qpc_prime, w_cb);
        inv_hadamard_2x2_chroma_dc_scaled_ext(&mut dc_cr, qpc_prime, w_cr);
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
                let cat = if plane_kind { 4 } else { 5 };
                let scale = *pic.scaling_lists.matrix_4x4(cat);
                dequantize_4x4_scaled_ext(&mut res, qpc_prime, &scale);
            }
            res[0] = dc[(br_row << 1) | br_col];
            idct_4x4(&mut res);
            let off_in_mb = br_row * 4 * cstride + br_col * 4;
            let plane = if plane_kind { &mut pic.cb16 } else { &mut pic.cr16 };
            for r in 0..4 {
                for c in 0..4 {
                    let base = plane[co + off_in_mb + r * cstride + c] as i32;
                    let v = (base + res[r * 4 + c]).clamp(0, max_sample);
                    plane[co + off_in_mb + r * cstride + c] = v as u16;
                }
            }
            nc_arr[(br_row << 1) | br_col] = total_coeff as u8;
            let info = pic.mb_info_mut(mb_x, mb_y);
            let dst = if plane_kind { &mut info.cb_nc } else { &mut info.cr_nc };
            dst[..4].copy_from_slice(&nc_arr);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers shared verbatim with the 8-bit path.
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
            "h264 10bit b-slice: negative ref_idx in {tag}"
        )));
    }
    let idx = ref_idx as usize;
    list.get(idx).copied().ok_or_else(|| {
        Error::invalid(format!(
            "h264 10bit b-slice: ref_idx {ref_idx} out of RefPicList{tag} range (len={})",
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
