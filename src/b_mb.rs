//! B-slice macroblock decode (CAVLC) — ITU-T H.264 §7.3.5 / §8.4.
//!
//! This mirrors [`crate::p_mb`] for the B-slice case:
//!
//! * B_Direct_16x16 and B_Skip — MVs / ref-idx derived either via
//!   **spatial direct** prediction (§8.4.1.2.2) when the slice header's
//!   `direct_spatial_mv_pred_flag == 1`, or **temporal direct** prediction
//!   (§8.4.1.2.3) when it is `0`. Temporal direct walks the colocated
//!   block in `RefPicList1[0]` (the colocated picture), reads its MV
//!   and the POC of the reference that MV pointed to, and rescales the
//!   MV via `DistScaleFactor` (POC arithmetic) to produce `(mvL0, mvL1)`.
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
//! (`weighted_bipred_idc == 2`) derives a per-block (w0, w1) pair from the
//! POC distances between the current slice and its two references, per
//! §8.4.2.3.3; long-term references or an out-of-range `DistScaleFactor`
//! short-circuit to the default unweighted average.

use oxideav_core::{Error, Result};

use crate::cavlc::{decode_residual_block, BlockKind};
use crate::golomb::BitReaderExt;
use crate::mb::LUMA_BLOCK_RASTER;
use crate::mb_type::{
    decode_b_slice_mb_type, decode_b_sub_mb_type, decode_cbp_inter, BMbType, BPartition,
    BSubPartition, PredDir,
};
use crate::motion::{
    apply_chroma_weight, apply_luma_weight, average_bipred, chroma_mc, implicit_bipred_weights,
    implicit_weighted_bipred_plane, luma_mc, predict_mv_list, weighted_bipred_plane,
    ImplicitBiWeights,
};
use crate::picture::{MbInfo, Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::{ChromaWeight, LumaWeight, PredWeightTable, SliceHeader};
use crate::sps::Sps;
use crate::transform::{
    chroma_qp, dequantize_4x4_scaled, idct_4x4, inv_hadamard_2x2_chroma_dc_scaled,
    inv_hadamard_2x4_chroma_dc_scaled,
};
use oxideav_core::bits::BitReader;

/// Per-slice context needed for temporal direct MV prediction (§8.4.1.2.3).
///
/// The current slice's POC and the POCs of every entry in RefPicList0 and
/// RefPicList1, plus per-entry short-term/long-term class. These are used
/// to derive `DistScaleFactor` (POC arithmetic) and to map a colocated
/// block's `refIdxCol` (an index into the colocated picture's own list)
/// back into the current slice's `RefPicList0` via POC matching.
#[derive(Clone, Copy, Debug)]
pub struct BSliceCtx<'a> {
    pub curr_poc: i32,
    /// Per-entry POCs of the current slice's `RefPicList0`.
    pub list0_pocs: &'a [i32],
    /// Per-entry POCs of the current slice's `RefPicList1`. `list1_pocs[0]`
    /// is the colocated picture's POC.
    pub list1_pocs: &'a [i32],
    /// `true` if the corresponding RefPicList0 entry is short-term, `false`
    /// if long-term. Same length as `list0_pocs`.
    pub list0_short_term: &'a [bool],
    /// Short-term / long-term class for each RefPicList1 entry.
    pub list1_short_term: &'a [bool],
    /// How to combine L0 and L1 predictions for bi-predicted blocks
    /// (§8.4.2.3). Selected from `pps.weighted_bipred_idc` and the presence
    /// of an explicit `pred_weight_table` at the slice header.
    pub bipred_mode: BipredMode,
}

/// §8.4.2.3 dispatch mode for how the two predictions are blended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BipredMode {
    /// `(predL0 + predL1 + 1) >> 1` — §8.4.2.3.1.
    Default,
    /// Explicit slice-header weights — §8.4.2.3.2.
    Explicit,
    /// POC-derived implicit weights — §8.4.2.3.3.
    Implicit,
}

impl<'a> BSliceCtx<'a> {
    /// Find the index in the current `RefPicList0` whose POC matches
    /// `target_poc`. Returns `None` if no entry matches — the caller should
    /// fall back to index 0 per the temporal-direct default.
    fn list0_idx_for_poc(&self, target_poc: i32) -> Option<i8> {
        self.list0_pocs
            .iter()
            .position(|&p| p == target_poc)
            .map(|i| i as i8)
    }
}

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
    ctx: &BSliceCtx<'_>,
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
        BMbType::Inter { partition } => {
            decode_b_inter_mb(
                br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, ref_list1, ctx, prev_qp, partition,
            )?;
            let info = pic.mb_info_mut(mb_x, mb_y);
            info.mb_type_b = Some(bmb);
            if matches!(partition, BPartition::Direct16x16) {
                info.b_direct_per_block = [true; 16];
            }
            Ok(())
        }
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
    ctx: &BSliceCtx<'_>,
    prev_qp: i32,
) -> Result<()> {
    // Initialise MB info for the direct MB. `b_direct_per_block` is
    // all-true because B_Skip is direct over the whole MB (§9.3.3.1.1
    // neighbour lookups consult this for ref_idx / MVD ctxIdxInc).
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        *info = MbInfo {
            qp_y: prev_qp,
            coded: true,
            intra: false,
            skipped: true,
            mb_type_p: None,
            p_partition: None,
            b_direct_per_block: [true; 16],
            ..Default::default()
        };
    }
    // Derive + compensate four 8×8 direct sub-MBs.
    if sh.direct_spatial_mv_pred_flag {
        direct_16x16_spatial_compensate(sh, sps, ctx, mb_x, mb_y, pic, ref_list0, ref_list1)?;
    } else {
        direct_16x16_temporal_compensate(sh, sps, mb_x, mb_y, pic, ref_list0, ref_list1, ctx)?;
    }
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
    ctx: &BSliceCtx<'_>,
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
            if sh.direct_spatial_mv_pred_flag {
                direct_16x16_spatial_compensate(
                    sh, sps, ctx, mb_x, mb_y, pic, ref_list0, ref_list1,
                )?;
            } else {
                direct_16x16_temporal_compensate(
                    sh, sps, mb_x, mb_y, pic, ref_list0, ref_list1, ctx,
                )?;
            }
        }
        BPartition::L0_16x16 => {
            decode_16x16_single_dir(
                br,
                sh,
                ctx,
                mb_x,
                mb_y,
                pic,
                ref_list0,
                ref_list1,
                PredDir::L0,
            )?;
        }
        BPartition::L1_16x16 => {
            decode_16x16_single_dir(
                br,
                sh,
                ctx,
                mb_x,
                mb_y,
                pic,
                ref_list0,
                ref_list1,
                PredDir::L1,
            )?;
        }
        BPartition::Bi_16x16 => {
            decode_16x16_single_dir(
                br,
                sh,
                ctx,
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
        info.cbp_luma = cbp_luma;
        info.cbp_chroma = cbp_chroma;
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
        Some((br.read_se()?, br.read_se()?))
    } else {
        None
    };
    let mvd_l1 = if ref_l1.is_some() {
        Some((br.read_se()?, br.read_se()?))
    } else {
        None
    };
    compensate_partition_b(
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
            mvd_l0[p] = Some((br.read_se()?, br.read_se()?));
        }
    }
    for p in 0..2 {
        if uses_l1(dirs[p]) {
            mvd_l1[p] = Some((br.read_se()?, br.read_se()?));
        }
    }

    for p in 0..2 {
        let (r0, c0, ph, pw) = rects[p];
        compensate_partition_b(
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
    sps: &Sps,
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
                    Some((br.read_se()?, br.read_se()?))
                } else {
                    None
                };
                let mvd_l1 = if uses_l1(dir) {
                    Some((br.read_se()?, br.read_se()?))
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
            // §9.3.3.1.1 direct-cache marker — 2×2 4×4-block grid is
            // direct under this sub-MB.
            {
                let info = pic.mb_info_mut(mb_x, mb_y);
                for rr in sr0..sr0 + 2 {
                    for cc in sc0..sc0 + 2 {
                        info.b_direct_per_block[rr * 4 + cc] = true;
                    }
                }
            }
            if sh.direct_spatial_mv_pred_flag {
                direct_8x8_spatial_compensate(
                    sh, sps, ctx, mb_x, mb_y, sr0, sc0, pic, ref_list0, ref_list1,
                )?;
            } else {
                direct_8x8_temporal_compensate(
                    sh, sps, mb_x, mb_y, sr0, sc0, pic, ref_list0, ref_list1, ctx,
                )?;
            }
            continue;
        }
        let dir = sp.pred_dir().expect("non-direct");
        for sub_idx in 0..sp.num_sub_partitions() {
            let (dr, dc, sh_h, sh_w) = sp.sub_rect(sub_idx);
            let r0 = sr0 + dr;
            let c0 = sc0 + dc;
            let m = mvds[s][sub_idx];
            compensate_partition_b(
                sh, ctx, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, sh_h, sh_w, dir, ref_l0[s],
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
        sh, ctx, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, ph, pw, dir, ref_idx_l0,
        ref_idx_l1, mv_l0, mv_l1,
    )
}

/// Public re-export of the private `apply_motion_compensation` routine so
/// the CABAC B-slice driver can feed it the per-partition MVs + refs. Takes
/// the same arguments verbatim.
#[allow(clippy::too_many_arguments)]
pub fn apply_motion_compensation_b(
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
    apply_motion_compensation(
        sh, ctx, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, ph, pw, dir, ref_idx_l0,
        ref_idx_l1, mv_l0, mv_l1,
    )
}

/// Public wrapper around the whole-MB spatial-or-temporal direct-mode MC
/// path so CABAC B can reuse the CAVLC helpers without duplicating §8.4.1.2.
pub fn direct_16x16_compensate_pub(
    sh: &SliceHeader,
    sps: &Sps,
    ctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
) -> Result<()> {
    if sh.direct_spatial_mv_pred_flag {
        direct_16x16_spatial_compensate(sh, sps, ctx, mb_x, mb_y, pic, ref_list0, ref_list1)
    } else {
        direct_16x16_temporal_compensate(sh, sps, mb_x, mb_y, pic, ref_list0, ref_list1, ctx)
    }
}

/// Direct mode per-4×4 MV / ref_idx record. Used by the 10-bit B-slice path
/// to derive direct MVs via the spatial or temporal rules without triggering
/// any u8-plane sample writes. The caller takes responsibility for running
/// motion compensation on the planes it cares about.
#[derive(Clone, Copy, Debug)]
pub struct DirectMv {
    pub ref_idx_l0: i8,
    pub ref_idx_l1: i8,
    pub mv_l0: (i16, i16),
    pub mv_l1: (i16, i16),
    pub dir: PredDir,
}

impl DirectMv {
    pub fn new(ref_idx_l0: i8, ref_idx_l1: i8, mv_l0: (i16, i16), mv_l1: (i16, i16)) -> Self {
        Self {
            ref_idx_l0,
            ref_idx_l1,
            mv_l0,
            mv_l1,
            dir: pick_bipred_dir(ref_idx_l0, ref_idx_l1),
        }
    }
}

/// Spatial-direct MV derivation (§8.4.1.2.2) for a whole 16×16 MB — returns
/// a single DirectMv shared by every 4×4 block plus writes the MVs / refs
/// into `pic.mb_info[mb_x, mb_y]` so subsequent neighbour lookups see them.
pub fn direct_16x16_spatial_mvs_pub(pic: &mut Picture, mb_x: u32, mb_y: u32) -> DirectMv {
    let (ri0, ri1) = direct_spatial_ref_idx(pic, mb_x, mb_y);
    let (mv0, mv1) = direct_spatial_mvs(pic, mb_x, mb_y, ri0, ri1);
    let d = DirectMv::new(ri0, ri1, mv0, mv1);
    for rr in 0..4usize {
        for cc in 0..4usize {
            let info = pic.mb_info_mut(mb_x, mb_y);
            info.ref_idx_l0[rr * 4 + cc] = ri0;
            info.ref_idx_l1[rr * 4 + cc] = ri1;
            info.mv_l0[rr * 4 + cc] = mv0;
            info.mv_l1[rr * 4 + cc] = mv1;
        }
    }
    d
}

/// Spatial-direct MV derivation for a single 8×8 sub-MB at (sr0, sc0) within
/// the MB. Mirrors [`direct_16x16_spatial_mvs_pub`] but scoped to the 2×2
/// grid of 4×4 blocks under the sub-MB.
pub fn direct_8x8_spatial_mvs_pub(
    pic: &mut Picture,
    mb_x: u32,
    mb_y: u32,
    sr0: usize,
    sc0: usize,
) -> DirectMv {
    let (ri0, ri1) = direct_spatial_ref_idx(pic, mb_x, mb_y);
    let (mv0, mv1) = direct_spatial_mvs(pic, mb_x, mb_y, ri0, ri1);
    let d = DirectMv::new(ri0, ri1, mv0, mv1);
    for rr in sr0..sr0 + 2 {
        for cc in sc0..sc0 + 2 {
            let info = pic.mb_info_mut(mb_x, mb_y);
            info.ref_idx_l0[rr * 4 + cc] = ri0;
            info.ref_idx_l1[rr * 4 + cc] = ri1;
            info.mv_l0[rr * 4 + cc] = mv0;
            info.mv_l1[rr * 4 + cc] = mv1;
        }
    }
    d
}

/// Temporal-direct MV derivation (§8.4.1.2.3) for a rect of 4×4 blocks at
/// `(r0..r0+ph, c0..c0+pw)`. Writes per-4×4 MVs/refs into `pic.mb_info` but
/// does not touch any plane samples; returns the per-4×4 MV grid so the
/// caller can run MC on its chosen planes.
#[allow(clippy::too_many_arguments)]
pub fn direct_temporal_mvs_pub(
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    ph: usize,
    pw: usize,
    pic: &mut Picture,
    ref_list1: &[&Picture],
    ctx: &BSliceCtx<'_>,
) -> Result<Vec<Vec<DirectMv>>> {
    if ref_list1.is_empty() {
        return Err(Error::invalid(
            "h264 b-slice temporal-direct: empty RefPicList1",
        ));
    }
    let ref_pic_col = ref_list1[0];
    let pic1_poc = *ctx.list1_pocs.first().ok_or_else(|| {
        Error::invalid("h264 b-slice temporal-direct: list1_pocs empty (DistScaleFactor input)")
    })?;
    let curr_poc = ctx.curr_poc;

    let mut out: Vec<Vec<DirectMv>> = Vec::with_capacity(ph);
    for rr in r0..r0 + ph {
        let mut row_vec = Vec::with_capacity(pw);
        for cc in c0..c0 + pw {
            let col = colocated_block(ref_pic_col, mb_x, mb_y, rr, cc);
            let (ri0, ri1, mv0, mv1) = if col.intra {
                (0i8, 0i8, (0i16, 0i16), (0i16, 0i16))
            } else {
                let ri0 = ctx.list0_idx_for_poc(col.ref_poc).unwrap_or(0);
                let ri1 = 0i8;
                let l0_short = ctx
                    .list0_short_term
                    .get(ri0 as usize)
                    .copied()
                    .unwrap_or(true);
                let long_term = !col.short_term_ref || !l0_short;
                let (mv0, mv1) =
                    temporal_scale_mv(col.mv, curr_poc, col.ref_poc, pic1_poc, long_term);
                (ri0, ri1, mv0, mv1)
            };
            let info = pic.mb_info_mut(mb_x, mb_y);
            info.ref_idx_l0[rr * 4 + cc] = ri0;
            info.ref_idx_l1[rr * 4 + cc] = ri1;
            info.mv_l0[rr * 4 + cc] = mv0;
            info.mv_l1[rr * 4 + cc] = mv1;
            row_vec.push(DirectMv::new(ri0, ri1, mv0, mv1));
        }
        out.push(row_vec);
    }
    Ok(out)
}

/// Public wrapper for the 8×8 direct-mode sub-MB compensate helpers. Used
/// from the CABAC `B_8x8` path when a sub-partition is `B_Direct_8x8`.
#[allow(clippy::too_many_arguments)]
pub fn direct_8x8_compensate_pub(
    sh: &SliceHeader,
    sps: &Sps,
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
        direct_8x8_spatial_compensate(
            sh, sps, ctx, mb_x, mb_y, sr0, sc0, pic, ref_list0, ref_list1,
        )
    } else {
        direct_8x8_temporal_compensate(
            sh, sps, mb_x, mb_y, sr0, sc0, pic, ref_list0, ref_list1, ctx,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_motion_compensation(
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
            let (mode, implicit_weights) = resolve_bipred_mode(ctx, r0_idx, r1_idx);
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
                mode,
                implicit_weights,
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
    // §8.4.2.2.1 ChromaArrayType == 2 — chroma height equals luma
    // height, so the MV y-component scales by 4 (not 8) and the chroma
    // partition vertical extent is 4*h samples (not 2*h).
    if pic.chroma_format_idc == 2 {
        mc_chroma_partition_422(
            pic,
            reference,
            mb_x,
            mb_y,
            r0,
            c0,
            h,
            w,
            mv_x_q,
            mv_y_q,
            chroma_weight,
        );
        return;
    }
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

/// 4:2:2 B-slice single-direction chroma MC (§8.4.2.2.1
/// ChromaArrayType = 2). Chroma width is halved but height matches
/// luma; `mv_y_q << 1` feeds `chroma_mc` so the 1/8-pel bilinear
/// filter sees `(mv_y_q >> 2, (mv_y_q & 3) << 1)` the way the spec
/// requires. Mirrors `crate::p_mb::mc_chroma_partition_422`.
#[allow(clippy::too_many_arguments)]
fn mc_chroma_partition_422(
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
    let ph = h * 4;
    let mut tmp_cb = vec![0u8; pw * ph];
    let mut tmp_cr = vec![0u8; pw * ph];
    let base_x = (mb_x as i32) * 8 + (c0 as i32) * 2;
    let base_y = (mb_y as i32) * 16 + (r0 as i32) * 4;
    let cstride = reference.chroma_stride();
    let cw = (reference.width / 2) as i32;
    let ch = reference.height as i32;
    let mv_y_c = mv_y_q << 1;
    chroma_mc(
        &mut tmp_cb,
        &reference.cb,
        cstride,
        cw,
        ch,
        base_x,
        base_y,
        mv_x_q,
        mv_y_c,
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
        mv_y_c,
        pw,
        ph,
    );
    if let Some(cw_entry) = chroma_weight {
        apply_chroma_weight(&mut tmp_cb, cw_entry, 0);
        apply_chroma_weight(&mut tmp_cr, cw_entry, 1);
    }
    let stride = pic.chroma_stride();
    let off = pic.chroma_off(mb_x, mb_y) + r0 * 4 * stride + c0 * 2;
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
    mode: BipredMode,
    implicit_weights: ImplicitBiWeights,
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
    match mode {
        BipredMode::Explicit => {
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
        }
        BipredMode::Implicit => match implicit_weights {
            ImplicitBiWeights::Weighted { w0, w1 } => {
                implicit_weighted_bipred_plane(&mut luma_out, &tmp_l0, &tmp_l1, w0, w1);
            }
            ImplicitBiWeights::Default => {
                // §8.4.2.3.3 long-term / clipped-DSF fallback.
                average_bipred(&mut luma_out, &tmp_l0, &tmp_l1);
            }
        },
        BipredMode::Default => {
            average_bipred(&mut luma_out, &tmp_l0, &tmp_l1);
        }
    }
    for rr in 0..lph {
        for cc in 0..lpw {
            pic.y[loff + rr * lstride + cc] = luma_out[rr * lpw + cc];
        }
    }

    // Chroma — dimensions + MV scaling depend on ChromaArrayType.
    // 4:2:0: width × 2, height × 2, mv passed verbatim.
    // 4:2:2: width × 2, height × 4, mv_y << 1 (§8.4.2.2.1).
    let is_422 = pic.chroma_format_idc == 2;
    let cpw = w * 2;
    let cph = if is_422 { h * 4 } else { h * 2 };
    let mut tmp_cb0 = vec![0u8; cpw * cph];
    let mut tmp_cb1 = vec![0u8; cpw * cph];
    let mut tmp_cr0 = vec![0u8; cpw * cph];
    let mut tmp_cr1 = vec![0u8; cpw * cph];
    let base_cx = (mb_x as i32) * 8 + (c0 as i32) * 2;
    let base_cy = if is_422 {
        (mb_y as i32) * 16 + (r0 as i32) * 4
    } else {
        (mb_y as i32) * 8 + (r0 as i32) * 2
    };
    let cstride0 = ref0.chroma_stride();
    let cstride1 = ref1.chroma_stride();
    let cw0 = (ref0.width / 2) as i32;
    let ch0 = if is_422 {
        ref0.height as i32
    } else {
        (ref0.height / 2) as i32
    };
    let cw1 = (ref1.width / 2) as i32;
    let ch1 = if is_422 {
        ref1.height as i32
    } else {
        (ref1.height / 2) as i32
    };
    // `chroma_mc` consumes `mv_y` as a luma quarter-pel. 4:2:2 needs
    // `yIntC = base_y + (mv_y >> 2)` and `yFracC = (mv_y & 3) << 1`,
    // which is what `mv_y << 1` reproduces inside `chroma_mc`'s
    // `>> 3` + `& 7` logic. §8.4.2.2.1 under ChromaArrayType == 2.
    let mv0_y_c = if is_422 { mv0_y_q << 1 } else { mv0_y_q };
    let mv1_y_c = if is_422 { mv1_y_q << 1 } else { mv1_y_q };
    chroma_mc(
        &mut tmp_cb0,
        &ref0.cb,
        cstride0,
        cw0,
        ch0,
        base_cx,
        base_cy,
        mv0_x_q,
        mv0_y_c,
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
        mv0_y_c,
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
        mv1_y_c,
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
        mv1_y_c,
        cpw,
        cph,
    );

    let mut cb_out = vec![0u8; cpw * cph];
    let mut cr_out = vec![0u8; cpw * cph];
    match mode {
        BipredMode::Explicit => {
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
        }
        BipredMode::Implicit => match implicit_weights {
            ImplicitBiWeights::Weighted { w0, w1 } => {
                // §8.4.2.3.3 — same weights apply to luma and chroma.
                implicit_weighted_bipred_plane(&mut cb_out, &tmp_cb0, &tmp_cb1, w0, w1);
                implicit_weighted_bipred_plane(&mut cr_out, &tmp_cr0, &tmp_cr1, w0, w1);
            }
            ImplicitBiWeights::Default => {
                average_bipred(&mut cb_out, &tmp_cb0, &tmp_cb1);
                average_bipred(&mut cr_out, &tmp_cr0, &tmp_cr1);
            }
        },
        BipredMode::Default => {
            average_bipred(&mut cb_out, &tmp_cb0, &tmp_cb1);
            average_bipred(&mut cr_out, &tmp_cr0, &tmp_cr1);
        }
    }
    let stride = pic.chroma_stride();
    let c_rstep = if is_422 { 4 } else { 2 };
    let coff = pic.chroma_off(mb_x, mb_y) + r0 * c_rstep * stride + c0 * 2;
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
        for (_, r) in items.iter().flatten() {
            if *r >= 0 {
                m = Some(m.map_or(*r, |cur| cur.min(*r)));
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
    ctx: &BSliceCtx<'_>,
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
        ctx,
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
    ctx: &BSliceCtx<'_>,
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
        ctx,
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
// Temporal direct MV prediction — §8.4.1.2.3.
// ---------------------------------------------------------------------------

/// The colocated block's selected MV + reference POC, plus the class
/// (short-term / long-term) of that reference and whether the block was
/// intra-coded. Per §8.4.1.2.3 the colocated picture is `RefPicList1[0]`;
/// the colocated 4×4 block is at the same `(br_row, br_col)` within its
/// macroblock. When the colocated block is bi-predicted we take the L0
/// side (per the spec's "L0 first" rule); when it is L1-only we fall
/// back to the L1 side.
#[derive(Clone, Copy, Debug)]
struct Colocated {
    intra: bool,
    /// `true` if the picture pointed to by `ref_poc` is a short-term
    /// reference. Long-term references force zero-scaled MVs
    /// (DistScaleFactor collapses to `mvCol` identity).
    short_term_ref: bool,
    mv: (i16, i16),
    /// POC of the picture the colocated block's MV pointed to. Used as
    /// `pic0PocL0` in the DistScaleFactor formula. Meaningless when
    /// `intra == true`.
    ref_poc: i32,
}

/// Read the colocated block at `(br_row, br_col)` of macroblock
/// `(mb_x, mb_y)` in `ref_pic_col` (= `RefPicList1[0]`). Falls back to
/// the L1 side if the L0 side is unreferenced at that 4×4 block.
fn colocated_block(
    ref_pic_col: &Picture,
    mb_x: u32,
    mb_y: u32,
    br_row: usize,
    br_col: usize,
) -> Colocated {
    let info = ref_pic_col.mb_info_at(mb_x, mb_y);
    if info.intra || !info.coded {
        return Colocated {
            intra: true,
            short_term_ref: true,
            mv: (0, 0),
            ref_poc: 0,
        };
    }
    let idx = br_row * 4 + br_col;
    let r0 = info.ref_idx_l0[idx];
    if r0 >= 0 {
        let poc = info.ref_poc_l0[idx];
        return Colocated {
            intra: false,
            // Heuristic: a short-term ref's POC snapshot is finite. If the
            // snapshot was unresolved (i32::MIN) we treat the colocated
            // block as intra to stay on the safe zero-MV path.
            short_term_ref: poc != i32::MIN,
            mv: info.mv_l0[idx],
            ref_poc: poc,
        };
    }
    let r1 = info.ref_idx_l1[idx];
    if r1 >= 0 {
        let poc = info.ref_poc_l1[idx];
        return Colocated {
            intra: false,
            short_term_ref: poc != i32::MIN,
            mv: info.mv_l1[idx],
            ref_poc: poc,
        };
    }
    // Neither list referenced — treat as intra-colocated.
    Colocated {
        intra: true,
        short_term_ref: true,
        mv: (0, 0),
        ref_poc: 0,
    }
}

/// `DistScaleFactor` per §8.4.1.2.3.
///
/// `tb = curr - pic0`, `td = pic1 - pic0` (each clipped to `[-128, 127]`).
/// Returns a scaled MV: `mvL0 = (DSF * mvCol + 128) >> 8`, `mvL1 = mvL0 - mvCol`.
/// When `td == 0` (reference and colocated share a POC) the scale collapses to
/// identity — `mvL0 = mvCol`, `mvL1 = (0, 0)`. Long-term references also force
/// identity (see §8.4.1.2.3 last paragraph).
fn temporal_scale_mv(
    mv_col: (i16, i16),
    curr_poc: i32,
    pic0_poc: i32,
    pic1_poc: i32,
    long_term: bool,
) -> ((i16, i16), (i16, i16)) {
    let tb = (curr_poc - pic0_poc).clamp(-128, 127);
    let td = (pic1_poc - pic0_poc).clamp(-128, 127);
    if td == 0 || long_term {
        // Degenerate: same-POC or LT ref → mvL0 = mvCol, mvL1 = 0.
        return (mv_col, (0, 0));
    }
    let td_abs = td.unsigned_abs() as i32;
    let tx = (16384 + (td_abs / 2)) / td;
    let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);

    let scale_component = |c: i16| -> i16 {
        let v = (dsf * (c as i32) + 128) >> 8;
        // MVs are stored as i16 quarter-pel; clamp to avoid overflow on
        // pathological inputs (real encoder output won't hit this).
        v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
    };
    let mv_l0 = (scale_component(mv_col.0), scale_component(mv_col.1));
    let mv_l1 = (
        mv_l0.0.wrapping_sub(mv_col.0),
        mv_l0.1.wrapping_sub(mv_col.1),
    );
    (mv_l0, mv_l1)
}

/// Derive the per-4×4 `(ref_idx_l0, ref_idx_l1, mv_l0, mv_l1)` for a
/// direct temporal macroblock covering `(r0..r0+ph, c0..c0+pw)` 4×4 grid.
///
/// Per §8.4.1.2.3: `refIdxL1 = 0` (refPicCol is by construction the L1[0]
/// entry). `refIdxL0` is set to the current-slice L0 index whose POC
/// matches the colocated block's reference POC. When no match is found
/// (colocated references a picture not in the current L0) the spec falls
/// back to 0 — this is also what we do on the intra-colocated path.
#[allow(clippy::too_many_arguments)]
fn direct_temporal_rect_compensate(
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    ph: usize,
    pw: usize,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    ctx: &BSliceCtx<'_>,
) -> Result<()> {
    if ref_list1.is_empty() {
        return Err(Error::invalid(
            "h264 b-slice temporal-direct: empty RefPicList1",
        ));
    }
    let ref_pic_col = ref_list1[0];
    let pic1_poc = *ctx.list1_pocs.first().ok_or_else(|| {
        Error::invalid("h264 b-slice temporal-direct: list1_pocs empty (DistScaleFactor input)")
    })?;
    let curr_poc = ctx.curr_poc;

    // Write per-4×4 MV / ref_idx into this MB's info for neighbour lookups
    // and deblocking, then run apply_motion_compensation over the rect.
    for rr in r0..r0 + ph {
        for cc in c0..c0 + pw {
            let col = colocated_block(ref_pic_col, mb_x, mb_y, rr, cc);
            let (ri0, ri1, mv0, mv1) = if col.intra {
                // §8.4.1.2.3: intra-colocated → zero MVs, ref_idx 0 on both
                // lists (the refs themselves are present in the current
                // RefPicLists by construction).
                (0i8, 0i8, (0i16, 0i16), (0i16, 0i16))
            } else {
                let ri0 = ctx.list0_idx_for_poc(col.ref_poc).unwrap_or(0);
                let ri1 = 0i8;
                // Long-term refs in either the colocated's target or the
                // current slice's L0[ri0] force identity scaling (spec
                // §8.4.1.2.3 long-term exception).
                let l0_short = ctx
                    .list0_short_term
                    .get(ri0 as usize)
                    .copied()
                    .unwrap_or(true);
                let long_term = !col.short_term_ref || !l0_short;
                let (mv0, mv1) =
                    temporal_scale_mv(col.mv, curr_poc, col.ref_poc, pic1_poc, long_term);
                (ri0, ri1, mv0, mv1)
            };
            let info = pic.mb_info_mut(mb_x, mb_y);
            info.ref_idx_l0[rr * 4 + cc] = ri0;
            info.ref_idx_l1[rr * 4 + cc] = ri1;
            info.mv_l0[rr * 4 + cc] = mv0;
            info.mv_l1[rr * 4 + cc] = mv1;
        }
    }

    // MC path: the entire rect uses bi-prediction on (ref0, ref1). Per-4×4
    // MVs vary, so run MC on each 4×4 sub-block independently rather than
    // assuming a single partition-wide MV.
    for rr in r0..r0 + ph {
        for cc in c0..c0 + pw {
            let info = pic.mb_info_at(mb_x, mb_y);
            let ri0 = info.ref_idx_l0[rr * 4 + cc];
            let ri1 = info.ref_idx_l1[rr * 4 + cc];
            let mv0 = info.mv_l0[rr * 4 + cc];
            let mv1 = info.mv_l1[rr * 4 + cc];
            let dir = pick_bipred_dir(ri0, ri1);
            apply_motion_compensation(
                sh,
                ctx,
                pic,
                ref_list0,
                ref_list1,
                mb_x,
                mb_y,
                rr,
                cc,
                1,
                1,
                dir,
                if ri0 >= 0 { Some(ri0) } else { None },
                if ri1 >= 0 { Some(ri1) } else { None },
                if ri0 >= 0 { Some(mv0) } else { None },
                if ri1 >= 0 { Some(mv1) } else { None },
            )?;
        }
    }
    Ok(())
}

/// Temporal-direct compensate for a whole 16×16 MB — §8.4.1.2.3.
#[allow(clippy::too_many_arguments)]
fn direct_16x16_temporal_compensate(
    sh: &SliceHeader,
    _sps: &Sps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    ctx: &BSliceCtx<'_>,
) -> Result<()> {
    direct_temporal_rect_compensate(sh, mb_x, mb_y, 0, 0, 4, 4, pic, ref_list0, ref_list1, ctx)
}

/// Temporal-direct compensate for a single 8×8 B_Direct_8x8 sub-MB.
#[allow(clippy::too_many_arguments)]
fn direct_8x8_temporal_compensate(
    sh: &SliceHeader,
    _sps: &Sps,
    mb_x: u32,
    mb_y: u32,
    sr0: usize,
    sc0: usize,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    ctx: &BSliceCtx<'_>,
) -> Result<()> {
    direct_temporal_rect_compensate(
        sh, mb_x, mb_y, sr0, sc0, 2, 2, pic, ref_list0, ref_list1, ctx,
    )
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
        // §7.4.2.2 — Inter-Y 4×4 is slot 3.
        let scale = *pic.scaling_lists.matrix_4x4(3);
        dequantize_4x4_scaled(&mut residual, qp_y, &scale);
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
    // §8.5.11.2 ChromaArrayType dispatch — 4:2:2 uses a 2×4 chroma DC
    // Hadamard and 8 AC blocks per plane instead of 4:2:0's 2×2 + 4.
    if pic.chroma_format_idc == 2 {
        return decode_inter_residual_chroma_422(br, pps, mb_x, mb_y, pic, cbp_chroma, qp_y);
    }
    let qpc = chroma_qp(qp_y, pps.chroma_qp_index_offset);
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
        inv_hadamard_2x2_chroma_dc_scaled(&mut dc_cb, qpc, w_cb);
        inv_hadamard_2x2_chroma_dc_scaled(&mut dc_cr, qpc, w_cr);
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
                // §7.4.2.2 — Inter-Cb/Cr (slots 4/5). B-slice chroma is inter.
                let cat = if plane_kind { 4 } else { 5 };
                let scale = *pic.scaling_lists.matrix_4x4(cat);
                dequantize_4x4_scaled(&mut res, qpc, &scale);
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
            let dst = if plane_kind {
                &mut info.cb_nc
            } else {
                &mut info.cr_nc
            };
            dst[..4].copy_from_slice(&nc_arr);
        }
    }
    Ok(())
}

/// 4:2:2 B-slice inter residual (§8.5.11.2, §9.2.1.2 ChromaArrayType == 2).
///
/// Same cbp_chroma encoding as 4:2:0 (0 = none, 1 = DC only, 2 = DC + AC),
/// but the plane is 8×16 samples = 2 cols × 4 rows of 4×4 AC blocks and
/// the chroma DC block is 2×4 (CAVLC kind `ChromaDc2x4`, nC = -2) fed
/// through `inv_hadamard_2x4_chroma_dc_scaled` with the Amendment-2
/// `QP'_C,DC = QP'_C + 3` offset baked in. B-slice chroma is always
/// inter → residual uses inter-Cb/Cr scaling slots 4/5.
fn decode_inter_residual_chroma_422(
    br: &mut BitReader<'_>,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    cbp_chroma: u8,
    qp_y: i32,
) -> Result<()> {
    let qpc_cb = chroma_qp(qp_y, pps.chroma_qp_index_offset);
    let qpc_cr = chroma_qp(qp_y, pps.second_chroma_qp_index_offset);
    let mut dc_cb = [0i32; 8];
    let mut dc_cr = [0i32; 8];
    if cbp_chroma >= 1 {
        let blk = decode_residual_block(br, -2, BlockKind::ChromaDc2x4)?;
        dc_cb.copy_from_slice(&blk.coeffs[..8]);
        let blk = decode_residual_block(br, -2, BlockKind::ChromaDc2x4)?;
        dc_cr.copy_from_slice(&blk.coeffs[..8]);
        let w_cb = pic.scaling_lists.matrix_4x4(4)[0];
        let w_cr = pic.scaling_lists.matrix_4x4(5)[0];
        inv_hadamard_2x4_chroma_dc_scaled(&mut dc_cb, qpc_cb, w_cb);
        inv_hadamard_2x4_chroma_dc_scaled(&mut dc_cr, qpc_cr, w_cr);
    }
    let cstride = pic.chroma_stride();
    let co = pic.chroma_off(mb_x, mb_y);
    for plane_cb in [true, false] {
        let dc = if plane_cb { &dc_cb } else { &dc_cr };
        let qpc = if plane_cb { qpc_cb } else { qpc_cr };
        let mut nc_arr = [0u8; 8];
        for blk_row in 0..4usize {
            for blk_col in 0..2usize {
                let blk_idx = blk_row * 2 + blk_col;
                let mut res = [0i32; 16];
                let mut total_coeff = 0u32;
                if cbp_chroma == 2 {
                    let nc =
                        predict_inter_nc_chroma_422(pic, mb_x, mb_y, plane_cb, blk_row, blk_col);
                    let ac = decode_residual_block(br, nc, BlockKind::ChromaAc)?;
                    total_coeff = ac.total_coeff;
                    res = ac.coeffs;
                    let cat = if plane_cb { 4 } else { 5 };
                    let scale = *pic.scaling_lists.matrix_4x4(cat);
                    dequantize_4x4_scaled(&mut res, qpc, &scale);
                }
                res[0] = dc[blk_idx];
                idct_4x4(&mut res);
                let off_in_mb = blk_row * 4 * cstride + blk_col * 4;
                let plane = if plane_cb { &mut pic.cb } else { &mut pic.cr };
                for r in 0..4 {
                    for c in 0..4 {
                        let base = plane[co + off_in_mb + r * cstride + c] as i32;
                        plane[co + off_in_mb + r * cstride + c] =
                            (base + res[r * 4 + c]).clamp(0, 255) as u8;
                    }
                }
                nc_arr[blk_idx] = total_coeff as u8;
                let info = pic.mb_info_mut(mb_x, mb_y);
                let dst = if plane_cb {
                    &mut info.cb_nc
                } else {
                    &mut info.cr_nc
                };
                dst[..8].copy_from_slice(&nc_arr);
            }
        }
    }
    Ok(())
}

/// 4:2:2 B-slice neighbour-nC predictor — mirrors the P-slice 4:2:2
/// predictor in [`crate::p_mb::predict_inter_nc_chroma_422`].
fn predict_inter_nc_chroma_422(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    cb: bool,
    blk_row: usize,
    blk_col: usize,
) -> i32 {
    let pick = |info: &MbInfo, idx: usize| -> u8 {
        if cb {
            info.cb_nc[idx]
        } else {
            info.cr_nc[idx]
        }
    };
    let info_here = pic.mb_info_at(mb_x, mb_y);
    let left = if blk_col > 0 {
        Some(pick(info_here, blk_row * 2 + (blk_col - 1)))
    } else if mb_x > 0 {
        let info = pic.mb_info_at(mb_x - 1, mb_y);
        if info.coded {
            Some(pick(info, blk_row * 2 + 1))
        } else {
            None
        }
    } else {
        None
    };
    let top = if blk_row > 0 {
        Some(pick(info_here, (blk_row - 1) * 2 + blk_col))
    } else if mb_y > 0 {
        let info = pic.mb_info_at(mb_x, mb_y - 1);
        if info.coded {
            Some(pick(info, 3 * 2 + blk_col))
        } else {
            None
        }
    } else {
        None
    };
    crate::mb::nc_from_neighbours(left, top)
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

/// Pick the bipred blend mode for `(ref_idx_l0, ref_idx_l1)` at the current
/// slice. For [`BipredMode::Implicit`] the per-ref POCs and long-term class
/// come from `ctx`; missing entries (out-of-range ref_idx) short-circuit to
/// the default equal-weight fallback per §8.4.2.3.3.
fn resolve_bipred_mode(
    ctx: &BSliceCtx<'_>,
    r0_idx: i8,
    r1_idx: i8,
) -> (BipredMode, ImplicitBiWeights) {
    if ctx.bipred_mode != BipredMode::Implicit {
        return (ctx.bipred_mode, ImplicitBiWeights::Default);
    }
    let i0 = r0_idx.max(0) as usize;
    let i1 = r1_idx.max(0) as usize;
    let Some(&l0_poc) = ctx.list0_pocs.get(i0) else {
        return (BipredMode::Implicit, ImplicitBiWeights::Default);
    };
    let Some(&l1_poc) = ctx.list1_pocs.get(i1) else {
        return (BipredMode::Implicit, ImplicitBiWeights::Default);
    };
    let l0_short = ctx.list0_short_term.get(i0).copied().unwrap_or(true);
    let l1_short = ctx.list1_short_term.get(i1).copied().unwrap_or(true);
    let weights = implicit_bipred_weights(ctx.curr_poc, l0_poc, l1_poc, !l0_short, !l1_short);
    (BipredMode::Implicit, weights)
}

fn list_weight(
    sh: &SliceHeader,
    list: u8,
    ref_idx: i8,
) -> (Option<&LumaWeight>, Option<&ChromaWeight>) {
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
