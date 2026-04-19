//! CABAC B-slice macroblock decode — ITU-T H.264 §7.3.5 + §9.3.
//!
//! Mirrors [`crate::b_mb::decode_b_slice_mb`] (the CAVLC B path) but reads
//! every syntax element through the CABAC engine. Motion compensation, MV
//! prediction, direct-mode derivation (spatial + temporal), and pixel
//! reconstruction all share the same helpers as CAVLC — only the entropy
//! layer differs.
//!
//! Coverage (matching CAVLC B):
//!   * `B_Skip` — signalled by `mb_skip_flag = 1`; direct-mode MC (spatial
//!     §8.4.1.2.2 or temporal §8.4.1.2.3).
//!   * Table 9-37 inter types: `B_Direct_16x16`, `B_{L0,L1,Bi}_16x16`, every
//!     16×8 / 8×16 pair variant, and `B_8x8` (with Table 9-38 sub_mb_type).
//!   * Intra-in-B MBs dispatch to the CABAC I-slice path.
//!
//! Out of scope — surfaces `Error::Unsupported`:
//!   * MBAFF B-slices.

use oxideav_core::{Error, Result};

use crate::b_mb::BSliceCtx;
use crate::cabac::binarize;
use crate::cabac::context::CabacContext;
use crate::cabac::engine::CabacDecoder;
use crate::cabac::mb::{
    decode_luma_8x8_residual_in_place, decode_residual_block_in_place,
    transform_size_8x8_flag_ctx_idx_inc,
};
use crate::cabac::residual::{BlockCat, CbfNeighbours};
use crate::cabac::tables::{
    CTX_IDX_CODED_BLOCK_PATTERN_LUMA, CTX_IDX_MB_QP_DELTA, CTX_IDX_MB_SKIP_FLAG_B,
    CTX_IDX_MB_TYPE_B, CTX_IDX_MB_TYPE_I, CTX_IDX_MVD_L0_X, CTX_IDX_MVD_L0_Y, CTX_IDX_MVD_L1_X,
    CTX_IDX_MVD_L1_Y, CTX_IDX_REF_IDX_L0, CTX_IDX_REF_IDX_L1, CTX_IDX_SUB_MB_TYPE_B,
    CTX_IDX_TRANSFORM_SIZE_8X8_FLAG,
};
use crate::mb::LUMA_BLOCK_RASTER;
use crate::mb_type::{
    decode_b_slice_mb_type, decode_b_sub_mb_type, BMbType, BPartition, BSubPartition, IMbType,
    PredDir,
};
use crate::motion::predict_mv_list;
use crate::picture::{MbInfo, Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::SliceHeader;
use crate::sps::Sps;
use crate::transform::{
    chroma_qp, dequantize_4x4_scaled, dequantize_8x8_scaled, idct_4x4, idct_8x8,
    inv_hadamard_2x2_chroma_dc_scaled,
};

const SUB_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 2), (2, 0), (2, 2)];

/// Decode one B-slice macroblock at `(mb_x, mb_y)` through CABAC.
///
/// Returns `Ok(true)` when the MB was signalled via `mb_skip_flag`
/// (direct-mode, no residual). The caller still reads the `end_of_slice_flag`
/// terminate bin after this returns.
#[allow(clippy::too_many_arguments)]
pub fn decode_b_mb_cabac(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    bctx: &BSliceCtx<'_>,
    prev_qp: &mut i32,
) -> Result<bool> {
    // §9.3.3.1.1.1 — mb_skip_flag for B slices.
    let skip_inc = mb_skip_flag_b_ctx_idx_inc(pic, mb_x, mb_y);
    let skipped = {
        let slice = &mut ctxs[CTX_IDX_MB_SKIP_FLAG_B..CTX_IDX_MB_SKIP_FLAG_B + 3];
        binarize::decode_mb_skip_flag_b(d, slice, skip_inc)?
    };
    if skipped {
        // B_Skip — delegate to the CAVLC path's direct-mode helper.
        crate::b_mb::decode_b_skip_mb(
            sh, sps, mb_x, mb_y, pic, ref_list0, ref_list1, bctx, *prev_qp,
        )?;
        return Ok(true);
    }

    // §9.3.3.1.1.3 — B-slice mb_type. Decoded value uses the Table 7-14
    // convention (0..=22 inter, 23..=47 intra-in-B, 48 = I_PCM).
    let mb_type_raw = {
        let (b_slice, i_slice) = if CTX_IDX_MB_TYPE_B < CTX_IDX_MB_TYPE_I {
            let (head, tail) = ctxs.split_at_mut(CTX_IDX_MB_TYPE_I);
            (
                &mut head[CTX_IDX_MB_TYPE_B..CTX_IDX_MB_TYPE_B + 9],
                &mut tail[..8],
            )
        } else {
            let (head, tail) = ctxs.split_at_mut(CTX_IDX_MB_TYPE_B);
            (
                &mut tail[..9],
                &mut head[CTX_IDX_MB_TYPE_I..CTX_IDX_MB_TYPE_I + 8],
            )
        };
        let inc = mb_type_b_ctx_idx_inc(pic, mb_x, mb_y);
        binarize::decode_mb_type_b(d, b_slice, i_slice, inc)?
    };

    // Map raw B-slice mb_type to the internal enum. I_PCM (48) is handled
    // separately — `decode_b_slice_mb_type` doesn't model 48.
    if mb_type_raw == 48 {
        return decode_intra_in_b(
            d,
            ctxs,
            sh,
            sps,
            pps,
            mb_x,
            mb_y,
            pic,
            prev_qp,
            IMbType::IPcm,
        )
        .map(|_| false);
    }
    let bmb = decode_b_slice_mb_type(mb_type_raw)
        .ok_or_else(|| Error::invalid(format!("h264 cabac b-slice: bad mb_type {mb_type_raw}")))?;
    match bmb {
        BMbType::IntraInB(imb) => {
            decode_intra_in_b(d, ctxs, sh, sps, pps, mb_x, mb_y, pic, prev_qp, imb)?;
        }
        BMbType::Inter { partition } => {
            decode_b_inter(
                d, ctxs, sh, sps, pps, mb_x, mb_y, pic, ref_list0, ref_list1, bctx, prev_qp,
                partition,
            )?;
            // §9.3.3.1.1.3 neighbour ctxIdxInc uses `mb_type_b` below —
            // publish the decoded B-MB type + "whole MB direct" flag so
            // later MBs' ctxIdxInc sees the right classification.
            let info = pic.mb_info_mut(mb_x, mb_y);
            info.mb_type_b = Some(bmb);
            if matches!(partition, BPartition::Direct16x16) {
                info.b_direct_per_block = [true; 16];
            }
        }
    }
    Ok(false)
}

// ---------------------------------------------------------------------------
// Inter-MB dispatch.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_b_inter(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    bctx: &BSliceCtx<'_>,
    prev_qp: &mut i32,
    partition: BPartition,
) -> Result<()> {
    // Reset this MB's MV state — L1 MVs from an earlier MB at the same slot
    // must not leak into neighbour lookups.
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

    // `B_8x8` populates `sub_parts` so the post-motion `transform_size_8x8_flag`
    // gate can check that every sub_mb_type is ≥ 8×8 (§7.3.5.1).
    let mut sub_parts: Option<[BSubPartition; 4]> = None;

    match partition {
        BPartition::Direct16x16 => {
            // §8.4.1.2 direct — reuse the CAVLC compensate helpers.
            direct_16x16_compensate(sh, sps, bctx, mb_x, mb_y, pic, ref_list0, ref_list1)?;
        }
        BPartition::L0_16x16 => {
            decode_16x16_single_dir(
                d,
                ctxs,
                sh,
                bctx,
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
                d,
                ctxs,
                sh,
                bctx,
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
                d,
                ctxs,
                sh,
                bctx,
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
            decode_two_partition(
                d, ctxs, sh, bctx, mb_x, mb_y, pic, ref_list0, ref_list1, &rects, &dirs,
            )?;
        }
        BPartition::TwoPart8x16 { dirs } => {
            let rects = [(0usize, 0usize, 4usize, 2usize), (0, 2, 4, 2)];
            decode_two_partition(
                d, ctxs, sh, bctx, mb_x, mb_y, pic, ref_list0, ref_list1, &rects, &dirs,
            )?;
        }
        BPartition::B8x8 => {
            let subs = decode_b_8x8(
                d, ctxs, sh, sps, mb_x, mb_y, pic, ref_list0, ref_list1, bctx,
            )?;
            sub_parts = Some(subs);
        }
    }

    // §9.3.3.1.1.4 — coded_block_pattern (inter CBP binarisation is shared
    // with P-slices).
    let cbp = {
        let slice =
            &mut ctxs[CTX_IDX_CODED_BLOCK_PATTERN_LUMA..CTX_IDX_CODED_BLOCK_PATTERN_LUMA + 12];
        binarize::decode_coded_block_pattern(
            d,
            slice,
            sps.chroma_format_idc as u8,
            |i, decoded| super::mb::cbp_luma_ctx_idx_inc(pic, mb_x, mb_y, i, decoded),
            |bin| {
                if bin == 0 {
                    super::mb::cbp_chroma_ctx_idx_inc_any(pic, mb_x, mb_y)
                } else {
                    super::mb::cbp_chroma_ctx_idx_inc_ac(pic, mb_x, mb_y)
                }
            },
        )?
    };
    let cbp_luma = (cbp & 0x0F) as u8;
    let cbp_chroma = ((cbp >> 4) & 0x03) as u8;

    // §7.3.5.1 — `transform_size_8x8_flag` is present for B inter MBs when
    // the PPS enables 8×8 AND cbp_luma != 0 AND the motion partition has no
    // sub-8×8 split. For `B_8x8`, every `sub_mb_type` must be `Direct_8x8`
    // or `L{0,1}/Bi_8x8` (no 8×4 / 4×8 / 4×4 sub-partition would straddle
    // an 8×8 transform block). All other B partitions (16×16 / 16×8 / 8×16)
    // are already ≥ 8×8 and qualify unconditionally when cbp_luma > 0.
    let partition_allows_8x8 = match partition {
        BPartition::B8x8 => sub_parts
            .as_ref()
            .map(|subs| subs.iter().all(|sp| sp.is_at_least_8x8()))
            .unwrap_or(false),
        _ => true,
    };
    let transform_8x8 = if pps.transform_8x8_mode_flag && cbp_luma != 0 && partition_allows_8x8 {
        let inc = transform_size_8x8_flag_ctx_idx_inc(pic, mb_x, mb_y);
        let slice = &mut ctxs[CTX_IDX_TRANSFORM_SIZE_8X8_FLAG..CTX_IDX_TRANSFORM_SIZE_8X8_FLAG + 3];
        binarize::decode_transform_size_8x8_flag(d, slice, inc)?
    } else {
        false
    };

    let needs_qp = cbp_luma != 0 || cbp_chroma != 0;
    if needs_qp {
        let inc = if pic.last_mb_qp_delta_was_nonzero {
            1u8
        } else {
            0u8
        };
        let slice = &mut ctxs[CTX_IDX_MB_QP_DELTA..CTX_IDX_MB_QP_DELTA + 4];
        let dqp = binarize::decode_mb_qp_delta(d, slice, inc)?;
        pic.last_mb_qp_delta_was_nonzero = dqp != 0;
        *prev_qp = ((*prev_qp + dqp + 52) % 52).clamp(0, 51);
    } else {
        pic.last_mb_qp_delta_was_nonzero = false;
    }
    let qp_y = *prev_qp;

    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        info.qp_y = qp_y;
        info.intra4x4_pred_mode = [INTRA_DC_FAKE; 16];
        info.cbp_luma = cbp_luma;
        info.cbp_chroma = cbp_chroma;
        info.transform_8x8 = transform_8x8;
    }
    if transform_8x8 {
        decode_inter_residual_luma_8x8(d, ctxs, mb_x, mb_y, pic, cbp_luma, qp_y)?;
    } else {
        decode_inter_residual_luma(d, ctxs, mb_x, mb_y, pic, cbp_luma, qp_y)?;
    }
    decode_inter_residual_chroma(d, ctxs, pps, mb_x, mb_y, pic, cbp_chroma, qp_y)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// 16x16 single-direction partition.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_16x16_single_dir(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sh: &SliceHeader,
    bctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    dir: PredDir,
) -> Result<()> {
    let ref_l0 = if uses_l0(dir) {
        Some(read_ref_idx_lx(
            d,
            ctxs,
            pic,
            mb_x,
            mb_y,
            0,
            0,
            sh.num_ref_idx_l0_active_minus1 + 1,
            0,
        )?)
    } else {
        None
    };
    let ref_l1 = if uses_l1(dir) {
        Some(read_ref_idx_lx(
            d,
            ctxs,
            pic,
            mb_x,
            mb_y,
            0,
            0,
            sh.num_ref_idx_l1_active_minus1 + 1,
            1,
        )?)
    } else {
        None
    };
    let mvd_l0 = if ref_l0.is_some() {
        let mvd = read_mvd(d, ctxs, pic, mb_x, mb_y, 0, 0, 0)?;
        write_mvd_abs(pic, mb_x, mb_y, 0, 0, 4, 4, 0, mvd);
        Some(mvd)
    } else {
        None
    };
    let mvd_l1 = if ref_l1.is_some() {
        let mvd = read_mvd(d, ctxs, pic, mb_x, mb_y, 0, 0, 1)?;
        write_mvd_abs(pic, mb_x, mb_y, 0, 0, 4, 4, 1, mvd);
        Some(mvd)
    } else {
        None
    };
    compensate_partition(
        sh, bctx, pic, ref_list0, ref_list1, mb_x, mb_y, 0, 0, 4, 4, dir, ref_l0, ref_l1, mvd_l0,
        mvd_l1,
    )
}

// ---------------------------------------------------------------------------
// 16x8 / 8x16 pair.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_two_partition(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sh: &SliceHeader,
    bctx: &BSliceCtx<'_>,
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

    // §7.3.5.1 syntax order: all ref_idx_l0, then all ref_idx_l1, then all
    // mvd_l0, then all mvd_l1 — per partition.
    for p in 0..2 {
        if uses_l0(dirs[p]) {
            let (r0, c0, _, _) = rects[p];
            ref_l0[p] = Some(read_ref_idx_lx(
                d, ctxs, pic, mb_x, mb_y, r0, c0, num_l0, 0,
            )?);
        }
    }
    for p in 0..2 {
        if uses_l1(dirs[p]) {
            let (r0, c0, _, _) = rects[p];
            ref_l1[p] = Some(read_ref_idx_lx(
                d, ctxs, pic, mb_x, mb_y, r0, c0, num_l1, 1,
            )?);
        }
    }
    // §9.3.3.1.1.7 — write absolute MVD magnitudes into `mvd_lX_abs`
    // immediately after each partition's MVD decode so the next
    // partition's MVD ctxIdxInc sees the updated neighbour block.
    // FFmpeg does this via `fill_rectangle(sl->mvd_cache[list], ...)`
    // inline with `DECODE_CABAC_MB_MVD`.
    for p in 0..2 {
        if uses_l0(dirs[p]) {
            let (r0, c0, ph, pw) = rects[p];
            let mvd = read_mvd(d, ctxs, pic, mb_x, mb_y, r0, c0, 0)?;
            write_mvd_abs(pic, mb_x, mb_y, r0, c0, ph, pw, 0, mvd);
            mvd_l0[p] = Some(mvd);
        }
    }
    for p in 0..2 {
        if uses_l1(dirs[p]) {
            let (r0, c0, ph, pw) = rects[p];
            let mvd = read_mvd(d, ctxs, pic, mb_x, mb_y, r0, c0, 1)?;
            write_mvd_abs(pic, mb_x, mb_y, r0, c0, ph, pw, 1, mvd);
            mvd_l1[p] = Some(mvd);
        }
    }
    for p in 0..2 {
        let (r0, c0, ph, pw) = rects[p];
        compensate_partition(
            sh, bctx, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, ph, pw, dirs[p], ref_l0[p],
            ref_l1[p], mvd_l0[p], mvd_l1[p],
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// B_8x8 — four 8x8 sub-MBs.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_b_8x8(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sh: &SliceHeader,
    sps: &Sps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    bctx: &BSliceCtx<'_>,
) -> Result<[BSubPartition; 4]> {
    let num_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let num_l1 = sh.num_ref_idx_l1_active_minus1 + 1;

    // Step 1: sub_mb_type for each of the four 8×8 sub-MBs.
    let mut subs = [BSubPartition::Direct8x8; 4];
    for s in 0..4 {
        let raw = {
            let slice = &mut ctxs[CTX_IDX_SUB_MB_TYPE_B..CTX_IDX_SUB_MB_TYPE_B + 4];
            binarize::decode_sub_mb_type_b(d, slice)?
        };
        subs[s] = decode_b_sub_mb_type(raw)
            .ok_or_else(|| Error::invalid(format!("h264 cabac b-slice: bad sub_mb_type {raw}")))?;
    }

    // Step 2: ref_idx_l0 for non-direct L0/BiPred sub-MBs.
    let mut ref_l0 = [None; 4];
    let mut ref_l1 = [None; 4];
    for s in 0..4 {
        if let Some(dir) = subs[s].pred_dir() {
            if uses_l0(dir) {
                let (sr0, sc0) = SUB_OFFSETS[s];
                ref_l0[s] = Some(read_ref_idx_lx(
                    d, ctxs, pic, mb_x, mb_y, sr0, sc0, num_l0, 0,
                )?);
            }
        }
    }
    for s in 0..4 {
        if let Some(dir) = subs[s].pred_dir() {
            if uses_l1(dir) {
                let (sr0, sc0) = SUB_OFFSETS[s];
                ref_l1[s] = Some(read_ref_idx_lx(
                    d, ctxs, pic, mb_x, mb_y, sr0, sc0, num_l1, 1,
                )?);
            }
        }
    }

    // Step 3: MVDs per sub-partition, list 0 first then list 1. Because
    // neighbour MV lookups for later sub-partitions depend on earlier ones,
    // we read + compensate each sub-MB in a single pass (the non-direct path
    // reads MVDs and runs MC; the direct path uses direct-mode MVs).
    //
    // Spec requires: all mvd_l0 across all sub-partitions, then all mvd_l1.
    // We pre-read both lists here, then do compensation afterwards.
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
        for sub_idx in 0..n {
            if let Some(dir) = dir_opt {
                let (sr0, sc0) = SUB_OFFSETS[s];
                let (dr, dc, sh_h, sh_w) = sp.sub_rect(sub_idx);
                let r0 = sr0 + dr;
                let c0 = sc0 + dc;
                let mvd_l0 = if uses_l0(dir) {
                    let mvd = read_mvd(d, ctxs, pic, mb_x, mb_y, r0, c0, 0)?;
                    write_mvd_abs(pic, mb_x, mb_y, r0, c0, sh_h, sh_w, 0, mvd);
                    Some(mvd)
                } else {
                    None
                };
                let mvd_l1 = if uses_l1(dir) {
                    let mvd = read_mvd(d, ctxs, pic, mb_x, mb_y, r0, c0, 1)?;
                    write_mvd_abs(pic, mb_x, mb_y, r0, c0, sh_h, sh_w, 1, mvd);
                    Some(mvd)
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

    // Step 4: compensate each sub-MB.
    for s in 0..4 {
        let sp = subs[s];
        let (sr0, sc0) = SUB_OFFSETS[s];
        if matches!(sp, BSubPartition::Direct8x8) {
            // §9.3.3.1.1.6 / .7 — mark this sub-MB's 2×2 4×4 grid as
            // "direct" for downstream ctxIdxInc lookups. Mirrors FFmpeg's
            // `fill_rectangle(&direct_cache[scan8[4*i]], 2, 2, 8, ...)` at
            // h264_cabac.c:2190.
            {
                let info = pic.mb_info_mut(mb_x, mb_y);
                for rr in sr0..sr0 + 2 {
                    for cc in sc0..sc0 + 2 {
                        info.b_direct_per_block[rr * 4 + cc] = true;
                    }
                }
            }
            direct_8x8_compensate(
                sh, sps, bctx, mb_x, mb_y, sr0, sc0, pic, ref_list0, ref_list1,
            )?;
            continue;
        }
        let dir = sp.pred_dir().expect("non-direct");
        for sub_idx in 0..sp.num_sub_partitions() {
            let (dr, dc, sh_h, sh_w) = sp.sub_rect(sub_idx);
            let r0 = sr0 + dr;
            let c0 = sc0 + dc;
            let m = mvds[s][sub_idx];
            compensate_partition(
                sh, bctx, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, sh_h, sh_w, dir,
                ref_l0[s], ref_l1[s], m.mvd_l0, m.mvd_l1,
            )?;
        }
    }
    Ok(subs)
}

// ---------------------------------------------------------------------------
// Partition-level MV derivation + motion compensation.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn compensate_partition(
    sh: &SliceHeader,
    bctx: &BSliceCtx<'_>,
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
    // Derive list-specific MVs via the median predictor + MVD add.
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
    // Publish MV + ref_idx + abs-MVD state before MC. §9.3.3.1.1.7 needs
    // the absolute MVD magnitudes of the left/above neighbour blocks to
    // derive the ctxIdxInc for later partitions within this MB (and the
    // first-row blocks of the MB below / first-col blocks of the MB to
    // the right). FFmpeg tracks this via `mvd_cache[list]` per scan8
    // entry; we fold it into `mb_info.mvd_lX_abs`.
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        let abs0 = mvd_l0.map(|(x, y)| {
            (
                (x.unsigned_abs()).min(u16::MAX as u32) as u16,
                (y.unsigned_abs()).min(u16::MAX as u32) as u16,
            )
        });
        let abs1 = mvd_l1.map(|(x, y)| {
            (
                (x.unsigned_abs()).min(u16::MAX as u32) as u16,
                (y.unsigned_abs()).min(u16::MAX as u32) as u16,
            )
        });
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
                if let Some(a) = abs0 {
                    info.mvd_l0_abs[idx] = a;
                }
                if let Some(a) = abs1 {
                    info.mvd_l1_abs[idx] = a;
                }
            }
        }
    }
    crate::b_mb::apply_motion_compensation_b(
        sh, bctx, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, ph, pw, dir, ref_idx_l0,
        ref_idx_l1, mv_l0, mv_l1,
    )
}

// ---------------------------------------------------------------------------
// Direct-mode dispatch — reuse the CAVLC compensate helpers verbatim.
// ---------------------------------------------------------------------------

fn direct_16x16_compensate(
    sh: &SliceHeader,
    sps: &Sps,
    bctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
) -> Result<()> {
    crate::b_mb::direct_16x16_compensate_pub(sh, sps, bctx, mb_x, mb_y, pic, ref_list0, ref_list1)
}

#[allow(clippy::too_many_arguments)]
fn direct_8x8_compensate(
    sh: &SliceHeader,
    sps: &Sps,
    bctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    sr0: usize,
    sc0: usize,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
) -> Result<()> {
    crate::b_mb::direct_8x8_compensate_pub(
        sh, sps, bctx, mb_x, mb_y, sr0, sc0, pic, ref_list0, ref_list1,
    )
}

// ---------------------------------------------------------------------------
// Intra-in-B — dispatch to the I-slice CABAC path.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_intra_in_b(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    prev_qp: &mut i32,
    imb: IMbType,
) -> Result<()> {
    super::mb::decode_intra_in_p_cabac(d, ctxs, sh, sps, pps, mb_x, mb_y, pic, prev_qp, imb)
}

// ---------------------------------------------------------------------------
// MVD + ref_idx helpers — shared between 16x16, 16x8/8x16, B_8x8 paths.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) fn read_ref_idx_lx(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    num_ref: u32,
    list: u8,
) -> Result<i8> {
    if num_ref <= 1 {
        return Ok(0);
    }
    let inc = ref_idx_ctx_idx_inc_at(pic, mb_x, mb_y, r0, c0, list);
    let base = if list == 0 {
        CTX_IDX_REF_IDX_L0
    } else {
        CTX_IDX_REF_IDX_L1
    };
    let slice = &mut ctxs[base..base + 6];
    let raw = binarize::decode_ref_idx_lx(d, slice, inc)?;
    if raw >= num_ref {
        return Err(Error::invalid(format!(
            "h264 cabac b-slice: ref_idx_l{list} {raw} out of range (num={num_ref})"
        )));
    }
    Ok(raw as i8)
}

/// Publish the absolute MVD magnitudes for a partition's 4×4 blocks
/// into `mvd_l{0,1}_abs`. §9.3.3.1.1.7 reads the neighbour block's
/// abs-MVD for the MVD ctxIdxInc, so a later partition within the
/// same MB needs to see the earlier partition's magnitudes. FFmpeg
/// tracks this via `mvd_cache[list]` and writes it inline with every
/// `DECODE_CABAC_MB_MVD` invocation.
pub(crate) fn write_mvd_abs(
    pic: &mut Picture,
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    ph: usize,
    pw: usize,
    list: u8,
    mvd: (i32, i32),
) {
    let ax = (mvd.0.unsigned_abs()).min(u16::MAX as u32) as u16;
    let ay = (mvd.1.unsigned_abs()).min(u16::MAX as u32) as u16;
    let info = pic.mb_info_mut(mb_x, mb_y);
    let dst = if list == 0 {
        &mut info.mvd_l0_abs
    } else {
        &mut info.mvd_l1_abs
    };
    for rr in r0..r0 + ph {
        for cc in c0..c0 + pw {
            dst[rr * 4 + cc] = (ax, ay);
        }
    }
}

pub(crate) fn read_mvd(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    list: u8,
) -> Result<(i32, i32)> {
    let inc_x = mvd_ctx_idx_inc(pic, mb_x, mb_y, r0, c0, true, list);
    let inc_y = mvd_ctx_idx_inc(pic, mb_x, mb_y, r0, c0, false, list);
    let (base_x, base_y) = if list == 0 {
        (CTX_IDX_MVD_L0_X, CTX_IDX_MVD_L0_Y)
    } else {
        (CTX_IDX_MVD_L1_X, CTX_IDX_MVD_L1_Y)
    };
    let mvd_x = {
        let slice = &mut ctxs[base_x..base_x + 7];
        binarize::decode_mvd_component(d, slice, inc_x)?
    };
    let mvd_y = {
        let slice = &mut ctxs[base_y..base_y + 7];
        binarize::decode_mvd_component(d, slice, inc_y)?
    };
    Ok((mvd_x, mvd_y))
}

// ---------------------------------------------------------------------------
// Residual decode — inter variant (gated on `intra = false`).
// ---------------------------------------------------------------------------

fn decode_inter_residual_luma(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
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
        let neighbours = cbf_neighbours_luma(pic, mb_x, mb_y, br_row, br_col);
        let coeffs =
            decode_residual_block_in_place(d, ctxs, BlockCat::Luma4x4, &neighbours, 16, false)?;
        let mut residual = coeffs;
        let total_coeff = coeffs.iter().filter(|&&v| v != 0).count() as u32;
        // §7.4.2.2 — Inter-Y 4×4 slot 3.
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

// Inter luma residual with `transform_size_8x8_flag = 1` (B-slice). Mirrors
// the P-slice CABAC 8×8 path (`cabac::p_mb::decode_inter_residual_luma_8x8`)
// — pulls the 64-coefficient block through `decode_luma_8x8_residual_in_place`
// (ctxBlockCat = 5, §9.3.3.1.1.9). `luma_nc` is populated per-4×4 with a
// consolidated NNZ so neighbour MV/mode derivations on subsequent MBs see a
// consistent "this 8×8 had residual" signal.
fn decode_inter_residual_luma_8x8(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    cbp_luma: u8,
    qp_y: i32,
) -> Result<()> {
    let lstride = pic.luma_stride();
    let lo_mb = pic.luma_off(mb_x, mb_y);

    for blk8 in 0..4usize {
        if (cbp_luma >> blk8) & 1 == 0 {
            continue;
        }
        let br8_row = blk8 >> 1;
        let br8_col = blk8 & 1;
        let r0 = br8_row * 2;
        let c0 = br8_col * 2;

        let neighbours = cbf_neighbours_luma(pic, mb_x, mb_y, r0, c0);
        let coeffs = decode_luma_8x8_residual_in_place(d, ctxs, &neighbours, false)?;
        let mut residual = coeffs;
        let nnz = residual.iter().filter(|&&v| v != 0).count() as u16;
        // §7.4.2.2 — Inter-Y 8×8 uses scaling-list slot 1.
        let scale = *pic.scaling_lists.matrix_8x8(1);
        dequantize_8x8_scaled(&mut residual, qp_y, &scale);
        idct_8x8(&mut residual);

        let info = pic.mb_info_mut(mb_x, mb_y);
        let byte = nnz.min(16) as u8;
        info.luma_nc[r0 * 4 + c0] = byte;
        info.luma_nc[r0 * 4 + c0 + 1] = byte;
        info.luma_nc[(r0 + 1) * 4 + c0] = byte;
        info.luma_nc[(r0 + 1) * 4 + c0 + 1] = byte;

        let lo = lo_mb + (br8_row * 8) * lstride + br8_col * 8;
        for r in 0..8 {
            for c in 0..8 {
                let base = pic.y[lo + r * lstride + c] as i32;
                let v = base + residual[r * 8 + c];
                pic.y[lo + r * lstride + c] = v.clamp(0, 255) as u8;
            }
        }
    }
    Ok(())
}

fn decode_inter_residual_chroma(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
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
        let neigh = CbfNeighbours::none();
        let cb_coeffs =
            decode_residual_block_in_place(d, ctxs, BlockCat::ChromaDc, &neigh, 4, false)?;
        for i in 0..4 {
            dc_cb[i] = cb_coeffs[i];
        }
        let cr_coeffs =
            decode_residual_block_in_place(d, ctxs, BlockCat::ChromaDc, &neigh, 4, false)?;
        for i in 0..4 {
            dc_cr[i] = cr_coeffs[i];
        }
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
                let neigh = CbfNeighbours::none();
                let ac =
                    decode_residual_block_in_place(d, ctxs, BlockCat::ChromaAc, &neigh, 15, false)?;
                total_coeff = ac.iter().filter(|&&v| v != 0).count() as u32;
                res = ac;
                // §7.4.2.2 — Inter chroma slots 4 (Cb) / 5 (Cr).
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
                    plane[co + off_in_mb + r * cstride + c] =
                        (base + res[r * 4 + c]).clamp(0, 255) as u8;
                }
            }
            nc_arr[(br_row << 1) | br_col] = total_coeff as u8;
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

// ---------------------------------------------------------------------------
// ctxIdxInc derivations.
// ---------------------------------------------------------------------------

pub(crate) fn mb_skip_flag_b_ctx_idx_inc(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
    // §9.3.3.1.1.1: condTermFlagN = 0 when neighbour N is B_Skip, 1
    // otherwise (unavailable neighbours contribute 0). `ctxIdxInc = A + B`.
    let a = mb_x > 0 && !pic.mb_info_at(mb_x - 1, mb_y).skipped;
    let b = mb_y > 0 && !pic.mb_info_at(mb_x, mb_y - 1).skipped;
    (a as u8) + (b as u8)
}

pub(crate) fn mb_type_b_ctx_idx_inc(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
    // §9.3.3.1.1.3 B-slice bin 0: condTermFlagN = 0 iff mbN is unavailable
    // or mbN is `B_Skip` or `B_Direct_16x16`, else 1. Mirrors FFmpeg's
    // `decode_cabac_mb_type` at h264_cabac.c:2032 — `!IS_DIRECT(top_type)`
    // checks MB-level MB_TYPE_DIRECT2 which is set for B_Skip and
    // B_Direct_16x16 only (B_8x8 with direct sub-MBs does NOT set it).
    let cond = |m: &crate::picture::MbInfo| -> bool {
        if !m.coded {
            return false;
        }
        if m.skipped {
            // B_Skip → condTermFlagN = 0.
            return false;
        }
        if matches!(
            m.mb_type_b,
            Some(crate::mb_type::BMbType::Inter {
                partition: crate::mb_type::BPartition::Direct16x16,
            })
        ) {
            return false;
        }
        true
    };
    let a = mb_x > 0 && cond(pic.mb_info_at(mb_x - 1, mb_y));
    let b = mb_y > 0 && cond(pic.mb_info_at(mb_x, mb_y - 1));
    (a as u8) + (b as u8)
}

fn ref_idx_ctx_idx_inc_at(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    list: u8,
) -> u8 {
    // §9.3.3.1.1.6 — FFmpeg `decode_cabac_mb_ref`: for B slices, a
    // neighbour block whose `direct_cache` bit is set contributes 0 to
    // ctxIdxInc regardless of its `ref_idx_lX`. For P slices it's just
    // `ref_idx > 0` with no direct gating.
    let a = neighbour_ref_idx_gt_zero_b(
        pic,
        mb_x as i32,
        mb_y as i32,
        r0 as i32,
        c0 as i32 - 1,
        list,
    );
    let b = neighbour_ref_idx_gt_zero_b(
        pic,
        mb_x as i32,
        mb_y as i32,
        r0 as i32 - 1,
        c0 as i32,
        list,
    );
    (a as u8) + 2 * (b as u8)
}

fn neighbour_ref_idx_gt_zero_b(
    pic: &Picture,
    mb_x: i32,
    mb_y: i32,
    row: i32,
    col: i32,
    list: u8,
) -> bool {
    let (mbx, mby, r, c) = match wrap_neighbour(mb_x, mb_y, row, col, pic.mb_width, pic.mb_height) {
        Some(v) => v,
        None => return false,
    };
    let info = pic.mb_info_at(mbx, mby);
    if !info.coded || info.intra {
        return false;
    }
    // B_Skip / B_Direct_16x16 / B8x8 Direct_8x8 sub → direct-cache gate
    // contributes 0 regardless of ref_idx.
    if info.b_direct_per_block[(r * 4 + c) as usize] {
        return false;
    }
    let ref_idx = if list == 0 {
        info.ref_idx_l0[(r * 4 + c) as usize]
    } else {
        info.ref_idx_l1[(r * 4 + c) as usize]
    };
    ref_idx > 0
}

fn mvd_ctx_idx_inc(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    is_x: bool,
    list: u8,
) -> u32 {
    // §9.3.3.1.1.7: absMvdComp(A) + absMvdComp(B) — summed across list-specific
    // neighbour magnitudes. Using the combined L0/L1 magnitudes matches JM's
    // neighbour-based predictor.
    let a = neighbour_abs_mvd(
        pic,
        mb_x as i32,
        mb_y as i32,
        r0 as i32,
        c0 as i32 - 1,
        is_x,
        list,
    );
    let b = neighbour_abs_mvd(
        pic,
        mb_x as i32,
        mb_y as i32,
        r0 as i32 - 1,
        c0 as i32,
        is_x,
        list,
    );
    a as u32 + b as u32
}

fn neighbour_abs_mvd(
    pic: &Picture,
    mb_x: i32,
    mb_y: i32,
    row: i32,
    col: i32,
    is_x: bool,
    list: u8,
) -> u16 {
    let (mbx, mby, r, c) = match wrap_neighbour(mb_x, mb_y, row, col, pic.mb_width, pic.mb_height) {
        Some(v) => v,
        None => return 0,
    };
    let info = pic.mb_info_at(mbx, mby);
    if !info.coded || info.intra {
        return 0;
    }
    // §9.3.3.1.1.7 — FFmpeg's `mvd_cache` is zeroed on direct / skip
    // blocks (no MVD was coded). Our per-MB `mvd_lX_abs` can hold stale
    // magnitudes from a previous decode; gate by the direct-cache flag
    // so the amvd accumulator mirrors FFmpeg's.
    if info.b_direct_per_block[(r * 4 + c) as usize] {
        return 0;
    }
    let (x, y) = if list == 0 {
        info.mvd_l0_abs[(r * 4 + c) as usize]
    } else {
        info.mvd_l1_abs[(r * 4 + c) as usize]
    };
    if is_x {
        x
    } else {
        y
    }
}

fn wrap_neighbour(
    mb_x: i32,
    mb_y: i32,
    row: i32,
    col: i32,
    mb_w: u32,
    mb_h: u32,
) -> Option<(u32, u32, u32, u32)> {
    let mut mbx = mb_x;
    let mut mby = mb_y;
    let mut r = row;
    let mut c = col;
    while c < 0 {
        mbx -= 1;
        c += 4;
    }
    while c >= 4 {
        mbx += 1;
        c -= 4;
    }
    while r < 0 {
        mby -= 1;
        r += 4;
    }
    while r >= 4 {
        mby += 1;
        r -= 4;
    }
    if mbx < 0 || mby < 0 || mbx as u32 >= mb_w || mby as u32 >= mb_h {
        return None;
    }
    Some((mbx as u32, mby as u32, r as u32, c as u32))
}

fn cbf_neighbours_luma(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    br_row: usize,
    br_col: usize,
) -> CbfNeighbours {
    let info_here = pic.mb_info_at(mb_x, mb_y);
    let left = if br_col > 0 {
        Some(info_here.luma_nc[br_row * 4 + br_col - 1] != 0)
    } else if mb_x > 0 {
        let info = pic.mb_info_at(mb_x - 1, mb_y);
        if info.coded {
            Some(info.luma_nc[br_row * 4 + 3] != 0)
        } else {
            None
        }
    } else {
        None
    };
    let above = if br_row > 0 {
        Some(info_here.luma_nc[(br_row - 1) * 4 + br_col] != 0)
    } else if mb_y > 0 {
        let info = pic.mb_info_at(mb_x, mb_y - 1);
        if info.coded {
            Some(info.luma_nc[12 + br_col] != 0)
        } else {
            None
        }
    } else {
        None
    };
    CbfNeighbours { left, above }
}

// ---------------------------------------------------------------------------
// List-direction helpers.
// ---------------------------------------------------------------------------

fn uses_l0(dir: PredDir) -> bool {
    matches!(dir, PredDir::L0 | PredDir::BiPred)
}
fn uses_l1(dir: PredDir) -> bool {
    matches!(dir, PredDir::L1 | PredDir::BiPred)
}
