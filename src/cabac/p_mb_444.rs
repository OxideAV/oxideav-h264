//! CABAC P- and B-slice macroblock decode for 4:4:4 (ChromaArrayType == 3).
//!
//! Structure mirrors the CAVLC 4:4:4 P / B paths in [`crate::p_mb_444`] /
//! [`crate::b_mb_444`]:
//!
//! * `mb_skip_flag` / `mb_type` are decoded through the shared CABAC
//!   ctxIdx banks at §9.3.3.1.1.1 / §9.3.3.1.1.3.
//! * Intra-in-P / intra-in-B re-enter the 4:4:4 I path in
//!   [`crate::cabac::mb_444`] so the intra-predictor + Table 9-42 Cb/Cr
//!   residual banks stay owned by one module.
//! * Inter partitions (P_L0_16×16 / P_L0_16×8 / P_L0_8×16 / P_8×8) read
//!   `ref_idx_l0` + two `mvd_l0` components per partition through CABAC,
//!   then run §8.4.2.2 Table 8-9 luma-filter motion compensation on all
//!   three planes via [`crate::p_mb_444::mc_partition_all_planes`].
//! * `coded_block_pattern` is parsed with `chroma_format_idc = 0` so the
//!   chroma bins are skipped — the 4-bit luma-only CBP gates every
//!   plane's residual stream (FFmpeg's `decode_chroma = false` rule).
//! * Per-plane residual decode reuses `decode_residual_block_in_place_plane`
//!   with `ResidualPlane::Luma / Cb444 / Cr444` so each plane walks its
//!   own §9.3.3.1.1.9 Table 9-42 context bank.
//!
//! B-slice inter partitions are not covered by this pass (they need the
//! full bi-prediction + direct-mode machinery); only intra-in-B and
//! B_Skip / B_Direct_16×16 route through the helpers here.

use oxideav_core::{Error, Result};

use crate::b_mb::BSliceCtx;
use crate::cabac::binarize;
use crate::cabac::context::CabacContext;
use crate::cabac::engine::CabacDecoder;
use crate::cabac::mb::{
    cbp_luma_ctx_idx_inc, decode_residual_block_in_place_plane,
    transform_size_8x8_flag_ctx_idx_inc, ResidualPlane,
};
use crate::cabac::mb_444::decode_intra_in_p_cabac_444;
use crate::cabac::p_mb as cabac_p;
use crate::cabac::residual::{BlockCat, CbfNeighbours};
use crate::cabac::tables::{
    CTX_IDX_CODED_BLOCK_PATTERN_LUMA, CTX_IDX_MB_QP_DELTA, CTX_IDX_MB_SKIP_FLAG_B,
    CTX_IDX_MB_SKIP_FLAG_P, CTX_IDX_MB_TYPE_B, CTX_IDX_MB_TYPE_I, CTX_IDX_MB_TYPE_INTRA_IN_P,
    CTX_IDX_MB_TYPE_P, CTX_IDX_REF_IDX_L0, CTX_IDX_SUB_MB_TYPE_B, CTX_IDX_TRANSFORM_SIZE_8X8_FLAG,
};
use crate::mb::LUMA_BLOCK_RASTER;
use crate::mb_444::{self as mb444, Plane};
use crate::mb_type::{
    decode_b_slice_mb_type, decode_b_sub_mb_type, decode_i_slice_mb_type, BMbType, BSubPartition,
    IMbType, PMbType, PPartition, PredDir,
};
use crate::motion::predict_mv_l0;
use crate::p_mb_444::{l0_luma_weight, lookup_ref_444, mc_partition_all_planes};
use crate::picture::{Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::SliceHeader;
use crate::sps::Sps;
use crate::transform::{chroma_qp, dequantize_4x4_scaled, idct_4x4};

// MVD contexts are laid out as two 7-slot banks for list-0 components.
const CTX_IDX_MVD_L0_X: usize = 40;
const CTX_IDX_MVD_L0_Y: usize = 47;

/// Decode one P-slice macroblock under ChromaArrayType = 3 through
/// CABAC. Intra-in-P routes to [`decode_intra_in_p_cabac_444`]; inter
/// partitions flow through the local 4:4:4 inter helpers.
#[allow(clippy::too_many_arguments)]
pub fn decode_p_mb_cabac_444(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    prev_qp: &mut i32,
) -> Result<bool> {
    debug_assert_eq!(sps.chroma_format_idc, 3);
    // §9.3.3.1.1.1 — mb_skip_flag for P slices.
    let skip_inc = cabac_p::mb_skip_flag_ctx_idx_inc(pic, mb_x, mb_y);
    let skipped = {
        let slice = &mut ctxs[CTX_IDX_MB_SKIP_FLAG_P..CTX_IDX_MB_SKIP_FLAG_P + 3];
        binarize::decode_mb_skip_flag_p(d, slice, skip_inc)?
    };
    if skipped {
        crate::p_mb_444::decode_p_skip_mb_444(sh, mb_x, mb_y, pic, ref_list0, *prev_qp)?;
        return Ok(true);
    }

    // §9.3.3.1.1.3 — mb_type for P/SP.
    let mb_type_raw = {
        let slice = &mut ctxs[CTX_IDX_MB_TYPE_P..CTX_IDX_MB_TYPE_P + 7];
        binarize::decode_mb_type_p(d, slice, CTX_IDX_MB_TYPE_INTRA_IN_P - CTX_IDX_MB_TYPE_P)?
    };

    match mb_type_raw {
        0 => decode_p_inter_444(
            d,
            ctxs,
            sh,
            pps,
            mb_x,
            mb_y,
            pic,
            ref_list0,
            prev_qp,
            PPartition::P16x16,
        )
        .map(|_| false),
        1 => decode_p_inter_444(
            d,
            ctxs,
            sh,
            pps,
            mb_x,
            mb_y,
            pic,
            ref_list0,
            prev_qp,
            PPartition::P16x8,
        )
        .map(|_| false),
        2 => decode_p_inter_444(
            d,
            ctxs,
            sh,
            pps,
            mb_x,
            mb_y,
            pic,
            ref_list0,
            prev_qp,
            PPartition::P8x16,
        )
        .map(|_| false),
        3 | 4 => {
            // P_8x8 / P_8x8_Ref0 not yet wired under CABAC 4:4:4 — the
            // sub_mb_type decode machinery is shared but the 4 × sub-8×8
            // MV reads need a separate helper; the testsrc fixtures
            // primarily exercise P16x16 / P8x16 / P16x8 / intra-in-P.
            Err(Error::unsupported(
                "h264: CABAC 4:4:4 P_8x8 not yet wired — P16x16/16x8/8x16 + intra-in-P + P_Skip only",
            ))
        }
        5..=30 => {
            let imb = decode_i_slice_mb_type(mb_type_raw - 5).ok_or_else(|| {
                Error::invalid(format!(
                    "h264 cabac 4:4:4 p-slice: bad intra-in-P mb_type {mb_type_raw}"
                ))
            })?;
            decode_intra_in_p_cabac_444(d, ctxs, sh, sps, pps, mb_x, mb_y, pic, prev_qp, imb)?;
            Ok(false)
        }
        _ => Err(Error::invalid(format!(
            "h264 cabac 4:4:4 p-slice: unexpected mb_type {mb_type_raw}"
        ))),
    }
}

/// Decode one B-slice macroblock under ChromaArrayType = 3 through
/// CABAC. Intra-in-B + B_Skip are wired; true B inter (bipred / direct)
/// lands on Error::Unsupported.
#[allow(clippy::too_many_arguments)]
pub fn decode_b_mb_cabac_444(
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
    debug_assert_eq!(sps.chroma_format_idc, 3);
    // §9.3.3.1.1.1 — mb_skip_flag for B slices.
    let skip_inc = mb_skip_flag_b_ctx_idx_inc(pic, mb_x, mb_y);
    let skipped = {
        let slice = &mut ctxs[CTX_IDX_MB_SKIP_FLAG_B..CTX_IDX_MB_SKIP_FLAG_B + 3];
        binarize::decode_mb_skip_flag_b(d, slice, skip_inc)?
    };
    if skipped {
        crate::b_mb_444::decode_b_skip_mb_444(
            sh, sps, mb_x, mb_y, pic, ref_list0, ref_list1, bctx, *prev_qp,
        )?;
        return Ok(true);
    }

    // §9.3.3.1.1.3 — B-slice mb_type.
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

    if mb_type_raw == 48 {
        decode_intra_in_p_cabac_444(
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
        )?;
        return Ok(false);
    }
    let bmb = decode_b_slice_mb_type(mb_type_raw).ok_or_else(|| {
        Error::invalid(format!(
            "h264 cabac 4:4:4 b-slice: bad mb_type {mb_type_raw}"
        ))
    })?;
    match bmb {
        BMbType::IntraInB(imb) => {
            decode_intra_in_p_cabac_444(d, ctxs, sh, sps, pps, mb_x, mb_y, pic, prev_qp, imb)?;
            Ok(false)
        }
        BMbType::Inter { partition } => {
            decode_b_inter_444(
                d, ctxs, sh, sps, pps, mb_x, mb_y, pic, ref_list0, ref_list1, bctx, prev_qp,
                partition,
            )?;
            Ok(false)
        }
    }
}

// ---------------------------------------------------------------------------
// B inter — Direct16x16 + L0/L1/Bi 16x16 (+ 16x8 / 8x16 two-part); B_8x8
// still returns Unsupported.
// ---------------------------------------------------------------------------

// CABAC 4:4:4 B-inter decode — dispatches by partition, reads list-0/1
// ref_idx and MVDs in §7.3.5.1 syntax order, then runs the 4:4:4
// luma-filter motion compensation on all three planes, followed by
// three plane-aligned luma-style residual streams gated by the 4-bit
// inter CBP.
#[allow(clippy::too_many_arguments)]
fn decode_b_inter_444(
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
    partition: crate::mb_type::BPartition,
) -> Result<()> {
    use crate::mb_type::{BPartition, PredDir};
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        *info = crate::picture::MbInfo {
            qp_y: *prev_qp,
            coded: true,
            intra: false,
            skipped: false,
            ..Default::default()
        };
    }

    match partition {
        BPartition::Direct16x16 => {
            // §8.4.1.2 direct — derive MVs via the shared helper and run MC
            // on 4:4:4 directly. We temporarily clone the picture to 4:2:0
            // scratch so the CAVLC direct-mode compensator (written against
            // 4:2:0 planes) can do its work, then mirror the MB MV state
            // back and re-apply MC on 4:4:4 via `apply_mvs_444`.
            let (mvs, dirs) =
                snapshot_b_direct_mv_state(sh, sps, bctx, mb_x, mb_y, pic, ref_list0, ref_list1)?;
            crate::b_mb_444::apply_mvs_444(
                pic, ref_list0, ref_list1, sh, bctx, mb_x, mb_y, &mvs, &dirs,
            );
        }
        BPartition::L0_16x16 => {
            decode_b_16x16_single(
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
            decode_b_16x16_single(
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
            decode_b_16x16_single(
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
            decode_b_two_partition(
                d, ctxs, sh, bctx, mb_x, mb_y, pic, ref_list0, ref_list1, &rects, &dirs,
            )?;
        }
        BPartition::TwoPart8x16 { dirs } => {
            let rects = [(0usize, 0usize, 4usize, 2usize), (0, 2, 4, 2)];
            decode_b_two_partition(
                d, ctxs, sh, bctx, mb_x, mb_y, pic, ref_list0, ref_list1, &rects, &dirs,
            )?;
        }
        BPartition::B8x8 => {
            decode_b_8x8_444(
                d, ctxs, sh, sps, bctx, mb_x, mb_y, pic, ref_list0, ref_list1,
            )?;
        }
    }

    // §9.3.3.1.1.4 — coded_block_pattern. 4:4:4 inter emits 4 luma bits only.
    let cbp_luma = {
        let slice =
            &mut ctxs[CTX_IDX_CODED_BLOCK_PATTERN_LUMA..CTX_IDX_CODED_BLOCK_PATTERN_LUMA + 12];
        let cbp = binarize::decode_coded_block_pattern(
            d,
            slice,
            0,
            |i, decoded| cbp_luma_ctx_idx_inc(pic, mb_x, mb_y, i, decoded),
            |_bin| 0u8,
        )?;
        (cbp & 0x0F) as u8
    };

    let transform_8x8 = if pps.transform_8x8_mode_flag && cbp_luma != 0 {
        let inc = transform_size_8x8_flag_ctx_idx_inc(pic, mb_x, mb_y);
        let slice = &mut ctxs[CTX_IDX_TRANSFORM_SIZE_8X8_FLAG..CTX_IDX_TRANSFORM_SIZE_8X8_FLAG + 3];
        binarize::decode_transform_size_8x8_flag(d, slice, inc)?
    } else {
        false
    };
    if transform_8x8 {
        return Err(Error::unsupported(
            "h264: CABAC 4:4:4 B inter transform_size_8x8_flag not wired — 4×4 transform only",
        ));
    }

    let needs_qp = cbp_luma != 0;
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
    let qp_cb = chroma_qp(qp_y, pps.chroma_qp_index_offset);
    let qp_cr = chroma_qp(qp_y, pps.second_chroma_qp_index_offset);

    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        info.qp_y = qp_y;
        info.intra4x4_pred_mode = [INTRA_DC_FAKE; 16];
        info.cbp_luma = cbp_luma;
        info.cbp_chroma = 0;
    }
    for (plane, qp, scale_idx) in [
        (Plane::Y, qp_y, 3usize),
        (Plane::Cb, qp_cb, 4usize),
        (Plane::Cr, qp_cr, 5usize),
    ] {
        decode_plane_inter_residual_4x4_cabac(
            d, ctxs, mb_x, mb_y, pic, plane, cbp_luma, qp, scale_idx,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_b_16x16_single(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sh: &SliceHeader,
    bctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    dir: crate::mb_type::PredDir,
) -> Result<()> {
    use crate::mb_type::PredDir;
    let uses_l0 = matches!(dir, PredDir::L0 | PredDir::BiPred);
    let uses_l1 = matches!(dir, PredDir::L1 | PredDir::BiPred);
    let ref_l0 = if uses_l0 {
        Some(crate::cabac::b_mb::read_ref_idx_lx(
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
    let ref_l1 = if uses_l1 {
        Some(crate::cabac::b_mb::read_ref_idx_lx(
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
    let mvd_l0 = if uses_l0 {
        let mvd = crate::cabac::b_mb::read_mvd(d, ctxs, pic, mb_x, mb_y, 0, 0, 0)?;
        crate::cabac::b_mb::write_mvd_abs(pic, mb_x, mb_y, 0, 0, 4, 4, 0, mvd);
        Some(mvd)
    } else {
        None
    };
    let mvd_l1 = if uses_l1 {
        let mvd = crate::cabac::b_mb::read_mvd(d, ctxs, pic, mb_x, mb_y, 0, 0, 1)?;
        crate::cabac::b_mb::write_mvd_abs(pic, mb_x, mb_y, 0, 0, 4, 4, 1, mvd);
        Some(mvd)
    } else {
        None
    };
    crate::b_mb_444::compensate_444_partition(
        sh, bctx, pic, ref_list0, ref_list1, mb_x, mb_y, 0, 0, 4, 4, dir, ref_l0, ref_l1, mvd_l0,
        mvd_l1,
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_b_two_partition(
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
    dirs: &[crate::mb_type::PredDir; 2],
) -> Result<()> {
    use crate::mb_type::PredDir;
    let uses_l0 = |d: PredDir| matches!(d, PredDir::L0 | PredDir::BiPred);
    let uses_l1 = |d: PredDir| matches!(d, PredDir::L1 | PredDir::BiPred);
    let num_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let num_l1 = sh.num_ref_idx_l1_active_minus1 + 1;
    let mut ref_l0 = [None; 2];
    let mut ref_l1 = [None; 2];
    let mut mvd_l0 = [None; 2];
    let mut mvd_l1 = [None; 2];
    // §7.3.5.1 — ref_idx_l0 list in raster order, then ref_idx_l1, then MVDs.
    for p in 0..2 {
        if uses_l0(dirs[p]) {
            let (r0, c0, _, _) = rects[p];
            ref_l0[p] = Some(crate::cabac::b_mb::read_ref_idx_lx(
                d, ctxs, pic, mb_x, mb_y, r0, c0, num_l0, 0,
            )?);
        }
    }
    for p in 0..2 {
        if uses_l1(dirs[p]) {
            let (r0, c0, _, _) = rects[p];
            ref_l1[p] = Some(crate::cabac::b_mb::read_ref_idx_lx(
                d, ctxs, pic, mb_x, mb_y, r0, c0, num_l1, 1,
            )?);
        }
    }
    // §9.3.3.1.1.7 — each MVD's ctxIdxInc depends on abs-MVD of left +
    // above neighbour blocks in the current MB's scan8 layout. FFmpeg
    // (`h264_cabac.c::DECODE_CABAC_MB_MVD`) fills `mvd_cache[list]`
    // immediately after each MVD decode via `fill_rectangle`, before
    // the next MVD decode. Mirror that here by writing `mvd_lX_abs`
    // into `pic.mb_info` as we go; otherwise a 16×8 / 8×16 second
    // partition's MVD ctx drifts because the first partition's blocks
    // still read back 0.
    for p in 0..2 {
        if uses_l0(dirs[p]) {
            let (r0, c0, ph, pw) = rects[p];
            let mvd = crate::cabac::b_mb::read_mvd(d, ctxs, pic, mb_x, mb_y, r0, c0, 0)?;
            crate::cabac::b_mb::write_mvd_abs(pic, mb_x, mb_y, r0, c0, ph, pw, 0, mvd);
            mvd_l0[p] = Some(mvd);
        }
    }
    for p in 0..2 {
        if uses_l1(dirs[p]) {
            let (r0, c0, ph, pw) = rects[p];
            let mvd = crate::cabac::b_mb::read_mvd(d, ctxs, pic, mb_x, mb_y, r0, c0, 1)?;
            crate::cabac::b_mb::write_mvd_abs(pic, mb_x, mb_y, r0, c0, ph, pw, 1, mvd);
            mvd_l1[p] = Some(mvd);
        }
    }
    for p in 0..2 {
        let (r0, c0, ph, pw) = rects[p];
        crate::b_mb_444::compensate_444_partition(
            sh, bctx, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, ph, pw, dirs[p], ref_l0[p],
            ref_l1[p], mvd_l0[p], mvd_l1[p],
        )?;
    }
    Ok(())
}

/// B_8x8 under ChromaArrayType = 3. Mirrors the 4:2:0 CABAC B_8x8
/// syntax walk (§7.3.5.1 — sub_mb_type for each sub, then ref_idx_l0,
/// ref_idx_l1, mvd_l0, mvd_l1 streamed per sub-MB) and then runs MC
/// on all three planes via [`crate::b_mb_444::compensate_444_partition`]
/// / the direct-8x8 snapshot helper.
#[allow(clippy::too_many_arguments)]
fn decode_b_8x8_444(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sh: &SliceHeader,
    sps: &Sps,
    bctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
) -> Result<()> {
    const SUB_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 2), (2, 0), (2, 2)];
    let uses_l0 = |dir: PredDir| matches!(dir, PredDir::L0 | PredDir::BiPred);
    let uses_l1 = |dir: PredDir| matches!(dir, PredDir::L1 | PredDir::BiPred);
    let num_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let num_l1 = sh.num_ref_idx_l1_active_minus1 + 1;
    let mut subs = [BSubPartition::Direct8x8; 4];
    for s in 0..4 {
        let raw = {
            let slice = &mut ctxs[CTX_IDX_SUB_MB_TYPE_B..CTX_IDX_SUB_MB_TYPE_B + 4];
            binarize::decode_sub_mb_type_b(d, slice)?
        };
        subs[s] = decode_b_sub_mb_type(raw).ok_or_else(|| {
            Error::invalid(format!("h264 cabac 4:4:4 b-slice: bad sub_mb_type {raw}"))
        })?;
    }

    // ref_idx_l0 for non-direct L0/BiPred sub-MBs.
    let mut ref_l0 = [None; 4];
    let mut ref_l1 = [None; 4];
    for s in 0..4 {
        if let Some(dir) = subs[s].pred_dir() {
            if uses_l0(dir) {
                let (sr0, sc0) = SUB_OFFSETS[s];
                ref_l0[s] = Some(crate::cabac::b_mb::read_ref_idx_lx(
                    d, ctxs, pic, mb_x, mb_y, sr0, sc0, num_l0, 0,
                )?);
            }
        }
    }
    for s in 0..4 {
        if let Some(dir) = subs[s].pred_dir() {
            if uses_l1(dir) {
                let (sr0, sc0) = SUB_OFFSETS[s];
                ref_l1[s] = Some(crate::cabac::b_mb::read_ref_idx_lx(
                    d, ctxs, pic, mb_x, mb_y, sr0, sc0, num_l1, 1,
                )?);
            }
        }
    }
    // mvd_l0 then mvd_l1 across all sub-partitions; fill mvd_abs inline
    // so subsequent sub-MBs' MVD ctx sees the updated neighbours.
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
                    let mvd = crate::cabac::b_mb::read_mvd(d, ctxs, pic, mb_x, mb_y, r0, c0, 0)?;
                    crate::cabac::b_mb::write_mvd_abs(pic, mb_x, mb_y, r0, c0, sh_h, sh_w, 0, mvd);
                    Some(mvd)
                } else {
                    None
                };
                let mvd_l1 = if uses_l1(dir) {
                    let mvd = crate::cabac::b_mb::read_mvd(d, ctxs, pic, mb_x, mb_y, r0, c0, 1)?;
                    crate::cabac::b_mb::write_mvd_abs(pic, mb_x, mb_y, r0, c0, sh_h, sh_w, 1, mvd);
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

    // Compensate each sub-MB on all three planes.
    for s in 0..4 {
        let sp = subs[s];
        let (sr0, sc0) = SUB_OFFSETS[s];
        if matches!(sp, BSubPartition::Direct8x8) {
            // Direct 8×8 — derive MVs through the CAVLC compensator on a
            // 4:2:0 scratch, then snapshot + re-MC on 4:4:4 via the
            // shared helper.
            let (mvs, dirs) = snapshot_b_direct_8x8_mv_state(
                sh, sps, bctx, mb_x, mb_y, sr0, sc0, pic, ref_list0, ref_list1,
            )?;
            crate::b_mb_444::apply_mvs_444_rect_pub(
                pic, ref_list0, ref_list1, sh, bctx, mb_x, mb_y, sr0, sc0, 2, 2, &mvs, &dirs,
            );
            continue;
        }
        let dir = sp.pred_dir().expect("non-direct");
        for sub_idx in 0..sp.num_sub_partitions() {
            let (dr, dc, sh_h, sh_w) = sp.sub_rect(sub_idx);
            let r0 = sr0 + dr;
            let c0 = sc0 + dc;
            let m = mvds[s][sub_idx];
            crate::b_mb_444::compensate_444_partition(
                sh, bctx, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, sh_h, sh_w, dir,
                ref_l0[s], ref_l1[s], m.mvd_l0, m.mvd_l1,
            )?;
        }
    }
    Ok(())
}

/// 8×8 direct-mode snapshot mirroring [`snapshot_b_direct_mv_state`]
/// — drives `direct_8x8_compensate_pub` on a 4:2:0 scratch picture so
/// §8.4.1.2 resolves the sub-MB MVs, then returns a 16-slot MV grid
/// restricted to the sub-MB rect.
#[allow(clippy::too_many_arguments)]
fn snapshot_b_direct_8x8_mv_state(
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
) -> Result<(
    crate::b_mb_444::PerBlockMv,
    [Option<crate::mb_type::PredDir>; 16],
)> {
    let mut scratch = crate::b_mb_444::make_420_scratch_pub(pic);
    let refs420_0: Vec<crate::picture::Picture> = ref_list0
        .iter()
        .map(|r| crate::b_mb_444::to_420_reference_pub(r))
        .collect();
    let refs420_1: Vec<crate::picture::Picture> = ref_list1
        .iter()
        .map(|r| crate::b_mb_444::to_420_reference_pub(r))
        .collect();
    let r0: Vec<&crate::picture::Picture> = refs420_0.iter().collect();
    let r1: Vec<&crate::picture::Picture> = refs420_1.iter().collect();
    crate::b_mb::direct_8x8_compensate_pub(
        sh,
        sps,
        ctx,
        mb_x,
        mb_y,
        sr0,
        sc0,
        &mut scratch,
        &r0,
        &r1,
    )?;
    // Merge scratch MB info back — the direct path only updates the
    // sub-MB rect (rows sr0..sr0+2, cols sc0..sc0+2).
    {
        let src = scratch.mb_info_at(mb_x, mb_y).clone();
        let dst = pic.mb_info_mut(mb_x, mb_y);
        for rr in sr0..sr0 + 2 {
            for cc in sc0..sc0 + 2 {
                let idx = rr * 4 + cc;
                dst.ref_idx_l0[idx] = src.ref_idx_l0[idx];
                dst.ref_idx_l1[idx] = src.ref_idx_l1[idx];
                dst.mv_l0[idx] = src.mv_l0[idx];
                dst.mv_l1[idx] = src.mv_l1[idx];
            }
        }
    }
    let info = pic.mb_info_at(mb_x, mb_y);
    let mut mvs = crate::b_mb_444::PerBlockMv::default();
    let mut dirs = [None; 16];
    for i in 0..16 {
        let ri0 = info.ref_idx_l0[i];
        let ri1 = info.ref_idx_l1[i];
        mvs.ref_l0[i] = ri0;
        mvs.ref_l1[i] = ri1;
        mvs.mv_l0[i] = info.mv_l0[i];
        mvs.mv_l1[i] = info.mv_l1[i];
        dirs[i] = match (ri0 >= 0, ri1 >= 0) {
            (true, true) => Some(PredDir::BiPred),
            (true, false) => Some(PredDir::L0),
            (false, true) => Some(PredDir::L1),
            (false, false) => None,
        };
    }
    Ok((mvs, dirs))
}

/// Resolve direct-mode MVs for a whole MB (§8.4.1.2) and materialise
/// the per-4×4 MV / ref / dir grid ready to hand to
/// [`crate::b_mb_444::apply_mvs_444`]. Uses the shared 4:2:0 scratch +
/// `direct_16x16_compensate_pub` pattern from the CAVLC 4:4:4 path.
#[allow(clippy::too_many_arguments)]
fn snapshot_b_direct_mv_state(
    sh: &SliceHeader,
    sps: &Sps,
    ctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
) -> Result<(
    crate::b_mb_444::PerBlockMv,
    [Option<crate::mb_type::PredDir>; 16],
)> {
    use crate::mb_type::PredDir;
    // Clone to 4:2:0 scratch so the CAVLC direct compensator can safely
    // write 4:2:0 chroma samples; we'll discard those and re-run MC on
    // 4:4:4 with the resulting MV grid.
    let mut scratch = crate::b_mb_444::make_420_scratch_pub(pic);
    let refs420_0: Vec<crate::picture::Picture> = ref_list0
        .iter()
        .map(|r| crate::b_mb_444::to_420_reference_pub(r))
        .collect();
    let refs420_1: Vec<crate::picture::Picture> = ref_list1
        .iter()
        .map(|r| crate::b_mb_444::to_420_reference_pub(r))
        .collect();
    let r0: Vec<&crate::picture::Picture> = refs420_0.iter().collect();
    let r1: Vec<&crate::picture::Picture> = refs420_1.iter().collect();
    crate::b_mb::direct_16x16_compensate_pub(sh, sps, ctx, mb_x, mb_y, &mut scratch, &r0, &r1)?;
    {
        let src = scratch.mb_info_at(mb_x, mb_y).clone();
        let dst = pic.mb_info_mut(mb_x, mb_y);
        *dst = src;
    }
    let info = pic.mb_info_at(mb_x, mb_y);
    let mut mvs = crate::b_mb_444::PerBlockMv::default();
    let mut dirs = [None; 16];
    for i in 0..16 {
        let ri0 = info.ref_idx_l0[i];
        let ri1 = info.ref_idx_l1[i];
        mvs.ref_l0[i] = ri0;
        mvs.ref_l1[i] = ri1;
        mvs.mv_l0[i] = info.mv_l0[i];
        mvs.mv_l1[i] = info.mv_l1[i];
        dirs[i] = match (ri0 >= 0, ri1 >= 0) {
            (true, true) => Some(PredDir::BiPred),
            (true, false) => Some(PredDir::L0),
            (false, true) => Some(PredDir::L1),
            (false, false) => None,
        };
    }
    Ok((mvs, dirs))
}

// ---------------------------------------------------------------------------
// P inter — 16x16 / 16x8 / 8x16 partitions.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_p_inter_444(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sh: &SliceHeader,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    prev_qp: &mut i32,
    partition: PPartition,
) -> Result<()> {
    let num_ref_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let n_parts = partition.num_partitions();
    let mut ref_idxs = [0i8; 4];
    for p in 0..n_parts {
        ref_idxs[p] = if num_ref_l0 > 1 {
            let inc = cabac_p::ref_idx_ctx_idx_inc(pic, mb_x, mb_y, partition, p)?;
            let slice = &mut ctxs[CTX_IDX_REF_IDX_L0..CTX_IDX_REF_IDX_L0 + 6];
            let raw = binarize::decode_ref_idx_lx(d, slice, inc)?;
            if raw >= num_ref_l0 {
                return Err(Error::invalid(format!(
                    "h264 cabac 4:4:4 p-slice: ref_idx_l0 {raw} out of range (num_ref={num_ref_l0})"
                )));
            }
            raw as i8
        } else {
            0
        };
    }

    // Read MVDs + do MV prediction.
    let mut mvs = [(0i16, 0i16); 4];
    for p in 0..n_parts {
        let (r0, c0, ph, pw) = partition.partition_rect(p);
        let inc_x = cabac_p::mvd_ctx_idx_inc(pic, mb_x, mb_y, r0, c0, true);
        let inc_y = cabac_p::mvd_ctx_idx_inc(pic, mb_x, mb_y, r0, c0, false);
        let mvd_x = {
            let slice = &mut ctxs[CTX_IDX_MVD_L0_X..CTX_IDX_MVD_L0_X + 7];
            binarize::decode_mvd_component(d, slice, inc_x)?
        };
        let mvd_y = {
            let slice = &mut ctxs[CTX_IDX_MVD_L0_Y..CTX_IDX_MVD_L0_Y + 7];
            binarize::decode_mvd_component(d, slice, inc_y)?
        };
        let pmv = predict_mv_l0(pic, mb_x, mb_y, r0, c0, ph, pw, ref_idxs[p]);
        let mv = (
            pmv.0.wrapping_add(mvd_x as i16),
            pmv.1.wrapping_add(mvd_y as i16),
        );
        mvs[p] = mv;
        cabac_p::fill_partition_mv(pic, mb_x, mb_y, r0, c0, ph, pw, mv, ref_idxs[p]);
        cabac_p::fill_partition_mvd_abs(
            pic,
            mb_x,
            mb_y,
            r0,
            c0,
            ph,
            pw,
            mvd_x.unsigned_abs(),
            mvd_y.unsigned_abs(),
        );
    }

    // MC across all three planes via the shared 4:4:4 helper.
    for p in 0..n_parts {
        let (r0, c0, ph, pw) = partition.partition_rect(p);
        let reference = lookup_ref_444(ref_list0, ref_idxs[p])?;
        let lw = l0_luma_weight(sh, ref_idxs[p]);
        let mv = mvs[p];
        mc_partition_all_planes(
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
    }

    // §9.3.3.1.1.4 — coded_block_pattern (luma-only under 4:4:4).
    let cbp_luma = {
        let slice =
            &mut ctxs[CTX_IDX_CODED_BLOCK_PATTERN_LUMA..CTX_IDX_CODED_BLOCK_PATTERN_LUMA + 12];
        let cbp = binarize::decode_coded_block_pattern(
            d,
            slice,
            0, // skip chroma bins — 4:4:4 reuses the 4-bit luma CBP
            |i, decoded| cbp_luma_ctx_idx_inc(pic, mb_x, mb_y, i, decoded),
            |_bin| 0u8,
        )?;
        (cbp & 0x0F) as u8
    };

    // §7.3.5.1 — transform_size_8x8_flag gate matches CAVLC 4:4:4:
    // refuse if the encoder selected 8×8 transform on any plane (not yet
    // wired for 4:4:4 chroma).
    let transform_8x8 = if pps.transform_8x8_mode_flag && cbp_luma != 0 {
        let inc = transform_size_8x8_flag_ctx_idx_inc(pic, mb_x, mb_y);
        let slice = &mut ctxs[CTX_IDX_TRANSFORM_SIZE_8X8_FLAG..CTX_IDX_TRANSFORM_SIZE_8X8_FLAG + 3];
        binarize::decode_transform_size_8x8_flag(d, slice, inc)?
    } else {
        false
    };
    if transform_8x8 {
        return Err(Error::unsupported(
            "h264: CABAC 4:4:4 inter transform_size_8x8_flag not wired — 4×4 transform only",
        ));
    }

    let needs_qp = cbp_luma != 0;
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
    let qp_cb = chroma_qp(qp_y, pps.chroma_qp_index_offset);
    let qp_cr = chroma_qp(qp_y, pps.second_chroma_qp_index_offset);

    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        info.qp_y = qp_y;
        info.coded = true;
        info.intra = false;
        info.skipped = false;
        info.mb_type_p = Some(PMbType::Inter { partition });
        info.p_partition = Some(partition);
        info.intra4x4_pred_mode = [INTRA_DC_FAKE; 16];
        info.transform_8x8 = false;
        info.cbp_luma = cbp_luma;
        info.cbp_chroma = 0;
    }

    for (plane, qp, scale_idx) in [
        (Plane::Y, qp_y, 3usize),
        (Plane::Cb, qp_cb, 4usize),
        (Plane::Cr, qp_cr, 5usize),
    ] {
        decode_plane_inter_residual_4x4_cabac(
            d, ctxs, mb_x, mb_y, pic, plane, cbp_luma, qp, scale_idx,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_plane_inter_residual_4x4_cabac(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    plane: Plane,
    cbp_luma: u8,
    qp: i32,
    scale_idx: usize,
) -> Result<()> {
    let stride = pic.luma_stride();
    let lo_mb = pic.luma_off(mb_x, mb_y);
    let rplane = match plane {
        Plane::Y => ResidualPlane::Luma,
        Plane::Cb => ResidualPlane::Cb444,
        Plane::Cr => ResidualPlane::Cr444,
    };
    for blk in 0..16usize {
        let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
        let eight_idx = (br_row / 2) * 2 + (br_col / 2);
        if (cbp_luma >> eight_idx) & 1 == 0 {
            continue;
        }
        let neighbours = cbf_neighbours_plane(pic, mb_x, mb_y, plane, br_row, br_col);
        let coeffs = decode_residual_block_in_place_plane(
            d,
            ctxs,
            BlockCat::Luma4x4,
            &neighbours,
            16,
            false,
            rplane,
        )?;
        let mut residual = coeffs;
        let total_coeff = coeffs.iter().filter(|&&v| v != 0).count() as u32;
        let scale = *pic.scaling_lists.matrix_4x4(scale_idx);
        dequantize_4x4_scaled(&mut residual, qp, &scale);
        idct_4x4(&mut residual);
        {
            let (buf, _) = mb444::plane_samples_mut(pic, plane);
            let lo = lo_mb + br_row * 4 * stride + br_col * 4;
            for r in 0..4 {
                for c in 0..4 {
                    let base = buf[lo + r * stride + c] as i32;
                    buf[lo + r * stride + c] = (base + residual[r * 4 + c]).clamp(0, 255) as u8;
                }
            }
        }
        let info = pic.mb_info_mut(mb_x, mb_y);
        mb444::plane_nc_at_mut(info, plane)[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Neighbour helpers: plane-aware CBF neighbour read.
// ---------------------------------------------------------------------------

fn cbf_neighbours_plane(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    plane: Plane,
    br_row: usize,
    br_col: usize,
) -> CbfNeighbours {
    let info_here = pic.mb_info_at(mb_x, mb_y);
    let here = mb444::plane_nc_at(info_here, plane);
    let left = if br_col > 0 {
        Some(here[br_row * 4 + br_col - 1] != 0)
    } else if mb_x > 0 {
        let info = pic.mb_info_at(mb_x - 1, mb_y);
        if info.coded {
            Some(mb444::plane_nc_at(info, plane)[br_row * 4 + 3] != 0)
        } else {
            None
        }
    } else {
        None
    };
    let above = if br_row > 0 {
        Some(here[(br_row - 1) * 4 + br_col] != 0)
    } else if mb_y > 0 {
        let info = pic.mb_info_at(mb_x, mb_y - 1);
        if info.coded {
            Some(mb444::plane_nc_at(info, plane)[12 + br_col] != 0)
        } else {
            None
        }
    } else {
        None
    };
    CbfNeighbours { left, above }
}

// ---------------------------------------------------------------------------
// B-slice ctxIdxInc derivations (shared with the 4:2:0 path).
// ---------------------------------------------------------------------------

fn mb_skip_flag_b_ctx_idx_inc(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
    let a = mb_x > 0 && !pic.mb_info_at(mb_x - 1, mb_y).skipped;
    let b = mb_y > 0 && !pic.mb_info_at(mb_x, mb_y - 1).skipped;
    (a as u8) + (b as u8)
}

fn mb_type_b_ctx_idx_inc(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
    let check = |info: &crate::picture::MbInfo| -> bool {
        if !info.coded {
            return false;
        }
        if info.skipped {
            return false;
        }
        true
    };
    let a = mb_x > 0 && check(pic.mb_info_at(mb_x - 1, mb_y));
    let b = mb_y > 0 && check(pic.mb_info_at(mb_x, mb_y - 1));
    (a as u8) + (b as u8)
}
