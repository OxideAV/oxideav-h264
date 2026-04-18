//! CABAC P- and B-slice macroblock decode for 4:4:4 (ChromaArrayType == 3).
//!
//! This first pass wires the entry points for the CABAC-in-4:4:4 P and B
//! paths. Intra-in-P / intra-in-B macroblocks route through the 4:4:4 I
//! path in [`crate::cabac::mb_444`]; inter macroblocks are parsed through
//! the shared 4:2:0 CABAC helpers (MV prediction, mvd / ref_idx / cbp
//! entropy) and then re-dispatched onto the 4:4:4 per-plane MC +
//! residual pipelines in [`crate::p_mb_444`] / [`crate::b_mb_444`] where
//! viable.
//!
//! The first-pass decode returns `Error::Unsupported` for P/B inter MBs —
//! the stretch-goal implementation lives in a follow-up. Intra-in-P and
//! intra-in-B both reach the 4:4:4 I macroblock helper cleanly, which in
//! practice covers solid-colour fixtures where every post-IDR coded
//! macroblock is either skipped or re-intra-coded.

use oxideav_core::{Error, Result};

use crate::b_mb::BSliceCtx;
use crate::cabac::binarize;
use crate::cabac::context::CabacContext;
use crate::cabac::engine::CabacDecoder;
use crate::cabac::mb_444::decode_intra_in_p_cabac_444;
use crate::cabac::tables::{
    CTX_IDX_MB_SKIP_FLAG_B, CTX_IDX_MB_SKIP_FLAG_P, CTX_IDX_MB_TYPE_B, CTX_IDX_MB_TYPE_I,
    CTX_IDX_MB_TYPE_INTRA_IN_P, CTX_IDX_MB_TYPE_P,
};
use crate::mb_type::{decode_b_slice_mb_type, decode_i_slice_mb_type, BMbType, IMbType};
use crate::picture::Picture;
use crate::pps::Pps;
use crate::slice::SliceHeader;
use crate::sps::Sps;

/// Decode one P-slice macroblock under ChromaArrayType = 3 through
/// CABAC. See module docs for coverage.
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
    let skip_inc = mb_skip_flag_p_ctx_idx_inc(pic, mb_x, mb_y);
    let skipped = {
        let slice = &mut ctxs[CTX_IDX_MB_SKIP_FLAG_P..CTX_IDX_MB_SKIP_FLAG_P + 3];
        binarize::decode_mb_skip_flag_p(d, slice, skip_inc)?
    };
    if skipped {
        crate::p_mb_444::decode_p_skip_mb_444(sh, mb_x, mb_y, pic, ref_list0, *prev_qp)?;
        return Ok(true);
    }

    // §9.3.3.1.1.3 — mb_type for P/SP. The 7-slot window at
    // [CTX_IDX_MB_TYPE_P .. +7] covers the inter prefix bank and the
    // intra-in-P suffix bank overlap at ctxIdx 17.
    let mb_type_raw = {
        let slice = &mut ctxs[CTX_IDX_MB_TYPE_P..CTX_IDX_MB_TYPE_P + 7];
        binarize::decode_mb_type_p(d, slice, CTX_IDX_MB_TYPE_INTRA_IN_P - CTX_IDX_MB_TYPE_P)?
    };

    match mb_type_raw {
        0..=4 => Err(Error::unsupported(
            "h264: CABAC 4:4:4 P-slice inter macroblocks not yet wired — \
             intra-in-P and P_Skip only (solid-colour clips)",
        )),
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
/// CABAC. See module docs for coverage.
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
        decode_intra_in_p_cabac_444(d, ctxs, sh, sps, pps, mb_x, mb_y, pic, prev_qp, IMbType::IPcm)?;
        return Ok(false);
    }
    let bmb = decode_b_slice_mb_type(mb_type_raw).ok_or_else(|| {
        Error::invalid(format!("h264 cabac 4:4:4 b-slice: bad mb_type {mb_type_raw}"))
    })?;
    match bmb {
        BMbType::IntraInB(imb) => {
            decode_intra_in_p_cabac_444(d, ctxs, sh, sps, pps, mb_x, mb_y, pic, prev_qp, imb)?;
            Ok(false)
        }
        BMbType::Inter { .. } => Err(Error::unsupported(
            "h264: CABAC 4:4:4 B-slice inter macroblocks not yet wired — \
             intra-in-B and B_Skip only (solid-colour clips)",
        )),
    }
}

// ---------------------------------------------------------------------------
// ctxIdxInc derivations shared with the 4:2:0 path.
// ---------------------------------------------------------------------------

fn mb_skip_flag_p_ctx_idx_inc(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
    let a = mb_x > 0 && !pic.mb_info_at(mb_x - 1, mb_y).skipped;
    let b = mb_y > 0 && !pic.mb_info_at(mb_x, mb_y - 1).skipped;
    (a as u8) + (b as u8)
}

fn mb_skip_flag_b_ctx_idx_inc(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
    let a = mb_x > 0 && !pic.mb_info_at(mb_x - 1, mb_y).skipped;
    let b = mb_y > 0 && !pic.mb_info_at(mb_x, mb_y - 1).skipped;
    (a as u8) + (b as u8)
}

fn mb_type_b_ctx_idx_inc(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
    // §9.3.3.1.1.3 — condTermFlagN = 0 if mbN is unavailable or mbN is
    // B_Skip / B_Direct_16x16, else 1.
    let check = |info: &crate::picture::MbInfo| -> bool {
        if !info.coded {
            return false;
        }
        if info.skipped {
            return false;
        }
        // Non-skip, non-direct inter / intra neighbour contributes 1.
        true
    };
    let a = mb_x > 0 && check(pic.mb_info_at(mb_x - 1, mb_y));
    let b = mb_y > 0 && check(pic.mb_info_at(mb_x, mb_y - 1));
    (a as u8) + (b as u8)
}
