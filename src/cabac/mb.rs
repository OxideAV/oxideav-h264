//! CABAC-driven I-slice macroblock decode — ITU-T H.264 §7.3.5 + §9.3.
//!
//! Mirrors [`crate::mb::decode_i_slice_data`] / `decode_one_mb` (the CAVLC
//! path) but reads every syntax element through the CABAC engine. The pixel
//! reconstruction stages (intra prediction, dequantisation, inverse
//! transforms) are shared verbatim with the CAVLC path — only the entropy
//! layer differs.
//!
//! The per-MB `ctxIdxInc` derivations here are the simple, spec-faithful
//! forms from §9.3.3.1.1:
//!
//! * `mb_type` bin 0: `condTermFlagA + condTermFlagB`, with
//!   `condTermFlagN = 0` for an unavailable neighbour and for any neighbour
//!   that is itself `I_NxN`.
//! * `mb_qp_delta` bin 0: `condTermFlagPrev = (prev_qp_delta != 0)`.
//! * `intra_chroma_pred_mode` bin 0:
//!   `condTermFlagN = (mbN intra && mbN.intra_chroma_pred_mode != 0)`.
//! * `coded_block_pattern` (luma sub-block i):
//!   `ctxIdxInc = condTermFlagA(i) + 2·condTermFlagB(i)`
//!   with `condTermFlagN(i) = (!mbN coded || cbp_luma_bit_N == 0)`.
//! * `coded_block_pattern` (chroma):
//!   `ctxIdxInc = condTermFlagA + 2·condTermFlagB`,
//!   `condTermFlagN = (mbN.cbp_chroma != 0)`.
//! * `coded_block_flag`: `condTermFlagA + 2·condTermFlagB` over the
//!   4×4 sub-block cbf values.

use oxideav_core::{Error, Result};

use crate::cabac::binarize;
use crate::cabac::context::CabacContext;
use crate::cabac::engine::CabacDecoder;
use crate::cabac::residual::{
    decode_residual_block_cabac, decode_residual_block_cabac_8x8,
    decode_residual_block_cabac_chroma_dc_422, BlockCat, CbfNeighbours,
};
use crate::cabac::tables::{
    CTX_IDX_ABSLVL_CB_LUMA16X16AC, CTX_IDX_ABSLVL_CB_LUMA16X16DC, CTX_IDX_ABSLVL_CB_LUMA4X4,
    CTX_IDX_ABSLVL_CB_LUMA8X8, CTX_IDX_ABSLVL_CR_LUMA16X16AC, CTX_IDX_ABSLVL_CR_LUMA16X16DC,
    CTX_IDX_ABSLVL_CR_LUMA4X4, CTX_IDX_ABSLVL_CR_LUMA8X8, CTX_IDX_CBF_CB_LUMA16X16AC,
    CTX_IDX_CBF_CB_LUMA16X16DC, CTX_IDX_CBF_CB_LUMA4X4, CTX_IDX_CBF_CB_LUMA8X8,
    CTX_IDX_CBF_CR_LUMA16X16AC, CTX_IDX_CBF_CR_LUMA16X16DC, CTX_IDX_CBF_CR_LUMA4X4,
    CTX_IDX_CBF_CR_LUMA8X8, CTX_IDX_CODED_BLOCK_FLAG, CTX_IDX_CODED_BLOCK_FLAG_LUMA8X8,
    CTX_IDX_CODED_BLOCK_PATTERN_LUMA, CTX_IDX_COEFF_ABS_LEVEL_MINUS1,
    CTX_IDX_COEFF_ABS_LEVEL_MINUS1_LUMA8X8, CTX_IDX_INTRA_CHROMA_PRED_MODE,
    CTX_IDX_LAST_CB_LUMA16X16AC, CTX_IDX_LAST_CB_LUMA16X16DC, CTX_IDX_LAST_CB_LUMA4X4,
    CTX_IDX_LAST_CB_LUMA8X8, CTX_IDX_LAST_CR_LUMA16X16AC, CTX_IDX_LAST_CR_LUMA16X16DC,
    CTX_IDX_LAST_CR_LUMA4X4, CTX_IDX_LAST_CR_LUMA8X8, CTX_IDX_LAST_SIGNIFICANT_COEFF_FLAG,
    CTX_IDX_LAST_SIGNIFICANT_COEFF_FLAG_LUMA8X8, CTX_IDX_MB_QP_DELTA, CTX_IDX_MB_TYPE_I,
    CTX_IDX_PREV_INTRA4X4_PRED_MODE_FLAG, CTX_IDX_REM_INTRA4X4_PRED_MODE,
    CTX_IDX_SIGNIFICANT_COEFF_FLAG,
    CTX_IDX_SIGNIFICANT_COEFF_FLAG_LUMA8X8, CTX_IDX_SIG_CB_LUMA16X16AC, CTX_IDX_SIG_CB_LUMA16X16DC,
    CTX_IDX_SIG_CB_LUMA4X4, CTX_IDX_SIG_CB_LUMA8X8, CTX_IDX_SIG_CR_LUMA16X16AC,
    CTX_IDX_SIG_CR_LUMA16X16DC, CTX_IDX_SIG_CR_LUMA4X4, CTX_IDX_SIG_CR_LUMA8X8,
    CTX_IDX_TRANSFORM_SIZE_8X8_FLAG,
};
use crate::cavlc::ZIGZAG_4X4;
use crate::intra_pred::{
    predict_intra_16x16, predict_intra_4x4, predict_intra_8x8, predict_intra_chroma,
    predict_intra_chroma_8x16, Intra16x16Mode, Intra16x16Neighbours, Intra4x4Mode,
    Intra4x4Neighbours, Intra8x8Mode, Intra8x8Neighbours, IntraChroma8x16Neighbours,
    IntraChromaMode, IntraChromaNeighbours,
};
use crate::mb::LUMA_BLOCK_RASTER;
use crate::mb_type::{decode_i_slice_mb_type, IMbType};
use crate::picture::{MbInfo, Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::SliceHeader;
use crate::sps::Sps;
use crate::transform::{
    chroma_qp, dequantize_4x4_scaled, dequantize_8x8_scaled, idct_4x4, idct_8x8,
    inv_hadamard_2x2_chroma_dc_scaled, inv_hadamard_2x4_chroma_dc_scaled,
    inv_hadamard_4x4_dc_scaled, ZIGZAG_8X8,
};

// ---------------------------------------------------------------------------
// ctxIdxInc derivation helpers.
// ---------------------------------------------------------------------------

fn mb_type_i_ctx_idx_inc(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
    // §9.3.3.1.1.3: condTermFlagN = 0 if mbN is unavailable or mbN is I_NxN.
    let a = if mb_x > 0 {
        let m = pic.mb_info_at(mb_x - 1, mb_y);
        m.coded && !matches!(m.mb_type_i, Some(IMbType::INxN))
    } else {
        false
    };
    let b = if mb_y > 0 {
        let m = pic.mb_info_at(mb_x, mb_y - 1);
        m.coded && !matches!(m.mb_type_i, Some(IMbType::INxN))
    } else {
        false
    };
    (a as u8) + (b as u8)
}

fn intra_chroma_pred_mode_ctx_idx_inc(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
    // §9.3.3.1.1.8.
    let check = |m: &MbInfo| m.coded && m.intra && m.intra_chroma_pred_mode != 0;
    let a = mb_x > 0 && check(pic.mb_info_at(mb_x - 1, mb_y));
    let b = mb_y > 0 && check(pic.mb_info_at(mb_x, mb_y - 1));
    (a as u8) + (b as u8)
}

/// `coded_block_pattern` luma ctxIdxInc for each 8×8 sub-block `i ∈ 0..4`
/// per §9.3.3.1.1.4 / Table 9-39.
///
/// For the i-th luma CBP bit we have a spatial A (left) neighbour and a
/// B (above) neighbour. Each neighbour is either:
///   * another 8×8 sub-block *inside* the current MB that is decoded
///     earlier in raster order — its CBP bit was just set on the four
///     already-decoded bins, so we consult a running `decoded_bits` mask.
///   * the matching 8×8 in an adjacent MB (written to `MbInfo::cbp_luma`).
///
/// `condTermFlagN = 1` when the neighbour is unavailable OR its CBP bit
/// is 0; `ctxIdxInc = condTermFlagA + 2·condTermFlagB`.
pub(crate) fn cbp_luma_ctx_idx_inc(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    i: usize,
    decoded_bits: u8,
) -> u8 {
    let (xi, yi) = sub_8x8_xy(i);
    // §9.3.3.1.1.4 / FFmpeg h264_mvpred.h: when a neighbour MB is
    // unavailable, the CBP luma nibble is treated as all-ones (0x0F for
    // inter, 0x7CF-masked for intra). That flips condTermFlag to 0 — NOT
    // the "all zero" default a naive read would assume.
    let a_bit = if xi > 0 {
        (decoded_bits >> (i - 1)) & 1 != 0
    } else if mb_x > 0 {
        let m = pic.mb_info_at(mb_x - 1, mb_y);
        if m.coded {
            let nbr_idx = i + 1;
            (m.cbp_luma >> nbr_idx) & 1 != 0
        } else {
            true
        }
    } else {
        true
    };
    let b_bit = if yi > 0 {
        (decoded_bits >> (i - 2)) & 1 != 0
    } else if mb_y > 0 {
        let m = pic.mb_info_at(mb_x, mb_y - 1);
        if m.coded {
            let nbr_idx = i + 2;
            (m.cbp_luma >> nbr_idx) & 1 != 0
        } else {
            true
        }
    } else {
        true
    };
    let a_cond = if a_bit { 0 } else { 1 };
    let b_cond = if b_bit { 0 } else { 1 };
    a_cond + 2 * b_cond
}

fn sub_8x8_xy(i: usize) -> (usize, usize) {
    match i {
        0 => (0, 0),
        1 => (1, 0),
        2 => (0, 1),
        3 => (1, 1),
        _ => unreachable!(),
    }
}

pub(crate) fn cbp_chroma_ctx_idx_inc_any(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
    // §9.3.3.1.1.4 / FFmpeg decode_cabac_mb_cbp_chroma bin 0:
    // ctxIdxInc increments by 1 iff left-MB cbp_chroma != 0, by 2 iff
    // top-MB cbp_chroma != 0. Unavailable neighbours default to
    // cbp_chroma = 0 (left_cbp/top_cbp bits 4..5 are cleared in both the
    // intra 0x7CF and inter 0x00F fall-backs).
    let a = mb_x > 0 && {
        let m = pic.mb_info_at(mb_x - 1, mb_y);
        m.coded && m.cbp_chroma != 0
    };
    let b = mb_y > 0 && {
        let m = pic.mb_info_at(mb_x, mb_y - 1);
        m.coded && m.cbp_chroma != 0
    };
    (a as u8) + 2 * (b as u8)
}

pub(crate) fn cbp_chroma_ctx_idx_inc_ac(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
    // §9.3.3.1.1.4 / FFmpeg decode_cabac_mb_cbp_chroma bin 1: ctxIdxInc
    // increments by 1/2 iff neighbour's cbp_chroma == 2 (DC+AC).
    // Unavailable neighbours contribute 0 (left_cbp/top_cbp default has
    // bits 4..5 zeroed in both intra and inter fall-backs). The +4 base
    // offset to reach ctxIdx 81..85 is applied by `decode_coded_block_pattern`.
    let a = mb_x > 0 && {
        let m = pic.mb_info_at(mb_x - 1, mb_y);
        m.coded && m.cbp_chroma == 2
    };
    let b = mb_y > 0 && {
        let m = pic.mb_info_at(mb_x, mb_y - 1);
        m.coded && m.cbp_chroma == 2
    };
    (a as u8) + 2 * (b as u8)
}

/// Select the 43-entry context window for a residual block per
/// §9.3.3.1.1.9 (Table 9-42). The layout returned into `out` is the flat
/// view `decode_residual_block_cabac` expects:
///
/// * `out[0]`    — coded_block_flag (with ctxIdxInc already applied).
/// * `out[1..=16]`  — significant_coeff_flag contexts.
/// * `out[17..=32]` — last_significant_coeff_flag contexts.
/// * `out[33..=42]` — coeff_abs_level_minus1 contexts.
///
/// The caller hands us ownership of the full 460-entry ctx vector; we
/// materialise a Vec<CabacContext> slice just for the block (shared-state
/// per-block is expected for a single pass — ctxs aren't shared between
/// blocks in this shim). For v1 we trade precise per-ctxBlockCat context
/// tracking for a simpler plumb: each residual block gets a fresh copy of
/// the appropriate ctx window, and writes back on return. This is
/// conservative vs. the spec's shared per-slice state but yields correct
/// decode for the reduced subset of streams we target (no residuals for
/// the I_16x16/cbp_luma=0 fixture).
/// Colour plane for the residual stream. Under 4:2:0 every residual
/// is `Luma` (chroma streams use the dedicated `BlockCat::Chroma*`
/// variants rather than a plane selector). Under 4:4:4 (ChromaArrayType
/// = 3), each of the three planes decodes luma-shaped residuals through
/// its own bank (§9.3.3.1.1.9 Table 9-42 extension cats 6..=13):
///
/// * `Plane::Luma` — cat 0/1/2/5 (Y)
/// * `Plane::Cb444` — cat 6/7/8/9 (Cb)
/// * `Plane::Cr444` — cat 10/11/12/13 (Cr)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResidualPlane {
    Luma,
    Cb444,
    Cr444,
}

/// Plan describing the ctxIdx backing each bin a residual decoder may read.
///
/// The 43-slot window (`coded_block_flag`, 16× `significant_coeff_flag`,
/// 16× `last_significant_coeff_flag`, 10× `coeff_abs_level_minus1`) maps
/// into specific indices of the slice-wide context array. After decode,
/// callers use [`ResidualCtxPlan::scatter`] to propagate probability-state
/// updates from the local window back to the slice-wide store so CABAC's
/// adaptive contexts persist across blocks (§9.3).
pub(crate) struct ResidualCtxPlan {
    pub indices: [usize; 43],
}

impl ResidualCtxPlan {
    /// Build the 43-entry ctxIdx plan for the given ctxBlockCat and the
    /// neighbour-derived `coded_block_flag` increment. `intra` selects the
    /// unavailable-neighbour default per §9.3.3.1.1.9 (condTermFlagN = 1
    /// for intra, 0 for inter).
    ///
    /// This is the 4:2:0 entry point — it always selects the luma-side
    /// context banks (cats 0/1/2/5 for Luma, cats 3/4 for Chroma). Under
    /// 4:4:4 the chroma planes need their own banks; use
    /// [`ResidualCtxPlan::new_for_plane`] instead.
    pub fn new(cat: BlockCat, neighbours: &CbfNeighbours, intra: bool) -> Self {
        Self::new_for_plane(cat, neighbours, intra, ResidualPlane::Luma)
    }

    /// Plane-aware variant of [`Self::new`]. Under `ResidualPlane::Luma`
    /// behaves identically to `new`; under `Cb444` / `Cr444` selects the
    /// §9.3.3.1.1.9 Table 9-42 extension banks at spec ctxIdx 460+.
    pub fn new_for_plane(
        cat: BlockCat,
        neighbours: &CbfNeighbours,
        intra: bool,
        plane: ResidualPlane,
    ) -> Self {
        let cbf_inc = coded_block_flag_inc(neighbours, intra) as usize;
        let (cbf_base, sig_base, last_base, lvl_base) = match plane {
            ResidualPlane::Luma => {
                let ctx_block_cat = cat.ctx_block_cat() as usize;
                (
                    CTX_IDX_CODED_BLOCK_FLAG + ctx_block_cat * 4,
                    CTX_IDX_SIGNIFICANT_COEFF_FLAG + SIG_BASE_OFFSETS[ctx_block_cat],
                    CTX_IDX_LAST_SIGNIFICANT_COEFF_FLAG + SIG_BASE_OFFSETS[ctx_block_cat],
                    CTX_IDX_COEFF_ABS_LEVEL_MINUS1 + LVL_BASE_OFFSETS[ctx_block_cat],
                )
            }
            ResidualPlane::Cb444 => match cat {
                BlockCat::Luma16x16Dc => (
                    CTX_IDX_CBF_CB_LUMA16X16DC,
                    CTX_IDX_SIG_CB_LUMA16X16DC,
                    CTX_IDX_LAST_CB_LUMA16X16DC,
                    CTX_IDX_ABSLVL_CB_LUMA16X16DC,
                ),
                BlockCat::Luma16x16Ac => (
                    CTX_IDX_CBF_CB_LUMA16X16AC,
                    CTX_IDX_SIG_CB_LUMA16X16AC,
                    CTX_IDX_LAST_CB_LUMA16X16AC,
                    CTX_IDX_ABSLVL_CB_LUMA16X16AC,
                ),
                BlockCat::Luma4x4 => (
                    CTX_IDX_CBF_CB_LUMA4X4,
                    CTX_IDX_SIG_CB_LUMA4X4,
                    CTX_IDX_LAST_CB_LUMA4X4,
                    CTX_IDX_ABSLVL_CB_LUMA4X4,
                ),
                BlockCat::Luma8x8 => (
                    CTX_IDX_CBF_CB_LUMA8X8,
                    CTX_IDX_SIG_CB_LUMA8X8,
                    CTX_IDX_LAST_CB_LUMA8X8,
                    CTX_IDX_ABSLVL_CB_LUMA8X8,
                ),
                // §6.4.1 under 4:4:4 there's no Chroma DC/AC bank — the
                // Cb path always uses luma-shaped cats. Fall back to Cb
                // 4×4 to keep the struct total; callers never hit this.
                _ => (
                    CTX_IDX_CBF_CB_LUMA4X4,
                    CTX_IDX_SIG_CB_LUMA4X4,
                    CTX_IDX_LAST_CB_LUMA4X4,
                    CTX_IDX_ABSLVL_CB_LUMA4X4,
                ),
            },
            ResidualPlane::Cr444 => match cat {
                BlockCat::Luma16x16Dc => (
                    CTX_IDX_CBF_CR_LUMA16X16DC,
                    CTX_IDX_SIG_CR_LUMA16X16DC,
                    CTX_IDX_LAST_CR_LUMA16X16DC,
                    CTX_IDX_ABSLVL_CR_LUMA16X16DC,
                ),
                BlockCat::Luma16x16Ac => (
                    CTX_IDX_CBF_CR_LUMA16X16AC,
                    CTX_IDX_SIG_CR_LUMA16X16AC,
                    CTX_IDX_LAST_CR_LUMA16X16AC,
                    CTX_IDX_ABSLVL_CR_LUMA16X16AC,
                ),
                BlockCat::Luma4x4 => (
                    CTX_IDX_CBF_CR_LUMA4X4,
                    CTX_IDX_SIG_CR_LUMA4X4,
                    CTX_IDX_LAST_CR_LUMA4X4,
                    CTX_IDX_ABSLVL_CR_LUMA4X4,
                ),
                BlockCat::Luma8x8 => (
                    CTX_IDX_CBF_CR_LUMA8X8,
                    CTX_IDX_SIG_CR_LUMA8X8,
                    CTX_IDX_LAST_CR_LUMA8X8,
                    CTX_IDX_ABSLVL_CR_LUMA8X8,
                ),
                _ => (
                    CTX_IDX_CBF_CR_LUMA4X4,
                    CTX_IDX_SIG_CR_LUMA4X4,
                    CTX_IDX_LAST_CR_LUMA4X4,
                    CTX_IDX_ABSLVL_CR_LUMA4X4,
                ),
            },
        };
        let mut indices = [0usize; 43];
        indices[0] = cbf_base + cbf_inc;
        for i in 0..16 {
            indices[1 + i] = sig_base + i;
            indices[17 + i] = last_base + i;
        }
        for i in 0..10 {
            indices[33 + i] = lvl_base + i;
        }
        Self { indices }
    }

    /// Materialise the 43-slot working window from the slice-wide array.
    pub fn gather(&self, ctxs: &[CabacContext]) -> Vec<CabacContext> {
        let n = ctxs.len();
        self.indices
            .iter()
            .map(|&i| ctxs[i.min(n.saturating_sub(1))])
            .collect()
    }

    /// Propagate post-decode context state back to the slice-wide array.
    pub fn scatter(&self, ctxs: &mut [CabacContext], window: &[CabacContext]) {
        let n = ctxs.len();
        for (slot, &i) in window.iter().zip(self.indices.iter()) {
            let idx = i.min(n.saturating_sub(1));
            ctxs[idx] = *slot;
        }
    }
}

/// Decode one residual block through the shared plan, updating the
/// slice-wide CABAC state in place. Wraps [`decode_residual_block_cabac`]
/// with the gather/scatter plumbing so adaptive probabilities persist.
/// `intra` selects the unavailable-neighbour default for the CBF ctxIdxInc
/// per §9.3.3.1.1.9 (1 when the current MB is intra-coded, 0 otherwise).
///
/// The output is returned in **4×4 raster order** (for 4×4-shaped blocks)
/// or **2×2 raster order** (for the 4:2:0 ChromaDc block). `decode_residual_block_cabac`
/// emits coefficients in coded scan order; §8.5.6 / §8.5.12 demand
/// zig-zag inverse before dequantise + IDCT, so this wrapper applies it
/// here — matching the CAVLC path in [`crate::cavlc::decode_residual_block`].
pub(crate) fn decode_residual_block_in_place(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    cat: BlockCat,
    neighbours: &CbfNeighbours,
    max_num_coeff: usize,
    intra: bool,
) -> Result<[i32; 16]> {
    decode_residual_block_in_place_plane(
        d,
        ctxs,
        cat,
        neighbours,
        max_num_coeff,
        intra,
        ResidualPlane::Luma,
    )
}

/// Plane-aware variant of [`decode_residual_block_in_place`]. Under
/// 4:4:4 (ChromaArrayType = 3) the Cb / Cr luma-shaped residuals run
/// against the §9.3.3.1.1.9 Table 9-42 extension banks (cats 6..=8 for
/// Cb, 10..=12 for Cr).
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_residual_block_in_place_plane(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    cat: BlockCat,
    neighbours: &CbfNeighbours,
    max_num_coeff: usize,
    intra: bool,
    plane: ResidualPlane,
) -> Result<[i32; 16]> {
    let plan = ResidualCtxPlan::new_for_plane(cat, neighbours, intra, plane);
    let mut window = plan.gather(ctxs);
    let coded = decode_residual_block_cabac(d, &mut window, cat, neighbours, max_num_coeff)?;
    plan.scatter(ctxs, &window);

    // §8.5.6 inverse scan: CABAC emits coefficients in scan order.
    // Subsequent §8.5.10 / §8.5.11 / §8.5.12 processing (inverse Hadamard,
    // dequantise, inverse 4×4/IDCT) works on the 4×4 (or 2×2) raster.
    let mut raster = [0i32; 16];
    match cat {
        BlockCat::Luma16x16Dc | BlockCat::Luma4x4 => {
            for i in 0..16 {
                raster[ZIGZAG_4X4[i]] = coded[i];
            }
        }
        BlockCat::Luma16x16Ac | BlockCat::ChromaAc | BlockCat::Chroma4x4 => {
            // Coded positions 0..=14 map into raster 1..=15; raster 0 (DC)
            // is filled from the separate DC block.
            for i in 0..15 {
                raster[ZIGZAG_4X4[i + 1]] = coded[i];
            }
        }
        BlockCat::ChromaDc => {
            // 2×2 chroma DC scan order matches raster (see §8.5.11.1).
            for i in 0..4 {
                raster[i] = coded[i];
            }
        }
        BlockCat::Luma8x8 => {
            // 8×8 blocks never flow through the 4×4-shaped helper; callers
            // must route cat = 5 through `decode_luma_8x8_residual_in_place`.
            return Err(Error::invalid(
                "h264 cabac residual: Luma8x8 routed through 4x4 helper",
            ));
        }
    }
    Ok(raster)
}

/// §9.3.3.1.1.9 — cumulative per-ctxBlockCat base into the 61-slot
/// significant/last coeff flag tables. Indexed by `BlockCat::ctx_block_cat()`.
pub(crate) const SIG_BASE_OFFSETS: [usize; 5] = [0, 15, 29, 44, 47];

/// Table 9-30 — `ctxIdxBlockCatOffset` for `coeff_abs_level_minus1`.
/// Note cat 3 (ChromaDC) consumes 9 slots, not 10, so cat 4 starts at 39.
pub(crate) const LVL_BASE_OFFSETS: [usize; 5] = [0, 10, 20, 30, 39];

/// §9.3.3.1.1.10 — ctxIdxInc for `transform_size_8x8_flag`. Neighbour
/// `condTermFlagN = 1` when the neighbour MB is coded AND used
/// `transform_size_8x8_flag = 1`; 0 otherwise (including unavailable
/// neighbours). Result = condTermFlagA + condTermFlagB, picking one of
/// contexts 399 / 400 / 401.
pub(crate) fn transform_size_8x8_flag_ctx_idx_inc(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
    let a = mb_x > 0 && {
        let m = pic.mb_info_at(mb_x - 1, mb_y);
        m.coded && m.transform_8x8
    };
    let b = mb_y > 0 && {
        let m = pic.mb_info_at(mb_x, mb_y - 1);
        m.coded && m.transform_8x8
    };
    (a as u8) + (b as u8)
}

/// Decode one Luma-8×8 residual block through the slice-wide CABAC state.
/// Gathers the 4 cbf + 15 sig + 9 last + 10 abs-level context slots from
/// `ctxs`, runs the 64-coefficient decode, scatters updated probability
/// states back, and returns the raster-order 8×8 block ready for
/// dequant + IDCT.
///
/// §9.3.3.1.1.9 — for an intra MB with unavailable neighbour, the CBF
/// condTermFlagN defaults to 1; inter MBs default to 0.
pub(crate) fn decode_luma_8x8_residual_in_place(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    neighbours: &CbfNeighbours,
    intra: bool,
) -> Result<[i32; 64]> {
    decode_luma_8x8_residual_in_place_plane(d, ctxs, neighbours, intra, ResidualPlane::Luma)
}

/// Plane-aware 8×8 residual decode. Under 4:4:4 Cb / Cr the banks are
/// cats 9 / 13 (§9.3.3.1.1.9 Table 9-42 extension, spec ctxIdx 660+).
///
/// §7.3.5.3.1 — `coded_block_flag` is present for this call *only* in the
/// 4:4:4 Cb/Cr planes (where `ChromaArrayType == 3` → cat 9 / 13). The
/// 4:2:0 / 4:2:2 luma variant (cat 5, `maxNumCoeff == 64`) skips the
/// flag because the per-block bit has already been signalled by
/// `cbp_luma`. FFmpeg's `decode_cabac_residual_nondc` enforces the same
/// gating (`cat != 5 || CHROMA444(h)`).
pub(crate) fn decode_luma_8x8_residual_in_place_plane(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    neighbours: &CbfNeighbours,
    intra: bool,
    plane: ResidualPlane,
) -> Result<[i32; 64]> {
    let cbf_inc = coded_block_flag_inc(neighbours, intra) as usize;
    let (cbf_base, sig_base, last_base, lvl_base, decode_cbf) = match plane {
        ResidualPlane::Luma => (
            CTX_IDX_CODED_BLOCK_FLAG_LUMA8X8,
            CTX_IDX_SIGNIFICANT_COEFF_FLAG_LUMA8X8,
            CTX_IDX_LAST_SIGNIFICANT_COEFF_FLAG_LUMA8X8,
            CTX_IDX_COEFF_ABS_LEVEL_MINUS1_LUMA8X8,
            false,
        ),
        ResidualPlane::Cb444 => (
            CTX_IDX_CBF_CB_LUMA8X8,
            CTX_IDX_SIG_CB_LUMA8X8,
            CTX_IDX_LAST_CB_LUMA8X8,
            CTX_IDX_ABSLVL_CB_LUMA8X8,
            true,
        ),
        ResidualPlane::Cr444 => (
            CTX_IDX_CBF_CR_LUMA8X8,
            CTX_IDX_SIG_CR_LUMA8X8,
            CTX_IDX_LAST_CR_LUMA8X8,
            CTX_IDX_ABSLVL_CR_LUMA8X8,
            true,
        ),
    };
    let cbf_ctx_idx = cbf_base + cbf_inc;

    // §9.3.3.1.1.9 last ≥ sig ≥ cbf: the three banks are disjoint, so
    // borrow them one at a time around the decode.
    let mut sig_last_window = [CabacContext::default(); 24];
    for k in 0..15 {
        sig_last_window[k] = ctxs[sig_base + k];
    }
    for k in 0..9 {
        sig_last_window[15 + k] = ctxs[last_base + k];
    }
    let mut abs_window = [CabacContext::default(); 10];
    for k in 0..10 {
        abs_window[k] = ctxs[lvl_base + k];
    }
    let mut cbf_ctx = ctxs[cbf_ctx_idx];
    let coded = decode_residual_block_cabac_8x8(
        d,
        &mut cbf_ctx,
        &mut sig_last_window,
        &mut abs_window,
        decode_cbf,
    )?;
    ctxs[cbf_ctx_idx] = cbf_ctx;
    for k in 0..15 {
        ctxs[sig_base + k] = sig_last_window[k];
    }
    for k in 0..9 {
        ctxs[last_base + k] = sig_last_window[15 + k];
    }
    for k in 0..10 {
        ctxs[lvl_base + k] = abs_window[k];
    }

    // §8.5.6 inverse scan: 8×8 zig-zag → raster.
    let mut raster = [0i32; 64];
    for i in 0..64 {
        raster[ZIGZAG_8X8[i]] = coded[i];
    }
    Ok(raster)
}

fn coded_block_flag_inc(neighbours: &CbfNeighbours, intra_default: bool) -> u8 {
    // §9.3.3.1.1.9: for an intra MB with unavailable neighbour,
    // condTermFlagN = 1 (§9.3.3.1.1.9 para "Otherwise, if any of the
    // following conditions is true, condTermFlagN is set equal to 1:
    // mbAddrN is not available and the current MB is coded in Intra
    // prediction mode"). Inter MBs default to 0.
    let a = neighbours.left.unwrap_or(intra_default) as u8;
    let b = neighbours.above.unwrap_or(intra_default) as u8;
    a + 2 * b
}

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

/// Decode a single I-slice macroblock via CABAC.
pub fn decode_i_mb_cabac(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    prev_qp: &mut i32,
) -> Result<()> {
    // --- mb_type ---
    let mb_type_inc = mb_type_i_ctx_idx_inc(pic, mb_x, mb_y);
    #[cfg(feature = "cabac-trace")]
    {
        d.trace_label = "mb_type";
    }
    let mb_type_raw = {
        // I-slice mb_type contexts live at ctxIdxOffset 3 (CTX_IDX_MB_TYPE_I),
        // 8 slots covering bin 0 neighbour-indexed + bins 2..6.
        let slice = &mut ctxs[CTX_IDX_MB_TYPE_I..CTX_IDX_MB_TYPE_I + 8];
        binarize::decode_mb_type_i(d, slice, mb_type_inc)?
    };
    let imb = decode_i_slice_mb_type(mb_type_raw)
        .ok_or_else(|| Error::invalid(format!("h264 cabac mb: bad I mb_type {mb_type_raw}")))?;
    decode_intra_mb_given_imb_cabac(d, ctxs, sh, sps, pps, mb_x, mb_y, pic, prev_qp, imb)
}

/// Decode an intra macroblock within a CABAC slice whose `mb_type` has
/// already been parsed. Used by the P-slice path for intra-in-P
/// macroblocks (`mb_type >= 5` in Table 7-13).
#[allow(clippy::too_many_arguments)]
pub fn decode_intra_in_p_cabac(
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
    decode_intra_mb_given_imb_cabac(d, ctxs, sh, sps, pps, mb_x, mb_y, pic, prev_qp, imb)
}

#[allow(clippy::too_many_arguments)]
fn decode_intra_mb_given_imb_cabac(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    _sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    prev_qp: &mut i32,
    imb: IMbType,
) -> Result<()> {
    if matches!(imb, IMbType::IPcm) {
        return decode_pcm_mb_cabac(d, sps, mb_x, mb_y, pic, *prev_qp);
    }

    // --- transform_size_8x8_flag (I_NxN, High Profile) ---
    let mut transform_8x8 = false;
    let mut intra4x4_modes = [INTRA_DC_FAKE; 16];
    if matches!(imb, IMbType::INxN) {
        // §9.3.3.1.1.10 — flag is present before the intra mode bits when
        // pps.transform_8x8_mode_flag is set.
        if pps.transform_8x8_mode_flag {
            let inc = transform_size_8x8_flag_ctx_idx_inc(pic, mb_x, mb_y);
            let slice =
                &mut ctxs[CTX_IDX_TRANSFORM_SIZE_8X8_FLAG..CTX_IDX_TRANSFORM_SIZE_8X8_FLAG + 3];
            transform_8x8 = binarize::decode_transform_size_8x8_flag(d, slice, inc)?;
        }
        if transform_8x8 {
            // §8.3.2 — four Intra_8×8 modes: each stored slot is replicated
            // into the four 4×4 positions the 8×8 covers so the in-MB
            // neighbour-mode predictor (keyed on 4×4 raster) picks up the
            // correct §8.3.2.1 min() neighbour for subsequent 8×8 blocks.
            for blk8 in 0..4usize {
                let br_row = (blk8 >> 1) * 2;
                let br_col = (blk8 & 1) * 2;
                let prev_flag = {
                    let slice = &mut ctxs[CTX_IDX_PREV_INTRA4X4_PRED_MODE_FLAG
                        ..CTX_IDX_PREV_INTRA4X4_PRED_MODE_FLAG + 1];
                    binarize::decode_prev_intra4x4_pred_mode_flag(d, slice)?
                };
                let predicted =
                    predict_intra4x4_mode_with(pic, mb_x, mb_y, br_row, br_col, &intra4x4_modes);
                let mode = if prev_flag {
                    predicted
                } else {
                    // rem_intra8x8_pred_mode shares the FL(3) regular-mode
                    // binarization with rem_intra4x4_pred_mode (§9.3.3.1.1.7).
                    let slice = &mut ctxs[CTX_IDX_REM_INTRA4X4_PRED_MODE
                        ..CTX_IDX_REM_INTRA4X4_PRED_MODE + 1];
                    let rem = binarize::decode_rem_intra4x4_pred_mode(d, slice)? as u8;
                    if rem < predicted {
                        rem
                    } else {
                        rem + 1
                    }
                };
                for dr in 0..2 {
                    for dc in 0..2 {
                        intra4x4_modes[(br_row + dr) * 4 + br_col + dc] = mode;
                    }
                }
            }
        } else {
            for blk in 0..16usize {
                let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
                let prev_flag = {
                    let slice = &mut ctxs[CTX_IDX_PREV_INTRA4X4_PRED_MODE_FLAG
                        ..CTX_IDX_PREV_INTRA4X4_PRED_MODE_FLAG + 1];
                    binarize::decode_prev_intra4x4_pred_mode_flag(d, slice)?
                };
                let predicted =
                    predict_intra4x4_mode_with(pic, mb_x, mb_y, br_row, br_col, &intra4x4_modes);
                let mode = if prev_flag {
                    predicted
                } else {
                    let slice = &mut ctxs[CTX_IDX_REM_INTRA4X4_PRED_MODE
                        ..CTX_IDX_REM_INTRA4X4_PRED_MODE + 1];
                    let rem = binarize::decode_rem_intra4x4_pred_mode(d, slice)? as u8;
                    if rem < predicted {
                        rem
                    } else {
                        rem + 1
                    }
                };
                intra4x4_modes[br_row * 4 + br_col] = mode;
            }
        }
    }

    // --- intra_chroma_pred_mode (always present when chroma_format_idc != 0) ---
    #[cfg(feature = "cabac-trace")]
    {
        d.trace_label = "intra_chroma_pred_mode";
    }
    let chroma_mode_val = if sps.chroma_format_idc != 0 {
        let inc = intra_chroma_pred_mode_ctx_idx_inc(pic, mb_x, mb_y);
        let slice = &mut ctxs[CTX_IDX_INTRA_CHROMA_PRED_MODE..CTX_IDX_INTRA_CHROMA_PRED_MODE + 4];
        binarize::decode_intra_chroma_pred_mode(d, slice, inc)?
    } else {
        0
    };
    let chroma_pred_mode = IntraChromaMode::from_u8(chroma_mode_val as u8).ok_or_else(|| {
        Error::invalid(format!(
            "h264 cabac mb: bad intra_chroma_pred_mode {chroma_mode_val}"
        ))
    })?;

    // --- coded_block_pattern (only for I_NxN) ---
    let (cbp_luma, cbp_chroma) = match imb {
        IMbType::INxN => {
            let slice =
                &mut ctxs[CTX_IDX_CODED_BLOCK_PATTERN_LUMA..CTX_IDX_CODED_BLOCK_PATTERN_LUMA + 12];
            let cbp = binarize::decode_coded_block_pattern(
                d,
                slice,
                sps.chroma_format_idc as u8,
                |i, decoded| cbp_luma_ctx_idx_inc(pic, mb_x, mb_y, i, decoded),
                |bin| {
                    if bin == 0 {
                        cbp_chroma_ctx_idx_inc_any(pic, mb_x, mb_y)
                    } else {
                        cbp_chroma_ctx_idx_inc_ac(pic, mb_x, mb_y)
                    }
                },
            )?;
            ((cbp & 0x0F) as u8, ((cbp >> 4) & 0x03) as u8)
        }
        IMbType::I16x16 {
            cbp_luma,
            cbp_chroma,
            ..
        } => (cbp_luma, cbp_chroma),
        IMbType::IPcm => unreachable!(),
    };

    // --- mb_qp_delta (when needed) ---
    let needs_qp_delta =
        matches!(imb, IMbType::I16x16 { .. }) || (cbp_luma != 0 || cbp_chroma != 0);
    if needs_qp_delta {
        #[cfg(feature = "cabac-trace")]
        {
            d.trace_label = "mb_qp_delta";
        }
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

    // Initialise the MB info.
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        *info = MbInfo {
            qp_y,
            coded: true,
            intra: true,
            intra4x4_pred_mode: intra4x4_modes,
            intra_chroma_pred_mode: chroma_mode_val as u8,
            mb_type_i: Some(imb),
            transform_8x8,
            cbp_luma,
            cbp_chroma,
            ..Default::default()
        };
    }

    // --- Luma reconstruction ---
    match imb {
        IMbType::I16x16 {
            intra16x16_pred_mode,
            ..
        } => decode_luma_intra_16x16(
            d,
            ctxs,
            sps,
            pps,
            mb_x,
            mb_y,
            pic,
            intra16x16_pred_mode,
            cbp_luma,
            qp_y,
        )?,
        IMbType::INxN if transform_8x8 => {
            decode_luma_intra_8x8_cabac(d, ctxs, mb_x, mb_y, pic, &intra4x4_modes, cbp_luma, qp_y)?
        }
        IMbType::INxN => decode_luma_intra_nxn(
            d,
            ctxs,
            sps,
            pps,
            mb_x,
            mb_y,
            pic,
            &intra4x4_modes,
            cbp_luma,
            qp_y,
        )?,
        IMbType::IPcm => unreachable!(),
    }

    // --- Chroma reconstruction ---
    #[cfg(feature = "cabac-trace")]
    {
        d.trace_label = "chroma";
    }
    if sps.chroma_format_idc == 2 {
        decode_chroma_422(
            d,
            ctxs,
            pps,
            mb_x,
            mb_y,
            pic,
            chroma_pred_mode,
            cbp_chroma,
            qp_y,
            true,
        )?;
    } else {
        decode_chroma(
            d,
            ctxs,
            sps,
            pps,
            mb_x,
            mb_y,
            pic,
            chroma_pred_mode,
            cbp_chroma,
            qp_y,
        )?;
    }

    #[cfg(feature = "cabac-trace")]
    {
        d.trace_label = "end_of_slice";
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// I_PCM — raw pixel bytes with a mid-slice CABAC re-init (§7.3.5.1 / §9.3.1.2).
// ---------------------------------------------------------------------------

/// Handle an I_PCM macroblock inside a CABAC slice.
///
/// Steps per §9.3.1.2:
/// 1. Align the stream cursor to the next byte boundary (the skipped bits
///    are `pcm_alignment_zero_bit`s — by spec they must be 0, but this
///    decoder doesn't validate that value).
/// 2. Read `256 × 2^(BitDepthY−8)` luma and `2 × MbWidthC × MbHeightC ×
///    2^(BitDepthC−8)` chroma bytes verbatim into the reconstruction
///    buffers. This crate is fixed at 8-bit 4:2:0 (MbWidthC =
///    MbHeightC = 8), so the totals are 256 luma + 2 × 64 chroma = 384.
/// 3. Re-seed the CABAC arithmetic engine (`codIRange = 0x01FE`,
///    `codIOffset = read_bits(9)`) — the context state array is NOT
///    touched.
///
/// Mirrors the CAVLC I_PCM path in [`crate::mb::decode_pcm_mb`] for the
/// sample-write bookkeeping so deblocking and neighbour lookups behave
/// identically regardless of entropy mode.
fn decode_pcm_mb_cabac(
    d: &mut CabacDecoder<'_>,
    sps: &Sps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    prev_qp: i32,
) -> Result<()> {
    // §9.3.1.2 step 1 — align to next byte boundary (pcm_alignment_zero_bit).
    d.align_to_byte_for_pcm();

    // §9.3.1.2 step 2 — raw luma + chroma bytes. This crate only supports
    // 8-bit samples and chroma_format_idc ∈ {0 (monochrome), 1 (4:2:0)};
    // reject anything else before touching the stream so a malformed or
    // unsupported bitstream fails cleanly.
    match sps.chroma_format_idc {
        0 | 1 => {}
        v => {
            return Err(Error::unsupported(format!(
                "h264: CABAC I_PCM with chroma_format_idc={v} not supported (only 4:2:0)"
            )));
        }
    }

    let lstride = pic.luma_stride();
    let lo = pic.luma_off(mb_x, mb_y);
    for r in 0..16 {
        for c in 0..16 {
            pic.y[lo + r * lstride + c] = d.read_pcm_byte()?;
        }
    }
    if sps.chroma_format_idc != 0 {
        let cstride = pic.chroma_stride();
        let co = pic.chroma_off(mb_x, mb_y);
        for r in 0..8 {
            for c in 0..8 {
                pic.cb[co + r * cstride + c] = d.read_pcm_byte()?;
            }
        }
        for r in 0..8 {
            for c in 0..8 {
                pic.cr[co + r * cstride + c] = d.read_pcm_byte()?;
            }
        }
    }

    // §9.3.1.2 step 3 — re-initialise the arithmetic engine.
    d.reinit_after_pcm()?;

    // §9.3.3.1.1.3 — for CABAC neighbour derivations, an I_PCM MB is treated
    // as coded and NOT as I_NxN. mb_type_i is left as None (the `I_NxN` check
    // in `mb_type_i_ctx_idx_inc` then naturally returns `coded=true`).
    // For intra_chroma_pred_mode neighbour derivation, I_PCM sets
    // `intra_chroma_pred_mode = 0` per the spec so condTermFlag = 0.
    // luma_nc/cb_nc/cr_nc are set to 16 mirroring the CAVLC path — any value
    // >0 marks the sub-block as having non-zero residual for neighbour
    // decisions. QP_Y is forced to the slice QP_Y (not inherited from
    // prev_qp after this MB per §8.5).
    let info = pic.mb_info_mut(mb_x, mb_y);
    *info = MbInfo {
        qp_y: prev_qp,
        coded: true,
        intra: true,
        luma_nc: [16; 16],
        cb_nc: [16; 16],
        cr_nc: [16; 16],
        intra4x4_pred_mode: [INTRA_DC_FAKE; 16],
        intra_chroma_pred_mode: 0,
        mb_type_i: Some(IMbType::IPcm),
        ..Default::default()
    };
    pic.last_mb_qp_delta_was_nonzero = false;
    Ok(())
}

// ---------------------------------------------------------------------------
// Luma — I_NxN.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_luma_intra_nxn(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    _sps: &Sps,
    _pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    modes: &[u8; 16],
    cbp_luma: u8,
    qp_y: i32,
) -> Result<()> {
    let lstride = pic.luma_stride();
    let lo_mb = pic.luma_off(mb_x, mb_y);

    for blk in 0..16usize {
        let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
        let mode_v = modes[br_row * 4 + br_col];
        let mode = Intra4x4Mode::from_u8(mode_v).ok_or_else(|| {
            Error::invalid(format!("h264 cabac mb: invalid intra4x4 mode {mode_v}"))
        })?;
        let neigh = collect_intra4x4_neighbours(pic, mb_x, mb_y, br_row, br_col);
        let mut pred = [0u8; 16];
        predict_intra_4x4(&mut pred, mode, &neigh);

        let cbp_bit_idx = (br_row / 2) * 2 + (br_col / 2);
        let has_residual = (cbp_luma >> cbp_bit_idx) & 1 != 0;

        let mut residual = [0i32; 16];
        let mut total_coeff = 0u32;
        if has_residual {
            let neighbours = cbf_neighbours_luma(pic, mb_x, mb_y, br_row, br_col);
            let coeffs =
                decode_residual_block_in_place(d, ctxs, BlockCat::Luma4x4, &neighbours, 16, true)?;
            residual = coeffs;
            total_coeff = coeffs.iter().filter(|&&v| v != 0).count() as u32;
            // §7.4.2.2 — Intra-Y 4×4 slot 0.
            let scale = *pic.scaling_lists.matrix_4x4(0);
            dequantize_4x4_scaled(&mut residual, qp_y, &scale);
            idct_4x4(&mut residual);
        }
        let lo = lo_mb + br_row * 4 * lstride + br_col * 4;
        for r in 0..4 {
            for c in 0..4 {
                let v = pred[r * 4 + c] as i32 + residual[r * 4 + c];
                pic.y[lo + r * lstride + c] = v.clamp(0, 255) as u8;
            }
        }
        pic.mb_info_mut(mb_x, mb_y).luma_nc[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Luma — I_NxN with transform_size_8x8_flag (High Profile, §8.3.2 + §8.5.13).
// ---------------------------------------------------------------------------
//
// Mirrors the CAVLC intra 8×8 path in `mb::decode_luma_intra_8x8` but feeds
// the residual through the CABAC 64-coefficient decoder (`ctxBlockCat = 5`,
// §9.3.3.1.1.9). Each 8×8 sub-block gets its own `coded_block_flag` bin
// (ctxIdxOffset 1012) + full significance map + UEGk levels. The decoded
// 64-coefficient vector is zig-zag-inverted, dequantised via the 8×8
// scaling list (intra-Y slot 0), and fed through `idct_8x8` before being
// added to the intra-8×8 prediction and clamped into `pic.y`.

#[allow(clippy::too_many_arguments)]
fn decode_luma_intra_8x8_cabac(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    modes: &[u8; 16],
    cbp_luma: u8,
    qp_y: i32,
) -> Result<()> {
    let lstride = pic.luma_stride();
    let lo_mb = pic.luma_off(mb_x, mb_y);

    for blk8 in 0..4usize {
        let br8_row = blk8 >> 1;
        let br8_col = blk8 & 1;
        let mode_v = modes[(br8_row * 2) * 4 + br8_col * 2];
        let mode = Intra8x8Mode::from_u8(mode_v).ok_or_else(|| {
            Error::invalid(format!("h264 cabac mb: invalid intra8x8 mode {mode_v}"))
        })?;
        let neigh = collect_intra8x8_neighbours(pic, mb_x, mb_y, br8_row, br8_col);
        let mut pred = [0u8; 64];
        predict_intra_8x8(&mut pred, mode, &neigh);

        let has_residual = (cbp_luma >> blk8) & 1 != 0;
        let mut residual = [0i32; 64];
        if has_residual {
            let r0 = br8_row * 2;
            let c0 = br8_col * 2;
            // §9.3.3.1.1.9 — CBF neighbour for the 8×8 block takes its
            // left and above from the 4×4 cache at the block's top-left
            // corner, same as the CAVLC nC predictor.
            let neighbours = cbf_neighbours_luma(pic, mb_x, mb_y, r0, c0);
            #[cfg(feature = "cabac-trace")]
            {
                d.trace_label = match blk8 {
                    0 => "luma8x8.blk0",
                    1 => "luma8x8.blk1",
                    2 => "luma8x8.blk2",
                    _ => "luma8x8.blk3",
                };
            }
            let coeffs = decode_luma_8x8_residual_in_place(d, ctxs, &neighbours, true)?;
            residual = coeffs;
            let nnz = residual.iter().filter(|&&v| v != 0).count() as u16;
            // §7.4.2.2 — Intra-Y 8×8 uses scaling-list slot 0.
            let scale = *pic.scaling_lists.matrix_8x8(0);
            dequantize_8x8_scaled(&mut residual, qp_y, &scale);
            idct_8x8(&mut residual);
            // Mirror the CAVLC path: write per-4×4 counts into `luma_nc`
            // so neighbour derivations on subsequent MBs see a consistent
            // "this 8×8 had residual" signal at the top-left slot.
            let info = pic.mb_info_mut(mb_x, mb_y);
            let byte = nnz.min(16) as u8;
            info.luma_nc[r0 * 4 + c0] = byte;
            info.luma_nc[r0 * 4 + c0 + 1] = byte;
            info.luma_nc[(r0 + 1) * 4 + c0] = byte;
            info.luma_nc[(r0 + 1) * 4 + c0 + 1] = byte;
        }

        let lo = lo_mb + (br8_row * 8) * lstride + br8_col * 8;
        for r in 0..8 {
            for c in 0..8 {
                let v = pred[r * 8 + c] as i32 + residual[r * 8 + c];
                pic.y[lo + r * lstride + c] = v.clamp(0, 255) as u8;
            }
        }
    }
    Ok(())
}

fn collect_intra8x8_neighbours(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    br8_row: usize,
    br8_col: usize,
) -> Intra8x8Neighbours {
    // Mirrors `crate::mb::collect_intra8x8_neighbours`; duplicated here to
    // keep CABAC path's dependencies self-contained.
    let lstride = pic.luma_stride();

    let top_avail = br8_row > 0 || mb_y > 0;
    let mut top = [0u8; 16];
    let mut top_right_available = false;
    if top_avail {
        let row_y_global = (mb_y as usize) * 16 + br8_row * 8 - 1;
        let row_off = row_y_global * lstride;
        let col_base = (mb_x as usize) * 16 + br8_col * 8;
        for i in 0..8 {
            top[i] = pic.y[row_off + col_base + i];
        }
        top_right_available = match (br8_row, br8_col) {
            (0, 0) => mb_y > 0,
            (0, 1) => {
                mb_y > 0 && mb_x + 1 < pic.mb_width && pic.mb_info_at(mb_x + 1, mb_y - 1).coded
            }
            (1, 0) => true,
            _ => false,
        };
        if top_right_available {
            for i in 0..8 {
                top[8 + i] = pic.y[row_off + col_base + 8 + i];
            }
        } else {
            for i in 0..8 {
                top[8 + i] = top[7];
            }
        }
    }

    let left_avail = br8_col > 0 || mb_x > 0;
    let mut left = [0u8; 8];
    if left_avail {
        let col_x_global = (mb_x as usize) * 16 + br8_col * 8 - 1;
        for i in 0..8 {
            let row = (mb_y as usize) * 16 + br8_row * 8 + i;
            left[i] = pic.y[row * lstride + col_x_global];
        }
    }

    let tl_avail = top_avail && left_avail;
    let top_left = if tl_avail {
        let row_y_global = (mb_y as usize) * 16 + br8_row * 8 - 1;
        let col_x_global = (mb_x as usize) * 16 + br8_col * 8 - 1;
        pic.y[row_y_global * lstride + col_x_global]
    } else {
        0
    };

    Intra8x8Neighbours {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
        top_right_available,
    }
}

// ---------------------------------------------------------------------------
// Luma — I_16x16.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_luma_intra_16x16(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    _sps: &Sps,
    _pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    intra16x16_pred_mode: u8,
    cbp_luma: u8,
    qp_y: i32,
) -> Result<()> {
    let neigh = collect_intra16x16_neighbours(pic, mb_x, mb_y);
    let mut pred = [0u8; 256];
    let mode = Intra16x16Mode::from_u8(intra16x16_pred_mode).ok_or_else(|| {
        Error::invalid(format!(
            "h264 cabac mb: invalid intra16x16 mode {intra16x16_pred_mode}"
        ))
    })?;
    predict_intra_16x16(&mut pred, mode, &neigh);

    // DC luma — always coded. §9.3.3.1.1.9: CBF ctxIdxInc derives from
    // the neighbour MB's luma16x16 DC CBF (not from per-4x4 AC counts).
    // For an I_16x16 neighbour the DC block is always present and its CBF
    // is recorded via `luma_nc[..].iter().any(|&v| v != 0)` proxy — but
    // that's an AC signal, not DC. We use `luma_dc_coded` stored in
    // MbInfo instead.
    let dc_neigh = luma16x16_dc_cbf_neighbours(pic, mb_x, mb_y);
    #[cfg(feature = "cabac-trace")]
    {
        d.trace_label = "luma16x16.dc";
    }
    let dc_coeffs =
        decode_residual_block_in_place(d, ctxs, BlockCat::Luma16x16Dc, &dc_neigh, 16, true)?;
    let mut dc = dc_coeffs;
    // Record that this MB's DC block had any nonzero coefficient so
    // subsequent neighbours can derive their Luma16x16 DC CBF ctxIdxInc
    // per §9.3.3.1.1.9.
    pic.mb_info_mut(mb_x, mb_y).luma16x16_dc_cbf = dc_coeffs.iter().any(|&v| v != 0);
    // §8.5.10 — Luma_16×16 DC dequant with Intra-Y (slot 0) weight.
    let w_dc = pic.scaling_lists.matrix_4x4(0)[0];
    inv_hadamard_4x4_dc_scaled(&mut dc, qp_y, w_dc);

    let lstride = pic.luma_stride();
    let lo_mb = pic.luma_off(mb_x, mb_y);
    for blk in 0..16usize {
        let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
        let mut residual = [0i32; 16];
        let mut total_coeff = 0u32;
        if cbp_luma != 0 {
            let neighbours = cbf_neighbours_luma(pic, mb_x, mb_y, br_row, br_col);
            let ac = decode_residual_block_in_place(
                d,
                ctxs,
                BlockCat::Luma16x16Ac,
                &neighbours,
                15,
                true,
            )?;
            residual = ac;
            total_coeff = ac.iter().filter(|&&v| v != 0).count() as u32;
            // §7.4.2.2 — Intra-Y AC uses the intra-Y 4×4 list (slot 0).
            let scale = *pic.scaling_lists.matrix_4x4(0);
            dequantize_4x4_scaled(&mut residual, qp_y, &scale);
        }
        // Splice DC into slot 0.
        residual[0] = dc[br_row * 4 + br_col];
        idct_4x4(&mut residual);

        let lo = lo_mb + br_row * 4 * lstride + br_col * 4;
        for r in 0..4 {
            for c in 0..4 {
                let v = pred[(br_row * 4 + r) * 16 + (br_col * 4 + c)] as i32 + residual[r * 4 + c];
                pic.y[lo + r * lstride + c] = v.clamp(0, 255) as u8;
            }
        }
        pic.mb_info_mut(mb_x, mb_y).luma_nc[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Chroma reconstruction.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_chroma(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    _sps: &Sps,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    chroma_mode: IntraChromaMode,
    cbp_chroma: u8,
    qp_y: i32,
) -> Result<()> {
    let qpc = chroma_qp(qp_y, pps.chroma_qp_index_offset);
    let neigh_cb = collect_chroma_neighbours(pic, mb_x, mb_y, true);
    let neigh_cr = collect_chroma_neighbours(pic, mb_x, mb_y, false);
    let mut pred_cb = [0u8; 64];
    let mut pred_cr = [0u8; 64];
    predict_intra_chroma(&mut pred_cb, chroma_mode, &neigh_cb);
    predict_intra_chroma(&mut pred_cr, chroma_mode, &neigh_cr);

    let mut dc_cb = [0i32; 4];
    let mut dc_cr = [0i32; 4];
    if cbp_chroma >= 1 {
        // §9.3.3.1.1.9 — chroma DC CBF ctxIdxInc derives from the
        // neighbour MB's chroma DC CBF (bit 6 for Cb, bit 7 for Cr in
        // FFmpeg's cbp_table). Track the per-plane flag in `MbInfo`.
        let neigh_cb_dc = chroma_dc_cbf_neighbours(pic, mb_x, mb_y, 0);
        let cb =
            decode_residual_block_in_place(d, ctxs, BlockCat::ChromaDc, &neigh_cb_dc, 4, true)?;
        for i in 0..4 {
            dc_cb[i] = cb[i];
        }
        pic.mb_info_mut(mb_x, mb_y).chroma_dc_cbf[0] = cb.iter().any(|&v| v != 0);
        let neigh_cr_dc = chroma_dc_cbf_neighbours(pic, mb_x, mb_y, 1);
        let cr =
            decode_residual_block_in_place(d, ctxs, BlockCat::ChromaDc, &neigh_cr_dc, 4, true)?;
        for i in 0..4 {
            dc_cr[i] = cr[i];
        }
        pic.mb_info_mut(mb_x, mb_y).chroma_dc_cbf[1] = cr.iter().any(|&v| v != 0);
        // §8.5.11.1 — CABAC I-slice chroma is intra.
        let w_cb = pic.scaling_lists.matrix_4x4(1)[0];
        let w_cr = pic.scaling_lists.matrix_4x4(2)[0];
        inv_hadamard_2x2_chroma_dc_scaled(&mut dc_cb, qpc, w_cb);
        inv_hadamard_2x2_chroma_dc_scaled(&mut dc_cr, qpc, w_cr);
    }

    let cstride = pic.chroma_stride();
    let co = pic.chroma_off(mb_x, mb_y);
    for plane_kind in [true, false] {
        let pred = if plane_kind { &pred_cb } else { &pred_cr };
        let dc = if plane_kind { &dc_cb } else { &dc_cr };
        let mut nc_arr = [0u8; 4];
        for blk_idx in 0..4u8 {
            let br_row = (blk_idx >> 1) as usize;
            let br_col = (blk_idx & 1) as usize;
            let mut res = [0i32; 16];
            let mut total_coeff = 0u32;
            if cbp_chroma == 2 {
                // §9.3.3.1.1.9 — for chroma AC, the CBF ctxIdxInc derives
                // from the neighbour 4×4 block's chroma AC CBF. Within the
                // MB we read the already-decoded block's `total_coeff`;
                // crossing the MB boundary we peek at the neighbour MB's
                // stored chroma nc. Unavailable intra neighbour → 1.
                let neigh =
                    chroma_ac_cbf_neighbours(pic, mb_x, mb_y, plane_kind, br_row, br_col, &nc_arr);
                let ac =
                    decode_residual_block_in_place(d, ctxs, BlockCat::ChromaAc, &neigh, 15, true)?;
                res = ac;
                total_coeff = ac.iter().filter(|&&v| v != 0).count() as u32;
                // §7.4.2.2 — Intra chroma slots 1 (Cb) / 2 (Cr). CABAC
                // I-slice chroma is always intra.
                let cat = if plane_kind { 1 } else { 2 };
                let scale = *pic.scaling_lists.matrix_4x4(cat);
                dequantize_4x4_scaled(&mut res, qpc, &scale);
            }
            res[0] = dc[(br_row << 1) | br_col];
            idct_4x4(&mut res);
            let off_in_mb = br_row * 4 * cstride + br_col * 4;
            let plane = if plane_kind { &mut pic.cb } else { &mut pic.cr };
            for r in 0..4 {
                for c in 0..4 {
                    let v = pred[(br_row * 4 + r) * 8 + (br_col * 4 + c)] as i32 + res[r * 4 + c];
                    plane[co + off_in_mb + r * cstride + c] = v.clamp(0, 255) as u8;
                }
            }
            nc_arr[(br_row << 1) | br_col] = total_coeff as u8;
        }
        let info = pic.mb_info_mut(mb_x, mb_y);
        let dst = if plane_kind {
            &mut info.cb_nc
        } else {
            &mut info.cr_nc
        };
        dst[..4].copy_from_slice(&nc_arr);
    }
    Ok(())
}

/// 4:2:2 chroma reconstruction under CABAC (§8.5.11.2, §8.3.4
/// ChromaArrayType == 2). Mirrors [`crate::mb::decode_chroma_422`] (the
/// CAVLC path) but reads the residual through CABAC — chroma DC is a
/// 2×4 block (8 coeffs, ctxBlockCat = 3, maxNumCoeff = 4·NumC8x8 with
/// NumC8x8 = 2 under ChromaArrayType = 2), dequantised via the
/// `QP'_C,DC = QP'_C + 3` offset baked into
/// [`inv_hadamard_2x4_chroma_dc_scaled`]. The 8 chroma-AC blocks per
/// plane share ctxBlockCat = 4 with the 4:2:0 path. `intra` selects the
/// `condTermFlagN` default + inter/intra scaling slot split (1/2 vs 4/5).
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_chroma_422(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    chroma_mode: IntraChromaMode,
    cbp_chroma: u8,
    qp_y: i32,
    intra: bool,
) -> Result<()> {
    let qpc_cb = chroma_qp(qp_y, pps.chroma_qp_index_offset);
    let qpc_cr = chroma_qp(qp_y, pps.second_chroma_qp_index_offset);

    let mut pred_cb = [0u8; 128];
    let mut pred_cr = [0u8; 128];
    if intra {
        let neigh_cb = collect_chroma_422_neighbours(pic, mb_x, mb_y, true);
        let neigh_cr = collect_chroma_422_neighbours(pic, mb_x, mb_y, false);
        predict_intra_chroma_8x16(&mut pred_cb, chroma_mode, &neigh_cb);
        predict_intra_chroma_8x16(&mut pred_cr, chroma_mode, &neigh_cr);
    }

    let mut dc_cb = [0i32; 8];
    let mut dc_cr = [0i32; 8];
    if cbp_chroma >= 1 {
        // §9.3.3.1.1.9 — ctxBlockCat = 3 ChromaDC shared with 4:2:0;
        // `max_num_coeff = 8` under ChromaArrayType = 2. The CABAC
        // residual decoder writes coded-scan positions 0..7 which match
        // the 2×4 raster directly per §6.4.3 (column-then-row walk).
        let neigh_cb_dc = chroma_dc_cbf_neighbours(pic, mb_x, mb_y, 0);
        let cb = decode_residual_block_cabac_8(d, ctxs, BlockCat::ChromaDc, &neigh_cb_dc, intra)?;
        dc_cb = cb;
        pic.mb_info_mut(mb_x, mb_y).chroma_dc_cbf[0] = cb.iter().any(|&v| v != 0);
        let neigh_cr_dc = chroma_dc_cbf_neighbours(pic, mb_x, mb_y, 1);
        let cr = decode_residual_block_cabac_8(d, ctxs, BlockCat::ChromaDc, &neigh_cr_dc, intra)?;
        dc_cr = cr;
        pic.mb_info_mut(mb_x, mb_y).chroma_dc_cbf[1] = cr.iter().any(|&v| v != 0);
        // §7.4.2.2 — intra chroma (slots 1/2) vs inter chroma (slots 4/5).
        let (cb_slot, cr_slot) = if intra { (1, 2) } else { (4, 5) };
        let w_cb = pic.scaling_lists.matrix_4x4(cb_slot)[0];
        let w_cr = pic.scaling_lists.matrix_4x4(cr_slot)[0];
        inv_hadamard_2x4_chroma_dc_scaled(&mut dc_cb, qpc_cb, w_cb);
        inv_hadamard_2x4_chroma_dc_scaled(&mut dc_cr, qpc_cr, w_cr);
    }

    let row_step = pic.chroma_row_stride_for_at(mb_x, mb_y);
    let co = pic.chroma_off(mb_x, mb_y);
    for plane_kind in [true, false] {
        let pred = if plane_kind { &pred_cb } else { &pred_cr };
        let dc = if plane_kind { &dc_cb } else { &dc_cr };
        let qpc = if plane_kind { qpc_cb } else { qpc_cr };
        let (intra_slot, inter_slot) = if plane_kind { (1, 4) } else { (2, 5) };
        let cat = if intra { intra_slot } else { inter_slot };

        // 4:2:2 AC grid: 4 rows × 2 cols of 4×4 blocks. Row-major order
        // so neighbour CBF lookups see already-decoded blocks above and
        // to the left (§9.3.3.1.1.9 Table 9-41).
        let mut nc_arr = [0u8; 8];
        for blk_row in 0..4usize {
            for blk_col in 0..2usize {
                let blk_idx = blk_row * 2 + blk_col;
                let mut res = [0i32; 16];
                let mut total_coeff = 0u32;
                if cbp_chroma == 2 {
                    let neigh = chroma_ac_cbf_neighbours_422(
                        pic, mb_x, mb_y, plane_kind, blk_row, blk_col, &nc_arr,
                    );
                    let ac = decode_residual_block_in_place(
                        d,
                        ctxs,
                        BlockCat::ChromaAc,
                        &neigh,
                        15,
                        intra,
                    )?;
                    res = ac;
                    total_coeff = ac.iter().filter(|&&v| v != 0).count() as u32;
                    let scale = *pic.scaling_lists.matrix_4x4(cat);
                    dequantize_4x4_scaled(&mut res, qpc, &scale);
                }
                res[0] = dc[blk_idx];
                idct_4x4(&mut res);
                let off_in_mb = blk_row * 4 * row_step + blk_col * 4;
                let plane = if plane_kind { &mut pic.cb } else { &mut pic.cr };
                if intra {
                    for r in 0..4 {
                        for c in 0..4 {
                            let p_idx = (blk_row * 4 + r) * 8 + (blk_col * 4 + c);
                            let v = pred[p_idx] as i32 + res[r * 4 + c];
                            plane[co + off_in_mb + r * row_step + c] = v.clamp(0, 255) as u8;
                        }
                    }
                } else {
                    for r in 0..4 {
                        for c in 0..4 {
                            let base = plane[co + off_in_mb + r * row_step + c] as i32;
                            let v = (base + res[r * 4 + c]).clamp(0, 255) as u8;
                            plane[co + off_in_mb + r * row_step + c] = v;
                        }
                    }
                }
                nc_arr[blk_idx] = total_coeff as u8;
                let info = pic.mb_info_mut(mb_x, mb_y);
                let dst = if plane_kind {
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

/// Helper: run [`decode_residual_block_cabac_chroma_dc_422`] for an
/// 8-coeff 4:2:2 ChromaDC block (ctxBlockCat = 3, NumC8x8 = 2).
/// Handles the gather/scatter against the slice-wide context array so
/// probability-state updates persist across blocks. Returns the 8 DC
/// coefficients in the 2×4 raster, which matches the coded scan order
/// per §6.4.3 — no inverse scan needed.
fn decode_residual_block_cabac_8(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    cat: BlockCat,
    neighbours: &CbfNeighbours,
    intra: bool,
) -> Result<[i32; 8]> {
    let plan = ResidualCtxPlan::new(cat, neighbours, intra);
    let mut window = plan.gather(ctxs);
    let out = decode_residual_block_cabac_chroma_dc_422(d, &mut window, neighbours)?;
    plan.scatter(ctxs, &window);
    Ok(out)
}

/// 4:2:2 chroma-AC CBF neighbour derivation (§9.3.3.1.1.9 Table 9-41
/// under ChromaArrayType = 2). The 2×4 grid stores per-block
/// `total_coeff` at `MbInfo::{cb,cr}_nc[blk_row*2 + blk_col]`. Cross-MB
/// left/top reads follow the §6.4.10.4 chroma-block neighbour rule for
/// 4:2:2 (8 blocks laid out as 4 rows × 2 cols).
fn chroma_ac_cbf_neighbours_422(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    plane_kind: bool,
    blk_row: usize,
    blk_col: usize,
    already_decoded: &[u8; 8],
) -> CbfNeighbours {
    let left = if blk_col > 0 {
        Some(already_decoded[blk_row * 2 + (blk_col - 1)] != 0)
    } else if mb_x > 0 {
        let m = pic.mb_info_at(mb_x - 1, mb_y);
        if m.coded {
            let nc = if plane_kind { &m.cb_nc } else { &m.cr_nc };
            Some(nc[blk_row * 2 + 1] != 0)
        } else {
            None
        }
    } else {
        None
    };
    let above = if blk_row > 0 {
        Some(already_decoded[(blk_row - 1) * 2 + blk_col] != 0)
    } else if mb_y > 0 {
        let m = pic.mb_info_at(mb_x, mb_y - 1);
        if m.coded {
            let nc = if plane_kind { &m.cb_nc } else { &m.cr_nc };
            // Bottom row of 2×4 grid is at blk_row = 3 → indices 6, 7.
            Some(nc[3 * 2 + blk_col] != 0)
        } else {
            None
        }
    } else {
        None
    };
    CbfNeighbours { left, above }
}

/// 4:2:2 chroma 8×16 neighbour gather for CABAC intra prediction —
/// mirrors [`crate::mb::collect_chroma_422_neighbours`].
fn collect_chroma_422_neighbours(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    cb: bool,
) -> IntraChroma8x16Neighbours {
    let row_stride = pic.chroma_row_stride_for_at(mb_x, mb_y);
    let co_mb = pic.chroma_off(mb_x, mb_y);
    let plane = if cb { &pic.cb } else { &pic.cr };
    let top_avail = pic.mb_top_available(mb_y);
    let mut top = [0u8; 8];
    if top_avail {
        let off = co_mb - row_stride;
        for i in 0..8 {
            top[i] = plane[off + i];
        }
    }
    let left_avail = mb_x > 0;
    let mut left = [0u8; 16];
    if left_avail {
        for i in 0..16 {
            left[i] = plane[co_mb + i * row_stride - 1];
        }
    }
    let tl_avail = top_avail && left_avail;
    let top_left = if tl_avail {
        plane[co_mb - row_stride - 1]
    } else {
        0
    };
    IntraChroma8x16Neighbours {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
    }
}

/// Derive Chroma DC CBF neighbours for plane `c` (0=Cb, 1=Cr) per
/// §9.3.3.1.1.9. The neighbour contributes `condTermFlagN` equal to
/// the neighbour MB's chroma DC CBF for plane `c` (tracked in
/// `MbInfo::chroma_dc_cbf`).
fn chroma_dc_cbf_neighbours(pic: &Picture, mb_x: u32, mb_y: u32, c: usize) -> CbfNeighbours {
    let probe = |mx: u32, my: u32| -> Option<bool> {
        let info = pic.mb_info_at(mx, my);
        if !info.coded {
            return None;
        }
        Some(info.chroma_dc_cbf[c])
    };
    let left = if mb_x > 0 {
        probe(mb_x - 1, mb_y)
    } else {
        None
    };
    let above = if mb_y > 0 {
        probe(mb_x, mb_y - 1)
    } else {
        None
    };
    CbfNeighbours { left, above }
}

/// Derive Luma16x16 DC CBF neighbours per §9.3.3.1.1.9. The neighbour
/// contributes to condTermFlagN iff it's an I_16x16 MB with luma DC
/// CBF set. I_NxN / inter neighbours have no DC block — their
/// condTermFlagN is 0 (even when coded), which matches `Some(false)`.
fn luma16x16_dc_cbf_neighbours(pic: &Picture, mb_x: u32, mb_y: u32) -> CbfNeighbours {
    let probe = |mx: u32, my: u32| -> Option<bool> {
        let info = pic.mb_info_at(mx, my);
        if !info.coded {
            return None;
        }
        Some(info.luma16x16_dc_cbf)
    };
    let left = if mb_x > 0 {
        probe(mb_x - 1, mb_y)
    } else {
        None
    };
    let above = if mb_y > 0 {
        probe(mb_x, mb_y - 1)
    } else {
        None
    };
    CbfNeighbours { left, above }
}

/// Derive chroma-AC CBF neighbours for block `(br_row, br_col)` in the
/// current 2×2 chroma grid. `already_decoded[i]` is this MB's running
/// `nc_arr` for the plane being decoded. Crossing the MB boundary reads
/// the left/top neighbour MB's stored chroma nc (Cb or Cr selected by
/// `plane_kind`). §9.3.3.1.1.9 Table 9-41 — condTermFlagN from the
/// chroma AC CBF.
fn chroma_ac_cbf_neighbours(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    plane_kind: bool,
    br_row: usize,
    br_col: usize,
    already_decoded: &[u8; 4],
) -> CbfNeighbours {
    let left = if br_col > 0 {
        Some(already_decoded[br_row * 2 + br_col - 1] != 0)
    } else if mb_x > 0 {
        let m = pic.mb_info_at(mb_x - 1, mb_y);
        if m.coded {
            let nc = if plane_kind { &m.cb_nc } else { &m.cr_nc };
            Some(nc[br_row * 2 + 1] != 0)
        } else {
            None
        }
    } else {
        None
    };
    let above = if br_row > 0 {
        Some(already_decoded[(br_row - 1) * 2 + br_col] != 0)
    } else if mb_y > 0 {
        let m = pic.mb_info_at(mb_x, mb_y - 1);
        if m.coded {
            let nc = if plane_kind { &m.cb_nc } else { &m.cr_nc };
            Some(nc[2 + br_col] != 0)
        } else {
            None
        }
    } else {
        None
    };
    CbfNeighbours { left, above }
}

// ---------------------------------------------------------------------------
// Neighbour helpers (duplicated from mb.rs — keeps the crate boundary with
// CAVLC simple without pulling on private items).
// ---------------------------------------------------------------------------

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

fn predict_intra4x4_mode_with(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    br_row: usize,
    br_col: usize,
    in_progress: &[u8; 16],
) -> u8 {
    let left_mode = if br_col > 0 {
        Some(in_progress[br_row * 4 + br_col - 1])
    } else if mb_x > 0 {
        let li = pic.mb_info_at(mb_x - 1, mb_y);
        if li.coded && li.intra {
            Some(li.intra4x4_pred_mode[br_row * 4 + 3])
        } else {
            None
        }
    } else {
        None
    };
    let top_mode = if br_row > 0 {
        Some(in_progress[(br_row - 1) * 4 + br_col])
    } else if mb_y > 0 {
        let ti = pic.mb_info_at(mb_x, mb_y - 1);
        if ti.coded && ti.intra {
            Some(ti.intra4x4_pred_mode[12 + br_col])
        } else {
            None
        }
    } else {
        None
    };
    match (left_mode, top_mode) {
        (Some(l), Some(t)) => l.min(t),
        _ => INTRA_DC_FAKE,
    }
}

fn collect_intra4x4_neighbours(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    br_row: usize,
    br_col: usize,
) -> Intra4x4Neighbours {
    let lstride = pic.luma_stride();

    let top_avail = br_row > 0 || mb_y > 0;
    let mut top = [0u8; 8];
    if top_avail {
        let row_y_global = if br_row > 0 {
            (mb_y as usize) * 16 + br_row * 4 - 1
        } else {
            (mb_y as usize) * 16 - 1
        };
        let row_off = row_y_global * lstride;
        for i in 0..4 {
            top[i] = pic.y[row_off + (mb_x as usize) * 16 + br_col * 4 + i];
        }
        // Top-right: replicate top[3] if unavailable.
        for i in 0..4 {
            top[4 + i] = top[3];
        }
    }

    let left_avail = br_col > 0 || mb_x > 0;
    let mut left = [0u8; 4];
    if left_avail {
        let col_x_global: usize = if br_col > 0 {
            (mb_x as usize) * 16 + br_col * 4 - 1
        } else {
            (mb_x as usize) * 16 - 1
        };
        for i in 0..4 {
            let row = (mb_y as usize) * 16 + br_row * 4 + i;
            left[i] = pic.y[row * lstride + col_x_global];
        }
    }

    let tl_avail = top_avail && left_avail;
    let top_left = if tl_avail {
        let row_y_global: usize = if br_row > 0 {
            (mb_y as usize) * 16 + br_row * 4 - 1
        } else {
            (mb_y as usize) * 16 - 1
        };
        let col_x_global: usize = if br_col > 0 {
            (mb_x as usize) * 16 + br_col * 4 - 1
        } else {
            (mb_x as usize) * 16 - 1
        };
        pic.y[row_y_global * lstride + col_x_global]
    } else {
        0
    };

    Intra4x4Neighbours {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
        top_right_available: false,
    }
}

fn collect_intra16x16_neighbours(pic: &Picture, mb_x: u32, mb_y: u32) -> Intra16x16Neighbours {
    let lstride = pic.luma_stride();
    let lo_mb = pic.luma_off(mb_x, mb_y);
    let top_avail = mb_y > 0;
    let mut top = [0u8; 16];
    if top_avail {
        let off = lo_mb - lstride;
        for i in 0..16 {
            top[i] = pic.y[off + i];
        }
    }
    let left_avail = mb_x > 0;
    let mut left = [0u8; 16];
    if left_avail {
        for i in 0..16 {
            left[i] = pic.y[lo_mb + i * lstride - 1];
        }
    }
    let tl_avail = top_avail && left_avail;
    let top_left = if tl_avail {
        pic.y[lo_mb - lstride - 1]
    } else {
        0
    };
    Intra16x16Neighbours {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
    }
}

fn collect_chroma_neighbours(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    cb: bool,
) -> IntraChromaNeighbours {
    let cstride = pic.chroma_stride();
    let co_mb = pic.chroma_off(mb_x, mb_y);
    let plane = if cb { &pic.cb } else { &pic.cr };
    let top_avail = mb_y > 0;
    let mut top = [0u8; 8];
    if top_avail {
        let off = co_mb - cstride;
        for i in 0..8 {
            top[i] = plane[off + i];
        }
    }
    let left_avail = mb_x > 0;
    let mut left = [0u8; 8];
    if left_avail {
        for i in 0..8 {
            left[i] = plane[co_mb + i * cstride - 1];
        }
    }
    let tl_avail = top_avail && left_avail;
    let top_left = if tl_avail {
        plane[co_mb - cstride - 1]
    } else {
        0
    };
    IntraChromaNeighbours {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
    }
}
