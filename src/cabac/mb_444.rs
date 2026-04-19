//! CABAC I-slice macroblock decode for 4:4:4 (ChromaArrayType == 3).
//!
//! Mirrors [`crate::cabac::mb::decode_i_mb_cabac`] but follows the §6.4.1
//! 4:4:4 pipeline:
//!
//! * `intra_chroma_pred_mode` is absent from the bitstream — each chroma
//!   plane inherits the luma plane's intra mode (FFmpeg's
//!   `decode_chroma = false` path at `CHROMA444(h)`).
//! * Each plane decodes three back-to-back luma-style residual streams
//!   (I_16×16 DC + 16 AC, or 16 × 4×4 for I_NxN). No chroma DC 2×2
//!   Hadamard; no chroma-specific ctxBlockCat. The three streams use the
//!   shared luma context banks (ctxBlockCat 0 / 1 / 2 / 5) — the spec's
//!   separate 4:4:4 Cb/Cr ctxBlockCat values 6..=13 are an accuracy
//!   refinement left for a follow-up.
//! * CBP is a 4-bit luma-only value (chroma CBP bits absent). We call the
//!   shared binariser with `chroma_format_idc = 0` so the chroma bins are
//!   skipped, then apply the same 4 bits to each plane's residual gate.
//! * Per-plane QPs come from `chroma_qp` with the PPS's two chroma
//!   `qp_index_offset`s; scaling-list slots are 0 (Y) / 1 (Cb) / 2 (Cr)
//!   for intra.

use oxideav_core::{Error, Result};

use crate::cabac::binarize;
use crate::cabac::context::CabacContext;
use crate::cabac::engine::CabacDecoder;
use crate::cabac::mb::{
    cbp_luma_ctx_idx_inc, decode_luma_8x8_residual_in_place_plane,
    decode_residual_block_in_place_plane, transform_size_8x8_flag_ctx_idx_inc, ResidualPlane,
};
use crate::cabac::residual::{BlockCat, CbfNeighbours};
use crate::cabac::tables::{
    CTX_IDX_CODED_BLOCK_PATTERN_LUMA, CTX_IDX_MB_QP_DELTA, CTX_IDX_MB_TYPE_I,
    CTX_IDX_PREV_INTRA4X4_PRED_MODE_FLAG, CTX_IDX_REM_INTRA4X4_PRED_MODE,
    CTX_IDX_TRANSFORM_SIZE_8X8_FLAG,
};
use crate::intra_pred::{
    predict_intra_16x16, predict_intra_4x4, Intra16x16Mode, Intra16x16Neighbours, Intra4x4Mode,
    Intra4x4Neighbours,
};
use crate::mb::{
    collect_intra16x16_neighbours, collect_intra4x4_neighbours, predict_intra4x4_mode_with,
    LUMA_BLOCK_RASTER,
};
use crate::mb_444::{self as mb444, Plane};
use crate::mb_type::{decode_i_slice_mb_type, IMbType};
use crate::picture::{MbInfo, Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::SliceHeader;
use crate::sps::Sps;
use crate::transform::{
    chroma_qp, dequantize_4x4_scaled, dequantize_8x8_scaled, idct_4x4, idct_8x8,
    inv_hadamard_4x4_dc_scaled,
};

/// Decode one I-slice macroblock under ChromaArrayType = 3.
#[allow(clippy::too_many_arguments)]
pub fn decode_i_mb_cabac_444(
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
    debug_assert_eq!(sps.chroma_format_idc, 3);
    let mb_type_inc = mb_type_i_ctx_idx_inc(pic, mb_x, mb_y);
    let mb_type_raw = {
        let slice = &mut ctxs[CTX_IDX_MB_TYPE_I..CTX_IDX_MB_TYPE_I + 8];
        binarize::decode_mb_type_i(d, slice, mb_type_inc)?
    };
    let imb = decode_i_slice_mb_type(mb_type_raw)
        .ok_or_else(|| Error::invalid(format!("h264 cabac 4:4:4: bad I mb_type {mb_type_raw}")))?;
    decode_intra_mb_given_imb_cabac_444(d, ctxs, sh, sps, pps, mb_x, mb_y, pic, prev_qp, imb)
}

/// Decode an intra macroblock within a 4:4:4 CABAC slice whose `mb_type`
/// has already been parsed. Used by the P-slice path for intra-in-P
/// macroblocks.
#[allow(clippy::too_many_arguments)]
pub fn decode_intra_in_p_cabac_444(
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
    decode_intra_mb_given_imb_cabac_444(d, ctxs, sh, sps, pps, mb_x, mb_y, pic, prev_qp, imb)
}

#[allow(clippy::too_many_arguments)]
fn decode_intra_mb_given_imb_cabac_444(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    _sh: &SliceHeader,
    _sps: &Sps,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    prev_qp: &mut i32,
    imb: IMbType,
) -> Result<()> {
    if matches!(imb, IMbType::IPcm) {
        return Err(Error::unsupported("h264: CABAC 4:4:4 I_PCM not yet wired"));
    }

    let mut transform_8x8 = false;
    let mut intra4x4_modes = [INTRA_DC_FAKE; 16];
    if matches!(imb, IMbType::INxN) {
        if pps.transform_8x8_mode_flag {
            let inc = transform_size_8x8_flag_ctx_idx_inc(pic, mb_x, mb_y);
            let slice =
                &mut ctxs[CTX_IDX_TRANSFORM_SIZE_8X8_FLAG..CTX_IDX_TRANSFORM_SIZE_8X8_FLAG + 3];
            transform_8x8 = binarize::decode_transform_size_8x8_flag(d, slice, inc)?;
        }
        if transform_8x8 {
            return Err(Error::unsupported(
                "h264: 4:4:4 CABAC I_NxN with 8×8 transform not yet wired — 4×4 transform only",
            ));
        }
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
                let slice =
                    &mut ctxs[CTX_IDX_REM_INTRA4X4_PRED_MODE..CTX_IDX_REM_INTRA4X4_PRED_MODE + 1];
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

    // §7.4.5 / Table 9-4(a) in CABAC — 4:4:4 I_NxN emits only 4 CBP
    // luma bits; no chroma CBP bits. Reuse the shared CBP binariser with
    // `chroma_format_idc = 0` so it skips the chroma bins. `cbp_luma` is
    // applied to every plane's 8×8 residual gate. I_16×16 bakes the CBP
    // into mb_type; 4:4:4 discards the chroma 2 bits there and uses
    // `cbp_luma` for all three planes.
    let cbp_luma = match imb {
        IMbType::INxN => {
            let slice =
                &mut ctxs[CTX_IDX_CODED_BLOCK_PATTERN_LUMA..CTX_IDX_CODED_BLOCK_PATTERN_LUMA + 12];
            let cbp = binarize::decode_coded_block_pattern(
                d,
                slice,
                0, // chroma_format_idc = 0 — skip chroma bins
                |i, decoded| cbp_luma_ctx_idx_inc(pic, mb_x, mb_y, i, decoded),
                |_bin| 0u8,
            )?;
            (cbp & 0x0F) as u8
        }
        IMbType::I16x16 { cbp_luma, .. } => cbp_luma,
        IMbType::IPcm => unreachable!(),
    };

    let needs_qp_delta = matches!(imb, IMbType::I16x16 { .. }) || cbp_luma != 0;
    if needs_qp_delta {
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
        *info = MbInfo {
            qp_y,
            coded: true,
            intra: true,
            intra4x4_pred_mode: intra4x4_modes,
            cb_intra4x4_pred_mode: intra4x4_modes,
            cr_intra4x4_pred_mode: intra4x4_modes,
            intra_chroma_pred_mode: 0,
            mb_type_i: Some(imb),
            transform_8x8,
            cbp_luma,
            cbp_chroma: 0,
            ..Default::default()
        };
    }

    match imb {
        IMbType::I16x16 {
            intra16x16_pred_mode,
            ..
        } => {
            for (plane, qp) in [(Plane::Y, qp_y), (Plane::Cb, qp_cb), (Plane::Cr, qp_cr)] {
                decode_plane_intra_16x16_cabac(
                    d,
                    ctxs,
                    mb_x,
                    mb_y,
                    pic,
                    plane,
                    intra16x16_pred_mode,
                    cbp_luma,
                    qp,
                )?;
            }
        }
        IMbType::INxN => {
            for (plane, qp) in [(Plane::Y, qp_y), (Plane::Cb, qp_cb), (Plane::Cr, qp_cr)] {
                decode_plane_intra_nxn_cabac(
                    d,
                    ctxs,
                    mb_x,
                    mb_y,
                    pic,
                    plane,
                    &intra4x4_modes,
                    cbp_luma,
                    qp,
                )?;
            }
        }
        IMbType::IPcm => unreachable!(),
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Per-plane residual pipelines — luma-shape for every colour component.
// ---------------------------------------------------------------------------

/// §9.3.3.1.1.9 Table 9-42 — translate a 4:4:4 [`Plane`] into the
/// CABAC residual bank selector so the Cb / Cr streams use their own
/// context sets (cats 6..=8 / 10..=12 for the 4×4-shaped helper,
/// 9 / 13 for the 8×8 helper).
fn residual_plane_for(plane: Plane) -> ResidualPlane {
    match plane {
        Plane::Y => ResidualPlane::Luma,
        Plane::Cb => ResidualPlane::Cb444,
        Plane::Cr => ResidualPlane::Cr444,
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_plane_intra_nxn_cabac(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    plane: Plane,
    modes: &[u8; 16],
    cbp_luma: u8,
    qp: i32,
) -> Result<()> {
    let scale_idx = plane.as_idx();
    let stride = pic.luma_stride();
    let lo_mb = pic.luma_off(mb_x, mb_y);

    for blk in 0..16usize {
        let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
        let mode_v = modes[br_row * 4 + br_col];
        let mode = Intra4x4Mode::from_u8(mode_v).ok_or_else(|| {
            Error::invalid(format!("h264 cabac 4:4:4: invalid intra4x4 mode {mode_v}"))
        })?;
        let neigh = collect_intra4x4_neighbours_plane(pic, mb_x, mb_y, plane, br_row, br_col);
        let mut pred = [0u8; 16];
        predict_intra_4x4(&mut pred, mode, &neigh);

        let cbp_bit_idx = (br_row / 2) * 2 + (br_col / 2);
        let has_residual = (cbp_luma >> cbp_bit_idx) & 1 != 0;

        let mut residual = [0i32; 16];
        let mut total_coeff = 0u32;
        if has_residual {
            let neighbours = cbf_neighbours_plane(pic, mb_x, mb_y, plane, br_row, br_col);
            let coeffs = decode_residual_block_in_place_plane(
                d,
                ctxs,
                BlockCat::Luma4x4,
                &neighbours,
                16,
                true,
                residual_plane_for(plane),
            )?;
            residual = coeffs;
            total_coeff = coeffs.iter().filter(|&&v| v != 0).count() as u32;
            let scale = *pic.scaling_lists.matrix_4x4(scale_idx);
            dequantize_4x4_scaled(&mut residual, qp, &scale);
            idct_4x4(&mut residual);
        }
        {
            let (buf, _) = mb444::plane_samples_mut(pic, plane);
            let lo = lo_mb + br_row * 4 * stride + br_col * 4;
            for r in 0..4 {
                for c in 0..4 {
                    let v = pred[r * 4 + c] as i32 + residual[r * 4 + c];
                    buf[lo + r * stride + c] = v.clamp(0, 255) as u8;
                }
            }
        }
        let info = pic.mb_info_mut(mb_x, mb_y);
        mb444::plane_nc_at_mut(info, plane)[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_plane_intra_16x16_cabac(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    plane: Plane,
    intra16x16_pred_mode: u8,
    cbp_luma: u8,
    qp: i32,
) -> Result<()> {
    let scale_idx = plane.as_idx();
    let mode = Intra16x16Mode::from_u8(intra16x16_pred_mode).ok_or_else(|| {
        Error::invalid(format!(
            "h264 cabac 4:4:4: invalid intra16x16 mode {intra16x16_pred_mode}"
        ))
    })?;
    let neigh = collect_intra16x16_neighbours_plane(pic, mb_x, mb_y, plane);
    let mut pred = [0u8; 256];
    predict_intra_16x16(&mut pred, mode, &neigh);

    let dc_neigh = plane_luma16x16_dc_cbf_neighbours(pic, mb_x, mb_y, plane);
    let dc_coeffs = decode_residual_block_in_place_plane(
        d,
        ctxs,
        BlockCat::Luma16x16Dc,
        &dc_neigh,
        16,
        true,
        residual_plane_for(plane),
    )?;
    let mut dc = dc_coeffs;
    match plane {
        Plane::Y => {
            pic.mb_info_mut(mb_x, mb_y).luma16x16_dc_cbf = dc_coeffs.iter().any(|&v| v != 0);
        }
        Plane::Cb => {
            pic.mb_info_mut(mb_x, mb_y).cb_luma16x16_dc_cbf = dc_coeffs.iter().any(|&v| v != 0);
        }
        Plane::Cr => {
            pic.mb_info_mut(mb_x, mb_y).cr_luma16x16_dc_cbf = dc_coeffs.iter().any(|&v| v != 0);
        }
    }
    let w_dc = pic.scaling_lists.matrix_4x4(scale_idx)[0];
    inv_hadamard_4x4_dc_scaled(&mut dc, qp, w_dc);

    let stride = pic.luma_stride();
    let lo_mb = pic.luma_off(mb_x, mb_y);
    for blk in 0..16usize {
        let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
        let mut residual = [0i32; 16];
        let mut total_coeff = 0u32;
        if cbp_luma != 0 {
            let neighbours = cbf_neighbours_plane(pic, mb_x, mb_y, plane, br_row, br_col);
            let ac = decode_residual_block_in_place_plane(
                d,
                ctxs,
                BlockCat::Luma16x16Ac,
                &neighbours,
                15,
                true,
                residual_plane_for(plane),
            )?;
            residual = ac;
            total_coeff = ac.iter().filter(|&&v| v != 0).count() as u32;
            let scale = *pic.scaling_lists.matrix_4x4(scale_idx);
            dequantize_4x4_scaled(&mut residual, qp, &scale);
        }
        residual[0] = dc[br_row * 4 + br_col];
        idct_4x4(&mut residual);

        {
            let (buf, _) = mb444::plane_samples_mut(pic, plane);
            let lo = lo_mb + br_row * 4 * stride + br_col * 4;
            for r in 0..4 {
                for c in 0..4 {
                    let v =
                        pred[(br_row * 4 + r) * 16 + (br_col * 4 + c)] as i32 + residual[r * 4 + c];
                    buf[lo + r * stride + c] = v.clamp(0, 255) as u8;
                }
            }
        }
        let info = pic.mb_info_mut(mb_x, mb_y);
        mb444::plane_nc_at_mut(info, plane)[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 8×8 residual — reused for inter MBs (P/B) with transform_size_8x8_flag = 1.
// ---------------------------------------------------------------------------

/// Decode one 8×8 residual block for `plane` through the shared CABAC
/// context banks. Returns unit; on exit the reconstructed 8×8 tile has
/// been added to `pic` and the MB's per-4×4 nC cache has been populated.
///
/// Currently unused — reserved for the follow-up P/B CABAC 4:4:4 wiring
/// (this first pass only exercises intra-in-P/B + Skip macroblocks).
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn decode_inter_residual_plane_8x8(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    plane: Plane,
    cbp_luma: u8,
    qp: i32,
) -> Result<()> {
    let scale_idx = match plane {
        Plane::Y => 1usize,              // Inter-Y 8×8
        Plane::Cb | Plane::Cr => 1usize, // fall back to inter-Y 8×8 list
    };
    let stride = pic.luma_stride();
    let lo_mb = pic.luma_off(mb_x, mb_y);

    for blk8 in 0..4usize {
        if (cbp_luma >> blk8) & 1 == 0 {
            continue;
        }
        let br8_row = blk8 >> 1;
        let br8_col = blk8 & 1;
        let r0 = br8_row * 2;
        let c0 = br8_col * 2;
        let neighbours = cbf_neighbours_plane(pic, mb_x, mb_y, plane, r0, c0);
        let coeffs = decode_luma_8x8_residual_in_place_plane(
            d,
            ctxs,
            &neighbours,
            false,
            residual_plane_for(plane),
        )?;
        let mut residual = coeffs;
        let nnz = residual.iter().filter(|&&v| v != 0).count() as u16;
        let scale = *pic.scaling_lists.matrix_8x8(scale_idx);
        dequantize_8x8_scaled(&mut residual, qp, &scale);
        idct_8x8(&mut residual);
        {
            let info = pic.mb_info_mut(mb_x, mb_y);
            let nc_arr = mb444::plane_nc_at_mut(info, plane);
            let byte = nnz.min(16) as u8;
            nc_arr[r0 * 4 + c0] = byte;
            nc_arr[r0 * 4 + c0 + 1] = byte;
            nc_arr[(r0 + 1) * 4 + c0] = byte;
            nc_arr[(r0 + 1) * 4 + c0 + 1] = byte;
        }
        let (buf, _) = mb444::plane_samples_mut(pic, plane);
        let lo = lo_mb + (br8_row * 8) * stride + br8_col * 8;
        for r in 0..8 {
            for c in 0..8 {
                let base = buf[lo + r * stride + c] as i32;
                let v = base + residual[r * 8 + c];
                buf[lo + r * stride + c] = v.clamp(0, 255) as u8;
            }
        }
    }
    Ok(())
}

/// Decode a 4×4-shaped inter residual for `plane`. Mirrors the 4:2:0
/// `decode_inter_residual_luma` but writes into the selected colour
/// plane and uses the scaling-list slot for that plane (3 = Inter-Y,
/// 4 = Inter-Cb, 5 = Inter-Cr).
///
/// Currently unused — reserved for the follow-up P/B CABAC 4:4:4 wiring
/// (this first pass only exercises intra-in-P/B + Skip macroblocks).
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn decode_inter_residual_plane_4x4(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    plane: Plane,
    cbp_luma: u8,
    qp: i32,
) -> Result<()> {
    let scale_idx = match plane {
        Plane::Y => 3usize,
        Plane::Cb => 4usize,
        Plane::Cr => 5usize,
    };
    let stride = pic.luma_stride();
    let lo_mb = pic.luma_off(mb_x, mb_y);
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
            residual_plane_for(plane),
        )?;
        let mut residual = coeffs;
        let total_coeff = coeffs.iter().filter(|&&v| v != 0).count() as u32;
        let scale = *pic.scaling_lists.matrix_4x4(scale_idx);
        dequantize_4x4_scaled(&mut residual, qp, &scale);
        idct_4x4(&mut residual);
        let (buf, _) = mb444::plane_samples_mut(pic, plane);
        let lo = lo_mb + br_row * 4 * stride + br_col * 4;
        for r in 0..4 {
            for c in 0..4 {
                let base = buf[lo + r * stride + c] as i32;
                buf[lo + r * stride + c] = (base + residual[r * 4 + c]).clamp(0, 255) as u8;
            }
        }
        let info = pic.mb_info_mut(mb_x, mb_y);
        mb444::plane_nc_at_mut(info, plane)[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Neighbour helpers for 4:4:4: same geometry as luma but keyed on the
// plane's own nC cache.
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

fn plane_luma16x16_dc_cbf_neighbours(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    plane: Plane,
) -> CbfNeighbours {
    let flag_for = |info: &MbInfo| match plane {
        Plane::Y => info.luma16x16_dc_cbf,
        Plane::Cb => info.cb_luma16x16_dc_cbf,
        Plane::Cr => info.cr_luma16x16_dc_cbf,
    };
    let probe = |mx: u32, my: u32| -> Option<bool> {
        let info = pic.mb_info_at(mx, my);
        if !info.coded {
            return None;
        }
        Some(flag_for(info))
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

fn collect_intra4x4_neighbours_plane(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    plane: Plane,
    br_row: usize,
    br_col: usize,
) -> Intra4x4Neighbours {
    if plane == Plane::Y {
        return collect_intra4x4_neighbours(pic, mb_x, mb_y, br_row, br_col);
    }
    let stride = pic.luma_stride();
    let (buf, _) = plane_samples(pic, plane);
    let lo_mb = pic.luma_off(mb_x, mb_y);
    let blk_tl = lo_mb + br_row * 4 * stride + br_col * 4;

    let top_avail = br_row > 0 || mb_y > 0;
    let mut top = [0u8; 8];
    if top_avail {
        let row_off = blk_tl - stride;
        for i in 0..4 {
            top[i] = buf[row_off + i];
        }
        // §6.4.10 — top-right availability matches the luma rules (the
        // 4:4:4 chroma planes decode identically to luma, so the shape of
        // Figure 6-12 / Table 8-7 is the same).
        let tr_avail = crate::mb::top_right_available_4x4(mb_x, mb_y, br_row, br_col, pic);
        if tr_avail {
            for i in 0..4 {
                top[4 + i] = buf[row_off + 4 + i];
            }
        } else {
            for i in 0..4 {
                top[4 + i] = top[3];
            }
        }
    }

    let left_avail = br_col > 0 || mb_x > 0;
    let mut left = [0u8; 4];
    if left_avail {
        for i in 0..4 {
            left[i] = buf[blk_tl + i * stride - 1];
        }
    }

    let tl_avail = top_avail && left_avail;
    let top_left = if tl_avail {
        buf[blk_tl - stride - 1]
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

fn collect_intra16x16_neighbours_plane(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    plane: Plane,
) -> Intra16x16Neighbours {
    if plane == Plane::Y {
        return collect_intra16x16_neighbours(pic, mb_x, mb_y);
    }
    let stride = pic.luma_stride();
    let (buf, _) = plane_samples(pic, plane);
    let lo_mb = pic.luma_off(mb_x, mb_y);
    let top_avail = mb_y > 0;
    let mut top = [0u8; 16];
    if top_avail {
        let off = lo_mb - stride;
        for i in 0..16 {
            top[i] = buf[off + i];
        }
    }
    let left_avail = mb_x > 0;
    let mut left = [0u8; 16];
    if left_avail {
        for i in 0..16 {
            left[i] = buf[lo_mb + i * stride - 1];
        }
    }
    let tl_avail = top_avail && left_avail;
    let top_left = if tl_avail { buf[lo_mb - stride - 1] } else { 0 };
    Intra16x16Neighbours {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
    }
}

fn plane_samples(pic: &Picture, plane: Plane) -> (&[u8], usize) {
    let stride = pic.luma_stride();
    let buf = match plane {
        Plane::Y => pic.y.as_slice(),
        Plane::Cb => pic.cb.as_slice(),
        Plane::Cr => pic.cr.as_slice(),
    };
    (buf, stride)
}

fn mb_type_i_ctx_idx_inc(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
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
