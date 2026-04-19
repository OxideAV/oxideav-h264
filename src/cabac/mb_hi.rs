//! 10-bit CABAC I-slice macroblock decode — ITU-T H.264 §7.3.5 / §9.3
//! with the bit-depth extensions from §7.4.2.1.1 and §8.5.12.1.
//!
//! Mirrors [`crate::cabac::mb::decode_i_mb_cabac`] but writes the
//! reconstructed samples into `Picture::y16 / cb16 / cr16` and routes the
//! residual dequant through the `*_ext` i64-widened helpers so the
//! extended `QpY + QpBdOffsetY` range (up to 63 at 10-bit) doesn't overflow.
//!
//! The CABAC entropy layer is bit-depth agnostic — context state,
//! binarisation, and residual decode all run unchanged. What differs:
//!
//! * Intra prediction + sample write target `u16` planes and clip to
//!   `(1 << BitDepth) - 1`.
//! * Residual dequant uses `dequantize_4x4_scaled_ext`,
//!   `inv_hadamard_4x4_dc_scaled_ext`, and
//!   `inv_hadamard_2x2_chroma_dc_scaled_ext` with `qp + QpBdOffset`.
//! * `mb_qp_delta` wraps modulo `52 + QpBdOffsetY` (§7.4.5 extended),
//!   keeping `prev_qp` in the canonical `[-QpBdOffsetY, 51]` window.
//!
//! Scope: 4:2:0, CABAC, I-slice (I_16x16 and I_NxN with 4×4 transform).
//! Intra_8×8 / transform_size_8x8_flag = 1 and I_PCM at 10-bit still
//! return `Error::Unsupported`. The 10-bit CABAC P/B paths now live in
//! [`crate::cabac::p_mb_hi`] / [`crate::cabac::b_mb_hi`] and dispatch
//! through [`decode_intra_mb_given_imb_cabac_hi`] for intra-in-P/B.

use oxideav_core::{Error, Result};

use crate::cabac::binarize;
use crate::cabac::context::CabacContext;
use crate::cabac::engine::CabacDecoder;
use crate::cabac::mb::{
    cbp_chroma_ctx_idx_inc_ac, cbp_chroma_ctx_idx_inc_any, cbp_luma_ctx_idx_inc,
    decode_residual_block_in_place, transform_size_8x8_flag_ctx_idx_inc,
};
use crate::cabac::residual::{BlockCat, CbfNeighbours};
use crate::cabac::tables::{
    CTX_IDX_CODED_BLOCK_PATTERN_LUMA, CTX_IDX_INTRA_CHROMA_PRED_MODE, CTX_IDX_MB_QP_DELTA,
    CTX_IDX_MB_TYPE_I, CTX_IDX_PREV_INTRA4X4_PRED_MODE_FLAG, CTX_IDX_REM_INTRA4X4_PRED_MODE,
    CTX_IDX_TRANSFORM_SIZE_8X8_FLAG,
};
use crate::intra_pred::{Intra16x16Mode, Intra4x4Mode, IntraChromaMode};
use crate::intra_pred_hi::{
    predict_intra_16x16 as pred16_hi, predict_intra_4x4 as pred4_hi,
    predict_intra_chroma as predc_hi, Intra16x16Neighbours16, Intra4x4Neighbours16,
    IntraChromaNeighbours16,
};
use crate::mb::LUMA_BLOCK_RASTER;
use crate::mb_type::{decode_i_slice_mb_type, IMbType};
use crate::picture::{MbInfo, Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::SliceHeader;
use crate::sps::Sps;
use crate::transform::{
    chroma_qp_hi, dequantize_4x4_scaled_ext, idct_4x4, inv_hadamard_2x2_chroma_dc_scaled_ext,
    inv_hadamard_4x4_dc_scaled_ext,
};

/// Helper mirroring [`crate::cabac::mb::mb_type_i_ctx_idx_inc`] but local to
/// the 10-bit path so the 8-bit helper can stay private to `cabac::mb`.
fn mb_type_i_ctx_idx_inc_hi(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
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

fn intra_chroma_pred_mode_ctx_idx_inc_hi(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
    let check = |m: &MbInfo| m.coded && m.intra && m.intra_chroma_pred_mode != 0;
    let a = mb_x > 0 && check(pic.mb_info_at(mb_x - 1, mb_y));
    let b = mb_y > 0 && check(pic.mb_info_at(mb_x, mb_y - 1));
    (a as u8) + (b as u8)
}

/// Decode a single I-slice macroblock at `(mb_x, mb_y)` via CABAC, writing
/// the reconstructed samples into the u16 planes on `pic`.
#[allow(clippy::too_many_arguments)]
pub fn decode_i_mb_cabac_hi(
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
    let mb_type_inc = mb_type_i_ctx_idx_inc_hi(pic, mb_x, mb_y);
    let mb_type_raw = {
        let slice = &mut ctxs[CTX_IDX_MB_TYPE_I..CTX_IDX_MB_TYPE_I + 8];
        binarize::decode_mb_type_i(d, slice, mb_type_inc)?
    };
    let imb = decode_i_slice_mb_type(mb_type_raw).ok_or_else(|| {
        Error::invalid(format!("h264 cabac mb (hi): bad I mb_type {mb_type_raw}"))
    })?;
    decode_intra_mb_given_imb_cabac_hi(d, ctxs, sh, sps, pps, mb_x, mb_y, pic, prev_qp, imb)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_intra_mb_given_imb_cabac_hi(
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
        return Err(Error::unsupported("h264 10bit cabac: I_PCM not yet wired"));
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
                "h264 10bit cabac: Intra_8x8 (transform_size_8x8_flag=1) not yet wired",
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

    let chroma_mode_val = if sps.chroma_format_idc != 0 {
        let inc = intra_chroma_pred_mode_ctx_idx_inc_hi(pic, mb_x, mb_y);
        let slice = &mut ctxs[CTX_IDX_INTRA_CHROMA_PRED_MODE..CTX_IDX_INTRA_CHROMA_PRED_MODE + 4];
        binarize::decode_intra_chroma_pred_mode(d, slice, inc)?
    } else {
        0
    };
    let chroma_pred_mode = IntraChromaMode::from_u8(chroma_mode_val as u8).ok_or_else(|| {
        Error::invalid(format!(
            "h264 cabac mb (hi): bad intra_chroma_pred_mode {chroma_mode_val}"
        ))
    })?;

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

    let needs_qp_delta =
        matches!(imb, IMbType::I16x16 { .. }) || (cbp_luma != 0 || cbp_chroma != 0);
    if needs_qp_delta {
        let inc = if pic.last_mb_qp_delta_was_nonzero {
            1u8
        } else {
            0u8
        };
        let slice = &mut ctxs[CTX_IDX_MB_QP_DELTA..CTX_IDX_MB_QP_DELTA + 4];
        let dqp = binarize::decode_mb_qp_delta(d, slice, inc)?;
        pic.last_mb_qp_delta_was_nonzero = dqp != 0;
        let qp_bd_offset_y = 6 * sps.bit_depth_luma_minus8 as i32;
        let modulus = 52 + qp_bd_offset_y;
        // §7.4.5 extended — same wrap as the CAVLC 10-bit path.
        let mut q = (*prev_qp + dqp + 2 * modulus) % modulus;
        if q > 51 {
            q -= modulus;
        }
        *prev_qp = q;
    } else {
        pic.last_mb_qp_delta_was_nonzero = false;
    }
    let qp_y = *prev_qp;
    let qp_bd_offset_y = 6 * sps.bit_depth_luma_minus8 as i32;
    let qp_y_prime = qp_y + qp_bd_offset_y;

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

    match imb {
        IMbType::I16x16 {
            intra16x16_pred_mode,
            ..
        } => decode_luma_intra_16x16_hi(
            d,
            ctxs,
            sps,
            mb_x,
            mb_y,
            pic,
            intra16x16_pred_mode,
            cbp_luma,
            qp_y_prime,
        )?,
        IMbType::INxN => decode_luma_intra_nxn_hi(
            d,
            ctxs,
            sps,
            mb_x,
            mb_y,
            pic,
            &intra4x4_modes,
            cbp_luma,
            qp_y_prime,
        )?,
        IMbType::IPcm => unreachable!(),
    }

    decode_chroma_hi(
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

    Ok(())
}

// ---------------------------------------------------------------------------
// Luma reconstruction — I_NxN (4×4 transform only at 10-bit for now).
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_luma_intra_nxn_hi(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sps: &Sps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    modes: &[u8; 16],
    cbp_luma: u8,
    qp_y_prime: i32,
) -> Result<()> {
    let bit_depth = (sps.bit_depth_luma_minus8 + 8) as u8;
    let max_sample = (1i32 << bit_depth) - 1;
    let lstride = pic.luma_stride();
    let lo_mb = pic.luma_off(mb_x, mb_y);

    for blk in 0..16usize {
        let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
        let mode_v = modes[br_row * 4 + br_col];
        let mode = Intra4x4Mode::from_u8(mode_v).ok_or_else(|| {
            Error::invalid(format!(
                "h264 cabac mb (hi): invalid intra4x4 mode {mode_v}"
            ))
        })?;
        let neigh = collect_intra4x4_neighbours_hi(pic, mb_x, mb_y, br_row, br_col);
        let mut pred = [0u16; 16];
        pred4_hi(&mut pred, mode, &neigh, bit_depth);

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
            let scale = *pic.scaling_lists.matrix_4x4(0);
            dequantize_4x4_scaled_ext(&mut residual, qp_y_prime, &scale);
            idct_4x4(&mut residual);
        }
        let lo = lo_mb + br_row * 4 * lstride + br_col * 4;
        for r in 0..4 {
            for c in 0..4 {
                let v = pred[r * 4 + c] as i32 + residual[r * 4 + c];
                pic.y16[lo + r * lstride + c] = v.clamp(0, max_sample) as u16;
            }
        }
        pic.mb_info_mut(mb_x, mb_y).luma_nc[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Luma reconstruction — I_16x16 (Hadamard DC + AC).
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_luma_intra_16x16_hi(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sps: &Sps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    intra16x16_pred_mode: u8,
    cbp_luma: u8,
    qp_y_prime: i32,
) -> Result<()> {
    let bit_depth = (sps.bit_depth_luma_minus8 + 8) as u8;
    let max_sample = (1i32 << bit_depth) - 1;
    let neigh = collect_intra16x16_neighbours_hi(pic, mb_x, mb_y);
    let mut pred = [0u16; 256];
    let mode = Intra16x16Mode::from_u8(intra16x16_pred_mode).ok_or_else(|| {
        Error::invalid(format!(
            "h264 cabac mb (hi): invalid intra16x16 mode {intra16x16_pred_mode}"
        ))
    })?;
    pred16_hi(&mut pred, mode, &neigh, bit_depth);

    let dc_neigh = luma16x16_dc_cbf_neighbours(pic, mb_x, mb_y);
    let dc_coeffs =
        decode_residual_block_in_place(d, ctxs, BlockCat::Luma16x16Dc, &dc_neigh, 16, true)?;
    let mut dc = dc_coeffs;
    pic.mb_info_mut(mb_x, mb_y).luma16x16_dc_cbf = dc_coeffs.iter().any(|&v| v != 0);
    let w_dc = pic.scaling_lists.matrix_4x4(0)[0];
    inv_hadamard_4x4_dc_scaled_ext(&mut dc, qp_y_prime, w_dc);

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
            let scale = *pic.scaling_lists.matrix_4x4(0);
            dequantize_4x4_scaled_ext(&mut residual, qp_y_prime, &scale);
        }
        residual[0] = dc[br_row * 4 + br_col];
        idct_4x4(&mut residual);

        let lo = lo_mb + br_row * 4 * lstride + br_col * 4;
        for r in 0..4 {
            for c in 0..4 {
                let v = pred[(br_row * 4 + r) * 16 + (br_col * 4 + c)] as i32 + residual[r * 4 + c];
                pic.y16[lo + r * lstride + c] = v.clamp(0, max_sample) as u16;
            }
        }
        pic.mb_info_mut(mb_x, mb_y).luma_nc[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Chroma reconstruction — 4:2:0, DC Hadamard + AC.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_chroma_hi(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sps: &Sps,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    chroma_mode: IntraChromaMode,
    cbp_chroma: u8,
    qp_y: i32,
) -> Result<()> {
    let bit_depth_c = (sps.bit_depth_chroma_minus8 + 8) as u8;
    let max_sample = (1i32 << bit_depth_c) - 1;
    let qp_bd_offset_c = 6 * sps.bit_depth_chroma_minus8 as i32;

    // §8.5.11 — chroma QP derivation with QpBdOffset.
    let qpi = (qp_y + pps.chroma_qp_index_offset).clamp(-qp_bd_offset_c, 51 + qp_bd_offset_c);
    let qpc = chroma_qp_hi(qpi);
    let qpc_prime = qpc + qp_bd_offset_c;

    let neigh_cb = collect_chroma_neighbours_hi(pic, mb_x, mb_y, true);
    let neigh_cr = collect_chroma_neighbours_hi(pic, mb_x, mb_y, false);
    let mut pred_cb = [0u16; 64];
    let mut pred_cr = [0u16; 64];
    predc_hi(&mut pred_cb, chroma_mode, &neigh_cb, bit_depth_c);
    predc_hi(&mut pred_cr, chroma_mode, &neigh_cr, bit_depth_c);

    let mut dc_cb = [0i32; 4];
    let mut dc_cr = [0i32; 4];
    if cbp_chroma >= 1 {
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
        let w_cb = pic.scaling_lists.matrix_4x4(1)[0];
        let w_cr = pic.scaling_lists.matrix_4x4(2)[0];
        inv_hadamard_2x2_chroma_dc_scaled_ext(&mut dc_cb, qpc_prime, w_cb);
        inv_hadamard_2x2_chroma_dc_scaled_ext(&mut dc_cr, qpc_prime, w_cr);
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
                let neigh =
                    chroma_ac_cbf_neighbours(pic, mb_x, mb_y, plane_kind, br_row, br_col, &nc_arr);
                let ac =
                    decode_residual_block_in_place(d, ctxs, BlockCat::ChromaAc, &neigh, 15, true)?;
                res = ac;
                total_coeff = ac.iter().filter(|&&v| v != 0).count() as u32;
                let cat = if plane_kind { 1 } else { 2 };
                let scale = *pic.scaling_lists.matrix_4x4(cat);
                dequantize_4x4_scaled_ext(&mut res, qpc_prime, &scale);
            }
            res[0] = dc[(br_row << 1) | br_col];
            idct_4x4(&mut res);
            let off_in_mb = br_row * 4 * cstride + br_col * 4;
            let plane = if plane_kind {
                &mut pic.cb16
            } else {
                &mut pic.cr16
            };
            for r in 0..4 {
                for c in 0..4 {
                    let v = pred[(br_row * 4 + r) * 8 + (br_col * 4 + c)] as i32 + res[r * 4 + c];
                    plane[co + off_in_mb + r * cstride + c] = v.clamp(0, max_sample) as u16;
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

// ---------------------------------------------------------------------------
// Neighbour helpers — all on u16 planes. Mirror the 8-bit CABAC path's
// private helpers (intentionally not reusing `crate::cabac::mb`'s internals
// so that module stays 8-bit-only without leaking a bit-depth-parametric
// shape).
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

fn collect_intra4x4_neighbours_hi(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    br_row: usize,
    br_col: usize,
) -> Intra4x4Neighbours16 {
    let lstride = pic.luma_stride();

    let top_avail = br_row > 0 || mb_y > 0;
    let mut top = [0u16; 8];
    if top_avail {
        let row_y_global = if br_row > 0 {
            (mb_y as usize) * 16 + br_row * 4 - 1
        } else {
            (mb_y as usize) * 16 - 1
        };
        let row_off = row_y_global * lstride;
        for i in 0..4 {
            top[i] = pic.y16[row_off + (mb_x as usize) * 16 + br_col * 4 + i];
        }
        let tr_avail = crate::mb::top_right_available_4x4(mb_x, mb_y, br_row, br_col, pic);
        if tr_avail {
            for i in 0..4 {
                top[4 + i] = pic.y16[row_off + (mb_x as usize) * 16 + br_col * 4 + 4 + i];
            }
        } else {
            for i in 0..4 {
                top[4 + i] = top[3];
            }
        }
    }

    let left_avail = br_col > 0 || mb_x > 0;
    let mut left = [0u16; 4];
    if left_avail {
        let col_x_global: usize = if br_col > 0 {
            (mb_x as usize) * 16 + br_col * 4 - 1
        } else {
            (mb_x as usize) * 16 - 1
        };
        for i in 0..4 {
            let row = (mb_y as usize) * 16 + br_row * 4 + i;
            left[i] = pic.y16[row * lstride + col_x_global];
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
        pic.y16[row_y_global * lstride + col_x_global]
    } else {
        0
    };

    Intra4x4Neighbours16 {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
        top_right_available: false,
    }
}

fn collect_intra16x16_neighbours_hi(pic: &Picture, mb_x: u32, mb_y: u32) -> Intra16x16Neighbours16 {
    let lstride = pic.luma_stride();
    let lo_mb = pic.luma_off(mb_x, mb_y);
    let top_avail = mb_y > 0;
    let mut top = [0u16; 16];
    if top_avail {
        let off = lo_mb - lstride;
        for i in 0..16 {
            top[i] = pic.y16[off + i];
        }
    }
    let left_avail = mb_x > 0;
    let mut left = [0u16; 16];
    if left_avail {
        for i in 0..16 {
            left[i] = pic.y16[lo_mb + i * lstride - 1];
        }
    }
    let tl_avail = top_avail && left_avail;
    let top_left = if tl_avail {
        pic.y16[lo_mb - lstride - 1]
    } else {
        0
    };
    Intra16x16Neighbours16 {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
    }
}

fn collect_chroma_neighbours_hi(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    cb: bool,
) -> IntraChromaNeighbours16 {
    let cstride = pic.chroma_stride();
    let co_mb = pic.chroma_off(mb_x, mb_y);
    let plane = if cb { &pic.cb16 } else { &pic.cr16 };
    let top_avail = mb_y > 0;
    let mut top = [0u16; 8];
    if top_avail {
        let off = co_mb - cstride;
        for i in 0..8 {
            top[i] = plane[off + i];
        }
    }
    let left_avail = mb_x > 0;
    let mut left = [0u16; 8];
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
    IntraChromaNeighbours16 {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
    }
}
