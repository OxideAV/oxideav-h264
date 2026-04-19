//! 10-bit 4:4:4 (ChromaArrayType == 3) I-slice CAVLC macroblock decode.
//!
//! Mirrors [`crate::mb_444`] but writes into the u16 planes on
//! [`crate::picture::Picture`]. The entropy layer and the block-layout
//! logic are identical — only the sample type, dequant intermediates
//! (widened to i64 through the `*_ext` helpers), and the QP clamps
//! (extended to `52 + QpBdOffsetY`) change.
//!
//! Scope: CAVLC I-slice, Intra_4×4 and Intra_16×16. `transform_size_8x8_flag`
//! still returns `Error::Unsupported` on the 4:4:4 10-bit path, same as the
//! 8-bit 4:4:4 module.

use oxideav_core::{Error, Result};

use crate::cavlc::{decode_residual_block, BlockKind};
use crate::golomb::BitReaderExt;
use crate::intra_pred::{Intra16x16Mode, Intra4x4Mode};
use crate::intra_pred_hi::{
    predict_intra_16x16 as pred16_hi, predict_intra_4x4 as pred4_hi, Intra16x16Neighbours16,
    Intra4x4Neighbours16,
};
use crate::mb::{top_right_available_4x4, LUMA_BLOCK_RASTER};
use crate::mb_444::{plane_nc_at_mut, predict_nc_plane, Plane, GOLOMB_TO_INTRA4X4_CBP_GRAY};
use crate::mb_hi::predict_intra4x4_mode_with;
use crate::mb_type::IMbType;
use crate::picture::{MbInfo, Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::SliceHeader;
use crate::sps::Sps;
use crate::transform::{
    chroma_qp_hi, dequantize_4x4_scaled_ext, idct_4x4, inv_hadamard_4x4_dc_scaled_ext,
};
use oxideav_core::bits::BitReader;

/// Top-level entry for a 10-bit 4:4:4 intra macroblock. `imb` has been
/// parsed by the slice loop. Reuses the 8-bit 4:4:4 plumbing for mode
/// / CBP parsing but takes the sample writes through the 10-bit
/// reconstruction path.
pub fn decode_intra_mb_hi_444(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    _sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    prev_qp: &mut i32,
    imb: IMbType,
) -> Result<()> {
    debug_assert_eq!(sps.chroma_format_idc, 3);

    let mut transform_8x8 = false;
    let mut intra4x4_modes = [INTRA_DC_FAKE; 16];
    if matches!(imb, IMbType::INxN) {
        if pps.transform_8x8_mode_flag {
            transform_8x8 = br.read_flag()?;
        }
        if transform_8x8 {
            return Err(Error::unsupported(
                "h264 10bit: 4:4:4 I_NxN with 8×8 transform not yet wired",
            ));
        }
        for blk in 0..16usize {
            let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
            let prev_flag = br.read_flag()?;
            let predicted =
                predict_intra4x4_mode_with(pic, mb_x, mb_y, br_row, br_col, &intra4x4_modes);
            let mode = if prev_flag {
                predicted
            } else {
                let rem = br.read_u32(3)? as u8;
                if rem < predicted {
                    rem
                } else {
                    rem + 1
                }
            };
            intra4x4_modes[br_row * 4 + br_col] = mode;
        }
    }

    // 4:4:4 has no transmitted chroma_pred_mode (§7.4.5) — chroma planes
    // reuse the luma modes.

    let (cbp_luma, _cbp_chroma_from_mb_type) = match imb {
        IMbType::INxN => {
            let cbp_raw = br.read_ue()?;
            if cbp_raw > 15 {
                return Err(Error::invalid(format!(
                    "h264 10bit 4:4:4 mb: I_NxN cbp {cbp_raw} > 15"
                )));
            }
            (GOLOMB_TO_INTRA4X4_CBP_GRAY[cbp_raw as usize], 0u8)
        }
        IMbType::I16x16 {
            cbp_luma,
            cbp_chroma,
            ..
        } => (cbp_luma, cbp_chroma),
        IMbType::IPcm => unreachable!(),
    };

    let needs_qp_delta = matches!(imb, IMbType::I16x16 { .. }) || cbp_luma != 0;
    if needs_qp_delta {
        let dqp = br.read_se()?;
        let qp_bd_offset_y = 6 * sps.bit_depth_luma_minus8 as i32;
        let modulus = 52 + qp_bd_offset_y;
        let mut q = (*prev_qp + dqp + 2 * modulus) % modulus;
        if q > 51 {
            q -= modulus;
        }
        *prev_qp = q;
    }
    let qp_y = *prev_qp;
    let qp_bd_offset_y = 6 * sps.bit_depth_luma_minus8 as i32;
    let qp_bd_offset_c = 6 * sps.bit_depth_chroma_minus8 as i32;
    let qp_y_prime = qp_y + qp_bd_offset_y;

    // Per-plane chroma QP — §8.5.11 with QpBdOffsetC support (the chroma
    // QP lookup runs on QpY + offset, then shifts by QpBdOffsetC for the
    // internal dequant).
    let cb_qpi = (qp_y + pps.chroma_qp_index_offset).clamp(-qp_bd_offset_c, 51 + qp_bd_offset_c);
    let cr_qpi =
        (qp_y + pps.second_chroma_qp_index_offset).clamp(-qp_bd_offset_c, 51 + qp_bd_offset_c);
    let qp_cb_prime = chroma_qp_hi(cb_qpi) + qp_bd_offset_c;
    let qp_cr_prime = chroma_qp_hi(cr_qpi) + qp_bd_offset_c;

    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        *info = MbInfo {
            qp_y,
            coded: true,
            intra: true,
            intra4x4_pred_mode: intra4x4_modes,
            cb_intra4x4_pred_mode: intra4x4_modes,
            cr_intra4x4_pred_mode: intra4x4_modes,
            transform_8x8,
            ..Default::default()
        };
    }

    match imb {
        IMbType::I16x16 {
            intra16x16_pred_mode,
            ..
        } => {
            for plane in [Plane::Y, Plane::Cb, Plane::Cr] {
                let qp_prime = match plane {
                    Plane::Y => qp_y_prime,
                    Plane::Cb => qp_cb_prime,
                    Plane::Cr => qp_cr_prime,
                };
                decode_plane_intra_16x16_hi(
                    br,
                    sps,
                    pic,
                    mb_x,
                    mb_y,
                    plane,
                    intra16x16_pred_mode,
                    cbp_luma,
                    qp_prime,
                )?;
            }
        }
        IMbType::INxN => {
            for plane in [Plane::Y, Plane::Cb, Plane::Cr] {
                let qp_prime = match plane {
                    Plane::Y => qp_y_prime,
                    Plane::Cb => qp_cb_prime,
                    Plane::Cr => qp_cr_prime,
                };
                decode_plane_intra_nxn_hi(
                    br,
                    sps,
                    pic,
                    mb_x,
                    mb_y,
                    plane,
                    &intra4x4_modes,
                    cbp_luma,
                    qp_prime,
                )?;
            }
        }
        IMbType::IPcm => unreachable!(),
    }

    Ok(())
}

fn plane_ref(pic: &Picture, plane: Plane) -> &Vec<u16> {
    match plane {
        Plane::Y => &pic.y16,
        Plane::Cb => &pic.cb16,
        Plane::Cr => &pic.cr16,
    }
}

fn plane_ref_mut(pic: &mut Picture, plane: Plane) -> &mut Vec<u16> {
    match plane {
        Plane::Y => &mut pic.y16,
        Plane::Cb => &mut pic.cb16,
        Plane::Cr => &mut pic.cr16,
    }
}

fn collect_intra4x4_neighbours_plane_hi(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    plane: Plane,
    br_row: usize,
    br_col: usize,
) -> Intra4x4Neighbours16 {
    let row_stride = pic.luma_row_stride_for_at(mb_x, mb_y);
    let lo_mb = pic.luma_off(mb_x, mb_y);
    let blk_tl = lo_mb + br_row * 4 * row_stride + br_col * 4;
    let buf = plane_ref(pic, plane);

    let top_avail = br_row > 0 || pic.mb_top_available(mb_y);
    let mut top = [0u16; 8];
    if top_avail {
        let row_off = blk_tl - row_stride;
        for i in 0..4 {
            top[i] = buf[row_off + i];
        }
        let tr_avail = top_right_available_4x4(mb_x, mb_y, br_row, br_col, pic);
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
    let mut left = [0u16; 4];
    if left_avail {
        for i in 0..4 {
            left[i] = buf[blk_tl + i * row_stride - 1];
        }
    }

    let tl_avail = top_avail && left_avail;
    let top_left = if tl_avail {
        buf[blk_tl - row_stride - 1]
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

fn collect_intra16x16_neighbours_plane_hi(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    plane: Plane,
) -> Intra16x16Neighbours16 {
    let row_stride = pic.luma_row_stride_for_at(mb_x, mb_y);
    let lo_mb = pic.luma_off(mb_x, mb_y);
    let buf = plane_ref(pic, plane);

    let top_avail = pic.mb_top_available(mb_y);
    let mut top = [0u16; 16];
    if top_avail {
        let off = lo_mb - row_stride;
        for i in 0..16 {
            top[i] = buf[off + i];
        }
    }
    let left_avail = mb_x > 0;
    let mut left = [0u16; 16];
    if left_avail {
        for i in 0..16 {
            left[i] = buf[lo_mb + i * row_stride - 1];
        }
    }
    let tl_avail = top_avail && left_avail;
    let top_left = if tl_avail {
        buf[lo_mb - row_stride - 1]
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

fn decode_plane_intra_nxn_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pic: &mut Picture,
    mb_x: u32,
    mb_y: u32,
    plane: Plane,
    modes: &[u8; 16],
    cbp_luma: u8,
    qp_prime: i32,
) -> Result<()> {
    let scale_idx = plane.as_idx();
    let bit_depth = (sps.bit_depth_luma_minus8 + 8) as u8;
    let max_sample: i32 = (1i32 << bit_depth) - 1;

    let lo_mb = pic.luma_off(mb_x, mb_y);
    let row_step = pic.luma_row_stride_for_at(mb_x, mb_y);

    for blk in 0..16usize {
        let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
        let mode_v = modes[br_row * 4 + br_col];
        let mode = Intra4x4Mode::from_u8(mode_v).ok_or_else(|| {
            Error::invalid(format!(
                "h264 10bit 4:4:4 mb: invalid intra4x4 mode {mode_v}"
            ))
        })?;
        let neigh = collect_intra4x4_neighbours_plane_hi(pic, mb_x, mb_y, plane, br_row, br_col);
        let mut pred = [0u16; 16];
        pred4_hi(&mut pred, mode, &neigh, bit_depth);

        let cbp_bit_idx = (br_row / 2) * 2 + (br_col / 2);
        let has_residual = (cbp_luma >> cbp_bit_idx) & 1 != 0;

        let mut residual = [0i32; 16];
        let mut total_coeff = 0u32;
        if has_residual {
            let nc = predict_nc_plane(pic, mb_x, mb_y, plane, br_row, br_col);
            let blk = decode_residual_block(br, nc, BlockKind::Luma4x4)?;
            total_coeff = blk.total_coeff;
            residual = blk.coeffs;
            let scale = *pic.scaling_lists.matrix_4x4(scale_idx);
            dequantize_4x4_scaled_ext(&mut residual, qp_prime, &scale);
            idct_4x4(&mut residual);
        }
        {
            let buf = plane_ref_mut(pic, plane);
            let lo = lo_mb + br_row * 4 * row_step + br_col * 4;
            for r in 0..4 {
                for c in 0..4 {
                    let v = pred[r * 4 + c] as i32 + residual[r * 4 + c];
                    buf[lo + r * row_step + c] = v.clamp(0, max_sample) as u16;
                }
            }
        }
        let info = pic.mb_info_mut(mb_x, mb_y);
        plane_nc_at_mut(info, plane)[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}

fn decode_plane_intra_16x16_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pic: &mut Picture,
    mb_x: u32,
    mb_y: u32,
    plane: Plane,
    intra16x16_pred_mode: u8,
    cbp_luma: u8,
    qp_prime: i32,
) -> Result<()> {
    let scale_idx = plane.as_idx();
    let bit_depth = (sps.bit_depth_luma_minus8 + 8) as u8;
    let max_sample: i32 = (1i32 << bit_depth) - 1;

    let mode = Intra16x16Mode::from_u8(intra16x16_pred_mode).ok_or_else(|| {
        Error::invalid(format!(
            "h264 10bit 4:4:4 mb: invalid intra16x16 mode {intra16x16_pred_mode}"
        ))
    })?;
    let neigh = collect_intra16x16_neighbours_plane_hi(pic, mb_x, mb_y, plane);
    let mut pred = [0u16; 256];
    pred16_hi(&mut pred, mode, &neigh, bit_depth);

    let nc_dc = predict_nc_plane(pic, mb_x, mb_y, plane, 0, 0);
    let dc_block = decode_residual_block(br, nc_dc, BlockKind::Luma16x16Dc)?;
    let mut dc = dc_block.coeffs;
    let w_dc = pic.scaling_lists.matrix_4x4(scale_idx)[0];
    inv_hadamard_4x4_dc_scaled_ext(&mut dc, qp_prime, w_dc);

    let lo_mb = pic.luma_off(mb_x, mb_y);
    let row_step = pic.luma_row_stride_for_at(mb_x, mb_y);
    for blk in 0..16usize {
        let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
        let mut residual = [0i32; 16];
        let mut total_coeff = 0u32;
        if cbp_luma != 0 {
            let nc = predict_nc_plane(pic, mb_x, mb_y, plane, br_row, br_col);
            let ac = decode_residual_block(br, nc, BlockKind::Luma16x16Ac)?;
            total_coeff = ac.total_coeff;
            residual = ac.coeffs;
            let scale = *pic.scaling_lists.matrix_4x4(scale_idx);
            dequantize_4x4_scaled_ext(&mut residual, qp_prime, &scale);
        }
        residual[0] = dc[br_row * 4 + br_col];
        idct_4x4(&mut residual);

        {
            let buf = plane_ref_mut(pic, plane);
            let lo = lo_mb + br_row * 4 * row_step + br_col * 4;
            for r in 0..4 {
                for c in 0..4 {
                    let v =
                        pred[(br_row * 4 + r) * 16 + (br_col * 4 + c)] as i32 + residual[r * 4 + c];
                    buf[lo + r * row_step + c] = v.clamp(0, max_sample) as u16;
                }
            }
        }
        let info = pic.mb_info_mut(mb_x, mb_y);
        plane_nc_at_mut(info, plane)[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}
