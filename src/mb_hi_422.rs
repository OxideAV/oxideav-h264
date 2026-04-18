//! 10-bit 4:2:2 chroma reconstruction (§8.5.11.2, ChromaArrayType == 2).
//!
//! Mirrors the private `decode_chroma_422` helper in [`crate::mb`] but
//! writes into `Picture::cb16 / cr16`. The CAVLC DC block is a
//! `ChromaDc2x4` stream (8 coefficients, `nC = -2`); per-plane AC is 8 ×
//! `ChromaAc` blocks gated on `cbp_chroma == 2`. The inverse 2×4
//! Hadamard + DC dequant runs through
//! [`crate::transform::inv_hadamard_2x4_chroma_dc_scaled_ext`] so the QP
//! at 10-bit (`QP'_C,DC` can hit 66) stays in the i64 intermediate range.

use oxideav_core::Result;

use crate::bitreader::BitReader;
use crate::cavlc::{decode_residual_block, BlockKind};
use crate::intra_pred::IntraChromaMode;
use crate::intra_pred_hi::{predict_intra_chroma_8x16, IntraChroma8x16Neighbours16};
use crate::picture::{MbInfo, Picture};
use crate::pps::Pps;
use crate::slice::SliceHeader;
use crate::sps::Sps;
use crate::transform::{
    chroma_qp_hi, dequantize_4x4_scaled_ext, idct_4x4, inv_hadamard_2x4_chroma_dc_scaled_ext,
};

/// Decode the two chroma planes of a 10-bit 4:2:2 macroblock. Analogous
/// to the 8-bit [`decode_chroma_422`](crate::mb) but operating on u16
/// samples. Called from the 10-bit CAVLC I-slice (and, when extended,
/// P/B) macroblock decoders.
pub fn decode_chroma_422_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    _sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    chroma_mode: IntraChromaMode,
    cbp_chroma: u8,
    qp_y: i32,
) -> Result<()> {
    let bit_depth_c = (sps.bit_depth_chroma_minus8 + 8) as u8;
    let max_sample: i32 = (1i32 << bit_depth_c) - 1;
    let qp_bd_offset_c = 6 * sps.bit_depth_chroma_minus8 as i32;

    let cb_qpi =
        (qp_y + pps.chroma_qp_index_offset).clamp(-qp_bd_offset_c, 51 + qp_bd_offset_c);
    let cr_qpi = (qp_y + pps.second_chroma_qp_index_offset)
        .clamp(-qp_bd_offset_c, 51 + qp_bd_offset_c);
    let qpc_cb_prime = chroma_qp_hi(cb_qpi) + qp_bd_offset_c;
    let qpc_cr_prime = chroma_qp_hi(cr_qpi) + qp_bd_offset_c;

    let neigh_cb = collect_chroma_422_neighbours_hi(pic, mb_x, mb_y, true);
    let neigh_cr = collect_chroma_422_neighbours_hi(pic, mb_x, mb_y, false);
    let mut pred_cb = [0u16; 128];
    let mut pred_cr = [0u16; 128];
    predict_intra_chroma_8x16(&mut pred_cb, chroma_mode, &neigh_cb, bit_depth_c);
    predict_intra_chroma_8x16(&mut pred_cr, chroma_mode, &neigh_cr, bit_depth_c);

    let mut dc_cb = [0i32; 8];
    let mut dc_cr = [0i32; 8];
    if cbp_chroma >= 1 {
        let blk = decode_residual_block(br, -2, BlockKind::ChromaDc2x4)?;
        for i in 0..8 {
            dc_cb[i] = blk.coeffs[i];
        }
        let blk = decode_residual_block(br, -2, BlockKind::ChromaDc2x4)?;
        for i in 0..8 {
            dc_cr[i] = blk.coeffs[i];
        }
        let w_cb = pic.scaling_lists.matrix_4x4(1)[0];
        let w_cr = pic.scaling_lists.matrix_4x4(2)[0];
        inv_hadamard_2x4_chroma_dc_scaled_ext(&mut dc_cb, qpc_cb_prime, w_cb);
        inv_hadamard_2x4_chroma_dc_scaled_ext(&mut dc_cr, qpc_cr_prime, w_cr);
    }

    let row_step = pic.chroma_row_stride_for_at(mb_x, mb_y);
    let co = pic.chroma_off(mb_x, mb_y);

    for plane_cb in [true, false] {
        let pred = if plane_cb { &pred_cb } else { &pred_cr };
        let dc = if plane_cb { &dc_cb } else { &dc_cr };
        let qpc_prime = if plane_cb { qpc_cb_prime } else { qpc_cr_prime };
        let mut nc_arr = [0u8; 8];
        for blk_row in 0..4usize {
            for blk_col in 0..2usize {
                let blk_idx = blk_row * 2 + blk_col;
                let mut res = [0i32; 16];
                let mut total_coeff = 0u32;
                if cbp_chroma == 2 {
                    let nc =
                        predict_nc_chroma_422_hi(pic, mb_x, mb_y, plane_cb, blk_row, blk_col);
                    let ac = decode_residual_block(br, nc, BlockKind::ChromaAc)?;
                    total_coeff = ac.total_coeff;
                    res = ac.coeffs;
                    let cat = if plane_cb { 1 } else { 2 };
                    let scale = *pic.scaling_lists.matrix_4x4(cat);
                    dequantize_4x4_scaled_ext(&mut res, qpc_prime, &scale);
                }
                res[0] = dc[blk_idx];
                idct_4x4(&mut res);
                let off_in_mb = (blk_row * 4) * row_step + blk_col * 4;
                let plane_buf = if plane_cb { &mut pic.cb16 } else { &mut pic.cr16 };
                for r in 0..4 {
                    for c in 0..4 {
                        let p_idx = (blk_row * 4 + r) * 8 + (blk_col * 4 + c);
                        let v = pred[p_idx] as i32 + res[r * 4 + c];
                        plane_buf[co + off_in_mb + r * row_step + c] =
                            v.clamp(0, max_sample) as u16;
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

fn predict_nc_chroma_422_hi(
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
    } else if let Some(above_y) = pic.mb_above_neighbour(mb_y) {
        let info = pic.mb_info_at(mb_x, above_y);
        if info.coded {
            Some(pick(info, 3 * 2 + blk_col))
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

fn collect_chroma_422_neighbours_hi(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    cb: bool,
) -> IntraChroma8x16Neighbours16 {
    let row_stride = pic.chroma_row_stride_for_at(mb_x, mb_y);
    let co_mb = pic.chroma_off(mb_x, mb_y);
    let plane = if cb { &pic.cb16 } else { &pic.cr16 };
    let top_avail = pic.mb_top_available(mb_y);
    let mut top = [0u16; 8];
    if top_avail {
        let off = co_mb - row_stride;
        for i in 0..8 {
            top[i] = plane[off + i];
        }
    }
    let left_avail = mb_x > 0;
    let mut left = [0u16; 16];
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
    IntraChroma8x16Neighbours16 {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
    }
}
