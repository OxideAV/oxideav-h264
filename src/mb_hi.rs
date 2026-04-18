//! 10-bit I-slice macroblock decode — §7.3.5 / §8.3 / §8.5 with the
//! bit-depth extensions from §7.4.2.1.1 and §8.5.12.1.
//!
//! Mirrors [`crate::mb::decode_i_slice_data`] but writes into the
//! u16 planes on [`crate::picture::Picture`]. The residual decode
//! (CAVLC, bit-depth agnostic) and the core transform math are
//! reused from the 8-bit pipeline; what changes is:
//!
//! * The effective dequant QP is `QP_Y + QpBdOffsetY` (up to 63 at
//!   10-bit), dispatched through the i64-widened `*_ext` variants.
//! * `mb_qp_delta` wraps modulo `52 + QpBdOffsetY`.
//! * Prediction samples live in `u16` and clip to `(1 << BitDepth) - 1`.
//! * Chroma QP table lookup is driven off `QpY + chroma_qp_index_offset`,
//!   then shifted by `QpBdOffsetC` before dequant.
//!
//! Scope: 4:2:0, CAVLC, I-slice (I_NxN 4×4 transform and I_16x16).
//! The 8×8 transform path for 10-bit is not yet wired — an I_NxN MB
//! with `transform_size_8x8_flag = 1` returns `Error::Unsupported`.
//! I_PCM at 10-bit is also not yet wired.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::cavlc::{decode_residual_block, BlockKind};
use crate::intra_pred::{Intra16x16Mode, Intra4x4Mode, IntraChromaMode};
use crate::intra_pred_hi::{
    predict_intra_16x16 as pred16_hi, predict_intra_4x4 as pred4_hi,
    predict_intra_chroma as predc_hi, Intra16x16Neighbours16, Intra4x4Neighbours16,
    IntraChromaNeighbours16,
};
use crate::mb::{predict_nc_luma, LUMA_BLOCK_RASTER};
use crate::mb_type::{decode_i_slice_mb_type, IMbType};
use crate::picture::{MbInfo, Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::SliceHeader;
use crate::sps::Sps;
use crate::tables::decode_cbp_intra;
use crate::transform::{
    chroma_qp_hi, dequantize_4x4_scaled_ext, idct_4x4, inv_hadamard_2x2_chroma_dc_scaled_ext,
    inv_hadamard_4x4_dc_scaled_ext,
};

/// Top-level entry for a 10-bit CAVLC I-slice. Walks every macroblock
/// in raster order. Caller must have already advanced the bit reader
/// past the slice header.
pub fn decode_i_slice_data_hi(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
) -> Result<()> {
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = sps.pic_height_in_map_units();
    let total_mbs = mb_w * mb_h;
    let mut mb_addr = sh.first_mb_in_slice;
    if mb_addr >= total_mbs {
        return Err(Error::invalid("h264 slice: first_mb_in_slice out of range"));
    }
    let qp_bd_offset_y = 6 * sps.bit_depth_luma_minus8 as i32;
    let mut prev_qp = pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta;
    if prev_qp < -qp_bd_offset_y || prev_qp > 51 {
        return Err(Error::invalid(format!(
            "h264 10bit slice: initial QP {prev_qp} out of range [{}..=51]",
            -qp_bd_offset_y
        )));
    }

    while mb_addr < total_mbs {
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        decode_one_mb_hi(br, sps, pps, sh, mb_x, mb_y, pic, &mut prev_qp)?;
        mb_addr += 1;
    }
    Ok(())
}

fn decode_one_mb_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    _sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    prev_qp: &mut i32,
) -> Result<()> {
    let mb_type = br.read_ue()?;
    let imb = decode_i_slice_mb_type(mb_type)
        .ok_or_else(|| Error::invalid(format!("h264 slice: bad I mb_type {mb_type}")))?;
    let _ = _sh;
    if matches!(imb, IMbType::IPcm) {
        return decode_pcm_mb_hi(br, sps, mb_x, mb_y, pic, *prev_qp);
    }

    // Intra4x4 modes (transform_8x8 at 10-bit is scoped out for now).
    let mut transform_8x8 = false;
    let mut intra4x4_modes = [INTRA_DC_FAKE; 16];
    if matches!(imb, IMbType::INxN) {
        if pps.transform_8x8_mode_flag {
            transform_8x8 = br.read_flag()?;
        }
        if transform_8x8 {
            return Err(Error::unsupported(
                "h264 10bit: Intra_8×8 (transform_size_8x8_flag=1) not yet wired",
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

    let chroma_pred_mode = {
        let v = br.read_ue()?;
        if v > 3 {
            return Err(Error::invalid(format!(
                "h264 mb: intra_chroma_pred_mode {v} > 3"
            )));
        }
        IntraChromaMode::from_u8(v as u8).unwrap()
    };

    let (cbp_luma, cbp_chroma) = match imb {
        IMbType::INxN => {
            let cbp_raw = br.read_ue()?;
            decode_cbp_intra(cbp_raw)
                .ok_or_else(|| Error::invalid(format!("h264 mb: bad CBP {cbp_raw}")))?
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
        let dqp = br.read_se()?;
        let qp_bd_offset_y = 6 * sps.bit_depth_luma_minus8 as i32;
        let modulus = 52 + qp_bd_offset_y;
        // §7.4.5 extended: QpY wraps mod (52 + QpBdOffsetY), carried through
        // the nominal range `[-QpBdOffsetY..=51]`.
        let mut q = (*prev_qp + dqp + 2 * modulus) % modulus;
        if q > 51 {
            q -= modulus;
        }
        *prev_qp = q;
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
            br,
            sps,
            mb_x,
            mb_y,
            pic,
            intra16x16_pred_mode,
            cbp_luma,
            qp_y_prime,
        )?,
        IMbType::INxN => decode_luma_intra_nxn_hi(
            br,
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
        br,
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

fn decode_pcm_mb_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    prev_qp: i32,
) -> Result<()> {
    while !br.is_byte_aligned() {
        let _ = br.read_u1()?;
    }
    let bdy = sps.bit_depth_luma_minus8 + 8;
    let bdc = sps.bit_depth_chroma_minus8 + 8;
    let lstride = pic.luma_stride();
    let lo = pic.luma_off(mb_x, mb_y);
    for r in 0..16 {
        for c in 0..16 {
            pic.y16[lo + r * lstride + c] = br.read_u32(bdy)? as u16;
        }
    }
    let cstride = pic.chroma_stride();
    let co = pic.chroma_off(mb_x, mb_y);
    let chroma_rows = pic.mb_height_chroma();
    let chroma_cols = pic.mb_width_chroma();
    for r in 0..chroma_rows {
        for c in 0..chroma_cols {
            pic.cb16[co + r * cstride + c] = br.read_u32(bdc)? as u16;
        }
    }
    for r in 0..chroma_rows {
        for c in 0..chroma_cols {
            pic.cr16[co + r * cstride + c] = br.read_u32(bdc)? as u16;
        }
    }
    let info = pic.mb_info_mut(mb_x, mb_y);
    info.qp_y = prev_qp;
    info.coded = true;
    info.intra = true;
    info.luma_nc = [16; 16];
    info.cb_nc = [16; 16];
    info.cr_nc = [16; 16];
    info.intra4x4_pred_mode = [INTRA_DC_FAKE; 16];
    info.cbp_luma = 0x0F;
    info.cbp_chroma = 2;
    Ok(())
}

fn decode_luma_intra_nxn_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    modes: &[u8; 16],
    cbp_luma: u8,
    qp_y_prime: i32,
) -> Result<()> {
    let bit_depth = (sps.bit_depth_luma_minus8 + 8) as u8;
    let max_sample = (1u16 << bit_depth) - 1;
    let lstride = pic.luma_stride();
    let lo_mb = pic.luma_off(mb_x, mb_y);

    for blk in 0..16usize {
        let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
        let mode_v = modes[br_row * 4 + br_col];
        let mode = Intra4x4Mode::from_u8(mode_v)
            .ok_or_else(|| Error::invalid(format!("h264 mb: invalid intra4x4 mode {mode_v}")))?;
        let neigh = collect_intra4x4_neighbours_hi(pic, mb_x, mb_y, br_row, br_col);
        let mut pred = [0u16; 16];
        pred4_hi(&mut pred, mode, &neigh, bit_depth);

        let cbp_bit_idx = (br_row / 2) * 2 + (br_col / 2);
        let has_residual = (cbp_luma >> cbp_bit_idx) & 1 != 0;

        let mut residual = [0i32; 16];
        let mut total_coeff = 0u32;
        if has_residual {
            let nc = predict_nc_luma(pic, mb_x, mb_y, br_row, br_col);
            let blk = decode_residual_block(br, nc, BlockKind::Luma4x4)?;
            total_coeff = blk.total_coeff;
            residual = blk.coeffs;
            let scale = *pic.scaling_lists.matrix_4x4(0);
            dequantize_4x4_scaled_ext(&mut residual, qp_y_prime, &scale);
            idct_4x4(&mut residual);
        }
        let lo = lo_mb + br_row * 4 * lstride + br_col * 4;
        for r in 0..4 {
            for c in 0..4 {
                let v = pred[r * 4 + c] as i32 + residual[r * 4 + c];
                pic.y16[lo + r * lstride + c] = v.clamp(0, max_sample as i32) as u16;
            }
        }
        pic.mb_info_mut(mb_x, mb_y).luma_nc[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}

fn decode_luma_intra_16x16_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    intra16x16_pred_mode: u8,
    cbp_luma: u8,
    qp_y_prime: i32,
) -> Result<()> {
    let bit_depth = (sps.bit_depth_luma_minus8 + 8) as u8;
    let max_sample = (1u16 << bit_depth) - 1;
    let neigh = collect_intra16x16_neighbours_hi(pic, mb_x, mb_y);
    let mut pred = [0u16; 256];
    let mode = Intra16x16Mode::from_u8(intra16x16_pred_mode).ok_or_else(|| {
        Error::invalid(format!(
            "h264 mb: invalid intra16x16 mode {intra16x16_pred_mode}"
        ))
    })?;
    pred16_hi(&mut pred, mode, &neigh, bit_depth);

    let nc_dc = predict_nc_luma(pic, mb_x, mb_y, 0, 0);
    let dc_block = decode_residual_block(br, nc_dc, BlockKind::Luma16x16Dc)?;
    let mut dc = dc_block.coeffs;
    let w_dc = pic.scaling_lists.matrix_4x4(0)[0];
    inv_hadamard_4x4_dc_scaled_ext(&mut dc, qp_y_prime, w_dc);

    let lstride = pic.luma_stride();
    let lo_mb = pic.luma_off(mb_x, mb_y);
    for blk in 0..16usize {
        let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
        let mut residual = [0i32; 16];
        let mut total_coeff = 0u32;
        if cbp_luma != 0 {
            let nc = predict_nc_luma(pic, mb_x, mb_y, br_row, br_col);
            let ac = decode_residual_block(br, nc, BlockKind::Luma16x16Ac)?;
            total_coeff = ac.total_coeff;
            residual = ac.coeffs;
            let scale = *pic.scaling_lists.matrix_4x4(0);
            dequantize_4x4_scaled_ext(&mut residual, qp_y_prime, &scale);
        }
        residual[0] = dc[br_row * 4 + br_col];
        idct_4x4(&mut residual);

        let lo = lo_mb + br_row * 4 * lstride + br_col * 4;
        for r in 0..4 {
            for c in 0..4 {
                let v = pred[(br_row * 4 + r) * 16 + (br_col * 4 + c)] as i32
                    + residual[r * 4 + c];
                pic.y16[lo + r * lstride + c] = v.clamp(0, max_sample as i32) as u16;
            }
        }
        pic.mb_info_mut(mb_x, mb_y).luma_nc[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}

fn decode_chroma_hi(
    br: &mut BitReader<'_>,
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
    let max_sample = (1u16 << bit_depth_c) - 1;
    let qp_bd_offset_c = 6 * sps.bit_depth_chroma_minus8 as i32;

    // §8.5.11 — chroma QP derivation with QpBdOffset support.
    // QPI = Clip3(-QpBdOffsetC, 51 + QpBdOffsetC, QpY + chroma_qp_index_offset)
    // If QPI >= 0 and QPI <= 51: QpC = QP_CHROMA_TABLE[QPI]; else QpC = QPI.
    // Internal dequant QP (QpC_prime) = QpC + QpBdOffsetC.
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
        let blk = decode_residual_block(br, 0, BlockKind::ChromaDc2x2)?;
        for i in 0..4 {
            dc_cb[i] = blk.coeffs[i];
        }
        let blk = decode_residual_block(br, 0, BlockKind::ChromaDc2x2)?;
        for i in 0..4 {
            dc_cr[i] = blk.coeffs[i];
        }
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
                let nc = predict_nc_chroma_hi(pic, mb_x, mb_y, plane_kind, br_row, br_col);
                let ac = decode_residual_block(br, nc, BlockKind::ChromaAc)?;
                total_coeff = ac.total_coeff;
                res = ac.coeffs;
                let cat = if plane_kind { 1 } else { 2 };
                let scale = *pic.scaling_lists.matrix_4x4(cat);
                dequantize_4x4_scaled_ext(&mut res, qpc_prime, &scale);
            }
            res[0] = dc[(br_row << 1) | br_col];
            idct_4x4(&mut res);
            let off_in_mb = br_row * 4 * cstride + br_col * 4;
            let plane = if plane_kind { &mut pic.cb16 } else { &mut pic.cr16 };
            for r in 0..4 {
                for c in 0..4 {
                    let v = pred[(br_row * 4 + r) * 8 + (br_col * 4 + c)] as i32
                        + res[r * 4 + c];
                    plane[co + off_in_mb + r * cstride + c] =
                        v.clamp(0, max_sample as i32) as u16;
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

fn predict_nc_chroma_hi(
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

pub(crate) fn predict_intra4x4_mode_with(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    br_row: usize,
    br_col: usize,
    in_progress_modes: &[u8; 16],
) -> u8 {
    let here_get = |row: usize, col: usize| in_progress_modes[row * 4 + col];

    let left_mode = if br_col > 0 {
        Some(here_get(br_row, br_col - 1))
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
        Some(here_get(br_row - 1, br_col))
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
        let tr_avail =
            crate::mb::top_right_available_4x4(mb_x, mb_y, br_row, br_col, pic);
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

fn collect_intra16x16_neighbours_hi(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
) -> Intra16x16Neighbours16 {
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
