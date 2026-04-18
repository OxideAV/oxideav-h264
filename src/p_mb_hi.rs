//! High-bit-depth P-slice macroblock decode — ITU-T H.264 §7.3.5 / §8.4
//! on u16 planes.
//!
//! Mirrors [`crate::p_mb`] but writes `Picture::y16 / cb16 / cr16` and
//! dispatches motion compensation through the u16 variants in
//! [`crate::motion`] (`luma_mc_hi`, `chroma_mc_hi`, `apply_luma_weight_hi`,
//! `apply_chroma_weight_hi`). Residual dequant goes through the
//! `*_scaled_ext` i64-widened helpers so the `QP_Y + QpBdOffsetY` range
//! (up to 63 at 10-bit) doesn't overflow; `mb_qp_delta` wraps mod
//! `52 + QpBdOffsetY`; intra-in-P dispatches back to
//! [`crate::mb_hi::decode_intra_mb_given_imb_hi`].
//!
//! Scope: 4:2:0 CAVLC, P_L0_16×16 / P_L0_16×8 / P_L0_8×16 / P_8×8 /
//! P_8×8ref0, P_Skip, intra-in-P. The 8×8 transform path at 10-bit is
//! gated the same way as the 8-bit path (§7.3.5.1 conditional on PPS +
//! `cbp_luma > 0` + no sub-8×8 motion partitions for P_8x8).

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::cavlc::{decode_residual_8x8_sub, decode_residual_block, BlockKind};
use crate::mb::LUMA_BLOCK_RASTER;
use crate::mb_type::{
    decode_cbp_inter, decode_p_slice_mb_type, decode_p_sub_mb_type, IMbType, PMbType, PPartition,
    PSubPartition,
};
use crate::motion::{
    apply_chroma_weight_hi, apply_luma_weight_hi, chroma_mc_hi, luma_mc_hi, predict_mv_l0,
    predict_mv_pskip,
};
use crate::picture::{MbInfo, Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::{ChromaWeight, LumaWeight, SliceHeader};
use crate::sps::Sps;
use crate::transform::{
    chroma_qp_hi, dequantize_4x4_scaled_ext, idct_4x4, idct_8x8,
    inv_hadamard_2x2_chroma_dc_scaled_ext,
};

/// Decode one coded P-slice macroblock at `(mb_x, mb_y)` in the 10-bit
/// pipeline. Mirrors [`crate::p_mb::decode_p_slice_mb`] on u16 planes.
pub fn decode_p_slice_mb_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    prev_qp: &mut i32,
) -> Result<()> {
    let mb_type = br.read_ue()?;
    let pmb = decode_p_slice_mb_type(mb_type)
        .ok_or_else(|| Error::invalid(format!("h264 p-slice: bad P mb_type {mb_type}")))?;

    match pmb {
        PMbType::IntraInP(imb) => decode_intra_in_p_hi(br, sps, pps, sh, mb_x, mb_y, pic, prev_qp, imb),
        PMbType::Inter { partition } => match partition {
            PPartition::P16x16 => {
                decode_p16x16_hi(br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, prev_qp)
            }
            PPartition::P16x8 => {
                decode_p16x8_hi(br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, prev_qp)
            }
            PPartition::P8x16 => {
                decode_p8x16_hi(br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, prev_qp)
            }
            PPartition::P8x8 | PPartition::P8x8Ref0 => decode_p8x8_hi(
                br,
                sps,
                pps,
                sh,
                mb_x,
                mb_y,
                pic,
                ref_list0,
                prev_qp,
                matches!(partition, PPartition::P8x8Ref0),
            ),
        },
    }
}

/// `P_Skip` derivation for the 10-bit path — no residual, MC from the
/// §8.4.1.1 derived MV. Chroma + luma weights honour the slice's
/// `pred_weight_table` if present.
pub fn decode_p_skip_mb_hi(
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    prev_qp: i32,
) -> Result<()> {
    let reference = lookup_ref(ref_list0, 0)?;
    let mv = predict_mv_pskip(pic, mb_x, mb_y);
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        *info = MbInfo {
            qp_y: prev_qp,
            coded: true,
            intra: false,
            skipped: true,
            mb_type_p: Some(PMbType::Inter {
                partition: PPartition::P16x16,
            }),
            p_partition: Some(PPartition::P16x16),
            ..Default::default()
        };
    }
    fill_partition_mv(pic, mb_x, mb_y, 0, 0, 4, 4, mv, 0);
    let (lw, cw) = l0_weight_for(sh, 0);
    mc_luma_partition_hi(pic, reference, mb_x, mb_y, 0, 0, 4, 4, mv.0 as i32, mv.1 as i32, lw);
    mc_chroma_partition_hi(pic, reference, mb_x, mb_y, 0, 0, 4, 4, mv.0 as i32, mv.1 as i32, cw);
    Ok(())
}

fn decode_intra_in_p_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    prev_qp: &mut i32,
    imb: IMbType,
) -> Result<()> {
    crate::mb_hi::decode_intra_mb_given_imb_hi(br, sps, pps, sh, mb_x, mb_y, pic, prev_qp, imb)
}

fn decode_p16x16_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    prev_qp: &mut i32,
) -> Result<()> {
    let num_ref_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let ref_idx = read_ref_idx(br, num_ref_l0)?;
    let reference = lookup_ref(ref_list0, ref_idx)?;
    let mvd_x = br.read_se()? as i32;
    let mvd_y = br.read_se()? as i32;

    let pmv = predict_mv_l0(pic, mb_x, mb_y, 0, 0, 4, 4, ref_idx);
    let mv = (
        pmv.0.wrapping_add(mvd_x as i16),
        pmv.1.wrapping_add(mvd_y as i16),
    );

    fill_partition_mv(pic, mb_x, mb_y, 0, 0, 4, 4, mv, ref_idx);

    let (lw, cw) = l0_weight_for(sh, ref_idx);
    mc_luma_partition_hi(pic, reference, mb_x, mb_y, 0, 0, 4, 4, mv.0 as i32, mv.1 as i32, lw);
    mc_chroma_partition_hi(pic, reference, mb_x, mb_y, 0, 0, 4, 4, mv.0 as i32, mv.1 as i32, cw);

    let cbp_raw = br.read_ue()?;
    let (cbp_luma, cbp_chroma) = decode_cbp_inter(cbp_raw)
        .ok_or_else(|| Error::invalid(format!("h264 p-slice: bad inter CBP {cbp_raw}")))?;
    let transform_8x8 = if pps.transform_8x8_mode_flag && cbp_luma != 0 {
        br.read_flag()?
    } else {
        false
    };
    let needs_qp = cbp_luma != 0 || cbp_chroma != 0;
    if needs_qp {
        let dqp = br.read_se()?;
        apply_qp_delta_hi(prev_qp, dqp, sps);
    }
    let qp_y = *prev_qp;
    let qp_y_prime = qp_y + 6 * sps.bit_depth_luma_minus8 as i32;

    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        info.qp_y = qp_y;
        info.coded = true;
        info.intra = false;
        info.skipped = false;
        info.mb_type_p = Some(PMbType::Inter {
            partition: PPartition::P16x16,
        });
        info.p_partition = Some(PPartition::P16x16);
        info.intra4x4_pred_mode = [INTRA_DC_FAKE; 16];
        info.cbp_luma = cbp_luma;
        info.cbp_chroma = cbp_chroma;
        info.transform_8x8 = transform_8x8;
    }

    if transform_8x8 {
        decode_inter_residual_luma_8x8_hi(br, mb_x, mb_y, pic, cbp_luma, qp_y_prime)?;
    } else {
        decode_inter_residual_luma_hi(br, mb_x, mb_y, pic, cbp_luma, qp_y_prime)?;
    }
    decode_inter_residual_chroma_hi(br, sps, pps, mb_x, mb_y, pic, cbp_chroma, qp_y)?;
    Ok(())
}

fn decode_p16x8_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    prev_qp: &mut i32,
) -> Result<()> {
    let num_ref_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let ref_idx_0 = read_ref_idx(br, num_ref_l0)?;
    let ref_idx_1 = read_ref_idx(br, num_ref_l0)?;

    let rects = [(0usize, 0usize, 2usize, 4usize), (2, 0, 2, 4)];
    let refs = [ref_idx_0, ref_idx_1];
    decode_two_partition_p_hi(
        br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, prev_qp, &rects, &refs, PPartition::P16x8,
    )
}

fn decode_p8x16_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    prev_qp: &mut i32,
) -> Result<()> {
    let num_ref_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let ref_idx_0 = read_ref_idx(br, num_ref_l0)?;
    let ref_idx_1 = read_ref_idx(br, num_ref_l0)?;

    let rects = [(0usize, 0usize, 4usize, 2usize), (0, 2, 4, 2)];
    let refs = [ref_idx_0, ref_idx_1];
    decode_two_partition_p_hi(
        br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, prev_qp, &rects, &refs, PPartition::P8x16,
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_two_partition_p_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    prev_qp: &mut i32,
    rects: &[(usize, usize, usize, usize); 2],
    refs: &[i8; 2],
    partition: PPartition,
) -> Result<()> {
    let mut mvs = [(0i16, 0i16); 2];
    for p in 0..2 {
        let (r0, c0, ph, pw) = rects[p];
        let reference = lookup_ref(ref_list0, refs[p])?;
        let mvd_x = br.read_se()? as i32;
        let mvd_y = br.read_se()? as i32;
        let pmv = predict_mv_l0(pic, mb_x, mb_y, r0, c0, ph, pw, refs[p]);
        let mv = (
            pmv.0.wrapping_add(mvd_x as i16),
            pmv.1.wrapping_add(mvd_y as i16),
        );
        mvs[p] = mv;
        fill_partition_mv(pic, mb_x, mb_y, r0, c0, ph, pw, mv, refs[p]);
        let (lw, cw) = l0_weight_for(sh, refs[p]);
        mc_luma_partition_hi(pic, reference, mb_x, mb_y, r0, c0, ph, pw, mv.0 as i32, mv.1 as i32, lw);
        mc_chroma_partition_hi(pic, reference, mb_x, mb_y, r0, c0, ph, pw, mv.0 as i32, mv.1 as i32, cw);
    }

    let cbp_raw = br.read_ue()?;
    let (cbp_luma, cbp_chroma) = decode_cbp_inter(cbp_raw)
        .ok_or_else(|| Error::invalid(format!("h264 p-slice: bad inter CBP {cbp_raw}")))?;
    let transform_8x8 = if pps.transform_8x8_mode_flag && cbp_luma != 0 {
        br.read_flag()?
    } else {
        false
    };
    let needs_qp = cbp_luma != 0 || cbp_chroma != 0;
    if needs_qp {
        let dqp = br.read_se()?;
        apply_qp_delta_hi(prev_qp, dqp, sps);
    }
    let qp_y = *prev_qp;
    let qp_y_prime = qp_y + 6 * sps.bit_depth_luma_minus8 as i32;

    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        info.qp_y = qp_y;
        info.coded = true;
        info.intra = false;
        info.skipped = false;
        info.p_partition = Some(partition);
        info.mb_type_p = Some(PMbType::Inter { partition });
        info.intra4x4_pred_mode = [INTRA_DC_FAKE; 16];
        info.cbp_luma = cbp_luma;
        info.cbp_chroma = cbp_chroma;
        info.transform_8x8 = transform_8x8;
    }

    if transform_8x8 {
        decode_inter_residual_luma_8x8_hi(br, mb_x, mb_y, pic, cbp_luma, qp_y_prime)?;
    } else {
        decode_inter_residual_luma_hi(br, mb_x, mb_y, pic, cbp_luma, qp_y_prime)?;
    }
    decode_inter_residual_chroma_hi(br, sps, pps, mb_x, mb_y, pic, cbp_chroma, qp_y)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_p8x8_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    prev_qp: &mut i32,
    is_ref0: bool,
) -> Result<()> {
    let num_ref_l0 = sh.num_ref_idx_l0_active_minus1 + 1;

    let mut sub_parts = [PSubPartition::Sub8x8; 4];
    for s in 0..4 {
        let v = br.read_ue()?;
        sub_parts[s] = decode_p_sub_mb_type(v)
            .ok_or_else(|| Error::invalid(format!("h264 p-slice: bad sub_mb_type {v}")))?;
    }
    let mut ref_idxs = [0i8; 4];
    if !is_ref0 {
        for s in 0..4 {
            ref_idxs[s] = read_ref_idx(br, num_ref_l0)?;
        }
    }
    const SUB_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 2), (2, 0), (2, 2)];
    for s in 0..4 {
        let sp = sub_parts[s];
        let (sr0, sc0) = SUB_OFFSETS[s];
        let ref_idx = ref_idxs[s];
        let reference = lookup_ref(ref_list0, ref_idx)?;
        for sub_idx in 0..sp.num_sub_partitions() {
            let (dr, dc, sh_h, sh_w) = sp.sub_rect(sub_idx);
            let r0 = sr0 + dr;
            let c0 = sc0 + dc;
            let mvd_x = br.read_se()? as i32;
            let mvd_y = br.read_se()? as i32;
            let pmv = predict_mv_l0(pic, mb_x, mb_y, r0, c0, sh_h, sh_w, ref_idx);
            let mv = (
                pmv.0.wrapping_add(mvd_x as i16),
                pmv.1.wrapping_add(mvd_y as i16),
            );
            fill_partition_mv(pic, mb_x, mb_y, r0, c0, sh_h, sh_w, mv, ref_idx);
            let (lw, cw) = l0_weight_for(sh, ref_idx);
            mc_luma_partition_hi(pic, reference, mb_x, mb_y, r0, c0, sh_h, sh_w, mv.0 as i32, mv.1 as i32, lw);
            mc_chroma_partition_hi(pic, reference, mb_x, mb_y, r0, c0, sh_h, sh_w, mv.0 as i32, mv.1 as i32, cw);
        }
    }

    let cbp_raw = br.read_ue()?;
    let (cbp_luma, cbp_chroma) = decode_cbp_inter(cbp_raw)
        .ok_or_else(|| Error::invalid(format!("h264 p-slice: bad inter CBP {cbp_raw}")))?;
    let all_sub_8x8 = sub_parts.iter().all(|sp| sp.is_at_least_8x8());
    let transform_8x8 =
        if pps.transform_8x8_mode_flag && cbp_luma != 0 && all_sub_8x8 {
            br.read_flag()?
        } else {
            false
        };
    let needs_qp = cbp_luma != 0 || cbp_chroma != 0;
    if needs_qp {
        let dqp = br.read_se()?;
        apply_qp_delta_hi(prev_qp, dqp, sps);
    }
    let qp_y = *prev_qp;
    let qp_y_prime = qp_y + 6 * sps.bit_depth_luma_minus8 as i32;

    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        info.qp_y = qp_y;
        info.coded = true;
        info.intra = false;
        info.skipped = false;
        let partition = if is_ref0 {
            PPartition::P8x8Ref0
        } else {
            PPartition::P8x8
        };
        info.p_partition = Some(partition);
        info.mb_type_p = Some(PMbType::Inter { partition });
        info.intra4x4_pred_mode = [INTRA_DC_FAKE; 16];
        info.cbp_luma = cbp_luma;
        info.cbp_chroma = cbp_chroma;
        info.transform_8x8 = transform_8x8;
    }

    if transform_8x8 {
        decode_inter_residual_luma_8x8_hi(br, mb_x, mb_y, pic, cbp_luma, qp_y_prime)?;
    } else {
        decode_inter_residual_luma_hi(br, mb_x, mb_y, pic, cbp_luma, qp_y_prime)?;
    }
    decode_inter_residual_chroma_hi(br, sps, pps, mb_x, mb_y, pic, cbp_chroma, qp_y)?;
    Ok(())
}

fn apply_qp_delta_hi(prev_qp: &mut i32, dqp: i32, sps: &Sps) {
    let qp_bd_offset_y = 6 * sps.bit_depth_luma_minus8 as i32;
    let modulus = 52 + qp_bd_offset_y;
    let mut q = (*prev_qp + dqp + 2 * modulus) % modulus;
    if q > 51 {
        q -= modulus;
    }
    *prev_qp = q;
}

fn decode_inter_residual_luma_hi(
    br: &mut BitReader<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    cbp_luma: u8,
    qp_y_prime: i32,
) -> Result<()> {
    let bit_depth = pic.bit_depth_y;
    let max_sample: i32 = (1i32 << bit_depth) - 1;
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
        let scale = *pic.scaling_lists.matrix_4x4(3);
        dequantize_4x4_scaled_ext(&mut residual, qp_y_prime, &scale);
        idct_4x4(&mut residual);
        let lo = lo_mb + br_row * 4 * lstride + br_col * 4;
        for r in 0..4 {
            for c in 0..4 {
                let base = pic.y16[lo + r * lstride + c] as i32;
                let v = (base + residual[r * 4 + c]).clamp(0, max_sample);
                pic.y16[lo + r * lstride + c] = v as u16;
            }
        }
        pic.mb_info_mut(mb_x, mb_y).luma_nc[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}

fn decode_inter_residual_luma_8x8_hi(
    br: &mut BitReader<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    cbp_luma: u8,
    qp_y_prime: i32,
) -> Result<()> {
    let bit_depth = pic.bit_depth_y;
    let max_sample: i32 = (1i32 << bit_depth) - 1;
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

        let mut residual = [0i32; 64];
        let slots = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)];
        for (sub, &(sr, sc)) in slots.iter().enumerate() {
            let nc = predict_inter_nc_luma(pic, mb_x, mb_y, sr, sc);
            let tc = decode_residual_8x8_sub(br, nc, sub, &mut residual)?;
            pic.mb_info_mut(mb_x, mb_y).luma_nc[sr * 4 + sc] = tc;
        }
        let scale = *pic.scaling_lists.matrix_8x8(1);
        // §8.5.13.1 extended-QP dequant — 10-bit pushes QpY' up to 63,
        // past the 8-bit path's `qp.clamp(0, 51)`, so widen the
        // intermediates to i64 and leave the shift pattern unbounded.
        dequantize_8x8_scaled_ext(&mut residual, qp_y_prime, &scale);
        idct_8x8(&mut residual);

        let info = pic.mb_info_mut(mb_x, mb_y);
        let sum = info.luma_nc[r0 * 4 + c0] as u16
            + info.luma_nc[r0 * 4 + c0 + 1] as u16
            + info.luma_nc[(r0 + 1) * 4 + c0] as u16
            + info.luma_nc[(r0 + 1) * 4 + c0 + 1] as u16;
        info.luma_nc[r0 * 4 + c0] = sum.min(16) as u8;

        let lo = lo_mb + (br8_row * 8) * lstride + br8_col * 8;
        for r in 0..8 {
            for c in 0..8 {
                let base = pic.y16[lo + r * lstride + c] as i32;
                let v = (base + residual[r * 8 + c]).clamp(0, max_sample);
                pic.y16[lo + r * lstride + c] = v as u16;
            }
        }
    }
    Ok(())
}

/// Inline 8×8 dequant widened to i64 for the extended-QP range.
/// Keeps the same LevelScale8x8 / shift pattern as
/// [`crate::transform::dequantize_8x8_scaled`] but doesn't clamp the QP
/// to 51. The `QP >= 36` branch is active at 10-bit once the caller adds
/// `QpBdOffsetY = 12`.
fn dequantize_8x8_scaled_ext(coeffs: &mut [i32; 64], qp: i32, weight_scale: &[i16; 64]) {
    // Table 8-15 — normAdjust8x8(qP%6, m).
    const V8X8_TABLE: [[i32; 6]; 6] = [
        [20, 18, 32, 19, 25, 24],
        [22, 19, 35, 21, 28, 26],
        [26, 23, 42, 24, 33, 31],
        [28, 25, 45, 26, 35, 33],
        [32, 28, 51, 30, 40, 38],
        [36, 32, 58, 34, 46, 43],
    ];
    const V8X8_POS_CLASS: [usize; 16] = [0, 3, 4, 3, 3, 1, 5, 1, 4, 5, 2, 5, 3, 1, 5, 1];
    fn pos_class_8x8(row: usize, col: usize) -> usize {
        V8X8_POS_CLASS[((row & 3) << 2) | (col & 3)]
    }
    let qp = qp.max(0);
    let qp6 = (qp / 6) as i32;
    let qmod = (qp % 6) as usize;
    if qp6 >= 6 {
        let shift = (qp6 - 6) as u32;
        for r in 0..8 {
            for c in 0..8 {
                let i = r * 8 + c;
                let w = weight_scale[i] as i64;
                let v = V8X8_TABLE[qmod][pos_class_8x8(r, c)] as i64;
                coeffs[i] = ((coeffs[i] as i64 * v * w) << shift) as i32;
            }
        }
    } else {
        let shift = (6 - qp6) as u32;
        let round = 1i64 << (shift - 1);
        for r in 0..8 {
            for c in 0..8 {
                let i = r * 8 + c;
                let w = weight_scale[i] as i64;
                let v = V8X8_TABLE[qmod][pos_class_8x8(r, c)] as i64;
                coeffs[i] = ((coeffs[i] as i64 * v * w + round) >> shift) as i32;
            }
        }
    }
}

fn decode_inter_residual_chroma_hi(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    cbp_chroma: u8,
    qp_y: i32,
) -> Result<()> {
    let bit_depth_c = pic.bit_depth_c;
    let max_sample: i32 = (1i32 << bit_depth_c) - 1;
    let qp_bd_offset_c = 6 * sps.bit_depth_chroma_minus8 as i32;
    let qpi = (qp_y + pps.chroma_qp_index_offset).clamp(-qp_bd_offset_c, 51 + qp_bd_offset_c);
    let qpc = chroma_qp_hi(qpi);
    let qpc_prime = qpc + qp_bd_offset_c;

    let mut dc_cb = [0i32; 4];
    let mut dc_cr = [0i32; 4];
    if cbp_chroma >= 1 {
        let blk = decode_residual_block(br, 0, BlockKind::ChromaDc2x2)?;
        dc_cb[..4].copy_from_slice(&blk.coeffs[..4]);
        let blk = decode_residual_block(br, 0, BlockKind::ChromaDc2x2)?;
        dc_cr[..4].copy_from_slice(&blk.coeffs[..4]);
        // P-slice chroma is inter → slot 4 (Cb) / slot 5 (Cr).
        let w_cb = pic.scaling_lists.matrix_4x4(4)[0];
        let w_cr = pic.scaling_lists.matrix_4x4(5)[0];
        inv_hadamard_2x2_chroma_dc_scaled_ext(&mut dc_cb, qpc_prime, w_cb);
        inv_hadamard_2x2_chroma_dc_scaled_ext(&mut dc_cr, qpc_prime, w_cr);
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
                let cat = if plane_kind { 4 } else { 5 };
                let scale = *pic.scaling_lists.matrix_4x4(cat);
                dequantize_4x4_scaled_ext(&mut res, qpc_prime, &scale);
            }
            res[0] = dc[(br_row << 1) | br_col];
            idct_4x4(&mut res);
            let off_in_mb = br_row * 4 * cstride + br_col * 4;
            let plane = if plane_kind { &mut pic.cb16 } else { &mut pic.cr16 };
            for r in 0..4 {
                for c in 0..4 {
                    let base = plane[co + off_in_mb + r * cstride + c] as i32;
                    let v = (base + res[r * 4 + c]).clamp(0, max_sample);
                    plane[co + off_in_mb + r * cstride + c] = v as u16;
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

fn read_ref_idx(br: &mut BitReader<'_>, num_ref: u32) -> Result<i8> {
    if num_ref == 0 {
        return Ok(0);
    }
    if num_ref == 1 {
        return Ok(0);
    }
    if num_ref == 2 {
        let bit = br.read_u1()?;
        return Ok(if bit == 1 { 0 } else { 1 });
    }
    Ok(br.read_ue()? as i8)
}

fn lookup_ref<'a>(ref_list0: &[&'a Picture], ref_idx: i8) -> Result<&'a Picture> {
    if ref_idx < 0 {
        return Err(Error::invalid(
            "h264 p-slice: negative ref_idx_l0 (should not be emitted by a valid encoder)",
        ));
    }
    let idx = ref_idx as usize;
    ref_list0.get(idx).copied().ok_or_else(|| {
        Error::invalid(format!(
            "h264 p-slice: ref_idx_l0 {} out of RefPicList0 range (len={})",
            ref_idx,
            ref_list0.len()
        ))
    })
}

fn fill_partition_mv(
    pic: &mut Picture,
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    h: usize,
    w: usize,
    mv: (i16, i16),
    ref_idx: i8,
) {
    let info = pic.mb_info_mut(mb_x, mb_y);
    for rr in r0..r0 + h {
        for cc in c0..c0 + w {
            info.mv_l0[rr * 4 + cc] = mv;
            info.ref_idx_l0[rr * 4 + cc] = ref_idx;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mc_luma_partition_hi(
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
    let mut tmp = vec![0u16; pw * ph];
    let base_x = (mb_x as i32) * 16 + (c0 as i32) * 4;
    let base_y = (mb_y as i32) * 16 + (r0 as i32) * 4;
    luma_mc_hi(&mut tmp, reference, base_x, base_y, mv_x_q, mv_y_q, pw, ph);
    if let Some(lw) = luma_weight {
        apply_luma_weight_hi(&mut tmp, lw, pic.bit_depth_y);
    }
    let lstride = pic.luma_stride();
    let off = pic.luma_off(mb_x, mb_y) + r0 * 4 * lstride + c0 * 4;
    for rr in 0..ph {
        for cc in 0..pw {
            pic.y16[off + rr * lstride + cc] = tmp[rr * pw + cc];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mc_chroma_partition_hi(
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
    let ph = h * 2;
    let mut tmp_cb = vec![0u16; pw * ph];
    let mut tmp_cr = vec![0u16; pw * ph];
    let base_x = (mb_x as i32) * 8 + (c0 as i32) * 2;
    let base_y = (mb_y as i32) * 8 + (r0 as i32) * 2;
    let cstride = reference.chroma_stride();
    let cw_plane = (reference.width / 2) as i32;
    let ch_plane = (reference.height / 2) as i32;
    chroma_mc_hi(
        &mut tmp_cb, &reference.cb16, cstride, cw_plane, ch_plane, base_x, base_y, mv_x_q, mv_y_q,
        pw, ph, reference.bit_depth_c,
    );
    chroma_mc_hi(
        &mut tmp_cr, &reference.cr16, cstride, cw_plane, ch_plane, base_x, base_y, mv_x_q, mv_y_q,
        pw, ph, reference.bit_depth_c,
    );
    if let Some(cw_entry) = chroma_weight {
        apply_chroma_weight_hi(&mut tmp_cb, cw_entry, 0, pic.bit_depth_c);
        apply_chroma_weight_hi(&mut tmp_cr, cw_entry, 1, pic.bit_depth_c);
    }
    let stride = pic.chroma_stride();
    let off = pic.chroma_off(mb_x, mb_y) + r0 * 2 * stride + c0 * 2;
    for rr in 0..ph {
        for cc in 0..pw {
            pic.cb16[off + rr * stride + cc] = tmp_cb[rr * pw + cc];
            pic.cr16[off + rr * stride + cc] = tmp_cr[rr * pw + cc];
        }
    }
}

fn l0_weight_for<'a>(
    sh: &'a SliceHeader,
    ref_idx: i8,
) -> (Option<&'a LumaWeight>, Option<&'a ChromaWeight>) {
    let Some(tbl) = sh.pred_weight_table.as_ref() else {
        return (None, None);
    };
    let idx = ref_idx.max(0) as usize;
    let lw = tbl.luma_l0.get(idx);
    let cw = tbl.chroma_l0.get(idx);
    (lw, cw)
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
