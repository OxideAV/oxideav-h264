//! 10-bit CABAC P-slice macroblock decode — ITU-T H.264 §7.3.5 + §9.3
//! with the bit-depth extensions from §7.4.2.1.1 and §8.5.12.1.
//!
//! Mirrors [`crate::cabac::p_mb::decode_p_mb_cabac`] but writes the
//! reconstructed samples into `Picture::y16 / cb16 / cr16` and routes the
//! residual dequant through the `*_ext` i64-widened helpers so the
//! extended `QpY + QpBdOffsetY` range (up to 63 at 10-bit) doesn't overflow.
//!
//! The CABAC entropy layer is bit-depth agnostic — context state,
//! binarisation, MVD + ref_idx derivation, CBP, and residual decode all run
//! unchanged. What differs:
//!
//! * Motion compensation targets `u16` planes via
//!   [`crate::motion::luma_mc_hi`] / [`crate::motion::chroma_mc_hi`] with
//!   per-bit-depth Clip1; uses weighted-prediction `*_hi` variants.
//! * Residual dequant uses `dequantize_4x4_scaled_ext` and
//!   `inv_hadamard_2x2_chroma_dc_scaled_ext` with `qp + QpBdOffset`.
//! * `mb_qp_delta` wraps modulo `52 + QpBdOffsetY` (§7.4.5 extended).
//! * Intra-in-P dispatches to the 10-bit CABAC I helper
//!   [`crate::cabac::mb_hi::decode_intra_mb_given_imb_cabac_hi`].
//!
//! Scope: 4:2:0, CABAC, P-slice (Inter partitions + Intra-in-P + P_Skip).
//! `transform_size_8x8_flag = 1` returns `Error::Unsupported` to match
//! the existing 10-bit CABAC I gate (Intra_8×8 / 8×8 transform is a
//! separate work item at 10-bit).

use oxideav_core::{Error, Result};

use crate::cabac::binarize;
use crate::cabac::context::CabacContext;
use crate::cabac::engine::CabacDecoder;
use crate::cabac::mb::{
    cbp_chroma_ctx_idx_inc_ac, cbp_chroma_ctx_idx_inc_any, cbp_luma_ctx_idx_inc,
    decode_residual_block_in_place, transform_size_8x8_flag_ctx_idx_inc,
};
use crate::cabac::p_mb as cabac_p;
use crate::cabac::residual::BlockCat;
use crate::cabac::tables::{
    CTX_IDX_CODED_BLOCK_PATTERN_LUMA, CTX_IDX_MB_QP_DELTA, CTX_IDX_MB_SKIP_FLAG_P,
    CTX_IDX_MB_TYPE_INTRA_IN_P, CTX_IDX_MB_TYPE_P, CTX_IDX_REF_IDX_L0, CTX_IDX_SUB_MB_TYPE_P,
    CTX_IDX_TRANSFORM_SIZE_8X8_FLAG,
};
use crate::mb::LUMA_BLOCK_RASTER;
use crate::mb_type::{PMbType, PPartition, PSubPartition};
use crate::motion::{
    apply_chroma_weight_hi, apply_luma_weight_hi, chroma_mc_hi, luma_mc_hi, predict_mv_l0,
};
use crate::picture::{Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::{ChromaWeight, LumaWeight, SliceHeader};
use crate::sps::Sps;
use crate::transform::{
    chroma_qp_hi, dequantize_4x4_scaled_ext, idct_4x4, inv_hadamard_2x2_chroma_dc_scaled_ext,
};

/// MVD context banks for list-0 components — mirrors the 8-bit CABAC P
/// module's layout (7-slot windows).
const CTX_IDX_MVD_L0_X: usize = 40;
const CTX_IDX_MVD_L0_Y: usize = 47;

const SUB_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 2), (2, 0), (2, 2)];

/// Decode one P-slice macroblock through CABAC on the 10-bit pipeline.
/// Returns `Ok(true)` when `mb_skip_flag = 1` (no residual), matching the
/// shape of [`crate::cabac::p_mb::decode_p_mb_cabac`].
#[allow(clippy::too_many_arguments)]
pub fn decode_p_mb_cabac_hi(
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
    let skip_inc = cabac_p::mb_skip_flag_ctx_idx_inc(pic, mb_x, mb_y);
    let skipped = {
        let slice = &mut ctxs[CTX_IDX_MB_SKIP_FLAG_P..CTX_IDX_MB_SKIP_FLAG_P + 3];
        binarize::decode_mb_skip_flag_p(d, slice, skip_inc)?
    };

    if skipped {
        crate::p_mb_hi::decode_p_skip_mb_hi(sh, mb_x, mb_y, pic, ref_list0, *prev_qp)?;
        return Ok(true);
    }

    let mb_type_raw = {
        let slice = &mut ctxs[CTX_IDX_MB_TYPE_P..CTX_IDX_MB_TYPE_P + 7];
        binarize::decode_mb_type_p(d, slice, CTX_IDX_MB_TYPE_INTRA_IN_P - CTX_IDX_MB_TYPE_P)?
    };

    match mb_type_raw {
        0 => decode_p_inter_hi(
            d,
            ctxs,
            sh,
            sps,
            pps,
            mb_x,
            mb_y,
            pic,
            ref_list0,
            prev_qp,
            PPartition::P16x16,
        ),
        1 => decode_p_inter_hi(
            d,
            ctxs,
            sh,
            sps,
            pps,
            mb_x,
            mb_y,
            pic,
            ref_list0,
            prev_qp,
            PPartition::P16x8,
        ),
        2 => decode_p_inter_hi(
            d,
            ctxs,
            sh,
            sps,
            pps,
            mb_x,
            mb_y,
            pic,
            ref_list0,
            prev_qp,
            PPartition::P8x16,
        ),
        3 => decode_p_8x8_hi(
            d, ctxs, sh, sps, pps, mb_x, mb_y, pic, ref_list0, prev_qp, false,
        ),
        5..=30 => {
            let imb = crate::mb_type::decode_i_slice_mb_type(mb_type_raw - 5).ok_or_else(|| {
                Error::invalid(format!(
                    "h264 10bit cabac p-slice: bad intra-in-P mb_type {mb_type_raw}"
                ))
            })?;
            crate::cabac::mb_hi::decode_intra_mb_given_imb_cabac_hi(
                d, ctxs, sh, sps, pps, mb_x, mb_y, pic, prev_qp, imb,
            )
        }
        _ => Err(Error::invalid(format!(
            "h264 10bit cabac p-slice: unexpected mb_type {mb_type_raw}"
        ))),
    }?;
    Ok(false)
}

// ---------------------------------------------------------------------------
// Inter partitions: P_L0_16×16 / 16×8 / 8×16.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_p_inter_hi(
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
                    "h264 10bit cabac p-slice: ref_idx_l0 {raw} out of range (num_ref={num_ref_l0})"
                )));
            }
            raw as i8
        } else {
            0
        };
    }

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

    for p in 0..n_parts {
        let (r0, c0, ph, pw) = partition.partition_rect(p);
        let reference = lookup_ref(ref_list0, ref_idxs[p])?;
        let (lw, cw) = l0_weight_for(sh, ref_idxs[p]);
        let mv = mvs[p];
        mc_luma_partition_hi(
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
        mc_chroma_partition_hi(
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
            cw,
        );
    }

    let cbp = decode_cbp_hi(d, ctxs, sps, pic, mb_x, mb_y)?;
    let cbp_luma = (cbp & 0x0F) as u8;
    let cbp_chroma = ((cbp >> 4) & 0x03) as u8;

    let transform_8x8 = if pps.transform_8x8_mode_flag && cbp_luma != 0 {
        let inc = transform_size_8x8_flag_ctx_idx_inc(pic, mb_x, mb_y);
        let slice = &mut ctxs[CTX_IDX_TRANSFORM_SIZE_8X8_FLAG..CTX_IDX_TRANSFORM_SIZE_8X8_FLAG + 3];
        binarize::decode_transform_size_8x8_flag(d, slice, inc)?
    } else {
        false
    };
    if transform_8x8 {
        return Err(Error::unsupported(
            "h264 10bit cabac p-slice: transform_size_8x8_flag=1 not yet wired",
        ));
    }

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
        apply_qp_delta_hi(prev_qp, dqp, sps);
    } else {
        pic.last_mb_qp_delta_was_nonzero = false;
    }
    let qp_y = *prev_qp;
    let qp_y_prime = qp_y + 6 * sps.bit_depth_luma_minus8 as i32;

    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        info.qp_y = qp_y;
        info.coded = true;
        info.intra = false;
        info.skipped = false;
        info.mb_type_p = Some(PMbType::Inter { partition });
        info.p_partition = Some(partition);
        info.intra4x4_pred_mode = [INTRA_DC_FAKE; 16];
        info.transform_8x8 = transform_8x8;
        info.cbp_luma = cbp_luma;
        info.cbp_chroma = cbp_chroma;
    }

    decode_inter_residual_luma_hi(d, ctxs, mb_x, mb_y, pic, cbp_luma, qp_y_prime)?;
    decode_inter_residual_chroma_hi(d, ctxs, sps, pps, mb_x, mb_y, pic, cbp_chroma, qp_y)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// P_8×8 — four 8×8 sub-partitions each with its own sub_mb_type.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_p_8x8_hi(
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
    is_ref0: bool,
) -> Result<()> {
    let num_ref_l0 = sh.num_ref_idx_l0_active_minus1 + 1;

    let mut sub_parts = [PSubPartition::Sub8x8; 4];
    for s in 0..4 {
        let raw = {
            let slice = &mut ctxs[CTX_IDX_SUB_MB_TYPE_P..CTX_IDX_SUB_MB_TYPE_P + 3];
            binarize::decode_sub_mb_type_p(d, slice)?
        };
        sub_parts[s] = crate::mb_type::decode_p_sub_mb_type(raw).ok_or_else(|| {
            Error::invalid(format!("h264 10bit cabac p-slice: bad sub_mb_type {raw}"))
        })?;
    }

    let mut ref_idxs = [0i8; 4];
    if !is_ref0 {
        for s in 0..4 {
            ref_idxs[s] = if num_ref_l0 > 1 {
                let (sr0, sc0) = SUB_OFFSETS[s];
                let inc = cabac_p::ref_idx_ctx_idx_inc_at(pic, mb_x, mb_y, sr0, sc0)?;
                let slice = &mut ctxs[CTX_IDX_REF_IDX_L0..CTX_IDX_REF_IDX_L0 + 6];
                let raw = binarize::decode_ref_idx_lx(d, slice, inc)?;
                if raw >= num_ref_l0 {
                    return Err(Error::invalid(format!(
                        "h264 10bit cabac p-slice: sub_ref_idx_l0 {raw} out of range"
                    )));
                }
                raw as i8
            } else {
                0
            };
        }
    }

    let partition = if is_ref0 {
        PPartition::P8x8Ref0
    } else {
        PPartition::P8x8
    };
    for s in 0..4 {
        let sp = sub_parts[s];
        let (sr0, sc0) = SUB_OFFSETS[s];
        let ref_idx = ref_idxs[s];
        let reference = lookup_ref(ref_list0, ref_idx)?;
        for sub_idx in 0..sp.num_sub_partitions() {
            let (dr, dc, sh_h, sh_w) = sp.sub_rect(sub_idx);
            let r0 = sr0 + dr;
            let c0 = sc0 + dc;
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
            let pmv = predict_mv_l0(pic, mb_x, mb_y, r0, c0, sh_h, sh_w, ref_idx);
            let mv = (
                pmv.0.wrapping_add(mvd_x as i16),
                pmv.1.wrapping_add(mvd_y as i16),
            );
            cabac_p::fill_partition_mv(pic, mb_x, mb_y, r0, c0, sh_h, sh_w, mv, ref_idx);
            cabac_p::fill_partition_mvd_abs(
                pic,
                mb_x,
                mb_y,
                r0,
                c0,
                sh_h,
                sh_w,
                mvd_x.unsigned_abs(),
                mvd_y.unsigned_abs(),
            );
            let (lw, cw) = l0_weight_for(sh, ref_idx);
            mc_luma_partition_hi(
                pic,
                reference,
                mb_x,
                mb_y,
                r0,
                c0,
                sh_h,
                sh_w,
                mv.0 as i32,
                mv.1 as i32,
                lw,
            );
            mc_chroma_partition_hi(
                pic,
                reference,
                mb_x,
                mb_y,
                r0,
                c0,
                sh_h,
                sh_w,
                mv.0 as i32,
                mv.1 as i32,
                cw,
            );
        }
    }

    let cbp = decode_cbp_hi(d, ctxs, sps, pic, mb_x, mb_y)?;
    let cbp_luma = (cbp & 0x0F) as u8;
    let cbp_chroma = ((cbp >> 4) & 0x03) as u8;

    let all_sub_8x8 = sub_parts.iter().all(|sp| sp.is_at_least_8x8());
    let transform_8x8 = if pps.transform_8x8_mode_flag && cbp_luma != 0 && all_sub_8x8 {
        let inc = transform_size_8x8_flag_ctx_idx_inc(pic, mb_x, mb_y);
        let slice = &mut ctxs[CTX_IDX_TRANSFORM_SIZE_8X8_FLAG..CTX_IDX_TRANSFORM_SIZE_8X8_FLAG + 3];
        binarize::decode_transform_size_8x8_flag(d, slice, inc)?
    } else {
        false
    };
    if transform_8x8 {
        return Err(Error::unsupported(
            "h264 10bit cabac p-slice: transform_size_8x8_flag=1 not yet wired",
        ));
    }

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
        apply_qp_delta_hi(prev_qp, dqp, sps);
    } else {
        pic.last_mb_qp_delta_was_nonzero = false;
    }
    let qp_y = *prev_qp;
    let qp_y_prime = qp_y + 6 * sps.bit_depth_luma_minus8 as i32;

    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        info.qp_y = qp_y;
        info.coded = true;
        info.intra = false;
        info.skipped = false;
        info.mb_type_p = Some(PMbType::Inter { partition });
        info.p_partition = Some(partition);
        info.intra4x4_pred_mode = [INTRA_DC_FAKE; 16];
        info.transform_8x8 = transform_8x8;
        info.cbp_luma = cbp_luma;
        info.cbp_chroma = cbp_chroma;
    }

    decode_inter_residual_luma_hi(d, ctxs, mb_x, mb_y, pic, cbp_luma, qp_y_prime)?;
    decode_inter_residual_chroma_hi(d, ctxs, sps, pps, mb_x, mb_y, pic, cbp_chroma, qp_y)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared CBP decode wrapper.
// ---------------------------------------------------------------------------

fn decode_cbp_hi(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sps: &Sps,
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
) -> Result<u32> {
    let slice = &mut ctxs[CTX_IDX_CODED_BLOCK_PATTERN_LUMA..CTX_IDX_CODED_BLOCK_PATTERN_LUMA + 12];
    binarize::decode_coded_block_pattern(
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
    )
}

// ---------------------------------------------------------------------------
// Inter residual — luma 4×4 and chroma 4:2:0 on u16 planes.
// ---------------------------------------------------------------------------

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
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
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
        let neighbours = cabac_p::cbf_neighbours_luma(pic, mb_x, mb_y, br_row, br_col);
        let coeffs =
            decode_residual_block_in_place(d, ctxs, BlockCat::Luma4x4, &neighbours, 16, false)?;
        let mut residual = coeffs;
        let total_coeff = coeffs.iter().filter(|&&v| v != 0).count() as u32;
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

#[allow(clippy::too_many_arguments)]
fn decode_inter_residual_chroma_hi(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
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
        let neigh_cb_dc = cabac_p::p_chroma_dc_cbf_neighbours(pic, mb_x, mb_y, 0);
        let cb_coeffs =
            decode_residual_block_in_place(d, ctxs, BlockCat::ChromaDc, &neigh_cb_dc, 4, false)?;
        for i in 0..4 {
            dc_cb[i] = cb_coeffs[i];
        }
        pic.mb_info_mut(mb_x, mb_y).chroma_dc_cbf[0] = cb_coeffs.iter().any(|&v| v != 0);
        let neigh_cr_dc = cabac_p::p_chroma_dc_cbf_neighbours(pic, mb_x, mb_y, 1);
        let cr_coeffs =
            decode_residual_block_in_place(d, ctxs, BlockCat::ChromaDc, &neigh_cr_dc, 4, false)?;
        for i in 0..4 {
            dc_cr[i] = cr_coeffs[i];
        }
        pic.mb_info_mut(mb_x, mb_y).chroma_dc_cbf[1] = cr_coeffs.iter().any(|&v| v != 0);
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
                let neigh = cabac_p::p_chroma_ac_cbf_neighbours(
                    pic, mb_x, mb_y, plane_kind, br_row, br_col, &nc_arr,
                );
                let ac =
                    decode_residual_block_in_place(d, ctxs, BlockCat::ChromaAc, &neigh, 15, false)?;
                total_coeff = ac.iter().filter(|&&v| v != 0).count() as u32;
                res = ac;
                let cat = if plane_kind { 4 } else { 5 };
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

// ---------------------------------------------------------------------------
// u16 MC kernels — variants of the CABAC 8-bit helpers in `cabac/p_mb.rs`.
// ---------------------------------------------------------------------------

fn lookup_ref<'a>(ref_list0: &[&'a Picture], ref_idx: i8) -> Result<&'a Picture> {
    if ref_idx < 0 {
        return Err(Error::invalid(
            "h264 10bit cabac p-slice: negative ref_idx_l0",
        ));
    }
    let idx = ref_idx as usize;
    ref_list0.get(idx).copied().ok_or_else(|| {
        Error::invalid(format!(
            "h264 10bit cabac p-slice: ref_idx_l0 {} out of range (len={})",
            ref_idx,
            ref_list0.len()
        ))
    })
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
        &mut tmp_cb,
        &reference.cb16,
        cstride,
        cw_plane,
        ch_plane,
        base_x,
        base_y,
        mv_x_q,
        mv_y_q,
        pw,
        ph,
        reference.bit_depth_c,
    );
    chroma_mc_hi(
        &mut tmp_cr,
        &reference.cr16,
        cstride,
        cw_plane,
        ch_plane,
        base_x,
        base_y,
        mv_x_q,
        mv_y_q,
        pw,
        ph,
        reference.bit_depth_c,
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
    (tbl.luma_l0.get(idx), tbl.chroma_l0.get(idx))
}
