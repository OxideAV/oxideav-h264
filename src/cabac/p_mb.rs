//! CABAC P-slice macroblock decode — ITU-T H.264 §7.3.5 + §9.3.
//!
//! Mirrors [`crate::p_mb::decode_p_slice_mb`] (the CAVLC P path) but reads
//! every syntax element through the CABAC engine. The motion-compensation,
//! MV prediction, and pixel reconstruction stages are shared verbatim with
//! the CAVLC path — only the entropy layer differs.
//!
//! Coverage (matching CAVLC P):
//!   * `P_Skip` — signalled by `mb_skip_flag = 1`; MV from §8.4.1.1.
//!   * `P_L0_16×16`, `P_L0_16×8`, `P_L0_8×16`.
//!   * `P_8x8` (with `sub_mb_type` ∈ {8×8, 8×4, 4×8, 4×4}).
//!   * Intra-in-P (`mb_type >= 5`), dispatched to the CABAC I-slice path.
//!
//! Out of scope — surfaces `Error::Unsupported`:
//!   * B-slice structures, MBAFF, 4:2:2 / 4:4:4.

use oxideav_core::{Error, Result};

use crate::cabac::binarize;
use crate::cabac::context::CabacContext;
use crate::cabac::engine::CabacDecoder;
use crate::cabac::mb::{
    decode_luma_8x8_residual_in_place, decode_residual_block_in_place,
    transform_size_8x8_flag_ctx_idx_inc,
};
use crate::cabac::residual::{BlockCat, CbfNeighbours};
use crate::cabac::tables::{
    CTX_IDX_CODED_BLOCK_PATTERN_LUMA, CTX_IDX_MB_QP_DELTA, CTX_IDX_MB_SKIP_FLAG_P,
    CTX_IDX_MB_TYPE_I, CTX_IDX_MB_TYPE_P, CTX_IDX_REF_IDX_L0, CTX_IDX_SUB_MB_TYPE_P,
    CTX_IDX_TRANSFORM_SIZE_8X8_FLAG,
};
use crate::mb::LUMA_BLOCK_RASTER;
use crate::mb_type::{IMbType, PMbType, PPartition, PSubPartition};
use crate::motion::{apply_chroma_weight, apply_luma_weight, chroma_mc, luma_mc, predict_mv_l0};
use crate::picture::{Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::{ChromaWeight, LumaWeight, SliceHeader};
use crate::sps::Sps;
use crate::transform::{
    chroma_qp, dequantize_4x4_scaled, dequantize_8x8_scaled, idct_4x4, idct_8x8,
    inv_hadamard_2x2_chroma_dc_scaled,
};

// MVD contexts are laid out as two 7-slot banks for list-0 components:
//   MVD_L0_X at ctxIdxOffset 40..=46
//   MVD_L0_Y at ctxIdxOffset 47..=53
const CTX_IDX_MVD_L0_X: usize = 40;
const CTX_IDX_MVD_L0_Y: usize = 47;
// transform_size_8x8_flag lives in the 460..1024 High-profile context
// range; we don't store it in tables.rs yet — refuse if the bit is set.

/// Decode one P-slice macroblock. Assumes the caller has already parsed
/// the slice header and placed the CABAC engine at the start of the
/// macroblock layer data (post any previous `end_of_slice_flag`).
///
/// On entry, `prev_qp` is the running QP_Y. On return the MB pixel buffer
/// has been filled and `prev_qp` reflects any `mb_qp_delta` that was read.
#[allow(clippy::too_many_arguments)]
pub fn decode_p_mb_cabac(
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
    // §9.3.3.1.1.1 — mb_skip_flag.
    let skip_inc = mb_skip_flag_ctx_idx_inc(pic, mb_x, mb_y);
    let skipped = {
        let slice = &mut ctxs[CTX_IDX_MB_SKIP_FLAG_P..CTX_IDX_MB_SKIP_FLAG_P + 3];
        binarize::decode_mb_skip_flag_p(d, slice, skip_inc)?
    };

    if skipped {
        decode_p_skip(sh, mb_x, mb_y, pic, ref_list0, *prev_qp)?;
        return Ok(true);
    }

    // §9.3.3.1.1.3 — mb_type for P/SP. Bin 0 ctxIdxInc is independent of
    // neighbours; subsequent bins within the P bank use fixed ctxIdxInc.
    // Intra-in-P bins share the I-slice ctxIdxOffset (CTX_IDX_MB_TYPE_I).
    let mb_type_raw = {
        let (p_slice, i_slice) = if CTX_IDX_MB_TYPE_P < CTX_IDX_MB_TYPE_I {
            // CTX_IDX_MB_TYPE_P (14) > CTX_IDX_MB_TYPE_I (3) in this crate,
            // so this branch is unreachable — kept for clarity.
            let (head, tail) = ctxs.split_at_mut(CTX_IDX_MB_TYPE_I);
            (
                &mut head[CTX_IDX_MB_TYPE_P..CTX_IDX_MB_TYPE_P + 7],
                &mut tail[..8],
            )
        } else {
            let (head, tail) = ctxs.split_at_mut(CTX_IDX_MB_TYPE_P);
            (
                &mut tail[..7],
                &mut head[CTX_IDX_MB_TYPE_I..CTX_IDX_MB_TYPE_I + 8],
            )
        };
        binarize::decode_mb_type_p(d, p_slice, i_slice)?
    };

    match mb_type_raw {
        0 => decode_p_inter(
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
        1 => decode_p_inter(
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
        2 => decode_p_inter(
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
        3 => decode_p_8x8(
            d, ctxs, sh, sps, pps, mb_x, mb_y, pic, ref_list0, prev_qp, false,
        ),
        5..=30 => {
            // Intra-in-P: dispatch to the CABAC I path with the already-
            // parsed IMbType.
            let imb = crate::mb_type::decode_i_slice_mb_type(mb_type_raw - 5).ok_or_else(|| {
                Error::invalid(format!(
                    "h264 cabac p-slice: bad intra-in-P mb_type {mb_type_raw}"
                ))
            })?;
            decode_intra_in_p(d, ctxs, sh, sps, pps, mb_x, mb_y, pic, prev_qp, imb)
        }
        _ => Err(Error::invalid(format!(
            "h264 cabac p-slice: unexpected mb_type {mb_type_raw}"
        ))),
    }?;
    Ok(false)
}

// ---------------------------------------------------------------------------
// P_Skip.
// ---------------------------------------------------------------------------

fn decode_p_skip(
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    prev_qp: i32,
) -> Result<()> {
    // Reuse the CAVLC P_Skip code path verbatim — the motion-compensation
    // and metadata bookkeeping are identical regardless of entropy mode.
    crate::p_mb::decode_p_skip_mb(sh, mb_x, mb_y, pic, ref_list0, prev_qp)
}

// ---------------------------------------------------------------------------
// Inter partitions: P_L0_16×16, P_L0_16×8, P_L0_8×16.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_p_inter(
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
    // Read ref_idx_l0 for each partition (only when more than one ref).
    for p in 0..n_parts {
        ref_idxs[p] = if num_ref_l0 > 1 {
            let inc = ref_idx_ctx_idx_inc(pic, mb_x, mb_y, partition, p)?;
            let slice = &mut ctxs[CTX_IDX_REF_IDX_L0..CTX_IDX_REF_IDX_L0 + 6];
            let raw = binarize::decode_ref_idx_lx(d, slice, inc)?;
            if raw >= num_ref_l0 {
                return Err(Error::invalid(format!(
                    "h264 cabac p-slice: ref_idx_l0 {raw} out of range (num_ref={num_ref_l0})"
                )));
            }
            raw as i8
        } else {
            0
        };
    }

    // Read MVDs + apply MV prediction + motion compensation.
    let mut mvs = [(0i16, 0i16); 4];
    let mut mvds = [(0i32, 0i32); 4];
    for p in 0..n_parts {
        let (r0, c0, ph, pw) = partition.partition_rect(p);
        let inc_x = mvd_ctx_idx_inc(pic, mb_x, mb_y, r0, c0, true);
        let inc_y = mvd_ctx_idx_inc(pic, mb_x, mb_y, r0, c0, false);
        let mvd_x = {
            let slice = &mut ctxs[CTX_IDX_MVD_L0_X..CTX_IDX_MVD_L0_X + 7];
            binarize::decode_mvd_component(d, slice, inc_x)?
        };
        let mvd_y = {
            let slice = &mut ctxs[CTX_IDX_MVD_L0_Y..CTX_IDX_MVD_L0_Y + 7];
            binarize::decode_mvd_component(d, slice, inc_y)?
        };
        mvds[p] = (mvd_x, mvd_y);

        let pmv = predict_mv_l0(pic, mb_x, mb_y, r0, c0, ph, pw, ref_idxs[p]);
        let mv = (
            pmv.0.wrapping_add(mvd_x as i16),
            pmv.1.wrapping_add(mvd_y as i16),
        );
        mvs[p] = mv;
        fill_partition_mv(pic, mb_x, mb_y, r0, c0, ph, pw, mv, ref_idxs[p]);
        fill_partition_mvd_abs(
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

    // Motion-compensation happens once all partition MVs are known so the
    // pixel writes don't collide with neighbour MV state reads.
    for p in 0..n_parts {
        let (r0, c0, ph, pw) = partition.partition_rect(p);
        let reference = lookup_ref(ref_list0, ref_idxs[p])?;
        let (lw, cw) = l0_weight_for(sh, ref_idxs[p]);
        let mv = mvs[p];
        mc_luma_partition(
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
        mc_chroma_partition(
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

    // §9.3.3.1.1.4 — coded_block_pattern.
    let cbp = {
        let slice =
            &mut ctxs[CTX_IDX_CODED_BLOCK_PATTERN_LUMA..CTX_IDX_CODED_BLOCK_PATTERN_LUMA + 12];
        binarize::decode_coded_block_pattern(
            d,
            slice,
            sps.chroma_format_idc as u8,
            |i, decoded| super::mb::cbp_luma_ctx_idx_inc(pic, mb_x, mb_y, i, decoded),
            |bin| {
                if bin == 0 {
                    super::mb::cbp_chroma_ctx_idx_inc_any(pic, mb_x, mb_y)
                } else {
                    super::mb::cbp_chroma_ctx_idx_inc_ac(pic, mb_x, mb_y)
                }
            },
        )?
    };
    let cbp_luma = (cbp & 0x0F) as u8;
    let cbp_chroma = ((cbp >> 4) & 0x03) as u8;

    // §7.3.5.1 — `transform_size_8x8_flag` is present for P inter MBs
    // (non-P_8x8) when the PPS enables 8×8 AND cbp_luma != 0. Partition
    // types 16×16 / 16×8 / 8×16 all qualify (no sub-8×8 motion splits).
    let transform_8x8 = if pps.transform_8x8_mode_flag && cbp_luma != 0 {
        let inc = transform_size_8x8_flag_ctx_idx_inc(pic, mb_x, mb_y);
        let slice =
            &mut ctxs[CTX_IDX_TRANSFORM_SIZE_8X8_FLAG..CTX_IDX_TRANSFORM_SIZE_8X8_FLAG + 3];
        binarize::decode_transform_size_8x8_flag(d, slice, inc)?
    } else {
        false
    };

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
        *prev_qp = ((*prev_qp + dqp + 52) % 52).clamp(0, 51);
    } else {
        pic.last_mb_qp_delta_was_nonzero = false;
    }
    let qp_y = *prev_qp;

    // Update MbInfo before residual decode — neighbour nC reads depend on it.
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

    if transform_8x8 {
        decode_inter_residual_luma_8x8(d, ctxs, mb_x, mb_y, pic, cbp_luma, qp_y)?;
    } else {
        decode_inter_residual_luma(d, ctxs, mb_x, mb_y, pic, cbp_luma, qp_y)?;
    }
    decode_inter_residual_chroma(d, ctxs, pps, mb_x, mb_y, pic, cbp_chroma, qp_y)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// P_8x8 — four 8×8 sub-partitions each with its own `sub_mb_type`.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_p_8x8(
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

    // Four sub_mb_type values first.
    let mut sub_parts = [PSubPartition::Sub8x8; 4];
    for s in 0..4 {
        let raw = {
            let slice = &mut ctxs[CTX_IDX_SUB_MB_TYPE_P..CTX_IDX_SUB_MB_TYPE_P + 3];
            binarize::decode_sub_mb_type_p(d, slice)?
        };
        sub_parts[s] = crate::mb_type::decode_p_sub_mb_type(raw)
            .ok_or_else(|| Error::invalid(format!("h264 cabac p-slice: bad sub_mb_type {raw}")))?;
    }

    // ref_idx_l0 for each 8×8 sub-MB.
    let mut ref_idxs = [0i8; 4];
    if !is_ref0 {
        for s in 0..4 {
            ref_idxs[s] = if num_ref_l0 > 1 {
                let (sr0, sc0) = SUB_OFFSETS[s];
                // ref_idx neighbour derivation uses the sub-MB's top-left block.
                let inc = ref_idx_ctx_idx_inc_at(pic, mb_x, mb_y, sr0, sc0)?;
                let slice = &mut ctxs[CTX_IDX_REF_IDX_L0..CTX_IDX_REF_IDX_L0 + 6];
                let raw = binarize::decode_ref_idx_lx(d, slice, inc)?;
                if raw >= num_ref_l0 {
                    return Err(Error::invalid(format!(
                        "h264 cabac p-slice: sub_ref_idx_l0 {raw} out of range"
                    )));
                }
                raw as i8
            } else {
                0
            };
        }
    }

    // Read MVDs + do MV prediction for every sub-partition.
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
            let inc_x = mvd_ctx_idx_inc(pic, mb_x, mb_y, r0, c0, true);
            let inc_y = mvd_ctx_idx_inc(pic, mb_x, mb_y, r0, c0, false);
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
            fill_partition_mv(pic, mb_x, mb_y, r0, c0, sh_h, sh_w, mv, ref_idx);
            fill_partition_mvd_abs(
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
            mc_luma_partition(
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
            mc_chroma_partition(
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

    let cbp = {
        let slice =
            &mut ctxs[CTX_IDX_CODED_BLOCK_PATTERN_LUMA..CTX_IDX_CODED_BLOCK_PATTERN_LUMA + 12];
        binarize::decode_coded_block_pattern(
            d,
            slice,
            sps.chroma_format_idc as u8,
            |i, decoded| super::mb::cbp_luma_ctx_idx_inc(pic, mb_x, mb_y, i, decoded),
            |bin| {
                if bin == 0 {
                    super::mb::cbp_chroma_ctx_idx_inc_any(pic, mb_x, mb_y)
                } else {
                    super::mb::cbp_chroma_ctx_idx_inc_ac(pic, mb_x, mb_y)
                }
            },
        )?
    };
    let cbp_luma = (cbp & 0x0F) as u8;
    let cbp_chroma = ((cbp >> 4) & 0x03) as u8;

    // §7.3.5.1 — P_8x8's `transform_size_8x8_flag` is gated on all four
    // sub_mb_types being at least 8×8 (no sub-8×8 split would straddle an
    // 8×8 transform). Matches the CAVLC gate in `p_mb::decode_p8x8`.
    let all_sub_8x8 = sub_parts.iter().all(|sp| sp.is_at_least_8x8());
    let transform_8x8 =
        if pps.transform_8x8_mode_flag && cbp_luma != 0 && all_sub_8x8 {
            let inc = transform_size_8x8_flag_ctx_idx_inc(pic, mb_x, mb_y);
            let slice =
                &mut ctxs[CTX_IDX_TRANSFORM_SIZE_8X8_FLAG..CTX_IDX_TRANSFORM_SIZE_8X8_FLAG + 3];
            binarize::decode_transform_size_8x8_flag(d, slice, inc)?
        } else {
            false
        };

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
        *prev_qp = ((*prev_qp + dqp + 52) % 52).clamp(0, 51);
    } else {
        pic.last_mb_qp_delta_was_nonzero = false;
    }
    let qp_y = *prev_qp;

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

    if transform_8x8 {
        decode_inter_residual_luma_8x8(d, ctxs, mb_x, mb_y, pic, cbp_luma, qp_y)?;
    } else {
        decode_inter_residual_luma(d, ctxs, mb_x, mb_y, pic, cbp_luma, qp_y)?;
    }
    decode_inter_residual_chroma(d, ctxs, pps, mb_x, mb_y, pic, cbp_chroma, qp_y)?;
    Ok(())
}

const SUB_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 2), (2, 0), (2, 2)];

// ---------------------------------------------------------------------------
// Intra-in-P: redirect to the CABAC I path's helpers.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_intra_in_p(
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
    super::mb::decode_intra_in_p_cabac(d, ctxs, sh, sps, pps, mb_x, mb_y, pic, prev_qp, imb)
}

// ---------------------------------------------------------------------------
// Inter residual — luma 4×4 and chroma.
// ---------------------------------------------------------------------------

fn decode_inter_residual_luma(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    cbp_luma: u8,
    qp_y: i32,
) -> Result<()> {
    let lstride = pic.luma_stride();
    let lo_mb = pic.luma_off(mb_x, mb_y);
    for blk in 0..16usize {
        let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
        let eight_idx = (br_row / 2) * 2 + (br_col / 2);
        if (cbp_luma >> eight_idx) & 1 == 0 {
            continue;
        }
        let neighbours = cbf_neighbours_luma(pic, mb_x, mb_y, br_row, br_col);
        let coeffs =
            decode_residual_block_in_place(d, ctxs, BlockCat::Luma4x4, &neighbours, 16, false)?;
        let mut residual = coeffs;
        let total_coeff = coeffs.iter().filter(|&&v| v != 0).count() as u32;
        // §7.4.2.2 — Inter-Y 4×4 slot 3.
        let scale = *pic.scaling_lists.matrix_4x4(3);
        dequantize_4x4_scaled(&mut residual, qp_y, &scale);
        idct_4x4(&mut residual);
        let lo = lo_mb + br_row * 4 * lstride + br_col * 4;
        for r in 0..4 {
            for c in 0..4 {
                let base = pic.y[lo + r * lstride + c] as i32;
                pic.y[lo + r * lstride + c] = (base + residual[r * 4 + c]).clamp(0, 255) as u8;
            }
        }
        pic.mb_info_mut(mb_x, mb_y).luma_nc[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}

// Inter luma residual with `transform_size_8x8_flag = 1`. Mirrors the
// CAVLC path (`p_mb::decode_inter_residual_luma_8x8`) but pulls the
// 64-coefficient block through `decode_luma_8x8_residual_in_place`
// (ctxBlockCat = 5, §9.3.3.1.1.9). `luma_nc` is populated per-4×4 with a
// consolidated NNZ so neighbour MV/mode derivations on subsequent MBs see
// a consistent "this 8×8 had residual" signal.
fn decode_inter_residual_luma_8x8(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    cbp_luma: u8,
    qp_y: i32,
) -> Result<()> {
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

        let neighbours = cbf_neighbours_luma(pic, mb_x, mb_y, r0, c0);
        let coeffs = decode_luma_8x8_residual_in_place(d, ctxs, &neighbours, false)?;
        let mut residual = coeffs;
        let nnz = residual.iter().filter(|&&v| v != 0).count() as u16;
        // §7.4.2.2 — Inter-Y 8×8 uses scaling-list slot 1.
        let scale = *pic.scaling_lists.matrix_8x8(1);
        dequantize_8x8_scaled(&mut residual, qp_y, &scale);
        idct_8x8(&mut residual);

        let info = pic.mb_info_mut(mb_x, mb_y);
        let byte = nnz.min(16) as u8;
        info.luma_nc[r0 * 4 + c0] = byte;
        info.luma_nc[r0 * 4 + c0 + 1] = byte;
        info.luma_nc[(r0 + 1) * 4 + c0] = byte;
        info.luma_nc[(r0 + 1) * 4 + c0 + 1] = byte;

        let lo = lo_mb + (br8_row * 8) * lstride + br8_col * 8;
        for r in 0..8 {
            for c in 0..8 {
                let base = pic.y[lo + r * lstride + c] as i32;
                let v = base + residual[r * 8 + c];
                pic.y[lo + r * lstride + c] = v.clamp(0, 255) as u8;
            }
        }
    }
    Ok(())
}

fn decode_inter_residual_chroma(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    cbp_chroma: u8,
    qp_y: i32,
) -> Result<()> {
    let qpc = chroma_qp(qp_y, pps.chroma_qp_index_offset);
    let mut dc_cb = [0i32; 4];
    let mut dc_cr = [0i32; 4];
    if cbp_chroma >= 1 {
        // §9.3.3.1.1.9 — chroma DC CBF neighbour per plane from
        // MbInfo::chroma_dc_cbf (same as the I-slice path).
        let neigh_cb_dc = p_chroma_dc_cbf_neighbours(pic, mb_x, mb_y, 0);
        let cb_coeffs =
            decode_residual_block_in_place(d, ctxs, BlockCat::ChromaDc, &neigh_cb_dc, 4, false)?;
        for i in 0..4 { dc_cb[i] = cb_coeffs[i]; }
        pic.mb_info_mut(mb_x, mb_y).chroma_dc_cbf[0] = cb_coeffs.iter().any(|&v| v != 0);
        let neigh_cr_dc = p_chroma_dc_cbf_neighbours(pic, mb_x, mb_y, 1);
        let cr_coeffs =
            decode_residual_block_in_place(d, ctxs, BlockCat::ChromaDc, &neigh_cr_dc, 4, false)?;
        for i in 0..4 { dc_cr[i] = cr_coeffs[i]; }
        pic.mb_info_mut(mb_x, mb_y).chroma_dc_cbf[1] = cr_coeffs.iter().any(|&v| v != 0);
        // CABAC P-slice chroma is inter.
        let w_cb = pic.scaling_lists.matrix_4x4(4)[0];
        let w_cr = pic.scaling_lists.matrix_4x4(5)[0];
        inv_hadamard_2x2_chroma_dc_scaled(&mut dc_cb, qpc, w_cb);
        inv_hadamard_2x2_chroma_dc_scaled(&mut dc_cr, qpc, w_cr);
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
                let neigh = p_chroma_ac_cbf_neighbours(pic, mb_x, mb_y, plane_kind, br_row, br_col, &nc_arr);
                let ac =
                    decode_residual_block_in_place(d, ctxs, BlockCat::ChromaAc, &neigh, 15, false)?;
                total_coeff = ac.iter().filter(|&&v| v != 0).count() as u32;
                res = ac;
                // §7.4.2.2 — Inter chroma slots 4 (Cb) / 5 (Cr).
                let cat = if plane_kind { 4 } else { 5 };
                let scale = *pic.scaling_lists.matrix_4x4(cat);
                dequantize_4x4_scaled(&mut res, qpc, &scale);
            }
            res[0] = dc[(br_row << 1) | br_col];
            idct_4x4(&mut res);
            let off_in_mb = br_row * 4 * cstride + br_col * 4;
            let plane = if plane_kind { &mut pic.cb } else { &mut pic.cr };
            for r in 0..4 {
                for c in 0..4 {
                    let base = plane[co + off_in_mb + r * cstride + c] as i32;
                    plane[co + off_in_mb + r * cstride + c] =
                        (base + res[r * 4 + c]).clamp(0, 255) as u8;
                }
            }
            nc_arr[(br_row << 1) | br_col] = total_coeff as u8;
            let info = pic.mb_info_mut(mb_x, mb_y);
            if plane_kind {
                info.cb_nc = nc_arr;
            } else {
                info.cr_nc = nc_arr;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ctxIdxInc derivation helpers.
// ---------------------------------------------------------------------------

fn mb_skip_flag_ctx_idx_inc(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
    // §9.3.3.1.1.1: condTermFlagN = 0 if mbN is unavailable OR mbN is
    // a P_Skip macroblock; otherwise 1. `ctxIdxInc = A + B`.
    let a = mb_x > 0 && !pic.mb_info_at(mb_x - 1, mb_y).skipped;
    let b = mb_y > 0 && !pic.mb_info_at(mb_x, mb_y - 1).skipped;
    (a as u8) + (b as u8)
}

fn ref_idx_ctx_idx_inc(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    partition: PPartition,
    part_idx: usize,
) -> Result<u8> {
    let (r0, c0, _, _) = partition.partition_rect(part_idx);
    ref_idx_ctx_idx_inc_at(pic, mb_x, mb_y, r0, c0)
}

fn ref_idx_ctx_idx_inc_at(pic: &Picture, mb_x: u32, mb_y: u32, r0: usize, c0: usize) -> Result<u8> {
    // §9.3.3.1.1.2: condTermFlagN = 1 when neighbour block N is inter and
    // has ref_idx_lX > 0; 0 otherwise (including intra and unavailable).
    let a = neighbour_ref_idx_gt_zero(pic, mb_x as i32, mb_y as i32, r0 as i32, c0 as i32 - 1);
    let b = neighbour_ref_idx_gt_zero(pic, mb_x as i32, mb_y as i32, r0 as i32 - 1, c0 as i32);
    Ok((a as u8) + 2 * (b as u8))
}

fn neighbour_ref_idx_gt_zero(pic: &Picture, mb_x: i32, mb_y: i32, row: i32, col: i32) -> bool {
    let (mbx, mby, r, c) = match wrap_neighbour(mb_x, mb_y, row, col, pic.mb_width, pic.mb_height) {
        Some(v) => v,
        None => return false,
    };
    let info = pic.mb_info_at(mbx, mby);
    if !info.coded || info.intra || info.skipped {
        return false;
    }
    info.ref_idx_l0[(r * 4 + c) as usize] > 0
}

fn mvd_ctx_idx_inc(pic: &Picture, mb_x: u32, mb_y: u32, r0: usize, c0: usize, is_x: bool) -> u32 {
    // §9.3.3.1.1.7: absMvdComp(A) + absMvdComp(B).
    let a = neighbour_abs_mvd(
        pic,
        mb_x as i32,
        mb_y as i32,
        r0 as i32,
        c0 as i32 - 1,
        is_x,
    );
    let b = neighbour_abs_mvd(
        pic,
        mb_x as i32,
        mb_y as i32,
        r0 as i32 - 1,
        c0 as i32,
        is_x,
    );
    a as u32 + b as u32
}

fn neighbour_abs_mvd(pic: &Picture, mb_x: i32, mb_y: i32, row: i32, col: i32, is_x: bool) -> u16 {
    let (mbx, mby, r, c) = match wrap_neighbour(mb_x, mb_y, row, col, pic.mb_width, pic.mb_height) {
        Some(v) => v,
        None => return 0,
    };
    let info = pic.mb_info_at(mbx, mby);
    if !info.coded || info.intra || info.skipped {
        return 0;
    }
    let (x, y) = info.mvd_l0_abs[(r * 4 + c) as usize];
    if is_x {
        x
    } else {
        y
    }
}

fn wrap_neighbour(
    mb_x: i32,
    mb_y: i32,
    row: i32,
    col: i32,
    mb_w: u32,
    mb_h: u32,
) -> Option<(u32, u32, u32, u32)> {
    let mut mbx = mb_x;
    let mut mby = mb_y;
    let mut r = row;
    let mut c = col;
    while c < 0 {
        mbx -= 1;
        c += 4;
    }
    while c >= 4 {
        mbx += 1;
        c -= 4;
    }
    while r < 0 {
        mby -= 1;
        r += 4;
    }
    while r >= 4 {
        mby += 1;
        r -= 4;
    }
    if mbx < 0 || mby < 0 || mbx as u32 >= mb_w || mby as u32 >= mb_h {
        return None;
    }
    Some((mbx as u32, mby as u32, r as u32, c as u32))
}

fn p_chroma_dc_cbf_neighbours(pic: &Picture, mb_x: u32, mb_y: u32, c: usize) -> CbfNeighbours {
    let probe = |mx: u32, my: u32| -> Option<bool> {
        let info = pic.mb_info_at(mx, my);
        if !info.coded {
            return None;
        }
        Some(info.chroma_dc_cbf[c])
    };
    let left = if mb_x > 0 { probe(mb_x - 1, mb_y) } else { None };
    let above = if mb_y > 0 { probe(mb_x, mb_y - 1) } else { None };
    CbfNeighbours { left, above }
}

fn p_chroma_ac_cbf_neighbours(
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

// ---------------------------------------------------------------------------
// MbInfo write helpers + MC adapters (cloned from the CAVLC path so we stay
// entirely within the CABAC entropy driver).
// ---------------------------------------------------------------------------

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

fn fill_partition_mvd_abs(
    pic: &mut Picture,
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    h: usize,
    w: usize,
    abs_x: u32,
    abs_y: u32,
) {
    let info = pic.mb_info_mut(mb_x, mb_y);
    let ax = abs_x.min(u16::MAX as u32) as u16;
    let ay = abs_y.min(u16::MAX as u32) as u16;
    for rr in r0..r0 + h {
        for cc in c0..c0 + w {
            info.mvd_l0_abs[rr * 4 + cc] = (ax, ay);
        }
    }
}

fn lookup_ref<'a>(ref_list0: &[&'a Picture], ref_idx: i8) -> Result<&'a Picture> {
    if ref_idx < 0 {
        return Err(Error::invalid("h264 cabac p-slice: negative ref_idx_l0"));
    }
    let idx = ref_idx as usize;
    ref_list0.get(idx).copied().ok_or_else(|| {
        Error::invalid(format!(
            "h264 cabac p-slice: ref_idx_l0 {} out of range (len={})",
            ref_idx,
            ref_list0.len()
        ))
    })
}

#[allow(clippy::too_many_arguments)]
fn mc_luma_partition(
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
    let mut tmp = vec![0u8; pw * ph];
    let base_x = (mb_x as i32) * 16 + (c0 as i32) * 4;
    let base_y = (mb_y as i32) * 16 + (r0 as i32) * 4;
    luma_mc(&mut tmp, reference, base_x, base_y, mv_x_q, mv_y_q, pw, ph);
    if let Some(lw) = luma_weight {
        apply_luma_weight(&mut tmp, lw);
    }
    let lstride = pic.luma_stride();
    let off = pic.luma_off(mb_x, mb_y) + r0 * 4 * lstride + c0 * 4;
    for rr in 0..ph {
        for cc in 0..pw {
            pic.y[off + rr * lstride + cc] = tmp[rr * pw + cc];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mc_chroma_partition(
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
    let mut tmp_cb = vec![0u8; pw * ph];
    let mut tmp_cr = vec![0u8; pw * ph];
    let base_x = (mb_x as i32) * 8 + (c0 as i32) * 2;
    let base_y = (mb_y as i32) * 8 + (r0 as i32) * 2;
    let cstride = reference.chroma_stride();
    let cw = (reference.width / 2) as i32;
    let ch = (reference.height / 2) as i32;
    chroma_mc(
        &mut tmp_cb,
        &reference.cb,
        cstride,
        cw,
        ch,
        base_x,
        base_y,
        mv_x_q,
        mv_y_q,
        pw,
        ph,
    );
    chroma_mc(
        &mut tmp_cr,
        &reference.cr,
        cstride,
        cw,
        ch,
        base_x,
        base_y,
        mv_x_q,
        mv_y_q,
        pw,
        ph,
    );
    if let Some(cw_entry) = chroma_weight {
        apply_chroma_weight(&mut tmp_cb, cw_entry, 0);
        apply_chroma_weight(&mut tmp_cr, cw_entry, 1);
    }
    let stride = pic.chroma_stride();
    let off = pic.chroma_off(mb_x, mb_y) + r0 * 2 * stride + c0 * 2;
    for rr in 0..ph {
        for cc in 0..pw {
            pic.cb[off + rr * stride + cc] = tmp_cb[rr * pw + cc];
            pic.cr[off + rr * stride + cc] = tmp_cr[rr * pw + cc];
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
