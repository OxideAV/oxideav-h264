//! P-slice macroblock decode — ITU-T H.264 §7.3.5 / §8.4.
//!
//! Baseline-profile subset:
//!
//! * `P_Skip` — `mb_skip_run` counted before the macroblock. No residual,
//!   MV derived from the §8.4.1.1 special predictor, ref_idx = 0.
//! * `P_L0_16×16`, `P_L0_16×8`, `P_L0_8×16`, `P_8x8(sub)`,
//!   `P_8x8_ref0(sub)` — motion-compensated with residual + IDCT. The
//!   sub-partition syntax for P_8x8 supports Sub8x8 / Sub8x4 / Sub4x8 /
//!   Sub4x4 (Table 7-17).
//! * Intra-in-P macroblocks (`mb_type >= 5`) — dispatched back into the
//!   existing I-slice decode helpers.
//!
//! The caller supplies a `RefPicList0` (a slice of `&Picture` built by
//! [`crate::dpb::Dpb`]) and each partition's `ref_idx_l0` indexes directly
//! into it. An out-of-range index is rejected as `Error::InvalidData`.

use oxideav_core::{Error, Result};

use crate::cavlc::{decode_residual_8x8_sub, decode_residual_block, BlockKind};
use crate::golomb::BitReaderExt;
use crate::mb::LUMA_BLOCK_RASTER;
use crate::mb_type::{
    decode_cbp_inter, decode_p_slice_mb_type, decode_p_sub_mb_type, IMbType, PMbType, PPartition,
    PSubPartition,
};
use crate::motion::{
    apply_chroma_weight, apply_luma_weight, chroma_mc, luma_mc, predict_mv_l0, predict_mv_pskip,
};
use crate::picture::{MbInfo, Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::{ChromaWeight, LumaWeight, SliceHeader};
use crate::sps::Sps;
use crate::transform::{
    chroma_qp, dequantize_4x4_scaled, dequantize_8x8_scaled, idct_4x4, idct_8x8,
    inv_hadamard_2x2_chroma_dc_scaled, inv_hadamard_2x4_chroma_dc_scaled,
};
use oxideav_core::bits::BitReader;

/// Decode one coded P-slice macroblock at `(mb_x, mb_y)`. On return the
/// pixel buffer + MbInfo at that address have been filled in; `prev_qp` is
/// updated to reflect the new `QP_Y` (for chained `mb_qp_delta` decoding
/// across macroblocks). The outer slice loop is responsible for emitting
/// `P_Skip` macroblocks via [`decode_p_skip_mb`] — this function starts
/// reading `mb_type` immediately.
pub fn decode_p_slice_mb(
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
        PMbType::IntraInP(imb) => {
            decode_intra_in_p(br, sps, pps, sh, mb_x, mb_y, pic, prev_qp, imb)
        }
        PMbType::Inter { partition } => match partition {
            PPartition::P16x16 => {
                decode_p16x16(br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, prev_qp)
            }
            PPartition::P16x8 => {
                decode_p16x8(br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, prev_qp)
            }
            PPartition::P8x16 => {
                decode_p8x16(br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, prev_qp)
            }
            PPartition::P8x8 | PPartition::P8x8Ref0 => decode_p8x8(
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

/// Apply the `P_Skip` derivation and fill in `(mb_x, mb_y)` — no residual,
/// motion vectors come from §8.4.1.1. The slice header is consulted for the
/// L0 pred-weight-table entry (§8.4.2.3.2); when the slice has no weighted
/// prediction enabled the MC output is written through unchanged.
pub fn decode_p_skip_mb(
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    prev_qp: i32,
) -> Result<()> {
    let reference = lookup_ref(ref_list0, 0)?;
    let mv = predict_mv_pskip(pic, mb_x, mb_y);
    // Reset MB info up-front, then record MV + ref_idx. All 16 sub-blocks
    // share the same MV / ref_idx = 0.
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
    mc_luma_partition(
        pic,
        reference,
        mb_x,
        mb_y,
        0,
        0,
        4,
        4,
        mv.0 as i32,
        mv.1 as i32,
        lw,
    );
    mc_chroma_partition(
        pic,
        reference,
        mb_x,
        mb_y,
        0,
        0,
        4,
        4,
        mv.0 as i32,
        mv.1 as i32,
        cw,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Intra-in-P passthrough.
// ---------------------------------------------------------------------------

fn decode_intra_in_p(
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
    crate::mb::decode_intra_mb_given_imb(br, sps, pps, sh, mb_x, mb_y, pic, prev_qp, imb)
}

// ---------------------------------------------------------------------------
// P_L0_16×16.
// ---------------------------------------------------------------------------

fn decode_p16x16(
    br: &mut BitReader<'_>,
    _sps: &Sps,
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
    let mvd_x = br.read_se()?;
    let mvd_y = br.read_se()?;

    let pmv = predict_mv_l0(pic, mb_x, mb_y, 0, 0, 4, 4, ref_idx);
    let mv = (
        pmv.0.wrapping_add(mvd_x as i16),
        pmv.1.wrapping_add(mvd_y as i16),
    );

    fill_partition_mv(pic, mb_x, mb_y, 0, 0, 4, 4, mv, ref_idx);

    // Predict luma + chroma from reference before residual is added. Apply
    // explicit weighted prediction (§8.4.2.3.2) to the predicted samples
    // when the slice carries a pred_weight_table.
    let (lw, cw) = l0_weight_for(sh, ref_idx);
    mc_luma_partition(
        pic,
        reference,
        mb_x,
        mb_y,
        0,
        0,
        4,
        4,
        mv.0 as i32,
        mv.1 as i32,
        lw,
    );
    mc_chroma_partition(
        pic,
        reference,
        mb_x,
        mb_y,
        0,
        0,
        4,
        4,
        mv.0 as i32,
        mv.1 as i32,
        cw,
    );

    // CBP (inter) + optional transform_8x8_flag + optional qp_delta + residual.
    let cbp_raw = br.read_ue()?;
    let (cbp_luma, cbp_chroma) = decode_cbp_inter(cbp_raw)
        .ok_or_else(|| Error::invalid(format!("h264 p-slice: bad inter CBP {cbp_raw}")))?;
    // §7.3.5.1: `transform_size_8x8_flag` is present when the PPS enables
    // the 8×8 transform, the MB has at least one coded luma residual block,
    // and the MB is not a P_8x8 with a sub-block smaller than 8×8. P_L0_16×16
    // satisfies the latter trivially (NumMbPart == 1).
    let transform_8x8 = if pps.transform_8x8_mode_flag && cbp_luma != 0 {
        br.read_flag()?
    } else {
        false
    };
    let needs_qp = cbp_luma != 0 || cbp_chroma != 0;
    if needs_qp {
        let dqp = br.read_se()?;
        *prev_qp = ((*prev_qp + dqp + 52) % 52).clamp(0, 51);
    }
    let qp_y = *prev_qp;

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
    }

    if transform_8x8 {
        decode_inter_residual_luma_8x8(br, mb_x, mb_y, pic, cbp_luma, qp_y)?;
    } else {
        decode_inter_residual_luma(br, mb_x, mb_y, pic, cbp_luma, qp_y)?;
    }
    decode_inter_residual_chroma(br, pps, mb_x, mb_y, pic, cbp_chroma, qp_y)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// P_L0_16×8 / P_L0_8×16 / P_8x8 — parse + MC only (residual same path).
// ---------------------------------------------------------------------------

fn decode_p16x8(
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
    decode_two_partition_p(
        br,
        sps,
        pps,
        sh,
        mb_x,
        mb_y,
        pic,
        ref_list0,
        prev_qp,
        &rects,
        &refs,
        PPartition::P16x8,
    )
}

fn decode_p8x16(
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
    decode_two_partition_p(
        br,
        sps,
        pps,
        sh,
        mb_x,
        mb_y,
        pic,
        ref_list0,
        prev_qp,
        &rects,
        &refs,
        PPartition::P8x16,
    )
}

fn decode_two_partition_p(
    br: &mut BitReader<'_>,
    _sps: &Sps,
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
    // First parse both MVDs so neighbour MV state from partition 0 feeds
    // partition 1's prediction correctly.
    let mut mvs = [(0i16, 0i16); 2];
    for p in 0..2 {
        let (r0, c0, ph, pw) = rects[p];
        let reference = lookup_ref(ref_list0, refs[p])?;
        let mvd_x = br.read_se()?;
        let mvd_y = br.read_se()?;
        let pmv = predict_mv_l0(pic, mb_x, mb_y, r0, c0, ph, pw, refs[p]);
        let mv = (
            pmv.0.wrapping_add(mvd_x as i16),
            pmv.1.wrapping_add(mvd_y as i16),
        );
        mvs[p] = mv;
        fill_partition_mv(pic, mb_x, mb_y, r0, c0, ph, pw, mv, refs[p]);
        let (lw, cw) = l0_weight_for(sh, refs[p]);
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

    let cbp_raw = br.read_ue()?;
    let (cbp_luma, cbp_chroma) = decode_cbp_inter(cbp_raw)
        .ok_or_else(|| Error::invalid(format!("h264 p-slice: bad inter CBP {cbp_raw}")))?;
    // §7.3.5.1 — `transform_size_8x8_flag` gates on PPS + `cbp_luma > 0`.
    // P_L0_16×8 / P_L0_8×16 both have NumMbPart == 2 (no sub-8×8 splits),
    // so the "no sub-block smaller than 8×8" clause is trivially true here.
    let transform_8x8 = if pps.transform_8x8_mode_flag && cbp_luma != 0 {
        br.read_flag()?
    } else {
        false
    };
    let needs_qp = cbp_luma != 0 || cbp_chroma != 0;
    if needs_qp {
        let dqp = br.read_se()?;
        *prev_qp = ((*prev_qp + dqp + 52) % 52).clamp(0, 51);
    }
    let qp_y = *prev_qp;

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
    }

    if transform_8x8 {
        decode_inter_residual_luma_8x8(br, mb_x, mb_y, pic, cbp_luma, qp_y)?;
    } else {
        decode_inter_residual_luma(br, mb_x, mb_y, pic, cbp_luma, qp_y)?;
    }
    decode_inter_residual_chroma(br, pps, mb_x, mb_y, pic, cbp_chroma, qp_y)?;
    Ok(())
}

fn decode_p8x8(
    br: &mut BitReader<'_>,
    _sps: &Sps,
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

    // Parse four sub_mb_type values up front — this is spec order.
    let mut sub_parts = [PSubPartition::Sub8x8; 4];
    for s in 0..4 {
        let v = br.read_ue()?;
        sub_parts[s] = decode_p_sub_mb_type(v)
            .ok_or_else(|| Error::invalid(format!("h264 p-slice: bad sub_mb_type {v}")))?;
    }
    // ref_idx_l0 for each of the four 8×8 sub-MBs (unless P_8x8ref0).
    let mut ref_idxs = [0i8; 4];
    if !is_ref0 {
        for s in 0..4 {
            ref_idxs[s] = read_ref_idx(br, num_ref_l0)?;
        }
    }
    // For each sub-MB and each of its sub-partitions, read mvd_l0.
    // sub-MB index 0..=3 maps to a 2×2 layout of 8×8 blocks (raster).
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
            let mvd_x = br.read_se()?;
            let mvd_y = br.read_se()?;
            let pmv = predict_mv_l0(pic, mb_x, mb_y, r0, c0, sh_h, sh_w, ref_idx);
            let mv = (
                pmv.0.wrapping_add(mvd_x as i16),
                pmv.1.wrapping_add(mvd_y as i16),
            );
            fill_partition_mv(pic, mb_x, mb_y, r0, c0, sh_h, sh_w, mv, ref_idx);
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

    let cbp_raw = br.read_ue()?;
    let (cbp_luma, cbp_chroma) = decode_cbp_inter(cbp_raw)
        .ok_or_else(|| Error::invalid(format!("h264 p-slice: bad inter CBP {cbp_raw}")))?;
    // §7.3.5.1 — `transform_size_8x8_flag` is present for a P_8x8 MB only
    // when every sub-MB is at least 8×8 (i.e. `P_L0_8x8` / `Sub8x8`). Any
    // sub-8×8 split would make an 8×8 transform span two motion partitions,
    // which the spec forbids. libx264's `mb_cache_transform_8x8` test
    // (`encoder/macroblock.c`) enforces the same rule.
    let all_sub_8x8 = sub_parts.iter().all(|sp| sp.is_at_least_8x8());
    let transform_8x8 = if pps.transform_8x8_mode_flag && cbp_luma != 0 && all_sub_8x8 {
        br.read_flag()?
    } else {
        false
    };
    let needs_qp = cbp_luma != 0 || cbp_chroma != 0;
    if needs_qp {
        let dqp = br.read_se()?;
        *prev_qp = ((*prev_qp + dqp + 52) % 52).clamp(0, 51);
    }
    let qp_y = *prev_qp;

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
    }

    if transform_8x8 {
        decode_inter_residual_luma_8x8(br, mb_x, mb_y, pic, cbp_luma, qp_y)?;
    } else {
        decode_inter_residual_luma(br, mb_x, mb_y, pic, cbp_luma, qp_y)?;
    }
    decode_inter_residual_chroma(br, pps, mb_x, mb_y, pic, cbp_chroma, qp_y)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Residual decode for inter macroblocks — luma 4×4 and chroma.
// ---------------------------------------------------------------------------

fn decode_inter_residual_luma(
    br: &mut BitReader<'_>,
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
        let nc = predict_inter_nc_luma(pic, mb_x, mb_y, br_row, br_col);
        let blk_parsed = decode_residual_block(br, nc, BlockKind::Luma4x4)?;
        let mut residual = blk_parsed.coeffs;
        let total_coeff = blk_parsed.total_coeff;
        // §7.4.2.2 — Inter-Y 4×4 is slot 3.
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
// intra 8×8 path (`mb::decode_luma_intra_8x8`) but adds the residual on top
// of the already motion-compensated samples in `pic.y` — no intra prediction
// and no (1,2,1) pre-filter of reference samples. CAVLC coding of each 8×8
// is four interleaved 4×4 sub-blocks (§7.3.5.3.2) spliced back into 8×8
// raster via `ZIGZAG_8X8_CAVLC`, with FFmpeg's per-sub-block
// `non_zero_count_cache` update between sub-blocks (`h264_cavlc.c`) and the
// post-decode `nnz[0] += nnz[1] + nnz[8] + nnz[9]` aggregation that lets
// neighbouring MBs read one consolidated count per 8×8 block.
fn decode_inter_residual_luma_8x8(
    br: &mut BitReader<'_>,
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

        let mut residual = [0i32; 64];
        let slots = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)];
        for (sub, &(sr, sc)) in slots.iter().enumerate() {
            let nc = predict_inter_nc_luma(pic, mb_x, mb_y, sr, sc);
            let tc = decode_residual_8x8_sub(br, nc, sub, &mut residual)?;
            pic.mb_info_mut(mb_x, mb_y).luma_nc[sr * 4 + sc] = tc;
        }
        // §7.4.2.2 — 8×8 inter-Y slot 1.
        let scale = *pic.scaling_lists.matrix_8x8(1);
        dequantize_8x8_scaled(&mut residual, qp_y, &scale);
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
                let base = pic.y[lo + r * lstride + c] as i32;
                let v = base + residual[r * 8 + c];
                pic.y[lo + r * lstride + c] = v.clamp(0, 255) as u8;
            }
        }
    }
    Ok(())
}

fn decode_inter_residual_chroma(
    br: &mut BitReader<'_>,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    cbp_chroma: u8,
    qp_y: i32,
) -> Result<()> {
    // §8.5.11 ChromaArrayType dispatch — 4:2:2 uses a 2×4 chroma DC
    // Hadamard and 8 AC blocks per plane instead of 4:2:0's 2×2 + 4.
    if pic.chroma_format_idc == 2 {
        return decode_inter_residual_chroma_422(br, pps, mb_x, mb_y, pic, cbp_chroma, qp_y);
    }
    let qpc = chroma_qp(qp_y, pps.chroma_qp_index_offset);
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
                let nc = predict_inter_nc_chroma(pic, mb_x, mb_y, plane_kind, br_row, br_col);
                let ac = decode_residual_block(br, nc, BlockKind::ChromaAc)?;
                total_coeff = ac.total_coeff;
                res = ac.coeffs;
                // §7.4.2.2 — Inter-Cb/Cr (slots 4/5). P-slice chroma is inter.
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
                    let v = (base + res[r * 4 + c]).clamp(0, 255) as u8;
                    plane[co + off_in_mb + r * cstride + c] = v;
                }
            }
            nc_arr[(br_row << 1) | br_col] = total_coeff as u8;
            // Publish the in-progress nc totals back to `MbInfo` after every
            // block so the next block's `predict_inter_nc_chroma` sees the
            // up-to-date counts (§9.2.1.1 neighbour prediction).
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

// 4:2:2 inter chroma residual (§8.5.11.2, §9.2.1.2 ChromaArrayType == 2).
//
// Same cbp_chroma encoding as 4:2:0 (0 = none, 1 = DC only, 2 = DC + AC),
// but the plane is 8×16 samples = 2 cols × 4 rows of 4×4 AC blocks and
// the chroma DC block is 2×4 (CAVLC kind `ChromaDc2x4`, nC = -2) fed
// through `inv_hadamard_2x4_chroma_dc_scaled` with the Amendment-2
// `QP'_C,DC = QP'_C + 3` offset baked in. For P-slices the chroma is
// always inter → residual uses inter-Cb/Cr scaling slots 4/5.
fn decode_inter_residual_chroma_422(
    br: &mut BitReader<'_>,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    cbp_chroma: u8,
    qp_y: i32,
) -> Result<()> {
    let qpc_cb = chroma_qp(qp_y, pps.chroma_qp_index_offset);
    let qpc_cr = chroma_qp(qp_y, pps.second_chroma_qp_index_offset);
    let mut dc_cb = [0i32; 8];
    let mut dc_cr = [0i32; 8];
    if cbp_chroma >= 1 {
        let blk = decode_residual_block(br, -2, BlockKind::ChromaDc2x4)?;
        dc_cb.copy_from_slice(&blk.coeffs[..8]);
        let blk = decode_residual_block(br, -2, BlockKind::ChromaDc2x4)?;
        dc_cr.copy_from_slice(&blk.coeffs[..8]);
        // P-slice chroma is inter → slot 4 (Cb) / slot 5 (Cr).
        let w_cb = pic.scaling_lists.matrix_4x4(4)[0];
        let w_cr = pic.scaling_lists.matrix_4x4(5)[0];
        inv_hadamard_2x4_chroma_dc_scaled(&mut dc_cb, qpc_cb, w_cb);
        inv_hadamard_2x4_chroma_dc_scaled(&mut dc_cr, qpc_cr, w_cr);
    }
    let cstride = pic.chroma_stride();
    let co = pic.chroma_off(mb_x, mb_y);
    for plane_cb in [true, false] {
        let dc = if plane_cb { &dc_cb } else { &dc_cr };
        let qpc = if plane_cb { qpc_cb } else { qpc_cr };
        // 4:2:2 AC grid: 4 rows × 2 cols of 4×4 blocks decoded in
        // row-major order so left/top neighbour NC lookups see
        // already-published counts (§9.2.1.1).
        let mut nc_arr = [0u8; 8];
        for blk_row in 0..4usize {
            for blk_col in 0..2usize {
                let blk_idx = blk_row * 2 + blk_col;
                let mut res = [0i32; 16];
                let mut total_coeff = 0u32;
                if cbp_chroma == 2 {
                    let nc =
                        predict_inter_nc_chroma_422(pic, mb_x, mb_y, plane_cb, blk_row, blk_col);
                    let ac = decode_residual_block(br, nc, BlockKind::ChromaAc)?;
                    total_coeff = ac.total_coeff;
                    res = ac.coeffs;
                    let cat = if plane_cb { 4 } else { 5 };
                    let scale = *pic.scaling_lists.matrix_4x4(cat);
                    dequantize_4x4_scaled(&mut res, qpc, &scale);
                }
                res[0] = dc[blk_idx];
                idct_4x4(&mut res);
                let off_in_mb = blk_row * 4 * cstride + blk_col * 4;
                let plane = if plane_cb { &mut pic.cb } else { &mut pic.cr };
                for r in 0..4 {
                    for c in 0..4 {
                        let base = plane[co + off_in_mb + r * cstride + c] as i32;
                        plane[co + off_in_mb + r * cstride + c] =
                            (base + res[r * 4 + c]).clamp(0, 255) as u8;
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

// §9.2.1.1 neighbour-nC predictor for 4:2:2 chroma AC blocks in an
// inter MB — the 2×4 block grid mirrors the intra 4:2:2 layout and
// `MbInfo::{cb,cr}_nc[blk_row*2 + blk_col]` slots store the per-block
// total_coeff so cross-MB left/top reads see the correct value.
fn predict_inter_nc_chroma_422(
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
    } else if mb_y > 0 {
        let info = pic.mb_info_at(mb_x, mb_y - 1);
        if info.coded {
            Some(pick(info, 3 * 2 + blk_col))
        } else {
            None
        }
    } else {
        None
    };
    crate::mb::nc_from_neighbours(left, top)
}

// ---------------------------------------------------------------------------
// Helpers — ref_idx reading, nC prediction, motion compensation glue.
// ---------------------------------------------------------------------------

fn read_ref_idx(br: &mut BitReader<'_>, num_ref: u32) -> Result<i8> {
    // `ref_idx_l0` is te(v): when `num_ref <= 2` it's a single-bit flag
    // (0 = ref 0, 1 = ref 1); otherwise it's ue(v).
    if num_ref == 0 {
        return Ok(0);
    }
    if num_ref == 1 {
        // Only one ref is active. Spec: single-bit te(v) — but with
        // `num_ref == 1` the spec skips the syntax entirely; callers pass
        // num_ref = 1 only when `num_ref_idx_l0_active_minus1 == 0`.
        return Ok(0);
    }
    if num_ref == 2 {
        let bit = br.read_u1()?;
        // te(v) with range [0..1]: 1 → ref 0, 0 → ref 1.
        return Ok(if bit == 1 { 0 } else { 1 });
    }
    Ok(br.read_ue()? as i8)
}

/// Index `ref_list0` by `ref_idx`, returning a reference to the `Picture`
/// or an `Error::InvalidData` when the bitstream addresses an index past
/// the slice's active list.
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
    // §8.4.2.2.1 ChromaArrayType dispatch. 4:2:2 keeps the same
    // horizontal chroma subsampling as 4:2:0 but the chroma height
    // matches luma height; the 1/8-pel bilinear filter still applies
    // but the MV y-component is divided by 4 (not 8) and its fraction
    // is `(mvLX[1] & 3) << 1` so fractional vertical positions only
    // span 0,2,4,6 (half the precision of 4:2:0).
    if pic.chroma_format_idc == 2 {
        mc_chroma_partition_422(
            pic,
            reference,
            mb_x,
            mb_y,
            r0,
            c0,
            h,
            w,
            mv_x_q,
            mv_y_q,
            chroma_weight,
        );
        return;
    }
    // Luma partition is (4h × 4w) samples at (mb*16 + r0*4, mb*16 + c0*4).
    // Chroma 4:2:0 — halve all coordinates; partition is (2h × 2w) samples.
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

// 4:2:2 chroma motion compensation (§8.4.2.2.1 ChromaArrayType == 2).
//
// Chroma width is half of luma width (same as 4:2:0) but chroma height
// equals luma height. Spec fractional derivation:
//   xFracC = mvLX[0] & 7;
//   yFracC = (mvLX[1] & 3) << 1;
//   xIntC  = xA + (mvLX[0] >> 3);
//   yIntC  = yA + (mvLX[1] >> 2);
// The 1/8-pel bilinear chroma filter is unchanged — the caller
// scales `yFracC` into the filter's 0..=7 range by shifting the
// luma MV's lower 2 bits left by 1 (effectively passing
// `mv_y_q << 1` to `chroma_mc`, which will do `& 7` internally).
// `mv_y_q >> 3` would underflow relative to the `>> 2` the spec
// wants, so the integer step is precomputed here.
#[allow(clippy::too_many_arguments)]
fn mc_chroma_partition_422(
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
    // Partition size in chroma samples: horizontally halved (SubWidthC=2),
    // vertically unscaled (SubHeightC=1).
    let pw = w * 2;
    let ph = h * 4;
    let mut tmp_cb = vec![0u8; pw * ph];
    let mut tmp_cr = vec![0u8; pw * ph];
    // Base chroma-sample origin — width halved, height matches luma.
    let base_x = (mb_x as i32) * 8 + (c0 as i32) * 2;
    let base_y = (mb_y as i32) * 16 + (r0 as i32) * 4;
    let cstride = reference.chroma_stride();
    let cw = (reference.width / 2) as i32;
    let ch = reference.height as i32;
    // `chroma_mc` consumes `mv_y_q` as a luma quarter-pel and does
    // `int_y = base_y + (mv_y_q >> 3)`, `dy = mv_y_q & 7`. For 4:2:2
    // the spec wants `int_y = base_y + (mv_y_q >> 2)` and
    // `dy = (mv_y_q & 3) << 1`, which is exactly what feeding
    // `mv_y_q << 1` reproduces: `(mv_y_q << 1) >> 3 == mv_y_q >> 2`
    // and `(mv_y_q << 1) & 7 == (mv_y_q & 3) << 1`.
    let mv_y_c = mv_y_q << 1;
    chroma_mc(
        &mut tmp_cb,
        &reference.cb,
        cstride,
        cw,
        ch,
        base_x,
        base_y,
        mv_x_q,
        mv_y_c,
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
        mv_y_c,
        pw,
        ph,
    );
    if let Some(cw_entry) = chroma_weight {
        apply_chroma_weight(&mut tmp_cb, cw_entry, 0);
        apply_chroma_weight(&mut tmp_cr, cw_entry, 1);
    }
    let stride = pic.chroma_stride();
    // Chroma MB is 8 wide × 16 tall — destination uses c0*2 horizontally
    // (half luma) and r0*4 vertically (same as luma).
    let off = pic.chroma_off(mb_x, mb_y) + r0 * 4 * stride + c0 * 2;
    for rr in 0..ph {
        for cc in 0..pw {
            pic.cb[off + rr * stride + cc] = tmp_cb[rr * pw + cc];
            pic.cr[off + rr * stride + cc] = tmp_cr[rr * pw + cc];
        }
    }
}

/// Return the (luma, chroma) weight pair for list 0 at reference index
/// `ref_idx`. Returns `None` when the slice header has no
/// `pred_weight_table` (explicit weighted prediction disabled for the
/// slice) — callers pass that straight through to MC which skips the
/// weighting step.
fn l0_weight_for(
    sh: &SliceHeader,
    ref_idx: i8,
) -> (Option<&LumaWeight>, Option<&ChromaWeight>) {
    let Some(tbl) = sh.pred_weight_table.as_ref() else {
        return (None, None);
    };
    let idx = ref_idx.max(0) as usize;
    let lw = tbl.luma_l0.get(idx);
    let cw = tbl.chroma_l0.get(idx);
    (lw, cw)
}

fn predict_inter_nc_luma(pic: &Picture, mb_x: u32, mb_y: u32, br_row: usize, br_col: usize) -> i32 {
    // Same predictor as intra (§9.2.1.1), but also walks MB boundaries where
    // the neighbour might be an inter MB with coded residual.
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
