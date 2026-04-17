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
//! `num_ref_idx_l0_active_minus1 > 0` is accepted in the bitstream but
//! rejected at decode time — the decoder keeps a single L0 reference.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::cavlc::{decode_residual_block, BlockKind};
use crate::mb::LUMA_BLOCK_RASTER;
use crate::mb_type::{
    decode_cbp_inter, decode_p_slice_mb_type, decode_p_sub_mb_type, IMbType, PMbType, PPartition,
    PSubPartition,
};
use crate::motion::{chroma_mc, luma_mc, predict_mv_l0, predict_mv_pskip};
use crate::picture::{MbInfo, Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::SliceHeader;
use crate::sps::Sps;
use crate::transform::{chroma_qp, dequantize_4x4, idct_4x4, inv_hadamard_2x2_chroma_dc};

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
    reference: &Picture,
    prev_qp: &mut i32,
) -> Result<()> {
    let mb_type = br.read_ue()?;
    let pmb = decode_p_slice_mb_type(mb_type)
        .ok_or_else(|| Error::invalid(format!("h264 p-slice: bad P mb_type {mb_type}")))?;

    match pmb {
        PMbType::IntraInP(imb) => decode_intra_in_p(br, sps, pps, sh, mb_x, mb_y, pic, prev_qp, imb),
        PMbType::Inter { partition } => match partition {
            PPartition::P16x16 => {
                decode_p16x16(br, sps, pps, sh, mb_x, mb_y, pic, reference, prev_qp)
            }
            PPartition::P16x8 => {
                decode_p16x8(br, sps, pps, sh, mb_x, mb_y, pic, reference, prev_qp)
            }
            PPartition::P8x16 => {
                decode_p8x16(br, sps, pps, sh, mb_x, mb_y, pic, reference, prev_qp)
            }
            PPartition::P8x8 | PPartition::P8x8Ref0 => decode_p8x8(
                br,
                sps,
                pps,
                sh,
                mb_x,
                mb_y,
                pic,
                reference,
                prev_qp,
                matches!(partition, PPartition::P8x8Ref0),
            ),
        },
    }
}

/// Apply the `P_Skip` derivation and fill in `(mb_x, mb_y)` — no residual,
/// motion vectors come from §8.4.1.1.
pub fn decode_p_skip_mb(
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    reference: &Picture,
    prev_qp: i32,
) -> Result<()> {
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
    mc_luma_partition(pic, reference, mb_x, mb_y, 0, 0, 4, 4, mv.0 as i32, mv.1 as i32);
    mc_chroma_partition(pic, reference, mb_x, mb_y, 0, 0, 4, 4, mv.0 as i32, mv.1 as i32);
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
    reference: &Picture,
    prev_qp: &mut i32,
) -> Result<()> {
    let num_ref_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let ref_idx = read_ref_idx(br, num_ref_l0)?;
    require_single_reference(ref_idx)?;
    let mvd_x = br.read_se()? as i32;
    let mvd_y = br.read_se()? as i32;

    let pmv = predict_mv_l0(pic, mb_x, mb_y, 0, 0, 4, 4, ref_idx);
    let mv = (
        pmv.0.wrapping_add(mvd_x as i16),
        pmv.1.wrapping_add(mvd_y as i16),
    );

    fill_partition_mv(pic, mb_x, mb_y, 0, 0, 4, 4, mv, ref_idx);

    // Predict luma + chroma from reference before residual is added.
    mc_luma_partition(pic, reference, mb_x, mb_y, 0, 0, 4, 4, mv.0 as i32, mv.1 as i32);
    mc_chroma_partition(pic, reference, mb_x, mb_y, 0, 0, 4, 4, mv.0 as i32, mv.1 as i32);

    // CBP (inter) + optional qp_delta + residual.
    let cbp_raw = br.read_ue()?;
    let (cbp_luma, cbp_chroma) = decode_cbp_inter(cbp_raw)
        .ok_or_else(|| Error::invalid(format!("h264 p-slice: bad inter CBP {cbp_raw}")))?;
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
    }

    decode_inter_residual_luma(br, mb_x, mb_y, pic, cbp_luma, qp_y)?;
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
    reference: &Picture,
    prev_qp: &mut i32,
) -> Result<()> {
    let num_ref_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let ref_idx_0 = read_ref_idx(br, num_ref_l0)?;
    let ref_idx_1 = read_ref_idx(br, num_ref_l0)?;
    require_single_reference(ref_idx_0)?;
    require_single_reference(ref_idx_1)?;

    let rects = [(0usize, 0usize, 2usize, 4usize), (2, 0, 2, 4)];
    let refs = [ref_idx_0, ref_idx_1];
    decode_two_partition_p(
        br, sps, pps, sh, mb_x, mb_y, pic, reference, prev_qp, &rects, &refs,
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
    reference: &Picture,
    prev_qp: &mut i32,
) -> Result<()> {
    let num_ref_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let ref_idx_0 = read_ref_idx(br, num_ref_l0)?;
    let ref_idx_1 = read_ref_idx(br, num_ref_l0)?;
    require_single_reference(ref_idx_0)?;
    require_single_reference(ref_idx_1)?;

    let rects = [(0usize, 0usize, 4usize, 2usize), (0, 2, 4, 2)];
    let refs = [ref_idx_0, ref_idx_1];
    decode_two_partition_p(
        br, sps, pps, sh, mb_x, mb_y, pic, reference, prev_qp, &rects, &refs,
        PPartition::P8x16,
    )
}

fn decode_two_partition_p(
    br: &mut BitReader<'_>,
    _sps: &Sps,
    pps: &Pps,
    _sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    reference: &Picture,
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
        let mvd_x = br.read_se()? as i32;
        let mvd_y = br.read_se()? as i32;
        let pmv = predict_mv_l0(pic, mb_x, mb_y, r0, c0, ph, pw, refs[p]);
        let mv = (
            pmv.0.wrapping_add(mvd_x as i16),
            pmv.1.wrapping_add(mvd_y as i16),
        );
        mvs[p] = mv;
        fill_partition_mv(pic, mb_x, mb_y, r0, c0, ph, pw, mv, refs[p]);
        mc_luma_partition(pic, reference, mb_x, mb_y, r0, c0, ph, pw, mv.0 as i32, mv.1 as i32);
        mc_chroma_partition(
            pic, reference, mb_x, mb_y, r0, c0, ph, pw, mv.0 as i32, mv.1 as i32,
        );
    }

    let cbp_raw = br.read_ue()?;
    let (cbp_luma, cbp_chroma) = decode_cbp_inter(cbp_raw)
        .ok_or_else(|| Error::invalid(format!("h264 p-slice: bad inter CBP {cbp_raw}")))?;
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
    }

    decode_inter_residual_luma(br, mb_x, mb_y, pic, cbp_luma, qp_y)?;
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
    reference: &Picture,
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
            require_single_reference(ref_idxs[s])?;
        }
    }
    // For each sub-MB and each of its sub-partitions, read mvd_l0.
    // sub-MB index 0..=3 maps to a 2×2 layout of 8×8 blocks (raster).
    const SUB_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 2), (2, 0), (2, 2)];
    for s in 0..4 {
        let sp = sub_parts[s];
        let (sr0, sc0) = SUB_OFFSETS[s];
        let ref_idx = ref_idxs[s];
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
            mc_luma_partition(
                pic, reference, mb_x, mb_y, r0, c0, sh_h, sh_w, mv.0 as i32, mv.1 as i32,
            );
            mc_chroma_partition(
                pic, reference, mb_x, mb_y, r0, c0, sh_h, sh_w, mv.0 as i32, mv.1 as i32,
            );
        }
    }

    let cbp_raw = br.read_ue()?;
    let (cbp_luma, cbp_chroma) = decode_cbp_inter(cbp_raw)
        .ok_or_else(|| Error::invalid(format!("h264 p-slice: bad inter CBP {cbp_raw}")))?;
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
    }

    decode_inter_residual_luma(br, mb_x, mb_y, pic, cbp_luma, qp_y)?;
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
        let blk = decode_residual_block(br, nc, BlockKind::Luma4x4)?;
        let mut residual = blk.coeffs;
        let total_coeff = blk.total_coeff;
        dequantize_4x4(&mut residual, qp_y);
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

fn decode_inter_residual_chroma(
    br: &mut BitReader<'_>,
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
        let blk = decode_residual_block(br, 0, BlockKind::ChromaDc2x2)?;
        dc_cb[..4].copy_from_slice(&blk.coeffs[..4]);
        let blk = decode_residual_block(br, 0, BlockKind::ChromaDc2x2)?;
        dc_cr[..4].copy_from_slice(&blk.coeffs[..4]);
        inv_hadamard_2x2_chroma_dc(&mut dc_cb, qpc);
        inv_hadamard_2x2_chroma_dc(&mut dc_cr, qpc);
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
                dequantize_4x4(&mut res, qpc);
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
        }
        let info = pic.mb_info_mut(mb_x, mb_y);
        if plane_kind {
            info.cb_nc = nc_arr;
        } else {
            info.cr_nc = nc_arr;
        }
    }
    Ok(())
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

fn require_single_reference(ref_idx: i8) -> Result<()> {
    if ref_idx != 0 {
        return Err(Error::unsupported(format!(
            "h264 p-slice: ref_idx_l0 {} (multi-reference DPB not implemented)",
            ref_idx
        )));
    }
    Ok(())
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
) {
    let pw = w * 4;
    let ph = h * 4;
    let mut tmp = vec![0u8; pw * ph];
    let base_x = (mb_x as i32) * 16 + (c0 as i32) * 4;
    let base_y = (mb_y as i32) * 16 + (r0 as i32) * 4;
    luma_mc(&mut tmp, reference, base_x, base_y, mv_x_q, mv_y_q, pw, ph);
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
) {
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
    let stride = pic.chroma_stride();
    let off = pic.chroma_off(mb_x, mb_y) + r0 * 2 * stride + c0 * 2;
    for rr in 0..ph {
        for cc in 0..pw {
            pic.cb[off + rr * stride + cc] = tmp_cb[rr * pw + cc];
            pic.cr[off + rr * stride + cc] = tmp_cr[rr * pw + cc];
        }
    }
}

fn predict_inter_nc_luma(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    br_row: usize,
    br_col: usize,
) -> i32 {
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
