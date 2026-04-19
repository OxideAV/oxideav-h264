//! 4:4:4 (ChromaArrayType == 3) P-slice CAVLC macroblock decode.
//!
//! Mirrors [`crate::p_mb`] for 4:4:4:
//!
//! * Motion compensation runs the luma 6-tap + bilinear quarter-pel filter
//!   on all three planes (§8.4.2.2.1). §8.4.2.2 Table 8-9 picks the
//!   "luma sample interpolation process" for chroma when
//!   `ChromaArrayType == 3` — the 4:2:0 chroma 1/8-pel bilinear is skipped.
//! * Residual per MB is three back-to-back luma-style streams. For inter
//!   MBs each plane decodes 16 × Luma4x4 blocks gated by a 4-bit CBP
//!   shared across all three planes. Luma uses scaling slot 3 (Inter-Y);
//!   Cb / Cr use slots 4 / 5. Per-plane QPs come from `chroma_qp`.
//! * Table 9-4(b) 4:4:4 inter CBP is 16 entries (luma-only, no chroma
//!   bits). FFmpeg's `golomb_to_inter_cbp_gray` matches.
//! * `transform_size_8x8_flag` on the 4:4:4 inter path routes back to
//!   `Error::Unsupported` — 4:4:4 8×8 chroma transform isn't wired yet.
//!
//! Intra-in-P macroblocks re-enter the 4:4:4 intra path
//! ([`crate::mb_444::decode_intra_mb_444`]) so the intra predictor + residual
//! schedule stays in one place.

use oxideav_core::{Error, Result};

use crate::cavlc::{decode_residual_block, BlockKind};
use crate::golomb::BitReaderExt;
use crate::mb::LUMA_BLOCK_RASTER;
use crate::mb_444::{self as mb444};
use crate::mb_type::{
    decode_p_slice_mb_type, decode_p_sub_mb_type, PMbType, PPartition, PSubPartition,
};
use crate::motion::{apply_luma_weight, luma_mc_plane, predict_mv_l0, predict_mv_pskip};
use crate::picture::{MbInfo, Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::{LumaWeight, SliceHeader};
use crate::sps::Sps;
use crate::transform::{chroma_qp, dequantize_4x4_scaled, idct_4x4};
use oxideav_core::bits::BitReader;

/// §9.1.2 / Table 9-4(b) — 4:4:4 and monochrome inter CBP gray code.
/// Source: FFmpeg `golomb_to_inter_cbp_gray` in `libavcodec/h264_cavlc.c`.
/// Maps `me(v)` (0..=15) → 4-bit `CodedBlockPatternLuma`. The chroma
/// columns of Table 9-4 are absent in 4:4:4; the luma CBP is applied to
/// every colour plane.
pub const GOLOMB_TO_INTER_CBP_GRAY: [u8; 16] =
    [0, 1, 2, 4, 8, 3, 5, 10, 12, 15, 7, 11, 13, 14, 6, 9];

/// Entry for one CAVLC P-slice macroblock under 4:4:4.
pub fn decode_p_slice_mb_444(
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
    debug_assert_eq!(sps.chroma_format_idc, 3);
    let mb_type = br.read_ue()?;
    let pmb = decode_p_slice_mb_type(mb_type)
        .ok_or_else(|| Error::invalid(format!("h264 p-slice 4:4:4: bad P mb_type {mb_type}")))?;

    match pmb {
        PMbType::IntraInP(imb) => {
            mb444::decode_intra_mb_444(br, sps, pps, sh, mb_x, mb_y, pic, prev_qp, imb)
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

/// 4:4:4 `P_Skip` — single MV derived from §8.4.1.1, applied to all three
/// planes through the luma filter.
pub fn decode_p_skip_mb_444(
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
    let lw = l0_luma_weight(sh, 0);
    mc_partition_all_planes(
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
    Ok(())
}

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
    let lw = l0_luma_weight(sh, ref_idx);
    mc_partition_all_planes(
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

    finalize_p_mb(br, pps, mb_x, mb_y, pic, prev_qp, PPartition::P16x16)
}

fn decode_p16x8(
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
    let ref_idx_0 = read_ref_idx(br, num_ref_l0)?;
    let ref_idx_1 = read_ref_idx(br, num_ref_l0)?;
    let rects = [(0usize, 0usize, 2usize, 4usize), (2, 0, 2, 4)];
    let refs = [ref_idx_0, ref_idx_1];
    decode_two_partition(
        br,
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
    let ref_idx_0 = read_ref_idx(br, num_ref_l0)?;
    let ref_idx_1 = read_ref_idx(br, num_ref_l0)?;
    let rects = [(0usize, 0usize, 4usize, 2usize), (0, 2, 4, 2)];
    let refs = [ref_idx_0, ref_idx_1];
    decode_two_partition(
        br,
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

fn decode_two_partition(
    br: &mut BitReader<'_>,
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
        fill_partition_mv(pic, mb_x, mb_y, r0, c0, ph, pw, mv, refs[p]);
        let lw = l0_luma_weight(sh, refs[p]);
        mc_partition_all_planes(
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
    }
    finalize_p_mb(br, pps, mb_x, mb_y, pic, prev_qp, partition)
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
    let mut sub_parts = [PSubPartition::Sub8x8; 4];
    for s in 0..4 {
        let v = br.read_ue()?;
        sub_parts[s] = decode_p_sub_mb_type(v)
            .ok_or_else(|| Error::invalid(format!("h264 p-slice 4:4:4: bad sub_mb_type {v}")))?;
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
            let mvd_x = br.read_se()?;
            let mvd_y = br.read_se()?;
            let pmv = predict_mv_l0(pic, mb_x, mb_y, r0, c0, sh_h, sh_w, ref_idx);
            let mv = (
                pmv.0.wrapping_add(mvd_x as i16),
                pmv.1.wrapping_add(mvd_y as i16),
            );
            fill_partition_mv(pic, mb_x, mb_y, r0, c0, sh_h, sh_w, mv, ref_idx);
            let lw = l0_luma_weight(sh, ref_idx);
            mc_partition_all_planes(
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
        }
    }
    let partition = if is_ref0 {
        PPartition::P8x8Ref0
    } else {
        PPartition::P8x8
    };
    finalize_p_mb(br, pps, mb_x, mb_y, pic, prev_qp, partition)
}

/// Parse CBP / qp_delta, write MbInfo, and decode per-plane 4×4 residuals.
fn finalize_p_mb(
    br: &mut BitReader<'_>,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    prev_qp: &mut i32,
    partition: PPartition,
) -> Result<()> {
    // §7.4.5 / Table 9-4(b) — 4:4:4 inter CBP is 4 bits (luma-only). The
    // chroma columns of the 4:2:0 table don't exist here; the same 4 bits
    // gate each plane's 8×8-bit residuals.
    let cbp_raw = br.read_ue()?;
    if cbp_raw > 15 {
        return Err(Error::invalid(format!(
            "h264 p-slice 4:4:4: inter CBP {cbp_raw} > 15"
        )));
    }
    let cbp_luma = GOLOMB_TO_INTER_CBP_GRAY[cbp_raw as usize];
    // 4:4:4 inter uses the same 4-bit CBP for all three planes (FFmpeg's
    // `decode_chroma = false` path in `ff_h264_decode_mb_cavlc`). No chroma
    // CBP bits are emitted.
    if pps.transform_8x8_mode_flag && cbp_luma != 0 {
        // §7.3.5.1 — `transform_size_8x8_flag` would be present here in
        // High444. We don't wire the 4:4:4 8×8 transform on any plane
        // yet; reject so a malformed/unsupported stream fails cleanly
        // rather than decoding the wrong residual shape.
        let flag = br.read_flag()?;
        if flag {
            return Err(Error::unsupported(
                "h264: 4:4:4 inter transform_size_8x8_flag not wired — 4×4 transform only",
            ));
        }
    }
    let needs_qp = cbp_luma != 0;
    if needs_qp {
        let dqp = br.read_se()?;
        *prev_qp = ((*prev_qp + dqp + 52) % 52).clamp(0, 51);
    }
    let qp_y = *prev_qp;
    let qp_cb = chroma_qp(qp_y, pps.chroma_qp_index_offset);
    let qp_cr = chroma_qp(qp_y, pps.second_chroma_qp_index_offset);

    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        info.qp_y = qp_y;
        info.coded = true;
        info.intra = false;
        info.skipped = false;
        info.mb_type_p = Some(PMbType::Inter { partition });
        info.p_partition = Some(partition);
        info.intra4x4_pred_mode = [INTRA_DC_FAKE; 16];
        info.cbp_luma = cbp_luma;
        info.cbp_chroma = 0;
    }

    // §7.4.2.2 — Inter 4×4 scaling-list slots: 3 (Y), 4 (Cb), 5 (Cr).
    for (plane, qp, scale_idx) in [
        (mb444::Plane::Y, qp_y, 3usize),
        (mb444::Plane::Cb, qp_cb, 4usize),
        (mb444::Plane::Cr, qp_cr, 5usize),
    ] {
        decode_plane_inter_4x4(br, mb_x, mb_y, pic, plane, cbp_luma, qp, scale_idx)?;
    }
    Ok(())
}

fn decode_plane_inter_4x4(
    br: &mut BitReader<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    plane: mb444::Plane,
    cbp_luma: u8,
    qp: i32,
    scale_idx: usize,
) -> Result<()> {
    let stride = pic.luma_stride();
    let lo_mb = pic.luma_off(mb_x, mb_y);
    for blk in 0..16usize {
        let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
        let eight_idx = (br_row / 2) * 2 + (br_col / 2);
        if (cbp_luma >> eight_idx) & 1 == 0 {
            continue;
        }
        let nc = mb444::predict_nc_plane(pic, mb_x, mb_y, plane, br_row, br_col);
        let blk = decode_residual_block(br, nc, BlockKind::Luma4x4)?;
        let mut residual = blk.coeffs;
        let total_coeff = blk.total_coeff;
        let scale = *pic.scaling_lists.matrix_4x4(scale_idx);
        dequantize_4x4_scaled(&mut residual, qp, &scale);
        idct_4x4(&mut residual);
        {
            let (buf, s) = mb444::plane_samples_mut(pic, plane);
            let lo = lo_mb + br_row * 4 * s + br_col * 4;
            for r in 0..4 {
                for c in 0..4 {
                    let base = buf[lo + r * s + c] as i32;
                    buf[lo + r * s + c] = (base + residual[r * 4 + c]).clamp(0, 255) as u8;
                }
            }
            let _ = stride;
        }
        let info = pic.mb_info_mut(mb_x, mb_y);
        mb444::plane_nc_at_mut(info, plane)[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Motion compensation — luma filter applied to every plane.
// ---------------------------------------------------------------------------

pub(crate) fn mc_partition_all_planes(
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
    // §8.4.2.2 Table 8-9 — ChromaArrayType = 3 routes chroma MC through
    // the luma sample interpolation process. All three planes share the
    // luma grid in 4:4:4 so base_x / base_y are identical.
    let pw = w * 4;
    let ph = h * 4;
    let base_x = (mb_x as i32) * 16 + (c0 as i32) * 4;
    let base_y = (mb_y as i32) * 16 + (r0 as i32) * 4;
    let stride = pic.luma_stride();
    let off = pic.luma_off(mb_x, mb_y) + r0 * 4 * stride + c0 * 4;

    for plane in [mb444::Plane::Y, mb444::Plane::Cb, mb444::Plane::Cr] {
        let src_buf: &[u8] = match plane {
            mb444::Plane::Y => &reference.y,
            mb444::Plane::Cb => &reference.cb,
            mb444::Plane::Cr => &reference.cr,
        };
        let mut tmp = vec![0u8; pw * ph];
        luma_mc_plane(
            &mut tmp,
            src_buf,
            reference.luma_stride(),
            reference.width as i32,
            reference.height as i32,
            base_x,
            base_y,
            mv_x_q,
            mv_y_q,
            pw,
            ph,
        );
        // §8.4.2.3.2 — explicit weighted prediction is keyed on the luma
        // weight entry. With the luma filter on chroma planes the weight
        // denom / offset stay the same (no separate chroma weight step).
        if let Some(lw) = luma_weight {
            apply_luma_weight(&mut tmp, lw);
        }
        let dst_buf: &mut Vec<u8> = match plane {
            mb444::Plane::Y => &mut pic.y,
            mb444::Plane::Cb => &mut pic.cb,
            mb444::Plane::Cr => &mut pic.cr,
        };
        for rr in 0..ph {
            for cc in 0..pw {
                dst_buf[off + rr * stride + cc] = tmp[rr * pw + cc];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers (copy of p_mb internals kept local for module boundaries).
// ---------------------------------------------------------------------------

fn read_ref_idx(br: &mut BitReader<'_>, num_ref: u32) -> Result<i8> {
    if num_ref == 0 || num_ref == 1 {
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
        return Err(Error::invalid("h264 p-slice 4:4:4: negative ref_idx_l0"));
    }
    let idx = ref_idx as usize;
    ref_list0.get(idx).copied().ok_or_else(|| {
        Error::invalid(format!(
            "h264 p-slice 4:4:4: ref_idx_l0 {} out of RefPicList0 range (len={})",
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

pub(crate) fn l0_luma_weight<'a>(sh: &'a SliceHeader, ref_idx: i8) -> Option<&'a LumaWeight> {
    let tbl = sh.pred_weight_table.as_ref()?;
    let idx = ref_idx.max(0) as usize;
    tbl.luma_l0.get(idx)
}

pub(crate) fn lookup_ref_444<'a>(ref_list0: &[&'a Picture], ref_idx: i8) -> Result<&'a Picture> {
    if ref_idx < 0 {
        return Err(Error::invalid("h264 p-slice 4:4:4: negative ref_idx"));
    }
    let idx = ref_idx as usize;
    ref_list0.get(idx).copied().ok_or_else(|| {
        Error::invalid(format!(
            "h264 p-slice 4:4:4: ref_idx {} out of RefPicList0 range (len={})",
            ref_idx,
            ref_list0.len()
        ))
    })
}

