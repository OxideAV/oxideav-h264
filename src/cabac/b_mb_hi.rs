//! 10-bit CABAC B-slice macroblock decode — ITU-T H.264 §7.3.5 + §9.3
//! with the bit-depth extensions from §7.4.2.1.1 and §8.5.12.1.
//!
//! Mirrors [`crate::cabac::b_mb::decode_b_mb_cabac`] but writes the
//! reconstructed samples into `Picture::y16 / cb16 / cr16` and routes the
//! residual dequant through the `*_ext` i64-widened helpers so the
//! extended `QpY + QpBdOffsetY` range (up to 63 at 10-bit) doesn't overflow.
//!
//! The CABAC entropy layer is bit-depth agnostic — context state,
//! binarisation, MVD + ref_idx_lX derivation, CBP, and residual decode run
//! unchanged. What differs:
//!
//! * Motion compensation uses the u16 helpers in [`crate::b_mb_hi`]
//!   (`apply_motion_compensation_b_hi`, `direct_16x16_compensate_hi`,
//!   `direct_8x8_compensate_hi`).
//! * Residual dequant uses `dequantize_4x4_scaled_ext` and
//!   `inv_hadamard_2x2_chroma_dc_scaled_ext` with `qp + QpBdOffset`.
//! * `mb_qp_delta` wraps modulo `52 + QpBdOffsetY` (§7.4.5 extended).
//! * Intra-in-B dispatches to the 10-bit CABAC I helper
//!   [`crate::cabac::mb_hi::decode_intra_mb_given_imb_cabac_hi`].
//!
//! Scope: 4:2:0, CABAC, B-slice — all `B_Direct` / `B_L0` / `B_L1` / `B_Bi`
//! shapes for 16×16, 16×8, 8×16, `B_8x8`, plus `B_Skip`. MBAFF and
//! `transform_size_8x8_flag = 1` return `Error::Unsupported` (same posture
//! as the 8-bit CABAC B path).

use oxideav_core::{Error, Result};

use crate::b_mb::BSliceCtx;
use crate::cabac::b_mb as cabac_b;
use crate::cabac::binarize;
use crate::cabac::context::CabacContext;
use crate::cabac::engine::CabacDecoder;
use crate::cabac::mb::{
    cbp_chroma_ctx_idx_inc_ac, cbp_chroma_ctx_idx_inc_any, cbp_luma_ctx_idx_inc,
    decode_residual_block_in_place,
};
use crate::cabac::p_mb as cabac_p;
use crate::cabac::residual::BlockCat;
use crate::cabac::tables::{
    CTX_IDX_CODED_BLOCK_PATTERN_LUMA, CTX_IDX_MB_QP_DELTA, CTX_IDX_MB_SKIP_FLAG_B,
    CTX_IDX_MB_TYPE_B, CTX_IDX_MB_TYPE_I, CTX_IDX_SUB_MB_TYPE_B,
};
use crate::mb::LUMA_BLOCK_RASTER;
use crate::mb_type::{
    decode_b_slice_mb_type, decode_b_sub_mb_type, BMbType, BPartition, BSubPartition, IMbType,
    PredDir,
};
use crate::motion::predict_mv_list;
use crate::picture::{MbInfo, Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::SliceHeader;
use crate::sps::Sps;
use crate::transform::{
    chroma_qp_hi, dequantize_4x4_scaled_ext, idct_4x4, inv_hadamard_2x2_chroma_dc_scaled_ext,
};

const SUB_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 2), (2, 0), (2, 2)];

/// Decode one B-slice macroblock through CABAC on the 10-bit pipeline.
/// Returns `Ok(true)` when `mb_skip_flag = 1`.
#[allow(clippy::too_many_arguments)]
pub fn decode_b_mb_cabac_hi(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    bctx: &BSliceCtx<'_>,
    prev_qp: &mut i32,
) -> Result<bool> {
    let skip_inc = cabac_b::mb_skip_flag_b_ctx_idx_inc(pic, mb_x, mb_y);
    let skipped = {
        let slice = &mut ctxs[CTX_IDX_MB_SKIP_FLAG_B..CTX_IDX_MB_SKIP_FLAG_B + 3];
        binarize::decode_mb_skip_flag_b(d, slice, skip_inc)?
    };
    if skipped {
        crate::b_mb_hi::decode_b_skip_mb_hi(
            sh, sps, mb_x, mb_y, pic, ref_list0, ref_list1, bctx, *prev_qp,
        )?;
        return Ok(true);
    }

    let mb_type_raw = {
        let (b_slice, i_slice) = if CTX_IDX_MB_TYPE_B < CTX_IDX_MB_TYPE_I {
            let (head, tail) = ctxs.split_at_mut(CTX_IDX_MB_TYPE_I);
            (
                &mut head[CTX_IDX_MB_TYPE_B..CTX_IDX_MB_TYPE_B + 9],
                &mut tail[..8],
            )
        } else {
            let (head, tail) = ctxs.split_at_mut(CTX_IDX_MB_TYPE_B);
            (
                &mut tail[..9],
                &mut head[CTX_IDX_MB_TYPE_I..CTX_IDX_MB_TYPE_I + 8],
            )
        };
        let inc = cabac_b::mb_type_b_ctx_idx_inc(pic, mb_x, mb_y);
        binarize::decode_mb_type_b(d, b_slice, i_slice, inc)?
    };

    if mb_type_raw == 48 {
        return crate::cabac::mb_hi::decode_intra_mb_given_imb_cabac_hi(
            d,
            ctxs,
            sh,
            sps,
            pps,
            mb_x,
            mb_y,
            pic,
            prev_qp,
            IMbType::IPcm,
        )
        .map(|_| false);
    }
    let bmb = decode_b_slice_mb_type(mb_type_raw).ok_or_else(|| {
        Error::invalid(format!(
            "h264 10bit cabac b-slice: bad mb_type {mb_type_raw}"
        ))
    })?;
    match bmb {
        BMbType::IntraInB(imb) => {
            crate::cabac::mb_hi::decode_intra_mb_given_imb_cabac_hi(
                d, ctxs, sh, sps, pps, mb_x, mb_y, pic, prev_qp, imb,
            )?;
        }
        BMbType::Inter { partition } => {
            decode_b_inter_hi(
                d, ctxs, sh, sps, pps, mb_x, mb_y, pic, ref_list0, ref_list1, bctx, prev_qp,
                partition,
            )?;
        }
    }
    Ok(false)
}

// ---------------------------------------------------------------------------
// Inter-MB dispatch.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_b_inter_hi(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    bctx: &BSliceCtx<'_>,
    prev_qp: &mut i32,
    partition: BPartition,
) -> Result<()> {
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        *info = MbInfo {
            qp_y: *prev_qp,
            coded: true,
            intra: false,
            skipped: false,
            ..Default::default()
        };
    }

    match partition {
        BPartition::Direct16x16 => {
            crate::b_mb_hi::direct_16x16_compensate_hi(
                sh, bctx, mb_x, mb_y, pic, ref_list0, ref_list1,
            )?;
        }
        BPartition::L0_16x16 => {
            decode_16x16_single_dir(
                d,
                ctxs,
                sh,
                bctx,
                mb_x,
                mb_y,
                pic,
                ref_list0,
                ref_list1,
                PredDir::L0,
            )?;
        }
        BPartition::L1_16x16 => {
            decode_16x16_single_dir(
                d,
                ctxs,
                sh,
                bctx,
                mb_x,
                mb_y,
                pic,
                ref_list0,
                ref_list1,
                PredDir::L1,
            )?;
        }
        BPartition::Bi_16x16 => {
            decode_16x16_single_dir(
                d,
                ctxs,
                sh,
                bctx,
                mb_x,
                mb_y,
                pic,
                ref_list0,
                ref_list1,
                PredDir::BiPred,
            )?;
        }
        BPartition::TwoPart16x8 { dirs } => {
            let rects = [(0usize, 0usize, 2usize, 4usize), (2, 0, 2, 4)];
            decode_two_partition(
                d, ctxs, sh, bctx, mb_x, mb_y, pic, ref_list0, ref_list1, &rects, &dirs,
            )?;
        }
        BPartition::TwoPart8x16 { dirs } => {
            let rects = [(0usize, 0usize, 4usize, 2usize), (0, 2, 4, 2)];
            decode_two_partition(
                d, ctxs, sh, bctx, mb_x, mb_y, pic, ref_list0, ref_list1, &rects, &dirs,
            )?;
        }
        BPartition::B8x8 => {
            decode_b_8x8(
                d, ctxs, sh, sps, mb_x, mb_y, pic, ref_list0, ref_list1, bctx,
            )?;
        }
    }

    // Parse-and-reject 8×8 transform — matches the 8-bit CABAC B path.
    if pps.transform_8x8_mode_flag {
        return Err(Error::unsupported(
            "h264 10bit cabac b-slice: transform_size_8x8_flag=1 not yet wired",
        ));
    }

    let cbp = {
        let slice =
            &mut ctxs[CTX_IDX_CODED_BLOCK_PATTERN_LUMA..CTX_IDX_CODED_BLOCK_PATTERN_LUMA + 12];
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
        )?
    };
    let cbp_luma = (cbp & 0x0F) as u8;
    let cbp_chroma = ((cbp >> 4) & 0x03) as u8;

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
        info.intra4x4_pred_mode = [INTRA_DC_FAKE; 16];
        info.cbp_luma = cbp_luma;
        info.cbp_chroma = cbp_chroma;
    }
    decode_inter_residual_luma_hi(d, ctxs, mb_x, mb_y, pic, cbp_luma, qp_y_prime)?;
    decode_inter_residual_chroma_hi(d, ctxs, sps, pps, mb_x, mb_y, pic, cbp_chroma, qp_y)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// 16×16 single-direction partition.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_16x16_single_dir(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sh: &SliceHeader,
    bctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    dir: PredDir,
) -> Result<()> {
    let ref_l0 = if uses_l0(dir) {
        Some(cabac_b::read_ref_idx_lx(
            d,
            ctxs,
            pic,
            mb_x,
            mb_y,
            0,
            0,
            sh.num_ref_idx_l0_active_minus1 + 1,
            0,
        )?)
    } else {
        None
    };
    let ref_l1 = if uses_l1(dir) {
        Some(cabac_b::read_ref_idx_lx(
            d,
            ctxs,
            pic,
            mb_x,
            mb_y,
            0,
            0,
            sh.num_ref_idx_l1_active_minus1 + 1,
            1,
        )?)
    } else {
        None
    };
    let mvd_l0 = if ref_l0.is_some() {
        Some(cabac_b::read_mvd(d, ctxs, pic, mb_x, mb_y, 0, 0, 0)?)
    } else {
        None
    };
    let mvd_l1 = if ref_l1.is_some() {
        Some(cabac_b::read_mvd(d, ctxs, pic, mb_x, mb_y, 0, 0, 1)?)
    } else {
        None
    };
    compensate_partition(
        sh, bctx, pic, ref_list0, ref_list1, mb_x, mb_y, 0, 0, 4, 4, dir, ref_l0, ref_l1, mvd_l0,
        mvd_l1,
    )
}

// ---------------------------------------------------------------------------
// 16×8 / 8×16 pair.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_two_partition(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sh: &SliceHeader,
    bctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    rects: &[(usize, usize, usize, usize); 2],
    dirs: &[PredDir; 2],
) -> Result<()> {
    let num_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let num_l1 = sh.num_ref_idx_l1_active_minus1 + 1;

    let mut ref_l0 = [None; 2];
    let mut ref_l1 = [None; 2];
    let mut mvd_l0 = [None; 2];
    let mut mvd_l1 = [None; 2];

    for p in 0..2 {
        if uses_l0(dirs[p]) {
            let (r0, c0, _, _) = rects[p];
            ref_l0[p] = Some(cabac_b::read_ref_idx_lx(
                d, ctxs, pic, mb_x, mb_y, r0, c0, num_l0, 0,
            )?);
        }
    }
    for p in 0..2 {
        if uses_l1(dirs[p]) {
            let (r0, c0, _, _) = rects[p];
            ref_l1[p] = Some(cabac_b::read_ref_idx_lx(
                d, ctxs, pic, mb_x, mb_y, r0, c0, num_l1, 1,
            )?);
        }
    }
    for p in 0..2 {
        if uses_l0(dirs[p]) {
            let (r0, c0, _, _) = rects[p];
            mvd_l0[p] = Some(cabac_b::read_mvd(d, ctxs, pic, mb_x, mb_y, r0, c0, 0)?);
        }
    }
    for p in 0..2 {
        if uses_l1(dirs[p]) {
            let (r0, c0, _, _) = rects[p];
            mvd_l1[p] = Some(cabac_b::read_mvd(d, ctxs, pic, mb_x, mb_y, r0, c0, 1)?);
        }
    }
    for p in 0..2 {
        let (r0, c0, ph, pw) = rects[p];
        compensate_partition(
            sh, bctx, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, ph, pw, dirs[p], ref_l0[p],
            ref_l1[p], mvd_l0[p], mvd_l1[p],
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// B_8×8 — four 8×8 sub-MBs.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_b_8x8(
    d: &mut CabacDecoder<'_>,
    ctxs: &mut [CabacContext],
    sh: &SliceHeader,
    _sps: &Sps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    bctx: &BSliceCtx<'_>,
) -> Result<()> {
    let num_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let num_l1 = sh.num_ref_idx_l1_active_minus1 + 1;

    let mut subs = [BSubPartition::Direct8x8; 4];
    for s in 0..4 {
        let raw = {
            let slice = &mut ctxs[CTX_IDX_SUB_MB_TYPE_B..CTX_IDX_SUB_MB_TYPE_B + 4];
            binarize::decode_sub_mb_type_b(d, slice)?
        };
        subs[s] = decode_b_sub_mb_type(raw).ok_or_else(|| {
            Error::invalid(format!("h264 10bit cabac b-slice: bad sub_mb_type {raw}"))
        })?;
    }

    let mut ref_l0 = [None; 4];
    let mut ref_l1 = [None; 4];
    for s in 0..4 {
        if let Some(dir) = subs[s].pred_dir() {
            if uses_l0(dir) {
                let (sr0, sc0) = SUB_OFFSETS[s];
                ref_l0[s] = Some(cabac_b::read_ref_idx_lx(
                    d, ctxs, pic, mb_x, mb_y, sr0, sc0, num_l0, 0,
                )?);
            }
        }
    }
    for s in 0..4 {
        if let Some(dir) = subs[s].pred_dir() {
            if uses_l1(dir) {
                let (sr0, sc0) = SUB_OFFSETS[s];
                ref_l1[s] = Some(cabac_b::read_ref_idx_lx(
                    d, ctxs, pic, mb_x, mb_y, sr0, sc0, num_l1, 1,
                )?);
            }
        }
    }

    #[derive(Clone, Copy, Default)]
    struct SubMvd {
        mvd_l0: Option<(i32, i32)>,
        mvd_l1: Option<(i32, i32)>,
    }
    let mut mvds: Vec<Vec<SubMvd>> = Vec::with_capacity(4);
    for s in 0..4 {
        let sp = subs[s];
        let n = sp.num_sub_partitions();
        let mut entry = Vec::with_capacity(n);
        let dir_opt = sp.pred_dir();
        for sub_idx in 0..n {
            if let Some(dir) = dir_opt {
                let (sr0, sc0) = SUB_OFFSETS[s];
                let (dr, dc, _sh_h, _sh_w) = sp.sub_rect(sub_idx);
                let r0 = sr0 + dr;
                let c0 = sc0 + dc;
                let mvd_l0 = if uses_l0(dir) {
                    Some(cabac_b::read_mvd(d, ctxs, pic, mb_x, mb_y, r0, c0, 0)?)
                } else {
                    None
                };
                let mvd_l1 = if uses_l1(dir) {
                    Some(cabac_b::read_mvd(d, ctxs, pic, mb_x, mb_y, r0, c0, 1)?)
                } else {
                    None
                };
                entry.push(SubMvd { mvd_l0, mvd_l1 });
            } else {
                entry.push(SubMvd::default());
            }
        }
        mvds.push(entry);
    }

    for s in 0..4 {
        let sp = subs[s];
        let (sr0, sc0) = SUB_OFFSETS[s];
        if matches!(sp, BSubPartition::Direct8x8) {
            crate::b_mb_hi::direct_8x8_compensate_hi(
                sh, bctx, mb_x, mb_y, sr0, sc0, pic, ref_list0, ref_list1,
            )?;
            continue;
        }
        let dir = sp.pred_dir().expect("non-direct");
        for sub_idx in 0..sp.num_sub_partitions() {
            let (dr, dc, sh_h, sh_w) = sp.sub_rect(sub_idx);
            let r0 = sr0 + dr;
            let c0 = sc0 + dc;
            let m = mvds[s][sub_idx];
            compensate_partition(
                sh, bctx, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, sh_h, sh_w, dir,
                ref_l0[s], ref_l1[s], m.mvd_l0, m.mvd_l1,
            )?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Partition-level MV derivation + motion compensation on u16 planes.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn compensate_partition(
    sh: &SliceHeader,
    bctx: &BSliceCtx<'_>,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    ph: usize,
    pw: usize,
    dir: PredDir,
    ref_idx_l0: Option<i8>,
    ref_idx_l1: Option<i8>,
    mvd_l0: Option<(i32, i32)>,
    mvd_l1: Option<(i32, i32)>,
) -> Result<()> {
    let mv_l0 = if let (Some(r), Some(d)) = (ref_idx_l0, mvd_l0) {
        let pmv = predict_mv_list(pic, mb_x, mb_y, r0, c0, ph, pw, r, 0);
        Some((
            pmv.0.wrapping_add(d.0 as i16),
            pmv.1.wrapping_add(d.1 as i16),
        ))
    } else {
        None
    };
    let mv_l1 = if let (Some(r), Some(d)) = (ref_idx_l1, mvd_l1) {
        let pmv = predict_mv_list(pic, mb_x, mb_y, r0, c0, ph, pw, r, 1);
        Some((
            pmv.0.wrapping_add(d.0 as i16),
            pmv.1.wrapping_add(d.1 as i16),
        ))
    } else {
        None
    };
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        for rr in r0..r0 + ph {
            for cc in c0..c0 + pw {
                let idx = rr * 4 + cc;
                if let (Some(r), Some(mv)) = (ref_idx_l0, mv_l0) {
                    info.ref_idx_l0[idx] = r;
                    info.mv_l0[idx] = mv;
                }
                if let (Some(r), Some(mv)) = (ref_idx_l1, mv_l1) {
                    info.ref_idx_l1[idx] = r;
                    info.mv_l1[idx] = mv;
                }
            }
        }
    }
    crate::b_mb_hi::apply_motion_compensation_b_hi(
        sh, bctx, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, ph, pw, dir, ref_idx_l0,
        ref_idx_l1, mv_l0, mv_l1,
    )
}

// ---------------------------------------------------------------------------
// Inter residual — u16 samples, i64-widened dequant.
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
// Helpers.
// ---------------------------------------------------------------------------

fn uses_l0(dir: PredDir) -> bool {
    matches!(dir, PredDir::L0 | PredDir::BiPred)
}
fn uses_l1(dir: PredDir) -> bool {
    matches!(dir, PredDir::L1 | PredDir::BiPred)
}
