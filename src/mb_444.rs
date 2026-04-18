//! 4:4:4 (ChromaArrayType == 3) I-slice CAVLC macroblock decode.
//!
//! ITU-T H.264 §6.4.1 Table 6-1 puts the chroma planes at the same
//! resolution as luma when `chroma_format_idc = 3`. The CAVLC
//! macroblock layer then coincides with the monochrome / luma code
//! path per plane: chroma uses luma-style Intra_4×4 / Intra_16×16
//! predictors (not the 4:2:0 chroma 8×8 modes), luma-style residual
//! blocks (no chroma DC 2×2 Hadamard and no 8 × ChromaAc slots), and
//! the standard luma scan. `intra_chroma_pred_mode` is absent from the
//! bitstream — each chroma plane inherits the luma plane's mode.
//!
//! The CBP encoding collapses too: for I_NxN `coded_block_pattern` is
//! an `me(v)` that maps through `golomb_to_intra4x4_cbp_gray` to a
//! single 4-bit value; the same 4 bits gate the per-8×8 residual for
//! EACH of the three colour components. The spec's `CodedBlockPatternChroma`
//! bits are unused in 4:4:4 (FFmpeg's `decode_chroma = false` path).
//!
//! This module is sequenced to match FFmpeg's
//! `ff_h264_decode_mb_cavlc` + `decode_luma_residual` invocations at
//! `CHROMA444(h)` — three `decode_luma_residual` calls, one per plane,
//! each using the plane's own QP and scaling list.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::cavlc::{decode_residual_block, BlockKind};
use crate::intra_pred::{
    predict_intra_16x16, predict_intra_4x4, Intra16x16Mode, Intra16x16Neighbours, Intra4x4Mode,
    Intra4x4Neighbours,
};
use crate::mb::{
    collect_intra16x16_neighbours, collect_intra4x4_neighbours, predict_intra4x4_mode_with,
    predict_nc_luma, top_right_available_4x4, LUMA_BLOCK_RASTER,
};
use crate::mb_type::IMbType;
use crate::picture::{MbInfo, Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::SliceHeader;
use crate::sps::Sps;
use crate::transform::{
    chroma_qp, dequantize_4x4_scaled, idct_4x4, inv_hadamard_4x4_dc_scaled,
};

/// §9.1.2 / Table 9-4 — 4:4:4 and monochrome CBP mapping for intra-4×4
/// macroblocks. Source: FFmpeg `golomb_to_intra4x4_cbp_gray` in
/// `libavcodec/h264_cavlc.c`. Maps the `me(v)` golomb value (0..=15)
/// to the 4-bit `CodedBlockPatternLuma` — the same pattern is applied
/// to every colour plane in 4:4:4.
pub const GOLOMB_TO_INTRA4X4_CBP_GRAY: [u8; 16] = [
    15, 0, 7, 11, 13, 14, 3, 5, 10, 12, 1, 2, 4, 8, 6, 9,
];

/// Identifies one of the three colour planes in 4:4:4. `Plane::Y` is the
/// luma plane; `Plane::Cb` / `Plane::Cr` are chroma planes that, in
/// 4:4:4, are the same dimensions as luma and decode through the same
/// luma intra / residual code paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plane {
    Y,
    Cb,
    Cr,
}

impl Plane {
    pub fn as_idx(self) -> usize {
        match self {
            Plane::Y => 0,
            Plane::Cb => 1,
            Plane::Cr => 2,
        }
    }
}

/// Entry point for 4:4:4 I-slice intra macroblock decode. Sequences
/// transform size flag → luma intra4x4 modes → CBP → qp_delta → per-plane
/// reconstruction, then records the full MbInfo.
pub fn decode_intra_mb_444(
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
    let _ = sh;
    debug_assert_eq!(sps.chroma_format_idc, 3);

    // §7.3.5.1 — transform_size_8x8_flag precedes intra mode syntax for
    // I_NxN when the PPS enables it. 4:4:4 CAVLC 8×8 transform is
    // treated the same as 4:2:0 on the luma path; chroma planes reuse
    // the same transform choice.
    let mut transform_8x8 = false;
    let mut intra4x4_modes = [INTRA_DC_FAKE; 16];
    if matches!(imb, IMbType::INxN) {
        if pps.transform_8x8_mode_flag {
            transform_8x8 = br.read_flag()?;
        }
        if transform_8x8 {
            return Err(Error::unsupported(
                "h264: 4:4:4 I_NxN with 8×8 transform not yet wired — 4×4 transform only",
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

    // In 4:4:4 the chroma_pred_mode syntax element is NOT transmitted —
    // each chroma plane inherits the luma plane's intra mode. FFmpeg
    // guards this with `decode_chroma = false` at chroma_format_idc == 3.

    // §7.4.5 / Table 9-4 — 4:4:4 I_NxN CBP is only 4 bits. I_16x16 has
    // the CBP baked into mb_type (luma 4 bits + chroma 2 bits as usual);
    // in 4:4:4 the chroma 2 bits are discarded and luma_cbp is applied
    // to all three planes for AC decode.
    let (cbp_luma, cbp_chroma_from_mb_type) = match imb {
        IMbType::INxN => {
            let cbp_raw = br.read_ue()?;
            if cbp_raw > 15 {
                return Err(Error::invalid(format!(
                    "h264 4:4:4 mb: I_NxN cbp {cbp_raw} > 15"
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
    let _ = cbp_chroma_from_mb_type;

    let needs_qp_delta = matches!(imb, IMbType::I16x16 { .. }) || cbp_luma != 0;
    if needs_qp_delta {
        let dqp = br.read_se()?;
        *prev_qp = ((*prev_qp + dqp + 52) % 52).clamp(0, 51);
    }
    let qp_y = *prev_qp;
    let qp_cb = chroma_qp(qp_y, pps.chroma_qp_index_offset);
    let qp_cr = chroma_qp(qp_y, pps.second_chroma_qp_index_offset);

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

    // Reconstruct each plane in order Y, Cb, Cr — matches FFmpeg's three
    // back-to-back `decode_luma_residual` calls for `p = 0, 1, 2` at
    // CHROMA444. Each plane's residual coefficients consume bits from
    // the same bitstream; its samples go into the plane's own buffer.
    match imb {
        IMbType::I16x16 {
            intra16x16_pred_mode,
            ..
        } => {
            for plane in [Plane::Y, Plane::Cb, Plane::Cr] {
                let qp = match plane {
                    Plane::Y => qp_y,
                    Plane::Cb => qp_cb,
                    Plane::Cr => qp_cr,
                };
                decode_plane_intra_16x16(
                    br,
                    pic,
                    mb_x,
                    mb_y,
                    plane,
                    intra16x16_pred_mode,
                    cbp_luma,
                    qp,
                )?;
            }
        }
        IMbType::INxN => {
            for plane in [Plane::Y, Plane::Cb, Plane::Cr] {
                let qp = match plane {
                    Plane::Y => qp_y,
                    Plane::Cb => qp_cb,
                    Plane::Cr => qp_cr,
                };
                decode_plane_intra_nxn(br, pic, mb_x, mb_y, plane, &intra4x4_modes, cbp_luma, qp)?;
            }
        }
        IMbType::IPcm => unreachable!(),
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// Plane access helpers.
// -----------------------------------------------------------------------------

/// Read a reference to a plane's sample buffer. Returns a raw slice plus the
/// stride in samples. All three planes share the luma stride under 4:4:4.
fn plane_samples(pic: &Picture, plane: Plane) -> (&[u8], usize) {
    let stride = pic.luma_stride();
    let buf = match plane {
        Plane::Y => pic.y.as_slice(),
        Plane::Cb => pic.cb.as_slice(),
        Plane::Cr => pic.cr.as_slice(),
    };
    (buf, stride)
}

pub fn plane_samples_mut(pic: &mut Picture, plane: Plane) -> (&mut [u8], usize) {
    let stride = pic.luma_stride();
    let buf = match plane {
        Plane::Y => pic.y.as_mut_slice(),
        Plane::Cb => pic.cb.as_mut_slice(),
        Plane::Cr => pic.cr.as_mut_slice(),
    };
    (buf, stride)
}

/// Read the per-4×4 nc (total-coeff) tracker for a plane. Mirrors
/// [`crate::mb::predict_nc_luma`] but lets chroma planes read/write
/// their own arrays. In 4:4:4 each chroma plane has a full 16-entry
/// array because the chroma grid matches luma's 4×4 raster.
pub(crate) fn plane_nc_at(info: &MbInfo, plane: Plane) -> &[u8; 16] {
    match plane {
        Plane::Y => &info.luma_nc,
        Plane::Cb => &info.cb_nc,
        Plane::Cr => &info.cr_nc,
    }
}

pub fn plane_nc_at_mut(info: &mut MbInfo, plane: Plane) -> &mut [u8; 16] {
    match plane {
        Plane::Y => &mut info.luma_nc,
        Plane::Cb => &mut info.cb_nc,
        Plane::Cr => &mut info.cr_nc,
    }
}

/// §9.2.1.1 predicted-nC for a 4×4 block of `plane`. For luma this
/// defers to the shared helper in `mb.rs` so the 4:2:0 path stays
/// bit-exact. Same function covers intra and inter neighbours —
/// §9.2.1.1 reads the neighbour's per-4×4 TotalCoeff (`nc`) regardless
/// of whether the neighbour MB is intra or inter.
pub(crate) fn predict_nc_plane(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    plane: Plane,
    br_row: usize,
    br_col: usize,
) -> i32 {
    if plane == Plane::Y {
        return predict_nc_luma(pic, mb_x, mb_y, br_row, br_col);
    }
    let info_here = pic.mb_info_at(mb_x, mb_y);
    let here = plane_nc_at(info_here, plane);
    let left = if br_col > 0 {
        Some(here[br_row * 4 + br_col - 1])
    } else if mb_x > 0 {
        let info = pic.mb_info_at(mb_x - 1, mb_y);
        if info.coded {
            Some(plane_nc_at(info, plane)[br_row * 4 + 3])
        } else {
            None
        }
    } else {
        None
    };
    let top = if br_row > 0 {
        Some(here[(br_row - 1) * 4 + br_col])
    } else if mb_y > 0 {
        let info = pic.mb_info_at(mb_x, mb_y - 1);
        if info.coded {
            Some(plane_nc_at(info, plane)[12 + br_col])
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

/// Gather 4×4 intra-prediction neighbours for `plane`. Luma defers to
/// the shared helper. Chroma planes read samples directly from the
/// chroma buffer but reuse the 4:2:0 luma neighbour geometry (the grid
/// dimensions match under 4:4:4).
fn collect_intra4x4_neighbours_plane(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    plane: Plane,
    br_row: usize,
    br_col: usize,
) -> Intra4x4Neighbours {
    if plane == Plane::Y {
        return collect_intra4x4_neighbours(pic, mb_x, mb_y, br_row, br_col);
    }
    let (buf, stride) = plane_samples(pic, plane);

    let top_avail = br_row > 0 || mb_y > 0;
    let mut top = [0u8; 8];
    if top_avail {
        let row_y_global = if br_row > 0 {
            (mb_y as usize) * 16 + br_row * 4 - 1
        } else {
            (mb_y as usize) * 16 - 1
        };
        let row_off = row_y_global * stride;
        for i in 0..4 {
            top[i] = buf[row_off + (mb_x as usize) * 16 + br_col * 4 + i];
        }
        let tr_avail = top_right_available_4x4(mb_x, mb_y, br_row, br_col, pic);
        if tr_avail {
            for i in 0..4 {
                top[4 + i] = buf[row_off + (mb_x as usize) * 16 + br_col * 4 + 4 + i];
            }
        } else {
            for i in 0..4 {
                top[4 + i] = top[3];
            }
        }
    }

    let left_avail = br_col > 0 || mb_x > 0;
    let mut left = [0u8; 4];
    if left_avail {
        let col_x_global: usize = if br_col > 0 {
            (mb_x as usize) * 16 + br_col * 4 - 1
        } else {
            (mb_x as usize) * 16 - 1
        };
        for i in 0..4 {
            let row = (mb_y as usize) * 16 + br_row * 4 + i;
            left[i] = buf[row * stride + col_x_global];
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
        buf[row_y_global * stride + col_x_global]
    } else {
        0
    };

    Intra4x4Neighbours {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
        top_right_available: false,
    }
}

/// Gather 16×16 intra-prediction neighbours for `plane`.
fn collect_intra16x16_neighbours_plane(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    plane: Plane,
) -> Intra16x16Neighbours {
    if plane == Plane::Y {
        return collect_intra16x16_neighbours(pic, mb_x, mb_y);
    }
    let (buf, stride) = plane_samples(pic, plane);
    let lo_mb = pic.luma_off(mb_x, mb_y);
    let top_avail = mb_y > 0;
    let mut top = [0u8; 16];
    if top_avail {
        let off = lo_mb - stride;
        for i in 0..16 {
            top[i] = buf[off + i];
        }
    }
    let left_avail = mb_x > 0;
    let mut left = [0u8; 16];
    if left_avail {
        for i in 0..16 {
            left[i] = buf[lo_mb + i * stride - 1];
        }
    }
    let tl_avail = top_avail && left_avail;
    let top_left = if tl_avail { buf[lo_mb - stride - 1] } else { 0 };
    Intra16x16Neighbours {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
    }
}

// -----------------------------------------------------------------------------
// Per-plane residual pipelines.
// -----------------------------------------------------------------------------

fn decode_plane_intra_nxn(
    br: &mut BitReader<'_>,
    pic: &mut Picture,
    mb_x: u32,
    mb_y: u32,
    plane: Plane,
    modes: &[u8; 16],
    cbp_luma: u8,
    qp: i32,
) -> Result<()> {
    // §7.4.2.2 — 4×4 scaling list indices: Intra-Y=0, Intra-Cb=1,
    // Intra-Cr=2. Same mapping applies in 4:4:4 because the scaling
    // matrices are per-component regardless of chroma format.
    let scale_idx = plane.as_idx();

    let lo_mb = pic.luma_off(mb_x, mb_y);
    let stride = pic.luma_stride();

    for blk in 0..16usize {
        let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
        let mode_v = modes[br_row * 4 + br_col];
        let mode = Intra4x4Mode::from_u8(mode_v)
            .ok_or_else(|| Error::invalid(format!("h264 4:4:4 mb: invalid intra4x4 mode {mode_v}")))?;
        let neigh = collect_intra4x4_neighbours_plane(pic, mb_x, mb_y, plane, br_row, br_col);
        let mut pred = [0u8; 16];
        predict_intra_4x4(&mut pred, mode, &neigh);

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
            dequantize_4x4_scaled(&mut residual, qp, &scale);
            idct_4x4(&mut residual);
        }
        {
            let (buf, s) = plane_samples_mut(pic, plane);
            let lo = lo_mb + br_row * 4 * s + br_col * 4;
            for r in 0..4 {
                for c in 0..4 {
                    let v = pred[r * 4 + c] as i32 + residual[r * 4 + c];
                    buf[lo + r * s + c] = v.clamp(0, 255) as u8;
                }
            }
        }
        let info = pic.mb_info_mut(mb_x, mb_y);
        plane_nc_at_mut(info, plane)[br_row * 4 + br_col] = total_coeff as u8;
        let _ = stride;
    }
    Ok(())
}

fn decode_plane_intra_16x16(
    br: &mut BitReader<'_>,
    pic: &mut Picture,
    mb_x: u32,
    mb_y: u32,
    plane: Plane,
    intra16x16_pred_mode: u8,
    cbp_luma: u8,
    qp: i32,
) -> Result<()> {
    let scale_idx = plane.as_idx();
    let mode = Intra16x16Mode::from_u8(intra16x16_pred_mode).ok_or_else(|| {
        Error::invalid(format!(
            "h264 4:4:4 mb: invalid intra16x16 mode {intra16x16_pred_mode}"
        ))
    })?;
    let neigh = collect_intra16x16_neighbours_plane(pic, mb_x, mb_y, plane);
    let mut pred = [0u8; 256];
    predict_intra_16x16(&mut pred, mode, &neigh);

    // DC block — always coded for I_16x16. The CAVLC coeff_token for the
    // DC block is decoded using the plane's nC prediction, same as luma.
    let nc_dc = predict_nc_plane(pic, mb_x, mb_y, plane, 0, 0);
    let dc_block = decode_residual_block(br, nc_dc, BlockKind::Luma16x16Dc)?;
    let mut dc = dc_block.coeffs;
    let w_dc = pic.scaling_lists.matrix_4x4(scale_idx)[0];
    inv_hadamard_4x4_dc_scaled(&mut dc, qp, w_dc);

    let lo_mb = pic.luma_off(mb_x, mb_y);
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
            dequantize_4x4_scaled(&mut residual, qp, &scale);
        }
        residual[0] = dc[br_row * 4 + br_col];
        idct_4x4(&mut residual);

        {
            let (buf, s) = plane_samples_mut(pic, plane);
            let lo = lo_mb + br_row * 4 * s + br_col * 4;
            for r in 0..4 {
                for c in 0..4 {
                    let v =
                        pred[(br_row * 4 + r) * 16 + (br_col * 4 + c)] as i32 + residual[r * 4 + c];
                    buf[lo + r * s + c] = v.clamp(0, 255) as u8;
                }
            }
        }
        let info = pic.mb_info_mut(mb_x, mb_y);
        plane_nc_at_mut(info, plane)[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}
