//! I-slice macroblock decode — ITU-T H.264 §7.3.5 / §8.3 / §8.5.
//!
//! Wires CAVLC residual decode, dequantisation, inverse transforms and intra
//! prediction together for every macroblock in an I-slice and writes
//! reconstructed samples into a [`Picture`].
//!
//! Macroblock layer syntax order (`§7.3.5.1`):
//!
//! 1. `mb_type`
//! 2. If `mb_type == I_NxN && pps.transform_8x8_mode_flag`:
//!    `transform_size_8x8_flag` — selects 4 × Intra_8×8 modes vs.
//!    16 × Intra_4×4 modes.
//! 3. If `mb_type == I_NxN` with 4×4 transform: 16 ×
//!    (`prev_intra4x4_pred_mode_flag` + optional `rem_intra4x4_pred_mode`);
//!    with 8×8 transform (High Profile §8.3.2): 4 ×
//!    (`prev_intra8x8_pred_mode_flag` + optional `rem_intra8x8_pred_mode`).
//! 4. If `chroma_format_idc != 0`: `intra_chroma_pred_mode`
//! 5. If `mb_type == I_NxN`: `coded_block_pattern` (`me(v)`)
//! 6. If `cbp != 0` or `mb_type starts I_16x16`: `mb_qp_delta`
//! 7. Residual: luma DC (Intra16x16 only) + 16 × 4×4 luma (or 4 × 8×8) +
//!    2 chroma DC + 8 chroma AC

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::cavlc::{decode_residual_8x8_sub, decode_residual_block, BlockKind};
use crate::intra_pred::{
    predict_intra_16x16, predict_intra_4x4, predict_intra_8x8, predict_intra_chroma,
    predict_intra_chroma_8x16, Intra16x16Mode, Intra16x16Neighbours, Intra4x4Mode,
    Intra4x4Neighbours, Intra8x8Mode, Intra8x8Neighbours, IntraChroma8x16Neighbours,
    IntraChromaMode, IntraChromaNeighbours,
};
use crate::mb_type::{decode_i_slice_mb_type, IMbType};
use crate::picture::{MbInfo, Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::SliceHeader;
use crate::sps::Sps;
use crate::tables::decode_cbp_intra;
use crate::transform::{
    chroma_qp, dequantize_4x4_scaled, dequantize_8x8_scaled, idct_4x4, idct_8x8,
    inv_hadamard_2x2_chroma_dc_scaled, inv_hadamard_2x4_chroma_dc_scaled,
    inv_hadamard_4x4_dc_scaled,
};

/// Per-block (4×4) raster ordering of the residual blocks within a
/// macroblock — §8.5.1, Figure 6-12. Block N covers `(LUMA_BLOCK_RASTER[N])`.
pub const LUMA_BLOCK_RASTER: [(usize, usize); 16] = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
];

/// Top-left slice decode entry — drives the macroblock loop.
///
/// Under MBAFF (§7.3.4) MBs arrive as top+bottom pairs; before the top MB
/// of each pair we read `mb_field_decoding_flag` and propagate it to
/// both MBs of the pair so the per-MB sample-offset / row-stride helpers
/// on [`Picture`] pick the right layout (§6.4.9.4). Non-MBAFF pictures
/// fall through the same loop with a single-step `pair_size = 1`.
pub fn decode_i_slice_data(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
) -> Result<()> {
    let mb_w = sps.pic_width_in_mbs();
    // §7.4.3 — a field-coded picture (PAFF) has `PicHeightInMapUnits` MB
    // rows (the field is half-height). MBAFF frame-mode and progressive
    // both want `pic_height_in_mbs` (the latter equals map_units, the
    // former 2× map_units).
    let mb_h = if sh.field_pic_flag {
        sps.pic_height_in_map_units()
    } else {
        sps.pic_height_in_mbs()
    };
    let total_mbs = mb_w * mb_h;
    if sh.first_mb_in_slice >= total_mbs {
        return Err(Error::invalid("h264 slice: first_mb_in_slice out of range"));
    }
    let mut prev_qp = (pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta).clamp(0, 51);

    if pic.mbaff_enabled {
        // §7.3.4 — `first_mb_in_slice` counts MB pairs in MBAFF, so the
        // starting (mb_x, mb_y_top) pair coord is `(fm %  mb_w, 2 * (fm /
        // mb_w))`. Each iteration consumes two MBs (top then bottom).
        let first_pair = sh.first_mb_in_slice as usize;
        let pair_rows = (mb_h / 2) as usize;
        let mb_w_us = mb_w as usize;
        let total_pairs = pair_rows * mb_w_us;
        for pair_idx in first_pair..total_pairs {
            let pair_x = (pair_idx % mb_w_us) as u32;
            let pair_y = (pair_idx / mb_w_us) as u32;
            let field = br.read_flag()?;
            let top_mb_y = pair_y * 2;
            let bot_mb_y = top_mb_y + 1;
            // Stamp the per-pair field flag before decode so
            // `Picture::luma_off` / `Picture::mb_luma_row_info` return
            // the field-aware offsets throughout `decode_one_mb`.
            pic.mb_info_mut(pair_x, top_mb_y).mb_field_decoding_flag = field;
            pic.mb_info_mut(pair_x, bot_mb_y).mb_field_decoding_flag = field;
            decode_one_mb(br, sps, pps, sh, pair_x, top_mb_y, pic, &mut prev_qp)?;
            decode_one_mb(br, sps, pps, sh, pair_x, bot_mb_y, pic, &mut prev_qp)?;
        }
    } else {
        let mut mb_addr = sh.first_mb_in_slice;
        while mb_addr < total_mbs {
            let mb_x = mb_addr % mb_w;
            let mb_y = mb_addr / mb_w;
            decode_one_mb(br, sps, pps, sh, mb_x, mb_y, pic, &mut prev_qp)?;
            mb_addr += 1;
        }
    }
    Ok(())
}

fn decode_one_mb(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    prev_qp: &mut i32,
) -> Result<()> {
    let mb_type = br.read_ue()?;
    let imb = decode_i_slice_mb_type(mb_type)
        .ok_or_else(|| Error::invalid(format!("h264 slice: bad I mb_type {mb_type}")))?;
    decode_intra_mb_given_imb(br, sps, pps, sh, mb_x, mb_y, pic, prev_qp, imb)
}

/// Public entry for decoding an intra macroblock whose `mb_type` has
/// already been parsed into an [`IMbType`]. Used by P-slice decode to reuse
/// the intra reconstruction path when a `mb_type` >= 5 appears inside a P
/// slice.
pub fn decode_intra_mb_given_imb(
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
    if matches!(imb, IMbType::IPcm) {
        return decode_pcm_mb(br, mb_x, mb_y, pic, *prev_qp);
    }
    if sps.chroma_format_idc == 3 {
        return crate::mb_444::decode_intra_mb_444(br, sps, pps, sh, mb_x, mb_y, pic, prev_qp, imb);
    }

    // Per §7.3.5.1, transform_size_8x8_flag comes before the intra mode
    // syntax for I_NxN when the PPS enables it. When the flag is set we
    // read 4 Intra_8×8 mode slots and dispatch to the High Profile §8.5.13
    // residual path; otherwise we read 16 Intra_4×4 mode slots.
    let mut transform_8x8 = false;
    let mut intra4x4_modes = [INTRA_DC_FAKE; 16];
    if matches!(imb, IMbType::INxN) {
        if pps.transform_8x8_mode_flag {
            transform_8x8 = br.read_flag()?;
        }
        if transform_8x8 {
            // §8.3.2 — four Intra_8×8 blocks in raster order. The chosen
            // mode is replicated into all four 4×4 slots the 8×8 covers,
            // so the 4×4-keyed `predict_intra4x4_mode_with` returns the
            // correct §8.3.2.1 neighbour-min for subsequent 8×8 blocks.
            for blk8 in 0..4usize {
                let br_row = (blk8 >> 1) * 2;
                let br_col = (blk8 & 1) * 2;
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
                for dr in 0..2 {
                    for dc in 0..2 {
                        intra4x4_modes[(br_row + dr) * 4 + br_col + dc] = mode;
                    }
                }
            }
        } else {
            // Iterate in raster of 4×4 sub-blocks per spec coding order
            // (LUMA_BLOCK_RASTER walks the zig-zag scan of 4×4 blocks).
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
    }

    // Chroma pred mode (always present when chroma_format_idc != 0).
    let chroma_pred_mode = if sps.chroma_format_idc != 0 {
        let v = br.read_ue()?;
        if v > 3 {
            return Err(Error::invalid(format!(
                "h264 mb: intra_chroma_pred_mode {v} > 3"
            )));
        }
        IntraChromaMode::from_u8(v as u8).unwrap()
    } else {
        IntraChromaMode::Dc
    };

    // CBP: only for I_NxN. For I_16x16 the cbp comes baked into mb_type.
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
        // QP_Y wraps mod 52 with a small offset; spec equation §7.4.5.
        *prev_qp = ((*prev_qp + dqp + 52) % 52).clamp(0, 51);
    }
    let qp_y = *prev_qp;

    // Initialise MB info. Preserve `mb_field_decoding_flag` which the
    // slice loop stamped before dispatch (§7.3.4) — the `..Default::default()`
    // below would otherwise reset it to `false` and flip the MB back to
    // frame-coded mid-decode.
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        let field = info.mb_field_decoding_flag;
        *info = MbInfo {
            qp_y,
            coded: true,
            intra: true,
            intra4x4_pred_mode: intra4x4_modes,
            cbp_luma,
            cbp_chroma,
            mb_field_decoding_flag: field,
            ..Default::default()
        };
    }

    // Predict + residual + reconstruct luma.
    match imb {
        IMbType::I16x16 {
            intra16x16_pred_mode,
            ..
        } => decode_luma_intra_16x16(
            br,
            sps,
            pps,
            sh,
            mb_x,
            mb_y,
            pic,
            intra16x16_pred_mode,
            cbp_luma,
            qp_y,
        )?,
        IMbType::INxN if transform_8x8 => {
            // §8.7 deblocking keys on `transform_8x8` to skip the internal
            // 4×4 edges that don't exist under an 8×8 transform.
            pic.mb_info_mut(mb_x, mb_y).transform_8x8 = true;
            decode_luma_intra_8x8(
                br,
                sps,
                pps,
                sh,
                mb_x,
                mb_y,
                pic,
                &intra4x4_modes,
                cbp_luma,
                qp_y,
            )?
        }
        IMbType::INxN => decode_luma_intra_nxn(
            br,
            sps,
            pps,
            sh,
            mb_x,
            mb_y,
            pic,
            &intra4x4_modes,
            cbp_luma,
            qp_y,
        )?,
        IMbType::IPcm => unreachable!(),
    }

    // Chroma reconstruction — per-format dispatch: 4:2:0 (the original
    // 8×8 chroma tile) vs 4:2:2 (8×16 chroma tile with the 2×4 DC
    // Hadamard + 8 AC blocks per plane).
    if sps.chroma_format_idc == 2 {
        decode_chroma_422(
            br,
            sps,
            pps,
            sh,
            mb_x,
            mb_y,
            pic,
            chroma_pred_mode,
            cbp_chroma,
            qp_y,
        )?;
    } else {
        decode_chroma(
            br,
            sps,
            pps,
            sh,
            mb_x,
            mb_y,
            pic,
            chroma_pred_mode,
            cbp_chroma,
            qp_y,
        )?;
    }
    // Record intra_chroma_pred_mode so neighbour lookups (used by CABAC) see
    // the right value.
    pic.mb_info_mut(mb_x, mb_y).intra_chroma_pred_mode = chroma_pred_mode as u8;

    Ok(())
}

// -----------------------------------------------------------------------------
// I_PCM
// -----------------------------------------------------------------------------

fn decode_pcm_mb(
    br: &mut BitReader<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    prev_qp: i32,
) -> Result<()> {
    while !br.is_byte_aligned() {
        let _ = br.read_u1()?;
    }
    let lstride = pic.luma_stride();
    let lo = pic.luma_off(mb_x, mb_y);
    for r in 0..16 {
        for c in 0..16 {
            pic.y[lo + r * lstride + c] = br.read_u32(8)? as u8;
        }
    }
    let cstride = pic.chroma_stride();
    let co = pic.chroma_off(mb_x, mb_y);
    let chroma_rows = pic.mb_height_chroma();
    let chroma_cols = pic.mb_width_chroma();
    for r in 0..chroma_rows {
        for c in 0..chroma_cols {
            pic.cb[co + r * cstride + c] = br.read_u32(8)? as u8;
        }
    }
    for r in 0..chroma_rows {
        for c in 0..chroma_cols {
            pic.cr[co + r * cstride + c] = br.read_u32(8)? as u8;
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
    // §9.3.3.1.1.4: for I_PCM neighbour derivations, treat all CBP bits as
    // "coded" so neighbour ctxIdxInc sees the expected condTermFlag=0.
    info.cbp_luma = 0x0F;
    info.cbp_chroma = 2;
    Ok(())
}

// -----------------------------------------------------------------------------
// Luma — I_NxN.
// -----------------------------------------------------------------------------

fn decode_luma_intra_nxn(
    br: &mut BitReader<'_>,
    _sps: &Sps,
    _pps: &Pps,
    _sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    modes: &[u8; 16],
    cbp_luma: u8,
    qp_y: i32,
) -> Result<()> {
    // `row_stride` accounts for MBAFF field-coded MBs which skip every
    // other plane row (§7.3.4) — non-MBAFF falls through to plain
    // `luma_stride()`.
    let row_stride = pic.luma_row_stride_for(mb_y);
    let lo_mb = pic.luma_off(mb_x, mb_y);

    for blk in 0..16usize {
        let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
        let mode_v = modes[br_row * 4 + br_col];
        let mode = Intra4x4Mode::from_u8(mode_v)
            .ok_or_else(|| Error::invalid(format!("h264 mb: invalid intra4x4 mode {mode_v}")))?;
        let neigh = collect_intra4x4_neighbours(pic, mb_x, mb_y, br_row, br_col);
        let mut pred = [0u8; 16];
        predict_intra_4x4(&mut pred, mode, &neigh);

        let cbp_bit_idx = (br_row / 2) * 2 + (br_col / 2);
        let has_residual = (cbp_luma >> cbp_bit_idx) & 1 != 0;

        let mut residual = [0i32; 16];
        let mut total_coeff = 0u32;
        if has_residual {
            let nc = predict_nc_luma(pic, mb_x, mb_y, br_row, br_col);
            let blk = decode_residual_block(br, nc, BlockKind::Luma4x4)?;
            total_coeff = blk.total_coeff;
            residual = blk.coeffs;
            // §7.4.2.2 index 0 — Intra-Y 4×4.
            let scale = *pic.scaling_lists.matrix_4x4(0);
            dequantize_4x4_scaled(&mut residual, qp_y, &scale);
            idct_4x4(&mut residual);
        }
        let lo = lo_mb + br_row * 4 * row_stride + br_col * 4;
        for r in 0..4 {
            for c in 0..4 {
                let v = pred[r * 4 + c] as i32 + residual[r * 4 + c];
                pic.y[lo + r * row_stride + c] = v.clamp(0, 255) as u8;
            }
        }
        pic.mb_info_mut(mb_x, mb_y).luma_nc[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Luma — I_NxN with transform_size_8x8_flag (High Profile, §8.3.2 + §8.5.13).
// -----------------------------------------------------------------------------
//
// Mirrors FFmpeg's `decode_luma_residual` when `IS_8x8DCT(mb_type)` is set
// (`libavcodec/h264_cavlc.c`). Each of the four 8×8 blocks (`i8x8` = 0..=3)
// decodes as four 4×4-style CAVLC sub-blocks (`i4x4` = 0..=3) spliced
// together via [`cavlc::ZIGZAG_8X8_CAVLC`]. Inside this loop the per-4×4
// `luma_nc` slots act as FFmpeg's `non_zero_count_cache`: each sub-block
// writes its own `total_coeff` at its 4×4 position so the NEXT sub-block's
// `pred_non_zero_count` lookup sees it. Once all four sub-blocks of one 8×8
// are decoded, the top-left slot is overwritten with the sum of all four
// (`nnz[0] += nnz[1] + nnz[8] + nnz[9]` in FFmpeg) so downstream neighbour
// MBs see a single consolidated count at the 8×8 block's top-left position.

fn decode_luma_intra_8x8(
    br: &mut BitReader<'_>,
    _sps: &Sps,
    _pps: &Pps,
    _sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    modes: &[u8; 16],
    cbp_luma: u8,
    qp_y: i32,
) -> Result<()> {
    let row_stride = pic.luma_row_stride_for(mb_y);
    let lo_mb = pic.luma_off(mb_x, mb_y);

    for blk8 in 0..4usize {
        let br8_row = blk8 >> 1;
        let br8_col = blk8 & 1;
        // Mode was replicated across all four 4×4 slots — any one works.
        let mode_v = modes[(br8_row * 2) * 4 + br8_col * 2];
        let mode = Intra8x8Mode::from_u8(mode_v)
            .ok_or_else(|| Error::invalid(format!("h264 mb: invalid intra8x8 mode {mode_v}")))?;
        let neigh = collect_intra8x8_neighbours(pic, mb_x, mb_y, br8_row, br8_col);
        let mut pred = [0u8; 64];
        predict_intra_8x8(&mut pred, mode, &neigh);

        let has_residual = (cbp_luma >> blk8) & 1 != 0;
        let mut residual = [0i32; 64];
        if has_residual {
            let r0 = br8_row * 2;
            let c0 = br8_col * 2;
            // Sub-block order matches FFmpeg's i4x4 = 0..=3 inside i8x8:
            // TL, TR, BL, BR — scan8 slots (r,c), (r,c+1), (r+1,c),
            // (r+1,c+1). Writing `luma_nc` between sub-blocks lets the
            // in-MB neighbour lookup for sub-blocks 1..=3 read the
            // previous sub-block's `total_coeff`.
            let slots = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)];
            for (sub, &(sr, sc)) in slots.iter().enumerate() {
                let nc = predict_nc_luma(pic, mb_x, mb_y, sr, sc);
                let tc = decode_residual_8x8_sub(br, nc, sub, &mut residual)?;
                pic.mb_info_mut(mb_x, mb_y).luma_nc[sr * 4 + sc] = tc;
            }
            // §7.4.2.2 — 8×8 intra-Y slot 0.
            let scale = *pic.scaling_lists.matrix_8x8(0);
            dequantize_8x8_scaled(&mut residual, qp_y, &scale);
            idct_8x8(&mut residual);

            // FFmpeg post-decode aggregation: collapse the 2×2 of
            // per-sub-block counts into the top-left slot so neighbouring
            // MBs see one representative nnz for the whole 8×8 block.
            let info = pic.mb_info_mut(mb_x, mb_y);
            let sum = info.luma_nc[r0 * 4 + c0] as u16
                + info.luma_nc[r0 * 4 + c0 + 1] as u16
                + info.luma_nc[(r0 + 1) * 4 + c0] as u16
                + info.luma_nc[(r0 + 1) * 4 + c0 + 1] as u16;
            info.luma_nc[r0 * 4 + c0] = sum.min(16) as u8;
        }

        let lo = lo_mb + (br8_row * 8) * row_stride + br8_col * 8;
        for r in 0..8 {
            for c in 0..8 {
                let v = pred[r * 8 + c] as i32 + residual[r * 8 + c];
                pic.y[lo + r * row_stride + c] = v.clamp(0, 255) as u8;
            }
        }
    }
    Ok(())
}

fn collect_intra8x8_neighbours(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    br8_row: usize,
    br8_col: usize,
) -> Intra8x8Neighbours {
    // §6.4.9.4 — field-coded MBs under MBAFF step by `2 * luma_stride`
    // per sample-row so the neighbour reads below follow the same
    // interleaved layout as the per-MB sample writes.
    let row_stride = pic.luma_row_stride_for(mb_y);
    let lo_mb = pic.luma_off(mb_x, mb_y);
    let blk_tl = lo_mb + br8_row * 8 * row_stride + br8_col * 8;

    let top_avail = br8_row > 0 || pic.mb_top_available(mb_y);
    let mut top = [0u8; 16];
    let mut top_right_available = false;
    if top_avail {
        // Row just above the block's row 0.
        let row_off = blk_tl - row_stride;
        for i in 0..8 {
            top[i] = pic.y[row_off + i];
        }
        // Top-right availability table for an 8×8 block (§6.4.11.4 adapted
        // to 8×8 raster): TL 8×8 reads top-right from MB above;
        // TR 8×8 reads from the above-right MB; BL 8×8 reads from the
        // top-right 8×8 in the same MB (already decoded); BR 8×8 would
        // read into the MB to the right, which is not yet decoded.
        top_right_available = match (br8_row, br8_col) {
            (0, 0) => pic.mb_top_available(mb_y),
            (0, 1) => {
                if let Some(above_y) = pic.mb_above_neighbour(mb_y) {
                    mb_x + 1 < pic.mb_width
                        && pic.mb_info_at(mb_x + 1, above_y).coded
                } else {
                    false
                }
            }
            (1, 0) => true,
            _ => false,
        };
        if top_right_available {
            for i in 0..8 {
                top[8 + i] = pic.y[row_off + 8 + i];
            }
        } else {
            for i in 0..8 {
                top[8 + i] = top[7];
            }
        }
    }

    let left_avail = br8_col > 0 || mb_x > 0;
    let mut left = [0u8; 8];
    if left_avail {
        for i in 0..8 {
            left[i] = pic.y[blk_tl + i * row_stride - 1];
        }
    }

    let tl_avail = top_avail && left_avail;
    let top_left = if tl_avail {
        pic.y[blk_tl - row_stride - 1]
    } else {
        0
    };

    Intra8x8Neighbours {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
        top_right_available,
    }
}

// -----------------------------------------------------------------------------
// Luma — I_16x16.
// -----------------------------------------------------------------------------

fn decode_luma_intra_16x16(
    br: &mut BitReader<'_>,
    _sps: &Sps,
    _pps: &Pps,
    _sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    intra16x16_pred_mode: u8,
    cbp_luma: u8,
    qp_y: i32,
) -> Result<()> {
    let neigh = collect_intra16x16_neighbours(pic, mb_x, mb_y);
    let mut pred = [0u8; 256];
    let mode = Intra16x16Mode::from_u8(intra16x16_pred_mode).ok_or_else(|| {
        Error::invalid(format!(
            "h264 mb: invalid intra16x16 mode {intra16x16_pred_mode}"
        ))
    })?;
    predict_intra_16x16(&mut pred, mode, &neigh);

    // DC luma block — always coded for I_16x16.
    let nc_dc = predict_nc_luma(pic, mb_x, mb_y, 0, 0);
    let dc_block = decode_residual_block(br, nc_dc, BlockKind::Luma16x16Dc)?;
    let mut dc = dc_block.coeffs;
    // §8.5.10 Luma_16×16 DC — use the Intra-Y 4×4 scaling list's
    // (0,0) entry.
    let w_dc = pic.scaling_lists.matrix_4x4(0)[0];
    inv_hadamard_4x4_dc_scaled(&mut dc, qp_y, w_dc);

    let row_stride = pic.luma_row_stride_for(mb_y);
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
            // §7.4.2.2 — Intra-Y 4×4 slot 0 (AC coefficients of I_16×16
            // share the intra-Y scaling list).
            let scale = *pic.scaling_lists.matrix_4x4(0);
            dequantize_4x4_scaled(&mut residual, qp_y, &scale);
        }
        // Insert DC sample at position 0 (already dequantised by the
        // Hadamard pass — it lives in the same scaled space as the AC
        // coefficients post-dequant).
        residual[0] = dc[br_row * 4 + br_col];
        idct_4x4(&mut residual);

        let lo = lo_mb + br_row * 4 * row_stride + br_col * 4;
        for r in 0..4 {
            for c in 0..4 {
                let v = pred[(br_row * 4 + r) * 16 + (br_col * 4 + c)] as i32 + residual[r * 4 + c];
                pic.y[lo + r * row_stride + c] = v.clamp(0, 255) as u8;
            }
        }
        pic.mb_info_mut(mb_x, mb_y).luma_nc[br_row * 4 + br_col] = total_coeff as u8;
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// Chroma reconstruction.
// -----------------------------------------------------------------------------

fn decode_chroma(
    br: &mut BitReader<'_>,
    _sps: &Sps,
    pps: &Pps,
    _sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    chroma_mode: IntraChromaMode,
    cbp_chroma: u8,
    qp_y: i32,
) -> Result<()> {
    let qpc = chroma_qp(qp_y, pps.chroma_qp_index_offset);

    // Predict both planes.
    let neigh_cb = collect_chroma_neighbours(pic, mb_x, mb_y, true);
    let neigh_cr = collect_chroma_neighbours(pic, mb_x, mb_y, false);
    let mut pred_cb = [0u8; 64];
    let mut pred_cr = [0u8; 64];
    predict_intra_chroma(&mut pred_cb, chroma_mode, &neigh_cb);
    predict_intra_chroma(&mut pred_cr, chroma_mode, &neigh_cr);

    // DC blocks — present when cbp_chroma >= 1.
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
        // §8.5.11.1 — Intra chroma DC uses the (0,0) entry of the
        // Intra-Cb (slot 1) and Intra-Cr (slot 2) 4×4 scaling lists.
        let w_cb = pic.scaling_lists.matrix_4x4(1)[0];
        let w_cr = pic.scaling_lists.matrix_4x4(2)[0];
        inv_hadamard_2x2_chroma_dc_scaled(&mut dc_cb, qpc, w_cb);
        inv_hadamard_2x2_chroma_dc_scaled(&mut dc_cr, qpc, w_cr);
    }

    let crow_stride = pic.chroma_row_stride_for(mb_y);
    let co = pic.chroma_off(mb_x, mb_y);

    // Reconstruct AC blocks for Cb then Cr. nC prediction within an MB uses
    // the per-block counts we are building up in this loop (§9.2.1.1 —
    // neighbours that are still being decoded must reflect their current
    // totals, not the pre-MB zeros). Write the in-progress `nc_arr` back
    // through `MbInfo` after every block so `predict_nc_chroma` sees it.
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
                let nc = predict_nc_chroma(pic, mb_x, mb_y, plane_kind, br_row, br_col);
                let ac = decode_residual_block(br, nc, BlockKind::ChromaAc)?;
                total_coeff = ac.total_coeff;
                res = ac.coeffs;
                // §7.4.2.2 — Intra-Cb (slot 1) for the Cb plane,
                // Intra-Cr (slot 2) for the Cr plane. The I-slice
                // chroma path is always intra.
                let cat = if plane_kind { 1 } else { 2 };
                let scale = *pic.scaling_lists.matrix_4x4(cat);
                dequantize_4x4_scaled(&mut res, qpc, &scale);
            }
            res[0] = dc[(br_row << 1) | br_col];
            idct_4x4(&mut res);
            let off_in_mb = br_row * 4 * crow_stride + br_col * 4;
            let plane = if plane_kind { &mut pic.cb } else { &mut pic.cr };
            for r in 0..4 {
                for c in 0..4 {
                    let v = pred[(br_row * 4 + r) * 8 + (br_col * 4 + c)] as i32 + res[r * 4 + c];
                    plane[co + off_in_mb + r * crow_stride + c] = v.clamp(0, 255) as u8;
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

// -----------------------------------------------------------------------------
// Predicted nC (§9.2.1.1).
// -----------------------------------------------------------------------------

pub(crate) fn predict_nc_luma(pic: &Picture, mb_x: u32, mb_y: u32, br_row: usize, br_col: usize) -> i32 {
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
    } else if let Some(above_y) = pic.mb_above_neighbour(mb_y) {
        // §6.4.9.4: under MBAFF the above neighbour of a field-coded MB
        // lives in the previous pair at matching field polarity, i.e.
        // `mb_y - 2`, not the default `mb_y - 1`.
        let info = pic.mb_info_at(mb_x, above_y);
        if info.coded {
            Some(info.luma_nc[12 + br_col])
        } else {
            None
        }
    } else {
        None
    };
    nc_from_neighbours(left, top)
}

fn predict_nc_chroma(
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
    } else if let Some(above_y) = pic.mb_above_neighbour(mb_y) {
        let info = pic.mb_info_at(mb_x, above_y);
        if info.coded {
            Some(pick(info, 2 + br_col))
        } else {
            None
        }
    } else {
        None
    };
    nc_from_neighbours(left, top)
}

fn nc_from_neighbours(left: Option<u8>, top: Option<u8>) -> i32 {
    match (left, top) {
        (Some(l), Some(t)) => (l as i32 + t as i32 + 1) >> 1,
        (Some(l), None) => l as i32,
        (None, Some(t)) => t as i32,
        (None, None) => 0,
    }
}

// -----------------------------------------------------------------------------
// Intra 4×4 mode prediction.
// -----------------------------------------------------------------------------

pub(crate) fn predict_intra4x4_mode_with(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    br_row: usize,
    br_col: usize,
    in_progress_modes: &[u8; 16],
) -> u8 {
    // Within this MB we use in_progress_modes[] for previously coded blocks
    // (raster of 4×4 blocks in the spec coding order corresponds to
    // LUMA_BLOCK_RASTER above). For the left/top neighbour outside the MB we
    // inspect the picture's stored intra4x4 modes.
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
    } else if let Some(above_y) = pic.mb_above_neighbour(mb_y) {
        let ti = pic.mb_info_at(mb_x, above_y);
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

// -----------------------------------------------------------------------------
// Neighbour collection.
// -----------------------------------------------------------------------------

pub(crate) fn collect_intra4x4_neighbours(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    br_row: usize,
    br_col: usize,
) -> Intra4x4Neighbours {
    // §6.4.9.4 — stride follows the MB's field-mode so every cross-MB
    // read hits the same-polarity field on a field-coded MBAFF pair.
    let row_stride = pic.luma_row_stride_for(mb_y);
    let lo_mb = pic.luma_off(mb_x, mb_y);
    let blk_tl = lo_mb + br_row * 4 * row_stride + br_col * 4;

    let top_avail = br_row > 0 || pic.mb_top_available(mb_y);
    let mut top = [0u8; 8];
    if top_avail {
        let row_off = blk_tl - row_stride;
        for i in 0..4 {
            top[i] = pic.y[row_off + i];
        }
        let tr_avail = top_right_available_4x4(mb_x, mb_y, br_row, br_col, pic);
        if tr_avail {
            for i in 0..4 {
                top[4 + i] = pic.y[row_off + 4 + i];
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
        for i in 0..4 {
            left[i] = pic.y[blk_tl + i * row_stride - 1];
        }
    }

    let tl_avail = top_avail && left_avail;
    let top_left = if tl_avail {
        pic.y[blk_tl - row_stride - 1]
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

/// Whether the 4 samples just above-right of this 4×4 block exist.
pub(crate) fn top_right_available_4x4(
    mb_x: u32,
    mb_y: u32,
    br_row: usize,
    br_col: usize,
    pic: &Picture,
) -> bool {
    // Per Figure 8-7 / §6.4.10:
    // - blocks where the upper-right 4×4 is *inside* the same MB and has
    //   already been decoded (i.e. its raster index < current's raster index)
    //   → available
    // - block (br_row=0, br_col=3): upper-right is in next MB to the right at
    //   block (3,0); available only if that MB sits in the row above.
    // - blocks at br_col == 3 with br_row > 0 → upper-right is in the next MB
    //   row, not yet decoded → unavailable.
    if br_col == 3 {
        if br_row == 0 {
            if let Some(above_y) = pic.mb_above_neighbour(mb_y) {
                mb_x + 1 < pic.mb_width
                    && pic.mb_info_at(mb_x + 1, above_y).coded
            } else {
                false
            }
        } else {
            false
        }
    } else {
        // Inside-the-MB cases.
        // Available: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2),
        //            (3,0), (3,1), (3,2). i.e. all (br_col != 3).
        // BUT — Figure 8-7 also marks blocks 3 (raster idx 3 = (1,1)),
        //   7 ((1,3)), 11 ((3,1)), 13 ((3,2)) as "n/a" for top-right.
        // Use the explicit table from the spec.
        const AVAIL: [[bool; 4]; 4] = [
            [true, true, true, false],
            [true, false, true, false],
            [true, true, true, false],
            [true, false, true, false],
        ];
        AVAIL[br_row][br_col]
    }
}

pub(crate) fn collect_intra16x16_neighbours(pic: &Picture, mb_x: u32, mb_y: u32) -> Intra16x16Neighbours {
    // §6.4.9.4 — under MBAFF a field-coded MB reads neighbour samples
    // from the same-polarity field of the previous pair, which in the
    // "uniform pair-mode" case translates to `row_stride = 2 *
    // luma_stride` on the reconstructed plane. For the top MB of pair
    // row 0 the "above" neighbour is unavailable (top edge of the
    // picture), matching the non-MBAFF guard.
    let row_stride = pic.luma_row_stride_for(mb_y);
    let lo_mb = pic.luma_off(mb_x, mb_y);
    let top_avail = pic.mb_top_available(mb_y);
    let mut top = [0u8; 16];
    if top_avail {
        let off = lo_mb - row_stride;
        for i in 0..16 {
            top[i] = pic.y[off + i];
        }
    }
    let left_avail = mb_x > 0;
    let mut left = [0u8; 16];
    if left_avail {
        for i in 0..16 {
            left[i] = pic.y[lo_mb + i * row_stride - 1];
        }
    }
    let tl_avail = top_avail && left_avail;
    let top_left = if tl_avail {
        pic.y[lo_mb - row_stride - 1]
    } else {
        0
    };
    Intra16x16Neighbours {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
    }
}

/// §8.3.4 neighbours for an 8×16 (4:2:2) chroma MB. Pulls 8 samples
/// from the row above, 16 from the column to the left, and the corner
/// sample for Plane mode. Mirrors [`collect_chroma_neighbours`] but
/// sized for 4:2:2's doubled vertical extent.
fn collect_chroma_422_neighbours(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    cb: bool,
) -> IntraChroma8x16Neighbours {
    let cstride = pic.chroma_stride();
    let co_mb = pic.chroma_off(mb_x, mb_y);
    let plane = if cb { &pic.cb } else { &pic.cr };
    let top_avail = mb_y > 0;
    let mut top = [0u8; 8];
    if top_avail {
        let off = co_mb - cstride;
        for i in 0..8 {
            top[i] = plane[off + i];
        }
    }
    let left_avail = mb_x > 0;
    let mut left = [0u8; 16];
    if left_avail {
        for i in 0..16 {
            left[i] = plane[co_mb + i * cstride - 1];
        }
    }
    let tl_avail = top_avail && left_avail;
    let top_left = if tl_avail {
        plane[co_mb - cstride - 1]
    } else {
        0
    };
    IntraChroma8x16Neighbours {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
    }
}

/// 4:2:2 chroma reconstruction (§8.5.11.2, §8.3.4 ChromaArrayType=2).
///
/// Layout per plane inside an MB:
/// * 8×16 samples, laid out as 2 cols × 4 rows of 4×4 AC blocks.
/// * CAVLC DC block: 8 coefficients (ChromaDc2x4 kind, nC = -2) fed
///   through a 2×4 inverse Hadamard + DC-slot dequant. The 8 DC values
///   are mapped into the 8 AC blocks' (0,0) raster slots.
/// * CAVLC AC blocks: 8 ChromaAc blocks per plane, decoded in raster
///   order (row-major over the 2×4 block grid) when `cbp_chroma == 2`.
///
/// `cbp_chroma` keeps the same 3-value encoding as 4:2:0 (Table 9-4
/// with ChromaArrayType-aware mapping): 0 = no chroma residual, 1 = DC
/// only, 2 = DC + AC.
fn decode_chroma_422(
    br: &mut BitReader<'_>,
    _sps: &Sps,
    pps: &Pps,
    _sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    chroma_mode: IntraChromaMode,
    cbp_chroma: u8,
    qp_y: i32,
) -> Result<()> {
    let qpc_cb = chroma_qp(qp_y, pps.chroma_qp_index_offset);
    let qpc_cr = chroma_qp(qp_y, pps.second_chroma_qp_index_offset);

    let neigh_cb = collect_chroma_422_neighbours(pic, mb_x, mb_y, true);
    let neigh_cr = collect_chroma_422_neighbours(pic, mb_x, mb_y, false);
    let mut pred_cb = [0u8; 128];
    let mut pred_cr = [0u8; 128];
    predict_intra_chroma_8x16(&mut pred_cb, chroma_mode, &neigh_cb);
    predict_intra_chroma_8x16(&mut pred_cr, chroma_mode, &neigh_cr);

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
        inv_hadamard_2x4_chroma_dc_scaled(&mut dc_cb, qpc_cb, w_cb);
        inv_hadamard_2x4_chroma_dc_scaled(&mut dc_cr, qpc_cr, w_cr);
    }

    let cstride = pic.chroma_stride();
    let co = pic.chroma_off(mb_x, mb_y);

    for plane_cb in [true, false] {
        let pred = if plane_cb { &pred_cb } else { &pred_cr };
        let dc = if plane_cb { &dc_cb } else { &dc_cr };
        let qpc = if plane_cb { qpc_cb } else { qpc_cr };
        // 4:2:2 AC grid: 4 rows × 2 cols of 4×4 blocks. Walk in
        // row-major order so neighbour NC lookups see the already-
        // decoded blocks above and to the left.
        let mut nc_arr = [0u8; 8];
        for blk_row in 0..4usize {
            for blk_col in 0..2usize {
                let blk_idx = blk_row * 2 + blk_col;
                let mut res = [0i32; 16];
                let mut total_coeff = 0u32;
                if cbp_chroma == 2 {
                    let nc = predict_nc_chroma_422(pic, mb_x, mb_y, plane_cb, blk_row, blk_col);
                    let ac = decode_residual_block(br, nc, BlockKind::ChromaAc)?;
                    total_coeff = ac.total_coeff;
                    res = ac.coeffs;
                    let cat = if plane_cb { 1 } else { 2 };
                    let scale = *pic.scaling_lists.matrix_4x4(cat);
                    dequantize_4x4_scaled(&mut res, qpc, &scale);
                }
                res[0] = dc[blk_idx];
                idct_4x4(&mut res);
                let off_in_mb = (blk_row * 4) * cstride + blk_col * 4;
                let plane_buf = if plane_cb { &mut pic.cb } else { &mut pic.cr };
                for r in 0..4 {
                    for c in 0..4 {
                        let p_idx = (blk_row * 4 + r) * 8 + (blk_col * 4 + c);
                        let v = pred[p_idx] as i32 + res[r * 4 + c];
                        plane_buf[co + off_in_mb + r * cstride + c] = v.clamp(0, 255) as u8;
                    }
                }
                nc_arr[blk_idx] = total_coeff as u8;
                let info = pic.mb_info_mut(mb_x, mb_y);
                let dst = if plane_cb {
                    &mut info.cb_nc
                } else {
                    &mut info.cr_nc
                };
                // Store the 4:2:2 NCs in a canonical linear layout so
                // [`predict_nc_chroma_422`] can read them back.
                dst[..8].copy_from_slice(&nc_arr);
            }
        }
    }
    Ok(())
}

/// §9.2.1.1 predicted nC for a 4×4 chroma AC block under
/// ChromaArrayType == 2. The chroma plane is 2 wide × 4 tall 4×4 blocks
/// inside each MB — we store per-block total_coeff at
/// `MbInfo::{cb,cr}_nc[blk_row*2 + blk_col]` and consult left/top
/// neighbours across MB boundaries just like the luma path.
fn predict_nc_chroma_422(
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
            // The bottom row of the 2×4 chroma AC grid is at block-row 3.
            Some(pick(info, 3 * 2 + blk_col))
        } else {
            None
        }
    } else {
        None
    };
    nc_from_neighbours(left, top)
}

fn collect_chroma_neighbours(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    cb: bool,
) -> IntraChromaNeighbours {
    // Mirrors the luma neighbour fetch under MBAFF — `chroma_row_stride_for`
    // doubles the plane stride for field-coded MBs so cross-MB reads
    // pick the same-polarity chroma field (§6.4.9.4 applied to 4:2:0).
    let row_stride = pic.chroma_row_stride_for(mb_y);
    let co_mb = pic.chroma_off(mb_x, mb_y);
    let plane = if cb { &pic.cb } else { &pic.cr };
    let top_avail = pic.mb_top_available(mb_y);
    let mut top = [0u8; 8];
    if top_avail {
        let off = co_mb - row_stride;
        for i in 0..8 {
            top[i] = plane[off + i];
        }
    }
    let left_avail = mb_x > 0;
    let mut left = [0u8; 8];
    if left_avail {
        for i in 0..8 {
            left[i] = plane[co_mb + i * row_stride - 1];
        }
    }
    let tl_avail = top_avail && left_avail;
    let top_left = if tl_avail {
        plane[co_mb - row_stride - 1]
    } else {
        0
    };
    IntraChromaNeighbours {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
    }
}
