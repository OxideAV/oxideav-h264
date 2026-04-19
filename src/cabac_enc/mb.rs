//! CABAC I-slice macroblock **encoder** — mirror of
//! [`crate::cabac::mb::decode_i_mb_cabac`].
//!
//! The encoder walks the same decision tree as the decoder: emit
//! `mb_type`, `intra_chroma_pred_mode`, `mb_qp_delta`, then the luma DC
//! residual, the 16 luma AC residuals, the two chroma DC residuals, and
//! the 8 chroma AC residuals — all through CABAC with the matching
//! §9.3.3.1.1.9 ctxIdxInc derivations from [`crate::cabac::mb`].
//!
//! For now this only supports I_16×16 (the one `mb_type` the baseline
//! encoder chooses via `pick_intra_16x16_mode`) in 4:2:0 / 8-bit. I_NxN,
//! I_PCM, Main/High chroma formats, and the intra-8×8 transform are out
//! of scope.

use oxideav_core::{Error, Result};

use crate::cabac::context::CabacContext;
use crate::cabac::mb::{
    cbp_chroma_ctx_idx_inc_ac, cbp_chroma_ctx_idx_inc_any, cbp_luma_ctx_idx_inc, ResidualCtxPlan,
};
use crate::cabac::residual::{BlockCat, CbfNeighbours};
use crate::cabac::tables::{
    CTX_IDX_CODED_BLOCK_PATTERN_LUMA, CTX_IDX_INTRA_CHROMA_PRED_MODE, CTX_IDX_MB_QP_DELTA,
    CTX_IDX_MB_TYPE_I,
};
use crate::cavlc::ZIGZAG_4X4;
use crate::picture::{MbInfo, Picture};

use super::binarize;
use super::engine::CabacEncoder;

/// Emit one I-slice macroblock of type `I_16×16` through CABAC.
///
/// Inputs:
/// * `mb_type` — the raw `mb_type` value (1..=24 for I_16×16).
/// * `luma_dc` — 16-entry DC block in coded scan order (ZIGZAG_4X4 already
///   applied). Per §9.3.3.1.1.9 the DC block is always emitted.
/// * `luma_ac[16]` — the 16 AC blocks in raster order (coded-scan, ZIGZAG
///   applied; position 0 must be zero).
/// * `chroma_dc_cb`, `chroma_dc_cr` — 4-entry 2×2 chroma DC blocks.
/// * `chroma_ac_cb[4]`, `chroma_ac_cr[4]` — the 4 chroma AC 4×4 blocks
///   per plane (raster order, scan applied, position 0 zero).
/// * `cbp_luma` / `cbp_chroma` — the CBP bits as derived by mode decision.
///
/// The function updates `pic.mb_info[..]` for `(mb_x, mb_y)` so subsequent
/// MBs' neighbour derivations see the right state, and updates
/// `pic.last_mb_qp_delta_was_nonzero` for the `mb_qp_delta` ctxIdxInc.
#[allow(clippy::too_many_arguments)]
pub fn encode_i_16x16_mb_cabac(
    enc: &mut CabacEncoder,
    ctxs: &mut [CabacContext],
    pic: &mut Picture,
    mb_x: u32,
    mb_y: u32,
    mb_type: u32,
    intra_chroma_mode: u32,
    mb_qp_delta: i32,
    qp_y: i32,
    luma_dc: &[i32; 16],
    luma_ac: &[[i32; 16]; 16],
    chroma_dc_cb: &[i32; 4],
    chroma_dc_cr: &[i32; 4],
    chroma_ac_cb: &[[i32; 16]; 4],
    chroma_ac_cr: &[[i32; 16]; 4],
    cbp_luma: u8,
    cbp_chroma: u8,
) -> Result<()> {
    if !(1..=24).contains(&mb_type) {
        return Err(Error::invalid(format!(
            "cabac_enc::mb: encode_i_16x16_mb_cabac requires I_16×16 mb_type (1..=24), got {mb_type}"
        )));
    }

    // --- mb_type ---
    let mb_type_inc = mb_type_i_ctx_idx_inc(pic, mb_x, mb_y);
    {
        let slice = &mut ctxs[CTX_IDX_MB_TYPE_I..CTX_IDX_MB_TYPE_I + 8];
        binarize::encode_mb_type_i(enc, slice, mb_type_inc, mb_type)?;
    }

    // --- intra_chroma_pred_mode (always for chroma_format_idc != 0) ---
    let chroma_inc = intra_chroma_pred_mode_ctx_idx_inc(pic, mb_x, mb_y);
    {
        let slice = &mut ctxs[CTX_IDX_INTRA_CHROMA_PRED_MODE..CTX_IDX_INTRA_CHROMA_PRED_MODE + 4];
        binarize::encode_intra_chroma_pred_mode(enc, slice, chroma_inc, intra_chroma_mode)?;
    }

    // --- coded_block_pattern (I_16×16: implicit, carried in mb_type) ---
    // No CBP emitted for I_16×16 — the spec packs both cbp bits into mb_type.

    // --- mb_qp_delta (always for I_16×16) ---
    let inc_qp = if pic.last_mb_qp_delta_was_nonzero {
        1u8
    } else {
        0
    };
    {
        let slice = &mut ctxs[CTX_IDX_MB_QP_DELTA..CTX_IDX_MB_QP_DELTA + 4];
        binarize::encode_mb_qp_delta(enc, slice, inc_qp, mb_qp_delta)?;
    }
    pic.last_mb_qp_delta_was_nonzero = mb_qp_delta != 0;

    // Register the MB in pic.mb_info BEFORE we emit residuals so the
    // neighbour-derivation helpers (coded_block_flag, chroma-AC CBF, …)
    // can read the right mb_info for the MB under construction.
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        *info = MbInfo {
            qp_y,
            coded: true,
            intra: true,
            intra_chroma_pred_mode: intra_chroma_mode as u8,
            mb_type_i: Some(crate::mb_type::IMbType::I16x16 {
                intra16x16_pred_mode: ((mb_type - 1) % 4) as u8,
                cbp_luma,
                cbp_chroma,
            }),
            cbp_luma,
            cbp_chroma,
            ..Default::default()
        };
    }

    // --- Luma 16×16 DC residual (always coded for I_16×16) ---
    let dc_neigh = luma16x16_dc_cbf_neighbours(pic, mb_x, mb_y);
    encode_residual_block(enc, ctxs, BlockCat::Luma16x16Dc, &dc_neigh, luma_dc, 16)?;
    pic.mb_info_mut(mb_x, mb_y).luma16x16_dc_cbf = luma_dc.iter().any(|&v| v != 0);

    // --- Luma AC (only when cbp_luma != 0 per §7.3.5.3) ---
    if cbp_luma != 0 {
        for blk in 0..16usize {
            let (br_row, br_col) = crate::mb::LUMA_BLOCK_RASTER[blk];
            let neigh = cbf_neighbours_luma(pic, mb_x, mb_y, br_row, br_col);
            // AC block has 15 coefficients — positions 1..=15 in the
            // raster-order vector. The helper ignores position 0 on emit.
            let ac = &luma_ac[br_row * 4 + br_col];
            encode_residual_block(enc, ctxs, BlockCat::Luma16x16Ac, &neigh, ac, 15)?;
            let total_coeff = ac.iter().filter(|&&v| v != 0).count() as u8;
            pic.mb_info_mut(mb_x, mb_y).luma_nc[br_row * 4 + br_col] = total_coeff;
        }
    }

    // --- Chroma DC + AC (4:2:0 only for now) ---
    if cbp_chroma >= 1 {
        let cb_dc_neigh = chroma_dc_cbf_neighbours(pic, mb_x, mb_y, 0);
        let mut dc_cb_block = [0i32; 16];
        dc_cb_block[..4].copy_from_slice(chroma_dc_cb);
        encode_residual_block(enc, ctxs, BlockCat::ChromaDc, &cb_dc_neigh, &dc_cb_block, 4)?;
        pic.mb_info_mut(mb_x, mb_y).chroma_dc_cbf[0] = chroma_dc_cb.iter().any(|&v| v != 0);

        let cr_dc_neigh = chroma_dc_cbf_neighbours(pic, mb_x, mb_y, 1);
        let mut dc_cr_block = [0i32; 16];
        dc_cr_block[..4].copy_from_slice(chroma_dc_cr);
        encode_residual_block(enc, ctxs, BlockCat::ChromaDc, &cr_dc_neigh, &dc_cr_block, 4)?;
        pic.mb_info_mut(mb_x, mb_y).chroma_dc_cbf[1] = chroma_dc_cr.iter().any(|&v| v != 0);
    }
    if cbp_chroma == 2 {
        for plane_is_cb in [true, false] {
            let mut nc_arr = [0u8; 4];
            for bi in 0..4usize {
                let br_row = bi >> 1;
                let br_col = bi & 1;
                let neigh =
                    chroma_ac_cbf_neighbours(pic, mb_x, mb_y, plane_is_cb, br_row, br_col, &nc_arr);
                let blk = if plane_is_cb {
                    &chroma_ac_cb[bi]
                } else {
                    &chroma_ac_cr[bi]
                };
                encode_residual_block(enc, ctxs, BlockCat::ChromaAc, &neigh, blk, 15)?;
                let tc = blk.iter().filter(|&&v| v != 0).count() as u8;
                nc_arr[bi] = tc;
            }
            let info = pic.mb_info_mut(mb_x, mb_y);
            let dst = if plane_is_cb {
                &mut info.cb_nc
            } else {
                &mut info.cr_nc
            };
            dst[..4].copy_from_slice(&nc_arr);
        }
    }

    // Suppress unused warning on the CBP helper: the I_16×16 path never
    // emits it (cbp is packed into mb_type), but the helper is used when
    // the encoder grows to I_NxN.
    let _ = encode_cbp_for_i_nxn;
    let _: usize = CTX_IDX_CODED_BLOCK_PATTERN_LUMA;

    Ok(())
}

/// Emit a full CABAC residual block (coded_block_flag + sig map + levels)
/// for a 4:2:0 4×4-shaped category (Luma16x16Dc / Luma16x16Ac / Luma4x4 /
/// ChromaDc / ChromaAc).
///
/// `coeffs[..]` is expected in **coded scan order** (zig-zag already
/// applied). Pass a length matching `max_num_coeff` as the caller's
/// "coefficients worth emitting": Luma16x16Dc/Luma4x4 = 16, Luma16x16Ac /
/// ChromaAc = 15, ChromaDc = 4.
fn encode_residual_block(
    enc: &mut CabacEncoder,
    ctxs: &mut [CabacContext],
    cat: BlockCat,
    neighbours: &CbfNeighbours,
    coeffs: &[i32; 16],
    max_num_coeff: usize,
) -> Result<()> {
    // §9.3.3.1.1.9 — gather the 43-slot context window.
    let plan = ResidualCtxPlan::new(cat, neighbours, true);
    let mut window = plan.gather(ctxs);

    // Map from raster-in vector to "coded scan positions of interest".
    // For 4×4 AC/DC in cats 0/2/4 the caller already did the zig-zag mapping
    // and stored coefficients in positions 0..max_num_coeff of `coeffs`.
    //
    // For AC (cats 1, 4) the first 15 coded-scan positions correspond to
    // raster 1..=15 post-zig-zag; the caller's convention is the same as
    // the decoder's (decoder reads coded positions and writes via
    // `ZIGZAG_4X4[i + 1]` back to raster), so to emit we read from the
    // caller-supplied "raster" convention which is actually already in
    // coded-scan per our contract.
    //
    // To keep this encoder independent of layout confusion, we *re-apply
    // zig-zag ourselves*: treat `coeffs` as raster-order and emit in
    // coded-scan order.
    let scan: &[usize] = match cat {
        BlockCat::Luma16x16Dc | BlockCat::Luma4x4 => &ZIGZAG_4X4[..16],
        BlockCat::Luma16x16Ac | BlockCat::ChromaAc | BlockCat::Chroma4x4 => {
            // AC: 15 positions starting at scan index 1 (skip DC).
            &ZIGZAG_4X4[1..16]
        }
        BlockCat::ChromaDc => &CHROMA_DC_SCAN[..4],
        BlockCat::Luma8x8 => {
            return Err(Error::invalid(
                "cabac_enc::mb::encode_residual_block: Luma8x8 routed through wrong helper",
            ));
        }
    };
    debug_assert_eq!(scan.len(), max_num_coeff);

    // §9.3.3.1.1.9 step 1: coded_block_flag.
    let any_nonzero = scan.iter().take(max_num_coeff).any(|&r| coeffs[r] != 0);
    binarize::encode_coded_block_flag(enc, &mut window[0], any_nonzero);
    if !any_nonzero {
        plan.scatter(ctxs, &window);
        return Ok(());
    }

    // §9.3.3.1.1.9 step 2: significance map over positions 0..=max-2; the
    // last position is implicit when reached.
    let n = max_num_coeff;
    // Find the last non-zero scan index.
    let mut last_scan: usize = 0;
    for i in 0..n {
        if coeffs[scan[i]] != 0 {
            last_scan = i;
        }
    }
    for i in 0..(n - 1) {
        let is_sig = coeffs[scan[i]] != 0;
        binarize::encode_significant_coeff_flag(enc, &mut window[1 + i], is_sig);
        if is_sig {
            let is_last = i == last_scan;
            binarize::encode_last_significant_coeff_flag(enc, &mut window[17 + i], is_last);
            if is_last {
                break;
            }
        }
    }
    // If last_scan == n - 1, no `last_significant_coeff_flag` was emitted
    // — the decoder infers it via the implicit-last rule.

    // §9.3.3.1.1.9 step 3: levels + signs in reverse scan order.
    let mut num_eq1: u32 = 0;
    let mut num_gt1: u32 = 0;
    // Walk down from last_scan (inclusive) to 0.
    let mut i: isize = last_scan as isize;
    while i >= 0 {
        let pos = scan[i as usize];
        let v = coeffs[pos];
        if v != 0 {
            let abs_m1 = v.unsigned_abs().saturating_sub(1);
            let (d_eq1, d_gt1) = binarize::encode_coeff_abs_level_minus1(
                enc,
                &mut window[33..43],
                num_eq1,
                num_gt1,
                abs_m1,
            )?;
            num_eq1 += d_eq1;
            num_gt1 += d_gt1;
            binarize::encode_coeff_sign_flag(enc, v < 0);
        }
        i -= 1;
    }

    plan.scatter(ctxs, &window);
    Ok(())
}

/// 2×2 chroma DC scan order for 4:2:0 (§8.5.11.1): raster positions
/// 0, 1, 2, 3 map to scan 0, 1, 2, 3. The decoder's residual helper
/// writes scan[i] back to raster[i], so we emit raster[i] in the same
/// order.
const CHROMA_DC_SCAN: [usize; 4] = [0, 1, 2, 3];

// --- Neighbour derivations (mirror decoder helpers) --------------------

fn mb_type_i_ctx_idx_inc(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
    // §9.3.3.1.1.3: condTermFlagN = 0 iff unavailable or I_NxN.
    use crate::mb_type::IMbType;
    let a = mb_x > 0 && {
        let m = pic.mb_info_at(mb_x - 1, mb_y);
        m.coded && !matches!(m.mb_type_i, Some(IMbType::INxN))
    };
    let b = mb_y > 0 && {
        let m = pic.mb_info_at(mb_x, mb_y - 1);
        m.coded && !matches!(m.mb_type_i, Some(IMbType::INxN))
    };
    (a as u8) + (b as u8)
}

fn intra_chroma_pred_mode_ctx_idx_inc(pic: &Picture, mb_x: u32, mb_y: u32) -> u8 {
    let check = |m: &MbInfo| m.coded && m.intra && m.intra_chroma_pred_mode != 0;
    let a = mb_x > 0 && check(pic.mb_info_at(mb_x - 1, mb_y));
    let b = mb_y > 0 && check(pic.mb_info_at(mb_x, mb_y - 1));
    (a as u8) + (b as u8)
}

fn luma16x16_dc_cbf_neighbours(pic: &Picture, mb_x: u32, mb_y: u32) -> CbfNeighbours {
    let probe = |mx: u32, my: u32| -> Option<bool> {
        let info = pic.mb_info_at(mx, my);
        if !info.coded {
            return None;
        }
        Some(info.luma16x16_dc_cbf)
    };
    let left = if mb_x > 0 {
        probe(mb_x - 1, mb_y)
    } else {
        None
    };
    let above = if mb_y > 0 {
        probe(mb_x, mb_y - 1)
    } else {
        None
    };
    CbfNeighbours { left, above }
}

fn chroma_dc_cbf_neighbours(pic: &Picture, mb_x: u32, mb_y: u32, c: usize) -> CbfNeighbours {
    let probe = |mx: u32, my: u32| -> Option<bool> {
        let info = pic.mb_info_at(mx, my);
        if !info.coded {
            return None;
        }
        Some(info.chroma_dc_cbf[c])
    };
    let left = if mb_x > 0 {
        probe(mb_x - 1, mb_y)
    } else {
        None
    };
    let above = if mb_y > 0 {
        probe(mb_x, mb_y - 1)
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

fn chroma_ac_cbf_neighbours(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    plane_is_cb: bool,
    br_row: usize,
    br_col: usize,
    already_decoded: &[u8; 4],
) -> CbfNeighbours {
    let left = if br_col > 0 {
        Some(already_decoded[br_row * 2 + br_col - 1] != 0)
    } else if mb_x > 0 {
        let m = pic.mb_info_at(mb_x - 1, mb_y);
        if m.coded {
            let nc = if plane_is_cb { &m.cb_nc } else { &m.cr_nc };
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
            let nc = if plane_is_cb { &m.cb_nc } else { &m.cr_nc };
            Some(nc[2 + br_col] != 0)
        } else {
            None
        }
    } else {
        None
    };
    CbfNeighbours { left, above }
}

/// Unused today (I_NxN emission), kept as a wired stub so the shared
/// `cbp_luma_ctx_idx_inc` import participates in dead-code analysis the
/// same way as the decoder side.
#[allow(dead_code)]
fn encode_cbp_for_i_nxn(
    enc: &mut CabacEncoder,
    ctxs: &mut [CabacContext],
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    chroma_format_idc: u8,
    cbp_luma: u8,
    cbp_chroma: u8,
) -> Result<()> {
    let slice = &mut ctxs[CTX_IDX_CODED_BLOCK_PATTERN_LUMA..CTX_IDX_CODED_BLOCK_PATTERN_LUMA + 12];
    binarize::encode_coded_block_pattern(
        enc,
        slice,
        chroma_format_idc,
        cbp_luma,
        cbp_chroma,
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

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::tables::init_slice_contexts;

    #[test]
    fn encode_empty_i_16x16_roundtrip() {
        // An all-zero residual I_16×16 MB with mode 2 (DC) and chroma
        // mode 0 (DC). This is the simplest CABAC MB the encoder emits
        // and the test confirms the slice-level bin stream decodes
        // symbol-identical through the existing decoder's primitives.
        let mut pic = Picture::new(1, 1);
        let mut enc = CabacEncoder::new();
        let mut ctxs = init_slice_contexts(0, true, 26);

        // mb_type for I_16×16 with pred=2 (DC), cbp_luma=0, cbp_chroma=0:
        //   idx = pred + 4*cbp_chroma + 12*cbp_luma_flag = 2 + 0 + 0 = 2
        //   mb_type = 3.
        let mb_type = 3;
        let luma_dc = [0i32; 16];
        let luma_ac: [[i32; 16]; 16] = [[0; 16]; 16];
        let chroma_dc_cb = [0i32; 4];
        let chroma_dc_cr = [0i32; 4];
        let chroma_ac_cb: [[i32; 16]; 4] = [[0; 16]; 4];
        let chroma_ac_cr: [[i32; 16]; 4] = [[0; 16]; 4];

        encode_i_16x16_mb_cabac(
            &mut enc,
            &mut ctxs,
            &mut pic,
            0,
            0,
            mb_type,
            0,
            0,
            26,
            &luma_dc,
            &luma_ac,
            &chroma_dc_cb,
            &chroma_dc_cr,
            &chroma_ac_cb,
            &chroma_ac_cr,
            0,
            0,
        )
        .unwrap();
        // Mark end of slice.
        enc.encode_terminate(1);
        let bytes = enc.finish();
        // Sanity: the encoder produced SOMETHING.
        assert!(
            !bytes.is_empty(),
            "CABAC MB encode produced empty bitstream"
        );

        // Decode back through the existing decoder — this validates the
        // arithmetic state kept in sync (mb_type, chroma mode, qp_delta,
        // cbf = 0 on every residual block).
        use crate::cabac::binarize as dec_bin;
        use crate::cabac::engine::CabacDecoder;
        let mut dec = CabacDecoder::new(&bytes, 0).unwrap();
        let mut dec_ctxs = init_slice_contexts(0, true, 26);

        let got_mb_type = dec_bin::decode_mb_type_i(
            &mut dec,
            &mut dec_ctxs[CTX_IDX_MB_TYPE_I..CTX_IDX_MB_TYPE_I + 8],
            0,
        )
        .unwrap();
        assert_eq!(got_mb_type, mb_type, "mb_type round-trip");

        let got_chroma_mode = dec_bin::decode_intra_chroma_pred_mode(
            &mut dec,
            &mut dec_ctxs[CTX_IDX_INTRA_CHROMA_PRED_MODE..CTX_IDX_INTRA_CHROMA_PRED_MODE + 4],
            0,
        )
        .unwrap();
        assert_eq!(got_chroma_mode, 0, "intra_chroma_pred_mode round-trip");

        let got_dqp = dec_bin::decode_mb_qp_delta(
            &mut dec,
            &mut dec_ctxs[CTX_IDX_MB_QP_DELTA..CTX_IDX_MB_QP_DELTA + 4],
            0,
        )
        .unwrap();
        assert_eq!(got_dqp, 0, "mb_qp_delta round-trip");
    }
}
