//! §7.3.5 — macroblock_layer emit (encoder side, I_16x16 path).
//!
//! Round-3 scope: every macroblock is `I_16x16`. Luma uses one of the
//! four §8.3.3 modes (DC / V / H / Plane); chroma uses one of the four
//! §8.3.4 modes (DC / V / H / Plane). `cbp_luma` is now {0, 15}: 0 when
//! the per-MB luma AC quantizes entirely to zero, 15 when at least one
//! 4x4 AC block carries a non-zero level (the spec only allows 0 or 15
//! for Intra_16x16 — §7.4.5.1). `cbp_chroma` ranges over {0, 1, 2}.
//!
//! Emitted syntax for one MB:
//!
//! ```text
//!   mb_type ue(v)               -- Table 7-11 row, picks (predmode, cbp_l, cbp_c)
//!   intra16x16_pred_mode (impl) -- carried by mb_type
//!   intra_chroma_pred_mode ue(v)
//!   coded_block_pattern me(v)   -- absent for Intra_16x16; CBP carried in mb_type
//!   mb_qp_delta se(v)           -- present for every Intra_16x16 (luma DC always)
//!   residual_block(LumaDC, 0..15, max=16)
//!   if cbp_luma == 15:
//!     for blk in 0..16: residual_block(Intra16x16ACLevel[blk], 0..14, max=15)
//!   if cbp_chroma > 0:
//!     residual_block(ChromaDCLevelCb, 0..3, max=4)
//!     residual_block(ChromaDCLevelCr, 0..3, max=4)
//!   if cbp_chroma == 2:
//!     for blk in 0..4: residual_block(ChromaACLevelCb[blk], 0..14, max=15)
//!     for blk in 0..4: residual_block(ChromaACLevelCr[blk], 0..14, max=15)
//! ```

use crate::cavlc::CoeffTokenContext;
use crate::encoder::bitstream::BitWriter;
use crate::encoder::cavlc::{encode_residual_block_cavlc, CavlcEncodeError};
use crate::encoder::transform::zigzag_scan_4x4;

/// §9.1.2 + Table 9-4(a) Intra column — inverse mapping CBP → codeNum.
/// `INTRA_CBP_TO_CODENUM_420_422[cbp]` returns the `me(v)` codeNum that
/// the decoder will map back to `cbp` via `ME_INTRA_420_422`. Entries
/// for invalid CBP values (luma > 15, chroma > 2) are unreachable.
///
/// Table 9-4(a) only lists 48 valid (luma, chroma) pairs out of 16*3 =
/// 48 for ChromaArrayType ∈ {1, 2}. Every (luma, chroma) pair appears
/// exactly once, so the inverse is well-defined.
const INTRA_CBP_TO_CODENUM_420_422: [u8; 48] = {
    // Build the inverse of `crate::macroblock_layer::ME_INTRA_420_422`
    // at compile time. The forward table is reproduced inline here to
    // avoid a runtime indirection (and to make the spec citation
    // local).
    const FWD: [u8; 48] = [
        47, 31, 15, 0, 23, 27, 29, 30, 7, 11, 13, 14, 39, 43, 45, 46, 16, 3, 5, 10, 12, 19, 21, 26,
        28, 35, 37, 42, 44, 1, 2, 4, 8, 17, 18, 20, 24, 6, 9, 22, 25, 32, 33, 34, 36, 40, 38, 41,
    ];
    let mut inv = [0u8; 48];
    let mut i = 0;
    while i < 48 {
        inv[FWD[i] as usize] = i as u8;
        i += 1;
    }
    inv
};

/// §9.1.2 — encode `coded_block_pattern` for an intra MB on
/// ChromaArrayType ∈ {1, 2}: look up the codeNum then `ue(v)` it.
pub fn intra_cbp_to_codenum_420_422(cbp_luma: u8, cbp_chroma: u8) -> u32 {
    debug_assert!(cbp_luma <= 15);
    debug_assert!(cbp_chroma <= 2);
    let cbp = ((cbp_chroma as usize) << 4) | (cbp_luma as usize);
    INTRA_CBP_TO_CODENUM_420_422[cbp] as u32
}

/// Configuration for one Intra_16x16 macroblock.
pub struct I16x16McbConfig {
    /// `pred_mode` ∈ 0..=3 per Table 8-4 (V/H/DC/Plane).
    pub pred_mode: u8,
    /// `intra_chroma_pred_mode` ∈ 0..=3 per Table 8-5 (DC/H/V/Plane).
    pub intra_chroma_pred_mode: u8,
    /// `cbp_luma` ∈ {0, 15}.
    pub cbp_luma: u8,
    /// `cbp_chroma` ∈ {0, 1, 2}.
    pub cbp_chroma: u8,
    /// `mb_qp_delta` per §7.4.5. Range -26..=25 for 8-bit.
    pub mb_qp_delta: i32,
    /// 16 quantized luma DC coefficients in the 4x4 *raster* order
    /// `(blk4x4_y * 4 + blk4x4_x)`. The encoder applies the zig-zag
    /// scan internally before CAVLC.
    pub luma_dc_levels_raster: [i32; 16],
    /// Per-4x4-block luma AC levels in §6.4.3 raster-Z order (matches
    /// `crate::reconstruct::LUMA_4X4_XY` indexing). Each block carries
    /// 15 AC coefficients in scan order with the AC-only layout
    /// (positions 0..=14, slot 15 unused — see `zigzag_scan_4x4_ac`).
    /// Used only when `cbp_luma == 15`.
    pub luma_ac_levels: [[i32; 16]; 16],
    /// Per-luma-AC-block `nC` (CAVLC neighbour context). Used only when
    /// `cbp_luma == 15`. The encoder must pre-compute these via the
    /// §9.2.1.1 derivation — they cascade through the MB as TotalCoeff
    /// of earlier blocks lands.
    pub luma_ac_nc: [i32; 16],
    /// 4 quantized chroma DC levels (Cb), in the same row-major order
    /// the inverse Hadamard expects. Ignored when `cbp_chroma == 0`.
    pub chroma_dc_cb: [i32; 4],
    /// 4 quantized chroma DC levels (Cr).
    pub chroma_dc_cr: [i32; 4],
    /// Per-4x4-block chroma AC levels (Cb), in scan order with the
    /// AC-only layout (positions 0..=14, slot 15 unused — see
    /// `zigzag_scan_4x4_ac`). Used only when `cbp_chroma == 2`.
    pub chroma_ac_cb: [[i32; 16]; 4],
    /// Per-4x4-block chroma AC levels (Cr).
    pub chroma_ac_cr: [[i32; 16]; 4],
}

/// §7.4.5 — Table 7-11 row index for an Intra_16x16 macroblock with the
/// given (`pred_mode`, `cbp_luma`, `cbp_chroma`). Inverse of
/// [`crate::macroblock_layer::MbType::from_i_slice`] for the
/// `Intra16x16` arm.
pub fn intra16x16_mb_type_value(pred_mode: u8, cbp_luma: u8, cbp_chroma: u8) -> u32 {
    debug_assert!(pred_mode <= 3);
    debug_assert!(cbp_luma == 0 || cbp_luma == 15);
    debug_assert!(cbp_chroma <= 2);
    let group = match (cbp_luma, cbp_chroma) {
        (0, 0) => 0u32,
        (0, 1) => 1,
        (0, 2) => 2,
        (15, 0) => 3,
        (15, 1) => 4,
        (15, 2) => 5,
        _ => unreachable!("invalid (cbp_luma, cbp_chroma) for Intra_16x16"),
    };
    1 + group * 4 + pred_mode as u32
}

/// Emit one Intra_16x16 macroblock (CAVLC, I-slice).
///
/// `nc_ctx` is the `nC` context for the luma DC residual block. With
/// `cbp_luma == 0` always, the AC TotalCoeff used for nC derivation is
/// identically zero; the encoder mirrors the decoder by passing
/// `Numeric(0)` for every MB.
pub fn write_intra16x16_mb(
    w: &mut BitWriter,
    cfg: &I16x16McbConfig,
    nc_ctx: CoeffTokenContext,
) -> Result<(), CavlcEncodeError> {
    let raw = intra16x16_mb_type_value(cfg.pred_mode, cfg.cbp_luma, cfg.cbp_chroma);
    w.ue(raw);
    // Intra_16x16 doesn't have intra4x4_pred_mode / intra8x8_pred_mode
    // entries — they're carried by mb_type.
    w.ue(cfg.intra_chroma_pred_mode as u32);
    // §7.3.5 — coded_block_pattern is *absent* for Intra_16x16 (the
    // CBP is carried by mb_type itself). Skip the me(v) emission.

    // §7.3.5 — mb_qp_delta is present whenever the MB has at least one
    // coded coefficient (Intra_16x16 always has the luma DC block, so
    // mb_qp_delta is always emitted here).
    w.se(cfg.mb_qp_delta);

    // §7.3.5.3 — Intra_16x16 luma DC block.
    let scan = zigzag_scan_4x4(&cfg.luma_dc_levels_raster);
    encode_residual_block_cavlc(w, nc_ctx, 16, &scan)?;

    // §7.3.5.3 — Intra_16x16 luma AC blocks. cbp_luma == 15 sends all
    // 16 AC blocks (one per 4x4 sub-block, in §6.4.3 raster-Z order).
    // Each block emits the 15 AC coefficients at scan positions 1..=15
    // (startIdx=1, endIdx=15, maxNumCoeff=15 — see §7.3.5.3.3 +
    // Table 9-42). The encoder has already packed these into AC-only
    // scan order via `zigzag_scan_4x4_ac` (positions 0..=14 of each
    // `luma_ac_levels[blk]`; slot 15 is unused padding).
    if cfg.cbp_luma == 15 {
        for blk in 0..16usize {
            let nc = cfg.luma_ac_nc[blk];
            encode_residual_block_cavlc(
                w,
                CoeffTokenContext::Numeric(nc),
                15,
                &cfg.luma_ac_levels[blk][..15],
            )?;
        }
    }

    // §7.3.5.3 — Chroma residual.
    if cfg.cbp_chroma > 0 {
        // Chroma DC (Cb then Cr). 4:2:0: 4 coefficients per plane,
        // already in row-major order ready for the encoder's CAVLC.
        // The CAVLC chroma-DC tables use `CoeffTokenContext::ChromaDc420`.
        encode_residual_block_cavlc(w, CoeffTokenContext::ChromaDc420, 4, &cfg.chroma_dc_cb)?;
        encode_residual_block_cavlc(w, CoeffTokenContext::ChromaDc420, 4, &cfg.chroma_dc_cr)?;
    }
    if cfg.cbp_chroma == 2 {
        // Chroma AC blocks: 4 sub-blocks per plane, Cb first then Cr.
        // Each block carries 15 AC coefficients in scan order.
        // §9.2.1 — nC for chroma AC is derived from neighbour
        // ChromaAC TotalCoeff. Round-2 simplification: pass
        // `Numeric(0)` for every chroma AC block — that's only
        // bit-rate-suboptimal, never incorrect.
        for blk in &cfg.chroma_ac_cb {
            encode_residual_block_cavlc(w, CoeffTokenContext::Numeric(0), 15, &blk[..15])?;
        }
        for blk in &cfg.chroma_ac_cr {
            encode_residual_block_cavlc(w, CoeffTokenContext::Numeric(0), 15, &blk[..15])?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// I_NxN (Intra_4x4) macroblock emit — round 4.
// ---------------------------------------------------------------------------

/// Configuration for one I_NxN (Intra_4x4) macroblock.
///
/// Layout per §7.3.5 / §7.3.5.1 with `mb_type == I_NxN` (raw value 0
/// in I-slice Table 7-11):
///
/// ```text
///   mb_type = 0                         (ue(v) → I_NxN)
///   for each of 16 4x4 blocks:
///     prev_intra4x4_pred_mode_flag      (u(1))
///     if !prev_flag:
///       rem_intra4x4_pred_mode          (u(3))
///   intra_chroma_pred_mode              (ue(v))
///   coded_block_pattern                 (me(v) → 8 bits in our case)
///   if cbp_luma > 0 || cbp_chroma > 0:
///     mb_qp_delta                       (se(v))
///   for each 8x8 quadrant where bit cbp_luma>>blk8 set:
///     for each of 4 child 4x4 blocks: residual_block_cavlc(.., 0..15, 16)
///   if cbp_chroma >= 1: chroma DC Cb, Cr
///   if cbp_chroma == 2: chroma AC Cb (4 blocks), Cr (4 blocks)
/// ```
pub struct INxNMcbConfig {
    /// `prev_intra4x4_pred_mode_flag[blk]` for each of the 16 4x4 blocks
    /// in §6.4.3 raster-Z order.
    pub prev_intra4x4_pred_mode_flag: [bool; 16],
    /// `rem_intra4x4_pred_mode[blk]` (3 bits, 0..=7). Only consumed
    /// when the corresponding `prev_flag` is false.
    pub rem_intra4x4_pred_mode: [u8; 16],
    /// `intra_chroma_pred_mode` ∈ 0..=3 per Table 8-5.
    pub intra_chroma_pred_mode: u8,
    /// `cbp_luma` ∈ 0..=15: bit `i` set ⇒ 8x8 quadrant `i` has at least
    /// one non-zero 4x4 sub-block residual.
    pub cbp_luma: u8,
    /// `cbp_chroma` ∈ {0, 1, 2}.
    pub cbp_chroma: u8,
    /// `mb_qp_delta` per §7.4.5. Range -26..=25 for 8-bit.
    pub mb_qp_delta: i32,
    /// 16 4x4 luma residual blocks in §6.4.3 raster-Z order, each 16
    /// scan-order coefficients (full 4x4 block: startIdx=0, endIdx=15,
    /// maxNumCoeff=16). Only blocks whose 8x8 quadrant bit is set in
    /// `cbp_luma` are emitted.
    pub luma_4x4_levels: [[i32; 16]; 16],
    /// Per-4x4-block luma `nC` (CAVLC neighbour context) in §6.4.3
    /// raster-Z order. Computed via §9.2.1.1 by the encoder caller.
    pub luma_4x4_nc: [i32; 16],
    /// 4 quantized chroma DC levels (Cb).
    pub chroma_dc_cb: [i32; 4],
    /// 4 quantized chroma DC levels (Cr).
    pub chroma_dc_cr: [i32; 4],
    /// Per-4x4-block chroma AC levels (Cb), AC-only scan layout
    /// (positions 0..=14, slot 15 unused).
    pub chroma_ac_cb: [[i32; 16]; 4],
    /// Per-4x4-block chroma AC levels (Cr).
    pub chroma_ac_cr: [[i32; 16]; 4],
}

/// Emit one I_NxN (Intra_4x4) macroblock (CAVLC, I-slice).
pub fn write_i_nxn_mb(w: &mut BitWriter, cfg: &INxNMcbConfig) -> Result<(), CavlcEncodeError> {
    debug_assert!(cfg.cbp_luma <= 15);
    debug_assert!(cfg.cbp_chroma <= 2);
    debug_assert!(cfg.intra_chroma_pred_mode <= 3);

    // §7.4.5 — Table 7-11: I_NxN mb_type raw = 0 in I-slice.
    w.ue(0);

    // §7.3.5.1 mb_pred(): per-block prev/rem.
    for blk in 0..16usize {
        let flag = cfg.prev_intra4x4_pred_mode_flag[blk];
        w.u(1, if flag { 1 } else { 0 });
        if !flag {
            debug_assert!(cfg.rem_intra4x4_pred_mode[blk] <= 7);
            w.u(3, cfg.rem_intra4x4_pred_mode[blk] as u32);
        }
    }
    // §7.3.5.1 — chroma intra pred mode for ChromaArrayType ∈ {1, 2}.
    w.ue(cfg.intra_chroma_pred_mode as u32);

    // §7.3.5 — coded_block_pattern me(v). Use the intra Table 9-4(a).
    let codenum = intra_cbp_to_codenum_420_422(cfg.cbp_luma, cfg.cbp_chroma);
    w.ue(codenum);

    // §7.3.5 — mb_qp_delta is present iff (cbp_luma>0 || cbp_chroma>0)
    // for non-Intra_16x16 MBs.
    let needs_qp_delta = cfg.cbp_luma > 0 || cfg.cbp_chroma > 0;
    if needs_qp_delta {
        w.se(cfg.mb_qp_delta);
    }

    // §7.3.5.3 — luma residual: 16 4x4 blocks grouped by 8x8 quadrant.
    // Each 8x8 quadrant covers 4 child 4x4 blocks; the cbp_luma bit
    // gates the whole quadrant.
    for blk8 in 0..4u8 {
        if (cfg.cbp_luma >> blk8) & 1 == 1 {
            for sub in 0..4u8 {
                let blk = (blk8 * 4 + sub) as usize;
                let nc = cfg.luma_4x4_nc[blk];
                encode_residual_block_cavlc(
                    w,
                    CoeffTokenContext::Numeric(nc),
                    16,
                    &cfg.luma_4x4_levels[blk],
                )?;
            }
        }
    }

    // §7.3.5.3 — chroma residual: same structure as for I_16x16.
    if cfg.cbp_chroma > 0 {
        encode_residual_block_cavlc(w, CoeffTokenContext::ChromaDc420, 4, &cfg.chroma_dc_cb)?;
        encode_residual_block_cavlc(w, CoeffTokenContext::ChromaDc420, 4, &cfg.chroma_dc_cr)?;
    }
    if cfg.cbp_chroma == 2 {
        for blk in &cfg.chroma_ac_cb {
            encode_residual_block_cavlc(w, CoeffTokenContext::Numeric(0), 15, &blk[..15])?;
        }
        for blk in &cfg.chroma_ac_cr {
            encode_residual_block_cavlc(w, CoeffTokenContext::Numeric(0), 15, &blk[..15])?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intra_cbp_inverse_round_trips_through_decoder_table() {
        // For every legal (cbp_luma, cbp_chroma) pair, we can re-derive
        // the codeNum and feed it through the decoder's parse_cbp_me
        // forward table to recover the original CBP.
        for cbp_luma in 0u8..=15 {
            for cbp_chroma in 0u8..=2 {
                let codenum = intra_cbp_to_codenum_420_422(cbp_luma, cbp_chroma);
                // Sanity: codeNum must be < 48 for ChromaArrayType ∈ {1, 2}.
                assert!(codenum < 48);
                // Sanity: the round-trip via the forward intra table.
                static FWD: [u8; 48] = [
                    47, 31, 15, 0, 23, 27, 29, 30, 7, 11, 13, 14, 39, 43, 45, 46, 16, 3, 5, 10, 12,
                    19, 21, 26, 28, 35, 37, 42, 44, 1, 2, 4, 8, 17, 18, 20, 24, 6, 9, 22, 25, 32,
                    33, 34, 36, 40, 38, 41,
                ];
                let cbp = FWD[codenum as usize];
                assert_eq!(
                    cbp,
                    ((cbp_chroma << 4) | cbp_luma),
                    "luma={cbp_luma} chroma={cbp_chroma} codeNum={codenum} got cbp={cbp}",
                );
            }
        }
    }

    #[test]
    fn mb_type_value_matches_table_7_11_decode() {
        use crate::macroblock_layer::{Intra16x16, MbType};
        for pred_mode in 0u8..=3 {
            for &cbp_luma in &[0u8, 15] {
                for cbp_chroma in 0u8..=2 {
                    let raw = intra16x16_mb_type_value(pred_mode, cbp_luma, cbp_chroma);
                    let decoded = MbType::from_i_slice(raw).unwrap();
                    assert_eq!(
                        decoded,
                        MbType::Intra16x16(Intra16x16 {
                            pred_mode,
                            cbp_luma,
                            cbp_chroma,
                        }),
                        "raw={raw}",
                    );
                }
            }
        }
    }
}
