//! §7.3.5 — macroblock_layer emit (encoder side, I_16x16 DC-only path).
//!
//! Round-1 scope: every macroblock is `I_16x16` with `pred_mode == 2`
//! (DC luma) and `intra_chroma_pred_mode == 0` (DC chroma). Both
//! `cbp_luma` and `cbp_chroma` are 0 — only the luma DC block is
//! transmitted (every Intra_16x16 MB transmits it unconditionally per
//! §7.3.5.3.1). No chroma residual, no AC residual.
//!
//! Emitted syntax for one MB:
//!
//! ```text
//!   mb_type ue(v)               -- Table 7-11 row, picks (predmode, cbp_l, cbp_c)
//!   intra16x16_pred_mode (impl) -- carried by mb_type
//!   intra_chroma_pred_mode ue(v)
//!   coded_block_pattern me(v)   -- absent for Intra_16x16; CBP carried in mb_type
//!   mb_qp_delta se(v)           -- present iff CBP != 0 OR Intra_16x16 ⇒ always
//!   residual_block(LumaDC, 0..15, max=16)
//!   -- LumaAC blocks omitted (cbp_luma == 0)
//!   -- chroma DC + AC omitted (cbp_chroma == 0)
//! ```

use crate::cavlc::CoeffTokenContext;
use crate::encoder::bitstream::BitWriter;
use crate::encoder::cavlc::{encode_residual_block_cavlc, CavlcEncodeError};
use crate::encoder::transform::zigzag_scan_4x4;

/// Configuration for one Intra_16x16 macroblock (round-1 fixed-mode).
pub struct I16x16McbConfig {
    /// `pred_mode` ∈ 0..=3 per Table 8-4 (V/H/DC/Plane). Round-1 uses 2.
    pub pred_mode: u8,
    /// `intra_chroma_pred_mode` ∈ 0..=3 per Table 8-5 (DC/H/V/Plane).
    /// Round-1 uses 0.
    pub intra_chroma_pred_mode: u8,
    /// `cbp_luma` ∈ {0, 15}.
    pub cbp_luma: u8,
    /// `cbp_chroma` ∈ {0, 1, 2}.
    pub cbp_chroma: u8,
    /// `mb_qp_delta` per §7.4.5. Range -26..=25 for 8-bit.
    pub mb_qp_delta: i32,
    /// 16 quantized luma DC coefficients in the 4x4 *raster* order
    /// `(blk4x4_y * 4 + blk4x4_x)` — i.e. row-major DC block. The
    /// encoder applies the zig-zag scan internally before CAVLC.
    pub luma_dc_levels_raster: [i32; 16],
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

/// Emit one Intra_16x16 macroblock (CAVLC, I-slice, round-1 path).
///
/// `nc_ctx` is the `nC` context for the luma DC residual block (always
/// `Numeric(0)` for the first MB; subsequent MBs derive it from
/// neighbouring TotalCoeff values per §9.2.1.1 — round-1 uses `0` for
/// every MB which is a valid (if sub-optimal) encoder choice; the
/// decoder accepts whichever nC the encoder picked because nC depends
/// only on neighbour state both sides see identically. NOTE: that's
/// not actually true — the decoder *derives* nC from neighbour state
/// itself, so the encoder must use the same derivation. Round-1 uses
/// the neighbour-aware `nC` value computed by the caller.).
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
    // §7.3.5 — coded_block_pattern is *absent* for Intra_16x16 (the CBP
    // is carried by mb_type itself). Skip the me(v) emission.

    // §7.3.5 — mb_qp_delta is present whenever the MB has at least one
    // coded coefficient (Intra_16x16 always has the luma DC block, so
    // mb_qp_delta is always emitted here).
    w.se(cfg.mb_qp_delta);

    // §7.3.5.3 — Intra_16x16 luma DC block.
    // Apply forward zig-zag, then CAVLC-encode with maxNumCoeff = 16.
    let scan = zigzag_scan_4x4(&cfg.luma_dc_levels_raster);
    encode_residual_block_cavlc(w, nc_ctx, 16, &scan)?;
    // cbp_luma == 0 → no Intra16x16ACLevel blocks.
    // cbp_chroma == 0 → no chroma DC nor chroma AC blocks.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
