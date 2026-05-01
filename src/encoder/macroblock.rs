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

// ---------------------------------------------------------------------------
// P_L0_16x16 (single-reference inter MB) — round 16.
// ---------------------------------------------------------------------------

/// §9.1.2 + Table 9-4(a) Inter column — inverse mapping CBP → codeNum
/// for inter macroblocks (P/B slices). `INTER_CBP_TO_CODENUM_420_422[cbp]`
/// returns the `me(v)` codeNum that the decoder will map back to `cbp`
/// via `ME_INTER_420_422`.
const INTER_CBP_TO_CODENUM_420_422: [u8; 48] = {
    // Forward (decoder) table reproduced inline so the spec citation is
    // local. Identical to `crate::macroblock_layer::ME_INTER_420_422`.
    const FWD: [u8; 48] = [
        0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37, 42, 44, 33,
        34, 36, 40, 39, 43, 45, 46, 17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
    ];
    let mut inv = [0u8; 48];
    let mut i = 0;
    while i < 48 {
        inv[FWD[i] as usize] = i as u8;
        i += 1;
    }
    inv
};

/// §9.1.2 — encode `coded_block_pattern` for a P/B inter MB on
/// ChromaArrayType ∈ {1, 2}.
pub fn inter_cbp_to_codenum_420_422(cbp_luma: u8, cbp_chroma: u8) -> u32 {
    debug_assert!(cbp_luma <= 15);
    debug_assert!(cbp_chroma <= 2);
    let cbp = ((cbp_chroma as usize) << 4) | (cbp_luma as usize);
    INTER_CBP_TO_CODENUM_420_422[cbp] as u32
}

/// Configuration for one P_L0_16x16 macroblock (round-16 inter encoder).
///
/// Layout per §7.3.5 / §7.3.5.1 with `mb_type == P_L0_16x16` (raw value
/// 0 in P-slice Table 7-13):
///
/// ```text
///   mb_type = 0                          (ue(v) → P_L0_16x16)
///   ref_idx_l0[0]                        (te(v); ABSENT when num_ref_idx_l0_active_minus1 == 0)
///   mvd_l0_x, mvd_l0_y                   (se(v), se(v))
///   coded_block_pattern                  (me(v); inter Table 9-4(a))
///   if cbp_luma > 0 || cbp_chroma > 0:
///     mb_qp_delta                        (se(v))
///   for each 8x8 quadrant where bit cbp_luma>>blk8 set:
///     for each of 4 child 4x4 blocks:    residual_block(0..15, 16)
///   if cbp_chroma >= 1: chroma DC Cb, Cr
///   if cbp_chroma == 2: chroma AC Cb (4), Cr (4)
/// ```
///
/// `mvd_l0_*` are in **quarter-pel units** (i.e. `mv = pel * 4`), per
/// §8.4.1.4 / §7.4.5.1.
pub struct PL016x16McbConfig {
    /// Motion vector difference, x component, in quarter-pel units.
    pub mvd_l0_x: i32,
    /// Motion vector difference, y component, in quarter-pel units.
    pub mvd_l0_y: i32,
    /// `cbp_luma` ∈ 0..=15.
    pub cbp_luma: u8,
    /// `cbp_chroma` ∈ {0, 1, 2}.
    pub cbp_chroma: u8,
    /// `mb_qp_delta` per §7.4.5. Range -26..=25 for 8-bit. Only emitted
    /// when (`cbp_luma > 0 || cbp_chroma > 0`).
    pub mb_qp_delta: i32,
    /// 16 4x4 luma residual blocks in §6.4.3 raster-Z order. Each block
    /// is 16 scan-order coefficients (full DC + AC; startIdx=0,
    /// endIdx=15, maxNumCoeff=16). Only blocks whose 8x8 quadrant bit
    /// is set in `cbp_luma` are emitted.
    pub luma_4x4_levels: [[i32; 16]; 16],
    /// Per-luma-4x4-block `nC` (CAVLC neighbour context) in §6.4.3
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

/// Emit one P_L0_16x16 macroblock (CAVLC, P-slice).
///
/// `num_ref_idx_l0_active_minus1` controls whether `ref_idx_l0` is
/// emitted: when 0 the field is **absent** per §7.3.5.1 (single ref).
/// Our slice header sets `num_ref_idx_active_override_flag = 0` and the
/// PPS default is 0 → callers should pass 0 here.
pub fn write_p_l0_16x16_mb(
    w: &mut BitWriter,
    cfg: &PL016x16McbConfig,
    num_ref_idx_l0_active_minus1: u32,
) -> Result<(), CavlcEncodeError> {
    debug_assert!(cfg.cbp_luma <= 15);
    debug_assert!(cfg.cbp_chroma <= 2);

    // §7.4.5 — Table 7-13: P_L0_16x16 mb_type raw = 0 in P-slice.
    w.ue(0);

    // §7.3.5.1 mb_pred(): ref_idx_l0 te(v) — present only when
    // (num_ref_idx_l0_active_minus1 > 0 || mbaff_field_change). Our
    // single-ref path skips it.
    if num_ref_idx_l0_active_minus1 > 0 {
        w.te(num_ref_idx_l0_active_minus1, 0);
    }

    // §7.3.5.1 — mvd_l0_x se(v), mvd_l0_y se(v) for the single
    // 16x16 partition. (MbPartPredMode != Pred_L1, not Direct.)
    w.se(cfg.mvd_l0_x);
    w.se(cfg.mvd_l0_y);

    // §7.3.5 — coded_block_pattern me(v) using the inter table.
    let codenum = inter_cbp_to_codenum_420_422(cfg.cbp_luma, cfg.cbp_chroma);
    w.ue(codenum);

    // §7.3.5 — mb_qp_delta is present iff (cbp_luma>0 || cbp_chroma>0).
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

// ---------------------------------------------------------------------------
// P_8x8 (4MV: four 8x8 sub-MB partitions, each PL08x8) — round 19.
// ---------------------------------------------------------------------------

/// Configuration for one P_8x8 macroblock with all four sub-blocks
/// using `sub_mb_type = PL08x8` (one MV per 8x8 partition — i.e. the
/// classic "4MV" mode). Round-19 keeps the simpler PL08x4/PL04x8/
/// PL04x4 sub-modes for later rounds.
///
/// Layout per §7.3.5 / §7.3.5.2 with `mb_type == P_8x8` (raw value 3
/// in P-slice Table 7-13) and four `sub_mb_type` entries (raw value 0
/// → PL08x8 in Table 7-17):
///
/// ```text
///   mb_type = 3                          (ue(v) → P_8x8)
///   for i in 0..4: sub_mb_type[i] = 0    (ue(v) → PL08x8)
///   for i in 0..4:  ref_idx_l0[i]        (te(v); ABSENT when num_ref_idx_l0_active_minus1 == 0)
///   for i in 0..4:  mvd_l0[i] = (x, y)   (se(v), se(v))
///   coded_block_pattern                  (me(v); inter Table 9-4(a))
///   if cbp_luma > 0 || cbp_chroma > 0:
///     mb_qp_delta                        (se(v))
///   for each 8x8 quadrant where bit cbp_luma>>blk8 set:
///     for each of 4 child 4x4 blocks:    residual_block(0..15, 16)
///   if cbp_chroma >= 1: chroma DC Cb, Cr
///   if cbp_chroma == 2: chroma AC Cb (4), Cr (4)
/// ```
///
/// `mvd_l0` are in **quarter-pel units** per §7.4.5.1 / §8.4.1.4.
pub struct P8x8AllPL08x8McbConfig {
    /// Per-8x8-partition mvd (quarter-pel units), in §6.4.3 8x8
    /// raster order: (0=TL, 1=TR, 2=BL, 3=BR). Same order as
    /// `LUMA_4X4_BLK[blk*4]` indexes the top-left 4x4 of each 8x8.
    pub mvd_l0: [(i32, i32); 4],
    /// `cbp_luma` ∈ 0..=15.
    pub cbp_luma: u8,
    /// `cbp_chroma` ∈ {0, 1, 2}.
    pub cbp_chroma: u8,
    /// `mb_qp_delta` per §7.4.5. Range -26..=25 for 8-bit. Only emitted
    /// when (`cbp_luma > 0 || cbp_chroma > 0`).
    pub mb_qp_delta: i32,
    /// 16 4x4 luma residual blocks in §6.4.3 raster-Z order. Same
    /// layout as `PL016x16McbConfig::luma_4x4_levels`.
    pub luma_4x4_levels: [[i32; 16]; 16],
    /// Per-luma-4x4-block `nC` (CAVLC neighbour context).
    pub luma_4x4_nc: [i32; 16],
    /// 4 quantized chroma DC levels (Cb).
    pub chroma_dc_cb: [i32; 4],
    /// 4 quantized chroma DC levels (Cr).
    pub chroma_dc_cr: [i32; 4],
    /// Per-4x4-block chroma AC levels (Cb).
    pub chroma_ac_cb: [[i32; 16]; 4],
    /// Per-4x4-block chroma AC levels (Cr).
    pub chroma_ac_cr: [[i32; 16]; 4],
}

/// Emit one P_8x8 macroblock with all four sub_mb_type set to
/// `PL08x8` (CAVLC, P-slice). See [`P8x8AllPL08x8McbConfig`] for the
/// syntax layout.
pub fn write_p_8x8_all_pl08x8_mb(
    w: &mut BitWriter,
    cfg: &P8x8AllPL08x8McbConfig,
    num_ref_idx_l0_active_minus1: u32,
) -> Result<(), CavlcEncodeError> {
    debug_assert!(cfg.cbp_luma <= 15);
    debug_assert!(cfg.cbp_chroma <= 2);

    // §7.4.5 — Table 7-13: P_8x8 mb_type raw = 3 in P-slice.
    w.ue(3);

    // §7.3.5.2 sub_mb_pred(): four sub_mb_type entries, all
    // PL08x8 (raw 0) here.
    for _ in 0..4 {
        w.ue(0);
    }

    // §7.3.5.2 ref_idx_l0 — gated by (num_ref_idx_l0_active_minus1 > 0)
    // and (sub_mb_type[i] != B_Direct_8x8) and (SubMbPredMode != Pred_L1).
    // PL08x8 has SubMbPredMode == Pred_L0, so the only gate is the
    // num_ref_idx check. Round-19 single-ref → absent.
    if num_ref_idx_l0_active_minus1 > 0 {
        for _ in 0..4 {
            w.te(num_ref_idx_l0_active_minus1, 0);
        }
    }

    // §7.3.5.2 mvd_l0 — for each 8x8 partition, NumSubMbPart(PL08x8) == 1
    // mvd. Emit four (mvd_x, mvd_y) pairs.
    for &(mvd_x, mvd_y) in cfg.mvd_l0.iter() {
        w.se(mvd_x);
        w.se(mvd_y);
    }

    // §7.3.5 — coded_block_pattern me(v) using the inter table.
    let codenum = inter_cbp_to_codenum_420_422(cfg.cbp_luma, cfg.cbp_chroma);
    w.ue(codenum);

    // §7.3.5 — mb_qp_delta is present iff (cbp_luma>0 || cbp_chroma>0).
    let needs_qp_delta = cfg.cbp_luma > 0 || cfg.cbp_chroma > 0;
    if needs_qp_delta {
        w.se(cfg.mb_qp_delta);
    }

    // §7.3.5.3 — luma residual: 16 4x4 blocks grouped by 8x8 quadrant.
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

    // §7.3.5.3 — chroma residual: same structure as I_16x16 / P_L0_16x16.
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

// ---------------------------------------------------------------------------
// B-slice 16x16 explicit-inter macroblocks — round 20.
// ---------------------------------------------------------------------------

/// §7.4.5 Table 7-14 — which list(s) a B 16x16 MB references.
///
/// Maps to raw mb_type values 1..=3. Round-20 only emits these three
/// "explicit, single 16x16 partition" types — B_Direct_16x16 (raw 0)
/// and B_Skip / B_Direct_8x8 / mixed B_Lx_Lx are all out of scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BPred16x16 {
    /// `B_L0_16x16` — single 16x16 partition predicted from L0 only.
    L0,
    /// `B_L1_16x16` — single 16x16 partition predicted from L1 only.
    L1,
    /// `B_Bi_16x16` — single 16x16 partition predicted from both lists,
    /// combined per §8.4.2.3 (default weighted = `(L0 + L1 + 1) >> 1`).
    Bi,
}

impl BPred16x16 {
    /// Table 7-14 raw mb_type value.
    pub fn mb_type_raw(self) -> u32 {
        match self {
            BPred16x16::L0 => 1,
            BPred16x16::L1 => 2,
            BPred16x16::Bi => 3,
        }
    }

    /// True iff this prediction reads from list 0 (L0 or Bi).
    pub fn uses_l0(self) -> bool {
        matches!(self, BPred16x16::L0 | BPred16x16::Bi)
    }

    /// True iff this prediction reads from list 1 (L1 or Bi).
    pub fn uses_l1(self) -> bool {
        matches!(self, BPred16x16::L1 | BPred16x16::Bi)
    }
}

/// Configuration for one B-slice 16x16 explicit-inter macroblock
/// (`B_L0_16x16` / `B_L1_16x16` / `B_Bi_16x16`).
///
/// Layout per §7.3.5 / §7.3.5.1 with `mb_type` ∈ {1, 2, 3} (raw values
/// in B-slice Table 7-14):
///
/// ```text
///   mb_type ue(v)                        (1 = B_L0, 2 = B_L1, 3 = B_Bi)
///   if uses_l0:
///     ref_idx_l0[0]                      (te(v); ABSENT when num_ref_idx_l0_active_minus1 == 0)
///   if uses_l1:
///     ref_idx_l1[0]                      (te(v); ABSENT when num_ref_idx_l1_active_minus1 == 0)
///   if uses_l0:
///     mvd_l0_x, mvd_l0_y                 (se(v), se(v))
///   if uses_l1:
///     mvd_l1_x, mvd_l1_y                 (se(v), se(v))
///   coded_block_pattern                  (me(v); inter Table 9-4(a))
///   if cbp_luma > 0 || cbp_chroma > 0:
///     mb_qp_delta                        (se(v))
///   for each 8x8 quadrant where bit cbp_luma>>blk8 set:
///     for each of 4 child 4x4 blocks:    residual_block(0..15, 16)
///   if cbp_chroma >= 1: chroma DC Cb, Cr
///   if cbp_chroma == 2: chroma AC Cb (4), Cr (4)
/// ```
///
/// `mvd_*` are in **quarter-pel units** per §7.4.5.1 / §8.4.1.4.
pub struct B16x16McbConfig {
    /// Which list(s) this MB uses (drives mb_type and which mvd fields
    /// are emitted).
    pub pred: BPred16x16,
    /// L0 motion vector difference, x component, in quarter-pel units.
    /// Ignored when `pred == L1`.
    pub mvd_l0_x: i32,
    /// L0 motion vector difference, y component. Ignored when `pred == L1`.
    pub mvd_l0_y: i32,
    /// L1 motion vector difference, x component. Ignored when `pred == L0`.
    pub mvd_l1_x: i32,
    /// L1 motion vector difference, y component. Ignored when `pred == L0`.
    pub mvd_l1_y: i32,
    /// `cbp_luma` ∈ 0..=15.
    pub cbp_luma: u8,
    /// `cbp_chroma` ∈ {0, 1, 2}.
    pub cbp_chroma: u8,
    /// `mb_qp_delta` per §7.4.5. Range -26..=25 for 8-bit. Only emitted
    /// when (`cbp_luma > 0 || cbp_chroma > 0`).
    pub mb_qp_delta: i32,
    /// 16 4x4 luma residual blocks in §6.4.3 raster-Z order. Same
    /// layout as `PL016x16McbConfig::luma_4x4_levels`.
    pub luma_4x4_levels: [[i32; 16]; 16],
    /// Per-luma-4x4-block `nC` (CAVLC neighbour context) in §6.4.3
    /// raster-Z order.
    pub luma_4x4_nc: [i32; 16],
    /// 4 quantized chroma DC levels (Cb).
    pub chroma_dc_cb: [i32; 4],
    /// 4 quantized chroma DC levels (Cr).
    pub chroma_dc_cr: [i32; 4],
    /// Per-4x4-block chroma AC levels (Cb), AC-only scan layout.
    pub chroma_ac_cb: [[i32; 16]; 4],
    /// Per-4x4-block chroma AC levels (Cr).
    pub chroma_ac_cr: [[i32; 16]; 4],
}

/// Emit one B-slice explicit 16x16 macroblock (CAVLC, B-slice).
///
/// `num_ref_idx_lN_active_minus1` controls whether `ref_idx_lN` is
/// emitted: when 0 the field is **absent** per §7.3.5.1 (single ref
/// in that list).
pub fn write_b_16x16_mb(
    w: &mut BitWriter,
    cfg: &B16x16McbConfig,
    num_ref_idx_l0_active_minus1: u32,
    num_ref_idx_l1_active_minus1: u32,
) -> Result<(), CavlcEncodeError> {
    debug_assert!(cfg.cbp_luma <= 15);
    debug_assert!(cfg.cbp_chroma <= 2);

    // §7.4.5 — Table 7-14: raw mb_type 1/2/3 for B_L0_16x16 / B_L1_16x16 /
    // B_Bi_16x16. All three are NumMbPart == 1 (single 16x16 partition).
    w.ue(cfg.pred.mb_type_raw());

    // §7.3.5.1 mb_pred(): ref_idx_l0 then ref_idx_l1 (te(v)). Each is
    // absent iff num_ref_idx_lN_active_minus1 == 0.
    if cfg.pred.uses_l0() && num_ref_idx_l0_active_minus1 > 0 {
        w.te(num_ref_idx_l0_active_minus1, 0);
    }
    if cfg.pred.uses_l1() && num_ref_idx_l1_active_minus1 > 0 {
        w.te(num_ref_idx_l1_active_minus1, 0);
    }

    // §7.3.5.1 — mvd_l0 then mvd_l1 (each as (mvd_x, mvd_y) se(v)
    // pair). Bipred emits both pairs; L0/L1-only emits one.
    if cfg.pred.uses_l0() {
        w.se(cfg.mvd_l0_x);
        w.se(cfg.mvd_l0_y);
    }
    if cfg.pred.uses_l1() {
        w.se(cfg.mvd_l1_x);
        w.se(cfg.mvd_l1_y);
    }

    // §7.3.5 — coded_block_pattern me(v) using the inter table.
    let codenum = inter_cbp_to_codenum_420_422(cfg.cbp_luma, cfg.cbp_chroma);
    w.ue(codenum);

    // §7.3.5 — mb_qp_delta is present iff (cbp_luma>0 || cbp_chroma>0).
    let needs_qp_delta = cfg.cbp_luma > 0 || cfg.cbp_chroma > 0;
    if needs_qp_delta {
        w.se(cfg.mb_qp_delta);
    }

    // §7.3.5.3 — luma residual: 16 4x4 blocks grouped by 8x8 quadrant.
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

    // §7.3.5.3 — chroma residual: same structure as P_L0_16x16.
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

/// Configuration for one B-slice **B_Direct_16x16** macroblock
/// (raw mb_type 0 in Table 7-14). Round-21 entry point.
///
/// B_Direct_16x16 has no `mvd_l0` / `mvd_l1` / `ref_idx_l0` / `ref_idx_l1`
/// in the bitstream — the decoder derives those itself per §8.4.1.2 (via
/// §8.4.1.2.2 spatial-direct or §8.4.1.2.3 temporal-direct depending on
/// `direct_spatial_mv_pred_flag`). The encoder is expected to mirror the
/// derivation locally and build a recon predictor from the same MVs.
///
/// Layout per §7.3.5 / §7.3.5.1 with `mb_type = 0`:
///
/// ```text
///   mb_type ue(v)                      (codeNum 0 = B_Direct_16x16)
///   coded_block_pattern me(v)          (inter Table 9-4(a))
///   if cbp_luma > 0 || cbp_chroma > 0:
///     mb_qp_delta se(v)
///   if cbp_luma > 0:
///     16 luma 4x4 residuals (raster-Z, by 8x8 quadrant)
///   if cbp_chroma >= 1: chroma DC Cb, Cr
///   if cbp_chroma == 2: chroma AC Cb (4), Cr (4)
/// ```
pub struct BDirect16x16McbConfig {
    /// `cbp_luma` ∈ 0..=15.
    pub cbp_luma: u8,
    /// `cbp_chroma` ∈ {0, 1, 2}.
    pub cbp_chroma: u8,
    /// `mb_qp_delta` per §7.4.5. Only emitted when (cbp_luma>0 || cbp_chroma>0).
    pub mb_qp_delta: i32,
    /// 16 4x4 luma residual blocks in §6.4.3 raster-Z order.
    pub luma_4x4_levels: [[i32; 16]; 16],
    /// Per-luma-4x4-block `nC` (CAVLC neighbour context).
    pub luma_4x4_nc: [i32; 16],
    /// 4 quantized chroma DC levels (Cb).
    pub chroma_dc_cb: [i32; 4],
    /// 4 quantized chroma DC levels (Cr).
    pub chroma_dc_cr: [i32; 4],
    /// Per-4x4-block chroma AC levels (Cb), AC-only scan layout.
    pub chroma_ac_cb: [[i32; 16]; 4],
    /// Per-4x4-block chroma AC levels (Cr).
    pub chroma_ac_cr: [[i32; 16]; 4],
}

/// Emit one B-slice `B_Direct_16x16` macroblock (CAVLC).
///
/// `mb_type = 0` (raw value in Table 7-14). No mvd / ref_idx in the
/// bitstream — the decoder derives them per §8.4.1.2 (spatial direct
/// when `direct_spatial_mv_pred_flag == 1`, temporal direct otherwise).
pub fn write_b_direct_16x16_mb(
    w: &mut BitWriter,
    cfg: &BDirect16x16McbConfig,
) -> Result<(), CavlcEncodeError> {
    debug_assert!(cfg.cbp_luma <= 15);
    debug_assert!(cfg.cbp_chroma <= 2);

    // §7.4.5 Table 7-14 — raw mb_type 0 = B_Direct_16x16.
    w.ue(0);

    // §7.3.5 — coded_block_pattern me(v) using the inter table.
    let codenum = inter_cbp_to_codenum_420_422(cfg.cbp_luma, cfg.cbp_chroma);
    w.ue(codenum);

    // §7.3.5 — mb_qp_delta is present iff (cbp_luma>0 || cbp_chroma>0).
    let needs_qp_delta = cfg.cbp_luma > 0 || cfg.cbp_chroma > 0;
    if needs_qp_delta {
        w.se(cfg.mb_qp_delta);
    }

    // §7.3.5.3 — luma residual blocks (when cbp_luma != 0).
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

    // §7.3.5.3 — chroma residual: same structure as P_L0_16x16 / B_L0_16x16.
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

// ---------------------------------------------------------------------------
// Round 22 — B-slice 16x8 / 8x16 partitions (Table 7-14 raw 4..=21).
// ---------------------------------------------------------------------------

/// §7.4.5 Table 7-14 — per-partition prediction list selector.
///
/// Each MB partition (one of two 16x8 halves or two 8x16 halves) picks
/// from {L0-only, L1-only, BiPred}. The pair (top, bottom) for 16x8 or
/// (left, right) for 8x16 enumerates the 9 mb_type rows 4..=21 in the
/// spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BPartPred {
    /// `Pred_L0` — one MV from list 0.
    L0,
    /// `Pred_L1` — one MV from list 1.
    L1,
    /// `BiPred` — both lists, combined per §8.4.2.3 (default weighted
    /// average `(L0 + L1 + 1) >> 1` when `weighted_bipred_idc == 0`).
    Bi,
}

impl BPartPred {
    /// True iff this partition reads from list 0 (L0 or Bi).
    pub fn uses_l0(self) -> bool {
        matches!(self, BPartPred::L0 | BPartPred::Bi)
    }

    /// True iff this partition reads from list 1 (L1 or Bi).
    pub fn uses_l1(self) -> bool {
        matches!(self, BPartPred::L1 | BPartPred::Bi)
    }
}

/// §7.4.5 Table 7-14 — raw mb_type for B_*_*_16x8 (rows 4..=20 even).
///
/// `(top, bottom)` → raw value mapping per the table:
/// ```text
///   (L0, L0)  →  4   (B_L0_L0_16x8)
///   (L1, L1)  →  6   (B_L1_L1_16x8)
///   (L0, L1)  →  8   (B_L0_L1_16x8)
///   (L1, L0)  → 10   (B_L1_L0_16x8)
///   (L0, Bi)  → 12   (B_L0_Bi_16x8)
///   (L1, Bi)  → 14   (B_L1_Bi_16x8)
///   (Bi, L0)  → 16   (B_Bi_L0_16x8)
///   (Bi, L1)  → 18   (B_Bi_L1_16x8)
///   (Bi, Bi)  → 20   (B_Bi_Bi_16x8)
/// ```
pub fn b_16x8_mb_type_raw(top: BPartPred, bottom: BPartPred) -> u32 {
    use BPartPred::*;
    match (top, bottom) {
        (L0, L0) => 4,
        (L1, L1) => 6,
        (L0, L1) => 8,
        (L1, L0) => 10,
        (L0, Bi) => 12,
        (L1, Bi) => 14,
        (Bi, L0) => 16,
        (Bi, L1) => 18,
        (Bi, Bi) => 20,
    }
}

/// §7.4.5 Table 7-14 — raw mb_type for B_*_*_8x16 (rows 5..=21 odd).
///
/// `(left, right)` → raw value mapping per the table:
/// ```text
///   (L0, L0)  →  5   (B_L0_L0_8x16)
///   (L1, L1)  →  7   (B_L1_L1_8x16)
///   (L0, L1)  →  9   (B_L0_L1_8x16)
///   (L1, L0)  → 11   (B_L1_L0_8x16)
///   (L0, Bi)  → 13   (B_L0_Bi_8x16)
///   (L1, Bi)  → 15   (B_L1_Bi_8x16)
///   (Bi, L0)  → 17   (B_Bi_L0_8x16)
///   (Bi, L1)  → 19   (B_Bi_L1_8x16)
///   (Bi, Bi)  → 21   (B_Bi_Bi_8x16)
/// ```
pub fn b_8x16_mb_type_raw(left: BPartPred, right: BPartPred) -> u32 {
    use BPartPred::*;
    match (left, right) {
        (L0, L0) => 5,
        (L1, L1) => 7,
        (L0, L1) => 9,
        (L1, L0) => 11,
        (L0, Bi) => 13,
        (L1, Bi) => 15,
        (Bi, L0) => 17,
        (Bi, L1) => 19,
        (Bi, Bi) => 21,
    }
}

/// Configuration for one B-slice 16x8 macroblock (mb_type ∈ {4, 6, 8,
/// 10, 12, 14, 16, 18, 20}). Two 16x8 partitions stacked vertically.
///
/// Layout per §7.3.5 / §7.3.5.1:
///
/// ```text
///   mb_type ue(v)                            (per b_16x8_mb_type_raw)
///   for part in 0..2:                       -- top, then bottom
///     if part uses L0 and num_ref_l0 > 0: ref_idx_l0[part]   (te(v))
///   for part in 0..2:
///     if part uses L1 and num_ref_l1 > 0: ref_idx_l1[part]   (te(v))
///   for part in 0..2:
///     if part uses L0: mvd_l0[part].x, mvd_l0[part].y        (se(v) x2)
///   for part in 0..2:
///     if part uses L1: mvd_l1[part].x, mvd_l1[part].y        (se(v) x2)
///   coded_block_pattern me(v)
///   if cbp != 0: mb_qp_delta se(v)
///   ... residuals (same layout as the 16x16 path) ...
/// ```
pub struct B16x8McbConfig {
    /// Top partition prediction list selection.
    pub top: BPartPred,
    /// Bottom partition prediction list selection.
    pub bottom: BPartPred,
    /// Per-partition L0 mvd. Index 0 = top, 1 = bottom. Ignored on
    /// partitions whose list selection doesn't use L0.
    pub mvd_l0: [(i32, i32); 2],
    /// Per-partition L1 mvd. Index 0 = top, 1 = bottom. Ignored on
    /// partitions whose list selection doesn't use L1.
    pub mvd_l1: [(i32, i32); 2],
    pub cbp_luma: u8,
    pub cbp_chroma: u8,
    pub mb_qp_delta: i32,
    /// 16 4x4 luma residual blocks in §6.4.3 raster-Z order.
    pub luma_4x4_levels: [[i32; 16]; 16],
    pub luma_4x4_nc: [i32; 16],
    pub chroma_dc_cb: [i32; 4],
    pub chroma_dc_cr: [i32; 4],
    pub chroma_ac_cb: [[i32; 16]; 4],
    pub chroma_ac_cr: [[i32; 16]; 4],
}

/// Emit one B-slice 16x8 macroblock (CAVLC, B-slice). Mirrors the field
/// order in §7.3.5.1 with `NumMbPart == 2`.
pub fn write_b_16x8_mb(
    w: &mut BitWriter,
    cfg: &B16x8McbConfig,
    num_ref_idx_l0_active_minus1: u32,
    num_ref_idx_l1_active_minus1: u32,
) -> Result<(), CavlcEncodeError> {
    debug_assert!(cfg.cbp_luma <= 15);
    debug_assert!(cfg.cbp_chroma <= 2);

    // §7.4.5 — Table 7-14 mb_type for the (top, bottom) pair.
    w.ue(b_16x8_mb_type_raw(cfg.top, cfg.bottom));

    write_b_two_part_pred(
        w,
        [cfg.top, cfg.bottom],
        cfg.mvd_l0,
        cfg.mvd_l1,
        num_ref_idx_l0_active_minus1,
        num_ref_idx_l1_active_minus1,
    );
    write_b_two_part_residuals(
        w,
        cfg.cbp_luma,
        cfg.cbp_chroma,
        cfg.mb_qp_delta,
        &cfg.luma_4x4_levels,
        &cfg.luma_4x4_nc,
        &cfg.chroma_dc_cb,
        &cfg.chroma_dc_cr,
        &cfg.chroma_ac_cb,
        &cfg.chroma_ac_cr,
    )
}

/// Configuration for one B-slice 8x16 macroblock (mb_type ∈ {5, 7, 9,
/// 11, 13, 15, 17, 19, 21}). Two 8x16 partitions side-by-side.
pub struct B8x16McbConfig {
    /// Left partition prediction list selection.
    pub left: BPartPred,
    /// Right partition prediction list selection.
    pub right: BPartPred,
    /// Per-partition L0 mvd. Index 0 = left, 1 = right.
    pub mvd_l0: [(i32, i32); 2],
    /// Per-partition L1 mvd. Index 0 = left, 1 = right.
    pub mvd_l1: [(i32, i32); 2],
    pub cbp_luma: u8,
    pub cbp_chroma: u8,
    pub mb_qp_delta: i32,
    pub luma_4x4_levels: [[i32; 16]; 16],
    pub luma_4x4_nc: [i32; 16],
    pub chroma_dc_cb: [i32; 4],
    pub chroma_dc_cr: [i32; 4],
    pub chroma_ac_cb: [[i32; 16]; 4],
    pub chroma_ac_cr: [[i32; 16]; 4],
}

/// Emit one B-slice 8x16 macroblock. Field order identical to 16x8 —
/// only the mb_type and the partition geometry differ; §7.3.5.1 lists
/// per-partition fields in mbPartIdx order.
pub fn write_b_8x16_mb(
    w: &mut BitWriter,
    cfg: &B8x16McbConfig,
    num_ref_idx_l0_active_minus1: u32,
    num_ref_idx_l1_active_minus1: u32,
) -> Result<(), CavlcEncodeError> {
    debug_assert!(cfg.cbp_luma <= 15);
    debug_assert!(cfg.cbp_chroma <= 2);

    // §7.4.5 — Table 7-14 mb_type for the (left, right) pair.
    w.ue(b_8x16_mb_type_raw(cfg.left, cfg.right));

    write_b_two_part_pred(
        w,
        [cfg.left, cfg.right],
        cfg.mvd_l0,
        cfg.mvd_l1,
        num_ref_idx_l0_active_minus1,
        num_ref_idx_l1_active_minus1,
    );
    write_b_two_part_residuals(
        w,
        cfg.cbp_luma,
        cfg.cbp_chroma,
        cfg.mb_qp_delta,
        &cfg.luma_4x4_levels,
        &cfg.luma_4x4_nc,
        &cfg.chroma_dc_cb,
        &cfg.chroma_dc_cr,
        &cfg.chroma_ac_cb,
        &cfg.chroma_ac_cr,
    )
}

/// §7.3.5.1 — emit ref_idx_l0[0..=1], ref_idx_l1[0..=1], mvd_l0[0..=1],
/// mvd_l1[0..=1] for the two partitions of a B 16x8 or B 8x16 MB.
///
/// Order is per the spec: all ref_idx_l0 first (for both partitions),
/// then all ref_idx_l1, then all mvd_l0, then all mvd_l1. Each per-
/// partition field is gated on the partition's prediction list usage.
fn write_b_two_part_pred(
    w: &mut BitWriter,
    parts: [BPartPred; 2],
    mvd_l0: [(i32, i32); 2],
    mvd_l1: [(i32, i32); 2],
    num_ref_idx_l0_active_minus1: u32,
    num_ref_idx_l1_active_minus1: u32,
) {
    // ref_idx_l0[part] for part = 0, 1.
    for &p in &parts {
        if p.uses_l0() && num_ref_idx_l0_active_minus1 > 0 {
            w.te(num_ref_idx_l0_active_minus1, 0);
        }
    }
    // ref_idx_l1[part] for part = 0, 1.
    for &p in &parts {
        if p.uses_l1() && num_ref_idx_l1_active_minus1 > 0 {
            w.te(num_ref_idx_l1_active_minus1, 0);
        }
    }
    // mvd_l0[part] (.x, .y) for part = 0, 1.
    for (i, &p) in parts.iter().enumerate() {
        if p.uses_l0() {
            w.se(mvd_l0[i].0);
            w.se(mvd_l0[i].1);
        }
    }
    // mvd_l1[part] (.x, .y) for part = 0, 1.
    for (i, &p) in parts.iter().enumerate() {
        if p.uses_l1() {
            w.se(mvd_l1[i].0);
            w.se(mvd_l1[i].1);
        }
    }
}

/// §7.3.5 — emit coded_block_pattern + mb_qp_delta + luma + chroma
/// residuals. Identical layout to the 16x16 and Direct paths since the
/// residual encoding is partition-shape-agnostic (every 4x4 block is
/// emitted independently).
#[allow(clippy::too_many_arguments)]
fn write_b_two_part_residuals(
    w: &mut BitWriter,
    cbp_luma: u8,
    cbp_chroma: u8,
    mb_qp_delta: i32,
    luma_4x4_levels: &[[i32; 16]; 16],
    luma_4x4_nc: &[i32; 16],
    chroma_dc_cb: &[i32; 4],
    chroma_dc_cr: &[i32; 4],
    chroma_ac_cb: &[[i32; 16]; 4],
    chroma_ac_cr: &[[i32; 16]; 4],
) -> Result<(), CavlcEncodeError> {
    let codenum = inter_cbp_to_codenum_420_422(cbp_luma, cbp_chroma);
    w.ue(codenum);

    let needs_qp_delta = cbp_luma > 0 || cbp_chroma > 0;
    if needs_qp_delta {
        w.se(mb_qp_delta);
    }

    for blk8 in 0..4u8 {
        if (cbp_luma >> blk8) & 1 == 1 {
            for sub in 0..4u8 {
                let blk = (blk8 * 4 + sub) as usize;
                let nc = luma_4x4_nc[blk];
                encode_residual_block_cavlc(
                    w,
                    CoeffTokenContext::Numeric(nc),
                    16,
                    &luma_4x4_levels[blk],
                )?;
            }
        }
    }

    if cbp_chroma > 0 {
        encode_residual_block_cavlc(w, CoeffTokenContext::ChromaDc420, 4, chroma_dc_cb)?;
        encode_residual_block_cavlc(w, CoeffTokenContext::ChromaDc420, 4, chroma_dc_cr)?;
    }
    if cbp_chroma == 2 {
        for blk in chroma_ac_cb {
            encode_residual_block_cavlc(w, CoeffTokenContext::Numeric(0), 15, &blk[..15])?;
        }
        for blk in chroma_ac_cr {
            encode_residual_block_cavlc(w, CoeffTokenContext::Numeric(0), 15, &blk[..15])?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Round 23 — B_8x8 with all four sub_mb_type = B_Direct_8x8
// ---------------------------------------------------------------------------
//
// Spec layout (§7.3.5 + §7.3.5.2 + §7.4.5 Table 7-14 + Table 7-18):
//
//   mb_type ue(v)                     -- raw 22 = B_8x8 (Table 7-14)
//   sub_mb_pred(B_8x8) {
//     for i in 0..4: sub_mb_type[i]   -- raw 0 = B_Direct_8x8 (Table 7-18),
//                                        ue(v) "1" each (4 bits total).
//     -- ref_idx_l0/_l1 are gated on `sub_mb_type[i] != B_Direct_8x8` and
//        `SubMbPredMode != Pred_L1/_L0`. With every sub_mb_type =
//        B_Direct_8x8 the gate fails for all four → no ref_idx fields.
//     -- mvd_l0/_l1 same gating → no mvd fields.
//   }
//   coded_block_pattern me(v)         -- inter table (Table 9-4 column 2)
//   if (cbp_luma>0 || cbp_chroma>0): mb_qp_delta se(v)
//   ... residuals (same layout as the 16x16 / 16x8 / 8x16 paths) ...
//
// The decoder fully re-derives the per-8x8 (refIdxL0, refIdxL1, mvL0,
// mvL1) per §8.4.1.2.2 (spatial direct, since
// `direct_spatial_mv_pred_flag = 1` in our slice header). With
// `direct_8x8_inference_flag = 1` (hard-wired in our SPS), every 8x8
// sub-MB carries one (mvL0, mvL1) pair shared across its four 4x4
// sub-partitions, so the encoder mirrors that derivation locally and
// builds the 16x16 luma + 8x8 chroma predictors as four independent
// 8x8 / 4x4 quadrants stitched together.
//
// Spec sections cited:
//   §7.3.5            — macroblock_layer (this file already covers
//                       I_16x16 / inter paths)
//   §7.3.5.2          — sub_mb_pred() syntax with sub_mb_type[mbPartIdx]
//   §7.4.5 Table 7-14 — mb_type=22 → B_8x8
//   §7.4.5 Table 7-18 — sub_mb_type=0 → B_Direct_8x8 (NumSubMbPart=4
//                       conceptually but with direct_8x8_inference_flag=1
//                       a single MV/refIdx pair drives the whole 8x8;
//                       see §8.4.1.2 NOTE 3)
//   §8.4.1.2          — direct mode entry; B_Direct_8x8 dispatch
//   §8.4.1.2.2        — spatial direct derivation (steps 4-7)
//   §9.1.2 / Table 9-4(a) — inter CBP me(v) (already inverse-tabled here)

/// Configuration for one B-slice **B_8x8** macroblock with every
/// `sub_mb_type` equal to `B_Direct_8x8`. Round-23 entry point.
///
/// No `mvd_*`, no `ref_idx_*` in the bitstream — the decoder derives all
/// four (mvL0, mvL1, refIdxL0, refIdxL1) pairs per §8.4.1.2.2.
///
/// Layout per §7.3.5 / §7.3.5.2 with `mb_type = 22`:
///
/// ```text
///   mb_type ue(v)                      (codeNum 22 → 0000 0010 111)
///   for i in 0..4:
///     sub_mb_type[i] ue(v)             (codeNum 0 = B_Direct_8x8 → "1")
///   coded_block_pattern me(v)          (inter Table 9-4(a))
///   if cbp_luma > 0 || cbp_chroma > 0:
///     mb_qp_delta se(v)
///   if cbp_luma > 0:
///     16 luma 4x4 residuals (raster-Z, by 8x8 quadrant)
///   if cbp_chroma >= 1: chroma DC Cb, Cr
///   if cbp_chroma == 2: chroma AC Cb (4), Cr (4)
/// ```
pub struct B8x8AllDirectMcbConfig {
    /// `cbp_luma` ∈ 0..=15.
    pub cbp_luma: u8,
    /// `cbp_chroma` ∈ {0, 1, 2}.
    pub cbp_chroma: u8,
    /// `mb_qp_delta` per §7.4.5. Only emitted when (cbp_luma>0 || cbp_chroma>0).
    pub mb_qp_delta: i32,
    /// 16 4x4 luma residual blocks in §6.4.3 raster-Z order.
    pub luma_4x4_levels: [[i32; 16]; 16],
    /// Per-luma-4x4-block `nC` (CAVLC neighbour context).
    pub luma_4x4_nc: [i32; 16],
    /// 4 quantized chroma DC levels (Cb).
    pub chroma_dc_cb: [i32; 4],
    /// 4 quantized chroma DC levels (Cr).
    pub chroma_dc_cr: [i32; 4],
    /// Per-4x4-block chroma AC levels (Cb), AC-only scan layout.
    pub chroma_ac_cb: [[i32; 16]; 4],
    /// Per-4x4-block chroma AC levels (Cr).
    pub chroma_ac_cr: [[i32; 16]; 4],
}

/// Emit one B-slice `B_8x8` macroblock with all four sub-MBs as
/// `B_Direct_8x8` (CAVLC, round 23).
///
/// `mb_type = 22` (raw value in Table 7-14). Each `sub_mb_type[i] = 0`
/// (B_Direct_8x8 per Table 7-18). No mvd / ref_idx in the bitstream —
/// the decoder derives all four per-8x8 (mvL0, mvL1, refIdxL0,
/// refIdxL1) pairs per §8.4.1.2 (spatial direct here).
pub fn write_b_8x8_all_direct_mb(
    w: &mut BitWriter,
    cfg: &B8x8AllDirectMcbConfig,
) -> Result<(), CavlcEncodeError> {
    debug_assert!(cfg.cbp_luma <= 15);
    debug_assert!(cfg.cbp_chroma <= 2);

    // §7.4.5 Table 7-14 — raw mb_type 22 = B_8x8.
    w.ue(22);

    // §7.3.5.2 — sub_mb_type[0..4]: each = 0 → B_Direct_8x8.
    for _ in 0..4 {
        w.ue(0);
    }

    // §7.3.5.2 — ref_idx_l0/_l1 gated on `sub_mb_type[i] != B_Direct_8x8
    // && SubMbPredMode != Pred_L1/_L0`. All four sub_mb_types ARE
    // B_Direct_8x8 → all four gates fail → no ref_idx_l0[i] / ref_idx_l1[i]
    // emitted. Same story for mvd_l0/_l1.

    // §7.3.5 — coded_block_pattern me(v) using the inter table.
    let codenum = inter_cbp_to_codenum_420_422(cfg.cbp_luma, cfg.cbp_chroma);
    w.ue(codenum);

    // §7.3.5 — mb_qp_delta is present iff (cbp_luma>0 || cbp_chroma>0).
    let needs_qp_delta = cfg.cbp_luma > 0 || cfg.cbp_chroma > 0;
    if needs_qp_delta {
        w.se(cfg.mb_qp_delta);
    }

    // §7.3.5.3 — luma residual blocks (when cbp_luma != 0).
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

    // §7.3.5.3 — chroma residual: same layout as B_Direct_16x16.
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

// ---------------------------------------------------------------------------
// Round 25 — B_8x8 with mixed sub_mb_type (Direct/L0/L1/Bi per cell)
// ---------------------------------------------------------------------------
//
// Spec layout (§7.3.5 + §7.3.5.2 + §7.4.5 Table 7-14 + Table 7-18):
//
//   mb_type ue(v)                     -- raw 22 = B_8x8 (Table 7-14)
//   sub_mb_pred(B_8x8) {
//     for i in 0..4: sub_mb_type[i]   -- raw ∈ {0, 1, 2, 3} per cell
//                                        (B_Direct_8x8 / B_L0_8x8 /
//                                        B_L1_8x8 / B_Bi_8x8). Round-25
//                                        scope is one MV per cell — sub-
//                                        8x8 partition shapes (raws
//                                        4..=12) deferred to r26.
//     for i in 0..4: ref_idx_l0[i]    -- te(v); gated on:
//                                        sub_mb_type[i] != B_Direct_8x8
//                                        AND SubMbPredMode != Pred_L1
//                                        AND num_ref_idx_l0_active_minus1>0
//                                        (round-25: single ref → absent).
//     for i in 0..4: ref_idx_l1[i]    -- te(v); same gate with PredL0.
//     for i in 0..4:
//       if sub_mb_type[i] != B_Direct_8x8 AND
//          SubMbPredMode != Pred_L1:
//         mvd_l0[i] = (x, y)          -- se(v) × 2 (NumSubMbPart == 1)
//     for i in 0..4:
//       if sub_mb_type[i] != B_Direct_8x8 AND
//          SubMbPredMode != Pred_L0:
//         mvd_l1[i] = (x, y)          -- se(v) × 2
//   }
//   coded_block_pattern me(v)         -- inter Table 9-4(a)
//   if (cbp_luma>0 || cbp_chroma>0): mb_qp_delta se(v)
//   ... residuals (same layout as the 16x16 / 16x8 / 8x16 paths) ...
//
// Cells coded as B_Direct_8x8 are derived via §8.4.1.2.2 (spatial direct,
// our current configuration) — no MVs in the bitstream for those cells.
// Cells coded as B_L0_8x8 / B_L1_8x8 / B_Bi_8x8 carry one explicit
// (mvd_l0, mvd_l1) pair as appropriate.
//
// Spec sections cited:
//   §7.3.5            — macroblock_layer (this file already covers the
//                       base inter / intra paths)
//   §7.3.5.2          — sub_mb_pred() syntax with sub_mb_type[mbPartIdx]
//   §7.4.5 Table 7-14 — mb_type=22 → B_8x8
//   §7.4.5 Table 7-18 — sub_mb_type ∈ {0..3} → 1 8x8 partition each
//   §8.4.1.2          — direct mode entry; B_Direct_8x8 dispatch
//   §8.4.1.2.2        — spatial direct derivation (steps 4-7)
//   §9.1.2 / Table 9-4(a) — inter CBP me(v) (already inverse-tabled here)

/// Round-25 — per-8x8-cell prediction kind for a `B_8x8` macroblock with
/// mixed sub_mb_types. `Direct` maps to raw sub_mb_type 0
/// (B_Direct_8x8); `L0`/`L1`/`Bi` map to raw 1/2/3 (B_L0_8x8/B_L1_8x8/
/// B_Bi_8x8 per Table 7-18).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BSubMbCell {
    /// `B_Direct_8x8` — derived (mvL0, mvL1, refIdxL0, refIdxL1) via
    /// §8.4.1.2 dispatch (spatial or temporal direct per slice header).
    Direct,
    /// `B_L0_8x8` — explicit L0-only single 8x8 partition.
    L0,
    /// `B_L1_8x8` — explicit L1-only single 8x8 partition.
    L1,
    /// `B_Bi_8x8` — explicit bipred single 8x8 partition.
    Bi,
}

impl BSubMbCell {
    /// Table 7-18 raw sub_mb_type value for this cell.
    pub fn sub_mb_type_raw(self) -> u32 {
        match self {
            BSubMbCell::Direct => 0,
            BSubMbCell::L0 => 1,
            BSubMbCell::L1 => 2,
            BSubMbCell::Bi => 3,
        }
    }

    /// True iff this cell reads from list 0 (L0 or Bi). Direct cells
    /// derive their own (refIdxL0, refIdxL1) per §8.4.1.2 — the gate
    /// here only governs whether the encoder writes mvd_l0/ref_idx_l0
    /// fields in the bitstream.
    pub fn writes_l0(self) -> bool {
        matches!(self, BSubMbCell::L0 | BSubMbCell::Bi)
    }

    /// True iff this cell reads from list 1 (L1 or Bi). Same caveat as
    /// `writes_l0` for direct cells.
    pub fn writes_l1(self) -> bool {
        matches!(self, BSubMbCell::L1 | BSubMbCell::Bi)
    }
}

/// Configuration for one B-slice **B_8x8** macroblock with mixed
/// `sub_mb_type` cells (Direct / L0 / L1 / Bi per 8x8 quadrant).
/// Round-25 entry point.
///
/// Each non-Direct cell carries one (mvd_l0, mvd_l1) pair as appropriate
/// (NumSubMbPart == 1 for sub_mb_type ∈ {0, 1, 2, 3}). Direct cells
/// have no MVs / ref_idxs in the bitstream — the decoder re-derives
/// them via §8.4.1.2.2 / §8.4.1.2.3.
///
/// Layout per §7.3.5 / §7.3.5.2 with `mb_type = 22`:
///
/// ```text
///   mb_type ue(v)                      (codeNum 22 → "0000 0010 111")
///   for i in 0..4:
///     sub_mb_type[i] ue(v)             (codeNum ∈ {0,1,2,3})
///   for i in 0..4: ref_idx_l0[i] te(v) (only if cell writes L0 + multi-ref)
///   for i in 0..4: ref_idx_l1[i] te(v) (only if cell writes L1 + multi-ref)
///   for i in 0..4:
///     if cell writes L0: mvd_l0[i].x, mvd_l0[i].y  (se(v) × 2)
///   for i in 0..4:
///     if cell writes L1: mvd_l1[i].x, mvd_l1[i].y  (se(v) × 2)
///   coded_block_pattern me(v)          (inter Table 9-4(a))
///   if cbp_luma > 0 || cbp_chroma > 0:
///     mb_qp_delta se(v)
///   ... residuals (16 luma 4x4 by 8x8 quadrant + chroma DC/AC) ...
/// ```
pub struct B8x8MixedMcbConfig {
    /// Per-cell sub_mb_type pick (raster Z-order: TL, TR, BL, BR).
    pub cells: [BSubMbCell; 4],
    /// Per-cell L0 mvd. Ignored on cells whose `writes_l0()` is false.
    pub mvd_l0: [(i32, i32); 4],
    /// Per-cell L1 mvd. Ignored on cells whose `writes_l1()` is false.
    pub mvd_l1: [(i32, i32); 4],
    pub cbp_luma: u8,
    pub cbp_chroma: u8,
    pub mb_qp_delta: i32,
    pub luma_4x4_levels: [[i32; 16]; 16],
    pub luma_4x4_nc: [i32; 16],
    pub chroma_dc_cb: [i32; 4],
    pub chroma_dc_cr: [i32; 4],
    pub chroma_ac_cb: [[i32; 16]; 4],
    pub chroma_ac_cr: [[i32; 16]; 4],
}

/// Emit one B-slice `B_8x8` macroblock with mixed sub-MB types per
/// quadrant (round-25). Per-cell sub_mb_type ∈ {B_Direct_8x8, B_L0_8x8,
/// B_L1_8x8, B_Bi_8x8}.
///
/// `mb_type = 22` (raw value in Table 7-14). Each `sub_mb_type[i]` per
/// Table 7-18; non-Direct cells emit explicit ref_idx (gated on multi-
/// ref) + mvd_lX as required.
pub fn write_b_8x8_mixed_mb(
    w: &mut BitWriter,
    cfg: &B8x8MixedMcbConfig,
    num_ref_idx_l0_active_minus1: u32,
    num_ref_idx_l1_active_minus1: u32,
) -> Result<(), CavlcEncodeError> {
    debug_assert!(cfg.cbp_luma <= 15);
    debug_assert!(cfg.cbp_chroma <= 2);

    // §7.4.5 Table 7-14 — raw mb_type 22 = B_8x8.
    w.ue(22);

    // §7.3.5.2 — sub_mb_type[0..4]: per-cell raw ∈ {0, 1, 2, 3}.
    for &c in &cfg.cells {
        w.ue(c.sub_mb_type_raw());
    }

    // §7.3.5.2 ref_idx_l0[i] gate: sub_mb_type[i] != B_Direct_8x8 AND
    // SubMbPredMode != Pred_L1 AND num_ref_idx_l0_active_minus1 > 0.
    // Round-25 always uses single-ref (num_ref_idx_l0_active_minus1==0)
    // so these te(v) are never emitted, but the gate is honoured.
    if num_ref_idx_l0_active_minus1 > 0 {
        for &c in &cfg.cells {
            if c.writes_l0() {
                w.te(num_ref_idx_l0_active_minus1, 0);
            }
        }
    }
    // ref_idx_l1[i] gate — symmetric for Pred_L0.
    if num_ref_idx_l1_active_minus1 > 0 {
        for &c in &cfg.cells {
            if c.writes_l1() {
                w.te(num_ref_idx_l1_active_minus1, 0);
            }
        }
    }

    // §7.3.5.2 mvd_l0[i] gate: sub_mb_type[i] != B_Direct_8x8 AND
    // SubMbPredMode != Pred_L1. NumSubMbPart(B_L0_8x8/B_L1_8x8/
    // B_Bi_8x8) == 1 → exactly one (x, y) pair per non-Direct L0 cell.
    for (i, &c) in cfg.cells.iter().enumerate() {
        if c.writes_l0() {
            w.se(cfg.mvd_l0[i].0);
            w.se(cfg.mvd_l0[i].1);
        }
    }
    // mvd_l1[i] gate — symmetric for Pred_L0.
    for (i, &c) in cfg.cells.iter().enumerate() {
        if c.writes_l1() {
            w.se(cfg.mvd_l1[i].0);
            w.se(cfg.mvd_l1[i].1);
        }
    }

    // §7.3.5 — coded_block_pattern me(v) using the inter table.
    let codenum = inter_cbp_to_codenum_420_422(cfg.cbp_luma, cfg.cbp_chroma);
    w.ue(codenum);

    // §7.3.5 — mb_qp_delta is present iff (cbp_luma>0 || cbp_chroma>0).
    let needs_qp_delta = cfg.cbp_luma > 0 || cfg.cbp_chroma > 0;
    if needs_qp_delta {
        w.se(cfg.mb_qp_delta);
    }

    // §7.3.5.3 — luma residual: 16 4x4 blocks grouped by 8x8 quadrant.
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

    // §7.3.5.3 — chroma residual.
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
    fn inter_cbp_inverse_round_trips_through_decoder_table() {
        // For every legal (cbp_luma, cbp_chroma) pair, we re-derive the
        // codeNum and verify it round-trips through the decoder's
        // ME_INTER_420_422 forward table.
        for cbp_luma in 0u8..=15 {
            for cbp_chroma in 0u8..=2 {
                let codenum = inter_cbp_to_codenum_420_422(cbp_luma, cbp_chroma);
                assert!(codenum < 48);
                static FWD: [u8; 48] = [
                    0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37,
                    42, 44, 33, 34, 36, 40, 39, 43, 45, 46, 17, 18, 20, 24, 19, 21, 26, 28, 23, 27,
                    29, 30, 22, 25, 38, 41,
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

    /// §7.4.5 Table 7-14 — B-slice 16x16 explicit mb_types decode back
    /// to the expected `MbType` variants. Round-20 only emits raw
    /// values 1, 2, 3 (B_L0_16x16 / B_L1_16x16 / B_Bi_16x16).
    #[test]
    fn b16x16_mb_type_raw_matches_table_7_14_decode() {
        use crate::macroblock_layer::MbType;
        assert_eq!(BPred16x16::L0.mb_type_raw(), 1);
        assert_eq!(BPred16x16::L1.mb_type_raw(), 2);
        assert_eq!(BPred16x16::Bi.mb_type_raw(), 3);
        assert_eq!(MbType::from_b_slice(1).unwrap(), MbType::BL016x16);
        assert_eq!(MbType::from_b_slice(2).unwrap(), MbType::BL116x16);
        assert_eq!(MbType::from_b_slice(3).unwrap(), MbType::BBi16x16);

        assert!(BPred16x16::L0.uses_l0() && !BPred16x16::L0.uses_l1());
        assert!(!BPred16x16::L1.uses_l0() && BPred16x16::L1.uses_l1());
        assert!(BPred16x16::Bi.uses_l0() && BPred16x16::Bi.uses_l1());
    }

    /// §7.4.5 Table 7-14 — raw mb_type 0 round-trips to
    /// `MbType::BDirect16x16` (round-21).
    #[test]
    fn b_direct_16x16_mb_type_raw_matches_table_7_14_decode() {
        use crate::macroblock_layer::MbType;
        assert_eq!(MbType::from_b_slice(0).unwrap(), MbType::BDirect16x16);
    }

    /// §7.4.5 Table 7-14 — raw mb_type 22 round-trips to `MbType::B8x8`
    /// (round-23). Each sub_mb_type=0 in §7.4.5 Table 7-18 round-trips
    /// to `SubMbType::BDirect8x8`.
    #[test]
    fn b_8x8_mb_type_raw_matches_table_7_14_decode() {
        use crate::macroblock_layer::{MbType, SubMbType};
        assert_eq!(MbType::from_b_slice(22).unwrap(), MbType::B8x8);
        assert_eq!(SubMbType::from_b(0).unwrap(), SubMbType::BDirect8x8);
    }

    /// Round-23 `B_8x8` + 4× `B_Direct_8x8` writer with `cbp_luma=0,
    /// cbp_chroma=0` emits exactly the expected ue(v) sequence:
    ///
    ///   * mb_type=22  → ue(v) "0000 0010 111" (9 bits — codeNum 22)
    ///   * sub_mb_type=0 (×4) → ue(v) "1" each (4 bits)
    ///   * cbp codeNum 0 → ue(v) "1" (1 bit)
    ///
    /// No mb_qp_delta (gated on cbp > 0). Total 14 bits.
    #[test]
    fn b_8x8_all_direct_zero_residual_emits_expected_bit_count() {
        let mut bw = BitWriter::new();
        let cfg = B8x8AllDirectMcbConfig {
            cbp_luma: 0,
            cbp_chroma: 0,
            mb_qp_delta: 0,
            luma_4x4_levels: [[0; 16]; 16],
            luma_4x4_nc: [0; 16],
            chroma_dc_cb: [0; 4],
            chroma_dc_cr: [0; 4],
            chroma_ac_cb: [[0; 16]; 4],
            chroma_ac_cr: [[0; 16]; 4],
        };
        write_b_8x8_all_direct_mb(&mut bw, &cfg).expect("write");
        // ue(22): binValue=23, leadingZeros=4 → 4 zeros + "1" + 4
        // bit-suffix "0111" = 9 bits. Plus 4× ue(0)=1 bit + ue(0)=1 bit
        // = 14 bits total before any trailing alignment.
        assert_eq!(bw.bits_emitted(), 14);
    }

    /// Round-21 B_Direct_16x16 writer with `cbp_luma=0, cbp_chroma=0`
    /// emits exactly two ue(v): mb_type=0 (single '1') + cbp codeNum=0
    /// (single '1'). No mb_qp_delta (gated on cbp > 0). That is two
    /// bits total in the bitstream window we control before
    /// `rbsp_trailing_bits()`.
    #[test]
    fn b_direct_16x16_zero_residual_emits_two_bits() {
        let mut bw = BitWriter::new();
        let cfg = BDirect16x16McbConfig {
            cbp_luma: 0,
            cbp_chroma: 0,
            mb_qp_delta: 0,
            luma_4x4_levels: [[0; 16]; 16],
            luma_4x4_nc: [0; 16],
            chroma_dc_cb: [0; 4],
            chroma_dc_cr: [0; 4],
            chroma_ac_cb: [[0; 16]; 4],
            chroma_ac_cr: [[0; 16]; 4],
        };
        write_b_direct_16x16_mb(&mut bw, &cfg).expect("write");
        // mb_type=0 → ue(v) "1" (1 bit). cbp codeNum=0 → ue(v) "1" (1 bit).
        // Total 2 bits.
        assert_eq!(bw.bits_emitted(), 2);
    }

    /// §7.4.5 Table 7-14 — round-22 B 16x8 mb_type raw values 4..=20
    /// even all decode back to the expected (mode_top, mode_bottom)
    /// `MbType` variant. Includes round-trip for every `(top, bottom)`
    /// pair in BPartPred^2.
    #[test]
    fn b_16x8_mb_type_raw_matches_table_7_14_decode() {
        use crate::macroblock_layer::MbType;
        use BPartPred::*;
        let cases: &[(BPartPred, BPartPred, u32, MbType)] = &[
            (L0, L0, 4, MbType::BL0L016x8),
            (L1, L1, 6, MbType::BL1L116x8),
            (L0, L1, 8, MbType::BL0L116x8),
            (L1, L0, 10, MbType::BL1L016x8),
            (L0, Bi, 12, MbType::BL0Bi16x8),
            (L1, Bi, 14, MbType::BL1Bi16x8),
            (Bi, L0, 16, MbType::BBiL016x8),
            (Bi, L1, 18, MbType::BBiL116x8),
            (Bi, Bi, 20, MbType::BBiBi16x8),
        ];
        for (top, bot, raw, expected) in cases.iter().cloned() {
            assert_eq!(b_16x8_mb_type_raw(top, bot), raw);
            assert_eq!(MbType::from_b_slice(raw).unwrap(), expected);
        }
    }

    /// §7.4.5 Table 7-14 — round-22 B 8x16 mb_type raw values 5..=21
    /// odd all decode back to the expected (mode_left, mode_right)
    /// `MbType` variant.
    #[test]
    fn b_8x16_mb_type_raw_matches_table_7_14_decode() {
        use crate::macroblock_layer::MbType;
        use BPartPred::*;
        let cases: &[(BPartPred, BPartPred, u32, MbType)] = &[
            (L0, L0, 5, MbType::BL0L08x16),
            (L1, L1, 7, MbType::BL1L18x16),
            (L0, L1, 9, MbType::BL0L18x16),
            (L1, L0, 11, MbType::BL1L08x16),
            (L0, Bi, 13, MbType::BL0Bi8x16),
            (L1, Bi, 15, MbType::BL1Bi8x16),
            (Bi, L0, 17, MbType::BBiL08x16),
            (Bi, L1, 19, MbType::BBiL18x16),
            (Bi, Bi, 21, MbType::BBiBi8x16),
        ];
        for (left, right, raw, expected) in cases.iter().cloned() {
            assert_eq!(b_8x16_mb_type_raw(left, right), raw);
            assert_eq!(MbType::from_b_slice(raw).unwrap(), expected);
        }
    }

    /// Round-22 B 16x8 zero-residual writer emits exactly the ue(v) for
    /// mb_type, the four mvds (Bi/Bi has 4 mvds = 8 se(v) ints), and the
    /// cbp codeNum. With both partitions Bi and all mvds=0 the bit count
    /// is: mb_type ue(v) for raw 20 (codeNum 20 → 9 bits) + 8 se(v) of
    /// "0" (1 bit each) + cbp codeNum=0 (1 bit) = 18 bits.
    #[test]
    fn b_16x8_writer_round_trip_smoke() {
        let mut bw = BitWriter::new();
        let cfg = B16x8McbConfig {
            top: BPartPred::Bi,
            bottom: BPartPred::Bi,
            mvd_l0: [(0, 0); 2],
            mvd_l1: [(0, 0); 2],
            cbp_luma: 0,
            cbp_chroma: 0,
            mb_qp_delta: 0,
            luma_4x4_levels: [[0; 16]; 16],
            luma_4x4_nc: [0; 16],
            chroma_dc_cb: [0; 4],
            chroma_dc_cr: [0; 4],
            chroma_ac_cb: [[0; 16]; 4],
            chroma_ac_cr: [[0; 16]; 4],
        };
        write_b_16x8_mb(&mut bw, &cfg, 0, 0).expect("write");
        // ue(20) is 9 bits; 8 se(0)=1 bit each; ue(0)=1 bit. Total 18.
        assert_eq!(bw.bits_emitted(), 18);
    }

    /// Round-22 B 8x16 zero-residual writer for (L0, L1) pair: mb_type
    /// raw 9 → ue(9) is 7 bits; mvd_l0[0] (.x, .y) = 2 bits; mvd_l1[1]
    /// (.x, .y) = 2 bits; cbp ue(0) = 1 bit. Total 12 bits.
    #[test]
    fn b_8x16_writer_l0_l1_zero_residual_bit_count() {
        let mut bw = BitWriter::new();
        let cfg = B8x16McbConfig {
            left: BPartPred::L0,
            right: BPartPred::L1,
            mvd_l0: [(0, 0); 2],
            mvd_l1: [(0, 0); 2],
            cbp_luma: 0,
            cbp_chroma: 0,
            mb_qp_delta: 0,
            luma_4x4_levels: [[0; 16]; 16],
            luma_4x4_nc: [0; 16],
            chroma_dc_cb: [0; 4],
            chroma_dc_cr: [0; 4],
            chroma_ac_cb: [[0; 16]; 4],
            chroma_ac_cr: [[0; 16]; 4],
        };
        write_b_8x16_mb(&mut bw, &cfg, 0, 0).expect("write");
        // ue(9) is 7 bits. Left uses L0 only → mvd_l0 (.x, .y) = 2 bits.
        // Right uses L1 only → mvd_l1 (.x, .y) = 2 bits. cbp ue(0) = 1 bit.
        // Total 12 bits.
        assert_eq!(bw.bits_emitted(), 12);
    }

    /// Round-25 — `BSubMbCell` raw values match Table 7-18 codenums for
    /// the four single-8x8-partition sub_mb_types {B_Direct_8x8,
    /// B_L0_8x8, B_L1_8x8, B_Bi_8x8} → raw {0, 1, 2, 3}. The decoder's
    /// `SubMbType::from_b` maps each raw back to the matching variant.
    #[test]
    fn b8x8_mixed_sub_mb_type_raws_match_table_7_18() {
        use crate::macroblock_layer::SubMbType;
        assert_eq!(BSubMbCell::Direct.sub_mb_type_raw(), 0);
        assert_eq!(BSubMbCell::L0.sub_mb_type_raw(), 1);
        assert_eq!(BSubMbCell::L1.sub_mb_type_raw(), 2);
        assert_eq!(BSubMbCell::Bi.sub_mb_type_raw(), 3);
        assert_eq!(SubMbType::from_b(0).unwrap(), SubMbType::BDirect8x8);
        assert_eq!(SubMbType::from_b(1).unwrap(), SubMbType::BL08x8);
        assert_eq!(SubMbType::from_b(2).unwrap(), SubMbType::BL18x8);
        assert_eq!(SubMbType::from_b(3).unwrap(), SubMbType::BBi8x8);

        // List-usage gate (writer side): Direct emits no MVs / refs;
        // L0/L1/Bi follow the standard pattern.
        assert!(!BSubMbCell::Direct.writes_l0() && !BSubMbCell::Direct.writes_l1());
        assert!(BSubMbCell::L0.writes_l0() && !BSubMbCell::L0.writes_l1());
        assert!(!BSubMbCell::L1.writes_l0() && BSubMbCell::L1.writes_l1());
        assert!(BSubMbCell::Bi.writes_l0() && BSubMbCell::Bi.writes_l1());
    }

    /// Round-25 — `B8x8MixedMcbConfig` writer with `cbp_luma == 0,
    /// cbp_chroma == 0` and a (Direct, L0, L1, Bi) cell layout (one of
    /// each non-trivial pick).
    ///
    /// Bit-count tally (Exp-Golomb ue(v) widths per §9.1.1: codeNum=k
    /// uses `2*floor(log2(k+1)) + 1` bits):
    ///   * mb_type=22 → 9 bits  (k+1=23, floor(log2)=4)
    ///   * sub_mb_type[0]=0 (Direct) → 1 bit
    ///   * sub_mb_type[1]=1 (L0)     → 3 bits  (ue(1))
    ///   * sub_mb_type[2]=2 (L1)     → 3 bits  (ue(2))
    ///   * sub_mb_type[3]=3 (Bi)     → 5 bits  (ue(3): k+1=4, floor(log2)=2)
    ///   * num_ref ∈ {0} → no ref_idx te(v).
    ///   * mvd_l0: cells writing L0 = {1 (L0), 3 (Bi)} → 2 cells × 2 se(v)
    ///     each at zero → 4 bits.
    ///   * mvd_l1: cells writing L1 = {2 (L1), 3 (Bi)} → 2 cells × 2 se(v)
    ///     each at zero → 4 bits.
    ///   * cbp codeNum 0 ue(v) → 1 bit.
    ///   * No mb_qp_delta (gated on cbp > 0).
    ///
    /// Total 9 + 1+3+3+5 + 4 + 4 + 1 = 30 bits.
    #[test]
    fn b_8x8_mixed_zero_residual_emits_expected_bit_count() {
        let mut bw = BitWriter::new();
        let cfg = B8x8MixedMcbConfig {
            cells: [
                BSubMbCell::Direct,
                BSubMbCell::L0,
                BSubMbCell::L1,
                BSubMbCell::Bi,
            ],
            mvd_l0: [(0, 0); 4],
            mvd_l1: [(0, 0); 4],
            cbp_luma: 0,
            cbp_chroma: 0,
            mb_qp_delta: 0,
            luma_4x4_levels: [[0; 16]; 16],
            luma_4x4_nc: [0; 16],
            chroma_dc_cb: [0; 4],
            chroma_dc_cr: [0; 4],
            chroma_ac_cb: [[0; 16]; 4],
            chroma_ac_cr: [[0; 16]; 4],
        };
        write_b_8x8_mixed_mb(&mut bw, &cfg, 0, 0).expect("write");
        assert_eq!(bw.bits_emitted(), 30);
    }

    /// Round-25 — `B8x8MixedMcbConfig` writer with all four cells set to
    /// `B_Direct_8x8` emits exactly the same bit pattern as the round-23
    /// `write_b_8x8_all_direct_mb` writer (with cbp == 0): 9 (mb_type 22)
    /// + 4×1 (sub_mb_type 0) + 1 (cbp ue(0)) = 14 bits.
    #[test]
    fn b_8x8_mixed_all_direct_matches_dedicated_all_direct_writer_bit_count() {
        let mut bw = BitWriter::new();
        let cfg = B8x8MixedMcbConfig {
            cells: [BSubMbCell::Direct; 4],
            mvd_l0: [(0, 0); 4],
            mvd_l1: [(0, 0); 4],
            cbp_luma: 0,
            cbp_chroma: 0,
            mb_qp_delta: 0,
            luma_4x4_levels: [[0; 16]; 16],
            luma_4x4_nc: [0; 16],
            chroma_dc_cb: [0; 4],
            chroma_dc_cr: [0; 4],
            chroma_ac_cb: [[0; 16]; 4],
            chroma_ac_cr: [[0; 16]; 4],
        };
        write_b_8x8_mixed_mb(&mut bw, &cfg, 0, 0).expect("write");
        assert_eq!(bw.bits_emitted(), 14);

        // Sanity: matches the dedicated all-Direct writer byte-for-byte.
        let mut bw_dedicated = BitWriter::new();
        let cfg_dedicated = B8x8AllDirectMcbConfig {
            cbp_luma: 0,
            cbp_chroma: 0,
            mb_qp_delta: 0,
            luma_4x4_levels: [[0; 16]; 16],
            luma_4x4_nc: [0; 16],
            chroma_dc_cb: [0; 4],
            chroma_dc_cr: [0; 4],
            chroma_ac_cb: [[0; 16]; 4],
            chroma_ac_cr: [[0; 16]; 4],
        };
        write_b_8x8_all_direct_mb(&mut bw_dedicated, &cfg_dedicated).expect("write");
        assert_eq!(bw.bits_emitted(), bw_dedicated.bits_emitted());
    }
}
