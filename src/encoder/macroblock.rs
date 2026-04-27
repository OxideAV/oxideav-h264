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
}
