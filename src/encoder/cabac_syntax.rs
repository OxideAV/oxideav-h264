//! §9.3 — CABAC per-syntax-element **encoder** binarisations and ctxIdx
//! derivation. Mirror of [`crate::cabac_ctx`] (the decoder side).
//!
//! The decoder defines, for each syntax element:
//! * a binarisation (TU / U / FL / UEGk / mb_type lookup tables), and
//! * a ctxIdx derivation procedure (§9.3.3.1.1.* — neighbour-aware).
//!
//! This module provides the matching ENCODE-side helpers: given the
//! element's value (and the same neighbour info the decoder consults),
//! produce the bin string and route each bin through
//! [`CabacEncoder`] using the appropriate ctxIdx.
//!
//! Round-30 scope: I + P slice elements only.
//! - `mb_skip_flag`           §9.3.3.1.1.1
//! - `mb_type`                §9.3.3.1.1.3 (I-slice + P-slice)
//! - `sub_mb_type`            §9.3.3.1.1.3 + Table 9-37 (P only)
//! - `mvd_lX`                 §9.3.3.1.1.7 / §9.3.3.1.2 + Table 9-39
//! - `ref_idx_lX`             §9.3.3.1.1.6
//! - `coded_block_pattern`    §9.3.3.1.1.4
//! - `coded_block_flag`       §9.3.3.1.1.9
//! - `significant_coeff_flag` §9.3.3.1.3
//! - `last_significant_coeff_flag`
//! - `coeff_abs_level_minus1` §9.3.3.1.3
//! - `coeff_sign_flag`        bypass
//! - `mb_qp_delta`            §9.3.3.1.1.5
//! - `intra_chroma_pred_mode` §9.3.3.1.1.8
//! - `prev_intra4x4_pred_mode_flag` / `rem_intra4x4_pred_mode`
//! - `end_of_slice_flag`      §9.3.3.2.4
//!
//! Out of scope: B-slice mb_type, transform_size_8x8_flag, 4:2:2/4:4:4
//! residual paths (the 4:2:0 paths cover the round-30 acceptance fixtures).

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use crate::cabac_ctx::{BlockType, CabacContexts, MvdComponent, NeighbourCtx, SliceKind};
use crate::encoder::cabac_engine::CabacEncoder;

// ---------------------------------------------------------------------------
// §9.3.3.1.1.1 — mb_skip_flag.
// ---------------------------------------------------------------------------

fn mb_skip_flag_ctx_offset(slice_kind: SliceKind) -> Option<u32> {
    match slice_kind {
        SliceKind::P | SliceKind::SP => Some(11),
        SliceKind::B => Some(24),
        _ => None,
    }
}

/// §9.3.3.1.1.1 — encode `mb_skip_flag`.
pub fn encode_mb_skip_flag(
    enc: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    slice_kind: SliceKind,
    neighbours: &NeighbourCtx,
    skip: bool,
) {
    let offset =
        mb_skip_flag_ctx_offset(slice_kind).expect("mb_skip_flag is not coded on I/SI slices");
    let cond_a = u32::from(neighbours.available_left && !neighbours.mb_skip_flag_left);
    let cond_b = u32::from(neighbours.available_above && !neighbours.mb_skip_flag_above);
    let ctx_idx = (offset + cond_a + cond_b) as usize;
    enc.encode_decision(ctxs.at_mut(ctx_idx), if skip { 1 } else { 0 });
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.3 — mb_type (I + P slice).
//
// Mirrors the decoder's `decode_mb_type_i` / `decode_mb_type_p` step
// for step. Each helper takes the spec value and emits the bin string
// that, when decoded, recovers the same value.
// ---------------------------------------------------------------------------

fn mb_type_cond_term_flag(offset: u32, neighbours: &NeighbourCtx, is_left: bool) -> u32 {
    let available = if is_left {
        neighbours.available_left
    } else {
        neighbours.available_above
    };
    if !available {
        return 0;
    }
    let filtered = match offset {
        0 => {
            if is_left {
                neighbours.left_is_si
            } else {
                neighbours.above_is_si
            }
        }
        3 => {
            if is_left {
                neighbours.left_is_i_nxn
            } else {
                neighbours.above_is_i_nxn
            }
        }
        27 => {
            if is_left {
                neighbours.left_is_b_skip_or_direct
            } else {
                neighbours.above_is_b_skip_or_direct
            }
        }
        _ => false,
    };
    u32::from(!filtered)
}

fn mb_type_ctx_inc_first3(offset: u32, bin_idx: u32, neighbours: &NeighbourCtx) -> u32 {
    if bin_idx == 0 {
        mb_type_cond_term_flag(offset, neighbours, true)
            + mb_type_cond_term_flag(offset, neighbours, false)
    } else {
        bin_idx
    }
}

/// §9.3.2.5 / Table 9-36 — encode an Intra mb_type "suffix".
///
/// `suffix` is the Table 9-36 row index (0..=25). `offset` selects the
/// ctxIdxOffset family (3 for I-slice, 17 for P intra-suffix, 32 for B).
fn encode_mb_type_intra_suffix(
    enc: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    offset: u32,
    suffix: u32,
    bin0_already_emitted: bool,
) {
    debug_assert!(suffix <= 25);
    if suffix == 0 {
        // Row 0 — I_NxN. Bin string is just "0" at bin 0. For the I-slice
        // path the caller has already emitted bin 0 (the value 0 came
        // out of `decode_mb_type_i`'s b0=0 branch). For P/B intra suffix
        // the caller has NOT emitted bin 0 yet; we must emit "0" here.
        if !bin0_already_emitted {
            enc.encode_decision(ctxs.at_mut(offset as usize), 0);
        }
        return;
    }
    // We reach this branch with the caller ensuring bin 0 == 1 was
    // emitted (or is about to be). For the I-slice path bin 0 == 1 is
    // the entry to the suffix tree; for P/B intra-suffix we still need
    // to emit bin 0 == 1 here.
    if !bin0_already_emitted {
        enc.encode_decision(ctxs.at_mut(offset as usize), 1);
    }

    if suffix == 25 {
        // Row 25 — I_PCM, bin string "1 1" on the I-slice tree. Bin 1
        // is the §9.3.4.5 terminate bin (ctxIdx 276 special).
        enc.encode_terminate(1);
        return;
    }
    // Rows 1..=24: bin 1 is terminate(0).
    enc.encode_terminate(0);

    // Per the decoder: binIdx 2..=6 ctxIdxInc family is offset-dependent.
    let (inc_b2, inc_b3, inc_b4_b3ne0, inc_b4_b3eq0, inc_b5_b3ne0, inc_b5_b3eq0, inc_b6) =
        if offset == 3 {
            (3u32, 4u32, 5u32, 6u32, 6u32, 7u32, 7u32)
        } else {
            (1u32, 2u32, 2u32, 3u32, 3u32, 3u32, 3u32)
        };

    // Decode tree:
    //   value = 12*b2 + 1 + ((b4<<1)|b5)              (b3 == 0, 6 bins total)
    //   value = 12*b2 + 5 + 4*b4 + ((b5<<1)|b6)       (b3 == 1, 7 bins total)
    let b2: u8 = if (1..=12).contains(&suffix) { 0 } else { 1 };
    let base = 12u32 * (b2 as u32);
    // Determine b3 from suffix: 6-bin rows use base+1..=base+4 (suffix-base in {1,2,3,4}).
    let off = suffix - base;
    let (b3, b4, b5, b6_opt) = if off <= 4 {
        // 6-bin row: b3=0; (b4<<1)|b5 = off-1.
        let v = off - 1;
        let b4 = ((v >> 1) & 1) as u8;
        let b5 = (v & 1) as u8;
        (0u8, b4, b5, None)
    } else {
        // 7-bin row: b3=1; off in 5..=12. b4 = (off>=9)?1:0;
        // (b5<<1)|b6 = (off - 5 - 4*b4).
        let b4 = if off >= 9 { 1u8 } else { 0u8 };
        let v = off - 5 - 4 * (b4 as u32);
        let b5 = ((v >> 1) & 1) as u8;
        let b6 = (v & 1) as u8;
        (1u8, b4, b5, Some(b6))
    };

    // Emit b2.
    enc.encode_decision(ctxs.at_mut((offset + inc_b2) as usize), b2);
    // b3.
    enc.encode_decision(ctxs.at_mut((offset + inc_b3) as usize), b3);
    // b4 ctxIdxInc depends on b3.
    let inc_b4 = if b3 != 0 { inc_b4_b3ne0 } else { inc_b4_b3eq0 };
    enc.encode_decision(ctxs.at_mut((offset + inc_b4) as usize), b4);
    // b5 ctxIdxInc depends on b3.
    let inc_b5 = if b3 != 0 { inc_b5_b3ne0 } else { inc_b5_b3eq0 };
    enc.encode_decision(ctxs.at_mut((offset + inc_b5) as usize), b5);
    // b6 only present in 7-bin rows.
    if let Some(b6) = b6_opt {
        enc.encode_decision(ctxs.at_mut((offset + inc_b6) as usize), b6);
    }
}

/// §9.3.2.5 / Table 9-36 — encode `mb_type` for I-slices.
/// `value` is the Table 9-36 row index (0..=25).
pub fn encode_mb_type_i(
    enc: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    neighbours: &NeighbourCtx,
    value: u32,
) {
    const OFFSET: u32 = 3;
    debug_assert!(value <= 25);
    let inc0 = mb_type_ctx_inc_first3(OFFSET, 0, neighbours);
    if value == 0 {
        // I_NxN — single bin "0".
        enc.encode_decision(ctxs.at_mut((OFFSET + inc0) as usize), 0);
        return;
    }
    // Bin 0 = 1 → enter Table 9-36 suffix tree.
    enc.encode_decision(ctxs.at_mut((OFFSET + inc0) as usize), 1);
    // The decoder calls `decode_mb_type_i_suffix(..., already_read_bin0=Some(1))`
    // so the helper does NOT re-read bin 0. We mirror that.
    encode_mb_type_intra_suffix(enc, ctxs, OFFSET, value, true);
}

/// §9.3.2.5 / Table 9-37 — encode `mb_type` for P/SP-slices.
///
/// `value` semantics:
/// * 0..=3 — Table 9-37 raw mb_type for P-slice inter (`P_L0_16x16`,
///   `P_L0_L0_16x8`, `P_L0_L0_8x16`, `P_8x8`). Bin string starts with "0".
/// * 5..=30 — intra mb_type (P-slice raw; suffix = value - 5 maps to
///   Table 9-36 row 0..=25). Bin string starts with "1" prefix at
///   ctxIdxOffset=14, then suffix at ctxIdxOffset=17.
///
/// Note: raw value 4 is not legal in P-slices.
pub fn encode_mb_type_p(
    enc: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    _neighbours: &NeighbourCtx,
    value: u32,
) {
    const OFFSET: u32 = 14;
    debug_assert!(
        value <= 4 || (5..=30).contains(&value),
        "invalid P mb_type {value}"
    );
    if value >= 5 {
        // Intra: bin0=1, then Table 9-36 suffix at offset=17.
        enc.encode_decision(ctxs.at_mut(OFFSET as usize), 1);
        encode_mb_type_intra_suffix(enc, ctxs, 17, value - 5, false);
        return;
    }
    // Inter: bin0=0.
    enc.encode_decision(ctxs.at_mut(OFFSET as usize), 0);
    // Per the decoder's table:
    //   0 (P_L0_16x16)    → 0 0 0
    //   1 (P_L0_L0_16x8)  → 0 1 1
    //   2 (P_L0_L0_8x16)  → 0 1 0
    //   3 (P_8x8)         → 0 0 1
    let (b1, b2) = match value {
        0 => (0u8, 0u8),
        1 => (1, 1),
        2 => (1, 0),
        3 => (0, 1),
        _ => unreachable!(),
    };
    enc.encode_decision(ctxs.at_mut((OFFSET + 1) as usize), b1);
    let inc_b2 = if b1 != 1 { 2 } else { 3 };
    enc.encode_decision(ctxs.at_mut((OFFSET + inc_b2) as usize), b2);
}

/// §9.3.2.5 / Table 9-37 — encode `mb_type` for B-slices.
///
/// `value` is the Table 7-14 raw mb_type:
/// * 0  — B_Direct_16x16 (bin string "0")
/// * 1  — B_L0_16x16     (bin string "1 0 0")
/// * 2  — B_L1_16x16     (bin string "1 0 1")
/// * 3  — B_Bi_16x16     (bin string "1 1 0 0 0 0")
/// * 4..=11 — 16x8 / 8x16 partition modes (bin strings 1 1 0 ... 1 1 1 1 1 0)
/// * 12..=21 — Bi mixed 16x8 / 8x16 (7-bin)
/// * 22 — B_8x8          (bin string "1 1 1 1 1 1")
/// * 23..=48 — intra (B intra prefix "1 1 1 1 0 1" + Table 9-36 suffix at offset=32)
///
/// Mirror of [`crate::cabac_ctx::decode_mb_type_b`].
pub fn encode_mb_type_b(
    enc: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    neighbours: &NeighbourCtx,
    value: u32,
) {
    const OFFSET: u32 = 27;
    debug_assert!(value <= 48, "invalid B mb_type {value}");
    // §9.3.3.1.1.3 — bin 0 ctxIdxInc using offset=27 special condTerm
    // ("mb_type for mbAddrN is B_Skip or B_Direct_16x16").
    let inc0 = mb_type_ctx_inc_first3(OFFSET, 0, neighbours);
    if value == 0 {
        // B_Direct_16x16 — single bin "0".
        enc.encode_decision(ctxs.at_mut((OFFSET + inc0) as usize), 0);
        return;
    }
    enc.encode_decision(ctxs.at_mut((OFFSET + inc0) as usize), 1);
    // bin 1 at ctxIdxInc=3.
    // bin 2 at ctxIdxInc = (b1 != 0) ? 4 : 5.
    if value == 1 {
        // B_L0_16x16 — "1 0 0".
        enc.encode_decision(ctxs.at_mut((OFFSET + 3) as usize), 0);
        // b1==0 → inc_b2 = 5.
        enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), 0);
        return;
    }
    if value == 2 {
        // B_L1_16x16 — "1 0 1".
        enc.encode_decision(ctxs.at_mut((OFFSET + 3) as usize), 0);
        enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), 1);
        return;
    }
    // value >= 3 — bin 1 = 1.
    enc.encode_decision(ctxs.at_mut((OFFSET + 3) as usize), 1);
    // b1==1 → inc_b2 = 4. b2 then determines further branching.
    if (3..=10).contains(&value) {
        // 6-bin rows "1 1 0 b3 b4 b5":
        //   3  B_Bi_16x16     1 1 0 0 0 0
        //   4  B_L0_L0_16x8   1 1 0 0 0 1
        //   5  B_L0_L0_8x16   1 1 0 0 1 0
        //   6  B_L1_L1_16x8   1 1 0 0 1 1
        //   7  B_L1_L1_8x16   1 1 0 1 0 0
        //   8  B_L0_L1_16x8   1 1 0 1 0 1
        //   9  B_L0_L1_8x16   1 1 0 1 1 0
        //   10 B_L1_L0_16x8   1 1 0 1 1 1
        enc.encode_decision(ctxs.at_mut((OFFSET + 4) as usize), 0);
        let v = value - 3;
        let b3 = ((v >> 2) & 1) as u8;
        let b4 = ((v >> 1) & 1) as u8;
        let b5 = (v & 1) as u8;
        enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), b3);
        enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), b4);
        enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), b5);
        return;
    }
    // value >= 11 — b2 = 1.
    enc.encode_decision(ctxs.at_mut((OFFSET + 4) as usize), 1);
    if value == 22 {
        // B_8x8 — "1 1 1 1 1 1".
        enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), 1);
        enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), 1);
        enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), 1);
        return;
    }
    if value == 11 {
        // B_L1_L0_8x16 — "1 1 1 1 1 0".
        enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), 1);
        enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), 1);
        enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), 0);
        return;
    }
    if (23..=48).contains(&value) {
        // Intra suffix prefix "1 1 1 1 0 1": b3=1,b4=0,b5=1.
        enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), 1);
        enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), 0);
        enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), 1);
        // Table 9-36 suffix at offset=32. Bin 0 of the suffix has not
        // been emitted; pass `bin0_already_emitted=false`.
        encode_mb_type_intra_suffix(enc, ctxs, 32, value - 23, false);
        return;
    }
    // 7-bin rows 12..=21 — "1 1 1 b3 b4 b5 b6":
    //   12 B_L0_Bi_16x8   1 1 1 0 0 0 0
    //   13 B_L0_Bi_8x16   1 1 1 0 0 0 1
    //   14 B_L1_Bi_16x8   1 1 1 0 0 1 0
    //   15 B_L1_Bi_8x16   1 1 1 0 0 1 1
    //   16 B_Bi_L0_16x8   1 1 1 0 1 0 0
    //   17 B_Bi_L0_8x16   1 1 1 0 1 0 1
    //   18 B_Bi_L1_16x8   1 1 1 0 1 1 0
    //   19 B_Bi_L1_8x16   1 1 1 0 1 1 1
    //   20 B_Bi_Bi_16x8   1 1 1 1 0 0 0
    //   21 B_Bi_Bi_8x16   1 1 1 1 0 0 1
    let v = value - 12;
    let b3 = ((v >> 3) & 1) as u8;
    let b4 = ((v >> 2) & 1) as u8;
    let b5 = ((v >> 1) & 1) as u8;
    let b6 = (v & 1) as u8;
    enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), b3);
    enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), b4);
    enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), b5);
    enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), b6);
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.5 — mb_qp_delta (signed Exp-Golomb-ish via Table 9-3 + unary).
// ---------------------------------------------------------------------------

/// §9.3.3.1.1.5 — encode `mb_qp_delta` (signed value mapped via Table 9-3
/// and emitted as a truncated-unary-style bin string).
pub fn encode_mb_qp_delta(
    enc: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    prev_mb_qp_delta_nonzero: bool,
    value: i32,
) {
    const OFFSET: u32 = 60;
    let inc0 = u32::from(prev_mb_qp_delta_nonzero);
    if value == 0 {
        enc.encode_decision(ctxs.at_mut((OFFSET + inc0) as usize), 0);
        return;
    }
    enc.encode_decision(ctxs.at_mut((OFFSET + inc0) as usize), 1);
    // Table 9-3 mapping inverse: positive k → mapped = 2*k - 1; non-positive → -2*k.
    // mapped value 1 → +1 (bin string "1 0").
    let mapped: u32 = if value > 0 {
        (2 * value - 1) as u32
    } else {
        (-2 * value) as u32
    };
    debug_assert!(mapped >= 1);
    if mapped == 1 {
        enc.encode_decision(ctxs.at_mut((OFFSET + 2) as usize), 0);
        return;
    }
    // mapped >= 2 — bin1=1, then (mapped-2) "1"s + a "0" at ctxIdx OFFSET+3.
    enc.encode_decision(ctxs.at_mut((OFFSET + 2) as usize), 1);
    for _ in 0..(mapped - 2) {
        enc.encode_decision(ctxs.at_mut((OFFSET + 3) as usize), 1);
    }
    enc.encode_decision(ctxs.at_mut((OFFSET + 3) as usize), 0);
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.6 — ref_idx_lX (unary).
// ---------------------------------------------------------------------------

/// §9.3.3.1.1.6 — encode `ref_idx_lX` (unary).
pub fn encode_ref_idx_lx(
    enc: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    neighbour_left_gt_0: bool,
    neighbour_above_gt_0: bool,
    value: u32,
) {
    const OFFSET: u32 = 54;
    let inc0 = u32::from(neighbour_left_gt_0) + 2 * u32::from(neighbour_above_gt_0);
    if value == 0 {
        enc.encode_decision(ctxs.at_mut((OFFSET + inc0) as usize), 0);
        return;
    }
    enc.encode_decision(ctxs.at_mut((OFFSET + inc0) as usize), 1);
    if value == 1 {
        enc.encode_decision(ctxs.at_mut((OFFSET + 4) as usize), 0);
        return;
    }
    enc.encode_decision(ctxs.at_mut((OFFSET + 4) as usize), 1);
    // Subsequent unary bins use ctxIdxInc = 5.
    for _ in 0..(value - 2) {
        enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), 1);
    }
    enc.encode_decision(ctxs.at_mut((OFFSET + 5) as usize), 0);
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.7 — mvd_lX (UEG3, signed).
// ---------------------------------------------------------------------------

/// §9.3.3.1.1.7 — encode `mvd_lX[][][compIdx]` (UEG3 binarisation,
/// uCoff=9, signedValFlag=1).
pub fn encode_mvd_lx(
    enc: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    component: MvdComponent,
    neighbour_abs_mvd_sum: u32,
    value: i32,
) {
    let offset = match component {
        MvdComponent::X => 40,
        MvdComponent::Y => 47,
    };
    let inc0 = if neighbour_abs_mvd_sum > 32 {
        2
    } else if neighbour_abs_mvd_sum > 2 {
        1
    } else {
        0
    };
    let abs_val = value.unsigned_abs();

    // UEG3 prefix: TU with cMax = 9.
    let prefix_val = abs_val.min(9);
    let bins: [u32; 9] = [inc0, 3, 4, 5, 6, 6, 6, 6, 6];
    for (i, inc) in bins.iter().enumerate() {
        let ctx_idx = (offset + inc) as usize;
        if (i as u32) < prefix_val {
            // Continue with '1' bins until we hit cMax.
            enc.encode_decision(ctxs.at_mut(ctx_idx), 1);
        } else {
            // Emit the terminating '0' bin.
            enc.encode_decision(ctxs.at_mut(ctx_idx), 0);
            break;
        }
    }
    if abs_val >= 9 {
        // Suffix: 0-th order Exp-Golomb (k0=3) bypass.
        let mut suf = abs_val - 9;
        let mut k: u32 = 3;
        // Emit the unary-of-1s phase: while suf >= (1 << k), emit '1', subtract, increment k.
        while suf >= (1u32 << k) {
            enc.encode_bypass(1);
            suf -= 1u32 << k;
            k += 1;
        }
        // Stop bit '0'.
        enc.encode_bypass(0);
        // Then write `k` bits of suf MSB-first.
        for i in (0..k).rev() {
            let b = ((suf >> i) & 1) as u8;
            enc.encode_bypass(b);
        }
    }
    if abs_val != 0 {
        let sign = if value < 0 { 1 } else { 0 };
        enc.encode_bypass(sign);
    }
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.8 — intra_chroma_pred_mode (TU, cMax=3).
// ---------------------------------------------------------------------------

/// §9.3.3.1.1.8 — encode `intra_chroma_pred_mode`.
pub fn encode_intra_chroma_pred_mode(
    enc: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    neighbours: &NeighbourCtx,
    value: u32,
) {
    const OFFSET: u32 = 64;
    debug_assert!(value <= 3);
    let cond_a = u32::from(
        neighbours.available_left
            && !neighbours.left_inter
            && !neighbours.left_is_i_pcm
            && neighbours.left_intra_chroma_pred_mode_nonzero,
    );
    let cond_b = u32::from(
        neighbours.available_above
            && !neighbours.above_inter
            && !neighbours.above_is_i_pcm
            && neighbours.above_intra_chroma_pred_mode_nonzero,
    );
    let inc0 = cond_a + cond_b;
    if value == 0 {
        enc.encode_decision(ctxs.at_mut((OFFSET + inc0) as usize), 0);
        return;
    }
    enc.encode_decision(ctxs.at_mut((OFFSET + inc0) as usize), 1);
    if value == 1 {
        enc.encode_decision(ctxs.at_mut((OFFSET + 3) as usize), 0);
        return;
    }
    enc.encode_decision(ctxs.at_mut((OFFSET + 3) as usize), 1);
    let bin2 = if value == 3 { 1u8 } else { 0 };
    enc.encode_decision(ctxs.at_mut((OFFSET + 3) as usize), bin2);
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.9 — coded_block_flag.
// ---------------------------------------------------------------------------

// Internal helpers to grab the same offsets the decoder side uses.
//
// We can't import the private methods on `BlockType`, so re-implement the
// small set we need locally (mirrors `cabac_ctx::BlockType::cbf_*`).
fn cbf_ctx_offset(bt: BlockType) -> u32 {
    let cat = bt as u32;
    match cat {
        0..=4 => 85,
        6..=8 => 460,
        10..=12 => 472,
        5 | 9 | 13 => 1012,
        _ => 85,
    }
}

fn cbf_block_cat_offset(bt: BlockType) -> u32 {
    match bt {
        BlockType::Luma16x16Dc => 0,
        BlockType::Luma16x16Ac => 4,
        BlockType::Luma4x4 => 8,
        BlockType::ChromaDc => 12,
        BlockType::ChromaAc => 16,
        BlockType::Luma8x8 => 0,
        BlockType::CbIntra16x16Dc => 0,
        BlockType::CbIntra16x16Ac => 4,
        BlockType::CbLuma4x4 => 8,
        BlockType::Cb8x8 => 4,
        BlockType::CrIntra16x16Dc => 0,
        BlockType::CrIntra16x16Ac => 4,
        BlockType::CrLuma4x4 => 8,
        BlockType::Cr8x8 => 8,
    }
}

fn sig_block_cat_offset(bt: BlockType) -> u32 {
    match bt {
        BlockType::Luma16x16Dc => 0,
        BlockType::Luma16x16Ac => 15,
        BlockType::Luma4x4 => 29,
        BlockType::ChromaDc => 44,
        BlockType::ChromaAc => 47,
        BlockType::Luma8x8 => 0,
        BlockType::CbIntra16x16Dc => 0,
        BlockType::CbIntra16x16Ac => 15,
        BlockType::CbLuma4x4 => 29,
        BlockType::Cb8x8 => 0,
        BlockType::CrIntra16x16Dc => 0,
        BlockType::CrIntra16x16Ac => 15,
        BlockType::CrLuma4x4 => 29,
        BlockType::Cr8x8 => 0,
    }
}

fn lvl_block_cat_offset(bt: BlockType) -> u32 {
    match bt {
        BlockType::Luma16x16Dc => 0,
        BlockType::Luma16x16Ac => 10,
        BlockType::Luma4x4 => 20,
        BlockType::ChromaDc => 30,
        BlockType::ChromaAc => 39,
        _ => 0,
    }
}

fn sig_coef_ctx_offset(bt: BlockType) -> u32 {
    let cat = bt as u32;
    match cat {
        0..=4 => 105,
        5 => 402,
        6..=8 => 484,
        9 => 660,
        10..=12 => 528,
        13 => 718,
        _ => 105,
    }
}

fn last_coef_ctx_offset(bt: BlockType) -> u32 {
    let cat = bt as u32;
    match cat {
        0..=4 => 166,
        5 => 417,
        6..=8 => 572,
        9 => 690,
        10..=12 => 616,
        13 => 748,
        _ => 166,
    }
}

fn abs_level_ctx_offset(bt: BlockType) -> u32 {
    let cat = bt as u32;
    match cat {
        0..=4 => 227,
        5 => 426,
        6..=8 => 952,
        9 => 708,
        10..=12 => 982,
        13 => 766,
        _ => 227,
    }
}

/// §9.3.3.1.1.9 — encode `coded_block_flag` for a transform block.
pub fn encode_coded_block_flag(
    enc: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    block_type: BlockType,
    neighbour_cbf_left: Option<bool>,
    neighbour_cbf_above: Option<bool>,
    flag: bool,
) {
    let base = cbf_ctx_offset(block_type);
    let cat_offset = cbf_block_cat_offset(block_type);
    let cond_a = u32::from(neighbour_cbf_left.unwrap_or(false));
    let cond_b = u32::from(neighbour_cbf_above.unwrap_or(false));
    let inc = cond_a + 2 * cond_b;
    let ctx_idx = (base + cat_offset + inc) as usize;
    enc.encode_decision(ctxs.at_mut(ctx_idx), if flag { 1 } else { 0 });
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.4 — coded_block_pattern.
// ---------------------------------------------------------------------------

/// §9.3.3.1.1.4 — encode `coded_block_pattern`. The 4-bit luma prefix is
/// emitted bit-by-bit with the §9.3.3.1.1.4 ctxIdxInc derivation; the
/// optional 2-bit chroma TU suffix follows.
pub fn encode_coded_block_pattern(
    enc: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    neighbours: &NeighbourCtx,
    chroma_array_type: u32,
    cbp_luma: u8,
    cbp_chroma: u8,
) {
    debug_assert!(cbp_luma <= 15);
    debug_assert!(cbp_chroma <= 2);
    const LUMA_OFFSET: u32 = 73;
    const CHROMA_OFFSET: u32 = 77;

    let cond_ext =
        |avail: bool, is_i_pcm: bool, is_skip: bool, cbp_luma: u8, bit_idx_n: u32| -> u32 {
            if !avail {
                return 0;
            }
            if is_i_pcm {
                return 0;
            }
            if is_skip {
                return 1;
            }
            let bit_set = ((cbp_luma >> bit_idx_n) & 1) != 0;
            if bit_set {
                0
            } else {
                1
            }
        };

    let mut prev_bins = [0u32; 4];
    for bin_idx in 0..4u32 {
        let bin = (cbp_luma >> bin_idx) & 1;
        let cond_a = match bin_idx {
            0 => cond_ext(
                neighbours.available_left,
                neighbours.left_is_i_pcm,
                neighbours.left_is_p_or_b_skip,
                neighbours.left_cbp_luma,
                1,
            ),
            1 => {
                let b_k = prev_bins[0];
                if b_k != 0 {
                    0
                } else {
                    1
                }
            }
            2 => cond_ext(
                neighbours.available_left,
                neighbours.left_is_i_pcm,
                neighbours.left_is_p_or_b_skip,
                neighbours.left_cbp_luma,
                3,
            ),
            3 => {
                let b_k = prev_bins[2];
                if b_k != 0 {
                    0
                } else {
                    1
                }
            }
            _ => 0,
        };
        let cond_b = match bin_idx {
            0 => cond_ext(
                neighbours.available_above,
                neighbours.above_is_i_pcm,
                neighbours.above_is_p_or_b_skip,
                neighbours.above_cbp_luma,
                2,
            ),
            1 => cond_ext(
                neighbours.available_above,
                neighbours.above_is_i_pcm,
                neighbours.above_is_p_or_b_skip,
                neighbours.above_cbp_luma,
                3,
            ),
            2 => {
                let b_k = prev_bins[0];
                if b_k != 0 {
                    0
                } else {
                    1
                }
            }
            3 => {
                let b_k = prev_bins[1];
                if b_k != 0 {
                    0
                } else {
                    1
                }
            }
            _ => 0,
        };
        let inc = cond_a + 2 * cond_b;
        let ctx_idx = (LUMA_OFFSET + inc) as usize;
        enc.encode_decision(ctxs.at_mut(ctx_idx), bin);
        prev_bins[bin_idx as usize] = bin as u32;
    }

    if chroma_array_type == 1 || chroma_array_type == 2 {
        let chroma_cond =
            |avail: bool, is_i_pcm: bool, is_skip: bool, cbp_chroma: u8, bin_idx: u32| -> u32 {
                if !avail {
                    return 0;
                }
                if is_i_pcm {
                    return 1;
                }
                if is_skip {
                    return 0;
                }
                let zero_branch = match bin_idx {
                    0 => cbp_chroma == 0,
                    1 => cbp_chroma != 2,
                    _ => false,
                };
                if zero_branch {
                    0
                } else {
                    1
                }
            };
        // binIdx 0.
        let cond_a0 = chroma_cond(
            neighbours.available_left,
            neighbours.left_is_i_pcm,
            neighbours.left_is_p_or_b_skip,
            neighbours.left_cbp_chroma,
            0,
        );
        let cond_b0 = chroma_cond(
            neighbours.available_above,
            neighbours.above_is_i_pcm,
            neighbours.above_is_p_or_b_skip,
            neighbours.above_cbp_chroma,
            0,
        );
        let inc0 = cond_a0 + 2 * cond_b0;
        let b0 = if cbp_chroma > 0 { 1 } else { 0 };
        enc.encode_decision(ctxs.at_mut((CHROMA_OFFSET + inc0) as usize), b0);
        if cbp_chroma > 0 {
            let cond_a1 = chroma_cond(
                neighbours.available_left,
                neighbours.left_is_i_pcm,
                neighbours.left_is_p_or_b_skip,
                neighbours.left_cbp_chroma,
                1,
            );
            let cond_b1 = chroma_cond(
                neighbours.available_above,
                neighbours.above_is_i_pcm,
                neighbours.above_is_p_or_b_skip,
                neighbours.above_cbp_chroma,
                1,
            );
            let inc1 = cond_a1 + 2 * cond_b1 + 4;
            let b1 = if cbp_chroma == 2 { 1 } else { 0 };
            enc.encode_decision(ctxs.at_mut((CHROMA_OFFSET + inc1) as usize), b1);
        }
    }
}

// ---------------------------------------------------------------------------
// §9.3.3.1.3 — significant_coeff_flag / last_significant_coeff_flag /
// coeff_abs_level_minus1.
// ---------------------------------------------------------------------------

// Mirror of `cabac_ctx::TBL_9_43` for ctxBlockCat ∈ {5, 9, 13}. Same
// 63 rows of (significant_frame, significant_field, last) triples.
static TBL_9_43_ENC: [(u8, u8, u8); 63] = [
    (0, 0, 0),
    (1, 1, 1),
    (2, 1, 1),
    (3, 2, 1),
    (4, 2, 1),
    (5, 3, 1),
    (5, 3, 1),
    (4, 4, 1),
    (4, 5, 1),
    (3, 6, 1),
    (3, 7, 1),
    (4, 7, 1),
    (4, 7, 1),
    (4, 8, 1),
    (5, 4, 1),
    (5, 5, 1),
    (4, 6, 2),
    (4, 9, 2),
    (4, 10, 2),
    (4, 10, 2),
    (3, 8, 2),
    (3, 11, 2),
    (6, 12, 2),
    (7, 11, 2),
    (7, 9, 2),
    (7, 9, 2),
    (8, 10, 2),
    (9, 10, 2),
    (10, 8, 2),
    (9, 11, 2),
    (8, 12, 2),
    (7, 11, 2),
    (7, 9, 3),
    (6, 9, 3),
    (11, 10, 3),
    (12, 10, 3),
    (13, 8, 3),
    (11, 11, 3),
    (6, 12, 3),
    (7, 11, 3),
    (8, 9, 4),
    (9, 9, 4),
    (14, 10, 4),
    (10, 10, 4),
    (9, 8, 4),
    (8, 13, 4),
    (6, 13, 4),
    (11, 9, 4),
    (12, 9, 5),
    (13, 10, 5),
    (11, 10, 5),
    (6, 8, 5),
    (9, 13, 6),
    (14, 13, 6),
    (10, 9, 6),
    (9, 9, 6),
    (11, 10, 7),
    (12, 10, 7),
    (13, 14, 7),
    (11, 14, 7),
    (14, 14, 8),
    (10, 14, 8),
    (12, 14, 8),
];

fn sig_coeff_inc_enc(block_type: BlockType, level_list_idx: u32, field: bool) -> u32 {
    let cat = block_type as u32;
    match cat {
        3 => core::cmp::min(level_list_idx, 2),
        5 | 9 | 13 => {
            let row = TBL_9_43_ENC[level_list_idx as usize];
            if field {
                row.1 as u32
            } else {
                row.0 as u32
            }
        }
        _ => level_list_idx,
    }
}

fn last_coeff_inc_enc(block_type: BlockType, level_list_idx: u32) -> u32 {
    let cat = block_type as u32;
    match cat {
        3 => core::cmp::min(level_list_idx, 2),
        5 | 9 | 13 => TBL_9_43_ENC[level_list_idx as usize].2 as u32,
        _ => level_list_idx,
    }
}

/// §9.3.3.1.1.9 + §9.3.3.1.3 — encode `significant_coeff_flag(coeff_idx)`.
pub fn encode_significant_coeff_flag(
    enc: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    block_type: BlockType,
    coeff_idx: u32,
    field: bool,
    flag: bool,
) {
    // Ignore field-specific offsets (round-30 is frame-only).
    let base = if field {
        // Mirror decoder's `sig_coef_ctx_offset(true)` if needed; keep
        // round-30 to frame.
        unimplemented!("CABAC encode does not yet support field-coded slices");
    } else {
        sig_coef_ctx_offset(block_type)
    };
    let cat_offset = sig_block_cat_offset(block_type);
    let inc = sig_coeff_inc_enc(block_type, coeff_idx, field);
    let ctx_idx = (base + cat_offset + inc) as usize;
    enc.encode_decision(ctxs.at_mut(ctx_idx), if flag { 1 } else { 0 });
}

/// §9.3.3.1.1.9 + §9.3.3.1.3 — encode `last_significant_coeff_flag`.
pub fn encode_last_significant_coeff_flag(
    enc: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    block_type: BlockType,
    coeff_idx: u32,
    field: bool,
    flag: bool,
) {
    if field {
        unimplemented!("CABAC encode does not yet support field-coded slices");
    }
    let base = last_coef_ctx_offset(block_type);
    let cat_offset = sig_block_cat_offset(block_type);
    let inc = last_coeff_inc_enc(block_type, coeff_idx);
    let ctx_idx = (base + cat_offset + inc) as usize;
    enc.encode_decision(ctxs.at_mut(ctx_idx), if flag { 1 } else { 0 });
}

/// §9.3.3.1.3 — encode `coeff_abs_level_minus1`. `value` is
/// `|level| - 1` (always >= 0). Updates the "running" decoded counters
/// the caller threads through `num_decoded_eq_1` / `num_decoded_gt_1`.
pub fn encode_coeff_abs_level_minus1(
    enc: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    block_type: BlockType,
    num_decoded_eq_1: u32,
    num_decoded_gt_1: u32,
    value: u32,
) {
    let base = abs_level_ctx_offset(block_type);
    let cat_offset = lvl_block_cat_offset(block_type);
    let u_coff: u32 = 14;
    let inc0 = if num_decoded_gt_1 != 0 {
        0
    } else {
        core::cmp::min(4, 1 + num_decoded_eq_1)
    };
    let ctx0 = (base + cat_offset + inc0) as usize;
    if value == 0 {
        enc.encode_decision(ctxs.at_mut(ctx0), 0);
        return;
    }
    enc.encode_decision(ctxs.at_mut(ctx0), 1);
    let cat_minus = if (block_type as u32) == 3 { 1 } else { 0 };
    let inc_rest = 5 + core::cmp::min(4u32 - cat_minus, num_decoded_gt_1);
    let ctx_rest = (base + cat_offset + inc_rest) as usize;
    if value < u_coff {
        // Emit (value-1) more '1' bins followed by a '0'.
        for _ in 0..(value - 1) {
            enc.encode_decision(ctxs.at_mut(ctx_rest), 1);
        }
        enc.encode_decision(ctxs.at_mut(ctx_rest), 0);
        return;
    }
    // value >= u_coff. The decoder reads bins until prefix_val (starting
    // at 1) reaches u_coff. We've already emitted bin0=1 (which the
    // decoder reads BEFORE the loop, accounting for prefix_val=1). So
    // the loop reads u_coff-1 more '1' bins (no terminating '0' — the
    // suffix takes over).
    for _ in 0..(u_coff - 1) {
        enc.encode_decision(ctxs.at_mut(ctx_rest), 1);
    }
    // Exp-Golomb (k=0) suffix bypass.
    let mut suf = value - u_coff;
    let mut k: u32 = 0;
    while suf >= (1u32 << k) {
        enc.encode_bypass(1);
        suf -= 1u32 << k;
        k += 1;
    }
    enc.encode_bypass(0);
    for i in (0..k).rev() {
        let b = ((suf >> i) & 1) as u8;
        enc.encode_bypass(b);
    }
}

// ---------------------------------------------------------------------------
// Bypass / terminator-coded elements.
// ---------------------------------------------------------------------------

pub fn encode_coeff_sign_flag(enc: &mut CabacEncoder, neg: bool) {
    enc.encode_bypass(if neg { 1 } else { 0 });
}

pub fn encode_end_of_slice_flag(enc: &mut CabacEncoder, end_of_slice: bool) {
    enc.encode_terminate(if end_of_slice { 1 } else { 0 });
}

/// §9.3.3.1.* — encode `prev_intra4x4_pred_mode_flag`. ctxIdx 68.
pub fn encode_prev_intra_pred_mode_flag(
    enc: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    flag: bool,
) {
    enc.encode_decision(ctxs.at_mut(68), if flag { 1 } else { 0 });
}

/// §9.3.3.1.* — encode `rem_intra4x4_pred_mode` (FL cMax=7, 3 bins).
pub fn encode_rem_intra_pred_mode(enc: &mut CabacEncoder, ctxs: &mut CabacContexts, value: u32) {
    debug_assert!(value <= 7);
    for i in 0..3 {
        let bin = ((value >> i) & 1) as u8;
        enc.encode_decision(ctxs.at_mut(69), bin);
    }
}

// ---------------------------------------------------------------------------
// §7.3.5.3 — full residual block CABAC encoder.
//
// Combines coded_block_flag + the per-coefficient significant /
// last-significant + abs-level + sign emission into a single helper
// that mirrors `crate::macroblock_layer::parse_residual_block_cabac`
// step-for-step.
// ---------------------------------------------------------------------------

/// §7.3.5.3 — encode one residual block under CABAC.
///
/// `coeffs` are the scan-order quantised levels (length must equal
/// `max_num_coeff` for full DC+AC blocks, 15 for AC-only blocks, 4
/// for 4:2:0 chroma DC). `block_type` selects the ctxBlockCat / Table
/// 9-40 family. `cbf_neighbour_left/above` carry the §9.3.3.1.1.9
/// left/above coded_block_flag values.
///
/// When the block has no non-zero coefficients (`coded_block_flag = 0`)
/// only a single CBF bin is emitted. Otherwise the full
/// significance-map + level + sign tree is emitted per §7.3.5.3.
///
/// Returns `coded_block_flag` (true if any non-zero coefficient was
/// emitted) so the caller can update its neighbour grid.
pub fn encode_residual_block_cabac(
    enc: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    block_type: BlockType,
    coeffs: &[i32],
    max_num_coeff: u32,
    cbf_neighbour_left: Option<bool>,
    cbf_neighbour_above: Option<bool>,
    skip_cbf: bool,
) -> bool {
    debug_assert_eq!(coeffs.len(), max_num_coeff as usize);
    let any_nz = coeffs.iter().any(|&c| c != 0);
    if !skip_cbf {
        encode_coded_block_flag(
            enc,
            ctxs,
            block_type,
            cbf_neighbour_left,
            cbf_neighbour_above,
            any_nz,
        );
    }
    if !any_nz {
        return false;
    }
    // Find the index of the last non-zero coefficient (in scan order).
    let last_idx = coeffs
        .iter()
        .enumerate()
        .rev()
        .find(|(_, c)| **c != 0)
        .map(|(i, _)| i as u32)
        .expect("any_nz implies a last non-zero exists");

    // §7.3.5.3 — emit (significant_coeff_flag, last_significant_coeff_flag)
    // pairs for positions 0..last_idx; at last_idx, emit
    // significant_coeff_flag = 1 followed by last_significant_coeff_flag = 1.
    // Positions beyond last_idx are not emitted.
    let last_position_exclusive = max_num_coeff - 1;
    for i in 0..last_idx {
        let sig = coeffs[i as usize] != 0;
        encode_significant_coeff_flag(enc, ctxs, block_type, i, false, sig);
        if sig {
            encode_last_significant_coeff_flag(enc, ctxs, block_type, i, false, false);
        }
    }
    // Position `last_idx` always has significant=1, last=1 (unless
    // last_idx == max_num_coeff - 1, in which case significance map
    // for the final position is implied — the spec stops the loop).
    if last_idx < last_position_exclusive {
        encode_significant_coeff_flag(enc, ctxs, block_type, last_idx, false, true);
        encode_last_significant_coeff_flag(enc, ctxs, block_type, last_idx, false, true);
    } else {
        // last_idx == max_num_coeff - 1. The decoder doesn't read sig/last
        // for the final position — it's implied. Nothing to emit here.
    }

    // §7.3.5.3 — emit |level| - 1 (abs value) and sign for each non-zero
    // coefficient, walked from the LAST non-zero back to the FIRST.
    // Counters track running "decoded" stats for ctxIdxInc derivation.
    let mut num_decoded_eq_1: u32 = 0;
    let mut num_decoded_gt_1: u32 = 0;
    for i in (0..=last_idx).rev() {
        let level = coeffs[i as usize];
        if level == 0 {
            continue;
        }
        let abs = level.unsigned_abs();
        let abs_minus1 = abs - 1;
        encode_coeff_abs_level_minus1(
            enc,
            ctxs,
            block_type,
            num_decoded_eq_1,
            num_decoded_gt_1,
            abs_minus1,
        );
        encode_coeff_sign_flag(enc, level < 0);
        if abs == 1 {
            num_decoded_eq_1 += 1;
        } else {
            num_decoded_gt_1 += 1;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Tests — encode a known value, decode it back and check equality.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitReader;
    use crate::cabac::CabacDecoder;
    use crate::cabac_ctx::{
        decode_coded_block_flag, decode_coded_block_pattern, decode_coeff_abs_level_minus1,
        decode_coeff_sign_flag, decode_end_of_slice_flag, decode_intra_chroma_pred_mode,
        decode_last_significant_coeff_flag, decode_mb_qp_delta, decode_mb_skip_flag,
        decode_mb_type_b, decode_mb_type_i, decode_mb_type_p, decode_mvd_lx,
        decode_prev_intra_pred_mode_flag, decode_ref_idx_lx, decode_rem_intra_pred_mode,
        decode_significant_coeff_flag,
    };

    fn roundtrip<F1, F2>(encode_side: F1, decode_side: F2) -> Vec<u8>
    where
        F1: FnOnce(&mut CabacEncoder, &mut CabacContexts),
        F2: FnOnce(&mut CabacDecoder<'_>, &mut CabacContexts),
    {
        // I-slice context init (no init_idc) at QP 26.
        let mut ctxs_enc = CabacContexts::init(SliceKind::I, None, 26).unwrap();
        let mut enc = CabacEncoder::new();
        encode_side(&mut enc, &mut ctxs_enc);
        // Terminate the stream with a '1' so the decoder can see end-of-data.
        enc.encode_terminate(1);
        let bytes = enc.finish_no_trailing();
        let mut ctxs_dec = CabacContexts::init(SliceKind::I, None, 26).unwrap();
        let mut dec = CabacDecoder::new(BitReader::new(&bytes)).unwrap();
        decode_side(&mut dec, &mut ctxs_dec);
        assert_eq!(dec.decode_terminate().unwrap(), 1);
        bytes
    }

    #[test]
    fn rt_mb_skip_flag_p_slice() {
        let mut ctxs_enc = CabacContexts::init(SliceKind::P, Some(0), 26).unwrap();
        let mut enc = CabacEncoder::new();
        encode_mb_skip_flag(
            &mut enc,
            &mut ctxs_enc,
            SliceKind::P,
            &NeighbourCtx::default(),
            true,
        );
        enc.encode_terminate(1);
        let bytes = enc.finish_no_trailing();
        let mut ctxs_dec = CabacContexts::init(SliceKind::P, Some(0), 26).unwrap();
        let mut dec = CabacDecoder::new(BitReader::new(&bytes)).unwrap();
        assert!(decode_mb_skip_flag(
            &mut dec,
            &mut ctxs_dec,
            SliceKind::P,
            &NeighbourCtx::default()
        )
        .unwrap());
        assert_eq!(dec.decode_terminate().unwrap(), 1);
    }

    #[test]
    fn rt_mb_type_i_all_values() {
        for v in 0u32..=25 {
            let nb = NeighbourCtx::default();
            let bytes = roundtrip(
                |enc, ctxs| encode_mb_type_i(enc, ctxs, &nb, v),
                |dec, ctxs| {
                    let got = decode_mb_type_i(dec, ctxs, &nb).unwrap();
                    assert_eq!(got, v, "mb_type_i value {v} mismatch");
                },
            );
            assert!(!bytes.is_empty(), "empty encode for mb_type {v}");
        }
    }

    #[test]
    fn rt_mb_type_p_inter_values() {
        for v in [0u32, 1, 2, 3] {
            let mut ctxs_enc = CabacContexts::init(SliceKind::P, Some(0), 26).unwrap();
            let mut enc = CabacEncoder::new();
            let nb = NeighbourCtx::default();
            encode_mb_type_p(&mut enc, &mut ctxs_enc, &nb, v);
            enc.encode_terminate(1);
            let bytes = enc.finish_no_trailing();
            let mut ctxs_dec = CabacContexts::init(SliceKind::P, Some(0), 26).unwrap();
            let mut dec = CabacDecoder::new(BitReader::new(&bytes)).unwrap();
            let got = decode_mb_type_p(&mut dec, &mut ctxs_dec, &nb).unwrap();
            assert_eq!(got, v, "P mb_type {v} round-trip");
            assert_eq!(dec.decode_terminate().unwrap(), 1);
        }
    }

    #[test]
    fn rt_mb_type_b_inter_values() {
        // Cover every Table 9-37 inter row + a couple intra and the
        // B_8x8 corner.
        let test_values: &[u32] = &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            30, 48,
        ];
        for &v in test_values {
            let mut ctxs_enc = CabacContexts::init(SliceKind::B, Some(0), 26).unwrap();
            let mut enc = CabacEncoder::new();
            let nb = NeighbourCtx::default();
            encode_mb_type_b(&mut enc, &mut ctxs_enc, &nb, v);
            enc.encode_terminate(1);
            let bytes = enc.finish_no_trailing();
            let mut ctxs_dec = CabacContexts::init(SliceKind::B, Some(0), 26).unwrap();
            let mut dec = CabacDecoder::new(BitReader::new(&bytes)).unwrap();
            let got = decode_mb_type_b(&mut dec, &mut ctxs_dec, &nb).unwrap();
            assert_eq!(got, v, "B mb_type {v} round-trip");
            assert_eq!(dec.decode_terminate().unwrap(), 1);
        }
    }

    #[test]
    fn rt_mb_type_p_intra_values() {
        for v in [5u32, 6, 12, 25, 30] {
            let mut ctxs_enc = CabacContexts::init(SliceKind::P, Some(0), 26).unwrap();
            let mut enc = CabacEncoder::new();
            let nb = NeighbourCtx::default();
            encode_mb_type_p(&mut enc, &mut ctxs_enc, &nb, v);
            enc.encode_terminate(1);
            let bytes = enc.finish_no_trailing();
            let mut ctxs_dec = CabacContexts::init(SliceKind::P, Some(0), 26).unwrap();
            let mut dec = CabacDecoder::new(BitReader::new(&bytes)).unwrap();
            let got = decode_mb_type_p(&mut dec, &mut ctxs_dec, &nb).unwrap();
            assert_eq!(got, v, "P intra mb_type {v} round-trip");
            assert_eq!(dec.decode_terminate().unwrap(), 1);
        }
    }

    #[test]
    fn rt_mb_qp_delta_values() {
        for v in [-5, -1, 0, 1, 7, 12] {
            roundtrip(
                |enc, ctxs| encode_mb_qp_delta(enc, ctxs, false, v),
                |dec, ctxs| {
                    let got = decode_mb_qp_delta(dec, ctxs, false).unwrap();
                    assert_eq!(got, v);
                },
            );
        }
    }

    #[test]
    fn rt_ref_idx_lx_unary() {
        for v in [0u32, 1, 2, 3, 7] {
            roundtrip(
                |enc, ctxs| encode_ref_idx_lx(enc, ctxs, false, false, v),
                |dec, ctxs| {
                    let got = decode_ref_idx_lx(dec, ctxs, false, false).unwrap();
                    assert_eq!(got, v);
                },
            );
        }
    }

    #[test]
    fn rt_mvd_lx_uneg3() {
        for v in [-15, -1, 0, 1, 8, 9, 16, 100] {
            roundtrip(
                |enc, ctxs| encode_mvd_lx(enc, ctxs, MvdComponent::X, 0, v),
                |dec, ctxs| {
                    let got = decode_mvd_lx(dec, ctxs, MvdComponent::X, 0).unwrap();
                    assert_eq!(got, v);
                },
            );
        }
    }

    #[test]
    fn rt_intra_chroma_pred_mode() {
        for v in 0u32..=3 {
            let nb = NeighbourCtx::default();
            roundtrip(
                |enc, ctxs| encode_intra_chroma_pred_mode(enc, ctxs, &nb, v),
                |dec, ctxs| {
                    let got = decode_intra_chroma_pred_mode(dec, ctxs, &nb).unwrap();
                    assert_eq!(got, v);
                },
            );
        }
    }

    #[test]
    fn rt_coded_block_pattern_intra_420() {
        // Intra-style CBPs for 4:2:0.
        for cbp_l in [0u8, 1, 5, 15] {
            for cbp_c in [0u8, 1, 2] {
                let nb = NeighbourCtx::default();
                let mut ctxs_enc = CabacContexts::init(SliceKind::I, None, 26).unwrap();
                let mut enc = CabacEncoder::new();
                encode_coded_block_pattern(&mut enc, &mut ctxs_enc, &nb, 1, cbp_l, cbp_c);
                enc.encode_terminate(1);
                let bytes = enc.finish_no_trailing();
                let mut ctxs_dec = CabacContexts::init(SliceKind::I, None, 26).unwrap();
                let mut dec = CabacDecoder::new(BitReader::new(&bytes)).unwrap();
                let got = decode_coded_block_pattern(&mut dec, &mut ctxs_dec, &nb, 1).unwrap();
                let expected = (cbp_l as u32) | ((cbp_c as u32) << 4);
                assert_eq!(got, expected, "cbp luma={cbp_l} chroma={cbp_c}");
                assert_eq!(dec.decode_terminate().unwrap(), 1);
            }
        }
    }

    #[test]
    fn rt_coded_block_flag() {
        for &flag in &[false, true] {
            roundtrip(
                |enc, ctxs| {
                    encode_coded_block_flag(
                        enc,
                        ctxs,
                        BlockType::Luma4x4,
                        Some(false),
                        Some(false),
                        flag,
                    )
                },
                |dec, ctxs| {
                    let got = decode_coded_block_flag(
                        dec,
                        ctxs,
                        BlockType::Luma4x4,
                        Some(false),
                        Some(false),
                    )
                    .unwrap();
                    assert_eq!(got, flag);
                },
            );
        }
    }

    #[test]
    fn rt_significant_and_last_coeff_flag() {
        for &sig in &[false, true] {
            for &last in &[false, true] {
                roundtrip(
                    |enc, ctxs| {
                        encode_significant_coeff_flag(enc, ctxs, BlockType::Luma4x4, 3, false, sig);
                        encode_last_significant_coeff_flag(
                            enc,
                            ctxs,
                            BlockType::Luma4x4,
                            3,
                            false,
                            last,
                        );
                    },
                    |dec, ctxs| {
                        let s =
                            decode_significant_coeff_flag(dec, ctxs, BlockType::Luma4x4, 3, false)
                                .unwrap();
                        assert_eq!(s, sig);
                        let l = decode_last_significant_coeff_flag(
                            dec,
                            ctxs,
                            BlockType::Luma4x4,
                            3,
                            false,
                        )
                        .unwrap();
                        assert_eq!(l, last);
                    },
                );
            }
        }
    }

    #[test]
    fn rt_coeff_abs_level_minus1_small() {
        for v in [0u32, 1, 5, 13, 14, 15, 30, 100] {
            roundtrip(
                |enc, ctxs| {
                    encode_coeff_abs_level_minus1(enc, ctxs, BlockType::Luma4x4, 0, 0, v);
                },
                |dec, ctxs| {
                    let got =
                        decode_coeff_abs_level_minus1(dec, ctxs, BlockType::Luma4x4, 0, 0).unwrap();
                    assert_eq!(got, v, "value {v}: decoder returned {got}");
                },
            );
        }
    }

    #[test]
    fn rt_coeff_sign_flag_bypass() {
        for &neg in &[false, true] {
            roundtrip(
                |enc, _ctxs| encode_coeff_sign_flag(enc, neg),
                |dec, _ctxs| {
                    let got = decode_coeff_sign_flag(dec).unwrap();
                    assert_eq!(got, neg);
                },
            );
        }
    }

    #[test]
    fn rt_end_of_slice_flag() {
        // end_of_slice = false (the "not yet" path) followed by another decision
        // the decoder can verify it round-trips clean.
        let mut ctxs_enc = CabacContexts::init(SliceKind::I, None, 26).unwrap();
        let mut enc = CabacEncoder::new();
        encode_end_of_slice_flag(&mut enc, false);
        encode_mb_type_i(&mut enc, &mut ctxs_enc, &NeighbourCtx::default(), 0);
        encode_end_of_slice_flag(&mut enc, true);
        let bytes = enc.finish_no_trailing();

        let mut ctxs_dec = CabacContexts::init(SliceKind::I, None, 26).unwrap();
        let mut dec = CabacDecoder::new(BitReader::new(&bytes)).unwrap();
        assert!(!decode_end_of_slice_flag(&mut dec).unwrap());
        let mb_type = decode_mb_type_i(&mut dec, &mut ctxs_dec, &NeighbourCtx::default()).unwrap();
        assert_eq!(mb_type, 0);
        assert!(decode_end_of_slice_flag(&mut dec).unwrap());
    }

    /// Simulate the decoder's `parse_residual_block_cabac` step-for-step
    /// against our encoder, for a few known coefficient layouts.
    fn rt_residual_block(coeffs: &[i32], block_type: BlockType, max_num_coeff: u32) {
        let mut ctxs_enc = CabacContexts::init(SliceKind::I, None, 26).unwrap();
        let mut enc = CabacEncoder::new();
        encode_residual_block_cabac(
            &mut enc,
            &mut ctxs_enc,
            block_type,
            coeffs,
            max_num_coeff,
            Some(false),
            Some(false),
            false,
        );
        enc.encode_terminate(1);
        let bytes = enc.finish_no_trailing();

        // Replay against the decoder.
        let mut ctxs_dec = CabacContexts::init(SliceKind::I, None, 26).unwrap();
        let mut dec = CabacDecoder::new(BitReader::new(&bytes)).unwrap();

        let any_nz = coeffs.iter().any(|&c| c != 0);
        let coded = decode_coded_block_flag(
            &mut dec,
            &mut ctxs_dec,
            block_type,
            Some(false),
            Some(false),
        )
        .unwrap();
        assert_eq!(coded, any_nz);
        if !any_nz {
            assert_eq!(dec.decode_terminate().unwrap(), 1);
            return;
        }

        let span = max_num_coeff;
        let mut significant = vec![false; max_num_coeff as usize];
        let mut num_coeff_in_scan = span;
        let mut i: u32 = 0;
        while i + 1 < num_coeff_in_scan {
            let sig = decode_significant_coeff_flag(&mut dec, &mut ctxs_dec, block_type, i, false)
                .unwrap();
            significant[i as usize] = sig;
            if sig {
                let last = decode_last_significant_coeff_flag(
                    &mut dec,
                    &mut ctxs_dec,
                    block_type,
                    i,
                    false,
                )
                .unwrap();
                if last {
                    num_coeff_in_scan = i + 1;
                    break;
                }
            }
            i += 1;
        }
        significant[(num_coeff_in_scan - 1) as usize] = true;

        let mut decoded = vec![0i32; max_num_coeff as usize];
        let mut num_eq1 = 0u32;
        let mut num_gt1 = 0u32;
        for pos in (0..num_coeff_in_scan).rev() {
            if !significant[pos as usize] {
                continue;
            }
            let abs_m1 = decode_coeff_abs_level_minus1(
                &mut dec,
                &mut ctxs_dec,
                block_type,
                num_eq1,
                num_gt1,
            )
            .unwrap();
            let sign = decode_coeff_sign_flag(&mut dec).unwrap();
            decoded[pos as usize] = ((abs_m1 + 1) as i32) * if sign { -1 } else { 1 };
            if abs_m1 == 0 {
                num_eq1 += 1;
            } else {
                num_gt1 += 1;
            }
        }
        assert_eq!(decoded, coeffs);
        assert_eq!(dec.decode_terminate().unwrap(), 1);
    }

    #[test]
    fn rt_residual_block_all_zero() {
        rt_residual_block(&[0i32; 16], BlockType::Luma4x4, 16);
    }

    #[test]
    fn rt_residual_block_single_dc() {
        let mut c = [0i32; 16];
        c[0] = 5;
        rt_residual_block(&c, BlockType::Luma4x4, 16);
    }

    #[test]
    fn rt_residual_block_dc_plus_ac() {
        let mut c = [0i32; 16];
        c[0] = 12;
        c[5] = -3;
        c[10] = 2;
        rt_residual_block(&c, BlockType::Luma4x4, 16);
    }

    #[test]
    fn rt_residual_block_last_pos_set() {
        // Last non-zero at scan pos 15 — exercises the "implicit final sig" path.
        let mut c = [0i32; 16];
        c[0] = 1;
        c[15] = -1;
        rt_residual_block(&c, BlockType::Luma4x4, 16);
    }

    #[test]
    fn rt_residual_block_large_levels() {
        let mut c = [0i32; 16];
        c[0] = 100;
        c[3] = -50;
        rt_residual_block(&c, BlockType::Luma4x4, 16);
    }

    #[test]
    fn rt_intra4x4_pred_mode_flags() {
        roundtrip(
            |enc, ctxs| {
                encode_prev_intra_pred_mode_flag(enc, ctxs, true);
                encode_prev_intra_pred_mode_flag(enc, ctxs, false);
                encode_rem_intra_pred_mode(enc, ctxs, 5);
            },
            |dec, ctxs| {
                assert!(decode_prev_intra_pred_mode_flag(dec, ctxs).unwrap());
                assert!(!decode_prev_intra_pred_mode_flag(dec, ctxs).unwrap());
                let v = decode_rem_intra_pred_mode(dec, ctxs).unwrap();
                assert_eq!(v, 5);
            },
        );
    }
}
