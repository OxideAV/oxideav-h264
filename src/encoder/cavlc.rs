//! §9.2 — CAVLC entropy coding (encoder side).
//!
//! Inverse of [`crate::cavlc`]: instead of reading bits to determine the
//! coefficient block, we encode an already-quantized coefficient block
//! into bits per §9.2.1 .. §9.2.4.
//!
//! Re-uses the decoder's transcribed tables (every table is a slice of
//! `(bits, length, value)` rows): instead of looking up by `bits` we
//! look up by `value`. Tables remain the single source of truth — any
//! spec correction made there is automatically picked up by the
//! encoder.
//!
//! No external implementation was consulted. The encoder follows the
//! "informative" §9.2 invertibility property that the spec relies on.

use crate::cavlc::{
    rb_table, tz_chroma_420_table, tz_chroma_422_table, tz_luma_table, CoeffTokenContext,
    CoeffTokenRow, TotalZerosTable,
};
use crate::encoder::bitstream::BitWriter;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CavlcEncodeError {
    #[error("coeff_token (TC={tc}, T1={t1}) not present in nC table")]
    UnknownCoeffToken { tc: u32, t1: u32 },
    #[error("total_zeros={tz} not present for total_coeff={tc}")]
    UnknownTotalZeros { tc: u32, tz: u32 },
    #[error("run_before={rb} not present for zeros_left={zl}")]
    UnknownRunBefore { zl: u32, rb: u32 },
    #[error("coefficient out of representable range: {0}")]
    LevelOutOfRange(i32),
}

pub type CavlcEncodeResult<T> = Result<T, CavlcEncodeError>;

/// §9.2.1 — encode `coeff_token` (TotalCoeff, TrailingOnes) using the
/// table chosen by `ctx`.
pub fn encode_coeff_token(
    w: &mut BitWriter,
    ctx: CoeffTokenContext,
    total_coeff: u32,
    trailing_ones: u32,
) -> CavlcEncodeResult<()> {
    let table = ctx.select_table();
    encode_coeff_token_in(w, table, total_coeff, trailing_ones)
}

fn encode_coeff_token_in(
    w: &mut BitWriter,
    table: &[CoeffTokenRow],
    total_coeff: u32,
    trailing_ones: u32,
) -> CavlcEncodeResult<()> {
    for &(bits, length, (tc, t1)) in table {
        if tc == total_coeff && t1 == trailing_ones {
            w.u(length as u32, bits);
            return Ok(());
        }
    }
    Err(CavlcEncodeError::UnknownCoeffToken {
        tc: total_coeff,
        t1: trailing_ones,
    })
}

/// §9.2.2 — encode one non-trailing-ones level value. Mirrors
/// [`crate::cavlc::decode_level`] step-by-step. Returns the updated
/// `suffix_length` for the next iteration (per §9.2.2 steps 9–10).
pub fn encode_level(
    w: &mut BitWriter,
    suffix_length: u32,
    is_first_level_after_t1_lt_3: bool,
    level_val: i32,
) -> CavlcEncodeResult<u32> {
    if level_val == 0 {
        return Err(CavlcEncodeError::LevelOutOfRange(level_val));
    }

    // §9.2.2 step 8 inverse: derive levelCode from levelVal.
    //   if levelVal > 0:  levelCode = 2 * levelVal - 2  (even)
    //   else:             levelCode = -2 * levelVal - 1 (odd)
    let mut level_code: i64 = if level_val > 0 {
        2 * level_val as i64 - 2
    } else {
        -2 * level_val as i64 - 1
    };

    // §9.2.2 step 7 inverse: when this is the first non-T1 level after a
    // T1 < 3 run, the decoder added 2 — so the encoder subtracts 2.
    if is_first_level_after_t1_lt_3 {
        level_code -= 2;
    }
    debug_assert!(level_code >= 0, "levelCode underflow: {level_code}");

    // §9.2.2 step 4..6 inverse:
    //   levelCode = (Min(15, levelPrefix) << suffixLength) + levelSuffix
    //   plus extra +15 when (levelPrefix>=15 && suffixLength==0)
    //   plus      +(1<<(levelPrefix-3))-4096 when levelPrefix>=16
    //
    // We need to find a (levelPrefix, levelSuffix, levelSuffixSize) tuple
    // that the decoder will turn back into our `level_code`.
    //
    // The encoder picks the *minimum* level_prefix that lets the suffix
    // fit, which matches the decoder's reading rules.

    let (level_prefix, level_suffix, level_suffix_size) = if suffix_length > 0 {
        let suffix_size = suffix_length;
        let suffix_max = 1u64 << suffix_size; // exclusive upper bound
        let prefix = (level_code >> suffix_size).min(15);
        if prefix < 15 {
            // Standard short-prefix path: levelCode fits in `prefix << SL + suffix`.
            let suffix = (level_code - (prefix << suffix_size)) as u64;
            debug_assert!(suffix < suffix_max);
            (prefix as u32, suffix as u32, suffix_size)
        } else {
            // Escape path: levelPrefix >= 15. Per the decoder, when
            // prefix >= 15 the suffix size becomes `prefix - 3` (not
            // `suffix_length`). For prefix == 15 the level_code is just
            // `(15 << SL) + level_suffix`; for prefix >= 16 there's an
            // additional `(1 << (prefix - 3)) - 4096` term.
            //
            // Search for the smallest prefix that accommodates the
            // value. Range covered by prefix p (p ≥ 15):
            //   base_p = (15 << SL) + (if p>=16 then (1<<(p-3)) - 4096 else 0)
            //   level_code ∈ [base_p, base_p + 2^(p-3))
            let constant_15_sl = 15i64 << suffix_size;
            let mut prefix = 15u32;
            loop {
                let suffix_size_p = prefix - 3;
                let extra = if prefix >= 16 {
                    (1i64 << suffix_size_p) - 4096
                } else {
                    0
                };
                let base = constant_15_sl + extra;
                let cap = 1i64 << suffix_size_p;
                if level_code >= base && level_code < base + cap {
                    let suffix = (level_code - base) as u32;
                    break (prefix, suffix, suffix_size_p);
                }
                prefix += 1;
                if prefix > 31 {
                    return Err(CavlcEncodeError::LevelOutOfRange(level_val));
                }
            }
        }
    } else {
        // suffix_length == 0 path. Decoder semantics:
        //   prefix < 14 : level_code = prefix                            (range 0..=13)
        //   prefix == 14: level_code = 14 + level_suffix (4-bit suffix)  (range 14..=29)
        //   prefix == 15: level_code = 30 + level_suffix (12-bit suffix) (range 30..=4125)
        //   prefix >= 16: level_code = 30 + level_suffix + (1<<(p-3)) - 4096
        //                              with (p-3)-bit suffix
        if level_code < 14 {
            (level_code as u32, 0u32, 0u32)
        } else if level_code < 30 {
            (14u32, (level_code - 14) as u32, 4u32)
        } else {
            let mut prefix = 15u32;
            loop {
                let suffix_size_p = prefix - 3;
                let extra = if prefix >= 16 {
                    (1i64 << suffix_size_p) - 4096
                } else {
                    0
                };
                let base = 30 + extra;
                let cap = 1i64 << suffix_size_p;
                if level_code >= base && level_code < base + cap {
                    let suffix = (level_code - base) as u32;
                    break (prefix, suffix, suffix_size_p);
                }
                prefix += 1;
                if prefix > 31 {
                    return Err(CavlcEncodeError::LevelOutOfRange(level_val));
                }
            }
        }
    };

    // Emit level_prefix as a unary code: `level_prefix` zeros + a
    // terminating `1`.
    for _ in 0..level_prefix {
        w.u(1, 0);
    }
    w.u(1, 1);
    // Emit level_suffix when its size is non-zero.
    if level_suffix_size > 0 {
        w.u(level_suffix_size, level_suffix);
    }

    // §9.2.2 steps 9–10: update suffix_length for the next call.
    let mut next_sl = suffix_length;
    if next_sl == 0 {
        next_sl = 1;
    }
    let abs_level = level_val.unsigned_abs();
    if abs_level > (3u32 << (next_sl - 1)) && next_sl < 6 {
        next_sl += 1;
    }
    Ok(next_sl)
}

/// §9.2.3 — encode `total_zeros`.
pub fn encode_total_zeros(
    w: &mut BitWriter,
    total_coeff: u32,
    table: TotalZerosTable,
    total_zeros: u32,
) -> CavlcEncodeResult<()> {
    let rows = match table {
        TotalZerosTable::Luma => tz_luma_table(total_coeff),
        TotalZerosTable::ChromaDc420 => tz_chroma_420_table(total_coeff),
        TotalZerosTable::ChromaDc422 => tz_chroma_422_table(total_coeff),
    }
    .ok_or(CavlcEncodeError::UnknownTotalZeros {
        tc: total_coeff,
        tz: total_zeros,
    })?;
    for &(bits, length, value) in rows {
        if value == total_zeros {
            w.u(length as u32, bits);
            return Ok(());
        }
    }
    Err(CavlcEncodeError::UnknownTotalZeros {
        tc: total_coeff,
        tz: total_zeros,
    })
}

/// §9.2.3 — encode one `run_before`.
pub fn encode_run_before(
    w: &mut BitWriter,
    zeros_left: u32,
    run_before: u32,
) -> CavlcEncodeResult<()> {
    if zeros_left == 0 {
        // Decoder doesn't read bits when zerosLeft == 0; match.
        return Ok(());
    }
    let rows = rb_table(zeros_left);
    for &(bits, length, value) in rows {
        if value == run_before {
            w.u(length as u32, bits);
            return Ok(());
        }
    }
    Err(CavlcEncodeError::UnknownRunBefore {
        zl: zeros_left,
        rb: run_before,
    })
}

// ---------------------------------------------------------------------------
// High-level encoder for one residual_block_cavlc(...) syntax element.
// ---------------------------------------------------------------------------

/// §7.3.5.3.1 + §9.2 — encode one `residual_block_cavlc(coeffLevel,
/// startIdx, endIdx, maxNumCoeff)` block.
///
/// `coeff_level_in_scan_order` holds the levels for indices
/// `startIdx..=endIdx`, **in scan order** (caller has already applied
/// the forward zig-zag scan). The slice length must equal
/// `endIdx - startIdx + 1`.
pub fn encode_residual_block_cavlc(
    w: &mut BitWriter,
    ctx: CoeffTokenContext,
    max_num_coeff: u32,
    coeff_level_in_scan_order: &[i32],
) -> CavlcEncodeResult<()> {
    // §9.2.4 — derive (TotalCoeff, TrailingOnes, levelVal[], runVal[]).
    let span = coeff_level_in_scan_order.len();

    // Walk from highest scan index down. Build:
    //  * `levels[i]`: the i-th non-zero coefficient encountered (high-
    //    to-low scan order). Matches spec `levelVal[i]` indexing.
    //  * `runs[i]`: spec `runVal[i]` — the count of zeros immediately
    //    *below* (lower scan than) `levels[i]`, but stopping at the
    //    next non-zero (i.e. above `levels[i+1]` in scan terms).
    //
    // The leading high-scan zeros (above the highest non-zero) are
    // not part of any `runVal` — they're not even part of
    // `total_zeros` (§9.2.3 only counts zeros below the highest
    // non-zero). We skip them silently.
    let mut levels: Vec<i32> = Vec::new();
    let mut runs: Vec<u32> = Vec::new();
    let mut current_run: u32 = 0;
    let mut found_first = false;
    for k in (0..span).rev() {
        let v = coeff_level_in_scan_order[k];
        if v == 0 {
            if found_first {
                current_run += 1;
            }
        } else if !found_first {
            // First non-zero (highest-scan): this is `levelVal[0]`.
            // No `runVal` push yet — `runVal[i]` describes zeros
            // *below* `levelVal[i]`, which we haven't seen.
            levels.push(v);
            current_run = 0;
            found_first = true;
        } else {
            // The zeros we just accumulated are immediately below the
            // previously-pushed level, so they are spec
            // `runVal[len(runs)]`.
            runs.push(current_run);
            current_run = 0;
            levels.push(v);
        }
    }
    // Trailing accumulator: zeros below the lowest-scan non-zero, up
    // to and including scan index 0. This is spec
    // `runVal[total_coeff - 1]` (the implicit one — *not* emitted by
    // the encoder, but consulted to validate consistency below).
    if found_first {
        runs.push(current_run);
    }
    let total_coeff = levels.len() as u32;

    // Trailing ones (T1): the count of |level| == 1 at the *high-scan* end,
    // capped at 3. Levels were collected high-to-low, so the first
    // entries in `levels` are the highest-scan ones.
    let mut trailing_ones: u32 = 0;
    for &v in levels.iter().take(3) {
        if v.abs() == 1 {
            trailing_ones += 1;
        } else {
            break;
        }
    }
    debug_assert!(trailing_ones <= total_coeff);

    encode_coeff_token(w, ctx, total_coeff, trailing_ones)?;
    if total_coeff == 0 {
        return Ok(());
    }

    // §9.2.2 — `trailing_ones_sign_flag`s, encoded high-scan first.
    for &v in levels.iter().take(trailing_ones as usize) {
        // sign 0 → +1, sign 1 → -1 (per §9.2.2).
        let sign_bit = if v > 0 { 0 } else { 1 };
        w.u(1, sign_bit);
    }

    // §9.2.2 — remaining non-zero levels.
    let mut suffix_length: u32 = if total_coeff > 10 && trailing_ones < 3 {
        1
    } else {
        0
    };
    for (i_in_levels, &v) in levels.iter().enumerate().skip(trailing_ones as usize) {
        let is_first_after = i_in_levels == trailing_ones as usize && trailing_ones < 3;
        suffix_length = encode_level(w, suffix_length, is_first_after, v)?;
    }

    // §9.2.3 — total_zeros (when total_coeff != max_num_coeff).
    if total_coeff < max_num_coeff {
        // total_zeros = sum of (zeros to the left of the first non-zero
        // in low-scan-order, inclusive) — i.e. the count of zeros in
        // positions ≤ highest_non_zero_scan_index that are NOT
        // themselves non-zero coefficients.
        // Equivalent: highest_non_zero_scan_index - (total_coeff - 1)
        //             counting positions from index 0.
        // Find the highest non-zero scan index.
        let mut highest_nonzero: i32 = -1;
        for k in (0..span).rev() {
            if coeff_level_in_scan_order[k] != 0 {
                highest_nonzero = k as i32;
                break;
            }
        }
        let total_zeros = (highest_nonzero - (total_coeff as i32 - 1)) as u32;
        let tz_table = match max_num_coeff {
            4 => TotalZerosTable::ChromaDc420,
            8 => TotalZerosTable::ChromaDc422,
            _ => TotalZerosTable::Luma,
        };
        encode_total_zeros(w, total_coeff, tz_table, total_zeros)?;

        // §9.2.3 — run_before loop. We have `total_coeff - 1` writes,
        // each consuming runs[i] zeros. The last entry (runs[total_coeff-1])
        // is implicit (absorbs whatever zeros_left remains).
        let mut zeros_left = total_zeros;
        for &rb in runs.iter().take((total_coeff as usize).saturating_sub(1)) {
            if zeros_left > 0 {
                encode_run_before(w, zeros_left, rb)?;
            }
            zeros_left = zeros_left.saturating_sub(rb);
        }
        // Sanity: the last run must absorb all remaining zeros.
        debug_assert_eq!(
            zeros_left,
            *runs.last().unwrap_or(&0),
            "implicit final run mismatch (zeros_left={zeros_left})",
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitReader;
    use crate::cavlc::{
        decode_coeff_token, decode_run_before, decode_total_zeros, parse_residual_block_cavlc,
    };

    #[test]
    fn coeff_token_roundtrips() {
        // Spot-check several (TC, T1) entries across all five tables.
        let cases: &[(CoeffTokenContext, u32, u32)] = &[
            (CoeffTokenContext::Numeric(0), 0, 0),
            (CoeffTokenContext::Numeric(0), 4, 3),
            (CoeffTokenContext::Numeric(2), 1, 0),
            (CoeffTokenContext::Numeric(5), 6, 2),
            (CoeffTokenContext::Numeric(10), 8, 1),
            (CoeffTokenContext::ChromaDc420, 2, 1),
            (CoeffTokenContext::ChromaDc422, 4, 2),
        ];
        for &(ctx, tc, t1) in cases {
            let mut w = BitWriter::new();
            encode_coeff_token(&mut w, ctx, tc, t1).expect("encode");
            // Pad to a byte so the BitReader doesn't EOF.
            w.align_to_byte_zero();
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            let (tc_back, t1_back) = decode_coeff_token(&mut r, ctx).expect("decode");
            assert_eq!((tc_back, t1_back), (tc, t1));
        }
    }

    #[test]
    fn total_zeros_roundtrips() {
        for tc in 1..=15 {
            for tz in 0..(16 - tc) {
                let mut w = BitWriter::new();
                encode_total_zeros(&mut w, tc, TotalZerosTable::Luma, tz).expect("encode");
                w.align_to_byte_zero();
                let bytes = w.into_bytes();
                let mut r = BitReader::new(&bytes);
                let back = decode_total_zeros(&mut r, tc, TotalZerosTable::Luma).expect("decode");
                assert_eq!(back, tz, "tc={tc} tz={tz}");
            }
        }
    }

    #[test]
    fn run_before_roundtrips() {
        for zl in 1..=14 {
            for rb in 0..=zl {
                let mut w = BitWriter::new();
                encode_run_before(&mut w, zl, rb).expect("encode");
                w.align_to_byte_zero();
                let bytes = w.into_bytes();
                let mut r = BitReader::new(&bytes);
                let back = decode_run_before(&mut r, zl).expect("decode");
                assert_eq!(back, rb, "zl={zl} rb={rb}");
            }
        }
    }

    fn roundtrip_block(scan: &[i32], ctx: CoeffTokenContext, max_num_coeff: u32) {
        let mut w = BitWriter::new();
        encode_residual_block_cavlc(&mut w, ctx, max_num_coeff, scan).expect("encode");
        w.align_to_byte_zero();
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        // For DC blocks startIdx=0, endIdx=maxNumCoeff-1.
        let start_idx = 0u32;
        let end_idx = scan.len() as u32 - 1;
        let decoded = parse_residual_block_cavlc(&mut r, ctx, start_idx, end_idx, max_num_coeff)
            .expect("decode");
        assert_eq!(decoded.len(), scan.len());
        for (k, (got, want)) in decoded.iter().zip(scan.iter()).enumerate() {
            assert_eq!(got, want, "k={k} got={got} want={want} scan={scan:?}");
        }
    }

    #[test]
    fn block_all_zeros_roundtrips() {
        let scan = [0i32; 16];
        roundtrip_block(&scan, CoeffTokenContext::Numeric(0), 16);
    }

    #[test]
    fn block_single_nonzero_roundtrips() {
        let mut scan = [0i32; 16];
        scan[0] = 5;
        roundtrip_block(&scan, CoeffTokenContext::Numeric(0), 16);
        scan[0] = 0;
        scan[3] = -3;
        roundtrip_block(&scan, CoeffTokenContext::Numeric(0), 16);
    }

    #[test]
    fn block_three_trailing_ones_roundtrips() {
        // T1 = 3, then a non-T1 level.
        let scan: [i32; 16] = [4, 0, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        roundtrip_block(&scan, CoeffTokenContext::Numeric(0), 16);
    }

    #[test]
    fn block_dense_roundtrips() {
        let scan: [i32; 16] = [12, -7, 5, -3, 2, -2, 1, -1, 1, 0, -1, 1, 0, 0, 1, 0];
        roundtrip_block(&scan, CoeffTokenContext::Numeric(5), 16);
    }

    #[test]
    fn encode_level_roundtrips_across_suffix_lengths() {
        use crate::cavlc::decode_level;
        // Cover small / mid / large levels and every relevant
        // suffix_length entering the decoder.
        let levels: &[i32] = &[
            1, -1, 2, -2, 7, -7, 16, -16, 60, -60, 200, -200, 1000, -1000, 5000, -5000,
        ];
        let suffix_lengths: &[u32] = &[0, 1, 2, 3, 4, 5, 6];
        for &lv in levels {
            for &sl in suffix_lengths {
                for is_first in [false, true] {
                    // is_first only matters when sl == 0 with the
                    // T1<3 path; spec gating: only relevant on the very
                    // first non-T1 level when T1 < 3.
                    if is_first && sl != 0 {
                        continue;
                    }
                    // §9.2.2 step 7: when is_first_after_t1_lt_3 fires
                    // the level absolute value is by construction >= 2
                    // (else it would have been counted as a T1).
                    if is_first && lv.abs() < 2 {
                        continue;
                    }
                    let mut w = BitWriter::new();
                    let next_sl = encode_level(&mut w, sl, is_first, lv).expect("encode");
                    w.align_to_byte_zero();
                    let bytes = w.into_bytes();
                    let mut r = BitReader::new(&bytes);
                    let (lv_back, next_sl_back) =
                        decode_level(&mut r, sl, is_first).expect("decode");
                    assert_eq!(
                        lv_back, lv,
                        "level mismatch lv={lv} sl={sl} is_first={is_first}"
                    );
                    assert_eq!(next_sl_back, next_sl, "next_sl mismatch lv={lv} sl={sl}",);
                }
            }
        }
    }

    #[test]
    fn chroma_dc_420_block_roundtrips() {
        let scan: [i32; 4] = [3, -1, 1, 0];
        roundtrip_block(&scan, CoeffTokenContext::ChromaDc420, 4);
    }
}
