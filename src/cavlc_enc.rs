//! CAVLC residual-block encoder — inverse of [`crate::cavlc`] emit path.
//!
//! Emits `residual_block_cavlc()` for all five block kinds the baseline
//! I-frame encoder needs:
//!
//! * `BlockKind::Luma4x4`         — 16 coefficients in 4×4 raster (unused
//!                                  in the current Intra_16×16-only encoder
//!                                  but exposed for completeness).
//! * `BlockKind::Luma16x16Dc`     — 16 DC coefficients post-Hadamard.
//! * `BlockKind::Luma16x16Ac`     — 15 AC coefficients (position 0 forced 0).
//! * `BlockKind::ChromaAc`        — 15 AC coefficients.
//! * `BlockKind::ChromaDc2x2`     — 4 DC coefficients, 4:2:0.
//!
//! Pipeline (inverse of §9.2):
//!
//! 1. Map raster → coded order via the zig-zag table.
//! 2. Determine `total_coeff` (non-zero count) and `trailing_ones` (<=3
//!    trailing ±1 values at the highest-frequency end of the coded order).
//! 3. Emit `coeff_token` from the table selected by `nC`.
//! 4. Emit trailing-one signs.
//! 5. Emit `level_prefix` + `level_suffix` for the remaining non-trailing
//!    levels, highest-frequency first.
//! 6. Emit `total_zeros` (the number of zero coefficients before the last
//!    non-zero in coded order).
//! 7. Emit `run_before` for each non-zero except the highest-frequency one.

use oxideav_core::{Error, Result};

use crate::bitwriter::BitWriter;
use crate::cavlc::{BlockKind, ZIGZAG_4X4};

/// Forward lookups mirroring the decoder's tables in [`crate::cavlc`].
/// We include the same (len, code) pairs — indexed by
/// `TotalCoeff * 4 + TrailingOnes`.
const COEFF_TOKEN_LEN: [[u8; 68]; 4] = [
    [
        1, 0, 0, 0, 6, 2, 0, 0, 8, 6, 3, 0, 9, 8, 7, 5, 10, 9, 8, 6, 11, 10, 9, 7, 13, 11, 10, 8,
        13, 13, 11, 9, 13, 13, 13, 10, 14, 14, 13, 11, 14, 14, 14, 13, 15, 15, 14, 14, 15, 15, 15,
        14, 16, 15, 15, 15, 16, 16, 16, 15, 16, 16, 16, 16, 16, 16, 16, 16,
    ],
    [
        2, 0, 0, 0, 6, 2, 0, 0, 6, 5, 3, 0, 7, 6, 6, 4, 8, 6, 6, 4, 8, 7, 7, 5, 9, 8, 8, 6, 11, 9,
        9, 6, 11, 11, 11, 7, 12, 11, 11, 9, 12, 12, 12, 11, 12, 12, 12, 11, 13, 13, 13, 12, 13, 13,
        13, 13, 13, 14, 13, 13, 14, 14, 14, 13, 14, 14, 14, 14,
    ],
    [
        4, 0, 0, 0, 6, 4, 0, 0, 6, 5, 4, 0, 6, 5, 5, 4, 7, 5, 5, 4, 7, 5, 5, 4, 7, 6, 6, 4, 7, 6,
        6, 4, 8, 7, 7, 5, 8, 8, 7, 6, 9, 8, 8, 7, 9, 9, 8, 8, 9, 9, 9, 8, 10, 9, 9, 9, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10, 10,
    ],
    [
        6, 0, 0, 0, 6, 6, 0, 0, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
    ],
];

const COEFF_TOKEN_BITS: [[u8; 68]; 4] = [
    [
        1, 0, 0, 0, 5, 1, 0, 0, 7, 4, 1, 0, 7, 6, 5, 3, 7, 6, 5, 3, 7, 6, 5, 4, 15, 6, 5, 4, 11,
        14, 5, 4, 8, 10, 13, 4, 15, 14, 9, 4, 11, 10, 13, 12, 15, 14, 9, 12, 11, 10, 13, 8, 15, 1,
        9, 12, 11, 14, 13, 8, 7, 10, 9, 12, 4, 6, 5, 8,
    ],
    [
        3, 0, 0, 0, 11, 2, 0, 0, 7, 7, 3, 0, 7, 10, 9, 5, 7, 6, 5, 4, 4, 6, 5, 6, 7, 6, 5, 8, 15,
        6, 5, 4, 11, 14, 13, 4, 15, 10, 9, 4, 11, 14, 13, 12, 8, 10, 9, 8, 15, 14, 13, 12, 11, 10,
        9, 12, 7, 11, 6, 8, 9, 8, 10, 1, 7, 6, 5, 4,
    ],
    [
        15, 0, 0, 0, 15, 14, 0, 0, 11, 15, 13, 0, 8, 12, 14, 12, 15, 10, 11, 11, 11, 8, 9, 10, 9,
        14, 13, 9, 8, 10, 9, 8, 15, 14, 13, 13, 11, 14, 10, 12, 15, 10, 13, 12, 11, 14, 9, 12, 8,
        10, 13, 8, 13, 7, 9, 12, 9, 12, 11, 10, 5, 8, 7, 6, 1, 4, 3, 2,
    ],
    [
        3, 0, 0, 0, 0, 1, 0, 0, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    ],
];

const CHROMA_DC_COEFF_TOKEN_LEN: [u8; 20] =
    [2, 0, 0, 0, 6, 1, 0, 0, 6, 6, 3, 0, 6, 7, 7, 6, 6, 8, 8, 7];
const CHROMA_DC_COEFF_TOKEN_BITS: [u8; 20] =
    [1, 0, 0, 0, 7, 1, 0, 0, 4, 6, 1, 0, 3, 3, 2, 5, 2, 3, 2, 0];

const TOTAL_ZEROS_LEN: [&[u8]; 15] = [
    &[1, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9],
    &[3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6],
    &[4, 3, 3, 3, 4, 4, 3, 3, 4, 5, 5, 6, 5, 6],
    &[5, 3, 4, 4, 3, 3, 3, 4, 3, 4, 5, 5, 5],
    &[4, 4, 4, 3, 3, 3, 3, 3, 4, 5, 4, 5],
    &[6, 5, 3, 3, 3, 3, 3, 3, 4, 3, 6],
    &[6, 5, 3, 3, 3, 2, 3, 4, 3, 6],
    &[6, 4, 5, 3, 2, 2, 3, 3, 6],
    &[6, 6, 4, 2, 2, 3, 2, 5],
    &[5, 5, 3, 2, 2, 2, 4],
    &[4, 4, 3, 3, 1, 3],
    &[4, 4, 2, 1, 3],
    &[3, 3, 1, 2],
    &[2, 2, 1],
    &[1, 1],
];

const TOTAL_ZEROS_BITS: [&[u8]; 15] = [
    &[1, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 1],
    &[7, 6, 5, 4, 3, 5, 4, 3, 2, 3, 2, 3, 2, 1, 0],
    &[5, 7, 6, 5, 4, 3, 4, 3, 2, 3, 2, 1, 1, 0],
    &[3, 7, 5, 4, 6, 5, 4, 3, 3, 2, 2, 1, 0],
    &[5, 4, 3, 7, 6, 5, 4, 3, 2, 1, 1, 0],
    &[1, 1, 7, 6, 5, 4, 3, 2, 1, 1, 0],
    &[1, 1, 5, 4, 3, 3, 2, 1, 1, 0],
    &[1, 1, 1, 3, 3, 2, 2, 1, 0],
    &[1, 0, 1, 3, 2, 1, 1, 1],
    &[1, 0, 1, 3, 2, 1, 1],
    &[0, 1, 1, 2, 1, 3],
    &[0, 1, 1, 1, 1],
    &[0, 1, 1, 1],
    &[0, 1, 1],
    &[0, 1],
];

const CHROMA_DC_TOTAL_ZEROS_LEN: [[u8; 4]; 3] = [[1, 2, 3, 3], [1, 2, 2, 0], [1, 1, 0, 0]];
const CHROMA_DC_TOTAL_ZEROS_BITS: [[u8; 4]; 3] = [[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]];

const RUN_LEN: [&[u8]; 7] = [
    &[1, 1],
    &[1, 2, 2],
    &[2, 2, 2, 2],
    &[2, 2, 2, 3, 3],
    &[2, 2, 3, 3, 3, 3],
    &[2, 3, 3, 3, 3, 3, 3],
    &[3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11],
];
const RUN_BITS: [&[u8]; 7] = [
    &[1, 0],
    &[1, 1, 0],
    &[3, 2, 1, 0],
    &[3, 2, 1, 1, 0],
    &[3, 2, 3, 2, 1, 0],
    &[3, 0, 1, 3, 2, 5, 4],
    &[7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
];

fn nc_class(nc: i32) -> usize {
    if nc < 2 {
        0
    } else if nc < 4 {
        1
    } else if nc < 8 {
        2
    } else {
        3
    }
}

/// Map a raster-indexed 4×4 block to coded order and encode the
/// `residual_block_cavlc()` syntax element into `w`.
///
/// Returns `total_coeff` so the caller can update per-sub-block nC tables
/// for neighbour prediction.
pub fn encode_residual_block(
    w: &mut BitWriter,
    raster: &[i32; 16],
    nc: i32,
    kind: BlockKind,
) -> Result<u32> {
    // Pull the coefficients into coded order according to block kind.
    let (coded, max_num_coeff) = match kind {
        BlockKind::Luma4x4 | BlockKind::Luma16x16Dc => {
            let mut coded = [0i32; 16];
            for i in 0..16 {
                coded[i] = raster[ZIGZAG_4X4[i]];
            }
            (coded, 16usize)
        }
        BlockKind::Luma16x16Ac | BlockKind::ChromaAc => {
            let mut coded = [0i32; 16];
            for i in 0..15 {
                coded[i] = raster[ZIGZAG_4X4[i + 1]];
            }
            (coded, 15usize)
        }
        BlockKind::ChromaDc2x2 => {
            let mut coded = [0i32; 16];
            for i in 0..4 {
                coded[i] = raster[i];
            }
            (coded, 4usize)
        }
    };

    // Count non-zeros and their positions.
    let mut nonzero_positions: Vec<usize> = Vec::with_capacity(max_num_coeff);
    for i in 0..max_num_coeff {
        if coded[i] != 0 {
            nonzero_positions.push(i);
        }
    }
    let total_coeff = nonzero_positions.len() as u32;

    // Trailing ones: up to 3 trailing ±1 values at the tail (highest-frequency
    // end of coded order). They are the non-zero coefficients whose absolute
    // value is 1, appearing contiguously from the end of the non-zero list.
    let mut trailing_ones = 0u32;
    if !nonzero_positions.is_empty() {
        for &p in nonzero_positions.iter().rev() {
            if coded[p].abs() == 1 && trailing_ones < 3 {
                trailing_ones += 1;
            } else {
                break;
            }
        }
    }

    // Avoid the unencodable edge case: when trailing_ones == 3 and
    // suffix_length starts at 0 (i.e. total_coeff <= 10), the first
    // non-trailing level of magnitude +8 produces level_code=14 which
    // has no representation in the sl=0 grammar. Reduce trailing_ones to
    // 2 so the +2 adjustment shifts level_code down to 12 (representable).
    if trailing_ones == 3 && total_coeff >= 4 && (total_coeff as usize) <= 10 {
        let first_nt_pos = nonzero_positions[nonzero_positions.len() - 4];
        let first_nt_level = coded[first_nt_pos];
        if first_nt_level == 8 {
            trailing_ones = 2;
        }
    }

    // 1. coeff_token.
    let token_idx = (total_coeff * 4 + trailing_ones) as usize;
    if total_coeff > max_num_coeff as u32 {
        return Err(Error::invalid(
            "h264 cavlc_enc: total_coeff > max_num_coeff",
        ));
    }
    match kind {
        BlockKind::ChromaDc2x2 => {
            if token_idx >= CHROMA_DC_COEFF_TOKEN_LEN.len() {
                return Err(Error::invalid("h264 cavlc_enc: chroma DC token oob"));
            }
            let len = CHROMA_DC_COEFF_TOKEN_LEN[token_idx];
            let bits = CHROMA_DC_COEFF_TOKEN_BITS[token_idx];
            if len == 0 {
                return Err(Error::invalid(format!(
                    "h264 cavlc_enc: no chroma DC token for (tc={total_coeff}, t1={trailing_ones})"
                )));
            }
            w.write_bits(bits as u32, len as u32);
        }
        _ => {
            let cls = nc_class(nc);
            if token_idx >= COEFF_TOKEN_LEN[cls].len() {
                return Err(Error::invalid("h264 cavlc_enc: token oob"));
            }
            let len = COEFF_TOKEN_LEN[cls][token_idx];
            let bits = COEFF_TOKEN_BITS[cls][token_idx];
            if len == 0 {
                return Err(Error::invalid(format!(
                    "h264 cavlc_enc: no token entry for (nc_cls={cls}, tc={total_coeff}, t1={trailing_ones})"
                )));
            }
            w.write_bits(bits as u32, len as u32);
        }
    }

    if total_coeff == 0 {
        return Ok(0);
    }

    // 2. Trailing-one signs (highest-frequency first).
    // The decoder's loop in [`crate::cavlc`] reads
    //   levels[0..trailing_ones]   = trailing-one signs (HF-first)
    //   levels[trailing_ones..tc]  = level_prefix/suffix pairs
    // and the code places them into `coded[target]` with `target` walking
    // backwards from the end of the coded range. So `levels[0]` corresponds
    // to the *last* non-zero in coded order (highest frequency).
    let mut levels = Vec::with_capacity(total_coeff as usize);
    for &p in nonzero_positions.iter().rev() {
        levels.push(coded[p]);
    }

    for i in 0..trailing_ones as usize {
        // levels[i] is ±1. Sign bit: 1 for negative, 0 for positive.
        let sign_bit = if levels[i] < 0 { 1 } else { 0 };
        w.write_bit(sign_bit);
    }

    // 3. Remaining levels with level_prefix + level_suffix.
    let mut suffix_length: u32 = if total_coeff > 10 && trailing_ones < 3 {
        1
    } else {
        0
    };
    for i in trailing_ones as usize..total_coeff as usize {
        let level = levels[i];
        encode_level(w, &mut suffix_length, level, i, trailing_ones)?;
    }

    // 4. total_zeros.
    let zeros_left = if (total_coeff as usize) < max_num_coeff {
        // total_zeros = (last_nonzero_coded_pos + 1) - total_coeff, but
        // equivalently: number of zero coefficients appearing before the
        // last non-zero in coded order.
        let last_nz = *nonzero_positions.last().unwrap();
        (last_nz + 1) as u32 - total_coeff
    } else {
        0
    };

    if (total_coeff as usize) < max_num_coeff {
        write_total_zeros(w, total_coeff, zeros_left, kind)?;
    }

    // 5. run_before — emitted for each non-zero except the lowest-frequency.
    // The decoder reads runs in the order i = 0..total_coeff-1 which
    // corresponds to levels[0..total_coeff-1] (HF-first). For each it
    // subtracts the run from `zeros_left`. We compute the same sequence of
    // runs by walking the non-zero positions from HF to LF.
    let mut runs = Vec::with_capacity(total_coeff as usize);
    // run[i] is the number of zeros immediately *preceding* the i-th
    // nonzero (HF-first) in coded order.
    for i in 0..total_coeff as usize {
        let hf_ix = nonzero_positions.len() - 1 - i;
        let p = nonzero_positions[hf_ix];
        let p_prev = if hf_ix == 0 {
            -1i32
        } else {
            nonzero_positions[hf_ix - 1] as i32
        };
        let run = (p as i32 - p_prev - 1) as u32;
        runs.push(run);
    }

    let mut zl = zeros_left;
    for i in 0..total_coeff as usize - 1 {
        if zl == 0 {
            // No more zeros to distribute — run must be 0, nothing to write.
            break;
        }
        let r = runs[i];
        write_run_before(w, zl, r)?;
        zl = zl.saturating_sub(r);
    }

    Ok(total_coeff)
}

/// §9.2.2.1 inverse of `read_level` — emits `level_prefix` + `level_suffix`
/// for a single non-trailing level.
fn encode_level(
    w: &mut BitWriter,
    suffix_length: &mut u32,
    level: i32,
    coeff_idx_among_nontrailing: usize,
    trailing_ones: u32,
) -> Result<()> {
    // Compute level_code from the level (inverse of decode).
    // Decode: level = ((level_code + 2) >> 1) if even else -((level_code + 1) >> 1)
    // So: if level > 0: level_code = 2*level - 2
    //     if level < 0: level_code = -2*level - 1
    let abs = level.unsigned_abs() as i32;
    let mut level_code: i32 = if level > 0 { 2 * abs - 2 } else { 2 * abs - 1 };

    // The decoder's "+2 if first non-TO coeff and trailing_ones < 3" adjustment
    // was applied during decode. Here, if that condition holds we subtract 2
    // before computing the prefix/suffix.
    let is_first_nontrailing =
        coeff_idx_among_nontrailing == trailing_ones as usize && trailing_ones < 3;
    if is_first_nontrailing {
        level_code -= 2;
    }

    // Now level_code >= 0. Determine level_prefix from level_code and suffix_length.
    let sl = *suffix_length;
    if sl == 0 {
        // Representable sub-ranges (from the decoder's §9.2.2.1 formulae):
        //   level_code 0..=13   → level_prefix = level_code, no suffix
        //   level_code 14       → unrepresentable (must be avoided upstream)
        //   level_code 15..=4110 → level_prefix = 15 with 12-bit suffix
        //                          (levelCode = 15 + suffix, since the
        //                          +(1<<12)-4096 escape term vanishes)
        //   level_code 29..=44  → level_prefix = 14 with 4-bit suffix
        //                          (levelCode = 14 + suffix + 15)
        // We use prefix=15 for level_code >= 15 (simpler; range overlaps
        // with prefix=14 at 29..44 but the decoder accepts either path and
        // prefix=15 is shorter at the low end).
        if level_code < 14 {
            let prefix = level_code as u32;
            for _ in 0..prefix {
                w.write_bit(0);
            }
            w.write_bit(1);
        } else if level_code == 14 {
            // Hit by the "trailing_ones==3, sl=0, first non-trailing |level|=+8"
            // edge case. The caller is supposed to have downgraded trailing_ones
            // to avoid this; if it slipped through, error out rather than lie.
            return Err(Error::invalid(
                "h264 cavlc_enc: level_code=14 at suffix_length=0 is not representable",
            ));
        } else if level_code <= 4110 {
            // prefix = 15, 12-bit suffix = level_code - 15.
            let suffix = (level_code - 15) as u32;
            for _ in 0..15 {
                w.write_bit(0);
            }
            w.write_bit(1);
            w.write_bits(suffix, 12);
        } else {
            return Err(Error::invalid(format!(
                "h264 cavlc_enc: level_code {level_code} out of sl=0 encodable range"
            )));
        }
    } else {
        // sl >= 1: standard case.
        // levelCode = (prefix << sl) + suffix, with suffix in [0, 2^sl).
        // For prefix < 15, level_code fits in [prefix << sl, prefix << sl + 2^sl).
        // We want the smallest prefix such that (prefix << sl) <= level_code.
        let sl_mask = (1u32 << sl) - 1;
        let prefix = (level_code as u32) >> sl;
        if prefix < 15 {
            let suffix = (level_code as u32) & sl_mask;
            for _ in 0..prefix {
                w.write_bit(0);
            }
            w.write_bit(1);
            w.write_bits(suffix, sl);
        } else {
            // Escape: prefix >= 15 with level_suffix_size = prefix - 3 (8-bit profile).
            // We'll use prefix = 15 for moderate overflows. The decoder math:
            //   levelCode = (15 << sl) + suffix + (1 << (15-3)) - 4096
            //            = (15 << sl) + suffix
            //   (because 1<<12 == 4096, so those cancel).
            // Wait — actually (1 << (prefix - 3)) with prefix=15 equals 1<<12=4096,
            // minus 4096 = 0, so the escape just adds the `Min(15,prefix) << sl` term
            // plus the suffix. That works.
            //
            // For prefix >= 16, the `(1 << (prefix-3)) - 4096` term adds a positive
            // offset so larger level_codes can be represented. Given our dc quants,
            // we don't expect to need prefix > 15 for typical intra at QP >= 10.
            // If we do, panic out — the caller's content is out of scope for a
            // minimal encoder.
            // Handle levels up through prefix=15's full range (suffix_size = 12).
            let max_with_prefix15 = (15u32 << sl) + (1u32 << 12) - 1;
            if (level_code as u32) > max_with_prefix15 {
                return Err(Error::invalid(format!(
                    "h264 cavlc_enc: level magnitude {} out of supported range",
                    level.abs()
                )));
            }
            let suffix = (level_code as u32) - (15u32 << sl);
            // 15 zeros, then 1, then 12-bit suffix.
            for _ in 0..15 {
                w.write_bit(0);
            }
            w.write_bit(1);
            w.write_bits(suffix, 12);
        }
    }

    // Now the suffix_length update step.
    if *suffix_length == 0 {
        *suffix_length = 1;
    }
    if (level.unsigned_abs() as i32) > (3i32 << (*suffix_length - 1)) && *suffix_length < 6 {
        *suffix_length += 1;
    }
    Ok(())
}

fn write_total_zeros(
    w: &mut BitWriter,
    total_coeff: u32,
    total_zeros: u32,
    kind: BlockKind,
) -> Result<()> {
    let row = (total_coeff - 1) as usize;
    let (lens, bits): (&[u8], &[u8]) = match kind {
        BlockKind::ChromaDc2x2 => (
            &CHROMA_DC_TOTAL_ZEROS_LEN[row][..],
            &CHROMA_DC_TOTAL_ZEROS_BITS[row][..],
        ),
        _ => (TOTAL_ZEROS_LEN[row], TOTAL_ZEROS_BITS[row]),
    };
    let idx = total_zeros as usize;
    if idx >= lens.len() {
        return Err(Error::invalid(format!(
            "h264 cavlc_enc: total_zeros {total_zeros} out of table for tc={total_coeff}"
        )));
    }
    let len = lens[idx];
    let code = bits[idx];
    if len == 0 {
        return Err(Error::invalid(format!(
            "h264 cavlc_enc: no total_zeros entry for tc={total_coeff}, zeros={total_zeros}"
        )));
    }
    w.write_bits(code as u32, len as u32);
    Ok(())
}

fn write_run_before(w: &mut BitWriter, zeros_left: u32, run: u32) -> Result<()> {
    let row = zeros_left.min(7) as usize - 1;
    let lens = RUN_LEN[row];
    let bits = RUN_BITS[row];
    let idx = run as usize;
    if idx >= lens.len() {
        return Err(Error::invalid(format!(
            "h264 cavlc_enc: run_before {run} out of table for zeros_left={zeros_left}"
        )));
    }
    let len = lens[idx];
    let code = bits[idx];
    if len == 0 {
        return Err(Error::invalid(format!(
            "h264 cavlc_enc: no run_before entry (zl={zeros_left}, run={run})"
        )));
    }
    w.write_bits(code as u32, len as u32);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;
    use crate::cavlc::decode_residual_block;

    fn roundtrip(block: [i32; 16], nc: i32, kind: BlockKind) {
        let mut w = BitWriter::new();
        // Encode; for the test harness we also append a sentinel bit so
        // the reader doesn't fall off the end.
        encode_residual_block(&mut w, &block, nc, kind).expect("encode");
        // Byte-align with a stop bit at the end (mimic rbsp trailing).
        w.write_rbsp_trailing_bits();
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        let decoded = decode_residual_block(&mut r, nc, kind).expect("decode");
        assert_eq!(
            decoded.coeffs, block,
            "roundtrip mismatch for kind={:?}",
            kind
        );
    }

    #[test]
    fn roundtrip_all_zero() {
        roundtrip([0; 16], 0, BlockKind::Luma4x4);
        roundtrip([0; 16], 0, BlockKind::Luma16x16Ac);
        roundtrip([0; 16], 0, BlockKind::Luma16x16Dc);
        roundtrip([0; 16], 0, BlockKind::ChromaAc);
        let mut b = [0; 16];
        b[0] = 0;
        roundtrip(b, 0, BlockKind::ChromaDc2x2);
    }

    #[test]
    fn roundtrip_single_dc() {
        let mut b = [0; 16];
        b[0] = 5;
        roundtrip(b, 0, BlockKind::Luma4x4);
        b[0] = -3;
        roundtrip(b, 0, BlockKind::Luma16x16Dc);
    }

    #[test]
    fn roundtrip_mixed_small() {
        let b = [3, 1, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        roundtrip(b, 2, BlockKind::Luma4x4);
        roundtrip(b, 5, BlockKind::Luma4x4);
    }

    #[test]
    fn roundtrip_chroma_dc_2x2() {
        let mut b = [0; 16];
        b[0] = 4;
        b[1] = -2;
        b[2] = 1;
        b[3] = -1;
        roundtrip(b, 0, BlockKind::ChromaDc2x2);
    }

    #[test]
    fn roundtrip_larger_levels_sl1_path() {
        // Levels big enough to push suffix_length > 0 after the first non-trailing.
        let b = [10, -7, 4, 2, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        roundtrip(b, 4, BlockKind::Luma4x4);
    }
}
