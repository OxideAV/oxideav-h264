//! §7.3.2.1.1.1 — scaling_list parsing.
//!
//! The `scaling_list(scalingList, sizeOfScalingList, useDefaultScalingMatrixFlag)`
//! syntax structure from §7.3.2.1.1.1 derives entries of a 4×4 (size 16) or
//! 8×8 (size 64) scaling matrix from a series of signed-Exp-Golomb
//! `delta_scale` values, and may signal that the caller should substitute
//! the default matrix from Tables 7-3 / 7-4 instead (semantics in
//! §7.4.2.1.1.1).
//!
//! The algorithm from the spec:
//!
//! ```text
//! lastScale = 8
//! nextScale = 8
//! for( j = 0; j < sizeOfScalingList; j++ ) {
//!     if( nextScale != 0 ) {
//!         delta_scale                                        // se(v)
//!         nextScale = ( lastScale + delta_scale + 256 ) % 256
//!         useDefaultScalingMatrixFlag = ( j == 0 && nextScale == 0 )
//!     }
//!     scalingList[ j ] = ( nextScale == 0 ) ? lastScale : nextScale
//!     lastScale = scalingList[ j ]
//! }
//! ```

use crate::bitstream::{BitError, BitReader};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ScalingListError {
    #[error("bitstream read failed: {0}")]
    Bitstream(#[from] BitError),
    #[error("scaling_list size must be 16 or 64 (got {0})")]
    InvalidSize(u32),
}

/// Result of parsing a single `scaling_list()` syntax structure
/// (§7.3.2.1.1.1). When `use_default` is `true` the caller should
/// substitute the default scaling matrix for this list per §7.4.2.1.1.1
/// (Tables 7-3 / 7-4) — the raw `scaling_list` field then holds the
/// values as derived by the pseudo-code, but they should be discarded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalingListResult {
    /// One value per scaling-list entry (length = sizeOfScalingList).
    pub scaling_list: Vec<i32>,
    /// `useDefaultScalingMatrixFlag` — when true, caller substitutes the
    /// default scaling matrix per Tables 7-3 / 7-4.
    pub use_default: bool,
}

/// §7.3.2.1.1.1 — parse one `scaling_list()` structure.
///
/// `size_of_scaling_list` must be 16 (4×4 list) or 64 (8×8 list); any
/// other size is rejected as malformed, per the constraints in
/// §7.4.2.1.1.1.
pub fn parse_scaling_list(
    r: &mut BitReader<'_>,
    size_of_scaling_list: u32,
) -> Result<ScalingListResult, ScalingListError> {
    if size_of_scaling_list != 16 && size_of_scaling_list != 64 {
        return Err(ScalingListError::InvalidSize(size_of_scaling_list));
    }

    // §7.3.2.1.1.1 — direct transcription of the spec pseudo-code.
    let n = size_of_scaling_list as usize;
    let mut scaling_list: Vec<i32> = Vec::with_capacity(n);
    let mut use_default = false;
    let mut last_scale: i32 = 8;
    let mut next_scale: i32 = 8;

    for j in 0..n {
        if next_scale != 0 {
            // se(v) delta_scale. §7.4.2.1.1.1 restricts delta_scale to
            // the range −128..=127 — we don't enforce that here since
            // the derivation is bit-exact regardless (the modulo-256 in
            // the spec masks the constraint violation away).
            let delta_scale = r.se()?;
            next_scale = (last_scale + delta_scale + 256).rem_euclid(256);
            if j == 0 && next_scale == 0 {
                use_default = true;
            }
        }
        let entry = if next_scale == 0 {
            last_scale
        } else {
            next_scale
        };
        scaling_list.push(entry);
        last_scale = entry;
    }

    Ok(ScalingListResult {
        scaling_list,
        use_default,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack a list of `(value, bit_width)` pairs into an MSB-first
    /// byte buffer. Used to assemble crafted bitstreams for tests.
    fn pack_bits(bits: &[(u32, u32)]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        let mut acc: u64 = 0;
        let mut nbits: u32 = 0;
        for &(val, w) in bits {
            assert!(w <= 32);
            // mask val to w bits
            let masked = if w == 32 {
                val
            } else {
                val & ((1u32 << w) - 1)
            };
            acc = (acc << w) | masked as u64;
            nbits += w;
            while nbits >= 8 {
                nbits -= 8;
                let byte = ((acc >> nbits) & 0xFF) as u8;
                out.push(byte);
            }
        }
        if nbits > 0 {
            // left-align the remaining bits into the final byte (MSB-first).
            let byte = ((acc << (8 - nbits)) & 0xFF) as u8;
            out.push(byte);
        }
        out
    }

    /// Encode signed Exp-Golomb (se(v), §9.1.1) as (value, bit-width).
    fn se_bits(value: i32) -> (u32, u32) {
        // Mapping §9.1.1: codeNum k → value  =  (-1)^(k+1) * ceil(k/2).
        //   k=0 → 0; k=1 → 1; k=2 → -1; k=3 → 2; k=4 → -2; ...
        let k = if value <= 0 {
            (-(value as i64) as u32) * 2
        } else {
            value as u32 * 2 - 1
        };
        // ue(v) codeword for k: floor(log2(k+1)) leading zeros, then
        // (k+1) in binary.
        let m = k + 1;
        let leading = 31 - m.leading_zeros();
        let total = 2 * leading + 1;
        (m, total)
    }

    #[test]
    fn size_must_be_16_or_64() {
        let mut r = BitReader::new(&[0xFF]);
        assert_eq!(
            parse_scaling_list(&mut r, 8).unwrap_err(),
            ScalingListError::InvalidSize(8)
        );
        let mut r = BitReader::new(&[0xFF]);
        assert_eq!(
            parse_scaling_list(&mut r, 0).unwrap_err(),
            ScalingListError::InvalidSize(0)
        );
    }

    #[test]
    fn all_zero_deltas_yield_all_eights() {
        // 16 × delta_scale = 0 → nextScale stays 8, scalingList[j] = 8.
        // However after the first iteration nextScale stays non-zero and we
        // keep reading delta_scale for every j. Each se(0) = ue codeword "1".
        let bits: Vec<(u32, u32)> = (0..16).map(|_| se_bits(0)).collect();
        let bytes = pack_bits(&bits);
        let mut r = BitReader::new(&bytes);
        let res = parse_scaling_list(&mut r, 16).unwrap();
        assert_eq!(res.scaling_list, vec![8; 16]);
        assert!(!res.use_default);
    }

    #[test]
    fn use_default_when_first_next_scale_is_zero() {
        // delta_scale = -8 at j=0 → nextScale = (8 + (-8) + 256) % 256 = 0
        // → useDefaultScalingMatrixFlag = true. Subsequent iterations
        // skip the delta_scale read (nextScale == 0) and all entries
        // become lastScale = 8.
        let bits: Vec<(u32, u32)> = vec![se_bits(-8)];
        let bytes = pack_bits(&bits);
        let mut r = BitReader::new(&bytes);
        let res = parse_scaling_list(&mut r, 16).unwrap();
        assert!(res.use_default);
        assert_eq!(res.scaling_list, vec![8; 16]);
    }

    #[test]
    fn non_zero_delta_at_j0_does_not_set_use_default() {
        // delta_scale = 2 at j=0 → nextScale = 10. use_default stays false.
        // Then 15 × delta = 0 keeps nextScale = 10 through every step.
        let mut bits: Vec<(u32, u32)> = vec![se_bits(2)];
        for _ in 0..15 {
            bits.push(se_bits(0));
        }
        let bytes = pack_bits(&bits);
        let mut r = BitReader::new(&bytes);
        let res = parse_scaling_list(&mut r, 16).unwrap();
        assert!(!res.use_default);
        assert_eq!(res.scaling_list, vec![10; 16]);
    }

    #[test]
    fn mixed_deltas_4x4() {
        // Walk through: last=8,next=8
        //   j=0 delta=+2 → next=10, entry=10, last=10
        //   j=1 delta=-2 → next=8, entry=8, last=8
        //   j=2 delta=0 → next=8, entry=8, last=8
        //   j=3 delta=+4 → next=12, entry=12, last=12
        //   j=4..15 delta=0 → next stays 12, entry=12
        let mut bits: Vec<(u32, u32)> = vec![se_bits(2), se_bits(-2), se_bits(0), se_bits(4)];
        for _ in 4..16 {
            bits.push(se_bits(0));
        }
        let bytes = pack_bits(&bits);
        let mut r = BitReader::new(&bytes);
        let res = parse_scaling_list(&mut r, 16).unwrap();
        assert!(!res.use_default);
        let expected = {
            let mut v = vec![10, 8, 8, 12];
            v.extend(std::iter::repeat(12).take(12));
            v
        };
        assert_eq!(res.scaling_list, expected);
    }

    #[test]
    fn eight_by_eight_list_size_64() {
        // 64 × delta=0 → all 8s.
        let bits: Vec<(u32, u32)> = (0..64).map(|_| se_bits(0)).collect();
        let bytes = pack_bits(&bits);
        let mut r = BitReader::new(&bytes);
        let res = parse_scaling_list(&mut r, 64).unwrap();
        assert_eq!(res.scaling_list.len(), 64);
        assert_eq!(res.scaling_list, vec![8; 64]);
        assert!(!res.use_default);
    }

    #[test]
    fn once_next_scale_hits_zero_midway_rest_copy_last() {
        // last=8,next=8.
        //   j=0 delta=+2 → next=10, entry=10, last=10
        //   j=1 delta=-10 → next=(10-10+256)%256=0. j!=0 so use_default stays false.
        //                   entry = lastScale = 10 (since next==0), last=10
        //   j=2..15: nextScale is 0 → skip delta_scale read. entry = last = 10.
        let bits: Vec<(u32, u32)> = vec![se_bits(2), se_bits(-10)];
        let bytes = pack_bits(&bits);
        let mut r = BitReader::new(&bytes);
        let res = parse_scaling_list(&mut r, 16).unwrap();
        assert!(!res.use_default);
        assert_eq!(res.scaling_list, vec![10; 16]);
    }

    #[test]
    fn modulo_256_wrap_behaves_like_spec() {
        // delta_scale = 250 at j=0: next = (8 + 250 + 256) % 256 = 2
        // (spec uses unsigned modular arithmetic; we model with rem_euclid
        // on i32 since delta_scale is signed in syntax).
        //
        // However se(v) can't directly encode 250 — the spec restricts
        // delta_scale to −128..127; but internally we still compute via
        // modulo. Use delta_scale = -6 instead: next = (8-6+256)%256 = 2.
        let mut bits: Vec<(u32, u32)> = Vec::new();
        bits.push(se_bits(-6));
        for _ in 0..15 {
            bits.push(se_bits(0));
        }
        let bytes = pack_bits(&bits);
        let mut r = BitReader::new(&bytes);
        let res = parse_scaling_list(&mut r, 16).unwrap();
        assert!(!res.use_default);
        assert_eq!(res.scaling_list[0], 2);
        assert_eq!(res.scaling_list[15], 2);
    }

    #[test]
    fn pack_bits_helper_roundtrip() {
        // Sanity check the test helper.
        // Pack "1 010 011" = ue codewords 0, 1, 2 into one byte 0b1010_0110.
        let bytes = pack_bits(&[(0b1, 1), (0b010, 3), (0b011, 3)]);
        assert_eq!(bytes, vec![0b1010_0110 >> 1 << 1]);
        // Actually the above left-aligns remaining 0 bits into final byte;
        // with nbits=7 we get one byte with low bit zero.
        // Let's verify with a cleaner case: 8-bit aligned.
        let bytes = pack_bits(&[(0b1010_0110, 8)]);
        assert_eq!(bytes, vec![0b1010_0110]);
    }
}
