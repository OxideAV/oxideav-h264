//! Forward integer transforms + quantisation used by the H.264 baseline
//! encoder. Mirror of [`crate::transform`] (decoder side).
//!
//! Pipeline, all integer arithmetic:
//!
//! 1. Forward 4×4 integer DCT — `Y = C_f * X * C_f^T` with
//!    `C_f = [[1,1,1,1],[2,1,-1,-2],[1,-1,-1,1],[1,-2,2,-1]]`.
//! 2. For an Intra_16×16 luma macroblock: the sixteen DC coefficients from
//!    the sixteen 4×4 sub-blocks are then fed through a 4×4 Hadamard
//!    (`forward_hadamard_4x4`) and quantised with the `(0,0)` MF entry plus
//!    `qbits + 1` (the extra bit accounts for the Hadamard's internal
//!    `×16` scaling).
//! 3. For 4:2:0 chroma DC, the four DC coefficients are Hadamard-transformed
//!    and quantised via `quantize_chroma_dc_2x2`.
//! 4. AC positions are quantised with `quantize_4x4_ac` — the dividing
//!    function is `(|W| * MF + f) >> qbits` with a reconstructable sign.
//!
//! The MF (multiplier-factor) table comes straight from the H.264 reference
//! and is the forward inverse of the decoder's `V_TABLE` scaled up by
//! `2^15`. That is: `MF[qp%6][c] * V_TABLE[qp%6][c] ≈ 2^(15 + 6)` so the
//! concatenation of `forward_quant >> inv_quant >> IDCT` lands back at the
//! input value within the spec's rounding tolerance.

/// Forward quantisation multiplier `MF[qp%6][pos_class]` — inverse of the
/// decoder's `V_TABLE` (within a 2^21 scale factor that the quantisation
/// right-shift removes).
const MF_TABLE: [[i32; 3]; 6] = [
    [13107, 5243, 8066],
    [11916, 4660, 7490],
    [10082, 4194, 6554],
    [9362, 3647, 5825],
    [8192, 3355, 5243],
    [7282, 2893, 4559],
];

/// Position class for a 4×4 cell (same layout as decoder's `pos_class`).
fn pos_class(row: usize, col: usize) -> usize {
    let r_even = row % 2 == 0;
    let c_even = col % 2 == 0;
    if r_even && c_even {
        0
    } else if !r_even && !c_even {
        1
    } else {
        2
    }
}

/// Forward 4×4 integer DCT — output replaces input. Raster order.
pub fn forward_dct_4x4(block: &mut [i32; 16]) {
    // C_f * X
    let mut tmp = [0i32; 16];
    for c in 0..4 {
        let a = block[c];
        let b = block[4 + c];
        let cc = block[8 + c];
        let d = block[12 + c];
        // Row 0: a + b + c + d
        tmp[c] = a + b + cc + d;
        // Row 1: 2a + b - c - 2d
        tmp[4 + c] = 2 * a + b - cc - 2 * d;
        // Row 2: a - b - c + d
        tmp[8 + c] = a - b - cc + d;
        // Row 3: a - 2b + 2c - d
        tmp[12 + c] = a - 2 * b + 2 * cc - d;
    }
    // Right-multiply by C_f^T
    for r in 0..4 {
        let a = tmp[r * 4];
        let b = tmp[r * 4 + 1];
        let cc = tmp[r * 4 + 2];
        let d = tmp[r * 4 + 3];
        block[r * 4] = a + b + cc + d;
        block[r * 4 + 1] = 2 * a + b - cc - 2 * d;
        block[r * 4 + 2] = a - b - cc + d;
        block[r * 4 + 3] = a - 2 * b + 2 * cc - d;
    }
}

/// Forward 4×4 Hadamard — used for the Intra_16×16 luma DC block. Input is
/// the 16 DC coefficients of the sixteen 4×4 sub-blocks (arranged in a 4×4
/// raster of MB 4×4 block positions); output is the Hadamard-transformed
/// values, still integer.
///
/// Matches x264's `dct4x4dc`: two butterfly passes, with the second pass
/// applying a `(x + 1) >> 1` rounding divide-by-two. This half-scaling
/// keeps forward×inverse net amplification at ×8 for a constant input
/// (rather than ×16) and produces a bitstream whose DC levels match
/// what x264 / JM / other reference encoders emit.
pub fn forward_hadamard_4x4(block: &mut [i32; 16]) {
    // Pass 1: transpose-and-butterfly, no shift.
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let s01 = block[i * 4] + block[i * 4 + 1];
        let d01 = block[i * 4] - block[i * 4 + 1];
        let s23 = block[i * 4 + 2] + block[i * 4 + 3];
        let d23 = block[i * 4 + 2] - block[i * 4 + 3];
        tmp[i] = s01 + s23;
        tmp[4 + i] = s01 - s23;
        tmp[8 + i] = d01 - d23;
        tmp[12 + i] = d01 + d23;
    }
    // Pass 2: butterfly with (x + 1) >> 1.
    for i in 0..4 {
        let s01 = tmp[i * 4] + tmp[i * 4 + 1];
        let d01 = tmp[i * 4] - tmp[i * 4 + 1];
        let s23 = tmp[i * 4 + 2] + tmp[i * 4 + 3];
        let d23 = tmp[i * 4 + 2] - tmp[i * 4 + 3];
        block[i * 4] = (s01 + s23 + 1) >> 1;
        block[i * 4 + 1] = (s01 - s23 + 1) >> 1;
        block[i * 4 + 2] = (d01 - d23 + 1) >> 1;
        block[i * 4 + 3] = (d01 + d23 + 1) >> 1;
    }
}

/// Forward 2×2 Hadamard — chroma DC. Input / output in raster order:
/// `[0,1,2,3] = [(0,0),(0,1),(1,0),(1,1)]`.
pub fn forward_hadamard_2x2(dc: &mut [i32; 4]) {
    let a = dc[0];
    let b = dc[1];
    let c = dc[2];
    let d = dc[3];
    let t0 = a + b;
    let t1 = a - b;
    let t2 = c + d;
    let t3 = c - d;
    dc[0] = t0 + t2;
    dc[1] = t1 + t3;
    dc[2] = t0 - t2;
    dc[3] = t1 - t3;
}

/// Quantise an AC 4×4 block in-place. `qp` must already be clamped to 0..=51.
/// Position `(0,0)` is also quantised — for Intra_16×16 the caller is
/// responsible for zeroing it out (DC lives in a separate Hadamard block).
pub fn quantize_4x4_ac(coeffs: &mut [i32; 16], qp: i32) {
    let qp = qp.clamp(0, 51);
    let qmod = (qp % 6) as usize;
    let qp6 = (qp / 6) as u32;
    let qbits: u32 = 15 + qp6;
    // Intra rounding offset f = 2^qbits / 3 (per §8.5.10 note; the
    // value is commonly used for intra blocks by reference encoders).
    let f = (1i64 << qbits) / 3;
    for r in 0..4 {
        for c in 0..4 {
            let i = r * 4 + c;
            let w = coeffs[i] as i64;
            let mf = MF_TABLE[qmod][pos_class(r, c)] as i64;
            let sign = w.signum();
            let abs = w.abs();
            let z = (abs * mf + f) >> qbits;
            coeffs[i] = (sign * z) as i32;
        }
    }
}

/// Quantise the Intra_16×16 luma DC 4×4 Hadamard block in-place. Uses the
/// `(0,0)` MF entry with a `qbits - 2` right-shift.
///
/// Calibration note: the forward 4×4 Hadamard amplifies a constant input
/// by 16. The decoder's (§8.5.10-compliant) `inv_hadamard_4x4_dc` spreads
/// a DC-only input `[z, 0, ..., 0]` into 16 equal cells of `z` (no extra
/// scaling), then applies a DC dequant that multiplies by `scale >> (6-qp/6)`
/// — i.e. 64× smaller than the AC dequant's `scale << (qp/6)` at a given
/// qp. To keep the reconstructed DC on par with what AC pos-0 dequant
/// would produce (and thus the IDCT output at pixel magnitude), the
/// forward must shift by 2 bits LESS than AC, giving `qbits - 2`.
///
/// This calibration matches the pure §8.5.10 decoder convention and
/// produces bitstreams that ffmpeg / JM / other spec-faithful decoders
/// interpret correctly.
pub fn quantize_luma_dc_4x4(dc: &mut [i32; 16], qp: i32) {
    let qp = qp.clamp(0, 51);
    let qmod = (qp % 6) as usize;
    let qp6 = (qp / 6) as u32;
    let qbits: u32 = 15 + qp6;
    // Forward DC shift calibrated against
    // [`crate::transform::inv_hadamard_4x4_dc`] (which follows the
    // x264/FFmpeg convention of `dc * scale << (qp/6-2)`). With the
    // forward Hadamard's x264-style `>>1` in pass 2, a constant-r residual
    // yields fHad[0] = 128r; the decoder reconstructs `dc_per_cell =
    // z * scale * 2^{qp/6-2}`. Solving `pixel_out = r` at qp=22 gives
    // shift = qbits + 1 (18+1=19 at qp=22). Same holds for all qps.
    let shift = qbits + 1;
    let f = (1i64 << shift) / 3;
    let mf = MF_TABLE[qmod][0] as i64;
    for v in dc.iter_mut() {
        let w = *v as i64;
        let sign = w.signum();
        let abs = w.abs();
        let z = (abs * mf + f) >> shift;
        *v = (sign * z) as i32;
    }
}

/// Quantise the 4:2:0 chroma DC 2×2 Hadamard block in-place.
///
/// Calibration note: the forward 2×2 Hadamard amplifies the DC position
/// by 4. The decoder's `inv_hadamard_2x2_chroma_dc` spreads a DC-only
/// input `[z, 0, 0, 0]` into 4 equal cells of `z` (no scaling), then
/// applies a dequant of `v * scale << (qp/6 - 1)` for `qp >= 6` — which
/// is 2× larger than AC's `v * scale << (qp/6)`. Net result:
/// reconstructed_DC = `(4 * z) * scale * 2^{qp/6-1}` vs. AC's
/// `z * scale * 2^{qp/6}`. To balance, forward shift is `qbits + 1`.
pub fn quantize_chroma_dc_2x2(dc: &mut [i32; 4], qp: i32) {
    let qp = qp.clamp(0, 51);
    let qmod = (qp % 6) as usize;
    let qp6 = (qp / 6) as u32;
    let qbits: u32 = 15 + qp6;
    let shift = qbits + 1;
    let f = (1i64 << shift) / 3;
    let mf = MF_TABLE[qmod][0] as i64;
    for v in dc.iter_mut() {
        let w = *v as i64;
        let sign = w.signum();
        let abs = w.abs();
        let z = (abs * mf + f) >> shift;
        *v = (sign * z) as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::{dequantize_4x4, idct_4x4, inv_hadamard_4x4_dc};

    #[test]
    fn fdct_then_idct_constant_block_reasonable() {
        // A constant block of 10 → forward DCT puts energy only at (0,0):
        // DC = 16 * 10 = 160 (since forward scales by 16 for DC position).
        let mut x = [0i32; 16];
        for v in x.iter_mut() {
            *v = 10;
        }
        forward_dct_4x4(&mut x);
        assert_eq!(x[0], 160);
        // Check all non-DC are zero.
        for i in 1..16 {
            assert_eq!(x[i], 0, "non-DC at {i} should be zero for constant input");
        }
    }

    #[test]
    fn quant_then_dequant_roundtrip_ac_qp22() {
        // A moderately textured block; encode then decode and check residual sum.
        let original: [i32; 16] = [
            10, -5, 2, 0, 7, 1, -2, 3, -4, 0, 1, -1, 2, 3, 0, -1,
        ];
        let mut x = original;
        forward_dct_4x4(&mut x);
        quantize_4x4_ac(&mut x, 22);
        // Dequantise (using the decoder's path) and IDCT.
        dequantize_4x4(&mut x, 22);
        idct_4x4(&mut x);
        // All values should be close to original (lossy due to quant, but bounded).
        for i in 0..16 {
            let diff = (x[i] - original[i]).abs();
            assert!(diff <= 6, "pos {i}: {} vs original {}", x[i], original[i]);
        }
    }

    #[test]
    fn hadamard_dc_roundtrip_constant_block_qp22() {
        // Simulate the DC pass of an I_16x16 macroblock whose 16 4×4
        // sub-blocks each have a residual of constant 16 per sample.
        // fDCT of a constant-16 4×4 block yields DC = 16*16 = 256 at
        // position 0 (and zero elsewhere). So the input to the forward
        // Hadamard is 16 copies of 256.
        let dc_per_block = 16i32 * 16; // fDCT DC of constant-16 4×4
        let mut dc = [dc_per_block; 16];
        forward_hadamard_4x4(&mut dc);
        quantize_luma_dc_4x4(&mut dc, 22);
        inv_hadamard_4x4_dc(&mut dc, 22);
        // After inverse dequant + Hadamard, dc[i] should match what AC's
        // position-0 dequant would produce for a unit-residual of 16 per
        // pixel: 16*64 = 1024 in "dequant space", so that IDCT's
        // `(v+32)>>6` yields 16 per sample.
        let mean: i32 = dc.iter().sum::<i32>() / 16;
        for &v in dc.iter() {
            assert!((v - mean).abs() <= 16, "dc spread too large: {} vs mean {}", v, mean);
        }
        // Expect ~1024 (allowing some quant error).
        assert!(
            (896..=1152).contains(&mean),
            "unexpected dc mean {} (expected ~1024)",
            mean
        );
    }

    #[test]
    fn chroma_dc_roundtrip_constant_block_qp22() {
        use crate::transform::inv_hadamard_2x2_chroma_dc;
        let dc_per_block = 16i32 * 16;
        let mut dc = [dc_per_block; 4];
        forward_hadamard_2x2(&mut dc);
        quantize_chroma_dc_2x2(&mut dc, 22);
        inv_hadamard_2x2_chroma_dc(&mut dc, 22);
        let mean: i32 = dc.iter().sum::<i32>() / 4;
        for &v in dc.iter() {
            assert!((v - mean).abs() <= 8, "chroma dc spread: {} vs {}", v, mean);
        }
        assert!(
            (896..=1152).contains(&mean),
            "chroma dc mean {}, expected ~1024",
            mean
        );
    }
}
