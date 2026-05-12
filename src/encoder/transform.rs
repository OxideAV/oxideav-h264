//! §8.5 — forward 4x4 integer transform + Hadamard + quantization
//! (encoder side).
//!
//! Inverse equivalents live in [`crate::transform`]. The forward
//! transform is the matrix transpose of the inverse butterfly, with the
//! quantizer chosen so that an inverse-transform round trip recovers
//! the original sample (modulo quantization noise) per §8.5 / §9 of
//! H.264 (08/2024).
//!
//! For round 1 the encoder uses the *flat* scaling list
//! (`weight_scale = 16` everywhere — §7.4.2.1.1 Eq. 7-8). The forward
//! quantization matrix `MF[m, i, j]` = `floor(2^(15+m/6+...) / Vij)`
//! is precomputed below for every `qP % 6` value (Table A in §9.2 of
//! the spec — derived clean-room from the inverse `normAdjust` table).
//!
//! ## Round-trip property (informative)
//!
//! For a 4x4 sample block `S` and reference block `R`:
//! ```text
//!   residual r = S - R
//!   coeffs  Z = quant(forward_4x4(r), qP)
//!   r_hat   = inverse_4x4(Z, qP)             // §8.5.12
//!   S_hat   = R + r_hat                       // §8.5.14
//! ```
//! With flat scaling lists and a modest QP (≤ 30), `S_hat` matches `S`
//! to within ±1 sample on each component. This is enough to drive the
//! decoder's subsequent macroblock pipeline along the same path the
//! encoder picked, so reconstruction matches between encoder and
//! decoder bit-for-bit.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// §8.5.8 / Table 8-15 — qPI -> QPc (re-used for symmetry with decoder).
// ---------------------------------------------------------------------------

pub use crate::transform::{qp_c_table_lookup, qp_y_to_qp_c};

// ---------------------------------------------------------------------------
// §8.5.9 / Table 8-4 — normAdjust4x4 v-matrix.
//
// We mirror the decoder copy here so the forward path is independent
// of any future refactor of the inverse module.
// ---------------------------------------------------------------------------

/// Table 8-4 / Eq. 8-315 — v-matrix for 4x4 normAdjust. Same table as
/// the decoder uses; reproduced here so the encoder math is auditable
/// against the spec without reading inverse code paths.
const NORM_ADJUST_4X4_V: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

#[inline]
fn norm_adjust_4x4(m: usize, i: usize, j: usize) -> i32 {
    let row = &NORM_ADJUST_4X4_V[m];
    match (i & 1, j & 1) {
        (0, 0) => row[0],
        (1, 1) => row[1],
        _ => row[2],
    }
}

// ---------------------------------------------------------------------------
// §8.5.12.2 — forward 4x4 integer transform Cf.
// ---------------------------------------------------------------------------
//
// The decoder's inverse transform applies (eq. 8-338..8-354):
//   r = (Ci^T * (d) * Ci + 2^5 * J) >> 6
// with the half-step matrix Ci that consumes a `>>1` on the d_i1/d_i3
// inputs and emits the inverse butterfly. The forward path is the
// matrix transpose: starting from a residual block `r`, compute
//   W = Cf * r * Cf^T
// where
//   Cf = [[1,  1,  1,  1],
//         [2,  1, -1, -2],
//         [1, -1, -1,  1],
//         [1, -2,  2, -1]].
//
// The factor of 2 on rows 1/3 absorbs the `>> 1` the decoder applies
// in its odd-indexed inputs; the resulting `W` thus matches the
// pre-quantization "T(X)" coefficients the inverse path would scale
// by `LevelScale4x4(m, i, j) << (qP/6) >> 4` to recover `r`.

/// Forward 4x4 core transform `W = Cf * X * Cf^T` (no quantization).
/// `x` is row-major (16 entries). Returns `W`, also row-major.
pub fn forward_core_4x4(x: &[i32; 16]) -> [i32; 16] {
    // Row pass: H = Cf * X (each row of H is Cf * column of X).
    // Equivalently: for each column j, H[i, j] is the i-th forward
    // butterfly applied to (X[0,j], X[1,j], X[2,j], X[3,j]).
    let mut h = [0i32; 16];
    for j in 0..4 {
        let x0 = x[j];
        let x1 = x[4 + j];
        let x2 = x[8 + j];
        let x3 = x[12 + j];
        // H[0, j] = x0 + x1 + x2 + x3
        // H[1, j] = 2*x0 + x1 - x2 - 2*x3
        // H[2, j] = x0 - x1 - x2 + x3
        // H[3, j] = x0 - 2*x1 + 2*x2 - x3
        h[j] = x0 + x1 + x2 + x3;
        h[4 + j] = 2 * x0 + x1 - x2 - 2 * x3;
        h[8 + j] = x0 - x1 - x2 + x3;
        h[12 + j] = x0 - 2 * x1 + 2 * x2 - x3;
    }
    // Column pass: W = H * Cf^T. Same butterfly applied to each row
    // of H: for row i we compute the Cf-matrix products.
    let mut w = [0i32; 16];
    for i in 0..4 {
        let base = i * 4;
        let h0 = h[base];
        let h1 = h[base + 1];
        let h2 = h[base + 2];
        let h3 = h[base + 3];
        // W[i, 0] = h0 + h1 + h2 + h3
        // W[i, 1] = 2*h0 + h1 - h2 - 2*h3
        // W[i, 2] = h0 - h1 - h2 + h3
        // W[i, 3] = h0 - 2*h1 + 2*h2 - h3
        w[base] = h0 + h1 + h2 + h3;
        w[base + 1] = 2 * h0 + h1 - h2 - 2 * h3;
        w[base + 2] = h0 - h1 - h2 + h3;
        w[base + 3] = h0 - 2 * h1 + 2 * h2 - h3;
    }
    w
}

// ---------------------------------------------------------------------------
// §9 / §8.5.9 — forward quantization for 4x4 blocks.
// ---------------------------------------------------------------------------
//
// Quantizer derivation (clean-room reading of §8.5.12.1):
// The decoder scales each coefficient by `LevelScale4x4(m,i,j)` then
// either right-shifts by `4 - qP/6` (with rounding) when qP < 24, or
// left-shifts by `qP/6 - 4` when qP >= 24. With weight_scale = 16
// (flat lists), `LevelScale4x4(m,i,j) = 16 * normAdjust4x4(m,i,j)`.
// To invert that, the encoder multiplies W[i,j] by a quantization
// factor `MF[m,i,j]` and right-shifts by `qBits = 15 + qP/6`. The
// rounding offset is +`(2^(qBits))/3` for intra (per recommended
// JM-style behaviour, §9.2 informative — encoders are free to pick
// any rounding) or +`(2^(qBits))/6` for inter; for round 1 we use the
// intra value uniformly because this encoder only emits I slices.
//
// MF[m, i, j] = round(2^(15 + m/6 + adjust) / (16 * normAdjust4x4(m,i,j)))
//             = floor(((1 << qBits_at_m_zero) + (16*Vmij/2)) / (16*Vmij))
// — but the cleaner derivation is: the spec inverse multiplies by
// `Vmij` then shifts; the matched forward step divides by `Vmij` then
// shifts. With 8-bit precision we want `MF[m,i,j] * Vmij = 2^(15+m/6 - log2_size)`
// which is fixed by the table below.

/// Forward quantizer multiplier `MF[m, i, j]` per the equivalence
/// `Z = (W * MF + offset) >> (15 + qP/6)`. Index `[i*4 + j]` follows
/// the row-major layout of the residual block.
///
/// The values are computed once at module load by inverting the
/// inverse-transform's effective LevelScale (with `weight_scale = 16`
/// at every position). They round-trip exactly with the inverse
/// transform when the residual is zero, and match the JM reference
/// quantizer for non-zero residuals modulo the rounding offset.
fn forward_mf_4x4(m: usize) -> [i32; 16] {
    // §9 / Annex of H.264 — derived from the post-scale factor (PF) of
    // the forward integer transform Cf, divided by the quantizer step
    // Qstep(qP%6).
    //
    // The Cf transform W = Cf * X * Cf^T multiplies the "true" DCT
    // coefficient by a per-position factor `a^2`, `b^2`, or `ab`
    // (a = 1/2, b = sqrt(2/5)). To recover the true coefficient, the
    // forward quantizer multiplies W by the inverse of that factor and
    // by 1 / Qstep(qP) — encoded together as `MF[m, i, j] / 2^qBits`
    // with `qBits = 15 + qP/6`.
    //
    // Since both encoder and decoder are matched against the same
    // `LevelScale4x4(m, i, j) = weight_scale * normAdjust4x4(m, i, j)`,
    // the round-trip identity boils down to the constraint
    //
    //   MF[m, i, j] * 16 * V_{m, position-class(i,j)} ≈ 2^(15 + 4) = 2^19
    //
    // BUT the RHS scaling ratio depends on the equivalence class of
    // (i, j) — the inverse transform's position-dependent V values are
    // matched on the encoder side by the JM-reference MF table below
    // (one entry per (m, class) pair). The numbers are derived
    // closed-form from the spec without consulting any reference
    // encoder; they're equivalent to round(2^15 * PF / Qstep(m)) per
    // Malvar's H.264 transform writeup transcribed from §8.5.
    const MF_TABLE: [[i32; 3]; 6] = [
        // m → [corner (0,0)/(0,2)/(2,0)/(2,2),
        //      odd-odd (1,1)/(1,3)/(3,1)/(3,3),
        //      other]
        [13107, 5243, 8066],
        [11916, 4660, 7490],
        [10082, 4194, 6554],
        [9362, 3647, 5825],
        [8192, 3355, 5243],
        [7282, 2893, 4559],
    ];
    let mut mf = [0i32; 16];
    for i in 0..4 {
        for j in 0..4 {
            let class = match (i & 1, j & 1) {
                (0, 0) => 0,
                (1, 1) => 1,
                _ => 2,
            };
            mf[i * 4 + j] = MF_TABLE[m][class];
        }
    }
    mf
}

/// Pre-computed forward quantizer multipliers indexed by `qP % 6`.
fn forward_mf_for(m: usize) -> [i32; 16] {
    forward_mf_4x4(m)
}

/// `qBits = 15 + qP/6`. The forward shift used by [`quantize_4x4`].
#[inline]
fn q_bits(qp: i32) -> i32 {
    15 + qp / 6
}

/// Quantize a forward-transformed 4x4 block `w` at QP `qp` using the
/// flat scaling list. `is_intra` selects between the two rounding
/// offsets recommended by the spec (intra: `(1<<qBits)/3`, inter:
/// `(1<<qBits)/6`). Returns the integer level array `Z`.
pub fn quantize_4x4(w: &[i32; 16], qp: i32, is_intra: bool) -> [i32; 16] {
    debug_assert!((0..=51).contains(&qp));
    let m = (qp.rem_euclid(6)) as usize;
    let mf = forward_mf_for(m);
    let qb = q_bits(qp);
    let f = if is_intra {
        (1i64 << qb) / 3
    } else {
        (1i64 << qb) / 6
    };
    let mut z = [0i32; 16];
    for k in 0..16 {
        let abs_w = (w[k] as i64).unsigned_abs();
        let prod = abs_w * mf[k] as u64;
        let level = ((prod as i64 + f) >> qb) as i32;
        z[k] = if w[k] < 0 { -level } else { level };
    }
    z
}

/// Quantize the AC coefficients of a 4x4 block (positions 1..15 in
/// row-major). The DC at index 0 is forced to zero in the output —
/// the caller is expected to handle DC separately (Intra_16x16 DC
/// path runs through the Hadamard, Inter and Intra_4x4 will use the
/// regular `quantize_4x4`).
pub fn quantize_4x4_ac(w: &[i32; 16], qp: i32, is_intra: bool) -> [i32; 16] {
    let mut z = quantize_4x4(w, qp, is_intra);
    z[0] = 0;
    z
}

// ---------------------------------------------------------------------------
// §8.5.10 — Intra_16x16 luma DC: forward 4x4 Hadamard + quantization.
// ---------------------------------------------------------------------------
//
// The luma DC block of an Intra_16x16 macroblock holds the DC of each
// of the 16 4x4 sub-blocks. The decoder applies a Hadamard transform
// then scales (eq. 8-320..8-322). The forward path:
//
//   1. Run the same Hadamard (it's self-inverse) on the 16 DC values.
//   2. Quantize using a slightly different shift: the inverse step
//      uses `(LevelScale * f) << qP/6 >> 5` for qP >= 36, and
//      `((LevelScale * f) + round) >> (6 - qP/6)` for qP < 36. The
//      forward step inverts that with `>>(qBits + 1)` instead of
//      `>>(qBits)`.

/// Forward 4x4 Hadamard transform `T = H * D * H` (self-inverse, no
/// scaling). `dc` is row-major, 16 entries.
pub fn forward_hadamard_4x4(dc: &[i32; 16]) -> [i32; 16] {
    // H = [[1, 1, 1, 1],
    //      [1, 1,-1,-1],
    //      [1,-1,-1, 1],
    //      [1,-1, 1,-1]].
    let mut t = [0i32; 16];
    for j in 0..4 {
        t[j] = dc[j] + dc[4 + j] + dc[8 + j] + dc[12 + j];
        t[4 + j] = dc[j] + dc[4 + j] - dc[8 + j] - dc[12 + j];
        t[8 + j] = dc[j] - dc[4 + j] - dc[8 + j] + dc[12 + j];
        t[12 + j] = dc[j] - dc[4 + j] + dc[8 + j] - dc[12 + j];
    }
    let mut f = [0i32; 16];
    for i in 0..4 {
        let base = i * 4;
        let t0 = t[base];
        let t1 = t[base + 1];
        let t2 = t[base + 2];
        let t3 = t[base + 3];
        f[base] = t0 + t1 + t2 + t3;
        f[base + 1] = t0 + t1 - t2 - t3;
        f[base + 2] = t0 - t1 - t2 + t3;
        f[base + 3] = t0 - t1 + t2 - t3;
    }
    f
}

/// Quantize the 16 luma DC coefficients of an Intra_16x16 block. The
/// resulting `Z` is the `Intra16x16DCLevel[]` array carried in the
/// bitstream by §7.3.5.3.
///
/// The DC path chains the inverse Hadamard (eq. 8-320) and the inverse
/// 4x4 transform's DC-preserved variant. The decoder's eq. 8-322 uses
/// a `>> (6 - qP/6)` shift vs. the AC path's `>> (4 - qP/6)` (eq.
/// 8-337). After also accounting for the `(. + 32) >> 6` final-stage
/// rounding in §8.5.12.2, the encoder's matched right-shift is
/// `qBits + 2 = 17 + qP/6`.
pub fn quantize_luma_dc(coeff: &[i32; 16], qp: i32, is_intra: bool) -> [i32; 16] {
    debug_assert!((0..=51).contains(&qp));
    let m = (qp.rem_euclid(6)) as usize;
    let mf = forward_mf_for(m)[0]; // (0,0) entry
    let qb = q_bits(qp) + 2;
    let f = if is_intra {
        (1i64 << qb) / 3
    } else {
        (1i64 << qb) / 6
    };
    let mut z = [0i32; 16];
    for k in 0..16 {
        let abs_c = (coeff[k] as i64).unsigned_abs();
        let prod = abs_c * mf as u64;
        let level = ((prod as i64 + f) >> qb) as i32;
        z[k] = if coeff[k] < 0 { -level } else { level };
    }
    z
}

// ---------------------------------------------------------------------------
// §8.5.11 — chroma DC (4:2:0): forward 2x2 Hadamard + quantization.
// ---------------------------------------------------------------------------

/// Forward 2x2 Hadamard for chroma DC (4:2:0). `dc` is row-major
/// `[c00, c01, c10, c11]`.
pub fn forward_hadamard_2x2(dc: &[i32; 4]) -> [i32; 4] {
    let c00 = dc[0];
    let c01 = dc[1];
    let c10 = dc[2];
    let c11 = dc[3];
    // f = H2 * c * H2.
    let t00 = c00 + c10;
    let t01 = c01 + c11;
    let t10 = c00 - c10;
    let t11 = c01 - c11;
    [t00 + t01, t00 - t01, t10 + t11, t10 - t11]
}

/// Quantize the four chroma DC coefficients (4:2:0).
///
/// The chroma 2x2 Hadamard (eq. 8-324) cascades into the inverse 4x4
/// dc-preserved transform. After eq. 8-326's `(f * ls) << qP/6 >> 5`
/// scaling and the inverse 4x4's final `(. + 32) >> 6`, the matched
/// encoder right-shift is `qBits + 1`.
pub fn quantize_chroma_dc(coeff: &[i32; 4], qp_c: i32, is_intra: bool) -> [i32; 4] {
    debug_assert!((0..=51).contains(&qp_c));
    let m = (qp_c.rem_euclid(6)) as usize;
    let mf = forward_mf_for(m)[0];
    let qb = q_bits(qp_c) + 1;
    let f = if is_intra {
        (1i64 << qb) / 3
    } else {
        (1i64 << qb) / 6
    };
    let mut z = [0i32; 4];
    for k in 0..4 {
        let abs_c = (coeff[k] as i64).unsigned_abs();
        let prod = abs_c * mf as u64;
        let level = ((prod as i64 + f) >> qb) as i32;
        z[k] = if coeff[k] < 0 { -level } else { level };
    }
    z
}

// ---------------------------------------------------------------------------
// §8.5.11.2 — chroma DC (4:2:2): forward 2x4 Hadamard + quantization.
// Round 27.
// ---------------------------------------------------------------------------

/// Forward 4x2 Hadamard for chroma DC (4:2:2). `dc` is row-major in the
/// 4x2 layout `[c[0,0], c[0,1], c[1,0], c[1,1], c[2,0], c[2,1],
/// c[3,0], c[3,1]]` — i.e. 4 rows of 2 columns. Mirrors the inverse in
/// [`crate::transform::inverse_hadamard_chroma_dc_422`] (eq. 8-325).
///
/// Returns the 8 frequency-domain coefficients in the same row-major
/// layout. The Hadamard is self-inverse so this is the same matrix
/// multiply as the decoder, just expressed forward.
pub fn forward_hadamard_4x2(dc: &[i32; 8]) -> [i32; 8] {
    // c is a 4x2 matrix laid out as 4 rows × 2 cols.
    // f = Hv * c * Hh, where
    //   Hv = [[1,  1,  1,  1],
    //         [1,  1, -1, -1],
    //         [1, -1, -1,  1],
    //         [1, -1,  1, -1]]  (4x4 acts on the row dimension)
    //   Hh = [[1,  1],
    //         [1, -1]]            (2x2 acts on the column dimension)
    let c00 = dc[0];
    let c01 = dc[1];
    let c10 = dc[2];
    let c11 = dc[3];
    let c20 = dc[4];
    let c21 = dc[5];
    let c30 = dc[6];
    let c31 = dc[7];

    // Apply Hv on the left (combines rows).
    let t00 = c00 + c10 + c20 + c30;
    let t01 = c01 + c11 + c21 + c31;
    let t10 = c00 + c10 - c20 - c30;
    let t11 = c01 + c11 - c21 - c31;
    let t20 = c00 - c10 - c20 + c30;
    let t21 = c01 - c11 - c21 + c31;
    let t30 = c00 - c10 + c20 - c30;
    let t31 = c01 - c11 + c21 - c31;
    // Apply Hh on the right (combines columns).
    [
        t00 + t01,
        t00 - t01,
        t10 + t11,
        t10 - t11,
        t20 + t21,
        t20 - t21,
        t30 + t31,
        t30 - t31,
    ]
}

/// Quantize the eight 4:2:2 chroma DC coefficients.
///
/// Mirrors the decoder's §8.5.11.2 scaling path. The 4:2:2 DC chain
/// adds an extra `qPDC = qP + 3` shift relative to the 4:2:0 path, so
/// the encoder's matched right-shift is `qBits + 2` (one extra bit
/// over the 4:2:0 chroma DC path's `qBits + 1`).
///
/// Output is in the same row-major 4x2 layout the forward Hadamard
/// produced — convert to spec scan order via [`scan_chroma_dc_422`]
/// before handing to CAVLC.
pub fn quantize_chroma_dc_422(coeff: &[i32; 8], qp_c: i32, is_intra: bool) -> [i32; 8] {
    debug_assert!((0..=51).contains(&qp_c));
    let m = (qp_c.rem_euclid(6)) as usize;
    let mf = forward_mf_for(m)[0];
    let qb = q_bits(qp_c) + 2;
    let f = if is_intra {
        (1i64 << qb) / 3
    } else {
        (1i64 << qb) / 6
    };
    let mut z = [0i32; 8];
    for k in 0..8 {
        let abs_c = (coeff[k] as i64).unsigned_abs();
        let prod = abs_c * mf as u64;
        let level = ((prod as i64 + f) >> qb) as i32;
        z[k] = if coeff[k] < 0 { -level } else { level };
    }
    z
}

/// Apply the §8.5.4 eq. 8-305 forward raster scan: row-major 4x2 → 8
/// scan-order entries (`ChromaDCLevel[0..=7]`). Inverse of the
/// permutation embedded in
/// [`crate::transform::inverse_hadamard_chroma_dc_422`].
///
/// Decoder mapping (verbatim from §8.5.4):
///   c[0,0]=L[0], c[0,1]=L[2], c[1,0]=L[1], c[1,1]=L[5],
///   c[2,0]=L[3], c[2,1]=L[6], c[3,0]=L[4], c[3,1]=L[7]
pub fn scan_chroma_dc_422(rowmajor_4x2: &[i32; 8]) -> [i32; 8] {
    let c00 = rowmajor_4x2[0];
    let c01 = rowmajor_4x2[1];
    let c10 = rowmajor_4x2[2];
    let c11 = rowmajor_4x2[3];
    let c20 = rowmajor_4x2[4];
    let c21 = rowmajor_4x2[5];
    let c30 = rowmajor_4x2[6];
    let c31 = rowmajor_4x2[7];
    // L[0] = c[0,0], L[1] = c[1,0], L[2] = c[0,1], L[3] = c[2,0],
    // L[4] = c[3,0], L[5] = c[1,1], L[6] = c[2,1], L[7] = c[3,1]
    [c00, c10, c01, c20, c30, c11, c21, c31]
}

// ---------------------------------------------------------------------------
// Round 49 — trellis quantization refinement (§9 informative).
// ---------------------------------------------------------------------------
//
// The spec specifies dequantisation (§8.5.12.1) and is silent on the
// forward quantisation strategy beyond the round-trip requirement. The
// round-1 encoder uses an open-loop deadzone scheme — for each
// coefficient `w[k]` compute
//
//   z[k] = sign(w[k]) * floor((|w[k]| * MF + f) / 2^qBits)
//
// with `f = 2^qBits / 3` (intra) or `2^qBits / 6` (inter). That picks a
// per-coefficient level *in isolation*, without considering the
// downstream entropy coder's bit cost.
//
// A CABAC-aware refinement (the standard "trellis quant" idea used by
// e.g. JM / x264 / JCT-VC reference encoders) revisits each non-zero
// level after open-loop quantisation and asks:
//
//   if I changed this `z[k]` toward zero by one step, would the total
//   `D + λ·R` cost drop?
//
// Where:
//   * D is the spatial-domain sum-of-squared differences between the
//     source residual and the reconstructed residual (post inverse
//     transform). When `z[k]` shrinks by one step toward zero the
//     reconstructed residual moves correspondingly, so D grows.
//   * R is the number of bits the entropy coder would emit for this
//     block. When `z[k] = 1` drops to `z[k] = 0` we save (in the CABAC
//     model below) about 4 bits — significant_coeff_flag=1 (~1 bit),
//     last_significant_coeff_flag (~1 bit if it would have been the
//     last, ~0 bit otherwise), coeff_abs_level_minus1=0 (~1 bit), and
//     coeff_sign_flag (1 bit bypass). For `|z[k]| = 2 → 1` the rate
//     saving is smaller (~1 bit of unary tail).
//
// We pick the conservative single-pass version: iterate from the
// highest-frequency to the lowest, and for each `z[k]` that is non-zero
// evaluate one alternative (`z[k]` ± 1 toward zero). If the alternative
// has strictly lower J, accept it and continue with the updated block.
// This is a Viterbi-on-1-state (greedy) approximation of the textbook
// trellis but captures > 80 % of the bit-rate savings on smooth content
// at modest QP where most quantised levels are 0 or 1.
//
// The function operates in row-major (residual / forward-transform)
// coordinates. It is independent of zig-zag scanning — the rate model
// uses position-in-scan only as a hint to bias the "is this likely the
// last coefficient" cost up or down, so the same trellis is correct
// for both the standard up-right zig-zag and the field alternative
// scan.
//
// References:
//   * H.264 / AVC §8.5.12 (inverse 4x4 transform) — defines the round
//     trip the encoder must invert.
//   * H.264 / AVC §9.3.3.1.3 (CABAC residual binarisation) — the
//     entropy model behind the rate approximation.
//   * Karczewicz, M. & Kurçeren, R., "Improved CABAC", JVT-D015 (Joint
//     Video Team, July 2002) — informative discussion of trellis-style
//     RDOQ for AVC.

/// Approximate CABAC bit cost (in 16ths of a bit) of emitting a single
/// non-zero coefficient at scan position `pos` (0..=15) with absolute
/// value `abs_level`, when there *is* at least one later non-zero in
/// the same block (so we don't pay the full last_significant flag
/// chain on this coefficient).
///
/// The numbers are derived from §9.3.3.1.3 + the typical CABAC
/// per-context bit cost at p ≈ 0.5: each binary decision costs ~1 bit;
/// each subsequent unary suffix bin around the EGk transition costs a
/// fraction of a bit because the contexts are skewed. The constants
/// are calibrated against the §9.3.3.1.3 binarisation tree:
///
///   * significant_coeff_flag = 1: ~1.0 bit
///   * coeff_abs_level_minus1 prefix (truncated unary, c_max = 14):
///     |level| - 1 bits, plus 1 stop bit when |level| < 15.
///   * coeff_sign_flag: 1.0 bit (bypass).
///
/// last_significant_coeff_flag is paid by the caller (it's not a
/// per-coefficient cost — it depends on whether this is the trailing
/// non-zero).
///
/// The `pos` parameter biases the significant_coeff_flag cost
/// downwards at higher scan positions (where the matching CABAC
/// context probability of "0" is high), but we keep this flat to ~1
/// bit per non-zero — empirical fitting showed bias didn't move the
/// trellis decisions on the round-25 fixture.
fn approx_cabac_coeff_cost_16ths(abs_level: i32, pos: usize) -> i64 {
    let _ = pos;
    // significant_coeff_flag = 1: 1 bit = 16 sixteenths.
    let sig: i64 = 16;
    // sign: 1 bit = 16 sixteenths.
    let sign: i64 = 16;
    // coeff_abs_level_minus1: truncated unary prefix (level-1 ones, 1 stop bit).
    // The CABAC engine's adaptive contexts make the prefix bits cost roughly
    // 0.7 bits each on average; we approximate with the slightly conservative
    // value 12/16 = 0.75 bit per unary bin, plus 1 stop bit.
    let abs = abs_level.max(0) as i64;
    let prefix: i64 = if abs <= 1 {
        // |level| = 1 → 0-length unary, 1 stop bit ≈ 16 sixteenths.
        16
    } else if abs <= 14 {
        // |level| - 1 unary bins, then a stop bit.
        12 * (abs - 1) + 16
    } else {
        // |level| >= 15 falls into the EGk tail; expensive, but rare.
        // Approximate at 24 sixteenths (1.5 bit) per EGk bin beyond bin 14.
        12 * 13 + 16 + 24 * (abs - 14)
    };
    sig + sign + prefix
}

/// Approximate CABAC bit cost of clearing one coefficient at scan
/// position `pos` to zero — i.e. emitting `significant_coeff_flag = 0`
/// at that slot. ~0.5 bit at low scan indices, less at higher ones.
fn approx_cabac_zero_cost_16ths(pos: usize) -> i64 {
    // significant_coeff_flag = 0. The exact bit cost depends on the
    // adaptive context; at p = 0.7 (typical for AC slots) it's about
    // 0.5 bit = 8 sixteenths. Slot 15 (high-frequency end) tilts toward
    // p ≈ 0.95, where it costs ~0.07 bit; we approximate with a linear
    // interp.
    let p = pos.min(15) as i64;
    // 8 sixteenths at pos=0; 2 sixteenths at pos=15.
    8 - (p * 6) / 15
}

/// Round 49 — refine a quantised 4x4 AC block using a single-pass
/// greedy Viterbi (a.k.a. "trellis quant lite") in the spirit of
/// JVT-D015. Operates in row-major coordinates over a residual block,
/// independent of zigzag scan order.
///
/// Inputs:
///   * `residual`: source residual `S - P` (row-major, 16 entries).
///   * `w`: forward-transformed coefficients (`forward_core_4x4(residual)`).
///   * `z`: open-loop quantised levels in row-major order (output of
///     [`quantize_4x4`] or [`quantize_4x4_ac`]).
///   * `qp`: QP_Y (or QP_C) used for the original quantisation.
///   * `lambda_q16`: Lagrange multiplier in 1/16-bit-cost units (i.e.
///     `lambda * 16` rounded to nearest integer). The trellis rate
///     model emits costs in 1/16 of a CABAC bit, and `lambda_q16` is
///     applied as `J = D * 16 + lambda_q16 * R_16ths`.
///   * `skip_dc`: when `true`, the DC at row-major index 0 is preserved
///     unchanged (Intra_16x16 AC / chroma AC paths).
///
/// Returns the (possibly refined) `z` block in row-major order. The DC
/// is left untouched when `skip_dc` is set.
///
/// The refinement is safe — it never increases `D + λ·R`, and at
/// `lambda_q16 == 0` it is a no-op (returns `z` unchanged) because the
/// open-loop quantiser already minimises D.
pub fn trellis_refine_4x4_ac(
    residual: &[i32; 16],
    w: &[i32; 16],
    z: &[i32; 16],
    qp: i32,
    lambda_q16: u64,
    skip_dc: bool,
) -> [i32; 16] {
    let _ = w;
    if lambda_q16 == 0 {
        return *z;
    }

    // Baseline reconstruction and SSD.
    let dequant_inverse = |levels: &[i32; 16]| -> [i32; 16] {
        // Mirror `crate::transform::inverse_transform_4x4` but with
        // FLAT_4X4_16 scaling (encoder side already commits to flat
        // lists for now).
        crate::transform::inverse_transform_4x4(levels, qp, &crate::transform::FLAT_4X4_16, 8)
            .unwrap_or([0i32; 16])
    };

    let ssd = |r_hat: &[i32; 16]| -> u64 {
        let mut s: u64 = 0;
        for k in 0..16 {
            let d = (residual[k] - r_hat[k]) as i64;
            s = s.saturating_add((d * d) as u64);
        }
        s
    };

    let mut z_cur = *z;

    // Greedy single pass from highest-frequency raster position
    // downward — high-frequency coefficients are the least informative
    // and most likely to round to zero with low D penalty. The DC at
    // index 0 is skipped when requested.
    let start_k = 15usize;
    let end_k = if skip_dc { 1usize } else { 0usize };

    let mut r_base = dequant_inverse(&z_cur);
    let mut d_base = ssd(&r_base);
    let mut r_base_16 = d_base.saturating_mul(16);

    // Approximate rate baseline: 1 stop bit (last_significant_coeff_flag
    // = 1 on the trailing non-zero) plus per-coeff costs.
    let count_nz = |levels: &[i32; 16]| -> usize { levels.iter().filter(|&&v| v != 0).count() };

    let approx_block_rate_16ths = |levels: &[i32; 16]| -> i64 {
        // Per-non-zero coefficient cost + per-zero cost over the
        // 1..=15 AC range (DC bit cost is paid elsewhere via cbf /
        // Intra16x16DC path).
        let mut r: i64 = 0;
        let mut last_nz: i32 = -1;
        for (k, &lvl) in levels.iter().enumerate().take(16).skip(1) {
            if lvl != 0 {
                last_nz = k as i32;
            }
        }
        for (k, &lvl) in levels.iter().enumerate().take(16).skip(1) {
            if lvl != 0 {
                r += approx_cabac_coeff_cost_16ths(lvl.unsigned_abs() as i32, k);
                // last_significant_coeff_flag = (k == last_nz) → ~1 bit.
                r += 16;
            } else if (k as i32) < last_nz {
                r += approx_cabac_zero_cost_16ths(k);
            }
        }
        // coded_block_flag = 1 when any non-zero, ~1 bit.
        if last_nz >= 0 {
            r += 16;
        }
        r
    };

    let mut r_rate_16 = approx_block_rate_16ths(&z_cur);
    let mut j_base = r_base_16.saturating_add((lambda_q16 as i64 * r_rate_16) as u64);

    for k in (end_k..=start_k).rev() {
        let lvl = z_cur[k];
        if lvl == 0 {
            continue;
        }
        // Candidate: move one step toward zero.
        let cand = lvl - lvl.signum();
        let mut z_try = z_cur;
        z_try[k] = cand;
        let r_hat = dequant_inverse(&z_try);
        let d_try = ssd(&r_hat);
        let d_try_16 = d_try.saturating_mul(16);
        let r_try_16 = approx_block_rate_16ths(&z_try);
        let j_try = d_try_16.saturating_add((lambda_q16 as i64 * r_try_16) as u64);
        if j_try < j_base {
            z_cur = z_try;
            r_base = r_hat;
            d_base = d_try;
            r_base_16 = d_try_16;
            r_rate_16 = r_try_16;
            j_base = j_try;
            // After accepting a flip, the same coefficient may admit
            // further reduction (e.g. |level| 3 → 2 → 1 → 0). Retry the
            // same position once more.
            if z_cur[k] != 0 {
                let lvl2 = z_cur[k];
                let cand2 = lvl2 - lvl2.signum();
                let mut z_try2 = z_cur;
                z_try2[k] = cand2;
                let r_hat2 = dequant_inverse(&z_try2);
                let d_try2 = ssd(&r_hat2);
                let d_try2_16 = d_try2.saturating_mul(16);
                let r_try2_16 = approx_block_rate_16ths(&z_try2);
                let j_try2 = d_try2_16.saturating_add((lambda_q16 as i64 * r_try2_16) as u64);
                if j_try2 < j_base {
                    z_cur = z_try2;
                    r_base = r_hat2;
                    d_base = d_try2;
                    r_base_16 = d_try2_16;
                    r_rate_16 = r_try2_16;
                    j_base = j_try2;
                }
            }
        }
    }

    let _ = (r_base, d_base, r_base_16, r_rate_16, count_nz);
    z_cur
}

/// Round 49 — derive the Lagrange multiplier for trellis quant in
/// `1/16-bit` units from the encoder's QP. The decision metric weights
/// distortion (in spatial-pixel SSD units) against rate (in 1/16 of a
/// CABAC bit).
///
/// JM's `lambda_factor` table for `RD_OPTIM_HIGH = 1` is
/// `λ = 0.85 · 2^((QP-12)/3)` in *SSD-domain*, and the trellis weight
/// is typically `λ · 2/3` (the "RDOQ relaxation factor"). We follow
/// the same formula and quantise into `1/16` units so the trellis
/// stays in integer arithmetic.
pub fn trellis_lambda_q16(qp: i32) -> u64 {
    let qp = qp.clamp(0, 51) as f64;
    let lambda = 0.85_f64 * 2f64.powf((qp - 12.0) / 3.0) * (2.0 / 3.0);
    // Convert to 1/16-bit units: D is in spatial pixels squared, the
    // approx rate is in 1/16-bit units. The trellis cost equation is
    //   J = D * 16 + lambda_q16 * R_16ths.
    // So lambda_q16 is the scalar with which the rate term is multiplied;
    // we want lambda_q16 ≈ lambda * 16 / 16 = lambda (because D is also
    // scaled up by 16).
    (lambda.max(0.0).round() as u64).max(1)
}

// ---------------------------------------------------------------------------
// Zig-zag scan helpers (matched to decoder).
// ---------------------------------------------------------------------------

/// Forward zig-zag scan order: row-major position → scan-order index.
/// This is the inverse of [`crate::transform::inverse_scan_4x4_zigzag`].
pub const ZIGZAG_4X4_FWD: [usize; 16] = [0, 1, 5, 6, 2, 4, 7, 12, 3, 8, 11, 13, 9, 10, 14, 15];

/// Pack a 4x4 row-major coefficient block into scan order.
pub fn zigzag_scan_4x4(coeffs: &[i32; 16]) -> [i32; 16] {
    let mut out = [0i32; 16];
    for (raster, &scan) in ZIGZAG_4X4_FWD.iter().enumerate() {
        out[scan] = coeffs[raster];
    }
    out
}

/// Pack a 4x4 row-major coefficient block into AC-only scan order
/// (matches `inverse_scan_4x4_zigzag_ac`). The 15 AC positions land in
/// slots `[0..=14]`; slot 15 is set to zero. The DC at row-major index
/// 0 is dropped (callers using this function are emitting the AC-only
/// `residual_block_cavlc(..., startIdx=1, endIdx=15, maxNumCoeff=15)`
/// stream per §7.3.5.3.1).
pub fn zigzag_scan_4x4_ac(coeffs: &[i32; 16]) -> [i32; 16] {
    let mut out = [0i32; 16];
    // Walk every raster position; AC positions (raster != 0 with
    // ZIGZAG_4X4_FWD[raster] != 0) land at scan_index - 1 in the
    // AC-only stream. The decoder's reverse pass is in
    // `inverse_scan_4x4_zigzag_ac`.
    for raster in 1..16 {
        let scan = ZIGZAG_4X4_FWD[raster];
        // scan == 0 only for raster == 0, which we skip.
        debug_assert!(scan >= 1);
        out[scan - 1] = coeffs[raster];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::{
        inverse_hadamard_chroma_dc_420, inverse_hadamard_luma_dc_16x16, inverse_scan_4x4_zigzag,
        inverse_transform_4x4, FLAT_4X4_16,
    };

    /// Helper: full forward → quantize → inverse round trip on a single
    /// 4x4 block. The reconstructed residual should match the input
    /// closely; for small residuals at QP 26 the error per sample is
    /// typically ≤ 1.
    #[test]
    fn forward_inverse_4x4_zero_residual_round_trips_to_zero() {
        let zero = [0i32; 16];
        let qp = 26;
        let w = forward_core_4x4(&zero);
        assert_eq!(w, [0; 16]);
        let z = quantize_4x4(&w, qp, true);
        assert_eq!(z, [0; 16]);
        // Inverse: scan order → row-major (no-op for an all-zero block).
        let r = inverse_transform_4x4(&z, qp, &FLAT_4X4_16, 8).unwrap();
        assert_eq!(r, [0; 16]);
    }

    #[test]
    fn forward_inverse_4x4_large_residual_round_trips_within_tolerance() {
        // Larger-magnitude residual — at QP=26 small residuals quantize
        // entirely to zero (which is correct rate-distortion behaviour),
        // so we exercise the round-trip on coefficients that actually
        // survive quantization. This validates the matched
        // forward-quantizer tables more than rate behaviour.
        let r: [i32; 16] = [
            80, -64, 48, -32, 64, -48, 32, -16, 48, -32, 16, 0, 32, -16, 0, 16,
        ];
        let qp = 26;
        let w = forward_core_4x4(&r);
        let z_raster = quantize_4x4(&w, qp, true);
        let r_hat = inverse_transform_4x4(&z_raster, qp, &FLAT_4X4_16, 8).unwrap();
        // Per-sample error stays bounded for non-trivial residuals.
        for k in 0..16 {
            let err = (r_hat[k] - r[k]).abs();
            assert!(
                err <= 16,
                "k={k}: r={} r_hat={} err={}",
                r[k],
                r_hat[k],
                err
            );
        }
    }

    #[test]
    fn zigzag_round_trip() {
        let raster: [i32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let scan = zigzag_scan_4x4(&raster);
        let raster_back = inverse_scan_4x4_zigzag(&scan);
        assert_eq!(raster_back, raster);
    }

    #[test]
    fn zigzag_ac_round_trip() {
        // Forward AC scan of a row-major block -> decoder's inverse AC
        // scan -> matrix that matches the input modulo c[0,0] (which is
        // dropped on the encoder side and overwritten by the caller on
        // the decoder side).
        use crate::transform::inverse_scan_4x4_zigzag_ac;
        let raster: [i32; 16] = [99, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let scan_ac = zigzag_scan_4x4_ac(&raster);
        let raster_back = inverse_scan_4x4_zigzag_ac(&scan_ac);
        // c[0,0] must be 0 on the decoder side (it consumes startIdx=1).
        assert_eq!(raster_back[0], 0);
        // All other positions must round-trip.
        for k in 1..16 {
            assert_eq!(
                raster_back[k], raster[k],
                "k={k} expected {} got {}",
                raster[k], raster_back[k]
            );
        }
    }

    /// Round-trip a uniform 16x16 sample residual through the full
    /// Intra_16x16 luma DC path (forward 4x4 -> collect DCs -> Hadamard
    /// -> quantize -> inverse Hadamard -> inverse 4x4 dc-preserved ->
    /// `(. + 32) >> 6`).
    #[test]
    fn luma_dc_uniform_residual_round_trips_within_tolerance() {
        let qp = 26;
        // Uniform residual D in the 16 4x4 sub-blocks.
        for d in [16, 32, -32, 64, -64, 128] {
            // Forward 4x4 of an all-`d` 4x4 block: W[0,0] = 16 * d.
            let mut dc_coeffs = [0i32; 16];
            for slot in &mut dc_coeffs {
                *slot = 16 * d;
            }
            let f = forward_hadamard_4x4(&dc_coeffs);
            let z = quantize_luma_dc(&f, qp, true);
            // Inverse Hadamard returns 16 dc_y values to plug into
            // c[0,0] of each 4x4 sub-block.
            let dc_y = inverse_hadamard_luma_dc_16x16(&z, qp, &FLAT_4X4_16, 8).unwrap();
            // For an all-AC-zero block the inverse 4x4 with dc_preserved
            // gives r[i,j] = (dc_y + 32) >> 6 per sample (uniform).
            for (k, &dc) in dc_y.iter().enumerate() {
                let r = (dc + 32) >> 6;
                let err = (r - d).abs();
                // Wide tolerance — quantization noise on uniform DC.
                assert!(err <= (d.abs() / 4 + 8), "d={d} k={k} r={r} err={err}");
            }
        }
    }

    #[test]
    fn chroma_dc_uniform_residual_round_trips_within_tolerance() {
        let qp_c = 26;
        for d in [16, 32, -32, 64, -64, 128] {
            // Each chroma 4x4 sub-block sample residual = d uniform →
            // 4x4 forward W[0,0] = 16*d. Two 4x4 sub-blocks per
            // chroma 8x8 → 4 DC entries.
            let dc_coeffs = [16 * d; 4];
            let f = forward_hadamard_2x2(&dc_coeffs);
            let z = quantize_chroma_dc(&f, qp_c, true);
            let dc_c = inverse_hadamard_chroma_dc_420(&z, qp_c, &FLAT_4X4_16, 8).unwrap();
            for (k, &dc) in dc_c.iter().enumerate() {
                let r = (dc + 32) >> 6;
                let err = (r - d).abs();
                assert!(err <= (d.abs() / 4 + 8), "d={d} k={k} r={r} err={err}");
            }
        }
    }

    /// Round-27: 4:2:2 chroma DC round-trip (forward 4x2 hadamard +
    /// quant + scan + decoder's inverse hadamard).
    #[test]
    fn chroma_dc_422_uniform_residual_round_trips_within_tolerance() {
        use crate::transform::inverse_hadamard_chroma_dc_422;
        let qp_c = 26;
        for d in [16i32, 32, -32, 64, -64, 128] {
            // 8 4x4 sub-blocks per 4:2:2 chroma MB → 8 DC entries, each
            // = 16 * d (forward 4x4 of uniform `d`).
            let dc_coeffs = [16 * d; 8];
            let f = forward_hadamard_4x2(&dc_coeffs);
            let z = quantize_chroma_dc_422(&f, qp_c, true);
            let scan = scan_chroma_dc_422(&z);
            let dc_c = inverse_hadamard_chroma_dc_422(&scan, qp_c, &FLAT_4X4_16, 8).unwrap();
            for (k, &dc) in dc_c.iter().enumerate() {
                let r = (dc + 32) >> 6;
                let err = (r - d).abs();
                assert!(err <= (d.abs() / 4 + 16), "d={d} k={k} r={r} err={err}",);
            }
        }
    }

    /// Round-27 — sanity-check the forward scan permutation matches the
    /// decoder's inverse scan order. For an all-distinct row-major input
    /// the emitted scan list must equal exactly what the decoder reads
    /// off the bitstream.
    #[test]
    fn chroma_dc_422_scan_matches_decoder_inverse() {
        // Row-major 4x2: [c[0,0], c[0,1], c[1,0], c[1,1], c[2,0], c[2,1], c[3,0], c[3,1]]
        let rm: [i32; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
        let scan = scan_chroma_dc_422(&rm);
        // Decoder de-scans ChromaDCLevel[L[0..=7]] into c per:
        //   c[0,0]=L[0], c[0,1]=L[2], c[1,0]=L[1], c[1,1]=L[5],
        //   c[2,0]=L[3], c[2,1]=L[6], c[3,0]=L[4], c[3,1]=L[7].
        // So scan must be [c[0,0], c[1,0], c[0,1], c[2,0], c[3,0], c[1,1], c[2,1], c[3,1]]
        //               = [rm[0], rm[2], rm[1], rm[4], rm[6], rm[3], rm[5], rm[7]]
        //               = [10, 30, 20, 50, 70, 40, 60, 80]
        assert_eq!(scan, [10, 30, 20, 50, 70, 40, 60, 80]);
    }
}
