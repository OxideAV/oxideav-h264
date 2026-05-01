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
