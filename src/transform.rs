//! §8.5 — Inverse transform and scaling.
//!
//! Pure integer math per ITU-T Rec. H.264 (08/2024). No bitstream, no
//! context. Takes parsed coefficient arrays and a quantiser parameter,
//! produces residual sample arrays for the reconstruction pipeline.
//!
//! Spec coverage:
//!   §8.5.6  — Inverse scanning process for 4x4 transform coefficients.
//!             Exposed separately as [`inverse_scan_4x4_zigzag`] /
//!             [`inverse_scan_4x4_field`].
//!   §8.5.8  — Derivation process for chroma quantization parameters
//!             (QPc via Table 8-15).
//!   §8.5.9  — Derivation process for scaling functions (LevelScale).
//!   §8.5.10 — Scaling + Hadamard for Intra_16x16 luma DC block.
//!   §8.5.11 — Scaling + Hadamard for chroma DC (2x2 for 4:2:0,
//!             2x4 for 4:2:2).
//!   §8.5.12 — Scaling + inverse transform for residual 4x4 blocks.
//!   §8.5.13 — Scaling + inverse transform for residual 8x8 blocks.
//!
//! All equation numbers below are from the 08/2024 edition.

#![allow(dead_code)]

use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors.
// ---------------------------------------------------------------------------

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TransformError {
    /// qP outside the valid 0..=51 range for 8-bit coding.
    #[error("QP out of range (got {0}, must be 0..=51)")]
    QpOutOfRange(i32),
    /// bit_depth outside the supported 8..=14 range.
    #[error("bit_depth out of range (got {0}, must be 8..=14)")]
    BitDepthOutOfRange(u32),
}

// ---------------------------------------------------------------------------
// §7.4.5 / Table 8-15 — QPy -> QPc mapping.
// ---------------------------------------------------------------------------

/// §8.5.8 / Table 8-15 — maps qPI (any value) to QPc.
/// For qPI < 30, the mapping is identity; for qPI ∈ [30,51] a table
/// lookup is used. Negative qPI values are returned unchanged because
/// the encoder is allowed to take qPI below zero (QpBdOffset range).
/// The caller is expected to have performed any required clipping.
pub fn qp_c_table_lookup(qp_i: i32) -> i32 {
    // Table 8-15 — QPc as a function of qPI, qPI ∈ [30,51].
    // Row index = qPI - 30.
    const QP_C_FROM_QPI_30_51: [i32; 22] = [
        29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39,
    ];
    if qp_i < 30 {
        qp_i
    } else if qp_i <= 51 {
        QP_C_FROM_QPI_30_51[(qp_i - 30) as usize]
    } else {
        // Out of normal bitstream range — clamp to the 51 entry.
        QP_C_FROM_QPI_30_51[21]
    }
}

/// §8.5.8 — full chroma QP derivation (eq. 8-311, 8-312 simplified for
/// BitDepth=8 where QpBdOffsetC is 0). For bit depths > 8 the caller
/// is expected to apply any QpBdOffset externally; this helper handles
/// the typical 8-bit path.
///
/// `qp_y` is the luma QP (post-adjustment) and `chroma_qp_index_offset`
/// comes from the PPS (`chroma_qp_index_offset` for Cb,
/// `second_chroma_qp_index_offset` for Cr). Returns QPc ∈ 0..=51.
pub fn qp_y_to_qp_c(qp_y: i32, chroma_qp_index_offset: i32) -> i32 {
    // §8.5.8 eq. 8-311: qPI = Clip3(-QpBdOffsetC, 51, QPY + qPOffset).
    // With QpBdOffsetC = 0 (8-bit chroma) the lower bound is 0.
    let qp_i = (qp_y + chroma_qp_index_offset).clamp(0, 51);
    qp_c_table_lookup(qp_i)
}

// ---------------------------------------------------------------------------
// §8.5.9 — normAdjust matrices (Tables 8-4, 8-5 transcribed verbatim).
// ---------------------------------------------------------------------------

/// Table 8-4 / Eq. 8-315 — v-matrix for 4x4 normAdjust.
///
/// Rows are indexed by m ∈ 0..=5 (= qP % 6), columns are the three
/// equivalence-class values selected by (i, j) parity per Eq. 8-314.
const NORM_ADJUST_4X4_V: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

/// Table 8-5 / Eq. 8-318 — v-matrix for 8x8 normAdjust.
const NORM_ADJUST_8X8_V: [[i32; 6]; 6] = [
    [20, 18, 32, 19, 25, 24],
    [22, 19, 35, 21, 28, 26],
    [26, 23, 42, 24, 33, 31],
    [28, 25, 45, 26, 35, 33],
    [32, 28, 51, 30, 40, 38],
    [36, 32, 58, 34, 46, 43],
];

/// §8.5.9 / Eq. 8-314 — normAdjust for 4x4 blocks.
///
/// Equivalence class selection:
///   (i%2, j%2) == (0, 0) -> v_{m,0}
///   (i%2, j%2) == (1, 1) -> v_{m,1}
///   otherwise            -> v_{m,2}
#[inline]
fn norm_adjust_4x4(m: usize, i: usize, j: usize) -> i32 {
    let row = &NORM_ADJUST_4X4_V[m];
    match (i & 1, j & 1) {
        (0, 0) => row[0],
        (1, 1) => row[1],
        _ => row[2],
    }
}

/// §8.5.9 / Eq. 8-317 — normAdjust for 8x8 blocks.
///
/// Equivalence class selection (in the order listed in the spec):
///   (i%4, j%4) == (0, 0)                      -> v_{m,0}
///   (i%2, j%2) == (1, 1)                      -> v_{m,1}
///   (i%4, j%4) == (2, 2)                      -> v_{m,2}
///   (i%4, j%2) == (0, 1) or (i%2, j%4) == (1, 0) -> v_{m,3}
///   (i%4, j%4) == (0, 2) or (i%4, j%4) == (2, 0) -> v_{m,4}
///   otherwise                                 -> v_{m,5}
#[inline]
fn norm_adjust_8x8(m: usize, i: usize, j: usize) -> i32 {
    let row = &NORM_ADJUST_8X8_V[m];
    let im4 = i & 3;
    let jm4 = j & 3;
    let im2 = i & 1;
    let jm2 = j & 1;
    if im4 == 0 && jm4 == 0 {
        row[0]
    } else if im2 == 1 && jm2 == 1 {
        row[1]
    } else if im4 == 2 && jm4 == 2 {
        row[2]
    } else if (im4 == 0 && jm2 == 1) || (im2 == 1 && jm4 == 0) {
        row[3]
    } else if (im4 == 0 && jm4 == 2) || (im4 == 2 && jm4 == 0) {
        row[4]
    } else {
        row[5]
    }
}

/// §8.5.9 Eq. 8-313 — LevelScale4x4(m, i, j) = weightScale4x4[i,j] *
/// normAdjust4x4(m, i, j).
#[inline]
fn level_scale_4x4(scaling_list: &[i32; 16], m: usize, i: usize, j: usize) -> i32 {
    scaling_list[i * 4 + j] * norm_adjust_4x4(m, i, j)
}

/// §8.5.9 Eq. 8-316 — LevelScale8x8(m, i, j) = weightScale8x8[i,j] *
/// normAdjust8x8(m, i, j).
#[inline]
fn level_scale_8x8(scaling_list: &[i32; 64], m: usize, i: usize, j: usize) -> i32 {
    scaling_list[i * 8 + j] * norm_adjust_8x8(m, i, j)
}

// ---------------------------------------------------------------------------
// §6.4 / §7.4.2.1.1.1 — default flat scaling lists.
// ---------------------------------------------------------------------------

/// Flat 4x4 scaling list. Used when scaling_matrix is not signalled
/// and "no scaling" applies per §7.4.2.1.1.1 (every entry set to 16
/// cancels out with the normAdjust / 16 factor implicit in the
/// derivation of the LevelScale tables).
pub fn default_scaling_list_4x4_flat() -> [i32; 16] {
    [16; 16]
}

/// Flat 8x8 scaling list.
pub fn default_scaling_list_8x8_flat() -> [i32; 64] {
    [16; 64]
}

// ---------------------------------------------------------------------------
// §8.5.6 — inverse zig-zag scanning for 4x4 transform coefficients.
//
// Table 8-13 row "zig-zag": idx -> c_ij. This utility helps callers that
// receive coefficients already in scan order.
// ---------------------------------------------------------------------------

/// Table 8-13, zig-zag row. `scan[k] = (i, j)` where k is the scan
/// index and (i, j) is the 2-D position in the 4x4 coefficient block.
const ZIGZAG_4X4: [(usize, usize); 16] = [
    (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
    (2, 1), (3, 0), (3, 1), (2, 2), (1, 3), (2, 3), (3, 2), (3, 3),
];

/// Table 8-13, field scan row.
const FIELD_4X4: [(usize, usize); 16] = [
    (0, 0), (1, 0), (0, 1), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1),
    (0, 2), (1, 2), (2, 2), (3, 2), (0, 3), (1, 3), (2, 3), (3, 3),
];

/// §8.5.6 — invert zig-zag scan; transforms a 16-entry scan-order list
/// into a 4x4 row-major coefficient matrix.
pub fn inverse_scan_4x4_zigzag(levels: &[i32; 16]) -> [i32; 16] {
    let mut out = [0i32; 16];
    for (k, &(i, j)) in ZIGZAG_4X4.iter().enumerate() {
        out[i * 4 + j] = levels[k];
    }
    out
}

/// §8.5.6 — invert the informative field scan for 4x4 blocks.
pub fn inverse_scan_4x4_field(levels: &[i32; 16]) -> [i32; 16] {
    let mut out = [0i32; 16];
    for (k, &(i, j)) in FIELD_4X4.iter().enumerate() {
        out[i * 4 + j] = levels[k];
    }
    out
}

// ---------------------------------------------------------------------------
// §8.5.12.1 — scaling for 4x4 residual blocks.
//
// Used internally by [`inverse_transform_4x4`]; exposed for callers that
// need to run scaling and transform as separate steps.
// ---------------------------------------------------------------------------

/// §8.5.12.1 — apply the LevelScale4x4 scaling to a 4x4 coefficient
/// block, producing the `d_ij` array ready for the inverse transform.
///
/// Equations 8-335..8-337. Does not touch c[0,0] for Intra_16x16 or
/// chroma residual blocks (the caller provides the already-scaled DC
/// value via the `is_dc_preserved` flag — true = leave d[0,0] as-is,
/// false = normal scaling).
fn scale_4x4(
    coeffs: &[i32; 16],
    qp: i32,
    scaling_list: &[i32; 16],
    is_dc_preserved: bool,
) -> [i32; 16] {
    // §8.5.9 — qP/6 and qP%6 control the log2 and modular parts of
    // the scaling factor.
    let qp_div = qp / 6;
    let qp_mod = (qp % 6) as usize;

    let mut d = [0i32; 16];
    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            if is_dc_preserved && i == 0 && j == 0 {
                // Eq. 8-335 — DC path: copy c_00 through unchanged.
                d[0] = coeffs[0];
                continue;
            }
            let ls = level_scale_4x4(scaling_list, qp_mod, i, j);
            let c = coeffs[idx];
            d[idx] = if qp >= 24 {
                // Eq. 8-336 — no rounding, left shift by (qP/6 - 4).
                (c * ls) << (qp_div - 4)
            } else {
                // Eq. 8-337 — right shift with rounding.
                let shift = 4 - qp_div;
                let round = 1 << (3 - qp_div);
                (c * ls + round) >> shift
            };
        }
    }
    d
}

// ---------------------------------------------------------------------------
// §8.5.12.2 — inverse 4x4 integer DCT (the core "bit-exact" transform).
//
// Equations 8-338..8-354.
// ---------------------------------------------------------------------------

/// §8.5.12.2 — 4x4 inverse transform (eq. 8-338..8-354). Takes the
/// scaled coefficient block `d` and returns the residual block `r`.
fn inverse_core_4x4(d: &[i32; 16]) -> [i32; 16] {
    // Row transform (eqs. 8-338..8-345).
    // f[i][j] = row-transformed value at position (i,j).
    let mut f = [0i32; 16];
    for i in 0..4 {
        let d_i0 = d[i * 4];
        let d_i1 = d[i * 4 + 1];
        let d_i2 = d[i * 4 + 2];
        let d_i3 = d[i * 4 + 3];

        // Intermediate e (eqs. 8-338..8-341).
        let e_i0 = d_i0 + d_i2;
        let e_i1 = d_i0 - d_i2;
        let e_i2 = (d_i1 >> 1) - d_i3;
        let e_i3 = d_i1 + (d_i3 >> 1);

        // Butterfly to f (eqs. 8-342..8-345).
        f[i * 4] = e_i0 + e_i3;
        f[i * 4 + 1] = e_i1 + e_i2;
        f[i * 4 + 2] = e_i1 - e_i2;
        f[i * 4 + 3] = e_i0 - e_i3;
    }

    // Column transform (eqs. 8-346..8-353). Produces h_ij.
    let mut h = [0i32; 16];
    for j in 0..4 {
        let f_0j = f[j];
        let f_1j = f[4 + j];
        let f_2j = f[8 + j];
        let f_3j = f[12 + j];

        // Intermediate g (eqs. 8-346..8-349).
        let g_0j = f_0j + f_2j;
        let g_1j = f_0j - f_2j;
        let g_2j = (f_1j >> 1) - f_3j;
        let g_3j = f_1j + (f_3j >> 1);

        // Butterfly to h (eqs. 8-350..8-353).
        h[j] = g_0j + g_3j;
        h[4 + j] = g_1j + g_2j;
        h[8 + j] = g_1j - g_2j;
        h[12 + j] = g_0j - g_3j;
    }

    // §8.5.12.2 eq. 8-354 — final right shift with rounding by 2^5 / 6.
    let mut r = [0i32; 16];
    for k in 0..16 {
        r[k] = (h[k] + 32) >> 6;
    }
    r
}

// ---------------------------------------------------------------------------
// §8.5.12 — public entry points for 4x4 residual blocks.
// ---------------------------------------------------------------------------

#[inline]
fn check_qp(qp: i32) -> Result<(), TransformError> {
    if !(0..=51).contains(&qp) {
        Err(TransformError::QpOutOfRange(qp))
    } else {
        Ok(())
    }
}

#[inline]
fn check_bit_depth(bit_depth: u32) -> Result<(), TransformError> {
    if !(8..=14).contains(&bit_depth) {
        Err(TransformError::BitDepthOutOfRange(bit_depth))
    } else {
        Ok(())
    }
}

/// §8.5.12 — scaling + inverse transform for a single 4x4 residual
/// block. `coeffs` is row-major (c[i,j] = coeffs[i*4+j]). `scaling_list`
/// is row-major weight_scale values (typically all 16s when scaling
/// matrices are not signalled).
pub fn inverse_transform_4x4(
    coeffs: &[i32; 16],
    qp: i32,
    scaling_list: &[i32; 16],
    bit_depth: u32,
) -> Result<[i32; 16], TransformError> {
    check_qp(qp)?;
    check_bit_depth(bit_depth)?;
    // Standard path: c[0,0] participates in scaling (not an Intra_16x16
    // luma DC block / not a chroma DC block).
    let d = scale_4x4(coeffs, qp, scaling_list, false);
    Ok(inverse_core_4x4(&d))
}

/// §8.5.12 with Intra_16x16 luma AC / chroma AC semantics: c[0,0] is a
/// pre-scaled DC value supplied by the caller (from §8.5.10 / §8.5.11)
/// and must not be rescaled here.
pub fn inverse_transform_4x4_dc_preserved(
    coeffs: &[i32; 16],
    qp: i32,
    scaling_list: &[i32; 16],
    bit_depth: u32,
) -> Result<[i32; 16], TransformError> {
    check_qp(qp)?;
    check_bit_depth(bit_depth)?;
    let d = scale_4x4(coeffs, qp, scaling_list, true);
    Ok(inverse_core_4x4(&d))
}

// ---------------------------------------------------------------------------
// §8.5.13 — 8x8 residual blocks: scaling + inverse transform.
// ---------------------------------------------------------------------------

/// §8.5.13.1 — scaling for 8x8 residual blocks (eqs. 8-356, 8-357).
fn scale_8x8(coeffs: &[i32; 64], qp: i32, scaling_list: &[i32; 64]) -> [i32; 64] {
    let qp_div = qp / 6;
    let qp_mod = (qp % 6) as usize;
    let mut d = [0i32; 64];
    for i in 0..8 {
        for j in 0..8 {
            let idx = i * 8 + j;
            let ls = level_scale_8x8(scaling_list, qp_mod, i, j);
            let c = coeffs[idx];
            d[idx] = if qp >= 36 {
                // Eq. 8-356.
                (c * ls) << (qp_div - 6)
            } else {
                // Eq. 8-357.
                let shift = 6 - qp_div;
                let round = 1 << (5 - qp_div);
                (c * ls + round) >> shift
            };
        }
    }
    d
}

/// §8.5.13.2 — 8x8 inverse transform core (eqs. 8-358..8-406).
fn inverse_core_8x8(d: &[i32; 64]) -> [i32; 64] {
    // Row (horizontal) transform.
    //
    // For each row i: produce g_i[0..8] by sequentially computing
    // e_i[0..8] (eqs. 8-358..8-365), f_i[0..8] (eqs. 8-366..8-373),
    // and g_i[0..8] (eqs. 8-374..8-381).
    let mut g = [0i32; 64];
    for i in 0..8 {
        let base = i * 8;
        let d0 = d[base];
        let d1 = d[base + 1];
        let d2 = d[base + 2];
        let d3 = d[base + 3];
        let d4 = d[base + 4];
        let d5 = d[base + 5];
        let d6 = d[base + 6];
        let d7 = d[base + 7];

        // eqs. 8-358..8-365.
        let e0 = d0 + d4;
        let e1 = -d3 + d5 - d7 - (d7 >> 1);
        let e2 = d0 - d4;
        let e3 = d1 + d7 - d3 - (d3 >> 1);
        let e4 = (d2 >> 1) - d6;
        let e5 = -d1 + d7 + d5 + (d5 >> 1);
        let e6 = d2 + (d6 >> 1);
        let e7 = d3 + d5 + d1 + (d1 >> 1);

        // eqs. 8-366..8-373.
        let f0 = e0 + e6;
        let f1 = e1 + (e7 >> 2);
        let f2 = e2 + e4;
        let f3 = e3 + (e5 >> 2);
        let f4 = e2 - e4;
        let f5 = (e3 >> 2) - e5;
        let f6 = e0 - e6;
        let f7 = e7 - (e1 >> 2);

        // eqs. 8-374..8-381.
        g[base] = f0 + f7;
        g[base + 1] = f2 + f5;
        g[base + 2] = f4 + f3;
        g[base + 3] = f6 + f1;
        g[base + 4] = f6 - f1;
        g[base + 5] = f4 - f3;
        g[base + 6] = f2 - f5;
        g[base + 7] = f0 - f7;
    }

    // Column (vertical) transform.
    //
    // Same 1-D inverse applied to each column. Produces m_ij, eqs.
    // 8-382..8-405.
    let mut m = [0i32; 64];
    for j in 0..8 {
        let g0 = g[j];
        let g1 = g[8 + j];
        let g2 = g[16 + j];
        let g3 = g[24 + j];
        let g4 = g[32 + j];
        let g5 = g[40 + j];
        let g6 = g[48 + j];
        let g7 = g[56 + j];

        // eqs. 8-382..8-389.
        let h0 = g0 + g4;
        let h1 = -g3 + g5 - g7 - (g7 >> 1);
        let h2 = g0 - g4;
        let h3 = g1 + g7 - g3 - (g3 >> 1);
        let h4 = (g2 >> 1) - g6;
        let h5 = -g1 + g7 + g5 + (g5 >> 1);
        let h6 = g2 + (g6 >> 1);
        let h7 = g3 + g5 + g1 + (g1 >> 1);

        // eqs. 8-390..8-397.
        let k0 = h0 + h6;
        let k1 = h1 + (h7 >> 2);
        let k2 = h2 + h4;
        let k3 = h3 + (h5 >> 2);
        let k4 = h2 - h4;
        let k5 = (h3 >> 2) - h5;
        let k6 = h0 - h6;
        let k7 = h7 - (h1 >> 2);

        // eqs. 8-398..8-405.
        m[j] = k0 + k7;
        m[8 + j] = k2 + k5;
        m[16 + j] = k4 + k3;
        m[24 + j] = k6 + k1;
        m[32 + j] = k6 - k1;
        m[40 + j] = k4 - k3;
        m[48 + j] = k2 - k5;
        m[56 + j] = k0 - k7;
    }

    // §8.5.13.2 eq. 8-406 — final right shift with rounding by 2^5 / 6.
    let mut r = [0i32; 64];
    for k in 0..64 {
        r[k] = (m[k] + 32) >> 6;
    }
    r
}

/// §8.5.13 — scaling + inverse transform for a single 8x8 block.
pub fn inverse_transform_8x8(
    coeffs: &[i32; 64],
    qp: i32,
    scaling_list: &[i32; 64],
    bit_depth: u32,
) -> Result<[i32; 64], TransformError> {
    check_qp(qp)?;
    check_bit_depth(bit_depth)?;
    let d = scale_8x8(coeffs, qp, scaling_list);
    Ok(inverse_core_8x8(&d))
}

// ---------------------------------------------------------------------------
// §8.5.10 — Intra_16x16 luma DC block: 4x4 Hadamard + scaling.
// ---------------------------------------------------------------------------

/// §8.5.10 — apply the 4x4 Hadamard-like transform (Eq. 8-320) to the
/// 16 DC coefficients of an Intra_16x16 luma macroblock, then scale
/// each output per Eq. 8-321 / 8-322. The returned `dcY` array is
/// arranged row-major and is meant to be plugged back into each 4x4
/// sub-block as its c[0,0] entry (see §8.5.2 step 2a).
pub fn inverse_hadamard_luma_dc_16x16(
    dc_coeffs: &[i32; 16],
    qp: i32,
    scaling_list: &[i32; 16],
    bit_depth: u32,
) -> Result<[i32; 16], TransformError> {
    check_qp(qp)?;
    check_bit_depth(bit_depth)?;

    // §8.5.10 Eq. 8-320 — f = H * c * H, with
    //     H = [[1, 1, 1, 1],
    //          [1, 1,-1,-1],
    //          [1,-1,-1, 1],
    //          [1,-1, 1,-1]].
    // Being a Hadamard this is self-inverse up to a factor of 4; the
    // factor is absorbed into the scaling step below.
    let c = dc_coeffs;
    let mut t = [0i32; 16]; // t = H * c
    for j in 0..4 {
        // Row 0 of H: [ 1,  1,  1,  1]
        t[j] = c[j] + c[4 + j] + c[8 + j] + c[12 + j];
        // Row 1: [ 1,  1, -1, -1]
        t[4 + j] = c[j] + c[4 + j] - c[8 + j] - c[12 + j];
        // Row 2: [ 1, -1, -1,  1]
        t[8 + j] = c[j] - c[4 + j] - c[8 + j] + c[12 + j];
        // Row 3: [ 1, -1,  1, -1]
        t[12 + j] = c[j] - c[4 + j] + c[8 + j] - c[12 + j];
    }
    let mut f = [0i32; 16]; // f = t * H
    for i in 0..4 {
        let base = i * 4;
        let t0 = t[base];
        let t1 = t[base + 1];
        let t2 = t[base + 2];
        let t3 = t[base + 3];
        // Right-multiplying by H above yields:
        //   f[i][0] = t[i][0]+t[i][1]+t[i][2]+t[i][3]
        //   f[i][1] = t[i][0]+t[i][1]-t[i][2]-t[i][3]
        //   f[i][2] = t[i][0]-t[i][1]-t[i][2]+t[i][3]
        //   f[i][3] = t[i][0]-t[i][1]+t[i][2]-t[i][3]
        f[base] = t0 + t1 + t2 + t3;
        f[base + 1] = t0 + t1 - t2 - t3;
        f[base + 2] = t0 - t1 - t2 + t3;
        f[base + 3] = t0 - t1 + t2 - t3;
    }

    // §8.5.10 — Scale per Eq. 8-321 / 8-322 using LevelScale4x4(qP%6, 0, 0).
    let qp_mod = (qp % 6) as usize;
    let qp_div = qp / 6;
    let ls = level_scale_4x4(scaling_list, qp_mod, 0, 0);

    let mut dc_y = [0i32; 16];
    if qp >= 36 {
        // Eq. 8-321.
        let shift = qp_div - 6;
        for k in 0..16 {
            dc_y[k] = (f[k] * ls) << shift;
        }
    } else {
        // Eq. 8-322.
        let shift = 6 - qp_div;
        let round = 1 << (5 - qp_div);
        for k in 0..16 {
            dc_y[k] = (f[k] * ls + round) >> shift;
        }
    }
    Ok(dc_y)
}

// ---------------------------------------------------------------------------
// §8.5.11 — chroma DC Hadamard + scaling.
// ---------------------------------------------------------------------------

/// §8.5.11.1 + §8.5.11.2 (4:2:0 sub-path) — 2x2 Hadamard on chroma DC,
/// followed by §8.5.11.2 Eq. 8-326 scaling.
///
/// `dc_coeffs` is the 2x2 coefficient array laid out row-major:
///   [c00, c01,
///    c10, c11].
///
/// `scaling_list_dc_entry` is LevelScale4x4(qp_c%6, 0, 0) worth of
/// state — callers must pass `scaling_list[0] * normAdjust4x4(qp%6,0,0)`;
/// the helper below recomputes it from a flat scaling list entry if
/// desired via [`level_scale_4x4`].
pub fn inverse_hadamard_chroma_dc_420(
    dc_coeffs: &[i32; 4],
    qp: i32,
    scaling_list: &[i32; 16],
    bit_depth: u32,
) -> Result<[i32; 4], TransformError> {
    check_qp(qp)?;
    check_bit_depth(bit_depth)?;

    // §8.5.11.1 Eq. 8-324 — f = H2 * c * H2 with
    //   H2 = [[1, 1],
    //         [1,-1]].
    //
    // Unrolled:
    //   t00 = c00 + c10,  t01 = c01 + c11
    //   t10 = c00 - c10,  t11 = c01 - c11
    //   f00 = t00 + t01,  f01 = t00 - t01
    //   f10 = t10 + t11,  f11 = t10 - t11
    let c00 = dc_coeffs[0];
    let c01 = dc_coeffs[1];
    let c10 = dc_coeffs[2];
    let c11 = dc_coeffs[3];
    let t00 = c00 + c10;
    let t01 = c01 + c11;
    let t10 = c00 - c10;
    let t11 = c01 - c11;
    let f = [t00 + t01, t00 - t01, t10 + t11, t10 - t11];

    // §8.5.11.2 Eq. 8-326 — scaling for 4:2:0:
    //   dcC_ij = ( (f_ij * LevelScale4x4(qP%6, 0, 0)) << (qP/6) ) >> 1
    let qp_mod = (qp % 6) as usize;
    let qp_div = qp / 6;
    let ls = level_scale_4x4(scaling_list, qp_mod, 0, 0);
    let mut out = [0i32; 4];
    for k in 0..4 {
        out[k] = ((f[k] * ls) << qp_div) >> 1;
    }
    Ok(out)
}

/// §8.5.11.1 + §8.5.11.2 (4:2:2 sub-path) — 2x4 Hadamard (Eq. 8-325)
/// followed by Eq. 8-327..8-329 scaling (qPDC = qP + 3).
///
/// The 2x4 chroma DC array is laid out row-major as a 4x2 matrix:
///   [c00, c01,
///    c10, c11,
///    c20, c21,
///    c30, c31].
///
/// (ChromaArrayType==2 gives a 4x2 = 8-entry DC array, as per Eq. 8-305.)
pub fn inverse_hadamard_chroma_dc_422(
    dc_coeffs: &[i32; 8],
    qp: i32,
    scaling_list: &[i32; 16],
    bit_depth: u32,
) -> Result<[i32; 8], TransformError> {
    check_qp(qp)?;
    check_bit_depth(bit_depth)?;

    // §8.5.11.1 Eq. 8-325: f = Hv * c * Hh, where
    //   Hv = [[1,  1,  1,  1],
    //         [1,  1, -1, -1],
    //         [1, -1, -1,  1],
    //         [1, -1,  1, -1]]  (4x4 Hadamard, acts on the 4-row side)
    //   Hh = [[1,  1],
    //         [1, -1]]           (2x2 on the 2-column side)
    //
    // Apply Hv to the left (combines rows).
    let c00 = dc_coeffs[0]; let c01 = dc_coeffs[1];
    let c10 = dc_coeffs[2]; let c11 = dc_coeffs[3];
    let c20 = dc_coeffs[4]; let c21 = dc_coeffs[5];
    let c30 = dc_coeffs[6]; let c31 = dc_coeffs[7];
    let t00 = c00 + c10 + c20 + c30;
    let t01 = c01 + c11 + c21 + c31;
    let t10 = c00 + c10 - c20 - c30;
    let t11 = c01 + c11 - c21 - c31;
    let t20 = c00 - c10 - c20 + c30;
    let t21 = c01 - c11 - c21 + c31;
    let t30 = c00 - c10 + c20 - c30;
    let t31 = c01 - c11 + c21 - c31;
    // Apply Hh to the right (combines columns).
    let f = [
        t00 + t01, t00 - t01,
        t10 + t11, t10 - t11,
        t20 + t21, t20 - t21,
        t30 + t31, t30 - t31,
    ];

    // §8.5.11.2 — scaling path with qPDC = qP + 3 (Eq. 8-327).
    let qp_dc = qp + 3;
    let qp_mod_for_ls = (qp % 6) as usize; // eq 8-328/329 keeps qP%6 for LevelScale
    let ls = level_scale_4x4(scaling_list, qp_mod_for_ls, 0, 0);
    let qp_dc_div = qp_dc / 6;
    let mut out = [0i32; 8];
    if qp_dc >= 36 {
        // Eq. 8-328.
        let shift = qp_dc_div - 6;
        for k in 0..8 {
            out[k] = (f[k] * ls) << shift;
        }
    } else {
        // Eq. 8-329.
        let shift = 6 - qp_dc_div;
        let round = 1 << (5 - qp_dc_div);
        for k in 0..8 {
            out[k] = (f[k] * ls + round) >> shift;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------ QPc lookup (§8.5.8 / Table 8-15) ----------------------

    #[test]
    fn qpc_identity_below_30() {
        for qp in 0..30 {
            assert_eq!(qp_c_table_lookup(qp), qp);
        }
    }

    #[test]
    fn qpc_table_boundaries() {
        // Per Table 8-15, qPI=30 -> 29, qPI=39 -> 35, qPI=51 -> 39.
        assert_eq!(qp_c_table_lookup(30), 29);
        assert_eq!(qp_c_table_lookup(31), 30);
        assert_eq!(qp_c_table_lookup(39), 35);
        assert_eq!(qp_c_table_lookup(40), 36);
        assert_eq!(qp_c_table_lookup(51), 39);
    }

    #[test]
    fn qpc_clips_high_qpi() {
        // Out of range saturates to the qPI=51 entry.
        assert_eq!(qp_c_table_lookup(100), 39);
    }

    #[test]
    fn qp_y_to_qp_c_with_offset() {
        // qp_y = 28 + offset 2 -> qPI=30 -> QPc=29.
        assert_eq!(qp_y_to_qp_c(28, 2), 29);
        // qp_y = 40 + offset 0 -> qPI=40 -> QPc=36.
        assert_eq!(qp_y_to_qp_c(40, 0), 36);
        // Negative accumulation clamps to 0.
        assert_eq!(qp_y_to_qp_c(-5, -10), 0);
        // Huge positive clamps to 51 -> 39.
        assert_eq!(qp_y_to_qp_c(80, 10), 39);
    }

    // ------------ 4x4 inverse transform ---------------------------------

    #[test]
    fn inverse_transform_4x4_all_zero() {
        let coeffs = [0i32; 16];
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_transform_4x4(&coeffs, 26, &sl, 8).unwrap();
        assert_eq!(out, [0i32; 16]);
    }

    #[test]
    fn inverse_transform_4x4_rejects_bad_qp() {
        let coeffs = [0i32; 16];
        let sl = default_scaling_list_4x4_flat();
        assert_eq!(
            inverse_transform_4x4(&coeffs, -1, &sl, 8).unwrap_err(),
            TransformError::QpOutOfRange(-1)
        );
        assert_eq!(
            inverse_transform_4x4(&coeffs, 52, &sl, 8).unwrap_err(),
            TransformError::QpOutOfRange(52)
        );
    }

    #[test]
    fn inverse_transform_4x4_rejects_bad_bit_depth() {
        let coeffs = [0i32; 16];
        let sl = default_scaling_list_4x4_flat();
        assert_eq!(
            inverse_transform_4x4(&coeffs, 26, &sl, 7).unwrap_err(),
            TransformError::BitDepthOutOfRange(7)
        );
        assert_eq!(
            inverse_transform_4x4(&coeffs, 26, &sl, 15).unwrap_err(),
            TransformError::BitDepthOutOfRange(15)
        );
    }

    #[test]
    fn inverse_transform_4x4_single_dc_qp24() {
        // qP = 24: qP/6 = 4, so qP-24 >=0 path (eq. 8-336), shift by 0.
        // LevelScale4x4(0,0,0) = weightScale(0,0)=16 * normAdjust(0,0,0)=10 = 160.
        // d[0,0] = c[0,0] * 160 << 0 = 160.
        // Inverse transform: single DC in e/f propagates to all positions
        // as d_00 (since e_i0 = d_00, e_i1 = d_00 for row 0; g_0j / h
        // similarly). The full derivation yields:
        //   row: f_i0 = f_i1 = f_i2 = f_i3 = d_00 for i=0 only; other rows zero.
        //   col: h_0j = f_0j for j=0..3; other rows zero.
        // So each h_ij = d_00 = 160 for i=j=0 only... wait: let's trace.
        //
        // Row transform, row 0 only (rest zero):
        //   e00 = d00+0 = 160; e01 = d00-0 = 160
        //   e02 = 0 - 0 = 0;  e03 = 0 + 0 = 0
        //   f00 = e00+e03 = 160; f01 = e01+e02 = 160
        //   f02 = e01-e02 = 160; f03 = e00-e03 = 160
        // So f = row 0 all 160, rest zero.
        //
        // Column transform for each j (only f_0j non-zero):
        //   g_0j = f_0j + 0 = 160; g_1j = f_0j - 0 = 160
        //   g_2j = 0 - 0 = 0;      g_3j = 0 + 0 = 0
        //   h_0j = g_0j+g_3j = 160
        //   h_1j = g_1j+g_2j = 160
        //   h_2j = g_1j-g_2j = 160
        //   h_3j = g_0j-g_3j = 160
        // => h is all-160.
        // r_ij = (160+32) >> 6 = 192 >> 6 = 3.
        let mut coeffs = [0i32; 16];
        coeffs[0] = 1;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_transform_4x4(&coeffs, 24, &sl, 8).unwrap();
        assert_eq!(out, [3i32; 16]);
    }

    #[test]
    fn inverse_transform_4x4_single_dc_qp30() {
        // qP = 30: qP/6 = 5, >=24, so eq. 8-336 with shift (5-4)=1.
        // LevelScale(0,0,0) at qP%6=0 = 16*10 = 160.
        // d[0,0] = 1*160 << 1 = 320.
        // By same argument, h is all-320 and r_ij = (320+32)>>6 = 352>>6 = 5.
        let mut coeffs = [0i32; 16];
        coeffs[0] = 1;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_transform_4x4(&coeffs, 30, &sl, 8).unwrap();
        assert_eq!(out, [5i32; 16]);
    }

    #[test]
    fn inverse_transform_4x4_single_dc_qp18() {
        // qP = 18: qP/6 = 3, <24, eq. 8-337 path.
        //   shift = 4-3 = 1, round = 1 << (3-3) = 1.
        // LevelScale(qP%6=0, 0, 0) = 16*10 = 160.
        // d[0,0] = (1*160 + 1) >> 1 = 161 >> 1 = 80.
        // h = all-80; r_ij = (80+32)>>6 = 112>>6 = 1.
        let mut coeffs = [0i32; 16];
        coeffs[0] = 1;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_transform_4x4(&coeffs, 18, &sl, 8).unwrap();
        assert_eq!(out, [1i32; 16]);
    }

    #[test]
    fn inverse_transform_4x4_dc_preserved_skips_scaling() {
        // With is_dc_preserved=true the input c[0,0] is used as the
        // scaled value directly. Setting c[0,0]=64 produces h_ij=64
        // across the board and r_ij = (64+32)>>6 = 1.
        let mut coeffs = [0i32; 16];
        coeffs[0] = 64;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_transform_4x4_dc_preserved(&coeffs, 24, &sl, 8).unwrap();
        assert_eq!(out, [1i32; 16]);
    }

    #[test]
    fn inverse_transform_4x4_known_pattern_qp0() {
        // qP = 0 -> qP/6=0, qP<24, eq. 8-337: shift=4, round=8.
        // LevelScale(0, i, j) uses normAdjust4x4(0, i, j) = {10,16,13}.
        //   norm(0,0,0)=10, norm(0,0,1)=13, norm(0,0,2)=10,  norm(0,0,3)=13
        //   norm(0,1,0)=13, norm(0,1,1)=16, norm(0,1,2)=13,  norm(0,1,3)=16
        //   norm(0,2,0)=10, norm(0,2,1)=13, norm(0,2,2)=10,  norm(0,2,3)=13
        //   norm(0,3,0)=13, norm(0,3,1)=16, norm(0,3,2)=13,  norm(0,3,3)=16
        // Weight scale = 16 everywhere, so LevelScale = 16*norm.
        // A single c[0,0]=1 -> d[0,0] = (1*160+8)>>4 = 168>>4 = 10.
        // Row xform: f row 0 all = 10. Col xform: h all = 10.
        // r_ij = (10+32)>>6 = 42>>6 = 0 (integer).
        //
        // Instead we pick c[0,0]=64 (scaled up to survive the DC path).
        //   d[0,0] = (64*160+8)>>4 = 10248>>4 = 640.
        //   f row 0 all 640; h all 640; r_ij = (640+32)>>6 = 10.
        let mut coeffs = [0i32; 16];
        coeffs[0] = 64;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_transform_4x4(&coeffs, 0, &sl, 8).unwrap();
        assert_eq!(out, [10i32; 16]);
    }

    // ------------ 8x8 inverse transform ---------------------------------

    #[test]
    fn inverse_transform_8x8_all_zero() {
        let coeffs = [0i32; 64];
        let sl = default_scaling_list_8x8_flat();
        let out = inverse_transform_8x8(&coeffs, 26, &sl, 8).unwrap();
        assert_eq!(out, [0i32; 64]);
    }

    #[test]
    fn inverse_transform_8x8_rejects_bad_qp() {
        let coeffs = [0i32; 64];
        let sl = default_scaling_list_8x8_flat();
        assert_eq!(
            inverse_transform_8x8(&coeffs, -5, &sl, 8).unwrap_err(),
            TransformError::QpOutOfRange(-5)
        );
        assert_eq!(
            inverse_transform_8x8(&coeffs, 60, &sl, 8).unwrap_err(),
            TransformError::QpOutOfRange(60)
        );
    }

    #[test]
    fn inverse_transform_8x8_single_dc_qp36() {
        // qP = 36: qP/6 = 6, >=36 path (eq. 8-356, shift by 0).
        // LevelScale8x8(0, 0, 0) = weightScale(0,0)=16 * normAdjust8x8(0,0,0)=20 = 320.
        // d[0,0] = 1 * 320 << 0 = 320.
        //
        // Inverse 8x8: with only d[0,0] non-zero, following the spec
        // butterflies row-by-row:
        //   Row 0 only non-zero column j=0, d0=320, rest 0.
        //   e0=d0+d4=320; e1=0; e2=d0-d4=320; e3=0; e4=0; e5=0; e6=0; e7=0.
        //   f0=e0+e6=320; f1=e1+(e7>>2)=0; f2=e2+e4=320; f3=e3+(e5>>2)=0;
        //   f4=e2-e4=320; f5=(e3>>2)-e5=0; f6=e0-e6=320; f7=e7-(e1>>2)=0.
        //   g0=f0+f7=320; g1=f2+f5=320; g2=f4+f3=320; g3=f6+f1=320;
        //   g4=f6-f1=320; g5=f4-f3=320; g6=f2-f5=320; g7=f0-f7=320.
        // So after row transform g[row=0, j=0..7] is all 320, all
        // other rows are zero.
        //
        // Column transform: for each column j, only g_0j = 320, rest 0.
        // By the same derivation, m[i=0..7, j] = 320 for all i.
        // Final r_ij = (320+32)>>6 = 352>>6 = 5.
        let mut coeffs = [0i32; 64];
        coeffs[0] = 1;
        let sl = default_scaling_list_8x8_flat();
        let out = inverse_transform_8x8(&coeffs, 36, &sl, 8).unwrap();
        assert_eq!(out, [5i32; 64]);
    }

    #[test]
    fn inverse_transform_8x8_single_dc_qp24() {
        // qP = 24: qP/6=4, <36, eq. 8-357: shift=2, round=2.
        // LevelScale8x8(0,0,0) = 16*20 = 320.
        //   d[0,0] = (1*320 + 2) >> 2 = 322>>2 = 80.
        // Same derivation as above -> r_ij = (80+32)>>6 = 112>>6 = 1.
        let mut coeffs = [0i32; 64];
        coeffs[0] = 1;
        let sl = default_scaling_list_8x8_flat();
        let out = inverse_transform_8x8(&coeffs, 24, &sl, 8).unwrap();
        assert_eq!(out, [1i32; 64]);
    }

    // ------------ 16x16 luma DC Hadamard (§8.5.10) ----------------------

    #[test]
    fn luma_dc_16x16_all_zero() {
        let dc = [0i32; 16];
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_luma_dc_16x16(&dc, 26, &sl, 8).unwrap();
        assert_eq!(out, [0i32; 16]);
    }

    #[test]
    fn luma_dc_16x16_rejects_bad_qp() {
        let dc = [0i32; 16];
        let sl = default_scaling_list_4x4_flat();
        assert_eq!(
            inverse_hadamard_luma_dc_16x16(&dc, 52, &sl, 8).unwrap_err(),
            TransformError::QpOutOfRange(52)
        );
    }

    #[test]
    fn luma_dc_16x16_single_dc_qp36() {
        // Only c[0,0]=1, others 0. The Hadamard H*c*H sends this
        // all-zero-except-(0,0)=1 matrix to a matrix of all 1s
        // (with sign = +1 at every position since H_ik * H_jk = ±1
        // and the (0,0,0,0) product is +1, but actually:
        //   f_ij = sum_{p,q} H_ip * c_pq * H_qj = H_i0 * c_00 * H_0j
        //        = H_i0 * H_0j * 1.
        // H_i0 = 1 for all i (first column of H). H_0j = 1 for all j
        // (first row of H). So f_ij = 1 for all i,j.
        //
        // qP=36 -> qp_div=6, >=36 path: shift=0.
        // LevelScale4x4(0, 0, 0) = 16*10 = 160.
        // dc_y_ij = (1 * 160) << 0 = 160 at every position.
        let mut dc = [0i32; 16];
        dc[0] = 1;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_luma_dc_16x16(&dc, 36, &sl, 8).unwrap();
        assert_eq!(out, [160i32; 16]);
    }

    #[test]
    fn luma_dc_16x16_hadamard_self_inverse() {
        // Feeding a "delta" Hadamard input (all equal) should, after the
        // double Hadamard, produce zero everywhere except the top-left.
        // Specifically, c_ij = 1 for every i,j:
        //   (H*c)_ij = sum_k H_ik * c_kj = sum_k H_ik (since c=1).
        //           sum_k H_0k = 4; sum_k H_{i!=0,k} = 0 (row sums).
        //   So (H*c) is a matrix whose only non-zero row is row 0 = [4,4,4,4].
        //   Then (H*c*H)_ij = sum_l (H*c)_il * H_lj. Only l=0 non-zero:
        //     = 4 * H_0j. Row 0 of H is all 1s, so _ = 4 everywhere in row 0 only.
        // Actually: (H*c*H)_ij = (H*c)_i0 * H_0j + ... but H acts on l-index
        // of (H*c)_il. (H*c)_il is 4 if i==0 and 0 otherwise, regardless
        // of l. So (H*c*H)_ij = sum_l (H*c)_il * H_lj.
        // For i=0: = sum_l 4 * H_lj = 4 * sum_l H_lj. Column sums: col 0 = 4, rest = 0.
        // For i!=0: =0.
        // So f is all zero except f[0,0] = 16.
        //
        // qP=36, LevelScale=160, dc_y[0,0] = 16*160 = 2560, rest 0.
        let dc = [1i32; 16];
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_luma_dc_16x16(&dc, 36, &sl, 8).unwrap();
        let mut expected = [0i32; 16];
        expected[0] = 2560;
        assert_eq!(out, expected);
    }

    // ------------ Chroma DC 2x2 (§8.5.11.1 / §8.5.11.2) -----------------

    #[test]
    fn chroma_dc_420_all_zero() {
        let dc = [0i32; 4];
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_chroma_dc_420(&dc, 20, &sl, 8).unwrap();
        assert_eq!(out, [0i32; 4]);
    }

    #[test]
    fn chroma_dc_420_single_dc_qp0() {
        // dc[0,0]=1, rest 0.
        // 2x2 Hadamard:
        //   t00=1+0=1, t01=0, t10=1-0=1, t11=0
        //   f00=t00+t01=1, f01=t00-t01=1, f10=t10+t11=1, f11=t10-t11=1
        // qP=0: qp_div=0, Eq. 8-326: dcC_ij = (f_ij * LevelScale(0,0,0)) << 0 >> 1
        //   LevelScale(0,0,0) = 16*10 = 160.
        //   dcC = (1*160)>>1 = 80 at every slot.
        let mut dc = [0i32; 4];
        dc[0] = 1;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_chroma_dc_420(&dc, 0, &sl, 8).unwrap();
        assert_eq!(out, [80i32; 4]);
    }

    #[test]
    fn chroma_dc_420_single_dc_qp6() {
        // qP=6 -> qp_div=1, LevelScale(0,0,0)=160 (qP%6=0).
        // dcC = (1*160 << 1) >> 1 = 320>>1 = 160.
        let mut dc = [0i32; 4];
        dc[0] = 1;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_chroma_dc_420(&dc, 6, &sl, 8).unwrap();
        assert_eq!(out, [160i32; 4]);
    }

    #[test]
    fn chroma_dc_420_hadamard_self_inverse() {
        // c = [[1,1],[1,1]] -> 2x2 Hadamard:
        //   t00=2, t01=2, t10=0, t11=0
        //   f00=4, f01=0, f10=0, f11=0
        let dc = [1i32; 4];
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_chroma_dc_420(&dc, 0, &sl, 8).unwrap();
        // f = [4, 0, 0, 0]. dcC = [(4*160)>>1, 0, 0, 0] = [320, 0, 0, 0].
        assert_eq!(out, [320, 0, 0, 0]);
    }

    // ------------ Chroma DC 2x4 (§8.5.11.2 / 4:2:2) ---------------------

    #[test]
    fn chroma_dc_422_all_zero() {
        let dc = [0i32; 8];
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_chroma_dc_422(&dc, 20, &sl, 8).unwrap();
        assert_eq!(out, [0i32; 8]);
    }

    #[test]
    fn chroma_dc_422_single_dc_qp0() {
        // dc[0,0]=1. 4x4 Hadamard on rows + 2x2 on columns:
        //   t_ij (after row transform) = H[i,0]*c[0,j] + 0 = H[i,0]*c[0,j]
        //   H[i,0] = 1 for all i. t00=1,t01=0; t10=1,t11=0; t20=1,t21=0; t30=1,t31=0.
        //   After col transform: f[i,0]=t_i0+t_i1 = 1; f[i,1]=t_i0-t_i1 = 1.
        //   So f is all 1s (8 entries).
        // qP=0 -> qpDC=3, qpDC<36, eq. 8-329: shift=5-0=5... wait qpDC/6=0.
        //   shift=6-0=6, round = 1<<(5-0) = 32.
        //   LevelScale(0,0,0) = 160.
        //   dcC = (1*160 + 32) >> 6 = 192>>6 = 3.
        let mut dc = [0i32; 8];
        dc[0] = 1;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_chroma_dc_422(&dc, 0, &sl, 8).unwrap();
        assert_eq!(out, [3i32; 8]);
    }

    #[test]
    fn chroma_dc_422_rejects_bad_qp() {
        let dc = [0i32; 8];
        let sl = default_scaling_list_4x4_flat();
        assert_eq!(
            inverse_hadamard_chroma_dc_422(&dc, -1, &sl, 8).unwrap_err(),
            TransformError::QpOutOfRange(-1)
        );
    }

    // ------------ Scaling list helpers ----------------------------------

    #[test]
    fn default_scaling_lists_all_16s() {
        assert_eq!(default_scaling_list_4x4_flat(), [16i32; 16]);
        assert_eq!(default_scaling_list_8x8_flat(), [16i32; 64]);
    }

    // ------------ Inverse scan (§8.5.6) ---------------------------------

    #[test]
    fn zigzag_scan_is_bijection() {
        // A scan-order list of 0..16 should map to a 4x4 matrix whose
        // entries, when read back in zig-zag, give 0..16.
        let mut levels = [0i32; 16];
        for k in 0..16 {
            levels[k] = k as i32;
        }
        let block = inverse_scan_4x4_zigzag(&levels);
        // Verify per Table 8-13: idx 0 -> c00, idx 1 -> c01, idx 2 -> c10, etc.
        assert_eq!(block[0 * 4 + 0], 0);
        assert_eq!(block[0 * 4 + 1], 1);
        assert_eq!(block[1 * 4 + 0], 2);
        assert_eq!(block[2 * 4 + 0], 3);
        assert_eq!(block[1 * 4 + 1], 4);
        assert_eq!(block[0 * 4 + 2], 5);
        assert_eq!(block[3 * 4 + 3], 15);
    }

    // ------------ NormAdjust spot checks (Tables 8-4, 8-5) --------------

    #[test]
    fn norm_adjust_4x4_spot_checks() {
        // Row 0 (qP%6=0): v = [10, 16, 13].
        assert_eq!(norm_adjust_4x4(0, 0, 0), 10); // (0,0) class
        assert_eq!(norm_adjust_4x4(0, 1, 1), 16); // (1,1) class
        assert_eq!(norm_adjust_4x4(0, 0, 1), 13); // other
        assert_eq!(norm_adjust_4x4(0, 2, 3), 13); // other
        // Row 5 (qP%6=5): v = [18, 29, 23].
        assert_eq!(norm_adjust_4x4(5, 0, 0), 18);
        assert_eq!(norm_adjust_4x4(5, 3, 3), 29);
        assert_eq!(norm_adjust_4x4(5, 2, 1), 23);
    }

    #[test]
    fn norm_adjust_8x8_spot_checks() {
        // Row 0: v = [20, 18, 32, 19, 25, 24].
        assert_eq!(norm_adjust_8x8(0, 0, 0), 20);   // (i%4,j%4)=(0,0)
        assert_eq!(norm_adjust_8x8(0, 4, 4), 20);   // (i%4,j%4)=(0,0) again
        assert_eq!(norm_adjust_8x8(0, 1, 1), 18);   // (i%2,j%2)=(1,1)
        assert_eq!(norm_adjust_8x8(0, 3, 3), 18);
        assert_eq!(norm_adjust_8x8(0, 2, 2), 32);   // (i%4,j%4)=(2,2)
        assert_eq!(norm_adjust_8x8(0, 6, 6), 32);
        assert_eq!(norm_adjust_8x8(0, 0, 1), 19);   // (i%4,j%2)=(0,1)
        assert_eq!(norm_adjust_8x8(0, 1, 0), 19);   // (i%2,j%4)=(1,0)
        assert_eq!(norm_adjust_8x8(0, 0, 2), 25);   // (i%4,j%4)=(0,2)
        assert_eq!(norm_adjust_8x8(0, 2, 0), 25);   // (i%4,j%4)=(2,0)
        // "otherwise" class:
        assert_eq!(norm_adjust_8x8(0, 1, 2), 24);
        assert_eq!(norm_adjust_8x8(0, 2, 1), 24);
        assert_eq!(norm_adjust_8x8(0, 2, 3), 24);
    }
}
