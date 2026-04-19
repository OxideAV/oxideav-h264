//! Inverse integer transforms and dequantisation — ITU-T H.264 §8.5.
//!
//! This module covers:
//!
//! * §8.5.6 — quantisation tables (`v[6][3]` / `LevelScale`).
//! * §8.5.10 — inverse Hadamard 4×4 (Intra16×16 DC luma).
//! * §8.5.11 — inverse Hadamard 2×2 (4:2:0 chroma DC).
//! * §8.5.12 — inverse 4×4 integer transform (§8.5.12.2 with the +(1<<5)
//!   rounding offset for AC and the simple `>> 6` for DC-loaded blocks).
//! * §8.5.13 — inverse 8×8 integer transform (§8.5.13.2) and its 6-class
//!   dequantisation using the `normAdjust8x8` table (§8.5.12.1, Table 8-15).
//! * §8.5 / §8.5.10 / §7.4.2.1 — dequantisation `c'[i,j] = c[i,j] * scale`.

/// `v[m][n]` — Table 8-13. Indexed by (qp_y % 6, position class). The three
/// position classes in a 4×4 block are:
///   0: positions (0,0) (0,2) (2,0) (2,2)
///   1: positions (1,1) (1,3) (3,1) (3,3)
///   2: all other positions
const V_TABLE: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

/// Position class for each cell of a 4×4 block.
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

/// Dequantise a 4×4 AC block in raster order using the flat (all-16)
/// scaling list. `qp` is the per-block QP value (luma or chroma already
/// adjusted via §8.5.11.1 / §8.5.11.2).
///
/// Per §8.5.10: `c'[i,j] = c[i,j] * LevelScale4x4(qP%6, i, j) << (qP/6)`
/// — applied to all 16 positions for a normal AC 4×4 block. For an Intra16×16
/// AC block, the DC slot (raster 0) is left at 0 and gets filled later from
/// the Hadamard pass.
pub fn dequantize_4x4(coeffs: &mut [i32; 16], qp: i32) {
    dequantize_4x4_scaled(coeffs, qp, &crate::scaling_list::FLAT_4X4);
}

/// Dequantise a 4×4 AC block using a custom weight matrix (§7.4.2.1.1
/// / §8.5.12.1). With `weight_scale` = flat-16 this matches
/// [`dequantize_4x4`] bit-exactly because the spec's `/16` folds
/// cleanly back into the shared IDCT's final `>>6`. For non-flat
/// weights the full spec formula is applied:
///
/// ```text
/// d = (c * w * V + round) >> shift    for qP <  24 (shift = 4 - qP/6)
/// d = (c * w * V)       << -shift     for qP >= 24
/// ```
///
/// where `V = normAdjust4x4(qP%6, i, j)` and `w = weightScale4x4[i,j]`.
/// The rounding keeps the low bits, avoiding the precision loss of
/// multiplying a pre-rounded `(V * w) >> 4` factor into the coefficient.
pub fn dequantize_4x4_scaled(coeffs: &mut [i32; 16], qp: i32, weight_scale: &[i16; 16]) {
    // Clamp to the 8-bit QP range. High-bit-depth callers that need
    // `qP > 51` use [`dequantize_4x4_scaled_ext`] which accepts the
    // QP range extended by `QpBdOffset` (§7.4.2.1.1).
    let qp = qp.clamp(0, 51);
    // Flat fast-path: bit-exact reproduction of the flat-16 pipeline.
    // With weight_scale = FLAT_4X4 (all 16) the JM-style formula
    // `(c * V * 16 + round) >> shift` collapses to `c * V` with the
    // `<< qp/6` absorbed in the shift difference.
    if weight_scale == &crate::scaling_list::FLAT_4X4 {
        let qp6 = (qp / 6) as u32;
        let qmod = (qp % 6) as usize;
        for r in 0..4 {
            for c in 0..4 {
                let i = r * 4 + c;
                let scale = V_TABLE[qmod][pos_class(r, c)];
                coeffs[i] = (coeffs[i] * scale) << qp6;
            }
        }
        return;
    }
    let qp6 = (qp / 6) as i32;
    let qmod = (qp % 6) as usize;
    if qp6 >= 4 {
        let shift = (qp6 - 4) as u32;
        for r in 0..4 {
            for c in 0..4 {
                let i = r * 4 + c;
                let w = weight_scale[i] as i32;
                let v = V_TABLE[qmod][pos_class(r, c)];
                coeffs[i] = (coeffs[i] * w * v) << shift;
            }
        }
    } else {
        let shift = (4 - qp6) as u32;
        let round = 1i32 << (shift - 1);
        for r in 0..4 {
            for c in 0..4 {
                let i = r * 4 + c;
                let w = weight_scale[i] as i32;
                let v = V_TABLE[qmod][pos_class(r, c)];
                coeffs[i] = (coeffs[i] * w * v + round) >> shift;
            }
        }
    }
}

/// Inverse 4×4 transform as per §8.5.12.2.
///
/// Input: dequantised coefficient block in raster order.
/// Output: residual sample block in raster order, scaled by `1 << 6`. The
/// caller is responsible for the final `(x + 32) >> 6` rounding when adding
/// to the prediction (matches the spec — the `(1<<5)` is inside the
/// transform but applied per sample).
pub fn idct_4x4(coeffs: &mut [i32; 16]) {
    let mut tmp = [0i32; 16];
    // Horizontal: rows.
    for r in 0..4 {
        let c0 = coeffs[r * 4];
        let c1 = coeffs[r * 4 + 1];
        let c2 = coeffs[r * 4 + 2];
        let c3 = coeffs[r * 4 + 3];
        let e = c0 + c2;
        let f = c0 - c2;
        let g = (c1 >> 1) - c3;
        let h = c1 + (c3 >> 1);
        tmp[r * 4] = e + h;
        tmp[r * 4 + 1] = f + g;
        tmp[r * 4 + 2] = f - g;
        tmp[r * 4 + 3] = e - h;
    }
    // Vertical: cols.
    for c in 0..4 {
        let c0 = tmp[c];
        let c1 = tmp[4 + c];
        let c2 = tmp[8 + c];
        let c3 = tmp[12 + c];
        let e = c0 + c2;
        let f = c0 - c2;
        let g = (c1 >> 1) - c3;
        let h = c1 + (c3 >> 1);
        coeffs[c] = e + h;
        coeffs[4 + c] = f + g;
        coeffs[8 + c] = f - g;
        coeffs[12 + c] = e - h;
    }
    // Spec: residual sample = (transformed + 32) >> 6.
    for v in coeffs.iter_mut() {
        *v = (*v + 32) >> 6;
    }
}

/// Extended-QP variant of [`dequantize_4x4_scaled`] — accepts `qP`
/// up to `51 + QpBdOffset` (§7.4.2.1.1). Used by the high-bit-depth
/// decode path where `QpY' = QpY + QpBdOffsetY` may exceed 51.
///
/// Internal precision is widened to i64 to avoid the i32 overflow
/// that occurs at `qP >= ~30` when coefficient magnitudes approach
/// the 16-bit residual bound. The shift pattern mirrors the spec's
/// §8.5.12.1 formula with an unbounded `qP/6` shift.
pub fn dequantize_4x4_scaled_ext(coeffs: &mut [i32; 16], qp: i32, weight_scale: &[i16; 16]) {
    let qp = qp.max(0);
    let qp6 = (qp / 6) as i32;
    let qmod = (qp % 6) as usize;
    if weight_scale == &crate::scaling_list::FLAT_4X4 {
        let shift = qp6 as u32;
        for r in 0..4 {
            for c in 0..4 {
                let i = r * 4 + c;
                let scale = V_TABLE[qmod][pos_class(r, c)] as i64;
                coeffs[i] = ((coeffs[i] as i64 * scale) << shift) as i32;
            }
        }
        return;
    }
    if qp6 >= 4 {
        let shift = (qp6 - 4) as u32;
        for r in 0..4 {
            for c in 0..4 {
                let i = r * 4 + c;
                let w = weight_scale[i] as i64;
                let v = V_TABLE[qmod][pos_class(r, c)] as i64;
                coeffs[i] = ((coeffs[i] as i64 * w * v) << shift) as i32;
            }
        }
    } else {
        let shift = (4 - qp6) as u32;
        let round = 1i64 << (shift - 1);
        for r in 0..4 {
            for c in 0..4 {
                let i = r * 4 + c;
                let w = weight_scale[i] as i64;
                let v = V_TABLE[qmod][pos_class(r, c)] as i64;
                coeffs[i] = ((coeffs[i] as i64 * w * v + round) >> shift) as i32;
            }
        }
    }
}

/// Extended-QP variant of [`inv_hadamard_4x4_dc_scaled`]. Accepts
/// `qP` up to `51 + QpBdOffset`. Same formula as the base variant
/// but widened to i64 so the left shifts at high QP don't overflow.
pub fn inv_hadamard_4x4_dc_scaled_ext(dc: &mut [i32; 16], qp: i32, weight_scale_00: i16) {
    let mut tmp = [0i32; 16];
    for r in 0..4 {
        let a = dc[r * 4];
        let b = dc[r * 4 + 1];
        let c = dc[r * 4 + 2];
        let d = dc[r * 4 + 3];
        tmp[r * 4] = a + b + c + d;
        tmp[r * 4 + 1] = a + b - c - d;
        tmp[r * 4 + 2] = a - b - c + d;
        tmp[r * 4 + 3] = a - b + c - d;
    }
    for c in 0..4 {
        let a = tmp[c];
        let b = tmp[4 + c];
        let cc = tmp[8 + c];
        let d = tmp[12 + c];
        dc[c] = a + b + cc + d;
        dc[4 + c] = a + b - cc - d;
        dc[8 + c] = a - b - cc + d;
        dc[12 + c] = a - b + cc - d;
    }

    let qp = qp.max(0);
    let qp6 = (qp / 6) as i32;
    let qmod = (qp % 6) as usize;
    let v = V_TABLE[qmod][0] as i64;
    if weight_scale_00 == 16 {
        if qp6 >= 2 {
            let shift = (qp6 - 2) as u32;
            for x in dc.iter_mut() {
                *x = ((*x as i64 * v) << shift) as i32;
            }
        } else {
            let shift = (2 - qp6) as u32;
            let round = 1i64 << (shift - 1);
            for x in dc.iter_mut() {
                *x = ((*x as i64 * v + round) >> shift) as i32;
            }
        }
    } else {
        let w = weight_scale_00 as i64;
        if qp6 >= 6 {
            let shift = (qp6 - 6) as u32;
            for x in dc.iter_mut() {
                *x = ((*x as i64 * w * v) << shift) as i32;
            }
        } else {
            let shift = (6 - qp6) as u32;
            let round = 1i64 << (shift - 1);
            for x in dc.iter_mut() {
                *x = ((*x as i64 * w * v + round) >> shift) as i32;
            }
        }
    }
}

/// Extended-QP variant of [`inv_hadamard_2x2_chroma_dc_scaled`].
pub fn inv_hadamard_2x2_chroma_dc_scaled_ext(dc: &mut [i32; 4], qp: i32, weight_scale_00: i16) {
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

    let qp = qp.max(0);
    let qp6 = (qp / 6) as u32;
    let qmod = (qp % 6) as usize;
    let v = V_TABLE[qmod][0] as i64;
    if weight_scale_00 == 16 {
        if qp >= 6 {
            let shift = qp6 - 1;
            for x in dc.iter_mut() {
                *x = ((*x as i64 * v) << shift) as i32;
            }
        } else {
            for x in dc.iter_mut() {
                *x = ((*x as i64 * v) >> 1) as i32;
            }
        }
    } else {
        let w = weight_scale_00 as i64;
        if qp >= 30 {
            let shift = (qp6 - 5) as u32;
            for x in dc.iter_mut() {
                *x = ((*x as i64 * w * v) << shift) as i32;
            }
        } else {
            let shift = (5 - qp6) as u32;
            let round = 1i64 << (shift - 1);
            for x in dc.iter_mut() {
                *x = ((*x as i64 * w * v + round) >> shift) as i32;
            }
        }
    }
}

/// Map a QP expressed in the nominal range `[-QpBdOffsetC..=51]` to
/// a chroma QP via Table 7-2 (§7.4.2.1). For values below 30 the
/// function is identity; above 30 it runs through `QP_CHROMA_TABLE`.
/// QpBdOffsetC is applied by the caller after the table lookup (see
/// [`crate::mb_hi`]).
pub fn chroma_qp_hi(qpi: i32) -> i32 {
    if qpi < 0 {
        qpi
    } else if qpi <= 51 {
        QP_CHROMA_TABLE[qpi as usize]
    } else {
        qpi
    }
}

/// Inverse 4×4 Hadamard for Intra16×16 DC luma block — §8.5.10.
///
/// Input/output: 16 DC coefficients in 4×4 raster (one per 4×4 sub-block of
/// the macroblock). After the Hadamard the values are dequantised with the
/// `(0,0)` LevelScale entry, and the result populates the DC slot of each
/// 4×4 luma residual block before its own inverse transform.
pub fn inv_hadamard_4x4_dc(dc: &mut [i32; 16], qp: i32) {
    inv_hadamard_4x4_dc_scaled(dc, qp, 16);
}

/// Inverse 4×4 Hadamard + dequant with a custom DC weight.
/// `weight_scale_00` is `weightScale4x4[0,0]` for the active Intra-Y
/// 4×4 scaling list (slot 0). The flat-16 weight matches
/// [`inv_hadamard_4x4_dc`] bit-exactly because the factor-of-16 is
/// baked into the `qp/6 - 2` shift (spec uses a +4 shift plus a /16
/// absorbed via the IDCT normalisation).
pub fn inv_hadamard_4x4_dc_scaled(dc: &mut [i32; 16], qp: i32, weight_scale_00: i16) {
    // Hadamard in two passes (rows then columns).
    let mut tmp = [0i32; 16];
    for r in 0..4 {
        let a = dc[r * 4];
        let b = dc[r * 4 + 1];
        let c = dc[r * 4 + 2];
        let d = dc[r * 4 + 3];
        tmp[r * 4] = a + b + c + d;
        tmp[r * 4 + 1] = a + b - c - d;
        tmp[r * 4 + 2] = a - b - c + d;
        tmp[r * 4 + 3] = a - b + c - d;
    }
    for c in 0..4 {
        let a = tmp[c];
        let b = tmp[4 + c];
        let cc = tmp[8 + c];
        let d = tmp[12 + c];
        dc[c] = a + b + cc + d;
        dc[4 + c] = a + b - cc - d;
        dc[8 + c] = a - b - cc + d;
        dc[12 + c] = a - b + cc - d;
    }

    // Dequantise (§8.5.10 Luma_16×16 DC). The spec formula is
    // `dc_ij = (f * w * V + 2^5) >> 6` when qP < 36, and
    // `(f * w * V) << (qP/6 - 6)` when qP >= 36. Our pipeline absorbs
    // a factor of 16 into the final IDCT `>>6` — so for flat weight
    // (w = 16), the shift reduces by 4 (= `qP/6 - 2` left shift or
    // `2 - qP/6` right shift with rounding). The scaled path below
    // generalises by factoring `w` into the intermediate.
    let qp = qp.clamp(0, 51);
    let qp6 = (qp / 6) as i32;
    let qmod = (qp % 6) as usize;
    let v = V_TABLE[qmod][0];
    if weight_scale_00 == 16 {
        if qp6 >= 2 {
            let shift = (qp6 - 2) as u32;
            for x in dc.iter_mut() {
                *x = (*x * v) << shift;
            }
        } else {
            let shift = (2 - qp6) as u32;
            let round = 1i32 << (shift - 1);
            for x in dc.iter_mut() {
                *x = (*x * v + round) >> shift;
            }
        }
    } else {
        // Generic spec-style dequant for the DC luma block with a
        // non-flat weight. Uses the full `(f * w * V + rnd) >> shift`
        // form so the factor of `w` isn't truncated to zero at high
        // QPs or pre-multiplied into the wrong intermediate.
        let w = weight_scale_00 as i32;
        if qp6 >= 6 {
            let shift = (qp6 - 6) as u32;
            for x in dc.iter_mut() {
                *x = (*x * w * v) << shift;
            }
        } else {
            let shift = (6 - qp6) as u32;
            let round = 1i32 << (shift - 1);
            for x in dc.iter_mut() {
                *x = (*x * w * v + round) >> shift;
            }
        }
    }
}

/// 2×4 chroma DC inverse Hadamard + dequant — §8.5.11.2
/// (ChromaArrayType == 2).
///
/// Input/output: 8 DC coefficients in coded order. The 2×4 DC array
/// layout is:
///
/// ```text
///   dc[0] dc[1]
///   dc[2] dc[3]
///   dc[4] dc[5]
///   dc[6] dc[7]
/// ```
///
/// Transform steps (FFmpeg `chroma422_dc_dequant_idct` reference):
///
/// 1. Row-side 2-pt butterfly on each of the 4 rows.
/// 2. 4-pt Hadamard down each of the two resulting columns.
/// 3. Dequantise each output by `w * V << (qp/6 - 2)` (left shift when
///    qP >= 12) or `(w * V + rnd) >> (2 - qp/6)` (right shift
///    otherwise). The 4:2:2 Hadamard's factor-of-8 gain costs 2 more
///    right-shifts than the AC dequant path uses for a flat weight;
///    this keeps the DC slot of the AC block in the same scaled
///    domain as the AC coefficients so `res[0] = dc[i]` + `idct_4x4`
///    produces the spec residual exactly.
pub fn inv_hadamard_2x4_chroma_dc_scaled(dc: &mut [i32; 8], qp: i32, weight_scale_00: i16) {
    let mut tmp = [0i32; 8];
    // Column 4-pt Hadamard first (down each of the 2 columns).
    for c in 0..2 {
        let a = dc[c]; // row 0
        let b = dc[2 + c]; // row 1
        let cc = dc[4 + c]; // row 2
        let d = dc[6 + c]; // row 3
        tmp[c] = a + b + cc + d;
        tmp[2 + c] = a + b - cc - d;
        tmp[4 + c] = a - b - cc + d;
        tmp[6 + c] = a - b + cc - d;
    }
    // Then row 2-pt butterfly.
    for r in 0..4 {
        let a = tmp[r * 2];
        let b = tmp[r * 2 + 1];
        dc[r * 2] = a + b;
        dc[r * 2 + 1] = a - b;
    }

    // Spec §8.5.11.2 / H.264 Amd 2 equations (8-328..8-330):
    //   QP'_C,DC = QP'_C + 3
    //   If QP'_C,DC >= 36:
    //       dcC_ij = (f_ij * LevelScale(QP'_C,DC%6, 0, 0)) << (QP'_C,DC/6 - 6)
    //   Else:
    //       dcC_ij = (f_ij * LevelScale(QP'_C,DC%6, 0, 0) + 2^(5-QP'_C,DC/6))
    //                >> (6 - QP'_C,DC/6)
    //
    // The +3 QP offset here is specific to ChromaArrayType==2 and accounts
    // for the factor-of-2 extra Hadamard gain (2×4 vs 2×2). Without it the
    // DC values come out ~1.38× too small, matching V_TABLE[(q+3)%6][0] /
    // V_TABLE[q%6][0] on typical QPs. LevelScale4x4 in our code folds
    // `weight_scale * V_TABLE[qmod][class]` so the full formula is the
    // same shape as the 2×2 code path, with QP replaced by QP + 3 and one
    // more right-shift absorbed into the shift constants.
    let qp_dc = (qp + 3).clamp(0, 51);
    let qp6 = (qp_dc / 6) as i32;
    let qmod = (qp_dc % 6) as usize;
    let v = V_TABLE[qmod][0];
    let w = weight_scale_00 as i32;
    if qp_dc >= 36 {
        let shift = (qp6 - 6) as u32;
        for x in dc.iter_mut() {
            *x = (*x * w * v) << shift;
        }
    } else {
        let shift = (6 - qp6) as u32;
        let round = 1i32 << (shift - 1);
        for x in dc.iter_mut() {
            *x = (*x * w * v + round) >> shift;
        }
    }
}

/// Extended-QP variant of [`inv_hadamard_2x4_chroma_dc_scaled`]. Accepts
/// `qP` up to `51 + QpBdOffsetC` (so `qp + 3` reaches 66 at 10-bit); uses
/// i64 intermediates so the `(qp_dc/6 - 6)` left shifts at high QP don't
/// overflow the i32 range. Mirrors the 8-bit shift pattern otherwise.
pub fn inv_hadamard_2x4_chroma_dc_scaled_ext(dc: &mut [i32; 8], qp: i32, weight_scale_00: i16) {
    let mut tmp = [0i32; 8];
    for c in 0..2 {
        let a = dc[c];
        let b = dc[2 + c];
        let cc = dc[4 + c];
        let d = dc[6 + c];
        tmp[c] = a + b + cc + d;
        tmp[2 + c] = a + b - cc - d;
        tmp[4 + c] = a - b - cc + d;
        tmp[6 + c] = a - b + cc - d;
    }
    for r in 0..4 {
        let a = tmp[r * 2];
        let b = tmp[r * 2 + 1];
        dc[r * 2] = a + b;
        dc[r * 2 + 1] = a - b;
    }

    let qp_dc = (qp + 3).max(0);
    let qp6 = (qp_dc / 6) as i32;
    let qmod = (qp_dc % 6) as usize;
    let v = V_TABLE[qmod][0] as i64;
    let w = weight_scale_00 as i64;
    if qp_dc >= 36 {
        let shift = (qp6 - 6) as u32;
        for x in dc.iter_mut() {
            *x = ((*x as i64 * w * v) << shift) as i32;
        }
    } else {
        let shift = (6 - qp6) as u32;
        let round = 1i64 << (shift - 1);
        for x in dc.iter_mut() {
            *x = ((*x as i64 * w * v + round) >> shift) as i32;
        }
    }
}

/// 2×2 chroma DC inverse Hadamard — §8.5.11.1.
///
/// Input: 4 DC coefficients in raster (00, 01, 10, 11).
pub fn inv_hadamard_2x2_chroma_dc(dc: &mut [i32; 4], qp: i32) {
    inv_hadamard_2x2_chroma_dc_scaled(dc, qp, 16);
}

/// Inverse 2×2 chroma DC Hadamard + dequant with a custom weight.
/// `weight_scale_00` is the chroma scaling list's (0,0) entry (the
/// DC slot). Flat (16) matches [`inv_hadamard_2x2_chroma_dc`].
pub fn inv_hadamard_2x2_chroma_dc_scaled(dc: &mut [i32; 4], qp: i32, weight_scale_00: i16) {
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

    let qp = qp.clamp(0, 51);
    let qp6 = (qp / 6) as u32;
    let qmod = (qp % 6) as usize;
    let v = V_TABLE[qmod][0];
    if weight_scale_00 == 16 {
        if qp >= 6 {
            let shift = qp6 - 1;
            for x in dc.iter_mut() {
                *x = (*x * v) << shift;
            }
        } else {
            for x in dc.iter_mut() {
                *x = (*x * v) >> 1;
            }
        }
    } else {
        let w = weight_scale_00 as i32;
        // §8.5.11.1 full form: (f * w * V + rnd) >> 5 for qP < 30,
        // shift-left for qP >= 30.
        if qp >= 30 {
            let shift = (qp6 - 5) as u32;
            for x in dc.iter_mut() {
                *x = (*x * w * v) << shift;
            }
        } else {
            let shift = (5 - qp6) as u32;
            let round = 1i32 << (shift - 1);
            for x in dc.iter_mut() {
                *x = (*x * w * v + round) >> shift;
            }
        }
    }
}

/// QP table for chroma — Table 7-2. Maps `qPI` (luma QP after offset) to
/// chroma QP `QPC`. For QPI < 30, output equals input; for 30..=51, use the
/// table.
pub const QP_CHROMA_TABLE: [i32; 52] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39,
    39, 39,
];

/// Map a luma-side QP+offset to the chroma QP per §7.4.2.1.
pub fn chroma_qp(qp_y: i32, chroma_qp_index_offset: i32) -> i32 {
    let qpi = (qp_y + chroma_qp_index_offset).clamp(-12, 51);
    if qpi < 0 {
        qpi
    } else {
        QP_CHROMA_TABLE[qpi as usize]
    }
}

// ---------------------------------------------------------------------------
// 8×8 dequantisation + inverse transform (§8.5.13 — High Profile).
// ---------------------------------------------------------------------------

/// `normAdjust8x8(qP%6, m)` — Table 8-15. Six position classes for the 8×8
/// block, indexed by the lookup table [`V8X8_POS_CLASS`].
const V8X8_TABLE: [[i32; 6]; 6] = [
    [20, 18, 32, 19, 25, 24],
    [22, 19, 35, 21, 28, 26],
    [26, 23, 42, 24, 33, 31],
    [28, 25, 45, 26, 35, 33],
    [32, 28, 51, 30, 40, 38],
    [36, 32, 58, 34, 46, 43],
];

/// 16-entry sub-table mapping `((row & 3) << 2) | (col & 3)` to one of the
/// six position classes of [`V8X8_TABLE`]. Matches FFmpeg's
/// `ff_h264_dequant8_coeff_init_scan`; equivalent to the spec's Table
/// 8-15 class assignment.
const V8X8_POS_CLASS: [usize; 16] = [0, 3, 4, 3, 3, 1, 5, 1, 4, 5, 2, 5, 3, 1, 5, 1];

/// Position class for an (row, col) pair in an 8×8 block.
fn pos_class_8x8(row: usize, col: usize) -> usize {
    V8X8_POS_CLASS[((row & 3) << 2) | (col & 3)]
}

/// Compute the 8×8 scaling factor `LevelScale8x8(qP%6, i, j)` from
/// §8.5.12.1 for a given per-position weight.
///
/// `weight` = 16 corresponds to the flat default scaling list — the
/// high-volume fast path. Custom scaling lists (§7.4.2.1.1 /
/// §7.4.2.2) replace `16` with the per-position weight parsed from
/// the SPS/PPS, resolved against the Table 7-2 fallback chain via
/// [`crate::scaling_list::ScalingLists`].
fn level_scale_8x8(qp_mod6: usize, row: usize, col: usize, weight: i32) -> i32 {
    V8X8_TABLE[qp_mod6][pos_class_8x8(row, col)] * weight
}

/// 8×8 zig-zag scan (frame coding, §8.5.6 Table 8-14). Maps coded-order
/// index 0..=63 → raster index (`row * 8 + col`).
#[rustfmt::skip]
pub const ZIGZAG_8X8: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

/// Dequantise an 8×8 AC block in raster order per §8.5.12.1 (flat scaling
/// list). Follows the reference decoder chain so the combined
/// (dequant + [`idct_8x8`]) matches FFmpeg / JM bit-exactly:
///
/// * `c'[i,j] = (c[i,j] * LevelScale8x8(qP%6, i, j) + rnd) >> shift`
///   with `shift = 6 - qP/6` and `rnd = 1 << (5 - qP/6)` when qP < 36;
/// * `c'[i,j] = (c[i,j] * LevelScale8x8(qP%6, i, j)) << (qP/6 - 6)` when
///   qP ≥ 36.
///
/// [`idct_8x8`] adds the spec's `32` rounding at DC and applies the final
/// `>> 6` per sample. The two stages together reproduce the spec's
/// `(TransformedResidual + 2^5) >> 6` semantics.
pub fn dequantize_8x8(coeffs: &mut [i32; 64], qp: i32) {
    dequantize_8x8_scaled(coeffs, qp, &crate::scaling_list::FLAT_8X8);
}

/// Dequantise an 8×8 AC block with a custom weight matrix (§7.4.2.1.1
/// / §8.5.12.1). With `weight_scale` = flat-16 this is bit-identical
/// to the non-scaled variant. The generic formula, matching the spec,
/// is `d = (c * V * w + round) >> shift` with `shift = 6 - qP/6`;
/// when qP/6 ≥ 6 the shift sign flips and the right shift becomes a
/// left shift of `qP/6 - 6`. The spec's `*16` for flat lists lives in
/// `w`.
pub fn dequantize_8x8_scaled(coeffs: &mut [i32; 64], qp: i32, weight_scale: &[i16; 64]) {
    let qp = qp.clamp(0, 51);
    let qp6 = (qp / 6) as i32;
    let qmod = (qp % 6) as usize;
    if qp6 >= 6 {
        let shift = (qp6 - 6) as u32;
        for r in 0..8 {
            for c in 0..8 {
                let i = r * 8 + c;
                let w = weight_scale[i] as i32;
                coeffs[i] = (coeffs[i] * level_scale_8x8(qmod, r, c, w)) << shift;
            }
        }
    } else {
        let shift = (6 - qp6) as u32;
        let round = 1i32 << (shift - 1);
        for r in 0..8 {
            for c in 0..8 {
                let i = r * 8 + c;
                let w = weight_scale[i] as i32;
                coeffs[i] = (coeffs[i] * level_scale_8x8(qmod, r, c, w) + round) >> shift;
            }
        }
    }
}

/// Extended-QP variant of [`dequantize_8x8_scaled`] — accepts `qP` up
/// to `51 + QpBdOffset` (§7.4.2.1.1). Widens intermediates to i64 to
/// survive the larger shifts at high QP, mirroring the 4×4
/// [`dequantize_4x4_scaled_ext`]. Used by the 10-bit Intra_8×8 and the
/// 10-bit P/B inter 8×8 residual paths.
pub fn dequantize_8x8_scaled_ext(coeffs: &mut [i32; 64], qp: i32, weight_scale: &[i16; 64]) {
    let qp = qp.max(0);
    let qp6 = (qp / 6) as i32;
    let qmod = (qp % 6) as usize;
    if qp6 >= 6 {
        let shift = (qp6 - 6) as u32;
        for r in 0..8 {
            for c in 0..8 {
                let i = r * 8 + c;
                let w = weight_scale[i] as i64;
                let v = V8X8_TABLE[qmod][pos_class_8x8(r, c)] as i64;
                coeffs[i] = ((coeffs[i] as i64 * v * w) << shift) as i32;
            }
        }
    } else {
        let shift = (6 - qp6) as u32;
        let round = 1i64 << (shift - 1);
        for r in 0..8 {
            for c in 0..8 {
                let i = r * 8 + c;
                let w = weight_scale[i] as i64;
                let v = V8X8_TABLE[qmod][pos_class_8x8(r, c)] as i64;
                coeffs[i] = ((coeffs[i] as i64 * v * w + round) >> shift) as i32;
            }
        }
    }
}

/// Inverse 8×8 integer transform as per §8.5.13.2.
///
/// Input: dequantised coefficient block in raster (`row * 8 + col`) order.
/// Output: residual sample block in raster order, `(h + 32) >> 6`-rounded
/// per sample and ready to be added to the prediction. The rounding is
/// absorbed once via `block[0] += 32` at the start of the two-pass butterfly
/// and applied as `>> 6` at the final stage — matches the JVT reference /
/// FFmpeg's `ff_h264_idct8_add` pipeline.
pub fn idct_8x8(coeffs: &mut [i32; 64]) {
    // The spec rounds after the two-pass transform: (h + 32) >> 6. Adding
    // 32 to the DC upfront is equivalent once both passes have completed
    // (the butterfly below preserves the DC-only constant-shift property).
    coeffs[0] = coeffs[0].wrapping_add(32);

    // Pass 1 — operate columnwise (read block[i + k*8] for k=0..7 down a
    // column, write back into the same column). Matches FFmpeg's first
    // loop where `i` is the column index and `k*8` selects the row.
    for i in 0..8 {
        let c0 = coeffs[i];
        let c1 = coeffs[i + 8];
        let c2 = coeffs[i + 16];
        let c3 = coeffs[i + 24];
        let c4 = coeffs[i + 32];
        let c5 = coeffs[i + 40];
        let c6 = coeffs[i + 48];
        let c7 = coeffs[i + 56];

        let a0 = c0.wrapping_add(c4);
        let a2 = c0.wrapping_sub(c4);
        let a4 = (c2 >> 1).wrapping_sub(c6);
        let a6 = (c6 >> 1).wrapping_add(c2);

        let b0 = a0.wrapping_add(a6);
        let b2 = a2.wrapping_add(a4);
        let b4 = a2.wrapping_sub(a4);
        let b6 = a0.wrapping_sub(a6);

        let a1 = -c3 + c5 - c7 - (c7 >> 1);
        let a3 = c1 + c7 - c3 - (c3 >> 1);
        let a5 = -c1 + c7 + c5 + (c5 >> 1);
        let a7 = c3 + c5 + c1 + (c1 >> 1);

        let b1 = (a7 >> 2) + a1;
        let b3 = a3 + (a5 >> 2);
        let b5 = (a3 >> 2) - a5;
        let b7 = a7 - (a1 >> 2);

        coeffs[i] = b0.wrapping_add(b7);
        coeffs[i + 56] = b0.wrapping_sub(b7);
        coeffs[i + 8] = b2.wrapping_add(b5);
        coeffs[i + 48] = b2.wrapping_sub(b5);
        coeffs[i + 16] = b4.wrapping_add(b3);
        coeffs[i + 40] = b4.wrapping_sub(b3);
        coeffs[i + 24] = b6.wrapping_add(b1);
        coeffs[i + 32] = b6.wrapping_sub(b1);
    }

    // Pass 2 — operate rowwise with final `>> 6`.
    for i in 0..8 {
        let base = i * 8;
        let c0 = coeffs[base];
        let c1 = coeffs[base + 1];
        let c2 = coeffs[base + 2];
        let c3 = coeffs[base + 3];
        let c4 = coeffs[base + 4];
        let c5 = coeffs[base + 5];
        let c6 = coeffs[base + 6];
        let c7 = coeffs[base + 7];

        let a0 = c0.wrapping_add(c4);
        let a2 = c0.wrapping_sub(c4);
        let a4 = (c2 >> 1).wrapping_sub(c6);
        let a6 = (c6 >> 1).wrapping_add(c2);

        let b0 = a0.wrapping_add(a6);
        let b2 = a2.wrapping_add(a4);
        let b4 = a2.wrapping_sub(a4);
        let b6 = a0.wrapping_sub(a6);

        let a1 = -c3 + c5 - c7 - (c7 >> 1);
        let a3 = c1 + c7 - c3 - (c3 >> 1);
        let a5 = -c1 + c7 + c5 + (c5 >> 1);
        let a7 = c3 + c5 + c1 + (c1 >> 1);

        let b1 = (a7 >> 2) + a1;
        let b3 = a3 + (a5 >> 2);
        let b5 = (a3 >> 2) - a5;
        let b7 = a7 - (a1 >> 2);

        coeffs[base] = (b0 + b7) >> 6;
        coeffs[base + 1] = (b2 + b5) >> 6;
        coeffs[base + 2] = (b4 + b3) >> 6;
        coeffs[base + 3] = (b6 + b1) >> 6;
        coeffs[base + 4] = (b6 - b1) >> 6;
        coeffs[base + 5] = (b4 - b3) >> 6;
        coeffs[base + 6] = (b2 - b5) >> 6;
        coeffs[base + 7] = (b0 - b7) >> 6;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct_zero_in_zero_out() {
        let mut z = [0i32; 16];
        idct_4x4(&mut z);
        assert!(z.iter().all(|&v| v == 0));
    }

    #[test]
    fn idct_pure_dc_constant() {
        // A pure DC coefficient should produce a constant block.
        // Using the H.264 4×4 transform conventions, a coefficient of 64 at
        // (0,0) (already dequantised) corresponds to a sample value of 1
        // (the >> 6 gives 1 per pixel after the +32 round).
        let mut c = [0i32; 16];
        c[0] = 64;
        idct_4x4(&mut c);
        // All cells should equal the same value.
        let v0 = c[0];
        for &v in c.iter() {
            assert_eq!(v, v0);
        }
    }

    #[test]
    fn idct8_zero_in_zero_out() {
        let mut z = [0i32; 64];
        idct_8x8(&mut z);
        // DC is increased by 32 up-front then >> 6 at output: 32 >> 6 = 0.
        assert!(z.iter().all(|&v| v == 0));
    }

    #[test]
    fn idct8_pure_dc_constant() {
        // A post-dequant DC coefficient of 64 should recover a flat block of
        // sample value 1 (the two butterfly passes preserve constant, +32
        // rounding aligns after >> 6). Matches FFmpeg's ff_h264_idct8_add
        // output for the same input.
        let mut c = [0i32; 64];
        c[0] = 64;
        idct_8x8(&mut c);
        let v0 = c[0];
        for &v in c.iter() {
            assert_eq!(v, v0, "pure DC should produce a constant block");
        }
        // Constant = ((64 * 1 * 1) + 32) >> 6 = 96 >> 6 = 1 — the 1D transform
        // of a pure-DC impulse is (1,1,...,1) per pass, so the 2D result is
        // (1,1,...,1) * 64.
        assert_eq!(v0, 1);
    }

    #[test]
    fn idct8_linearity_small() {
        // The transform is linear — 2× input = 2× output. Verify on a small
        // random-ish coefficient pattern.
        let mut c1 = [0i32; 64];
        c1[0] = 128;
        c1[1] = 32;
        c1[8] = 16;
        let mut c2 = c1;
        for v in c2.iter_mut() {
            *v *= 2;
        }
        idct_8x8(&mut c1);
        idct_8x8(&mut c2);
        // Because of the +32 rounding offset at DC plus >> 6, scaling isn't
        // strictly 2× element-wise — but the difference must be bounded by
        // 1 per sample. Check that relation holds.
        for i in 0..64 {
            let diff = c2[i] - 2 * c1[i];
            assert!(
                diff.abs() <= 2,
                "non-linear 8×8 IDCT at pos {i}: c1={} c2={}",
                c1[i],
                c2[i]
            );
        }
    }

    /// Flat-16 scaling lists must produce identical output to the
    /// non-scaled path. This locks in the factor-of-16 reorganisation:
    /// with `weightScale = 16`, `V_TABLE * 16 >> 4 = V_TABLE`, so the
    /// custom-matrix dequant path collapses back to the original.
    #[test]
    fn dequantize_4x4_flat_matches_unscaled() {
        use crate::scaling_list::FLAT_4X4;
        let mut a = [1i32; 16];
        a[5] = 3;
        a[9] = -7;
        let mut b = a;
        dequantize_4x4(&mut a, 18);
        dequantize_4x4_scaled(&mut b, 18, &FLAT_4X4);
        assert_eq!(a, b);
    }

    /// Scaled-4×4 vs unscaled: a weight of 32 doubles the effective
    /// level-scale vs flat-16 at each position.
    #[test]
    fn dequantize_4x4_doubled_weights_are_doubled_output() {
        let mut flat = [1i32; 16];
        let mut doubled = [1i32; 16];
        let w = [32i16; 16];
        dequantize_4x4(&mut flat, 6);
        dequantize_4x4_scaled(&mut doubled, 6, &w);
        for i in 0..16 {
            assert_eq!(doubled[i], 2 * flat[i], "mismatch at pos {i}");
        }
    }

    #[test]
    fn dequantize_8x8_flat_matches_unscaled() {
        use crate::scaling_list::FLAT_8X8;
        let mut a = [1i32; 64];
        a[3] = 5;
        a[27] = -4;
        let mut b = a;
        dequantize_8x8(&mut a, 24);
        dequantize_8x8_scaled(&mut b, 24, &FLAT_8X8);
        assert_eq!(a, b);
    }

    #[test]
    fn dequantize_8x8_flat_qp0() {
        // Dequantising a flat-zero block should give flat zero.
        let mut c = [0i32; 64];
        dequantize_8x8(&mut c, 0);
        assert!(c.iter().all(|&v| v == 0));
    }

    #[test]
    fn dequantize_8x8_dc_matches_spec() {
        // At qP = 0 with LevelScale8x8(0, 0, 0) = 20 * 16 = 320, and
        // shift 6, rnd = 32: c' = (1 * 320 + 32) >> 6 = 352 >> 6 = 5.
        let mut c = [0i32; 64];
        c[0] = 1;
        dequantize_8x8(&mut c, 0);
        assert_eq!(c[0], 5, "qp=0 DC dequant should equal 5");
        // At qP = 6 with shift = 0, rnd = 32, c' = (1*320 * 2 + 32) >> 6 but
        // qp=6 hits the qp/6 = 1 branch (shift right 5, rnd 16):
        // c' = (1*320 + 16) >> 5 = 10. (qp/6 == 1, 6 - 1 = 5 shift.)
        let mut c = [0i32; 64];
        c[0] = 1;
        dequantize_8x8(&mut c, 6);
        assert_eq!(c[0], 10);
    }
}
