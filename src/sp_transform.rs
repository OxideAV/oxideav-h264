//! §8.6 — Decoding process for P macroblocks in SP slices or SI
//! macroblocks.
//!
//! SP slices carry Inter-predicted macroblocks whose reconstruction is
//! defined **in the transform domain**: the prediction samples are
//! forward-transformed (eq. 8-415), combined with the transmitted
//! residual coefficients, re-quantised with the slice's QSY / QSC
//! (eq. 7-33 `QSY = 26 + pic_init_qs_minus26 + slice_qs_delta`), and
//! only then run through the ordinary §8.5.12 scaling + inverse
//! transform. This makes the reconstructed samples a function of the
//! *quantised* transform coefficients alone — which is what allows a
//! switching picture (sp_for_switch_flag == 1, §8.6.2) or an SI picture
//! to reproduce a bit-identical reconstruction from a *different*
//! prediction (different reference pictures, or intra prediction).
//!
//! This module holds the pure coefficient-domain math of §8.6:
//!
//! * eq. 8-415 forward 4x4 core transform of prediction samples,
//! * eq. 8-417 `Aij` and eqs. 8-418/8-419 `LevelScale2(m, i, j)`,
//! * §8.6.1 (non-switching: eqs. 8-416, 8-420, 8-424, 8-425 and the
//!   chroma DC chain 8-427..8-431),
//! * §8.6.2 (switching / SI: eqs. 8-432, 8-433, 8-435, 8-437 and the
//!   chroma DC chain 8-439..8-441).
//!
//! The outputs of these helpers are 4x4 coefficient arrays `c` ready
//! for the §8.5.12 "scaling and transformation process for residual
//! 4x4 blocks" invoked with qP = QSY (luma) / QSC (chroma) per
//! eqs. 8-331 / 8-333 (`sMbFlag == 1`), followed by `uij = Clip1(rij)`
//! (eqs. 8-421 / 8-426 / 8-434 / 8-438 — note: the prediction is NOT
//! added again; it already entered through the transform domain).

/// §8.5.9 eq. 8-313 norm-adjust column classes shared by eq. 8-417
/// `Aij` and eq. 8-418 `LevelScale2`: class 0 for
/// (i, j) ∈ {(0,0), (0,2), (2,0), (2,2)}, class 1 for
/// {(1,1), (1,3), (3,1), (3,3)}, class 2 otherwise.
#[inline]
fn coeff_class(i: usize, j: usize) -> usize {
    match (i % 2, j % 2) {
        (0, 0) => 0,
        (1, 1) => 1,
        _ => 2,
    }
}

/// Eq. 8-417 — `Aij`.
#[inline]
fn a_ij(i: usize, j: usize) -> i64 {
    const A: [i64; 3] = [16, 25, 20];
    A[coeff_class(i, j)]
}

/// Eq. 8-419 — the `w` matrix backing `LevelScale2(m, i, j)`
/// (eq. 8-418). Row index = m (qP % 6), column index = coefficient
/// class.
const LEVEL_SCALE2_W: [[i64; 3]; 6] = [
    [13107, 5243, 8066],
    [11916, 4660, 7490],
    [10082, 4194, 6554],
    [9362, 3647, 5825],
    [8192, 3355, 5243],
    [7282, 2893, 4559],
];

/// Eq. 8-418 — `LevelScale2(m, i, j)`.
#[inline]
fn level_scale2(m: usize, i: usize, j: usize) -> i64 {
    LEVEL_SCALE2_W[m][coeff_class(i, j)]
}

/// §8.5.9 eq. 8-313 — `LevelScale4x4(m, i, j)` with a caller-supplied
/// raster weightScale list (flat 16s outside High-profile scaling
/// matrices; SP/SI live in the Extended profile where the SPS/PPS
/// carry no scaling matrices).
#[inline]
fn level_scale4x4(weight_scale: &[i32; 16], m: usize, i: usize, j: usize) -> i64 {
    const NORM_ADJUST: [[i64; 3]; 6] = [
        [10, 16, 13],
        [11, 18, 14],
        [13, 20, 16],
        [14, 23, 18],
        [16, 25, 20],
        [18, 29, 23],
    ];
    weight_scale[i * 4 + j] as i64 * NORM_ADJUST[m][coeff_class(i, j)]
}

/// Eq. 8-415 — forward 4x4 core transform of prediction samples:
/// `cp = Cf · p · CfT` with `Cf` rows
/// `[1 1 1 1; 2 1 −1 −2; 1 −1 −1 1; 1 −2 2 −1]`. Input and output are
/// row-major (`p[i*4+j] = p_ij`).
pub fn forward_core_4x4(p: &[i32; 16]) -> [i32; 16] {
    // Rows: t = Cf · p.
    let mut t = [0i32; 16];
    for j in 0..4 {
        let (a, b, c, d) = (p[j], p[4 + j], p[8 + j], p[12 + j]);
        t[j] = a + b + c + d;
        t[4 + j] = 2 * a + b - c - 2 * d;
        t[8 + j] = a - b - c + d;
        t[12 + j] = a - 2 * b + 2 * c - d;
    }
    // Columns: cp = t · CfT.
    let mut cp = [0i32; 16];
    for i in 0..4 {
        let (a, b, c, d) = (t[i * 4], t[i * 4 + 1], t[i * 4 + 2], t[i * 4 + 3]);
        cp[i * 4] = a + b + c + d;
        cp[i * 4 + 1] = 2 * a + b - c - 2 * d;
        cp[i * 4 + 2] = a - b - c + d;
        cp[i * 4 + 3] = a - 2 * b + 2 * c - d;
    }
    cp
}

/// 2x2 Hadamard sandwich `H · x · H` with `H = [1 1; 1 −1]`
/// (eqs. 8-427 / 8-430). Row-major 2x2 (`x[i*2+j] = x_ij`).
#[inline]
pub fn hadamard_2x2(x: &[i64; 4]) -> [i64; 4] {
    let (a, b, c, d) = (x[0], x[1], x[2], x[3]);
    [a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d]
}

/// Eq. 8-416 dequant term for one coefficient:
/// `( ( cr · LevelScale4x4(qP % 6, i, j) · Aij ) << ( qP / 6 ) ) >> 10`.
#[inline]
fn dequant_8_416(cr: i32, qp: i32, weight_scale: &[i32; 16], i: usize, j: usize) -> i64 {
    let m = (qp % 6) as usize;
    ((cr as i64 * level_scale4x4(weight_scale, m, i, j) * a_ij(i, j)) << (qp / 6)) >> 10
}

/// Eq. 8-420 (luma, sign applied AFTER the shift):
/// `c = Sign(cs) · ( ( Abs(cs) · LevelScale2(QS % 6, i, j)
///                    + ( 1 << ( 14 + QS / 6 ) ) ) >> ( 15 + QS / 6 ) )`.
#[inline]
fn quant_qs_sign_outside(cs: i64, qs: i32, i: usize, j: usize) -> i32 {
    let m = (qs % 6) as usize;
    let shift = 15 + qs / 6;
    let mag = (cs.abs() * level_scale2(m, i, j) + (1i64 << (14 + qs / 6))) >> shift;
    (cs.signum() * mag) as i32
}

/// Eqs. 8-425 / 8-435 (chroma AC, sign applied BEFORE the arithmetic
/// shift): `c = ( Sign(cs) · ( Abs(cs) · LevelScale2(QS % 6, i, j)
///                + ( 1 << ( 14 + QS / 6 ) ) ) ) >> ( 15 + QS / 6 )`.
#[inline]
fn quant_qs_sign_inside(cs: i64, qs: i32, i: usize, j: usize) -> i32 {
    let m = (qs % 6) as usize;
    let shift = 15 + qs / 6;
    ((cs.signum() * (cs.abs() * level_scale2(m, i, j) + (1i64 << (14 + qs / 6)))) >> shift) as i32
}

/// Eqs. 8-429 / 8-439 (chroma DC, one extra bit in the rounding term
/// and the shift; sign applied BEFORE the arithmetic shift):
/// `dc = ( Sign(x) · ( Abs(x) · LevelScale2(QS % 6, 0, 0)
///          + ( 1 << ( 15 + QS / 6 ) ) ) ) >> ( 16 + QS / 6 )`.
#[inline]
fn quant_qs_dc(x: i64, qs: i32) -> i64 {
    let m = (qs % 6) as usize;
    let shift = 16 + qs / 6;
    (x.signum() * (x.abs() * level_scale2(m, 0, 0) + (1i64 << (15 + qs / 6)))) >> shift
}

/// §8.6.1.1 — luma coefficients for one 4x4 block of a P macroblock in
/// an SP slice with `sp_for_switch_flag == 0`.
///
/// * `pred` — the §8.4 Inter prediction samples `p` (row-major 4x4,
///   eq. 8-414).
/// * `cr` — the parsed prediction residual transform coefficients after
///   the §8.5.6 inverse scan (row-major 4x4).
/// * `qp_y` — the current macroblock's QPY (eq. 8-416 dequant of `cr`).
/// * `qs_y` — the slice QSY (eq. 8-420 re-quantisation).
///
/// Returns the 4x4 array `c` to feed §8.5.12 with qP = QSY
/// (eq. 8-331).
pub fn sp_luma_non_switching(
    pred: &[i32; 16],
    cr: &[i32; 16],
    qp_y: i32,
    qs_y: i32,
    weight_scale: &[i32; 16],
) -> [i32; 16] {
    let cp = forward_core_4x4(pred);
    let mut c = [0i32; 16];
    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            // Eq. 8-416.
            let cs = cp[idx] as i64 + dequant_8_416(cr[idx], qp_y, weight_scale, i, j);
            // Eq. 8-420.
            c[idx] = quant_qs_sign_outside(cs, qs_y, i, j);
        }
    }
    c
}

/// §8.6.2.1 — luma coefficients for one 4x4 block of a P macroblock in
/// an SP slice with `sp_for_switch_flag == 1`, or of an SI macroblock
/// (the prediction is then the §8.3 Intra_4x4 prediction).
///
/// Eq. 8-432 quantises the transformed prediction with QSY; eq. 8-433
/// adds the parsed residual coefficients `cr` (inverse-scanned,
/// row-major). Returns `c` for §8.5.12 with qP = QSY.
pub fn sp_luma_switching(pred: &[i32; 16], cr: &[i32; 16], qs_y: i32) -> [i32; 16] {
    let cp = forward_core_4x4(pred);
    let mut c = [0i32; 16];
    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            // Eq. 8-432 (sign outside the shift, like the luma 8-420).
            let cs = quant_qs_sign_outside(cp[idx] as i64, qs_y, i, j);
            // Eq. 8-433.
            c[idx] = cr[idx] + cs;
        }
    }
    c
}

/// §8.6.1.2 — chroma coefficients for one component of a P macroblock
/// in a non-switching SP slice (4:2:0: four 4x4 blocks).
///
/// * `pred` — the 8x8 Inter prediction samples of this chroma
///   component (row-major).
/// * `dc_levels` — parsed `ChromaDCLevel[iCbCr][k]`, k = 0..3.
/// * `ac` — per-4x4-block inverse-scanned residual coefficients
///   (row-major; entry 0 of each block is ignored — the DC comes from
///   the 2x2 chain).
///
/// Returns four 4x4 arrays `c(chroma4x4BlkIdx)` (with `c00` filled per
/// eq. 8-431) ready for §8.5.12 with qP = QSC. §8.5.12.1 leaves chroma
/// `d00 = c00` untouched (eq. 8-335), so the eq. 8-431 pre-scaled DC
/// flows through exactly like the ordinary §8.5.11 chroma DC output.
pub fn sp_chroma_non_switching(
    pred: &[i32; 64],
    dc_levels: &[i32; 4],
    ac: &[[i32; 16]; 4],
    qp_c: i32,
    qs_c: i32,
    weight_scale: &[i32; 16],
) -> [[i32; 16]; 4] {
    // Steps 2..3 — per-block forward transform of the prediction.
    let cp = chroma_pred_transform(pred);

    let mut out = [[0i32; 16]; 4];
    for blk in 0..4 {
        for i in 0..4 {
            for j in 0..4 {
                if i == 0 && j == 0 {
                    continue; // DC handled below.
                }
                let idx = i * 4 + j;
                // Eq. 8-424.
                let cs =
                    cp[blk][idx] as i64 + dequant_8_416(ac[blk][idx], qp_c, weight_scale, i, j);
                // Eq. 8-425.
                out[blk][idx] = quant_qs_sign_inside(cs, qs_c, i, j);
            }
        }
    }

    // Eq. 8-427 — 2x2 transform of the prediction DC coefficients.
    // Matrix element (i, j) is c00p(i * 2 + j).
    let dcp = hadamard_2x2(&[
        cp[0][0] as i64,
        cp[1][0] as i64,
        cp[2][0] as i64,
        cp[3][0] as i64,
    ]);
    let m = (qp_c % 6) as usize;
    let ls_dc_qp = level_scale4x4(weight_scale, m, 0, 0);
    let mut dcr = [0i64; 4];
    for i in 0..2 {
        for j in 0..2 {
            let pos = i * 2 + j;
            // Eq. 8-428 — note the ChromaDCLevel index is j * 2 + i and
            // the shift is >> 9 (one less than the AC's >> 10 to account
            // for the un-normalised 2x2 Hadamard).
            let level = dc_levels[j * 2 + i] as i64;
            let dcs = dcp[pos] + (((level * ls_dc_qp * a_ij(0, 0)) << (qp_c / 6)) >> 9);
            // Eq. 8-429.
            dcr[pos] = quant_qs_dc(dcs, qs_c);
        }
    }
    // Eq. 8-430.
    let f = hadamard_2x2(&dcr);
    // Eq. 8-431 — scale and distribute to the per-block DC slots.
    let ls_dc_qs = level_scale4x4(weight_scale, (qs_c % 6) as usize, 0, 0);
    for i in 0..2 {
        for j in 0..2 {
            out[j * 2 + i][0] = (((f[i * 2 + j] * ls_dc_qs) << (qs_c / 6)) >> 5) as i32;
        }
    }
    out
}

/// §8.6.2.2 — chroma coefficients for one component of a P macroblock
/// in a switching SP slice, or of an SI macroblock (4:2:0).
///
/// Same inputs as [`sp_chroma_non_switching`]. Returns four 4x4 `c`
/// arrays for §8.5.12 with qP = QSC; per eqs. 8-440/8-441 the DC slot
/// carries `f` verbatim (no eq. 8-431 scaling), which §8.5.12.1
/// preserves (eq. 8-335).
pub fn sp_chroma_switching(
    pred: &[i32; 64],
    dc_levels: &[i32; 4],
    ac: &[[i32; 16]; 4],
    qs_c: i32,
) -> [[i32; 16]; 4] {
    let cp = chroma_pred_transform(pred);

    let mut out = [[0i32; 16]; 4];
    for blk in 0..4 {
        for i in 0..4 {
            for j in 0..4 {
                if i == 0 && j == 0 {
                    continue;
                }
                let idx = i * 4 + j;
                // Eq. 8-435.
                let cs = quant_qs_sign_inside(cp[blk][idx] as i64, qs_c, i, j);
                // Eq. 8-437.
                out[blk][idx] = ac[blk][idx] + cs;
            }
        }
    }

    // Eq. 8-427 on the prediction DCs.
    let dcp = hadamard_2x2(&[
        cp[0][0] as i64,
        cp[1][0] as i64,
        cp[2][0] as i64,
        cp[3][0] as i64,
    ]);
    let mut dcr = [0i64; 4];
    for i in 0..2 {
        for j in 0..2 {
            let pos = i * 2 + j;
            // Eq. 8-439.
            let dcs = quant_qs_dc(dcp[pos], qs_c);
            // Eq. 8-440 — ChromaDCLevel index j * 2 + i.
            dcr[pos] = dcs + dc_levels[j * 2 + i] as i64;
        }
    }
    // Eq. 8-430.
    let f = hadamard_2x2(&dcr);
    // Eq. 8-441 — copied verbatim into the DC slots.
    for i in 0..2 {
        for j in 0..2 {
            out[j * 2 + i][0] = f[i * 2 + j] as i32;
        }
    }
    out
}

/// Steps shared by both chroma paths: split the 8x8 component
/// prediction into the four 4x4 blocks (§6.4.7 raster order) and apply
/// eq. 8-415 to each.
fn chroma_pred_transform(pred: &[i32; 64]) -> [[i32; 16]; 4] {
    let mut cp = [[0i32; 16]; 4];
    for (blk, dst) in cp.iter_mut().enumerate() {
        let (bx, by) = ((blk % 2) * 4, (blk / 2) * 4);
        let mut p = [0i32; 16];
        for y in 0..4 {
            for x in 0..4 {
                p[y * 4 + x] = pred[(by + y) * 8 + bx + x];
            }
        }
        *dst = forward_core_4x4(&p);
    }
    cp
}

// ---------------------------------------------------------------------------
// Encoder-side §8.6 helpers (non-normative): forward quantisers that
// invert the eq. 8-416 / 8-428 dequant terms so an SP encoder can pick
// residual levels approximating a target in the transform domain.
// ---------------------------------------------------------------------------

/// Encoder choice (non-normative): the residual level whose eq. 8-416
/// dequant best approximates `target` at the given qP.
pub fn encoder_quant_8_416(
    target: i64,
    qp: i32,
    weight_scale: &[i32; 16],
    i: usize,
    j: usize,
) -> i32 {
    let m = (qp % 6) as usize;
    let den = (level_scale4x4(weight_scale, m, i, j) * a_ij(i, j)) << (qp / 6);
    // round(target * 2^10 / den) with round-half-away-from-zero.
    let num = target << 10;
    let r = if num >= 0 {
        (num + den / 2) / den
    } else {
        (num - den / 2) / den
    };
    r as i32
}

/// Encoder choice (non-normative): the chroma DC level whose eq. 8-428
/// dequant best approximates `target` at the given qP.
pub fn encoder_quant_8_428(target: i64, qp_c: i32, weight_scale: &[i32; 16]) -> i32 {
    let m = (qp_c % 6) as usize;
    let den = (level_scale4x4(weight_scale, m, 0, 0) * a_ij(0, 0)) << (qp_c / 6);
    let num = target << 9;
    let r = if num >= 0 {
        (num + den / 2) / den
    } else {
        (num - den / 2) / den
    };
    r as i32
}

/// Eq. 8-431 as a per-element integer scale: with `QSC >= 6` the
/// operation `((f · LevelScale4x4(QSC % 6, 0, 0)) << (QSC / 6)) >> 5`
/// is exactly `f · k` with
/// `k = LevelScale4x4 · 2^(QSC / 6) / 32` (an integer because
/// LevelScale4x4(m, 0, 0) is a multiple of 16 for flat weightScale and
/// `QSC / 6 >= 1`). A switching-picture encoder uses this to reproduce
/// a non-switching picture's chroma DC reconstruction exactly
/// (`dcr_switch = k · dcr_primary`). Returns `None` when the scale is
/// not an exact integer (QSC < 6 with an odd norm-adjust product).
pub fn chroma_dc_switch_scale(qs_c: i32, weight_scale: &[i32; 16]) -> Option<i64> {
    let ls = level_scale4x4(weight_scale, (qs_c % 6) as usize, 0, 0);
    let num = ls << (qs_c / 6);
    if num % 32 == 0 {
        Some(num / 32)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLAT: [i32; 16] = [16; 16];

    /// Eq. 8-415 on a constant block: only the DC survives, equal to
    /// 16 · v (sum over the 4x4 after both 1-D passes).
    #[test]
    fn forward_core_dc_of_constant_block() {
        let p = [7i32; 16];
        let cp = forward_core_4x4(&p);
        assert_eq!(cp[0], 16 * 7);
        assert!(cp[1..].iter().all(|&v| v == 0));
    }

    /// The eq. 8-415 core matches the §8.5.12.2 inverse: pushing a
    /// quantised version of T(p) through the §8.5.12 chain at the same
    /// qP reproduces p to within the quantisation error bound.
    #[test]
    fn sp_switching_luma_roundtrip_close_to_prediction() {
        // A gradient prediction block.
        let mut pred = [0i32; 16];
        for (k, v) in pred.iter_mut().enumerate() {
            *v = 60 + (k as i32) * 3;
        }
        let qs_y = 24;
        let c = sp_luma_switching(&pred, &[0i32; 16], qs_y);
        let r = crate::transform::inverse_transform_4x4(&c, qs_y, &FLAT, 8).unwrap();
        for k in 0..16 {
            let err = (r[k] - pred[k]).abs();
            assert!(
                err <= 6,
                "blk[{k}]: recon {} vs pred {} (err {err})",
                r[k],
                pred[k]
            );
        }
    }

    /// §8.6.1.1 with an all-zero residual still re-quantises the
    /// prediction (an SP P_Skip is NOT a sample-domain copy).
    #[test]
    fn sp_non_switching_zero_residual_requantises() {
        let mut pred = [0i32; 16];
        for (k, v) in pred.iter_mut().enumerate() {
            *v = 100 + ((k as i32 * 37) % 23);
        }
        let qp_y = 28;
        let qs_y = 28;
        let c = sp_luma_non_switching(&pred, &[0i32; 16], qp_y, qs_y, &FLAT);
        let r = crate::transform::inverse_transform_4x4(&c, qs_y, &FLAT, 8).unwrap();
        // The output is the QS-quantised prediction, not the original.
        let max_err = (0..16).map(|k| (r[k] - pred[k]).abs()).max().unwrap();
        assert!(max_err > 0, "QS=28 must not be lossless on this block");
        assert!(max_err <= 12, "but must stay within quantiser range");
    }

    /// The defining SP property at coefficient level: a switching block
    /// with `cr = c_target − cs` reproduces the target coefficients
    /// exactly, whatever the prediction.
    #[test]
    fn switching_identity_luma_exact() {
        let mut pred_a = [0i32; 16];
        let mut pred_b = [0i32; 16];
        for k in 0..16 {
            pred_a[k] = 90 + ((k as i32 * 17) % 31);
            pred_b[k] = 15 + ((k as i32 * 29) % 61);
        }
        let qp_y = 27;
        let qs_y = 25;
        // Primary SP block (stream A): some residual.
        let mut cr_a = [0i32; 16];
        cr_a[0] = 3;
        cr_a[5] = -2;
        let c_target = sp_luma_non_switching(&pred_a, &cr_a, qp_y, qs_y, &FLAT);

        // Switching block predicted from unrelated pred_b: choose
        // cr = c_target − Q_QS(T(pred_b)).
        let cs_b = sp_luma_switching(&pred_b, &[0i32; 16], qs_y);
        let mut cr_sw = [0i32; 16];
        for k in 0..16 {
            cr_sw[k] = c_target[k] - cs_b[k];
        }
        let c_sw = sp_luma_switching(&pred_b, &cr_sw, qs_y);
        assert_eq!(c_sw, c_target);
    }

    /// Chroma DC switching identity: with QSC >= 6 the eq. 8-431 scale
    /// is an exact integer `k`, and `dcr_switch = k · dcr_primary`
    /// reproduces the primary's scaled DC verbatim through the
    /// eq. 8-430/8-441 path.
    #[test]
    fn switching_identity_chroma_dc_exact() {
        let mut pred_a = [0i32; 64];
        let mut pred_b = [0i32; 64];
        for k in 0..64 {
            pred_a[k] = 120 + ((k as i32 * 13) % 17);
            pred_b[k] = 40 + ((k as i32 * 7) % 29);
        }
        let (qp_c, qs_c) = (28, 26);
        let dc_a = [5, -3, 2, 0];
        let ac_zero = [[0i32; 16]; 4];
        let c_a = sp_chroma_non_switching(&pred_a, &dc_a, &ac_zero, qp_c, qs_c, &FLAT);

        let k = chroma_dc_switch_scale(qs_c, &FLAT).expect("QSC >= 6 scale is integral");

        // Recover the primary's dcr from its c00 values: c00 = k · f,
        // f = H2(dcr) ⇒ H2(f) = 4 · dcr (H2 is 2·orthogonal).
        let f_scaled = [
            c_a[0][0] as i64,
            c_a[1][0] as i64,
            c_a[2][0] as i64,
            c_a[3][0] as i64,
        ];
        // f_scaled[blk = j*2+i] = k · f[i*2+j] — undo the transposed
        // distribution to get k·f in matrix order.
        let kf = [f_scaled[0], f_scaled[2], f_scaled[1], f_scaled[3]];
        let h_kf = hadamard_2x2(&kf); // = 4k · dcr (matrix order)
        assert!(h_kf.iter().all(|v| v % 4 == 0));
        let target_dcr = [h_kf[0] / 4, h_kf[1] / 4, h_kf[2] / 4, h_kf[3] / 4];
        assert!(target_dcr.iter().all(|v| v % k == 0 || k == 1));

        // Switching side: dc level = target_dcr − dcs(pred_b).
        let cp_b = chroma_pred_transform(&pred_b);
        let dcp_b = hadamard_2x2(&[
            cp_b[0][0] as i64,
            cp_b[1][0] as i64,
            cp_b[2][0] as i64,
            cp_b[3][0] as i64,
        ]);
        let mut dc_levels_sw = [0i32; 4];
        for i in 0..2 {
            for j in 0..2 {
                let pos = i * 2 + j;
                let dcs = quant_qs_dc(dcp_b[pos], qs_c);
                dc_levels_sw[j * 2 + i] = (target_dcr[pos] - dcs) as i32;
            }
        }
        let c_sw = sp_chroma_switching(&pred_b, &dc_levels_sw, &ac_zero, qs_c);
        // The switching DC slots must equal the primary's scaled DCs.
        for blk in 0..4 {
            assert_eq!(c_sw[blk][0], c_a[blk][0], "blk {blk} DC");
        }
    }

    /// Encoder forward quant inverts eq. 8-416 within one step.
    #[test]
    fn encoder_quant_8_416_inverts_dequant() {
        for qp in [10, 22, 30, 40] {
            for target in [-2000i64, -37, 0, 5, 419, 3000] {
                for (i, j) in [(0, 0), (1, 1), (2, 3)] {
                    let level = encoder_quant_8_416(target, qp, &FLAT, i, j);
                    let back = dequant_8_416(level, qp, &FLAT, i, j);
                    let den =
                        (level_scale4x4(&FLAT, (qp % 6) as usize, i, j) * a_ij(i, j)) << (qp / 6);
                    let step = den >> 10;
                    assert!(
                        (back - target).abs() <= step.max(1),
                        "qp {qp} target {target} ({i},{j}): back {back} step {step}"
                    );
                }
            }
        }
    }
}
