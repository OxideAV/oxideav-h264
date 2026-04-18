//! Motion-vector prediction and motion compensation for P-slices —
//! ITU-T H.264 §8.4.
//!
//! This module covers the baseline-profile subset:
//!
//! * `mv_l0` prediction from 4×4 neighbour blocks (§8.4.1.3), including the
//!   special cases for the P_L0_16×16 / 16×8 / 8×16 partition shapes.
//! * `P_Skip` motion-vector derivation (§8.4.1.1).
//! * Luma motion compensation: integer + 6-tap half-pel + bilinear
//!   quarter-pel (§8.4.2.2.1).
//! * Chroma motion compensation: bilinear 1/8-pel (§8.4.2.2.2).
//! * Explicit weighted prediction (§8.4.2.3.2) for single-list-0 samples
//!   applied on top of the interpolated predictor. B-slice bi-prediction
//!   and implicit weighted bi-prediction (§8.4.2.3.3) are out of scope.
//!
//! Motion vectors use quarter-pel units for luma (spec convention); chroma
//! MVs are derived from luma / 2 to produce 1/8-pel chroma offsets.

use crate::picture::Picture;
use crate::slice::{ChromaWeight, LumaWeight};

/// 4×4-block-local neighbour lookup: returns (mv_x, mv_y, ref_idx) for the
/// block at position `(br_row, br_col)` of macroblock `(mb_x, mb_y)`.
///
/// `Some((mv, ref_idx))` when the neighbour is available and inter-coded.
/// `None` when the neighbour doesn't exist (picture edge) or is intra.
pub fn neighbour_mv(
    pic: &Picture,
    mb_x: i32,
    mb_y: i32,
    br_row: i32,
    br_col: i32,
) -> Option<((i16, i16), i8)> {
    // Walk MB coordinates first, then sub-block position within that MB.
    let mut mbx = mb_x;
    let mut mby = mb_y;
    let mut row = br_row;
    let mut col = br_col;
    while col < 0 {
        mbx -= 1;
        col += 4;
    }
    while col >= 4 {
        mbx += 1;
        col -= 4;
    }
    while row < 0 {
        mby -= 1;
        row += 4;
    }
    while row >= 4 {
        mby += 1;
        row -= 4;
    }
    if mbx < 0 || mby < 0 || mbx as u32 >= pic.mb_width || mby as u32 >= pic.mb_height {
        return None;
    }
    let info = pic.mb_info_at(mbx as u32, mby as u32);
    if !info.coded || info.intra {
        return None;
    }
    let idx = (row * 4 + col) as usize;
    let ref_idx = info.ref_idx_l0[idx];
    if ref_idx < 0 {
        return None;
    }
    Some((info.mv_l0[idx], ref_idx))
}

/// Predict `mv_l0` for a partition whose top-left 4×4 block is
/// `(br_row, br_col)` and whose size is (`part_h`, `part_w`) blocks
/// (each unit = 4 samples).
///
/// Implements the §8.4.1.3.1 median over neighbours A (left), B (top),
/// C (top-right), with the partition-specific single-predictor special
/// cases in §8.4.1.3:
///
/// * P_L0_16×8 top: use B if B.ref matches; P_L0_16×8 bottom: use A if A.ref
///   matches.
/// * P_L0_8×16 left: use A if A.ref matches; right: use C if C.ref matches.
/// * Otherwise: median of the three components.
pub fn predict_mv_l0(
    pic: &Picture,
    mb_x: u32,
    mb_y: u32,
    br_row: usize,
    br_col: usize,
    part_h: usize,
    part_w: usize,
    ref_idx: i8,
) -> (i16, i16) {
    let mbx = mb_x as i32;
    let mby = mb_y as i32;
    let br_row = br_row as i32;
    let br_col = br_col as i32;

    // A — block to the left of the partition's top-left.
    let a = neighbour_mv(pic, mbx, mby, br_row, br_col - 1);
    // B — block just above the partition's top-left.
    let b = neighbour_mv(pic, mbx, mby, br_row - 1, br_col);
    // C — block above-right of the partition's top-right.
    let tr_col = br_col + part_w as i32;
    let tr_row = br_row - 1;
    let mut c = neighbour_mv(pic, mbx, mby, tr_row, tr_col);
    if c.is_none() {
        // Fall back to D (above-left of the partition's top-left).
        c = neighbour_mv(pic, mbx, mby, br_row - 1, br_col - 1);
    }

    // §8.4.1.3 partition-specific special cases — only when exactly one of
    // A/B/C shares `ref_idx`; otherwise fall through to median.
    if part_h == 2 && part_w == 4 {
        // P_L0_16×8. Top half (br_row == 0) → use B; bottom (br_row == 2) → use A.
        if br_row == 0 {
            if let Some((mv, r)) = b {
                if r == ref_idx {
                    return mv;
                }
            }
        } else if br_row == 2 {
            if let Some((mv, r)) = a {
                if r == ref_idx {
                    return mv;
                }
            }
        }
    } else if part_h == 4 && part_w == 2 {
        // P_L0_8×16. Left half (br_col == 0) → use A; right (br_col == 2) → use C.
        if br_col == 0 {
            if let Some((mv, r)) = a {
                if r == ref_idx {
                    return mv;
                }
            }
        } else if br_col == 2 {
            if let Some((mv, r)) = c {
                if r == ref_idx {
                    return mv;
                }
            }
        }
    }

    // Median rule:
    //  * If exactly one of A/B/C matches ref_idx, use its MV.
    //  * Otherwise median(MV_A, MV_B, MV_C) component-wise, treating
    //    unavailable neighbours as zero.
    let matches_a = a.map(|(_, r)| r == ref_idx).unwrap_or(false);
    let matches_b = b.map(|(_, r)| r == ref_idx).unwrap_or(false);
    let matches_c = c.map(|(_, r)| r == ref_idx).unwrap_or(false);
    let only_matches = [matches_a, matches_b, matches_c]
        .iter()
        .filter(|&&m| m)
        .count();

    // Spec §8.4.1.3.1 first-row availability oddity: when B and C are both
    // unavailable but A is available, use A directly (not median).
    if b.is_none() && c.is_none() && a.is_some() {
        let mv = a.unwrap().0;
        return mv;
    }

    if only_matches == 1 {
        if matches_a {
            return a.unwrap().0;
        }
        if matches_b {
            return b.unwrap().0;
        }
        return c.unwrap().0;
    }

    let ax = a.map(|(mv, _)| mv.0).unwrap_or(0);
    let ay = a.map(|(mv, _)| mv.1).unwrap_or(0);
    let bx = b.map(|(mv, _)| mv.0).unwrap_or(0);
    let by = b.map(|(mv, _)| mv.1).unwrap_or(0);
    let cx = c.map(|(mv, _)| mv.0).unwrap_or(0);
    let cy = c.map(|(mv, _)| mv.1).unwrap_or(0);
    (median3(ax, bx, cx), median3(ay, by, cy))
}

/// P_Skip MV derivation — §8.4.1.1. Returns `(0, 0)` when either the A or B
/// neighbour is unavailable / has ref_idx != 0 / has mv == 0. Otherwise
/// returns the standard P16×16 median predictor with `ref_idx = 0`.
pub fn predict_mv_pskip(pic: &Picture, mb_x: u32, mb_y: u32) -> (i16, i16) {
    let a = neighbour_mv(pic, mb_x as i32, mb_y as i32, 0, -1);
    let b = neighbour_mv(pic, mb_x as i32, mb_y as i32, -1, 0);
    let trivial_a = match a {
        None => true,
        Some(((0, 0), 0)) => true,
        _ => false,
    };
    let trivial_b = match b {
        None => true,
        Some(((0, 0), 0)) => true,
        _ => false,
    };
    if trivial_a || trivial_b {
        return (0, 0);
    }
    predict_mv_l0(pic, mb_x, mb_y, 0, 0, 4, 4, 0)
}

fn median3(a: i16, b: i16, c: i16) -> i16 {
    // Median without extra allocations.
    a.max(b).min(c).max(a.min(b))
}

// ---------------------------------------------------------------------------
// Motion compensation — §8.4.2.
// ---------------------------------------------------------------------------

/// 6-tap half-pel luma filter (§8.4.2.2.1): `(1, -5, 20, 20, -5, 1) / 32`.
#[inline]
fn luma_tap6(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
    a - 5 * b + 20 * c + 20 * d - 5 * e + f
}

/// Clamp integer-pel coordinate to the picture edges (§8.4.2.2.2.1 — the
/// 6-tap taps are allowed to read outside the picture, with samples
/// replicated from the nearest edge).
#[inline]
fn clamp_ref(x: i32, w: i32) -> usize {
    x.clamp(0, w - 1) as usize
}

/// Motion-compensate a rectangular luma region into `dst` (raster order,
/// row-major, stride = `part_w`) at (mv_x_q, mv_y_q) quarter-pel offset
/// from (`base_x`, `base_y`) within the reference picture `ref_pic`.
///
/// `part_w` / `part_h` are in samples.
pub fn luma_mc(
    dst: &mut [u8],
    ref_pic: &Picture,
    base_x: i32,
    base_y: i32,
    mv_x_q: i32,
    mv_y_q: i32,
    part_w: usize,
    part_h: usize,
) {
    let fx = mv_x_q & 3;
    let fy = mv_y_q & 3;
    let int_x = base_x + (mv_x_q >> 2);
    let int_y = base_y + (mv_y_q >> 2);
    let w = ref_pic.width as i32;
    let h = ref_pic.height as i32;
    let stride = ref_pic.luma_stride();
    let src = &ref_pic.y;
    let get = |y: i32, x: i32| -> i32 { src[clamp_ref(y, h) * stride + clamp_ref(x, w)] as i32 };

    // Compute interpolated samples at (int_x + fx/4, int_y + fy/4) for each
    // output position. Implements the §8.4.2.2.1 quarter-pel subset by
    // falling through to the appropriate 6-tap / bilinear combination.
    //
    // H-half = 6-tap filter in the horizontal direction.
    // V-half = 6-tap filter in the vertical direction.
    // J (center-half) = 6-tap of the H-half intermediates.
    let half_h = |y: i32, x: i32| -> i32 {
        luma_tap6(
            get(y, x - 2),
            get(y, x - 1),
            get(y, x),
            get(y, x + 1),
            get(y, x + 2),
            get(y, x + 3),
        )
    };
    let half_v = |y: i32, x: i32| -> i32 {
        luma_tap6(
            get(y - 2, x),
            get(y - 1, x),
            get(y, x),
            get(y + 1, x),
            get(y + 2, x),
            get(y + 3, x),
        )
    };

    // Intermediate 6-tap on the H-half row (pre-rounding): values of
    // half-samples at (x+1/2, y-2..y+3). Use to compute the centre half
    // (J) or to bilinear-mix against a V-half for quarter positions.
    let half_h_raw = |y: i32, x: i32| -> i32 {
        luma_tap6(
            get(y, x - 2),
            get(y, x - 1),
            get(y, x),
            get(y, x + 1),
            get(y, x + 2),
            get(y, x + 3),
        )
    };
    let center_j = |y: i32, x: i32| -> i32 {
        // Six half_h_raw rows, then 6-tap vertically.
        let r0 = half_h_raw(y - 2, x);
        let r1 = half_h_raw(y - 1, x);
        let r2 = half_h_raw(y, x);
        let r3 = half_h_raw(y + 1, x);
        let r4 = half_h_raw(y + 2, x);
        let r5 = half_h_raw(y + 3, x);
        luma_tap6(r0, r1, r2, r3, r4, r5)
    };

    let round_shift = |v: i32, shift: u32| -> u8 {
        let bias = 1 << (shift - 1);
        ((v + bias) >> shift).clamp(0, 255) as u8
    };

    for r in 0..part_h as i32 {
        for c in 0..part_w as i32 {
            let y = int_y + r;
            let x = int_x + c;
            let out: u8 = match (fx, fy) {
                (0, 0) => get(y, x) as u8,
                // Horizontal-only positions.
                (1, 0) => {
                    // a = round((G + b + 1) / 2) where b = half-sample.
                    let g = get(y, x);
                    let b = round_shift(half_h(y, x), 5) as i32;
                    ((g + b + 1) >> 1) as u8
                }
                (2, 0) => round_shift(half_h(y, x), 5),
                (3, 0) => {
                    let g = get(y, x + 1);
                    let b = round_shift(half_h(y, x), 5) as i32;
                    ((g + b + 1) >> 1) as u8
                }
                // Vertical-only positions.
                (0, 1) => {
                    let g = get(y, x);
                    let h = round_shift(half_v(y, x), 5) as i32;
                    ((g + h + 1) >> 1) as u8
                }
                (0, 2) => round_shift(half_v(y, x), 5),
                (0, 3) => {
                    let g = get(y + 1, x);
                    let h = round_shift(half_v(y, x), 5) as i32;
                    ((g + h + 1) >> 1) as u8
                }
                // Diagonal half-half-center mixes.
                (2, 2) => round_shift(center_j(y, x), 10),
                (1, 1) => {
                    let b = round_shift(half_h(y, x), 5) as i32;
                    let h = round_shift(half_v(y, x), 5) as i32;
                    ((b + h + 1) >> 1) as u8
                }
                (3, 1) => {
                    let b = round_shift(half_h(y, x), 5) as i32;
                    let h = round_shift(half_v(y, x + 1), 5) as i32;
                    ((b + h + 1) >> 1) as u8
                }
                (1, 3) => {
                    let b = round_shift(half_h(y + 1, x), 5) as i32;
                    let h = round_shift(half_v(y, x), 5) as i32;
                    ((b + h + 1) >> 1) as u8
                }
                (3, 3) => {
                    let b = round_shift(half_h(y + 1, x), 5) as i32;
                    let h = round_shift(half_v(y, x + 1), 5) as i32;
                    ((b + h + 1) >> 1) as u8
                }
                // Half-horizontal / quarter-vertical (and vice versa).
                (2, 1) => {
                    let b = round_shift(half_h(y, x), 5) as i32;
                    let j = round_shift(center_j(y, x), 10) as i32;
                    ((b + j + 1) >> 1) as u8
                }
                (2, 3) => {
                    let b = round_shift(half_h(y + 1, x), 5) as i32;
                    let j = round_shift(center_j(y, x), 10) as i32;
                    ((b + j + 1) >> 1) as u8
                }
                (1, 2) => {
                    let h = round_shift(half_v(y, x), 5) as i32;
                    let j = round_shift(center_j(y, x), 10) as i32;
                    ((h + j + 1) >> 1) as u8
                }
                (3, 2) => {
                    let h = round_shift(half_v(y, x + 1), 5) as i32;
                    let j = round_shift(center_j(y, x), 10) as i32;
                    ((h + j + 1) >> 1) as u8
                }
                _ => unreachable!(),
            };
            dst[r as usize * part_w + c as usize] = out;
        }
    }
}

/// Motion-compensate a rectangular chroma region (Cb or Cr) into `dst`.
/// Chroma uses bilinear 1/8-pel interpolation (§8.4.2.2.2).
///
/// `base_x`/`base_y` are in chroma-plane samples; `mv_x_q`/`mv_y_q` are
/// *luma* quarter-pel vectors — the spec halves them for 4:2:0 chroma to
/// produce 1/8-pel fractions.
pub fn chroma_mc(
    dst: &mut [u8],
    ref_plane: &[u8],
    plane_stride: usize,
    plane_w: i32,
    plane_h: i32,
    base_x: i32,
    base_y: i32,
    mv_x_q: i32,
    mv_y_q: i32,
    part_w: usize,
    part_h: usize,
) {
    // 1/8-pel chroma offsets (mv is quarter-pel luma; chroma divides by 2).
    let dx = mv_x_q & 7; // 0..7
    let dy = mv_y_q & 7;
    let int_x = base_x + (mv_x_q >> 3);
    let int_y = base_y + (mv_y_q >> 3);

    for r in 0..part_h as i32 {
        for c in 0..part_w as i32 {
            let y0 = int_y + r;
            let x0 = int_x + c;
            let y1 = y0 + 1;
            let x1 = x0 + 1;
            let p00 = ref_plane[clamp_ref(y0, plane_h) * plane_stride + clamp_ref(x0, plane_w)]
                as i32;
            let p01 = ref_plane[clamp_ref(y0, plane_h) * plane_stride + clamp_ref(x1, plane_w)]
                as i32;
            let p10 = ref_plane[clamp_ref(y1, plane_h) * plane_stride + clamp_ref(x0, plane_w)]
                as i32;
            let p11 = ref_plane[clamp_ref(y1, plane_h) * plane_stride + clamp_ref(x1, plane_w)]
                as i32;
            let v = ((8 - dx) * (8 - dy) * p00
                + dx * (8 - dy) * p01
                + (8 - dx) * dy * p10
                + dx * dy * p11
                + 32)
                >> 6;
            dst[r as usize * part_w + c as usize] = v.clamp(0, 255) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Explicit weighted prediction — §8.4.2.3.2.
// ---------------------------------------------------------------------------

/// Apply explicit luma weighted prediction in place over a `part_w × part_h`
/// block stored in raster order with stride = `part_w`.
///
/// Implements §8.4.2.3.2 for single-list prediction (P-slice L0 only):
///   `predSampleLX = Clip1Y(((x * w) + (1 << (logWD - 1))) >> logWD) + o)`
/// with the `logWD == 0` special case `predSampleLX = Clip1Y(x * w + o)`.
/// When the weight entry isn't `present` in the slice header the default
/// weight `1 << logWD` and offset `0` leave the input unchanged.
pub fn apply_luma_weight(dst: &mut [u8], w: &LumaWeight) {
    apply_weight_plane(dst, w.weight, w.offset, w.log2_denom);
}

/// Apply explicit chroma weighted prediction in place to a single chroma
/// plane (`plane_idx == 0` for Cb, `1` for Cr). Same formula as
/// [`apply_luma_weight`]; `plane_idx` picks the (weight, offset) pair.
pub fn apply_chroma_weight(dst: &mut [u8], w: &ChromaWeight, plane_idx: usize) {
    apply_weight_plane(dst, w.weight[plane_idx], w.offset[plane_idx], w.log2_denom);
}

#[inline]
fn apply_weight_plane(dst: &mut [u8], weight: i32, offset: i32, log2_denom: u32) {
    if log2_denom == 0 {
        for p in dst.iter_mut() {
            let x = *p as i32;
            *p = (x * weight + offset).clamp(0, 255) as u8;
        }
        return;
    }
    let round = 1i32 << (log2_denom - 1);
    for p in dst.iter_mut() {
        let x = *p as i32;
        let v = ((x * weight + round) >> log2_denom) + offset;
        *p = v.clamp(0, 255) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_solid_picture(mb_w: u32, mb_h: u32, v: u8) -> Picture {
        let mut pic = Picture::new(mb_w, mb_h);
        for p in pic.y.iter_mut() {
            *p = v;
        }
        for p in pic.cb.iter_mut() {
            *p = v;
        }
        for p in pic.cr.iter_mut() {
            *p = v;
        }
        pic
    }

    #[test]
    fn integer_pel_mc_copies_source() {
        let pic = make_solid_picture(2, 2, 77);
        let mut dst = vec![0u8; 16 * 16];
        luma_mc(&mut dst, &pic, 0, 0, 0, 0, 16, 16);
        assert!(dst.iter().all(|&x| x == 77));
    }

    #[test]
    fn half_pel_mc_constant_region_is_constant() {
        let pic = make_solid_picture(2, 2, 100);
        let mut dst = vec![0u8; 16 * 16];
        // fx=2, fy=0 — pure horizontal half-pel.
        luma_mc(&mut dst, &pic, 0, 0, 2, 0, 16, 16);
        assert!(dst.iter().all(|&x| x == 100));
        // fx=0, fy=2 — pure vertical half-pel.
        luma_mc(&mut dst, &pic, 0, 0, 0, 2, 16, 16);
        assert!(dst.iter().all(|&x| x == 100));
        // fx=2, fy=2 — center.
        luma_mc(&mut dst, &pic, 0, 0, 2, 2, 16, 16);
        assert!(dst.iter().all(|&x| x == 100));
    }

    #[test]
    fn chroma_mc_constant_region_is_constant() {
        let pic = make_solid_picture(2, 2, 50);
        let mut dst = vec![0u8; 8 * 8];
        chroma_mc(
            &mut dst,
            &pic.cb,
            pic.chroma_stride(),
            (pic.width / 2) as i32,
            (pic.height / 2) as i32,
            0,
            0,
            3,
            5,
            8,
            8,
        );
        assert!(dst.iter().all(|&x| x == 50));
    }

    #[test]
    fn median3_returns_middle() {
        assert_eq!(median3(1, 2, 3), 2);
        assert_eq!(median3(3, 1, 2), 2);
        assert_eq!(median3(-5, 10, 0), 0);
    }

    #[test]
    fn weight_default_is_identity() {
        // present=false → weight = 1<<log2Denom, offset=0; result equals input.
        let mut buf = [10u8, 40, 80, 200];
        let w = LumaWeight {
            weight: 1 << 3,
            offset: 0,
            log2_denom: 3,
            present: false,
        };
        let expected = buf;
        apply_luma_weight(&mut buf, &w);
        assert_eq!(buf, expected);
    }

    #[test]
    fn weight_half_brightness() {
        // weight=1, log2_denom=1 → x>>1 (rounded). offset=0.
        //   10 → (10 + 1) >> 1 = 5
        //   40 → 20
        //  200 → 100
        let mut buf = [10u8, 40, 200];
        let w = LumaWeight {
            weight: 1,
            offset: 0,
            log2_denom: 1,
            present: true,
        };
        apply_luma_weight(&mut buf, &w);
        assert_eq!(buf, [5, 20, 100]);
    }

    #[test]
    fn weight_offset_is_additive() {
        // weight = 1<<denom, offset=+20 → result = x + 20 (clipped to 255).
        let mut buf = [10u8, 100, 250];
        let w = LumaWeight {
            weight: 1 << 5,
            offset: 20,
            log2_denom: 5,
            present: true,
        };
        apply_luma_weight(&mut buf, &w);
        assert_eq!(buf, [30, 120, 255]);
    }

    #[test]
    fn weight_clips_to_valid_range() {
        // weight = 2<<denom, offset=0 → result = 2*x (clipped to 255).
        let mut buf = [10u8, 100, 200];
        let w = LumaWeight {
            weight: 2 << 4,
            offset: 0,
            log2_denom: 4,
            present: true,
        };
        apply_luma_weight(&mut buf, &w);
        assert_eq!(buf, [20, 200, 255]);
        // Negative offset clips at 0.
        let mut buf = [10u8, 100, 200];
        let w = LumaWeight {
            weight: 1 << 4,
            offset: -50,
            log2_denom: 4,
            present: true,
        };
        apply_luma_weight(&mut buf, &w);
        assert_eq!(buf, [0, 50, 150]);
    }

    #[test]
    fn weight_log2_zero_uses_plain_formula() {
        // Special case: logWD == 0 → predSampleLX = Clip1(x*w + o).
        let mut buf = [5u8, 10, 20];
        let w = LumaWeight {
            weight: 3,
            offset: -5,
            log2_denom: 0,
            present: true,
        };
        apply_luma_weight(&mut buf, &w);
        assert_eq!(buf, [10, 25, 55]);
    }

    #[test]
    fn chroma_weight_per_plane() {
        let mut cb = [50u8, 100, 150];
        let mut cr = [50u8, 100, 150];
        let w = ChromaWeight {
            weight: [2 << 3, 1 << 3],
            offset: [-10, 30],
            log2_denom: 3,
            present: true,
        };
        apply_chroma_weight(&mut cb, &w, 0);
        apply_chroma_weight(&mut cr, &w, 1);
        // Cb: (x * 16 + 4) >> 3 - 10 = 2x - 10
        assert_eq!(cb, [90, 190, 255]);
        // Cr: (x * 8 + 4) >> 3 + 30 = x + 30
        assert_eq!(cr, [80, 130, 180]);
    }
}
