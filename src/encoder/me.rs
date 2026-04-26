//! §8.4 — Motion estimation (encoder side).
//!
//! Round-16 introduced a minimal **integer-pel motion estimation** for
//! P-slice support. Round-17 layered **half-pel refinement** on top
//! using the §8.4.2.2.1 6-tap luma interpolator. Round-18 adds the
//! third stage: **quarter-pel refinement** around the half-pel winner
//! covering all 8 surrounding ¼-pel offsets. The quarter-pel positions
//! `a/c/d/f/i/k/n/q` (and `e/g/p/r` on the diagonals) are produced
//! per §8.4.2.2.1 eq. 8-250..8-261 — they are bilinear averages of
//! the integer / half-pel samples that the round-17 path already
//! computes. We never re-implement the formulas here: the encoder
//! invokes [`interpolate_luma`] (the same routine the decoder calls)
//! with the chosen `(xFrac, yFrac)` ∈ {0, 1, 2, 3}², so encoder and
//! decoder agree bit-for-bit on the predictor for any sub-pel MV the
//! encoder picks.
//!
//! Search strategy (16×16 luma):
//!
//! 1. **Integer-pel full search** centred at (0, 0) within
//!    `±range_x` / `±range_y` pixels. Picks the integer (dx, dy)
//!    minimising SAD against the 16×16 source block.
//! 2. **Half-pel refinement** around the integer winner: try the 8
//!    ½-pel neighbours, keep the best by SAD against the 6-tap
//!    interpolated predictor.
//! 3. **Quarter-pel refinement** around the half-pel winner: try the
//!    8 ¼-pel neighbours (offsets `±1` in qpel units around the
//!    half-pel best). Keep the best by SAD against the
//!    bilinear-of-integer-and-half-pel predictor.
//!
//! All MVs are returned in **quarter-pel units** (mv = pel × 4), matching
//! `mvd_l0_*` syntax in §7.4.5.1 and the decoder's expectations in
//! §8.4.1.2. Integer-pel MVs are multiples of 4; half-pel multiples of
//! 2; quarter-pel can be any integer in qpel units.
//!
//! Reference-picture edge replication is via §8.4.2.2.1 eq. 8-239 /
//! 8-240. The integer-pel SAD path uses local clamping; the half- and
//! quarter-pel paths inherit replication from
//! [`inter_pred::interpolate_luma`].
//!
//! 4MV partitions are deferred to later rounds.

#![allow(clippy::too_many_arguments)]

use crate::encoder::bitstream::BitWriter;
use crate::inter_pred::interpolate_luma;

/// Result of motion estimation for one MB partition.
#[derive(Debug, Clone, Copy)]
pub struct MeResult {
    /// Best MV in **quarter-pel units** (encoded as `mv = mv_pel * 4`),
    /// so the bitstream can carry it directly. Integer-pel ME yields
    /// multiples of 4; half-pel refinement can introduce multiples of 2.
    pub mv_x: i32,
    pub mv_y: i32,
    /// SAD of the chosen (mv_x, mv_y) predictor block.
    pub sad: u32,
}

/// Compute SAD between a 16×16 source block and a 16×16 integer-pel
/// reference block at `(int_x, int_y)` of the reference plane. Edges
/// are replicated per §8.4.2.2.1.
fn sad_16x16_at(
    src_y: &[u8],
    src_stride: usize,
    src_mb_x: usize,
    src_mb_y: usize,
    ref_y: &[u8],
    ref_stride: usize,
    ref_w: i32,
    ref_h: i32,
    int_x: i32,
    int_y: i32,
) -> u32 {
    let mut sad = 0u32;
    for j in 0..16i32 {
        let ry = (int_y + j).clamp(0, ref_h - 1) as usize;
        let row_ref = &ref_y[ry * ref_stride..ry * ref_stride + ref_stride];
        let row_src_off = (src_mb_y * 16 + j as usize) * src_stride + src_mb_x * 16;
        let row_src = &src_y[row_src_off..row_src_off + 16];
        for i in 0..16i32 {
            let rx = (int_x + i).clamp(0, ref_w - 1) as usize;
            let s = row_src[i as usize] as i32;
            let r = row_ref[rx] as i32;
            sad = sad.saturating_add((s - r).unsigned_abs());
        }
    }
    sad
}

/// Run integer-pel **full-search** motion estimation for a single 16×16
/// luma macroblock. `(mb_x, mb_y)` is the MB position (in MB units) of
/// both source and (initial) reference origin. The search centres at
/// (0, 0) MV (no PredMV bias yet). Returns the best MV in quarter-pel
/// units.
pub fn search_integer_16x16(
    src_y: &[u8],
    src_stride: usize,
    src_w: u32,
    src_h: u32,
    ref_y: &[u8],
    ref_stride: usize,
    ref_w: u32,
    ref_h: u32,
    mb_x: usize,
    mb_y: usize,
    range_x: i32,
    range_y: i32,
) -> MeResult {
    let _ = src_w;
    let _ = src_h;
    let ref_w_i = ref_w as i32;
    let ref_h_i = ref_h as i32;
    let centre_x = (mb_x * 16) as i32;
    let centre_y = (mb_y * 16) as i32;

    let mut best_sad = u32::MAX;
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;
    let mut best_mv_cost = u32::MAX;
    for dy in -range_y..=range_y {
        for dx in -range_x..=range_x {
            let int_x = centre_x + dx;
            let int_y = centre_y + dy;
            let sad = sad_16x16_at(
                src_y, src_stride, mb_x, mb_y, ref_y, ref_stride, ref_w_i, ref_h_i, int_x, int_y,
            );
            // Tie-break: prefer the MV with the smaller `|dx| + |dy|`
            // magnitude. With predicted MV at (0, 0), this minimises
            // mvd magnitude → fewer bits. Without this, identical-frame
            // searches pick the first MV walked (the most negative one),
            // which is a non-trivial bit cost.
            let mv_cost = (dx.unsigned_abs()) + (dy.unsigned_abs());
            if sad < best_sad || (sad == best_sad && mv_cost < best_mv_cost) {
                best_sad = sad;
                best_dx = dx;
                best_dy = dy;
                best_mv_cost = mv_cost;
            }
        }
    }
    MeResult {
        mv_x: best_dx * 4, // integer-pel → quarter-pel units (mv = pel * 4)
        mv_y: best_dy * 4,
        sad: best_sad,
    }
}

/// Compute SAD between a 16×16 source block and a 16×16 reference
/// predictor produced by the §8.4.2.2.1 6-tap interpolator at the
/// fractional MV `(mv_qpel_x, mv_qpel_y)` (quarter-pel units, anchored
/// at MB origin `(mb_x*16, mb_y*16)`). The predictor is built via
/// `interpolate_luma`, which transparently handles edge replication.
fn sad_16x16_at_qpel(
    src_y: &[u8],
    src_stride: usize,
    src_mb_x: usize,
    src_mb_y: usize,
    ref_i32: &[i32],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    mb_x: usize,
    mb_y: usize,
    mv_qpel_x: i32,
    mv_qpel_y: i32,
) -> u32 {
    let int_x = (mb_x * 16) as i32 + (mv_qpel_x >> 2);
    let int_y = (mb_y * 16) as i32 + (mv_qpel_y >> 2);
    let x_frac = (mv_qpel_x & 3) as u8;
    let y_frac = (mv_qpel_y & 3) as u8;
    let mut pred = [0i32; 256];
    interpolate_luma(
        ref_i32, ref_stride, ref_w, ref_h, int_x, int_y, x_frac, y_frac, 16, 16, 8, &mut pred, 16,
    )
    .expect("luma interpolation");

    let mut sad = 0u32;
    for j in 0..16usize {
        let row_src_off = (src_mb_y * 16 + j) * src_stride + src_mb_x * 16;
        for i in 0..16usize {
            let s = src_y[row_src_off + i] as i32;
            let p = pred[j * 16 + i].clamp(0, 255);
            sad = sad.saturating_add((s - p).unsigned_abs());
        }
    }
    sad
}

/// §8.4.2.2.1 — Half-pel refinement around an integer-pel best MV.
///
/// `start_mv_qpel_x` / `start_mv_qpel_y` are the quarter-pel-unit MV
/// returned by [`search_integer_16x16`] (multiples of 4). This routine
/// tries the 8 surrounding **half-pel** offsets `(±2, 0)`, `(0, ±2)`,
/// `(±2, ±2)` (still in quarter-pel units, so `±2` = ½-pel). The 6-tap
/// interpolator from §8.4.2.2.1 (eq. 8-241..8-247) is applied via
/// [`interpolate_luma`]. Returns the best MV by SAD; ties prefer the
/// smaller (|mvd_x| + |mvd_y|) magnitude relative to PredMV at (0, 0)
/// to keep mvd-bit cost low.
///
/// Note: chroma cost is intentionally not factored in — chroma 1/8-pel
/// derives mechanically from the luma MV (§8.4.1.4) and the chroma
/// bilinear filter is a much smoother surface.
pub fn refine_half_pel_16x16(
    src_y: &[u8],
    src_stride: usize,
    src_w: u32,
    src_h: u32,
    ref_y: &[u8],
    ref_stride: usize,
    ref_w: u32,
    ref_h: u32,
    mb_x: usize,
    mb_y: usize,
    start_mv_qpel_x: i32,
    start_mv_qpel_y: i32,
    start_sad: u32,
) -> MeResult {
    let _ = src_w;
    let _ = src_h;
    let ref_w_us = ref_w as usize;
    let ref_h_us = ref_h as usize;
    // Promote the reference plane to i32 once for `interpolate_luma`.
    let ref_i32: Vec<i32> = ref_y
        .iter()
        .take(ref_stride * ref_h_us)
        .map(|&v| v as i32)
        .collect();

    let mut best_mv_x = start_mv_qpel_x;
    let mut best_mv_y = start_mv_qpel_y;
    let mut best_sad = start_sad;
    let mut best_mv_cost = (start_mv_qpel_x.unsigned_abs()) + (start_mv_qpel_y.unsigned_abs());

    // 8 half-pel neighbours (in quarter-pel units, ±2 = ½-pel).
    const HALF_OFFS: [(i32, i32); 8] = [
        (-2, -2),
        (0, -2),
        (2, -2),
        (-2, 0),
        (2, 0),
        (-2, 2),
        (0, 2),
        (2, 2),
    ];
    for &(dx, dy) in &HALF_OFFS {
        let mv_x = start_mv_qpel_x + dx;
        let mv_y = start_mv_qpel_y + dy;
        let sad = sad_16x16_at_qpel(
            src_y, src_stride, mb_x, mb_y, &ref_i32, ref_stride, ref_w_us, ref_h_us, mb_x, mb_y,
            mv_x, mv_y,
        );
        let mv_cost = (mv_x.unsigned_abs()) + (mv_y.unsigned_abs());
        if sad < best_sad || (sad == best_sad && mv_cost < best_mv_cost) {
            best_sad = sad;
            best_mv_x = mv_x;
            best_mv_y = mv_y;
            best_mv_cost = mv_cost;
        }
    }

    MeResult {
        mv_x: best_mv_x,
        mv_y: best_mv_y,
        sad: best_sad,
    }
}

/// Convenience wrapper: integer-pel full search (±range) followed by
/// half-pel refinement around the integer winner. Returns the refined
/// MV in quarter-pel units. Equivalent to calling
/// [`search_integer_16x16`] then [`refine_half_pel_16x16`].
pub fn search_half_pel_16x16(
    src_y: &[u8],
    src_stride: usize,
    src_w: u32,
    src_h: u32,
    ref_y: &[u8],
    ref_stride: usize,
    ref_w: u32,
    ref_h: u32,
    mb_x: usize,
    mb_y: usize,
    range_x: i32,
    range_y: i32,
) -> MeResult {
    let int_best = search_integer_16x16(
        src_y, src_stride, src_w, src_h, ref_y, ref_stride, ref_w, ref_h, mb_x, mb_y, range_x,
        range_y,
    );
    refine_half_pel_16x16(
        src_y,
        src_stride,
        src_w,
        src_h,
        ref_y,
        ref_stride,
        ref_w,
        ref_h,
        mb_x,
        mb_y,
        int_best.mv_x,
        int_best.mv_y,
        int_best.sad,
    )
}

/// §8.4.2.2.1 — Quarter-pel refinement around a half-pel best MV.
///
/// `start_mv_qpel_x` / `start_mv_qpel_y` is the quarter-pel-unit MV
/// returned by [`refine_half_pel_16x16`] (multiples of 2). This routine
/// tries the 8 surrounding **quarter-pel** offsets `(±1, 0)`, `(0, ±1)`,
/// `(±1, ±1)` (in quarter-pel units). The fractional position
/// `(xFrac, yFrac) = (mv_qpel_x & 3, mv_qpel_y & 3)` selects one of the
/// 16 sub-pel positions in §8.4.2.2.1 eq. 8-243 .. 8-261. The
/// interpolator from [`interpolate_luma`] handles all 16 cases
/// uniformly, so the encoder picks the same predictor the decoder will
/// derive at decode time.
///
/// Returns the best MV by SAD; ties prefer the smaller
/// `(|mv_qpel_x| + |mv_qpel_y|)` magnitude relative to PredMV at (0, 0)
/// to keep mvd-bit cost low. As with the half-pel pass, chroma cost is
/// not factored in — chroma 1/8-pel derives mechanically from the luma
/// MV per §8.4.1.4 and the chroma bilinear filter is much smoother.
pub fn refine_quarter_pel_16x16(
    src_y: &[u8],
    src_stride: usize,
    src_w: u32,
    src_h: u32,
    ref_y: &[u8],
    ref_stride: usize,
    ref_w: u32,
    ref_h: u32,
    mb_x: usize,
    mb_y: usize,
    start_mv_qpel_x: i32,
    start_mv_qpel_y: i32,
    start_sad: u32,
) -> MeResult {
    let _ = src_w;
    let _ = src_h;
    let ref_w_us = ref_w as usize;
    let ref_h_us = ref_h as usize;
    // Promote the reference plane to i32 once for `interpolate_luma`.
    let ref_i32: Vec<i32> = ref_y
        .iter()
        .take(ref_stride * ref_h_us)
        .map(|&v| v as i32)
        .collect();

    let mut best_mv_x = start_mv_qpel_x;
    let mut best_mv_y = start_mv_qpel_y;
    let mut best_sad = start_sad;
    let mut best_mv_cost = (start_mv_qpel_x.unsigned_abs()) + (start_mv_qpel_y.unsigned_abs());

    // 8 quarter-pel neighbours around the half-pel winner. In
    // quarter-pel units, ±1 = ¼-pel.
    const QUARTER_OFFS: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
    for &(dx, dy) in &QUARTER_OFFS {
        let mv_x = start_mv_qpel_x + dx;
        let mv_y = start_mv_qpel_y + dy;
        let sad = sad_16x16_at_qpel(
            src_y, src_stride, mb_x, mb_y, &ref_i32, ref_stride, ref_w_us, ref_h_us, mb_x, mb_y,
            mv_x, mv_y,
        );
        let mv_cost = (mv_x.unsigned_abs()) + (mv_y.unsigned_abs());
        if sad < best_sad || (sad == best_sad && mv_cost < best_mv_cost) {
            best_sad = sad;
            best_mv_x = mv_x;
            best_mv_y = mv_y;
            best_mv_cost = mv_cost;
        }
    }

    MeResult {
        mv_x: best_mv_x,
        mv_y: best_mv_y,
        sad: best_sad,
    }
}

/// Convenience wrapper: integer-pel full search (±range), then half-pel
/// refinement, then quarter-pel refinement. Returns the refined MV in
/// quarter-pel units. Equivalent to calling [`search_integer_16x16`],
/// [`refine_half_pel_16x16`], and [`refine_quarter_pel_16x16`] in
/// sequence.
pub fn search_quarter_pel_16x16(
    src_y: &[u8],
    src_stride: usize,
    src_w: u32,
    src_h: u32,
    ref_y: &[u8],
    ref_stride: usize,
    ref_w: u32,
    ref_h: u32,
    mb_x: usize,
    mb_y: usize,
    range_x: i32,
    range_y: i32,
) -> MeResult {
    let half_best = search_half_pel_16x16(
        src_y, src_stride, src_w, src_h, ref_y, ref_stride, ref_w, ref_h, mb_x, mb_y, range_x,
        range_y,
    );
    refine_quarter_pel_16x16(
        src_y,
        src_stride,
        src_w,
        src_h,
        ref_y,
        ref_stride,
        ref_w,
        ref_h,
        mb_x,
        mb_y,
        half_best.mv_x,
        half_best.mv_y,
        half_best.sad,
    )
}

/// Trial-encode the L0 P_L0_16x16 mb_pred syntax (mb_type + mvd_l0)
/// to measure its bit cost for RDO. `ref_idx_l0` is implicit when
/// `num_ref_idx_l0_active_minus1 == 0` — no `te(v)` is emitted.
///
/// Returns the bit count of `mb_type ue(0)` + `mvd_l0_x se(mvdx)` +
/// `mvd_l0_y se(mvdy)`.
pub fn p_l0_16x16_pred_bits(mvd_x: i32, mvd_y: i32) -> u64 {
    let mut w = BitWriter::new();
    w.ue(0); // mb_type = P_L0_16x16
    w.se(mvd_x);
    w.se(mvd_y);
    w.bits_emitted() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_solid_frame(w: usize, h: usize, val: u8) -> Vec<u8> {
        vec![val; w * h]
    }

    #[test]
    fn search_zero_motion_on_identical_frames() {
        let src = make_solid_frame(64, 64, 128);
        let r = make_solid_frame(64, 64, 128);
        let m = search_integer_16x16(&src, 64, 64, 64, &r, 64, 64, 64, 1, 1, 4, 4);
        assert_eq!(m.mv_x, 0);
        assert_eq!(m.mv_y, 0);
        assert_eq!(m.sad, 0);
    }

    #[test]
    fn search_finds_offset_motion() {
        // Source at MB (1,1) is a checkerboard tile; the reference has the
        // same tile shifted right by 4 pixels.
        let mut src = vec![0u8; 64 * 64];
        let mut r = vec![0u8; 64 * 64];
        for j in 16..32 {
            for i in 16..32 {
                let v = if ((i + j) % 2) == 0 { 200u8 } else { 50 };
                src[j * 64 + i] = v;
                // Reference: same tile but shifted +4 in x → MV should be (+4, 0).
                r[j * 64 + i + 4] = v;
            }
        }
        let m = search_integer_16x16(&src, 64, 64, 64, &r, 64, 64, 64, 1, 1, 8, 8);
        assert_eq!(m.mv_x, 4 * 4); // +4 pel in quarter-pel units
        assert_eq!(m.mv_y, 0);
        assert_eq!(m.sad, 0);
    }

    #[test]
    fn search_handles_picture_boundary_via_replication() {
        // When the search would step past the right edge, the integer
        // sample is clamped — so a constant-value reference still gives
        // SAD=0 for any (mv_x, mv_y) within range, including out-of-bounds
        // MVs. We just verify zero SAD with offsets at the boundary.
        let src = make_solid_frame(32, 32, 100);
        let r = make_solid_frame(32, 32, 100);
        let m = search_integer_16x16(&src, 32, 32, 32, &r, 32, 32, 32, 1, 1, 16, 16);
        assert_eq!(m.sad, 0);
    }

    // ---- Half-pel refinement tests ----

    /// Apply the spec's horizontal half-pel filter (eq. 8-241/8-242)
    /// at integer position (x, y), with edge-replicated reference
    /// samples — this is what the decoder will compute for x_frac = 2,
    /// y_frac = 0 at the same coordinate.
    fn b_at(plane: &[u8], stride: usize, w: i32, h: i32, x: i32, y: i32) -> i32 {
        let s = |xx: i32| -> i32 {
            let cx = xx.clamp(0, w - 1) as usize;
            let cy = y.clamp(0, h - 1) as usize;
            plane[cy * stride + cx] as i32
        };
        let b1 = s(x - 2) - 5 * s(x - 1) + 20 * s(x) + 20 * s(x + 1) - 5 * s(x + 2) + s(x + 3);
        ((b1 + 16) >> 5).clamp(0, 255)
    }

    #[test]
    fn half_pel_refinement_finds_horizontal_half_pel_motion() {
        // Reference: linear ramp in x, constant per y. The 6-tap FIR
        // applied at any position with xFrac=2 returns the linearly
        // interpolated value at (x+½), which is `(r[x]+r[x+1])/2` for
        // a linear ramp. Source at MB (1,1) is exactly the half-pel-
        // right interpolation of the reference at the same MB.
        // Integer (0,0) SAD is non-zero (offset by ½-pel of slope);
        // mv_qpel=+2 reproduces the source bit-exactly.
        let mut r = vec![0u8; 64 * 64];
        for j in 0..64 {
            for i in 0..64 {
                // Slope of 3 per pixel in x, range stays in [0, 255]
                // for i in 0..64 → 0..189 plus a +30 bias.
                r[j * 64 + i] = (30 + i * 3).clamp(0, 255) as u8;
            }
        }
        let mut src = vec![0u8; 64 * 64];
        for j in 0..16i32 {
            for i in 0..16i32 {
                let x = 16 + i;
                let y = 16 + j;
                src[(y as usize) * 64 + x as usize] = b_at(&r, 64, 64, 64, x, y) as u8;
            }
        }
        let res = search_half_pel_16x16(&src, 64, 64, 64, &r, 64, 64, 64, 1, 1, 4, 4);
        assert_eq!(res.mv_x, 2, "expected mv_x = +½ pel (qpel +2)");
        assert_eq!(res.mv_y, 0, "expected mv_y = 0");
        assert_eq!(res.sad, 0, "predictor should match the synthetic source");
    }

    #[test]
    fn half_pel_refinement_keeps_integer_when_better() {
        // Identical frames — the integer (0, 0) MV gives SAD = 0.
        // Half-pel refinement must NOT replace it (no neighbour can
        // beat 0). Tie-break must prefer the smaller-magnitude MV
        // (which is the integer winner here).
        let src = vec![123u8; 64 * 64];
        let r = vec![123u8; 64 * 64];
        let res = search_half_pel_16x16(&src, 64, 64, 64, &r, 64, 64, 64, 1, 1, 4, 4);
        assert_eq!(res.mv_x, 0);
        assert_eq!(res.mv_y, 0);
        assert_eq!(res.sad, 0);
    }

    #[test]
    fn half_pel_refinement_preserves_integer_winner_on_clean_integer_motion() {
        // Reference shifted +4 pel from source. Integer search returns
        // (16, 0) qpel. Half-pel refinement should NOT degrade — it
        // must find SAD <= the integer one.
        let mut src = vec![100u8; 64 * 64];
        let mut r = vec![100u8; 64 * 64];
        for j in 16..32 {
            for i in 16..32 {
                let v = if ((i + j) % 2) == 0 { 200u8 } else { 50 };
                src[j * 64 + i] = v;
                r[j * 64 + i + 4] = v;
            }
        }
        let int_only = search_integer_16x16(&src, 64, 64, 64, &r, 64, 64, 64, 1, 1, 8, 8);
        let refined = search_half_pel_16x16(&src, 64, 64, 64, &r, 64, 64, 64, 1, 1, 8, 8);
        assert_eq!(int_only.mv_x, 16);
        assert!(
            refined.sad <= int_only.sad,
            "half-pel refinement must not degrade SAD: int={} refined={}",
            int_only.sad,
            refined.sad,
        );
        // For perfectly integer motion, the half-pel offsets cannot
        // beat SAD=0; refinement should keep the integer MV.
        assert_eq!(refined.mv_x, 16);
        assert_eq!(refined.mv_y, 0);
        assert_eq!(refined.sad, 0);
    }

    #[test]
    fn pred_bits_simple_mvd() {
        // mb_type ue(0) = 1 bit. se(0) = 1 bit each. Total = 3.
        assert_eq!(p_l0_16x16_pred_bits(0, 0), 3);
        // se(1) = ue(1) = 3 bits → mb_type 1 + 3 + 3 = 7.
        assert_eq!(p_l0_16x16_pred_bits(1, 1), 7);
    }

    // ---- Quarter-pel refinement tests ----

    /// §8.4.2.2.1 eq. 8-250 — `a = (G + b + 1) >> 1`, the quarter-pel
    /// position at xFrac=1, yFrac=0. Reference values come from the
    /// integer sample G and the horizontal half-pel `b` (already
    /// validated in the half-pel test helper above).
    fn a_at(plane: &[u8], stride: usize, w: i32, h: i32, x: i32, y: i32) -> i32 {
        let cx = x.clamp(0, w - 1) as usize;
        let cy = y.clamp(0, h - 1) as usize;
        let g = plane[cy * stride + cx] as i32;
        let b = b_at(plane, stride, w, h, x, y);
        (g + b + 1) >> 1
    }

    #[test]
    fn quarter_pel_refinement_finds_quarter_pel_motion() {
        // Reference: linear ramp in x. For a slope-3 ramp the integer
        // sample at i is `30 + 3*i`, half-pel `b` is `32 + 3*i`, and
        // quarter-pel `a = (G + b + 1) >> 1` evaluates to `31 + 3*i`.
        // Source at MB (1,1) is the `a` sample of the reference at the
        // same MB → the encoder must pick mv_qpel = +1 to recover
        // SAD = 0.
        let mut r = vec![0u8; 64 * 64];
        for j in 0..64 {
            for i in 0..64 {
                r[j * 64 + i] = (30 + i * 3).clamp(0, 255) as u8;
            }
        }
        let mut src = vec![0u8; 64 * 64];
        for j in 0..16i32 {
            for i in 0..16i32 {
                let x = 16 + i;
                let y = 16 + j;
                src[(y as usize) * 64 + x as usize] = a_at(&r, 64, 64, 64, x, y) as u8;
            }
        }
        let res = search_quarter_pel_16x16(&src, 64, 64, 64, &r, 64, 64, 64, 1, 1, 4, 4);
        assert_eq!(res.mv_x, 1, "expected mv_x = +¼ pel (qpel +1)");
        assert_eq!(res.mv_y, 0, "expected mv_y = 0");
        assert_eq!(res.sad, 0, "predictor should match the synthetic source");
    }

    #[test]
    fn quarter_pel_refinement_keeps_half_pel_when_better() {
        // Synthetic source = exactly the half-pel `b` predictor at
        // MB (1,1). Quarter-pel refinement must keep the half-pel MV
        // (mv_qpel = +2) — no quarter-pel neighbour can beat SAD = 0.
        let mut r = vec![0u8; 64 * 64];
        for j in 0..64 {
            for i in 0..64 {
                r[j * 64 + i] = (30 + i * 3).clamp(0, 255) as u8;
            }
        }
        let mut src = vec![0u8; 64 * 64];
        for j in 0..16i32 {
            for i in 0..16i32 {
                let x = 16 + i;
                let y = 16 + j;
                src[(y as usize) * 64 + x as usize] = b_at(&r, 64, 64, 64, x, y) as u8;
            }
        }
        let half = search_half_pel_16x16(&src, 64, 64, 64, &r, 64, 64, 64, 1, 1, 4, 4);
        let qpel = search_quarter_pel_16x16(&src, 64, 64, 64, &r, 64, 64, 64, 1, 1, 4, 4);
        assert_eq!(half.mv_x, 2);
        assert_eq!(half.sad, 0);
        // Quarter-pel refinement cannot beat SAD = 0; with the
        // smaller-magnitude tie break it must keep the half-pel winner.
        assert!(qpel.sad <= half.sad);
        assert_eq!(qpel.mv_x, 2);
        assert_eq!(qpel.mv_y, 0);
        assert_eq!(qpel.sad, 0);
    }

    #[test]
    fn quarter_pel_refinement_keeps_integer_when_better() {
        // Identical frames — integer (0,0) gives SAD = 0. Half-pel
        // refinement keeps it; quarter-pel refinement must too.
        let src = vec![55u8; 64 * 64];
        let r = vec![55u8; 64 * 64];
        let res = search_quarter_pel_16x16(&src, 64, 64, 64, &r, 64, 64, 64, 1, 1, 4, 4);
        assert_eq!(res.mv_x, 0);
        assert_eq!(res.mv_y, 0);
        assert_eq!(res.sad, 0);
    }

    #[test]
    fn quarter_pel_refinement_does_not_degrade_clean_integer_motion() {
        // Reference shifted +4 pel from source. The ME chain must keep
        // the integer winner all the way through quarter-pel refinement
        // (SAD = 0 is unbeatable).
        let mut src = vec![100u8; 64 * 64];
        let mut r = vec![100u8; 64 * 64];
        for j in 16..32 {
            for i in 16..32 {
                let v = if ((i + j) % 2) == 0 { 200u8 } else { 50 };
                src[j * 64 + i] = v;
                r[j * 64 + i + 4] = v;
            }
        }
        let qpel = search_quarter_pel_16x16(&src, 64, 64, 64, &r, 64, 64, 64, 1, 1, 8, 8);
        assert_eq!(qpel.mv_x, 16);
        assert_eq!(qpel.mv_y, 0);
        assert_eq!(qpel.sad, 0);
    }
}
