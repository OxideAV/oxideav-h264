//! §8.4 — Motion estimation (encoder side, integer-pel).
//!
//! Round-16 introduces a minimal **integer-pel motion estimation** for
//! P-slice support. Search strategy:
//!
//! * Centre at (0, 0) — i.e. the predicted MV (PredMV is the same when
//!   no neighbour MVs are available, which is the common case for our
//!   single-reference encoder + sliding-window DPB).
//! * **Square diamond / full search** within `±range_x` / `±range_y`
//!   pixels. Picks the (mv_x, mv_y) minimising **SAD** between the
//!   source 16×16 luma block and the integer-pel reference block.
//! * Reference-picture edge replication is via [`Self::sample_clipped`],
//!   matching §8.4.2.2.1 eq. 8-239 / 8-240. So MVs that point outside
//!   the picture are still legal — the decoder will replicate the same
//!   way through `inter_pred::interpolate_luma` at integer fractional
//!   coords.
//!
//! Half-pel and 4MV refinement are deferred to a later round. Here we
//! stay strictly integer-pel single-MV-per-MB.

#![allow(clippy::too_many_arguments)]

use crate::encoder::bitstream::BitWriter;

/// Result of motion estimation for one MB partition.
#[derive(Debug, Clone, Copy)]
pub struct MeResult {
    /// Best MV in **quarter-pel units** (encoded as `mv = mv_pel * 4`),
    /// so the bitstream can carry it directly. For integer-pel ME this
    /// is always a multiple of 4.
    pub mv_x: i32,
    pub mv_y: i32,
    /// SAD of the chosen (mv_x, mv_y) integer-pel block.
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

    #[test]
    fn pred_bits_simple_mvd() {
        // mb_type ue(0) = 1 bit. se(0) = 1 bit each. Total = 3.
        assert_eq!(p_l0_16x16_pred_bits(0, 0), 3);
        // se(1) = ue(1) = 3 bits → mb_type 1 + 3 + 3 = 7.
        assert_eq!(p_l0_16x16_pred_bits(1, 1), 7);
    }
}
