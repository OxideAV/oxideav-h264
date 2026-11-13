//! Reference-sample windows for the encoder's motion search and
//! predictor builds.
//!
//! The §8.4.2.2 interpolators work on `i32` sample planes and read a
//! bounded neighbourhood of the block they predict: the luma 6-tap
//! filter reaches 2 samples before and 3 after the block in each
//! direction, the chroma bilinear filter 1 after. Coordinates outside
//! the picture are clamped onto its edge (eqs. 8-239 / 8-240,
//! 8-262..8-269).
//!
//! A [`RefWindow`] holds just that neighbourhood, fetched from the
//! `u8` reference plane with the clamping already applied. Running an
//! interpolator on the window with the block position translated into
//! window coordinates reads exactly the sample values it would read on
//! the whole plane, so the prediction is identical — without widening
//! the whole plane to `i32` for every block.

use crate::inter_pred::{interpolate_chroma, interpolate_luma};

/// Largest window side: a 16-sample block, the 6-tap reach (2 + 3) and
/// one whole sample of sub-pel refinement on either side.
const MAX_SIDE: usize = 16 + 5 + 2 + 3;

/// `i32` copy of the plane samples `x0..x0 + w` x `y0..y0 + h`
/// (coordinates clamped into the plane).
pub(crate) struct RefWindow {
    data: [i32; MAX_SIDE * MAX_SIDE],
    w: usize,
    h: usize,
    x0: i32,
    y0: i32,
}

impl RefWindow {
    /// Fetch the window from a `u8` plane of `plane_w` x `plane_h`
    /// samples with rows `stride` apart.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        plane: &[u8],
        stride: usize,
        plane_w: usize,
        plane_h: usize,
        x0: i32,
        y0: i32,
        w: usize,
        h: usize,
    ) -> Self {
        assert!(w <= MAX_SIDE && h <= MAX_SIDE, "reference window size");
        assert!(plane_w >= 1 && plane_h >= 1, "empty reference plane");
        let mut data = [0i32; MAX_SIDE * MAX_SIDE];
        let inside_x = x0 >= 0 && x0 as usize + w <= plane_w;
        let (xmax, ymax) = (plane_w as i32 - 1, plane_h as i32 - 1);
        for (j, out) in data.chunks_exact_mut(w.max(1)).take(h).enumerate() {
            let y = (y0 + j as i32).clamp(0, ymax) as usize;
            let row = &plane[y * stride..y * stride + plane_w];
            if inside_x {
                for (o, &s) in out.iter_mut().zip(&row[x0 as usize..x0 as usize + w]) {
                    *o = i32::from(s);
                }
            } else {
                for (i, o) in out.iter_mut().enumerate() {
                    *o = i32::from(row[(x0 + i as i32).clamp(0, xmax) as usize]);
                }
            }
        }
        Self { data, w, h, x0, y0 }
    }

    /// The window a luma interpolation of a `bw` x `bh` block at
    /// integer position `(int_x, int_y)` reads, widened by `slack`
    /// whole samples on every side (sub-pel refinement around the
    /// position).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_luma(
        plane: &[u8],
        stride: usize,
        plane_w: usize,
        plane_h: usize,
        int_x: i32,
        int_y: i32,
        bw: usize,
        bh: usize,
        slack: usize,
    ) -> Self {
        let s = slack as i32;
        Self::new(
            plane,
            stride,
            plane_w,
            plane_h,
            int_x - 2 - s,
            int_y - 2 - s,
            bw + 5 + 2 * slack,
            bh + 5 + 2 * slack,
        )
    }

    /// §8.4.2.2.1 luma prediction of a `w` x `h` block whose integer
    /// position in the PLANE is `(int_x, int_y)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn luma(
        &self,
        int_x: i32,
        int_y: i32,
        x_frac: u8,
        y_frac: u8,
        w: u32,
        h: u32,
        dst: &mut [i32],
        dst_stride: usize,
    ) {
        let (x, y) = (int_x - self.x0, int_y - self.y0);
        // The 6-tap reach must stay inside the window, or the
        // interpolator would clamp onto the window edge instead of
        // the picture edge.
        assert!(
            x >= 2
                && y >= 2
                && x as usize + w as usize + 3 <= self.w
                && y as usize + h as usize + 3 <= self.h,
            "luma block outside its reference window"
        );
        interpolate_luma(
            &self.data[..self.w * self.h],
            self.w,
            self.w,
            self.h,
            x,
            y,
            x_frac,
            y_frac,
            w,
            h,
            8,
            dst,
            dst_stride,
        )
        .expect("luma interpolation");
    }
}

/// §8.4.2.2.1 luma prediction of a `w` x `h` block (sides up to 16)
/// from a `u8` plane: the same samples as `interpolate_luma` on the
/// plane widened to `i32`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_from_plane(
    plane: &[u8],
    stride: usize,
    plane_w: usize,
    plane_h: usize,
    int_x: i32,
    int_y: i32,
    x_frac: u8,
    y_frac: u8,
    w: u32,
    h: u32,
    dst: &mut [i32],
    dst_stride: usize,
) {
    RefWindow::for_luma(
        plane, stride, plane_w, plane_h, int_x, int_y, w as usize, h as usize, 0,
    )
    .luma(int_x, int_y, x_frac, y_frac, w, h, dst, dst_stride);
}

/// §8.4.2.2.2 chroma prediction of a `w` x `h` block (sides up to 16)
/// from a `u8` plane: the same samples as `interpolate_chroma` on the
/// plane widened to `i32`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_from_plane(
    plane: &[u8],
    stride: usize,
    plane_w: usize,
    plane_h: usize,
    int_x: i32,
    int_y: i32,
    x_frac: u8,
    y_frac: u8,
    w: u32,
    h: u32,
    dst: &mut [i32],
    dst_stride: usize,
) {
    // The bilinear filter reads one sample past the block.
    let win = RefWindow::new(
        plane,
        stride,
        plane_w,
        plane_h,
        int_x,
        int_y,
        w as usize + 1,
        h as usize + 1,
    );
    interpolate_chroma(
        &win.data[..win.w * win.h],
        win.w,
        win.w,
        win.h,
        0,
        0,
        x_frac,
        y_frac,
        w,
        h,
        8,
        dst,
        dst_stride,
    )
    .expect("chroma interpolation");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(w: usize, h: usize) -> Vec<u8> {
        (0..w * h)
            .map(|k| ((k * 73 + (k / w) * 31 + (k % 7) * 19) % 251) as u8)
            .collect()
    }

    /// Window-based prediction equals whole-plane prediction for every
    /// fractional position, block size and block position — including
    /// positions far outside the picture, where every read clamps.
    #[test]
    fn window_prediction_equals_whole_plane_prediction() {
        let (pw, ph) = (40usize, 28usize);
        let p = plane(pw, ph);
        let wide: Vec<i32> = p.iter().map(|&v| i32::from(v)).collect();
        for &(w, h) in &[(16u32, 16u32), (8, 8), (16, 8), (8, 16), (4, 4)] {
            for &(x, y) in &[
                (0i32, 0i32),
                (-7, -5),
                (-40, 3),
                (pw as i32 - 9, ph as i32 - 3),
                (pw as i32 + 20, ph as i32 + 9),
                (11, 6),
                (3, -30),
            ] {
                for frac in 0..16u8 {
                    let (xf, yf) = (frac & 3, frac >> 2);
                    let n = (w * h) as usize;
                    let mut a = vec![0i32; n];
                    let mut b = vec![0i32; n];
                    interpolate_luma(&wide, pw, pw, ph, x, y, xf, yf, w, h, 8, &mut a, w as usize)
                        .unwrap();
                    luma_from_plane(&p, pw, pw, ph, x, y, xf, yf, w, h, &mut b, w as usize);
                    assert_eq!(a, b, "luma {w}x{h} at ({x}, {y}) frac ({xf}, {yf})");
                    // A refinement window serves every position within
                    // its slack.
                    let win = RefWindow::for_luma(&p, pw, pw, ph, x, y, w as usize, h as usize, 1);
                    for (dx, dy) in [(-1, -1), (1, 0), (0, 1), (1, 1), (-1, 1)] {
                        interpolate_luma(
                            &wide,
                            pw,
                            pw,
                            ph,
                            x + dx,
                            y + dy,
                            xf,
                            yf,
                            w,
                            h,
                            8,
                            &mut a,
                            w as usize,
                        )
                        .unwrap();
                        win.luma(x + dx, y + dy, xf, yf, w, h, &mut b, w as usize);
                        assert_eq!(a, b, "window luma at ({x}, {y}) + ({dx}, {dy})");
                    }
                }
                for frac in 0..64u8 {
                    let (xf, yf) = (frac & 7, frac >> 3);
                    let n = (w * h) as usize;
                    let mut a = vec![0i32; n];
                    let mut b = vec![0i32; n];
                    interpolate_chroma(
                        &wide, pw, pw, ph, x, y, xf, yf, w, h, 8, &mut a, w as usize,
                    )
                    .unwrap();
                    chroma_from_plane(&p, pw, pw, ph, x, y, xf, yf, w, h, &mut b, w as usize);
                    assert_eq!(a, b, "chroma {w}x{h} at ({x}, {y}) frac ({xf}, {yf})");
                }
            }
        }
    }
}
