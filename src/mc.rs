//! §8.4.2.2 — fractional-sample interpolation on a clipped sample
//! window.
//!
//! The motion-compensation kernels used by the reconstruction path.
//! Each kernel first gathers the integer samples a block can touch —
//! for a `w`x`h` luma block the `(w + 5)`x`(h + 5)` neighbourhood of
//! eqs. 8-239 / 8-240 (Table 8-11 reaches 2 samples left / above and 3
//! right / below), for chroma the `(w + 1)`x`(h + 1)` neighbourhood of
//! eqs. 8-262..8-269 — with the spec's `Clip3` coordinate clamping
//! applied while gathering. The filters (eqs. 8-241..8-261 and 8-270)
//! then run on the window without any further bounds handling, so the
//! result equals the per-sample clamped derivation for every block
//! position, including blocks reaching outside the picture.
//!
//! The source plane is read through [`PlaneSrc`], either one flat
//! row-major plane or a plane published as per-macroblock-row bands by
//! a picture still being decoded on another thread (frame threading);
//! both give the same samples.
//!
//! Clean-room: derived only from ITU-T Rec. H.264 (08/2024) §8.4.2.2.

use std::sync::OnceLock;

/// Largest window a kernel gathers: a 16x16 luma block plus the
/// 6-tap filter margins (2 before, 3 after) in each direction.
const WIN: usize = 21;

/// A reference sample plane, addressed by row.
#[derive(Clone, Copy)]
pub(crate) enum PlaneSrc<'a> {
    /// A flat row-major plane: row `y` starts at `data[y * stride]`.
    /// A field of a frame is a flat plane with a doubled stride.
    Flat {
        data: &'a [i32],
        stride: usize,
        width: usize,
        height: usize,
    },
    /// A plane published in bands of `band_rows` rows (each band a
    /// tightly packed `band_rows * width` buffer). Every band a block
    /// reads must already be published.
    #[cfg_attr(not(test), allow(dead_code))]
    Banded {
        bands: &'a [OnceLock<Box<[i32]>>],
        band_rows: usize,
        width: usize,
        height: usize,
    },
}

impl<'a> PlaneSrc<'a> {
    #[inline]
    pub(crate) fn width(&self) -> usize {
        match *self {
            PlaneSrc::Flat { width, .. } | PlaneSrc::Banded { width, .. } => width,
        }
    }

    #[inline]
    pub(crate) fn height(&self) -> usize {
        match *self {
            PlaneSrc::Flat { height, .. } | PlaneSrc::Banded { height, .. } => height,
        }
    }

    /// Row `y` (`y < height`), `width` samples long.
    #[inline]
    fn row(&self, y: usize) -> &'a [i32] {
        match *self {
            PlaneSrc::Flat {
                data,
                stride,
                width,
                ..
            } => &data[y * stride..y * stride + width],
            PlaneSrc::Banded {
                bands,
                band_rows,
                width,
                ..
            } => {
                let band = bands[y / band_rows]
                    .get()
                    .expect("reference band read before it was published");
                let o = (y % band_rows) * width;
                &band[o..o + width]
            }
        }
    }
}

/// Gather the `ww`x`wh` window whose top-left integer sample is
/// `(x0, y0)` into `out` (row stride `ostride`), clamping every
/// coordinate into the plane (eqs. 8-239 / 8-240, 8-262..8-269).
#[inline]
#[allow(clippy::too_many_arguments)]
fn gather(
    src: &PlaneSrc<'_>,
    x0: i32,
    y0: i32,
    ww: usize,
    wh: usize,
    out: &mut [i32],
    ostride: usize,
) {
    let width = src.width() as i32;
    let ymax = src.height() as i32 - 1;
    let inside_x = x0 >= 0 && x0 + ww as i32 <= width;
    for r in 0..wh {
        let y = (y0 + r as i32).clamp(0, ymax) as usize;
        let row = src.row(y);
        let o = &mut out[r * ostride..r * ostride + ww];
        if inside_x {
            copy_short(o, &row[x0 as usize..x0 as usize + ww]);
        } else {
            for (c, slot) in o.iter_mut().enumerate() {
                *slot = row[(x0 + c as i32).clamp(0, width - 1) as usize];
            }
        }
    }
}

/// `dst.copy_from_slice(src)` for the short rows of a block window,
/// in fixed 4-sample steps (a library `memcpy` call per row costs more
/// than the copy itself at these lengths).
#[inline(always)]
fn copy_short(dst: &mut [i32], src: &[i32]) {
    let src = &src[..dst.len()];
    let (d4, dt) = dst.as_chunks_mut::<4>();
    let (s4, st) = src.as_chunks::<4>();
    for (d, s) in d4.iter_mut().zip(s4) {
        *d = *s;
    }
    for (d, &s) in dt.iter_mut().zip(st) {
        *d = s;
    }
}

#[inline(always)]
fn tap6(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
    a - 5 * b + 20 * c + 20 * d - 5 * e + f
}

#[inline(always)]
fn clip1(v: i32, max_v: i32) -> i32 {
    v.clamp(0, max_v)
}

/// §8.4.2.2.1 — predict a `w`x`h` luma block (`w`, `h` ≤ 16) at the
/// integer position `(int_x, int_y)` with quarter-sample offset
/// `(x_frac, y_frac)` (each 0..=3), writing `dst[y * dst_stride + x]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn luma(
    src: &PlaneSrc<'_>,
    int_x: i32,
    int_y: i32,
    x_frac: u8,
    y_frac: u8,
    w: usize,
    h: usize,
    max_v: i32,
    dst: &mut [i32],
    dst_stride: usize,
) {
    debug_assert!(w <= 16 && h <= 16 && x_frac < 4 && y_frac < 4);
    if x_frac == 0 && y_frac == 0 {
        // Table 8-12 — G: the integer samples themselves.
        gather(src, int_x, int_y, w, h, dst, dst_stride);
        return;
    }
    let ws = w + 5;
    let mut win = [0i32; WIN * WIN];
    gather(src, int_x - 2, int_y - 2, ws, h + 5, &mut win, ws);
    // G at (x, y) of the block is win[(y + 2) * ws + x + 2].
    let g = |x: usize, y: usize| win[(y + 2) * ws + x + 2];
    // eq. 8-241 — b1 at (x, y + dy) (dy = 0: b, dy = 1: s).
    let b1 = |x: usize, yy: usize| {
        let r = &win[yy * ws + x..yy * ws + x + 6];
        tap6(r[0], r[1], r[2], r[3], r[4], r[5])
    };
    // eq. 8-242 — h1 at (x + dx, y) (dx = 0: h, dx = 1: m), `xx` the
    // window column of the sample.
    let h1 = |xx: usize, y: usize| {
        let c = y * ws + xx;
        tap6(
            win[c],
            win[c + ws],
            win[c + 2 * ws],
            win[c + 3 * ws],
            win[c + 4 * ws],
            win[c + 5 * ws],
        )
    };
    let half = |v1: i32| clip1((v1 + 16) >> 5, max_v);
    match (x_frac, y_frac) {
        // a / b / c — horizontal half sample, averaged with G or H.
        (_, 0) => {
            for y in 0..h {
                let drow = &mut dst[y * dst_stride..y * dst_stride + w];
                for (x, d) in drow.iter_mut().enumerate() {
                    let b = half(b1(x, y + 2));
                    *d = match x_frac {
                        1 => (g(x, y) + b + 1) >> 1,
                        2 => b,
                        _ => (g(x + 1, y) + b + 1) >> 1,
                    };
                }
            }
        }
        // d / h / n — vertical half sample, averaged with G or M.
        (0, _) => {
            for y in 0..h {
                let drow = &mut dst[y * dst_stride..y * dst_stride + w];
                for (x, d) in drow.iter_mut().enumerate() {
                    let hh = half(h1(x + 2, y));
                    *d = match y_frac {
                        1 => (g(x, y) + hh + 1) >> 1,
                        2 => hh,
                        _ => (g(x, y + 1) + hh + 1) >> 1,
                    };
                }
            }
        }
        // f / j / q — j from the b1 intermediates (eq. 8-246),
        // averaged with b or s.
        (2, _) => {
            let mut mid = [0i32; WIN * 16];
            for yy in 0..h + 5 {
                for x in 0..w {
                    mid[yy * w + x] = b1(x, yy);
                }
            }
            for y in 0..h {
                let drow = &mut dst[y * dst_stride..y * dst_stride + w];
                for (x, d) in drow.iter_mut().enumerate() {
                    let c = y * w + x;
                    let j1 = tap6(
                        mid[c],
                        mid[c + w],
                        mid[c + 2 * w],
                        mid[c + 3 * w],
                        mid[c + 4 * w],
                        mid[c + 5 * w],
                    );
                    let j = clip1((j1 + 512) >> 10, max_v);
                    *d = match y_frac {
                        1 => (half(mid[c + 2 * w]) + j + 1) >> 1,
                        2 => j,
                        _ => (half(mid[c + 3 * w]) + j + 1) >> 1,
                    };
                }
            }
        }
        // i / k — j from the h1 intermediates (eq. 8-245), averaged
        // with h or m.
        (_, 2) => {
            let mut mid = [0i32; 16 * WIN];
            for y in 0..h {
                for xx in 0..ws {
                    mid[y * ws + xx] = h1(xx, y);
                }
            }
            let side = if x_frac == 1 { 2 } else { 3 };
            for y in 0..h {
                let drow = &mut dst[y * dst_stride..y * dst_stride + w];
                let r = &mid[y * ws..y * ws + ws];
                for (x, d) in drow.iter_mut().enumerate() {
                    let j1 = tap6(r[x], r[x + 1], r[x + 2], r[x + 3], r[x + 4], r[x + 5]);
                    let j = clip1((j1 + 512) >> 10, max_v);
                    *d = (half(r[x + side]) + j + 1) >> 1;
                }
            }
        }
        // e / g / p / r — diagonal averages of b or s with h or m.
        _ => {
            let by = if y_frac == 1 { 2 } else { 3 };
            let hx = if x_frac == 1 { 2 } else { 3 };
            for y in 0..h {
                let drow = &mut dst[y * dst_stride..y * dst_stride + w];
                for (x, d) in drow.iter_mut().enumerate() {
                    let b = half(b1(x, y + by));
                    let hh = half(h1(x + hx, y));
                    *d = (b + hh + 1) >> 1;
                }
            }
        }
    }
}

/// §8.4.2.2.2 — predict a `w`x`h` chroma block (`w`, `h` ≤ 16) at the
/// integer position `(int_x, int_y)` with eighth-sample offset
/// `(x_frac, y_frac)` (each 0..=7), eq. 8-270.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma(
    src: &PlaneSrc<'_>,
    int_x: i32,
    int_y: i32,
    x_frac: u8,
    y_frac: u8,
    w: usize,
    h: usize,
    dst: &mut [i32],
    dst_stride: usize,
) {
    debug_assert!(w <= 16 && h <= 16 && x_frac < 8 && y_frac < 8);
    if x_frac == 0 && y_frac == 0 {
        gather(src, int_x, int_y, w, h, dst, dst_stride);
        return;
    }
    let ws = w + 1;
    let mut win = [0i32; 17 * 17];
    gather(src, int_x, int_y, ws, h + 1, &mut win, ws);
    let xf = x_frac as i32;
    let yf = y_frac as i32;
    let c00 = (8 - xf) * (8 - yf);
    let c10 = xf * (8 - yf);
    let c01 = (8 - xf) * yf;
    let c11 = xf * yf;
    for y in 0..h {
        let ra = &win[y * ws..y * ws + ws];
        let rc = &win[(y + 1) * ws..(y + 1) * ws + ws];
        let drow = &mut dst[y * dst_stride..y * dst_stride + w];
        for (x, d) in drow.iter_mut().enumerate() {
            *d = (c00 * ra[x] + c10 * ra[x + 1] + c01 * rc[x] + c11 * rc[x + 1] + 32) >> 6;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(w: usize, h: usize, seed: u32) -> Vec<i32> {
        let mut s = seed;
        (0..w * h)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((s >> 16) & 0xFF) as i32
            })
            .collect()
    }

    /// The window kernels equal the reference per-sample clamped
    /// interpolators for every fractional position, block size and a
    /// sweep of positions straddling every picture edge — through both
    /// the flat and the banded plane views.
    #[test]
    fn luma_and_chroma_match_reference_interpolators() {
        let (pw, ph) = (40usize, 36usize);
        let data = plane(pw, ph, 7);
        let bands: Vec<OnceLock<Box<[i32]>>> = data
            .chunks(pw * 4)
            .map(|c| {
                let l = OnceLock::new();
                let _ = l.set(c.to_vec().into_boxed_slice());
                l
            })
            .collect();
        let srcs = [
            PlaneSrc::Flat {
                data: &data,
                stride: pw,
                width: pw,
                height: ph,
            },
            PlaneSrc::Banded {
                bands: &bands,
                band_rows: 4,
                width: pw,
                height: ph,
            },
        ];
        let sizes = [
            (4usize, 4usize),
            (8, 4),
            (4, 8),
            (8, 8),
            (16, 8),
            (8, 16),
            (16, 16),
        ];
        for src in &srcs {
            for &(w, h) in &sizes {
                for int_y in [-25i32, -3, 0, 5, 20, 33, 50] {
                    for int_x in [-30i32, -2, 0, 7, 24, 37, 60] {
                        for xf in 0..4u8 {
                            for yf in 0..4u8 {
                                let mut a = vec![0i32; w * h];
                                let mut b = vec![0i32; w * h];
                                luma(src, int_x, int_y, xf, yf, w, h, 255, &mut a, w);
                                crate::simd::scalar::interpolate_luma(
                                    &data, pw, pw, ph, int_x, int_y, xf, yf, w as u32, h as u32, 8,
                                    &mut b, w,
                                )
                                .unwrap();
                                assert_eq!(
                                    a, b,
                                    "luma {w}x{h} at ({int_x},{int_y}) frac ({xf},{yf})"
                                );
                            }
                        }
                        for xf in 0..8u8 {
                            for yf in 0..8u8 {
                                let (cw, chh) = (w / 2, h / 2);
                                let mut a = vec![0i32; cw * chh];
                                let mut b = vec![0i32; cw * chh];
                                chroma(src, int_x, int_y, xf, yf, cw, chh, &mut a, cw);
                                crate::simd::scalar::interpolate_chroma(
                                    &data, pw, pw, ph, int_x, int_y, xf, yf, cw as u32, chh as u32,
                                    8, &mut b, cw,
                                )
                                .unwrap();
                                assert_eq!(
                                    a, b,
                                    "chroma {cw}x{chh} at ({int_x},{int_y}) frac ({xf},{yf})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
