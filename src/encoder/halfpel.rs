//! Half-sample planes of a reference picture for the motion search.
//!
//! §8.4.2.2.1 builds every fractional luma sample from the integer
//! samples `G` and three half-sample values: `b` (horizontal 6-tap,
//! eq. 8-241 / 8-243), `h` (vertical 6-tap, eq. 8-242 / 8-244) and `j`
//! (the 6-tap filter applied in both directions on the unrounded
//! intermediates, eq. 8-245..8-247). The twelve quarter-sample
//! positions are rounded averages of two of those (eq. 8-250..8-261).
//!
//! A sub-sample refinement that interpolates a block per candidate
//! recomputes the same half-sample values again and again — 16
//! candidates per search, several searches per macroblock. This module
//! filters the whole reference once into `b`, `h` and `j` planes
//! (padded, with the picture-edge clamp of eq. 8-239 / 8-240 applied
//! per sample), after which the prediction at any quarter-sample
//! position is one plane or the rounded average of two, and a SAD
//! against it is a plain row walk. The values are exactly those of
//! `inter_pred::interpolate_luma` (the test below compares every
//! fraction at positions inside, across and beyond the picture edge).

/// Clip1 at 8 bits.
#[inline]
fn clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// The 6-tap filter of eq. 8-241 on six consecutive values.
#[inline]
fn tap6(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
    a - 5 * b + 20 * c + 20 * d - 5 * e + f
}

/// Padded integer and half-sample planes of one 8-bit reference plane.
pub(crate) struct HalfPel {
    /// Samples of padding on every side.
    pad: usize,
    stride: usize,
    rows: usize,
    /// Integer samples `G`.
    g: Vec<u8>,
    /// Horizontal half samples `b` (between `G(x, y)` and `G(x+1, y)`).
    b: Vec<u8>,
    /// Vertical half samples `h` (between `G(x, y)` and `G(x, y+1)`).
    h: Vec<u8>,
    /// Centre half samples `j`.
    j: Vec<u8>,
}

/// Which plane a term of the prediction comes from.
#[derive(Clone, Copy)]
enum Src {
    G,
    B,
    H,
    J,
}

/// One term: a plane and its whole-sample displacement.
type Term = (Src, usize, usize);

/// The one or two terms of the prediction at fraction `(xf, yf)`
/// (eq. 8-250..8-261; `m` is `h` one column right, `s` is `b` one row
/// down).
fn terms(xf: u8, yf: u8) -> (Term, Option<Term>) {
    use Src::{B, G, H, J};
    match (xf & 3, yf & 3) {
        (0, 0) => ((G, 0, 0), None),
        (1, 0) => ((G, 0, 0), Some((B, 0, 0))),
        (2, 0) => ((B, 0, 0), None),
        (3, 0) => ((G, 1, 0), Some((B, 0, 0))),
        (0, 1) => ((G, 0, 0), Some((H, 0, 0))),
        (0, 2) => ((H, 0, 0), None),
        (0, 3) => ((G, 0, 1), Some((H, 0, 0))),
        (1, 1) => ((B, 0, 0), Some((H, 0, 0))),
        (2, 1) => ((B, 0, 0), Some((J, 0, 0))),
        (3, 1) => ((B, 0, 0), Some((H, 1, 0))),
        (1, 2) => ((H, 0, 0), Some((J, 0, 0))),
        (2, 2) => ((J, 0, 0), None),
        (3, 2) => ((J, 0, 0), Some((H, 1, 0))),
        (1, 3) => ((H, 0, 0), Some((B, 0, 1))),
        (2, 3) => ((J, 0, 0), Some((B, 0, 1))),
        _ => ((H, 1, 0), Some((B, 0, 1))),
    }
}

impl HalfPel {
    /// Filter the `w` x `h` plane `plane` (row stride `stride`) with
    /// `pad` samples of padding, on up to `threads` threads.
    pub(crate) fn new(
        plane: &[u8],
        stride: usize,
        w: usize,
        h: usize,
        pad: usize,
        threads: usize,
    ) -> Self {
        let ps = w + 2 * pad;
        let rows = h + 2 * pad;
        let mut g = vec![0u8; ps * rows];
        let mut b = vec![0u8; ps * rows];
        let mut hh = vec![0u8; ps * rows];
        let mut j = vec![0u8; ps * rows];
        if w == 0 || h == 0 {
            return Self {
                pad,
                stride: ps,
                rows,
                g,
                b,
                h: hh,
                j,
            };
        }
        // Unrounded horizontal intermediates `b1` of the picture rows,
        // over the padded width.
        let mut b1 = vec![0i16; ps * h];
        let workers = threads.clamp(1, h);
        let rows_per = h.div_ceil(workers);
        // Pass 1 — per picture row: G with its horizontal padding, b1
        // and b.
        let pass1 = |y0: usize, g_rows: &mut [u8], b_rows: &mut [u8], b1_rows: &mut [i16]| {
            // The row extended by the clamp, 2 samples further left and
            // 3 further right than the padded width (the filter reach).
            let mut ext = vec![0i32; ps + 5];
            for (k, ((g_row, b_row), b1_row)) in g_rows
                .chunks_mut(ps)
                .zip(b_rows.chunks_mut(ps))
                .zip(b1_rows.chunks_mut(ps))
                .enumerate()
            {
                let src = &plane[(y0 + k) * stride..][..w];
                for (i, e) in ext.iter_mut().enumerate() {
                    let x = (i as isize - (pad as isize + 2)).clamp(0, w as isize - 1);
                    *e = i32::from(src[x as usize]);
                }
                for x in 0..ps {
                    g_row[x] = ext[x + 2] as u8;
                    let v = tap6(
                        ext[x],
                        ext[x + 1],
                        ext[x + 2],
                        ext[x + 3],
                        ext[x + 4],
                        ext[x + 5],
                    );
                    b1_row[x] = v as i16;
                    b_row[x] = clip8((v + 16) >> 5);
                }
            }
        };
        {
            let g_pic = &mut g[pad * ps..(pad + h) * ps];
            let b_pic = &mut b[pad * ps..(pad + h) * ps];
            if workers == 1 {
                pass1(0, g_pic, b_pic, &mut b1);
            } else {
                std::thread::scope(|s| {
                    for (n, ((gc, bc), b1c)) in g_pic
                        .chunks_mut(rows_per * ps)
                        .zip(b_pic.chunks_mut(rows_per * ps))
                        .zip(b1.chunks_mut(rows_per * ps))
                        .enumerate()
                    {
                        let pass1 = &pass1;
                        s.spawn(move || pass1(n * rows_per, gc, bc, b1c));
                    }
                });
            }
        }
        // Rows above and below the picture repeat its first / last row
        // (eq. 8-240) in G and b.
        for plane_buf in [&mut g, &mut b] {
            let (top, rest) = plane_buf.split_at_mut(pad * ps);
            for r in top.chunks_mut(ps) {
                r.copy_from_slice(&rest[..ps]);
            }
            let (head, bottom) = plane_buf.split_at_mut((pad + h) * ps);
            let last = &head[(pad + h - 1) * ps..];
            for r in bottom.chunks_mut(ps) {
                r.copy_from_slice(last);
            }
        }
        // Pass 2 — per padded row: h from G, j from b1, both through
        // the vertical filter with the row clamp.
        let g_ref = &g;
        let b1_ref = &b1;
        let pass2 = |r0: usize, h_rows: &mut [u8], j_rows: &mut [u8]| {
            for (k, (h_row, j_row)) in h_rows.chunks_mut(ps).zip(j_rows.chunks_mut(ps)).enumerate()
            {
                // Picture row of this padded row and of its six taps.
                let y = (r0 + k) as isize - pad as isize;
                let row = |d: isize| (y + d).clamp(0, h as isize - 1) as usize;
                let rs = [row(-2), row(-1), row(0), row(1), row(2), row(3)];
                let gr: [&[u8]; 6] = std::array::from_fn(|t| &g_ref[(pad + rs[t]) * ps..][..ps]);
                let br: [&[i16]; 6] = std::array::from_fn(|t| &b1_ref[rs[t] * ps..][..ps]);
                for x in 0..ps {
                    let v = tap6(
                        i32::from(gr[0][x]),
                        i32::from(gr[1][x]),
                        i32::from(gr[2][x]),
                        i32::from(gr[3][x]),
                        i32::from(gr[4][x]),
                        i32::from(gr[5][x]),
                    );
                    h_row[x] = clip8((v + 16) >> 5);
                    let v = tap6(
                        i32::from(br[0][x]),
                        i32::from(br[1][x]),
                        i32::from(br[2][x]),
                        i32::from(br[3][x]),
                        i32::from(br[4][x]),
                        i32::from(br[5][x]),
                    );
                    j_row[x] = clip8((v + 512) >> 10);
                }
            }
        };
        let workers = threads.clamp(1, rows);
        if workers == 1 {
            pass2(0, &mut hh, &mut j);
        } else {
            let rows_per = rows.div_ceil(workers);
            std::thread::scope(|s| {
                for (n, (hc, jc)) in hh
                    .chunks_mut(rows_per * ps)
                    .zip(j.chunks_mut(rows_per * ps))
                    .enumerate()
                {
                    let pass2 = &pass2;
                    s.spawn(move || pass2(n * rows_per, hc, jc));
                }
            });
        }
        Self {
            pad,
            stride: ps,
            rows,
            g,
            b,
            h: hh,
            j,
        }
    }

    /// The integer samples from `(x, y)` on (reference coordinates,
    /// the picture-edge clamp applied) to the end of the padded plane,
    /// with the padded row stride; `None` when a `w` x `h` block there
    /// reaches beyond the padding.
    pub(crate) fn integer_block(
        &self,
        x: i32,
        y: i32,
        w: usize,
        h: usize,
    ) -> Option<(&[u8], usize)> {
        let (px, py) = (x + self.pad as i32, y + self.pad as i32);
        if px < 0 || py < 0 || px as usize + w > self.stride || py as usize + h > self.rows {
            return None;
        }
        Some((
            &self.g[py as usize * self.stride + px as usize..],
            self.stride,
        ))
    }

    fn plane(&self, s: Src) -> &[u8] {
        match s {
            Src::G => &self.g,
            Src::B => &self.b,
            Src::H => &self.h,
            Src::J => &self.j,
        }
    }

    /// SAD between the `N` x `N` source block at `(src_x, src_y0)` and
    /// the §8.4.2.2.1 prediction whose integer position in the
    /// reference is `(int_x, int_y)` at fraction `(x_frac, y_frac)`.
    /// `None` when the block reaches beyond the padding.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sad<const N: usize>(
        &self,
        src_y: &[u8],
        src_stride: usize,
        src_x: usize,
        src_y0: usize,
        int_x: i32,
        int_y: i32,
        x_frac: u8,
        y_frac: u8,
    ) -> Option<u32> {
        let x = int_x + self.pad as i32;
        let y = int_y + self.pad as i32;
        // One more column / row for the displaced terms.
        if x < 0 || y < 0 || x as usize + N + 1 > self.stride || y as usize + N + 1 > self.rows {
            return None;
        }
        let (x, y) = (x as usize, y as usize);
        let (first, second) = terms(x_frac, y_frac);
        let at = |t: Term| &self.plane(t.0)[(y + t.2) * self.stride + x + t.1..];
        let pa = at(first);
        let mut sad = 0u32;
        match second {
            None => {
                for r in 0..N {
                    let s: &[u8; N] = src_y[(src_y0 + r) * src_stride + src_x..][..N]
                        .try_into()
                        .expect("source row");
                    let a: &[u8; N] = pa[r * self.stride..][..N].try_into().expect("row");
                    sad += s
                        .iter()
                        .zip(a)
                        .map(|(&s, &a)| u32::from(s.abs_diff(a)))
                        .sum::<u32>();
                }
            }
            Some(t) => {
                let pb = at(t);
                for r in 0..N {
                    let s: &[u8; N] = src_y[(src_y0 + r) * src_stride + src_x..][..N]
                        .try_into()
                        .expect("source row");
                    let a: &[u8; N] = pa[r * self.stride..][..N].try_into().expect("row");
                    let b: &[u8; N] = pb[r * self.stride..][..N].try_into().expect("row");
                    sad += s
                        .iter()
                        .zip(a.iter().zip(b))
                        .map(|(&s, (&a, &b))| {
                            let p = ((u16::from(a) + u16::from(b) + 1) >> 1) as u8;
                            u32::from(s.abs_diff(p))
                        })
                        .sum::<u32>();
                }
            }
        }
        Some(sad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::refwin::RefWindow;

    /// The plane SAD equals the SAD against the block interpolated by
    /// the decoder's routine, for every fraction, at positions inside
    /// the picture, across each edge and wholly outside it, and for
    /// every thread count.
    #[test]
    fn plane_sad_equals_interpolated_block_sad() {
        let (w, h) = (40usize, 36usize);
        let stride = w + 3;
        let reference: Vec<u8> = (0..stride * h)
            .map(|k| {
                let (x, y) = (k % stride, k / stride);
                ((x * 53 + y * 31 + (x * y) % 17 * 9 + (x / 5) * 40) % 256) as u8
            })
            .collect();
        let src: Vec<u8> = (0..stride * h)
            .map(|k| ((k * 97 + 13) % 251) as u8)
            .collect();
        let pad = 24usize;
        let serial = HalfPel::new(&reference, stride, w, h, pad, 1);
        for threads in [2usize, 3, 7] {
            let par = HalfPel::new(&reference, stride, w, h, pad, threads);
            assert!(par.g == serial.g && par.b == serial.b, "threads={threads}");
            assert!(par.h == serial.h && par.j == serial.j, "threads={threads}");
        }
        let positions = [
            (0i32, 0i32),
            (5, 7),
            (-3, 4),
            (-20, -20),
            (w as i32 - 8, 3),
            (w as i32 - 3, h as i32 - 2),
            (w as i32 + 6, h as i32 + 5),
            (9, -17),
            (11, h as i32 + 3),
        ];
        for &(ix, iy) in &positions {
            for yf in 0..4u8 {
                for xf in 0..4u8 {
                    for n in [8usize, 16] {
                        let window = RefWindow::for_luma(&reference, stride, w, h, ix, iy, n, n, 1);
                        let mut pred = vec![0i32; n * n];
                        window.luma(ix, iy, xf, yf, n as u32, n as u32, &mut pred, n);
                        let want: u32 = (0..n * n)
                            .map(|k| {
                                let s = i32::from(src[(2 + k / n) * stride + 3 + k % n]);
                                (s - pred[k].clamp(0, 255)).unsigned_abs()
                            })
                            .sum();
                        let got = if n == 8 {
                            serial.sad::<8>(&src, stride, 3, 2, ix, iy, xf, yf)
                        } else {
                            serial.sad::<16>(&src, stride, 3, 2, ix, iy, xf, yf)
                        };
                        assert_eq!(got, Some(want), "pos ({ix},{iy}) frac ({xf},{yf}) n={n}");
                    }
                }
            }
        }
        // Beyond the padding there is no answer.
        assert_eq!(serial.sad::<16>(&src, stride, 3, 2, -25, 0, 0, 0), None);
        assert_eq!(
            serial.sad::<16>(&src, stride, 3, 2, w as i32 + 8, 0, 1, 1),
            None
        );
    }
}
