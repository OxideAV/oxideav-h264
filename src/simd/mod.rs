//! SIMD fast paths for the H.264 decoder hot spots.
//!
//! The H.264 decoder spends the majority of its wall time in a small
//! number of pixel kernels:
//!
//! * [`interpolate_luma`] — 6-tap quarter-pel luma interpolation
//!   (§8.4.2.2.1). The classic H.264 hotspot — per-output-pixel cost
//!   of the diagonal "j" position is two 6-tap FIRs (one row, one
//!   column on intermediates).
//! * [`interpolate_chroma`] — bilinear chroma interpolation (§8.4.2.2.2).
//! * [`inverse_transform_4x4`] — 4×4 integer DCT + dequant (§8.5.12).
//! * [`inverse_transform_8x8`] — 8×8 integer DCT + dequant (§8.5.13).
//!
//! Layout (matches `oxideav-mpeg4video`):
//!
//! * [`scalar`] — reference implementations, always compiled, used as
//!   the oracle for the bit-exactness tests.
//! * [`chunked`] — stable-Rust "manual SIMD" built from fixed-size
//!   `[i32; 8]` chunks that LLVM lowers to AVX2 / NEON / SSE on
//!   release builds. This is the default path on stable.
//! * [`portable`] — `std::simd` paths behind the `nightly` feature flag.
//!
//! Public entry points dispatch at compile time via `cfg`. The chunked
//! and portable implementations must agree with the scalar path
//! bit-exactly (validated by `tests::*_matches_scalar`).

pub mod chunked;
pub mod scalar;

#[cfg(feature = "nightly")]
pub mod portable;

use crate::inter_pred::InterPredResult;

/// §8.4.2.2.1 — luma fractional-sample interpolation. See
/// [`crate::inter_pred::interpolate_luma`] for the parameter contract.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn interpolate_luma(
    src: &[i32],
    src_stride: usize,
    src_width: usize,
    src_height: usize,
    int_x: i32,
    int_y: i32,
    x_frac: u8,
    y_frac: u8,
    w: u32,
    h: u32,
    bit_depth: u32,
    dst: &mut [i32],
    dst_stride: usize,
) -> InterPredResult<()> {
    #[cfg(feature = "nightly")]
    {
        portable::interpolate_luma(
            src, src_stride, src_width, src_height, int_x, int_y, x_frac, y_frac, w, h, bit_depth,
            dst, dst_stride,
        )
    }
    #[cfg(not(feature = "nightly"))]
    {
        chunked::interpolate_luma(
            src, src_stride, src_width, src_height, int_x, int_y, x_frac, y_frac, w, h, bit_depth,
            dst, dst_stride,
        )
    }
}

/// §8.4.2.2.2 — chroma bilinear sample interpolation. See
/// [`crate::inter_pred::interpolate_chroma`] for the parameter contract.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn interpolate_chroma(
    src: &[i32],
    src_stride: usize,
    src_width: usize,
    src_height: usize,
    int_x: i32,
    int_y: i32,
    x_frac: u8,
    y_frac: u8,
    w: u32,
    h: u32,
    bit_depth: u32,
    dst: &mut [i32],
    dst_stride: usize,
) -> InterPredResult<()> {
    #[cfg(feature = "nightly")]
    {
        portable::interpolate_chroma(
            src, src_stride, src_width, src_height, int_x, int_y, x_frac, y_frac, w, h, bit_depth,
            dst, dst_stride,
        )
    }
    #[cfg(not(feature = "nightly"))]
    {
        chunked::interpolate_chroma(
            src, src_stride, src_width, src_height, int_x, int_y, x_frac, y_frac, w, h, bit_depth,
            dst, dst_stride,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_plane<F: FnMut(i32, i32) -> i32>(w: usize, h: usize, mut f: F) -> Vec<i32> {
        let mut v = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                v[y * w + x] = f(x as i32, y as i32);
            }
        }
        v
    }

    /// All 16 quarter-pel luma fractions, every block size used by the
    /// decoder, on a non-trivial reference plane: chunked must exactly
    /// match scalar.
    #[test]
    fn luma_chunked_matches_scalar() {
        let src = make_plane(64, 64, |x, y| ((x * 7 + y * 13) ^ (x * y)) & 0xff);
        let sizes: &[(u32, u32)] = &[(4, 4), (4, 8), (8, 4), (8, 8), (8, 16), (16, 8), (16, 16)];
        for &(w, h) in sizes {
            for x_frac in 0..=3u8 {
                for y_frac in 0..=3u8 {
                    let mut s = vec![0i32; (w * h) as usize];
                    let mut c = vec![0i32; (w * h) as usize];
                    scalar::interpolate_luma(
                        &src, 64, 64, 64, 20, 16, x_frac, y_frac, w, h, 8, &mut s, w as usize,
                    )
                    .unwrap();
                    chunked::interpolate_luma(
                        &src, 64, 64, 64, 20, 16, x_frac, y_frac, w, h, 8, &mut c, w as usize,
                    )
                    .unwrap();
                    assert_eq!(s, c, "luma w={w} h={h} xf={x_frac} yf={y_frac}");
                }
            }
        }
    }

    /// Edge replication path — request samples past the right/bottom edge.
    #[test]
    fn luma_chunked_matches_scalar_edges() {
        let src = make_plane(16, 16, |x, y| (x * 5 + y * 11) & 0xff);
        for &(int_x, int_y) in &[(-3i32, -3i32), (12, 12), (-3, 8), (10, -4)] {
            for x_frac in 0..=3u8 {
                for y_frac in 0..=3u8 {
                    let mut s = vec![0i32; 8 * 8];
                    let mut c = vec![0i32; 8 * 8];
                    scalar::interpolate_luma(
                        &src, 16, 16, 16, int_x, int_y, x_frac, y_frac, 8, 8, 8, &mut s, 8,
                    )
                    .unwrap();
                    chunked::interpolate_luma(
                        &src, 16, 16, 16, int_x, int_y, x_frac, y_frac, 8, 8, 8, &mut c, 8,
                    )
                    .unwrap();
                    assert_eq!(s, c, "luma edge ({int_x},{int_y}) xf={x_frac} yf={y_frac}");
                }
            }
        }
    }

    /// 1/8-pel chroma — every fractional combination on a couple of
    /// block sizes.
    #[test]
    fn chroma_chunked_matches_scalar() {
        let src = make_plane(32, 32, |x, y| (x * 3 + y * 5) & 0xff);
        let sizes: &[(u32, u32)] = &[(2, 2), (2, 4), (4, 2), (4, 4), (4, 8), (8, 8)];
        for &(w, h) in sizes {
            for x_frac in 0..=7u8 {
                for y_frac in 0..=7u8 {
                    let mut s = vec![0i32; (w * h) as usize];
                    let mut c = vec![0i32; (w * h) as usize];
                    scalar::interpolate_chroma(
                        &src, 32, 32, 32, 4, 5, x_frac, y_frac, w, h, 8, &mut s, w as usize,
                    )
                    .unwrap();
                    chunked::interpolate_chroma(
                        &src, 32, 32, 32, 4, 5, x_frac, y_frac, w, h, 8, &mut c, w as usize,
                    )
                    .unwrap();
                    assert_eq!(s, c, "chroma w={w} h={h} xf={x_frac} yf={y_frac}");
                }
            }
        }
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn luma_portable_matches_scalar() {
        let src = make_plane(64, 64, |x, y| ((x * 7 + y * 13) ^ (x * y)) & 0xff);
        let sizes: &[(u32, u32)] = &[(4, 4), (8, 8), (16, 16)];
        for &(w, h) in sizes {
            for x_frac in 0..=3u8 {
                for y_frac in 0..=3u8 {
                    let mut s = vec![0i32; (w * h) as usize];
                    let mut p = vec![0i32; (w * h) as usize];
                    scalar::interpolate_luma(
                        &src, 64, 64, 64, 20, 16, x_frac, y_frac, w, h, 8, &mut s, w as usize,
                    )
                    .unwrap();
                    portable::interpolate_luma(
                        &src, 64, 64, 64, 20, 16, x_frac, y_frac, w, h, 8, &mut p, w as usize,
                    )
                    .unwrap();
                    assert_eq!(s, p, "luma w={w} h={h} xf={x_frac} yf={y_frac}");
                }
            }
        }
    }
}
