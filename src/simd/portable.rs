//! `std::simd` paths behind the `nightly` feature flag.
//!
//! This module is only compiled when the `nightly` cargo feature is
//! enabled. The default (stable) build path goes through `chunked`.

use crate::inter_pred::InterPredResult;

/// §8.4.2.2.1 — luma fractional-sample interpolation (portable path).
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
    // Until a portable_simd rewrite lands, defer to the chunked path
    // (which is itself auto-vectorised by LLVM on release builds).
    super::chunked::interpolate_luma(
        src, src_stride, src_width, src_height, int_x, int_y, x_frac, y_frac, w, h, bit_depth, dst,
        dst_stride,
    )
}

/// §8.4.2.2.2 — chroma bilinear interpolation (portable path).
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
    super::chunked::interpolate_chroma(
        src, src_stride, src_width, src_height, int_x, int_y, x_frac, y_frac, w, h, bit_depth, dst,
        dst_stride,
    )
}
