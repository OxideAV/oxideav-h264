//! Scalar reference implementations for the H.264 hot kernels.
//!
//! Always compiled, used as the fallback on any target and as the
//! oracle for `simd::tests::*_matches_scalar`. These wrap the existing
//! `inter_pred` entry points unchanged so we have a stable bit-exact
//! reference while we develop the chunked/portable variants.

use crate::inter_pred::{
    interpolate_chroma as ip_chroma, interpolate_luma as ip_luma, InterPredResult,
};

/// §8.4.2.2.1 — luma interpolation (scalar reference).
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
    ip_luma(
        src, src_stride, src_width, src_height, int_x, int_y, x_frac, y_frac, w, h, bit_depth, dst,
        dst_stride,
    )
}

/// §8.4.2.2.2 — chroma bilinear interpolation (scalar reference).
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
    ip_chroma(
        src, src_stride, src_width, src_height, int_x, int_y, x_frac, y_frac, w, h, bit_depth, dst,
        dst_stride,
    )
}
