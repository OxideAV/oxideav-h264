//! Stable-Rust "manual SIMD" kernels for the H.264 decoder hot paths.
//!
//! These are written so LLVM's auto-vectoriser produces tight AVX2 /
//! NEON code on release builds — no nightly required. The two big
//! wins over the per-pixel scalar reference in `inter_pred.rs` are:
//!
//! 1. **Once-per-row edge clipping.** The scalar path calls `Clip3`
//!    on every coordinate of every tap inside `ref_sample`. The
//!    chunked path materialises an `(w+5)`-wide row of integer
//!    samples once (clipped at the row edges), then runs the FIR on
//!    contiguous indices.
//! 2. **Hoisted horizontal FIR for the diagonal "j" position.** The
//!    scalar `j1_at` recomputes 6 horizontal FIRs per output pixel.
//!    The chunked path computes a `(h+5)`-row strip of horizontal
//!    intermediates exactly once and then runs the vertical FIR on
//!    that strip — turning O(w*h*36) work into O((h+5)*w*6 + w*h*6).

// SIMD inner loops are written as indexed loops so multiple arrays can
// be touched at the same index — clippy::needless_range_loop wants
// `iter_mut().enumerate()` instead, but that gets in the way of LLVM's
// auto-vectoriser because the resulting iterator pattern isn't loop-
// hoistable to a flat AVX2 / NEON SIMD body. Same rationale as
// `oxideav-mpeg4video::simd::chunked` (mod-level `#![allow]`).
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::inter_pred::{InterPredError, InterPredResult};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn clip3(x: i32, y: i32, z: i32) -> i32 {
    if z < x {
        x
    } else if z > y {
        y
    } else {
        z
    }
}

#[inline(always)]
fn clip1(x: i32, max_v: i32) -> i32 {
    if x < 0 {
        0
    } else if x > max_v {
        max_v
    } else {
        x
    }
}

/// 6-tap H.264 luma FIR `s0 - 5*s1 + 20*s2 + 20*s3 - 5*s4 + s5`.
#[inline(always)]
fn tap6(s0: i32, s1: i32, s2: i32, s3: i32, s4: i32, s5: i32) -> i32 {
    s0 - 5 * s1 + 20 * s2 + 20 * s3 - 5 * s4 + s5
}

/// Materialise a row of `w + 5` integer samples starting at horizontal
/// offset `int_x - 2` into `out[0..w+5]`, with §8.4.2.2.1 eq. 8-239
/// edge replication at the row's left and right boundaries. The row
/// itself is `iy` (already vertically clipped by the caller).
#[inline(always)]
fn fill_row(
    out: &mut [i32],
    src: &[i32],
    src_stride: usize,
    src_w: usize,
    iy: usize,
    int_x: i32,
    n: usize,
) {
    // out[k] = src[iy * src_stride + clip3(0, src_w-1, int_x - 2 + k)]
    let row_base = iy * src_stride;
    let xmax = src_w as i32 - 1;
    let x0 = int_x - 2;
    let xn = x0 + n as i32 - 1;
    if x0 >= 0 && xn <= xmax {
        // Fast path: contiguous, no clipping.
        let start = (row_base as i32 + x0) as usize;
        out[..n].copy_from_slice(&src[start..start + n]);
    } else {
        for (k, slot) in out.iter_mut().take(n).enumerate() {
            let cx = clip3(0, xmax, x0 + k as i32) as usize;
            *slot = src[row_base + cx];
        }
    }
}

// ---------------------------------------------------------------------------
// §8.4.2.2.1 — luma fractional-sample interpolation
// ---------------------------------------------------------------------------

/// Maximum partition height we'll handle on the stack. Inter
/// partitions are at most 16x16 (§7.3.5 macroblock_layer); chroma
/// 4:2:0 partitions reach 8x8. We oversize to allow w+5 rows.
const MAX_H: usize = 16 + 5;
/// Maximum partition width + 5-tap halo (so 16+5).
const MAX_W: usize = 16 + 5;

/// Workspace for one luma block. Each row holds `w` H-FIR intermediates
/// (a "b1" strip) for vertical filtering.
struct LumaWs {
    /// `(h + 5)` rows of `w` H-FIR intermediates.
    h_strip: [[i32; MAX_W]; MAX_H],
    /// One scratch row of `w + 5` integer samples for H-FIR.
    int_row: [i32; MAX_W + 8],
}

impl LumaWs {
    #[inline(always)]
    fn new() -> Self {
        Self {
            h_strip: [[0i32; MAX_W]; MAX_H],
            int_row: [0i32; MAX_W + 8],
        }
    }
}

/// Build the integer-sample row at vertical offset `iy_rel` from
/// `int_y` (clipped). Returns the materialised row of `w` samples
/// starting at horizontal offset `int_x` for use as a "G column".
#[inline(always)]
fn integer_row_at(
    src: &[i32],
    src_stride: usize,
    src_w: usize,
    src_h: usize,
    int_x: i32,
    int_y: i32,
    iy_rel: i32,
    w: usize,
    out: &mut [i32; MAX_W],
) {
    let iy = clip3(0, src_h as i32 - 1, int_y + iy_rel) as usize;
    let xmax = src_w as i32 - 1;
    let row_base = iy * src_stride;
    if int_x >= 0 && int_x + w as i32 - 1 <= xmax {
        out[..w].copy_from_slice(&src[row_base + int_x as usize..row_base + int_x as usize + w]);
    } else {
        for (k, slot) in out.iter_mut().take(w).enumerate() {
            let cx = clip3(0, xmax, int_x + k as i32) as usize;
            *slot = src[row_base + cx];
        }
    }
}

/// H-FIR strip: for each row in `int_y - 2 .. int_y + h + 3` (length
/// `h+5`), compute the b1 intermediate at columns `int_x .. int_x + w`
/// and store into `ws.h_strip[r][c]`. Uses `ws.int_row` as scratch.
#[inline(always)]
fn build_h_strip(
    src: &[i32],
    src_stride: usize,
    src_w: usize,
    src_h: usize,
    int_x: i32,
    int_y: i32,
    w: usize,
    h: usize,
    ws: &mut LumaWs,
) {
    let n_int = w + 5; // taps -2..+3 → 6 contiguous samples per output
    let nrows = h + 5;
    for r in 0..nrows {
        let iy_abs = int_y + r as i32 - 2;
        let iy = clip3(0, src_h as i32 - 1, iy_abs) as usize;
        // Materialise the integer row covering taps -2..+3 around int_x..int_x+w.
        fill_row(&mut ws.int_row, src, src_stride, src_w, iy, int_x, n_int);
        // Apply the H-FIR.
        let strip_row = &mut ws.h_strip[r];
        for i in 0..w {
            strip_row[i] = tap6(
                ws.int_row[i],
                ws.int_row[i + 1],
                ws.int_row[i + 2],
                ws.int_row[i + 3],
                ws.int_row[i + 4],
                ws.int_row[i + 5],
            );
        }
    }
}

/// Pure horizontal half-pel block: `(0,0) → b` only requires 1 row of
/// H-FIR per output row.
#[inline(always)]
fn fill_b(
    src: &[i32],
    src_stride: usize,
    src_w: usize,
    src_h: usize,
    int_x: i32,
    int_y: i32,
    w: usize,
    h: usize,
    max_v: i32,
    dst: &mut [i32],
    dst_stride: usize,
    ws: &mut LumaWs,
) {
    let n_int = w + 5;
    for yl in 0..h {
        let iy = clip3(0, src_h as i32 - 1, int_y + yl as i32) as usize;
        fill_row(&mut ws.int_row, src, src_stride, src_w, iy, int_x, n_int);
        let drow = &mut dst[yl * dst_stride..yl * dst_stride + w];
        for i in 0..w {
            let b1 = tap6(
                ws.int_row[i],
                ws.int_row[i + 1],
                ws.int_row[i + 2],
                ws.int_row[i + 3],
                ws.int_row[i + 4],
                ws.int_row[i + 5],
            );
            drow[i] = clip1((b1 + 16) >> 5, max_v);
        }
    }
}

/// Pure vertical half-pel block: `(0,2) → h`.
#[inline(always)]
fn fill_h_vert(
    src: &[i32],
    src_stride: usize,
    src_w: usize,
    src_h: usize,
    int_x: i32,
    int_y: i32,
    w: usize,
    h: usize,
    max_v: i32,
    dst: &mut [i32],
    dst_stride: usize,
) {
    // Materialise (h+5) integer columns at int_x .. int_x+w.
    // We take one row at a time so the cache touches src linearly.
    let mut col_strip: [[i32; MAX_W]; MAX_H] = [[0; MAX_W]; MAX_H];
    let nrows = h + 5;
    for r in 0..nrows {
        let iy = clip3(0, src_h as i32 - 1, int_y + r as i32 - 2) as usize;
        let row_base = iy * src_stride;
        let xmax = src_w as i32 - 1;
        if int_x >= 0 && int_x + w as i32 - 1 <= xmax {
            col_strip[r][..w]
                .copy_from_slice(&src[row_base + int_x as usize..row_base + int_x as usize + w]);
        } else {
            for i in 0..w {
                let cx = clip3(0, xmax, int_x + i as i32) as usize;
                col_strip[r][i] = src[row_base + cx];
            }
        }
    }
    for yl in 0..h {
        let drow = &mut dst[yl * dst_stride..yl * dst_stride + w];
        for i in 0..w {
            let h1 = tap6(
                col_strip[yl][i],
                col_strip[yl + 1][i],
                col_strip[yl + 2][i],
                col_strip[yl + 3][i],
                col_strip[yl + 4][i],
                col_strip[yl + 5][i],
            );
            drow[i] = clip1((h1 + 16) >> 5, max_v);
        }
    }
}

/// §8.4.2.2.1 — luma fractional-sample interpolation (chunked path).
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
    if x_frac > 3 {
        return Err(InterPredError::LumaFracOutOfRange(x_frac));
    }
    if y_frac > 3 {
        return Err(InterPredError::LumaFracOutOfRange(y_frac));
    }
    if w == 0 || h == 0 || w % 4 != 0 || h % 4 != 0 {
        return Err(InterPredError::InvalidBlockSize { w, h });
    }
    // Clamp the workspace size; partitions are at most 16x16 in H.264.
    let wu = w as usize;
    let hu = h as usize;
    if wu > MAX_W || hu > MAX_H {
        // Fall back to scalar for over-large blocks (defensive — the
        // spec caps inter partitions at 16x16, but the bit-exactness
        // contract still holds).
        return super::scalar::interpolate_luma(
            src, src_stride, src_width, src_height, int_x, int_y, x_frac, y_frac, w, h, bit_depth,
            dst, dst_stride,
        );
    }
    let max_v = (1i32 << bit_depth) - 1;

    // Pure-integer (0,0): a strided copy with edge replication.
    if x_frac == 0 && y_frac == 0 {
        for yl in 0..hu {
            let iy = clip3(0, src_height as i32 - 1, int_y + yl as i32) as usize;
            let row_base = iy * src_stride;
            let xmax = src_width as i32 - 1;
            let drow = &mut dst[yl * dst_stride..yl * dst_stride + wu];
            if int_x >= 0 && int_x + wu as i32 - 1 <= xmax {
                drow.copy_from_slice(
                    &src[row_base + int_x as usize..row_base + int_x as usize + wu],
                );
            } else {
                for (i, slot) in drow.iter_mut().enumerate() {
                    let cx = clip3(0, xmax, int_x + i as i32) as usize;
                    *slot = src[row_base + cx];
                }
            }
        }
        return Ok(());
    }

    // Pure-horizontal half-pel (2, 0).
    if x_frac == 2 && y_frac == 0 {
        let mut ws = LumaWs::new();
        fill_b(
            src, src_stride, src_width, src_height, int_x, int_y, wu, hu, max_v, dst, dst_stride,
            &mut ws,
        );
        return Ok(());
    }

    // Pure-vertical half-pel (0, 2).
    if x_frac == 0 && y_frac == 2 {
        fill_h_vert(
            src, src_stride, src_width, src_height, int_x, int_y, wu, hu, max_v, dst, dst_stride,
        );
        return Ok(());
    }

    // Everything else needs at most: a `b` value at (x, y), a `b` value
    // at (x, y+1) (for s/p/q/r/n quarter-pel), a `h` (vertical) at (x, y),
    // a `h` at (x+1, y) (for m, used in 3,*), a `j` (diagonal) at (x, y).
    //
    // For the diagonal-bearing positions (x_frac == 2 || y_frac == 2,
    // and at least one of them being 1/3 makes things mixed) we build
    // the H-strip once, then derive everything from it.
    //
    // Strategy: always build the H-FIR strip when we need any `b` /
    // `j` / `s` / `m`-derived value (which is essentially every
    // non-(0,0)/(0,2)/(2,0) position). Then assemble the position.
    let mut ws = LumaWs::new();
    build_h_strip(
        src, src_stride, src_width, src_height, int_x, int_y, wu, hu, &mut ws,
    );

    // Helpers to read intermediates from the strip.
    // strip[yl + 2][xl] is the H-FIR b1 intermediate at the *target*
    // integer location (xl, yl). The strip rows -2..+2 cover y-2..y+2;
    // strip rows -2 corresponds to ws.h_strip[0], ..., strip row +2 is
    // ws.h_strip[4], the target row (offset 0) is ws.h_strip[2], and
    // row +1 is ws.h_strip[3].

    // Apply the H-strip to assemble the requested position.
    match (x_frac, y_frac) {
        // (2, 0) handled above.
        // (0, 2) handled above.
        // (2, 2): j only — V-FIR over rows -2..+3 of the H-strip.
        (2, 2) => {
            for yl in 0..hu {
                let r0 = &ws.h_strip[yl];
                let r1 = &ws.h_strip[yl + 1];
                let r2 = &ws.h_strip[yl + 2];
                let r3 = &ws.h_strip[yl + 3];
                let r4 = &ws.h_strip[yl + 4];
                let r5 = &ws.h_strip[yl + 5];
                let drow = &mut dst[yl * dst_stride..yl * dst_stride + wu];
                for i in 0..wu {
                    let j1 = tap6(r0[i], r1[i], r2[i], r3[i], r4[i], r5[i]);
                    drow[i] = clip1((j1 + 512) >> 10, max_v);
                }
            }
        }
        // (1, 0) = a: G + b averaged.
        (1, 0) => {
            for yl in 0..hu {
                let b_row = &ws.h_strip[yl + 2]; // y-offset 0 in strip = row 2
                let drow = &mut dst[yl * dst_stride..yl * dst_stride + wu];
                let iy = clip3(0, src_height as i32 - 1, int_y + yl as i32) as usize;
                let row_base = iy * src_stride;
                let xmax = src_width as i32 - 1;
                for i in 0..wu {
                    let cx = if int_x >= 0 && int_x + i as i32 <= xmax {
                        (int_x + i as i32) as usize
                    } else {
                        clip3(0, xmax, int_x + i as i32) as usize
                    };
                    let g = src[row_base + cx];
                    let b_s = clip1((b_row[i] + 16) >> 5, max_v);
                    drow[i] = (g + b_s + 1) >> 1;
                }
            }
        }
        // (3, 0) = c: H (G at x+1) + b averaged.
        (3, 0) => {
            for yl in 0..hu {
                let b_row = &ws.h_strip[yl + 2];
                let drow = &mut dst[yl * dst_stride..yl * dst_stride + wu];
                let iy = clip3(0, src_height as i32 - 1, int_y + yl as i32) as usize;
                let row_base = iy * src_stride;
                let xmax = src_width as i32 - 1;
                for i in 0..wu {
                    let xi = int_x + i as i32 + 1;
                    let cx = clip3(0, xmax, xi) as usize;
                    let h_int = src[row_base + cx];
                    let b_s = clip1((b_row[i] + 16) >> 5, max_v);
                    drow[i] = (h_int + b_s + 1) >> 1;
                }
            }
        }
        // (0, 1) = d: G + h averaged.  (0, 3) = n: M + h averaged.
        (0, 1) | (0, 3) => {
            // Need vertical FIR per output row; reuse the column-strip
            // approach of fill_h_vert but combined with G or M.
            let mut col_strip: [[i32; MAX_W]; MAX_H] = [[0; MAX_W]; MAX_H];
            let nrows = hu + 5;
            for r in 0..nrows {
                let iy = clip3(0, src_height as i32 - 1, int_y + r as i32 - 2) as usize;
                integer_row_at(
                    src,
                    src_stride,
                    src_width,
                    src_height,
                    int_x,
                    iy as i32,
                    0,
                    wu,
                    &mut col_strip[r],
                );
            }
            let g_offset = if y_frac == 1 { 0 } else { 1 };
            for yl in 0..hu {
                let drow = &mut dst[yl * dst_stride..yl * dst_stride + wu];
                for i in 0..wu {
                    let h1 = tap6(
                        col_strip[yl][i],
                        col_strip[yl + 1][i],
                        col_strip[yl + 2][i],
                        col_strip[yl + 3][i],
                        col_strip[yl + 4][i],
                        col_strip[yl + 5][i],
                    );
                    let h_s = clip1((h1 + 16) >> 5, max_v);
                    let g = col_strip[yl + 2 + g_offset][i];
                    drow[i] = (g + h_s + 1) >> 1;
                }
            }
        }
        // (1, 1) = e: b + h averaged.
        // (1, 2) = i: h + j averaged.
        // (1, 3) = p: h + s averaged (s = b at y+1).
        // (3, 1) = g: b + m averaged (m = h at x+1).
        // (3, 2) = k: j + m averaged.
        // (3, 3) = r: m + s averaged.
        // (2, 1) = f: b + j averaged.
        // (2, 3) = q: j + s averaged.
        _ => {
            // General path — derive requested averages from the H-strip
            // plus, when needed, a fresh vertical-FIR on the integer
            // column at the relevant x-offset.
            // For positions involving `m` (x-offset 1) or `h` (x-offset 0)
            // we run the vertical FIR on the integer column.
            // For positions involving `s` (x_frac 2 H-half at y+1) we
            // pick that from the H-strip.
            //
            // Pre-compute the vertical-h column strip if needed.
            let need_h = matches!(
                (x_frac, y_frac),
                (1, 1) | (1, 2) | (1, 3) | (3, 1) | (3, 2) | (3, 3)
            );
            let need_j = matches!((x_frac, y_frac), (1, 2) | (3, 2) | (2, 1) | (2, 3));
            let need_b = matches!(
                (x_frac, y_frac),
                (1, 1) | (3, 1) | (2, 1) | (1, 3) | (3, 3) | (2, 3)
            );

            // h: V-FIR on integer column at (int_x, *).
            // m: V-FIR on integer column at (int_x + 1, *).
            // We always build a column strip wide enough to cover x ∈ 0..w
            // for h, and 0..w (shifted by 1 in x) for m.
            let mut col_strip: [[i32; MAX_W]; MAX_H] = [[0; MAX_W]; MAX_H];
            let mut col_strip_p1: [[i32; MAX_W]; MAX_H] = [[0; MAX_W]; MAX_H];
            let nrows = hu + 5;
            if need_h {
                for r in 0..nrows {
                    let iy = clip3(0, src_height as i32 - 1, int_y + r as i32 - 2) as usize;
                    integer_row_at(
                        src,
                        src_stride,
                        src_width,
                        src_height,
                        int_x,
                        iy as i32,
                        0,
                        wu,
                        &mut col_strip[r],
                    );
                }
            }
            // For x_frac == 3 we additionally need m at (x+1, y).
            let need_m = matches!((x_frac, y_frac), (3, 1) | (3, 2) | (3, 3));
            if need_m {
                for r in 0..nrows {
                    let iy = clip3(0, src_height as i32 - 1, int_y + r as i32 - 2) as usize;
                    integer_row_at(
                        src,
                        src_stride,
                        src_width,
                        src_height,
                        int_x + 1,
                        iy as i32,
                        0,
                        wu,
                        &mut col_strip_p1[r],
                    );
                }
            }

            for yl in 0..hu {
                let drow = &mut dst[yl * dst_stride..yl * dst_stride + wu];
                for i in 0..wu {
                    // b at (x, y): from H-strip at row yl+2.
                    let b_s = if need_b {
                        clip1((ws.h_strip[yl + 2][i] + 16) >> 5, max_v)
                    } else {
                        0
                    };
                    // j at (x, y).
                    let j_s = if need_j {
                        let j1 = tap6(
                            ws.h_strip[yl][i],
                            ws.h_strip[yl + 1][i],
                            ws.h_strip[yl + 2][i],
                            ws.h_strip[yl + 3][i],
                            ws.h_strip[yl + 4][i],
                            ws.h_strip[yl + 5][i],
                        );
                        clip1((j1 + 512) >> 10, max_v)
                    } else {
                        0
                    };
                    // h at (x, y): V-FIR over col_strip rows yl..yl+6 at column i.
                    let h_s = if need_h {
                        let h1 = tap6(
                            col_strip[yl][i],
                            col_strip[yl + 1][i],
                            col_strip[yl + 2][i],
                            col_strip[yl + 3][i],
                            col_strip[yl + 4][i],
                            col_strip[yl + 5][i],
                        );
                        clip1((h1 + 16) >> 5, max_v)
                    } else {
                        0
                    };
                    // m at (x+1, y).
                    let m_s = if need_m {
                        let m1 = tap6(
                            col_strip_p1[yl][i],
                            col_strip_p1[yl + 1][i],
                            col_strip_p1[yl + 2][i],
                            col_strip_p1[yl + 3][i],
                            col_strip_p1[yl + 4][i],
                            col_strip_p1[yl + 5][i],
                        );
                        clip1((m1 + 16) >> 5, max_v)
                    } else {
                        0
                    };
                    // s at (x, y+1): from H-strip at row yl+3.
                    let need_s = matches!((x_frac, y_frac), (1, 3) | (3, 3) | (2, 3));
                    let s_s = if need_s {
                        clip1((ws.h_strip[yl + 3][i] + 16) >> 5, max_v)
                    } else {
                        0
                    };

                    drow[i] = match (x_frac, y_frac) {
                        (1, 1) => (b_s + h_s + 1) >> 1, // e
                        (1, 2) => (h_s + j_s + 1) >> 1, // i
                        (1, 3) => (h_s + s_s + 1) >> 1, // p
                        (2, 1) => (b_s + j_s + 1) >> 1, // f
                        (2, 3) => (j_s + s_s + 1) >> 1, // q
                        (3, 1) => (b_s + m_s + 1) >> 1, // g
                        (3, 2) => (j_s + m_s + 1) >> 1, // k
                        (3, 3) => (m_s + s_s + 1) >> 1, // r
                        _ => unreachable!(),
                    };
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// §8.4.2.2.2 — chroma bilinear interpolation
// ---------------------------------------------------------------------------

/// §8.4.2.2.2 — chroma bilinear interpolation (chunked path).
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
    _bit_depth: u32,
    dst: &mut [i32],
    dst_stride: usize,
) -> InterPredResult<()> {
    if x_frac > 7 {
        return Err(InterPredError::ChromaFracOutOfRange(x_frac));
    }
    if y_frac > 7 {
        return Err(InterPredError::ChromaFracOutOfRange(y_frac));
    }
    if w == 0 || h == 0 {
        return Err(InterPredError::InvalidBlockSize { w, h });
    }
    let wu = w as usize;
    let hu = h as usize;
    let xf = x_frac as i32;
    let yf = y_frac as i32;
    let w8mxf = 8 - xf;
    let w8myf = 8 - yf;
    let xmax = src_width as i32 - 1;
    let ymax = src_height as i32 - 1;

    // Common case: the entire 4x4 / 8x8 chroma window is inside the
    // reference plane → no per-pixel clip3.
    let inside =
        int_x >= 0 && int_y >= 0 && (int_x + wu as i32) <= xmax && (int_y + hu as i32) <= ymax;

    if inside && xf == 0 && yf == 0 {
        // Pure integer copy.
        for yl in 0..hu {
            let iy = (int_y + yl as i32) as usize;
            let row_base = iy * src_stride + int_x as usize;
            dst[yl * dst_stride..yl * dst_stride + wu]
                .copy_from_slice(&src[row_base..row_base + wu]);
        }
        return Ok(());
    }
    if inside {
        // Bilinear, no clipping.
        // Coefficients per (eq. 8-270) are constants over the block.
        let c00 = w8mxf * w8myf;
        let c10 = xf * w8myf;
        let c01 = w8mxf * yf;
        let c11 = xf * yf;
        for yl in 0..hu {
            let iy = (int_y + yl as i32) as usize;
            let row_a = iy * src_stride + int_x as usize;
            let row_c = (iy + 1) * src_stride + int_x as usize;
            let drow = &mut dst[yl * dst_stride..yl * dst_stride + wu];
            for i in 0..wu {
                let sa = src[row_a + i];
                let sb = src[row_a + i + 1];
                let sc = src[row_c + i];
                let sd = src[row_c + i + 1];
                drow[i] = (c00 * sa + c10 * sb + c01 * sc + c11 * sd + 32) >> 6;
            }
        }
        return Ok(());
    }

    // Edge-replicated slow path.
    for yl in 0..hu {
        for xl in 0..wu {
            let ix = int_x + xl as i32;
            let iy = int_y + yl as i32;
            let cxa = clip3(0, xmax, ix) as usize;
            let cxb = clip3(0, xmax, ix + 1) as usize;
            let cya = clip3(0, ymax, iy) as usize;
            let cyc = clip3(0, ymax, iy + 1) as usize;
            let sa = src[cya * src_stride + cxa];
            let sb = src[cya * src_stride + cxb];
            let sc = src[cyc * src_stride + cxa];
            let sd = src[cyc * src_stride + cxb];
            let v =
                (w8mxf * w8myf * sa + xf * w8myf * sb + w8mxf * yf * sc + xf * yf * sd + 32) >> 6;
            dst[yl * dst_stride + xl] = v;
        }
    }
    Ok(())
}
