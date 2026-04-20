//! §8.4.2 — Inter-prediction sample generation.
//!
//! Fractional-sample interpolation (§8.4.2.2) and weighted-sample
//! prediction (§8.4.2.3) per ITU-T Rec. H.264 (08/2024). Pure pixel
//! transforms — reference samples in, predicted block out.
//!
//! Conventions:
//! - Samples are carried as `i32` so that the 6-tap FIR intermediate
//!   values (potentially negative, up to ~21 bits at 8-bit depth) do
//!   not overflow and the final `Clip1` step clamps to
//!   `[0, (1 << bit_depth) - 1]`.
//! - `src_stride` is in samples, row-major.
//! - `dst_stride` must be `>= w`.
//! - Reference-picture edge handling is done per §8.4.2.2.1 eq. 8-239 /
//!   8-240 — coordinates are clipped to `[0, width-1]` / `[0, height-1]`
//!   (edge replication), so the caller does NOT need to pad.

#![allow(dead_code)]

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum InterPredError {
    #[error("fractional luma component {0} out of range 0..=3")]
    LumaFracOutOfRange(u8),
    #[error("fractional chroma component {0} out of range 0..=7")]
    ChromaFracOutOfRange(u8),
    #[error("block dimensions must be multiples of 4, got {w}x{h}")]
    InvalidBlockSize { w: u32, h: u32 },
}

pub type InterPredResult<T> = Result<T, InterPredError>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// §5 — `Clip3(x, y, z) = min(max(z, x), y)`.
#[inline]
fn clip3(x: i32, y: i32, z: i32) -> i32 {
    z.max(x).min(y)
}

/// §5 — `Clip1Y(x) = Clip3(0, (1 << BitDepthY) - 1, x)`; likewise Clip1C.
#[inline]
fn clip1(x: i32, bit_depth: u32) -> i32 {
    let max_v = (1i32 << bit_depth) - 1;
    clip3(0, max_v, x)
}

/// Fetch a single reference-plane sample with edge replication, per
/// §8.4.2.2.1 eq. 8-239 / 8-240.
#[inline]
fn ref_sample(
    src: &[i32],
    src_stride: usize,
    src_w: usize,
    src_h: usize,
    x: i32,
    y: i32,
) -> i32 {
    let cx = clip3(0, src_w as i32 - 1, x) as usize;
    let cy = clip3(0, src_h as i32 - 1, y) as usize;
    src[cy * src_stride + cx]
}

/// 6-tap FIR with taps `{1, -5, 20, 20, -5, 1}` applied to six samples
/// `s0..s5` (§8.4.2.2.1 eq. 8-241 / 8-242).
#[inline]
fn tap6(s0: i32, s1: i32, s2: i32, s3: i32, s4: i32, s5: i32) -> i32 {
    s0 - 5 * s1 + 20 * s2 + 20 * s3 - 5 * s4 + s5
}

// ---------------------------------------------------------------------------
// §8.4.2.2.1 — Luma sample interpolation process
// ---------------------------------------------------------------------------

/// Compute the horizontal-half-pel intermediate (`b1`-style) at the
/// integer location (`x`, `y`) by applying the 6-tap FIR horizontally:
/// `E - 5F + 20G + 20H - 5I + J` with E=(x-2,y)..J=(x+3,y).
#[inline]
fn horiz_b1(
    src: &[i32],
    src_stride: usize,
    src_w: usize,
    src_h: usize,
    x: i32,
    y: i32,
) -> i32 {
    let s0 = ref_sample(src, src_stride, src_w, src_h, x - 2, y);
    let s1 = ref_sample(src, src_stride, src_w, src_h, x - 1, y);
    let s2 = ref_sample(src, src_stride, src_w, src_h, x, y);
    let s3 = ref_sample(src, src_stride, src_w, src_h, x + 1, y);
    let s4 = ref_sample(src, src_stride, src_w, src_h, x + 2, y);
    let s5 = ref_sample(src, src_stride, src_w, src_h, x + 3, y);
    tap6(s0, s1, s2, s3, s4, s5)
}

/// Compute the vertical-half-pel intermediate (`h1`-style) at (`x`,`y`)
/// by applying the 6-tap FIR vertically: `A - 5C + 20G + 20M - 5R + T`
/// with A=(x,y-2)..T=(x,y+3).
#[inline]
fn vert_h1(
    src: &[i32],
    src_stride: usize,
    src_w: usize,
    src_h: usize,
    x: i32,
    y: i32,
) -> i32 {
    let s0 = ref_sample(src, src_stride, src_w, src_h, x, y - 2);
    let s1 = ref_sample(src, src_stride, src_w, src_h, x, y - 1);
    let s2 = ref_sample(src, src_stride, src_w, src_h, x, y);
    let s3 = ref_sample(src, src_stride, src_w, src_h, x, y + 1);
    let s4 = ref_sample(src, src_stride, src_w, src_h, x, y + 2);
    let s5 = ref_sample(src, src_stride, src_w, src_h, x, y + 3);
    tap6(s0, s1, s2, s3, s4, s5)
}

/// Vertical 6-tap filter applied to a column of horizontally-filtered
/// intermediate values. Used for the "j" center position (§8.4.2.2.1
/// eq. 8-245): `j1 = cc - 5*dd + 20*h1 + 20*m1 - 5*ee + ff` where
/// cc, dd, h1, m1, ee, ff are all horizontal-FIR intermediates at
/// x = (target) and y = (-2..3 relative to target).
#[inline]
fn j1_at(
    src: &[i32],
    src_stride: usize,
    src_w: usize,
    src_h: usize,
    x: i32,
    y: i32,
) -> i32 {
    // Re-use horiz_b1 at six different y lines.
    let b_m2 = horiz_b1(src, src_stride, src_w, src_h, x, y - 2);
    let b_m1 = horiz_b1(src, src_stride, src_w, src_h, x, y - 1);
    let b_0 = horiz_b1(src, src_stride, src_w, src_h, x, y);
    let b_p1 = horiz_b1(src, src_stride, src_w, src_h, x, y + 1);
    let b_p2 = horiz_b1(src, src_stride, src_w, src_h, x, y + 2);
    let b_p3 = horiz_b1(src, src_stride, src_w, src_h, x, y + 3);
    tap6(b_m2, b_m1, b_0, b_p1, b_p2, b_p3)
}

/// §8.4.2.2.1 — compute `predPartLX_L[xL, yL]` for one output pixel.
/// `ix`/`iy` is the integer sample location (Clip3-handled inside);
/// `(xFrac, yFrac)` selects one of the 16 sub-pel positions per
/// Table 8-12.
fn luma_pred_one(
    src: &[i32],
    src_stride: usize,
    src_w: usize,
    src_h: usize,
    ix: i32,
    iy: i32,
    x_frac: u8,
    y_frac: u8,
    bit_depth: u32,
) -> i32 {
    // Full-pel sample G = reference at (ix, iy).
    #[inline]
    fn g(s: &[i32], st: usize, w: usize, h: usize, x: i32, y: i32) -> i32 {
        ref_sample(s, st, w, h, x, y)
    }

    // Half-pel horizontal sample b at (ix, iy): Clip1Y((b1 + 16) >> 5).
    #[inline]
    fn b_at(
        s: &[i32], st: usize, w: usize, h: usize, x: i32, y: i32, bd: u32,
    ) -> i32 {
        let b1 = horiz_b1(s, st, w, h, x, y);
        clip1((b1 + 16) >> 5, bd)
    }

    // Half-pel vertical sample h at (ix, iy): Clip1Y((h1 + 16) >> 5).
    #[inline]
    fn h_at(
        s: &[i32], st: usize, w: usize, h: usize, x: i32, y: i32, bd: u32,
    ) -> i32 {
        let h1 = vert_h1(s, st, w, h, x, y);
        clip1((h1 + 16) >> 5, bd)
    }

    // Half-pel diagonal sample j at (ix, iy): Clip1Y((j1 + 512) >> 10).
    #[inline]
    fn j_at(
        s: &[i32], st: usize, w: usize, h: usize, x: i32, y: i32, bd: u32,
    ) -> i32 {
        let j1 = j1_at(s, st, w, h, x, y);
        clip1((j1 + 512) >> 10, bd)
    }

    match (x_frac, y_frac) {
        // (0,0) = G
        (0, 0) => g(src, src_stride, src_w, src_h, ix, iy),

        // Table 8-12 column x=0: d, h, n.
        // (0,1) = d = (G + h + 1) >> 1 — eq. 8-252
        (0, 1) => {
            let g = g(src, src_stride, src_w, src_h, ix, iy);
            let h_s = h_at(src, src_stride, src_w, src_h, ix, iy, bit_depth);
            (g + h_s + 1) >> 1
        }
        // (0,2) = h — eq. 8-244
        (0, 2) => h_at(src, src_stride, src_w, src_h, ix, iy, bit_depth),
        // (0,3) = n = (M + h + 1) >> 1 — eq. 8-253, where M=(ix,iy+1).
        (0, 3) => {
            let m = g(src, src_stride, src_w, src_h, ix, iy + 1);
            // h is the half-pel at the SAME integer row (between G above
            // and M below — i.e., centred between G and M). n averages
            // the lower integer with that same h.
            let h_s = h_at(src, src_stride, src_w, src_h, ix, iy, bit_depth);
            (m + h_s + 1) >> 1
        }

        // Column x=1: a, e, i, p.
        // (1,0) = a = (G + b + 1) >> 1 — eq. 8-250
        (1, 0) => {
            let g = g(src, src_stride, src_w, src_h, ix, iy);
            let b_s = b_at(src, src_stride, src_w, src_h, ix, iy, bit_depth);
            (g + b_s + 1) >> 1
        }
        // (1,1) = e = (b + h + 1) >> 1 — eq. 8-258
        (1, 1) => {
            let b_s = b_at(src, src_stride, src_w, src_h, ix, iy, bit_depth);
            let h_s = h_at(src, src_stride, src_w, src_h, ix, iy, bit_depth);
            (b_s + h_s + 1) >> 1
        }
        // (1,2) = i = (h + j + 1) >> 1 — eq. 8-255
        (1, 2) => {
            let h_s = h_at(src, src_stride, src_w, src_h, ix, iy, bit_depth);
            let j_s = j_at(src, src_stride, src_w, src_h, ix, iy, bit_depth);
            (h_s + j_s + 1) >> 1
        }
        // (1,3) = p = (h + s + 1) >> 1 — eq. 8-260, where s is the
        // horizontal half-pel at the row below (between M and N).
        (1, 3) => {
            let h_s = h_at(src, src_stride, src_w, src_h, ix, iy, bit_depth);
            // s lives at (ix, iy+1) as a horizontal-half-pel.
            let s_b1 = horiz_b1(src, src_stride, src_w, src_h, ix, iy + 1);
            let s_s = clip1((s_b1 + 16) >> 5, bit_depth);
            (h_s + s_s + 1) >> 1
        }

        // Column x=2: b, f, j, q.
        // (2,0) = b — eq. 8-243
        (2, 0) => b_at(src, src_stride, src_w, src_h, ix, iy, bit_depth),
        // (2,1) = f = (b + j + 1) >> 1 — eq. 8-254
        (2, 1) => {
            let b_s = b_at(src, src_stride, src_w, src_h, ix, iy, bit_depth);
            let j_s = j_at(src, src_stride, src_w, src_h, ix, iy, bit_depth);
            (b_s + j_s + 1) >> 1
        }
        // (2,2) = j — eq. 8-247
        (2, 2) => j_at(src, src_stride, src_w, src_h, ix, iy, bit_depth),
        // (2,3) = q = (j + s + 1) >> 1 — eq. 8-257
        (2, 3) => {
            let j_s = j_at(src, src_stride, src_w, src_h, ix, iy, bit_depth);
            let s_b1 = horiz_b1(src, src_stride, src_w, src_h, ix, iy + 1);
            let s_s = clip1((s_b1 + 16) >> 5, bit_depth);
            (j_s + s_s + 1) >> 1
        }

        // Column x=3: c, g, k, r.
        // (3,0) = c = (H + b + 1) >> 1 — eq. 8-251, H=(ix+1, iy).
        (3, 0) => {
            let h_int = g(src, src_stride, src_w, src_h, ix + 1, iy);
            let b_s = b_at(src, src_stride, src_w, src_h, ix, iy, bit_depth);
            (h_int + b_s + 1) >> 1
        }
        // (3,1) = g = (b + m + 1) >> 1 — eq. 8-259. m is vertical-half
        // pel at (ix+1, iy).
        (3, 1) => {
            let b_s = b_at(src, src_stride, src_w, src_h, ix, iy, bit_depth);
            let m_h1 = vert_h1(src, src_stride, src_w, src_h, ix + 1, iy);
            let m_s = clip1((m_h1 + 16) >> 5, bit_depth);
            (b_s + m_s + 1) >> 1
        }
        // (3,2) = k = (j + m + 1) >> 1 — eq. 8-256
        (3, 2) => {
            let j_s = j_at(src, src_stride, src_w, src_h, ix, iy, bit_depth);
            let m_h1 = vert_h1(src, src_stride, src_w, src_h, ix + 1, iy);
            let m_s = clip1((m_h1 + 16) >> 5, bit_depth);
            (j_s + m_s + 1) >> 1
        }
        // (3,3) = r = (m + s + 1) >> 1 — eq. 8-261
        (3, 3) => {
            let m_h1 = vert_h1(src, src_stride, src_w, src_h, ix + 1, iy);
            let m_s = clip1((m_h1 + 16) >> 5, bit_depth);
            let s_b1 = horiz_b1(src, src_stride, src_w, src_h, ix, iy + 1);
            let s_s = clip1((s_b1 + 16) >> 5, bit_depth);
            (m_s + s_s + 1) >> 1
        }

        // Unreachable — validated up front.
        _ => unreachable!(),
    }
}

/// Luma fractional-sample interpolation. Output `dst` is row-major with
/// stride `dst_stride` (must be >= `w`). `src` is the reference plane
/// with stride `src_stride`. `int_x`/`int_y` point at the integer-
/// position sample corresponding to fractional position (0, 0) in the
/// output block. `x_frac`, `y_frac` ∈ 0..=3.
///
/// Samples outside the picture are replicated from the edge per §8.4.2.2.
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
    for yl in 0..h as i32 {
        for xl in 0..w as i32 {
            let v = luma_pred_one(
                src,
                src_stride,
                src_width,
                src_height,
                int_x + xl,
                int_y + yl,
                x_frac,
                y_frac,
                bit_depth,
            );
            dst[(yl as usize) * dst_stride + xl as usize] = v;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// §8.4.2.2.2 — Chroma sample interpolation process
// ---------------------------------------------------------------------------

/// §8.4.2.2.3 — chroma fractional-sample interpolation.
/// `x_frac`/`y_frac` ∈ 0..=7 (1/8-pel for ChromaArrayType==1; caller is
/// responsible for deriving the fractional components per §8.4.2.2
/// eq. 8-227 / 8-229 etc.).
///
/// Formula (eq. 8-270):
/// `((8 - xF) * (8 - yF) * A + xF * (8 - yF) * B
///   + (8 - xF) * yF * C + xF * yF * D + 32) >> 6`
///
/// where A,B,C,D are the four surrounding integer chroma samples
/// after edge replication per eq. 8-262 .. 8-269.
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
    if w == 0 || h == 0 || w % 4 != 0 || h % 4 != 0 {
        return Err(InterPredError::InvalidBlockSize { w, h });
    }

    let xf = x_frac as i32;
    let yf = y_frac as i32;
    let w8mxf = 8 - xf;
    let w8myf = 8 - yf;

    for yl in 0..h as i32 {
        for xl in 0..w as i32 {
            // Eq. 8-262 .. 8-269 — four integer neighbours.
            let ix = int_x + xl;
            let iy = int_y + yl;
            let sa = ref_sample(src, src_stride, src_width, src_height, ix, iy);
            let sb = ref_sample(src, src_stride, src_width, src_height, ix + 1, iy);
            let sc = ref_sample(src, src_stride, src_width, src_height, ix, iy + 1);
            let sd = ref_sample(src, src_stride, src_width, src_height, ix + 1, iy + 1);

            // Eq. 8-270.
            let v = (w8mxf * w8myf * sa
                + xf * w8myf * sb
                + w8mxf * yf * sc
                + xf * yf * sd
                + 32)
                >> 6;
            dst[(yl as usize) * dst_stride + xl as usize] = v;
            // Note: chroma bilinear result is already in [0, (1<<BitDepthC)-1]
            // when A..D are — no separate Clip1C is specified.
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// §8.4.2.3 — Weighted sample prediction
// ---------------------------------------------------------------------------

/// Weighted-pred parameters for a single list entry.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct WeightedEntry {
    pub weight: i32,
    pub offset: i32,
}

/// Which lists contribute to this prediction.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BiPredMode {
    L0Only,
    L1Only,
    Bipred,
}

/// §8.4.2.3.1 — default weighted prediction.
///
/// - If only one list is available (L0 or L1), passthrough.
/// - If both lists contribute, `(predL0 + predL1 + 1) >> 1`.
///
/// This entry point always takes both buffers and averages them
/// (eq. 8-273). For single-list default behaviour callers can simply
/// copy the list buffer through themselves, or pass the same buffer
/// twice — the caller is typically `weighted_pred_explicit` when the
/// weighted_pred_flag is off.
pub fn weighted_pred_default(
    pred_l0: &[i32],
    pred_l1: &[i32],
    stride: usize,
    w: u32,
    h: u32,
    _bit_depth: u32,
    dst: &mut [i32],
    dst_stride: usize,
) {
    for y in 0..h as usize {
        for x in 0..w as usize {
            let a = pred_l0[y * stride + x];
            let b = pred_l1[y * stride + x];
            // Eq. 8-273.
            dst[y * dst_stride + x] = (a + b + 1) >> 1;
        }
    }
}

/// §8.4.2.3.2 — explicit weighted sample prediction.
///
/// Formulas (eq. 8-274 / 8-275 / 8-276):
///
/// * L0 only, `logWD >= 1`:
///   `Clip1(((predL0 * w0 + 2^(logWD-1)) >> logWD) + o0)`
///   `logWD == 0`:
///   `Clip1(predL0 * w0 + o0)`
/// * L1 only — symmetric.
/// * Bipred:
///   `Clip1(((predL0*w0 + predL1*w1 + 2^logWD) >> (logWD+1))
///           + ((o0 + o1 + 1) >> 1))`
pub fn weighted_pred_explicit(
    pred_l0: Option<&[i32]>,
    pred_l1: Option<&[i32]>,
    stride: usize,
    w: u32,
    h: u32,
    mode: BiPredMode,
    weight_l0: WeightedEntry,
    weight_l1: WeightedEntry,
    log2_wd: u32,
    bit_depth: u32,
    dst: &mut [i32],
    dst_stride: usize,
) {
    match mode {
        BiPredMode::L0Only => {
            let l0 = pred_l0.expect("L0 buffer required in L0Only mode");
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let p = l0[y * stride + x];
                    let v = if log2_wd >= 1 {
                        // Eq. 8-274, branch logWD>=1.
                        ((p * weight_l0.weight + (1i32 << (log2_wd - 1)))
                            >> log2_wd)
                            + weight_l0.offset
                    } else {
                        // Eq. 8-274, branch logWD==0.
                        p * weight_l0.weight + weight_l0.offset
                    };
                    dst[y * dst_stride + x] = clip1(v, bit_depth);
                }
            }
        }
        BiPredMode::L1Only => {
            let l1 = pred_l1.expect("L1 buffer required in L1Only mode");
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let p = l1[y * stride + x];
                    // Eq. 8-275.
                    let v = if log2_wd >= 1 {
                        ((p * weight_l1.weight + (1i32 << (log2_wd - 1)))
                            >> log2_wd)
                            + weight_l1.offset
                    } else {
                        p * weight_l1.weight + weight_l1.offset
                    };
                    dst[y * dst_stride + x] = clip1(v, bit_depth);
                }
            }
        }
        BiPredMode::Bipred => {
            let l0 = pred_l0.expect("L0 buffer required in Bipred mode");
            let l1 = pred_l1.expect("L1 buffer required in Bipred mode");
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let p0 = l0[y * stride + x];
                    let p1 = l1[y * stride + x];
                    // Eq. 8-276.
                    let sum = p0 * weight_l0.weight
                        + p1 * weight_l1.weight
                        + (1i32 << log2_wd);
                    let v = (sum >> (log2_wd + 1))
                        + ((weight_l0.offset + weight_l1.offset + 1) >> 1);
                    dst[y * dst_stride + x] = clip1(v, bit_depth);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a ref-plane of size `w x h` filled by a closure.
    fn make_plane<F: FnMut(i32, i32) -> i32>(w: usize, h: usize, mut f: F) -> Vec<i32> {
        let mut v = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..w {
                v[y * w + x] = f(x as i32, y as i32);
            }
        }
        v
    }

    // ---- Luma integer-position tests ----

    #[test]
    fn luma_integer_4x4_copy() {
        let src = make_plane(16, 16, |x, y| (x + y) & 0xff);
        let mut dst = vec![0i32; 4 * 4];
        interpolate_luma(&src, 16, 16, 16, 4, 4, 0, 0, 4, 4, 8, &mut dst, 4)
            .unwrap();
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dst[y * 4 + x], ((x + 4 + y + 4) & 0xff) as i32);
            }
        }
    }

    #[test]
    fn luma_integer_8x8_copy() {
        let src = make_plane(32, 32, |x, y| ((x * 3 + y * 5) & 0xff) as i32);
        let mut dst = vec![0i32; 8 * 8];
        interpolate_luma(&src, 32, 32, 32, 8, 10, 0, 0, 8, 8, 8, &mut dst, 8)
            .unwrap();
        for y in 0..8 {
            for x in 0..8 {
                let ex = ((8 + x as i32) * 3 + (10 + y as i32) * 5) & 0xff;
                assert_eq!(dst[y * 8 + x], ex);
            }
        }
    }

    #[test]
    fn luma_integer_16x16_copy() {
        let src = make_plane(64, 64, |x, y| ((x ^ y) & 0xff) as i32);
        let mut dst = vec![0i32; 16 * 16];
        interpolate_luma(&src, 64, 64, 64, 12, 20, 0, 0, 16, 16, 8, &mut dst, 16)
            .unwrap();
        for y in 0..16 {
            for x in 0..16 {
                let ex = ((12 + x as i32) ^ (20 + y as i32)) & 0xff;
                assert_eq!(dst[y * 16 + x], ex);
            }
        }
    }

    // ---- Luma half-pel horizontal (xFrac=2, yFrac=0) ----

    #[test]
    fn luma_half_pel_horizontal_constant_plane() {
        // Constant plane -> 6-tap FIR yields constant: (1-5+20+20-5+1)*v/32 = 32v/32 = v.
        let src = make_plane(32, 32, |_, _| 128);
        let mut dst = vec![0i32; 4 * 4];
        interpolate_luma(&src, 32, 32, 32, 10, 10, 2, 0, 4, 4, 8, &mut dst, 4)
            .unwrap();
        for v in &dst {
            assert_eq!(*v, 128);
        }
    }

    #[test]
    fn luma_half_pel_horizontal_known_sample() {
        // Hand-compute the 6-tap filter at one location with crafted input.
        // We build a plane where row 10 columns 8..=13 are {10, 20, 30, 40, 50, 60}.
        // For output at (ix=10, iy=10, xFrac=2, yFrac=0):
        //   b1 = E(-2) - 5*F(-1) + 20*G(0) + 20*H(1) - 5*I(2) + J(3)
        //       at x = 10-2..10+3 = 8..13 -> 10 -5*20 + 20*30 + 20*40 - 5*50 + 60
        //       = 10 - 100 + 600 + 800 - 250 + 60 = 1120
        //   b  = Clip1_8((1120 + 16) >> 5) = 1136 >> 5 = 35.5 -> 35.
        let mut src = vec![0i32; 32 * 32];
        let row_vals: [i32; 6] = [10, 20, 30, 40, 50, 60];
        for (i, v) in row_vals.iter().enumerate() {
            src[10 * 32 + 8 + i] = *v;
        }
        let mut dst = vec![0i32; 4];
        // Compute just one sample (4x4 block is fine — other pixels depend
        // on zeros around, but we check only the target pixel (0,0) which
        // corresponds to ix=10,iy=10).
        interpolate_luma(&src, 32, 32, 32, 10, 10, 2, 0, 4, 4, 8, &mut vec![0i32; 16], 4)
            .unwrap();
        // Recompute for that pixel explicitly:
        let mut dst_one = vec![0i32; 16];
        interpolate_luma(&src, 32, 32, 32, 10, 10, 2, 0, 4, 4, 8, &mut dst_one, 4).unwrap();
        let _ = &mut dst; // silence unused
        assert_eq!(dst_one[0], 35);
    }

    // ---- Luma half-pel vertical (xFrac=0, yFrac=2) ----

    #[test]
    fn luma_half_pel_vertical_constant() {
        let src = make_plane(32, 32, |_, _| 200);
        let mut dst = vec![0i32; 4 * 4];
        interpolate_luma(&src, 32, 32, 32, 10, 10, 0, 2, 4, 4, 8, &mut dst, 4)
            .unwrap();
        for v in &dst {
            assert_eq!(*v, 200);
        }
    }

    #[test]
    fn luma_half_pel_vertical_known_sample() {
        // Column 10, rows 8..=13 = {10, 20, 30, 40, 50, 60}.
        // h1 = A - 5C + 20G + 20M - 5R + T = 10 - 5*20 + 20*30 + 20*40 - 5*50 + 60 = 1120
        // h  = Clip1((1120+16)>>5) = 1136>>5 = 35.
        let mut src = vec![0i32; 32 * 32];
        let col_vals: [i32; 6] = [10, 20, 30, 40, 50, 60];
        for (i, v) in col_vals.iter().enumerate() {
            src[(8 + i) * 32 + 10] = *v;
        }
        let mut dst = vec![0i32; 16];
        interpolate_luma(&src, 32, 32, 32, 10, 10, 0, 2, 4, 4, 8, &mut dst, 4).unwrap();
        assert_eq!(dst[0], 35);
    }

    // ---- Luma diagonal j (xFrac=2, yFrac=2) ----

    #[test]
    fn luma_half_pel_diagonal_j_constant() {
        // Constant plane — j1 = 32 * horiz-filter-of-constant = 32 * (32*v)/? Actually
        // we compute j1 = tap6 applied to six b1 values, each equal to 32*v.
        // tap6 of six constants `c`: c - 5c + 20c + 20c - 5c + c = 32*c.
        // So j1 = 32 * (32*v) = 1024*v. j = Clip1((1024v + 512) >> 10) = v.
        let src = make_plane(32, 32, |_, _| 100);
        let mut dst = vec![0i32; 4 * 4];
        interpolate_luma(&src, 32, 32, 32, 10, 10, 2, 2, 4, 4, 8, &mut dst, 4)
            .unwrap();
        for v in &dst {
            assert_eq!(*v, 100);
        }
    }

    #[test]
    fn luma_half_pel_diagonal_j_known_sample() {
        // Construct a plane where six consecutive rows each have the
        // same "row pattern" as the horizontal test (cols 8..=13 =
        // 10,20,30,40,50,60). Rows 8..=13.
        //
        //   For any of those 6 rows, horiz_b1 at (x=10, y) = 1120.
        //   So j1 = 1120 - 5*1120 + 20*1120 + 20*1120 - 5*1120 + 1120
        //         = (1 -5 +20 +20 -5 +1) * 1120 = 32 * 1120 = 35840.
        //   j = Clip1((35840 + 512) >> 10) = 36352 >> 10 = 35.5 -> 35.
        let mut src = vec![0i32; 32 * 32];
        let row_vals: [i32; 6] = [10, 20, 30, 40, 50, 60];
        for row in 8..=13 {
            for (i, v) in row_vals.iter().enumerate() {
                src[row * 32 + 8 + i] = *v;
            }
        }
        let mut dst = vec![0i32; 16];
        interpolate_luma(&src, 32, 32, 32, 10, 10, 2, 2, 4, 4, 8, &mut dst, 4).unwrap();
        assert_eq!(dst[0], 35);
    }

    // ---- Luma quarter-pel (xFrac=1, yFrac=0: position a) ----

    #[test]
    fn luma_quarter_pel_position_a() {
        // Column-block with known values at row 10, cols 8..=13 = {10,20,30,40,50,60}.
        // Integer G = ref(10, 10) = 30 (col 10).
        // b (half-pel at (10,10)) = 35 as above.
        // a = (G + b + 1) >> 1 = (30 + 35 + 1) >> 1 = 66>>1 = 33.
        let mut src = vec![0i32; 32 * 32];
        let row_vals: [i32; 6] = [10, 20, 30, 40, 50, 60];
        for (i, v) in row_vals.iter().enumerate() {
            src[10 * 32 + 8 + i] = *v;
        }
        let mut dst = vec![0i32; 16];
        interpolate_luma(&src, 32, 32, 32, 10, 10, 1, 0, 4, 4, 8, &mut dst, 4).unwrap();
        assert_eq!(dst[0], 33);
    }

    // ---- Luma edge replication ----

    #[test]
    fn luma_edge_replication() {
        // 8x8 ref plane filled with 50 everywhere. Ask for an output at
        // int_x = -5, int_y = -5 (far off the upper-left corner). With
        // edge replication every tap reads 50, so output = 50 everywhere.
        let src = make_plane(8, 8, |_, _| 50);
        let mut dst = vec![0i32; 4 * 4];
        interpolate_luma(&src, 8, 8, 8, -5, -5, 2, 2, 4, 4, 8, &mut dst, 4).unwrap();
        for v in &dst {
            assert_eq!(*v, 50);
        }
    }

    #[test]
    fn luma_edge_replication_integer_past_right() {
        // Right-edge column = 200, rest = 100. Request output starting
        // int_x = src_width (past right edge). Every reference column
        // read clamps to src_width-1 (= col 7), so we should read 200
        // for every sample -> integer copy outputs 200.
        let mut src = vec![100i32; 8 * 8];
        for y in 0..8 {
            src[y * 8 + 7] = 200;
        }
        let mut dst = vec![0i32; 16];
        interpolate_luma(&src, 8, 8, 8, 8, 2, 0, 0, 4, 4, 8, &mut dst, 4).unwrap();
        for v in &dst {
            assert_eq!(*v, 200);
        }
    }

    // ---- Chroma integer copy ----

    #[test]
    fn chroma_integer_copy() {
        let src = make_plane(16, 16, |x, y| ((x * 7 + y) & 0xff) as i32);
        let mut dst = vec![0i32; 4 * 4];
        interpolate_chroma(&src, 16, 16, 16, 2, 3, 0, 0, 4, 4, 8, &mut dst, 4)
            .unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let ex = (((2 + x as i32) * 7 + (3 + y as i32)) & 0xff) as i32;
                assert_eq!(dst[y * 4 + x], ex);
            }
        }
    }

    // ---- Chroma bilinear at known corners ----

    #[test]
    fn chroma_bilinear_center_known_corners() {
        // 2x2 ref with corners A=10, B=20, C=30, D=40.
        // Actually use a plane where (0,0)=10, (1,0)=20, (0,1)=30,
        // (1,1)=40, padded further with whatever (we only read A/B/C/D
        // and the 4x4 block reads from (0,0)..(4,4)). For this test we
        // only inspect output[0] where A/B/C/D are those four.
        //
        // xF=4, yF=4 -> (4*4 + 4*4 + 4*4 + 4*4 -- wait let's compute):
        //   ((8-4)*(8-4)*10 + 4*(8-4)*20 + (8-4)*4*30 + 4*4*40 + 32) >> 6
        //   = (16*10 + 16*20 + 16*30 + 16*40 + 32) >> 6
        //   = (160 + 320 + 480 + 640 + 32) >> 6
        //   = 1632 >> 6
        //   = 25.5 -> 25
        //
        // Build a larger plane and place the 4 corners at (2,2)..(3,3).
        let mut src = vec![0i32; 8 * 8];
        src[2 * 8 + 2] = 10;
        src[2 * 8 + 3] = 20;
        src[3 * 8 + 2] = 30;
        src[3 * 8 + 3] = 40;
        let mut dst = vec![0i32; 16];
        // int_x=2, int_y=2, so output[0] uses (2,2), (3,2), (2,3), (3,3).
        interpolate_chroma(&src, 8, 8, 8, 2, 2, 4, 4, 4, 4, 8, &mut dst, 4)
            .unwrap();
        assert_eq!(dst[0], 25);
    }

    #[test]
    fn chroma_bilinear_zero_frac_passthrough() {
        // xFrac=yFrac=0 -> output = (64*A + 32)>>6 = A (since 32<64).
        let src = make_plane(16, 16, |x, y| ((x * 11 + y * 13) & 0xff) as i32);
        let mut dst = vec![0i32; 4 * 4];
        interpolate_chroma(&src, 16, 16, 16, 1, 2, 0, 0, 4, 4, 8, &mut dst, 4)
            .unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let ex = (((1 + x as i32) * 11 + (2 + y as i32) * 13) & 0xff) as i32;
                assert_eq!(dst[y * 4 + x], ex);
            }
        }
    }

    #[test]
    fn chroma_bilinear_full_right_frac() {
        // xFrac=8 is out of range (1/8-pel is 0..=7). We check xFrac=7
        // is accepted.
        let src = make_plane(16, 16, |x, _| x);
        let mut dst = vec![0i32; 4 * 4];
        assert!(interpolate_chroma(&src, 16, 16, 16, 2, 2, 7, 0, 4, 4, 8, &mut dst, 4).is_ok());
        // xFrac=8 -> error
        assert!(matches!(
            interpolate_chroma(&src, 16, 16, 16, 2, 2, 8, 0, 4, 4, 8, &mut dst, 4),
            Err(InterPredError::ChromaFracOutOfRange(8))
        ));
    }

    // ---- Weighted pred default ----

    #[test]
    fn weighted_default_bipred_roundup() {
        let l0 = vec![10i32; 16];
        let l1 = vec![20i32; 16];
        let mut dst = vec![0i32; 16];
        weighted_pred_default(&l0, &l1, 4, 4, 4, 8, &mut dst, 4);
        // (10 + 20 + 1) >> 1 = 31>>1 = 15.
        for v in &dst {
            assert_eq!(*v, 15);
        }
    }

    #[test]
    fn weighted_default_bipred_zero_plus_one() {
        let l0 = vec![0i32; 16];
        let l1 = vec![1i32; 16];
        let mut dst = vec![0i32; 16];
        weighted_pred_default(&l0, &l1, 4, 4, 4, 8, &mut dst, 4);
        // (0 + 1 + 1) >> 1 = 1.
        for v in &dst {
            assert_eq!(*v, 1);
        }
    }

    // ---- Weighted pred explicit ----

    #[test]
    fn weighted_explicit_l0_only_logwd_nonzero() {
        let l0 = vec![50i32; 16];
        let mut dst = vec![0i32; 16];
        // w0 = 64, o0 = 10, log2_wd = 6.
        // v = ((50 * 64 + (1<<5)) >> 6) + 10 = ((3200 + 32) >> 6) + 10
        //   = (3232 >> 6) + 10 = 50 + 10 = 60.
        weighted_pred_explicit(
            Some(&l0),
            None,
            4,
            4,
            4,
            BiPredMode::L0Only,
            WeightedEntry { weight: 64, offset: 10 },
            WeightedEntry::default(),
            6,
            8,
            &mut dst,
            4,
        );
        for v in &dst {
            assert_eq!(*v, 60);
        }
    }

    #[test]
    fn weighted_explicit_l0_only_logwd_zero() {
        let l0 = vec![10i32; 16];
        let mut dst = vec![0i32; 16];
        // w0=2, o0=5, logWD=0 -> v = 10*2 + 5 = 25. Clip1Y (8-bit) -> 25.
        weighted_pred_explicit(
            Some(&l0),
            None,
            4,
            4,
            4,
            BiPredMode::L0Only,
            WeightedEntry { weight: 2, offset: 5 },
            WeightedEntry::default(),
            0,
            8,
            &mut dst,
            4,
        );
        for v in &dst {
            assert_eq!(*v, 25);
        }
    }

    #[test]
    fn weighted_explicit_l1_only() {
        let l1 = vec![30i32; 16];
        let mut dst = vec![0i32; 16];
        // w1=32, o1=-5, logWD=5 -> v = ((30*32 + 16) >> 5) + (-5)
        //   = (960 + 16) >> 5 - 5 = 976 >> 5 - 5 = 30 - 5 = 25.
        weighted_pred_explicit(
            None,
            Some(&l1),
            4,
            4,
            4,
            BiPredMode::L1Only,
            WeightedEntry::default(),
            WeightedEntry { weight: 32, offset: -5 },
            5,
            8,
            &mut dst,
            4,
        );
        for v in &dst {
            assert_eq!(*v, 25);
        }
    }

    #[test]
    fn weighted_explicit_bipred_known() {
        let l0 = vec![100i32; 16];
        let l1 = vec![200i32; 16];
        let mut dst = vec![0i32; 16];
        // w0=32, w1=32, o0=0, o1=0, logWD=5.
        // sum = 100*32 + 200*32 + (1<<5) = 3200 + 6400 + 32 = 9632
        // v = (9632 >> 6) + ((0+0+1)>>1) = 150 + 0 = 150.
        weighted_pred_explicit(
            Some(&l0),
            Some(&l1),
            4,
            4,
            4,
            BiPredMode::Bipred,
            WeightedEntry { weight: 32, offset: 0 },
            WeightedEntry { weight: 32, offset: 0 },
            5,
            8,
            &mut dst,
            4,
        );
        for v in &dst {
            assert_eq!(*v, 150);
        }
    }

    #[test]
    fn weighted_explicit_bipred_with_offset() {
        let l0 = vec![40i32; 16];
        let l1 = vec![60i32; 16];
        let mut dst = vec![0i32; 16];
        // w0=w1=32, o0=5, o1=7, logWD=5.
        // sum = 40*32 + 60*32 + 32 = 1280 + 1920 + 32 = 3232
        // v = (3232 >> 6) + ((5+7+1)>>1) = 50 + (13>>1) = 50 + 6 = 56.
        weighted_pred_explicit(
            Some(&l0),
            Some(&l1),
            4,
            4,
            4,
            BiPredMode::Bipred,
            WeightedEntry { weight: 32, offset: 5 },
            WeightedEntry { weight: 32, offset: 7 },
            5,
            8,
            &mut dst,
            4,
        );
        for v in &dst {
            assert_eq!(*v, 56);
        }
    }

    #[test]
    fn weighted_explicit_clips_to_max() {
        let l0 = vec![255i32; 16];
        let mut dst = vec![0i32; 16];
        // w0=64, o0=100, logWD=6 -> v = ((255*64 + 32)>>6) + 100 = 255 + 100 = 355
        // Clip1_8 -> 255.
        weighted_pred_explicit(
            Some(&l0),
            None,
            4,
            4,
            4,
            BiPredMode::L0Only,
            WeightedEntry { weight: 64, offset: 100 },
            WeightedEntry::default(),
            6,
            8,
            &mut dst,
            4,
        );
        for v in &dst {
            assert_eq!(*v, 255);
        }
    }

    #[test]
    fn weighted_explicit_clips_to_zero() {
        let l0 = vec![10i32; 16];
        let mut dst = vec![0i32; 16];
        // w0=64, o0=-100 -> v = 10 - 100 = -90. Clip1 -> 0.
        weighted_pred_explicit(
            Some(&l0),
            None,
            4,
            4,
            4,
            BiPredMode::L0Only,
            WeightedEntry { weight: 64, offset: -100 },
            WeightedEntry::default(),
            6,
            8,
            &mut dst,
            4,
        );
        for v in &dst {
            assert_eq!(*v, 0);
        }
    }

    // ---- Clip1 boundary ----

    #[test]
    fn clip1_bounds() {
        assert_eq!(clip1(-5, 8), 0);
        assert_eq!(clip1(0, 8), 0);
        assert_eq!(clip1(255, 8), 255);
        assert_eq!(clip1(1000, 8), 255);
        assert_eq!(clip1(1023, 10), 1023);
        assert_eq!(clip1(4095, 12), 4095);
    }

    // ---- API validation ----

    #[test]
    fn luma_rejects_bad_frac() {
        let src = vec![0i32; 16 * 16];
        let mut dst = vec![0i32; 16];
        assert!(matches!(
            interpolate_luma(&src, 16, 16, 16, 0, 0, 4, 0, 4, 4, 8, &mut dst, 4),
            Err(InterPredError::LumaFracOutOfRange(4))
        ));
        assert!(matches!(
            interpolate_luma(&src, 16, 16, 16, 0, 0, 0, 5, 4, 4, 8, &mut dst, 4),
            Err(InterPredError::LumaFracOutOfRange(5))
        ));
    }

    #[test]
    fn luma_rejects_bad_block_size() {
        let src = vec![0i32; 16 * 16];
        let mut dst = vec![0i32; 16];
        assert!(matches!(
            interpolate_luma(&src, 16, 16, 16, 0, 0, 0, 0, 5, 4, 8, &mut dst, 5),
            Err(InterPredError::InvalidBlockSize { w: 5, h: 4 })
        ));
    }

    // ---- 10-bit luma Clip1 ----

    #[test]
    fn luma_half_pel_horizontal_10bit_clip() {
        // At 10-bit depth clip1 bound is 1023. Use a pattern that would
        // overshoot an 8-bit clamp but not 10-bit.
        //   row values 200..700 evenly spaced -> b1 intermediate
        //   A valid 10-bit clamp should not saturate a mid-range output.
        let mut src = vec![0i32; 32 * 32];
        let row_vals: [i32; 6] = [200, 300, 400, 500, 600, 700];
        for (i, v) in row_vals.iter().enumerate() {
            src[10 * 32 + 8 + i] = *v;
        }
        let mut dst = vec![0i32; 16];
        interpolate_luma(&src, 32, 32, 32, 10, 10, 2, 0, 4, 4, 10, &mut dst, 4).unwrap();
        // b1 = 200 - 5*300 + 20*400 + 20*500 - 5*600 + 700
        //    = 200 - 1500 + 8000 + 10000 - 3000 + 700 = 14400
        // b  = (14400 + 16) >> 5 = 14416 >> 5 = 450 (floor).
        assert_eq!(dst[0], 450);
    }

    // ---- Chroma full range of fractions ----

    #[test]
    fn chroma_full_offset_corner_d() {
        // xFrac=7, yFrac=7 biases almost entirely toward D.
        // A=0, B=0, C=0, D=64 -> ((1)(1)(0) + 7*1*0 + 1*7*0 + 7*7*64 + 32) >> 6
        //                     = (0 + 0 + 0 + 3136 + 32) >> 6 = 3168 >> 6 = 49.
        let mut src = vec![0i32; 8 * 8];
        src[3 * 8 + 3] = 64; // corner D when int=(2,2)
        let mut dst = vec![0i32; 16];
        interpolate_chroma(&src, 8, 8, 8, 2, 2, 7, 7, 4, 4, 8, &mut dst, 4)
            .unwrap();
        assert_eq!(dst[0], 49);
    }
}
