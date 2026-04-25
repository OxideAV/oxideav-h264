//! §8.3.3 Intra_16x16 prediction (encoder side) and §8.3.4 chroma
//! prediction. The decoder owns the canonical implementation in
//! [`crate::intra_pred`]; the encoder side is a thin re-derivation that
//! works on raw `u8` reconstruction buffers (no `Samples16x16` plumbing
//! required) so we can run mode-decision per macroblock without copying
//! into the decoder's neighbour-collection types.
//!
//! The arithmetic mirrors the decoder line-by-line; see
//! `predict_16x16_*` and `predict_chroma_*` in `intra_pred.rs` for the
//! authoritative spec reference. Spec citations refer to ITU-T Rec.
//! H.264 (08/2024).

#![allow(dead_code)]

/// All four Intra_16x16 prediction modes (§8.3.3 Table 8-4 indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum I16x16Mode {
    /// `Intra_16x16_Vertical` — copy top row downward.
    Vertical = 0,
    /// `Intra_16x16_Horizontal` — copy left column rightward.
    Horizontal = 1,
    /// `Intra_16x16_DC` — single DC value (mean of neighbours).
    Dc = 2,
    /// `Intra_16x16_Plane` — bilinear plane fit.
    Plane = 3,
}

impl I16x16Mode {
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// All four IntraChromaPred modes (§8.3.4 Table 8-5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraChromaMode {
    /// DC (Table 8-5 index 0).
    Dc = 0,
    /// Horizontal (Table 8-5 index 1).
    Horizontal = 1,
    /// Vertical (Table 8-5 index 2).
    Vertical = 2,
    /// Plane (Table 8-5 index 3).
    Plane = 3,
}

impl IntraChromaMode {
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Predict a single 16x16 luma block from the reconstruction buffer.
/// Returns the 256 predicted samples in row-major order, or `None` if
/// the mode is unavailable for this MB position (e.g. Vertical at the
/// top edge — §8.3.3 forbids picking modes whose required neighbours
/// are missing).
///
/// `mb_x`, `mb_y` are the macroblock's coordinates inside the picture
/// (in MB units). `recon_y` holds samples that have already been
/// committed (left/above MBs only — caller guarantees raster-scan).
pub fn predict_16x16(
    mode: I16x16Mode,
    recon_y: &[u8],
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
) -> Option<[i32; 256]> {
    let (top, left, tl) = availability(mb_x, mb_y);
    match mode {
        I16x16Mode::Vertical if !top => None,
        I16x16Mode::Horizontal if !left => None,
        I16x16Mode::Plane if !(top && left && tl) => None,
        I16x16Mode::Vertical => Some(predict_16x16_vertical(recon_y, width, height, mb_x, mb_y)),
        I16x16Mode::Horizontal => {
            Some(predict_16x16_horizontal(recon_y, width, height, mb_x, mb_y))
        }
        I16x16Mode::Dc => Some(predict_16x16_dc(recon_y, width, height, mb_x, mb_y)),
        I16x16Mode::Plane => Some(predict_16x16_plane(recon_y, width, height, mb_x, mb_y)),
    }
}

/// Availability mask derived from MB coordinates and a single-slice
/// picture (raster-scan, no FMO). For Intra_16x16:
///  * top available iff `mb_y > 0`
///  * left available iff `mb_x > 0`
///  * top-left available iff both
fn availability(mb_x: usize, mb_y: usize) -> (bool, bool, bool) {
    (mb_y > 0, mb_x > 0, mb_y > 0 && mb_x > 0)
}

/// §8.3.3.1 — Intra_16x16_Vertical. (8-116) `pred[x,y] = p[x,-1]`.
/// Caller must check `top_available`; this routine returns a mid-grey
/// fill when unavailable so the SAD/SATD path doesn't pick it.
fn predict_16x16_vertical(
    recon_y: &[u8],
    width: usize,
    _height: usize,
    mb_x: usize,
    mb_y: usize,
) -> [i32; 256] {
    let (top, _left, _tl) = availability(mb_x, mb_y);
    let mut out = [128i32; 256];
    if !top {
        return out;
    }
    let row_above = (mb_y * 16 - 1) * width;
    for x in 0..16 {
        let s = recon_y[row_above + mb_x * 16 + x] as i32;
        for y in 0..16 {
            out[y * 16 + x] = s;
        }
    }
    out
}

/// §8.3.3.2 — Intra_16x16_Horizontal. (8-117) `pred[x,y] = p[-1,y]`.
fn predict_16x16_horizontal(
    recon_y: &[u8],
    width: usize,
    _height: usize,
    mb_x: usize,
    mb_y: usize,
) -> [i32; 256] {
    let (_top, left, _tl) = availability(mb_x, mb_y);
    let mut out = [128i32; 256];
    if !left {
        return out;
    }
    for y in 0..16 {
        let s = recon_y[(mb_y * 16 + y) * width + mb_x * 16 - 1] as i32;
        for x in 0..16 {
            out[y * 16 + x] = s;
        }
    }
    out
}

/// §8.3.3.3 — Intra_16x16_DC, equations (8-118)..(8-121).
fn predict_16x16_dc(
    recon_y: &[u8],
    width: usize,
    _height: usize,
    mb_x: usize,
    mb_y: usize,
) -> [i32; 256] {
    let (top, left, _tl) = availability(mb_x, mb_y);
    let dc = if top && left {
        let mut s_top = 0i32;
        let mut s_left = 0i32;
        let row_above = (mb_y * 16 - 1) * width;
        for i in 0..16 {
            s_top += recon_y[row_above + mb_x * 16 + i] as i32;
        }
        for j in 0..16 {
            s_left += recon_y[(mb_y * 16 + j) * width + mb_x * 16 - 1] as i32;
        }
        (s_top + s_left + 16) >> 5
    } else if left {
        let mut s = 0i32;
        for j in 0..16 {
            s += recon_y[(mb_y * 16 + j) * width + mb_x * 16 - 1] as i32;
        }
        (s + 8) >> 4
    } else if top {
        let mut s = 0i32;
        let row_above = (mb_y * 16 - 1) * width;
        for i in 0..16 {
            s += recon_y[row_above + mb_x * 16 + i] as i32;
        }
        (s + 8) >> 4
    } else {
        128
    };
    [dc; 256]
}

/// §8.3.3.4 — Intra_16x16_Plane, equations (8-122)..(8-127). Falls
/// back to DC when neighbour samples missing.
fn predict_16x16_plane(
    recon_y: &[u8],
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
) -> [i32; 256] {
    let (top, left, tl) = availability(mb_x, mb_y);
    if !(top && left && tl) {
        return predict_16x16_dc(recon_y, width, height, mb_x, mb_y);
    }
    // Sample fetchers for the spec's `p[x,-1]` / `p[-1,y]` / `p[-1,-1]`.
    let row_above = (mb_y * 16 - 1) * width;
    let p_top = |x: i32| -> i32 {
        if x == -1 {
            recon_y[row_above + mb_x * 16 - 1] as i32
        } else {
            recon_y[row_above + mb_x * 16 + x as usize] as i32
        }
    };
    let p_left = |y: i32| -> i32 {
        if y == -1 {
            recon_y[row_above + mb_x * 16 - 1] as i32
        } else {
            recon_y[(mb_y * 16 + y as usize) * width + mb_x * 16 - 1] as i32
        }
    };
    // (8-126) H = sum_{x'=0..7} (x'+1) * (p[8+x',-1] - p[6-x',-1])
    let mut h = 0i32;
    for xp in 0..8i32 {
        h += (xp + 1) * (p_top(8 + xp) - p_top(6 - xp));
    }
    // (8-127) V = sum_{y'=0..7} (y'+1) * (p[-1,8+y'] - p[-1,6-y'])
    let mut v = 0i32;
    for yp in 0..8i32 {
        v += (yp + 1) * (p_left(8 + yp) - p_left(6 - yp));
    }
    // (8-123) a = 16 * (p[-1,15] + p[15,-1])
    let a = 16 * (p_left(15) + p_top(15));
    // (8-124) b = (5*H + 32) >> 6 ; (8-125) c = (5*V + 32) >> 6
    let b = (5 * h + 32) >> 6;
    let c = (5 * v + 32) >> 6;
    let mut out = [0i32; 256];
    for y in 0..16i32 {
        for x in 0..16i32 {
            let raw = (a + b * (x - 7) + c * (y - 7) + 16) >> 5;
            out[(y * 16 + x) as usize] = raw.clamp(0, 255);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// §8.3.4 — chroma intra prediction (4:2:0 only — 8x8 plane).
// ---------------------------------------------------------------------------

/// Predict a single 8x8 chroma block. Returns 64 predicted samples
/// row-major, or `None` when the mode is unavailable at this MB
/// position.
pub fn predict_chroma_8x8(
    mode: IntraChromaMode,
    recon: &[u8],
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
) -> Option<[i32; 64]> {
    let (top, left, tl) = availability(mb_x, mb_y);
    match mode {
        IntraChromaMode::Vertical if !top => None,
        IntraChromaMode::Horizontal if !left => None,
        IntraChromaMode::Plane if !(top && left && tl) => None,
        IntraChromaMode::Dc => Some(predict_chroma_dc_8x8(recon, width, height, mb_x, mb_y)),
        IntraChromaMode::Horizontal => Some(predict_chroma_h_8x8(recon, width, height, mb_x, mb_y)),
        IntraChromaMode::Vertical => Some(predict_chroma_v_8x8(recon, width, height, mb_x, mb_y)),
        IntraChromaMode::Plane => Some(predict_chroma_plane_8x8(recon, width, height, mb_x, mb_y)),
    }
}

/// §8.3.4.1 — Intra_Chroma_DC (4:2:0). The spec actually splits the
/// 8x8 block into four 4x4 sub-blocks and derives a DC per sub-block
/// using rules (8-132)..(8-141). We reproduce the same per-sub-block
/// derivation so the predictor matches the decoder's reconstruction
/// down to the last bit.
fn predict_chroma_dc_8x8(
    recon: &[u8],
    width: usize,
    _height: usize,
    mb_x: usize,
    mb_y: usize,
) -> [i32; 64] {
    let (top, left, _tl) = availability(mb_x, mb_y);
    let default = 128i32;
    let row_above = if top { (mb_y * 8 - 1) * width } else { 0 };
    let sum_top = |x_o: usize| -> i32 {
        let mut s = 0i32;
        for i in 0..4 {
            s += recon[row_above + mb_x * 8 + x_o + i] as i32;
        }
        s
    };
    let sum_left = |y_o: usize| -> i32 {
        let mut s = 0i32;
        for j in 0..4 {
            s += recon[(mb_y * 8 + y_o + j) * width + mb_x * 8 - 1] as i32;
        }
        s
    };
    // Compute DC per 4x4 sub-block at (xO, yO).
    let dc_sub = |x_o: usize, y_o: usize| -> i32 {
        if x_o == 0 && y_o == 0 {
            // (8-132)..(8-135): both → average; one → /4; neither → default.
            if top && left {
                (sum_top(0) + sum_left(0) + 4) >> 3
            } else if top {
                (sum_top(0) + 2) >> 2
            } else if left {
                (sum_left(0) + 2) >> 2
            } else {
                default
            }
        } else if x_o > 0 && y_o == 0 {
            // Top row (xO=4 in 4:2:0): prefer top (8-136..8-138).
            if top {
                (sum_top(x_o) + 2) >> 2
            } else if left {
                (sum_left(0) + 2) >> 2
            } else {
                default
            }
        } else if x_o == 0 && y_o > 0 {
            // Left column (yO=4 in 4:2:0): prefer left (8-139..8-141).
            if left {
                (sum_left(y_o) + 2) >> 2
            } else if top {
                (sum_top(0) + 2) >> 2
            } else {
                default
            }
        } else {
            // Inner sub-block (xO=4, yO=4): mix top + left when both
            // available (8-142..8-145).
            if top && left {
                (sum_top(x_o) + sum_left(y_o) + 4) >> 3
            } else if top {
                (sum_top(x_o) + 2) >> 2
            } else if left {
                (sum_left(y_o) + 2) >> 2
            } else {
                default
            }
        }
    };
    let mut out = [0i32; 64];
    for y_o in [0, 4] {
        for x_o in [0, 4] {
            let dc = dc_sub(x_o, y_o);
            for yy in 0..4 {
                for xx in 0..4 {
                    out[(y_o + yy) * 8 + (x_o + xx)] = dc;
                }
            }
        }
    }
    out
}

/// §8.3.4.2 — Intra_Chroma_Horizontal (8-128). `pred[x,y] = p[-1,y]`.
fn predict_chroma_h_8x8(
    recon: &[u8],
    width: usize,
    _height: usize,
    mb_x: usize,
    mb_y: usize,
) -> [i32; 64] {
    let (_top, left, _tl) = availability(mb_x, mb_y);
    let mut out = [128i32; 64];
    if !left {
        return out;
    }
    for y in 0..8 {
        let s = recon[(mb_y * 8 + y) * width + mb_x * 8 - 1] as i32;
        for x in 0..8 {
            out[y * 8 + x] = s;
        }
    }
    out
}

/// §8.3.4.3 — Intra_Chroma_Vertical (8-129). `pred[x,y] = p[x,-1]`.
fn predict_chroma_v_8x8(
    recon: &[u8],
    width: usize,
    _height: usize,
    mb_x: usize,
    mb_y: usize,
) -> [i32; 64] {
    let (top, _left, _tl) = availability(mb_x, mb_y);
    let mut out = [128i32; 64];
    if !top {
        return out;
    }
    let row_above = (mb_y * 8 - 1) * width;
    for x in 0..8 {
        let s = recon[row_above + mb_x * 8 + x] as i32;
        for y in 0..8 {
            out[y * 8 + x] = s;
        }
    }
    out
}

/// §8.3.4.4 — Intra_Chroma_Plane (4:2:0). Equations (8-150)..(8-156).
fn predict_chroma_plane_8x8(
    recon: &[u8],
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
) -> [i32; 64] {
    let (top, left, tl) = availability(mb_x, mb_y);
    if !(top && left && tl) {
        return predict_chroma_dc_8x8(recon, width, height, mb_x, mb_y);
    }
    let row_above = (mb_y * 8 - 1) * width;
    let p_top = |x: i32| -> i32 {
        if x == -1 {
            recon[row_above + mb_x * 8 - 1] as i32
        } else {
            recon[row_above + mb_x * 8 + x as usize] as i32
        }
    };
    let p_left = |y: i32| -> i32 {
        if y == -1 {
            recon[row_above + mb_x * 8 - 1] as i32
        } else {
            recon[(mb_y * 8 + y as usize) * width + mb_x * 8 - 1] as i32
        }
    };
    // (8-153) H = sum_{x'=0..3} (x'+1) * (p[4+x',-1] - p[2-x',-1])
    let mut h = 0i32;
    for xp in 0..4i32 {
        h += (xp + 1) * (p_top(4 + xp) - p_top(2 - xp));
    }
    // (8-154) V = sum_{y'=0..3} (y'+1) * (p[-1,4+y'] - p[-1,2-y'])
    let mut v = 0i32;
    for yp in 0..4i32 {
        v += (yp + 1) * (p_left(4 + yp) - p_left(2 - yp));
    }
    // (8-150) a = 16 * (p[-1,7] + p[7,-1])
    let a = 16 * (p_left(7) + p_top(7));
    // (8-151) b = (34*H + 32) >> 6 ; (8-152) c = (34*V + 32) >> 6
    let b = (34 * h + 32) >> 6;
    let c = (34 * v + 32) >> 6;
    let mut out = [0i32; 64];
    for y in 0..8i32 {
        for x in 0..8i32 {
            let raw = (a + b * (x - 3) + c * (y - 3) + 16) >> 5;
            out[(y * 8 + x) as usize] = raw.clamp(0, 255);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Mode-decision metric.
// ---------------------------------------------------------------------------

/// Sum of absolute differences between a 16x16 source block and a
/// candidate 16x16 predictor. Used by the encoder's mode selector.
pub fn sad_16x16(
    src: &[u8],
    src_stride: usize,
    src_x: usize,
    src_y: usize,
    pred: &[i32; 256],
) -> u64 {
    let mut s = 0u64;
    for j in 0..16 {
        for i in 0..16 {
            let p = pred[j * 16 + i];
            let q = src[(src_y + j) * src_stride + (src_x + i)] as i32;
            s += (p - q).unsigned_abs() as u64;
        }
    }
    s
}

/// SAD for 8x8 chroma blocks. Same idea as `sad_16x16`.
pub fn sad_8x8(src: &[u8], src_stride: usize, src_x: usize, src_y: usize, pred: &[i32; 64]) -> u64 {
    let mut s = 0u64;
    for j in 0..8 {
        for i in 0..8 {
            let p = pred[j * 8 + i];
            let q = src[(src_y + j) * src_stride + (src_x + i)] as i32;
            s += (p - q).unsigned_abs() as u64;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_recon(width: usize, height: usize, val: u8) -> Vec<u8> {
        vec![val; width * height]
    }

    #[test]
    fn dc_with_no_neighbours_returns_128() {
        let recon = flat_recon(64, 64, 200);
        let pred = predict_16x16_dc(&recon, 64, 64, 0, 0);
        assert!(pred.iter().all(|&p| p == 128));
    }

    #[test]
    fn vertical_unavailable_at_top_edge() {
        let recon = flat_recon(64, 64, 200);
        assert!(predict_16x16(I16x16Mode::Vertical, &recon, 64, 64, 0, 0).is_none());
        assert!(predict_16x16(I16x16Mode::Horizontal, &recon, 64, 64, 0, 0).is_none());
        assert!(predict_16x16(I16x16Mode::Plane, &recon, 64, 64, 0, 0).is_none());
        assert!(predict_16x16(I16x16Mode::Dc, &recon, 64, 64, 0, 0).is_some());
    }

    #[test]
    fn vertical_copies_top_row() {
        // Place a known top row at y = 15 (which is the row above MB
        // (0, 1) in the 64x64 picture).
        let mut recon = vec![0u8; 64 * 64];
        for x in 0..64 {
            recon[15 * 64 + x] = (x * 4) as u8;
        }
        let pred = predict_16x16_vertical(&recon, 64, 64, 0, 1);
        // Every column should equal the top sample; sample x=5 → 20.
        for y in 0..16 {
            assert_eq!(pred[y * 16 + 5], 20);
        }
    }

    #[test]
    fn horizontal_copies_left_column() {
        let mut recon = vec![0u8; 64 * 64];
        for y in 0..64 {
            recon[y * 64 + 15] = (y * 3) as u8;
        }
        let pred = predict_16x16_horizontal(&recon, 64, 64, 1, 0);
        // Every row should equal the left sample.
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(pred[y * 16 + x], ((y * 3) & 0xff) as i32);
            }
        }
    }

    #[test]
    fn plane_constant_neighbours_returns_constant() {
        // All neighbour samples = 100 → plane reduces to DC = 100.
        let recon = flat_recon(64, 64, 100);
        let pred = predict_16x16_plane(&recon, 64, 64, 1, 1);
        // Plane: a = 16*(100+100) = 3200, b = 0, c = 0, pred = (3200+16)>>5 = 100.
        assert!(pred.iter().all(|&p| p == 100), "got {:?}", &pred[..8]);
    }

    #[test]
    fn chroma_dc_with_no_neighbours_returns_128() {
        let recon = flat_recon(32, 32, 200);
        let pred = predict_chroma_dc_8x8(&recon, 32, 32, 0, 0);
        assert!(pred.iter().all(|&p| p == 128));
    }

    #[test]
    fn chroma_h_copies_left() {
        let mut recon = vec![0u8; 32 * 32];
        for y in 0..32 {
            recon[y * 32 + 7] = (y * 2) as u8;
        }
        let pred = predict_chroma_h_8x8(&recon, 32, 32, 1, 0);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(pred[y * 8 + x], ((y * 2) & 0xff) as i32);
            }
        }
    }

    #[test]
    fn sad_zero_when_pred_matches_src() {
        let src = vec![100u8; 64 * 64];
        let pred = [100i32; 256];
        assert_eq!(sad_16x16(&src, 64, 0, 0, &pred), 0);
    }
}
