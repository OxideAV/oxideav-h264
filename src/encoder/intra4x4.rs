//! §8.3.1 — Intra_4x4 prediction (encoder side, all 9 modes).
//!
//! Mirrors [`crate::intra_pred`]'s 4x4 path but operates on the encoder's
//! raw `u8` reconstruction buffer with simple `(mb_x, mb_y, blk_x, blk_y)`
//! coordinates rather than the decoder's `Samples4x4` plumbing.
//!
//! All spec citations refer to ITU-T Rec. H.264 (08/2024).
//!
//! Per §8.3.1, the 9 modes per 4x4 sub-block are:
//!
//! ```text
//!   0  Intra_4x4_Vertical              (8-46)
//!   1  Intra_4x4_Horizontal            (8-47)
//!   2  Intra_4x4_DC                    (8-48)..(8-51)
//!   3  Intra_4x4_Diagonal_Down_Left    (8-52)..(8-53)
//!   4  Intra_4x4_Diagonal_Down_Right   (8-54)..(8-56)
//!   5  Intra_4x4_Vertical_Right        (8-57)..(8-60)
//!   6  Intra_4x4_Horizontal_Down       (8-61)..(8-64)
//!   7  Intra_4x4_Vertical_Left         (8-65)..(8-66)
//!   8  Intra_4x4_Horizontal_Up         (8-67)..(8-70)
//! ```
//!
//! Each mode reads from a small set of neighbour samples
//! `p[x, -1]` / `p[-1, y]` / `p[-1, -1]`. Whether a mode is *available*
//! depends on which of those samples are present (see
//! [`Intra4x4Availability`] below).
//!
//! Per §8.3.1.1 NOTE: the §6.4.11.4 neighbour map uses (xN, yN) =
//! (bx-1, by) for A and (bx, by-1) for B at the block-of-4-samples
//! granularity. The "top-right" availability of a 4x4 block depends on
//! the block's position inside the macroblock per Table 8-2's footnote
//! that some block indices have their "top-right" inside the same MB
//! (already reconstructed) while others would need MB-to-the-right's
//! samples (not yet available in raster scan).

#![allow(dead_code)]

/// All 9 Intra_4x4 prediction modes (§8.3.1.1 Table 8-2 indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intra4x4Mode {
    Vertical = 0,
    Horizontal = 1,
    Dc = 2,
    DiagonalDownLeft = 3,
    DiagonalDownRight = 4,
    VerticalRight = 5,
    HorizontalDown = 6,
    VerticalLeft = 7,
    HorizontalUp = 8,
}

impl Intra4x4Mode {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Vertical,
            1 => Self::Horizontal,
            2 => Self::Dc,
            3 => Self::DiagonalDownLeft,
            4 => Self::DiagonalDownRight,
            5 => Self::VerticalRight,
            6 => Self::HorizontalDown,
            7 => Self::VerticalLeft,
            8 => Self::HorizontalUp,
            _ => return None,
        })
    }

    pub const ALL: [Intra4x4Mode; 9] = [
        Intra4x4Mode::Vertical,
        Intra4x4Mode::Horizontal,
        Intra4x4Mode::Dc,
        Intra4x4Mode::DiagonalDownLeft,
        Intra4x4Mode::DiagonalDownRight,
        Intra4x4Mode::VerticalRight,
        Intra4x4Mode::HorizontalDown,
        Intra4x4Mode::VerticalLeft,
        Intra4x4Mode::HorizontalUp,
    ];
}

/// Neighbour-sample availability for a single 4x4 block, computed from
/// the MB / sub-block position and what's been reconstructed.
///
/// Per §8.3.1.2's "top-right substitution" rule we treat top-right as
/// available whenever the 4x4 block has *any* set of `p[4..=7, -1]`
/// samples reachable inside the local MB or in an already-reconstructed
/// MB above (raster scan order — never to the right of the current MB).
#[derive(Debug, Clone, Copy, Default)]
pub struct Intra4x4Availability {
    pub top_left: bool,
    pub top: bool,
    pub top_right: bool,
    pub left: bool,
}

/// §6.4.3 Figure 6-10 — luma 4x4 block scan: index 0..=15 → (bx, by)
/// offset of the block's top-left sample inside a macroblock (in pixels,
/// not 4-sample units). Identical to `crate::reconstruct::LUMA_4X4_XY`
/// but in `usize` so we can use it inline for buffer indexing.
pub const LUMA_4X4_XY: [(usize, usize); 16] = [
    (0, 0),
    (4, 0),
    (0, 4),
    (4, 4),
    (8, 0),
    (12, 0),
    (8, 4),
    (12, 4),
    (0, 8),
    (4, 8),
    (0, 12),
    (4, 12),
    (8, 8),
    (12, 8),
    (8, 12),
    (12, 12),
];

/// §8.3.1.2.4/8 — for each luma 4x4 block index, can the top-right
/// samples `p[4..=7, -1]` come from a reconstructed neighbour?
///
/// Per the spec, in raster scan order the top-right samples of certain
/// 4x4 blocks lie within already-encoded neighbours (top-right MB or
/// inside the same MB), while others would require samples from the
/// MB-to-the-right (not yet encoded). For those blocks the "top-right
/// substitution" rule in §8.3.1.2 fills `p[4..=7, -1]` with `p[3, -1]`.
///
/// Block indices that have a *real* top-right available (assuming top
/// neighbour is itself available):
///   inside-MB neighbours: blocks 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14
///   no real top-right (substitute from p[3,-1]): 3, 7, 11, 15
///
/// Note for block 3, 7, 11: their "top-right" block per §6.4.3 lies in
/// the same MB but has NOT been encoded yet in raster-Z order. Those
/// follow the substitution rule too. Block 5, 13: their top-right is
/// encoded already (block 4/12 of the MB above-right side, or in the
/// same MB earlier in scan); we treat block 5 as having real top-right
/// only if its row's top-right MB is available too.
///
/// Conservative encoder rule: we treat "top-right available" as true
/// whenever the top neighbour is available AND the top-right samples
/// can be read from the local MB or from above-row MBs already in the
/// recon buffer. For simplicity we restrict to:
///   * blocks in the top row of the MB (0, 1, 4, 5) → true if top
///     neighbour is available AND (block is 0/1: top neighbour MB has
///     enough samples; block 4/5: top-right MB available, OR substitute
///     using p[3,-1]). To keep the implementation simple and correct we
///     always allow the substitution path: top_right is "available" iff
///     top is, and the caller's [`gather_top_row`] handles the
///     substitution.
///
/// In practice the substitution is *always* legal per §8.3.1.2, so
/// `top_right` is identical to `top` for the encoder's purposes. We
/// keep the distinction for documentation only.
#[allow(unused)]
pub fn block_has_real_top_right(blk: usize) -> bool {
    !matches!(blk, 3 | 7 | 11 | 15 | 5 | 13)
}

/// Read the 8 top-row samples `p[0..=7, -1]` for the 4x4 block at
/// MB-relative position `(bx, by)` from the reconstruction buffer.
/// `(mb_x, mb_y)` are the MB coordinates in the picture (in MB units).
///
/// Applies §8.3.1.2's substitution rule: when `p[4..=7, -1]` would
/// require samples to the right of the picture (or to the right of the
/// current MB column — not yet encoded), the value `p[3, -1]` is
/// substituted into all four positions.
pub fn gather_top_row(
    recon: &[u8],
    pic_w: usize,
    mb_x: usize,
    mb_y: usize,
    bx: usize,
    by: usize,
    top_avail: bool,
) -> [i32; 8] {
    let mut p = [0i32; 8];
    if !top_avail {
        return p;
    }
    let abs_y = (mb_y * 16 + by).wrapping_sub(1);
    // p[0..=3, -1] always reachable when top_avail (by==0 → row above MB,
    // by>0 → samples inside same MB at (bx..bx+4, by-1) already
    // reconstructed earlier in raster-Z order).
    for (i, slot) in p[..4].iter_mut().enumerate() {
        let abs_x = mb_x * 16 + bx + i;
        *slot = recon[abs_y * pic_w + abs_x] as i32;
    }
    // p[4..=7, -1]: legal samples come from
    //   (a) the current MB above row (by==0): bx+4..bx+8 may be inside
    //       the current MB above-row (when bx+8 <= 16), or in the
    //       above-right MB (when bx+8 > 16).
    //   (b) the current MB internally (by>0): bx+4..bx+8 inside same
    //       row of MB above-row (already reconstructed).
    //
    // Per §8.3.1.2 the substitution rule fills p[4..7] with p[3,-1]
    // *only* when the original p[4..7,-1] is "not available". To keep
    // bit-exact match with the decoder's neighbour-sample gather, we
    // perform the substitution for the same blocks the decoder would:
    //   * block index 3, 7, 11, 15 → top-right inside MB but NOT yet
    //     encoded in raster-Z order → substitute.
    //   * block index 5, 13 → top-right would be in the MB to the
    //     right (not yet encoded) when at top edge → substitute.
    //   * Otherwise → real samples.
    //
    // The caller passes `(bx, by)` so we can recompute the block index.
    let blk = LUMA_4X4_XY
        .iter()
        .position(|&(x, y)| x == bx && y == by)
        .unwrap_or(0);
    let need_substitute = matches!(blk, 3 | 5 | 7 | 11 | 13 | 15);
    // Also substitute when the would-be-fetched samples are off the
    // picture's right edge (last MB column with bx+4..bx+8 spilling
    // beyond row above us).
    let abs_x_start = mb_x * 16 + bx + 4;
    let off_right = abs_x_start + 4 > pic_w;
    if need_substitute || off_right {
        let v = p[3];
        for slot in &mut p[4..8] {
            *slot = v;
        }
    } else {
        for (i, slot) in p[4..8].iter_mut().enumerate() {
            let abs_x = mb_x * 16 + bx + 4 + i;
            *slot = recon[abs_y * pic_w + abs_x] as i32;
        }
    }
    p
}

/// Read the 4 left-column samples `p[-1, 0..=3]` for the 4x4 block at
/// MB-relative position `(bx, by)` from the reconstruction buffer.
pub fn gather_left_col(
    recon: &[u8],
    pic_w: usize,
    mb_x: usize,
    mb_y: usize,
    bx: usize,
    by: usize,
    left_avail: bool,
) -> [i32; 4] {
    let mut l = [0i32; 4];
    if !left_avail {
        return l;
    }
    let abs_x = (mb_x * 16 + bx).wrapping_sub(1);
    for (j, slot) in l.iter_mut().enumerate() {
        let abs_y = mb_y * 16 + by + j;
        *slot = recon[abs_y * pic_w + abs_x] as i32;
    }
    l
}

/// Read `p[-1, -1]` for the 4x4 block.
pub fn gather_top_left(
    recon: &[u8],
    pic_w: usize,
    mb_x: usize,
    mb_y: usize,
    bx: usize,
    by: usize,
    top_left_avail: bool,
) -> i32 {
    if !top_left_avail {
        return 0;
    }
    let abs_x = (mb_x * 16 + bx).wrapping_sub(1);
    let abs_y = (mb_y * 16 + by).wrapping_sub(1);
    recon[abs_y * pic_w + abs_x] as i32
}

/// Per §6.4.11.4 + §8.3.1.1 — availability of the four neighbour
/// samples (top, left, top-left, top-right) for the 4x4 block at
/// MB-relative position `(bx, by)` inside macroblock `(mb_x, mb_y)`
/// of a single-slice raster-scan picture. `pic_w_mbs` is the picture
/// width in MBs.
///
/// Top-right availability follows the §8.3.1.2 substitution rule: even
/// when the "real" top-right samples are not present, the substitution
/// rule lets the predictor proceed by replicating `p[3, -1]`. So
/// top_right is reported as available iff `top` is available — the
/// `gather_top_row` helper handles the substitution silently.
pub fn availability(
    mb_x: usize,
    mb_y: usize,
    bx: usize,
    by: usize,
    _pic_w_mbs: usize,
) -> Intra4x4Availability {
    // Top edge of picture / top edge of MB w/ no MB above:
    //   top samples are at absolute y = mb_y*16 + by - 1.
    let top = mb_y > 0 || by > 0;
    // Left edge of picture / left edge of MB w/ no MB to the left:
    //   left samples are at absolute x = mb_x*16 + bx - 1.
    let left = mb_x > 0 || bx > 0;
    // Top-left corner is the (mb_x*16+bx-1, mb_y*16+by-1) sample. It
    // exists iff both top and left exist.
    let top_left = top && left;
    // Top-right: see doc above. Always equal to `top` per the
    // substitution rule.
    let top_right = top;
    Intra4x4Availability {
        top_left,
        top,
        top_right,
        left,
    }
}

/// Run a 4x4 predictor against the reconstruction buffer and return
/// 16 row-major predicted samples. Returns `None` if the chosen mode's
/// required neighbour samples are not available (per §8.3.1.1).
#[allow(clippy::too_many_arguments)] // mirrors the spec's parameterisation
pub fn predict_4x4(
    mode: Intra4x4Mode,
    recon: &[u8],
    pic_w: usize,
    mb_x: usize,
    mb_y: usize,
    bx: usize,
    by: usize,
    avail: Intra4x4Availability,
) -> Option<[i32; 16]> {
    // §8.3.1.1 mode availability rules:
    //   * Vertical / DiagonalDownLeft / VerticalLeft: need top.
    //   * Horizontal / HorizontalUp: need left.
    //   * DC: always available (uses defaults).
    //   * DiagonalDownRight / VerticalRight / HorizontalDown: need top
    //     AND left AND top-left.
    match mode {
        Intra4x4Mode::Vertical | Intra4x4Mode::DiagonalDownLeft | Intra4x4Mode::VerticalLeft
            if !avail.top =>
        {
            return None
        }
        Intra4x4Mode::Horizontal | Intra4x4Mode::HorizontalUp if !avail.left => return None,
        Intra4x4Mode::DiagonalDownRight
        | Intra4x4Mode::VerticalRight
        | Intra4x4Mode::HorizontalDown
            if !(avail.top && avail.left && avail.top_left) =>
        {
            return None;
        }
        _ => {}
    }
    let top = gather_top_row(recon, pic_w, mb_x, mb_y, bx, by, avail.top);
    let left = gather_left_col(recon, pic_w, mb_x, mb_y, bx, by, avail.left);
    let tl = gather_top_left(recon, pic_w, mb_x, mb_y, bx, by, avail.top_left);
    let mut out = [0i32; 16];
    match mode {
        Intra4x4Mode::Vertical => p_vertical(&top, &mut out),
        Intra4x4Mode::Horizontal => p_horizontal(&left, &mut out),
        Intra4x4Mode::Dc => p_dc(&top, &left, avail, &mut out),
        Intra4x4Mode::DiagonalDownLeft => p_ddl(&top, &mut out),
        Intra4x4Mode::DiagonalDownRight => p_ddr(&top, &left, tl, &mut out),
        Intra4x4Mode::VerticalRight => p_vr(&top, &left, tl, &mut out),
        Intra4x4Mode::HorizontalDown => p_hd(&top, &left, tl, &mut out),
        Intra4x4Mode::VerticalLeft => p_vl(&top, &mut out),
        Intra4x4Mode::HorizontalUp => p_hu(&left, &mut out),
    }
    // §8.3.1.2 final clip: predicted samples are 0..=255 (8-bit luma).
    for s in out.iter_mut() {
        *s = (*s).clamp(0, 255);
    }
    Some(out)
}

#[inline]
fn at4(x: usize, y: usize) -> usize {
    y * 4 + x
}

// §8.3.1.2.1 — Intra_4x4_Vertical (8-46): pred[x,y] = p[x,-1].
fn p_vertical(top: &[i32; 8], out: &mut [i32; 16]) {
    for y in 0..4 {
        for x in 0..4 {
            out[at4(x, y)] = top[x];
        }
    }
}

// §8.3.1.2.2 — Intra_4x4_Horizontal (8-47): pred[x,y] = p[-1,y].
fn p_horizontal(left: &[i32; 4], out: &mut [i32; 16]) {
    for y in 0..4 {
        for x in 0..4 {
            out[at4(x, y)] = left[y];
        }
    }
}

// §8.3.1.2.3 — Intra_4x4_DC (8-48..8-51).
fn p_dc(top: &[i32; 8], left: &[i32; 4], avail: Intra4x4Availability, out: &mut [i32; 16]) {
    let dc = if avail.top && avail.left {
        // (8-48)
        (top[0] + top[1] + top[2] + top[3] + left[0] + left[1] + left[2] + left[3] + 4) >> 3
    } else if avail.left {
        // (8-49)
        (left[0] + left[1] + left[2] + left[3] + 2) >> 2
    } else if avail.top {
        // (8-50)
        (top[0] + top[1] + top[2] + top[3] + 2) >> 2
    } else {
        // (8-51) bit_depth = 8 → default 128.
        128
    };
    for s in out.iter_mut() {
        *s = dc;
    }
}

// §8.3.1.2.4 — Intra_4x4_Diagonal_Down_Left (8-52..8-53).
fn p_ddl(top: &[i32; 8], out: &mut [i32; 16]) {
    let p = top;
    for y in 0..4 {
        for x in 0..4 {
            let v = if x == 3 && y == 3 {
                (p[6] + 3 * p[7] + 2) >> 2
            } else {
                (p[x + y] + 2 * p[x + y + 1] + p[x + y + 2] + 2) >> 2
            };
            out[at4(x, y)] = v;
        }
    }
}

// §8.3.1.2.5 — Intra_4x4_Diagonal_Down_Right (8-54..8-56).
fn p_ddr(top: &[i32; 8], left: &[i32; 4], tl: i32, out: &mut [i32; 16]) {
    let p_top = |x: i32| -> i32 {
        if x == -1 {
            tl
        } else {
            top[x as usize]
        }
    };
    let p_left = |y: i32| -> i32 {
        if y == -1 {
            tl
        } else {
            left[y as usize]
        }
    };
    for y in 0..4i32 {
        for x in 0..4i32 {
            let v = if x > y {
                (p_top(x - y - 2) + 2 * p_top(x - y - 1) + p_top(x - y) + 2) >> 2
            } else if x < y {
                (p_left(y - x - 2) + 2 * p_left(y - x - 1) + p_left(y - x) + 2) >> 2
            } else {
                (p_top(0) + 2 * tl + p_left(0) + 2) >> 2
            };
            out[at4(x as usize, y as usize)] = v;
        }
    }
}

// §8.3.1.2.6 — Intra_4x4_Vertical_Right (8-57..8-60).
fn p_vr(top: &[i32; 8], left: &[i32; 4], tl: i32, out: &mut [i32; 16]) {
    let p_top = |x: i32| -> i32 {
        if x == -1 {
            tl
        } else {
            top[x as usize]
        }
    };
    let p_left = |y: i32| -> i32 {
        if y == -1 {
            tl
        } else {
            left[y as usize]
        }
    };
    for y in 0..4i32 {
        for x in 0..4i32 {
            let z_vr = 2 * x - y;
            let v = match z_vr {
                0 | 2 | 4 | 6 => {
                    let xp = x - (y >> 1) - 1;
                    (p_top(xp) + p_top(xp + 1) + 1) >> 1
                }
                1 | 3 | 5 => {
                    let xp = x - (y >> 1) - 2;
                    (p_top(xp) + 2 * p_top(xp + 1) + p_top(xp + 2) + 2) >> 2
                }
                -1 => (p_left(0) + 2 * tl + p_top(0) + 2) >> 2,
                _ => (p_left(y - 1) + 2 * p_left(y - 2) + p_left(y - 3) + 2) >> 2,
            };
            out[at4(x as usize, y as usize)] = v;
        }
    }
}

// §8.3.1.2.7 — Intra_4x4_Horizontal_Down (8-61..8-64).
fn p_hd(top: &[i32; 8], left: &[i32; 4], tl: i32, out: &mut [i32; 16]) {
    let p_top = |x: i32| -> i32 {
        if x == -1 {
            tl
        } else {
            top[x as usize]
        }
    };
    let p_left = |y: i32| -> i32 {
        if y == -1 {
            tl
        } else {
            left[y as usize]
        }
    };
    for y in 0..4i32 {
        for x in 0..4i32 {
            let z_hd = 2 * y - x;
            let v = match z_hd {
                0 | 2 | 4 | 6 => {
                    let yp = y - (x >> 1) - 1;
                    (p_left(yp) + p_left(yp + 1) + 1) >> 1
                }
                1 | 3 | 5 => {
                    let yp = y - (x >> 1) - 2;
                    (p_left(yp) + 2 * p_left(yp + 1) + p_left(yp + 2) + 2) >> 2
                }
                -1 => (p_left(0) + 2 * tl + p_top(0) + 2) >> 2,
                _ => (p_top(x - 1) + 2 * p_top(x - 2) + p_top(x - 3) + 2) >> 2,
            };
            out[at4(x as usize, y as usize)] = v;
        }
    }
}

// §8.3.1.2.8 — Intra_4x4_Vertical_Left (8-65..8-66).
fn p_vl(top: &[i32; 8], out: &mut [i32; 16]) {
    let p = top;
    for y in 0..4i32 {
        for x in 0..4i32 {
            let v = if y == 0 || y == 2 {
                let xp = (x + (y >> 1)) as usize;
                (p[xp] + p[xp + 1] + 1) >> 1
            } else {
                let xp = (x + (y >> 1)) as usize;
                (p[xp] + 2 * p[xp + 1] + p[xp + 2] + 2) >> 2
            };
            out[at4(x as usize, y as usize)] = v;
        }
    }
}

// §8.3.1.2.9 — Intra_4x4_Horizontal_Up (8-67..8-70).
fn p_hu(left: &[i32; 4], out: &mut [i32; 16]) {
    let l = left;
    for y in 0..4i32 {
        for x in 0..4i32 {
            let z_hu = x + 2 * y;
            let v = match z_hu {
                0 | 2 | 4 => {
                    let yp = (y + (x >> 1)) as usize;
                    (l[yp] + l[yp + 1] + 1) >> 1
                }
                1 | 3 => {
                    let yp = (y + (x >> 1)) as usize;
                    let p2 = if yp + 2 < 4 { l[yp + 2] } else { l[3] };
                    let p1 = if yp + 1 < 4 { l[yp + 1] } else { l[3] };
                    (l[yp] + 2 * p1 + p2 + 2) >> 2
                }
                5 => (l[2] + 3 * l[3] + 2) >> 2,
                _ => l[3],
            };
            out[at4(x as usize, y as usize)] = v;
        }
    }
}

/// Sum of absolute differences between a 4x4 source patch and a
/// candidate 4x4 predictor. `src_x`, `src_y` are absolute pixel
/// coordinates of the patch's top-left in the source plane.
pub fn sad_4x4(src: &[u8], src_w: usize, src_x: usize, src_y: usize, pred: &[i32; 16]) -> u64 {
    let mut s = 0u64;
    for j in 0..4 {
        for i in 0..4 {
            let p = pred[j * 4 + i];
            let q = src[(src_y + j) * src_w + (src_x + i)] as i32;
            s += (p - q).unsigned_abs() as u64;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(w: usize, h: usize, val: u8) -> Vec<u8> {
        vec![val; w * h]
    }

    #[test]
    fn dc_no_neighbours_returns_128() {
        let recon = flat(64, 64, 200);
        let avail = Intra4x4Availability::default();
        let out = predict_4x4(Intra4x4Mode::Dc, &recon, 64, 0, 0, 0, 0, avail).unwrap();
        assert!(out.iter().all(|&v| v == 128));
    }

    #[test]
    fn vertical_unavailable_at_top_edge() {
        let recon = flat(64, 64, 200);
        let avail = availability(0, 0, 0, 0, 4);
        assert!(predict_4x4(Intra4x4Mode::Vertical, &recon, 64, 0, 0, 0, 0, avail).is_none());
        assert!(predict_4x4(Intra4x4Mode::Horizontal, &recon, 64, 0, 0, 0, 0, avail).is_none());
        assert!(predict_4x4(
            Intra4x4Mode::DiagonalDownRight,
            &recon,
            64,
            0,
            0,
            0,
            0,
            avail
        )
        .is_none());
        assert!(predict_4x4(Intra4x4Mode::Dc, &recon, 64, 0, 0, 0, 0, avail).is_some());
    }

    #[test]
    fn vertical_copies_top_row() {
        // Non-edge block: (mb_x, mb_y) = (0, 1), bx=0, by=0 → samples at
        // (0..4, 15) of recon. Set those to 10, 20, 30, 40.
        let mut recon = vec![0u8; 64 * 64];
        for x in 0..4 {
            recon[15 * 64 + x] = ((x + 1) * 10) as u8;
        }
        // Also set top-right samples (x=4..=7) so substitution doesn't kick in.
        for x in 4..8 {
            recon[15 * 64 + x] = ((x + 1) * 10) as u8;
        }
        let avail = availability(0, 1, 0, 0, 4);
        let out = predict_4x4(Intra4x4Mode::Vertical, &recon, 64, 0, 1, 0, 0, avail).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out[at4(x, y)], ((x + 1) * 10) as i32);
            }
        }
    }

    #[test]
    fn horizontal_copies_left_col() {
        // (mb_x, mb_y) = (1, 0), bx=0, by=0 → samples at (15, 0..4).
        let mut recon = vec![0u8; 64 * 64];
        for y in 0..4 {
            recon[y * 64 + 15] = ((y + 1) * 7) as u8;
        }
        let avail = availability(1, 0, 0, 0, 4);
        let out = predict_4x4(Intra4x4Mode::Horizontal, &recon, 64, 1, 0, 0, 0, avail).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out[at4(x, y)], ((y + 1) * 7) as i32);
            }
        }
    }

    #[test]
    fn sad_zero_when_pred_matches_src() {
        let mut src = vec![0u8; 64 * 64];
        for j in 0..4 {
            for i in 0..4 {
                src[j * 64 + i] = (j * 10 + i) as u8;
            }
        }
        let mut pred = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                pred[at4(i, j)] = (j * 10 + i) as i32;
            }
        }
        assert_eq!(sad_4x4(&src, 64, 0, 0, &pred), 0);
    }
}
