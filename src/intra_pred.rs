//! §8.3 — Intra prediction.
//!
//! All Intra_NxN and Intra_16x16 / chroma modes per ITU-T Rec. H.264
//! (08/2024). Pure pixel-level transforms — no bitstream, no context.
//!
//! Sample reference layout mirrors the spec's p[x, y]:
//!   p[-1, -1]       = top_left (scalar)
//!   p[0..=3,  -1]   = top
//!   p[4..=7,  -1]   = top_right   (for 4x4; for 8x8 p[8..=15, -1])
//!   p[-1,   0..=3]  = left        (for 4x4; for 8x8 0..=7)
//!
//! All prediction outputs are clipped to the bit-depth range
//! `0..=(1<<bit_depth)-1` (Clip1_Y / Clip1_C in the spec).

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// §5 — Clip1(x) = Clip3(0, (1<<BitDepth)-1, x).
#[inline]
fn clip1(x: i32, bit_depth: u32) -> i32 {
    let max = (1i32 << bit_depth) - 1;
    if x < 0 {
        0
    } else if x > max {
        max
    } else {
        x
    }
}

/// DC fallback when no neighbours are available.
#[inline]
fn dc_default(bit_depth: u32) -> i32 {
    1 << (bit_depth - 1)
}

// ---------------------------------------------------------------------------
// Mode enums
// ---------------------------------------------------------------------------

/// §8.3.1.1 Table 8-2.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Intra4x4Mode {
    Vertical,
    Horizontal,
    Dc,
    DiagonalDownLeft,
    DiagonalDownRight,
    VerticalRight,
    HorizontalDown,
    VerticalLeft,
    HorizontalUp,
}

impl Intra4x4Mode {
    pub fn from_index(idx: u8) -> Option<Self> {
        Some(match idx {
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

    pub fn as_index(self) -> u8 {
        match self {
            Self::Vertical => 0,
            Self::Horizontal => 1,
            Self::Dc => 2,
            Self::DiagonalDownLeft => 3,
            Self::DiagonalDownRight => 4,
            Self::VerticalRight => 5,
            Self::HorizontalDown => 6,
            Self::VerticalLeft => 7,
            Self::HorizontalUp => 8,
        }
    }
}

/// §8.3.2.1 Table 8-3.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Intra8x8Mode {
    Vertical,
    Horizontal,
    Dc,
    DiagonalDownLeft,
    DiagonalDownRight,
    VerticalRight,
    HorizontalDown,
    VerticalLeft,
    HorizontalUp,
}

impl Intra8x8Mode {
    pub fn from_index(idx: u8) -> Option<Self> {
        Some(match idx {
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

    pub fn as_index(self) -> u8 {
        match self {
            Self::Vertical => 0,
            Self::Horizontal => 1,
            Self::Dc => 2,
            Self::DiagonalDownLeft => 3,
            Self::DiagonalDownRight => 4,
            Self::VerticalRight => 5,
            Self::HorizontalDown => 6,
            Self::VerticalLeft => 7,
            Self::HorizontalUp => 8,
        }
    }
}

/// §8.3.3 Table 8-4.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Intra16x16Mode {
    Vertical,
    Horizontal,
    Dc,
    Plane,
}

impl Intra16x16Mode {
    pub fn from_index(idx: u8) -> Option<Self> {
        Some(match idx {
            0 => Self::Vertical,
            1 => Self::Horizontal,
            2 => Self::Dc,
            3 => Self::Plane,
            _ => return None,
        })
    }

    pub fn as_index(self) -> u8 {
        match self {
            Self::Vertical => 0,
            Self::Horizontal => 1,
            Self::Dc => 2,
            Self::Plane => 3,
        }
    }
}

/// §8.3.4 Table 8-5.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum IntraChromaMode {
    Dc,
    Horizontal,
    Vertical,
    Plane,
}

impl IntraChromaMode {
    pub fn from_index(idx: u8) -> Option<Self> {
        Some(match idx {
            0 => Self::Dc,
            1 => Self::Horizontal,
            2 => Self::Vertical,
            3 => Self::Plane,
            _ => return None,
        })
    }

    pub fn as_index(self) -> u8 {
        match self {
            Self::Dc => 0,
            Self::Horizontal => 1,
            Self::Vertical => 2,
            Self::Plane => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Neighbour availability and sample containers
// ---------------------------------------------------------------------------

/// Availability flags for neighbour samples.
/// Per §8.3.1.1 — some Intra_4x4 modes (Diagonal_Down_Left,
/// Vertical_Left) read p[4..=7, -1]. When those samples are not
/// available but p[0..=3, -1] is, the spec substitutes p[3, -1]
/// into p[4..=7, -1]. That substitution happens in `resolve_top_right`.
#[derive(Debug, Copy, Clone, Default)]
pub struct Neighbour4x4Availability {
    pub top_left: bool,
    pub top: bool,
    pub top_right: bool,
    pub left: bool,
}

/// Reference samples for a 4x4 predictor, per §8.3.1.1.
/// Layout:
///   p[-1, -1]   = top_left
///   p[0..=3, -1] = top
///   p[4..=7, -1] = top_right
///   p[-1, 0..=3] = left
#[derive(Debug, Copy, Clone)]
pub struct Samples4x4 {
    pub top_left: i32,
    pub top: [i32; 4],
    pub top_right: [i32; 4],
    pub left: [i32; 4],
    pub availability: Neighbour4x4Availability,
}

/// §8.3.2.1 — reference samples for an 8x8 block.
/// Layout:
///   p[-1, -1]   = top_left
///   p[0..=7, -1] = top
///   p[8..=15, -1] = top_right
///   p[-1, 0..=7] = left
#[derive(Debug, Copy, Clone)]
pub struct Samples8x8 {
    pub top_left: i32,
    pub top: [i32; 8],
    pub top_right: [i32; 8],
    pub left: [i32; 8],
    pub availability: Neighbour4x4Availability,
}

/// §8.3.3 — reference samples for a 16x16 block.
#[derive(Debug, Copy, Clone)]
pub struct Samples16x16 {
    pub top_left: i32,
    pub top: [i32; 16],
    pub left: [i32; 16],
    pub availability: Neighbour4x4Availability,
}

/// §8.3.4 — ChromaArrayType gates the chroma block size.
///   ChromaArrayType == 1 (4:2:0): 8x8
///   ChromaArrayType == 2 (4:2:2): 8x16
/// ChromaArrayType == 3 (4:4:4) is treated as luma (use the luma
/// predict_* helpers instead).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ChromaArrayType {
    Yuv420,
    Yuv422,
}

impl ChromaArrayType {
    /// MbWidthC: chroma block width in samples.
    pub fn width(self) -> usize {
        8
    }
    /// MbHeightC: chroma block height in samples.
    pub fn height(self) -> usize {
        match self {
            Self::Yuv420 => 8,
            Self::Yuv422 => 16,
        }
    }
}

/// Reference samples for an 8x8 or 8x16 chroma block per §8.3.4.
#[derive(Debug, Clone)]
pub struct SamplesChroma {
    pub top_left: i32,
    pub top: Vec<i32>,  // length MbWidthC (== 8)
    pub left: Vec<i32>, // length MbHeightC (8 or 16)
    pub availability: Neighbour4x4Availability,
}

// ---------------------------------------------------------------------------
// §8.3.1 Intra_4x4 prediction
// ---------------------------------------------------------------------------

/// Build the full p[0..=7, -1] array applying §8.3.1.2's substitution
/// rule: "When samples p[x, -1], with x = 4..7, are marked as 'not
/// available for Intra_4x4 prediction,' and the sample p[3, -1] is
/// marked as 'available for Intra_4x4 prediction,' the sample value
/// of p[3, -1] is substituted for sample values p[x, -1], with x = 4..7".
fn top_row_4x4(s: &Samples4x4) -> [i32; 8] {
    let mut p = [0i32; 8];
    p[0..4].copy_from_slice(&s.top);
    if s.availability.top_right {
        p[4..8].copy_from_slice(&s.top_right);
    } else if s.availability.top {
        // §8.3.1.2 substitution rule.
        for i in 4..8 {
            p[i] = s.top[3];
        }
    }
    p
}

/// Predict a 4x4 block. Output is row-major 4x4 (16 samples).
pub fn predict_4x4(mode: Intra4x4Mode, samples: &Samples4x4, bit_depth: u32, out: &mut [i32; 16]) {
    match mode {
        Intra4x4Mode::Vertical => predict_4x4_vertical(samples, out),
        Intra4x4Mode::Horizontal => predict_4x4_horizontal(samples, out),
        Intra4x4Mode::Dc => predict_4x4_dc(samples, bit_depth, out),
        Intra4x4Mode::DiagonalDownLeft => predict_4x4_ddl(samples, out),
        Intra4x4Mode::DiagonalDownRight => predict_4x4_ddr(samples, out),
        Intra4x4Mode::VerticalRight => predict_4x4_vr(samples, out),
        Intra4x4Mode::HorizontalDown => predict_4x4_hd(samples, out),
        Intra4x4Mode::VerticalLeft => predict_4x4_vl(samples, out),
        Intra4x4Mode::HorizontalUp => predict_4x4_hu(samples, out),
    }
}

#[inline]
fn at4(x: usize, y: usize) -> usize {
    y * 4 + x
}

// §8.3.1.2.1 — Intra_4x4_Vertical.
// pred4x4[x, y] = p[x, -1]
fn predict_4x4_vertical(s: &Samples4x4, out: &mut [i32; 16]) {
    for y in 0..4 {
        for x in 0..4 {
            out[at4(x, y)] = s.top[x];
        }
    }
}

// §8.3.1.2.2 — Intra_4x4_Horizontal.
// pred4x4[x, y] = p[-1, y]
fn predict_4x4_horizontal(s: &Samples4x4, out: &mut [i32; 16]) {
    for y in 0..4 {
        for x in 0..4 {
            out[at4(x, y)] = s.left[y];
        }
    }
}

// §8.3.1.2.3 — Intra_4x4_DC.
fn predict_4x4_dc(s: &Samples4x4, bit_depth: u32, out: &mut [i32; 16]) {
    let dc = if s.availability.top && s.availability.left {
        // (8-48)
        (s.top[0]
            + s.top[1]
            + s.top[2]
            + s.top[3]
            + s.left[0]
            + s.left[1]
            + s.left[2]
            + s.left[3]
            + 4)
            >> 3
    } else if s.availability.left {
        // (8-49) only left available
        (s.left[0] + s.left[1] + s.left[2] + s.left[3] + 2) >> 2
    } else if s.availability.top {
        // (8-50) only top available
        (s.top[0] + s.top[1] + s.top[2] + s.top[3] + 2) >> 2
    } else {
        // (8-51) neither available
        dc_default(bit_depth)
    };
    for sample in out.iter_mut() {
        *sample = dc;
    }
}

// §8.3.1.2.4 — Intra_4x4_Diagonal_Down_Left.
fn predict_4x4_ddl(s: &Samples4x4, out: &mut [i32; 16]) {
    let p = top_row_4x4(s);
    for y in 0..4 {
        for x in 0..4 {
            let v = if x == 3 && y == 3 {
                // (8-52) pred4x4[3,3] = (p[6,-1] + 3*p[7,-1] + 2) >> 2
                (p[6] + 3 * p[7] + 2) >> 2
            } else {
                // (8-53) (p[x+y,-1] + 2*p[x+y+1,-1] + p[x+y+2,-1] + 2) >> 2
                (p[x + y] + 2 * p[x + y + 1] + p[x + y + 2] + 2) >> 2
            };
            out[at4(x, y)] = v;
        }
    }
}

// §8.3.1.2.5 — Intra_4x4_Diagonal_Down_Right.
fn predict_4x4_ddr(s: &Samples4x4, out: &mut [i32; 16]) {
    // Reference samples: top_left plus p[0..=3,-1] and p[-1,0..=3].
    // Use helpers to read p[x,-1] for x in -1..=3 and p[-1,y] for y in -1..=3.
    let tl = s.top_left;
    let t = &s.top;
    let l = &s.left;
    // Helper to fetch p[x,-1] for x in -1..=3.
    let p_top = |x: i32| -> i32 {
        if x == -1 {
            tl
        } else {
            t[x as usize]
        }
    };
    // Helper to fetch p[-1,y] for y in -1..=3.
    let p_left = |y: i32| -> i32 {
        if y == -1 {
            tl
        } else {
            l[y as usize]
        }
    };
    for y in 0..4i32 {
        for x in 0..4i32 {
            let v = if x > y {
                // (8-54) (p[x-y-2,-1] + 2*p[x-y-1,-1] + p[x-y,-1] + 2) >> 2
                (p_top(x - y - 2) + 2 * p_top(x - y - 1) + p_top(x - y) + 2) >> 2
            } else if x < y {
                // (8-55) (p[-1,y-x-2] + 2*p[-1,y-x-1] + p[-1,y-x] + 2) >> 2
                (p_left(y - x - 2) + 2 * p_left(y - x - 1) + p_left(y - x) + 2) >> 2
            } else {
                // x == y: (8-56) (p[0,-1] + 2*p[-1,-1] + p[-1,0] + 2) >> 2
                (p_top(0) + 2 * tl + p_left(0) + 2) >> 2
            };
            out[at4(x as usize, y as usize)] = v;
        }
    }
}

// §8.3.1.2.6 — Intra_4x4_Vertical_Right.
fn predict_4x4_vr(s: &Samples4x4, out: &mut [i32; 16]) {
    let tl = s.top_left;
    let t = &s.top;
    let l = &s.left;
    let p_top = |x: i32| -> i32 {
        if x == -1 {
            tl
        } else {
            t[x as usize]
        }
    };
    let p_left = |y: i32| -> i32 {
        if y == -1 {
            tl
        } else {
            l[y as usize]
        }
    };
    for y in 0..4i32 {
        for x in 0..4i32 {
            let z_vr = 2 * x - y;
            let v = match z_vr {
                0 | 2 | 4 | 6 => {
                    // (8-57) (p[x-(y>>1)-1,-1] + p[x-(y>>1),-1] + 1) >> 1
                    let xp = x - (y >> 1) - 1;
                    (p_top(xp) + p_top(xp + 1) + 1) >> 1
                }
                1 | 3 | 5 => {
                    // (8-58) (p[x-(y>>1)-2,-1] + 2*p[x-(y>>1)-1,-1] + p[x-(y>>1),-1] + 2) >> 2
                    let xp = x - (y >> 1) - 2;
                    (p_top(xp) + 2 * p_top(xp + 1) + p_top(xp + 2) + 2) >> 2
                }
                -1 => {
                    // (8-59) (p[-1,0] + 2*p[-1,-1] + p[0,-1] + 2) >> 2
                    (p_left(0) + 2 * tl + p_top(0) + 2) >> 2
                }
                _ => {
                    // z_vr == -2 or -3: (8-60) (p[-1,y-1] + 2*p[-1,y-2] + p[-1,y-3] + 2) >> 2
                    (p_left(y - 1) + 2 * p_left(y - 2) + p_left(y - 3) + 2) >> 2
                }
            };
            out[at4(x as usize, y as usize)] = v;
        }
    }
}

// §8.3.1.2.7 — Intra_4x4_Horizontal_Down.
fn predict_4x4_hd(s: &Samples4x4, out: &mut [i32; 16]) {
    let tl = s.top_left;
    let t = &s.top;
    let l = &s.left;
    let p_top = |x: i32| -> i32 {
        if x == -1 {
            tl
        } else {
            t[x as usize]
        }
    };
    let p_left = |y: i32| -> i32 {
        if y == -1 {
            tl
        } else {
            l[y as usize]
        }
    };
    for y in 0..4i32 {
        for x in 0..4i32 {
            let z_hd = 2 * y - x;
            let v = match z_hd {
                0 | 2 | 4 | 6 => {
                    // (8-61) (p[-1,y-(x>>1)-1] + p[-1,y-(x>>1)] + 1) >> 1
                    let yp = y - (x >> 1) - 1;
                    (p_left(yp) + p_left(yp + 1) + 1) >> 1
                }
                1 | 3 | 5 => {
                    // (8-62) (p[-1,y-(x>>1)-2] + 2*p[-1,y-(x>>1)-1] + p[-1,y-(x>>1)] + 2) >> 2
                    let yp = y - (x >> 1) - 2;
                    (p_left(yp) + 2 * p_left(yp + 1) + p_left(yp + 2) + 2) >> 2
                }
                -1 => {
                    // (8-63) (p[-1,0] + 2*p[-1,-1] + p[0,-1] + 2) >> 2
                    (p_left(0) + 2 * tl + p_top(0) + 2) >> 2
                }
                _ => {
                    // z_hd == -2 or -3: (8-64) (p[x-1,-1] + 2*p[x-2,-1] + p[x-3,-1] + 2) >> 2
                    (p_top(x - 1) + 2 * p_top(x - 2) + p_top(x - 3) + 2) >> 2
                }
            };
            out[at4(x as usize, y as usize)] = v;
        }
    }
}

// §8.3.1.2.8 — Intra_4x4_Vertical_Left.
fn predict_4x4_vl(s: &Samples4x4, out: &mut [i32; 16]) {
    let p = top_row_4x4(s);
    for y in 0..4i32 {
        for x in 0..4i32 {
            let v = if y == 0 || y == 2 {
                // (8-65) (p[x+(y>>1),-1] + p[x+(y>>1)+1,-1] + 1) >> 1
                let xp = (x + (y >> 1)) as usize;
                (p[xp] + p[xp + 1] + 1) >> 1
            } else {
                // y == 1 or 3: (8-66) (p[x+(y>>1),-1] + 2*p[x+(y>>1)+1,-1] + p[x+(y>>1)+2,-1] + 2) >> 2
                let xp = (x + (y >> 1)) as usize;
                (p[xp] + 2 * p[xp + 1] + p[xp + 2] + 2) >> 2
            };
            out[at4(x as usize, y as usize)] = v;
        }
    }
}

// §8.3.1.2.9 — Intra_4x4_Horizontal_Up.
fn predict_4x4_hu(s: &Samples4x4, out: &mut [i32; 16]) {
    let l = &s.left;
    for y in 0..4i32 {
        for x in 0..4i32 {
            let z_hu = x + 2 * y;
            let v = match z_hu {
                0 | 2 | 4 => {
                    // (8-67) (p[-1,y+(x>>1)] + p[-1,y+(x>>1)+1] + 1) >> 1
                    let yp = (y + (x >> 1)) as usize;
                    (l[yp] + l[yp + 1] + 1) >> 1
                }
                1 | 3 => {
                    // (8-68) (p[-1,y+(x>>1)] + 2*p[-1,y+(x>>1)+1] + p[-1,y+(x>>1)+2] + 2) >> 2
                    let yp = (y + (x >> 1)) as usize;
                    let p2 = if yp + 2 < 4 { l[yp + 2] } else { l[3] };
                    let p1 = if yp + 1 < 4 { l[yp + 1] } else { l[3] };
                    (l[yp] + 2 * p1 + p2 + 2) >> 2
                }
                5 => {
                    // (8-69) (p[-1,2] + 3*p[-1,3] + 2) >> 2
                    (l[2] + 3 * l[3] + 2) >> 2
                }
                _ => {
                    // zHU > 5: (8-70) p[-1,3]
                    l[3]
                }
            };
            out[at4(x as usize, y as usize)] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// §8.3.2 Intra_8x8 prediction
// ---------------------------------------------------------------------------

/// Build a 16-entry array of p'[x, -1] after applying the reference
/// sample filter's substitution rule for x = 8..15 when p[8..=15,-1]
/// is not available.
fn top_row_8x8_raw(s: &Samples8x8) -> [i32; 16] {
    let mut p = [0i32; 16];
    p[0..8].copy_from_slice(&s.top);
    if s.availability.top_right {
        p[8..16].copy_from_slice(&s.top_right);
    } else if s.availability.top {
        // §8.3.2.2 — "When samples p[x, -1], with x = 8..15, are
        // marked as 'not available for Intra_8x8 prediction,' and the
        // sample p[7, -1] is marked as 'available for Intra_8x8
        // prediction,' the sample value of p[7, -1] is substituted".
        for i in 8..16 {
            p[i] = s.top[7];
        }
    }
    p
}

/// §8.3.2.2.1 — reference sample filter for Intra_8x8. Given raw
/// reference samples, produce the filtered p' used by the prediction
/// process.
pub fn filter_samples_8x8(raw: &Samples8x8, _bit_depth: u32) -> Samples8x8 {
    let raw_top = top_row_8x8_raw(raw);

    // Availability of p[-1, -1] and p[0,-1] and p[-1,0] is needed.
    let tl_avail = raw.availability.top_left;
    let top_avail = raw.availability.top;
    let left_avail = raw.availability.left;

    // Filtered top row p'[x, -1], x = 0..15. Only computed when top
    // samples p[x, -1] with x = 0..15 are available (which — with
    // substitution — is implied by top_avail).
    let mut filt_top = [0i32; 16];
    let mut filt_top_right = [0i32; 8];

    if top_avail {
        // p'[0,-1]: (8-78) with p[-1,-1] available, else (8-79).
        filt_top[0] = if tl_avail {
            (raw.top_left + 2 * raw_top[0] + raw_top[1] + 2) >> 2
        } else {
            (3 * raw_top[0] + raw_top[1] + 2) >> 2
        };
        // p'[x,-1], x = 1..14: (8-80)
        for x in 1..15 {
            filt_top[x] = (raw_top[x - 1] + 2 * raw_top[x] + raw_top[x + 1] + 2) >> 2;
        }
        // p'[15,-1]: (8-81)
        filt_top[15] = (raw_top[14] + 3 * raw_top[15] + 2) >> 2;
    }

    // Split filt_top into top (0..8) and top_right (8..16).
    let mut filt_top_8 = [0i32; 8];
    filt_top_8.copy_from_slice(&filt_top[0..8]);
    filt_top_right.copy_from_slice(&filt_top[8..16]);

    // p'[-1,-1]: derivation depends on availability of p[0,-1] and p[-1,0].
    // Per §8.3.2.2.1 "When the sample p[-1, -1] is marked as 'available'".
    let filt_tl = if tl_avail {
        if !top_avail || !left_avail {
            // One of p[0,-1] / p[-1,0] unavailable.
            if top_avail {
                // p[0,-1] available, p[-1,0] not: (8-82)
                (3 * raw.top_left + raw_top[0] + 2) >> 2
            } else if left_avail {
                // p[-1,0] available, p[0,-1] not: (8-83)
                (3 * raw.top_left + raw.left[0] + 2) >> 2
            } else {
                // Both p[0,-1] and p[-1,0] unavailable.
                // Per spec NOTE: p'[-1,-1] is not used, but assign
                // raw.top_left as a harmless default.
                raw.top_left
            }
        } else {
            // Both p[0,-1] and p[-1,0] available: (8-84)
            (raw_top[0] + 2 * raw.top_left + raw.left[0] + 2) >> 2
        }
    } else {
        raw.top_left
    };

    // Filtered left column p'[-1, y], y = 0..7.
    let mut filt_left = [0i32; 8];
    if left_avail {
        // p'[-1, 0]: (8-85) with p[-1,-1] available, else (8-86).
        filt_left[0] = if tl_avail {
            (raw.top_left + 2 * raw.left[0] + raw.left[1] + 2) >> 2
        } else {
            (3 * raw.left[0] + raw.left[1] + 2) >> 2
        };
        // p'[-1, y], y = 1..6: (8-87)
        for y in 1..7 {
            filt_left[y] = (raw.left[y - 1] + 2 * raw.left[y] + raw.left[y + 1] + 2) >> 2;
        }
        // p'[-1, 7]: (8-88)
        filt_left[7] = (raw.left[6] + 3 * raw.left[7] + 2) >> 2;
    }

    Samples8x8 {
        top_left: filt_tl,
        top: filt_top_8,
        top_right: filt_top_right,
        left: filt_left,
        availability: raw.availability,
    }
}

/// Build p'[x, -1] for x = 0..=15 from a filtered Samples8x8 (top
/// plus top_right concatenated).
fn filtered_top_16(s: &Samples8x8) -> [i32; 16] {
    let mut p = [0i32; 16];
    p[0..8].copy_from_slice(&s.top);
    p[8..16].copy_from_slice(&s.top_right);
    p
}

/// Predict an 8x8 block. `samples` is assumed to already contain the
/// filtered reference samples (i.e. the caller has run
/// `filter_samples_8x8`). `out` is row-major 8x8 (64 samples).
pub fn predict_8x8(mode: Intra8x8Mode, samples: &Samples8x8, bit_depth: u32, out: &mut [i32; 64]) {
    match mode {
        Intra8x8Mode::Vertical => predict_8x8_vertical(samples, out),
        Intra8x8Mode::Horizontal => predict_8x8_horizontal(samples, out),
        Intra8x8Mode::Dc => predict_8x8_dc(samples, bit_depth, out),
        Intra8x8Mode::DiagonalDownLeft => predict_8x8_ddl(samples, out),
        Intra8x8Mode::DiagonalDownRight => predict_8x8_ddr(samples, out),
        Intra8x8Mode::VerticalRight => predict_8x8_vr(samples, out),
        Intra8x8Mode::HorizontalDown => predict_8x8_hd(samples, out),
        Intra8x8Mode::VerticalLeft => predict_8x8_vl(samples, out),
        Intra8x8Mode::HorizontalUp => predict_8x8_hu(samples, out),
    }
}

#[inline]
fn at8(x: usize, y: usize) -> usize {
    y * 8 + x
}

// §8.3.2.2.2 — Intra_8x8_Vertical. pred[x,y] = p'[x,-1]
fn predict_8x8_vertical(s: &Samples8x8, out: &mut [i32; 64]) {
    for y in 0..8 {
        for x in 0..8 {
            out[at8(x, y)] = s.top[x];
        }
    }
}

// §8.3.2.2.3 — Intra_8x8_Horizontal. pred[x,y] = p'[-1,y]
fn predict_8x8_horizontal(s: &Samples8x8, out: &mut [i32; 64]) {
    for y in 0..8 {
        for x in 0..8 {
            out[at8(x, y)] = s.left[y];
        }
    }
}

// §8.3.2.2.4 — Intra_8x8_DC.
fn predict_8x8_dc(s: &Samples8x8, bit_depth: u32, out: &mut [i32; 64]) {
    let dc = if s.availability.top && s.availability.left {
        // (8-91)
        let sum_top: i32 = s.top.iter().sum();
        let sum_left: i32 = s.left.iter().sum();
        (sum_top + sum_left + 8) >> 4
    } else if s.availability.left {
        // (8-92)
        let sum_left: i32 = s.left.iter().sum();
        (sum_left + 4) >> 3
    } else if s.availability.top {
        // (8-93)
        let sum_top: i32 = s.top.iter().sum();
        (sum_top + 4) >> 3
    } else {
        // (8-94)
        dc_default(bit_depth)
    };
    for sample in out.iter_mut() {
        *sample = dc;
    }
}

// §8.3.2.2.5 — Intra_8x8_Diagonal_Down_Left.
fn predict_8x8_ddl(s: &Samples8x8, out: &mut [i32; 64]) {
    let p = filtered_top_16(s);
    for y in 0..8 {
        for x in 0..8 {
            let v = if x == 7 && y == 7 {
                // (8-95) (p'[14,-1] + 3*p'[15,-1] + 2) >> 2
                (p[14] + 3 * p[15] + 2) >> 2
            } else {
                // (8-96) (p'[x+y,-1] + 2*p'[x+y+1,-1] + p'[x+y+2,-1] + 2) >> 2
                (p[x + y] + 2 * p[x + y + 1] + p[x + y + 2] + 2) >> 2
            };
            out[at8(x, y)] = v;
        }
    }
}

// §8.3.2.2.6 — Intra_8x8_Diagonal_Down_Right.
fn predict_8x8_ddr(s: &Samples8x8, out: &mut [i32; 64]) {
    let tl = s.top_left;
    let t = &s.top;
    let l = &s.left;
    let p_top = |x: i32| -> i32 {
        if x == -1 {
            tl
        } else {
            t[x as usize]
        }
    };
    let p_left = |y: i32| -> i32 {
        if y == -1 {
            tl
        } else {
            l[y as usize]
        }
    };
    for y in 0..8i32 {
        for x in 0..8i32 {
            let v = if x > y {
                // (8-97) (p'[x-y-2,-1] + 2*p'[x-y-1,-1] + p'[x-y,-1] + 2) >> 2
                (p_top(x - y - 2) + 2 * p_top(x - y - 1) + p_top(x - y) + 2) >> 2
            } else if x < y {
                // (8-98) (p'[-1,y-x-2] + 2*p'[-1,y-x-1] + p'[-1,y-x] + 2) >> 2
                (p_left(y - x - 2) + 2 * p_left(y - x - 1) + p_left(y - x) + 2) >> 2
            } else {
                // (8-99) (p'[0,-1] + 2*p'[-1,-1] + p'[-1,0] + 2) >> 2
                (p_top(0) + 2 * tl + p_left(0) + 2) >> 2
            };
            out[at8(x as usize, y as usize)] = v;
        }
    }
}

// §8.3.2.2.7 — Intra_8x8_Vertical_Right.
fn predict_8x8_vr(s: &Samples8x8, out: &mut [i32; 64]) {
    let tl = s.top_left;
    let p = filtered_top_16(s);
    let l = &s.left;
    let p_top = |x: i32| -> i32 {
        if x == -1 {
            tl
        } else {
            p[x as usize]
        }
    };
    let p_left = |y: i32| -> i32 {
        if y == -1 {
            tl
        } else {
            l[y as usize]
        }
    };
    for y in 0..8i32 {
        for x in 0..8i32 {
            let z_vr = 2 * x - y;
            let v = match z_vr {
                0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 => {
                    // (8-100) (p'[x-(y>>1)-1,-1] + p'[x-(y>>1),-1] + 1) >> 1
                    let xp = x - (y >> 1) - 1;
                    (p_top(xp) + p_top(xp + 1) + 1) >> 1
                }
                1 | 3 | 5 | 7 | 9 | 11 | 13 => {
                    // (8-101) (p'[x-(y>>1)-2,-1] + 2*p'[x-(y>>1)-1,-1] + p'[x-(y>>1),-1] + 2) >> 2
                    let xp = x - (y >> 1) - 2;
                    (p_top(xp) + 2 * p_top(xp + 1) + p_top(xp + 2) + 2) >> 2
                }
                -1 => {
                    // (8-102) (p'[-1,0] + 2*p'[-1,-1] + p'[0,-1] + 2) >> 2
                    (p_left(0) + 2 * tl + p_top(0) + 2) >> 2
                }
                _ => {
                    // -2..=-7: (8-103) (p'[-1,y-2*x-1] + 2*p'[-1,y-2*x-2] + p'[-1,y-2*x-3] + 2) >> 2
                    (p_left(y - 2 * x - 1) + 2 * p_left(y - 2 * x - 2) + p_left(y - 2 * x - 3) + 2)
                        >> 2
                }
            };
            out[at8(x as usize, y as usize)] = v;
        }
    }
}

// §8.3.2.2.8 — Intra_8x8_Horizontal_Down.
fn predict_8x8_hd(s: &Samples8x8, out: &mut [i32; 64]) {
    let tl = s.top_left;
    let t = &s.top;
    let l = &s.left;
    let p_top = |x: i32| -> i32 {
        if x == -1 {
            tl
        } else {
            t[x as usize]
        }
    };
    let p_left = |y: i32| -> i32 {
        if y == -1 {
            tl
        } else {
            l[y as usize]
        }
    };
    for y in 0..8i32 {
        for x in 0..8i32 {
            let z_hd = 2 * y - x;
            let v = match z_hd {
                0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 => {
                    // (8-104) (p'[-1,y-(x>>1)-1] + p'[-1,y-(x>>1)] + 1) >> 1
                    let yp = y - (x >> 1) - 1;
                    (p_left(yp) + p_left(yp + 1) + 1) >> 1
                }
                1 | 3 | 5 | 7 | 9 | 11 | 13 => {
                    // (8-105) (p'[-1,y-(x>>1)-2] + 2*p'[-1,y-(x>>1)-1] + p'[-1,y-(x>>1)] + 2) >> 2
                    let yp = y - (x >> 1) - 2;
                    (p_left(yp) + 2 * p_left(yp + 1) + p_left(yp + 2) + 2) >> 2
                }
                -1 => {
                    // (8-106) (p'[-1,0] + 2*p'[-1,-1] + p'[0,-1] + 2) >> 2
                    (p_left(0) + 2 * tl + p_top(0) + 2) >> 2
                }
                _ => {
                    // -2..=-7: (8-107) (p'[x-2*y-1,-1] + 2*p'[x-2*y-2,-1] + p'[x-2*y-3,-1] + 2) >> 2
                    (p_top(x - 2 * y - 1) + 2 * p_top(x - 2 * y - 2) + p_top(x - 2 * y - 3) + 2)
                        >> 2
                }
            };
            out[at8(x as usize, y as usize)] = v;
        }
    }
}

// §8.3.2.2.9 — Intra_8x8_Vertical_Left.
fn predict_8x8_vl(s: &Samples8x8, out: &mut [i32; 64]) {
    let p = filtered_top_16(s);
    for y in 0..8 {
        for x in 0..8 {
            let v = if y == 0 || y == 2 || y == 4 || y == 6 {
                // (8-108) (p'[x+(y>>1),-1] + p'[x+(y>>1)+1,-1] + 1) >> 1
                let xp = x + (y >> 1);
                (p[xp] + p[xp + 1] + 1) >> 1
            } else {
                // (8-109) (p'[x+(y>>1),-1] + 2*p'[x+(y>>1)+1,-1] + p'[x+(y>>1)+2,-1] + 2) >> 2
                let xp = x + (y >> 1);
                (p[xp] + 2 * p[xp + 1] + p[xp + 2] + 2) >> 2
            };
            out[at8(x, y)] = v;
        }
    }
}

// §8.3.2.2.10 — Intra_8x8_Horizontal_Up.
fn predict_8x8_hu(s: &Samples8x8, out: &mut [i32; 64]) {
    let l = &s.left;
    for y in 0..8i32 {
        for x in 0..8i32 {
            let z_hu = x + 2 * y;
            let v = match z_hu {
                0 | 2 | 4 | 6 | 8 | 10 | 12 => {
                    // (8-110) (p'[-1,y+(x>>1)] + p'[-1,y+(x>>1)+1] + 1) >> 1
                    let yp = (y + (x >> 1)) as usize;
                    (l[yp] + l[yp + 1] + 1) >> 1
                }
                1 | 3 | 5 | 7 | 9 | 11 => {
                    // (8-111) (p'[-1,y+(x>>1)] + 2*p'[-1,y+(x>>1)+1] + p'[-1,y+(x>>1)+2] + 2) >> 2
                    let yp = (y + (x >> 1)) as usize;
                    let p1 = if yp + 1 < 8 { l[yp + 1] } else { l[7] };
                    let p2 = if yp + 2 < 8 { l[yp + 2] } else { l[7] };
                    (l[yp] + 2 * p1 + p2 + 2) >> 2
                }
                13 => {
                    // (8-112) (p'[-1,6] + 3*p'[-1,7] + 2) >> 2
                    (l[6] + 3 * l[7] + 2) >> 2
                }
                _ => {
                    // z_hu > 13: (8-113) p'[-1,7]
                    l[7]
                }
            };
            out[at8(x as usize, y as usize)] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// §8.3.3 Intra_16x16 prediction
// ---------------------------------------------------------------------------

/// Predict a 16x16 block. `out` is row-major 16x16 (256 samples).
pub fn predict_16x16(
    mode: Intra16x16Mode,
    samples: &Samples16x16,
    bit_depth: u32,
    out: &mut [i32; 256],
) {
    match mode {
        Intra16x16Mode::Vertical => predict_16x16_vertical(samples, out),
        Intra16x16Mode::Horizontal => predict_16x16_horizontal(samples, out),
        Intra16x16Mode::Dc => predict_16x16_dc(samples, bit_depth, out),
        Intra16x16Mode::Plane => predict_16x16_plane(samples, bit_depth, out),
    }
}

#[inline]
fn at16(x: usize, y: usize) -> usize {
    y * 16 + x
}

// §8.3.3.1 — Intra_16x16_Vertical. (8-116) pred[x,y] = p[x,-1]
fn predict_16x16_vertical(s: &Samples16x16, out: &mut [i32; 256]) {
    for y in 0..16 {
        for x in 0..16 {
            out[at16(x, y)] = s.top[x];
        }
    }
}

// §8.3.3.2 — Intra_16x16_Horizontal. (8-117) pred[x,y] = p[-1,y]
fn predict_16x16_horizontal(s: &Samples16x16, out: &mut [i32; 256]) {
    for y in 0..16 {
        for x in 0..16 {
            out[at16(x, y)] = s.left[y];
        }
    }
}

// §8.3.3.3 — Intra_16x16_DC.
fn predict_16x16_dc(s: &Samples16x16, bit_depth: u32, out: &mut [i32; 256]) {
    let dc = if s.availability.top && s.availability.left {
        // (8-118)
        let sum_top: i32 = s.top.iter().sum();
        let sum_left: i32 = s.left.iter().sum();
        (sum_top + sum_left + 16) >> 5
    } else if s.availability.left {
        // (8-119)
        let sum_left: i32 = s.left.iter().sum();
        (sum_left + 8) >> 4
    } else if s.availability.top {
        // (8-120)
        let sum_top: i32 = s.top.iter().sum();
        (sum_top + 8) >> 4
    } else {
        // (8-121)
        dc_default(bit_depth)
    };
    for sample in out.iter_mut() {
        *sample = dc;
    }
}

// §8.3.3.4 — Intra_16x16_Plane.
fn predict_16x16_plane(s: &Samples16x16, bit_depth: u32, out: &mut [i32; 256]) {
    // (8-126) H = sum over x'=0..=7 of (x'+1) * (p[8+x', -1] - p[6-x', -1])
    // (8-127) V = sum over y'=0..=7 of (y'+1) * (p[-1, 8+y'] - p[-1, 6-y'])
    //
    // Note that p[7,-1] for x'=-1 is not referenced; the loop starts
    // at x'=0 which uses p[8,-1] and p[6,-1]. When x'=7 we use p[15,-1]
    // and p[-1,-1]. When y'=7 we use p[-1,15] and p[-1,-1].
    let p_top = |x: i32| -> i32 {
        if x == -1 {
            s.top_left
        } else {
            s.top[x as usize]
        }
    };
    let p_left = |y: i32| -> i32 {
        if y == -1 {
            s.top_left
        } else {
            s.left[y as usize]
        }
    };

    let mut h: i32 = 0;
    for xp in 0..8i32 {
        h += (xp + 1) * (p_top(8 + xp) - p_top(6 - xp));
    }
    let mut v: i32 = 0;
    for yp in 0..8i32 {
        v += (yp + 1) * (p_left(8 + yp) - p_left(6 - yp));
    }

    // (8-123) a = 16 * (p[-1,15] + p[15,-1])
    let a = 16 * (s.left[15] + s.top[15]);
    // (8-124) b = (5 * H + 32) >> 6
    let b = (5 * h + 32) >> 6;
    // (8-125) c = (5 * V + 32) >> 6
    let c = (5 * v + 32) >> 6;

    for y in 0..16i32 {
        for x in 0..16i32 {
            // (8-122) pred[x,y] = Clip1_Y((a + b*(x-7) + c*(y-7) + 16) >> 5)
            let raw = (a + b * (x - 7) + c * (y - 7) + 16) >> 5;
            out[at16(x as usize, y as usize)] = clip1(raw, bit_depth);
        }
    }
}

// ---------------------------------------------------------------------------
// §8.3.4 Intra chroma prediction
// ---------------------------------------------------------------------------

/// Predict a chroma block. `out` must be exactly `8 * chroma_type.height()`
/// samples (64 for 4:2:0, 128 for 4:2:2), row-major width-first.
pub fn predict_chroma(
    mode: IntraChromaMode,
    samples: &SamplesChroma,
    chroma_type: ChromaArrayType,
    bit_depth: u32,
    out: &mut [i32],
) {
    let w = chroma_type.width();
    let h = chroma_type.height();
    assert_eq!(out.len(), w * h, "chroma output buffer size mismatch");
    assert_eq!(samples.top.len(), w, "chroma top row length mismatch");
    assert_eq!(samples.left.len(), h, "chroma left column length mismatch");

    match mode {
        IntraChromaMode::Dc => predict_chroma_dc(samples, chroma_type, bit_depth, out),
        IntraChromaMode::Horizontal => predict_chroma_horizontal(samples, chroma_type, out),
        IntraChromaMode::Vertical => predict_chroma_vertical(samples, chroma_type, out),
        IntraChromaMode::Plane => predict_chroma_plane(samples, chroma_type, bit_depth, out),
    }
}

// §8.3.4.1 — Intra_Chroma_DC.
// Chroma is partitioned into 4x4 sub-blocks for DC derivation.
// For 4:2:0 (8x8): 4 sub-blocks at (xO,yO) in {(0,0),(4,0),(0,4),(4,4)}.
// For 4:2:2 (8x16): 8 sub-blocks at (xO,yO) in
//   {(0,0),(4,0),(0,4),(4,4),(0,8),(4,8),(0,12),(4,12)}.
// The spec's decision tree by (xO, yO) gives different DC formulas.
fn predict_chroma_dc(
    s: &SamplesChroma,
    chroma_type: ChromaArrayType,
    bit_depth: u32,
    out: &mut [i32],
) {
    let w = chroma_type.width();
    let h = chroma_type.height();
    let default_dc = dc_default(bit_depth);

    // Iterate over 4x4 sub-blocks, indexed by their top-left (xO, yO).
    let mut y_o = 0;
    while y_o < h {
        let mut x_o = 0;
        while x_o < w {
            let dc = chroma_dc_subblock(s, x_o, y_o, w, h, default_dc);
            // Fill the 4x4 sub-block with `dc`.
            for yy in 0..4 {
                for xx in 0..4 {
                    out[(y_o + yy) * w + (x_o + xx)] = dc;
                }
            }
            x_o += 4;
        }
        y_o += 4;
    }
}

/// Compute DC for a single 4x4 chroma sub-block at (xO, yO).
fn chroma_dc_subblock(
    s: &SamplesChroma,
    x_o: usize,
    y_o: usize,
    _w: usize,
    _h: usize,
    default_dc: i32,
) -> i32 {
    // Availability of the four relevant reference samples:
    //   top samples p[xO..xO+3, -1]
    //   left samples p[-1, yO..yO+3]
    let top_avail = s.availability.top;
    let left_avail = s.availability.left;

    let sum_top = || -> i32 { s.top[x_o] + s.top[x_o + 1] + s.top[x_o + 2] + s.top[x_o + 3] };
    let sum_left = || -> i32 { s.left[y_o] + s.left[y_o + 1] + s.left[y_o + 2] + s.left[y_o + 3] };

    if x_o == 0 && y_o == 0 {
        // (xO, yO) == (0, 0): use both top and left. §8.3.4.1
        if top_avail && left_avail {
            // (8-132)
            (sum_top() + sum_left() + 4) >> 3
        } else if top_avail {
            // (8-133) uses only top  -- wait, (8-133) is left-only.
            // Re-read: (8-133) uses p[-1, y'+yO], y'=0..3. That's the
            // left column. (8-134) uses p[x'+xO, -1]. That's the top row.
            // (8-134) top-only: (sum_top + 2) >> 2
            (sum_top() + 2) >> 2
        } else if left_avail {
            // (8-133) left-only: (sum_left + 2) >> 2
            (sum_left() + 2) >> 2
        } else {
            // (8-135) neither: (1 << (BitDepth - 1))
            default_dc
        }
    } else if x_o > 0 && y_o == 0 {
        // Top row, not leftmost: prefer top samples.
        // (8-136) top available: (sum_top + 2) >> 2
        // (8-137) top not available but left available: (sum_left + 2) >> 2
        // (8-138) neither: default
        if top_avail {
            (sum_top() + 2) >> 2
        } else if left_avail {
            (sum_left() + 2) >> 2
        } else {
            default_dc
        }
    } else if x_o == 0 && y_o > 0 {
        // Left column, not top: prefer left samples.
        // (8-139) left available: (sum_left + 2) >> 2
        // (8-140) left not available but top available: (sum_top + 2) >> 2
        // (8-141) neither: default
        if left_avail {
            (sum_left() + 2) >> 2
        } else if top_avail {
            (sum_top() + 2) >> 2
        } else {
            default_dc
        }
    } else {
        // x_o > 0 && y_o > 0: the spec folds this into the
        // "otherwise" branch of the xO>0, yO>0 case which uses
        // BOTH when available. The standard treatment (per a close
        // reading of §8.3.4.1's ordered cases) mirrors (xO, yO) == (0,0):
        // use both top and left if available, else whichever is
        // available, else default.
        if top_avail && left_avail {
            (sum_top() + sum_left() + 4) >> 3
        } else if top_avail {
            (sum_top() + 2) >> 2
        } else if left_avail {
            (sum_left() + 2) >> 2
        } else {
            default_dc
        }
    }
}

// §8.3.4.2 — Intra_Chroma_Horizontal. (8-142) pred[x,y] = p[-1, y]
fn predict_chroma_horizontal(s: &SamplesChroma, chroma_type: ChromaArrayType, out: &mut [i32]) {
    let w = chroma_type.width();
    let h = chroma_type.height();
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = s.left[y];
        }
    }
}

// §8.3.4.3 — Intra_Chroma_Vertical. (8-143) pred[x,y] = p[x, -1]
fn predict_chroma_vertical(s: &SamplesChroma, chroma_type: ChromaArrayType, out: &mut [i32]) {
    let w = chroma_type.width();
    let h = chroma_type.height();
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = s.top[x];
        }
    }
}

// §8.3.4.4 — Intra_Chroma_Plane.
fn predict_chroma_plane(
    s: &SamplesChroma,
    chroma_type: ChromaArrayType,
    bit_depth: u32,
    out: &mut [i32],
) {
    let w = chroma_type.width();
    let h = chroma_type.height();
    let is_yuv422 = chroma_type == ChromaArrayType::Yuv422;
    // xCF = (ChromaArrayType == 3) ? 4 : 0 — we never get here when ==3.
    let x_cf: i32 = 0;
    // yCF = (ChromaArrayType != 1) ? 4 : 0
    let y_cf: i32 = if is_yuv422 { 4 } else { 0 };

    let p_top = |x: i32| -> i32 {
        if x == -1 {
            s.top_left
        } else {
            s.top[x as usize]
        }
    };
    let p_left = |y: i32| -> i32 {
        if y == -1 {
            s.top_left
        } else {
            s.left[y as usize]
        }
    };

    // (8-148) H = sum over x'=0..=3+xCF of (x'+1) * (p[4+xCF+x', -1] - p[2+xCF-x', -1])
    let mut h_sum: i32 = 0;
    for xp in 0..=(3 + x_cf) {
        h_sum += (xp + 1) * (p_top(4 + x_cf + xp) - p_top(2 + x_cf - xp));
    }
    // (8-149) V = sum over y'=0..=3+yCF of (y'+1) * (p[-1, 4+yCF+y'] - p[-1, 2+yCF-y'])
    let mut v_sum: i32 = 0;
    for yp in 0..=(3 + y_cf) {
        v_sum += (yp + 1) * (p_left(4 + y_cf + yp) - p_left(2 + y_cf - yp));
    }

    // (8-145) a = 16 * (p[-1, MbHeightC - 1] + p[MbWidthC - 1, -1])
    let a = 16 * (s.left[h - 1] + s.top[w - 1]);
    // (8-146) b = ((34 - 29 * (ChromaArrayType == 3)) * H + 32) >> 6
    //         For ChromaArrayType != 3, coefficient is 34.
    let b = (34 * h_sum + 32) >> 6;
    // (8-147) c = ((34 - 29 * (ChromaArrayType != 1)) * V + 32) >> 6
    //         For 4:2:0 (== 1): coefficient is 34.
    //         For 4:2:2 (!= 1, != 3 since we're not 4:4:4 here): 34-29 = 5.
    let c_coef = if is_yuv422 { 5 } else { 34 };
    let c = (c_coef * v_sum + 32) >> 6;

    // (8-144) pred[x,y] = Clip1_C((a + b*(x-3-xCF) + c*(y-3-yCF) + 16) >> 5)
    for y in 0..h {
        for x in 0..w {
            let xi = x as i32;
            let yi = y as i32;
            let raw = (a + b * (xi - 3 - x_cf) + c * (yi - 3 - y_cf) + 16) >> 5;
            out[y * w + x] = clip1(raw, bit_depth);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn all_avail() -> Neighbour4x4Availability {
        Neighbour4x4Availability {
            top_left: true,
            top: true,
            top_right: true,
            left: true,
        }
    }

    fn no_avail() -> Neighbour4x4Availability {
        Neighbour4x4Availability::default()
    }

    // ----- Mode enum round-trips -----

    #[test]
    fn intra4x4_mode_index_roundtrip() {
        for i in 0..=8u8 {
            let m = Intra4x4Mode::from_index(i).unwrap();
            assert_eq!(m.as_index(), i);
        }
        assert!(Intra4x4Mode::from_index(9).is_none());
    }

    #[test]
    fn intra8x8_mode_index_roundtrip() {
        for i in 0..=8u8 {
            let m = Intra8x8Mode::from_index(i).unwrap();
            assert_eq!(m.as_index(), i);
        }
    }

    #[test]
    fn intra16x16_mode_index_roundtrip() {
        for i in 0..=3u8 {
            let m = Intra16x16Mode::from_index(i).unwrap();
            assert_eq!(m.as_index(), i);
        }
        assert!(Intra16x16Mode::from_index(4).is_none());
    }

    #[test]
    fn intra_chroma_mode_index_roundtrip() {
        for i in 0..=3u8 {
            let m = IntraChromaMode::from_index(i).unwrap();
            assert_eq!(m.as_index(), i);
        }
    }

    // ----- Intra 4x4 tests -----

    fn mk_4x4(top: [i32; 4], top_right: [i32; 4], left: [i32; 4], tl: i32) -> Samples4x4 {
        Samples4x4 {
            top_left: tl,
            top,
            top_right,
            left,
            availability: all_avail(),
        }
    }

    #[test]
    fn intra_4x4_dc_all_neighbours() {
        // top=[10,20,30,40], left=[5,15,25,35] → (10+20+30+40+5+15+25+35+4)/8 = 184/8 = 23
        let s = mk_4x4([10, 20, 30, 40], [50, 60, 70, 80], [5, 15, 25, 35], 0);
        let mut out = [0i32; 16];
        predict_4x4(Intra4x4Mode::Dc, &s, 8, &mut out);
        for v in out.iter() {
            assert_eq!(*v, 23);
        }
    }

    #[test]
    fn intra_4x4_dc_no_neighbours() {
        let mut s = mk_4x4([0; 4], [0; 4], [0; 4], 0);
        s.availability = no_avail();
        let mut out = [0i32; 16];
        predict_4x4(Intra4x4Mode::Dc, &s, 8, &mut out);
        for v in out.iter() {
            assert_eq!(*v, 128);
        }
    }

    #[test]
    fn intra_4x4_dc_only_left() {
        // left = [4,8,12,16] → (4+8+12+16+2)/4 = 42/4 = 10
        let mut s = mk_4x4([0; 4], [0; 4], [4, 8, 12, 16], 0);
        s.availability = Neighbour4x4Availability {
            top_left: false,
            top: false,
            top_right: false,
            left: true,
        };
        let mut out = [0i32; 16];
        predict_4x4(Intra4x4Mode::Dc, &s, 8, &mut out);
        for v in out.iter() {
            assert_eq!(*v, 10);
        }
    }

    #[test]
    fn intra_4x4_dc_only_top() {
        // top = [1,3,5,7] → (1+3+5+7+2)/4 = 18/4 = 4
        let mut s = mk_4x4([1, 3, 5, 7], [0; 4], [0; 4], 0);
        s.availability = Neighbour4x4Availability {
            top_left: false,
            top: true,
            top_right: false,
            left: false,
        };
        let mut out = [0i32; 16];
        predict_4x4(Intra4x4Mode::Dc, &s, 8, &mut out);
        for v in out.iter() {
            assert_eq!(*v, 4);
        }
    }

    #[test]
    fn intra_4x4_vertical() {
        let s = mk_4x4([11, 22, 33, 44], [0; 4], [0; 4], 0);
        let mut out = [0i32; 16];
        predict_4x4(Intra4x4Mode::Vertical, &s, 8, &mut out);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out[at4(x, y)], s.top[x]);
            }
        }
    }

    #[test]
    fn intra_4x4_horizontal() {
        let s = mk_4x4([0; 4], [0; 4], [7, 14, 21, 28], 0);
        let mut out = [0i32; 16];
        predict_4x4(Intra4x4Mode::Horizontal, &s, 8, &mut out);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out[at4(x, y)], s.left[y]);
            }
        }
    }

    #[test]
    fn intra_4x4_ddl_hand() {
        // top = [0,1,2,3], top_right = [4,5,6,7]
        // p = [0,1,2,3,4,5,6,7]
        // pred[0,0] = (p[0]+2*p[1]+p[2]+2)>>2 = (0+2+2+2)>>2 = 6>>2 = 1
        // pred[3,3] = (p[6]+3*p[7]+2)>>2 = (6+21+2)>>2 = 29>>2 = 7
        let s = mk_4x4([0, 1, 2, 3], [4, 5, 6, 7], [0; 4], 0);
        let mut out = [0i32; 16];
        predict_4x4(Intra4x4Mode::DiagonalDownLeft, &s, 8, &mut out);
        assert_eq!(out[at4(0, 0)], 1);
        assert_eq!(out[at4(3, 3)], 7);
        // pred[1,0] = (p[1]+2*p[2]+p[3]+2)>>2 = (1+4+3+2)>>2 = 10>>2 = 2
        assert_eq!(out[at4(1, 0)], 2);
        // pred[0,1] same as pred[1,0]: (p[1]+2*p[2]+p[3]+2)>>2 = 2
        assert_eq!(out[at4(0, 1)], 2);
    }

    #[test]
    fn intra_4x4_ddl_top_right_substitution() {
        // When top_right is not available but top is, replicate p[3,-1].
        // top = [10,20,30,40], top_right effectively = [40,40,40,40].
        // pred[3,0] = (p[3]+2*p[4]+p[5]+2)>>2 = (40+80+40+2)>>2 = 162>>2 = 40
        let mut s = mk_4x4([10, 20, 30, 40], [99; 4], [0; 4], 0);
        s.availability.top_right = false;
        let mut out = [0i32; 16];
        predict_4x4(Intra4x4Mode::DiagonalDownLeft, &s, 8, &mut out);
        assert_eq!(out[at4(3, 0)], 40);
    }

    #[test]
    fn intra_4x4_ddr_hand() {
        // top=[10,20,30,40], left=[50,60,70,80], tl=100
        // pred[0,0] = (p[0,-1] + 2*p[-1,-1] + p[-1,0] + 2)>>2 = (10+200+50+2)>>2 = 262>>2 = 65
        // pred[1,0] (x>y): (p[-1,-1]+2*p[0,-1]+p[1,-1]+2)>>2 = (100+20+20+2)>>2
        //  = (100+2*10+20+2)>>2... wait: p[x-y-2,-1] with x=1,y=0 = p[-1,-1]=tl=100;
        //  p[x-y-1,-1] = p[0,-1]=10; p[x-y,-1]=p[1,-1]=20 → (100+20+20+2)>>2 = 142>>2 = 35
        let s = mk_4x4([10, 20, 30, 40], [0; 4], [50, 60, 70, 80], 100);
        let mut out = [0i32; 16];
        predict_4x4(Intra4x4Mode::DiagonalDownRight, &s, 8, &mut out);
        assert_eq!(out[at4(0, 0)], 65);
        assert_eq!(out[at4(1, 0)], 35);
        // pred[0,1] (x<y): (p[-1,y-x-2]+2*p[-1,y-x-1]+p[-1,y-x]+2)>>2
        //  y-x-2=-1 → tl=100; y-x-1=0 → left[0]=50; y-x=1 → left[1]=60
        //  (100+100+60+2)>>2 = 262>>2 = 65
        assert_eq!(out[at4(0, 1)], 65);
    }

    #[test]
    fn intra_4x4_vr_hand() {
        // top=[10,20,30,40], tl=50, left=[60,70,80,90]
        // pred[0,0]: zVR=0, (p[-1,-1]+p[0,-1]+1)>>1 = (50+10+1)>>1 = 61>>1 = 30
        let s = mk_4x4([10, 20, 30, 40], [0; 4], [60, 70, 80, 90], 50);
        let mut out = [0i32; 16];
        predict_4x4(Intra4x4Mode::VerticalRight, &s, 8, &mut out);
        assert_eq!(out[at4(0, 0)], 30);
        // pred[0,1]: x=0,y=1, zVR=-1 → (p[-1,0]+2*p[-1,-1]+p[0,-1]+2)>>2 = (60+100+10+2)>>2 = 172>>2 = 43
        assert_eq!(out[at4(0, 1)], 43);
    }

    #[test]
    fn intra_4x4_hd_hand() {
        let s = mk_4x4([10, 20, 30, 40], [0; 4], [60, 70, 80, 90], 50);
        let mut out = [0i32; 16];
        predict_4x4(Intra4x4Mode::HorizontalDown, &s, 8, &mut out);
        // pred[0,0]: zHD=0 → (p[-1,-1]+p[-1,0]+1)>>1 = (50+60+1)>>1 = 55
        assert_eq!(out[at4(0, 0)], 55);
        // pred[1,0]: zHD=-1 → (p[-1,0]+2*p[-1,-1]+p[0,-1]+2)>>2 = (60+100+10+2)>>2 = 43
        assert_eq!(out[at4(1, 0)], 43);
    }

    #[test]
    fn intra_4x4_vl_hand() {
        // top=[10,20,30,40], top_right=[50,60,70,80]
        // pred[0,0]: y=0 → (p[0,-1]+p[1,-1]+1)>>1 = (10+20+1)>>1 = 31>>1 = 15
        // pred[0,1]: y=1 → (p[0,-1]+2*p[1,-1]+p[2,-1]+2)>>2 = (10+40+30+2)>>2 = 82>>2 = 20
        let s = mk_4x4([10, 20, 30, 40], [50, 60, 70, 80], [0; 4], 0);
        let mut out = [0i32; 16];
        predict_4x4(Intra4x4Mode::VerticalLeft, &s, 8, &mut out);
        assert_eq!(out[at4(0, 0)], 15);
        assert_eq!(out[at4(0, 1)], 20);
    }

    #[test]
    fn intra_4x4_hu_hand() {
        // left = [10, 20, 30, 40]
        // pred[0,0]: zHU=0 → (p[-1,0]+p[-1,1]+1)>>1 = (10+20+1)>>1 = 15
        // pred[1,0]: zHU=1 → (p[-1,0]+2*p[-1,1]+p[-1,2]+2)>>2 = (10+40+30+2)>>2 = 20
        // pred[3,3]: zHU=9 >5 → p[-1,3] = 40
        let s = mk_4x4([0; 4], [0; 4], [10, 20, 30, 40], 0);
        let mut out = [0i32; 16];
        predict_4x4(Intra4x4Mode::HorizontalUp, &s, 8, &mut out);
        assert_eq!(out[at4(0, 0)], 15);
        assert_eq!(out[at4(1, 0)], 20);
        assert_eq!(out[at4(3, 3)], 40);
        // pred[1,2]: zHU=5 → (p[-1,2]+3*p[-1,3]+2)>>2 = (30+120+2)>>2 = 152>>2 = 38
        assert_eq!(out[at4(1, 2)], 38);
    }

    // ----- Intra 8x8 tests -----

    fn mk_8x8(top: [i32; 8], top_right: [i32; 8], left: [i32; 8], tl: i32) -> Samples8x8 {
        Samples8x8 {
            top_left: tl,
            top,
            top_right,
            left,
            availability: all_avail(),
        }
    }

    #[test]
    fn intra_8x8_filter_basic() {
        // All 100s -> filter should still produce 100s.
        let raw = mk_8x8([100; 8], [100; 8], [100; 8], 100);
        let filtered = filter_samples_8x8(&raw, 8);
        for v in filtered.top.iter() {
            assert_eq!(*v, 100);
        }
        for v in filtered.top_right.iter() {
            assert_eq!(*v, 100);
        }
        for v in filtered.left.iter() {
            assert_eq!(*v, 100);
        }
        assert_eq!(filtered.top_left, 100);
    }

    #[test]
    fn intra_8x8_filter_known() {
        // top = [0,4,8,12,16,20,24,28], top_right=[32,36,40,44,48,52,56,60]
        // tl = 50, left = [100,100,100,100,100,100,100,100]
        //
        // p'[0,-1] = (tl + 2*top[0] + top[1] + 2) >> 2 = (50+0+4+2)>>2 = 56>>2 = 14
        let raw = mk_8x8(
            [0, 4, 8, 12, 16, 20, 24, 28],
            [32, 36, 40, 44, 48, 52, 56, 60],
            [100; 8],
            50,
        );
        let filtered = filter_samples_8x8(&raw, 8);
        assert_eq!(filtered.top[0], 14);
        // p'[1,-1] = (top[0]+2*top[1]+top[2]+2)>>2 = (0+8+8+2)>>2 = 18>>2 = 4
        assert_eq!(filtered.top[1], 4);
        // p'[15,-1] = (top_right[6]+3*top_right[7]+2)>>2 = (56+180+2)>>2 = 238>>2 = 59
        assert_eq!(filtered.top_right[7], 59);
        // p'[-1,0] = (tl+2*left[0]+left[1]+2)>>2 = (50+200+100+2)>>2 = 352>>2 = 88
        assert_eq!(filtered.left[0], 88);
        // p'[-1,7] = (left[6]+3*left[7]+2)>>2 = (100+300+2)>>2 = 402>>2 = 100
        assert_eq!(filtered.left[7], 100);
        // p'[-1,-1] = (top[0]+2*tl+left[0]+2)>>2 = (0+100+100+2)>>2 = 202>>2 = 50
        assert_eq!(filtered.top_left, 50);
    }

    #[test]
    fn intra_8x8_dc_all() {
        // All 100s (filtered == raw). Sum = 800+800=1600, (1600+8)>>4 = 1608>>4 = 100.
        let raw = mk_8x8([100; 8], [100; 8], [100; 8], 100);
        let filtered = filter_samples_8x8(&raw, 8);
        let mut out = [0i32; 64];
        predict_8x8(Intra8x8Mode::Dc, &filtered, 8, &mut out);
        for v in out.iter() {
            assert_eq!(*v, 100);
        }
    }

    #[test]
    fn intra_8x8_dc_no_neighbours() {
        let mut raw = mk_8x8([0; 8], [0; 8], [0; 8], 0);
        raw.availability = no_avail();
        let filtered = filter_samples_8x8(&raw, 8);
        let mut out = [0i32; 64];
        predict_8x8(Intra8x8Mode::Dc, &filtered, 8, &mut out);
        for v in out.iter() {
            assert_eq!(*v, 128);
        }
    }

    #[test]
    fn intra_8x8_vertical() {
        let raw = mk_8x8([1, 2, 3, 4, 5, 6, 7, 8], [9; 8], [0; 8], 0);
        let filtered = filter_samples_8x8(&raw, 8);
        let mut out = [0i32; 64];
        predict_8x8(Intra8x8Mode::Vertical, &filtered, 8, &mut out);
        // Output rows should repeat the filtered top row.
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(out[at8(x, y)], filtered.top[x]);
            }
        }
    }

    #[test]
    fn intra_8x8_horizontal() {
        let raw = mk_8x8([0; 8], [0; 8], [10, 20, 30, 40, 50, 60, 70, 80], 0);
        let filtered = filter_samples_8x8(&raw, 8);
        let mut out = [0i32; 64];
        predict_8x8(Intra8x8Mode::Horizontal, &filtered, 8, &mut out);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(out[at8(x, y)], filtered.left[y]);
            }
        }
    }

    #[test]
    fn intra_8x8_ddl_constant() {
        // All-constant reference → every prediction should equal that constant.
        let raw = mk_8x8([77; 8], [77; 8], [77; 8], 77);
        let filtered = filter_samples_8x8(&raw, 8);
        let mut out = [0i32; 64];
        predict_8x8(Intra8x8Mode::DiagonalDownLeft, &filtered, 8, &mut out);
        for v in out.iter() {
            assert_eq!(*v, 77);
        }
    }

    #[test]
    fn intra_8x8_ddr_constant() {
        let raw = mk_8x8([42; 8], [42; 8], [42; 8], 42);
        let filtered = filter_samples_8x8(&raw, 8);
        let mut out = [0i32; 64];
        predict_8x8(Intra8x8Mode::DiagonalDownRight, &filtered, 8, &mut out);
        for v in out.iter() {
            assert_eq!(*v, 42);
        }
    }

    // ----- Intra 16x16 tests -----

    fn mk_16x16(top: [i32; 16], left: [i32; 16], tl: i32) -> Samples16x16 {
        Samples16x16 {
            top_left: tl,
            top,
            left,
            availability: all_avail(),
        }
    }

    #[test]
    fn intra_16x16_vertical() {
        let mut top = [0i32; 16];
        for (i, t) in top.iter_mut().enumerate() {
            *t = i as i32 * 2;
        }
        let s = mk_16x16(top, [0; 16], 0);
        let mut out = [0i32; 256];
        predict_16x16(Intra16x16Mode::Vertical, &s, 8, &mut out);
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(out[at16(x, y)], top[x]);
            }
        }
    }

    #[test]
    fn intra_16x16_horizontal() {
        let mut left = [0i32; 16];
        for (i, l) in left.iter_mut().enumerate() {
            *l = i as i32 + 5;
        }
        let s = mk_16x16([0; 16], left, 0);
        let mut out = [0i32; 256];
        predict_16x16(Intra16x16Mode::Horizontal, &s, 8, &mut out);
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(out[at16(x, y)], left[y]);
            }
        }
    }

    #[test]
    fn intra_16x16_dc_all_100() {
        let s = mk_16x16([100; 16], [100; 16], 100);
        let mut out = [0i32; 256];
        predict_16x16(Intra16x16Mode::Dc, &s, 8, &mut out);
        for v in out.iter() {
            assert_eq!(*v, 100);
        }
    }

    #[test]
    fn intra_16x16_dc_no_neighbours() {
        let mut s = mk_16x16([0; 16], [0; 16], 0);
        s.availability = no_avail();
        let mut out = [0i32; 256];
        predict_16x16(Intra16x16Mode::Dc, &s, 8, &mut out);
        for v in out.iter() {
            assert_eq!(*v, 128);
        }
    }

    #[test]
    fn intra_16x16_plane_constant() {
        // All samples identical → H = V = 0, b = c = 0, a = 16*(v+v) = 32v
        // pred = (32v + 16) >> 5 = v (for v < 128 this is fine).
        let s = mk_16x16([80; 16], [80; 16], 80);
        let mut out = [0i32; 256];
        predict_16x16(Intra16x16Mode::Plane, &s, 8, &mut out);
        for v in out.iter() {
            assert_eq!(*v, 80);
        }
    }

    #[test]
    fn intra_16x16_plane_hand() {
        // Construct: top = 0..=15, left = 0..=15, tl = 0.
        // H = sum_{x'=0..7} (x'+1)*(top[8+x']-top[6-x'])
        //   x'=0: 1 * (8 - 6) = 2
        //   x'=1: 2 * (9 - 5) = 8
        //   x'=2: 3 * (10- 4) = 18
        //   x'=3: 4 * (11 - 3) = 32
        //   x'=4: 5 * (12 - 2) = 50
        //   x'=5: 6 * (13 - 1) = 72
        //   x'=6: 7 * (14 - 0) = 98
        //   x'=7: 8 * (15 - p[-1,-1]) = 8 * 15 = 120
        // H = 2+8+18+32+50+72+98+120 = 400
        // V = 400 (symmetry)
        // a = 16 * (left[15] + top[15]) = 16 * 30 = 480
        // b = (5*400 + 32) >> 6 = 2032 >> 6 = 31
        // c = 31
        // pred[7,7] = Clip1((480 + 31*0 + 31*0 + 16) >> 5) = 496>>5 = 15
        let mut top = [0i32; 16];
        let mut left = [0i32; 16];
        for i in 0..16 {
            top[i] = i as i32;
            left[i] = i as i32;
        }
        let s = mk_16x16(top, left, 0);
        let mut out = [0i32; 256];
        predict_16x16(Intra16x16Mode::Plane, &s, 8, &mut out);
        assert_eq!(out[at16(7, 7)], 15);
    }

    // ----- Intra chroma tests -----

    fn mk_chroma_420(top: Vec<i32>, left: Vec<i32>, tl: i32) -> SamplesChroma {
        SamplesChroma {
            top_left: tl,
            top,
            left,
            availability: all_avail(),
        }
    }

    #[test]
    fn intra_chroma_dc_420_all_avail() {
        // top=[10,20,30,40,50,60,70,80], left=[5,15,25,35,45,55,65,75]
        // Subblock (0,0): sum_top = 10+20+30+40=100, sum_left=5+15+25+35=80
        //   dc = (100+80+4)>>3 = 184>>3 = 23.
        // Subblock (4,0): top+right not leftmost → top-preferred.
        //   top samples p[4..=7,-1] = 50+60+70+80 = 260
        //   dc = (260+2)>>2 = 65
        // Subblock (0,4): leftmost non-top → left-preferred.
        //   left samples p[-1,4..=7] = 45+55+65+75 = 240
        //   dc = (240+2)>>2 = 60
        // Subblock (4,4): both available → use both.
        //   sum_top[4..=7] = 260, sum_left[4..=7] = 240
        //   dc = (260+240+4)>>3 = 504>>3 = 63
        let s = mk_chroma_420(
            vec![10, 20, 30, 40, 50, 60, 70, 80],
            vec![5, 15, 25, 35, 45, 55, 65, 75],
            0,
        );
        let mut out = vec![0i32; 64];
        predict_chroma(
            IntraChromaMode::Dc,
            &s,
            ChromaArrayType::Yuv420,
            8,
            &mut out,
        );
        // Top-left 4x4 (x=0..4, y=0..4) should all be 23.
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out[y * 8 + x], 23, "TL subblock at ({},{})", x, y);
            }
        }
        for y in 0..4 {
            for x in 4..8 {
                assert_eq!(out[y * 8 + x], 65, "TR subblock at ({},{})", x, y);
            }
        }
        for y in 4..8 {
            for x in 0..4 {
                assert_eq!(out[y * 8 + x], 60, "BL subblock at ({},{})", x, y);
            }
        }
        for y in 4..8 {
            for x in 4..8 {
                assert_eq!(out[y * 8 + x], 63, "BR subblock at ({},{})", x, y);
            }
        }
    }

    #[test]
    fn intra_chroma_dc_420_no_neighbours() {
        let mut s = mk_chroma_420(vec![0; 8], vec![0; 8], 0);
        s.availability = no_avail();
        let mut out = vec![0i32; 64];
        predict_chroma(
            IntraChromaMode::Dc,
            &s,
            ChromaArrayType::Yuv420,
            8,
            &mut out,
        );
        for v in out.iter() {
            assert_eq!(*v, 128);
        }
    }

    #[test]
    fn intra_chroma_vertical_420() {
        let s = mk_chroma_420(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![0; 8], 0);
        let mut out = vec![0i32; 64];
        predict_chroma(
            IntraChromaMode::Vertical,
            &s,
            ChromaArrayType::Yuv420,
            8,
            &mut out,
        );
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(out[y * 8 + x], s.top[x]);
            }
        }
    }

    #[test]
    fn intra_chroma_horizontal_420() {
        let s = mk_chroma_420(vec![0; 8], vec![10, 20, 30, 40, 50, 60, 70, 80], 0);
        let mut out = vec![0i32; 64];
        predict_chroma(
            IntraChromaMode::Horizontal,
            &s,
            ChromaArrayType::Yuv420,
            8,
            &mut out,
        );
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(out[y * 8 + x], s.left[y]);
            }
        }
    }

    #[test]
    fn intra_chroma_plane_420_constant() {
        let s = mk_chroma_420(vec![60; 8], vec![60; 8], 60);
        let mut out = vec![0i32; 64];
        predict_chroma(
            IntraChromaMode::Plane,
            &s,
            ChromaArrayType::Yuv420,
            8,
            &mut out,
        );
        for v in out.iter() {
            assert_eq!(*v, 60);
        }
    }

    #[test]
    fn intra_chroma_plane_420_hand() {
        // For 4:2:0: xCF=0, yCF=0, b coeff=34, c coeff=34.
        // H = sum_{x'=0..=3} (x'+1)*(p[4+x',-1]-p[2-x',-1])
        //   x'=0: 1*(top[4]-top[2])
        //   x'=1: 2*(top[5]-top[1])
        //   x'=2: 3*(top[6]-top[0])
        //   x'=3: 4*(top[7]-p[-1,-1])  — where p[2-x',-1]=p[-1,-1]=tl for x'=3.
        // top = [0,2,4,6,8,10,12,14], tl = 0.
        //   x'=0: 1*(8-4)=4
        //   x'=1: 2*(10-2)=16
        //   x'=2: 3*(12-0)=36
        //   x'=3: 4*(14-0)=56
        // H = 112
        // left mirrors top; V = 112.
        // a = 16 * (left[7] + top[7]) = 16*(14+14) = 448
        // b = (34*112 + 32) >> 6 = (3808+32)>>6 = 3840>>6 = 60
        // c = 60
        // pred[3,3] = Clip1((448 + 60*(3-3-0) + 60*(3-3-0) + 16) >> 5) = 464>>5 = 14
        let s = mk_chroma_420(
            vec![0, 2, 4, 6, 8, 10, 12, 14],
            vec![0, 2, 4, 6, 8, 10, 12, 14],
            0,
        );
        let mut out = vec![0i32; 64];
        predict_chroma(
            IntraChromaMode::Plane,
            &s,
            ChromaArrayType::Yuv420,
            8,
            &mut out,
        );
        assert_eq!(out[3 * 8 + 3], 14);
    }

    #[test]
    fn intra_chroma_vertical_422() {
        let s = SamplesChroma {
            top_left: 0,
            top: vec![1, 2, 3, 4, 5, 6, 7, 8],
            left: vec![0; 16],
            availability: all_avail(),
        };
        let mut out = vec![0i32; 128];
        predict_chroma(
            IntraChromaMode::Vertical,
            &s,
            ChromaArrayType::Yuv422,
            8,
            &mut out,
        );
        for y in 0..16 {
            for x in 0..8 {
                assert_eq!(out[y * 8 + x], s.top[x]);
            }
        }
    }

    #[test]
    fn intra_chroma_plane_422_constant() {
        let s = SamplesChroma {
            top_left: 60,
            top: vec![60; 8],
            left: vec![60; 16],
            availability: all_avail(),
        };
        let mut out = vec![0i32; 128];
        predict_chroma(
            IntraChromaMode::Plane,
            &s,
            ChromaArrayType::Yuv422,
            8,
            &mut out,
        );
        for v in out.iter() {
            assert_eq!(*v, 60);
        }
    }

    #[test]
    fn clip1_boundary() {
        assert_eq!(clip1(-1, 8), 0);
        assert_eq!(clip1(0, 8), 0);
        assert_eq!(clip1(255, 8), 255);
        assert_eq!(clip1(256, 8), 255);
        assert_eq!(clip1(1023, 10), 1023);
        assert_eq!(clip1(1024, 10), 1023);
    }
}
