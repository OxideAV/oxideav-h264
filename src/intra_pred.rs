//! Intra prediction — ITU-T H.264 §8.3.
//!
//! Four families:
//! * Intra_4×4 (§8.3.1) — 9 modes, per 4×4 luma sub-block.
//! * Intra_8×8 (§8.3.2, High Profile) — 9 modes over 8×8 luma blocks with
//!   reference-sample low-pass filtering per §8.3.2.2.2.
//! * Intra_16×16 (§8.3.3) — 4 modes (Vertical / Horizontal / DC / Plane).
//! * Intra chroma 8×8 (§8.3.4) — 4 modes (DC / Horizontal / Vertical / Plane).
//!
//! Functions take samples from neighbouring rows / columns and write the
//! predicted block. They never touch the bitstream — entropy / mode decode
//! happens in `mb.rs`.

// Indices into `top` for an Intra_4×4 prediction. The 4×4 block needs
// `top[0..=7]` plus `top_left` plus `left[0..=3]`.

/// Intra_4×4 prediction modes (§8.3.1.1, Table 8-2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
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
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Vertical),
            1 => Some(Self::Horizontal),
            2 => Some(Self::Dc),
            3 => Some(Self::DiagonalDownLeft),
            4 => Some(Self::DiagonalDownRight),
            5 => Some(Self::VerticalRight),
            6 => Some(Self::HorizontalDown),
            7 => Some(Self::VerticalLeft),
            8 => Some(Self::HorizontalUp),
            _ => None,
        }
    }
}

/// Intra_16×16 prediction modes (§8.3.3, Table 8-4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Intra16x16Mode {
    Vertical = 0,
    Horizontal = 1,
    Dc = 2,
    Plane = 3,
}

impl Intra16x16Mode {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Vertical),
            1 => Some(Self::Horizontal),
            2 => Some(Self::Dc),
            3 => Some(Self::Plane),
            _ => None,
        }
    }
}

/// Chroma intra prediction modes (§8.3.4, Table 8-5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum IntraChromaMode {
    Dc = 0,
    Horizontal = 1,
    Vertical = 2,
    Plane = 3,
}

impl IntraChromaMode {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Dc),
            1 => Some(Self::Horizontal),
            2 => Some(Self::Vertical),
            3 => Some(Self::Plane),
            _ => None,
        }
    }
}

/// Neighbour samples for an Intra_4×4 block (§8.3.1.2, Figure 8-7).
///
/// `top[0..=7]` = the row above the block (8 samples, used by diagonal
/// modes that consume the upper-right neighbour). When the block sits at
/// the right edge of a macroblock, the upper-right neighbour might be
/// unavailable — callers should populate `top[4..=7]` by replicating
/// `top[3]` per §8.3.1.2.1.
///
/// `left[0..=3]` = the column to the left.
/// `top_left` = the corner sample at (-1, -1).
#[derive(Clone, Debug)]
pub struct Intra4x4Neighbours {
    pub top: [u8; 8],
    pub left: [u8; 4],
    pub top_left: u8,
    pub top_available: bool,
    pub left_available: bool,
    pub top_left_available: bool,
    /// True if top[4..=7] are real (i.e. the block to the upper-right is
    /// available). When false, callers must replicate top[3] before calling.
    pub top_right_available: bool,
}

/// Predict a 4×4 luma block. Writes the prediction into `out` (raster).
pub fn predict_intra_4x4(out: &mut [u8; 16], mode: Intra4x4Mode, n: &Intra4x4Neighbours) {
    use Intra4x4Mode::*;
    match mode {
        Vertical => {
            // p[x,y] = top[x]
            for y in 0..4 {
                for x in 0..4 {
                    out[y * 4 + x] = n.top[x];
                }
            }
        }
        Horizontal => {
            for y in 0..4 {
                for x in 0..4 {
                    out[y * 4 + x] = n.left[y];
                }
            }
        }
        Dc => {
            let dc: u32 = match (n.top_available, n.left_available) {
                (true, true) => {
                    (n.top[0] as u32
                        + n.top[1] as u32
                        + n.top[2] as u32
                        + n.top[3] as u32
                        + n.left[0] as u32
                        + n.left[1] as u32
                        + n.left[2] as u32
                        + n.left[3] as u32
                        + 4)
                        >> 3
                }
                (true, false) => {
                    (n.top[0] as u32 + n.top[1] as u32 + n.top[2] as u32 + n.top[3] as u32 + 2) >> 2
                }
                (false, true) => {
                    (n.left[0] as u32 + n.left[1] as u32 + n.left[2] as u32 + n.left[3] as u32 + 2)
                        >> 2
                }
                (false, false) => 128,
            };
            let v = dc.min(255) as u8;
            for px in out.iter_mut() {
                *px = v;
            }
        }
        DiagonalDownLeft => {
            // §8.3.1.2.4. For (x,y) != (3,3):
            //   pred[x,y] = (top[x+y] + 2*top[x+y+1] + top[x+y+2] + 2) >> 2
            // For (x,y) == (3,3):
            //   pred[x,y] = (top[6] + 3*top[7] + 2) >> 2
            let t = &n.top;
            let mut z = [0u8; 8];
            for i in 0..6 {
                let a = t[i] as u32;
                let b = t[i + 1] as u32;
                let c = t[i + 2] as u32;
                z[i] = (((a + 2 * b + c + 2) >> 2).min(255)) as u8;
            }
            z[7] = (((t[6] as u32 + 3 * t[7] as u32 + 2) >> 2).min(255)) as u8;
            for y in 0..4 {
                for x in 0..4 {
                    let zi = if x == 3 && y == 3 { 7 } else { x + y };
                    out[y * 4 + x] = z[zi];
                }
            }
        }
        DiagonalDownRight => {
            // §8.3.1.2.5: needs top + top_left + left.
            // p[x,y] = filtered samples on a diagonal.
            // Build a unified "edge" array indexed -3..=3 around the corner.
            let mut e = [0u8; 9]; // -4..=4 for safety
                                  // e index 4 = top_left.
            e[4] = n.top_left;
            for i in 0..4 {
                e[5 + i] = n.top[i]; // e[5..=8] = top[0..=3]
                e[3 - i] = n.left[i]; // e[3..=0] = left[0..=3] reversed
            }
            // Filter triplets.
            let mut f = [0u8; 7]; // f[0..=6]
            for i in 0..7 {
                let a = e[i] as u32;
                let b = e[i + 1] as u32;
                let c = e[i + 2] as u32;
                f[i] = (((a + 2 * b + c + 2) >> 2).min(255)) as u8;
            }
            // p[x,y] = f[3 + x - y]
            for y in 0..4 {
                for x in 0..4 {
                    out[y * 4 + x] = f[(3 + x as i32 - y as i32) as usize];
                }
            }
        }
        VerticalRight => {
            // §8.3.1.2.6. e[] linearises the neighbour ring so the index
            // arithmetic from the spec maps to a single array: e[4]=p[-1,-1],
            // e[5..=8]=p[0..=3,-1], e[3..=0]=p[-1,0..=3]. Then zVR = 2x-y
            // selects one of the five cases below.
            let mut e = [0u8; 9];
            e[4] = n.top_left;
            for i in 0..4 {
                e[5 + i] = n.top[i];
                e[3 - i] = n.left[i];
            }
            for y in 0..4 {
                for x in 0..4 {
                    let zvr = 2 * x as i32 - y as i32;
                    let v;
                    if zvr == 0 || zvr == 2 || zvr == 4 || zvr == 6 {
                        let i0 = (4 + x as i32 - (y as i32 >> 1)) as usize;
                        v = ((e[i0] as u32 + e[i0 + 1] as u32 + 1) >> 1) as u8;
                    } else if zvr == 1 || zvr == 3 || zvr == 5 {
                        let i0 = (3 + x as i32 - (y as i32 >> 1)) as usize;
                        v = ((e[i0] as u32 + 2 * e[i0 + 1] as u32 + e[i0 + 2] as u32 + 2) >> 2)
                            as u8;
                    } else if zvr == -1 {
                        v = ((e[3] as u32 + 2 * e[4] as u32 + e[5] as u32 + 2) >> 2) as u8;
                    } else {
                        // zVR == -2 at (0,2) → uses left[1], left[0], tl.
                        // zVR == -3 at (0,3) → uses left[2], left[1], left[0].
                        // Spec: pred = (p[-1,y-2x-1] + 2*p[-1,y-2x-2] + p[-1,y-2x-3] + 2) >> 2.
                        let off = (-zvr - 1) as usize;
                        let i = 4 - off;
                        v = ((e[i - 1] as u32 + 2 * e[i] as u32 + e[i + 1] as u32 + 2) >> 2) as u8;
                    }
                    out[y * 4 + x] = v;
                }
            }
        }
        HorizontalDown => {
            // §8.3.1.2.7 — like VerticalRight but rotated.
            let mut e = [0u8; 9];
            e[4] = n.top_left;
            for i in 0..4 {
                e[5 + i] = n.top[i];
                e[3 - i] = n.left[i];
            }
            for y in 0..4 {
                for x in 0..4 {
                    let zhd = 2 * y as i32 - x as i32;
                    let v;
                    if zhd == 0 || zhd == 2 || zhd == 4 || zhd == 6 {
                        // even ≥ 0: take two from "left" side mirror
                        let i0 = (3 - (y as i32 - (x as i32 >> 1))) as usize;
                        v = ((e[i0] as u32 + e[i0 + 1] as u32 + 1) >> 1) as u8;
                    } else if zhd == 1 || zhd == 3 || zhd == 5 {
                        // i0 maps to p[-1, y-(x>>1)]; spec's 3-tap is
                        // (p[-1,y-(x>>1)-2] + 2*p[-1,y-(x>>1)-1] + p[-1,y-(x>>1)] + 2) >> 2,
                        // which in e[]-space is (e[i0+2] + 2*e[i0+1] + e[i0] + 2) >> 2.
                        let i0 = (3 - (y as i32 - (x as i32 >> 1))) as usize;
                        v = ((e[i0] as u32 + 2 * e[i0 + 1] as u32 + e[i0 + 2] as u32 + 2) >> 2)
                            as u8;
                    } else if zhd == -1 {
                        v = ((e[3] as u32 + 2 * e[4] as u32 + e[5] as u32 + 2) >> 2) as u8;
                    } else {
                        // zhd = -2 or -3.
                        let off = (-zhd - 1) as usize;
                        let i = 4 + off;
                        v = ((e[i - 1] as u32 + 2 * e[i] as u32 + e[i + 1] as u32 + 2) >> 2) as u8;
                    }
                    out[y * 4 + x] = v;
                }
            }
        }
        VerticalLeft => {
            // §8.3.1.2.8 — needs top and (often) top right.
            let t = &n.top;
            for y in 0..4 {
                for x in 0..4 {
                    let i = x + (y >> 1);
                    let v = if y % 2 == 0 {
                        ((t[i] as u32 + t[i + 1] as u32 + 1) >> 1) as u8
                    } else {
                        ((t[i] as u32 + 2 * t[i + 1] as u32 + t[i + 2] as u32 + 2) >> 2) as u8
                    };
                    out[y * 4 + x] = v;
                }
            }
        }
        HorizontalUp => {
            // §8.3.1.2.9 — needs left only.
            // zHU = x + 2y. For zHU >= 6 the spec replicates left[3];
            // the earlier impl wrongly averaged/filtered for zHU = 6 and 7.
            let l = &n.left;
            for y in 0..4 {
                for x in 0..4 {
                    let zhu = x + 2 * y;
                    let v = match zhu {
                        0 | 2 | 4 => {
                            let yy = y + (x >> 1);
                            ((l[yy] as u32 + l[yy + 1] as u32 + 1) >> 1) as u8
                        }
                        1 | 3 => {
                            let yy = y + (x >> 1);
                            ((l[yy] as u32 + 2 * l[yy + 1] as u32 + l[yy + 2] as u32 + 2) >> 2)
                                as u8
                        }
                        5 => ((l[2] as u32 + 3 * l[3] as u32 + 2) >> 2) as u8,
                        _ => l[3],
                    };
                    out[y * 4 + x] = v;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Intra_16×16 (§8.3.3)
// ---------------------------------------------------------------------------

/// Neighbours for a 16×16 block (luma macroblock).
#[derive(Clone, Debug)]
pub struct Intra16x16Neighbours {
    pub top: [u8; 16],
    pub left: [u8; 16],
    pub top_left: u8,
    pub top_available: bool,
    pub left_available: bool,
    pub top_left_available: bool,
}

pub fn predict_intra_16x16(out: &mut [u8; 256], mode: Intra16x16Mode, n: &Intra16x16Neighbours) {
    use Intra16x16Mode::*;
    // Substitute mode if required neighbour is unavailable (§8.7.2.2 chroma
    // analogue). Vertical without top → DC; Horizontal without left → DC.
    // Plane requires both — fallback to DC if either missing.
    let mode = match mode {
        Vertical if !n.top_available => Dc,
        Horizontal if !n.left_available => Dc,
        Plane if !n.top_available || !n.left_available || !n.top_left_available => Dc,
        m => m,
    };
    match mode {
        Vertical => {
            for y in 0..16 {
                for x in 0..16 {
                    out[y * 16 + x] = n.top[x];
                }
            }
        }
        Horizontal => {
            for y in 0..16 {
                for x in 0..16 {
                    out[y * 16 + x] = n.left[y];
                }
            }
        }
        Dc => {
            let dc: u32 = match (n.top_available, n.left_available) {
                (true, true) => {
                    let s: u32 = n.top.iter().map(|&v| v as u32).sum::<u32>()
                        + n.left.iter().map(|&v| v as u32).sum::<u32>();
                    (s + 16) >> 5
                }
                (true, false) => {
                    let s: u32 = n.top.iter().map(|&v| v as u32).sum();
                    (s + 8) >> 4
                }
                (false, true) => {
                    let s: u32 = n.left.iter().map(|&v| v as u32).sum();
                    (s + 8) >> 4
                }
                (false, false) => 128,
            };
            let v = dc.min(255) as u8;
            for px in out.iter_mut() {
                *px = v;
            }
        }
        Plane => {
            // §8.3.3.4
            // H = sum_{i=0..7} (i+1) * (top[8+i] - top[6-i])
            // V = sum_{i=0..7} (i+1) * (left[8+i] - left[6-i])
            // When i == 7, the `-i` term hits `top[-1]` / `left[-1]`, which is
            // the top-left corner sample stored separately.
            let top_at = |k: i32| -> i32 {
                if k < 0 {
                    n.top_left as i32
                } else {
                    n.top[k as usize] as i32
                }
            };
            let left_at = |k: i32| -> i32 {
                if k < 0 {
                    n.top_left as i32
                } else {
                    n.left[k as usize] as i32
                }
            };
            let mut h: i32 = 0;
            for i in 0..8i32 {
                h += (i + 1) * (top_at(8 + i) - top_at(6 - i));
            }
            let mut v: i32 = 0;
            for i in 0..8i32 {
                v += (i + 1) * (left_at(8 + i) - left_at(6 - i));
            }
            let b = (5 * h + 32) >> 6;
            let c = (5 * v + 32) >> 6;
            let a = 16 * (n.left[15] as i32 + n.top[15] as i32);
            for y in 0..16 {
                for x in 0..16 {
                    let p = (a + b * (x as i32 - 7) + c * (y as i32 - 7) + 16) >> 5;
                    out[y * 16 + x] = p.clamp(0, 255) as u8;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Intra chroma 8×8 (§8.3.4)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct IntraChromaNeighbours {
    pub top: [u8; 8],
    pub left: [u8; 8],
    pub top_left: u8,
    pub top_available: bool,
    pub left_available: bool,
    pub top_left_available: bool,
}

pub fn predict_intra_chroma(out: &mut [u8; 64], mode: IntraChromaMode, n: &IntraChromaNeighbours) {
    use IntraChromaMode::*;
    let mode = match mode {
        Vertical if !n.top_available => Dc,
        Horizontal if !n.left_available => Dc,
        Plane if !n.top_available || !n.left_available || !n.top_left_available => Dc,
        m => m,
    };
    match mode {
        Vertical => {
            for y in 0..8 {
                for x in 0..8 {
                    out[y * 8 + x] = n.top[x];
                }
            }
        }
        Horizontal => {
            for y in 0..8 {
                for x in 0..8 {
                    out[y * 8 + x] = n.left[y];
                }
            }
        }
        Dc => {
            // Each 4x4 chroma quadrant gets its own DC value (§8.3.4.2).
            // Quadrant (0,0): use top[0..=3] AND left[0..=3] when both available;
            //                 else fall back to whichever is available (or 128).
            // Quadrant (1,0): right-top -> use top[4..=7] only (not left!).
            // Quadrant (0,1): left-bottom -> use left[4..=7] only.
            // Quadrant (1,1): right-bottom -> top[4..=7] AND left[4..=7] when both available.
            let avg = |samples: &[u8]| -> u32 {
                let s: u32 = samples.iter().map(|&v| v as u32).sum();
                (s + (samples.len() as u32 / 2)) / samples.len() as u32
            };
            let dc = |topa: bool, lefta: bool, tslice: &[u8], lslice: &[u8]| -> u8 {
                let val = match (topa, lefta) {
                    (true, true) => {
                        let s: u32 = tslice.iter().map(|&v| v as u32).sum::<u32>()
                            + lslice.iter().map(|&v| v as u32).sum::<u32>();
                        (s + 4) >> 3
                    }
                    (true, false) => avg(tslice),
                    (false, true) => avg(lslice),
                    (false, false) => 128,
                };
                val.min(255) as u8
            };
            // Quadrant 0 (top-left 4x4)
            let q00 = dc(
                n.top_available,
                n.left_available,
                &n.top[0..4],
                &n.left[0..4],
            );
            // Quadrant 1 (top-right 4x4): only top[4..=7]
            let q10 = if n.top_available {
                let s: u32 = n.top[4..8].iter().map(|&v| v as u32).sum();
                ((s + 2) >> 2).min(255) as u8
            } else if n.left_available {
                let s: u32 = n.left[0..4].iter().map(|&v| v as u32).sum();
                ((s + 2) >> 2).min(255) as u8
            } else {
                128
            };
            // Quadrant 2 (bottom-left 4x4): only left[4..=7]
            let q01 = if n.left_available {
                let s: u32 = n.left[4..8].iter().map(|&v| v as u32).sum();
                ((s + 2) >> 2).min(255) as u8
            } else if n.top_available {
                let s: u32 = n.top[0..4].iter().map(|&v| v as u32).sum();
                ((s + 2) >> 2).min(255) as u8
            } else {
                128
            };
            // Quadrant 3 (bottom-right 4x4)
            let q11 = dc(
                n.top_available,
                n.left_available,
                &n.top[4..8],
                &n.left[4..8],
            );
            // Fill quadrants.
            for y in 0..8 {
                for x in 0..8 {
                    let v = match (x / 4, y / 4) {
                        (0, 0) => q00,
                        (1, 0) => q10,
                        (0, 1) => q01,
                        (1, 1) => q11,
                        _ => unreachable!(),
                    };
                    out[y * 8 + x] = v;
                }
            }
        }
        Plane => {
            // §8.3.4.4. The sums reach index `-1` of the top/left rows,
            // which is the top-left corner sample (stored separately).
            let top_at = |k: i32| -> i32 {
                if k < 0 {
                    n.top_left as i32
                } else {
                    n.top[k as usize] as i32
                }
            };
            let left_at = |k: i32| -> i32 {
                if k < 0 {
                    n.top_left as i32
                } else {
                    n.left[k as usize] as i32
                }
            };
            let mut h: i32 = 0;
            for i in 0..4i32 {
                h += (i + 1) * (top_at(4 + i) - top_at(2 - i));
            }
            let mut v: i32 = 0;
            for i in 0..4i32 {
                v += (i + 1) * (left_at(4 + i) - left_at(2 - i));
            }
            let b = (34 * h + 32) >> 6;
            let c = (34 * v + 32) >> 6;
            let a = 16 * (n.left[7] as i32 + n.top[7] as i32);
            for y in 0..8 {
                for x in 0..8 {
                    let p = (a + b * (x as i32 - 3) + c * (y as i32 - 3) + 16) >> 5;
                    out[y * 8 + x] = p.clamp(0, 255) as u8;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Intra_8×8 (§8.3.2) — High Profile.
// ---------------------------------------------------------------------------

/// Intra_8×8 prediction modes (§8.3.2.1, Table 8-3). Same numeric values as
/// [`Intra4x4Mode`] but applied over an 8×8 block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Intra8x8Mode {
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

impl Intra8x8Mode {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Vertical),
            1 => Some(Self::Horizontal),
            2 => Some(Self::Dc),
            3 => Some(Self::DiagonalDownLeft),
            4 => Some(Self::DiagonalDownRight),
            5 => Some(Self::VerticalRight),
            6 => Some(Self::HorizontalDown),
            7 => Some(Self::VerticalLeft),
            8 => Some(Self::HorizontalUp),
            _ => None,
        }
    }
}

/// Neighbour samples needed by an Intra_8×8 prediction (§8.3.2.2.1). The
/// prediction consumes:
///
/// * `top[0..=15]`   — the 16 samples of the row above (top + top-right).
/// * `left[0..=7]`   — the 8 samples of the column to the left.
/// * `top_left`      — the corner sample at (-1, -1).
///
/// The `top_right_available` flag mirrors the Intra_4×4 case: when false,
/// `top[8..=15]` must be replicated from `top[7]` by the caller per
/// §8.3.2.2.1 step 3.
#[derive(Clone, Debug)]
pub struct Intra8x8Neighbours {
    pub top: [u8; 16],
    pub left: [u8; 8],
    pub top_left: u8,
    pub top_available: bool,
    pub left_available: bool,
    pub top_left_available: bool,
    pub top_right_available: bool,
}

/// Reference-sample low-pass filtering per §8.3.2.2.2. Produces the
/// filtered references used by every Intra_8×8 prediction mode. Returns
/// `(filt_top[0..=15], filt_left[0..=7], filt_top_left)` — same layout as
/// the input neighbour struct but with the (1,2,1) filter applied.
///
/// The three corner rules from the spec:
///
/// * `p'[-1,-1] = (p[0,-1] + 2·p[-1,-1] + p[-1,0] + 2) >> 2` when both top
///   and left are available.
/// * `p'[-1,-1] = (3·p[-1,-1] + p[0,-1] + 2) >> 2` when only top available.
/// * `p'[-1,-1] = (3·p[-1,-1] + p[-1,0] + 2) >> 2` when only left available.
///
/// The two last-sample rules:
///
/// * `p'[15,-1] = (p[14,-1] + 3·p[15,-1] + 2) >> 2` (top row's tail).
/// * `p'[-1,7]  = (p[-1,6] + 3·p[-1,7] + 2) >> 2`   (left column's tail).
fn filter_ref_8x8(n: &Intra8x8Neighbours) -> ([u8; 16], [u8; 8], u8) {
    let mut ft = [0u8; 16];
    let mut fl = [0u8; 8];
    let ftl;

    // Top row filtering: uses top_left + top[0..=15] as input samples.
    if n.top_available {
        // Position 0 uses top_left on the low side.
        let tl_src = if n.top_left_available {
            n.top_left
        } else {
            n.top[0]
        };
        ft[0] = ((tl_src as u32 + 2 * n.top[0] as u32 + n.top[1] as u32 + 2) >> 2) as u8;
        for i in 1..15 {
            ft[i] =
                ((n.top[i - 1] as u32 + 2 * n.top[i] as u32 + n.top[i + 1] as u32 + 2) >> 2) as u8;
        }
        ft[15] = ((n.top[14] as u32 + 3 * n.top[15] as u32 + 2) >> 2) as u8;
    }

    // Left column filtering: uses top_left + left[0..=7].
    if n.left_available {
        let tl_src = if n.top_left_available {
            n.top_left
        } else {
            n.left[0]
        };
        fl[0] = ((tl_src as u32 + 2 * n.left[0] as u32 + n.left[1] as u32 + 2) >> 2) as u8;
        for i in 1..7 {
            fl[i] = ((n.left[i - 1] as u32 + 2 * n.left[i] as u32 + n.left[i + 1] as u32 + 2) >> 2)
                as u8;
        }
        fl[7] = ((n.left[6] as u32 + 3 * n.left[7] as u32 + 2) >> 2) as u8;
    }

    // Top-left corner filtering.
    if n.top_left_available {
        ftl = match (n.top_available, n.left_available) {
            (true, true) => {
                ((n.top[0] as u32 + 2 * n.top_left as u32 + n.left[0] as u32 + 2) >> 2) as u8
            }
            (true, false) => ((3 * n.top_left as u32 + n.top[0] as u32 + 2) >> 2) as u8,
            (false, true) => ((3 * n.top_left as u32 + n.left[0] as u32 + 2) >> 2) as u8,
            (false, false) => n.top_left,
        };
    } else {
        ftl = 0;
    }

    (ft, fl, ftl)
}

/// Predict an 8×8 luma block. Writes the prediction into `out` (raster
/// layout `row * 8 + col`). Callers must already have populated
/// `n.top[8..=15]` with replicated `top[7]` samples when
/// `top_right_available == false` (same convention as Intra_4×4).
pub fn predict_intra_8x8(out: &mut [u8; 64], mode: Intra8x8Mode, n: &Intra8x8Neighbours) {
    use Intra8x8Mode::*;
    // Mode substitution follows §8.3.2.1 — a mode whose required neighbour
    // is unavailable would be an invalid bitstream; but to be robust we fall
    // back to DC so we never read out-of-range samples.
    let mode = match mode {
        Vertical if !n.top_available => Dc,
        Horizontal if !n.left_available => Dc,
        DiagonalDownLeft | VerticalLeft if !n.top_available => Dc,
        DiagonalDownRight | VerticalRight | HorizontalDown
            if !n.top_available || !n.left_available || !n.top_left_available =>
        {
            Dc
        }
        HorizontalUp if !n.left_available => Dc,
        m => m,
    };

    let (ft, fl, ftl) = filter_ref_8x8(n);

    // Helper — replicate top[7] into top[8..=15] when caller didn't. The
    // filtering above consumed them so the helper isn't needed for ft[].

    match mode {
        Vertical => {
            // p[x, y] = ft[x], x=0..=7, y=0..=7.
            for y in 0..8 {
                for x in 0..8 {
                    out[y * 8 + x] = ft[x];
                }
            }
        }
        Horizontal => {
            for y in 0..8 {
                for x in 0..8 {
                    out[y * 8 + x] = fl[y];
                }
            }
        }
        Dc => {
            let dc: u32 = match (n.top_available, n.left_available) {
                (true, true) => {
                    (ft[0..8].iter().map(|&v| v as u32).sum::<u32>()
                        + fl.iter().map(|&v| v as u32).sum::<u32>()
                        + 8)
                        >> 4
                }
                (true, false) => (ft[0..8].iter().map(|&v| v as u32).sum::<u32>() + 4) >> 3,
                (false, true) => (fl.iter().map(|&v| v as u32).sum::<u32>() + 4) >> 3,
                (false, false) => 128,
            };
            let v = dc.min(255) as u8;
            for px in out.iter_mut() {
                *px = v;
            }
        }
        DiagonalDownLeft => {
            // §8.3.2.2.4 — uses ft[0..=15] only.
            for y in 0..8 {
                for x in 0..8 {
                    let v = if x == 7 && y == 7 {
                        ((ft[14] as u32 + 3 * ft[15] as u32 + 2) >> 2) as u8
                    } else {
                        ((ft[x + y] as u32 + 2 * ft[x + y + 1] as u32 + ft[x + y + 2] as u32 + 2)
                            >> 2) as u8
                    };
                    out[y * 8 + x] = v;
                }
            }
        }
        DiagonalDownRight => {
            // §8.3.2.2.5 — needs ft, fl, ftl. Shares the 17-entry `e[]`
            // layout with HD/VR so the k=-7 case at (x=0,y=7) can reach
            // fl[7] via e[1] (spec taps p[-1,6], p[-1,7], ...). For k=x-y
            // in -7..=7, the spec 3-tap is centred at e[8+k].
            //
            // Layout:
            //   e[0..=7] = fl[7..=0] reversed (so fl[7]=e[0], fl[0]=e[7])
            //   e[8]     = ftl
            //   e[9..=16] = ft[0..=7]
            let mut e = [0u8; 17];
            for i in 0..8 {
                e[i] = fl[7 - i];
            }
            e[8] = ftl;
            for i in 0..8 {
                e[9 + i] = ft[i];
            }
            for y in 0..8 {
                for x in 0..8 {
                    let k = x as i32 - y as i32;
                    let idx = (8 + k) as usize;
                    let l = e[idx - 1] as u32;
                    let m = e[idx] as u32;
                    let r = e[idx + 1] as u32;
                    out[y * 8 + x] = ((l + 2 * m + r + 2) >> 2) as u8;
                }
            }
        }
        VerticalRight => {
            // §8.3.2.2.6. Linearised reference ring `e[0..=16]`:
            // * `e[0..=7]`  = filtered left column reversed (`fl[7..=0]`).
            // * `e[8]`      = filtered top-left.
            // * `e[9..=16]` = filtered top row (`ft[0..=7]`).
            // `fl[7]` must live in `e[]` so the HorizontalDown sibling at
            // zhd=14 can reach it; we keep a single shared layout so the
            // two mirror-symmetric modes share index arithmetic.
            let mut e = [0u8; 17];
            for i in 0..8 {
                e[i] = fl[7 - i];
            }
            e[8] = ftl;
            for i in 0..8 {
                e[9 + i] = ft[i];
            }
            for y in 0..8 {
                for x in 0..8 {
                    let zvr = 2 * x as i32 - y as i32;
                    let v = if zvr >= 0 && zvr % 2 == 0 {
                        let base = 8 + x as i32 - (y as i32 >> 1);
                        let i = base as usize;
                        ((e[i] as u32 + e[i + 1] as u32 + 1) >> 1) as u8
                    } else if zvr >= 1 {
                        let base = 8 + x as i32 - ((y as i32 + 1) >> 1);
                        let i = base as usize;
                        ((e[i] as u32 + 2 * e[i + 1] as u32 + e[i + 2] as u32 + 2) >> 2) as u8
                    } else {
                        // zvr ∈ {-1..=-7} — left-column 3-tap around e[i+1].
                        let k = -zvr - 1;
                        let i = (7 - k) as usize;
                        ((e[i] as u32 + 2 * e[i + 1] as u32 + e[i + 2] as u32 + 2) >> 2) as u8
                    };
                    out[y * 8 + x] = v;
                }
            }
        }
        HorizontalDown => {
            // §8.3.2.2.7 — mirror of VerticalRight. Shares the 17-entry
            // `e[]` layout so the zhd=14 case at (x=0,y=7) can reach fl[7],
            // which was out-of-range in the prior 16-entry layout.
            //
            // Indexing matches FFmpeg's pred8x8l_horizontal_down reference:
            // * even zhd∈{0..14}: 2-tap `(e[i-1]+e[i]+1)>>1`.
            // * odd  zhd∈{1..13}: 3-tap `(e[i+1]+2·e[i]+e[i-1]+2)>>2`.
            // * zhd=-1 merges into the top-row 3-tap centred at ftl.
            // * zhd∈{-2..=-7}: 3-tap along the top row.
            let mut e = [0u8; 17];
            for i in 0..8 {
                e[i] = fl[7 - i];
            }
            e[8] = ftl;
            for i in 0..8 {
                e[9 + i] = ft[i];
            }
            for y in 0..8 {
                for x in 0..8 {
                    let zhd = 2 * y as i32 - x as i32;
                    let v = if zhd >= 0 && zhd % 2 == 0 {
                        let base = 8 - (y as i32 - (x as i32 >> 1));
                        let i = base as usize;
                        ((e[i] as u32 + e[i - 1] as u32 + 1) >> 1) as u8
                    } else if zhd >= 1 {
                        let base = 8 - (y as i32 - (x as i32 >> 1));
                        let i = base as usize;
                        ((e[i + 1] as u32 + 2 * e[i] as u32 + e[i - 1] as u32 + 2) >> 2) as u8
                    } else {
                        // zhd ∈ {-1..=-7} — top-row 3-tap centred at
                        // e[i] (zhd=-1 centre is ftl, zhd=-k centre is ft[k-2]).
                        let i = (7 - zhd) as usize;
                        ((e[i - 1] as u32 + 2 * e[i] as u32 + e[i + 1] as u32 + 2) >> 2) as u8
                    };
                    out[y * 8 + x] = v;
                }
            }
        }
        VerticalLeft => {
            // §8.3.2.2.8 — needs ft (with top-right replication) only.
            for y in 0..8 {
                for x in 0..8 {
                    let i = x + (y >> 1);
                    let v = if y % 2 == 0 {
                        ((ft[i] as u32 + ft[i + 1] as u32 + 1) >> 1) as u8
                    } else {
                        ((ft[i] as u32 + 2 * ft[i + 1] as u32 + ft[i + 2] as u32 + 2) >> 2) as u8
                    };
                    out[y * 8 + x] = v;
                }
            }
        }
        HorizontalUp => {
            // §8.3.2.2.9 — needs fl only.
            for y in 0..8 {
                for x in 0..8 {
                    let zhu = x + 2 * y;
                    let v = if zhu <= 12 && zhu % 2 == 0 {
                        let i = y + (x >> 1);
                        ((fl[i] as u32 + fl[i + 1] as u32 + 1) >> 1) as u8
                    } else if zhu <= 11 {
                        let i = y + (x >> 1);
                        ((fl[i] as u32 + 2 * fl[i + 1] as u32 + fl[i + 2] as u32 + 2) >> 2) as u8
                    } else if zhu == 13 {
                        ((fl[6] as u32 + 3 * fl[7] as u32 + 2) >> 2) as u8
                    } else {
                        fl[7]
                    };
                    out[y * 8 + x] = v;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intra4x4_vertical() {
        let n = Intra4x4Neighbours {
            top: [10, 20, 30, 40, 0, 0, 0, 0],
            left: [0; 4],
            top_left: 0,
            top_available: true,
            left_available: false,
            top_left_available: false,
            top_right_available: false,
        };
        let mut out = [0u8; 16];
        predict_intra_4x4(&mut out, Intra4x4Mode::Vertical, &n);
        for y in 0..4 {
            assert_eq!(&out[y * 4..y * 4 + 4], &[10u8, 20, 30, 40]);
        }
    }

    #[test]
    fn intra16x16_dc_neither_available() {
        let n = Intra16x16Neighbours {
            top: [0; 16],
            left: [0; 16],
            top_left: 0,
            top_available: false,
            left_available: false,
            top_left_available: false,
        };
        let mut out = [0u8; 256];
        predict_intra_16x16(&mut out, Intra16x16Mode::Dc, &n);
        assert!(out.iter().all(|&v| v == 128));
    }

    #[test]
    fn intra8x8_dc_neither_available() {
        let n = Intra8x8Neighbours {
            top: [0; 16],
            left: [0; 8],
            top_left: 0,
            top_available: false,
            left_available: false,
            top_left_available: false,
            top_right_available: false,
        };
        let mut out = [0u8; 64];
        predict_intra_8x8(&mut out, Intra8x8Mode::Dc, &n);
        assert!(out.iter().all(|&v| v == 128));
    }

    #[test]
    fn intra8x8_vertical_constant_top() {
        // A constant top row must yield a constant prediction (the (1,2,1)
        // reference filter preserves a flat signal modulo the corner rules).
        let mut top = [100u8; 16];
        // Ensure the filter's corner rules don't perturb a uniform top row.
        // With top_left = 100, left_available = true, the filter is
        // (p[-1,-1] + 2*p[0,-1] + p[1,-1] + 2) >> 2 = (100 + 200 + 100 + 2) >> 2 = 100.
        // Also position 15: (p[14,-1] + 3*p[15,-1] + 2) >> 2 = (100 + 300 + 2) >> 2 = 100.
        top[15] = 100;
        let n = Intra8x8Neighbours {
            top,
            left: [100; 8],
            top_left: 100,
            top_available: true,
            left_available: true,
            top_left_available: true,
            top_right_available: true,
        };
        let mut out = [0u8; 64];
        predict_intra_8x8(&mut out, Intra8x8Mode::Vertical, &n);
        assert!(out.iter().all(|&v| v == 100));
    }

    #[test]
    fn intra8x8_horizontal_constant_left() {
        let n = Intra8x8Neighbours {
            top: [50; 16],
            left: [50; 8],
            top_left: 50,
            top_available: true,
            left_available: true,
            top_left_available: true,
            top_right_available: true,
        };
        let mut out = [0u8; 64];
        predict_intra_8x8(&mut out, Intra8x8Mode::Horizontal, &n);
        assert!(out.iter().all(|&v| v == 50));
    }

    #[test]
    fn intra4x4_diagonal_down_left_spec_pred_0_0() {
        // §8.3.1.2.4: pred[0,0] = (top[0] + 2*top[1] + top[2] + 2) >> 2.
        // Earlier impl computed (3*top[0] + top[1] + 2) >> 2 which differed
        // on non-uniform top rows.
        let n = Intra4x4Neighbours {
            top: [10, 100, 200, 40, 40, 40, 40, 40],
            left: [0; 4],
            top_left: 0,
            top_available: true,
            left_available: false,
            top_left_available: false,
            top_right_available: true,
        };
        let mut out = [0u8; 16];
        predict_intra_4x4(&mut out, Intra4x4Mode::DiagonalDownLeft, &n);
        // (10 + 200 + 200 + 2) >> 2 = 412 >> 2 = 103.
        assert_eq!(out[0], 103);
    }

    #[test]
    fn intra4x4_horizontal_up_replicates_bottom_for_high_zhu() {
        // §8.3.1.2.9: for zHU >= 6 pred = p[-1,3]. Earlier impl used
        // averaging / 3-tap filters for zHU == 6 and 7 which deviated
        // from the spec on non-uniform left columns.
        let n = Intra4x4Neighbours {
            top: [0; 8],
            left: [10, 20, 30, 200],
            top_left: 0,
            top_available: false,
            left_available: true,
            top_left_available: false,
            top_right_available: false,
        };
        let mut out = [0u8; 16];
        predict_intra_4x4(&mut out, Intra4x4Mode::HorizontalUp, &n);
        // (x=0, y=3) zHU=6 → must be left[3] = 200.
        assert_eq!(out[3 * 4 + 0], 200);
        // (x=1, y=3) zHU=7 → must also be left[3] = 200.
        assert_eq!(out[3 * 4 + 1], 200);
        // (x=2, y=3) zHU=8, (x=3, y=3) zHU=9 → left[3] too.
        assert_eq!(out[3 * 4 + 2], 200);
        assert_eq!(out[3 * 4 + 3], 200);
    }

    #[test]
    fn intra8x8_horizontal_down_zhd14_reaches_fl7() {
        // §8.3.2.2.7 at (x=0,y=7), zhd=14: pred = (fl[6] + fl[7] + 1) >> 1.
        // Before the e[] resize fl[7] was out of scope so this would either
        // panic or return garbage. Uniform left gives 200.
        let n = Intra8x8Neighbours {
            top: [5; 16],
            left: [200; 8],
            top_left: 200,
            top_available: true,
            left_available: true,
            top_left_available: true,
            top_right_available: true,
        };
        let mut out = [0u8; 64];
        predict_intra_8x8(&mut out, Intra8x8Mode::HorizontalDown, &n);
        assert_eq!(out[7 * 8 + 0], 200, "(x=0,y=7) must be (fl[6]+fl[7]+1)/2");
    }

    #[test]
    fn intra8x8_horizontal_down_zhd_minus1_ffmpeg() {
        // FFmpeg SRC(1,0) for pred8x8l_horizontal_down:
        //   (l0 + 2*lt + t0 + 2) >> 2  (zhd=-1 corner-crossing 3-tap).
        // Prior impl used a top-row 3-tap here; this test pins the fix.
        let n = Intra8x8Neighbours {
            top: [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80],
            left: [100, 100, 100, 100, 100, 100, 100, 100],
            top_left: 40,
            top_available: true,
            left_available: true,
            top_left_available: true,
            top_right_available: true,
        };
        let mut out = [0u8; 64];
        predict_intra_8x8(&mut out, Intra8x8Mode::HorizontalDown, &n);
        // Filtered samples: fl[0] = (tl + 2*l[0] + l[1] + 2)>>2 = (40+200+100+2)/4 = 85.
        // ftl = (t[0] + 2*tl + l[0] + 2)>>2 = (80+80+100+2)/4 = 65.
        // ft[0] = (tl + 2*t[0] + t[1] + 2)>>2 = (40+160+80+2)/4 = 70.
        // SRC(1,0) = (fl[0] + 2*ftl + ft[0] + 2)>>2 = (85+130+70+2)/4 = 71.
        assert_eq!(out[0 * 8 + 1], 71, "(x=1,y=0) zhd=-1 corner-crossing 3-tap");
    }

    #[test]
    fn intra8x8_vertical_right_zvr14_reaches_ft7() {
        // §8.3.2.2.6 at (x=7, y=0), zvr=14: pred = (ft[6] + ft[7] + 1) >> 1.
        // Prior impl's range-check excluded zvr=14 from the even branch,
        // pushing it into the top-row 3-tap and then OOB-indexing.
        let n = Intra8x8Neighbours {
            top: [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
            left: [5; 8],
            top_left: 5,
            top_available: true,
            left_available: true,
            top_left_available: true,
            top_right_available: true,
        };
        let mut out = [0u8; 64];
        predict_intra_8x8(&mut out, Intra8x8Mode::VerticalRight, &n);
        // Constant top row → every filtered ft[] is 20.
        assert_eq!(out[0 * 8 + 7], 20, "(x=7,y=0) must be (ft[6]+ft[7]+1)/2");
    }

    #[test]
    fn intra4x4_vertical_right_low_left_tail_uses_spec_indices() {
        // §8.3.1.2.6: zVR=-2 at (0,2) uses p[-1,1], p[-1,0], p[-1,-1].
        // Earlier impl used p[-1,2], p[-1,1], p[-1,0] (off-by-one),
        // which diverged when top_left !≈ left[2].
        let n = Intra4x4Neighbours {
            top: [50; 8],
            left: [10, 20, 30, 40],
            top_left: 200,
            top_available: true,
            left_available: true,
            top_left_available: true,
            top_right_available: true,
        };
        let mut out = [0u8; 16];
        predict_intra_4x4(&mut out, Intra4x4Mode::VerticalRight, &n);
        // (x=0,y=2) zVR=-2: (left[1] + 2*left[0] + tl + 2) >> 2
        //                 = (20 + 20 + 200 + 2) >> 2 = 242 >> 2 = 60.
        assert_eq!(out[2 * 4 + 0], 60);
        // (x=0,y=3) zVR=-3: (left[2] + 2*left[1] + left[0] + 2) >> 2
        //                 = (30 + 40 + 10 + 2) >> 2 = 82 >> 2 = 20.
        assert_eq!(out[3 * 4 + 0], 20);
    }
}
