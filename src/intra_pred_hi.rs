//! High-bit-depth intra prediction — ITU-T H.264 §8.3, u16 variants.
//!
//! Mirrors [`crate::intra_pred`] with three differences that follow
//! from the bit-depth change:
//!
//! * Sample storage is `u16` per the spec's `Clip1Y(x) = Clip3(0,
//!   (1 << BitDepthY) - 1, x)` (§7.4.2.1).
//! * The DC "no neighbour" fallback returns `1 << (bit_depth - 1)`
//!   (§8.3.1.2.3 / §8.3.3.3 / §8.3.4.2 / §8.3.2.2.3) rather than 128.
//! * Final plane-mode clamping honours `(1 << bit_depth) - 1`.
//!
//! Intermediate arithmetic is widened to `i32` / `u32`. The 3-tap
//! filter / 2-tap averaging coefficients are identical to the 8-bit
//! module — only the input/output types change.

use crate::intra_pred::{
    Intra16x16Mode, Intra4x4Mode, Intra8x8Mode, IntraChromaMode,
};

#[derive(Clone, Debug)]
pub struct Intra4x4Neighbours16 {
    pub top: [u16; 8],
    pub left: [u16; 4],
    pub top_left: u16,
    pub top_available: bool,
    pub left_available: bool,
    pub top_left_available: bool,
    pub top_right_available: bool,
}

#[derive(Clone, Debug)]
pub struct Intra16x16Neighbours16 {
    pub top: [u16; 16],
    pub left: [u16; 16],
    pub top_left: u16,
    pub top_available: bool,
    pub left_available: bool,
    pub top_left_available: bool,
}

#[derive(Clone, Debug)]
pub struct IntraChromaNeighbours16 {
    pub top: [u16; 8],
    pub left: [u16; 8],
    pub top_left: u16,
    pub top_available: bool,
    pub left_available: bool,
    pub top_left_available: bool,
}

#[derive(Clone, Debug)]
pub struct Intra8x8Neighbours16 {
    pub top: [u16; 16],
    pub left: [u16; 8],
    pub top_left: u16,
    pub top_available: bool,
    pub left_available: bool,
    pub top_left_available: bool,
    pub top_right_available: bool,
}

#[inline]
fn clip1(v: i32, max_sample: u16) -> u16 {
    v.clamp(0, max_sample as i32) as u16
}

pub fn predict_intra_4x4(
    out: &mut [u16; 16],
    mode: Intra4x4Mode,
    n: &Intra4x4Neighbours16,
    bit_depth: u8,
) {
    use Intra4x4Mode::*;
    let max_sample: u16 = (1u32 << bit_depth) as u16 - 1;
    let mid: u32 = 1u32 << (bit_depth - 1);
    match mode {
        Vertical => {
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
                (false, false) => mid,
            };
            let v = clip1(dc as i32, max_sample);
            for px in out.iter_mut() {
                *px = v;
            }
        }
        DiagonalDownLeft => {
            let t = &n.top;
            let mut z = [0u16; 8];
            for i in 0..6 {
                let a = t[i] as u32;
                let b = t[i + 1] as u32;
                let c = t[i + 2] as u32;
                z[i] = clip1(((a + 2 * b + c + 2) >> 2) as i32, max_sample);
            }
            z[7] = clip1(((t[6] as u32 + 3 * t[7] as u32 + 2) >> 2) as i32, max_sample);
            for y in 0..4 {
                for x in 0..4 {
                    let zi = if x == 3 && y == 3 { 7 } else { x + y };
                    out[y * 4 + x] = z[zi];
                }
            }
        }
        DiagonalDownRight => {
            let mut e = [0u16; 9];
            e[4] = n.top_left;
            for i in 0..4 {
                e[5 + i] = n.top[i];
                e[3 - i] = n.left[i];
            }
            let mut f = [0u16; 7];
            for i in 0..7 {
                let a = e[i] as u32;
                let b = e[i + 1] as u32;
                let c = e[i + 2] as u32;
                f[i] = clip1(((a + 2 * b + c + 2) >> 2) as i32, max_sample);
            }
            for y in 0..4 {
                for x in 0..4 {
                    out[y * 4 + x] = f[(3 + x as i32 - y as i32) as usize];
                }
            }
        }
        VerticalRight => {
            let mut e = [0u16; 9];
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
                        v = clip1(((e[i0] as u32 + e[i0 + 1] as u32 + 1) >> 1) as i32, max_sample);
                    } else if zvr == 1 || zvr == 3 || zvr == 5 {
                        let i0 = (3 + x as i32 - (y as i32 >> 1)) as usize;
                        v = clip1(
                            ((e[i0] as u32 + 2 * e[i0 + 1] as u32 + e[i0 + 2] as u32 + 2) >> 2) as i32,
                            max_sample,
                        );
                    } else if zvr == -1 {
                        v = clip1(
                            ((e[3] as u32 + 2 * e[4] as u32 + e[5] as u32 + 2) >> 2) as i32,
                            max_sample,
                        );
                    } else {
                        let off = (-zvr - 1) as usize;
                        let i = 4 - off;
                        v = clip1(
                            ((e[i - 1] as u32 + 2 * e[i] as u32 + e[i + 1] as u32 + 2) >> 2) as i32,
                            max_sample,
                        );
                    }
                    out[y * 4 + x] = v;
                }
            }
        }
        HorizontalDown => {
            let mut e = [0u16; 9];
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
                        let i0 = (3 - (y as i32 - (x as i32 >> 1))) as usize;
                        v = clip1(((e[i0] as u32 + e[i0 + 1] as u32 + 1) >> 1) as i32, max_sample);
                    } else if zhd == 1 || zhd == 3 || zhd == 5 {
                        let i0 = (3 - (y as i32 - (x as i32 >> 1))) as usize;
                        v = clip1(
                            ((e[i0] as u32 + 2 * e[i0 + 1] as u32 + e[i0 + 2] as u32 + 2) >> 2) as i32,
                            max_sample,
                        );
                    } else if zhd == -1 {
                        v = clip1(
                            ((e[3] as u32 + 2 * e[4] as u32 + e[5] as u32 + 2) >> 2) as i32,
                            max_sample,
                        );
                    } else {
                        let off = (-zhd - 1) as usize;
                        let i = 4 + off;
                        v = clip1(
                            ((e[i - 1] as u32 + 2 * e[i] as u32 + e[i + 1] as u32 + 2) >> 2) as i32,
                            max_sample,
                        );
                    }
                    out[y * 4 + x] = v;
                }
            }
        }
        VerticalLeft => {
            let t = &n.top;
            for y in 0..4 {
                for x in 0..4 {
                    let i = x + (y >> 1);
                    let v = if y % 2 == 0 {
                        clip1(((t[i] as u32 + t[i + 1] as u32 + 1) >> 1) as i32, max_sample)
                    } else {
                        clip1(
                            ((t[i] as u32 + 2 * t[i + 1] as u32 + t[i + 2] as u32 + 2) >> 2) as i32,
                            max_sample,
                        )
                    };
                    out[y * 4 + x] = v;
                }
            }
        }
        HorizontalUp => {
            let l = &n.left;
            for y in 0..4 {
                for x in 0..4 {
                    let zhu = x + 2 * y;
                    let v = match zhu {
                        0 | 2 | 4 => {
                            let yy = y + (x >> 1);
                            clip1(((l[yy] as u32 + l[yy + 1] as u32 + 1) >> 1) as i32, max_sample)
                        }
                        1 | 3 => {
                            let yy = y + (x >> 1);
                            clip1(
                                ((l[yy] as u32 + 2 * l[yy + 1] as u32 + l[yy + 2] as u32 + 2) >> 2)
                                    as i32,
                                max_sample,
                            )
                        }
                        5 => clip1(((l[2] as u32 + 3 * l[3] as u32 + 2) >> 2) as i32, max_sample),
                        _ => l[3],
                    };
                    out[y * 4 + x] = v;
                }
            }
        }
    }
}

pub fn predict_intra_16x16(
    out: &mut [u16; 256],
    mode: Intra16x16Mode,
    n: &Intra16x16Neighbours16,
    bit_depth: u8,
) {
    use Intra16x16Mode::*;
    let max_sample: u16 = (1u32 << bit_depth) as u16 - 1;
    let mid: u32 = 1u32 << (bit_depth - 1);
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
                (false, false) => mid,
            };
            let v = clip1(dc as i32, max_sample);
            for px in out.iter_mut() {
                *px = v;
            }
        }
        Plane => {
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
            let mut v_sum: i32 = 0;
            for i in 0..8i32 {
                v_sum += (i + 1) * (left_at(8 + i) - left_at(6 - i));
            }
            let b = (5 * h + 32) >> 6;
            let c = (5 * v_sum + 32) >> 6;
            let a = 16 * (n.left[15] as i32 + n.top[15] as i32);
            for y in 0..16 {
                for x in 0..16 {
                    let p = (a + b * (x as i32 - 7) + c * (y as i32 - 7) + 16) >> 5;
                    out[y * 16 + x] = clip1(p, max_sample);
                }
            }
        }
    }
}

pub fn predict_intra_chroma(
    out: &mut [u16; 64],
    mode: IntraChromaMode,
    n: &IntraChromaNeighbours16,
    bit_depth: u8,
) {
    use IntraChromaMode::*;
    let max_sample: u16 = (1u32 << bit_depth) as u16 - 1;
    let mid: u32 = 1u32 << (bit_depth - 1);
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
            let dc = |topa: bool, lefta: bool, tslice: &[u16], lslice: &[u16]| -> u16 {
                let val = match (topa, lefta) {
                    (true, true) => {
                        let s: u32 = tslice.iter().map(|&v| v as u32).sum::<u32>()
                            + lslice.iter().map(|&v| v as u32).sum::<u32>();
                        (s + 4) >> 3
                    }
                    (true, false) => {
                        let s: u32 = tslice.iter().map(|&v| v as u32).sum();
                        (s + (tslice.len() as u32 / 2)) / tslice.len() as u32
                    }
                    (false, true) => {
                        let s: u32 = lslice.iter().map(|&v| v as u32).sum();
                        (s + (lslice.len() as u32 / 2)) / lslice.len() as u32
                    }
                    (false, false) => mid,
                };
                clip1(val as i32, max_sample)
            };
            let q00 = dc(
                n.top_available,
                n.left_available,
                &n.top[0..4],
                &n.left[0..4],
            );
            let q10 = if n.top_available {
                let s: u32 = n.top[4..8].iter().map(|&v| v as u32).sum();
                clip1(((s + 2) >> 2) as i32, max_sample)
            } else if n.left_available {
                let s: u32 = n.left[0..4].iter().map(|&v| v as u32).sum();
                clip1(((s + 2) >> 2) as i32, max_sample)
            } else {
                clip1(mid as i32, max_sample)
            };
            let q01 = if n.left_available {
                let s: u32 = n.left[4..8].iter().map(|&v| v as u32).sum();
                clip1(((s + 2) >> 2) as i32, max_sample)
            } else if n.top_available {
                let s: u32 = n.top[0..4].iter().map(|&v| v as u32).sum();
                clip1(((s + 2) >> 2) as i32, max_sample)
            } else {
                clip1(mid as i32, max_sample)
            };
            let q11 = dc(
                n.top_available,
                n.left_available,
                &n.top[4..8],
                &n.left[4..8],
            );
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
            let mut v_sum: i32 = 0;
            for i in 0..4i32 {
                v_sum += (i + 1) * (left_at(4 + i) - left_at(2 - i));
            }
            let b = (34 * h + 32) >> 6;
            let c = (34 * v_sum + 32) >> 6;
            let a = 16 * (n.left[7] as i32 + n.top[7] as i32);
            for y in 0..8 {
                for x in 0..8 {
                    let p = (a + b * (x as i32 - 3) + c * (y as i32 - 3) + 16) >> 5;
                    out[y * 8 + x] = clip1(p, max_sample);
                }
            }
        }
    }
}

fn filter_ref_8x8(n: &Intra8x8Neighbours16) -> ([u16; 16], [u16; 8], u16) {
    let mut ft = [0u16; 16];
    let mut fl = [0u16; 8];
    let ftl;

    if n.top_available {
        let tl_src = if n.top_left_available {
            n.top_left
        } else {
            n.top[0]
        };
        ft[0] = ((tl_src as u32 + 2 * n.top[0] as u32 + n.top[1] as u32 + 2) >> 2) as u16;
        for i in 1..15 {
            ft[i] = ((n.top[i - 1] as u32 + 2 * n.top[i] as u32 + n.top[i + 1] as u32 + 2) >> 2)
                as u16;
        }
        ft[15] = ((n.top[14] as u32 + 3 * n.top[15] as u32 + 2) >> 2) as u16;
    }

    if n.left_available {
        let tl_src = if n.top_left_available {
            n.top_left
        } else {
            n.left[0]
        };
        fl[0] = ((tl_src as u32 + 2 * n.left[0] as u32 + n.left[1] as u32 + 2) >> 2) as u16;
        for i in 1..7 {
            fl[i] = ((n.left[i - 1] as u32 + 2 * n.left[i] as u32 + n.left[i + 1] as u32 + 2) >> 2)
                as u16;
        }
        fl[7] = ((n.left[6] as u32 + 3 * n.left[7] as u32 + 2) >> 2) as u16;
    }

    if n.top_left_available {
        ftl = match (n.top_available, n.left_available) {
            (true, true) => {
                ((n.top[0] as u32 + 2 * n.top_left as u32 + n.left[0] as u32 + 2) >> 2) as u16
            }
            (true, false) => ((3 * n.top_left as u32 + n.top[0] as u32 + 2) >> 2) as u16,
            (false, true) => ((3 * n.top_left as u32 + n.left[0] as u32 + 2) >> 2) as u16,
            (false, false) => n.top_left,
        };
    } else {
        ftl = 0;
    }

    (ft, fl, ftl)
}

pub fn predict_intra_8x8(
    out: &mut [u16; 64],
    mode: Intra8x8Mode,
    n: &Intra8x8Neighbours16,
    bit_depth: u8,
) {
    use Intra8x8Mode::*;
    let max_sample: u16 = (1u32 << bit_depth) as u16 - 1;
    let mid: u32 = 1u32 << (bit_depth - 1);
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

    match mode {
        Vertical => {
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
                (false, false) => mid,
            };
            let v = clip1(dc as i32, max_sample);
            for px in out.iter_mut() {
                *px = v;
            }
        }
        DiagonalDownLeft => {
            for y in 0..8 {
                for x in 0..8 {
                    let v = if x == 7 && y == 7 {
                        clip1(((ft[14] as u32 + 3 * ft[15] as u32 + 2) >> 2) as i32, max_sample)
                    } else {
                        clip1(
                            ((ft[x + y] as u32 + 2 * ft[x + y + 1] as u32 + ft[x + y + 2] as u32 + 2)
                                >> 2) as i32,
                            max_sample,
                        )
                    };
                    out[y * 8 + x] = v;
                }
            }
        }
        DiagonalDownRight => {
            let mut e = [0u16; 17];
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
                    out[y * 8 + x] = clip1(((l + 2 * m + r + 2) >> 2) as i32, max_sample);
                }
            }
        }
        VerticalRight => {
            let mut e = [0u16; 17];
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
                        clip1(((e[i] as u32 + e[i + 1] as u32 + 1) >> 1) as i32, max_sample)
                    } else if zvr >= 1 {
                        let base = 8 + x as i32 - ((y as i32 + 1) >> 1);
                        let i = base as usize;
                        clip1(
                            ((e[i] as u32 + 2 * e[i + 1] as u32 + e[i + 2] as u32 + 2) >> 2) as i32,
                            max_sample,
                        )
                    } else {
                        let k = -zvr - 1;
                        let i = (7 - k) as usize;
                        clip1(
                            ((e[i] as u32 + 2 * e[i + 1] as u32 + e[i + 2] as u32 + 2) >> 2) as i32,
                            max_sample,
                        )
                    };
                    out[y * 8 + x] = v;
                }
            }
        }
        HorizontalDown => {
            let mut e = [0u16; 17];
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
                        clip1(((e[i] as u32 + e[i - 1] as u32 + 1) >> 1) as i32, max_sample)
                    } else if zhd >= 1 {
                        let base = 8 - (y as i32 - (x as i32 >> 1));
                        let i = base as usize;
                        clip1(
                            ((e[i + 1] as u32 + 2 * e[i] as u32 + e[i - 1] as u32 + 2) >> 2) as i32,
                            max_sample,
                        )
                    } else {
                        let i = (7 - zhd) as usize;
                        clip1(
                            ((e[i - 1] as u32 + 2 * e[i] as u32 + e[i + 1] as u32 + 2) >> 2) as i32,
                            max_sample,
                        )
                    };
                    out[y * 8 + x] = v;
                }
            }
        }
        VerticalLeft => {
            for y in 0..8 {
                for x in 0..8 {
                    let i = x + (y >> 1);
                    let v = if y % 2 == 0 {
                        clip1(((ft[i] as u32 + ft[i + 1] as u32 + 1) >> 1) as i32, max_sample)
                    } else {
                        clip1(
                            ((ft[i] as u32 + 2 * ft[i + 1] as u32 + ft[i + 2] as u32 + 2) >> 2)
                                as i32,
                            max_sample,
                        )
                    };
                    out[y * 8 + x] = v;
                }
            }
        }
        HorizontalUp => {
            for y in 0..8 {
                for x in 0..8 {
                    let zhu = x + 2 * y;
                    let v = if zhu <= 12 && zhu % 2 == 0 {
                        let i = y + (x >> 1);
                        clip1(((fl[i] as u32 + fl[i + 1] as u32 + 1) >> 1) as i32, max_sample)
                    } else if zhu <= 11 {
                        let i = y + (x >> 1);
                        clip1(
                            ((fl[i] as u32 + 2 * fl[i + 1] as u32 + fl[i + 2] as u32 + 2) >> 2)
                                as i32,
                            max_sample,
                        )
                    } else if zhu == 13 {
                        clip1(((fl[6] as u32 + 3 * fl[7] as u32 + 2) >> 2) as i32, max_sample)
                    } else {
                        fl[7]
                    };
                    out[y * 8 + x] = v;
                }
            }
        }
    }
}
