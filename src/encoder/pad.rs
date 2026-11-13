//! §7.4.2.1.1 frame cropping on the encoder side: pictures whose size
//! is not a whole number of macroblocks (or, with
//! `frame_mbs_only_flag = 0`, of macroblock pairs) are coded on a
//! padded sample grid and the SPS cropping rectangle
//! (`frame_cropping_flag` + `frame_crop_{left,right,top,bottom}_offset`)
//! tells the decoder which part of it to output.
//!
//! The padding always sits to the right of and below the picture
//! (`frame_crop_left_offset = frame_crop_top_offset = 0`) and repeats
//! the last picture column / row — for field coding the last *two*
//! rows alternately, so every padded row repeats a row of its own
//! field parity. The padded samples are never output, so their value
//! is an encoder choice; edge repetition keeps the intra and inter
//! predictions of the last macroblock row / column cheap.

use crate::sps::FrameCropping;

/// §7.4.2.1.1 eq. 7-19..7-22 — `(CropUnitX, CropUnitY)` in luma
/// samples for a `chroma_array_type` (0 = monochrome or separate colour
/// planes, 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4) and `frame_mbs_only_flag`.
pub fn crop_units(chroma_array_type: u32, frame_mbs_only: bool) -> (u32, u32) {
    let field_factor = if frame_mbs_only { 1 } else { 2 };
    let (sub_width_c, sub_height_c) = match chroma_array_type {
        1 => (2, 2),
        2 => (2, 1),
        _ => (1, 1),
    };
    (sub_width_c, sub_height_c * field_factor)
}

/// Why a picture size cannot be coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GeometryError {
    /// Width or height is zero.
    Empty,
    /// The width is not a multiple of `CropUnitX` (the chroma sampling
    /// has no column for a partial luma group).
    Width {
        /// `CropUnitX` for the layout.
        unit: u32,
    },
    /// The height is not a multiple of `CropUnitY`.
    Height {
        /// `CropUnitY` for the layout.
        unit: u32,
    },
}

impl std::fmt::Display for GeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "picture width and height must be non-zero"),
            Self::Width { unit } => write!(
                f,
                "picture width must be a multiple of {unit} luma samples for this chroma format (§7.4.2.1.1 CropUnitX)"
            ),
            Self::Height { unit } => write!(
                f,
                "picture height must be a multiple of {unit} luma samples for this chroma format / field coding (§7.4.2.1.1 CropUnitY)"
            ),
        }
    }
}

impl std::error::Error for GeometryError {}

/// Coded sample grid for a picture of a given output size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodedGeometry {
    /// Output (cropped) width in luma samples.
    pub display_width: u32,
    /// Output (cropped) height in luma samples.
    pub display_height: u32,
    /// `PicWidthInSamplesL`: the width rounded up to 16.
    pub coded_width: u32,
    /// `16 * FrameHeightInMbs`: the height rounded up to 16
    /// (`frame_mbs_only`) or 32 (field / MBAFF coding).
    pub coded_height: u32,
    /// `frame_mbs_only_flag` the geometry was derived for.
    pub frame_mbs_only: bool,
    /// The SPS cropping offsets, `None` when nothing is cropped.
    pub cropping: Option<FrameCropping>,
}

impl CodedGeometry {
    /// Derive the coded grid and cropping offsets for a
    /// `display_width` x `display_height` picture.
    pub fn new(
        display_width: u32,
        display_height: u32,
        chroma_array_type: u32,
        frame_mbs_only: bool,
    ) -> Result<Self, GeometryError> {
        if display_width == 0 || display_height == 0 {
            return Err(GeometryError::Empty);
        }
        let (cux, cuy) = crop_units(chroma_array_type, frame_mbs_only);
        if !display_width.is_multiple_of(cux) {
            return Err(GeometryError::Width { unit: cux });
        }
        if !display_height.is_multiple_of(cuy) {
            return Err(GeometryError::Height { unit: cuy });
        }
        let coded_width = display_width.next_multiple_of(16);
        let coded_height = display_height.next_multiple_of(if frame_mbs_only { 16 } else { 32 });
        let right = (coded_width - display_width) / cux;
        let bottom = (coded_height - display_height) / cuy;
        Ok(Self {
            display_width,
            display_height,
            coded_width,
            coded_height,
            frame_mbs_only,
            cropping: (right != 0 || bottom != 0).then_some(FrameCropping {
                left: 0,
                right,
                top: 0,
                bottom,
            }),
        })
    }

    /// `true` when the coded grid is larger than the picture.
    pub fn is_padded(&self) -> bool {
        self.cropping.is_some()
    }

    /// Pad one `w` x `h` sample plane of the picture to `cw` x `ch` (the
    /// same plane of the coded grid).
    pub fn pad_plane<T: Copy>(
        &self,
        src: &[T],
        w: usize,
        h: usize,
        cw: usize,
        ch: usize,
    ) -> Vec<T> {
        pad_plane(src, w, h, cw, ch, if self.frame_mbs_only { 1 } else { 2 })
    }
}

/// Pad a row-major `w` x `h` plane to `cw` x `ch` by repeating its last
/// column, then its last `row_period` rows cyclically (1 = frame
/// coding; 2 = field coding, each padded row repeating the last row of
/// its own parity; falls back to 1 when the plane has a single row).
pub fn pad_plane<T: Copy>(
    src: &[T],
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    row_period: usize,
) -> Vec<T> {
    assert!(w >= 1 && h >= 1 && cw >= w && ch >= h, "pad_plane geometry");
    assert_eq!(src.len(), w * h, "pad_plane source size");
    let period = if h >= row_period {
        row_period.max(1)
    } else {
        1
    };
    let mut out = Vec::with_capacity(cw * ch);
    for row in 0..ch {
        let from = if row < h {
            row
        } else {
            h - period + (row - h) % period
        };
        let line = &src[from * w..(from + 1) * w];
        out.extend_from_slice(line);
        out.resize(out.len() + (cw - w), line[w - 1]);
    }
    out
}

/// Crop a row-major `cw`-wide plane to its top-left `w` x `h` window.
pub fn crop_plane<T: Copy>(src: &[T], cw: usize, w: usize, h: usize) -> Vec<T> {
    assert!(w <= cw && src.len() >= cw * h, "crop_plane geometry");
    let mut out = Vec::with_capacity(w * h);
    for row in src.chunks_exact(cw).take(h) {
        out.extend_from_slice(&row[..w]);
    }
    out
}

/// Owned 8-bit (Y, Cb, Cr) planes.
pub type Planes8 = (Vec<u8>, Vec<u8>, Vec<u8>);

impl CodedGeometry {
    /// Pad 8-bit 4:2:0 pictures of the output size to the coded grid.
    pub fn pad_420_frames(&self, frames: &[(&[u8], &[u8], &[u8])]) -> Vec<Planes8> {
        let (w, h) = (self.display_width as usize, self.display_height as usize);
        let (cw, ch) = (self.coded_width as usize, self.coded_height as usize);
        frames
            .iter()
            .map(|&(y, u, v)| {
                (
                    self.pad_plane(y, w, h, cw, ch),
                    self.pad_plane(u, w / 2, h / 2, cw / 2, ch / 2),
                    self.pad_plane(v, w / 2, h / 2, cw / 2, ch / 2),
                )
            })
            .collect()
    }

    /// Crop an 8-bit 4:2:0 picture of the coded grid to the output size.
    pub fn crop_420_frame(&self, f: &Planes8) -> Planes8 {
        let (w, h) = (self.display_width as usize, self.display_height as usize);
        let cw = self.coded_width as usize;
        (
            crop_plane(&f.0, cw, w, h),
            crop_plane(&f.1, cw / 2, w / 2, h / 2),
            crop_plane(&f.2, cw / 2, w / 2, h / 2),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_units_follow_eq_7_19_to_7_22() {
        assert_eq!(crop_units(0, true), (1, 1));
        assert_eq!(crop_units(0, false), (1, 2));
        assert_eq!(crop_units(1, true), (2, 2));
        assert_eq!(crop_units(1, false), (2, 4));
        assert_eq!(crop_units(2, true), (2, 1));
        assert_eq!(crop_units(2, false), (2, 2));
        assert_eq!(crop_units(3, true), (1, 1));
        assert_eq!(crop_units(3, false), (1, 2));
    }

    #[test]
    fn geometry_rounds_up_and_counts_crop_units() {
        let g = CodedGeometry::new(1920, 1080, 1, true).unwrap();
        assert_eq!((g.coded_width, g.coded_height), (1920, 1088));
        assert_eq!(
            g.cropping,
            Some(FrameCropping {
                left: 0,
                right: 0,
                top: 0,
                bottom: 4
            })
        );
        // Field coding: 1080 rounds to 1088 (34 pairs), CropUnitY = 4.
        let g = CodedGeometry::new(1920, 1080, 1, false).unwrap();
        assert_eq!(g.coded_height, 1088);
        assert_eq!(g.cropping.unwrap().bottom, 2);
        let g = CodedGeometry::new(33, 17, 3, true).unwrap();
        assert_eq!((g.coded_width, g.coded_height), (48, 32));
        let c = g.cropping.unwrap();
        assert_eq!((c.right, c.bottom), (15, 15));
        assert!(!CodedGeometry::new(64, 48, 1, true).unwrap().is_padded());
        assert_eq!(
            CodedGeometry::new(33, 16, 1, true),
            Err(GeometryError::Width { unit: 2 })
        );
        assert_eq!(
            CodedGeometry::new(32, 18, 1, false),
            Err(GeometryError::Height { unit: 4 })
        );
        assert_eq!(CodedGeometry::new(0, 4, 3, true), Err(GeometryError::Empty));
    }

    #[test]
    fn pad_repeats_edges_per_parity() {
        let src = [1u8, 2, 3, 4, 5, 6];
        // 3x2 → 5x5, frame rows.
        let p = pad_plane(&src, 3, 2, 5, 5, 1);
        assert_eq!(&p[..5], &[1, 2, 3, 3, 3]);
        assert_eq!(&p[5..10], &[4, 5, 6, 6, 6]);
        assert_eq!(&p[20..25], &[4, 5, 6, 6, 6]);
        // Field rows: row 2 repeats row 0, row 3 repeats row 1.
        let p = pad_plane(&src, 3, 2, 3, 5, 2);
        assert_eq!(&p[6..9], &[1, 2, 3]);
        assert_eq!(&p[9..12], &[4, 5, 6]);
        assert_eq!(&p[12..15], &[1, 2, 3]);
        assert_eq!(crop_plane(&p, 3, 2, 2), vec![1, 2, 4, 5]);
    }
}
