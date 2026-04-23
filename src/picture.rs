//! Picture sample buffer for a decoded frame/field.
//!
//! Holds luma + chroma sample planes sized from the active SPS.
//! Samples are stored as `i32` to accommodate bit_depth up to 14; the
//! actual value range is `0..(1 << bit_depth)`.
//!
//! Spec references:
//! * §6.2 — ChromaArrayType and sample-array layout.
//! * §7.4.2.1.1 — picture-size derivations (PicWidthInSamplesL,
//!   PicHeightInSamplesL, MbWidthC, MbHeightC).
//! * §5.7 — Clip3 / Clip1 used when sampling outside the picture
//!   (returns the nearest edge sample, per §8.4.2.2.1).
//!
//! Clean-room: derived only from ITU-T Rec. H.264 (08/2024).

/// A decoded picture sample buffer.
///
/// `luma` is a row-major `i32` buffer of size
/// `width_in_samples * height_in_samples`. Chroma planes `cb` / `cr`
/// are sized from `chroma_array_type` (and are empty for monochrome).
///
/// POC and frame_num are set by the caller after reconstruction.
#[derive(Debug, Clone)]
pub struct Picture {
    pub width_in_samples: u32,
    pub height_in_samples: u32,
    /// §6.2 — 0 = monochrome (or separate_colour_plane_flag),
    /// 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4.
    pub chroma_array_type: u32,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub luma: Vec<i32>,
    pub cb: Vec<i32>,
    pub cr: Vec<i32>,
    /// Set by caller after reconstruction.
    pub pic_order_cnt: i32,
    pub frame_num: u32,
    /// §8.4.1.2.3 — co-located motion data for temporal-direct
    /// derivation. When populated, `mv_l0_grid[mb_addr * 16 + blk4]`
    /// holds the list-0 MV (in 1/4-pel units) of the 4x4 block, and
    /// `ref_idx_l0_grid[mb_addr * 4 + blk8]` holds the list-0 refIdx
    /// (−1 when the list is not used). `is_intra_grid[mb_addr]` flags
    /// intra-coded macroblocks (colZeroFlag wants to treat intra as
    /// "L0 unavailable").
    ///
    /// The grids are sized as:
    ///   - `mv_*_grid`: `width_in_mbs * height_in_mbs * 16`
    ///   - `ref_idx_*_grid`: `width_in_mbs * height_in_mbs * 4`
    ///   - `is_intra_grid`: `width_in_mbs * height_in_mbs`
    ///
    /// Absent for I-only pictures or for pictures whose motion data is
    /// not needed by later B slices.
    pub mb_width_in_picture: u32,
    pub mv_l0_grid: Vec<(i16, i16)>,
    pub mv_l1_grid: Vec<(i16, i16)>,
    pub ref_idx_l0_grid: Vec<i8>,
    pub ref_idx_l1_grid: Vec<i8>,
    pub is_intra_grid: Vec<bool>,
}

impl Picture {
    /// Allocate a zero-filled picture of the requested geometry.
    pub fn new(
        width_in_samples: u32,
        height_in_samples: u32,
        chroma_array_type: u32,
        bit_depth_luma: u32,
        bit_depth_chroma: u32,
    ) -> Self {
        let luma_len = (width_in_samples as usize) * (height_in_samples as usize);
        // §6.2 / Table 6-1 — MbWidthC / MbHeightC.
        //   ChromaArrayType == 0: no chroma.
        //   ChromaArrayType == 1 (4:2:0): MbWidthC=8, MbHeightC=8 → half W/H.
        //   ChromaArrayType == 2 (4:2:2): MbWidthC=8, MbHeightC=16 → half W, full H.
        //   ChromaArrayType == 3 (4:4:4): MbWidthC=16, MbHeightC=16 → full W, full H.
        let (cw, ch) = chroma_dims(chroma_array_type, width_in_samples, height_in_samples);
        let chroma_len = (cw as usize) * (ch as usize);
        Self {
            width_in_samples,
            height_in_samples,
            chroma_array_type,
            bit_depth_luma,
            bit_depth_chroma,
            luma: vec![0; luma_len],
            cb: vec![0; chroma_len],
            cr: vec![0; chroma_len],
            pic_order_cnt: 0,
            frame_num: 0,
            mb_width_in_picture: 0,
            mv_l0_grid: Vec::new(),
            mv_l1_grid: Vec::new(),
            ref_idx_l0_grid: Vec::new(),
            ref_idx_l1_grid: Vec::new(),
            is_intra_grid: Vec::new(),
        }
    }

    /// §6.2 — chroma plane width in samples.
    pub fn chroma_width(&self) -> u32 {
        chroma_dims(
            self.chroma_array_type,
            self.width_in_samples,
            self.height_in_samples,
        )
        .0
    }

    /// §6.2 — chroma plane height in samples.
    pub fn chroma_height(&self) -> u32 {
        chroma_dims(
            self.chroma_array_type,
            self.width_in_samples,
            self.height_in_samples,
        )
        .1
    }

    /// Clamped luma sample access — coordinates outside the picture are
    /// clipped to the picture edge per §8.4.2.2.1 / Clip3 of §5.7. This
    /// matches the neighbour-sampling rule used by intra prediction when
    /// a neighbour is available but the sample index goes off-picture.
    pub fn luma_at(&self, x: i32, y: i32) -> i32 {
        if self.width_in_samples == 0 || self.height_in_samples == 0 {
            return 0;
        }
        let xi = clip3_i32(0, self.width_in_samples as i32 - 1, x) as usize;
        let yi = clip3_i32(0, self.height_in_samples as i32 - 1, y) as usize;
        self.luma[yi * (self.width_in_samples as usize) + xi]
    }

    /// Clamped chroma Cb access (§6.2).
    pub fn cb_at(&self, x: i32, y: i32) -> i32 {
        let cw = self.chroma_width();
        let ch = self.chroma_height();
        if cw == 0 || ch == 0 {
            return 0;
        }
        let xi = clip3_i32(0, cw as i32 - 1, x) as usize;
        let yi = clip3_i32(0, ch as i32 - 1, y) as usize;
        self.cb[yi * (cw as usize) + xi]
    }

    /// Clamped chroma Cr access.
    pub fn cr_at(&self, x: i32, y: i32) -> i32 {
        let cw = self.chroma_width();
        let ch = self.chroma_height();
        if cw == 0 || ch == 0 {
            return 0;
        }
        let xi = clip3_i32(0, cw as i32 - 1, x) as usize;
        let yi = clip3_i32(0, ch as i32 - 1, y) as usize;
        self.cr[yi * (cw as usize) + xi]
    }

    /// Write a luma sample. Out-of-bounds writes are silently ignored
    /// (callers should not produce such writes — this is a hard guard
    /// for robustness during reconstruction edge handling).
    pub fn set_luma(&mut self, x: i32, y: i32, v: i32) {
        if x < 0 || y < 0 {
            return;
        }
        let xi = x as u32;
        let yi = y as u32;
        if xi >= self.width_in_samples || yi >= self.height_in_samples {
            return;
        }
        let idx = (yi as usize) * (self.width_in_samples as usize) + (xi as usize);
        self.luma[idx] = v;
    }

    /// Write a Cb sample.
    pub fn set_cb(&mut self, x: i32, y: i32, v: i32) {
        if x < 0 || y < 0 {
            return;
        }
        let cw = self.chroma_width();
        let ch = self.chroma_height();
        let xi = x as u32;
        let yi = y as u32;
        if xi >= cw || yi >= ch {
            return;
        }
        let idx = (yi as usize) * (cw as usize) + (xi as usize);
        self.cb[idx] = v;
    }

    /// Write a Cr sample.
    pub fn set_cr(&mut self, x: i32, y: i32, v: i32) {
        if x < 0 || y < 0 {
            return;
        }
        let cw = self.chroma_width();
        let ch = self.chroma_height();
        let xi = x as u32;
        let yi = y as u32;
        if xi >= cw || yi >= ch {
            return;
        }
        let idx = (yi as usize) * (cw as usize) + (xi as usize);
        self.cr[idx] = v;
    }

    /// Fetch the colocated L0 MV + refIdx for the 4x4 block at
    /// (`mb_addr`, `blk4`). Returns `None` if motion data was not
    /// populated for this picture (e.g. I-only pic) or `mb_addr` /
    /// `blk4` is out of range.
    ///
    /// `blk4` uses the Figure 6-10 raster index (0..=15).
    pub fn colocated_l0(&self, mb_addr: u32, blk4: usize) -> Option<((i16, i16), i8, bool)> {
        if self.mv_l0_grid.is_empty() {
            return None;
        }
        let mv_idx = (mb_addr as usize).checked_mul(16)?.checked_add(blk4)?;
        let mv = *self.mv_l0_grid.get(mv_idx)?;
        let ref_idx = *self.ref_idx_l0_grid.get((mb_addr as usize) * 4 + (blk4 / 4))?;
        let is_intra = self
            .is_intra_grid
            .get(mb_addr as usize)
            .copied()
            .unwrap_or(false);
        Some((mv, ref_idx, is_intra))
    }

    /// Fetch the colocated L1 MV + refIdx. Used when the colocated
    /// block has predFlagL0 == 0 and the caller needs to fall back on
    /// its L1 MV per §8.4.1.2.3.
    pub fn colocated_l1(&self, mb_addr: u32, blk4: usize) -> Option<((i16, i16), i8, bool)> {
        if self.mv_l1_grid.is_empty() {
            return None;
        }
        let mv_idx = (mb_addr as usize).checked_mul(16)?.checked_add(blk4)?;
        let mv = *self.mv_l1_grid.get(mv_idx)?;
        let ref_idx = *self.ref_idx_l1_grid.get((mb_addr as usize) * 4 + (blk4 / 4))?;
        let is_intra = self
            .is_intra_grid
            .get(mb_addr as usize)
            .copied()
            .unwrap_or(false);
        Some((mv, ref_idx, is_intra))
    }
}

/// §6.2 / Table 6-1 — chroma plane (width, height) given
/// `chroma_array_type` and the luma frame dimensions.
fn chroma_dims(chroma_array_type: u32, w: u32, h: u32) -> (u32, u32) {
    match chroma_array_type {
        0 => (0, 0),
        1 => (w / 2, h / 2),
        2 => (w / 2, h),
        3 => (w, h),
        _ => (0, 0),
    }
}

/// §5.7 — `Clip3(x, y, z) = min(y, max(x, z))`. Inlined here so that
/// Picture stays a self-contained module.
#[inline]
fn clip3_i32(x: i32, y: i32, z: i32) -> i32 {
    if z < x {
        x
    } else if z > y {
        y
    } else {
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_monochrome() {
        let p = Picture::new(32, 16, 0, 8, 8);
        assert_eq!(p.luma.len(), 32 * 16);
        assert!(p.cb.is_empty());
        assert!(p.cr.is_empty());
        assert_eq!(p.chroma_width(), 0);
        assert_eq!(p.chroma_height(), 0);
    }

    #[test]
    fn allocation_yuv420() {
        let p = Picture::new(32, 16, 1, 8, 8);
        assert_eq!(p.luma.len(), 32 * 16);
        assert_eq!(p.cb.len(), 16 * 8);
        assert_eq!(p.cr.len(), 16 * 8);
        assert_eq!(p.chroma_width(), 16);
        assert_eq!(p.chroma_height(), 8);
    }

    #[test]
    fn allocation_yuv422() {
        let p = Picture::new(32, 16, 2, 8, 8);
        assert_eq!(p.luma.len(), 32 * 16);
        assert_eq!(p.cb.len(), 16 * 16);
        assert_eq!(p.cr.len(), 16 * 16);
        assert_eq!(p.chroma_width(), 16);
        assert_eq!(p.chroma_height(), 16);
    }

    #[test]
    fn allocation_yuv444() {
        let p = Picture::new(32, 16, 3, 10, 10);
        assert_eq!(p.luma.len(), 32 * 16);
        assert_eq!(p.cb.len(), 32 * 16);
        assert_eq!(p.cr.len(), 32 * 16);
        assert_eq!(p.chroma_width(), 32);
        assert_eq!(p.chroma_height(), 16);
    }

    #[test]
    fn luma_set_and_get() {
        let mut p = Picture::new(16, 16, 1, 8, 8);
        p.set_luma(3, 4, 123);
        assert_eq!(p.luma_at(3, 4), 123);
        // Other samples untouched.
        assert_eq!(p.luma_at(0, 0), 0);
    }

    #[test]
    fn luma_edge_clamp() {
        let mut p = Picture::new(4, 4, 1, 8, 8);
        // Corner plant so we can detect the clamp.
        p.set_luma(0, 0, 42);
        p.set_luma(3, 3, 77);
        // Off the top-left clamps to (0, 0).
        assert_eq!(p.luma_at(-1, -1), 42);
        // Off the bottom-right clamps to (3, 3).
        assert_eq!(p.luma_at(10, 10), 77);
        // Off the top clamps to row 0.
        p.set_luma(2, 0, 55);
        assert_eq!(p.luma_at(2, -3), 55);
    }

    #[test]
    fn chroma_set_and_get() {
        let mut p = Picture::new(16, 16, 1, 8, 8);
        p.set_cb(2, 3, 10);
        p.set_cr(4, 5, 20);
        assert_eq!(p.cb_at(2, 3), 10);
        assert_eq!(p.cr_at(4, 5), 20);
    }

    #[test]
    fn chroma_edge_clamp() {
        let mut p = Picture::new(8, 8, 1, 8, 8);
        p.set_cb(0, 0, 1);
        p.set_cb(3, 3, 2);
        assert_eq!(p.cb_at(-1, -1), 1);
        assert_eq!(p.cb_at(5, 5), 2);
    }

    #[test]
    fn set_out_of_bounds_silently_ignored() {
        let mut p = Picture::new(4, 4, 1, 8, 8);
        // Should not panic.
        p.set_luma(100, 100, 42);
        p.set_luma(-1, -1, 42);
        assert_eq!(p.luma, vec![0; 16]);
    }
}
