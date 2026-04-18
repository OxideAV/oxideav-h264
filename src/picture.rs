//! Reconstructed picture buffer (YUV420P) with macroblock helpers.
//!
//! Used by the decoder to gather reconstructed luma and chroma samples per
//! macroblock and to feed them to the deblocking pass.

use oxideav_core::{frame::VideoPlane, PixelFormat, TimeBase, VideoFrame};

use crate::mb_type::{IMbType, PMbType, PPartition};

/// §6.4.1 — (SubWidthC, SubHeightC) for a given `chroma_format_idc`.
/// Chroma plane sample dimensions are `PicWidthInSamplesL / SubWidthC`
/// by `PicHeightInSamplesL / SubHeightC`.
///
/// | chroma_format_idc | meaning     | SubWidthC | SubHeightC |
/// |-------------------|-------------|-----------|------------|
/// | 0                 | monochrome  |     -     |     -      |
/// | 1                 | 4:2:0       |     2     |     2      |
/// | 2                 | 4:2:2       |     2     |     1      |
/// | 3                 | 4:4:4       |     1     |     1      |
pub fn chroma_subsampling(chroma_format_idc: u32) -> (u32, u32) {
    match chroma_format_idc {
        1 => (2, 2),
        2 => (2, 1),
        3 => (1, 1),
        // Monochrome falls back to (2, 2) so the chroma stride math stays
        // benign; callers must gate on `chroma_format_idc == 0` before
        // touching chroma planes.
        _ => (2, 2),
    }
}

/// §6.4.1 — chroma plane width (luma width ÷ SubWidthC).
pub fn chroma_plane_w(luma_w: u32, chroma_format_idc: u32) -> u32 {
    let (sub_w, _) = chroma_subsampling(chroma_format_idc);
    luma_w / sub_w
}

/// §6.4.1 — chroma plane height (luma height ÷ SubHeightC).
pub fn chroma_plane_h(luma_h: u32, chroma_format_idc: u32) -> u32 {
    let (_, sub_h) = chroma_subsampling(chroma_format_idc);
    luma_h / sub_h
}

#[derive(Clone, Debug)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub mb_width: u32,
    pub mb_height: u32,
    /// Chroma format of this picture's chroma planes (§6.4.1 Table 6-1).
    /// 1 = 4:2:0 (default for existing code paths), 3 = 4:4:4. The
    /// chroma plane size, stride, and offset helpers key on this field.
    pub chroma_format_idc: u32,
    /// Luma bit depth (`bit_depth_luma_minus8 + 8`, §7.4.2.1.1). Defaults
    /// to 8. When set to 10, the high-bit-depth u16 planes
    /// ([`Self::y16`] / [`Self::cb16`] / [`Self::cr16`]) hold the
    /// reconstructed samples and the 8-bit `y / cb / cr` planes stay
    /// zero-sized. All downstream emission honours this field.
    pub bit_depth_y: u8,
    /// Chroma bit depth. Must equal [`Self::bit_depth_y`] in the current
    /// decoder since the high-bit-depth path only wires the 10-bit luma
    /// = 10-bit chroma case.
    pub bit_depth_c: u8,
    /// Luma plane, raw stride == width. Holds 8-bit samples when
    /// [`Self::bit_depth_y`] == 8; empty otherwise (the u16 planes own
    /// the data).
    pub y: Vec<u8>,
    /// Cb plane, raw stride follows [`Self::chroma_stride`] (width / SubWidthC).
    pub cb: Vec<u8>,
    /// Cr plane, raw stride follows [`Self::chroma_stride`] (width / SubWidthC).
    pub cr: Vec<u8>,
    /// 10-bit luma plane (used when [`Self::bit_depth_y`] > 8). Stored in
    /// the range `0..=(1 << bit_depth_y) - 1` per §8.5 `Clip1Y`. Length
    /// matches the 8-bit `y` shape when active. Empty in 8-bit mode.
    pub y16: Vec<u16>,
    /// 10-bit Cb plane. See [`Self::y16`].
    pub cb16: Vec<u16>,
    /// 10-bit Cr plane. See [`Self::y16`].
    pub cr16: Vec<u16>,
    /// Per-macroblock state needed for predictor neighbour resolution and
    /// deblocking. Stored in raster order.
    pub mb_info: Vec<MbInfo>,
    /// Tracks the most recently decoded `mb_qp_delta` for CABAC ctxIdxInc
    /// derivation (§9.3.3.1.1.5). Separate from per-MB state because the
    /// spec keys on "the previous MB in decoding order", not any neighbour.
    pub last_mb_qp_delta_was_nonzero: bool,
    /// Active scaling-list matrices for the current slice, resolved per
    /// §7.4.2.2 Table 7-2. Set by the decoder at slice-entry time;
    /// reset to flat on picture allocation. Referenced by the
    /// dequant paths in `mb.rs` / `p_mb.rs` / `b_mb.rs` / `cabac/*.rs`.
    pub scaling_lists: crate::scaling_list::ScalingLists,
}

#[derive(Clone, Debug)]
pub struct MbInfo {
    /// QP_Y for the macroblock (after slice_qp_delta + per-MB delta).
    pub qp_y: i32,
    /// True if this macroblock decoded successfully (otherwise its samples
    /// are placeholder zeros and shouldn't be used as predictors).
    pub coded: bool,
    /// Per-4×4 luma block coefficient counts (TotalCoeff). 16 entries indexed
    /// in raster order within the MB (`row*4 + col`). Used by §9.2.1.1 to
    /// compute the predicted nC for the block to the right / below.
    pub luma_nc: [u8; 16],
    /// Per-4×4 chroma Cb coefficient counts. For 4:2:0 the first 4 entries
    /// (2×2 sub-blocks) are used; for 4:4:4 all 16 entries matter (the
    /// chroma plane is the same size as luma and follows the luma 4×4
    /// grid). Entries beyond the active count stay zero.
    pub cb_nc: [u8; 16],
    /// Per-4×4 chroma Cr coefficient counts. Same shape as
    /// [`Self::cb_nc`] — 4 active entries in 4:2:0, 16 in 4:4:4.
    pub cr_nc: [u8; 16],
    /// Intra4x4 modes per chroma-Cb 4×4 sub-block — used in 4:4:4 for
    /// `predict_intra4x4_mode_with`-style neighbour mode prediction on
    /// the chroma plane. Unused in 4:2:0.
    pub cb_intra4x4_pred_mode: [u8; 16],
    /// Intra4x4 modes per chroma-Cr 4×4 sub-block. See
    /// [`Self::cb_intra4x4_pred_mode`].
    pub cr_intra4x4_pred_mode: [u8; 16],
    /// Intra4x4 modes per sub-block (16 entries) — used for mode prediction.
    /// For Intra16x16/PCM macroblocks, all entries are `INTRA_DC_FAKE` so
    /// neighbouring 4×4 blocks fall through to the DC fallback (per §8.3.1.1).
    pub intra4x4_pred_mode: [u8; 16],
    /// `intra_chroma_pred_mode` for this MB — used by CABAC's
    /// §9.3.3.1.1.8 ctxIdxInc derivation for the neighbour MBs.
    pub intra_chroma_pred_mode: u8,
    /// Parsed I-slice macroblock type (CABAC ctxIdxInc for `mb_type`
    /// §9.3.3.1.1.3 depends on whether the neighbour MB is `I_NxN` or not).
    pub mb_type_i: Option<IMbType>,
    /// Parsed P-slice macroblock type. `None` for intra macroblocks.
    pub mb_type_p: Option<PMbType>,
    /// P-slice partition layout for inter macroblocks — cached here so
    /// neighbour MV lookups don't re-parse `mb_type_p`.
    pub p_partition: Option<PPartition>,
    /// Per-4×4-sub-block list-0 motion vectors in quarter-pel units
    /// (`mv[row*4+col]`). Zero for intra macroblocks.
    pub mv_l0: [(i16, i16); 16],
    /// Per-4×4-sub-block list-0 reference indices. `-1` for "no prediction"
    /// (intra or not yet decoded); `0..` otherwise.
    pub ref_idx_l0: [i8; 16],
    /// Per-4×4-sub-block list-1 motion vectors in quarter-pel units.
    /// Populated only for B-slice inter macroblocks; zero otherwise.
    pub mv_l1: [(i16, i16); 16],
    /// Per-4×4-sub-block list-1 reference indices. `-1` means "list-1
    /// prediction not used at this 4×4 location" (P macroblocks, L0-only
    /// B sub-partitions, intra MBs).
    pub ref_idx_l1: [i8; 16],
    /// POC of the list-0 reference pointed to by `ref_idx_l0[i]`, snapshotted
    /// at slice-decode end so that a later B-slice can resolve this picture's
    /// colocated block's MV reference via POC (§8.4.1.2.3 temporal direct).
    /// `i32::MIN` sentinel means "no list-0 reference at this 4×4 sub-block"
    /// (intra, or L1-only B sub-partition). Populated by
    /// [`Picture::snapshot_ref_pocs`] after slice decode completes.
    pub ref_poc_l0: [i32; 16],
    /// POC of the list-1 reference at each 4×4 sub-block. `i32::MIN` for
    /// "no list-1 reference." See [`Self::ref_poc_l0`].
    pub ref_poc_l1: [i32; 16],
    /// Per-4×4-sub-block absolute list-0 MVD components. Used by CABAC
    /// §9.3.3.1.1.7 ctxIdxInc derivation (`absMvdComp_A + absMvdComp_B`).
    /// Set on the partition's top-left 4×4 block and replicated into
    /// every 4×4 block the partition covers so neighbour lookups into
    /// adjacent partitions / MBs read the right summed magnitude.
    pub mvd_l0_abs: [(u16, u16); 16],
    /// Per-4×4-sub-block absolute list-1 MVD components. Same role as
    /// [`Self::mvd_l0_abs`] but for `mvd_l1_{x,y}` under §9.3.3.1.1.7.
    /// Populated only by the CABAC B-slice entropy driver.
    pub mvd_l1_abs: [(u16, u16); 16],
    /// True when this macroblock is intra. For an I-slice all MBs are intra,
    /// but the field exists to support future P-slice deblocking edges.
    pub intra: bool,
    /// True when this macroblock was `P_Skip` in a P-slice. Used by
    /// deblocking and neighbour lookups so the zero-cbp P_Skip case can be
    /// recognised without re-parsing.
    pub skipped: bool,
    /// True when this macroblock used `transform_size_8x8_flag = 1`
    /// (High Profile 8×8 residual transform). Consumed by §8.7 deblocking
    /// to skip the internal 4×4 edges that don't exist under an 8×8
    /// transform (only the mid-MB edge at sample offset 8 is filtered).
    pub transform_8x8: bool,
    /// True when this is an I_16x16 macroblock whose luma DC block had
    /// `coded_block_flag = 1` (i.e., at least one nonzero DC coefficient).
    /// Used by §9.3.3.1.1.9 CABAC CBF ctxIdxInc derivation for the
    /// Luma16x16 DC block of a neighbouring MB. False for I_NxN (no DC
    /// block) and for MBs where DC CBF was 0.
    pub luma16x16_dc_cbf: bool,
    /// Per-plane chroma DC `coded_block_flag` (Cb=0, Cr=1). Used by
    /// §9.3.3.1.1.9 to derive the CBF ctxIdxInc for a neighbouring MB's
    /// chroma DC block.
    pub chroma_dc_cbf: [bool; 2],
    /// `coded_block_pattern` luma nibble (one bit per 8×8 sub-block,
    /// raster order: bit0 TL, bit1 TR, bit2 BL, bit3 BR). Needed by
    /// §9.3.3.1.1.4 Table 9-39 so a neighbour MB's CBP-luma bit can be
    /// consulted directly rather than reconstructed from 4×4 NNZ counts
    /// (the latter conflates inter 8×8 transform with 4×4 residuals).
    pub cbp_luma: u8,
    /// `coded_block_pattern` chroma value in {0,1,2} per Table 7-16:
    /// 0 = no chroma residual, 1 = chroma DC only, 2 = chroma DC+AC.
    /// Consumed by the §9.3.3.1.1.4 chroma CBP ctxIdxInc derivation on
    /// neighbour MBs.
    pub cbp_chroma: u8,
}

impl Default for MbInfo {
    fn default() -> Self {
        Self {
            qp_y: 0,
            coded: false,
            luma_nc: [0; 16],
            cb_nc: [0; 16],
            cr_nc: [0; 16],
            cb_intra4x4_pred_mode: [INTRA_DC_FAKE; 16],
            cr_intra4x4_pred_mode: [INTRA_DC_FAKE; 16],
            intra4x4_pred_mode: [INTRA_DC_FAKE; 16],
            intra_chroma_pred_mode: 0,
            mb_type_i: None,
            mb_type_p: None,
            p_partition: None,
            mv_l0: [(0, 0); 16],
            ref_idx_l0: [-1; 16],
            mv_l1: [(0, 0); 16],
            ref_idx_l1: [-1; 16],
            ref_poc_l0: [i32::MIN; 16],
            ref_poc_l1: [i32::MIN; 16],
            mvd_l0_abs: [(0, 0); 16],
            mvd_l1_abs: [(0, 0); 16],
            intra: false,
            skipped: false,
            transform_8x8: false,
            luma16x16_dc_cbf: false,
            chroma_dc_cbf: [false; 2],
            cbp_luma: 0,
            cbp_chroma: 0,
        }
    }
}

/// Sentinel used as "intra mode unavailable for prediction" — see §8.3.1.1.
pub const INTRA_DC_FAKE: u8 = 2;

impl Picture {
    /// Allocate a 4:2:0 picture (backwards-compatible default). New call sites
    /// that need a different chroma format must go through
    /// [`Self::new_with_format`].
    pub fn new(mb_width: u32, mb_height: u32) -> Self {
        Self::new_with_format(mb_width, mb_height, 1)
    }

    /// Allocate a picture with the chroma plane sized per §6.4.1 Table 6-1.
    /// For `chroma_format_idc = 3` (4:4:4) the chroma planes match the
    /// luma dimensions and each chroma MB occupies 16 samples per
    /// row/column — the helpers below scale the stride and per-MB
    /// offsets accordingly.
    pub fn new_with_format(mb_width: u32, mb_height: u32, chroma_format_idc: u32) -> Self {
        Self::new_with_bit_depth(mb_width, mb_height, chroma_format_idc, 8, 8)
    }

    /// Allocate a picture at the given chroma format and bit depths.
    /// When `bit_depth_y` is 8, the 8-bit planes are allocated and the
    /// 10-bit companions stay empty; above 8, only the u16 planes are
    /// populated with the mid-grey value `1 << (bit_depth - 1)`.
    pub fn new_with_bit_depth(
        mb_width: u32,
        mb_height: u32,
        chroma_format_idc: u32,
        bit_depth_y: u8,
        bit_depth_c: u8,
    ) -> Self {
        let width = mb_width * 16;
        let height = mb_height * 16;
        let cw = chroma_plane_w(width, chroma_format_idc);
        let ch = chroma_plane_h(height, chroma_format_idc);
        let luma_pixels = (width * height) as usize;
        let chroma_pixels = (cw * ch) as usize;
        let hi_y = bit_depth_y > 8;
        let hi_c = bit_depth_c > 8;
        let mid_y: u16 = 1 << (bit_depth_y - 1);
        let mid_c: u16 = 1 << (bit_depth_c - 1);
        Self {
            width,
            height,
            mb_width,
            mb_height,
            chroma_format_idc,
            bit_depth_y,
            bit_depth_c,
            y: if hi_y { Vec::new() } else { vec![0u8; luma_pixels] },
            cb: if hi_c {
                Vec::new()
            } else {
                vec![128u8; chroma_pixels]
            },
            cr: if hi_c {
                Vec::new()
            } else {
                vec![128u8; chroma_pixels]
            },
            y16: if hi_y {
                vec![0u16; luma_pixels]
            } else {
                Vec::new()
            },
            cb16: if hi_c {
                vec![mid_c; chroma_pixels]
            } else {
                Vec::new()
            },
            cr16: if hi_c {
                vec![mid_c; chroma_pixels]
            } else {
                Vec::new()
            },
            mb_info: vec![MbInfo::default(); (mb_width * mb_height) as usize],
            last_mb_qp_delta_was_nonzero: false,
            scaling_lists: crate::scaling_list::ScalingLists::flat(),
        }
        // `mid_y` exists for future symmetry; luma recon always overwrites
        // so there's no need to pre-fill the luma plane with grey.
        .mid_y_reserved(mid_y)
    }

    fn mid_y_reserved(self, _mid_y: u16) -> Self {
        self
    }

    /// True when this picture holds samples in the u16 high-bit-depth
    /// planes. Downstream code keys on this to pick the u8 vs u16 emit
    /// path without re-deriving from `bit_depth_y`.
    pub fn is_high_bit_depth(&self) -> bool {
        self.bit_depth_y > 8 || self.bit_depth_c > 8
    }

    /// Stride (bytes per row) of the luma plane.
    pub fn luma_stride(&self) -> usize {
        self.width as usize
    }
    /// Stride (bytes per row) of either chroma plane — `width / SubWidthC`
    /// per §6.4.1. For 4:2:0 this equals `width / 2`; for 4:4:4 it equals
    /// the luma width.
    pub fn chroma_stride(&self) -> usize {
        let (sub_w, _) = chroma_subsampling(self.chroma_format_idc);
        (self.width / sub_w) as usize
    }
    /// Width of a chroma macroblock in samples (16 / SubWidthC). 8 for 4:2:0,
    /// 16 for 4:4:4. Drives the per-MB chroma `chroma_off` stride stride.
    pub fn mb_width_chroma(&self) -> usize {
        let (sub_w, _) = chroma_subsampling(self.chroma_format_idc);
        16 / sub_w as usize
    }
    /// Height of a chroma macroblock in samples (16 / SubHeightC). 8 for
    /// 4:2:0, 16 for 4:4:4.
    pub fn mb_height_chroma(&self) -> usize {
        let (_, sub_h) = chroma_subsampling(self.chroma_format_idc);
        16 / sub_h as usize
    }

    /// Linear address of the top-left luma sample of `(mb_x, mb_y)`.
    pub fn luma_off(&self, mb_x: u32, mb_y: u32) -> usize {
        (mb_y as usize * 16) * self.luma_stride() + (mb_x as usize * 16)
    }
    pub fn chroma_off(&self, mb_x: u32, mb_y: u32) -> usize {
        let mbw = self.mb_width_chroma();
        let mbh = self.mb_height_chroma();
        (mb_y as usize * mbh) * self.chroma_stride() + (mb_x as usize * mbw)
    }

    /// Get a mutable reference to a single MB's bookkeeping.
    pub fn mb_info_mut(&mut self, mb_x: u32, mb_y: u32) -> &mut MbInfo {
        let idx = (mb_y * self.mb_width + mb_x) as usize;
        &mut self.mb_info[idx]
    }
    pub fn mb_info_at(&self, mb_x: u32, mb_y: u32) -> &MbInfo {
        let idx = (mb_y * self.mb_width + mb_x) as usize;
        &self.mb_info[idx]
    }

    /// Snapshot the POC of each 4×4 block's list-0 / list-1 reference so that
    /// a later B-slice using this picture as a colocated reference can resolve
    /// `refIdxCol` → POC without holding the RefPicList references alive.
    /// Walks every MB and resolves `ref_idx_lN[i]` against the supplied lists.
    /// An out-of-range index or `ref_idx < 0` maps to `i32::MIN` (the "none"
    /// sentinel). Called from the decoder at end-of-slice, before the picture
    /// is inserted into the DPB (§8.4.1.2.3 temporal direct colocated lookup).
    pub fn snapshot_ref_pocs(&mut self, list0_pocs: &[i32], list1_pocs: &[i32]) {
        for info in &mut self.mb_info {
            for i in 0..16usize {
                let r0 = info.ref_idx_l0[i];
                info.ref_poc_l0[i] = if r0 >= 0 {
                    list0_pocs.get(r0 as usize).copied().unwrap_or(i32::MIN)
                } else {
                    i32::MIN
                };
                let r1 = info.ref_idx_l1[i];
                info.ref_poc_l1[i] = if r1 >= 0 {
                    list1_pocs.get(r1 as usize).copied().unwrap_or(i32::MIN)
                } else {
                    i32::MIN
                };
            }
        }
    }

    /// Crop to (visible_w × visible_h) and emit a `VideoFrame`. The picture's
    /// raw dimensions are MB-aligned; the encoder requested any smaller
    /// visible area via SPS frame cropping.
    pub fn into_video_frame(
        self,
        visible_w: u32,
        visible_h: u32,
        pts: Option<i64>,
        time_base: TimeBase,
    ) -> VideoFrame {
        self.to_video_frame(visible_w, visible_h, pts, time_base)
    }

    /// Non-consuming variant of [`Self::into_video_frame`] — used by the
    /// decoder when it needs to both emit a frame and keep the picture
    /// around as a reference for subsequent P-slice decodes.
    pub fn to_video_frame(
        &self,
        visible_w: u32,
        visible_h: u32,
        pts: Option<i64>,
        time_base: TimeBase,
    ) -> VideoFrame {
        if self.is_high_bit_depth() {
            return self.to_video_frame_hi(visible_w, visible_h, pts, time_base);
        }
        let (sub_w, sub_h) = chroma_subsampling(self.chroma_format_idc);
        let cw = visible_w.div_ceil(sub_w);
        let ch = visible_h.div_ceil(sub_h);
        let l_stride = self.luma_stride();
        let c_stride = self.chroma_stride();

        let mut y_out = Vec::with_capacity((visible_w * visible_h) as usize);
        for r in 0..visible_h as usize {
            let off = r * l_stride;
            y_out.extend_from_slice(&self.y[off..off + visible_w as usize]);
        }
        let mut cb_out = Vec::with_capacity((cw * ch) as usize);
        let mut cr_out = Vec::with_capacity((cw * ch) as usize);
        for r in 0..ch as usize {
            let off = r * c_stride;
            cb_out.extend_from_slice(&self.cb[off..off + cw as usize]);
            cr_out.extend_from_slice(&self.cr[off..off + cw as usize]);
        }

        let format = match self.chroma_format_idc {
            3 => PixelFormat::Yuv444P,
            _ => PixelFormat::Yuv420P,
        };
        VideoFrame {
            format,
            width: visible_w,
            height: visible_h,
            pts,
            time_base,
            planes: vec![
                VideoPlane {
                    stride: visible_w as usize,
                    data: y_out,
                },
                VideoPlane {
                    stride: cw as usize,
                    data: cb_out,
                },
                VideoPlane {
                    stride: cw as usize,
                    data: cr_out,
                },
            ],
        }
    }

    /// Emit a `Yuv420P10Le` / `Yuv444P10Le` frame by packing the u16
    /// planes in little-endian order. Strides are reported in **bytes**
    /// per `VideoPlane::stride` convention, i.e. `width * 2` for 10-bit.
    fn to_video_frame_hi(
        &self,
        visible_w: u32,
        visible_h: u32,
        pts: Option<i64>,
        time_base: TimeBase,
    ) -> VideoFrame {
        let (sub_w, sub_h) = chroma_subsampling(self.chroma_format_idc);
        let cw = visible_w.div_ceil(sub_w);
        let ch = visible_h.div_ceil(sub_h);
        let l_stride = self.luma_stride();
        let c_stride = self.chroma_stride();

        let pack_row = |src: &[u16], dst: &mut Vec<u8>, w: usize| {
            for &s in &src[..w] {
                dst.push((s & 0xFF) as u8);
                dst.push((s >> 8) as u8);
            }
        };

        let mut y_bytes = Vec::with_capacity((visible_w * visible_h * 2) as usize);
        for r in 0..visible_h as usize {
            let off = r * l_stride;
            pack_row(&self.y16[off..off + visible_w as usize], &mut y_bytes, visible_w as usize);
        }
        let mut cb_bytes = Vec::with_capacity((cw * ch * 2) as usize);
        let mut cr_bytes = Vec::with_capacity((cw * ch * 2) as usize);
        for r in 0..ch as usize {
            let off = r * c_stride;
            pack_row(&self.cb16[off..off + cw as usize], &mut cb_bytes, cw as usize);
            pack_row(&self.cr16[off..off + cw as usize], &mut cr_bytes, cw as usize);
        }

        let format = match self.chroma_format_idc {
            3 => PixelFormat::Yuv444P10Le,
            _ => PixelFormat::Yuv420P10Le,
        };
        VideoFrame {
            format,
            width: visible_w,
            height: visible_h,
            pts,
            time_base,
            planes: vec![
                VideoPlane {
                    stride: (visible_w as usize) * 2,
                    data: y_bytes,
                },
                VideoPlane {
                    stride: (cw as usize) * 2,
                    data: cb_bytes,
                },
                VideoPlane {
                    stride: (cw as usize) * 2,
                    data: cr_bytes,
                },
            ],
        }
    }
}
