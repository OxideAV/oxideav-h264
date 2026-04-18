//! Reconstructed picture buffer (YUV420P) with macroblock helpers.
//!
//! Used by the decoder to gather reconstructed luma and chroma samples per
//! macroblock and to feed them to the deblocking pass.

use oxideav_core::{frame::VideoPlane, PixelFormat, TimeBase, VideoFrame};

use crate::mb_type::{IMbType, PMbType, PPartition};

#[derive(Clone, Debug)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub mb_width: u32,
    pub mb_height: u32,
    /// Luma plane, raw stride == width.
    pub y: Vec<u8>,
    /// Cb plane, raw stride == width / 2.
    pub cb: Vec<u8>,
    /// Cr plane, raw stride == width / 2.
    pub cr: Vec<u8>,
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
    /// Per-4×4 chroma Cb coefficient counts (4 entries: 2×2 sub-blocks).
    pub cb_nc: [u8; 4],
    /// Per-4×4 chroma Cr coefficient counts (4 entries).
    pub cr_nc: [u8; 4],
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
}

impl Default for MbInfo {
    fn default() -> Self {
        Self {
            qp_y: 0,
            coded: false,
            luma_nc: [0; 16],
            cb_nc: [0; 4],
            cr_nc: [0; 4],
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
        }
    }
}

/// Sentinel used as "intra mode unavailable for prediction" — see §8.3.1.1.
pub const INTRA_DC_FAKE: u8 = 2;

impl Picture {
    pub fn new(mb_width: u32, mb_height: u32) -> Self {
        let width = mb_width * 16;
        let height = mb_height * 16;
        let cw = width / 2;
        let ch = height / 2;
        Self {
            width,
            height,
            mb_width,
            mb_height,
            y: vec![0u8; (width * height) as usize],
            cb: vec![128u8; (cw * ch) as usize],
            cr: vec![128u8; (cw * ch) as usize],
            mb_info: vec![MbInfo::default(); (mb_width * mb_height) as usize],
            last_mb_qp_delta_was_nonzero: false,
            scaling_lists: crate::scaling_list::ScalingLists::flat(),
        }
    }

    /// Stride (bytes per row) of the luma plane.
    pub fn luma_stride(&self) -> usize {
        self.width as usize
    }
    pub fn chroma_stride(&self) -> usize {
        (self.width / 2) as usize
    }

    /// Linear address of the top-left luma sample of `(mb_x, mb_y)`.
    pub fn luma_off(&self, mb_x: u32, mb_y: u32) -> usize {
        (mb_y as usize * 16) * self.luma_stride() + (mb_x as usize * 16)
    }
    pub fn chroma_off(&self, mb_x: u32, mb_y: u32) -> usize {
        (mb_y as usize * 8) * self.chroma_stride() + (mb_x as usize * 8)
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
        let cw = visible_w.div_ceil(2);
        let ch = visible_h.div_ceil(2);
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

        VideoFrame {
            format: PixelFormat::Yuv420P,
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
}
