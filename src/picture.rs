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
    /// `true` when this picture represents a single field from a PAFF
    /// coded video sequence (§7.3.3 `field_pic_flag = 1`). The luma /
    /// chroma buffers already hold the half-height field samples; the
    /// decoder flags this so `to_video_frame` can report the field
    /// dimensions verbatim without attempting frame-pair combination.
    pub field_pic: bool,
    /// `true` when this picture is the *bottom* field of a PAFF pair
    /// (§7.3.3 `bottom_field_flag = 1`). Ignored when `field_pic == false`.
    /// Carried on the picture so downstream field-pairing can identify
    /// parity without replaying the slice header.
    pub bottom_field: bool,
    /// §7.3.4 — true when the slice the picture is being decoded from
    /// has `frame_mbs_only_flag = 0 && mb_adaptive_frame_field_flag = 1`
    /// and `field_pic_flag = 0`. Under MBAFF an MB pair (top + bottom
    /// MBs at the same pair coordinate) may be frame- or field-coded
    /// independently; [`MbInfo::mb_field_decoding_flag`] records the
    /// per-pair choice and the sample-offset / row-stride helpers on
    /// this struct key on it to implement §6.4.9.4 neighbour lookups.
    pub mbaff_enabled: bool,
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
    /// 4:4:4 mirror of [`Self::luma16x16_dc_cbf`] for the Cb plane:
    /// under ChromaArrayType = 3 the chroma plane runs a luma-shaped
    /// I_16×16 DC pipeline with its own `coded_block_flag` context, so
    /// neighbour derivations need a per-plane cbf flag.
    pub cb_luma16x16_dc_cbf: bool,
    /// 4:4:4 mirror of [`Self::luma16x16_dc_cbf`] for the Cr plane.
    pub cr_luma16x16_dc_cbf: bool,
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
    /// §7.4.4 `mb_field_decoding_flag` — `true` for a field-coded MB
    /// in an MBAFF pair. Zero for non-MBAFF pictures; per §7.3.4 every
    /// MB in the same pair shares this value (both the top and bottom
    /// MB see the same flag). Drives the per-MB row-stride / sample
    /// offset helpers on [`Picture`].
    pub mb_field_decoding_flag: bool,
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
            cb_luma16x16_dc_cbf: false,
            cr_luma16x16_dc_cbf: false,
            chroma_dc_cbf: [false; 2],
            cbp_luma: 0,
            cbp_chroma: 0,
            mb_field_decoding_flag: false,
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

    /// Allocate a PAFF field picture — `mb_height` is the field's MB row
    /// count (§7.3.3, equal to `PicHeightInMapUnits` when
    /// `field_pic_flag = 1`). The luma plane is `mb_width * 16` wide by
    /// `mb_height * 16` rows (half the parent frame's rows); the chroma
    /// planes follow §6.4.1 subsampling of the field dimensions.
    /// `bottom_field` records which parity this field belongs to so a
    /// downstream pair-combiner can stack top + bottom back into a
    /// full-height frame.
    pub fn new_field(
        mb_width: u32,
        mb_height: u32,
        chroma_format_idc: u32,
        bottom_field: bool,
    ) -> Self {
        let mut pic = Self::new_with_bit_depth(mb_width, mb_height, chroma_format_idc, 8, 8);
        pic.field_pic = true;
        pic.bottom_field = bottom_field;
        pic
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
            field_pic: false,
            bottom_field: false,
            mbaff_enabled: false,
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
    ///
    /// Under MBAFF (§7.3.4) an MB at `mb_y = 2*pair_y + which` in a
    /// field-coded pair occupies rows `pair_y*32 + which + r*2` for
    /// `r = 0..15` — the top MB sits on even plane rows and the bottom
    /// MB on odd rows. In the frame-coded (or non-MBAFF) case the MB
    /// covers contiguous rows `mb_y*16 + r`. The helper always returns
    /// the byte offset of that MB's `(0, 0)` sample; callers walk rows
    /// with [`Self::luma_row_stride_for`] rather than the raw plane
    /// stride.
    pub fn luma_off(&self, mb_x: u32, mb_y: u32) -> usize {
        let lstride = self.luma_stride();
        let (first_row, _) = self.mb_luma_row_info_at(mb_x, mb_y);
        first_row * lstride + (mb_x as usize * 16)
    }
    pub fn chroma_off(&self, mb_x: u32, mb_y: u32) -> usize {
        let cstride = self.chroma_stride();
        let mbw = self.mb_width_chroma();
        let (first_row, _) = self.mb_chroma_row_info_at(mb_x, mb_y);
        first_row * cstride + (mb_x as usize * mbw)
    }

    /// §7.3.4 — per-MB row stride in luma samples. Returns `2 *
    /// luma_stride()` for a field-coded MB in an MBAFF pair (the
    /// MB samples interleave with its sibling) and `luma_stride()`
    /// otherwise. Callers that walk rows inside an MB must use this
    /// instead of [`Self::luma_stride`] once MBAFF is enabled.
    pub fn luma_row_stride_for(&self, mb_y: u32) -> usize {
        self.luma_row_stride_for_at(0, mb_y)
    }

    /// Column-aware variant of [`Self::luma_row_stride_for`]. Needed
    /// on mixed-mode MBAFF pictures where neighbour pairs may carry
    /// different `mb_field_decoding_flag` values.
    pub fn luma_row_stride_for_at(&self, mb_x: u32, mb_y: u32) -> usize {
        self.luma_stride() * if self.is_mb_field_at(mb_x, mb_y) { 2 } else { 1 }
    }

    /// Chroma counterpart of [`Self::luma_row_stride_for`].
    pub fn chroma_row_stride_for(&self, mb_y: u32) -> usize {
        self.chroma_row_stride_for_at(0, mb_y)
    }

    /// Column-aware variant of [`Self::chroma_row_stride_for`].
    pub fn chroma_row_stride_for_at(&self, mb_x: u32, mb_y: u32) -> usize {
        self.chroma_stride() * if self.is_mb_field_at(mb_x, mb_y) { 2 } else { 1 }
    }

    /// True when the pair at (mb_x, mb_y) is field-coded. The current
    /// MVP uses `mb_x = 0` as a proxy (queries keyed on just `mb_y`),
    /// which is correct for uniformly-field-coded pictures. Mixed-mode
    /// pictures still need the more precise `is_mb_field_at(mb_x, mb_y)`
    /// variant below.
    pub fn is_mb_field(&self, mb_y: u32) -> bool {
        self.is_mb_field_at(0, mb_y)
    }

    /// True when the specific MB at `(mb_x, mb_y)` is field-coded. Per
    /// §7.3.4 both MBs of a pair carry the same flag so the column
    /// affects the result only on mixed-mode MBAFF pictures.
    pub fn is_mb_field_at(&self, mb_x: u32, mb_y: u32) -> bool {
        if !self.mbaff_enabled {
            return false;
        }
        let idx = (mb_y * self.mb_width + mb_x) as usize;
        self.mb_info
            .get(idx)
            .map(|m| m.mb_field_decoding_flag)
            .unwrap_or(false)
    }

    /// Whether MB `(mb_x, mb_y)` has an "above" neighbour in the §6.4.9.4
    /// sense. Non-MBAFF and frame-coded MBAFF pairs: true iff `mb_y > 0`.
    /// Field-coded MBAFF pairs: both top and bottom MBs look at the
    /// **previous pair** (same-polarity field), so availability requires
    /// `pair_y > 0`.
    pub fn mb_top_available(&self, mb_y: u32) -> bool {
        if !self.mbaff_enabled {
            return mb_y > 0;
        }
        if self.is_mb_field(mb_y) {
            (mb_y / 2) > 0
        } else {
            mb_y > 0
        }
    }

    /// Index of the "above" neighbour MB for `mb_y` per §6.4.9.4.
    /// Returns `None` when unavailable. For non-MBAFF and frame-coded
    /// MBAFF pairs the above neighbour is `mb_y - 1`. For field-coded
    /// MBAFF pairs the neighbour sits in the **previous pair** at the
    /// same field polarity, so `mb_y - 2`.
    pub fn mb_above_neighbour(&self, mb_y: u32) -> Option<u32> {
        if !self.mb_top_available(mb_y) {
            return None;
        }
        if self.mbaff_enabled && self.is_mb_field(mb_y) {
            // Top MB of pair P (which=0): above = top MB of pair P-1
            //   → mb_y - 2.
            // Bottom MB of pair P (which=1): above = bottom MB of pair
            //   P-1 → mb_y - 2.
            Some(mb_y - 2)
        } else {
            Some(mb_y - 1)
        }
    }

    /// (first luma plane row, row stride in rows) for MB row `mb_y`.
    /// Under MBAFF with a field-coded pair the top MB sits on even plane
    /// rows (row_stride_in_rows = 2) and the bottom MB on odd rows.
    /// Frame-coded or non-MBAFF MBs return `(mb_y*16, 1)`.
    pub fn mb_luma_row_info(&self, mb_y: u32) -> (usize, usize) {
        self.mb_luma_row_info_at(0, mb_y)
    }

    /// Column-aware variant of [`Self::mb_luma_row_info`].
    pub fn mb_luma_row_info_at(&self, mb_x: u32, mb_y: u32) -> (usize, usize) {
        if self.mbaff_enabled && self.is_mb_field_at(mb_x, mb_y) {
            let pair_y = (mb_y / 2) as usize;
            let which = (mb_y & 1) as usize;
            (pair_y * 32 + which, 2)
        } else {
            ((mb_y as usize) * 16, 1)
        }
    }

    /// Chroma counterpart of [`Self::mb_luma_row_info`]. Returns `(first
    /// chroma plane row, row-step-in-rows)`. For 4:2:0 the chroma MB is
    /// 8 rows tall per frame-MB and steps by 2 under field-coded MBAFF.
    pub fn mb_chroma_row_info(&self, mb_y: u32) -> (usize, usize) {
        self.mb_chroma_row_info_at(0, mb_y)
    }

    /// Column-aware variant of [`Self::mb_chroma_row_info`].
    pub fn mb_chroma_row_info_at(&self, mb_x: u32, mb_y: u32) -> (usize, usize) {
        let mbh = self.mb_height_chroma();
        if self.mbaff_enabled && self.is_mb_field_at(mb_x, mb_y) {
            let pair_y = (mb_y / 2) as usize;
            let which = (mb_y & 1) as usize;
            (pair_y * (mbh * 2) + which, 2)
        } else {
            ((mb_y as usize) * mbh, 1)
        }
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
            2 => PixelFormat::Yuv422P,
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

        // §7.4.2.1.1 — pick a PixelFormat variant that matches the luma
        // bit depth stored in `y16` / `cb16` / `cr16`. The u16 LE packing
        // is identical across depths; the enum variant only annotates the
        // valid sample range so downstream consumers know the
        // Clip1Y bound. oxideav-core 0.0 exposes 10-bit and 12-bit 4:2:0
        // variants but not 14-bit; the 14-bit path therefore reuses
        // `Yuv420P10Le` as a 16-bit-container fallback — the actual u16
        // words still carry full 14-bit samples (0..=16383).
        let format = match (self.chroma_format_idc, self.bit_depth_y) {
            (3, _) => PixelFormat::Yuv444P10Le,
            (_, 12) => PixelFormat::Yuv420P12Le,
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
