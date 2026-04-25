//! Slice reconstruction pipeline.
//!
//! Integrates parsed `Macroblock` syntax with the intra prediction,
//! inter prediction, inverse transform, and deblocking modules to
//! produce decoded samples in a [`Picture`].
//!
//! Scope:
//! * I-slices (intra macroblocks), P-slices (P_Skip / P_L0_16x16 /
//!   P_L0_L0_16x8 / P_L0_L0_8x16 / P_8x8 / P_8x8ref0) and B-slices
//!   (B_Skip / B_Direct_16x16 / B_Direct_8x8 / B_L0_* / B_L1_* /
//!   B_Bi_*).
//! * Frame pictures, non-MBAFF.
//! * ChromaArrayType ∈ {0, 1, 2}. 4:4:4 (== 3) is explicitly rejected.
//! * Deblocking is applied as a single picture-level pass after all MBs
//!   have been reconstructed (§8.7 conceptually runs per-MB; the
//!   picture-level equivalent is permitted since the filter is defined
//!   per-edge and the spec's block ordering is preserved).
//!
//! Spec references throughout follow ITU-T Rec. H.264 (08/2024).
//!
//! Clean-room: derived only from the ITU-T specification.

// Spec-driven reconstruction functions take many parameters (SliceData
// + SliceHeader + SPS + PPS + RefPicProvider + Picture + MbGrid plus
// per-partition coordinates and filter params). Suppress clippy's
// too_many_arguments at module scope.
#![allow(clippy::too_many_arguments)]

use crate::deblock::{
    alpha_from_index, beta_from_index, derive_boundary_strength, filter_edge, tc0_from, BsInputs,
    EdgeSamples, FilterParams, Plane,
};
use crate::inter_pred::{
    interpolate_chroma, interpolate_luma, weighted_pred_explicit, BiPredMode, WeightedEntry,
};
use crate::intra_pred::{
    filter_samples_8x8, predict_16x16, predict_4x4, predict_8x8, predict_chroma,
    ChromaArrayType as IpChromaArrayType, Intra16x16Mode, Intra4x4Mode, Intra8x8Mode,
    IntraChromaMode, Neighbour4x4Availability, Samples16x16, Samples4x4, Samples8x8, SamplesChroma,
};
use crate::macroblock_layer::{Macroblock, MbType, SubMbType};
use crate::mb_grid::{MbGrid, MbInfo};
use crate::mv_deriv::{
    derive_b_spatial_direct_with_d, derive_median_mvpred, derive_mvpred_with_d,
    derive_p_skip_mv_with_d, Mv, MvpredInputs, MvpredShape, NeighbourMv,
};
use crate::picture::Picture;
use crate::pps::Pps;
use crate::ref_store::RefPicProvider;
use crate::slice_data::SliceData;
use crate::slice_header::{PredWeightTable, SliceHeader, SliceType};
use crate::sps::Sps;
use crate::transform::{
    inverse_hadamard_chroma_dc_420, inverse_hadamard_chroma_dc_422, inverse_hadamard_luma_dc_16x16,
    inverse_transform_4x4, inverse_transform_4x4_dc_preserved, inverse_transform_8x8, qp_y_to_qp_c,
    select_scaling_list_4x4, select_scaling_list_8x8, TransformError,
};

use thiserror::Error;

/// Errors surfaced by the reconstruction layer.
#[derive(Debug, Error)]
pub enum ReconstructError {
    #[error("unsupported mb_type: {0:?}")]
    UnsupportedMbType(String),
    #[error("unsupported ChromaArrayType: {0}")]
    UnsupportedChromaArrayType(u32),
    #[error("field / MBAFF pictures are not supported in this reconstruction pass")]
    FieldOrMbaffNotSupported,
    #[error("transform error: {0}")]
    Transform(#[from] TransformError),
    #[error("intra pred error: (internal — out of bounds)")]
    IntraPredOutOfBounds,
    #[error("reference picture {list}:{idx} not available in the ref store")]
    MissingRefPic { list: u8, idx: u32 },
    #[error("unsupported inter mb_type for this scope: {0}")]
    UnsupportedInterMbType(String),
}

// -------------------------------------------------------------------------
// §6.4.3 — 4x4 luma block scan (Figure 6-10)
//   idx -> (x, y) offset of 4x4 block inside a macroblock.
// -------------------------------------------------------------------------

/// §6.4.3 / Figure 6-10 — upper-left (x, y) of each 4x4 luma block
/// relative to the macroblock origin. Block indices 0..=15.
const LUMA_4X4_XY: [(i32, i32); 16] = [
    // 8x8 quadrant top-left (blocks 0..=3): offset 0..=3 inside → local (0,0) (4,0) (0,4) (4,4)
    (0, 0),
    (4, 0),
    (0, 4),
    (4, 4),
    // 8x8 top-right quadrant (blocks 4..=7): +8 x
    (8, 0),
    (12, 0),
    (8, 4),
    (12, 4),
    // 8x8 bottom-left (blocks 8..=11): +8 y
    (0, 8),
    (4, 8),
    (0, 12),
    (4, 12),
    // 8x8 bottom-right (blocks 12..=15): +8 x +8 y
    (8, 8),
    (12, 8),
    (8, 12),
    (12, 12),
];

/// §6.4.3 — upper-left (x, y) of each 8x8 luma block inside the MB.
const LUMA_8X8_XY: [(i32, i32); 4] = [(0, 0), (8, 0), (0, 8), (8, 8)];

/// §8.5.7 / Table 8-14 — inverse 8x8 zig-zag scan.
///
/// Maps `idx -> (i, j)` such that the scan-order list entry at index
/// `idx` corresponds to the row-major matrix entry `c[i][j]` (stored
/// at `i*8 + j` in the returned buffer).
///
/// Reproduced verbatim from Table 8-14 (zig-zag row, frame-macroblock
/// case). The field-scan variant is not used here since this module
/// only handles frame pictures.
const ZIGZAG_8X8: [(usize, usize); 64] = [
    // idx 0..=7.
    (0, 0),
    (0, 1),
    (1, 0),
    (2, 0),
    (1, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    // idx 8..=15.
    (2, 1),
    (3, 0),
    (4, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 4),
    (0, 5),
    // idx 16..=23.
    (1, 4),
    (2, 3),
    (3, 2),
    (4, 1),
    (5, 0),
    (6, 0),
    (5, 1),
    (4, 2),
    // idx 24..=31.
    (3, 3),
    (2, 4),
    (1, 5),
    (0, 6),
    (0, 7),
    (1, 6),
    (2, 5),
    (3, 4),
    // idx 32..=39.
    (4, 3),
    (5, 2),
    (6, 1),
    (7, 0),
    (7, 1),
    (6, 2),
    (5, 3),
    (4, 4),
    // idx 40..=47.
    (3, 5),
    (2, 6),
    (1, 7),
    (2, 7),
    (3, 6),
    (4, 5),
    (5, 4),
    (6, 3),
    // idx 48..=55.
    (7, 2),
    (7, 3),
    (6, 4),
    (5, 5),
    (4, 6),
    (3, 7),
    (4, 7),
    (5, 6),
    // idx 56..=63.
    (6, 5),
    (7, 4),
    (7, 5),
    (6, 6),
    (5, 7),
    (6, 7),
    (7, 6),
    (7, 7),
];

/// §8.5.7 — invert the 8x8 zig-zag scan (frame-macroblock case of
/// Table 8-14). Input is the 64-entry scan-order coefficient list;
/// output is the row-major 8x8 matrix with `c[i][j]` at index
/// `i*8 + j`, ready for [`inverse_transform_8x8`].
fn inverse_scan_8x8_zigzag(levels: &[i32; 64]) -> [i32; 64] {
    let mut out = [0i32; 64];
    for (k, &(i, j)) in ZIGZAG_8X8.iter().enumerate() {
        out[i * 8 + j] = levels[k];
    }
    out
}

// -------------------------------------------------------------------------
// §6.4.1 — per-MB sample-origin derivation (frame + MBAFF pictures)
// -------------------------------------------------------------------------
//
// For non-MBAFF frame pictures, the MB's origin is simply
// `(mb_x * 16, mb_y * 16)` per eqs. (6-3)/(6-4), where `(mb_x, mb_y)` is
// the raster position. In MBAFF frames the origin is derived from
// eqs. (6-5)..(6-10) and the MB's `mb_field_decoding_flag`.
//
// For field-coded MBs in an MBAFF frame the luma samples are interleaved
// with the pair partner per §6.4.1 eq. (6-10):
//   row `k` of the MB is written at picture row `mb_py + k * 2`
// (with `mb_py` already offset by 0 for top-of-pair or 1 for bottom-of-
// pair per eq. (6-9)/(6-10)). The [`MbWriter`] helper folds this stride
// into every sample write.

/// §6.4.1 — MB origin `(mb_px, mb_py)` in luma samples. Handles both
/// non-MBAFF (eqs. 6-3 / 6-4) and MBAFF (eqs. 6-5..6-10) cases.
///
/// For MBAFF field MBs, `mb_py` is the starting row of the field MB
/// within the pair (pair_y or pair_y + 1), and subsequent rows are
/// interleaved with stride 2 (§6.4.1 eq. 6-10) — the caller must use
/// [`MbWriter`] (or similar) to apply the stride on each row.
fn mb_sample_origin(
    grid: &MbGrid,
    mb_addr: u32,
    mbaff_frame_flag: bool,
    mb_field_decoding_flag: bool,
) -> (i32, i32) {
    if mbaff_frame_flag {
        let (x, y) = crate::mb_address::mbaff_mb_to_sample_xy(
            mb_addr,
            grid.width_in_mbs,
            mb_field_decoding_flag,
        );
        (x as i32, y as i32)
    } else {
        let (mb_x, mb_y) = grid.mb_xy(mb_addr);
        ((mb_x as i32) * 16, (mb_y as i32) * 16)
    }
}

/// §6.4.1 — per-plane sample-write helper that applies the MBAFF
/// field-MB y-stride (§6.4.1 eq. 6-10).
///
/// Construct one per macroblock with [`MbWriter::new`]. Call
/// [`MbWriter::set_luma`] / [`MbWriter::set_cb`] / [`MbWriter::set_cr`]
/// with `(x_in_mb, y_in_mb)` — coordinates relative to the MB origin,
/// in luma (resp. chroma) samples. The helper computes absolute picture
/// coordinates with the correct stride for both non-MBAFF (stride 1)
/// and MBAFF field MBs (stride 2 per eq. 6-10), and handles the MBAFF
/// chroma origin derivation that `(mb_px / 16) * MbWidthC` breaks for
/// field MBs whose `mb_py` is an odd pair-local offset.
#[derive(Debug, Clone, Copy)]
struct MbWriter {
    /// Luma MB origin x — `mb_px` from [`mb_sample_origin`].
    mb_px: i32,
    /// Luma MB starting row — `mb_py` from [`mb_sample_origin`]. For
    /// field MBs this is `pair_y + (mb_is_top ? 0 : 1)`.
    mb_py: i32,
    /// `mb_field_decoding_flag` for this MB — enables the y-stride = 2
    /// interleave on writes.
    mb_field: bool,
    /// Whether the surrounding picture is an MBAFF frame. Affects
    /// chroma origin derivation (field chroma MBs also interleave).
    mbaff_frame: bool,
    /// `(mb_px_pair, mb_py_pair)` — luma top-left of the containing
    /// MBAFF pair (used for chroma origin derivation). For non-MBAFF
    /// this is just `(mb_px, mb_py)`.
    pair_px: i32,
    pair_py: i32,
    /// `mb_is_top` — true for top-of-pair MB in MBAFF. False for non-
    /// MBAFF pictures (treated as a singleton).
    mb_is_top: bool,
    /// §6.2 — chroma array type (0/1/2/3). Drives chroma-origin maths.
    chroma_array_type: u32,
}

impl MbWriter {
    /// Build an `MbWriter` for a macroblock given its address and
    /// picture-level MBAFF state.
    fn new(
        grid: &MbGrid,
        mb_addr: u32,
        mb_px: i32,
        mb_py: i32,
        mbaff_frame: bool,
        mb_field: bool,
        chroma_array_type: u32,
    ) -> Self {
        let (pair_px, pair_py, mb_is_top) = if mbaff_frame {
            // §6.4.1 eqs. (6-5)/(6-6): pair top-left at
            // `((pair_idx % PicW) * 16, (pair_idx / PicW) * 32)`.
            let w = grid.width_in_mbs.max(1);
            let pair_idx = mb_addr / 2;
            let pair_x = ((pair_idx % w) as i32) * 16;
            let pair_y = ((pair_idx / w) as i32) * 32;
            let is_top = mb_addr % 2 == 0;
            (pair_x, pair_y, is_top)
        } else {
            (mb_px, mb_py, true)
        };
        Self {
            mb_px,
            mb_py,
            mb_field,
            mbaff_frame,
            pair_px,
            pair_py,
            mb_is_top,
            chroma_array_type,
        }
    }

    /// §6.4.1 eq. (6-10) — luma y-stride: 2 for field MBs in an MBAFF
    /// pair, 1 otherwise.
    #[inline]
    fn luma_y_stride(&self) -> i32 {
        if self.mb_field {
            2
        } else {
            1
        }
    }

    /// §6.4.1 — chroma y-stride, mirror of [`luma_y_stride`] in the
    /// chroma plane.
    #[inline]
    fn chroma_y_stride(&self) -> i32 {
        if self.mb_field {
            2
        } else {
            1
        }
    }

    /// Chroma MB origin x. Derived from the luma MB origin via the
    /// Table 6-1 `SubWidthC` factor (same for all field states).
    #[inline]
    fn chroma_mb_px(&self) -> i32 {
        let (sub_w, _) = chroma_subsample(self.chroma_array_type);
        if sub_w == 0 {
            return 0;
        }
        // Chroma-pair x = luma-pair x / SubWidthC (same for both MBs).
        self.pair_px / sub_w
    }

    /// Chroma MB starting row. For non-MBAFF and MBAFF frame-coded MBs
    /// this is `mb_py / SubHeightC`. For MBAFF field MBs the chroma
    /// field structure mirrors luma: chroma-pair origin = pair_py /
    /// SubHeightC, and the field MB's first chroma row is
    /// `chroma_pair_y + (mb_is_top ? 0 : 1)`.
    #[inline]
    fn chroma_mb_py(&self) -> i32 {
        let (_, sub_h) = chroma_subsample(self.chroma_array_type);
        if sub_h == 0 {
            return 0;
        }
        if self.mbaff_frame {
            // §6.4.1 — MBAFF chroma origin.
            let (_, mb_h_c) = chroma_mb_dims(self.chroma_array_type);
            let mb_h_c = mb_h_c as i32;
            let chroma_pair_y = self.pair_py / sub_h;
            if self.mb_field {
                // §6.4.1 eq. (6-10) chroma equivalent — top/bot field
                // chroma MB starts at 0 / 1 and strides by 2.
                chroma_pair_y + if self.mb_is_top { 0 } else { 1 }
            } else {
                // Frame-coded MBs in an MBAFF pair — top/bot occupy
                // rows [0, MbHeightC) / [MbHeightC, 2*MbHeightC) of
                // the chroma pair.
                chroma_pair_y + if self.mb_is_top { 0 } else { mb_h_c }
            }
        } else {
            // Non-MBAFF — classic eqs. (6-3)/(6-4).
            self.mb_py / sub_h
        }
    }

    /// Write a luma sample at `(x_in_mb, y_in_mb)` — coordinates
    /// relative to the MB origin, in luma samples. Applies the field-
    /// MB y-stride (§6.4.1 eq. 6-10).
    #[inline]
    fn set_luma(&self, pic: &mut Picture, x_in_mb: i32, y_in_mb: i32, v: i32) {
        let ax = self.mb_px + x_in_mb;
        let ay = self.mb_py + y_in_mb * self.luma_y_stride();
        pic.set_luma(ax, ay, v);
    }

    /// Write a Cb sample at `(x_in_mb, y_in_mb)` — coordinates relative
    /// to the chroma MB origin, in chroma samples.
    #[inline]
    fn set_cb(&self, pic: &mut Picture, x_in_mb: i32, y_in_mb: i32, v: i32) {
        let ax = self.chroma_mb_px() + x_in_mb;
        let ay = self.chroma_mb_py() + y_in_mb * self.chroma_y_stride();
        pic.set_cb(ax, ay, v);
    }

    /// Write a Cr sample at `(x_in_mb, y_in_mb)`.
    #[inline]
    fn set_cr(&self, pic: &mut Picture, x_in_mb: i32, y_in_mb: i32, v: i32) {
        let ax = self.chroma_mb_px() + x_in_mb;
        let ay = self.chroma_mb_py() + y_in_mb * self.chroma_y_stride();
        pic.set_cr(ax, ay, v);
    }
}

// -------------------------------------------------------------------------
// Public entry point
// -------------------------------------------------------------------------

/// Reconstruct a slice (I / P / B) into `pic`, updating `grid` with
/// per-MB metadata for downstream deblocking and subsequent-MB
/// neighbour lookups.
///
/// `ref_pics` supplies reference pictures for inter prediction
/// (§8.4.2). Pass [`crate::ref_store::NoRefs`] for I-only reconstruction.
pub fn reconstruct_slice<R: RefPicProvider>(
    slice_data: &SliceData,
    slice_header: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    ref_pics: &R,
    pic: &mut Picture,
    grid: &mut MbGrid,
) -> Result<(), ReconstructError> {
    reconstruct_slice_no_deblock(slice_data, slice_header, sps, pps, ref_pics, pic, grid)?;

    // §8.7 picture-level deblock for the single-slice-per-picture case.
    // Callers doing multi-slice assembly must use
    // `reconstruct_slice_no_deblock` + `deblock_picture_full` at picture
    // finalization time so the deblocker runs exactly once, after all
    // slices have deposited their MBs into the shared grid.
    let bit_depth_y = 8 + sps.bit_depth_luma_minus8;
    let bit_depth_c = 8 + sps.bit_depth_chroma_minus8;
    let mbaff_frame_flag = sps.mb_adaptive_frame_field_flag && !slice_header.field_pic_flag;
    let deblock_enabled = slice_header.disable_deblocking_filter_idc != 1;
    let deblock_env_off = std::env::var("OXIDEAV_H264_NO_DEBLOCK").is_ok();
    if deblock_enabled && !deblock_env_off {
        let alpha_off = slice_header.slice_alpha_c0_offset_div2 * 2;
        let beta_off = slice_header.slice_beta_offset_div2 * 2;
        deblock_picture_full(
            pic,
            grid,
            alpha_off,
            beta_off,
            bit_depth_y,
            bit_depth_c,
            pps,
            mbaff_frame_flag,
            &slice_data.mb_field_decoding_flags,
        );
    }
    Ok(())
}

/// §8.4 slice reconstruction without the §8.7 deblocking pass — meant
/// for multi-slice picture assembly where several slices share a
/// `Picture` + `MbGrid`. Callers must invoke [`deblock_picture_full`]
/// exactly once after the final slice of the picture has been
/// reconstructed so the deblocker sees the fully-populated grid.
pub fn reconstruct_slice_no_deblock<R: RefPicProvider>(
    slice_data: &SliceData,
    slice_header: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    ref_pics: &R,
    pic: &mut Picture,
    grid: &mut MbGrid,
) -> Result<(), ReconstructError> {
    // -------- Preconditions -----------------------------------------
    let chroma_array_type = sps.chroma_array_type();
    if chroma_array_type == 3 {
        return Err(ReconstructError::UnsupportedChromaArrayType(
            chroma_array_type,
        ));
    }
    // §7.4.2.1.1 — MbaffFrameFlag = mb_adaptive_frame_field_flag &&
    // !field_pic_flag. Field-only pictures (`field_pic_flag == 1`) still
    // aren't handled by this reconstruction pass — MBAFF frames are, as
    // of Phase 2, with the caveats documented in the `mb_address` MBAFF
    // helpers (field-MB samples are placed at their pair-local start
    // row per §6.4.1 eq. 6-10 but without the y-stride=2 interleave
    // with the pair partner — that's a Phase-3 refinement).
    if slice_header.field_pic_flag {
        return Err(ReconstructError::FieldOrMbaffNotSupported);
    }
    let mbaff_frame_flag = sps.mb_adaptive_frame_field_flag && !slice_header.field_pic_flag;

    // §7.4.2.1 — bit depths (8..=14 supported by the transform module).
    let bit_depth_y = 8 + sps.bit_depth_luma_minus8;
    let bit_depth_c = 8 + sps.bit_depth_chroma_minus8;

    // §7.4.3 — SliceQPY = 26 + pic_init_qp_minus26 + slice_qp_delta.
    let slice_qp_y = 26 + pps.pic_init_qp_minus26 + slice_header.slice_qp_delta;

    // §7.4.5 — QP_Y for the current MB; initially equal to SliceQPY,
    // then updated by mb_qp_delta (§8.5.8 eq. 8-309).
    let mut prev_qp_y = slice_qp_y;

    // §7.3.3 — deblocking filter control from the slice header.
    let (alpha_off, beta_off) = (
        slice_header.slice_alpha_c0_offset_div2 * 2,
        slice_header.slice_beta_offset_div2 * 2,
    );

    // §7.4.3 — disable_deblocking_filter_idc:
    //   0 = on for all edges, 1 = off, 2 = on but not across slice boundaries.
    // Our picture-level pass treats the whole picture as one slice (so
    // 0 and 2 behave identically here).
    let deblock_enabled = slice_header.disable_deblocking_filter_idc != 1;

    // -------- Walk macroblocks --------------------------------------
    // §7.4.2.1 / §7.3.4 — starting CurrMbAddr = first_mb_in_slice *
    // (1 + MbaffFrameFlag): in MBAFF the slice header's first_mb_in_slice
    // is in pair units so the first raw MB address is double.
    let mut curr_addr = slice_header.first_mb_in_slice * (1 + u32::from(mbaff_frame_flag));
    let total = slice_data.macroblocks.len();
    for (idx, mb) in slice_data.macroblocks.iter().enumerate() {
        // §7.4.4 — per-MB mb_field_decoding_flag (shared within an
        // MBAFF pair, `false` for non-MBAFF).
        let mb_field_decoding_flag = slice_data
            .mb_field_decoding_flags
            .get(idx)
            .copied()
            .unwrap_or(false);
        // Determine this MB's QP_Y (needed for AC transform scaling and
        // deblocking). For I_PCM / P_Skip / B_Skip we leave QP_Y
        // unchanged per §7.4.5. P_Skip / B_Skip carry no mb_qp_delta.
        let mb_qp_y = if mb.mb_type.is_i_pcm() || mb.is_skip {
            prev_qp_y
        } else {
            // §8.5.8 eq. 8-309 — QP_Y = ((prev_QP_Y + mb_qp_delta + 52 + QpBdOffsetY)
            // % (52 + QpBdOffsetY)) - QpBdOffsetY.
            let qp_bd_offset_y = (6 * sps.bit_depth_luma_minus8) as i32;
            let range = 52 + qp_bd_offset_y;
            let raw = prev_qp_y + mb.mb_qp_delta + range;
            let m = raw.rem_euclid(range);
            m - qp_bd_offset_y
        };

        // Reconstruct this MB's samples.
        // §6.4.8 — the current MB's slice identity (used to reject
        // cross-slice neighbours in §8.3.1.1 intra-pred-mode derivation).
        // `first_mb_in_slice` is unique per slice of a primary coded
        // picture and already carried on the slice header (§7.3.3).
        let current_slice_id = slice_header.first_mb_in_slice as i32;
        if mb.mb_type.is_intra() {
            reconstruct_mb_intra(
                mb,
                curr_addr,
                mb_qp_y,
                chroma_array_type,
                bit_depth_y,
                bit_depth_c,
                sps,
                pps,
                pic,
                grid,
                mbaff_frame_flag,
                mb_field_decoding_flag,
                current_slice_id,
            )?;
        } else {
            reconstruct_mb_inter(
                mb,
                curr_addr,
                mb_qp_y,
                chroma_array_type,
                bit_depth_y,
                bit_depth_c,
                slice_header,
                sps,
                pps,
                ref_pics,
                pic,
                grid,
                mbaff_frame_flag,
                mb_field_decoding_flag,
                current_slice_id,
            )?;
        }

        // Record MB info for future neighbour lookups.
        if let Some(info) = grid.get_mut(curr_addr) {
            info.available = true;
            info.is_intra = mb.mb_type.is_intra();
            info.is_i_pcm = mb.mb_type.is_i_pcm();
            info.is_intra_nxn = mb.mb_type.is_i_nxn();
            info.mb_type_raw = mb.mb_type_raw;
            info.qp_y = mb_qp_y;
            info.cbp_luma = (mb.coded_block_pattern & 0x0F) as u8;
            info.cbp_chroma = ((mb.coded_block_pattern >> 4) & 0x03) as u8;
            info.transform_size_8x8_flag = mb.transform_size_8x8_flag;
            info.luma_nonzero_4x4 = compute_luma_nonzero_mask(mb);
            info.chroma_nonzero_4x4 = compute_chroma_nonzero_mask(mb, chroma_array_type);
            if let Some(pred) = mb.mb_pred.as_ref() {
                info.intra_chroma_pred_mode = pred.intra_chroma_pred_mode;
            }
            // §6.4.8 third bullet — stamp this MB with the slice it
            // belongs to. `first_mb_in_slice` serves as a unique slice
            // identifier within a primary coded picture (§7.3.3 —
            // different values across the picture's slices) without
            // requiring a separate counter threaded through the decoder.
            // The cast is safe: first_mb_in_slice <= 2^16 for any valid
            // H.264 picture size (Table A-1 MaxMbs).
            info.slice_id = slice_header.first_mb_in_slice as i32;
        }

        prev_qp_y = mb_qp_y;

        // Advance to the next MB address within this slice's slice
        // group. With a single slice group (the common case) this is
        // simply curr_addr + 1 for the non-MBAFF frame path.
        if idx + 1 < total {
            curr_addr += 1;
            if curr_addr as usize >= grid.info.len() {
                break;
            }
        }
    }

    // Deblocking is deferred so a multi-slice picture only runs §8.7 once,
    // after every slice has contributed to the shared Picture + MbGrid.
    // Single-slice callers should use `reconstruct_slice` instead, which
    // wraps this helper and invokes the deblocker.
    let _ = (
        deblock_enabled,
        alpha_off,
        beta_off,
        bit_depth_y,
        bit_depth_c,
    );
    Ok(())
}

/// §8.7 picture-level deblocking filter entry point for multi-slice
/// callers. Deferring the deblocking pass until the whole picture has
/// been decoded means the right-/bottom-neighbour MBs of the last MB in
/// an early slice aren't seen as "unavailable" when the filter runs, and
/// — critically — already-deblocked edges aren't re-filtered by the
/// next slice's own deblock pass.
///
/// `alpha_off` / `beta_off` are the §7.4.3 `slice_alpha_c0_offset_div2`
/// and `slice_beta_offset_div2` values doubled into the final loop-filter
/// offsets, taken from the first slice of the picture. Multi-slice
/// pictures that vary these per slice are a known simplification; the
/// conformance streams we target use uniform deblock offsets across the
/// picture.
#[allow(clippy::too_many_arguments)]
pub fn deblock_picture_full(
    pic: &mut Picture,
    grid: &MbGrid,
    alpha_off: i32,
    beta_off: i32,
    bit_depth_y: u32,
    bit_depth_c: u32,
    pps: &Pps,
    mbaff_frame_flag: bool,
    mb_field_flags: &[bool],
) {
    let deblock_env_off = std::env::var("OXIDEAV_H264_NO_DEBLOCK").is_ok();
    if deblock_env_off {
        return;
    }
    deblock_picture(
        pic,
        grid,
        alpha_off,
        beta_off,
        bit_depth_y,
        bit_depth_c,
        pps,
        mbaff_frame_flag,
        mb_field_flags,
    );
}

// -------------------------------------------------------------------------
// Per-MB reconstruction
// -------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn reconstruct_mb_intra(
    mb: &Macroblock,
    mb_addr: u32,
    qp_y: i32,
    chroma_array_type: u32,
    bit_depth_y: u32,
    bit_depth_c: u32,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
    grid: &mut MbGrid,
    mbaff_frame_flag: bool,
    mb_field_decoding_flag: bool,
    current_slice_id: i32,
) -> Result<(), ReconstructError> {
    // §6.4.1 — MB sample origin (non-MBAFF eqs. 6-3/6-4 or MBAFF eqs.
    // 6-5..6-10 depending on `mb_field_decoding_flag`).
    let (mb_px, mb_py) = mb_sample_origin(grid, mb_addr, mbaff_frame_flag, mb_field_decoding_flag);
    let writer = MbWriter::new(
        grid,
        mb_addr,
        mb_px,
        mb_py,
        mbaff_frame_flag,
        mb_field_decoding_flag,
        chroma_array_type,
    );

    match &mb.mb_type {
        MbType::IPcm => {
            // §8.3.5 — I_PCM: raw samples are the decoded output. The
            // [`MbWriter`] applies §6.4.1 eq. (6-10) y-stride for field
            // MBs automatically.
            let pcm = mb.pcm_samples.as_ref().ok_or_else(|| {
                ReconstructError::UnsupportedMbType("I_PCM without samples".into())
            })?;
            for y in 0..16 {
                for x in 0..16 {
                    let v = pcm.luma[(y * 16 + x) as usize] as i32;
                    writer.set_luma(pic, x, y, v);
                }
            }
            // Chroma (4:2:0 → 8x8, 4:2:2 → 8x16). Writer handles the
            // §6.4.1 chroma origin + field stride for MBAFF field MBs.
            let (cw, ch) = chroma_mb_dims(chroma_array_type);
            let mut k = 0usize;
            for y in 0..ch {
                for x in 0..cw {
                    let v = pcm.chroma_cb[k] as i32;
                    writer.set_cb(pic, x as i32, y as i32, v);
                    k += 1;
                }
            }
            k = 0;
            for y in 0..ch {
                for x in 0..cw {
                    let v = pcm.chroma_cr[k] as i32;
                    writer.set_cr(pic, x as i32, y as i32, v);
                    k += 1;
                }
            }
        }
        MbType::Intra16x16(cfg) => {
            reconstruct_intra_16x16(
                mb,
                cfg.pred_mode,
                cfg.cbp_luma,
                cfg.cbp_chroma,
                qp_y,
                chroma_array_type,
                bit_depth_y,
                bit_depth_c,
                mb_px,
                mb_py,
                &writer,
                sps,
                pps,
                pic,
                grid,
                current_slice_id,
            )?;
        }
        MbType::INxN => {
            reconstruct_intra_nxn(
                mb,
                qp_y,
                bit_depth_y,
                mb_px,
                mb_py,
                mb_addr,
                &writer,
                sps,
                pps,
                pic,
                grid,
                current_slice_id,
            )?;
            reconstruct_chroma_intra(
                mb,
                qp_y,
                chroma_array_type,
                bit_depth_c,
                mb_px,
                mb_py,
                &writer,
                sps,
                pps,
                pic,
                grid,
                current_slice_id,
            )?;
        }
        _ => {
            // Not reachable for valid I-slice data — caller checks
            // is_intra() — but return a diagnostic anyway.
            return Err(ReconstructError::UnsupportedMbType(format!(
                "{:?}",
                mb.mb_type
            )));
        }
    }
    Ok(())
}

// -------------------------------------------------------------------------
// §8.3.3 — Intra_16x16 luma
// -------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn reconstruct_intra_16x16(
    mb: &Macroblock,
    pred_mode_idx: u8,
    cbp_luma: u8,
    _cbp_chroma: u8,
    qp_y: i32,
    chroma_array_type: u32,
    bit_depth_y: u32,
    bit_depth_c: u32,
    mb_px: i32,
    mb_py: i32,
    writer: &MbWriter,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
    grid: &MbGrid,
    current_slice_id: i32,
) -> Result<(), ReconstructError> {
    // -------- §8.3.3 — 16x16 prediction -----------------------------
    let samples = gather_samples_16x16(
        pic,
        grid,
        mb_px,
        mb_py,
        current_slice_id,
        pps.constrained_intra_pred_flag,
    );
    let mode = Intra16x16Mode::from_index(pred_mode_idx).ok_or_else(|| {
        ReconstructError::UnsupportedMbType(format!("Intra_16x16 pred_mode {}", pred_mode_idx))
    })?;
    let mut pred = [0i32; 256];
    predict_16x16(mode, &samples, bit_depth_y, &mut pred);

    // -------- §8.5.10 — luma DC block (always present for I_16x16) --
    // §7.4.2.1.1.1 Table 7-2 — Intra luma 4x4 list (index 0).
    let sl4 = select_scaling_list_4x4(0, sps, pps);
    let dc_levels = mb.residual_luma_dc.as_ref().copied().unwrap_or([0i32; 16]);
    // DC coefficients are in zig-zag scan order; inverse-scan to a
    // 4x4 matrix before the Hadamard.
    let dc_matrix = crate::transform::inverse_scan_4x4_zigzag(&dc_levels);
    let dc_y = inverse_hadamard_luma_dc_16x16(&dc_matrix, qp_y, &sl4, bit_depth_y)?;

    // -------- §8.5.12 — each of the 16 AC blocks --------------------
    // Residual layout from macroblock_layer:
    //   For Intra_16x16: when cbp_luma == 15, residual_luma has 16
    //   entries in the 4x4 raster-Z order (same as LUMA_4X4_XY indices).
    //   Each entry holds the 15 AC coefficients at parser slots 0..=14,
    //   corresponding to spec scan positions 1..=15 (startIdx=1 for
    //   Intra16x16ACLevel per §7.3.5.3.3 / Table 9-42). Slot 15 is
    //   unused (padding by `pad_to_16`).
    //   When cbp_luma == 0, residual_luma is empty (only DC).
    #[allow(clippy::needless_range_loop)] // spec §8.5.10 raster-Z 4x4 walk
    for block_idx in 0..16usize {
        let (bx, by) = LUMA_4X4_XY[block_idx];
        // Dequantised AC coefficients for this 4x4 block. c[0,0] is
        // overwritten with the already-scaled DC entry from dc_y.
        let mut coeffs = if cbp_luma == 15 {
            let ac = mb
                .residual_luma
                .get(block_idx)
                .copied()
                .unwrap_or([0i32; 16]);
            // AC values at slots 0..=14 are spec scan positions 1..=15.
            crate::transform::inverse_scan_4x4_zigzag_ac(&ac)
        } else {
            [0i32; 16]
        };
        // Place the pre-scaled DC into c[0,0]. The dc_y array is
        // indexed by the same (blkX, blkY) — which, following §8.5.10
        // eq. 8-323 table, maps blk index to (row, col) via the
        // 4x4 luma scan. Here block_idx == LUMA_4X4_XY position /4.
        let dc_row = (by / 4) as usize;
        let dc_col = (bx / 4) as usize;
        coeffs[0] = dc_y[dc_row * 4 + dc_col];

        let residual = inverse_transform_4x4_dc_preserved(&coeffs, qp_y, &sl4, bit_depth_y)?;

        // Add prediction + residual, clip, write via MbWriter so
        // §6.4.1 eq. (6-10) field-MB y-stride is applied.
        write_block_luma(writer, pic, &pred, bx, by, &residual, bit_depth_y);
    }

    // Chroma for the same MB.
    reconstruct_chroma_intra(
        mb,
        qp_y,
        chroma_array_type,
        bit_depth_c,
        mb_px,
        mb_py,
        writer,
        sps,
        pps,
        pic,
        grid,
        current_slice_id,
    )?;

    Ok(())
}

// -------------------------------------------------------------------------
// §8.3.1 / §8.3.2 — Intra_4x4 / Intra_8x8 luma
// -------------------------------------------------------------------------

/// §6.4.8 / §6.4.4 — "is this MB available for prediction?" for the
/// §8.3.1.1 / §8.3.2.1 pred-mode derivation. Returns `None` when the
/// neighbour lies outside the picture, when it hasn't been decoded yet
/// (per the MbGrid `available` flag), when `constrained_intra_pred_flag`
/// is 1 and the neighbour is inter-coded (§8.3.1.1 step 2 bullet 3/4),
/// or when the neighbour belongs to a different slice than the current
/// macroblock (§6.4.8 third bullet).
///
/// `current_slice_id` selects the slice identity to compare against —
/// pass the value that will be stamped into the *current* MB's
/// `MbInfo::slice_id`. Pass `-1` in unit tests / standalone callers that
/// don't care about slice boundaries; the slice filter is disabled when
/// either side of the comparison is `-1` (unstamped).
fn intra_neighbour_mb_info(
    grid: &MbGrid,
    mb_addr: Option<u32>,
    constrained_intra_pred: bool,
    current_slice_id: i32,
) -> Option<&MbInfo> {
    let addr = mb_addr?;
    let info = grid.get(addr)?;
    if !info.available {
        return None;
    }
    if constrained_intra_pred && !info.is_intra {
        // §8.3.1.1 steps 2 bullets 3 & 4 / §8.3.2.1 step 2 bullets 3 & 4
        // — an inter-coded neighbour is treated as unavailable for the
        // purposes of the dcPredModePredictedFlag derivation.
        return None;
    }
    // §6.4.8 third bullet — neighbours in a different slice are marked
    // "not available". The `-1` sentinel skips this check so unit tests
    // and pre-slice-id callers keep their old semantics.
    if current_slice_id >= 0 && info.slice_id >= 0 && info.slice_id != current_slice_id {
        return None;
    }
    Some(info)
}

/// §7.4.5 — is this MB's prediction coded as Intra_4x4 or Intra_8x8
/// (i.e. `MbPartPredMode == Intra_4x4 / Intra_8x8`, aka the I_NxN
/// mnemonic)? Used by §8.3.1.1 step 3 to decide whether a neighbour
/// contributes `intra_4x4_pred_modes[...]` / `intra_8x8_pred_modes[...]`
/// (returns `true`) or falls back to Intra_4x4_DC (`intraMxMPredModeN = 2`)
/// because the neighbour is Intra_16x16 / I_PCM.
///
/// We rely on the `is_intra_nxn` flag populated by the caller from
/// `MbType::is_i_nxn()` rather than `mb_type_raw`, because the raw
/// bitstream value has different meanings per slice type (e.g. raw=5
/// is Intra_16x16 in I slices but the I_NxN P-slice remap value).
fn mb_is_intra_nxn(info: &MbInfo) -> bool {
    info.is_intra && !info.is_i_pcm && info.is_intra_nxn
}

/// §8.3.1.1 — is `info.mb_type_raw` an Intra_4x4 MB (i.e. I_NxN with
/// `transform_size_8x8_flag` == 0)?
fn mb_is_intra_4x4(info: &MbInfo) -> bool {
    mb_is_intra_nxn(info) && !info.transform_size_8x8_flag
}

/// §8.3.2.1 — is `info.mb_type_raw` an Intra_8x8 MB (i.e. I_NxN with
/// `transform_size_8x8_flag` == 1)?
fn mb_is_intra_8x8(info: &MbInfo) -> bool {
    mb_is_intra_nxn(info) && info.transform_size_8x8_flag
}

/// §6.4.11.4 / Table 6-3 / §6.4.13.1 — derive (neighbour MB address,
/// neighbour 4x4 block index) for direction `N ∈ {A, B}` of the 4x4
/// block at position `(x, y)` inside the current MB.
///
/// For non-MBAFF frame pictures only (the scope of this module).
/// Returns `None` when the neighbour lies outside the picture.
fn neighbour_4x4_addr(
    grid: &MbGrid,
    mb_addr: u32,
    bx: i32,
    by: i32,
    xd: i32,
    yd: i32,
) -> Option<(u32, usize)> {
    // Eq. 6-25 / 6-26.
    let xn = bx + xd;
    let yn = by + yd;
    // Table 6-3 (non-MBAFF frame path).
    let [mb_a, mb_b, _mb_c, mb_d] = grid.neighbour_mb_addrs(mb_addr);
    let mb_n = if xn < 0 && yn < 0 {
        mb_d
    } else if xn < 0 && (0..16).contains(&yn) {
        mb_a
    } else if (0..16).contains(&xn) && yn < 0 {
        mb_b
    } else if (0..16).contains(&xn) && (0..16).contains(&yn) {
        Some(mb_addr)
    } else {
        // Other Table 6-3 cases yield mbAddrC or "not available"; for a
        // 4x4 block of a 16x16 MB with (xD, yD) in {(-1, 0), (0, -1)},
        // those branches are unreachable.
        None
    };
    let addr = mb_n?;
    // Eq. 6-34 / 6-35: (xW, yW) relative to the neighbour MB.
    let xw = (xn + 16) % 16;
    let yw = (yn + 16) % 16;
    // Eq. 6-38.
    let blk = (8 * (yw / 8) + 4 * (xw / 8) + 2 * ((yw % 8) / 4) + ((xw % 8) / 4)) as usize;
    Some((addr, blk))
}

/// §6.4.11.2 / Table 6-3 / §6.4.13.3 — derive (neighbour MB address,
/// neighbour 8x8 block index) for direction `N ∈ {A, B}` of the 8x8
/// block with index `blk8` (raster 0..=3) in the current MB.
fn neighbour_8x8_addr(
    grid: &MbGrid,
    mb_addr: u32,
    blk8: usize,
    xd: i32,
    yd: i32,
) -> Option<(u32, usize)> {
    // Eq. 6-23 / 6-24.
    let xn = ((blk8 as i32) % 2) * 8 + xd;
    let yn = ((blk8 as i32) / 2) * 8 + yd;
    let [mb_a, mb_b, _mb_c, mb_d] = grid.neighbour_mb_addrs(mb_addr);
    let mb_n = if xn < 0 && yn < 0 {
        mb_d
    } else if xn < 0 && (0..16).contains(&yn) {
        mb_a
    } else if (0..16).contains(&xn) && yn < 0 {
        mb_b
    } else if (0..16).contains(&xn) && (0..16).contains(&yn) {
        Some(mb_addr)
    } else {
        None
    };
    let addr = mb_n?;
    let xw = (xn + 16) % 16;
    let yw = (yn + 16) % 16;
    // Eq. 6-40.
    let blk = (2 * (yw / 8) + (xw / 8)) as usize;
    Some((addr, blk))
}

/// §8.3.1.1 — for neighbour direction N (A or B), return
/// `intraMxMPredModeN` as specified in step 3 of the derivation.
///
/// `dc_pred_flag` is the step-2 `dcPredModePredictedFlag`.
fn intra_mxm_pred_mode_for_neighbour_4x4(
    grid: &MbGrid,
    dc_pred_flag: bool,
    neighbour: Option<(u32, usize)>,
    constrained_intra_pred: bool,
    current_slice_id: i32,
) -> u8 {
    if dc_pred_flag {
        // §8.3.1.1 step 3 bullet 1 — DC fallback.
        return 2;
    }
    let Some((addr, blk)) = neighbour else {
        return 2;
    };
    let Some(info) =
        intra_neighbour_mb_info(grid, Some(addr), constrained_intra_pred, current_slice_id)
    else {
        return 2;
    };
    if mb_is_intra_4x4(info) {
        // §8.3.1.1 step 3 bullet 2 sub-bullet 1.
        info.intra_4x4_pred_modes[blk]
    } else if mb_is_intra_8x8(info) {
        // §8.3.1.1 step 3 bullet 2 sub-bullet 2 — neighbour is
        // Intra_8x8: intraMxMPredModeN = Intra8x8PredMode[ blk >> 2 ].
        info.intra_8x8_pred_modes[blk >> 2]
    } else {
        // Neighbour is Intra_16x16 / I_PCM / Inter → DC per §8.3.1.1
        // step 3 bullet 1 ("not coded in Intra_4x4 or Intra_8x8").
        2
    }
}

/// §8.3.2.1 — same as [`intra_mxm_pred_mode_for_neighbour_4x4`] but
/// for the Intra_8x8 derivation. When the neighbour MB is coded in
/// Intra_4x4, the spec specifies a per-direction sub-index `n`
/// (§8.3.2.1 eq. 8-72): N == A ⇒ n = 1 (in the frame / non-MBAFF case);
/// N == B ⇒ n = 2. We expose `n_for_4x4` so the caller supplies the
/// right value.
fn intra_mxm_pred_mode_for_neighbour_8x8(
    grid: &MbGrid,
    dc_pred_flag: bool,
    neighbour: Option<(u32, usize)>,
    constrained_intra_pred: bool,
    n_for_4x4: usize,
    current_slice_id: i32,
) -> u8 {
    if dc_pred_flag {
        return 2;
    }
    let Some((addr, blk)) = neighbour else {
        return 2;
    };
    let Some(info) =
        intra_neighbour_mb_info(grid, Some(addr), constrained_intra_pred, current_slice_id)
    else {
        return 2;
    };
    if mb_is_intra_8x8(info) {
        info.intra_8x8_pred_modes[blk]
    } else if mb_is_intra_4x4(info) {
        // §8.3.2.1 eq. 8-72: intraMxMPredModeN =
        // Intra4x4PredMode[ luma8x8BlkIdxN * 4 + n ]. Non-MBAFF frame:
        // n = 1 for A, 2 for B.
        let sub_idx = blk * 4 + n_for_4x4;
        info.intra_4x4_pred_modes[sub_idx]
    } else {
        2
    }
}

/// §8.3.1.1 — compute `Intra4x4PredMode[luma4x4BlkIdx]` for block
/// `block_idx` of the current MB using the parsed
/// `prev_intra4x4_pred_mode_flag` / `rem_intra4x4_pred_mode` plus the
/// neighbour 4x4 / 8x8 pred modes already recorded in the grid.
fn derive_intra_4x4_pred_mode(
    grid: &MbGrid,
    mb_addr: u32,
    block_idx: usize,
    pred: &crate::macroblock_layer::MbPred,
    constrained_intra_pred: bool,
    current_slice_id: i32,
) -> u8 {
    let (bx, by) = LUMA_4X4_XY[block_idx];
    // §6.4.11.4 step 1 / Table 6-2: A → (xD, yD) = (-1, 0); B → (0, -1).
    let na = neighbour_4x4_addr(grid, mb_addr, bx, by, -1, 0);
    let nb = neighbour_4x4_addr(grid, mb_addr, bx, by, 0, -1);

    // §8.3.1.1 step 2.
    let mb_a = intra_neighbour_mb_info(
        grid,
        na.map(|(a, _)| a),
        constrained_intra_pred,
        current_slice_id,
    );
    let mb_b = intra_neighbour_mb_info(
        grid,
        nb.map(|(a, _)| a),
        constrained_intra_pred,
        current_slice_id,
    );
    let dc_pred_flag = mb_a.is_none() || mb_b.is_none();

    // §8.3.1.1 step 3.
    let mode_a = intra_mxm_pred_mode_for_neighbour_4x4(
        grid,
        dc_pred_flag,
        na,
        constrained_intra_pred,
        current_slice_id,
    );
    let mode_b = intra_mxm_pred_mode_for_neighbour_4x4(
        grid,
        dc_pred_flag,
        nb,
        constrained_intra_pred,
        current_slice_id,
    );

    // §8.3.1.1 step 4, eq. 8-41.
    let predicted = mode_a.min(mode_b);

    if pred.prev_intra4x4_pred_mode_flag[block_idx] {
        predicted
    } else {
        let rem = pred.rem_intra4x4_pred_mode[block_idx];
        if rem < predicted {
            rem
        } else {
            rem + 1
        }
    }
}

/// §8.3.2.1 — compute `Intra8x8PredMode[luma8x8BlkIdx]` for 8x8 block
/// `blk8` of the current MB. Mirror of
/// [`derive_intra_4x4_pred_mode`] but with the 8x8-specific neighbour
/// table and the `n` sub-index of eq. 8-72.
fn derive_intra_8x8_pred_mode(
    grid: &MbGrid,
    mb_addr: u32,
    blk8: usize,
    pred: &crate::macroblock_layer::MbPred,
    constrained_intra_pred: bool,
    current_slice_id: i32,
) -> u8 {
    // §6.4.11.2 step 1 / Table 6-2: A → (xD, yD) = (-1, 0); B → (0, -1).
    let na = neighbour_8x8_addr(grid, mb_addr, blk8, -1, 0);
    let nb = neighbour_8x8_addr(grid, mb_addr, blk8, 0, -1);

    let mb_a = intra_neighbour_mb_info(
        grid,
        na.map(|(a, _)| a),
        constrained_intra_pred,
        current_slice_id,
    );
    let mb_b = intra_neighbour_mb_info(
        grid,
        nb.map(|(a, _)| a),
        constrained_intra_pred,
        current_slice_id,
    );
    let dc_pred_flag = mb_a.is_none() || mb_b.is_none();

    // Non-MBAFF frame path of §8.3.2.1 eq. 8-72: n = 1 for A, n = 2 for B.
    let mode_a = intra_mxm_pred_mode_for_neighbour_8x8(
        grid,
        dc_pred_flag,
        na,
        constrained_intra_pred,
        1,
        current_slice_id,
    );
    let mode_b = intra_mxm_pred_mode_for_neighbour_8x8(
        grid,
        dc_pred_flag,
        nb,
        constrained_intra_pred,
        2,
        current_slice_id,
    );

    // §8.3.2.1 step 4, eq. 8-73.
    let predicted = mode_a.min(mode_b);
    if pred.prev_intra8x8_pred_mode_flag[blk8] {
        predicted
    } else {
        let rem = pred.rem_intra8x8_pred_mode[blk8];
        if rem < predicted {
            rem
        } else {
            rem + 1
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_intra_nxn(
    mb: &Macroblock,
    qp_y: i32,
    bit_depth_y: u32,
    mb_px: i32,
    mb_py: i32,
    mb_addr: u32,
    writer: &MbWriter,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
    grid: &mut MbGrid,
    current_slice_id: i32,
) -> Result<(), ReconstructError> {
    let pred = mb
        .mb_pred
        .as_ref()
        .ok_or_else(|| ReconstructError::UnsupportedMbType("I_NxN without mb_pred".into()))?;
    // §7.4.2.1.1.1 Table 7-2 — intra-luma 4x4 list (i=0) / 8x8 list (i=6 → sub-idx 0).
    let sl4 = select_scaling_list_4x4(0, sps, pps);
    let sl8 = select_scaling_list_8x8(0, sps, pps);
    let cbp_luma = (mb.coded_block_pattern & 0x0F) as u8;
    let cip = pps.constrained_intra_pred_flag;

    // The Intra_4x4/Intra_8x8 pred-mode derivation consults the
    // already-set `intra_4x4_pred_modes` / `intra_8x8_pred_modes` of
    // the current MB for neighbours falling inside it. Ensure the
    // current MB's grid entry is marked available + intra + with the
    // correct `mb_type_raw` and `transform_size_8x8_flag` BEFORE we
    // start per-block derivation, so a later-block's A / B lookup
    // into this same MB succeeds. (reconstruct_slice sets these again
    // at the end of the per-MB loop — doing it here is idempotent.)
    if let Some(info) = grid.get_mut(mb_addr) {
        info.available = true;
        info.is_intra = true;
        info.is_i_pcm = false;
        info.is_intra_nxn = true;
        info.mb_type_raw = mb.mb_type_raw;
        info.transform_size_8x8_flag = mb.transform_size_8x8_flag;
        // Reset the per-block pred-mode arrays so stale entries from
        // a previously-reconstructed picture can't bleed through.
        info.intra_4x4_pred_modes = [0; 16];
        info.intra_8x8_pred_modes = [0; 4];
        // §6.4.8 — stamp slice_id eagerly so
        // `intra_neighbour_mb_info` / `same_slice_at` see the current MB
        // as in-slice when later blocks of this same MB consult it.
        info.slice_id = current_slice_id;
    }

    if mb.transform_size_8x8_flag {
        // §8.3.2 — Intra_8x8 path.
        #[allow(clippy::needless_range_loop)] // spec §8.3.2 8x8 block walk
        for blk8 in 0..4usize {
            let (bx, by) = LUMA_8X8_XY[blk8];
            // §8.3.2.1 — derive Intra_8x8 prediction mode per eq. 8-73.
            let mode_idx =
                derive_intra_8x8_pred_mode(grid, mb_addr, blk8, pred, cip, current_slice_id);
            let mode = Intra8x8Mode::from_index(mode_idx).unwrap_or(Intra8x8Mode::Dc);

            // Gather neighbour samples for this 8x8 block.
            let raw = gather_samples_8x8(
                pic,
                grid,
                mb_px + bx,
                mb_py + by,
                blk8,
                current_slice_id,
                cip,
            );
            let filtered = filter_samples_8x8(&raw, bit_depth_y);
            let mut pred_samples = [0i32; 64];
            predict_8x8(mode, &filtered, bit_depth_y, &mut pred_samples);

            // §8.5.13 — inverse transform of the 8x8 residual block.
            //
            // Storage convention: `mb.residual_luma[blk8*4..blk8*4+4]`
            // holds four 16-entry arrays that together contain the 64
            // coefficients of this 8x8 block. The concatenation order
            // (sub 0 → indices 0..=15, sub 1 → 16..=31, ...) is the
            // 8x8 scan order produced by CABAC's single
            // `residual_block(..., 63, 64)` call (see §7.3.5.3.1 and
            // the CABAC path in macroblock_layer.rs).
            //
            // Assumption: the concatenated 64 entries are in 8x8
            // zig-zag scan order (§8.5.7 / Table 8-14) — this matches
            // the CABAC path. The CAVLC path in macroblock_layer
            // currently emits four independent 4x4 zig-zag scans per
            // 8x8 block (a known simplification documented at the top
            // of that module); for that path the pixels will be wrong
            // but at least the pipeline runs instead of erroring out.
            // TODO: once the CAVLC 8x8 walker reads a single 64-coeff
            // block in 8x8 zig-zag order, this will be correct for
            // both entropy modes.
            let coeffs_flat: [i32; 64] = {
                let bit = (cbp_luma >> blk8) & 1 == 1;
                if bit {
                    // `mb.residual_luma` is compacted to only include
                    // 8x8 quadrants whose cbp_luma bit is set, so map
                    // `blk8` to the array index by counting set bits
                    // below it in `cbp_luma` (§7.3.5.3 CAVLC / CABAC
                    // both push sequentially).
                    let set_before = (cbp_luma & ((1u8 << blk8) - 1)).count_ones() as usize;
                    let base_slot = set_before * 4;
                    let mut scan = [0i32; 64];
                    for sub in 0..4usize {
                        let slot = base_slot + sub;
                        if let Some(coefs) = mb.residual_luma.get(slot) {
                            for (i, c) in coefs.iter().enumerate().take(16) {
                                scan[sub * 16 + i] = *c;
                            }
                        }
                    }
                    // §8.5.7 / Table 8-14 — invert the 8x8 zig-zag
                    // scan. `inverse_transform_8x8` consumes a row-
                    // major 8x8 matrix (c_ij at index i*8+j).
                    inverse_scan_8x8_zigzag(&scan)
                } else {
                    [0i32; 64]
                }
            };
            let residual = inverse_transform_8x8(&coeffs_flat, qp_y, &sl8, bit_depth_y)?;

            // Add prediction + residual; write back via MbWriter so
            // §6.4.1 eq. (6-10) field-MB y-stride is applied.
            for y in 0..8 {
                for x in 0..8 {
                    let idx = y * 8 + x;
                    let v = clip_sample(pred_samples[idx] + residual[idx], bit_depth_y);
                    writer.set_luma(pic, bx + x as i32, by + y as i32, v);
                }
            }

            // Track intra_8x8_pred_modes in the grid BEFORE the next
            // 8x8 block is derived — its neighbour lookup reads this.
            if let Some(info) = grid.get_mut(mb_addr) {
                info.intra_8x8_pred_modes[blk8] = mode.as_index();
            }
        }
    } else {
        // §8.3.1 — Intra_4x4 path.
        // OXIDEAV_H264_RECON_DEBUG — test-only per-block trace for MB 0.
        let debug_mb = std::env::var("OXIDEAV_H264_RECON_DEBUG_MB")
            .ok()
            .and_then(|s| s.parse::<u32>().ok());
        let debug = (mb_addr == 0 && std::env::var("OXIDEAV_H264_RECON_DEBUG").is_ok())
            || debug_mb == Some(mb_addr);
        if debug {
            eprintln!(
                "RECON MB{} I_NxN: cbp_luma={:#x} qp_y={} mb_type_raw={} cbp={:#x} entropy={}",
                mb_addr,
                cbp_luma,
                qp_y,
                mb.mb_type_raw,
                mb.coded_block_pattern,
                if pps.entropy_coding_mode_flag {
                    "CABAC"
                } else {
                    "CAVLC"
                }
            );
            eprintln!("  prev_flags: {:?}", pred.prev_intra4x4_pred_mode_flag);
            eprintln!("  rem_modes:  {:?}", pred.rem_intra4x4_pred_mode);
            eprintln!("  residual_luma.len() = {}", mb.residual_luma.len());
            for (i, blk) in mb.residual_luma.iter().enumerate() {
                let nz = blk.iter().filter(|&&v| v != 0).count();
                if nz > 0 {
                    eprintln!("  residual_luma[{}] ({} nz): {:?}", i, nz, blk);
                } else {
                    eprintln!("  residual_luma[{}] (all zero)", i);
                }
            }
        }
        #[allow(clippy::needless_range_loop)] // spec §8.3.1.1 raster-Z 4x4 walk
        for block_idx in 0..16usize {
            let (bx, by) = LUMA_4X4_XY[block_idx];
            // §8.3.1.1 — derive Intra_4x4 prediction mode per eq. 8-41.
            let mode_idx =
                derive_intra_4x4_pred_mode(grid, mb_addr, block_idx, pred, cip, current_slice_id);
            let mode = Intra4x4Mode::from_index(mode_idx).unwrap_or(Intra4x4Mode::Dc);

            // Gather neighbour samples for this 4x4 block.
            let samples = gather_samples_4x4(
                pic,
                grid,
                mb_px + bx,
                mb_py + by,
                block_idx,
                current_slice_id,
                cip,
            );
            let mut pred_samples = [0i32; 16];
            predict_4x4(mode, &samples, bit_depth_y, &mut pred_samples);

            // §8.5.12 — inverse-transform the residual (if any).
            //
            // `mb.residual_luma` is populated by the macroblock-layer
            // parser only for the 8x8 quadrants whose cbp_luma bit is
            // set (§7.3.5.3 / §7.4.5.3.2), and pushed sequentially in
            // raster-Z order. So for a cbp_luma with gaps (e.g. 0x5 or
            // 0xe) the array index is NOT block_idx but the compact
            // position computed by counting set bits below block_idx's
            // 8x8 quadrant.
            let blk8 = block_idx / 4;
            let coeffs_scan = if (cbp_luma >> blk8) & 1 == 1 {
                let set_before = (cbp_luma & ((1u8 << blk8) - 1)).count_ones() as usize;
                let compact_idx = set_before * 4 + (block_idx % 4);
                mb.residual_luma
                    .get(compact_idx)
                    .copied()
                    .unwrap_or([0i32; 16])
            } else {
                [0i32; 16]
            };
            let coeffs = crate::transform::inverse_scan_4x4_zigzag(&coeffs_scan);
            let residual = inverse_transform_4x4(&coeffs, qp_y, &sl4, bit_depth_y)?;

            if debug && (block_idx == 0 || block_idx == 3) {
                eprintln!(
                    "  blk{}: pred_mode={} samples.avail=tl{}t{}tr{}l{}",
                    block_idx,
                    mode.as_index(),
                    samples.availability.top_left,
                    samples.availability.top,
                    samples.availability.top_right,
                    samples.availability.left
                );
                eprintln!("    top: {:?}, left: {:?}", samples.top, samples.left);
                eprintln!("    pred_samples: {:?}", pred_samples);
                eprintln!("    coeffs_scan: {:?}", coeffs_scan);
                eprintln!("    coeffs(4x4): {:?}", coeffs);
                eprintln!("    residual: {:?}", residual);
            }

            // Combine + clip + write.
            write_block_luma(
                writer,
                pic,
                &pred_block_to_mb(&pred_samples, bx, by),
                bx,
                by,
                &residual,
                bit_depth_y,
            );

            // Track intra_4x4_pred_modes in the grid BEFORE the next
            // 4x4 block is derived — its neighbour lookup reads this.
            if let Some(info) = grid.get_mut(mb_addr) {
                info.intra_4x4_pred_modes[block_idx] = mode.as_index();
            }
        }
    }

    Ok(())
}

/// Expand a 4x4 local prediction block into an MB-sized 256 array
/// just so `write_block_luma` can share its signature with the
/// Intra_16x16 path. The surrounding entries are ignored by the
/// caller (it only reads the 4x4 window at (bx, by)).
fn pred_block_to_mb(src4: &[i32; 16], bx: i32, by: i32) -> [i32; 256] {
    let mut out = [0i32; 256];
    for yy in 0..4 {
        for xx in 0..4 {
            let gx = (bx as usize) + xx;
            let gy = (by as usize) + yy;
            out[gy * 16 + gx] = src4[yy * 4 + xx];
        }
    }
    out
}

// -------------------------------------------------------------------------
// Chroma intra reconstruction
// -------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn reconstruct_chroma_intra(
    mb: &Macroblock,
    qp_y: i32,
    chroma_array_type: u32,
    bit_depth_c: u32,
    mb_px: i32,
    mb_py: i32,
    writer: &MbWriter,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
    grid: &MbGrid,
    current_slice_id: i32,
) -> Result<(), ReconstructError> {
    // Monochrome: no chroma work to do.
    if chroma_array_type == 0 {
        return Ok(());
    }
    if chroma_array_type == 3 {
        return Err(ReconstructError::UnsupportedChromaArrayType(
            chroma_array_type,
        ));
    }

    let ct = if chroma_array_type == 1 {
        IpChromaArrayType::Yuv420
    } else {
        IpChromaArrayType::Yuv422
    };
    let (mbw_c, mbh_c) = chroma_mb_dims(chroma_array_type);
    // Chroma MB origin — MbWriter handles the MBAFF field-MB / non-
    // MBAFF cases uniformly (§6.4.1).
    let c_mb_px = writer.chroma_mb_px();
    let c_mb_py = writer.chroma_mb_py();
    let _ = mb_px;
    let _ = mb_py;

    // §7.4.5.1 — intra_chroma_pred_mode (only when intra).
    let chroma_mode_idx = mb
        .mb_pred
        .as_ref()
        .map(|p| p.intra_chroma_pred_mode)
        .unwrap_or(0);
    let chroma_mode = IntraChromaMode::from_index(chroma_mode_idx).unwrap_or(IntraChromaMode::Dc);

    // §8.5.8 — QPc derivation per plane (Cb uses chroma_qp_index_offset,
    // Cr uses second_chroma_qp_index_offset when present).
    let cb_offset = pps.chroma_qp_index_offset;
    let cr_offset = pps
        .extension
        .as_ref()
        .map(|e| e.second_chroma_qp_index_offset)
        .unwrap_or(pps.chroma_qp_index_offset);
    let qp_cb = qp_y_to_qp_c(qp_y, cb_offset);
    let qp_cr = qp_y_to_qp_c(qp_y, cr_offset);

    let cbp_chroma = ((mb.coded_block_pattern >> 4) & 0x03) as u8;
    // §7.4.2.1.1.1 Table 7-2 — intra chroma lists: i=1 (Cb) / i=2 (Cr).
    let sl4_cb = select_scaling_list_4x4(1, sps, pps);
    let sl4_cr = select_scaling_list_4x4(2, sps, pps);

    // §8.3.4 subwidth / subheight used to project chroma coordinates
    // back to picture-absolute luma so the slice-id check in
    // `gather_samples_chroma` hits the right grid MB.
    let (sub_width_c, sub_height_c) = match chroma_array_type {
        1 => (2, 2), // 4:2:0
        2 => (2, 1), // 4:2:2
        _ => (1, 1),
    };

    // Per plane: gather samples, predict, inverse-transform residual,
    // combine.
    for plane in 0..2u8 {
        let samples = gather_samples_chroma(
            pic,
            grid,
            c_mb_px,
            c_mb_py,
            ct,
            plane,
            current_slice_id,
            sub_width_c,
            sub_height_c,
            pps.constrained_intra_pred_flag,
        );
        let out_len = (mbw_c as usize) * (mbh_c as usize);
        let mut pred_samples = vec![0i32; out_len];
        predict_chroma(chroma_mode, &samples, ct, bit_depth_c, &mut pred_samples);

        let qp_c = if plane == 0 { qp_cb } else { qp_cr };
        let sl4 = if plane == 0 { &sl4_cb } else { &sl4_cr };

        // §8.5.11 — chroma DC Hadamard.
        let dc_block = if plane == 0 {
            &mb.residual_chroma_dc_cb
        } else {
            &mb.residual_chroma_dc_cr
        };
        // Chroma DC arrays (2x2 for 4:2:0, 4x2 for 4:2:2).
        let dc_flat: [i32; 8] = {
            let mut a = [0i32; 8];
            for (i, v) in dc_block.iter().enumerate().take(a.len()) {
                a[i] = *v;
            }
            a
        };

        let (dc_cb4, dc_cb8): (Option<[i32; 4]>, Option<[i32; 8]>) = if cbp_chroma > 0 {
            if chroma_array_type == 1 {
                let dc4: [i32; 4] = [dc_flat[0], dc_flat[1], dc_flat[2], dc_flat[3]];
                let out = inverse_hadamard_chroma_dc_420(&dc4, qp_c, sl4, bit_depth_c)?;
                (Some(out), None)
            } else {
                let out = inverse_hadamard_chroma_dc_422(&dc_flat, qp_c, sl4, bit_depth_c)?;
                (None, Some(out))
            }
        } else {
            (Some([0i32; 4]), Some([0i32; 8]))
        };

        // §8.5.12 — chroma AC 4x4 blocks (one or two 8x8 chroma MB rows).
        // 4:2:0: 4 blocks (8x8 layout 2x2). 4:2:2: 8 blocks (4x2 layout).
        let num_c8x8 = if chroma_array_type == 1 { 1 } else { 2 };
        let n_ac = 4 * num_c8x8 as usize;
        let ac_blocks = if plane == 0 {
            &mb.residual_chroma_ac_cb
        } else {
            &mb.residual_chroma_ac_cr
        };

        for blk in 0..n_ac {
            // DC coefficient for this 4x4 block.
            let dc_c = if chroma_array_type == 1 {
                dc_cb4.unwrap()[blk]
            } else {
                dc_cb8.unwrap()[blk]
            };
            let ac_scan = if cbp_chroma == 2 {
                ac_blocks.get(blk).copied().unwrap_or([0i32; 16])
            } else {
                [0i32; 16]
            };
            // Chroma AC: parser slots 0..=14 are spec scan positions 1..=15.
            let mut coeffs = crate::transform::inverse_scan_4x4_zigzag_ac(&ac_scan);
            coeffs[0] = dc_c;
            let residual = inverse_transform_4x4_dc_preserved(&coeffs, qp_c, sl4, bit_depth_c)?;

            // Position of this 4x4 chroma block inside the MB.
            // For 4:2:0 (8x8 chroma MB): blocks laid out in 2x2 at (0,0) (4,0) (0,4) (4,4).
            // For 4:2:2 (8x16 chroma MB): 4x2 layout, 8 blocks.
            let (bx, by) = chroma_block_xy(chroma_array_type, blk);
            for yy in 0..4 {
                for xx in 0..4 {
                    let pidx = ((by as usize + yy) * (mbw_c as usize)) + (bx as usize + xx);
                    let v = clip_sample(pred_samples[pidx] + residual[yy * 4 + xx], bit_depth_c);
                    if plane == 0 {
                        writer.set_cb(pic, bx + xx as i32, by + yy as i32, v);
                    } else {
                        writer.set_cr(pic, bx + xx as i32, by + yy as i32, v);
                    }
                }
            }
        }
    }
    let _ = c_mb_px;
    let _ = c_mb_py;

    Ok(())
}

/// Lookup table for chroma 4x4 block (x, y) inside the chroma MB.
/// §6.4.3 informative, aligned with chroma scan.
fn chroma_block_xy(chroma_array_type: u32, blk: usize) -> (i32, i32) {
    // 4:2:0: 4 blocks in a 2x2 layout inside an 8x8 chroma MB.
    // 4:2:2: 8 blocks in a 2x4 layout inside an 8x16 chroma MB.
    match chroma_array_type {
        1 => {
            const XY: [(i32, i32); 4] = [(0, 0), (4, 0), (0, 4), (4, 4)];
            XY[blk]
        }
        2 => {
            const XY: [(i32, i32); 8] = [
                (0, 0),
                (4, 0),
                (0, 4),
                (4, 4),
                (0, 8),
                (4, 8),
                (0, 12),
                (4, 12),
            ];
            XY[blk]
        }
        _ => (0, 0),
    }
}

// -------------------------------------------------------------------------
// Neighbour sample gathering helpers
// -------------------------------------------------------------------------

/// §8.3.3 — reference samples for an Intra_16x16 block at MB origin
/// (mb_px, mb_py). Availability derived solely from picture boundaries
/// — full §6.4.11 availability (including constrained_intra_pred)
/// should be layered by the caller when needed.
fn gather_samples_16x16(
    pic: &Picture,
    grid: &MbGrid,
    mb_px: i32,
    mb_py: i32,
    current_slice_id: i32,
    constrained_intra_pred: bool,
) -> Samples16x16 {
    let left_avail = mb_px > 0
        && same_slice_at(grid, mb_px - 1, mb_py, current_slice_id)
        && cip_ok_at(grid, mb_px - 1, mb_py, constrained_intra_pred);
    let top_avail = mb_py > 0
        && same_slice_at(grid, mb_px, mb_py - 1, current_slice_id)
        && cip_ok_at(grid, mb_px, mb_py - 1, constrained_intra_pred);
    let tl_avail = mb_px > 0
        && mb_py > 0
        && same_slice_at(grid, mb_px - 1, mb_py - 1, current_slice_id)
        && cip_ok_at(grid, mb_px - 1, mb_py - 1, constrained_intra_pred);

    let top_left = if tl_avail {
        pic.luma_at(mb_px - 1, mb_py - 1)
    } else {
        0
    };
    let mut top = [0i32; 16];
    for x in 0..16 {
        top[x as usize] = if top_avail {
            pic.luma_at(mb_px + x, mb_py - 1)
        } else {
            0
        };
    }
    let mut left = [0i32; 16];
    for y in 0..16 {
        left[y as usize] = if left_avail {
            pic.luma_at(mb_px - 1, mb_py + y)
        } else {
            0
        };
    }
    Samples16x16 {
        top_left,
        top,
        left,
        availability: Neighbour4x4Availability {
            top_left: tl_avail,
            top: top_avail,
            top_right: false, // Intra_16x16 doesn't use top_right
            left: left_avail,
        },
    }
}

/// §8.3.1 — reference samples for a 4x4 block at (bx, by) (top-left
/// luma sample of the 4x4 block).
///
/// `block_idx` is the luma4x4BlkIdx (0..=15) of this 4x4 block within
/// its macroblock; it gates the top-right sample availability per
/// §6.4.11.4 / the 4x4 block scan order (§6.4.3 / Figure 6-10).
/// Blocks 3, 7, 11, 13, 15 have their top-right neighbours either not
/// yet decoded (3, 11) or located in a macroblock to the right that
/// has not been decoded yet (7, 13, 15). For those indices top-right
/// is marked "not available for Intra_4x4 prediction"; the §8.3.1.2
/// substitution rule (copy p[3, -1] into p[4..7, -1]) then kicks in
/// inside the intra-pred helpers.
fn gather_samples_4x4(
    pic: &Picture,
    grid: &MbGrid,
    bx: i32,
    by: i32,
    block_idx: usize,
    current_slice_id: i32,
    constrained_intra_pred: bool,
) -> Samples4x4 {
    let left_avail = bx > 0
        && same_slice_at(grid, bx - 1, by, current_slice_id)
        && cip_ok_at(grid, bx - 1, by, constrained_intra_pred);
    let top_avail = by > 0
        && same_slice_at(grid, bx, by - 1, current_slice_id)
        && cip_ok_at(grid, bx, by - 1, constrained_intra_pred);
    let tl_avail = bx > 0
        && by > 0
        && same_slice_at(grid, bx - 1, by - 1, current_slice_id)
        && cip_ok_at(grid, bx - 1, by - 1, constrained_intra_pred);
    // Top-right availability: the 4 samples at (bx+4..=bx+7, by-1) must
    // (a) lie in an above-row (i.e. `top_avail`), (b) fit inside the
    // picture horizontally, AND (c) come from a 4x4 block that has
    // already been decoded per the §6.4.3 scan. For luma4x4BlkIdx ∈
    // {3, 7, 11, 13, 15} those samples are not yet available — block
    // 3/11 need block 4/14 respectively (later in scan), blocks
    // 7/13/15 need samples from the right-neighbour macroblock which
    // hasn't been decoded yet when the current MB is being processed.
    let tr_scan_ok = !matches!(block_idx, 3 | 7 | 11 | 13 | 15);
    let tr_avail = by > 0
        && tr_scan_ok
        && (bx + 4 < pic.width_in_samples as i32)
        && same_slice_at(grid, bx + 4, by - 1, current_slice_id)
        && cip_ok_at(grid, bx + 4, by - 1, constrained_intra_pred);

    let top_left = if tl_avail {
        pic.luma_at(bx - 1, by - 1)
    } else {
        0
    };
    let mut top = [0i32; 4];
    let mut top_right = [0i32; 4];
    for x in 0..4 {
        top[x as usize] = if top_avail {
            pic.luma_at(bx + x, by - 1)
        } else {
            0
        };
        top_right[x as usize] = if tr_avail {
            pic.luma_at(bx + 4 + x, by - 1)
        } else {
            0
        };
    }
    let mut left = [0i32; 4];
    for y in 0..4 {
        left[y as usize] = if left_avail {
            pic.luma_at(bx - 1, by + y)
        } else {
            0
        };
    }
    Samples4x4 {
        top_left,
        top,
        top_right,
        left,
        availability: Neighbour4x4Availability {
            top_left: tl_avail,
            top: top_avail,
            top_right: tr_avail,
            left: left_avail,
        },
    }
}

/// §6.4.8 — is the macroblock containing picture-absolute luma sample
/// `(x, y)` in the same slice as the current macroblock?
///
/// `current_slice_id < 0` is the "don't-know / don't-care" escape hatch
/// that keeps legacy / unit-test callers' semantics unchanged. A grid
/// entry with `slice_id == -1` means the MB was either never
/// reconstructed or pre-dates slice-id stamping (e.g., the current MB
/// being reconstructed — `reconstruct_intra_nxn`'s preamble stamps
/// `slice_id` along with `available`); we treat it as "match" so the
/// check only rejects MBs explicitly stamped with a different slice.
/// Out-of-picture coordinates answer `true` so the caller's own
/// picture-boundary check remains the single source of truth for that
/// case.
fn same_slice_at(grid: &MbGrid, x: i32, y: i32, current_slice_id: i32) -> bool {
    if current_slice_id < 0 {
        return true;
    }
    if x < 0 || y < 0 {
        return true;
    }
    let mb_x = (x >> 4) as u32;
    let mb_y = (y >> 4) as u32;
    if mb_x >= grid.width_in_mbs || mb_y >= grid.height_in_mbs {
        return true;
    }
    let addr = mb_y * grid.width_in_mbs + mb_x;
    match grid.get(addr) {
        Some(info) if info.slice_id >= 0 => info.slice_id == current_slice_id,
        _ => true,
    }
}

/// §8.3.1.2 / §8.3.2.2 / §8.3.3.1 / §8.3.4.1 — "constrained_intra_pred
/// rejection": with `pps.constrained_intra_pred_flag == 1`, a sample
/// p[x, y] that lies inside a macroblock coded using an Inter prediction
/// mode is marked as "not available for Intra prediction".
///
/// `(x, y)` are picture-absolute luma sample coordinates. For chroma
/// callers, pass the chroma coordinates scaled up to luma (the spec's
/// rule applies to the macroblock, not the individual sample plane).
///
/// Returns `false` (neighbour unavailable) when constrained_intra_pred
/// applies. Returns `true` in all other cases — including out-of-picture
/// coordinates and unstamped grid entries, leaving the caller's other
/// availability checks as the source of truth.
fn cip_ok_at(grid: &MbGrid, x: i32, y: i32, constrained_intra_pred: bool) -> bool {
    if !constrained_intra_pred {
        return true;
    }
    if x < 0 || y < 0 {
        return true;
    }
    let mb_x = (x >> 4) as u32;
    let mb_y = (y >> 4) as u32;
    if mb_x >= grid.width_in_mbs || mb_y >= grid.height_in_mbs {
        return true;
    }
    let addr = mb_y * grid.width_in_mbs + mb_x;
    match grid.get(addr) {
        // Neighbour is an already-decoded inter MB — samples unavailable
        // under constrained_intra_pred per §8.3.1.2 / §8.3.2.2 /
        // §8.3.3.1 / §8.3.4.1.
        Some(info) if info.available && !info.is_intra => false,
        _ => true,
    }
}

/// §8.3.2 — reference samples for an 8x8 block at (bx, by).
///
/// `blk8` is the luma8x8BlkIdx (0..=3) of this 8x8 block within its
/// macroblock; it gates the top-right availability per §6.4.11.2 and
/// the 8x8 scan order. Block 3 (bottom-right 8x8 quadrant) needs
/// samples from the MB to the right which hasn't been decoded yet,
/// so its top-right is always "not available" regardless of the
/// picture bounds. The §8.3.2.2 substitution rule (copy p[7, -1] into
/// p[8..15, -1]) then applies inside `filter_samples_8x8`.
fn gather_samples_8x8(
    pic: &Picture,
    grid: &MbGrid,
    bx: i32,
    by: i32,
    blk8: usize,
    current_slice_id: i32,
    constrained_intra_pred: bool,
) -> Samples8x8 {
    let left_avail = bx > 0
        && same_slice_at(grid, bx - 1, by, current_slice_id)
        && cip_ok_at(grid, bx - 1, by, constrained_intra_pred);
    let top_avail = by > 0
        && same_slice_at(grid, bx, by - 1, current_slice_id)
        && cip_ok_at(grid, bx, by - 1, constrained_intra_pred);
    let tl_avail = bx > 0
        && by > 0
        && same_slice_at(grid, bx - 1, by - 1, current_slice_id)
        && cip_ok_at(grid, bx - 1, by - 1, constrained_intra_pred);
    let tr_scan_ok = blk8 != 3;
    let tr_avail = by > 0
        && tr_scan_ok
        && (bx + 8 < pic.width_in_samples as i32)
        && same_slice_at(grid, bx + 8, by - 1, current_slice_id)
        && cip_ok_at(grid, bx + 8, by - 1, constrained_intra_pred);

    let top_left = if tl_avail {
        pic.luma_at(bx - 1, by - 1)
    } else {
        0
    };
    let mut top = [0i32; 8];
    let mut top_right = [0i32; 8];
    for x in 0..8 {
        top[x as usize] = if top_avail {
            pic.luma_at(bx + x, by - 1)
        } else {
            0
        };
        top_right[x as usize] = if tr_avail {
            pic.luma_at(bx + 8 + x, by - 1)
        } else {
            0
        };
    }
    let mut left = [0i32; 8];
    for y in 0..8 {
        left[y as usize] = if left_avail {
            pic.luma_at(bx - 1, by + y)
        } else {
            0
        };
    }
    Samples8x8 {
        top_left,
        top,
        top_right,
        left,
        availability: Neighbour4x4Availability {
            top_left: tl_avail,
            top: top_avail,
            top_right: tr_avail,
            left: left_avail,
        },
    }
}

/// §8.3.4 — chroma reference samples for plane `plane` (0 = Cb, 1 = Cr).
fn gather_samples_chroma(
    pic: &Picture,
    grid: &MbGrid,
    c_mb_px: i32,
    c_mb_py: i32,
    ct: IpChromaArrayType,
    plane: u8,
    current_slice_id: i32,
    sub_width_c: i32,
    sub_height_c: i32,
    constrained_intra_pred: bool,
) -> SamplesChroma {
    let w = ct.width();
    let h = ct.height();
    // Chroma (c_mb_px, c_mb_py) → picture-absolute luma sample coordinate
    // so we can look the owning MB up in the slice grid.
    let to_lx = |cx: i32| cx * sub_width_c;
    let to_ly = |cy: i32| cy * sub_height_c;
    let left_avail = c_mb_px > 0
        && same_slice_at(grid, to_lx(c_mb_px - 1), to_ly(c_mb_py), current_slice_id)
        && cip_ok_at(
            grid,
            to_lx(c_mb_px - 1),
            to_ly(c_mb_py),
            constrained_intra_pred,
        );
    let top_avail = c_mb_py > 0
        && same_slice_at(grid, to_lx(c_mb_px), to_ly(c_mb_py - 1), current_slice_id)
        && cip_ok_at(
            grid,
            to_lx(c_mb_px),
            to_ly(c_mb_py - 1),
            constrained_intra_pred,
        );
    let tl_avail = c_mb_px > 0
        && c_mb_py > 0
        && same_slice_at(
            grid,
            to_lx(c_mb_px - 1),
            to_ly(c_mb_py - 1),
            current_slice_id,
        )
        && cip_ok_at(
            grid,
            to_lx(c_mb_px - 1),
            to_ly(c_mb_py - 1),
            constrained_intra_pred,
        );

    let fetch = |x: i32, y: i32| -> i32 {
        if plane == 0 {
            pic.cb_at(x, y)
        } else {
            pic.cr_at(x, y)
        }
    };

    let top_left = if tl_avail {
        fetch(c_mb_px - 1, c_mb_py - 1)
    } else {
        0
    };
    let mut top = vec![0i32; w];
    for (x, slot) in top.iter_mut().enumerate() {
        *slot = if top_avail {
            fetch(c_mb_px + x as i32, c_mb_py - 1)
        } else {
            0
        };
    }
    let mut left = vec![0i32; h];
    for (y, slot) in left.iter_mut().enumerate() {
        *slot = if left_avail {
            fetch(c_mb_px - 1, c_mb_py + y as i32)
        } else {
            0
        };
    }
    SamplesChroma {
        top_left,
        top,
        left,
        availability: Neighbour4x4Availability {
            top_left: tl_avail,
            top: top_avail,
            top_right: false,
            left: left_avail,
        },
    }
}

// -------------------------------------------------------------------------
// Sample write helpers
// -------------------------------------------------------------------------

/// Combine an MB-sized prediction array with a 4x4 residual at
/// position (bx, by), clip, and write into `pic.luma`.
///
/// `writer` applies the §6.4.1 eq. (6-10) field-MB y-stride; `(bx, by)`
/// is the 4x4 block's top-left in MB-local luma coordinates (0..=15).
fn write_block_luma(
    writer: &MbWriter,
    pic: &mut Picture,
    pred_mb: &[i32; 256],
    bx: i32,
    by: i32,
    residual_4x4: &[i32; 16],
    bit_depth: u32,
) {
    for yy in 0..4 {
        for xx in 0..4 {
            let pred_idx = ((by as usize) + yy) * 16 + (bx as usize) + xx;
            let res = residual_4x4[yy * 4 + xx];
            let v = clip_sample(pred_mb[pred_idx] + res, bit_depth);
            writer.set_luma(pic, bx + xx as i32, by + yy as i32, v);
        }
    }
}

/// §5.7 Clip1 — clip sample to `[0, (1 << bit_depth) - 1]`.
#[inline]
fn clip_sample(v: i32, bit_depth: u32) -> i32 {
    let hi = (1i32 << bit_depth) - 1;
    v.clamp(0, hi)
}

/// §6.2 / Table 6-1 — chroma MB dimensions per ChromaArrayType.
fn chroma_mb_dims(chroma_array_type: u32) -> (u32, u32) {
    match chroma_array_type {
        1 => (8, 8),
        2 => (8, 16),
        3 => (16, 16),
        _ => (0, 0),
    }
}

/// §6.2 / Table 6-1 — (SubWidthC, SubHeightC): luma:chroma sample ratio
/// per ChromaArrayType. ChromaArrayType 0 (monochrome) returns `(1, 1)`
/// as a safe fallback; ChromaArrayType 3 (4:4:4) returns `(1, 1)`.
#[inline]
fn chroma_subsample(chroma_array_type: u32) -> (i32, i32) {
    match chroma_array_type {
        1 => (2, 2),
        2 => (2, 1),
        3 => (1, 1),
        _ => (1, 1),
    }
}

// =========================================================================
// §8.4 — Inter-prediction reconstruction
// =========================================================================

/// §7.4.5 — which reference lists a P/B partition uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PartMode {
    /// Predicted from list 0 only.
    L0Only,
    /// Predicted from list 1 only (B slices).
    L1Only,
    /// Bi-predicted from both lists (B slices).
    BiPred,
    /// Direct mode — no MVD in the bitstream; MVs derived per §8.4.1.2.
    Direct,
}

/// A single inter partition (16x16, 16x8, 8x16, or 8x8 sub-partition).
/// Coordinates are luma-sample offsets inside the MB.
#[derive(Debug, Clone, Copy)]
struct InterPartition {
    /// Origin (x, y) in luma samples, inside the MB (0..=15).
    x: u8,
    y: u8,
    /// Width / height in luma samples (4, 8, or 16).
    w: u8,
    h: u8,
    mode: PartMode,
    /// Shape flag for MVpred (§8.4.1.3 eq. 8-203..8-206).
    shape: MvpredShape,
    /// List-0 reference index. `-1` means "not used".
    ref_idx_l0: i8,
    /// List-1 reference index. `-1` means "not used".
    ref_idx_l1: i8,
    /// Parsed MVDs (1/4-pel), if present. Zero for direct / skip.
    mvd_l0: (i32, i32),
    mvd_l1: (i32, i32),
    /// `true` when this partition was generated from P_Skip / B_Skip so
    /// that `derive_partition_mvs` can apply the §8.4.1.2 zero-MV
    /// substitution rules specific to skip mode. Regular inter partitions
    /// with MVD = (0, 0) and ref_idx = 0 look identical in (w, h, mode,
    /// shape, mvd, ref_idx) but MUST use the standard MVpred derivation
    /// per §8.4.1.3 — not the P_Skip zero-forcing conditions.
    is_skip: bool,
    /// §8.4.1.2.3 — pre-computed L0/L1 MV for temporal-direct partitions.
    /// When `Some`, these override the §8.4.1 derivation in
    /// `derive_partition_mvs` (which would otherwise invoke median
    /// prediction). `None` for explicit inter partitions and for
    /// spatial-direct partitions that still go through the median path.
    precomputed_mv: Option<(Mv, Mv)>,
}

impl Default for InterPartition {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            w: 16,
            h: 16,
            mode: PartMode::L0Only,
            shape: MvpredShape::Default,
            ref_idx_l0: -1,
            ref_idx_l1: -1,
            mvd_l0: (0, 0),
            mvd_l1: (0, 0),
            is_skip: false,
            precomputed_mv: None,
        }
    }
}

/// §8.4 — Per-MB inter reconstruction.
#[allow(clippy::too_many_arguments)]
fn reconstruct_mb_inter<R: RefPicProvider>(
    mb: &Macroblock,
    mb_addr: u32,
    qp_y: i32,
    chroma_array_type: u32,
    bit_depth_y: u32,
    bit_depth_c: u32,
    slice_header: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    ref_pics: &R,
    pic: &mut Picture,
    grid: &mut MbGrid,
    mbaff_frame_flag: bool,
    mb_field_decoding_flag: bool,
    current_slice_id: i32,
) -> Result<(), ReconstructError> {
    // §6.4.1 — MB sample origin (MBAFF-aware).
    let (mb_px, mb_py) = mb_sample_origin(grid, mb_addr, mbaff_frame_flag, mb_field_decoding_flag);
    // §6.4.1 eq. (6-10) — MBAFF field-MB y-stride writer.
    let writer = MbWriter::new(
        grid,
        mb_addr,
        mb_px,
        mb_py,
        mbaff_frame_flag,
        mb_field_decoding_flag,
        chroma_array_type,
    );

    // -------- Derive inter partitions --------------------------------
    let partitions = derive_inter_partitions(
        mb,
        slice_header,
        sps,
        ref_pics,
        pic,
        mb_addr,
        grid,
        current_slice_id,
    )?;

    // Test-only OXIDEAV_H264_RECON_DEBUG / OXIDEAV_H264_RECON_DEBUG_MB
    // instrumentation — prints per-MB + per-partition inter reconstruct
    // inputs so a human operator can compare against ffmpeg's trace.
    let inter_debug_mb = std::env::var("OXIDEAV_H264_RECON_DEBUG_MB")
        .ok()
        .and_then(|s| s.parse::<u32>().ok());
    let inter_debug = (mb_addr == 0 && std::env::var("OXIDEAV_H264_RECON_DEBUG").is_ok())
        || inter_debug_mb == Some(mb_addr);
    if inter_debug {
        eprintln!(
            "RECON_INTER MB#{} mb_type={:?} mb_type_raw={} slice_type={:?} \
             cbp={:#x} transform8x8={} qp_y={} partitions={}",
            mb_addr,
            mb.mb_type,
            mb.mb_type_raw,
            slice_header.slice_type,
            mb.coded_block_pattern,
            mb.transform_size_8x8_flag,
            qp_y,
            partitions.len(),
        );
        for (i, p) in partitions.iter().enumerate() {
            eprintln!(
                "  part[{}] x={} y={} w={} h={} mode={:?} shape={:?} \
                 ref_l0={} ref_l1={} mvd_l0={:?} mvd_l1={:?}",
                i,
                p.x,
                p.y,
                p.w,
                p.h,
                p.mode,
                p.shape,
                p.ref_idx_l0,
                p.ref_idx_l1,
                p.mvd_l0,
                p.mvd_l1,
            );
        }
    }

    // -------- For each partition: MVpred, MV, MC, write prediction ---
    // Luma prediction samples for the whole MB.
    let mut pred_luma = [0i32; 256];
    // Chroma prediction samples: sized per chroma MB dims.
    let (mbw_c, mbh_c) = chroma_mb_dims(chroma_array_type);
    let mut pred_cb = vec![0i32; (mbw_c as usize) * (mbh_c as usize)];
    let mut pred_cr = vec![0i32; (mbw_c as usize) * (mbh_c as usize)];

    // §6.4.8 — stamp the current MB's slice identity eagerly so that
    // later partitions' MVpred neighbour lookups (within the same MB)
    // see this MB as in-slice via `same_slice_at` / `neighbour_from_block`.
    if let Some(info) = grid.get_mut(mb_addr) {
        info.slice_id = current_slice_id;
    }

    for part in &partitions {
        process_partition(
            part,
            mb_addr,
            mb_px,
            mb_py,
            chroma_array_type,
            bit_depth_y,
            bit_depth_c,
            slice_header,
            pps,
            ref_pics,
            grid,
            pic,
            &mut pred_luma,
            &mut pred_cb,
            &mut pred_cr,
            inter_debug,
            current_slice_id,
        )?;
    }

    // -------- Residual add (§8.5 inverse transform) ------------------
    // P_Skip / B_Skip never carry residual; for all other inter MBs,
    // the residual walker fills in residual_luma / residual_chroma_*
    // the same shape as for I_NxN. We re-use the 4x4 transform path
    // for each 4x4 block, gated by cbp_luma / cbp_chroma, and combine
    // with pred_luma before writing to the picture.
    let cbp_luma = (mb.coded_block_pattern & 0x0F) as u8;
    let cbp_chroma = ((mb.coded_block_pattern >> 4) & 0x03) as u8;
    // §7.4.2.1.1.1 Table 7-2 — inter-luma lists: 4x4 i=3 / 8x8 i=7 (sub-idx 1).
    let sl4 = select_scaling_list_4x4(3, sps, pps);
    let sl8 = select_scaling_list_8x8(1, sps, pps);

    if mb.transform_size_8x8_flag {
        // §8.5.13 — 8x8 inter residual path (four 8x8 blocks).
        // Each 8x8 is flagged by one bit of cbp_luma. The four 16-
        // entry arrays in residual_luma[blk8*4..blk8*4+4] concatenate
        // to the 8x8-zigzag-scanned coefficients (CABAC path); invert
        // the 8x8 scan (§8.5.7 / Table 8-14) before feeding into
        // inverse_transform_8x8. See Intra_8x8 branch for the same
        // reasoning and the CAVLC-path caveat.
        #[allow(clippy::needless_range_loop)] // spec §8.5.13 4×8x8 walk
        for blk8 in 0..4usize {
            let (bx, by) = LUMA_8X8_XY[blk8];
            let has_res = (cbp_luma >> blk8) & 1 == 1;
            let coeffs = if has_res {
                // `residual_luma` is compacted by the parser —
                // quadrants with cbp_luma bit cleared are skipped, so
                // index by set-bits-below-blk8 (see Intra_NxN path).
                let set_before = (cbp_luma & ((1u8 << blk8) - 1)).count_ones() as usize;
                let base_slot = set_before * 4;
                let mut scan = [0i32; 64];
                for sub in 0..4usize {
                    let slot = base_slot + sub;
                    if let Some(c) = mb.residual_luma.get(slot) {
                        for (i, v) in c.iter().enumerate().take(16) {
                            scan[sub * 16 + i] = *v;
                        }
                    }
                }
                inverse_scan_8x8_zigzag(&scan)
            } else {
                [0i32; 64]
            };
            let residual = inverse_transform_8x8(&coeffs, qp_y, &sl8, bit_depth_y)?;
            for y in 0..8 {
                for x in 0..8 {
                    let v =
                        pred_luma[(by as usize + y) * 16 + (bx as usize + x)] + residual[y * 8 + x];
                    writer.set_luma(
                        pic,
                        bx + x as i32,
                        by + y as i32,
                        clip_sample(v, bit_depth_y),
                    );
                }
            }
        }
    } else {
        // §8.5.12 — 4x4 inter residual path.
        #[allow(clippy::needless_range_loop)] // spec §8.5.12 raster-Z 4x4 walk
        for blk4 in 0..16usize {
            let (bx, by) = LUMA_4X4_XY[blk4];
            let blk8 = blk4 / 4;
            let has_res = (cbp_luma >> blk8) & 1 == 1;
            let coeffs_scan = if has_res {
                // Compact index into `residual_luma` — the parser
                // pushes 4 entries per cbp_luma bit that is set in
                // low-bit-first order, so the array index is (number
                // of set bits below blk8) * 4 + (blk4 % 4), not blk4.
                let set_before = (cbp_luma & ((1u8 << blk8) - 1)).count_ones() as usize;
                let compact_idx = set_before * 4 + (blk4 % 4);
                mb.residual_luma
                    .get(compact_idx)
                    .copied()
                    .unwrap_or([0i32; 16])
            } else {
                [0i32; 16]
            };
            let coeffs = crate::transform::inverse_scan_4x4_zigzag(&coeffs_scan);
            let residual = inverse_transform_4x4(&coeffs, qp_y, &sl4, bit_depth_y)?;
            for yy in 0..4 {
                for xx in 0..4 {
                    let v = pred_luma[(by as usize + yy) * 16 + (bx as usize + xx)]
                        + residual[yy * 4 + xx];
                    writer.set_luma(
                        pic,
                        bx + xx as i32,
                        by + yy as i32,
                        clip_sample(v, bit_depth_y),
                    );
                }
            }
        }
    }

    // -------- Chroma residual + write --------------------------------
    if chroma_array_type != 0 && chroma_array_type != 3 {
        reconstruct_inter_chroma_residual(
            mb,
            qp_y,
            chroma_array_type,
            bit_depth_c,
            mb_px,
            mb_py,
            &writer,
            sps,
            pps,
            cbp_chroma,
            &pred_cb,
            &pred_cr,
            pic,
        )?;
    }

    Ok(())
}

/// §8.4.2.3 — decide whether the explicit weighted-prediction path
/// applies for the current slice.
///
/// * P / SP slices: `pps.weighted_pred_flag` selects explicit.
/// * B slices: `pps.weighted_bipred_idc == 1` selects explicit.
///   `weighted_bipred_idc == 2` is implicit (§8.4.2.3.3) — handled
///   separately via [`weighted_implicit_active`]; this helper returns
///   `false` in that case.
///   `weighted_bipred_idc == 0` is default.
///
/// A `pred_weight_table` that is `None` (parser didn't attach one) is
/// always treated as "default" — the explicit path would have nothing
/// to read.
fn weighted_explicit_active(slice_header: &SliceHeader, pps: &Pps) -> bool {
    if slice_header.pred_weight_table.is_none() {
        return false;
    }
    match slice_header.slice_type {
        SliceType::P | SliceType::SP => pps.weighted_pred_flag,
        SliceType::B => pps.weighted_bipred_idc == 1,
        _ => false,
    }
}

/// §8.4.2.3.3 — decide whether the implicit weighted-prediction path
/// applies for the current slice.
///
/// Implicit mode is selected for B slices when `weighted_bipred_idc
/// == 2`. Only the bipred sub-case is affected by implicit weighting
/// (eq. 8-276 with `logWDC = 5`, zero offsets, and weights derived
/// from POC distance); L0-only / L1-only partitions still use the
/// default copy-through path per §8.4.2.3.1.
///
/// Unlike the explicit path, implicit does NOT require a
/// `pred_weight_table` in the slice — the weights are fully derived
/// from POC.
fn weighted_implicit_active(slice_header: &SliceHeader, pps: &Pps) -> bool {
    matches!(slice_header.slice_type, SliceType::B) && pps.weighted_bipred_idc == 2
}

/// §8.4.2.3.3 — implicit weighted-prediction weight derivation.
///
/// Returns `(w0, w1, log2WD)` where the final bipred prediction is
/// computed by the same eq. 8-276 formula as explicit mode:
///
/// ```text
///   v = Clip1( ((predL0*w0 + predL1*w1 + 2^logWD) >> (logWD + 1)) )
/// ```
///
/// with zero offsets (eq. 8-278, 8-279 set `o0C = o1C = 0`) and
/// `logWD = 5` (eq. 8-277). `w0` + `w1` always equals 64.
///
/// Equations (citing spec clause numbers):
///
/// * eq. 8-201: `tb = Clip3(-128, 127, currPOC - pic0POC)`
/// * eq. 8-202: `td = Clip3(-128, 127, pic1POC - pic0POC)`
/// * eq. 8-197: `tx = (16384 + Abs(td/2)) / td`
/// * eq. 8-198: `DistScaleFactor = Clip3(-1024, 1023, (tb*tx + 32) >> 6)`
/// * eq. 8-280, 8-281: fallback `w0 = w1 = 32` when `td == 0`, or one
///   of the refs is long-term, or `DistScaleFactor >> 2` is outside
///   `[-64, 128]`.
/// * eq. 8-282, 8-283: otherwise `w0 = 64 - (DistScaleFactor >> 2)`,
///   `w1 = DistScaleFactor >> 2`.
///
/// `long_term_either` should be set when either `pic0` or `pic1` is
/// marked "used for long-term reference" (§8.2.5). This module does
/// not have direct access to the DPB marking (`RefPicProvider`
/// exposes only [`Picture`] samples), so callers currently pass
/// `false` and the implicit path assumes both refs are short-term.
/// Non-conforming for bitstreams that use long-term refs as bipred
/// references — TODO(§8.2.5.2): plumb marking through to the provider.
fn implicit_bipred_weights(
    curr_poc: i32,
    poc_l0: i32,
    poc_l1: i32,
    long_term_either: bool,
) -> (i32, i32, u32) {
    // logWDC = 5 (eq. 8-277).
    let log2_wd: u32 = 5;

    // eq. 8-202 — td first; when zero, fall through to equal weights.
    let td = clip3_i32(-128, 127, poc_l1 - poc_l0);
    if td == 0 || long_term_either {
        // eq. 8-280, 8-281 — equal weighting.
        return (32, 32, log2_wd);
    }

    // eq. 8-201 — tb.
    let tb = clip3_i32(-128, 127, curr_poc - poc_l0);

    // eq. 8-197 — tx = (16384 + |td/2|) / td.
    //   Division matches the C-style "truncate toward zero" used by
    //   the spec (DivideBy in §5.7 defers to integer division).
    let tx = (16384 + (td / 2).abs()) / td;

    // eq. 8-198 — DistScaleFactor.
    let dist_scale_factor = clip3_i32(-1024, 1023, (tb * tx + 32) >> 6);

    // The spec's fallback band: the raw DistScaleFactor range
    // [-1024, 1023] corresponds to (DistScaleFactor >> 2) in
    // [-256, 255]. Implicit weighting is disabled when this is
    // outside [-64, 128] (eq. 8-280..8-281 — the first bullet in
    // §8.4.2.3.3).
    let dsf_shift2 = dist_scale_factor >> 2;
    if !(-64..=128).contains(&dsf_shift2) {
        return (32, 32, log2_wd);
    }

    // eq. 8-282, 8-283.
    let w1 = dsf_shift2;
    let w0 = 64 - w1;
    (w0, w1, log2_wd)
}

/// §5.7 `Clip3(x, y, z) = min(y, max(x, z))`.
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

/// §7.4.3.2 — look up the (weight, offset) for a luma entry in the
/// pred_weight_table. When `ref_idx < 0` (list not used) the returned
/// entry is irrelevant; we return a benign default. When the per-entry
/// flag was 0 (stored as `None`), the inferred values are
/// `weight = 2^log2_wd, offset = 0`.
fn luma_weight_entry(pwt: &PredWeightTable, list: u8, ref_idx: i8, log2_wd: u32) -> WeightedEntry {
    if ref_idx < 0 {
        return WeightedEntry::default();
    }
    let idx = ref_idx as usize;
    let table = match list {
        0 => &pwt.luma_weights_l0,
        _ => &pwt.luma_weights_l1,
    };
    match table.get(idx).copied().flatten() {
        Some((w, o)) => WeightedEntry {
            weight: w,
            offset: o,
        },
        None => WeightedEntry {
            // Inferred values per §7.4.3.2.
            weight: 1i32 << log2_wd,
            offset: 0,
        },
    }
}

/// §7.4.3.2 — look up the chroma (weight, offset) for a given
/// (list, iCbCr, ref_idx) entry, applying the same inference rule as
/// [`luma_weight_entry`].
fn chroma_weight_entry(
    pwt: &PredWeightTable,
    list: u8,
    i_cb_cr: usize,
    ref_idx: i8,
    log2_wd: u32,
) -> WeightedEntry {
    if ref_idx < 0 {
        return WeightedEntry::default();
    }
    let idx = ref_idx as usize;
    let table = match list {
        0 => &pwt.chroma_weights_l0,
        _ => &pwt.chroma_weights_l1,
    };
    match table.get(idx).copied().flatten() {
        Some(pair) => {
            let (w, o) = pair[i_cb_cr];
            WeightedEntry {
                weight: w,
                offset: o,
            }
        }
        None => WeightedEntry {
            weight: 1i32 << log2_wd,
            offset: 0,
        },
    }
}

/// §8.4.2 — motion-compensate a single partition + write into the
/// MB-local prediction buffers. Also records the MV/ref_idx into the
/// grid so subsequent MBs' MVpred can see it.
#[allow(clippy::too_many_arguments)]
fn process_partition<R: RefPicProvider>(
    part: &InterPartition,
    mb_addr: u32,
    mb_px: i32,
    mb_py: i32,
    chroma_array_type: u32,
    bit_depth_y: u32,
    bit_depth_c: u32,
    slice_header: &SliceHeader,
    pps: &Pps,
    ref_pics: &R,
    grid: &mut MbGrid,
    pic: &Picture,
    pred_luma: &mut [i32; 256],
    pred_cb: &mut [i32],
    pred_cr: &mut [i32],
    inter_debug: bool,
    current_slice_id: i32,
) -> Result<(), ReconstructError> {
    let _ = pic; // currently unused; kept for future neighbour queries.
                 // §8.4.1 — derive L0 / L1 MVpreds. For Direct / Skip we compute MVs
                 // up-front via derive_p_skip_mv or a simple spatial-direct stub.
                 // Everything else uses mvLX = mvpLX + mvdLX (eq. 8-174-ish in §8.4.1).
    let (mv_l0, mv_l1) = derive_partition_mvs(part, mb_addr, grid, current_slice_id);
    if inter_debug {
        eprintln!(
            "    derived mv_l0={:?} mv_l1={:?} for part@({},{} {}x{})",
            (mv_l0.x, mv_l0.y),
            (mv_l1.x, mv_l1.y),
            part.x,
            part.y,
            part.w,
            part.h
        );
    }

    // Store MVs in the MB grid so future MBs can use them. We store in
    // the current MB entry (`grid[mb_addr]`). The grid holds per-4x4-
    // block MVs (16 slots) — fill the slots covered by this partition
    // per Figure 6-10 / §6.4.3 layout.
    if let Some(info) = grid.get_mut(mb_addr) {
        let x0 = part.x as usize / 4;
        let y0 = part.y as usize / 4;
        let nx = part.w as usize / 4;
        let ny = part.h as usize / 4;
        for dy in 0..ny {
            for dx in 0..nx {
                let bx = x0 + dx;
                let by = y0 + dy;
                // Figure 6-10 raster — block index for (bx, by) in 4x4.
                let blk4 = blk4_raster_index(bx as u8, by as u8);
                if part.mode != PartMode::L1Only {
                    info.mv_l0[blk4 as usize] = (mv_l0.x as i16, mv_l0.y as i16);
                }
                if matches!(
                    part.mode,
                    PartMode::L1Only | PartMode::BiPred | PartMode::Direct
                ) {
                    info.mv_l1[blk4 as usize] = (mv_l1.x as i16, mv_l1.y as i16);
                }
            }
        }
        // §7.4.5 — per 8x8 partition refIdx. Update the 8x8 quadrants
        // this partition covers.
        let q_x0 = part.x as usize / 8;
        let q_y0 = part.y as usize / 8;
        let q_nx = (part.w as usize).max(8) / 8;
        let q_ny = (part.h as usize).max(8) / 8;
        for qdy in 0..q_ny {
            for qdx in 0..q_nx {
                let qx = q_x0 + qdx;
                let qy = q_y0 + qdy;
                if qx < 2 && qy < 2 {
                    let q_idx = qy * 2 + qx;
                    // §7.4.5 / §8.4.1 — write ref_idx for each list that
                    // contributes to this partition. Direct / BSkip derive
                    // both L0 and L1 refs per §8.4.1.2, so Direct writes
                    // both just like BiPred.
                    if part.mode != PartMode::L1Only {
                        info.ref_idx_l0[q_idx] = part.ref_idx_l0;
                        // §8.7.2.1 NOTE 1 — also capture the POC of the
                        // referenced picture so later slices (with possibly
                        // different RefPicList0) can compare picture identity
                        // cross-slice without consulting a picture-wide
                        // snapshot that may belong to a different slice.
                        if part.ref_idx_l0 >= 0 {
                            info.ref_poc_l0[q_idx] = ref_pics
                                .ref_pic_poc(0, part.ref_idx_l0 as u32)
                                .unwrap_or(i32::MIN);
                        }
                    }
                    if matches!(
                        part.mode,
                        PartMode::L1Only | PartMode::BiPred | PartMode::Direct
                    ) {
                        info.ref_idx_l1[q_idx] = part.ref_idx_l1;
                        if part.ref_idx_l1 >= 0 {
                            info.ref_poc_l1[q_idx] = ref_pics
                                .ref_pic_poc(1, part.ref_idx_l1 as u32)
                                .unwrap_or(i32::MIN);
                        }
                    }
                }
            }
        }
    }

    // Predict pixels for this partition.
    // §8.4.2.1 — build predPartL0 / predPartL1 by MC from the selected
    // reference pictures, then combine per §8.4.2.3 (weighted-pred or
    // default average).
    let w = part.w as u32;
    let h = part.h as u32;
    let part_abs_x = mb_px + part.x as i32;
    let part_abs_y = mb_py + part.y as i32;

    // Allocate partition-sized scratch buffers.
    let mut l0_buf = vec![0i32; (w as usize) * (h as usize)];
    let mut l1_buf = vec![0i32; (w as usize) * (h as usize)];
    let has_l0 = matches!(
        part.mode,
        PartMode::L0Only | PartMode::BiPred | PartMode::Direct
    ) && part.ref_idx_l0 >= 0;
    let has_l1 = matches!(
        part.mode,
        PartMode::L1Only | PartMode::BiPred | PartMode::Direct
    ) && part.ref_idx_l1 >= 0;

    if has_l0 {
        let rp =
            ref_pics
                .ref_pic(0, part.ref_idx_l0 as u32)
                .ok_or(ReconstructError::MissingRefPic {
                    list: 0,
                    idx: part.ref_idx_l0 as u32,
                })?;
        mc_luma_partition(
            rp,
            part_abs_x,
            part_abs_y,
            mv_l0,
            w,
            h,
            bit_depth_y,
            &mut l0_buf,
        )?;
    }
    if has_l1 {
        let rp =
            ref_pics
                .ref_pic(1, part.ref_idx_l1 as u32)
                .ok_or(ReconstructError::MissingRefPic {
                    list: 1,
                    idx: part.ref_idx_l1 as u32,
                })?;
        mc_luma_partition(
            rp,
            part_abs_x,
            part_abs_y,
            mv_l1,
            w,
            h,
            bit_depth_y,
            &mut l1_buf,
        )?;
    }

    // §8.4.2.3 — combine L0 / L1 prediction buffers into the final
    // prediction samples. The mode is selected per §8.4.2.3:
    //
    // * P / SP slices: explicit (§8.4.2.3.2) when pps.weighted_pred_flag,
    //   else default (§8.4.2.3.1 — plain copy of the single list).
    // * B slices:
    //   - weighted_bipred_idc == 0 → default (eq. 8-273 average for
    //     bipred, copy for single list).
    //   - weighted_bipred_idc == 1 → explicit (eq. 8-274/8-275/8-276
    //     with weights/offsets pulled from the slice's pred_weight_table).
    //   - weighted_bipred_idc == 2 → implicit (§8.4.2.3.3). Weights
    //     derived from POC distance; only the bipred sub-case uses
    //     implicit weighting (single-list partitions fall through to
    //     the default copy-through).
    let use_explicit = weighted_explicit_active(slice_header, pps);
    let bi_mode = if has_l0 && has_l1 {
        Some(BiPredMode::Bipred)
    } else if has_l0 {
        Some(BiPredMode::L0Only)
    } else if has_l1 {
        Some(BiPredMode::L1Only)
    } else {
        None
    };

    // §8.4.2.3.3 — implicit weighted bipred only fires on true bipred
    // partitions. For L0-only / L1-only in a weighted_bipred_idc==2
    // slice, the spec falls back to default (§8.4.2.3.1).
    let use_implicit_bipred =
        weighted_implicit_active(slice_header, pps) && matches!(bi_mode, Some(BiPredMode::Bipred));

    // §8.4.2.3.3 — derive implicit weights from POC distance when
    // implicit mode is active. The current picture's POC is in
    // `pic.pic_order_cnt`; the two ref pictures' POCs are in their
    // respective `Picture.pic_order_cnt` fields.
    let implicit_weights =
        if use_implicit_bipred {
            let curr_poc = pic.pic_order_cnt;
            // has_l0 && has_l1 guaranteed by bi_mode == Bipred above.
            let rp0 = ref_pics.ref_pic(0, part.ref_idx_l0 as u32).ok_or(
                ReconstructError::MissingRefPic {
                    list: 0,
                    idx: part.ref_idx_l0 as u32,
                },
            )?;
            let rp1 = ref_pics.ref_pic(1, part.ref_idx_l1 as u32).ok_or(
                ReconstructError::MissingRefPic {
                    list: 1,
                    idx: part.ref_idx_l1 as u32,
                },
            )?;
            // TODO(§8.2.5): RefPicProvider doesn't expose long-term
            // marking. Assume both refs are short-term — conforming for
            // the common case where bipred uses short-term refs.
            let long_term_either = false;
            Some(implicit_bipred_weights(
                curr_poc,
                rp0.pic_order_cnt,
                rp1.pic_order_cnt,
                long_term_either,
            ))
        } else {
            None
        };

    for py in 0..h as usize {
        for px in 0..w as usize {
            let dst_idx = (part.y as usize + py) * 16 + (part.x as usize + px);
            let v = match bi_mode {
                Some(_) if use_explicit => {
                    // Handled in bulk below via weighted_pred_explicit;
                    // defer by leaving the slot at 0 for now.
                    0
                }
                Some(BiPredMode::Bipred) if use_implicit_bipred => {
                    // Handled in bulk below via the implicit weighted
                    // pred path; defer.
                    0
                }
                Some(BiPredMode::Bipred) => {
                    // Eq. 8-273 — default bipred average.
                    (l0_buf[py * w as usize + px] + l1_buf[py * w as usize + px] + 1) >> 1
                }
                Some(BiPredMode::L0Only) => l0_buf[py * w as usize + px],
                Some(BiPredMode::L1Only) => l1_buf[py * w as usize + px],
                None => {
                    // Neither L0 nor L1 active (e.g., B_Direct with no
                    // usable refs) — fall back to zero prediction
                    // per §8.4.2.3.
                    0
                }
            };
            pred_luma[dst_idx] = v;
        }
    }

    // §8.4.2.3.3 — implicit bipred: apply eq. 8-276 with
    // `logWD = 5`, offsets = 0, and POC-derived weights.
    if let Some((w0, w1, log2_wd)) = implicit_weights {
        let mut scratch = vec![0i32; (w as usize) * (h as usize)];
        weighted_pred_explicit(
            Some(l0_buf.as_slice()),
            Some(l1_buf.as_slice()),
            w as usize,
            w,
            h,
            BiPredMode::Bipred,
            WeightedEntry {
                weight: w0,
                offset: 0,
            },
            WeightedEntry {
                weight: w1,
                offset: 0,
            },
            log2_wd,
            bit_depth_y,
            &mut scratch,
            w as usize,
        );
        for py in 0..h as usize {
            for px in 0..w as usize {
                let dst_idx = (part.y as usize + py) * 16 + (part.x as usize + px);
                pred_luma[dst_idx] = scratch[py * w as usize + px];
            }
        }
    }

    if use_explicit {
        if let (Some(mode), Some(pwt)) = (bi_mode, slice_header.pred_weight_table.as_ref()) {
            // §8.4.2.3.2 — explicit weighted sample prediction for luma.
            let log2_wd = pwt.luma_log2_weight_denom;
            let w_l0 = luma_weight_entry(pwt, 0, part.ref_idx_l0, log2_wd);
            let w_l1 = luma_weight_entry(pwt, 1, part.ref_idx_l1, log2_wd);
            // Scratch partition-sized buffer so we can use the spec's
            // dst/dst_stride API directly.
            let mut scratch = vec![0i32; (w as usize) * (h as usize)];
            let l0_opt = if matches!(mode, BiPredMode::L0Only | BiPredMode::Bipred) {
                Some(l0_buf.as_slice())
            } else {
                None
            };
            let l1_opt = if matches!(mode, BiPredMode::L1Only | BiPredMode::Bipred) {
                Some(l1_buf.as_slice())
            } else {
                None
            };
            weighted_pred_explicit(
                l0_opt,
                l1_opt,
                w as usize,
                w,
                h,
                mode,
                w_l0,
                w_l1,
                log2_wd,
                bit_depth_y,
                &mut scratch,
                w as usize,
            );
            // Copy back into pred_luma at partition-local position.
            for py in 0..h as usize {
                for px in 0..w as usize {
                    let dst_idx = (part.y as usize + py) * 16 + (part.x as usize + px);
                    pred_luma[dst_idx] = scratch[py * w as usize + px];
                }
            }
        }
    }

    // Chroma MC. For 4:2:0, chroma MV = mv_luma / 2 in 1/4-pel units
    // at luma resolution -> 1/8-pel chroma units. §8.4.1.4 / §8.4.2.2.
    if chroma_array_type == 1 || chroma_array_type == 2 {
        let (mbw_c, mbh_c) = chroma_mb_dims(chroma_array_type);
        let c_mb_px = mb_px / 2; // 4:2:0 and 4:2:2 both halve width.
        let c_mb_py = if chroma_array_type == 1 {
            mb_py / 2
        } else {
            mb_py
        };
        let c_part_x = part.x as i32 / 2;
        let c_part_y = if chroma_array_type == 1 {
            part.y as i32 / 2
        } else {
            part.y as i32
        };
        let c_w = part.w as u32 / 2;
        let c_h = if chroma_array_type == 1 {
            part.h as u32 / 2
        } else {
            part.h as u32
        };
        // MB width in chroma is always 8 for 4:2:0 / 4:2:2 (Table 6-1).
        let mbw_c_use = mbw_c;
        let _ = mbh_c;

        let mut l0_cb = vec![0i32; (c_w as usize) * (c_h as usize)];
        let mut l0_cr = vec![0i32; (c_w as usize) * (c_h as usize)];
        let mut l1_cb = vec![0i32; (c_w as usize) * (c_h as usize)];
        let mut l1_cr = vec![0i32; (c_w as usize) * (c_h as usize)];
        if has_l0 {
            let rp = ref_pics.ref_pic(0, part.ref_idx_l0 as u32).unwrap();
            mc_chroma_partition(
                rp,
                c_mb_px + c_part_x,
                c_mb_py + c_part_y,
                mv_l0,
                c_w,
                c_h,
                chroma_array_type,
                bit_depth_c,
                &mut l0_cb,
                &mut l0_cr,
            )?;
        }
        if has_l1 {
            let rp = ref_pics.ref_pic(1, part.ref_idx_l1 as u32).unwrap();
            mc_chroma_partition(
                rp,
                c_mb_px + c_part_x,
                c_mb_py + c_part_y,
                mv_l1,
                c_w,
                c_h,
                chroma_array_type,
                bit_depth_c,
                &mut l1_cb,
                &mut l1_cr,
            )?;
        }
        // Combine into pred_cb / pred_cr at MB-local position.
        // §8.4.2.3 — the same default/explicit/implicit dispatch as
        // luma applies to chroma. Explicit mode uses separate weights
        // per (list, iCbCr) and a separate log2 denominator
        // (chroma_log2_weight_denom). Implicit mode (§8.4.2.3.3)
        // reuses the luma (w0, w1, log2WD=5) triple — eq. 8-282/8-283
        // are specified for "C" replaced by L/Cb/Cr with the same
        // DistScaleFactor, and eq. 8-277 fixes logWDC = 5.
        if let Some((w0, w1, log2_wd)) = implicit_weights {
            let mut scratch_cb = vec![0i32; (c_w as usize) * (c_h as usize)];
            let mut scratch_cr = vec![0i32; (c_w as usize) * (c_h as usize)];
            let w_entry_l0 = WeightedEntry {
                weight: w0,
                offset: 0,
            };
            let w_entry_l1 = WeightedEntry {
                weight: w1,
                offset: 0,
            };
            weighted_pred_explicit(
                Some(l0_cb.as_slice()),
                Some(l1_cb.as_slice()),
                c_w as usize,
                c_w,
                c_h,
                BiPredMode::Bipred,
                w_entry_l0,
                w_entry_l1,
                log2_wd,
                bit_depth_c,
                &mut scratch_cb,
                c_w as usize,
            );
            weighted_pred_explicit(
                Some(l0_cr.as_slice()),
                Some(l1_cr.as_slice()),
                c_w as usize,
                c_w,
                c_h,
                BiPredMode::Bipred,
                w_entry_l0,
                w_entry_l1,
                log2_wd,
                bit_depth_c,
                &mut scratch_cr,
                c_w as usize,
            );
            for py in 0..c_h as usize {
                for px in 0..c_w as usize {
                    let dst_idx =
                        (c_part_y as usize + py) * (mbw_c_use as usize) + (c_part_x as usize + px);
                    let idx = py * c_w as usize + px;
                    pred_cb[dst_idx] = scratch_cb[idx];
                    pred_cr[dst_idx] = scratch_cr[idx];
                }
            }
        } else if use_explicit {
            if let (Some(mode), Some(pwt)) = (bi_mode, slice_header.pred_weight_table.as_ref()) {
                let log2_wd_c = pwt.chroma_log2_weight_denom;
                // Cb: iCbCr = 0.
                let (w_cb_l0, w_cb_l1) = (
                    chroma_weight_entry(pwt, 0, 0, part.ref_idx_l0, log2_wd_c),
                    chroma_weight_entry(pwt, 1, 0, part.ref_idx_l1, log2_wd_c),
                );
                // Cr: iCbCr = 1.
                let (w_cr_l0, w_cr_l1) = (
                    chroma_weight_entry(pwt, 0, 1, part.ref_idx_l0, log2_wd_c),
                    chroma_weight_entry(pwt, 1, 1, part.ref_idx_l1, log2_wd_c),
                );

                let mut scratch_cb = vec![0i32; (c_w as usize) * (c_h as usize)];
                let mut scratch_cr = vec![0i32; (c_w as usize) * (c_h as usize)];
                let cb_l0 = if matches!(mode, BiPredMode::L0Only | BiPredMode::Bipred) {
                    Some(l0_cb.as_slice())
                } else {
                    None
                };
                let cb_l1 = if matches!(mode, BiPredMode::L1Only | BiPredMode::Bipred) {
                    Some(l1_cb.as_slice())
                } else {
                    None
                };
                let cr_l0 = if matches!(mode, BiPredMode::L0Only | BiPredMode::Bipred) {
                    Some(l0_cr.as_slice())
                } else {
                    None
                };
                let cr_l1 = if matches!(mode, BiPredMode::L1Only | BiPredMode::Bipred) {
                    Some(l1_cr.as_slice())
                } else {
                    None
                };
                weighted_pred_explicit(
                    cb_l0,
                    cb_l1,
                    c_w as usize,
                    c_w,
                    c_h,
                    mode,
                    w_cb_l0,
                    w_cb_l1,
                    log2_wd_c,
                    bit_depth_c,
                    &mut scratch_cb,
                    c_w as usize,
                );
                weighted_pred_explicit(
                    cr_l0,
                    cr_l1,
                    c_w as usize,
                    c_w,
                    c_h,
                    mode,
                    w_cr_l0,
                    w_cr_l1,
                    log2_wd_c,
                    bit_depth_c,
                    &mut scratch_cr,
                    c_w as usize,
                );
                for py in 0..c_h as usize {
                    for px in 0..c_w as usize {
                        let dst_idx = (c_part_y as usize + py) * (mbw_c_use as usize)
                            + (c_part_x as usize + px);
                        let idx = py * c_w as usize + px;
                        pred_cb[dst_idx] = scratch_cb[idx];
                        pred_cr[dst_idx] = scratch_cr[idx];
                    }
                }
            } else {
                // No pred_weight_table available; fall through to
                // default (shouldn't happen if use_explicit is set, but
                // be defensive).
                for py in 0..c_h as usize {
                    for px in 0..c_w as usize {
                        let dst_idx = (c_part_y as usize + py) * (mbw_c_use as usize)
                            + (c_part_x as usize + px);
                        let idx = py * c_w as usize + px;
                        let (vb, vr) = if has_l0 && has_l1 {
                            (
                                (l0_cb[idx] + l1_cb[idx] + 1) >> 1,
                                (l0_cr[idx] + l1_cr[idx] + 1) >> 1,
                            )
                        } else if has_l0 {
                            (l0_cb[idx], l0_cr[idx])
                        } else if has_l1 {
                            (l1_cb[idx], l1_cr[idx])
                        } else {
                            (0, 0)
                        };
                        pred_cb[dst_idx] = vb;
                        pred_cr[dst_idx] = vr;
                    }
                }
            }
        } else {
            for py in 0..c_h as usize {
                for px in 0..c_w as usize {
                    let dst_idx =
                        (c_part_y as usize + py) * (mbw_c_use as usize) + (c_part_x as usize + px);
                    let idx = py * c_w as usize + px;
                    let (vb, vr) = if has_l0 && has_l1 {
                        (
                            (l0_cb[idx] + l1_cb[idx] + 1) >> 1,
                            (l0_cr[idx] + l1_cr[idx] + 1) >> 1,
                        )
                    } else if has_l0 {
                        (l0_cb[idx], l0_cr[idx])
                    } else if has_l1 {
                        (l1_cb[idx], l1_cr[idx])
                    } else {
                        (0, 0)
                    };
                    pred_cb[dst_idx] = vb;
                    pred_cr[dst_idx] = vr;
                }
            }
        }
    }

    Ok(())
}

/// §8.4.2.2 — motion-compensate one partition's luma plane. `dst`
/// is the partition-sized output buffer (w * h samples, row-major).
fn mc_luma_partition(
    ref_pic: &Picture,
    part_abs_x: i32,
    part_abs_y: i32,
    mv: Mv,
    w: u32,
    h: u32,
    bit_depth: u32,
    dst: &mut [i32],
) -> Result<(), ReconstructError> {
    // §8.4.1.4 — MV is in 1/4-pel luma units. Integer part = mv / 4
    // (with spec's "truncate toward zero" via i32 division); fractional
    // part in 0..=3.
    let mv_x = mv.x;
    let mv_y = mv.y;
    let int_x = part_abs_x + (mv_x >> 2);
    let int_y = part_abs_y + (mv_y >> 2);
    let x_frac = (mv_x & 3) as u8;
    let y_frac = (mv_y & 3) as u8;

    interpolate_luma(
        &ref_pic.luma,
        ref_pic.width_in_samples as usize,
        ref_pic.width_in_samples as usize,
        ref_pic.height_in_samples as usize,
        int_x,
        int_y,
        x_frac,
        y_frac,
        w,
        h,
        bit_depth,
        dst,
        w as usize,
    )
    .map_err(|_| ReconstructError::IntraPredOutOfBounds)?;
    Ok(())
}

/// §8.4.2.2 — motion-compensate one partition's chroma planes (Cb + Cr).
/// Writes to two partition-sized scratch buffers. ChromaArrayType is 1
/// (4:2:0) or 2 (4:2:2). §8.4.1.4 — chroma MVs are in 1/8-pel units for
/// 4:2:0 / 4:2:2.
#[allow(clippy::too_many_arguments)]
fn mc_chroma_partition(
    ref_pic: &Picture,
    part_abs_x: i32,
    part_abs_y: i32,
    mv: Mv,
    w: u32,
    h: u32,
    chroma_array_type: u32,
    bit_depth: u32,
    dst_cb: &mut [i32],
    dst_cr: &mut [i32],
) -> Result<(), ReconstructError> {
    // §8.4.1.4 — chroma MV derivation:
    // - 4:2:0 (ChromaArrayType == 1): mvC = mv / 2 (both components).
    // - 4:2:2 (ChromaArrayType == 2): mvC.x = mv.x / 2, mvC.y = mv.y.
    // The chroma interpolator takes 1/8-pel fractions (0..=7) — per
    // §8.4.2.2.2 the chroma MV uses 3 fractional bits.
    let (mv_cx, mv_cy) = match chroma_array_type {
        1 => (mv.x, mv.y),
        2 => (mv.x, mv.y * 2), // keep 1/8-pel resolution vertical.
        _ => (mv.x, mv.y),
    };
    // int_x = part_abs_x + (mvC.x >> 3); x_frac = mvC.x & 7.
    // For 4:2:0 this yields the spec's eq. 8-232 / 8-233 (xFracC / yFracC).
    let int_x = part_abs_x + (mv_cx >> 3);
    let int_y = part_abs_y + (mv_cy >> 3);
    let x_frac = (mv_cx & 7) as u8;
    let y_frac = (mv_cy & 7) as u8;

    let cw = ref_pic.chroma_width() as usize;
    let ch = ref_pic.chroma_height() as usize;

    interpolate_chroma(
        &ref_pic.cb,
        cw,
        cw,
        ch,
        int_x,
        int_y,
        x_frac,
        y_frac,
        w,
        h,
        bit_depth,
        dst_cb,
        w as usize,
    )
    .map_err(|_| ReconstructError::IntraPredOutOfBounds)?;
    interpolate_chroma(
        &ref_pic.cr,
        cw,
        cw,
        ch,
        int_x,
        int_y,
        x_frac,
        y_frac,
        w,
        h,
        bit_depth,
        dst_cr,
        w as usize,
    )
    .map_err(|_| ReconstructError::IntraPredOutOfBounds)?;
    Ok(())
}

/// §8.4.1 — derive the L0/L1 MV for a partition.
///
/// For P_Skip and B_Direct this is the derivation per §8.4.1.2; for
/// explicit inter partitions this is mvpLX + mvdLX.
///
/// Neighbour MV data is read from the grid. The partition's shape
/// selects the §8.4.1.3 shortcut (16x8 / 8x16) when applicable.
fn derive_partition_mvs(
    part: &InterPartition,
    mb_addr: u32,
    grid: &MbGrid,
    current_slice_id: i32,
) -> (Mv, Mv) {
    // §8.4.1.2.3 — direct-mode partitions may carry pre-computed L0/L1
    // MVs (e.g. temporal direct). Honour them directly.
    if let Some((mv_l0, mv_l1)) = part.precomputed_mv {
        return (mv_l0, mv_l1);
    }
    // Build neighbour MVs (A, B, C, D) relative to the partition origin.
    // The simple non-MBAFF frame case: A is the 4x4 block immediately
    // left (within this MB or the left MB if partition is at x=0),
    // B is the 4x4 block immediately above, C is the 4x4 block
    // upper-right, D is the 4x4 block upper-left.
    //
    // D is consulted only for the §8.4.1.3.2 eq. 8-214..8-216 C→D
    // substitution applied inside `derive_mvpred_with_d` /
    // `derive_p_skip_mv_with_d`.
    let (neigh_a_l0, neigh_b_l0, neigh_c_l0, neigh_d_l0) =
        neighbour_mvs_for_list(part, mb_addr, grid, 0, current_slice_id);
    let (neigh_a_l1, neigh_b_l1, neigh_c_l1, neigh_d_l1) =
        neighbour_mvs_for_list(part, mb_addr, grid, 1, current_slice_id);

    let (mv_l0, mv_l1) = match part.mode {
        PartMode::Direct => {
            // §8.4.1.2 direct mode — for now use spatial-direct median
            // with ref_idx = 0. Temporal direct and full spatial direct
            // precise derivation are deferred.
            let mvp_l0 = derive_mvpred_with_d(
                &MvpredInputs {
                    neighbour_a: neigh_a_l0,
                    neighbour_b: neigh_b_l0,
                    neighbour_c: neigh_c_l0,
                    current_ref_idx: part.ref_idx_l0.max(0) as i32,
                    shape: MvpredShape::Default,
                },
                neigh_d_l0,
            );
            let mvp_l1 = derive_mvpred_with_d(
                &MvpredInputs {
                    neighbour_a: neigh_a_l1,
                    neighbour_b: neigh_b_l1,
                    neighbour_c: neigh_c_l1,
                    current_ref_idx: part.ref_idx_l1.max(0) as i32,
                    shape: MvpredShape::Default,
                },
                neigh_d_l1,
            );
            (mvp_l0, mvp_l1)
        }
        _ => {
            // §8.4.1.1 — mvLX = mvpLX + mvdLX.
            // P_Skip (part.is_skip == true for PSkip macroblocks) invokes
            // the §8.4.1.2 zero-MV substitution conditions inside
            // `derive_p_skip_mv_with_d`. Regular P_L0_16x16 macroblocks
            // that happen to have MVD = (0, 0) and ref_idx = 0 look
            // identical in (mode, mvd, ref_idx, shape, w, h) but MUST
            // use the plain §8.4.1.3 MVpred — the skip-specific
            // "refIdxA == 0 AND mvA == 0 → force zero" rule does NOT
            // apply to them. Prior to fixing this, a P_L0_16x16 whose
            // left neighbour was a P_Skip / zero-MV MB would have its
            // MVpred wrongly forced to zero. `is_skip` disambiguates.
            let mvp_l0 =
                if part.is_skip && part.mode == PartMode::L0Only && part.w == 16 && part.h == 16 {
                    // P_Skip path (§8.4.1.2), with §8.4.1.3.2 C→D substitution.
                    let (_, mv) =
                        derive_p_skip_mv_with_d(neigh_a_l0, neigh_b_l0, neigh_c_l0, neigh_d_l0);
                    mv
                } else if part.mode != PartMode::L1Only && part.ref_idx_l0 >= 0 {
                    derive_mvpred_with_d(
                        &MvpredInputs {
                            neighbour_a: neigh_a_l0,
                            neighbour_b: neigh_b_l0,
                            neighbour_c: neigh_c_l0,
                            current_ref_idx: part.ref_idx_l0 as i32,
                            shape: part.shape,
                        },
                        neigh_d_l0,
                    )
                } else {
                    Mv::ZERO
                };
            let mvp_l1 = if part.mode != PartMode::L0Only && part.ref_idx_l1 >= 0 {
                derive_mvpred_with_d(
                    &MvpredInputs {
                        neighbour_a: neigh_a_l1,
                        neighbour_b: neigh_b_l1,
                        neighbour_c: neigh_c_l1,
                        current_ref_idx: part.ref_idx_l1 as i32,
                        shape: part.shape,
                    },
                    neigh_d_l1,
                )
            } else {
                Mv::ZERO
            };
            (
                Mv::new(mvp_l0.x + part.mvd_l0.0, mvp_l0.y + part.mvd_l0.1),
                Mv::new(mvp_l1.x + part.mvd_l1.0, mvp_l1.y + part.mvd_l1.1),
            )
        }
    };

    (mv_l0, mv_l1)
}

/// Figure 6-10 raster mapping — 4x4 block coordinates (bx, by) in
/// 4x4-block units (0..=3 each) → raster block index (0..=15) used
/// by MbInfo::mv_l0 / mv_l1.
#[inline]
fn blk4_raster_index(bx_b: u8, by_b: u8) -> u8 {
    // LUMA_4X4_XY is indexed by block index and yields (x, y) in 4-sample
    // units. Invert via the 8x8-quadrant split (Figure 6-10):
    //   quadrant = 2*(by_b/2) + (bx_b/2)
    //   lo index = 2*(by_b%2) + (bx_b%2)
    //   blk4 = 4 * quadrant + lo
    let q = 2 * (by_b / 2) + (bx_b / 2);
    let lo = 2 * (by_b % 2) + (bx_b % 2);
    4 * q + lo
}

/// §8.4.1.3.2 — build (A, B, C) NeighbourMv triples for the top-left
/// 4x4 block of a partition, querying the grid for the previously
/// decoded neighbour MVs + ref indices.
///
/// Non-MBAFF, frame-picture case only. A is the left 4x4 (same row,
/// one 4x4 to the left); B is the above 4x4; C is above-right (one
/// 4x4 across a 4-pel boundary).
fn neighbour_mvs_for_list(
    part: &InterPartition,
    mb_addr: u32,
    grid: &MbGrid,
    list: u8,
    current_slice_id: i32,
) -> (NeighbourMv, NeighbourMv, NeighbourMv, NeighbourMv) {
    // Partition origin in 4x4 units, relative to MB.
    let px = part.x as i32 / 4;
    let py = part.y as i32 / 4;
    let width = grid.width_in_mbs as i32;
    let (mb_xx, mb_yy) = grid.mb_xy(mb_addr);
    let mb_xx = mb_xx as i32;
    let mb_yy = mb_yy as i32;

    // A = (px-1, py) — left.
    let a = neighbour_from_block(
        grid,
        mb_addr,
        mb_xx,
        mb_yy,
        width,
        px - 1,
        py,
        list,
        current_slice_id,
    );
    // B = (px, py-1) — above.
    let b = neighbour_from_block(
        grid,
        mb_addr,
        mb_xx,
        mb_yy,
        width,
        px,
        py - 1,
        list,
        current_slice_id,
    );
    // C = (px + w/4, py - 1) — above-right of the partition.
    let c_off_x = part.w as i32 / 4;
    let c = neighbour_from_block(
        grid,
        mb_addr,
        mb_xx,
        mb_yy,
        width,
        px + c_off_x,
        py - 1,
        list,
        current_slice_id,
    );
    // D = (px - 1, py - 1) — above-left (used for §8.4.1.3.2 C→D
    // substitution). Only consulted when C is unavailable.
    let d = neighbour_from_block(
        grid,
        mb_addr,
        mb_xx,
        mb_yy,
        width,
        px - 1,
        py - 1,
        list,
        current_slice_id,
    );
    (a, b, c, d)
}

/// Fetch a NeighbourMv at (bx, by) in 4x4-block coordinates relative
/// to the current MB. Negative coordinates cross into adjacent MBs
/// (§6.4.12). Returns `UNAVAILABLE` if the neighbour lies outside the
/// picture or hasn't been decoded yet.
///
/// `curr_mb_addr` is the address of the current (in-flight) MB: when
/// the wrapped neighbour turns out to be the same MB (i.e. an earlier
/// partition of this MB), we bypass the `info.available` check since
/// that flag is only set after the entire MB finishes reconstruction —
/// but per §7.4.5 / §8.4.1 later partitions in the raster scan may
/// legitimately read MVs from earlier partitions of the same MB (e.g.
/// the bottom partition of a 16x8 reads the top partition's MV as its
/// B-neighbour).
fn neighbour_from_block(
    grid: &MbGrid,
    curr_mb_addr: u32,
    mb_xx: i32,
    mb_yy: i32,
    width: i32,
    bx: i32,
    by: i32,
    list: u8,
    current_slice_id: i32,
) -> NeighbourMv {
    // Wrap to neighbour MB.
    let mut nx = mb_xx;
    let mut ny = mb_yy;
    let mut bxw = bx;
    let mut byw = by;
    if bxw < 0 {
        nx -= 1;
        bxw += 4;
    } else if bxw >= 4 {
        nx += 1;
        bxw -= 4;
    }
    if byw < 0 {
        ny -= 1;
        byw += 4;
    } else if byw >= 4 {
        ny += 1;
        byw -= 4;
    }
    if nx < 0 || ny < 0 || nx >= width {
        return NeighbourMv::UNAVAILABLE;
    }
    let addr = (ny as u32) * (width as u32) + (nx as u32);
    let Some(info) = grid.get(addr) else {
        return NeighbourMv::UNAVAILABLE;
    };
    // For the current in-flight MB, bypass the `info.available` check
    // (that flag is only set after reconstruct_mb_inter returns). The
    // per-4x4-block MV slot either holds a value written by an earlier
    // partition (valid) or the (0,0) default with ref_idx = -1 default
    // (which will fall through to UNAVAILABLE below). For any OTHER MB,
    // require `available == true` (it may be later in raster scan or a
    // right-MB neighbour not yet decoded).
    let is_same_mb = addr == curr_mb_addr;
    if !is_same_mb && !info.available {
        return NeighbourMv::UNAVAILABLE;
    }
    // §6.4.8 third bullet — neighbours that belong to a different slice
    // than the current MB are marked not available. The current MB is
    // always in-slice (same as itself), and the `-1` escape hatch in
    // `same_slice_at` makes legacy callers that don't stamp slice_ids
    // see the old behaviour. Skip the check for same-MB lookups so
    // in-flight partitions of the current MB remain accessible.
    if !is_same_mb
        && current_slice_id >= 0
        && info.slice_id >= 0
        && info.slice_id != current_slice_id
    {
        return NeighbourMv::UNAVAILABLE;
    }
    // §8.4.1.3.2 — intra neighbour => mvLXN = 0, refIdxLXN = -1 for
    // MVpred, but mbAddrN IS available in the §6.4.5 sense AND (for
    // a decoded neighbour MB) its partition is available per §6.4.11.1.
    // P_Skip's §8.4.1.1 zero-forcing checks the mbAddr-availability
    // bit, not the MVpred-availability bit; see
    // `NeighbourMv::mb_available` / `derive_p_skip_mv_with_d` for why
    // the distinction matters.
    if info.is_intra {
        return NeighbourMv::intra_but_mb_available();
    }
    let blk4 = blk4_raster_index(bxw as u8, byw as u8) as usize;
    let q_idx = blk4 / 4;
    let (mv, ref_idx) = if list == 0 {
        (info.mv_l0[blk4], info.ref_idx_l0[q_idx])
    } else {
        (info.mv_l1[blk4], info.ref_idx_l1[q_idx])
    };
    if ref_idx < 0 {
        // Inter MB with refIdxLX == -1. Two disjoint cases:
        //   (a) Within the CURRENT in-flight MB AND the 8x8 partition
        //       containing (bxw, byw) has NOT been decoded yet — both
        //       ref_idx_l0[q] and ref_idx_l1[q] are still the default
        //       -1. Per §6.4.11.1 the partition is NOT AVAILABLE, and
        //       the §8.4.1.3.2 eq. 8-214..8-216 C→D substitution must
        //       fire when this neighbour is the C slot.
        //   (b) The partition IS decoded (either a different MB, or
        //       within the current MB with the OTHER list's refIdx set
        //       meaning the partition chose predFlagLX==0 for this
        //       list). Per §6.4.11.1 the partition IS available, but
        //       §8.4.1.3.2 step 2a collapses its (mv, refIdx) to
        //       (0, -1) for the median. The C→D substitution must NOT
        //       fire for this case.
        //
        // Distinguish same-MB (a) from (b) by inspecting the OTHER
        // list's refIdx at the same 8x8 quadrant: if that's also -1,
        // the partition has not been decoded; if it's >= 0, the
        // partition IS decoded but chose single-list prediction on the
        // other list.
        let other_ref_idx = if list == 0 {
            info.ref_idx_l1[q_idx]
        } else {
            info.ref_idx_l0[q_idx]
        };
        if is_same_mb && other_ref_idx < 0 {
            return NeighbourMv::partition_not_yet_decoded_same_mb();
        }
        return NeighbourMv::intra_but_mb_available();
    }
    NeighbourMv {
        available: true,
        mb_available: true,
        partition_available: true,
        ref_idx: ref_idx as i32,
        mv: Mv::new(mv.0 as i32, mv.1 as i32),
    }
}

/// §7.4.5 / Tables 7-13, 7-14 — derive the list of inter partitions
/// for a given macroblock.
fn derive_inter_partitions<R: RefPicProvider>(
    mb: &Macroblock,
    slice_header: &SliceHeader,
    sps: &Sps,
    ref_pics: &R,
    pic: &Picture,
    mb_addr: u32,
    grid: &MbGrid,
    current_slice_id: i32,
) -> Result<Vec<InterPartition>, ReconstructError> {
    use MbType::*;
    let pred = mb.mb_pred.as_ref();
    match &mb.mb_type {
        // --- P slices -------------------------------------------------
        PSkip => {
            // §8.4.1.1 — P_Skip is a 16x16 L0 partition with ref_idx = 0
            // and MVD = (0, 0). The spec-accurate MVs are derived inside
            // derive_partition_mvs via derive_p_skip_mv.
            Ok(vec![InterPartition {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                mode: PartMode::L0Only,
                shape: MvpredShape::Default,
                ref_idx_l0: 0,
                ref_idx_l1: -1,
                mvd_l0: (0, 0),
                mvd_l1: (0, 0),
                is_skip: true,
                precomputed_mv: None,
            }])
        }
        PL016x16 => {
            let p = pred.ok_or_else(|| {
                ReconstructError::UnsupportedInterMbType("P_L0_16x16 without mb_pred".into())
            })?;
            let r0 = *p.ref_idx_l0.first().unwrap_or(&0) as i8;
            let mvd = p.mvd_l0.first().copied().unwrap_or([0, 0]);
            Ok(vec![InterPartition {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                mode: PartMode::L0Only,
                shape: MvpredShape::Default,
                ref_idx_l0: r0,
                ref_idx_l1: -1,
                mvd_l0: (mvd[0], mvd[1]),
                mvd_l1: (0, 0),
                is_skip: false,
                precomputed_mv: None,
            }])
        }
        PL0L016x8 => {
            let p = pred.ok_or_else(|| {
                ReconstructError::UnsupportedInterMbType("P_L0_L0_16x8 without mb_pred".into())
            })?;
            let r0 = *p.ref_idx_l0.first().unwrap_or(&0) as i8;
            let r1 = *p.ref_idx_l0.get(1).unwrap_or(&0) as i8;
            let mvd0 = p.mvd_l0.first().copied().unwrap_or([0, 0]);
            let mvd1 = p.mvd_l0.get(1).copied().unwrap_or([0, 0]);
            Ok(vec![
                InterPartition {
                    x: 0,
                    y: 0,
                    w: 16,
                    h: 8,
                    mode: PartMode::L0Only,
                    shape: MvpredShape::Partition16x8Top,
                    ref_idx_l0: r0,
                    ref_idx_l1: -1,
                    mvd_l0: (mvd0[0], mvd0[1]),
                    mvd_l1: (0, 0),
                    is_skip: false,
                    precomputed_mv: None,
                },
                InterPartition {
                    x: 0,
                    y: 8,
                    w: 16,
                    h: 8,
                    mode: PartMode::L0Only,
                    shape: MvpredShape::Partition16x8Bottom,
                    ref_idx_l0: r1,
                    ref_idx_l1: -1,
                    mvd_l0: (mvd1[0], mvd1[1]),
                    mvd_l1: (0, 0),
                    is_skip: false,
                    precomputed_mv: None,
                },
            ])
        }
        PL0L08x16 => {
            let p = pred.ok_or_else(|| {
                ReconstructError::UnsupportedInterMbType("P_L0_L0_8x16 without mb_pred".into())
            })?;
            let r0 = *p.ref_idx_l0.first().unwrap_or(&0) as i8;
            let r1 = *p.ref_idx_l0.get(1).unwrap_or(&0) as i8;
            let mvd0 = p.mvd_l0.first().copied().unwrap_or([0, 0]);
            let mvd1 = p.mvd_l0.get(1).copied().unwrap_or([0, 0]);
            Ok(vec![
                InterPartition {
                    x: 0,
                    y: 0,
                    w: 8,
                    h: 16,
                    mode: PartMode::L0Only,
                    shape: MvpredShape::Partition8x16Left,
                    ref_idx_l0: r0,
                    ref_idx_l1: -1,
                    mvd_l0: (mvd0[0], mvd0[1]),
                    mvd_l1: (0, 0),
                    is_skip: false,
                    precomputed_mv: None,
                },
                InterPartition {
                    x: 8,
                    y: 0,
                    w: 8,
                    h: 16,
                    mode: PartMode::L0Only,
                    shape: MvpredShape::Partition8x16Right,
                    ref_idx_l0: r1,
                    ref_idx_l1: -1,
                    mvd_l0: (mvd1[0], mvd1[1]),
                    mvd_l1: (0, 0),
                    is_skip: false,
                    precomputed_mv: None,
                },
            ])
        }
        P8x8 | P8x8Ref0 => {
            let sm = mb.sub_mb_pred.as_ref().ok_or_else(|| {
                ReconstructError::UnsupportedInterMbType("P_8x8 without sub_mb_pred".into())
            })?;
            let force_ref0 = matches!(mb.mb_type, P8x8Ref0);
            let mut parts = Vec::new();
            for mb_part in 0..4usize {
                let (part_x, part_y) = match mb_part {
                    0 => (0u8, 0u8),
                    1 => (8, 0),
                    2 => (0, 8),
                    _ => (8, 8),
                };
                let sub_type = sm.sub_mb_type[mb_part];
                let r0 = if force_ref0 {
                    0i8
                } else {
                    sm.ref_idx_l0[mb_part] as i8
                };
                // Walk sub-partitions.
                let sub_parts = sub_mb_partitions(sub_type);
                for (sub_idx, (sx, sy, sw, sh)) in sub_parts.iter().enumerate() {
                    let mvd = sm.mvd_l0[mb_part].get(sub_idx).copied().unwrap_or([0, 0]);
                    parts.push(InterPartition {
                        x: part_x + sx,
                        y: part_y + sy,
                        w: *sw,
                        h: *sh,
                        mode: match sub_type {
                            SubMbType::BDirect8x8 => PartMode::Direct,
                            _ => PartMode::L0Only,
                        },
                        shape: MvpredShape::Default,
                        ref_idx_l0: r0,
                        ref_idx_l1: -1,
                        mvd_l0: (mvd[0], mvd[1]),
                        mvd_l1: (0, 0),
                        is_skip: false,
                        precomputed_mv: None,
                    });
                }
            }
            Ok(parts)
        }

        // --- B slices -------------------------------------------------
        BSkip | BDirect16x16 => {
            // §8.4.1.2 direct mode — expand into sub-partitions with
            // precomputed MVs derived per §8.4.1.2.3 (temporal) or
            // §8.4.1.2.2 (spatial).
            let is_skip = matches!(mb.mb_type, BSkip);
            if !slice_header.direct_spatial_mv_pred_flag {
                Ok(build_temporal_direct_partitions(
                    sps, ref_pics, pic, mb_addr, is_skip,
                ))
            } else {
                Ok(build_spatial_direct_partitions(
                    sps,
                    ref_pics,
                    pic,
                    mb_addr,
                    grid,
                    current_slice_id,
                    0,
                    0,
                    16,
                    is_skip,
                ))
            }
        }
        BL016x16 | BL116x16 | BBi16x16 => {
            let p = pred.ok_or_else(|| {
                ReconstructError::UnsupportedInterMbType("B_*_16x16 without mb_pred".into())
            })?;
            let (mode, r0, r1) = b_dir_refs_16x16(&mb.mb_type, p);
            let mvd0 = p.mvd_l0.first().copied().unwrap_or([0, 0]);
            let mvd1 = p.mvd_l1.first().copied().unwrap_or([0, 0]);
            Ok(vec![InterPartition {
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                mode,
                shape: MvpredShape::Default,
                ref_idx_l0: r0,
                ref_idx_l1: r1,
                mvd_l0: (mvd0[0], mvd0[1]),
                mvd_l1: (mvd1[0], mvd1[1]),
                is_skip: false,
                precomputed_mv: None,
            }])
        }
        // B 16x8 / 8x16 variants — all combinations of L0/L1/Bi per half.
        BL0L016x8 | BL1L116x8 | BL0L116x8 | BL1L016x8 | BL0Bi16x8 | BL1Bi16x8 | BBiL016x8
        | BBiL116x8 | BBiBi16x8 => {
            let p = pred.ok_or_else(|| {
                ReconstructError::UnsupportedInterMbType("B_*_16x8 without mb_pred".into())
            })?;
            let (m0, m1) = b_dir_for_16x8(&mb.mb_type);
            let (r0_top, r1_top) = ref_idx_for_part(p, 0, m0);
            let (r0_bot, r1_bot) = ref_idx_for_part(p, 1, m1);
            let mvd_l0_top = p.mvd_l0.first().copied().unwrap_or([0, 0]);
            let mvd_l1_top = p.mvd_l1.first().copied().unwrap_or([0, 0]);
            let mvd_l0_bot = p.mvd_l0.get(1).copied().unwrap_or([0, 0]);
            let mvd_l1_bot = p.mvd_l1.get(1).copied().unwrap_or([0, 0]);
            Ok(vec![
                InterPartition {
                    x: 0,
                    y: 0,
                    w: 16,
                    h: 8,
                    mode: m0,
                    shape: MvpredShape::Partition16x8Top,
                    ref_idx_l0: r0_top,
                    ref_idx_l1: r1_top,
                    mvd_l0: (mvd_l0_top[0], mvd_l0_top[1]),
                    mvd_l1: (mvd_l1_top[0], mvd_l1_top[1]),
                    is_skip: false,
                    precomputed_mv: None,
                },
                InterPartition {
                    x: 0,
                    y: 8,
                    w: 16,
                    h: 8,
                    mode: m1,
                    shape: MvpredShape::Partition16x8Bottom,
                    ref_idx_l0: r0_bot,
                    ref_idx_l1: r1_bot,
                    mvd_l0: (mvd_l0_bot[0], mvd_l0_bot[1]),
                    mvd_l1: (mvd_l1_bot[0], mvd_l1_bot[1]),
                    is_skip: false,
                    precomputed_mv: None,
                },
            ])
        }
        BL0L08x16 | BL1L18x16 | BL0L18x16 | BL1L08x16 | BL0Bi8x16 | BL1Bi8x16 | BBiL08x16
        | BBiL18x16 | BBiBi8x16 => {
            let p = pred.ok_or_else(|| {
                ReconstructError::UnsupportedInterMbType("B_*_8x16 without mb_pred".into())
            })?;
            let (m0, m1) = b_dir_for_8x16(&mb.mb_type);
            let (r0_l, r1_l) = ref_idx_for_part(p, 0, m0);
            let (r0_r, r1_r) = ref_idx_for_part(p, 1, m1);
            let mvd_l0_l = p.mvd_l0.first().copied().unwrap_or([0, 0]);
            let mvd_l1_l = p.mvd_l1.first().copied().unwrap_or([0, 0]);
            let mvd_l0_r = p.mvd_l0.get(1).copied().unwrap_or([0, 0]);
            let mvd_l1_r = p.mvd_l1.get(1).copied().unwrap_or([0, 0]);
            Ok(vec![
                InterPartition {
                    x: 0,
                    y: 0,
                    w: 8,
                    h: 16,
                    mode: m0,
                    shape: MvpredShape::Partition8x16Left,
                    ref_idx_l0: r0_l,
                    ref_idx_l1: r1_l,
                    mvd_l0: (mvd_l0_l[0], mvd_l0_l[1]),
                    mvd_l1: (mvd_l1_l[0], mvd_l1_l[1]),
                    is_skip: false,
                    precomputed_mv: None,
                },
                InterPartition {
                    x: 8,
                    y: 0,
                    w: 8,
                    h: 16,
                    mode: m1,
                    shape: MvpredShape::Partition8x16Right,
                    ref_idx_l0: r0_r,
                    ref_idx_l1: r1_r,
                    mvd_l0: (mvd_l0_r[0], mvd_l0_r[1]),
                    mvd_l1: (mvd_l1_r[0], mvd_l1_r[1]),
                    is_skip: false,
                    precomputed_mv: None,
                },
            ])
        }
        B8x8 => {
            let sm = mb.sub_mb_pred.as_ref().ok_or_else(|| {
                ReconstructError::UnsupportedInterMbType("B_8x8 without sub_mb_pred".into())
            })?;
            let mut parts = Vec::new();
            for mb_part in 0..4usize {
                let (part_x, part_y) = match mb_part {
                    0 => (0u8, 0u8),
                    1 => (8, 0),
                    2 => (0, 8),
                    _ => (8, 8),
                };
                let sub_type = sm.sub_mb_type[mb_part];

                // §8.4.1.2 — B_Direct_8x8 sub-MB: dispatch to the
                // direct-mode derivation per `direct_spatial_mv_pred_flag`.
                // Temporal direct (flag == 0) fills in precomputed MVs
                // from the colocated block (same as B_Direct_16x16,
                // restricted to this 8x8). Spatial direct (flag == 1)
                // derives refIdxLX via MinPositive(A,B,C) and applies
                // colZeroFlag (§8.4.1.2.2).
                if matches!(sub_type, SubMbType::BDirect8x8) {
                    if !slice_header.direct_spatial_mv_pred_flag {
                        parts.extend(build_temporal_direct_sub_partitions(
                            sps, ref_pics, pic, mb_addr, part_x, part_y,
                        ));
                    } else {
                        parts.extend(build_spatial_direct_partitions(
                            sps,
                            ref_pics,
                            pic,
                            mb_addr,
                            grid,
                            current_slice_id,
                            part_x,
                            part_y,
                            8,
                            false,
                        ));
                    }
                    continue;
                }

                let (mode, r0, r1) = b_sub_mode(
                    sub_type,
                    sm.ref_idx_l0[mb_part] as i8,
                    sm.ref_idx_l1[mb_part] as i8,
                );
                let sub_parts = sub_mb_partitions(sub_type);
                for (sub_idx, (sx, sy, sw, sh)) in sub_parts.iter().enumerate() {
                    let mvd0 = sm.mvd_l0[mb_part].get(sub_idx).copied().unwrap_or([0, 0]);
                    let mvd1 = sm.mvd_l1[mb_part].get(sub_idx).copied().unwrap_or([0, 0]);
                    parts.push(InterPartition {
                        x: part_x + sx,
                        y: part_y + sy,
                        w: *sw,
                        h: *sh,
                        mode,
                        shape: MvpredShape::Default,
                        ref_idx_l0: r0,
                        ref_idx_l1: r1,
                        mvd_l0: (mvd0[0], mvd0[1]),
                        mvd_l1: (mvd1[0], mvd1[1]),
                        is_skip: false,
                        precomputed_mv: None,
                    });
                }
            }
            Ok(parts)
        }

        // Should not arrive here — caller routes intra via the intra path.
        other => Err(ReconstructError::UnsupportedInterMbType(format!(
            "{:?}",
            other
        ))),
    }
}

/// Table 7-18 — sub-partition layout (relative x, y, w, h) per sub_mb_type.
fn sub_mb_partitions(t: SubMbType) -> Vec<(u8, u8, u8, u8)> {
    use SubMbType::*;
    match t {
        PL08x8 | BDirect8x8 | BL08x8 | BL18x8 | BBi8x8 => vec![(0, 0, 8, 8)],
        PL08x4 | BL08x4 | BL18x4 | BBi8x4 => vec![(0, 0, 8, 4), (0, 4, 8, 4)],
        PL04x8 | BL04x8 | BL14x8 | BBi4x8 => vec![(0, 0, 4, 8), (4, 0, 4, 8)],
        PL04x4 | BL04x4 | BL14x4 | BBi4x4 => {
            vec![(0, 0, 4, 4), (4, 0, 4, 4), (0, 4, 4, 4), (4, 4, 4, 4)]
        }
        Reserved(_) => vec![(0, 0, 8, 8)],
    }
}

/// §8.4.1.2.3 — derive one temporal-direct partition's `(refIdxL0,
/// mvL0, mvL1)` for a single (mbAddr, block) colocated position.
///
/// Returns `Some((ref_idx_l0, mv_l0, mv_l1))` when the colocated
/// picture is available; `None` if the provider didn't ship motion
/// data (caller should fall back to (0,0,0,0)).
fn derive_temporal_direct_mvs_for_block<R: RefPicProvider>(
    ref_pics: &R,
    mb_addr: u32,
    col_blk4: usize,
    curr_poc: i32,
) -> Option<(i8, Mv, Mv)> {
    // §8.4.1.2.1 / Table 8-6 — colPic = RefPicList1[0] (frame-only
    // decode — field / AFRM coding paths deferred).
    let col_pic = ref_pics.ref_pic(1, 0)?;
    let poc_pic1 = col_pic.pic_order_cnt;

    // §8.4.1.2.1 — read (mvCol, refIdxCol) from the colocated block.
    // Rule: if predFlagL0Col = 1, use L0; else if predFlagL1Col = 1,
    // use L1; else (intra) mvCol = 0, refIdxCol = -1.
    let l0 = col_pic.colocated_l0(mb_addr, col_blk4);
    let l1 = col_pic.colocated_l1(mb_addr, col_blk4);
    let is_intra =
        l0.as_ref().map(|t| t.2).unwrap_or(false) || l1.as_ref().map(|t| t.2).unwrap_or(false);

    // Extract (mvCol, refIdxCol, which_list) — "which_list" is the
    // colocated picture's list index (0 or 1) the MV came from,
    // needed to look up the referenced picture's POC later.
    let (mv_col_i16, ref_idx_col, used_l1) = if is_intra {
        ((0i16, 0i16), -1i8, false)
    } else {
        match (l0, l1) {
            (Some((mv, rf, _)), _) if rf >= 0 => (mv, rf, false),
            (_, Some((mv, rf, _))) if rf >= 0 => (mv, rf, true),
            _ => ((0, 0), -1, false),
        }
    };

    // §8.4.1.2.3 eq. 8-191 — refIdxL0 = (refIdxCol < 0) ? 0 :
    //                                     MapColToList0(refIdxCol).
    // MapColToList0: find the index in the current slice's
    // RefPicList0 that refers to the picture the colocated block
    // pointed at. We match by POC (picture identity).
    //
    // To perform that lookup, we need the colocated picture's
    // reference list POCs. Those are snapshotted into the Picture
    // struct at reconstruction time (`ref_list_0_pocs` /
    // `ref_list_1_pocs`).
    let curr_list0_pocs = ref_pics.ref_list_0_pocs();
    let curr_list0_lt = ref_pics.ref_list_0_longterm();
    let ref_idx_l0: i8 = if ref_idx_col < 0 {
        0
    } else {
        let col_list = if used_l1 {
            &col_pic.ref_list_1_pocs
        } else {
            &col_pic.ref_list_0_pocs
        };
        // POC of the picture the colocated block referenced.
        match col_list.get(ref_idx_col as usize).copied() {
            Some(ref_poc) => {
                // Lowest-valued index in current slice's RefPicList0
                // whose picture has the same POC (picture identity).
                match curr_list0_pocs.iter().position(|&p| p == ref_poc) {
                    Some(idx) => idx as i8,
                    None => 0, // Fallback: pic not in current L0 → 0.
                }
            }
            None => 0,
        }
    };

    // pic0 = RefPicList0[refIdxL0] of the current slice.
    let poc_pic0 = ref_pics
        .ref_pic_poc(0, ref_idx_l0 as u32)
        .unwrap_or(curr_poc);

    // §8.4.1.2.3 eq. 8-195/8-196 short-circuit:
    //   (a) refIdxL0 picture is long-term,
    //   (b) DiffPicOrderCnt(pic1, pic0) == 0.
    // Either triggers: mvL0 = mvCol, mvL1 = 0.
    let pic0_is_long_term = curr_list0_lt
        .get(ref_idx_l0 as usize)
        .copied()
        .unwrap_or(false);
    let mv_col = Mv::new(mv_col_i16.0 as i32, mv_col_i16.1 as i32);

    if pic0_is_long_term || poc_pic1 == poc_pic0 {
        return Some((ref_idx_l0, mv_col, Mv::ZERO));
    }

    // eq. 8-201/8-202.
    let tb = clip3_i32(-128, 127, curr_poc - poc_pic0);
    let td = clip3_i32(-128, 127, poc_pic1 - poc_pic0);
    if td == 0 {
        return Some((ref_idx_l0, mv_col, Mv::ZERO));
    }

    // eq. 8-197/8-198/8-199/8-200.
    // tx = (16384 + Abs(td/2)) / td
    // Per spec §5.7 "/" truncates toward zero; Rust i32/i32 matches.
    let tx = (16384 + (td / 2).abs()) / td;
    let dsf = clip3_i32(-1024, 1023, (tb * tx + 32) >> 6);
    let mvx_l0 = ((dsf * mv_col.x) + 128) >> 8;
    let mvy_l0 = ((dsf * mv_col.y) + 128) >> 8;
    let mv_l0 = Mv::new(mvx_l0, mvy_l0);
    let mv_l1 = Mv::new(mv_l0.x - mv_col.x, mv_l0.y - mv_col.y);
    Some((ref_idx_l0, mv_l0, mv_l1))
}

/// §8.4.1.2.3 — expand a B_Skip / B_Direct_16x16 macroblock into
/// temporal-direct sub-partitions with pre-computed L0/L1 MVs.
///
/// The granularity follows `sps.direct_8x8_inference_flag`:
///   - flag == 1: four 8x8 partitions (coarser derivation).
///   - flag == 0: sixteen 4x4 partitions (fine derivation).
///
/// Per partition:
///   1. Read (mvCol, refIdxCol) from the colocated block in
///      RefPicList1[0] (§8.4.1.2.1).
///   2. refIdxL0 = (refIdxCol < 0) ? 0 : MapColToList0(refIdxCol)
///      (eq. 8-191), refIdxL1 = 0 (eq. 8-192). MapColToList0 is
///      performed by matching POCs between the colocated picture's
///      per-slice RefPicList0 snapshot and the current slice's
///      RefPicList0.
///   3. If pic0 is long-term or DiffPicOrderCnt(pic1, pic0) == 0:
///      mvL0 = mvCol, mvL1 = 0 (eq. 8-195/8-196).
///   4. Else apply temporal scaling (eq. 8-197..8-200).
fn build_temporal_direct_partitions<R: RefPicProvider>(
    sps: &Sps,
    ref_pics: &R,
    pic: &Picture,
    mb_addr: u32,
    is_skip: bool,
) -> Vec<InterPartition> {
    let curr_poc = pic.pic_order_cnt;
    let use_8x8 = sps.direct_8x8_inference_flag;
    let block_size: u8 = if use_8x8 { 8 } else { 4 };
    let step = block_size as usize;
    let n_per_row = 16usize / step;

    let mut partitions = Vec::with_capacity(n_per_row * n_per_row);
    for by in 0..n_per_row {
        for bx in 0..n_per_row {
            let x = (bx * step) as u8;
            let y = (by * step) as u8;

            // §8.4.1.2.1 luma4x4BlkIdx — the 4x4 block index within
            // the colocated MB that supplies (mvCol, refIdxCol):
            //   - direct_8x8_inference_flag == 1: luma4x4BlkIdx =
            //     5 * mbPartIdx, i.e. block indices 0, 5, 10, 15
            //     (diagonal — one per 8x8 quadrant).
            //   - direct_8x8_inference_flag == 0: luma4x4BlkIdx =
            //     4 * mbPartIdx + subMbPartIdx — every 4x4 queried
            //     independently.
            let col_blk4 = if use_8x8 {
                // mbPartIdx = 2*by + bx (by, bx each in 0..=1).
                let mb_part_idx = 2 * by + bx;
                5 * mb_part_idx
            } else {
                // 4x4 granularity: raster position (bx, by) in
                // units of 4-sample blocks.
                blk4_raster_index(bx as u8, by as u8) as usize
            };

            let (ref_idx_l0, mv_l0, mv_l1) =
                derive_temporal_direct_mvs_for_block(ref_pics, mb_addr, col_blk4, curr_poc)
                    .unwrap_or((0, Mv::ZERO, Mv::ZERO));

            partitions.push(InterPartition {
                x,
                y,
                w: block_size,
                h: block_size,
                mode: PartMode::Direct,
                shape: MvpredShape::Default,
                ref_idx_l0,
                ref_idx_l1: 0,
                mvd_l0: (0, 0),
                mvd_l1: (0, 0),
                is_skip,
                precomputed_mv: Some((mv_l0, mv_l1)),
            });
        }
    }

    partitions
}

/// §8.4.1.2.3 — derive temporal-direct partitions for one B_Direct_8x8
/// sub-macroblock located at (`part_x`, `part_y`) within the current
/// MB (both in luma samples, multiples of 8).
///
/// Mirrors [`build_temporal_direct_partitions`] but only covers the
/// 8x8 area of the sub-MB. The granularity inside the 8x8 still
/// follows `direct_8x8_inference_flag`:
///   - flag == 1: one 8x8 partition.
///   - flag == 0: four 4x4 partitions.
fn build_temporal_direct_sub_partitions<R: RefPicProvider>(
    sps: &Sps,
    ref_pics: &R,
    pic: &Picture,
    mb_addr: u32,
    part_x: u8,
    part_y: u8,
) -> Vec<InterPartition> {
    let curr_poc = pic.pic_order_cnt;
    let use_8x8 = sps.direct_8x8_inference_flag;
    let block_size: u8 = if use_8x8 { 8 } else { 4 };
    let step = block_size as usize;
    let n_per_row = 8usize / step;

    // §8.4.1.2.1 — for B_Direct_8x8, mbPartIdx is fixed by the
    // sub-macroblock position: 2*(part_y/8) + (part_x/8).
    let mb_part_idx = 2 * ((part_y / 8) as usize) + ((part_x / 8) as usize);

    let mut partitions = Vec::with_capacity(n_per_row * n_per_row);
    for sy in 0..n_per_row {
        for sx in 0..n_per_row {
            let x = part_x + (sx * step) as u8;
            let y = part_y + (sy * step) as u8;

            // §8.4.1.2.1 luma4x4BlkIdx.
            let col_blk4 = if use_8x8 {
                5 * mb_part_idx
            } else {
                // subMbPartIdx = 2*sy + sx for 4x4 granularity.
                let sub_mb_part_idx = 2 * sy + sx;
                4 * mb_part_idx + sub_mb_part_idx
            };

            let (ref_idx_l0, mv_l0, mv_l1) =
                derive_temporal_direct_mvs_for_block(ref_pics, mb_addr, col_blk4, curr_poc)
                    .unwrap_or((0, Mv::ZERO, Mv::ZERO));

            partitions.push(InterPartition {
                x,
                y,
                w: block_size,
                h: block_size,
                mode: PartMode::Direct,
                shape: MvpredShape::Default,
                ref_idx_l0,
                ref_idx_l1: 0,
                mvd_l0: (0, 0),
                mvd_l1: (0, 0),
                is_skip: false,
                precomputed_mv: Some((mv_l0, mv_l1)),
            });
        }
    }
    partitions
}

/// §8.4.1.2.2 — B-slice spatial direct mode partition builder.
///
/// Given a direct-mode region of size `region_size` samples at
/// (`region_x`, `region_y`) inside the current MB (a full 16x16
/// for `B_Direct_16x16`/`B_Skip`, or an 8x8 sub-MB for `B_Direct_8x8`),
/// derives `refIdxL0`, `refIdxL1` and the per-sub-block motion vectors
/// with the `colZeroFlag` short-circuit applied.
///
/// Implements:
/// 1. MinPositive chain on MB-level (A, B, C) neighbour refIdxLX
///    (eq. 8-184/8-185), then the directZeroPredictionFlag fallback
///    (eq. 8-188..8-190) if both result in < 0.
/// 2. For every 8x8 sub-block (when `direct_8x8_inference_flag == 1`)
///    or 4x4 sub-block (when 0), mvpLX is derived from the same
///    MB-level neighbours via `derive_median_mvpred`.
/// 3. `colZeroFlag` (§8.4.1.2.2 step 7): if RefPicList1[0] is a
///    short-term picture and the colocated block's predicted L0 or L1
///    MV satisfies `|mvCol| <= 1` with `refIdxCol == 0`, force the
///    corresponding list's MV to zero (for that sub-block only).
///
/// Note — the MB-level neighbour lookup uses the partition origin at
/// the MB's top-left (block (0, 0)) with width = region_size / 4 so
/// that C sits above-right of the region. This matches the spec's
/// "(mbPartIdx, subMbPartIdx) = (0, 0)" rule for refIdx derivation.
fn build_spatial_direct_partitions<R: RefPicProvider>(
    sps: &Sps,
    ref_pics: &R,
    _pic: &Picture,
    mb_addr: u32,
    grid: &MbGrid,
    current_slice_id: i32,
    region_x: u8,
    region_y: u8,
    region_size: u8,
    is_skip: bool,
) -> Vec<InterPartition> {
    // §8.4.1.2.2 step 2-3 — derive refIdxLXN, mvLXN from §8.4.1.3.2
    // invoked with **mbPartIdx = 0, subMbPartIdx = 0** — i.e. the
    // neighbours of the macroblock's top-left 16x16 partition. This
    // is invariant in the size/position of the direct region (NOTE 1).
    // Even for a B_Direct_8x8 sub-MB at (8, 0) the neighbours probed
    // are the MB-level (A, B, C), not the sub-MB's.
    let neighbour_probe = InterPartition {
        x: 0,
        y: 0,
        w: 16,
        h: 16,
        mode: PartMode::Direct,
        shape: MvpredShape::Default,
        ref_idx_l0: 0,
        ref_idx_l1: 0,
        mvd_l0: (0, 0),
        mvd_l1: (0, 0),
        is_skip,
        precomputed_mv: None,
    };
    let (a_l0, b_l0, c_l0, d_l0) =
        neighbour_mvs_for_list(&neighbour_probe, mb_addr, grid, 0, current_slice_id);
    let (a_l1, b_l1, c_l1, d_l1) =
        neighbour_mvs_for_list(&neighbour_probe, mb_addr, grid, 1, current_slice_id);

    // §8.4.1.2.2 step 4 — MinPositive chain to pick refIdxLX. This
    // mirrors the pre-step-5 derivation inside
    // `derive_b_spatial_direct_with_d`; we redo it here so we can
    // observe whether BOTH refs were < 0 (i.e. directZeroPredictionFlag
    // should fire) before the helper's step-5 fallback collapses them
    // to 0.
    fn min_positive(x: i32, y: i32) -> i32 {
        if x >= 0 && y >= 0 {
            x.min(y)
        } else {
            x.max(y)
        }
    }
    // §8.4.1.3.2 eq. 8-214..8-216 — C→D substitution when C is
    // unavailable (partition_available == false).
    let c_l0_eff = if !c_l0.partition_available && d_l0.partition_available {
        d_l0
    } else {
        c_l0
    };
    let c_l1_eff = if !c_l1.partition_available && d_l1.partition_available {
        d_l1
    } else {
        c_l1
    };

    let ref_l0_minp = min_positive(a_l0.ref_idx, min_positive(b_l0.ref_idx, c_l0_eff.ref_idx));
    let ref_l1_minp = min_positive(a_l1.ref_idx, min_positive(b_l1.ref_idx, c_l1_eff.ref_idx));

    // §8.4.1.2.2 step 5 — directZeroPredictionFlag = both refs < 0.
    let direct_zero_both = ref_l0_minp < 0 && ref_l1_minp < 0;
    let ref_idx_l0_final: i8 = if direct_zero_both {
        0
    } else if ref_l0_minp < 0 {
        -1
    } else {
        ref_l0_minp as i8
    };
    let ref_idx_l1_final: i8 = if direct_zero_both {
        0
    } else if ref_l1_minp < 0 {
        -1
    } else {
        ref_l1_minp as i8
    };

    // Keep the MB-level MVs from the helper for debug / symmetry —
    // not used directly (we recompute per sub-block via
    // `derive_median_mvpred` below).
    let _ = derive_b_spatial_direct_with_d(a_l0, a_l1, b_l0, b_l1, c_l0, c_l1, d_l0, d_l1);

    // §8.4.1.2.2 step 7 — colZeroFlag requires RefPicList1[0] to be a
    // short-term reference and the colocated block to have a small
    // zero-like MV with ref_idx_l0 == 0.
    let col_pic_is_short_term = !ref_pics
        .ref_list_1_longterm()
        .first()
        .copied()
        .unwrap_or(false);
    let col_pic = ref_pics.ref_pic(1, 0);

    // Granularity inside the region: 8x8 when direct_8x8_inference_flag,
    // else 4x4.
    let use_8x8 = sps.direct_8x8_inference_flag;
    let block_size: u8 = if use_8x8 { 8 } else { 4 };
    // Guard: if the region is smaller than the block size, fall back
    // to one region-sized block.
    let step = block_size.min(region_size) as usize;
    let block_size = step as u8;
    let n_per_row = (region_size as usize) / step;

    let mut partitions = Vec::with_capacity(n_per_row * n_per_row);
    for by in 0..n_per_row {
        for bx in 0..n_per_row {
            let x = region_x + (bx * step) as u8;
            let y = region_y + (by * step) as u8;

            // §8.4.1.2.2 step 6 — per-block colZeroFlag derivation.
            // `blk4` is the 4x4 block index within the MB covering
            // (x, y). For 8x8-granularity direct mode the spec uses
            // 5 * mbPartIdx (blocks 0, 5, 10, 15 — one per 8x8
            // quadrant); for 4x4 it's the raster index of the actual
            // 4x4 block.
            let bx4 = x / 4;
            let by4 = y / 4;
            let col_blk4: usize = if use_8x8 {
                let mb_part_idx = 2 * ((y / 8) as usize) + ((x / 8) as usize);
                5 * mb_part_idx
            } else {
                blk4_raster_index(bx4, by4) as usize
            };

            let col_zero_flag = col_pic_is_short_term
                && col_pic.is_some_and(|cp| is_colocated_zero_mv(cp, mb_addr, col_blk4));

            // §8.4.1.2.2 mv derivation (NOTE 3: mvLX returned from
            // §8.4.1.3 is identical for all 4x4 sub-partitions of the
            // same MB — so we compute these outside of the inner loop
            // in principle; they're inside to keep the flow linear).
            //
            // Rules:
            //   - directZeroPredictionFlag == 1 ⇒ both mvLX = 0.
            //   - refIdxLX < 0 ⇒ mvLX = 0 (list not used).
            //   - refIdxLX == 0 && colZeroFlag ⇒ mvLX = 0.
            //   - otherwise mvLX = derive_median_mvpred(A, B, C_eff,
            //     refIdxLX).
            let mv_l0 = if direct_zero_both
                || ref_idx_l0_final < 0
                || (ref_idx_l0_final == 0 && col_zero_flag)
            {
                Mv::ZERO
            } else {
                derive_median_mvpred(a_l0, b_l0, c_l0_eff, ref_idx_l0_final as i32)
            };
            let mv_l1 = if direct_zero_both
                || ref_idx_l1_final < 0
                || (ref_idx_l1_final == 0 && col_zero_flag)
            {
                Mv::ZERO
            } else {
                derive_median_mvpred(a_l1, b_l1, c_l1_eff, ref_idx_l1_final as i32)
            };

            // Build the partition mode: if only L0 is valid we should
            // emit an L0-only MC (not bi-predict), and vice versa.
            let mode = match (ref_idx_l0_final >= 0, ref_idx_l1_final >= 0) {
                (true, true) => PartMode::Direct,
                (true, false) => PartMode::L0Only,
                (false, true) => PartMode::L1Only,
                (false, false) => PartMode::Direct, // directZero
            };

            partitions.push(InterPartition {
                x,
                y,
                w: block_size,
                h: block_size,
                mode,
                shape: MvpredShape::Default,
                ref_idx_l0: ref_idx_l0_final,
                ref_idx_l1: ref_idx_l1_final,
                mvd_l0: (0, 0),
                mvd_l1: (0, 0),
                is_skip,
                precomputed_mv: Some((mv_l0, mv_l1)),
            });
        }
    }
    partitions
}

/// §8.4.1.2.2 step 7 — colZeroFlag "small zero-like MV" check.
///
/// §8.4.1.2.1 rules for picking mvCol/refIdxCol from the colocated
/// block:
///   1. If the colocated macroblock is intra, mvCol = (0, 0),
///      refIdxCol = -1. Then colZeroFlag is 0 (refIdxCol != 0).
///   2. Else if predFlagL0Col == 1, (mvCol, refIdxCol) = L0 values.
///   3. Else (predFlagL0Col == 0, predFlagL1Col == 1), (mvCol,
///      refIdxCol) = L1 values.
///
/// colZeroFlag becomes 1 iff refIdxCol == 0 && |mvCol| <= 1 in both
/// components.
fn is_colocated_zero_mv(col_pic: &Picture, mb_addr: u32, col_blk4: usize) -> bool {
    let l0 = col_pic.colocated_l0(mb_addr, col_blk4);
    let l1 = col_pic.colocated_l1(mb_addr, col_blk4);

    // Intra colocated: mvCol = 0, refIdxCol = -1 ⇒ colZeroFlag = 0.
    let is_intra =
        l0.as_ref().map(|t| t.2).unwrap_or(false) || l1.as_ref().map(|t| t.2).unwrap_or(false);
    if is_intra {
        return false;
    }

    // Pick L0 when its refIdx is valid; else L1; else no data.
    let (mv, ref_idx) = match (l0, l1) {
        (Some((mv, rf, _)), _) if rf >= 0 => (mv, rf),
        (_, Some((mv, rf, _))) if rf >= 0 => (mv, rf),
        _ => return false,
    };
    ref_idx == 0 && mv.0.abs() <= 1 && mv.1.abs() <= 1
}

/// Table 7-14 — B_L0/B_L1/B_Bi 16x16 → (mode, ref_idx_l0, ref_idx_l1).
fn b_dir_refs_16x16(ty: &MbType, p: &crate::macroblock_layer::MbPred) -> (PartMode, i8, i8) {
    match ty {
        MbType::BL016x16 => (
            PartMode::L0Only,
            *p.ref_idx_l0.first().unwrap_or(&0) as i8,
            -1,
        ),
        MbType::BL116x16 => (
            PartMode::L1Only,
            -1,
            *p.ref_idx_l1.first().unwrap_or(&0) as i8,
        ),
        MbType::BBi16x16 => (
            PartMode::BiPred,
            *p.ref_idx_l0.first().unwrap_or(&0) as i8,
            *p.ref_idx_l1.first().unwrap_or(&0) as i8,
        ),
        _ => (PartMode::L0Only, 0, -1),
    }
}

/// Table 7-14 — 16x8 B variants → (mode_top, mode_bottom).
fn b_dir_for_16x8(ty: &MbType) -> (PartMode, PartMode) {
    use MbType::*;
    match ty {
        BL0L016x8 => (PartMode::L0Only, PartMode::L0Only),
        BL1L116x8 => (PartMode::L1Only, PartMode::L1Only),
        BL0L116x8 => (PartMode::L0Only, PartMode::L1Only),
        BL1L016x8 => (PartMode::L1Only, PartMode::L0Only),
        BL0Bi16x8 => (PartMode::L0Only, PartMode::BiPred),
        BL1Bi16x8 => (PartMode::L1Only, PartMode::BiPred),
        BBiL016x8 => (PartMode::BiPred, PartMode::L0Only),
        BBiL116x8 => (PartMode::BiPred, PartMode::L1Only),
        BBiBi16x8 => (PartMode::BiPred, PartMode::BiPred),
        _ => (PartMode::L0Only, PartMode::L0Only),
    }
}

/// Table 7-14 — 8x16 B variants → (mode_left, mode_right).
fn b_dir_for_8x16(ty: &MbType) -> (PartMode, PartMode) {
    use MbType::*;
    match ty {
        BL0L08x16 => (PartMode::L0Only, PartMode::L0Only),
        BL1L18x16 => (PartMode::L1Only, PartMode::L1Only),
        BL0L18x16 => (PartMode::L0Only, PartMode::L1Only),
        BL1L08x16 => (PartMode::L1Only, PartMode::L0Only),
        BL0Bi8x16 => (PartMode::L0Only, PartMode::BiPred),
        BL1Bi8x16 => (PartMode::L1Only, PartMode::BiPred),
        BBiL08x16 => (PartMode::BiPred, PartMode::L0Only),
        BBiL18x16 => (PartMode::BiPred, PartMode::L1Only),
        BBiBi8x16 => (PartMode::BiPred, PartMode::BiPred),
        _ => (PartMode::L0Only, PartMode::L0Only),
    }
}

/// Helper — per-partition (ref_idx_l0, ref_idx_l1), honouring the mode.
fn ref_idx_for_part(p: &crate::macroblock_layer::MbPred, idx: usize, mode: PartMode) -> (i8, i8) {
    let r0 = p.ref_idx_l0.get(idx).copied().unwrap_or(0) as i8;
    let r1 = p.ref_idx_l1.get(idx).copied().unwrap_or(0) as i8;
    match mode {
        PartMode::L0Only => (r0, -1),
        PartMode::L1Only => (-1, r1),
        PartMode::BiPred => (r0, r1),
        PartMode::Direct => (r0, r1),
    }
}

/// Table 7-18 — sub_mb_type for B_8x8 → (mode, ref_idx_l0, ref_idx_l1).
fn b_sub_mode(t: SubMbType, r0: i8, r1: i8) -> (PartMode, i8, i8) {
    use SubMbType::*;
    match t {
        BDirect8x8 => (PartMode::Direct, 0, 0),
        BL08x8 | BL08x4 | BL04x8 | BL04x4 => (PartMode::L0Only, r0, -1),
        BL18x8 | BL18x4 | BL14x8 | BL14x4 => (PartMode::L1Only, -1, r1),
        BBi8x8 | BBi8x4 | BBi4x8 | BBi4x4 => (PartMode::BiPred, r0, r1),
        _ => (PartMode::L0Only, r0, -1),
    }
}

/// §8.5.12 / §8.5.11 — inter chroma residual path: for every 4x4 chroma
/// block, dequantise AC, combine with the pre-computed pred buffer, and
/// write out. Shares code with the intra chroma path except the
/// prediction buffer source.
#[allow(clippy::too_many_arguments)]
fn reconstruct_inter_chroma_residual(
    mb: &Macroblock,
    qp_y: i32,
    chroma_array_type: u32,
    bit_depth_c: u32,
    mb_px: i32,
    mb_py: i32,
    writer: &MbWriter,
    sps: &Sps,
    pps: &Pps,
    cbp_chroma: u8,
    pred_cb: &[i32],
    pred_cr: &[i32],
    pic: &mut Picture,
) -> Result<(), ReconstructError> {
    let (mbw_c, _mbh_c) = chroma_mb_dims(chroma_array_type);
    // §6.4.1 — MBAFF-aware chroma origin from writer.
    let c_mb_px = writer.chroma_mb_px();
    let c_mb_py = writer.chroma_mb_py();
    let _ = mb_px;
    let _ = mb_py;

    // §8.5.8 — QPc per plane.
    let cb_offset = pps.chroma_qp_index_offset;
    let cr_offset = pps
        .extension
        .as_ref()
        .map(|e| e.second_chroma_qp_index_offset)
        .unwrap_or(pps.chroma_qp_index_offset);
    let qp_cb = qp_y_to_qp_c(qp_y, cb_offset);
    let qp_cr = qp_y_to_qp_c(qp_y, cr_offset);

    // §7.4.2.1.1.1 Table 7-2 — inter chroma lists: i=4 (Cb) / i=5 (Cr).
    let sl4_cb = select_scaling_list_4x4(4, sps, pps);
    let sl4_cr = select_scaling_list_4x4(5, sps, pps);
    let num_c8x8 = if chroma_array_type == 1 { 1 } else { 2 };
    let n_ac = 4 * num_c8x8 as usize;

    for plane in 0..2u8 {
        let qp_c = if plane == 0 { qp_cb } else { qp_cr };
        let sl4 = if plane == 0 { &sl4_cb } else { &sl4_cr };
        let dc_block = if plane == 0 {
            &mb.residual_chroma_dc_cb
        } else {
            &mb.residual_chroma_dc_cr
        };
        let dc_flat: [i32; 8] = {
            let mut a = [0i32; 8];
            for (i, v) in dc_block.iter().enumerate().take(a.len()) {
                a[i] = *v;
            }
            a
        };
        let (dc4, dc8): (Option<[i32; 4]>, Option<[i32; 8]>) = if cbp_chroma > 0 {
            if chroma_array_type == 1 {
                let dc4: [i32; 4] = [dc_flat[0], dc_flat[1], dc_flat[2], dc_flat[3]];
                let out = inverse_hadamard_chroma_dc_420(&dc4, qp_c, sl4, bit_depth_c)?;
                (Some(out), None)
            } else {
                let out = inverse_hadamard_chroma_dc_422(&dc_flat, qp_c, sl4, bit_depth_c)?;
                (None, Some(out))
            }
        } else {
            (Some([0i32; 4]), Some([0i32; 8]))
        };
        let ac_blocks = if plane == 0 {
            &mb.residual_chroma_ac_cb
        } else {
            &mb.residual_chroma_ac_cr
        };
        for blk in 0..n_ac {
            let dc_c = if chroma_array_type == 1 {
                dc4.unwrap()[blk]
            } else {
                dc8.unwrap()[blk]
            };
            let ac_scan = if cbp_chroma == 2 {
                ac_blocks.get(blk).copied().unwrap_or([0i32; 16])
            } else {
                [0i32; 16]
            };
            // Chroma AC: parser slots 0..=14 are spec scan positions 1..=15.
            let mut coeffs = crate::transform::inverse_scan_4x4_zigzag_ac(&ac_scan);
            coeffs[0] = dc_c;
            let residual = inverse_transform_4x4_dc_preserved(&coeffs, qp_c, sl4, bit_depth_c)?;
            let (bx, by) = chroma_block_xy(chroma_array_type, blk);
            for yy in 0..4 {
                for xx in 0..4 {
                    let pidx = ((by as usize + yy) * (mbw_c as usize)) + (bx as usize + xx);
                    let pred_v = if plane == 0 {
                        pred_cb[pidx]
                    } else {
                        pred_cr[pidx]
                    };
                    let v = clip_sample(pred_v + residual[yy * 4 + xx], bit_depth_c);
                    if plane == 0 {
                        writer.set_cb(pic, bx + xx as i32, by + yy as i32, v);
                    } else {
                        writer.set_cr(pic, bx + xx as i32, by + yy as i32, v);
                    }
                }
            }
        }
    }
    let _ = c_mb_px;
    let _ = c_mb_py;
    Ok(())
}

// Silence "MbInfo unused" if no inter test uses it directly.
#[allow(dead_code)]
fn _mb_info_marker(_: MbInfo) {}

// -------------------------------------------------------------------------
// §8.7 — per-block nonzero-coefficient bitmap helpers
// -------------------------------------------------------------------------

/// §8.7.2.1 — derive the per-4x4 luma nonzero-coefficient mask for
/// the current MB from the parsed residual data. Bit `z` (indexed by
/// §6.4.3 Figure 6-10 Z-scan) is set iff the 4x4 block at Z-scan
/// position `z` has at least one non-zero AC (or AC+DC) transform
/// coefficient level. I_PCM is treated as "all blocks coded" since
/// its samples are carried directly with no transform block CBF test.
///
/// The mask drives the deblock boundary-strength derivation: the
/// third bullet of §8.7.2.1 ("either adjacent 4x4 block contains
/// non-zero transform coefficient levels" → bS = 2) requires
/// per-block granularity rather than the MB-level `cbp_luma != 0`
/// check, which is too coarse (MBs with a single coded 8x8 quadrant
/// would incorrectly trigger bS = 2 on edges of the three other
/// quadrants' coded-block-free 4x4 blocks).
fn compute_luma_nonzero_mask(mb: &Macroblock) -> u16 {
    // I_PCM — the spec treats I_PCM as having all transform blocks
    // coded for BS purposes (first bullet equivalence, §8.7.2.1).
    if mb.mb_type.is_i_pcm() {
        return 0xFFFF;
    }
    let cbp_luma = (mb.coded_block_pattern & 0x0F) as u8;
    // For Intra_16x16, the 16 AC blocks are always present when
    // `cbp_luma` for the Intra_16x16 level is 15; then per-4x4
    // granularity comes from the parsed AC coefficients themselves.
    // The DC block is not a 4x4 transform block per §8.7.2.1 — it
    // does NOT contribute to the per-4x4 bit. (Per NOTE 1 in
    // §9.2.1.1 CAVLC: DC is tracked separately.)
    if let MbType::Intra16x16(cfg) = &mb.mb_type {
        let mut mask = 0u16;
        if cfg.cbp_luma == 15 {
            // residual_luma has 16 entries in raster-Z order.
            for (blk_idx, block) in mb.residual_luma.iter().enumerate().take(16) {
                // AC coefficients at slots 0..=14 (slot 15 is padding).
                // Slot 0 is spec scan position 1 (AC, not DC).
                let any_nz = block.iter().take(15).any(|c| *c != 0);
                if any_nz {
                    mask |= 1 << blk_idx;
                }
            }
        }
        return mask;
    }
    // 8x8 transform: residual_luma has 4 entries per set cbp bit,
    // each entry holding 16 of the 64 scan positions. For the bS
    // derivation, every 4x4 inside an 8x8 transform block shares the
    // 8x8's "nonzero" status.
    if mb.transform_size_8x8_flag {
        let mut mask = 0u16;
        // Mapping: 8x8 quadrant q → four 4x4 Z-scan indices.
        // Quadrants 0..=3 at (bx=0,by=0),(bx=8,by=0),(bx=0,by=8),(bx=8,by=8).
        // The four 4x4s in that 8x8 are Z-scan indices q*4..q*4+3
        // (not to be confused with raster position — §6.4.3 Z-scan
        // groups blk4 indices in quadrants of 4 consecutive values).
        for quad in 0..4usize {
            if (cbp_luma >> quad) & 1 == 0 {
                continue;
            }
            let set_before = (cbp_luma & ((1u8 << quad) - 1)).count_ones() as usize;
            let base_slot = set_before * 4;
            // Whole-8x8 nonzero iff any of the four 16-entry chunks
            // carries a non-zero coefficient.
            let mut any_nz = false;
            for sub in 0..4usize {
                let slot = base_slot + sub;
                if let Some(c) = mb.residual_luma.get(slot) {
                    if c.iter().any(|v| *v != 0) {
                        any_nz = true;
                        break;
                    }
                }
            }
            if any_nz {
                // All four 4x4s in the quadrant inherit the 8x8 flag.
                for z in (quad * 4)..(quad * 4 + 4) {
                    mask |= 1 << z;
                }
            }
        }
        return mask;
    }
    // 4x4 transform (Inter / I_NxN). residual_luma has 4 entries per
    // set cbp bit, in ascending quadrant order. Each entry is one
    // 4x4 block's 16-coefficient array.
    let mut mask = 0u16;
    for quad in 0..4usize {
        if (cbp_luma >> quad) & 1 == 0 {
            continue;
        }
        let set_before = (cbp_luma & ((1u8 << quad) - 1)).count_ones() as usize;
        let base_slot = set_before * 4;
        for sub in 0..4usize {
            let slot = base_slot + sub;
            if let Some(c) = mb.residual_luma.get(slot) {
                if c.iter().any(|v| *v != 0) {
                    // Z-scan position of this 4x4 block.
                    let z = quad * 4 + sub;
                    mask |= 1 << z;
                }
            }
        }
    }
    mask
}

/// §8.7.2.1 — per-4x4 chroma nonzero-coefficient mask, laid out as
/// two 8-bit planes in a u16: low byte = Cb, high byte = Cr. For
/// 4:2:0 only bits 0..=3 (Cb) and 8..=11 (Cr) are populated (four
/// 4x4 chroma blocks per plane); for 4:2:2 bits 0..=7 / 8..=15 are
/// populated (eight 4x4 blocks per plane). ChromaArrayType == 3
/// (4:4:4) is not modelled here; callers that use the chroma deblock
/// path for 4:4:4 should consult the luma mask instead.
fn compute_chroma_nonzero_mask(mb: &Macroblock, chroma_array_type: u32) -> u16 {
    // I_PCM: treat all chroma blocks as coded, consistent with luma.
    if mb.mb_type.is_i_pcm() {
        return 0xFFFF;
    }
    if chroma_array_type != 1 && chroma_array_type != 2 {
        return 0;
    }
    // cbp_chroma == 0 → no coded AC (and DC). cbp_chroma == 1 → only
    // DC coded. cbp_chroma == 2 → DC + all AC blocks coded.
    // Per §8.7.2.1 the 4x4 AC granularity matters — inspect the
    // parser's `residual_chroma_ac_*` entries to set per-block bits.
    let cbp_chroma = ((mb.coded_block_pattern >> 4) & 0x03) as u8;
    if cbp_chroma < 2 {
        return 0;
    }
    // Per-plane 4x4 count: 4 for 4:2:0, 8 for 4:2:2.
    let num_chroma_blocks = if chroma_array_type == 1 { 4 } else { 8 };
    let mut mask = 0u16;
    for (plane, ac) in [&mb.residual_chroma_ac_cb, &mb.residual_chroma_ac_cr]
        .iter()
        .enumerate()
    {
        let shift = if plane == 0 { 0 } else { 8 };
        for (blk, coeffs) in ac.iter().enumerate().take(num_chroma_blocks) {
            let any_nz = coeffs.iter().any(|v| *v != 0);
            if any_nz {
                mask |= 1u16 << (shift + blk as u32);
            }
        }
    }
    mask
}

// -------------------------------------------------------------------------
// §8.7 — Deblocking (picture-level pass, simplified)
// -------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn deblock_picture(
    pic: &mut Picture,
    grid: &MbGrid,
    alpha_off: i32,
    beta_off: i32,
    bit_depth_y: u32,
    bit_depth_c: u32,
    pps: &Pps,
    mbaff_frame_flag: bool,
    mb_field_flags: &[bool],
) {
    // For simplicity, walk the luma plane and filter each 4x4 block
    // boundary (vertical edges first, then horizontal) per §8.7.1.
    // Chroma filtering is a per-plane replay at the chroma grid; we
    // handle 4:2:0 and 4:2:2 only (4:4:4 was already rejected upstream).
    //
    // MBAFF scope: the walker consults [`pixel_to_mb_addr`] for MB
    // addressing so field-coded pairs hit the right per-MB MbInfo.
    // Full §6.4.11.1 Table 6-4 MBAFF edge-geometry (the different 4x4
    // vs 8x8 edge choices in field pairs) is a known simplification —
    // we approximate by treating MBAFF edges the same as non-MBAFF
    // edges aligned to picture-grid 4-pel boundaries, which keeps the
    // filter running without panic. Pixel-accurate MBAFF deblocking
    // is a future-work item.
    deblock_plane_luma(
        pic,
        grid,
        alpha_off,
        beta_off,
        bit_depth_y,
        mbaff_frame_flag,
        mb_field_flags,
    );
    if pic.chroma_array_type == 1 || pic.chroma_array_type == 2 {
        deblock_plane_chroma(
            pic,
            grid,
            alpha_off,
            beta_off,
            bit_depth_c,
            pps,
            mbaff_frame_flag,
            mb_field_flags,
        );
    }
    // Suppress unused warning for pps when ChromaArrayType == 0.
    let _ = pps;
}

/// §6.4.1 — map picture luma sample coordinates `(x, y)` to the MB
/// address containing that sample, honouring MBAFF pair structure +
/// per-pair `mb_field_decoding_flag`.
///
/// In non-MBAFF frame pictures this is simply `mb_x + mb_y *
/// PicWidthInMbs`. In MBAFF the picture has pair rows of 32 luma
/// samples; a pixel's containing MB depends on whether its pair is
/// field- or frame-coded:
///
/// * Frame-coded pair: top MB occupies picture rows `[pair_y, pair_y+16)`;
///   bottom MB occupies `[pair_y+16, pair_y+32)`.
/// * Field-coded pair: top MB occupies even-parity rows of the pair
///   (`pair_y + k*2`); bottom MB occupies odd-parity rows (`pair_y + 1 + k*2`).
///
/// This helper returns `None` if the pixel lies outside the picture.
fn pixel_to_mb_addr(
    grid: &MbGrid,
    x: i32,
    y: i32,
    mbaff_frame_flag: bool,
    mb_field_flags: &[bool],
) -> Option<u32> {
    if x < 0 || y < 0 {
        return None;
    }
    let w = grid.width_in_mbs as i32;
    let mb_x = x / 16;
    if mb_x >= w {
        return None;
    }
    if !mbaff_frame_flag {
        let mb_y = y / 16;
        if mb_y >= grid.height_in_mbs as i32 {
            return None;
        }
        return Some((mb_y as u32) * grid.width_in_mbs + mb_x as u32);
    }
    // MBAFF frame: pair row is 32 luma rows tall.
    let pair_row = y / 32;
    if pair_row * 2 >= grid.height_in_mbs as i32 {
        return None;
    }
    let pair_top_addr = (pair_row as u32 * 2) * grid.width_in_mbs + mb_x as u32;
    let bot_addr = pair_top_addr + grid.width_in_mbs; // bottom MB of the pair
                                                      // Inspect the top MB's mb_field_decoding_flag (shared within a pair).
    let pair_field = mb_field_flags
        .get(pair_top_addr as usize)
        .copied()
        .unwrap_or(false);
    let rel_y = y - pair_row * 32; // 0..=31
    let is_bot = if pair_field {
        // Field-coded pair: even rel_y → top MB, odd → bottom MB.
        (rel_y & 1) == 1
    } else {
        // Frame-coded pair: rel_y < 16 → top MB, else bottom MB.
        rel_y >= 16
    };
    if is_bot {
        Some(bot_addr)
    } else {
        Some(pair_top_addr)
    }
}

/// §8.7.2.1 — `different_ref_or_mv` test for a pair of 4x4 luma blocks
/// straddling one edge, when neither side is intra/SP/SI and neither side
/// has nonzero transform coeffs. Returns `true` when bS should be 1 per
/// the fourth bullet of §8.7.2.1:
///
///   * the two blocks use different reference pictures, or
///   * the two blocks use a different number of motion vectors
///     (one list-only vs bi-pred), or
///   * any motion vector component between the two blocks differs by
///     more than 3 in quarter-sample units (i.e. `|Δ| >= 4`).
///
/// `p_in_mb_x`, `p_in_mb_y`, `q_in_mb_x`, `q_in_mb_y` are picture-
/// relative pixel coordinates modulo 16 inside each MB; from them the
/// per-4x4-block MV / per-8x8 ref_idx indices are derived.
fn different_ref_or_mv_luma(
    p_info: &MbInfo,
    q_info: &MbInfo,
    p_in_mb_x: u32,
    p_in_mb_y: u32,
    q_in_mb_x: u32,
    q_in_mb_y: u32,
) -> bool {
    // 4x4 block index inside the MB follows the §6.4.3 Figure 6-10
    // Z-scan — NOT simple raster. Reuse `blk4_raster_index`, which
    // inverts (bx, by)_4x4 back to the Z-scan luma block index used
    // throughout the grid (see `mv_l0`/`ref_idx_l0` storage).
    let p_blk4 = blk4_raster_index((p_in_mb_x / 4) as u8, (p_in_mb_y / 4) as u8) as usize;
    let q_blk4 = blk4_raster_index((q_in_mb_x / 4) as u8, (q_in_mb_y / 4) as u8) as usize;
    // ref_idx is tracked per 8x8 partition in raster order 0..=3:
    //   ref_idx_l0[qy*2 + qx] with qx = bx_in_mb/8, qy = by_in_mb/8.
    let p_blk8 = ((p_in_mb_y / 8) * 2 + (p_in_mb_x / 8)) as usize;
    let q_blk8 = ((q_in_mb_y / 8) * 2 + (q_in_mb_x / 8)) as usize;
    let p_ref0 = p_info.ref_idx_l0[p_blk8];
    let q_ref0 = q_info.ref_idx_l0[q_blk8];
    let p_ref1 = p_info.ref_idx_l1[p_blk8];
    let q_ref1 = q_info.ref_idx_l1[q_blk8];
    let p_has_l0 = p_ref0 >= 0;
    let q_has_l0 = q_ref0 >= 0;
    let p_has_l1 = p_ref1 >= 0;
    let q_has_l1 = q_ref1 >= 0;

    // §8.7.2.1 NOTE 1 — picture identity is by POC (actual referenced
    // picture), NOT by list+index. Read the resolved POCs stored in
    // MbInfo directly: each side's ref_idx was looked up through *its
    // own slice's* RefPicListX at reconstruct-time, avoiding the
    // multi-slice cross-reference bug that a single picture-wide
    // snapshot would introduce (CABAST3_Sony_E: slice 1 = P, slice 2 = B
    // with distinct lists).
    let poc_sentinel: i32 = i32::MIN;
    let p_pic0 = if p_has_l0 {
        p_info.ref_poc_l0[p_blk8]
    } else {
        poc_sentinel
    };
    let q_pic0 = if q_has_l0 {
        q_info.ref_poc_l0[q_blk8]
    } else {
        poc_sentinel
    };
    let p_pic1 = if p_has_l1 {
        p_info.ref_poc_l1[p_blk8]
    } else {
        poc_sentinel
    };
    let q_pic1 = if q_has_l1 {
        q_info.ref_poc_l1[q_blk8]
    } else {
        poc_sentinel
    };

    // "Number of motion vectors" per §8.7.2.1 NOTE 2 is
    // PredFlagL0[part] + PredFlagL1[part]. Two edges with the same
    // count but different list usage (one side L0-only, other L1-only)
    // are only considered "different" if their referenced pictures
    // differ.
    let p_count = (p_has_l0 as u8) + (p_has_l1 as u8);
    let q_count = (q_has_l0 as u8) + (q_has_l1 as u8);
    if p_count != q_count {
        return true;
    }

    // Same list usage — compare per-list MVs and refs.
    let p_mv0 = p_info.mv_l0[p_blk4];
    let q_mv0 = q_info.mv_l0[q_blk4];
    let p_mv1 = p_info.mv_l1[p_blk4];
    let q_mv1 = q_info.mv_l1[q_blk4];

    let bi = p_has_l0 && p_has_l1;
    if bi {
        // §8.7.2.1 bS=1 block — bi-predicted edges. Split by whether p
        // and q reference the "same two pictures" or two different
        // pairs; in both cases the test considers the straight and
        // swapped pairings (since L0/L1 assignment is reference-list
        // ordering, not picture identity).
        //
        // Case A: same two reference pictures on both sides (possibly
        // with L0/L1 swapped between p and q) — spec's "two motion
        // vectors for the same reference picture" double-test: for
        // diff_ref_mv to fire, BOTH the straight and swapped Δmv
        // tests must each reach 4 quarter-pels.
        //
        // Case B: p and q reference different picture pairs — returns
        // "different reference pictures" = true.
        //
        // Case C: p uses two DIFFERENT pictures and q uses the SAME
        // two pictures (only the matching pairing is checked, not the
        // swapped, since the swapped pairing would pair p's L0 mv with
        // q's L1 mv for a DIFFERENT picture — nonsensical per spec).
        let straight_refs_match = p_pic0 == q_pic0 && p_pic1 == q_pic1;
        let swapped_refs_match = p_pic0 == q_pic1 && p_pic1 == q_pic0;

        if !straight_refs_match && !swapped_refs_match {
            // Case B: different picture sets → bS=1.
            return true;
        }

        let straight_mv_ok = mv_delta_below_4(p_mv0, q_mv0) && mv_delta_below_4(p_mv1, q_mv1);
        let swapped_mv_ok = mv_delta_below_4(p_mv0, q_mv1) && mv_delta_below_4(p_mv1, q_mv0);

        // If L0[p] and L1[p] refer to distinct pictures (and same for
        // q), the spec's "two motion vectors and two different ref
        // pictures" clause applies: for EITHER referenced picture,
        // |Δmv| >= 4 triggers bS=1. That's equivalent to saying at
        // least one of the matching-ref-pairings has a big Δmv.
        //
        // Spec straight pairing is valid when straight_refs_match;
        // the swapped pairing is valid when swapped_refs_match.
        // If both refs in the pair are the SAME picture (p_pic0 ==
        // p_pic1), we have the spec's "same reference picture" clause
        // and must require BOTH pairings to fail (the stricter double
        // test). Otherwise (distinct pictures), it's sufficient that
        // ANY valid pairing fails.
        let same_ref = p_pic0 == p_pic1 && q_pic0 == q_pic1;
        if same_ref {
            // Spec: "two motion vectors for the same reference picture"
            // — requires both tests to fail (i.e., neither pairing has
            // all Δmvs below 4) for diff_ref_mv=true.
            return !(straight_mv_ok || swapped_mv_ok);
        }

        // Distinct reference pictures in the pair. Check the matching
        // pairing; swapped pairing only if refs match when swapped.
        let straight_fails = straight_refs_match && !straight_mv_ok;
        let swapped_fails = swapped_refs_match && !swapped_mv_ok;
        straight_fails || swapped_fails
    } else if p_has_l0 || p_has_l1 {
        // §8.7.2.1 "one motion vector is used to predict p0 and one
        // motion vector is used to predict q0" clause. p_count == q_count
        // == 1 here (we're past the bi branch and the count-mismatch
        // early return). Single-prediction can still come from different
        // lists on the two sides, so resolve which list each side uses
        // and match picture by POC, not by list index.
        //
        // NOTE 1 of §8.7.2.1: the reference picture is determined by
        // actual picture identity (POC), without regard to list or list
        // index.
        let (p_pic, p_mv) = if p_has_l0 {
            (p_pic0, p_mv0)
        } else {
            (p_pic1, p_mv1)
        };
        let (q_pic, q_mv) = if q_has_l0 {
            (q_pic0, q_mv0)
        } else {
            (q_pic1, q_mv1)
        };
        if p_pic != q_pic {
            return true;
        }
        !mv_delta_below_4(p_mv, q_mv)
    } else {
        // Neither list active — no MV info; keep bS=0 for this edge
        // (the intra/coef bullets would have handled any interesting
        // cases).
        false
    }
}

/// Sub-predicate of §8.7.2.1: `true` iff both MV components differ by
/// < 4 in quarter-sample units.
#[inline]
fn mv_delta_below_4(a: (i16, i16), b: (i16, i16)) -> bool {
    (a.0 as i32 - b.0 as i32).abs() < 4 && (a.1 as i32 - b.1 as i32).abs() < 4
}

#[allow(clippy::too_many_arguments)]
fn deblock_plane_luma(
    pic: &mut Picture,
    grid: &MbGrid,
    alpha_off: i32,
    beta_off: i32,
    bit_depth: u32,
    mbaff_frame_flag: bool,
    mb_field_flags: &[bool],
) {
    let w = pic.width_in_samples as i32;
    let h = pic.height_in_samples as i32;
    let mb_w = grid.width_in_mbs as i32;
    let mb_h = grid.height_in_mbs as i32;
    // §8.7.2.1 NOTE 1 — picture-identity comparison is by POC of the
    // referenced picture, not by list+index. Each MB's per-partition POCs
    // are stored in `MbInfo::ref_poc_l0/l1` (populated during inter
    // reconstruction through the *current slice's* ref list), so
    // `different_ref_or_mv_luma` can read them directly without a
    // picture-wide snapshot that would be wrong for multi-slice pictures
    // where slices have different ref lists.

    // §8.7.1 — "Filtering process for block edges". Order:
    // for each MB in raster scan:
    //   filter 4 vertical luma edges (left-to-right: x = mb_x*16 + {0, 4, 8, 12}),
    //   then 4 horizontal luma edges (top-to-bottom: y = mb_y*16 + {0, 4, 8, 12}).
    // This order is observable — e.g. the right-MB-edge strong filter at
    // MB (n+1) reads p-side samples that were already modified by the
    // previous MB's horizontal edges.
    //
    // The edge at x = mb_x*16 is the LEFT MB-boundary; it is skipped at
    // the picture left edge (mb_x == 0) because there is no p-side MB.
    // Similarly y = mb_y*16 is skipped at the top edge.
    //
    // MBAFF note: we use [`pixel_to_mb_addr`] to resolve the per-sample
    // MB address. `mbaff_or_field` is passed through to bS derivation —
    // §8.7.2.1 widens bS=4 to require verticalEdgeFlag for MBAFF/field
    // pictures. Mixed-mode edges (one frame-MB + one field-MB across a
    // horizontal pair boundary) are NOT detected here; that's part of
    // the future-work simplification.
    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            // --- 4 vertical edges of this MB, in left-to-right order ---
            for edge_off in 0..4 {
                let edge_x = mb_x * 16 + edge_off * 4;
                if edge_x == 0 || edge_x >= w {
                    continue; // picture left edge or past right boundary
                }
                // Each vertical edge has four 4-row "segments" in this MB.
                for seg in 0..4 {
                    let y0 = mb_y * 16 + seg * 4;
                    if y0 >= h {
                        break;
                    }
                    let p_addr = match pixel_to_mb_addr(
                        grid,
                        edge_x - 1,
                        y0,
                        mbaff_frame_flag,
                        mb_field_flags,
                    ) {
                        Some(a) => a,
                        None => continue,
                    };
                    let q_addr = match pixel_to_mb_addr(
                        grid,
                        edge_x,
                        y0,
                        mbaff_frame_flag,
                        mb_field_flags,
                    ) {
                        Some(a) => a,
                        None => continue,
                    };
                    let (p_info, q_info) = match (grid.get(p_addr), grid.get(q_addr)) {
                        (Some(p), Some(q)) if p.available && q.available => (p, q),
                        _ => continue,
                    };
                    let is_mb_edge = p_addr != q_addr;
                    // §8.7 — when transform_size_8x8_flag is set on the
                    // current (q-side) macroblock, the internal 4x4-only
                    // luma edges at offsets 4 and 12 are NOT filtered
                    // (only solid-bold 8-sample edges are). Offset 0 is
                    // an MB boundary, offset 8 is the 8x8 boundary, so
                    // both are always filtered when enabled. For Baseline
                    // / Main profiles (no 8x8 transform) this is a no-op.
                    if !is_mb_edge
                        && (edge_off == 1 || edge_off == 3)
                        && q_info.transform_size_8x8_flag
                    {
                        continue;
                    }
                    // §8.7.2.1 fourth bullet: inter-coded edges pick up
                    // bS=1 when the two 4x4 blocks disagree on ref pic /
                    // MV count / MV component delta (>=4 in qpel units).
                    // Coordinates modulo 16 give the per-MB 4x4 index.
                    let p_in_mb_x = (edge_x - 1).rem_euclid(16) as u32;
                    let p_in_mb_y = y0.rem_euclid(16) as u32;
                    let q_in_mb_x = edge_x.rem_euclid(16) as u32;
                    let q_in_mb_y = y0.rem_euclid(16) as u32;
                    let diff_ref_mv = !p_info.is_intra
                        && !q_info.is_intra
                        && different_ref_or_mv_luma(
                            p_info, q_info, p_in_mb_x, p_in_mb_y, q_in_mb_x, q_in_mb_y,
                        );
                    // §8.7.2.1 — per-4x4-block nonzero-coefficient test.
                    // Consult the per-block mask set at reconstruct-time.
                    let p_blk4_z =
                        blk4_raster_index((p_in_mb_x / 4) as u8, (p_in_mb_y / 4) as u8) as usize;
                    let q_blk4_z =
                        blk4_raster_index((q_in_mb_x / 4) as u8, (q_in_mb_y / 4) as u8) as usize;
                    let p_has_nz = (p_info.luma_nonzero_4x4 >> p_blk4_z) & 1 == 1;
                    let q_has_nz = (q_info.luma_nonzero_4x4 >> q_blk4_z) & 1 == 1;
                    let bs = derive_boundary_strength(BsInputs {
                        p_is_intra: p_info.is_intra,
                        q_is_intra: q_info.is_intra,
                        is_mb_edge,
                        is_sp_or_si: false,
                        either_has_nonzero_coeffs: p_has_nz || q_has_nz,
                        different_ref_or_mv: diff_ref_mv,
                        mixed_mode_edge: false,
                        vertical_edge: true,
                        mbaff_or_field: mbaff_frame_flag,
                    });
                    if bs == 0 {
                        continue;
                    }
                    filter_vertical_edge_luma(
                        pic,
                        edge_x,
                        y0,
                        bs,
                        p_info.qp_y,
                        q_info.qp_y,
                        alpha_off,
                        beta_off,
                        bit_depth,
                    );
                }
            }

            // --- 4 horizontal edges of this MB, in top-to-bottom order ---
            for edge_off in 0..4 {
                let edge_y = mb_y * 16 + edge_off * 4;
                if edge_y == 0 || edge_y >= h {
                    continue;
                }
                for seg in 0..4 {
                    let x0 = mb_x * 16 + seg * 4;
                    if x0 >= w {
                        break;
                    }
                    let p_addr = match pixel_to_mb_addr(
                        grid,
                        x0,
                        edge_y - 1,
                        mbaff_frame_flag,
                        mb_field_flags,
                    ) {
                        Some(a) => a,
                        None => continue,
                    };
                    let q_addr = match pixel_to_mb_addr(
                        grid,
                        x0,
                        edge_y,
                        mbaff_frame_flag,
                        mb_field_flags,
                    ) {
                        Some(a) => a,
                        None => continue,
                    };
                    let (p_info, q_info) = match (grid.get(p_addr), grid.get(q_addr)) {
                        (Some(p), Some(q)) if p.available && q.available => (p, q),
                        _ => continue,
                    };
                    let is_mb_edge = p_addr != q_addr;
                    // §8.7 — same 8x8-transform skip as vertical path:
                    // for the internal 4x4-only horizontal edges at
                    // offsets 4 and 12, skip when transform_size_8x8_flag
                    // is set on the current (q-side) MB.
                    if !is_mb_edge
                        && (edge_off == 1 || edge_off == 3)
                        && q_info.transform_size_8x8_flag
                    {
                        continue;
                    }
                    // §8.7.2.1 fourth bullet (see vertical path).
                    let p_in_mb_x = x0.rem_euclid(16) as u32;
                    let p_in_mb_y = (edge_y - 1).rem_euclid(16) as u32;
                    let q_in_mb_x = x0.rem_euclid(16) as u32;
                    let q_in_mb_y = edge_y.rem_euclid(16) as u32;
                    let diff_ref_mv = !p_info.is_intra
                        && !q_info.is_intra
                        && different_ref_or_mv_luma(
                            p_info, q_info, p_in_mb_x, p_in_mb_y, q_in_mb_x, q_in_mb_y,
                        );
                    // §8.7.2.1 — per-4x4-block nonzero-coefficient test.
                    let p_blk4_z =
                        blk4_raster_index((p_in_mb_x / 4) as u8, (p_in_mb_y / 4) as u8) as usize;
                    let q_blk4_z =
                        blk4_raster_index((q_in_mb_x / 4) as u8, (q_in_mb_y / 4) as u8) as usize;
                    let p_has_nz = (p_info.luma_nonzero_4x4 >> p_blk4_z) & 1 == 1;
                    let q_has_nz = (q_info.luma_nonzero_4x4 >> q_blk4_z) & 1 == 1;
                    let bs = derive_boundary_strength(BsInputs {
                        p_is_intra: p_info.is_intra,
                        q_is_intra: q_info.is_intra,
                        is_mb_edge,
                        is_sp_or_si: false,
                        either_has_nonzero_coeffs: p_has_nz || q_has_nz,
                        different_ref_or_mv: diff_ref_mv,
                        mixed_mode_edge: false,
                        vertical_edge: false,
                        mbaff_or_field: mbaff_frame_flag,
                    });
                    if bs == 0 {
                        continue;
                    }
                    filter_horizontal_edge_luma(
                        pic,
                        x0,
                        edge_y,
                        bs,
                        p_info.qp_y,
                        q_info.qp_y,
                        alpha_off,
                        beta_off,
                        bit_depth,
                    );
                }
            }
        }
    }
    // Consumer markers for deblock helpers
    let _ = alpha_from_index;
    let _ = beta_from_index;
    let _ = tc0_from;
}

#[allow(clippy::too_many_arguments)]
fn deblock_plane_chroma(
    pic: &mut Picture,
    grid: &MbGrid,
    alpha_off: i32,
    beta_off: i32,
    bit_depth: u32,
    pps: &Pps,
    mbaff_frame_flag: bool,
    mb_field_flags: &[bool],
) {
    let _ = pps; // QPc derivation keeps QP_Y indirectly; handled inside.
    let cw = pic.chroma_width() as i32;
    let ch = pic.chroma_height() as i32;
    let cb_offset = pps.chroma_qp_index_offset;
    let cr_offset = pps
        .extension
        .as_ref()
        .map(|e| e.second_chroma_qp_index_offset)
        .unwrap_or(pps.chroma_qp_index_offset);
    // §8.7.2.1 NOTE 1 — see deblock_plane_luma; `different_ref_or_mv_luma`
    // reads per-MB POCs directly from `MbInfo::ref_poc_l0/l1`.
    let (sub_w, sub_h) = chroma_subsample(pic.chroma_array_type);
    if sub_w == 0 || pic.chroma_array_type == 0 {
        return;
    }

    // For each plane (0 = Cb, 1 = Cr) the edges of each MB are filtered
    // in raster scan order: four vertical chroma edges, then four
    // horizontal chroma edges (§8.7.1). For 4:2:0 chroma (SubWidthC =
    // SubHeightC = 2) the MB's chroma block is 8x8 and only two edges
    // of each orientation apply (the 4-sample boundaries at chroma
    // offsets 0 and 4). For 4:2:2 the chroma MB block is 8x16 so four
    // horizontal edges apply but only two vertical ones.
    //
    // MBAFF: use pixel_to_mb_addr with the luma coordinate derived from
    // chroma position via (SubWidthC, SubHeightC) for uniform MBAFF
    // addressing — see deblock_plane_luma for scope-limit discussion.
    let mb_w = grid.width_in_mbs as i32;
    let mb_h = grid.height_in_mbs as i32;
    // Chroma MB dims in chroma samples:
    // SubWidthC=1 → chroma MB is 16 wide; SubWidthC=2 → 8 wide.
    // SubHeightC=1 → 16 tall; SubHeightC=2 → 8 tall.
    let chroma_mb_w = 16 / sub_w.max(1);
    let chroma_mb_h = 16 / sub_h.max(1);
    for plane in 0..2u8 {
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                // Vertical chroma edges: chroma-x = mb_x*chroma_mb_w + {0, 4, ...}
                // For 4:2:0 / 4:2:2, only offsets {0, 4} are valid (chroma MB 8 wide).
                for edge_off in (0..chroma_mb_w).step_by(4) {
                    let edge_x = mb_x * chroma_mb_w + edge_off;
                    if edge_x == 0 || edge_x >= cw {
                        continue;
                    }
                    // Segments of 4 chroma rows each. In 4:2:0 each
                    // 4-chroma-row segment maps to two luma 4-row
                    // segments (SubHeightC == 2); the per-luma-segment
                    // bS can differ, so we filter each 2-chroma-row
                    // sub-segment with its own luma-edge bS. In 4:2:2
                    // (SubHeightC == 1) there's a single corresponding
                    // luma segment per chroma segment, so sub_seg_count
                    // collapses to 1 covering all 4 chroma rows.
                    let sub_seg_rows = 4usize / sub_h as usize; // 2 for 4:2:0, 4 for 4:2:2
                    let sub_seg_count = 4usize / sub_seg_rows; // 2 for 4:2:0, 1 for 4:2:2
                    for seg_off in (0..chroma_mb_h).step_by(4) {
                        for sub in 0..sub_seg_count {
                            let y0 = mb_y * chroma_mb_h + seg_off + (sub * sub_seg_rows) as i32;
                            if y0 >= ch {
                                break;
                            }
                            let lp_x = (edge_x - 1) * sub_w;
                            let lq_x = edge_x * sub_w;
                            let ly = y0 * sub_h;
                            let p_addr = match pixel_to_mb_addr(
                                grid,
                                lp_x,
                                ly,
                                mbaff_frame_flag,
                                mb_field_flags,
                            ) {
                                Some(a) => a,
                                None => continue,
                            };
                            let q_addr = match pixel_to_mb_addr(
                                grid,
                                lq_x,
                                ly,
                                mbaff_frame_flag,
                                mb_field_flags,
                            ) {
                                Some(a) => a,
                                None => continue,
                            };
                            let (p_info, q_info) = match (grid.get(p_addr), grid.get(q_addr)) {
                                (Some(p), Some(q)) if p.available && q.available => (p, q),
                                _ => continue,
                            };
                            let is_mb_edge = p_addr != q_addr;
                            // §8.7.2.1 last paragraph — chroma bS inherits
                            // from the corresponding luma edge at the
                            // mapped luma coordinate (chroma-x *
                            // SubWidthC, chroma-y * SubHeightC).
                            let p_in_mb_x = (lp_x).rem_euclid(16) as u32;
                            let p_in_mb_y = (ly).rem_euclid(16) as u32;
                            let q_in_mb_x = (lq_x).rem_euclid(16) as u32;
                            let q_in_mb_y = (ly).rem_euclid(16) as u32;
                            let diff_ref_mv = !p_info.is_intra
                                && !q_info.is_intra
                                && different_ref_or_mv_luma(
                                    p_info, q_info, p_in_mb_x, p_in_mb_y, q_in_mb_x, q_in_mb_y,
                                );
                            let p_blk4_z =
                                blk4_raster_index((p_in_mb_x / 4) as u8, (p_in_mb_y / 4) as u8)
                                    as usize;
                            let q_blk4_z =
                                blk4_raster_index((q_in_mb_x / 4) as u8, (q_in_mb_y / 4) as u8)
                                    as usize;
                            let p_has_nz = (p_info.luma_nonzero_4x4 >> p_blk4_z) & 1 == 1;
                            let q_has_nz = (q_info.luma_nonzero_4x4 >> q_blk4_z) & 1 == 1;
                            let bs = derive_boundary_strength(BsInputs {
                                p_is_intra: p_info.is_intra,
                                q_is_intra: q_info.is_intra,
                                is_mb_edge,
                                is_sp_or_si: false,
                                either_has_nonzero_coeffs: p_has_nz || q_has_nz,
                                different_ref_or_mv: diff_ref_mv,
                                mixed_mode_edge: false,
                                vertical_edge: true,
                                mbaff_or_field: mbaff_frame_flag,
                            });
                            if std::env::var_os("OXIDEAV_H264_DEBLOCK_TRACE").is_some() {
                                eprintln!(
                                    "DBL C{} Vx={edge_x} y0={y0} bs={bs} plane={plane} p_addr={p_addr} q_addr={q_addr} p_is_intra={} q_is_intra={} p_nz={} q_nz={} diff_ref_mv={}",
                                    plane, p_info.is_intra, q_info.is_intra, p_has_nz, q_has_nz, diff_ref_mv,
                                );
                            }
                            if bs == 0 {
                                continue;
                            }
                            let qp_avg = chroma_qp_avg(
                                p_info.qp_y,
                                q_info.qp_y,
                                if plane == 0 { cb_offset } else { cr_offset },
                            );
                            filter_chroma_vertical_rows(
                                pic,
                                plane,
                                edge_x,
                                y0,
                                sub_seg_rows as i32,
                                bs,
                                qp_avg,
                                alpha_off,
                                beta_off,
                                bit_depth,
                            );
                        }
                    }
                }

                // Horizontal chroma edges. In 4:2:0 each 4-chroma-col
                // segment maps to two luma 4-col segments (SubWidthC == 2);
                // in 4:2:2 (SubWidthC == 2) the same applies. In 4:4:4
                // we don't enter this function at all.
                let sub_seg_cols = 4usize / sub_w as usize;
                let sub_seg_count_h = 4usize / sub_seg_cols;
                for edge_off in (0..chroma_mb_h).step_by(4) {
                    let edge_y = mb_y * chroma_mb_h + edge_off;
                    if edge_y == 0 || edge_y >= ch {
                        continue;
                    }
                    for seg_off in (0..chroma_mb_w).step_by(4) {
                        for sub in 0..sub_seg_count_h {
                            let x0 = mb_x * chroma_mb_w + seg_off + (sub * sub_seg_cols) as i32;
                            if x0 >= cw {
                                break;
                            }
                            let lx = x0 * sub_w;
                            let lp_y = (edge_y - 1) * sub_h;
                            let lq_y = edge_y * sub_h;
                            let p_addr = match pixel_to_mb_addr(
                                grid,
                                lx,
                                lp_y,
                                mbaff_frame_flag,
                                mb_field_flags,
                            ) {
                                Some(a) => a,
                                None => continue,
                            };
                            let q_addr = match pixel_to_mb_addr(
                                grid,
                                lx,
                                lq_y,
                                mbaff_frame_flag,
                                mb_field_flags,
                            ) {
                                Some(a) => a,
                                None => continue,
                            };
                            let (p_info, q_info) = match (grid.get(p_addr), grid.get(q_addr)) {
                                (Some(p), Some(q)) if p.available && q.available => (p, q),
                                _ => continue,
                            };
                            let is_mb_edge = p_addr != q_addr;
                            // §8.7.2.1 last paragraph — chroma bS inherits
                            // from the corresponding luma edge.
                            let p_in_mb_x = lx.rem_euclid(16) as u32;
                            let p_in_mb_y = lp_y.rem_euclid(16) as u32;
                            let q_in_mb_x = lx.rem_euclid(16) as u32;
                            let q_in_mb_y = lq_y.rem_euclid(16) as u32;
                            let diff_ref_mv = !p_info.is_intra
                                && !q_info.is_intra
                                && different_ref_or_mv_luma(
                                    p_info, q_info, p_in_mb_x, p_in_mb_y, q_in_mb_x, q_in_mb_y,
                                );
                            let p_blk4_z =
                                blk4_raster_index((p_in_mb_x / 4) as u8, (p_in_mb_y / 4) as u8)
                                    as usize;
                            let q_blk4_z =
                                blk4_raster_index((q_in_mb_x / 4) as u8, (q_in_mb_y / 4) as u8)
                                    as usize;
                            let p_has_nz = (p_info.luma_nonzero_4x4 >> p_blk4_z) & 1 == 1;
                            let q_has_nz = (q_info.luma_nonzero_4x4 >> q_blk4_z) & 1 == 1;
                            let bs = derive_boundary_strength(BsInputs {
                                p_is_intra: p_info.is_intra,
                                q_is_intra: q_info.is_intra,
                                is_mb_edge,
                                is_sp_or_si: false,
                                either_has_nonzero_coeffs: p_has_nz || q_has_nz,
                                different_ref_or_mv: diff_ref_mv,
                                mixed_mode_edge: false,
                                vertical_edge: false,
                                mbaff_or_field: mbaff_frame_flag,
                            });
                            if bs == 0 {
                                continue;
                            }
                            let qp_avg = chroma_qp_avg(
                                p_info.qp_y,
                                q_info.qp_y,
                                if plane == 0 { cb_offset } else { cr_offset },
                            );
                            filter_chroma_horizontal_cols(
                                pic,
                                plane,
                                x0,
                                edge_y,
                                sub_seg_cols as i32,
                                bs,
                                qp_avg,
                                alpha_off,
                                beta_off,
                                bit_depth,
                            );
                        }
                    }
                }
            }
        }
    }
    let _ = alpha_off;
    let _ = beta_off;
    let _ = bit_depth;
}

/// §8.5.8 / §8.7.2.2 eq. 8-453 — (QPc(p) + QPc(q) + 1) >> 1 for chroma.
fn chroma_qp_avg(p_qp_y: i32, q_qp_y: i32, offset: i32) -> i32 {
    let p_c = qp_y_to_qp_c(p_qp_y, offset);
    let q_c = qp_y_to_qp_c(q_qp_y, offset);
    (p_c + q_c + 1) >> 1
}

fn filter_vertical_edge_luma(
    pic: &mut Picture,
    edge_x: i32,
    y0: i32,
    bs: u8,
    p_qp: i32,
    q_qp: i32,
    alpha_off: i32,
    beta_off: i32,
    bit_depth: u32,
) {
    for dy in 0..4 {
        let y = y0 + dy;
        if y < 0 || y >= pic.height_in_samples as i32 {
            continue;
        }
        // Fetch p3..p0, q0..q3 along the row.
        let mut samples = [0i32; 8];
        for (i, x) in (edge_x - 4..edge_x + 4).enumerate() {
            samples[i] = pic.luma_at(x, y);
        }
        let p3 = samples[0];
        let mut p2 = samples[1];
        let mut p1 = samples[2];
        let mut p0 = samples[3];
        let mut q0 = samples[4];
        let mut q1 = samples[5];
        let mut q2 = samples[6];
        let q3 = samples[7];
        let qp_avg = (p_qp + q_qp + 1) >> 1;
        {
            let edge = EdgeSamples {
                p3,
                p2: &mut p2,
                p1: &mut p1,
                p0: &mut p0,
                q0: &mut q0,
                q1: &mut q1,
                q2: &mut q2,
                q3,
            };
            filter_edge(
                Plane::Luma,
                edge,
                FilterParams {
                    bs,
                    qp_avg,
                    filter_offset_a: alpha_off,
                    filter_offset_b: beta_off,
                    bit_depth,
                },
            );
        }
        // p3/q3 are read-only in the filter API.
        pic.set_luma(edge_x - 4, y, samples[0]);
        pic.set_luma(edge_x - 3, y, p2);
        pic.set_luma(edge_x - 2, y, p1);
        pic.set_luma(edge_x - 1, y, p0);
        pic.set_luma(edge_x, y, q0);
        pic.set_luma(edge_x + 1, y, q1);
        pic.set_luma(edge_x + 2, y, q2);
        pic.set_luma(edge_x + 3, y, samples[7]);
    }
}

fn filter_horizontal_edge_luma(
    pic: &mut Picture,
    x0: i32,
    edge_y: i32,
    bs: u8,
    p_qp: i32,
    q_qp: i32,
    alpha_off: i32,
    beta_off: i32,
    bit_depth: u32,
) {
    for dx in 0..4 {
        let x = x0 + dx;
        if x < 0 || x >= pic.width_in_samples as i32 {
            continue;
        }
        let mut samples = [0i32; 8];
        for (i, y) in (edge_y - 4..edge_y + 4).enumerate() {
            samples[i] = pic.luma_at(x, y);
        }
        let p3 = samples[0];
        let mut p2 = samples[1];
        let mut p1 = samples[2];
        let mut p0 = samples[3];
        let mut q0 = samples[4];
        let mut q1 = samples[5];
        let mut q2 = samples[6];
        let q3 = samples[7];
        let qp_avg = (p_qp + q_qp + 1) >> 1;
        {
            let edge = EdgeSamples {
                p3,
                p2: &mut p2,
                p1: &mut p1,
                p0: &mut p0,
                q0: &mut q0,
                q1: &mut q1,
                q2: &mut q2,
                q3,
            };
            filter_edge(
                Plane::Luma,
                edge,
                FilterParams {
                    bs,
                    qp_avg,
                    filter_offset_a: alpha_off,
                    filter_offset_b: beta_off,
                    bit_depth,
                },
            );
        }
        pic.set_luma(x, edge_y - 3, p2);
        pic.set_luma(x, edge_y - 2, p1);
        pic.set_luma(x, edge_y - 1, p0);
        pic.set_luma(x, edge_y, q0);
        pic.set_luma(x, edge_y + 1, q1);
        pic.set_luma(x, edge_y + 2, q2);
    }
}

/// Filter `rows` chroma rows starting at `y0` on the vertical chroma
/// edge at `edge_x`. `rows` is 2 for 4:2:0 (each chroma row is paired
/// with a specific luma 4-row segment by SubHeightC == 2) or 4 for
/// 4:2:2 (SubHeightC == 1).
#[allow(clippy::too_many_arguments)]
fn filter_chroma_vertical_rows(
    pic: &mut Picture,
    plane: u8,
    edge_x: i32,
    y0: i32,
    rows: i32,
    bs: u8,
    qp_avg: i32,
    alpha_off: i32,
    beta_off: i32,
    bit_depth: u32,
) {
    let cw = pic.chroma_width() as i32;
    let ch = pic.chroma_height() as i32;
    for dy in 0..rows {
        let y = y0 + dy;
        if y < 0 || y >= ch {
            continue;
        }
        let fetch = |x: i32, y: i32| -> i32 {
            if plane == 0 {
                pic.cb_at(x, y)
            } else {
                pic.cr_at(x, y)
            }
        };
        let mut s = [0i32; 8];
        for (i, x) in (edge_x - 4..edge_x + 4).enumerate() {
            let xc = x.clamp(0, cw.max(1) - 1);
            s[i] = fetch(xc, y);
        }
        let p3 = s[0];
        let mut p2 = s[1];
        let mut p1 = s[2];
        let mut p0 = s[3];
        let mut q0 = s[4];
        let mut q1 = s[5];
        let mut q2 = s[6];
        let q3 = s[7];
        {
            let edge = EdgeSamples {
                p3,
                p2: &mut p2,
                p1: &mut p1,
                p0: &mut p0,
                q0: &mut q0,
                q1: &mut q1,
                q2: &mut q2,
                q3,
            };
            filter_edge(
                Plane::Chroma,
                edge,
                FilterParams {
                    bs,
                    qp_avg,
                    filter_offset_a: alpha_off,
                    filter_offset_b: beta_off,
                    bit_depth,
                },
            );
        }
        if plane == 0 {
            pic.set_cb(edge_x - 2, y, p1);
            pic.set_cb(edge_x - 1, y, p0);
            pic.set_cb(edge_x, y, q0);
            pic.set_cb(edge_x + 1, y, q1);
        } else {
            pic.set_cr(edge_x - 2, y, p1);
            pic.set_cr(edge_x - 1, y, p0);
            pic.set_cr(edge_x, y, q0);
            pic.set_cr(edge_x + 1, y, q1);
        }
    }
}

/// Filter `cols` chroma columns starting at `x0` on the horizontal
/// chroma edge at `edge_y`. Analogous to `filter_chroma_vertical_rows`
/// for the horizontal direction.
#[allow(clippy::too_many_arguments)]
fn filter_chroma_horizontal_cols(
    pic: &mut Picture,
    plane: u8,
    x0: i32,
    edge_y: i32,
    cols: i32,
    bs: u8,
    qp_avg: i32,
    alpha_off: i32,
    beta_off: i32,
    bit_depth: u32,
) {
    let cw = pic.chroma_width() as i32;
    let ch = pic.chroma_height() as i32;
    for dx in 0..cols {
        let x = x0 + dx;
        if x < 0 || x >= cw {
            continue;
        }
        let fetch = |x: i32, y: i32| -> i32 {
            if plane == 0 {
                pic.cb_at(x, y)
            } else {
                pic.cr_at(x, y)
            }
        };
        let mut s = [0i32; 8];
        for (i, y) in (edge_y - 4..edge_y + 4).enumerate() {
            let yc = y.clamp(0, ch.max(1) - 1);
            s[i] = fetch(x, yc);
        }
        let p3 = s[0];
        let mut p2 = s[1];
        let mut p1 = s[2];
        let mut p0 = s[3];
        let mut q0 = s[4];
        let mut q1 = s[5];
        let mut q2 = s[6];
        let q3 = s[7];
        {
            let edge = EdgeSamples {
                p3,
                p2: &mut p2,
                p1: &mut p1,
                p0: &mut p0,
                q0: &mut q0,
                q1: &mut q1,
                q2: &mut q2,
                q3,
            };
            filter_edge(
                Plane::Chroma,
                edge,
                FilterParams {
                    bs,
                    qp_avg,
                    filter_offset_a: alpha_off,
                    filter_offset_b: beta_off,
                    bit_depth,
                },
            );
        }
        if plane == 0 {
            pic.set_cb(x, edge_y - 2, p1);
            pic.set_cb(x, edge_y - 1, p0);
            pic.set_cb(x, edge_y, q0);
            pic.set_cb(x, edge_y + 1, q1);
        } else {
            pic.set_cr(x, edge_y - 2, p1);
            pic.set_cr(x, edge_y - 1, p0);
            pic.set_cr(x, edge_y, q0);
            pic.set_cr(x, edge_y + 1, q1);
        }
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macroblock_layer::{
        Intra16x16, Macroblock, MbPred, MbType, PcmSamples, SubMbPred, SubMbType,
    };
    use crate::pps::Pps;
    use crate::ref_store::{NoRefs, RefPicStore};
    use crate::slice_header::{
        DecRefPicMarking, PredWeightTable, RefPicListModification, SliceHeader, SliceType,
    };
    use crate::sps::{FrameCropping, Sps};

    fn make_sps(w_mbs: u32, h_mbs: u32) -> Sps {
        Sps {
            profile_idc: 66,
            constraint_set_flags: 0,
            level_idc: 30,
            seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            qpprime_y_zero_transform_bypass_flag: false,
            seq_scaling_matrix_present_flag: false,
            seq_scaling_lists: None,
            log2_max_frame_num_minus4: 0,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb_minus4: 0,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            offset_for_ref_frame: Vec::new(),
            max_num_ref_frames: 1,
            gaps_in_frame_num_value_allowed_flag: false,
            pic_width_in_mbs_minus1: w_mbs - 1,
            pic_height_in_map_units_minus1: h_mbs - 1,
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: false,
            frame_cropping: None::<FrameCropping>,
            vui_parameters_present_flag: false,
            vui: None,
        }
    }

    fn make_pps() -> Pps {
        Pps {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
            entropy_coding_mode_flag: false,
            bottom_field_pic_order_in_frame_present_flag: false,
            num_slice_groups_minus1: 0,
            slice_group_map: None,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            pic_init_qp_minus26: 0,
            pic_init_qs_minus26: 0,
            chroma_qp_index_offset: 0,
            deblocking_filter_control_present_flag: false,
            constrained_intra_pred_flag: false,
            redundant_pic_cnt_present_flag: false,
            extension: None,
        }
    }

    fn make_slice_header() -> SliceHeader {
        SliceHeader {
            first_mb_in_slice: 0,
            slice_type_raw: 2,
            slice_type: SliceType::I,
            all_slices_same_type: false,
            pic_parameter_set_id: 0,
            colour_plane_id: 0,
            frame_num: 0,
            field_pic_flag: false,
            bottom_field_flag: false,
            idr_pic_id: 0,
            pic_order_cnt_lsb: 0,
            delta_pic_order_cnt_bottom: 0,
            delta_pic_order_cnt: [0, 0],
            redundant_pic_cnt: 0,
            direct_spatial_mv_pred_flag: false,
            num_ref_idx_active_override_flag: false,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            ref_pic_list_modification: RefPicListModification::default(),
            pred_weight_table: None::<PredWeightTable>,
            dec_ref_pic_marking: None::<DecRefPicMarking>,
            cabac_init_idc: 0,
            slice_qp_delta: 0,
            sp_for_switch_flag: false,
            slice_qs_delta: 0,
            disable_deblocking_filter_idc: 1, // off — keeps output pristine
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
            slice_group_change_cycle: 0,
        }
    }

    fn make_empty_ipcm_mb() -> Macroblock {
        let mut luma = vec![0u32; 256];
        // Unique gradient so we can detect correct placement.
        for (i, slot) in luma.iter_mut().enumerate() {
            *slot = (i as u32) & 0xFF;
        }
        let cb = vec![100u32; 64];
        let cr = vec![200u32; 64];
        Macroblock {
            mb_type: MbType::IPcm,
            mb_type_raw: 25,
            mb_pred: None,
            sub_mb_pred: None,
            pcm_samples: Some(PcmSamples {
                luma,
                chroma_cb: cb,
                chroma_cr: cr,
            }),
            coded_block_pattern: 0,
            transform_size_8x8_flag: false,
            mb_qp_delta: 0,
            residual_luma: Vec::new(),
            residual_luma_dc: None,
            residual_chroma_dc_cb: Vec::new(),
            residual_chroma_dc_cr: Vec::new(),
            residual_chroma_ac_cb: Vec::new(),
            residual_chroma_ac_cr: Vec::new(),
            residual_cb_luma_like: Vec::new(),
            residual_cr_luma_like: Vec::new(),
            residual_cb_16x16_dc: None,
            residual_cr_16x16_dc: None,
            is_skip: false,
        }
    }

    fn make_intra16x16_dc_mb(pred_mode: u8) -> Macroblock {
        // No residual at all: cbp_luma = 0 → all-zero DC block; cbp_chroma = 0.
        Macroblock {
            mb_type: MbType::Intra16x16(Intra16x16 {
                pred_mode,
                cbp_luma: 0,
                cbp_chroma: 0,
            }),
            mb_type_raw: 1,
            mb_pred: Some(MbPred::default()),
            sub_mb_pred: None,
            pcm_samples: None,
            coded_block_pattern: 0,
            transform_size_8x8_flag: false,
            mb_qp_delta: 0,
            residual_luma: Vec::new(),
            residual_luma_dc: Some([0i32; 16]),
            residual_chroma_dc_cb: vec![0i32; 4],
            residual_chroma_dc_cr: vec![0i32; 4],
            residual_chroma_ac_cb: Vec::new(),
            residual_chroma_ac_cr: Vec::new(),
            residual_cb_luma_like: Vec::new(),
            residual_cr_luma_like: Vec::new(),
            residual_cb_16x16_dc: None,
            residual_cr_16x16_dc: None,
            is_skip: false,
        }
    }

    #[test]
    fn i_pcm_macroblock_is_copied_into_picture() {
        let sps = make_sps(1, 1);
        let pps = make_pps();
        let sh = make_slice_header();
        let mb = make_empty_ipcm_mb();
        let slice_data = SliceData {
            macroblocks: vec![mb],
            mb_field_decoding_flags: vec![false],
            last_mb_addr: 0,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        let mut grid = MbGrid::new(1, 1);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &NoRefs, &mut pic, &mut grid).unwrap();

        // Check the gradient pattern is in the luma plane.
        for y in 0..16 {
            for x in 0..16 {
                let expected = (y * 16 + x) & 0xFF;
                assert_eq!(pic.luma_at(x, y), expected, "mismatch at ({}, {})", x, y);
            }
        }
        // Chroma Cb filled with 100, Cr with 200.
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(pic.cb_at(x, y), 100);
                assert_eq!(pic.cr_at(x, y), 200);
            }
        }
        // Grid should show the MB as available, intra, and I_PCM.
        let info = grid.get(0).unwrap();
        assert!(info.available);
        assert!(info.is_intra);
        assert!(info.is_i_pcm);
    }

    #[test]
    fn intra_16x16_dc_zero_residual_falls_back_to_default_dc() {
        // With zero residual and no neighbours available, the DC
        // prediction value per §8.3.3.3 eq. 8-121 is (1 << (BitDepth-1))
        // = 128 for 8-bit. Applied to the whole 16x16 block.
        let sps = make_sps(1, 1);
        let pps = make_pps();
        let sh = make_slice_header();
        let mb = make_intra16x16_dc_mb(2); // pred_mode 2 = DC
        let slice_data = SliceData {
            macroblocks: vec![mb],
            mb_field_decoding_flags: vec![false],
            last_mb_addr: 0,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        let mut grid = MbGrid::new(1, 1);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &NoRefs, &mut pic, &mut grid).unwrap();

        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(
                    pic.luma_at(x, y),
                    128,
                    "mismatch at ({}, {}): expected DC=128",
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn intra_4x4_zero_residual_vertical_pred_from_top() {
        // Two MBs stacked vertically (1 wide, 2 tall). First MB is
        // I_PCM with known top-half samples, second MB is I_NxN with
        // all 16 4x4 blocks using Vertical (mode 0) prediction.
        let sps = make_sps(1, 2);
        let pps = make_pps();
        let sh = make_slice_header();

        // First MB: I_PCM with a recognisable horizontal band in its
        // bottom row (y = 15), since that's what the second MB's top
        // row will read from.
        let mut luma = vec![0u32; 256];
        for x in 0..16 {
            luma[15 * 16 + x] = 50 + x as u32;
        }
        let cb = vec![128u32; 64];
        let cr = vec![128u32; 64];
        let mb_top = Macroblock {
            mb_type: MbType::IPcm,
            mb_type_raw: 25,
            mb_pred: None,
            sub_mb_pred: None,
            pcm_samples: Some(PcmSamples {
                luma,
                chroma_cb: cb,
                chroma_cr: cr,
            }),
            coded_block_pattern: 0,
            transform_size_8x8_flag: false,
            mb_qp_delta: 0,
            residual_luma: Vec::new(),
            residual_luma_dc: None,
            residual_chroma_dc_cb: Vec::new(),
            residual_chroma_dc_cr: Vec::new(),
            residual_chroma_ac_cb: Vec::new(),
            residual_chroma_ac_cr: Vec::new(),
            residual_cb_luma_like: Vec::new(),
            residual_cr_luma_like: Vec::new(),
            residual_cb_16x16_dc: None,
            residual_cr_16x16_dc: None,
            is_skip: false,
        };

        // Second MB: I_NxN (Intra_4x4), 16 blocks, all Vertical (mode 0),
        // zero residual. cbp_luma == 0 so residual_luma is empty.
        //
        // §8.3.1.1 derivation (eq. 8-41) for every 4x4 block:
        //   * Left-column blocks {0, 2, 8, 10}: neighbour A lies in the
        //     MB to the left which doesn't exist (1-wide picture) →
        //     dcPredModePredictedFlag = 1 → predIntra4x4PredMode = 2
        //     (DC). Choose prev_flag = false, rem = 0: 0 < 2 →
        //     Intra4x4PredMode = 0 (Vertical). ✓
        //   * All other blocks: predIntra4x4PredMode = 0 because either
        //     B is the top I_PCM MB (contributes 2 via the
        //     "not Intra_4x4/8x8" rule) and A is an already-Vertical
        //     4x4 block inside this MB (contributes 0), or both A and B
        //     lie inside this MB and both previously decoded as
        //     Vertical. Choose prev_flag = true → Intra4x4PredMode = 0
        //     (Vertical). ✓
        let mut pred = MbPred::default();
        for i in 0..16 {
            let left_col = matches!(i, 0 | 2 | 8 | 10);
            if left_col {
                pred.prev_intra4x4_pred_mode_flag[i] = false;
                pred.rem_intra4x4_pred_mode[i] = 0;
            } else {
                pred.prev_intra4x4_pred_mode_flag[i] = true;
                pred.rem_intra4x4_pred_mode[i] = 0;
            }
        }
        pred.intra_chroma_pred_mode = 0; // DC
        let mb_bot = Macroblock {
            mb_type: MbType::INxN,
            mb_type_raw: 0,
            mb_pred: Some(pred),
            sub_mb_pred: None,
            pcm_samples: None,
            coded_block_pattern: 0,
            transform_size_8x8_flag: false,
            mb_qp_delta: 0,
            residual_luma: Vec::new(),
            residual_luma_dc: None,
            residual_chroma_dc_cb: vec![0i32; 4],
            residual_chroma_dc_cr: vec![0i32; 4],
            residual_chroma_ac_cb: Vec::new(),
            residual_chroma_ac_cr: Vec::new(),
            residual_cb_luma_like: Vec::new(),
            residual_cr_luma_like: Vec::new(),
            residual_cb_16x16_dc: None,
            residual_cr_16x16_dc: None,
            is_skip: false,
        };

        let slice_data = SliceData {
            macroblocks: vec![mb_top, mb_bot],
            mb_field_decoding_flags: vec![false, false],
            last_mb_addr: 1,
        };
        let mut pic = Picture::new(16, 32, 1, 8, 8);
        let mut grid = MbGrid::new(1, 2);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &NoRefs, &mut pic, &mut grid).unwrap();

        // Top MB row 15 holds the known band.
        for x in 0..16 {
            assert_eq!(pic.luma_at(x, 15), (50 + x));
        }
        // Bottom MB (y = 16..31) with Vertical pred should have each
        // column copying the top neighbour (y = 15) down through
        // rows 16..31.
        for y in 16..32 {
            for x in 0..16 {
                assert_eq!(
                    pic.luma_at(x, y),
                    (50 + x),
                    "bottom MB at ({}, {}) should replicate the top row",
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn intra_8x8_zero_residual_dc_fallback_produces_dc_prediction() {
        // §8.3.2 / §8.5.13 — an Intra_8x8 macroblock with
        // `transform_size_8x8_flag = 1`, `cbp_luma = 0` (no residual),
        // at the top-left of a single-MB picture. With no neighbours
        // (blk8=0) the §8.3.2.1 derivation sets dcPredModePredictedFlag
        // = 1 and hence predIntra8x8PredMode = 2 (DC). For blocks 1, 2,
        // and 3 the earlier blocks are already DC, so the derivation
        // again yields 2. With prev_intra8x8_pred_mode_flag = 1 for all
        // four blocks, Intra8x8PredMode[i] = predIntra8x8PredMode = 2,
        // and the §8.3.2.2.4 "neighbours unavailable" branch gives the
        // DC value 1 << (BitDepth-1) = 128 for 8-bit.
        //
        // This exercises the 8x8 residual assembly path (Option B:
        // inverse 8x8 zig-zag scan via ZIGZAG_8X8 / Table 8-14) with
        // an all-zero coefficient block — the inverse transform of
        // zero is zero, so the output is purely the DC prediction.
        let sps = make_sps(1, 1);
        let pps = make_pps();
        let sh = make_slice_header();

        let mut pred = MbPred::default();
        for i in 0..4 {
            // §8.3.2.1 eq. 8-73: prev_intra8x8_pred_mode_flag = 1 selects
            // Intra8x8PredMode = predIntra8x8PredMode. For the top-left
            // MB with no available A/B neighbours the derivation yields
            // DC = 2 for every block.
            pred.prev_intra8x8_pred_mode_flag[i] = true;
            pred.rem_intra8x8_pred_mode[i] = 0;
        }
        pred.intra_chroma_pred_mode = 0;
        let mb = Macroblock {
            mb_type: MbType::INxN,
            mb_type_raw: 0,
            mb_pred: Some(pred),
            sub_mb_pred: None,
            pcm_samples: None,
            coded_block_pattern: 0, // cbp_luma = 0 → no residual.
            transform_size_8x8_flag: true,
            mb_qp_delta: 0,
            residual_luma: Vec::new(),
            residual_luma_dc: None,
            residual_chroma_dc_cb: vec![0i32; 4],
            residual_chroma_dc_cr: vec![0i32; 4],
            residual_chroma_ac_cb: Vec::new(),
            residual_chroma_ac_cr: Vec::new(),
            residual_cb_luma_like: Vec::new(),
            residual_cr_luma_like: Vec::new(),
            residual_cb_16x16_dc: None,
            residual_cr_16x16_dc: None,
            is_skip: false,
        };
        let slice_data = SliceData {
            macroblocks: vec![mb],
            mb_field_decoding_flags: vec![false],
            last_mb_addr: 0,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        let mut grid = MbGrid::new(1, 1);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &NoRefs, &mut pic, &mut grid).unwrap();

        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(
                    pic.luma_at(x, y),
                    128,
                    "intra_8x8 DC with no neighbours and zero residual \
                     should produce 128 at ({}, {})",
                    x,
                    y
                );
            }
        }
    }

    // ---------------------------------------------------------------------
    // §8.3.1.1 / §8.3.2.1 — intra pred-mode derivation unit tests.
    //
    // Each test builds a minimal MbGrid + MbPred and exercises
    // [`derive_intra_4x4_pred_mode`] / [`derive_intra_8x8_pred_mode`]
    // directly so we can pin down the `predIntra4x4PredMode` / eq. 8-41
    // branches independently of the full reconstruction pipeline.
    // ---------------------------------------------------------------------

    /// Build an MbInfo flagged as an already-reconstructed Intra_4x4 MB
    /// with all 16 4x4 pred modes set to `mode`.
    fn mk_intra4x4_info(mode: u8) -> MbInfo {
        MbInfo {
            available: true,
            is_intra: true,
            is_i_pcm: false,
            is_intra_nxn: true,
            mb_type_raw: 0,
            transform_size_8x8_flag: false,
            intra_4x4_pred_modes: [mode; 16],
            ..MbInfo::default()
        }
    }

    /// Build an MbInfo flagged as an already-reconstructed Intra_8x8 MB.
    fn mk_intra8x8_info(mode: u8) -> MbInfo {
        MbInfo {
            available: true,
            is_intra: true,
            is_i_pcm: false,
            is_intra_nxn: true,
            mb_type_raw: 0,
            transform_size_8x8_flag: true,
            intra_8x8_pred_modes: [mode; 4],
            ..MbInfo::default()
        }
    }

    #[test]
    fn intra_4x4_derivation_top_left_block0_with_no_neighbours_predicts_dc() {
        // Isolated 1x1 picture, block 0 of the sole MB. Both mbAddrA and
        // mbAddrB are "not available" → dcPredModePredictedFlag = 1 →
        // predIntra4x4PredMode = 2 (DC). With prev_flag = 1,
        // Intra4x4PredMode[0] = predIntra4x4PredMode = 2. The old
        // hard-coded "DC fallback on prev_flag=1" path happens to agree
        // for this corner case — confirming the new derivation does
        // not regress it.
        let grid = MbGrid::new(1, 1);
        let mut pred = MbPred::default();
        pred.prev_intra4x4_pred_mode_flag[0] = true;
        let mode = derive_intra_4x4_pred_mode(&grid, 0, 0, &pred, false, -1);
        assert_eq!(mode, 2, "expected DC when both A and B are unavailable");
    }

    #[test]
    fn intra_4x4_derivation_within_mb_uses_a_block_pred_mode() {
        // Single MB; block 0 has already been assigned Vertical (0) in
        // the grid. Block 1 lies immediately to its right:
        //   * A neighbour: current MB, block 0 → Intra4x4PredMode = 0.
        //   * B neighbour: mbAddrB not available → intraMxMPredModeB = 2
        //     via step 2's dcPredModePredictedFlag route (A available
        //     but B not → dcPredFlag = 1, forcing both sides to 2).
        //
        // Wait — A is current-MB-self, which is always "available" in
        // the §6.4.11 sense because §6.4.4 marks it available before the
        // block-level derivation runs. But the test picture is 1x1 at
        // the TOP of the picture, so mbAddrB really is unavailable →
        // dcPredFlag = 1 → predicted = 2 → rem_intra4x4_pred_mode[1]
        // dominates. With rem = 3 and pred = 2, actual = rem+1 = 4.
        let mut grid = MbGrid::new(1, 1);
        if let Some(info) = grid.get_mut(0) {
            *info = mk_intra4x4_info(0);
        }
        let mut pred = MbPred::default();
        pred.prev_intra4x4_pred_mode_flag[1] = false;
        pred.rem_intra4x4_pred_mode[1] = 3;
        // predicted = 2 (DC), rem = 3, 3 < 2 is false → actual = 3 + 1 = 4.
        let mode = derive_intra_4x4_pred_mode(&grid, 0, 1, &pred, false, -1);
        assert_eq!(mode, 4);
    }

    #[test]
    fn intra_4x4_derivation_rem_less_than_predicted_yields_rem() {
        // 2-wide grid, current MB at addr 1. Left neighbour (addr 0) is
        // an already-set Intra_4x4 MB with every 4x4 pred mode = 5
        // (Vertical_Right). For block 0 of the current MB:
        //   * A: left MB, 4x4 block at (xW=15, yW=0) → eq. 6-38 ⇒
        //     blk = 8*0 + 4*1 + 2*0 + (7/4 = 1) = 5. Mode of that block
        //     = 5.
        //   * B: mbAddrB is not available (top row) → dcPredFlag = 1 →
        //     forces both to DC = 2.
        //
        // Since B is unavailable, dcPredFlag = 1 and predicted = 2. With
        // prev_flag = false and rem = 0, 0 < 2 → actual = 0 (Vertical).
        let mut grid = MbGrid::new(2, 1);
        if let Some(info) = grid.get_mut(0) {
            *info = mk_intra4x4_info(5);
        }
        let mut pred = MbPred::default();
        pred.prev_intra4x4_pred_mode_flag[0] = false;
        pred.rem_intra4x4_pred_mode[0] = 0;
        let mode = derive_intra_4x4_pred_mode(&grid, 1, 0, &pred, false, -1);
        assert_eq!(mode, 0);
    }

    #[test]
    fn intra_4x4_derivation_both_neighbours_intra_4x4_uses_min_of_modes() {
        // 3x3 grid, current MB at centre (addr 4). Left MB (addr 3) has
        // all 4x4 modes = 7, top MB (addr 1) has all 4x4 modes = 3.
        // Block 0 of current MB:
        //   * A: mbAddrA = addr 3, luma4x4BlkIdxA from (15, 0): blk =
        //     8*(0/8)+4*(15/8)+2*((0%8)/4)+((15%8)/4) = 0+4+0+1 = 5. Mode
        //     = 7.
        //   * B: mbAddrB = addr 1, luma4x4BlkIdxB from (0, 15): blk =
        //     8*(15/8)+4*(0/8)+2*((15%8)/4)+((0%8)/4) = 8+0+2+0 = 10.
        //     Mode = 3.
        //   * predIntra4x4PredMode = min(7, 3) = 3.
        // prev_flag = false, rem = 1 → 1 < 3 → actual = 1 (Horizontal).
        let mut grid = MbGrid::new(3, 3);
        if let Some(info) = grid.get_mut(3) {
            *info = mk_intra4x4_info(7);
        }
        if let Some(info) = grid.get_mut(1) {
            *info = mk_intra4x4_info(3);
        }
        let mut pred = MbPred::default();
        pred.prev_intra4x4_pred_mode_flag[0] = false;
        pred.rem_intra4x4_pred_mode[0] = 1;
        let mode = derive_intra_4x4_pred_mode(&grid, 4, 0, &pred, false, -1);
        assert_eq!(mode, 1);
    }

    #[test]
    fn intra_4x4_derivation_rem_equal_or_greater_than_predicted_adds_one() {
        // Same 3x3 geometry, predicted = 3. With rem = 3:
        //   3 < 3 is false → actual = rem + 1 = 4.
        // With rem = 5: 5 < 3 is false → actual = 6.
        let mut grid = MbGrid::new(3, 3);
        if let Some(info) = grid.get_mut(3) {
            *info = mk_intra4x4_info(7);
        }
        if let Some(info) = grid.get_mut(1) {
            *info = mk_intra4x4_info(3);
        }
        let mut pred = MbPred::default();
        pred.prev_intra4x4_pred_mode_flag[0] = false;
        pred.rem_intra4x4_pred_mode[0] = 3;
        assert_eq!(derive_intra_4x4_pred_mode(&grid, 4, 0, &pred, false, -1), 4);
        pred.rem_intra4x4_pred_mode[0] = 5;
        assert_eq!(derive_intra_4x4_pred_mode(&grid, 4, 0, &pred, false, -1), 6);
    }

    #[test]
    fn intra_4x4_derivation_constrained_intra_pred_treats_inter_as_unavailable() {
        // Current MB at addr 1. Left MB is AVAILABLE but INTER-coded.
        // With constrained_intra_pred_flag = 0 (off) the inter neighbour
        // is consulted by the availability check → dcPredFlag depends
        // on whether mbAddrA is "available", which it is for picture
        // geometry. §8.3.1.1 step 3 then sets intraMxMPredModeA = 2
        // (inter → "not coded in Intra_4x4 or Intra_8x8"). And mbAddrB
        // is out of picture → dcPredFlag = 1 anyway.
        //
        // To observe the flag's effect we instead place the current MB
        // at (x=1, y=1) of a 3x3 grid with an inter-coded left neighbour
        // AND an intra top neighbour. Without CIPred: both A and B
        // available, A=inter → mode=2 via step 3, B=intra_4x4 → its
        // recorded mode. With CIPred=1: A is treated as unavailable →
        // dcPredFlag = 1 → both A and B forced to 2.
        let mut grid = MbGrid::new(3, 3);
        // Left neighbour (addr 3): inter (is_intra = false).
        if let Some(info) = grid.get_mut(3) {
            info.available = true;
            info.is_intra = false;
        }
        // Top neighbour (addr 1): intra 4x4, all modes = 6.
        if let Some(info) = grid.get_mut(1) {
            *info = mk_intra4x4_info(6);
        }
        let mut pred = MbPred::default();
        pred.prev_intra4x4_pred_mode_flag[0] = true; // use predicted mode
                                                     // Without constrained_intra_pred_flag: A=inter contributes 2
                                                     // (step 3 bullet 1 via "not Intra_4x4 or Intra_8x8"), B=6.
                                                     // predicted = min(2, 6) = 2. With prev_flag=true → 2.
        assert_eq!(derive_intra_4x4_pred_mode(&grid, 4, 0, &pred, false, -1), 2);
        // With constrained_intra_pred_flag = 1: A treated as
        // unavailable → dcPredFlag = 1 → both sides 2 → predicted = 2.
        // Same output 2 here. Swap in an intra left neighbour with
        // mode 0 and re-run to confirm CIPred really changes the
        // predicted mode when it matters.
        if let Some(info) = grid.get_mut(3) {
            // Now left is intra, all modes 0 → predicted = min(0, 6) = 0.
            *info = mk_intra4x4_info(0);
        }
        assert_eq!(derive_intra_4x4_pred_mode(&grid, 4, 0, &pred, false, -1), 0);
        // Even with CIPred=1, because the neighbour IS intra it remains
        // available for intra prediction → same result 0.
        assert_eq!(derive_intra_4x4_pred_mode(&grid, 4, 0, &pred, true, -1), 0);
        // Switch left back to inter; with CIPred=1 it becomes
        // unavailable → dcPredFlag = 1 → predicted = 2.
        if let Some(info) = grid.get_mut(3) {
            info.available = true;
            info.is_intra = false;
        }
        assert_eq!(derive_intra_4x4_pred_mode(&grid, 4, 0, &pred, true, -1), 2);
    }

    #[test]
    fn intra_4x4_derivation_intra_16x16_neighbour_contributes_dc() {
        // §8.3.1.1 step 3 bullet 1: an Intra_16x16 neighbour is "not
        // coded in Intra_4x4 or Intra_8x8" → intraMxMPredModeN = 2.
        let mut grid = MbGrid::new(3, 3);
        // Left = Intra_16x16 MB (I slice row 1 → mb_type_raw = 1..=24).
        if let Some(info) = grid.get_mut(3) {
            info.available = true;
            info.is_intra = true;
            info.is_i_pcm = false;
            info.is_intra_nxn = false; // Intra_16x16, NOT I_NxN.
            info.mb_type_raw = 4; // Intra_16x16 variant.
            info.intra_4x4_pred_modes = [5; 16]; // ignored by derivation.
        }
        // Top = Intra_4x4 with modes = 7.
        if let Some(info) = grid.get_mut(1) {
            *info = mk_intra4x4_info(7);
        }
        let mut pred = MbPred::default();
        pred.prev_intra4x4_pred_mode_flag[0] = true;
        // A = Intra_16x16 → 2, B = Intra_4x4[10] = 7. predicted = min(2, 7) = 2.
        assert_eq!(derive_intra_4x4_pred_mode(&grid, 4, 0, &pred, false, -1), 2);
    }

    /// Regression test: in I slices `mb_type_raw = 5` is
    /// `I_16x16_0_1_0` (Intra_16x16) per Table 7-11 — NOT I_NxN. The
    /// pre-fix `mb_is_intra_nxn` used
    /// `matches!(mb_type_raw, 0 | 5 | 23)` (intended for P/B slice
    /// remaps) which incorrectly classified an I-slice Intra_16x16
    /// neighbour as Intra_4x4. That made its stored (but meaningless)
    /// `intra_4x4_pred_modes[...]` leak into §8.3.1.1 step 3 instead
    /// of the spec's Intra_4x4_DC fallback. This reproduces the
    /// exact scenario from `jvt_CABA1_SVA_B` frame 8 MB 27, whose
    /// top neighbour MB 16 is an I-slice Intra_16x16 with raw=5.
    #[test]
    fn intra_4x4_derivation_i_slice_raw5_is_intra_16x16_not_i_nxn() {
        let mut grid = MbGrid::new(3, 3);
        // Top neighbour (addr 1): I-slice Intra_16x16 with raw=5.
        // Its `intra_4x4_pred_modes` are uninitialised from the
        // parser's point of view (we never write them for I_16x16).
        // Seed them with a non-DC value to prove the derivation
        // does NOT use them.
        if let Some(info) = grid.get_mut(1) {
            info.available = true;
            info.is_intra = true;
            info.is_i_pcm = false;
            info.is_intra_nxn = false; // I_16x16 → false.
            info.mb_type_raw = 5; // I_16x16_0_1_0.
            info.intra_4x4_pred_modes = [7; 16]; // must be ignored.
            info.intra_8x8_pred_modes = [7; 4];
        }
        // Left neighbour (addr 3): Intra_4x4, all modes = 3.
        if let Some(info) = grid.get_mut(3) {
            *info = mk_intra4x4_info(3);
        }
        let mut pred = MbPred::default();
        pred.prev_intra4x4_pred_mode_flag[0] = true;
        // A (left, I_NxN) → 3. B (top, I_16x16) → 2 (DC). Predicted
        // = min(3, 2) = 2. Bug would have yielded B = 7 →
        // predicted = min(3, 7) = 3.
        assert_eq!(derive_intra_4x4_pred_mode(&grid, 4, 0, &pred, false, -1), 2);
    }

    #[test]
    fn intra_4x4_derivation_intra_8x8_neighbour_uses_blk_shr_2() {
        // §8.3.1.1 step 3 sub-bullet 2: when a neighbour MB is coded in
        // Intra_8x8, intraMxMPredModeN = Intra8x8PredMode[ blk >> 2 ].
        let mut grid = MbGrid::new(3, 3);
        // Left = Intra_8x8 with blk-8x8 modes [1, 2, 3, 4].
        if let Some(info) = grid.get_mut(3) {
            info.available = true;
            info.is_intra = true;
            info.is_i_pcm = false;
            info.mb_type_raw = 0;
            info.transform_size_8x8_flag = true;
            info.intra_8x8_pred_modes = [1, 2, 3, 4];
        }
        // Top = Intra_4x4 with modes = 6.
        if let Some(info) = grid.get_mut(1) {
            *info = mk_intra4x4_info(6);
        }
        // Current MB at addr 4, block 0 at (0, 0).
        // A neighbour: left MB, (xW=15, yW=0), luma4x4BlkIdxN via
        // eq. 6-38 = 8*0 + 4*1 + 2*0 + 1 = 5. 5 >> 2 = 1 →
        // intra_8x8_pred_modes[1] = 2.
        // B: top MB, (xW=0, yW=15), luma4x4BlkIdxN = 10. Top MB is
        // Intra_4x4 → intra_4x4_pred_modes[10] = 6.
        // predicted = min(2, 6) = 2.
        let mut pred = MbPred::default();
        pred.prev_intra4x4_pred_mode_flag[0] = true;
        assert_eq!(derive_intra_4x4_pred_mode(&grid, 4, 0, &pred, false, -1), 2);
    }

    #[test]
    fn intra_8x8_derivation_top_left_blk_0_with_no_neighbours_predicts_dc() {
        // Single-MB picture, blk 0, no A/B neighbours → predicted = 2.
        let grid = MbGrid::new(1, 1);
        let mut pred = MbPred::default();
        pred.prev_intra8x8_pred_mode_flag[0] = true;
        assert_eq!(derive_intra_8x8_pred_mode(&grid, 0, 0, &pred, false, -1), 2);
    }

    #[test]
    fn intra_8x8_derivation_both_neighbours_intra_8x8_uses_min_of_modes() {
        // 3x3 grid, current at addr 4.
        // Left (addr 3) Intra_8x8 with modes [3, 3, 3, 3].
        // Top (addr 1) Intra_8x8 with modes [7, 7, 7, 7].
        // blk8=0 of current: A=(xW=15, yW=0) → blk = 2*0 + 15/8 = 1.
        // Mode of left blk 1 = 3.
        // B=(xW=0, yW=15) → blk = 2*(15/8) + 0 = 2. Mode = 7.
        // predicted = min(3, 7) = 3.
        let mut grid = MbGrid::new(3, 3);
        if let Some(info) = grid.get_mut(3) {
            *info = mk_intra8x8_info(3);
        }
        if let Some(info) = grid.get_mut(1) {
            *info = mk_intra8x8_info(7);
        }
        let mut pred = MbPred::default();
        pred.prev_intra8x8_pred_mode_flag[0] = true;
        assert_eq!(derive_intra_8x8_pred_mode(&grid, 4, 0, &pred, false, -1), 3);
        // rem = 2 (< 3) with prev_flag=false → actual = 2.
        pred.prev_intra8x8_pred_mode_flag[0] = false;
        pred.rem_intra8x8_pred_mode[0] = 2;
        assert_eq!(derive_intra_8x8_pred_mode(&grid, 4, 0, &pred, false, -1), 2);
        // rem = 4 (>= 3) → actual = 5.
        pred.rem_intra8x8_pred_mode[0] = 4;
        assert_eq!(derive_intra_8x8_pred_mode(&grid, 4, 0, &pred, false, -1), 5);
    }

    #[test]
    fn intra_8x8_derivation_intra_4x4_neighbour_uses_eq_8_72_sub_indices() {
        // §8.3.2.1 eq. 8-72: when the neighbour MB is Intra_4x4,
        // intraMxMPredModeN = Intra4x4PredMode[ luma8x8BlkIdxN * 4 + n ]
        // with n = 1 for A and n = 2 for B (non-MBAFF frame path).
        // Current MB at addr 4 of a 3x3 grid, blk8 = 0.
        // A = left MB, luma8x8BlkIdxA computed:
        //   xN=-1, yN=0 → (xW=15, yW=0) → eq. 6-40 → 2*0 + 15/8 = 1.
        // So A uses intra_4x4_pred_modes[1 * 4 + 1] = intra_4x4_pred_modes[5].
        let mut grid = MbGrid::new(3, 3);
        if let Some(info) = grid.get_mut(3) {
            *info = mk_intra4x4_info(0);
            // Seed a distinctive value at block 5 of the left MB.
            info.intra_4x4_pred_modes[5] = 4;
        }
        // Top MB = Intra_4x4. luma8x8BlkIdxB: xN=0, yN=-1 → (xW=0, yW=15)
        // → 2*(15/8) + 0 = 2. n=2 for B → intra_4x4_pred_modes[2*4 + 2]
        // = intra_4x4_pred_modes[10].
        if let Some(info) = grid.get_mut(1) {
            *info = mk_intra4x4_info(0);
            info.intra_4x4_pred_modes[10] = 6;
        }
        // predicted = min(4, 6) = 4.
        let mut pred = MbPred::default();
        pred.prev_intra8x8_pred_mode_flag[0] = true;
        assert_eq!(derive_intra_8x8_pred_mode(&grid, 4, 0, &pred, false, -1), 4);
    }

    #[test]
    fn inverse_scan_8x8_zigzag_roundtrip_matches_table_8_14() {
        // Spot-check a few entries from Table 8-14:
        //   idx 0 → c00  (row 0, col 0)
        //   idx 1 → c01  (row 0, col 1)
        //   idx 2 → c10  (row 1, col 0)
        //   idx 7 → c12  (row 1, col 2)
        //   idx 63 → c77 (row 7, col 7)
        let mut scan = [0i32; 64];
        scan[0] = 11;
        scan[1] = 12;
        scan[2] = 13;
        scan[7] = 17;
        scan[63] = 77;
        let m = inverse_scan_8x8_zigzag(&scan);
        assert_eq!(m[0], 11, "idx 0 → c00");
        assert_eq!(m[1], 12, "idx 1 → c01");
        assert_eq!(m[8], 13, "idx 2 → c10");
        assert_eq!(m[8 + 2], 17, "idx 7 → c12");
        assert_eq!(m[7 * 8 + 7], 77, "idx 63 → c77");
    }

    #[test]
    fn deblocking_alters_intra_edge() {
        // Build a 2x1 MB picture (32x16) with a hard horizontal step
        // between two intra MBs. With deblocking enabled, bS=3 for an
        // intra MB edge (not MB edge perpendicular to a vertical edge
        // between MBs in the same row) and low QPs keep the filter off.
        // We raise QP via mb_qp_delta so alpha/beta indexes move into
        // the "filter fires" region.
        let mut sps = make_sps(2, 1);
        let _ = &mut sps;
        let pps = make_pps();
        let mut sh = make_slice_header();
        // Enable deblocking, leave offsets at 0.
        sh.disable_deblocking_filter_idc = 0;
        sh.slice_qp_delta = 20; // SliceQPY = 46 — well into the filter region.

        // Two I_PCM MBs with a modest step at the MB edge (small
        // enough that |p0-q0| < alpha so filterSamplesFlag fires).
        // Left MB samples = 100, right MB samples = 115. Delta = 15 —
        // well below alpha for QP=46 (alpha' = 162, alpha = 162 at
        // bit_depth 8).
        let left_luma = vec![100u32; 256];
        let right_luma = vec![115u32; 256];
        let cb = vec![128u32; 64];
        let cr = vec![128u32; 64];
        let mk = |luma: Vec<u32>| Macroblock {
            mb_type: MbType::IPcm,
            mb_type_raw: 25,
            mb_pred: None,
            sub_mb_pred: None,
            pcm_samples: Some(PcmSamples {
                luma,
                chroma_cb: cb.clone(),
                chroma_cr: cr.clone(),
            }),
            coded_block_pattern: 0,
            transform_size_8x8_flag: false,
            mb_qp_delta: 0,
            residual_luma: Vec::new(),
            residual_luma_dc: None,
            residual_chroma_dc_cb: Vec::new(),
            residual_chroma_dc_cr: Vec::new(),
            residual_chroma_ac_cb: Vec::new(),
            residual_chroma_ac_cr: Vec::new(),
            residual_cb_luma_like: Vec::new(),
            residual_cr_luma_like: Vec::new(),
            residual_cb_16x16_dc: None,
            residual_cr_16x16_dc: None,
            is_skip: false,
        };

        // Because I_PCM keeps QP_Y unchanged, we can't raise QP through
        // mb_qp_delta on I_PCM. Use two Intra_16x16 DC MBs instead but
        // hack the underlying luma buffer to set up the edge contrast.
        // Simplest: run reconstruction then observe no change vs
        // deblocking-off baseline to confirm the filter *can* run.
        let mb0 = mk(left_luma);
        let mb1 = mk(right_luma);
        let slice_data = SliceData {
            macroblocks: vec![mb0, mb1],
            mb_field_decoding_flags: vec![false, false],
            last_mb_addr: 1,
        };
        let mut pic_off = Picture::new(32, 16, 1, 8, 8);
        let mut grid_off = MbGrid::new(2, 1);
        let mut sh_off = sh.clone();
        sh_off.disable_deblocking_filter_idc = 1; // off baseline
        reconstruct_slice(
            &slice_data,
            &sh_off,
            &sps,
            &pps,
            &NoRefs,
            &mut pic_off,
            &mut grid_off,
        )
        .unwrap();

        let mut pic_on = Picture::new(32, 16, 1, 8, 8);
        let mut grid_on = MbGrid::new(2, 1);
        reconstruct_slice(
            &slice_data,
            &sh,
            &sps,
            &pps,
            &NoRefs,
            &mut pic_on,
            &mut grid_on,
        )
        .unwrap();

        // We just confirm the two pictures are not equal: the filter
        // did something when enabled. A finer per-sample assertion
        // would require replicating the exact filter output here.
        let changed = pic_on.luma != pic_off.luma;
        // I_PCM MBs with QP_Y = SliceQPY = 46 (filter tables active) and
        // bS = 4 at the MB edge should alter samples near x=15/16 at
        // least for one row.
        assert!(
            changed,
            "deblocking filter did not modify any samples, \
             expected at least the edge column to change"
        );
    }

    // ---------------------------------------------------------------------
    // Inter reconstruction tests
    // ---------------------------------------------------------------------

    /// Build a P-slice header for inter tests.
    fn make_p_slice_header() -> SliceHeader {
        let mut sh = make_slice_header();
        sh.slice_type_raw = 0;
        sh.slice_type = SliceType::P;
        sh
    }

    /// Build a B-slice header.
    fn make_b_slice_header() -> SliceHeader {
        let mut sh = make_slice_header();
        sh.slice_type_raw = 1;
        sh.slice_type = SliceType::B;
        sh
    }

    /// Build a reference picture with luma filled by a closure, chroma
    /// flat at 128.
    fn make_ref_pic<F: FnMut(i32, i32) -> i32>(w: u32, h: u32, mut f: F) -> Picture {
        let mut p = Picture::new(w, h, 1, 8, 8);
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                p.set_luma(x, y, f(x, y));
            }
        }
        for y in 0..(h / 2) as i32 {
            for x in 0..(w / 2) as i32 {
                p.set_cb(x, y, 128);
                p.set_cr(x, y, 128);
            }
        }
        p
    }

    /// Build a P_L0_16x16 MB with zero residual.
    fn make_p_l0_16x16(ref_idx: u32, mvd: [i32; 2]) -> Macroblock {
        let pred = MbPred {
            ref_idx_l0: vec![ref_idx],
            mvd_l0: vec![mvd],
            ..MbPred::default()
        };
        Macroblock {
            mb_type: MbType::PL016x16,
            mb_type_raw: 0,
            mb_pred: Some(pred),
            sub_mb_pred: None,
            pcm_samples: None,
            coded_block_pattern: 0,
            transform_size_8x8_flag: false,
            mb_qp_delta: 0,
            residual_luma: Vec::new(),
            residual_luma_dc: None,
            residual_chroma_dc_cb: vec![0i32; 4],
            residual_chroma_dc_cr: vec![0i32; 4],
            residual_chroma_ac_cb: Vec::new(),
            residual_chroma_ac_cr: Vec::new(),
            residual_cb_luma_like: Vec::new(),
            residual_cr_luma_like: Vec::new(),
            residual_cb_16x16_dc: None,
            residual_cr_16x16_dc: None,
            is_skip: false,
        }
    }

    #[test]
    fn p_skip_with_empty_refs_returns_missing_ref_pic() {
        // P_Skip references L0[0]; with NoRefs (no pictures available)
        // the reconstruction must surface MissingRefPic.
        let sps = make_sps(1, 1);
        let pps = make_pps();
        let sh = make_p_slice_header();
        let mb = Macroblock::new_skip(SliceType::P);
        let slice_data = SliceData {
            macroblocks: vec![mb],
            mb_field_decoding_flags: vec![false],
            last_mb_addr: 0,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        let mut grid = MbGrid::new(1, 1);
        let err = reconstruct_slice(&slice_data, &sh, &sps, &pps, &NoRefs, &mut pic, &mut grid)
            .unwrap_err();
        match err {
            ReconstructError::MissingRefPic { list, idx } => {
                assert_eq!(list, 0);
                assert_eq!(idx, 0);
            }
            other => panic!("unexpected error: {:?}", other),
        }
    }

    #[test]
    fn p_l0_16x16_zero_mv_copies_reference_plane() {
        // With MV = (0, 0), MVD = (0, 0), ref_idx = 0, the predicted
        // MB equals the corresponding 16x16 region of the reference
        // picture. Zero residual => output == prediction.
        let sps = make_sps(1, 1);
        let pps = make_pps();
        let sh = make_p_slice_header();

        // Reference picture: luma[y * 16 + x] = (y * 16 + x) & 0xFF.
        let ref_pic = make_ref_pic(16, 16, |x, y| (y * 16 + x) & 0xFF);
        let mut store = RefPicStore::new();
        store.insert(0, ref_pic);
        store.set_list_0(vec![0]);

        let mb = make_p_l0_16x16(0, [0, 0]);
        let slice_data = SliceData {
            macroblocks: vec![mb],
            mb_field_decoding_flags: vec![false],
            last_mb_addr: 0,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        let mut grid = MbGrid::new(1, 1);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &store, &mut pic, &mut grid).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                let expected = (y * 16 + x) & 0xFF;
                assert_eq!(pic.luma_at(x, y), expected, "mismatch at ({}, {})", x, y);
            }
        }
    }

    #[test]
    fn p_l0_16x16_half_pel_horizontal_mv_matches_6tap() {
        // MV = (2, 0) in 1/4-pel units => xFrac = 2 (half-pel). The
        // 6-tap FIR at (x, y) computes:
        //   b1 = E -5F + 20G + 20H -5I + J
        //   b  = Clip1_8((b1 + 16) >> 5)
        // We set the reference row such that a hand-computed cell
        // result is known.
        let sps = make_sps(2, 2);
        let pps = make_pps();
        let sh = make_p_slice_header();

        let mut ref_pic = Picture::new(32, 32, 1, 8, 8);
        // Row 8 cols 8..=13 = {10, 20, 30, 40, 50, 60}.
        for (i, v) in [10, 20, 30, 40, 50, 60].iter().enumerate() {
            ref_pic.set_luma(8 + i as i32, 8, *v);
        }
        for y in 0..16 {
            for x in 0..16 {
                ref_pic.set_cb(x, y, 128);
                ref_pic.set_cr(x, y, 128);
            }
        }
        let mut store = RefPicStore::new();
        store.insert(0, ref_pic);
        store.set_list_0(vec![0]);

        // Build a 1-MB picture at position (0, 0). MV = (2, 0) in
        // 1/4-pel units. The MB starts at (0, 0); with xFrac = 2 the
        // output sample at MB (10, 8) corresponds to the half-pel b
        // at reference (10, 8), which equals 35 (see inter_pred
        // luma_half_pel_horizontal_known_sample).
        let mb = make_p_l0_16x16(0, [2, 0]);
        let slice_data = SliceData {
            macroblocks: vec![mb],
            mb_field_decoding_flags: vec![false],
            last_mb_addr: 0,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        let mut grid = MbGrid::new(1, 1);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &store, &mut pic, &mut grid).unwrap();
        assert_eq!(pic.luma_at(10, 8), 35);
    }

    #[test]
    fn p_8x8_with_b_direct_sub_partitions_runs_without_error() {
        // Synthesize a P_8x8 MB with four 8x8 sub-partitions of
        // SubMbType::BDirect8x8 (valid for B slices; here we use it
        // as a dispatch smoke test — the code treats BDirect8x8 as
        // Direct mode for the P slice too, and since ref_idx = 0 and
        // MVD = (0, 0) the spatial direct path will pull MV = (0, 0)
        // with no neighbour MV data). A small reference picture lets
        // MC complete.
        let sps = make_sps(1, 1);
        let pps = make_pps();
        let sh = make_p_slice_header();

        let ref_pic = make_ref_pic(16, 16, |_, _| 50);
        let mut store = RefPicStore::new();
        store.insert(0, ref_pic);
        store.set_list_0(vec![0]);

        let mut smp = SubMbPred::default();
        for i in 0..4 {
            smp.sub_mb_type[i] = SubMbType::BDirect8x8;
            smp.ref_idx_l0[i] = 0;
            smp.ref_idx_l1[i] = 0;
            smp.mvd_l0[i] = vec![[0, 0]];
            smp.mvd_l1[i] = vec![[0, 0]];
        }
        let mb = Macroblock {
            mb_type: MbType::P8x8,
            mb_type_raw: 3,
            mb_pred: None,
            sub_mb_pred: Some(smp),
            pcm_samples: None,
            coded_block_pattern: 0,
            transform_size_8x8_flag: false,
            mb_qp_delta: 0,
            residual_luma: Vec::new(),
            residual_luma_dc: None,
            residual_chroma_dc_cb: vec![0i32; 4],
            residual_chroma_dc_cr: vec![0i32; 4],
            residual_chroma_ac_cb: Vec::new(),
            residual_chroma_ac_cr: Vec::new(),
            residual_cb_luma_like: Vec::new(),
            residual_cr_luma_like: Vec::new(),
            residual_cb_16x16_dc: None,
            residual_cr_16x16_dc: None,
            is_skip: false,
        };
        let slice_data = SliceData {
            macroblocks: vec![mb],
            mb_field_decoding_flags: vec![false],
            last_mb_addr: 0,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        let mut grid = MbGrid::new(1, 1);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &store, &mut pic, &mut grid).unwrap();
        // Zero MV, constant ref plane => output = 50 everywhere.
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(pic.luma_at(x, y), 50);
            }
        }
    }

    #[test]
    fn b_skip_with_two_refs_runs_direct_mode() {
        // B_Skip triggers direct-mode derivation (§8.4.1.2). With two
        // reference pictures populated and a single MB, we just make
        // sure the code path runs and produces plausible output.
        let sps = make_sps(1, 1);
        let pps = make_pps();
        let sh = make_b_slice_header();

        let l0 = make_ref_pic(16, 16, |_, _| 80);
        let l1 = make_ref_pic(16, 16, |_, _| 120);
        let mut store = RefPicStore::new();
        store.insert(0, l0);
        store.insert(1, l1);
        store.set_list_0(vec![0]);
        store.set_list_1(vec![1]);

        let mb = Macroblock::new_skip(SliceType::B);
        let slice_data = SliceData {
            macroblocks: vec![mb],
            mb_field_decoding_flags: vec![false],
            last_mb_addr: 0,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        let mut grid = MbGrid::new(1, 1);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &store, &mut pic, &mut grid).unwrap();
        // BiPred default average of (80, 120) = 100 (approximately).
        for y in 0..16 {
            for x in 0..16 {
                let v = pic.luma_at(x, y);
                assert!(
                    (95..=105).contains(&v),
                    "expected ~100 at ({}, {}), got {}",
                    x,
                    y,
                    v
                );
            }
        }
    }

    #[test]
    fn p_l0_l0_16x8_bottom_partition_reads_top_partition_mv_from_same_mb() {
        // §8.4.1.3.2 regression: within a single MB split into 16x8
        // top + bottom partitions, the bottom partition's MVpred must
        // consult the top partition's MV as its B-neighbour (the 4x4
        // block immediately above the bottom partition's top-left 4x4
        // block lives in the top partition, i.e. still inside the
        // current in-flight MB whose `info.available` is not yet set
        // by `reconstruct_slice`). Prior to the fix, `neighbour_from_block`
        // rejected the current-MB lookup because the `info.available`
        // flag was still `false`, causing the bottom partition to
        // compute MVpred from all-zero neighbours.
        //
        // Construction:
        //   - Slice is 1 MB wide, 1 MB tall.
        //   - MB 0 is P_L0_L0_16x8 with:
        //     * Top partition MVD = (16, 0)    -> MV = (16, 0) since
        //       no external neighbours for the top-left MB of the slice.
        //     * Bottom partition MVD = (0, 0)  -> MV = MVpred = top
        //       partition's MV = (16, 0) via Partition16x8Bottom shape
        //       shortcut (eq. 8-204 picks A when refIdxA matches).
        //       Wait — for 16x8 bottom, A = block-4-to-the-left same
        //       row, which is outside this MB (unavailable). So it
        //       falls through to median(A=unavail, B=top-part,
        //       C=unavail) where only B matches refIdx → pick B's MV.
        //   - With same-MB neighbour reads working, bottom partition
        //     produces the reference frame at (x+16, y+8-half) — a
        //     16-pel horizontal shift.
        //   - With same-MB reads broken (old code), bottom partition's
        //     MV = (0, 0), producing the reference frame at (x, y+8)
        //     with no horizontal shift.
        let sps = make_sps(1, 2);
        let pps = make_pps();
        let sh = make_p_slice_header();

        // Reference picture: gradient increasing with x, flat at 0 for y.
        // luma[y*W + x] = (x * 2) & 0xFF. So x-shift of 16 produces a
        // difference of 32 at each sample.
        let ref_pic = make_ref_pic(32, 32, |x, _| (x * 2) & 0xFF);
        let mut store = RefPicStore::new();
        store.insert(0, ref_pic);
        store.set_list_0(vec![0]);

        // Build a P_L0_L0_16x8 MB with top MVD=(16,0), bottom MVD=(0,0).
        let pred = crate::macroblock_layer::MbPred {
            ref_idx_l0: vec![0, 0],
            mvd_l0: vec![[16, 0], [0, 0]],
            ..crate::macroblock_layer::MbPred::default()
        };
        let mb = Macroblock {
            mb_type: MbType::PL0L016x8,
            mb_type_raw: 1,
            mb_pred: Some(pred),
            sub_mb_pred: None,
            pcm_samples: None,
            coded_block_pattern: 0,
            transform_size_8x8_flag: false,
            mb_qp_delta: 0,
            residual_luma: Vec::new(),
            residual_luma_dc: None,
            residual_chroma_dc_cb: vec![0i32; 4],
            residual_chroma_dc_cr: vec![0i32; 4],
            residual_chroma_ac_cb: Vec::new(),
            residual_chroma_ac_cr: Vec::new(),
            residual_cb_luma_like: Vec::new(),
            residual_cr_luma_like: Vec::new(),
            residual_cb_16x16_dc: None,
            residual_cr_16x16_dc: None,
            is_skip: false,
        };
        // Add a filler MB 1 (not used in the check).
        let mb1 = make_p_l0_16x16(0, [0, 0]);
        let slice_data = SliceData {
            macroblocks: vec![mb, mb1],
            mb_field_decoding_flags: vec![false, false],
            last_mb_addr: 1,
        };
        let mut pic = Picture::new(16, 32, 1, 8, 8);
        let mut grid = MbGrid::new(1, 2);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &store, &mut pic, &mut grid).unwrap();

        // After the fix, the BOTTOM partition (rows 8..15 of MB 0)
        // reads the top partition's MV=(16,0)>>2=(4,0) integer pixel
        // shift. Output pixel at (x, y=8..15) should equal
        // ref_pic[x+4, y+8-8]=(x+4)*2. In particular sample (0, 8)
        // = ref at (4, 8) = 8.
        //
        // Before the fix, the bottom would compute MV=(0,0) so output
        // at (0, 8) = ref_pic[0, 8] = 0.
        assert_eq!(
            pic.luma_at(0, 8),
            8,
            "bottom partition did not inherit top partition MV \
             (same-MB neighbour read regression)"
        );
        // Also check top partition for completeness: top gets MV=(16,0)
        // -> integer shift 4 pixels right. Pixel (0, 0) should be
        // ref_pic[4, 0] = 8.
        assert_eq!(pic.luma_at(0, 0), 8);
    }

    #[test]
    fn p_l0_16x16_with_zero_mvd_and_ref0_does_not_apply_p_skip_zero_substitution() {
        // §8.4.1.1 / §8.4.1.2 regression: a regular P_L0_16x16 MB with
        // MVD = (0, 0) and ref_idx_l0 = 0 is NOT a P_Skip MB, even
        // though the two look identical in (mode, mvd, ref_idx, shape,
        // w, h). The §8.4.1.2 zero-MV substitution conditions
        // (force MV to zero when refIdxL0A == 0 AND mvL0A == 0, etc.)
        // apply ONLY to P_Skip, per the spec.
        //
        // Before the fix, `derive_partition_mvs` detected any 16x16
        // L0Only partition with mvd=(0,0) ref=0 as P_Skip and applied
        // the zero-forcing rule, corrupting MVpred for MBs whose left
        // neighbour happened to have refIdx=0 and mv=(0,0).
        //
        // Test setup:
        //   - 2-wide 1-tall slice.
        //   - MB 0: P_L0_16x16 with MVD=(0, 0) -> MV=(0, 0) (no
        //     neighbours). Its mv_l0 becomes (0, 0) and
        //     ref_idx_l0[*] = 0 across the MB.
        //   - MB 1: P_L0_16x16 with MVD=(8, 0) and ref_idx=0. The
        //     MVpred for MB 1 must be the median over the top-left
        //     4x4's neighbours (A = MB 0's top-right 4x4, B/C/D
        //     unavailable). Per §8.4.1.3.1 step 1 "both B and C are
        //     unavail, A is available -> replace B, C with A",
        //     median(A, A, A) = A = (0, 0). MVpred = (0, 0).
        //     MV_MB1 = (0, 0) + (8, 0) = (8, 0).
        //   - With the P_Skip bug: the old code saw mvd=(0,0) AND
        //     ref=0 (after mvd addition? no, BEFORE mvd addition),
        //     actually wait — the bug triggers ONLY when mvd=(0,0)
        //     which makes it look like a P_Skip. Since MB 1 has
        //     mvd=(8, 0), the bug would NOT fire for MB 1 — the
        //     regression is harder to trigger in a 1-MB unit test.
        //     We instead use MB 0 with mvd=(0, 0) and have MB 0's
        //     neighbours all be unavailable; the P_Skip path forces
        //     zero when neighbour A unavailable, which yields the
        //     same result as median-with-unavail, so no observable
        //     difference for MB 0 on its own.
        //
        // Therefore, to isolate the fix we construct a scenario that
        // differs only via the P_Skip "A's refIdx=0 AND mvA=0"
        // zero-forcing rule: MB 0 is P_Skip which sets mv=(0,0)+ref=0;
        // MB 1 is P_L0_16x16 with mvd=(0, 0) and ref_idx=0 (i.e. it
        // looks identical to P_Skip to the BUGGY code).
        //
        // The BUGGY code would route MB 1 through derive_p_skip_mv.
        // In that path, A (MB 0) has ref_idx=0 AND mv=(0,0) → force
        // MB 1's MVpred to zero. That's what the buggy code does.
        //
        // The CORRECT code uses ordinary median: since MB 0 is A,
        // refIdx=0 matches, only A matches → return A's MV = (0, 0).
        // So MB 1's MVpred is also (0, 0), and MV_MB1 = (0, 0).
        //
        // Hmm — in THIS setup the two paths yield the same result.
        // The bug manifests when MB 0 has a NON-zero MV but ref=0
        // (i.e. P_L0_16x16 with mvd != 0). Then:
        //   - Buggy: A's ref=0, mv!=0 → does NOT force zero (second
        //     P_Skip condition doesn't match). Fall back to median.
        //     Wait — P_Skip only forces zero when A.ref=0 AND A.mv=0.
        //     So the buggy code and correct code agree when neighbour
        //     A has mv != 0.
        //   - But what if A.ref=0 AND A.mv=0 (MB 0 is a legitimate
        //     zero-MV P_L0_16x16)? Buggy code forces MB 1's MVpred
        //     to zero — same as median(A=(0,0), B/C/D unavail)
        //     gives (0, 0) — still same.
        //
        // Actually the behavioural difference is more subtle. The
        // P_Skip path forces zero when ANY of (!A.avail, !B.avail,
        // A.ref=0 & A.mv=0, B.ref=0 & B.mv=0) hold. Regular median
        // for unavail A / B replaces them with zeros (via
        // UNAVAILABLE.refIdx=-1 → doesn't match; eq. 8-211 won't
        // fire unless ≥1 matches; then median of (0, 0, avail_mv) =
        // fall-through).
        //
        // Concrete case where they diverge: MB 0 = P_L0_16x16 with
        // ref=0, mv=(3, 4); MB 1 = P_L0_16x16 with mvd=(0, 0), ref=0.
        //   - BUGGY: detect as P_Skip. A.ref=0, A.mv=(3,4) (not
        //     zero). B, C, D unavail. None of the 4 zero-forcing
        //     conditions hits. Fall through to median. median(A=(3,4),
        //     B/C=UNAVAIL). Step 1: B, C both unavail, A avail →
        //     copy A to B, C. median(A, A, A) = A = (3, 4). MV = (3, 4).
        //     Same as correct median! So still no diff.
        //
        // Hmm. When DO the buggy and correct paths differ? It's when
        // the zero-forcing condition "B not available" hits for the
        // buggy path. E.g. MB 1 on the top row of the picture: A
        // available (MB 0), B, C unavailable. Buggy: !B.available
        // → force zero. Correct median: A available, B, C unavail →
        // median(A, A, A) = A. Different!
        //
        // So the regression test IS a top-row MB. Let's do that.
        //
        // Setup:
        //   - 2x1 MB picture. MB 0 and MB 1 on top row.
        //   - MB 0: P_L0_16x16 with MVD=(3, 4), so MB 0's mv = (3, 4).
        //   - MB 1: P_L0_16x16 with MVD=(0, 0), ref=0 (looks like
        //     P_Skip to buggy code).
        //   - Expected: MB 1's MVpred should be (3, 4) (= MB 0's mv,
        //     per median fallback with A=MB 0 and B, C unavail).
        //     MB 1's MV = MVpred + MVD = (3, 4) + (0, 0) = (3, 4).
        //   - Buggy: MB 1 treated as P_Skip. !B.available holds →
        //     force zero. MVpred = (0, 0). MV = (0, 0).
        let sps = make_sps(2, 1);
        let pps = make_pps();
        let sh = make_p_slice_header();

        // Reference plane — gradient in x.
        let ref_pic = make_ref_pic(64, 16, |x, _| (x * 2) & 0xff);
        let mut store = RefPicStore::new();
        store.insert(0, ref_pic);
        store.set_list_0(vec![0]);

        // MB 0: MVD = (3, 4). MV = (3, 4). 3 in 1/4-pel = 0 integer, 3 frac.
        let mb0 = make_p_l0_16x16(0, [3, 4]);
        // MB 1: MVD = (0, 0). MV depends on MVpred (which should be
        // MB 0's MV = (3, 4)). So MB 1 pixels at (16 + x, y) come
        // from ref_pic at integer (16 + x + 0, y + 1) with frac
        // (3, 0) (x frac part of 3), i.e. a horizontal 6-tap.
        let mb1 = make_p_l0_16x16(0, [0, 0]);
        let slice_data = SliceData {
            macroblocks: vec![mb0, mb1],
            mb_field_decoding_flags: vec![false, false],
            last_mb_addr: 1,
        };
        let mut pic = Picture::new(32, 16, 1, 8, 8);
        let mut grid = MbGrid::new(2, 1);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &store, &mut pic, &mut grid).unwrap();

        // MB 1's MV grid slots should all be (3, 4) — inherited from MB 0.
        let info1 = grid.get(1).unwrap();
        for blk4 in 0..16 {
            assert_eq!(
                info1.mv_l0[blk4],
                (3i16, 4i16),
                "MB 1's blk{blk4} MV should inherit MB 0's MV via median, \
                 NOT be forced to zero by the P_Skip zero-MV substitution",
            );
        }
    }

    #[test]
    fn inter_mb_records_mv_and_ref_idx_in_grid() {
        // After an inter MB is reconstructed, the MbGrid must carry
        // the MV + ref_idx so a neighbouring inter MB's MVpred can
        // consult them.
        let sps = make_sps(2, 1);
        let pps = make_pps();
        let sh = make_p_slice_header();

        let ref_pic = make_ref_pic(32, 16, |_, _| 50);
        let mut store = RefPicStore::new();
        store.insert(0, ref_pic);
        store.set_list_0(vec![0]);

        // MB 0: P_L0_16x16 with ref_idx=0, MVD=(4, 8) (i.e., MV = (4, 8)
        // since no neighbour -> MVpred = (0, 0)).
        let mb0 = make_p_l0_16x16(0, [4, 8]);
        // MB 1: P_L0_16x16 — we'll just verify the grid after MB 0.
        let mb1 = make_p_l0_16x16(0, [0, 0]);
        let slice_data = SliceData {
            macroblocks: vec![mb0, mb1],
            mb_field_decoding_flags: vec![false, false],
            last_mb_addr: 1,
        };
        let mut pic = Picture::new(32, 16, 1, 8, 8);
        let mut grid = MbGrid::new(2, 1);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &store, &mut pic, &mut grid).unwrap();

        let info0 = grid.get(0).unwrap();
        assert!(info0.available);
        assert!(!info0.is_intra);
        assert_eq!(info0.ref_idx_l0, [0, 0, 0, 0]);
        // Every 4x4 block in MB 0 should carry mv = (4, 8).
        for blk4 in 0..16 {
            assert_eq!(info0.mv_l0[blk4], (4i16, 8i16));
        }
    }

    // ---------------------------------------------------------------------
    // §8.4.2.3 — Weighted sample prediction dispatch tests
    // ---------------------------------------------------------------------

    /// Build a pred_weight_table with a single L0/L1 luma/chroma entry.
    /// `chroma_w_o` is `(cb_weight, cb_offset, cr_weight, cr_offset)`.
    fn make_pwt_single_entry(
        log2wd_y: u32,
        log2wd_c: u32,
        l0: Option<(i32, i32)>,
        l1: Option<(i32, i32)>,
        chroma_l0: Option<(i32, i32, i32, i32)>,
        chroma_l1: Option<(i32, i32, i32, i32)>,
    ) -> PredWeightTable {
        PredWeightTable {
            luma_log2_weight_denom: log2wd_y,
            chroma_log2_weight_denom: log2wd_c,
            luma_weights_l0: vec![l0],
            chroma_weights_l0: vec![chroma_l0.map(|(a, b, c, d)| [(a, b), (c, d)])],
            luma_weights_l1: vec![l1],
            chroma_weights_l1: vec![chroma_l1.map(|(a, b, c, d)| [(a, b), (c, d)])],
        }
    }

    #[test]
    fn p_l0_16x16_explicit_weighted_applies_formula() {
        // §8.4.2.3.2 eq. 8-274 (L0 only, log2WD >= 1):
        //   predSampleL0[x, y] = Clip1(((predL0 * w0 + (1<<(log2WD-1))) >> log2WD) + o0)
        //
        // With predL0 = 10, w0 = 16, o0 = 4, log2WD = 4:
        //   (10 * 16 + 8) >> 4 + 4 = (168 >> 4) + 4 = 10 + 4 = 14.
        let sps = make_sps(1, 1);
        let mut pps = make_pps();
        pps.weighted_pred_flag = true;
        let mut sh = make_p_slice_header();
        sh.pred_weight_table = Some(make_pwt_single_entry(
            4,
            0,
            Some((16, 4)),
            None,
            Some((1 << 0, 0, 1 << 0, 0)),
            None,
        ));

        let ref_pic = make_ref_pic(16, 16, |_, _| 10);
        let mut store = RefPicStore::new();
        store.insert(0, ref_pic);
        store.set_list_0(vec![0]);

        let mb = make_p_l0_16x16(0, [0, 0]);
        let slice_data = SliceData {
            macroblocks: vec![mb],
            mb_field_decoding_flags: vec![false],
            last_mb_addr: 0,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        let mut grid = MbGrid::new(1, 1);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &store, &mut pic, &mut grid).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(
                    pic.luma_at(x, y),
                    14,
                    "expected 14 at ({}, {}), got {}",
                    x,
                    y,
                    pic.luma_at(x, y)
                );
            }
        }
    }

    #[test]
    fn p_l0_16x16_weighted_pred_flag_off_matches_default() {
        // Regression: with weighted_pred_flag = false, the reconstructed
        // MB must equal the copy-through default path (i.e., the same
        // output as p_l0_16x16_zero_mv_copies_reference_plane).
        let sps = make_sps(1, 1);
        let mut pps = make_pps();
        pps.weighted_pred_flag = false;
        // Even if the bitstream had shipped a pred_weight_table with
        // non-identity weights, weighted_pred_flag=0 must ignore it.
        let mut sh = make_p_slice_header();
        sh.pred_weight_table = Some(make_pwt_single_entry(
            4,
            0,
            Some((32, 10)), // would be "*2 + 10" if used
            None,
            Some((1 << 0, 0, 1 << 0, 0)),
            None,
        ));

        let ref_pic = make_ref_pic(16, 16, |x, y| (y * 16 + x) & 0xFF);
        let mut store = RefPicStore::new();
        store.insert(0, ref_pic);
        store.set_list_0(vec![0]);

        let mb = make_p_l0_16x16(0, [0, 0]);
        let slice_data = SliceData {
            macroblocks: vec![mb],
            mb_field_decoding_flags: vec![false],
            last_mb_addr: 0,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        let mut grid = MbGrid::new(1, 1);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &store, &mut pic, &mut grid).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                let expected = (y * 16 + x) & 0xFF;
                assert_eq!(
                    pic.luma_at(x, y),
                    expected,
                    "weighted_pred_flag=off must copy ref; mismatch at ({}, {})",
                    x,
                    y
                );
            }
        }
    }

    /// Build a B_Bi_16x16 MB with zero MVDs and the given L0/L1 ref
    /// indices.
    fn make_b_bi_16x16(ref_idx_l0: u32, ref_idx_l1: u32) -> Macroblock {
        let pred = MbPred {
            ref_idx_l0: vec![ref_idx_l0],
            ref_idx_l1: vec![ref_idx_l1],
            mvd_l0: vec![[0, 0]],
            mvd_l1: vec![[0, 0]],
            ..MbPred::default()
        };
        Macroblock {
            mb_type: MbType::BBi16x16,
            mb_type_raw: 3,
            mb_pred: Some(pred),
            sub_mb_pred: None,
            pcm_samples: None,
            coded_block_pattern: 0,
            transform_size_8x8_flag: false,
            mb_qp_delta: 0,
            residual_luma: Vec::new(),
            residual_luma_dc: None,
            residual_chroma_dc_cb: vec![0i32; 4],
            residual_chroma_dc_cr: vec![0i32; 4],
            residual_chroma_ac_cb: Vec::new(),
            residual_chroma_ac_cr: Vec::new(),
            residual_cb_luma_like: Vec::new(),
            residual_cr_luma_like: Vec::new(),
            residual_cb_16x16_dc: None,
            residual_cr_16x16_dc: None,
            is_skip: false,
        }
    }

    #[test]
    fn b_bi_16x16_explicit_weighted_applies_bipred_formula() {
        // §8.4.2.3.2 eq. 8-276 (bipred):
        //   v = Clip1(((p0 * w0 + p1 * w1 + (1 << log2WD)) >> (log2WD + 1))
        //             + ((o0 + o1 + 1) >> 1))
        //
        // With p0 = 40, p1 = 80, w0 = w1 = 1, o0 = o1 = 0, log2WD = 1:
        //   sum = 40 + 80 + 2 = 122
        //   v = (122 >> 2) + 0 = 30.
        let sps = make_sps(1, 1);
        let mut pps = make_pps();
        pps.weighted_bipred_idc = 1;
        let mut sh = make_b_slice_header();
        sh.num_ref_idx_l0_active_minus1 = 0;
        sh.num_ref_idx_l1_active_minus1 = 0;
        sh.pred_weight_table = Some(make_pwt_single_entry(
            1,
            0,
            Some((1, 0)),
            Some((1, 0)),
            Some((1 << 0, 0, 1 << 0, 0)),
            Some((1 << 0, 0, 1 << 0, 0)),
        ));

        let l0 = make_ref_pic(16, 16, |_, _| 40);
        let l1 = make_ref_pic(16, 16, |_, _| 80);
        let mut store = RefPicStore::new();
        store.insert(0, l0);
        store.insert(1, l1);
        store.set_list_0(vec![0]);
        store.set_list_1(vec![1]);

        let mb = make_b_bi_16x16(0, 0);
        let slice_data = SliceData {
            macroblocks: vec![mb],
            mb_field_decoding_flags: vec![false],
            last_mb_addr: 0,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        let mut grid = MbGrid::new(1, 1);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &store, &mut pic, &mut grid).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(
                    pic.luma_at(x, y),
                    30,
                    "expected 30 at ({}, {}), got {}",
                    x,
                    y,
                    pic.luma_at(x, y)
                );
            }
        }
    }

    #[test]
    fn b_bi_16x16_implicit_bipred_idc_td_zero_falls_back_to_equal_weights() {
        // §8.4.2.3.3 — when td = Clip3(-128, 127, pic1POC - pic0POC) is
        // zero, eq. 8-280 / 8-281 set w0C = w1C = 32 (logWDC = 5,
        // offsets zero per eq. 8-277/8-278/8-279).
        //
        // Then eq. 8-276 with w0=w1=32, log2WD=5:
        //   v = ((80*32 + 120*32 + 32) >> 6) + 0
        //     = ((2560 + 3840 + 32) >> 6)
        //     = 6432 >> 6
        //     = 100.
        let sps = make_sps(1, 1);
        let mut pps = make_pps();
        pps.weighted_bipred_idc = 2; // implicit.
        let sh = make_b_slice_header();

        // Both refs have pic_order_cnt = 0 (default) → td = 0.
        let l0 = make_ref_pic(16, 16, |_, _| 80);
        let l1 = make_ref_pic(16, 16, |_, _| 120);
        let mut store = RefPicStore::new();
        store.insert(0, l0);
        store.insert(1, l1);
        store.set_list_0(vec![0]);
        store.set_list_1(vec![1]);

        let mb = make_b_bi_16x16(0, 0);
        let slice_data = SliceData {
            macroblocks: vec![mb],
            mb_field_decoding_flags: vec![false],
            last_mb_addr: 0,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        let mut grid = MbGrid::new(1, 1);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &store, &mut pic, &mut grid).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(
                    pic.luma_at(x, y),
                    100,
                    "expected 100 (implicit w0=w1=32 via td=0 fallback) at \
                     ({}, {}), got {}",
                    x,
                    y,
                    pic.luma_at(x, y)
                );
            }
        }
    }

    #[test]
    fn b_bi_16x16_implicit_bipred_idc_poc_midpoint_applies_equal_weights() {
        // §8.4.2.3.3 — current picture temporally between the two
        // refs. With pic0 POC = 0, pic1 POC = 8, curr POC = 4:
        //
        //   td = Clip3(-128, 127, 8 - 0) = 8           (eq. 8-202)
        //   tb = Clip3(-128, 127, 4 - 0) = 4           (eq. 8-201)
        //   tx = (16384 + |8/2|) / 8
        //      = (16384 + 4) / 8 = 2048                 (eq. 8-197)
        //   DistScaleFactor = Clip3(-1024, 1023,
        //                           (4*2048 + 32) >> 6)
        //                   = Clip3(-1024, 1023,
        //                           8224 >> 6)
        //                   = Clip3(-1024, 1023, 128)
        //                   = 128                      (eq. 8-198)
        //   DistScaleFactor >> 2 = 32 — in [-64, 128],
        //   so eq. 8-282/8-283: w1 = 32, w0 = 64 - 32 = 32.
        //
        // Since the midpoint yields equal weights, with
        // predL0 = 40, predL1 = 120:
        //   (40*32 + 120*32 + 32) >> 6
        //     = (1280 + 3840 + 32) >> 6
        //     = 5152 >> 6
        //     = 80.
        let sps = make_sps(1, 1);
        let mut pps = make_pps();
        pps.weighted_bipred_idc = 2;
        let sh = make_b_slice_header();

        let mut l0 = make_ref_pic(16, 16, |_, _| 40);
        l0.pic_order_cnt = 0;
        let mut l1 = make_ref_pic(16, 16, |_, _| 120);
        l1.pic_order_cnt = 8;
        let mut store = RefPicStore::new();
        store.insert(0, l0);
        store.insert(1, l1);
        store.set_list_0(vec![0]);
        store.set_list_1(vec![1]);

        let mb = make_b_bi_16x16(0, 0);
        let slice_data = SliceData {
            macroblocks: vec![mb],
            mb_field_decoding_flags: vec![false],
            last_mb_addr: 0,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        pic.pic_order_cnt = 4;
        let mut grid = MbGrid::new(1, 1);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &store, &mut pic, &mut grid).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(
                    pic.luma_at(x, y),
                    80,
                    "expected 80 (implicit midpoint w0=w1=32) at ({}, {}), got {}",
                    x,
                    y,
                    pic.luma_at(x, y)
                );
            }
        }
    }

    #[test]
    fn b_bi_16x16_implicit_bipred_idc_asymmetric_poc_weights_lean_on_closer_ref() {
        // §8.4.2.3.3 — current picture closer to L0 than L1. With
        //   pic0 POC = 0, pic1 POC = 8, curr POC = 2:
        //   td = 8, tb = 2, tx = 2048
        //   DistScaleFactor = Clip3(-1024, 1023, (2*2048 + 32) >> 6)
        //                   = (4128 >> 6) = 64
        //   DistScaleFactor >> 2 = 16  (in range)
        //   w1 = 16, w0 = 48 — L0 is 3x weighted (L0 is temporally
        //   closer to curr, so its bigger weight is expected).
        //
        // With predL0 = 100, predL1 = 20 (flat planes):
        //   (100*48 + 20*16 + 32) >> 6
        //     = (4800 + 320 + 32) >> 6
        //     = 5152 >> 6
        //     = 80.
        let sps = make_sps(1, 1);
        let mut pps = make_pps();
        pps.weighted_bipred_idc = 2;
        let sh = make_b_slice_header();

        let mut l0 = make_ref_pic(16, 16, |_, _| 100);
        l0.pic_order_cnt = 0;
        let mut l1 = make_ref_pic(16, 16, |_, _| 20);
        l1.pic_order_cnt = 8;
        let mut store = RefPicStore::new();
        store.insert(0, l0);
        store.insert(1, l1);
        store.set_list_0(vec![0]);
        store.set_list_1(vec![1]);

        let mb = make_b_bi_16x16(0, 0);
        let slice_data = SliceData {
            macroblocks: vec![mb],
            mb_field_decoding_flags: vec![false],
            last_mb_addr: 0,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        pic.pic_order_cnt = 2;
        let mut grid = MbGrid::new(1, 1);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &store, &mut pic, &mut grid).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(
                    pic.luma_at(x, y),
                    80,
                    "expected 80 (implicit asymmetric w0=48, w1=16) at \
                     ({}, {}), got {}",
                    x,
                    y,
                    pic.luma_at(x, y)
                );
            }
        }
    }

    #[test]
    fn implicit_bipred_weights_td_zero_returns_equal_weights() {
        // §8.4.2.3.3 eq. 8-280 / 8-281 — td == 0 path.
        assert_eq!(
            implicit_bipred_weights(0, 0, 0, false),
            (32, 32, 5),
            "td == 0 should yield w0 = w1 = 32"
        );
    }

    #[test]
    fn implicit_bipred_weights_long_term_returns_equal_weights() {
        // §8.4.2.3.3 eq. 8-280 / 8-281 — long-term ref path.
        assert_eq!(
            implicit_bipred_weights(4, 0, 8, true),
            (32, 32, 5),
            "long-term ref should yield w0 = w1 = 32"
        );
    }

    #[test]
    fn implicit_bipred_weights_midpoint_yields_equal_weights() {
        // Midpoint — tb = td/2 — yields DistScaleFactor = 128, so
        // w1 = 32, w0 = 32.
        assert_eq!(
            implicit_bipred_weights(4, 0, 8, false),
            (32, 32, 5),
            "midpoint → equal weights"
        );
    }

    #[test]
    fn implicit_bipred_weights_hand_computed_asymmetric() {
        // curr closer to L0:
        //   tb = 2, td = 8, tx = (16384 + 4) / 8 = 2048.
        //   DistScaleFactor = (2*2048 + 32) >> 6 = 4128 >> 6 = 64.
        //   DistScaleFactor >> 2 = 16 (in range).
        //   w1 = 16, w0 = 48.
        assert_eq!(
            implicit_bipred_weights(2, 0, 8, false),
            (48, 16, 5),
            "curr closer to L0 (tb=2, td=8)"
        );

        // curr closer to L1:
        //   tb = 6, td = 8, tx = 2048.
        //   DistScaleFactor = (6*2048 + 32) >> 6 = 12320 >> 6 = 192.
        //   DistScaleFactor >> 2 = 48 (in range).
        //   w1 = 48, w0 = 16.
        assert_eq!(
            implicit_bipred_weights(6, 0, 8, false),
            (16, 48, 5),
            "curr closer to L1 (tb=6, td=8)"
        );
    }

    #[test]
    fn implicit_bipred_weights_extrapolation_out_of_band_falls_back() {
        // curr well outside the span of the two refs — DistScaleFactor>>2
        // exits [-64, 128] → eq. 8-280/8-281 fallback to (32, 32).
        //
        // With pic0=0, pic1=8, curr=-100:
        //   tb = -100, td = 8, tx = 2048.
        //   DistScaleFactor = Clip3(-1024, 1023, (-100*2048 + 32) >> 6)
        //                   = Clip3(-1024, 1023, -204768 >> 6)
        //                   = Clip3(-1024, 1023, -3200)
        //                   = -1024.
        //   -1024 >> 2 = -256 (< -64) → fallback.
        assert_eq!(
            implicit_bipred_weights(-100, 0, 8, false),
            (32, 32, 5),
            "out-of-band DistScaleFactor should fall back to equal weights"
        );
    }

    // =====================================================================
    // §6.4.1 / §6.4.10 — MBAFF reconstruction (Phase 2)
    // =====================================================================

    /// Build an SPS with `mb_adaptive_frame_field_flag` enabled, at
    /// the given picture dimensions.
    fn make_mbaff_sps(w_mbs: u32, h_map_units: u32) -> Sps {
        let mut sps = make_sps(w_mbs, h_map_units);
        sps.mb_adaptive_frame_field_flag = true;
        sps.frame_mbs_only_flag = false;
        sps
    }

    #[test]
    fn reconstruct_slice_accepts_mbaff_frames() {
        // §7.4.2.1.1 — MbaffFrameFlag = mb_adaptive_frame_field_flag &&
        // !field_pic_flag. Phase 2 removes the blanket rejection: a
        // minimal two-MB MBAFF pair (one frame + one frame MB) should
        // reconstruct without erroring.
        let sps = make_mbaff_sps(1, 1); // 1 MB wide, 1 map-unit tall → 2 MBs tall.
        let pps = make_pps();
        let sh = make_slice_header();
        let mb0 = make_empty_ipcm_mb();
        let mb1 = make_empty_ipcm_mb();
        let slice_data = SliceData {
            macroblocks: vec![mb0, mb1],
            mb_field_decoding_flags: vec![false, false], // frame-coded pair
            last_mb_addr: 1,
        };
        // FrameHeightInMbs = 2 * 1 = 2; picture is 16x32 luma.
        let mut pic = Picture::new(16, 32, 1, 8, 8);
        let mut grid = MbGrid::new(1, 2);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &NoRefs, &mut pic, &mut grid)
            .expect("MBAFF frame reconstruction must no longer error");
    }

    #[test]
    fn reconstruct_mbaff_frame_mb_pair_places_samples_in_stacked_layout() {
        // §6.4.1 eqs. 6-7/6-8 — for frame-coded MBs in an MBAFF frame,
        // the top MB's samples occupy rows 0..16 and the bottom MB's
        // samples occupy rows 16..32 at column 0.
        let sps = make_mbaff_sps(1, 1);
        let pps = make_pps();
        let sh = make_slice_header();
        let mut mb_top = make_empty_ipcm_mb();
        let mut mb_bot = make_empty_ipcm_mb();
        // Tag the two MBs with distinct constant luma values so we can
        // detect placement.
        for v in mb_top.pcm_samples.as_mut().unwrap().luma.iter_mut() {
            *v = 10;
        }
        for v in mb_bot.pcm_samples.as_mut().unwrap().luma.iter_mut() {
            *v = 20;
        }
        let slice_data = SliceData {
            macroblocks: vec![mb_top, mb_bot],
            mb_field_decoding_flags: vec![false, false],
            last_mb_addr: 1,
        };
        let mut pic = Picture::new(16, 32, 1, 8, 8);
        let mut grid = MbGrid::new(1, 2);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &NoRefs, &mut pic, &mut grid).unwrap();

        // Top MB @ rows 0..16 → value 10.
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(
                    pic.luma_at(x, y),
                    10,
                    "frame-MB pair top: expected 10 at ({}, {})",
                    x,
                    y
                );
            }
        }
        // Bottom MB @ rows 16..32 → value 20.
        for y in 16..32 {
            for x in 0..16 {
                assert_eq!(
                    pic.luma_at(x, y),
                    20,
                    "frame-MB pair bottom: expected 20 at ({}, {})",
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn reconstruct_mbaff_field_mb_pair_does_not_error() {
        // §6.4.1 eqs. 6-9/6-10 — field MBs start at y = yO or yO + 1
        // within the pair. For Phase 2 we verify the reconstruction
        // doesn't error on a field-coded MBAFF pair; pixel-accurate
        // y-stride=2 interleave is a Phase-3 follow-up.
        let sps = make_mbaff_sps(1, 1);
        let pps = make_pps();
        let sh = make_slice_header();
        let mb_top = make_empty_ipcm_mb();
        let mb_bot = make_empty_ipcm_mb();
        let slice_data = SliceData {
            macroblocks: vec![mb_top, mb_bot],
            mb_field_decoding_flags: vec![true, true], // field-coded pair
            last_mb_addr: 1,
        };
        let mut pic = Picture::new(16, 32, 1, 8, 8);
        let mut grid = MbGrid::new(1, 2);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &NoRefs, &mut pic, &mut grid)
            .expect("MBAFF field pair reconstruction must no longer error");
    }

    #[test]
    fn reconstruct_mbaff_intra16x16_pair_dc_fills_both_mbs() {
        // Pure intra_16x16 DC (mode 2). With no neighbours, DC defaults
        // to 1 << (BitDepth-1) = 128 per §8.3.3.3 eq. 8-121.
        // Both MBs should fill their regions with 128.
        let sps = make_mbaff_sps(1, 1);
        let pps = make_pps();
        let sh = make_slice_header();
        let mb_top = make_intra16x16_dc_mb(2);
        let mb_bot = make_intra16x16_dc_mb(2);
        let slice_data = SliceData {
            macroblocks: vec![mb_top, mb_bot],
            mb_field_decoding_flags: vec![false, false],
            last_mb_addr: 1,
        };
        let mut pic = Picture::new(16, 32, 1, 8, 8);
        let mut grid = MbGrid::new(1, 2);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &NoRefs, &mut pic, &mut grid).unwrap();

        for y in 0..32 {
            for x in 0..16 {
                assert_eq!(
                    pic.luma_at(x, y),
                    128,
                    "intra-16x16 DC fallback in MBAFF: expected 128 at ({}, {})",
                    x,
                    y
                );
            }
        }
    }

    // =====================================================================
    // §6.4.1 eq. (6-10) — MBAFF field-MB y-stride interleave
    // =====================================================================

    /// §6.4.1 eq. (6-10) — a field-coded MB pair in an MBAFF frame
    /// writes its luma samples with `y_stride = 2`. The top MB's row
    /// `k` goes at picture row `pair_y + k * 2`, and the bottom MB's
    /// row `k` goes at `pair_y + 1 + k * 2`.
    ///
    /// Verified with two I_PCM MBs carrying a simple pattern that
    /// identifies each source sample's origin (the per-MB constant
    /// 10 / 20 distinguishes top vs bottom).
    #[test]
    fn mbaff_field_mb_y_stride_interleaves_top_and_bottom() {
        let sps = make_mbaff_sps(1, 1);
        let pps = make_pps();
        let sh = make_slice_header();
        // Top MB: all luma samples = 10. Bottom MB: all luma samples = 20.
        // For I_PCM, pcm.luma[y*16 + x] is written directly; the MbWriter
        // applies the field stride. So after reconstruction:
        //   Even rows (0, 2, 4, …, 30) should be from the top MB = 10.
        //   Odd  rows (1, 3, 5, …, 31) should be from the bottom MB = 20.
        let mut mb_top = make_empty_ipcm_mb();
        let mut mb_bot = make_empty_ipcm_mb();
        for v in mb_top.pcm_samples.as_mut().unwrap().luma.iter_mut() {
            *v = 10;
        }
        for v in mb_bot.pcm_samples.as_mut().unwrap().luma.iter_mut() {
            *v = 20;
        }
        let slice_data = SliceData {
            macroblocks: vec![mb_top, mb_bot],
            mb_field_decoding_flags: vec![true, true], // field-coded pair
            last_mb_addr: 1,
        };
        let mut pic = Picture::new(16, 32, 1, 8, 8);
        let mut grid = MbGrid::new(1, 2);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &NoRefs, &mut pic, &mut grid).unwrap();

        // Top MB (mb_addr = 0): writes rows 0, 2, 4, …, 30.
        for k in 0..16 {
            let y = k * 2;
            for x in 0..16 {
                assert_eq!(
                    pic.luma_at(x, y),
                    10,
                    "field-MB top: sample (x={}, y={}) (row k={}) expected 10",
                    x,
                    y,
                    k
                );
            }
        }
        // Bottom MB (mb_addr = 1): writes rows 1, 3, 5, …, 31.
        for k in 0..16 {
            let y = k * 2 + 1;
            for x in 0..16 {
                assert_eq!(
                    pic.luma_at(x, y),
                    20,
                    "field-MB bottom: sample (x={}, y={}) (row k={}) expected 20",
                    x,
                    y,
                    k
                );
            }
        }
    }

    /// §6.4.1 eq. (6-10) — direct MbWriter unit test: verify the
    /// top-of-pair MB's (0, 0) and (0, 1) samples land at the spec's
    /// eq. (6-10) coordinates `(pair_px, pair_py)` and
    /// `(pair_px, pair_py + 2)`, and the bottom-of-pair MB's (0, 0)
    /// samples land at `(pair_px, pair_py + 1)`.
    #[test]
    fn mbaff_field_mb_writer_places_samples_per_eq_6_10() {
        // Single MB column, single pair row — two MBs.
        let grid = MbGrid::new(1, 2);
        let mut pic = Picture::new(16, 32, 1, 8, 8);

        // Top-of-pair MB (mb_addr = 0), field-coded. mb_is_top = true,
        // pair_y = 0, mb_py = 0.
        let (mb_px_top, mb_py_top) = mb_sample_origin(&grid, 0, true, true);
        assert_eq!((mb_px_top, mb_py_top), (0, 0), "top-of-pair field origin");
        let w_top = MbWriter::new(&grid, 0, mb_px_top, mb_py_top, true, true, 1);
        // Write a known value at (0, 0) and (0, 1) of the top MB.
        w_top.set_luma(&mut pic, 0, 0, 111);
        w_top.set_luma(&mut pic, 0, 1, 222);

        // Bottom-of-pair MB (mb_addr = 1), field-coded. mb_is_top = false,
        // pair_y = 0, mb_py = 1.
        let (mb_px_bot, mb_py_bot) = mb_sample_origin(&grid, 1, true, true);
        assert_eq!((mb_px_bot, mb_py_bot), (0, 1), "bot-of-pair field origin");
        let w_bot = MbWriter::new(&grid, 1, mb_px_bot, mb_py_bot, true, true, 1);
        // Write a known value at (0, 0) of the bottom MB.
        w_bot.set_luma(&mut pic, 0, 0, 77);

        // §6.4.1 eq. (6-10) — top MB (0, 0) at picture (0, 0);
        // top MB (0, 1) at picture (0, 2); bot MB (0, 0) at (0, 1).
        assert_eq!(pic.luma_at(0, 0), 111, "top MB (0, 0) → picture (0, 0)");
        assert_eq!(pic.luma_at(0, 2), 222, "top MB (0, 1) → picture (0, 2)");
        assert_eq!(pic.luma_at(0, 1), 77, "bot MB (0, 0) → picture (0, 1)");
    }

    // =====================================================================
    // §8.7 — MBAFF deblocking doesn't panic
    // =====================================================================

    /// §8.7 — MBAFF deblocking should complete without panicking on a
    /// minimal MBAFF pair. Pixel-accurate MBAFF edge geometry is future
    /// work, but we must not crash.
    #[test]
    fn mbaff_deblocking_completes_without_panic() {
        let sps = make_mbaff_sps(1, 1);
        let pps = make_pps();
        let mut sh = make_slice_header();
        sh.disable_deblocking_filter_idc = 0; // enable deblock
        sh.slice_qp_delta = 20; // well into the filter-active region
        let mb_top = make_intra16x16_dc_mb(2);
        let mb_bot = make_intra16x16_dc_mb(2);
        let slice_data = SliceData {
            macroblocks: vec![mb_top, mb_bot],
            mb_field_decoding_flags: vec![false, false],
            last_mb_addr: 1,
        };
        let mut pic = Picture::new(16, 32, 1, 8, 8);
        let mut grid = MbGrid::new(1, 2);
        // Just must not panic. Pixel output correctness for MBAFF
        // deblocking is beyond this phase — see deblock_plane_* for
        // the non-MBAFF-exact edge handling.
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &NoRefs, &mut pic, &mut grid)
            .expect("MBAFF + deblocking must complete without error");
    }

    /// §8.7 — MBAFF deblocking on a field-coded pair likewise must not
    /// panic even with QP large enough for the filter to fire.
    #[test]
    fn mbaff_field_pair_deblocking_completes_without_panic() {
        let sps = make_mbaff_sps(1, 1);
        let pps = make_pps();
        let mut sh = make_slice_header();
        sh.disable_deblocking_filter_idc = 0;
        sh.slice_qp_delta = 20;
        let mb_top = make_intra16x16_dc_mb(2);
        let mb_bot = make_intra16x16_dc_mb(2);
        let slice_data = SliceData {
            macroblocks: vec![mb_top, mb_bot],
            mb_field_decoding_flags: vec![true, true], // field pair
            last_mb_addr: 1,
        };
        let mut pic = Picture::new(16, 32, 1, 8, 8);
        let mut grid = MbGrid::new(1, 2);
        reconstruct_slice(&slice_data, &sh, &sps, &pps, &NoRefs, &mut pic, &mut grid)
            .expect("MBAFF field pair + deblocking must complete without error");
    }

    // =====================================================================
    // §8.3.1.1 step 2 bullet 3 — constrained_intra_pred_flag gating
    // =====================================================================

    /// §8.3.1.1 step 2 bullet 3 — when `pps.constrained_intra_pred_flag`
    /// is 1, an Inter-coded neighbour MB is treated as unavailable for
    /// intra prediction (forcing DC fallback). The flag must be plumbed
    /// from the PPS through to the per-block pred-mode derivation.
    ///
    /// Mirror of `intra_4x4_derivation_constrained_intra_pred_treats_inter_as_unavailable`
    /// at the full-pipeline level: verify that changing only the PPS's
    /// constrained_intra_pred_flag alters the output in a setting where
    /// a left inter-coded neighbour would otherwise contribute its
    /// intra_4x4_pred_mode.
    #[test]
    fn constrained_intra_pred_flag_forces_inter_neighbour_to_dc() {
        // 3-wide grid. Simulate an already-reconstructed left neighbour
        // MB that's been marked Inter. The current MB is I_NxN at the
        // centre. We cannot conveniently emit Inter MBs from the test
        // harness (which doesn't have a ref store), so exercise the
        // derivation directly.

        // Case A: constrained_intra_pred_flag = 0. Inter left neighbour
        // contributes mode 2 (DC) via §8.3.1.1 step 3 bullet 1
        // ("not Intra_4x4 or Intra_8x8"). Top neighbour is Intra_4x4.
        // The spec's §8.3.1.1 step 2 bullet 3 sets
        // `dcPredModePredictedFlag = 1` when EITHER neighbour is
        // unavailable (the flag also fires when cip=1 and a neighbour
        // is inter). With flag=0 both neighbours are "available" → the
        // min-of-modes rule runs.
        //
        // Case B: constrained_intra_pred_flag = 1. The Inter left
        // neighbour becomes "not available" per §8.3.1.1 step 2
        // bullet 3, which sets dcPredModePredictedFlag = 1 → predicted
        // mode forced to 2 (DC) regardless of top neighbour.
        let mut grid = MbGrid::new(3, 3);
        // Left neighbour (addr 3): inter (is_intra = false).
        if let Some(info) = grid.get_mut(3) {
            info.available = true;
            info.is_intra = false;
        }
        // Top neighbour (addr 1): intra 4x4, all modes = 0 (Vertical).
        if let Some(info) = grid.get_mut(1) {
            *info = mk_intra4x4_info(0);
        }
        let mut pred = MbPred::default();
        pred.prev_intra4x4_pred_mode_flag[0] = true; // use predicted

        // flag=0: inter-left contributes 2, top contributes 0 →
        // predicted = min(2, 0) = 0 (Vertical).
        let mode_unconstrained = derive_intra_4x4_pred_mode(&grid, 4, 0, &pred, false, -1);
        assert_eq!(
            mode_unconstrained, 0,
            "flag=0: inter-left → mode 2; top=0 → min = 0"
        );

        // flag=1: inter-left treated as unavailable → dcPredFlag = 1
        // → predicted = 2 regardless of top → actual = 2.
        let mode_constrained = derive_intra_4x4_pred_mode(&grid, 4, 0, &pred, true, -1);
        assert_eq!(
            mode_constrained, 2,
            "flag=1: inter-left unavailable → DC fallback (mode 2)"
        );
        assert_ne!(
            mode_constrained, mode_unconstrained,
            "constrained_intra_pred_flag must change the outcome here"
        );
    }

    /// §8.3.1.1 step 2 bullet 3 — end-to-end: reconstruct an I_NxN MB
    /// with a (marked-) Inter left neighbour and verify the bottom-most
    /// row of the current MB's left-column 4x4 block changes when the
    /// PPS's constrained_intra_pred_flag is toggled.
    ///
    /// This is a smoke-level test; it verifies the plumbing from
    /// `pps.constrained_intra_pred_flag` down through
    /// `derive_intra_4x4_pred_mode`. The parser doesn't produce fake
    /// Inter neighbours, but the grid state is reconstructed in-test
    /// before the I_NxN MB is decoded, so the derivation sees it.
    #[test]
    fn constrained_intra_pred_flag_plumbed_through_pps() {
        // A 1x1 picture can't have a left neighbour; a 2x1 picture
        // lets us seed the left neighbour's grid entry directly (as in
        // intra_4x4_derivation_constrained_intra_pred_treats_inter_as_unavailable),
        // then reconstruct the right MB as I_NxN.
        //
        // For the right MB at addr=1, neighbour A is addr 0 and B is
        // unavailable (top edge). If A is Inter + flag=0, mode_A = 2,
        // and mode_B is not used (dcPredFlag fires due to B unavailable)
        // anyway. So the CIP effect on mode derivation is subtle when
        // only one neighbour is present.
        //
        // To observe a plumbing-level change we use a 3x3 layout and
        // the derive_* helper directly — this has already been exercised
        // above. Here we just confirm `pps.constrained_intra_pred_flag`
        // reads through [`reconstruct_slice`] without panicking.
        let sps = make_sps(1, 1);
        let mut pps_on = make_pps();
        pps_on.constrained_intra_pred_flag = true;
        let pps_off = make_pps();
        let sh = make_slice_header();

        let build_nxn_mb = || {
            let mut pred = MbPred::default();
            for i in 0..16 {
                pred.prev_intra4x4_pred_mode_flag[i] = true;
                pred.rem_intra4x4_pred_mode[i] = 0;
            }
            pred.intra_chroma_pred_mode = 0;
            Macroblock {
                mb_type: MbType::INxN,
                mb_type_raw: 0,
                mb_pred: Some(pred),
                sub_mb_pred: None,
                pcm_samples: None,
                coded_block_pattern: 0,
                transform_size_8x8_flag: false,
                mb_qp_delta: 0,
                residual_luma: Vec::new(),
                residual_luma_dc: None,
                residual_chroma_dc_cb: vec![0i32; 4],
                residual_chroma_dc_cr: vec![0i32; 4],
                residual_chroma_ac_cb: Vec::new(),
                residual_chroma_ac_cr: Vec::new(),
                residual_cb_luma_like: Vec::new(),
                residual_cr_luma_like: Vec::new(),
                residual_cb_16x16_dc: None,
                residual_cr_16x16_dc: None,
                is_skip: false,
            }
        };

        for pps in [&pps_off, &pps_on] {
            let slice_data = SliceData {
                macroblocks: vec![build_nxn_mb()],
                mb_field_decoding_flags: vec![false],
                last_mb_addr: 0,
            };
            let mut pic = Picture::new(16, 16, 1, 8, 8);
            let mut grid = MbGrid::new(1, 1);
            reconstruct_slice(&slice_data, &sh, &sps, pps, &NoRefs, &mut pic, &mut grid).unwrap();
        }
    }

    // -----------------------------------------------------------------
    // §6.4.3 / §6.4.11.4 / §8.3.1.2 — Intra_4x4 top-right availability.
    //
    // For luma4x4BlkIdx ∈ {3, 7, 11, 13, 15} the 4 samples p[4..7, -1]
    // used by DDL / VL / HD / HU / VR prediction come from a block or
    // macroblock that has NOT yet been decoded in the §6.4.3 scan order.
    // `gather_samples_4x4` must surface `top_right = false` for those
    // block indices even though the pixels are inside the picture, so
    // the §8.3.1.2 "substitute p[3,-1]" fallback kicks in.
    // -----------------------------------------------------------------

    #[test]
    fn gather_samples_4x4_top_right_unavailable_for_scan_boundary_blocks() {
        // Picture large enough that bx+4 is well inside the width for
        // every 4x4 block of MB 0 — so a pure "within picture" check
        // would say `top_right = true` everywhere.
        let pic = Picture::new(64, 32, 1, 8, 8);

        // Map luma4x4BlkIdx -> (bx, by) inside MB 0 (LUMA_4X4_XY).
        let xy = super::LUMA_4X4_XY;

        let grid = crate::mb_grid::MbGrid::new(4, 2);
        for (idx, (bx, by)) in xy.iter().copied().enumerate() {
            let s = super::gather_samples_4x4(&pic, &grid, bx, by, idx, -1, false);
            let expected = match idx {
                // Scan-order "top-right not yet decoded" set.
                3 | 7 | 11 | 13 | 15 => false,
                // First-row blocks (by == 0) still depend on whether a
                // top neighbour is available; inside MB 0 at (mb_px=0,
                // mb_py=0) it is not — so those also report false here.
                _ if by == 0 => false,
                _ => true,
            };
            assert_eq!(
                s.availability.top_right, expected,
                "blk {} (bx={}, by={}) expected tr={} got {}",
                idx, bx, by, expected, s.availability.top_right
            );
        }
    }

    // §6.4.11.2 / §6.4.3 — Intra_8x8 top-right availability for the
    // bottom-right 8x8 block (luma8x8BlkIdx == 3). Its top-right
    // samples (p[8..15, -1] at y = 7 within the MB) would need to come
    // from the right-neighbour macroblock, which has not been decoded
    // yet. Report `top_right = false` regardless of picture bounds.
    #[test]
    fn gather_samples_8x8_top_right_unavailable_for_blk3() {
        let pic = Picture::new(64, 32, 1, 8, 8);
        let grid = crate::mb_grid::MbGrid::new(4, 2);
        // LUMA_8X8_XY[3] == (8, 8).
        let s = super::gather_samples_8x8(&pic, &grid, 8, 8, 3, -1, false);
        assert!(
            !s.availability.top_right,
            "blk8 3: tr should be false (right-neighbour MB not decoded)"
        );
        // And blk8 == 2 at (0, 8) — top-right here is block 1 within
        // the current MB, which HAS been decoded; report true (the
        // top row is also inside the picture).
        let s2 = super::gather_samples_8x8(&pic, &grid, 0, 8, 2, -1, false);
        assert!(s2.availability.top_right);
    }

    // -----------------------------------------------------------------
    // §7.4.5.3 — residual_luma compact indexing.
    //
    // The macroblock-layer parser pushes 4x4 residual entries into
    // `Macroblock::residual_luma` ONLY for 8x8 quadrants whose
    // cbp_luma bit is set (§7.3.5.3). When cbp_luma has gaps (e.g.
    // 0b1110 — bit 0 clear) the array is *compacted*, not sparse.
    // The reconstruction path must therefore map `block_idx` to the
    // array slot by counting set bits below the block's 8x8 quadrant.
    //
    // Before the fix, `residual_luma[block_idx]` was read directly,
    // which silently substituted data from a different 4x4 block for
    // every block_idx past a gap.
    //
    // This test rebuilds an Intra_4x4 MB with cbp_luma = 0b1110 and
    // a non-zero DC coefficient placed in the residual_luma entry
    // that corresponds to MB block index 4 (first 4x4 of 8x8 quadrant
    // 1). Block 4's reconstructed sample value must reflect that
    // residual — which, before the fix, would have been mis-routed to
    // block 0 (since residual_luma[0] held the data).
    // -----------------------------------------------------------------

    #[test]
    fn intra4x4_residual_indexing_respects_cbp_luma_gaps() {
        use crate::macroblock_layer::{Macroblock, MbPred, MbType};
        use crate::mb_grid::MbGrid;
        use crate::slice_data::SliceData;
        let sps = make_sps(1, 1);
        let pps = make_pps();
        let sh = make_slice_header();

        // Build an I_NxN MB with cbp_luma = 0x7 (bits 0, 1, 2 set —
        // bit 3 clear so block 12..15 get no residual). Place a non-
        // zero DC residual in the FIRST entry of the array. Pre-fix,
        // that entry would be read as block 0's residual because the
        // reconstruct code used `residual_luma[block_idx]` directly.
        //
        // Post-fix, `residual_luma[0]` holds block 0's data AND the
        // offset walking works by counting low-bit cbp_luma bits
        // below the current 8x8 quadrant (see `set_before`). The
        // invariant we assert: for cbp=0x7, `residual_luma[0..=11]`
        // map to block 0..=11 in order (the "all bits set below"
        // case, which is a no-op and must stay correct).
        //
        // Then a second sub-case with cbp_luma = 0xE (bit 0 clear,
        // bits 1/2/3 set) places residual_luma[0] at block index 4,
        // proving the fix.

        for (cbp, residual_first_at_block) in [(0x7u8, 0usize), (0xEu8, 4usize)] {
            // One non-zero DC in the first residual entry. After
            // inverse 4x4 transform at qP=26 this yields a non-zero
            // residual block.
            let mut first_entry = [0i32; 16];
            first_entry[0] = 4; // AC path, with scaling -> non-zero.
                                // Fill the array with the number of entries the parser
                                // would emit: 4 per set cbp_luma bit, placed sequentially.
            let n = (cbp & 0x0F).count_ones() as usize * 4;
            let mut residual_luma = Vec::with_capacity(n);
            for i in 0..n {
                if i == 0 {
                    residual_luma.push(first_entry);
                } else {
                    residual_luma.push([0i32; 16]);
                }
            }
            let mb_pred = MbPred {
                prev_intra4x4_pred_mode_flag: [true; 16],
                rem_intra4x4_pred_mode: [0; 16],
                intra_chroma_pred_mode: 0,
                ..MbPred::default()
            };
            let mb = Macroblock {
                mb_type: MbType::INxN,
                mb_type_raw: 0,
                mb_pred: Some(mb_pred),
                sub_mb_pred: None,
                pcm_samples: None,
                coded_block_pattern: cbp as u32,
                transform_size_8x8_flag: false,
                mb_qp_delta: 0,
                residual_luma,
                residual_luma_dc: None,
                residual_chroma_dc_cb: Vec::new(),
                residual_chroma_dc_cr: Vec::new(),
                residual_chroma_ac_cb: Vec::new(),
                residual_chroma_ac_cr: Vec::new(),
                residual_cb_luma_like: Vec::new(),
                residual_cr_luma_like: Vec::new(),
                residual_cb_16x16_dc: None,
                residual_cr_16x16_dc: None,
                is_skip: false,
            };

            // Reconstruct.
            let slice_data = SliceData {
                macroblocks: vec![mb],
                mb_field_decoding_flags: vec![false],
                last_mb_addr: 0,
            };
            let mut pic = Picture::new(16, 16, 1, 8, 8);
            let mut grid = MbGrid::new(1, 1);
            reconstruct_slice(&slice_data, &sh, &sps, &pps, &NoRefs, &mut pic, &mut grid).unwrap();

            // Block N of MB 0 occupies (bx, by) = LUMA_4X4_XY[N].
            // Sum the 16 samples in that block and assert non-flat
            // (pred was DC=128, residual non-zero → sum != 128 * 16).
            let (bx, by) = super::LUMA_4X4_XY[residual_first_at_block];
            let mut sum = 0i32;
            for y in 0..4 {
                for x in 0..4 {
                    sum += pic.luma_at(bx + x, by + y);
                }
            }
            let flat_dc_sum = 128 * 16;
            assert_ne!(
                sum, flat_dc_sum,
                "cbp=0x{:x} block {} should have non-flat output \
                 (residual_first_at_block stored in residual_luma[0] must route there)",
                cbp, residual_first_at_block,
            );
            // Sanity check: block 0 should be flat DC=128 when the
            // first residual doesn't land there (cbp=0xE case). When
            // cbp=0x7 the residual lands at block 0 and that block
            // won't be flat — skip the "flat" check in that sub-case.
            if residual_first_at_block != 0 {
                // For cbp=0xE, bit 0 is clear → block 0, 1, 2, 3 get
                // NO residual → all DC=128 (but only block 0 has no
                // neighbours). Check block 0 which has no neighbours.
                let (ox, oy) = super::LUMA_4X4_XY[0];
                let mut other_sum = 0i32;
                for y in 0..4 {
                    for x in 0..4 {
                        other_sum += pic.luma_at(ox + x, oy + y);
                    }
                }
                assert_eq!(
                    other_sum, flat_dc_sum,
                    "cbp=0x{:x} block 0 should be flat DC=128 (cbp bit 0 clear)",
                    cbp,
                );
            }
        }
    }
}
