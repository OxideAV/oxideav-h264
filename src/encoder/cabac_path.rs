//! Round-30 — H.264 **CABAC encode path** for IDR + P slices.
//!
//! Self-contained CABAC entry points layered on top of the round-1..29
//! CAVLC encoder primitives (motion estimation, intra prediction,
//! transform, quantisation, deblock). The CAVLC bitstream emitter is
//! replaced by the CABAC arithmetic encoder + per-syntax helpers in
//! [`crate::encoder::cabac_engine`] / [`crate::encoder::cabac_syntax`].
//!
//! Scope (per H.264 §9.3 + Tables 9-12..9-26):
//!
//! * **PPS**: `entropy_coding_mode_flag = 1`. `weighted_pred_flag = 0`,
//!   `weighted_bipred_idc = 0`, `deblocking_filter_control_present = 1`.
//! * **Profile**: Main (77) — Baseline (66) forbids CABAC per §A.2.1.
//! * **Slice header**: writes `cabac_init_idc = 0` for P slices
//!   (sufficient for the round-30 acceptance fixture; idcs 1..2 give
//!   different ctxIdx init seeds but the engine works the same way).
//! * **Slice data**: §7.3.4 CABAC alignment to byte boundary, then per
//!   §7.3.5 macroblock_layer:
//!   - I-slice: every MB is `I_16x16` with all four §8.3.3 luma modes
//!     trialled by SAD on predictor + chroma DC/V/H/Plane.
//!   - P-slice: per-MB pick between `P_Skip` and `P_L0_16x16` with
//!     integer-pel ME ±16. Residual is full luma 16 4x4 + chroma DC + AC.
//! * **End-of-slice**: `end_of_slice_flag` via `EncodeTerminate(1)`.
//!
//! Out of scope for round-30: B slices, I_NxN under CABAC, half-pel /
//! quarter-pel ME, intra fallback in P (the existing CAVLC path's
//! round-29 RDO is decoupled from this entry point), 4:2:2 / 4:4:4.

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::unnecessary_cast)]

use crate::cabac_ctx::{BlockType, CabacContexts, MvdComponent, NeighbourCtx, SliceKind};
use crate::encoder::bitstream::BitWriter;
use crate::encoder::cabac_engine::CabacEncoder;
use crate::encoder::cabac_syntax::{
    encode_coded_block_pattern, encode_end_of_slice_flag, encode_intra_chroma_pred_mode,
    encode_mb_qp_delta, encode_mb_skip_flag, encode_mb_type_b, encode_mb_type_i, encode_mb_type_p,
    encode_mvd_lx, encode_residual_block_cabac, encode_sub_mb_type_b,
};
use crate::encoder::deblock::{
    chroma_nz_mask_from_blocks, deblock_recon, luma_nz_mask_from_blocks, MbDeblockInfo,
};
use crate::encoder::intra_pred::{
    predict_16x16, predict_chroma_8x8, sad_8x8, I16x16Mode, IntraChromaMode,
};
use crate::encoder::macroblock::{b_16x8_mb_type_raw, b_8x16_mb_type_raw, BPartPred, BSubMbCell};
use crate::encoder::me::{search_quarter_pel_16x16, search_quarter_pel_8x8};
use crate::encoder::nal::build_nal_unit;
use crate::encoder::pps::{build_baseline_pps_rbsp, BaselinePpsConfig};
use crate::encoder::slice::{
    write_b_slice_header, write_idr_i_slice_header, write_p_slice_header, BSliceHeaderConfig,
    CabacSliceParams, IdrSliceHeaderConfig, PSliceHeaderConfig,
};
use crate::encoder::sps::{build_baseline_sps_rbsp, BaselineSpsConfig};
use crate::encoder::transform::{
    forward_core_4x4, forward_hadamard_2x2, forward_hadamard_4x4, quantize_4x4_ac,
    quantize_chroma_dc, quantize_luma_dc, zigzag_scan_4x4, zigzag_scan_4x4_ac,
};
use crate::encoder::{
    best_partition_mode_b, build_b_explicit_4x4_cell_chroma, build_b_explicit_8x8_cell_luma,
    build_b_partition_pred_chroma_16x8, build_b_partition_pred_chroma_8x16,
    build_b_partition_pred_luma_16x8, build_b_partition_pred_luma_8x16,
    build_inter_pred_chroma_4x4, build_inter_pred_luma_8x8, partition_sad_b_16x8,
    partition_sad_b_8x16, pick_best_b8x8_cell, EncodedB, EncodedFrameRef, EncodedIdr, EncodedP,
    Encoder, FrameRefPartitionMv, PartitionShape, YuvFrame,
};
use crate::inter_pred::{interpolate_chroma, interpolate_luma};
use crate::mv_deriv::{
    derive_b_spatial_direct_with_d, derive_mvpred_with_d, derive_p_skip_mv_with_d, Mv,
    MvpredInputs, MvpredShape, NeighbourMv,
};

// `build_inter_pred_luma` and `build_inter_pred_chroma` are private to the
// `crate::encoder` module's `mod.rs`; we replicate them locally here.
use crate::nal::NalUnitType;
use crate::transform::{
    inverse_hadamard_chroma_dc_420, inverse_hadamard_luma_dc_16x16,
    inverse_transform_4x4_dc_preserved, qp_bd_offset, qp_y_to_qp_c_with_bd_offset, FLAT_4X4_16,
};

/// §6.4.3 / Figure 6-10 — luma 4x4 block scan (block index → (bx, by)).
const LUMA_4X4_BLK: [(usize, usize); 16] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (3, 0),
    (2, 1),
    (3, 1),
    (0, 2),
    (1, 2),
    (0, 3),
    (1, 3),
    (2, 2),
    (3, 2),
    (2, 3),
    (3, 3),
];

// ---------------------------------------------------------------------------
// CABAC neighbour state for the encoder side.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CabacEncMbInfo {
    available: bool,
    is_intra: bool,
    is_skip: bool,
    is_i_pcm: bool,
    is_i_nxn: bool,
    /// §9.3.3.1.1.3 — true when this MB's mb_type ∈ {B_Skip, B_Direct_16x16}.
    /// Used by the B-slice mb_type ctxIdxOffset=27 bin-0 condTerm.
    is_b_skip_or_direct: bool,
    cbp_luma: u8,
    cbp_chroma: u8,
    intra_chroma_pred_mode: u8,
    /// Per-luma-4x4-block coded_block_flag.
    cbf_luma_4x4: [bool; 16],
    cbf_luma_dc: bool,
    cbf_luma_ac: [bool; 16],
    cbf_cb_dc: bool,
    cbf_cr_dc: bool,
    cbf_cb_ac: [bool; 4],
    cbf_cr_ac: [bool; 4],
    /// 4:4:4 chroma AC CBF tracking — 16 4x4 blocks per plane (§7.3.5.3).
    cbf_cb_ac_444: [bool; 16],
    cbf_cr_ac_444: [bool; 16],
    /// Per-4x4 |mvd_x| / |mvd_y| (round-30 P MBs use one MV per MB).
    abs_mvd_l0_x: [u32; 16],
    abs_mvd_l0_y: [u32; 16],
    /// L1 mvd magnitudes — only populated on B-slice MBs.
    abs_mvd_l1_x: [u32; 16],
    abs_mvd_l1_y: [u32; 16],
    /// Round-32 — per-8x8 MV / refIdx tracking for §8.4.1.3 mvp
    /// derivation in P + B CABAC paths. Round-30/31 forced MVs to (0, 0)
    /// so this was unused; the round-32 real-ME wiring populates one
    /// 16x16 MV (replicated across all 4 8x8 partitions) per MB. The
    /// per-8x8 layout matches `MvGridSlot::mv_l0_8x8` so future per-
    /// partition encoding (16x8 / 8x16 / 8x8) can plug into the same
    /// structure. `ref_idx_l0_8x8 < 0` means "this list unused"
    /// (intra MB or B mb_type that doesn't use L0).
    ref_idx_l0_8x8: [i32; 4],
    mv_l0_8x8: [Mv; 4],
    ref_idx_l1_8x8: [i32; 4],
    mv_l1_8x8: [Mv; 4],
}

impl Default for CabacEncMbInfo {
    fn default() -> Self {
        Self {
            available: false,
            is_intra: false,
            is_skip: false,
            is_i_pcm: false,
            is_i_nxn: false,
            is_b_skip_or_direct: false,
            cbp_luma: 0,
            cbp_chroma: 0,
            intra_chroma_pred_mode: 0,
            cbf_luma_4x4: [false; 16],
            cbf_luma_dc: false,
            cbf_luma_ac: [false; 16],
            cbf_cb_dc: false,
            cbf_cr_dc: false,
            cbf_cb_ac: [false; 4],
            cbf_cr_ac: [false; 4],
            cbf_cb_ac_444: [false; 16],
            cbf_cr_ac_444: [false; 16],
            abs_mvd_l0_x: [0; 16],
            abs_mvd_l0_y: [0; 16],
            abs_mvd_l1_x: [0; 16],
            abs_mvd_l1_y: [0; 16],
            ref_idx_l0_8x8: [-1; 4],
            mv_l0_8x8: [Mv::ZERO; 4],
            ref_idx_l1_8x8: [-1; 4],
            mv_l1_8x8: [Mv::ZERO; 4],
        }
    }
}

impl CabacEncMbInfo {
    /// Build a [`NeighbourMv`] for this MB, looking at its 8x8 partition
    /// `part`. Returns [`NeighbourMv::UNAVAILABLE`] when the MB hasn't
    /// been encoded yet, or [`NeighbourMv::intra_but_mb_available`] for
    /// intra MBs (so the §8.4.1.3.2 step 2a folding fires).
    fn neighbour_mv_for(&self, part: usize, list: u8) -> NeighbourMv {
        if !self.available {
            return NeighbourMv::UNAVAILABLE;
        }
        if self.is_intra {
            return NeighbourMv::intra_but_mb_available();
        }
        let (ref_idx, mv) = if list == 0 {
            (self.ref_idx_l0_8x8[part], self.mv_l0_8x8[part])
        } else {
            (self.ref_idx_l1_8x8[part], self.mv_l1_8x8[part])
        };
        NeighbourMv {
            available: ref_idx >= 0,
            mb_available: true,
            partition_available: true,
            ref_idx,
            mv,
        }
    }
}

struct CabacEncGrid {
    width_mbs: usize,
    mbs: Vec<CabacEncMbInfo>,
}

impl CabacEncGrid {
    fn new(width_mbs: usize, height_mbs: usize) -> Self {
        Self {
            width_mbs,
            mbs: vec![CabacEncMbInfo::default(); width_mbs * height_mbs],
        }
    }
    fn at(&self, x: usize, y: usize) -> &CabacEncMbInfo {
        &self.mbs[y * self.width_mbs + x]
    }
    fn at_mut(&mut self, x: usize, y: usize) -> &mut CabacEncMbInfo {
        &mut self.mbs[y * self.width_mbs + x]
    }
    fn left(&self, x: usize, y: usize) -> Option<&CabacEncMbInfo> {
        if x == 0 {
            None
        } else {
            Some(self.at(x - 1, y))
        }
    }
    fn above(&self, x: usize, y: usize) -> Option<&CabacEncMbInfo> {
        if y == 0 {
            None
        } else {
            Some(self.at(x, y - 1))
        }
    }
}

// ---------------------------------------------------------------------------
// Round-32 — §8.4.1.3 MV-predictor derivation reading from the CABAC
// encoder grid.
//
// Mirrors the CAVLC path's `neighbour_mvs_16x16` + `mvp_for_16x16` /
// `p_skip_mv` helpers but operates on `CabacEncGrid`, which carries
// per-8x8 MVs / refIdxs in [`CabacEncMbInfo::mv_l0_8x8`] / `..._l1_8x8`.
//
// Per §6.4.11.7 the four neighbours of a 16x16 partition's top-left
// 4x4 block are (A=left, B=above, C=above-right, D=above-left). Each
// neighbour's representative 8x8 partition is the one spatially adjacent
// to the current MB's top-left 4x4 (block 0):
//   * A → left MB's 8x8 #1 (top-right)
//   * B → above MB's 8x8 #2 (bottom-left)
//   * C → above-right MB's 8x8 #2 (bottom-left)
//   * D → above-left MB's 8x8 #3 (bottom-right)
// ---------------------------------------------------------------------------

fn cabac_neighbour_mvs_16x16(
    grid: &CabacEncGrid,
    mb_x: usize,
    mb_y: usize,
    list: u8,
) -> (NeighbourMv, NeighbourMv, NeighbourMv, NeighbourMv) {
    let make = |x: i32, y: i32, part: usize| -> NeighbourMv {
        if x < 0 || y < 0 || (x as usize) >= grid.width_mbs {
            return NeighbourMv::UNAVAILABLE;
        }
        let h = grid.mbs.len() / grid.width_mbs;
        if (y as usize) >= h {
            return NeighbourMv::UNAVAILABLE;
        }
        grid.at(x as usize, y as usize).neighbour_mv_for(part, list)
    };
    let mb_xi = mb_x as i32;
    let mb_yi = mb_y as i32;
    let a = make(mb_xi - 1, mb_yi, 1);
    let b = make(mb_xi, mb_yi - 1, 2);
    let c = make(mb_xi + 1, mb_yi - 1, 2);
    let d = make(mb_xi - 1, mb_yi - 1, 3);
    (a, b, c, d)
}

/// §8.4.1.3 — derive mvpLX for a 16x16 partition reading neighbours
/// from the CABAC encoder grid. `list ∈ {0, 1}`. `ref_idx` is the
/// current MB's refIdxLX (always 0 in our single-ref-per-list setup).
fn cabac_mvp_for_16x16(
    grid: &CabacEncGrid,
    mb_x: usize,
    mb_y: usize,
    list: u8,
    ref_idx: i32,
) -> Mv {
    let (a, b, c, d) = cabac_neighbour_mvs_16x16(grid, mb_x, mb_y, list);
    let inputs = MvpredInputs::from_abc(a, b, c, ref_idx, MvpredShape::Default);
    derive_mvpred_with_d(&inputs, d)
}

/// §8.4.1.2 — derive (refIdxL0, mvL0) for a P_Skip MB at (mb_x, mb_y)
/// using the CABAC encoder grid. Single-ref P-slice → refIdxL0 = 0.
fn cabac_p_skip_mv(grid: &CabacEncGrid, mb_x: usize, mb_y: usize) -> (i32, Mv) {
    let (a, b, c, d) = cabac_neighbour_mvs_16x16(grid, mb_x, mb_y, 0);
    derive_p_skip_mv_with_d(a, b, c, d)
}

/// §8.4.1.2.2 — encoder-side spatial-direct derivation reading from the
/// CABAC encoder grid. Mirrors `b_spatial_direct_derive` in
/// `encoder::mod` but on the CABAC grid layout. Returns a
/// [`CabacBDirect`] with per-8x8 MVs + refIdxs for both lists. The
/// caller checks `is_uniform()` to gate `B_Direct_16x16` / `B_Skip`
/// emission.
///
/// `direct_8x8_inference_flag = 1` is hard-wired in our SPS so the
/// granularity is per-8x8 (4 partitions per MB).
fn cabac_b_spatial_direct_derive(
    grid: &CabacEncGrid,
    l1_partition_mvs: &[FrameRefPartitionMv],
    width_mbs: usize,
    mb_x: usize,
    mb_y: usize,
) -> CabacBDirect {
    // §8.4.1.2.2 step 2-3 — MB-level (A, B, C, D) for both lists.
    let (a_l0, b_l0, c_l0, d_l0) = cabac_neighbour_mvs_16x16(grid, mb_x, mb_y, 0);
    let (a_l1, b_l1, c_l1, d_l1) = cabac_neighbour_mvs_16x16(grid, mb_x, mb_y, 1);

    // §8.4.1.2.2 step 4-5 derivation via the shared helper.
    let (ref_idx_l0_helper, mv_l0_mb, ref_idx_l1_helper, mv_l1_mb) =
        derive_b_spatial_direct_with_d(a_l0, a_l1, b_l0, b_l1, c_l0, c_l1, d_l0, d_l1);

    // Repeat step 4 locally to detect the `directZero` fallback (the
    // helper folds it into the output). Same logic as the CAVLC path.
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
    let min_pos = |x: i32, y: i32| -> i32 {
        if x >= 0 && y >= 0 {
            x.min(y)
        } else {
            x.max(y)
        }
    };
    let r0_minp = min_pos(a_l0.ref_idx, min_pos(b_l0.ref_idx, c_l0_eff.ref_idx));
    let r1_minp = min_pos(a_l1.ref_idx, min_pos(b_l1.ref_idx, c_l1_eff.ref_idx));
    let direct_zero = r0_minp < 0 && r1_minp < 0;

    let ref_idx_l0_final: i32 = if direct_zero {
        0
    } else if r0_minp < 0 {
        -1
    } else {
        ref_idx_l0_helper
    };
    let ref_idx_l1_final: i32 = if direct_zero {
        0
    } else if r1_minp < 0 {
        -1
    } else {
        ref_idx_l1_helper
    };

    // §8.4.1.2.2 step 6/7 — per 8x8 partition MV with colZeroFlag override.
    let mut mv_l0_per_8x8 = [Mv::ZERO; 4];
    let mut mv_l1_per_8x8 = [Mv::ZERO; 4];
    for part in 0..4usize {
        let cz = cabac_col_zero_flag(l1_partition_mvs, width_mbs, mb_x, mb_y, part);
        let l0_force_zero = direct_zero || ref_idx_l0_final < 0 || (ref_idx_l0_final == 0 && cz);
        let l1_force_zero = direct_zero || ref_idx_l1_final < 0 || (ref_idx_l1_final == 0 && cz);
        mv_l0_per_8x8[part] = if l0_force_zero { Mv::ZERO } else { mv_l0_mb };
        mv_l1_per_8x8[part] = if l1_force_zero { Mv::ZERO } else { mv_l1_mb };
    }

    CabacBDirect {
        ref_idx_l0: ref_idx_l0_final,
        ref_idx_l1: ref_idx_l1_final,
        mv_l0_per_8x8,
        mv_l1_per_8x8,
        direct_zero,
    }
}

/// §8.4.1.2.2 step 7 — colZeroFlag check on the L1 anchor's colocated
/// 8x8 partition. Returns true iff the colocated block is short-term-
/// coded with `refIdxCol == 0` and `|mvCol| <= 1` in both components.
fn cabac_col_zero_flag(
    l1_partition_mvs: &[FrameRefPartitionMv],
    width_mbs: usize,
    mb_x: usize,
    mb_y: usize,
    part_8x8: usize,
) -> bool {
    let idx = (mb_y * width_mbs + mb_x) * 4 + part_8x8;
    let Some(slot) = l1_partition_mvs.get(idx) else {
        return false;
    };
    if slot.is_intra {
        return false;
    }
    if slot.ref_idx_l0 != 0 {
        return false;
    }
    slot.mv_l0.0.abs() <= 1 && slot.mv_l0.1.abs() <= 1
}

/// Round-32 — §8.4.1.2.2 spatial direct derivation output (CABAC side).
/// Same shape as the CAVLC `BDirectDerivation` but local to the CABAC
/// path so we don't have to expose the CAVLC type publicly.
#[derive(Debug, Clone, Copy)]
struct CabacBDirect {
    ref_idx_l0: i32,
    ref_idx_l1: i32,
    mv_l0_per_8x8: [Mv; 4],
    mv_l1_per_8x8: [Mv; 4],
    #[allow(dead_code)]
    direct_zero: bool,
}

impl CabacBDirect {
    /// True iff every 8x8 partition shares the same MV across both
    /// lists. Round-32 only emits `B_Direct_16x16` / `B_Skip` when this
    /// holds (the per-8x8 / mixed paths are deferred — they need the
    /// `B_8x8` syntax + sub_mb_type encoder which doesn't exist for
    /// CABAC yet).
    fn is_uniform(&self) -> bool {
        let l0_uniform = self
            .mv_l0_per_8x8
            .iter()
            .all(|m| *m == self.mv_l0_per_8x8[0]);
        let l1_uniform = self
            .mv_l1_per_8x8
            .iter()
            .all(|m| *m == self.mv_l1_per_8x8[0]);
        l0_uniform && l1_uniform
    }

    fn mv_l0(&self) -> Mv {
        self.mv_l0_per_8x8[0]
    }
    fn mv_l1(&self) -> Mv {
        self.mv_l1_per_8x8[0]
    }
}

// ---------------------------------------------------------------------------
// Round-475 — §8.4.1.3 partition-shape MV-predictor derivation reading
// from the CABAC encoder grid. Mirrors the CAVLC-side
// `mvp_for_b_16x8_partition` / `mvp_for_b_8x16_partition` but operates
// on `CabacEncGrid` and is per-list aware (B-slices need separate L0
// and L1 MVPs).
//
// `inflight_*` arrays describe partitions of the CURRENT MB that have
// already been decoded. For a 16x8 MB the second (bottom) partition
// reads from the first (top) partition's MVs through these arrays;
// likewise for 8x16's right reading from left.
// ---------------------------------------------------------------------------

/// Decide the (mb_x', mb_y', 4x4-x, 4x4-y) coordinate of a §6.4.11 4x4
/// neighbour, allowing -1 / >=4 indices that wrap to neighbour MBs. The
/// CABAC-grid analogue of the CAVLC `neighbour_8x8_partition_mv` mapping
/// step. Returns `None` for unavailable / off-grid neighbours and an
/// inflight partition index when the neighbour is in the same MB.
fn cabac_neighbour_partition_mv(
    grid: &CabacEncGrid,
    mb_x: usize,
    mb_y: usize,
    bx_4x4: i32,
    by_4x4: i32,
    list: u8,
    inflight_avail: [bool; 4],
    mv_inflight: [Mv; 4],
    ref_inflight: [i32; 4],
) -> NeighbourMv {
    let h = grid.mbs.len() / grid.width_mbs;
    let (nmb_x, nmb_y, lx, ly) = if bx_4x4 < 0 && (0..4).contains(&by_4x4) {
        // Left MB.
        if mb_x == 0 {
            return NeighbourMv::UNAVAILABLE;
        }
        (mb_x - 1, mb_y, 3, by_4x4)
    } else if (0..4).contains(&bx_4x4) && by_4x4 < 0 {
        // Above MB.
        if mb_y == 0 {
            return NeighbourMv::UNAVAILABLE;
        }
        (mb_x, mb_y - 1, bx_4x4, 3)
    } else if bx_4x4 < 0 && by_4x4 < 0 {
        // Above-left MB (D).
        if mb_x == 0 || mb_y == 0 {
            return NeighbourMv::UNAVAILABLE;
        }
        (mb_x - 1, mb_y - 1, 3, 3)
    } else if (0..4).contains(&bx_4x4) && (0..4).contains(&by_4x4) {
        // Same MB — use inflight.
        let part = 2 * (by_4x4 as usize / 2) + (bx_4x4 as usize / 2);
        if inflight_avail[part] {
            return NeighbourMv {
                available: ref_inflight[part] >= 0,
                mb_available: true,
                partition_available: true,
                ref_idx: ref_inflight[part],
                mv: mv_inflight[part],
            };
        } else {
            return NeighbourMv::partition_not_yet_decoded_same_mb();
        }
    } else if (4..).contains(&bx_4x4) && by_4x4 < 0 {
        // Above-right MB.
        if mb_y == 0 || mb_x + 1 >= grid.width_mbs {
            return NeighbourMv::UNAVAILABLE;
        }
        (mb_x + 1, mb_y - 1, bx_4x4 - 4, 3)
    } else {
        return NeighbourMv::UNAVAILABLE;
    };
    if nmb_y >= h {
        return NeighbourMv::UNAVAILABLE;
    }
    let s = grid.at(nmb_x, nmb_y);
    if !s.available {
        return NeighbourMv::UNAVAILABLE;
    }
    if s.is_intra {
        return NeighbourMv::intra_but_mb_available();
    }
    let part = 2 * (ly as usize / 2) + (lx as usize / 2);
    let (ref_idx, mv) = if list == 0 {
        (s.ref_idx_l0_8x8[part], s.mv_l0_8x8[part])
    } else {
        (s.ref_idx_l1_8x8[part], s.mv_l1_8x8[part])
    };
    NeighbourMv {
        available: ref_idx >= 0,
        mb_available: true,
        partition_available: true,
        ref_idx,
        mv,
    }
}

/// §8.4.1.3 — derive mvpLX for one 16x8 partition (top or bottom) of a
/// B MB at `(mb_x, mb_y)` using the CABAC encoder grid. Mirror of the
/// CAVLC-side `mvp_for_b_16x8_partition`.
#[allow(clippy::too_many_arguments)]
fn cabac_mvp_for_b_16x8_partition(
    grid: &CabacEncGrid,
    mb_x: usize,
    mb_y: usize,
    is_top: bool,
    ref_idx: i32,
    list: u8,
    inflight_avail: [bool; 4],
    mv_inflight: [Mv; 4],
    ref_inflight: [i32; 4],
) -> Mv {
    let by = if is_top { 0i32 } else { 2 };
    let a = cabac_neighbour_partition_mv(
        grid,
        mb_x,
        mb_y,
        -1,
        by,
        list,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let b = cabac_neighbour_partition_mv(
        grid,
        mb_x,
        mb_y,
        0,
        by - 1,
        list,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let c = cabac_neighbour_partition_mv(
        grid,
        mb_x,
        mb_y,
        4,
        by - 1,
        list,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let d = cabac_neighbour_partition_mv(
        grid,
        mb_x,
        mb_y,
        -1,
        by - 1,
        list,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let shape = if is_top {
        MvpredShape::Partition16x8Top
    } else {
        MvpredShape::Partition16x8Bottom
    };
    let inputs = MvpredInputs::from_abc(a, b, c, ref_idx, shape);
    derive_mvpred_with_d(&inputs, d)
}

/// §8.4.1.3 — derive mvpLX for one 8x16 partition (left or right) of a
/// B MB at `(mb_x, mb_y)` using the CABAC encoder grid.
#[allow(clippy::too_many_arguments)]
fn cabac_mvp_for_b_8x16_partition(
    grid: &CabacEncGrid,
    mb_x: usize,
    mb_y: usize,
    is_left: bool,
    ref_idx: i32,
    list: u8,
    inflight_avail: [bool; 4],
    mv_inflight: [Mv; 4],
    ref_inflight: [i32; 4],
) -> Mv {
    let bx = if is_left { 0i32 } else { 2 };
    let a = cabac_neighbour_partition_mv(
        grid,
        mb_x,
        mb_y,
        bx - 1,
        0,
        list,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let b = cabac_neighbour_partition_mv(
        grid,
        mb_x,
        mb_y,
        bx,
        -1,
        list,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let c = cabac_neighbour_partition_mv(
        grid,
        mb_x,
        mb_y,
        bx + 2,
        -1,
        list,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let d = cabac_neighbour_partition_mv(
        grid,
        mb_x,
        mb_y,
        bx - 1,
        -1,
        list,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let shape = if is_left {
        MvpredShape::Partition8x16Left
    } else {
        MvpredShape::Partition8x16Right
    };
    let inputs = MvpredInputs::from_abc(a, b, c, ref_idx, shape);
    derive_mvpred_with_d(&inputs, d)
}

/// §8.4.1.3 — derive mvpLX for one 8x8 sub-MB partition (cell idx ∈
/// {0, 1, 2, 3} in raster Z-order TL/TR/BL/BR) of a B `B_8x8` MB at
/// `(mb_x, mb_y)` using the CABAC encoder grid. Round-475 mixed-cell
/// path. Mirrors the CAVLC-side `mvp_for_p_8x8_partition` (extended for
/// per-list operation in B-slices).
///
/// `cell_decoded[c]` carries whether cell `c` has been processed
/// (decoder analogue: whether `process_partition` has been called on it
/// yet). `mv_inflight[c]` and `ref_inflight[c]` carry that cell's
/// per-list MV and refIdx (`-1` if the list is not used by that cell —
/// e.g. a Direct cell with refIdxL0 < 0, or an explicit L1-only cell on
/// the L0 lookup). When `cell_decoded[c]` is false the MVP function
/// returns the [`NeighbourMv::partition_not_yet_decoded_same_mb`]
/// sentinel; when true but `ref_inflight[c] < 0` it returns
/// `intra_but_mb_available` (matching the decoder's grid behaviour for a
/// processed cell whose other-list-only entry leaves the queried list
/// at the default `(0, 0), -1`).
#[allow(clippy::too_many_arguments)]
fn cabac_mvp_for_b_8x8_partition(
    grid: &CabacEncGrid,
    mb_x: usize,
    mb_y: usize,
    sub_idx: usize,
    ref_idx: i32,
    list: u8,
    cell_decoded: [bool; 4],
    mv_inflight: [Mv; 4],
    ref_inflight: [i32; 4],
) -> Mv {
    // Top-left 4x4 of this 8x8 partition, in 4x4-block units.
    let sub_x = (sub_idx % 2) as i32;
    let sub_y = (sub_idx / 2) as i32;
    let bx = sub_x * 2;
    let by = sub_y * 2;
    // Partition width = 8 = 2 4x4 blocks → C is at (bx + 2, by - 1).
    let a = cabac_neighbour_partition_mv_b8x8(
        grid,
        mb_x,
        mb_y,
        bx - 1,
        by,
        list,
        cell_decoded,
        mv_inflight,
        ref_inflight,
    );
    let b = cabac_neighbour_partition_mv_b8x8(
        grid,
        mb_x,
        mb_y,
        bx,
        by - 1,
        list,
        cell_decoded,
        mv_inflight,
        ref_inflight,
    );
    let c = cabac_neighbour_partition_mv_b8x8(
        grid,
        mb_x,
        mb_y,
        bx + 2,
        by - 1,
        list,
        cell_decoded,
        mv_inflight,
        ref_inflight,
    );
    let d = cabac_neighbour_partition_mv_b8x8(
        grid,
        mb_x,
        mb_y,
        bx - 1,
        by - 1,
        list,
        cell_decoded,
        mv_inflight,
        ref_inflight,
    );
    let inputs = MvpredInputs::from_abc(a, b, c, ref_idx, MvpredShape::Default);
    derive_mvpred_with_d(&inputs, d)
}

/// Variant of `cabac_neighbour_partition_mv` for the B_8x8 mixed path.
/// The same-MB case distinguishes between "cell decoded" (returns
/// `intra_but_mb_available` when the queried list is unused — partition
/// IS available per §6.4.11.1) and "cell not yet decoded" (returns
/// `partition_not_yet_decoded_same_mb` — partition IS NOT available).
#[allow(clippy::too_many_arguments)]
fn cabac_neighbour_partition_mv_b8x8(
    grid: &CabacEncGrid,
    mb_x: usize,
    mb_y: usize,
    bx_4x4: i32,
    by_4x4: i32,
    list: u8,
    cell_decoded: [bool; 4],
    mv_inflight: [Mv; 4],
    ref_inflight: [i32; 4],
) -> NeighbourMv {
    let h = grid.mbs.len() / grid.width_mbs;
    let (nmb_x, nmb_y, lx, ly) = if bx_4x4 < 0 && (0..4).contains(&by_4x4) {
        if mb_x == 0 {
            return NeighbourMv::UNAVAILABLE;
        }
        (mb_x - 1, mb_y, 3, by_4x4)
    } else if (0..4).contains(&bx_4x4) && by_4x4 < 0 {
        if mb_y == 0 {
            return NeighbourMv::UNAVAILABLE;
        }
        (mb_x, mb_y - 1, bx_4x4, 3)
    } else if bx_4x4 < 0 && by_4x4 < 0 {
        if mb_x == 0 || mb_y == 0 {
            return NeighbourMv::UNAVAILABLE;
        }
        (mb_x - 1, mb_y - 1, 3, 3)
    } else if (0..4).contains(&bx_4x4) && (0..4).contains(&by_4x4) {
        // Same MB — use cell_decoded + inflight.
        let part = 2 * (by_4x4 as usize / 2) + (bx_4x4 as usize / 2);
        if !cell_decoded[part] {
            return NeighbourMv::partition_not_yet_decoded_same_mb();
        }
        // Cell IS decoded. If the queried list isn't used by that cell
        // (ref_inflight < 0), the decoder grid has the default (0,0)/-1
        // pair → `intra_but_mb_available` (partition available, but list
        // unused → mvLXN=0, refIdxLXN=-1 in §8.4.1.3.2 step 2a sense).
        if ref_inflight[part] < 0 {
            return NeighbourMv::intra_but_mb_available();
        }
        return NeighbourMv {
            available: true,
            mb_available: true,
            partition_available: true,
            ref_idx: ref_inflight[part],
            mv: mv_inflight[part],
        };
    } else if (4..).contains(&bx_4x4) && by_4x4 < 0 {
        if mb_y == 0 || mb_x + 1 >= grid.width_mbs {
            return NeighbourMv::UNAVAILABLE;
        }
        (mb_x + 1, mb_y - 1, bx_4x4 - 4, 3)
    } else {
        return NeighbourMv::UNAVAILABLE;
    };
    if nmb_y >= h {
        return NeighbourMv::UNAVAILABLE;
    }
    let s = grid.at(nmb_x, nmb_y);
    if !s.available {
        return NeighbourMv::UNAVAILABLE;
    }
    if s.is_intra {
        return NeighbourMv::intra_but_mb_available();
    }
    let part = 2 * (ly as usize / 2) + (lx as usize / 2);
    let (ref_idx, mv) = if list == 0 {
        (s.ref_idx_l0_8x8[part], s.mv_l0_8x8[part])
    } else {
        (s.ref_idx_l1_8x8[part], s.mv_l1_8x8[part])
    };
    if ref_idx < 0 {
        return NeighbourMv::intra_but_mb_available();
    }
    NeighbourMv {
        available: true,
        mb_available: true,
        partition_available: true,
        ref_idx,
        mv,
    }
}

/// §9.3.3.1.1.9 — CBF neighbour conditions for a block at the MB level
/// (LumaDC, ChromaDC). Returns `(cond_left, cond_above)`.
///
/// For unavailable neighbour: `current_is_intra ? true : false` (the spec's
/// `unavail_cbf` rule). For I_PCM neighbour: `true`. For skip: `false`.
fn cbf_neighbour_mb_level(
    grid: &CabacEncGrid,
    mb_x: usize,
    mb_y: usize,
    current_is_intra: bool,
    plane: CbfPlane,
) -> (bool, bool) {
    let pick = |o: Option<&CabacEncMbInfo>| -> bool {
        match o {
            None => current_is_intra,
            Some(info) => {
                if !info.available {
                    current_is_intra
                } else if info.is_i_pcm {
                    true
                } else if info.is_skip {
                    false
                } else {
                    match plane {
                        CbfPlane::LumaDc => info.cbf_luma_dc,
                        CbfPlane::CbDc => info.cbf_cb_dc,
                        CbfPlane::CrDc => info.cbf_cr_dc,
                    }
                }
            }
        }
    };
    (pick(grid.left(mb_x, mb_y)), pick(grid.above(mb_x, mb_y)))
}

#[derive(Clone, Copy)]
enum CbfPlane {
    LumaDc,
    CbDc,
    CrDc,
}

/// §9.3.3.1.1.9 — CBF neighbour conditions for a 4x4 luma AC block of the
/// CURRENT MB. `blk_idx` is the §6.4.3 raster-Z scan index. The "internal
/// neighbour" branch reads the running CBF state of the current MB; the
/// "external neighbour" branch reads from the left/above MB's CBF arrays.
#[allow(clippy::too_many_arguments)]
fn cbf_neighbour_luma_ac(
    grid: &CabacEncGrid,
    mb_x: usize,
    mb_y: usize,
    current: &CabacEncMbInfo,
    current_is_intra: bool,
    block_type: BlockType,
    blk_idx: u8,
) -> (bool, bool) {
    // 4x4 (x,y) inside the MB for `blk_idx`.
    let (bx, by) = blk4x4_xy(blk_idx);
    let pick = |xn: i32, yn: i32, is_left: bool| -> bool {
        if xn >= 0 && yn >= 0 {
            // Internal — read running CBF of current MB.
            let bi = blk4x4_idx(xn, yn) as usize;
            cbf_luma_for_local(current, block_type, bi)
        } else {
            // External MB.
            let nbr = if is_left {
                grid.left(mb_x, mb_y)
            } else {
                grid.above(mb_x, mb_y)
            };
            let Some(info) = nbr else {
                return current_is_intra;
            };
            if !info.available {
                return current_is_intra;
            }
            if info.is_i_pcm {
                return true;
            }
            if info.is_skip {
                return false;
            }
            // Wrap into neighbour MB.
            let (wx, wy) = if is_left {
                (xn + 16, yn)
            } else {
                (xn, yn + 16)
            };
            let bi = blk4x4_idx(wx, wy) as usize;
            // Check 8x8 CBP gate.
            let blk8 = (bi >> 2) as u8;
            if ((info.cbp_luma >> blk8) & 1) == 0 {
                return false;
            }
            cbf_luma_for_local(info, block_type, bi)
        }
    };
    let cond_a = pick(bx - 1, by, true);
    let cond_b = pick(bx, by - 1, false);
    (cond_a, cond_b)
}

#[inline]
fn blk4x4_xy(idx: u8) -> (i32, i32) {
    let hi = (idx / 4) as i32;
    let lo = (idx % 4) as i32;
    let x_hi = (hi % 2) * 8;
    let y_hi = (hi / 2) * 8;
    let x_lo = (lo % 2) * 4;
    let y_lo = (lo / 2) * 4;
    (x_hi + x_lo, y_hi + y_lo)
}

#[inline]
fn blk4x4_idx(x: i32, y: i32) -> u8 {
    (8 * (y / 8) + 4 * (x / 8) + 2 * ((y % 8) / 4) + ((x % 8) / 4)) as u8
}

/// §9.3.3.1.1.7 — derive `absMvdCompA + absMvdCompB` for the current
/// 4x4 block `blk4x4` of the given list / component, mirroring the
/// decoder-side [`crate::macroblock_layer::cabac_mvd_abs_sum`]. Used
/// for the encoder ctxIdxInc derivation when emitting `mvd_lN` so the
/// encoder and decoder pick the same context.
///
/// `inflight` carries the current MB's per-4x4 |mvd| values for any
/// already-emitted partitions (when the (A, B) lookup falls inside
/// the current MB).
#[allow(clippy::too_many_arguments)]
fn cabac_enc_mvd_abs_sum(
    grid: &CabacEncGrid,
    mb_x: usize,
    mb_y: usize,
    list: u8,
    comp: MvdComponent,
    blk4x4: u8,
    inflight: &[u32; 16],
) -> u32 {
    let (bx, by) = blk4x4_xy(blk4x4);
    let pick = |xn: i32, yn: i32| -> u32 {
        if xn >= 0 && yn >= 0 {
            // Internal — current MB's already-decoded partition state.
            let idx = blk4x4_idx(xn, yn) as usize;
            return inflight[idx];
        }
        // External neighbour MB.
        let nbr = if xn < 0 && yn >= 0 {
            // Left MB — wrap xn into right edge (xn + 16).
            if mb_x == 0 {
                return 0;
            }
            let info = grid.at(mb_x - 1, mb_y);
            (info, xn + 16, yn)
        } else if xn >= 0 && yn < 0 {
            if mb_y == 0 {
                return 0;
            }
            let info = grid.at(mb_x, mb_y - 1);
            (info, xn, yn + 16)
        } else {
            return 0;
        };
        let (info, nxn, nyn) = nbr;
        if !info.available {
            return 0;
        }
        if info.is_skip || info.is_intra {
            return 0;
        }
        let idx = blk4x4_idx(nxn, nyn) as usize;
        match (list, comp) {
            (0, MvdComponent::X) => info.abs_mvd_l0_x[idx],
            (0, MvdComponent::Y) => info.abs_mvd_l0_y[idx],
            (1, MvdComponent::X) => info.abs_mvd_l1_x[idx],
            _ => info.abs_mvd_l1_y[idx],
        }
    };
    pick(bx - 1, by) + pick(bx, by - 1)
}

#[inline]
fn cbf_luma_for_local(info: &CabacEncMbInfo, block_type: BlockType, bi: usize) -> bool {
    match block_type {
        BlockType::Luma4x4 | BlockType::Luma16x16Ac => {
            info.cbf_luma_4x4.get(bi).copied().unwrap_or(false)
                || info.cbf_luma_ac.get(bi).copied().unwrap_or(false)
        }
        // 4:4:4 chroma planes: coded "like luma" with 16 4x4 AC blocks.
        BlockType::CbIntra16x16Ac | BlockType::CbLuma4x4 => {
            info.cbf_cb_ac_444.get(bi).copied().unwrap_or(false)
        }
        BlockType::CrIntra16x16Ac | BlockType::CrLuma4x4 => {
            info.cbf_cr_ac_444.get(bi).copied().unwrap_or(false)
        }
        _ => false,
    }
}

/// §9.3.3.1.1.9 — CBF for chroma AC sub-blocks. Internal neighbour
/// reads the current MB's running CBF state, external goes to the
/// neighbour MB. `is_cr` selects between Cb and Cr.
fn cbf_neighbour_chroma_ac(
    grid: &CabacEncGrid,
    mb_x: usize,
    mb_y: usize,
    current: &CabacEncMbInfo,
    current_is_intra: bool,
    blk_idx: u8,
    is_cr: bool,
) -> (bool, bool) {
    // Chroma 4:2:0 layout: 2x2 blocks. Coordinates in 4-pixel units.
    let cx = ((blk_idx as i32) % 2) * 4;
    let cy = ((blk_idx as i32) / 2) * 4;
    let pick = |xn: i32, yn: i32, is_left: bool| -> bool {
        if xn >= 0 && yn >= 0 {
            // Internal: read current MB's running chroma AC state.
            let bi = ((yn / 4) * 2 + (xn / 4)) as usize;
            if is_cr {
                current.cbf_cr_ac.get(bi).copied().unwrap_or(false)
            } else {
                current.cbf_cb_ac.get(bi).copied().unwrap_or(false)
            }
        } else {
            let nbr = if is_left {
                grid.left(mb_x, mb_y)
            } else {
                grid.above(mb_x, mb_y)
            };
            let Some(info) = nbr else {
                return current_is_intra;
            };
            if !info.available {
                return current_is_intra;
            }
            if info.is_i_pcm {
                return true;
            }
            if info.is_skip {
                return false;
            }
            // Wrap into the neighbour's chroma.
            let (wx, wy) = if is_left { (xn + 8, yn) } else { (xn, yn + 8) };
            let bi = ((wy / 4) * 2 + (wx / 4)) as usize;
            if is_cr {
                info.cbf_cr_ac.get(bi).copied().unwrap_or(false)
            } else {
                info.cbf_cb_ac.get(bi).copied().unwrap_or(false)
            }
        }
    };
    let cond_a = pick(cx - 1, cy, true);
    let cond_b = pick(cx, cy - 1, false);
    (cond_a, cond_b)
}

/// Build a NeighbourCtx from the encoder's grid.
fn build_neighbour_ctx(grid: &CabacEncGrid, mb_x: usize, mb_y: usize) -> NeighbourCtx {
    let left = grid.left(mb_x, mb_y);
    let above = grid.above(mb_x, mb_y);
    let mk = |o: Option<&CabacEncMbInfo>| -> (bool, bool, bool, bool, bool, u8, u8, u8, bool) {
        match o {
            None => (false, false, false, false, false, 0, 0, 0, false),
            Some(m) => (
                m.available,
                m.is_skip,
                !m.is_intra,
                m.is_i_pcm,
                m.is_i_nxn,
                m.cbp_luma,
                m.cbp_chroma,
                m.intra_chroma_pred_mode,
                m.is_b_skip_or_direct,
            ),
        }
    };
    let (la, ls, li, lpcm, linxn, lcbpl, lcbpc, licpm, lbsd) = mk(left);
    let (aa, as_, ai, apcm, ainxn, acbpl, acbpc, aicpm, absd) = mk(above);
    NeighbourCtx {
        available_left: la,
        available_above: aa,
        mb_skip_flag_left: ls,
        mb_skip_flag_above: as_,
        left_is_si: false,
        above_is_si: false,
        left_is_i_nxn: linxn,
        above_is_i_nxn: ainxn,
        left_is_b_skip_or_direct: lbsd,
        above_is_b_skip_or_direct: absd,
        left_inter: li,
        above_inter: ai,
        left_is_i_pcm: lpcm,
        above_is_i_pcm: apcm,
        left_intra_chroma_pred_mode_nonzero: licpm != 0,
        above_intra_chroma_pred_mode_nonzero: aicpm != 0,
        left_transform_8x8: false,
        above_transform_8x8: false,
        left_cbp_luma: lcbpl,
        above_cbp_luma: acbpl,
        left_cbp_chroma: lcbpc,
        above_cbp_chroma: acbpc,
        left_is_p_or_b_skip: ls,
        above_is_p_or_b_skip: as_,
    }
}

// ---------------------------------------------------------------------------
// Per-MB encode helpers.
// ---------------------------------------------------------------------------

fn pick_intra16x16_mode(
    frame: &YuvFrame<'_>,
    recon_y: &[u8],
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
) -> (I16x16Mode, [i32; 256]) {
    let mut best_mode = I16x16Mode::Dc;
    let mut best_pred = [0i32; 256];
    let mut best_sad = u64::MAX;
    for &mode in &[
        I16x16Mode::Vertical,
        I16x16Mode::Horizontal,
        I16x16Mode::Dc,
        I16x16Mode::Plane,
    ] {
        let Some(pred) = predict_16x16(mode, recon_y, width, height, mb_x, mb_y) else {
            continue;
        };
        // SAD against source.
        let mut sad: u64 = 0;
        for j in 0..16usize {
            for i in 0..16usize {
                let s = frame.y[(mb_y * 16 + j) * width + (mb_x * 16 + i)] as i32;
                sad += (s - pred[j * 16 + i]).unsigned_abs() as u64;
            }
        }
        if sad < best_sad {
            best_sad = sad;
            best_mode = mode;
            best_pred = pred;
        }
    }
    (best_mode, best_pred)
}

fn pick_chroma_mode(
    frame_u: &[u8],
    frame_v: &[u8],
    recon_u: &[u8],
    recon_v: &[u8],
    chroma_w: usize,
    chroma_h: usize,
    mb_x: usize,
    mb_y: usize,
) -> (IntraChromaMode, [i32; 64], [i32; 64]) {
    let mut best_mode = IntraChromaMode::Dc;
    let mut best_pred_u = [0i32; 64];
    let mut best_pred_v = [0i32; 64];
    let mut best_sad = u64::MAX;
    for &mode in &[
        IntraChromaMode::Dc,
        IntraChromaMode::Horizontal,
        IntraChromaMode::Vertical,
        IntraChromaMode::Plane,
    ] {
        let Some(pred_u) = predict_chroma_8x8(mode, recon_u, chroma_w, chroma_h, mb_x, mb_y) else {
            continue;
        };
        let Some(pred_v) = predict_chroma_8x8(mode, recon_v, chroma_w, chroma_h, mb_x, mb_y) else {
            continue;
        };
        let sad_u = sad_8x8(frame_u, chroma_w, mb_x, mb_y, &pred_u);
        let sad_v = sad_8x8(frame_v, chroma_w, mb_x, mb_y, &pred_v);
        let sad = sad_u as u64 + sad_v as u64;
        if sad < best_sad {
            best_sad = sad;
            best_mode = mode;
            best_pred_u = pred_u;
            best_pred_v = pred_v;
        }
    }
    (best_mode, best_pred_u, best_pred_v)
}

/// Quantise a 16x16 luma residual: returns per-block (DC, AC quant raster,
/// AC scan) plus a flag for any AC nonzero.
fn quantize_intra16x16_luma(
    residual: &[i32; 256],
    qp_y: i32,
) -> ([i32; 16], [[i32; 16]; 16], [[i32; 16]; 16], bool) {
    // Per-block forward 4x4.
    let mut coeffs_4x4 = [[0i32; 16]; 16];
    for blk in 0..16usize {
        let bx = blk % 4;
        let by = blk / 4;
        let mut block = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                block[j * 4 + i] = residual[(by * 4 + j) * 16 + bx * 4 + i];
            }
        }
        coeffs_4x4[blk] = forward_core_4x4(&block);
    }
    // DC matrix.
    let mut dc_coeffs = [0i32; 16];
    for (blk, c) in coeffs_4x4.iter().enumerate() {
        let bx = blk % 4;
        let by = blk / 4;
        dc_coeffs[by * 4 + bx] = c[0];
    }
    let dc_hadamard = forward_hadamard_4x4(&dc_coeffs);
    let dc_levels = quantize_luma_dc(&dc_hadamard, qp_y, true);

    // Per-block AC quantise.
    let mut ac_quant_raster = [[0i32; 16]; 16];
    let mut ac_scan = [[0i32; 16]; 16];
    let mut any_ac_nz = false;
    for blkz in 0..16usize {
        let (bx, by) = LUMA_4X4_BLK[blkz];
        let raster = by * 4 + bx;
        let z = quantize_4x4_ac(&coeffs_4x4[raster], qp_y, true);
        ac_quant_raster[raster] = z;
        let scan_ac = zigzag_scan_4x4_ac(&z);
        ac_scan[blkz] = scan_ac;
        if scan_ac.iter().any(|&v| v != 0) {
            any_ac_nz = true;
        }
    }
    (dc_levels, ac_quant_raster, ac_scan, any_ac_nz)
}

/// Reconstruct an Intra_16x16 luma MB into `recon_y`.
fn reconstruct_intra16x16_luma(
    mb_x: usize,
    mb_y: usize,
    width: usize,
    pred: &[i32; 256],
    dc_levels: &[i32; 16],
    ac_quant_raster: &[[i32; 16]; 16],
    cbp_luma: u8,
    qp_y: i32,
    recon_y: &mut [u8],
) {
    let inv_dc =
        inverse_hadamard_luma_dc_16x16(dc_levels, qp_y, &FLAT_4X4_16, 8).expect("luma DC inverse");
    for blk in 0..16usize {
        let bx = blk % 4;
        let by = blk / 4;
        let mut coeffs_in = if cbp_luma == 15 {
            ac_quant_raster[blk]
        } else {
            [0i32; 16]
        };
        coeffs_in[0] = inv_dc[by * 4 + bx];
        let r = inverse_transform_4x4_dc_preserved(&coeffs_in, qp_y, &FLAT_4X4_16, 8)
            .expect("inverse 4x4");
        for j in 0..4 {
            for i in 0..4 {
                let px = mb_x * 16 + bx * 4 + i;
                let py = mb_y * 16 + by * 4 + j;
                let p = pred[(by * 4 + j) * 16 + bx * 4 + i];
                recon_y[py * width + px] = (p + r[j * 4 + i]).clamp(0, 255) as u8;
            }
        }
    }
}

/// Encode a chroma plane (Cb or Cr) for an Intra_16x16 MB at 4:2:0.
/// Returns (dc_levels[4], ac_scan[4][16], ac_quant_raster[4][16],
/// any_dc_nz, any_ac_nz, recon_residual[64]).
#[allow(clippy::type_complexity)]
fn encode_chroma_intra16x16_420(
    frame_c: &[u8],
    recon_c: &[u8],
    chroma_w: usize,
    chroma_h: usize,
    mb_x: usize,
    mb_y: usize,
    mode: IntraChromaMode,
    qp_c: i32,
) -> (
    [i32; 4],
    [[i32; 16]; 4],
    [[i32; 16]; 4],
    bool,
    bool,
    [i32; 64],
) {
    let _ = recon_c;
    let pred = predict_chroma_8x8(mode, recon_c, chroma_w, chroma_h, mb_x, mb_y).unwrap();
    // Residual.
    let mut residual = [0i32; 64];
    for j in 0..8usize {
        for i in 0..8usize {
            let s = frame_c[(mb_y * 8 + j) * chroma_w + (mb_x * 8 + i)] as i32;
            residual[j * 8 + i] = s - pred[j * 8 + i];
        }
    }
    // Per-block 4x4 forward.
    let mut blocks = [[0i32; 16]; 4];
    for blk in 0..4usize {
        let bx = blk % 2;
        let by = blk / 2;
        for j in 0..4 {
            for i in 0..4 {
                blocks[blk][j * 4 + i] = residual[(by * 4 + j) * 8 + bx * 4 + i];
            }
        }
        blocks[blk] = forward_core_4x4(&blocks[blk]);
    }
    // DC matrix (4 entries).
    let dc_in = [blocks[0][0], blocks[1][0], blocks[2][0], blocks[3][0]];
    let dc_had = forward_hadamard_2x2(&dc_in);
    let dc_levels = quantize_chroma_dc(&dc_had, qp_c, true);
    // Per-block AC.
    let mut ac_quant = [[0i32; 16]; 4];
    let mut ac_scan = [[0i32; 16]; 4];
    let mut any_ac_nz = false;
    for blk in 0..4usize {
        let z = quantize_4x4_ac(&blocks[blk], qp_c, true);
        ac_quant[blk] = z;
        let s = zigzag_scan_4x4_ac(&z);
        ac_scan[blk] = s;
        if s.iter().any(|&v| v != 0) {
            any_ac_nz = true;
        }
    }
    let any_dc_nz = dc_levels.iter().any(|&v| v != 0);

    // Inverse path for recon (use the inverse Hadamard + inverse 4x4 with
    // DC overwritten).
    let inv_dc = inverse_hadamard_chroma_dc_420(&dc_levels, qp_c, &FLAT_4X4_16, 8)
        .expect("chroma DC inverse");
    let mut recon_residual = [0i32; 64];
    for blk in 0..4usize {
        let bx = blk % 2;
        let by = blk / 2;
        let mut coeffs_in = ac_quant[blk];
        // Note: the decoder reads chroma AC blocks ONLY when cbp_chroma==2.
        // For DC-only (cbp_chroma==1) the AC slots are zero when decoded.
        // We'll handle this in the caller by zeroing if DC-only.
        coeffs_in[0] = inv_dc[blk];
        let r = inverse_transform_4x4_dc_preserved(&coeffs_in, qp_c, &FLAT_4X4_16, 8)
            .expect("inverse 4x4 chroma");
        for j in 0..4 {
            for i in 0..4 {
                recon_residual[(by * 4 + j) * 8 + bx * 4 + i] = r[j * 4 + i];
            }
        }
    }
    (
        dc_levels,
        ac_scan,
        ac_quant,
        any_dc_nz,
        any_ac_nz,
        recon_residual,
    )
}

// ---------------------------------------------------------------------------
// IDR / P entry points.
// ---------------------------------------------------------------------------

impl Encoder {
    /// Round-30 — encode an IDR access unit using **CABAC** entropy coding.
    /// `EncoderConfig::cabac` must be `true` and `profile_idc >= 77`.
    pub fn encode_idr_cabac(&self, frame: &YuvFrame<'_>) -> EncodedIdr {
        let cfg = self.config();
        assert!(
            cfg.cabac,
            "encode_idr_cabac requires EncoderConfig::cabac = true",
        );
        assert!(
            cfg.profile_idc >= 77,
            "CABAC requires Main profile or higher (got profile_idc={})",
            cfg.profile_idc,
        );
        assert!(
            matches!(cfg.chroma_format_idc, 1 | 3),
            "encode_idr_cabac supports chroma_format_idc 1 (4:2:0) and 3 (4:4:4); \
             4:2:2 CABAC IDR is not yet wired",
        );
        let width = cfg.width as usize;
        let height = cfg.height as usize;
        let width_mbs = (cfg.width / 16) as usize;
        let height_mbs = (cfg.height / 16) as usize;
        // §6.2 Table 6-1 — chroma plane dimensions.
        let chroma_w = match cfg.chroma_format_idc {
            3 => width,
            _ => width / 2,
        };
        let chroma_h = match cfg.chroma_format_idc {
            3 => height,
            _ => height / 2,
        };

        let qp_y = cfg.qp;
        // §8.5.8 BD-aware chroma-QP. `qp_bd_offset_c = 6 *
        // bit_depth_chroma_minus8` extends the eq. 8-311 qPI clamp
        // from 0..=51 to −QpBdOffsetC..=51. Today the SPS pins
        // bit_depth_chroma_minus8=0 (round-30 scope), so this is
        // equivalent to the legacy 8-bit shim; the BD-aware call lets
        // a future High10/422/444 round wire a non-zero BD without
        // re-touching the encoder math.
        let qp_bd_offset_c = qp_bd_offset(cfg.bit_depth_chroma_minus8);
        let qp_c = qp_y_to_qp_c_with_bd_offset(qp_y, 0, qp_bd_offset_c);

        let sps = build_baseline_sps_rbsp(&BaselineSpsConfig {
            seq_parameter_set_id: 0,
            level_idc: cfg.level_idc,
            width_in_mbs: width_mbs as u32,
            height_in_mbs: height_mbs as u32,
            log2_max_frame_num_minus4: 4,
            log2_max_poc_lsb_minus4: 4,
            max_num_ref_frames: cfg.max_num_ref_frames,
            profile_idc: cfg.profile_idc,
            chroma_format_idc: cfg.chroma_format_idc,
        });
        let pps_cfg = BaselinePpsConfig {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
            pic_init_qp_minus26: cfg.qp - 26,
            chroma_qp_index_offset: 0,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            entropy_coding_mode_flag: true,
        };
        let pps = build_baseline_pps_rbsp(&pps_cfg);
        let mut stream: Vec<u8> = Vec::new();
        stream.extend_from_slice(&build_nal_unit(3, NalUnitType::Sps, &sps));
        stream.extend_from_slice(&build_nal_unit(3, NalUnitType::Pps, &pps));

        // Slice header.
        let mut sw = BitWriter::new();
        write_idr_i_slice_header(
            &mut sw,
            &IdrSliceHeaderConfig {
                first_mb_in_slice: 0,
                slice_type_raw: 7,
                pic_parameter_set_id: 0,
                frame_num: 0,
                frame_num_bits: 8,
                idr_pic_id: 0,
                pic_order_cnt_lsb: 0,
                poc_lsb_bits: 8,
                slice_qp_delta: 0,
                // Disable deblocking for 4:4:4 — the deblock module's
                // chroma filter assumes 4:2:0 8x8 planes; 4:4:4 uses the
                // luma filter on full-size chroma planes which is not yet
                // wired in the deblock layer.
                disable_deblocking_filter_idc: if cfg.chroma_format_idc == 3 { 1 } else { 0 },
                slice_alpha_c0_offset_div2: 0,
                slice_beta_offset_div2: 0,
            },
        );
        // §7.3.4 — cabac_alignment_one_bit until byte aligned.
        while !sw.byte_aligned() {
            sw.u(1, 1);
        }

        // ---- CABAC engine setup ----
        let mut cabac = CabacEncoder::new();
        let mut ctxs = CabacContexts::init(SliceKind::I, None, qp_y).expect("ctx init");

        let mut recon_y = vec![0u8; width * height];
        let mut recon_u = vec![0u8; chroma_w * chroma_h];
        let mut recon_v = vec![0u8; chroma_w * chroma_h];
        let mut grid = CabacEncGrid::new(width_mbs, height_mbs);
        let mut mb_dbl: Vec<MbDeblockInfo> = vec![MbDeblockInfo::default(); width_mbs * height_mbs];

        // §9.3.3.1.1.5 — prev_mb_qp_delta_nonzero is per-slice rolling state.
        let mut prev_mb_qp_delta_nonzero = false;

        for mb_y in 0..height_mbs {
            for mb_x in 0..width_mbs {
                let mb_addr = mb_y * width_mbs + mb_x;
                let nb = build_neighbour_ctx(&grid, mb_x, mb_y);

                // Pick intra modes.
                let (luma_mode, luma_pred) =
                    pick_intra16x16_mode(frame, &recon_y, width, height, mb_x, mb_y);
                let (chroma_mode, _, _) = pick_chroma_mode(
                    frame.u, frame.v, &recon_u, &recon_v, chroma_w, chroma_h, mb_x, mb_y,
                );

                // Residual + transform + quant.
                let mut residual = [0i32; 256];
                for j in 0..16usize {
                    for i in 0..16usize {
                        let s = frame.y[(mb_y * 16 + j) * width + (mb_x * 16 + i)] as i32;
                        residual[j * 16 + i] = s - luma_pred[j * 16 + i];
                    }
                }
                let (luma_dc, luma_ac_quant, luma_ac_scan, any_luma_ac_nz) =
                    quantize_intra16x16_luma(&residual, qp_y);
                let mut cbp_luma: u8 = if any_luma_ac_nz { 15 } else { 0 };

                // --------------- chroma encode (4:2:0 vs 4:4:4) ---------------
                //
                // 4:4:4: chroma coded "like luma" per §7.3.5.3:
                //   • encode_chroma_intra16x16_444 both modifies recon_u/v and
                //     returns the quantized DC + AC arrays for CABAC emit.
                //   • cbp_chroma stays 0 (the 4:4:4 cbp is folded into mb_type
                //     via cbp_luma per §7.4.5.1 Table 7-11).
                //   • intra_chroma_pred_mode is NOT emitted (§7.3.5.1 note).
                //
                // 4:2:0: legacy path with separate ChromaDc/ChromaAc blocks.

                // 4:4:4 chroma residual arrays (only populated when fmt==3).
                let mut cb_dc_444 = [0i32; 16];
                let mut cr_dc_444 = [0i32; 16];
                let mut cb_ac_scan_444 = [[0i32; 16]; 16];
                let mut cr_ac_scan_444 = [[0i32; 16]; 16];
                // 4:2:0 chroma residual arrays (only populated when fmt==1).
                let mut cb_dc_420 = [0i32; 4];
                let mut cr_dc_420 = [0i32; 4];
                let mut cb_ac_scan_420 = [[0i32; 16]; 4];
                let mut cr_ac_scan_420 = [[0i32; 16]; 4];
                let cbp_chroma: u8;

                if cfg.chroma_format_idc == 3 {
                    // 4:4:4 — encode and reconstruct directly into recon_u/v.
                    let (cdc, cac, _, any_cb_ac_nz, _) = encode_chroma_intra16x16_444(
                        frame.u,
                        &mut recon_u,
                        chroma_w,
                        chroma_h,
                        mb_x,
                        mb_y,
                        luma_mode,
                        qp_c,
                    );
                    cb_dc_444 = cdc;
                    cb_ac_scan_444 = cac;
                    let (cdc2, cac2, _, any_cr_ac_nz, _) = encode_chroma_intra16x16_444(
                        frame.v,
                        &mut recon_v,
                        chroma_w,
                        chroma_h,
                        mb_x,
                        mb_y,
                        luma_mode,
                        qp_c,
                    );
                    cr_dc_444 = cdc2;
                    cr_ac_scan_444 = cac2;
                    // §7.4.5.1 / §7.3.5.3 — for 4:4:4 cbp_chroma is always 0
                    // (chroma coded "like luma"). The chroma AC gate is
                    // cbp_luma == 15, so promote cbp_luma when any chroma
                    // plane has nonzero AC, mirroring the CAVLC path (mod.rs
                    // line "if cb_nz || cr_nz { cbp_luma = 15 }").
                    if any_cb_ac_nz || any_cr_ac_nz {
                        cbp_luma = 15;
                    }
                    cbp_chroma = 0;
                } else {
                    // 4:2:0 encode.
                    let (cdc, cac, _, _, any_cb_ac_nz, cb_res) = encode_chroma_intra16x16_420(
                        frame.u,
                        &recon_u,
                        chroma_w,
                        chroma_h,
                        mb_x,
                        mb_y,
                        chroma_mode,
                        qp_c,
                    );
                    let (cdc2, cac2, _, _, any_cr_ac_nz, cr_res) = encode_chroma_intra16x16_420(
                        frame.v,
                        &recon_v,
                        chroma_w,
                        chroma_h,
                        mb_x,
                        mb_y,
                        chroma_mode,
                        qp_c,
                    );
                    cb_dc_420 = cdc;
                    cr_dc_420 = cdc2;
                    cb_ac_scan_420 = cac;
                    cr_ac_scan_420 = cac2;
                    let any_dc_nz =
                        cb_dc_420.iter().any(|&v| v != 0) || cr_dc_420.iter().any(|&v| v != 0);
                    let any_ac_nz = any_cb_ac_nz || any_cr_ac_nz;
                    cbp_chroma = if any_ac_nz {
                        2
                    } else if any_dc_nz {
                        1
                    } else {
                        0
                    };

                    // Reconstruct chroma into recon buffers.
                    let chroma_pred_u =
                        predict_chroma_8x8(chroma_mode, &recon_u, chroma_w, chroma_h, mb_x, mb_y)
                            .unwrap();
                    let chroma_pred_v =
                        predict_chroma_8x8(chroma_mode, &recon_v, chroma_w, chroma_h, mb_x, mb_y)
                            .unwrap();
                    let recon_cb_res = if cbp_chroma == 2 {
                        cb_res
                    } else if cbp_chroma == 1 {
                        let inv_dc =
                            inverse_hadamard_chroma_dc_420(&cb_dc_420, qp_c, &FLAT_4X4_16, 8)
                                .unwrap();
                        let mut out = [0i32; 64];
                        for blk in 0..4usize {
                            let bx = blk % 2;
                            let by = blk / 2;
                            let mut ci = [0i32; 16];
                            ci[0] = inv_dc[blk];
                            let r = inverse_transform_4x4_dc_preserved(&ci, qp_c, &FLAT_4X4_16, 8)
                                .unwrap();
                            for j in 0..4 {
                                for i in 0..4 {
                                    out[(by * 4 + j) * 8 + bx * 4 + i] = r[j * 4 + i];
                                }
                            }
                        }
                        out
                    } else {
                        [0i32; 64]
                    };
                    let recon_cr_res = if cbp_chroma == 2 {
                        cr_res
                    } else if cbp_chroma == 1 {
                        let inv_dc =
                            inverse_hadamard_chroma_dc_420(&cr_dc_420, qp_c, &FLAT_4X4_16, 8)
                                .unwrap();
                        let mut out = [0i32; 64];
                        for blk in 0..4usize {
                            let bx = blk % 2;
                            let by = blk / 2;
                            let mut ci = [0i32; 16];
                            ci[0] = inv_dc[blk];
                            let r = inverse_transform_4x4_dc_preserved(&ci, qp_c, &FLAT_4X4_16, 8)
                                .unwrap();
                            for j in 0..4 {
                                for i in 0..4 {
                                    out[(by * 4 + j) * 8 + bx * 4 + i] = r[j * 4 + i];
                                }
                            }
                        }
                        out
                    } else {
                        [0i32; 64]
                    };
                    for j in 0..8usize {
                        for i in 0..8usize {
                            let cy = mb_y * 8 + j;
                            let cx = mb_x * 8 + i;
                            recon_u[cy * chroma_w + cx] =
                                (chroma_pred_u[j * 8 + i] + recon_cb_res[j * 8 + i]).clamp(0, 255)
                                    as u8;
                            recon_v[cy * chroma_w + cx] =
                                (chroma_pred_v[j * 8 + i] + recon_cr_res[j * 8 + i]).clamp(0, 255)
                                    as u8;
                        }
                    }
                }

                // Reconstruct luma.
                reconstruct_intra16x16_luma(
                    mb_x,
                    mb_y,
                    width,
                    &luma_pred,
                    &luma_dc,
                    &luma_ac_quant,
                    cbp_luma,
                    qp_y,
                    &mut recon_y,
                );

                // ---- CABAC emit ----
                // mb_type. Table 9-36 row index = §7.4.5.1 Intra_16x16
                // group: 1 + group*4 + pred_mode where group is from
                // (cbp_luma, cbp_chroma).
                let group = match (cbp_luma, cbp_chroma) {
                    (0, 0) => 0u32,
                    (0, 1) => 1,
                    (0, 2) => 2,
                    (15, 0) => 3,
                    (15, 1) => 4,
                    (15, 2) => 5,
                    _ => unreachable!(),
                };
                let mb_type_value = 1 + group * 4 + luma_mode as u32;
                encode_mb_type_i(&mut cabac, &mut ctxs, &nb, mb_type_value);

                // intra_chroma_pred_mode — absent for 4:4:4 (§7.3.5.1 note:
                // "not present if ChromaArrayType is equal to 3").
                if cfg.chroma_format_idc != 3 {
                    encode_intra_chroma_pred_mode(&mut cabac, &mut ctxs, &nb, chroma_mode as u32);
                }

                // No coded_block_pattern (carried by mb_type for Intra_16x16).
                // mb_qp_delta — always present for Intra_16x16.
                let qp_delta = 0i32;
                encode_mb_qp_delta(&mut cabac, &mut ctxs, prev_mb_qp_delta_nonzero, qp_delta);
                prev_mb_qp_delta_nonzero = qp_delta != 0;

                // §7.3.5.3 — Luma DC block (Intra_16x16). CBF neighbour
                // context is MB-level (LumaDC).
                let (cbf_a, cbf_b) =
                    cbf_neighbour_mb_level(&grid, mb_x, mb_y, true, CbfPlane::LumaDc);
                let luma_dc_scan = zigzag_scan_4x4(&luma_dc);
                let cbf_luma_dc = encode_residual_block_cabac(
                    &mut cabac,
                    &mut ctxs,
                    BlockType::Luma16x16Dc,
                    &luma_dc_scan,
                    16,
                    Some(cbf_a),
                    Some(cbf_b),
                    false,
                );
                // Update current MB grid slot's running state for downstream
                // neighbour reads (luma AC blocks and chroma DC/AC).
                {
                    let cur = grid.at_mut(mb_x, mb_y);
                    cur.is_intra = true;
                    cur.cbf_luma_dc = cbf_luma_dc;
                }
                let mut cbf_luma_ac = [false; 16];
                if cbp_luma == 15 {
                    for blkz in 0..16usize {
                        // Snapshot current MB info for in-MB neighbour reads.
                        let cur_snapshot = grid.at(mb_x, mb_y).clone();
                        let (ca, cb) = cbf_neighbour_luma_ac(
                            &grid,
                            mb_x,
                            mb_y,
                            &cur_snapshot,
                            true,
                            BlockType::Luma16x16Ac,
                            blkz as u8,
                        );
                        let coded = encode_residual_block_cabac(
                            &mut cabac,
                            &mut ctxs,
                            BlockType::Luma16x16Ac,
                            &luma_ac_scan[blkz][..15],
                            15,
                            Some(ca),
                            Some(cb),
                            false,
                        );
                        cbf_luma_ac[blkz] = coded;
                        grid.at_mut(mb_x, mb_y).cbf_luma_ac[blkz] = coded;
                    }
                }

                // Chroma residual — 4:2:0 or 4:4:4 path.
                if cfg.chroma_format_idc == 3 {
                    // 4:4:4: Cb plane first (CbIntra16x16Dc + CbIntra16x16Ac),
                    // then Cr (CrIntra16x16Dc + CrIntra16x16Ac), per §7.3.5.3.
                    // The cbp_luma gate (cbp_luma == 15) applies here too.
                    let (cbf_cb_dc, _cbf_cb_ac) = emit_444_chroma_plane_cabac(
                        &mut cabac,
                        &mut ctxs,
                        &mut grid,
                        mb_x,
                        mb_y,
                        &cb_dc_444,
                        &cb_ac_scan_444,
                        cbp_luma,
                        BlockType::CbIntra16x16Dc,
                        BlockType::CbIntra16x16Ac,
                    );
                    grid.at_mut(mb_x, mb_y).cbf_cb_dc = cbf_cb_dc;
                    let (cbf_cr_dc, _cbf_cr_ac) = emit_444_chroma_plane_cabac(
                        &mut cabac,
                        &mut ctxs,
                        &mut grid,
                        mb_x,
                        mb_y,
                        &cr_dc_444,
                        &cr_ac_scan_444,
                        cbp_luma,
                        BlockType::CrIntra16x16Dc,
                        BlockType::CrIntra16x16Ac,
                    );
                    grid.at_mut(mb_x, mb_y).cbf_cr_dc = cbf_cr_dc;
                } else {
                    // 4:2:0 chroma residual.
                    if cbp_chroma > 0 {
                        let (cb_a, cb_b) =
                            cbf_neighbour_mb_level(&grid, mb_x, mb_y, true, CbfPlane::CbDc);
                        let cb_dc_coded = encode_residual_block_cabac(
                            &mut cabac,
                            &mut ctxs,
                            BlockType::ChromaDc,
                            &cb_dc_420,
                            4,
                            Some(cb_a),
                            Some(cb_b),
                            false,
                        );
                        grid.at_mut(mb_x, mb_y).cbf_cb_dc = cb_dc_coded;
                        let (cr_a, cr_b) =
                            cbf_neighbour_mb_level(&grid, mb_x, mb_y, true, CbfPlane::CrDc);
                        let cr_dc_coded = encode_residual_block_cabac(
                            &mut cabac,
                            &mut ctxs,
                            BlockType::ChromaDc,
                            &cr_dc_420,
                            4,
                            Some(cr_a),
                            Some(cr_b),
                            false,
                        );
                        grid.at_mut(mb_x, mb_y).cbf_cr_dc = cr_dc_coded;
                    }
                    if cbp_chroma == 2 {
                        for blk in 0..4usize {
                            let cur = grid.at(mb_x, mb_y).clone();
                            let (ca, cb) = cbf_neighbour_chroma_ac(
                                &grid, mb_x, mb_y, &cur, true, blk as u8, false,
                            );
                            let coded = encode_residual_block_cabac(
                                &mut cabac,
                                &mut ctxs,
                                BlockType::ChromaAc,
                                &cb_ac_scan_420[blk][..15],
                                15,
                                Some(ca),
                                Some(cb),
                                false,
                            );
                            grid.at_mut(mb_x, mb_y).cbf_cb_ac[blk] = coded;
                        }
                        for blk in 0..4usize {
                            let cur = grid.at(mb_x, mb_y).clone();
                            let (ca, cb) = cbf_neighbour_chroma_ac(
                                &grid, mb_x, mb_y, &cur, true, blk as u8, true,
                            );
                            let coded = encode_residual_block_cabac(
                                &mut cabac,
                                &mut ctxs,
                                BlockType::ChromaAc,
                                &cr_ac_scan_420[blk][..15],
                                15,
                                Some(ca),
                                Some(cb),
                                false,
                            );
                            grid.at_mut(mb_x, mb_y).cbf_cr_ac[blk] = coded;
                        }
                    }
                }

                // Update grid (final MB state for next-MB neighbour reads).
                let info = grid.at_mut(mb_x, mb_y);
                info.available = true;
                info.is_intra = true;
                info.is_skip = false;
                info.is_i_pcm = false;
                info.is_i_nxn = false;
                info.cbp_luma = cbp_luma;
                info.cbp_chroma = cbp_chroma;
                info.intra_chroma_pred_mode = chroma_mode as u8;

                // §8.7 deblock info.
                let any_ac_per4x4: [bool; 16] = std::array::from_fn(|i| {
                    cbp_luma == 15 && luma_ac_scan[i].iter().any(|&v| v != 0)
                });
                let any_chroma_ac_per4x4_cb: [bool; 4] = if cfg.chroma_format_idc == 3 {
                    // 4:4:4: no separate 4-block chroma; deblock uses luma filter.
                    [false; 4]
                } else {
                    std::array::from_fn(|i| {
                        cbp_chroma == 2 && cb_ac_scan_420[i].iter().any(|&v| v != 0)
                    })
                };
                let any_chroma_ac_per4x4_cr: [bool; 4] = if cfg.chroma_format_idc == 3 {
                    [false; 4]
                } else {
                    std::array::from_fn(|i| {
                        cbp_chroma == 2 && cr_ac_scan_420[i].iter().any(|&v| v != 0)
                    })
                };
                let dbl = MbDeblockInfo {
                    is_intra: true,
                    qp_y,
                    luma_nonzero_4x4: luma_nz_mask_from_blocks(&any_ac_per4x4),
                    chroma_nonzero_4x4: chroma_nz_mask_from_blocks(
                        &any_chroma_ac_per4x4_cb,
                        &any_chroma_ac_per4x4_cr,
                    ),
                    ..Default::default()
                };
                mb_dbl[mb_addr] = dbl;

                // §7.3.4 — end_of_slice_flag (terminate(0) until last MB).
                let is_last = mb_x + 1 == width_mbs && mb_y + 1 == height_mbs;
                encode_end_of_slice_flag(&mut cabac, is_last);
            }
        }
        // §7.3.4 — flush CABAC payload + rbsp_stop_one_bit + alignment.
        // EncodeTerminate(1) on the last MB already finalised the
        // arithmetic state; finish() now writes the rbsp_stop_one_bit.
        let cabac_payload = cabac.finish();
        // Splice the byte-aligned slice header + cabac payload.
        // The slice header writer leaves us at a byte boundary because
        // the cabac_alignment_one_bit padding aligned the writer; we now
        // append the cabac payload bytes directly.
        debug_assert!(sw.byte_aligned());
        let mut slice_rbsp = sw.into_bytes();
        slice_rbsp.extend_from_slice(&cabac_payload);
        stream.extend_from_slice(&build_nal_unit(3, NalUnitType::SliceIdr, &slice_rbsp));

        // Picture-level deblocking (skip for 4:4:4 — disabled in slice header).
        if cfg.chroma_format_idc != 3 {
            deblock_recon(
                cfg.width,
                cfg.height,
                chroma_w as u32,
                chroma_h as u32,
                &mut recon_y,
                &mut recon_u,
                &mut recon_v,
                &mb_dbl,
                0,
                cfg.width / 16,
                cfg.height / 16,
            );
        }

        // Build partition_mvs (all intra → is_intra=true, mv=0, ref=-1).
        let n_parts = width_mbs * height_mbs * 4;
        let partition_mvs = vec![
            FrameRefPartitionMv {
                mv_l0: (0, 0),
                ref_idx_l0: -1,
                is_intra: true,
            };
            n_parts
        ];

        EncodedIdr {
            annex_b: stream,
            recon_y,
            recon_u,
            recon_v,
            recon_width: cfg.width,
            recon_height: cfg.height,
            partition_mvs,
        }
    }

    /// Round-30 — encode a P-slice using **CABAC** entropy coding.
    ///
    /// Strategy per MB: integer-pel ME ±16. If chosen MV equals the
    /// §8.4.1.2 skip-MV AND CBP would be 0 → P_Skip. Otherwise emit
    /// P_L0_16x16 with full residual (luma 4x4 DC+AC + chroma DC ± AC).
    pub fn encode_p_cabac(
        &self,
        frame: &YuvFrame<'_>,
        prev: &EncodedFrameRef<'_>,
        frame_num: u32,
        pic_order_cnt_lsb: u32,
    ) -> EncodedP {
        let cfg = self.config();
        assert!(cfg.cabac);
        assert!(cfg.profile_idc >= 77);
        assert_eq!(cfg.chroma_format_idc, 1);
        let width = cfg.width as usize;
        let height = cfg.height as usize;
        let width_mbs = (cfg.width / 16) as usize;
        let height_mbs = (cfg.height / 16) as usize;
        let chroma_w = width / 2;
        let chroma_h = height / 2;
        let qp_y = cfg.qp;
        // §8.5.8 BD-aware chroma-QP — see encode_idr_cabac; identical reasoning.
        let qp_bd_offset_c = qp_bd_offset(cfg.bit_depth_chroma_minus8);
        let qp_c = qp_y_to_qp_c_with_bd_offset(qp_y, 0, qp_bd_offset_c);

        // SPS / PPS not re-emitted (caller shipped them with IDR).
        let mut stream: Vec<u8> = Vec::new();

        let mut sw = BitWriter::new();
        write_p_slice_header(
            &mut sw,
            &PSliceHeaderConfig {
                first_mb_in_slice: 0,
                slice_type_raw: 5,
                pic_parameter_set_id: 0,
                frame_num,
                frame_num_bits: 8,
                pic_order_cnt_lsb,
                poc_lsb_bits: 8,
                slice_qp_delta: 0,
                disable_deblocking_filter_idc: 0,
                slice_alpha_c0_offset_div2: 0,
                slice_beta_offset_div2: 0,
                nal_ref_idc: 2,
                cabac: Some(CabacSliceParams { cabac_init_idc: 0 }),
            },
        );
        while !sw.byte_aligned() {
            sw.u(1, 1);
        }

        let mut cabac = CabacEncoder::new();
        let mut ctxs = CabacContexts::init(SliceKind::P, Some(0), qp_y).expect("ctx init");

        let mut recon_y = vec![0u8; width * height];
        let mut recon_u = vec![0u8; chroma_w * chroma_h];
        let mut recon_v = vec![0u8; chroma_w * chroma_h];
        let mut grid = CabacEncGrid::new(width_mbs, height_mbs);
        let mut mb_dbl: Vec<MbDeblockInfo> = vec![MbDeblockInfo::default(); width_mbs * height_mbs];
        let mut prev_mb_qp_delta_nonzero = false;

        // Per-MB MV grid for skip-MV derivation. Round-30 simplification:
        // skip-MV is just (0, 0) (the §8.4.1.2 derivation reduces to
        // zero-MV for the all-zero case which dominates static content).
        // For non-trivial MV neighbours we still use 0 — the encoder
        // simply won't pick P_Skip in those cases (cbp will be non-zero
        // or MV will differ).
        for mb_y in 0..height_mbs {
            for mb_x in 0..width_mbs {
                let mb_addr = mb_y * width_mbs + mb_x;

                // Round-32 — real motion estimation (integer + ½-pel +
                // ¼-pel) via the existing `search_quarter_pel_16x16`
                // helper, and §8.4.1.3 / §8.4.1.2 mvp / skip-MV using
                // `cabac_mvp_for_16x16` / `cabac_p_skip_mv` reading the
                // per-MB MV grid we now maintain in `CabacEncMbInfo`.
                let me = search_quarter_pel_16x16(
                    frame.y,
                    width,
                    cfg.width,
                    cfg.height,
                    prev.recon_y,
                    prev.width as usize,
                    prev.width,
                    prev.height,
                    mb_x,
                    mb_y,
                    16,
                    16,
                );
                let chosen_mv = Mv::new(me.mv_x, me.mv_y);
                let mvp = cabac_mvp_for_16x16(&grid, mb_x, mb_y, 0, 0);
                let (_skip_ref, skip_mv) = cabac_p_skip_mv(&grid, mb_x, mb_y);

                // Build inter predictor (luma + chroma).
                let pred_y = build_inter_pred_luma_local(
                    prev.recon_y,
                    prev.width,
                    prev.height,
                    mb_x,
                    mb_y,
                    chosen_mv,
                );
                let pred_u = build_inter_pred_chroma_local(
                    prev.recon_u,
                    chroma_w as u32,
                    chroma_h as u32,
                    mb_x,
                    mb_y,
                    chosen_mv,
                );
                let pred_v = build_inter_pred_chroma_local(
                    prev.recon_v,
                    chroma_w as u32,
                    chroma_h as u32,
                    mb_x,
                    mb_y,
                    chosen_mv,
                );

                // Luma residual + transform + AC quantise (no DC Hadamard
                // for inter MBs).
                let mut residual = [0i32; 256];
                for j in 0..16usize {
                    for i in 0..16usize {
                        let s = frame.y[(mb_y * 16 + j) * width + (mb_x * 16 + i)] as i32;
                        residual[j * 16 + i] = s - pred_y[j * 16 + i];
                    }
                }
                let mut luma_blocks = [[0i32; 16]; 16];
                let mut blk_has_nz = [false; 16];
                let mut recon_residual_y = [0i32; 256];
                for blkz in 0..16usize {
                    let (bx, by) = LUMA_4X4_BLK[blkz];
                    let mut block = [0i32; 16];
                    for j in 0..4 {
                        for i in 0..4 {
                            block[j * 4 + i] = residual[(by * 4 + j) * 16 + bx * 4 + i];
                        }
                    }
                    let coeffs = forward_core_4x4(&block);
                    // Inter MBs quantise both DC and AC together (no Hadamard).
                    // We use the AC quantise function on the *full* 4x4 block
                    // by passing it through quantize_4x4_ac... wait, that
                    // clears DC. We need a full 4x4 quantiser. Re-use
                    // `quantize_4x4` from the transform module.
                    let mut q = crate::encoder::transform::quantize_4x4(&coeffs, qp_y, false);
                    // Round 49 — trellis-quant refinement (informative,
                    // §9): pull small-magnitude coefficients toward zero
                    // when the resulting `D + λ·R` cost is strictly
                    // lower. Decoder-transparent; only the encoded
                    // bitstream shrinks.
                    if cfg.trellis_quant && q.iter().any(|&v| v != 0) {
                        let lambda_q16 = crate::encoder::transform::trellis_lambda_q16(qp_y);
                        q = crate::encoder::transform::trellis_refine_4x4_ac(
                            &block, &coeffs, &q, qp_y, lambda_q16,
                            false, // inter — DC participates
                        );
                    }
                    let scan = zigzag_scan_4x4(&q);
                    luma_blocks[blkz] = scan;
                    if scan.iter().any(|&v| v != 0) {
                        blk_has_nz[blkz] = true;
                    }
                    // Inverse path for recon.
                    let r = crate::transform::inverse_transform_4x4(&q, qp_y, &FLAT_4X4_16, 8)
                        .expect("inverse 4x4");
                    for j in 0..4 {
                        for i in 0..4 {
                            recon_residual_y[(by * 4 + j) * 16 + bx * 4 + i] = r[j * 4 + i];
                        }
                    }
                }
                // Per-8x8 cbp_luma.
                let mut cbp_luma: u8 = 0;
                for blk8 in 0..4usize {
                    if (0..4).any(|sub| blk_has_nz[blk8 * 4 + sub]) {
                        cbp_luma |= 1 << blk8;
                    }
                }

                // Chroma residual.
                let (cb_dc, cb_ac_scan, cb_ac_quant, _cb_dc_nz, cb_ac_nz, cb_recon_res) =
                    encode_chroma_inter_420(frame.u, &pred_u, chroma_w, mb_x, mb_y, qp_c);
                let (cr_dc, cr_ac_scan, _cr_ac_quant, _cr_dc_nz, cr_ac_nz, cr_recon_res) =
                    encode_chroma_inter_420(frame.v, &pred_v, chroma_w, mb_x, mb_y, qp_c);
                let any_dc_nz = cb_dc.iter().any(|&v| v != 0) || cr_dc.iter().any(|&v| v != 0);
                let any_ac_nz = cb_ac_nz || cr_ac_nz;
                let cbp_chroma: u8 = if any_ac_nz {
                    2
                } else if any_dc_nz {
                    1
                } else {
                    0
                };
                let _ = cb_ac_quant;

                // P_Skip eligibility: chosen MV == §8.4.1.2 skip MV AND
                // CBP == 0. Round-32 — with real ME wired the MV check
                // is no longer trivial; many MBs on static / near-static
                // content will still satisfy chosen_mv == skip_mv == (0, 0)
                // and skip out, but moving content forces a mvd emit.
                let is_skip = chosen_mv == skip_mv && cbp_luma == 0 && cbp_chroma == 0;

                let nb = build_neighbour_ctx(&grid, mb_x, mb_y);

                // mb_skip_flag.
                encode_mb_skip_flag(&mut cabac, &mut ctxs, SliceKind::P, &nb, is_skip);

                if is_skip {
                    // §9.3.3.1.1.5 — P_Skip forces the next MB's
                    // mb_qp_delta ctxIdxInc(bin 0) → 0. Mirror the
                    // decoder's slice-walker behaviour.
                    prev_mb_qp_delta_nonzero = false;
                    // Apply pure predictor recon.
                    for j in 0..16usize {
                        for i in 0..16usize {
                            let py = mb_y * 16 + j;
                            let px = mb_x * 16 + i;
                            recon_y[py * width + px] = pred_y[j * 16 + i].clamp(0, 255) as u8;
                        }
                    }
                    for j in 0..8usize {
                        for i in 0..8usize {
                            let cy = mb_y * 8 + j;
                            let cx = mb_x * 8 + i;
                            recon_u[cy * chroma_w + cx] = pred_u[j * 8 + i].clamp(0, 255) as u8;
                            recon_v[cy * chroma_w + cx] = pred_v[j * 8 + i].clamp(0, 255) as u8;
                        }
                    }
                    // Update grid.
                    let info = grid.at_mut(mb_x, mb_y);
                    info.available = true;
                    info.is_intra = false;
                    info.is_skip = true;
                    info.cbp_luma = 0;
                    info.cbp_chroma = 0;
                    // P_Skip — record the §8.4.1.2 skip MV across all 4
                    // 8x8 partitions for downstream neighbour reads.
                    info.ref_idx_l0_8x8 = [0; 4];
                    info.mv_l0_8x8 = [skip_mv; 4];
                    let skip_mv_t = (
                        skip_mv.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                        skip_mv.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                    );
                    mb_dbl[mb_addr] = MbDeblockInfo {
                        is_intra: false,
                        qp_y,
                        luma_nonzero_4x4: 0,
                        chroma_nonzero_4x4: 0,
                        mv_l0: [skip_mv_t; 16],
                        ref_idx_l0: [0; 4],
                        ref_poc_l0: [0; 4],
                        ..Default::default()
                    };
                    let is_last = mb_x + 1 == width_mbs && mb_y + 1 == height_mbs;
                    encode_end_of_slice_flag(&mut cabac, is_last);
                    continue;
                }

                // mb_type = P_L0_16x16 (raw 0 in P-slice Table 7-13).
                encode_mb_type_p(&mut cabac, &mut ctxs, &nb, 0);

                // mvd_l0 (single-ref → no ref_idx_l0 emit). Round-32 —
                // mvp via §8.4.1.3 median (`cabac_mvp_for_16x16`).
                let mvd_x = chosen_mv.x - mvp.x;
                let mvd_y = chosen_mv.y - mvp.y;
                // §9.3.3.1.1.7 — neighbour |mvd| sum for ctxIdxInc(bin 0)
                // of mvd_l0_x / mvd_l0_y. For 4x4 block 0 the A neighbour
                // is left MB block 1 (top-right of the left MB at MB-pixel
                // (15, 0) = block (3, 0) = block index 5 → wait, idx 0
                // top-left maps to block 1 right side. In our simplified
                // encoder every MB has uniform MV so reading ANY 4x4 block
                // of the left/above MB returns the same |mvd|.
                let abs_mvd_left_x = grid
                    .left(mb_x, mb_y)
                    .map(|m| m.abs_mvd_l0_x[0])
                    .unwrap_or(0);
                let abs_mvd_above_x = grid
                    .above(mb_x, mb_y)
                    .map(|m| m.abs_mvd_l0_x[0])
                    .unwrap_or(0);
                let sum_x = abs_mvd_left_x + abs_mvd_above_x;
                encode_mvd_lx(&mut cabac, &mut ctxs, MvdComponent::X, sum_x, mvd_x);
                let abs_mvd_left_y = grid
                    .left(mb_x, mb_y)
                    .map(|m| m.abs_mvd_l0_y[0])
                    .unwrap_or(0);
                let abs_mvd_above_y = grid
                    .above(mb_x, mb_y)
                    .map(|m| m.abs_mvd_l0_y[0])
                    .unwrap_or(0);
                let sum_y = abs_mvd_left_y + abs_mvd_above_y;
                encode_mvd_lx(&mut cabac, &mut ctxs, MvdComponent::Y, sum_y, mvd_y);

                // coded_block_pattern.
                encode_coded_block_pattern(&mut cabac, &mut ctxs, &nb, 1, cbp_luma, cbp_chroma);

                // mb_qp_delta if cbp > 0.
                let needs_qpd = cbp_luma > 0 || cbp_chroma > 0;
                if needs_qpd {
                    let qpd = 0i32;
                    encode_mb_qp_delta(&mut cabac, &mut ctxs, prev_mb_qp_delta_nonzero, qpd);
                    prev_mb_qp_delta_nonzero = qpd != 0;
                } else {
                    // §9.3.3.1.1.5 — when mb_qp_delta is not present, the
                    // decoder treats it as 0 → prev_mb_qp_delta_nonzero
                    // becomes false. Mirror that here so the next MB's
                    // ctxIdxInc(bin 0) lookup is identical.
                    prev_mb_qp_delta_nonzero = false;
                }

                // Mark current MB available so in-MB neighbour reads see
                // the right state (we'll update CBP / CBFs as we go).
                {
                    let cur = grid.at_mut(mb_x, mb_y);
                    cur.is_intra = false;
                    cur.is_skip = false;
                    cur.cbp_luma = cbp_luma;
                    cur.cbp_chroma = cbp_chroma;
                }
                // Luma residual: per 8x8 quadrant gated by cbp_luma.
                for blk8 in 0..4u8 {
                    if (cbp_luma >> blk8) & 1 == 1 {
                        for sub in 0..4u8 {
                            let blkz = (blk8 * 4 + sub) as usize;
                            let cur = grid.at(mb_x, mb_y).clone();
                            let (ca, cb) = cbf_neighbour_luma_ac(
                                &grid,
                                mb_x,
                                mb_y,
                                &cur,
                                false,
                                BlockType::Luma4x4,
                                blkz as u8,
                            );
                            let coded = encode_residual_block_cabac(
                                &mut cabac,
                                &mut ctxs,
                                BlockType::Luma4x4,
                                &luma_blocks[blkz],
                                16,
                                Some(ca),
                                Some(cb),
                                false,
                            );
                            grid.at_mut(mb_x, mb_y).cbf_luma_4x4[blkz] = coded;
                        }
                    }
                }

                if cbp_chroma > 0 {
                    let (cb_a, cb_b) =
                        cbf_neighbour_mb_level(&grid, mb_x, mb_y, false, CbfPlane::CbDc);
                    let cb_dc_coded = encode_residual_block_cabac(
                        &mut cabac,
                        &mut ctxs,
                        BlockType::ChromaDc,
                        &cb_dc,
                        4,
                        Some(cb_a),
                        Some(cb_b),
                        false,
                    );
                    grid.at_mut(mb_x, mb_y).cbf_cb_dc = cb_dc_coded;
                    let (cr_a, cr_b) =
                        cbf_neighbour_mb_level(&grid, mb_x, mb_y, false, CbfPlane::CrDc);
                    let cr_dc_coded = encode_residual_block_cabac(
                        &mut cabac,
                        &mut ctxs,
                        BlockType::ChromaDc,
                        &cr_dc,
                        4,
                        Some(cr_a),
                        Some(cr_b),
                        false,
                    );
                    grid.at_mut(mb_x, mb_y).cbf_cr_dc = cr_dc_coded;
                }
                if cbp_chroma == 2 {
                    for blk in 0..4usize {
                        let cur = grid.at(mb_x, mb_y).clone();
                        let (ca, cb) = cbf_neighbour_chroma_ac(
                            &grid, mb_x, mb_y, &cur, false, blk as u8, false,
                        );
                        let coded = encode_residual_block_cabac(
                            &mut cabac,
                            &mut ctxs,
                            BlockType::ChromaAc,
                            &cb_ac_scan[blk][..15],
                            15,
                            Some(ca),
                            Some(cb),
                            false,
                        );
                        grid.at_mut(mb_x, mb_y).cbf_cb_ac[blk] = coded;
                    }
                    for blk in 0..4usize {
                        let cur = grid.at(mb_x, mb_y).clone();
                        let (ca, cb) = cbf_neighbour_chroma_ac(
                            &grid, mb_x, mb_y, &cur, false, blk as u8, true,
                        );
                        let coded = encode_residual_block_cabac(
                            &mut cabac,
                            &mut ctxs,
                            BlockType::ChromaAc,
                            &cr_ac_scan[blk][..15],
                            15,
                            Some(ca),
                            Some(cb),
                            false,
                        );
                        grid.at_mut(mb_x, mb_y).cbf_cr_ac[blk] = coded;
                    }
                }

                // Reconstruct luma (pred + residual) per 8x8 quadrant.
                for blk8 in 0..4usize {
                    let send = (cbp_luma >> blk8) & 1 == 1;
                    for sub in 0..4usize {
                        let blkz = blk8 * 4 + sub;
                        let (bx, by) = LUMA_4X4_BLK[blkz];
                        for j in 0..4usize {
                            for i in 0..4usize {
                                let p = pred_y[(by * 4 + j) * 16 + bx * 4 + i];
                                let r = if send {
                                    recon_residual_y[(by * 4 + j) * 16 + bx * 4 + i]
                                } else {
                                    0
                                };
                                let py = mb_y * 16 + by * 4 + j;
                                let px = mb_x * 16 + bx * 4 + i;
                                recon_y[py * width + px] = (p + r).clamp(0, 255) as u8;
                            }
                        }
                    }
                }
                // Reconstruct chroma — apply DC-only / DC+AC residual to predictor.
                let recon_cb_residual = if cbp_chroma == 2 {
                    cb_recon_res
                } else if cbp_chroma == 1 {
                    chroma_residual_dc_only(&cb_dc, qp_c)
                } else {
                    [0i32; 64]
                };
                let recon_cr_residual = if cbp_chroma == 2 {
                    cr_recon_res
                } else if cbp_chroma == 1 {
                    chroma_residual_dc_only(&cr_dc, qp_c)
                } else {
                    [0i32; 64]
                };
                for j in 0..8usize {
                    for i in 0..8usize {
                        let cy = mb_y * 8 + j;
                        let cx = mb_x * 8 + i;
                        recon_u[cy * chroma_w + cx] =
                            (pred_u[j * 8 + i] + recon_cb_residual[j * 8 + i]).clamp(0, 255) as u8;
                        recon_v[cy * chroma_w + cx] =
                            (pred_v[j * 8 + i] + recon_cr_residual[j * 8 + i]).clamp(0, 255) as u8;
                    }
                }

                // Update grid.
                let info = grid.at_mut(mb_x, mb_y);
                info.available = true;
                info.is_intra = false;
                info.is_skip = false;
                info.cbp_luma = cbp_luma;
                info.cbp_chroma = cbp_chroma;
                info.abs_mvd_l0_x = [mvd_x.unsigned_abs(); 16];
                info.abs_mvd_l0_y = [mvd_y.unsigned_abs(); 16];
                info.ref_idx_l0_8x8 = [0; 4];
                info.mv_l0_8x8 = [chosen_mv; 4];

                let mv_t = (
                    chosen_mv.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                    chosen_mv.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                );
                let cb_ac_per4x4: [bool; 4] = std::array::from_fn(|i| {
                    cbp_chroma == 2 && cb_ac_scan[i].iter().any(|&v| v != 0)
                });
                let cr_ac_per4x4: [bool; 4] = std::array::from_fn(|i| {
                    cbp_chroma == 2 && cr_ac_scan[i].iter().any(|&v| v != 0)
                });
                mb_dbl[mb_addr] = MbDeblockInfo {
                    is_intra: false,
                    qp_y,
                    luma_nonzero_4x4: luma_nz_mask_from_blocks(&blk_has_nz),
                    chroma_nonzero_4x4: chroma_nz_mask_from_blocks(&cb_ac_per4x4, &cr_ac_per4x4),
                    mv_l0: [mv_t; 16],
                    ref_idx_l0: [0; 4],
                    ref_poc_l0: [0; 4],
                    ..Default::default()
                };

                let is_last = mb_x + 1 == width_mbs && mb_y + 1 == height_mbs;
                encode_end_of_slice_flag(&mut cabac, is_last);
            }
        }

        let cabac_payload = cabac.finish();
        let mut slice_rbsp = sw.into_bytes();
        slice_rbsp.extend_from_slice(&cabac_payload);
        stream.extend_from_slice(&build_nal_unit(2, NalUnitType::SliceNonIdr, &slice_rbsp));

        deblock_recon(
            cfg.width,
            cfg.height,
            chroma_w as u32,
            chroma_h as u32,
            &mut recon_y,
            &mut recon_u,
            &mut recon_v,
            &mb_dbl,
            0,
            cfg.width / 16,
            cfg.height / 16,
        );

        // Round-32 — capture per-8x8 MV state from the grid so a
        // downstream B-slice's colZeroFlag check (§8.4.1.2.2 step 7)
        // sees real MVs, not all-zero. Intra MBs and unencoded slots
        // map to (mv=0, ref=-1, is_intra=true) per §8.4.1.2.1.
        let n_parts = width_mbs * height_mbs * 4;
        let mut partition_mvs = Vec::with_capacity(n_parts);
        for slot in &grid.mbs {
            for part in 0..4usize {
                let (mv, ref_idx, is_intra) = if !slot.available || slot.is_intra {
                    (Mv::ZERO, -1i32, true)
                } else {
                    (slot.mv_l0_8x8[part], slot.ref_idx_l0_8x8[part], false)
                };
                partition_mvs.push(FrameRefPartitionMv {
                    mv_l0: (
                        mv.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                        mv.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                    ),
                    ref_idx_l0: ref_idx.clamp(-1, 127) as i8,
                    is_intra,
                });
            }
        }

        EncodedP {
            annex_b: stream,
            recon_y,
            recon_u,
            recon_v,
            recon_width: cfg.width,
            recon_height: cfg.height,
            frame_num,
            pic_order_cnt_lsb,
            partition_mvs,
        }
    }

    /// Round-31 — encode a non-reference B-slice using **CABAC** entropy
    /// coding.
    ///
    /// Per-MB strategy (intentionally simple, matching the round-30 P
    /// path): compute four candidate inter predictors at forced MV =
    /// (0, 0):
    ///
    /// 1. `B_L0_16x16` — list-0 predictor only.
    /// 2. `B_L1_16x16` — list-1 predictor only.
    /// 3. `B_Bi_16x16` — §8.4.2.3.1 default merge `(L0 + L1 + 1) >> 1`.
    ///
    /// Pick the lowest-SAD candidate, transform/quantise the residual,
    /// emit `mb_skip_flag = 0`, the chosen mb_type, the L0/L1 mvds (each
    /// at zero), `coded_block_pattern`, optional `mb_qp_delta`, and the
    /// residual blocks.
    ///
    /// `B_Skip` and `B_Direct_16x16` are intentionally **not** emitted by
    /// this round. Both rely on the §8.4.1.2 direct-mode derivation which
    /// is non-trivial to mirror in the CABAC encoder; with the encoder /
    /// decoder mvp both fixed at zero, the explicit `B_L*_16x16` types
    /// already deliver the correct samples.
    ///
    /// Caller is responsible for emitting the SPS / PPS / IDR before
    /// calling this method (the same as for [`Encoder::encode_p_cabac`]).
    /// The SPS must declare `max_num_ref_frames >= 2` and
    /// `profile_idc >= 77` (Main); the PPS must have
    /// `entropy_coding_mode_flag = 1`.
    pub fn encode_b_cabac(
        &self,
        frame: &YuvFrame<'_>,
        ref_l0: &EncodedFrameRef<'_>,
        ref_l1: &EncodedFrameRef<'_>,
        frame_num: u32,
        pic_order_cnt_lsb: u32,
    ) -> EncodedB {
        let cfg = self.config();
        assert!(
            cfg.cabac,
            "encode_b_cabac requires EncoderConfig::cabac = true",
        );
        assert!(
            cfg.profile_idc >= 77,
            "CABAC + B-slices require Main profile or higher (got profile_idc={})",
            cfg.profile_idc,
        );
        assert_eq!(cfg.chroma_format_idc, 1, "round-31 CABAC B is 4:2:0 only");
        assert_eq!(frame.width, cfg.width);
        assert_eq!(frame.height, cfg.height);
        assert_eq!(ref_l0.width, cfg.width);
        assert_eq!(ref_l0.height, cfg.height);
        assert_eq!(ref_l1.width, cfg.width);
        assert_eq!(ref_l1.height, cfg.height);

        let width = cfg.width as usize;
        let height = cfg.height as usize;
        let width_mbs = (cfg.width / 16) as usize;
        let height_mbs = (cfg.height / 16) as usize;
        let chroma_w = width / 2;
        let chroma_h = height / 2;

        let qp_y = cfg.qp;
        let qp_bd_offset_c = qp_bd_offset(cfg.bit_depth_chroma_minus8);
        let qp_c = qp_y_to_qp_c_with_bd_offset(qp_y, 0, qp_bd_offset_c);

        let mut stream: Vec<u8> = Vec::new();

        let mut sw = BitWriter::new();
        write_b_slice_header(
            &mut sw,
            &BSliceHeaderConfig {
                first_mb_in_slice: 0,
                slice_type_raw: 6, // B, all-same-type
                pic_parameter_set_id: 0,
                frame_num,
                frame_num_bits: 8,
                pic_order_cnt_lsb,
                poc_lsb_bits: 8,
                // §8.4.1.2.2 spatial direct flag — set 1 (the value is
                // irrelevant: this round never emits B_Direct_16x16 /
                // B_Skip, but the syntax field is mandatory in §7.3.3).
                direct_spatial_mv_pred_flag: true,
                slice_qp_delta: 0,
                disable_deblocking_filter_idc: 0,
                slice_alpha_c0_offset_div2: 0,
                slice_beta_offset_div2: 0,
                nal_ref_idc: 0, // non-reference B
                pred_weight_table: None,
                cabac: Some(CabacSliceParams { cabac_init_idc: 0 }),
            },
        );
        // §7.3.4 — cabac_alignment_one_bit until byte aligned.
        while !sw.byte_aligned() {
            sw.u(1, 1);
        }

        let mut cabac = CabacEncoder::new();
        let mut ctxs = CabacContexts::init(SliceKind::B, Some(0), qp_y).expect("ctx init");

        let mut recon_y = vec![0u8; width * height];
        let mut recon_u = vec![0u8; chroma_w * chroma_h];
        let mut recon_v = vec![0u8; chroma_w * chroma_h];
        let mut grid = CabacEncGrid::new(width_mbs, height_mbs);
        let mut mb_dbl: Vec<MbDeblockInfo> = vec![MbDeblockInfo::default(); width_mbs * height_mbs];
        let mut prev_mb_qp_delta_nonzero = false;

        for mb_y in 0..height_mbs {
            for mb_x in 0..width_mbs {
                let mb_addr = mb_y * width_mbs + mb_x;

                // Round-32 — per-list quarter-pel ME. Same routine as the
                // CAVLC encode_b_mb path; only the reference plane differs.
                let me_l0 = search_quarter_pel_16x16(
                    frame.y,
                    width,
                    cfg.width,
                    cfg.height,
                    ref_l0.recon_y,
                    ref_l0.width as usize,
                    ref_l0.width,
                    ref_l0.height,
                    mb_x,
                    mb_y,
                    16,
                    16,
                );
                let me_l1 = search_quarter_pel_16x16(
                    frame.y,
                    width,
                    cfg.width,
                    cfg.height,
                    ref_l1.recon_y,
                    ref_l1.width as usize,
                    ref_l1.width,
                    ref_l1.height,
                    mb_x,
                    mb_y,
                    16,
                    16,
                );
                let mv_l0 = Mv::new(me_l0.mv_x, me_l0.mv_y);
                let mv_l1 = Mv::new(me_l1.mv_x, me_l1.mv_y);
                // §8.4.1.3 mvp per list (single ref → ref_idx = 0). Used
                // when we emit the explicit `B_L0/L1/Bi_16x16` rows.
                let mvp_l0 = cabac_mvp_for_16x16(&grid, mb_x, mb_y, 0, 0);
                let mvp_l1 = cabac_mvp_for_16x16(&grid, mb_x, mb_y, 1, 0);

                // Build the three explicit-inter predictor candidates at
                // the per-list ME-chosen MVs.
                let pred_y_l0 = build_inter_pred_luma_local(
                    ref_l0.recon_y,
                    ref_l0.width,
                    ref_l0.height,
                    mb_x,
                    mb_y,
                    mv_l0,
                );
                let pred_u_l0 = build_inter_pred_chroma_local(
                    ref_l0.recon_u,
                    chroma_w as u32,
                    chroma_h as u32,
                    mb_x,
                    mb_y,
                    mv_l0,
                );
                let pred_v_l0 = build_inter_pred_chroma_local(
                    ref_l0.recon_v,
                    chroma_w as u32,
                    chroma_h as u32,
                    mb_x,
                    mb_y,
                    mv_l0,
                );
                let pred_y_l1 = build_inter_pred_luma_local(
                    ref_l1.recon_y,
                    ref_l1.width,
                    ref_l1.height,
                    mb_x,
                    mb_y,
                    mv_l1,
                );
                let pred_u_l1 = build_inter_pred_chroma_local(
                    ref_l1.recon_u,
                    chroma_w as u32,
                    chroma_h as u32,
                    mb_x,
                    mb_y,
                    mv_l1,
                );
                let pred_v_l1 = build_inter_pred_chroma_local(
                    ref_l1.recon_v,
                    chroma_w as u32,
                    chroma_h as u32,
                    mb_x,
                    mb_y,
                    mv_l1,
                );
                // §8.4.2.3.1 default-merge bipred = ((L0 + L1 + 1) >> 1)
                // followed by Clip1Y. The two operands are already
                // Clip1Y-ed by the per-list interpolation, so we only
                // need the average + clip.
                let mut pred_y_bi = [0i32; 256];
                for i in 0..256 {
                    pred_y_bi[i] = ((pred_y_l0[i] + pred_y_l1[i] + 1) >> 1).clamp(0, 255);
                }
                let mut pred_u_bi = [0i32; 64];
                let mut pred_v_bi = [0i32; 64];
                for i in 0..64 {
                    pred_u_bi[i] = ((pred_u_l0[i] + pred_u_l1[i] + 1) >> 1).clamp(0, 255);
                    pred_v_bi[i] = ((pred_v_l0[i] + pred_v_l1[i] + 1) >> 1).clamp(0, 255);
                }

                // Round-32 — §8.4.1.2.2 spatial-direct derivation.
                // Build the direct predictor when the derivation produces
                // a uniform per-MB MV across all 4 8x8 partitions
                // (round-32 only emits B_Direct_16x16 / B_Skip in that
                // case; per-8x8 / mixed B_8x8+all-Direct under CABAC is
                // out of scope for this round). When at least one list
                // is used, build the direct predictor and SAD it.
                let direct = cabac_b_spatial_direct_derive(
                    &grid,
                    ref_l1.partition_mvs,
                    width_mbs,
                    mb_x,
                    mb_y,
                );
                let direct_uniform = direct.is_uniform();
                let direct_pred_kind: Option<u32> =
                    match (direct.ref_idx_l0 >= 0, direct.ref_idx_l1 >= 0) {
                        (true, true) => Some(3),  // Bi
                        (true, false) => Some(1), // L0
                        (false, true) => Some(2), // L1
                        (false, false) => None,
                    };
                let (direct_competitive, direct_pred_y, direct_pred_u, direct_pred_v) =
                    match (direct_uniform, direct_pred_kind) {
                        (true, Some(dpk)) => {
                            let dy = match dpk {
                                1 => build_inter_pred_luma_local(
                                    ref_l0.recon_y,
                                    ref_l0.width,
                                    ref_l0.height,
                                    mb_x,
                                    mb_y,
                                    direct.mv_l0(),
                                ),
                                2 => build_inter_pred_luma_local(
                                    ref_l1.recon_y,
                                    ref_l1.width,
                                    ref_l1.height,
                                    mb_x,
                                    mb_y,
                                    direct.mv_l1(),
                                ),
                                _ => {
                                    let l0 = build_inter_pred_luma_local(
                                        ref_l0.recon_y,
                                        ref_l0.width,
                                        ref_l0.height,
                                        mb_x,
                                        mb_y,
                                        direct.mv_l0(),
                                    );
                                    let l1 = build_inter_pred_luma_local(
                                        ref_l1.recon_y,
                                        ref_l1.width,
                                        ref_l1.height,
                                        mb_x,
                                        mb_y,
                                        direct.mv_l1(),
                                    );
                                    let mut bi = [0i32; 256];
                                    for i in 0..256 {
                                        bi[i] = ((l0[i] + l1[i] + 1) >> 1).clamp(0, 255);
                                    }
                                    bi
                                }
                            };
                            let du = match dpk {
                                1 => build_inter_pred_chroma_local(
                                    ref_l0.recon_u,
                                    chroma_w as u32,
                                    chroma_h as u32,
                                    mb_x,
                                    mb_y,
                                    direct.mv_l0(),
                                ),
                                2 => build_inter_pred_chroma_local(
                                    ref_l1.recon_u,
                                    chroma_w as u32,
                                    chroma_h as u32,
                                    mb_x,
                                    mb_y,
                                    direct.mv_l1(),
                                ),
                                _ => {
                                    let l0 = build_inter_pred_chroma_local(
                                        ref_l0.recon_u,
                                        chroma_w as u32,
                                        chroma_h as u32,
                                        mb_x,
                                        mb_y,
                                        direct.mv_l0(),
                                    );
                                    let l1 = build_inter_pred_chroma_local(
                                        ref_l1.recon_u,
                                        chroma_w as u32,
                                        chroma_h as u32,
                                        mb_x,
                                        mb_y,
                                        direct.mv_l1(),
                                    );
                                    let mut bi = [0i32; 64];
                                    for i in 0..64 {
                                        bi[i] = ((l0[i] + l1[i] + 1) >> 1).clamp(0, 255);
                                    }
                                    bi
                                }
                            };
                            let dv = match dpk {
                                1 => build_inter_pred_chroma_local(
                                    ref_l0.recon_v,
                                    chroma_w as u32,
                                    chroma_h as u32,
                                    mb_x,
                                    mb_y,
                                    direct.mv_l0(),
                                ),
                                2 => build_inter_pred_chroma_local(
                                    ref_l1.recon_v,
                                    chroma_w as u32,
                                    chroma_h as u32,
                                    mb_x,
                                    mb_y,
                                    direct.mv_l1(),
                                ),
                                _ => {
                                    let l0 = build_inter_pred_chroma_local(
                                        ref_l0.recon_v,
                                        chroma_w as u32,
                                        chroma_h as u32,
                                        mb_x,
                                        mb_y,
                                        direct.mv_l0(),
                                    );
                                    let l1 = build_inter_pred_chroma_local(
                                        ref_l1.recon_v,
                                        chroma_w as u32,
                                        chroma_h as u32,
                                        mb_x,
                                        mb_y,
                                        direct.mv_l1(),
                                    );
                                    let mut bi = [0i32; 64];
                                    for i in 0..64 {
                                        bi[i] = ((l0[i] + l1[i] + 1) >> 1).clamp(0, 255);
                                    }
                                    bi
                                }
                            };
                            (true, dy, du, dv)
                        }
                        _ => (false, [0i32; 256], [0i32; 64], [0i32; 64]),
                    };

                // Round-475 — `B_8x8` with all four sub_mb_type =
                // `B_Direct_8x8`. This honours the per-8x8 MVs from the
                // §8.4.1.2.2 step-7 colZeroFlag override (which can drive
                // individual quadrants' MVs to (0,0) while leaving others
                // on the median MVpred). Predictor is stitched per-8x8.
                //
                // Only worth competing when the MB-level direct kind
                // is well-defined AND the per-8x8 MVs disagree (else the
                // simpler B_Direct_16x16 wins on syntax cost).
                let (
                    direct_8x8_competitive,
                    direct_8x8_pred_y,
                    direct_8x8_pred_u,
                    direct_8x8_pred_v,
                ) = if direct_pred_kind.is_some() && !direct_uniform {
                    let mut py = [0i32; 256];
                    let mut pu = [0i32; 64];
                    let mut pv = [0i32; 64];
                    for q in 0..4usize {
                        let sub_x = q % 2;
                        let sub_y = q / 2;
                        let mv0 = direct.mv_l0_per_8x8[q];
                        let mv1 = direct.mv_l1_per_8x8[q];
                        let l0_used = direct.ref_idx_l0 >= 0;
                        let l1_used = direct.ref_idx_l1 >= 0;
                        let l0_y = if l0_used {
                            Some(build_inter_pred_luma_8x8(
                                ref_l0.recon_y,
                                ref_l0.width,
                                ref_l0.height,
                                mb_x,
                                mb_y,
                                sub_x,
                                sub_y,
                                mv0,
                            ))
                        } else {
                            None
                        };
                        let l1_y = if l1_used {
                            Some(build_inter_pred_luma_8x8(
                                ref_l1.recon_y,
                                ref_l1.width,
                                ref_l1.height,
                                mb_x,
                                mb_y,
                                sub_x,
                                sub_y,
                                mv1,
                            ))
                        } else {
                            None
                        };
                        let q_y: [i32; 64] = match (l0_used, l1_used) {
                            (true, true) => {
                                let l0 = l0_y.unwrap();
                                let l1 = l1_y.unwrap();
                                let mut out = [0i32; 64];
                                for i in 0..64 {
                                    out[i] = ((l0[i] + l1[i] + 1) >> 1).clamp(0, 255);
                                }
                                out
                            }
                            (true, false) => l0_y.unwrap(),
                            (false, true) => l1_y.unwrap(),
                            (false, false) => [0i32; 64],
                        };
                        for j in 0..8usize {
                            for i in 0..8usize {
                                py[(sub_y * 8 + j) * 16 + sub_x * 8 + i] = q_y[j * 8 + i];
                            }
                        }
                        // Chroma 4x4 quadrant per list.
                        let l0_cb = if l0_used {
                            Some(build_inter_pred_chroma_4x4(
                                ref_l0.recon_u,
                                chroma_w as u32,
                                chroma_h as u32,
                                mb_x,
                                mb_y,
                                sub_x,
                                sub_y,
                                mv0,
                            ))
                        } else {
                            None
                        };
                        let l1_cb = if l1_used {
                            Some(build_inter_pred_chroma_4x4(
                                ref_l1.recon_u,
                                chroma_w as u32,
                                chroma_h as u32,
                                mb_x,
                                mb_y,
                                sub_x,
                                sub_y,
                                mv1,
                            ))
                        } else {
                            None
                        };
                        let q_cb: [i32; 16] = match (l0_used, l1_used) {
                            (true, true) => {
                                let l0 = l0_cb.unwrap();
                                let l1 = l1_cb.unwrap();
                                let mut out = [0i32; 16];
                                for i in 0..16 {
                                    out[i] = ((l0[i] + l1[i] + 1) >> 1).clamp(0, 255);
                                }
                                out
                            }
                            (true, false) => l0_cb.unwrap(),
                            (false, true) => l1_cb.unwrap(),
                            (false, false) => [0i32; 16],
                        };
                        let l0_cr = if l0_used {
                            Some(build_inter_pred_chroma_4x4(
                                ref_l0.recon_v,
                                chroma_w as u32,
                                chroma_h as u32,
                                mb_x,
                                mb_y,
                                sub_x,
                                sub_y,
                                mv0,
                            ))
                        } else {
                            None
                        };
                        let l1_cr = if l1_used {
                            Some(build_inter_pred_chroma_4x4(
                                ref_l1.recon_v,
                                chroma_w as u32,
                                chroma_h as u32,
                                mb_x,
                                mb_y,
                                sub_x,
                                sub_y,
                                mv1,
                            ))
                        } else {
                            None
                        };
                        let q_cr: [i32; 16] = match (l0_used, l1_used) {
                            (true, true) => {
                                let l0 = l0_cr.unwrap();
                                let l1 = l1_cr.unwrap();
                                let mut out = [0i32; 16];
                                for i in 0..16 {
                                    out[i] = ((l0[i] + l1[i] + 1) >> 1).clamp(0, 255);
                                }
                                out
                            }
                            (true, false) => l0_cr.unwrap(),
                            (false, true) => l1_cr.unwrap(),
                            (false, false) => [0i32; 16],
                        };
                        for j in 0..4usize {
                            for i in 0..4usize {
                                pu[(sub_y * 4 + j) * 8 + sub_x * 4 + i] = q_cb[j * 4 + i];
                                pv[(sub_y * 4 + j) * 8 + sub_x * 4 + i] = q_cr[j * 4 + i];
                            }
                        }
                    }
                    (true, py, pu, pv)
                } else {
                    (false, [0i32; 256], [0i32; 64], [0i32; 64])
                };

                // SAD against source (luma only — chroma trail-along).
                let sad = |pred: &[i32; 256]| -> u64 {
                    let mut s: u64 = 0;
                    for j in 0..16usize {
                        for i in 0..16usize {
                            let v = frame.y[(mb_y * 16 + j) * width + (mb_x * 16 + i)] as i32;
                            s += (v - pred[j * 16 + i]).unsigned_abs() as u64;
                        }
                    }
                    s
                };
                let sad_l0 = sad(&pred_y_l0);
                let sad_l1 = sad(&pred_y_l1);
                let sad_bi = sad(&pred_y_bi);
                let sad_direct = if direct_competitive {
                    sad(&direct_pred_y)
                } else {
                    u64::MAX
                };
                let sad_direct_8x8 = if direct_8x8_competitive {
                    sad(&direct_8x8_pred_y)
                } else {
                    u64::MAX
                };

                // Lagrangian rate bias for direct mode. Direct emits
                // ~1 bit (mb_type=0), explicit-inter rows emit ~12-20
                // bits (mvds + mb_type). Bias the comparison so direct
                // wins on near-tie SADs. λ scales coarsely with QP.
                let lambda = (qp_y / 6).clamp(0, 8) as u64;
                let direct_bias_sad: u64 = (lambda + 1) * 16;

                // Round-475 — also consider direct_8x8 (B_8x8 + 4×
                // B_Direct_8x8). It carries ~5 bits of mb_type/sub_mb_type
                // overhead compared to ~1 bit for B_Direct_16x16, so we
                // bias it modestly.
                let direct_8x8_bias_sad: u64 = (lambda + 1) * 4;
                let prefer_direct_8x8 = direct_8x8_competitive
                    && sad_direct_8x8.saturating_add(direct_8x8_bias_sad) < sad_direct
                    && sad_direct_8x8 <= sad_l0
                    && sad_direct_8x8 <= sad_l1
                    && sad_direct_8x8 <= sad_bi;

                // Pick best — prefer direct when its SAD plus bias still
                // beats the explicit candidates' SAD; otherwise pick the
                // best explicit row by raw SAD.
                let prefer_direct = !prefer_direct_8x8
                    && direct_competitive
                    && sad_direct <= sad_l0.saturating_add(direct_bias_sad)
                    && sad_direct <= sad_l1.saturating_add(direct_bias_sad)
                    && sad_direct <= sad_bi.saturating_add(direct_bias_sad);
                // chosen_mb_type: 0 = B_Direct_16x16, 1 = B_L0, 2 = B_L1,
                // 3 = B_Bi. (22 = B_8x8 carried separately by
                // `is_direct_per_8x8` flag.)
                let (chosen_mb_type, pred_y_16, pred_u_16, pred_v_16) = if prefer_direct_8x8 {
                    // Use B_8x8 all-Direct predictor; chosen_mb_type=22
                    // is the marker emitted later, but for the SAD path
                    // we still use the per-8x8 predictor as the base.
                    (
                        22u32,
                        direct_8x8_pred_y,
                        direct_8x8_pred_u,
                        direct_8x8_pred_v,
                    )
                } else if prefer_direct {
                    (0u32, direct_pred_y, direct_pred_u, direct_pred_v)
                } else if sad_l0 <= sad_l1 && sad_l0 <= sad_bi {
                    (1u32, pred_y_l0, pred_u_l0, pred_v_l0) // B_L0_16x16
                } else if sad_l1 <= sad_bi {
                    (2u32, pred_y_l1, pred_u_l1, pred_v_l1) // B_L1_16x16
                } else {
                    (3u32, pred_y_bi, pred_u_bi, pred_v_bi) // B_Bi_16x16
                };
                let mut is_direct = chosen_mb_type == 0;
                let mut is_direct_per_8x8 = chosen_mb_type == 22;

                // -------------------------------------------------------
                // Round-475 — B 16x8 / 8x16 partition decision.
                //
                // Run a qpel ME per 8x8 cell × per list (4 cells × 2
                // lists), use those plus the 16x16 ME pick as candidates
                // for each partition's per-list MV, and pick the best
                // (mode, mv_l0, mv_l1) per partition with §6.4 SAD.
                //
                // If a partition shape's combined SAD beats the best
                // 16x16 candidate by more than the Lagrangian bias,
                // override to that partition mode (mb_type ∈ 4..21 per
                // Table 7-14). Direct / B_Skip stays exclusive — its bin
                // cost is ~1 bit so we don't override it with partitions.
                // -------------------------------------------------------
                let best_16x16_sad = match chosen_mb_type {
                    0 => sad_direct,
                    1 => sad_l0,
                    2 => sad_l1,
                    3 => sad_bi,
                    _ => unreachable!(),
                };
                // Per-cell qpel ME on each list (4 cells × 2 lists).
                let mut me_l0_cells = [Mv::ZERO; 4];
                let mut me_l1_cells = [Mv::ZERO; 4];
                for cell in 0..4usize {
                    let sub_x = cell % 2;
                    let sub_y = cell / 2;
                    let r0 = search_quarter_pel_8x8(
                        frame.y,
                        width,
                        cfg.width,
                        cfg.height,
                        ref_l0.recon_y,
                        ref_l0.width as usize,
                        ref_l0.width,
                        ref_l0.height,
                        mb_x,
                        mb_y,
                        sub_x,
                        sub_y,
                        4,
                        4,
                    );
                    let r1 = search_quarter_pel_8x8(
                        frame.y,
                        width,
                        cfg.width,
                        cfg.height,
                        ref_l1.recon_y,
                        ref_l1.width as usize,
                        ref_l1.width,
                        ref_l1.height,
                        mb_x,
                        mb_y,
                        sub_x,
                        sub_y,
                        4,
                        4,
                    );
                    me_l0_cells[cell] = Mv::new(r0.mv_x, r0.mv_y);
                    me_l1_cells[cell] = Mv::new(r1.mv_x, r1.mv_y);
                }
                // 16x8: top covers cells 0/1; bottom covers 2/3.
                let cand_top_l0 = [mv_l0, me_l0_cells[0], me_l0_cells[1]];
                let cand_top_l1 = [mv_l1, me_l1_cells[0], me_l1_cells[1]];
                let cand_bot_l0 = [mv_l0, me_l0_cells[2], me_l0_cells[3]];
                let cand_bot_l1 = [mv_l1, me_l1_cells[2], me_l1_cells[3]];
                let (top_mode, top_mv_l0, top_mv_l1, sad_top) =
                    best_partition_mode_b(&cand_top_l0, &cand_top_l1, |pred, mv0, mv1| {
                        partition_sad_b_16x8(
                            pred, frame.y, width, ref_l0, ref_l1, mb_x, mb_y, 0, mv0, mv1,
                        )
                    });
                let (bot_mode, bot_mv_l0, bot_mv_l1, sad_bot) =
                    best_partition_mode_b(&cand_bot_l0, &cand_bot_l1, |pred, mv0, mv1| {
                        partition_sad_b_16x8(
                            pred, frame.y, width, ref_l0, ref_l1, mb_x, mb_y, 8, mv0, mv1,
                        )
                    });
                let sad_16x8_total = (sad_top as u64).saturating_add(sad_bot as u64);
                // 8x16: left = cells 0/2; right = cells 1/3.
                let cand_left_l0 = [mv_l0, me_l0_cells[0], me_l0_cells[2]];
                let cand_left_l1 = [mv_l1, me_l1_cells[0], me_l1_cells[2]];
                let cand_right_l0 = [mv_l0, me_l0_cells[1], me_l0_cells[3]];
                let cand_right_l1 = [mv_l1, me_l1_cells[1], me_l1_cells[3]];
                let (left_mode, left_mv_l0, left_mv_l1, sad_left) =
                    best_partition_mode_b(&cand_left_l0, &cand_left_l1, |pred, mv0, mv1| {
                        partition_sad_b_8x16(
                            pred, frame.y, width, ref_l0, ref_l1, mb_x, mb_y, 0, mv0, mv1,
                        )
                    });
                let (right_mode, right_mv_l0, right_mv_l1, sad_right) =
                    best_partition_mode_b(&cand_right_l0, &cand_right_l1, |pred, mv0, mv1| {
                        partition_sad_b_8x16(
                            pred, frame.y, width, ref_l0, ref_l1, mb_x, mb_y, 8, mv0, mv1,
                        )
                    });
                let sad_8x16_total = (sad_left as u64).saturating_add(sad_right as u64);

                // Lagrangian rate penalty for partition modes vs 16x16.
                let part_bias_sad: u64 = (lambda + 1) * 8;
                let part_threshold = best_16x16_sad.saturating_sub(part_bias_sad);
                let (part_winner, part_winner_sad) = if sad_16x8_total <= sad_8x16_total {
                    (PartitionShape::P16x8, sad_16x8_total)
                } else {
                    (PartitionShape::P8x16, sad_8x16_total)
                };

                // Don't override Direct: it may collapse to B_Skip (no
                // bins). Partition override only beats explicit-inter
                // 16x16 candidates.
                let mut partition_choice: Option<(
                    PartitionShape,
                    BPartPred,
                    BPartPred,
                    Mv,
                    Mv,
                    Mv,
                    Mv,
                )> = None;
                if !is_direct && !is_direct_per_8x8 && part_winner_sad < part_threshold {
                    partition_choice = Some(match part_winner {
                        PartitionShape::P16x8 => (
                            PartitionShape::P16x8,
                            top_mode,
                            bot_mode,
                            top_mv_l0,
                            top_mv_l1,
                            bot_mv_l0,
                            bot_mv_l1,
                        ),
                        PartitionShape::P8x16 => (
                            PartitionShape::P8x16,
                            left_mode,
                            right_mode,
                            left_mv_l0,
                            left_mv_l1,
                            right_mv_l0,
                            right_mv_l1,
                        ),
                    });
                    is_direct = false;
                }

                // -------------------------------------------------------
                // Round-475 — per-cell mixed `B_8x8` decision.
                //
                // Per-cell pick from {Direct, L0, L1, Bi}. Each non-
                // Direct cell pays ~6-12 bits of MB syntax (sub_mb_type
                // + per-cell mvd_l0/_l1) — `pick_best_b8x8_cell` already
                // bakes that rate cost into its Lagrangian score. The
                // mixed mode is only worth emitting when at least one
                // cell picks non-Direct AND the total cell SAD beats the
                // best alternative by enough to amortise the +12-bit MB
                // header cost vs B_Direct_16x16 (or 0 vs B_8x8+all-
                // Direct). Mirror of the CAVLC `Mixed8x8` selector.
                // -------------------------------------------------------
                let mut mixed_choice: Option<(
                    [BSubMbCell; 4],
                    [Mv; 4], // per-cell mv L0
                    [Mv; 4], // per-cell mv L1
                )> = None;
                if direct_pred_kind.is_some() {
                    let mut mixed_cells = [BSubMbCell::Direct; 4];
                    let mut mixed_mv_l0 = [Mv::ZERO; 4];
                    let mut mixed_mv_l1 = [Mv::ZERO; 4];
                    let mut mixed_pred_per_cell = [[0i32; 64]; 4];
                    let mut mixed_total_sad: u32 = 0;
                    for cell in 0..4usize {
                        let sub_x = cell % 2;
                        let sub_y = cell / 2;
                        // Build per-cell Direct predictor directly from
                        // §8.4.1.2.2-derived MVs (don't slice from
                        // `direct_8x8_pred_y` — that's only populated
                        // when `!direct_uniform`).
                        let l0_used = direct.ref_idx_l0 >= 0;
                        let l1_used = direct.ref_idx_l1 >= 0;
                        let mv0d = direct.mv_l0_per_8x8[cell];
                        let mv1d = direct.mv_l1_per_8x8[cell];
                        let direct_cell_pred = match (l0_used, l1_used) {
                            (true, true) => {
                                let a = build_inter_pred_luma_8x8(
                                    ref_l0.recon_y,
                                    ref_l0.width,
                                    ref_l0.height,
                                    mb_x,
                                    mb_y,
                                    sub_x,
                                    sub_y,
                                    mv0d,
                                );
                                let b = build_inter_pred_luma_8x8(
                                    ref_l1.recon_y,
                                    ref_l1.width,
                                    ref_l1.height,
                                    mb_x,
                                    mb_y,
                                    sub_x,
                                    sub_y,
                                    mv1d,
                                );
                                let mut out = [0i32; 64];
                                for i in 0..64 {
                                    out[i] = ((a[i] + b[i] + 1) >> 1).clamp(0, 255);
                                }
                                out
                            }
                            (true, false) => build_inter_pred_luma_8x8(
                                ref_l0.recon_y,
                                ref_l0.width,
                                ref_l0.height,
                                mb_x,
                                mb_y,
                                sub_x,
                                sub_y,
                                mv0d,
                            ),
                            (false, true) => build_inter_pred_luma_8x8(
                                ref_l1.recon_y,
                                ref_l1.width,
                                ref_l1.height,
                                mb_x,
                                mb_y,
                                sub_x,
                                sub_y,
                                mv1d,
                            ),
                            (false, false) => [0i32; 64],
                        };
                        let (k, m0, m1, p, s) = pick_best_b8x8_cell(
                            frame.y,
                            width,
                            ref_l0,
                            ref_l1,
                            mb_x,
                            mb_y,
                            sub_x,
                            sub_y,
                            &direct_cell_pred,
                            me_l0_cells[cell],
                            me_l1_cells[cell],
                            qp_y,
                        );
                        mixed_cells[cell] = k;
                        mixed_mv_l0[cell] = m0;
                        mixed_mv_l1[cell] = m1;
                        mixed_pred_per_cell[cell] = p;
                        mixed_total_sad = mixed_total_sad.saturating_add(s);
                    }
                    let mixed_has_non_direct = mixed_cells.iter().any(|&c| c != BSubMbCell::Direct);
                    if mixed_has_non_direct {
                        // Compare mixed-total-SAD vs the best current
                        // candidate (after partition / direct decisions).
                        // Mixed costs +12 bits MB header vs B_Direct_16x16
                        // and 0 vs DirectPer8x8.
                        let mixed_mb_header_bias_sad: u64 = if is_direct_per_8x8 {
                            0
                        } else {
                            (lambda + 1) * 12
                        };
                        let mixed_tiebreak_bias: u64 = (lambda + 1) * 8;
                        let mixed_score = (mixed_total_sad as u64)
                            .saturating_add(mixed_mb_header_bias_sad)
                            .saturating_add(mixed_tiebreak_bias);
                        // Reference SAD from current best.
                        let current_best_sad: u64 = if partition_choice.is_some() {
                            part_winner_sad
                        } else {
                            best_16x16_sad
                        };
                        if mixed_score < current_best_sad {
                            // Build per-cell predictors (luma + chroma)
                            // and stitch them into the MB-level buffers.
                            // Done after the gating to avoid wasted work.
                            mixed_choice = Some((mixed_cells, mixed_mv_l0, mixed_mv_l1));
                            // Mixed wins; clear partition / direct flags.
                            partition_choice = None;
                            is_direct = false;
                            is_direct_per_8x8 = false;
                        }
                    }
                }

                // Build the working luma + chroma predictors. For
                // partition modes, build the composite from the two
                // per-partition predictors; for mixed B_8x8, stitch the
                // four cell predictors; otherwise use the 16x16
                // predictor selected above.
                let (pred_y, pred_u, pred_v) = if let Some((cells, mv0s, mv1s)) = mixed_choice {
                    let mut py = [0i32; 256];
                    let mut pu = [0i32; 64];
                    let mut pv = [0i32; 64];
                    let chroma_w_l0 = ref_l0.width / 2;
                    let chroma_h_l0 = ref_l0.height / 2;
                    let chroma_w_l1 = ref_l1.width / 2;
                    let chroma_h_l1 = ref_l1.height / 2;
                    for cell in 0..4usize {
                        let sub_x = cell % 2;
                        let sub_y = cell / 2;
                        // Luma 8x8.
                        let cell_y = match cells[cell] {
                            BSubMbCell::Direct => {
                                // Direct cell uses §8.4.1.2.2-derived
                                // per-8x8 MVs. Build per-cell predictor
                                // directly (we can't slice from
                                // `direct_8x8_pred_y` because that's
                                // only populated when `!direct_uniform`,
                                // and mixed mode is gated on
                                // `direct_pred_kind.is_some()` only).
                                let l0_used = direct.ref_idx_l0 >= 0;
                                let l1_used = direct.ref_idx_l1 >= 0;
                                let mv0 = direct.mv_l0_per_8x8[cell];
                                let mv1 = direct.mv_l1_per_8x8[cell];
                                match (l0_used, l1_used) {
                                    (true, true) => {
                                        let a = build_inter_pred_luma_8x8(
                                            ref_l0.recon_y,
                                            ref_l0.width,
                                            ref_l0.height,
                                            mb_x,
                                            mb_y,
                                            sub_x,
                                            sub_y,
                                            mv0,
                                        );
                                        let b = build_inter_pred_luma_8x8(
                                            ref_l1.recon_y,
                                            ref_l1.width,
                                            ref_l1.height,
                                            mb_x,
                                            mb_y,
                                            sub_x,
                                            sub_y,
                                            mv1,
                                        );
                                        let mut out = [0i32; 64];
                                        for i in 0..64 {
                                            out[i] = ((a[i] + b[i] + 1) >> 1).clamp(0, 255);
                                        }
                                        out
                                    }
                                    (true, false) => build_inter_pred_luma_8x8(
                                        ref_l0.recon_y,
                                        ref_l0.width,
                                        ref_l0.height,
                                        mb_x,
                                        mb_y,
                                        sub_x,
                                        sub_y,
                                        mv0,
                                    ),
                                    (false, true) => build_inter_pred_luma_8x8(
                                        ref_l1.recon_y,
                                        ref_l1.width,
                                        ref_l1.height,
                                        mb_x,
                                        mb_y,
                                        sub_x,
                                        sub_y,
                                        mv1,
                                    ),
                                    (false, false) => [0i32; 64],
                                }
                            }
                            other => build_b_explicit_8x8_cell_luma(
                                other, ref_l0, ref_l1, mb_x, mb_y, sub_x, sub_y, mv0s[cell],
                                mv1s[cell],
                            ),
                        };
                        for j in 0..8usize {
                            for i in 0..8usize {
                                py[(sub_y * 8 + j) * 16 + sub_x * 8 + i] = cell_y[j * 8 + i];
                            }
                        }
                        // Chroma 4x4 per cell.
                        let (mv0_eff, mv1_eff, chroma_kind) = match cells[cell] {
                            BSubMbCell::Direct => {
                                let kind = match (direct.ref_idx_l0 >= 0, direct.ref_idx_l1 >= 0) {
                                    (true, true) => BSubMbCell::Bi,
                                    (true, false) => BSubMbCell::L0,
                                    (false, true) => BSubMbCell::L1,
                                    (false, false) => BSubMbCell::L0, // defensive
                                };
                                (direct.mv_l0_per_8x8[cell], direct.mv_l1_per_8x8[cell], kind)
                            }
                            other => (mv0s[cell], mv1s[cell], other),
                        };
                        let cu = build_b_explicit_4x4_cell_chroma(
                            chroma_kind,
                            ref_l0.recon_u,
                            ref_l1.recon_u,
                            chroma_w_l0,
                            chroma_h_l0,
                            chroma_w_l1,
                            chroma_h_l1,
                            mb_x,
                            mb_y,
                            sub_x,
                            sub_y,
                            mv0_eff,
                            mv1_eff,
                        );
                        let cv = build_b_explicit_4x4_cell_chroma(
                            chroma_kind,
                            ref_l0.recon_v,
                            ref_l1.recon_v,
                            chroma_w_l0,
                            chroma_h_l0,
                            chroma_w_l1,
                            chroma_h_l1,
                            mb_x,
                            mb_y,
                            sub_x,
                            sub_y,
                            mv0_eff,
                            mv1_eff,
                        );
                        for j in 0..4usize {
                            for i in 0..4usize {
                                pu[(sub_y * 4 + j) * 8 + sub_x * 4 + i] = cu[j * 4 + i];
                                pv[(sub_y * 4 + j) * 8 + sub_x * 4 + i] = cv[j * 4 + i];
                            }
                        }
                    }
                    (py, pu, pv)
                } else if let Some((shape, mode_a, mode_b, mv0_a, mv1_a, mv0_b, mv1_b)) =
                    partition_choice
                {
                    let mut py = [0i32; 256];
                    let mut pu = [0i32; 64];
                    let mut pv = [0i32; 64];
                    match shape {
                        PartitionShape::P16x8 => {
                            let p0 = build_b_partition_pred_luma_16x8(
                                mode_a, ref_l0, ref_l1, mb_x, mb_y, 0, mv0_a, mv1_a,
                            );
                            let p1 = build_b_partition_pred_luma_16x8(
                                mode_b, ref_l0, ref_l1, mb_x, mb_y, 8, mv0_b, mv1_b,
                            );
                            for j in 0..8usize {
                                for i in 0..16usize {
                                    py[j * 16 + i] = p0[j * 16 + i];
                                    py[(8 + j) * 16 + i] = p1[j * 16 + i];
                                }
                            }
                            let (cb0, cr0) = build_b_partition_pred_chroma_16x8(
                                mode_a, ref_l0, ref_l1, mb_x, mb_y, 0, mv0_a, mv1_a,
                            );
                            let (cb1, cr1) = build_b_partition_pred_chroma_16x8(
                                mode_b, ref_l0, ref_l1, mb_x, mb_y, 8, mv0_b, mv1_b,
                            );
                            for j in 0..4usize {
                                for i in 0..8usize {
                                    pu[j * 8 + i] = cb0[j * 8 + i];
                                    pv[j * 8 + i] = cr0[j * 8 + i];
                                    pu[(4 + j) * 8 + i] = cb1[j * 8 + i];
                                    pv[(4 + j) * 8 + i] = cr1[j * 8 + i];
                                }
                            }
                        }
                        PartitionShape::P8x16 => {
                            let p0 = build_b_partition_pred_luma_8x16(
                                mode_a, ref_l0, ref_l1, mb_x, mb_y, 0, mv0_a, mv1_a,
                            );
                            let p1 = build_b_partition_pred_luma_8x16(
                                mode_b, ref_l0, ref_l1, mb_x, mb_y, 8, mv0_b, mv1_b,
                            );
                            for j in 0..16usize {
                                for i in 0..8usize {
                                    py[j * 16 + i] = p0[j * 8 + i];
                                    py[j * 16 + 8 + i] = p1[j * 8 + i];
                                }
                            }
                            let (cb0, cr0) = build_b_partition_pred_chroma_8x16(
                                mode_a, ref_l0, ref_l1, mb_x, mb_y, 0, mv0_a, mv1_a,
                            );
                            let (cb1, cr1) = build_b_partition_pred_chroma_8x16(
                                mode_b, ref_l0, ref_l1, mb_x, mb_y, 8, mv0_b, mv1_b,
                            );
                            for j in 0..8usize {
                                for i in 0..4usize {
                                    pu[j * 8 + i] = cb0[j * 4 + i];
                                    pv[j * 8 + i] = cr0[j * 4 + i];
                                    pu[j * 8 + 4 + i] = cb1[j * 4 + i];
                                    pv[j * 8 + 4 + i] = cr1[j * 4 + i];
                                }
                            }
                        }
                    }
                    (py, pu, pv)
                } else {
                    (pred_y_16, pred_u_16, pred_v_16)
                };

                // Luma residual + per-4x4 quantise.
                let mut residual = [0i32; 256];
                for j in 0..16usize {
                    for i in 0..16usize {
                        let s = frame.y[(mb_y * 16 + j) * width + (mb_x * 16 + i)] as i32;
                        residual[j * 16 + i] = s - pred_y[j * 16 + i];
                    }
                }
                let mut luma_blocks = [[0i32; 16]; 16];
                let mut blk_has_nz = [false; 16];
                let mut recon_residual_y = [0i32; 256];
                for blkz in 0..16usize {
                    let (bx, by) = LUMA_4X4_BLK[blkz];
                    let mut block = [0i32; 16];
                    for j in 0..4 {
                        for i in 0..4 {
                            block[j * 4 + i] = residual[(by * 4 + j) * 16 + bx * 4 + i];
                        }
                    }
                    let coeffs = forward_core_4x4(&block);
                    let mut q = crate::encoder::transform::quantize_4x4(&coeffs, qp_y, false);
                    // Round 49 — trellis-quant refinement on the B-slice
                    // inter luma path. Mirrors the P-CABAC call above.
                    if cfg.trellis_quant && q.iter().any(|&v| v != 0) {
                        let lambda_q16 = crate::encoder::transform::trellis_lambda_q16(qp_y);
                        q = crate::encoder::transform::trellis_refine_4x4_ac(
                            &block, &coeffs, &q, qp_y, lambda_q16, false,
                        );
                    }
                    let scan = zigzag_scan_4x4(&q);
                    luma_blocks[blkz] = scan;
                    if scan.iter().any(|&v| v != 0) {
                        blk_has_nz[blkz] = true;
                    }
                    let r = crate::transform::inverse_transform_4x4(&q, qp_y, &FLAT_4X4_16, 8)
                        .expect("inverse 4x4");
                    for j in 0..4 {
                        for i in 0..4 {
                            recon_residual_y[(by * 4 + j) * 16 + bx * 4 + i] = r[j * 4 + i];
                        }
                    }
                }
                let mut cbp_luma: u8 = 0;
                for blk8 in 0..4usize {
                    if (0..4).any(|sub| blk_has_nz[blk8 * 4 + sub]) {
                        cbp_luma |= 1 << blk8;
                    }
                }

                // Chroma residual.
                let (cb_dc, cb_ac_scan, _cb_ac_quant, _cb_dc_nz, cb_ac_nz, cb_recon_res) =
                    encode_chroma_inter_420(frame.u, &pred_u, chroma_w, mb_x, mb_y, qp_c);
                let (cr_dc, cr_ac_scan, _cr_ac_quant, _cr_dc_nz, cr_ac_nz, cr_recon_res) =
                    encode_chroma_inter_420(frame.v, &pred_v, chroma_w, mb_x, mb_y, qp_c);
                let any_dc_nz = cb_dc.iter().any(|&v| v != 0) || cr_dc.iter().any(|&v| v != 0);
                let any_ac_nz = cb_ac_nz || cr_ac_nz;
                let cbp_chroma: u8 = if any_ac_nz {
                    2
                } else if any_dc_nz {
                    1
                } else {
                    0
                };

                let nb = build_neighbour_ctx(&grid, mb_x, mb_y);

                // Round-32 — B_Skip emission (§7.3.4 mb_skip_flag = 1).
                // Eligible iff:
                //   * direct mode is the chosen winner above, AND
                //   * residual (luma + chroma) quantises to all-zero.
                //
                // B_Skip carries NO further MB syntax — the decoder
                // re-derives MVs / refIdxs via §8.4.1.2.2 spatial
                // direct, applies the predictor, and moves on. The
                // encoder's cbp_luma / cbp_chroma must therefore be 0.
                let is_b_skip = is_direct && cbp_luma == 0 && cbp_chroma == 0;

                // mb_skip_flag.
                encode_mb_skip_flag(&mut cabac, &mut ctxs, SliceKind::B, &nb, is_b_skip);

                if is_b_skip {
                    // §9.3.3.1.1.5 — skipped MBs reset
                    // prev_mb_qp_delta_nonzero (no qpd is emitted).
                    prev_mb_qp_delta_nonzero = false;

                    // Apply the direct predictor to the recon (no residual).
                    for j in 0..16usize {
                        for i in 0..16usize {
                            let py = mb_y * 16 + j;
                            let px = mb_x * 16 + i;
                            recon_y[py * width + px] = pred_y[j * 16 + i].clamp(0, 255) as u8;
                        }
                    }
                    for j in 0..8usize {
                        for i in 0..8usize {
                            let cy = mb_y * 8 + j;
                            let cx = mb_x * 8 + i;
                            recon_u[cy * chroma_w + cx] = pred_u[j * 8 + i].clamp(0, 255) as u8;
                            recon_v[cy * chroma_w + cx] = pred_v[j * 8 + i].clamp(0, 255) as u8;
                        }
                    }
                    // Update grid — record the per-8x8 direct MVs so
                    // downstream MBs reading our slot get the right
                    // §8.4.1.3 neighbour data.
                    let info = grid.at_mut(mb_x, mb_y);
                    info.available = true;
                    info.is_intra = false;
                    info.is_skip = true;
                    info.is_b_skip_or_direct = true;
                    info.cbp_luma = 0;
                    info.cbp_chroma = 0;
                    info.ref_idx_l0_8x8 = [direct.ref_idx_l0; 4];
                    info.mv_l0_8x8 = direct.mv_l0_per_8x8;
                    info.ref_idx_l1_8x8 = [direct.ref_idx_l1; 4];
                    info.mv_l1_8x8 = direct.mv_l1_per_8x8;
                    // mvd magnitudes stay zero (B_Skip carries no mvd).
                    info.abs_mvd_l0_x = [0; 16];
                    info.abs_mvd_l0_y = [0; 16];
                    info.abs_mvd_l1_x = [0; 16];
                    info.abs_mvd_l1_y = [0; 16];

                    let mv_l0_t = (
                        direct.mv_l0().x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                        direct.mv_l0().y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                    );
                    mb_dbl[mb_addr] = MbDeblockInfo {
                        is_intra: false,
                        qp_y,
                        luma_nonzero_4x4: 0,
                        chroma_nonzero_4x4: 0,
                        mv_l0: [mv_l0_t; 16],
                        ref_idx_l0: [direct.ref_idx_l0.max(0) as i8; 4],
                        ref_poc_l0: [0; 4],
                        ..Default::default()
                    };

                    let is_last = mb_x + 1 == width_mbs && mb_y + 1 == height_mbs;
                    encode_end_of_slice_flag(&mut cabac, is_last);
                    continue;
                }

                // mb_type.
                //
                // Round-475 — partition mode override (mb_type ∈ 4..21
                // per Table 7-14) collapses to a single ue(v) with the
                // shape-specific raw value. Direct / Bi / L0 / L1 16x16
                // (raw 0..3) goes through the unchanged code path.
                // B_8x8 + 4× B_Direct_8x8 emits raw mb_type=22 + 4
                // sub_mb_type=0 (B_Direct_8x8).
                let chosen_mb_type_emitted =
                    if let Some((shape, mode_a, mode_b, ..)) = partition_choice {
                        match shape {
                            PartitionShape::P16x8 => b_16x8_mb_type_raw(mode_a, mode_b),
                            PartitionShape::P8x16 => b_8x16_mb_type_raw(mode_a, mode_b),
                        }
                    } else if is_direct_per_8x8 || mixed_choice.is_some() {
                        22u32
                    } else {
                        chosen_mb_type
                    };
                encode_mb_type_b(&mut cabac, &mut ctxs, &nb, chosen_mb_type_emitted);
                // §7.3.5.2 — B_8x8 sub_mb_pred(): four sub_mb_type entries.
                // For all-Direct (round-475 round-32) all four are 0
                // (B_Direct_8x8). For mixed (round-475 task #475), per-
                // cell ∈ {0, 1, 2, 3} per Table 7-18.
                if is_direct_per_8x8 {
                    for _ in 0..4 {
                        encode_sub_mb_type_b(&mut cabac, &mut ctxs, 0);
                    }
                } else if let Some((cells, _, _)) = mixed_choice {
                    for &c in &cells {
                        encode_sub_mb_type_b(&mut cabac, &mut ctxs, c.sub_mb_type_raw());
                    }
                }

                // 16x16 explicit branch flags (only meaningful when
                // partition_choice is None and mixed_choice is None).
                let pred_kind = chosen_mb_type; // 0 = Direct, 1 = L0, 2 = L1, 3 = Bi
                let uses_l0_explicit_16x16 = partition_choice.is_none()
                    && mixed_choice.is_none()
                    && (pred_kind == 1 || pred_kind == 3);
                let uses_l1_explicit_16x16 = partition_choice.is_none()
                    && mixed_choice.is_none()
                    && (pred_kind == 2 || pred_kind == 3);

                // num_ref_idx_lN_active_minus1 = 0 in our PPS, so ref_idx_lN
                // is NOT emitted (single ref per list). B_Direct_16x16
                // also emits no mvd / ref_idx — the decoder re-derives
                // them via §8.4.1.2.2.

                // Per-partition MVDs for the 16x8 / 8x16 path.
                //
                // §7.3.5.1 emits per-partition mvd_l0 then mvd_l1 in
                // partition order (top→bottom or left→right). Within-MB
                // neighbour reads (for partition 1) see partition 0's
                // MVs through the inflight arrays.
                let (part_mvd_l0_x, part_mvd_l0_y, part_mvd_l1_x, part_mvd_l1_y) =
                    if let Some((shape, mode_a, mode_b, mv0_a, mv1_a, mv0_b, mv1_b)) =
                        partition_choice
                    {
                        let mut mvd0_x = [0i32; 2];
                        let mut mvd0_y = [0i32; 2];
                        let mut mvd1_x = [0i32; 2];
                        let mut mvd1_y = [0i32; 2];
                        let inflight_avail0 = [false; 4];
                        let mv_inflight0 = [Mv::ZERO; 4];
                        let ref_inflight0 = [-1i32; 4];

                        // Partition 0.
                        let (mvp0_l0, mvp0_l1) = match shape {
                            PartitionShape::P16x8 => (
                                cabac_mvp_for_b_16x8_partition(
                                    &grid,
                                    mb_x,
                                    mb_y,
                                    true,
                                    0,
                                    0,
                                    inflight_avail0,
                                    mv_inflight0,
                                    ref_inflight0,
                                ),
                                cabac_mvp_for_b_16x8_partition(
                                    &grid,
                                    mb_x,
                                    mb_y,
                                    true,
                                    0,
                                    1,
                                    inflight_avail0,
                                    mv_inflight0,
                                    ref_inflight0,
                                ),
                            ),
                            PartitionShape::P8x16 => (
                                cabac_mvp_for_b_8x16_partition(
                                    &grid,
                                    mb_x,
                                    mb_y,
                                    true,
                                    0,
                                    0,
                                    inflight_avail0,
                                    mv_inflight0,
                                    ref_inflight0,
                                ),
                                cabac_mvp_for_b_8x16_partition(
                                    &grid,
                                    mb_x,
                                    mb_y,
                                    true,
                                    0,
                                    1,
                                    inflight_avail0,
                                    mv_inflight0,
                                    ref_inflight0,
                                ),
                            ),
                        };
                        if mode_a.uses_l0() {
                            mvd0_x[0] = mv0_a.x - mvp0_l0.x;
                            mvd0_y[0] = mv0_a.y - mvp0_l0.y;
                        }
                        if mode_a.uses_l1() {
                            mvd1_x[0] = mv1_a.x - mvp0_l1.x;
                            mvd1_y[0] = mv1_a.y - mvp0_l1.y;
                        }

                        // Partition 1 — inflight 8x8 layout depends on shape.
                        // 16x8: top covers 8x8 #0 + #1 → inflight[0,1]=true.
                        // 8x16: left covers 8x8 #0 + #2 → inflight[0,2]=true.
                        let (
                            inflight_avail1,
                            mv_inflight1_l0,
                            mv_inflight1_l1,
                            ref_inflight1_l0,
                            ref_inflight1_l1,
                        ) = match shape {
                            PartitionShape::P16x8 => (
                                [true, true, false, false],
                                [mv0_a, mv0_a, Mv::ZERO, Mv::ZERO],
                                [mv1_a, mv1_a, Mv::ZERO, Mv::ZERO],
                                [
                                    if mode_a.uses_l0() { 0 } else { -1 },
                                    if mode_a.uses_l0() { 0 } else { -1 },
                                    -1,
                                    -1,
                                ],
                                [
                                    if mode_a.uses_l1() { 0 } else { -1 },
                                    if mode_a.uses_l1() { 0 } else { -1 },
                                    -1,
                                    -1,
                                ],
                            ),
                            PartitionShape::P8x16 => (
                                [true, false, true, false],
                                [mv0_a, Mv::ZERO, mv0_a, Mv::ZERO],
                                [mv1_a, Mv::ZERO, mv1_a, Mv::ZERO],
                                [
                                    if mode_a.uses_l0() { 0 } else { -1 },
                                    -1,
                                    if mode_a.uses_l0() { 0 } else { -1 },
                                    -1,
                                ],
                                [
                                    if mode_a.uses_l1() { 0 } else { -1 },
                                    -1,
                                    if mode_a.uses_l1() { 0 } else { -1 },
                                    -1,
                                ],
                            ),
                        };

                        let (mvp1_l0, mvp1_l1) = match shape {
                            PartitionShape::P16x8 => (
                                cabac_mvp_for_b_16x8_partition(
                                    &grid,
                                    mb_x,
                                    mb_y,
                                    false,
                                    0,
                                    0,
                                    inflight_avail1,
                                    mv_inflight1_l0,
                                    ref_inflight1_l0,
                                ),
                                cabac_mvp_for_b_16x8_partition(
                                    &grid,
                                    mb_x,
                                    mb_y,
                                    false,
                                    0,
                                    1,
                                    inflight_avail1,
                                    mv_inflight1_l1,
                                    ref_inflight1_l1,
                                ),
                            ),
                            PartitionShape::P8x16 => (
                                cabac_mvp_for_b_8x16_partition(
                                    &grid,
                                    mb_x,
                                    mb_y,
                                    false,
                                    0,
                                    0,
                                    inflight_avail1,
                                    mv_inflight1_l0,
                                    ref_inflight1_l0,
                                ),
                                cabac_mvp_for_b_8x16_partition(
                                    &grid,
                                    mb_x,
                                    mb_y,
                                    false,
                                    0,
                                    1,
                                    inflight_avail1,
                                    mv_inflight1_l1,
                                    ref_inflight1_l1,
                                ),
                            ),
                        };
                        if mode_b.uses_l0() {
                            mvd0_x[1] = mv0_b.x - mvp1_l0.x;
                            mvd0_y[1] = mv0_b.y - mvp1_l0.y;
                        }
                        if mode_b.uses_l1() {
                            mvd1_x[1] = mv1_b.x - mvp1_l1.x;
                            mvd1_y[1] = mv1_b.y - mvp1_l1.y;
                        }
                        (mvd0_x, mvd0_y, mvd1_x, mvd1_y)
                    } else {
                        ([0i32; 2], [0i32; 2], [0i32; 2], [0i32; 2])
                    };

                // Track |mvd|-per-4x4 inflight for within-MB neighbour
                // reads. For 16x16 path the only mvd is at blk4=0, so
                // populate that slot after emission. For partition path
                // we fan out to the partition's 4x4 range.
                let mut inflight_abs_l0_x = [0u32; 16];
                let mut inflight_abs_l0_y = [0u32; 16];
                let mut inflight_abs_l1_x = [0u32; 16];
                let mut inflight_abs_l1_y = [0u32; 16];

                // mvd_l0 (only for explicit B_L0_16x16 / B_Bi_16x16).
                let (mvd_l0_x, mvd_l0_y) = if uses_l0_explicit_16x16 {
                    let mvd_x = mv_l0.x - mvp_l0.x;
                    let mvd_y = mv_l0.y - mvp_l0.y;
                    let sum_x = cabac_enc_mvd_abs_sum(
                        &grid,
                        mb_x,
                        mb_y,
                        0,
                        MvdComponent::X,
                        0,
                        &inflight_abs_l0_x,
                    );
                    encode_mvd_lx(&mut cabac, &mut ctxs, MvdComponent::X, sum_x, mvd_x);
                    let sum_y = cabac_enc_mvd_abs_sum(
                        &grid,
                        mb_x,
                        mb_y,
                        0,
                        MvdComponent::Y,
                        0,
                        &inflight_abs_l0_y,
                    );
                    encode_mvd_lx(&mut cabac, &mut ctxs, MvdComponent::Y, sum_y, mvd_y);
                    let ax = mvd_x.unsigned_abs();
                    let ay = mvd_y.unsigned_abs();
                    for slot in inflight_abs_l0_x.iter_mut() {
                        *slot = ax;
                    }
                    for slot in inflight_abs_l0_y.iter_mut() {
                        *slot = ay;
                    }
                    (mvd_x, mvd_y)
                } else {
                    (0i32, 0i32)
                };
                // mvd_l1.
                let (mvd_l1_x, mvd_l1_y) = if uses_l1_explicit_16x16 {
                    let mvd_x = mv_l1.x - mvp_l1.x;
                    let mvd_y = mv_l1.y - mvp_l1.y;
                    let sum_x = cabac_enc_mvd_abs_sum(
                        &grid,
                        mb_x,
                        mb_y,
                        1,
                        MvdComponent::X,
                        0,
                        &inflight_abs_l1_x,
                    );
                    encode_mvd_lx(&mut cabac, &mut ctxs, MvdComponent::X, sum_x, mvd_x);
                    let sum_y = cabac_enc_mvd_abs_sum(
                        &grid,
                        mb_x,
                        mb_y,
                        1,
                        MvdComponent::Y,
                        0,
                        &inflight_abs_l1_y,
                    );
                    encode_mvd_lx(&mut cabac, &mut ctxs, MvdComponent::Y, sum_y, mvd_y);
                    let ax = mvd_x.unsigned_abs();
                    let ay = mvd_y.unsigned_abs();
                    for slot in inflight_abs_l1_x.iter_mut() {
                        *slot = ax;
                    }
                    for slot in inflight_abs_l1_y.iter_mut() {
                        *slot = ay;
                    }
                    (mvd_x, mvd_y)
                } else {
                    (0i32, 0i32)
                };

                // Partition-mode mvd emission. §7.3.5.1 order:
                //   for part: ref_idx_l0[part]   (none — single ref)
                //   for part: ref_idx_l1[part]   (none — single ref)
                //   for part: if uses L0: mvd_l0[part].x, .y
                //   for part: if uses L1: mvd_l1[part].x, .y
                if let Some((shape, mode_a, mode_b, ..)) = partition_choice {
                    let parts = [mode_a, mode_b];
                    // Per-partition top-left 4x4 block index (mirror of
                    // decoder's `mb_part_top_left_4x4`):
                    //   16x8: part 0 → blk 0, part 1 → blk 8.
                    //   8x16: part 0 → blk 0, part 1 → blk 4.
                    let top_left_blk = match shape {
                        PartitionShape::P16x8 => [0u8, 8u8],
                        PartitionShape::P8x16 => [0u8, 4u8],
                    };
                    // Per-partition 4x4-block index ranges (mirror of
                    // `mb_part_4x4_range`): each partition covers 8 blocks.
                    let part_4x4_ranges: [[u8; 8]; 2] = match shape {
                        PartitionShape::P16x8 => {
                            [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
                        }
                        PartitionShape::P8x16 => {
                            [[0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15]]
                        }
                    };
                    // mvd_l0[0..=1].
                    for p in 0..2 {
                        if parts[p].uses_l0() {
                            let blk4 = top_left_blk[p];
                            let sum_x = cabac_enc_mvd_abs_sum(
                                &grid,
                                mb_x,
                                mb_y,
                                0,
                                MvdComponent::X,
                                blk4,
                                &inflight_abs_l0_x,
                            );
                            let sum_y = cabac_enc_mvd_abs_sum(
                                &grid,
                                mb_x,
                                mb_y,
                                0,
                                MvdComponent::Y,
                                blk4,
                                &inflight_abs_l0_y,
                            );
                            encode_mvd_lx(
                                &mut cabac,
                                &mut ctxs,
                                MvdComponent::X,
                                sum_x,
                                part_mvd_l0_x[p],
                            );
                            encode_mvd_lx(
                                &mut cabac,
                                &mut ctxs,
                                MvdComponent::Y,
                                sum_y,
                                part_mvd_l0_y[p],
                            );
                            // Fan out this partition's |mvd| into all
                            // 4x4 blocks it covers, per decoder
                            // `mb_part_4x4_range`.
                            let ax = part_mvd_l0_x[p].unsigned_abs();
                            let ay = part_mvd_l0_y[p].unsigned_abs();
                            for &b4 in &part_4x4_ranges[p] {
                                inflight_abs_l0_x[b4 as usize] = ax;
                                inflight_abs_l0_y[b4 as usize] = ay;
                            }
                        }
                    }
                    // mvd_l1[0..=1].
                    for p in 0..2 {
                        if parts[p].uses_l1() {
                            let blk4 = top_left_blk[p];
                            let sum_x = cabac_enc_mvd_abs_sum(
                                &grid,
                                mb_x,
                                mb_y,
                                1,
                                MvdComponent::X,
                                blk4,
                                &inflight_abs_l1_x,
                            );
                            let sum_y = cabac_enc_mvd_abs_sum(
                                &grid,
                                mb_x,
                                mb_y,
                                1,
                                MvdComponent::Y,
                                blk4,
                                &inflight_abs_l1_y,
                            );
                            encode_mvd_lx(
                                &mut cabac,
                                &mut ctxs,
                                MvdComponent::X,
                                sum_x,
                                part_mvd_l1_x[p],
                            );
                            encode_mvd_lx(
                                &mut cabac,
                                &mut ctxs,
                                MvdComponent::Y,
                                sum_y,
                                part_mvd_l1_y[p],
                            );
                            let ax = part_mvd_l1_x[p].unsigned_abs();
                            let ay = part_mvd_l1_y[p].unsigned_abs();
                            for &b4 in &part_4x4_ranges[p] {
                                inflight_abs_l1_x[b4 as usize] = ax;
                                inflight_abs_l1_y[b4 as usize] = ay;
                            }
                        }
                    }
                }

                // Round-475 — per-cell mixed `B_8x8` mvd emission.
                //
                // §7.3.5.2 sub_mb_pred() field order (per Table 7-18,
                // single-ref → no ref_idx_lN te(v) since
                // num_ref_idx_l0_active_minus1 == 0):
                //   for i in 0..4: if cell uses L0: mvd_l0[i].x, .y
                //   for i in 0..4: if cell uses L1: mvd_l1[i].x, .y
                //
                // Per-cell MVPs come from `cabac_mvp_for_b_8x8_partition`
                // with the inflight per-quadrant MV / ref state of cells
                // emitted earlier in the MB.
                //
                // Per-4x4 |mvd| fan-out: each 8x8 cell covers a 2x2 block
                // of 4x4 indices. Cell `c` (sub_x, sub_y) covers blk4 in
                // {(2*sub_y+0)*4 + 2*sub_x + 0, +1, (2*sub_y+1)*4 + ...}
                // — but we use the §6.4.3 raster-Z layout below for
                // consistency with the rest of the encoder.
                let mut mixed_per_cell_mvd_l0_x = [0i32; 4];
                let mut mixed_per_cell_mvd_l0_y = [0i32; 4];
                let mut mixed_per_cell_mvd_l1_x = [0i32; 4];
                let mut mixed_per_cell_mvd_l1_y = [0i32; 4];
                if let Some((cells, mv0s, mv1s)) = mixed_choice {
                    // Per-cell decoded flag + per-cell L0/L1 inflight
                    // state, updated in raster order to mirror the
                    // decoder's `process_partition` loop. The decoder
                    // iterates partitions in raster cell order (0, 1,
                    // 2, 3); for each partition it derives mvp_l0 +
                    // mvp_l1 from the current grid state, then stores
                    // (mv_l0, mv_l1) into the grid. So cell c's MVPs
                    // for both lists read inflight reflecting cells
                    // 0..(c-1)'s MVs.
                    //
                    // `cell_decoded[c]` carries whether cell c has been
                    // processed (regardless of which list); the
                    // per-list inflight slot may have ref_idx = -1 when
                    // the cell uses only the OTHER list. The MVP helper
                    // uses this distinction for the §8.4.1.3.2 C→D
                    // substitution (decoded-but-other-list-only cells
                    // are partition-available + intra-like; not-yet-
                    // decoded cells are partition-unavailable).
                    let mut cell_decoded = [false; 4];
                    let mut cell_mv_inflight_l0 = [Mv::ZERO; 4];
                    let mut cell_mv_inflight_l1 = [Mv::ZERO; 4];
                    let mut cell_ref_inflight_l0 = [-1i32; 4];
                    let mut cell_ref_inflight_l1 = [-1i32; 4];

                    // Per-cell top-left 4x4 block index (Table 6-10
                    // raster-Z layout): cell 0 → blk 0, cell 1 → blk 4,
                    // cell 2 → blk 8, cell 3 → blk 12. (cell index `c`
                    // maps to block `4*c` because the §6.4.3 LUMA_4X4_BLK
                    // groups all 4 4x4s of one 8x8 quadrant first.)
                    let cell_top_left_blk: [u8; 4] = [0, 4, 8, 12];
                    let cell_4x4_ranges: [[u8; 4]; 4] = [
                        [0, 1, 2, 3],     // cell 0 → blocks 0..=3
                        [4, 5, 6, 7],     // cell 1 → blocks 4..=7
                        [8, 9, 10, 11],   // cell 2 → blocks 8..=11
                        [12, 13, 14, 15], // cell 3 → blocks 12..=15
                    ];

                    // §7.3.5.2 — emit mvd_l0[0..4] in cell order. After
                    // each cell's MVP derivation, mark cell as decoded
                    // and populate L0 + L1 inflight (Direct cells use
                    // §8.4.1.2.2 MVs / refs; explicit cells use ME MVs
                    // with refIdx = 0). This mirrors the decoder's per-
                    // cell `process_partition` grid update.
                    for c in 0..4usize {
                        if cells[c].writes_l0() {
                            let mvp = cabac_mvp_for_b_8x8_partition(
                                &grid,
                                mb_x,
                                mb_y,
                                c,
                                0,
                                0,
                                cell_decoded,
                                cell_mv_inflight_l0,
                                cell_ref_inflight_l0,
                            );
                            let mvd_x = mv0s[c].x - mvp.x;
                            let mvd_y = mv0s[c].y - mvp.y;
                            let blk4 = cell_top_left_blk[c];
                            let sum_x = cabac_enc_mvd_abs_sum(
                                &grid,
                                mb_x,
                                mb_y,
                                0,
                                MvdComponent::X,
                                blk4,
                                &inflight_abs_l0_x,
                            );
                            let sum_y = cabac_enc_mvd_abs_sum(
                                &grid,
                                mb_x,
                                mb_y,
                                0,
                                MvdComponent::Y,
                                blk4,
                                &inflight_abs_l0_y,
                            );
                            encode_mvd_lx(&mut cabac, &mut ctxs, MvdComponent::X, sum_x, mvd_x);
                            encode_mvd_lx(&mut cabac, &mut ctxs, MvdComponent::Y, sum_y, mvd_y);
                            mixed_per_cell_mvd_l0_x[c] = mvd_x;
                            mixed_per_cell_mvd_l0_y[c] = mvd_y;
                            let ax = mvd_x.unsigned_abs();
                            let ay = mvd_y.unsigned_abs();
                            for &b4 in &cell_4x4_ranges[c] {
                                inflight_abs_l0_x[b4 as usize] = ax;
                                inflight_abs_l0_y[b4 as usize] = ay;
                            }
                        }
                        // Update both L0 and L1 inflight at end of cell's
                        // iteration (mirrors decoder).
                        cell_decoded[c] = true;
                        match cells[c] {
                            BSubMbCell::Direct => {
                                if direct.ref_idx_l0 >= 0 {
                                    cell_mv_inflight_l0[c] = direct.mv_l0_per_8x8[c];
                                    cell_ref_inflight_l0[c] = direct.ref_idx_l0;
                                }
                                if direct.ref_idx_l1 >= 0 {
                                    cell_mv_inflight_l1[c] = direct.mv_l1_per_8x8[c];
                                    cell_ref_inflight_l1[c] = direct.ref_idx_l1;
                                }
                            }
                            BSubMbCell::L0 => {
                                cell_mv_inflight_l0[c] = mv0s[c];
                                cell_ref_inflight_l0[c] = 0;
                            }
                            BSubMbCell::L1 => {
                                cell_mv_inflight_l1[c] = mv1s[c];
                                cell_ref_inflight_l1[c] = 0;
                            }
                            BSubMbCell::Bi => {
                                cell_mv_inflight_l0[c] = mv0s[c];
                                cell_ref_inflight_l0[c] = 0;
                                cell_mv_inflight_l1[c] = mv1s[c];
                                cell_ref_inflight_l1[c] = 0;
                            }
                        }
                    }
                    // §7.3.5.2 — emit mvd_l1[0..4] in cell order. The L1
                    // mvp for cell c reads cells 0..(c-1)'s L1 grid
                    // state (already populated by the L0 loop above for
                    // ALL cells). To match the decoder we need fresh
                    // per-cell state that reflects cells 0..(c-1) only.
                    let mut cell_decoded_walk = [false; 4];
                    let mut cell_mv_inflight_l1_walk = [Mv::ZERO; 4];
                    let mut cell_ref_inflight_l1_walk = [-1i32; 4];
                    for c in 0..4usize {
                        if cells[c].writes_l1() {
                            let mvp = cabac_mvp_for_b_8x8_partition(
                                &grid,
                                mb_x,
                                mb_y,
                                c,
                                0,
                                1,
                                cell_decoded_walk,
                                cell_mv_inflight_l1_walk,
                                cell_ref_inflight_l1_walk,
                            );
                            let mvd_x = mv1s[c].x - mvp.x;
                            let mvd_y = mv1s[c].y - mvp.y;
                            let blk4 = cell_top_left_blk[c];
                            let sum_x = cabac_enc_mvd_abs_sum(
                                &grid,
                                mb_x,
                                mb_y,
                                1,
                                MvdComponent::X,
                                blk4,
                                &inflight_abs_l1_x,
                            );
                            let sum_y = cabac_enc_mvd_abs_sum(
                                &grid,
                                mb_x,
                                mb_y,
                                1,
                                MvdComponent::Y,
                                blk4,
                                &inflight_abs_l1_y,
                            );
                            encode_mvd_lx(&mut cabac, &mut ctxs, MvdComponent::X, sum_x, mvd_x);
                            encode_mvd_lx(&mut cabac, &mut ctxs, MvdComponent::Y, sum_y, mvd_y);
                            mixed_per_cell_mvd_l1_x[c] = mvd_x;
                            mixed_per_cell_mvd_l1_y[c] = mvd_y;
                            let ax = mvd_x.unsigned_abs();
                            let ay = mvd_y.unsigned_abs();
                            for &b4 in &cell_4x4_ranges[c] {
                                inflight_abs_l1_x[b4 as usize] = ax;
                                inflight_abs_l1_y[b4 as usize] = ay;
                            }
                        }
                        cell_decoded_walk[c] = true;
                        match cells[c] {
                            BSubMbCell::Direct if direct.ref_idx_l1 >= 0 => {
                                cell_mv_inflight_l1_walk[c] = direct.mv_l1_per_8x8[c];
                                cell_ref_inflight_l1_walk[c] = direct.ref_idx_l1;
                            }
                            BSubMbCell::L1 | BSubMbCell::Bi => {
                                cell_mv_inflight_l1_walk[c] = mv1s[c];
                                cell_ref_inflight_l1_walk[c] = 0;
                            }
                            _ => {}
                        }
                    }
                    let _ = (
                        cell_decoded,
                        cell_mv_inflight_l0,
                        cell_ref_inflight_l0,
                        cell_mv_inflight_l1,
                        cell_ref_inflight_l1,
                    );
                }

                // coded_block_pattern.
                encode_coded_block_pattern(&mut cabac, &mut ctxs, &nb, 1, cbp_luma, cbp_chroma);

                // mb_qp_delta only when CBP > 0.
                let needs_qpd = cbp_luma > 0 || cbp_chroma > 0;
                if needs_qpd {
                    let qpd = 0i32;
                    encode_mb_qp_delta(&mut cabac, &mut ctxs, prev_mb_qp_delta_nonzero, qpd);
                    prev_mb_qp_delta_nonzero = qpd != 0;
                } else {
                    prev_mb_qp_delta_nonzero = false;
                }

                {
                    let cur = grid.at_mut(mb_x, mb_y);
                    cur.is_intra = false;
                    cur.is_skip = false;
                    cur.is_b_skip_or_direct = is_direct;
                    cur.cbp_luma = cbp_luma;
                    cur.cbp_chroma = cbp_chroma;
                }

                // Luma residual blocks (per-8x8 gated).
                for blk8 in 0..4u8 {
                    if (cbp_luma >> blk8) & 1 == 1 {
                        for sub in 0..4u8 {
                            let blkz = (blk8 * 4 + sub) as usize;
                            let cur = grid.at(mb_x, mb_y).clone();
                            let (ca, cb) = cbf_neighbour_luma_ac(
                                &grid,
                                mb_x,
                                mb_y,
                                &cur,
                                false,
                                BlockType::Luma4x4,
                                blkz as u8,
                            );
                            let coded = encode_residual_block_cabac(
                                &mut cabac,
                                &mut ctxs,
                                BlockType::Luma4x4,
                                &luma_blocks[blkz],
                                16,
                                Some(ca),
                                Some(cb),
                                false,
                            );
                            grid.at_mut(mb_x, mb_y).cbf_luma_4x4[blkz] = coded;
                        }
                    }
                }

                // Chroma DC + AC residual.
                if cbp_chroma > 0 {
                    let (cb_a, cb_b) =
                        cbf_neighbour_mb_level(&grid, mb_x, mb_y, false, CbfPlane::CbDc);
                    let cb_dc_coded = encode_residual_block_cabac(
                        &mut cabac,
                        &mut ctxs,
                        BlockType::ChromaDc,
                        &cb_dc,
                        4,
                        Some(cb_a),
                        Some(cb_b),
                        false,
                    );
                    grid.at_mut(mb_x, mb_y).cbf_cb_dc = cb_dc_coded;
                    let (cr_a, cr_b) =
                        cbf_neighbour_mb_level(&grid, mb_x, mb_y, false, CbfPlane::CrDc);
                    let cr_dc_coded = encode_residual_block_cabac(
                        &mut cabac,
                        &mut ctxs,
                        BlockType::ChromaDc,
                        &cr_dc,
                        4,
                        Some(cr_a),
                        Some(cr_b),
                        false,
                    );
                    grid.at_mut(mb_x, mb_y).cbf_cr_dc = cr_dc_coded;
                }
                if cbp_chroma == 2 {
                    for blk in 0..4usize {
                        let cur = grid.at(mb_x, mb_y).clone();
                        let (ca, cb) = cbf_neighbour_chroma_ac(
                            &grid, mb_x, mb_y, &cur, false, blk as u8, false,
                        );
                        let coded = encode_residual_block_cabac(
                            &mut cabac,
                            &mut ctxs,
                            BlockType::ChromaAc,
                            &cb_ac_scan[blk][..15],
                            15,
                            Some(ca),
                            Some(cb),
                            false,
                        );
                        grid.at_mut(mb_x, mb_y).cbf_cb_ac[blk] = coded;
                    }
                    for blk in 0..4usize {
                        let cur = grid.at(mb_x, mb_y).clone();
                        let (ca, cb) = cbf_neighbour_chroma_ac(
                            &grid, mb_x, mb_y, &cur, false, blk as u8, true,
                        );
                        let coded = encode_residual_block_cabac(
                            &mut cabac,
                            &mut ctxs,
                            BlockType::ChromaAc,
                            &cr_ac_scan[blk][..15],
                            15,
                            Some(ca),
                            Some(cb),
                            false,
                        );
                        grid.at_mut(mb_x, mb_y).cbf_cr_ac[blk] = coded;
                    }
                }

                // Reconstruct luma.
                for blk8 in 0..4usize {
                    let send = (cbp_luma >> blk8) & 1 == 1;
                    for sub in 0..4usize {
                        let blkz = blk8 * 4 + sub;
                        let (bx, by) = LUMA_4X4_BLK[blkz];
                        for j in 0..4usize {
                            for i in 0..4usize {
                                let p = pred_y[(by * 4 + j) * 16 + bx * 4 + i];
                                let r = if send {
                                    recon_residual_y[(by * 4 + j) * 16 + bx * 4 + i]
                                } else {
                                    0
                                };
                                let py = mb_y * 16 + by * 4 + j;
                                let px = mb_x * 16 + bx * 4 + i;
                                recon_y[py * width + px] = (p + r).clamp(0, 255) as u8;
                            }
                        }
                    }
                }

                // Reconstruct chroma.
                let recon_cb_residual = if cbp_chroma == 2 {
                    cb_recon_res
                } else if cbp_chroma == 1 {
                    chroma_residual_dc_only(&cb_dc, qp_c)
                } else {
                    [0i32; 64]
                };
                let recon_cr_residual = if cbp_chroma == 2 {
                    cr_recon_res
                } else if cbp_chroma == 1 {
                    chroma_residual_dc_only(&cr_dc, qp_c)
                } else {
                    [0i32; 64]
                };
                for j in 0..8usize {
                    for i in 0..8usize {
                        let cy = mb_y * 8 + j;
                        let cx = mb_x * 8 + i;
                        recon_u[cy * chroma_w + cx] =
                            (pred_u[j * 8 + i] + recon_cb_residual[j * 8 + i]).clamp(0, 255) as u8;
                        recon_v[cy * chroma_w + cx] =
                            (pred_v[j * 8 + i] + recon_cr_residual[j * 8 + i]).clamp(0, 255) as u8;
                    }
                }

                // Update grid — record real per-8x8 MVs / refIdxs for
                // both lists so downstream MBs reading our slot get the
                // right §8.4.1.3 neighbour data + §8.4.1.2.2 step-7
                // colZeroFlag input. For B_Direct we use the §8.4.1.2.2
                // derivation's per-8x8 MVs; for explicit-inter we use
                // the chosen ME MVs (replicated across all 4 partitions
                // since round-32 only emits 16x16 partitions in CABAC B).
                let info = grid.at_mut(mb_x, mb_y);
                info.available = true;
                info.is_intra = false;
                info.is_skip = false;
                info.is_b_skip_or_direct = is_direct;
                info.cbp_luma = cbp_luma;
                info.cbp_chroma = cbp_chroma;
                if is_direct || is_direct_per_8x8 {
                    info.ref_idx_l0_8x8 = [direct.ref_idx_l0; 4];
                    info.mv_l0_8x8 = direct.mv_l0_per_8x8;
                    info.ref_idx_l1_8x8 = [direct.ref_idx_l1; 4];
                    info.mv_l1_8x8 = direct.mv_l1_per_8x8;
                    info.abs_mvd_l0_x = [0; 16];
                    info.abs_mvd_l0_y = [0; 16];
                    info.abs_mvd_l1_x = [0; 16];
                    info.abs_mvd_l1_y = [0; 16];
                } else if let Some((cells, mv0s, mv1s)) = mixed_choice {
                    // Round-475 — mixed B_8x8 per-cell grid update.
                    let mut ref_l0 = [-1i32; 4];
                    let mut ref_l1 = [-1i32; 4];
                    let mut mv_l0_arr = [Mv::ZERO; 4];
                    let mut mv_l1_arr = [Mv::ZERO; 4];
                    for c in 0..4usize {
                        match cells[c] {
                            BSubMbCell::Direct => {
                                ref_l0[c] = direct.ref_idx_l0;
                                ref_l1[c] = direct.ref_idx_l1;
                                mv_l0_arr[c] = direct.mv_l0_per_8x8[c];
                                mv_l1_arr[c] = direct.mv_l1_per_8x8[c];
                            }
                            BSubMbCell::L0 => {
                                ref_l0[c] = 0;
                                mv_l0_arr[c] = mv0s[c];
                            }
                            BSubMbCell::L1 => {
                                ref_l1[c] = 0;
                                mv_l1_arr[c] = mv1s[c];
                            }
                            BSubMbCell::Bi => {
                                ref_l0[c] = 0;
                                ref_l1[c] = 0;
                                mv_l0_arr[c] = mv0s[c];
                                mv_l1_arr[c] = mv1s[c];
                            }
                        }
                    }
                    info.ref_idx_l0_8x8 = ref_l0;
                    info.mv_l0_8x8 = mv_l0_arr;
                    info.ref_idx_l1_8x8 = ref_l1;
                    info.mv_l1_8x8 = mv_l1_arr;
                    // Per-4x4 |mvd| layout. Cell c covers blocks {4c..4c+3}
                    // in §6.4.3 raster-Z order (cell groups by 8x8).
                    let mut abs_mvd_l0_x = [0u32; 16];
                    let mut abs_mvd_l0_y = [0u32; 16];
                    let mut abs_mvd_l1_x = [0u32; 16];
                    let mut abs_mvd_l1_y = [0u32; 16];
                    for c in 0..4usize {
                        let ax_l0 = mixed_per_cell_mvd_l0_x[c].unsigned_abs();
                        let ay_l0 = mixed_per_cell_mvd_l0_y[c].unsigned_abs();
                        let ax_l1 = mixed_per_cell_mvd_l1_x[c].unsigned_abs();
                        let ay_l1 = mixed_per_cell_mvd_l1_y[c].unsigned_abs();
                        for sub in 0..4usize {
                            let blkz = c * 4 + sub;
                            abs_mvd_l0_x[blkz] = ax_l0;
                            abs_mvd_l0_y[blkz] = ay_l0;
                            abs_mvd_l1_x[blkz] = ax_l1;
                            abs_mvd_l1_y[blkz] = ay_l1;
                        }
                    }
                    info.abs_mvd_l0_x = abs_mvd_l0_x;
                    info.abs_mvd_l0_y = abs_mvd_l0_y;
                    info.abs_mvd_l1_x = abs_mvd_l1_x;
                    info.abs_mvd_l1_y = abs_mvd_l1_y;
                } else if let Some((shape, mode_a, mode_b, mv0_a, mv1_a, mv0_b, mv1_b)) =
                    partition_choice
                {
                    // Per-partition per-8x8 MV / ref / |mvd| layout.
                    // 16x8: 8x8 #0/#1 = top, #2/#3 = bottom.
                    // 8x16: 8x8 #0/#2 = left, #1/#3 = right.
                    let (cell_a_idx, cell_b_idx): ([usize; 2], [usize; 2]) = match shape {
                        PartitionShape::P16x8 => ([0, 1], [2, 3]),
                        PartitionShape::P8x16 => ([0, 2], [1, 3]),
                    };
                    let mut ref_l0 = [-1i32; 4];
                    let mut ref_l1 = [-1i32; 4];
                    let mut mv_l0_arr = [Mv::ZERO; 4];
                    let mut mv_l1_arr = [Mv::ZERO; 4];
                    for &i in &cell_a_idx {
                        if mode_a.uses_l0() {
                            ref_l0[i] = 0;
                            mv_l0_arr[i] = mv0_a;
                        }
                        if mode_a.uses_l1() {
                            ref_l1[i] = 0;
                            mv_l1_arr[i] = mv1_a;
                        }
                    }
                    for &i in &cell_b_idx {
                        if mode_b.uses_l0() {
                            ref_l0[i] = 0;
                            mv_l0_arr[i] = mv0_b;
                        }
                        if mode_b.uses_l1() {
                            ref_l1[i] = 0;
                            mv_l1_arr[i] = mv1_b;
                        }
                    }
                    info.ref_idx_l0_8x8 = ref_l0;
                    info.mv_l0_8x8 = mv_l0_arr;
                    info.ref_idx_l1_8x8 = ref_l1;
                    info.mv_l1_8x8 = mv_l1_arr;
                    // |mvd| arrays — coarsely set per 8x8 quadrant. The
                    // 4x4 layout below mirrors the 8x8 → 4x4 fan-out
                    // (each 8x8 cell covers a 2x2 block of 4x4s in
                    // raster Z-order). For the |mvd|-sum neighbour
                    // calculation we only ever read index 0 (top-left
                    // 4x4 of left/above MB), so the partition layout
                    // matters: top-left 4x4 → cell 0.
                    let abs_l0_x_a: u32 = part_mvd_l0_x[0].unsigned_abs();
                    let abs_l0_y_a: u32 = part_mvd_l0_y[0].unsigned_abs();
                    let abs_l1_x_a: u32 = part_mvd_l1_x[0].unsigned_abs();
                    let abs_l1_y_a: u32 = part_mvd_l1_y[0].unsigned_abs();
                    let abs_l0_x_b: u32 = part_mvd_l0_x[1].unsigned_abs();
                    let abs_l0_y_b: u32 = part_mvd_l0_y[1].unsigned_abs();
                    let abs_l1_x_b: u32 = part_mvd_l1_x[1].unsigned_abs();
                    let abs_l1_y_b: u32 = part_mvd_l1_y[1].unsigned_abs();
                    let mut abs_mvd_l0_x = [0u32; 16];
                    let mut abs_mvd_l0_y = [0u32; 16];
                    let mut abs_mvd_l1_x = [0u32; 16];
                    let mut abs_mvd_l1_y = [0u32; 16];
                    for blkz in 0..16usize {
                        let (bx, by) = LUMA_4X4_BLK[blkz];
                        let cell = (by / 2) * 2 + (bx / 2);
                        let in_a = cell_a_idx.contains(&cell);
                        if in_a {
                            abs_mvd_l0_x[blkz] = abs_l0_x_a;
                            abs_mvd_l0_y[blkz] = abs_l0_y_a;
                            abs_mvd_l1_x[blkz] = abs_l1_x_a;
                            abs_mvd_l1_y[blkz] = abs_l1_y_a;
                        } else {
                            abs_mvd_l0_x[blkz] = abs_l0_x_b;
                            abs_mvd_l0_y[blkz] = abs_l0_y_b;
                            abs_mvd_l1_x[blkz] = abs_l1_x_b;
                            abs_mvd_l1_y[blkz] = abs_l1_y_b;
                        }
                    }
                    info.abs_mvd_l0_x = abs_mvd_l0_x;
                    info.abs_mvd_l0_y = abs_mvd_l0_y;
                    info.abs_mvd_l1_x = abs_mvd_l1_x;
                    info.abs_mvd_l1_y = abs_mvd_l1_y;
                } else {
                    if uses_l0_explicit_16x16 {
                        info.ref_idx_l0_8x8 = [0; 4];
                        info.mv_l0_8x8 = [mv_l0; 4];
                        info.abs_mvd_l0_x = [mvd_l0_x.unsigned_abs(); 16];
                        info.abs_mvd_l0_y = [mvd_l0_y.unsigned_abs(); 16];
                    } else {
                        info.ref_idx_l0_8x8 = [-1; 4];
                        info.mv_l0_8x8 = [Mv::ZERO; 4];
                        info.abs_mvd_l0_x = [0; 16];
                        info.abs_mvd_l0_y = [0; 16];
                    }
                    if uses_l1_explicit_16x16 {
                        info.ref_idx_l1_8x8 = [0; 4];
                        info.mv_l1_8x8 = [mv_l1; 4];
                        info.abs_mvd_l1_x = [mvd_l1_x.unsigned_abs(); 16];
                        info.abs_mvd_l1_y = [mvd_l1_y.unsigned_abs(); 16];
                    } else {
                        info.ref_idx_l1_8x8 = [-1; 4];
                        info.mv_l1_8x8 = [Mv::ZERO; 4];
                        info.abs_mvd_l1_x = [0; 16];
                        info.abs_mvd_l1_y = [0; 16];
                    }
                }

                // Pick a representative L0 MV for the deblocker fallback
                // (legacy 16x16 path). The partition / direct_per_8x8
                // paths fill `mv_l0_arr` per-4x4 above and ignore this.
                let dbl_mv_l0 = if is_direct {
                    direct.mv_l0()
                } else if is_direct_per_8x8 {
                    direct.mv_l0_per_8x8[0]
                } else if let Some((cells, mv0s, _)) = mixed_choice {
                    // Top-left cell (idx 0) is the representative.
                    match cells[0] {
                        BSubMbCell::Direct => direct.mv_l0_per_8x8[0],
                        BSubMbCell::L0 | BSubMbCell::Bi => mv0s[0],
                        BSubMbCell::L1 => Mv::ZERO,
                    }
                } else if let Some((_, mode_a, _, mv0_a, _, _, _)) = partition_choice {
                    if mode_a.uses_l0() {
                        mv0_a
                    } else {
                        Mv::ZERO
                    }
                } else if uses_l0_explicit_16x16 {
                    mv_l0
                } else {
                    Mv::ZERO
                };
                let mv_l0_t = (
                    dbl_mv_l0.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                    dbl_mv_l0.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                );

                let cb_ac_per4x4: [bool; 4] = std::array::from_fn(|i| {
                    cbp_chroma == 2 && cb_ac_scan[i].iter().any(|&v| v != 0)
                });
                let cr_ac_per4x4: [bool; 4] = std::array::from_fn(|i| {
                    cbp_chroma == 2 && cr_ac_scan[i].iter().any(|&v| v != 0)
                });

                // §8.7 deblock — per-4x4 MV + per-8x8 ref / POC arrays.
                // For partition modes (16x8 / 8x16) each half carries its
                // own MV / ref; the §8.7.2.1 bS derivation reads these
                // per-block values, so the encoder's deblock state must
                // mirror what the decoder will compute (else internal
                // partition boundaries get the wrong filter strength).
                // For B_8x8 + all-Direct each 8x8 quadrant carries the
                // §8.4.1.2.2 derivation's per-quadrant (mvL0, mvL1).
                let (mv_l0_arr, ref_idx_l0_arr, ref_poc_l0_arr): (
                    [(i16, i16); 16],
                    [i8; 4],
                    [i32; 4],
                ) = if is_direct_per_8x8 {
                    let l0_used = direct.ref_idx_l0 >= 0;
                    let mut mvs = [(0i16, 0i16); 16];
                    let mut refs = [-1i8; 4];
                    let mut pocs = [i32::MIN; 4];
                    if l0_used {
                        for blk in 0..16usize {
                            let (bx, by) = LUMA_4X4_BLK[blk];
                            let q = (by / 2) * 2 + (bx / 2);
                            let mv = direct.mv_l0_per_8x8[q];
                            mvs[blk] = (
                                mv.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                                mv.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                            );
                        }
                        for q in 0..4 {
                            refs[q] = direct.ref_idx_l0.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
                            pocs[q] = 0;
                        }
                    }
                    (mvs, refs, pocs)
                } else if let Some((cells, mv0s, _)) = mixed_choice {
                    // Round-475 mixed B_8x8 — per-cell L0 MV / ref / poc.
                    let mut mvs = [(0i16, 0i16); 16];
                    let mut refs = [-1i8; 4];
                    let mut pocs = [i32::MIN; 4];
                    for c in 0..4usize {
                        let (mv, ref_idx, poc) = match cells[c] {
                            BSubMbCell::Direct => (
                                direct.mv_l0_per_8x8[c],
                                direct.ref_idx_l0,
                                if direct.ref_idx_l0 >= 0 { 0 } else { i32::MIN },
                            ),
                            BSubMbCell::L0 | BSubMbCell::Bi => (mv0s[c], 0, 0),
                            BSubMbCell::L1 => (Mv::ZERO, -1, i32::MIN),
                        };
                        let mvt = (
                            mv.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                            mv.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                        );
                        for sub in 0..4usize {
                            mvs[c * 4 + sub] = mvt;
                        }
                        refs[c] = ref_idx.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
                        pocs[c] = poc;
                    }
                    (mvs, refs, pocs)
                } else if let Some((shape, mode_a, mode_b, mv0_a, _, mv0_b, _)) = partition_choice {
                    let used_a = mode_a.uses_l0();
                    let used_b = mode_b.uses_l0();
                    let mv_a_t = (
                        mv0_a.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                        mv0_a.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                    );
                    let mv_b_t = (
                        mv0_b.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                        mv0_b.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                    );
                    let mut mvs = [(0i16, 0i16); 16];
                    let mut refs = [-1i8; 4];
                    let mut pocs = [i32::MIN; 4];
                    match shape {
                        PartitionShape::P16x8 => {
                            for (blk, slot) in mvs.iter_mut().enumerate() {
                                let (_bx, by) = LUMA_4X4_BLK[blk];
                                *slot = if by < 2 {
                                    if used_a {
                                        mv_a_t
                                    } else {
                                        (0, 0)
                                    }
                                } else if used_b {
                                    mv_b_t
                                } else {
                                    (0, 0)
                                };
                            }
                            refs[0] = if used_a { 0 } else { -1 };
                            refs[1] = if used_a { 0 } else { -1 };
                            refs[2] = if used_b { 0 } else { -1 };
                            refs[3] = if used_b { 0 } else { -1 };
                            pocs[0] = if used_a { 0 } else { i32::MIN };
                            pocs[1] = if used_a { 0 } else { i32::MIN };
                            pocs[2] = if used_b { 0 } else { i32::MIN };
                            pocs[3] = if used_b { 0 } else { i32::MIN };
                        }
                        PartitionShape::P8x16 => {
                            for (blk, slot) in mvs.iter_mut().enumerate() {
                                let (bx, _by) = LUMA_4X4_BLK[blk];
                                *slot = if bx < 2 {
                                    if used_a {
                                        mv_a_t
                                    } else {
                                        (0, 0)
                                    }
                                } else if used_b {
                                    mv_b_t
                                } else {
                                    (0, 0)
                                };
                            }
                            refs[0] = if used_a { 0 } else { -1 };
                            refs[2] = if used_a { 0 } else { -1 };
                            refs[1] = if used_b { 0 } else { -1 };
                            refs[3] = if used_b { 0 } else { -1 };
                            pocs[0] = if used_a { 0 } else { i32::MIN };
                            pocs[2] = if used_a { 0 } else { i32::MIN };
                            pocs[1] = if used_b { 0 } else { i32::MIN };
                            pocs[3] = if used_b { 0 } else { i32::MIN };
                        }
                    }
                    (mvs, refs, pocs)
                } else {
                    ([mv_l0_t; 16], [0; 4], [0; 4])
                };
                let (mv_l1_arr, ref_idx_l1_arr, ref_poc_l1_arr): (
                    [(i16, i16); 16],
                    [i8; 4],
                    [i32; 4],
                ) = if is_direct_per_8x8 {
                    let l1_used = direct.ref_idx_l1 >= 0;
                    let mut mvs = [(0i16, 0i16); 16];
                    let mut refs = [-1i8; 4];
                    let mut pocs = [i32::MIN; 4];
                    if l1_used {
                        for blk in 0..16usize {
                            let (bx, by) = LUMA_4X4_BLK[blk];
                            let q = (by / 2) * 2 + (bx / 2);
                            let mv = direct.mv_l1_per_8x8[q];
                            mvs[blk] = (
                                mv.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                                mv.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                            );
                        }
                        for q in 0..4 {
                            refs[q] = direct.ref_idx_l1.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
                            pocs[q] = 1000;
                        }
                    }
                    (mvs, refs, pocs)
                } else if let Some((cells, _, mv1s)) = mixed_choice {
                    // Round-475 mixed B_8x8 — per-cell L1 MV / ref / poc.
                    let mut mvs = [(0i16, 0i16); 16];
                    let mut refs = [-1i8; 4];
                    let mut pocs = [i32::MIN; 4];
                    for c in 0..4usize {
                        let (mv, ref_idx, poc) = match cells[c] {
                            BSubMbCell::Direct => (
                                direct.mv_l1_per_8x8[c],
                                direct.ref_idx_l1,
                                if direct.ref_idx_l1 >= 0 {
                                    1000
                                } else {
                                    i32::MIN
                                },
                            ),
                            BSubMbCell::L1 | BSubMbCell::Bi => (mv1s[c], 0, 1000),
                            BSubMbCell::L0 => (Mv::ZERO, -1, i32::MIN),
                        };
                        let mvt = (
                            mv.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                            mv.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                        );
                        for sub in 0..4usize {
                            mvs[c * 4 + sub] = mvt;
                        }
                        refs[c] = ref_idx.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
                        pocs[c] = poc;
                    }
                    (mvs, refs, pocs)
                } else if let Some((shape, mode_a, mode_b, _, mv1_a, _, mv1_b)) = partition_choice {
                    let used_a = mode_a.uses_l1();
                    let used_b = mode_b.uses_l1();
                    let mv_a_t = (
                        mv1_a.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                        mv1_a.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                    );
                    let mv_b_t = (
                        mv1_b.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                        mv1_b.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                    );
                    let mut mvs = [(0i16, 0i16); 16];
                    let mut refs = [-1i8; 4];
                    let mut pocs = [i32::MIN; 4];
                    match shape {
                        PartitionShape::P16x8 => {
                            for (blk, slot) in mvs.iter_mut().enumerate() {
                                let (_bx, by) = LUMA_4X4_BLK[blk];
                                *slot = if by < 2 {
                                    if used_a {
                                        mv_a_t
                                    } else {
                                        (0, 0)
                                    }
                                } else if used_b {
                                    mv_b_t
                                } else {
                                    (0, 0)
                                };
                            }
                            refs[0] = if used_a { 0 } else { -1 };
                            refs[1] = if used_a { 0 } else { -1 };
                            refs[2] = if used_b { 0 } else { -1 };
                            refs[3] = if used_b { 0 } else { -1 };
                            pocs[0] = if used_a { 1000 } else { i32::MIN };
                            pocs[1] = if used_a { 1000 } else { i32::MIN };
                            pocs[2] = if used_b { 1000 } else { i32::MIN };
                            pocs[3] = if used_b { 1000 } else { i32::MIN };
                        }
                        PartitionShape::P8x16 => {
                            for (blk, slot) in mvs.iter_mut().enumerate() {
                                let (bx, _by) = LUMA_4X4_BLK[blk];
                                *slot = if bx < 2 {
                                    if used_a {
                                        mv_a_t
                                    } else {
                                        (0, 0)
                                    }
                                } else if used_b {
                                    mv_b_t
                                } else {
                                    (0, 0)
                                };
                            }
                            refs[0] = if used_a { 0 } else { -1 };
                            refs[2] = if used_a { 0 } else { -1 };
                            refs[1] = if used_b { 0 } else { -1 };
                            refs[3] = if used_b { 0 } else { -1 };
                            pocs[0] = if used_a { 1000 } else { i32::MIN };
                            pocs[2] = if used_a { 1000 } else { i32::MIN };
                            pocs[1] = if used_b { 1000 } else { i32::MIN };
                            pocs[3] = if used_b { 1000 } else { i32::MIN };
                        }
                    }
                    (mvs, refs, pocs)
                } else {
                    ([(0, 0); 16], [-1; 4], [i32::MIN; 4])
                };

                mb_dbl[mb_addr] = MbDeblockInfo {
                    is_intra: false,
                    qp_y,
                    luma_nonzero_4x4: luma_nz_mask_from_blocks(&blk_has_nz),
                    chroma_nonzero_4x4: chroma_nz_mask_from_blocks(&cb_ac_per4x4, &cr_ac_per4x4),
                    mv_l0: mv_l0_arr,
                    ref_idx_l0: ref_idx_l0_arr,
                    ref_poc_l0: ref_poc_l0_arr,
                    mv_l1: mv_l1_arr,
                    ref_idx_l1: ref_idx_l1_arr,
                    ref_poc_l1: ref_poc_l1_arr,
                };

                let is_last = mb_x + 1 == width_mbs && mb_y + 1 == height_mbs;
                encode_end_of_slice_flag(&mut cabac, is_last);
            }
        }

        let cabac_payload = cabac.finish();
        let mut slice_rbsp = sw.into_bytes();
        slice_rbsp.extend_from_slice(&cabac_payload);
        // nal_ref_idc = 0 (non-reference B).
        stream.extend_from_slice(&build_nal_unit(0, NalUnitType::SliceNonIdr, &slice_rbsp));

        deblock_recon(
            cfg.width,
            cfg.height,
            chroma_w as u32,
            chroma_h as u32,
            &mut recon_y,
            &mut recon_u,
            &mut recon_v,
            &mb_dbl,
            0,
            cfg.width / 16,
            cfg.height / 16,
        );

        let n_parts = width_mbs * height_mbs * 4;
        let partition_mvs = vec![
            FrameRefPartitionMv {
                mv_l0: (0, 0),
                ref_idx_l0: 0,
                is_intra: false,
            };
            n_parts
        ];

        EncodedB {
            annex_b: stream,
            recon_y,
            recon_u,
            recon_v,
            recon_width: cfg.width,
            recon_height: cfg.height,
            frame_num,
            pic_order_cnt_lsb,
            partition_mvs,
        }
    }
}

// ---------------------------------------------------------------------------
// Inter helpers (luma + chroma predictor build using the spec's
// interpolation primitives).
// ---------------------------------------------------------------------------

fn build_inter_pred_luma_local(
    ref_y: &[u8],
    ref_w: u32,
    ref_h: u32,
    mb_x: usize,
    mb_y: usize,
    mv: Mv,
) -> [i32; 256] {
    let stride = ref_w as usize;
    let h = ref_h as usize;
    let ref_i32: Vec<i32> = ref_y.iter().take(stride * h).map(|&v| v as i32).collect();
    let mv_int_x = mv.x >> 2;
    let mv_int_y = mv.y >> 2;
    let x_frac = (mv.x & 3) as u8;
    let y_frac = (mv.y & 3) as u8;
    let int_x = (mb_x * 16) as i32 + mv_int_x;
    let int_y = (mb_y * 16) as i32 + mv_int_y;
    let mut dst = [0i32; 256];
    interpolate_luma(
        &ref_i32, stride, stride, h, int_x, int_y, x_frac, y_frac, 16, 16, 8, &mut dst, 16,
    )
    .expect("luma interp");
    dst
}

fn build_inter_pred_chroma_local(
    ref_c: &[u8],
    ref_cw: u32,
    ref_ch: u32,
    mb_x: usize,
    mb_y: usize,
    mv: Mv,
) -> [i32; 64] {
    let stride = ref_cw as usize;
    let h = ref_ch as usize;
    let ref_i32: Vec<i32> = ref_c.iter().take(stride * h).map(|&v| v as i32).collect();
    let int_x = (mb_x * 8) as i32 + (mv.x >> 3);
    let int_y = (mb_y * 8) as i32 + (mv.y >> 3);
    let x_frac = (mv.x & 7) as u8;
    let y_frac = (mv.y & 7) as u8;
    let mut dst = [0i32; 64];
    interpolate_chroma(
        &ref_i32, stride, stride, h, int_x, int_y, x_frac, y_frac, 8, 8, 8, &mut dst, 8,
    )
    .expect("chroma interp");
    dst
}

#[allow(clippy::type_complexity)]
fn encode_chroma_inter_420(
    frame_c: &[u8],
    pred: &[i32; 64],
    chroma_w: usize,
    mb_x: usize,
    mb_y: usize,
    qp_c: i32,
) -> (
    [i32; 4],
    [[i32; 16]; 4],
    [[i32; 16]; 4],
    bool,
    bool,
    [i32; 64],
) {
    let mut residual = [0i32; 64];
    for j in 0..8usize {
        for i in 0..8usize {
            let s = frame_c[(mb_y * 8 + j) * chroma_w + (mb_x * 8 + i)] as i32;
            residual[j * 8 + i] = s - pred[j * 8 + i];
        }
    }
    let mut blocks = [[0i32; 16]; 4];
    for blk in 0..4usize {
        let bx = blk % 2;
        let by = blk / 2;
        let mut block = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                block[j * 4 + i] = residual[(by * 4 + j) * 8 + bx * 4 + i];
            }
        }
        blocks[blk] = forward_core_4x4(&block);
    }
    let dc_in = [blocks[0][0], blocks[1][0], blocks[2][0], blocks[3][0]];
    let dc_had = forward_hadamard_2x2(&dc_in);
    let dc_levels = quantize_chroma_dc(&dc_had, qp_c, false);
    let mut ac_scan = [[0i32; 16]; 4];
    let mut ac_quant = [[0i32; 16]; 4];
    let mut any_ac_nz = false;
    for blk in 0..4usize {
        let z = quantize_4x4_ac(&blocks[blk], qp_c, false);
        ac_quant[blk] = z;
        let s = zigzag_scan_4x4_ac(&z);
        ac_scan[blk] = s;
        if s.iter().any(|&v| v != 0) {
            any_ac_nz = true;
        }
    }
    let any_dc_nz = dc_levels.iter().any(|&v| v != 0);
    let inv_dc = inverse_hadamard_chroma_dc_420(&dc_levels, qp_c, &FLAT_4X4_16, 8).unwrap();
    let mut recon_res = [0i32; 64];
    for blk in 0..4usize {
        let bx = blk % 2;
        let by = blk / 2;
        let mut coeffs_in = ac_quant[blk];
        coeffs_in[0] = inv_dc[blk];
        let r = inverse_transform_4x4_dc_preserved(&coeffs_in, qp_c, &FLAT_4X4_16, 8).unwrap();
        for j in 0..4 {
            for i in 0..4 {
                recon_res[(by * 4 + j) * 8 + bx * 4 + i] = r[j * 4 + i];
            }
        }
    }
    (
        dc_levels, ac_scan, ac_quant, any_dc_nz, any_ac_nz, recon_res,
    )
}

fn chroma_residual_dc_only(dc: &[i32; 4], qp_c: i32) -> [i32; 64] {
    let inv_dc = inverse_hadamard_chroma_dc_420(dc, qp_c, &FLAT_4X4_16, 8).unwrap();
    let mut out = [0i32; 64];
    for blk in 0..4usize {
        let bx = blk % 2;
        let by = blk / 2;
        let mut coeffs_in = [0i32; 16];
        coeffs_in[0] = inv_dc[blk];
        let r = inverse_transform_4x4_dc_preserved(&coeffs_in, qp_c, &FLAT_4X4_16, 8).unwrap();
        for j in 0..4 {
            for i in 0..4 {
                out[(by * 4 + j) * 8 + bx * 4 + i] = r[j * 4 + i];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Round-33 — 4:4:4 chroma plane CABAC encode helper (used by encode_idr_cabac
// when chroma_format_idc == 3). Each chroma plane is coded "like luma" per
// §7.3.5.3 / §8.5.10: a 16-entry DC Hadamard block + 16 4x4 AC blocks in
// §6.4.3 raster-Z order, using the luma Intra_16x16 prediction mode.
// ---------------------------------------------------------------------------

/// Encode one 4:4:4 chroma plane (Cb or Cr) for an Intra_16x16 CABAC MB.
///
/// Returns `(dc_levels[16], ac_scan[16][16], ac_quant_raster[16][16],
///           any_ac_nz, recon_444[256])`.
///
/// `recon_c` contains the prior-MB recon (for predictor). After this call
/// the caller should write the 16x16 tile into the chroma recon buffer.
/// The tile is returned in `recon_444` already as pred+residual.
#[allow(clippy::type_complexity)]
fn encode_chroma_intra16x16_444(
    src_plane: &[u8],
    recon_c: &mut [u8],
    chroma_width: usize,
    chroma_height: usize,
    mb_x: usize,
    mb_y: usize,
    mode: I16x16Mode,
    qp_c: i32,
) -> (
    [i32; 16],
    [[i32; 16]; 16],
    [[i32; 16]; 16],
    bool,
    [i32; 256],
) {
    use crate::encoder::transform::{forward_core_4x4, forward_hadamard_4x4, quantize_luma_dc};
    use crate::transform::{inverse_hadamard_luma_dc_16x16, inverse_transform_4x4_dc_preserved};

    // (1) Predictor from chroma recon plane (treated like luma).
    let pred = predict_16x16(mode, recon_c, chroma_width, chroma_height, mb_x, mb_y)
        .unwrap_or([128i32; 256]);

    // (2) Residual.
    let mut residual = [0i32; 256];
    for j in 0..16usize {
        for i in 0..16usize {
            let s = src_plane[(mb_y * 16 + j) * chroma_width + (mb_x * 16 + i)] as i32;
            residual[j * 16 + i] = s - pred[j * 16 + i];
        }
    }

    // (3) Per-4x4 forward integer transform; pull DCs.
    let mut coeffs_4x4 = [[0i32; 16]; 16];
    for blk in 0..16usize {
        let bx = blk % 4;
        let by = blk / 4;
        let mut block = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                block[j * 4 + i] = residual[(by * 4 + j) * 16 + bx * 4 + i];
            }
        }
        coeffs_4x4[blk] = forward_core_4x4(&block);
    }
    let mut dc_coeffs = [0i32; 16];
    for (blk, c) in coeffs_4x4.iter().enumerate() {
        let bx = blk % 4;
        let by = blk / 4;
        dc_coeffs[by * 4 + bx] = c[0];
    }
    let dc_t = forward_hadamard_4x4(&dc_coeffs);
    let dc_levels = quantize_luma_dc(&dc_t, qp_c, true);

    // (4) Per-block AC quantize (raster and scan order).
    let mut ac_quant_raster = [[0i32; 16]; 16];
    let mut ac_scan = [[0i32; 16]; 16];
    let mut any_ac_nz = false;
    for blkz in 0..16usize {
        let (bx, by) = LUMA_4X4_BLK[blkz];
        let raster = by * 4 + bx;
        let z = quantize_4x4_ac(&coeffs_4x4[raster], qp_c, true);
        ac_quant_raster[raster] = z;
        ac_scan[blkz] = zigzag_scan_4x4_ac(&z);
        if ac_scan[blkz].iter().any(|&v| v != 0) {
            any_ac_nz = true;
        }
    }

    // (5) Inverse path → produce decoder-bit-exact recon tile.
    let inv_dc = inverse_hadamard_luma_dc_16x16(&dc_levels, qp_c, &FLAT_4X4_16, 8)
        .expect("4:4:4 chroma DC inverse");
    let mut recon_tile = [0i32; 256];
    #[allow(clippy::needless_range_loop)]
    for blk in 0..16usize {
        let bx = blk % 4;
        let by = blk / 4;
        let mut coeffs_in = if any_ac_nz {
            ac_quant_raster[blk]
        } else {
            [0i32; 16]
        };
        coeffs_in[0] = inv_dc[by * 4 + bx];
        let r = inverse_transform_4x4_dc_preserved(&coeffs_in, qp_c, &FLAT_4X4_16, 8)
            .expect("4:4:4 chroma 4x4 inverse");
        for j in 0..4 {
            for i in 0..4 {
                let p = pred[(by * 4 + j) * 16 + bx * 4 + i];
                let px = mb_x * 16 + bx * 4 + i;
                let py = mb_y * 16 + by * 4 + j;
                let v = (p + r[j * 4 + i]).clamp(0, 255);
                recon_tile[(by * 4 + j) * 16 + bx * 4 + i] = v;
                recon_c[py * chroma_width + px] = v as u8;
            }
        }
    }

    (dc_levels, ac_scan, ac_quant_raster, any_ac_nz, recon_tile)
}

/// Emit a 4:4:4 chroma plane residual over CABAC (Cb or Cr).
///
/// The DC block uses `dc_block_type` (`CbIntra16x16Dc` or `CrIntra16x16Dc`).
/// The AC blocks use `ac_block_type` (`CbIntra16x16Ac` or `CrIntra16x16Ac`).
/// Returns `cbf_dc: bool` and `cbf_ac: [bool; 16]` for grid state updates.
#[allow(clippy::too_many_arguments)]
fn emit_444_chroma_plane_cabac(
    cabac: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    grid: &mut CabacEncGrid,
    mb_x: usize,
    mb_y: usize,
    dc_levels: &[i32; 16],
    ac_scan: &[[i32; 16]; 16],
    cbp_luma: u8,
    dc_block_type: BlockType,
    ac_block_type: BlockType,
) -> (bool, [bool; 16]) {
    // The CBF neighbour for the DC block uses the correct chroma-DC plane.
    // §9.3.3.1.1.9 Table 9-40: CbIntra16x16Dc → blockCat 6, CrIntra16x16Dc → blockCat 9.
    // For the DC context we derive from the appropriate neighbour cbf_cb_dc / cbf_cr_dc.
    let dc_plane = match dc_block_type {
        BlockType::CrIntra16x16Dc => CbfPlane::CrDc,
        _ => CbfPlane::CbDc,
    };
    let (cbf_a, cbf_b) = cbf_neighbour_mb_level(grid, mb_x, mb_y, true, dc_plane);
    let dc_scan = zigzag_scan_4x4(dc_levels);
    let cbf_dc = encode_residual_block_cabac(
        cabac,
        ctxs,
        dc_block_type,
        &dc_scan,
        16,
        Some(cbf_a),
        Some(cbf_b),
        false,
    );

    let mut cbf_ac = [false; 16];
    if cbp_luma == 15 {
        let is_cr = matches!(
            ac_block_type,
            BlockType::CrIntra16x16Ac | BlockType::CrLuma4x4
        );
        for blkz in 0..16usize {
            let cur_snap = grid.at(mb_x, mb_y).clone();
            let (ca, cb) =
                cbf_neighbour_luma_ac(grid, mb_x, mb_y, &cur_snap, true, ac_block_type, blkz as u8);
            let coded = encode_residual_block_cabac(
                cabac,
                ctxs,
                ac_block_type,
                &ac_scan[blkz][..15],
                15,
                Some(ca),
                Some(cb),
                false,
            );
            cbf_ac[blkz] = coded;
            // Update the plane-specific 4:4:4 chroma AC CBF array so
            // intra-MB neighbours can read the correct CBF state.
            if is_cr {
                grid.at_mut(mb_x, mb_y).cbf_cr_ac_444[blkz] |= coded;
            } else {
                grid.at_mut(mb_x, mb_y).cbf_cb_ac_444[blkz] |= coded;
            }
        }
    }

    (cbf_dc, cbf_ac)
}
