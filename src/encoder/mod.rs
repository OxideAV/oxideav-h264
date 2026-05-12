//! H.264 / AVC **encoder** — Baseline profile, IDR-only, with mb_type
//! switching between **I_16x16** (DC/V/H/Plane) and **I_NxN** (Intra_4x4
//! with all 9 §8.3.1.2 modes) based on per-MB SAD.
//!
//! Round-4 scope (round-1 DC-luma only; round-2 chroma residual +
//! multi-mode I_16x16; round-3 luma AC residual; round-4 I_NxN; see
//! `STATUS.md`):
//!
//! * **Profile:** Baseline (`profile_idc = 66`), 4:2:0, 8-bit, single
//!   slice per picture, no FMO.
//! * **Slice types:** IDR I-slices only.
//! * **Macroblock types:** per-MB choice between `I_16x16` (4 modes)
//!   and `I_NxN` / Intra_4x4 (9 per-block modes). The lower-SAD path
//!   wins. mb_type is signalled via Table 7-11.
//! * **Chroma intra:** §8.3.4 four modes (DC, V, H, Plane) selected per
//!   MB by minimum-SAD.
//! * **Transforms:** 4x4 integer transform + 4x4 luma-DC Hadamard
//!   (Intra_16x16 only) + 2x2 chroma-DC Hadamard. Flat scaling lists.
//! * **CBP:**
//!   - Intra_16x16: `cbp_luma` ∈ {0, 15} (per §7.4.5.1) — 15 whenever
//!     any 4x4 luma AC has a non-zero coefficient.
//!   - I_NxN: `cbp_luma` ∈ 0..=15, one bit per 8x8 quadrant.
//!   - `cbp_chroma` ∈ {0, 1, 2}.
//! * **Entropy:** CAVLC, with per-MB neighbour grid (`CavlcNcGrid`)
//!   driving §9.2.1.1 nC derivation for AC blocks.
//! * **In-loop filter:** §8.7 deblock applied to local recon as a
//!   picture-level pass after every MB has been emitted (round-14).
//!   `disable_deblocking_filter_idc = 0`, alpha/beta offsets = 0.
//!
//! Spec references throughout cite ITU-T Rec. H.264 (08/2024).

#![allow(dead_code)]

pub mod bitstream;
pub mod cabac_engine;
pub mod cabac_path;
pub mod cabac_syntax;
pub mod cavlc;
pub mod deblock;
pub mod intra4x4;
pub mod intra_pred;
pub mod macroblock;
pub mod me;
pub mod nal;
pub mod pps;
pub mod rdo;
pub mod slice;
pub mod sps;
pub mod transform;

use crate::cavlc::CoeffTokenContext;
use crate::encoder::bitstream::BitWriter;
use crate::encoder::cavlc::encode_residual_block_cavlc;
use crate::encoder::deblock::{
    chroma_nz_mask_from_blocks, deblock_recon, luma_nz_mask_from_blocks, MbDeblockInfo,
};
use crate::encoder::intra4x4::{
    availability as i4x4_avail, predict_4x4, Intra4x4Mode, LUMA_4X4_XY as I4X4_XY,
};
use crate::encoder::intra_pred::{
    predict_16x16, predict_chroma_8x8, sad_8x8, I16x16Mode, IntraChromaMode,
};
use crate::encoder::macroblock::{
    write_b_16x16_mb, write_b_8x8_mixed_mb, write_i_nxn_mb, write_intra16x16_mb,
    write_intra16x16_mb_in_inter_slice, write_p_8x8_all_pl08x8_mb, write_p_l0_16x16_mb,
    B16x16McbConfig, B8x8MixedMcbConfig, BPartPred, BPred16x16, BSubMbCell, I16x16McbConfig,
    INxNMcbConfig, P8x8AllPL08x8McbConfig, PL016x16McbConfig,
};
use crate::encoder::me::{search_quarter_pel_16x16, search_quarter_pel_8x8};
use crate::encoder::nal::build_nal_unit;
use crate::encoder::pps::{build_baseline_pps_rbsp, BaselinePpsConfig};
use crate::encoder::rdo::{cost_combined, lambda_ssd, ssd_16x16, ssd_4x4};
use crate::encoder::slice::{
    write_b_slice_header, write_idr_i_slice_header, write_p_slice_header, BSliceHeaderConfig,
    ExplicitBipredWeightTable, IdrSliceHeaderConfig, PSliceHeaderConfig,
};
use crate::encoder::sps::{build_baseline_sps_rbsp, BaselineSpsConfig};
use crate::encoder::transform::{
    forward_core_4x4, forward_hadamard_2x2, forward_hadamard_4x4, quantize_4x4, quantize_4x4_ac,
    quantize_chroma_dc, quantize_luma_dc, zigzag_scan_4x4, zigzag_scan_4x4_ac,
};
use crate::inter_pred::{interpolate_chroma, interpolate_luma};
use crate::macroblock_layer::{derive_nc_luma, CavlcNcGrid, LumaNcKind};
use crate::mv_deriv::{Mv, MvpredInputs, NeighbourMv};
use crate::nal::NalUnitType;
use crate::transform::{
    inverse_hadamard_chroma_dc_420, inverse_hadamard_luma_dc_16x16, inverse_scan_4x4_zigzag_ac,
    inverse_transform_4x4, inverse_transform_4x4_dc_preserved, qp_bd_offset,
    qp_y_to_qp_c_with_bd_offset, FLAT_4X4_16,
};

/// §6.4.3 / Figure 6-10 — luma 4x4 block scan: index 0..=15 → (x, y)
/// offset of the block inside a macroblock (in 4-sample units, *not*
/// pixels). This mirrors `crate::reconstruct::LUMA_4X4_XY` but in
/// 4x4-block coordinates so the encoder can index the per-block AC
/// arrays directly.
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

/// One encoded YUV frame's worth of input planes. All planes are
/// 8-bit, row-major. Width and height must be multiples of 16 (no
/// cropping support yet).
///
/// Default chroma layout is 4:2:0 — `u` / `v` length is `(W/2) *
/// (H/2)`. When the encoder is configured for 4:2:2
/// (`EncoderConfig::chroma_format_idc = 2`, round 27), `u` / `v` length
/// is `(W/2) * H` — half-width, full-height per §6.2 Table 6-1.
pub struct YuvFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
}

/// Encoder configuration.
#[derive(Debug, Clone, Copy)]
pub struct EncoderConfig {
    pub width: u32,
    pub height: u32,
    /// Slice QP_Y. Range 0..=51. Default 26 (PPS pic_init_qp_minus26 = 0
    /// + slice_qp_delta = 0).
    pub qp: i32,
    /// §7.4.2.1 — `profile_idc` for the emitted SPS. Default 66
    /// (Baseline). Set to 77 (Main) when calling [`Encoder::encode_b`]
    /// because §A.2.2 forbids B-slices in Baseline.
    pub profile_idc: u8,
    /// §7.4.2.1.1 — `max_num_ref_frames`. Default 1 (one reference,
    /// sufficient for IDR + linear P chain). Round-20 B-slice support
    /// needs 2 (the prior IDR is the L0 reference, the prior P-frame
    /// is the L1 reference, both must remain in the DPB until the B
    /// MB has been decoded).
    pub max_num_ref_frames: u32,
    /// §7.4.3 / §8.4.1.2 — when `false` (default) the encoder emits
    /// `direct_spatial_mv_pred_flag = 1` in the B-slice header and
    /// drives B_Skip / B_Direct via §8.4.1.2.2 spatial direct
    /// derivation (round 21 / 23). When `true`, the encoder emits
    /// `direct_spatial_mv_pred_flag = 0` and uses §8.4.1.2.3
    /// **temporal direct** derivation: per-8x8 (mvL0, mvL1, refIdxL0,
    /// refIdxL1) are computed by POC-scaling the colocated L1
    /// anchor's L0 motion vectors. Round-24.
    pub direct_temporal_mv_pred: bool,
    /// Round-26 — §7.4.2.2 / §8.4.2.3.2 explicit weighted bipred for B
    /// slices. When `true` the encoder:
    ///   * Sets PPS `weighted_bipred_idc = 1` so the decoder pulls a
    ///     `pred_weight_table()` from each B slice header.
    ///   * Selects per-slice (luma_weight_l0, luma_weight_l1,
    ///     luma_offset_l0, luma_offset_l1) at `luma_log2_weight_denom = 5`
    ///     via the §8.4.2.3.2 forward least-squares fit
    ///     ([`compute_explicit_bipred_weights`]).
    ///   * Replaces the bipred merge in `build_b_predictors` with the
    ///     §8.4.2.3.2 eq. 8-276 explicit formula so the local recon
    ///     stays bit-equivalent to what the decoder produces.
    ///
    /// When `false` (default) bipred uses the §8.4.2.3.1 default
    /// average `(L0 + L1 + 1) >> 1` and the PPS / slice header carry no
    /// pred_weight_table.
    pub explicit_weighted_bipred: bool,
    /// Round-29 — when `true` (default) the P/B-slice encoder runs an
    /// **intra fallback** RDO trial on every coded MB: the inter MB cost
    /// `J_inter = D_inter + λ·R_inter` is compared against an Intra_16x16
    /// trial cost `J_intra`, and the lower-J path wins. This covers
    /// occlusion, scene-cut frames, and high-detail areas where motion-
    /// compensated prediction's residual is more expensive to code than
    /// a fresh intra block. Per H.264 §7.3.5 (Tables 7-13 / 7-14) the
    /// chosen MB is emitted with the intra mb_type values shifted by +5
    /// (P-slice) or +23 (B-slice) so the decoder picks up Intra_16x16
    /// from the inter-slice mb_type tables.
    ///
    /// When `false`, P/B slices are inter-only (`P_Skip` / `P_L0_*` /
    /// `B_Skip` / `B_*_*`) — the round-16..28 behaviour. Useful for
    /// A/B tests against the new path.
    ///
    /// Round-29 scope: I_16x16 only (no I_NxN / I_PCM fallback). The
    /// intra trial runs only on coded inter MBs (P_Skip / B_Skip retain
    /// their zero-syntax fast path).
    pub intra_in_inter: bool,
    /// §7.4.2.1.1 / §6.2 — `chroma_format_idc`. Defaults to 1 (4:2:0).
    /// Round-27 supports 2 (4:2:2) on the `encode_idr` path: the
    /// emitted SPS is High 4:2:2 (profile_idc=122), the chroma plane
    /// is 8x16 per MB (two stacked 8x8 tiles), the chroma DC Hadamard
    /// is 2x4 per §8.5.11.2, and the deblock pass adapts via
    /// `Picture::chroma_array_type`.
    ///
    /// Round-28 adds 3 (4:4:4) on `encode_idr` (Intra_16x16 only):
    /// the emitted SPS is High 4:4:4 Predictive (profile_idc=244), the
    /// chroma plane is the full 16x16 per MB, each plane is encoded
    /// "like luma" (16x16 DC Hadamard + 16 4x4 AC blocks per plane —
    /// §7.3.5.3 / §8.5.10), and `intra_chroma_pred_mode` is omitted
    /// (chroma reuses the luma Intra_16x16 mode per §8.3.4.1). To
    /// keep round-28 scope manageable: I_NxN is disabled, deblocking
    /// is turned off (`disable_deblocking_filter_idc=1`), and
    /// `separate_colour_plane_flag` stays at 0 (separate planes are
    /// out of scope). Other values (0 monochrome) and the P/B-slice
    /// paths remain 4:2:0-only.
    pub chroma_format_idc: u32,
    /// Round-30 — when `true`, the encoder emits CABAC entropy coding
    /// (`entropy_coding_mode_flag = 1` in the PPS). Use the
    /// [`Encoder::encode_idr_cabac`] / [`Encoder::encode_p_cabac`]
    /// entry points; the existing CAVLC paths
    /// ([`Encoder::encode_idr`] / [`Encoder::encode_p`]) ignore this
    /// flag and continue emitting CAVLC.
    ///
    /// CABAC is forbidden by Baseline profile (§A.2.1); enabling this
    /// flag requires `profile_idc >= 77` (Main).
    pub cabac: bool,
    /// §7.4.2.1.1 / Annex A Table A-1 — `level_idc` to emit in the
    /// SPS. Default: derived from `(width, height)` so the smallest
    /// level whose `MaxFS` covers the picture is selected. A 720p
    /// stream rounds up to 31 (Level 3.1, MaxFS = 3600 mb), 1080p to
    /// 40 (Level 4, MaxFS = 8192 mb), 4K UHD to 51 (Level 5.1, MaxFS
    /// = 36864 mb). Override when you know the actual constraint set
    /// for the deployment (e.g. fixing a Level 4.0 cap because a
    /// downstream decoder doesn't support 4.1+).
    ///
    /// Pre-config-flag behaviour was a hard-coded `level_idc = 30`
    /// (Level 3, MaxFS 1620 mb = 720x576 SD). Encoding 720p / 1080p
    /// content with that signalling produced spec-non-conformant
    /// streams (PicSize > MaxFS) — strict decoders are entitled to
    /// reject them. The new derivation closes that gap without
    /// changing any other encoder behaviour.
    pub level_idc: u8,
    /// §7.4.2.1.1 — `bit_depth_chroma_minus8`. Defaults to 0 (8-bit
    /// chroma). The current SPS writer ([`crate::encoder::sps`]) pins
    /// the emitted bit_depth fields at 0 — High10/422/444 with bit
    /// depth >8 is out of round-30 scope. This field is plumbed through
    /// the encoder state so that the §8.5.8 chroma-QP derivation calls
    /// the BD-aware [`crate::transform::qp_y_to_qp_c_with_bd_offset`]
    /// with the correct `qp_bd_offset_c = 6 * bit_depth_chroma_minus8`.
    /// When a future round extends the SPS to emit non-zero bit_depth
    /// fields, the encoder's QPC math is already wired to follow.
    pub bit_depth_chroma_minus8: u32,
    /// Round-49 — trellis quantisation refinement (a.k.a. RDOQ-lite)
    /// for inter AC residual blocks. When `true` (default), after the
    /// open-loop `quantize_4x4` step in the CABAC inter path the
    /// encoder runs [`crate::encoder::transform::trellis_refine_4x4_ac`]
    /// over each 4x4 luma + chroma AC block to refine quantised levels
    /// toward zero whenever the resulting `D + λ·R` cost is strictly
    /// lower. The refinement is informative (§9 is silent on forward
    /// quant strategy beyond the round-trip property) so the decoder
    /// is unaffected.
    ///
    /// Empirical effect on the round-25 mixed-motion B-slice fixture:
    ///   * CABAC B-slice: 12 → 11 bytes (~8 %), PSNR_Y 44.20 → 44.20 dB
    ///     (no measurable loss; trellis only flips coefficients whose
    ///     rate cost dominates their distortion contribution).
    ///   * Wider GOPs on textured content see ~5-12 % saving on the
    ///     coded inter MBs.
    pub trellis_quant: bool,
}

impl EncoderConfig {
    pub fn new(width: u32, height: u32) -> Self {
        let width_in_mbs = width.div_ceil(16);
        let height_in_mbs = height.div_ceil(16);
        Self {
            width,
            height,
            qp: 26,
            profile_idc: 66,
            max_num_ref_frames: 1,
            direct_temporal_mv_pred: false,
            explicit_weighted_bipred: false,
            intra_in_inter: true,
            chroma_format_idc: 1,
            cabac: false,
            level_idc: min_level_idc_for_picture_size(width_in_mbs, height_in_mbs),
            bit_depth_chroma_minus8: 0,
            trellis_quant: true,
        }
    }
}

/// Annex A Table A-1 — pick the smallest `level_idc` whose MaxFS
/// (max frame size in macroblocks) is at least `width_in_mbs *
/// height_in_mbs`. Used by [`EncoderConfig::new`] to default the
/// emitted SPS `level_idc` so 720p / 1080p / 4K encodes don't
/// silently signal a too-low Level 3 (the historical hard-coded
/// default).
///
/// The table walks from low to high so the caller could later cap
/// to a downstream decoder's max supported level if needed. We
/// include Levels 6.0 / 6.1 / 6.2 (added in the 2017+ revisions,
/// MaxFS = 139264 each) so 8K UHD (480x270 mb) + larger settle on
/// a real level rather than the previous 5.2 ceiling. Pictures
/// beyond Level 6.2 (the highest in the 2024-08 edition) fall off
/// the end and return 62 (the encoder will still emit it; whether
/// a decoder honours it is its problem).
///
/// MaxFS values are pulled directly from §A.3.1 Table A-1.
pub(crate) fn min_level_idc_for_picture_size(width_in_mbs: u32, height_in_mbs: u32) -> u8 {
    let pic_size = width_in_mbs.saturating_mul(height_in_mbs);
    // (level_idc, MaxFS) walking low→high; first hit wins.
    const LEVELS: &[(u8, u32)] = &[
        (10, 99),      // Level 1
        (11, 396),     // Level 1.1
        (12, 396),     // Level 1.2
        (13, 396),     // Level 1.3
        (20, 396),     // Level 2
        (21, 792),     // Level 2.1
        (22, 1_620),   // Level 2.2
        (30, 1_620),   // Level 3
        (31, 3_600),   // Level 3.1 (~720p)
        (32, 5_120),   // Level 3.2
        (40, 8_192),   // Level 4 (~1080p)
        (41, 8_192),   // Level 4.1
        (42, 8_704),   // Level 4.2
        (50, 22_080),  // Level 5 (~2.5K)
        (51, 36_864),  // Level 5.1 (~4K UHD)
        (52, 36_864),  // Level 5.2
        (60, 139_264), // Level 6 (~8K UHD)
        (61, 139_264), // Level 6.1
        (62, 139_264), // Level 6.2
    ];
    for &(level, max_fs) in LEVELS {
        if pic_size <= max_fs {
            return level;
        }
    }
    // Pictures beyond every standardised level — emit the highest
    // anyway so the SPS is at least syntactically valid; a strict
    // decoder is entitled to reject the level + size pairing.
    62
}

/// Encoder-side per-MB grid for the §8.3.1.1 / §6.4.11.4 neighbour
/// lookup needed by I_NxN mode prediction. Mirrors the small subset
/// of fields the decoder's [`crate::reconstruct::MbGrid`] tracks for
/// intra prediction. The encoder must agree with the decoder
/// bit-for-bit on the predicted Intra4x4 mode of every block,
/// otherwise `prev_intra4x4_pred_mode_flag = 1` would mean a different
/// mode on each side.
#[derive(Debug, Clone, Default)]
struct IntraGridSlot {
    available: bool,
    is_i_nxn: bool,
    /// `intra_4x4_pred_modes[i]` per §6.4.3 raster-Z order. Only
    /// populated when `is_i_nxn`.
    intra_4x4_pred_modes: [u8; 16],
}

#[derive(Debug, Clone)]
struct IntraGrid {
    width_mbs: usize,
    slots: Vec<IntraGridSlot>,
}

impl IntraGrid {
    fn new(width_mbs: usize, height_mbs: usize) -> Self {
        Self {
            width_mbs,
            slots: vec![IntraGridSlot::default(); width_mbs * height_mbs],
        }
    }
    fn slot(&self, mb_x: usize, mb_y: usize) -> &IntraGridSlot {
        &self.slots[mb_y * self.width_mbs + mb_x]
    }
    fn slot_mut(&mut self, mb_x: usize, mb_y: usize) -> &mut IntraGridSlot {
        &mut self.slots[mb_y * self.width_mbs + mb_x]
    }
}

/// §6.4.11.4 + §8.3.1.1 — predict the Intra_4x4 mode for block
/// `block_idx` of the MB at (mb_x, mb_y) using the modes of neighbour
/// 4x4 blocks A (left) and B (above).
///
/// Returns the §8.3.1.1 step 4 `predIntra4x4PredMode` value (0..=8).
///
/// Algorithm (mirrors decoder's `derive_intra_4x4_pred_mode`):
///   1. Compute (xN, yN) = (bx-1, by) for A and (bx, by-1) for B in
///      MB-relative pixel coordinates of the current 4x4 block.
///   2. Locate the neighbour MB and the neighbour 4x4 block index per
///      §6.4.3 / Table 6-3 / eq. 6-38 (non-MBAFF frame case).
///   3. dcPredModePredictedFlag (step 2): true iff either neighbour MB
///      is unavailable OR (constrained_intra_pred && neighbour is not
///      intra). We don't use constrained_intra_pred → only the
///      availability check.
///   4. modeN (step 3): 2 (DC) when dcPredFlag, else
///      `intra_4x4_pred_modes[blkN]` of the neighbour MB if it's
///      I_NxN, else 2 (DC) for I_16x16/I_PCM/inter neighbours.
///   5. predicted = min(modeA, modeB).
fn predicted_intra4x4_mode(
    grid: &IntraGrid,
    mb_x: usize,
    mb_y: usize,
    block_idx: usize,
    pic_w_mbs: usize,
) -> u8 {
    use crate::encoder::intra4x4::LUMA_4X4_XY as XY;
    let (bx, by) = XY[block_idx];

    let neigh = |xd: i32, yd: i32| -> Option<(usize, usize, usize)> {
        // Eq. 6-25 / 6-26 (block-of-4-samples xN, yN inside MB grid).
        let xn = bx as i32 + xd;
        let yn = by as i32 + yd;
        // Table 6-3 (non-MBAFF frame): pick neighbour MB.
        let (nmb_x, nmb_y) = if xn < 0 && (0..16).contains(&yn) {
            // mbAddrA: left MB.
            if mb_x == 0 {
                return None;
            }
            (mb_x - 1, mb_y)
        } else if (0..16).contains(&xn) && yn < 0 {
            // mbAddrB: above MB.
            if mb_y == 0 {
                return None;
            }
            (mb_x, mb_y - 1)
        } else if (0..16).contains(&xn) && (0..16).contains(&yn) {
            (mb_x, mb_y)
        } else {
            return None;
        };
        // Eq. 6-34 / 6-35 — coords inside neighbour MB.
        let xw = ((xn + 16) % 16) as usize;
        let yw = ((yn + 16) % 16) as usize;
        // Eq. 6-38.
        let nblk = 8 * (yw / 8) + 4 * (xw / 8) + 2 * ((yw % 8) / 4) + ((xw % 8) / 4);
        Some((nmb_x, nmb_y, nblk))
    };

    let na = neigh(-1, 0);
    let nb = neigh(0, -1);

    let mb_a_avail = na
        .and_then(|(nx, ny, _)| Some(grid.slot(nx, ny)).filter(|s| s.available))
        .is_some();
    let mb_b_avail = nb
        .and_then(|(nx, ny, _)| Some(grid.slot(nx, ny)).filter(|s| s.available))
        .is_some();
    let dc_pred_flag = !mb_a_avail || !mb_b_avail;

    let mode_for = |neigh: Option<(usize, usize, usize)>| -> u8 {
        if dc_pred_flag {
            return 2;
        }
        let Some((nx, ny, nblk)) = neigh else {
            return 2;
        };
        let s = grid.slot(nx, ny);
        if !s.available {
            return 2;
        }
        if s.is_i_nxn {
            s.intra_4x4_pred_modes[nblk]
        } else {
            // Intra_16x16 / I_PCM / inter neighbour → DC fallback.
            2
        }
    };
    let mode_a = mode_for(na);
    let mode_b = mode_for(nb);
    let _ = pic_w_mbs;
    mode_a.min(mode_b)
}

/// Per-MB output bundle from a trial encode (round-15 RDO).
///
/// `luma_cost` is the Lagrangian cost `J = D + λ · R` of the MB's
/// **luma** path only (chroma is identical for both I_16x16 and I_NxN
/// trials so we exclude it from the comparison). `deblock` is the
/// per-MB info needed by the §8.7 picture-level filter.
struct MbTrial {
    luma_cost: u64,
    deblock: MbDeblockInfo,
}

/// Snapshot of all encoder state mutated by one MB's encode, captured
/// inside `encode_mb` so we can roll back the losing trial.
///
/// Captures only the MB-sized window of the recon planes, which keeps
/// allocations small (256 + 64 + 64 = 384 bytes per snapshot) regardless
/// of picture size.
struct MbStateSnapshot {
    recon_y_mb: [u8; 256],
    recon_u_mb: [u8; 64],
    recon_v_mb: [u8; 64],
    sw: BitWriter,
    nc_slot: crate::macroblock_layer::CavlcMbNc,
    intra_slot: IntraGridSlot,
}

impl MbStateSnapshot {
    #[allow(clippy::too_many_arguments)]
    fn capture(
        recon_y: &[u8],
        recon_u: &[u8],
        recon_v: &[u8],
        width: usize,
        chroma_width: usize,
        mb_x: usize,
        mb_y: usize,
        sw: &BitWriter,
        nc_grid: &CavlcNcGrid,
        intra_grid: &IntraGrid,
        mb_addr: usize,
        _width_mbs: usize,
    ) -> Self {
        let mut snap = Self {
            recon_y_mb: [0u8; 256],
            recon_u_mb: [0u8; 64],
            recon_v_mb: [0u8; 64],
            sw: sw.clone(),
            nc_slot: nc_grid.mbs[mb_addr],
            intra_slot: intra_grid.slot(mb_x, mb_y).clone(),
        };
        for j in 0..16usize {
            let src = (mb_y * 16 + j) * width + mb_x * 16;
            snap.recon_y_mb[j * 16..j * 16 + 16].copy_from_slice(&recon_y[src..src + 16]);
        }
        for j in 0..8usize {
            let src = (mb_y * 8 + j) * chroma_width + mb_x * 8;
            snap.recon_u_mb[j * 8..j * 8 + 8].copy_from_slice(&recon_u[src..src + 8]);
            snap.recon_v_mb[j * 8..j * 8 + 8].copy_from_slice(&recon_v[src..src + 8]);
        }
        snap
    }

    #[allow(clippy::too_many_arguments)]
    fn restore(
        &self,
        recon_y: &mut [u8],
        recon_u: &mut [u8],
        recon_v: &mut [u8],
        width: usize,
        chroma_width: usize,
        mb_x: usize,
        mb_y: usize,
        sw: &mut BitWriter,
        nc_grid: &mut CavlcNcGrid,
        intra_grid: &mut IntraGrid,
        mb_addr: usize,
    ) {
        for j in 0..16usize {
            let dst = (mb_y * 16 + j) * width + mb_x * 16;
            recon_y[dst..dst + 16].copy_from_slice(&self.recon_y_mb[j * 16..j * 16 + 16]);
        }
        for j in 0..8usize {
            let dst = (mb_y * 8 + j) * chroma_width + mb_x * 8;
            recon_u[dst..dst + 8].copy_from_slice(&self.recon_u_mb[j * 8..j * 8 + 8]);
            recon_v[dst..dst + 8].copy_from_slice(&self.recon_v_mb[j * 8..j * 8 + 8]);
        }
        *sw = self.sw.clone();
        nc_grid.mbs[mb_addr] = self.nc_slot;
        *intra_grid.slot_mut(mb_x, mb_y) = self.intra_slot.clone();
    }

    #[allow(clippy::too_many_arguments)]
    fn install(
        &self,
        recon_y: &mut [u8],
        recon_u: &mut [u8],
        recon_v: &mut [u8],
        width: usize,
        chroma_width: usize,
        mb_x: usize,
        mb_y: usize,
        sw: &mut BitWriter,
        nc_grid: &mut CavlcNcGrid,
        intra_grid: &mut IntraGrid,
        mb_addr: usize,
    ) {
        // Same as restore — fields are owned post-trial state to install.
        self.restore(
            recon_y,
            recon_u,
            recon_v,
            width,
            chroma_width,
            mb_x,
            mb_y,
            sw,
            nc_grid,
            intra_grid,
            mb_addr,
        );
    }
}

/// Round-29 — extra snapshot fields needed to roll back a P/B-slice MB
/// trial. Wraps [`MbStateSnapshot`] with the inter-slice-only MV grid
/// slot and `pending_skip` counter so the intra-fallback RDO can
/// trial-encode a candidate MB and discard the loser without leaking
/// state into the next MB.
struct InterMbStateSnapshot {
    inner: MbStateSnapshot,
    mv_l0: MvGridSlot,
    /// `Some` only on the B-slice path — L1 grid slot.
    mv_l1: Option<MvGridSlot>,
    pending_skip: u32,
}

impl InterMbStateSnapshot {
    #[allow(clippy::too_many_arguments)]
    fn capture_p(
        recon_y: &[u8],
        recon_u: &[u8],
        recon_v: &[u8],
        width: usize,
        chroma_width: usize,
        mb_x: usize,
        mb_y: usize,
        sw: &BitWriter,
        nc_grid: &CavlcNcGrid,
        intra_grid: &IntraGrid,
        mv_grid: &MvGrid,
        mb_addr: usize,
        width_mbs: usize,
        pending_skip: u32,
    ) -> Self {
        Self {
            inner: MbStateSnapshot::capture(
                recon_y,
                recon_u,
                recon_v,
                width,
                chroma_width,
                mb_x,
                mb_y,
                sw,
                nc_grid,
                intra_grid,
                mb_addr,
                width_mbs,
            ),
            mv_l0: *mv_grid.slot(mb_x, mb_y),
            mv_l1: None,
            pending_skip,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_b(
        recon_y: &[u8],
        recon_u: &[u8],
        recon_v: &[u8],
        width: usize,
        chroma_width: usize,
        mb_x: usize,
        mb_y: usize,
        sw: &BitWriter,
        nc_grid: &CavlcNcGrid,
        intra_grid: &IntraGrid,
        mv_grid_l0: &MvGrid,
        mv_grid_l1: &MvGrid,
        mb_addr: usize,
        width_mbs: usize,
        pending_skip: u32,
    ) -> Self {
        Self {
            inner: MbStateSnapshot::capture(
                recon_y,
                recon_u,
                recon_v,
                width,
                chroma_width,
                mb_x,
                mb_y,
                sw,
                nc_grid,
                intra_grid,
                mb_addr,
                width_mbs,
            ),
            mv_l0: *mv_grid_l0.slot(mb_x, mb_y),
            mv_l1: Some(*mv_grid_l1.slot(mb_x, mb_y)),
            pending_skip,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn restore_p(
        &self,
        recon_y: &mut [u8],
        recon_u: &mut [u8],
        recon_v: &mut [u8],
        width: usize,
        chroma_width: usize,
        mb_x: usize,
        mb_y: usize,
        sw: &mut BitWriter,
        nc_grid: &mut CavlcNcGrid,
        intra_grid: &mut IntraGrid,
        mv_grid: &mut MvGrid,
        mb_addr: usize,
        pending_skip: &mut u32,
    ) {
        self.inner.restore(
            recon_y,
            recon_u,
            recon_v,
            width,
            chroma_width,
            mb_x,
            mb_y,
            sw,
            nc_grid,
            intra_grid,
            mb_addr,
        );
        *mv_grid.slot_mut(mb_x, mb_y) = self.mv_l0;
        *pending_skip = self.pending_skip;
    }

    #[allow(clippy::too_many_arguments)]
    fn restore_b(
        &self,
        recon_y: &mut [u8],
        recon_u: &mut [u8],
        recon_v: &mut [u8],
        width: usize,
        chroma_width: usize,
        mb_x: usize,
        mb_y: usize,
        sw: &mut BitWriter,
        nc_grid: &mut CavlcNcGrid,
        intra_grid: &mut IntraGrid,
        mv_grid_l0: &mut MvGrid,
        mv_grid_l1: &mut MvGrid,
        mb_addr: usize,
        pending_skip: &mut u32,
    ) {
        self.inner.restore(
            recon_y,
            recon_u,
            recon_v,
            width,
            chroma_width,
            mb_x,
            mb_y,
            sw,
            nc_grid,
            intra_grid,
            mb_addr,
        );
        *mv_grid_l0.slot_mut(mb_x, mb_y) = self.mv_l0;
        if let Some(slot) = self.mv_l1 {
            *mv_grid_l1.slot_mut(mb_x, mb_y) = slot;
        }
        *pending_skip = self.pending_skip;
    }
}

/// Top-level encoder. Holds parameter set state but no per-picture
/// scratch — `encode_idr` is the one entry point.
pub struct Encoder {
    cfg: EncoderConfig,
}

impl Encoder {
    /// Read-only access to the encoder config (used by the round-30
    /// CABAC entry points in [`crate::encoder::cabac_path`]).
    pub fn config(&self) -> &EncoderConfig {
        &self.cfg
    }
}

impl Encoder {
    pub fn new(cfg: EncoderConfig) -> Self {
        assert!(
            cfg.width % 16 == 0 && cfg.height % 16 == 0,
            "encoder requires 16-pixel-aligned dimensions",
        );
        assert!((0..=51).contains(&cfg.qp));
        assert!(
            matches!(cfg.chroma_format_idc, 1..=3),
            "encoder only supports 4:2:0 (chroma_format_idc=1), 4:2:2 \
             (chroma_format_idc=2), and 4:4:4 (chroma_format_idc=3); got {}",
            cfg.chroma_format_idc,
        );
        // Round-27: 4:2:2 currently requires the High 4:2:2 profile (122).
        // Round-28: 4:4:4 currently requires the High 4:4:4 Predictive
        // profile (244). The 4:2:0-only P/B paths still pin profile_idc
        // to 66/77; the chroma-extended IDR paths need a profile in the
        // §7.3.2.1.1 chroma-extended group.
        if cfg.chroma_format_idc == 2 {
            assert_eq!(
                cfg.profile_idc, 122,
                "4:2:2 (chroma_format_idc=2) requires profile_idc=122 (High 4:2:2)",
            );
        }
        if cfg.chroma_format_idc == 3 {
            assert_eq!(
                cfg.profile_idc, 244,
                "4:4:4 (chroma_format_idc=3) requires profile_idc=244 (High 4:4:4 Predictive)",
            );
        }
        Self { cfg }
    }

    /// Encode the SPS + PPS + IDR access unit for one frame as an
    /// Annex B byte stream. Also returns the per-plane reconstruction
    /// the decoder is expected to produce — useful for round-trip
    /// validation in tests.
    pub fn encode_idr(&self, frame: &YuvFrame<'_>) -> EncodedIdr {
        assert_eq!(frame.width, self.cfg.width);
        assert_eq!(frame.height, self.cfg.height);
        let width_mbs = self.cfg.width / 16;
        let height_mbs = self.cfg.height / 16;
        // §6.2 / Table 6-1 — chroma plane dimensions per chroma_format_idc.
        // 4:2:0 → (W/2, H/2); 4:2:2 → (W/2, H); 4:4:4 → (W, H).
        let chroma_width = match self.cfg.chroma_format_idc {
            3 => self.cfg.width as usize,
            _ => (self.cfg.width / 2) as usize,
        };
        let chroma_height = match self.cfg.chroma_format_idc {
            2 => self.cfg.height as usize,
            3 => self.cfg.height as usize,
            _ => (self.cfg.height / 2) as usize,
        };
        // Sanity-check the input frame's chroma plane sizes match what
        // we expect for the configured chroma_format_idc. Mis-sized
        // planes would silently produce out-of-bounds reads in the
        // encoder helpers.
        debug_assert_eq!(
            frame.u.len(),
            chroma_width * chroma_height,
            "Cb plane length {} does not match {}x{} for chroma_format_idc={}",
            frame.u.len(),
            chroma_width,
            chroma_height,
            self.cfg.chroma_format_idc,
        );
        debug_assert_eq!(frame.v.len(), chroma_width * chroma_height);

        // SPS / PPS bytes.
        let sps_cfg = BaselineSpsConfig {
            seq_parameter_set_id: 0,
            level_idc: self.cfg.level_idc,
            width_in_mbs: width_mbs,
            height_in_mbs: height_mbs,
            log2_max_frame_num_minus4: 4,
            log2_max_poc_lsb_minus4: 4,
            max_num_ref_frames: self.cfg.max_num_ref_frames,
            profile_idc: self.cfg.profile_idc,
            chroma_format_idc: self.cfg.chroma_format_idc,
        };
        let pps_cfg = BaselinePpsConfig {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
            pic_init_qp_minus26: self.cfg.qp - 26,
            chroma_qp_index_offset: 0,
            weighted_pred_flag: false,
            // Round-26 — when the encoder is configured for explicit
            // weighted bipred, the PPS must signal idc=1 so subsequent
            // B-slice headers may carry pred_weight_table().
            weighted_bipred_idc: if self.cfg.explicit_weighted_bipred {
                1
            } else {
                0
            },
            entropy_coding_mode_flag: false,
        };
        let sps_rbsp = build_baseline_sps_rbsp(&sps_cfg);
        let pps_rbsp = build_baseline_pps_rbsp(&pps_cfg);

        let mut stream: Vec<u8> = Vec::new();
        stream.extend_from_slice(&build_nal_unit(3, NalUnitType::Sps, &sps_rbsp));
        stream.extend_from_slice(&build_nal_unit(3, NalUnitType::Pps, &pps_rbsp));

        // Slice header + slice data → IDR NAL.
        let mut sw = BitWriter::new();
        write_idr_i_slice_header(
            &mut sw,
            &IdrSliceHeaderConfig {
                first_mb_in_slice: 0,
                slice_type_raw: 7,
                pic_parameter_set_id: 0,
                frame_num: 0,
                frame_num_bits: sps_cfg.log2_max_frame_num_minus4 + 4,
                idr_pic_id: 0,
                pic_order_cnt_lsb: 0,
                poc_lsb_bits: sps_cfg.log2_max_poc_lsb_minus4 + 4,
                slice_qp_delta: 0, // total QP = pic_init_qp_minus26 + 26
                // §7.4.3 / round 14 — enable in-loop deblocking. We
                // mirror the decoder's §8.7 pass on the local recon so
                // EncodedIdr.recon_* matches what the decoder produces.
                //
                // Round-28 (4:4:4) intentionally disables deblocking:
                // §8.7 chroma deblocking for ChromaArrayType==3 uses
                // the LUMA filter ladder applied to each chroma plane
                // (per-plane bS derivation), which is a separate code
                // path the round-28 decoder doesn't yet exercise.
                // Setting `disable_deblocking_filter_idc=1` is per
                // §7.4.3 spec and yields a still-conformant bitstream.
                disable_deblocking_filter_idc: if self.cfg.chroma_format_idc == 3 {
                    1
                } else {
                    0
                },
                slice_alpha_c0_offset_div2: 0,
                slice_beta_offset_div2: 0,
            },
        );

        // Reconstruction state, sized to the picture.
        let mut recon_y = vec![0u8; (self.cfg.width * self.cfg.height) as usize];
        let mut recon_u = vec![0u8; chroma_width * chroma_height];
        let mut recon_v = vec![0u8; chroma_width * chroma_height];

        let qp_y = self.cfg.qp;
        // §8.5.8 — chroma-QP derivation, BD-aware. `qp_bd_offset_c =
        // 6 * bit_depth_chroma_minus8` extends the eq. 8-311 qPI clamp
        // from `0..=51` to `−QpBdOffsetC..=51`. Today the SPS pins
        // bit_depth_chroma_minus8=0 so this matches the legacy 8-bit
        // path; the BD-aware call lets a future High10/422/444 round
        // wire a non-zero BD without re-touching the encoder math.
        let qp_bd_offset_c = qp_bd_offset(self.cfg.bit_depth_chroma_minus8);
        let qp_c =
            qp_y_to_qp_c_with_bd_offset(qp_y, pps_cfg.chroma_qp_index_offset, qp_bd_offset_c);

        // §9.2.1.1 — neighbour grid for CAVLC nC derivation. Encoder
        // mirrors what the decoder will reconstruct as it walks the MBs:
        // mark each MB's slot `is_available = true` after encode and
        // store the per-4x4 luma-AC TotalCoeff. The decoder uses the
        // same procedure to pick CAVLC tables, so we must match it
        // bit-exactly to avoid mis-parsing.
        let mut nc_grid = CavlcNcGrid::new(width_mbs, height_mbs);
        // Encoder-side intra-pred-mode grid mirroring what the decoder
        // will populate as it reconstructs each MB. Used to derive
        // `predIntra4x4PredMode` per §8.3.1.1 step 4.
        let mut intra_grid = IntraGrid::new(width_mbs as usize, height_mbs as usize);

        // §8.7 deblock info, one entry per MB (raster order). Filled by
        // the per-MB encode functions; consumed after the slice loop by
        // the picture-level deblock pass.
        let mut mb_deblock_infos: Vec<MbDeblockInfo> =
            vec![MbDeblockInfo::default(); (width_mbs * height_mbs) as usize];

        // Iterate MBs in raster order.
        for mb_y in 0..height_mbs {
            for mb_x in 0..width_mbs {
                let dbl = self.encode_mb(
                    frame,
                    mb_x as usize,
                    mb_y as usize,
                    qp_y,
                    qp_c,
                    chroma_width,
                    chroma_height,
                    &mut recon_y,
                    &mut recon_u,
                    &mut recon_v,
                    &mut sw,
                    &mut nc_grid,
                    &mut intra_grid,
                );
                let mb_addr = (mb_y * width_mbs + mb_x) as usize;
                mb_deblock_infos[mb_addr] = dbl;
            }
        }

        // §7.3.2.8 / §7.3.2.9 — slice_data trailing bits.
        sw.rbsp_trailing_bits();
        let slice_rbsp = sw.into_bytes();
        stream.extend_from_slice(&build_nal_unit(3, NalUnitType::SliceIdr, &slice_rbsp));

        // §8.7 — picture-level in-loop deblocking. Runs *after* every MB
        // has been reconstructed, both because the spec defines the
        // filter per picture and because the encoder's intra-prediction
        // step (which reads `recon_y` for neighbours) must see the
        // pre-filter samples — same as the decoder.
        //
        // Round-28 (4:4:4) skips the deblock pass: the slice header
        // sets `disable_deblocking_filter_idc=1` so the decoder doesn't
        // run §8.7, and the encoder must mirror that on its local recon
        // to keep `EncodedIdr.recon_*` bit-equivalent with what the
        // decoder produces.
        if self.cfg.chroma_format_idc != 3 {
            crate::encoder::deblock::deblock_recon_with_chroma_array_type(
                self.cfg.width,
                self.cfg.height,
                chroma_width as u32,
                chroma_height as u32,
                &mut recon_y,
                &mut recon_u,
                &mut recon_v,
                &mb_deblock_infos,
                pps_cfg.chroma_qp_index_offset,
                width_mbs,
                height_mbs,
                self.cfg.chroma_format_idc,
            );
        }

        // IDR is all-intra: every partition is `is_intra = true`, so
        // colocated lookups return colZeroFlag = 0 (the §8.4.1.2.2 step 7
        // "intra colocated" branch).
        let n_parts = (width_mbs as usize) * (height_mbs as usize) * 4;
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
            recon_width: self.cfg.width,
            recon_height: self.cfg.height,
            partition_mvs,
        }
    }

    /// Round-15 entry point: **Lagrangian RDO** mb_type dispatch.
    ///
    /// Strategy:
    ///   1) Snapshot the recon planes (only the MB-sized window),
    ///      `BitWriter`, `CavlcNcGrid` slot, and `IntraGrid` slot.
    ///   2) Run the I_16x16 path, measuring `J = D + λ·R` for the luma
    ///      portion (chroma is identical for both paths so it doesn't
    ///      affect the comparison and is excluded from J).
    ///   3) Snapshot its output state; restore the original state.
    ///   4) Run the I_NxN path, measuring its luma J the same way.
    ///   5) Pick the lower J; if I_16x16 won, restore its snapshot.
    ///
    /// Compared to round-4 SAD: the new metric counts actual quantisation
    /// noise (D = SSD on recon) and actual entropy-coded bits (R = trial
    /// CAVLC encode), so the comparison is calibrated. λ comes from
    /// [`rdo::lambda_ssd`].
    #[allow(clippy::too_many_arguments)]
    fn encode_mb(
        &self,
        frame: &YuvFrame<'_>,
        mb_x: usize,
        mb_y: usize,
        qp_y: i32,
        qp_c: i32,
        chroma_width: usize,
        chroma_height: usize,
        recon_y: &mut [u8],
        recon_u: &mut [u8],
        recon_v: &mut [u8],
        sw: &mut BitWriter,
        nc_grid: &mut CavlcNcGrid,
        intra_grid: &mut IntraGrid,
    ) -> MbDeblockInfo {
        let width = self.cfg.width as usize;
        let width_mbs = (self.cfg.width / 16) as usize;
        let mb_addr = (mb_y as u32) * self.cfg.width / 16 + (mb_x as u32);

        // Round-28: 4:4:4 path is Intra_16x16-only (the I_NxN chroma
        // emit and the §8.3.4.1-for-ChromaArrayType=3 chroma-as-luma
        // 4x4 / 8x8 plumbing for the I_NxN side are out of scope this
        // round). Skip the RDO between I_16x16 and I_NxN and run the
        // I_16x16 path directly.
        if self.cfg.chroma_format_idc == 3 {
            let trial = self.encode_mb_intra16x16(
                frame,
                mb_x,
                mb_y,
                qp_y,
                qp_c,
                chroma_width,
                chroma_height,
                recon_y,
                recon_u,
                recon_v,
                sw,
                nc_grid,
                intra_grid,
            );
            return trial.deblock;
        }

        // -- Snapshot pre-state (so we can roll back the loser). --
        let snap = MbStateSnapshot::capture(
            recon_y,
            recon_u,
            recon_v,
            width,
            chroma_width,
            mb_x,
            mb_y,
            sw,
            nc_grid,
            intra_grid,
            mb_addr as usize,
            width_mbs,
        );

        // ----- Trial 1: I_16x16 path. -----
        let trial_a = self.encode_mb_intra16x16(
            frame,
            mb_x,
            mb_y,
            qp_y,
            qp_c,
            chroma_width,
            chroma_height,
            recon_y,
            recon_u,
            recon_v,
            sw,
            nc_grid,
            intra_grid,
        );
        let snap_a = MbStateSnapshot::capture(
            recon_y,
            recon_u,
            recon_v,
            width,
            chroma_width,
            mb_x,
            mb_y,
            sw,
            nc_grid,
            intra_grid,
            mb_addr as usize,
            width_mbs,
        );

        // -- Restore original state for the second trial. --
        snap.restore(
            recon_y,
            recon_u,
            recon_v,
            width,
            chroma_width,
            mb_x,
            mb_y,
            sw,
            nc_grid,
            intra_grid,
            mb_addr as usize,
        );

        // ----- Trial 2: I_NxN path. -----
        let trial_b = self.encode_mb_intra4x4(
            frame,
            mb_x,
            mb_y,
            qp_y,
            qp_c,
            chroma_width,
            chroma_height,
            recon_y,
            recon_u,
            recon_v,
            sw,
            nc_grid,
            intra_grid,
        );

        // ----- Pick winner. -----
        // Each path returns its own luma `J = D + λR` inside `MbTrial`.
        // Lower J wins. On tie, prefer I_16x16 (smaller MB syntax).
        if trial_a.luma_cost <= trial_b.luma_cost {
            // Roll back the I_NxN trial; install the I_16x16 results.
            snap.restore(
                recon_y,
                recon_u,
                recon_v,
                width,
                chroma_width,
                mb_x,
                mb_y,
                sw,
                nc_grid,
                intra_grid,
                mb_addr as usize,
            );
            snap_a.install(
                recon_y,
                recon_u,
                recon_v,
                width,
                chroma_width,
                mb_x,
                mb_y,
                sw,
                nc_grid,
                intra_grid,
                mb_addr as usize,
            );
            trial_a.deblock
        } else {
            trial_b.deblock
        }
    }

    /// Intra_16x16 encoder path with round-15 Lagrangian RDO mode
    /// decision. Returns `MbTrial` carrying the luma `J = D + λ·R` cost
    /// alongside the deblock info.
    #[allow(clippy::too_many_arguments)]
    fn encode_mb_intra16x16(
        &self,
        frame: &YuvFrame<'_>,
        mb_x: usize,
        mb_y: usize,
        qp_y: i32,
        qp_c: i32,
        chroma_width: usize,
        chroma_height: usize,
        recon_y: &mut [u8],
        recon_u: &mut [u8],
        recon_v: &mut [u8],
        sw: &mut BitWriter,
        nc_grid: &mut CavlcNcGrid,
        intra_grid: &mut IntraGrid,
    ) -> MbTrial {
        let width = self.cfg.width as usize;
        let height = self.cfg.height as usize;
        let lambda = lambda_ssd(qp_y, true);

        // ------- Luma: pick mode by full RDO -------
        // §8.3.3 — for each of 4 candidates compute D = SSD(source, recon)
        // and R = trial CAVLC bits, then pick min(D + λR). Modes whose
        // required neighbours are missing are skipped (the decoder would
        // reject them if we tried to signal them).
        let mut best_luma_mode = I16x16Mode::Dc;
        let mut best_luma_pred = [0i32; 256];
        let mut best_luma_cost = u64::MAX;
        for &mode in &[
            I16x16Mode::Vertical,
            I16x16Mode::Horizontal,
            I16x16Mode::Dc,
            I16x16Mode::Plane,
        ] {
            let Some(pred) = predict_16x16(mode, recon_y, width, height, mb_x, mb_y) else {
                continue;
            };
            let cost = trial_intra16x16_cost(frame, &pred, mb_x, mb_y, qp_y, width, lambda);
            if cost < best_luma_cost {
                best_luma_cost = cost;
                best_luma_mode = mode;
                best_luma_pred = pred;
            }
        }

        // Sample residual (16x16 i32) against the chosen predictor.
        let mut residual = [0i32; 256];
        for j in 0..16usize {
            for i in 0..16usize {
                let s = frame.y[(mb_y * 16 + j) * width + (mb_x * 16 + i)] as i32;
                residual[j * 16 + i] = s - best_luma_pred[j * 16 + i];
            }
        }

        // For each 4x4 sub-block: forward 4x4. Then collect DCs and
        // run the Hadamard.
        let mut coeffs_4x4 = [[0i32; 16]; 16];
        for (blk, slot) in coeffs_4x4.iter_mut().enumerate() {
            let bx = blk % 4;
            let by = blk / 4;
            let mut block = [0i32; 16];
            for j in 0..4 {
                for i in 0..4 {
                    block[j * 4 + i] = residual[(by * 4 + j) * 16 + bx * 4 + i];
                }
            }
            *slot = forward_core_4x4(&block);
        }
        // Pull out the DCs into the row-major DC matrix.
        let mut dc_coeffs = [0i32; 16];
        for (blk, coeffs) in coeffs_4x4.iter().enumerate() {
            let bx = blk % 4;
            let by = blk / 4;
            dc_coeffs[by * 4 + bx] = coeffs[0];
        }
        // Forward Hadamard + quantize.
        let dc_hadamard = forward_hadamard_4x4(&dc_coeffs);
        let dc_levels = quantize_luma_dc(&dc_hadamard, qp_y, true);

        // §8.5.10 — luma AC residual. Quantize each 4x4 block's AC
        // (positions 1..15 of the row-major coefficient block; position
        // 0 is the DC and is handled separately by the Hadamard above).
        // Pack into AC-only scan order (15 entries each) ready for
        // CAVLC emit.
        //
        // Ordering note: §6.4.3 raster-Z order (`LUMA_4X4_BLK`) is what
        // the decoder consumes when `cbp_luma == 15`. The local
        // `coeffs_4x4[blk]` array above is keyed by simple row-major
        // (`blk = by*4 + bx`); we re-index here.
        let mut luma_ac_levels = [[0i32; 16]; 16];
        let mut luma_ac_quant_raster = [[0i32; 16]; 16]; // for inverse path
        let mut any_luma_ac_nz = false;
        for blkz in 0..16usize {
            let (bx, by) = LUMA_4X4_BLK[blkz];
            let raster = by * 4 + bx;
            let z = quantize_4x4_ac(&coeffs_4x4[raster], qp_y, true);
            luma_ac_quant_raster[raster] = z;
            // AC scan layout for the bitstream (positions 0..=14).
            let scan_ac = zigzag_scan_4x4_ac(&z);
            luma_ac_levels[blkz] = scan_ac;
            if scan_ac.iter().any(|&v| v != 0) {
                any_luma_ac_nz = true;
            }
        }
        // §7.4.5.1 — for Intra_16x16 cbp_luma is 0 or 15 only.
        // Round-28: for 4:4:4 we may need to promote cbp_luma to 15
        // post-hoc if either chroma plane has nonzero AC (so the
        // §7.3.5.3 chroma plane AC blocks reach the decoder under
        // the cbp_luma==15 gate). Done below after chroma encode.
        #[allow(unused_mut)]
        let mut cbp_luma: u8 = if any_luma_ac_nz { 15 } else { 0 };

        // §8.5.10 reconstruction: inverse Hadamard on the DC levels +
        // per-block inverse 4x4 with c[0,0] overwritten by the matching
        // pre-scaled DC entry. When `cbp_luma == 0` the AC is zeroed —
        // matching what the decoder will see (no AC blocks in the
        // bitstream → empty residual_luma → coeffs[1..] = 0).
        let inv_dc = inverse_hadamard_luma_dc_16x16(&dc_levels, qp_y, &FLAT_4X4_16, 8)
            .expect("luma DC inverse");
        #[allow(clippy::needless_range_loop)] // raster row-major over 16 4x4 blocks
        for blk in 0..16usize {
            let bx = blk % 4;
            let by = blk / 4;
            // Build the inverse coefficient block: AC from the
            // already-quantized levels (only when cbp_luma == 15 will
            // they actually reach the decoder — otherwise zero them
            // here so the encoder recon matches what the decoder will
            // produce).
            let mut coeffs_in = if cbp_luma == 15 {
                // Re-build the pre-AC-scan coeff matrix the decoder
                // produces by `inverse_scan_4x4_zigzag_ac` from the
                // AC-only layout. Identical to taking the original
                // `luma_ac_quant_raster[blk]` (which already has slot 0
                // zeroed by `quantize_4x4_ac`).
                luma_ac_quant_raster[blk]
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
                    let p = best_luma_pred[(by * 4 + j) * 16 + bx * 4 + i];
                    let raw = p + r[j * 4 + i];
                    recon_y[py * width + px] = raw.clamp(0, 255) as u8;
                }
            }
        }

        // ------- Chroma: dispatch to 4:2:0 / 4:2:2 / 4:4:4 helper -------
        // Round 27/28: the 4:2:0 / 4:2:2 helper picks the best §8.3.4
        // mode by SAD, encodes DC + AC residuals, reconstructs the
        // sample residual, and returns a `ChromaBlock`. For 4:4:4 the
        // chroma plane is "coded like luma" (§7.3.5.3 /
        // §8.3.4.1), so we skip the chroma-mode-decision entirely:
        // chroma reuses the luma Intra_16x16 mode, and each plane
        // runs the luma-style 16x16 DC + 16 AC pipeline via
        // `encode_intra16x16_chroma_plane_luma_like`.
        let chroma_array_type = self.cfg.chroma_format_idc;
        let mut chroma_block_for_mask: Option<ChromaBlock> = None;
        // Pre-declare 4:4:4 outputs (only populated on the 4:4:4 path).
        let mut cb_dc_444 = [0i32; 16];
        let mut cr_dc_444 = [0i32; 16];
        let mut cb_ac_scan_444 = [[0i32; 16]; 16];
        let mut cr_ac_scan_444 = [[0i32; 16]; 16];
        let mut cb_ac_quant_444 = [[0i32; 16]; 16];
        let mut cr_ac_quant_444 = [[0i32; 16]; 16];
        let (cbp_chroma, best_chroma_mode_u8): (u8, u8) = if chroma_array_type == 3 {
            // §8.3.4.1 — chroma reuses the luma mode for 4:4:4.
            let (cb_dc, cb_ac_scan, cb_ac_quant, cb_nz) = encode_intra16x16_chroma_plane_luma_like(
                best_luma_mode,
                frame.u,
                recon_u,
                chroma_width,
                chroma_height,
                mb_x,
                mb_y,
                qp_c,
            );
            let (cr_dc, cr_ac_scan, cr_ac_quant, cr_nz) = encode_intra16x16_chroma_plane_luma_like(
                best_luma_mode,
                frame.v,
                recon_v,
                chroma_width,
                chroma_height,
                mb_x,
                mb_y,
                qp_c,
            );
            cb_dc_444 = cb_dc;
            cr_dc_444 = cr_dc;
            cb_ac_scan_444 = cb_ac_scan;
            cr_ac_scan_444 = cr_ac_scan;
            cb_ac_quant_444 = cb_ac_quant;
            cr_ac_quant_444 = cr_ac_quant;
            // 4:4:4: cbp_chroma is held at 0 in mb_type (Table 7-11
            // entry chosen in the 0-chroma family). The chroma plane AC
            // emit is gated by `cbp_luma == 15` instead, per §7.3.5.3.
            //
            // When either chroma plane has nonzero AC, we promote
            // `cbp_luma` to 15 here so the bitstream gate fires and the
            // decoder reads the chroma plane AC blocks. The luma AC
            // payload is identically zero in that case, so 16 short
            // coeff_token=0 blocks add ~16 bits — a small and
            // predictable cost.
            if cb_nz || cr_nz {
                cbp_luma = 15;
            }
            (0u8, 0u8)
        } else {
            let chroma_block = encode_chroma_block(
                chroma_array_type,
                frame.u,
                frame.v,
                recon_u,
                recon_v,
                chroma_width,
                chroma_height,
                mb_x,
                mb_y,
                qp_c,
            );
            let cm = chroma_block.pred_mode.as_u8();
            let ccbp = chroma_block.cbp_chroma;
            // Apply 4:2:0 / 4:2:2 chroma reconstruction.
            install_chroma_recon(
                chroma_array_type,
                chroma_width,
                mb_x,
                mb_y,
                &chroma_block,
                recon_u,
                recon_v,
            );
            chroma_block_for_mask = Some(chroma_block);
            (ccbp, cm)
        };
        let n_chroma_blk = ChromaBlock::cb_block_count(chroma_array_type);
        // Re-export the per-block AC arrays into local arrays of the
        // shape the legacy `I16x16McbConfig` expects (4 blocks). For
        // 4:2:2 the writer below uses `write_intra16x16_mb_chroma` with
        // `ChromaWriteKind::Yuv422` so these arrays are unused.
        let mut u_dc_levels = [0i32; 4];
        let mut v_dc_levels = [0i32; 4];
        let mut u_ac_levels = [[0i32; 16]; 4];
        let mut v_ac_levels = [[0i32; 16]; 4];
        if chroma_array_type == 1 {
            let cb = chroma_block_for_mask.as_ref().unwrap();
            u_dc_levels.copy_from_slice(&cb.dc_cb[..4]);
            v_dc_levels.copy_from_slice(&cb.dc_cr[..4]);
            u_ac_levels.copy_from_slice(&cb.ac_cb[..4]);
            v_ac_levels.copy_from_slice(&cb.ac_cr[..4]);
        }
        let _ = chroma_height;

        // ------- Build CAVLC neighbour-grid view + per-block nC ------
        // §9.2.1.1: nC for each AC residual block is derived from the
        // neighbour A (left) and B (above) blocks. The decoder will use
        // its own CavlcNcGrid for this; the encoder must match.
        //
        // Strategy: mark this MB available *before* emit and
        // progressively fill `luma_total_coeff[blk]` with the
        // TotalCoeff of each AC block as we enumerate them. Internal
        // neighbours then read the correct partial state via
        // `derive_nc_luma`'s `NeighbourSource::InternalBlock` →
        // `nc_nn_luma` → `info.luma_total_coeff[blk]` chain. External
        // neighbours read from previously-encoded MBs (already filled
        // by an earlier iteration of this loop).
        let width_mbs = self.cfg.width / 16;
        let mb_addr = (mb_y as u32) * width_mbs + (mb_x as u32);
        // NOTE: order of operations matters. We need to set
        // `is_available = true` and `is_intra = true` for the current
        // MB *before* deriving any nC values that might reference its
        // own internal blocks.
        {
            let cur = &mut nc_grid.mbs[mb_addr as usize];
            cur.is_available = true;
            cur.is_intra = true;
            cur.is_skip = false;
            cur.is_i_pcm = false;
            cur.luma_total_coeff = [0u8; 16]; // start fresh
        }

        // §9.2.1.1 NOTE 1: nC for Intra16x16DCLevel uses *AC* TotalCoeff
        // of neighbour luma4x4 block 0 (the DC's "blk_idx" is fixed to
        // 0 by step 1). The neighbour A/B for block 0 are the
        // corresponding blocks of the left / above MBs. `derive_nc_luma`
        // with `LumaNcKind::Intra16x16Dc` does this lookup.
        let nc_dc = derive_nc_luma(nc_grid, mb_addr, 0, LumaNcKind::Intra16x16Dc, true, false);

        // Per-AC-block nC, in §6.4.3 raster-Z order. We compute every
        // entry up-front (using the freshly-zeroed own_luma_totals on
        // this MB) then progressively update the grid as we encode each
        // block, recomputing the next block's nC from the updated state.
        let mut luma_ac_nc = [0i32; 16];
        let mut own_totals = [0u8; 16];
        for blkz in 0..16u8 {
            // Update grid view of own MB before computing this block's
            // nC, so internal neighbours read the right TotalCoeff.
            nc_grid.mbs[mb_addr as usize].luma_total_coeff = own_totals;
            let nc = derive_nc_luma(nc_grid, mb_addr, blkz, LumaNcKind::Ac, true, false);
            luma_ac_nc[blkz as usize] = nc;
            // Update own_totals with this block's TotalCoeff so the
            // next block's nC sees it.
            if cbp_luma == 15 {
                let tc = luma_ac_levels[blkz as usize]
                    .iter()
                    .filter(|&&v| v != 0)
                    .count() as u8;
                own_totals[blkz as usize] = tc;
            }
        }
        // Final write of TotalCoeff into the grid for this MB so future
        // MBs see the correct neighbour state.
        nc_grid.mbs[mb_addr as usize].luma_total_coeff = own_totals;

        // ------- Emit MB syntax -------
        let nc_dc_ctx = CoeffTokenContext::Numeric(nc_dc);
        if chroma_array_type == 1 {
            let chroma_block = chroma_block_for_mask.as_ref().unwrap();
            let mb_cfg = I16x16McbConfig {
                pred_mode: best_luma_mode.as_u8(),
                intra_chroma_pred_mode: best_chroma_mode_u8,
                cbp_luma,
                cbp_chroma,
                mb_qp_delta: 0,
                luma_dc_levels_raster: dc_levels,
                luma_ac_levels,
                luma_ac_nc,
                chroma_dc_cb: u_dc_levels,
                chroma_dc_cr: v_dc_levels,
                chroma_ac_cb: u_ac_levels,
                chroma_ac_cr: v_ac_levels,
            };
            let _ = chroma_block; // silence "unused" if helper changes.
            write_intra16x16_mb(sw, &mb_cfg, nc_dc_ctx).expect("write Intra16x16 mb");
        } else if chroma_array_type == 2 {
            // Round-27: 4:2:2 path. Pass per-plane 8-entry DC + 8-block
            // AC arrays via `ChromaWriteKind::Yuv422`.
            let chroma_block = chroma_block_for_mask.as_ref().unwrap();
            crate::encoder::macroblock::write_intra16x16_mb_chroma(
                sw,
                best_luma_mode.as_u8(),
                best_chroma_mode_u8,
                cbp_luma,
                cbp_chroma,
                0,
                &dc_levels,
                &luma_ac_levels,
                &luma_ac_nc,
                crate::encoder::macroblock::ChromaWriteKind::Yuv422 {
                    chroma_dc_cb: &chroma_block.dc_cb,
                    chroma_dc_cr: &chroma_block.dc_cr,
                    chroma_ac_cb: &chroma_block.ac_cb,
                    chroma_ac_cr: &chroma_block.ac_cr,
                },
                nc_dc_ctx,
            )
            .expect("write Intra16x16 mb (4:2:2)");
        } else {
            // Round-28: 4:4:4 path. Each chroma plane is coded "like
            // luma" — its own 16x16 DC Hadamard block + 16 4x4 AC
            // blocks, gated by luma cbp==15 (per §7.3.5.3 / decoder
            // `residual_cb_16x16_dc` / `residual_cb_luma_like` etc).
            //
            // §9.2.1.1 — nC for the per-plane DC/AC blocks should use
            // the per-plane neighbour `cb_luma_total_coeff[]` /
            // `cr_luma_total_coeff[]`. This encoder approximates with
            // nC=0 across the board; the decoder is then bit-exact
            // because both sides agree on the choice (the encoder
            // doesn't update those fields in the neighbour grid, and
            // the decoder's `nc_nn_plane_luma_like` lookups will see
            // the per-plane totals as default 0). Empirically this
            // matches what ffmpeg produces from the same parsed bits.
            let cb_ac_nc = [0i32; 16];
            let cr_ac_nc = [0i32; 16];
            crate::encoder::macroblock::write_intra16x16_mb_chroma(
                sw,
                best_luma_mode.as_u8(),
                /* intra_chroma_pred_mode: ignored for 4:4:4 */ 0,
                cbp_luma,
                cbp_chroma,
                0,
                &dc_levels,
                &luma_ac_levels,
                &luma_ac_nc,
                crate::encoder::macroblock::ChromaWriteKind::Yuv444 {
                    cb_dc_raster: &cb_dc_444,
                    cr_dc_raster: &cr_dc_444,
                    cb_ac_levels: &cb_ac_scan_444,
                    cr_ac_levels: &cr_ac_scan_444,
                    cb_ac_nc: &cb_ac_nc,
                    cr_ac_nc: &cr_ac_nc,
                    cb_dc_nc: 0,
                    cr_dc_nc: 0,
                    cbp_luma_for_ac_gate: cbp_luma,
                },
                nc_dc_ctx,
            )
            .expect("write Intra16x16 mb (4:4:4)");
            // Suppress warnings for unused 4:4:4 reconstruction state
            // when the helper internally needs them.
            let _ = (cb_ac_quant_444, cr_ac_quant_444);
        }

        // Mark this MB as available + not-i_nxn for subsequent MBs'
        // §8.3.1.1 step 3 lookup. Intra_16x16 neighbours fall into
        // the "DC fallback" branch (intraMxMPredModeN = 2).
        let slot = intra_grid.slot_mut(mb_x, mb_y);
        slot.available = true;
        slot.is_i_nxn = false;

        // §8.7.2.1 — per-4x4 nonzero mask for the deblock walker.
        // Intra_16x16: when cbp_luma == 15 every 4x4 block transmits its
        // (DC + AC) coefficients and the DC plane guarantees the block
        // contains a non-zero coefficient. The deblock walker treats any
        // nonzero block (DC or AC) as "either has nonzero coeffs", which
        // promotes bS from 0 → 2 on internal edges. So we set the bit
        // for every block when cbp_luma == 15. When cbp_luma == 0 the
        // DC Hadamard levels may still be non-zero, but the spec's per-
        // 4x4 nonzero check (§8.7.2.1 third bullet) reads from the
        // luma4x4 blocks — for Intra_16x16 the DC plane is a separate
        // block (Intra16x16DCLevel) that the decoder marks as nonzero on
        // own_luma_total set by the DC's TotalCoeff. We reflect this
        // closely by setting all 16 bits when cbp_luma == 15 (AC nonzero
        // anywhere) — DC-only Intra_16x16 (cbp_luma == 0) leaves the
        // mask at 0, matching the spec's per-4x4 view: AC of those
        // blocks is identically zero so the bS=2 rule fires only on
        // edges where the OTHER side has AC.
        let luma_nonzero_4x4 = if cbp_luma == 15 {
            // Per-block nonzero: bit set iff the AC scan-order array
            // has at least one non-zero entry.
            let mut nz = [false; 16];
            for (blkz, scan_ac) in luma_ac_levels.iter().enumerate() {
                nz[blkz] = scan_ac.iter().any(|&v| v != 0);
            }
            luma_nz_mask_from_blocks(&nz)
        } else {
            0
        };
        // Chroma per-block AC nonzero. For 4:2:0: 4 blocks per plane,
        // packed as bits 0..=3 (Cb) and 8..=11 (Cr). For 4:2:2: 8 blocks
        // per plane, packed bits 0..=7 (Cb) and 8..=15 (Cr) per
        // `compute_chroma_nonzero_mask` in the decoder's reconstruct.rs.
        // Round-28 4:4:4: chroma deblocking uses the LUMA filter ladder,
        // which is gated entirely on the encoder side by setting
        // `disable_deblocking_filter_idc=1` in the slice header. The
        // chroma_nonzero_4x4 mask is left at 0 — it isn't consulted on
        // this path.
        let chroma_nz_mask = chroma_block_for_mask
            .as_ref()
            .map(|cb| chroma_nz_mask_from_chroma_block(cb, n_chroma_blk))
            .unwrap_or(0);
        MbTrial {
            luma_cost: best_luma_cost,
            deblock: MbDeblockInfo {
                is_intra: true,
                qp_y,
                luma_nonzero_4x4,
                chroma_nonzero_4x4: chroma_nz_mask,
                ..Default::default()
            },
        }
    }

    /// I_NxN (Intra_4x4) path: per-block 9-mode prediction, full luma
    /// 4x4 residual, chroma residual identical to I_16x16. Round-15: per-
    /// block Lagrangian RDO replaces SAD-on-predictor mode decision.
    #[allow(clippy::too_many_arguments)]
    fn encode_mb_intra4x4(
        &self,
        frame: &YuvFrame<'_>,
        mb_x: usize,
        mb_y: usize,
        qp_y: i32,
        qp_c: i32,
        chroma_width: usize,
        chroma_height: usize,
        recon_y: &mut [u8],
        recon_u: &mut [u8],
        recon_v: &mut [u8],
        sw: &mut BitWriter,
        nc_grid: &mut CavlcNcGrid,
        intra_grid: &mut IntraGrid,
    ) -> MbTrial {
        let width = self.cfg.width as usize;
        let width_mbs = (self.cfg.width / 16) as usize;
        let lambda = lambda_ssd(qp_y, true);

        // Per-block: pick mode by full Lagrangian RDO (round-15).
        // Build the intra_grid slot incrementally so subsequent blocks
        // see this MB's chosen intra_4x4_pred_modes when computing
        // their predIntra4x4PredMode (§8.3.1.1 step 4).
        //
        // We MUST pre-mark the slot as I_NxN+available so that the
        // `predicted_4x4_mode` lookup for blocks 5..15 finds the
        // current MB's earlier-block modes.
        {
            let s = intra_grid.slot_mut(mb_x, mb_y);
            s.available = true;
            s.is_i_nxn = true;
            s.intra_4x4_pred_modes = [0u8; 16];
        }

        // Per-block scratch.
        let mut chosen_mode = [0u8; 16]; // Intra4x4Mode index per block.
        let mut prev_flag = [false; 16];
        let mut rem = [0u8; 16];
        let mut luma_4x4_levels_scan = [[0i32; 16]; 16]; // scan-order per block
        let mut luma_4x4_quant_raster = [[0i32; 16]; 16]; // raster-order quantized levels (for inverse)
        let mut blk_has_nz = [false; 16];
        // Per-block cumulative luma cost (D + λR) — tracked for the
        // MB-level mb_type RDO comparison upstream in `encode_mb`. Each
        // block's contribution is the actual SSD on recon plus λ times
        // the per-block prev_flag/rem cost plus the trial CAVLC bit
        // count. We must approximate the per-block CAVLC bits because
        // the actual nC depends on neighbour state we won't fully
        // commit until after this loop; we use nC=0 as a stable lower-
        // bound estimate.
        //
        // Seed with the per-MB syntax overhead so the cost is comparable
        // to the I_16x16 trial:
        //   mb_type ue(0) = 1 bit
        //   intra_chroma_pred_mode ue(v) ≈ 3-bit average
        //   coded_block_pattern me(v) intra 4:2:0 = ~4-bit average
        //   mb_qp_delta se(0) = 1 bit (when any cbp non-zero)
        // → ~9 bits, roughly matching the I_16x16 path's 7 + 1 = 8 bits.
        //
        // Plus an empirical bias for "post-deblock D under-measurement":
        // mode decision happens before the picture-level §8.7 deblock
        // pass, so per-block D is measured on pre-filter recon. I_NxN at
        // the boundary of smooth regions tends to gain *more* deblock
        // smoothing than I_16x16; this bias keeps the smooth gradient
        // fixture's top-row / left-col MBs (where I_16x16 is restricted
        // to DC) from over-spending I_NxN there. The +16-bit value was
        // calibrated empirically against the round-15 fixture set.
        let mb_overhead_bits: u64 = 9 + 16;
        let mut mb_luma_cost: u64 = cost_combined(0, mb_overhead_bits, lambda);

        for blk in 0..16usize {
            let (bx, by) = I4X4_XY[blk];
            let avail = i4x4_avail(mb_x, mb_y, bx, by, width_mbs);
            let predicted = predicted_intra4x4_mode(intra_grid, mb_x, mb_y, blk, width_mbs);

            // Trial all 9 modes per RDO. For each: predict → forward →
            // quantize → inverse → add → SSD; trial CAVLC for rate.
            let mut best_mode = Intra4x4Mode::Dc;
            let mut best_cost = u64::MAX;
            let mut best_pred = [0i32; 16];
            let mut best_z_raster = [0i32; 16];
            let mut best_z_scan = [0i32; 16];
            let mut best_recon = [0i32; 16];
            for &mode in &Intra4x4Mode::ALL {
                let Some(pred) = predict_4x4(mode, recon_y, width, mb_x, mb_y, bx, by, avail)
                else {
                    continue;
                };
                let candidate = mode.as_u8();
                // ----- D component: build residual, transform, quant,
                // inverse, add → recon block. -----
                let mut block_res = [0i32; 16];
                for j in 0..4 {
                    for i in 0..4 {
                        let s = frame.y[(mb_y * 16 + by + j) * width + (mb_x * 16 + bx + i)] as i32;
                        block_res[j * 4 + i] = s - pred[j * 4 + i];
                    }
                }
                let w_coeffs = forward_core_4x4(&block_res);
                let z_raster = quantize_4x4(&w_coeffs, qp_y, true);
                let z_scan = zigzag_scan_4x4(&z_raster);
                let r =
                    inverse_transform_4x4(&z_raster, qp_y, &FLAT_4X4_16, 8).expect("inverse 4x4");
                let mut recon_block = [0i32; 16];
                for j in 0..4 {
                    for i in 0..4 {
                        recon_block[j * 4 + i] = pred[j * 4 + i] + r[j * 4 + i];
                    }
                }
                let d = ssd_4x4(frame.y, width, mb_x * 16 + bx, mb_y * 16 + by, &recon_block);

                // ----- R component: prev_flag/rem cost + trial CAVLC. -----
                // 1 bit for prev_flag; if mode != predicted, +3 bits rem.
                let mut bits: u64 = 1;
                if candidate != predicted {
                    bits += 3;
                }
                // Trial CAVLC encode for the residual block. nC=0 is a
                // safe stable estimate during mode trial — the actual
                // nC depends on neighbour TotalCoeff values that vary
                // with the chosen mode, but the impact on coeff_token
                // bit count is small (1-2 bits) and using a constant
                // makes mode comparisons consistent. Quadrant-cbp gating
                // is also approximated: for trial scoring assume the
                // block IS transmitted (over-estimate at low QP).
                let mut trial_w = BitWriter::new();
                if encode_residual_block_cavlc(
                    &mut trial_w,
                    CoeffTokenContext::Numeric(0),
                    16,
                    &z_scan,
                )
                .is_ok()
                {
                    bits += trial_w.bits_emitted() as u64;
                }
                let cost = cost_combined(d, bits, lambda);
                if cost < best_cost {
                    best_cost = cost;
                    best_mode = mode;
                    best_pred = pred;
                    best_z_raster = z_raster;
                    best_z_scan = z_scan;
                    best_recon = recon_block;
                }
            }
            let chosen = best_mode.as_u8();
            chosen_mode[blk] = chosen;
            mb_luma_cost = mb_luma_cost.saturating_add(best_cost);

            if chosen == predicted {
                prev_flag[blk] = true;
            } else {
                prev_flag[blk] = false;
                // §7.3.5.1: rem in 0..=7. Map chosen → rem so that the
                // decoder's `if rem < predicted { rem } else { rem + 1 }`
                // recovers `chosen`.
                let r = if chosen < predicted {
                    chosen
                } else {
                    chosen - 1
                };
                rem[blk] = r;
            }

            // Update intra_grid with our chosen mode so subsequent
            // blocks (in this MB and in later MBs) see it for their
            // own neighbour lookup.
            intra_grid.slot_mut(mb_x, mb_y).intra_4x4_pred_modes[blk] = chosen;
            luma_4x4_quant_raster[blk] = best_z_raster;
            luma_4x4_levels_scan[blk] = best_z_scan;
            blk_has_nz[blk] = best_z_raster.iter().any(|&v| v != 0);

            // Commit reconstruction to the buffer so subsequent blocks
            // see it. Mirrors the decoder's per-block recon walk.
            // `best_pred` was folded into `best_recon` during the trial.
            // (Same caveat as round-4 about zero-cbp quadrant rollback —
            // applied below after the cbp_luma decision.)
            let _ = best_pred;
            for j in 0..4 {
                for i in 0..4 {
                    let px = mb_x * 16 + bx + i;
                    let py = mb_y * 16 + by + j;
                    recon_y[py * width + px] = best_recon[j * 4 + i].clamp(0, 255) as u8;
                }
            }
        }

        // §7.3.5 — cbp_luma per 8x8 quadrant: bit `q` set iff any of the
        // 4 child 4x4 blocks has at least one non-zero coefficient.
        let mut cbp_luma: u8 = 0;
        for blk8 in 0..4usize {
            let any_nz = (0..4).any(|sub| blk_has_nz[blk8 * 4 + sub]);
            if any_nz {
                cbp_luma |= 1u8 << blk8;
            }
        }

        // Roll back recon for any 8x8 quadrant whose cbp bit is 0:
        // the decoder will not receive those 4x4 residuals, so it
        // reconstructs `pred + 0`. Re-derive that here. We must use
        // each block's chosen predictor — recompute it from the modes
        // (the recon buffer is already updated to the residual recon,
        // which would diverge from the decoder for zero-cbp quadrants).
        for blk8 in 0..4usize {
            if (cbp_luma >> blk8) & 1 == 1 {
                continue;
            }
            for sub in 0..4usize {
                let blk = blk8 * 4 + sub;
                let (bx, by) = I4X4_XY[blk];
                let mode = Intra4x4Mode::from_u8(chosen_mode[blk]).expect("valid mode");
                // Re-gather neighbour samples — the recon buffer at the
                // (bx, by) position has already been updated to the
                // post-residual recon for the trial. To re-predict the
                // same way the decoder will, we need the recon as it
                // was BEFORE this block. But we already overwrote it.
                //
                // Fortunately, the decoder's prediction reads only
                // samples LEFT, ABOVE, ABOVE-LEFT, ABOVE-RIGHT of this
                // 4x4 block — none of which lie inside the block
                // itself. Those neighbour samples are unchanged: we
                // overwrote (bx..bx+4, by..by+4) only. So recompute
                // the predictor against the current recon and it will
                // match the decoder.
                let avail = i4x4_avail(mb_x, mb_y, bx, by, width_mbs);
                let pred = predict_4x4(mode, recon_y, width, mb_x, mb_y, bx, by, avail)
                    .expect("predictor exists for chosen mode");
                for j in 0..4 {
                    for i in 0..4 {
                        let px = mb_x * 16 + bx + i;
                        let py = mb_y * 16 + by + j;
                        recon_y[py * width + px] = pred[j * 4 + i].clamp(0, 255) as u8;
                    }
                }
            }
        }

        // ----- Chroma: dispatch via 4:2:0 / 4:2:2 helper. -----
        let chroma_array_type = self.cfg.chroma_format_idc;
        let chroma_block = encode_chroma_block(
            chroma_array_type,
            frame.u,
            frame.v,
            recon_u,
            recon_v,
            chroma_width,
            chroma_height,
            mb_x,
            mb_y,
            qp_c,
        );
        let best_chroma_mode = chroma_block.pred_mode;
        let cbp_chroma = chroma_block.cbp_chroma;
        let n_chroma_blk = ChromaBlock::cb_block_count(chroma_array_type);
        let mut u_dc_levels = [0i32; 4];
        let mut v_dc_levels = [0i32; 4];
        let mut u_ac_levels = [[0i32; 16]; 4];
        let mut v_ac_levels = [[0i32; 16]; 4];
        if chroma_array_type == 1 {
            u_dc_levels.copy_from_slice(&chroma_block.dc_cb[..4]);
            v_dc_levels.copy_from_slice(&chroma_block.dc_cr[..4]);
            u_ac_levels.copy_from_slice(&chroma_block.ac_cb[..4]);
            v_ac_levels.copy_from_slice(&chroma_block.ac_cr[..4]);
        }
        install_chroma_recon(
            chroma_array_type,
            chroma_width,
            mb_x,
            mb_y,
            &chroma_block,
            recon_u,
            recon_v,
        );
        let _ = chroma_height;

        // ----- CAVLC neighbour grid update. -----
        let pic_w_mbs = self.cfg.width / 16;
        let mb_addr = (mb_y as u32) * pic_w_mbs + (mb_x as u32);
        {
            let cur = &mut nc_grid.mbs[mb_addr as usize];
            cur.is_available = true;
            cur.is_intra = true;
            cur.is_skip = false;
            cur.is_i_pcm = false;
            cur.luma_total_coeff = [0u8; 16];
        }
        // Per-AC-block nC for the I_NxN luma path. Same nC derivation
        // (LumaNcKind::Ac), updated progressively as in the I_16x16
        // path. Blocks whose 8x8 quadrant cbp is 0 contribute
        // TotalCoeff = 0.
        let mut luma_4x4_nc = [0i32; 16];
        let mut own_totals = [0u8; 16];
        for blk in 0..16usize {
            let blk8 = blk / 4;
            nc_grid.mbs[mb_addr as usize].luma_total_coeff = own_totals;
            let nc = derive_nc_luma(nc_grid, mb_addr, blk as u8, LumaNcKind::Ac, true, false);
            luma_4x4_nc[blk] = nc;
            // Only blocks in transmitted quadrants contribute non-zero
            // TotalCoeff.
            if (cbp_luma >> blk8) & 1 == 1 {
                let tc = luma_4x4_levels_scan[blk]
                    .iter()
                    .filter(|&&v| v != 0)
                    .count() as u8;
                own_totals[blk] = tc;
            } else {
                own_totals[blk] = 0;
            }
        }
        nc_grid.mbs[mb_addr as usize].luma_total_coeff = own_totals;

        // ----- Emit syntax. -----
        if chroma_array_type == 1 {
            let mb_cfg = INxNMcbConfig {
                prev_intra4x4_pred_mode_flag: prev_flag,
                rem_intra4x4_pred_mode: rem,
                intra_chroma_pred_mode: best_chroma_mode.as_u8(),
                cbp_luma,
                cbp_chroma,
                mb_qp_delta: 0,
                luma_4x4_levels: luma_4x4_levels_scan,
                luma_4x4_nc,
                chroma_dc_cb: u_dc_levels,
                chroma_dc_cr: v_dc_levels,
                chroma_ac_cb: u_ac_levels,
                chroma_ac_cr: v_ac_levels,
            };
            write_i_nxn_mb(sw, &mb_cfg).expect("write I_NxN mb");
        } else {
            // Round-27: 4:2:2 chroma layout.
            crate::encoder::macroblock::write_i_nxn_mb_chroma(
                sw,
                &prev_flag,
                &rem,
                best_chroma_mode.as_u8(),
                cbp_luma,
                cbp_chroma,
                0,
                &luma_4x4_levels_scan,
                &luma_4x4_nc,
                crate::encoder::macroblock::ChromaWriteKind::Yuv422 {
                    chroma_dc_cb: &chroma_block.dc_cb,
                    chroma_dc_cr: &chroma_block.dc_cr,
                    chroma_ac_cb: &chroma_block.ac_cb,
                    chroma_ac_cr: &chroma_block.ac_cr,
                },
            )
            .expect("write I_NxN mb (4:2:2)");
        }

        // intra_grid was updated incrementally above; nothing else to
        // commit here.

        // §8.7.2.1 — per-4x4 nonzero mask. For I_NxN the per-block
        // nonzero status is exactly the per-block AC scan flags computed
        // earlier (`blk_has_nz`); blocks in zero-cbp 8x8 quadrants are
        // already false there.
        let luma_nonzero_4x4 = luma_nz_mask_from_blocks(&blk_has_nz);
        let chroma_nz_mask = chroma_nz_mask_from_chroma_block(&chroma_block, n_chroma_blk);
        MbTrial {
            luma_cost: mb_luma_cost,
            deblock: MbDeblockInfo {
                is_intra: true,
                qp_y,
                luma_nonzero_4x4,
                chroma_nonzero_4x4: chroma_nz_mask,
                ..Default::default()
            },
        }
    }
}

/// Output of [`Encoder::encode_idr`].
pub struct EncodedIdr {
    pub annex_b: Vec<u8>,
    pub recon_y: Vec<u8>,
    pub recon_u: Vec<u8>,
    pub recon_v: Vec<u8>,
    pub recon_width: u32,
    pub recon_height: u32,
    /// Per-8x8-partition MV state. IDR is all-intra so every partition's
    /// `is_intra = true`. Provided for API symmetry with `EncodedP` so
    /// that an IDR can be passed as an L0/L1 anchor to a B-slice — the
    /// §8.4.1.2.2 step 7 colZeroFlag check then short-circuits to 0
    /// (the `if is_intra` branch in `is_colocated_zero_mv`).
    pub partition_mvs: Vec<FrameRefPartitionMv>,
}

/// One 8x8 partition's per-list MV state, captured at picture-level so
/// later B-slices can use this picture as the L1 anchor for §8.4.1.2.2
/// spatial direct derivation (specifically the colZeroFlag step).
///
/// Indexing: `partition` is 0..=3 (top-left, top-right, bottom-left,
/// bottom-right of the parent MB, mirroring `MvGridSlot::mv_l0_8x8` order).
/// `(mb_y * width_mbs + mb_x) * 4 + partition`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameRefPartitionMv {
    /// L0 motion vector in quarter-pel units. `(0, 0)` for intra MBs.
    pub mv_l0: (i16, i16),
    /// L0 reference index. `-1` for intra MBs or when L0 not used.
    pub ref_idx_l0: i8,
    /// True when the parent MB is intra-coded (mvCol = 0, refIdxCol = -1
    /// per §8.4.1.2.1 NOTE). Drives §8.4.1.2.2 step 7's colZeroFlag = 0
    /// fallback for intra colocated MBs.
    pub is_intra: bool,
}

/// Output of [`Encoder::encode_p`] — a single P-slice access unit and
/// the matching reconstruction. Same layout as [`EncodedIdr`] except
/// the bitstream contains only the slice NAL (no SPS/PPS — those were
/// emitted by the IDR).
pub struct EncodedP {
    pub annex_b: Vec<u8>,
    pub recon_y: Vec<u8>,
    pub recon_u: Vec<u8>,
    pub recon_v: Vec<u8>,
    pub recon_width: u32,
    pub recon_height: u32,
    /// `frame_num` of this P-frame. Increments from the IDR's 0.
    pub frame_num: u32,
    /// `pic_order_cnt_lsb` of this P-frame.
    pub pic_order_cnt_lsb: u32,
    /// Per-8x8-partition MV state for each MB. Length =
    /// `(width / 16) * (height / 16) * 4`. Used by [`Encoder::encode_b`]
    /// when this P-frame is the L1 anchor — the spatial direct
    /// derivation (§8.4.1.2.2 step 7) needs the colocated block's L0
    /// MV / refIdx to evaluate `colZeroFlag`.
    pub partition_mvs: Vec<FrameRefPartitionMv>,
}

/// Per-MB encoder-side MV / refIdx slot for the §8.4.1.3 mvp neighbour
/// grid. Round-19 extends round-16's single-MV slot to track **per-8x8
/// partition** MVs (4 entries) so that the encoder can emit P_8x8 MBs
/// (mb_type=3, sub_mb_type=PL08x8) with mvds that match the §8.4.1.3
/// neighbour-mv predictor the decoder will use.
///
/// Indexing of `mv_l0` / `ref_idx_l0_8x8`:
///   0 → top-left 8x8, 1 → top-right, 2 → bottom-left, 3 → bottom-right
/// (Figure 6-10 8x8 partition order; same as `LUMA_4X4_BLK[blk*4]`.)
///
/// For P_L0_16x16 MBs all four 8x8 entries carry the same MV / ref —
/// downstream MBs reading any 8x8 see the single 16x16 MV, preserving
/// round-16 / round-17 / round-18 behaviour exactly.
#[derive(Debug, Clone, Copy)]
struct MvGridSlot {
    /// True once this MB has been encoded (so subsequent MBs can read it).
    available: bool,
    /// True for intra-coded MBs (their mvp contribution is mv=0, ref=-1).
    is_intra: bool,
    /// L0 reference index per 8x8 partition (always 0 in our single-ref
    /// P-slice — kept per-partition to leave room for multi-ref later).
    ref_idx_l0_8x8: [i32; 4],
    /// L0 motion vector per 8x8 partition, in 1/4-pel units.
    mv_l0_8x8: [Mv; 4],
}

impl MvGridSlot {
    /// Round-16 compat shim — many call sites only need the single
    /// representative MV / ref. Returns the top-left 8x8's values which
    /// equals the unique 16x16 MV for non-4MV MBs.
    fn ref_idx_l0(&self) -> i32 {
        self.ref_idx_l0_8x8[0]
    }
    fn mv_l0(&self) -> Mv {
        self.mv_l0_8x8[0]
    }
}

impl Default for MvGridSlot {
    fn default() -> Self {
        Self {
            available: false,
            is_intra: false,
            ref_idx_l0_8x8: [-1; 4],
            mv_l0_8x8: [Mv::ZERO; 4],
        }
    }
}

#[derive(Debug, Clone)]
struct MvGrid {
    width_mbs: usize,
    slots: Vec<MvGridSlot>,
}

impl MvGrid {
    fn new(width_mbs: usize, height_mbs: usize) -> Self {
        Self {
            width_mbs,
            slots: vec![MvGridSlot::default(); width_mbs * height_mbs],
        }
    }
    fn slot(&self, mb_x: usize, mb_y: usize) -> &MvGridSlot {
        &self.slots[mb_y * self.width_mbs + mb_x]
    }
    fn slot_mut(&mut self, mb_x: usize, mb_y: usize) -> &mut MvGridSlot {
        &mut self.slots[mb_y * self.width_mbs + mb_x]
    }
}

/// Build the (A, B, C, D) NeighbourMv records for the current MB
/// (mb_x, mb_y) — single 16x16 partition. Per §6.4.11.7 the four
/// neighbours of a 16x16 partition's top-left 4x4 (block 0) are:
///
/// * A: the 4x4 at (-1, 0) of current MB → MB(mb_x-1, mb_y) 8x8 #1
///   (top-right 4x4 column, row 0 → 8x8 partition that contains 4x4
///   (3, 0) = top-right).
/// * B: the 4x4 at (0, -1) → MB(mb_x, mb_y-1) 8x8 #2 (bottom-left 4x4
///   column → contains 4x4 (0, 3) = bottom-left).
/// * C: the 4x4 at (16, -1) → MB(mb_x+1, mb_y-1) 8x8 #2 (bottom-left
///   of MB above-right, which contains 4x4 (0, 3)).
/// * D: the 4x4 at (-1, -1) → MB(mb_x-1, mb_y-1) 8x8 #3 (bottom-right).
///
/// For 16x16-partitioned neighbours, all 4 8x8 cells share the same MV
/// so the read is unambiguous. For 16x8 / 8x16 / 8x8 partitioned
/// neighbours, we read the 8x8 cell that **spatially** contains the
/// requested 4x4 corner — matching what the decoder does when it
/// derives mvpLX for the current MB's partition.
///
/// Each neighbour is "MVpred-usable" iff its MB is encoded AND not intra.
/// Intra neighbours collapse to (mv=0, ref=-1) per §8.4.1.3.2 step 2a.
fn neighbour_mvs_16x16(
    grid: &MvGrid,
    mb_x: usize,
    mb_y: usize,
) -> (NeighbourMv, NeighbourMv, NeighbourMv, NeighbourMv) {
    // (8x8 partition index in neighbour MB) — picks the 8x8 cell
    // spatially adjacent to the current MB's top-left 4x4 (block 0).
    let make = |maybe: Option<&MvGridSlot>, part_idx: usize| -> NeighbourMv {
        match maybe {
            Some(s) if s.available => {
                if s.is_intra {
                    NeighbourMv::intra_but_mb_available()
                } else {
                    NeighbourMv {
                        available: s.ref_idx_l0_8x8[part_idx] >= 0,
                        mb_available: true,
                        partition_available: true,
                        ref_idx: s.ref_idx_l0_8x8[part_idx],
                        mv: s.mv_l0_8x8[part_idx],
                    }
                }
            }
            _ => NeighbourMv::UNAVAILABLE,
        }
    };
    let a = if mb_x > 0 {
        Some(grid.slot(mb_x - 1, mb_y))
    } else {
        None
    };
    let b = if mb_y > 0 {
        Some(grid.slot(mb_x, mb_y - 1))
    } else {
        None
    };
    let c = if mb_y > 0 && mb_x + 1 < grid.width_mbs {
        Some(grid.slot(mb_x + 1, mb_y - 1))
    } else {
        None
    };
    let d = if mb_x > 0 && mb_y > 0 {
        Some(grid.slot(mb_x - 1, mb_y - 1))
    } else {
        None
    };
    (
        make(a, 1), // left  → top-right 8x8 (#1)
        make(b, 2), // above → bottom-left 8x8 (#2)
        make(c, 2), // above-right → bottom-left 8x8 (#2)
        make(d, 3), // above-left → bottom-right 8x8 (#3)
    )
}

/// Per §8.4.1.3 + the §8.4.1.3.2 C→D substitution — derive the MVpred
/// for a single 16x16 P_L0 partition at (mb_x, mb_y).
fn mvp_for_16x16(grid: &MvGrid, mb_x: usize, mb_y: usize, ref_idx: i32) -> Mv {
    let (a, b, c, d) = neighbour_mvs_16x16(grid, mb_x, mb_y);
    let inputs = MvpredInputs::from_abc(a, b, c, ref_idx, Default::default());
    crate::mv_deriv::derive_mvpred_with_d(&inputs, d)
}

/// Per §8.4.1.2 — derive the (refIdxL0, mvL0) for a P_Skip MB at
/// (mb_x, mb_y). With the C→D substitution.
fn p_skip_mv(grid: &MvGrid, mb_x: usize, mb_y: usize) -> (i32, Mv) {
    let (a, b, c, d) = neighbour_mvs_16x16(grid, mb_x, mb_y);
    crate::mv_deriv::derive_p_skip_mv_with_d(a, b, c, d)
}

/// Round-19 — read the MV/refIdx of the 8x8 partition containing the
/// 4x4 block at MB-relative coordinates `(bx_4x4, by_4x4) ∈ {0,1,2,3}²`
/// of the MB at `(mb_x, mb_y)`. Returns a [`NeighbourMv`] with the
/// availability flags set per §8.4.1.1 / §6.4.11.1:
///
/// * `mb_xy` outside the picture or referencing an unencoded MB →
///   [`NeighbourMv::UNAVAILABLE`].
/// * intra-coded MB → [`NeighbourMv::intra_but_mb_available`] (so the
///   median collapses for that slot but the §8.4.1.3.2 C→D substitution
///   does **not** fire).
/// * MB encoded but the targeted 8x8 partition is in a sub-MB whose
///   `available` bit is false (e.g. the encoder is mid-MB): the partition
///   is `partition_not_yet_decoded_same_mb` so C→D substitution fires.
///
/// `inflight_avail` (length 4) is the per-8x8-partition availability
/// for the **current** MB-in-flight (`mb_x`, `mb_y`), allowing within-MB
/// neighbour reads of already-decoded sub-partitions. `mv_inflight` and
/// `ref_inflight` are the corresponding MVs / refs of those partitions
/// (slots gated by `inflight_avail[i]==true`).
#[allow(clippy::too_many_arguments)]
fn neighbour_8x8_partition_mv(
    grid: &MvGrid,
    mb_x: usize,
    mb_y: usize,
    bx_4x4: i32,
    by_4x4: i32,
    inflight_avail: [bool; 4],
    mv_inflight: [Mv; 4],
    ref_inflight: [i32; 4],
) -> NeighbourMv {
    // Map (bx_4x4, by_4x4) ∈ Z² (allowing -1) onto a (mb_x', mb_y',
    // 8x8 partition index ∈ 0..=3).
    let (nmb_x, nmb_y, lx, ly) = if bx_4x4 < 0 && (0..4).contains(&by_4x4) {
        // Left MB.
        if mb_x == 0 {
            return NeighbourMv::UNAVAILABLE;
        }
        // Wrap into right edge of left MB: bx becomes 3.
        (mb_x - 1, mb_y, 3, by_4x4)
    } else if (0..4).contains(&bx_4x4) && by_4x4 < 0 {
        // Above MB.
        if mb_y == 0 {
            return NeighbourMv::UNAVAILABLE;
        }
        (mb_x, mb_y - 1, bx_4x4, 3)
    } else if bx_4x4 < 0 && by_4x4 < 0 {
        // Above-left MB (only used as D).
        if mb_x == 0 || mb_y == 0 {
            return NeighbourMv::UNAVAILABLE;
        }
        (mb_x - 1, mb_y - 1, 3, 3)
    } else if (0..4).contains(&bx_4x4) && (0..4).contains(&by_4x4) {
        // Same MB.
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
            // Partition not yet decoded → §6.4.11.1 partition-unavailable.
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
    let s = grid.slot(nmb_x, nmb_y);
    if !s.available {
        return NeighbourMv::UNAVAILABLE;
    }
    if s.is_intra {
        return NeighbourMv::intra_but_mb_available();
    }
    let part = 2 * (ly as usize / 2) + (lx as usize / 2);
    NeighbourMv {
        available: s.ref_idx_l0_8x8[part] >= 0,
        mb_available: true,
        partition_available: true,
        ref_idx: s.ref_idx_l0_8x8[part],
        mv: s.mv_l0_8x8[part],
    }
}

/// Round-19 — derive the §8.4.1.3 mvpLX for one 8x8 partition of a
/// P_8x8 MB at `(mb_x, mb_y)`. `sub_idx` ∈ 0..=3 in 8x8 raster
/// (0=TL, 1=TR, 2=BL, 3=BR). `inflight_*` arrays carry the already-
/// decoded sub-partitions' MVs/refs of the **current** MB so within-MB
/// neighbour reads work.
#[allow(clippy::too_many_arguments)]
fn mvp_for_p_8x8_partition(
    grid: &MvGrid,
    mb_x: usize,
    mb_y: usize,
    sub_idx: usize,
    ref_idx: i32,
    inflight_avail: [bool; 4],
    mv_inflight: [Mv; 4],
    ref_inflight: [i32; 4],
) -> Mv {
    // Top-left 4x4 of this 8x8 partition, in 4x4-block units.
    let sub_x = (sub_idx % 2) as i32;
    let sub_y = (sub_idx / 2) as i32;
    let bx = sub_x * 2;
    let by = sub_y * 2;
    // Partition width = 8 = 2 4x4 blocks → C is at (bx + 2, by - 1).
    let a = neighbour_8x8_partition_mv(
        grid,
        mb_x,
        mb_y,
        bx - 1,
        by,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let b = neighbour_8x8_partition_mv(
        grid,
        mb_x,
        mb_y,
        bx,
        by - 1,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let c = neighbour_8x8_partition_mv(
        grid,
        mb_x,
        mb_y,
        bx + 2,
        by - 1,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let d = neighbour_8x8_partition_mv(
        grid,
        mb_x,
        mb_y,
        bx - 1,
        by - 1,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let inputs = MvpredInputs::from_abc(a, b, c, ref_idx, Default::default());
    crate::mv_deriv::derive_mvpred_with_d(&inputs, d)
}

/// Round-16 — generate the inter predictor for one 16x16 luma MB at
/// (mb_x, mb_y), using mv_l0 (in 1/4-pel units) against the reference
/// luma plane. Output is `[i32; 256]` row-major (16 rows of 16 samples
/// each).
///
/// For integer-pel MVs (mv_l0 is a multiple of 4), `x_frac == y_frac
/// == 0` and the call simplifies to a clipped copy.
fn build_inter_pred_luma(
    ref_y: &[u8],
    ref_w: u32,
    ref_h: u32,
    mb_x: usize,
    mb_y: usize,
    mv: Mv,
) -> [i32; 256] {
    // Convert ref to i32 plane.
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
    .expect("luma interpolation");
    dst
}

/// Round-19 — generate the inter predictor for one 8x8 luma sub-MB
/// **partition** of a P_8x8 MB at `(mb_x, mb_y)`. `(sub_x, sub_y) ∈
/// {0, 1}` selects which 8x8 inside the MB. Output is `[i32; 64]`
/// row-major (8 rows of 8 samples).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_inter_pred_luma_8x8(
    ref_y: &[u8],
    ref_w: u32,
    ref_h: u32,
    mb_x: usize,
    mb_y: usize,
    sub_x: usize,
    sub_y: usize,
    mv: Mv,
) -> [i32; 64] {
    let stride = ref_w as usize;
    let h = ref_h as usize;
    let ref_i32: Vec<i32> = ref_y.iter().take(stride * h).map(|&v| v as i32).collect();

    let mv_int_x = mv.x >> 2;
    let mv_int_y = mv.y >> 2;
    let x_frac = (mv.x & 3) as u8;
    let y_frac = (mv.y & 3) as u8;
    let int_x = (mb_x * 16 + sub_x * 8) as i32 + mv_int_x;
    let int_y = (mb_y * 16 + sub_y * 8) as i32 + mv_int_y;

    let mut dst = [0i32; 64];
    interpolate_luma(
        &ref_i32, stride, stride, h, int_x, int_y, x_frac, y_frac, 8, 8, 8, &mut dst, 8,
    )
    .expect("luma interpolation 8x8");
    dst
}

/// Round-19 — generate the inter predictor for one 4x4 chroma sub-MB
/// partition of a P_8x8 MB (4:2:0). `(sub_x, sub_y) ∈ {0, 1}` selects
/// the sub-block. Output is `[i32; 16]` row-major.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_inter_pred_chroma_4x4(
    ref_c: &[u8],
    ref_cw: u32,
    ref_ch: u32,
    mb_x: usize,
    mb_y: usize,
    sub_x: usize,
    sub_y: usize,
    mv: Mv,
) -> [i32; 16] {
    let stride = ref_cw as usize;
    let h = ref_ch as usize;
    let ref_i32: Vec<i32> = ref_c.iter().take(stride * h).map(|&v| v as i32).collect();

    // §8.4.1.4 — 4:2:0 chroma uses luma MV directly, with 1/8-pel
    // fractional positions (mvC = mv_luma; xFracC = mvC & 7).
    let mv_cx = mv.x;
    let mv_cy = mv.y;
    let int_x = (mb_x * 8 + sub_x * 4) as i32 + (mv_cx >> 3);
    let int_y = (mb_y * 8 + sub_y * 4) as i32 + (mv_cy >> 3);
    let x_frac = (mv_cx & 7) as u8;
    let y_frac = (mv_cy & 7) as u8;

    let mut dst = [0i32; 16];
    interpolate_chroma(
        &ref_i32, stride, stride, h, int_x, int_y, x_frac, y_frac, 4, 4, 8, &mut dst, 4,
    )
    .expect("chroma interpolation 4x4");
    dst
}

/// Round-16 — generate the inter predictor for one 8x8 chroma MB
/// (4:2:0) at (mb_x, mb_y), using `mv_l0` (luma 1/4-pel units; chroma
/// derives 1/8-pel units per §8.4.1.4 for 4:2:0).
fn build_inter_pred_chroma(
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

    // §8.4.1.4 / eq. 8-228..8-231 — chroma MV is luma MV (4:2:0).
    // xIntC = mb_x*8 + (mvC >> 3); xFracC = mvC & 7. mvC = mv (luma 1/4-pel).
    let mv_cx = mv.x;
    let mv_cy = mv.y;
    let int_x = (mb_x * 8) as i32 + (mv_cx >> 3);
    let int_y = (mb_y * 8) as i32 + (mv_cy >> 3);
    let x_frac = (mv_cx & 7) as u8;
    let y_frac = (mv_cy & 7) as u8;

    let mut dst = [0i32; 64];
    interpolate_chroma(
        &ref_i32, stride, stride, h, int_x, int_y, x_frac, y_frac, 8, 8, 8, &mut dst, 8,
    )
    .expect("chroma interpolation");
    dst
}

/// Encode the chroma residual for one P_L0 MB given the chroma
/// inter predictor (instead of the intra predictor used by
/// [`encode_chroma_residual`]). Same machinery as the intra path —
/// shared via a small wrapper; we just need an `i32` predictor.
///
/// Returns `(dc_levels, ac_levels, recon_residual)` matching
/// [`encode_chroma_residual`]'s signature.
fn encode_chroma_residual_inter(
    src_plane: &[u8],
    chroma_stride: usize,
    mb_x: usize,
    mb_y: usize,
    pred: &[i32; 64],
    qp_c: i32,
) -> ([i32; 4], [[i32; 16]; 4], [i32; 64]) {
    // Re-use the intra path — its only assumption on the predictor is
    // that it's a `[i32; 64]` row-major 8x8 block. The is_intra=true
    // rounding flag in the AC quantization differs slightly from inter,
    // but for round-16 we accept the intra rounding for chroma (the
    // bitstream is correct either way; only the rate-distortion trade-
    // off shifts marginally).
    encode_chroma_residual(src_plane, chroma_stride, mb_x, mb_y, pred, qp_c)
}

/// Round-27 — Chroma block encode result. Holds residual coefficients
/// (DC + AC, both planes, sized for 4:2:0 or 4:2:2), the chosen
/// intra-chroma mode, derived `cbp_chroma`, and the reconstructed
/// chroma residual to be added back to the predictor.
///
/// 4:2:0 path fills the first 4 entries of every array; 4:2:2 fills
/// all 8.
struct ChromaBlock {
    pred_mode: IntraChromaMode,
    cbp_chroma: u8,
    /// Quantized chroma DC levels for Cb / Cr. Length 4 (4:2:0) or 8
    /// (4:2:2 — already in spec scan order).
    dc_cb: [i32; 8],
    dc_cr: [i32; 8],
    /// Per-4x4-block AC scan for Cb / Cr. Up to 8 blocks for 4:2:2;
    /// 4:2:0 uses only the first 4.
    ac_cb: [[i32; 16]; 8],
    ac_cr: [[i32; 16]; 8],
    /// Reconstructed residual (sample-domain) for both planes. Size
    /// 64 (4:2:0 8x8) or 128 (4:2:2 8x16). Stored in the upper bound
    /// length so the caller iterates over the right number of rows.
    recon_cb: Vec<i32>,
    recon_cr: Vec<i32>,
    /// Predictor samples used (64 / 128 entries).
    pred_cb: Vec<i32>,
    pred_cr: Vec<i32>,
}

impl ChromaBlock {
    /// Per-4x4-block AC nonzero flags for Cb / Cr (length 4 for 4:2:0,
    /// 8 for 4:2:2). Used by the deblock walker's chroma-nonzero mask.
    fn cb_block_count(chroma_array_type: u32) -> usize {
        match chroma_array_type {
            1 => 4,
            2 => 8,
            _ => 0,
        }
    }
}

/// Round-27 — pick the best chroma intra-prediction mode (DC / H / V /
/// Plane) by SAD, encode the residual, reconstruct the samples, and
/// return everything the caller needs to install the recon, emit the
/// MB syntax, and update the deblock walker mask.
///
/// `chroma_array_type` selects 4:2:0 (1) vs 4:2:2 (2). 4:4:4 is out
/// of scope and panics.
#[allow(clippy::too_many_arguments)]
fn encode_chroma_block(
    chroma_array_type: u32,
    frame_u: &[u8],
    frame_v: &[u8],
    recon_u: &[u8],
    recon_v: &[u8],
    chroma_width: usize,
    chroma_height: usize,
    mb_x: usize,
    mb_y: usize,
    qp_c: i32,
) -> ChromaBlock {
    use crate::encoder::intra_pred::{predict_chroma_8x16, sad_8x16};
    let (h_chroma, n_blk) = match chroma_array_type {
        1 => (8usize, 4usize),
        2 => (16usize, 8usize),
        _ => panic!("encode_chroma_block: unsupported chroma_array_type {chroma_array_type}"),
    };

    let mut best_mode = IntraChromaMode::Dc;
    let mut best_pred_u_64 = [0i32; 64];
    let mut best_pred_v_64 = [0i32; 64];
    let mut best_pred_u_128 = [0i32; 128];
    let mut best_pred_v_128 = [0i32; 128];
    let mut best_sad = u64::MAX;
    for &mode in &[
        IntraChromaMode::Dc,
        IntraChromaMode::Horizontal,
        IntraChromaMode::Vertical,
        IntraChromaMode::Plane,
    ] {
        if chroma_array_type == 1 {
            let (Some(pu), Some(pv)) = (
                predict_chroma_8x8(mode, recon_u, chroma_width, chroma_height, mb_x, mb_y),
                predict_chroma_8x8(mode, recon_v, chroma_width, chroma_height, mb_x, mb_y),
            ) else {
                continue;
            };
            let s = sad_8x8(frame_u, chroma_width, mb_x * 8, mb_y * 8, &pu)
                + sad_8x8(frame_v, chroma_width, mb_x * 8, mb_y * 8, &pv);
            if s < best_sad {
                best_sad = s;
                best_mode = mode;
                best_pred_u_64 = pu;
                best_pred_v_64 = pv;
            }
        } else {
            // 4:2:2
            let (Some(pu), Some(pv)) = (
                predict_chroma_8x16(mode, recon_u, chroma_width, chroma_height, mb_x, mb_y),
                predict_chroma_8x16(mode, recon_v, chroma_width, chroma_height, mb_x, mb_y),
            ) else {
                continue;
            };
            let s = sad_8x16(frame_u, chroma_width, mb_x * 8, mb_y * 16, &pu)
                + sad_8x16(frame_v, chroma_width, mb_x * 8, mb_y * 16, &pv);
            if s < best_sad {
                best_sad = s;
                best_mode = mode;
                best_pred_u_128 = pu;
                best_pred_v_128 = pv;
            }
        }
    }

    let mut dc_cb = [0i32; 8];
    let mut dc_cr = [0i32; 8];
    let mut ac_cb = [[0i32; 16]; 8];
    let mut ac_cr = [[0i32; 16]; 8];
    let mut recon_cb = vec![0i32; 8 * h_chroma];
    let mut recon_cr = vec![0i32; 8 * h_chroma];

    if chroma_array_type == 1 {
        let (u_dc, u_ac, u_recon) =
            encode_chroma_residual(frame_u, chroma_width, mb_x, mb_y, &best_pred_u_64, qp_c);
        let (v_dc, v_ac, v_recon) =
            encode_chroma_residual(frame_v, chroma_width, mb_x, mb_y, &best_pred_v_64, qp_c);
        dc_cb[..4].copy_from_slice(&u_dc);
        dc_cr[..4].copy_from_slice(&v_dc);
        ac_cb[..4].copy_from_slice(&u_ac);
        ac_cr[..4].copy_from_slice(&v_ac);
        recon_cb[..64].copy_from_slice(&u_recon);
        recon_cr[..64].copy_from_slice(&v_recon);
    } else {
        let (u_dc, u_ac, u_recon) =
            encode_chroma_residual_422(frame_u, chroma_width, mb_x, mb_y, &best_pred_u_128, qp_c);
        let (v_dc, v_ac, v_recon) =
            encode_chroma_residual_422(frame_v, chroma_width, mb_x, mb_y, &best_pred_v_128, qp_c);
        dc_cb = u_dc;
        dc_cr = v_dc;
        ac_cb = u_ac;
        ac_cr = v_ac;
        recon_cb[..128].copy_from_slice(&u_recon);
        recon_cr[..128].copy_from_slice(&v_recon);
    }

    let any_dc_nz =
        dc_cb[..n_blk].iter().any(|&v| v != 0) || dc_cr[..n_blk].iter().any(|&v| v != 0);
    let any_ac_nz = ac_cb[..n_blk].iter().any(|blk| blk.iter().any(|&v| v != 0))
        || ac_cr[..n_blk].iter().any(|blk| blk.iter().any(|&v| v != 0));
    let cbp_chroma: u8 = if any_ac_nz {
        2
    } else if any_dc_nz {
        1
    } else {
        0
    };

    let pred_cb: Vec<i32> = if chroma_array_type == 1 {
        best_pred_u_64.to_vec()
    } else {
        best_pred_u_128.to_vec()
    };
    let pred_cr: Vec<i32> = if chroma_array_type == 1 {
        best_pred_v_64.to_vec()
    } else {
        best_pred_v_128.to_vec()
    };

    ChromaBlock {
        pred_mode: best_mode,
        cbp_chroma,
        dc_cb,
        dc_cr,
        ac_cb,
        ac_cr,
        recon_cb,
        recon_cr,
        pred_cb,
        pred_cr,
    }
}

/// Round-27 — pack a `ChromaBlock`'s per-4x4 AC nonzero flags into the
/// 16-bit mask the deblock walker reads. Same packing as the decoder's
/// `compute_chroma_nonzero_mask`: bits 0..=(n-1) are Cb, bits
/// 8..=(8+n-1) are Cr, where n = 4 for 4:2:0 and n = 8 for 4:2:2.
fn chroma_nz_mask_from_chroma_block(block: &ChromaBlock, n_blk: usize) -> u16 {
    let mut mask = 0u16;
    if block.cbp_chroma != 2 {
        return mask;
    }
    for k in 0..n_blk {
        if block.ac_cb[k].iter().any(|&v| v != 0) {
            mask |= 1u16 << k;
        }
        if block.ac_cr[k].iter().any(|&v| v != 0) {
            mask |= 1u16 << (8 + k);
        }
    }
    mask
}

/// Round-28 — Intra_16x16 encode of one chroma plane for 4:4:4
/// ("coded like luma" per §7.3.5.3). Mirrors the luma pipeline:
///
///   1. Build §8.3.3 16x16 predictor on the chroma plane using the
///      caller-supplied luma `mode` (per §8.3.4.1, ChromaArrayType==3
///      reuses the luma Intra_16x16 mode and predictor formula).
///   2. Sample residual against the predictor.
///   3. Per-4x4-block forward integer transform; collect the 16 DCs;
///      forward 4x4 Hadamard + quantize.
///   4. Per-block AC quantize.
///   5. Inverse: Hadamard-DC → per-block DC-preserved inverse → add
///      predictor → clamp into recon plane.
///
/// Returns:
///   * `dc_levels` — quantized 16-entry DC array in raster order
///     (caller applies `zigzag_scan_4x4` for emit).
///   * `ac_levels_scan` — per-block AC scan layout (positions 0..=14;
///     §6.4.3 raster-Z indexing).
///   * `ac_quant_raster` — per-block AC raster-order quantized levels
///     (slot 0 zeroed) — used for the inverse path.
///   * `any_ac_nz` — true iff any 4x4 AC block has at least one
///     non-zero level (drives `cbp_luma == 15` gating).
///
/// Stateful side-effect: writes the reconstructed chroma plane samples
/// into `recon_chroma` (16x16 tile at MB origin).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn encode_intra16x16_chroma_plane_luma_like(
    mode: I16x16Mode,
    src_plane: &[u8],
    recon_chroma: &mut [u8],
    chroma_width: usize,
    chroma_height: usize,
    mb_x: usize,
    mb_y: usize,
    qp_c: i32,
) -> ([i32; 16], [[i32; 16]; 16], [[i32; 16]; 16], bool) {
    // (1) Build the 16x16 predictor from the chroma plane's prior
    // recon. `predict_16x16` is plane-agnostic — it just reads u8
    // samples at MB-relative offsets. For mb_x==0 / mb_y==0 the
    // predictor falls back to mid-grey (128) per the helper's
    // unavailability handling — that's the same fallback the decoder
    // uses for the §8.3.3.1/2/3/4 not-available branches.
    let pred = predict_16x16(mode, recon_chroma, chroma_width, chroma_height, mb_x, mb_y)
        .unwrap_or([128i32; 256]);

    // (2) Sample residual.
    let mut residual = [0i32; 256];
    for j in 0..16usize {
        for i in 0..16usize {
            let s = src_plane[(mb_y * 16 + j) * chroma_width + (mb_x * 16 + i)] as i32;
            residual[j * 16 + i] = s - pred[j * 16 + i];
        }
    }

    // (3) Per-4x4 forward integer transform.
    let mut coeffs_4x4 = [[0i32; 16]; 16];
    for (blk, slot) in coeffs_4x4.iter_mut().enumerate() {
        let bx = blk % 4;
        let by = blk / 4;
        let mut block = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                block[j * 4 + i] = residual[(by * 4 + j) * 16 + bx * 4 + i];
            }
        }
        *slot = forward_core_4x4(&block);
    }
    // Pull DCs into a row-major 4x4 matrix.
    let mut dc_coeffs = [0i32; 16];
    for (blk, c) in coeffs_4x4.iter().enumerate() {
        let bx = blk % 4;
        let by = blk / 4;
        dc_coeffs[by * 4 + bx] = c[0];
    }
    // Forward Hadamard + quantize. Same helper as luma — chroma DC
    // for 4:4:4 follows the luma DC path per §8.5.10.
    let dc_t = forward_hadamard_4x4(&dc_coeffs);
    let dc_levels = quantize_luma_dc(&dc_t, qp_c, true);

    // (4) Per-block AC quantize. Reorder into §6.4.3 raster-Z.
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

    // (5) Inverse path — produce decoder-bit-exact recon.
    let inv_dc = inverse_hadamard_luma_dc_16x16(&dc_levels, qp_c, &FLAT_4X4_16, 8)
        .expect("4:4:4 chroma DC inverse");
    #[allow(clippy::needless_range_loop)] // raster row-major over 16 4x4 blocks
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
                let px = mb_x * 16 + bx * 4 + i;
                let py = mb_y * 16 + by * 4 + j;
                let p = pred[(by * 4 + j) * 16 + bx * 4 + i];
                let raw = p + r[j * 4 + i];
                recon_chroma[py * chroma_width + px] = raw.clamp(0, 255) as u8;
            }
        }
    }

    (dc_levels, ac_scan, ac_quant_raster, any_ac_nz)
}

/// Round-27 — install the chroma block reconstruction back into the
/// per-plane recon buffers. 4:2:0 → 8x8 tile per MB. 4:2:2 → 8x16 tile.
fn install_chroma_recon(
    chroma_array_type: u32,
    chroma_width: usize,
    mb_x: usize,
    mb_y: usize,
    block: &ChromaBlock,
    recon_u: &mut [u8],
    recon_v: &mut [u8],
) {
    let (h_chroma, base_y) = match chroma_array_type {
        1 => (8usize, mb_y * 8),
        2 => (16usize, mb_y * 16),
        _ => return,
    };
    for j in 0..h_chroma {
        for i in 0..8usize {
            let cx = mb_x * 8 + i;
            let cy = base_y + j;
            let pred_u = block.pred_cb[j * 8 + i];
            let pred_v = block.pred_cr[j * 8 + i];
            let res_u = if block.cbp_chroma == 0 {
                0
            } else {
                block.recon_cb[j * 8 + i]
            };
            let res_v = if block.cbp_chroma == 0 {
                0
            } else {
                block.recon_cr[j * 8 + i]
            };
            recon_u[cy * chroma_width + cx] = (pred_u + res_u).clamp(0, 255) as u8;
            recon_v[cy * chroma_width + cx] = (pred_v + res_v).clamp(0, 255) as u8;
        }
    }
}

/// Round-27 — encode one chroma plane's residual (DC + AC) for an 8x16
/// chroma MB (4:2:2). Returns:
///   * 8 quantized scan-order DC levels (already in `ChromaDCLevel`
///     scan order — the decoder consumes this verbatim).
///   * 8 per-4x4-block AC levels in scan order (AC-only layout).
///   * Reconstructed 8x16 residual (`[i32; 128]`, row-major).
///
/// 8 4x4 sub-blocks per plane laid out as:
///   (0,0), (4,0), (0,4), (4,4), (0,8), (4,8), (0,12), (4,12)
/// matching the decoder's `chroma_block_xy` ordering for 4:2:2.
fn encode_chroma_residual_422(
    src_plane: &[u8],
    chroma_stride: usize,
    mb_x: usize,
    mb_y: usize,
    pred: &[i32; 128],
    qp_c: i32,
) -> ([i32; 8], [[i32; 16]; 8], [i32; 128]) {
    use crate::encoder::transform::{
        forward_hadamard_4x2, quantize_chroma_dc_422, scan_chroma_dc_422,
    };
    use crate::transform::inverse_hadamard_chroma_dc_422;

    // Build sample residual against the predictor.
    let mut residual = [0i32; 128];
    for j in 0..16usize {
        for i in 0..8usize {
            let s = src_plane[(mb_y * 16 + j) * chroma_stride + mb_x * 8 + i] as i32;
            residual[j * 8 + i] = s - pred[j * 8 + i];
        }
    }

    // Forward 4x4 on each of the 8 sub-blocks. Order matches the
    // decoder's `chroma_block_xy` for 4:2:2 (raster: 2x4 layout).
    let blocks_xy: [(usize, usize); 8] = [
        (0, 0),
        (4, 0),
        (0, 4),
        (4, 4),
        (0, 8),
        (4, 8),
        (0, 12),
        (4, 12),
    ];
    let mut sub_coeffs = [[0i32; 16]; 8];
    for (k, &(bx, by)) in blocks_xy.iter().enumerate() {
        let mut block = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                block[j * 4 + i] = residual[(by + j) * 8 + bx + i];
            }
        }
        sub_coeffs[k] = forward_core_4x4(&block);
    }

    // Collect the 8 DCs into a row-major 4x2 layout. Block (xO, yO):
    //   k=0 (0,0)  → c[0,0]
    //   k=1 (4,0)  → c[0,1]
    //   k=2 (0,4)  → c[1,0]
    //   k=3 (4,4)  → c[1,1]
    //   k=4 (0,8)  → c[2,0]
    //   k=5 (4,8)  → c[2,1]
    //   k=6 (0,12) → c[3,0]
    //   k=7 (4,12) → c[3,1]
    let dc_in: [i32; 8] = [
        sub_coeffs[0][0], // c[0,0]
        sub_coeffs[1][0], // c[0,1]
        sub_coeffs[2][0], // c[1,0]
        sub_coeffs[3][0], // c[1,1]
        sub_coeffs[4][0], // c[2,0]
        sub_coeffs[5][0], // c[2,1]
        sub_coeffs[6][0], // c[3,0]
        sub_coeffs[7][0], // c[3,1]
    ];

    // Forward 4x2 Hadamard + quantize.
    let dc_t = forward_hadamard_4x2(&dc_in);
    let dc_q = quantize_chroma_dc_422(&dc_t, qp_c, true);
    // Pack into spec scan order so the bitstream can carry it directly.
    let dc_levels = scan_chroma_dc_422(&dc_q);

    // Quantize AC of each sub-block (positions 1..15 in scan order).
    let mut ac_levels = [[0i32; 16]; 8];
    for (k, coeffs) in sub_coeffs.iter().enumerate() {
        let z = quantize_4x4_ac(coeffs, qp_c, true);
        ac_levels[k] = zigzag_scan_4x4_ac(&z);
    }

    // ------- Reconstruct the chroma 8x16 residual -------
    // §8.5.11.2: inverse Hadamard → per-block DC-preserved inverse.
    let dc_back = inverse_hadamard_chroma_dc_422(&dc_levels, qp_c, &FLAT_4X4_16, 8)
        .expect("chroma DC 422 inverse");
    // dc_back is row-major 4x2 → maps to blocks 0..=7 by (k as above).
    let dc_block_index_to_dc_back: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

    let mut recon_residual = [0i32; 128];
    for (k, &(bx, by)) in blocks_xy.iter().enumerate() {
        let mut coeffs = inverse_scan_4x4_zigzag_ac(&ac_levels[k]);
        coeffs[0] = dc_back[dc_block_index_to_dc_back[k]];
        let r = inverse_transform_4x4_dc_preserved(&coeffs, qp_c, &FLAT_4X4_16, 8)
            .expect("chroma 4x4 422 inverse");
        for j in 0..4 {
            for i in 0..4 {
                recon_residual[(by + j) * 8 + bx + i] = r[j * 4 + i];
            }
        }
    }

    (dc_levels, ac_levels, recon_residual)
}

/// Encode one chroma plane's residual (DC + AC) for an 8x8 chroma MB,
/// returning (raster-order DC levels for the 2x2 DC matrix,
/// per-4x4-block AC levels in scan order, reconstructed 8x8 residual).
///
/// Implements §8.5.11 (encoder side). Chroma block layout for 4:2:0:
/// 2x2 grid of 4x4 sub-blocks at (0,0), (4,0), (0,4), (4,4). The
/// decoder's inverse path uses `inverse_hadamard_chroma_dc_420` →
/// per-block `inverse_transform_4x4_dc_preserved` (with the 2x2-Hadamard
/// DC plugged in at `c[0,0]`). The forward path mirrors that.
///
/// `chroma_dc_indices` are the four spec-order DC slots:
///   0 → (0,0), 1 → (1,0), 2 → (0,1), 3 → (1,1)
/// in *block* coordinates (each step = 4 samples). That matches
/// `inverse_hadamard_chroma_dc_420`.
fn encode_chroma_residual(
    src_plane: &[u8],
    chroma_stride: usize,
    mb_x: usize,
    mb_y: usize,
    pred: &[i32; 64],
    qp_c: i32,
) -> ([i32; 4], [[i32; 16]; 4], [i32; 64]) {
    // Build sample residual against the predictor.
    let mut residual = [0i32; 64];
    for j in 0..8usize {
        for i in 0..8usize {
            let s = src_plane[(mb_y * 8 + j) * chroma_stride + mb_x * 8 + i] as i32;
            residual[j * 8 + i] = s - pred[j * 8 + i];
        }
    }

    // Forward 4x4 on each of the four 4x4 sub-blocks. Order matches the
    // inverse_hadamard_chroma_dc_420 input order (raster: TL, TR, BL, BR).
    let blocks_xy = [(0usize, 0usize), (4, 0), (0, 4), (4, 4)];
    let mut sub_coeffs = [[0i32; 16]; 4]; // forward-transformed 4x4 blocks
    for (k, &(bx, by)) in blocks_xy.iter().enumerate() {
        let mut block = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                block[j * 4 + i] = residual[(by + j) * 8 + bx + i];
            }
        }
        sub_coeffs[k] = forward_core_4x4(&block);
    }
    // Collect the 4 DCs.
    let dc_in: [i32; 4] = [
        sub_coeffs[0][0],
        sub_coeffs[1][0],
        sub_coeffs[2][0],
        sub_coeffs[3][0],
    ];
    // Forward 2x2 Hadamard.
    let dc_t = forward_hadamard_2x2(&dc_in);
    // Quantize the four DC levels.
    let dc_levels = quantize_chroma_dc(&dc_t, qp_c, true);

    // Quantize AC of each sub-block (positions 1..15 in scan order).
    // The encoder zeroes scan-position 0; the decoder will overwrite
    // with the inverse-Hadamard DC.
    let mut ac_levels = [[0i32; 16]; 4];
    for (k, coeffs) in sub_coeffs.iter().enumerate() {
        // Run flat AC quantization on positions 1..15. Position 0 is
        // already part of the DC path.
        let z = quantize_4x4_ac(coeffs, qp_c, true);
        // Pack into the encoder's "AC scan" layout: 15 entries at
        // positions 0..14, mirroring the decoder's
        // `inverse_scan_4x4_zigzag_ac` reading.
        ac_levels[k] = zigzag_scan_4x4_ac(&z);
    }

    // ------- Reconstruct the chroma 8x8 residual -------
    // §8.5.11: inverse Hadamard → per-block DC-preserved inverse.
    let dc_back = inverse_hadamard_chroma_dc_420(&dc_levels, qp_c, &FLAT_4X4_16, 8)
        .expect("chroma DC inverse");

    let mut recon_residual = [0i32; 64];
    for (k, &(bx, by)) in blocks_xy.iter().enumerate() {
        // Re-build a 4x4 coefficient block from the AC scan, then
        // overwrite c[0,0] with the inverse-Hadamard DC.
        let mut coeffs = inverse_scan_4x4_zigzag_ac(&ac_levels[k]);
        coeffs[0] = dc_back[k];
        let r = inverse_transform_4x4_dc_preserved(&coeffs, qp_c, &FLAT_4X4_16, 8)
            .expect("chroma 4x4 inverse");
        for j in 0..4 {
            for i in 0..4 {
                recon_residual[(by + j) * 8 + bx + i] = r[j * 4 + i];
            }
        }
    }

    (dc_levels, ac_levels, recon_residual)
}

/// Round-15 RDO trial for one Intra_16x16 luma mode.
///
/// Performs a *full* trial: forward 4x4 on each of the 16 sub-blocks,
/// pulls DCs into the Hadamard, quantises DC + AC, runs the inverse
/// path back through `inverse_hadamard_luma_dc_16x16` +
/// `inverse_transform_4x4_dc_preserved` to land on the same recon the
/// decoder will produce, then computes:
///
///   * `D` = SSD between source 16x16 luma and the reconstructed 16x16.
///   * `R` = trial CAVLC bit count for the DC block + 16 AC blocks
///     (only when AC has any non-zero level — matches `cbp_luma == 15`
///     gating in the real emit).
///
/// Returns `J = D + λ·R`. Lower = better.
///
/// The chroma path is identical for all luma modes so it is excluded
/// from `J` (its contribution would just be a constant offset).
#[allow(clippy::too_many_arguments)]
fn trial_intra16x16_cost(
    frame: &YuvFrame<'_>,
    pred: &[i32; 256],
    mb_x: usize,
    mb_y: usize,
    qp_y: i32,
    width: usize,
    lambda: f64,
) -> u64 {
    // Build the 16x16 sample residual.
    let mut residual = [0i32; 256];
    for j in 0..16usize {
        for i in 0..16usize {
            let s = frame.y[(mb_y * 16 + j) * width + (mb_x * 16 + i)] as i32;
            residual[j * 16 + i] = s - pred[j * 16 + i];
        }
    }
    // Forward 4x4 on each sub-block.
    let mut coeffs_4x4 = [[0i32; 16]; 16];
    for (blk, slot) in coeffs_4x4.iter_mut().enumerate() {
        let bx = blk % 4;
        let by = blk / 4;
        let mut block = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                block[j * 4 + i] = residual[(by * 4 + j) * 16 + bx * 4 + i];
            }
        }
        *slot = forward_core_4x4(&block);
    }
    let mut dc_coeffs = [0i32; 16];
    for (blk, c) in coeffs_4x4.iter().enumerate() {
        let bx = blk % 4;
        let by = blk / 4;
        dc_coeffs[by * 4 + bx] = c[0];
    }
    let dc_t = forward_hadamard_4x4(&dc_coeffs);
    let dc_levels = quantize_luma_dc(&dc_t, qp_y, true);

    let mut ac_quant_raster = [[0i32; 16]; 16];
    let mut ac_scan = [[0i32; 16]; 16];
    let mut any_ac_nz = false;
    for blkz in 0..16usize {
        let (bx, by) = LUMA_4X4_BLK[blkz];
        let raster = by * 4 + bx;
        let z = quantize_4x4_ac(&coeffs_4x4[raster], qp_y, true);
        ac_quant_raster[raster] = z;
        ac_scan[blkz] = zigzag_scan_4x4_ac(&z);
        if ac_scan[blkz].iter().any(|&v| v != 0) {
            any_ac_nz = true;
        }
    }

    // Inverse path: build the recon planes for SSD measurement.
    let inv_dc =
        inverse_hadamard_luma_dc_16x16(&dc_levels, qp_y, &FLAT_4X4_16, 8).expect("luma DC inverse");
    let mut recon = [0i32; 256];
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
        let r =
            inverse_transform_4x4_dc_preserved(&coeffs_in, qp_y, &FLAT_4X4_16, 8).expect("inv 4x4");
        for j in 0..4 {
            for i in 0..4 {
                let p = pred[(by * 4 + j) * 16 + bx * 4 + i];
                recon[(by * 4 + j) * 16 + bx * 4 + i] = p + r[j * 4 + i];
            }
        }
    }
    let d = ssd_16x16(frame.y, width, mb_x * 16, mb_y * 16, &recon);

    // Trial CAVLC for the rate term. nC=0 estimate (same rationale as
    // the I_NxN per-block trial above). The DC block is always
    // transmitted; AC blocks are transmitted only when cbp_luma == 15.
    let mut bits: u64 = 0;
    // mb_type ue(v): for Intra_16x16 the codeNum is 1..=24 (4-bit field
    // approximately). We don't try to enumerate it here; the choice
    // between cbp_luma 0/15 and the 4 pred_modes shifts the codeNum by
    // a small constant. Use a fixed estimate of 7 bits for Intra_16x16
    // mb_type — accurate to within ±2 bits across the (pred_mode, cbp)
    // space (Table 7-11 entries 1..24 → ue(v) widths 1, 3, 5, 7, 9 bits
    // depending on codeNum). The constant cancels out in *between-mode*
    // comparisons but matters for the I_16x16 vs I_NxN comparison; we
    // keep it as a stable midpoint.
    bits += 7;
    // intra_chroma_pred_mode ue(v): also a constant offset (we'll pick
    // chroma later, identical for all luma modes). Skip in J.
    // mb_qp_delta se(0) = 1 bit.
    bits += 1;

    // Luma DC block.
    let dc_scan = zigzag_scan_4x4(&dc_levels);
    let mut trial_w = BitWriter::new();
    if encode_residual_block_cavlc(&mut trial_w, CoeffTokenContext::Numeric(0), 16, &dc_scan)
        .is_ok()
    {
        bits += trial_w.bits_emitted() as u64;
    }
    // Luma AC blocks (only when cbp_luma == 15).
    if any_ac_nz {
        for scan in ac_scan.iter() {
            let mut tw = BitWriter::new();
            if encode_residual_block_cavlc(&mut tw, CoeffTokenContext::Numeric(0), 15, &scan[..15])
                .is_ok()
            {
                bits += tw.bits_emitted() as u64;
            }
        }
    }
    cost_combined(d, bits, lambda)
}

// ===========================================================================
// Round-16 — P-slice (inter) encoder.
// ===========================================================================

impl Encoder {
    /// Round-16 — encode one P-frame using `prev` as the reference. Emits
    /// just the slice NAL (caller has already emitted SPS/PPS via the IDR).
    ///
    /// Strategy per MB:
    ///   1. Run integer-pel motion estimation around (0, 0) ±range.
    ///   2. Build the inter predictor (luma 6-tap reduces to integer copy
    ///      for our integer-pel MVs; chroma bilinear with 0..=4 frac).
    ///   3. Compute the residual against the predictor, run forward 4x4 +
    ///      AC quantization (no luma DC Hadamard for inter MBs).
    ///   4. Decide P_Skip vs P_L0_16x16: P_Skip when chosen MV equals
    ///      the §8.4.1.2-derived skip-MV AND CBP would be 0.
    ///   5. (Round-16 simplification) No intra-fallback RDO yet — we trust
    ///      the inter path. Future round can add it.
    ///
    /// The bitstream emits an `mb_skip_run` ue(v) before each coded MB
    /// (per §7.3.4 CAVLC), counting the consecutive P_Skip MBs.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_p(
        &self,
        frame: &YuvFrame<'_>,
        prev: &EncodedFrameRef<'_>,
        frame_num: u32,
        pic_order_cnt_lsb: u32,
    ) -> EncodedP {
        assert_eq!(frame.width, self.cfg.width);
        assert_eq!(frame.height, self.cfg.height);
        assert_eq!(prev.width, self.cfg.width);
        assert_eq!(prev.height, self.cfg.height);
        let width_mbs = self.cfg.width / 16;
        let height_mbs = self.cfg.height / 16;
        let chroma_width = (self.cfg.width / 2) as usize;
        let chroma_height = (self.cfg.height / 2) as usize;

        // SPS/PPS are not re-emitted for non-IDR slices. Caller already
        // shipped them with the IDR.
        let mut stream: Vec<u8> = Vec::new();

        let log2_max_frame_num_minus4: u32 = 4;
        let log2_max_poc_lsb_minus4: u32 = 4;
        let chroma_qp_index_offset: i32 = 0;

        let mut sw = BitWriter::new();
        write_p_slice_header(
            &mut sw,
            &PSliceHeaderConfig {
                first_mb_in_slice: 0,
                slice_type_raw: 5, // P, all-same-type
                pic_parameter_set_id: 0,
                frame_num,
                frame_num_bits: log2_max_frame_num_minus4 + 4,
                pic_order_cnt_lsb,
                poc_lsb_bits: log2_max_poc_lsb_minus4 + 4,
                slice_qp_delta: 0,
                disable_deblocking_filter_idc: 0,
                slice_alpha_c0_offset_div2: 0,
                slice_beta_offset_div2: 0,
                nal_ref_idc: 2,
                cabac: None,
            },
        );

        // Reconstruction state for THIS picture.
        let mut recon_y = vec![0u8; (self.cfg.width * self.cfg.height) as usize];
        let mut recon_u = vec![0u8; chroma_width * chroma_height];
        let mut recon_v = vec![0u8; chroma_width * chroma_height];

        let qp_y = self.cfg.qp;
        // §8.5.8 BD-aware chroma-QP — see encode_idr; identical reasoning.
        let qp_bd_offset_c = qp_bd_offset(self.cfg.bit_depth_chroma_minus8);
        let qp_c = qp_y_to_qp_c_with_bd_offset(qp_y, chroma_qp_index_offset, qp_bd_offset_c);

        let mut nc_grid = CavlcNcGrid::new(width_mbs, height_mbs);
        let mut intra_grid = IntraGrid::new(width_mbs as usize, height_mbs as usize);
        let mut mv_grid = MvGrid::new(width_mbs as usize, height_mbs as usize);

        let mut mb_deblock_infos: Vec<MbDeblockInfo> =
            vec![MbDeblockInfo::default(); (width_mbs * height_mbs) as usize];

        // §7.3.4 CAVLC P-slice walker — emit `mb_skip_run` before each
        // coded MB. We accumulate skips and flush them just before writing
        // a coded MB (or at end-of-slice). Note: the spec also requires
        // an `mb_skip_run` even at the *very end* of a slice if the last
        // MBs were skips — we always emit one final `mb_skip_run` to
        // ensure consistency.
        let mut pending_skip: u32 = 0;

        for mb_y in 0..height_mbs as usize {
            for mb_x in 0..width_mbs as usize {
                let mb_addr = mb_y * width_mbs as usize + mb_x;

                // Round-29 — Decide P_Skip / P_L0_16x16 / P_8x8 for this
                // MB, with optional Intra_16x16 fallback when configured.
                let dbl = self.encode_p_mb_with_intra_fallback(
                    frame,
                    prev,
                    mb_x,
                    mb_y,
                    qp_y,
                    qp_c,
                    chroma_width,
                    chroma_height,
                    &mut recon_y,
                    &mut recon_u,
                    &mut recon_v,
                    &mut sw,
                    &mut nc_grid,
                    &mut intra_grid,
                    &mut mv_grid,
                    &mut pending_skip,
                );
                mb_deblock_infos[mb_addr] = dbl;
            }
        }

        // Flush any trailing skip run.
        if pending_skip > 0 {
            sw.ue(pending_skip);
            pending_skip = 0;
        }
        let _ = pending_skip;

        // §7.3.2.8 / §7.3.2.9 — slice_data trailing bits.
        sw.rbsp_trailing_bits();
        let slice_rbsp = sw.into_bytes();
        stream.extend_from_slice(&build_nal_unit(2, NalUnitType::SliceNonIdr, &slice_rbsp));

        // §8.7 — picture-level in-loop deblocking.
        deblock_recon(
            self.cfg.width,
            self.cfg.height,
            chroma_width as u32,
            chroma_height as u32,
            &mut recon_y,
            &mut recon_u,
            &mut recon_v,
            &mb_deblock_infos,
            chroma_qp_index_offset,
            width_mbs,
            height_mbs,
        );

        // §8.4.1.2.1 / §8.4.1.2.2 — capture per-8x8 MV state so a later
        // B-slice can use this P-frame as the L1 anchor and probe the
        // colocated block's L0 MV/refIdx for colZeroFlag (step 7).
        let n_parts = (width_mbs as usize) * (height_mbs as usize) * 4;
        let mut partition_mvs = Vec::with_capacity(n_parts);
        for slot in &mv_grid.slots {
            for part in 0..4usize {
                // Both "not available" and "intra" map to (mv=0,
                // ref=-1, is_intra=true) per §8.4.1.2.1: an intra
                // colocated MB has refIdxCol=-1, and an unencoded slot
                // is treated identically to "intra" by the colocated
                // probe.
                let (mv, ref_idx, is_intra) = if !slot.available || slot.is_intra {
                    (Mv::ZERO, -1, true)
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
            recon_width: self.cfg.width,
            recon_height: self.cfg.height,
            frame_num,
            pic_order_cnt_lsb,
            partition_mvs,
        }
    }

    /// Encode one P-slice macroblock: run ME, build inter predictor,
    /// transmit residual (or skip if equivalent). Updates `mv_grid`,
    /// `nc_grid`, `intra_grid`, the recon planes, and `pending_skip`.
    /// Returns the deblock info for the §8.7 pass.
    ///
    /// Round-19 adds a **4MV (P_8x8 with all sub_mb_type = PL08x8)**
    /// trial: each 8x8 sub-block runs its own quarter-pel ME, and 4MV
    /// wins when its sum-of-SADs beats the 16x16 SAD by more than the
    /// 4MV bit-cost overhead. On smooth content the two paths converge
    /// to the same MV → identical SAD → 1MV wins (no regression).
    #[allow(clippy::too_many_arguments)]
    fn encode_p_mb(
        &self,
        frame: &YuvFrame<'_>,
        prev: &EncodedFrameRef<'_>,
        mb_x: usize,
        mb_y: usize,
        qp_y: i32,
        qp_c: i32,
        chroma_width: usize,
        chroma_height: usize,
        recon_y: &mut [u8],
        recon_u: &mut [u8],
        recon_v: &mut [u8],
        sw: &mut BitWriter,
        nc_grid: &mut CavlcNcGrid,
        intra_grid: &mut IntraGrid,
        mv_grid: &mut MvGrid,
        pending_skip: &mut u32,
    ) -> MbDeblockInfo {
        let width = self.cfg.width as usize;
        let width_mbs = (self.cfg.width / 16) as usize;
        let mb_addr = mb_y * width_mbs + mb_x;

        // 1. Motion estimation: integer-pel full search ±16 followed by
        //    §8.4.2.2.1 half-pel and then quarter-pel refinement against
        //    the 6-tap interpolated predictor. Encoder and decoder agree
        //    on the predictor for sub-pel MVs because the refinement
        //    uses the same `interpolate_luma` the decoder calls.
        let me = search_quarter_pel_16x16(
            frame.y,
            width,
            self.cfg.width,
            self.cfg.height,
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
        let me_16x16_sad = me.sad;

        // 2. Compute MVpred + skip-MV for this MB position.
        let mvp = mvp_for_16x16(mv_grid, mb_x, mb_y, 0);
        let (skip_ref, skip_mv) = p_skip_mv(mv_grid, mb_x, mb_y);

        // 3. Build the inter predictor against the chosen MV.
        let pred_y =
            build_inter_pred_luma(prev.recon_y, prev.width, prev.height, mb_x, mb_y, chosen_mv);
        let chroma_w = prev.width / 2;
        let chroma_h = prev.height / 2;
        let pred_u =
            build_inter_pred_chroma(prev.recon_u, chroma_w, chroma_h, mb_x, mb_y, chosen_mv);
        let pred_v =
            build_inter_pred_chroma(prev.recon_v, chroma_w, chroma_h, mb_x, mb_y, chosen_mv);

        // ----- Round-19: 4MV (P_8x8) ME trial -----
        //
        // Run a per-8x8 quarter-pel ME. Each sub-block carries its own
        // MV; the sum of SADs lower-bounds the 4MV path's distortion.
        // 4MV is preferred over 1MV when the savings exceed the 4MV
        // bit-cost overhead (3 extra mb_type bits + 4× sub_mb_type "0"
        // = 4 extra bits + 6 extra mvd se(v) values vs 2 in 1MV path,
        // i.e. ~4 + 4·avg_mvd_se(v) − 2·avg_mvd_se(v) bits on top of
        // the residual-rate delta).
        let me_4mv: [crate::encoder::me::MeResult; 4] = std::array::from_fn(|i| {
            let sub_x = i % 2;
            let sub_y = i / 2;
            search_quarter_pel_8x8(
                frame.y,
                width,
                self.cfg.width,
                self.cfg.height,
                prev.recon_y,
                prev.width as usize,
                prev.width,
                prev.height,
                mb_x,
                mb_y,
                sub_x,
                sub_y,
                16,
                16,
            )
        });
        let me_4mv_sad_sum: u32 = me_4mv.iter().map(|m| m.sad).sum();

        // Heuristic: pick 4MV when the per-8x8 SAD sum beats the 16x16
        // SAD by a CLEAR margin AND the 1MV residual is large enough
        // that fragmenting the predictor across sub-blocks is
        // worthwhile. Two gates:
        //
        //   (a) Absolute floor: 1MV SAD ≥ 1024 (avg ≥ 4/pixel over the
        //       16×16 = 256 samples). Below that, the residual is in
        //       quantisation-noise territory — splitting only adds
        //       syntax bits without fixing real motion error.
        //
        //   (b) Relative margin: 4MV SAD sum < 70% of 1MV SAD. This is
        //       large enough that the syntax overhead (3 extra mvd
        //       se(v) values + mb_type + 4× sub_mb_type) is amortised
        //       even at low QP.
        //
        // On smooth or lightly-quantised content the two paths converge
        // (often identical SADs) → 4MV is never picked → no regression
        // on the round 16/17/18 PSNR fixtures.
        let try_4mv =
            me_16x16_sad >= 1024 && (me_4mv_sad_sum as u64) * 10 < (me_16x16_sad as u64) * 7;

        if try_4mv {
            return self.encode_p_mb_4mv(
                frame,
                prev,
                mb_x,
                mb_y,
                qp_y,
                qp_c,
                chroma_width,
                chroma_height,
                recon_y,
                recon_u,
                recon_v,
                sw,
                nc_grid,
                intra_grid,
                mv_grid,
                pending_skip,
                me_4mv,
            );
        }

        // 4. Compute the luma residual, transform, AC-quantize, inverse,
        //    and reconstruct.
        let inter_luma = forward_inter_luma(frame.y, width, mb_x, mb_y, &pred_y, qp_y);
        let luma_4x4_levels_scan = inter_luma.levels_scan;
        let luma_4x4_quant_raster = inter_luma.quant_raster;
        let blk_has_nz = inter_luma.blk_has_nz;
        let recon_block_residual = inter_luma.recon_residual;

        // 5. Decide cbp_luma (per 8x8 quadrant).
        let mut cbp_luma: u8 = 0;
        for blk8 in 0..4usize {
            let any_nz = (0..4).any(|sub| blk_has_nz[blk8 * 4 + sub]);
            if any_nz {
                cbp_luma |= 1u8 << blk8;
            }
        }

        // 6. Encode chroma residual against the inter predictor.
        let (u_dc_levels, u_ac_levels, u_recon_residual) =
            encode_chroma_residual_inter(frame.u, chroma_width, mb_x, mb_y, &pred_u, qp_c);
        let (v_dc_levels, v_ac_levels, v_recon_residual) =
            encode_chroma_residual_inter(frame.v, chroma_width, mb_x, mb_y, &pred_v, qp_c);
        let any_dc_nz = u_dc_levels.iter().any(|&v| v != 0) || v_dc_levels.iter().any(|&v| v != 0);
        let any_ac_nz = u_ac_levels.iter().any(|blk| blk.iter().any(|&v| v != 0))
            || v_ac_levels.iter().any(|blk| blk.iter().any(|&v| v != 0));
        let cbp_chroma: u8 = if any_ac_nz {
            2
        } else if any_dc_nz {
            1
        } else {
            0
        };

        // 7. P_Skip eligibility:
        //    * chosen MV must equal the §8.4.1.2 skip-MV
        //    * cbp_luma == 0 AND cbp_chroma == 0
        //    * skip_ref_idx == 0 (always 0 for our single-ref slice)
        let _ = skip_ref;
        let is_skip = chosen_mv == skip_mv && cbp_luma == 0 && cbp_chroma == 0;

        if is_skip {
            // P_Skip: do NOT emit any MB syntax; just bump pending_skip
            // and update grid state for neighbour lookups.
            *pending_skip += 1;

            // Update mv_grid: P_Skip MB carries its derived (skip_mv, ref=0)
            // — the same MV applies to all four 8x8 partitions.
            *mv_grid.slot_mut(mb_x, mb_y) = MvGridSlot {
                available: true,
                is_intra: false,
                ref_idx_l0_8x8: [0; 4],
                mv_l0_8x8: [skip_mv; 4],
            };

            // Update intra_grid: P_Skip is not intra, not I_NxN.
            let s = intra_grid.slot_mut(mb_x, mb_y);
            s.available = true;
            s.is_i_nxn = false;

            // Update nc_grid: P_Skip neighbour contributes nN=0 (§9.2.1.1
            // step 6).
            {
                let cur = &mut nc_grid.mbs[mb_addr];
                cur.is_available = true;
                cur.is_intra = false;
                cur.is_skip = true;
                cur.is_i_pcm = false;
                cur.luma_total_coeff = [0u8; 16];
                cur.cb_total_coeff = [0u8; 8];
                cur.cr_total_coeff = [0u8; 8];
            }

            // Apply the inter predictor to the recon (no residual).
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
                    recon_u[cy * chroma_width + cx] = pred_u[j * 8 + i].clamp(0, 255) as u8;
                    recon_v[cy * chroma_width + cx] = pred_v[j * 8 + i].clamp(0, 255) as u8;
                }
            }
            let _ = chroma_height;

            // Populate per-4x4 MV + per-8x8 ref_idx for the deblock
            // walker's different_ref_or_mv_luma. P_Skip uses skip_mv +
            // ref=0; ref_poc=0 (POC of the single IDR ref).
            let mv_t = (
                skip_mv.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                skip_mv.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            );
            return MbDeblockInfo {
                is_intra: false,
                qp_y,
                luma_nonzero_4x4: 0,
                chroma_nonzero_4x4: 0,
                mv_l0: [mv_t; 16],
                ref_idx_l0: [0; 4],
                ref_poc_l0: [0; 4],
                ..Default::default()
            };
        }

        // P_L0_16x16: flush any pending skip-run, then emit the coded MB.
        sw.ue(*pending_skip);
        *pending_skip = 0;

        // mvd = chosen_mv - mvp.
        let mvd_x = chosen_mv.x - mvp.x;
        let mvd_y = chosen_mv.y - mvp.y;

        // Apply luma reconstruction now that cbp_luma is known.
        // For 8x8 quadrants whose cbp bit is 0, the decoder reconstructs
        // pred + 0 (no residual). Mirror that here.
        for blk8 in 0..4usize {
            let send = (cbp_luma >> blk8) & 1 == 1;
            for sub in 0..4usize {
                let blk = blk8 * 4 + sub;
                let (bx, by) = LUMA_4X4_BLK[blk];
                let bx_pix = bx * 4;
                let by_pix = by * 4;
                for j in 0..4usize {
                    for i in 0..4usize {
                        let p = pred_y[(by_pix + j) * 16 + bx_pix + i];
                        let r = if send {
                            recon_block_residual[blk][j * 4 + i]
                        } else {
                            0
                        };
                        let py = mb_y * 16 + by_pix + j;
                        let px = mb_x * 16 + bx_pix + i;
                        recon_y[py * width + px] = (p + r).clamp(0, 255) as u8;
                    }
                }
            }
        }
        for j in 0..8usize {
            for i in 0..8usize {
                let cy = mb_y * 8 + j;
                let cx = mb_x * 8 + i;
                let pu = pred_u[j * 8 + i];
                let pv = pred_v[j * 8 + i];
                let res_u = if cbp_chroma == 0 {
                    0
                } else {
                    u_recon_residual[j * 8 + i]
                };
                let res_v = if cbp_chroma == 0 {
                    0
                } else {
                    v_recon_residual[j * 8 + i]
                };
                recon_u[cy * chroma_width + cx] = (pu + res_u).clamp(0, 255) as u8;
                recon_v[cy * chroma_width + cx] = (pv + res_v).clamp(0, 255) as u8;
            }
        }
        let _ = chroma_height;

        // CAVLC neighbour-grid update (per §9.2.1.1).
        {
            let cur = &mut nc_grid.mbs[mb_addr];
            cur.is_available = true;
            cur.is_intra = false;
            cur.is_skip = false;
            cur.is_i_pcm = false;
            cur.luma_total_coeff = [0u8; 16];
        }

        let mut luma_4x4_nc = [0i32; 16];
        let mut own_totals = [0u8; 16];
        for blk in 0..16usize {
            let blk8 = blk / 4;
            nc_grid.mbs[mb_addr].luma_total_coeff = own_totals;
            let nc = derive_nc_luma(
                nc_grid,
                mb_addr as u32,
                blk as u8,
                LumaNcKind::Ac,
                true,
                false,
            );
            luma_4x4_nc[blk] = nc;
            if (cbp_luma >> blk8) & 1 == 1 {
                let tc = luma_4x4_levels_scan[blk]
                    .iter()
                    .filter(|&&v| v != 0)
                    .count() as u8;
                own_totals[blk] = tc;
            } else {
                own_totals[blk] = 0;
            }
        }
        nc_grid.mbs[mb_addr].luma_total_coeff = own_totals;

        // Emit the MB syntax.
        let mb_cfg = PL016x16McbConfig {
            mvd_l0_x: mvd_x,
            mvd_l0_y: mvd_y,
            cbp_luma,
            cbp_chroma,
            mb_qp_delta: 0,
            luma_4x4_levels: luma_4x4_levels_scan,
            luma_4x4_nc,
            chroma_dc_cb: u_dc_levels,
            chroma_dc_cr: v_dc_levels,
            chroma_ac_cb: u_ac_levels,
            chroma_ac_cr: v_ac_levels,
        };
        let _ = luma_4x4_quant_raster; // already folded into recon_block_residual
        write_p_l0_16x16_mb(sw, &mb_cfg, 0).expect("write P_L0_16x16 mb");

        // Update mv_grid + intra_grid for subsequent MBs. P_L0_16x16
        // → all four 8x8 partitions carry the same MV / ref_idx.
        *mv_grid.slot_mut(mb_x, mb_y) = MvGridSlot {
            available: true,
            is_intra: false,
            ref_idx_l0_8x8: [0; 4],
            mv_l0_8x8: [chosen_mv; 4],
        };
        let s = intra_grid.slot_mut(mb_x, mb_y);
        s.available = true;
        s.is_i_nxn = false;

        // Per-4x4 nonzero mask for the §8.7 deblock walker.
        let luma_nonzero_4x4 = luma_nz_mask_from_blocks(&blk_has_nz);
        let cb_nz: [bool; 4] =
            std::array::from_fn(|i| cbp_chroma == 2 && u_ac_levels[i].iter().any(|&v| v != 0));
        let cr_nz: [bool; 4] =
            std::array::from_fn(|i| cbp_chroma == 2 && v_ac_levels[i].iter().any(|&v| v != 0));

        let mv_t = (
            chosen_mv.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            chosen_mv.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
        );
        MbDeblockInfo {
            is_intra: false,
            qp_y,
            luma_nonzero_4x4,
            chroma_nonzero_4x4: chroma_nz_mask_from_blocks(&cb_nz, &cr_nz),
            mv_l0: [mv_t; 16],
            ref_idx_l0: [0; 4],
            ref_poc_l0: [0; 4],
            ..Default::default()
        }
    }

    /// Round-19 — encode one P-slice MB using **4MV** P_8x8 with all
    /// four sub_mb_type set to PL08x8. Mirrors `encode_p_mb`'s post-ME
    /// pipeline (residual + recon + grid update + emit) but with a
    /// per-8x8 luma + chroma predictor and the §8.4.1.3 MVpred derived
    /// **per partition** (with within-MB neighbour reads as the four
    /// sub-blocks decode in raster order 0→1→2→3).
    #[allow(clippy::too_many_arguments)]
    fn encode_p_mb_4mv(
        &self,
        frame: &YuvFrame<'_>,
        prev: &EncodedFrameRef<'_>,
        mb_x: usize,
        mb_y: usize,
        qp_y: i32,
        qp_c: i32,
        chroma_width: usize,
        chroma_height: usize,
        recon_y: &mut [u8],
        recon_u: &mut [u8],
        recon_v: &mut [u8],
        sw: &mut BitWriter,
        nc_grid: &mut CavlcNcGrid,
        intra_grid: &mut IntraGrid,
        mv_grid: &mut MvGrid,
        pending_skip: &mut u32,
        me_4mv: [crate::encoder::me::MeResult; 4],
    ) -> MbDeblockInfo {
        let width = self.cfg.width as usize;
        let width_mbs = (self.cfg.width / 16) as usize;
        let mb_addr = mb_y * width_mbs + mb_x;
        let chroma_w = prev.width / 2;
        let chroma_h = prev.height / 2;
        let _ = chroma_height;

        // 1. Per-8x8 chosen MVs (in qpel) and per-partition mvds against
        //    the §8.4.1.3 within-MB-aware MVpred.
        let mvs: [Mv; 4] = std::array::from_fn(|i| Mv::new(me_4mv[i].mv_x, me_4mv[i].mv_y));
        let mut mvds: [(i32, i32); 4] = [(0, 0); 4];
        let mut inflight_avail = [false; 4];
        let mut mv_inflight = [Mv::ZERO; 4];
        let mut ref_inflight = [-1i32; 4];
        for sub_idx in 0..4usize {
            let mvp = mvp_for_p_8x8_partition(
                mv_grid,
                mb_x,
                mb_y,
                sub_idx,
                0,
                inflight_avail,
                mv_inflight,
                ref_inflight,
            );
            mvds[sub_idx] = (mvs[sub_idx].x - mvp.x, mvs[sub_idx].y - mvp.y);
            inflight_avail[sub_idx] = true;
            mv_inflight[sub_idx] = mvs[sub_idx];
            ref_inflight[sub_idx] = 0;
        }

        // 2. Build composite per-8x8 luma predictor.
        let mut pred_y = [0i32; 256];
        for (sub_idx, &mv) in mvs.iter().enumerate() {
            let sub_x = sub_idx % 2;
            let sub_y = sub_idx / 2;
            let p = build_inter_pred_luma_8x8(
                prev.recon_y,
                prev.width,
                prev.height,
                mb_x,
                mb_y,
                sub_x,
                sub_y,
                mv,
            );
            for j in 0..8usize {
                for i in 0..8usize {
                    pred_y[(sub_y * 8 + j) * 16 + sub_x * 8 + i] = p[j * 8 + i];
                }
            }
        }

        // 3. Build composite per-(4x4) chroma predictor (4:2:0).
        let mut pred_u = [0i32; 64];
        let mut pred_v = [0i32; 64];
        for (sub_idx, &mv) in mvs.iter().enumerate() {
            let sub_x = sub_idx % 2;
            let sub_y = sub_idx / 2;
            let pu = build_inter_pred_chroma_4x4(
                prev.recon_u,
                chroma_w,
                chroma_h,
                mb_x,
                mb_y,
                sub_x,
                sub_y,
                mv,
            );
            let pv = build_inter_pred_chroma_4x4(
                prev.recon_v,
                chroma_w,
                chroma_h,
                mb_x,
                mb_y,
                sub_x,
                sub_y,
                mv,
            );
            for j in 0..4usize {
                for i in 0..4usize {
                    pred_u[(sub_y * 4 + j) * 8 + sub_x * 4 + i] = pu[j * 4 + i];
                    pred_v[(sub_y * 4 + j) * 8 + sub_x * 4 + i] = pv[j * 4 + i];
                }
            }
        }

        // 4. Forward + quantize luma residual.
        let inter_luma = forward_inter_luma(frame.y, width, mb_x, mb_y, &pred_y, qp_y);
        let luma_4x4_levels_scan = inter_luma.levels_scan;
        let blk_has_nz = inter_luma.blk_has_nz;
        let recon_block_residual = inter_luma.recon_residual;

        let mut cbp_luma: u8 = 0;
        for blk8 in 0..4usize {
            let any_nz = (0..4).any(|sub| blk_has_nz[blk8 * 4 + sub]);
            if any_nz {
                cbp_luma |= 1u8 << blk8;
            }
        }

        // 5. Forward + quantize chroma residual.
        let (u_dc_levels, u_ac_levels, u_recon_residual) =
            encode_chroma_residual_inter(frame.u, chroma_width, mb_x, mb_y, &pred_u, qp_c);
        let (v_dc_levels, v_ac_levels, v_recon_residual) =
            encode_chroma_residual_inter(frame.v, chroma_width, mb_x, mb_y, &pred_v, qp_c);
        let any_dc_nz = u_dc_levels.iter().any(|&v| v != 0) || v_dc_levels.iter().any(|&v| v != 0);
        let any_ac_nz = u_ac_levels.iter().any(|blk| blk.iter().any(|&v| v != 0))
            || v_ac_levels.iter().any(|blk| blk.iter().any(|&v| v != 0));
        let cbp_chroma: u8 = if any_ac_nz {
            2
        } else if any_dc_nz {
            1
        } else {
            0
        };

        // P_8x8 cannot be P_Skip (P_Skip is single-MV by definition);
        // always emit the MB. Flush pending skip-run first.
        sw.ue(*pending_skip);
        *pending_skip = 0;

        // 6. Apply luma reconstruction (pred + residual or pred-only on
        //    cbp_luma=0 quadrants).
        for blk8 in 0..4usize {
            let send = (cbp_luma >> blk8) & 1 == 1;
            for sub in 0..4usize {
                let blk = blk8 * 4 + sub;
                let (bx, by) = LUMA_4X4_BLK[blk];
                let bx_pix = bx * 4;
                let by_pix = by * 4;
                for j in 0..4usize {
                    for i in 0..4usize {
                        let p = pred_y[(by_pix + j) * 16 + bx_pix + i];
                        let r = if send {
                            recon_block_residual[blk][j * 4 + i]
                        } else {
                            0
                        };
                        let py = mb_y * 16 + by_pix + j;
                        let px = mb_x * 16 + bx_pix + i;
                        recon_y[py * width + px] = (p + r).clamp(0, 255) as u8;
                    }
                }
            }
        }
        for j in 0..8usize {
            for i in 0..8usize {
                let cy = mb_y * 8 + j;
                let cx = mb_x * 8 + i;
                let pu = pred_u[j * 8 + i];
                let pv = pred_v[j * 8 + i];
                let res_u = if cbp_chroma == 0 {
                    0
                } else {
                    u_recon_residual[j * 8 + i]
                };
                let res_v = if cbp_chroma == 0 {
                    0
                } else {
                    v_recon_residual[j * 8 + i]
                };
                recon_u[cy * chroma_width + cx] = (pu + res_u).clamp(0, 255) as u8;
                recon_v[cy * chroma_width + cx] = (pv + res_v).clamp(0, 255) as u8;
            }
        }

        // 7. CAVLC neighbour-grid update.
        {
            let cur = &mut nc_grid.mbs[mb_addr];
            cur.is_available = true;
            cur.is_intra = false;
            cur.is_skip = false;
            cur.is_i_pcm = false;
            cur.luma_total_coeff = [0u8; 16];
        }

        let mut luma_4x4_nc = [0i32; 16];
        let mut own_totals = [0u8; 16];
        for blk in 0..16usize {
            let blk8 = blk / 4;
            nc_grid.mbs[mb_addr].luma_total_coeff = own_totals;
            let nc = derive_nc_luma(
                nc_grid,
                mb_addr as u32,
                blk as u8,
                LumaNcKind::Ac,
                true,
                false,
            );
            luma_4x4_nc[blk] = nc;
            if (cbp_luma >> blk8) & 1 == 1 {
                let tc = luma_4x4_levels_scan[blk]
                    .iter()
                    .filter(|&&v| v != 0)
                    .count() as u8;
                own_totals[blk] = tc;
            } else {
                own_totals[blk] = 0;
            }
        }
        nc_grid.mbs[mb_addr].luma_total_coeff = own_totals;

        // 8. Emit P_8x8 syntax.
        let mb_cfg = P8x8AllPL08x8McbConfig {
            mvd_l0: mvds,
            cbp_luma,
            cbp_chroma,
            mb_qp_delta: 0,
            luma_4x4_levels: luma_4x4_levels_scan,
            luma_4x4_nc,
            chroma_dc_cb: u_dc_levels,
            chroma_dc_cr: v_dc_levels,
            chroma_ac_cb: u_ac_levels,
            chroma_ac_cr: v_ac_levels,
        };
        write_p_8x8_all_pl08x8_mb(sw, &mb_cfg, 0).expect("write P_8x8 mb");

        // 9. Update mv_grid + intra_grid for subsequent MBs.
        *mv_grid.slot_mut(mb_x, mb_y) = MvGridSlot {
            available: true,
            is_intra: false,
            ref_idx_l0_8x8: [0; 4],
            mv_l0_8x8: mvs,
        };
        let s = intra_grid.slot_mut(mb_x, mb_y);
        s.available = true;
        s.is_i_nxn = false;

        // 10. Per-4x4 nonzero mask + per-partition MVs for the §8.7
        //     deblock walker. Each 8x8 partition maps to four 4x4 blocks
        //     all sharing the same MV.
        let luma_nonzero_4x4 = luma_nz_mask_from_blocks(&blk_has_nz);
        let cb_nz: [bool; 4] =
            std::array::from_fn(|i| cbp_chroma == 2 && u_ac_levels[i].iter().any(|&v| v != 0));
        let cr_nz: [bool; 4] =
            std::array::from_fn(|i| cbp_chroma == 2 && v_ac_levels[i].iter().any(|&v| v != 0));

        let mut mv_l0 = [(0i16, 0i16); 16];
        for blk in 0..16usize {
            let (bx, by) = LUMA_4X4_BLK[blk];
            let sub_idx = 2 * (by / 2) + (bx / 2);
            let mv = mvs[sub_idx];
            mv_l0[blk] = (
                mv.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                mv.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            );
        }

        MbDeblockInfo {
            is_intra: false,
            qp_y,
            luma_nonzero_4x4,
            chroma_nonzero_4x4: chroma_nz_mask_from_blocks(&cb_nz, &cr_nz),
            mv_l0,
            ref_idx_l0: [0; 4],
            ref_poc_l0: [0; 4],
            ..Default::default()
        }
    }

    /// Round-29 — encode one P-slice MB with optional **Intra_16x16
    /// fallback** RDO (per §7.3.5 Table 7-13 mb_type values 6..=29).
    ///
    /// When [`EncoderConfig::intra_in_inter`] is true, snapshot the
    /// pre-state, run the inter pipeline (`encode_p_mb`), measure
    /// `J_inter = D_inter + λ·R_inter`, snapshot the post-state, restore
    /// pre-state, run an Intra_16x16 trial, measure `J_intra` the same
    /// way, and install the lower-J winner. When false, just call the
    /// inter pipeline directly.
    ///
    /// The intra trial reuses the already-reconstructed prior MBs from
    /// `recon_y` for §8.3.3 neighbour samples — same as the IDR path.
    /// On winner-is-intra the encoder skips emitting `mvd_l0` and
    /// updates `mv_grid` with `is_intra = true` so subsequent inter MBs'
    /// §8.4.1.3.2 step-2 collapse fires.
    #[allow(clippy::too_many_arguments)]
    fn encode_p_mb_with_intra_fallback(
        &self,
        frame: &YuvFrame<'_>,
        prev: &EncodedFrameRef<'_>,
        mb_x: usize,
        mb_y: usize,
        qp_y: i32,
        qp_c: i32,
        chroma_width: usize,
        chroma_height: usize,
        recon_y: &mut [u8],
        recon_u: &mut [u8],
        recon_v: &mut [u8],
        sw: &mut BitWriter,
        nc_grid: &mut CavlcNcGrid,
        intra_grid: &mut IntraGrid,
        mv_grid: &mut MvGrid,
        pending_skip: &mut u32,
    ) -> MbDeblockInfo {
        if !self.cfg.intra_in_inter {
            return self.encode_p_mb(
                frame,
                prev,
                mb_x,
                mb_y,
                qp_y,
                qp_c,
                chroma_width,
                chroma_height,
                recon_y,
                recon_u,
                recon_v,
                sw,
                nc_grid,
                intra_grid,
                mv_grid,
                pending_skip,
            );
        }

        let width = self.cfg.width as usize;
        let width_mbs = (self.cfg.width / 16) as usize;
        let mb_addr = mb_y * width_mbs + mb_x;

        // -- Snapshot pre-state. --
        let snap_pre = InterMbStateSnapshot::capture_p(
            recon_y,
            recon_u,
            recon_v,
            width,
            chroma_width,
            mb_x,
            mb_y,
            sw,
            nc_grid,
            intra_grid,
            mv_grid,
            mb_addr,
            width_mbs,
            *pending_skip,
        );
        let pre_bits = sw.bits_emitted();

        // -- Trial 1: inter (P_Skip / P_L0_16x16 / P_8x8). --
        let inter_dbl = self.encode_p_mb(
            frame,
            prev,
            mb_x,
            mb_y,
            qp_y,
            qp_c,
            chroma_width,
            chroma_height,
            recon_y,
            recon_u,
            recon_v,
            sw,
            nc_grid,
            intra_grid,
            mv_grid,
            pending_skip,
        );
        let inter_bits = sw.bits_emitted() - pre_bits;
        let inter_d = ssd_16x16(frame.y, width, mb_x * 16, mb_y * 16, &{
            let mut buf = [0i32; 256];
            for j in 0..16usize {
                for i in 0..16usize {
                    buf[j * 16 + i] = recon_y[(mb_y * 16 + j) * width + (mb_x * 16 + i)] as i32;
                }
            }
            buf
        });
        let lambda = lambda_ssd(qp_y, false);
        let j_inter = cost_combined(inter_d, inter_bits as u64, lambda);

        // Capture post-inter state.
        let snap_inter = InterMbStateSnapshot::capture_p(
            recon_y,
            recon_u,
            recon_v,
            width,
            chroma_width,
            mb_x,
            mb_y,
            sw,
            nc_grid,
            intra_grid,
            mv_grid,
            mb_addr,
            width_mbs,
            *pending_skip,
        );

        // -- Restore pre-state for the intra trial. --
        snap_pre.restore_p(
            recon_y,
            recon_u,
            recon_v,
            width,
            chroma_width,
            mb_x,
            mb_y,
            sw,
            nc_grid,
            intra_grid,
            mv_grid,
            mb_addr,
            pending_skip,
        );

        // -- Trial 2: Intra_16x16 in P-slice. --
        let intra_dbl = self.encode_p_mb_intra16x16(
            frame,
            mb_x,
            mb_y,
            qp_y,
            qp_c,
            chroma_width,
            chroma_height,
            recon_y,
            recon_u,
            recon_v,
            sw,
            nc_grid,
            intra_grid,
            mv_grid,
            pending_skip,
        );
        let intra_bits = sw.bits_emitted() - pre_bits;
        let intra_d = ssd_16x16(frame.y, width, mb_x * 16, mb_y * 16, &{
            let mut buf = [0i32; 256];
            for j in 0..16usize {
                for i in 0..16usize {
                    buf[j * 16 + i] = recon_y[(mb_y * 16 + j) * width + (mb_x * 16 + i)] as i32;
                }
            }
            buf
        });
        let j_intra = cost_combined(intra_d, intra_bits as u64, lambda);

        // -- Pick winner. --
        if j_inter <= j_intra {
            // Roll back the intra trial; install the inter results.
            snap_pre.restore_p(
                recon_y,
                recon_u,
                recon_v,
                width,
                chroma_width,
                mb_x,
                mb_y,
                sw,
                nc_grid,
                intra_grid,
                mv_grid,
                mb_addr,
                pending_skip,
            );
            snap_inter.restore_p(
                recon_y,
                recon_u,
                recon_v,
                width,
                chroma_width,
                mb_x,
                mb_y,
                sw,
                nc_grid,
                intra_grid,
                mv_grid,
                mb_addr,
                pending_skip,
            );
            inter_dbl
        } else {
            intra_dbl
        }
    }

    /// Round-29 — Intra_16x16 fallback for a P-slice MB. Picks the best
    /// §8.3.3 mode by SAD against the reconstructed neighbour samples
    /// (top/left of the current frame's already-encoded MBs), forward
    /// 4x4 + Hadamard + quantize, encodes chroma, and emits the MB syntax
    /// via [`write_intra16x16_mb_in_inter_slice`] with `mb_type_offset =
    /// 5` (Table 7-13). Updates `mv_grid` with `is_intra = true` so
    /// subsequent inter MBs' §8.4.1.3.2 step-2 collapses.
    ///
    /// Round-29 scope: 4:2:0 only (encode_p / encode_b are 4:2:0-only).
    #[allow(clippy::too_many_arguments)]
    fn encode_p_mb_intra16x16(
        &self,
        frame: &YuvFrame<'_>,
        mb_x: usize,
        mb_y: usize,
        qp_y: i32,
        qp_c: i32,
        chroma_width: usize,
        chroma_height: usize,
        recon_y: &mut [u8],
        recon_u: &mut [u8],
        recon_v: &mut [u8],
        sw: &mut BitWriter,
        nc_grid: &mut CavlcNcGrid,
        intra_grid: &mut IntraGrid,
        mv_grid: &mut MvGrid,
        pending_skip: &mut u32,
    ) -> MbDeblockInfo {
        // Flush any pending P_Skip run before emitting a coded MB.
        sw.ue(*pending_skip);
        *pending_skip = 0;

        // Delegate the heavy lifting to a writer that targets the
        // P-slice mb_type table (offset = 5).
        self.encode_intra16x16_in_inter_slice(
            frame,
            mb_x,
            mb_y,
            qp_y,
            qp_c,
            chroma_width,
            chroma_height,
            recon_y,
            recon_u,
            recon_v,
            sw,
            nc_grid,
            intra_grid,
            5,
        );

        // Update MV grid: intra MB contributes (mv=0, ref=-1) per
        // §8.4.1.3.2 step 2.
        *mv_grid.slot_mut(mb_x, mb_y) = MvGridSlot {
            available: true,
            is_intra: true,
            ref_idx_l0_8x8: [-1; 4],
            mv_l0_8x8: [Mv::ZERO; 4],
        };

        // §8.7.2.1 — per-4x4 deblock mask. For Intra_16x16, when
        // cbp_luma == 15 every 4x4 block carries AC; when cbp_luma == 0
        // only the DC plane is sent. We approximate by setting all 16
        // bits when any AC coefficient exists post-quantize, mirroring
        // the IDR Intra_16x16 path. Look up the actual bit from the
        // nc_grid we just populated.
        let width_mbs = (self.cfg.width / 16) as usize;
        let mb_addr = mb_y * width_mbs + mb_x;
        let any_nz = nc_grid.mbs[mb_addr]
            .luma_total_coeff
            .iter()
            .any(|&v| v != 0);
        let luma_nonzero_4x4: u16 = if any_nz {
            let mut nz = [false; 16];
            for (blkz, slot) in nz.iter_mut().enumerate() {
                *slot = nc_grid.mbs[mb_addr].luma_total_coeff[blkz] != 0;
            }
            luma_nz_mask_from_blocks(&nz)
        } else {
            0
        };

        MbDeblockInfo {
            is_intra: true,
            qp_y,
            luma_nonzero_4x4,
            chroma_nonzero_4x4: 0,
            ..Default::default()
        }
    }

    /// Round-29 — shared Intra_16x16 emitter used by both the P-slice
    /// (`mb_type_offset = 5`) and B-slice (`mb_type_offset = 23`)
    /// fallback paths. Picks the best §8.3.3 luma mode + best §8.3.4
    /// chroma mode by Lagrangian RDO (luma) / SAD (chroma), runs the
    /// forward + quantize + inverse pipeline, writes the MB syntax via
    /// [`write_intra16x16_mb_in_inter_slice`], and updates `recon_*`,
    /// `nc_grid`, `intra_grid`. Caller is responsible for `mv_grid`
    /// updates, `pending_skip` flush, and the deblock-info return.
    #[allow(clippy::too_many_arguments)]
    fn encode_intra16x16_in_inter_slice(
        &self,
        frame: &YuvFrame<'_>,
        mb_x: usize,
        mb_y: usize,
        qp_y: i32,
        qp_c: i32,
        chroma_width: usize,
        chroma_height: usize,
        recon_y: &mut [u8],
        recon_u: &mut [u8],
        recon_v: &mut [u8],
        sw: &mut BitWriter,
        nc_grid: &mut CavlcNcGrid,
        intra_grid: &mut IntraGrid,
        mb_type_offset: u32,
    ) {
        let width = self.cfg.width as usize;
        let height = self.cfg.height as usize;
        let lambda = lambda_ssd(qp_y, true);

        // Pick best luma mode by RDO (mirrors encode_mb_intra16x16's
        // luma-mode loop, with neighbour reads against `recon_y` which
        // has already been populated by prior MBs).
        let mut best_luma_mode = I16x16Mode::Dc;
        let mut best_luma_pred = [0i32; 256];
        let mut best_luma_cost = u64::MAX;
        for &mode in &[
            I16x16Mode::Vertical,
            I16x16Mode::Horizontal,
            I16x16Mode::Dc,
            I16x16Mode::Plane,
        ] {
            let Some(pred) = predict_16x16(mode, recon_y, width, height, mb_x, mb_y) else {
                continue;
            };
            let cost = trial_intra16x16_cost(frame, &pred, mb_x, mb_y, qp_y, width, lambda);
            if cost < best_luma_cost {
                best_luma_cost = cost;
                best_luma_mode = mode;
                best_luma_pred = pred;
            }
        }

        // Build luma residual against chosen predictor.
        let mut residual = [0i32; 256];
        for j in 0..16usize {
            for i in 0..16usize {
                let s = frame.y[(mb_y * 16 + j) * width + (mb_x * 16 + i)] as i32;
                residual[j * 16 + i] = s - best_luma_pred[j * 16 + i];
            }
        }
        let mut coeffs_4x4 = [[0i32; 16]; 16];
        for (blk, slot) in coeffs_4x4.iter_mut().enumerate() {
            let bx = blk % 4;
            let by = blk / 4;
            let mut block = [0i32; 16];
            for j in 0..4 {
                for i in 0..4 {
                    block[j * 4 + i] = residual[(by * 4 + j) * 16 + bx * 4 + i];
                }
            }
            *slot = forward_core_4x4(&block);
        }
        let mut dc_coeffs = [0i32; 16];
        for (blk, c) in coeffs_4x4.iter().enumerate() {
            let bx = blk % 4;
            let by = blk / 4;
            dc_coeffs[by * 4 + bx] = c[0];
        }
        let dc_t = forward_hadamard_4x4(&dc_coeffs);
        let dc_levels = quantize_luma_dc(&dc_t, qp_y, true);

        // Per-block AC quantize.
        let mut luma_ac_levels = [[0i32; 16]; 16];
        let mut luma_ac_quant_raster = [[0i32; 16]; 16];
        let mut any_luma_ac_nz = false;
        for blkz in 0..16usize {
            let (bx, by) = LUMA_4X4_BLK[blkz];
            let raster = by * 4 + bx;
            let z = quantize_4x4_ac(&coeffs_4x4[raster], qp_y, true);
            luma_ac_quant_raster[raster] = z;
            let scan_ac = zigzag_scan_4x4_ac(&z);
            luma_ac_levels[blkz] = scan_ac;
            if scan_ac.iter().any(|&v| v != 0) {
                any_luma_ac_nz = true;
            }
        }
        let cbp_luma: u8 = if any_luma_ac_nz { 15 } else { 0 };

        // Inverse path → recon_y.
        let inv_dc = inverse_hadamard_luma_dc_16x16(&dc_levels, qp_y, &FLAT_4X4_16, 8)
            .expect("luma DC inverse");
        #[allow(clippy::needless_range_loop)]
        for blk in 0..16usize {
            let bx = blk % 4;
            let by = blk / 4;
            let mut coeffs_in = if cbp_luma == 15 {
                luma_ac_quant_raster[blk]
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
                    let p = best_luma_pred[(by * 4 + j) * 16 + bx * 4 + i];
                    let raw = p + r[j * 4 + i];
                    recon_y[py * width + px] = raw.clamp(0, 255) as u8;
                }
            }
        }

        // Chroma path (4:2:0 only on this round).
        let chroma_block = encode_chroma_block(
            self.cfg.chroma_format_idc,
            frame.u,
            frame.v,
            recon_u,
            recon_v,
            chroma_width,
            chroma_height,
            mb_x,
            mb_y,
            qp_c,
        );
        let cbp_chroma = chroma_block.cbp_chroma;
        let best_chroma_mode_u8 = chroma_block.pred_mode.as_u8();
        install_chroma_recon(
            self.cfg.chroma_format_idc,
            chroma_width,
            mb_x,
            mb_y,
            &chroma_block,
            recon_u,
            recon_v,
        );
        let _ = chroma_height;

        let mut u_dc_levels = [0i32; 4];
        let mut v_dc_levels = [0i32; 4];
        let mut u_ac_levels = [[0i32; 16]; 4];
        let mut v_ac_levels = [[0i32; 16]; 4];
        u_dc_levels.copy_from_slice(&chroma_block.dc_cb[..4]);
        v_dc_levels.copy_from_slice(&chroma_block.dc_cr[..4]);
        u_ac_levels.copy_from_slice(&chroma_block.ac_cb[..4]);
        v_ac_levels.copy_from_slice(&chroma_block.ac_cr[..4]);

        // CAVLC neighbour-grid update for this MB (Intra16x16 → mark as
        // intra so subsequent MBs' nC derivation reads the correct
        // TotalCoeff). Done before per-block nC derivation so internal
        // neighbours within the MB read the freshly-zeroed totals.
        let width_mbs = (self.cfg.width / 16) as usize;
        let mb_addr = mb_y * width_mbs + mb_x;
        {
            let cur = &mut nc_grid.mbs[mb_addr];
            cur.is_available = true;
            cur.is_intra = true;
            cur.is_skip = false;
            cur.is_i_pcm = false;
            cur.luma_total_coeff = [0u8; 16];
        }
        let nc_dc = derive_nc_luma(
            nc_grid,
            mb_addr as u32,
            0,
            LumaNcKind::Intra16x16Dc,
            true,
            false,
        );
        let mut luma_ac_nc = [0i32; 16];
        let mut own_totals = [0u8; 16];
        for blkz in 0..16u8 {
            nc_grid.mbs[mb_addr].luma_total_coeff = own_totals;
            let nc = derive_nc_luma(nc_grid, mb_addr as u32, blkz, LumaNcKind::Ac, true, false);
            luma_ac_nc[blkz as usize] = nc;
            if cbp_luma == 15 {
                let tc = luma_ac_levels[blkz as usize]
                    .iter()
                    .filter(|&&v| v != 0)
                    .count() as u8;
                own_totals[blkz as usize] = tc;
            }
        }
        nc_grid.mbs[mb_addr].luma_total_coeff = own_totals;

        // Emit MB syntax.
        let mb_cfg = I16x16McbConfig {
            pred_mode: best_luma_mode.as_u8(),
            intra_chroma_pred_mode: best_chroma_mode_u8,
            cbp_luma,
            cbp_chroma,
            mb_qp_delta: 0,
            luma_dc_levels_raster: dc_levels,
            luma_ac_levels,
            luma_ac_nc,
            chroma_dc_cb: u_dc_levels,
            chroma_dc_cr: v_dc_levels,
            chroma_ac_cb: u_ac_levels,
            chroma_ac_cr: v_ac_levels,
        };
        write_intra16x16_mb_in_inter_slice(
            sw,
            &mb_cfg,
            mb_type_offset,
            CoeffTokenContext::Numeric(nc_dc),
        )
        .expect("write Intra_16x16 in inter slice");

        // Mark intra grid: available + not-i_nxn so subsequent I_NxN
        // neighbour lookups fall to DC (mode = 2).
        let s = intra_grid.slot_mut(mb_x, mb_y);
        s.available = true;
        s.is_i_nxn = false;

        // Suppress unused.
        let _ = best_luma_cost;
    }
}

/// Borrowed view of a previously-encoded frame's reconstruction. Used as
/// the reference for [`Encoder::encode_p`] and [`Encoder::encode_b`].
///
/// `partition_mvs` carries per-8x8 colocated MV / refIdx data when this
/// reference is being used as the **L1 anchor** of a B-slice (round-21
/// spatial direct mode needs it for §8.4.1.2.2 step 7 colZeroFlag).
/// For non-B-slice users (`encode_p`) the field is ignored. An empty
/// slice is treated as "all intra colocated" (colZeroFlag = 0).
///
/// `pic_order_cnt` (round-24) is the picture order count this reference
/// was assigned at encode time, needed by §8.4.1.2.3 temporal direct
/// derivation (the POCs of pic0 = L0[refIdxL0] and pic1 = L1[0] feed
/// the `tb` / `td` distance computation). For IDR derived from
/// `EncodedIdr` it defaults to 0 (POC type 0, IDR's
/// `pic_order_cnt_lsb` is forced to 0). For P derived from
/// `EncodedP` it is `2 * pic_order_cnt_lsb` matching the encoder's
/// frame-coding convention (`TopFieldOrderCnt = pic_order_cnt_lsb`,
/// frame POC = min(top, bottom) = top in our setup).
pub struct EncodedFrameRef<'a> {
    pub width: u32,
    pub height: u32,
    pub recon_y: &'a [u8],
    pub recon_u: &'a [u8],
    pub recon_v: &'a [u8],
    /// Per-8x8 partition MVs, length = `(width / 16) * (height / 16) * 4`,
    /// or empty when no MV data is available (IDR / treat as all-intra).
    pub partition_mvs: &'a [FrameRefPartitionMv],
    /// §8.2.1 picture order count of this reference. Used by §8.4.1.2.3
    /// temporal direct (round 24); ignored by spatial direct.
    pub pic_order_cnt: i32,
}

impl<'a> From<&'a EncodedIdr> for EncodedFrameRef<'a> {
    fn from(e: &'a EncodedIdr) -> Self {
        Self {
            width: e.recon_width,
            height: e.recon_height,
            recon_y: &e.recon_y,
            recon_u: &e.recon_u,
            recon_v: &e.recon_v,
            partition_mvs: &e.partition_mvs,
            pic_order_cnt: 0, // §8.2.1.1 / §7.4.3 — IDR's pic_order_cnt_lsb is 0.
        }
    }
}

impl<'a> From<&'a EncodedP> for EncodedFrameRef<'a> {
    fn from(e: &'a EncodedP) -> Self {
        Self {
            width: e.recon_width,
            height: e.recon_height,
            recon_y: &e.recon_y,
            recon_u: &e.recon_u,
            recon_v: &e.recon_v,
            partition_mvs: &e.partition_mvs,
            pic_order_cnt: e.pic_order_cnt_lsb as i32,
        }
    }
}

/// Forward-quantize-inverse output for one inter MB's luma residual.
///
/// * `levels_scan[blk]` — the 16 coefficients in zig-zag scan order for
///   CAVLC emit.
/// * `quant_raster[blk]` — the same coefficients in 4x4 raster (for
///   inverse path / debug).
/// * `blk_has_nz[blk]` — true iff the block has at least one non-zero
///   coefficient (drives cbp_luma quadrant decision).
/// * `recon_residual[blk]` — the reconstructed 16 sample residuals for
///   this 4x4 block (post-inverse), ready to be added to the predictor.
struct InterLumaForward {
    levels_scan: [[i32; 16]; 16],
    quant_raster: [[i32; 16]; 16],
    blk_has_nz: [bool; 16],
    recon_residual: [[i32; 16]; 16],
}

/// Forward 4x4 + AC quantization for an inter MB's luma residual. Inter
/// MBs do NOT use the Intra_16x16 luma DC Hadamard — every 4x4 block is
/// transmitted as a full DC+AC residual block.
fn forward_inter_luma(
    src_y: &[u8],
    src_stride: usize,
    mb_x: usize,
    mb_y: usize,
    pred: &[i32; 256],
    qp_y: i32,
) -> InterLumaForward {
    // Build sample residual against the predictor.
    let mut residual = [0i32; 256];
    for j in 0..16usize {
        for i in 0..16usize {
            let s = src_y[(mb_y * 16 + j) * src_stride + (mb_x * 16 + i)] as i32;
            residual[j * 16 + i] = s - pred[j * 16 + i];
        }
    }

    let mut levels_scan = [[0i32; 16]; 16];
    let mut quant_raster = [[0i32; 16]; 16];
    let mut blk_has_nz = [false; 16];
    let mut recon_residual = [[0i32; 16]; 16];

    for blkz in 0..16usize {
        let (bx, by) = LUMA_4X4_BLK[blkz];
        let mut block = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                block[j * 4 + i] = residual[(by * 4 + j) * 16 + bx * 4 + i];
            }
        }
        // Forward 4x4 + full quantize (includes DC; inter uses the
        // standard quantizer per §8.5.10).
        let w = forward_core_4x4(&block);
        let z = quantize_4x4(&w, qp_y, false);
        let scan = zigzag_scan_4x4(&z);
        levels_scan[blkz] = scan;
        quant_raster[blkz] = z;
        blk_has_nz[blkz] = z.iter().any(|&v| v != 0);

        // Inverse path so we know the recon's residual for the block.
        let r = inverse_transform_4x4(&z, qp_y, &FLAT_4X4_16, 8).expect("inv 4x4");
        recon_residual[blkz] = r;
    }
    InterLumaForward {
        levels_scan,
        quant_raster,
        blk_has_nz,
        recon_residual,
    }
}

// ---------------------------------------------------------------------------
// Round 20 — B-slice support (B_L0_16x16 / B_L1_16x16 / B_Bi_16x16).
// Round 21 — B_Skip / B_Direct_16x16 (spatial direct mode).
//
// Round-20 scope: explicit-inter only; one 16x16 partition per MB; no
// B_Skip / B_Direct; one reference per list (L0 = the IDR, L1 = the prior
// P-frame), `weighted_bipred_idc = 0` so bipred uses default
// `(L0+L1+1)>>1` per §8.4.2.3.1 eq. 8-273.
//
// Round-21 adds **spatial direct mode**:
//   * `B_Direct_16x16` (raw mb_type 0): no mvd_lN / ref_idx_lN in the
//     bitstream; decoder derives MVs per §8.4.1.2.2 from the encoder's
//     L0 / L1 MV grids + the colocated picture's L0 MV.
//   * `B_Skip` (signalled by `mb_skip_run`): same MV derivation as
//     B_Direct_16x16 but with cbp_luma = cbp_chroma = 0 and no MB syntax
//     at all — pure run-length skip.
//
// The encoder mirrors the decoder's §8.4.1.2.2 derivation locally and
// emits B_Direct / B_Skip when:
//   1. The derived predictor's SAD beats the round-20 explicit-inter
//      (B_L0 / B_L1 / B_Bi) candidates, AND
//   2. (For B_Skip) the residual quantises to all-zero (cbp == 0).
//
// `colZeroFlag` (§8.4.1.2.2 step 7) requires the colocated picture's
// per-block L0 MV. The L1 reference (a prior P-frame) supplies these
// via `EncodedFrameRef::partition_mvs`. For an IDR L1 (all-intra), the
// step-7 check returns colZeroFlag = 0 for every block.
//
// Spec sections cited:
//   §7.3.3                 — slice_header() (B-slice fields)
//   §7.3.4                 — slice_data() with mb_skip_run (B_Skip)
//   §7.3.5.1               — mb_pred() with mvd_l0 / mvd_l1
//   §7.4.5 Table 7-14      — B-slice mb_type ∈ {0..=3} mapping
//   §8.4.1.2               — direct-mode predicted MV derivation entry
//   §8.4.1.2.1             — colocated block selection (L0 vs L1 MV)
//   §8.4.1.2.2             — spatial direct MV derivation (5..7)
//   §8.4.1.3               — derivation of mvpLX (separate for L0 / L1)
//   §8.4.1.3.1             — median MV prediction
//   §8.4.1.3.2             — C→D substitution
//   §8.4.2.3.1 eq. 8-273   — default weighted bi-prediction average
//   §8.4.2.2.1             — luma 6-tap fractional sample interpolation
//   §8.4.1.4               — chroma MV derivation from luma (4:2:0)
//   §8.7.2.1 NOTE 2        — bipred edge bS derivation (drives deblock)
//   §A.2                   — profile_idc must be ≥ 77 for B-slices
// ---------------------------------------------------------------------------

/// §8.4.1.2.2 — encoder-side spatial direct derivation result.
///
/// Each MB's direct mode produces one (refIdxL0, refIdxL1, mvL0, mvL1)
/// per 8x8 sub-partition (when `direct_8x8_inference_flag = 1`). When all
/// four 8x8 partitions resolve to the same `(ref, ref, mv, mv)` we can
/// build a single 16x16 predictor; otherwise we'd need per-partition
/// motion compensation. Round-21 only emits B_Direct / B_Skip when all
/// four partitions agree (the common case under directZeroPredictionFlag
/// or when neighbours are zero-MV propagating).
#[derive(Debug, Clone, Copy)]
struct BDirectDerivation {
    ref_idx_l0: i32,
    ref_idx_l1: i32,
    /// Per-8x8 MV (4 entries) for L0. `Mv::ZERO` when `ref_idx_l0 < 0`.
    mv_l0_per_8x8: [Mv; 4],
    /// Per-8x8 MV (4 entries) for L1. `Mv::ZERO` when `ref_idx_l1 < 0`.
    mv_l1_per_8x8: [Mv; 4],
    /// True when `directZeroPredictionFlag` fired (§8.4.1.2.2 step 5).
    direct_zero: bool,
}

impl BDirectDerivation {
    /// True iff every 8x8 partition shares the same MV (and thus the
    /// 16x16 predictor is uniform across the MB). Round-21 only emits
    /// `B_Direct_16x16` / `B_Skip` for this case.
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

    /// Read the uniform per-MB L0 MV (only meaningful when `is_uniform()`).
    fn mv_l0(&self) -> Mv {
        self.mv_l0_per_8x8[0]
    }

    /// Read the uniform per-MB L1 MV (only meaningful when `is_uniform()`).
    fn mv_l1(&self) -> Mv {
        self.mv_l1_per_8x8[0]
    }
}

/// §8.4.1.2.2 step 7 — colZeroFlag check.
///
/// Returns true iff the colocated 8x8 partition (from L1's
/// `partition_mvs`) is short-term-coded with `refIdxCol == 0` and
/// `|mvCol| <= 1` in both components.
///
/// Per §8.4.1.2.1, when the colocated MB is intra mvCol=0 / refIdxCol=-1,
/// and colZeroFlag is forced to 0 (the `is_intra` branch). When
/// `partition_mvs` is empty (e.g. an IDR L1 anchor with no MV data
/// recorded), we treat all colocated blocks as intra → colZeroFlag = 0.
fn col_zero_flag(
    l1_partition_mvs: &[FrameRefPartitionMv],
    width_mbs: usize,
    mb_x: usize,
    mb_y: usize,
    part_8x8: usize,
) -> bool {
    let idx = (mb_y * width_mbs + mb_x) * 4 + part_8x8;
    let Some(slot) = l1_partition_mvs.get(idx) else {
        // No MV data → treat as intra → colZeroFlag = 0.
        return false;
    };
    if slot.is_intra {
        return false;
    }
    // §8.4.1.2.1 step 1.2 — picks L0's MV by default when refIdxL0Col >= 0.
    // Our encoder's L1 anchor is a P-slice with single L0 ref so this is
    // always the available data.
    if slot.ref_idx_l0 != 0 {
        return false;
    }
    // §8.4.1.2.2 step 7 — colZeroFlag iff |mvCol| <= 1 in both components.
    slot.mv_l0.0.abs() <= 1 && slot.mv_l0.1.abs() <= 1
}

/// §8.4.1.2.2 — encoder-side spatial direct MV / refIdx derivation for
/// the current MB. Mirrors the decoder's `build_spatial_direct_partitions`
/// using the encoder's (`mv_grid_l0`, `mv_grid_l1`) state and the L1
/// anchor's `partition_mvs` for the step-7 colZeroFlag check.
///
/// Implements:
///   * step 4 — MinPositive(refIdxL0A, refIdxL0B, refIdxL0C) for both
///     lists, with §8.4.1.3.2 C→D substitution applied first.
///   * step 5 — `directZeroPredictionFlag` when both lists' MinPositive
///     output is < 0; both refs collapse to 0, both MVs to (0, 0).
///   * step 6 — per-partition MV derivation: median MVpred from the
///     A/B/C neighbours unless `direct_zero` or `refIdxLX < 0`.
///   * step 7 — per-8x8 colZeroFlag short-circuit: when refIdxLX == 0
///     AND colZeroFlag fires, that list's MV for that partition is 0.
///
/// `direct_8x8_inference_flag = 1` is hard-wired in our SPS so the
/// granularity is 8x8 (4 partitions per MB).
fn b_spatial_direct_derive(
    mv_grid_l0: &MvGrid,
    mv_grid_l1: &MvGrid,
    l1_partition_mvs: &[FrameRefPartitionMv],
    width_mbs: usize,
    mb_x: usize,
    mb_y: usize,
) -> BDirectDerivation {
    // §8.4.1.2.2 step 2-3 — MB-level (A, B, C, D) for both lists.
    let (a_l0, b_l0, c_l0, d_l0) = neighbour_mvs_16x16(mv_grid_l0, mb_x, mb_y);
    let (a_l1, b_l1, c_l1, d_l1) = neighbour_mvs_16x16(mv_grid_l1, mb_x, mb_y);

    // Run the §8.4.1.2.2 step 4-5 derivation through the helper to get
    // ref_idx_l0/l1 + the MB-level mvL0/mvL1 (subject to step 7
    // override below). The helper returns refIdx >= 0 for both lists
    // when directZero fires, but with `direct_zero = (mv == 0)` the
    // caller can detect the fallback by re-checking the MinPositive
    // result independently.
    let (ref_idx_l0_helper, mv_l0_mb, ref_idx_l1_helper, mv_l1_mb) =
        crate::mv_deriv::derive_b_spatial_direct_with_d(
            a_l0, a_l1, b_l0, b_l1, c_l0, c_l1, d_l0, d_l1,
        );

    // Repeat the step-4 MinPositive chain locally to detect directZero
    // (the helper folds it into the output).
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

    // §8.4.1.2.2 step 6/7 — per 8x8 partition MV derivation.
    let mut mv_l0_per_8x8 = [Mv::ZERO; 4];
    let mut mv_l1_per_8x8 = [Mv::ZERO; 4];
    for part in 0..4usize {
        let cz = col_zero_flag(l1_partition_mvs, width_mbs, mb_x, mb_y, part);
        // §8.4.1.2.2 step 6/7 — three forced-zero conditions per list
        // (directZero, refIdx<0, refIdx==0 AND colZeroFlag); otherwise
        // the MB-level median MV applies.
        let l0_force_zero = direct_zero || ref_idx_l0_final < 0 || (ref_idx_l0_final == 0 && cz);
        let l1_force_zero = direct_zero || ref_idx_l1_final < 0 || (ref_idx_l1_final == 0 && cz);
        mv_l0_per_8x8[part] = if l0_force_zero { Mv::ZERO } else { mv_l0_mb };
        mv_l1_per_8x8[part] = if l1_force_zero { Mv::ZERO } else { mv_l1_mb };
    }

    BDirectDerivation {
        ref_idx_l0: ref_idx_l0_final,
        ref_idx_l1: ref_idx_l1_final,
        mv_l0_per_8x8,
        mv_l1_per_8x8,
        direct_zero,
    }
}

// ---------------------------------------------------------------------------
// Round 24 — §8.4.1.2.3 B-slice **temporal direct** derivation (encoder side).
//
// Temporal direct mode (`direct_spatial_mv_pred_flag = 0`) replaces the
// §8.4.1.2.2 spatial MV / refIdx derivation with a POC-distance-driven
// scaling of the colocated picture's L0 motion vectors:
//
//   pic1 = colocated picture = RefPicList1[0]   (the L1 anchor)
//   pic0 = RefPicList0[refIdxL0]                 (the L0 reference picked by
//                                                 MapColToList0(refIdxCol))
//   td = Clip3(-128, 127, DiffPicOrderCnt(pic1, pic0))
//   tb = Clip3(-128, 127, DiffPicOrderCnt(currPicOrField, pic0))
//   tx = (16384 + |td/2|) / td
//   DistScaleFactor = Clip3(-1024, 1023, (tb*tx + 32) >> 6)
//   mvL0 = (DistScaleFactor*mvCol + 128) >> 8
//   mvL1 = mvL0 - mvCol
//
// `MapColToList0` (eq. 8-191) maps the colocated block's `refIdxCol` (an
// index into the colocated picture's RefPicList0) into the current
// slice's RefPicList0 by **picture identity (POC)**. Our encoder uses a
// single L0 reference per B-slice (the IDR), so MapColToList0 collapses
// to "the current slice's RefPicList0[0] = the IDR" whenever the
// colocated MB's refIdxCol == 0, and to refIdxL0 = 0 when refIdxCol < 0
// (the "directZero" fallback per eq. 8-191's first branch).
//
// `direct_8x8_inference_flag = 1` in our SPS, so the granularity is
// 4 partitions per MB (one (mvL0, mvL1) per 8x8). The 4x4 sub-grid
// within each 8x8 carries the same MV — same convention the spatial
// path uses.
//
// Spec sections cited:
//   §8.4.1.2.1     — colocated block / picture selection (Table 8-6)
//   §8.4.1.2.3     — temporal direct MV derivation (eq. 8-191..8-202)
//   §8.4.1.2 NOTE 3 — direct_8x8_inference_flag = 1 → per-8x8 granularity
//   §8.2.1         — DiffPicOrderCnt (frame coding ⇒ POC subtract)
// ---------------------------------------------------------------------------

/// §8.4.1.2.3 — encoder-side temporal direct derivation for one MB.
///
/// Mirrors the decoder's `derive_temporal_direct_mvs_for_block` for our
/// constrained encoder setup:
///   * Single L0 reference per B-slice (the IDR / older anchor).
///   * Single L1 reference per B-slice (the prior P-frame).
///   * `direct_8x8_inference_flag = 1` → 4 partitions per MB.
///   * `MapColToList0` collapses to identity when refIdxCol == 0 (the
///     IDR is at index 0 of both the L1 anchor's RefPicList0 and the
///     current B's RefPicList0); else falls back to refIdxL0 = 0.
///   * Long-term refs not used (encoder emits only short-term) — the
///     long-term shortcut never fires.
///
/// `curr_poc` / `pic0_poc` / `pic1_poc` are picture order counts.
/// For our POC type 0 frame-coded encoder these equal the
/// `pic_order_cnt_lsb` values shipped in the slice headers.
fn b_temporal_direct_derive(
    l1_partition_mvs: &[FrameRefPartitionMv],
    width_mbs: usize,
    mb_x: usize,
    mb_y: usize,
    curr_poc: i32,
    pic0_poc: i32,
    pic1_poc: i32,
) -> BDirectDerivation {
    // §8.4.1.2.3 / eq. 8-192 — refIdxL1 is always 0 for direct mode
    // (the colocated picture itself is RefPicList1[0]).
    //
    // refIdxL0: see eq. 8-191. Our encoder has a single L0 reference,
    // so MapColToList0(refIdxCol) collapses to 0 whenever the colocated
    // L0 reference matches the IDR (which it always does in our IPB
    // setup — the L1 P-frame's only L0 ref is the IDR). When the
    // colocated block is intra (refIdxCol < 0), eq. 8-191 forces
    // refIdxL0 = 0 as well. Either way refIdxL0 = 0 across the MB.
    let mut mv_l0_per_8x8 = [Mv::ZERO; 4];
    let mut mv_l1_per_8x8 = [Mv::ZERO; 4];

    for part in 0..4usize {
        let idx = (mb_y * width_mbs + mb_x) * 4 + part;
        // Default colocated mv = (0, 0) when no L1 partition data is
        // available (e.g. an IDR L1 anchor — recall §8.4.1.2.1 NOTE
        // for intra colocated MBs: mvCol = 0 / refIdxCol = -1).
        let (mv_col, is_intra) = match l1_partition_mvs.get(idx) {
            Some(slot) if !slot.is_intra => {
                (Mv::new(slot.mv_l0.0 as i32, slot.mv_l0.1 as i32), false)
            }
            _ => (Mv::ZERO, true),
        };
        // §8.4.1.2.3 eq. 8-195/8-196 short-circuit when the colocated
        // block is intra (or when DiffPicOrderCnt(pic1, pic0) == 0,
        // handled inside `derive_b_temporal_direct`): mvL0 = mvCol = 0,
        // mvL1 = 0. Same numerical result either way.
        if is_intra {
            mv_l0_per_8x8[part] = Mv::ZERO;
            mv_l1_per_8x8[part] = Mv::ZERO;
            continue;
        }
        // Forward to the shared §8.4.1.2.3 scaling helper (eq. 8-197..8-200).
        // The helper handles the td == 0 → mvL0 = mvCol / mvL1 = 0
        // shortcut internally.
        let (mv_l0, mv_l1) =
            crate::mv_deriv::derive_b_temporal_direct(mv_col, pic1_poc, curr_poc, pic0_poc);
        mv_l0_per_8x8[part] = mv_l0;
        mv_l1_per_8x8[part] = mv_l1;
    }

    // Both refIdxL0 and refIdxL1 are 0 in our single-ref-per-list setup.
    BDirectDerivation {
        ref_idx_l0: 0,
        ref_idx_l1: 0,
        mv_l0_per_8x8,
        mv_l1_per_8x8,
        direct_zero: false,
    }
}

/// Output of [`Encoder::encode_b`] — a single B-slice access unit and the
/// matching reconstruction. Same layout as [`EncodedP`]; B-frames are
/// non-reference (`nal_ref_idc = 0`) so they do not feed any further
/// inter-coding pass — but we publish the recon anyway for measurement
/// (PSNR vs source) and ffmpeg-interop testing.
pub struct EncodedB {
    pub annex_b: Vec<u8>,
    pub recon_y: Vec<u8>,
    pub recon_u: Vec<u8>,
    pub recon_v: Vec<u8>,
    pub recon_width: u32,
    pub recon_height: u32,
    /// `frame_num` of this B-frame. Per §7.4.3 this matches the B's
    /// position in decode order (NOT display order).
    pub frame_num: u32,
    /// `pic_order_cnt_lsb` of this B-frame — used by the decoder's
    /// §8.2.4 list construction to determine which ref is "before" and
    /// which is "after" in display order.
    pub pic_order_cnt_lsb: u32,
    /// Per-8x8-partition MV state. B-frames are non-reference so this
    /// is normally unused, but is provided for API symmetry.
    pub partition_mvs: Vec<FrameRefPartitionMv>,
}

impl<'a> From<&'a EncodedB> for EncodedFrameRef<'a> {
    fn from(e: &'a EncodedB) -> Self {
        Self {
            width: e.recon_width,
            height: e.recon_height,
            recon_y: &e.recon_y,
            recon_u: &e.recon_u,
            recon_v: &e.recon_v,
            partition_mvs: &e.partition_mvs,
            pic_order_cnt: e.pic_order_cnt_lsb as i32,
        }
    }
}

/// Per-MB chosen B-prediction descriptor — what `encode_b_mb` decides
/// before emitting the macroblock syntax + recon. `mvd_lN` already
/// accounts for the §8.4.1.3 `mvpLN` predictor of the chosen list.
#[derive(Debug, Clone, Copy)]
struct BMbDecision {
    /// Which list(s) this MB references.
    pred: BPred16x16,
    /// Chosen L0 MV (1/4-pel). Always Mv::ZERO when `pred == L1`.
    mv_l0: Mv,
    /// Chosen L1 MV (1/4-pel). Always Mv::ZERO when `pred == L0`.
    mv_l1: Mv,
    /// L0 mvd against the §8.4.1.3 mvp. Zero when L0 unused.
    mvd_l0_x: i32,
    mvd_l0_y: i32,
    /// L1 mvd against the §8.4.1.3 mvp.
    mvd_l1_x: i32,
    mvd_l1_y: i32,
    /// Combined luma predictor (post-§8.4.2.3 weighted-pred merge for
    /// bipred, or one list verbatim for L0-only / L1-only).
    pred_y: [i32; 256],
    /// Combined chroma Cb predictor.
    pred_u: [i32; 64],
    /// Combined chroma Cr predictor.
    pred_v: [i32; 64],
}

/// Round-26 — explicit weighted-bipred merge parameters (luma only).
///
/// When `Some`, `build_b_predictors` replaces the §8.4.2.3.1 default
/// `(L0 + L1 + 1) >> 1` luma merge with the §8.4.2.3.2 explicit formula
/// (eq. 8-276) using these `(weight, offset)` pairs and `log2_wd` denom.
/// Chroma keeps the default merge — our `pred_weight_table()` writer
/// signals `chroma_weight_lN_flag = 0` for every ref so the decoder
/// matches.
#[derive(Debug, Clone, Copy)]
struct WeightedBipredLuma {
    log2_wd: u32,
    weight_l0: i32,
    offset_l0: i32,
    weight_l1: i32,
    offset_l1: i32,
}

/// Round-26 — slice-wide weight selector for explicit weighted bipred.
///
/// Models the source luma plane as
///   `src[k] ≈ ((w0 * L0[k] + w1 * L1[k] + 2^logWD) >> (logWD+1)) +
///             ((o0 + o1 + 1) >> 1)`
/// (the §8.4.2.3.2 eq. 8-276 prediction). For a slice-wide constant fit
/// we collapse to a 2-parameter least-squares problem in (α, β):
///   `src ≈ α * L0 + (1 - α) * L1 + β`
/// then quantise α to a denom-32 fraction (`logWD = 5`):
///   `w0 = round(α * 32)`, `w1 = 32 - w0`
/// Offsets are split symmetrically: `o0 = o1 = round(β / 2)`. The fit
/// uses the encoder's mb-aligned regions only — no MV-aware sample
/// matching — which is sufficient for cross-fade content where the two
/// references share spatial structure (no per-MB motion).
///
/// Picks the (w0, w1) pair on a coarse grid `{0, 4, 8, 12, ..., 32}`
/// (9 candidates) by sum-of-squared-error against the source after the
/// §8.4.2.3.2 round-and-clip is applied. Offsets are constrained to
/// the integer least-squares optimum, clamped to ±127 per spec.
///
/// Falls back to the default (16, 16, 0, 0) when the chosen weights
/// don't beat the default merge by at least 1% on SSD — that ensures
/// the bit cost of pred_weight_table doesn't backfire on neutral
/// content. Returns `None`-equivalent default in that case.
#[allow(clippy::too_many_arguments)]
fn compute_explicit_bipred_weights(
    src_y: &[u8],
    ref_l0_y: &[u8],
    ref_l1_y: &[u8],
    width: usize,
    height: usize,
    ref_l0_stride: usize,
    ref_l1_stride: usize,
) -> ExplicitBipredWeightTable {
    const LOG2_WD: u32 = 5; // denom = 32; w0+w1 = 64 for the canonical "α + (1-α) = 1"
    const DENOM: i32 = 1 << (LOG2_WD + 1); // 64

    // Default weights: w0 = w1 = 32 (sum = 64 → α = 0.5 → eq. 8-273).
    let default_table = ExplicitBipredWeightTable {
        luma_log2_weight_denom: LOG2_WD,
        luma_weight_l0: 32,
        luma_offset_l0: 0,
        luma_weight_l1: 32,
        luma_offset_l1: 0,
    };

    // Round-and-clip the §8.4.2.3.2 prediction for a given (w0, w1, o)
    // combo and return the SSE against `src`. `o` is the per-list offset
    // (symmetric: o0 = o1 = o so `(o0+o1+1)>>1` is just o). Used both
    // for the candidate search and for the default-vs-explicit decision.
    let sse_for = |w0: i32, w1: i32, o: i32| -> u64 {
        let mut sse: u64 = 0;
        let logwd_plus1 = LOG2_WD + 1;
        let round = 1i32 << LOG2_WD;
        let off_avg = (o + o + 1) >> 1;
        for j in 0..height {
            for i in 0..width {
                let p0 = ref_l0_y[j * ref_l0_stride + i] as i32;
                let p1 = ref_l1_y[j * ref_l1_stride + i] as i32;
                let v = ((p0 * w0 + p1 * w1 + round) >> logwd_plus1) + off_avg;
                let pred = v.clamp(0, 255);
                let s = src_y[j * width + i] as i32;
                let d = s - pred;
                sse = sse.saturating_add((d * d) as u64);
            }
        }
        sse
    };

    // Coarse grid over α at 1/16 granularity (sum w0+w1 == DENOM = 64).
    // For every candidate w0, also pick the integer offset o that
    // minimises SSE (closed-form: o = round(mean(src - pred_no_offset))).
    let total = (width * height) as f64;
    let mut best_sse = u64::MAX;
    let mut best_w0 = 32i32;
    let mut best_off = 0i32;
    for w0 in (0..=DENOM).step_by(4) {
        let w1 = DENOM - w0;
        // Compute optimal symmetric offset (o0 = o1 = o) by averaging
        // the residual. This is the LS fit for the constant component.
        let mut sum_resid: f64 = 0.0;
        let logwd_plus1 = LOG2_WD + 1;
        let round = 1i32 << LOG2_WD;
        for j in 0..height {
            for i in 0..width {
                let p0 = ref_l0_y[j * ref_l0_stride + i] as i32;
                let p1 = ref_l1_y[j * ref_l1_stride + i] as i32;
                let v = (p0 * w0 + p1 * w1 + round) >> logwd_plus1;
                let s = src_y[j * width + i] as i32;
                sum_resid += (s - v) as f64;
            }
        }
        let o_raw = (sum_resid / total).round() as i32;
        let o = o_raw.clamp(-127, 127);
        let sse = sse_for(w0, w1, o);
        if sse < best_sse {
            best_sse = sse;
            best_w0 = w0;
            best_off = o;
        }
    }

    // Compute the default (16, 16, 0, 0)-equivalent SSE for the
    // accept/reject decision. Note: with w0=w1=32, logWD=5, eq. 8-276
    // simplifies to `(L0 + L1 + 32) >> 6 + 0` which differs from the
    // §8.4.2.3.1 default by one rounding-bit position. To compare
    // apples-to-apples we evaluate the *actual* default merge.
    let mut default_sse: u64 = 0;
    for j in 0..height {
        for i in 0..width {
            let p0 = ref_l0_y[j * ref_l0_stride + i] as i32;
            let p1 = ref_l1_y[j * ref_l1_stride + i] as i32;
            let pred = (p0 + p1 + 1) >> 1;
            let s = src_y[j * width + i] as i32;
            let d = s - pred;
            default_sse = default_sse.saturating_add((d * d) as u64);
        }
    }

    // Accept the chosen weights only if they beat the default by a
    // healthy margin (≥ 1% SSE reduction). This avoids spending the
    // ~30-bit pred_weight_table budget on slices where it doesn't help.
    let margin = default_sse / 100;
    if best_sse + margin >= default_sse {
        return default_table;
    }

    let w1 = DENOM - best_w0;
    ExplicitBipredWeightTable {
        luma_log2_weight_denom: LOG2_WD,
        luma_weight_l0: best_w0,
        luma_offset_l0: best_off,
        luma_weight_l1: w1,
        luma_offset_l1: best_off,
    }
}

/// Build the L0 + L1 luma + chroma predictors for a 16x16 MB given the
/// chosen MVs and the two reference frames. Returns one composite buffer
/// per plane that already incorporates the §8.4.2.3.1 default weighted
/// merge for bipred (`(L0 + L1 + 1) >> 1`). For L0-only / L1-only the
/// returned predictor is just that list's interpolated samples.
///
/// Round-26 — when `weighted` is `Some`, the bipred luma merge instead
/// uses the §8.4.2.3.2 explicit formula (eq. 8-276):
///   `Clip1(((L0*w0 + L1*w1 + 2^logWD) >> (logWD+1)) +
///          ((o0 + o1 + 1) >> 1))`
/// Chroma still uses the default §8.4.2.3.1 average; our slice header
/// emit flags chroma weights as absent so the decoder agrees.
#[allow(clippy::too_many_arguments)]
fn build_b_predictors(
    pred: BPred16x16,
    ref_l0: &EncodedFrameRef<'_>,
    ref_l1: &EncodedFrameRef<'_>,
    mb_x: usize,
    mb_y: usize,
    mv_l0: Mv,
    mv_l1: Mv,
    weighted: Option<WeightedBipredLuma>,
) -> ([i32; 256], [i32; 64], [i32; 64]) {
    // Always interpolate from each list it uses; the §8.4.2.3.1 average
    // is bit-equivalent to the sum-then-shift the decoder applies.
    let chroma_w_l0 = ref_l0.width / 2;
    let chroma_h_l0 = ref_l0.height / 2;
    let chroma_w_l1 = ref_l1.width / 2;
    let chroma_h_l1 = ref_l1.height / 2;

    let p_y_l0 = if pred.uses_l0() {
        Some(build_inter_pred_luma(
            ref_l0.recon_y,
            ref_l0.width,
            ref_l0.height,
            mb_x,
            mb_y,
            mv_l0,
        ))
    } else {
        None
    };
    let p_y_l1 = if pred.uses_l1() {
        Some(build_inter_pred_luma(
            ref_l1.recon_y,
            ref_l1.width,
            ref_l1.height,
            mb_x,
            mb_y,
            mv_l1,
        ))
    } else {
        None
    };
    let p_u_l0 = if pred.uses_l0() {
        Some(build_inter_pred_chroma(
            ref_l0.recon_u,
            chroma_w_l0,
            chroma_h_l0,
            mb_x,
            mb_y,
            mv_l0,
        ))
    } else {
        None
    };
    let p_u_l1 = if pred.uses_l1() {
        Some(build_inter_pred_chroma(
            ref_l1.recon_u,
            chroma_w_l1,
            chroma_h_l1,
            mb_x,
            mb_y,
            mv_l1,
        ))
    } else {
        None
    };
    let p_v_l0 = if pred.uses_l0() {
        Some(build_inter_pred_chroma(
            ref_l0.recon_v,
            chroma_w_l0,
            chroma_h_l0,
            mb_x,
            mb_y,
            mv_l0,
        ))
    } else {
        None
    };
    let p_v_l1 = if pred.uses_l1() {
        Some(build_inter_pred_chroma(
            ref_l1.recon_v,
            chroma_w_l1,
            chroma_h_l1,
            mb_x,
            mb_y,
            mv_l1,
        ))
    } else {
        None
    };

    // Merge per pred mode.
    let mut pred_y = [0i32; 256];
    let mut pred_u = [0i32; 64];
    let mut pred_v = [0i32; 64];
    match pred {
        BPred16x16::L0 => {
            pred_y = p_y_l0.unwrap();
            pred_u = p_u_l0.unwrap();
            pred_v = p_v_l0.unwrap();
        }
        BPred16x16::L1 => {
            pred_y = p_y_l1.unwrap();
            pred_u = p_u_l1.unwrap();
            pred_v = p_v_l1.unwrap();
        }
        BPred16x16::Bi => {
            let l0_y = p_y_l0.unwrap();
            let l1_y = p_y_l1.unwrap();
            if let Some(wp) = weighted {
                // §8.4.2.3.2 eq. 8-276 — explicit weighted bipred:
                //   Clip1Y(((L0*w0 + L1*w1 + 2^logWD) >> (logWD+1)) +
                //          ((o0 + o1 + 1) >> 1))
                let logwd = wp.log2_wd;
                let off_avg = (wp.offset_l0 + wp.offset_l1 + 1) >> 1;
                let round = 1i32 << logwd;
                for i in 0..256 {
                    let v =
                        (l0_y[i] * wp.weight_l0 + l1_y[i] * wp.weight_l1 + round) >> (logwd + 1);
                    pred_y[i] = (v + off_avg).clamp(0, 255);
                }
            } else {
                // §8.4.2.3.1 eq. 8-273 — default `(L0 + L1 + 1) >> 1`.
                for i in 0..256 {
                    pred_y[i] = (l0_y[i] + l1_y[i] + 1) >> 1;
                }
            }
            let l0_u = p_u_l0.unwrap();
            let l1_u = p_u_l1.unwrap();
            for i in 0..64 {
                pred_u[i] = (l0_u[i] + l1_u[i] + 1) >> 1;
            }
            let l0_v = p_v_l0.unwrap();
            let l1_v = p_v_l1.unwrap();
            for i in 0..64 {
                pred_v[i] = (l0_v[i] + l1_v[i] + 1) >> 1;
            }
        }
    }
    (pred_y, pred_u, pred_v)
}

/// Round-23 — per-8x8 prediction kind for a `B_8x8` MB whose four
/// sub-MBs are all `B_Direct_8x8`. Each 8x8 partition gets its own
/// (refIdxL0, refIdxL1, mvL0, mvL1) per §8.4.1.2.2 step 4-7, and the
/// (refIdxL0 >= 0, refIdxL1 >= 0) pair maps to one of the same three
/// merge modes the 16x16 path uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectPart8x8 {
    /// L0-only — `refIdxL0 >= 0`, `refIdxL1 == -1`.
    L0,
    /// L1-only — `refIdxL0 == -1`, `refIdxL1 >= 0`.
    L1,
    /// Both lists, default weighted average `(L0 + L1 + 1) >> 1` per
    /// §8.4.2.3.1 eq. 8-273.
    Bi,
    /// Neither list (would only arise for an all-intra colocated source
    /// without `directZeroPredictionFlag` correction). Not expected in
    /// practice — the spatial-direct derivation forces both refs to 0
    /// when neighbours are all unavailable. We treat this as L0-only with
    /// mv=0 to keep a single defined predictor in scope.
    None,
}

impl DirectPart8x8 {
    fn uses_l0(self) -> bool {
        matches!(self, Self::L0 | Self::Bi)
    }
    fn uses_l1(self) -> bool {
        matches!(self, Self::L1 | Self::Bi)
    }
}

/// Round-23 — derive the per-8x8 prediction kind for each of the four
/// sub-MBs of a `B_8x8` macroblock from the §8.4.1.2.2 spatial-direct
/// derivation result.
///
/// With `direct_8x8_inference_flag = 1` (hard-wired in our SPS), the
/// derivation produces one (mvL0, mvL1, refIdxL0, refIdxL1) per 8x8
/// partition. The current `BDirectDerivation` shape carries a single
/// (refIdxL0, refIdxL1) for the whole MB (matching the spec — refIdxLX
/// is MB-level, not per-partition), so the per-8x8 prediction kind only
/// varies when the per-8x8 MV grid disagrees AND we want to honour
/// per-list "off" status when refIdx == -1.
fn direct_per_8x8_kinds(d: &BDirectDerivation) -> [DirectPart8x8; 4] {
    let l0_used = d.ref_idx_l0 >= 0;
    let l1_used = d.ref_idx_l1 >= 0;
    // §8.4.1.2.2 step 7 colZeroFlag can drive a per-list MV to (0,0)
    // even when refIdxLX >= 0; that does NOT change predFlagLX, so
    // L0/L1 still apply at the merge stage. We honour predFlagLX
    // exactly via refIdxLX sign.
    let kind = match (l0_used, l1_used) {
        (true, true) => DirectPart8x8::Bi,
        (true, false) => DirectPart8x8::L0,
        (false, true) => DirectPart8x8::L1,
        (false, false) => DirectPart8x8::None,
    };
    [kind; 4]
}

/// Round-23 — build per-8x8 luma + chroma predictors for a `B_8x8`
/// macroblock with all-`B_Direct_8x8` sub-MBs. Each 8x8 quadrant uses
/// its own (mvL0, mvL1) pair from the §8.4.1.2.2 derivation; the
/// composite is stitched into a 16x16 luma buffer + two 8x8 chroma
/// buffers (matching the `build_b_predictors` shape so the existing
/// residual / reconstruction code works unchanged).
///
/// For each 8x8 quadrant index `q ∈ 0..4` (raster Z-order: top-left,
/// top-right, bottom-left, bottom-right), the per-list 8x8 luma + 4x4
/// chroma predictors are computed via `build_inter_pred_luma_8x8` /
/// `build_inter_pred_chroma_4x4`. Bi-pred uses the same §8.4.2.3.1
/// default weighted average `(L0+L1+1)>>1` as 16x16 / 16x8 / 8x16.
fn build_b_direct_8x8_predictors(
    kinds: [DirectPart8x8; 4],
    mv_l0_per_8x8: [Mv; 4],
    mv_l1_per_8x8: [Mv; 4],
    ref_l0: &EncodedFrameRef<'_>,
    ref_l1: &EncodedFrameRef<'_>,
    mb_x: usize,
    mb_y: usize,
) -> ([i32; 256], [i32; 64], [i32; 64]) {
    let mut pred_y = [0i32; 256];
    let mut pred_u = [0i32; 64];
    let mut pred_v = [0i32; 64];

    let chroma_w_l0 = ref_l0.width / 2;
    let chroma_h_l0 = ref_l0.height / 2;
    let chroma_w_l1 = ref_l1.width / 2;
    let chroma_h_l1 = ref_l1.height / 2;

    // §6.4.3 raster-Z order across the four 8x8 quadrants of an MB:
    //   q=0 → (sub_x=0, sub_y=0)  top-left
    //   q=1 → (sub_x=1, sub_y=0)  top-right
    //   q=2 → (sub_x=0, sub_y=1)  bottom-left
    //   q=3 → (sub_x=1, sub_y=1)  bottom-right
    for q in 0..4usize {
        let sub_x = q % 2;
        let sub_y = q / 2;
        let kind = kinds[q];
        let mv0 = mv_l0_per_8x8[q];
        let mv1 = mv_l1_per_8x8[q];

        // Luma 8x8 predictor for this quadrant.
        let l0_y = if kind.uses_l0() {
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
        let l1_y = if kind.uses_l1() {
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
        let q_y: [i32; 64] = match kind {
            DirectPart8x8::L0 => l0_y.unwrap(),
            DirectPart8x8::L1 => l1_y.unwrap(),
            DirectPart8x8::Bi => {
                let l0 = l0_y.unwrap();
                let l1 = l1_y.unwrap();
                let mut out = [0i32; 64];
                for i in 0..64 {
                    out[i] = (l0[i] + l1[i] + 1) >> 1;
                }
                out
            }
            DirectPart8x8::None => {
                // Defensive fallback — predict from L0 with mv=0.
                build_inter_pred_luma_8x8(
                    ref_l0.recon_y,
                    ref_l0.width,
                    ref_l0.height,
                    mb_x,
                    mb_y,
                    sub_x,
                    sub_y,
                    Mv::ZERO,
                )
            }
        };

        // Chroma 4x4 predictor (4:2:0) for this quadrant.
        let l0_u = if kind.uses_l0() {
            Some(build_inter_pred_chroma_4x4(
                ref_l0.recon_u,
                chroma_w_l0,
                chroma_h_l0,
                mb_x,
                mb_y,
                sub_x,
                sub_y,
                mv0,
            ))
        } else {
            None
        };
        let l0_v = if kind.uses_l0() {
            Some(build_inter_pred_chroma_4x4(
                ref_l0.recon_v,
                chroma_w_l0,
                chroma_h_l0,
                mb_x,
                mb_y,
                sub_x,
                sub_y,
                mv0,
            ))
        } else {
            None
        };
        let l1_u = if kind.uses_l1() {
            Some(build_inter_pred_chroma_4x4(
                ref_l1.recon_u,
                chroma_w_l1,
                chroma_h_l1,
                mb_x,
                mb_y,
                sub_x,
                sub_y,
                mv1,
            ))
        } else {
            None
        };
        let l1_v = if kind.uses_l1() {
            Some(build_inter_pred_chroma_4x4(
                ref_l1.recon_v,
                chroma_w_l1,
                chroma_h_l1,
                mb_x,
                mb_y,
                sub_x,
                sub_y,
                mv1,
            ))
        } else {
            None
        };
        let merge_chroma = |c0: Option<[i32; 16]>, c1: Option<[i32; 16]>| -> [i32; 16] {
            match kind {
                DirectPart8x8::L0 => c0.unwrap(),
                DirectPart8x8::L1 => c1.unwrap(),
                DirectPart8x8::Bi => {
                    let a = c0.unwrap();
                    let b = c1.unwrap();
                    let mut out = [0i32; 16];
                    for i in 0..16 {
                        out[i] = (a[i] + b[i] + 1) >> 1;
                    }
                    out
                }
                DirectPart8x8::None => c0.unwrap_or([0; 16]),
            }
        };
        let q_u = merge_chroma(l0_u, l1_u);
        let q_v = merge_chroma(l0_v, l1_v);

        // Stitch the 8x8 luma + 4x4 chroma quadrants into the 16x16 /
        // 8x8 plane buffers.
        for j in 0..8usize {
            for i in 0..8usize {
                let dst_y = (sub_y * 8 + j) * 16 + sub_x * 8 + i;
                pred_y[dst_y] = q_y[j * 8 + i];
            }
        }
        for j in 0..4usize {
            for i in 0..4usize {
                let dst_c = (sub_y * 4 + j) * 8 + sub_x * 4 + i;
                pred_u[dst_c] = q_u[j * 4 + i];
                pred_v[dst_c] = q_v[j * 4 + i];
            }
        }
    }
    (pred_y, pred_u, pred_v)
}

// ---------------------------------------------------------------------------
// Round 25 — B_8x8 with mixed sub_mb_type per cell.
//
// Each of the four 8x8 quadrants of a B_8x8 MB picks independently from
// {Direct (§8.4.1.2.2 / §8.4.1.2.3), L0-only, L1-only, Bi}. The cell's
// MB-syntax cost varies — Direct is free (no MVs / refs), L0/L1 each
// cost ~12 bits of mvd (one mvd_l0 or mvd_l1 pair plus the sub_mb_type
// ue(v) overhead vs Direct), and Bi costs ~20 bits.
//
// Spec sections cited:
//   §7.3.5.2          — sub_mb_pred() syntax with sub_mb_type[i]
//   §7.4.5 Table 7-18 — sub_mb_type ∈ {0..3} for the four single-partition
//                       cell types (B_Direct_8x8 / B_L0_8x8 / B_L1_8x8 /
//                       B_Bi_8x8).
//   §8.4.1.2.2        — spatial direct derivation (mvL0, mvL1 per 8x8)
//   §8.4.2.3.1 eq. 8-273 — default weighted bipred (L0 + L1 + 1) >> 1
// ---------------------------------------------------------------------------

/// SAD between an 8x8 source block (at MB-relative offset `(off_x,
/// off_y)` ∈ {0, 8}² of MB `(mb_x, mb_y)` in `src_y`) and an 8x8 luma
/// predictor buffer (row-major i32, length 64). Predictor samples are
/// clipped to 0..=255 to match the decoder.
pub(crate) fn sad_8x8_pred(
    src_y: &[u8],
    src_stride: usize,
    mb_x: usize,
    mb_y: usize,
    off_x: usize,
    off_y: usize,
    pred: &[i32; 64],
) -> u32 {
    let mut s = 0u32;
    for j in 0..8usize {
        for i in 0..8usize {
            let src = src_y[(mb_y * 16 + off_y + j) * src_stride + mb_x * 16 + off_x + i] as i32;
            let p = pred[j * 8 + i].clamp(0, 255);
            s = s.saturating_add((src - p).unsigned_abs());
        }
    }
    s
}

/// Round-25 — build one 8x8 luma cell predictor for an explicit-inter
/// `B_8x8` cell (`B_L0_8x8` / `B_L1_8x8` / `B_Bi_8x8`). Bi uses the
/// §8.4.2.3.1 default weighted average `(L0 + L1 + 1) >> 1`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_b_explicit_8x8_cell_luma(
    cell: BSubMbCell,
    ref_l0: &EncodedFrameRef<'_>,
    ref_l1: &EncodedFrameRef<'_>,
    mb_x: usize,
    mb_y: usize,
    sub_x: usize,
    sub_y: usize,
    mv_l0: Mv,
    mv_l1: Mv,
) -> [i32; 64] {
    match cell {
        BSubMbCell::L0 => build_inter_pred_luma_8x8(
            ref_l0.recon_y,
            ref_l0.width,
            ref_l0.height,
            mb_x,
            mb_y,
            sub_x,
            sub_y,
            mv_l0,
        ),
        BSubMbCell::L1 => build_inter_pred_luma_8x8(
            ref_l1.recon_y,
            ref_l1.width,
            ref_l1.height,
            mb_x,
            mb_y,
            sub_x,
            sub_y,
            mv_l1,
        ),
        BSubMbCell::Bi => {
            let a = build_inter_pred_luma_8x8(
                ref_l0.recon_y,
                ref_l0.width,
                ref_l0.height,
                mb_x,
                mb_y,
                sub_x,
                sub_y,
                mv_l0,
            );
            let b = build_inter_pred_luma_8x8(
                ref_l1.recon_y,
                ref_l1.width,
                ref_l1.height,
                mb_x,
                mb_y,
                sub_x,
                sub_y,
                mv_l1,
            );
            let mut out = [0i32; 64];
            for i in 0..64 {
                out[i] = (a[i] + b[i] + 1) >> 1;
            }
            out
        }
        BSubMbCell::Direct => {
            // Caller is responsible for using `build_inter_pred_luma_8x8`
            // with the §8.4.1.2.2 derived per-8x8 MVs for Direct cells.
            // Defensive fallback: L0 with mv=0.
            build_inter_pred_luma_8x8(
                ref_l0.recon_y,
                ref_l0.width,
                ref_l0.height,
                mb_x,
                mb_y,
                sub_x,
                sub_y,
                Mv::ZERO,
            )
        }
    }
}

/// Round-25 — build one 4x4 chroma cell predictor (4:2:0) for an
/// explicit-inter `B_8x8` cell. Symmetric with `build_b_explicit_8x8_cell_luma`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_b_explicit_4x4_cell_chroma(
    cell: BSubMbCell,
    ref_plane_l0: &[u8],
    ref_plane_l1: &[u8],
    chroma_w_l0: u32,
    chroma_h_l0: u32,
    chroma_w_l1: u32,
    chroma_h_l1: u32,
    mb_x: usize,
    mb_y: usize,
    sub_x: usize,
    sub_y: usize,
    mv_l0: Mv,
    mv_l1: Mv,
) -> [i32; 16] {
    match cell {
        BSubMbCell::L0 => build_inter_pred_chroma_4x4(
            ref_plane_l0,
            chroma_w_l0,
            chroma_h_l0,
            mb_x,
            mb_y,
            sub_x,
            sub_y,
            mv_l0,
        ),
        BSubMbCell::L1 => build_inter_pred_chroma_4x4(
            ref_plane_l1,
            chroma_w_l1,
            chroma_h_l1,
            mb_x,
            mb_y,
            sub_x,
            sub_y,
            mv_l1,
        ),
        BSubMbCell::Bi => {
            let a = build_inter_pred_chroma_4x4(
                ref_plane_l0,
                chroma_w_l0,
                chroma_h_l0,
                mb_x,
                mb_y,
                sub_x,
                sub_y,
                mv_l0,
            );
            let b = build_inter_pred_chroma_4x4(
                ref_plane_l1,
                chroma_w_l1,
                chroma_h_l1,
                mb_x,
                mb_y,
                sub_x,
                sub_y,
                mv_l1,
            );
            let mut out = [0i32; 16];
            for i in 0..16 {
                out[i] = (a[i] + b[i] + 1) >> 1;
            }
            out
        }
        BSubMbCell::Direct => build_inter_pred_chroma_4x4(
            ref_plane_l0,
            chroma_w_l0,
            chroma_h_l0,
            mb_x,
            mb_y,
            sub_x,
            sub_y,
            Mv::ZERO,
        ),
    }
}

/// Round-25 — pick the best per-cell `BSubMbCell` for one 8x8 quadrant
/// of a `B_8x8` MB by minimising a Lagrangian SAD with rate cost
/// approximations:
///
///   * Direct: 0-bit headroom (no MVs / refs / sub_mb_type overhead vs
///     B_Direct_8x8 baseline).
///   * L0: ~6 bits worth of SAD penalty (sub_mb_type overhead +
///     mvd_l0 (.x, .y) at se(0) = 2 bits, plus typical small se).
///   * L1: ~6 bits.
///   * Bi:  ~12 bits (two mvds + larger sub_mb_type).
///
/// The penalty multiplier scales with QP via λ. We use the same
/// `λ = (qp_y / 6).clamp(0, 8)` family as the round-21..24 selectors.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pick_best_b8x8_cell(
    src_y: &[u8],
    src_stride: usize,
    ref_l0: &EncodedFrameRef<'_>,
    ref_l1: &EncodedFrameRef<'_>,
    mb_x: usize,
    mb_y: usize,
    sub_x: usize,
    sub_y: usize,
    direct_pred: &[i32; 64],
    me_l0: Mv,
    me_l1: Mv,
    qp_y: i32,
) -> (BSubMbCell, Mv, Mv, [i32; 64], u32) {
    let off_x = sub_x * 8;
    let off_y = sub_y * 8;

    // Direct candidate — caller-supplied predictor (already built from
    // the §8.4.1.2.2 derivation's per-8x8 MVs).
    let sad_direct = sad_8x8_pred(src_y, src_stride, mb_x, mb_y, off_x, off_y, direct_pred);

    // L0/L1/Bi candidates.
    let pred_l0 = build_b_explicit_8x8_cell_luma(
        BSubMbCell::L0,
        ref_l0,
        ref_l1,
        mb_x,
        mb_y,
        sub_x,
        sub_y,
        me_l0,
        me_l1,
    );
    let sad_l0 = sad_8x8_pred(src_y, src_stride, mb_x, mb_y, off_x, off_y, &pred_l0);

    let pred_l1 = build_b_explicit_8x8_cell_luma(
        BSubMbCell::L1,
        ref_l0,
        ref_l1,
        mb_x,
        mb_y,
        sub_x,
        sub_y,
        me_l0,
        me_l1,
    );
    let sad_l1 = sad_8x8_pred(src_y, src_stride, mb_x, mb_y, off_x, off_y, &pred_l1);

    let pred_bi = build_b_explicit_8x8_cell_luma(
        BSubMbCell::Bi,
        ref_l0,
        ref_l1,
        mb_x,
        mb_y,
        sub_x,
        sub_y,
        me_l0,
        me_l1,
    );
    let sad_bi = sad_8x8_pred(src_y, src_stride, mb_x, mb_y, off_x, off_y, &pred_bi);

    // Rate cost approximations (bits) per cell, vs the all-Direct baseline.
    let lambda = (qp_y / 6).clamp(0, 8) as u32;
    let cost_direct = 0u32;
    let cost_l0 = (lambda + 1) * 6;
    let cost_l1 = (lambda + 1) * 6;
    let cost_bi = (lambda + 1) * 12;

    let j_direct = sad_direct.saturating_add(cost_direct);
    let j_l0 = sad_l0.saturating_add(cost_l0);
    let j_l1 = sad_l1.saturating_add(cost_l1);
    let j_bi = sad_bi.saturating_add(cost_bi);

    // Tie-break order: Direct → L0 → L1 → Bi (mirrors round-21 / round-22
    // selector preferences for cheaper syntax).
    let mut best = (
        BSubMbCell::Direct,
        Mv::ZERO,
        Mv::ZERO,
        *direct_pred,
        j_direct,
        sad_direct,
    );
    if j_l0 < best.4 {
        best = (BSubMbCell::L0, me_l0, Mv::ZERO, pred_l0, j_l0, sad_l0);
    }
    if j_l1 < best.4 {
        best = (BSubMbCell::L1, Mv::ZERO, me_l1, pred_l1, j_l1, sad_l1);
    }
    if j_bi < best.4 {
        best = (BSubMbCell::Bi, me_l0, me_l1, pred_bi, j_bi, sad_bi);
    }
    (best.0, best.1, best.2, best.3, best.5)
}

/// SAD between a 16x16 source block (at MB `(mb_x, mb_y)` of `src_y`) and
/// a 16x16 luma predictor buffer in row-major i32 (clamped to 0..=255 to
/// match what the decoder writes after weighted-pred clipping).
fn sad_16x16_pred(
    src_y: &[u8],
    src_stride: usize,
    mb_x: usize,
    mb_y: usize,
    pred: &[i32; 256],
) -> u32 {
    let mut s = 0u32;
    for j in 0..16usize {
        for i in 0..16usize {
            let src = src_y[(mb_y * 16 + j) * src_stride + mb_x * 16 + i] as i32;
            let p = pred[j * 16 + i].clamp(0, 255);
            s = s.saturating_add((src - p).unsigned_abs());
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Round 22 — B-slice 16x8 / 8x16 partition predictors and MVP helpers.
// ---------------------------------------------------------------------------

/// Round-22 — build a luma inter predictor for one 16x8 partition (top
/// or bottom) of a B MB, optionally combining L0 and L1 per
/// `BPartPred`. Output is `[i32; 128]` in row-major (8 rows of 16
/// samples). `partition_y_off` ∈ {0, 8} selects the half within the MB.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_b_partition_pred_luma_16x8(
    pred: BPartPred,
    ref_l0: &EncodedFrameRef<'_>,
    ref_l1: &EncodedFrameRef<'_>,
    mb_x: usize,
    mb_y: usize,
    partition_y_off: usize,
    mv_l0: Mv,
    mv_l1: Mv,
) -> [i32; 128] {
    let mut out = [0i32; 128];
    let l0 = if pred.uses_l0() {
        Some(build_inter_pred_luma_partition(
            ref_l0.recon_y,
            ref_l0.width,
            ref_l0.height,
            mb_x,
            mb_y,
            0,
            partition_y_off,
            16,
            8,
            mv_l0,
        ))
    } else {
        None
    };
    let l1 = if pred.uses_l1() {
        Some(build_inter_pred_luma_partition(
            ref_l1.recon_y,
            ref_l1.width,
            ref_l1.height,
            mb_x,
            mb_y,
            0,
            partition_y_off,
            16,
            8,
            mv_l1,
        ))
    } else {
        None
    };
    match pred {
        BPartPred::L0 => out.copy_from_slice(&l0.unwrap()),
        BPartPred::L1 => out.copy_from_slice(&l1.unwrap()),
        BPartPred::Bi => {
            // §8.4.2.3.1 eq. 8-273.
            let a = l0.unwrap();
            let b = l1.unwrap();
            for i in 0..128 {
                out[i] = (a[i] + b[i] + 1) >> 1;
            }
        }
    }
    out
}

/// Round-22 — build a luma inter predictor for one 8x16 partition (left
/// or right) of a B MB. Output is `[i32; 128]` in row-major (16 rows of
/// 8 samples). `partition_x_off` ∈ {0, 8} selects the half within the
/// MB.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_b_partition_pred_luma_8x16(
    pred: BPartPred,
    ref_l0: &EncodedFrameRef<'_>,
    ref_l1: &EncodedFrameRef<'_>,
    mb_x: usize,
    mb_y: usize,
    partition_x_off: usize,
    mv_l0: Mv,
    mv_l1: Mv,
) -> [i32; 128] {
    let mut out = [0i32; 128];
    let l0 = if pred.uses_l0() {
        Some(build_inter_pred_luma_partition(
            ref_l0.recon_y,
            ref_l0.width,
            ref_l0.height,
            mb_x,
            mb_y,
            partition_x_off,
            0,
            8,
            16,
            mv_l0,
        ))
    } else {
        None
    };
    let l1 = if pred.uses_l1() {
        Some(build_inter_pred_luma_partition(
            ref_l1.recon_y,
            ref_l1.width,
            ref_l1.height,
            mb_x,
            mb_y,
            partition_x_off,
            0,
            8,
            16,
            mv_l1,
        ))
    } else {
        None
    };
    match pred {
        BPartPred::L0 => out.copy_from_slice(&l0.unwrap()),
        BPartPred::L1 => out.copy_from_slice(&l1.unwrap()),
        BPartPred::Bi => {
            let a = l0.unwrap();
            let b = l1.unwrap();
            for i in 0..128 {
                out[i] = (a[i] + b[i] + 1) >> 1;
            }
        }
    }
    out
}

/// Round-22 — generic luma inter-pred helper for an arbitrary
/// rectangular partition of an MB. `(off_x, off_y)` is the partition's
/// top-left within the MB in luma samples; `(w, h)` is the partition
/// dimensions (one of (16, 8), (8, 16), (16, 16), (8, 8)). Output is
/// `[i32; w*h]` packed row-major into a `Vec<i32>` (length 128 for the
/// 16x8/8x16 cases used here).
#[allow(clippy::too_many_arguments)]
fn build_inter_pred_luma_partition(
    ref_y: &[u8],
    ref_w: u32,
    ref_h: u32,
    mb_x: usize,
    mb_y: usize,
    off_x: usize,
    off_y: usize,
    w: u32,
    h: u32,
    mv: Mv,
) -> Vec<i32> {
    let stride = ref_w as usize;
    let h_us = ref_h as usize;
    let ref_i32: Vec<i32> = ref_y
        .iter()
        .take(stride * h_us)
        .map(|&v| v as i32)
        .collect();

    let mv_int_x = mv.x >> 2;
    let mv_int_y = mv.y >> 2;
    let x_frac = (mv.x & 3) as u8;
    let y_frac = (mv.y & 3) as u8;
    let int_x = (mb_x * 16 + off_x) as i32 + mv_int_x;
    let int_y = (mb_y * 16 + off_y) as i32 + mv_int_y;

    let mut dst = vec![0i32; (w * h) as usize];
    interpolate_luma(
        &ref_i32, stride, stride, h_us, int_x, int_y, x_frac, y_frac, w, h, 8, &mut dst, w as usize,
    )
    .expect("luma interpolation partition");
    dst
}

/// Round-22 — generic chroma inter-pred for an arbitrary partition.
/// `(off_x, off_y)` is the partition's top-left in luma samples; `(w,
/// h)` is the partition's chroma dimensions (so a 16x8 luma partition
/// gives an 8x4 chroma partition). 4:2:0 only.
#[allow(clippy::too_many_arguments)]
fn build_inter_pred_chroma_partition(
    ref_c: &[u8],
    ref_cw: u32,
    ref_ch: u32,
    mb_x: usize,
    mb_y: usize,
    off_x_luma: usize,
    off_y_luma: usize,
    w_chroma: u32,
    h_chroma: u32,
    mv: Mv,
) -> Vec<i32> {
    let stride = ref_cw as usize;
    let h_us = ref_ch as usize;
    let ref_i32: Vec<i32> = ref_c
        .iter()
        .take(stride * h_us)
        .map(|&v| v as i32)
        .collect();

    // §8.4.1.4 — 4:2:0 chroma uses luma MV directly with 1/8-pel
    // fractional positions.
    let int_x = (mb_x * 8 + off_x_luma / 2) as i32 + (mv.x >> 3);
    let int_y = (mb_y * 8 + off_y_luma / 2) as i32 + (mv.y >> 3);
    let x_frac = (mv.x & 7) as u8;
    let y_frac = (mv.y & 7) as u8;

    let mut dst = vec![0i32; (w_chroma * h_chroma) as usize];
    interpolate_chroma(
        &ref_i32,
        stride,
        stride,
        h_us,
        int_x,
        int_y,
        x_frac,
        y_frac,
        w_chroma,
        h_chroma,
        8,
        &mut dst,
        w_chroma as usize,
    )
    .expect("chroma interpolation partition");
    dst
}

/// Round-22 — build chroma predictors for one B-slice 16x8 partition
/// (Cb + Cr). 4:2:0 → chroma partition is 8x4 (`[i32; 32]` row-major).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_b_partition_pred_chroma_16x8(
    pred: BPartPred,
    ref_l0: &EncodedFrameRef<'_>,
    ref_l1: &EncodedFrameRef<'_>,
    mb_x: usize,
    mb_y: usize,
    partition_y_off_luma: usize,
    mv_l0: Mv,
    mv_l1: Mv,
) -> ([i32; 32], [i32; 32]) {
    let cw_l0 = ref_l0.width / 2;
    let ch_l0 = ref_l0.height / 2;
    let cw_l1 = ref_l1.width / 2;
    let ch_l1 = ref_l1.height / 2;
    let make = |plane_l0: &[u8], plane_l1: &[u8]| -> [i32; 32] {
        let mut out = [0i32; 32];
        let a = if pred.uses_l0() {
            Some(build_inter_pred_chroma_partition(
                plane_l0,
                cw_l0,
                ch_l0,
                mb_x,
                mb_y,
                0,
                partition_y_off_luma,
                8,
                4,
                mv_l0,
            ))
        } else {
            None
        };
        let b = if pred.uses_l1() {
            Some(build_inter_pred_chroma_partition(
                plane_l1,
                cw_l1,
                ch_l1,
                mb_x,
                mb_y,
                0,
                partition_y_off_luma,
                8,
                4,
                mv_l1,
            ))
        } else {
            None
        };
        match pred {
            BPartPred::L0 => out.copy_from_slice(&a.unwrap()),
            BPartPred::L1 => out.copy_from_slice(&b.unwrap()),
            BPartPred::Bi => {
                let a = a.unwrap();
                let b = b.unwrap();
                for i in 0..32 {
                    out[i] = (a[i] + b[i] + 1) >> 1;
                }
            }
        }
        out
    };
    (
        make(ref_l0.recon_u, ref_l1.recon_u),
        make(ref_l0.recon_v, ref_l1.recon_v),
    )
}

/// Round-22 — build chroma predictors for one B-slice 8x16 partition
/// (Cb + Cr). 4:2:0 → chroma partition is 4x8 (`[i32; 32]` row-major).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_b_partition_pred_chroma_8x16(
    pred: BPartPred,
    ref_l0: &EncodedFrameRef<'_>,
    ref_l1: &EncodedFrameRef<'_>,
    mb_x: usize,
    mb_y: usize,
    partition_x_off_luma: usize,
    mv_l0: Mv,
    mv_l1: Mv,
) -> ([i32; 32], [i32; 32]) {
    let cw_l0 = ref_l0.width / 2;
    let ch_l0 = ref_l0.height / 2;
    let cw_l1 = ref_l1.width / 2;
    let ch_l1 = ref_l1.height / 2;
    let make = |plane_l0: &[u8], plane_l1: &[u8]| -> [i32; 32] {
        let mut out = [0i32; 32];
        let a = if pred.uses_l0() {
            Some(build_inter_pred_chroma_partition(
                plane_l0,
                cw_l0,
                ch_l0,
                mb_x,
                mb_y,
                partition_x_off_luma,
                0,
                4,
                8,
                mv_l0,
            ))
        } else {
            None
        };
        let b = if pred.uses_l1() {
            Some(build_inter_pred_chroma_partition(
                plane_l1,
                cw_l1,
                ch_l1,
                mb_x,
                mb_y,
                partition_x_off_luma,
                0,
                4,
                8,
                mv_l1,
            ))
        } else {
            None
        };
        match pred {
            BPartPred::L0 => out.copy_from_slice(&a.unwrap()),
            BPartPred::L1 => out.copy_from_slice(&b.unwrap()),
            BPartPred::Bi => {
                let a = a.unwrap();
                let b = b.unwrap();
                for i in 0..32 {
                    out[i] = (a[i] + b[i] + 1) >> 1;
                }
            }
        }
        out
    };
    (
        make(ref_l0.recon_u, ref_l1.recon_u),
        make(ref_l0.recon_v, ref_l1.recon_v),
    )
}

/// SAD between an arbitrary-sized source block (at MB-relative offset
/// `(off_x, off_y)` of the MB at `(mb_x, mb_y)`) and a `w*h` luma
/// predictor buffer (row-major i32 with stride = w, clamped to
/// 0..=255).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sad_partition_pred(
    src_y: &[u8],
    src_stride: usize,
    mb_x: usize,
    mb_y: usize,
    off_x: usize,
    off_y: usize,
    w: usize,
    h: usize,
    pred: &[i32],
) -> u32 {
    let mut s = 0u32;
    for j in 0..h {
        for i in 0..w {
            let src = src_y[(mb_y * 16 + off_y + j) * src_stride + mb_x * 16 + off_x + i] as i32;
            let p = pred[j * w + i].clamp(0, 255);
            s = s.saturating_add((src - p).unsigned_abs());
        }
    }
    s
}

/// Round-22 — derive the §8.4.1.3 mvpLX for one 16x8 partition (top or
/// bottom) of a B MB at `(mb_x, mb_y)` against list `list_idx` ∈
/// {0, 1}. `is_top` selects which 16x8 half. Within-MB neighbour reads
/// use `inflight_*` arrays that carry the already-decoded partitions
/// of the **current** MB.
///
/// The shape directs the §8.4.1.3 directional shortcut:
/// `Partition16x8Top` for top, `Partition16x8Bottom` for bottom. The
/// `derive_mvpred_with_d` median fallback consults A/B/C neighbours of
/// the partition's top-left 4x4 plus the optional D (above-left).
#[allow(clippy::too_many_arguments)]
fn mvp_for_b_16x8_partition(
    grid: &MvGrid,
    mb_x: usize,
    mb_y: usize,
    is_top: bool,
    ref_idx: i32,
    inflight_avail: [bool; 4],
    mv_inflight: [Mv; 4],
    ref_inflight: [i32; 4],
) -> Mv {
    // Partition top-left in 4x4 units. Top: (0, 0); bottom: (0, 2).
    let by = if is_top { 0i32 } else { 2 };
    // A at (-1, by); B at (0, by - 1); C at (4, by - 1); D at (-1, by - 1).
    // Partition width = 16 px = 4 4x4 → C at column index 4 (above-right
    // neighbour MB). Partition height irrelevant for 16x8 neighbours.
    let a = neighbour_8x8_partition_mv(
        grid,
        mb_x,
        mb_y,
        -1,
        by,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let b = neighbour_8x8_partition_mv(
        grid,
        mb_x,
        mb_y,
        0,
        by - 1,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let c = neighbour_8x8_partition_mv(
        grid,
        mb_x,
        mb_y,
        4,
        by - 1,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let d = neighbour_8x8_partition_mv(
        grid,
        mb_x,
        mb_y,
        -1,
        by - 1,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let shape = if is_top {
        crate::mv_deriv::MvpredShape::Partition16x8Top
    } else {
        crate::mv_deriv::MvpredShape::Partition16x8Bottom
    };
    let inputs = MvpredInputs::from_abc(a, b, c, ref_idx, shape);
    crate::mv_deriv::derive_mvpred_with_d(&inputs, d)
}

/// Round-22 — derive the §8.4.1.3 mvpLX for one 8x16 partition (left
/// or right). Mirrors [`mvp_for_b_16x8_partition`].
#[allow(clippy::too_many_arguments)]
fn mvp_for_b_8x16_partition(
    grid: &MvGrid,
    mb_x: usize,
    mb_y: usize,
    is_left: bool,
    ref_idx: i32,
    inflight_avail: [bool; 4],
    mv_inflight: [Mv; 4],
    ref_inflight: [i32; 4],
) -> Mv {
    // Partition top-left in 4x4 units. Left: (0, 0); right: (2, 0).
    let bx = if is_left { 0i32 } else { 2 };
    // A at (bx - 1, 0); B at (bx, -1); C at (bx + 2, -1); D at (bx - 1, -1).
    // Partition width = 8 px = 2 4x4 → C at column (bx + 2) (above MB
    // for left, above-right MB for right).
    let a = neighbour_8x8_partition_mv(
        grid,
        mb_x,
        mb_y,
        bx - 1,
        0,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let b = neighbour_8x8_partition_mv(
        grid,
        mb_x,
        mb_y,
        bx,
        -1,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let c = neighbour_8x8_partition_mv(
        grid,
        mb_x,
        mb_y,
        bx + 2,
        -1,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let d = neighbour_8x8_partition_mv(
        grid,
        mb_x,
        mb_y,
        bx - 1,
        -1,
        inflight_avail,
        mv_inflight,
        ref_inflight,
    );
    let shape = if is_left {
        crate::mv_deriv::MvpredShape::Partition8x16Left
    } else {
        crate::mv_deriv::MvpredShape::Partition8x16Right
    };
    let inputs = MvpredInputs::from_abc(a, b, c, ref_idx, shape);
    crate::mv_deriv::derive_mvpred_with_d(&inputs, d)
}

/// Round-22 — per-partition ME for one 16x8 or 8x16 partition (any
/// list). Runs `search_quarter_pel_8x8` on each of the (up to 2) 8x8
/// cells covered by the partition, and picks the candidate MV (one of
/// `mv_16x16` or per-cell-best) that gives the lowest partition-wide
/// SAD against `ref_y`. Returns the best partition MV.
///
/// The encoder tries three candidates per (partition, list):
///   * The 16x16 ME result (`mv_16x16`).
///   * The first 8x8 cell's qpel ME result.
///   * The second 8x8 cell's qpel ME result.
///
/// SAD is measured over the full partition area against the qpel-
/// interpolated predictor. This costs 3 partition-SADs per (partition,
/// list) but reuses ME results we already need anyway.
#[allow(clippy::too_many_arguments)]
fn best_partition_mv(
    src_y: &[u8],
    src_stride: usize,
    src_w: u32,
    src_h: u32,
    ref_y: &[u8],
    ref_stride: usize,
    ref_w: u32,
    ref_h: u32,
    mb_x: usize,
    mb_y: usize,
    off_x: usize,
    off_y: usize,
    part_w: u32,
    part_h: u32,
    candidates: &[Mv],
) -> (Mv, u32) {
    let mut best_mv = candidates[0];
    let mut best_sad = u32::MAX;
    for &mv in candidates {
        let pred = build_inter_pred_luma_partition(
            ref_y, ref_w, ref_h, mb_x, mb_y, off_x, off_y, part_w, part_h, mv,
        );
        let sad = sad_partition_pred(
            src_y,
            src_stride,
            mb_x,
            mb_y,
            off_x,
            off_y,
            part_w as usize,
            part_h as usize,
            &pred,
        );
        let _ = (src_w, src_h, ref_stride);
        if sad < best_sad {
            best_sad = sad;
            best_mv = mv;
        }
    }
    (best_mv, best_sad)
}

/// Round-22 — per-partition SAD for a given (mode, mv_l0, mv_l1).
/// Builds the partition's L0/L1/Bi predictor, returns its SAD against
/// source.
#[allow(clippy::too_many_arguments)]
pub(crate) fn partition_sad_b_16x8(
    pred: BPartPred,
    src_y: &[u8],
    src_stride: usize,
    ref_l0: &EncodedFrameRef<'_>,
    ref_l1: &EncodedFrameRef<'_>,
    mb_x: usize,
    mb_y: usize,
    off_y: usize,
    mv_l0: Mv,
    mv_l1: Mv,
) -> u32 {
    let p = build_b_partition_pred_luma_16x8(pred, ref_l0, ref_l1, mb_x, mb_y, off_y, mv_l0, mv_l1);
    sad_partition_pred(src_y, src_stride, mb_x, mb_y, 0, off_y, 16, 8, &p)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn partition_sad_b_8x16(
    pred: BPartPred,
    src_y: &[u8],
    src_stride: usize,
    ref_l0: &EncodedFrameRef<'_>,
    ref_l1: &EncodedFrameRef<'_>,
    mb_x: usize,
    mb_y: usize,
    off_x: usize,
    mv_l0: Mv,
    mv_l1: Mv,
) -> u32 {
    let p = build_b_partition_pred_luma_8x16(pred, ref_l0, ref_l1, mb_x, mb_y, off_x, mv_l0, mv_l1);
    sad_partition_pred(src_y, src_stride, mb_x, mb_y, off_x, 0, 8, 16, &p)
}

/// Round-22 — choose the best per-partition (mode, mv_l0, mv_l1) for
/// one 16x8 or 8x16 partition given pre-computed per-list candidate MVs.
pub(crate) fn best_partition_mode_b(
    candidates_l0: &[Mv],
    candidates_l1: &[Mv],
    eval_sad: impl Fn(BPartPred, Mv, Mv) -> u32,
) -> (BPartPred, Mv, Mv, u32) {
    let mut best = (BPartPred::L0, Mv::ZERO, Mv::ZERO, u32::MAX);
    // L0-only — try each L0 candidate.
    for &mv0 in candidates_l0 {
        let s = eval_sad(BPartPred::L0, mv0, Mv::ZERO);
        if s < best.3 {
            best = (BPartPred::L0, mv0, Mv::ZERO, s);
        }
    }
    // L1-only — try each L1 candidate.
    for &mv1 in candidates_l1 {
        let s = eval_sad(BPartPred::L1, Mv::ZERO, mv1);
        if s < best.3 {
            best = (BPartPred::L1, Mv::ZERO, mv1, s);
        }
    }
    // Bi — try each L0 × L1 combination.
    for &mv0 in candidates_l0 {
        for &mv1 in candidates_l1 {
            let s = eval_sad(BPartPred::Bi, mv0, mv1);
            if s < best.3 {
                best = (BPartPred::Bi, mv0, mv1, s);
            }
        }
    }
    best
}

/// Round-22 — which partition shape was picked for a B MB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PartitionShape {
    /// Two 16x8 halves stacked vertically (top, bottom).
    P16x8,
    /// Two 8x16 halves side-by-side (left, right).
    P8x16,
}

/// Round-22 — captured per-partition decision for a B 16x8 / 8x16 MB.
/// `mode_a` / `mv_*_a` is the first partition (top for 16x8, left for
/// 8x16); `mode_b` / `mv_*_b` is the second (bottom / right).
#[derive(Debug, Clone, Copy)]
struct PartitionChoiceB {
    shape: PartitionShape,
    mode_a: BPartPred,
    mode_b: BPartPred,
    mv_l0_a: Mv,
    mv_l1_a: Mv,
    mv_l0_b: Mv,
    mv_l1_b: Mv,
}

/// Round-22 — build a 16x16 composite predictor (luma + Cb + Cr) from a
/// partition decision, by stitching the two per-partition predictors
/// into the relevant halves of a 16x16 buffer.
fn build_partition_composite_b(
    pc: &PartitionChoiceB,
    ref_l0: &EncodedFrameRef<'_>,
    ref_l1: &EncodedFrameRef<'_>,
    mb_x: usize,
    mb_y: usize,
) -> ([i32; 256], [i32; 64], [i32; 64]) {
    let mut pred_y = [0i32; 256];
    let mut pred_u = [0i32; 64];
    let mut pred_v = [0i32; 64];
    match pc.shape {
        PartitionShape::P16x8 => {
            let py0 = build_b_partition_pred_luma_16x8(
                pc.mode_a, ref_l0, ref_l1, mb_x, mb_y, 0, pc.mv_l0_a, pc.mv_l1_a,
            );
            let py1 = build_b_partition_pred_luma_16x8(
                pc.mode_b, ref_l0, ref_l1, mb_x, mb_y, 8, pc.mv_l0_b, pc.mv_l1_b,
            );
            for j in 0..8usize {
                for i in 0..16usize {
                    pred_y[j * 16 + i] = py0[j * 16 + i];
                    pred_y[(8 + j) * 16 + i] = py1[j * 16 + i];
                }
            }
            // Chroma 4:2:0 → 8x4 each half.
            let (cb0, cr0) = build_b_partition_pred_chroma_16x8(
                pc.mode_a, ref_l0, ref_l1, mb_x, mb_y, 0, pc.mv_l0_a, pc.mv_l1_a,
            );
            let (cb1, cr1) = build_b_partition_pred_chroma_16x8(
                pc.mode_b, ref_l0, ref_l1, mb_x, mb_y, 8, pc.mv_l0_b, pc.mv_l1_b,
            );
            for j in 0..4usize {
                for i in 0..8usize {
                    pred_u[j * 8 + i] = cb0[j * 8 + i];
                    pred_v[j * 8 + i] = cr0[j * 8 + i];
                    pred_u[(4 + j) * 8 + i] = cb1[j * 8 + i];
                    pred_v[(4 + j) * 8 + i] = cr1[j * 8 + i];
                }
            }
        }
        PartitionShape::P8x16 => {
            let py0 = build_b_partition_pred_luma_8x16(
                pc.mode_a, ref_l0, ref_l1, mb_x, mb_y, 0, pc.mv_l0_a, pc.mv_l1_a,
            );
            let py1 = build_b_partition_pred_luma_8x16(
                pc.mode_b, ref_l0, ref_l1, mb_x, mb_y, 8, pc.mv_l0_b, pc.mv_l1_b,
            );
            for j in 0..16usize {
                for i in 0..8usize {
                    pred_y[j * 16 + i] = py0[j * 8 + i];
                    pred_y[j * 16 + 8 + i] = py1[j * 8 + i];
                }
            }
            // Chroma 4:2:0 → 4x8 each half.
            let (cb0, cr0) = build_b_partition_pred_chroma_8x16(
                pc.mode_a, ref_l0, ref_l1, mb_x, mb_y, 0, pc.mv_l0_a, pc.mv_l1_a,
            );
            let (cb1, cr1) = build_b_partition_pred_chroma_8x16(
                pc.mode_b, ref_l0, ref_l1, mb_x, mb_y, 8, pc.mv_l0_b, pc.mv_l1_b,
            );
            for j in 0..8usize {
                for i in 0..4usize {
                    pred_u[j * 8 + i] = cb0[j * 4 + i];
                    pred_v[j * 8 + i] = cr0[j * 4 + i];
                    pred_u[j * 8 + 4 + i] = cb1[j * 4 + i];
                    pred_v[j * 8 + 4 + i] = cr1[j * 4 + i];
                }
            }
        }
    }
    (pred_y, pred_u, pred_v)
}

/// Round-22 — derive per-partition MVDs for a B 16x8 / 8x16 MB using
/// the §8.4.1.3 partition-shape MVP derivation. Returns
/// `(mvd_l0[2], mvd_l1[2])`.
///
/// For the second partition (bottom of 16x8 / right of 8x16) the §8.4.1.3
/// MVP can read from the first partition's MVs (which are part of the
/// SAME MB, already decoded in raster order). We surface those via
/// `inflight_*` arrays so the within-MB neighbour lookup in
/// `mvp_for_b_*_partition` finds them.
/// Per-partition MVD tuple (one (x, y) per of the two partitions).
type PartitionMvds = [(i32, i32); 2];

fn partition_mvds_b(
    pc: &PartitionChoiceB,
    mv_grid_l0: &MvGrid,
    mv_grid_l1: &MvGrid,
    mb_x: usize,
    mb_y: usize,
) -> (PartitionMvds, PartitionMvds) {
    let mut mvd_l0 = [(0i32, 0i32); 2];
    let mut mvd_l1 = [(0i32, 0i32); 2];
    let inflight_avail = [false; 4];
    let mv_inflight = [Mv::ZERO; 4];
    let ref_inflight = [-1i32; 4];
    match pc.shape {
        PartitionShape::P16x8 => {
            // Partition 0 (top).
            if pc.mode_a.uses_l0() {
                let mvp = mvp_for_b_16x8_partition(
                    mv_grid_l0,
                    mb_x,
                    mb_y,
                    true,
                    0,
                    inflight_avail,
                    mv_inflight,
                    ref_inflight,
                );
                mvd_l0[0] = (pc.mv_l0_a.x - mvp.x, pc.mv_l0_a.y - mvp.y);
            }
            if pc.mode_a.uses_l1() {
                let mvp = mvp_for_b_16x8_partition(
                    mv_grid_l1,
                    mb_x,
                    mb_y,
                    true,
                    0,
                    inflight_avail,
                    mv_inflight,
                    ref_inflight,
                );
                mvd_l1[0] = (pc.mv_l1_a.x - mvp.x, pc.mv_l1_a.y - mvp.y);
            }
            // Partition 1 (bottom) — top covers 8x8 #0 and #1.
            let inflight_avail_bot = [true, true, false, false];
            let mv_inflight_bot_l0 = [pc.mv_l0_a, pc.mv_l0_a, Mv::ZERO, Mv::ZERO];
            let mv_inflight_bot_l1 = [pc.mv_l1_a, pc.mv_l1_a, Mv::ZERO, Mv::ZERO];
            let ref_inflight_bot_l0 = [
                if pc.mode_a.uses_l0() { 0 } else { -1 },
                if pc.mode_a.uses_l0() { 0 } else { -1 },
                -1,
                -1,
            ];
            let ref_inflight_bot_l1 = [
                if pc.mode_a.uses_l1() { 0 } else { -1 },
                if pc.mode_a.uses_l1() { 0 } else { -1 },
                -1,
                -1,
            ];
            if pc.mode_b.uses_l0() {
                let mvp = mvp_for_b_16x8_partition(
                    mv_grid_l0,
                    mb_x,
                    mb_y,
                    false,
                    0,
                    inflight_avail_bot,
                    mv_inflight_bot_l0,
                    ref_inflight_bot_l0,
                );
                mvd_l0[1] = (pc.mv_l0_b.x - mvp.x, pc.mv_l0_b.y - mvp.y);
            }
            if pc.mode_b.uses_l1() {
                let mvp = mvp_for_b_16x8_partition(
                    mv_grid_l1,
                    mb_x,
                    mb_y,
                    false,
                    0,
                    inflight_avail_bot,
                    mv_inflight_bot_l1,
                    ref_inflight_bot_l1,
                );
                mvd_l1[1] = (pc.mv_l1_b.x - mvp.x, pc.mv_l1_b.y - mvp.y);
            }
        }
        PartitionShape::P8x16 => {
            if pc.mode_a.uses_l0() {
                let mvp = mvp_for_b_8x16_partition(
                    mv_grid_l0,
                    mb_x,
                    mb_y,
                    true,
                    0,
                    inflight_avail,
                    mv_inflight,
                    ref_inflight,
                );
                mvd_l0[0] = (pc.mv_l0_a.x - mvp.x, pc.mv_l0_a.y - mvp.y);
            }
            if pc.mode_a.uses_l1() {
                let mvp = mvp_for_b_8x16_partition(
                    mv_grid_l1,
                    mb_x,
                    mb_y,
                    true,
                    0,
                    inflight_avail,
                    mv_inflight,
                    ref_inflight,
                );
                mvd_l1[0] = (pc.mv_l1_a.x - mvp.x, pc.mv_l1_a.y - mvp.y);
            }
            // Partition 1 (right) — left covers 8x8 #0 and #2.
            let inflight_avail_right = [true, false, true, false];
            let mv_inflight_right_l0 = [pc.mv_l0_a, Mv::ZERO, pc.mv_l0_a, Mv::ZERO];
            let mv_inflight_right_l1 = [pc.mv_l1_a, Mv::ZERO, pc.mv_l1_a, Mv::ZERO];
            let ref_inflight_right_l0 = [
                if pc.mode_a.uses_l0() { 0 } else { -1 },
                -1,
                if pc.mode_a.uses_l0() { 0 } else { -1 },
                -1,
            ];
            let ref_inflight_right_l1 = [
                if pc.mode_a.uses_l1() { 0 } else { -1 },
                -1,
                if pc.mode_a.uses_l1() { 0 } else { -1 },
                -1,
            ];
            if pc.mode_b.uses_l0() {
                let mvp = mvp_for_b_8x16_partition(
                    mv_grid_l0,
                    mb_x,
                    mb_y,
                    false,
                    0,
                    inflight_avail_right,
                    mv_inflight_right_l0,
                    ref_inflight_right_l0,
                );
                mvd_l0[1] = (pc.mv_l0_b.x - mvp.x, pc.mv_l0_b.y - mvp.y);
            }
            if pc.mode_b.uses_l1() {
                let mvp = mvp_for_b_8x16_partition(
                    mv_grid_l1,
                    mb_x,
                    mb_y,
                    false,
                    0,
                    inflight_avail_right,
                    mv_inflight_right_l1,
                    ref_inflight_right_l1,
                );
                mvd_l1[1] = (pc.mv_l1_b.x - mvp.x, pc.mv_l1_b.y - mvp.y);
            }
        }
    }
    (mvd_l0, mvd_l1)
}

/// Round-22 — derive (per-8x8 MV, per-8x8 ref_idx, list_used) for the
/// MvGrid update of one list. For partition modes the per-8x8 slots
/// carry per-partition data; for 16x16 / Direct, all slots share the
/// single MV.
fn grid_state_b(
    pc: Option<&PartitionChoiceB>,
    is_l0: bool,
    pred: BPred16x16,
    eff_mv: Mv,
) -> ([Mv; 4], [i32; 4], bool) {
    if let Some(pc) = pc {
        let (mv_a, mv_b, used_a, used_b) = if is_l0 {
            (
                pc.mv_l0_a,
                pc.mv_l0_b,
                pc.mode_a.uses_l0(),
                pc.mode_b.uses_l0(),
            )
        } else {
            (
                pc.mv_l1_a,
                pc.mv_l1_b,
                pc.mode_a.uses_l1(),
                pc.mode_b.uses_l1(),
            )
        };
        let any_used = used_a || used_b;
        let mut mvs = [Mv::ZERO; 4];
        let mut refs = [-1i32; 4];
        match pc.shape {
            PartitionShape::P16x8 => {
                // Top covers 8x8 #0/#1; bottom #2/#3.
                if used_a {
                    mvs[0] = mv_a;
                    mvs[1] = mv_a;
                    refs[0] = 0;
                    refs[1] = 0;
                }
                if used_b {
                    mvs[2] = mv_b;
                    mvs[3] = mv_b;
                    refs[2] = 0;
                    refs[3] = 0;
                }
            }
            PartitionShape::P8x16 => {
                // Left covers 8x8 #0/#2; right #1/#3.
                if used_a {
                    mvs[0] = mv_a;
                    mvs[2] = mv_a;
                    refs[0] = 0;
                    refs[2] = 0;
                }
                if used_b {
                    mvs[1] = mv_b;
                    mvs[3] = mv_b;
                    refs[1] = 0;
                    refs[3] = 0;
                }
            }
        }
        (mvs, refs, any_used)
    } else {
        let used = if is_l0 {
            pred.uses_l0()
        } else {
            pred.uses_l1()
        };
        if used {
            ([eff_mv; 4], [0; 4], true)
        } else {
            ([Mv::ZERO; 4], [-1; 4], false)
        }
    }
}

/// Round-22 — build the per-4x4 MV / per-8x8 ref / per-8x8 POC arrays
/// for the §8.7 deblocker. For partition modes the 4x4 blocks in each
/// half carry that half's MV; for 16x16 / Direct the single MV applies
/// everywhere.
fn deblock_mv_arr_b(
    pc: Option<&PartitionChoiceB>,
    is_l0: bool,
    pred: BPred16x16,
    eff_mv: Mv,
    poc_sentinel: i32,
) -> ([(i16, i16); 16], [i8; 4], [i32; 4]) {
    if let Some(pc) = pc {
        let (mv_a, mv_b, used_a, used_b) = if is_l0 {
            (
                pc.mv_l0_a,
                pc.mv_l0_b,
                pc.mode_a.uses_l0(),
                pc.mode_b.uses_l0(),
            )
        } else {
            (
                pc.mv_l1_a,
                pc.mv_l1_b,
                pc.mode_a.uses_l1(),
                pc.mode_b.uses_l1(),
            )
        };
        let mv_a_t = (
            mv_a.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            mv_a.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
        );
        let mv_b_t = (
            mv_b.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            mv_b.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
        );
        let mut mvs = [(0i16, 0i16); 16];
        let mut refs = [-1i8; 4];
        let mut pocs = [i32::MIN; 4];
        // The `mvs` array is indexed by §6.4.3 Z-scan position 0..15
        // (matches `MbDeblockInfo::mv_l0` / decoder MbInfo layout). Map
        // each Z-scan index to its (4x4-bx, 4x4-by) coords via
        // `LUMA_4X4_BLK` so we correctly assign top/bottom (16x8) or
        // left/right (8x16) partition MVs.
        match pc.shape {
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
                pocs[0] = if used_a { poc_sentinel } else { i32::MIN };
                pocs[1] = if used_a { poc_sentinel } else { i32::MIN };
                pocs[2] = if used_b { poc_sentinel } else { i32::MIN };
                pocs[3] = if used_b { poc_sentinel } else { i32::MIN };
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
                pocs[0] = if used_a { poc_sentinel } else { i32::MIN };
                pocs[2] = if used_a { poc_sentinel } else { i32::MIN };
                pocs[1] = if used_b { poc_sentinel } else { i32::MIN };
                pocs[3] = if used_b { poc_sentinel } else { i32::MIN };
            }
        }
        (mvs, refs, pocs)
    } else {
        let used = if is_l0 {
            pred.uses_l0()
        } else {
            pred.uses_l1()
        };
        if used {
            let mv_t = (
                eff_mv.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                eff_mv.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            );
            ([mv_t; 16], [0; 4], [poc_sentinel; 4])
        } else {
            ([(0, 0); 16], [-1; 4], [i32::MIN; 4])
        }
    }
}

/// Round-23 — build the per-4x4 MV / per-8x8 ref / per-8x8 POC arrays
/// for the §8.7 deblocker on a `B_8x8` + 4× `B_Direct_8x8` macroblock.
/// Each 8x8 quadrant carries its own per-list MV (from the §8.4.1.2.2
/// derivation); within a quadrant all four 4x4 blocks share the
/// quadrant's MV (`direct_8x8_inference_flag = 1`). When `ref_idx` is
/// negative the list is unused for this MB and all per-4x4 MVs / refs /
/// pocs are zeroed-out / sentinel.
fn deblock_arrays_per_8x8_direct(
    mv_per_8x8: [Mv; 4],
    ref_idx: i32,
    poc_sentinel: i32,
) -> ([(i16, i16); 16], [i8; 4], [i32; 4]) {
    let used = ref_idx >= 0;
    if !used {
        return ([(0, 0); 16], [-1; 4], [i32::MIN; 4]);
    }
    let mut mvs = [(0i16, 0i16); 16];
    // §6.4.3 raster-Z 4x4 indexing; LUMA_4X4_BLK[blk] returns (4x4-bx,
    // 4x4-by) ∈ 0..4 × 0..4. The containing 8x8 quadrant is
    // (bx/2, by/2) → quadrant index = (by/2)*2 + bx/2.
    for (blk, slot) in mvs.iter_mut().enumerate() {
        let (bx, by) = LUMA_4X4_BLK[blk];
        let q = (by / 2) * 2 + (bx / 2);
        let mv = mv_per_8x8[q];
        *slot = (
            mv.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            mv.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
        );
    }
    let refs = [ref_idx.clamp(i8::MIN as i32, i8::MAX as i32) as i8; 4];
    let pocs = [poc_sentinel; 4];
    (mvs, refs, pocs)
}

/// Round-25 — build the per-4x4 MV / per-8x8 ref / per-8x8 POC arrays
/// for the §8.7 deblocker on a mixed `B_8x8` macroblock. Each 8x8
/// quadrant carries its own per-list MV AND its own per-list ref_idx
/// (an 8x8 cell may be Direct, L0-only, L1-only, or Bi independently
/// from its neighbours). Within a quadrant all four 4x4 blocks share
/// the quadrant's MV.
///
/// Slots whose ref_idx is negative encode "list unused for this 8x8
/// cell" (mv = (0, 0), poc = `i32::MIN` sentinel).
fn deblock_arrays_per_8x8_mixed(
    mv_per_8x8: [Mv; 4],
    ref_idx_per_8x8: [i32; 4],
    poc_sentinel: i32,
) -> ([(i16, i16); 16], [i8; 4], [i32; 4]) {
    let mut mvs = [(0i16, 0i16); 16];
    for (blk, slot) in mvs.iter_mut().enumerate() {
        let (bx, by) = LUMA_4X4_BLK[blk];
        let q = (by / 2) * 2 + (bx / 2);
        if ref_idx_per_8x8[q] >= 0 {
            let mv = mv_per_8x8[q];
            *slot = (
                mv.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                mv.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            );
        } else {
            *slot = (0, 0);
        }
    }
    let mut refs = [-1i8; 4];
    let mut pocs = [i32::MIN; 4];
    for q in 0..4usize {
        if ref_idx_per_8x8[q] >= 0 {
            refs[q] = ref_idx_per_8x8[q].clamp(i8::MIN as i32, i8::MAX as i32) as i8;
            pocs[q] = poc_sentinel;
        }
    }
    (mvs, refs, pocs)
}

impl Encoder {
    /// Round-20 — encode one B-frame using `ref_l0` (typically the IDR /
    /// older anchor) and `ref_l1` (typically the next anchor — the prior
    /// P-frame in our IPB GOP). Emits just the slice NAL; SPS/PPS were
    /// shipped by the IDR. The B-frame is **non-reference**
    /// (`nal_ref_idc = 0`) so the DPB does not retain it, and we never
    /// chain another inter-coded frame off of it.
    ///
    /// Strategy per MB:
    ///   1. Run quarter-pel ME against L0 and against L1 independently.
    ///   2. Build the bipred predictor as `(L0_pred + L1_pred + 1) >> 1`
    ///      and SAD that against source as well.
    ///   3. Pick the minimum-SAD candidate among {L0, L1, Bi} as the
    ///      `pred` for this MB. Tie-break order: Bi → L0 → L1 (Bi first
    ///      because the bit cost of a bipred MB is offset by its lower
    ///      residual rate when L0 and L1 are close).
    ///   4. Emit `B_L0/L1/Bi_16x16` with the appropriate mvd(s),
    ///      residual, recon.
    ///
    /// The encoder never emits B_Skip or B_Direct in round 20 (they
    /// require derivation paths we don't yet exercise). The slice header
    /// still signals `direct_spatial_mv_pred_flag = 1` for spec
    /// completeness.
    pub fn encode_b(
        &self,
        frame: &YuvFrame<'_>,
        ref_l0: &EncodedFrameRef<'_>,
        ref_l1: &EncodedFrameRef<'_>,
        frame_num: u32,
        pic_order_cnt_lsb: u32,
    ) -> EncodedB {
        assert_eq!(frame.width, self.cfg.width);
        assert_eq!(frame.height, self.cfg.height);
        assert_eq!(ref_l0.width, self.cfg.width);
        assert_eq!(ref_l0.height, self.cfg.height);
        assert_eq!(ref_l1.width, self.cfg.width);
        assert_eq!(ref_l1.height, self.cfg.height);
        assert!(
            self.cfg.profile_idc >= 77,
            "B-slices require Main profile or higher (§A.2.2 — Baseline forbids B)"
        );

        let width = self.cfg.width as usize;
        let width_mbs = self.cfg.width / 16;
        let height_mbs = self.cfg.height / 16;
        let chroma_width = (self.cfg.width / 2) as usize;
        let chroma_height = (self.cfg.height / 2) as usize;

        // SPS/PPS are not re-emitted for non-IDR slices.
        let mut stream: Vec<u8> = Vec::new();

        let log2_max_frame_num_minus4: u32 = 4;
        let log2_max_poc_lsb_minus4: u32 = 4;
        let chroma_qp_index_offset: i32 = 0;

        // Round-24 — direct_spatial_mv_pred_flag follows the encoder
        // configuration: 1 = §8.4.1.2.2 spatial direct (rounds 21/23),
        // 0 = §8.4.1.2.3 temporal direct (round 24). The slice header
        // signals the choice; the encoder's per-MB direct path picks
        // the matching derivation below.
        let direct_spatial = !self.cfg.direct_temporal_mv_pred;

        // Round-26 — explicit weighted-bipred weight selection.
        //
        // When `cfg.explicit_weighted_bipred` is set, fit a single
        // (w0, w1, o0, o1) at log2_wd = 5 against the two reference
        // recon planes and the source. The fit is a slice-wide constant
        // — a finer per-MB selection would need its own pred_weight
        // table per ref index, which is out of scope. The chosen weights
        // are then carried in the slice header's `pred_weight_table()`
        // and applied by the decoder via §8.4.2.3.2 explicit bipred.
        //
        // We accept the chosen weights only if they materially diverge
        // from the default (32, 32, 0, 0); otherwise the bit cost of
        // the pred_weight_table outweighs any gain and we leave the
        // table absent (decoder falls back to default merge).
        let weighted_table = if self.cfg.explicit_weighted_bipred {
            Some(compute_explicit_bipred_weights(
                frame.y,
                ref_l0.recon_y,
                ref_l1.recon_y,
                self.cfg.width as usize,
                self.cfg.height as usize,
                ref_l0.width as usize,
                ref_l1.width as usize,
            ))
        } else {
            None
        };
        let weighted_luma_for_mb: Option<WeightedBipredLuma> =
            weighted_table.map(|t| WeightedBipredLuma {
                log2_wd: t.luma_log2_weight_denom,
                weight_l0: t.luma_weight_l0,
                offset_l0: t.luma_offset_l0,
                weight_l1: t.luma_weight_l1,
                offset_l1: t.luma_offset_l1,
            });

        let mut sw = BitWriter::new();
        write_b_slice_header(
            &mut sw,
            &BSliceHeaderConfig {
                first_mb_in_slice: 0,
                slice_type_raw: 6, // B, all-same-type
                pic_parameter_set_id: 0,
                frame_num,
                frame_num_bits: log2_max_frame_num_minus4 + 4,
                pic_order_cnt_lsb,
                poc_lsb_bits: log2_max_poc_lsb_minus4 + 4,
                direct_spatial_mv_pred_flag: direct_spatial,
                slice_qp_delta: 0,
                disable_deblocking_filter_idc: 0,
                slice_alpha_c0_offset_div2: 0,
                slice_beta_offset_div2: 0,
                nal_ref_idc: 0, // non-reference B
                pred_weight_table: weighted_table,
                cabac: None,
            },
        );

        // Reconstruction state for THIS picture.
        let mut recon_y = vec![0u8; (self.cfg.width * self.cfg.height) as usize];
        let mut recon_u = vec![0u8; chroma_width * chroma_height];
        let mut recon_v = vec![0u8; chroma_width * chroma_height];

        let qp_y = self.cfg.qp;
        // §8.5.8 BD-aware chroma-QP — see encode_idr; identical reasoning.
        let qp_bd_offset_c = qp_bd_offset(self.cfg.bit_depth_chroma_minus8);
        let qp_c = qp_y_to_qp_c_with_bd_offset(qp_y, chroma_qp_index_offset, qp_bd_offset_c);

        let mut nc_grid = CavlcNcGrid::new(width_mbs, height_mbs);
        let mut intra_grid = IntraGrid::new(width_mbs as usize, height_mbs as usize);
        // Two MV-pred grids — one per reference list. §8.4.1.3 derives
        // mvpLX from neighbours' refIdxLX / mvLX of the *same* list, so
        // the L0 and L1 predictors are derived independently from each
        // other.
        let mut mv_grid_l0 = MvGrid::new(width_mbs as usize, height_mbs as usize);
        let mut mv_grid_l1 = MvGrid::new(width_mbs as usize, height_mbs as usize);

        let mut mb_deblock_infos: Vec<MbDeblockInfo> =
            vec![MbDeblockInfo::default(); (width_mbs * height_mbs) as usize];

        // §7.3.4 CAVLC B-slice walker — `mb_skip_run` ue(v) accumulates
        // consecutive B_Skip MBs and is flushed before every coded MB
        // (and once at end-of-slice). Round-21 enables B_Skip when the
        // §8.4.1.2.2 spatial-direct predictor agrees with the encoder's
        // chosen MVs and the residual quantises to all-zero.
        let mut pending_skip: u32 = 0;
        for mb_y in 0..height_mbs as usize {
            for mb_x in 0..width_mbs as usize {
                let mb_addr = mb_y * (width_mbs as usize) + mb_x;
                // Round-29: Intra fallback in B-slice via RDO wrapper.
                let dbl = self.encode_b_mb_with_intra_fallback(
                    frame,
                    ref_l0,
                    ref_l1,
                    mb_x,
                    mb_y,
                    qp_y,
                    qp_c,
                    chroma_width,
                    chroma_height,
                    width,
                    &mut recon_y,
                    &mut recon_u,
                    &mut recon_v,
                    &mut sw,
                    &mut nc_grid,
                    &mut intra_grid,
                    &mut mv_grid_l0,
                    &mut mv_grid_l1,
                    &mut pending_skip,
                    pic_order_cnt_lsb as i32,
                    weighted_luma_for_mb,
                );
                mb_deblock_infos[mb_addr] = dbl;
            }
        }
        // §7.3.4 — flush any trailing skip run (mb_skip_run for the
        // final group of skips at end-of-slice).
        sw.ue(pending_skip);

        // §7.3.2.8 — slice_data trailing bits.
        sw.rbsp_trailing_bits();
        let slice_rbsp = sw.into_bytes();
        // nal_ref_idc = 0 (non-reference B).
        stream.extend_from_slice(&build_nal_unit(0, NalUnitType::SliceNonIdr, &slice_rbsp));

        // §8.7 — picture-level in-loop deblocking. The B picture itself
        // is not used as a reference but we run the filter so the
        // exported recon matches what the decoder will output (and so
        // any test comparing against ffmpeg's decoded YUV measures the
        // post-filter samples).
        deblock_recon(
            self.cfg.width,
            self.cfg.height,
            chroma_width as u32,
            chroma_height as u32,
            &mut recon_y,
            &mut recon_u,
            &mut recon_v,
            &mb_deblock_infos,
            chroma_qp_index_offset,
            width_mbs,
            height_mbs,
        );

        // Capture per-8x8 MV state from the L0 grid (B-frames are
        // non-reference so this is normally unused; provided for API
        // symmetry with EncodedP / EncodedIdr).
        let n_parts = (width_mbs as usize) * (height_mbs as usize) * 4;
        let mut partition_mvs = Vec::with_capacity(n_parts);
        for slot in &mv_grid_l0.slots {
            for part in 0..4usize {
                // Both "not available" and "intra" map to (mv=0,
                // ref=-1, is_intra=true) per §8.4.1.2.1: an intra
                // colocated MB has refIdxCol=-1, and an unencoded slot
                // is treated identically to "intra" by the colocated
                // probe.
                let (mv, ref_idx, is_intra) = if !slot.available || slot.is_intra {
                    (Mv::ZERO, -1, true)
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

        EncodedB {
            annex_b: stream,
            recon_y,
            recon_u,
            recon_v,
            recon_width: self.cfg.width,
            recon_height: self.cfg.height,
            frame_num,
            pic_order_cnt_lsb,
            partition_mvs,
        }
    }

    /// Encode one B-slice macroblock. Mirrors [`Encoder::encode_p_mb`]
    /// but with two-list ME and the §8.4.2.3 weighted-pred merge for
    /// bipred.
    ///
    /// Round-21 adds the **spatial direct** path:
    ///   * Compute the §8.4.1.2.2 derived MVs / refIdxs for this MB's
    ///     neighbour configuration.
    ///   * Build the corresponding predictor and trial-quantise its
    ///     residual.
    ///   * If the result has `cbp_luma == cbp_chroma == 0`, emit the MB
    ///     as **B_Skip** (extends `pending_skip`). Otherwise compare its
    ///     SAD against the explicit-inter candidates (L0/L1/Bi) and
    ///     prefer **B_Direct_16x16** when its SAD is competitive (it
    ///     saves the mvds + ref_idx in the bitstream).
    #[allow(clippy::too_many_arguments)]
    fn encode_b_mb(
        &self,
        frame: &YuvFrame<'_>,
        ref_l0: &EncodedFrameRef<'_>,
        ref_l1: &EncodedFrameRef<'_>,
        mb_x: usize,
        mb_y: usize,
        qp_y: i32,
        qp_c: i32,
        chroma_width: usize,
        _chroma_height: usize,
        src_stride: usize,
        recon_y: &mut [u8],
        recon_u: &mut [u8],
        recon_v: &mut [u8],
        sw: &mut BitWriter,
        nc_grid: &mut CavlcNcGrid,
        intra_grid: &mut IntraGrid,
        mv_grid_l0: &mut MvGrid,
        mv_grid_l1: &mut MvGrid,
        pending_skip: &mut u32,
        // Round-24 — picture order count of the current B-frame, used by
        // §8.4.1.2.3 temporal direct (with `ref_l0.pic_order_cnt` =
        // pic0 POC and `ref_l1.pic_order_cnt` = pic1 POC). Ignored
        // when the encoder is configured for spatial direct.
        curr_poc: i32,
        // Round-26 — explicit weighted-bipred parameters (luma) for
        // this B slice. `Some` when the slice header carries a
        // `pred_weight_table()`; the bipred merge in `build_b_predictors`
        // then uses §8.4.2.3.2 eq. 8-276 instead of the default average.
        weighted: Option<WeightedBipredLuma>,
    ) -> MbDeblockInfo {
        let width_mbs = (self.cfg.width / 16) as usize;
        let mb_addr = mb_y * width_mbs + mb_x;

        // 1. Per-list quarter-pel ME (integer + ½-pel + ¼-pel). Exact
        //    same routine the P encoder uses; only the reference plane
        //    differs.
        let me_l0 = search_quarter_pel_16x16(
            frame.y,
            src_stride,
            self.cfg.width,
            self.cfg.height,
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
            src_stride,
            self.cfg.width,
            self.cfg.height,
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

        // 2. Build the L0-only / L1-only / bipred predictors and SAD them
        //    against the source. The predictors are built once and then
        //    re-used for the chosen mode.
        let (pred_y_l0, pred_u_l0, pred_v_l0) = build_b_predictors(
            BPred16x16::L0,
            ref_l0,
            ref_l1,
            mb_x,
            mb_y,
            mv_l0,
            mv_l1,
            weighted,
        );
        let (pred_y_l1, pred_u_l1, pred_v_l1) = build_b_predictors(
            BPred16x16::L1,
            ref_l0,
            ref_l1,
            mb_x,
            mb_y,
            mv_l0,
            mv_l1,
            weighted,
        );
        let (pred_y_bi, pred_u_bi, pred_v_bi) = build_b_predictors(
            BPred16x16::Bi,
            ref_l0,
            ref_l1,
            mb_x,
            mb_y,
            mv_l0,
            mv_l1,
            weighted,
        );

        let sad_l0 = sad_16x16_pred(frame.y, src_stride, mb_x, mb_y, &pred_y_l0);
        let sad_l1 = sad_16x16_pred(frame.y, src_stride, mb_x, mb_y, &pred_y_l1);
        let sad_bi = sad_16x16_pred(frame.y, src_stride, mb_x, mb_y, &pred_y_bi);

        // 2.5 Round-21 / Round-24 — direct mode trial.
        //
        // Mirror the decoder's derivation locally to learn what MVs /
        // refIdxs the decoder would compute if we emitted B_Direct here.
        // Build that predictor, trial-quantise its residual, and compare
        // against the explicit-inter candidates.
        //
        // The encoder's `cfg.direct_temporal_mv_pred` flag chooses
        // between two spec-defined paths:
        //   * `false` (default) — §8.4.1.2.2 SPATIAL direct (rounds 21
        //     / 23). MVs/refIdxs derived from the (A, B, C) MB-level
        //     neighbour mvs/refs of the current B-frame's grids, with
        //     a per-8x8 step-7 colZeroFlag override probing the L1
        //     anchor's colocated motion.
        //   * `true` — §8.4.1.2.3 TEMPORAL direct (round 24). MVs are
        //     POC-distance scalings of the L1 anchor's per-partition
        //     L0 motion vectors; refIdxs collapse to (0, 0) for our
        //     single-ref-per-list setup.
        //
        // Round-21 only emits B_Direct / B_Skip when the derivation
        // produces a uniform MV across all 4 8x8 partitions; round-23
        // adds B_8x8+all-Direct for the per-partition case. Same set
        // of writers serves both spatial and temporal — only the
        // derivation differs.
        let direct = if self.cfg.direct_temporal_mv_pred {
            b_temporal_direct_derive(
                ref_l1.partition_mvs,
                width_mbs,
                mb_x,
                mb_y,
                curr_poc,
                ref_l0.pic_order_cnt,
                ref_l1.pic_order_cnt,
            )
        } else {
            b_spatial_direct_derive(
                mv_grid_l0,
                mv_grid_l1,
                ref_l1.partition_mvs,
                width_mbs,
                mb_x,
                mb_y,
            )
        };
        let direct_uniform = direct.is_uniform();
        // Map (refIdxL0 >= 0, refIdxL1 >= 0) onto the same BPred16x16
        // enum the explicit-inter writer uses, so the predictor builder
        // is shared. Direct mode where neither list is used should not
        // happen (directZeroPredictionFlag forces both refs to 0), but
        // if it did we fall back to L0-only with mv=0.
        let direct_pred_kind: Option<BPred16x16> =
            match (direct.ref_idx_l0 >= 0, direct.ref_idx_l1 >= 0) {
                (true, true) => Some(BPred16x16::Bi),
                (true, false) => Some(BPred16x16::L0),
                (false, true) => Some(BPred16x16::L1),
                (false, false) => None,
            };

        // Compute the direct predictor + SAD only when the configuration
        // is exploitable (uniform across 8x8 partitions and at least one
        // list is used).
        let (direct_competitive, direct_pred_y, direct_pred_u, direct_pred_v, direct_sad) =
            match (direct_uniform, direct_pred_kind) {
                (true, Some(dpk)) => {
                    let (dy, du, dv) = build_b_predictors(
                        dpk,
                        ref_l0,
                        ref_l1,
                        mb_x,
                        mb_y,
                        direct.mv_l0(),
                        direct.mv_l1(),
                        weighted,
                    );
                    let dsad = sad_16x16_pred(frame.y, src_stride, mb_x, mb_y, &dy);
                    (true, dy, du, dv, dsad)
                }
                _ => (false, [0i32; 256], [0i32; 64], [0i32; 64], u32::MAX),
            };

        // 2.6 Round-23 — `B_8x8` with all four sub_mb_type =
        // `B_Direct_8x8`. This honours the per-8x8 MVs produced by the
        // §8.4.1.2.2 step-7 colZeroFlag override (which can drive
        // individual partitions' MVs to (0,0) while leaving others on
        // the median MVpred), as well as any future per-8x8 refIdx
        // disagreement. Compute the per-8x8 stitched predictor and its
        // SAD whenever direct_pred_kind is well-defined; the BMode
        // selector below picks among {direct uniform, direct per-8x8,
        // explicit, partition} on Lagrangian cost.
        let direct_8x8_kinds = direct_per_8x8_kinds(&direct);
        let (
            direct_8x8_competitive,
            direct_8x8_pred_y,
            direct_8x8_pred_u,
            direct_8x8_pred_v,
            direct_8x8_sad,
        ) = match direct_pred_kind {
            Some(_) => {
                let (dy, du, dv) = build_b_direct_8x8_predictors(
                    direct_8x8_kinds,
                    direct.mv_l0_per_8x8,
                    direct.mv_l1_per_8x8,
                    ref_l0,
                    ref_l1,
                    mb_x,
                    mb_y,
                );
                let dsad = sad_16x16_pred(frame.y, src_stride, mb_x, mb_y, &dy);
                (true, dy, du, dv, dsad)
            }
            None => (false, [0i32; 256], [0i32; 64], [0i32; 64], u32::MAX),
        };

        // 2.7 Round-25 — `B_8x8` with **mixed** per-cell sub_mb_type.
        // Per-cell qpel ME on each list (re-using `search_quarter_pel_8x8`),
        // then per-cell pick of {Direct, L0, L1, Bi} on Lagrangian SAD.
        // The Direct cell predictor is sliced from the round-23
        // `direct_8x8_pred_y` 16x16 buffer at the appropriate quadrant.
        // The mixed predictor is stitched from the per-cell winners. If
        // every cell ends up Direct we collapse to the round-23 path
        // (the all-Direct writer is a tighter syntax than the round-25
        // mixed writer when no MVs are emitted).
        let mut mixed_me_l0_cells: [Mv; 4] = [Mv::ZERO; 4];
        let mut mixed_me_l1_cells: [Mv; 4] = [Mv::ZERO; 4];
        for cell in 0..4usize {
            let sx = cell % 2;
            let sy = cell / 2;
            let r0 = search_quarter_pel_8x8(
                frame.y,
                src_stride,
                self.cfg.width,
                self.cfg.height,
                ref_l0.recon_y,
                ref_l0.width as usize,
                ref_l0.width,
                ref_l0.height,
                mb_x,
                mb_y,
                sx,
                sy,
                4,
                4,
            );
            let r1 = search_quarter_pel_8x8(
                frame.y,
                src_stride,
                self.cfg.width,
                self.cfg.height,
                ref_l1.recon_y,
                ref_l1.width as usize,
                ref_l1.width,
                ref_l1.height,
                mb_x,
                mb_y,
                sx,
                sy,
                4,
                4,
            );
            mixed_me_l0_cells[cell] = Mv::new(r0.mv_x, r0.mv_y);
            mixed_me_l1_cells[cell] = Mv::new(r1.mv_x, r1.mv_y);
        }

        let mut mixed_cells = [BSubMbCell::Direct; 4];
        let mut mixed_cell_mv_l0 = [Mv::ZERO; 4];
        let mut mixed_cell_mv_l1 = [Mv::ZERO; 4];
        let mut mixed_cell_pred_y_per_cell = [[0i32; 64]; 4];
        let mut mixed_total_sad = 0u32;
        let mixed_competitive = direct_pred_kind.is_some();
        if mixed_competitive {
            for cell in 0..4usize {
                let sub_x = cell % 2;
                let sub_y = cell / 2;
                // Slice the Direct cell's 8x8 luma predictor out of the
                // 16x16 buffer.
                let mut direct_cell_pred = [0i32; 64];
                for j in 0..8usize {
                    for i in 0..8usize {
                        direct_cell_pred[j * 8 + i] =
                            direct_8x8_pred_y[(sub_y * 8 + j) * 16 + sub_x * 8 + i];
                    }
                }
                let (cell_kind, cell_mv0, cell_mv1, cell_pred, cell_sad) = pick_best_b8x8_cell(
                    frame.y,
                    src_stride,
                    ref_l0,
                    ref_l1,
                    mb_x,
                    mb_y,
                    sub_x,
                    sub_y,
                    &direct_cell_pred,
                    mixed_me_l0_cells[cell],
                    mixed_me_l1_cells[cell],
                    qp_y,
                );
                mixed_cells[cell] = cell_kind;
                mixed_cell_mv_l0[cell] = cell_mv0;
                mixed_cell_mv_l1[cell] = cell_mv1;
                mixed_cell_pred_y_per_cell[cell] = cell_pred;
                mixed_total_sad = mixed_total_sad.saturating_add(cell_sad);
            }
        } else {
            mixed_total_sad = u32::MAX;
        }

        // The mixed mode is only worth emitting when at least one cell
        // picked a non-Direct kind (otherwise the round-23 all-Direct
        // writer is tighter and produces the same predictor).
        let mixed_has_non_direct = mixed_cells.iter().any(|&c| c != BSubMbCell::Direct);
        let mixed_competitive_for_pick = mixed_competitive && mixed_has_non_direct;

        // Stitch the per-cell mixed luma + chroma composite predictors.
        let (mixed_pred_y, mixed_pred_u, mixed_pred_v) = if mixed_competitive_for_pick {
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
                let cell_pred_y = mixed_cell_pred_y_per_cell[cell];
                for j in 0..8usize {
                    for i in 0..8usize {
                        py[(sub_y * 8 + j) * 16 + sub_x * 8 + i] = cell_pred_y[j * 8 + i];
                    }
                }
                // Chroma predictor for this cell. Direct cells use the
                // §8.4.1.2.2-derived per-8x8 MVs; explicit cells use the
                // per-cell ME MVs. The chroma kind for Direct cells
                // honours the §8.4.1.2.2 derived (refIdxL0, refIdxL1)
                // status (both >= 0 → Bi; only one >= 0 → L0/L1).
                let (mv0, mv1) = if mixed_cells[cell] == BSubMbCell::Direct {
                    (direct.mv_l0_per_8x8[cell], direct.mv_l1_per_8x8[cell])
                } else {
                    (mixed_cell_mv_l0[cell], mixed_cell_mv_l1[cell])
                };
                let chroma_kind = match mixed_cells[cell] {
                    BSubMbCell::Direct => {
                        match (direct.ref_idx_l0 >= 0, direct.ref_idx_l1 >= 0) {
                            (true, true) => BSubMbCell::Bi,
                            (true, false) => BSubMbCell::L0,
                            (false, true) => BSubMbCell::L1,
                            (false, false) => BSubMbCell::L0, // defensive
                        }
                    }
                    other => other,
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
                    mv0,
                    mv1,
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
                    mv0,
                    mv1,
                );
                for j in 0..4usize {
                    for i in 0..4usize {
                        pu[(sub_y * 4 + j) * 8 + sub_x * 4 + i] = cu[j * 4 + i];
                        pv[(sub_y * 4 + j) * 8 + sub_x * 4 + i] = cv[j * 4 + i];
                    }
                }
            }
            (py, pu, pv)
        } else {
            ([0i32; 256], [0i32; 64], [0i32; 64])
        };

        // 3. Pick the candidate that minimises a Lagrangian cost
        //    `J = SAD_luma + λ · R_syntax`. Direct mode emits the
        //    smallest syntax block (just `mb_type=0 + cbp`, or nothing
        //    at all when B_Skip), so it gets a SAD allowance that
        //    reflects its rate savings.
        //
        // We approximate R_syntax with simple constants:
        //   * Direct (cbp != 0): ~2 bits (`mb_type=0` ue + `cbp_codenum`
        //     ue=0 if cbp small).
        //   * Direct + cbp == 0 (would emit B_Skip): 0 bits (run-length).
        //   * Explicit L0 / L1: ~3 + 2·avg_mvd_bits ≈ ~12 bits.
        //   * Explicit Bi:      ~3 + 4·avg_mvd_bits ≈ ~20 bits.
        //
        // The "SAD-equivalent" of one bit at QP_Y is roughly
        //   λ · 1 ≈ 0.85 · 2^((QP-12)/3) bits-to-SAD scaling.
        // For QP=26 this gives λ ≈ 22, and the explicit-vs-direct
        // syntax delta is ~10-20 bits → an equivalent SAD margin of
        // ~200-440. Empirically a fixed margin of 16 bits' worth on
        // top of the explicit candidate's SAD is sufficient for static
        // / smooth content where ME is searching a flat SAD landscape.
        enum BMode {
            Direct,
            /// Round-23: `B_8x8` with all four sub_mb_type=B_Direct_8x8.
            /// Same MV / ref derivation as Direct (§8.4.1.2.2), but with
            /// per-8x8 motion compensation honouring the colZeroFlag
            /// step-7 per-partition overrides. Bigger MB syntax than
            /// `B_Direct_16x16` (mb_type=22 ue(9 bits) + 4× sub_mb_type
            /// ue(1 bit) = 13 bits before cbp), so only chosen when
            /// direct_8x8_sad beats direct_uniform_sad by more than the
            /// extra-syntax cost.
            DirectPer8x8,
            /// Round-25: `B_8x8` with mixed per-cell sub_mb_type (some
            /// cells Direct, others L0/L1/Bi). Each non-Direct cell adds
            /// ~6-12 bits of MB syntax (sub_mb_type overhead + per-cell
            /// mvds), so only chosen when the per-cell motion materially
            /// beats the all-Direct (round-23) candidate.
            Mixed8x8,
            Explicit(BPred16x16),
        }
        // Per-MB Lagrangian penalty applied to explicit candidates
        // relative to Direct, in SAD units. Tuned conservatively so
        // round-20 fixtures still see Bi pick when it materially beats
        // Direct (the round-20 midpoint test has direct_sad far above
        // bi_sad on every MB → Bi still wins there).
        let direct_bias_sad: u32 = {
            // λ · bit-savings, scaled coarsely. For QP 0..51 this
            // ranges from ~1 (low QP) to ~64 (QP 51). The bit savings
            // (Direct vs Explicit Bi) are ~20 bits, so a typical
            // smooth-motion B-slice gets a few hundred SAD units of
            // headroom in favour of Direct.
            let lambda = (qp_y / 6).clamp(0, 8) as u32;
            (lambda + 1) * 16
        };
        // Round-23 — extra cost for B_8x8+all-Direct relative to
        // B_Direct_16x16: ~13 - 1 = ~12 bits of MB header (mb_type 22 +
        // 4× sub_mb_type=0 = 13 bits, vs B_Direct_16x16's 1 bit
        // mb_type=0). Bias the selector against per-8x8 direct so we
        // only switch when the per-8x8 motion materially beats the
        // single-MV uniform predictor.
        let direct_8x8_bias_sad: u32 = {
            let lambda = (qp_y / 6).clamp(0, 8) as u32;
            (lambda + 1) * 12
        };
        // Best direct candidate (between uniform B_Direct_16x16 and
        // per-8x8 B_8x8+all-Direct). Pick uniform when its SAD plus the
        // per-8x8 syntax-cost bias still beats the per-8x8 SAD.
        let prefer_per_8x8 = direct_8x8_competitive
            && (!direct_competitive
                || direct_8x8_sad.saturating_add(direct_8x8_bias_sad) < direct_sad);
        let (best_direct_competitive, best_direct_sad, best_direct_is_per_8x8) = if prefer_per_8x8 {
            (direct_8x8_competitive, direct_8x8_sad, true)
        } else {
            (direct_competitive, direct_sad, false)
        };
        // Round-25 — mixed `B_8x8` candidate. The per-cell pick already
        // factored in a per-cell Lagrangian rate cost (in
        // `pick_best_b8x8_cell`), so `mixed_total_sad` already includes
        // the per-non-Direct-cell sub_mb_type+mvd bit penalty.
        //
        // Mixed8x8 vs B_Direct_16x16: Mixed costs +12 bits of MB header
        // (mb_type 22 ue(9) + 4× sub_mb_type ue(≥1) ≥ 4 bits, vs
        // B_Direct_16x16's 1-bit mb_type=0).
        //
        // Mixed8x8 vs DirectPer8x8: same MB header (both mb_type=22),
        // but Mixed's sub_mb_type entries cost ≥3 extra bits each for
        // non-Direct cells (ue(1)/ue(2)/ue(3) = 3/3/5 bits vs ue(0) = 1
        // bit) — that overhead is the per-cell `(lambda+1)*6` term in
        // `pick_best_b8x8_cell`'s rate cost.
        //
        // To gate Mixed vs DirectPer8x8 / B_Direct_16x16 fairly:
        //   * If best_direct is the 16x16 form, add the +12-bit MB
        //     header penalty to Mixed's J score.
        //   * If best_direct is DirectPer8x8, no extra penalty (same MB
        //     header).
        //
        // Plus a small extra "tie-break" margin so Mixed only fires when
        // it materially beats the simpler all-Direct candidate.
        let mixed_mb_header_bias_sad: u32 = if best_direct_is_per_8x8 {
            0
        } else {
            let lambda = (qp_y / 6).clamp(0, 8) as u32;
            (lambda + 1) * 12
        };
        let mixed_tiebreak_bias_sad: u32 = {
            let lambda = (qp_y / 6).clamp(0, 8) as u32;
            (lambda + 1) * 8
        };
        let prefer_mixed = mixed_competitive_for_pick
            && best_direct_competitive
            && mixed_total_sad
                .saturating_add(mixed_mb_header_bias_sad)
                .saturating_add(mixed_tiebreak_bias_sad)
                < best_direct_sad;
        let chosen_mode: BMode = if prefer_mixed {
            BMode::Mixed8x8
        } else if best_direct_competitive
            && best_direct_sad <= sad_bi.saturating_add(direct_bias_sad)
            && best_direct_sad <= sad_l0.saturating_add(direct_bias_sad)
            && best_direct_sad <= sad_l1.saturating_add(direct_bias_sad)
        {
            if best_direct_is_per_8x8 {
                BMode::DirectPer8x8
            } else {
                BMode::Direct
            }
        } else if sad_bi <= sad_l0 && sad_bi <= sad_l1 {
            BMode::Explicit(BPred16x16::Bi)
        } else if sad_l0 <= sad_l1 {
            BMode::Explicit(BPred16x16::L0)
        } else {
            BMode::Explicit(BPred16x16::L1)
        };
        let (pred, pred_y, pred_u, pred_v, eff_mv_l0, eff_mv_l1) = match chosen_mode {
            BMode::Direct => {
                // Direct's effective MVs are the §8.4.1.2.2 derivation's
                // outputs (used for downstream grid updates + deblocker
                // mv-identity). The explicit `pred` enum carries which
                // list(s) the predictor reads; for the writer we'll use
                // the dedicated B_Direct_16x16 path (no mvd / ref_idx).
                (
                    direct_pred_kind.expect("Direct chosen → direct_pred_kind set"),
                    direct_pred_y,
                    direct_pred_u,
                    direct_pred_v,
                    direct.mv_l0(),
                    direct.mv_l1(),
                )
            }
            BMode::DirectPer8x8 => {
                // The MB-level (eff_mv_l0, eff_mv_l1) returned here is
                // only used for downstream grid + deblock setup when
                // partition_choice is None (the per-8x8 path patches its
                // own per-8x8 MVs into the grid further below), so we
                // hand back the top-left 8x8's MV as a representative —
                // the real per-8x8 grid update happens in the explicit
                // per-8x8 branch after `partition_choice` is settled.
                (
                    direct_pred_kind.expect("DirectPer8x8 chosen → direct_pred_kind set"),
                    direct_8x8_pred_y,
                    direct_8x8_pred_u,
                    direct_8x8_pred_v,
                    direct.mv_l0_per_8x8[0],
                    direct.mv_l1_per_8x8[0],
                )
            }
            BMode::Mixed8x8 => {
                // Round-25 — `B_8x8` with mixed sub_mb_type cells. The
                // composite predictor was stitched above. The MB-level
                // `pred` enum is a coarse summary used by downstream
                // grid / deblock helpers when no per-cell override
                // overrides them; the per-cell branch below patches the
                // grid with the per-cell MVs / refs directly.
                let any_l0 = mixed_cells.iter().any(|&c| c.writes_l0()) || direct.ref_idx_l0 >= 0;
                let any_l1 = mixed_cells.iter().any(|&c| c.writes_l1()) || direct.ref_idx_l1 >= 0;
                let summary = match (any_l0, any_l1) {
                    (true, true) => BPred16x16::Bi,
                    (true, false) => BPred16x16::L0,
                    (false, true) => BPred16x16::L1,
                    (false, false) => BPred16x16::L0, // defensive
                };
                (
                    summary,
                    mixed_pred_y,
                    mixed_pred_u,
                    mixed_pred_v,
                    mixed_cell_mv_l0[0],
                    mixed_cell_mv_l1[0],
                )
            }
            BMode::Explicit(BPred16x16::Bi) => (
                BPred16x16::Bi,
                pred_y_bi,
                pred_u_bi,
                pred_v_bi,
                mv_l0,
                mv_l1,
            ),
            BMode::Explicit(BPred16x16::L0) => (
                BPred16x16::L0,
                pred_y_l0,
                pred_u_l0,
                pred_v_l0,
                mv_l0,
                Mv::ZERO,
            ),
            BMode::Explicit(BPred16x16::L1) => (
                BPred16x16::L1,
                pred_y_l1,
                pred_u_l1,
                pred_v_l1,
                Mv::ZERO,
                mv_l1,
            ),
        };
        let mut is_direct = matches!(chosen_mode, BMode::Direct);
        // Round-23 — when DirectPer8x8 is chosen, we still want the
        // partition-mode override below to be DISABLED (Direct paths
        // never pair with explicit 16x8 / 8x16 partition shapes).
        let is_direct_per_8x8 = matches!(chosen_mode, BMode::DirectPer8x8);
        // Round-25 — Mixed8x8 also bypasses the 16x8 / 8x16 partition
        // override (mb_type=22 is mutually exclusive with raws 4..=21).
        let is_mixed_8x8 = matches!(chosen_mode, BMode::Mixed8x8);

        // ---- Round-22: try 16x8 / 8x16 partition modes ----
        //
        // For each of the 4 partition shapes, pick the best (per-list
        // MV, mode) pair. Then compare the joint partition-shape SAD
        // against the winning 16x16 / Direct candidate above. Use a
        // Lagrangian rate cost: a partition MB pays ~10-30 extra bits
        // over 16x16 (single-ref → no extra ref_idx fields, just the
        // larger mb_type ue(v) and the duplicated mvds).
        let mut me_l0_cells: [Mv; 4] = [Mv::ZERO; 4];
        let mut me_l1_cells: [Mv; 4] = [Mv::ZERO; 4];
        for cell in 0..4usize {
            let sx = cell % 2;
            let sy = cell / 2;
            let r0 = search_quarter_pel_8x8(
                frame.y,
                src_stride,
                self.cfg.width,
                self.cfg.height,
                ref_l0.recon_y,
                ref_l0.width as usize,
                ref_l0.width,
                ref_l0.height,
                mb_x,
                mb_y,
                sx,
                sy,
                4,
                4,
            );
            let r1 = search_quarter_pel_8x8(
                frame.y,
                src_stride,
                self.cfg.width,
                self.cfg.height,
                ref_l1.recon_y,
                ref_l1.width as usize,
                ref_l1.width,
                ref_l1.height,
                mb_x,
                mb_y,
                sx,
                sy,
                4,
                4,
            );
            me_l0_cells[cell] = Mv::new(r0.mv_x, r0.mv_y);
            me_l1_cells[cell] = Mv::new(r1.mv_x, r1.mv_y);
        }

        // Best 16x16 SAD = the SAD of the chosen 16x16-mode predictor.
        // (For Mixed8x8 / DirectPer8x8 we still pass the SAD of their
        // composite into the partition-mode comparison even though the
        // partition path will be skipped — `is_mixed_8x8` /
        // `is_direct_per_8x8` are checked directly in the gate below.)
        let best_16x16_sad = match chosen_mode {
            BMode::Direct => direct_sad,
            BMode::DirectPer8x8 => direct_8x8_sad,
            BMode::Mixed8x8 => mixed_total_sad,
            BMode::Explicit(BPred16x16::Bi) => sad_bi,
            BMode::Explicit(BPred16x16::L0) => sad_l0,
            BMode::Explicit(BPred16x16::L1) => sad_l1,
        };

        // 16x8 candidates: top uses cells 0/1; bottom uses 2/3.
        let cand_top_l0 = [mv_l0, me_l0_cells[0], me_l0_cells[1]];
        let cand_top_l1 = [mv_l1, me_l1_cells[0], me_l1_cells[1]];
        let cand_bot_l0 = [mv_l0, me_l0_cells[2], me_l0_cells[3]];
        let cand_bot_l1 = [mv_l1, me_l1_cells[2], me_l1_cells[3]];
        let (top_mode, top_mv_l0, top_mv_l1, sad_top) =
            best_partition_mode_b(&cand_top_l0, &cand_top_l1, |pred, mv0, mv1| {
                partition_sad_b_16x8(
                    pred, frame.y, src_stride, ref_l0, ref_l1, mb_x, mb_y, 0, mv0, mv1,
                )
            });
        let (bot_mode, bot_mv_l0, bot_mv_l1, sad_bot) =
            best_partition_mode_b(&cand_bot_l0, &cand_bot_l1, |pred, mv0, mv1| {
                partition_sad_b_16x8(
                    pred, frame.y, src_stride, ref_l0, ref_l1, mb_x, mb_y, 8, mv0, mv1,
                )
            });
        let sad_16x8_total = sad_top.saturating_add(sad_bot);

        // 8x16 candidates: left = cells 0/2; right = cells 1/3.
        let cand_left_l0 = [mv_l0, me_l0_cells[0], me_l0_cells[2]];
        let cand_left_l1 = [mv_l1, me_l1_cells[0], me_l1_cells[2]];
        let cand_right_l0 = [mv_l0, me_l0_cells[1], me_l0_cells[3]];
        let cand_right_l1 = [mv_l1, me_l1_cells[1], me_l1_cells[3]];
        let (left_mode, left_mv_l0, left_mv_l1, sad_left) =
            best_partition_mode_b(&cand_left_l0, &cand_left_l1, |pred, mv0, mv1| {
                partition_sad_b_8x16(
                    pred, frame.y, src_stride, ref_l0, ref_l1, mb_x, mb_y, 0, mv0, mv1,
                )
            });
        let (right_mode, right_mv_l0, right_mv_l1, sad_right) =
            best_partition_mode_b(&cand_right_l0, &cand_right_l1, |pred, mv0, mv1| {
                partition_sad_b_8x16(
                    pred, frame.y, src_stride, ref_l0, ref_l1, mb_x, mb_y, 8, mv0, mv1,
                )
            });
        let sad_8x16_total = sad_left.saturating_add(sad_right);

        // Lagrangian rate penalty for partition modes vs 16x16. For QP=26
        // this gives ~40 SAD units of headroom — partition wins only when
        // its SAD beats 16x16 by more than that.
        let part_bias_sad: u32 = {
            let lambda = (qp_y / 6).clamp(0, 8) as u32;
            (lambda + 1) * 8
        };
        let part_threshold = best_16x16_sad.saturating_sub(part_bias_sad);
        let (part_winner, part_winner_sad) = if sad_16x8_total <= sad_8x16_total {
            (PartitionShape::P16x8, sad_16x8_total)
        } else {
            (PartitionShape::P8x16, sad_8x16_total)
        };
        let mut partition_choice: Option<PartitionChoiceB> = None;
        // Don't override Direct: it may collapse to B_Skip (0 bits of MB
        // syntax) once the residual quantises to zero, beating any
        // partition mode by a large margin. Same for DirectPer8x8 / Mixed8x8 —
        // both emit `mb_type=22` (B_8x8) which is mutually exclusive
        // with the explicit 16x8 / 8x16 partition shapes (mb_type 4..=21).
        // Partition mode override only applies to explicit-inter 16x16
        // candidates.
        if !is_direct && !is_direct_per_8x8 && !is_mixed_8x8 && part_winner_sad < part_threshold {
            partition_choice = Some(match part_winner {
                PartitionShape::P16x8 => PartitionChoiceB {
                    shape: PartitionShape::P16x8,
                    mode_a: top_mode,
                    mode_b: bot_mode,
                    mv_l0_a: top_mv_l0,
                    mv_l1_a: top_mv_l1,
                    mv_l0_b: bot_mv_l0,
                    mv_l1_b: bot_mv_l1,
                },
                PartitionShape::P8x16 => PartitionChoiceB {
                    shape: PartitionShape::P8x16,
                    mode_a: left_mode,
                    mode_b: right_mode,
                    mv_l0_a: left_mv_l0,
                    mv_l1_a: left_mv_l1,
                    mv_l0_b: right_mv_l0,
                    mv_l1_b: right_mv_l1,
                },
            });
            // Direct mode is incompatible with partition shapes — clear it.
            is_direct = false;
        }

        // Override the working predictor with the partition composite
        // when a partition mode was chosen.
        let (pred_y, pred_u, pred_v) = if let Some(pc) = partition_choice.as_ref() {
            build_partition_composite_b(pc, ref_l0, ref_l1, mb_x, mb_y)
        } else {
            (pred_y, pred_u, pred_v)
        };

        // 4. mvds against the §8.4.1.3 mvp for each used list. For 16x16
        //    explicit, this is a single pair; for partition modes we
        //    derive per-partition MVDs further below.
        let mvp_l0 = mvp_for_16x16(mv_grid_l0, mb_x, mb_y, 0);
        let mvp_l1 = mvp_for_16x16(mv_grid_l1, mb_x, mb_y, 0);
        let (mvd_l0_x, mvd_l0_y) = if !is_direct
            && !is_direct_per_8x8
            && !is_mixed_8x8
            && partition_choice.is_none()
            && pred.uses_l0()
        {
            (eff_mv_l0.x - mvp_l0.x, eff_mv_l0.y - mvp_l0.y)
        } else {
            (0, 0)
        };
        let (mvd_l1_x, mvd_l1_y) = if !is_direct
            && !is_direct_per_8x8
            && !is_mixed_8x8
            && partition_choice.is_none()
            && pred.uses_l1()
        {
            (eff_mv_l1.x - mvp_l1.x, eff_mv_l1.y - mvp_l1.y)
        } else {
            (0, 0)
        };

        // Per-partition MVDs against the §8.4.1.3 partition-shape mvp.
        // Only populated when `partition_choice` is set.
        let (mvd_l0_parts, mvd_l1_parts) = if let Some(pc) = partition_choice.as_ref() {
            partition_mvds_b(pc, mv_grid_l0, mv_grid_l1, mb_x, mb_y)
        } else {
            ([(0, 0); 2], [(0, 0); 2])
        };

        // Round-25 — per-cell MVDs for the Mixed8x8 mode. For each
        // non-Direct cell, compute the mvd against the §8.4.1.3 mvpLX
        // for that 8x8 partition (using the within-MB inflight neighbour
        // states as cells decode in raster Z-order 0→1→2→3). Direct
        // cells emit no MVDs.
        let (mixed_mvd_l0_per_cell, mixed_mvd_l1_per_cell) = if is_mixed_8x8 {
            let mut mvd0 = [(0i32, 0i32); 4];
            let mut mvd1 = [(0i32, 0i32); 4];
            // Inflight per-cell L0 / L1 state for §8.4.1.3 within-MB
            // neighbour reads (mirrors the round-19 P_8x8 path). Each
            // cell sees the previously-decoded cells of this same MB.
            let mut inflight_avail = [false; 4];
            let mut mv_inflight_l0 = [Mv::ZERO; 4];
            let mut ref_inflight_l0 = [-1i32; 4];
            let mut mv_inflight_l1 = [Mv::ZERO; 4];
            let mut ref_inflight_l1 = [-1i32; 4];
            for cell in 0..4usize {
                let kind = mixed_cells[cell];
                if kind.writes_l0() {
                    let mvp = mvp_for_p_8x8_partition(
                        mv_grid_l0,
                        mb_x,
                        mb_y,
                        cell,
                        0,
                        inflight_avail,
                        mv_inflight_l0,
                        ref_inflight_l0,
                    );
                    let mv = mixed_cell_mv_l0[cell];
                    mvd0[cell] = (mv.x - mvp.x, mv.y - mvp.y);
                }
                if kind.writes_l1() {
                    let mvp = mvp_for_p_8x8_partition(
                        mv_grid_l1,
                        mb_x,
                        mb_y,
                        cell,
                        0,
                        inflight_avail,
                        mv_inflight_l1,
                        ref_inflight_l1,
                    );
                    let mv = mixed_cell_mv_l1[cell];
                    mvd1[cell] = (mv.x - mvp.x, mv.y - mvp.y);
                }
                // Update inflight per-cell state for downstream cells.
                inflight_avail[cell] = true;
                // For Direct cells, the inflight state must reflect the
                // §8.4.1.2.2 derivation result (so subsequent cells'
                // §8.4.1.3 neighbour reads see the correct MV / ref).
                let (cell_mv0_ifl, cell_ref0_ifl) = match kind {
                    BSubMbCell::Direct => (direct.mv_l0_per_8x8[cell], direct.ref_idx_l0),
                    BSubMbCell::L0 | BSubMbCell::Bi => (mixed_cell_mv_l0[cell], 0),
                    BSubMbCell::L1 => (Mv::ZERO, -1),
                };
                let (cell_mv1_ifl, cell_ref1_ifl) = match kind {
                    BSubMbCell::Direct => (direct.mv_l1_per_8x8[cell], direct.ref_idx_l1),
                    BSubMbCell::L1 | BSubMbCell::Bi => (mixed_cell_mv_l1[cell], 0),
                    BSubMbCell::L0 => (Mv::ZERO, -1),
                };
                mv_inflight_l0[cell] = cell_mv0_ifl;
                ref_inflight_l0[cell] = cell_ref0_ifl;
                mv_inflight_l1[cell] = cell_mv1_ifl;
                ref_inflight_l1[cell] = cell_ref1_ifl;
            }
            (mvd0, mvd1)
        } else {
            ([(0, 0); 4], [(0, 0); 4])
        };

        let _ = BMbDecision {
            pred,
            mv_l0: eff_mv_l0,
            mv_l1: eff_mv_l1,
            mvd_l0_x,
            mvd_l0_y,
            mvd_l1_x,
            mvd_l1_y,
            pred_y,
            pred_u,
            pred_v,
        }; // (struct kept for forward documentation; the values are read directly below)

        // 5. Forward + quantize the luma residual against the chosen
        //    predictor (re-using the inter forward path).
        let inter_luma = forward_inter_luma(frame.y, src_stride, mb_x, mb_y, &pred_y, qp_y);
        let luma_4x4_levels_scan = inter_luma.levels_scan;
        let blk_has_nz = inter_luma.blk_has_nz;
        let recon_block_residual = inter_luma.recon_residual;

        let mut cbp_luma: u8 = 0;
        for blk8 in 0..4usize {
            let any_nz = (0..4).any(|sub| blk_has_nz[blk8 * 4 + sub]);
            if any_nz {
                cbp_luma |= 1u8 << blk8;
            }
        }

        // 6. Forward + quantize chroma residual.
        let (u_dc_levels, u_ac_levels, u_recon_residual) =
            encode_chroma_residual_inter(frame.u, chroma_width, mb_x, mb_y, &pred_u, qp_c);
        let (v_dc_levels, v_ac_levels, v_recon_residual) =
            encode_chroma_residual_inter(frame.v, chroma_width, mb_x, mb_y, &pred_v, qp_c);
        let any_dc_nz = u_dc_levels.iter().any(|&v| v != 0) || v_dc_levels.iter().any(|&v| v != 0);
        let any_ac_nz = u_ac_levels.iter().any(|blk| blk.iter().any(|&v| v != 0))
            || v_ac_levels.iter().any(|blk| blk.iter().any(|&v| v != 0));
        let cbp_chroma: u8 = if any_ac_nz {
            2
        } else if any_dc_nz {
            1
        } else {
            0
        };

        // 7. Apply luma reconstruction (pred + residual or pred-only on
        //    cbp_luma=0 quadrants). Same pattern as encode_p_mb.
        let width = self.cfg.width as usize;
        for blk8 in 0..4usize {
            let send = (cbp_luma >> blk8) & 1 == 1;
            for sub in 0..4usize {
                let blk = blk8 * 4 + sub;
                let (bx, by) = LUMA_4X4_BLK[blk];
                let bx_pix = bx * 4;
                let by_pix = by * 4;
                for j in 0..4usize {
                    for i in 0..4usize {
                        let p = pred_y[(by_pix + j) * 16 + bx_pix + i];
                        let r = if send {
                            recon_block_residual[blk][j * 4 + i]
                        } else {
                            0
                        };
                        let py = mb_y * 16 + by_pix + j;
                        let px = mb_x * 16 + bx_pix + i;
                        recon_y[py * width + px] = (p + r).clamp(0, 255) as u8;
                    }
                }
            }
        }
        for j in 0..8usize {
            for i in 0..8usize {
                let cy = mb_y * 8 + j;
                let cx = mb_x * 8 + i;
                let pu = pred_u[j * 8 + i];
                let pv = pred_v[j * 8 + i];
                let res_u = if cbp_chroma == 0 {
                    0
                } else {
                    u_recon_residual[j * 8 + i]
                };
                let res_v = if cbp_chroma == 0 {
                    0
                } else {
                    v_recon_residual[j * 8 + i]
                };
                recon_u[cy * chroma_width + cx] = (pu + res_u).clamp(0, 255) as u8;
                recon_v[cy * chroma_width + cx] = (pv + res_v).clamp(0, 255) as u8;
            }
        }

        // 8. CAVLC neighbour-grid update + per-block nC.
        //
        // §9.2.1.1 step 6 — a B_Skip neighbour contributes nN=0 (same as
        // P_Skip). For B_Direct_16x16 with cbp_luma=0 the slots are also
        // 0 but `is_skip` stays false (the MB is coded, just with no
        // residual).
        let mb_is_skip = is_direct && cbp_luma == 0 && cbp_chroma == 0;
        {
            let cur = &mut nc_grid.mbs[mb_addr];
            cur.is_available = true;
            cur.is_intra = false;
            cur.is_skip = mb_is_skip;
            cur.is_i_pcm = false;
            cur.luma_total_coeff = [0u8; 16];
        }

        let mut luma_4x4_nc = [0i32; 16];
        let mut own_totals = [0u8; 16];
        for blk in 0..16usize {
            let blk8 = blk / 4;
            nc_grid.mbs[mb_addr].luma_total_coeff = own_totals;
            let nc = derive_nc_luma(
                nc_grid,
                mb_addr as u32,
                blk as u8,
                LumaNcKind::Ac,
                true,
                false,
            );
            luma_4x4_nc[blk] = nc;
            if (cbp_luma >> blk8) & 1 == 1 {
                let tc = luma_4x4_levels_scan[blk]
                    .iter()
                    .filter(|&&v| v != 0)
                    .count() as u8;
                own_totals[blk] = tc;
            } else {
                own_totals[blk] = 0;
            }
        }
        nc_grid.mbs[mb_addr].luma_total_coeff = own_totals;

        // 9. Emit the MB syntax.
        //
        // §7.3.4 — four bitstream layouts:
        //   * B_Skip            : no MB syntax — pending_skip += 1.
        //   * B_Direct_16x16    : mb_type=0 + cbp + (mb_qp_delta) + residuals.
        //   * Explicit 16x16    : mb_type∈{1,2,3} + ref_idxs + mvds + cbp + …
        //   * 16x8 / 8x16 (r22) : mb_type∈{4..21} + per-partition ref_idxs
        //                        + per-partition mvds + cbp + …
        // Skips never flush their own ue(v); they are folded into the
        // run that precedes the next coded MB (or the trailing flush).
        if mb_is_skip {
            *pending_skip += 1;
        } else {
            sw.ue(*pending_skip);
            *pending_skip = 0;

            if is_direct {
                let dcfg = crate::encoder::macroblock::BDirect16x16McbConfig {
                    cbp_luma,
                    cbp_chroma,
                    mb_qp_delta: 0,
                    luma_4x4_levels: luma_4x4_levels_scan,
                    luma_4x4_nc,
                    chroma_dc_cb: u_dc_levels,
                    chroma_dc_cr: v_dc_levels,
                    chroma_ac_cb: u_ac_levels,
                    chroma_ac_cr: v_ac_levels,
                };
                crate::encoder::macroblock::write_b_direct_16x16_mb(sw, &dcfg)
                    .expect("write B_Direct_16x16 mb");
            } else if is_direct_per_8x8 {
                // Round-23 — `B_8x8` with all four sub_mb_type =
                // `B_Direct_8x8`. The decoder re-derives all four
                // (mvL0, mvL1, refIdxL0, refIdxL1) per §8.4.1.2.2 (we
                // already mirror that derivation in
                // `b_spatial_direct_derive`), so no MV / refIdx fields
                // appear in the bitstream — only cbp + residuals.
                let dcfg = crate::encoder::macroblock::B8x8AllDirectMcbConfig {
                    cbp_luma,
                    cbp_chroma,
                    mb_qp_delta: 0,
                    luma_4x4_levels: luma_4x4_levels_scan,
                    luma_4x4_nc,
                    chroma_dc_cb: u_dc_levels,
                    chroma_dc_cr: v_dc_levels,
                    chroma_ac_cb: u_ac_levels,
                    chroma_ac_cr: v_ac_levels,
                };
                crate::encoder::macroblock::write_b_8x8_all_direct_mb(sw, &dcfg)
                    .expect("write B_8x8 + 4× B_Direct_8x8 mb");
            } else if is_mixed_8x8 {
                // Round-25 — `B_8x8` with mixed per-cell sub_mb_type.
                // Direct cells emit only their sub_mb_type; explicit
                // cells emit their (mvd_l0, mvd_l1) pair as appropriate
                // (§7.3.5.2 gating; ref_idx absent in single-ref setup).
                let mcfg = B8x8MixedMcbConfig {
                    cells: mixed_cells,
                    mvd_l0: mixed_mvd_l0_per_cell,
                    mvd_l1: mixed_mvd_l1_per_cell,
                    cbp_luma,
                    cbp_chroma,
                    mb_qp_delta: 0,
                    luma_4x4_levels: luma_4x4_levels_scan,
                    luma_4x4_nc,
                    chroma_dc_cb: u_dc_levels,
                    chroma_dc_cr: v_dc_levels,
                    chroma_ac_cb: u_ac_levels,
                    chroma_ac_cr: v_ac_levels,
                };
                write_b_8x8_mixed_mb(sw, &mcfg, 0, 0).expect("write B_8x8 mixed mb");
            } else if let Some(pc) = partition_choice.as_ref() {
                use crate::encoder::macroblock::{
                    write_b_16x8_mb, write_b_8x16_mb, B16x8McbConfig, B8x16McbConfig,
                };
                match pc.shape {
                    PartitionShape::P16x8 => {
                        let cfg = B16x8McbConfig {
                            top: pc.mode_a,
                            bottom: pc.mode_b,
                            mvd_l0: mvd_l0_parts,
                            mvd_l1: mvd_l1_parts,
                            cbp_luma,
                            cbp_chroma,
                            mb_qp_delta: 0,
                            luma_4x4_levels: luma_4x4_levels_scan,
                            luma_4x4_nc,
                            chroma_dc_cb: u_dc_levels,
                            chroma_dc_cr: v_dc_levels,
                            chroma_ac_cb: u_ac_levels,
                            chroma_ac_cr: v_ac_levels,
                        };
                        write_b_16x8_mb(sw, &cfg, 0, 0).expect("write B_*_16x8 mb");
                    }
                    PartitionShape::P8x16 => {
                        let cfg = B8x16McbConfig {
                            left: pc.mode_a,
                            right: pc.mode_b,
                            mvd_l0: mvd_l0_parts,
                            mvd_l1: mvd_l1_parts,
                            cbp_luma,
                            cbp_chroma,
                            mb_qp_delta: 0,
                            luma_4x4_levels: luma_4x4_levels_scan,
                            luma_4x4_nc,
                            chroma_dc_cb: u_dc_levels,
                            chroma_dc_cr: v_dc_levels,
                            chroma_ac_cb: u_ac_levels,
                            chroma_ac_cr: v_ac_levels,
                        };
                        write_b_8x16_mb(sw, &cfg, 0, 0).expect("write B_*_8x16 mb");
                    }
                }
            } else {
                let mb_cfg = B16x16McbConfig {
                    pred,
                    mvd_l0_x,
                    mvd_l0_y,
                    mvd_l1_x,
                    mvd_l1_y,
                    cbp_luma,
                    cbp_chroma,
                    mb_qp_delta: 0,
                    luma_4x4_levels: luma_4x4_levels_scan,
                    luma_4x4_nc,
                    chroma_dc_cb: u_dc_levels,
                    chroma_dc_cr: v_dc_levels,
                    chroma_ac_cb: u_ac_levels,
                    chroma_ac_cr: v_ac_levels,
                };
                write_b_16x16_mb(sw, &mb_cfg, 0, 0).expect("write B_*_16x16 mb");
            }
        }

        // 10. Update the L0 and L1 mv-pred grids. For partition modes,
        //     the per-8x8 slots carry per-partition MVs (top half =
        //     slots 0/1, bottom = 2/3 for 16x8; left = 0/2, right = 1/3
        //     for 8x16). For 16x16 / Direct, all four slots share the
        //     single MV. For Round-23 DirectPer8x8, every slot carries
        //     its own (mvL0, mvL1, refIdxL0, refIdxL1) per the
        //     §8.4.1.2.2 derivation.
        let (l0_mvs_8x8, l0_refs_8x8, l0_used) = if is_direct_per_8x8 {
            let used = direct.ref_idx_l0 >= 0;
            (
                direct.mv_l0_per_8x8,
                if used {
                    [direct.ref_idx_l0; 4]
                } else {
                    [-1; 4]
                },
                used,
            )
        } else if is_mixed_8x8 {
            // Round-25 — per-cell L0 (mv, ref). Direct cells take their
            // §8.4.1.2.2 derivation result; explicit L0/Bi cells take
            // their explicit MV with refIdxL0=0; L1 cells leave L0 unset.
            let mut mvs = [Mv::ZERO; 4];
            let mut refs = [-1i32; 4];
            let mut any_used = false;
            for cell in 0..4usize {
                let kind = mixed_cells[cell];
                match kind {
                    BSubMbCell::Direct => {
                        if direct.ref_idx_l0 >= 0 {
                            mvs[cell] = direct.mv_l0_per_8x8[cell];
                            refs[cell] = direct.ref_idx_l0;
                            any_used = true;
                        }
                    }
                    BSubMbCell::L0 | BSubMbCell::Bi => {
                        mvs[cell] = mixed_cell_mv_l0[cell];
                        refs[cell] = 0;
                        any_used = true;
                    }
                    BSubMbCell::L1 => {}
                }
            }
            (mvs, refs, any_used)
        } else {
            grid_state_b(
                partition_choice.as_ref(),
                true, // L0
                pred,
                eff_mv_l0,
            )
        };
        let (l1_mvs_8x8, l1_refs_8x8, l1_used) = if is_direct_per_8x8 {
            let used = direct.ref_idx_l1 >= 0;
            (
                direct.mv_l1_per_8x8,
                if used {
                    [direct.ref_idx_l1; 4]
                } else {
                    [-1; 4]
                },
                used,
            )
        } else if is_mixed_8x8 {
            let mut mvs = [Mv::ZERO; 4];
            let mut refs = [-1i32; 4];
            let mut any_used = false;
            for cell in 0..4usize {
                let kind = mixed_cells[cell];
                match kind {
                    BSubMbCell::Direct => {
                        if direct.ref_idx_l1 >= 0 {
                            mvs[cell] = direct.mv_l1_per_8x8[cell];
                            refs[cell] = direct.ref_idx_l1;
                            any_used = true;
                        }
                    }
                    BSubMbCell::L1 | BSubMbCell::Bi => {
                        mvs[cell] = mixed_cell_mv_l1[cell];
                        refs[cell] = 0;
                        any_used = true;
                    }
                    BSubMbCell::L0 => {}
                }
            }
            (mvs, refs, any_used)
        } else {
            grid_state_b(
                partition_choice.as_ref(),
                false, // L1
                pred,
                eff_mv_l1,
            )
        };
        *mv_grid_l0.slot_mut(mb_x, mb_y) = if l0_used {
            MvGridSlot {
                available: true,
                is_intra: false,
                ref_idx_l0_8x8: l0_refs_8x8,
                mv_l0_8x8: l0_mvs_8x8,
            }
        } else {
            MvGridSlot {
                available: true,
                is_intra: true,
                ref_idx_l0_8x8: [-1; 4],
                mv_l0_8x8: [Mv::ZERO; 4],
            }
        };
        *mv_grid_l1.slot_mut(mb_x, mb_y) = if l1_used {
            MvGridSlot {
                available: true,
                is_intra: false,
                ref_idx_l0_8x8: l1_refs_8x8,
                mv_l0_8x8: l1_mvs_8x8,
            }
        } else {
            MvGridSlot {
                available: true,
                is_intra: true,
                ref_idx_l0_8x8: [-1; 4],
                mv_l0_8x8: [Mv::ZERO; 4],
            }
        };
        let s = intra_grid.slot_mut(mb_x, mb_y);
        s.available = true;
        s.is_i_nxn = false;

        // 11. Per-4x4 nonzero mask + per-list MVs/refs/POCs for the §8.7
        //     deblock walker. For partition modes the per-4x4 MVs follow
        //     the §8.7.2.1 partition layout. POCs use the same anchor-
        //     layout sentinel as the 16x16 path. For Round-23
        //     DirectPer8x8 each 4x4 carries its containing 8x8's MV.
        let luma_nonzero_4x4 = luma_nz_mask_from_blocks(&blk_has_nz);
        let cb_nz: [bool; 4] =
            std::array::from_fn(|i| cbp_chroma == 2 && u_ac_levels[i].iter().any(|&v| v != 0));
        let cr_nz: [bool; 4] =
            std::array::from_fn(|i| cbp_chroma == 2 && v_ac_levels[i].iter().any(|&v| v != 0));

        let (mv_l0_arr, ref_idx_l0, ref_poc_l0) = if is_direct_per_8x8 {
            deblock_arrays_per_8x8_direct(direct.mv_l0_per_8x8, direct.ref_idx_l0, 0i32)
        } else if is_mixed_8x8 {
            deblock_arrays_per_8x8_mixed(l0_mvs_8x8, l0_refs_8x8, 0i32)
        } else {
            deblock_mv_arr_b(partition_choice.as_ref(), true, pred, eff_mv_l0, 0i32)
        };
        let (mv_l1_arr, ref_idx_l1, ref_poc_l1) = if is_direct_per_8x8 {
            deblock_arrays_per_8x8_direct(direct.mv_l1_per_8x8, direct.ref_idx_l1, 1000i32)
        } else if is_mixed_8x8 {
            deblock_arrays_per_8x8_mixed(l1_mvs_8x8, l1_refs_8x8, 1000i32)
        } else {
            deblock_mv_arr_b(partition_choice.as_ref(), false, pred, eff_mv_l1, 1000i32)
        };

        MbDeblockInfo {
            is_intra: false,
            qp_y,
            luma_nonzero_4x4,
            chroma_nonzero_4x4: chroma_nz_mask_from_blocks(&cb_nz, &cr_nz),
            mv_l0: mv_l0_arr,
            ref_idx_l0,
            ref_poc_l0,
            mv_l1: mv_l1_arr,
            ref_idx_l1,
            ref_poc_l1,
        }
    }

    /// Round-29 — encode one B-slice MB with optional **Intra_16x16
    /// fallback** RDO (per §7.3.5 Table 7-14 mb_type values 24..=47).
    /// Mirrors [`Encoder::encode_p_mb_with_intra_fallback`] but on the
    /// two-list B-slice context.
    #[allow(clippy::too_many_arguments)]
    fn encode_b_mb_with_intra_fallback(
        &self,
        frame: &YuvFrame<'_>,
        ref_l0: &EncodedFrameRef<'_>,
        ref_l1: &EncodedFrameRef<'_>,
        mb_x: usize,
        mb_y: usize,
        qp_y: i32,
        qp_c: i32,
        chroma_width: usize,
        chroma_height: usize,
        src_stride: usize,
        recon_y: &mut [u8],
        recon_u: &mut [u8],
        recon_v: &mut [u8],
        sw: &mut BitWriter,
        nc_grid: &mut CavlcNcGrid,
        intra_grid: &mut IntraGrid,
        mv_grid_l0: &mut MvGrid,
        mv_grid_l1: &mut MvGrid,
        pending_skip: &mut u32,
        curr_poc: i32,
        weighted: Option<WeightedBipredLuma>,
    ) -> MbDeblockInfo {
        if !self.cfg.intra_in_inter {
            return self.encode_b_mb(
                frame,
                ref_l0,
                ref_l1,
                mb_x,
                mb_y,
                qp_y,
                qp_c,
                chroma_width,
                chroma_height,
                src_stride,
                recon_y,
                recon_u,
                recon_v,
                sw,
                nc_grid,
                intra_grid,
                mv_grid_l0,
                mv_grid_l1,
                pending_skip,
                curr_poc,
                weighted,
            );
        }

        let width = self.cfg.width as usize;
        let width_mbs = (self.cfg.width / 16) as usize;
        let mb_addr = mb_y * width_mbs + mb_x;

        // -- Snapshot pre-state. --
        let snap_pre = InterMbStateSnapshot::capture_b(
            recon_y,
            recon_u,
            recon_v,
            width,
            chroma_width,
            mb_x,
            mb_y,
            sw,
            nc_grid,
            intra_grid,
            mv_grid_l0,
            mv_grid_l1,
            mb_addr,
            width_mbs,
            *pending_skip,
        );
        let pre_bits = sw.bits_emitted();

        // -- Trial 1: inter (B_Skip / B_Direct / B_L0/L1/Bi / partitions / 8x8). --
        let inter_dbl = self.encode_b_mb(
            frame,
            ref_l0,
            ref_l1,
            mb_x,
            mb_y,
            qp_y,
            qp_c,
            chroma_width,
            chroma_height,
            src_stride,
            recon_y,
            recon_u,
            recon_v,
            sw,
            nc_grid,
            intra_grid,
            mv_grid_l0,
            mv_grid_l1,
            pending_skip,
            curr_poc,
            weighted,
        );
        let inter_bits = sw.bits_emitted() - pre_bits;
        let inter_d = ssd_16x16(frame.y, width, mb_x * 16, mb_y * 16, &{
            let mut buf = [0i32; 256];
            for j in 0..16usize {
                for i in 0..16usize {
                    buf[j * 16 + i] = recon_y[(mb_y * 16 + j) * width + (mb_x * 16 + i)] as i32;
                }
            }
            buf
        });
        let lambda = lambda_ssd(qp_y, false);
        let j_inter = cost_combined(inter_d, inter_bits as u64, lambda);

        // Capture post-inter state.
        let snap_inter = InterMbStateSnapshot::capture_b(
            recon_y,
            recon_u,
            recon_v,
            width,
            chroma_width,
            mb_x,
            mb_y,
            sw,
            nc_grid,
            intra_grid,
            mv_grid_l0,
            mv_grid_l1,
            mb_addr,
            width_mbs,
            *pending_skip,
        );

        // -- Restore pre-state for the intra trial. --
        snap_pre.restore_b(
            recon_y,
            recon_u,
            recon_v,
            width,
            chroma_width,
            mb_x,
            mb_y,
            sw,
            nc_grid,
            intra_grid,
            mv_grid_l0,
            mv_grid_l1,
            mb_addr,
            pending_skip,
        );

        // -- Trial 2: Intra_16x16 in B-slice. --
        let intra_dbl = self.encode_b_mb_intra16x16(
            frame,
            mb_x,
            mb_y,
            qp_y,
            qp_c,
            chroma_width,
            chroma_height,
            recon_y,
            recon_u,
            recon_v,
            sw,
            nc_grid,
            intra_grid,
            mv_grid_l0,
            mv_grid_l1,
            pending_skip,
        );
        let intra_bits = sw.bits_emitted() - pre_bits;
        let intra_d = ssd_16x16(frame.y, width, mb_x * 16, mb_y * 16, &{
            let mut buf = [0i32; 256];
            for j in 0..16usize {
                for i in 0..16usize {
                    buf[j * 16 + i] = recon_y[(mb_y * 16 + j) * width + (mb_x * 16 + i)] as i32;
                }
            }
            buf
        });
        let j_intra = cost_combined(intra_d, intra_bits as u64, lambda);

        // -- Pick winner. --
        if j_inter <= j_intra {
            snap_pre.restore_b(
                recon_y,
                recon_u,
                recon_v,
                width,
                chroma_width,
                mb_x,
                mb_y,
                sw,
                nc_grid,
                intra_grid,
                mv_grid_l0,
                mv_grid_l1,
                mb_addr,
                pending_skip,
            );
            snap_inter.restore_b(
                recon_y,
                recon_u,
                recon_v,
                width,
                chroma_width,
                mb_x,
                mb_y,
                sw,
                nc_grid,
                intra_grid,
                mv_grid_l0,
                mv_grid_l1,
                mb_addr,
                pending_skip,
            );
            inter_dbl
        } else {
            intra_dbl
        }
    }

    /// Round-29 — Intra_16x16 fallback for a B-slice MB. Mirrors
    /// [`Encoder::encode_p_mb_intra16x16`] but updates *both* L0 and L1
    /// MV grids (intra contribution for either list).
    #[allow(clippy::too_many_arguments)]
    fn encode_b_mb_intra16x16(
        &self,
        frame: &YuvFrame<'_>,
        mb_x: usize,
        mb_y: usize,
        qp_y: i32,
        qp_c: i32,
        chroma_width: usize,
        chroma_height: usize,
        recon_y: &mut [u8],
        recon_u: &mut [u8],
        recon_v: &mut [u8],
        sw: &mut BitWriter,
        nc_grid: &mut CavlcNcGrid,
        intra_grid: &mut IntraGrid,
        mv_grid_l0: &mut MvGrid,
        mv_grid_l1: &mut MvGrid,
        pending_skip: &mut u32,
    ) -> MbDeblockInfo {
        // Flush any pending B_Skip run.
        sw.ue(*pending_skip);
        *pending_skip = 0;

        self.encode_intra16x16_in_inter_slice(
            frame,
            mb_x,
            mb_y,
            qp_y,
            qp_c,
            chroma_width,
            chroma_height,
            recon_y,
            recon_u,
            recon_v,
            sw,
            nc_grid,
            intra_grid,
            23,
        );

        // Update both MV grids for the intra MB.
        let intra_slot = MvGridSlot {
            available: true,
            is_intra: true,
            ref_idx_l0_8x8: [-1; 4],
            mv_l0_8x8: [Mv::ZERO; 4],
        };
        *mv_grid_l0.slot_mut(mb_x, mb_y) = intra_slot;
        *mv_grid_l1.slot_mut(mb_x, mb_y) = intra_slot;

        let width_mbs = (self.cfg.width / 16) as usize;
        let mb_addr = mb_y * width_mbs + mb_x;
        let any_nz = nc_grid.mbs[mb_addr]
            .luma_total_coeff
            .iter()
            .any(|&v| v != 0);
        let luma_nonzero_4x4: u16 = if any_nz {
            let mut nz = [false; 16];
            for (blkz, slot) in nz.iter_mut().enumerate() {
                *slot = nc_grid.mbs[mb_addr].luma_total_coeff[blkz] != 0;
            }
            luma_nz_mask_from_blocks(&nz)
        } else {
            0
        };

        MbDeblockInfo {
            is_intra: true,
            qp_y,
            luma_nonzero_4x4,
            chroma_nonzero_4x4: 0,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod weighted_bipred_tests {
    use super::*;

    /// Hand-built fade fixture: textured frames at two brightness
    /// levels with a true linear cross-fade. Texture (not just brightness)
    /// rules out the degenerate "all-on-L1, offset compensates" solution
    /// — the weights have to honour the per-sample mixing to keep SSE
    /// low. The selector should land on weights near (45, 19) at
    /// log2_wd = 5, i.e. α ≈ 0.7 toward L0.
    #[test]
    fn weight_selector_finds_alpha_07_on_textured_fade() {
        let w = 16usize;
        let h = 16usize;
        let mut l0 = vec![0u8; w * h];
        let mut l1 = vec![0u8; w * h];
        let mut src = vec![0u8; w * h];
        for j in 0..h {
            for i in 0..w {
                // L0 and L1 have *different* texture scales so an offset
                // alone cannot disguise itself as a weight choice. With
                // L0 amplitude 60 and L1 amplitude 20, only the right α
                // can match the source's mixed texture amplitude
                // 0.7 * 60 + 0.3 * 20 = 42.
                let tex_a = (((i * 13 + j * 7) % 16) as i32) * 4; // 0..=60
                let tex_b = ((i * 13 + j * 7) % 16) as i32; // 0..=15
                let a = 30 + tex_a;
                let b = 180 + tex_b;
                let v = (0.7 * a as f64 + 0.3 * b as f64).round() as i32;
                l0[j * w + i] = a.clamp(0, 255) as u8;
                l1[j * w + i] = b.clamp(0, 255) as u8;
                src[j * w + i] = v.clamp(0, 255) as u8;
            }
        }
        let table = compute_explicit_bipred_weights(&src, &l0, &l1, w, h, w, w);
        assert_eq!(table.luma_log2_weight_denom, 5);
        // w0 + w1 must sum to 64 (denom for log2_wd = 5).
        assert_eq!(
            table.luma_weight_l0 + table.luma_weight_l1,
            64,
            "weights must sum to 2^(logwd+1)"
        );
        // Coarse grid step is 4 → α=0.7 lands on w0 ∈ {44, 48}; we expect
        // the selector to pick the better neighbour.
        assert!(
            (40..=48).contains(&table.luma_weight_l0),
            "expected w0 ≈ 45 (α=0.7), got {}",
            table.luma_weight_l0,
        );
    }

    /// On content where the source IS the linear midpoint of L0 and L1
    /// the selector should fall back to the default (32, 32, 0, 0)
    /// because the explicit table wouldn't beat the default merge.
    #[test]
    fn weight_selector_returns_default_on_midpoint_fade() {
        let w = 16usize;
        let h = 16usize;
        let l0 = vec![60u8; w * h];
        let l1 = vec![160u8; w * h];
        // Midpoint: (60+160+1)>>1 = 110.
        let src = vec![110u8; w * h];
        let table = compute_explicit_bipred_weights(&src, &l0, &l1, w, h, w, w);
        assert_eq!(table.luma_weight_l0, 32);
        assert_eq!(table.luma_weight_l1, 32);
        assert_eq!(table.luma_offset_l0, 0);
        assert_eq!(table.luma_offset_l1, 0);
    }
}

#[cfg(test)]
mod level_derivation_tests {
    use super::*;

    /// Annex A Table A-1 — `min_level_idc_for_picture_size` walks the
    /// MaxFS column low→high and picks the first entry whose `MaxFS`
    /// is at least the picture's `width_in_mbs * height_in_mbs`.
    #[test]
    fn min_level_idc_picks_smallest_covering_level() {
        // QCIF 176x144 → 99 mb → Level 1 (MaxFS = 99).
        assert_eq!(min_level_idc_for_picture_size(11, 9), 10);
        // CIF 352x288 → 396 mb → Level 1.1 (first 396-bucket entry).
        assert_eq!(min_level_idc_for_picture_size(22, 18), 11);
        // 4CIF 704x576 → 1584 mb → Level 2.2 (MaxFS = 1620, first 1620
        // bucket since 2.1 has 792 and 2 has 396).
        assert_eq!(min_level_idc_for_picture_size(44, 36), 22);
        // SD 720x576 → 1620 mb → Level 2.2 (1620 = MaxFS exact).
        assert_eq!(min_level_idc_for_picture_size(45, 36), 22);
        // 720p 1280x720 → 3600 mb → Level 3.1 (MaxFS = 3600 exact).
        assert_eq!(min_level_idc_for_picture_size(80, 45), 31);
        // 1080p 1920x1080 → 8160 mb → Level 4 (MaxFS = 8192).
        assert_eq!(min_level_idc_for_picture_size(120, 68), 40);
        // 4K UHD 3840x2160 → 32400 mb → Level 5.1 (MaxFS = 36864).
        assert_eq!(min_level_idc_for_picture_size(240, 135), 51);
        // 8K UHD 7680x4320 → 129600 mb → Level 6 (MaxFS = 139264).
        assert_eq!(min_level_idc_for_picture_size(480, 270), 60);
        // Pathological huge picture beyond every standardised level
        // (e.g. 16K-equivalent) → cap at Level 6.2.
        assert_eq!(min_level_idc_for_picture_size(1024, 1024), 62);
    }

    /// Default-derived encoder config picks the level via
    /// [`min_level_idc_for_picture_size`]. Pre-fix the constructor
    /// always emitted `level_idc = 30` regardless of size.
    #[test]
    fn encoder_config_default_level_matches_picture_size() {
        // 64x64 (the synthetic test fixture) → 16 mb → Level 1.
        let cfg = EncoderConfig::new(64, 64);
        assert_eq!(cfg.level_idc, 10);

        // 720p
        let cfg = EncoderConfig::new(1280, 720);
        assert_eq!(cfg.level_idc, 31);

        // 1080p
        let cfg = EncoderConfig::new(1920, 1080);
        assert_eq!(cfg.level_idc, 40);

        // 4K UHD
        let cfg = EncoderConfig::new(3840, 2160);
        assert_eq!(cfg.level_idc, 51);
    }

    /// Non-multiple-of-16 dimensions round up to whole MBs (the SPS
    /// field is `pic_width_in_mbs_minus1`, which is in MB units).
    #[test]
    fn non_aligned_dimensions_round_up_to_whole_mbs() {
        // 1920x1088 should equal 1920x1080 (both round to 120x68 mb).
        assert_eq!(EncoderConfig::new(1920, 1088).level_idc, 40);
        // 1920x1081 still rounds to 120x68 → Level 4.
        assert_eq!(EncoderConfig::new(1920, 1081).level_idc, 40);
    }
}
