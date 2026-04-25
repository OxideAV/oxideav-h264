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
pub mod cavlc;
pub mod deblock;
pub mod intra4x4;
pub mod intra_pred;
pub mod macroblock;
pub mod nal;
pub mod pps;
pub mod slice;
pub mod sps;
pub mod transform;

use crate::cavlc::CoeffTokenContext;
use crate::encoder::bitstream::BitWriter;
use crate::encoder::deblock::{
    chroma_nz_mask_from_blocks, deblock_recon, luma_nz_mask_from_blocks, MbDeblockInfo,
};
use crate::encoder::intra4x4::{
    availability as i4x4_avail, predict_4x4, sad_4x4, Intra4x4Mode, LUMA_4X4_XY as I4X4_XY,
};
use crate::encoder::intra_pred::{
    predict_16x16, predict_chroma_8x8, sad_16x16, sad_8x8, I16x16Mode, IntraChromaMode,
};
use crate::encoder::macroblock::{
    write_i_nxn_mb, write_intra16x16_mb, I16x16McbConfig, INxNMcbConfig,
};
use crate::encoder::nal::build_nal_unit;
use crate::encoder::pps::{build_baseline_pps_rbsp, BaselinePpsConfig};
use crate::encoder::slice::{write_idr_i_slice_header, IdrSliceHeaderConfig};
use crate::encoder::sps::{build_baseline_sps_rbsp, BaselineSpsConfig};
use crate::encoder::transform::{
    forward_core_4x4, forward_hadamard_2x2, forward_hadamard_4x4, quantize_4x4, quantize_4x4_ac,
    quantize_chroma_dc, quantize_luma_dc, zigzag_scan_4x4, zigzag_scan_4x4_ac,
};
use crate::macroblock_layer::{derive_nc_luma, CavlcNcGrid, LumaNcKind};
use crate::nal::NalUnitType;
use crate::transform::{
    inverse_hadamard_chroma_dc_420, inverse_hadamard_luma_dc_16x16, inverse_scan_4x4_zigzag_ac,
    inverse_transform_4x4, inverse_transform_4x4_dc_preserved, qp_y_to_qp_c, FLAT_4X4_16,
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

/// One encoded YUV 4:2:0 frame's worth of input planes. All planes are
/// 8-bit, row-major. Width and height must be multiples of 16 (no
/// cropping support yet).
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
}

impl EncoderConfig {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            qp: 26,
        }
    }
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

/// Top-level encoder. Holds parameter set state but no per-picture
/// scratch — `encode_idr` is the one entry point.
pub struct Encoder {
    cfg: EncoderConfig,
}

impl Encoder {
    pub fn new(cfg: EncoderConfig) -> Self {
        assert!(
            cfg.width % 16 == 0 && cfg.height % 16 == 0,
            "encoder requires 16-pixel-aligned dimensions",
        );
        assert!((0..=51).contains(&cfg.qp));
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
        let chroma_width = (self.cfg.width / 2) as usize;
        let chroma_height = (self.cfg.height / 2) as usize;

        // SPS / PPS bytes.
        let sps_cfg = BaselineSpsConfig {
            seq_parameter_set_id: 0,
            level_idc: 30,
            width_in_mbs: width_mbs,
            height_in_mbs: height_mbs,
            log2_max_frame_num_minus4: 4,
            log2_max_poc_lsb_minus4: 4,
            max_num_ref_frames: 1,
        };
        let pps_cfg = BaselinePpsConfig {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
            pic_init_qp_minus26: self.cfg.qp - 26,
            chroma_qp_index_offset: 0,
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
                disable_deblocking_filter_idc: 0,
                slice_alpha_c0_offset_div2: 0,
                slice_beta_offset_div2: 0,
            },
        );

        // Reconstruction state, sized to the picture.
        let mut recon_y = vec![0u8; (self.cfg.width * self.cfg.height) as usize];
        let mut recon_u = vec![0u8; chroma_width * chroma_height];
        let mut recon_v = vec![0u8; chroma_width * chroma_height];

        let qp_y = self.cfg.qp;
        let qp_c = qp_y_to_qp_c(qp_y, pps_cfg.chroma_qp_index_offset);

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
        deblock_recon(
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
        );

        EncodedIdr {
            annex_b: stream,
            recon_y,
            recon_u,
            recon_v,
            recon_width: self.cfg.width,
            recon_height: self.cfg.height,
        }
    }

    /// Round-4 entry point: pick the lower-SAD mb_type (I_16x16 vs
    /// I_NxN) and dispatch.
    ///
    /// Strategy:
    ///   1) Compute the best I_16x16 mode + its 16x16 SAD against source.
    ///   2) Compute the best I_NxN per-block modes (9 candidates each)
    ///      against source, summing SAD.
    ///   3) Whichever has lower total SAD wins. The winning encoder
    ///      path runs the actual residual + reconstruction + bitstream
    ///      emit. The losing path is discarded.
    ///
    /// The I_NxN trial uses prediction-only SAD (no residual), so the
    /// comparison is biased: at low QP, I_NxN should always win on
    /// detailed content (more granular prediction); at high QP, AC
    /// residual on I_16x16 may produce an equivalent result. We accept
    /// the bias for now — round 5+ can add full RDO with residual cost.
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
        let height = self.cfg.height as usize;

        // ----- I_16x16 mode-decision SAD -----
        let mut best_16x16_sad = u64::MAX;
        for &mode in &[
            I16x16Mode::Vertical,
            I16x16Mode::Horizontal,
            I16x16Mode::Dc,
            I16x16Mode::Plane,
        ] {
            let Some(pred) = predict_16x16(mode, recon_y, width, height, mb_x, mb_y) else {
                continue;
            };
            let sad = sad_16x16(frame.y, width, mb_x * 16, mb_y * 16, &pred);
            if sad < best_16x16_sad {
                best_16x16_sad = sad;
            }
        }

        // ----- I_NxN per-block prediction-only SAD -----
        // Note: this scores predictors against the *source* before
        // residual quantisation. It picks a winner mode per block but
        // the absolute SAD comparison is approximate vs I_16x16 (which
        // includes residual encoding implicitly via the same ranking).
        let intra4x4_sad = self.score_i_nxn_sad(frame, recon_y, mb_x, mb_y);

        if intra4x4_sad < best_16x16_sad {
            self.encode_mb_intra4x4(
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
            )
        } else {
            self.encode_mb_intra16x16(
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
            )
        }
    }

    /// Compute the best I_NxN total SAD (sum of best per-block SADs).
    /// Used by the mb_type decision in [`Encoder::encode_mb`].
    fn score_i_nxn_sad(
        &self,
        frame: &YuvFrame<'_>,
        recon_y: &[u8],
        mb_x: usize,
        mb_y: usize,
    ) -> u64 {
        let width = self.cfg.width as usize;
        let width_mbs = (self.cfg.width / 16) as usize;
        let mut total: u64 = 0;
        for &(bx, by) in &I4X4_XY {
            let avail = i4x4_avail(mb_x, mb_y, bx, by, width_mbs);
            let mut best = u64::MAX;
            for &mode in &Intra4x4Mode::ALL {
                let Some(pred) = predict_4x4(mode, recon_y, width, mb_x, mb_y, bx, by, avail)
                else {
                    continue;
                };
                let sad = sad_4x4(frame.y, width, mb_x * 16 + bx, mb_y * 16 + by, &pred);
                if sad < best {
                    best = sad;
                }
            }
            // If somehow no mode was available (shouldn't happen — DC
            // is always available), penalize but don't crash.
            if best == u64::MAX {
                best = u64::MAX / 2;
            }
            total = total.saturating_add(best);
        }
        total
    }

    /// Original Intra_16x16 path (round-3 logic).
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
    ) -> MbDeblockInfo {
        let width = self.cfg.width as usize;
        let height = self.cfg.height as usize;

        // ------- Luma: pick mode by SAD -------
        // §8.3.3 — compute all available candidates against the local
        // recon buffer, score against the source, pick the lowest.
        // Modes whose required neighbours are missing are skipped (the
        // decoder would reject them if we tried to signal them).
        let mut best_luma_mode = I16x16Mode::Dc;
        let mut best_luma_pred = [0i32; 256];
        let mut best_luma_sad = u64::MAX;
        for &mode in &[
            I16x16Mode::Vertical,
            I16x16Mode::Horizontal,
            I16x16Mode::Dc,
            I16x16Mode::Plane,
        ] {
            let Some(pred) = predict_16x16(mode, recon_y, width, height, mb_x, mb_y) else {
                continue;
            };
            let sad = sad_16x16(frame.y, width, mb_x * 16, mb_y * 16, &pred);
            if sad < best_luma_sad {
                best_luma_sad = sad;
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
        let cbp_luma: u8 = if any_luma_ac_nz { 15 } else { 0 };

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

        // ------- Chroma: pick mode by SAD on both planes -------
        // §8.3.4 — same idea as luma but jointly across Cb/Cr (one
        // chroma_pred_mode covers both planes).
        let mut best_chroma_mode = IntraChromaMode::Dc;
        let mut best_chroma_pred_u = [0i32; 64];
        let mut best_chroma_pred_v = [0i32; 64];
        let mut best_chroma_sad = u64::MAX;
        for &mode in &[
            IntraChromaMode::Dc,
            IntraChromaMode::Horizontal,
            IntraChromaMode::Vertical,
            IntraChromaMode::Plane,
        ] {
            let (Some(pu), Some(pv)) = (
                predict_chroma_8x8(mode, recon_u, chroma_width, chroma_height, mb_x, mb_y),
                predict_chroma_8x8(mode, recon_v, chroma_width, chroma_height, mb_x, mb_y),
            ) else {
                continue;
            };
            let s = sad_8x8(frame.u, chroma_width, mb_x * 8, mb_y * 8, &pu)
                + sad_8x8(frame.v, chroma_width, mb_x * 8, mb_y * 8, &pv);
            if s < best_chroma_sad {
                best_chroma_sad = s;
                best_chroma_mode = mode;
                best_chroma_pred_u = pu;
                best_chroma_pred_v = pv;
            }
        }

        // Encode chroma residual for both planes. The decoder consumes
        // both planes' DC residual when cbp_chroma >= 1, and both
        // planes' AC residual when cbp_chroma == 2. Our encoder always
        // emits chroma DC, and emits AC when at least one AC level is
        // non-zero.
        let (u_dc_levels, u_ac_levels, u_recon_residual) =
            encode_chroma_residual(frame.u, chroma_width, mb_x, mb_y, &best_chroma_pred_u, qp_c);
        let (v_dc_levels, v_ac_levels, v_recon_residual) =
            encode_chroma_residual(frame.v, chroma_width, mb_x, mb_y, &best_chroma_pred_v, qp_c);

        // Decide cbp_chroma: 1 if either plane has any non-zero DC,
        // 2 if either plane has any non-zero AC. (Round-2 emits both
        // planes' DC unconditionally when cbp_chroma >= 1.)
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

        // Apply chroma reconstruction. When cbp_chroma == 0 nothing was
        // sent, so the decoder will fall back to the predictor — match.
        for j in 0..8usize {
            for i in 0..8usize {
                let cx = mb_x * 8 + i;
                let cy = mb_y * 8 + j;
                let pred_u = best_chroma_pred_u[j * 8 + i];
                let pred_v = best_chroma_pred_v[j * 8 + i];
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
                recon_u[cy * chroma_width + cx] = (pred_u + res_u).clamp(0, 255) as u8;
                recon_v[cy * chroma_width + cx] = (pred_v + res_v).clamp(0, 255) as u8;
            }
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
        let mb_cfg = I16x16McbConfig {
            pred_mode: best_luma_mode.as_u8(),
            intra_chroma_pred_mode: best_chroma_mode.as_u8(),
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
        write_intra16x16_mb(sw, &mb_cfg, CoeffTokenContext::Numeric(nc_dc))
            .expect("write Intra16x16 mb");

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
        // Chroma per-block AC nonzero (4 blocks × 2 planes).
        let cb_nz: [bool; 4] =
            std::array::from_fn(|i| cbp_chroma == 2 && u_ac_levels[i].iter().any(|&v| v != 0));
        let cr_nz: [bool; 4] =
            std::array::from_fn(|i| cbp_chroma == 2 && v_ac_levels[i].iter().any(|&v| v != 0));
        MbDeblockInfo {
            is_intra: true,
            qp_y,
            luma_nonzero_4x4,
            chroma_nonzero_4x4: chroma_nz_mask_from_blocks(&cb_nz, &cr_nz),
        }
    }

    /// I_NxN (Intra_4x4) path: per-block 9-mode prediction, full luma
    /// 4x4 residual, chroma residual identical to I_16x16. Round-4.
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
    ) -> MbDeblockInfo {
        let width = self.cfg.width as usize;
        let width_mbs = (self.cfg.width / 16) as usize;

        // Per-block: pick mode by SAD against source.
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

        for blk in 0..16usize {
            let (bx, by) = I4X4_XY[blk];
            let avail = i4x4_avail(mb_x, mb_y, bx, by, width_mbs);

            // Pick best mode by SAD on the predictor.
            let mut best_mode = Intra4x4Mode::Dc;
            let mut best_sad = u64::MAX;
            let mut best_pred = [0i32; 16];
            for &mode in &Intra4x4Mode::ALL {
                let Some(pred) = predict_4x4(mode, recon_y, width, mb_x, mb_y, bx, by, avail)
                else {
                    continue;
                };
                let sad = sad_4x4(frame.y, width, mb_x * 16 + bx, mb_y * 16 + by, &pred);
                if sad < best_sad {
                    best_sad = sad;
                    best_mode = mode;
                    best_pred = pred;
                }
            }
            let chosen = best_mode.as_u8();
            chosen_mode[blk] = chosen;

            // Derive predicted mode per §8.3.1.1 step 4 and the
            // `prev_flag` / `rem` syntax that signals our chosen mode.
            let predicted = predicted_intra4x4_mode(intra_grid, mb_x, mb_y, blk, width_mbs);
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

            // Build the residual: source - predictor.
            let mut block_res = [0i32; 16];
            for j in 0..4 {
                for i in 0..4 {
                    let s = frame.y[(mb_y * 16 + by + j) * width + (mb_x * 16 + bx + i)] as i32;
                    block_res[j * 4 + i] = s - best_pred[j * 4 + i];
                }
            }
            // Forward 4x4 + quantize (full block, including DC slot 0).
            let w_coeffs = forward_core_4x4(&block_res);
            let z_raster = quantize_4x4(&w_coeffs, qp_y, true);
            luma_4x4_quant_raster[blk] = z_raster;
            // Pack into scan order (raster → scan).
            let z_scan = zigzag_scan_4x4(&z_raster);
            luma_4x4_levels_scan[blk] = z_scan;
            blk_has_nz[blk] = z_raster.iter().any(|&v| v != 0);

            // Inverse: decode the residual back, add to predictor,
            // write to recon buffer so subsequent blocks see it.
            // Note: when the block's 8x8 quadrant cbp bit is 0 the
            // decoder will use [0; 16] for this block — we have to
            // reflect that during the cbp_luma decision below. For
            // now write the *post-residual* recon assuming the block
            // is transmitted; we'll roll back zero-quadrant blocks
            // in a second pass.
            let r = inverse_transform_4x4(&z_raster, qp_y, &FLAT_4X4_16, 8).expect("inverse 4x4");
            for j in 0..4 {
                for i in 0..4 {
                    let px = mb_x * 16 + bx + i;
                    let py = mb_y * 16 + by + j;
                    let p = best_pred[j * 4 + i];
                    let raw = p + r[j * 4 + i];
                    recon_y[py * width + px] = raw.clamp(0, 255) as u8;
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

        // ----- Chroma: same path as I_16x16. -----
        let mut best_chroma_mode = IntraChromaMode::Dc;
        let mut best_chroma_pred_u = [0i32; 64];
        let mut best_chroma_pred_v = [0i32; 64];
        let mut best_chroma_sad = u64::MAX;
        for &mode in &[
            IntraChromaMode::Dc,
            IntraChromaMode::Horizontal,
            IntraChromaMode::Vertical,
            IntraChromaMode::Plane,
        ] {
            let (Some(pu), Some(pv)) = (
                predict_chroma_8x8(mode, recon_u, chroma_width, chroma_height, mb_x, mb_y),
                predict_chroma_8x8(mode, recon_v, chroma_width, chroma_height, mb_x, mb_y),
            ) else {
                continue;
            };
            let s = sad_8x8(frame.u, chroma_width, mb_x * 8, mb_y * 8, &pu)
                + sad_8x8(frame.v, chroma_width, mb_x * 8, mb_y * 8, &pv);
            if s < best_chroma_sad {
                best_chroma_sad = s;
                best_chroma_mode = mode;
                best_chroma_pred_u = pu;
                best_chroma_pred_v = pv;
            }
        }
        let (u_dc_levels, u_ac_levels, u_recon_residual) =
            encode_chroma_residual(frame.u, chroma_width, mb_x, mb_y, &best_chroma_pred_u, qp_c);
        let (v_dc_levels, v_ac_levels, v_recon_residual) =
            encode_chroma_residual(frame.v, chroma_width, mb_x, mb_y, &best_chroma_pred_v, qp_c);
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
        for j in 0..8usize {
            for i in 0..8usize {
                let cx = mb_x * 8 + i;
                let cy = mb_y * 8 + j;
                let pred_u = best_chroma_pred_u[j * 8 + i];
                let pred_v = best_chroma_pred_v[j * 8 + i];
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
                recon_u[cy * chroma_width + cx] = (pred_u + res_u).clamp(0, 255) as u8;
                recon_v[cy * chroma_width + cx] = (pred_v + res_v).clamp(0, 255) as u8;
            }
        }
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

        // intra_grid was updated incrementally above; nothing else to
        // commit here.

        // §8.7.2.1 — per-4x4 nonzero mask. For I_NxN the per-block
        // nonzero status is exactly the per-block AC scan flags computed
        // earlier (`blk_has_nz`); blocks in zero-cbp 8x8 quadrants are
        // already false there.
        let luma_nonzero_4x4 = luma_nz_mask_from_blocks(&blk_has_nz);
        let cb_nz: [bool; 4] =
            std::array::from_fn(|i| cbp_chroma == 2 && u_ac_levels[i].iter().any(|&v| v != 0));
        let cr_nz: [bool; 4] =
            std::array::from_fn(|i| cbp_chroma == 2 && v_ac_levels[i].iter().any(|&v| v != 0));
        MbDeblockInfo {
            is_intra: true,
            qp_y,
            luma_nonzero_4x4,
            chroma_nonzero_4x4: chroma_nz_mask_from_blocks(&cb_nz, &cr_nz),
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
