//! H.264 / AVC **encoder** — Baseline profile, IDR-only, I_16x16
//! (DC/V/H/Plane) with chroma DC + AC residual.
//!
//! Round-2 scope (round-1 was DC-luma only, no chroma residual; see
//! `STATUS.md`):
//!
//! * **Profile:** Baseline (`profile_idc = 66`), 4:2:0, 8-bit, single
//!   slice per picture, no FMO.
//! * **Slice types:** IDR I-slices only.
//! * **Macroblock types:** every MB is `I_16x16`, with the four
//!   §8.3.3 prediction modes (V, H, DC, Plane) selected per MB by
//!   minimum-SAD against the source.
//! * **Chroma intra:** §8.3.4 with the four modes (DC, V, H, Plane)
//!   selected per MB by minimum-SAD.
//! * **Transforms:** 4x4 integer transform + 4x4 luma-DC Hadamard +
//!   2x2 chroma-DC Hadamard. Flat scaling lists.
//! * **CBP:** `cbp_luma` is still 0 (no luma AC residual). `cbp_chroma`
//!   is 1 (chroma DC) by default; promoted to 2 (chroma DC + AC) when
//!   the AC residual on either plane carries non-zero levels.
//! * **Entropy:** CAVLC.
//! * **In-loop filter:** disabled (`disable_deblocking_filter_idc = 1`).
//!
//! Spec references throughout cite ITU-T Rec. H.264 (08/2024).

#![allow(dead_code)]

pub mod bitstream;
pub mod cavlc;
pub mod intra_pred;
pub mod macroblock;
pub mod nal;
pub mod pps;
pub mod slice;
pub mod sps;
pub mod transform;

use crate::cavlc::CoeffTokenContext;
use crate::encoder::bitstream::BitWriter;
use crate::encoder::intra_pred::{
    predict_16x16, predict_chroma_8x8, sad_16x16, sad_8x8, I16x16Mode, IntraChromaMode,
};
use crate::encoder::macroblock::{write_intra16x16_mb, I16x16McbConfig};
use crate::encoder::nal::build_nal_unit;
use crate::encoder::pps::{build_baseline_pps_rbsp, BaselinePpsConfig};
use crate::encoder::slice::{write_idr_i_slice_header, IdrSliceHeaderConfig};
use crate::encoder::sps::{build_baseline_sps_rbsp, BaselineSpsConfig};
use crate::encoder::transform::{
    forward_core_4x4, forward_hadamard_2x2, forward_hadamard_4x4, quantize_4x4_ac,
    quantize_chroma_dc, quantize_luma_dc, zigzag_scan_4x4_ac,
};
use crate::nal::NalUnitType;
use crate::transform::{
    inverse_hadamard_chroma_dc_420, inverse_hadamard_luma_dc_16x16, inverse_scan_4x4_zigzag_ac,
    inverse_transform_4x4_dc_preserved, qp_y_to_qp_c, FLAT_4X4_16,
};

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
            },
        );

        // Reconstruction state, sized to the picture.
        let mut recon_y = vec![0u8; (self.cfg.width * self.cfg.height) as usize];
        let mut recon_u = vec![0u8; chroma_width * chroma_height];
        let mut recon_v = vec![0u8; chroma_width * chroma_height];

        let qp_y = self.cfg.qp;
        let qp_c = qp_y_to_qp_c(qp_y, pps_cfg.chroma_qp_index_offset);

        // Iterate MBs in raster order.
        for mb_y in 0..height_mbs {
            for mb_x in 0..width_mbs {
                self.encode_mb(
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
                );
            }
        }

        // §7.3.2.8 / §7.3.2.9 — slice_data trailing bits.
        sw.rbsp_trailing_bits();
        let slice_rbsp = sw.into_bytes();
        stream.extend_from_slice(&build_nal_unit(3, NalUnitType::SliceIdr, &slice_rbsp));

        EncodedIdr {
            annex_b: stream,
            recon_y,
            recon_u,
            recon_v,
            recon_width: self.cfg.width,
            recon_height: self.cfg.height,
        }
    }

    /// Encode one macroblock and update the reconstruction buffers.
    /// Splits the encode into:
    ///   1) Luma mode decision (V/H/DC/Plane) by SAD.
    ///   2) Luma DC + AC encode + reconstruction.
    ///   3) Chroma mode decision (DC/V/H/Plane) by SAD.
    ///   4) Chroma DC + AC encode + reconstruction.
    ///   5) Macroblock layer emit.
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
    ) {
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

        // Round-2 still uses cbp_luma == 0: discard luma AC. The
        // inverse path uses only the DC reconstruction (matches
        // §8.5.10 / §8.5.12 with all AC coefficients == 0).
        let inv_dc = inverse_hadamard_luma_dc_16x16(&dc_levels, qp_y, &FLAT_4X4_16, 8)
            .expect("luma DC inverse");
        for blk in 0..16usize {
            let bx = blk % 4;
            let by = blk / 4;
            let mut coeffs_in = [0i32; 16];
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

        // ------- Emit MB syntax -------
        // §9.2.1.1 NOTE 1: nC for Intra16x16DCLevel uses *AC* TotalCoeff
        // of neighbour luma4x4 block 0. With cbp_luma == 0 always, the
        // AC TotalCoeff is identically zero, so nC = 0 across the
        // picture — the encoder mirrors the decoder.
        let nc_dc = 0;
        let mb_cfg = I16x16McbConfig {
            pred_mode: best_luma_mode.as_u8(),
            intra_chroma_pred_mode: best_chroma_mode.as_u8(),
            cbp_luma: 0,
            cbp_chroma,
            mb_qp_delta: 0,
            luma_dc_levels_raster: dc_levels,
            chroma_dc_cb: u_dc_levels,
            chroma_dc_cr: v_dc_levels,
            chroma_ac_cb: u_ac_levels,
            chroma_ac_cr: v_ac_levels,
        };
        write_intra16x16_mb(sw, &mb_cfg, CoeffTokenContext::Numeric(nc_dc))
            .expect("write Intra16x16 mb");
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
