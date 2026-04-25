//! H.264 / AVC **encoder** — Baseline profile, IDR-only, I_16x16 DC.
//!
//! Round-1 scope (see `STATUS.md` in this folder for the full plan):
//!
//! * **Profile:** Baseline (`profile_idc = 66`), 4:2:0, 8-bit, single
//!   slice per picture, no FMO.
//! * **Slice types:** IDR I-slices only.
//! * **Macroblock types:** every MB is `I_16x16` with `pred_mode = 2`
//!   (DC luma) and `intra_chroma_pred_mode = 0` (DC chroma).
//! * **Transforms:** 4x4 integer transform + 4x4 luma-DC Hadamard +
//!   2x2 chroma-DC Hadamard. Flat scaling lists (`weight_scale = 16`).
//! * **Entropy:** CAVLC (no CABAC).
//! * **In-loop filter:** disabled (`disable_deblocking_filter_idc = 1`).
//!
//! The encoder maintains a local reconstruction buffer that mirrors what
//! the decoder will produce sample-for-sample — the same DC prediction,
//! quantization, inverse transform, and clip steps run on both sides.
//! That gives us a deterministic round-trip target: encode → decode
//! should reproduce the encoder's local recon exactly.
//!
//! Spec references throughout cite ITU-T Rec. H.264 (08/2024).

#![allow(dead_code)]

pub mod bitstream;
pub mod cavlc;
pub mod macroblock;
pub mod nal;
pub mod pps;
pub mod slice;
pub mod sps;
pub mod transform;

use crate::cavlc::CoeffTokenContext;
use crate::encoder::bitstream::BitWriter;
use crate::encoder::macroblock::{intra16x16_mb_type_value, write_intra16x16_mb, I16x16McbConfig};
use crate::encoder::nal::build_nal_unit;
use crate::encoder::pps::{build_baseline_pps_rbsp, BaselinePpsConfig};
use crate::encoder::slice::{write_idr_i_slice_header, IdrSliceHeaderConfig};
use crate::encoder::sps::{build_baseline_sps_rbsp, BaselineSpsConfig};
use crate::encoder::transform::{
    forward_core_4x4, forward_hadamard_2x2, forward_hadamard_4x4, quantize_4x4, quantize_chroma_dc,
    quantize_luma_dc,
};
use crate::nal::NalUnitType;
use crate::transform::{
    inverse_hadamard_chroma_dc_420, inverse_hadamard_luma_dc_16x16, inverse_transform_4x4,
    qp_y_to_qp_c, FLAT_4X4_16,
};

/// One encoded YUV 4:2:0 frame's worth of input planes. All planes are
/// 8-bit, row-major. Width and height must be multiples of 16 (no
/// cropping support in round 1).
pub struct YuvFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
}

/// Round-1 encoder configuration.
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
            "round-1 encoder requires 16-pixel-aligned dimensions",
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
                let mb_addr = mb_y * width_mbs + mb_x;
                // ------- Luma (16x16) DC prediction + residual -------
                // §8.3.3.3 — DC = (sum_top + sum_left + 16) >> 5 when
                // both are available; the half-sum forms when only one
                // side is present; default 128 when neither is.
                let dc_pred = predict_dc_luma_16x16(
                    &recon_y,
                    self.cfg.width as usize,
                    self.cfg.height as usize,
                    mb_x as usize,
                    mb_y as usize,
                );
                // Sample residual (16x16).
                let mut residual = [0i32; 256];
                for j in 0..16usize {
                    for i in 0..16usize {
                        let px = (mb_x as usize) * 16 + i;
                        let py = (mb_y as usize) * 16 + j;
                        let s = frame.y[py * (self.cfg.width as usize) + px] as i32;
                        residual[j * 16 + i] = s - dc_pred;
                    }
                }
                // For each 4x4 sub-block: forward 4x4. Then collect DCs
                // and run the Hadamard.
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

                // For round 1 cbp_luma = 0 — we discard AC. So inverse
                // path uses only the DC reconstruction:
                let inv_dc = inverse_hadamard_luma_dc_16x16(&dc_levels, qp_y, &FLAT_4X4_16, 8)
                    .expect("luma DC inverse");

                // Reconstruct each 4x4 sub-block as DC-only (zero AC):
                // build a coefficient block with only c[0,0] = inv_dc[blk]
                // and run the inverse transform with `dc_preserved`.
                for blk in 0..16 {
                    let bx = blk % 4;
                    let by = blk / 4;
                    let mut coeffs_in = [0i32; 16];
                    coeffs_in[0] = inv_dc[by * 4 + bx];
                    let r = inverse_transform_4x4(
                        // dc_preserved variant expects c[0,0] to be the
                        // already-scaled DC; that matches `inv_dc` per
                        // §8.5.10.
                        &coeffs_in,
                        qp_y,
                        &FLAT_4X4_16,
                        8,
                    );
                    // Use the DC-preserved variant so c[0,0] passes
                    // through unscaled and the rest of the AC stays 0.
                    let _ = r; // computed below via dc_preserved
                    let r = crate::transform::inverse_transform_4x4_dc_preserved(
                        &coeffs_in,
                        qp_y,
                        &FLAT_4X4_16,
                        8,
                    )
                    .expect("inverse 4x4");
                    for j in 0..4 {
                        for i in 0..4 {
                            let px = (mb_x as usize) * 16 + bx * 4 + i;
                            let py = (mb_y as usize) * 16 + by * 4 + j;
                            let raw = dc_pred + r[j * 4 + i];
                            recon_y[py * (self.cfg.width as usize) + px] = raw.clamp(0, 255) as u8;
                        }
                    }
                }

                // ------- Emit MB syntax -------
                // §9.2.1.1 NOTE 1 — `nC` for Intra16x16DCLevel uses the
                // *AC* TotalCoeff of neighbour luma4x4 block 0, not the
                // DC. For round-1 cbp_luma is always 0 → no AC ever
                // contributes → `nC` is identically 0. (The DC counts
                // do *not* feed back into the AC nC pool.)
                let nc_dc = 0;
                if std::env::var_os("OXIDEAV_H264_ENC_DEBUG").is_some() {
                    eprintln!(
                        "[ENC MB {:>3}] dc_pred={} dc_levels={:?} nc={}",
                        mb_addr, dc_pred, dc_levels, nc_dc
                    );
                }
                // total_coeff for the DC block we just emitted, for
                // future MBs' nC derivation.
                let mb_cfg = I16x16McbConfig {
                    pred_mode: 2,              // DC
                    intra_chroma_pred_mode: 0, // DC
                    cbp_luma: 0,
                    cbp_chroma: 0,
                    mb_qp_delta: 0,
                    luma_dc_levels_raster: dc_levels,
                };
                write_intra16x16_mb(&mut sw, &mb_cfg, CoeffTokenContext::Numeric(nc_dc))
                    .expect("write Intra16x16 mb");

                // ------- Chroma: DC prediction + reconstruction -------
                // §8.3.4.1 — chroma DC. With cbp_chroma=0 no residual
                // coefficients are sent, so the recon is exactly the
                // predictor sample.
                for plane in 0..2 {
                    let (src_plane, recon_plane): (&[u8], &mut [u8]) = if plane == 0 {
                        (frame.u, &mut recon_u)
                    } else {
                        (frame.v, &mut recon_v)
                    };
                    let _ = src_plane;
                    let pred = predict_dc_chroma_8x8(
                        recon_plane,
                        chroma_width,
                        chroma_height,
                        mb_x as usize,
                        mb_y as usize,
                    );
                    // Fill the 8x8 block with the predictor (one DC
                    // per 4x4 sub-block per §8.3.4.1; round-1 just uses
                    // a single MB-wide DC because we're not signalling
                    // chroma residual either way).
                    for j in 0..8 {
                        for i in 0..8 {
                            let cx = (mb_x as usize) * 8 + i;
                            let cy = (mb_y as usize) * 8 + j;
                            recon_plane[cy * chroma_width + cx] = pred.clamp(0, 255) as u8;
                        }
                    }
                }

                // Suppress unused-import warnings until the chroma
                // residual path is wired in (round 2). We import these
                // symbols here so the encoder's module surface mirrors
                // the decoder's, even though round-1 only writes the
                // luma DC residual.
                let _ = (
                    qp_c,
                    quantize_4x4,
                    quantize_chroma_dc,
                    forward_hadamard_2x2,
                    inverse_hadamard_chroma_dc_420,
                    intra16x16_mb_type_value,
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

// ---------------------------------------------------------------------------
// Local DC prediction helpers (mirror §8.3.3.3 / §8.3.4.1 exactly).
// ---------------------------------------------------------------------------

/// §8.3.3.3 — Intra_16x16_DC predictor, 8-bit luma, computed against the
/// reconstruction buffer in `recon_y`. Returns the single DC value used
/// for every sample in the 16x16 block.
fn predict_dc_luma_16x16(
    recon_y: &[u8],
    width: usize,
    _height: usize,
    mb_x: usize,
    mb_y: usize,
) -> i32 {
    let top_avail = mb_y > 0;
    let left_avail = mb_x > 0;
    if top_avail && left_avail {
        let mut s_top = 0i32;
        let mut s_left = 0i32;
        let row_above = (mb_y * 16 - 1) * width;
        for i in 0..16 {
            s_top += recon_y[row_above + mb_x * 16 + i] as i32;
        }
        for j in 0..16 {
            s_left += recon_y[(mb_y * 16 + j) * width + mb_x * 16 - 1] as i32;
        }
        (s_top + s_left + 16) >> 5
    } else if left_avail {
        let mut s_left = 0i32;
        for j in 0..16 {
            s_left += recon_y[(mb_y * 16 + j) * width + mb_x * 16 - 1] as i32;
        }
        (s_left + 8) >> 4
    } else if top_avail {
        let mut s_top = 0i32;
        let row_above = (mb_y * 16 - 1) * width;
        for i in 0..16 {
            s_top += recon_y[row_above + mb_x * 16 + i] as i32;
        }
        (s_top + 8) >> 4
    } else {
        128
    }
}

/// §8.3.4.1 — Intra_Chroma_DC predictor (4:2:0, 8x8). Round-1
/// simplification: returns the single 8x8-wide DC instead of one per
/// 4x4 sub-block. The decoder's per-sub-block derivation reduces to
/// the same value when neighbour samples are constant, which is the
/// case we hit here because each previous MB filled its chroma 8x8
/// with a single DC value (no residual transmitted).
fn predict_dc_chroma_8x8(
    recon: &[u8],
    width: usize,
    _height: usize,
    mb_x: usize,
    mb_y: usize,
) -> i32 {
    let top_avail = mb_y > 0;
    let left_avail = mb_x > 0;
    if top_avail && left_avail {
        let mut s_top = 0i32;
        let mut s_left = 0i32;
        let row_above = (mb_y * 8 - 1) * width;
        for i in 0..8 {
            s_top += recon[row_above + mb_x * 8 + i] as i32;
        }
        for j in 0..8 {
            s_left += recon[(mb_y * 8 + j) * width + mb_x * 8 - 1] as i32;
        }
        (s_top + s_left + 8) >> 4
    } else if left_avail {
        let mut s = 0i32;
        for j in 0..8 {
            s += recon[(mb_y * 8 + j) * width + mb_x * 8 - 1] as i32;
        }
        (s + 4) >> 3
    } else if top_avail {
        let mut s = 0i32;
        let row_above = (mb_y * 8 - 1) * width;
        for i in 0..8 {
            s += recon[row_above + mb_x * 8 + i] as i32;
        }
        (s + 4) >> 3
    } else {
        128
    }
}
