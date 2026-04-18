//! Pure-Rust H.264 / AVC (ITU-T H.264 | ISO/IEC 14496-10) decoder + Baseline
//! I-frame-only encoder.
//!
//! # What is decoded today
//!
//! * NAL unit framing in **both** Annex B (`00 00 [00] 01` start codes) and
//!   AVCC length-prefixed form (used inside MP4 `mdat`), emulation-prevention
//!   byte stripping (§7.4.1.1), and `AVCDecoderConfigurationRecord` parsing
//!   (ISO/IEC 14496-15 §5.2.4.1).
//! * SPS / PPS / slice-header parsing (§7.3.2.1.1, §7.3.2.2, §7.3.3).
//! * **I-slice** pixel reconstruction for **both CAVLC and CABAC** — intra
//!   prediction, residual decode, 4×4 IDCT + Hadamard, optional deblocking.
//!   The CABAC I-path covers I_16×16, I_NxN and I_PCM (with the mid-stream
//!   arithmetic-engine re-init mandated by §9.3.1.2).
//! * **CAVLC baseline P-slice** pixel reconstruction (§8.4) — P_L0_16×16 /
//!   16×8 / 8×16, P_8×8 (all four sub-partition shapes) and P_8×8ref0,
//!   P_Skip, plus intra-in-P macroblocks. Luma 6-tap half-pel + bilinear
//!   quarter-pel MC; bilinear 1/8-pel chroma MC; edge-replicated
//!   out-of-picture reads. A single-slot L0 reference (the previous frame
//!   output) is retained across packets.
//! * **Explicit weighted P-slice prediction** (§7.3.3.2 / §8.4.2.3.2) —
//!   the slice's `pred_weight_table` is parsed when the PPS has
//!   `weighted_pred_flag = 1`, and the per-reference (weight, offset,
//!   log2_denom) triples are applied to the MC output before the residual
//!   is added. List-0 only (the L1 side of the weight table is parsed for
//!   bitstream conformance but unused; B-slice bi-prediction remains out
//!   of scope).
//!
//! # What is encoded today
//!
//! See [`encoder::H264Encoder`] — Baseline Profile, I-frames only,
//! Intra_16×16 DC_PRED, CAVLC, single slice per frame, 4:2:0, 8-bit,
//! fixed QP, deblocking disabled. The encoder and decoder round-trip
//! to PSNR ≥ 28 dB for gradient content at QP 22 and ≥ 40 dB for solid
//! content.
//!
//! # High Profile 8×8 transform — primitives only, not wired
//!
//! The math primitives for the High-Profile 8×8 residual path are
//! implemented and unit-tested in isolation but are **not** wired into the
//! macroblock decode loop:
//!
//! * 8×8 inverse integer transform ([`transform::idct_8x8`]) and its
//!   6-class dequantisation ([`transform::dequantize_8x8`]) — §8.5.13.
//! * Nine Intra_8×8 prediction modes ([`intra_pred::predict_intra_8x8`])
//!   including reference-sample low-pass filtering — §8.3.2.
//!
//! When `pps.transform_8x8_mode_flag = 1` and a macroblock sets
//! `transform_size_8x8_flag = 1`, both the CAVLC I-slice path
//! ([`mb::decode_i_slice_data`]) and the CABAC I-slice path
//! ([`cabac::mb`]) surface `Error::Unsupported`. The CAVLC §7.3.5.3.2
//! residual decode and the CABAC 8×8 context coding (§9.3.3.1.1.10) are
//! not implemented — a prior attempt hit a bit-accounting desync against
//! real x264 output and was rolled back so the public surface stays
//! honest about what's wired.
//!
//! # Out of scope (returns `Error::Unsupported` or the encoder refuses)
//!
//! * High-Profile 8×8 residual decode (both CAVLC §7.3.5.3.2 and CABAC
//!   §9.3.3.1.1.10), and P-slice inter macroblocks with
//!   `transform_size_8x8_flag = 1`.
//! * Custom scaling-list matrices — the SPS/PPS fields are parsed and
//!   skipped, and only the flat-16 default list is used for 4×4 dequant.
//! * CABAC P-slices (decode); any CABAC encoding; any P/B slice encoding.
//! * B-slices (bi-prediction). Implicit weighted bi-prediction
//!   (`weighted_bipred_idc == 2`, §8.4.2.3.3) is not implemented either —
//!   only the explicit P-slice form (§8.4.2.3.2) is supported today.
//! * Multi-reference DPB (`ref_idx_l0 > 0`), reference picture list
//!   modification operations.
//! * Interlaced coding / MBAFF / PAFF.
//! * 4:2:2 / 4:4:4 chroma, bit depths above 8.
//! * Rate control, adaptive QP, mode decision (Intra4×4 / Plane / Vertical /
//!   Horizontal), or any psychovisual tuning. The encoder always emits
//!   Intra_16×16 with DC_PRED and a fixed QP.
//!
//! This crate has no runtime dependencies beyond `oxideav-core` and
//! `oxideav-codec`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::derivable_impls)]

pub mod bitreader;
pub mod bitwriter;
pub mod cabac;
pub mod cavlc;
pub mod cavlc_enc;
pub mod deblock;
pub mod decoder;
pub mod encoder;
pub mod fwd_transform;
pub mod intra_pred;
pub mod mb;
pub mod mb_type;
pub mod motion;
pub mod nal;
pub mod p_mb;
pub mod picture;
pub mod pps;
pub mod slice;
pub mod sps;
pub mod tables;
pub mod transform;

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecCapabilities, CodecId, PixelFormat};

/// The canonical oxideav codec id for H.264 / AVC video.
pub const CODEC_ID_STR: &str = "h264";

/// Register this crate's decoder and baseline I-only encoder with a codec
/// registry. Both are registered under the same `"h264"` id; the registry's
/// preference + priority logic picks one at resolve time.
pub fn register(reg: &mut CodecRegistry) {
    let dec_caps = CodecCapabilities::video("h264_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(8192, 8192);
    reg.register_decoder_impl(CodecId::new(CODEC_ID_STR), dec_caps, decoder::make_decoder);

    // Encoder: Baseline I-only. The registered caps advertise
    // `intra_only = true` and cap the output at level 3.0 maxima.
    let enc_caps = CodecCapabilities::video("h264_baseline_i_sw")
        .with_lossy(true)
        .with_intra_only(true)
        .with_max_size(720, 576)
        .with_pixel_format(PixelFormat::Yuv420P);
    reg.register_encoder_impl(CodecId::new(CODEC_ID_STR), enc_caps, encoder::make_encoder);
}
