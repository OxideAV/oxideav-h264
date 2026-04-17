//! Pure-Rust H.264 / AVC (ITU-T H.264 | ISO/IEC 14496-10) decoder.
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
//!   The CABAC I-path covers I_16×16 and I_NxN; I_PCM inside CABAC is not
//!   yet wired.
//! * **CAVLC baseline P-slice** pixel reconstruction (§8.4) — P_L0_16×16 /
//!   16×8 / 8×16, P_8×8 (all four sub-partition shapes) and P_8×8ref0,
//!   P_Skip, plus intra-in-P macroblocks. Luma 6-tap half-pel + bilinear
//!   quarter-pel MC; bilinear 1/8-pel chroma MC; edge-replicated
//!   out-of-picture reads. A single-slot L0 reference (the previous frame
//!   output) is retained across packets.
//!
//! # Out of scope (returns `Error::Unsupported`)
//!
//! * CABAC P-slices.
//! * B-slices, weighted prediction, bi-prediction.
//! * Multi-reference DPB (`ref_idx_l0 > 0`), reference picture list
//!   modification operations.
//! * Interlaced coding / MBAFF / PAFF.
//! * 8×8 transform (§8.5.13), 4:2:2 / 4:4:4 chroma, bit depths above 8.
//! * Encoder.
//!
//! This crate has no runtime dependencies beyond `oxideav-core` and
//! `oxideav-codec`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::derivable_impls)]

pub mod bitreader;
pub mod cabac;
pub mod cavlc;
pub mod deblock;
pub mod decoder;
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
use oxideav_core::{CodecCapabilities, CodecId};

/// The canonical oxideav codec id for H.264 / AVC video.
pub const CODEC_ID_STR: &str = "h264";

/// Register this decoder with a codec registry.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("h264_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(8192, 8192);
    reg.register_decoder_impl(CodecId::new(CODEC_ID_STR), caps, decoder::make_decoder);
}
