//! Pure-Rust **H.264 / AVC** codec for [`oxideav`].
//!
//! **This crate is currently empty.** A previous implementation lives on
//! the [`old`](https://github.com/OxideAV/oxideav-h264/tree/old) branch
//! but is being rewritten from scratch with the ITU-T Rec. H.264 |
//! ISO/IEC 14496-10 specification (2024-08 edition) as the single
//! authoritative source. No external decoder code (libavcodec / openh264
//! / JM / etc.) is consulted while writing this implementation.
//!
//! The crate exposes a registration entry point so the workspace
//! aggregator keeps building, but no decoder or encoder is registered
//! yet — every packet routed here errors out with `Unsupported`.
//!
//! See `README.md` for the spec coverage matrix as it grows.

pub(crate) mod bitstream;
pub mod nal;
pub mod sps;
pub mod pps;
pub mod scaling_list;
pub mod vui;
pub mod slice_header;
pub mod non_vcl;
pub mod poc;
pub(crate) mod cavlc;
pub(crate) mod cabac;
pub mod intra_pred;
pub mod inter_pred;
pub mod mv_deriv;
pub mod transform;
pub mod deblock;
pub mod ref_list;
pub(crate) mod cabac_ctx;
pub mod sei;
pub mod mb_address;
pub mod macroblock_layer;
pub mod slice_data;
pub mod picture;
pub mod mb_grid;
pub mod ref_store;
pub mod reconstruct;
pub mod dpb_output;
pub mod decoder;

use oxideav_codec::CodecRegistry;

/// Codec id constant — matches the historical `"h264"` id used by
/// containers (MKV `V_MPEG4/ISO/AVC`, MP4 `avc1`, AVI `H264`/`X264`).
pub const CODEC_ID_STR: &str = "h264";

/// Currently a no-op so the workspace aggregator can keep wiring this
/// crate in without breaking. Will register a real decoder + encoder
/// once the spec-driven rebuild has both ends working.
pub fn register(_codecs: &mut CodecRegistry) {}
