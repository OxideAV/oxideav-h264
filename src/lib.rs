//! Pure-Rust **H.264 / AVC** codec for [`oxideav`].
//!
//! **This crate is currently empty.** A previous implementation lives on
//! the [`old`](https://github.com/OxideAV/oxideav-h264/tree/old) branch
//! but is being rewritten from scratch with the ITU-T Rec. H.264 |
//! ISO/IEC 14496-10 specification (2024-08 edition) as the single
//! authoritative source. No external decoder code is consulted while
//! writing this implementation.
//!
//! The crate exposes a registration entry point so the workspace
//! aggregator keeps building, but no decoder or encoder is registered
//! yet — every packet routed here errors out with `Unsupported`.
//!
//! See `README.md` for the spec coverage matrix as it grows.

pub(crate) mod bitstream;
pub(crate) mod cabac;
pub(crate) mod cabac_ctx;
pub(crate) mod cavlc;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod deblock;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod decoder;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod dpb_output;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod inter_pred;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod intra_pred;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod macroblock_layer;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod mb_address;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod mb_grid;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod mv_deriv;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod nal;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod non_vcl;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod picture;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod poc;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod pps;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod reconstruct;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod ref_list;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod ref_store;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod scaling_list;
pub mod sei;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod simd;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod slice_data;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod slice_header;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod sps;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod sps_extension;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod subset_sps;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod transform;
#[doc(hidden)] // internal — exposed for tests/fuzz; not part of the stable API
pub mod vui;

pub mod h264_decoder;

pub mod encoder;

use oxideav_core::{CodecCapabilities, CodecId, CodecTag};
use oxideav_core::{CodecInfo, CodecRegistry, RuntimeContext};

/// Codec id constant — matches the historical `"h264"` id used by
/// containers (MKV `V_MPEG4/ISO/AVC`, MP4 `avc1`, AVI `H264`/`X264`).
pub const CODEC_ID_STR: &str = "h264";

/// Register the H.264 decoder with a codec registry.
///
/// Claims the historical FourCCs used by MKV (`V_MPEG4/ISO/AVC` maps
/// to `AVC1`), MP4 (`avc1`, `avc3`), and AVI (`H264`, `X264`, `h264`).
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("h264_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(8192, 8192);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(h264_decoder::make_decoder)
            .tags([
                CodecTag::fourcc(b"H264"),
                CodecTag::fourcc(b"h264"),
                CodecTag::fourcc(b"AVC1"),
                CodecTag::fourcc(b"avc1"),
                CodecTag::fourcc(b"avc3"),
                CodecTag::fourcc(b"X264"),
                CodecTag::fourcc(b"x264"),
            ]),
    );
}

/// Unified registration entry point: install the H.264 codec factories
/// into the codec sub-registry of a [`RuntimeContext`].
///
/// This is the preferred entry point for new code — it matches the
/// convention every sibling crate now follows. Direct callers that need
/// only the codec sub-registry can keep using [`register_codecs`].
///
/// Also wired into [`oxideav_meta::register_all`] via the
/// [`oxideav_core::register!`] macro below.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
}

oxideav_core::register!("h264", register);

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{CodecId, CodecParameters, RuntimeContext};

    #[test]
    fn register_via_runtime_context_installs_codec_factory() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let dec = ctx
            .codecs
            .first_decoder(&params)
            .expect("h264 decoder factory");
        assert_eq!(dec.codec_id().as_str(), CODEC_ID_STR);
    }
}
