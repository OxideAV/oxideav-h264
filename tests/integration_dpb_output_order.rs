//! Integration test for §C.2.2 / §C.4 output ordering through
//! [`H264CodecDecoder`].
//!
//! The full end-to-end path (real bitstream with B-frames) needs a
//! sample clip that reliably exercises reorder. Rather than depend on
//! a specific sample and the CABAC-residual reconstruction path (still
//! under repair), we drive the decoder's `receive_frame` /
//! `Decoder::flush` surface in the basic NeedMore / Eof contract.
//!
//! The detailed POC-ordering logic is exercised by the unit tests in
//! `h264_decoder::tests::ipbbb_reorder_delivers_ascending_poc` and
//! `flush_at_eof_drains_in_poc_order`, which have direct access to
//! the internal [`crate::dpb_output::DpbOutput`] so they can fabricate
//! decode-order sequences without a real slice stream.
//!
//! Spec references:
//!   * §C.2.2 — "Storage and output of decoded pictures"
//!   * §C.4   — "Bumping process" / output timing

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Error};
use oxideav_h264::h264_decoder::H264CodecDecoder;

/// §C.4 — a fresh decoder with no frames pushed yields `NeedMore`
/// on `receive_frame`, not `Eof`.
#[test]
fn fresh_decoder_reports_need_more() {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let res = dec.receive_frame();
    assert!(matches!(res, Err(Error::NeedMore)), "got {res:?}");
}

/// §C.4 — after `flush()` with no frames produced, `receive_frame`
/// reports `Eof` (not `NeedMore`), signalling end-of-stream.
#[test]
fn flushed_empty_decoder_reports_eof() {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    dec.flush().expect("flush");
    let res = dec.receive_frame();
    assert!(matches!(res, Err(Error::Eof)), "got {res:?}");
}

/// Post-`reset()` the decoder behaves as freshly constructed: no
/// pending frames, no sticky EOF.
#[test]
fn reset_clears_eof_flag() {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    dec.flush().expect("flush");
    assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    dec.reset().expect("reset");
    assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
}
