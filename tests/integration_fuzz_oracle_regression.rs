//! Regression tests for divergences uncovered by the
//! `ffmpeg_oracle_decode` cross-decode fuzz target. Each test below
//! pins down a specific input shape that previously made our decoder
//! emit a `Frame::Video` while libavcodec rejected the whole access
//! unit — i.e. an over-permissive accept that the fuzz oracle flagged
//! as a strictness divergence.
//!
//! The tests do NOT depend on libavcodec being installed: they
//! exercise our decoder alone and assert it does the right thing for
//! the input the oracle reported. The cross-decode oracle itself
//! lives in `fuzz/fuzz_targets/ffmpeg_oracle_decode.rs`.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;

/// Crash `2ad9589faea57c09b55364bf7ac4af0373e66c18` (175 bytes).
///
/// The input carries a PPS, an SPS, then three non-IDR slices whose
/// CABAC bitstreams all run past EOF. libavcodec rejects each slice
/// with "CABAC decoding failed: read past end of bitstream", produces
/// no frame, and treats the access unit as gone.
///
/// Before the fix, our decoder seeded a `PictureInProgress` on the
/// first slice's parse-OK header, then `eprintln!`'d the per-slice
/// reconstruction failures and silently finalized the never-painted
/// picture into the DPB. `receive_frame` then produced a
/// `Frame::Video` with all-zero luma + stale chroma — a strictness
/// divergence from libavcodec.
///
/// Spec citation: §7.4.1.2.4 ("primary coded picture" / decoded
/// reference picture marking) governs which pictures enter the DPB.
/// A picture whose every slice failed §7.3.4 / §9 entropy decode is
/// not a successfully decoded picture and must not be added.
///
/// The fix tracks `any_slice_succeeded` per `PictureInProgress` and
/// drops the picture in `finalize_in_progress_picture` when the flag
/// is still false at finalize time.
#[test]
fn fuzz_crash_2ad9589f_all_slices_fail_no_frame_emitted() {
    // The exact 175-byte payload from the libfuzzer artifact, embedded
    // verbatim so the test is self-contained and runs in CI without
    // needing the original `crash-…` file from the artifact upload.
    const CRASH_BYTES: &[u8] = &[
        0x00, 0x00, 0x01, 0x08, 0xf9, 0x1a, 0x18, 0x08, 0x26, 0x08, 0x00, 0x08, 0x5f, 0x61, 0x6c,
        0x28, 0xff, 0xa6, 0xa6, 0xa6, 0x01, 0x20, 0x00, 0x00, 0x00, 0x00, 0xd3, 0xd3, 0xd3, 0xd3,
        0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0xd3, 0x00,
        0x00, 0x00, 0x01, 0x07, 0x53, 0x78, 0x35, 0xa6, 0xa6, 0xa6, 0x9e, 0xa6, 0xa6, 0x1a, 0x18,
        0x08, 0x26, 0x08, 0xa6, 0x2a, 0xac, 0xac, 0x7e, 0xac, 0xaa, 0xaa, 0xaa, 0x73, 0x65, 0x6c,
        0x69, 0x6e, 0x75, 0x78, 0x66, 0x73, 0x00, 0x00, 0x00, 0x01, 0x01, 0xb8, 0xb8, 0xb8, 0x25,
        0x00, 0xf6, 0xfb, 0x47, 0x08, 0xf9, 0x01, 0xb8, 0xb8, 0xb8, 0x25, 0x00, 0xf6, 0x00, 0x48,
        0xf6, 0xa6, 0x2a, 0xac, 0xac, 0x7e, 0xac, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xff, 0x18, 0x00,
        0x08, 0x0e, 0x00, 0x00, 0x00, 0x01, 0x01, 0x08, 0xb8, 0x01, 0xf9, 0xb8, 0xb8, 0x00, 0x00,
        0x00, 0x01, 0x01, 0xbd, 0x08, 0xf9, 0x01, 0xb8, 0xb8, 0xb8, 0x00, 0xdf, 0x00, 0x48, 0xf6,
        0x00, 0xf6, 0x56, 0x00, 0x00, 0xfa, 0xf6, 0xf9, 0x01, 0x0a, 0x00, 0x36, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x03, 0x66, 0x73, 0xf6, 0xf6, 0xf9, 0x01, 0xb8,
    ];

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), CRASH_BYTES.to_vec());

    // `send_packet` returns Ok because parameter-set NALs parse fine
    // and per-slice errors are caught + downgraded to log-only events
    // (see `H264CodecDecoder::send_packet`'s error-tolerant inner
    // loop). The test cares about what comes out of `receive_frame`,
    // not whether `send_packet` returned Ok.
    let _ = dec.send_packet(&pkt);
    dec.flush().expect("flush");

    // No frame should be emitted: every slice failed CABAC parse, so
    // there is no successfully-decoded primary coded picture to push
    // into the DPB. `receive_frame` should report Eof (queue empty +
    // EOF after flush).
    match dec.receive_frame() {
        Err(oxideav_core::Error::Eof) => {
            // Desired behaviour — libavcodec parity.
        }
        Err(e) => {
            // Any other error is also acceptable (no frame emitted),
            // but Eof is the precise expected shape.
            panic!("expected Error::Eof after all slices failed, got Error: {e:?}");
        }
        Ok(Frame::Video(_)) => {
            panic!(
                "REGRESSION: decoder emitted a Frame::Video for an input \
                 whose every slice failed CABAC parse — this is the \
                 over-permissive-accept the fuzz oracle flagged on \
                 crash-2ad9589f… (libavcodec rejects + emits no frame)."
            );
        }
        Ok(other) => {
            panic!("expected Eof, got {other:?}");
        }
    }
}
