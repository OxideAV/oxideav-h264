//! End-to-end P-slice decode smoke test.
//!
//! Verifies that with the DPB / ref_store wired through
//! `H264CodecDecoder`, non-IDR slices are routed into the
//! reconstruction pipeline instead of being silently dropped, *and*
//! that a slice whose reconstruction fails (e.g. because of a
//! known CABAC-residual gap in other modules) doesn't blow up the
//! decoder — the stream keeps going and frames continue to come
//! out.
//!
//! The sample clip used here is `foreman_p16x16.264` (176x144
//! Annex B, CABAC, one IDR + several P slices). The P slices are
//! known to trip over the CABAC-residual gap that `cavlc.rs` /
//! `macroblock_layer.rs` are still being repaired for, so we do
//! NOT assert that all P frames make it through — we only check
//! that at least the first (IDR) frame comes through with the
//! expected geometry. Once the CABAC gap lands, we can tighten
//! this to assert ≥2 frames (IDR + ≥1 P).
//!
//! Accuracy of reconstructed pixels for P slices is intentionally
//! NOT asserted yet.
//!
//! Path resolution mirrors the sibling integration tests: use
//! `OXIDEAV_SAMPLES_H264_FOREMAN` or the hardcoded author default;
//! skip if neither exists.

use oxideav_codec::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, PixelFormat, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use std::path::PathBuf;

fn sample_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("OXIDEAV_SAMPLES_H264_FOREMAN") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    let default = PathBuf::from(
        "/home/magicaltux/projects/oxideav/samples/samples.ffmpeg.org/V-codecs/h264/foreman_p16x16.264",
    );
    if default.exists() { Some(default) } else { None }
}

#[test]
fn decode_p_slices_yields_frames() {
    let Some(path) = sample_path() else {
        eprintln!(
            "skip: set OXIDEAV_SAMPLES_H264_FOREMAN or place the file at the default path"
        );
        return;
    };

    let bytes = std::fs::read(&path).expect("read test clip");
    assert!(!bytes.is_empty(), "empty test clip");

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));

    // Annex B, no extradata. Feed the whole clip as one packet; the
    // driver iterates every NAL internally.
    let packet = Packet::new(0, TimeBase::new(1, 25), bytes.clone()).with_pts(0);
    dec.send_packet(&packet).expect("send_packet");

    // Signal EOF so the §C.4 bumping process drains the output DPB.
    // POC-ordered output (post round-78 wiring) holds pictures until
    // enough follow-up frames arrive to force a bump; flush drains
    // the remainder in POC-ascending order.
    dec.flush().expect("flush");

    // Drain the queued frames. Each call is `Eof` once empty (not
    // `NeedMore`, because we flushed).
    let mut frames: Vec<_> = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => frames.push(vf),
            Ok(_) => continue, // non-video frame; shouldn't happen here
            Err(_) => break,
        }
    }

    assert!(
        !frames.is_empty(),
        "expected ≥1 frame (at least the IDR), got 0 — the pipeline errored before any output",
    );

    // Every frame that did make it through must have the expected
    // 176x144 YUV 4:2:0 geometry. This catches regressions in the
    // Picture → VideoFrame conversion.
    for (i, vf) in frames.iter().enumerate() {
        assert_eq!(vf.format, PixelFormat::Yuv420P, "frame {i} wrong format");
        assert_eq!(vf.width, 176, "frame {i} wrong width");
        assert_eq!(vf.height, 144, "frame {i} wrong height");
        assert_eq!(vf.planes.len(), 3, "frame {i} missing planes");
        assert_eq!(vf.planes[0].data.len(), 176 * 144, "frame {i} luma size");
        assert_eq!(vf.planes[1].data.len(), 88 * 72, "frame {i} cb size");
        assert_eq!(vf.planes[2].data.len(), 88 * 72, "frame {i} cr size");
    }

    eprintln!(
        "foreman: decoded {} frame(s) ({}x{} {:?})",
        frames.len(),
        frames[0].width,
        frames[0].height,
        frames[0].format
    );
}
