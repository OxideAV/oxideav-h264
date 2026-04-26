//! End-to-end decode smoke test through the `oxideav_core::Decoder`
//! trait, using `foreman_p16x16.264` — the same Annex B sample used by
//! `integration_foreman.rs`, but exercising the public trait surface
//! instead of the internal Picture pipeline.
//!
//! Verifies that feeding the raw Annex B bytes through
//! `H264CodecDecoder::send_packet` produces a `Frame::Video` with the
//! expected geometry and non-empty pixel data for the first IDR.
//!
//! Like its sibling, this test looks for the clip at the path in
//! `OXIDEAV_SAMPLES_H264_FOREMAN` (or a hardcoded author default) and
//! skips if the sample is missing.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
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
    if default.exists() {
        Some(default)
    } else {
        None
    }
}

#[test]
fn decode_foreman_p16x16_first_idr_through_trait() {
    let Some(path) = sample_path() else {
        eprintln!("skip: set OXIDEAV_SAMPLES_H264_FOREMAN or place the file at the default path");
        return;
    };

    let bytes = std::fs::read(&path).expect("read test clip");
    assert!(!bytes.is_empty(), "empty test clip");

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));

    // Annex B, no extradata; single packet carries the whole clip.
    let packet = Packet::new(0, TimeBase::new(1, 25), bytes.clone()).with_pts(0);
    dec.send_packet(&packet).expect("send_packet");

    // Signal end-of-stream so the §C.4 bumping process drains the
    // output DPB. Before `dpb_output` wiring went in, decode-order
    // output meant `receive_frame` produced the IDR immediately; with
    // POC-ordered output the queue holds up to `max_num_reorder_frames`
    // pictures and the trailing IDR is only released once we flush.
    dec.flush().expect("flush");

    // First IDR (lowest POC) should have been reconstructed and queued.
    let frame = dec.receive_frame().expect("receive_frame");
    let vf = match frame {
        Frame::Video(vf) => vf,
        other => panic!("expected Frame::Video, got {:?}", other),
    };

    // foreman_p16x16.264 is 176x144 (QCIF) YUV 4:2:0 8-bit. The slim
    // VideoFrame shape no longer carries width/height/format directly —
    // those live on the stream's CodecParameters. Geometry is asserted
    // through the plane layout instead (stride == width, data ==
    // stride * height), and the 3-plane / chroma-half pattern stands in
    // for the Yuv420P format check.
    assert_eq!(vf.planes.len(), 3, "expected 3-plane (Yuv420P) layout");

    // Luma plane stride matches width, and data size matches stride*height.
    assert_eq!(vf.planes[0].stride, 176);
    assert_eq!(vf.planes[0].data.len(), 176 * 144);

    // Chroma planes are half-resolution for 4:2:0.
    assert_eq!(vf.planes[1].stride, 88);
    assert_eq!(vf.planes[1].data.len(), 88 * 72);
    assert_eq!(vf.planes[2].stride, 88);
    assert_eq!(vf.planes[2].data.len(), 88 * 72);

    // Top-left luma sample: liveness check that reconstruction wrote
    // something. The exact value is encoder-defined; we just require
    // that not all samples are zero (a black or unwritten buffer would
    // be a red flag for foreman, which opens on the speaker's face).
    let has_nonzero = vf.planes[0].data.iter().any(|&b| b != 0);
    assert!(
        has_nonzero,
        "luma plane was all zero — reconstruction likely did not run"
    );

    // pts carried through from the packet.
    assert_eq!(vf.pts, Some(0));
    eprintln!(
        "first IDR: luma={}x{} top-left-luma={} pts={:?}",
        vf.planes[0].stride,
        vf.planes[0].data.len() / vf.planes[0].stride,
        vf.planes[0].data[0],
        vf.pts
    );
}
