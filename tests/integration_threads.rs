//! A decoder granted a worker budget through
//! `Decoder::set_execution_context` must emit exactly the frames the
//! single-threaded decoder emits, for every budget.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, ExecutionContext, Frame, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;

fn decode_all(bytes: &[u8], threads: usize) -> Vec<Vec<u8>> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    dec.set_execution_context(&ExecutionContext::with_threads(threads));
    let packet = Packet::new(0, TimeBase::new(1, 25), bytes.to_vec()).with_pts(0);
    dec.send_packet(&packet).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    while let Ok(Frame::Video(v)) = dec.receive_frame() {
        let mut out = Vec::new();
        for p in v.image_planes() {
            out.extend_from_slice(&p.data);
        }
        frames.push(out);
    }
    frames
}

fn assert_thread_invariant(name: &str) {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).expect("read fixture");
    let serial = decode_all(&bytes, 1);
    assert!(!serial.is_empty(), "{name}: no frames decoded");
    for threads in [2, 3, 4, 8] {
        let threaded = decode_all(&bytes, threads);
        assert_eq!(
            serial.len(),
            threaded.len(),
            "{name}: frame count at {threads} threads"
        );
        for (i, (a, b)) in serial.iter().zip(&threaded).enumerate() {
            assert!(a == b, "{name}: frame {i} differs at {threads} threads");
        }
    }
}

#[test]
fn x264_p_b_stream_is_thread_invariant() {
    assert_thread_invariant("x264_testsrc_128x96_25f.h264");
}

#[test]
fn mbaff_stream_is_thread_invariant() {
    assert_thread_invariant("mbaff_iframe_128x96.h264");
}
