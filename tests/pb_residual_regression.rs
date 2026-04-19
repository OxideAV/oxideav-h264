//! Regression for the P/B residual bit-exactness ceiling.
//!
//! `testsrc_p_64x64.{es,yuv}` — a 3-frame 64×64 CAVLC Main-profile P clip
//! generated from `testsrc` with no B-frames (`-bf 0`) and no scenecut so
//! frames 1-2 are pure P. Every MB on row 3 decodes through
//! [`decode_p16x16`] with `cbp_luma = 15`, and two MBs on row 2 are
//! Intra_16×16 with `cbp_luma = 0`.
//!
//! Root cause of the original divergence (§8.4.1.3.1 median MV
//! prediction): an intra-coded neighbour is "available" with
//! `refIdxLX = -1` and `mvLX = 0`, NOT "not available". The earlier
//! implementation short-circuited `neighbour_mv_list` to `None` for
//! intra neighbours, which in turn triggered the §6.4.11.7 C→D
//! geometric-fallback incorrectly. The corrected predictor matches
//! FFmpeg's `pred_motion` match_count path and yields bit-exact
//! reconstruction on this fixture.
//!
//! Generated with:
//!   ffmpeg -y -f lavfi -i testsrc=duration=1:size=64x64:rate=3 -vframes 3 \
//!     -pix_fmt yuv420p -c:v libx264 -profile:v main -preset ultrafast \
//!     -coder 0 -bf 0 -x264opts no-scenecut:keyint=3:min-keyint=3 -qp 10 \
//!     /tmp/testsrc_p.mp4

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, PixelFormat, TimeBase};
use oxideav_h264::decoder::H264Decoder;

fn read_fixture(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn single_packet(es: Vec<u8>) -> Packet {
    Packet::new(0, TimeBase::new(1, 90_000), es)
        .with_pts(0)
        .with_keyframe(true)
}

fn collect_frames(dec: &mut H264Decoder) -> Vec<oxideav_core::VideoFrame> {
    let mut out = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => out.push(v),
            _ => break,
        }
    }
    dec.flush().unwrap();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => out.push(v),
            _ => break,
        }
    }
    out
}

fn flatten_frames_yuv420(
    frames: &[oxideav_core::VideoFrame],
    n: usize,
    w: usize,
    h: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * w * h * 3 / 2);
    for frame in frames.iter().take(n) {
        assert_eq!(frame.format, PixelFormat::Yuv420P);
        let (pw, ph) = [(w, h), (w / 2, h / 2), (w / 2, h / 2)]
            .iter()
            .copied()
            .next()
            .unwrap();
        let planes_dims = [(w, h), (w / 2, h / 2), (w / 2, h / 2)];
        let _ = (pw, ph);
        for (i, p) in frame.planes.iter().enumerate() {
            let (pw, ph) = planes_dims[i];
            for row in 0..ph {
                out.extend_from_slice(&p.data[row * p.stride..row * p.stride + pw]);
            }
        }
    }
    out
}

fn ratio_match(a: &[u8], b: &[u8]) -> f64 {
    let matched = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    matched as f64 / a.len() as f64
}

/// The 4:2:0 testsrc P-slice fixture must decode bit-exact — the
/// §8.4.1.3.1 median MV predictor now treats intra neighbours as
/// `(mvLX = 0, refIdxLX = -1)` (available, non-matching) instead of
/// `None` (unavailable), so the earlier 95.7% ceiling caused by the
/// incorrect C→D geometric-fallback on row 3 is gone.
#[test]
fn testsrc_p_decode_above_current_floor() {
    let es = read_fixture("tests/fixtures/testsrc_p_64x64.es");
    let ref_yuv = read_fixture("tests/fixtures/testsrc_p_64x64.yuv");
    let mut dec = H264Decoder::new(CodecId::new("h264"));
    dec.send_packet(&single_packet(es))
        .expect("decode succeeds");
    let frames = collect_frames(&mut dec);
    assert!(
        frames.len() >= 3,
        "expected >=3 frames, got {}",
        frames.len()
    );
    let got = flatten_frames_yuv420(&frames, 3, 64, 64);
    assert_eq!(got.len(), ref_yuv.len(), "decoded length mismatch");
    let r = ratio_match(&got, &ref_yuv);
    assert!(
        (r - 1.0).abs() < f64::EPSILON,
        "4:2:0 testsrc P-slice decode accuracy {:.3}% — expected 100%",
        r * 100.0
    );
}
