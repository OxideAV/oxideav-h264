//! ffmpeg-gated conformance test for **MBAFF (interlaced) intra decode**.
//!
//! The fixture `tests/fixtures/mbaff_iframe_128x96.h264` is a synthetic,
//! clean-licensed Annex B clip: a single IDR picture of an
//! `ffmpeg -f lavfi testsrc2` source, encoded by libx264 with
//! macroblock-adaptive frame/field coding (MBAFF). The SPS carries
//! `frame_mbs_only_flag == 0` and `mb_adaptive_frame_field_flag == 1`, so
//! macroblock addresses are pair-interleaved per §6.4.10 and the
//! intra-prediction neighbour derivation must follow Table 6-4 rather
//! than the raster §6.4.9 path. (This is the bug fixed alongside this
//! test: routing `neighbour_4x4_addr`/`neighbour_8x8_addr` through the
//! MBAFF derivation lifted luma PSNR on this clip from ~9 dB to ~54 dB.)
//!
//! The test is **gated on the `ffmpeg` CLI** being on `PATH`: ffmpeg is
//! invoked as an opaque black box to produce the reference raw YUV
//! (we never read its source). When ffmpeg is absent the test skips.
//!
//! ## Encoder command used to author the fixture
//!
//! ```text
//! ffmpeg -y -f lavfi -i 'testsrc2=size=128x96:rate=50:duration=0.04' \
//!        -flags +ildct+ilme \
//!        -c:v libx264 -frames:v 1 -profile:v high -level 30 \
//!        -preset placebo -bf 0 -refs 1 -coder 1 \
//!        -x264-params keyint=1:scenecut=0:tff=1:interlaced=1 \
//!        -pix_fmt yuv420p tests/fixtures/mbaff_iframe_128x96.h264
//! ```

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use std::path::PathBuf;
use std::process::Command;

const WIDTH: usize = 128;
const HEIGHT: usize = 96;

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("mbaff_iframe_128x96.h264")
}

/// Decode the fixture's reference luma plane via ffmpeg (black box) into
/// a flat `yuv420p` byte buffer for the single coded frame.
fn ffmpeg_reference_yuv(h264: &PathBuf) -> Option<Vec<u8>> {
    let out = Command::new("ffmpeg")
        .args(["-y", "-i"])
        .arg(h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(out.stdout)
}

#[test]
fn mbaff_iframe_intra_decode_psnr() {
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not on PATH (black-box reference unavailable)");
        return;
    }
    let path = fixture_path();
    assert!(path.exists(), "fixture missing: {}", path.display());

    let reference = match ffmpeg_reference_yuv(&path) {
        Some(r) => r,
        None => {
            eprintln!("skip: ffmpeg failed to decode the fixture");
            return;
        }
    };
    let frame_sz = WIDTH * HEIGHT * 3 / 2;
    assert!(
        reference.len() >= frame_sz,
        "ffmpeg reference too small: {} < {}",
        reference.len(),
        frame_sz
    );

    let bytes = std::fs::read(&path).expect("read fixture");
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let packet = Packet::new(0, TimeBase::new(1, 50), bytes).with_pts(0);
    dec.send_packet(&packet).expect("send_packet");
    dec.flush().expect("flush");

    let frame = dec
        .receive_frame()
        .expect("MBAFF I-frame must reconstruct and be queued");
    let vf = match frame {
        Frame::Video(vf) => vf,
        other => panic!("expected Frame::Video, got {other:?}"),
    };

    // Geometry: 128x96 luma, 3-plane 4:2:0.
    assert_eq!(vf.planes.len(), 3, "expected 3-plane Yuv420P layout");
    assert_eq!(vf.planes[0].stride, WIDTH);
    assert_eq!(vf.planes[0].data.len(), WIDTH * HEIGHT);

    // Luma PSNR vs the ffmpeg black-box decode.
    let mut sse = 0i64;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let got = vf.planes[0].data[y * vf.planes[0].stride + x] as i64;
            let exp = reference[y * WIDTH + x] as i64;
            let d = got - exp;
            sse += d * d;
        }
    }
    let mse = sse as f64 / (WIDTH * HEIGHT) as f64;
    let psnr = if mse == 0.0 {
        99.0
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    };
    eprintln!("MBAFF I-frame luma PSNR = {psnr:.2} dB (mse={mse:.3})");

    // Conformance gate: the MBAFF pair-interleaved intra path must
    // reconstruct the picture to within typical deblock-rounding noise
    // of the reference decode (PSNR >= 40 dB).
    assert!(
        psnr >= 40.0,
        "MBAFF I-frame luma PSNR {psnr:.2} dB below 40 dB conformance gate"
    );
}
