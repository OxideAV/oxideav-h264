//! Optional ffmpeg cross-check: encode a frame with `oxideav-h264`, pipe
//! the resulting Annex B stream into `ffmpeg`, decode to raw YUV, and
//! compare against the encoder's input. Skipped if the `ffmpeg` binary
//! is not on `PATH`.
//!
//! Asserts luma PSNR ≥ 28 dB on a 64×48 gradient at QP 22 — the same
//! bar the self-round-trip tests use. If this passes, the output is
//! standard-compliant enough for a reference decoder.

use std::io::Write;
use std::process::{Command, Stdio};

use oxideav_codec::Encoder;
use oxideav_core::{frame::VideoPlane, CodecId, Frame, PixelFormat, TimeBase, VideoFrame};
use oxideav_h264::encoder::{H264Encoder, H264EncoderOptions};

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn make_gradient_frame(w: u32, h: u32) -> VideoFrame {
    let mut y = vec![0u8; (w * h) as usize];
    let cw = w / 2;
    let ch = h / 2;
    let cb = vec![128u8; (cw * ch) as usize];
    let cr = vec![128u8; (cw * ch) as usize];
    for r in 0..h {
        for c in 0..w {
            y[(r * w + c) as usize] = ((r + c).min(255)) as u8;
        }
    }
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width: w,
        height: h,
        pts: Some(0),
        time_base: TimeBase::new(1, 30),
        planes: vec![
            VideoPlane {
                stride: w as usize,
                data: y,
            },
            VideoPlane {
                stride: cw as usize,
                data: cb,
            },
            VideoPlane {
                stride: cw as usize,
                data: cr,
            },
        ],
    }
}

#[test]
fn ffmpeg_decodes_our_h264_output() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not on PATH — skipping");
        return;
    }
    let w = 64u32;
    let h = 48u32;
    let src = make_gradient_frame(w, h);
    let mut enc = H264Encoder::new(
        CodecId::new("h264"),
        w,
        h,
        H264EncoderOptions {
            qp: 22,
            ..Default::default()
        },
    )
    .expect("encoder::new");
    enc.send_frame(&Frame::Video(src.clone()))
        .expect("send_frame");
    enc.flush().expect("flush");
    let pkt = enc.receive_packet().expect("receive_packet");

    // Pipe Annex B stream into ffmpeg and read raw YUV from stdout.
    let mut cmd = Command::new("ffmpeg")
        .args([
            "-loglevel",
            "error",
            "-f",
            "h264",
            "-i",
            "-",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffmpeg");

    {
        let stdin = cmd.stdin.as_mut().expect("stdin");
        stdin.write_all(&pkt.data).expect("write_all");
    }
    let out = cmd.wait_with_output().expect("wait");
    assert!(
        out.status.success(),
        "ffmpeg exit: {:?}, stderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let expected_size = (w * h) as usize + (w * h / 2) as usize; // YUV420P
    assert_eq!(
        out.stdout.len(),
        expected_size,
        "ffmpeg produced {} bytes, expected {} (Y+Cb+Cr for {}×{})",
        out.stdout.len(),
        expected_size,
        w,
        h
    );
    let y_out = &out.stdout[..(w * h) as usize];
    let y_src = &src.planes[0].data;
    let mse: f64 = y_out
        .iter()
        .zip(y_src.iter())
        .map(|(a, b)| {
            let d = *a as f64 - *b as f64;
            d * d
        })
        .sum::<f64>()
        / (y_out.len() as f64);
    let psnr = if mse < 1e-9 {
        99.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    };
    eprintln!(
        "ffmpeg-decode gradient 64×48 qp22: packet={} bytes, luma psnr = {:.2} dB",
        pkt.data.len(),
        psnr
    );
    assert!(
        psnr >= 28.0,
        "ffmpeg-decoded luma psnr {:.2} < 28 dB — our bitstream diverged from spec?",
        psnr
    );
}
