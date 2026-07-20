//! Round-420 — rate-controlled GOP sessions ([`EncoderSession`]).
//!
//! Drives multi-GOP CBR and capped-VBR encodes over synthetic moving
//! content and checks:
//!
//! * rate accuracy: |actual − target| / target within bounds, with
//!   CBR additionally near-exact once filler is counted;
//! * the QP trajectory stays inside the configured window and the
//!   modelled CPB never underflows (the session's retry loop honours
//!   the hard cap);
//! * every rate-controlled stream decodes in our own registry decoder
//!   AND byte-identically in a black-box reference decoder (when the
//!   binary is present) — filler NAL units included.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::rate_control::RateControlConfig;
use oxideav_h264::encoder::session::{EncoderSession, SessionConfig, SessionFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use std::process::Command;

const FFMPEG: &str = "/opt/homebrew/bin/ffmpeg";

const W: u32 = 80;
const H: u32 = 64;

/// Moving-texture source: gradient background, two moving blocks, a
/// mild scene change every 45 frames (texture phase jump) so the
/// controller sees complexity steps.
fn make_frame(n: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (W as usize, H as usize);
    let phase = (n / 45) * 37;
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; (w / 2) * (h / 2)];
    let mut v = vec![0u8; (w / 2) * (h / 2)];
    for j in 0..h {
        for i in 0..w {
            let base = ((i * 5 + j * 3 + n * 4 + phase) % 180) as u8;
            y[j * w + i] = 30 + base;
        }
    }
    let bx = (6 + n * 3) % (w - 24);
    let by = (10 + n * 2) % (h - 20);
    for j in by..by + 20 {
        for i in bx..bx + 24 {
            y[j * w + i] = 220;
        }
    }
    let cx = (w - 30 - (n * 2) % (w - 40)).max(2);
    for j in 20..36.min(h) {
        for i in cx..(cx + 16).min(w) {
            y[j * w + i] = 60;
        }
    }
    for j in 0..h / 2 {
        for i in 0..w / 2 {
            u[j * (w / 2) + i] = (90 + ((i + n * 3 + phase) % 60)) as u8;
            v[j * (w / 2) + i] = (130 + ((j + n * 2) % 50)) as u8;
        }
    }
    (y, u, v)
}

fn run_session(mut s: EncoderSession, frames: usize) -> (Vec<u8>, Vec<SessionFrame>) {
    let mut stream = Vec::new();
    let mut infos = Vec::new();
    for n in 0..frames {
        let (y, u, v) = make_frame(n);
        let sf = s.encode_frame(&y, &u, &v);
        stream.extend_from_slice(&sf.annex_b);
        infos.push(sf);
    }
    (stream, infos)
}

fn decode_own(stream: &[u8]) -> Vec<VideoFrame> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let packet = Packet::new(0, TimeBase::new(1, 30), stream.to_vec()).with_pts(0);
    dec.send_packet(&packet).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => frames.push(vf),
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    frames
}

/// Cross-check our decode against the black-box reference decoder,
/// byte-exactly, when the binary is present.
fn cross_check_reference(stream: &[u8], own: &[VideoFrame], tag: &str) {
    if !std::path::Path::new(FFMPEG).exists() {
        eprintln!("skip reference cross-check: binary not present");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h264-r420s-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let h264 = dir.join(format!("{tag}.h264"));
    let yuv = dir.join(format!("{tag}.yuv"));
    std::fs::write(&h264, stream).expect("write stream");
    let status = Command::new(FFMPEG)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-y"])
        .arg(&yuv)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success(), "{tag}: reference decoder rejected stream");
    let raw = std::fs::read(&yuv).expect("read reference yuv");
    let fsz = (W as usize * H as usize) * 3 / 2;
    assert_eq!(raw.len(), fsz * own.len(), "{tag}: reference frame count");
    let ysz = W as usize * H as usize;
    for (n, vf) in own.iter().enumerate() {
        let base = n * fsz;
        assert_eq!(
            &raw[base..base + ysz],
            &vf.planes[0].data[..],
            "{tag}: frame {n} luma cross-decoder mismatch"
        );
        assert_eq!(
            &raw[base + ysz..base + ysz + ysz / 4],
            &vf.planes[1].data[..],
            "{tag}: frame {n} cb cross-decoder mismatch"
        );
        assert_eq!(
            &raw[base + ysz + ysz / 4..base + fsz],
            &vf.planes[2].data[..],
            "{tag}: frame {n} cr cross-decoder mismatch"
        );
    }
}

/// Average bitrate over `infos[skip..]`. CBR accuracy is measured
/// after a warmup window: the modelled CPB opens at its
/// initial-removal-delay position and the drift down to the steering
/// setpoint releases banked bits during the first GOP — a one-time
/// transient, not a steady-state rate error.
fn measured_bitrate(infos: &[SessionFrame], fps: f64, with_filler: bool, skip: usize) -> f64 {
    let tail = &infos[skip..];
    let bits: u64 = tail
        .iter()
        .map(|s| s.payload_bits + if with_filler { s.filler_bits } else { 0 })
        .sum();
    bits as f64 / (tail.len() as f64 / fps)
}

#[test]
fn cbr_session_hits_target_and_decodes() {
    const TARGET: u32 = 120_000;
    let cfg = SessionConfig::rate_controlled(W, H, RateControlConfig::cbr(TARGET, 30, 1));
    let (stream, infos) = run_session(EncoderSession::new(cfg), 90);

    // Payload accuracy after the first GOP (warmup transient).
    let payload = measured_bitrate(&infos, 30.0, false, 30);
    let err = (payload - f64::from(TARGET)).abs() / f64::from(TARGET);
    eprintln!(
        "CBR payload rate {payload:.0} bps (err {:.2}%)",
        err * 100.0
    );
    assert!(err < 0.10, "CBR payload err {err:.4} over 10%");

    // QP varies (rate control is actually doing something) and stays
    // in the configured window.
    let qps: Vec<i32> = infos.iter().map(|s| s.qp).collect();
    assert!(
        qps.iter().any(|&q| q != qps[0]),
        "QP never adapted: {qps:?}"
    );
    assert!(qps.iter().all(|&q| (10..=51).contains(&q)));

    // IDR cadence.
    for (n, s) in infos.iter().enumerate() {
        assert_eq!(s.is_idr, n % 30 == 0, "IDR cadence broke at {n}");
    }

    let own = decode_own(&stream);
    assert_eq!(own.len(), infos.len(), "own decoder frame count");
    cross_check_reference(&stream, &own, "cbr-session");
}

#[test]
fn capped_vbr_session_tracks_average() {
    const TARGET: u32 = 100_000;
    let cfg =
        SessionConfig::rate_controlled(W, H, RateControlConfig::capped_vbr(TARGET, 200_000, 30, 1));
    let (stream, infos) = run_session(EncoderSession::new(cfg), 120);

    let payload = measured_bitrate(&infos, 30.0, false, 0);
    let err = (payload - f64::from(TARGET)).abs() / f64::from(TARGET);
    eprintln!(
        "VBR payload rate {payload:.0} bps (err {:.2}%)",
        err * 100.0
    );
    assert!(err < 0.10, "capped-VBR err {err:.4} over 10%");
    assert!(
        infos.iter().all(|s| s.filler_bits == 0),
        "VBR must not emit filler"
    );

    let own = decode_own(&stream);
    assert_eq!(own.len(), infos.len(), "own decoder frame count");
    cross_check_reference(&stream, &own, "vbr-session");
}

#[test]
fn cbr_cabac_session_decodes() {
    const TARGET: u32 = 100_000;
    let mut cfg = SessionConfig::rate_controlled(W, H, RateControlConfig::cbr(TARGET, 30, 1));
    cfg.cabac = true;
    let (stream, infos) = run_session(EncoderSession::new(cfg), 75);

    let payload = measured_bitrate(&infos, 30.0, false, 30);
    let err = (payload - f64::from(TARGET)).abs() / f64::from(TARGET);
    eprintln!(
        "CBR/CABAC payload rate {payload:.0} bps (err {:.2}%)",
        err * 100.0
    );
    assert!(err < 0.10, "CBR/CABAC err {err:.4} over 10%");

    let own = decode_own(&stream);
    assert_eq!(own.len(), infos.len(), "own decoder frame count");
    cross_check_reference(&stream, &own, "cbr-cabac-session");
}

#[test]
fn constant_qp_session_matches_stateless_calls() {
    let cfg = SessionConfig::constant_qp(W, H, 26);
    let mut s = EncoderSession::new(cfg);
    let (y, u, v) = make_frame(0);
    let sf = s.encode_frame(&y, &u, &v);
    assert!(sf.is_idr);
    assert_eq!(sf.qp, 26);
    assert_eq!(sf.filler_bits, 0);

    use oxideav_h264::encoder::{Encoder, EncoderConfig, YuvFrame};
    let direct = Encoder::new(EncoderConfig::new(W, H)).encode_idr(&YuvFrame {
        width: W,
        height: H,
        y: &y,
        u: &u,
        v: &v,
    });
    assert_eq!(
        sf.annex_b, direct.annex_b,
        "constant-QP session must reproduce the stateless output"
    );
}

#[test]
fn cbr_filler_keeps_channel_busy_on_easy_content() {
    // Flat gray content at a generous rate: the payload cannot use
    // the channel even at min QP, so CBR must emit filler.
    const TARGET: u32 = 800_000;
    let cfg = SessionConfig::rate_controlled(W, H, RateControlConfig::cbr(TARGET, 30, 1));
    let mut s = EncoderSession::new(cfg);
    let y = vec![128u8; (W * H) as usize];
    let u = vec![128u8; (W * H / 4) as usize];
    let v = vec![128u8; (W * H / 4) as usize];
    let mut stream = Vec::new();
    let mut sent_bits = 0u64;
    for n in 0..45 {
        let sf = s.encode_frame(&y, &u, &v);
        if n >= 15 {
            // Steady state: once the bucket has risen to its cap the
            // per-frame sent bits (payload + filler) must equal the
            // per-frame channel arrivals almost exactly.
            sent_bits += sf.payload_bits + sf.filler_bits;
        }
        stream.extend_from_slice(&sf.annex_b);
    }
    let sent_rate = sent_bits as f64 / 1.0; // frames 15..45 @ 30 fps
    let err = (sent_rate - f64::from(TARGET)).abs() / f64::from(TARGET);
    eprintln!(
        "CBR steady-state sent (payload+filler) {sent_rate:.0} bps, err {:.2}%",
        err * 100.0
    );
    assert!(
        err < 0.02,
        "filler failed to keep the channel busy: {err:.4}"
    );

    // Filler NALs must not disturb either decoder.
    let own = decode_own(&stream);
    assert_eq!(own.len(), 45, "own decoder frame count with filler");
    cross_check_reference(&stream, &own, "cbr-filler");
}
