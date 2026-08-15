//! Round-443 — B-frame rate control ([`EncoderSession`] with
//! `b_frames > 0`).
//!
//! Drives multi-GOP CBR and capped-VBR encodes whose GOPs interleave
//! non-reference B pictures between the anchors (display order
//! `I B..B P B..B P …`, coded anchor-first) and checks:
//!
//! * mini-GOP scheduling: kind / display-index sequence including the
//!   truncated pre-IDR tail and the [`EncoderSession::finish`] drain;
//! * rate accuracy: |actual − target| / target within bounds after
//!   the first-GOP warmup, for CBR and capped VBR, CAVLC and CABAC;
//! * the λ-scaled QP hierarchy: non-reference B pictures settle at a
//!   coarser QP than the P anchors they predict from;
//! * rate-distortion sanity: at a comparable rate the B-GOP CBR
//!   encode stays within a bounded PSNR_Y margin of the fixed-QP
//!   B-GOP anchor curve;
//! * every stream decodes in our own registry decoder AND
//!   byte-identically in a black-box reference decoder (when the
//!   binary is present) — reordered output, SEI and filler included.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::rate_control::RateControlConfig;
use oxideav_h264::encoder::session::{
    EncoderSession, SessionConfig, SessionFrame, SessionFrameKind,
};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use std::process::Command;

const FFMPEG: &str = "/opt/homebrew/bin/ffmpeg";

const W: u32 = 80;
const H: u32 = 64;
const FPS: f64 = 30.0;

/// Moving-texture source: gradient background, two moving blocks, a
/// mild scene change every 45 frames — same family as the round-420
/// session suite so the B results compare against known behaviour.
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

/// Push `frames` display-order pictures + finish; returns the decode-
/// order Annex B stream, the per-AU infos (decode order) and the
/// display-order source luma planes.
fn run_b_session(
    mut s: EncoderSession,
    frames: usize,
) -> (Vec<u8>, Vec<SessionFrame>, Vec<Vec<u8>>) {
    let mut stream = Vec::new();
    let mut infos = Vec::new();
    let mut sources = Vec::new();
    for n in 0..frames {
        let (y, u, v) = make_frame(n);
        for sf in s.push_frame(&y, &u, &v) {
            stream.extend_from_slice(&sf.annex_b);
            infos.push(sf);
        }
        sources.push(y);
    }
    for sf in s.finish() {
        stream.extend_from_slice(&sf.annex_b);
        infos.push(sf);
    }
    assert_eq!(infos.len(), frames, "one AU per pushed picture");
    (stream, infos, sources)
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
    let dir = std::env::temp_dir().join(format!("oxideav-h264-r443b-{}", std::process::id()));
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

/// Mean PSNR_Y of the decoded stream (display order) vs the sources.
fn psnr_y(sources: &[Vec<u8>], decoded: &[VideoFrame]) -> f64 {
    assert_eq!(sources.len(), decoded.len());
    let mut sse = 0u64;
    for (src, vf) in sources.iter().zip(decoded.iter()) {
        for (a, b) in src.iter().zip(vf.planes[0].data.iter()) {
            let d = i64::from(*a) - i64::from(*b);
            sse += (d * d) as u64;
        }
    }
    let n = (sources.len() as u64) * u64::from(W) * u64::from(H);
    let mse = sse as f64 / n as f64;
    if mse > 0.0 {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    } else {
        f64::INFINITY
    }
}

fn mean_qp(infos: &[SessionFrame], kind: SessionFrameKind, skip: usize) -> f64 {
    let v: Vec<i32> = infos[skip..]
        .iter()
        .filter(|s| s.kind == kind)
        .map(|s| s.qp)
        .collect();
    assert!(!v.is_empty(), "no {kind:?} AUs after warmup");
    v.iter().sum::<i32>() as f64 / v.len() as f64
}

#[test]
fn b_gop_structure_and_kinds() {
    // gop 8, b_frames 3, 18 pictures: exercises the regular mini-GOP,
    // the truncated pre-IDR tail (8 % 4 != 0) and the finish() drain.
    let mut cfg = SessionConfig::constant_qp(W, H, 30);
    cfg.gop_length = 8;
    cfg.b_frames = 3;
    let (stream, infos, sources) = run_b_session(EncoderSession::new(cfg), 18);

    use SessionFrameKind::{Idr, B, P};
    let expected: &[(SessionFrameKind, u64)] = &[
        (Idr, 0),
        (P, 4),
        (B, 1),
        (B, 2),
        (B, 3),
        (P, 7), // truncated tail before the IDR at 8
        (B, 5),
        (B, 6),
        (Idr, 8),
        (P, 12),
        (B, 9),
        (B, 10),
        (B, 11),
        (P, 15),
        (B, 13),
        (B, 14),
        (Idr, 16),
        (P, 17), // finish(): single-picture tail codes as P
    ];
    let got: Vec<(SessionFrameKind, u64)> =
        infos.iter().map(|s| (s.kind, s.display_index)).collect();
    assert_eq!(got, expected, "mini-GOP schedule mismatch");
    for s in &infos {
        assert_eq!(s.is_idr, s.kind == Idr);
    }

    let own = decode_own(&stream);
    assert_eq!(own.len(), 18, "own decoder frame count");
    // Display order must come back monotone: PSNR vs the display-order
    // sources collapses if the DPB mis-orders the Bs.
    let p = psnr_y(&sources, &own);
    eprintln!(
        "b-gop structure stream: {} bytes, PSNR_Y {p:.2} dB",
        stream.len()
    );
    assert!(p > 30.0, "display-order PSNR_Y {p:.2} dB implausibly low");
    cross_check_reference(&stream, &own, "b-structure");
}

#[test]
fn cbr_b_session_hits_target_and_decodes() {
    const TARGET: u32 = 120_000;
    let mut cfg = SessionConfig::rate_controlled(W, H, RateControlConfig::cbr(TARGET, 30, 1));
    cfg.b_frames = 2;
    let (stream, infos, sources) = run_b_session(EncoderSession::new(cfg), 92);

    // Payload accuracy after the first GOP (warmup transient).
    let skip = 30;
    let tail: u64 = infos[skip..].iter().map(|s| s.payload_bits).sum();
    let payload = tail as f64 / ((infos.len() - skip) as f64 / FPS);
    let err = (payload - f64::from(TARGET)).abs() / f64::from(TARGET);
    eprintln!(
        "CBR B-GOP payload rate {payload:.0} bps (err {:.2}%)",
        err * 100.0
    );
    assert!(err < 0.10, "CBR B-GOP payload err {err:.4} over 10%");

    // λ-scaled hierarchy: B settles coarser than P, IDR no coarser
    // than P.
    let (qi, qp, qb) = (
        mean_qp(&infos, SessionFrameKind::Idr, skip),
        mean_qp(&infos, SessionFrameKind::P, skip),
        mean_qp(&infos, SessionFrameKind::B, skip),
    );
    eprintln!("mean QP: IDR {qi:.2}  P {qp:.2}  B {qb:.2}");
    assert!(
        qb > qp - 0.5,
        "B mean QP {qb:.2} not above P mean QP {qp:.2}"
    );
    assert!(
        qi <= qp + 0.5,
        "IDR mean QP {qi:.2} above P mean QP {qp:.2}"
    );

    // All QPs in window; the controller genuinely adapted.
    let qps: Vec<i32> = infos.iter().map(|s| s.qp).collect();
    assert!(qps.iter().all(|&q| (10..=51).contains(&q)));
    assert!(qps.iter().any(|&q| q != qps[0]), "QP never adapted");

    let own = decode_own(&stream);
    assert_eq!(own.len(), infos.len(), "own decoder frame count");
    let p = psnr_y(&sources, &own);
    eprintln!("CBR B-GOP PSNR_Y {p:.2} dB");
    cross_check_reference(&stream, &own, "cbr-b-session");
}

#[test]
fn capped_vbr_b_session_tracks_average() {
    const TARGET: u32 = 100_000;
    let mut cfg =
        SessionConfig::rate_controlled(W, H, RateControlConfig::capped_vbr(TARGET, 200_000, 30, 1));
    cfg.b_frames = 2;
    let (stream, infos, sources) = run_b_session(EncoderSession::new(cfg), 121);

    let bits: u64 = infos.iter().map(|s| s.payload_bits).sum();
    let payload = bits as f64 / (infos.len() as f64 / FPS);
    let err = (payload - f64::from(TARGET)).abs() / f64::from(TARGET);
    eprintln!(
        "VBR B-GOP payload rate {payload:.0} bps (err {:.2}%)",
        err * 100.0
    );
    assert!(err < 0.10, "capped-VBR B-GOP err {err:.4} over 10%");
    assert!(
        infos.iter().all(|s| s.filler_bits == 0),
        "VBR must not emit filler"
    );

    let own = decode_own(&stream);
    assert_eq!(own.len(), infos.len(), "own decoder frame count");
    let p = psnr_y(&sources, &own);
    eprintln!("VBR B-GOP PSNR_Y {p:.2} dB");
    cross_check_reference(&stream, &own, "vbr-b-session");
}

#[test]
fn cbr_cabac_b_session_decodes() {
    const TARGET: u32 = 100_000;
    let mut cfg = SessionConfig::rate_controlled(W, H, RateControlConfig::cbr(TARGET, 30, 1));
    cfg.cabac = true;
    cfg.b_frames = 2;
    let (stream, infos, sources) = run_b_session(EncoderSession::new(cfg), 76);

    let skip = 30;
    let tail: u64 = infos[skip..].iter().map(|s| s.payload_bits).sum();
    let payload = tail as f64 / ((infos.len() - skip) as f64 / FPS);
    let err = (payload - f64::from(TARGET)).abs() / f64::from(TARGET);
    eprintln!(
        "CBR/CABAC B-GOP payload rate {payload:.0} bps (err {:.2}%)",
        err * 100.0
    );
    assert!(err < 0.10, "CBR/CABAC B-GOP err {err:.4} over 10%");

    let qb = mean_qp(&infos, SessionFrameKind::B, skip);
    let qp = mean_qp(&infos, SessionFrameKind::P, skip);
    assert!(qb > qp - 0.5, "CABAC B mean QP {qb:.2} not above P {qp:.2}");

    let own = decode_own(&stream);
    assert_eq!(own.len(), infos.len(), "own decoder frame count");
    let p = psnr_y(&sources, &own);
    eprintln!("CBR/CABAC B-GOP PSNR_Y {p:.2} dB");
    cross_check_reference(&stream, &own, "cbr-cabac-b-session");
}

#[test]
fn constant_qp_b_session_decodes_and_matches_reference() {
    let mut cfg = SessionConfig::constant_qp(W, H, 28);
    cfg.gop_length = 15;
    cfg.b_frames = 2;
    let (stream, infos, sources) = run_b_session(EncoderSession::new(cfg), 45);
    assert!(infos.iter().all(|s| s.qp == 28));
    assert!(infos.iter().all(|s| s.sei_bits == 0 && s.filler_bits == 0));
    let own = decode_own(&stream);
    assert_eq!(own.len(), infos.len());
    let p = psnr_y(&sources, &own);
    eprintln!("constant-QP B-GOP PSNR_Y {p:.2} dB");
    assert!(p > 30.0);
    cross_check_reference(&stream, &own, "cqp-b-session");
}

#[test]
fn b_rate_control_holds_the_rd_curve() {
    // Fixed-QP B-GOP anchors (gop 30, b 2 — the same GOP shape the
    // controlled session uses).
    let mut anchors: Vec<(f64, f64)> = Vec::new(); // (bps, psnr_y)
    for qp in [26, 30, 34] {
        let mut cfg = SessionConfig::constant_qp(W, H, qp);
        cfg.b_frames = 2;
        let (stream, infos, sources) = run_b_session(EncoderSession::new(cfg), 60);
        let bits: u64 = infos.iter().map(|s| s.payload_bits).sum();
        let bps = bits as f64 / (infos.len() as f64 / FPS);
        let own = decode_own(&stream);
        assert_eq!(own.len(), infos.len());
        let p = psnr_y(&sources, &own);
        eprintln!("B-GOP anchor QP {qp}: {bps:.0} bps, PSNR_Y {p:.2} dB");
        anchors.push((bps, p));
    }
    assert!(anchors[0].0 > anchors[1].0 && anchors[1].0 > anchors[2].0);
    assert!(anchors[0].1 > anchors[1].1 && anchors[1].1 > anchors[2].1);

    // CBR B-GOP at the middle anchor's rate.
    let target = anchors[1].0.round() as u32;
    let mut cfg = SessionConfig::rate_controlled(W, H, RateControlConfig::cbr(target, 30, 1));
    cfg.b_frames = 2;
    let (stream, infos, sources) = run_b_session(EncoderSession::new(cfg), 60);
    let payload: u64 = infos.iter().map(|s| s.payload_bits).sum();
    let sei: u64 = infos.iter().map(|s| s.sei_bits).sum();
    let cbr_channel_bps = payload as f64 / (infos.len() as f64 / FPS);
    // RD comparison on picture-coding bits (the fixed-QP anchors are
    // unannotated; HRD SEI is channel payload, not picture coding).
    let cbr_bps = (payload - sei) as f64 / (infos.len() as f64 / FPS);
    let own = decode_own(&stream);
    assert_eq!(own.len(), infos.len());
    let cbr_psnr = psnr_y(&sources, &own);
    eprintln!(
        "CBR B-GOP @ {target} bps: {cbr_bps:.0} bps, PSNR_Y {cbr_psnr:.2} dB (anchor {:.2} dB)",
        anchors[1].1
    );

    let curve_psnr = {
        let (lo, hi) = if cbr_bps <= anchors[1].0 {
            (anchors[2], anchors[1])
        } else {
            (anchors[1], anchors[0])
        };
        let t = ((cbr_bps.ln() - lo.0.ln()) / (hi.0.ln() - lo.0.ln())).clamp(0.0, 1.0);
        lo.1 + t * (hi.1 - lo.1)
    };
    let loss = curve_psnr - cbr_psnr;
    eprintln!("B-GOP RD check: curve {curve_psnr:.2} dB, CBR {cbr_psnr:.2} dB, loss {loss:.2} dB");
    assert!(
        loss < 1.5,
        "B-GOP rate control lost {loss:.2} dB versus the fixed-QP RD curve"
    );
    let err = (cbr_channel_bps - f64::from(target)).abs() / f64::from(target);
    assert!(err < 0.12, "B-GOP RD-check CBR rate err {err:.4}");
}
