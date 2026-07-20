//! Round-420 — per-frame QP overrides (`encode_*_with_qp`).
//!
//! Rate control needs every picture to pick its own QP while the
//! sequence keeps a single PPS. The §7.4.3 route is `slice_qp_delta`
//! (`SliceQP_Y = 26 + pic_init_qp_minus26 + slice_qp_delta`). These
//! tests drive IDR + P chains whose per-frame QP differs from the PPS
//! `pic_init_qp_minus26` on both entropy paths and require:
//!
//! * our own registry decoder reproduces the encoder's local recon
//!   byte-exactly for every frame;
//! * a black-box reference decoder (when present) produces the same
//!   planes byte-exactly.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use std::path::PathBuf;
use std::process::Command;

const FFMPEG: &str = "/opt/homebrew/bin/ffmpeg";

const W: u32 = 64;
const H: u32 = 64;

/// Deterministic moving-texture frame: diagonal gradient + a bright
/// block that shifts per frame (so P frames have real motion + coded
/// residual at moderate QP).
fn make_frame(n: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (W as usize, H as usize);
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; (w / 2) * (h / 2)];
    let mut v = vec![0u8; (w / 2) * (h / 2)];
    for j in 0..h {
        for i in 0..w {
            let base = ((i * 3 + j * 2 + n * 5) % 200) as u8;
            y[j * w + i] = 20 + base;
        }
    }
    // Moving 12x12 bright block.
    let bx = (4 + n * 6) % (w - 12);
    let by = (8 + n * 4) % (h - 12);
    for j in by..by + 12 {
        for i in bx..bx + 12 {
            y[j * w + i] = 235;
        }
    }
    for j in 0..h / 2 {
        for i in 0..w / 2 {
            u[j * (w / 2) + i] = (100 + ((i + n * 2) % 40)) as u8;
            v[j * (w / 2) + i] = (140 + ((j + n) % 30)) as u8;
        }
    }
    (y, u, v)
}

fn decode_own(stream: &[u8]) -> Vec<VideoFrame> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let packet = Packet::new(0, TimeBase::new(1, 25), stream.to_vec()).with_pts(0);
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

fn assert_frame_matches(vf: &VideoFrame, recon: (&[u8], &[u8], &[u8]), tag: &str) {
    assert_eq!(vf.planes[0].data, recon.0, "{tag}: luma mismatch");
    assert_eq!(vf.planes[1].data, recon.1, "{tag}: cb mismatch");
    assert_eq!(vf.planes[2].data, recon.2, "{tag}: cr mismatch");
}

/// Black-box reference decode: returns the concatenated planar YUV of
/// all frames, or `None` when the binary is absent (test soft-skips).
fn reference_decode(stream: &[u8], tag: &str) -> Option<Vec<u8>> {
    if !std::path::Path::new(FFMPEG).exists() {
        eprintln!("skip reference cross-check: binary not present");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h264-r420-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let h264: PathBuf = dir.join(format!("{tag}.h264"));
    let yuv: PathBuf = dir.join(format!("{tag}.yuv"));
    std::fs::write(&h264, stream).expect("write stream");
    let status = Command::new(FFMPEG)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-y"])
        .arg(&yuv)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success(), "{tag}: reference decoder rejected stream");
    Some(std::fs::read(&yuv).expect("read reference yuv"))
}

fn frame_size_yuv420() -> usize {
    (W as usize * H as usize) * 3 / 2
}

fn check_reference(stream: &[u8], recons: &[(&[u8], &[u8], &[u8])], tag: &str) {
    let Some(raw) = reference_decode(stream, tag) else {
        return;
    };
    assert_eq!(
        raw.len(),
        frame_size_yuv420() * recons.len(),
        "{tag}: reference decoded frame count"
    );
    for (n, recon) in recons.iter().enumerate() {
        let base = n * frame_size_yuv420();
        let ysz = W as usize * H as usize;
        let csz = ysz / 4;
        assert_eq!(
            &raw[base..base + ysz],
            recon.0,
            "{tag}: frame {n} reference luma"
        );
        assert_eq!(
            &raw[base + ysz..base + ysz + csz],
            recon.1,
            "{tag}: frame {n} reference cb"
        );
        assert_eq!(
            &raw[base + ysz + csz..base + ysz + 2 * csz],
            recon.2,
            "{tag}: frame {n} reference cr"
        );
    }
}

#[test]
fn cavlc_per_frame_qp_roundtrip() {
    let cfg = EncoderConfig::new(W, H); // PPS pic_init_qp = 26
    let enc = Encoder::new(cfg);

    let (y0, u0, v0) = make_frame(0);
    let (y1, u1, v1) = make_frame(1);
    let (y2, u2, v2) = make_frame(2);
    let f0 = YuvFrame {
        width: W,
        height: H,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let f1 = YuvFrame {
        width: W,
        height: H,
        y: &y1,
        u: &u1,
        v: &v1,
    };
    let f2 = YuvFrame {
        width: W,
        height: H,
        y: &y2,
        u: &u2,
        v: &v2,
    };

    // IDR far above the PPS QP, P frames stepping down.
    let idr = enc.encode_idr_with_qp(&f0, 38);
    let p1 = enc.encode_p_with_qp(&f1, &EncodedFrameRef::from(&idr), 1, 2, 31);
    let p2 = enc.encode_p_with_qp(&f2, &EncodedFrameRef::from(&p1), 2, 4, 22);

    let mut stream = idr.annex_b.clone();
    stream.extend_from_slice(&p1.annex_b);
    stream.extend_from_slice(&p2.annex_b);

    let frames = decode_own(&stream);
    assert_eq!(frames.len(), 3, "expected IDR + 2 P frames");
    assert_frame_matches(
        &frames[0],
        (&idr.recon_y, &idr.recon_u, &idr.recon_v),
        "idr@38",
    );
    assert_frame_matches(&frames[1], (&p1.recon_y, &p1.recon_u, &p1.recon_v), "p1@31");
    assert_frame_matches(&frames[2], (&p2.recon_y, &p2.recon_u, &p2.recon_v), "p2@22");

    check_reference(
        &stream,
        &[
            (&idr.recon_y, &idr.recon_u, &idr.recon_v),
            (&p1.recon_y, &p1.recon_u, &p1.recon_v),
            (&p2.recon_y, &p2.recon_u, &p2.recon_v),
        ],
        "cavlc-frame-qp",
    );
}

#[test]
fn cabac_per_frame_qp_roundtrip() {
    let mut cfg = EncoderConfig::new(W, H);
    cfg.cabac = true;
    cfg.profile_idc = 77;
    let enc = Encoder::new(cfg);

    let (y0, u0, v0) = make_frame(0);
    let (y1, u1, v1) = make_frame(1);
    let (y2, u2, v2) = make_frame(2);

    let idr = enc.encode_idr_cabac_with_qp(
        &YuvFrame {
            width: W,
            height: H,
            y: &y0,
            u: &u0,
            v: &v0,
        },
        40,
    );
    let p1 = enc.encode_p_cabac_with_qp(
        &YuvFrame {
            width: W,
            height: H,
            y: &y1,
            u: &u1,
            v: &v1,
        },
        &EncodedFrameRef::from(&idr),
        1,
        2,
        30,
    );
    let p2 = enc.encode_p_cabac_with_qp(
        &YuvFrame {
            width: W,
            height: H,
            y: &y2,
            u: &u2,
            v: &v2,
        },
        &EncodedFrameRef::from(&p1),
        2,
        4,
        24,
    );

    let mut stream = idr.annex_b.clone();
    stream.extend_from_slice(&p1.annex_b);
    stream.extend_from_slice(&p2.annex_b);

    let frames = decode_own(&stream);
    assert_eq!(frames.len(), 3, "expected IDR + 2 P frames");
    assert_frame_matches(
        &frames[0],
        (&idr.recon_y, &idr.recon_u, &idr.recon_v),
        "idr@40",
    );
    assert_frame_matches(&frames[1], (&p1.recon_y, &p1.recon_u, &p1.recon_v), "p1@30");
    assert_frame_matches(&frames[2], (&p2.recon_y, &p2.recon_u, &p2.recon_v), "p2@24");

    check_reference(
        &stream,
        &[
            (&idr.recon_y, &idr.recon_u, &idr.recon_v),
            (&p1.recon_y, &p1.recon_u, &p1.recon_v),
            (&p2.recon_y, &p2.recon_u, &p2.recon_v),
        ],
        "cabac-frame-qp",
    );
}

#[test]
fn frame_qp_actually_changes_rate() {
    let cfg = EncoderConfig::new(W, H);
    let enc = Encoder::new(cfg);
    let (y0, u0, v0) = make_frame(0);
    let frame = YuvFrame {
        width: W,
        height: H,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let hi = enc.encode_idr_with_qp(&frame, 20);
    let lo = enc.encode_idr_with_qp(&frame, 44);
    assert!(
        hi.annex_b.len() > 2 * lo.annex_b.len(),
        "QP 20 ({}) should cost far more than QP 44 ({})",
        hi.annex_b.len(),
        lo.annex_b.len()
    );
    // Default-QP wrapper must be identical to the explicit form.
    let a = enc.encode_idr(&frame);
    let b = enc.encode_idr_with_qp(&frame, 26);
    assert_eq!(a.annex_b, b.annex_b, "wrapper/default-QP equivalence");
}
