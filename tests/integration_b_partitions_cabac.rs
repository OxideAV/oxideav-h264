//! Round-475 — CABAC B-slice 16x8 / 8x16 partition modes (Table 7-14
//! mb_type 4..=21).
//!
//! Mirrors the round-22 CAVLC partition fixtures (`integration_b_partitions.rs`)
//! but routes through `Encoder::encode_b_cabac`. Per-half motion forces
//! the encoder's per-partition ME to pick mb_type ∈ {B_L0_L1_16x8,
//! B_L0_L1_8x16}, exercising the new CABAC partition wiring.
//!
//! Acceptance: bit-exact (≤ 1 LSB) decoder roundtrip on the B-frame
//! local recon, and PSNR_Y ≥ 38 dB.
//!
//! Cross-reference:
//!   * §7.3.5.1 mb_pred() — per-partition ref_idx + mvd field order
//!   * §7.4.5 Table 7-14 — B mb_type raw 4..=21
//!   * §8.4.1.3 — partition-shape mvp (Partition16x8Top/Bottom, Partition8x16Left/Right)
//!   * §9.3.3.1.1.3 — CABAC mb_type binarisation for B (Table 9-37 rows 4..=21)

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 64;
const H: usize = 64;

fn psnr(orig: &[u8], recon: &[u8]) -> f64 {
    let n = orig.len() as f64;
    let sse: f64 = orig
        .iter()
        .zip(recon.iter())
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum();
    if sse <= 0.0 {
        return 99.0;
    }
    let m = sse / n;
    10.0 * (255.0_f64 * 255.0 / m).log10()
}

/// IDR: smooth horizontal ramp.
fn make_idr() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            y[j * W + i] = (60 + i + j) as u8;
        }
    }
    let u = vec![128u8; (W / 2) * (H / 2)];
    let v = vec![128u8; (W / 2) * (H / 2)];
    (y, u, v)
}

/// P: same as IDR shifted right by 4 luma px.
fn make_p() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let i_src = i.saturating_sub(4);
            y[j * W + i] = (60 + i_src + j) as u8;
        }
    }
    let u = vec![128u8; (W / 2) * (H / 2)];
    let v = vec![128u8; (W / 2) * (H / 2)];
    (y, u, v)
}

/// B with per-half motion: top half of each MB exactly matches the IDR,
/// bottom half exactly matches the P. The encoder's 16x8 partition ME
/// should pick (top=L0, bottom=L1) → mb_type raw 8 (B_L0_L1_16x8).
fn make_b_per_half_motion(idr_y: &[u8], p_y: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for mb_y in 0..(H / 16) {
        for mb_x in 0..(W / 16) {
            for j in 0..16 {
                for i in 0..16 {
                    let py = mb_y * 16 + j;
                    let px = mb_x * 16 + i;
                    let src = if j < 8 {
                        idr_y[py * W + px]
                    } else {
                        p_y[py * W + px]
                    };
                    y[py * W + px] = src;
                }
            }
        }
    }
    let u = vec![128u8; (W / 2) * (H / 2)];
    let v = vec![128u8; (W / 2) * (H / 2)];
    (y, u, v)
}

/// B with vertical-half motion split (LEFT vs RIGHT). Encoder should
/// pick 8x16 with (left=L0, right=L1) → mb_type raw 9 (B_L0_L1_8x16).
fn make_b_per_vhalf_motion(idr_y: &[u8], p_y: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for mb_y in 0..(H / 16) {
        for mb_x in 0..(W / 16) {
            for j in 0..16 {
                for i in 0..16 {
                    let py = mb_y * 16 + j;
                    let px = mb_x * 16 + i;
                    let src = if i < 8 {
                        idr_y[py * W + px]
                    } else {
                        p_y[py * W + px]
                    };
                    y[py * W + px] = src;
                }
            }
        }
    }
    let u = vec![128u8; (W / 2) * (H / 2)];
    let v = vec![128u8; (W / 2) * (H / 2)];
    (y, u, v)
}

fn encode_ipb_cabac(
    f0: &YuvFrame<'_>,
    f1: &YuvFrame<'_>,
    f2: &YuvFrame<'_>,
) -> (
    oxideav_h264::encoder::EncodedIdr,
    oxideav_h264::encoder::EncodedP,
    oxideav_h264::encoder::EncodedB,
) {
    let mut cfg = EncoderConfig::new(W as u32, H as u32);
    cfg.cabac = true;
    cfg.profile_idc = 77; // Main — required for CABAC + B
    cfg.max_num_ref_frames = 2;
    cfg.qp = 26;
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr_cabac(f0);
    let p = enc.encode_p_cabac(f1, &EncodedFrameRef::from(&idr), 1, 4);
    let b = enc.encode_b_cabac(
        f2,
        &EncodedFrameRef::from(&idr),
        &EncodedFrameRef::from(&p),
        1,
        2,
    );
    (idr, p, b)
}

#[test]
fn round475_b_16x8_cabac_self_roundtrip_matches_local_recon() {
    let (y0, u0, v0) = make_idr();
    let (y1, u1, v1) = make_p();
    let (y2, u2, v2) = make_b_per_half_motion(&y0, &y1);

    let f0 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let f1 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y1,
        u: &u1,
        v: &v1,
    };
    let f2 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y2,
        u: &u2,
        v: &v2,
    };

    let (idr, p, b) = encode_ipb_cabac(&f0, &f1, &f2);

    let psnr_local = psnr(&y2, &b.recon_y);
    eprintln!(
        "round-475 CABAC B 16x8: stream={} bytes (IDR {}, P {}, B {}), local PSNR_Y = {:.2} dB",
        idr.annex_b.len() + p.annex_b.len() + b.annex_b.len(),
        idr.annex_b.len(),
        p.annex_b.len(),
        b.annex_b.len(),
        psnr_local,
    );
    assert!(
        psnr_local >= 38.0,
        "round-475 CABAC B 16x8 local PSNR_Y {psnr_local:.2} dB below floor 38.0",
    );

    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p.annex_b);
    combined.extend_from_slice(&b.annex_b);
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), combined.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");

    let mut frames = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => frames.push(vf),
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    assert_eq!(frames.len(), 3, "expected 3 decoded frames");

    let b_decoded = &frames[1];
    let mut max_diff_y = 0i32;
    for (k, (&a, &local)) in b_decoded.planes[0]
        .data
        .iter()
        .zip(b.recon_y.iter())
        .enumerate()
    {
        let d = (a as i32 - local as i32).abs();
        if d > max_diff_y {
            max_diff_y = d;
        }
        assert!(
            d <= 1,
            "CABAC B 16x8 luma mismatch at idx {k}: dec={a}, enc={local} (diff {d})",
        );
    }
    let psnr_dec = psnr(&y2, &b_decoded.planes[0].data);
    eprintln!(
        "round-475 CABAC B 16x8: decoder PSNR_Y = {:.2} dB (max enc/dec diff {max_diff_y})",
        psnr_dec,
    );
    assert!(
        psnr_dec >= 38.0,
        "round-475 CABAC B 16x8 decoder PSNR_Y {psnr_dec:.2} dB below floor 38.0",
    );
}

#[test]
fn round475_b_8x16_cabac_self_roundtrip_matches_local_recon() {
    let (y0, u0, v0) = make_idr();
    let (y1, u1, v1) = make_p();
    let (y2, u2, v2) = make_b_per_vhalf_motion(&y0, &y1);

    let f0 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let f1 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y1,
        u: &u1,
        v: &v1,
    };
    let f2 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y2,
        u: &u2,
        v: &v2,
    };

    let (idr, p, b) = encode_ipb_cabac(&f0, &f1, &f2);

    let psnr_local = psnr(&y2, &b.recon_y);
    eprintln!(
        "round-475 CABAC B 8x16: stream={} bytes, local PSNR_Y = {:.2} dB",
        idr.annex_b.len() + p.annex_b.len() + b.annex_b.len(),
        psnr_local,
    );
    assert!(
        psnr_local >= 38.0,
        "round-475 CABAC B 8x16 local PSNR_Y {psnr_local:.2} dB below floor 38.0",
    );

    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p.annex_b);
    combined.extend_from_slice(&b.annex_b);
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), combined.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");

    let mut frames = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => frames.push(vf),
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    assert_eq!(frames.len(), 3);

    let b_decoded = &frames[1];
    let mut max_diff = 0i32;
    for (k, (&a, &local)) in b_decoded.planes[0]
        .data
        .iter()
        .zip(b.recon_y.iter())
        .enumerate()
    {
        let d = (a as i32 - local as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d <= 1,
            "CABAC B 8x16 luma mismatch at idx {k}: dec={a}, enc={local} (diff {d})",
        );
    }
    eprintln!(
        "round-475 CABAC B 8x16: decoder match (max enc/dec diff {max_diff}, PSNR_Y = {:.2} dB)",
        psnr(&y2, &b_decoded.planes[0].data)
    );
}

#[test]
fn round475_b_partitions_cabac_ffmpeg_interop() {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }

    let (y0, u0, v0) = make_idr();
    let (y1, u1, v1) = make_p();
    let (y2, u2, v2) = make_b_per_half_motion(&y0, &y1);

    let f0 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let f1 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y1,
        u: &u1,
        v: &v1,
    };
    let f2 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y2,
        u: &u2,
        v: &v2,
    };

    let (idr, p, b) = encode_ipb_cabac(&f0, &f1, &f2);

    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p.annex_b);
    combined.extend_from_slice(&b.annex_b);

    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!(
        "oxideav_h264_round475_b_cabac_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_round475_b_cabac_{}.yuv",
        std::process::id()
    ));
    std::fs::write(&h264_path, &combined).expect("write h264");

    let status = std::process::Command::new(ffmpeg)
        .arg("-loglevel")
        .arg("error")
        .arg("-y")
        .arg("-f")
        .arg("h264")
        .arg("-i")
        .arg(&h264_path)
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(&yuv_path)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed");
    let raw = std::fs::read(&yuv_path).expect("read yuv");
    let _ = std::fs::remove_file(&h264_path);
    let _ = std::fs::remove_file(&yuv_path);

    let y_len = W * H;
    let c_len = (W / 2) * (H / 2);
    let frame_len = y_len + 2 * c_len;
    assert_eq!(raw.len(), 3 * frame_len, "expected 3 frames from ffmpeg");

    // Display order: IDR (POC 0) → B (POC 2) → P (POC 4). frames[1] = B.
    let b_off = frame_len;
    let b_y = &raw[b_off..b_off + y_len];
    let mut max_diff = 0i32;
    for (k, (&a, &local)) in b_y.iter().zip(b.recon_y.iter()).enumerate() {
        let d = (a as i32 - local as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d <= 1,
            "ffmpeg CABAC B luma {k}: ff={a} enc={local} (diff {d})",
        );
    }
    let psnr_ff = psnr(&y2, b_y);
    eprintln!(
        "round-475 CABAC B partitions ffmpeg interop: bit-equivalent (max diff {max_diff}, stream {} bytes, PSNR_Y={:.2} dB)",
        combined.len(),
        psnr_ff,
    );
    assert!(
        psnr_ff >= 38.0,
        "round-475 CABAC B partitions ffmpeg PSNR_Y {psnr_ff:.2} dB below floor 38.0",
    );
}

// ============================================================================
// Round-475 — B_8x8 + 4× B_Direct_8x8 CABAC test (mb_type=22, sub_mb_type=0).
// ============================================================================

const B8_W: usize = 32;
const B8_H: usize = 16;

fn b8_make_idr() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; B8_W * B8_H];
    for j in 0..B8_H {
        for i in 0..B8_W {
            y[j * B8_W + i] = (60 + i + j) as u8;
        }
    }
    let u = vec![128u8; (B8_W / 2) * (B8_H / 2)];
    let v = vec![128u8; (B8_W / 2) * (B8_H / 2)];
    (y, u, v)
}

fn b8_make_p() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; B8_W * B8_H];
    for j in 0..B8_H {
        for i in 0..B8_W {
            let i_src = i.saturating_sub(4);
            y[j * B8_W + i] = (60 + i_src + j) as u8;
        }
    }
    let u = vec![128u8; (B8_W / 2) * (B8_H / 2)];
    let v = vec![128u8; (B8_W / 2) * (B8_H / 2)];
    (y, u, v)
}

fn b8_make_b(idr_y: &[u8], p_y: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; B8_W * B8_H];
    for j in 0..B8_H {
        for i in 0..B8_W {
            y[j * B8_W + i] =
                (idr_y[j * B8_W + i] as u32 + p_y[j * B8_W + i] as u32).div_ceil(2) as u8;
        }
    }
    let u = vec![128u8; (B8_W / 2) * (B8_H / 2)];
    let v = vec![128u8; (B8_W / 2) * (B8_H / 2)];
    (y, u, v)
}

/// CABAC `B_8x8` + 4× `B_Direct_8x8` smoke test. The fixture is the
/// same shape as the round-23 CAVLC `B_8x8` direct test — bipred-on-
/// midpoint between an IDR and a 4-px-shifted P. The §8.4.1.2.2
/// derivation produces per-8x8 MVs (the L1 anchor's colocated MV is
/// non-zero, driving step-7 colZeroFlag = false on every quadrant), so
/// the encoder picks the new B_8x8 path over uniform B_Direct_16x16.
///
/// Acceptance: self-roundtrip ≤ 1 LSB on the B-frame's local recon.
#[test]
fn round475_b_8x8_all_direct_cabac_self_roundtrip() {
    let (y0, u0, v0) = b8_make_idr();
    let (y1, u1, v1) = b8_make_p();
    let (y2, u2, v2) = b8_make_b(&y0, &y1);

    let mut cfg = EncoderConfig::new(B8_W as u32, B8_H as u32);
    cfg.cabac = true;
    cfg.profile_idc = 77;
    cfg.max_num_ref_frames = 2;
    cfg.qp = 26;
    let enc = Encoder::new(cfg);

    let f0 = YuvFrame {
        width: B8_W as u32,
        height: B8_H as u32,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let f1 = YuvFrame {
        width: B8_W as u32,
        height: B8_H as u32,
        y: &y1,
        u: &u1,
        v: &v1,
    };
    let f2 = YuvFrame {
        width: B8_W as u32,
        height: B8_H as u32,
        y: &y2,
        u: &u2,
        v: &v2,
    };

    let idr = enc.encode_idr_cabac(&f0);
    let p = enc.encode_p_cabac(&f1, &EncodedFrameRef::from(&idr), 1, 4);
    let b = enc.encode_b_cabac(
        &f2,
        &EncodedFrameRef::from(&idr),
        &EncodedFrameRef::from(&p),
        1,
        2,
    );

    let psnr_local = psnr(&y2, &b.recon_y);
    eprintln!(
        "round-475 CABAC B 8x8 all-Direct: stream={} bytes (B {}), local PSNR_Y = {:.2} dB",
        idr.annex_b.len() + p.annex_b.len() + b.annex_b.len(),
        b.annex_b.len(),
        psnr_local,
    );
    assert!(
        psnr_local >= 38.0,
        "round-475 CABAC B 8x8 local PSNR_Y {psnr_local:.2} dB below floor 38.0",
    );

    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p.annex_b);
    combined.extend_from_slice(&b.annex_b);
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), combined.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");

    let mut frames = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => frames.push(vf),
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    assert_eq!(frames.len(), 3);

    let b_decoded = &frames[1];
    let mut max_diff = 0i32;
    for (k, (&a, &local)) in b_decoded.planes[0]
        .data
        .iter()
        .zip(b.recon_y.iter())
        .enumerate()
    {
        let d = (a as i32 - local as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d <= 1,
            "CABAC B 8x8 luma mismatch at idx {k}: dec={a}, enc={local} (diff {d})",
        );
    }
    eprintln!("round-475 CABAC B 8x8: decoder match (max enc/dec diff {max_diff})",);
}

#[test]
fn round475_b_8x8_all_direct_cabac_ffmpeg_interop() {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }

    let (y0, u0, v0) = b8_make_idr();
    let (y1, u1, v1) = b8_make_p();
    let (y2, u2, v2) = b8_make_b(&y0, &y1);

    let mut cfg = EncoderConfig::new(B8_W as u32, B8_H as u32);
    cfg.cabac = true;
    cfg.profile_idc = 77;
    cfg.max_num_ref_frames = 2;
    cfg.qp = 26;
    let enc = Encoder::new(cfg);

    let f0 = YuvFrame {
        width: B8_W as u32,
        height: B8_H as u32,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let f1 = YuvFrame {
        width: B8_W as u32,
        height: B8_H as u32,
        y: &y1,
        u: &u1,
        v: &v1,
    };
    let f2 = YuvFrame {
        width: B8_W as u32,
        height: B8_H as u32,
        y: &y2,
        u: &u2,
        v: &v2,
    };

    let idr = enc.encode_idr_cabac(&f0);
    let p = enc.encode_p_cabac(&f1, &EncodedFrameRef::from(&idr), 1, 4);
    let b = enc.encode_b_cabac(
        &f2,
        &EncodedFrameRef::from(&idr),
        &EncodedFrameRef::from(&p),
        1,
        2,
    );

    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p.annex_b);
    combined.extend_from_slice(&b.annex_b);

    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!(
        "oxideav_h264_round475_b8x8_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_round475_b8x8_{}.yuv",
        std::process::id()
    ));
    std::fs::write(&h264_path, &combined).expect("write h264");

    let status = std::process::Command::new(ffmpeg)
        .arg("-loglevel")
        .arg("error")
        .arg("-y")
        .arg("-f")
        .arg("h264")
        .arg("-i")
        .arg(&h264_path)
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(&yuv_path)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed");
    let raw = std::fs::read(&yuv_path).expect("read yuv");
    let _ = std::fs::remove_file(&h264_path);
    let _ = std::fs::remove_file(&yuv_path);

    let y_len = B8_W * B8_H;
    let c_len = (B8_W / 2) * (B8_H / 2);
    let frame_len = y_len + 2 * c_len;
    assert_eq!(raw.len(), 3 * frame_len, "expected 3 frames from ffmpeg");

    let b_off = frame_len;
    let b_y = &raw[b_off..b_off + y_len];
    let mut max_diff = 0i32;
    for (k, (&a, &local)) in b_y.iter().zip(b.recon_y.iter()).enumerate() {
        let d = (a as i32 - local as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d <= 1,
            "ffmpeg CABAC B 8x8 luma {k}: ff={a} enc={local} (diff {d})",
        );
    }
    let psnr_ff = psnr(&y2, b_y);
    eprintln!(
        "round-475 CABAC B 8x8 ffmpeg: bit-equivalent (max diff {max_diff}, stream {} bytes, PSNR_Y={:.2} dB)",
        combined.len(),
        psnr_ff,
    );
    assert!(
        psnr_ff >= 38.0,
        "round-475 CABAC B 8x8 ffmpeg PSNR_Y {psnr_ff:.2} dB below floor 38.0",
    );
}
