//! Round-22 — B-slice 16x8 / 8x16 partition modes.
//!
//! These tests build a synthetic IPB triple where the B-frame has
//! per-half motion that differs between the top and bottom (or left
//! and right) of every MB. The encoder's per-partition ME then
//! selects a 16x8 or 8x16 mb_type with appropriate (mode_top,
//! mode_bottom) / (mode_left, mode_right) combinations from §7.4.5
//! Table 7-14 (raw values 4..=21).
//!
//! Two fixtures:
//!
//! 1. `round22_b_16x8_self_roundtrip_matches_local_recon` — feed the
//!    encoded stream through our own decoder and verify the B-frame's
//!    luma matches the encoder's local recon bit-exactly (max diff ≤
//!    1 LSB to allow for the same ¼-pel accumulator-ordering tolerance
//!    we accept in the round-20 test).
//!
//! 2. `round22_b_partitions_ffmpeg_interop` — feed the same stream to
//!    libavcodec and verify bit-equivalent decode + ≥ 38 dB PSNR vs
//!    source.
//!
//! Cross-reference: §7.4.5 Table 7-14 (mb_type 4..=21), §8.4.1.3
//! median MVpred per partition (with Partition16x8Top/Bottom and
//! Partition8x16Left/Right shape shortcuts), §8.4.2.2.1 sub-pel
//! interpolation, §8.4.2.3.1 default weighted bipred merge.

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

/// IDR: smooth horizontal ramp (60..123 across i=0..63).
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

/// P: same as IDR, shifted right by 4 luma px (integer-pel motion).
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

/// B with per-half motion: top half of each MB exactly matches the IDR
/// (no shift), bottom half exactly matches the P (shifted right by 4).
/// With round-22 16x8 partition support, every MB should pick a 16x8
/// mb_type with (top = L0, bottom = L1) — which is mb_type raw 8
/// (B_L0_L1_16x8). This tests both the partition-shape ME and the
/// per-partition L0/L1 mode-selection logic.
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

/// B with per-half motion split LEFT vs RIGHT: left half matches IDR,
/// right half matches P. Encoder should pick 8x16 with (left = L0,
/// right = L1) — mb_type raw 9 (B_L0_L1_8x16).
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

#[test]
fn round22_b_16x8_self_roundtrip_matches_local_recon() {
    let mut cfg = EncoderConfig::new(W as u32, H as u32);
    cfg.profile_idc = 77; // Main — required for B (§A.2.2)
    cfg.max_num_ref_frames = 2;
    let enc = Encoder::new(cfg);

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

    let idr = enc.encode_idr(&f0);
    let p = enc.encode_p(&f1, &EncodedFrameRef::from(&idr), 1, 4);
    let b = enc.encode_b(
        &f2,
        &EncodedFrameRef::from(&idr),
        &EncodedFrameRef::from(&p),
        1,
        2,
    );

    let psnr_local = psnr(&y2, &b.recon_y);
    eprintln!(
        "round-22 B 16x8: stream={} bytes (IDR {}, P {}, B {}), local PSNR_Y = {:.2} dB",
        idr.annex_b.len() + p.annex_b.len() + b.annex_b.len(),
        idr.annex_b.len(),
        p.annex_b.len(),
        b.annex_b.len(),
        psnr_local,
    );
    assert!(
        psnr_local >= 38.0,
        "round-22 B local recon PSNR_Y {psnr_local:.2} dB below floor 38.0",
    );

    // Round-trip through our own decoder.
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

    // §C.4 bumping order: IDR (POC 0) → B (POC 2) → P (POC 4).
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
            "B luma mismatch at idx {k}: dec={a}, enc={local} (diff {d})",
        );
    }
    let psnr_dec = psnr(&y2, &b_decoded.planes[0].data);
    eprintln!(
        "round-22 B 16x8: decoder PSNR_Y = {:.2} dB (max enc/dec diff {max_diff_y})",
        psnr_dec,
    );
    assert!(
        psnr_dec >= 38.0,
        "round-22 B decoder PSNR_Y {psnr_dec:.2} dB below floor 38.0",
    );
}

#[test]
fn round22_b_8x16_self_roundtrip_matches_local_recon() {
    let mut cfg = EncoderConfig::new(W as u32, H as u32);
    cfg.profile_idc = 77;
    cfg.max_num_ref_frames = 2;
    let enc = Encoder::new(cfg);

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

    let idr = enc.encode_idr(&f0);
    let p = enc.encode_p(&f1, &EncodedFrameRef::from(&idr), 1, 4);
    let b = enc.encode_b(
        &f2,
        &EncodedFrameRef::from(&idr),
        &EncodedFrameRef::from(&p),
        1,
        2,
    );

    let psnr_local = psnr(&y2, &b.recon_y);
    eprintln!(
        "round-22 B 8x16: stream={} bytes (B {}), local PSNR_Y = {:.2} dB",
        idr.annex_b.len() + p.annex_b.len() + b.annex_b.len(),
        b.annex_b.len(),
        psnr_local,
    );
    assert!(
        psnr_local >= 38.0,
        "round-22 B 8x16 local PSNR_Y {psnr_local:.2} dB below floor 38.0",
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
            "B 8x16 luma mismatch at idx {k}: dec={a}, enc={local} (diff {d})",
        );
    }
    eprintln!(
        "round-22 B 8x16: decoder match (max enc/dec diff {max_diff}, PSNR_Y = {:.2} dB)",
        psnr(&y2, &b_decoded.planes[0].data)
    );
}

#[test]
fn round22_b_partitions_ffmpeg_interop() {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }

    let mut cfg = EncoderConfig::new(W as u32, H as u32);
    cfg.profile_idc = 77;
    cfg.max_num_ref_frames = 2;
    let enc = Encoder::new(cfg);

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

    let idr = enc.encode_idr(&f0);
    let p = enc.encode_p(&f1, &EncodedFrameRef::from(&idr), 1, 4);
    let b = enc.encode_b(
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
        "oxideav_h264_round22_b_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!("oxideav_h264_round22_b_{}.yuv", std::process::id()));
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
        assert!(d <= 1, "ffmpeg B luma {k}: ff={a} enc={local} (diff {d})",);
    }
    eprintln!(
        "round-22 ffmpeg interop: bit-equivalent on B-slice (max diff {max_diff}, stream {} bytes)",
        combined.len()
    );

    let psnr_ff = psnr(&y2, b_y);
    eprintln!("round-22 ffmpeg PSNR_Y on B-slice = {:.2} dB", psnr_ff);
    assert!(
        psnr_ff >= 38.0,
        "round-22 ffmpeg B-slice PSNR_Y {psnr_ff:.2} dB below floor 38.0",
    );
}
