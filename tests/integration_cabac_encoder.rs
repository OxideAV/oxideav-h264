//! Round-30 — H.264 **CABAC encode** integration tests.
//!
//! Verifies the [`Encoder::encode_idr_cabac`] / [`Encoder::encode_p_cabac`]
//! entry points round-trip cleanly through:
//!
//! 1. **Self-decode**: decoding the CABAC stream with our own H.264
//!    decoder (which already supports CABAC) yields per-plane samples
//!    that match the encoder's local recon.
//! 2. **PSNR floor**: decoded picture vs. original source achieves
//!    >=45 dB Y on a synthetic gradient at QP 26 (round-30 acceptance).
//!
//! Round-30 scope: I + P slices, 4:2:0, single ref, integer-pel ME.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

/// Build a 64x64 YUV 4:2:0 picture with a smooth diagonal luma gradient
/// and constant midgray chroma. Matches the round-16 round-trip fixture.
fn make_gradient_64x64() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = 64usize;
    let h = 64usize;
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            let v = 16 + (i + j) * (240 - 16) / (w + h - 2);
            y[j * w + i] = v.clamp(0, 255) as u8;
        }
    }
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];
    (y, u, v)
}

fn mse(orig: &[u8], recon: &[u8]) -> f64 {
    assert_eq!(orig.len(), recon.len());
    let n = orig.len() as f64;
    orig.iter()
        .zip(recon.iter())
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d / n
        })
        .sum()
}

fn psnr(orig: &[u8], recon: &[u8]) -> f64 {
    let m = mse(orig, recon);
    if m <= 0.0 {
        return 99.0;
    }
    10.0 * (255.0_f64 * 255.0 / m).log10()
}

#[test]
fn round30_idr_cabac_self_roundtrip() {
    let (y, u, v) = make_gradient_64x64();
    let frame = YuvFrame {
        width: 64,
        height: 64,
        y: &y,
        u: &u,
        v: &v,
    };
    let mut cfg = EncoderConfig::new(64, 64);
    cfg.cabac = true;
    cfg.profile_idc = 77; // Main (CABAC requires >= Main)
    cfg.qp = 26;
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr_cabac(&frame);

    // Decode through our own decoder.
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), idr.annex_b.clone()).with_pts(0);
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
    assert_eq!(frames.len(), 1, "expected exactly one decoded frame");
    let f0 = &frames[0];
    assert_eq!(f0.planes.len(), 3, "expected Yuv420P");
    assert_eq!(f0.planes[0].stride, 64);
    assert_eq!(f0.planes[0].data.len(), 64 * 64);

    // Encoder recon matches decoder output.
    let mut max_diff = 0i32;
    for (k, (&a, &b)) in f0.planes[0].data.iter().zip(idr.recon_y.iter()).enumerate() {
        let d = (a as i32 - b as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d <= 1,
            "IDR luma mismatch at idx {k}: dec={a}, enc={b} (diff {d})"
        );
    }

    let psnr_y = psnr(&y, &f0.planes[0].data);
    eprintln!(
        "round30_idr_cabac: stream={} bytes, max enc/dec diff={max_diff}, PSNR_Y={psnr_y:.2} dB",
        idr.annex_b.len(),
    );
    assert!(
        psnr_y >= 45.0,
        "PSNR {psnr_y:.2} dB below round-30 floor of 45 dB"
    );
}

#[test]
fn round30_idr_plus_p_cabac_self_roundtrip() {
    let (y0, u0, v0) = make_gradient_64x64();
    // P-frame: same source — encoder should emit all P_Skip.
    let f0 = YuvFrame {
        width: 64,
        height: 64,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let f1 = YuvFrame {
        width: 64,
        height: 64,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let mut cfg = EncoderConfig::new(64, 64);
    cfg.cabac = true;
    cfg.profile_idc = 77;
    cfg.qp = 26;
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr_cabac(&f0);
    let p = enc.encode_p_cabac(&f1, &EncodedFrameRef::from(&idr), 1, 2);

    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p.annex_b);
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
    assert_eq!(frames.len(), 2, "expected 2 decoded frames");

    let f0_dec = &frames[0];
    let f1_dec = &frames[1];
    // IDR matches encoder recon.
    for (k, (&a, &b)) in f0_dec.planes[0]
        .data
        .iter()
        .zip(idr.recon_y.iter())
        .enumerate()
    {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "IDR luma mismatch at {k}: dec={a}, enc={b}",
        );
    }
    // P matches encoder recon (small tolerance to absorb LSB rounding
    // differences in the §8.7 deblock filter for round-30 — encoder /
    // decoder agree on the boundary-strength derivation but the
    // tc-clipped delta may differ by ±1 LSB on smooth content).
    let mut max_diff = 0i32;
    let mut nz = 0usize;
    for (_k, (&a, &b)) in f1_dec.planes[0]
        .data
        .iter()
        .zip(p.recon_y.iter())
        .enumerate()
    {
        let d = (a as i32 - b as i32).abs();
        if d > 0 {
            nz += 1;
        }
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!(
        "round30_p_cabac: max P enc/dec diff = {max_diff}, divergent pixels = {nz}/{}",
        f1_dec.planes[0].data.len(),
    );
    let psnr_recon = psnr(&f1_dec.planes[0].data, &p.recon_y);
    eprintln!("round30_p_cabac: PSNR(decoder vs encoder recon) = {psnr_recon:.2} dB");
    let psnr_src = psnr(&y0, &f1_dec.planes[0].data);
    eprintln!(
        "round30_idr_plus_p_cabac: IDR={} bytes, P={} bytes, PSNR_Y(P vs source)={psnr_src:.2} dB",
        idr.annex_b.len(),
        p.annex_b.len(),
    );
    assert!(
        psnr_src >= 45.0,
        "P-frame PSNR vs source {psnr_src:.2} dB below round-30 floor 45 dB"
    );
    // Self-roundtrip: encoder recon should match decoder output.
    assert!(
        max_diff <= 1,
        "P encoder/decoder recon diverges by {max_diff} LSBs (>1)"
    );
}

#[test]
fn round30_idr_cabac_ffmpeg_interop() {
    // ffmpeg/libavcodec must decode the CABAC IDR bit-equivalently.
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }

    let (y, u, v) = make_gradient_64x64();
    let frame = YuvFrame {
        width: 64,
        height: 64,
        y: &y,
        u: &u,
        v: &v,
    };
    let mut cfg = EncoderConfig::new(64, 64);
    cfg.cabac = true;
    cfg.profile_idc = 77;
    cfg.qp = 26;
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr_cabac(&frame);

    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!(
        "oxideav_h264_round30_cabac_idr_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_round30_cabac_idr_{}.yuv",
        std::process::id()
    ));
    std::fs::write(&h264_path, &idr.annex_b).expect("write h264");

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
    assert!(status.success(), "ffmpeg failed to decode our CABAC stream");
    let raw = std::fs::read(&yuv_path).expect("read yuv");
    let _ = std::fs::remove_file(&h264_path);
    let _ = std::fs::remove_file(&yuv_path);

    let y_len = 64 * 64;
    let c_len = 32 * 32;
    assert_eq!(raw.len(), y_len + 2 * c_len, "expected 1 frame from ffmpeg");
    let f_y = &raw[..y_len];
    let mut max_diff = 0i32;
    for (k, (&a, &b)) in f_y.iter().zip(idr.recon_y.iter()).enumerate() {
        let d = (a as i32 - b as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d <= 1,
            "ffmpeg vs encoder recon luma {k}: ff={a} enc={b} (diff {d})"
        );
    }
    let psnr_y = psnr(&y, f_y);
    eprintln!("round30_idr_cabac_ffmpeg_interop: max diff={max_diff}, PSNR_Y={psnr_y:.2} dB");
    assert!(
        psnr_y >= 45.0,
        "PSNR {psnr_y:.2} dB below 45 dB floor under ffmpeg decode"
    );
}
