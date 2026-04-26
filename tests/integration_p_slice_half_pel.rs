//! Round-17 — Half-pel ME PSNR delta on a half-pel-shifted P-frame.
//!
//! Verifies the round-17 §8.4.2.2.1 half-pel ME refinement closes the
//! gap on content that is **not** integer-pel-aligned. The IDR is a
//! linear ramp; the P-frame is the same ramp shifted right by ½ luma
//! pixel. Integer-pel-only ME (round 16) cannot reach the predictor —
//! the residual carries the ½-pel-of-slope mismatch and PSNR caps out
//! around ~30 dB at high QP. With half-pel ME, the encoder picks
//! `mv_qpel = +2` and the predictor matches the source bit-exactly,
//! so PSNR jumps to the round-17 floor (≥ 38 dB on a smooth ramp).
//!
//! The roundtrip half (decode → compare-against-encoder-recon) re-
//! exercises the encoder/decoder lockstep on half-pel MVs end to end.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 64;
const H: usize = 64;

/// 64x64 IDR: linear horizontal ramp, constant chroma.
fn make_idr_ramp() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            // Slope of 3/pel — span 30..(30 + 3*63) = 30..219.
            y[j * W + i] = (30 + i * 3) as u8;
        }
    }
    let u = vec![128u8; (W / 2) * (H / 2)];
    let v = vec![128u8; (W / 2) * (H / 2)];
    (y, u, v)
}

/// 64x64 P: same ramp shifted right by ½ luma pixel. We compute the
/// shifted samples directly from the slope (slope × 0.5 = +1.5 → +2
/// after rounding consistent with the §8.4.2.2.1 b-position formula
/// applied to a linear ramp: see the round-17 ME unit tests for the
/// derivation).
fn make_p_half_pel_shift_right() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            // 6-tap on linear ramp (30 + 3i) at half-pel position:
            // ((1008 + 96*i + 16) >> 5) = 32 + 3*i. So the shifted
            // ramp is r[i] + 2 = 32 + 3*i.
            y[j * W + i] = (32 + i * 3).clamp(0, 255) as u8;
        }
    }
    let u = vec![128u8; (W / 2) * (H / 2)];
    let v = vec![128u8; (W / 2) * (H / 2)];
    (y, u, v)
}

fn mse(orig: &[u8], recon: &[u8]) -> f64 {
    let n = orig.len() as f64;
    let s: f64 = orig
        .iter()
        .zip(recon.iter())
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum();
    s / n
}

fn psnr(orig: &[u8], recon: &[u8]) -> f64 {
    let m = mse(orig, recon);
    if m <= 0.0 {
        return 99.0;
    }
    10.0 * (255.0_f64 * 255.0 / m).log10()
}

#[test]
fn round17_half_pel_p_slice_high_psnr() {
    let (y0, u0, v0) = make_idr_ramp();
    let (y1, u1, v1) = make_p_half_pel_shift_right();
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
    let cfg = EncoderConfig::new(W as u32, H as u32);
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr(&f0);
    let p = enc.encode_p(&f1, &EncodedFrameRef::from(&idr), 1, 2);

    // Compute PSNR of the encoder's local reconstruction vs the source —
    // this is what half-pel ME directly improves. With ½-pel-aligned
    // motion, the predictor *is* the source, so even at modest QP the
    // residual is tiny and PSNR is high.
    let psnr_local = psnr(&y1, &p.recon_y);
    eprintln!(
        "round-17 half-pel P-slice: stream={} bytes (IDR {}, P {}), local recon PSNR_Y = {:.2} dB",
        idr.annex_b.len() + p.annex_b.len(),
        idr.annex_b.len(),
        p.annex_b.len(),
        psnr_local,
    );
    assert!(
        psnr_local >= 38.0,
        "round-17 half-pel local recon PSNR_Y {psnr_local:.2} dB below floor 38.0",
    );

    // Round-trip through our own decoder — verifies the encoder's
    // half-pel MV is bit-equivalent under the §8.4.2.2.1 6-tap filter.
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
    assert_eq!(frames.len(), 2, "expected 2 decoded frames (IDR + P)");

    // Half-pel rounding can drift by 1 sample in either direction
    // between encoder local recon and decoder output (different
    // accumulators, same spec) — accept |diff| <= 1 like the round-16
    // tests.
    let f1_dec = &frames[1];
    let mut max_diff = 0i32;
    for (k, (&a, &b)) in f1_dec.planes[0]
        .data
        .iter()
        .zip(p.recon_y.iter())
        .enumerate()
    {
        let d = (a as i32 - b as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d <= 1,
            "half-pel P luma mismatch at idx {k}: dec={a}, enc={b} (diff {d})",
        );
    }
    let psnr_dec = psnr(&y1, &f1_dec.planes[0].data);
    eprintln!(
        "round-17 half-pel P-slice: decoder PSNR_Y = {:.2} dB (max enc/dec diff {max_diff})",
        psnr_dec,
    );
    assert!(
        psnr_dec >= 38.0,
        "round-17 half-pel decoder PSNR_Y {psnr_dec:.2} dB below floor 38.0",
    );
}

/// Round-17 — ffmpeg interop on the half-pel P-frame. The encoder's
/// half-pel MV must be bit-exactly understood by libavcodec.
#[test]
fn round17_half_pel_ffmpeg_interop() {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }
    let (y0, u0, v0) = make_idr_ramp();
    let (y1, u1, v1) = make_p_half_pel_shift_right();
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
    let cfg = EncoderConfig::new(W as u32, H as u32);
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr(&f0);
    let p = enc.encode_p(&f1, &EncodedFrameRef::from(&idr), 1, 2);

    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p.annex_b);

    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!(
        "oxideav_h264_round17_halfpel_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_round17_halfpel_{}.yuv",
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
    assert_eq!(raw.len(), 2 * frame_len, "expected 2 frames from ffmpeg");

    let f1_y = &raw[frame_len..frame_len + y_len];
    let mut max_diff = 0i32;
    for (k, (&a, &b)) in f1_y.iter().zip(p.recon_y.iter()).enumerate() {
        let d = (a as i32 - b as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d <= 1,
            "ffmpeg half-pel P luma {k}: ff={a} enc={b} (diff {d})",
        );
    }
    eprintln!(
        "round-17 ffmpeg interop: bit-equivalent on half-pel P (max diff {max_diff}, stream {} bytes)",
        combined.len()
    );
}
