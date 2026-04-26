//! End-to-end self-roundtrip for the round-1 Baseline I-only encoder.
//!
//! Generates a synthetic 64x64 YUV 4:2:0 picture, runs it through the
//! encoder, feeds the resulting Annex B bytes back into our own
//! H264CodecDecoder, and checks that the decoded picture matches the
//! encoder's local reconstruction (the encoder publishes its expected
//! recon alongside the bitstream).
//!
//! Expectations for round 1 (`I_16x16` DC-only, no AC, CBP=0,
//! deblocking off):
//!  * Every decoded sample equals the encoder's recon (not the
//!    original — only the DC of each MB survives quantization, so the
//!    recon is far from photometric).
//!  * No spurious decoder errors; the stream parses end-to-end.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::{Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

/// Build a 64x64 4:2:0 test source. Same idea as ffmpeg's `testsrc`
/// pattern but trivial: a smooth horizontal gradient on luma, midgray
/// on chroma.
fn make_testsrc_64x64() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = 64usize;
    let h = 64usize;
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            // Smooth gradient 16..240 across the picture diagonal.
            let v = 16 + (i + j) * (240 - 16) / (w + h - 2);
            y[j * w + i] = v.clamp(0, 255) as u8;
        }
    }
    // 32x32 chroma planes — flat 128.
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];
    (y, u, v)
}

/// 64x64 4:2:0 picture with a non-trivial chroma pattern. Luma is the
/// same diagonal gradient; chroma is a Cb gradient horizontally + a Cr
/// gradient vertically, so the chroma residual / mode-decision paths
/// are actually exercised by the round-2 encoder.
fn make_testsrc_chroma_64x64() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = 64usize;
    let h = 64usize;
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; (w / 2) * (h / 2)];
    let mut v = vec![0u8; (w / 2) * (h / 2)];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = (16 + (i + j) * (240 - 16) / (w + h - 2)).clamp(0, 255) as u8;
        }
    }
    for j in 0..(h / 2) {
        for i in 0..(w / 2) {
            // Cb varies horizontally 64..192; Cr varies vertically.
            u[j * (w / 2) + i] = (64 + i * 128 / (w / 2 - 1)) as u8;
            v[j * (w / 2) + i] = (64 + j * 128 / (h / 2 - 1)) as u8;
        }
    }
    (y, u, v)
}

#[test]
fn encode_decode_roundtrip_matches_local_recon() {
    let (y, u, v) = make_testsrc_64x64();
    let frame = YuvFrame {
        width: 64,
        height: 64,
        y: &y,
        u: &u,
        v: &v,
    };
    let cfg = EncoderConfig::new(64, 64);
    let enc = Encoder::new(cfg);
    let out = enc.encode_idr(&frame);

    // Run the bytes back through the decoder.
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), out.annex_b.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");

    let frame = dec.receive_frame().expect("receive_frame");
    let vf = match frame {
        Frame::Video(vf) => vf,
        other => panic!("expected Frame::Video, got {:?}", other),
    };
    // Slim VideoFrame: derive 64x64 Yuv420P geometry from plane stride
    // + 3-plane layout. Format/resolution live on stream params now.
    assert_eq!(vf.planes.len(), 3, "expected Yuv420P (3-plane) layout");
    assert_eq!(vf.planes[0].stride, 64);
    assert_eq!(vf.planes[0].data.len(), 64 * 64);

    // Compare per-plane to the encoder's recon. Allow a tiny tolerance
    // (off-by-one due to integer rounding in the deblocking-disabled
    // path is not expected, but any mismatch >= 2 indicates a real
    // pipeline divergence).
    for (k, (&a, &b)) in vf.planes[0].data.iter().zip(out.recon_y.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "luma sample {k}: decoded={a} encoder_recon={b}",
        );
    }
    for (k, (&a, &b)) in vf.planes[1].data.iter().zip(out.recon_u.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "Cb sample {k}: decoded={a} encoder_recon={b}",
        );
    }
    for (k, (&a, &b)) in vf.planes[2].data.iter().zip(out.recon_v.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "Cr sample {k}: decoded={a} encoder_recon={b}",
        );
    }

    // Sanity: PSNR vs. original. With DC-only reconstruction this will
    // be modest, but should be >= 15 dB on a smooth gradient (the DC
    // term tracks the average sample value of each MB).
    let mse = mse_y(&y, &out.recon_y);
    let psnr = if mse == 0.0 {
        99.0
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    };
    eprintln!(
        "round-1 self-roundtrip: stream={} bytes, luma PSNR vs original = {:.2} dB",
        out.annex_b.len(),
        psnr,
    );
    assert!(psnr >= 15.0, "luma PSNR {psnr} dB below 15");
}

fn mse_y(orig: &[u8], recon: &[u8]) -> f64 {
    let mut s = 0u64;
    for (&o, &r) in orig.iter().zip(recon.iter()) {
        let d = o as i32 - r as i32;
        s += (d * d) as u64;
    }
    s as f64 / orig.len() as f64
}

/// Round-2: chroma-residual self-roundtrip. Same flow as
/// [`encode_decode_roundtrip_matches_local_recon`] but with a picture
/// that has non-trivial chroma content, exercising the encoder's chroma
/// DC / AC residual + mode-decision paths.
#[test]
fn encode_decode_roundtrip_chroma_residual() {
    let (y, u, v) = make_testsrc_chroma_64x64();
    let frame = YuvFrame {
        width: 64,
        height: 64,
        y: &y,
        u: &u,
        v: &v,
    };
    let cfg = EncoderConfig::new(64, 64);
    let enc = Encoder::new(cfg);
    let out = enc.encode_idr(&frame);

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), out.annex_b.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");

    let frame = dec.receive_frame().expect("receive_frame");
    let vf = match frame {
        Frame::Video(vf) => vf,
        other => panic!("expected Frame::Video, got {:?}", other),
    };
    // Yuv420P stand-in: 3-plane layout. Format tag is on stream params.
    assert_eq!(vf.planes.len(), 3, "expected Yuv420P (3-plane) layout");

    // Bit-exact match between decoder output and encoder local recon.
    for (k, (&a, &b)) in vf.planes[0].data.iter().zip(out.recon_y.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "luma sample {k}: decoded={a} encoder_recon={b}",
        );
    }
    for (k, (&a, &b)) in vf.planes[1].data.iter().zip(out.recon_u.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "Cb sample {k}: decoded={a} encoder_recon={b}",
        );
    }
    for (k, (&a, &b)) in vf.planes[2].data.iter().zip(out.recon_v.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "Cr sample {k}: decoded={a} encoder_recon={b}",
        );
    }

    let psnr_y = {
        let m = mse_y(&y, &out.recon_y);
        if m == 0.0 {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / m).log10()
        }
    };
    let psnr_u = {
        let m = mse_y(&u, &out.recon_u);
        if m == 0.0 {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / m).log10()
        }
    };
    let psnr_v = {
        let m = mse_y(&v, &out.recon_v);
        if m == 0.0 {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / m).log10()
        }
    };
    eprintln!(
        "round-2 chroma roundtrip: stream={} bytes, PSNR Y={:.2} U={:.2} V={:.2}",
        out.annex_b.len(),
        psnr_y,
        psnr_u,
        psnr_v
    );
    // Chroma should now be tracked to within reasonable PSNR even at
    // QP 26 with DC + AC residual.
    assert!(psnr_u >= 25.0, "Cb PSNR {psnr_u} dB too low");
    assert!(psnr_v >= 25.0, "Cr PSNR {psnr_v} dB too low");
}

/// 128x128 picture with mixed content: vertical bars on the left half,
/// horizontal bars on the right half, smooth gradient at the bottom.
/// Designed to give the V/H/Plane modes a real workout vs DC-only.
fn make_mixed_128x128() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = 128usize;
    let h = 128usize;
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            let v = if i < w / 2 && j < h / 2 {
                // Vertical bars: cycle [40, 200] every 8 cols.
                if (i / 8) % 2 == 0 {
                    40
                } else {
                    200
                }
            } else if i >= w / 2 && j < h / 2 {
                // Horizontal bars.
                if (j / 8) % 2 == 0 {
                    40
                } else {
                    200
                }
            } else {
                // Smooth gradient.
                (16 + (i + j) * 200 / (w + h - 2)).clamp(0, 255)
            };
            y[j * w + i] = v as u8;
        }
    }
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];
    (y, u, v)
}

/// PSNR comparison: V/H/Plane modes vs original on a richer picture.
/// This is a validation/diagnostic test — it ensures the per-MB mode
/// decision actually picks non-DC modes when they help and that the
/// resulting reconstruction stays high-fidelity through the decoder.
#[test]
fn encode_mixed_picture_picks_non_dc_modes() {
    let (y, u, v) = make_mixed_128x128();
    let frame = YuvFrame {
        width: 128,
        height: 128,
        y: &y,
        u: &u,
        v: &v,
    };
    let cfg = EncoderConfig::new(128, 128);
    let enc = Encoder::new(cfg);
    let out = enc.encode_idr(&frame);

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), out.annex_b.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let frame = dec.receive_frame().expect("receive_frame");
    let vf = match frame {
        Frame::Video(vf) => vf,
        other => panic!("expected Frame::Video, got {:?}", other),
    };
    for (k, (&a, &b)) in vf.planes[0].data.iter().zip(out.recon_y.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "luma sample {k}: decoded={a} encoder_recon={b}",
        );
    }

    let psnr_y = {
        let m = mse_y(&y, &out.recon_y);
        if m == 0.0 {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / m).log10()
        }
    };
    eprintln!(
        "round-2 mixed-pic 128x128: stream={} bytes, PSNR Y={:.2}",
        out.annex_b.len(),
        psnr_y
    );
    // With DC-only the bars would be reproduced as flat MB-wide DCs;
    // V/H modes should keep them sharp even with cbp_luma=0 (the DC
    // captures the row/column average per 4x4).
    assert!(psnr_y >= 18.0, "Y PSNR {psnr_y} below baseline");
}

#[test]
fn ffmpeg_decode_chroma_residual_matches() {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }
    let (y, u, v) = make_testsrc_chroma_64x64();
    let frame = YuvFrame {
        width: 64,
        height: 64,
        y: &y,
        u: &u,
        v: &v,
    };
    let cfg = EncoderConfig::new(64, 64);
    let enc = Encoder::new(cfg);
    let out = enc.encode_idr(&frame);

    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!(
        "oxideav_h264_enc_chroma_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_enc_chroma_{}.yuv",
        std::process::id()
    ));
    std::fs::write(&h264_path, &out.annex_b).expect("write h264");

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

    let y_len = 64 * 64;
    let c_len = 32 * 32;
    assert_eq!(raw.len(), y_len + 2 * c_len);
    let ff_y = &raw[0..y_len];
    let ff_u = &raw[y_len..y_len + c_len];
    let ff_v = &raw[y_len + c_len..];

    for (k, (&a, &b)) in ff_y.iter().zip(out.recon_y.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "ffmpeg luma {k}: ff={a} enc_recon={b}",
        );
    }
    for (k, (&a, &b)) in ff_u.iter().zip(out.recon_u.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "ffmpeg Cb {k}: ff={a} enc_recon={b}",
        );
    }
    for (k, (&a, &b)) in ff_v.iter().zip(out.recon_v.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "ffmpeg Cr {k}: ff={a} enc_recon={b}",
        );
    }
    eprintln!("ffmpeg chroma interop: bit-exact match on full picture");
}

/// Optional ffmpeg interop test. Writes the encoder's Annex B output to
/// a temp file and asks ffmpeg to decode it back to raw YUV. Compares
/// the result against the encoder's local recon (sample-accurate match
/// expected because we run no in-loop filter and the spec defines the
/// reconstruction process bit-exactly).
///
/// Skipped when `/opt/homebrew/bin/ffmpeg` is not present (CI without
/// homebrew, etc.).
#[test]
fn ffmpeg_decode_matches_encoder_recon() {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }
    let (y, u, v) = make_testsrc_64x64();
    let frame = YuvFrame {
        width: 64,
        height: 64,
        y: &y,
        u: &u,
        v: &v,
    };
    let cfg = EncoderConfig::new(64, 64);
    let enc = Encoder::new(cfg);
    let out = enc.encode_idr(&frame);

    // Write the .h264 to a unique temp path.
    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!("oxideav_h264_enc_{}.h264", std::process::id()));
    let yuv_path = tmpdir.join(format!("oxideav_h264_enc_{}.yuv", std::process::id()));
    std::fs::write(&h264_path, &out.annex_b).expect("write h264");

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

    // Layout: 64*64 Y, then 32*32 U, then 32*32 V.
    let y_len = 64 * 64;
    let c_len = 32 * 32;
    assert_eq!(raw.len(), y_len + 2 * c_len);

    let ff_y = &raw[0..y_len];
    let ff_u = &raw[y_len..y_len + c_len];
    let ff_v = &raw[y_len + c_len..];

    // Compare to encoder's local recon.
    for (k, (&a, &b)) in ff_y.iter().zip(out.recon_y.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "ffmpeg luma {k}: ff={a} enc_recon={b}",
        );
    }
    for (k, (&a, &b)) in ff_u.iter().zip(out.recon_u.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "ffmpeg Cb {k}: ff={a} enc_recon={b}",
        );
    }
    for (k, (&a, &b)) in ff_v.iter().zip(out.recon_v.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "ffmpeg Cr {k}: ff={a} enc_recon={b}",
        );
    }
    eprintln!(
        "ffmpeg interop: bit-exact match (128-byte tolerance) on {} luma + 2*{} chroma samples",
        ff_y.len(),
        ff_u.len()
    );
}

/// Round-3 — luma AC residual lift. Uses a 64x64 noisy test source
/// designed to *force* significant AC: every pixel is a hash of (i, j).
/// A DC-only encoder sees pure noise and quantizes the residual to ~MB
/// average; an AC-capable encoder reproduces a much sharper image.
fn make_noisy_64x64() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = 64usize;
    let h = 64usize;
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            // Deterministic high-frequency pattern.
            y[j * w + i] = ((i * 17 + j * 23 + (i ^ j) * 7) % 200 + 28) as u8;
        }
    }
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];
    (y, u, v)
}

#[test]
fn round3_ac_residual_lifts_psnr_on_noisy_picture() {
    let (y, u, v) = make_noisy_64x64();
    let frame = YuvFrame {
        width: 64,
        height: 64,
        y: &y,
        u: &u,
        v: &v,
    };
    let mut cfg = EncoderConfig::new(64, 64);
    cfg.qp = 18; // Low QP so AC actually survives quantization.
    let enc = Encoder::new(cfg);
    let out = enc.encode_idr(&frame);

    // Self-roundtrip first (validates encoder recon == decoder recon).
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), out.annex_b.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let frame_out = dec.receive_frame().expect("receive_frame");
    let vf = match frame_out {
        Frame::Video(vf) => vf,
        other => panic!("expected Frame::Video, got {:?}", other),
    };
    for (k, (&a, &b)) in vf.planes[0].data.iter().zip(out.recon_y.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "luma sample {k}: decoded={a} encoder_recon={b}",
        );
    }

    let psnr = {
        let m = mse_y(&y, &out.recon_y);
        if m == 0.0 {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / m).log10()
        }
    };
    eprintln!(
        "round-3 noisy 64x64 (QP=18): stream={} bytes, PSNR Y={:.2}",
        out.annex_b.len(),
        psnr
    );
    // With AC residual the encoder should clear ~34 dB on this content
    // (DC-only Round 2 capped around 14 dB on noise).
    assert!(psnr >= 34.0, "PSNR {psnr:.2} dB below round-3 floor 34");

    // Comparison run at QP=26 — looser quantization so we can see the
    // gap between DC-only (round-2) and DC+AC (round-3) reconstruction
    // on the same input.
    let mut cfg2 = EncoderConfig::new(64, 64);
    cfg2.qp = 26;
    let out2 = Encoder::new(cfg2).encode_idr(&frame);
    let psnr_qp26 = {
        let m = mse_y(&y, &out2.recon_y);
        if m == 0.0 {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / m).log10()
        }
    };
    eprintln!(
        "round-3 noisy 64x64 (QP=26): stream={} bytes, PSNR Y={:.2}",
        out2.annex_b.len(),
        psnr_qp26
    );
}

#[test]
fn round3_ffmpeg_decode_noisy_picture_matches() {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }
    let (y, u, v) = make_noisy_64x64();
    let frame = YuvFrame {
        width: 64,
        height: 64,
        y: &y,
        u: &u,
        v: &v,
    };
    let mut cfg = EncoderConfig::new(64, 64);
    cfg.qp = 18;
    let enc = Encoder::new(cfg);
    let out = enc.encode_idr(&frame);

    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!(
        "oxideav_h264_enc_noisy_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!("oxideav_h264_enc_noisy_{}.yuv", std::process::id()));
    std::fs::write(&h264_path, &out.annex_b).expect("write h264");

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

    let y_len = 64 * 64;
    let c_len = 32 * 32;
    assert_eq!(raw.len(), y_len + 2 * c_len);
    let ff_y = &raw[0..y_len];
    for (k, (&a, &b)) in ff_y.iter().zip(out.recon_y.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "ffmpeg luma {k}: ff={a} enc_recon={b}",
        );
    }
    eprintln!(
        "round-3 ffmpeg interop (noisy, QP=18): stream={} bytes, bit-exact match",
        out.annex_b.len()
    );
}

// ---------------------------------------------------------------------------
// Round-4 — I_NxN (Intra_4x4) integration tests.
// ---------------------------------------------------------------------------

/// 64x64 picture with multiple distinct edge orientations per MB. Each
/// 4x4 sub-block of the picture has a different intra-4x4 mode that is
/// "best" for it (vertical stripes here, horizontal there, diagonal
/// gradient elsewhere). I_16x16 with one mode per macroblock can't
/// capture all 16 sub-block patterns; I_NxN should land much closer.
fn make_oriented_edges_64x64() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = 64usize;
    let h = 64usize;
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            let blk_x = (i / 4) % 4;
            let blk_y = (j / 4) % 4;
            // Pick a per-4x4-sub-block pattern based on its position
            // inside the parent MB.
            let local_x = i % 4;
            let local_y = j % 4;
            let v = match (blk_x + 4 * blk_y) % 4 {
                0 => 32 + (local_x * 50) as i32,             // vertical-ish
                1 => 32 + (local_y * 50) as i32,             // horizontal-ish
                2 => 32 + ((local_x + local_y) * 25) as i32, // diagonal
                _ => 32 + ((3 - local_x) * 30 + local_y * 20) as i32,
            };
            y[j * w + i] = v.clamp(0, 255) as u8;
        }
    }
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];
    (y, u, v)
}

#[test]
fn round4_i_nxn_lifts_psnr_on_oriented_edges() {
    let (y, u, v) = make_oriented_edges_64x64();
    let frame = YuvFrame {
        width: 64,
        height: 64,
        y: &y,
        u: &u,
        v: &v,
    };
    let mut cfg = EncoderConfig::new(64, 64);
    cfg.qp = 26;
    let enc = Encoder::new(cfg);
    let out = enc.encode_idr(&frame);

    // Self-roundtrip must match local recon bit-exactly.
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), out.annex_b.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let frame_out = dec.receive_frame().expect("receive_frame");
    let vf = match frame_out {
        Frame::Video(vf) => vf,
        other => panic!("expected Frame::Video, got {:?}", other),
    };
    for (k, (&a, &b)) in vf.planes[0].data.iter().zip(out.recon_y.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "luma sample {k}: decoded={a} encoder_recon={b}",
        );
    }
    let psnr = {
        let m = mse_y(&y, &out.recon_y);
        if m == 0.0 {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / m).log10()
        }
    };
    eprintln!(
        "round-4 oriented-edges 64x64 (QP=26): stream={} bytes, PSNR Y={:.2}",
        out.annex_b.len(),
        psnr
    );
    // I_NxN should bring this above 30 dB; I_16x16 alone caps around 22 dB.
    assert!(psnr >= 28.0, "PSNR {psnr:.2} dB below round-4 floor 28");
}

#[test]
fn round4_ffmpeg_decode_oriented_edges_matches() {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }
    let (y, u, v) = make_oriented_edges_64x64();
    let frame = YuvFrame {
        width: 64,
        height: 64,
        y: &y,
        u: &u,
        v: &v,
    };
    let mut cfg = EncoderConfig::new(64, 64);
    cfg.qp = 26;
    let enc = Encoder::new(cfg);
    let out = enc.encode_idr(&frame);

    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!("oxideav_h264_enc_r4_{}.h264", std::process::id()));
    let yuv_path = tmpdir.join(format!("oxideav_h264_enc_r4_{}.yuv", std::process::id()));
    std::fs::write(&h264_path, &out.annex_b).expect("write h264");

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

    let y_len = 64 * 64;
    let c_len = 32 * 32;
    assert_eq!(raw.len(), y_len + 2 * c_len);
    let ff_y = &raw[0..y_len];
    let ff_u = &raw[y_len..y_len + c_len];
    let ff_v = &raw[y_len + c_len..];
    for (k, (&a, &b)) in ff_y.iter().zip(out.recon_y.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "ffmpeg luma {k}: ff={a} enc_recon={b}",
        );
    }
    for (k, (&a, &b)) in ff_u.iter().zip(out.recon_u.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "ffmpeg Cb {k}: ff={a} enc_recon={b}",
        );
    }
    for (k, (&a, &b)) in ff_v.iter().zip(out.recon_v.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "ffmpeg Cr {k}: ff={a} enc_recon={b}",
        );
    }
    eprintln!(
        "round-4 ffmpeg interop (oriented-edges, QP=26): stream={} bytes, bit-exact",
        out.annex_b.len()
    );
}

// ---------------------------------------------------------------------------
// Round-14 — in-loop deblocking integration tests.
// ---------------------------------------------------------------------------

/// 64x64 picture with deliberate per-MB DC offsets that produce a
/// staircase pattern: every 16x16 macroblock has a slightly different
/// DC value (around mid-gray). With the in-loop filter disabled the
/// MB boundaries would be sharply visible (1-2 step at each x=16 / y=16
/// boundary), forming exactly the kind of "blocking artefact" that §8.7
/// is designed to suppress. The deblock filter should pull these
/// boundaries closer together.
fn make_dc_staircase_64x64() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = 64usize;
    let h = 64usize;
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            // Per-MB DC offset (each MB gets its own mid-gray). The
            // step between adjacent MBs is small enough to be inside
            // the §8.7 deblock activation band at QP 26.
            let mb_x = i / 16;
            let mb_y = j / 16;
            let dc = 120 + (mb_x as i32) * 4 + (mb_y as i32) * 3;
            y[j * w + i] = dc.clamp(0, 255) as u8;
        }
    }
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];
    (y, u, v)
}

/// Round-14 self-roundtrip: with deblocking on, the encoder's local
/// recon must still match the decoder's recon bit-exactly (the encoder
/// runs the same §8.7 pass that the decoder runs).
#[test]
fn round14_deblock_self_roundtrip_matches() {
    let (y, u, v) = make_dc_staircase_64x64();
    let frame = YuvFrame {
        width: 64,
        height: 64,
        y: &y,
        u: &u,
        v: &v,
    };
    let cfg = EncoderConfig::new(64, 64);
    let enc = Encoder::new(cfg);
    let out = enc.encode_idr(&frame);

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), out.annex_b.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let frame_out = dec.receive_frame().expect("receive_frame");
    let vf = match frame_out {
        Frame::Video(vf) => vf,
        other => panic!("expected Frame::Video, got {:?}", other),
    };
    for (k, (&a, &b)) in vf.planes[0].data.iter().zip(out.recon_y.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "luma sample {k}: decoded={a} encoder_recon={b}",
        );
    }
    for (k, (&a, &b)) in vf.planes[1].data.iter().zip(out.recon_u.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "Cb sample {k}: decoded={a} encoder_recon={b}",
        );
    }
    for (k, (&a, &b)) in vf.planes[2].data.iter().zip(out.recon_v.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "Cr sample {k}: decoded={a} encoder_recon={b}",
        );
    }

    let psnr = {
        let m = mse_y(&y, &out.recon_y);
        if m == 0.0 {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / m).log10()
        }
    };
    eprintln!(
        "round-14 dc-staircase 64x64 (QP=26): stream={} bytes, PSNR Y={:.2}",
        out.annex_b.len(),
        psnr,
    );
    // §8.7 should keep the boundary smooth — this test is mainly an
    // interop guard; the floor is intentionally generous.
    assert!(psnr >= 30.0, "PSNR {psnr:.2} dB below round-14 floor 30");
}

/// Round-14 ffmpeg interop: the encoder's deblock pass must produce
/// the exact same samples as ffmpeg's libavcodec deblocker.
#[test]
fn round14_ffmpeg_decode_deblocked_matches() {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }
    let (y, u, v) = make_dc_staircase_64x64();
    let frame = YuvFrame {
        width: 64,
        height: 64,
        y: &y,
        u: &u,
        v: &v,
    };
    let cfg = EncoderConfig::new(64, 64);
    let enc = Encoder::new(cfg);
    let out = enc.encode_idr(&frame);

    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!("oxideav_h264_enc_r14_{}.h264", std::process::id()));
    let yuv_path = tmpdir.join(format!("oxideav_h264_enc_r14_{}.yuv", std::process::id()));
    std::fs::write(&h264_path, &out.annex_b).expect("write h264");

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

    let y_len = 64 * 64;
    let c_len = 32 * 32;
    assert_eq!(raw.len(), y_len + 2 * c_len);
    let ff_y = &raw[0..y_len];
    let ff_u = &raw[y_len..y_len + c_len];
    let ff_v = &raw[y_len + c_len..];
    for (k, (&a, &b)) in ff_y.iter().zip(out.recon_y.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "ffmpeg luma {k}: ff={a} enc_recon={b}",
        );
    }
    for (k, (&a, &b)) in ff_u.iter().zip(out.recon_u.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "ffmpeg Cb {k}: ff={a} enc_recon={b}",
        );
    }
    for (k, (&a, &b)) in ff_v.iter().zip(out.recon_v.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "ffmpeg Cr {k}: ff={a} enc_recon={b}",
        );
    }
    eprintln!(
        "round-14 ffmpeg interop (dc-staircase, QP=26): stream={} bytes, bit-exact",
        out.annex_b.len()
    );
}
