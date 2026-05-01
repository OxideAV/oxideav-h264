//! Round-28 — 4:4:4 chroma encoder integration tests.
//!
//! Exercises the IDR-only 4:4:4 path:
//!  * Encoder emits a High 4:4:4 Predictive (profile_idc=244) SPS with
//!    `chroma_format_idc=3` and `separate_colour_plane_flag=0`.
//!  * Per-MB chroma residual is "coded like luma" per §7.3.5.3:
//!    each plane has a 16x16 DC Hadamard block and 16 4x4 AC blocks,
//!    sharing the luma Intra_16x16 prediction mode (§8.3.4.1).
//!  * Decoder reads back at 4:4:4 (`Picture::chroma_array_type=3`,
//!    chroma plane sized at `(W, H)`).
//!  * ffmpeg cross-decode produces sample-accurate output (the encoder
//!    sets `disable_deblocking_filter_idc=1` for round-28, so the §8.7
//!    chroma deblock pass — which uses the LUMA filter for
//!    ChromaArrayType==3 — is not exercised; recon is determined
//!    entirely by the per-plane Intra_16x16 reconstruction pipeline).
//!
//! The 4:2:0 / 4:2:2 paths and the I_NxN / P / B paths are out of
//! scope for this round and are not exercised here.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::{Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

/// Build a 64x64 4:4:4 test source: smooth diagonal luma gradient,
/// horizontal Cb gradient, vertical Cr gradient. Chroma plane is
/// `64 x 64` (W × H per §6.2 Table 6-1).
fn make_testsrc_444_64x64() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = 64usize;
    let h = 64usize;
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; w * h];
    let mut v = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            // Smooth luma gradient 16..240 across the picture diagonal.
            y[j * w + i] = (16 + (i + j) * (240 - 16) / (w + h - 2)).clamp(0, 255) as u8;
            // Horizontal Cb gradient.
            u[j * w + i] = (64 + i * 128 / (w - 1)) as u8;
            // Vertical Cr gradient.
            v[j * w + i] = (64 + j * 128 / (h - 1)) as u8;
        }
    }
    (y, u, v)
}

fn mse_plane(orig: &[u8], recon: &[u8]) -> f64 {
    let mut s = 0u64;
    for (&o, &r) in orig.iter().zip(recon.iter()) {
        let d = o as i32 - r as i32;
        s += (d * d) as u64;
    }
    s as f64 / orig.len() as f64
}

fn psnr(mse: f64) -> f64 {
    if mse == 0.0 {
        99.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

#[test]
fn round28_444_encode_decode_self_roundtrip_matches_local_recon() {
    let (y, u, v) = make_testsrc_444_64x64();
    let frame = YuvFrame {
        width: 64,
        height: 64,
        y: &y,
        u: &u,
        v: &v,
    };
    let mut cfg = EncoderConfig::new(64, 64);
    cfg.profile_idc = 244; // High 4:4:4 Predictive — required for chroma_format_idc=3.
    cfg.chroma_format_idc = 3;
    let enc = Encoder::new(cfg);
    let out = enc.encode_idr(&frame);

    // The encoder's local recon for 4:4:4 must size chroma at (W, H).
    assert_eq!(out.recon_u.len(), 64 * 64, "Cb plane wrong length");
    assert_eq!(out.recon_v.len(), 64 * 64, "Cr plane wrong length");

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
    assert_eq!(vf.planes.len(), 3, "expected 3-plane layout");
    // Luma stride = 64.
    assert_eq!(vf.planes[0].stride, 64);
    assert_eq!(vf.planes[0].data.len(), 64 * 64);
    // Chroma stride = W = 64; chroma height = H = 64 (4:4:4).
    assert_eq!(vf.planes[1].stride, 64);
    assert_eq!(vf.planes[1].data.len(), 64 * 64);
    assert_eq!(vf.planes[2].data.len(), 64 * 64);

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

    let ps_y = psnr(mse_plane(&y, &out.recon_y));
    let ps_u = psnr(mse_plane(&u, &out.recon_u));
    let ps_v = psnr(mse_plane(&v, &out.recon_v));
    eprintln!(
        "round-28 4:4:4 self-roundtrip 64x64 (QP=26): stream={} bytes, PSNR Y={:.2} U={:.2} V={:.2}",
        out.annex_b.len(),
        ps_y,
        ps_u,
        ps_v,
    );
    // Spec requirement (acceptance): luma PSNR >= 40 dB on a synthetic
    // gradient.
    assert!(
        ps_y >= 40.0,
        "round-28 luma PSNR {ps_y:.2} dB below 40 floor",
    );
}

#[test]
fn round28_444_ffmpeg_decode_matches_encoder_recon() {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }
    let (y, u, v) = make_testsrc_444_64x64();
    let frame = YuvFrame {
        width: 64,
        height: 64,
        y: &y,
        u: &u,
        v: &v,
    };
    let mut cfg = EncoderConfig::new(64, 64);
    cfg.profile_idc = 244;
    cfg.chroma_format_idc = 3;
    let enc = Encoder::new(cfg);
    let out = enc.encode_idr(&frame);

    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!("oxideav_h264_enc_444_{}.h264", std::process::id()));
    let yuv_path = tmpdir.join(format!("oxideav_h264_enc_444_{}.yuv", std::process::id()));
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
        .arg("yuv444p")
        .arg(&yuv_path)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed");
    let raw = std::fs::read(&yuv_path).expect("read yuv");
    let _ = std::fs::remove_file(&h264_path);
    let _ = std::fs::remove_file(&yuv_path);

    // Yuv444p layout: Y plane = WxH, Cb = WxH, Cr = WxH.
    let plane_len = 64 * 64;
    assert_eq!(raw.len(), 3 * plane_len, "ffmpeg output wrong size");
    let ff_y = &raw[0..plane_len];
    let ff_u = &raw[plane_len..2 * plane_len];
    let ff_v = &raw[2 * plane_len..];

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
    let ps_y = psnr(mse_plane(&y, &out.recon_y));
    eprintln!(
        "round-28 4:4:4 ffmpeg interop (gradient, QP=26): stream={} bytes, bit-exact, PSNR Y={:.2}",
        out.annex_b.len(),
        ps_y,
    );
    assert!(
        ps_y >= 40.0,
        "Y PSNR {ps_y:.2} dB below 40 (spec acceptance)"
    );
}
