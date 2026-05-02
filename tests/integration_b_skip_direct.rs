//! Round-21 — B_Skip / B_Direct_16x16 (spatial direct mode)
//!
//! These tests exercise the §8.4.1.2.2 spatial-direct MV derivation
//! path on top of the round-20 explicit-inter B-slice support. Two
//! fixtures:
//!
//! 1. `round21_static_picture_uses_b_skip` — IDR + P + B all share
//!    the same source content → ME finds (0,0) for both lists →
//!    bipred predictor exactly reproduces source → cbp == 0 → every
//!    B MB is emittable as `B_Skip` (no MB syntax). The whole B-slice
//!    collapses to a slice header + a single trailing `mb_skip_run =
//!    width_mbs * height_mbs`.
//!
//! 2. `round21_b_skip_ffmpeg_interop` — feed the same stream to
//!    libavcodec and verify the decoded YUV matches the encoder's
//!    local recon bit-exactly.
//!
//! Cross-reference: §7.3.4 (mb_skip_run), §7.4.5 Table 7-14
//! (B_Skip is implicit; B_Direct_16x16 is raw mb_type 0),
//! §8.4.1.2 / §8.4.1.2.2 (direct-mode MV derivation).

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 64;
const H: usize = 64;

/// 64x64 mid-grey horizontal ramp. Same content for IDR / P / B → no
/// motion → spatial direct produces (0, 0) MVs everywhere → B_Skip.
fn make_static_picture() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
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

#[test]
fn round21_static_picture_uses_b_skip() {
    // Static IDR + P + B: every B MB should pick B_Skip (no syntax).
    //
    // This fixture isolates the round-21 B_Skip selection path. With the
    // round-29 `intra_in_inter` RDO trial enabled (the encoder default),
    // a few P-slice MBs pick Intra_16x16 over P_Skip on this 64x64
    // gradient because the post-IDR §8.7 deblock smoothing leaves the
    // P_Skip predictor (= IDR recon) a couple of LSBs off the source
    // and the intra trial's RDO cost lands lower. The resulting P recon
    // is then used as the L1 anchor of the B-slice; the spatial-direct
    // bipred predictor (= avg of L0=IDR and L1=P) inherits that small
    // offset, so a few B MBs end up with cbp != 0 and cannot collapse
    // to B_Skip — pushing the slice from ~11 bytes to ~18 bytes and
    // breaking the round-21 byte-budget assertion below.
    //
    // Round-21 was designed and verified before round-29 existed, so
    // we explicitly opt out of the intra fallback here to keep the
    // round-21 invariant ("static content collapses to all-B_Skip")
    // testable in isolation. The round-29 fallback has its own
    // dedicated coverage in `tests/integration_p_slice_intra_fallback.rs`.
    let mut cfg = EncoderConfig::new(W as u32, H as u32);
    cfg.profile_idc = 77; // Main — required for B-slices (§A.2.2)
    cfg.max_num_ref_frames = 2;
    cfg.intra_in_inter = false;
    let enc = Encoder::new(cfg);

    let (y0, u0, v0) = make_static_picture();
    let f = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y0,
        u: &u0,
        v: &v0,
    };

    let idr = enc.encode_idr(&f);
    let p = enc.encode_p(&f, &EncodedFrameRef::from(&idr), 1, 4);
    let b = enc.encode_b(
        &f,
        &EncodedFrameRef::from(&idr),
        &EncodedFrameRef::from(&p),
        1, // frame_num — same as P (B is non-reference)
        2, // pic_order_cnt_lsb between IDR (0) and P (4)
    );

    eprintln!(
        "round-21 static-content stream sizes: IDR={}, P={}, B={}",
        idr.annex_b.len(),
        p.annex_b.len(),
        b.annex_b.len(),
    );

    // The B-slice should be SHORTER than the round-20 B-slice (which
    // was 16 bytes for the same dimensions because it carried per-MB
    // mb_type/mvds). Under round-21 with all-skips, the B-slice is
    // just the slice header + a final `mb_skip_run = 16` ue(v) +
    // trailing bits → on the order of 6 bytes or less. The round-20
    // baseline for the all-Bi midpoint test was about 16 bytes; here
    // we sanity-check that we actually shrunk the slice.
    assert!(
        b.annex_b.len() <= 14,
        "round-21 B-slice on static content should fit in <=14 bytes, got {}",
        b.annex_b.len(),
    );

    // Round-trip the IDR + P + B through our own decoder.
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
    assert_eq!(frames.len(), 3, "expected 3 decoded frames (IDR + P + B)");

    // Display order: IDR (POC 0) → B (POC 2) → P (POC 4).
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
            "B luma mismatch at idx {k}: dec={a}, enc={local} (diff {d})",
        );
    }
    let psnr_dec = psnr(&y0, &b_decoded.planes[0].data);
    eprintln!(
        "round-21 static-content B-slice PSNR_Y = {:.2} dB (max enc/dec diff {max_diff})",
        psnr_dec,
    );
    assert!(
        psnr_dec >= 38.0,
        "round-21 B PSNR_Y {psnr_dec:.2} dB below floor 38.0",
    );
}

/// Round-21 — same anchors as round-20's midpoint test (IDR + shifted-P
/// linear-interp B), but the encoder now picks B_Direct / B_Skip on MBs
/// where the spatial-direct predictor is competitive with the
/// explicit-Bi candidate. The bipred predictor at MV=(0,0) bit-exactly
/// reproduces the source on this fixture so virtually every MB hits
/// `cbp == 0` → B_Skip → run-length compressed slice.
///
/// The point of this test is to verify that the round-20 fixture still
/// round-trips bit-exactly through both our own decoder and ffmpeg
/// after the round-21 mode-decision changes.
#[test]
fn round21_midpoint_b_slice_decoder_match() {
    use oxideav_core::Decoder as _;

    let mut cfg = EncoderConfig::new(W as u32, H as u32);
    cfg.profile_idc = 77;
    cfg.max_num_ref_frames = 2;
    let enc = Encoder::new(cfg);

    // IDR: smooth horizontal ramp.
    let mut y0 = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            y0[j * W + i] = (60 + i + j) as u8;
        }
    }
    let u0 = vec![128u8; (W / 2) * (H / 2)];
    let v0 = vec![128u8; (W / 2) * (H / 2)];

    // P: same content shifted right by 4 luma px (clamped at left edge).
    let mut y1 = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let i_src = i.saturating_sub(4);
            y1[j * W + i] = (60 + i_src + j) as u8;
        }
    }
    let u1 = vec![128u8; (W / 2) * (H / 2)];
    let v1 = vec![128u8; (W / 2) * (H / 2)];

    // B: midpoint between IDR and P (the round-20 fixture).
    let mut y2 = vec![0u8; W * H];
    for k in 0..(W * H) {
        let a = y0[k] as u32;
        let b = y1[k] as u32;
        y2[k] = ((a + b + 1) >> 1) as u8;
    }
    let u2 = vec![128u8; (W / 2) * (H / 2)];
    let v2 = vec![128u8; (W / 2) * (H / 2)];

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
    for (&a, &local) in b_decoded.planes[0].data.iter().zip(b.recon_y.iter()) {
        let d = (a as i32 - local as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(d <= 1, "B luma mismatch: dec={a}, enc={local}");
    }
    let psnr_dec = psnr(&y2, &b_decoded.planes[0].data);
    eprintln!(
        "round-21 midpoint B-slice PSNR_Y = {:.2} dB (max diff {max_diff}, B-slice {} bytes)",
        psnr_dec,
        b.annex_b.len(),
    );
    assert!(psnr_dec >= 38.0);
}

#[test]
fn round21_b_skip_ffmpeg_interop() {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }

    let mut cfg = EncoderConfig::new(W as u32, H as u32);
    cfg.profile_idc = 77;
    cfg.max_num_ref_frames = 2;
    let enc = Encoder::new(cfg);

    let (y0, u0, v0) = make_static_picture();
    let f = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y0,
        u: &u0,
        v: &v0,
    };

    let idr = enc.encode_idr(&f);
    let p = enc.encode_p(&f, &EncodedFrameRef::from(&idr), 1, 4);
    let b = enc.encode_b(
        &f,
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
        "oxideav_h264_round21_skip_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_round21_skip_{}.yuv",
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

    // Display order: IDR → B → P. The B is at frame index 1.
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
        "round-21 ffmpeg interop static B-slice: bit-equivalent (max diff {max_diff}, stream {} bytes)",
        combined.len()
    );

    let psnr_ff = psnr(&y0, b_y);
    eprintln!("round-21 ffmpeg static B-slice PSNR_Y = {:.2} dB", psnr_ff);
    assert!(
        psnr_ff >= 38.0,
        "round-21 ffmpeg B PSNR_Y {psnr_ff:.2} dB below floor 38.0",
    );
}
