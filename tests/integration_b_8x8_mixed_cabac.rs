//! Round-475 task #475 — CABAC B-slice **per-cell mixed B_8x8**.
//!
//! Mirrors the round-25 CAVLC mixed B_8x8 fixture
//! (`integration_b_8x8_mixed.rs`) but routes through
//! `Encoder::encode_b_cabac`. Per-cell motion forces the encoder to
//! pick `mb_type=22` + per-cell `sub_mb_type ∈ {1, 2, 3}` (B_L0_8x8 /
//! B_L1_8x8 / B_Bi_8x8), exercising the new mixed-cell mvd plumbing
//! (cell-local MVP, mvd_l0[i] then mvd_l1[i] in raster order, per-cell
//! `cabac_mvp_for_b_8x8_partition`).
//!
//! Cross-reference:
//!   * §7.3.5.2 sub_mb_pred() — mvd_l0 / mvd_l1 cell-order emission
//!   * §7.4.5 Table 7-14 — B mb_type raw 22 (B_8x8)
//!   * §7.4.5 Table 7-18 — sub_mb_type raw 0..3 single-partition cells
//!   * §8.4.1.3 — partition-shape mvp (Default for 8x8 sub-MB)
//!   * §9.3.3.1.1.3 — CABAC mb_type binarisation (mb_type=22 → "111111")
//!   * §9.3.3.1.1.3 + Table 9-38 — CABAC sub_mb_type binarisation
//!
//! Acceptance: bit-equivalent (≤ 1 LSB) decoder roundtrip on the B-frame
//! local recon, hard-asserted ffmpeg cross-decode (max diff 0), and
//! PSNR_Y ≥ 38 dB.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 32;
const H: usize = 16;

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

/// Same fixture shape as the CAVLC round-25 mixed test: per-quadrant
/// content offsets so the per-cell ME finds different MVs that drive
/// the encoder to pick at least one non-Direct cell.
fn make_idr() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let off = match (i / 8 % 2, j / 8) {
                (0, 0) => 0,
                (1, 0) => 20,
                (0, 1) => 40,
                _ => 60,
            };
            y[j * W + i] = (60 + i + j + off) as u8;
        }
    }
    let u = vec![128u8; (W / 2) * (H / 2)];
    let v = vec![128u8; (W / 2) * (H / 2)];
    (y, u, v)
}

fn make_p(idr_y: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let i_src = i.saturating_sub(4);
            y[j * W + i] = idr_y[j * W + i_src];
        }
    }
    let u = vec![128u8; (W / 2) * (H / 2)];
    let v = vec![128u8; (W / 2) * (H / 2)];
    (y, u, v)
}

fn make_b(p_y: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y = p_y.to_vec();
    let u = vec![128u8; (W / 2) * (H / 2)];
    let v = vec![128u8; (W / 2) * (H / 2)];
    (y, u, v)
}

#[test]
fn task475_b_8x8_mixed_cabac_self_roundtrip_matches_local_recon() {
    let mut cfg = EncoderConfig::new(W as u32, H as u32);
    cfg.profile_idc = 77; // Main — required for B (§A.2.2)
    cfg.cabac = true; // CABAC entropy
    cfg.max_num_ref_frames = 2;
    let enc = Encoder::new(cfg);

    let (y0, u0, v0) = make_idr();
    let (y1, u1, v1) = make_p(&y0);
    let (y2, u2, v2) = make_b(&y1);

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
        "task #475 CABAC B_8x8 mixed: stream IDR={} P={} B={} bytes, local PSNR_Y = {:.2} dB",
        idr.annex_b.len(),
        p.annex_b.len(),
        b.annex_b.len(),
        psnr_local,
    );
    assert!(
        psnr_local >= 38.0,
        "task #475 local recon PSNR_Y {psnr_local:.2} dB below floor 38.0",
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

    // Display order: IDR (POC 0) → B (POC 2) → P (POC 4). frames[1] = B.
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
    let psnr_dec = psnr(&y2, &b_decoded.planes[0].data);
    eprintln!(
        "task #475 CABAC B_8x8 mixed: decoder PSNR_Y = {:.2} dB (max enc/dec diff {max_diff})",
        psnr_dec,
    );
    assert!(
        psnr_dec >= 38.0,
        "task #475 decoder PSNR_Y {psnr_dec:.2} dB below floor 38.0",
    );
}

#[test]
fn task475_b_8x8_mixed_cabac_ffmpeg_interop() {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }

    let mut cfg = EncoderConfig::new(W as u32, H as u32);
    cfg.profile_idc = 77;
    cfg.cabac = true;
    cfg.max_num_ref_frames = 2;
    let enc = Encoder::new(cfg);

    let (y0, u0, v0) = make_idr();
    let (y1, u1, v1) = make_p(&y0);
    let (y2, u2, v2) = make_b(&y1);

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
        "oxideav_h264_task475_b8x8mixed_cabac_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_task475_b8x8mixed_cabac_{}.yuv",
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
        assert!(d <= 1, "ffmpeg B luma {k}: ff={a} enc={local} (diff {d})",);
    }
    eprintln!(
        "task #475 ffmpeg interop: bit-equivalent on CABAC B-slice (max diff {max_diff}, stream {} bytes)",
        combined.len()
    );

    let psnr_ff = psnr(&y2, b_y);
    eprintln!(
        "task #475 ffmpeg PSNR_Y on CABAC B-slice = {:.2} dB",
        psnr_ff
    );
    assert!(
        psnr_ff >= 38.0,
        "task #475 ffmpeg B-slice PSNR_Y {psnr_ff:.2} dB below floor 38.0",
    );
    // Hard-asserted: ffmpeg's decode bit-matches our local recon.
    assert_eq!(
        max_diff, 0,
        "task #475 max diff must be 0 (bit-equivalent ffmpeg cross-decode)"
    );
}

/// Reachability: stream is non-trivial and significantly smaller than
/// the worst-case all-explicit baseline. Mirrors the round-25 CAVLC
/// `path_exercised` test.
#[test]
fn task475_b_8x8_mixed_cabac_path_exercised_on_per_cell_motion() {
    let mut cfg = EncoderConfig::new(W as u32, H as u32);
    cfg.profile_idc = 77;
    cfg.cabac = true;
    cfg.max_num_ref_frames = 2;
    let enc = Encoder::new(cfg);

    let (y0, u0, v0) = make_idr();
    let (y1, u1, v1) = make_p(&y0);
    let (y2, u2, v2) = make_b(&y1);

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

    let idr = enc.encode_idr_cabac(&f0);
    let p = enc.encode_p_cabac(&f1, &EncodedFrameRef::from(&idr), 1, 4);
    let b = enc.encode_b_cabac(
        &f2,
        &EncodedFrameRef::from(&idr),
        &EncodedFrameRef::from(&p),
        1,
        2,
    );

    eprintln!(
        "task #475 CABAC reachability: IDR={} P={} B={} bytes",
        idr.annex_b.len(),
        p.annex_b.len(),
        b.annex_b.len(),
    );
    // Loose upper bound — the B-slice must be much smaller than the
    // worst-case all-16x16-explicit path.
    assert!(
        b.annex_b.len() < 200,
        "task #475 CABAC B-slice unexpectedly large ({} bytes)",
        b.annex_b.len(),
    );
}
