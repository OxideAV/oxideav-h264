//! Round-20 — B-slice integration: encode IDR + P + B, decode through
//! both our own decoder and ffmpeg, measure PSNR.
//!
//! The B-frame uses **two reference frames**: L0 = IDR (POC 0), L1 = the
//! intermediate P-frame (POC 4). The B itself is at POC 2 — i.e. between
//! the two anchors in display order, after them in decode order
//! (canonical IPB GOP).
//!
//! Per-MB the encoder picks among {B_L0_16x16, B_L1_16x16, B_Bi_16x16}
//! by minimum-SAD on the chosen list's predictor (`(L0 + L1 + 1) >> 1`
//! for bipred per §8.4.2.3.1 eq. 8-273). With a textbook IPB sequence
//! built by **linear interpolation between IDR and P**, every MB should
//! prefer the bipred candidate (it bit-exactly recovers the source) so
//! PSNR_Y stays high and the residual is small.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 64;
const H: usize = 64;

/// 64x64 IDR: smooth horizontal ramp, mid-grey chroma.
fn make_idr() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            // Mid-greys, bounded so the linear-interp B stays in 0..=255.
            y[j * W + i] = (60 + i + j) as u8;
        }
    }
    let u = vec![128u8; (W / 2) * (H / 2)];
    let v = vec![128u8; (W / 2) * (H / 2)];
    (y, u, v)
}

/// 64x64 P: same content shifted right by 4 luma px (an integer-pel
/// motion the round-16 ME finds exactly). Stays clamped in 0..=255.
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

/// 64x64 B (intermediate): exact midpoint between IDR and P, sample-by-
/// sample. With the encoder's bipred merge `(L0 + L1 + 1) >> 1`, the
/// B-frame predictor reconstructs the source bit-exactly — every MB
/// trips into B_Bi_16x16 with cbp_luma == 0.
fn make_b(idr_y: &[u8], p_y: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for k in 0..(W * H) {
        let a = idr_y[k] as u32;
        let b = p_y[k] as u32;
        // §8.4.2.3.1 eq. 8-273 average.
        y[k] = ((a + b + 1) >> 1) as u8;
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
fn round20_b_slice_self_roundtrip_matches_local_recon() {
    // IDR + P (round-16/17/18 path) + B (round-20).
    let mut cfg = EncoderConfig::new(W as u32, H as u32);
    cfg.profile_idc = 77; // Main — required for B-slices (§A.2.2)
    cfg.max_num_ref_frames = 2; // both anchors must stay in DPB
    let enc = Encoder::new(cfg);

    let (y0, u0, v0) = make_idr();
    let (y1, u1, v1) = make_p();
    let (y2, u2, v2) = make_b(&y0, &y1);

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
    // P frame: decode-order index 1, POC LSB 4 (display position 2 in
    // an IPB GOP where the B at POC 2 sits between IDR @ POC 0 and P @
    // POC 4). frame_num = 1 (P is a reference).
    let p = enc.encode_p(&f1, &EncodedFrameRef::from(&idr), 1, 4);
    // B frame: decode-order index 2, POC LSB 2 (between IDR and P in
    // display order). frame_num is not incremented for non-reference B.
    let b = enc.encode_b(
        &f2,
        &EncodedFrameRef::from(&idr),
        &EncodedFrameRef::from(&p),
        1, // frame_num — same as the P that preceded it (B is non-reference)
        2, // pic_order_cnt_lsb — between IDR (0) and P (4)
    );

    let psnr_local = psnr(&y2, &b.recon_y);
    eprintln!(
        "round-20 B-slice: stream={} bytes (IDR {}, P {}, B {}), local recon PSNR_Y = {:.2} dB",
        idr.annex_b.len() + p.annex_b.len() + b.annex_b.len(),
        idr.annex_b.len(),
        p.annex_b.len(),
        b.annex_b.len(),
        psnr_local,
    );
    // With perfect bipred-on-midpoint content the encoder's local recon
    // should be very close to the source.
    assert!(
        psnr_local >= 38.0,
        "round-20 B local recon PSNR_Y {psnr_local:.2} dB below floor 38.0",
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

    // The decoder delivers frames in *display* order via §C.4 bumping.
    // Display order: IDR (POC 0) → B (POC 2) → P (POC 4).
    // So frames[0] = IDR, frames[1] = B, frames[2] = P.
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
        // Bipred chains should be bit-equivalent encoder ↔ decoder; the
        // ¼-pel test in round 18 allows ≤1 LSB drift due to accumulator
        // ordering, so we accept the same tolerance here even though
        // most MBs use integer/half-pel MVs.
        assert!(
            d <= 1,
            "B luma mismatch at idx {k}: dec={a}, enc={local} (diff {d})",
        );
    }
    let psnr_dec = psnr(&y2, &b_decoded.planes[0].data);
    eprintln!(
        "round-20 B-slice: decoder PSNR_Y = {:.2} dB (max enc/dec diff {max_diff_y})",
        psnr_dec,
    );
    assert!(
        psnr_dec >= 38.0,
        "round-20 B decoder PSNR_Y {psnr_dec:.2} dB below floor 38.0",
    );
}

#[test]
fn round20_b_slice_ffmpeg_interop() {
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
    let (y2, u2, v2) = make_b(&y0, &y1);

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
        "oxideav_h264_round20_b_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!("oxideav_h264_round20_b_{}.yuv", std::process::id()));
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

    // ffmpeg outputs in display order (it consumes the bumping order).
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
        "round-20 ffmpeg interop: bit-equivalent on B-slice (max diff {max_diff}, stream {} bytes)",
        combined.len()
    );

    let psnr_ff = psnr(&y2, b_y);
    eprintln!("round-20 ffmpeg PSNR_Y on B-slice = {:.2} dB", psnr_ff,);
    assert!(
        psnr_ff >= 38.0,
        "round-20 ffmpeg B-slice PSNR_Y {psnr_ff:.2} dB below floor 38.0",
    );
}
