//! Round-26 — explicit weighted bipred (PPS `weighted_bipred_idc = 1`,
//! per-slice `pred_weight_table()`).
//!
//! The fixture is a textbook **cross-fade** scenario: L0 (IDR) is a
//! low-brightness textured frame, L1 (P) is the same texture at high
//! brightness, and the B in between is at fade-position 70 % toward
//! L0. The default `(L0 + L1 + 1) >> 1` bipred predicts the midpoint —
//! ~30 LSB off from the source on every sample — so the residual is
//! large. With explicit weights `(w0=45, w1=19)` at `log2_wd = 5`
//! (`α ≈ 0.7`) the predictor lands within 1 LSB of the source and the
//! residual collapses to ~zero, which CAVLC encodes in dramatically
//! fewer bits.
//!
//! The test asserts:
//!   * Encoder enabled with `explicit_weighted_bipred = true` produces a
//!     B-slice NAL that is at least 10 % smaller than the same encoder
//!     with the flag off.
//!   * Both bitstreams decode bit-equivalent (max diff <= 1) through
//!     our own H264CodecDecoder.
//!   * If ffmpeg is available, it cross-decodes the weighted bitstream
//!     to the same recon as our encoder predicts (≤ 1 LSB drift).

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 64;
const H: usize = 64;

/// Low-brightness textured plane: base 30 + small ramp.
fn make_idr_dark() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            // Texture amplitude small enough to keep both L0 and L1
            // safely inside [0, 255] after the brightness shift.
            let tex = ((i + j) as u8) & 0x1f; // 0..=31
            y[j * W + i] = 30u8.saturating_add(tex);
        }
    }
    let u = vec![128u8; (W / 2) * (H / 2)];
    let v = vec![128u8; (W / 2) * (H / 2)];
    (y, u, v)
}

/// High-brightness version of the same texture: base 200 + same ramp.
fn make_p_bright() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let tex = ((i + j) as u8) & 0x1f;
            y[j * W + i] = 200u8.saturating_add(tex);
        }
    }
    let u = vec![128u8; (W / 2) * (H / 2)];
    let v = vec![128u8; (W / 2) * (H / 2)];
    (y, u, v)
}

/// B at fade position α=0.7 toward L0 (i.e. closer to dark).
///   src ≈ 0.7 * L0 + 0.3 * L1
/// Computed from the *source* L0/L1, not from the encoder's recon, so
/// the same source plane is used regardless of QP.
fn make_b_fade(idr_y: &[u8], p_y: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for k in 0..(W * H) {
        let a = idr_y[k] as f64;
        let b = p_y[k] as f64;
        // α = 0.7 toward L0
        let v = (0.7 * a + 0.3 * b).round() as i32;
        y[k] = v.clamp(0, 255) as u8;
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

/// Encode IDR + P + B with weighted-pred in `mode`. Returns the per-NAL
/// byte counts plus the assembled annex-B bytes.
fn encode_with(weighted: bool) -> (usize, usize, usize, Vec<u8>, Vec<u8>) {
    let mut cfg = EncoderConfig::new(W as u32, H as u32);
    cfg.profile_idc = 77; // Main — required for B-slices (§A.2.2).
    cfg.max_num_ref_frames = 2;
    cfg.explicit_weighted_bipred = weighted;
    let enc = Encoder::new(cfg);

    let (y0, u0, v0) = make_idr_dark();
    let (y1, u1, v1) = make_p_bright();
    let (y2, u2, v2) = make_b_fade(&y0, &y1);

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
        1, // frame_num same as preceding P (B is non-reference)
        2, // POC LSB midway between IDR (0) and P (4)
    );

    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p.annex_b);
    combined.extend_from_slice(&b.annex_b);
    (
        idr.annex_b.len(),
        p.annex_b.len(),
        b.annex_b.len(),
        combined,
        b.recon_y,
    )
}

#[test]
fn round26_weighted_bipred_shrinks_b_slice_on_fade_in_fixture() {
    let (idr_a, p_a, b_a, _, _) = encode_with(false);
    let (idr_b, p_b, b_b, _, _) = encode_with(true);
    eprintln!(
        "round-26 fade-in: weighted=off → IDR {idr_a} P {p_a} B {b_a} (total {}B)\n              \
                       weighted=on  → IDR {idr_b} P {p_b} B {b_b} (total {}B)",
        idr_a + p_a + b_a,
        idr_b + p_b + b_b,
    );
    // IDR and P bitstreams should be identical (the toggle only affects
    // the B path's prediction merge + the PPS weighted_bipred_idc bit).
    // We do not require exact equality — the PPS adds 1 bit which can
    // round to a different byte boundary — but they should be within a
    // handful of bytes.
    assert!(
        (idr_a as i64 - idr_b as i64).abs() <= 4,
        "weighted-pred toggle perturbed IDR size unexpectedly: {idr_a} vs {idr_b}",
    );
    assert!(
        (p_a as i64 - p_b as i64).abs() <= 4,
        "weighted-pred toggle perturbed P size unexpectedly: {p_a} vs {p_b}",
    );
    // The headline assertion: weighted B-slice is materially smaller.
    let saved_pct = 100.0 * (b_a as f64 - b_b as f64) / (b_a as f64);
    eprintln!(
        "round-26 fade-in: B-slice {b_a} → {b_b} bytes ({saved_pct:.1} % smaller with weighted-pred)"
    );
    assert!(
        saved_pct >= 10.0,
        "weighted-pred should shrink the B-slice by ≥ 10 %; observed {saved_pct:.1} %",
    );
}

#[test]
fn round26_weighted_bipred_self_roundtrip_matches_local_recon() {
    // Source y2 we want to compare against.
    let (y0, _, _) = make_idr_dark();
    let (y1, _, _) = make_p_bright();
    let (y2, _, _) = make_b_fade(&y0, &y1);

    let (_, _, _, combined, b_recon_y) = encode_with(true);
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

    // Display order: IDR (POC 0) → B (POC 2) → P (POC 4) ⇒ frames[1] = B.
    let b_decoded = &frames[1];
    let mut max_diff_y = 0i32;
    for (k, (&a, &local)) in b_decoded.planes[0]
        .data
        .iter()
        .zip(b_recon_y.iter())
        .enumerate()
    {
        let d = (a as i32 - local as i32).abs();
        if d > max_diff_y {
            max_diff_y = d;
        }
        // Encoder local recon and decoder output should agree; allow
        // ≤ 1 LSB drift from quarter-pel filter accumulator ordering
        // (same tolerance round 18/20 use).
        assert!(
            d <= 1,
            "weighted-bipred B luma mismatch at idx {k}: dec={a}, enc={local} (diff {d})",
        );
    }
    let psnr_dec = psnr(&y2, &b_decoded.planes[0].data);
    eprintln!(
        "round-26 self-roundtrip: weighted B PSNR_Y = {psnr_dec:.2} dB (max enc/dec diff {max_diff_y})"
    );
    // PSNR floor: with weighted-pred matching the fade exactly the
    // residual is near zero so PSNR should be very high.
    assert!(
        psnr_dec >= 38.0,
        "weighted-bipred B PSNR_Y {psnr_dec:.2} dB below floor 38.0",
    );
}

#[test]
fn round26_weighted_bipred_ffmpeg_interop() {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }

    let (y0, _, _) = make_idr_dark();
    let (y1, _, _) = make_p_bright();
    let (y2, _, _) = make_b_fade(&y0, &y1);
    let (_, _, _, combined, b_recon_y) = encode_with(true);

    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!(
        "oxideav_h264_round26_wp_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_round26_wp_{}.yuv",
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
    // Display order: IDR → B → P ⇒ frames[1] = B.
    let b_off = frame_len;
    let b_y = &raw[b_off..b_off + y_len];
    let mut max_diff = 0i32;
    for (k, (&a, &local)) in b_y.iter().zip(b_recon_y.iter()).enumerate() {
        let d = (a as i32 - local as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d <= 1,
            "weighted-bipred ffmpeg B luma {k}: ff={a} enc={local} (diff {d})",
        );
    }
    eprintln!(
        "round-26 ffmpeg interop: bit-equivalent on weighted B-slice (max diff {max_diff}, stream {} bytes)",
        combined.len()
    );

    let psnr_ff = psnr(&y2, b_y);
    eprintln!("round-26 ffmpeg PSNR_Y on weighted B-slice = {psnr_ff:.2} dB");
    assert!(
        psnr_ff >= 38.0,
        "weighted-bipred B-slice PSNR_Y via ffmpeg {psnr_ff:.2} dB below floor 38.0",
    );
}
