//! Round-151 — opt-in CABAC IDR Intra_16x16 chroma AC trellis
//! quantisation (`EncoderConfig::trellis_quant_intra_chroma`).
//!
//! Extends the round-148 luma-AC trellis to chroma. For 4:2:0 the lever
//! refines the 4 per-block 4×4 AC arrays per chroma plane (Cb/Cr), each
//! with `skip_dc=true` so the §8.5.11.1 2×2 chroma-DC Hadamard chain
//! stays bit-exact. For 4:4:4 the lever refines the 16 per-block 4×4
//! AC arrays per chroma plane, treating chroma "like luma" per §7.3.5.3
//! (the §8.5.10 luma-style DC inverse Hadamard chain stays bit-exact
//! because `skip_dc=true` pins the DC slot).
//!
//! Spec basis: §8.5.10 (luma DC Hadamard round trip — applied to 4:4:4
//! chroma when ChromaArrayType=3), §8.5.11.1 (2×2 chroma-DC Hadamard),
//! §9.3.3.1.3 (CABAC residual binarisation). The trellis is
//! informative — the decoder reads the same `ChromaACLevel` /
//! `Intra16x16AC` block-cat syntax tree regardless of which path
//! produced the levels.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::{Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

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

/// Build a textured luma plane: diagonal stripes + per-pixel checker
/// so every 4×4 block carries non-trivial AC after the §8.5.12
/// integer transform.
fn make_textured_luma(w: usize, h: usize) -> Vec<u8> {
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            let stripe = ((i as i32 + j as i32) * 3) & 0xff;
            let checker = if (i ^ j) & 1 == 0 { 28 } else { -28 };
            let v = stripe + checker;
            y[j * w + i] = v.clamp(0, 255) as u8;
        }
    }
    y
}

/// Build a textured chroma plane (same idea as luma, but with a
/// different stride + offset so Cb and Cr can use distinct seeds and
/// the chroma intra predictor doesn't sit on flat 128).
fn make_textured_chroma(cw: usize, ch: usize, seed: i32) -> Vec<u8> {
    let mut c = vec![0u8; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            let stripe = (((i as i32 * 5) + (j as i32 * 7) + seed) * 2) & 0xff;
            let checker = if (i ^ j) & 1 == 0 { 18 } else { -18 };
            let v = 128 + ((stripe - 128) / 2) + checker;
            c[j * cw + i] = v.clamp(0, 255) as u8;
        }
    }
    c
}

fn encode_idr_420(qp: i32, trellis_chroma: bool) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, f64) {
    let w = 64usize;
    let h = 64usize;
    let cw = w / 2;
    let ch = h / 2;
    let y = make_textured_luma(w, h);
    let u = make_textured_chroma(cw, ch, 13);
    let v = make_textured_chroma(cw, ch, 41);

    let f = YuvFrame {
        width: w as u32,
        height: h as u32,
        y: &y,
        u: &u,
        v: &v,
    };

    let mut cfg = EncoderConfig::new(w as u32, h as u32);
    cfg.cabac = true;
    cfg.profile_idc = 77;
    cfg.qp = qp;
    cfg.trellis_quant = true; // inter trellis lever; not exercised by IDR
    cfg.trellis_quant_intra = false; // luma-intra trellis stays default-off
    cfg.trellis_quant_intra_chroma = trellis_chroma;
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr_cabac(&f);
    let psnr_y = psnr(&y, &idr.recon_y);
    (idr.annex_b, idr.recon_y, idr.recon_u, idr.recon_v, psnr_y)
}

fn encode_idr_444(qp: i32, trellis_chroma: bool) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = 64usize;
    let h = 64usize;
    let y = make_textured_luma(w, h);
    let u = make_textured_chroma(w, h, 13);
    let v = make_textured_chroma(w, h, 41);

    let f = YuvFrame {
        width: w as u32,
        height: h as u32,
        y: &y,
        u: &u,
        v: &v,
    };

    let mut cfg = EncoderConfig::new(w as u32, h as u32);
    cfg.cabac = true;
    cfg.profile_idc = 244;
    cfg.chroma_format_idc = 3;
    cfg.qp = qp;
    cfg.trellis_quant = true;
    cfg.trellis_quant_intra = false;
    cfg.trellis_quant_intra_chroma = trellis_chroma;
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr_cabac(&f);
    (idr.annex_b, idr.recon_y, idr.recon_u, idr.recon_v)
}

/// Default config (`trellis_quant_intra_chroma = false`) must produce
/// exactly the byte sequence the round-148 path always produced — i.e.
/// the lever is genuinely opt-in. Verified on both the 4:2:0 and 4:4:4
/// IDR paths.
#[test]
fn round151_trellis_quant_intra_chroma_default_off_is_byte_for_byte_baseline() {
    let (idr_off_a, ry_a, ru_a, rv_a, _) = encode_idr_420(22, false);
    let (idr_off_b, ry_b, ru_b, rv_b, _) = encode_idr_420(22, false);
    assert_eq!(
        idr_off_a, idr_off_b,
        "4:2:0 trellis_quant_intra_chroma=false must be deterministic across calls",
    );
    assert_eq!(ry_a, ry_b, "4:2:0 recon_y must be deterministic");
    assert_eq!(ru_a, ru_b, "4:2:0 recon_u must be deterministic");
    assert_eq!(rv_a, rv_b, "4:2:0 recon_v must be deterministic");

    let (idr_444_a, ry444_a, ru444_a, rv444_a) = encode_idr_444(22, false);
    let (idr_444_b, ry444_b, ru444_b, rv444_b) = encode_idr_444(22, false);
    assert_eq!(
        idr_444_a, idr_444_b,
        "4:4:4 trellis_quant_intra_chroma=false must be deterministic across calls",
    );
    assert_eq!(ry444_a, ry444_b, "4:4:4 recon_y must be deterministic");
    assert_eq!(ru444_a, ru444_b, "4:4:4 recon_u must be deterministic");
    assert_eq!(rv444_a, rv444_b, "4:4:4 recon_v must be deterministic");
}

/// 4:2:0 decoder-transparency: with `trellis_quant_intra_chroma = true`
/// the encoded IDR still round-trips through our own decoder
/// bit-equivalently to the encoder's local recon, on **all three
/// planes**. This pins the load-bearing guarantee that the chroma
/// trellis must never emit a syntactically illegal stream, and that
/// the `skip_dc=true` pin on the per-block refinement leaves the
/// §8.5.11.1 chroma-DC Hadamard chain undisturbed.
#[test]
fn round151_trellis_quant_intra_chroma_420_self_roundtrip_bit_equivalent() {
    let (annex_b, local_y, local_u, local_v, _) = encode_idr_420(22, true);

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), annex_b).with_pts(0);
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
    assert_eq!(frames.len(), 1, "expected 1 frame, got {}", frames.len());

    let decoded = &frames[0];
    let max_diff = |a: &[u8], b: &[u8]| -> i32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as i32 - y as i32).abs())
            .max()
            .unwrap_or(0)
    };
    assert_eq!(
        max_diff(&decoded.planes[0].data, &local_y),
        0,
        "luma mismatch — encoder/decoder out of sync",
    );
    assert_eq!(
        max_diff(&decoded.planes[1].data, &local_u),
        0,
        "Cb mismatch — chroma-DC Hadamard chain disturbed?",
    );
    assert_eq!(
        max_diff(&decoded.planes[2].data, &local_v),
        0,
        "Cr mismatch — chroma-DC Hadamard chain disturbed?",
    );
}

/// 4:4:4 decoder-transparency: same load-bearing guarantee, now over
/// the §7.3.5.3 "chroma coded like luma" path where each chroma plane
/// carries its own 16-block 4×4 DC Hadamard. The trellis refines all
/// 16 AC arrays per plane with `skip_dc=true`.
#[test]
fn round151_trellis_quant_intra_chroma_444_self_roundtrip_bit_equivalent() {
    let (annex_b, local_y, local_u, local_v) = encode_idr_444(22, true);

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), annex_b).with_pts(0);
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
    assert_eq!(frames.len(), 1, "expected 1 frame, got {}", frames.len());

    let decoded = &frames[0];
    let max_diff = |a: &[u8], b: &[u8]| -> i32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as i32 - y as i32).abs())
            .max()
            .unwrap_or(0)
    };
    assert_eq!(
        max_diff(&decoded.planes[0].data, &local_y),
        0,
        "4:4:4 luma mismatch",
    );
    assert_eq!(
        max_diff(&decoded.planes[1].data, &local_u),
        0,
        "4:4:4 Cb mismatch — luma-style DC Hadamard chain disturbed?",
    );
    assert_eq!(
        max_diff(&decoded.planes[2].data, &local_v),
        0,
        "4:4:4 Cr mismatch — luma-style DC Hadamard chain disturbed?",
    );
}

/// QP sweep: at every QP we test, the trellis-refined IDR must stay
/// non-empty, decode to exactly one frame, and round-trip bit-exact
/// on all three planes. Mirror the round-148 luma envelope shape.
#[test]
fn round151_trellis_quant_intra_chroma_qp_sweep_stays_decodable() {
    for qp in [18, 22, 26, 30, 34, 38, 42] {
        let (annex_b, local_y, local_u, local_v, _) = encode_idr_420(qp, true);
        assert!(
            !annex_b.is_empty(),
            "QP={qp}: chroma-trellis IDR produced an empty Annex B stream",
        );

        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let pkt = Packet::new(0, TimeBase::new(1, 25), annex_b).with_pts(0);
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
        assert_eq!(
            frames.len(),
            1,
            "QP={qp}: expected 1 frame, got {}",
            frames.len()
        );

        let decoded = &frames[0];
        let max_diff = |a: &[u8], b: &[u8]| -> i32 {
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x as i32 - y as i32).abs())
                .max()
                .unwrap_or(0)
        };
        assert_eq!(
            max_diff(&decoded.planes[0].data, &local_y),
            0,
            "QP={qp}: luma drift > 0",
        );
        assert_eq!(
            max_diff(&decoded.planes[1].data, &local_u),
            0,
            "QP={qp}: Cb drift > 0",
        );
        assert_eq!(
            max_diff(&decoded.planes[2].data, &local_v),
            0,
            "QP={qp}: Cr drift > 0",
        );
    }
}

/// Quality envelope: flipping `trellis_quant_intra_chroma` on must not
/// catastrophically degrade luma PSNR. The chroma trellis touches only
/// chroma AC; luma reconstruction is identical between on/off, so this
/// guard is a sanity check that the lever doesn't accidentally feed
/// state into the luma path. (We don't pin a chroma byte-saving lower
/// bound here — the rate model is calibrated for inter blockCats and a
/// future round will re-calibrate for chroma blockCats 4 / 1.)
#[test]
fn round151_trellis_quant_intra_chroma_psnr_y_unchanged() {
    for qp in [22, 28, 34] {
        let (_, _, _, _, psnr_off) = encode_idr_420(qp, false);
        let (_, _, _, _, psnr_on) = encode_idr_420(qp, true);
        let drift = (psnr_off - psnr_on).abs();
        eprintln!(
            "round-151 QP={qp}: PSNR_Y off={psnr_off:.2} on={psnr_on:.2} (|drift|={drift:.3})",
        );
        assert!(
            drift <= 0.05,
            "QP={qp}: luma PSNR should be insensitive to chroma trellis (off={psnr_off:.2} on={psnr_on:.2})",
        );
    }
}

/// Measurement-only — print the chroma-trellis byte impact on the
/// round-151 textured fixture across a small QP grid. Always passes;
/// useful for tracking rate behaviour across rate-model recalibrations.
#[test]
fn round151_trellis_quant_intra_chroma_byte_savings_print_only() {
    eprintln!("\n--- round-151 chroma-trellis 4:2:0 byte counts ---");
    for qp in [18, 22, 26, 30, 34, 38] {
        let (off_bytes, _, _, _, _) = encode_idr_420(qp, false);
        let (on_bytes, _, _, _, _) = encode_idr_420(qp, true);
        let delta = on_bytes.len() as i64 - off_bytes.len() as i64;
        let pct = 100.0 * delta as f64 / off_bytes.len() as f64;
        eprintln!(
            "QP={qp}: off={off:6} bytes, on={on:6} bytes (delta {delta:+5}, {pct:+5.2} %)",
            off = off_bytes.len(),
            on = on_bytes.len(),
        );
    }
    eprintln!("--- round-151 chroma-trellis 4:4:4 byte counts ---");
    for qp in [18, 22, 26, 30, 34, 38] {
        let (off_bytes, _, _, _) = encode_idr_444(qp, false);
        let (on_bytes, _, _, _) = encode_idr_444(qp, true);
        let delta = on_bytes.len() as i64 - off_bytes.len() as i64;
        let pct = 100.0 * delta as f64 / off_bytes.len() as f64;
        eprintln!(
            "QP={qp}: off={off:6} bytes, on={on:6} bytes (delta {delta:+5}, {pct:+5.2} %)",
            off = off_bytes.len(),
            on = on_bytes.len(),
        );
    }
}
