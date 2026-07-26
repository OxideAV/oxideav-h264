//! Round-430 — MB-row QP modulation on the **CABAC P** path
//! (`Encoder::encode_p_cabac_rate_adaptive`) and on **IDR** pictures
//! (`Encoder::encode_idr_rate_adaptive` /
//! `Encoder::encode_idr_cabac_rate_adaptive`). Round 420 landed the
//! CAVLC-P controller only; these are the r420 follow-ups.
//!
//! What the byte-exact roundtrips prove:
//!
//! * CAVLC: the §7.4.5 `mb_qp_delta` chain (emitted only on MBs with
//!   coded residual; Intra_16x16 always) and the §8.7 deblock-QP chain
//!   match the decoder's derivation.
//! * CABAC: additionally the §9.3.3.1.1.5 context chain — ctxIdxInc
//!   (bin 0) of `mb_qp_delta` is 1 exactly when the previous MB in
//!   decoding order carried a NON-ZERO delta. Before round 430 every
//!   emitted delta was 0, so that context path was never exercised
//!   with a hot state; a mis-mirrored chain desynchronises the
//!   arithmetic decoder, which the byte-exact recon comparison (and
//!   the black-box reference decode) would catch immediately.
//!
//! Content: a vertical complexity cliff (smooth top half, noisy
//! bottom) so the per-row feedback has something to correct.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use std::process::Command;

const FFMPEG: &str = "/opt/homebrew/bin/ffmpeg";

const W: u32 = 96;
const H: u32 = 96;

fn xorshift(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// Vertical complexity cliff: top half smooth gradient, bottom half
/// dense noise. `n` shifts both so P frames carry residual everywhere.
fn make_cliff_frame(n: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (W as usize, H as usize);
    let mut y = vec![0u8; w * h];
    let mut u = vec![128u8; (w / 2) * (h / 2)];
    let mut v = vec![128u8; (w / 2) * (h / 2)];
    let mut rng = 0x1234_5678u32 ^ (n as u32).wrapping_mul(0x9E37_79B9);
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = if j < h / 2 {
                (40 + ((i + j + n * 3) % 60)) as u8
            } else {
                (xorshift(&mut rng) % 200 + 20) as u8
            };
        }
    }
    for j in 0..h / 2 {
        for i in 0..w / 2 {
            u[j * (w / 2) + i] = (110 + ((i + n) % 30)) as u8;
            v[j * (w / 2) + i] = (120 + ((j + n) % 30)) as u8;
        }
    }
    (y, u, v)
}

fn decode_own(stream: &[u8]) -> Vec<VideoFrame> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let packet = Packet::new(0, TimeBase::new(1, 30), stream.to_vec()).with_pts(0);
    dec.send_packet(&packet).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => frames.push(vf),
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    frames
}

fn assert_planes(vf: &VideoFrame, y: &[u8], u: &[u8], v: &[u8], tag: &str) {
    assert_eq!(vf.planes[0].data, y, "{tag}: luma");
    assert_eq!(vf.planes[1].data, u, "{tag}: cb");
    assert_eq!(vf.planes[2].data, v, "{tag}: cr");
}

fn reference_agrees(stream: &[u8], own: &[VideoFrame], tag: &str) {
    if !std::path::Path::new(FFMPEG).exists() {
        eprintln!("skip reference cross-check: binary not present");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h264-r430rq-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let h264 = dir.join(format!("{tag}.h264"));
    let yuv = dir.join(format!("{tag}.yuv"));
    std::fs::write(&h264, stream).expect("write");
    let status = Command::new(FFMPEG)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-y"])
        .arg(&yuv)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success(), "{tag}: reference decoder rejected stream");
    let raw = std::fs::read(&yuv).expect("read yuv");
    let fsz = (W as usize * H as usize) * 3 / 2;
    assert_eq!(raw.len(), fsz * own.len(), "{tag}: frame count");
    let ysz = W as usize * H as usize;
    for (n, vf) in own.iter().enumerate() {
        let b = n * fsz;
        assert_eq!(&raw[b..b + ysz], &vf.planes[0].data[..], "{tag}: f{n} luma");
        assert_eq!(
            &raw[b + ysz..b + ysz + ysz / 4],
            &vf.planes[1].data[..],
            "{tag}: f{n} cb"
        );
        assert_eq!(
            &raw[b + ysz + ysz / 4..b + fsz],
            &vf.planes[2].data[..],
            "{tag}: f{n} cr"
        );
    }
}

fn cabac_encoder() -> Encoder {
    let mut cfg = EncoderConfig::new(W, H);
    cfg.cabac = true;
    cfg.profile_idc = 77; // Main — CABAC requires >= Main
    Encoder::new(cfg)
}

type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

fn frames01() -> (Planes, Planes) {
    (make_cliff_frame(0), make_cliff_frame(1))
}

// ---------------------------------------------------------------------
// CABAC P path
// ---------------------------------------------------------------------

#[test]
fn cabac_p_tight_budget_rows_coarsen_and_land_closer() {
    let enc = cabac_encoder();
    let ((y0, u0, v0), (y1, u1, v1)) = frames01();
    let f0 = YuvFrame {
        width: W,
        height: H,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let f1 = YuvFrame {
        width: W,
        height: H,
        y: &y1,
        u: &u1,
        v: &v1,
    };

    let idr = enc.encode_idr_cabac_with_qp(&f0, 28);
    let r = EncodedFrameRef::from(&idr);
    let budget = 10_000u64;
    let p_flat = enc.encode_p_cabac_with_qp(&f1, &r, 1, 2, 28);
    let p_adapt = enc.encode_p_cabac_rate_adaptive(&f1, &r, 1, 2, 28, budget);
    let flat_bits = 8 * p_flat.annex_b.len() as u64;
    let adapt_bits = 8 * p_adapt.annex_b.len() as u64;
    eprintln!("cabac-p tight: flat {flat_bits} bits, adapted {adapt_bits} bits (budget {budget})");
    assert!(
        flat_bits > budget + budget / 4,
        "premise: flat CABAC encode must overshoot the budget, got {flat_bits}"
    );
    assert!(
        adapt_bits.abs_diff(budget) < flat_bits.abs_diff(budget),
        "row adaptation must land closer to budget: adapted {adapt_bits} vs flat {flat_bits}"
    );

    let mut stream = idr.annex_b.clone();
    stream.extend_from_slice(&p_adapt.annex_b);
    let own = decode_own(&stream);
    assert_eq!(own.len(), 2);
    assert_planes(&own[0], &idr.recon_y, &idr.recon_u, &idr.recon_v, "cp f0");
    assert_planes(
        &own[1],
        &p_adapt.recon_y,
        &p_adapt.recon_u,
        &p_adapt.recon_v,
        "cp f1",
    );
    reference_agrees(&stream, &own, "cabac-p-tight");
}

#[test]
fn cabac_p_skip_runs_across_qp_rows_stay_consistent() {
    // Static content: every P MB skips, so no MB can carry the delta —
    // the decoder keeps SliceQP_Y, mb_skip resets the §9.3.3.1.1.5
    // chain, and the encoder must mirror both. Byte-exact roundtrip
    // proves it.
    let enc = cabac_encoder();
    let ((y0, u0, v0), _) = frames01();
    let f0 = YuvFrame {
        width: W,
        height: H,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let idr = enc.encode_idr_cabac_with_qp(&f0, 30);
    let r = EncodedFrameRef::from(&idr);
    let p = enc.encode_p_cabac_rate_adaptive(&f0, &r, 1, 2, 30, 500);

    let mut stream = idr.annex_b.clone();
    stream.extend_from_slice(&p.annex_b);
    let own = decode_own(&stream);
    assert_eq!(own.len(), 2);
    assert_planes(&own[1], &p.recon_y, &p.recon_u, &p.recon_v, "cp-skip f1");
    reference_agrees(&stream, &own, "cabac-p-skip");
}

// ---------------------------------------------------------------------
// IDR row modulation (CAVLC + CABAC)
// ---------------------------------------------------------------------

#[test]
fn cavlc_idr_tight_budget_rows_coarsen_and_land_closer() {
    let enc = Encoder::new(EncoderConfig::new(W, H));
    let ((y0, u0, v0), _) = frames01();
    let f0 = YuvFrame {
        width: W,
        height: H,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let budget = 14_000u64;
    let flat = enc.encode_idr_with_qp(&f0, 28);
    let adapt = enc.encode_idr_rate_adaptive(&f0, 28, budget);
    let flat_bits = 8 * flat.annex_b.len() as u64;
    let adapt_bits = 8 * adapt.annex_b.len() as u64;
    eprintln!(
        "cavlc-idr tight: flat {flat_bits} bits, adapted {adapt_bits} bits (budget {budget})"
    );
    assert!(
        flat_bits > budget + budget / 4,
        "premise: flat IDR must overshoot the budget, got {flat_bits}"
    );
    assert!(
        adapt_bits.abs_diff(budget) < flat_bits.abs_diff(budget),
        "IDR row adaptation must land closer to budget: adapted {adapt_bits} vs flat {flat_bits}"
    );

    let own = decode_own(&adapt.annex_b);
    assert_eq!(own.len(), 1);
    assert_planes(
        &own[0],
        &adapt.recon_y,
        &adapt.recon_u,
        &adapt.recon_v,
        "ci",
    );
    reference_agrees(&adapt.annex_b, &own, "cavlc-idr-tight");
}

#[test]
fn cavlc_idr_generous_budget_rows_refine() {
    let enc = Encoder::new(EncoderConfig::new(W, H));
    let ((y0, u0, v0), _) = frames01();
    let f0 = YuvFrame {
        width: W,
        height: H,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let flat = enc.encode_idr_with_qp(&f0, 40);
    let adapt = enc.encode_idr_rate_adaptive(&f0, 40, 200_000);
    let flat_bits = 8 * flat.annex_b.len() as u64;
    let adapt_bits = 8 * adapt.annex_b.len() as u64;
    eprintln!("cavlc-idr generous: flat {flat_bits} bits, adapted {adapt_bits} bits");
    assert!(
        adapt_bits > flat_bits,
        "under budget the rows must refine: adapted {adapt_bits} vs flat {flat_bits}"
    );
    let own = decode_own(&adapt.annex_b);
    assert_eq!(own.len(), 1);
    assert_planes(
        &own[0],
        &adapt.recon_y,
        &adapt.recon_u,
        &adapt.recon_v,
        "ci-gen",
    );
    reference_agrees(&adapt.annex_b, &own, "cavlc-idr-generous");
}

#[test]
fn cabac_idr_tight_budget_rows_coarsen_and_land_closer() {
    let enc = cabac_encoder();
    let ((y0, u0, v0), _) = frames01();
    let f0 = YuvFrame {
        width: W,
        height: H,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let budget = 13_000u64;
    let flat = enc.encode_idr_cabac_with_qp(&f0, 28);
    let adapt = enc.encode_idr_cabac_rate_adaptive(&f0, 28, budget);
    let flat_bits = 8 * flat.annex_b.len() as u64;
    let adapt_bits = 8 * adapt.annex_b.len() as u64;
    eprintln!(
        "cabac-idr tight: flat {flat_bits} bits, adapted {adapt_bits} bits (budget {budget})"
    );
    assert!(
        flat_bits > budget + budget / 4,
        "premise: flat CABAC IDR must overshoot the budget, got {flat_bits}"
    );
    assert!(
        adapt_bits.abs_diff(budget) < flat_bits.abs_diff(budget),
        "CABAC IDR row adaptation must land closer: adapted {adapt_bits} vs flat {flat_bits}"
    );

    let own = decode_own(&adapt.annex_b);
    assert_eq!(own.len(), 1);
    assert_planes(
        &own[0],
        &adapt.recon_y,
        &adapt.recon_u,
        &adapt.recon_v,
        "cbi",
    );
    reference_agrees(&adapt.annex_b, &own, "cabac-idr-tight");
}

/// The IDR + adapted-CABAC-P pair through a rate-controlled session
/// shape: IDR at one QP, P rows modulating, all decoded byte-exactly
/// by both decoders. Exercises non-zero deltas on BOTH slice kinds of
/// one stream (the §9.3.3.1.1.5 chain resets per slice).
#[test]
fn cabac_idr_plus_p_adapted_stream_roundtrips() {
    let enc = cabac_encoder();
    let ((y0, u0, v0), (y1, u1, v1)) = frames01();
    let f0 = YuvFrame {
        width: W,
        height: H,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let f1 = YuvFrame {
        width: W,
        height: H,
        y: &y1,
        u: &u1,
        v: &v1,
    };
    let idr = enc.encode_idr_cabac_rate_adaptive(&f0, 30, 16_000);
    let r = EncodedFrameRef::from(&idr);
    let p = enc.encode_p_cabac_rate_adaptive(&f1, &r, 1, 2, 30, 9_000);

    let mut stream = idr.annex_b.clone();
    stream.extend_from_slice(&p.annex_b);
    let own = decode_own(&stream);
    assert_eq!(own.len(), 2);
    assert_planes(&own[0], &idr.recon_y, &idr.recon_u, &idr.recon_v, "mix f0");
    assert_planes(&own[1], &p.recon_y, &p.recon_u, &p.recon_v, "mix f1");
    reference_agrees(&stream, &own, "cabac-mixed-adapted");
}
