//! Round-420 — MB-row QP modulation (`Encoder::encode_p_rate_adaptive`).
//!
//! The per-row controller steps the working QP toward a whole-slice
//! bit budget; QP changes ride §7.4.5 `mb_qp_delta` (present only on
//! MBs with coded residual — skipped / cbp==0 MBs inherit the
//! previous QP_Y, and the §8.7 deblock chain follows the decoder's
//! derivation). These tests use content with a vertical complexity
//! cliff so the row feedback has something to correct, and check:
//!
//! * our registry decoder reproduces the encoder recon byte-exactly
//!   (proves the emitted delta chain and the deblock-QP chain agree
//!   with the decoder);
//! * a black-box reference decoder agrees byte-exactly (when
//!   present);
//! * the adapted frame lands measurably closer to the budget than
//!   the flat-QP encode of the same picture.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use std::process::Command;

const FFMPEG: &str = "/opt/homebrew/bin/ffmpeg";

const W: u32 = 96;
const H: u32 = 96;

/// Deterministic xorshift PRNG so the "noisy" half is reproducible.
fn xorshift(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// Vertical complexity cliff: top half is a smooth gradient (cheap),
/// bottom half is dense noise (expensive). `n` shifts both so P
/// frames carry real residual everywhere.
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
    let dir = std::env::temp_dir().join(format!("oxideav-h264-r420m-{}", std::process::id()));
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

/// Encode IDR + adapted P + flat P; return everything needed by the
/// assertions.
struct CliffRun {
    stream_adapt: Vec<u8>,
    adapt_bits: u64,
    flat_bits: u64,
    recons: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
}

fn run_cliff(base_qp: i32, budget_bits: u64) -> CliffRun {
    let enc = Encoder::new(EncoderConfig::new(W, H));
    let (y0, u0, v0) = make_cliff_frame(0);
    let (y1, u1, v1) = make_cliff_frame(1);
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

    let idr = enc.encode_idr_with_qp(&f0, base_qp);
    let r = EncodedFrameRef::from(&idr);
    let p_adapt = enc.encode_p_rate_adaptive(&f1, &r, 1, 2, base_qp, budget_bits);
    let p_flat = enc.encode_p_with_qp(&f1, &r, 1, 2, base_qp);

    let mut stream_adapt = idr.annex_b.clone();
    stream_adapt.extend_from_slice(&p_adapt.annex_b);

    CliffRun {
        stream_adapt,
        adapt_bits: 8 * p_adapt.annex_b.len() as u64,
        flat_bits: 8 * p_flat.annex_b.len() as u64,
        recons: vec![
            (
                idr.recon_y.clone(),
                idr.recon_u.clone(),
                idr.recon_v.clone(),
            ),
            (
                p_adapt.recon_y.clone(),
                p_adapt.recon_u.clone(),
                p_adapt.recon_v.clone(),
            ),
        ],
    }
}

#[test]
fn tight_budget_rows_coarsen_and_land_closer() {
    // Flat-QP cost of this P frame at QP 28 is ~22 kbit (noisy
    // bottom half); a 12 kbit budget forces the later rows to climb.
    let run = run_cliff(28, 12_000);
    assert!(
        run.flat_bits > 15_000,
        "premise: flat encode must overshoot the budget, got {}",
        run.flat_bits
    );
    let d_adapt = run.adapt_bits.abs_diff(12_000);
    let d_flat = run.flat_bits.abs_diff(12_000);
    eprintln!(
        "tight budget: flat {} bits, adapted {} bits (budget 12000)",
        run.flat_bits, run.adapt_bits
    );
    assert!(
        d_adapt < d_flat,
        "row adaptation must land closer to budget: adapted {} vs flat {}",
        run.adapt_bits,
        run.flat_bits
    );

    // Decode both ways; recon must match byte-exactly.
    let own = decode_own(&run.stream_adapt);
    assert_eq!(own.len(), 2);
    for (n, (ry, ru, rv)) in run.recons.iter().enumerate() {
        assert_planes(&own[n], ry, ru, rv, &format!("tight f{n}"));
    }
    reference_agrees(&run.stream_adapt, &own, "row-qp-tight");
}

#[test]
fn generous_budget_rows_refine_and_spend_more() {
    // Budget far above the flat cost: rows step DOWN in QP and spend
    // quality bits.
    let run = run_cliff(40, 120_000);
    eprintln!(
        "generous budget: flat {} bits, adapted {} bits (budget 120000)",
        run.flat_bits, run.adapt_bits
    );
    assert!(
        run.adapt_bits > run.flat_bits,
        "under budget the rows must refine: adapted {} vs flat {}",
        run.adapt_bits,
        run.flat_bits
    );

    let own = decode_own(&run.stream_adapt);
    assert_eq!(own.len(), 2);
    for (n, (ry, ru, rv)) in run.recons.iter().enumerate() {
        assert_planes(&own[n], ry, ru, rv, &format!("generous f{n}"));
    }
    reference_agrees(&run.stream_adapt, &own, "row-qp-generous");
}

#[test]
fn skip_runs_across_qp_rows_stay_consistent() {
    // Static content: every P MB is a skip candidate. The row
    // controller may want a different QP but no coded MB exists to
    // carry the delta — the decoder keeps SliceQP_Y and so must the
    // encoder's deblock chain. Byte-exact roundtrip proves it.
    let enc = Encoder::new(EncoderConfig::new(W, H));
    let (y0, u0, v0) = make_cliff_frame(0);
    let f0 = YuvFrame {
        width: W,
        height: H,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let idr = enc.encode_idr_with_qp(&f0, 30);
    let r = EncodedFrameRef::from(&idr);
    // Same frame again → skips; absurdly tight budget.
    let p = enc.encode_p_rate_adaptive(&f0, &r, 1, 2, 30, 500);

    let mut stream = idr.annex_b.clone();
    stream.extend_from_slice(&p.annex_b);
    let own = decode_own(&stream);
    assert_eq!(own.len(), 2);
    assert_planes(&own[1], &p.recon_y, &p.recon_u, &p.recon_v, "skip-p");
    reference_agrees(&stream, &own, "row-qp-skip");
}
