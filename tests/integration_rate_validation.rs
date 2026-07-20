//! Round-420 — rate-control validation matrix.
//!
//! Measures, over a matrix of synthetic signals:
//!
//! * **rate accuracy** — |actual − target| / target of the payload
//!   bitrate after the first-GOP warmup, for CBR and capped VBR;
//! * **rate-distortion sanity** — a CBR encode is compared against
//!   fixed-QP anchor encodes of the same content: at a comparable
//!   rate the rate-controlled stream must not lose more than a
//!   bounded PSNR_Y margin versus the interpolated fixed-QP curve
//!   (rate control redistributes bits, it must not squander them);
//! * **decodability** — every stream decodes through our registry
//!   decoder (frame count checked; byte-exactness of the recon chain
//!   is covered by the session / row-QP suites).
//!
//! The measured numbers print with `--nocapture` and feed the crate
//! README's rate-control section.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::rate_control::RateControlConfig;
use oxideav_h264::encoder::session::{EncoderSession, SessionConfig};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: u32 = 80;
const H: u32 = 64;
const FPS: f64 = 30.0;

fn xorshift(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Signal {
    /// Smooth moving gradient — cheap, highly predictable.
    Smooth,
    /// Textured noise field with global motion — expensive.
    Texture,
    /// Smooth content with a hard scene cut every 25 frames.
    SceneCut,
}

fn make_frame(sig: Signal, n: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (W as usize, H as usize);
    let mut y = vec![0u8; w * h];
    let mut u = vec![128u8; (w / 2) * (h / 2)];
    let mut v = vec![128u8; (w / 2) * (h / 2)];
    match sig {
        Signal::Smooth => {
            for j in 0..h {
                for i in 0..w {
                    y[j * w + i] = (40 + ((i * 2 + j + n * 3) % 160)) as u8;
                }
            }
        }
        Signal::Texture => {
            // Per-frame reseeded noise with a deterministic phase so
            // motion compensation finds partial matches only.
            let mut rng = 0xC0FF_EE00u32 ^ ((n as u32 / 4).wrapping_mul(0x9E37_79B9));
            let shift = n % 4;
            for j in 0..h {
                for i in 0..w {
                    let base = (xorshift(&mut rng) % 160 + 40) as u8;
                    y[j * w + (i + shift) % w] = base;
                }
            }
        }
        Signal::SceneCut => {
            let scene = n / 25;
            let phase = scene * 61;
            for j in 0..h {
                for i in 0..w {
                    y[j * w + i] =
                        (30 + ((i * (1 + scene % 3) + j * 2 + n * 2 + phase) % 170)) as u8;
                }
            }
        }
    }
    for j in 0..h / 2 {
        for i in 0..w / 2 {
            u[j * (w / 2) + i] = (100 + ((i + n) % 40)) as u8;
            v[j * (w / 2) + i] = (125 + ((j + n * 2) % 35)) as u8;
        }
    }
    (y, u, v)
}

struct RunResult {
    payload_bits_total: u64,
    payload_bits_tail: u64,
    /// Payload + CBR filler over the tail window.
    sent_bits_tail: u64,
    /// Modelled CPB fullness at the start / end of the tail window
    /// (rate-controlled sessions only).
    fullness_at_skip: f64,
    fullness_end: f64,
    tail_frames: usize,
    frames: usize,
    psnr_y: f64,
}

/// Encode `frames` frames of `sig` through `session`, decode the whole
/// stream with our decoder, and return rate + PSNR_Y stats. `skip`
/// frames are excluded from the tail rate measurement (warmup).
fn run_and_measure(
    mut session: EncoderSession,
    sig: Signal,
    frames: usize,
    skip: usize,
) -> RunResult {
    let mut stream = Vec::new();
    let mut sources: Vec<Vec<u8>> = Vec::new();
    let mut total = 0u64;
    let mut tail = 0u64;
    let mut sent_tail = 0u64;
    let mut fullness_at_skip = 0.0f64;
    for n in 0..frames {
        if n == skip {
            fullness_at_skip = session
                .rate_controller()
                .map(|rc| rc.cpb_fullness())
                .unwrap_or(0.0);
        }
        let (y, u, v) = make_frame(sig, n);
        let sf = session.encode_frame(&y, &u, &v);
        total += sf.payload_bits;
        if n >= skip {
            tail += sf.payload_bits;
            sent_tail += sf.payload_bits + sf.filler_bits;
        }
        stream.extend_from_slice(&sf.annex_b);
        sources.push(y);
    }
    let fullness_end = session
        .rate_controller()
        .map(|rc| rc.cpb_fullness())
        .unwrap_or(0.0);

    // Decode with our decoder; PSNR_Y against the source.
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let packet = Packet::new(0, TimeBase::new(1, 30), stream).with_pts(0);
    dec.send_packet(&packet).expect("send_packet");
    dec.flush().expect("flush");
    let mut decoded: Vec<VideoFrame> = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => decoded.push(vf),
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    assert_eq!(decoded.len(), frames, "decoder frame count for {sig:?}");

    let mut sse = 0u64;
    for (src, vf) in sources.iter().zip(decoded.iter()) {
        for (a, b) in src.iter().zip(vf.planes[0].data.iter()) {
            let d = i64::from(*a) - i64::from(*b);
            sse += (d * d) as u64;
        }
    }
    let n_samples = (frames as u64) * u64::from(W) * u64::from(H);
    let mse = sse as f64 / n_samples as f64;
    let psnr_y = if mse > 0.0 {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    } else {
        f64::INFINITY
    };

    RunResult {
        payload_bits_total: total,
        payload_bits_tail: tail,
        sent_bits_tail: sent_tail,
        fullness_at_skip,
        fullness_end,
        tail_frames: frames - skip,
        frames,
        psnr_y,
    }
}

fn cbr_session(bitrate: u32) -> EncoderSession {
    EncoderSession::new(SessionConfig::rate_controlled(
        W,
        H,
        RateControlConfig::cbr(bitrate, 30, 1),
    ))
}

#[test]
fn rate_accuracy_matrix() {
    // (signal, mode-label, target bps, session)
    let cases: Vec<(Signal, &str, u32, EncoderSession)> = vec![
        // Note: the smooth signal saturates around ~65 kbps even at
        // min QP; its 50 kbps target is deliberately inside what the
        // content can absorb, while the CBR *channel* check below
        // covers the fill-to-rate contract regardless.
        (Signal::Smooth, "cbr", 50_000, cbr_session(50_000)),
        (Signal::Texture, "cbr", 300_000, cbr_session(300_000)),
        // SceneCut saturates near ~83 kbps at min QP (smooth scenes
        // between the cuts) — 60 kbps keeps the target absorbable.
        (Signal::SceneCut, "cbr", 60_000, cbr_session(60_000)),
        (
            Signal::Texture,
            "vbr",
            200_000,
            EncoderSession::new(SessionConfig::rate_controlled(
                W,
                H,
                RateControlConfig::capped_vbr(200_000, 400_000, 30, 1),
            )),
        ),
        (
            Signal::SceneCut,
            "vbr",
            60_000,
            EncoderSession::new(SessionConfig::rate_controlled(
                W,
                H,
                RateControlConfig::capped_vbr(60_000, 120_000, 30, 1),
            )),
        ),
    ];

    let mut worst = 0.0f64;
    for (sig, label, target, session) in cases {
        let r = run_and_measure(session, sig, 75, 30);
        let seconds = r.tail_frames as f64 / FPS;
        let rate = r.payload_bits_tail as f64 / seconds;
        let err = (rate - f64::from(target)).abs() / f64::from(target);
        eprintln!(
            "{label} {sig:?} target {target}: payload {rate:.0} bps err {:.2}% PSNR_Y {:.2} dB",
            err * 100.0,
            r.psnr_y
        );
        if label == "vbr" {
            // Capped VBR: the integrator drives the raw payload
            // average itself onto the target.
            assert!(
                err < 0.10,
                "{label} {sig:?} rate err {err:.4} exceeds 10% (rate {rate:.0} vs {target})"
            );
            worst = worst.max(err);
        } else {
            // CBR: over a finite window the raw payload rate is
            // dominated by the bucket position moving between the cap
            // and the 55% setpoint (banked channel bits being spent /
            // repaid — on-rate behaviour by Annex C). The controller
            // metric is the drift-corrected payload rate, whose error
            // versus target is exactly the filler share.
            let corrected =
                (r.payload_bits_tail as f64 - (r.fullness_at_skip - r.fullness_end)) / seconds;
            let cerr = (corrected - f64::from(target)).abs() / f64::from(target);
            eprintln!(
                "  cbr drift-corrected payload: {corrected:.0} bps err {:.2}%",
                cerr * 100.0
            );
            assert!(
                cerr < 0.06,
                "{label} {sig:?} corrected rate err {cerr:.4} exceeds 6%"
            );
            worst = worst.max(cerr);
        }
        if label == "cbr" {
            // CBR channel contract via Annex C bucket conservation:
            // over the tail window, payload + filler must equal the
            // channel arrivals plus the net bucket drain (fullness is
            // decoder-side: draining the bucket means the encoder
            // spent banked channel bits — still on-rate).
            let expected_sent = f64::from(target) * seconds + (r.fullness_at_skip - r.fullness_end);
            let sent = r.sent_bits_tail as f64;
            let sent_err = (sent - expected_sent).abs() / (f64::from(target) * seconds);
            eprintln!(
                "  cbr channel (payload+filler+bucket): err {:.2}%",
                sent_err * 100.0
            );
            assert!(
                sent_err < 0.02,
                "{label} {sig:?} channel conservation err {sent_err:.4} exceeds 2%"
            );
        }
    }
    eprintln!(
        "worst-case post-warmup payload rate error: {:.2}%",
        worst * 100.0
    );
}

#[test]
fn rate_control_holds_the_rd_curve() {
    // Fixed-QP anchors on the scene-cut signal (IDR + P, gop 30 — the
    // same GOP shape the controlled session uses).
    let mut anchors: Vec<(f64, f64)> = Vec::new(); // (bps, psnr_y)
    for qp in [26, 30, 34] {
        let mut cfg = SessionConfig::constant_qp(W, H, qp);
        cfg.gop_length = 30;
        let r = run_and_measure(EncoderSession::new(cfg), Signal::SceneCut, 60, 0);
        let bps = r.payload_bits_total as f64 / (r.frames as f64 / FPS);
        eprintln!("anchor QP {qp}: {bps:.0} bps, PSNR_Y {:.2} dB", r.psnr_y);
        anchors.push((bps, r.psnr_y));
    }
    // Anchors are rate-descending with QP; sanity-check monotony.
    assert!(anchors[0].0 > anchors[1].0 && anchors[1].0 > anchors[2].0);
    assert!(anchors[0].1 > anchors[1].1 && anchors[1].1 > anchors[2].1);

    // CBR at the middle anchor's rate.
    let target = anchors[1].0.round() as u32;
    let r = run_and_measure(cbr_session(target), Signal::SceneCut, 60, 0);
    let cbr_bps = r.payload_bits_total as f64 / (r.frames as f64 / FPS);
    eprintln!(
        "CBR @ {target} bps: {cbr_bps:.0} bps, PSNR_Y {:.2} dB (anchor {:.2} dB)",
        r.psnr_y, anchors[1].1
    );

    // Interpolate the anchor RD curve at the CBR's achieved rate
    // (log-rate linear interpolation between the two bracketing
    // anchors) and require the controlled encode to stay within a
    // bounded PSNR margin below the curve.
    let curve_psnr = {
        let (lo, hi) = if cbr_bps <= anchors[1].0 {
            (anchors[2], anchors[1])
        } else {
            (anchors[1], anchors[0])
        };
        let t = ((cbr_bps.ln() - lo.0.ln()) / (hi.0.ln() - lo.0.ln())).clamp(0.0, 1.0);
        lo.1 + t * (hi.1 - lo.1)
    };
    let loss = curve_psnr - r.psnr_y;
    eprintln!(
        "RD check: curve {curve_psnr:.2} dB, CBR {:.2} dB, loss {loss:.2} dB",
        r.psnr_y
    );
    assert!(
        loss < 1.5,
        "rate control lost {loss:.2} dB versus the fixed-QP RD curve"
    );
    // And the rate itself must have been respected.
    let err = (cbr_bps - f64::from(target)).abs() / f64::from(target);
    assert!(err < 0.12, "RD-check CBR rate err {err:.4}");
}
