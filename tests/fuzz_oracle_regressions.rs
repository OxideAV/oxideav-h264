//! Regressions for the `ffmpeg_oracle_decode` differential-fuzz harness
//! premises and the decode-side strictness fixes they drove
//! (2026-07-12 .. 2026-07-16 scheduled-Fuzz reds + local refuzz).
//!
//! The harness classifies our decoder's output before asserting parity
//! against the reference decoder:
//!
//! * output produced while `decode_error_count() > 0` is
//!   partial/concealed — exempt from parity checks and never
//!   pixel-compared;
//! * streams whose stored SPSes declare >8-bit or
//!   non-planar-comparable chroma are incomparable (the oracle wrapper
//!   only represents 8-bit planar YUV);
//! * "reference rejected, we decoded cleanly" is advisory (the
//!   reference layers stream-discipline heuristics beyond bitstream
//!   conformance).
//!
//! The fixtures under `tests/fixtures/fuzz_regressions/` are the
//! fuzzer-found inputs from the failed scheduled runs plus the local
//! refuzz that followed the harness fix. Each one pins a decoder-side
//! property the harness classification relies on.

use oxideav_core::{CodecId, Decoder, Frame, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;

/// What the decoder must do with an archived fuzzer input.
enum Expect {
    /// Any frame squeezed out of the stream must be flagged: skipped
    /// slices counted (`decode_error_count > 0`) or an SPS outside the
    /// 8-bit-planar comparison domain. A clean comparable decode of
    /// these streams would be a real differential-fuzz divergence.
    FlaggedIfFrames,
    /// The stream must register decode errors (its defect is a
    /// conformance violation the decoder now refuses: §8.3.x intra
    /// mode with unavailable neighbours, §8.2.4 empty-DPB inter slice,
    /// slice-data-partition NALs, ...).
    Errors,
    /// Decoding cleanly is accepted behaviour (reference-side
    /// heuristic strictness only, e.g. refusing non-paired fields);
    /// the pin is just "no panic".
    NoPanic,
}

const FIXTURES: &[(&str, &[u8], Expect)] = &[
    // -- 2026-07-12 .. 07-16 scheduled-run artifacts (oracle was bound
    //    to the wrong decoder; these are hostile streams our decoder
    //    salvages degraded/incomparable output from) ----------------
    (
        "28d2a5c6 (High 4:4:4 9-bit, broken slices)",
        include_bytes!("fixtures/fuzz_regressions/crash-28d2a5c63be321cb8e69303316d1d79ec32f91aa"),
        Expect::FlaggedIfFrames,
    ),
    (
        "8f49a0cf",
        include_bytes!("fixtures/fuzz_regressions/crash-8f49a0cffd9443dc4a4389d043a9069fa4fbd5e8"),
        Expect::FlaggedIfFrames,
    ),
    (
        "df32d506",
        include_bytes!("fixtures/fuzz_regressions/crash-df32d5060d483d08cebb46452646d81c6c253546"),
        Expect::FlaggedIfFrames,
    ),
    (
        "f57b1693",
        include_bytes!("fixtures/fuzz_regressions/crash-f57b16934e6a53f10e92048ee476e1145333ff86"),
        Expect::FlaggedIfFrames,
    ),
    // -- local refuzz after the oracle fix: decode-side strictness
    //    gaps, each now enforced with a spec cite -------------------
    (
        "560624b8 (§8.3.x intra mode needs unavailable neighbours)",
        include_bytes!("fixtures/fuzz_regressions/crash-560624b82fc3234c42284c67e95b3b70a971ee55"),
        Expect::Errors,
    ),
    (
        "874295dc (§8.2.4 P slice with empty reference DPB)",
        include_bytes!("fixtures/fuzz_regressions/crash-874295dc5eb97498eabc3cfa844f74af658f8540"),
        Expect::Errors,
    ),
    (
        "c9b7f418 (slice data partition NALs, unimplemented VCL content)",
        include_bytes!("fixtures/fuzz_regressions/crash-c9b7f418e56b1d191950778dc6507f50dedf5010"),
        Expect::Errors,
    ),
    (
        "b741d946 (non-paired fields; reference-side heuristic strictness)",
        include_bytes!("fixtures/fuzz_regressions/crash-b741d946292d96dd1a04b334523ddcd8e5846431"),
        Expect::NoPanic,
    ),
    // -- 2026-07-25 scheduled-run OOM artifact (round 430). The input
    //    carries one SPS our parser rejects (frame-cropping window
    //    outside the 16x16 picture) plus a parseable 16x16 SPS with
    //    gaps_in_frame_num_value_allowed_flag=1. The libFuzzer OOM was
    //    oracle-side (an older reference build balloons on the stream's
    //    gap/concealment handling; current builds reject it), but the
    //    triage exposed two decoder-side unbounded-memory hazards fixed
    //    in the same arc: the reference store never released evicted
    //    pictures, and §8.2.5.2 gap fill allocated one full placeholder
    //    picture per missing frame_num. The harness now also skips any
    //    input whose SPS NALs cannot all be parsed and size-vetted.
    (
        "oom-ccf7388c (gap-fill / ref-store memory bounds, oracle-side OOM)",
        include_bytes!("fixtures/fuzz_regressions/oom-ccf7388c0de1c3e0ac24b05afffb58a959f76e26"),
        Expect::NoPanic,
    ),
];

/// Decode one input the way the fuzz harness does: single Annex-B
/// packet, flush, drain. Returns (frames_emitted, error_count,
/// any_sps_beyond_8bit_planar).
fn drive(data: &[u8]) -> (usize, u64, bool) {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), data.to_vec());
    let mut frames = 0usize;
    if dec.send_packet(&pkt).is_ok() && dec.flush().is_ok() {
        while let Ok(Frame::Video(_)) = dec.receive_frame() {
            frames += 1;
            if frames > 64 {
                break;
            }
        }
    }
    let beyond_8bit = (0..32).any(|id| {
        dec.stored_sps(id).is_some_and(|sps| {
            sps.bit_depth_luma_minus8 != 0
                || sps.bit_depth_chroma_minus8 != 0
                || sps.chroma_format_idc == 0
                || sps.separate_colour_plane_flag
        })
    });
    (frames, dec.decode_error_count(), beyond_8bit)
}

#[test]
fn archived_fuzz_inputs_hold_their_pinned_classification() {
    for (name, data, expect) in FIXTURES {
        let (frames, errors, beyond_8bit) = drive(data);
        eprintln!(
            "fixture {name}: {} bytes -> frames={frames} errors={errors} beyond_8bit={beyond_8bit}",
            data.len()
        );
        match expect {
            Expect::FlaggedIfFrames => {
                if frames > 0 {
                    assert!(
                        errors > 0 || beyond_8bit,
                        "fixture {name}: emitted {frames} frame(s) from a hostile stream with \
                         decode_error_count == 0 and a plain 8-bit SPS — the harness would treat \
                         the output as a clean decode"
                    );
                }
            }
            Expect::Errors => {
                assert!(
                    errors > 0,
                    "fixture {name}: expected the decoder to register slice errors \
                     (conformance-violation stream), got a clean run with {frames} frame(s)"
                );
            }
            Expect::NoPanic => { /* reaching here without a panic is the pin */ }
        }
    }
}

/// The 2026-07-16 input carries a High 4:4:4 (profile 244) SPS with a
/// 9-bit depth. The harness's incomparability gate depends on seeing
/// that through `stored_sps`; pin it.
#[test]
fn stored_sps_exposes_high_bit_depth_parameters() {
    let (_, _, beyond_8bit) = drive(FIXTURES[0].1);
    assert!(
        beyond_8bit,
        "expected the profile-244 fixture to store an SPS outside the 8-bit-planar domain"
    );
}

/// A clean conforming stream must keep `decode_error_count()` at zero —
/// otherwise the harness would demote every valid decode to "concealed"
/// and the frame-parity assertions would never run.
#[test]
fn conforming_stream_reports_zero_decode_errors() {
    let data = include_bytes!("fixtures/mbaff_iframe_128x96.h264");
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), data.to_vec());
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = 0usize;
    while let Ok(Frame::Video(_)) = dec.receive_frame() {
        frames += 1;
    }
    assert!(frames > 0, "conforming fixture produced no frames");
    assert_eq!(
        dec.decode_error_count(),
        0,
        "conforming fixture must decode without slice errors"
    );
    assert!(
        dec.active_sps().is_some(),
        "active SPS should be exposed after decoding"
    );
}

/// Round 430 (2026-07-25 scheduled-fuzz OOM triage) — the reference
/// picture store must stay DPB-bounded across a long decode session.
/// Before the fix, `RefPicStore` kept EVERY reference picture ever
/// decoded (the sparse key-indexed vector had no release path), so a
/// 64-frame IPP session held 64 pictures at EOF and a real-time stream
/// grew without bound. Post-fix the store holds exactly the live DPB.
#[test]
fn long_session_reference_store_stays_dpb_bounded() {
    use oxideav_h264::encoder::session::{EncoderSession, SessionConfig};

    const W: u32 = 48;
    const H: u32 = 48;
    const FRAMES: usize = 64;

    // One IDR + 63 P frames referencing the previous picture.
    let mut cfg = SessionConfig::constant_qp(W, H, 30);
    cfg.gop_length = FRAMES as u32;
    let mut session = EncoderSession::new(cfg);
    let mut stream = Vec::new();
    for n in 0..FRAMES {
        let (w, h) = (W as usize, H as usize);
        let mut y = vec![0u8; w * h];
        for j in 0..h {
            for i in 0..w {
                y[j * w + i] = ((i * 3 + j * 5 + n * 7) % 200) as u8 + 20;
            }
        }
        let u = vec![110u8; (w / 2) * (h / 2)];
        let v = vec![140u8; (w / 2) * (h / 2)];
        stream.extend_from_slice(&session.encode_frame(&y, &u, &v).annex_b);
    }

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), stream);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = 0usize;
    while let Ok(Frame::Video(_)) = dec.receive_frame() {
        frames += 1;
    }
    assert_eq!(frames, FRAMES, "own decoder must emit every frame");
    assert_eq!(dec.decode_error_count(), 0, "session stream is conforming");

    let held = dec.ref_picture_count();
    assert!(
        held <= 4,
        "reference store holds {held} pictures after a {FRAMES}-frame session — \
         must stay bounded by the DPB (pre-round-430 bug held one per reference frame)"
    );
}
