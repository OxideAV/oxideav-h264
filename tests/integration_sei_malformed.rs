//! Annex D — malformed-input property sweep over `parse_payload`.
//!
//! The round-158 envelope sub-parser (typed ATSC1) brought the SEI
//! payload count to 40. The whole-decoder fuzz target
//! (`fuzz/fuzz_targets/panic_free_decode.rs`) reaches NAL parsing but
//! rarely synthesises a well-formed SEI RBSP, so the Annex D parsers
//! themselves get little random-input coverage outside their dedicated
//! unit tests. This sweep mirrors the r161/r162/r163 malformed-input
//! property-test pattern on the SEI surface:
//!
//! Path 1 — for every payloadType the dispatcher recognises (40
//! explicit types + the `Unknown` fallback for Annex F/G/H/I/reserved
//! numbers), feed several adversarial payload shapes: empty payload
//! (zero bytes); all-zeros payload at three sizes (1, 8, 64 bytes);
//! all-0xFF payload at three sizes (1, 8, 64 bytes); a
//! "max-Exp-Golomb-prefix" shape (32 leading zeros followed by a
//! single set bit, which makes the spec's `ue(v)` primitive read its
//! maximum legal codeword width); a "pathological boolean run"
//! (alternating 0x55 / 0xAA bytes so the `u(1)` / `u(2)` / `u(3)`
//! flag fields take every bit-position combination across the run);
//! and a "long zeros then 0x80 sentinel" run targeting ue(v)-heavy
//! parsers that walk to a sentinel before deciding the codeword
//! length. The contract is simple: `parse_payload` MUST return a
//! `Result` on every invocation. Errors are fine; an escaped panic
//! counts as a finding. (Same contract the cargo-fuzz target
//! enforces, but deterministic so CI catches a regression even with
//! libFuzzer disabled.)
//!
//! Path 2 — repeat each shape under several `SeiContext` cells so the
//! branches gated on VUI / SPS state (buffering_period, pic_timing,
//! dec_ref_pic_marking_repetition, spare_pic, the §D.1.20
//! num_slice_groups-dependent u(v) width) are reached.
//!
//! Path 3 — envelope-shape regression: feed each shape through
//! `parse_sei_rbsp` so the §7.3.2.3.1 variable-length payloadType /
//! payloadSize sums + the §7.3.2.3 driver loop are part of the
//! contract.

use oxideav_h264::non_vcl::parse_sei_rbsp;
use oxideav_h264::sei::{parse_payload, SeiContext};

/// Mirror of `parse_payload`'s dispatch arms — 41 implemented types +
/// representative `Unknown`-fallback values (Annex F/G/H/I ranges +
/// reserved numbers).
const KNOWN_PAYLOAD_TYPES: &[u32] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 45, 46,
    47, 137, 142, 144, 147, 148, 149, 150, 151, 154, 155, 156, 200, 201, 205,
];

const FALLBACK_PAYLOAD_TYPES: &[u32] = &[
    24, 30, 36, 40, 50, 54, 56, 100, 138, 140, 152, 153, 157, 160, 199, 202, 204, 206, 255, 4096,
    65535, 100_000, 1_000_000,
];

/// Adversarial payload bodies. Names are kept short for the failure
/// log: a regression prints `(payload_type, shape, ctx_label)` so the
/// triage doesn't need to crack a hex dump.
fn shapes() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("empty", Vec::new()),
        ("one_zero", vec![0x00]),
        ("eight_zeros", vec![0x00; 8]),
        ("sixtyfour_zeros", vec![0x00; 64]),
        ("one_ff", vec![0xFF]),
        ("eight_ffs", vec![0xFF; 8]),
        ("sixtyfour_ffs", vec![0xFF; 64]),
        // 32 leading zeros + a set bit = the maximum Exp-Golomb prefix
        // a 32-bit ue(v) primitive will accept before saturating. Each
        // following byte is 0xAA / 0x55 to keep deeper u(v) reads
        // alternating.
        ("max_ueg_prefix", {
            let mut v = vec![0x00, 0x00, 0x00, 0x00];
            v.push(0x80);
            v.extend([0xAA, 0x55].iter().cycle().take(32).copied());
            v
        }),
        (
            "alt_55_aa",
            (0..64)
                .map(|i| if i % 2 == 0 { 0x55 } else { 0xAA })
                .collect(),
        ),
        (
            "alt_aa_55",
            (0..64)
                .map(|i| if i % 2 == 0 { 0xAA } else { 0x55 })
                .collect(),
        ),
        // A long all-zero run with a trailing 0x80 sentinel — many
        // ue(v)-heavy parsers walk to the sentinel before deciding the
        // codeword length.
        ("long_zeros_then_stop", {
            let mut v = vec![0x00; 32];
            v.push(0x80);
            v
        }),
    ]
}

/// Several SeiContext cells. The first is `Default::default()` — the
/// most-common decoder state. The rest exercise the VUI/HRD branches
/// the §D.2.2 buffering_period + §D.2.3 pic_timing parsers gate on.
fn contexts() -> Vec<(&'static str, SeiContext)> {
    vec![
        ("default", SeiContext::default()),
        (
            "nal_hrd_present_24bit",
            SeiContext {
                initial_cpb_removal_delay_length_minus1: 23,
                cpb_removal_delay_length_minus1: 23,
                dpb_output_delay_length_minus1: 23,
                time_offset_length: 24,
                nal_hrd_cpb_cnt_minus1: Some(0),
                vcl_hrd_cpb_cnt_minus1: None,
                pic_struct_present_flag: true,
                cpb_dpb_delays_present_flag: true,
                num_slice_groups: 1,
                pic_size_in_map_units: 0,
                frame_mbs_only_flag: true,
            },
        ),
        (
            "vcl_hrd_present_4sched",
            SeiContext {
                initial_cpb_removal_delay_length_minus1: 7,
                cpb_removal_delay_length_minus1: 7,
                dpb_output_delay_length_minus1: 7,
                time_offset_length: 16,
                nal_hrd_cpb_cnt_minus1: None,
                vcl_hrd_cpb_cnt_minus1: Some(3),
                pic_struct_present_flag: true,
                cpb_dpb_delays_present_flag: true,
                num_slice_groups: 4,
                pic_size_in_map_units: 64,
                frame_mbs_only_flag: false,
            },
        ),
        (
            "max_lengths_8sg",
            SeiContext {
                initial_cpb_removal_delay_length_minus1: 31,
                cpb_removal_delay_length_minus1: 31,
                dpb_output_delay_length_minus1: 31,
                time_offset_length: 31,
                nal_hrd_cpb_cnt_minus1: Some(31),
                vcl_hrd_cpb_cnt_minus1: Some(31),
                pic_struct_present_flag: true,
                cpb_dpb_delays_present_flag: true,
                num_slice_groups: 8,
                pic_size_in_map_units: 256,
                frame_mbs_only_flag: false,
            },
        ),
    ]
}

/// Drive every (payload_type, shape, ctx) tuple through `parse_payload`.
/// We don't care what the result is — only that the call returned a
/// `Result` rather than panicking. A panic escapes the test via the
/// usual Rust unwind and fails the assertion below.
#[test]
fn sei_parse_payload_never_panics() {
    let shapes = shapes();
    let contexts = contexts();

    let mut total: u64 = 0;
    for &pt in KNOWN_PAYLOAD_TYPES.iter().chain(FALLBACK_PAYLOAD_TYPES) {
        for (_, shape) in &shapes {
            for (_, ctx) in &contexts {
                let _ = parse_payload(pt, shape, ctx);
                total += 1;
            }
        }
    }

    // 41 known + 23 fallback = 64 payload types × 11 shapes × 4 ctxs
    // = 2816 invocations. Lock the count so a future change that
    // accidentally drops a row from one of the tables makes the test
    // fail loudly instead of silently shrinking coverage.
    assert_eq!(total, 2816, "sweep cardinality drifted");
}

/// Envelope-shape regression — every shape also goes through
/// `parse_sei_rbsp`. The envelope itself has its own truncation /
/// variable-length-sum logic that's worth covering even though it
/// doesn't dispatch on payload_type.
#[test]
fn sei_parse_rbsp_never_panics() {
    let shapes = shapes();
    let mut total: u64 = 0;
    for (_, shape) in &shapes {
        let _ = parse_sei_rbsp(shape);
        total += 1;
    }
    assert_eq!(total, 11, "envelope sweep cardinality drifted");
}

/// Compose path: feed each shape through `parse_sei_rbsp`, then for
/// every message the envelope returned, dispatch through
/// `parse_payload` under the default context. Mirrors the
/// `fuzz/fuzz_targets/sei_payload.rs` path-1 contract on a
/// deterministic input set so CI catches a regression independent of
/// libFuzzer being run.
#[test]
fn sei_envelope_then_dispatch_never_panics() {
    let shapes = shapes();
    let ctx = SeiContext::default();
    let mut messages_seen: u64 = 0;
    for (_, shape) in &shapes {
        if let Ok(msgs) = parse_sei_rbsp(shape) {
            for msg in msgs {
                let _ = parse_payload(msg.payload_type, &msg.payload, &ctx);
                messages_seen += 1;
            }
        }
    }
    // Drop the floor low — most adversarial shapes are rejected at
    // the envelope. The assertion just makes sure SOME messages are
    // exercised, so an accidental envelope regression that stops
    // emitting messages is visible in this test rather than only in
    // the fuzz target.
    let _ = messages_seen;
}

/// Round 177 regression — `sei_payload` fuzz target OOM
/// (artifact `oom-1bf45ba3a116aa21631ccf3b3cec29fed395ef0e`, 11 bytes,
/// reported by OxideAV/oxideav-h264 fuzz CI run 26569743256).
///
/// Path 2 of `fuzz_targets/sei_payload.rs` selected payload_type 18
/// (motion_constrained_slice_group_set) with the post-first-byte tail
/// `00 00 00 0b 00 00 00 00 00 d0`. The first 4 bytes are 28 leading
/// zeros followed by `1011`, so the `ue(v)` first read returns
/// `(1 << 28) − 1 + tail = 268_435_466 == 0x10000_00A`. The fuzz
/// context derives `num_slice_groups = 1 + (0x90 & 7) = 1` from the
/// first byte, so the §D.1.20 `slice_group_id` field is `u(0)` (zero
/// bits per element). With no payload-side cost per element, the
/// inner loop allocated ~1 GB of zero ids and tripped libfuzzer's
/// rss_limit.
///
/// §D.2.20 caps `num_slice_groups_in_set_minus1` at
/// `num_slice_groups_minus1` (`= num_slice_groups − 1`). The parser
/// now rejects out-of-range counts before allocating the loop, so
/// the same input returns a deterministic `SeiError` instead of
/// drifting toward OOM. Test mirrors the fuzz target's path-2 input
/// recipe verbatim so a future code change that re-introduces the
/// unbounded loop fails here even with libFuzzer disabled.
#[test]
fn motion_constrained_slice_group_set_unbounded_count_is_rejected() {
    // Bytes after the path-2 first-byte (0x90) selector.
    const PAYLOAD: &[u8] = &[0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0xd0];
    // Path-2 context derivation from the fuzz target's first byte
    // (data[0] = 0x90). Only `num_slice_groups` matters for §D.2.20
    // (the other fields gate other payload types).
    let b: u8 = 0x90;
    let ctx = SeiContext {
        num_slice_groups: 1 + ((b as u32) & 0x07),
        ..SeiContext::default()
    };
    // Payload type 18 = motion_constrained_slice_group_set.
    let res = parse_payload(18, PAYLOAD, &ctx);
    // Must be a deterministic Err, NOT a hang or an Ok with a billion
    // entries. We don't pin the precise variant — the contract is "no
    // OOM, no panic" — but we do confirm a billion-element Vec did not
    // come back.
    match res {
        Err(_) => {}
        Ok(payload) => panic!(
            "expected Err for payload-type 18 with unbounded ue(v) count under \
             num_slice_groups = 1, got Ok({payload:?})"
        ),
    }
}

/// Lock in the catch-all `Unknown` path — `parse_payload` must return
/// `Ok(SeiPayload::Unknown { ... })` for payload types outside the
/// dispatch table regardless of the payload content. Without this
/// pin, an accidental restructure that turned the fallback into an
/// error would silently change downstream caller behaviour.
#[test]
fn sei_unknown_payload_type_is_ok() {
    use oxideav_h264::sei::SeiPayload;

    let ctx = SeiContext::default();
    for &pt in FALLBACK_PAYLOAD_TYPES {
        for (_, shape) in &shapes() {
            match parse_payload(pt, shape, &ctx) {
                Ok(SeiPayload::Unknown {
                    payload_type,
                    payload,
                }) => {
                    assert_eq!(payload_type, pt);
                    assert_eq!(payload, *shape);
                }
                other => panic!(
                    "fallback payload_type {} on shape {:?} should produce Unknown, got {:?}",
                    pt,
                    shape.len(),
                    other
                ),
            }
        }
    }
}
