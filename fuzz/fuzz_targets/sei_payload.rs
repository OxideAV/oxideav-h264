#![no_main]
//! Fuzz: arbitrary bytes → SEI parsing.
//!
//! The whole-decoder fuzz targets feed arbitrary bytes through the NAL
//! pipeline; reaching the Annex D SEI payload parsers behind that path
//! requires a well-formed Annex-B stream (start code → NAL header
//! `forbidden_zero_bit==0 && nal_unit_type==6` → emulation-prevention
//! stripping → `parse_sei_rbsp` → `parse_payload`), which random-byte
//! fuzzing rarely synthesises by accident. This target lifts the SEI
//! surface and feeds bytes directly at three layers:
//!
//! 1. `parse_sei_rbsp` on the raw input (envelope robustness — the
//!    ff_byte runs, payloadType/payloadSize variable-length sums,
//!    payload truncation, the `more_rbsp_data` driver loop).
//! 2. For every message the envelope did extract, `parse_payload` is
//!    invoked under a default-ish `SeiContext` so the dispatch table
//!    covering all 40 implemented payload types (§D.1.2..§D.1.38) gets
//!    exercised.
//! 3. The first input byte additionally selects a payload type from a
//!    fixed pool (every type the dispatcher recognises plus several
//!    "Unknown" sentinels) and feeds the *rest* of the input as raw
//!    payload directly into `parse_payload`. This bypasses path 1's
//!    envelope arithmetic so the rarer high-payloadType branches
//!    (137/142/144/147..156/200/201/205) are hit with frequency
//!    independent of how often the envelope happens to encode them.
//!
//! Contract: every public SEI entry-point must return a `Result` for
//! malformed input — never panic, never abort. Errors from
//! `parse_sei_rbsp` / `parse_payload` are fine; only an escaped panic
//! counts as a finding.

use libfuzzer_sys::fuzz_target;
use oxideav_h264::non_vcl::parse_sei_rbsp;
use oxideav_h264::sei::{parse_payload, SeiContext};

/// Fixed dispatch pool. Mirrors the `match` arms in
/// `oxideav_h264::sei::parse_payload`. The trailing entries fall through
/// to the `Unknown` arm so the catch-all path is also covered.
const PAYLOAD_TYPES: &[u32] = &[
    // §D.1.2..§D.1.27 + §D.1.38 — Annex D base
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 45, 47,
    // §D.1.29..§D.1.35 — HDR + 360 family
    137, 142, 144, 147, 148, 149, 150, 151, 154, 155, 156,
    // Annex H §H.13.2.6 — alternative_depth_info
    181,
    // §D.1.36..§D.1.38 — transport-layer hints
    200, 201, 205,
    // Annex F/G/H/I numeric ranges + reserved values — hit the `Unknown`
    // fallback path so the catch-all doesn't go unfuzzed.
    24, 30, 36, 40, 50, 54, 56, 100, 138, 140, 152, 153, 157, 160, 199, 202, 204, 206, 255, 4096,
    65535, 100_000, 1_000_000,
];

fn context_from_byte(b: u8) -> SeiContext {
    // Build a varied-but-bounded SeiContext from a single input byte so
    // the buffering_period / pic_timing / dec_ref_pic_marking_repetition
    // / spare_pic / motion_constrained_slice_group_set branches get
    // explored under several VUI/SPS shapes. Every field is bounded
    // inside the spec ranges (e.g. *_length_minus1 ≤ 31, cpb_cnt_minus1
    // ≤ 31). `default()` is one specific cell of this space; this
    // function expands it.
    SeiContext {
        initial_cpb_removal_delay_length_minus1: b & 0x1F,
        cpb_removal_delay_length_minus1: (b >> 3) & 0x1F,
        dpb_output_delay_length_minus1: (b >> 1) & 0x1F,
        time_offset_length: (b >> 2) & 0x1F,
        nal_hrd_cpb_cnt_minus1: if b & 0x80 != 0 {
            Some((b & 0x1F) as u32)
        } else {
            None
        },
        vcl_hrd_cpb_cnt_minus1: if b & 0x40 != 0 {
            Some(((b >> 1) & 0x1F) as u32)
        } else {
            None
        },
        pic_struct_present_flag: b & 0x20 != 0,
        cpb_dpb_delays_present_flag: b & 0x10 != 0,
        // num_slice_groups in 1..=8 keeps the §D.1.20 u(v) width bounded
        // (slice_group_id width = ceil(log2(num_slice_groups)) bits).
        num_slice_groups: 1 + ((b as u32) & 0x07),
        // pic_size_in_map_units small enough that the spare_pic loops
        // can't allocate gigabytes if the dispatch lands there.
        pic_size_in_map_units: ((b as u32) & 0x0F) * 16,
        frame_mbs_only_flag: b & 0x08 == 0,
        // num_depth_views in 0..=7 covers the §H.13.1.5 depth_timing
        // unknown-rejection (0) and the per-view loop (1..). The Annex H
        // ceiling is 1024; small values keep the per-view Vec bounded.
        num_depth_views: ((b as u32) >> 5) & 0x07,
    }
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let ctx = context_from_byte(data[0]);

    // Path 1: feed the full input through the SEI envelope parser, then
    // dispatch every extracted message through parse_payload.
    if let Ok(messages) = parse_sei_rbsp(data) {
        for msg in &messages {
            let _ = parse_payload(msg.payload_type, &msg.payload, &ctx);
        }
    }

    // Path 2: bypass the envelope so the per-type Annex D parsers get
    // hit with the input as raw payload. The first byte selects the
    // payloadType from the fixed pool above; the rest is the payload.
    let pt = PAYLOAD_TYPES[(data[0] as usize) % PAYLOAD_TYPES.len()];
    let _ = parse_payload(pt, &data[1..], &ctx);

    // Path 3: also feed the *whole* input as raw payload under a
    // payloadType determined by the second byte (when present), so the
    // first byte's effect on `ctx` doesn't fully determine which arm we
    // hit on path 2.
    if data.len() >= 2 {
        let pt2 = PAYLOAD_TYPES[(data[1] as usize) % PAYLOAD_TYPES.len()];
        let _ = parse_payload(pt2, data, &ctx);
    }
});
