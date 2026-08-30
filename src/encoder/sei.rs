//! Round-430 — write side of the SEI messages the rate-controlled
//! sessions emit (§7.3.2.3 / §D.1): `buffering_period` (payloadType 0)
//! and `pic_timing` (payloadType 1), so CBR streams carry the Annex C
//! HRD annotation in-band alongside the SPS's §E.1.2 NAL HRD block.
//!
//! The parse side lives in [`crate::sei`] / [`crate::non_vcl`]; these
//! writers are field-for-field mirrors of those parsers.
//!
//! Framing (§7.3.2.3.1): each `sei_message()` codes `payloadType` and
//! `payloadSize` as sequences of 0xFF bytes plus a final non-0xFF
//! byte, followed by `payloadSize` payload bytes. Payloads whose
//! syntax is not byte-aligned end with the §D.1 `sei_payload` bit
//! alignment: a stop bit (`bit_equal_to_one`) then zero bits. The
//! whole `sei_rbsp()` then gets `rbsp_trailing_bits()`.
//!
//! Clean-room: derived only from ITU-T Rec. H.264 (08/2024) §7.3.2.3,
//! §D.1.2, §D.1.3, Annex E.

use crate::encoder::bitstream::BitWriter;
use crate::encoder::nal::build_nal_unit;
use crate::nal::NalUnitType;
use crate::vui::HrdParameters;

/// §D.1.2 — build a `buffering_period` payload (payloadType 0).
///
/// `nal_delays` carries one `(initial_cpb_removal_delay,
/// initial_cpb_removal_delay_offset)` pair per SchedSelIdx of the NAL
/// HRD block (we emit no VCL HRD). Field widths come from the SPS's
/// [`HrdParameters::initial_cpb_removal_delay_length_minus1`].
pub fn build_buffering_period_payload(
    seq_parameter_set_id: u32,
    hrd: &HrdParameters,
    nal_delays: &[(u32, u32)],
) -> Vec<u8> {
    debug_assert_eq!(
        nal_delays.len(),
        hrd.cpb_cnt_minus1 as usize + 1,
        "one delay pair per SchedSelIdx (§D.1.2)"
    );
    let bits = hrd.initial_cpb_removal_delay_length_minus1 as u32 + 1;
    let mut w = BitWriter::new();
    w.ue(seq_parameter_set_id);
    for &(delay, offset) in nal_delays {
        debug_assert!(delay > 0, "initial_cpb_removal_delay shall be > 0 (§D.2.2)");
        w.u(bits, delay);
        w.u(bits, offset);
    }
    finish_sei_payload(w)
}

/// §D.1.3 — build a `pic_timing` payload (payloadType 1) for a stream
/// whose VUI declares NAL HRD (CpbDpbDelaysPresentFlag = 1) and
/// `pic_struct_present_flag = 0`.
///
/// `cpb_removal_delay` counts clock ticks since the last
/// buffering-period access unit; `dpb_output_delay` is the removal-to-
/// output distance (0 for a no-reorder IPP stream whose pictures are
/// output at removal time). Field widths come from the SPS HRD block.
pub fn build_pic_timing_payload(
    hrd: &HrdParameters,
    cpb_removal_delay: u32,
    dpb_output_delay: u32,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.u(
        hrd.cpb_removal_delay_length_minus1 as u32 + 1,
        cpb_removal_delay,
    );
    w.u(
        hrd.dpb_output_delay_length_minus1 as u32 + 1,
        dpb_output_delay,
    );
    finish_sei_payload(w)
}

/// Round-453 — §D.1.8 — build a `recovery_point` payload (payloadType
/// 6): `recovery_frame_cnt` ue(v), `exact_match_flag` u(1),
/// `broken_link_flag` u(1), `changing_slice_group_idc` u(2). Attached
/// to random-access pictures so a decoder starting there knows after
/// how many frames (in output order) the output is correct.
pub fn build_recovery_point_payload(
    recovery_frame_cnt: u32,
    exact_match_flag: bool,
    broken_link_flag: bool,
    changing_slice_group_idc: u8,
) -> Vec<u8> {
    debug_assert!(changing_slice_group_idc <= 2, "§D.2.8: idc in 0..=2");
    let mut w = BitWriter::new();
    w.ue(recovery_frame_cnt);
    w.u(1, u32::from(exact_match_flag));
    w.u(1, u32::from(broken_link_flag));
    w.u(2, u32::from(changing_slice_group_idc));
    finish_sei_payload(w)
}

/// §D.1 tail of every `sei_payload()` whose syntax is not guaranteed
/// byte-aligned: when the writer sits mid-byte, emit the stop bit +
/// zero padding (`bit_equal_to_one` / `bit_equal_to_zero`).
fn finish_sei_payload(mut w: BitWriter) -> Vec<u8> {
    if !w.byte_aligned() {
        w.u(1, 1);
        while !w.byte_aligned() {
            w.u(1, 0);
        }
    }
    w.into_bytes()
}

/// §7.3.2.3 — assemble one `sei_rbsp()` from `(payloadType, payload)`
/// messages and wrap it into a complete Annex B SEI NAL unit
/// (`nal_ref_idc = 0`, type 6).
pub fn build_sei_nal(messages: &[(u32, Vec<u8>)]) -> Vec<u8> {
    let mut rbsp = Vec::new();
    for (payload_type, payload) in messages {
        // §7.3.2.3.1 — ff_byte* + last_payload_type_byte.
        let mut t = *payload_type;
        while t >= 255 {
            rbsp.push(0xFF);
            t -= 255;
        }
        rbsp.push(t as u8);
        // ff_byte* + last_payload_size_byte.
        let mut sz = payload.len();
        while sz >= 255 {
            rbsp.push(0xFF);
            sz -= 255;
        }
        rbsp.push(sz as u8);
        rbsp.extend_from_slice(payload);
    }
    // §7.3.2.3 — rbsp_trailing_bits(): payloads are byte-aligned, so
    // this is the 0x80 stop byte.
    rbsp.push(0x80);
    build_nal_unit(0, NalUnitType::Sei, &rbsp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal::{parse_nal_unit, AnnexBSplitter};
    use crate::non_vcl::parse_sei_rbsp;
    use crate::sei::{parse_buffering_period, parse_pic_timing, SeiContext};

    fn test_hrd() -> HrdParameters {
        HrdParameters {
            cpb_cnt_minus1: 0,
            bit_rate_scale: 0,
            cpb_size_scale: 0,
            bit_rate_value_minus1: vec![1874], // 120_000 bps at scale 0
            cpb_size_value_minus1: vec![7499], // 120_000 bits at scale 0
            cbr_flag: vec![true],
            initial_cpb_removal_delay_length_minus1: 23,
            cpb_removal_delay_length_minus1: 23,
            dpb_output_delay_length_minus1: 23,
            time_offset_length: 24,
        }
    }

    fn ctx_for(hrd: &HrdParameters) -> SeiContext {
        SeiContext {
            initial_cpb_removal_delay_length_minus1: hrd.initial_cpb_removal_delay_length_minus1,
            cpb_removal_delay_length_minus1: hrd.cpb_removal_delay_length_minus1,
            dpb_output_delay_length_minus1: hrd.dpb_output_delay_length_minus1,
            time_offset_length: hrd.time_offset_length,
            nal_hrd_cpb_cnt_minus1: Some(hrd.cpb_cnt_minus1),
            vcl_hrd_cpb_cnt_minus1: None,
            pic_struct_present_flag: false,
            cpb_dpb_delays_present_flag: true,
            ..SeiContext::default()
        }
    }

    /// The emitted buffering_period + pic_timing SEI NAL parses back
    /// through the crate's own §7.3.2.3 envelope + §D.1 payload
    /// parsers, field-exact.
    #[test]
    fn bp_and_pt_sei_round_trip_through_own_parsers() {
        let hrd = test_hrd();
        let bp = build_buffering_period_payload(0, &hrd, &[(54_000, 36_000)]);
        let pt = build_pic_timing_payload(&hrd, 6, 0);
        let nal = build_sei_nal(&[(0, bp), (1, pt)]);

        let mut nals: Vec<&[u8]> = AnnexBSplitter::new(&nal).collect();
        assert_eq!(nals.len(), 1);
        let nu = parse_nal_unit(nals.remove(0)).expect("nal parses");
        assert_eq!(nu.header.nal_unit_type, crate::nal::NalUnitType::Sei);
        let msgs = parse_sei_rbsp(&nu.rbsp).expect("sei envelope parses");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].payload_type, 0);
        assert_eq!(msgs[1].payload_type, 1);

        let ctx = ctx_for(&hrd);
        let bp = parse_buffering_period(&msgs[0].payload, &ctx).expect("bp parses");
        assert_eq!(bp.seq_parameter_set_id, 0);
        let nal_hrd = bp.nal_hrd.expect("nal hrd delays present");
        assert_eq!(nal_hrd.len(), 1);
        assert_eq!(nal_hrd[0].initial_cpb_removal_delay, 54_000);
        assert_eq!(nal_hrd[0].initial_cpb_removal_delay_offset, 36_000);
        assert!(bp.vcl_hrd.is_none());

        let pt = parse_pic_timing(&msgs[1].payload, &ctx).expect("pt parses");
        assert_eq!(pt.cpb_removal_delay, 6);
        assert_eq!(pt.dpb_output_delay, 0);
        assert!(pt.pic_struct.is_none());
    }

    /// Non-byte-multiple field widths force the §D.1 payload bit
    /// alignment tail; the parser must still read the values exactly.
    #[test]
    fn odd_width_delays_are_bit_aligned_correctly() {
        let mut hrd = test_hrd();
        hrd.initial_cpb_removal_delay_length_minus1 = 17; // 18-bit fields
        hrd.cpb_removal_delay_length_minus1 = 9; // 10-bit
        hrd.dpb_output_delay_length_minus1 = 4; // 5-bit
        let bp = build_buffering_period_payload(3, &hrd, &[(0x2_0001, 0x1_5555)]);
        let pt = build_pic_timing_payload(&hrd, 0x3FF, 0x1F);
        let ctx = ctx_for(&hrd);
        let parsed_bp = parse_buffering_period(&bp, &ctx).expect("bp parses");
        assert_eq!(parsed_bp.seq_parameter_set_id, 3);
        let d = &parsed_bp.nal_hrd.unwrap()[0];
        assert_eq!(d.initial_cpb_removal_delay, 0x2_0001);
        assert_eq!(d.initial_cpb_removal_delay_offset, 0x1_5555);
        let parsed_pt = parse_pic_timing(&pt, &ctx).expect("pt parses");
        assert_eq!(parsed_pt.cpb_removal_delay, 0x3FF);
        assert_eq!(parsed_pt.dpb_output_delay, 0x1F);
    }
}
