//! `separate_colour_plane_flag = 1` narrowing tests — ITU-T H.264
//! §7.4.2.1.1.
//!
//! The 4:4:4 + separate-colour-plane coding mode (profile 244, three
//! independent monochrome pictures keyed by `colour_plane_id`) is not
//! wired in this decoder. These tests pin three things we DO support
//! without running the full MB pipeline:
//!
//! 1. `parse_sps` accepts a High 4:4:4 Predictive SPS that carries
//!    `chroma_format_idc = 3` + `separate_colour_plane_flag = 1` and
//!    round-trips both fields.
//! 2. `parse_slice_header` consumes the 2-bit `colour_plane_id` that
//!    §7.3.3 places before `frame_num` when the SPS flag is set. Each
//!    valid id ∈ {0, 1, 2} is tested.
//! 3. `H264Decoder` feeds an Annex-B packet combining such SPS/PPS and
//!    a minimal slice, and returns `Error::Unsupported` referencing
//!    `separate_colour_plane` — i.e. the reject is specific to this
//!    mode, not swallowed by the generic chroma-format gate. This keeps
//!    the upstream failure mode observable for downstream callers that
//!    want to route around it.

use oxideav_core::{CodecId, Packet, TimeBase};
use oxideav_h264::bitwriter::{rbsp_to_ebsp, BitWriter};
use oxideav_h264::decoder::H264Decoder;
use oxideav_h264::nal::{NalHeader, NalUnitType};
use oxideav_h264::pps::parse_pps;
use oxideav_h264::slice::parse_slice_header;
use oxideav_h264::sps::parse_sps;

/// Build a High 4:4:4 Predictive (profile_idc = 244) SPS with
/// `chroma_format_idc = 3` + `separate_colour_plane_flag = 1`. 16×16
/// coded picture, no scaling matrices, `pic_order_cnt_type = 2`,
/// `log2_max_frame_num_minus4 = 0`.
fn build_separate_colour_plane_sps() -> Vec<u8> {
    let mut bw = BitWriter::new();
    // profile_idc = 244 (High 4:4:4 Predictive) — enables the high-profile
    // block that carries chroma_format_idc + separate_colour_plane_flag.
    bw.write_bits(244, 8);
    // constraint_set flags + reserved bits.
    bw.write_bits(0, 8);
    // level_idc = 30.
    bw.write_bits(30, 8);
    bw.write_ue(0); // seq_parameter_set_id
    bw.write_ue(3); // chroma_format_idc = 3 (4:4:4)
    bw.write_flag(true); // separate_colour_plane_flag = 1
    bw.write_ue(0); // bit_depth_luma_minus8 (8-bit)
    bw.write_ue(0); // bit_depth_chroma_minus8
    bw.write_flag(false); // qpprime_y_zero_transform_bypass_flag
    bw.write_flag(false); // seq_scaling_matrix_present_flag
    bw.write_ue(0); // log2_max_frame_num_minus4
    bw.write_ue(2); // pic_order_cnt_type = 2
    bw.write_ue(1); // max_num_ref_frames
    bw.write_flag(false); // gaps_in_frame_num_value_allowed_flag
    bw.write_ue(0); // pic_width_in_mbs_minus1 → 16-wide
    bw.write_ue(0); // pic_height_in_map_units_minus1 → 16-tall
    bw.write_flag(true); // frame_mbs_only_flag
    bw.write_flag(true); // direct_8x8_inference_flag
    bw.write_flag(false); // frame_cropping_flag
    bw.write_flag(false); // vui_parameters_present_flag
    bw.write_rbsp_trailing_bits();
    bw.finish()
}

fn build_minimal_pps() -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.write_ue(0); // pic_parameter_set_id
    bw.write_ue(0); // seq_parameter_set_id
    bw.write_flag(false); // entropy_coding_mode_flag (CAVLC)
    bw.write_flag(false); // bottom_field_pic_order_in_frame_present_flag
    bw.write_ue(0); // num_slice_groups_minus1
    bw.write_ue(0); // num_ref_idx_l0_default_active_minus1
    bw.write_ue(0); // num_ref_idx_l1_default_active_minus1
    bw.write_flag(false); // weighted_pred_flag
    bw.write_bits(0, 2); // weighted_bipred_idc
    bw.write_se(0); // pic_init_qp_minus26
    bw.write_se(0); // pic_init_qs_minus26
    bw.write_se(0); // chroma_qp_index_offset
    bw.write_flag(true); // deblocking_filter_control_present_flag
    bw.write_flag(false); // constrained_intra_pred_flag
    bw.write_flag(false); // redundant_pic_cnt_present_flag
    bw.write_rbsp_trailing_bits();
    bw.finish()
}

/// Craft an IDR slice header RBSP for a picture whose SPS has
/// `separate_colour_plane_flag = 1`. `plane_id` is written as the 2-bit
/// `colour_plane_id` (§7.3.3) immediately after `pic_parameter_set_id`.
fn build_colour_plane_slice_header_rbsp(plane_id: u8) -> Vec<u8> {
    assert!(plane_id < 4, "colour_plane_id is u(2)");
    let mut bw = BitWriter::new();
    bw.write_ue(0); // first_mb_in_slice
    bw.write_ue(7); // slice_type = I (single-type variant)
    bw.write_ue(0); // pic_parameter_set_id
    bw.write_bits(plane_id as u32, 2); // colour_plane_id
    bw.write_bits(0, 4); // frame_num (4 bits, log2_max_frame_num_minus4 = 0)
    // frame_mbs_only_flag = 1 → no field_pic_flag bits.
    bw.write_ue(0); // idr_pic_id
    // pic_order_cnt_type = 2 in the SPS → no pic_order_cnt_lsb bits.
    // weighted_pred_flag = 0 in PPS → no pred_weight_table.
    // dec_ref_pic_marking (is_idr, nal_ref_idc != 0):
    bw.write_flag(false); // no_output_of_prior_pics_flag
    bw.write_flag(false); // long_term_reference_flag
    // entropy_coding_mode_flag = 0 → no cabac_init_idc.
    bw.write_se(0); // slice_qp_delta
    // deblocking_filter_control_present_flag = 1 in PPS:
    bw.write_ue(1); // disable_deblocking_filter_idc = 1 (off)
    bw.write_rbsp_trailing_bits();
    bw.finish()
}

#[test]
fn sps_parses_separate_colour_plane_flag() {
    let rbsp = build_separate_colour_plane_sps();
    let header = NalHeader {
        forbidden_zero_bit: 0,
        nal_ref_idc: 3,
        nal_unit_type: NalUnitType::Sps,
    };
    let sps = parse_sps(&header, &rbsp).expect("parse sps");
    assert_eq!(sps.profile_idc, 244);
    assert_eq!(sps.chroma_format_idc, 3);
    assert!(
        sps.separate_colour_plane_flag,
        "SPS must round-trip separate_colour_plane_flag = 1"
    );
    assert_eq!(sps.bit_depth_luma_minus8, 0);
    assert_eq!(sps.bit_depth_chroma_minus8, 0);
}

#[test]
fn slice_header_parses_colour_plane_id_all_three() {
    let sps_rbsp = build_separate_colour_plane_sps();
    let sps_header = NalHeader {
        forbidden_zero_bit: 0,
        nal_ref_idc: 3,
        nal_unit_type: NalUnitType::Sps,
    };
    let sps = parse_sps(&sps_header, &sps_rbsp).expect("parse sps");

    let pps_rbsp = build_minimal_pps();
    let pps_header = NalHeader {
        forbidden_zero_bit: 0,
        nal_ref_idc: 3,
        nal_unit_type: NalUnitType::Pps,
    };
    let pps = parse_pps(&pps_header, &pps_rbsp, None).expect("parse pps");

    for plane_id in 0u8..=2 {
        let rbsp = build_colour_plane_slice_header_rbsp(plane_id);
        let slice_header = NalHeader {
            forbidden_zero_bit: 0,
            nal_ref_idc: 3,
            nal_unit_type: NalUnitType::SliceIdr,
        };
        let sh =
            parse_slice_header(&slice_header, &rbsp, &sps, &pps).expect("parse slice header");
        assert_eq!(
            sh.colour_plane_id, plane_id,
            "§7.3.3 colour_plane_id round-trip failed for plane {plane_id}"
        );
        assert!(sh.is_idr);
        assert_eq!(sh.frame_num, 0);
    }
}

#[test]
fn decoder_rejects_separate_colour_plane_flag_cleanly() {
    // Wire the SPS + PPS + a plane-0 IDR slice into an Annex-B packet and
    // feed it to the decoder. The decoder must reject with an Unsupported
    // error that mentions separate_colour_plane (i.e. the reject is
    // specific to this mode, not funnelled through the generic chroma-
    // format gate).
    let sps_rbsp = build_separate_colour_plane_sps();
    let pps_rbsp = build_minimal_pps();
    let slice_rbsp = build_colour_plane_slice_header_rbsp(0);

    let mut sps_nal = vec![0x67u8]; // forbidden=0, nal_ref_idc=3, type=7 SPS
    sps_nal.extend_from_slice(&rbsp_to_ebsp(&sps_rbsp));
    let mut pps_nal = vec![0x68u8]; // type=8 PPS
    pps_nal.extend_from_slice(&rbsp_to_ebsp(&pps_rbsp));
    let mut idr_nal = vec![0x65u8]; // type=5 SliceIdr
    idr_nal.extend_from_slice(&rbsp_to_ebsp(&slice_rbsp));

    let mut packet = Vec::new();
    packet.extend_from_slice(&[0, 0, 0, 1]);
    packet.extend_from_slice(&sps_nal);
    packet.extend_from_slice(&[0, 0, 0, 1]);
    packet.extend_from_slice(&pps_nal);
    packet.extend_from_slice(&[0, 0, 0, 1]);
    packet.extend_from_slice(&idr_nal);

    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 90_000), packet).with_pts(0);
    use oxideav_codec::Decoder;
    let err = dec.send_packet(&pkt).expect_err(
        "decoder must reject separate_colour_plane_flag = 1 — not silently decode",
    );
    let msg = format!("{err}").to_lowercase();
    assert!(
        msg.contains("separate_colour_plane"),
        "reject message must identify the unsupported mode, got: {err}",
    );
}
