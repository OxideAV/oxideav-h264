//! Integration test for reference picture list modification (RPLM) —
//! §7.3.3.1 / §7.4.3.1 / §8.2.4.3.
//!
//! A hand-crafted slice header RBSP is run through the full
//! [`parse_slice_header`] path and the parsed [`RplmCommand`] sequence is
//! replayed through [`apply_rplm`] on top of a default-built L0 list. This
//! exercises the parser wiring and the DPB integration that the decoder
//! uses at the slice boundary — a real encoded RPLM-using fixture is
//! not feasible without a hand-built macroblock layer.

use oxideav_core::bits::BitWriter;
use oxideav_h264::dpb::{apply_rplm, Dpb, RplmCommand};
use oxideav_h264::golomb::BitWriterExt;
use oxideav_h264::nal::{NalHeader, NalUnitType};
use oxideav_h264::picture::Picture;
use oxideav_h264::pps::{parse_pps, Pps};
use oxideav_h264::slice::parse_slice_header;
use oxideav_h264::sps::{parse_sps, Sps};

/// Build a minimal baseline SPS (4:2:0, 8-bit, 64×64, `max_num_ref_frames=4`,
/// `pic_order_cnt_type=2`, `log2_max_frame_num_minus4=4`).
fn build_sps() -> (Vec<u8>, Sps) {
    let mut bw = BitWriter::new();
    // profile_idc = 66 (Baseline).
    bw.write_bits(66, 8);
    // constraint_set_flags = 0, reserved_zero_2bits = 0.
    bw.write_bits(0, 8);
    // level_idc = 30.
    bw.write_bits(30, 8);
    // seq_parameter_set_id = 0.
    bw.write_ue(0);
    // log2_max_frame_num_minus4 = 4 → MaxFrameNum = 256.
    bw.write_ue(4);
    // pic_order_cnt_type = 2.
    bw.write_ue(2);
    // max_num_ref_frames = 4.
    bw.write_ue(4);
    // gaps_in_frame_num_value_allowed_flag = 0.
    bw.write_flag(false);
    // pic_width_in_mbs_minus1 = 3 → 4 MBs wide = 64.
    bw.write_ue(3);
    // pic_height_in_map_units_minus1 = 3 → 64 tall.
    bw.write_ue(3);
    // frame_mbs_only_flag = 1.
    bw.write_flag(true);
    // direct_8x8_inference_flag = 0.
    bw.write_flag(false);
    // frame_cropping_flag = 0.
    bw.write_flag(false);
    // vui_parameters_present_flag = 0.
    bw.write_flag(false);
    bw.write_rbsp_trailing_bits();
    let rbsp = bw.finish();
    let header = NalHeader {
        forbidden_zero_bit: 0,
        nal_ref_idc: 3,
        nal_unit_type: NalUnitType::Sps,
    };
    let sps = parse_sps(&header, &rbsp).expect("parse sps");
    (rbsp, sps)
}

/// Build a minimal baseline PPS (CAVLC, no weighted prediction).
fn build_pps() -> (Vec<u8>, Pps) {
    let mut bw = BitWriter::new();
    bw.write_ue(0); // pic_parameter_set_id
    bw.write_ue(0); // seq_parameter_set_id
    bw.write_flag(false); // entropy_coding_mode_flag
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
    let rbsp = bw.finish();
    let header = NalHeader {
        forbidden_zero_bit: 0,
        nal_ref_idc: 3,
        nal_unit_type: NalUnitType::Pps,
    };
    let pps = parse_pps(&header, &rbsp, None).expect("parse pps");
    (rbsp, pps)
}

/// A P-slice header whose RPLM flag is set and whose command list contains
/// one idc=0 short-term subtract survives the parser and produces an
/// [`RplmCommand::ShortTermSubtract`] entry that, when fed through
/// [`apply_rplm`], reorders a default-built L0.
#[test]
fn p_slice_rplm_idc0_end_to_end() {
    let (_sps_rbsp, sps) = build_sps();
    let (_pps_rbsp, pps) = build_pps();

    let mut bw = BitWriter::new();
    // first_mb_in_slice = 0.
    bw.write_ue(0);
    // slice_type = 5 (P, single-slice-type-per-picture variant).
    bw.write_ue(5);
    // pic_parameter_set_id = 0.
    bw.write_ue(0);
    // frame_num (8 bits for log2_max_frame_num_minus4 = 4) = 5.
    bw.write_bits(5, 8);
    // (pic_order_cnt_type == 2, no lsb/delta to emit.)
    // num_ref_idx_active_override_flag = 1, num_ref_idx_l0_active_minus1 = 3.
    bw.write_flag(true);
    bw.write_ue(3);
    // ref_pic_list_modification_flag_l0 = 1.
    bw.write_flag(true);
    // idc=0, delta=2.
    bw.write_ue(0);
    bw.write_ue(2);
    // idc=3 terminator.
    bw.write_ue(3);
    // dec_ref_pic_marking() — not IDR, adaptive_flag = 0.
    bw.write_flag(false);
    // slice_qp_delta = 0.
    bw.write_se(0);
    // disable_deblocking_filter_idc = 1 (no deblocking offsets).
    bw.write_ue(1);
    bw.write_rbsp_trailing_bits();
    let rbsp = bw.finish();

    let header = NalHeader {
        forbidden_zero_bit: 0,
        nal_ref_idc: 2,
        nal_unit_type: NalUnitType::SliceNonIdr,
    };
    let sh = parse_slice_header(&header, &rbsp, &sps, &pps).expect("parse slice header");
    assert_eq!(sh.rplm_l0.len(), 1);
    assert_eq!(
        sh.rplm_l0[0],
        RplmCommand::ShortTermSubtract {
            abs_diff_pic_num_minus1: 2
        }
    );
    assert!(sh.rplm_l1.is_empty());

    // Apply on top of a default-built L0 with four short-term refs.
    let mut dpb = Dpb::new(4, 4);
    for fn_ in 0..4u32 {
        dpb.insert_reference(Picture::new(4, 4), fn_, fn_ as i32, fn_ as i32);
    }
    dpb.update_frame_num_wrap(4, 256);
    let default = dpb.build_list0_p(sh.num_ref_idx_l0_active_minus1);
    let modified = apply_rplm(
        default,
        &sh.rplm_l0,
        sh.frame_num as i32,
        256,
        sh.num_ref_idx_l0_active_minus1,
        dpb.frames_raw(),
    )
    .expect("apply rplm");
    // CurrPicNum=5, delta=3 → picNum=2 → ref with frame_num_wrap=2 to pos 0.
    assert_eq!(modified[0].frame_num_wrap, 2);
    assert_eq!(modified.len(), 4);
}

/// A B-slice header with RPLM commands on both lists parses into
/// independent command vectors.
#[test]
fn b_slice_rplm_both_lists() {
    let (_sps_rbsp, sps) = build_sps();
    let (_pps_rbsp, pps) = build_pps();

    let mut bw = BitWriter::new();
    // first_mb_in_slice = 0.
    bw.write_ue(0);
    // slice_type = 6 (B, single-slice-type-per-picture variant).
    bw.write_ue(6);
    // pic_parameter_set_id = 0.
    bw.write_ue(0);
    // frame_num.
    bw.write_bits(8, 8);
    // direct_spatial_mv_pred_flag = 1 (B-slice).
    bw.write_flag(true);
    // num_ref_idx_active_override_flag = 1; L0=3, L1=3.
    bw.write_flag(true);
    bw.write_ue(3);
    bw.write_ue(3);
    // ref_pic_list_modification_flag_l0 = 1, idc=1 delta=0, idc=3.
    bw.write_flag(true);
    bw.write_ue(1);
    bw.write_ue(0);
    bw.write_ue(3);
    // ref_pic_list_modification_flag_l1 = 1, idc=2 lt=3, idc=3.
    bw.write_flag(true);
    bw.write_ue(2);
    bw.write_ue(3);
    bw.write_ue(3);
    // dec_ref_pic_marking: adaptive_flag = 0.
    bw.write_flag(false);
    bw.write_se(0); // slice_qp_delta
    bw.write_ue(1); // disable_deblocking_filter_idc = 1
    bw.write_rbsp_trailing_bits();
    let rbsp = bw.finish();

    let header = NalHeader {
        forbidden_zero_bit: 0,
        nal_ref_idc: 2,
        nal_unit_type: NalUnitType::SliceNonIdr,
    };
    let sh = parse_slice_header(&header, &rbsp, &sps, &pps).expect("parse b-slice header");
    assert_eq!(
        sh.rplm_l0,
        vec![RplmCommand::ShortTermAdd {
            abs_diff_pic_num_minus1: 0
        }]
    );
    assert_eq!(
        sh.rplm_l1,
        vec![RplmCommand::LongTerm {
            long_term_pic_num: 3
        }]
    );
}
