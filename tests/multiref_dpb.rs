//! Multi-reference DPB integration tests — exercises the new
//! [`oxideav_h264::dpb::Dpb`] path and the `decode_p_skip_mb` call site
//! with a `RefPicList0` longer than one entry.
//!
//! This is a direct API-level test that avoids the need for a real
//! `ref=2` fixture encoded by x264. It drives [`decode_p_skip_mb`] with
//! two candidate references (one bright, one dark) and flips `ref_idx`
//! between them to assert the L0 lookup actually threads through the
//! multi-entry list.

use oxideav_h264::p_mb::decode_p_skip_mb;
use oxideav_h264::picture::Picture;
use oxideav_h264::slice::{SliceHeader, SliceType};

fn skeleton_slice(num_active_minus1: u32) -> SliceHeader {
    SliceHeader {
        first_mb_in_slice: 0,
        slice_type_raw: 0,
        slice_type: SliceType::P,
        pic_parameter_set_id: 0,
        colour_plane_id: 0,
        frame_num: 0,
        field_pic_flag: false,
        bottom_field_flag: false,
        idr_pic_id: 0,
        pic_order_cnt_lsb: 0,
        delta_pic_order_cnt_bottom: 0,
        delta_pic_order_cnt: [0, 0],
        redundant_pic_cnt: 0,
        direct_spatial_mv_pred_flag: false,
        num_ref_idx_active_override_flag: false,
        num_ref_idx_l0_active_minus1: num_active_minus1,
        num_ref_idx_l1_active_minus1: 0,
        cabac_init_idc: 0,
        slice_qp_delta: 0,
        sp_for_switch_flag: false,
        slice_qs_delta: 0,
        disable_deblocking_filter_idc: 0,
        slice_alpha_c0_offset_div2: 0,
        slice_beta_offset_div2: 0,
        slice_data_bit_offset: 0,
        is_idr: false,
        pred_weight_table: None,
        idr_no_output_of_prior_pics_flag: false,
        idr_long_term_reference_flag: false,
        adaptive_ref_pic_marking_mode_flag: false,
        mmco_commands: Vec::new(),
        rplm_l0: Vec::new(),
        rplm_l1: Vec::new(),
    }
}

/// Build a picture with every sample set to `y_val` / `uv_val`.
fn constant_pic(mb_w: u32, mb_h: u32, y_val: u8, uv_val: u8) -> Picture {
    let mut pic = Picture::new(mb_w, mb_h);
    for p in pic.y.iter_mut() {
        *p = y_val;
    }
    for p in pic.cb.iter_mut() {
        *p = uv_val;
    }
    for p in pic.cr.iter_mut() {
        *p = uv_val;
    }
    pic
}

/// `decode_p_skip_mb` reads `ref_idx = 0` and must pick the first entry
/// in the list — demonstrated by changing the first entry between two
/// calls and seeing the output follow.
#[test]
fn p_skip_reads_list0_first_entry() {
    let ref_a = constant_pic(2, 2, 60, 128);
    let ref_b = constant_pic(2, 2, 200, 128);
    let sh = skeleton_slice(1);

    // ref_list0 = [&ref_a, &ref_b] → ref_idx 0 points to ref_a (luma 60).
    let mut out = Picture::new(2, 2);
    let list0 = [&ref_a, &ref_b];
    decode_p_skip_mb(&sh, 0, 0, &mut out, &list0, 26).expect("p_skip");
    let lstride = out.luma_stride();
    // Check the 16×16 luma region the MB covers.
    for y in 0..16 {
        for x in 0..16 {
            assert_eq!(
                out.y[y * lstride + x],
                60,
                "ref_a should dominate at ({x},{y}), got {}",
                out.y[y * lstride + x]
            );
        }
    }

    // Swap the order: ref_b first → ref_idx 0 should now pick luma 200.
    let mut out2 = Picture::new(2, 2);
    let list0b = [&ref_b, &ref_a];
    decode_p_skip_mb(&sh, 0, 0, &mut out2, &list0b, 26).expect("p_skip");
    for y in 0..16 {
        for x in 0..16 {
            assert_eq!(
                out2.y[y * lstride + x],
                200,
                "ref_b should dominate at ({x},{y}), got {}",
                out2.y[y * lstride + x]
            );
        }
    }
}

/// When `ref_list0` is empty the P_Skip call returns Err (an invalid bitstream).
#[test]
fn empty_list0_surfaces_error() {
    let sh = skeleton_slice(0);
    let mut out = Picture::new(2, 2);
    let empty: [&Picture; 0] = [];
    let err = decode_p_skip_mb(&sh, 0, 0, &mut out, &empty, 26);
    assert!(err.is_err(), "empty list0 must error, got {:?}", err);
}

/// List-0 construction in the DPB: after inserting three short-term
/// references with ascending frame_num, `build_list0_p(1)` returns the
/// two most recent ones in descending-frame_num order.
#[test]
fn dpb_list0_descending_short_term() {
    use oxideav_h264::dpb::Dpb;
    let mut dpb = Dpb::new(4, 0);
    for fn_ in 0..3u32 {
        dpb.insert_reference(Picture::new(1, 1), fn_, fn_ as i32, fn_ as i32);
    }
    dpb.update_frame_num_wrap(2, 16);
    let list = dpb.build_list0_p(1);
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].frame_num, 2);
    assert_eq!(list[1].frame_num, 1);
}

/// End-to-end: drive two I-frames through the front-end decoder,
/// confirm the second one is decoded (meaning the DPB cleared properly
/// on the second IDR and a fresh reference was installed).
#[test]
fn decoder_emits_both_of_two_idrs() {
    use oxideav_codec::CodecRegistry;
    use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
    use oxideav_h264::encoder::{H264Encoder, H264EncoderOptions};

    let mut codecs = CodecRegistry::new();
    oxideav_h264::register(&mut codecs);

    // Encode two gray I-frames at 16×16.
    let mut enc = H264Encoder::new(
        CodecId::new("h264"),
        16,
        16,
        H264EncoderOptions {
            qp: 22,
            ..Default::default()
        },
    )
    .expect("encoder");

    let mk = |pts: i64| oxideav_core::VideoFrame {
        format: oxideav_core::PixelFormat::Yuv420P,
        width: 16,
        height: 16,
        pts: Some(pts),
        time_base: TimeBase::new(1, 30),
        planes: vec![
            oxideav_core::frame::VideoPlane {
                stride: 16,
                data: vec![128u8; 16 * 16],
            },
            oxideav_core::frame::VideoPlane {
                stride: 8,
                data: vec![128u8; 8 * 8],
            },
            oxideav_core::frame::VideoPlane {
                stride: 8,
                data: vec![128u8; 8 * 8],
            },
        ],
    };

    oxideav_codec::Encoder::send_frame(&mut enc, &Frame::Video(mk(0))).unwrap();
    oxideav_codec::Encoder::send_frame(&mut enc, &Frame::Video(mk(1))).unwrap();
    oxideav_codec::Encoder::flush(&mut enc).unwrap();
    let mut bytes = Vec::new();
    loop {
        match oxideav_codec::Encoder::receive_packet(&mut enc) {
            Ok(p) => bytes.extend_from_slice(&p.data),
            Err(oxideav_core::Error::Eof) => break,
            Err(oxideav_core::Error::NeedMore) => break,
            Err(e) => panic!("recv packet: {e:?}"),
        }
    }
    assert!(!bytes.is_empty());

    let params = CodecParameters::video(CodecId::new("h264"));
    let mut dec = codecs.make_decoder(&params).expect("decoder");

    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");

    let mut count = 0;
    while let Ok(Frame::Video(_)) = dec.receive_frame() {
        count += 1;
    }
    assert_eq!(count, 2, "expected two decoded frames");
}
