//! Round-451 — redundant coded picture gates (§7.4.2.2 / §7.4.1.2).
//!
//! When the active PPS carries `redundant_pic_cnt_present_flag`,
//! every slice header codes a §7.3.3 `redundant_pic_cnt`; slices
//! with a value greater than 0 belong to a REDUNDANT coded picture —
//! an approximation of (part of) the primary picture that a decoder
//! may use for error recovery and may otherwise discard. Decoding
//! primary data only, the decoder must produce the SAME output
//! whether or not redundant slices (with deliberately DIFFERENT
//! coded content) are present in the stream.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::bitstream::BitWriter;
use oxideav_h264::encoder::cavlc::CoeffTokenContext;
use oxideav_h264::encoder::macroblock::{write_intra16x16_mb, I16x16McbConfig};
use oxideav_h264::encoder::nal::build_nal_unit;
use oxideav_h264::encoder::pps::{build_baseline_pps_rbsp, BaselinePpsConfig};
use oxideav_h264::encoder::sps::{build_baseline_sps_rbsp, BaselineSpsConfig};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264::nal::NalUnitType;

const W: usize = 48;
const H: usize = 48;
const W_MBS: usize = W / 16;
const H_MBS: usize = H / 16;

fn parameter_sets() -> Vec<u8> {
    let sps = build_baseline_sps_rbsp(&BaselineSpsConfig {
        bit_depth_luma_minus8: 0,
        bit_depth_chroma_minus8: 0,
        seq_parameter_set_id: 0,
        level_idc: 20,
        width_in_mbs: W_MBS as u32,
        height_in_mbs: H_MBS as u32,
        log2_max_frame_num_minus4: 4,
        log2_max_poc_lsb_minus4: 4,
        max_num_ref_frames: 1,
        profile_idc: 66,
        chroma_format_idc: 1,
        separate_colour_plane: false,
        seq_scaling_lists: None,
        interlaced_fields: false,
        mbaff: false,
        vui: None,
    });
    let pps = build_baseline_pps_rbsp(&BaselinePpsConfig {
        redundant_pic_cnt_present_flag: true,
        pic_parameter_set_id: 0,
        seq_parameter_set_id: 0,
        pic_init_qp_minus26: 0,
        chroma_qp_index_offset: 0,
        weighted_pred_flag: false,
        weighted_bipred_idc: 0,
        entropy_coding_mode_flag: false,
        transform_8x8_mode_flag: false,
        pic_scaling_lists: None,
        chroma_format_idc: 1,
    });
    let mut out = build_nal_unit(3, NalUnitType::Sps, &sps);
    out.extend_from_slice(&build_nal_unit(3, NalUnitType::Pps, &pps));
    out
}

/// One I16x16-DC IDR access unit whose slice header codes the §7.3.3
/// `redundant_pic_cnt` (the active PPS carries the flag).
fn i16x16_idr(redundant_pic_cnt: u32, seed: u32) -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.ue(0); // first_mb_in_slice
    bw.ue(7); // slice_type: I, all slices in picture
    bw.ue(0); // pic_parameter_set_id
    bw.u(8, 0); // frame_num
    bw.ue(0); // idr_pic_id
    bw.u(8, 0); // pic_order_cnt_lsb
    bw.ue(redundant_pic_cnt); // §7.3.3 — redundant_pic_cnt
    bw.u(1, 0); // no_output_of_prior_pics_flag
    bw.u(1, 0); // long_term_reference_flag
    bw.se(0); // slice_qp_delta (QP 26)
    bw.ue(1); // disable_deblocking_filter_idc = 1

    let mut state = seed | 1;
    let mut next = |range: i32| -> i32 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state % (2 * range as u32 + 1)) as i32 - range
    };
    for _mb in 0..W_MBS * H_MBS {
        let mut dc = [0i32; 16];
        for slot in dc.iter_mut() {
            *slot = next(6);
        }
        let mut dc_cb = [0i32; 4];
        let mut dc_cr = [0i32; 4];
        for slot in dc_cb.iter_mut() {
            *slot = next(4);
        }
        for slot in dc_cr.iter_mut() {
            *slot = next(4);
        }
        write_intra16x16_mb(
            &mut bw,
            &I16x16McbConfig {
                pred_mode: 2,
                intra_chroma_pred_mode: 0,
                cbp_luma: 0,
                cbp_chroma: 1,
                mb_qp_delta: 0,
                luma_dc_levels_raster: dc,
                luma_ac_levels: [[0i32; 16]; 16],
                luma_ac_nc: [0i32; 16],
                chroma_dc_cb: dc_cb,
                chroma_dc_cr: dc_cr,
                chroma_ac_cb: [[0i32; 16]; 4],
                chroma_ac_cr: [[0i32; 16]; 4],
                chroma_ac_nc_cb: [0i32; 8],
                chroma_ac_nc_cr: [0i32; 8],
            },
            CoeffTokenContext::Numeric(0),
        )
        .expect("I_16x16 emit");
    }
    bw.rbsp_trailing_bits();
    build_nal_unit(3, NalUnitType::SliceIdr, &bw.into_bytes())
}

fn decode_all(stream: &[u8]) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream.to_vec()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    assert_eq!(dec.decode_error_count(), 0, "no skipped slices");
    let mut out = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => {
                assert_eq!(vf.planes.len(), 3);
                out.push((
                    vf.planes[0].data.to_vec(),
                    vf.planes[1].data.to_vec(),
                    vf.planes[2].data.to_vec(),
                ));
            }
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    out
}

/// A redundant slice (deliberately DIFFERENT coded content) following
/// the primary picture must not change the decode: one frame, equal to
/// the primary-only decode byte-for-byte.
#[test]
fn redundant_coded_picture_is_discarded() {
    let params = parameter_sets();
    let primary = i16x16_idr(0, 0x1DEA1);
    let redundant = i16x16_idr(1, 0x0DDBA11);

    let mut primary_only = params.clone();
    primary_only.extend_from_slice(&primary);
    let baseline = decode_all(&primary_only);
    assert_eq!(baseline.len(), 1);

    let mut with_redundant = params;
    with_redundant.extend_from_slice(&primary);
    with_redundant.extend_from_slice(&redundant);
    let frames = decode_all(&with_redundant);
    assert_eq!(
        frames.len(),
        1,
        "the redundant slice must not open a picture of its own"
    );
    assert_eq!(frames[0].0, baseline[0].0, "luma unchanged by redundancy");
    assert_eq!(frames[0].1, baseline[0].1, "Cb unchanged by redundancy");
    assert_eq!(frames[0].2, baseline[0].2, "Cr unchanged by redundancy");
}
