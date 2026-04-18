//! Hand-crafted 10-bit CAVLC I_PCM test. A single 16×16 macroblock with
//! `mb_type = I_PCM` carries 16 × 16 + 2 × 64 raw 10-bit samples byte-
//! aligned after the slice header. The decoder must read each sample as
//! a `bit_depth`-wide unsigned integer and write it into the u16
//! `Picture` planes unclipped.
//!
//! x264 doesn't emit I_PCM macroblocks in normal encoder operation, so
//! this test builds the bitstream itself — a minimal High 10 SPS /
//! baseline-ish PPS / IDR slice pair that the decoder's 10-bit gate
//! accepts.

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, PixelFormat, TimeBase};
use oxideav_h264::decoder::H264Decoder;

struct BitWriter {
    bits: Vec<u8>,
}

impl BitWriter {
    fn new() -> Self {
        Self { bits: Vec::new() }
    }
    fn write_bit(&mut self, b: u8) {
        self.bits.push(b & 1);
    }
    fn write_bits(&mut self, v: u64, n: u32) {
        for i in (0..n).rev() {
            self.bits.push(((v >> i) & 1) as u8);
        }
    }
    fn write_u(&mut self, v: u64, n: u32) {
        self.write_bits(v, n);
    }
    fn write_ue(&mut self, v: u32) {
        let x = (v + 1) as u64;
        let bits = 64 - x.leading_zeros();
        let zeros = bits - 1;
        for _ in 0..zeros {
            self.write_bit(0);
        }
        self.write_bits(x, bits);
    }
    fn write_se(&mut self, v: i32) {
        let mapped = if v <= 0 {
            (-v) as u32 * 2
        } else {
            v as u32 * 2 - 1
        };
        self.write_ue(mapped);
    }
    fn align_to_byte_with_stop_bit(&mut self) {
        self.write_bit(1);
        while self.bits.len() % 8 != 0 {
            self.write_bit(0);
        }
    }
    fn into_bytes(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.bits.len() / 8);
        for chunk in self.bits.chunks(8) {
            let mut b = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                b |= bit << (7 - i);
            }
            out.push(b);
        }
        out
    }
}

fn build_sps_high10_rbsp() -> Vec<u8> {
    // High 10 profile (110), 4:2:0, bit depth 10.
    let profile_idc: u8 = 110;
    let constraint_flags: u8 = 0;
    let level_idc: u8 = 30;

    let mut bw = BitWriter::new();
    bw.write_ue(0); // seq_parameter_set_id
    // High profile — chroma_format_idc, bit depths, scaling lists.
    bw.write_ue(1); // chroma_format_idc = 1 (4:2:0)
    bw.write_ue(2); // bit_depth_luma_minus8 = 2 → 10-bit
    bw.write_ue(2); // bit_depth_chroma_minus8 = 2 → 10-bit
    bw.write_bit(0); // qpprime_y_zero_transform_bypass_flag
    bw.write_bit(0); // seq_scaling_matrix_present_flag
    bw.write_ue(0); // log2_max_frame_num_minus4
    bw.write_ue(0); // pic_order_cnt_type
    bw.write_ue(0); // log2_max_pic_order_cnt_lsb_minus4
    bw.write_ue(1); // max_num_ref_frames
    bw.write_bit(0); // gaps_in_frame_num_value_allowed_flag
    bw.write_ue(0); // pic_width_in_mbs_minus1 = 0 → 16 px
    bw.write_ue(0); // pic_height_in_map_units_minus1 = 0 → 16 px
    bw.write_bit(1); // frame_mbs_only_flag
    bw.write_bit(0); // direct_8x8_inference_flag
    bw.write_bit(0); // frame_cropping_flag
    bw.write_bit(0); // vui_parameters_present_flag
    bw.align_to_byte_with_stop_bit();
    let body = bw.into_bytes();

    let mut out = Vec::with_capacity(3 + body.len());
    out.push(profile_idc);
    out.push(constraint_flags);
    out.push(level_idc);
    out.extend_from_slice(&body);
    out
}

fn build_pps_cavlc_rbsp() -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.write_ue(0); // pic_parameter_set_id
    bw.write_ue(0); // seq_parameter_set_id
    bw.write_bit(0); // entropy_coding_mode_flag = 0 → CAVLC
    bw.write_bit(0); // bottom_field_pic_order_in_frame_present_flag
    bw.write_ue(0); // num_slice_groups_minus1
    bw.write_ue(0); // num_ref_idx_l0_default_active_minus1
    bw.write_ue(0); // num_ref_idx_l1_default_active_minus1
    bw.write_bit(0); // weighted_pred_flag
    bw.write_u(0, 2); // weighted_bipred_idc
    bw.write_se(0); // pic_init_qp_minus26 → QP = 26
    bw.write_se(0); // pic_init_qs_minus26
    bw.write_se(0); // chroma_qp_index_offset
    bw.write_bit(1); // deblocking_filter_control_present_flag
    bw.write_bit(0); // constrained_intra_pred_flag
    bw.write_bit(0); // redundant_pic_cnt_present_flag
    bw.align_to_byte_with_stop_bit();
    bw.into_bytes()
}

/// Build an IDR slice RBSP consisting of a single CAVLC `mb_type = 25`
/// (I_PCM) macroblock with explicit 10-bit samples for every plane.
fn build_idr_slice_cavlc_ipcm_10bit(
    luma: &[u16; 256],
    cb: &[u16; 64],
    cr: &[u16; 64],
) -> Vec<u8> {
    let mut bw = BitWriter::new();
    // Slice header — §7.3.3.
    bw.write_ue(0); // first_mb_in_slice
    bw.write_ue(7); // slice_type = I (7)
    bw.write_ue(0); // pic_parameter_set_id
    bw.write_u(0, 4); // frame_num
    bw.write_ue(0); // idr_pic_id
    bw.write_u(0, 4); // pic_order_cnt_lsb
    bw.write_bit(0); // no_output_of_prior_pics_flag
    bw.write_bit(0); // long_term_reference_flag
    bw.write_se(0); // slice_qp_delta
    bw.write_ue(1); // disable_deblocking_filter_idc = 1

    // Macroblock — `mb_type = 25` is the 26th code in the I-slice table
    // (I_PCM). `read_ue` of 25 encodes as 26 → prefix zeros + bits.
    bw.write_ue(25);

    // §7.3.5.3 pcm_alignment_zero_bit — pad to byte boundary with zeros.
    while bw.bits.len() % 8 != 0 {
        bw.write_bit(0);
    }

    // 10-bit PCM samples, big-endian per §7.4.5.
    for &v in luma.iter() {
        bw.write_u(v as u64, 10);
    }
    for &v in cb.iter() {
        bw.write_u(v as u64, 10);
    }
    for &v in cr.iter() {
        bw.write_u(v as u64, 10);
    }

    // End-of-slice marker — RBSP stop bit + byte alignment.
    bw.align_to_byte_with_stop_bit();
    bw.into_bytes()
}

fn wrap_nalu(nal_unit_type: u8, nal_ref_idc: u8, rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + rbsp.len() + 4);
    let header = ((nal_ref_idc & 0x03) << 5) | (nal_unit_type & 0x1F);
    out.push(header);
    let mut zero_run = 0u8;
    for &b in rbsp {
        if zero_run >= 2 && b <= 0x03 {
            out.push(0x03);
            zero_run = 0;
        }
        out.push(b);
        if b == 0 {
            zero_run += 1;
        } else {
            zero_run = 0;
        }
    }
    out
}

fn build_annex_b_packet(
    luma: &[u16; 256],
    cb: &[u16; 64],
    cr: &[u16; 64],
) -> Vec<u8> {
    let sps_rbsp = build_sps_high10_rbsp();
    let pps_rbsp = build_pps_cavlc_rbsp();
    let idr_rbsp = build_idr_slice_cavlc_ipcm_10bit(luma, cb, cr);

    let sps = wrap_nalu(7, 3, &sps_rbsp);
    let pps = wrap_nalu(8, 3, &pps_rbsp);
    let idr = wrap_nalu(5, 3, &idr_rbsp);

    let mut pkt = Vec::new();
    pkt.extend_from_slice(&[0, 0, 0, 1]);
    pkt.extend_from_slice(&sps);
    pkt.extend_from_slice(&[0, 0, 0, 1]);
    pkt.extend_from_slice(&pps);
    pkt.extend_from_slice(&[0, 0, 0, 1]);
    pkt.extend_from_slice(&idr);
    pkt
}

fn unpack_u16_le(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

#[test]
fn cavlc_ipcm_10bit_single_mb_round_trip() {
    // Unique per-pixel values across 10-bit range so any
    // stride/bit-depth bug shows up immediately.
    let mut luma = [0u16; 256];
    for (i, v) in luma.iter_mut().enumerate() {
        *v = (i * 4 & 0x3FF) as u16;
    }
    let mut cb = [0u16; 64];
    for (i, v) in cb.iter_mut().enumerate() {
        *v = (512 + i * 8) as u16 & 0x3FF;
    }
    let mut cr = [0u16; 64];
    for (i, v) in cr.iter_mut().enumerate() {
        *v = (100 + i * 13) as u16 & 0x3FF;
    }

    let pkt_data = build_annex_b_packet(&luma, &cb, &cr);

    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 90_000), pkt_data)
        .with_pts(0)
        .with_keyframe(true);
    dec.send_packet(&pkt).expect("send_packet");
    let frame = match dec.receive_frame().expect("receive_frame") {
        Frame::Video(f) => f,
        other => panic!(
            "expected video frame, got {:?}",
            std::mem::discriminant(&other)
        ),
    };

    assert_eq!(frame.width, 16);
    assert_eq!(frame.height, 16);
    assert_eq!(frame.format, PixelFormat::Yuv420P10Le);
    assert_eq!(frame.planes.len(), 3);

    let dec_y = unpack_u16_le(&frame.planes[0].data);
    let dec_cb = unpack_u16_le(&frame.planes[1].data);
    let dec_cr = unpack_u16_le(&frame.planes[2].data);

    for (i, &v) in dec_y.iter().enumerate() {
        assert_eq!(v, luma[i], "Y[{i}] expected {}, got {v}", luma[i]);
    }
    for (i, &v) in dec_cb.iter().enumerate() {
        assert_eq!(v, cb[i], "Cb[{i}] expected {}, got {v}", cb[i]);
    }
    for (i, &v) in dec_cr.iter().enumerate() {
        assert_eq!(v, cr[i], "Cr[{i}] expected {}, got {v}", cr[i]);
    }
}
