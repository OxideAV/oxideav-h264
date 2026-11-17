//! Explicit weighted prediction at a bit depth above 8 (section 8.4.2.3.2).
//!
//! `luma_offset_l0` and `chroma_offset_l0` are coded in 8-bit units. Equations
//! 8-449 to 8-451 first form `o = offset * (1 << (BitDepth - 8))`, so a High 10
//! stream's offsets are four times their coded values when they are added to
//! the samples.
//!
//! The stream is built by hand: a 16x16 High 10 (4:2:0) IDR picture holding one
//! I_PCM macroblock, then a P picture holding one P_Skip macroblock (zero
//! motion, reference index 0) whose slice header carries a
//! `pred_weight_table()`. Every sample of the P picture is then fixed by the
//! standard's equation 8-449 applied to the PCM samples, with no residual and
//! no deblocking in the way.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const BIT_DEPTH: u32 = 10;
const MAX: i32 = (1 << BIT_DEPTH) - 1;

#[derive(Default)]
struct Bits {
    bytes: Vec<u8>,
    used: u32,
}

impl Bits {
    fn bit(&mut self, b: u32) {
        if self.used == 0 {
            self.bytes.push(0);
        }
        if b != 0 {
            *self.bytes.last_mut().unwrap() |= 0x80 >> self.used;
        }
        self.used = (self.used + 1) % 8;
    }
    fn u(&mut self, n: u32, v: u32) {
        for i in (0..n).rev() {
            self.bit((v >> i) & 1);
        }
    }
    fn ue(&mut self, v: u32) {
        let x = v + 1;
        let len = 32 - x.leading_zeros();
        self.u(len - 1, 0);
        self.u(len, x);
    }
    fn se(&mut self, v: i32) {
        self.ue(if v > 0 {
            2 * v as u32 - 1
        } else {
            (-2 * v) as u32
        });
    }
    fn align_zero(&mut self) {
        while self.used != 0 {
            self.bit(0);
        }
    }
    fn trailing(&mut self) {
        self.bit(1);
        self.align_zero();
    }
}

/// A NAL unit with a start code and emulation prevention bytes.
fn nal(ref_idc: u8, unit_type: u8, rbsp: &[u8]) -> Vec<u8> {
    let mut out = vec![0, 0, 0, 1, (ref_idc << 5) | unit_type];
    let mut zeros = 0;
    for &b in rbsp {
        if zeros >= 2 && b <= 3 {
            out.push(3);
            zeros = 0;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
    }
    out
}

fn sps() -> Vec<u8> {
    let mut b = Bits::default();
    b.u(8, 110); // High 10
    b.u(8, 0);
    b.u(8, 30);
    b.ue(0); // seq_parameter_set_id
    b.ue(1); // chroma_format_idc: 4:2:0
    b.ue(BIT_DEPTH - 8);
    b.ue(BIT_DEPTH - 8);
    b.bit(0); // qpprime_y_zero_transform_bypass_flag
    b.bit(0); // seq_scaling_matrix_present_flag
    b.ue(0); // log2_max_frame_num_minus4
    b.ue(2); // pic_order_cnt_type
    b.ue(1); // max_num_ref_frames
    b.bit(0); // gaps_in_frame_num_value_allowed_flag
    b.ue(0); // pic_width_in_mbs_minus1
    b.ue(0); // pic_height_in_map_units_minus1
    b.bit(1); // frame_mbs_only_flag
    b.bit(1); // direct_8x8_inference_flag
    b.bit(0); // frame_cropping_flag
    b.bit(0); // vui_parameters_present_flag
    b.trailing();
    nal(3, 7, &b.bytes)
}

fn pps() -> Vec<u8> {
    let mut b = Bits::default();
    b.ue(0); // pic_parameter_set_id
    b.ue(0); // seq_parameter_set_id
    b.bit(0); // entropy_coding_mode_flag: CAVLC
    b.bit(0); // bottom_field_pic_order_in_frame_present_flag
    b.ue(0); // num_slice_groups_minus1
    b.ue(0); // num_ref_idx_l0_default_active_minus1
    b.ue(0); // num_ref_idx_l1_default_active_minus1
    b.bit(1); // weighted_pred_flag
    b.u(2, 0); // weighted_bipred_idc
    b.se(0); // pic_init_qp_minus26
    b.se(0); // pic_init_qs_minus26
    b.se(0); // chroma_qp_index_offset
    b.bit(1); // deblocking_filter_control_present_flag
    b.bit(0); // constrained_intra_pred_flag
    b.bit(0); // redundant_pic_cnt_present_flag
    b.trailing();
    nal(3, 8, &b.bytes)
}

/// Samples of the I_PCM macroblock: luma, then Cb, then Cr.
fn pcm_samples() -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let y = (0..256).map(|i| 100 + (i * 3) % 800).collect();
    let cb = (0..64).map(|i| 200 + (i * 11) % 500).collect();
    let cr = (0..64).map(|i| 300 + (i * 7) % 600).collect();
    (y, cb, cr)
}

fn idr_slice() -> Vec<u8> {
    let (y, cb, cr) = pcm_samples();
    let mut b = Bits::default();
    b.ue(0); // first_mb_in_slice
    b.ue(7); // slice_type: I
    b.ue(0); // pic_parameter_set_id
    b.u(4, 0); // frame_num
    b.ue(0); // idr_pic_id
    b.bit(0); // no_output_of_prior_pics_flag
    b.bit(0); // long_term_reference_flag
    b.se(0); // slice_qp_delta
    b.ue(1); // disable_deblocking_filter_idc
    b.ue(25); // mb_type: I_PCM
    b.align_zero();
    for &s in y.iter().chain(&cb).chain(&cr) {
        b.u(BIT_DEPTH, s as u32);
    }
    b.trailing();
    nal(3, 5, &b.bytes)
}

/// Coded (8-bit unit) explicit weights: (weight, offset) for luma, Cb and Cr.
struct Coded {
    luma_log2_denom: u32,
    chroma_log2_denom: u32,
    luma: (i32, i32),
    cb: (i32, i32),
    cr: (i32, i32),
}

fn p_slice(c: &Coded) -> Vec<u8> {
    let mut b = Bits::default();
    b.ue(0); // first_mb_in_slice
    b.ue(5); // slice_type: P
    b.ue(0); // pic_parameter_set_id
    b.u(4, 1); // frame_num
    b.bit(0); // num_ref_idx_active_override_flag
    b.bit(0); // ref_pic_list_modification_flag_l0
    b.ue(c.luma_log2_denom);
    b.ue(c.chroma_log2_denom);
    b.bit(1); // luma_weight_l0_flag
    b.se(c.luma.0);
    b.se(c.luma.1);
    b.bit(1); // chroma_weight_l0_flag
    b.se(c.cb.0);
    b.se(c.cb.1);
    b.se(c.cr.0);
    b.se(c.cr.1);
    b.bit(0); // adaptive_ref_pic_marking_mode_flag
    b.se(0); // slice_qp_delta
    b.ue(1); // disable_deblocking_filter_idc
    b.ue(1); // mb_skip_run: the one macroblock is P_Skip
    b.trailing();
    nal(2, 1, &b.bytes)
}

fn decode(stream: &[u8]) -> Vec<[Vec<i32>; 3]> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream.to_vec()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut out = Vec::new();
    while let Ok(f) = dec.receive_frame() {
        if let Frame::Video(vf) = f {
            assert_eq!(vf.planes.len(), 3);
            let plane = |i: usize| -> Vec<i32> {
                vf.planes[i]
                    .data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| i32::from(u16::from_le_bytes(*c)))
                    .collect()
            };
            out.push([plane(0), plane(1), plane(2)]);
        }
    }
    assert_eq!(dec.decode_error_count(), 0);
    out
}

/// Equation 8-449 (single list) with the offset scaled for the bit depth.
fn weighted(sample: i32, log2_denom: u32, (w, o): (i32, i32)) -> i32 {
    let o = o << (BIT_DEPTH - 8);
    let v = if log2_denom >= 1 {
        ((sample * w + (1 << (log2_denom - 1))) >> log2_denom) + o
    } else {
        sample * w + o
    };
    v.clamp(0, MAX)
}

fn check(c: &Coded) {
    let mut stream = sps();
    stream.extend(pps());
    stream.extend(idr_slice());
    stream.extend(p_slice(c));
    let frames = decode(&stream);
    assert_eq!(frames.len(), 2, "IDR picture and P picture");

    let (y, cb, cr) = pcm_samples();
    assert_eq!(frames[0][0], y, "the I_PCM luma samples come back exactly");
    assert_eq!(frames[0][1], cb);
    assert_eq!(frames[0][2], cr);

    let expect_y: Vec<i32> = y
        .iter()
        .map(|&s| weighted(s, c.luma_log2_denom, c.luma))
        .collect();
    let expect_cb: Vec<i32> = cb
        .iter()
        .map(|&s| weighted(s, c.chroma_log2_denom, c.cb))
        .collect();
    let expect_cr: Vec<i32> = cr
        .iter()
        .map(|&s| weighted(s, c.chroma_log2_denom, c.cr))
        .collect();
    assert_eq!(frames[1][0], expect_y, "luma");
    assert_eq!(frames[1][1], expect_cb, "Cb");
    assert_eq!(frames[1][2], expect_cr, "Cr");
}

/// Unit weights with nonzero offsets: the P picture is the reference plus four
/// times each coded offset at 10 bits.
#[test]
fn offsets_are_scaled_by_bit_depth_with_unit_weights() {
    check(&Coded {
        luma_log2_denom: 5,
        chroma_log2_denom: 5,
        luma: (32, 7),
        cb: (32, -3),
        cr: (32, 5),
    });
}

/// Non-unit weights and a different denominator for chroma.
#[test]
fn offsets_are_scaled_by_bit_depth_with_scaling_weights() {
    check(&Coded {
        luma_log2_denom: 6,
        chroma_log2_denom: 4,
        luma: (40, -9),
        cb: (20, 12),
        cr: (11, -6),
    });
}
