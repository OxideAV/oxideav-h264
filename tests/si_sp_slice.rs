//! Integration tests for SI / SP slice decode — ITU-T H.264 §7.3.5 /
//! §7.4.5 (Table 7-12 / 7-13) / §8.6.
//!
//! x264 does not emit SI or SP slices, so these tests hand-craft the
//! smallest legal SI and SP pictures (16×16 = one macroblock each) and
//! feed the bytes to `H264Decoder`.
//!
//! Coverage:
//!
//! * SI — an IDR picture whose one macroblock is SI (`mb_type = 0`
//!   under Table 7-12). The SI macroblock is wire-identical to I_NxN;
//!   every 4×4 block uses DC intra prediction and emits a zero-residual
//!   CAVLC payload, so the reconstruction is Y = Cb = Cr = 128.
//!
//! * SP — a two-picture stream: IDR I-slice (reference) followed by a
//!   primary SP slice (`sp_for_switch_flag = 0`, `QS == QP`) whose one
//!   macroblock is `P_Skip`. With zero MV, the SP picture duplicates
//!   the IDR samples (Y = Cb = Cr = 128).

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::decoder::H264Decoder;

// ---------------------------------------------------------------------------
// BitWriter — MSB-first, supports ue/se Exp-Golomb.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// SPS / PPS.
// ---------------------------------------------------------------------------

fn build_sps_rbsp() -> Vec<u8> {
    // Baseline profile (66) is what x264 emits for CAVLC streams. The
    // spec gates SI/SP at Extended profile but the decoder is agnostic —
    // the slice-type dispatch looks at `slice_type` alone.
    let profile_idc: u8 = 66;
    let constraint_flags: u8 = 0;
    let level_idc: u8 = 30;

    let mut bw = BitWriter::new();
    bw.write_ue(0); // seq_parameter_set_id
    bw.write_ue(0); // log2_max_frame_num_minus4
    bw.write_ue(2); // pic_order_cnt_type = 2 (no lsb/delta — every pic advances POC)
    bw.write_ue(1); // max_num_ref_frames (SP needs a reference)
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

fn build_pps_rbsp() -> Vec<u8> {
    // CAVLC PPS — entropy_coding_mode_flag=0.
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
    bw.write_se(0); // pic_init_qs_minus26 → QS = 26 (matches QP)
    bw.write_se(0); // chroma_qp_index_offset
    bw.write_bit(1); // deblocking_filter_control_present_flag
    bw.write_bit(0); // constrained_intra_pred_flag
    bw.write_bit(0); // redundant_pic_cnt_present_flag
    bw.align_to_byte_with_stop_bit();
    bw.into_bytes()
}

// ---------------------------------------------------------------------------
// NAL emission helpers.
// ---------------------------------------------------------------------------

fn emulation_prevent(rbsp: &[u8]) -> Vec<u8> {
    // §7.4.1.1 — insert 0x03 after any 0x0000 pair that would otherwise
    // create a start-code emulation. The hand-crafted payloads are short
    // and rarely trigger this, but we run the check for safety.
    let mut out = Vec::with_capacity(rbsp.len());
    let mut zeros = 0;
    for &b in rbsp {
        if zeros >= 2 && b <= 0x03 {
            out.push(0x03);
            zeros = 0;
        }
        out.push(b);
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
    }
    out
}

fn nalu(nal_ref_idc: u8, nal_unit_type: u8, rbsp: &[u8]) -> Vec<u8> {
    let header = (nal_ref_idc << 5) | nal_unit_type;
    let mut out = vec![header];
    out.extend_from_slice(&emulation_prevent(rbsp));
    out
}

// ---------------------------------------------------------------------------
// CAVLC helpers — emit the minimal payload for an I_NxN/SI MB with all
// DC intra modes and zero residual, and for a P_Skip macroblock.
// ---------------------------------------------------------------------------

/// Emit the macroblock layer for a zero-residual I_NxN (or SI) macroblock
/// using I4x4 DC_PRED on every block and IntraChromaDC. All 16 luma
/// AC blocks, both chroma DC, and 8 chroma AC residuals are coded as
/// `coeff_token` = 0 (TotalCoeff=0, TrailingOnes=0 — a single '1' bit in
/// the nC=0 VLC class). No mb_qp_delta is emitted because `cbp == 0`
/// and the MB is I_NxN (I_16x16 is the only I MB that requires
/// mb_qp_delta when cbp is zero).
fn emit_inxn_zero_residual(bw: &mut BitWriter, mb_type: u32) {
    bw.write_ue(mb_type); // SI: 0 under Table 7-12; I_NxN: 0 under Table 7-11.
                          // transform_size_8x8_flag is conditional on
                          // pps.transform_8x8_mode_flag, which is 0 in
                          // `build_pps_rbsp`.
                          // 16 × prev_intra4x4_pred_mode_flag = 1 (use predicted mode).
    for _ in 0..16 {
        bw.write_bit(1);
    }
    bw.write_ue(0); // intra_chroma_pred_mode = DC (0).
                    // coded_block_pattern = 0 → me(v) codeNum = 3 per Table 9-4
                    // (`ME_INTRA_4_2_0[3] == 0`). Emitted as ue(v) codeNum.
    bw.write_ue(3); // ChromaArrayType=1, intra CBP 0 → me(v) codeNum 3.
                    // No mb_qp_delta because CBP == 0 and MB is I_NxN.
                    // Residual: 16 AC blocks + chroma DC + chroma AC, all zero. But
                    // when cbp_luma == 0 no luma AC blocks are read, and when
                    // cbp_chroma == 0 neither chroma DC nor chroma AC are read. So
                    // we emit nothing more.
}

/// Emit a single P_Skip macroblock as CAVLC mb_skip_run = 1 (then 0 to
/// signal "no more skips" for the next coded MB, but when we have a
/// 1-MB picture the trailing skip_run is all we need and the loop ends).
/// Wait — for a 1-MB picture, skip_run == 1 means "skip this MB"; after
/// the loop ends with `mb_addr >= total_mbs`, no coded MB syntax is
/// consumed. So a single `skip_run` ue(v) of value 1 suffices.
fn emit_p_skip_one_mb(bw: &mut BitWriter) {
    bw.write_ue(1);
}

// ---------------------------------------------------------------------------
// Slice construction.
// ---------------------------------------------------------------------------

/// SI-slice IDR — `slice_type = 9` (SI, single-type-per-picture variant).
/// One macroblock: SI mb_type = 0 (wire-identical to I_NxN with DC
/// intra modes and zero residual).
fn build_si_idr_slice_rbsp() -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.write_ue(0); // first_mb_in_slice
                    // slice_type = 9 → SI (Table 7-6: 4=SI, 9=SI when
                    // every slice in the picture is the same type).
    bw.write_ue(9);
    bw.write_ue(0); // pic_parameter_set_id
                    // frame_num (4 bits: log2_max_frame_num_minus4 = 0 → 4).
    bw.write_u(0, 4);
    bw.write_ue(0); // idr_pic_id
                    // pic_order_cnt_type == 2 — no POC syntax.
                    // No ref_pic_list_modification for SI (same as I).
                    // dec_ref_pic_marking (IDR): no_output_of_prior_pics_flag,
                    // long_term_reference_flag.
    bw.write_bit(0); // no_output_of_prior_pics_flag
    bw.write_bit(0); // long_term_reference_flag
                     // cabac_init_idc skipped (CAVLC).
    bw.write_se(0); // slice_qp_delta → QP = 26.
                    // slice_qs_delta for SP/SI → QS = 26 (matches QP, §8.6.1 identity).
    bw.write_se(0);
    bw.write_ue(1); // disable_deblocking_filter_idc = 1
                    // (no alpha/beta offsets since idc == 1)

    // slice_data — single SI MB, zero-residual I_NxN DC.
    emit_inxn_zero_residual(&mut bw, 0);
    bw.align_to_byte_with_stop_bit();
    bw.into_bytes()
}

/// I-slice IDR for use as a reference before the SP slice. One MB,
/// I_NxN with DC intra modes, zero residual. `slice_type = 7` (I-only).
fn build_i_idr_slice_rbsp() -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.write_ue(0); // first_mb_in_slice
    bw.write_ue(7); // slice_type = 7 → I (single-type-per-picture)
    bw.write_ue(0); // pic_parameter_set_id
    bw.write_u(0, 4); // frame_num
    bw.write_ue(0); // idr_pic_id
    bw.write_bit(0); // no_output_of_prior_pics_flag
    bw.write_bit(0); // long_term_reference_flag
    bw.write_se(0); // slice_qp_delta
    bw.write_ue(1); // disable_deblocking_filter_idc = 1

    emit_inxn_zero_residual(&mut bw, 0);
    bw.align_to_byte_with_stop_bit();
    bw.into_bytes()
}

/// Non-IDR SP slice that references the preceding IDR. `slice_type = 8`
/// (SP single-type-per-picture). One macroblock: P_Skip (MV = 0,
/// ref_idx = 0) — reproduces the IDR's samples bit-exact.
fn build_sp_slice_rbsp() -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.write_ue(0); // first_mb_in_slice
                    // slice_type = 8 → SP (Table 7-6: 3=SP, 8=SP single-type).
    bw.write_ue(8);
    bw.write_ue(0); // pic_parameter_set_id
    bw.write_u(1, 4); // frame_num = 1
                      // SP is P-shaped: num_ref_idx_active_override_flag for P/SP/B.
    bw.write_bit(0); // num_ref_idx_active_override_flag = 0 (use PPS default = 1)
                     // ref_pic_list_modification: SP has list 0.
                     // ref_pic_list_modification_flag_l0 = 0 (keep default list).
    bw.write_bit(0);
    // pred_weight_table skipped (weighted_pred_flag = 0 in PPS).
    // dec_ref_pic_marking: nal_ref_idc != 0 → adaptive_ref_pic_marking_mode_flag.
    bw.write_bit(0); // adaptive_ref_pic_marking_mode_flag = 0 (sliding window)
                     // cabac_init_idc skipped (CAVLC).
    bw.write_se(0); // slice_qp_delta → QP = 26.
                    // SP specific: sp_for_switch_flag + slice_qs_delta.
    bw.write_bit(0); // sp_for_switch_flag = 0 (primary SP)
    bw.write_se(0); // slice_qs_delta → QS = 26 (matches QP).
    bw.write_ue(1); // disable_deblocking_filter_idc = 1

    emit_p_skip_one_mb(&mut bw);
    bw.align_to_byte_with_stop_bit();
    bw.into_bytes()
}

// ---------------------------------------------------------------------------
// Stream framing and decode.
// ---------------------------------------------------------------------------

fn prefix(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + nal.len());
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.extend_from_slice(nal);
    out
}

#[test]
fn si_slice_single_mb_decodes() {
    let sps = nalu(3, 7, &build_sps_rbsp());
    let pps = nalu(3, 8, &build_pps_rbsp());
    let slice = nalu(3, 5, &build_si_idr_slice_rbsp());

    let mut packet_data = Vec::new();
    packet_data.extend_from_slice(&prefix(&sps));
    packet_data.extend_from_slice(&prefix(&pps));
    packet_data.extend_from_slice(&prefix(&slice));

    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 90_000), packet_data)
        .with_pts(0)
        .with_keyframe(true);
    dec.send_packet(&pkt).expect("send_packet");
    let frame = match dec.receive_frame() {
        Ok(Frame::Video(f)) => f,
        other => panic!("expected video frame, got {:?}", other.map(|_| ())),
    };
    assert_eq!(frame.width, 16);
    assert_eq!(frame.height, 16);
    // DC intra prediction on the first macroblock (no neighbours) yields
    // Y = Cb = Cr = 128 (§8.3.1.2.3 / §8.3.4.3 "no neighbour available"
    // clause). With zero residual the reconstruction equals the prediction.
    let y = &frame.planes[0].data;
    let cb = &frame.planes[1].data;
    let cr = &frame.planes[2].data;
    assert_eq!(y.len(), 16 * 16);
    assert_eq!(cb.len(), 8 * 8);
    assert_eq!(cr.len(), 8 * 8);
    for &p in y {
        assert_eq!(p, 128, "SI luma expected Y=128, got {p}");
    }
    for &p in cb {
        assert_eq!(p, 128, "SI chroma Cb expected 128, got {p}");
    }
    for &p in cr {
        assert_eq!(p, 128, "SI chroma Cr expected 128, got {p}");
    }
}

#[test]
fn sp_slice_single_mb_decodes() {
    let sps = nalu(3, 7, &build_sps_rbsp());
    let pps = nalu(3, 8, &build_pps_rbsp());
    let idr = nalu(3, 5, &build_i_idr_slice_rbsp());
    // SP slice carries `nal_ref_idc != 0` so the decoder snapshots its
    // output into the DPB, but we only decode one reference + one SP,
    // so nal_ref_idc = 2 is safe.
    let sp = nalu(2, 1, &build_sp_slice_rbsp());

    let mut packet_data = Vec::new();
    packet_data.extend_from_slice(&prefix(&sps));
    packet_data.extend_from_slice(&prefix(&pps));
    packet_data.extend_from_slice(&prefix(&idr));
    packet_data.extend_from_slice(&prefix(&sp));

    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 90_000), packet_data)
        .with_pts(0)
        .with_keyframe(true);
    dec.send_packet(&pkt).expect("send_packet");

    // First frame: the IDR. Second frame: the SP-slice picture.
    let idr_frame = match dec.receive_frame() {
        Ok(Frame::Video(f)) => f,
        other => panic!("expected IDR video frame, got {:?}", other.map(|_| ())),
    };
    assert_eq!(idr_frame.width, 16);
    assert_eq!(idr_frame.height, 16);
    for &p in &idr_frame.planes[0].data {
        assert_eq!(p, 128, "IDR luma expected 128, got {p}");
    }
    let sp_frame = match dec.receive_frame() {
        Ok(Frame::Video(f)) => f,
        other => panic!("expected SP video frame, got {:?}", other.map(|_| ())),
    };
    assert_eq!(sp_frame.width, 16);
    assert_eq!(sp_frame.height, 16);
    // P_Skip reproduces the reference's samples — same grey picture.
    for &p in &sp_frame.planes[0].data {
        assert_eq!(p, 128, "SP luma expected 128, got {p}");
    }
    for &p in &sp_frame.planes[1].data {
        assert_eq!(p, 128, "SP chroma Cb expected 128, got {p}");
    }
    for &p in &sp_frame.planes[2].data {
        assert_eq!(p, 128, "SP chroma Cr expected 128, got {p}");
    }
}
