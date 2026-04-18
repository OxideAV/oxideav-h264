//! Synthetic CAVLC I-slice decode at 12-bit and 14-bit.
//!
//! x264 on this host is compiled as a 10-bit-only build, so the standard
//! ffmpeg fixture-generation recipe can't produce a 12/14-bit bitstream.
//! Instead this test hand-crafts a minimal High Profile 4:2:0 IDR
//! containing a single I_NxN macroblock with:
//!
//! * all 16 `prev_intra4x4_pred_mode_flag = 1` (every 4×4 block inherits
//!   the neighbour-predicted `Intra_DC` mode — §8.3.1.1 clause that sets
//!   the predicted mode to `Intra_DC` when either neighbour is missing),
//! * `intra_chroma_pred_mode = 0` (DC, §8.3.4.1),
//! * `coded_block_pattern = 0` (§7.4.5.3 — ue(v) code `3` through
//!   Table 9-4(b)), so no luma / chroma residuals and no `mb_qp_delta`
//!   is emitted.
//!
//! With no residuals and no neighbours, §8.3.1.2.3 DC fallback fills
//! every luma sample with `1 << (BitDepth_Y - 1)` and §8.3.4.2 chroma DC
//! fallback fills every chroma sample with `1 << (BitDepth_C - 1)`. The
//! resulting frame is a flat mid-grey — the exact sample value proves
//! the decode pipeline threaded the SPS bit depth through intra
//! prediction, the u16 Picture planes, and the emit path.
//!
//! This exercises:
//! * The slice-entry gate accepts `bit_depth_luma_minus8 ∈ {4, 6}`.
//! * The `Picture` allocator at 12/14-bit.
//! * CAVLC I-slice routing for 12/14-bit → [`oxideav_h264::mb_hi`].
//! * u16 LE frame packing for 12-bit (`Yuv420P12Le`) and 14-bit (reuses
//!   `Yuv420P10Le` as a 16-bit-container fallback — see `picture.rs`).

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, PixelFormat, TimeBase};
use oxideav_h264::decoder::H264Decoder;

// ---------------------------------------------------------------------------
// MSB-first bit writer.
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
    fn write_u(&mut self, v: u64, n: u32) {
        for i in (0..n).rev() {
            self.bits.push(((v >> i) & 1) as u8);
        }
    }
    fn write_ue(&mut self, v: u32) {
        // Exp-Golomb: x = v+1 in binary, prefixed with bits-1 zeros.
        let x = (v as u64) + 1;
        let bits = 64 - x.leading_zeros();
        for _ in 0..(bits - 1) {
            self.bits.push(0);
        }
        for i in (0..bits).rev() {
            self.bits.push(((x >> i) & 1) as u8);
        }
    }
    fn write_se(&mut self, v: i32) {
        let mapped = if v <= 0 {
            (-v) as u32 * 2
        } else {
            v as u32 * 2 - 1
        };
        self.write_ue(mapped);
    }
    /// RBSP trailing bits: `1` then zero-pad to byte boundary.
    fn write_rbsp_trailing(&mut self) {
        self.write_bit(1);
        while self.bits.len() % 8 != 0 {
            self.write_bit(0);
        }
    }
    fn into_bytes(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.bits.len().div_ceil(8));
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
// NALU wrapping — inserts emulation-prevention bytes and prepends header.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// SPS / PPS / slice construction for High Profile 4:2:0 at bit_depth_y.
// ---------------------------------------------------------------------------

/// Build a High Profile (100) SPS with the requested `bit_depth_luma_minus8`
/// / `bit_depth_chroma_minus8` and a 16×16 frame.
fn build_sps_rbsp(bit_depth_minus8: u32) -> Vec<u8> {
    // profile_idc = 100 (High) so the bit-depth / chroma-format syntax is
    // emitted per §7.3.2.1.1 (High Profile sub-block).
    let profile_idc: u8 = 100;
    let constraint_flags: u8 = 0;
    let level_idc: u8 = 30; // level 3.0

    let mut bw = BitWriter::new();
    bw.write_ue(0); // seq_parameter_set_id
    bw.write_ue(1); // chroma_format_idc = 1 (4:2:0)
    bw.write_ue(bit_depth_minus8); // bit_depth_luma_minus8
    bw.write_ue(bit_depth_minus8); // bit_depth_chroma_minus8
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
    bw.write_rbsp_trailing();
    let body = bw.into_bytes();

    let mut out = Vec::with_capacity(3 + body.len());
    out.push(profile_idc);
    out.push(constraint_flags);
    out.push(level_idc);
    out.extend_from_slice(&body);
    out
}

fn build_pps_rbsp() -> Vec<u8> {
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
    bw.write_se(0); // pic_init_qp_minus26 → QpY = 26
    bw.write_se(0); // pic_init_qs_minus26
    bw.write_se(0); // chroma_qp_index_offset
    bw.write_bit(1); // deblocking_filter_control_present_flag
    bw.write_bit(0); // constrained_intra_pred_flag
    bw.write_bit(0); // redundant_pic_cnt_present_flag
    bw.write_rbsp_trailing();
    bw.into_bytes()
}

/// Build an IDR slice RBSP containing a single I_NxN macroblock with
/// DC prediction inherited on every sub-block and no residuals.
fn build_idr_slice_rbsp() -> Vec<u8> {
    let mut bw = BitWriter::new();
    // ---- Slice header (§7.3.3) ----
    bw.write_ue(0); // first_mb_in_slice
    bw.write_ue(7); // slice_type = 7 (I, single-type-per-picture)
    bw.write_ue(0); // pic_parameter_set_id
    bw.write_u(0, 4); // frame_num (4 bits)
    bw.write_ue(0); // idr_pic_id
    bw.write_u(0, 4); // pic_order_cnt_lsb (4 bits)
    bw.write_bit(0); // no_output_of_prior_pics_flag
    bw.write_bit(0); // long_term_reference_flag
    bw.write_se(0); // slice_qp_delta → QpY = 26
    bw.write_ue(1); // disable_deblocking_filter_idc = 1 (no deblock)

    // ---- MB layer: single I_NxN macroblock ----
    // mb_type = 0 → I_NxN.
    bw.write_ue(0);
    // 16 × prev_intra4x4_pred_mode_flag = 1. When set, the coded mode is
    // the predicted one; §8.3.1.1 returns Intra_DC (value 2) as the
    // fallback when any neighbour is missing — which, for a lone MB at
    // (0,0), applies to every block.
    for _ in 0..16 {
        bw.write_bit(1);
    }
    // intra_chroma_pred_mode = 0 (DC).
    bw.write_ue(0);
    // coded_block_pattern: me(v) encoded as ue(v) per Table 9-4(b).
    // ue(v) = 3 maps via ME_INTRA_4_2_0[3] = 0 → (cbp_luma = 0, cbp_chroma = 0).
    bw.write_ue(3);
    // With cbp == 0 the decoder skips `mb_qp_delta` and the full residual
    // layer (§7.3.5 conditional on `CodedBlockPatternLuma | CodedBlockPatternChroma`).

    bw.write_rbsp_trailing();
    bw.into_bytes()
}

/// Build an IDR slice RBSP containing a single I_16x16 macroblock with
/// DC luma prediction and DC chroma prediction. `cbp_luma = 0`,
/// `cbp_chroma = 0`, so no AC residuals are emitted — but the
/// Intra16x16 Hadamard DC block is still coded (with TotalCoeff = 0)
/// per §7.3.5.3. This exercises the 12/14-bit Hadamard dequant math
/// (i64-widened `inv_hadamard_4x4_dc_scaled_ext`) even though the
/// decoded DC is zero.
fn build_idr_slice_rbsp_i16x16_dc() -> Vec<u8> {
    let mut bw = BitWriter::new();
    // ---- Slice header ----
    bw.write_ue(0); // first_mb_in_slice
    bw.write_ue(7); // slice_type = I
    bw.write_ue(0); // pic_parameter_set_id
    bw.write_u(0, 4); // frame_num
    bw.write_ue(0); // idr_pic_id
    bw.write_u(0, 4); // pic_order_cnt_lsb
    bw.write_bit(0); // no_output_of_prior_pics_flag
    bw.write_bit(0); // long_term_reference_flag
    bw.write_se(0); // slice_qp_delta
    bw.write_ue(1); // disable_deblocking_filter_idc = 1

    // ---- MB layer ----
    // Table 7-11: mb_type = 3 → (pred = 2 / DC, cbp_chroma_class = 0,
    // cbp_luma_flag = 0). So `cbp_luma = 0`, `cbp_chroma = 0`, and the
    // luma Intra16x16 prediction mode is DC.
    bw.write_ue(3);
    // intra_chroma_pred_mode = 0 (DC).
    bw.write_ue(0);
    // I_16x16 always emits mb_qp_delta even when cbp == 0.
    bw.write_se(0);
    // Luma 16x16 DC block — TC=0, T1=0 under nC class 0 (`nc < 2`) is a
    // single `1` bit (`COEFF_TOKEN_BITS[0][0] = 1`, len = 1).
    bw.write_bit(1);
    bw.write_rbsp_trailing();
    bw.into_bytes()
}

fn build_annex_b_packet(bit_depth_minus8: u32) -> Vec<u8> {
    build_packet(bit_depth_minus8, &build_idr_slice_rbsp())
}

fn build_annex_b_packet_i16x16(bit_depth_minus8: u32) -> Vec<u8> {
    build_packet(bit_depth_minus8, &build_idr_slice_rbsp_i16x16_dc())
}

fn build_packet(bit_depth_minus8: u32, idr_rbsp: &[u8]) -> Vec<u8> {
    let sps_rbsp = build_sps_rbsp(bit_depth_minus8);
    let pps_rbsp = build_pps_rbsp();

    let sps = wrap_nalu(7, 3, &sps_rbsp);
    let pps = wrap_nalu(8, 3, &pps_rbsp);
    let idr = wrap_nalu(5, 3, idr_rbsp);

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

fn decode_synthetic(bit_depth_minus8: u32) -> oxideav_core::VideoFrame {
    let pkt_data = build_annex_b_packet(bit_depth_minus8);
    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 90_000), pkt_data)
        .with_pts(0)
        .with_keyframe(true);
    dec.send_packet(&pkt).expect("send_packet");
    match dec.receive_frame().expect("receive_frame") {
        Frame::Video(f) => f,
        other => panic!(
            "expected video frame, got {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[test]
fn decode_12bit_iframe_dc_fallback() {
    let frame = decode_synthetic(4);

    assert_eq!(frame.width, 16);
    assert_eq!(frame.height, 16);
    // 12-bit 4:2:0 emits the core Yuv420P12Le variant.
    assert_eq!(frame.format, PixelFormat::Yuv420P12Le);
    assert_eq!(frame.planes.len(), 3);

    // §8.3.1.2.3 / §8.3.4.2 no-neighbour DC fallback = 1 << (BitDepth - 1).
    let mid: u16 = 1 << (12 - 1); // 2048

    let y = unpack_u16_le(&frame.planes[0].data);
    let cb = unpack_u16_le(&frame.planes[1].data);
    let cr = unpack_u16_le(&frame.planes[2].data);
    assert_eq!(y.len(), 16 * 16);
    assert_eq!(cb.len(), 8 * 8);
    assert_eq!(cr.len(), 8 * 8);
    for (i, &v) in y.iter().enumerate() {
        assert_eq!(v, mid, "12-bit Y[{i}] expected {mid}, got {v}");
    }
    for (i, &v) in cb.iter().enumerate() {
        assert_eq!(v, mid, "12-bit Cb[{i}] expected {mid}, got {v}");
    }
    for (i, &v) in cr.iter().enumerate() {
        assert_eq!(v, mid, "12-bit Cr[{i}] expected {mid}, got {v}");
    }
}

#[test]
fn decode_14bit_iframe_dc_fallback() {
    let frame = decode_synthetic(6);

    assert_eq!(frame.width, 16);
    assert_eq!(frame.height, 16);
    // oxideav-core 0.0.3 has no `Yuv420P14Le` variant; the decoder
    // reuses `Yuv420P10Le` as a 16-bit-container fallback (u16 words
    // still hold 14-bit samples 0..=16383).
    assert_eq!(frame.format, PixelFormat::Yuv420P10Le);

    // §8.3.1.2.3 / §8.3.4.2 no-neighbour DC fallback = 1 << (BitDepth - 1).
    let mid: u16 = 1 << (14 - 1); // 8192

    let y = unpack_u16_le(&frame.planes[0].data);
    let cb = unpack_u16_le(&frame.planes[1].data);
    let cr = unpack_u16_le(&frame.planes[2].data);
    for (i, &v) in y.iter().enumerate() {
        assert_eq!(v, mid, "14-bit Y[{i}] expected {mid}, got {v}");
    }
    for &v in cb.iter().chain(cr.iter()) {
        assert_eq!(v, mid);
    }
}

/// Loosened gate still rejects odd depths (e.g. `bit_depth_luma_minus8 = 3`).
#[test]
fn reject_unsupported_odd_bit_depth() {
    let pkt_data = build_annex_b_packet(3);
    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 90_000), pkt_data)
        .with_pts(0)
        .with_keyframe(true);
    // Per spec §7.4.2.1.1 bit_depth_luma_minus8 can be 0..=6, but every
    // High-family profile constrains it to even values. Our gate follows
    // that — odd values are rejected.
    let err = dec.send_packet(&pkt).err();
    assert!(
        err.is_some(),
        "odd bit_depth_luma_minus8 must be rejected at slice entry"
    );
}

fn decode_i16x16_dc(bit_depth_minus8: u32) -> oxideav_core::VideoFrame {
    let pkt_data = build_annex_b_packet_i16x16(bit_depth_minus8);
    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 90_000), pkt_data)
        .with_pts(0)
        .with_keyframe(true);
    dec.send_packet(&pkt).expect("send_packet (i16x16)");
    match dec.receive_frame().expect("receive_frame (i16x16)") {
        Frame::Video(f) => f,
        other => panic!(
            "expected video frame, got {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

/// I_16x16 with mb_type = 3 (DC + cbp = 0) exercises the high-bit-depth
/// Hadamard DC chain even though the decoded DC is zero. With no
/// neighbours, §8.3.1.2.3 Intra_16x16 DC fallback yields the luma
/// mid-grey, and the zero Hadamard output lands every sample on that
/// value after the `+ residual` add + clip.
#[test]
fn decode_12bit_i16x16_dc_hadamard_path() {
    let frame = decode_i16x16_dc(4);
    assert_eq!(frame.width, 16);
    assert_eq!(frame.height, 16);
    assert_eq!(frame.format, PixelFormat::Yuv420P12Le);
    let mid: u16 = 1 << 11;
    let y = unpack_u16_le(&frame.planes[0].data);
    for &v in y.iter() {
        assert_eq!(v, mid);
    }
}

#[test]
fn decode_14bit_i16x16_dc_hadamard_path() {
    let frame = decode_i16x16_dc(6);
    assert_eq!(frame.width, 16);
    assert_eq!(frame.height, 16);
    assert_eq!(frame.format, PixelFormat::Yuv420P10Le);
    let mid: u16 = 1 << 13;
    let y = unpack_u16_le(&frame.planes[0].data);
    for &v in y.iter() {
        assert_eq!(v, mid);
    }
}
