//! End-to-end CABAC I_PCM macroblock decode test.
//!
//! Hand-crafts a single 16×16 CABAC IDR picture whose lone macroblock is
//! coded as `I_PCM` — bypassing the arithmetic engine for pixel samples and
//! re-seeding `codIRange`/`codIOffset` per §9.3.1.2 before the
//! `end_of_slice_flag`. Verifies the raw pixel payload round-trips through
//! `H264Decoder` unchanged.
//!
//! The test shares no code with `cabac_iframe.rs`; it carries its own
//! spec-faithful miniature CABAC encoder that understands the I_PCM mid-
//! stream flush (§9.3.4.6) *without* the `| 1` RBSP-stop-bit trick used at
//! end-of-slice.

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::decoder::H264Decoder;

// ---------------------------------------------------------------------------
// BitWriter — MSB-first.
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
// CabacEncoder — mirrors ITU-T H.264 (07/2019) §9.3.4.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
struct CabacCtx {
    p_state_idx: u8,
    val_mps: u8,
}

impl CabacCtx {
    fn init(m: i32, n: i32, qpy: i32) -> Self {
        let qpy = qpy.clamp(0, 51);
        let pre = (((m * qpy) >> 4) + n).clamp(1, 126);
        if pre <= 63 {
            Self {
                p_state_idx: (63 - pre) as u8,
                val_mps: 0,
            }
        } else {
            Self {
                p_state_idx: (pre - 64) as u8,
                val_mps: 1,
            }
        }
    }
}

#[rustfmt::skip]
const RANGE_TAB_LPS: [[u16; 4]; 64] = [
    [128, 176, 208, 240], [128, 167, 197, 227], [128, 158, 187, 216], [123, 150, 178, 205],
    [116, 142, 169, 195], [111, 135, 160, 185], [105, 128, 152, 175], [100, 123, 144, 166],
    [ 95, 116, 137, 158], [ 90, 110, 130, 150], [ 85, 104, 123, 142], [ 81,  99, 117, 135],
    [ 77,  94, 111, 128], [ 73,  89, 105, 122], [ 69,  85, 100, 116], [ 66,  80,  95, 110],
    [ 62,  76,  90, 104], [ 59,  72,  86,  99], [ 56,  69,  81,  94], [ 53,  65,  77,  89],
    [ 51,  62,  73,  85], [ 48,  59,  69,  80], [ 46,  56,  66,  76], [ 43,  53,  63,  72],
    [ 41,  50,  59,  69], [ 39,  48,  56,  65], [ 37,  45,  54,  62], [ 35,  43,  51,  59],
    [ 33,  41,  48,  56], [ 32,  39,  46,  53], [ 30,  37,  43,  50], [ 29,  35,  41,  48],
    [ 27,  33,  39,  45], [ 26,  31,  37,  43], [ 24,  30,  35,  41], [ 23,  28,  33,  39],
    [ 22,  27,  32,  37], [ 21,  26,  30,  35], [ 20,  24,  29,  33], [ 19,  23,  27,  31],
    [ 18,  22,  26,  30], [ 17,  21,  25,  28], [ 16,  20,  23,  27], [ 15,  19,  22,  25],
    [ 14,  18,  21,  24], [ 14,  17,  20,  23], [ 13,  16,  19,  22], [ 12,  15,  18,  21],
    [ 12,  14,  17,  20], [ 11,  14,  16,  19], [ 11,  13,  15,  18], [ 10,  12,  15,  17],
    [ 10,  12,  14,  16], [  9,  11,  13,  15], [  9,  11,  12,  14], [  8,  10,  12,  14],
    [  8,   9,  11,  13], [  7,   9,  11,  12], [  7,   9,  10,  12], [  7,   8,  10,  11],
    [  6,   8,   9,  11], [  6,   7,   9,  10], [  6,   7,   8,   9], [  2,   2,   2,   2],
];

#[rustfmt::skip]
const TRANS_IDX_LPS: [u8; 64] = [
     0,  0,  1,  2,  2,  4,  4,  5,  6,  7,  8,  9,  9, 11, 11, 12,
    13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24,
    24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33,
    33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

#[rustfmt::skip]
const TRANS_IDX_MPS: [u8; 64] = [
     1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
];

struct CabacEncoder {
    low: u32,
    range: u32,
    outstanding: u32,
    first: bool,
    out_bits: Vec<u8>,
}

impl CabacEncoder {
    fn new() -> Self {
        Self {
            low: 0,
            range: 0x01FE,
            outstanding: 0,
            first: true,
            out_bits: Vec::new(),
        }
    }

    fn put_bit(&mut self, bit: u8) {
        if self.first {
            self.first = false;
        } else {
            self.out_bits.push(bit);
        }
        for _ in 0..self.outstanding {
            self.out_bits.push(1 - bit);
        }
        self.outstanding = 0;
    }

    fn renorm(&mut self) {
        while self.range < 0x0100 {
            if self.low < 0x0100 {
                self.put_bit(0);
            } else if self.low >= 0x0200 {
                self.low -= 0x0200;
                self.put_bit(1);
            } else {
                self.low -= 0x0100;
                self.outstanding += 1;
            }
            self.low <<= 1;
            self.range <<= 1;
        }
    }

    fn encode_bin(&mut self, ctx: &mut CabacCtx, bin: u8) {
        let rlps_idx = ((self.range >> 6) & 3) as usize;
        let p = ctx.p_state_idx as usize;
        let rlps = RANGE_TAB_LPS[p][rlps_idx] as u32;
        self.range -= rlps;
        if bin != ctx.val_mps {
            self.low += self.range;
            self.range = rlps;
            if ctx.p_state_idx == 0 {
                ctx.val_mps = 1 - ctx.val_mps;
            }
            ctx.p_state_idx = TRANS_IDX_LPS[p];
        } else {
            ctx.p_state_idx = TRANS_IDX_MPS[p];
        }
        self.renorm();
    }

    /// §9.3.4.5 — encode_terminate. Pass `bin = 1` for I_PCM or end-of-slice.
    fn encode_terminate(&mut self, bin: u8) {
        self.range -= 2;
        if bin == 1 {
            self.low += self.range;
            self.range = 2;
            // No RenormE here — the caller drives it via one of the flush
            // variants below.
        } else {
            self.renorm();
        }
    }

    /// §9.3.4.6 — EncodingFlush for I_PCM (mid-stream). Drains `codILow` /
    /// `codIRange` into the bitstream without embedding an RBSP stop bit,
    /// then pads to byte alignment with `pcm_alignment_zero_bit` (spec:
    /// zero-valued pad bits).
    fn flush_for_pcm(&mut self) {
        // Drain: range = 2, then renorm pushes out one "high" bit + 2 trailing
        // bits (mirrors end_of_slice flush but without the stop-bit trick).
        self.range = 2;
        self.renorm();
        let hi = ((self.low >> 9) & 1) as u8;
        self.put_bit(hi);
        let tail = ((self.low >> 7) & 3) as u8;
        self.out_bits.push((tail >> 1) & 1);
        self.out_bits.push(tail & 1);
        // pcm_alignment_zero_bit — pad to byte alignment with 0-bits.
        while self.out_bits.len() % 8 != 0 {
            self.out_bits.push(0);
        }
    }

    /// Reset the arithmetic engine state per §9.3.1.2 after I_PCM — context
    /// state is left untouched.
    fn reinit_after_pcm(&mut self) {
        self.low = 0;
        self.range = 0x01FE;
        self.outstanding = 0;
        self.first = true;
    }

    /// Write a raw 8-bit PCM sample. Must be called only between
    /// [`flush_for_pcm`](Self::flush_for_pcm) and
    /// [`reinit_after_pcm`](Self::reinit_after_pcm).
    fn write_pcm_byte(&mut self, v: u8) {
        for i in (0..8).rev() {
            self.out_bits.push((v >> i) & 1);
        }
    }

    /// §9.3.4.6 — end-of-slice flush. Also embeds the RBSP stop bit via the
    /// `| 1` trick (WriteBits( ((low >> 7) & 3) | 1, 2 )).
    fn finish_flush(&mut self) {
        self.range = 2;
        self.renorm();
        let hi = ((self.low >> 9) & 1) as u8;
        self.put_bit(hi);
        let tail = (((self.low >> 7) & 3) | 1) as u8;
        self.out_bits.push((tail >> 1) & 1);
        self.out_bits.push(tail & 1);
    }

    fn finish_bytes_aligned(mut self) -> Vec<u8> {
        while self.out_bits.len() % 8 != 0 {
            self.out_bits.push(0);
        }
        for _ in 0..64 {
            self.out_bits.push(0);
        }
        let mut out = Vec::with_capacity(self.out_bits.len() / 8);
        for chunk in self.out_bits.chunks(8) {
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
// SPS / PPS / slice construction.
// ---------------------------------------------------------------------------

fn build_sps_rbsp() -> Vec<u8> {
    // Main profile (77) — CABAC-capable, 4:2:0 default.
    let profile_idc: u8 = 77;
    let constraint_flags: u8 = 0;
    let level_idc: u8 = 30;

    let mut bw = BitWriter::new();
    bw.write_ue(0); // seq_parameter_set_id
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

fn build_pps_rbsp() -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.write_ue(0); // pic_parameter_set_id
    bw.write_ue(0); // seq_parameter_set_id
    bw.write_bit(1); // entropy_coding_mode_flag = 1 → CABAC
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

/// Build the IDR slice RBSP. The single macroblock is `I_PCM` with a caller-
/// supplied 256-byte luma + 64-byte Cb + 64-byte Cr payload.
fn build_idr_slice_rbsp_ipcm(
    luma: &[u8; 256],
    cb: &[u8; 64],
    cr: &[u8; 64],
) -> Vec<u8> {
    // ----- Slice header -----
    let mut bw = BitWriter::new();
    bw.write_ue(0); // first_mb_in_slice
    bw.write_ue(7); // slice_type = I (7)
    bw.write_ue(0); // pic_parameter_set_id
    bw.write_u(0, 4); // frame_num (4 bits per SPS)
    bw.write_ue(0); // idr_pic_id
    bw.write_u(0, 4); // pic_order_cnt_lsb
    bw.write_bit(0); // no_output_of_prior_pics_flag
    bw.write_bit(0); // long_term_reference_flag
    bw.write_se(0); // slice_qp_delta
    bw.write_ue(1); // disable_deblocking_filter_idc = 1 → no alpha/beta
    // cabac_alignment_one_bit pad up to byte boundary.
    while bw.bits.len() % 8 != 0 {
        bw.write_bit(1);
    }
    let header_bytes = bw.into_bytes();

    // ----- CABAC macroblock: I_PCM (mb_type = 25) -----
    let slice_qpy = 26;
    let mut enc = CabacEncoder::new();

    // mb_type bin 0 — ctxIdx 3 (ctxIdxOffset for I-slice mb_type); value 1
    //   means "not I_NxN". (m, n) for ctx 3 in column 0 = (20, -15) per
    //   Table 9-12 row 3.
    let mut ctx_mb_type_b0 = CabacCtx::init(20, -15, slice_qpy);
    enc.encode_bin(&mut ctx_mb_type_b0, 1);
    // Terminate bin = 1 → I_PCM.
    enc.encode_terminate(1);

    // §9.3.1.2 I_PCM path — flush, byte-align, raw bytes, re-init.
    enc.flush_for_pcm();
    for &v in luma.iter() {
        enc.write_pcm_byte(v);
    }
    for &v in cb.iter() {
        enc.write_pcm_byte(v);
    }
    for &v in cr.iter() {
        enc.write_pcm_byte(v);
    }
    enc.reinit_after_pcm();

    // end_of_slice_flag — terminate(1) + final flush with RBSP stop bit.
    enc.encode_terminate(1);
    enc.finish_flush();

    let cabac_bytes = enc.finish_bytes_aligned();
    let mut out = header_bytes;
    out.extend_from_slice(&cabac_bytes);
    out
}

// ---------------------------------------------------------------------------
// NALU / Annex B packaging.
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

fn build_annex_b_packet(luma: &[u8; 256], cb: &[u8; 64], cr: &[u8; 64]) -> Vec<u8> {
    let sps_rbsp = build_sps_rbsp();
    let pps_rbsp = build_pps_rbsp();
    let idr_rbsp = build_idr_slice_rbsp_ipcm(luma, cb, cr);

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

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[test]
fn cabac_ipcm_single_mb_round_trip() {
    // Unique per-pixel values so any stride/offset bug shows up immediately.
    let mut luma = [0u8; 256];
    for (i, v) in luma.iter_mut().enumerate() {
        *v = (i & 0xFF) as u8;
    }
    let mut cb = [0u8; 64];
    for (i, v) in cb.iter_mut().enumerate() {
        *v = (200 - i as u8).wrapping_add(0);
    }
    let mut cr = [0u8; 64];
    for (i, v) in cr.iter_mut().enumerate() {
        *v = (30 + i as u8).wrapping_add(0);
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
    assert_eq!(frame.planes.len(), 3);

    // Luma plane.
    for (i, &v) in frame.planes[0].data.iter().enumerate() {
        assert_eq!(v, luma[i], "Y[{i}] expected {}, got {v}", luma[i]);
    }
    // Cb / Cr planes.
    for (i, &v) in frame.planes[1].data.iter().enumerate() {
        assert_eq!(v, cb[i], "Cb[{i}] expected {}, got {v}", cb[i]);
    }
    for (i, &v) in frame.planes[2].data.iter().enumerate() {
        assert_eq!(v, cr[i], "Cr[{i}] expected {}, got {v}", cr[i]);
    }
}
