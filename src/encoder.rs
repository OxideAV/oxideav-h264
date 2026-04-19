//! H.264 **Baseline Profile** encoder.
//!
//! Minimum viable encoder that can be driven through the
//! [`oxideav_codec::Encoder`] trait and whose output is decodable by this
//! crate's [`crate::decoder::H264Decoder`] (and by ffmpeg, subject to fixture
//! availability).
//!
//! # Advertised scope
//!
//! * Baseline Profile (profile_idc = 66), level 3.0.
//! * Progressive path: IDR I-frame + optional CAVLC `P_L0_16×16` /
//!   `P_Skip` P-frames referencing the previous reconstructed frame
//!   (single-slot L0). Driven by
//!   [`H264EncoderOptions::p_slice_interval`] — `0` keeps every output
//!   an IDR (legacy behaviour); `N > 0` emits one IDR followed by
//!   P-frames until the next interval boundary.
//! * PAFF path (`paff_field = Some(..)`): the first encoded field is an
//!   IDR I-slice at the selected parity; subsequent encoded fields at the
//!   same parity are emitted as zero-residual P-slices (one `mb_skip_run`
//!   covering every MB, no coded macroblock) referencing the prior I
//!   field. This exercises the decoder's PAFF P path without needing a
//!   full inter-prediction writer on the encoder side.
//! * CAVLC entropy coding (entropy_coding_mode_flag = 0).
//! * Single slice per picture (`num_slice_groups_minus1 = 0`).
//! * 4:2:0 chroma, 8-bit luma / chroma, single colour plane.
//! * I-MB path: Intra_16×16 DC_PRED for every luma macroblock;
//!   chroma DC for chroma.
//! * P-MB path: `P_L0_16×16` with a single 16×16 partition, ref_idx = 0,
//!   integer-pel ME via SAD over a ±16-pixel window; MVD = mv − §8.4.1.3
//!   median predictor; 4×4 residual forward DCT + quant + CAVLC; inter
//!   CBP via me(v); `P_Skip` when MVD = 0 and residual is all-zero.
//! * Fixed QP (configurable, default = 26). No rate control, no adaptive QP.
//! * Deblocking disabled on emit (`disable_deblocking_filter_idc = 1`).
//! * Annex B framing with 4-byte start codes.
//!
//! # Packet layout
//!
//! Every `receive_packet()` returns a single self-contained packet shaped
//! as one of:
//!
//! ```text
//! IDR:
//!   [start code] [SPS NAL]
//!   [start code] [PPS NAL]
//!   [start code] [IDR slice NAL]
//!
//! P-frame:
//!   [start code] [SPS NAL]
//!   [start code] [PPS NAL]
//!   [start code] [non-IDR slice NAL]
//! ```
//!
//! IDR packets carry `PacketFlags::keyframe = true`; P packets do not.
//!
//! # Quantisation
//!
//! The forward pipeline matches [`crate::fwd_transform`]: DCT → quant
//! (per-position MF table) → the decoder's `dequantize_4x4` and
//! `idct_4x4` on the encoder side for *local reconstruction*. That means
//! the encoder's `last-reconstructed` buffer mirrors what the decoder will
//! produce exactly (bit-exact for a given qp), so subsequent macroblock
//! intra-prediction uses the same neighbour samples the decoder would.

use std::collections::VecDeque;

use oxideav_codec::Encoder;
use oxideav_core::{
    packet::PacketFlags, CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat,
    Result, VideoFrame,
};

use crate::cavlc::BlockKind;
use crate::cavlc_enc::encode_residual_block;
use crate::fwd_transform::{
    forward_dct_4x4, forward_hadamard_2x2, forward_hadamard_4x4, quantize_4x4_ac,
    quantize_chroma_dc_2x2, quantize_luma_dc_4x4,
};
use crate::golomb::BitWriterExt;
use crate::intra_pred::{
    predict_intra_16x16, predict_intra_chroma, Intra16x16Mode, Intra16x16Neighbours,
    IntraChromaMode, IntraChromaNeighbours,
};
use crate::mb::LUMA_BLOCK_RASTER;
use crate::mb_type::{PMbType, PPartition, ME_INTER_4_2_0};
use crate::motion::{chroma_mc, luma_mc_plane, predict_mv_l0, predict_mv_pskip};
use crate::nal::rbsp_to_ebsp;
use crate::picture::{MbInfo, Picture};
use crate::tables::ME_INTRA_4_2_0;
use crate::transform::{
    chroma_qp, dequantize_4x4, idct_4x4, inv_hadamard_2x2_chroma_dc, inv_hadamard_4x4_dc,
};
use oxideav_core::bits::BitWriter;

/// Options for the H.264 encoder.
#[derive(Clone, Debug)]
pub struct H264EncoderOptions {
    /// Quantisation parameter (0-51). Higher QP → smaller files, more
    /// blur. Default 26.
    pub qp: i32,
    /// SPS seq_parameter_set_id. Default 0.
    pub sps_id: u32,
    /// PPS pic_parameter_set_id. Default 0.
    pub pps_id: u32,
    /// `Some(false)` → emit PAFF top-field slices (§7.3.3
    /// `field_pic_flag = 1`, `bottom_field_flag = 0`).
    /// `Some(true)`  → emit PAFF bottom-field slices.
    /// `None`        → emit as a progressive frame (default).
    ///
    /// The input [`VideoFrame`] is the field samples themselves — the
    /// encoder does *not* deinterleave a full-height frame. When set,
    /// the SPS carries `frame_mbs_only_flag = 0` and
    /// `mb_adaptive_frame_field_flag = 0` so the slice's
    /// `field_pic_flag` bit is emitted.
    ///
    /// The first PAFF frame fed to the encoder is an IDR I-slice at the
    /// selected parity. Subsequent PAFF frames become "all-skip" P
    /// slices — a single `mb_skip_run = TotalMbs` ue(v) with no coded
    /// macroblock, which decodes as a zero-MV copy of the prior I field
    /// (§8.4.1.1 `P_Skip` with unavailable neighbours returns the zero
    /// predictor). This is enough to drive the decoder's PAFF P path
    /// end-to-end; it is not a reconstruction-quality inter encoder.
    pub paff_field: Option<bool>,
    /// Progressive P-slice cadence. When `0` (default), every output
    /// frame is an IDR (legacy I-only behaviour). When `N > 0`, the
    /// encoder emits an IDR every `N` frames and a CAVLC
    /// `P_L0_16×16` / `P_Skip` P-slice for the frames in between,
    /// referencing the previous reconstructed frame (single-slot L0).
    /// Ignored on the PAFF path — PAFF still emits one IDR + all-skip P
    /// fields as documented on [`Self::paff_field`].
    pub p_slice_interval: u32,
}

impl Default for H264EncoderOptions {
    fn default() -> Self {
        Self {
            qp: 26,
            sps_id: 0,
            pps_id: 0,
            paff_field: None,
            p_slice_interval: 0,
        }
    }
}

/// Baseline H.264 encoder supporting IDR + CAVLC P-slices.
pub struct H264Encoder {
    codec_id: CodecId,
    opts: H264EncoderOptions,
    /// Output width, height. Rounded up to the nearest multiple of 16
    /// internally; `SPS.frame_cropping` reports the visible size.
    width: u32,
    height: u32,
    coded_width: u32,
    coded_height: u32,
    mb_width: u32,
    mb_height: u32,
    /// Cached SPS / PPS NAL bodies (including 1-byte NAL header).
    sps_nal: Vec<u8>,
    pps_nal: Vec<u8>,
    output_params: CodecParameters,
    packets: VecDeque<Packet>,
    /// Slice-layer `frame_num` (§7.4.3) — advanced for every non-IDR
    /// reference frame; reset to 0 at every IDR.
    frame_num: u32,
    /// Count of frames emitted since the last IDR; when it hits
    /// `p_slice_interval` the encoder rotates back to an IDR.
    frames_since_idr: u32,
    /// Reconstructed frame that sits in L0 for the *next* P-slice encode.
    /// `None` before the first frame and whenever the next frame must be
    /// an IDR.
    ref_pic: Option<Picture>,
    eof: bool,
}

impl H264Encoder {
    /// Build an encoder targeting `width` × `height`. Dimensions must be
    /// within Baseline Level 3.0 (up to 720×576 conventionally; we don't
    /// enforce the exact macroblocks-per-second cap since we have no
    /// frame-rate knowledge up front).
    pub fn new(
        codec_id: CodecId,
        width: u32,
        height: u32,
        opts: H264EncoderOptions,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::invalid("h264 encoder: width/height must be > 0"));
        }
        if !(0..=51).contains(&opts.qp) {
            return Err(Error::invalid(format!(
                "h264 encoder: qp {} out of range 0..=51",
                opts.qp
            )));
        }
        let mb_width = width.div_ceil(16);
        let mb_height = height.div_ceil(16);
        let coded_width = mb_width * 16;
        let coded_height = mb_height * 16;
        let sps_nal = build_sps_nal(width, height, opts.sps_id, opts.paff_field.is_some())?;
        let pps_nal = build_pps_nal(opts.pps_id, opts.sps_id)?;

        let mut output_params = CodecParameters::video(codec_id.clone());
        output_params.media_type = MediaType::Video;
        output_params.width = Some(width);
        output_params.height = Some(height);
        output_params.pixel_format = Some(PixelFormat::Yuv420P);
        // AVCDecoderConfigurationRecord as the extradata (for MP4 wrapping).
        output_params.extradata = build_avcc(&sps_nal, &pps_nal);

        Ok(Self {
            codec_id,
            opts,
            width,
            height,
            coded_width,
            coded_height,
            mb_width,
            mb_height,
            sps_nal,
            pps_nal,
            output_params,
            packets: VecDeque::new(),
            frame_num: 0,
            frames_since_idr: 0,
            ref_pic: None,
            eof: false,
        })
    }

    /// Encode one YUV 4:2:0 frame into an Annex B packet and stash it in
    /// the output queue.
    fn encode_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        if frame.format != PixelFormat::Yuv420P {
            return Err(Error::invalid(format!(
                "h264 encoder: unsupported pixel format {:?} (need Yuv420P)",
                frame.format
            )));
        }
        if frame.width != self.width || frame.height != self.height {
            return Err(Error::invalid(format!(
                "h264 encoder: frame size {}×{} does not match encoder size {}×{}",
                frame.width, frame.height, self.width, self.height
            )));
        }
        if frame.planes.len() < 3 {
            return Err(Error::invalid(
                "h264 encoder: expected 3 planes for YUV420P",
            ));
        }
        // PAFF + `frame_num > 0` → emit an all-skip P-field referencing the
        // prior I-field rather than a second IDR. Progressive mode always
        // takes the I-only path.
        if self.opts.paff_field.is_some() && self.frame_num > 0 {
            return self.encode_paff_p_skip_field(frame);
        }
        // Progressive P-slice cadence (ignored under PAFF).
        if self.opts.paff_field.is_none()
            && self.opts.p_slice_interval > 0
            && self.frames_since_idr > 0
            && self.frames_since_idr < self.opts.p_slice_interval
            && self.ref_pic.is_some()
        {
            return self.encode_p_frame(frame);
        }

        // Build the MB-aligned raw-sample buffer (padding replicate-edges
        // if width/height are not already multiples of 16).
        let y_src = plane_to_mb_aligned(
            &frame.planes[0].data,
            frame.planes[0].stride,
            self.width as usize,
            self.height as usize,
            self.coded_width as usize,
            self.coded_height as usize,
        );
        let cw = (self.width / 2) as usize;
        let ch = (self.height / 2) as usize;
        let coded_cw = (self.coded_width / 2) as usize;
        let coded_ch = (self.coded_height / 2) as usize;
        let cb_src = plane_to_mb_aligned(
            &frame.planes[1].data,
            frame.planes[1].stride,
            cw,
            ch,
            coded_cw,
            coded_ch,
        );
        let cr_src = plane_to_mb_aligned(
            &frame.planes[2].data,
            frame.planes[2].stride,
            cw,
            ch,
            coded_cw,
            coded_ch,
        );

        // Reconstruction buffer — encoder's local view of what the decoder
        // will produce. Populated MB-by-MB so later MBs' intra prediction
        // uses bit-exact neighbour samples.
        let mut rec_y = vec![0u8; (self.coded_width * self.coded_height) as usize];
        let mut rec_cb = vec![0u8; coded_cw * coded_ch];
        let mut rec_cr = vec![0u8; coded_cw * coded_ch];
        let l_stride = self.coded_width as usize;
        let c_stride = coded_cw;

        // Build IDR slice RBSP. frame_num is 0 for the IDR.
        let mut slice_rbsp = BitWriter::new();
        self.write_slice_header(&mut slice_rbsp);
        // The I-slice is always emitted with frame_num = 0 in this encoder
        // (it is the IDR that opens the CVS); subsequent PAFF P-fields
        // advance `self.frame_num` to drive the DPB on the decoder side.
        // nC neighbour state (§9.2.1.1) tracked per 4×4 luma block and per
        // 4×4 chroma block. Raster-indexed across the coded picture.
        let mut luma_nc = vec![0u8; (self.mb_width * 4 * self.mb_height * 4) as usize];
        let mut cb_nc = vec![0u8; (self.mb_width * 2 * self.mb_height * 2) as usize];
        let mut cr_nc = vec![0u8; (self.mb_width * 2 * self.mb_height * 2) as usize];

        let mb_w = self.mb_width;
        let mb_h = self.mb_height;
        let mut last_qp = self.opts.qp;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                self.encode_one_mb(
                    mb_x,
                    mb_y,
                    &y_src,
                    &cb_src,
                    &cr_src,
                    &mut rec_y,
                    &mut rec_cb,
                    &mut rec_cr,
                    l_stride,
                    c_stride,
                    &mut slice_rbsp,
                    &mut luma_nc,
                    &mut cb_nc,
                    &mut cr_nc,
                    &mut last_qp,
                )?;
            }
        }
        // Stop bit + zero-align.
        slice_rbsp.write_rbsp_trailing_bits();
        let rbsp = slice_rbsp.finish();

        // NAL header for an IDR slice: forbidden_zero_bit=0, nal_ref_idc=3,
        // nal_unit_type=5 → 0x65.
        let nal_header = 0x65u8;
        let mut slice_nal = Vec::with_capacity(1 + rbsp.len());
        slice_nal.push(nal_header);
        slice_nal.extend_from_slice(&rbsp);
        let slice_ebsp = rbsp_to_ebsp(&slice_nal);

        // Packet payload: Annex B framing. Start codes (4-byte) before each
        // of the three NALs.
        let mut out =
            Vec::with_capacity(slice_ebsp.len() + self.sps_nal.len() + self.pps_nal.len() + 12);
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&self.sps_nal);
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&self.pps_nal);
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&slice_ebsp);

        let mut pkt = Packet::new(0, frame.time_base, out);
        pkt.pts = frame.pts;
        pkt.dts = frame.pts;
        pkt.flags = PacketFlags {
            keyframe: true,
            ..Default::default()
        };
        self.packets.push_back(pkt);
        // IDR advances `frame_num` (the PAFF encoder keys on
        // `frame_num > 0` to dispatch the all-skip P-field path, and the
        // progressive P path reads the monotonic value unchanged). For
        // the progressive encoder the next P-slice writes
        // `frame_num = 1`; the IDR itself was written with `frame_num = 0`
        // inline by `write_slice_header`.
        self.frame_num = self.frame_num.wrapping_add(1);
        self.frames_since_idr = 1;
        // Stash the reconstructed picture so the next P-frame can use it
        // as its single L0 reference. A fresh `Picture` seeded with the
        // sample buffers we just produced; `MbInfo.coded = true` (all
        // intra) so MV predictor lookups walking into this picture return
        // the "available, intra" answer defined by §8.4.1.3.1.
        let pic = self.build_ref_picture_intra(rec_y, rec_cb, rec_cr);
        self.ref_pic = Some(pic);
        Ok(())
    }

    /// PAFF P-field — §7.3.3 P-slice header + §7.3.4 one `mb_skip_run`
    /// ue(v) equal to `TotalMbs`. The field carries no coded macroblock;
    /// every MB is derived by the decoder as `P_Skip` (§8.4.1.1) which,
    /// with both A and B neighbours unavailable on the first MB and all
    /// subsequent MBs inheriting ref_idx = 0 / MV = (0, 0), copies the
    /// prior I-field sample-for-sample. The caller-supplied [`VideoFrame`]
    /// is ignored on this path — its role is exercising the decoder, not
    /// rate-distortion.
    fn encode_paff_p_skip_field(&mut self, _frame: &VideoFrame) -> Result<()> {
        let bottom = self.opts.paff_field.expect("paff_field set by caller");
        let total_mbs = self.mb_width * self.mb_height;

        let mut slice_rbsp = BitWriter::new();
        // §7.3.3 P-slice header. slice_type = 5 (single-type-per-picture
        // variant of P), so `slice_type_raw % 5 == 0 == SliceType::P`.
        slice_rbsp.write_ue(0); // first_mb_in_slice
        slice_rbsp.write_ue(5); // slice_type = P
        slice_rbsp.write_ue(self.opts.pps_id);
        // frame_num — 4 bits (log2_max_frame_num_minus4 = 0). One P-field
        // follows the IDR, so frame_num = 1.
        slice_rbsp.write_bits(1, 4);
        // field_pic_flag + bottom_field_flag (PAFF path always).
        slice_rbsp.write_flag(true);
        slice_rbsp.write_flag(bottom);
        // pic_order_cnt_lsb — 4 bits. Pick 2 so the P-field's POC is
        // strictly greater than the IDR's 0 (avoids a POC tie at the
        // decoder's output reorder queue).
        slice_rbsp.write_bits(2, 4);
        // num_ref_idx_active_override_flag = 0 → inherit PPS default of
        // num_ref_idx_l0_active_minus1 = 0.
        slice_rbsp.write_flag(false);
        // ref_pic_list_modification_flag_l0 = 0.
        slice_rbsp.write_flag(false);
        // dec_ref_pic_marking — non-IDR with nal_ref_idc != 0:
        //   adaptive_ref_pic_marking_mode_flag = 0 (sliding window).
        slice_rbsp.write_flag(false);
        // slice_qp_delta — the slice data reads no mb_qp_delta so this
        // value is inherited as QP for every implied P_Skip MB.
        slice_rbsp.write_se(self.opts.qp - 26);
        // disable_deblocking_filter_idc = 1 (disabled) — matches the IDR
        // path and avoids the two boundary-offset se(v) fields.
        slice_rbsp.write_ue(1);

        // §7.3.4 slice_data() — single `mb_skip_run` covering every MB
        // followed by the stop bit + alignment. No `mb_type` is ever
        // parsed because the outer loop breaks on `mb_addr >= total_mbs`
        // immediately after consuming the skip run.
        slice_rbsp.write_ue(total_mbs);
        slice_rbsp.write_rbsp_trailing_bits();
        let rbsp = slice_rbsp.finish();

        // NAL header: nal_ref_idc = 3 (reference picture), nal_unit_type = 1
        // (SliceNonIdr). 0x61 = 0b0110_0001.
        let nal_header = 0x61u8;
        let mut slice_nal = Vec::with_capacity(1 + rbsp.len());
        slice_nal.push(nal_header);
        slice_nal.extend_from_slice(&rbsp);
        let slice_ebsp = rbsp_to_ebsp(&slice_nal);

        // P-fields after the IDR re-emit the slice only — SPS/PPS are
        // cached by the decoder from the opening IDR packet. Include
        // them anyway so the packet is self-contained (mirrors the IDR
        // path's three-NAL layout and keeps the test harness happy when
        // a downstream parser wants to re-read parameter sets).
        let mut out =
            Vec::with_capacity(slice_ebsp.len() + self.sps_nal.len() + self.pps_nal.len() + 12);
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&self.sps_nal);
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&self.pps_nal);
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&slice_ebsp);

        let mut pkt = Packet::new(0, _frame.time_base, out);
        pkt.pts = _frame.pts;
        pkt.dts = _frame.pts;
        pkt.flags = PacketFlags::default();
        self.packets.push_back(pkt);
        self.frame_num = self.frame_num.wrapping_add(1);
        Ok(())
    }

    fn write_slice_header(&self, w: &mut BitWriter) {
        // §7.3.3 IDR slice header for Baseline/CAVLC/single slice.
        w.write_ue(0); // first_mb_in_slice
        w.write_ue(7); // slice_type = 7 (I, single-type-per-picture variant)
        w.write_ue(self.opts.pps_id); // pic_parameter_set_id
                                      // frame_num — 4 bits total (log2_max_frame_num_minus4 = 0 in our SPS).
        w.write_bits(0, 4);
        // §7.3.3 — field_pic_flag / bottom_field_flag only present when the
        // SPS has frame_mbs_only_flag = 0. Our PAFF path sets the SPS flag
        // off and encodes a single field per packet.
        if let Some(bottom) = self.opts.paff_field {
            w.write_flag(true); // field_pic_flag = 1
            w.write_flag(bottom); // bottom_field_flag
        }
        // idr_pic_id (IDR only).
        w.write_ue(0);
        // pic_order_cnt_lsb — log2_max_pic_order_cnt_lsb_minus4 = 0 → 4 bits.
        w.write_bits(0, 4);
        // No reference picture list modification (I-slice, skipped per §7.3.3.1).
        // No pred_weight_table (weighted_pred_flag = 0).
        // dec_ref_pic_marking (nal_ref_idc != 0, IDR):
        w.write_flag(false); // no_output_of_prior_pics_flag
        w.write_flag(false); // long_term_reference_flag
                             // slice_qp_delta = qp - (pic_init_qp - 26) ⇒ we set pic_init_qp_minus26
                             // = 0 in PPS, so slice_qp_delta = qp - 26.
        w.write_se(self.opts.qp - 26);
        // Deblocking filter control — disable_deblocking_filter_idc = 1 (off).
        w.write_ue(1);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_one_mb(
        &self,
        mb_x: u32,
        mb_y: u32,
        y_src: &[u8],
        cb_src: &[u8],
        cr_src: &[u8],
        rec_y: &mut [u8],
        rec_cb: &mut [u8],
        rec_cr: &mut [u8],
        l_stride: usize,
        c_stride: usize,
        w: &mut BitWriter,
        luma_nc: &mut [u8],
        cb_nc: &mut [u8],
        cr_nc: &mut [u8],
        last_qp: &mut i32,
    ) -> Result<()> {
        // Spec Table 7-11: mb_type 23 = I_16x16_2_2_15 — Intra16x16PredMode=2
        // (DC), CodedBlockPatternLuma=15, CodedBlockPatternChroma=2. This
        // encoder always emits that combination: DC prediction is the most
        // neighbour-robust (works for the first MB with no available
        // neighbours), coded AC and chroma AC guarantee the decoder
        // receives residual for every 4×4 sub-block, and the fixed QP
        // keeps the whole pipeline deterministic.
        let mb_type: u32 = 23;
        w.write_ue(mb_type);

        // Collect intra prediction neighbours from reconstruction buffer.
        let lneigh = collect_intra16x16_neighbours(rec_y, l_stride, mb_x, mb_y);
        let cneigh_cb = collect_chroma_neighbours(rec_cb, c_stride, mb_x, mb_y);
        let cneigh_cr = collect_chroma_neighbours(rec_cr, c_stride, mb_x, mb_y);

        // Intra chroma pred mode — DC (0). Emitted as ue(v).
        w.write_ue(0);

        // I_16x16 has no explicit CBP — it's packed into mb_type.

        // mb_qp_delta. We use a fixed QP; the first MB of the slice needs
        // a delta to reach opts.qp from the slice start (already == opts.qp),
        // so delta is always 0 here.
        let qp_y = self.opts.qp;
        let mb_qp_delta = 0;
        let _ = last_qp; // unused — fixed QP throughout.
        w.write_se(mb_qp_delta);

        // ------------------------------------------------------------------
        // Luma pipeline.
        // ------------------------------------------------------------------
        let mut pred = [0u8; 256];
        predict_intra_16x16(&mut pred, Intra16x16Mode::Dc, &lneigh);

        // Compute residuals for all 16 4×4 blocks in MB.
        // Layout: luma_blocks[row_4x4 * 4 + col_4x4] = [i32; 16] raster.
        let mut ac_blocks: [[i32; 16]; 16] = [[0; 16]; 16];
        let mut dc_block = [0i32; 16];
        let y_off_mb = (mb_y as usize * 16) * l_stride + mb_x as usize * 16;
        for br in 0..4usize {
            for bc in 0..4usize {
                let mut residual = [0i32; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        let src = y_src[y_off_mb + (br * 4 + r) * l_stride + (bc * 4 + c)] as i32;
                        let p = pred[(br * 4 + r) * 16 + (bc * 4 + c)] as i32;
                        residual[r * 4 + c] = src - p;
                    }
                }
                forward_dct_4x4(&mut residual);
                // Capture DC into dc_block, zero it from AC.
                dc_block[br * 4 + bc] = residual[0];
                residual[0] = 0;
                ac_blocks[br * 4 + bc] = residual;
            }
        }
        // Quantise the AC blocks.
        for b in ac_blocks.iter_mut() {
            quantize_4x4_ac(b, qp_y);
        }
        // Quantise the DC via Hadamard + quant_luma_dc.
        forward_hadamard_4x4(&mut dc_block);
        quantize_luma_dc_4x4(&mut dc_block, qp_y);

        // Emit the residual: DC block first (§7.3.5.3), then 16 AC blocks.
        // Compute nC for the DC block (nC = 0 since the DC block has its own
        // neighbour prediction; for luma 16x16 DC the standard uses nC
        // predicted from the top-left 4×4 neighbour just like any other
        // block, but with special rules. The decoder passes the result of
        // `predict_nc_luma(pic, mb_x, mb_y, 0, 0)` in
        // [`crate::mb::decode_luma_intra_16x16`]. We mirror that.
        let nc_dc = predict_nc_luma(luma_nc, mb_x, mb_y, 0, 0, mb_w_of(self.mb_width));
        encode_residual_block(w, &dc_block, nc_dc, BlockKind::Luma16x16Dc)?;

        // Each 4×4 AC block.
        let mut luma_nc_local = [0u8; 16];
        for blk in 0..16usize {
            let (br, bc) = LUMA_BLOCK_RASTER[blk];
            let nc = predict_nc_luma_combined(
                luma_nc,
                &luma_nc_local,
                mb_x,
                mb_y,
                br,
                bc,
                self.mb_width,
            );
            let total_coeff =
                encode_residual_block(w, &ac_blocks[br * 4 + bc], nc, BlockKind::Luma16x16Ac)?;
            luma_nc_local[br * 4 + bc] = total_coeff as u8;
        }
        // Commit luma_nc_local to the picture-wide luma_nc table.
        for br in 0..4 {
            for bc in 0..4 {
                let gy = (mb_y as usize * 4 + br) * (self.mb_width as usize * 4)
                    + (mb_x as usize * 4 + bc);
                luma_nc[gy] = luma_nc_local[br * 4 + bc];
            }
        }

        // ------------------------------------------------------------------
        // Local luma reconstruction.
        // ------------------------------------------------------------------
        // Inverse of what we just emitted: dequant each AC block, dequant+
        // Hadamard the DC, re-inject DC[i] into position 0 of each AC block,
        // IDCT, add prediction, clamp.
        let mut dc_reconstructed = dc_block;
        inv_hadamard_4x4_dc(&mut dc_reconstructed, qp_y);
        for br in 0..4 {
            for bc in 0..4 {
                let mut res = ac_blocks[br * 4 + bc];
                dequantize_4x4(&mut res, qp_y);
                res[0] = dc_reconstructed[br * 4 + bc];
                idct_4x4(&mut res);
                for r in 0..4 {
                    for c in 0..4 {
                        let v = pred[(br * 4 + r) * 16 + (bc * 4 + c)] as i32 + res[r * 4 + c];
                        rec_y[y_off_mb + (br * 4 + r) * l_stride + (bc * 4 + c)] =
                            v.clamp(0, 255) as u8;
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Chroma pipeline (Cb then Cr).
        // ------------------------------------------------------------------
        let qpc = chroma_qp(qp_y, 0 /* chroma_qp_index_offset = 0 in our PPS */);
        let c_off_mb = (mb_y as usize * 8) * c_stride + mb_x as usize * 8;

        // DC Cb then DC Cr block (always coded when cbp_chroma>=1).
        let mut pred_cb = [0u8; 64];
        let mut pred_cr = [0u8; 64];
        predict_intra_chroma(&mut pred_cb, IntraChromaMode::Dc, &cneigh_cb);
        predict_intra_chroma(&mut pred_cr, IntraChromaMode::Dc, &cneigh_cr);

        let mut cb_ac: [[i32; 16]; 4] = [[0; 16]; 4];
        let mut cr_ac: [[i32; 16]; 4] = [[0; 16]; 4];
        let mut cb_dc = [0i32; 4];
        let mut cr_dc = [0i32; 4];

        for plane_is_cb in [true, false] {
            let (src, pred, ac_dst, dc_dst) = if plane_is_cb {
                (cb_src, &pred_cb, &mut cb_ac, &mut cb_dc)
            } else {
                (cr_src, &pred_cr, &mut cr_ac, &mut cr_dc)
            };
            for bi in 0..4usize {
                let br = bi >> 1;
                let bc = bi & 1;
                let mut residual = [0i32; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        let s = src[c_off_mb + (br * 4 + r) * c_stride + (bc * 4 + c)] as i32;
                        let p = pred[(br * 4 + r) * 8 + (bc * 4 + c)] as i32;
                        residual[r * 4 + c] = s - p;
                    }
                }
                forward_dct_4x4(&mut residual);
                dc_dst[bi] = residual[0];
                residual[0] = 0;
                ac_dst[bi] = residual;
                quantize_4x4_ac(&mut ac_dst[bi], qpc);
            }
            forward_hadamard_2x2(dc_dst);
            quantize_chroma_dc_2x2(dc_dst, qpc);
        }

        // Emit: chroma DC Cb, chroma DC Cr, then 4 Cb AC, then 4 Cr AC.
        {
            let mut dc_block_cb = [0i32; 16];
            for i in 0..4 {
                dc_block_cb[i] = cb_dc[i];
            }
            encode_residual_block(w, &dc_block_cb, 0, BlockKind::ChromaDc2x2)?;
            let mut dc_block_cr = [0i32; 16];
            for i in 0..4 {
                dc_block_cr[i] = cr_dc[i];
            }
            encode_residual_block(w, &dc_block_cr, 0, BlockKind::ChromaDc2x2)?;
        }

        let mut cb_nc_local = [0u8; 4];
        for bi in 0..4usize {
            let br = bi >> 1;
            let bc = bi & 1;
            let nc =
                predict_nc_chroma_combined(cb_nc, &cb_nc_local, mb_x, mb_y, br, bc, self.mb_width);
            let tc = encode_residual_block(w, &cb_ac[bi], nc, BlockKind::ChromaAc)?;
            cb_nc_local[bi] = tc as u8;
        }
        let mut cr_nc_local = [0u8; 4];
        for bi in 0..4usize {
            let br = bi >> 1;
            let bc = bi & 1;
            let nc =
                predict_nc_chroma_combined(cr_nc, &cr_nc_local, mb_x, mb_y, br, bc, self.mb_width);
            let tc = encode_residual_block(w, &cr_ac[bi], nc, BlockKind::ChromaAc)?;
            cr_nc_local[bi] = tc as u8;
        }
        // Commit chroma nc to picture-wide tables.
        for br in 0..2 {
            for bc in 0..2 {
                let gy = (mb_y as usize * 2 + br) * (self.mb_width as usize * 2)
                    + (mb_x as usize * 2 + bc);
                cb_nc[gy] = cb_nc_local[br * 2 + bc];
                cr_nc[gy] = cr_nc_local[br * 2 + bc];
            }
        }

        // ------------------------------------------------------------------
        // Local chroma reconstruction.
        // ------------------------------------------------------------------
        let mut cb_dc_rec = cb_dc;
        let mut cr_dc_rec = cr_dc;
        inv_hadamard_2x2_chroma_dc(&mut cb_dc_rec, qpc);
        inv_hadamard_2x2_chroma_dc(&mut cr_dc_rec, qpc);
        for plane_is_cb in [true, false] {
            let (rec, pred, ac, dc) = if plane_is_cb {
                (&mut *rec_cb, &pred_cb, &cb_ac, &cb_dc_rec)
            } else {
                (&mut *rec_cr, &pred_cr, &cr_ac, &cr_dc_rec)
            };
            for bi in 0..4usize {
                let br = bi >> 1;
                let bc = bi & 1;
                let mut res = ac[bi];
                dequantize_4x4(&mut res, qpc);
                res[0] = dc[bi];
                idct_4x4(&mut res);
                for r in 0..4 {
                    for c in 0..4 {
                        let v = pred[(br * 4 + r) * 8 + (bc * 4 + c)] as i32 + res[r * 4 + c];
                        rec[c_off_mb + (br * 4 + r) * c_stride + (bc * 4 + c)] =
                            v.clamp(0, 255) as u8;
                    }
                }
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------
    // Reference-picture + P-slice encoding.
    // -----------------------------------------------------------------

    /// Wrap the IDR's reconstruction buffers into a [`Picture`] tagged
    /// as a fully-intra frame so the next P-slice's MV predictor treats
    /// every neighbour as "available, intra" (§8.4.1.3.1: `refIdxLX = -1`
    /// with `mvLX = 0`). Chroma planes are copied as-is.
    fn build_ref_picture_intra(&self, rec_y: Vec<u8>, rec_cb: Vec<u8>, rec_cr: Vec<u8>) -> Picture {
        let mut pic = Picture::new_with_format(self.mb_width, self.mb_height, 1);
        pic.y = rec_y;
        pic.cb = rec_cb;
        pic.cr = rec_cr;
        for mb_y in 0..self.mb_height {
            for mb_x in 0..self.mb_width {
                let info = pic.mb_info_mut(mb_x, mb_y);
                *info = MbInfo {
                    qp_y: self.opts.qp,
                    coded: true,
                    intra: true,
                    ..Default::default()
                };
            }
        }
        pic
    }

    /// Encode one progressive P-frame referencing the previous
    /// reconstructed picture.
    fn encode_p_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        let l_stride = self.coded_width as usize;
        let coded_cw = (self.coded_width / 2) as usize;
        let coded_ch = (self.coded_height / 2) as usize;

        let y_src = plane_to_mb_aligned(
            &frame.planes[0].data,
            frame.planes[0].stride,
            self.width as usize,
            self.height as usize,
            self.coded_width as usize,
            self.coded_height as usize,
        );
        let cw = (self.width / 2) as usize;
        let ch = (self.height / 2) as usize;
        let cb_src = plane_to_mb_aligned(
            &frame.planes[1].data,
            frame.planes[1].stride,
            cw,
            ch,
            coded_cw,
            coded_ch,
        );
        let cr_src = plane_to_mb_aligned(
            &frame.planes[2].data,
            frame.planes[2].stride,
            cw,
            ch,
            coded_cw,
            coded_ch,
        );

        // The reference picture is consumed (replaced with a freshly
        // reconstructed `Picture` below). Take it out, decode into a new
        // sibling, then install the new one at the end.
        let ref_pic = self.ref_pic.take().expect("encode_p_frame: ref_pic");
        let mut cur = Picture::new_with_format(self.mb_width, self.mb_height, 1);

        let mut slice_rbsp = BitWriter::new();
        self.write_p_slice_header(&mut slice_rbsp);

        // §7.3.4 CAVLC P-slice data loop — walk MBs in raster order. A
        // contiguous run of P_Skip MBs is counted into `pending_skip_run`
        // and flushed (as an ue(v)) right before the next coded MB, or at
        // the very end of the slice. Picture-wide nC tables (§9.2.1.1) for
        // coded-MB CAVLC neighbour prediction live on `cur.mb_info`.
        let mut pending_skip_run: u32 = 0;
        let mut last_qp = self.opts.qp;

        for mb_y in 0..self.mb_height {
            for mb_x in 0..self.mb_width {
                // Integer-pel motion search (±16), SAD.
                let (best_mv, best_sad) = integer_me_16x16(
                    &y_src,
                    &ref_pic.y,
                    l_stride,
                    mb_x,
                    mb_y,
                    self.coded_width as usize,
                    self.coded_height as usize,
                );
                // Try P_Skip first: need the §8.4.1.1 skip predictor to
                // match, and the resulting motion-compensated residual to
                // quantise-to-zero.
                let skip_mv = predict_mv_pskip(&cur, mb_x, mb_y);
                let skip_ok = self.try_skip(
                    &cur, &ref_pic, &y_src, &cb_src, &cr_src, mb_x, mb_y, skip_mv,
                );
                if skip_ok {
                    pending_skip_run += 1;
                    // Record the skip on the cur picture for neighbour
                    // prediction (§8.4.1.1 ref_idx=0, mv=skip_mv; all
                    // luma_nc entries remain zero).
                    self.commit_skip_mb(&mut cur, &ref_pic, mb_x, mb_y, skip_mv, last_qp);
                    continue;
                }

                // Non-skip coded MB: flush any pending skip run.
                slice_rbsp.write_ue(pending_skip_run);
                pending_skip_run = 0;

                self.encode_p_mb(
                    &mut slice_rbsp,
                    &mut cur,
                    &ref_pic,
                    &y_src,
                    &cb_src,
                    &cr_src,
                    mb_x,
                    mb_y,
                    best_mv,
                    best_sad,
                    &mut last_qp,
                )?;
            }
        }
        // Trailing skip run (if the tail of the slice is all skips).
        slice_rbsp.write_ue(pending_skip_run);
        slice_rbsp.write_rbsp_trailing_bits();
        let rbsp = slice_rbsp.finish();

        // NAL header: non-IDR, nal_ref_idc = 2 (reference). 0x41 =
        // 0b0100_0001. Using nal_ref_idc = 2 distinguishes short-term
        // reference from the IDR's nal_ref_idc = 3 without changing
        // reference-picture marking semantics.
        let nal_header = 0x41u8;
        let mut slice_nal = Vec::with_capacity(1 + rbsp.len());
        slice_nal.push(nal_header);
        slice_nal.extend_from_slice(&rbsp);
        let slice_ebsp = rbsp_to_ebsp(&slice_nal);

        let mut out =
            Vec::with_capacity(slice_ebsp.len() + self.sps_nal.len() + self.pps_nal.len() + 12);
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&self.sps_nal);
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&self.pps_nal);
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&slice_ebsp);

        let mut pkt = Packet::new(0, frame.time_base, out);
        pkt.pts = frame.pts;
        pkt.dts = frame.pts;
        pkt.flags = PacketFlags::default();
        self.packets.push_back(pkt);

        self.frame_num = self.frame_num.wrapping_add(1);
        self.frames_since_idr = self.frames_since_idr.saturating_add(1);
        self.ref_pic = Some(cur);
        Ok(())
    }

    /// §7.3.3 P-slice header (non-IDR) — single slice, single ref, no
    /// weighted prediction, sliding-window DPB marking.
    fn write_p_slice_header(&self, w: &mut BitWriter) {
        w.write_ue(0); // first_mb_in_slice
        w.write_ue(5); // slice_type = 5 (P, single-type-per-picture variant)
        w.write_ue(self.opts.pps_id); // pic_parameter_set_id
                                      // frame_num — 4 bits (log2_max_frame_num_minus4 = 0).
        w.write_bits(self.frame_num & 0xF, 4);
        // No field_pic_flag when frame_mbs_only_flag = 1 (progressive SPS).
        // pic_order_cnt_lsb — 4 bits. Use `2 * frame_num` so each P-slice
        // sits strictly above the IDR's POC 0.
        w.write_bits((self.frame_num * 2) & 0xF, 4);
        // num_ref_idx_active_override_flag = 0 — inherits the PPS's
        // num_ref_idx_l0_default_active_minus1 = 0.
        w.write_flag(false);
        // ref_pic_list_modification_flag_l0 = 0.
        w.write_flag(false);
        // dec_ref_pic_marking — non-IDR:
        //   adaptive_ref_pic_marking_mode_flag = 0 (sliding window).
        w.write_flag(false);
        // slice_qp_delta.
        w.write_se(self.opts.qp - 26);
        // disable_deblocking_filter_idc = 1.
        w.write_ue(1);
    }

    /// P_Skip feasibility check (§8.4.1.1) — the implied MV comes from
    /// `predict_mv_pskip`, ref_idx = 0, and the resulting motion
    /// compensation must already be within the quant round-trip's
    /// tolerance of the source (i.e. all residual blocks quantise to
    /// zero).
    fn try_skip(
        &self,
        cur: &Picture,
        ref_pic: &Picture,
        y_src: &[u8],
        cb_src: &[u8],
        cr_src: &[u8],
        mb_x: u32,
        mb_y: u32,
        skip_mv: (i16, i16),
    ) -> bool {
        let _ = cur;
        let qp_y = self.opts.qp;
        let qpc = chroma_qp(qp_y, 0);
        let l_stride = self.coded_width as usize;
        let c_stride = (self.coded_width / 2) as usize;

        // Motion-compensate luma 16×16 at integer-pel offset from the ref.
        let mut pred_y = [0u8; 256];
        luma_mc_plane(
            &mut pred_y,
            &ref_pic.y,
            l_stride,
            ref_pic.width as i32,
            ref_pic.height as i32,
            (mb_x as i32) * 16,
            (mb_y as i32) * 16,
            skip_mv.0 as i32, // already quarter-pel (integer)
            skip_mv.1 as i32,
            16,
            16,
        );
        let y_off_mb = (mb_y as usize * 16) * l_stride + mb_x as usize * 16;
        for br in 0..4usize {
            for bc in 0..4usize {
                let mut residual = [0i32; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        let src = y_src[y_off_mb + (br * 4 + r) * l_stride + (bc * 4 + c)] as i32;
                        let p = pred_y[(br * 4 + r) * 16 + (bc * 4 + c)] as i32;
                        residual[r * 4 + c] = src - p;
                    }
                }
                forward_dct_4x4(&mut residual);
                quantize_4x4_ac(&mut residual, qp_y);
                for &v in residual.iter() {
                    if v != 0 {
                        return false;
                    }
                }
            }
        }

        // Chroma skip check — chroma MC at half-MV, DC Hadamard + AC both
        // must quantise to zero.
        let mut pred_cb = [0u8; 64];
        let mut pred_cr = [0u8; 64];
        let c_plane_w = (ref_pic.width / 2) as i32;
        let c_plane_h = (ref_pic.height / 2) as i32;
        chroma_mc(
            &mut pred_cb,
            &ref_pic.cb,
            c_stride,
            c_plane_w,
            c_plane_h,
            (mb_x as i32) * 8,
            (mb_y as i32) * 8,
            skip_mv.0 as i32,
            skip_mv.1 as i32,
            8,
            8,
        );
        chroma_mc(
            &mut pred_cr,
            &ref_pic.cr,
            c_stride,
            c_plane_w,
            c_plane_h,
            (mb_x as i32) * 8,
            (mb_y as i32) * 8,
            skip_mv.0 as i32,
            skip_mv.1 as i32,
            8,
            8,
        );
        let c_off_mb = (mb_y as usize * 8) * c_stride + mb_x as usize * 8;
        for plane_is_cb in [true, false] {
            let (src, pred) = if plane_is_cb {
                (cb_src, &pred_cb)
            } else {
                (cr_src, &pred_cr)
            };
            let mut dc = [0i32; 4];
            let mut ac: [[i32; 16]; 4] = [[0; 16]; 4];
            for bi in 0..4usize {
                let br = bi >> 1;
                let bc = bi & 1;
                let mut residual = [0i32; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        let s = src[c_off_mb + (br * 4 + r) * c_stride + (bc * 4 + c)] as i32;
                        let p = pred[(br * 4 + r) * 8 + (bc * 4 + c)] as i32;
                        residual[r * 4 + c] = s - p;
                    }
                }
                forward_dct_4x4(&mut residual);
                dc[bi] = residual[0];
                residual[0] = 0;
                ac[bi] = residual;
                quantize_4x4_ac(&mut ac[bi], qpc);
            }
            forward_hadamard_2x2(&mut dc);
            quantize_chroma_dc_2x2(&mut dc, qpc);
            if dc.iter().any(|&v| v != 0) {
                return false;
            }
            for b in ac.iter() {
                if b.iter().any(|&v| v != 0) {
                    return false;
                }
            }
        }
        true
    }

    /// Commit a P_Skip MB into the current picture: copies the MC
    /// prediction into the reconstruction buffers and records the implied
    /// MV / ref_idx on `MbInfo` so subsequent MVP lookups and chroma-nC
    /// prediction see the right neighbour state.
    fn commit_skip_mb(
        &self,
        cur: &mut Picture,
        ref_pic: &Picture,
        mb_x: u32,
        mb_y: u32,
        skip_mv: (i16, i16),
        qp_y: i32,
    ) {
        let l_stride = self.coded_width as usize;
        let c_stride = (self.coded_width / 2) as usize;
        let mut pred_y = [0u8; 256];
        luma_mc_plane(
            &mut pred_y,
            &ref_pic.y,
            l_stride,
            ref_pic.width as i32,
            ref_pic.height as i32,
            (mb_x as i32) * 16,
            (mb_y as i32) * 16,
            skip_mv.0 as i32,
            skip_mv.1 as i32,
            16,
            16,
        );
        let y_off_mb = (mb_y as usize * 16) * l_stride + mb_x as usize * 16;
        for r in 0..16 {
            for c in 0..16 {
                cur.y[y_off_mb + r * l_stride + c] = pred_y[r * 16 + c];
            }
        }
        let mut pred_cb = [0u8; 64];
        let mut pred_cr = [0u8; 64];
        let c_plane_w = (ref_pic.width / 2) as i32;
        let c_plane_h = (ref_pic.height / 2) as i32;
        chroma_mc(
            &mut pred_cb,
            &ref_pic.cb,
            c_stride,
            c_plane_w,
            c_plane_h,
            (mb_x as i32) * 8,
            (mb_y as i32) * 8,
            skip_mv.0 as i32,
            skip_mv.1 as i32,
            8,
            8,
        );
        chroma_mc(
            &mut pred_cr,
            &ref_pic.cr,
            c_stride,
            c_plane_w,
            c_plane_h,
            (mb_x as i32) * 8,
            (mb_y as i32) * 8,
            skip_mv.0 as i32,
            skip_mv.1 as i32,
            8,
            8,
        );
        let c_off_mb = (mb_y as usize * 8) * c_stride + mb_x as usize * 8;
        for r in 0..8 {
            for c in 0..8 {
                cur.cb[c_off_mb + r * c_stride + c] = pred_cb[r * 8 + c];
                cur.cr[c_off_mb + r * c_stride + c] = pred_cr[r * 8 + c];
            }
        }
        let info = cur.mb_info_mut(mb_x, mb_y);
        *info = MbInfo {
            qp_y,
            coded: true,
            intra: false,
            skipped: true,
            mb_type_p: Some(PMbType::Inter {
                partition: PPartition::P16x16,
            }),
            p_partition: Some(PPartition::P16x16),
            mv_l0: [skip_mv; 16],
            ref_idx_l0: [0; 16],
            ..Default::default()
        };
    }

    /// Encode one P_L0_16×16 coded MB: mb_type, mvd, CBP, qp_delta,
    /// residual (DC Hadamard + 16 AC for luma when cbp_luma = 0xF; chroma
    /// DC Hadamard + 4 AC when cbp_chroma = 2; nothing if luma / chroma
    /// residual fully quantised to zero).
    #[allow(clippy::too_many_arguments)]
    fn encode_p_mb(
        &self,
        w: &mut BitWriter,
        cur: &mut Picture,
        ref_pic: &Picture,
        y_src: &[u8],
        cb_src: &[u8],
        cr_src: &[u8],
        mb_x: u32,
        mb_y: u32,
        best_mv: (i16, i16),
        _best_sad: u32,
        last_qp: &mut i32,
    ) -> Result<()> {
        let qp_y = self.opts.qp;
        let qpc = chroma_qp(qp_y, 0);
        let l_stride = self.coded_width as usize;
        let c_stride = (self.coded_width / 2) as usize;
        let _ = last_qp;

        // MV predictor: ref_idx=0, P_L0_16×16 partition (br=bc=0, 4×4).
        let pmv = predict_mv_l0(cur, mb_x, mb_y, 0, 0, 4, 4, 0);
        let mvd = (
            best_mv.0.wrapping_sub(pmv.0) as i32,
            best_mv.1.wrapping_sub(pmv.1) as i32,
        );

        // Motion-compensate luma from reference at best_mv.
        let mut pred_y = [0u8; 256];
        luma_mc_plane(
            &mut pred_y,
            &ref_pic.y,
            l_stride,
            ref_pic.width as i32,
            ref_pic.height as i32,
            (mb_x as i32) * 16,
            (mb_y as i32) * 16,
            best_mv.0 as i32,
            best_mv.1 as i32,
            16,
            16,
        );
        // Chroma MC.
        let mut pred_cb = [0u8; 64];
        let mut pred_cr = [0u8; 64];
        let c_plane_w = (ref_pic.width / 2) as i32;
        let c_plane_h = (ref_pic.height / 2) as i32;
        chroma_mc(
            &mut pred_cb,
            &ref_pic.cb,
            c_stride,
            c_plane_w,
            c_plane_h,
            (mb_x as i32) * 8,
            (mb_y as i32) * 8,
            best_mv.0 as i32,
            best_mv.1 as i32,
            8,
            8,
        );
        chroma_mc(
            &mut pred_cr,
            &ref_pic.cr,
            c_stride,
            c_plane_w,
            c_plane_h,
            (mb_x as i32) * 8,
            (mb_y as i32) * 8,
            best_mv.0 as i32,
            best_mv.1 as i32,
            8,
            8,
        );

        // Forward 4×4 DCT + quant for each of the 16 luma AC blocks. For
        // P-slice inter the 16×16 partition does NOT go through a DC
        // Hadamard — every 4×4 is a plain Luma4x4 block and the CBP-luma
        // 8×8 bits are set when any sub-block has a non-zero coefficient.
        let y_off_mb = (mb_y as usize * 16) * l_stride + mb_x as usize * 16;
        let mut luma_blocks: [[i32; 16]; 16] = [[0; 16]; 16];
        for br in 0..4usize {
            for bc in 0..4usize {
                let mut residual = [0i32; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        let src = y_src[y_off_mb + (br * 4 + r) * l_stride + (bc * 4 + c)] as i32;
                        let p = pred_y[(br * 4 + r) * 16 + (bc * 4 + c)] as i32;
                        residual[r * 4 + c] = src - p;
                    }
                }
                forward_dct_4x4(&mut residual);
                quantize_4x4_ac(&mut residual, qp_y);
                luma_blocks[br * 4 + bc] = residual;
            }
        }

        // Compute cbp_luma (one bit per 8×8 sub-block).
        let mut cbp_luma: u8 = 0;
        for blk8 in 0..4u8 {
            let br8 = (blk8 >> 1) as usize;
            let bc8 = (blk8 & 1) as usize;
            let mut nz = false;
            for br_off in 0..2 {
                for bc_off in 0..2 {
                    let br = br8 * 2 + br_off;
                    let bc = bc8 * 2 + bc_off;
                    if luma_blocks[br * 4 + bc].iter().any(|&v| v != 0) {
                        nz = true;
                    }
                }
            }
            if nz {
                cbp_luma |= 1 << blk8;
            }
        }

        // Chroma residual: DC Hadamard pass + 4 AC blocks per plane.
        let c_off_mb = (mb_y as usize * 8) * c_stride + mb_x as usize * 8;
        let mut cb_ac: [[i32; 16]; 4] = [[0; 16]; 4];
        let mut cr_ac: [[i32; 16]; 4] = [[0; 16]; 4];
        let mut cb_dc = [0i32; 4];
        let mut cr_dc = [0i32; 4];
        for plane_is_cb in [true, false] {
            let (src, pred, ac_dst, dc_dst) = if plane_is_cb {
                (cb_src, &pred_cb, &mut cb_ac, &mut cb_dc)
            } else {
                (cr_src, &pred_cr, &mut cr_ac, &mut cr_dc)
            };
            for bi in 0..4usize {
                let br = bi >> 1;
                let bc = bi & 1;
                let mut residual = [0i32; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        let s = src[c_off_mb + (br * 4 + r) * c_stride + (bc * 4 + c)] as i32;
                        let p = pred[(br * 4 + r) * 8 + (bc * 4 + c)] as i32;
                        residual[r * 4 + c] = s - p;
                    }
                }
                forward_dct_4x4(&mut residual);
                dc_dst[bi] = residual[0];
                residual[0] = 0;
                ac_dst[bi] = residual;
                quantize_4x4_ac(&mut ac_dst[bi], qpc);
            }
            forward_hadamard_2x2(dc_dst);
            quantize_chroma_dc_2x2(dc_dst, qpc);
        }
        let chroma_dc_nz = cb_dc.iter().any(|&v| v != 0) || cr_dc.iter().any(|&v| v != 0);
        let chroma_ac_nz = cb_ac.iter().any(|b| b.iter().any(|&v| v != 0))
            || cr_ac.iter().any(|b| b.iter().any(|&v| v != 0));
        let cbp_chroma: u8 = if chroma_ac_nz {
            2
        } else if chroma_dc_nz {
            1
        } else {
            0
        };

        // Write mb_type = 0 (P_L0_16×16).
        w.write_ue(0);
        // ref_idx_l0 omitted — num_ref_idx_l0_active_minus1 = 0, so the
        // decoder infers ref_idx = 0 without consuming any bits.
        // mvd_l0_x, mvd_l0_y.
        w.write_se(mvd.0);
        w.write_se(mvd.1);

        // Inter CBP via me(v): find index in ME_INTER_4_2_0.
        let cbp_value = ((cbp_chroma & 0x3) << 4) | (cbp_luma & 0xF);
        let cbp_idx = ME_INTER_4_2_0
            .iter()
            .position(|&v| v == cbp_value)
            .ok_or_else(|| {
                Error::invalid(format!(
                    "h264 encoder: no me(v) inter CBP index for value {cbp_value}"
                ))
            })? as u32;
        w.write_ue(cbp_idx);

        // §7.3.5.1 — mb_qp_delta is present only when cbp_luma != 0 or
        // cbp_chroma != 0. Our PPS has `transform_8x8_mode_flag = 0`, so
        // no transform_size_8x8_flag.
        if cbp_luma != 0 || cbp_chroma != 0 {
            // Fixed QP → delta = 0 every time.
            w.write_se(0);
        }

        // Luma residual: 16 Luma4x4 blocks. Order: for each of 4 8×8 sub-MBs,
        // if cbp_luma bit set, emit its 4 child 4×4 blocks in raster order.
        // §7.3.5.3 specifies inter 4×4 residuals use BlockKind::Luma4x4.
        for blk8 in 0..4u8 {
            if (cbp_luma >> blk8) & 1 == 0 {
                continue;
            }
            let br8 = (blk8 >> 1) as usize;
            let bc8 = (blk8 & 1) as usize;
            for sub in 0..4usize {
                let sub_br = sub >> 1;
                let sub_bc = sub & 1;
                let br = br8 * 2 + sub_br;
                let bc = bc8 * 2 + sub_bc;
                let nc = predict_inter_nc_luma_local(cur, mb_x, mb_y, br, bc);
                let tc =
                    encode_residual_block(w, &luma_blocks[br * 4 + bc], nc, BlockKind::Luma4x4)?;
                cur.mb_info_mut(mb_x, mb_y).luma_nc[br * 4 + bc] = tc as u8;
            }
        }

        // Chroma residual — chroma DC first (both Cb and Cr), then AC per
        // plane if cbp_chroma == 2.
        if cbp_chroma >= 1 {
            let mut dcb = [0i32; 16];
            let mut dcr = [0i32; 16];
            dcb[..4].copy_from_slice(&cb_dc);
            dcr[..4].copy_from_slice(&cr_dc);
            encode_residual_block(w, &dcb, 0, BlockKind::ChromaDc2x2)?;
            encode_residual_block(w, &dcr, 0, BlockKind::ChromaDc2x2)?;
        }
        if cbp_chroma == 2 {
            for plane_is_cb in [true, false] {
                for bi in 0..4usize {
                    let br = bi >> 1;
                    let bc = bi & 1;
                    let nc = predict_inter_nc_chroma_local(cur, mb_x, mb_y, plane_is_cb, br, bc);
                    let ac = if plane_is_cb { &cb_ac[bi] } else { &cr_ac[bi] };
                    let tc = encode_residual_block(w, ac, nc, BlockKind::ChromaAc)?;
                    let info = cur.mb_info_mut(mb_x, mb_y);
                    if plane_is_cb {
                        info.cb_nc[bi] = tc as u8;
                    } else {
                        info.cr_nc[bi] = tc as u8;
                    }
                }
            }
        }

        // Local reconstruction: inverse quant + IDCT, re-inject chroma DC,
        // add to MC prediction, clamp, write into `cur`.
        // Luma.
        for br in 0..4usize {
            for bc in 0..4usize {
                let mut res = luma_blocks[br * 4 + bc];
                dequantize_4x4(&mut res, qp_y);
                idct_4x4(&mut res);
                for r in 0..4 {
                    for c in 0..4 {
                        let p = pred_y[(br * 4 + r) * 16 + (bc * 4 + c)] as i32;
                        let v = p + res[r * 4 + c];
                        cur.y[y_off_mb + (br * 4 + r) * l_stride + (bc * 4 + c)] =
                            v.clamp(0, 255) as u8;
                    }
                }
            }
        }
        // Chroma.
        let mut cb_dc_rec = cb_dc;
        let mut cr_dc_rec = cr_dc;
        inv_hadamard_2x2_chroma_dc(&mut cb_dc_rec, qpc);
        inv_hadamard_2x2_chroma_dc(&mut cr_dc_rec, qpc);
        for plane_is_cb in [true, false] {
            let (rec, pred, ac, dc) = if plane_is_cb {
                (cur.cb.as_mut_slice(), &pred_cb, &cb_ac, &cb_dc_rec)
            } else {
                (cur.cr.as_mut_slice(), &pred_cr, &cr_ac, &cr_dc_rec)
            };
            for bi in 0..4usize {
                let br = bi >> 1;
                let bc = bi & 1;
                let mut res = ac[bi];
                dequantize_4x4(&mut res, qpc);
                res[0] = dc[bi];
                idct_4x4(&mut res);
                for r in 0..4 {
                    for c in 0..4 {
                        let p = pred[(br * 4 + r) * 8 + (bc * 4 + c)] as i32;
                        let v = p + res[r * 4 + c];
                        rec[c_off_mb + (br * 4 + r) * c_stride + (bc * 4 + c)] =
                            v.clamp(0, 255) as u8;
                    }
                }
            }
        }

        // MbInfo: mark inter, record MV + ref_idx + CBP + qp.
        let info = cur.mb_info_mut(mb_x, mb_y);
        info.qp_y = qp_y;
        info.coded = true;
        info.intra = false;
        info.skipped = false;
        info.mb_type_p = Some(PMbType::Inter {
            partition: PPartition::P16x16,
        });
        info.p_partition = Some(PPartition::P16x16);
        info.mv_l0 = [best_mv; 16];
        info.ref_idx_l0 = [0; 16];
        info.cbp_luma = cbp_luma;
        info.cbp_chroma = cbp_chroma;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Prep source bytes for encode: if the incoming plane is already the
/// coded size, re-pack its bytes into a contiguous `coded_w × coded_h`
/// buffer. If smaller, replicate edges on the right / bottom.
fn plane_to_mb_aligned(
    src: &[u8],
    src_stride: usize,
    src_w: usize,
    src_h: usize,
    coded_w: usize,
    coded_h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; coded_w * coded_h];
    for r in 0..coded_h {
        let src_r = r.min(src_h.saturating_sub(1));
        for c in 0..coded_w {
            let src_c = c.min(src_w.saturating_sub(1));
            out[r * coded_w + c] = src[src_r * src_stride + src_c];
        }
    }
    out
}

fn mb_w_of(mb_width: u32) -> u32 {
    mb_width
}

/// Predict luma nC for the block at (mb_x, mb_y, br, bc), using the
/// picture-wide `luma_nc` (global, 4×4-block raster) plus in-progress
/// values for blocks within this MB (`local`, 16 entries in 4×4 MB raster).
fn predict_nc_luma_combined(
    global: &[u8],
    local: &[u8; 16],
    mb_x: u32,
    mb_y: u32,
    br: usize,
    bc: usize,
    mb_width: u32,
) -> i32 {
    let stride = mb_width as usize * 4;
    // Left neighbour.
    let left: Option<u8> = if bc > 0 {
        Some(local[br * 4 + bc - 1])
    } else if mb_x > 0 {
        // Block just to the left of current MB: at (mb_x-1, mb_y), 4×4 block (br, 3).
        let gy = (mb_y as usize * 4 + br) * stride + ((mb_x - 1) as usize * 4 + 3);
        Some(global[gy])
    } else {
        None
    };
    let top: Option<u8> = if br > 0 {
        Some(local[(br - 1) * 4 + bc])
    } else if mb_y > 0 {
        let gy = ((mb_y - 1) as usize * 4 + 3) * stride + (mb_x as usize * 4 + bc);
        Some(global[gy])
    } else {
        None
    };
    nc_from(left, top)
}

fn predict_nc_luma(
    global: &[u8],
    mb_x: u32,
    mb_y: u32,
    br: usize,
    bc: usize,
    mb_width: u32,
) -> i32 {
    let stride = mb_width as usize * 4;
    let left: Option<u8> = if bc > 0 {
        // This overload is only used for the first block of the MB where nothing
        // inside the MB has been written yet, so the left neighbour at (br,0)
        // must come from the previous MB.
        None
    } else if mb_x > 0 {
        let gy = (mb_y as usize * 4 + br) * stride + ((mb_x - 1) as usize * 4 + 3);
        Some(global[gy])
    } else {
        None
    };
    let top: Option<u8> = if br > 0 {
        None
    } else if mb_y > 0 {
        let gy = ((mb_y - 1) as usize * 4 + 3) * stride + (mb_x as usize * 4 + bc);
        Some(global[gy])
    } else {
        None
    };
    nc_from(left, top)
}

fn predict_nc_chroma_combined(
    global: &[u8],
    local: &[u8; 4],
    mb_x: u32,
    mb_y: u32,
    br: usize,
    bc: usize,
    mb_width: u32,
) -> i32 {
    let stride = mb_width as usize * 2;
    let left: Option<u8> = if bc > 0 {
        Some(local[br * 2 + bc - 1])
    } else if mb_x > 0 {
        let gy = (mb_y as usize * 2 + br) * stride + ((mb_x - 1) as usize * 2 + 1);
        Some(global[gy])
    } else {
        None
    };
    let top: Option<u8> = if br > 0 {
        Some(local[(br - 1) * 2 + bc])
    } else if mb_y > 0 {
        let gy = ((mb_y - 1) as usize * 2 + 1) * stride + (mb_x as usize * 2 + bc);
        Some(global[gy])
    } else {
        None
    };
    nc_from(left, top)
}

fn nc_from(left: Option<u8>, top: Option<u8>) -> i32 {
    match (left, top) {
        (Some(l), Some(t)) => (l as i32 + t as i32 + 1) >> 1,
        (Some(l), None) => l as i32,
        (None, Some(t)) => t as i32,
        (None, None) => 0,
    }
}

/// Integer-pel motion estimation over the 16×16 luma macroblock at
/// `(mb_x, mb_y)` against a ±16-pixel window of the reference plane.
/// Returns `(mv, sad)` — `mv` is in quarter-pel units so downstream MC
/// callers can read it unchanged; SAD is kept so callers can compare two
/// alternatives (e.g. zero-MV vs best-search) without re-reading.
fn integer_me_16x16(
    src: &[u8],
    ref_y: &[u8],
    stride: usize,
    mb_x: u32,
    mb_y: u32,
    coded_w: usize,
    coded_h: usize,
) -> ((i16, i16), u32) {
    let mb_ox = mb_x as usize * 16;
    let mb_oy = mb_y as usize * 16;
    let search_radius: i32 = 16;
    let mut best = (0i32, 0i32);
    // Zero-MV first pass so an exactly-identical frame yields `P_Skip`.
    let mut best_sad = sad_16x16(src, ref_y, stride, mb_ox, mb_oy, 0, 0, coded_w, coded_h);
    // Full search over ±search_radius pixels.
    for dy in -search_radius..=search_radius {
        for dx in -search_radius..=search_radius {
            if dx == 0 && dy == 0 {
                continue;
            }
            let sad = sad_16x16(src, ref_y, stride, mb_ox, mb_oy, dx, dy, coded_w, coded_h);
            if sad < best_sad {
                best_sad = sad;
                best = (dx, dy);
            }
        }
    }
    // Convert integer-pel (dx, dy) to quarter-pel MV by × 4.
    (((best.0 * 4) as i16, (best.1 * 4) as i16), best_sad)
}

fn sad_16x16(
    src: &[u8],
    ref_y: &[u8],
    stride: usize,
    mb_ox: usize,
    mb_oy: usize,
    dx: i32,
    dy: i32,
    coded_w: usize,
    coded_h: usize,
) -> u32 {
    let mut sad: u32 = 0;
    for r in 0..16i32 {
        for c in 0..16i32 {
            let sy = mb_oy as i32 + r;
            let sx = mb_ox as i32 + c;
            let ry = (sy + dy).clamp(0, coded_h as i32 - 1) as usize;
            let rx = (sx + dx).clamp(0, coded_w as i32 - 1) as usize;
            let s = src[sy as usize * stride + sx as usize] as i32;
            let p = ref_y[ry * stride + rx] as i32;
            sad += (s - p).unsigned_abs();
        }
    }
    sad
}

/// Predict nC for an inter 4×4 luma block at `(br, bc)` within MB
/// `(mb_x, mb_y)` using only the `cur` [`Picture`]'s MbInfo — mirrors
/// [`crate::p_mb::predict_inter_nc_luma`] but bounded to the encoder's
/// single-slice scope.
fn predict_inter_nc_luma_local(cur: &Picture, mb_x: u32, mb_y: u32, br: usize, bc: usize) -> i32 {
    let info_here = cur.mb_info_at(mb_x, mb_y);
    let left = if bc > 0 {
        Some(info_here.luma_nc[br * 4 + bc - 1])
    } else if mb_x > 0 {
        let info = cur.mb_info_at(mb_x - 1, mb_y);
        if info.coded {
            Some(info.luma_nc[br * 4 + 3])
        } else {
            None
        }
    } else {
        None
    };
    let top = if br > 0 {
        Some(info_here.luma_nc[(br - 1) * 4 + bc])
    } else if mb_y > 0 {
        let info = cur.mb_info_at(mb_x, mb_y - 1);
        if info.coded {
            Some(info.luma_nc[12 + bc])
        } else {
            None
        }
    } else {
        None
    };
    nc_from(left, top)
}

/// Same shape as [`predict_inter_nc_luma_local`] but for chroma 4×4
/// sub-blocks (2×2 layout in 4:2:0).
fn predict_inter_nc_chroma_local(
    cur: &Picture,
    mb_x: u32,
    mb_y: u32,
    cb: bool,
    br: usize,
    bc: usize,
) -> i32 {
    let pick = |info: &MbInfo, sub: usize| -> u8 {
        if cb {
            info.cb_nc[sub]
        } else {
            info.cr_nc[sub]
        }
    };
    let info_here = cur.mb_info_at(mb_x, mb_y);
    let left = if bc > 0 {
        Some(pick(info_here, br * 2 + bc - 1))
    } else if mb_x > 0 {
        let info = cur.mb_info_at(mb_x - 1, mb_y);
        if info.coded {
            Some(pick(info, br * 2 + 1))
        } else {
            None
        }
    } else {
        None
    };
    let top = if br > 0 {
        Some(pick(info_here, (br - 1) * 2 + bc))
    } else if mb_y > 0 {
        let info = cur.mb_info_at(mb_x, mb_y - 1);
        if info.coded {
            Some(pick(info, 2 + bc))
        } else {
            None
        }
    } else {
        None
    };
    nc_from(left, top)
}

fn collect_intra16x16_neighbours(
    rec: &[u8],
    stride: usize,
    mb_x: u32,
    mb_y: u32,
) -> Intra16x16Neighbours {
    let mb_ox = mb_x as usize * 16;
    let mb_oy = mb_y as usize * 16;
    let top_avail = mb_y > 0;
    let left_avail = mb_x > 0;
    let tl_avail = top_avail && left_avail;
    let mut top = [0u8; 16];
    if top_avail {
        let row = (mb_oy - 1) * stride;
        for i in 0..16 {
            top[i] = rec[row + mb_ox + i];
        }
    }
    let mut left = [0u8; 16];
    if left_avail {
        for i in 0..16 {
            left[i] = rec[(mb_oy + i) * stride + mb_ox - 1];
        }
    }
    let top_left = if tl_avail {
        rec[(mb_oy - 1) * stride + mb_ox - 1]
    } else {
        0
    };
    Intra16x16Neighbours {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
    }
}

fn collect_chroma_neighbours(
    rec: &[u8],
    stride: usize,
    mb_x: u32,
    mb_y: u32,
) -> IntraChromaNeighbours {
    let mb_ox = mb_x as usize * 8;
    let mb_oy = mb_y as usize * 8;
    let top_avail = mb_y > 0;
    let left_avail = mb_x > 0;
    let tl_avail = top_avail && left_avail;
    let mut top = [0u8; 8];
    if top_avail {
        let row = (mb_oy - 1) * stride;
        for i in 0..8 {
            top[i] = rec[row + mb_ox + i];
        }
    }
    let mut left = [0u8; 8];
    if left_avail {
        for i in 0..8 {
            left[i] = rec[(mb_oy + i) * stride + mb_ox - 1];
        }
    }
    let top_left = if tl_avail {
        rec[(mb_oy - 1) * stride + mb_ox - 1]
    } else {
        0
    };
    IntraChromaNeighbours {
        top,
        left,
        top_left,
        top_available: top_avail,
        left_available: left_avail,
        top_left_available: tl_avail,
    }
}

// ---------------------------------------------------------------------------
// SPS / PPS builders.
// ---------------------------------------------------------------------------

/// Build an SPS NAL (header byte + RBSP + emulation prevention). Baseline
/// profile, level 3.0, chroma 4:2:0, 8-bit, single colour plane. Includes
/// frame_cropping when width/height are not multiples of 16.
fn build_sps_nal(width: u32, height: u32, sps_id: u32, paff: bool) -> Result<Vec<u8>> {
    let mb_w = width.div_ceil(16);
    let mb_h = height.div_ceil(16);
    let coded_w = mb_w * 16;
    let coded_h = mb_h * 16;
    let crop_right = (coded_w - width) / 2;
    let crop_bottom = (coded_h - height) / 2;
    let needs_crop = (coded_w - width) != 0 || (coded_h - height) != 0;

    // Build RBSP body: 3 header bytes (profile, constraints, level) + bit stream.
    let mut body = Vec::new();
    body.push(66u8); // profile_idc = Baseline
                     // constraint_set{0,1}_flag = 1, others 0 → 0xC0
    body.push(0xC0);
    body.push(30u8); // level_idc = 30 → level 3.0

    let mut w = BitWriter::new();
    w.write_ue(sps_id);
    // log2_max_frame_num_minus4 = 0 (frame_num is 4 bits).
    w.write_ue(0);
    // pic_order_cnt_type = 0, log2_max_pic_order_cnt_lsb_minus4 = 0 (4 bits).
    w.write_ue(0);
    w.write_ue(0);
    // num_ref_frames = 1, gaps_in_frame_num_value_allowed = 0.
    w.write_ue(1);
    w.write_flag(false);
    // pic_width_in_mbs_minus1, pic_height_in_map_units_minus1.
    w.write_ue(mb_w - 1);
    w.write_ue(mb_h - 1);
    // §7.3.2.1.1 — frame_mbs_only_flag: 1 for progressive, 0 when the CVS
    // may carry PAFF field slices. When 0 we follow with
    // mb_adaptive_frame_field_flag = 0 (no MBAFF, so every slice with
    // field_pic_flag = 0 is a plain frame — which our progressive
    // encoder doesn't emit alongside PAFF fields anyway).
    w.write_flag(!paff);
    if paff {
        w.write_flag(false); // mb_adaptive_frame_field_flag
    }
    // direct_8x8_inference_flag = 1.
    w.write_flag(true);
    // frame_cropping_flag.
    w.write_flag(needs_crop);
    if needs_crop {
        w.write_ue(0); // crop_left
        w.write_ue(crop_right);
        w.write_ue(0); // crop_top
        w.write_ue(crop_bottom);
    }
    // vui_parameters_present_flag = 0.
    w.write_flag(false);
    // rbsp_trailing_bits.
    w.write_rbsp_trailing_bits();
    let rbsp_tail = w.finish();
    body.extend_from_slice(&rbsp_tail);

    // NAL header byte: 0x67 (forbidden=0, nal_ref_idc=3, type=7 SPS).
    let mut nal = Vec::with_capacity(1 + body.len());
    nal.push(0x67);
    nal.extend_from_slice(&rbsp_to_ebsp(&body));
    Ok(nal)
}

fn build_pps_nal(pps_id: u32, sps_id: u32) -> Result<Vec<u8>> {
    let mut w = BitWriter::new();
    w.write_ue(pps_id);
    w.write_ue(sps_id);
    w.write_flag(false); // entropy_coding_mode_flag = 0 (CAVLC)
    w.write_flag(false); // bottom_field_pic_order_in_frame_present_flag
    w.write_ue(0); // num_slice_groups_minus1 = 0
                   // no slice group map
    w.write_ue(0); // num_ref_idx_l0_default_active_minus1
    w.write_ue(0); // num_ref_idx_l1_default_active_minus1
    w.write_flag(false); // weighted_pred_flag
    w.write_bits(0, 2); // weighted_bipred_idc = 0
    w.write_se(0); // pic_init_qp_minus26
    w.write_se(0); // pic_init_qs_minus26
    w.write_se(0); // chroma_qp_index_offset
    w.write_flag(true); // deblocking_filter_control_present_flag
    w.write_flag(false); // constrained_intra_pred_flag
    w.write_flag(false); // redundant_pic_cnt_present_flag
    w.write_rbsp_trailing_bits();
    let rbsp = w.finish();

    let mut nal = Vec::with_capacity(1 + rbsp.len());
    nal.push(0x68); // forbidden=0, nal_ref_idc=3, type=8 PPS
    nal.extend_from_slice(&rbsp_to_ebsp(&rbsp));
    Ok(nal)
}

/// Build an `AVCDecoderConfigurationRecord` ("avcC") for MP4 wrapping.
fn build_avcc(sps_nal: &[u8], pps_nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(sps_nal.len() + pps_nal.len() + 16);
    out.push(1); // configurationVersion
                 // The SPS body (after the 1-byte NAL header) starts with profile, compat, level.
    out.push(sps_nal[1]); // profile_indication
    out.push(sps_nal[2]); // profile_compatibility
    out.push(sps_nal[3]); // level_indication
    out.push(0xFF); // reserved 111111 | lengthSizeMinusOne=3
    out.push(0xE1); // reserved 111 | numSPS=1
    out.extend_from_slice(&(sps_nal.len() as u16).to_be_bytes());
    out.extend_from_slice(sps_nal);
    out.push(1); // numPPS
    out.extend_from_slice(&(pps_nal.len() as u16).to_be_bytes());
    out.extend_from_slice(pps_nal);
    out
}

// ---------------------------------------------------------------------------
// Encoder trait impl.
// ---------------------------------------------------------------------------

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let w = params
        .width
        .ok_or_else(|| Error::invalid("h264 encoder: CodecParameters.width is required"))?;
    let h = params
        .height
        .ok_or_else(|| Error::invalid("h264 encoder: CodecParameters.height is required"))?;
    let opts = H264EncoderOptions::default();
    let enc = H264Encoder::new(params.codec_id.clone(), w, h, opts)?;
    Ok(Box::new(enc))
}

impl Encoder for H264Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Video(v) => self.encode_frame(v),
            _ => Err(Error::invalid("h264 encoder: expected a video frame")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.packets.pop_front() {
            return Ok(p);
        }
        if self.eof {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

// Verify the mb_type packing at compile time so the encoder's fixed choice
// lines up with the decoder's table.
const _: () = {
    // mb_type 21 in i_mb_type_info (1..=24 range) corresponds to entry
    // (pred=2 [DC], cbp_chroma=2, cbp_luma=15). That's what we want.
    // We can't easily assert this in const eval since the decoder's table
    // is embedded in a match, but the layout here is documented in
    // `mb_type.rs`. ME_INTRA_4_2_0 table is used for I_NxN CBP — which we
    // never emit (I_NxN is not used by this encoder), so it's only referenced
    // here to keep the import wired.
    let _ = ME_INTRA_4_2_0[0];
};
