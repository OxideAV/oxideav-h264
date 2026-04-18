//! H.264 **Baseline Profile, I-frames only** encoder.
//!
//! Minimum viable encoder that can be driven through the
//! [`oxideav_codec::Encoder`] trait and whose output is decodable by this
//! crate's [`crate::decoder::H264Decoder`] (and by ffmpeg, subject to fixture
//! availability).
//!
//! # Advertised scope
//!
//! * Baseline Profile (profile_idc = 66), level 3.0.
//! * I-frames only — every output frame is emitted as an IDR NAL.
//! * CAVLC entropy coding (entropy_coding_mode_flag = 0).
//! * Single slice per picture (`num_slice_groups_minus1 = 0`).
//! * 4:2:0 chroma, 8-bit luma / chroma, single colour plane.
//! * Intra_16×16 DC_PRED for every luma macroblock; chroma DC for chroma.
//! * Fixed QP (configurable, default = 26). No rate control, no adaptive QP.
//! * Deblocking disabled on emit (`disable_deblocking_filter_idc = 1`).
//! * Annex B framing with 4-byte start codes.
//!
//! # Packet layout
//!
//! Every `receive_packet()` returns a single self-contained packet shaped
//! as:
//!
//! ```text
//! [start code] [SPS NAL]
//! [start code] [PPS NAL]
//! [start code] [IDR slice NAL]
//! ```
//!
//! Every packet is a keyframe (`PacketFlags::keyframe = true`).
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

use crate::bitwriter::{rbsp_to_ebsp, BitWriter};
use crate::cavlc::BlockKind;
use crate::cavlc_enc::encode_residual_block;
use crate::fwd_transform::{
    forward_dct_4x4, forward_hadamard_2x2, forward_hadamard_4x4, quantize_4x4_ac,
    quantize_chroma_dc_2x2, quantize_luma_dc_4x4,
};
use crate::intra_pred::{
    predict_intra_16x16, predict_intra_chroma, Intra16x16Mode, Intra16x16Neighbours,
    IntraChromaMode, IntraChromaNeighbours,
};
use crate::mb::LUMA_BLOCK_RASTER;
use crate::tables::ME_INTRA_4_2_0;
use crate::transform::{
    chroma_qp, dequantize_4x4, idct_4x4, inv_hadamard_2x2_chroma_dc, inv_hadamard_4x4_dc,
};

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
}

impl Default for H264EncoderOptions {
    fn default() -> Self {
        Self {
            qp: 26,
            sps_id: 0,
            pps_id: 0,
        }
    }
}

/// Baseline I-only H.264 encoder.
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
    /// Monotonic IDR frame number.
    frame_num: u32,
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
        let sps_nal = build_sps_nal(width, height, opts.sps_id)?;
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
        let mut rec_cb = vec![0u8; (coded_cw * coded_ch) as usize];
        let mut rec_cr = vec![0u8; (coded_cw * coded_ch) as usize];
        let l_stride = self.coded_width as usize;
        let c_stride = coded_cw;

        // Build IDR slice RBSP.
        let mut slice_rbsp = BitWriter::new();
        self.write_slice_header(&mut slice_rbsp);
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
                (rec_cb.as_mut(), &pred_cb, &cb_ac, &cb_dc_rec)
            } else {
                (rec_cr.as_mut(), &pred_cr, &cr_ac, &cr_dc_rec)
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
fn build_sps_nal(width: u32, height: u32, sps_id: u32) -> Result<Vec<u8>> {
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
    // frame_mbs_only_flag = 1; direct_8x8_inference_flag = 1.
    w.write_flag(true);
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
