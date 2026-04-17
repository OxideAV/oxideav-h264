//! Front-end H.264 decoder.
//!
//! * Detects whether incoming packet bytes are Annex B (start-code) or AVCC
//!   (length-prefixed) form. AVCC mode is selected when the codec parameters
//!   carry a non-empty `extradata` matching an
//!   AVCDecoderConfigurationRecord; otherwise Annex B framing is assumed and
//!   start codes are scanned.
//! * Walks NAL units and updates SPS / PPS tables.
//! * Parses slice headers and runs the pixel reconstruction pipeline for:
//!   - I-slices in CAVLC or CABAC entropy mode (§7.3.5 / §8.3 / §8.5).
//!   - P-slices in CAVLC entropy mode (§8.4 motion compensation) —
//!     P16×16 / P16×8 / P8×16 / P8×8 / P8x8ref0 / P_Skip with a
//!     single-reference list 0. Intra-in-P macroblocks reuse the I-slice
//!     intra path.
//!
//! Out of scope (returns `Error::Unsupported`):
//! * CABAC P-slices.
//! * B-slices (bi-prediction, weighted prediction).
//! * Multi-reference DPB / ref_idx > 0 / reference picture list modification.
//! * Interlaced coding / MBAFF.
//! * 8×8 transform, 4:2:2 / 4:4:4 chroma, bit depth > 8.

use std::collections::{HashMap, VecDeque};

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational, Result, TimeBase,
    VideoFrame,
};

use crate::bitreader::BitReader;
use crate::cabac::{engine::CabacDecoder, mb::decode_i_mb_cabac, tables::init_slice_contexts};
use crate::mb::decode_i_slice_data;
use crate::p_mb::{decode_p_slice_mb, decode_p_skip_mb};
use crate::nal::{
    extract_rbsp, split_annex_b, split_length_prefixed, AvcConfig, NalHeader, NalUnitType,
};
use crate::picture::Picture;
use crate::pps::{parse_pps, Pps};
use crate::slice::{parse_slice_header, SliceHeader, SliceType};
use crate::sps::{parse_sps, Sps};

/// Build a decoder from `params`.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let mut dec = H264Decoder::new(params.codec_id.clone());
    if !params.extradata.is_empty() {
        dec.set_avc_config(&params.extradata)?;
    }
    Ok(Box::new(dec))
}

/// Front-end H.264 decoder. Holds parsed parameter sets keyed by id.
pub struct H264Decoder {
    codec_id: CodecId,
    /// `Some(length_size)` if the wire format is AVCC (length-prefixed).
    /// `None` for Annex B byte-stream form.
    length_size: Option<u8>,
    sps_by_id: HashMap<u32, Sps>,
    pps_by_id: HashMap<u32, Pps>,
    last_avc_config: Option<AvcConfig>,
    pending_pts: Option<i64>,
    pending_tb: TimeBase,
    ready_frames: VecDeque<VideoFrame>,
    eof: bool,
    /// Slice headers from the last packet, for diagnostics.
    last_slice_headers: Vec<SliceHeader>,
    /// Picture under construction — held across slices of a single frame.
    current_pic: Option<Picture>,
    /// The most recently decoded picture, retained as the single L0
    /// reference for subsequent P-slices. Cleared on IDR / reset.
    last_reference: Option<Picture>,
}

impl H264Decoder {
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            length_size: None,
            sps_by_id: HashMap::new(),
            pps_by_id: HashMap::new(),
            last_avc_config: None,
            pending_pts: None,
            pending_tb: TimeBase::new(1, 90_000),
            ready_frames: VecDeque::new(),
            eof: false,
            last_slice_headers: Vec::new(),
            current_pic: None,
            last_reference: None,
        }
    }

    /// Feed an `avcC` (`AVCDecoderConfigurationRecord`) blob. Sets the
    /// length-prefix size and pre-loads SPS/PPS.
    pub fn set_avc_config(&mut self, avcc: &[u8]) -> Result<()> {
        let cfg = AvcConfig::parse(avcc)?;
        self.length_size = Some(cfg.length_size);
        for sps_nalu in &cfg.sps {
            self.ingest_nalu(sps_nalu)?;
        }
        for pps_nalu in &cfg.pps {
            self.ingest_nalu(pps_nalu)?;
        }
        self.last_avc_config = Some(cfg);
        Ok(())
    }

    /// Walk a single NAL unit (header byte + payload) and update state.
    fn ingest_nalu(&mut self, nalu: &[u8]) -> Result<()> {
        if nalu.is_empty() {
            return Ok(());
        }
        let header = NalHeader::parse(nalu[0])?;
        let rbsp = extract_rbsp(&nalu[1..]);
        match header.nal_unit_type {
            NalUnitType::Sps => {
                let sps = parse_sps(&header, &rbsp)?;
                self.sps_by_id.insert(sps.seq_parameter_set_id, sps);
            }
            NalUnitType::Pps => {
                let pps = parse_pps(&header, &rbsp, None)?;
                self.pps_by_id.insert(pps.pic_parameter_set_id, pps);
            }
            t if t.is_slice() => {
                let pps_id = peek_pps_id(&rbsp)?;
                let pps = self.pps_by_id.get(&pps_id).cloned().ok_or_else(|| {
                    Error::invalid(format!(
                        "h264 slice: references unknown PPS id {pps_id} (need PPS NAL first)"
                    ))
                })?;
                let sps = self
                    .sps_by_id
                    .get(&pps.seq_parameter_set_id)
                    .cloned()
                    .ok_or_else(|| {
                        Error::invalid(format!(
                            "h264 slice: PPS {pps_id} references unknown SPS {}",
                            pps.seq_parameter_set_id
                        ))
                    })?;
                let sh = parse_slice_header(&header, &rbsp, &sps, &pps)?;
                self.last_slice_headers.push(sh.clone());
                match sh.slice_type {
                    SliceType::I => {}
                    SliceType::P => {
                        if pps.entropy_coding_mode_flag {
                            return Err(Error::unsupported(
                                "h264: CABAC P-slices not implemented — use CAVLC (entropy_coding_mode_flag=0) for P",
                            ));
                        }
                    }
                    other => {
                        return Err(Error::unsupported(format!(
                            "h264: slice type {:?} not implemented (only I + CAVLC P supported)",
                            other
                        )));
                    }
                }
                if !sps.frame_mbs_only_flag {
                    return Err(Error::unsupported(
                        "h264: interlaced / MBAFF (§7.3.2.1.1 frame_mbs_only_flag=0) not supported",
                    ));
                }
                // Macroblock-layer decode (§7.3.5 / §8.3 / §8.4 / §8.5 / §9.2).
                let mb_w = sps.pic_width_in_mbs();
                let mb_h = sps.pic_height_in_map_units();
                let mut pic = self
                    .current_pic
                    .take()
                    .unwrap_or_else(|| Picture::new(mb_w, mb_h));
                if pic.mb_width != mb_w || pic.mb_height != mb_h {
                    pic = Picture::new(mb_w, mb_h);
                }

                if sh.is_idr {
                    // IDR — wipe any previously stored reference so no
                    // future P-slice accidentally predicts from a stale
                    // picture.
                    self.last_reference = None;
                }

                match sh.slice_type {
                    SliceType::I => {
                        if pps.entropy_coding_mode_flag {
                            decode_cabac_i_slice(&rbsp, &sh, &sps, &pps, &mut pic)?;
                        } else {
                            let mut br = BitReader::new(&rbsp);
                            br.skip(sh.slice_data_bit_offset as u32)?;
                            decode_i_slice_data(&mut br, &sh, &sps, &pps, &mut pic)?;
                        }
                    }
                    SliceType::P => {
                        let reference = self.last_reference.as_ref().ok_or_else(|| {
                            Error::invalid(
                                "h264 p-slice: no reference picture available (no prior IDR?)",
                            )
                        })?;
                        if reference.mb_width != mb_w || reference.mb_height != mb_h {
                            return Err(Error::invalid(
                                "h264 p-slice: reference picture size mismatch",
                            ));
                        }
                        let mut br = BitReader::new(&rbsp);
                        br.skip(sh.slice_data_bit_offset as u32)?;
                        decode_p_slice_data(&mut br, &sh, &sps, &pps, &mut pic, reference)?;
                    }
                    _ => unreachable!(),
                }

                // Optional in-loop deblocking — §8.7.
                if sh.disable_deblocking_filter_idc != 1 {
                    crate::deblock::deblock_picture(&mut pic, &pps, &sh);
                }

                // Crop and emit a frame, then stash the decoded picture as
                // the single L0 reference for subsequent P-slices.
                let (vw, vh) = sps.visible_size();
                let frame = pic.to_video_frame(vw, vh, self.pending_pts, self.pending_tb);
                self.ready_frames.push_back(frame);
                self.last_reference = Some(pic);
            }
            NalUnitType::Aud
            | NalUnitType::Sei
            | NalUnitType::EndOfSeq
            | NalUnitType::EndOfStream
            | NalUnitType::Filler => {
                // Ignored in this version.
            }
            _ => {
                // Unknown / unspecified: ignore.
            }
        }
        Ok(())
    }

    /// Look up the active SPS for a given id (for tests / diagnostics).
    pub fn sps(&self, id: u32) -> Option<&Sps> {
        self.sps_by_id.get(&id)
    }

    /// Look up the active PPS for a given id.
    pub fn pps(&self, id: u32) -> Option<&Pps> {
        self.pps_by_id.get(&id)
    }

    /// Slice headers parsed from the most recent batch (diagnostics).
    pub fn last_slice_headers(&self) -> &[SliceHeader] {
        &self.last_slice_headers
    }
}

/// CAVLC P-slice data loop — §7.3.4.
///
/// CAVLC P-slice syntax emits one `mb_skip_run` ue(v) before each coded
/// macroblock (including the very first MB of the slice). The value is the
/// number of `P_Skip` macroblocks that precede the next coded MB. The last
/// "run" before the end of the slice is also emitted, letting the encoder
/// trail a run of skipped MBs with no following coded MB.
fn decode_p_slice_data(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
    reference: &Picture,
) -> Result<()> {
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = sps.pic_height_in_map_units();
    let total_mbs = mb_w * mb_h;
    let mut mb_addr = sh.first_mb_in_slice;
    if mb_addr >= total_mbs {
        return Err(Error::invalid("h264 p-slice: first_mb_in_slice out of range"));
    }
    let mut prev_qp = (pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta).clamp(0, 51);

    while mb_addr < total_mbs {
        let skip_run = br.read_ue()?;
        // Emit `skip_run` skipped MBs then one coded MB (if any remain).
        for _ in 0..skip_run {
            if mb_addr >= total_mbs {
                break;
            }
            let mb_x = mb_addr % mb_w;
            let mb_y = mb_addr / mb_w;
            decode_p_skip_mb(mb_x, mb_y, pic, reference, prev_qp)?;
            mb_addr += 1;
        }
        if mb_addr >= total_mbs {
            break;
        }
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        decode_p_slice_mb(br, sps, pps, sh, mb_x, mb_y, pic, reference, &mut prev_qp)?;
        mb_addr += 1;
    }
    Ok(())
}

/// §9.3 CABAC entropy-coded I-slice data loop. Mirrors
/// [`crate::mb::decode_i_slice_data`] but feeds every macroblock through
/// [`crate::cabac::mb::decode_i_mb_cabac`] and relies on
/// [`CabacDecoder::decode_terminate`] for the `end_of_slice_flag`.
fn decode_cabac_i_slice(
    rbsp: &[u8],
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
) -> Result<()> {
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = sps.pic_height_in_map_units();
    let total_mbs = mb_w * mb_h;
    let slice_qpy = (pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta).clamp(0, 51);
    let mut ctxs = init_slice_contexts(sh.cabac_init_idc, true, slice_qpy);
    let mut dec = CabacDecoder::new(rbsp, sh.slice_data_bit_offset as usize)?;
    let mut prev_qp = slice_qpy;
    pic.last_mb_qp_delta_was_nonzero = false;
    let mut mb_addr = sh.first_mb_in_slice;
    if mb_addr >= total_mbs {
        return Err(Error::invalid(
            "h264 cabac slice: first_mb_in_slice out of range",
        ));
    }
    loop {
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        decode_i_mb_cabac(
            &mut dec,
            &mut ctxs,
            sh,
            sps,
            pps,
            mb_x,
            mb_y,
            pic,
            &mut prev_qp,
        )?;
        mb_addr += 1;
        // §9.3.3.2.4 end_of_slice_flag — one terminate bin after each MB.
        let end = dec.decode_terminate()?;
        if end == 1 {
            break;
        }
        if mb_addr >= total_mbs {
            return Err(Error::invalid(
                "h264 cabac slice: end_of_slice_flag did not fire before last MB",
            ));
        }
    }
    Ok(())
}

/// Peek the `pic_parameter_set_id` from a slice RBSP without consuming.
fn peek_pps_id(rbsp: &[u8]) -> Result<u32> {
    let mut br = crate::bitreader::BitReader::new(rbsp);
    let _first_mb_in_slice = br.read_ue()?;
    let _slice_type_raw = br.read_ue()?;
    let pps_id = br.read_ue()?;
    Ok(pps_id)
}

impl Decoder for H264Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.pending_pts = packet.pts;
        self.pending_tb = packet.time_base;
        self.last_slice_headers.clear();
        let nalus: Vec<&[u8]> = match self.length_size {
            Some(ls) => split_length_prefixed(&packet.data, ls)?,
            None => split_annex_b(&packet.data),
        };
        for nalu in nalus {
            self.ingest_nalu(nalu)?;
        }
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.ready_frames.pop_front() {
            return Ok(Frame::Video(f));
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

    fn reset(&mut self) -> Result<()> {
        // Drop the in-flight picture, pending slice-header cache, and
        // ready frame queue. SPS/PPS tables + AVCC config (length_size,
        // last_avc_config) are stream-level — they come from extradata or
        // out-of-band SPS/PPS NALs that won't be retransmitted post-seek,
        // so they stay put. The L0 reference picture is dropped too so
        // the next P-slice sees a clean slate (it must be preceded by an
        // IDR or another I-slice).
        self.pending_pts = None;
        self.ready_frames.clear();
        self.eof = false;
        self.last_slice_headers.clear();
        self.current_pic = None;
        self.last_reference = None;
        Ok(())
    }
}

/// Build a `CodecParameters` from a parsed SPS — useful when wiring a
/// container that carries Annex B data.
pub fn codec_parameters_from_sps(sps: &Sps) -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
    let (w, h) = sps.visible_size();
    params.width = Some(w);
    params.height = Some(h);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    // Frame rate isn't known from SPS without VUI timing. Leave unset.
    let _ = Rational::new(0, 1);
    params
}

/// Surface the frame the decoder *would* emit for an I-slice, once the
/// pixel pipeline lands. Returns a black `VideoFrame` of the right size.
/// Useful for downstream callers who want to test sizing without driving the
/// full decode loop.
pub fn placeholder_frame(sps: &Sps, pts: Option<i64>, tb: TimeBase) -> VideoFrame {
    use oxideav_core::frame::VideoPlane;
    let (w, h) = sps.visible_size();
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let y = vec![0u8; (w * h) as usize];
    let cb = vec![128u8; (cw * ch) as usize];
    let cr = vec![128u8; (cw * ch) as usize];
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width: w,
        height: h,
        pts,
        time_base: tb,
        planes: vec![
            VideoPlane {
                stride: w as usize,
                data: y,
            },
            VideoPlane {
                stride: cw as usize,
                data: cb,
            },
            VideoPlane {
                stride: cw as usize,
                data: cr,
            },
        ],
    }
}
