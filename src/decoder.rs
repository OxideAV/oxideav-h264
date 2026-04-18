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
//!     P16×16 / P16×8 / P8×16 / P8×8 / P8x8ref0 / P_Skip. Intra-in-P
//!     macroblocks reuse the I-slice intra path. Explicit weighted
//!     prediction (§8.4.2.3.2) is applied to MC output when the PPS +
//!     slice header enable it.
//!   - B-slices in CAVLC entropy mode with both spatial direct
//!     (§8.4.1.2.2) and temporal direct (§8.4.1.2.3) MV derivation,
//!     default + explicit weighted bi-prediction, and all Table 7-14 /
//!     7-17 non-MBAFF partition shapes. The same B-slice pixel pipeline
//!     is driven from CABAC via [`crate::cabac::b_mb`].
//! * Owns a [`crate::dpb::Dpb`] holding up to `sps.max_num_ref_frames`
//!   reconstructed reference frames with POC ordering, sliding-window
//!   marking (§8.2.5.3), and MMCO operations 1 / 2 / 3 / 4 / 5 / 6
//!   (§8.2.5.4). Each P-slice sees a freshly-built `RefPicList0`
//!   (§8.2.4.2.1); each B-slice sees both `RefPicList0` and
//!   `RefPicList1` (§8.2.4.2.3). The DPB's POC-ordered reorder queue
//!   is activated (size = `max_num_ref_frames`) on the first B-slice
//!   so frames emerge in presentation order.
//!
//! Out of scope (returns `Error::Unsupported`):
//! * Interlaced coding / MBAFF (P and B).
//! * 8×8 transform on the CABAC path, bit depth > 8.
//! * Monochrome (`chroma_format_idc = 0`) and
//!   `separate_colour_plane_flag = 1` are rejected at slice entry
//!   (§7.4.2.1.1 / §6.4.1). 4:4:4 (`chroma_format_idc = 3`) is wired
//!   for CAVLC I-slices only; 4:2:2 (`chroma_format_idc = 2`) is
//!   wired for CAVLC I/P/B and CABAC I/P slices. CABAC 4:2:2 B-slices
//!   and 4:4:4 CABAC still return `Error::Unsupported`.
//!   See [`crate::mb_444`] for the 4:4:4 CAVLC
//!   path, [`crate::mb::decode_chroma_422`] (private) for the 4:2:2
//!   chroma pipeline, and [`crate::picture::chroma_plane_w`] /
//!   [`crate::picture::chroma_plane_h`] for the format-aware
//!   sample-layout helpers.
//!
//! Reference picture list modification (RPLM, §7.3.3.1 / §8.2.4.3) is
//! parsed into [`crate::slice::SliceHeader::rplm_l0`] /
//! [`crate::slice::SliceHeader::rplm_l1`] and applied via
//! [`crate::dpb::apply_rplm`] on top of the default-built lists.

use std::collections::{HashMap, VecDeque};

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational, Result, TimeBase,
    VideoFrame,
};

use crate::b_mb::{decode_b_skip_mb, decode_b_slice_mb};
use crate::bitreader::BitReader;
use crate::cabac::{
    b_mb::decode_b_mb_cabac, b_mb_hi::decode_b_mb_cabac_hi, engine::CabacDecoder,
    mb::decode_i_mb_cabac, mb_444::decode_i_mb_cabac_444, mb_hi::decode_i_mb_cabac_hi,
    p_mb::decode_p_mb_cabac, p_mb_444::{decode_b_mb_cabac_444, decode_p_mb_cabac_444},
    p_mb_hi::decode_p_mb_cabac_hi, tables::init_slice_contexts,
};
use crate::dpb::{
    apply_rplm, derive_poc_type0, derive_poc_type1, derive_poc_type2, Dpb, MmcoCommand, PocState,
    PocType1Params,
};
use crate::mb::decode_i_slice_data;
use crate::nal::{
    extract_rbsp, split_annex_b, split_length_prefixed, AvcConfig, NalHeader, NalUnitType,
};
use crate::p_mb::{decode_p_skip_mb, decode_p_slice_mb};
use crate::p_mb_444::{decode_p_skip_mb_444, decode_p_slice_mb_444};
use crate::b_mb_444::{decode_b_skip_mb_444, decode_b_slice_mb_444};
use crate::p_mb_hi::{decode_p_skip_mb_hi, decode_p_slice_mb_hi};
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
    /// Decoded Picture Buffer — `sps.max_num_ref_frames` reference slots
    /// plus a small reorder queue. Lazily initialised on the first slice.
    dpb: Option<Dpb>,
    /// Running POC state for `pic_order_cnt_type == 0` (§8.2.1.1).
    poc_state: PocState,
    /// Running state for `pic_order_cnt_type == 2` (§8.2.1.3).
    prev_frame_num: u32,
    prev_frame_num_offset: i32,
    /// `frame_num` of the most recently decoded **reference** picture —
    /// drives `FrameNumWrap` for DPB short-term entries (§8.2.4.1).
    prev_ref_frame_num: u32,
    /// FIFO of (pts, timebase) pairs aligned with the decode-order
    /// pictures being queued for POC-reordered output. Drained as the
    /// DPB releases frames.
    pts_fifo: std::collections::VecDeque<(Option<i64>, TimeBase)>,
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
            dpb: None,
            poc_state: PocState::default(),
            prev_frame_num: 0,
            prev_frame_num_offset: 0,
            prev_ref_frame_num: 0,
            pts_fifo: VecDeque::new(),
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
                        // CABAC P-slice support shipped in this crate; the
                        // dispatch below routes to the CABAC data loop when
                        // `entropy_coding_mode_flag = 1`.
                    }
                    SliceType::B => {
                        // CABAC B-slices are wired below through
                        // `decode_cabac_b_slice`; the old early-reject stays
                        // removed so both entropy modes share the same path.
                    }
                    other => {
                        return Err(Error::unsupported(format!(
                            "h264: slice type {:?} not implemented",
                            other
                        )));
                    }
                }
                // §7.3.2.1.1 / §7.3.3: `frame_mbs_only_flag = 0` allows
                // three interlaced modes — MBAFF frame coding (slice has
                // `field_pic_flag = 0` and the SPS has
                // `mb_adaptive_frame_field_flag = 1`), PAFF field coding
                // (slice has `field_pic_flag = 1`), and plain field-mode
                // frames. This decoder wires PAFF I-slice CAVLC and MBAFF
                // I-slice CAVLC MVPs for 4:2:0 / 8-bit; other slice types /
                // entropy modes continue to return `Error::Unsupported`.
                if !sps.frame_mbs_only_flag {
                    if !sh.field_pic_flag {
                        // MBAFF frame coding path.
                        if !sps.mb_adaptive_frame_field_flag {
                            return Err(Error::unsupported(
                                "h264: frame-mode slice under frame_mbs_only_flag=0 without MBAFF/PAFF not supported",
                            ));
                        }
                        let mbaff_ok = sh.slice_type == SliceType::I
                            && !pps.entropy_coding_mode_flag
                            && (sps.chroma_format_idc == 1
                                || sps.chroma_format_idc == 2
                                || sps.chroma_format_idc == 3)
                            && sps.bit_depth_luma_minus8 == 0
                            && sps.bit_depth_chroma_minus8 == 0
                            && !sps.separate_colour_plane_flag;
                        if !mbaff_ok {
                            return Err(Error::unsupported(
                                "h264: only MBAFF I-slice CAVLC 4:2:0/4:2:2/4:4:4 8-bit interlaced is wired (§7.3.4); \
                                 MBAFF+P/B/CABAC/10-bit still reject",
                            ));
                        }
                    } else {
                        // PAFF field slice: only CAVLC I-slices in 4:2:0 / 8-bit
                        // land on the field-reconstruction path below.
                        if sh.slice_type != SliceType::I {
                            return Err(Error::unsupported(
                                "h264: PAFF (field_pic_flag=1) supports I-slices only",
                            ));
                        }
                        if pps.entropy_coding_mode_flag {
                            return Err(Error::unsupported(
                                "h264: PAFF CABAC entropy decode not wired — CAVLC I-slice only",
                            ));
                        }
                        if sps.chroma_format_idc != 1
                            || sps.bit_depth_luma_minus8 != 0
                            || sps.bit_depth_chroma_minus8 != 0
                        {
                            return Err(Error::unsupported(
                                "h264: PAFF only wired for 4:2:0 / 8-bit (no 4:2:2 / 4:4:4 / 10-bit)",
                            ));
                        }
                        if sps.mb_adaptive_frame_field_flag {
                            return Err(Error::unsupported(
                                "h264: PAFF + MBAFF mixed within the same CVS not supported",
                            ));
                        }
                    }
                }
                // §7.4.2.1.1 — chroma format. The MB-layer pipeline below
                // (intra prediction, inverse transforms, motion
                // compensation, deblocking, CAVLC/CABAC residual) is
                // wired for `chroma_format_idc = 1` (4:2:0) only; the
                // chroma plane dimensions, DC transform size, per-MB
                // block count, intra-mode set, sub-pel filter taps and
                // deblock internal-edge layout all differ for
                // `chroma_format_idc ∈ {2, 3}` and for
                // `separate_colour_plane_flag = 1`. Fail fast so
                // upstream callers can route to a C-free fallback
                // (rather than produce silently-corrupt YUV).
                if sps.separate_colour_plane_flag {
                    return Err(Error::unsupported(
                        "h264: separate_colour_plane_flag=1 (§7.4.2.1.1 three independent colour planes) not supported",
                    ));
                }
                match sps.chroma_format_idc {
                    0 => {
                        return Err(Error::unsupported(
                            "h264: chroma_format_idc=0 (monochrome, §6.4.1) not supported",
                        ));
                    }
                    1 => {}
                    2 => {
                        // 4:2:2 — CAVLC I/P/B and CABAC I/P slices wired.
                        // Chroma plane is w/2 × h (half-width, full-height);
                        // chroma DC uses a 2×4 Hadamard (§8.5.11.2), 8 AC
                        // blocks per plane, and intra prediction runs over
                        // 8×16 tiles (§8.3.4). CABAC B-slices under 4:2:2
                        // still route to Error::Unsupported; chroma MC y is
                        // unscaled (§8.4.2.2.1 ChromaArrayType == 2) in
                        // the P/B paths (vertical MV divided by 4, not 8).
                        if matches!(sh.slice_type, SliceType::SP | SliceType::SI) {
                            return Err(Error::unsupported(
                                "h264: 4:2:2 (chroma_format_idc=2) supports I/P/B slices only",
                            ));
                        }
                        if pps.entropy_coding_mode_flag
                            && matches!(sh.slice_type, SliceType::B)
                        {
                            return Err(Error::unsupported(
                                "h264: 4:2:2 CABAC B-slice not yet wired — CABAC I/P + CAVLC I/P/B only",
                            ));
                        }
                    }
                    3 => {
                        // 4:4:4 — CAVLC I/P/B land on the ChromaArrayType=3
                        // pipeline (chroma planes the same size as luma,
                        // luma-style residual / intra prediction per plane,
                        // luma 6-tap MC filter on chroma per §8.4.2.2
                        // Table 8-9).
                        //
                        // CABAC 4:4:4 is gated until the spec's extended
                        // ctxBlockCat banks (6..=13 in §9.3.3.1.1.9
                        // Table 9-42, ctxIdxOffsets 1012+) are wired —
                        // sharing the luma cat=0/1/2/5 banks across all
                        // three planes diverges from the encoder's
                        // probability state after the first MB.
                        // `crate::cabac::mb_444` holds the entry-point
                        // plumbing (mb_type + per-plane intra prediction +
                        // per-plane residual dispatch) for the follow-up
                        // that lights those banks up.
                        if pps.entropy_coding_mode_flag {
                            return Err(Error::unsupported(
                                "h264: 4:4:4 CABAC entropy decode not yet wired — \
                                 needs §9.3.3.1.1.9 Table 9-42 extended Cb/Cr \
                                 ctxBlockCat banks (CAVLC I/P/B supported)",
                            ));
                        }
                    }
                    v => {
                        return Err(Error::unsupported(format!(
                            "h264: chroma_format_idc={v} invalid (§7.4.2.1.1) / not supported"
                        )));
                    }
                }
                // §7.4.2.1.1 bit depth — `bit_depth_luma_minus8 ∈ {0,2,4,6}`
                // map to 8 / 10 / 12 / 14 bits. The spec also defines odd
                // values (9, 11, 13 bits) via `bit_depth_luma_minus8 ∈
                // {1, 3, 5}` but no profile permits those in practice and
                // none of the conformance fixtures exercise them.
                match sps.bit_depth_luma_minus8 {
                    0 | 2 | 4 | 6 => {}
                    v => {
                        return Err(Error::unsupported(format!(
                            "h264: bit_depth_luma_minus8={v} not supported \
                             (only 0/2/4/6 = 8/10/12/14-bit are wired)"
                        )));
                    }
                }
                let high_bit_depth =
                    sps.bit_depth_luma_minus8 > 0 || sps.bit_depth_chroma_minus8 > 0;
                if high_bit_depth {
                    if sps.bit_depth_luma_minus8 != sps.bit_depth_chroma_minus8 {
                        return Err(Error::unsupported(
                            "h264: bit_depth_luma != bit_depth_chroma not supported",
                        ));
                    }
                    if sps.chroma_format_idc != 1 {
                        return Err(Error::unsupported(
                            "h264: high-bit-depth decode only wired for 4:2:0 (chroma_format_idc=1)",
                        ));
                    }
                    match sh.slice_type {
                        SliceType::I | SliceType::P | SliceType::B => {}
                        _ => {
                            return Err(Error::unsupported(
                                "h264: high-bit-depth decode wired for I/P/B slices only",
                            ));
                        }
                    }
                    // 10-bit: CAVLC I/P/B + CABAC I/P/B are wired.
                    // 12/14-bit: CAVLC I only — P/B and CABAC return
                    // Error::Unsupported while the rest of the pipeline
                    // (motion comp u16 filters, CABAC ctx clamps, etc.)
                    // is audited for higher `QpBdOffset`.
                    if sps.bit_depth_luma_minus8 > 2 {
                        if pps.entropy_coding_mode_flag {
                            return Err(Error::unsupported(
                                "h264: 12/14-bit CABAC not yet wired — CAVLC I only",
                            ));
                        }
                        if sh.slice_type != SliceType::I {
                            return Err(Error::unsupported(
                                "h264: 12/14-bit CAVLC wired for I-slices only (P/B not yet audited)",
                            ));
                        }
                    }
                }

                // Macroblock-layer decode (§7.3.5 / §8.3 / §8.4 / §8.5 / §9.2).
                // §7.4.3 — MB grid height depends on frame/field mode:
                //   frame_mbs_only_flag = 1: PicHeightInMapUnits (progressive)
                //   MBAFF frame-mode (field_pic_flag = 0): 2 × map_units
                //     (every MB, top + bottom of each MBAFF pair, gets a slot)
                //   PAFF field picture (field_pic_flag = 1): PicHeightInMapUnits
                //     (one field = half-height, already the map-unit count)
                let mb_w = sps.pic_width_in_mbs();
                let mb_h = if sh.field_pic_flag {
                    sps.pic_height_in_map_units()
                } else {
                    sps.pic_height_in_mbs()
                };
                let bit_depth_y = (sps.bit_depth_luma_minus8 + 8) as u8;
                let bit_depth_c = (sps.bit_depth_chroma_minus8 + 8) as u8;
                let make_pic = || -> Picture {
                    if sh.field_pic_flag {
                        Picture::new_field(mb_w, mb_h, sps.chroma_format_idc, sh.bottom_field_flag)
                    } else {
                        Picture::new_with_bit_depth(
                            mb_w,
                            mb_h,
                            sps.chroma_format_idc,
                            bit_depth_y,
                            bit_depth_c,
                        )
                    }
                };
                let mut pic = self.current_pic.take().unwrap_or_else(make_pic);
                if pic.mb_width != mb_w
                    || pic.mb_height != mb_h
                    || pic.chroma_format_idc != sps.chroma_format_idc
                    || pic.bit_depth_y != bit_depth_y
                    || pic.bit_depth_c != bit_depth_c
                    || pic.field_pic != sh.field_pic_flag
                    || (sh.field_pic_flag && pic.bottom_field != sh.bottom_field_flag)
                {
                    pic = make_pic();
                }
                pic.mbaff_enabled =
                    !sps.frame_mbs_only_flag && sps.mb_adaptive_frame_field_flag;
                // §7.4.2.2 Table 7-2 — resolve the active scaling
                // matrices (SPS + PPS, with fallback). Stored on the
                // picture so every MB-level dequant call can index
                // the per-category list without re-doing the fallback.
                pic.scaling_lists = crate::scaling_list::ScalingLists::resolve(&sps, &pps);

                // Lazily (re)create the DPB when we first see an SPS —
                // its size depends on `max_num_ref_frames`. Size the
                // output-reorder window from the VUI's
                // `max_num_reorder_frames` (§E.2.1) when present — it is
                // the spec-canonical bound. Falls back to
                // `max_num_ref_frames` (over-conservative) otherwise.
                // When no B-slice is in the stream the window collapses
                // to 0 so frames emerge immediately.
                let want_dpb_size = sps.max_num_ref_frames.max(1);
                let vui_reorder = sps.vui.as_ref().and_then(|v| v.max_num_reorder_frames);
                let want_reorder = if sh.slice_type == SliceType::B {
                    vui_reorder.unwrap_or(sps.max_num_ref_frames).max(1)
                } else {
                    0
                };
                match self.dpb.as_mut() {
                    Some(d) if d.max_num_ref_frames == want_dpb_size => {
                        // Bump the reorder window if a B-slice raises it.
                        d.set_reorder_window(d.reorder_window().max(want_reorder));
                    }
                    _ => {
                        self.dpb = Some(Dpb::new(want_dpb_size, want_reorder));
                    }
                }

                let max_frame_num = 1u32 << (sps.log2_max_frame_num_minus4 + 4);

                if sh.is_idr {
                    // IDR — drain the DPB's reorder queue and wipe references.
                    let out = if let Some(dpb) = self.dpb.as_mut() {
                        if !sh.idr_no_output_of_prior_pics_flag {
                            let out = dpb.flush_all();
                            dpb.clear();
                            out
                        } else {
                            dpb.clear();
                            Vec::new()
                        }
                    } else {
                        Vec::new()
                    };
                    for pic_out in out {
                        self.queue_ready_frame(&sps, pic_out);
                    }
                    self.poc_state = PocState::default();
                    self.prev_frame_num = 0;
                    self.prev_frame_num_offset = 0;
                    self.prev_ref_frame_num = 0;
                }

                // ---- POC derivation (§8.2.1) -------------------------
                let is_ref = header.nal_ref_idc != 0;
                let poc = match sps.pic_order_cnt_type {
                    0 => {
                        let (p, new_state) = derive_poc_type0(
                            sh.pic_order_cnt_lsb,
                            sps.log2_max_pic_order_cnt_lsb_minus4,
                            self.poc_state,
                            sh.is_idr,
                        );
                        if is_ref {
                            self.poc_state = new_state;
                        }
                        p
                    }
                    2 => {
                        let (p, off) = derive_poc_type2(
                            sh.frame_num,
                            self.prev_frame_num,
                            self.prev_frame_num_offset,
                            max_frame_num,
                            header.nal_ref_idc,
                            sh.is_idr,
                        );
                        if is_ref {
                            self.prev_frame_num_offset = off;
                        }
                        p
                    }
                    1 => {
                        let (p, off) = derive_poc_type1(PocType1Params {
                            delta_pic_order_always_zero_flag: sps.delta_pic_order_always_zero_flag,
                            offset_for_non_ref_pic: sps.offset_for_non_ref_pic,
                            offset_for_top_to_bottom_field: sps.offset_for_top_to_bottom_field,
                            num_ref_frames_in_pic_order_cnt_cycle: sps
                                .num_ref_frames_in_pic_order_cnt_cycle,
                            offset_for_ref_frame: &sps.offset_for_ref_frame,
                            max_frame_num,
                            nal_ref_idc: header.nal_ref_idc,
                            is_idr: sh.is_idr,
                            frame_num: sh.frame_num,
                            prev_frame_num: self.prev_frame_num,
                            prev_frame_num_offset: self.prev_frame_num_offset,
                            delta_pic_order_cnt: sh.delta_pic_order_cnt,
                        });
                        // §8.2.1.2 — every picture advances `prev_frame_num_offset`
                        // (unlike type 2, which only updates on reference
                        // pictures). This matches the spec's
                        // "FrameNumOffset for the next picture" clause.
                        self.prev_frame_num_offset = off;
                        p
                    }
                    other => {
                        return Err(Error::invalid(format!(
                            "h264: unknown pic_order_cnt_type {other}"
                        )));
                    }
                };

                match sh.slice_type {
                    SliceType::I => {
                        if pps.entropy_coding_mode_flag && high_bit_depth {
                            decode_cabac_i_slice_hi(&rbsp, &sh, &sps, &pps, &mut pic)?;
                        } else if pps.entropy_coding_mode_flag
                            && sps.chroma_format_idc == 3
                        {
                            decode_cabac_i_slice_444(&rbsp, &sh, &sps, &pps, &mut pic)?;
                        } else if pps.entropy_coding_mode_flag {
                            decode_cabac_i_slice(&rbsp, &sh, &sps, &pps, &mut pic)?;
                        } else if high_bit_depth {
                            let mut br = BitReader::new(&rbsp);
                            br.skip(sh.slice_data_bit_offset as u32)?;
                            crate::mb_hi::decode_i_slice_data_hi(
                                &mut br, &sh, &sps, &pps, &mut pic,
                            )?;
                        } else {
                            let mut br = BitReader::new(&rbsp);
                            br.skip(sh.slice_data_bit_offset as u32)?;
                            decode_i_slice_data(&mut br, &sh, &sps, &pps, &mut pic)?;
                        }
                    }
                    SliceType::P => {
                        let dpb = self.dpb.as_mut().expect("DPB initialised above");
                        dpb.update_frame_num_wrap(self.prev_ref_frame_num, max_frame_num);
                        // Re-borrow immutably to build the list + apply RPLM.
                        let dpb_ref: &Dpb = dpb;
                        let default_list = dpb_ref.build_list0_p(sh.num_ref_idx_l0_active_minus1);
                        let ref_entries = if sh.rplm_l0.is_empty() {
                            default_list
                        } else {
                            // §7.4.3 CurrPicNum == frame_num for non-field
                            // frames; MaxPicNum == MaxFrameNum.
                            let curr_pic_num = sh.frame_num as i32;
                            apply_rplm(
                                default_list,
                                &sh.rplm_l0,
                                curr_pic_num,
                                max_frame_num as i32,
                                sh.num_ref_idx_l0_active_minus1,
                                dpb_ref.frames_raw(),
                            )?
                        };
                        if ref_entries.is_empty() {
                            return Err(Error::invalid(
                                "h264 p-slice: no reference picture available (no prior IDR?)",
                            ));
                        }
                        for rf in &ref_entries {
                            if rf.pic.mb_width != mb_w || rf.pic.mb_height != mb_h {
                                return Err(Error::invalid(
                                    "h264 p-slice: reference picture size mismatch",
                                ));
                            }
                        }
                        let list0: Vec<&Picture> = ref_entries.iter().map(|r| &r.pic).collect();
                        let list0_pocs: Vec<i32> = ref_entries.iter().map(|r| r.poc).collect();
                        if pps.entropy_coding_mode_flag && high_bit_depth {
                            decode_cabac_p_slice_hi(&rbsp, &sh, &sps, &pps, &mut pic, &list0)?;
                        } else if pps.entropy_coding_mode_flag && sps.chroma_format_idc == 3 {
                            decode_cabac_p_slice_444(&rbsp, &sh, &sps, &pps, &mut pic, &list0)?;
                        } else if pps.entropy_coding_mode_flag {
                            decode_cabac_p_slice(&rbsp, &sh, &sps, &pps, &mut pic, &list0)?;
                        } else if high_bit_depth {
                            let mut br = BitReader::new(&rbsp);
                            br.skip(sh.slice_data_bit_offset as u32)?;
                            decode_p_slice_data_hi(&mut br, &sh, &sps, &pps, &mut pic, &list0)?;
                        } else if sps.chroma_format_idc == 3 {
                            let mut br = BitReader::new(&rbsp);
                            br.skip(sh.slice_data_bit_offset as u32)?;
                            decode_p_slice_data_444(&mut br, &sh, &sps, &pps, &mut pic, &list0)?;
                        } else {
                            let mut br = BitReader::new(&rbsp);
                            br.skip(sh.slice_data_bit_offset as u32)?;
                            decode_p_slice_data(&mut br, &sh, &sps, &pps, &mut pic, &list0)?;
                        }
                        // Snapshot POCs so a subsequent B-slice that picks
                        // this P-picture as its colocated L1[0] reference
                        // can resolve colocated `refIdxCol` → POC
                        // (§8.4.1.2.3 temporal direct). P-slices have no
                        // list1, so only list0 POCs are recorded.
                        pic.snapshot_ref_pocs(&list0_pocs, &[]);
                    }
                    SliceType::B => {
                        let dpb = self.dpb.as_mut().expect("DPB initialised above");
                        dpb.update_frame_num_wrap(self.prev_ref_frame_num, max_frame_num);
                        let dpb_ref: &Dpb = dpb;
                        let default_l0 =
                            dpb_ref.build_list0_b(poc, sh.num_ref_idx_l0_active_minus1);
                        let default_l1 =
                            dpb_ref.build_list1_b(poc, sh.num_ref_idx_l1_active_minus1);
                        let curr_pic_num = sh.frame_num as i32;
                        let l0_entries = if sh.rplm_l0.is_empty() {
                            default_l0
                        } else {
                            apply_rplm(
                                default_l0,
                                &sh.rplm_l0,
                                curr_pic_num,
                                max_frame_num as i32,
                                sh.num_ref_idx_l0_active_minus1,
                                dpb_ref.frames_raw(),
                            )?
                        };
                        let l1_entries = if sh.rplm_l1.is_empty() {
                            default_l1
                        } else {
                            apply_rplm(
                                default_l1,
                                &sh.rplm_l1,
                                curr_pic_num,
                                max_frame_num as i32,
                                sh.num_ref_idx_l1_active_minus1,
                                dpb_ref.frames_raw(),
                            )?
                        };
                        if l0_entries.is_empty() || l1_entries.is_empty() {
                            return Err(Error::invalid(
                                "h264 b-slice: empty RefPicList0/1 (no prior references?)",
                            ));
                        }
                        for rf in l0_entries.iter().chain(l1_entries.iter()) {
                            if rf.pic.mb_width != mb_w || rf.pic.mb_height != mb_h {
                                return Err(Error::invalid(
                                    "h264 b-slice: reference picture size mismatch",
                                ));
                            }
                        }
                        let list0: Vec<&Picture> = l0_entries.iter().map(|r| &r.pic).collect();
                        let list1: Vec<&Picture> = l1_entries.iter().map(|r| &r.pic).collect();
                        let list0_pocs: Vec<i32> = l0_entries.iter().map(|r| r.poc).collect();
                        let list1_pocs: Vec<i32> = l1_entries.iter().map(|r| r.poc).collect();
                        let list0_short: Vec<bool> =
                            l0_entries.iter().map(|r| r.short_term).collect();
                        let list1_short: Vec<bool> =
                            l1_entries.iter().map(|r| r.short_term).collect();
                        // Pick the bipred combine mode per §8.4.2.3:
                        // explicit table present → §8.4.2.3.2,
                        // PPS `weighted_bipred_idc == 2` → §8.4.2.3.3 implicit,
                        // otherwise default average (§8.4.2.3.1).
                        let bipred_mode = if sh.pred_weight_table.is_some() {
                            crate::b_mb::BipredMode::Explicit
                        } else if pps.weighted_bipred_idc == 2 {
                            crate::b_mb::BipredMode::Implicit
                        } else {
                            crate::b_mb::BipredMode::Default
                        };
                        let b_ctx = crate::b_mb::BSliceCtx {
                            curr_poc: poc,
                            list0_pocs: &list0_pocs,
                            list1_pocs: &list1_pocs,
                            list0_short_term: &list0_short,
                            list1_short_term: &list1_short,
                            bipred_mode,
                        };
                        if pps.entropy_coding_mode_flag && high_bit_depth {
                            decode_cabac_b_slice_hi(
                                &rbsp, &sh, &sps, &pps, &mut pic, &list0, &list1, &b_ctx,
                            )?;
                        } else if pps.entropy_coding_mode_flag && sps.chroma_format_idc == 3 {
                            decode_cabac_b_slice_444(
                                &rbsp, &sh, &sps, &pps, &mut pic, &list0, &list1, &b_ctx,
                            )?;
                        } else if pps.entropy_coding_mode_flag {
                            decode_cabac_b_slice(
                                &rbsp, &sh, &sps, &pps, &mut pic, &list0, &list1, &b_ctx,
                            )?;
                        } else if high_bit_depth {
                            let mut br = BitReader::new(&rbsp);
                            br.skip(sh.slice_data_bit_offset as u32)?;
                            decode_b_slice_data_hi(
                                &mut br, &sh, &sps, &pps, &mut pic, &list0, &list1, &b_ctx,
                            )?;
                        } else if sps.chroma_format_idc == 3 {
                            let mut br = BitReader::new(&rbsp);
                            br.skip(sh.slice_data_bit_offset as u32)?;
                            decode_b_slice_data_444(
                                &mut br, &sh, &sps, &pps, &mut pic, &list0, &list1, &b_ctx,
                            )?;
                        } else {
                            let mut br = BitReader::new(&rbsp);
                            br.skip(sh.slice_data_bit_offset as u32)?;
                            decode_b_slice_data(
                                &mut br, &sh, &sps, &pps, &mut pic, &list0, &list1, &b_ctx,
                            )?;
                        }
                        // Snapshot POCs so future B-slices that use this
                        // picture as a colocated reference can resolve
                        // `refIdxCol` to a POC without holding reference
                        // list pointers (§8.4.1.2.3).
                        pic.snapshot_ref_pocs(&list0_pocs, &list1_pocs);
                    }
                    _ => unreachable!(),
                }

                // Optional in-loop deblocking — §8.7. The 8-bit kernel
                // touches `pic.y/cb/cr`; the 10-bit kernel touches the u16
                // planes and scales α / β / tC0 by `1 << (BitDepth - 8)`
                // per §8.7.2.1.
                //
                // §8.7.1.1 — MBAFF pictures use a dedicated deblock
                // driver that walks MB pairs and keys the edge schedule
                // on each pair's `mb_field_decoding_flag`. Frame-coded
                // pairs follow the progressive schedule; field-coded
                // pairs step through the interleaved plane at
                // `row_step = 2` and skip the inside-pair boundary
                // between the top and bottom MB. 10-bit MBAFF falls
                // back to skipping deblock for now — that path isn't
                // wired (MBAFF + high-bit-depth returns Unsupported at
                // slice entry).
                if sh.disable_deblocking_filter_idc != 1 {
                    if pic.mbaff_enabled {
                        if !pic.is_high_bit_depth() {
                            crate::deblock::deblock_picture_mbaff(&mut pic, &pps, &sh);
                        }
                    } else if pic.is_high_bit_depth() {
                        crate::deblock_hi::deblock_picture_hi(&mut pic, &pps, &sh);
                    } else {
                        crate::deblock::deblock_picture(&mut pic, &pps, &sh);
                    }
                }

                // ---- DPB insertion + reference marking ---------------
                // Record the packet PTS paired with this decode-order
                // picture for later POC-reordered delivery.
                self.pts_fifo.push_back((self.pending_pts, self.pending_tb));
                let dpb = self.dpb.as_mut().expect("DPB initialised above");
                if is_ref {
                    dpb.update_frame_num_wrap(self.prev_ref_frame_num, max_frame_num);
                    let frame_num_wrap = Dpb::compute_frame_num_wrap(
                        sh.frame_num,
                        self.prev_ref_frame_num,
                        max_frame_num,
                    );
                    if sh.adaptive_ref_pic_marking_mode_flag {
                        // MMCO mode — ops 3 and 6 may reference the
                        // current picture, so insert before replaying.
                        dpb.insert_reference(pic.clone(), sh.frame_num, frame_num_wrap, poc);
                        for cmd in &sh.mmco_commands {
                            dpb.apply_mmco(*cmd, sh.frame_num, max_frame_num)?;
                        }
                    } else {
                        // Sliding-window mode — age the oldest short-term
                        // ref out when at capacity, then insert.
                        dpb.apply_sliding_window();
                        dpb.insert_reference(pic.clone(), sh.frame_num, frame_num_wrap, poc);
                        if sh.is_idr && sh.idr_long_term_reference_flag {
                            dpb.apply_mmco(
                                MmcoCommand::AssignCurrentLongTerm {
                                    long_term_frame_idx: 0,
                                },
                                sh.frame_num,
                                max_frame_num,
                            )?;
                        }
                    }
                    dpb.mark_last_output();
                    self.prev_ref_frame_num = sh.frame_num;
                }
                self.prev_frame_num = sh.frame_num;

                // Queue the picture for POC-ordered output and drain any
                // pictures that the reorder window now allows.
                dpb.queue_output(pic, poc);
                let ready = dpb.take_ready_outputs();
                for out in ready {
                    self.queue_ready_frame(&sps, out);
                }
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

    /// Turn a decoded `Picture` (emerging from DPB POC-ordering) into a
    /// `VideoFrame` and enqueue it for the caller. Picks pts/timebase
    /// from the head of `pts_fifo` (decode-order), which is a sensible
    /// default when POC order matches decode order (no B-slices).
    fn queue_ready_frame(&mut self, sps: &Sps, pic: Picture) {
        let (mut vw, mut vh) = sps.visible_size();
        // §7.4.3 PAFF: a field picture holds only half the frame's rows.
        // Emit it at half-visible-height so downstream consumers see
        // the correct sample layout (pairing back into a full frame is
        // left to the caller — see README).
        if pic.field_pic {
            vh /= 2;
            // Guard against pathological odd visible heights that would
            // otherwise zero-collapse on division. Clamp to at least 1
            // row so `to_video_frame` sees non-empty planes.
            if vh == 0 {
                vh = 1;
            }
            // Clamp to the allocated field dimensions so we never read
            // past the buffer when visible crop differs from coded.
            vw = vw.min(pic.width);
            vh = vh.min(pic.height);
        }
        let (pts, tb) = self
            .pts_fifo
            .pop_front()
            .unwrap_or((self.pending_pts, self.pending_tb));
        let frame = pic.to_video_frame(vw, vh, pts, tb);
        self.ready_frames.push_back(frame);
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
    ref_list0: &[&Picture],
) -> Result<()> {
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = sps.pic_height_in_map_units();
    let total_mbs = mb_w * mb_h;
    let mut mb_addr = sh.first_mb_in_slice;
    if mb_addr >= total_mbs {
        return Err(Error::invalid(
            "h264 p-slice: first_mb_in_slice out of range",
        ));
    }
    let mut prev_qp = (pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta).clamp(0, 51);

    while mb_addr < total_mbs {
        let skip_run = br.read_ue()?;
        for _ in 0..skip_run {
            if mb_addr >= total_mbs {
                break;
            }
            let mb_x = mb_addr % mb_w;
            let mb_y = mb_addr / mb_w;
            decode_p_skip_mb(sh, mb_x, mb_y, pic, ref_list0, prev_qp)?;
            mb_addr += 1;
        }
        if mb_addr >= total_mbs {
            break;
        }
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        decode_p_slice_mb(br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, &mut prev_qp)?;
        mb_addr += 1;
    }
    Ok(())
}

/// CAVLC P-slice data loop — 10-bit pipeline. Mirrors
/// [`decode_p_slice_data`] but drives the u16 macroblock decoder
/// [`decode_p_slice_mb_hi`] and uses the 10-bit P_Skip helper. The
/// initial QpY is allowed to start in `[-QpBdOffsetY, 51]` per
/// §7.4.2.1.1, and `prev_qp` is extended to the same range before being
/// fed through the macroblock loop.
fn decode_p_slice_data_hi(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
    ref_list0: &[&Picture],
) -> Result<()> {
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = sps.pic_height_in_map_units();
    let total_mbs = mb_w * mb_h;
    let mut mb_addr = sh.first_mb_in_slice;
    if mb_addr >= total_mbs {
        return Err(Error::invalid(
            "h264 p-slice: first_mb_in_slice out of range",
        ));
    }
    let qp_bd_offset_y = 6 * sps.bit_depth_luma_minus8 as i32;
    let mut prev_qp = pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta;
    if prev_qp < -qp_bd_offset_y || prev_qp > 51 {
        return Err(Error::invalid(format!(
            "h264 10bit p-slice: initial QP {prev_qp} out of range [{}..=51]",
            -qp_bd_offset_y
        )));
    }

    while mb_addr < total_mbs {
        let skip_run = br.read_ue()?;
        for _ in 0..skip_run {
            if mb_addr >= total_mbs {
                break;
            }
            let mb_x = mb_addr % mb_w;
            let mb_y = mb_addr / mb_w;
            decode_p_skip_mb_hi(sh, mb_x, mb_y, pic, ref_list0, prev_qp)?;
            mb_addr += 1;
        }
        if mb_addr >= total_mbs {
            break;
        }
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        decode_p_slice_mb_hi(br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, &mut prev_qp)?;
        mb_addr += 1;
    }
    Ok(())
}

/// CAVLC B-slice data loop — §7.3.4.
///
/// Mirrors the P-slice loop: each coded MB is preceded by `mb_skip_run` that
/// counts how many `B_Skip` MBs come before the next coded MB. Trailing
/// `B_Skip` runs are also permitted (no coded MB follows).
fn decode_b_slice_data(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    ctx: &crate::b_mb::BSliceCtx<'_>,
) -> Result<()> {
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = sps.pic_height_in_map_units();
    let total_mbs = mb_w * mb_h;
    let mut mb_addr = sh.first_mb_in_slice;
    if mb_addr >= total_mbs {
        return Err(Error::invalid(
            "h264 b-slice: first_mb_in_slice out of range",
        ));
    }
    let mut prev_qp = (pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta).clamp(0, 51);

    while mb_addr < total_mbs {
        let skip_run = br.read_ue()?;
        for _ in 0..skip_run {
            if mb_addr >= total_mbs {
                break;
            }
            let mb_x = mb_addr % mb_w;
            let mb_y = mb_addr / mb_w;
            decode_b_skip_mb(sh, sps, mb_x, mb_y, pic, ref_list0, ref_list1, ctx, prev_qp)?;
            mb_addr += 1;
        }
        if mb_addr >= total_mbs {
            break;
        }
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        decode_b_slice_mb(
            br,
            sps,
            pps,
            sh,
            mb_x,
            mb_y,
            pic,
            ref_list0,
            ref_list1,
            ctx,
            &mut prev_qp,
        )?;
        mb_addr += 1;
    }
    Ok(())
}

/// CAVLC P-slice data loop for 4:4:4. Mirrors [`decode_p_slice_data`] but
/// drives the 4:4:4 macroblock decoder in [`crate::p_mb_444`]. The spec
/// wire-layer is unchanged (`mb_skip_run` preceding each coded MB); only
/// the per-MB decoding shape differs under ChromaArrayType = 3.
fn decode_p_slice_data_444(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
    ref_list0: &[&Picture],
) -> Result<()> {
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = sps.pic_height_in_map_units();
    let total_mbs = mb_w * mb_h;
    let mut mb_addr = sh.first_mb_in_slice;
    if mb_addr >= total_mbs {
        return Err(Error::invalid(
            "h264 p-slice 4:4:4: first_mb_in_slice out of range",
        ));
    }
    let mut prev_qp = (pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta).clamp(0, 51);
    while mb_addr < total_mbs {
        let skip_run = br.read_ue()?;
        for _ in 0..skip_run {
            if mb_addr >= total_mbs {
                break;
            }
            let mb_x = mb_addr % mb_w;
            let mb_y = mb_addr / mb_w;
            decode_p_skip_mb_444(sh, mb_x, mb_y, pic, ref_list0, prev_qp)?;
            mb_addr += 1;
        }
        if mb_addr >= total_mbs {
            break;
        }
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        decode_p_slice_mb_444(br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, &mut prev_qp)?;
        mb_addr += 1;
    }
    Ok(())
}

/// CAVLC B-slice data loop for 4:4:4. Mirrors [`decode_b_slice_data`].
#[allow(clippy::too_many_arguments)]
fn decode_b_slice_data_444(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    ctx: &crate::b_mb::BSliceCtx<'_>,
) -> Result<()> {
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = sps.pic_height_in_map_units();
    let total_mbs = mb_w * mb_h;
    let mut mb_addr = sh.first_mb_in_slice;
    if mb_addr >= total_mbs {
        return Err(Error::invalid(
            "h264 b-slice 4:4:4: first_mb_in_slice out of range",
        ));
    }
    let mut prev_qp = (pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta).clamp(0, 51);
    while mb_addr < total_mbs {
        let skip_run = br.read_ue()?;
        for _ in 0..skip_run {
            if mb_addr >= total_mbs {
                break;
            }
            let mb_x = mb_addr % mb_w;
            let mb_y = mb_addr / mb_w;
            decode_b_skip_mb_444(sh, sps, mb_x, mb_y, pic, ref_list0, ref_list1, ctx, prev_qp)?;
            mb_addr += 1;
        }
        if mb_addr >= total_mbs {
            break;
        }
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        decode_b_slice_mb_444(
            br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, ref_list1, ctx, &mut prev_qp,
        )?;
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

/// 10-bit CAVLC B-slice data loop. Mirrors [`decode_b_slice_data`] but
/// dispatches the coded MB decode through [`crate::b_mb_hi::decode_b_slice_mb_hi`]
/// so reconstruction lands on the `u16` planes. `prev_qp` is carried in
/// the extended `[-QpBdOffsetY, 51]` window and wrapped modulo
/// `52 + QpBdOffsetY` per §7.4.5 extended.
#[allow(clippy::too_many_arguments)]
fn decode_b_slice_data_hi(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    ctx: &crate::b_mb::BSliceCtx<'_>,
) -> Result<()> {
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = sps.pic_height_in_map_units();
    let total_mbs = mb_w * mb_h;
    let mut mb_addr = sh.first_mb_in_slice;
    if mb_addr >= total_mbs {
        return Err(Error::invalid(
            "h264 10bit b-slice: first_mb_in_slice out of range",
        ));
    }
    let qp_bd_offset_y = 6 * sps.bit_depth_luma_minus8 as i32;
    let mut prev_qp = pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta;
    if prev_qp < -qp_bd_offset_y || prev_qp > 51 {
        return Err(Error::invalid(format!(
            "h264 10bit b-slice: initial QP {prev_qp} out of range [{}..=51]",
            -qp_bd_offset_y
        )));
    }

    while mb_addr < total_mbs {
        let skip_run = br.read_ue()?;
        for _ in 0..skip_run {
            if mb_addr >= total_mbs {
                break;
            }
            let mb_x = mb_addr % mb_w;
            let mb_y = mb_addr / mb_w;
            crate::b_mb_hi::decode_b_skip_mb_hi(
                sh, sps, mb_x, mb_y, pic, ref_list0, ref_list1, ctx, prev_qp,
            )?;
            mb_addr += 1;
        }
        if mb_addr >= total_mbs {
            break;
        }
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        crate::b_mb_hi::decode_b_slice_mb_hi(
            br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, ref_list1, ctx, &mut prev_qp,
        )?;
        mb_addr += 1;
    }
    Ok(())
}

/// §9.3 CABAC entropy-coded 4:4:4 I-slice loop. Mirrors
/// [`decode_cabac_i_slice`] but dispatches each macroblock through
/// [`crate::cabac::mb_444::decode_i_mb_cabac_444`] so the three colour
/// planes (Y / Cb / Cr) each run through a luma-shaped residual pipeline.
fn decode_cabac_i_slice_444(
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
            "h264 cabac 4:4:4 i-slice: first_mb_in_slice out of range",
        ));
    }
    loop {
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        decode_i_mb_cabac_444(
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
        let end = dec.decode_terminate()?;
        if end == 1 {
            break;
        }
        if mb_addr >= total_mbs {
            return Err(Error::invalid(
                "h264 cabac 4:4:4 i-slice: end_of_slice_flag did not fire before last MB",
            ));
        }
    }
    Ok(())
}

/// §9.3 CABAC entropy-coded 4:4:4 P-slice loop.
fn decode_cabac_p_slice_444(
    rbsp: &[u8],
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
    ref_list0: &[&Picture],
) -> Result<()> {
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = sps.pic_height_in_map_units();
    let total_mbs = mb_w * mb_h;
    let slice_qpy = (pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta).clamp(0, 51);
    let mut ctxs = init_slice_contexts(sh.cabac_init_idc, false, slice_qpy);
    let mut dec = CabacDecoder::new(rbsp, sh.slice_data_bit_offset as usize)?;
    let mut prev_qp = slice_qpy;
    pic.last_mb_qp_delta_was_nonzero = false;
    let mut mb_addr = sh.first_mb_in_slice;
    if mb_addr >= total_mbs {
        return Err(Error::invalid(
            "h264 cabac 4:4:4 p-slice: first_mb_in_slice out of range",
        ));
    }
    loop {
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        decode_p_mb_cabac_444(
            &mut dec,
            &mut ctxs,
            sh,
            sps,
            pps,
            mb_x,
            mb_y,
            pic,
            ref_list0,
            &mut prev_qp,
        )?;
        mb_addr += 1;
        let end = dec.decode_terminate()?;
        if end == 1 {
            break;
        }
        if mb_addr >= total_mbs {
            return Err(Error::invalid(
                "h264 cabac 4:4:4 p-slice: end_of_slice_flag did not fire before last MB",
            ));
        }
    }
    Ok(())
}

/// §9.3 CABAC entropy-coded 4:4:4 B-slice loop.
#[allow(clippy::too_many_arguments)]
fn decode_cabac_b_slice_444(
    rbsp: &[u8],
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    bctx: &crate::b_mb::BSliceCtx<'_>,
) -> Result<()> {
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = sps.pic_height_in_map_units();
    let total_mbs = mb_w * mb_h;
    let slice_qpy = (pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta).clamp(0, 51);
    let mut ctxs = init_slice_contexts(sh.cabac_init_idc, false, slice_qpy);
    let mut dec = CabacDecoder::new(rbsp, sh.slice_data_bit_offset as usize)?;
    let mut prev_qp = slice_qpy;
    pic.last_mb_qp_delta_was_nonzero = false;
    let mut mb_addr = sh.first_mb_in_slice;
    if mb_addr >= total_mbs {
        return Err(Error::invalid(
            "h264 cabac 4:4:4 b-slice: first_mb_in_slice out of range",
        ));
    }
    loop {
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        decode_b_mb_cabac_444(
            &mut dec,
            &mut ctxs,
            sh,
            sps,
            pps,
            mb_x,
            mb_y,
            pic,
            ref_list0,
            ref_list1,
            bctx,
            &mut prev_qp,
        )?;
        mb_addr += 1;
        let end = dec.decode_terminate()?;
        if end == 1 {
            break;
        }
        if mb_addr >= total_mbs {
            return Err(Error::invalid(
                "h264 cabac 4:4:4 b-slice: end_of_slice_flag did not fire before last MB",
            ));
        }
    }
    Ok(())
}

/// §9.3 CABAC entropy-coded 10-bit I-slice loop — mirrors
/// [`decode_cabac_i_slice`] but dispatches each macroblock through
/// [`crate::cabac::mb_hi::decode_i_mb_cabac_hi`] so the reconstructed
/// samples land on the `u16` planes. CABAC's slice-QP init clips
/// `SliceQpY` to `[0, 51]` internally (§9.3.1.1), so the 10-bit
/// `[-QpBdOffsetY, 51]` window for `prev_qp` is still honoured while the
/// per-context `preCtxState` derivation stays on its spec-mandated range.
fn decode_cabac_i_slice_hi(
    rbsp: &[u8],
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
) -> Result<()> {
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = sps.pic_height_in_map_units();
    let total_mbs = mb_w * mb_h;
    let qp_bd_offset_y = 6 * sps.bit_depth_luma_minus8 as i32;
    let slice_qpy = pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta;
    if slice_qpy < -qp_bd_offset_y || slice_qpy > 51 {
        return Err(Error::invalid(format!(
            "h264 10bit cabac i-slice: initial QP {slice_qpy} out of range [{}..=51]",
            -qp_bd_offset_y
        )));
    }
    let mut ctxs = init_slice_contexts(sh.cabac_init_idc, true, slice_qpy);
    let mut dec = CabacDecoder::new(rbsp, sh.slice_data_bit_offset as usize)?;
    let mut prev_qp = slice_qpy;
    pic.last_mb_qp_delta_was_nonzero = false;
    let mut mb_addr = sh.first_mb_in_slice;
    if mb_addr >= total_mbs {
        return Err(Error::invalid(
            "h264 10bit cabac i-slice: first_mb_in_slice out of range",
        ));
    }
    loop {
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        decode_i_mb_cabac_hi(
            &mut dec, &mut ctxs, sh, sps, pps, mb_x, mb_y, pic, &mut prev_qp,
        )?;
        mb_addr += 1;
        let end = dec.decode_terminate()?;
        if end == 1 {
            break;
        }
        if mb_addr >= total_mbs {
            return Err(Error::invalid(
                "h264 10bit cabac i-slice: end_of_slice_flag did not fire before last MB",
            ));
        }
    }
    Ok(())
}

/// §9.3 CABAC entropy-coded P-slice data loop. Mirrors `decode_p_slice_data`
/// but feeds every macroblock through [`crate::cabac::p_mb::decode_p_mb_cabac`]
/// and relies on [`CabacDecoder::decode_terminate`] for the
/// `end_of_slice_flag`. `mb_skip_flag` replaces CAVLC's `mb_skip_run`.
fn decode_cabac_p_slice(
    rbsp: &[u8],
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
    ref_list0: &[&Picture],
) -> Result<()> {
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = sps.pic_height_in_map_units();
    let total_mbs = mb_w * mb_h;
    let slice_qpy = (pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta).clamp(0, 51);
    let mut ctxs = init_slice_contexts(sh.cabac_init_idc, false, slice_qpy);
    let mut dec = CabacDecoder::new(rbsp, sh.slice_data_bit_offset as usize)?;
    let mut prev_qp = slice_qpy;
    pic.last_mb_qp_delta_was_nonzero = false;
    let mut mb_addr = sh.first_mb_in_slice;
    if mb_addr >= total_mbs {
        return Err(Error::invalid(
            "h264 cabac p-slice: first_mb_in_slice out of range",
        ));
    }
    loop {
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        let skipped = decode_p_mb_cabac(
            &mut dec,
            &mut ctxs,
            sh,
            sps,
            pps,
            mb_x,
            mb_y,
            pic,
            ref_list0,
            &mut prev_qp,
        )?;
        mb_addr += 1;
        // §9.3.3.2.4 end_of_slice_flag — terminate bin after every MB, even
        // skipped ones. If a skip runs us past the last MB without the
        // terminate bit firing we flag a malformed bitstream.
        let _ = skipped;
        let end = dec.decode_terminate()?;
        if end == 1 {
            break;
        }
        if mb_addr >= total_mbs {
            return Err(Error::invalid(
                "h264 cabac p-slice: end_of_slice_flag did not fire before last MB",
            ));
        }
    }
    Ok(())
}

/// §9.3 CABAC entropy-coded B-slice data loop. Mirrors `decode_b_slice_data`
/// but feeds every macroblock through [`crate::cabac::b_mb::decode_b_mb_cabac`]
/// and relies on [`CabacDecoder::decode_terminate`] for the
/// `end_of_slice_flag`. `mb_skip_flag` replaces CAVLC's `mb_skip_run`.
#[allow(clippy::too_many_arguments)]
fn decode_cabac_b_slice(
    rbsp: &[u8],
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    bctx: &crate::b_mb::BSliceCtx<'_>,
) -> Result<()> {
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = sps.pic_height_in_map_units();
    let total_mbs = mb_w * mb_h;
    let slice_qpy = (pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta).clamp(0, 51);
    let mut ctxs = init_slice_contexts(sh.cabac_init_idc, false, slice_qpy);
    let mut dec = CabacDecoder::new(rbsp, sh.slice_data_bit_offset as usize)?;
    let mut prev_qp = slice_qpy;
    pic.last_mb_qp_delta_was_nonzero = false;
    let mut mb_addr = sh.first_mb_in_slice;
    if mb_addr >= total_mbs {
        return Err(Error::invalid(
            "h264 cabac b-slice: first_mb_in_slice out of range",
        ));
    }
    loop {
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        decode_b_mb_cabac(
            &mut dec,
            &mut ctxs,
            sh,
            sps,
            pps,
            mb_x,
            mb_y,
            pic,
            ref_list0,
            ref_list1,
            bctx,
            &mut prev_qp,
        )?;
        mb_addr += 1;
        let end = dec.decode_terminate()?;
        if end == 1 {
            break;
        }
        if mb_addr >= total_mbs {
            return Err(Error::invalid(
                "h264 cabac b-slice: end_of_slice_flag did not fire before last MB",
            ));
        }
    }
    Ok(())
}

/// §9.3 CABAC entropy-coded 10-bit P-slice data loop. Mirrors
/// [`decode_cabac_p_slice`] but feeds every macroblock through
/// [`crate::cabac::p_mb_hi::decode_p_mb_cabac_hi`] so the u16 planes stay
/// the reconstruction target and residual dequant widens through
/// `*_scaled_ext`. CABAC clamps `SliceQpY` to `[0, 51]` (§9.3.1.1), matching
/// the 10-bit CABAC I-slice loop.
fn decode_cabac_p_slice_hi(
    rbsp: &[u8],
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
    ref_list0: &[&Picture],
) -> Result<()> {
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = sps.pic_height_in_map_units();
    let total_mbs = mb_w * mb_h;
    let slice_qpy = (pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta).clamp(0, 51);
    let qp_bd_offset_y = 6 * sps.bit_depth_luma_minus8 as i32;
    let mut prev_qp = slice_qpy;
    if prev_qp < -qp_bd_offset_y || prev_qp > 51 {
        return Err(Error::invalid(format!(
            "h264 10bit cabac p-slice: initial QP {prev_qp} out of range [{}..=51]",
            -qp_bd_offset_y
        )));
    }
    let mut ctxs = init_slice_contexts(sh.cabac_init_idc, false, slice_qpy);
    let mut dec = CabacDecoder::new(rbsp, sh.slice_data_bit_offset as usize)?;
    pic.last_mb_qp_delta_was_nonzero = false;
    let mut mb_addr = sh.first_mb_in_slice;
    if mb_addr >= total_mbs {
        return Err(Error::invalid(
            "h264 10bit cabac p-slice: first_mb_in_slice out of range",
        ));
    }
    loop {
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        let _skipped = decode_p_mb_cabac_hi(
            &mut dec,
            &mut ctxs,
            sh,
            sps,
            pps,
            mb_x,
            mb_y,
            pic,
            ref_list0,
            &mut prev_qp,
        )?;
        mb_addr += 1;
        let end = dec.decode_terminate()?;
        if end == 1 {
            break;
        }
        if mb_addr >= total_mbs {
            return Err(Error::invalid(
                "h264 10bit cabac p-slice: end_of_slice_flag did not fire before last MB",
            ));
        }
    }
    Ok(())
}

/// §9.3 CABAC entropy-coded 10-bit B-slice data loop. Mirrors
/// [`decode_cabac_b_slice`] but dispatches each MB through
/// [`crate::cabac::b_mb_hi::decode_b_mb_cabac_hi`].
#[allow(clippy::too_many_arguments)]
fn decode_cabac_b_slice_hi(
    rbsp: &[u8],
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    bctx: &crate::b_mb::BSliceCtx<'_>,
) -> Result<()> {
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = sps.pic_height_in_map_units();
    let total_mbs = mb_w * mb_h;
    let slice_qpy = (pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta).clamp(0, 51);
    let qp_bd_offset_y = 6 * sps.bit_depth_luma_minus8 as i32;
    let mut prev_qp = slice_qpy;
    if prev_qp < -qp_bd_offset_y || prev_qp > 51 {
        return Err(Error::invalid(format!(
            "h264 10bit cabac b-slice: initial QP {prev_qp} out of range [{}..=51]",
            -qp_bd_offset_y
        )));
    }
    let mut ctxs = init_slice_contexts(sh.cabac_init_idc, false, slice_qpy);
    let mut dec = CabacDecoder::new(rbsp, sh.slice_data_bit_offset as usize)?;
    pic.last_mb_qp_delta_was_nonzero = false;
    let mut mb_addr = sh.first_mb_in_slice;
    if mb_addr >= total_mbs {
        return Err(Error::invalid(
            "h264 10bit cabac b-slice: first_mb_in_slice out of range",
        ));
    }
    loop {
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        decode_b_mb_cabac_hi(
            &mut dec,
            &mut ctxs,
            sh,
            sps,
            pps,
            mb_x,
            mb_y,
            pic,
            ref_list0,
            ref_list1,
            bctx,
            &mut prev_qp,
        )?;
        mb_addr += 1;
        let end = dec.decode_terminate()?;
        if end == 1 {
            break;
        }
        if mb_addr >= total_mbs {
            return Err(Error::invalid(
                "h264 10bit cabac b-slice: end_of_slice_flag did not fire before last MB",
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
        // Drain any pictures still parked in the DPB reorder queue. We
        // have to clone a reference SPS so we can release the DPB borrow
        // before calling `queue_ready_frame`.
        let (remaining, sps_opt) = if let Some(dpb) = self.dpb.as_mut() {
            (dpb.flush_all(), self.sps_by_id.values().next().cloned())
        } else {
            (Vec::new(), None)
        };
        if let Some(sps) = sps_opt {
            for pic in remaining {
                self.queue_ready_frame(&sps, pic);
            }
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // Drop the in-flight picture, pending slice-header cache, and
        // ready frame queue. SPS/PPS tables + AVCC config (length_size,
        // last_avc_config) are stream-level — they come from extradata or
        // out-of-band SPS/PPS NALs that won't be retransmitted post-seek,
        // so they stay put. The DPB + POC state are dropped too so the
        // next P-slice sees a clean slate (it must be preceded by an
        // IDR or another I-slice).
        self.pending_pts = None;
        self.ready_frames.clear();
        self.eof = false;
        self.last_slice_headers.clear();
        self.current_pic = None;
        if let Some(dpb) = self.dpb.as_mut() {
            dpb.clear();
        }
        self.poc_state = PocState::default();
        self.prev_frame_num = 0;
        self.prev_frame_num_offset = 0;
        self.prev_ref_frame_num = 0;
        self.pts_fifo.clear();
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
