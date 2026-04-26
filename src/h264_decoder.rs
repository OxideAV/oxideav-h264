//! H.264 decoder scaffold exposed through the `oxideav_core::Decoder`
//! trait so containers + players can route packets at us.
//!
//! What this scaffold currently does:
//! 1. Accepts Annex B byte-stream packets or AVCC-framed packets
//!    (length prefix size taken from `extradata` when present — an
//!    `AVCDecoderConfigurationRecord` per ISO/IEC 14496-15).
//! 2. Walks NAL units through [`crate::decoder::Decoder`] which
//!    captures SPS/PPS and emits parsed slice headers.
//! 3. Maintains a decoded picture store ([`crate::ref_store::RefPicStore`])
//!    across pictures so P/B slices have reference pictures for
//!    motion compensation (§8.2.4 / §8.2.5).
//! 4. For every slice it parses, the decoder derives POC (§8.2.1),
//!    builds the RefPicList0 / RefPicList1 that §8.4.2 inter
//!    prediction consumes, runs `reconstruct::reconstruct_slice`, and
//!    queues the reconstructed Picture as a `VideoFrame` for
//!    `receive_frame`.
//!
//! Known simplifications (see inline comments for details):
//! - **Access unit assembly**: §7.4.1.2.4 multi-slice assembly IS
//!   implemented — continuation slices land in the same Picture +
//!   MbGrid as the first slice, and we finalize the picture (push to
//!   DPB + output queue) when a slice opens a new primary coded picture
//!   or on AUD / EndOfSequence / flush. Caveat: the deblocking pass in
//!   `reconstruct::reconstruct_slice` runs per-slice and walks the
//!   whole grid, so continuation slices may re-filter earlier slices'
//!   interior edges. See the comment on `handle_slice` for details.
//! - **Output ordering**: frames are emitted in display (POC) order
//!   via [`crate::dpb_output::DpbOutput`] per Annex C §C.2.2 / §C.4
//!   bumping process.
//! - **Field pictures / MBAFF**: not supported. `field_pic_flag == 1`
//!   or `mb_adaptive_frame_field_flag == 1` causes the reconstruct
//!   layer to reject the slice.
//! - **Reference picture list modification (RPLM)**: the ops are
//!   applied via `ref_list::modify_ref_pic_list`, but the underlying
//!   short-term / long-term derivation is a first pass and may not
//!   be fully spec-accurate for all streams.
//!
//! The trait is what oxideplay/oxideav-pipeline consumes; registering
//! this decoder (via [`crate::register`]) stops the "codec not found"
//! error on the first h264 packet.

use std::collections::VecDeque;

use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, Result, TimeBase, VideoFrame, VideoPlane,
};

use crate::decoder::{Decoder as H264Driver, Event};
use crate::dpb_output::{DpbOutput, OutputEntry};
use crate::mb_grid::MbGrid;
use crate::picture::Picture;
use crate::poc::{derive_poc, PocResult, PocSlice, PocSps, PocState};
use crate::ref_list::{self, DpbEntry, MmcoOp as RefMmcoOp, PicStructure, RefMarking, RplmOp};
use crate::ref_store::{RefPicProvider, RefPicStore};
use crate::slice_header::{
    MmcoOp as SliceMmcoOp, RefPicListModificationOp as SliceRplmOp, SliceHeader, SliceType,
};
use crate::sps::Sps;
use crate::{reconstruct, slice_data};

/// §7.4.1.2 / §7.4.1.2.4 — state carried forward across slices that
/// belong to the *same* primary coded picture.
///
/// Once a slice with first_mb_in_slice == 0 (and/or an AUD) opens a new
/// primary coded picture, a `PictureInProgress` is allocated. Each
/// subsequent slice that passes the §7.4.1.2.4 "same picture" test is
/// reconstructed into the *same* `pic` / `grid` so its macroblocks are
/// laid down alongside the earlier slices'. When a slice fails the test
/// (new primary coded picture) or an AUD / flush fires, the in-progress
/// picture is finalized (pushed into the DPB and output queue) and a
/// fresh one is started from the triggering slice.
struct PictureInProgress {
    /// Reconstructed samples.
    pic: Picture,
    /// MB metadata for the assembled picture. Carries §6.4.11 availability
    /// plus per-MB QP/CBP/etc. across slice boundaries so continuation
    /// slices see prior slices' MBs as neighbours during intra prediction
    /// and deblocking.
    grid: MbGrid,
    /// The `(nal_unit_type, nal_ref_idc, header)` of the *first* slice of
    /// this picture — the identity used for §7.4.1.2.4 comparisons against
    /// subsequent slices.
    first_nal_unit_type: u8,
    first_nal_ref_idc: u8,
    first_header: SliceHeader,
    /// True if any slice in the picture so far was a reference slice. A
    /// picture is a reference picture if *any* of its VCL NALs carries
    /// nal_ref_idc != 0 — §7.4.1.2.1 / §7.4.1.2.4 require all slices to
    /// share the zero-ness of nal_ref_idc, so this is effectively the
    /// first slice's is_reference bit, but we OR it to be defensive.
    is_reference: bool,
    /// True if this is an IDR picture (any slice has nal_unit_type == 5).
    is_idr: bool,
    /// §8.2.1 POC result derived at the first slice. All slices of the
    /// same picture share the same POC per §7.4.1.2.4.
    poc: PocResult,
    /// §7.4.3 picture structure for DPB bookkeeping.
    structure: PicStructure,
    /// Packet pts to stamp onto the finalized VideoFrame.
    pts: Option<i64>,
    /// Packet time_base for rescaling downstream.
    time_base: TimeBase,
    /// §8.7 deblocking state from the *first* slice of the picture.
    /// Multi-slice pictures that vary deblocking_filter_idc / alpha_off /
    /// beta_off per slice are a known simplification — the JVT
    /// conformance streams we target encode uniform per-picture
    /// deblock offsets. Kept so `finalize_in_progress_picture` can run
    /// the §8.7 pass exactly once, after every slice has populated the
    /// shared Picture + MbGrid.
    deblock_enabled: bool,
    deblock_alpha_off: i32,
    deblock_beta_off: i32,
    /// §7.4.4 — per-MB `mb_field_decoding_flag` aggregated across every
    /// slice of the picture, indexed by picture-level macroblock
    /// address. `slice_data.mb_field_decoding_flags` is slice-local so
    /// we copy it into this picture-wide vector using the raw
    /// CurrMbAddr walk the slice data parser produced.
    mb_field_flags: Vec<bool>,
    /// §7.4.1.2.1 — SPS + PPS snapshots captured at the first slice's
    /// header-parse time. Used by `finalize_in_progress_picture` for
    /// deblocking instead of reading the driver's current "active"
    /// parameter sets, which may have been overwritten by a later PPS
    /// NAL carrying the same id but different scaling / qp-offset
    /// values (JVT CACQP3 exercises this path).
    sps: Sps,
    pps: crate::pps::Pps,
}

/// Registry factory — called by the codec registry when a container
/// wants a decoder for H.264.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let mut dec = H264CodecDecoder::new(params.codec_id.clone());
    if !params.extradata.is_empty() {
        dec.consume_extradata(&params.extradata)?;
    }
    Ok(Box::new(dec))
}

/// Full per-slice decoder with DPB wiring.
pub struct H264CodecDecoder {
    codec_id: CodecId,
    /// NAL unit length-prefix size from `avcC`, when present. `None`
    /// means the input is treated as Annex B byte-stream.
    length_size: Option<u8>,
    driver: H264Driver,
    /// Last slice header we parsed — useful for probes / asserts.
    pub last_slice: Option<SliceHeader>,
    eof: bool,
    /// §C.2.2 / §C.4 — POC-ordered output DPB. Entries live here until
    /// the bumping process releases them to `receive_frame`. Created
    /// lazily (or recreated on SPS change) from the active SPS's VUI
    /// bitstream restriction (§E.2.1), with an Annex A Table A-1
    /// fallback when the VUI block is absent.
    output_dpb: DpbOutput<VideoFrame>,
    /// Pictures that have already been "bumped" from the DPB and are
    /// waiting for `receive_frame`. This covers both:
    /// 1. entries evicted by `DpbOutput::push` when the queue is full,
    ///    and
    /// 2. entries drained from the queue at an IDR / MMCO-5 so the
    ///    previous sequence's pictures are delivered in POC order
    ///    *before* the new sequence's first frames (§C.4).
    ///
    /// Also used to carry the `flush()` drain at EOF.
    ready: VecDeque<VideoFrame>,
    /// Packet-level pts passed on the most recent `send_packet`. We
    /// stamp the first frame produced from that packet with it.
    pending_pts: Option<i64>,
    /// Packet time_base so downstream consumers can rescale.
    pending_time_base: TimeBase,

    // ---- DPB / cross-picture state (§8.2.1 / §8.2.4 / §8.2.5) -------
    /// Long-lived decoded picture store. Holds reconstructed Pictures
    /// by DPB slot key. The per-slice ref_pic_list_0 / _1 arrays are
    /// repopulated for every slice via `set_list_0` / `set_list_1`.
    ref_store: RefPicStore,
    /// DPB entry metadata (marking, POC, frame_num, …) in decode
    /// order. Parallel to the keys held in `ref_store`. §8.2.4 /
    /// §8.2.5 state-machine ops live on this vector.
    dpb_entries: Vec<DpbEntry>,
    /// §8.2.1 POC derivation state (prev_pic_order_cnt_msb, etc.).
    poc_state: PocState,
    /// Monotonic counter used to mint fresh DPB slot keys. Never
    /// reused within a stream so the sparse `RefPicStore` never
    /// aliases — if this wraps we recycle (stream length in the
    /// billions of frames is not something we need to worry about).
    next_dpb_key: u32,
    /// Set when the previous *reference* picture's
    /// `dec_ref_pic_marking()` contained MMCO-5. Consumed by the next
    /// call to `derive_poc`. §8.2.1 NOTE 1.
    prev_had_mmco5: bool,
    /// For the §8.2.1.1 MMCO-5 hint — the previous reference
    /// picture's `TopFieldOrderCnt`, used only when `prev_had_mmco5`.
    prev_reference_top_foc: i32,
    /// §8.2.5.2 — `PrevRefFrameNum` kept for the gap-in-frame_num
    /// detection / non-existing reference-frame synthesis. Holds the
    /// `frame_num` of the most recent *reference* picture decoded in
    /// the current coded video sequence. Reset on IDR.
    ///
    /// When the next picture arrives with `frame_num` not equal to
    /// `(prev_ref_frame_num + 1) mod MaxFrameNum`, the spec §8.2.5.2
    /// procedure inserts synthetic "non-existing" short-term reference
    /// frames for each missing `frame_num` value.
    prev_ref_frame_num: Option<u32>,

    /// §7.4.1.2 / §7.4.1.2.4 — picture currently being assembled across
    /// one-or-more slice NAL units. `None` means no slice of the current
    /// access unit has been processed yet (either we haven't started, or
    /// the last picture was just finalized). Populated by the first
    /// slice of a primary coded picture and consumed by `finalize_picture`
    /// when the picture boundary is detected.
    in_progress: Option<PictureInProgress>,
}

impl H264CodecDecoder {
    pub fn new(codec_id: CodecId) -> Self {
        // Start with a "generous" placeholder sizing (16 frames, the
        // Annex A upper bound from §A.3.1 item h, `Min(…, 16)`). The
        // first slice updates the sizing in `ensure_output_dpb_sized`
        // from the active SPS.
        Self {
            codec_id,
            length_size: None,
            driver: H264Driver::new(),
            last_slice: None,
            eof: false,
            output_dpb: DpbOutput::<VideoFrame>::new(16, 16),
            ready: VecDeque::new(),
            pending_pts: None,
            pending_time_base: TimeBase::new(1, 1),
            ref_store: RefPicStore::new(),
            dpb_entries: Vec::new(),
            poc_state: PocState::default(),
            next_dpb_key: 0,
            prev_had_mmco5: false,
            prev_reference_top_foc: 0,
            prev_ref_frame_num: None,
            in_progress: None,
        }
    }

    /// §4.1.3.1 ISO/IEC 14496-15 — `AVCDecoderConfigurationRecord`.
    /// Minimal parse: pick up `lengthSizeMinusOne` and feed the stored
    /// SPS + PPS NAL units through the driver.
    ///
    /// Layout:
    /// ```text
    ///   u8  configurationVersion (= 1)
    ///   u8  AVCProfileIndication
    ///   u8  profile_compatibility
    ///   u8  AVCLevelIndication
    ///   u8  reserved (6 bits, 111111) + lengthSizeMinusOne (2 bits)
    ///   u8  reserved (3 bits, 111) + numOfSequenceParameterSets (5 bits)
    ///     repeated numOfSequenceParameterSets times:
    ///       u16 sequenceParameterSetLength
    ///       <that many> sequenceParameterSetNALUnit
    ///   u8  numOfPictureParameterSets
    ///     repeated:
    ///       u16 pictureParameterSetLength
    ///       <that many> pictureParameterSetNALUnit
    /// ```
    pub fn consume_extradata(&mut self, extra: &[u8]) -> Result<()> {
        if extra.len() < 7 {
            return Err(Error::invalid("h264: extradata shorter than avcC header"));
        }
        if extra[0] != 1 {
            return Err(Error::invalid("h264: avcC configurationVersion must be 1"));
        }
        self.length_size = Some((extra[4] & 0x03) + 1);
        let num_sps = (extra[5] & 0x1f) as usize;
        let mut pos = 6;
        for _ in 0..num_sps {
            if pos + 2 > extra.len() {
                return Err(Error::invalid("h264: avcC truncated at SPS length"));
            }
            let len = u16::from_be_bytes([extra[pos], extra[pos + 1]]) as usize;
            pos += 2;
            if pos + len > extra.len() {
                return Err(Error::invalid("h264: avcC truncated at SPS body"));
            }
            let _ = self
                .driver
                .process_nal(&extra[pos..pos + len])
                .map_err(|e| Error::invalid(format!("h264 avcC SPS: {e}")))?;
            pos += len;
        }
        if pos >= extra.len() {
            return Err(Error::invalid("h264: avcC truncated at PPS count"));
        }
        let num_pps = extra[pos] as usize;
        pos += 1;
        for _ in 0..num_pps {
            if pos + 2 > extra.len() {
                return Err(Error::invalid("h264: avcC truncated at PPS length"));
            }
            let len = u16::from_be_bytes([extra[pos], extra[pos + 1]]) as usize;
            pos += 2;
            if pos + len > extra.len() {
                return Err(Error::invalid("h264: avcC truncated at PPS body"));
            }
            let _ = self
                .driver
                .process_nal(&extra[pos..pos + len])
                .map_err(|e| Error::invalid(format!("h264 avcC PPS: {e}")))?;
            pos += len;
        }
        Ok(())
    }

    /// Handle a single emitted driver event.
    fn handle_event(&mut self, ev: Event) -> Result<()> {
        match ev {
            Event::Slice {
                nal_unit_type,
                nal_ref_idc,
                header,
                rbsp,
                slice_data_cursor,
                pps,
                sps,
            } => {
                self.last_slice = Some(header.clone());
                self.handle_slice(
                    nal_unit_type,
                    nal_ref_idc,
                    header,
                    rbsp,
                    slice_data_cursor,
                    sps,
                    pps,
                )
            }
            // §7.4.1.2.3 — Access Unit Delimiter explicitly marks an
            // access unit boundary. Any picture we've been assembling is
            // finalized here so the next slice opens a fresh one.
            Event::AccessUnitDelimiter(_) => {
                self.finalize_in_progress_picture()?;
                Ok(())
            }
            // §7.3.2.5 / §7.3.2.6 — end of sequence / stream close any
            // picture currently being assembled.
            Event::EndOfSequence | Event::EndOfStream => {
                self.finalize_in_progress_picture()?;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// §7.4.1.2.4 — decide whether `header` opens a new primary coded
    /// picture, by comparing to the first slice of the
    /// `in_progress` picture. Returns true when any of the listed
    /// conditions in §7.4.1.2.4 differs (new picture) or when there is
    /// no picture currently in progress.
    ///
    /// The conditions enumerated in the spec (and used here):
    ///   * `frame_num` differs
    ///   * `pic_parameter_set_id` differs
    ///   * `field_pic_flag` differs
    ///   * `nal_ref_idc` is 0 for one and non-0 for the other
    ///   * `pic_order_cnt_lsb` differs  (pic_order_cnt_type == 0)
    ///   * `delta_pic_order_cnt_bottom` differs (pic_order_cnt_type == 0
    ///     and bottom_field_pic_order_in_frame_present_flag == 1)
    ///   * `delta_pic_order_cnt[0]` differs (pic_order_cnt_type == 1)
    ///   * `delta_pic_order_cnt[1]` differs (pic_order_cnt_type == 1 and
    ///     bottom_field_pic_order_in_frame_present_flag == 1)
    ///   * `IdrPicFlag` differs (one is IDR, the other isn't)
    ///   * `IdrPicFlag == 1` AND `idr_pic_id` differs
    fn is_first_vcl_of_new_picture(
        &self,
        nal_unit_type: u8,
        nal_ref_idc: u8,
        header: &SliceHeader,
    ) -> bool {
        let Some(in_progress) = self.in_progress.as_ref() else {
            return true;
        };
        let prev = &in_progress.first_header;
        let prev_idr = in_progress.first_nal_unit_type == 5;
        let curr_idr = nal_unit_type == 5;
        let prev_is_ref = in_progress.first_nal_ref_idc != 0;
        let curr_is_ref = nal_ref_idc != 0;

        if prev.frame_num != header.frame_num {
            return true;
        }
        if prev.pic_parameter_set_id != header.pic_parameter_set_id {
            return true;
        }
        if prev.field_pic_flag != header.field_pic_flag {
            return true;
        }
        if prev_is_ref != curr_is_ref {
            return true;
        }
        if prev.pic_order_cnt_lsb != header.pic_order_cnt_lsb {
            return true;
        }
        if prev.delta_pic_order_cnt_bottom != header.delta_pic_order_cnt_bottom {
            return true;
        }
        if prev.delta_pic_order_cnt[0] != header.delta_pic_order_cnt[0] {
            return true;
        }
        if prev.delta_pic_order_cnt[1] != header.delta_pic_order_cnt[1] {
            return true;
        }
        if prev_idr != curr_idr {
            return true;
        }
        if curr_idr && prev.idr_pic_id != header.idr_pic_id {
            return true;
        }
        false
    }

    /// Drive reconstruction for one slice NAL. Covers both the IDR
    /// and P/B paths.
    ///
    /// §7.4.1.2.4 multi-slice assembly: when this slice is a continuation
    /// of the in-progress picture (same frame_num / POC / IDR status etc.)
    /// we reconstruct straight into the existing Picture + MbGrid so the
    /// slice's macroblocks land alongside the earlier slices'. When this
    /// slice opens a new primary coded picture we first finalize the
    /// previous in-progress one (pushing it into the DPB + output queue)
    /// and then start a fresh Picture + MbGrid.
    ///
    /// Known limitation: reconstruct_slice runs a full-picture deblocking
    /// pass at the end (§8.7). For a continuation slice the earlier
    /// slices' macroblocks are still in the grid with `available == true`,
    /// so the deblocker may re-filter their interior edges — a slight
    /// over-filter. Proper behaviour would defer deblocking until all
    /// slices are in, but the deblocking helper is private to
    /// `reconstruct.rs` and cannot be invoked separately from here.
    /// In practice the artifact is minor compared to the coarse blocking
    /// you get with one-slice-per-picture assembly, which is the bug this
    /// replaces.
    #[allow(clippy::too_many_arguments)] // mirrors Event::Slice's flat layout
    fn handle_slice(
        &mut self,
        nal_unit_type: u8,
        nal_ref_idc: u8,
        header: SliceHeader,
        rbsp: Vec<u8>,
        cursor: (usize, u8),
        sps: Sps,
        pps: crate::pps::Pps,
    ) -> Result<()> {
        // The SPS and PPS have been snapshotted at slice-header parse
        // time (see [`Event::Slice::pps`] for why — same-id PPS
        // re-transmission at access-unit boundaries, as in JVT CACQP3).
        let is_idr = nal_unit_type == 5;
        let is_reference = nal_ref_idc != 0;

        // §7.4.1.2.4 — first VCL of a primary coded picture?
        let starts_new_picture =
            self.is_first_vcl_of_new_picture(nal_unit_type, nal_ref_idc, &header);

        if starts_new_picture {
            // Finalize whatever was in progress before starting the new
            // picture with this slice.
            self.finalize_in_progress_picture()?;

            // §8.2.5.2 — gaps in frame_num. When the stream signals a
            // gap (`frame_num != (PrevRefFrameNum + 1) mod MaxFrameNum`)
            // and the SPS allows gaps, synthesize "non-existing" short-term
            // reference frames for each missing value to keep the
            // sliding window + RPLM picNumX arithmetic aligned with the
            // encoder's view of the DPB.
            if !is_idr && sps.gaps_in_frame_num_value_allowed_flag {
                self.fill_frame_num_gap(&sps, header.frame_num)?;
            }

            // §8.2.1 — derive POC for this picture. All slices of the
            // same picture will share this value (§7.4.1.2.4).
            let poc_sps = make_poc_sps(&sps);
            let poc_slice = PocSlice {
                is_reference,
                is_idr,
                frame_num: header.frame_num,
                field_pic_flag: header.field_pic_flag,
                bottom_field_flag: header.bottom_field_flag,
                pic_order_cnt_lsb: header.pic_order_cnt_lsb,
                delta_pic_order_cnt_bottom: header.delta_pic_order_cnt_bottom,
                delta_pic_order_cnt: header.delta_pic_order_cnt,
                prev_had_mmco5: self.prev_had_mmco5,
                prev_reference_top_foc_for_mmco5: self.prev_reference_top_foc,
            };
            let poc = derive_poc(&poc_sps, &poc_slice, &mut self.poc_state)
                .map_err(|e| Error::invalid(format!("h264 POC: {e:?}")))?;

            let width_samples = sps.pic_width_in_mbs() * 16;
            let height_samples = sps.frame_height_in_mbs() * 16;
            let chroma_array_type = sps.chroma_array_type();
            let pic = Picture::new(
                width_samples,
                height_samples,
                chroma_array_type,
                sps.bit_depth_luma_minus8 + 8,
                sps.bit_depth_chroma_minus8 + 8,
            );
            let grid = MbGrid::new(sps.pic_width_in_mbs(), sps.frame_height_in_mbs());
            let structure =
                pic_structure_from_flags(header.field_pic_flag, header.bottom_field_flag);

            // Consume the packet-level pts exactly once per access unit
            // — the first slice to open a picture gets it.
            let pts = self.pending_pts.take();
            let time_base = self.pending_time_base;

            let deblock_enabled = header.disable_deblocking_filter_idc != 1;
            let deblock_alpha_off = header.slice_alpha_c0_offset_div2 * 2;
            let deblock_beta_off = header.slice_beta_offset_div2 * 2;
            let mb_count = (sps.pic_width_in_mbs() * sps.frame_height_in_mbs()) as usize;
            let mb_field_flags = vec![false; mb_count];

            self.in_progress = Some(PictureInProgress {
                pic,
                grid,
                first_nal_unit_type: nal_unit_type,
                first_nal_ref_idc: nal_ref_idc,
                first_header: header.clone(),
                is_reference,
                is_idr,
                poc,
                structure,
                pts,
                time_base,
                deblock_enabled,
                deblock_alpha_off,
                deblock_beta_off,
                mb_field_flags,
                sps: sps.clone(),
                pps: pps.clone(),
            });
        }

        // Reconstruct this slice into the in-progress picture.
        self.reconstruct_slice_into_in_progress(nal_ref_idc, &header, &rbsp, cursor, &sps, &pps)?;

        Ok(())
    }

    /// Run `reconstruct::reconstruct_slice` against the currently
    /// in-progress picture's pic + grid. Also picks up any OR of
    /// `is_reference` so a picture is marked a reference picture as
    /// soon as any of its slices carries nal_ref_idc != 0 (§7.4.1.2.4
    /// requires this to be uniform across slices, but we're tolerant).
    fn reconstruct_slice_into_in_progress(
        &mut self,
        nal_ref_idc: u8,
        header: &SliceHeader,
        rbsp: &[u8],
        cursor: (usize, u8),
        sps: &Sps,
        pps: &crate::pps::Pps,
    ) -> Result<()> {
        // Parse slice_data — common to I / P / B paths.
        let sd = slice_data::parse_slice_data(rbsp, cursor.0, cursor.1, header, sps, pps)
            .map_err(|e| Error::invalid(format!("h264 slice_data: {e}")))?;

        let in_progress = self
            .in_progress
            .as_mut()
            .expect("in_progress must have been seeded by handle_slice");

        // Update the reference bit for the whole picture if any slice
        // is a reference slice.
        if nal_ref_idc != 0 {
            in_progress.is_reference = true;
        }

        let current_structure = in_progress.structure;
        let current_bottom = matches!(current_structure, PicStructure::BottomField);
        let current_is_field = header.field_pic_flag;
        let pic_order_cnt = in_progress.poc.pic_order_cnt;
        let is_idr = in_progress.is_idr;

        // Build per-slice RefPicList0 / RefPicList1.
        let (list0, list1) = if is_idr {
            (Vec::new(), Vec::new())
        } else {
            let max_frame_num = 1u32 << (sps.log2_max_frame_num_minus4 + 4);
            let (mut l0, mut l1) = match header.slice_type {
                SliceType::P | SliceType::SP => (
                    ref_list::init_ref_pic_list_p(
                        &self.dpb_entries,
                        header.frame_num,
                        max_frame_num,
                        current_structure,
                        current_bottom,
                    ),
                    Vec::new(),
                ),
                SliceType::B => ref_list::init_ref_pic_lists_b(
                    &self.dpb_entries,
                    pic_order_cnt,
                    current_structure,
                    current_bottom,
                ),
                SliceType::I | SliceType::SI => (Vec::new(), Vec::new()),
            };

            if header.slice_type.has_list_0() {
                let ops_l0: Vec<RplmOp> = header
                    .ref_pic_list_modification
                    .modifications_l0
                    .iter()
                    .map(slice_rplm_to_ref_rplm)
                    .collect();
                ref_list::modify_ref_pic_list(
                    &mut l0,
                    &ops_l0,
                    &self.dpb_entries,
                    header.num_ref_idx_l0_active_minus1 + 1,
                    header.frame_num,
                    max_frame_num,
                    current_is_field,
                    current_bottom,
                );
            }
            if header.slice_type.has_list_1() {
                let ops_l1: Vec<RplmOp> = header
                    .ref_pic_list_modification
                    .modifications_l1
                    .iter()
                    .map(slice_rplm_to_ref_rplm)
                    .collect();
                ref_list::modify_ref_pic_list(
                    &mut l1,
                    &ops_l1,
                    &self.dpb_entries,
                    header.num_ref_idx_l1_active_minus1 + 1,
                    header.frame_num,
                    max_frame_num,
                    current_is_field,
                    current_bottom,
                );
            }

            (l0, l1)
        };

        // §8.4.* — pixel reconstruction into the in-progress picture.
        // Stamp the current picture's POC + frame_num so §8.4.1.2.3
        // temporal-direct derivation can consult it. Idempotent across
        // the slices of a coded picture (§7.4.1.2.4 requires POC
        // consistency).
        in_progress.pic.pic_order_cnt = in_progress.poc.pic_order_cnt;
        in_progress.pic.frame_num = header.frame_num;

        // §8.4.1.2.3 — precompute POCs + long-term flags for the
        // slice's RefPicList0. A later B-slice uses this picture as
        // the colocated picture and invokes MapColToList0 which
        // requires picture-identity lookup (by POC) back into the
        // list that was active when this picture was decoded.
        let list_0_pocs: Vec<i32> = list0
            .iter()
            .map(|&key| {
                self.ref_store
                    .get_by_key(key)
                    .map(|p| p.pic_order_cnt)
                    .unwrap_or(0)
            })
            .collect();
        let list_0_longterm: Vec<bool> = list0
            .iter()
            .map(|&key| {
                self.dpb_entries
                    .iter()
                    .find(|e| e.dpb_key == key)
                    .map(|e| e.is_long_term())
                    .unwrap_or(false)
            })
            .collect();
        let list_1_pocs: Vec<i32> = list1
            .iter()
            .map(|&key| {
                self.ref_store
                    .get_by_key(key)
                    .map(|p| p.pic_order_cnt)
                    .unwrap_or(0)
            })
            .collect();
        let list_1_longterm: Vec<bool> = list1
            .iter()
            .map(|&key| {
                self.dpb_entries
                    .iter()
                    .find(|e| e.dpb_key == key)
                    .map(|e| e.is_long_term())
                    .unwrap_or(false)
            })
            .collect();
        // Idempotent across slices: once set for a picture, only
        // update if still empty (shared lists across slices of one
        // primary coded picture have the same POCs — but RPLM may
        // differ. Using the first non-empty snapshot matches the
        // common "first slice wins" convention for primary MB 0).
        if in_progress.pic.ref_list_0_pocs.is_empty() {
            in_progress.pic.ref_list_0_pocs = list_0_pocs.clone();
            in_progress.pic.ref_list_0_longterm = list_0_longterm.clone();
            in_progress.pic.ref_list_1_pocs = list_1_pocs.clone();
            in_progress.pic.ref_list_1_longterm = list_1_longterm.clone();
        }
        let _ = list_1_pocs;
        let provider = BorrowedRefProvider {
            store: &self.ref_store,
            list_0: &list0,
            list_1: &list1,
            list_0_pocs,
            list_0_longterm,
            list_1_longterm,
        };
        reconstruct::reconstruct_slice_no_deblock(
            &sd,
            header,
            sps,
            pps,
            &provider,
            &mut in_progress.pic,
            &mut in_progress.grid,
        )
        .map_err(|e| Error::invalid(format!("h264 reconstruct: {e}")))?;

        // §7.4.4 — copy this slice's per-MB mb_field_decoding_flag
        // values into the picture-wide array so `finalize_in_progress_picture`
        // can hand the whole picture's flags to the deblocker in one shot.
        // `sd.macroblocks[i]` corresponds to macroblock address
        // `first_mb_in_slice * (1 + MbaffFrameFlag) + i` in the raster /
        // slice-group-0 walk.
        let mbaff_frame_flag = sps.mb_adaptive_frame_field_flag && !header.field_pic_flag;
        let mut curr_addr = (header.first_mb_in_slice * (1 + u32::from(mbaff_frame_flag))) as usize;
        for flag in sd.mb_field_decoding_flags.iter().copied() {
            if let Some(slot) = in_progress.mb_field_flags.get_mut(curr_addr) {
                *slot = flag;
            }
            curr_addr = curr_addr.saturating_add(1);
        }

        Ok(())
    }

    /// Complete the picture currently held in `self.in_progress`: run the
    /// §8.2.5 decoded reference picture marking, insert the picture into
    /// the DPB if it's a reference, and push the finalized `VideoFrame`
    /// through the §C.4 output bumping process. Clears `in_progress`.
    ///
    /// No-op when no picture is in progress.
    fn finalize_in_progress_picture(&mut self) -> Result<()> {
        let Some(in_progress) = self.in_progress.take() else {
            return Ok(());
        };
        let PictureInProgress {
            mut pic,
            grid,
            first_nal_unit_type: _,
            first_nal_ref_idc: _,
            first_header,
            is_reference,
            is_idr,
            poc,
            structure,
            pts,
            time_base,
            deblock_enabled,
            deblock_alpha_off,
            deblock_beta_off,
            mb_field_flags,
            sps,
            pps,
        } = in_progress;

        // §8.4.1.2.3 temporal direct needs the colocated block's MVs
        // of any B slice that references this picture. Snapshot the
        // decoded MV grid into the Picture so `ref_store` carries it
        // forward. Idempotent: overwrites whatever was in the Picture
        // before.
        snapshot_grid_into_picture(&mut pic, &grid);

        // SPS and PPS were snapshotted at the first slice's header-parse
        // time (via [`Event::Slice`]). Using the driver's current
        // `active_pps()` here would be wrong whenever a later NAL
        // overwrites `pps_by_id[id]` before the picture is finalized
        // — as happens in JVT CACQP3 where every access unit re-sends
        // PPS id 0 with a different `chroma_qp_index_offset`.

        // §8.7 — one picture-level deblocking pass, AFTER every slice of
        // this primary coded picture has populated the shared
        // Picture + MbGrid. Running it per-slice (the old behaviour)
        // would re-filter already-deblocked edges when a later slice's
        // pass revisits them with more MBs marked `available`, corrupting
        // the pixels near slice boundaries. Multi-slice pictures are
        // exercised by the JVT SVA_Base_B (CAVLC IP, 3 slices/pic) and
        // SL1_SVA_B (CAVLC IPB, 3 slices/pic) conformance streams.
        let bit_depth_y = 8 + sps.bit_depth_luma_minus8;
        let bit_depth_c = 8 + sps.bit_depth_chroma_minus8;
        let mbaff_frame_flag = sps.mb_adaptive_frame_field_flag && !first_header.field_pic_flag;
        if deblock_enabled {
            reconstruct::deblock_picture_full(
                &mut pic,
                &grid,
                deblock_alpha_off,
                deblock_beta_off,
                bit_depth_y,
                bit_depth_c,
                &pps,
                mbaff_frame_flag,
                &mb_field_flags,
            );
        }

        // §8.2.5 — decoded reference picture marking.
        let mut current_entry = DpbEntry {
            frame_num: first_header.frame_num,
            top_field_order_cnt: poc.top_field_order_cnt,
            bottom_field_order_cnt: poc.bottom_field_order_cnt,
            pic_order_cnt: poc.pic_order_cnt,
            structure,
            marking: if is_reference {
                RefMarking::ShortTerm
            } else {
                RefMarking::Unused
            },
            long_term_frame_idx: 0,
            dpb_key: self.mint_dpb_key(),
        };

        let mut mmco5_triggered = false;
        if is_reference {
            let marking = first_header.dec_ref_pic_marking.as_ref();
            let long_term_ref_flag = marking.is_some_and(|m| m.long_term_reference_flag);
            let no_output = marking.is_some_and(|m| m.no_output_of_prior_pics_flag);
            let adaptive_ops_vec: Option<Vec<RefMmcoOp>> = marking
                .and_then(|m| m.adaptive_marking.as_ref())
                .map(|ops| ops.iter().map(slice_mmco_to_ref_mmco).collect());

            mmco5_triggered = ref_list::perform_marking(
                &mut self.dpb_entries,
                &mut current_entry,
                sps.max_num_ref_frames,
                is_idr,
                long_term_ref_flag,
                no_output,
                adaptive_ops_vec.as_deref(),
                first_header.frame_num,
                1u32 << (sps.log2_max_frame_num_minus4 + 4),
            );

            self.dpb_entries
                .retain(|e| !matches!(e.marking, RefMarking::Unused));

            // §7.4.1.2.4 / §8.2.1 NOTE 1 — when the current picture
            // carries MMCO 5, its frame_num is inferred to 0 and its
            // Top/BottomFieldOrderCnt (hence PicOrderCnt) are reset
            // to post-subtraction values for *all* subsequent uses —
            // reference-picture-list construction included. Store the
            // post-reset values in the DPB so §8.2.4.1 PicNum /
            // FrameNumWrap arithmetic on later slices sees the
            // spec-sanctioned zeroed identity. The same reset must be
            // applied to the Picture that is inserted into `ref_store`
            // so that downstream POC-based queries (ref_pic_poc /
            // weighted bipred / temporal-direct) see the post-reset
            // identity too — otherwise later slices referencing this
            // picture compare against its pre-reset POC and mis-identify
            // picture identity at deblock-time.
            if mmco5_triggered {
                let temp = current_entry.pic_order_cnt;
                current_entry.frame_num = 0;
                current_entry.top_field_order_cnt -= temp;
                current_entry.bottom_field_order_cnt -= temp;
                current_entry.pic_order_cnt = current_entry
                    .top_field_order_cnt
                    .min(current_entry.bottom_field_order_cnt);
                pic.pic_order_cnt = current_entry.pic_order_cnt;
                pic.frame_num = 0;
            }

            self.ref_store.insert(current_entry.dpb_key, pic.clone());
            self.dpb_entries.push(current_entry);
        }

        if is_reference {
            self.prev_had_mmco5 = mmco5_triggered;
            // §8.2.1.1 — prevPicOrderCntLsb after an MMCO-5 picture is
            // the post-reset TopFieldOrderCnt (§8.2.1 NOTE 1 subtracts
            // tempPicOrderCnt from Top/BottomFieldOrderCnt after decode).
            self.prev_reference_top_foc = if mmco5_triggered {
                poc.top_field_order_cnt - poc.pic_order_cnt
            } else {
                0
            };
            // §8.2.5.2 — remember this reference picture's `frame_num`
            // so the next access unit can detect (and fill) any gap.
            // MMCO-5 resets the CVS, so `prev_ref_frame_num` is cleared
            // per §8.2.1 NOTE 1 / §8.2.5.4.5.
            self.prev_ref_frame_num = if mmco5_triggered {
                // Per §8.2.5.4.5 the picture that carried MMCO 5 is
                // renumbered to frame_num 0; treat its successor as if
                // it were the frame after frame_num 0.
                Some(0)
            } else {
                Some(first_header.frame_num)
            };
        }

        // `time_base` was previously stamped onto the VideoFrame for
        // downstream rescaling; the slim VideoFrame shape only carries
        // pts + planes now, so the time base lives on the stream's
        // CodecParameters instead. Bind to `_` to keep the destructure
        // total and document the intent.
        let _ = time_base;
        let vf = picture_to_video_frame(&pic, pts);

        // §C.4 — at IDR / MMCO-5 drain the prior sequence.
        if is_idr || mmco5_triggered {
            for drained in self.output_dpb.flush() {
                self.ready.push_back(drained.picture);
            }
            self.output_dpb.reset();
        }

        self.ensure_output_dpb_sized(&sps);

        // §8.2.1 NOTE 1 — after decoding of the MMCO-5 picture:
        //   tempPicOrderCnt = PicOrderCnt(CurrPic);
        //   TopFieldOrderCnt -= tempPicOrderCnt;
        //   BottomFieldOrderCnt -= tempPicOrderCnt;
        // For a frame (field_pic_flag == 0 and TopFOC == BotFOC) this
        // zeros both, so PicOrderCnt(CurrPic) for output ordering
        // becomes 0. Downstream POC-ordered bumping MUST see the
        // post-reset POC — otherwise the MMCO-5 picture is sorted by
        // its pre-reset POC (ordinarily the largest in the outgoing
        // CVS) and later CVS pictures, which use POC starting from 0
        // again, are bumped ahead of it.
        let output_poc = if mmco5_triggered {
            let temp = poc.pic_order_cnt;
            let top_after = poc.top_field_order_cnt - temp;
            let bot_after = poc.bottom_field_order_cnt - temp;
            if first_header.field_pic_flag && first_header.bottom_field_flag {
                bot_after
            } else if first_header.field_pic_flag {
                top_after
            } else {
                top_after.min(bot_after)
            }
        } else {
            poc.pic_order_cnt
        };

        let entry = OutputEntry {
            picture: vf,
            pic_order_cnt: output_poc,
            frame_num: first_header.frame_num,
            needed_for_output: true,
        };
        if let Some(bumped) = self.output_dpb.push(entry) {
            self.ready.push_back(bumped.picture);
        }

        Ok(())
    }

    /// Resize the output DPB capacity from the active SPS's VUI
    /// `bitstream_restriction` (§E.2.1) when present, else fall back
    /// to the Annex A Table A-1 per-level default derived from
    /// `MaxDpbMbs` (§A.3.1 item h). Only rebuilds the internal queue
    /// when the capacity would actually change — the common case of a
    /// steady SPS is a cheap no-op.
    fn ensure_output_dpb_sized(&mut self, sps: &Sps) {
        let (reorder, buffering) = output_dpb_sizing(sps);
        if self.output_dpb.max_num_reorder_frames != reorder
            || self.output_dpb.max_dec_frame_buffering != buffering
        {
            // The DpbOutput has no "resize" primitive, so move any
            // entries currently queued into a fresh DpbOutput with the
            // new capacity. Using push() on the new queue preserves
            // §C.4 bumping semantics — if the new cap is smaller, the
            // excess is pushed to `ready` in POC order.
            let pending = self.output_dpb.flush();
            let mut new_dpb = DpbOutput::<VideoFrame>::new(reorder, buffering);
            // Iterate in the POC-ascending order flush() produced.
            for e in pending {
                if let Some(bumped) = new_dpb.push(e) {
                    self.ready.push_back(bumped.picture);
                }
            }
            self.output_dpb = new_dpb;
        }
    }

    fn mint_dpb_key(&mut self) -> u32 {
        let k = self.next_dpb_key;
        self.next_dpb_key = self.next_dpb_key.wrapping_add(1);
        k
    }

    /// §8.2.5.2 — synthesise "non-existing" short-term reference frames
    /// to fill a gap between `PrevRefFrameNum` and the current picture's
    /// `frame_num`. Each synthetic frame advances the sliding-window
    /// eviction state (§8.2.5.3) and the POC-state `prev_frame_num` for
    /// `pic_order_cnt_type` 1/2, matching the encoder's assumption that
    /// those references exist in the DPB when it emits RPLM / MMCO ops
    /// targeting their PicNum values.
    ///
    /// The synthetic picture's pixel samples are marked "not available
    /// for prediction of other pictures" per §8.2.5.2; we still insert
    /// a placeholder [`Picture`] in the ref store so any erroneous
    /// motion-compensation reference produces neutral (gray) output
    /// rather than a crash. Conformant streams do not actually sample
    /// a non-existing reference's pixels.
    fn fill_frame_num_gap(&mut self, sps: &Sps, current_frame_num: u32) -> Result<()> {
        let max_frame_num: u32 = 1u32 << (sps.log2_max_frame_num_minus4 + 4);
        // §8.2.5.2 is a no-op when the previous reference picture is
        // the immediate predecessor (or when no reference has yet been
        // seen — we let the first non-IDR reference picture seed
        // `prev_ref_frame_num`).
        let Some(prev) = self.prev_ref_frame_num else {
            return Ok(());
        };
        let mut expected = (prev + 1) % max_frame_num;
        if expected == current_frame_num {
            return Ok(());
        }

        let width_samples = sps.pic_width_in_mbs() * 16;
        let height_samples = sps.frame_height_in_mbs() * 16;
        let chroma_array_type = sps.chroma_array_type();
        let bit_depth_y = sps.bit_depth_luma_minus8 + 8;
        let bit_depth_c = sps.bit_depth_chroma_minus8 + 8;

        // Guard against a runaway loop from a bogus `current_frame_num`;
        // the spec allows up to MaxFrameNum iterations.
        let mut iterations: u32 = 0;
        while expected != current_frame_num && iterations < max_frame_num {
            // §8.2.5.2 — a neutral placeholder picture. Samples are
            // mid-grey (2^(bit_depth-1)) because the spec only guarantees
            // "not available for prediction"; mid-grey keeps any
            // accidental reference from producing wildly out-of-range
            // residuals.
            let pic = gray_picture(
                width_samples,
                height_samples,
                chroma_array_type,
                bit_depth_y,
                bit_depth_c,
            );

            // §8.2.1.3 POC type 2 (and also §8.2.1.2 type 1) treat each
            // non-existing frame as a reference picture with its own
            // `frame_num`, so step the POC state's `prev_frame_num` and
            // `prev_frame_num_offset` exactly as a real reference would.
            if matches!(sps.pic_order_cnt_type, 1 | 2) {
                let prev_fnum_offset = self.poc_state.prev_frame_num_offset;
                let new_offset = if self.poc_state.prev_frame_num > expected {
                    prev_fnum_offset + max_frame_num as i64
                } else {
                    prev_fnum_offset
                };
                self.poc_state.prev_frame_num_offset = new_offset;
            }
            self.poc_state.prev_frame_num = expected;

            // Derive a POC for the non-existing frame so RefPicList
            // construction for B slices (unused here but kept correct
            // in general) has a consistent ordering key.
            let (top_foc, bot_foc, poc_value) = non_existing_poc(sps, &self.poc_state, expected);

            // §8.2.5.2 — apply the sliding window before adding, so the
            // DPB never exceeds `max_num_ref_frames` short+long refs.
            ref_list::sliding_window_marking(
                &mut self.dpb_entries,
                sps.max_num_ref_frames,
                expected,
                max_frame_num,
            );
            self.dpb_entries
                .retain(|e| !matches!(e.marking, RefMarking::Unused));

            let key = self.mint_dpb_key();
            let entry = DpbEntry {
                frame_num: expected,
                top_field_order_cnt: top_foc,
                bottom_field_order_cnt: bot_foc,
                pic_order_cnt: poc_value,
                structure: PicStructure::Frame,
                marking: RefMarking::ShortTerm,
                long_term_frame_idx: 0,
                dpb_key: key,
            };
            self.ref_store.insert(key, pic);
            self.dpb_entries.push(entry);

            self.prev_ref_frame_num = Some(expected);
            expected = (expected + 1) % max_frame_num;
            iterations += 1;
        }

        Ok(())
    }
}

/// §8.2.1 — compute `(TopFieldOrderCnt, BottomFieldOrderCnt, PicOrderCnt)`
/// for a synthetic non-existing reference frame. Only type 2 and type 1
/// need real values; type 0 is left at (0, 0, 0) since non-existing
/// frames with type 0 are never used to predict other pictures' POCs.
fn non_existing_poc(sps: &Sps, state: &PocState, frame_num: u32) -> (i32, i32, i32) {
    match sps.pic_order_cnt_type {
        2 => {
            // §8.2.1.3 treats a reference picture as
            // `2 * (FrameNumOffset + frame_num)`. We use the already-
            // updated `prev_frame_num_offset` because the caller has
            // stepped it for this non-existing frame.
            let poc = 2 * (state.prev_frame_num_offset + frame_num as i64);
            let poc = poc.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            (poc, poc, poc)
        }
        1 => {
            // A conservative stand-in — type-1 streams rarely hit the
            // gap path and we do not need pixel-exact reproduction of
            // the expectedPicOrderCnt arithmetic here.
            (0, 0, 0)
        }
        _ => (0, 0, 0),
    }
}

/// §8.2.5.2 — mid-grey placeholder picture for a synthetic non-existing
/// reference frame. Samples are set to `2^(bit_depth - 1)` per plane so
/// that accidental motion-compensation references produce neutral
/// output instead of zeroes (which would bias the residual).
fn gray_picture(
    width_samples: u32,
    height_samples: u32,
    chroma_array_type: u32,
    bit_depth_y: u32,
    bit_depth_c: u32,
) -> Picture {
    let mut p = Picture::new(
        width_samples,
        height_samples,
        chroma_array_type,
        bit_depth_y,
        bit_depth_c,
    );
    let grey_y: i32 = 1 << (bit_depth_y.saturating_sub(1));
    let grey_c: i32 = 1 << (bit_depth_c.saturating_sub(1));
    for v in p.luma.iter_mut() {
        *v = grey_y;
    }
    for v in p.cb.iter_mut() {
        *v = grey_c;
    }
    for v in p.cr.iter_mut() {
        *v = grey_c;
    }
    p
}

/// Per-slice [`RefPicProvider`] that borrows pictures from a long-running
/// [`RefPicStore`] but carries its own RefPicList0 / RefPicList1 key
/// arrays. Avoids cloning every DPB Picture on every slice.
struct BorrowedRefProvider<'a> {
    store: &'a RefPicStore,
    list_0: &'a [u32],
    list_1: &'a [u32],
    /// §8.4.1.2.3 — precomputed POCs of the pictures in `list_0`,
    /// supplied to the temporal-direct MapColToList0 derivation.
    list_0_pocs: Vec<i32>,
    /// §8.4.1.2.3 — long-term flag parallel to `list_0_pocs`.
    list_0_longterm: Vec<bool>,
    /// §8.4.1.2.2 — long-term flag for RefPicList1. Spatial-direct
    /// mode suppresses colZeroFlag when `RefPicList1[0]` is long-term.
    list_1_longterm: Vec<bool>,
}

impl RefPicProvider for BorrowedRefProvider<'_> {
    fn ref_pic(&self, list: u8, idx: u32) -> Option<&Picture> {
        let keys = match list {
            0 => self.list_0,
            1 => self.list_1,
            _ => return None,
        };
        let key = *keys.get(idx as usize)?;
        self.store.get_by_key(key)
    }

    fn ref_list_0_pocs(&self) -> &[i32] {
        &self.list_0_pocs
    }

    fn ref_list_0_longterm(&self) -> &[bool] {
        &self.list_0_longterm
    }

    fn ref_list_1_longterm(&self) -> &[bool] {
        &self.list_1_longterm
    }
}

/// Derive the `(max_num_reorder_frames, max_dec_frame_buffering)`
/// pair for sizing [`DpbOutput`] from the active SPS.
///
/// Preferred source is the VUI `bitstream_restriction` block
/// (§E.2.1). When it's absent we infer the spec default per
/// §A.3.1:
///   1. `max_dec_frame_buffering` ← Annex A Table A-1 per-level
///      cap, i.e. Min(MaxDpbMbs / (PicWidthInMbs * FrameHeightInMbs), 16)
///      (§A.3.1 item h).
///   2. `max_num_reorder_frames` ← inferred to `max_dec_frame_buffering`
///      when the VUI `bitstream_restriction` flag is absent
///      (§A.3.1 item j / §E.2.1 — unspecified defaults to maximum).
///      A smaller upper bound breaks streams that issue an MMCO-5
///      deep into a POC range because §C.4 bumps pictures by POC in
///      a single CVS, and clipping the reorder window prematurely
///      forces out-of-POC-order emission when a later CVS reuses
///      the low POC range.
///
/// Both values floor at 1 so `DpbOutput::push` can always bump when
/// the queue is full (cap of 0 would deadlock the queue).
fn output_dpb_sizing(sps: &Sps) -> (u32, u32) {
    if let Some(br) = sps
        .vui
        .as_ref()
        .and_then(|v| v.bitstream_restriction.as_ref())
    {
        // §E.2.1 — max_dec_frame_buffering must be ≥ max_num_reorder_frames.
        // Clamp to at least 1 so the queue can ever bump.
        let reorder = br.max_num_reorder_frames;
        let buffering = br.max_dec_frame_buffering.max(reorder).max(1);
        return (reorder, buffering);
    }

    // Fallback — level-derived Annex A Table A-1 / §A.3.1 item h cap.
    let max_dpb_mbs = max_dpb_mbs_for_level(sps.level_idc);
    let pic_size_mbs = sps
        .pic_width_in_mbs()
        .saturating_mul(sps.frame_height_in_mbs())
        .max(1);
    let max_dpb_frames = (max_dpb_mbs / pic_size_mbs).clamp(1, 16);

    // §A.3.1 item j — when bitstream_restriction is absent, the
    // spec's default for `max_num_reorder_frames` is the full
    // buffering cap. Under-sizing it would clamp POC reordering and
    // bump pictures in decode order instead of POC order.
    (max_dpb_frames, max_dpb_frames)
}

/// Annex A Table A-1 — `MaxDpbMbs` per `level_idc`.
///
/// `level_idc` is the raw `u8` from §7.4.2.1.1; intermediate levels
/// (e.g. 2.1) are encoded as `10 * <level>` so `21 → 2.1`. Level 1b
/// is signalled via `constraint_set3_flag` alongside `level_idc == 11`
/// which carries MaxDpbMbs = 396 (same as level 1.1), so we treat 11
/// conservatively as 1.1.
///
/// Unknown `level_idc` values fall through to the lowest bucket
/// (MaxDpbMbs = 396) to minimise over-allocation while still
/// permitting at least one reference picture.
fn max_dpb_mbs_for_level(level_idc: u8) -> u32 {
    match level_idc {
        10 => 396,   // level 1
        11 => 900,   // level 1.1 (also 1b, per the conservative note above)
        12 => 2_376, // level 1.2
        13 => 2_376, // level 1.3
        20 => 2_376, // level 2
        21 => 4_752, // level 2.1
        22 => 8_100, // level 2.2
        30 => 8_100, // level 3
        31 => 18_000,
        32 => 20_480,
        40 => 32_768,
        41 => 32_768,
        42 => 34_816,
        50 => 110_400,
        51 => 184_320,
        52 => 184_320,
        60 => 696_320,
        61 => 696_320,
        62 => 696_320,
        _ => 396, // conservative fallback
    }
}

/// §7.4.3 — map `(field_pic_flag, bottom_field_flag)` into the
/// [`PicStructure`] that `ref_list` / DPB bookkeeping consumes.
fn pic_structure_from_flags(field_pic_flag: bool, bottom_field_flag: bool) -> PicStructure {
    match (field_pic_flag, bottom_field_flag) {
        (false, _) => PicStructure::Frame,
        (true, false) => PicStructure::TopField,
        (true, true) => PicStructure::BottomField,
    }
}

/// Project an [`Sps`] into the subset of fields [`derive_poc`] needs.
fn make_poc_sps(sps: &Sps) -> PocSps {
    PocSps {
        pic_order_cnt_type: sps.pic_order_cnt_type,
        log2_max_frame_num_minus4: sps.log2_max_frame_num_minus4,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
        delta_pic_order_always_zero_flag: sps.delta_pic_order_always_zero_flag,
        offset_for_non_ref_pic: sps.offset_for_non_ref_pic,
        offset_for_top_to_bottom_field: sps.offset_for_top_to_bottom_field,
        num_ref_frames_in_pic_order_cnt_cycle: sps.num_ref_frames_in_pic_order_cnt_cycle,
        offset_for_ref_frame: sps.offset_for_ref_frame.clone(),
        frame_mbs_only_flag: sps.frame_mbs_only_flag,
    }
}

/// Convert §7.3.3.1 RPLM op to the §8.2.4.3 ref_list equivalent.
fn slice_rplm_to_ref_rplm(op: &SliceRplmOp) -> RplmOp {
    match *op {
        SliceRplmOp::Subtract(v) => RplmOp::Subtract(v),
        SliceRplmOp::Add(v) => RplmOp::Add(v),
        SliceRplmOp::LongTerm(v) => RplmOp::LongTerm(v),
    }
}

/// Convert §7.3.3.3 MMCO op to the §8.2.5.4 ref_list equivalent.
fn slice_mmco_to_ref_mmco(op: &SliceMmcoOp) -> RefMmcoOp {
    match *op {
        SliceMmcoOp::MarkShortTermUnused(v) => RefMmcoOp::MarkShortTermUnused(v),
        SliceMmcoOp::MarkLongTermUnused(v) => RefMmcoOp::MarkLongTermUnused(v),
        SliceMmcoOp::AssignLongTerm(d, ltfi) => RefMmcoOp::AssignLongTerm(d, ltfi),
        SliceMmcoOp::SetMaxLongTermIdx(v) => RefMmcoOp::SetMaxLongTermIdx(v),
        SliceMmcoOp::MarkAllUnused => RefMmcoOp::MarkAllUnused,
        SliceMmcoOp::AssignCurrentLongTerm(v) => RefMmcoOp::AssignCurrentLongTerm(v),
    }
}

impl Decoder for H264CodecDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.pending_pts = packet.pts;
        self.pending_time_base = packet.time_base;
        let data = packet.data.clone();
        match self.length_size {
            Some(n) => {
                // AVCC framing — walk length-prefixed NAL units and
                // hand each to the driver.
                let mut i = 0usize;
                let n = n as usize;
                while i < data.len() {
                    if i + n > data.len() {
                        return Err(Error::invalid("h264: AVCC length prefix truncated"));
                    }
                    let mut len = 0usize;
                    for k in 0..n {
                        len = (len << 8) | data[i + k] as usize;
                    }
                    i += n;
                    if i + len > data.len() {
                        return Err(Error::invalid("h264: AVCC NAL payload truncated"));
                    }
                    let ev = self
                        .driver
                        .process_nal(&data[i..i + len])
                        .map_err(|e| Error::invalid(format!("h264 NAL parse: {e}")))?;
                    // Ignore per-slice errors so the stream can keep
                    // feeding. Real errors in parse step 1 (NAL parse)
                    // already aborted above; these are reconstruction
                    // errors the caller may want to log, but we drop
                    // them for now to avoid killing the stream on one
                    // broken slice (e.g. unsupported MB type).
                    if let Err(e) = self.handle_event(ev) {
                        eprintln!("h264 slice skipped: {e}");
                    }
                    i += len;
                }
                Ok(())
            }
            None => {
                // Annex B framing. Collect events first so the driver
                // borrow ends before we recurse into handle_event
                // (which re-borrows self.driver to read active_sps /
                // pps).
                let events: Vec<_> = self.driver.process_annex_b(&data).collect();
                for ev in events {
                    match ev {
                        Ok(ev) => {
                            if let Err(e) = self.handle_event(ev) {
                                eprintln!("h264 slice skipped: {e}");
                            }
                        }
                        Err(e) => {
                            return Err(Error::invalid(format!("h264 NAL parse: {e}")));
                        }
                    }
                }
                Ok(())
            }
        }
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        // §C.4 — bumped / already-released pictures come out first in
        // the order the bumping process produced them.
        if let Some(vf) = self.ready.pop_front() {
            return Ok(Frame::Video(vf));
        }
        // Try a conservative bump on the output DPB. Per §C.4 / the
        // `pop_ready` semantics, this only yields a picture when the
        // queue is genuinely over capacity (mid-stream backpressure).
        if let Some(bumped) = self.output_dpb.pop_ready() {
            return Ok(Frame::Video(bumped.picture));
        }
        // EOF: drain everything remaining in POC-ascending order
        // (§C.4 "no_output_of_prior_pics_flag == 0" / end-of-stream).
        // We drain once into `ready` and then hand out one by one,
        // so subsequent receive_frame calls pull from `ready` above
        // until exhausted.
        if self.eof {
            let drained = self.output_dpb.flush();
            if drained.is_empty() {
                return Err(Error::Eof);
            }
            for e in drained {
                self.ready.push_back(e.picture);
            }
            // Safe to unwrap: we just confirmed non-empty drain above.
            if let Some(vf) = self.ready.pop_front() {
                return Ok(Frame::Video(vf));
            }
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        // §7.4.1.2 — close any picture we've been assembling so it reaches
        // the DPB + output queue before the caller drains at EOF.
        if let Err(e) = self.finalize_in_progress_picture() {
            eprintln!("h264 flush: final picture skipped: {e}");
        }
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.driver = H264Driver::new();
        self.last_slice = None;
        self.eof = false;
        // §C.4 — wipe the output queue and any picture that was
        // already bumped but not yet consumed.
        self.output_dpb.reset();
        self.ready.clear();
        self.pending_pts = None;
        self.ref_store = RefPicStore::new();
        self.dpb_entries.clear();
        self.poc_state = PocState::default();
        self.next_dpb_key = 0;
        self.prev_had_mmco5 = false;
        self.prev_reference_top_foc = 0;
        self.prev_ref_frame_num = None;
        // Drop any picture currently being assembled — reset implies we
        // discard in-flight state, not deliver it.
        self.in_progress = None;
        Ok(())
    }
}

/// §8.4.1.2.3 — snapshot the per-4x4-block motion data from the
/// freshly-decoded picture's [`MbGrid`] into its [`Picture`] so
/// subsequent B slices can consult the colocated block for temporal
/// direct mode.
///
/// The `Picture`'s sample buffer already contains the decoded samples;
/// this only populates the optional mv / refIdx / intra grids. Called
/// by `finalize_in_progress_picture` whether the picture is a
/// reference or not — non-reference pictures never feed a later B
/// slice, but the per-picture cost of the copy is tiny and it keeps
/// the code path uniform.
fn snapshot_grid_into_picture(pic: &mut Picture, grid: &MbGrid) {
    let w = grid.width_in_mbs as usize;
    let h = grid.height_in_mbs as usize;
    let nmb = w * h;
    pic.mb_width_in_picture = grid.width_in_mbs;
    pic.mv_l0_grid = vec![(0i16, 0i16); nmb * 16];
    pic.mv_l1_grid = vec![(0i16, 0i16); nmb * 16];
    pic.ref_idx_l0_grid = vec![-1i8; nmb * 4];
    pic.ref_idx_l1_grid = vec![-1i8; nmb * 4];
    pic.is_intra_grid = vec![false; nmb];
    for (addr, info) in grid.info.iter().enumerate() {
        let base_mv = addr * 16;
        let base_r = addr * 4;
        for blk4 in 0..16 {
            pic.mv_l0_grid[base_mv + blk4] = info.mv_l0[blk4];
            pic.mv_l1_grid[base_mv + blk4] = info.mv_l1[blk4];
        }
        for blk8 in 0..4 {
            pic.ref_idx_l0_grid[base_r + blk8] = info.ref_idx_l0[blk8];
            pic.ref_idx_l1_grid[base_r + blk8] = info.ref_idx_l1[blk8];
        }
        pic.is_intra_grid[addr] = info.is_intra;
    }
}

/// Convert a reconstructed [`Picture`] to a [`VideoFrame`]. Samples are
/// clamped to 8-bit (0..=255) for now; higher bit depths will get
/// dedicated pixel formats later.
///
/// The slim `VideoFrame` shape only carries `pts` + `planes`; the pixel
/// format / resolution / time_base live on the stream's
/// [`CodecParameters`] (set by the decoder from the SPS).
fn picture_to_video_frame(pic: &Picture, pts: Option<i64>) -> VideoFrame {
    let w = pic.width_in_samples as usize;
    let cw = pic.chroma_width() as usize;

    let clamp_u8 = |s: i32| s.clamp(0, 255) as u8;

    let luma: Vec<u8> = pic.luma.iter().map(|&s| clamp_u8(s)).collect();
    let mut planes = vec![VideoPlane {
        stride: w,
        data: luma,
    }];
    if pic.chroma_array_type != 0 {
        let cb: Vec<u8> = pic.cb.iter().map(|&s| clamp_u8(s)).collect();
        let cr: Vec<u8> = pic.cr.iter().map(|&s| clamp_u8(s)).collect();
        planes.push(VideoPlane {
            stride: cw,
            data: cb,
        });
        planes.push(VideoPlane {
            stride: cw,
            data: cr,
        });
    }

    VideoFrame { pts, planes }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the §C.2.2 / §C.4 wiring through
    //! [`H264CodecDecoder`]. These tests bypass `handle_slice` and push
    //! fabricated [`VideoFrame`] + POC pairs straight into the output
    //! path so we exercise only the DPB-output plumbing without
    //! needing a real H.264 bitstream with B-frames in the samples
    //! dir. The bumping-logic correctness itself is already covered by
    //! `crate::dpb_output::tests`; what we assert here is that the
    //! wrapper delivers entries in POC order through `receive_frame`,
    //! honours IDR resets (§C.4), and flushes at EOF.
    //!
    //! Spec references:
    //! * §C.2.2 — "Storage and output of decoded pictures"
    //! * §C.4   — "Bumping process"
    //! * §8.2.5.4 — MMCO op 5 resets the DPB
    //! * Annex A Table A-1 — level-derived MaxDpbMbs defaults
    use super::*;
    use crate::dpb_output::OutputEntry;

    /// Build a tiny VideoFrame so tests can track individual pictures
    /// without carrying real pixel data.
    fn vf(tag: u8) -> VideoFrame {
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 1,
                data: vec![tag],
            }],
        }
    }

    /// Test access: pull the single-byte "tag" out of a VideoFrame
    /// planted by `vf`.
    fn vf_tag(f: &Frame) -> u8 {
        match f {
            Frame::Video(v) => v.planes[0].data[0],
            _ => panic!("non-video frame"),
        }
    }

    fn push_entry(dec: &mut H264CodecDecoder, tag: u8, poc: i32, frame_num: u32) {
        let entry = OutputEntry {
            picture: vf(tag),
            pic_order_cnt: poc,
            frame_num,
            needed_for_output: true,
        };
        if let Some(bumped) = dec.output_dpb.push(entry) {
            dec.ready.push_back(bumped.picture);
        }
    }

    /// §C.4 — with `max_num_reorder_frames == 2`, feeding a decode
    /// order that reorders POC mid-stream must eventually deliver
    /// every picture in POC-ascending order. Mid-stream order depends
    /// on when the bumping process fires (queue-full threshold); the
    /// end-of-stream flush cleans up the tail.
    ///
    /// Trace (cap = 2, bump runs BEFORE each insertion when len == cap):
    ///   push (POC 0, tag 10): queue=[0]
    ///   push (POC 4, tag 11): queue=[0,4]
    ///   push (POC 2, tag 12): bump lowest POC 0 (tag 10) → ready;
    ///                         queue=[4,2]
    ///   push (POC 1, tag 13): bump lowest POC 2 (tag 12) → ready;
    ///                         queue=[4,1]
    ///   push (POC 3, tag 14): bump lowest POC 1 (tag 13) → ready;
    ///                         queue=[4,3]
    ///   flush: sorted ascending → [POC 3 (tag 14), POC 4 (tag 11)]
    ///
    /// Ready drain: 10, 12, 13. Flush: 14, 11. Total: 10, 12, 13, 14, 11.
    /// This is NOT strictly POC-ascending because the bumping process
    /// is conservative — it emits the lowest-POC entry *currently
    /// queued* when capacity is reached, not the lowest POC across
    /// the whole stream. That matches §C.4's real-decoder behaviour.
    /// For a genuinely ascending output the bitstream must give the
    /// decoder enough slack (i.e. a `max_num_reorder_frames` large
    /// enough to hold every picture that could reorder past the one
    /// being bumped).
    #[test]
    fn reorder_with_small_dpb_matches_conservative_bumping() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        dec.output_dpb = DpbOutput::<VideoFrame>::new(2, 3);

        push_entry(&mut dec, 10, 0, 0); // IDR
        push_entry(&mut dec, 11, 4, 1); // P
        push_entry(&mut dec, 12, 2, 2); // B
        push_entry(&mut dec, 13, 1, 3); // B
        push_entry(&mut dec, 14, 3, 4); // B

        dec.flush().expect("flush");

        let mut tags = Vec::new();
        while let Ok(f) = dec.receive_frame() {
            tags.push(vf_tag(&f));
        }

        assert_eq!(tags, vec![10, 12, 13, 14, 11]);
    }

    /// §C.4 — with a reorder window that *is* large enough to hold
    /// every out-of-order picture, the flush at EOF produces the
    /// fully POC-ascending output order expected for display.
    #[test]
    fn reorder_with_sufficient_dpb_yields_strictly_ascending_poc() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        // Cap = 5 ≥ number of pictures → nothing bumps mid-stream,
        // every picture goes through the end-of-stream flush in POC
        // order.
        dec.output_dpb = DpbOutput::<VideoFrame>::new(5, 5);

        // Same IPBBB decode order as above.
        push_entry(&mut dec, 10, 0, 0); // IDR
        push_entry(&mut dec, 11, 4, 1); // P
        push_entry(&mut dec, 12, 2, 2); // B
        push_entry(&mut dec, 13, 1, 3); // B
        push_entry(&mut dec, 14, 3, 4); // B

        dec.flush().expect("flush");

        let mut tags = Vec::new();
        let mut pocs = Vec::new();
        while let Ok(f) = dec.receive_frame() {
            tags.push(vf_tag(&f));
            // Reconstruct POC from our tagging scheme (tag -> poc):
            // 10->0, 11->4, 12->2, 13->1, 14->3.
            let poc = match vf_tag(&f) {
                10 => 0,
                11 => 4,
                12 => 2,
                13 => 1,
                14 => 3,
                _ => unreachable!(),
            };
            pocs.push(poc);
        }

        // Expected POC-ascending order: 0, 1, 2, 3, 4 → tags 10, 13, 12, 14, 11.
        assert_eq!(tags, vec![10, 13, 12, 14, 11]);
        assert_eq!(pocs, vec![0, 1, 2, 3, 4]);
    }

    /// §C.4 — a full flush drains the DPB in POC order at EOF.
    #[test]
    fn flush_at_eof_drains_in_poc_order() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        dec.output_dpb = DpbOutput::<VideoFrame>::new(8, 8);

        // Decode order: [POC 3, POC 1, POC 2] — nothing bumped mid-stream
        // because we stay below capacity.
        push_entry(&mut dec, 0xA0, 3, 0);
        push_entry(&mut dec, 0xA1, 1, 1);
        push_entry(&mut dec, 0xA2, 2, 2);

        // Before flush(), nothing is over capacity → receive_frame gets
        // NeedMore.
        assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));

        dec.flush().expect("flush");

        let tags: Vec<u8> =
            std::iter::from_fn(|| dec.receive_frame().ok().map(|f| vf_tag(&f))).collect();
        // POC ascending: 1, 2, 3 → tags 0xA1, 0xA2, 0xA0.
        assert_eq!(tags, vec![0xA1, 0xA2, 0xA0]);
    }

    /// §C.4 — at EOF with an empty queue, `receive_frame` returns
    /// `Error::Eof` not `NeedMore`.
    #[test]
    fn eof_on_empty_queue_after_flush() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        dec.flush().expect("flush");
        assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    }

    /// §C.4 — `receive_frame` returns `NeedMore` mid-stream when the
    /// queue is under capacity.
    #[test]
    fn need_more_before_any_push() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
    }

    /// §C.4 — the pre-IDR drain: when a new IDR lands, any pictures
    /// still pending from the previous coded video sequence must be
    /// delivered in POC order before the IDR itself.
    ///
    /// We simulate this by pushing a few entries, then doing the same
    /// flush-into-ready + reset dance `handle_slice` does on an IDR
    /// boundary, then pushing the new IDR. The POC counter restarts
    /// from 0 on IDR, but the old sequence's pictures still need to
    /// come out first.
    #[test]
    fn idr_drains_pending_pictures_in_poc_order() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        dec.output_dpb = DpbOutput::<VideoFrame>::new(4, 4);

        // Sequence 1: POCs 0, 4, 2, 1 — four frames queued, none bumped.
        push_entry(&mut dec, 1, 0, 0);
        push_entry(&mut dec, 2, 4, 1);
        push_entry(&mut dec, 3, 2, 2);
        push_entry(&mut dec, 4, 1, 3);

        // Mimic `handle_slice`'s IDR branch: drain pending into `ready`
        // then reset the output DPB.
        for drained in dec.output_dpb.flush() {
            dec.ready.push_back(drained.picture);
        }
        dec.output_dpb.reset();

        // IDR at POC 0 kicks off sequence 2.
        push_entry(&mut dec, 100, 0, 0);

        // EOF drains the IDR itself too.
        dec.flush().expect("flush");

        let tags: Vec<u8> =
            std::iter::from_fn(|| dec.receive_frame().ok().map(|f| vf_tag(&f))).collect();
        // Sequence 1 in POC order (0, 1, 2, 4) → tags [1, 4, 3, 2]
        // followed by sequence 2's IDR (100).
        assert_eq!(tags, vec![1, 4, 3, 2, 100]);
    }

    /// §C.4 — `reset()` wipes both the output DPB and the "already
    /// bumped" queue.
    #[test]
    fn reset_clears_output_dpb_and_ready() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        dec.output_dpb = DpbOutput::<VideoFrame>::new(2, 2);
        // Fill + overflow so one entry lands in `ready`.
        push_entry(&mut dec, 1, 0, 0);
        push_entry(&mut dec, 2, 1, 1);
        push_entry(&mut dec, 3, 2, 2); // bumps lowest POC (0) → ready.
        assert_eq!(dec.ready.len(), 1);
        assert_eq!(dec.output_dpb.len(), 2);

        dec.reset().expect("reset");
        assert_eq!(dec.output_dpb.len(), 0);
        assert!(dec.ready.is_empty());
        assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
    }

    /// Annex A Table A-1 / §A.3.1 item h — when the SPS carries a VUI
    /// `bitstream_restriction` block, we honour it; otherwise we fall
    /// back to the level-derived default.
    #[test]
    fn dpb_sizing_prefers_vui() {
        use crate::sps::Sps;
        use crate::vui::{BitstreamRestriction, VuiParameters};

        // Construct a minimal SPS with a VUI bitstream_restriction
        // block that says "max 3 reorder, max 5 buffering".
        let sps = Sps {
            profile_idc: 66,
            constraint_set_flags: 0,
            level_idc: 30,
            seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            qpprime_y_zero_transform_bypass_flag: false,
            seq_scaling_matrix_present_flag: false,
            seq_scaling_lists: None,
            log2_max_frame_num_minus4: 0,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb_minus4: 0,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            offset_for_ref_frame: Vec::new(),
            max_num_ref_frames: 4,
            gaps_in_frame_num_value_allowed_flag: false,
            pic_width_in_mbs_minus1: 10,
            pic_height_in_map_units_minus1: 8,
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: true,
            frame_cropping: None,
            vui_parameters_present_flag: true,
            vui: Some(VuiParameters {
                bitstream_restriction: Some(BitstreamRestriction {
                    motion_vectors_over_pic_boundaries_flag: true,
                    max_bytes_per_pic_denom: 0,
                    max_bits_per_mb_denom: 0,
                    log2_max_mv_length_horizontal: 0,
                    log2_max_mv_length_vertical: 0,
                    max_num_reorder_frames: 3,
                    max_dec_frame_buffering: 5,
                }),
                ..Default::default()
            }),
        };
        assert_eq!(output_dpb_sizing(&sps), (3, 5));
    }

    /// Annex A Table A-1 fallback — when VUI is absent, use the
    /// level-derived MaxDpbMbs cap as `max_dec_frame_buffering`, and
    /// (per §A.3.1 item j) infer `max_num_reorder_frames` equal to
    /// that cap (maximum reorder window). Level 3.0 (level_idc == 30)
    /// has MaxDpbMbs = 8100; for a 176x144 (11x9 MB) picture that's
    /// Min(8100/99, 16) = 16 on both values.
    #[test]
    fn dpb_sizing_falls_back_to_level_default() {
        use crate::sps::Sps;

        let sps = Sps {
            profile_idc: 66,
            constraint_set_flags: 0,
            level_idc: 30, // Level 3.0 → MaxDpbMbs 8100.
            seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            qpprime_y_zero_transform_bypass_flag: false,
            seq_scaling_matrix_present_flag: false,
            seq_scaling_lists: None,
            log2_max_frame_num_minus4: 0,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb_minus4: 0,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            offset_for_ref_frame: Vec::new(),
            max_num_ref_frames: 4,
            gaps_in_frame_num_value_allowed_flag: false,
            pic_width_in_mbs_minus1: 10, // width_in_mbs = 11 (176 px)
            pic_height_in_map_units_minus1: 8, // height_in_mbs = 9 (144 px)
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: true,
            frame_cropping: None,
            vui_parameters_present_flag: false,
            vui: None,
        };
        // PicSize = 11 * 9 = 99; 8100 / 99 = 81 → capped at 16
        // (max_dec_frame_buffering). Reorder inferred to match
        // buffering per §A.3.1 item j.
        assert_eq!(output_dpb_sizing(&sps), (16, 16));
    }

    /// Annex A Table A-1 — per-level MaxDpbMbs sanity. Levels that
    /// share the same MaxDpbMbs row in Table A-1 must yield the same
    /// value.
    #[test]
    fn max_dpb_mbs_per_level_table() {
        assert_eq!(max_dpb_mbs_for_level(10), 396);
        assert_eq!(max_dpb_mbs_for_level(12), 2_376);
        assert_eq!(max_dpb_mbs_for_level(21), 4_752);
        assert_eq!(max_dpb_mbs_for_level(30), 8_100);
        assert_eq!(max_dpb_mbs_for_level(31), 18_000);
        assert_eq!(max_dpb_mbs_for_level(40), 32_768);
        assert_eq!(max_dpb_mbs_for_level(42), 34_816);
        assert_eq!(max_dpb_mbs_for_level(50), 110_400);
        assert_eq!(max_dpb_mbs_for_level(51), 184_320);
        assert_eq!(max_dpb_mbs_for_level(60), 696_320);
        // Unknown level → conservative fallback.
        assert_eq!(max_dpb_mbs_for_level(200), 396);
    }

    // -- §7.4.1.2.4 first-VCL-of-primary-coded-picture detection -------
    //
    // These tests seed an `in_progress` PictureInProgress by hand and
    // then probe `is_first_vcl_of_new_picture` with different trailing
    // slice headers. The assembly code itself is exercised end-to-end
    // by `tests/integration_multislice_assembly.rs`; these tests cover
    // the boundary-condition matrix without needing a real bitstream.

    use crate::poc::PocResult;
    use crate::ref_list::PicStructure;
    use crate::slice_header::{RefPicListModification, SliceHeader as Hdr, SliceType as ST};

    /// Minimal SPS for the seed helpers — exact field values do not
    /// matter since the seeded in-progress picture is never actually
    /// reconstructed; these tests only exercise
    /// `is_first_vcl_of_new_picture` which consults the slice header.
    fn test_sps() -> crate::sps::Sps {
        crate::sps::Sps {
            profile_idc: 66,
            constraint_set_flags: 0,
            level_idc: 10,
            seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            qpprime_y_zero_transform_bypass_flag: false,
            seq_scaling_matrix_present_flag: false,
            seq_scaling_lists: None,
            log2_max_frame_num_minus4: 0,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb_minus4: 0,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            offset_for_ref_frame: Vec::new(),
            max_num_ref_frames: 1,
            gaps_in_frame_num_value_allowed_flag: false,
            pic_width_in_mbs_minus1: 0,
            pic_height_in_map_units_minus1: 0,
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: false,
            frame_cropping: None,
            vui_parameters_present_flag: false,
            vui: None,
        }
    }

    fn test_pps() -> crate::pps::Pps {
        crate::pps::Pps {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
            entropy_coding_mode_flag: false,
            bottom_field_pic_order_in_frame_present_flag: false,
            num_slice_groups_minus1: 0,
            slice_group_map: None,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            pic_init_qp_minus26: 0,
            pic_init_qs_minus26: 0,
            chroma_qp_index_offset: 0,
            deblocking_filter_control_present_flag: false,
            constrained_intra_pred_flag: false,
            redundant_pic_cnt_present_flag: false,
            extension: None,
        }
    }

    /// Build a minimal SliceHeader for boundary-detection testing. All
    /// fields default to "non-IDR P frame at frame_num=0, POC lsb=0" —
    /// tests tweak the specific fields they want to compare.
    fn hdr_base() -> Hdr {
        Hdr {
            first_mb_in_slice: 0,
            slice_type_raw: 0,
            slice_type: ST::P,
            all_slices_same_type: false,
            pic_parameter_set_id: 0,
            colour_plane_id: 0,
            frame_num: 0,
            field_pic_flag: false,
            bottom_field_flag: false,
            idr_pic_id: 0,
            pic_order_cnt_lsb: 0,
            delta_pic_order_cnt_bottom: 0,
            delta_pic_order_cnt: [0, 0],
            redundant_pic_cnt: 0,
            direct_spatial_mv_pred_flag: false,
            num_ref_idx_active_override_flag: false,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            ref_pic_list_modification: RefPicListModification::default(),
            pred_weight_table: None,
            dec_ref_pic_marking: None,
            cabac_init_idc: 0,
            slice_qp_delta: 0,
            sp_for_switch_flag: false,
            slice_qs_delta: 0,
            disable_deblocking_filter_idc: 0,
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
            slice_group_change_cycle: 0,
        }
    }

    /// Seed a dummy PictureInProgress on a decoder so
    /// `is_first_vcl_of_new_picture` has something to compare against.
    fn seed_in_progress(dec: &mut H264CodecDecoder, nut: u8, nri: u8, header: Hdr) {
        let pic = Picture::new(16, 16, 1, 8, 8);
        let grid = MbGrid::new(1, 1);
        dec.in_progress = Some(PictureInProgress {
            pic,
            grid,
            first_nal_unit_type: nut,
            first_nal_ref_idc: nri,
            first_header: header,
            is_reference: nri != 0,
            is_idr: nut == 5,
            poc: PocResult {
                top_field_order_cnt: 0,
                bottom_field_order_cnt: 0,
                pic_order_cnt: 0,
            },
            structure: PicStructure::Frame,
            pts: None,
            time_base: TimeBase::new(1, 1),
            deblock_enabled: false,
            deblock_alpha_off: 0,
            deblock_beta_off: 0,
            mb_field_flags: Vec::new(),
            sps: test_sps(),
            pps: test_pps(),
        });
    }

    /// §7.4.1.2.4 — no picture in progress ⇒ any slice starts a new
    /// primary coded picture.
    #[test]
    fn first_vcl_when_no_picture_in_progress() {
        let dec = H264CodecDecoder::new(CodecId::new("h264"));
        assert!(dec.is_first_vcl_of_new_picture(1, 2, &hdr_base()));
    }

    /// §7.4.1.2.4 — identical header + nal_unit_type + nal_ref_idc ⇒
    /// SAME primary coded picture (continuation slice).
    #[test]
    fn same_picture_when_all_conditions_match() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        // Same everything except first_mb_in_slice (which is NOT in
        // the §7.4.1.2.4 list of differing conditions).
        let mut h = hdr_base();
        h.first_mb_in_slice = 384;
        assert!(!dec.is_first_vcl_of_new_picture(1, 2, &h));
    }

    /// §7.4.1.2.4 — `frame_num` differs ⇒ new picture.
    #[test]
    fn different_frame_num_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        let mut h = hdr_base();
        h.frame_num = 1;
        assert!(dec.is_first_vcl_of_new_picture(1, 2, &h));
    }

    /// §7.4.1.2.4 — `pic_parameter_set_id` differs ⇒ new picture.
    #[test]
    fn different_pps_id_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        let mut h = hdr_base();
        h.pic_parameter_set_id = 3;
        assert!(dec.is_first_vcl_of_new_picture(1, 2, &h));
    }

    /// §7.4.1.2.4 — `field_pic_flag` differs ⇒ new picture.
    #[test]
    fn different_field_pic_flag_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        let mut h = hdr_base();
        h.field_pic_flag = true;
        assert!(dec.is_first_vcl_of_new_picture(1, 2, &h));
    }

    /// §7.4.1.2.4 — nal_ref_idc zero-ness differs (prev ref, new
    /// non-ref) ⇒ new picture.
    #[test]
    fn different_nal_ref_idc_zero_ness_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        // Old nal_ref_idc = 2 (non-zero), new = 0.
        assert!(dec.is_first_vcl_of_new_picture(1, 0, &hdr_base()));
    }

    /// §7.4.1.2.4 — both nal_ref_idc non-zero but different value
    /// (e.g. 1 vs 2) is NOT a new picture — only the *zero-ness*
    /// matters.
    #[test]
    fn same_nal_ref_idc_nonzero_is_same_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        assert!(!dec.is_first_vcl_of_new_picture(1, 1, &hdr_base()));
    }

    /// §7.4.1.2.4 — `pic_order_cnt_lsb` differs ⇒ new picture.
    #[test]
    fn different_poc_lsb_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        let mut h = hdr_base();
        h.pic_order_cnt_lsb = 4;
        assert!(dec.is_first_vcl_of_new_picture(1, 2, &h));
    }

    /// §7.4.1.2.4 — `delta_pic_order_cnt_bottom` differs ⇒ new picture.
    #[test]
    fn different_delta_poc_bottom_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        let mut h = hdr_base();
        h.delta_pic_order_cnt_bottom = 1;
        assert!(dec.is_first_vcl_of_new_picture(1, 2, &h));
    }

    /// §7.4.1.2.4 — `delta_pic_order_cnt[0]` differs ⇒ new picture.
    #[test]
    fn different_delta_poc_0_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        let mut h = hdr_base();
        h.delta_pic_order_cnt[0] = 2;
        assert!(dec.is_first_vcl_of_new_picture(1, 2, &h));
    }

    /// §7.4.1.2.4 — `delta_pic_order_cnt[1]` differs ⇒ new picture.
    #[test]
    fn different_delta_poc_1_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        let mut h = hdr_base();
        h.delta_pic_order_cnt[1] = 3;
        assert!(dec.is_first_vcl_of_new_picture(1, 2, &h));
    }

    /// §7.4.1.2.4 — IdrPicFlag differs (one IDR, other not) ⇒ new.
    #[test]
    fn different_idr_flag_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        seed_in_progress(&mut dec, 1, 2, hdr_base());
        // IDR is nal_unit_type == 5. The new slice type 5 differs from
        // prev type 1.
        let mut h = hdr_base();
        h.slice_type_raw = 2;
        h.slice_type = ST::I;
        assert!(dec.is_first_vcl_of_new_picture(5, 2, &h));
    }

    /// §7.4.1.2.4 — both IDR but `idr_pic_id` differs ⇒ new.
    #[test]
    fn different_idr_pic_id_is_new_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let mut prev = hdr_base();
        prev.idr_pic_id = 0;
        prev.slice_type_raw = 2;
        prev.slice_type = ST::I;
        seed_in_progress(&mut dec, 5, 3, prev);
        let mut h = hdr_base();
        h.idr_pic_id = 1;
        h.slice_type_raw = 2;
        h.slice_type = ST::I;
        assert!(dec.is_first_vcl_of_new_picture(5, 3, &h));
    }

    /// §7.4.1.2.4 — both IDR with SAME idr_pic_id and all other fields
    /// match ⇒ SAME picture.
    #[test]
    fn same_idr_pic_id_is_same_picture() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let mut prev = hdr_base();
        prev.idr_pic_id = 7;
        prev.slice_type_raw = 2;
        prev.slice_type = ST::I;
        seed_in_progress(&mut dec, 5, 3, prev.clone());
        // Continuation slice in a multi-slice IDR picture.
        let mut h = prev;
        h.first_mb_in_slice = 384;
        assert!(!dec.is_first_vcl_of_new_picture(5, 3, &h));
    }
}
