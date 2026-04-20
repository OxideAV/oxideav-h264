//! H.264 decoder scaffold exposed through the `oxideav_codec::Decoder`
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
//! - **Access unit assembly**: each slice NAL is treated as its own
//!   picture. Multi-slice pictures (several slice NALs that together
//!   form one coded picture per §7.4.1.2.4) are not assembled —
//!   decode artifacts may occur on streams that rely on that.
//! - **Output ordering**: frames are emitted in decode order, not
//!   display (POC) order. A follow-up will wire through
//!   [`crate::dpb_output::DpbOutput`] for proper reordering.
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

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, TimeBase, VideoFrame,
    VideoPlane,
};

use crate::decoder::{Decoder as H264Driver, Event};
use crate::mb_grid::MbGrid;
use crate::picture::Picture;
use crate::poc::{derive_poc, PocSlice, PocSps, PocState};
use crate::ref_list::{self, DpbEntry, MmcoOp as RefMmcoOp, PicStructure, RefMarking, RplmOp};
use crate::ref_store::{RefPicProvider, RefPicStore};
use crate::slice_header::{
    MmcoOp as SliceMmcoOp, RefPicListModificationOp as SliceRplmOp, SliceHeader, SliceType,
};
use crate::sps::Sps;
use crate::{reconstruct, slice_data};

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
    /// Reconstructed frames queued for `receive_frame`.
    frames: VecDeque<VideoFrame>,
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
}

impl H264CodecDecoder {
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            length_size: None,
            driver: H264Driver::new(),
            last_slice: None,
            eof: false,
            frames: VecDeque::new(),
            pending_pts: None,
            pending_time_base: TimeBase::new(1, 1),
            ref_store: RefPicStore::new(),
            dpb_entries: Vec::new(),
            poc_state: PocState::default(),
            next_dpb_key: 0,
            prev_had_mmco5: false,
            prev_reference_top_foc: 0,
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
            return Err(Error::invalid(
                "h264: avcC configurationVersion must be 1",
            ));
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
            } => {
                self.last_slice = Some(header.clone());
                self.handle_slice(
                    nal_unit_type,
                    nal_ref_idc,
                    header,
                    rbsp,
                    slice_data_cursor,
                )
            }
            // §7.4.1.2.4 AUD marks the start of a new access unit;
            // in our simplified one-slice-per-picture model there's
            // no additional bookkeeping to do at the AUD itself — the
            // next slice's frame_num boundary check handles picture
            // transitions.
            _ => Ok(()),
        }
    }

    /// Drive reconstruction for one slice NAL. Covers both the IDR
    /// and P/B paths.
    ///
    /// Simplification: each slice NAL is treated as its own coded
    /// picture (§7.4.1.2.4). Multi-slice pictures (several slice NALs
    /// sharing the same frame_num / POC) are not assembled — doing so
    /// correctly requires gathering all slices of the access unit into
    /// a single Picture before running deblocking, and is a known
    /// follow-up. In practice most streams we feed through oxideplay
    /// use one slice per picture, so this gets us moving.
    fn handle_slice(
        &mut self,
        nal_unit_type: u8,
        nal_ref_idc: u8,
        header: SliceHeader,
        rbsp: Vec<u8>,
        cursor: (usize, u8),
    ) -> Result<()> {
        // Snapshot SPS and PPS to detach from the driver borrow so we
        // can mutate self.frames / self.ref_store below.
        let sps = self
            .driver
            .active_sps()
            .cloned()
            .ok_or_else(|| Error::invalid("h264: slice with no active SPS"))?;
        let pps = self
            .driver
            .active_pps()
            .cloned()
            .ok_or_else(|| Error::invalid("h264: slice with no active PPS"))?;

        let is_idr = nal_unit_type == 5;
        let is_reference = nal_ref_idc != 0;

        // §8.2.1 — derive POC for this picture.
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

        // Parse slice_data — common to I / P / B paths.
        let sd = slice_data::parse_slice_data(&rbsp, cursor.0, cursor.1, &header, &sps, &pps)
            .map_err(|e| Error::invalid(format!("h264 slice_data: {e}")))?;

        // Build the destination Picture.
        let width_samples = sps.pic_width_in_mbs() * 16;
        let height_samples = sps.frame_height_in_mbs() * 16;
        let chroma_array_type = sps.chroma_array_type();
        let mut pic = Picture::new(
            width_samples,
            height_samples,
            chroma_array_type,
            sps.bit_depth_luma_minus8 + 8,
            sps.bit_depth_chroma_minus8 + 8,
        );
        let mut grid = MbGrid::new(sps.pic_width_in_mbs(), sps.frame_height_in_mbs());

        let current_structure = pic_structure_from_flags(header.field_pic_flag, header.bottom_field_flag);
        let current_bottom = matches!(current_structure, PicStructure::BottomField);
        let current_is_field = header.field_pic_flag;

        // Build the per-slice RefPicList0 / RefPicList1. For IDRs
        // both stay empty; for P/B slices we run §8.2.4.2 init then
        // apply §8.2.4.3 RPLM if the slice signalled it.
        let (list0, list1) = if is_idr {
            (Vec::new(), Vec::new())
        } else {
            let max_frame_num = 1u32 << (sps.log2_max_frame_num_minus4 + 4);

            // §8.2.4.2.1 P/SP, §8.2.4.2.3 B, §8.2.4.2 fall-through for I/SI.
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
                    poc.pic_order_cnt,
                    current_structure,
                    current_bottom,
                ),
                SliceType::I | SliceType::SI => (Vec::new(), Vec::new()),
            };

            // §8.2.4.3 — apply RPLM ops (if any). `modify_ref_pic_list`
            // also truncates / pads to `num_active`, so we call it
            // unconditionally for lists the slice type owns.
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

        // §8.4.* — run the actual pixel reconstruction. The provider
        // borrows the long-running ref_store for Picture lookups and
        // carries the per-slice list arrays; no Pictures are cloned
        // for inter prediction.
        let provider = BorrowedRefProvider {
            store: &self.ref_store,
            list_0: &list0,
            list_1: &list1,
        };
        reconstruct::reconstruct_slice(
            &sd,
            &header,
            &sps,
            &pps,
            &provider,
            &mut pic,
            &mut grid,
        )
        .map_err(|e| Error::invalid(format!("h264 reconstruct: {e}")))?;

        // §8.2.5 — decoded reference picture marking. Build the DpbEntry
        // for the current picture, run the marking state machine, then
        // (for reference pictures) insert the Picture + entry into the
        // long-running DPB.
        let mut current_entry = DpbEntry {
            frame_num: header.frame_num,
            top_field_order_cnt: poc.top_field_order_cnt,
            bottom_field_order_cnt: poc.bottom_field_order_cnt,
            pic_order_cnt: poc.pic_order_cnt,
            structure: current_structure,
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
            // §8.2.5.1 — dispatch into IDR / sliding-window / adaptive
            // paths. `perform_marking` does not add the current entry
            // to the DPB — we insert it into `ref_store` + push the
            // DpbEntry ourselves once the marking settles.
            //
            // `nal_ref_idc != 0` guarantees `dec_ref_pic_marking` is
            // present per §7.3.3.
            let marking = header.dec_ref_pic_marking.as_ref();
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
                header.frame_num,
                1u32 << (sps.log2_max_frame_num_minus4 + 4),
            );

            // Evict "Unused" entries so they don't participate in
            // subsequent list initialisation (§8.2.4.2 only considers
            // short- / long-term refs). The long-running RefPicStore
            // still holds the Pictures at those keys, but without a
            // DpbEntry pointing at them they're effectively dead.
            self.dpb_entries
                .retain(|e| !matches!(e.marking, RefMarking::Unused));

            // Insert the current picture keyed by the DpbEntry's
            // dpb_key so ref-pic-list lookups resolve to it.
            self.ref_store.insert(current_entry.dpb_key, pic.clone());
            self.dpb_entries.push(current_entry);
        }

        // §8.2.1 NOTE 1 — only reference pictures count as the
        // "previous reference picture" that feeds the next POC
        // derivation. Non-reference pictures leave the hint alone.
        if is_reference {
            self.prev_had_mmco5 = mmco5_triggered;
            self.prev_reference_top_foc = if mmco5_triggered {
                poc.top_field_order_cnt
            } else {
                0
            };
        }

        // Convert the reconstructed Picture into a VideoFrame and
        // queue it. Output is currently in *decode* order; a later
        // patch will thread `DpbOutput` (§C.4) in to reorder by POC.
        let pts = self.pending_pts.take();
        let time_base = self.pending_time_base;
        let vf = picture_to_video_frame(&pic, pts, time_base);
        self.frames.push_back(vf);

        Ok(())
    }

    fn mint_dpb_key(&mut self) -> u32 {
        let k = self.next_dpb_key;
        self.next_dpb_key = self.next_dpb_key.wrapping_add(1);
        k
    }
}

/// Per-slice [`RefPicProvider`] that borrows pictures from a long-running
/// [`RefPicStore`] but carries its own RefPicList0 / RefPicList1 key
/// arrays. Avoids cloning every DPB Picture on every slice.
struct BorrowedRefProvider<'a> {
    store: &'a RefPicStore,
    list_0: &'a [u32],
    list_1: &'a [u32],
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
        if let Some(vf) = self.frames.pop_front() {
            return Ok(Frame::Video(vf));
        }
        if self.eof {
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.driver = H264Driver::new();
        self.last_slice = None;
        self.eof = false;
        self.frames.clear();
        self.pending_pts = None;
        self.ref_store = RefPicStore::new();
        self.dpb_entries.clear();
        self.poc_state = PocState::default();
        self.next_dpb_key = 0;
        self.prev_had_mmco5 = false;
        self.prev_reference_top_foc = 0;
        Ok(())
    }
}

/// Convert a reconstructed [`Picture`] to a [`VideoFrame`]. Samples are
/// clamped to 8-bit (0..=255) for now; higher bit depths will get
/// dedicated pixel formats later.
fn picture_to_video_frame(pic: &Picture, pts: Option<i64>, time_base: TimeBase) -> VideoFrame {
    let format = match (pic.chroma_array_type, pic.bit_depth_luma) {
        (0, 8) => PixelFormat::Gray8,
        (1, 8) => PixelFormat::Yuv420P,
        (2, 8) => PixelFormat::Yuv422P,
        (3, 8) => PixelFormat::Yuv444P,
        // Fall back to 4:2:0 for higher bit depths — samples are still
        // clamped to 8-bit below, which is lossy but keeps the pipeline
        // moving. High-bit-depth streams need a wider pixel format later.
        _ => PixelFormat::Yuv420P,
    };

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

    VideoFrame {
        format,
        width: pic.width_in_samples,
        height: pic.height_in_samples,
        pts,
        time_base,
        planes,
    }
}
