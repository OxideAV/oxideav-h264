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

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, TimeBase, VideoFrame,
    VideoPlane,
};

use crate::decoder::{Decoder as H264Driver, Event};
use crate::dpb_output::{DpbOutput, OutputEntry};
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
        // route it through the §C.4 bumping process.
        let pts = self.pending_pts.take();
        let time_base = self.pending_time_base;
        let vf = picture_to_video_frame(&pic, pts, time_base);

        // §C.4 — at an IDR, any pictures still held in the output DPB
        // belong to the *previous* coded video sequence and must be
        // delivered in POC order before the IDR itself. Drain them
        // into `ready` first, then reset the output queue, then push
        // the IDR. MMCO op 5 inside a reference picture also triggers
        // this reset (§8.2.5.4 / §C.4); we already detected that as
        // `mmco5_triggered` above.
        if is_idr || mmco5_triggered {
            for drained in self.output_dpb.flush() {
                self.ready.push_back(drained.picture);
            }
            self.output_dpb.reset();
        }

        // Size the output DPB against the active SPS (§E.2.1 /
        // §A.3.1 item h). Cheap to call every slice — the helper
        // short-circuits when the derived capacity hasn't changed.
        self.ensure_output_dpb_sized(&sps);

        // Push into the output DPB. §C.4: if the queue is full, the
        // lowest-POC entry is bumped out and becomes immediately
        // available to `receive_frame`.
        let entry = OutputEntry {
            picture: vf,
            pic_order_cnt: poc.pic_order_cnt,
            frame_num: header.frame_num,
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

/// Derive the `(max_num_reorder_frames, max_dec_frame_buffering)`
/// pair for sizing [`DpbOutput`] from the active SPS.
///
/// Preferred source is the VUI `bitstream_restriction` block
/// (§E.2.1). When it's absent we infer a sensible default:
///   1. `max_dec_frame_buffering` ← Annex A Table A-1 per-level
///      cap, i.e. Min(MaxDpbMbs / (PicWidthInMbs * FrameHeightInMbs), 16)
///      (§A.3.1 item h).
///   2. `max_num_reorder_frames` ← `sps.max_num_ref_frames`, clamped
///      to the buffering cap. This is a *pragmatic* default: the
///      spec allows max_num_reorder_frames up to MaxDpbFrames but
///      streams without a `bitstream_restriction` block typically
///      don't reorder further than their reference-count ceiling.
///      Using MaxDpbFrames directly would force the decoder to hold
///      pictures far longer than the stream actually requires, which
///      breaks low-latency consumers that flush one frame at a time.
///
/// Both values floor at 1 so `DpbOutput::push` can always bump when
/// the queue is full (cap of 0 would deadlock the queue).
fn output_dpb_sizing(sps: &Sps) -> (u32, u32) {
    if let Some(br) = sps.vui.as_ref().and_then(|v| v.bitstream_restriction.as_ref()) {
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
    let max_dpb_frames = (max_dpb_mbs / pic_size_mbs).min(16).max(1);

    // Pragmatic reorder default when the VUI block is absent: use
    // max_num_ref_frames (§7.4.2.1.1). A stream with
    // max_num_ref_frames == 1 never reorders; a B-capable stream
    // typically signals reorder explicitly via VUI.
    let reorder = sps.max_num_ref_frames.max(1).min(max_dpb_frames);
    (reorder, max_dpb_frames)
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
            format: PixelFormat::Gray8,
            width: 1,
            height: 1,
            pts: None,
            time_base: TimeBase::new(1, 1),
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

        let tags: Vec<u8> = std::iter::from_fn(|| dec.receive_frame().ok().map(|f| vf_tag(&f)))
            .collect();
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

        let tags: Vec<u8> = std::iter::from_fn(|| dec.receive_frame().ok().map(|f| vf_tag(&f)))
            .collect();
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
    /// `sps.max_num_ref_frames` (clamped to the cap) as the reorder
    /// window. Level 3.0 (level_idc == 30) has MaxDpbMbs = 8100; for
    /// a 176x144 (11x9 MB) picture that's Min(8100/99, 16) = 16.
    /// With `max_num_ref_frames == 4` in the fabricated SPS the
    /// expected result is (4, 16).
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
        // (max_dec_frame_buffering). Reorder defaults to
        // max_num_ref_frames = 4 clamped to 16 → 4.
        assert_eq!(output_dpb_sizing(&sps), (4, 16));
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
}
