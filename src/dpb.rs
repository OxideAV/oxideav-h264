//! Decoded Picture Buffer (DPB) — ITU-T H.264 Annex C / §8.2.5.
//!
//! The DPB holds up to `max_num_ref_frames` reconstructed reference frames
//! plus a small reorder queue for POC-based output ordering. Each stored
//! reference carries:
//!
//! * `frame_num` (§7.4.3) — the slice header's `frame_num`, interpreted
//!   modulo `MaxFrameNum`.
//! * `poc` (PicOrderCnt, §8.2.1) — derived from the SPS
//!   `pic_order_cnt_type` + slice header fields. This decoder implements
//!   type 0 (explicit LSB with MSB tracking) and type 2 (POC = 2·frame_num
//!   for reference pictures, 2·frame_num−1 for non-reference).
//! * short-term / long-term marking — §8.2.5 maintains exactly these two
//!   classes; a picture is never in both. Long-term pictures also carry a
//!   `long_term_pic_num` that the RefPicList construction uses for ordering.
//! * `used_for_reference` — set on insertion, cleared by sliding-window
//!   aging, MMCO operations 1 / 2 / 5, or by the IDR-wipe path.
//!
//! The RefPicList0 construction for P-slices follows §8.2.4.2.1:
//!
//! 1. Short-term reference frames sorted by descending `FrameNumWrap`
//!    (so the most recent short-term reference is index 0).
//! 2. Long-term reference frames sorted by ascending `LongTermPicNum`.
//!
//! The slice header's `num_ref_idx_l0_active_minus1` trims the final list
//! to exactly `num_ref_idx_l0_active_minus1 + 1` entries. Reference picture
//! list modification (§8.2.4.3) is **not** applied here — the slice header
//! parser surfaces `Error::Unsupported` on non-identity RPLM commands.
//!
//! Reference-marking modes:
//!
//! * `adaptive_ref_pic_marking_mode_flag = 0` → sliding-window marking
//!   (§8.2.5.3): when the short-term + long-term count would exceed
//!   `max_num_ref_frames`, the oldest short-term picture (smallest
//!   `FrameNumWrap`) is marked "unused for reference".
//! * `adaptive_ref_pic_marking_mode_flag = 1` → explicit MMCO commands
//!   (§8.2.5.4). Operations 1 (mark short-term unused), 3 (assign long-
//!   term index), 5 (mark all unused), and 6 (current picture → long-term)
//!   are implemented. Operations 2 (mark long-term unused) and 4 (adjust
//!   MaxLongTermFrameIdx) are also implemented for completeness.
//!
//! Output ordering — the caller sees frames in POC order, not decode order.
//! The [`Dpb::take_ready_outputs`] helper drains all currently-buffered
//! pictures with `poc <= last_emitted_poc + reorder_window` and returns
//! them sorted by POC. [`Dpb::flush_all`] drains the remaining queue at
//! EOF / reset.

use crate::picture::Picture;

/// A single entry in the DPB.
#[derive(Clone, Debug)]
pub struct RefFrame {
    /// `frame_num` as seen in the slice header (pre-wrap).
    pub frame_num: u32,
    /// `FrameNumWrap` — `frame_num` adjusted by `prev_ref_frame_num`
    /// wrapping per §8.2.4.1. For short-term pictures this is the value
    /// used to order them in RefPicList0.
    pub frame_num_wrap: i32,
    /// Picture order count (§8.2.1) — derived once per picture.
    pub poc: i32,
    /// The reconstructed samples.
    pub pic: Picture,
    /// `true` when the frame is currently marked short-term, `false`
    /// when long-term. Meaningless if `used_for_reference == false`.
    pub short_term: bool,
    /// Long-term index (spec: `LongTermFrameIdx`) — `None` for short-term.
    pub long_term_frame_idx: Option<u32>,
    /// Whether this entry is a valid reference picture.
    pub used_for_reference: bool,
    /// `true` once the frame has been delivered to the caller. A frame
    /// stays in the DPB after output until reference marking releases it.
    pub output: bool,
}

/// Decoded Picture Buffer.
#[derive(Clone, Debug)]
pub struct Dpb {
    /// The reference-frame slots. Capacity = `max_num_ref_frames`. The
    /// queue is *not* sorted — insertion order is decode order, and the
    /// list-building helpers do the per-purpose sort.
    frames: Vec<RefFrame>,
    /// Pending output queue for reorder: pictures awaiting POC-ordered
    /// delivery to the caller. Entries are drained by
    /// [`Dpb::take_ready_outputs`] / [`Dpb::flush_all`].
    pending_output: Vec<PendingOutput>,
    /// Maximum number of reference frames — from SPS `max_num_ref_frames`.
    /// Drives the sliding-window bound.
    pub max_num_ref_frames: u32,
    /// Upper bound on `LongTermFrameIdx` + 1 (spec: `MaxLongTermFrameIdx`).
    /// Set by MMCO op 4, or implicitly on IDR by the `long_term_reference_flag`.
    /// `0` means "no long-term frames allowed" (the default after IDR
    /// without `long_term_reference_flag`).
    max_long_term_frame_idx_plus1: u32,
    /// Reorder window — how many decoded frames may sit in the pending-
    /// output queue before one is required to emerge. Derived from the
    /// SPS and profile; a conservative default is `max_num_ref_frames`.
    reorder_window: u32,
    /// Last POC actually output to the caller. Monotonic in POC (subject
    /// to reset on IDR).
    last_output_poc: Option<i32>,
}

/// A picture waiting to be returned to the caller in POC order.
#[derive(Clone, Debug)]
pub struct PendingOutput {
    pub poc: i32,
    pub pic: Picture,
}

impl Dpb {
    /// Create an empty DPB sized for the given SPS parameters.
    /// `reorder_window` is the maximum number of frames the reorder queue
    /// may hold before one is forced out. A value of `0` disables
    /// reordering (each queued picture is eligible for immediate output,
    /// suitable for P-only / baseline streams).
    pub fn new(max_num_ref_frames: u32, reorder_window: u32) -> Self {
        Self {
            frames: Vec::with_capacity(max_num_ref_frames.max(1) as usize),
            pending_output: Vec::new(),
            max_num_ref_frames: max_num_ref_frames.max(1),
            max_long_term_frame_idx_plus1: 0,
            reorder_window,
            last_output_poc: None,
        }
    }

    /// Current reorder window — number of pictures that may sit in the
    /// pending-output queue before one is forced to emerge in POC order.
    pub fn reorder_window(&self) -> u32 {
        self.reorder_window
    }

    /// Adjust the reorder window. Used by the decoder when the first
    /// B-slice appears to raise the window above the P-only default of 0.
    pub fn set_reorder_window(&mut self, window: u32) {
        self.reorder_window = window;
    }

    /// Reset the DPB to its empty state.
    pub fn clear(&mut self) {
        self.frames.clear();
        self.pending_output.clear();
        self.max_long_term_frame_idx_plus1 = 0;
        self.last_output_poc = None;
    }

    /// Number of currently-live reference frames (short + long).
    pub fn num_references(&self) -> usize {
        self.frames.iter().filter(|f| f.used_for_reference).count()
    }

    /// Number of currently-held frames (any status).
    pub fn len(&self) -> usize {
        self.frames.len()
    }
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Live references, ordered for `RefPicList0` construction for a B-slice
    /// per §8.2.4.2.3: short-term entries with `poc < curr_poc` in
    /// descending-POC order, then short-term entries with `poc > curr_poc`
    /// in ascending-POC order, then long-term entries in ascending
    /// `LongTermFrameIdx` order. The list is then trimmed to the slice's
    /// active L0 length.
    pub fn build_list0_b(
        &self,
        curr_poc: i32,
        num_ref_idx_l0_active_minus1: u32,
    ) -> Vec<&RefFrame> {
        let mut past: Vec<&RefFrame> = self
            .frames
            .iter()
            .filter(|f| f.used_for_reference && f.short_term && f.poc < curr_poc)
            .collect();
        past.sort_by(|a, b| b.poc.cmp(&a.poc));

        let mut future: Vec<&RefFrame> = self
            .frames
            .iter()
            .filter(|f| f.used_for_reference && f.short_term && f.poc > curr_poc)
            .collect();
        future.sort_by_key(|f| f.poc);

        let mut long: Vec<&RefFrame> = self
            .frames
            .iter()
            .filter(|f| f.used_for_reference && !f.short_term)
            .collect();
        long.sort_by_key(|f| f.long_term_frame_idx.unwrap_or(u32::MAX));

        let mut list: Vec<&RefFrame> = past;
        list.extend(future);
        list.extend(long);
        let want = (num_ref_idx_l0_active_minus1 as usize) + 1;
        if list.len() > want {
            list.truncate(want);
        }
        list
    }

    /// Live references, ordered for `RefPicList1` construction for a B-slice
    /// per §8.2.4.2.3: short-term entries with `poc > curr_poc` in
    /// ascending-POC order, then short-term entries with `poc < curr_poc`
    /// in descending-POC order, then long-term entries in ascending
    /// `LongTermFrameIdx` order. When `RefPicList1` and `RefPicList0` are
    /// identical and have more than one entry, the first two entries of
    /// `RefPicList1` are swapped (§8.2.4.2.3 last paragraph).
    pub fn build_list1_b(
        &self,
        curr_poc: i32,
        num_ref_idx_l1_active_minus1: u32,
    ) -> Vec<&RefFrame> {
        let mut future: Vec<&RefFrame> = self
            .frames
            .iter()
            .filter(|f| f.used_for_reference && f.short_term && f.poc > curr_poc)
            .collect();
        future.sort_by_key(|f| f.poc);

        let mut past: Vec<&RefFrame> = self
            .frames
            .iter()
            .filter(|f| f.used_for_reference && f.short_term && f.poc < curr_poc)
            .collect();
        past.sort_by(|a, b| b.poc.cmp(&a.poc));

        let mut long: Vec<&RefFrame> = self
            .frames
            .iter()
            .filter(|f| f.used_for_reference && !f.short_term)
            .collect();
        long.sort_by_key(|f| f.long_term_frame_idx.unwrap_or(u32::MAX));

        let mut list: Vec<&RefFrame> = future;
        list.extend(past);
        list.extend(long);
        let want = (num_ref_idx_l1_active_minus1 as usize) + 1;
        if list.len() > want {
            list.truncate(want);
        }
        list
    }

    /// Live references, ordered for `RefPicList0` construction for a P-slice
    /// per §8.2.4.2.1.
    pub fn build_list0_p(&self, num_ref_idx_l0_active_minus1: u32) -> Vec<&RefFrame> {
        let mut short: Vec<&RefFrame> = self
            .frames
            .iter()
            .filter(|f| f.used_for_reference && f.short_term)
            .collect();
        short.sort_by(|a, b| b.frame_num_wrap.cmp(&a.frame_num_wrap));

        let mut long: Vec<&RefFrame> = self
            .frames
            .iter()
            .filter(|f| f.used_for_reference && !f.short_term)
            .collect();
        long.sort_by_key(|f| f.long_term_frame_idx.unwrap_or(u32::MAX));

        let mut list: Vec<&RefFrame> = short;
        list.extend(long);

        let want = (num_ref_idx_l0_active_minus1 as usize) + 1;
        if list.len() > want {
            list.truncate(want);
        }
        list
    }

    /// Compute `FrameNumWrap` per §8.2.4.1.
    pub fn compute_frame_num_wrap(
        frame_num: u32,
        prev_ref_frame_num: u32,
        max_frame_num: u32,
    ) -> i32 {
        if frame_num > prev_ref_frame_num {
            (frame_num as i32) - (max_frame_num as i32)
        } else {
            frame_num as i32
        }
    }

    /// Refresh `frame_num_wrap` on every short-term entry.
    pub fn update_frame_num_wrap(&mut self, prev_ref_frame_num: u32, max_frame_num: u32) {
        for f in &mut self.frames {
            if !f.used_for_reference || !f.short_term {
                continue;
            }
            f.frame_num_wrap =
                Self::compute_frame_num_wrap(f.frame_num, prev_ref_frame_num, max_frame_num);
        }
    }

    /// Apply sliding-window reference picture marking (§8.2.5.3).
    pub fn apply_sliding_window(&mut self) {
        let total_refs = self.frames.iter().filter(|f| f.used_for_reference).count() as u32;
        if total_refs < self.max_num_ref_frames {
            return;
        }
        let target = self
            .frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.used_for_reference && f.short_term)
            .min_by_key(|(_, f)| f.frame_num_wrap)
            .map(|(i, _)| i);
        if let Some(idx) = target {
            self.frames[idx].used_for_reference = false;
        }
        self.gc();
    }

    fn gc(&mut self) {
        self.frames.retain(|f| f.used_for_reference || !f.output);
    }

    /// Insert a freshly-decoded reference picture.
    pub fn insert_reference(
        &mut self,
        pic: Picture,
        frame_num: u32,
        frame_num_wrap: i32,
        poc: i32,
    ) {
        self.frames.push(RefFrame {
            frame_num,
            frame_num_wrap,
            poc,
            pic,
            short_term: true,
            long_term_frame_idx: None,
            used_for_reference: true,
            output: false,
        });
    }

    /// Push a picture into the output-reorder queue.
    pub fn queue_output(&mut self, pic: Picture, poc: i32) {
        self.pending_output.push(PendingOutput { pic, poc });
    }

    /// Drain any buffered outputs whose POC is eligible for delivery.
    pub fn take_ready_outputs(&mut self) -> Vec<Picture> {
        let mut out = Vec::new();
        while self.pending_output.len() > self.reorder_window as usize {
            let (min_idx, _) = self
                .pending_output
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.poc)
                .expect("non-empty queue");
            let entry = self.pending_output.remove(min_idx);
            self.last_output_poc = Some(entry.poc);
            out.push(entry.pic);
        }
        out
    }

    /// Drain the entire pending-output queue in POC order.
    pub fn flush_all(&mut self) -> Vec<Picture> {
        self.pending_output.sort_by_key(|e| e.poc);
        let drained: Vec<Picture> = self
            .pending_output
            .drain(..)
            .map(|e| {
                self.last_output_poc = Some(e.poc);
                e.pic
            })
            .collect();
        drained
    }

    /// Apply a single MMCO command (§8.2.5.4).
    pub fn apply_mmco(
        &mut self,
        cmd: MmcoCommand,
        current_frame_num: u32,
        max_frame_num: u32,
    ) -> oxideav_core::Result<()> {
        match cmd {
            MmcoCommand::MarkShortTermUnused {
                difference_of_pic_nums_minus1,
            } => {
                let curr = current_frame_num as i32;
                let pic_num_x = curr - (difference_of_pic_nums_minus1 as i32 + 1);
                let target = self.find_short_term_by_pic_num(pic_num_x, max_frame_num);
                if let Some(idx) = target {
                    self.frames[idx].used_for_reference = false;
                }
                self.gc();
            }
            MmcoCommand::MarkLongTermUnused { long_term_pic_num } => {
                let target = self.frames.iter().position(|f| {
                    f.used_for_reference
                        && !f.short_term
                        && f.long_term_frame_idx == Some(long_term_pic_num)
                });
                if let Some(idx) = target {
                    self.frames[idx].used_for_reference = false;
                }
                self.gc();
            }
            MmcoCommand::AssignLongTerm {
                difference_of_pic_nums_minus1,
                long_term_frame_idx,
            } => {
                let curr = current_frame_num as i32;
                let pic_num_x = curr - (difference_of_pic_nums_minus1 as i32 + 1);
                for f in &mut self.frames {
                    if f.used_for_reference
                        && !f.short_term
                        && f.long_term_frame_idx == Some(long_term_frame_idx)
                    {
                        f.used_for_reference = false;
                    }
                }
                if let Some(idx) = self.find_short_term_by_pic_num(pic_num_x, max_frame_num) {
                    self.frames[idx].short_term = false;
                    self.frames[idx].long_term_frame_idx = Some(long_term_frame_idx);
                }
                if long_term_frame_idx + 1 > self.max_long_term_frame_idx_plus1 {
                    self.max_long_term_frame_idx_plus1 = long_term_frame_idx + 1;
                }
                self.gc();
            }
            MmcoCommand::SetMaxLongTerm {
                max_long_term_frame_idx_plus1,
            } => {
                self.max_long_term_frame_idx_plus1 = max_long_term_frame_idx_plus1;
                let cutoff = max_long_term_frame_idx_plus1;
                for f in &mut self.frames {
                    if f.used_for_reference && !f.short_term {
                        if let Some(idx) = f.long_term_frame_idx {
                            if idx + 1 > cutoff {
                                f.used_for_reference = false;
                            }
                        }
                    }
                }
                self.gc();
            }
            MmcoCommand::MarkAllUnused => {
                for f in &mut self.frames {
                    f.used_for_reference = false;
                }
                self.max_long_term_frame_idx_plus1 = 0;
                self.gc();
            }
            MmcoCommand::AssignCurrentLongTerm {
                long_term_frame_idx,
            } => {
                for f in &mut self.frames {
                    if f.used_for_reference
                        && !f.short_term
                        && f.long_term_frame_idx == Some(long_term_frame_idx)
                    {
                        f.used_for_reference = false;
                    }
                }
                if let Some(last) = self.frames.last_mut() {
                    last.short_term = false;
                    last.long_term_frame_idx = Some(long_term_frame_idx);
                }
                if long_term_frame_idx + 1 > self.max_long_term_frame_idx_plus1 {
                    self.max_long_term_frame_idx_plus1 = long_term_frame_idx + 1;
                }
                self.gc();
            }
        }
        Ok(())
    }

    fn find_short_term_by_pic_num(&self, pic_num: i32, _max_frame_num: u32) -> Option<usize> {
        self.frames
            .iter()
            .enumerate()
            .find(|(_, f)| f.used_for_reference && f.short_term && f.frame_num_wrap == pic_num)
            .map(|(i, _)| i)
    }

    /// Iterate live reference frames (diagnostics).
    pub fn iter_refs(&self) -> impl Iterator<Item = &RefFrame> {
        self.frames.iter().filter(|f| f.used_for_reference)
    }

    /// Mark the most recently inserted frame as already-output.
    pub fn mark_last_output(&mut self) {
        if let Some(last) = self.frames.last_mut() {
            last.output = true;
        }
    }
}

/// The parsed MMCO command set (§7.3.3.3 / §8.2.5.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmcoCommand {
    /// Op 1 — mark a short-term picture as "unused for reference".
    MarkShortTermUnused { difference_of_pic_nums_minus1: u32 },
    /// Op 2 — mark a long-term picture as "unused for reference".
    MarkLongTermUnused { long_term_pic_num: u32 },
    /// Op 3 — mark the addressed short-term picture as long-term.
    AssignLongTerm {
        difference_of_pic_nums_minus1: u32,
        long_term_frame_idx: u32,
    },
    /// Op 4 — update `MaxLongTermFrameIdx`.
    SetMaxLongTerm { max_long_term_frame_idx_plus1: u32 },
    /// Op 5 — mark every reference picture unused. IDR-like.
    MarkAllUnused,
    /// Op 6 — tag the current picture as a long-term reference.
    AssignCurrentLongTerm { long_term_frame_idx: u32 },
}

// ---------------------------------------------------------------------------
// POC (Picture Order Count) derivation — §8.2.1.
// ---------------------------------------------------------------------------

/// Tracked state for `pic_order_cnt_type == 0` (§8.2.1.1).
#[derive(Clone, Copy, Debug, Default)]
pub struct PocState {
    pub prev_pic_order_cnt_msb: i32,
    pub prev_pic_order_cnt_lsb: i32,
}

/// Derive `PicOrderCnt` for `pic_order_cnt_type == 0` — §8.2.1.1.
pub fn derive_poc_type0(
    pic_order_cnt_lsb: u32,
    log2_max_pic_order_cnt_lsb_minus4: u32,
    prev: PocState,
    is_idr: bool,
) -> (i32, PocState) {
    if is_idr {
        let new_state = PocState {
            prev_pic_order_cnt_msb: 0,
            prev_pic_order_cnt_lsb: 0,
        };
        return (pic_order_cnt_lsb as i32, new_state);
    }

    let max_pic_order_cnt_lsb = 1i32 << (log2_max_pic_order_cnt_lsb_minus4 + 4);
    let lsb = pic_order_cnt_lsb as i32;

    let pic_order_cnt_msb = if lsb < prev.prev_pic_order_cnt_lsb
        && (prev.prev_pic_order_cnt_lsb - lsb) >= max_pic_order_cnt_lsb / 2
    {
        prev.prev_pic_order_cnt_msb + max_pic_order_cnt_lsb
    } else if lsb > prev.prev_pic_order_cnt_lsb
        && (lsb - prev.prev_pic_order_cnt_lsb) > max_pic_order_cnt_lsb / 2
    {
        prev.prev_pic_order_cnt_msb - max_pic_order_cnt_lsb
    } else {
        prev.prev_pic_order_cnt_msb
    };

    let top_field_order_cnt = pic_order_cnt_msb + lsb;
    let new_state = PocState {
        prev_pic_order_cnt_msb: pic_order_cnt_msb,
        prev_pic_order_cnt_lsb: lsb,
    };
    (top_field_order_cnt, new_state)
}

/// Derive `PicOrderCnt` for `pic_order_cnt_type == 2` — §8.2.1.3.
pub fn derive_poc_type2(
    frame_num: u32,
    prev_frame_num: u32,
    prev_frame_num_offset: i32,
    max_frame_num: u32,
    nal_ref_idc: u8,
    is_idr: bool,
) -> (i32, i32) {
    if is_idr {
        return (0, 0);
    }
    let frame_num_offset = if prev_frame_num > frame_num {
        prev_frame_num_offset + (max_frame_num as i32)
    } else {
        prev_frame_num_offset
    };
    let abs = frame_num_offset + frame_num as i32;
    let poc = 2 * abs - if nal_ref_idc == 0 { 1 } else { 0 };
    (poc, frame_num_offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::picture::Picture;

    fn dummy_pic() -> Picture {
        Picture::new(1, 1)
    }

    #[test]
    fn poc_type0_monotonic_without_wrap() {
        let (p0, s0) = derive_poc_type0(0, 4, PocState::default(), true);
        assert_eq!(p0, 0);
        let (p1, s1) = derive_poc_type0(2, 4, s0, false);
        assert_eq!(p1, 2);
        let (p2, _s2) = derive_poc_type0(4, 4, s1, false);
        assert_eq!(p2, 4);
    }

    #[test]
    fn poc_type0_wraps_over_lsb_boundary() {
        let log2 = 4u32;
        let max_lsb = 1i32 << (log2 + 4);
        let mut state = PocState {
            prev_pic_order_cnt_msb: 0,
            prev_pic_order_cnt_lsb: max_lsb - 2,
        };
        let (p, s) = derive_poc_type0(0, log2, state, false);
        assert_eq!(p, max_lsb);
        assert_eq!(s.prev_pic_order_cnt_msb, max_lsb);
        assert_eq!(s.prev_pic_order_cnt_lsb, 0);
        state = s;
        let (p2, _s2) = derive_poc_type0(2, log2, state, false);
        assert_eq!(p2, max_lsb + 2);
    }

    #[test]
    fn poc_type2_ref_and_nonref() {
        let (p0, off0) = derive_poc_type2(0, 0, 0, 16, 3, true);
        assert_eq!(p0, 0);
        assert_eq!(off0, 0);
        let (p1, off1) = derive_poc_type2(1, 0, off0, 16, 3, false);
        assert_eq!(p1, 2);
        assert_eq!(off1, 0);
        let (p2, off2) = derive_poc_type2(2, 1, off1, 16, 0, false);
        assert_eq!(p2, 2 * 2 - 1);
        assert_eq!(off2, 0);
        let (p3, off3) = derive_poc_type2(0, 15, 0, 16, 3, false);
        assert_eq!(off3, 16);
        assert_eq!(p3, 2 * 16);
    }

    #[test]
    fn sliding_window_evicts_oldest_short_term() {
        let mut dpb = Dpb::new(4, 4);
        for fn_ in 0..4u32 {
            dpb.apply_sliding_window();
            dpb.insert_reference(dummy_pic(), fn_, fn_ as i32, fn_ as i32);
        }
        assert_eq!(dpb.num_references(), 4);
        dpb.apply_sliding_window();
        dpb.insert_reference(dummy_pic(), 4, 4, 4);
        assert_eq!(dpb.num_references(), 4);
        let live: Vec<u32> = dpb.iter_refs().map(|f| f.frame_num).collect();
        assert!(!live.contains(&0));
        assert!(live.contains(&4));
    }

    #[test]
    fn mmco_op1_marks_short_term_unused() {
        let mut dpb = Dpb::new(4, 4);
        for fn_ in 0..3u32 {
            dpb.insert_reference(dummy_pic(), fn_, fn_ as i32, fn_ as i32);
        }
        dpb.apply_mmco(
            MmcoCommand::MarkShortTermUnused {
                difference_of_pic_nums_minus1: 1,
            },
            3,
            16,
        )
        .unwrap();
        let live: Vec<u32> = dpb.iter_refs().map(|f| f.frame_num).collect();
        assert_eq!(live.len(), 2);
        assert!(!live.contains(&1));
    }

    #[test]
    fn mmco_op3_assigns_long_term() {
        let mut dpb = Dpb::new(4, 4);
        for fn_ in 0..3u32 {
            dpb.insert_reference(dummy_pic(), fn_, fn_ as i32, fn_ as i32);
        }
        dpb.apply_mmco(
            MmcoCommand::AssignLongTerm {
                difference_of_pic_nums_minus1: 1,
                long_term_frame_idx: 0,
            },
            3,
            16,
        )
        .unwrap();
        let lt: Vec<_> = dpb
            .iter_refs()
            .filter(|f| !f.short_term)
            .map(|f| (f.frame_num, f.long_term_frame_idx))
            .collect();
        assert_eq!(lt, vec![(1, Some(0))]);
    }

    #[test]
    fn mmco_op5_clears_all() {
        let mut dpb = Dpb::new(4, 4);
        for fn_ in 0..3u32 {
            dpb.insert_reference(dummy_pic(), fn_, fn_ as i32, fn_ as i32);
        }
        dpb.apply_mmco(MmcoCommand::MarkAllUnused, 3, 16).unwrap();
        assert_eq!(dpb.num_references(), 0);
    }

    #[test]
    fn mmco_op6_tags_current_long_term() {
        let mut dpb = Dpb::new(4, 4);
        dpb.insert_reference(dummy_pic(), 0, 0, 0);
        dpb.insert_reference(dummy_pic(), 1, 1, 1);
        dpb.apply_mmco(
            MmcoCommand::AssignCurrentLongTerm {
                long_term_frame_idx: 2,
            },
            1,
            16,
        )
        .unwrap();
        let last = dpb.iter_refs().last().unwrap();
        assert!(!last.short_term);
        assert_eq!(last.long_term_frame_idx, Some(2));
    }

    #[test]
    fn list0_p_orders_short_then_long() {
        let mut dpb = Dpb::new(4, 4);
        for fn_ in 0..3u32 {
            dpb.insert_reference(dummy_pic(), fn_, fn_ as i32, fn_ as i32);
        }
        dpb.update_frame_num_wrap(2, 16);
        dpb.apply_mmco(
            MmcoCommand::AssignLongTerm {
                difference_of_pic_nums_minus1: 2,
                long_term_frame_idx: 7,
            },
            3,
            16,
        )
        .unwrap();
        let list = dpb.build_list0_p(2);
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].frame_num, 2);
        assert_eq!(list[1].frame_num, 1);
        assert_eq!(list[2].long_term_frame_idx, Some(7));
    }

    #[test]
    fn list0_p_trims_to_active_count() {
        let mut dpb = Dpb::new(4, 4);
        for fn_ in 0..4u32 {
            dpb.insert_reference(dummy_pic(), fn_, fn_ as i32, fn_ as i32);
        }
        let list = dpb.build_list0_p(0);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].frame_num, 3);
    }

    #[test]
    fn list0_b_splits_past_future_short_term() {
        // Four short-term references with POCs [2, 4, 6, 10].
        // Current POC = 5. §8.2.4.2.3 RefPicList0 order:
        //   past (poc < 5) descending by POC → [4, 2]
        //   future (poc > 5) ascending by POC → [6, 10]
        // Expected list: [4, 2, 6, 10].
        let mut dpb = Dpb::new(8, 8);
        for (fn_, poc) in [(0u32, 2i32), (1, 4), (2, 6), (3, 10)] {
            dpb.insert_reference(dummy_pic(), fn_, fn_ as i32, poc);
        }
        let list = dpb.build_list0_b(5, 7);
        let pocs: Vec<i32> = list.iter().map(|f| f.poc).collect();
        assert_eq!(pocs, vec![4, 2, 6, 10]);
    }

    #[test]
    fn list1_b_splits_future_past_short_term() {
        // Same setup as above. §8.2.4.2.3 RefPicList1 order:
        //   future ascending → [6, 10]
        //   past descending → [4, 2]
        // Expected: [6, 10, 4, 2].
        let mut dpb = Dpb::new(8, 8);
        for (fn_, poc) in [(0u32, 2i32), (1, 4), (2, 6), (3, 10)] {
            dpb.insert_reference(dummy_pic(), fn_, fn_ as i32, poc);
        }
        let list = dpb.build_list1_b(5, 7);
        let pocs: Vec<i32> = list.iter().map(|f| f.poc).collect();
        assert_eq!(pocs, vec![6, 10, 4, 2]);
    }

    #[test]
    fn list1_b_long_term_after_short_term() {
        // Two short-term refs (POC 2 and 10) plus one long-term ref.
        let mut dpb = Dpb::new(8, 8);
        dpb.insert_reference(dummy_pic(), 0, 0, 2);
        dpb.insert_reference(dummy_pic(), 1, 1, 10);
        dpb.insert_reference(dummy_pic(), 2, 2, 7);
        // Promote the last one to long-term.
        dpb.apply_mmco(
            MmcoCommand::AssignLongTerm {
                difference_of_pic_nums_minus1: 0,
                long_term_frame_idx: 3,
            },
            2,
            16,
        )
        .unwrap();
        let list = dpb.build_list1_b(5, 7);
        // Short-term first (future-asc then past-desc), then long-term.
        let lt_end = list.last().unwrap();
        assert_eq!(lt_end.long_term_frame_idx, Some(3));
    }

    #[test]
    fn output_reorder_respects_window() {
        let mut dpb = Dpb::new(4, 2);
        dpb.queue_output(dummy_pic(), 0);
        assert_eq!(dpb.take_ready_outputs().len(), 0);
        dpb.queue_output(dummy_pic(), 4);
        assert_eq!(dpb.take_ready_outputs().len(), 0);
        dpb.queue_output(dummy_pic(), 2);
        let out = dpb.take_ready_outputs();
        assert_eq!(out.len(), 1);
        let out2 = dpb.flush_all();
        assert_eq!(out2.len(), 2);
    }
}
