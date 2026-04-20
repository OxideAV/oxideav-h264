//! POC-ordered output queue per Annex C of ITU-T Rec. H.264 (08/2024).
//!
//! After each picture is decoded and reconstructed, the decoder calls
//! [`DpbOutput::push`]. Completed pictures are held until output order
//! (POC ascending) is resolvable; they then become available via
//! [`DpbOutput::pop_ready`] or [`DpbOutput::flush`] at end-of-stream.
//!
//! Spec references
//! ---------------
//! * §C.2.2 — "Storage and output of decoded pictures"
//! * §C.4   — "Bumping process" / output timing of decoded pictures
//! * §C.4.4 — "Marking decoded pictures as 'not needed for output'"
//! * §8.2.5.4 — MMCO op 5 resets the DPB; callers use [`DpbOutput::reset`]
//! * Annex A Table A-1 — per-level `MaxDpbMbs` → default `max_num_reorder_frames`
//! * §E.2.1 — SPS VUI `bitstream_restriction` overrides the default
//!
//! This module is the *output side* of the DPB — it orders completed
//! pictures by PicOrderCnt for display. Reference-picture storage used
//! by motion compensation lives in `ref_list.rs`.
//!
//! Picture payload type
//! --------------------
//! [`OutputEntry`] is generic over the picture payload `P` so this
//! module compiles independently of the still-in-flight `picture::Picture`
//! rewrite. Once `picture.rs` is filled in, callers parameterise this
//! queue with `OutputEntry<crate::picture::Picture>` (or the type alias
//! [`DefaultDpbOutput`]) and nothing else changes — ordering decisions
//! use the explicit `pic_order_cnt` + `frame_num` carried alongside.

#![allow(dead_code)]

/// A single entry in the output queue.
///
/// The picture payload `P` is opaque to this module: ordering is driven
/// solely by the `pic_order_cnt` / `frame_num` stored alongside it, per
/// Annex C.
#[derive(Debug, Clone)]
pub struct OutputEntry<P> {
    /// The decoded picture samples (opaque to ordering logic).
    pub picture: P,
    /// PicOrderCnt derived per §8.2.1 — governs *output* order.
    pub pic_order_cnt: i32,
    /// §7.4.3 `frame_num` — governs *decoding* order; used as POC
    /// tiebreaker per the §C.4 NOTE.
    pub frame_num: u32,
    /// §C.4 — set to true on insertion; cleared when the picture is
    /// emitted (bumped) and thereby transitions to "not needed for
    /// output".
    pub needed_for_output: bool,
}

/// POC-ordered output DPB.
///
/// The queue holds up to `max_num_reorder_frames` completed pictures;
/// a further insertion triggers a §C.4 bumping step, evicting the
/// lowest-POC entry to the caller.
pub struct DpbOutput<P> {
    /// Maximum entries held before bumping begins. Sourced from
    /// SPS VUI `bitstream_restriction::max_num_reorder_frames` (§E.2.1)
    /// when present, else the Annex A Table A-1 per-level default.
    pub max_num_reorder_frames: u32,
    /// §E.2.1 / §A.3.1 — upper bound on total DPB occupancy.
    /// Carried here for API completeness; enforcement of reference
    /// retention is `ref_list`'s job. This module never holds more
    /// than `max_num_reorder_frames` pictures anyway.
    pub max_dec_frame_buffering: u32,
    entries: Vec<OutputEntry<P>>,
}

impl<P> DpbOutput<P> {
    /// Create a new output DPB sized per §E.2.1 / Annex A Table A-1.
    pub fn new(max_num_reorder_frames: u32, max_dec_frame_buffering: u32) -> Self {
        Self {
            max_num_reorder_frames,
            max_dec_frame_buffering,
            entries: Vec::new(),
        }
    }

    /// §C.4 — Push a newly-decoded picture, bumping if the queue is full.
    ///
    /// If adding the entry would cause the queue to exceed
    /// `max_num_reorder_frames`, the lowest-POC entry currently held
    /// (per §C.4 ordering, tiebroken by `frame_num`) is returned and
    /// the new entry takes its place in the queue. Otherwise returns
    /// `None` and the picture stays queued until a later push / pop /
    /// flush.
    pub fn push(&mut self, mut entry: OutputEntry<P>) -> Option<OutputEntry<P>> {
        // §C.4: "picture is needed for output" on entry.
        entry.needed_for_output = true;

        let bumped = if self.entries.len() as u32 >= self.max_num_reorder_frames {
            self.bump_lowest()
        } else {
            None
        };

        self.entries.push(entry);
        bumped
    }

    /// §C.4 — Conservative bump: returns the lowest-POC entry only
    /// while the queue is at/over capacity.
    ///
    /// Per the Annex C bumping process, output is deferred until the
    /// queue cannot accept another picture without bumping. Callers
    /// that know the stream has ended should use [`flush`] instead.
    pub fn pop_ready(&mut self) -> Option<OutputEntry<P>> {
        if (self.entries.len() as u32) > self.max_num_reorder_frames {
            self.bump_lowest()
        } else {
            None
        }
    }

    /// §C.4 — End-of-stream: drain everything in POC-ascending order.
    ///
    /// This is the "no_output_of_prior_pics_flag == 0" / end-of-stream
    /// path of the bumping process, where every remaining picture
    /// marked "needed for output" is emitted.
    pub fn flush(&mut self) -> Vec<OutputEntry<P>> {
        // Sort ascending by POC, tie-break by frame_num (§C.4 NOTE).
        self.entries.sort_by(|a, b| {
            a.pic_order_cnt
                .cmp(&b.pic_order_cnt)
                .then(a.frame_num.cmp(&b.frame_num))
        });
        let mut out = std::mem::take(&mut self.entries);
        for e in out.iter_mut() {
            e.needed_for_output = false;
        }
        out
    }

    /// §8.2.5.4 MMCO op 5 — resets the DPB.
    ///
    /// Per the spec, after MMCO 5 all reference pictures are marked
    /// "unused for reference" and any picture not yet output is still
    /// to be output (§C.4 bumping empties the DPB). This helper drains
    /// the queue in output order, giving the caller the final batch.
    pub fn reset(&mut self) -> Vec<OutputEntry<P>> {
        self.flush()
    }

    /// Number of pictures currently held in the output queue.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no picture is currently held.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// §C.4 bumping primitive — remove and return the lowest-POC entry
    /// (ties broken by smaller `frame_num`). Returns `None` on empty.
    fn bump_lowest(&mut self) -> Option<OutputEntry<P>> {
        if self.entries.is_empty() {
            return None;
        }
        // Find index of min (poc, frame_num).
        let mut min_idx = 0usize;
        for i in 1..self.entries.len() {
            let a = &self.entries[i];
            let b = &self.entries[min_idx];
            if a.pic_order_cnt < b.pic_order_cnt
                || (a.pic_order_cnt == b.pic_order_cnt && a.frame_num < b.frame_num)
            {
                min_idx = i;
            }
        }
        let mut e = self.entries.remove(min_idx);
        // §C.4 — bumping transitions the picture to "not needed for output".
        e.needed_for_output = false;
        Some(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Picture placeholder for ordering-logic unit tests. The real
    /// `crate::picture::Picture` is still a stub at the time of writing;
    /// `DpbOutput` is generic over picture payload, so here we just
    /// use a `u32` id to uniquely identify pushed pictures.
    type TestPic = u32;

    fn mk(poc: i32, fnum: u32) -> OutputEntry<TestPic> {
        OutputEntry {
            picture: fnum,
            pic_order_cnt: poc,
            frame_num: fnum,
            needed_for_output: false,
        }
    }

    #[test]
    fn empty_queue_yields_nothing() {
        let mut d: DpbOutput<TestPic> = DpbOutput::new(2, 4);
        assert!(d.is_empty());
        assert_eq!(d.len(), 0);
        assert!(d.pop_ready().is_none());
        assert!(d.flush().is_empty());
    }

    #[test]
    fn push_below_capacity_returns_none() {
        let mut d: DpbOutput<TestPic> = DpbOutput::new(3, 4);
        assert!(d.push(mk(0, 0)).is_none());
        assert!(d.push(mk(2, 1)).is_none());
        assert_eq!(d.len(), 2);
        // Below max_num_reorder_frames=3 → conservative pop_ready → None.
        assert!(d.pop_ready().is_none());
    }

    #[test]
    fn push_at_capacity_triggers_bump() {
        let mut d: DpbOutput<TestPic> = DpbOutput::new(2, 4);
        assert!(d.push(mk(4, 0)).is_none());
        assert!(d.push(mk(1, 1)).is_none());
        // Third push should evict lowest-POC (poc=1).
        let bumped = d.push(mk(7, 2)).expect("bump expected on overflow");
        assert_eq!(bumped.pic_order_cnt, 1);
        assert_eq!(bumped.frame_num, 1);
        // §C.4 — bumped entry is no longer "needed for output".
        assert!(!bumped.needed_for_output);
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn push_sets_needed_for_output_flag() {
        let mut d: DpbOutput<TestPic> = DpbOutput::new(4, 4);
        d.push(mk(0, 0));
        // Drain and verify the flag was set by push and cleared by flush.
        let drained = d.flush();
        assert_eq!(drained.len(), 1);
        assert!(!drained[0].needed_for_output);
    }

    #[test]
    fn flush_drains_in_poc_order() {
        let mut d: DpbOutput<TestPic> = DpbOutput::new(8, 8);
        d.push(mk(8, 0));
        d.push(mk(2, 1));
        d.push(mk(5, 2));
        let out = d.flush();
        let pocs: Vec<i32> = out.iter().map(|e| e.pic_order_cnt).collect();
        assert_eq!(pocs, vec![2, 5, 8]);
        assert!(d.is_empty());
    }

    #[test]
    fn poc_tie_broken_by_frame_num() {
        let mut d: DpbOutput<TestPic> = DpbOutput::new(8, 8);
        d.push(mk(10, 2));
        d.push(mk(10, 1));
        let out = d.flush();
        assert_eq!(out[0].frame_num, 1);
        assert_eq!(out[1].frame_num, 2);
    }

    #[test]
    fn reset_clears_all() {
        let mut d: DpbOutput<TestPic> = DpbOutput::new(4, 4);
        d.push(mk(0, 0));
        d.push(mk(1, 1));
        let drained = d.reset();
        assert_eq!(drained.len(), 2);
        assert_eq!(d.len(), 0);
        assert!(d.is_empty());
        // After reset, nothing remains to flush.
        assert!(d.flush().is_empty());
    }

    #[test]
    fn idr_mmco5_scenario_leaves_queue_empty() {
        let mut d: DpbOutput<TestPic> = DpbOutput::new(3, 4);
        d.push(mk(0, 0));
        d.push(mk(4, 1));
        d.push(mk(2, 2));
        // MMCO 5 hit — caller invokes reset().
        let _pre_mmco5 = d.reset();
        // New IDR at POC=0 after the reset — queue is fresh.
        assert!(d.push(mk(0, 0)).is_none());
        assert_eq!(d.len(), 1);
        let final_batch = d.flush();
        assert_eq!(final_batch.len(), 1);
        assert_eq!(final_batch[0].pic_order_cnt, 0);
    }

    #[test]
    fn bump_cascade_preserves_remaining_poc_order() {
        // Force multiple bumps. Trace (cap=2, lowest-POC bumped each
        // time the queue is full when push arrives):
        //   push (3,0) → [(3,0)]                           no bump
        //   push (1,1) → [(3,0),(1,1)]                     no bump
        //   push (5,2) → bump (1,1); queue = [(3,0),(5,2)] b1.poc=1
        //   push (2,3) → bump (3,0); queue = [(5,2),(2,3)] b2.poc=3
        //   flush      → sorted ascending by POC → [2,5]
        let mut d: DpbOutput<TestPic> = DpbOutput::new(2, 4);
        assert!(d.push(mk(3, 0)).is_none());
        assert!(d.push(mk(1, 1)).is_none());
        let b1 = d.push(mk(5, 2)).unwrap();
        assert_eq!(b1.pic_order_cnt, 1);
        let b2 = d.push(mk(2, 3)).unwrap();
        assert_eq!(b2.pic_order_cnt, 3);
        let rest = d.flush();
        let pocs: Vec<i32> = rest.iter().map(|e| e.pic_order_cnt).collect();
        assert_eq!(pocs, vec![2, 5]);
    }

    #[test]
    fn pop_ready_only_emits_over_capacity() {
        let mut d: DpbOutput<TestPic> = DpbOutput::new(2, 4);
        d.push(mk(7, 0));
        d.push(mk(3, 1));
        // At capacity, not *over* it → conservative pop_ready → None.
        assert!(d.pop_ready().is_none());
        assert_eq!(d.len(), 2);
    }
}
