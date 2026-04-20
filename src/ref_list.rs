//! §8.2.4 + §8.2.5 — Reference picture list construction and marking.
//!
//! Pure state-machine operations on a DPB (decoded picture buffer)
//! and per-slice reference lists. No bitstream dependency.
//!
//! This module implements:
//!  - §8.2.4.1 — Decoding process for picture numbers.
//!  - §8.2.4.2.1 — Initialisation process for reference picture list
//!    for P and SP slices in frames.
//!  - §8.2.4.2.3 — Initialisation process for reference picture lists
//!    for B slices in frames.
//!  - §8.2.4.3  — Modification process for reference picture lists.
//!  - §8.2.5.1  — Sequence of operations for decoded reference picture
//!    marking.
//!  - §8.2.5.3  — Sliding window decoded reference picture marking.
//!  - §8.2.5.4  — Adaptive memory control decoded reference picture
//!    marking (MMCO 1..=6).
//!
//! Field-specific initialisation (§8.2.4.2.2 / §8.2.4.2.4 / §8.2.4.2.5)
//! is partially supported: the API takes a `PicStructure` and `current_bottom`
//! so that `PicNum` derivations for fields are correct, but list
//! interleaving by parity is not yet modelled — fields are treated as
//! single-parity reference frames. This keeps the state machine correct
//! for frame-only coded streams, which is the common case.

#![allow(dead_code)]

/// Marking state of a decoded picture in the DPB.
///
/// §8.2.5 — a reference picture is either short-term, long-term, or
/// "unused for reference" (which we represent as `Unused` and which
/// may then be evicted from the DPB by the caller at its discretion).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefMarking {
    ShortTerm,
    LongTerm,
    /// Marked as "unused for reference".
    Unused,
}

/// Field / frame descriptor per §8.2.4.1. Used for both DPB entries and
/// slice header data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PicStructure {
    TopField,
    BottomField,
    Frame,
    /// Complementary field pair (two fields that share a frame_num).
    FieldPair,
}

impl PicStructure {
    /// True when this is a coded field (`field_pic_flag == 1` at decode
    /// time).
    fn is_field(self) -> bool {
        matches!(self, PicStructure::TopField | PicStructure::BottomField)
    }

    /// True when this is a bottom field.
    fn is_bottom(self) -> bool {
        matches!(self, PicStructure::BottomField)
    }
}

/// A decoded picture as held in the DPB.
#[derive(Debug, Clone)]
pub struct DpbEntry {
    /// `frame_num` from the slice header(s) of the picture. Only
    /// meaningful for short-term refs (§8.2.4.1).
    pub frame_num: u32,
    pub top_field_order_cnt: i32,
    pub bottom_field_order_cnt: i32,
    /// `PicOrderCnt(picX)` per equation 8-1 — min of top/bottom for a
    /// frame or complementary field pair, or the single field's POC.
    pub pic_order_cnt: i32,
    pub structure: PicStructure,
    pub marking: RefMarking,
    /// `LongTermFrameIdx` when `marking == LongTerm`. Unused otherwise.
    pub long_term_frame_idx: u32,
    /// Storage key — caller-supplied, this module doesn't interpret.
    pub dpb_key: u32,
}

impl DpbEntry {
    /// §8.2.4.1 eq. 8-27 / 8-28 / 8-30 / 8-31 — derive `FrameNumWrap`
    /// and `PicNum` for a short-term reference.
    ///
    /// `current_frame_num` is the `frame_num` value of the current
    /// picture being decoded; `max_frame_num` is `MaxFrameNum` per
    /// eq. 7-10 (`2 ^ (log2_max_frame_num_minus4 + 4)`).
    pub fn pic_num(
        &self,
        current_frame_num: u32,
        max_frame_num: u32,
        current_is_field: bool,
        current_bottom: bool,
    ) -> i32 {
        // eq. 8-27 — FrameNumWrap = FrameNum - MaxFrameNum when
        // FrameNum > frame_num (current), else FrameNum.
        let frame_num_wrap = if self.frame_num > current_frame_num {
            self.frame_num as i64 - max_frame_num as i64
        } else {
            self.frame_num as i64
        };

        if !current_is_field {
            // Frame picture — eq. 8-28: PicNum = FrameNumWrap.
            frame_num_wrap as i32
        } else {
            // Field picture — eq. 8-30 / 8-31. Same parity gets the
            // "+1" boost, opposite parity doesn't. For a reference
            // frame used as a field reference, both parities are
            // considered: we pick the one whose parity matches the
            // referencing field's parity. For field pair / single
            // field DPB entries, the entry's own structure determines
            // parity.
            let same_parity = match self.structure {
                PicStructure::TopField => !current_bottom,
                PicStructure::BottomField => current_bottom,
                // Frame / field pair: treat as if its parity matches
                // the current field (both fields available).
                PicStructure::Frame | PicStructure::FieldPair => true,
            };
            if same_parity {
                (2 * frame_num_wrap + 1) as i32
            } else {
                (2 * frame_num_wrap) as i32
            }
        }
    }

    /// §8.2.4.1 eq. 8-29 / 8-32 / 8-33 — `LongTermPicNum` for a
    /// long-term reference.
    pub fn long_term_pic_num(&self, current_is_field: bool, current_bottom: bool) -> i32 {
        if !current_is_field {
            // eq. 8-29
            self.long_term_frame_idx as i32
        } else {
            let same_parity = match self.structure {
                PicStructure::TopField => !current_bottom,
                PicStructure::BottomField => current_bottom,
                PicStructure::Frame | PicStructure::FieldPair => true,
            };
            if same_parity {
                // eq. 8-32
                (2 * self.long_term_frame_idx + 1) as i32
            } else {
                // eq. 8-33
                (2 * self.long_term_frame_idx) as i32
            }
        }
    }

    /// True if this entry is a short-term reference.
    fn is_short_term(&self) -> bool {
        matches!(self.marking, RefMarking::ShortTerm)
    }

    /// True if this entry is a long-term reference.
    fn is_long_term(&self) -> bool {
        matches!(self.marking, RefMarking::LongTerm)
    }

    /// True if this entry is used for reference at all.
    fn is_ref(&self) -> bool {
        !matches!(self.marking, RefMarking::Unused)
    }
}

/// `modification_of_pic_nums_idc` operation from §7.3.3.1 / Table 7-7.
/// Mirrors the shape of `crate::slice_header::RefPicListModificationOp`
/// but re-declared here so this module has no upward dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RplmOp {
    /// idc 0 — `abs_diff_pic_num_minus1`, subtract from picNumPred.
    Subtract(u32),
    /// idc 1 — `abs_diff_pic_num_minus1`, add to picNumPred.
    Add(u32),
    /// idc 2 — `long_term_pic_num` of the picture to splice in.
    LongTerm(u32),
}

/// MMCO op descriptor — mirrors `slice_header::MmcoOp`.
///
/// §7.3.3.3 Table 7-9 — see each variant's doc comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmcoOp {
    /// op 1 — `difference_of_pic_nums_minus1`.
    MarkShortTermUnused(u32),
    /// op 2 — `long_term_pic_num`.
    MarkLongTermUnused(u32),
    /// op 3 — `(difference_of_pic_nums_minus1, long_term_frame_idx)`.
    AssignLongTerm(u32, u32),
    /// op 4 — `max_long_term_frame_idx_plus1`.
    SetMaxLongTermIdx(u32),
    /// op 5 — reset all refs.
    MarkAllUnused,
    /// op 6 — `long_term_frame_idx` for the current picture.
    AssignCurrentLongTerm(u32),
}

// ---------------------------------------------------------------------
// §8.2.4.2 — Initialisation of reference picture lists.
// ---------------------------------------------------------------------

/// §8.2.4.2.1 — Initialisation process for the reference picture list
/// for P and SP slices in frames.
///
/// Ordering:
///   1. Short-term frames/complementary field pairs, sorted by
///      descending `PicNum`.
///   2. Long-term frames/complementary field pairs, sorted by
///      ascending `LongTermPicNum`.
///
/// Returns a list of `dpb_key` values in the order they should occupy
/// RefPicList0.
pub fn init_ref_pic_list_p(
    dpb: &[DpbEntry],
    current_frame_num: u32,
    max_frame_num: u32,
    current_structure: PicStructure,
    current_bottom: bool,
) -> Vec<u32> {
    let current_is_field = current_structure.is_field();

    // 1. Short-term, descending PicNum.
    let mut short: Vec<(i32, u32)> = dpb
        .iter()
        .filter(|e| e.is_short_term())
        .map(|e| {
            (
                e.pic_num(
                    current_frame_num,
                    max_frame_num,
                    current_is_field,
                    current_bottom,
                ),
                e.dpb_key,
            )
        })
        .collect();
    short.sort_by(|a, b| b.0.cmp(&a.0));

    // 2. Long-term, ascending LongTermPicNum.
    let mut long: Vec<(i32, u32)> = dpb
        .iter()
        .filter(|e| e.is_long_term())
        .map(|e| {
            (
                e.long_term_pic_num(current_is_field, current_bottom),
                e.dpb_key,
            )
        })
        .collect();
    long.sort_by(|a, b| a.0.cmp(&b.0));

    let mut out: Vec<u32> = Vec::with_capacity(short.len() + long.len());
    out.extend(short.into_iter().map(|(_, k)| k));
    out.extend(long.into_iter().map(|(_, k)| k));
    out
}

/// §8.2.4.2.3 — Initialisation process for reference picture lists for
/// B slices in frames.
///
/// RefPicList0 ordering:
///   1a. Short-term refs with `PicOrderCnt(entryShortTerm)` less than
///       `PicOrderCnt(CurrPic)`, sorted by descending PicOrderCnt.
///   1b. Then short-term refs with `PicOrderCnt >= PicOrderCnt(CurrPic)`,
///       sorted by ascending PicOrderCnt.
///   2.  Then long-term refs, sorted by ascending `LongTermPicNum`.
///
/// RefPicList1 ordering:
///   1a. Short-term refs with `PicOrderCnt(entryShortTerm)` greater than
///       `PicOrderCnt(CurrPic)`, sorted by ascending PicOrderCnt.
///   1b. Then short-term refs with `PicOrderCnt <= PicOrderCnt(CurrPic)`,
///       sorted by descending PicOrderCnt.
///   2.  Then long-term refs, sorted by ascending `LongTermPicNum`.
///
/// Plus the tie-breaker: if `RefPicList1 == RefPicList0` and the list
/// has more than one entry, swap positions [0] and [1] of List1.
pub fn init_ref_pic_lists_b(
    dpb: &[DpbEntry],
    current_poc: i32,
    current_structure: PicStructure,
    current_bottom: bool,
) -> (Vec<u32>, Vec<u32>) {
    let current_is_field = current_structure.is_field();

    // Partition short-term refs by POC vs current.
    let mut st_less: Vec<(i32, u32)> = Vec::new();
    let mut st_geq: Vec<(i32, u32)> = Vec::new();
    let mut st_greater: Vec<(i32, u32)> = Vec::new();
    let mut st_leq: Vec<(i32, u32)> = Vec::new();

    for e in dpb.iter().filter(|e| e.is_short_term()) {
        let poc = e.pic_order_cnt;
        if poc < current_poc {
            st_less.push((poc, e.dpb_key));
        } else {
            st_geq.push((poc, e.dpb_key));
        }
        if poc > current_poc {
            st_greater.push((poc, e.dpb_key));
        } else {
            st_leq.push((poc, e.dpb_key));
        }
    }

    // RefPicList0: st_less desc by POC, then st_geq asc by POC.
    st_less.sort_by(|a, b| b.0.cmp(&a.0));
    st_geq.sort_by(|a, b| a.0.cmp(&b.0));

    // RefPicList1: st_greater asc, then st_leq desc.
    st_greater.sort_by(|a, b| a.0.cmp(&b.0));
    st_leq.sort_by(|a, b| b.0.cmp(&a.0));

    // Long-term refs, ascending LongTermPicNum (shared by both lists).
    let mut long: Vec<(i32, u32)> = dpb
        .iter()
        .filter(|e| e.is_long_term())
        .map(|e| {
            (
                e.long_term_pic_num(current_is_field, current_bottom),
                e.dpb_key,
            )
        })
        .collect();
    long.sort_by(|a, b| a.0.cmp(&b.0));

    let mut list0: Vec<u32> = Vec::new();
    list0.extend(st_less.iter().map(|(_, k)| *k));
    list0.extend(st_geq.iter().map(|(_, k)| *k));
    list0.extend(long.iter().map(|(_, k)| *k));

    let mut list1: Vec<u32> = Vec::new();
    list1.extend(st_greater.iter().map(|(_, k)| *k));
    list1.extend(st_leq.iter().map(|(_, k)| *k));
    list1.extend(long.iter().map(|(_, k)| *k));

    // §8.2.4.2.3 final rule — when RefPicList1 has more than one entry
    // and is identical to RefPicList0, swap [0] and [1] of List1.
    if list1.len() > 1 && list1 == list0 {
        list1.swap(0, 1);
    }

    (list0, list1)
}

// ---------------------------------------------------------------------
// §8.2.4.3 — Modification process for reference picture lists.
// ---------------------------------------------------------------------

/// §8.2.4.3 — apply RPLM ops to a reference picture list.
///
/// The initial list provided by the caller is first truncated to the
/// target `num_active` entries (per §8.2.4.2) if it has more, or padded
/// with a sentinel `u32::MAX` ("no reference picture") if it has fewer.
/// Each op then either splices a short-term or long-term ref into the
/// current `refIdxLX` position per §8.2.4.3.1 / §8.2.4.3.2.
///
/// Implementation notes:
///  - picNumLXPred is initialised to `CurrPicNum` and updated after
///    each short-term modification (§8.2.4.3.1).
///  - The temporary "length + 1" trick from the spec is reproduced
///    here: we push the new entry, shift tail entries, then truncate
///    back to `num_active`.
///  - `u32::MAX` is used as the sentinel for "no reference picture".
pub fn modify_ref_pic_list(
    list: &mut Vec<u32>,
    ops: &[RplmOp],
    dpb: &[DpbEntry],
    num_active: u32,
    current_frame_num: u32,
    max_frame_num: u32,
    current_is_field: bool,
    current_bottom: bool,
) {
    // Normalise to exactly `num_active` entries (§8.2.4.2 fall-through).
    let target = num_active as usize;
    if list.len() > target {
        list.truncate(target);
    }
    while list.len() < target {
        list.push(u32::MAX); // sentinel — "no reference picture".
    }

    if ops.is_empty() {
        return;
    }

    // §7.4.3 — CurrPicNum derivation.
    let curr_pic_num: i32 = if current_is_field {
        (2 * current_frame_num + 1) as i32
    } else {
        current_frame_num as i32
    };
    // §7.4.3 — MaxPicNum derivation.
    let max_pic_num: i32 = if current_is_field {
        (2 * max_frame_num) as i32
    } else {
        max_frame_num as i32
    };

    let mut ref_idx_lx: usize = 0;
    let mut pic_num_lx_pred: i32 = curr_pic_num;

    for op in ops {
        match *op {
            RplmOp::Subtract(abs_diff) | RplmOp::Add(abs_diff) => {
                // §8.2.4.3.1 — short-term modification.
                let delta = (abs_diff + 1) as i32;

                // eq. 8-34 / 8-35 — picNumLXNoWrap.
                let pic_num_lx_no_wrap: i32 = if matches!(op, RplmOp::Subtract(_)) {
                    if pic_num_lx_pred - delta < 0 {
                        pic_num_lx_pred - delta + max_pic_num
                    } else {
                        pic_num_lx_pred - delta
                    }
                } else {
                    // Add
                    if pic_num_lx_pred + delta >= max_pic_num {
                        pic_num_lx_pred + delta - max_pic_num
                    } else {
                        pic_num_lx_pred + delta
                    }
                };

                pic_num_lx_pred = pic_num_lx_no_wrap;

                // eq. 8-36 — picNumLX.
                let pic_num_lx: i32 = if pic_num_lx_no_wrap > curr_pic_num {
                    pic_num_lx_no_wrap - max_pic_num
                } else {
                    pic_num_lx_no_wrap
                };

                // Locate the matching short-term ref in the DPB.
                let target_key = dpb
                    .iter()
                    .find(|e| {
                        e.is_short_term()
                            && e.pic_num(
                                current_frame_num,
                                max_frame_num,
                                current_is_field,
                                current_bottom,
                            ) == pic_num_lx
                    })
                    .map(|e| e.dpb_key)
                    .unwrap_or(u32::MAX);

                splice_into_list(
                    list,
                    ref_idx_lx,
                    num_active as usize,
                    target_key,
                    |k| {
                        // PicNumF filter — for this routine, the
                        // "short-term matching" path should skip the
                        // entry we just inserted. An entry that isn't
                        // marked short-term gets PicNumF = MaxPicNum
                        // (per §8.2.4.3.1), which won't equal any
                        // short-term picNumLX — so we only need to
                        // suppress re-matching the same dpb_key. The
                        // spec uses the "PicNumF(RefPicListX[cIdx]) !=
                        // picNumLX" test; we implement the equivalent
                        // "dbp_key != target" check because an entry
                        // at a new list index corresponds to exactly
                        // one DPB pic per our mapping.
                        k != target_key
                    },
                );
                ref_idx_lx += 1;
            }
            RplmOp::LongTerm(long_term_pic_num) => {
                // §8.2.4.3.2 — long-term modification.
                let target_key = dpb
                    .iter()
                    .find(|e| {
                        e.is_long_term()
                            && e.long_term_pic_num(current_is_field, current_bottom)
                                == long_term_pic_num as i32
                    })
                    .map(|e| e.dpb_key)
                    .unwrap_or(u32::MAX);

                splice_into_list(list, ref_idx_lx, num_active as usize, target_key, |k| {
                    k != target_key
                });
                ref_idx_lx += 1;
            }
        }
    }
}

/// Common helper for §8.2.4.3.1 eq. 8-37 and §8.2.4.3.2 eq. 8-38 —
/// shift entries right from `ref_idx_lx`, insert `target_key`, then
/// drop any existing occurrence of `target_key` further along in the
/// list (the `filter_retain` predicate picks which entries survive
/// the compact step).
///
/// The spec temporarily extends the list length by 1 during the
/// procedure and truncates back to `num_active` at the end; we emulate
/// that by working on a temporary vec.
fn splice_into_list<F>(
    list: &mut Vec<u32>,
    ref_idx_lx: usize,
    num_active: usize,
    target_key: u32,
    filter_retain: F,
) where
    F: Fn(u32) -> bool,
{
    if ref_idx_lx >= num_active {
        return;
    }
    // Temporary "one element longer" vec per the spec's pseudo-code.
    let mut tmp: Vec<u32> = Vec::with_capacity(num_active + 1);
    // Copy [0 .. ref_idx_lx] unchanged.
    tmp.extend_from_slice(&list[..ref_idx_lx]);
    // Insert the target.
    tmp.push(target_key);
    // Append the rest, filtering out the duplicate of target_key.
    for &k in &list[ref_idx_lx..] {
        if filter_retain(k) {
            tmp.push(k);
        }
    }
    // Re-pad / truncate to num_active.
    while tmp.len() < num_active {
        tmp.push(u32::MAX);
    }
    tmp.truncate(num_active);
    *list = tmp;
}

// ---------------------------------------------------------------------
// §8.2.5 — Decoded reference picture marking.
// ---------------------------------------------------------------------

/// §8.2.5.3 — Sliding window decoded reference picture marking.
///
/// When `numShortTerm + numLongTerm == Max(max_num_ref_frames, 1)`,
/// mark as "unused for reference" the short-term ref with the smallest
/// `FrameNumWrap` value. We need the current `frame_num` to compute
/// `FrameNumWrap` correctly, but the spec §8.2.5.3 specifies "smallest
/// value of FrameNumWrap" — i.e., the oldest short-term in display
/// order modulo wrap. Callers that haven't yet assigned a frame_num to
/// the current picture can pass 0 / `max_frame_num` safely as the
/// ordering by FrameNumWrap of existing refs is invariant under the
/// wrap mapping so long as all refs are compared consistently.
///
/// For simplicity and to avoid threading yet another parameter in, we
/// sort by `frame_num` treating `frame_num > current_frame_num` values
/// as "wrapped" (i.e., `frame_num - max_frame_num`) as §8.2.4.1
/// eq. 8-27 prescribes.
pub fn sliding_window_marking(
    dpb: &mut [DpbEntry],
    max_num_ref_frames: u32,
    current_frame_num: u32,
    max_frame_num: u32,
) {
    let num_short = dpb.iter().filter(|e| e.is_short_term()).count() as u32;
    let num_long = dpb.iter().filter(|e| e.is_long_term()).count() as u32;
    let cap = max_num_ref_frames.max(1);

    if num_short + num_long < cap {
        return; // No eviction needed.
    }
    if num_short == 0 {
        return; // Nothing short-term to evict (spec's "numShortTerm > 0"
                // precondition — a conformant stream shouldn't hit this
                // branch, but we guard anyway).
    }

    // Find index of short-term ref with smallest FrameNumWrap.
    let mut best: Option<(usize, i64)> = None;
    for (i, e) in dpb.iter().enumerate() {
        if !e.is_short_term() {
            continue;
        }
        let fnw: i64 = if e.frame_num > current_frame_num {
            e.frame_num as i64 - max_frame_num as i64
        } else {
            e.frame_num as i64
        };
        match best {
            None => best = Some((i, fnw)),
            Some((_, cur_fnw)) if fnw < cur_fnw => best = Some((i, fnw)),
            _ => {}
        }
    }
    if let Some((i, _)) = best {
        dpb[i].marking = RefMarking::Unused;
    }
}

/// §8.2.5.4 — Adaptive memory control decoded reference picture
/// marking. Applies MMCO ops 1..=6 in order.
///
/// `current_entry_ref` represents the *current* picture that's about
/// to be added to the DPB as a reference; this function does NOT add
/// it — it only adjusts the marking of `current_entry_ref` for MMCO 6
/// (mark current as long-term) and updates existing DPB entries for
/// ops 1..=5. Ops 3 and 6 may also evict a previous picture that
/// shares the target `LongTermFrameIdx`.
///
/// Returns `true` if MMCO 5 was applied — caller then resets POC /
/// frame_num state per §8.2.1 NOTE 1.
pub fn apply_mmco(
    dpb: &mut Vec<DpbEntry>,
    ops: &[MmcoOp],
    current_entry_ref: &mut DpbEntry,
    current_frame_num: u32,
    max_frame_num: u32,
) -> bool {
    let mut mmco5 = false;

    // §7.4.3 — CurrPicNum.
    let current_is_field = current_entry_ref.structure.is_field();
    let current_bottom = current_entry_ref.structure.is_bottom();
    let curr_pic_num: i32 = if current_is_field {
        (2 * current_frame_num + 1) as i32
    } else {
        current_frame_num as i32
    };

    for op in ops {
        match *op {
            MmcoOp::MarkShortTermUnused(diff) => {
                // §8.2.5.4.1 eq. 8-39.
                let pic_num_x = curr_pic_num - (diff as i32 + 1);
                for e in dpb.iter_mut() {
                    if e.is_short_term()
                        && e.pic_num(
                            current_frame_num,
                            max_frame_num,
                            current_is_field,
                            current_bottom,
                        ) == pic_num_x
                    {
                        e.marking = RefMarking::Unused;
                    }
                }
            }
            MmcoOp::MarkLongTermUnused(ltpn) => {
                // §8.2.5.4.2.
                for e in dpb.iter_mut() {
                    if e.is_long_term()
                        && e.long_term_pic_num(current_is_field, current_bottom) == ltpn as i32
                    {
                        e.marking = RefMarking::Unused;
                    }
                }
            }
            MmcoOp::AssignLongTerm(diff, ltfi) => {
                // §8.2.5.4.3. First, evict any existing long-term with
                // the same LongTermFrameIdx.
                for e in dpb.iter_mut() {
                    if e.is_long_term() && e.long_term_frame_idx == ltfi {
                        e.marking = RefMarking::Unused;
                    }
                }
                // Then promote the short-term ref identified by
                // picNumX to long-term.
                let pic_num_x = curr_pic_num - (diff as i32 + 1);
                for e in dpb.iter_mut() {
                    if e.is_short_term()
                        && e.pic_num(
                            current_frame_num,
                            max_frame_num,
                            current_is_field,
                            current_bottom,
                        ) == pic_num_x
                    {
                        e.marking = RefMarking::LongTerm;
                        e.long_term_frame_idx = ltfi;
                    }
                }
            }
            MmcoOp::SetMaxLongTermIdx(max_plus1) => {
                // §8.2.5.4.4 — MaxLongTermFrameIdx = max_plus1 - 1 when
                // max_plus1 > 0; "no long-term frame indices" when 0.
                if max_plus1 == 0 {
                    // All long-terms become unused.
                    for e in dpb.iter_mut() {
                        if e.is_long_term() {
                            e.marking = RefMarking::Unused;
                        }
                    }
                } else {
                    let max_ltfi = max_plus1 - 1;
                    for e in dpb.iter_mut() {
                        if e.is_long_term() && e.long_term_frame_idx > max_ltfi {
                            e.marking = RefMarking::Unused;
                        }
                    }
                }
            }
            MmcoOp::MarkAllUnused => {
                // §8.2.5.4.5. Also sets MaxLongTermFrameIdx to "no
                // long-term frame indices" — we express that implicitly
                // by leaving no long-term entries in the DPB; a caller
                // that tracks `MaxLongTermFrameIdx` separately should
                // reset it when `mmco5_triggered == true`.
                for e in dpb.iter_mut() {
                    e.marking = RefMarking::Unused;
                }
                mmco5 = true;
            }
            MmcoOp::AssignCurrentLongTerm(ltfi) => {
                // §8.2.5.4.6 — mark the *current* picture as long-term.
                // If a different pic already carries this LongTermFrameIdx,
                // mark it as unused first.
                for e in dpb.iter_mut() {
                    if e.is_long_term() && e.long_term_frame_idx == ltfi {
                        e.marking = RefMarking::Unused;
                    }
                }
                current_entry_ref.marking = RefMarking::LongTerm;
                current_entry_ref.long_term_frame_idx = ltfi;
            }
        }
    }

    mmco5
}

/// §8.2.5.1 — combined decoded reference picture marking process.
///
/// This is a convenience wrapper that dispatches to:
///  - IDR path: mark all existing refs unused, then mark the current
///    as short-term (or long-term if `long_term_reference_flag_for_idr`).
///  - Non-IDR adaptive path: invoke `apply_mmco`.
///  - Non-IDR sliding-window path: invoke `sliding_window_marking`.
///
/// Returns `true` iff MMCO 5 was triggered. `no_output_of_prior_pics_flag`
/// is accepted here for future use (it affects output decisions in
/// Annex C, not marking) and currently unused.
pub fn perform_marking(
    dpb: &mut Vec<DpbEntry>,
    current_entry: &mut DpbEntry,
    max_num_ref_frames: u32,
    is_idr: bool,
    long_term_reference_flag_for_idr: bool,
    _no_output_of_prior_pics_flag: bool,
    adaptive_ops: Option<&[MmcoOp]>,
    current_frame_num: u32,
    max_frame_num: u32,
) -> bool {
    if is_idr {
        // §8.2.5.1 — all reference pictures are marked as "unused for
        // reference". Then, based on long_term_reference_flag, either
        // mark the IDR itself as short-term (and reset MaxLongTermFrameIdx
        // to "no long-term frame indices") or as long-term with
        // LongTermFrameIdx = 0 and MaxLongTermFrameIdx = 0.
        for e in dpb.iter_mut() {
            e.marking = RefMarking::Unused;
        }
        if long_term_reference_flag_for_idr {
            current_entry.marking = RefMarking::LongTerm;
            current_entry.long_term_frame_idx = 0;
        } else {
            current_entry.marking = RefMarking::ShortTerm;
        }
        return false;
    }

    match adaptive_ops {
        Some(ops) => {
            let mmco5 =
                apply_mmco(dpb, ops, current_entry, current_frame_num, max_frame_num);
            // §8.2.5.1 step 3 — if current not marked as long-term by
            // MMCO 6, mark it as short-term.
            if !matches!(current_entry.marking, RefMarking::LongTerm) {
                current_entry.marking = RefMarking::ShortTerm;
            }
            mmco5
        }
        None => {
            sliding_window_marking(dpb, max_num_ref_frames, current_frame_num, max_frame_num);
            // §8.2.5.1 step 3.
            current_entry.marking = RefMarking::ShortTerm;
            false
        }
    }
}

// ---------------------------------------------------------------------
// Tests — hand-computed from spec equations.
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn st_frame(frame_num: u32, poc: i32, key: u32) -> DpbEntry {
        DpbEntry {
            frame_num,
            top_field_order_cnt: poc,
            bottom_field_order_cnt: poc,
            pic_order_cnt: poc,
            structure: PicStructure::Frame,
            marking: RefMarking::ShortTerm,
            long_term_frame_idx: 0,
            dpb_key: key,
        }
    }

    fn lt_frame(ltfi: u32, poc: i32, key: u32) -> DpbEntry {
        DpbEntry {
            frame_num: 0,
            top_field_order_cnt: poc,
            bottom_field_order_cnt: poc,
            pic_order_cnt: poc,
            structure: PicStructure::Frame,
            marking: RefMarking::LongTerm,
            long_term_frame_idx: ltfi,
            dpb_key: key,
        }
    }

    // §8.2.4.1 eq. 8-27 / 8-28 — PicNum derivation with frame_num wrap.
    #[test]
    fn pic_num_no_wrap() {
        // max_frame_num = 16, current frame_num = 5, ref frame_num = 3.
        let e = st_frame(3, 0, 1);
        let pn = e.pic_num(5, 16, false, false);
        // FrameNumWrap = FrameNum = 3 (since FrameNum <= frame_num).
        assert_eq!(pn, 3);
    }

    #[test]
    fn pic_num_wrap() {
        // max_frame_num = 16, current frame_num = 2, ref frame_num = 15.
        let e = st_frame(15, 0, 1);
        let pn = e.pic_num(2, 16, false, false);
        // FrameNumWrap = 15 - 16 = -1, so PicNum = -1.
        assert_eq!(pn, -1);
    }

    #[test]
    fn long_term_pic_num_frame() {
        let e = lt_frame(7, 0, 1);
        assert_eq!(e.long_term_pic_num(false, false), 7);
    }

    #[test]
    fn long_term_pic_num_field_same_parity() {
        let mut e = lt_frame(3, 0, 1);
        e.structure = PicStructure::TopField;
        // current field is top (bottom=false), ref is top — same parity.
        assert_eq!(e.long_term_pic_num(true, false), 2 * 3 + 1);
    }

    #[test]
    fn long_term_pic_num_field_opposite_parity() {
        let mut e = lt_frame(3, 0, 1);
        e.structure = PicStructure::TopField;
        // current field is bottom, ref is top — opposite parity.
        assert_eq!(e.long_term_pic_num(true, true), 2 * 3);
    }

    // §8.2.4.2.1 — P slice RefPicList0 init with only short-term refs.
    #[test]
    fn init_p_list_short_only() {
        // DPB frame_nums = [1,2,3,4], current frame_num = 5,
        // max_frame_num = 16. PicNums = FrameNumWrap = [1,2,3,4].
        // RefPicList0 should be ordered descending by PicNum: [4,3,2,1].
        let dpb = vec![
            st_frame(1, 0, 101),
            st_frame(2, 0, 102),
            st_frame(3, 0, 103),
            st_frame(4, 0, 104),
        ];
        let list = init_ref_pic_list_p(&dpb, 5, 16, PicStructure::Frame, false);
        assert_eq!(list, vec![104, 103, 102, 101]);
    }

    // §8.2.4.2.1 example from spec (page 125):
    // "three ST refs with PicNum 300,302,303 and two LT refs with
    //  LongTermPicNum 0 and 3".
    #[test]
    fn init_p_list_spec_example() {
        let dpb = vec![
            st_frame(300, 0, 300),
            st_frame(302, 0, 302),
            st_frame(303, 0, 303),
            lt_frame(0, 0, 1000),
            lt_frame(3, 0, 1003),
        ];
        // current frame_num = 304 (so no wrap — FrameNumWrap = frame_num).
        let list = init_ref_pic_list_p(&dpb, 304, 1 << 12, PicStructure::Frame, false);
        assert_eq!(list, vec![303, 302, 300, 1000, 1003]);
    }

    // §8.2.4.2.3 — B slice: current POC = 10, DPB at POCs {5,6,15,20}.
    // RefPicList0:
    //   st_less (<10) sorted desc POC: [6, 5]
    //   st_geq  (>=10) sorted asc POC: [15, 20]
    //   ⇒ [6, 5, 15, 20].
    // RefPicList1:
    //   st_greater (>10) sorted asc POC: [15, 20]
    //   st_leq     (<=10) sorted desc POC: [6, 5]
    //   ⇒ [15, 20, 6, 5].
    #[test]
    fn init_b_lists_spec_rules() {
        let dpb = vec![
            st_frame(1, 5, 105),
            st_frame(2, 6, 106),
            st_frame(3, 15, 115),
            st_frame(4, 20, 120),
        ];
        let (l0, l1) = init_ref_pic_lists_b(&dpb, 10, PicStructure::Frame, false);
        assert_eq!(l0, vec![106, 105, 115, 120]);
        assert_eq!(l1, vec![115, 120, 106, 105]);
    }

    // §8.2.4.2.3 tie-breaker: if List1 == List0 and len > 1, swap [0]/[1].
    #[test]
    fn init_b_lists_identical_swap() {
        // All refs have POC < current, so List0 = [desc POCs] and
        // List1's "greater" partition is empty and the "leq" partition
        // sorted desc would produce the same order as List0 — triggering
        // the swap.
        let dpb = vec![st_frame(1, 1, 1), st_frame(2, 2, 2), st_frame(3, 3, 3)];
        let (l0, l1) = init_ref_pic_lists_b(&dpb, 10, PicStructure::Frame, false);
        // l0 = st_less desc: [3,2,1].
        // l1 = st_greater asc (empty) + st_leq desc: [3,2,1] — same.
        // Swap [0]/[1] of l1 ⇒ [2,3,1].
        assert_eq!(l0, vec![3, 2, 1]);
        assert_eq!(l1, vec![2, 3, 1]);
    }

    // §8.2.4.2.3 — long-term refs come after short-term, asc LongTermPicNum.
    #[test]
    fn init_b_lists_with_long_term() {
        let dpb = vec![
            st_frame(1, 5, 5),
            st_frame(2, 15, 15),
            lt_frame(2, 0, 92),
            lt_frame(0, 0, 90),
        ];
        let (l0, l1) = init_ref_pic_lists_b(&dpb, 10, PicStructure::Frame, false);
        // l0: st_less=[5], st_geq=[15], long=[90 (ltpn=0), 92 (ltpn=2)]
        assert_eq!(l0, vec![5, 15, 90, 92]);
        // l1: st_greater=[15], st_leq=[5], long=[90, 92]
        assert_eq!(l1, vec![15, 5, 90, 92]);
    }

    // §8.2.4.3.1 — Subtract op: picNumLXPred starts at CurrPicNum.
    #[test]
    fn modify_subtract_applies() {
        // DPB has short-term frames with frame_num 1..=4 (PicNums 1..=4).
        let dpb = vec![
            st_frame(1, 0, 1),
            st_frame(2, 0, 2),
            st_frame(3, 0, 3),
            st_frame(4, 0, 4),
        ];
        // Current frame_num = 5, so CurrPicNum = 5.
        // Initial list = init_ref_pic_list_p: [4,3,2,1].
        let mut list = init_ref_pic_list_p(&dpb, 5, 16, PicStructure::Frame, false);
        // Subtract(0) ⇒ picNumLXNoWrap = 5 - 1 = 4, pic_num_lx = 4.
        // Splice PicNum=4's key (which is 4) into list position 0,
        // shifting duplicates.
        let ops = [RplmOp::Subtract(0)];
        modify_ref_pic_list(&mut list, &ops, &dpb, 4, 5, 16, false, false);
        assert_eq!(list[0], 4);
        // The rest of the list retains its relative order minus the
        // duplicated '4'.
        assert_eq!(&list[1..], &[3, 2, 1]);
    }

    // §8.2.4.3.1 — Add op.
    #[test]
    fn modify_add_applies() {
        // DPB only has frame_num 5 — CurrPicNum = 10, max_frame_num = 16.
        // But MaxPicNum = 16 for frames.
        // If we start pred=10 and Add(5) (abs_diff+1=6), picNumLXNoWrap
        // = 10 + 6 = 16 >= MaxPicNum=16, so NoWrap = 16-16 = 0.
        // Then 0 <= 10 so picNumLX = 0. So we look for a short-term
        // ref with PicNum = 0. Let's give the DPB one: frame_num=0.
        let dpb = vec![st_frame(0, 0, 42)];
        let mut list = vec![42];
        let ops = [RplmOp::Add(5)];
        modify_ref_pic_list(&mut list, &ops, &dpb, 1, 10, 16, false, false);
        assert_eq!(list, vec![42]);
    }

    // §8.2.4.3.2 — LongTerm op.
    #[test]
    fn modify_long_term_applies() {
        let dpb = vec![
            st_frame(1, 0, 10),
            lt_frame(3, 0, 20), // LongTermPicNum = 3 (frame).
            lt_frame(5, 0, 30), // LongTermPicNum = 5.
        ];
        // Initial list = [10, 20, 30] (st desc then lt asc).
        let mut list = init_ref_pic_list_p(&dpb, 5, 16, PicStructure::Frame, false);
        assert_eq!(list, vec![10, 20, 30]);
        let ops = [RplmOp::LongTerm(5)];
        modify_ref_pic_list(&mut list, &ops, &dpb, 3, 5, 16, false, false);
        // Splice LongTermPicNum=5's key (30) to position 0.
        assert_eq!(list, vec![30, 10, 20]);
    }

    // §8.2.5.3 — sliding window.
    #[test]
    fn sliding_window_evicts_oldest() {
        let mut dpb = vec![
            st_frame(1, 0, 1),
            st_frame(2, 0, 2),
            st_frame(3, 0, 3),
        ];
        // max_num_ref=2 with 3 short-term refs ⇒ evict smallest
        // FrameNumWrap = 1 (dpb_key=1).
        sliding_window_marking(&mut dpb, 2, 4, 16);
        assert_eq!(dpb[0].marking, RefMarking::Unused);
        assert_eq!(dpb[1].marking, RefMarking::ShortTerm);
        assert_eq!(dpb[2].marking, RefMarking::ShortTerm);
    }

    #[test]
    fn sliding_window_no_eviction_when_below_cap() {
        let mut dpb = vec![st_frame(1, 0, 1)];
        sliding_window_marking(&mut dpb, 2, 2, 16);
        assert_eq!(dpb[0].marking, RefMarking::ShortTerm);
    }

    // §8.2.5.4.1 — MMCO 1 (mark short-term unused).
    #[test]
    fn mmco_mark_short_term_unused() {
        let mut dpb = vec![st_frame(1, 0, 1), st_frame(2, 0, 2), st_frame(3, 0, 3)];
        let mut current = st_frame(4, 0, 4);
        // CurrPicNum = 4, MMCO 1 diff=0 ⇒ picNumX = 4 - 1 = 3.
        // So DPB entry with PicNum=3 (frame_num=3, dpb_key=3) gets
        // marked unused.
        let ops = [MmcoOp::MarkShortTermUnused(0)];
        let mmco5 = apply_mmco(&mut dpb, &ops, &mut current, 4, 16);
        assert!(!mmco5);
        assert_eq!(dpb[0].marking, RefMarking::ShortTerm);
        assert_eq!(dpb[1].marking, RefMarking::ShortTerm);
        assert_eq!(dpb[2].marking, RefMarking::Unused);
    }

    // §8.2.5.4.2 — MMCO 2 (mark long-term unused).
    #[test]
    fn mmco_mark_long_term_unused() {
        let mut dpb = vec![lt_frame(0, 0, 100), lt_frame(3, 0, 103)];
        let mut current = st_frame(4, 0, 4);
        let ops = [MmcoOp::MarkLongTermUnused(3)];
        let mmco5 = apply_mmco(&mut dpb, &ops, &mut current, 4, 16);
        assert!(!mmco5);
        assert_eq!(dpb[0].marking, RefMarking::LongTerm);
        assert_eq!(dpb[1].marking, RefMarking::Unused);
    }

    // §8.2.5.4.3 — MMCO 3 (assign short-term to long-term).
    #[test]
    fn mmco_assign_long_term() {
        let mut dpb = vec![
            st_frame(2, 0, 2),
            st_frame(3, 0, 3),
            lt_frame(5, 0, 105), // will be evicted if new target has ltfi=5
        ];
        let mut current = st_frame(4, 0, 4);
        // CurrPicNum=4, diff=0 ⇒ picNumX=3 (frame_num=3, dpb_key=3).
        // Assign ltfi=5. The existing LT with ltfi=5 (dpb_key=105) must
        // be evicted first.
        let ops = [MmcoOp::AssignLongTerm(0, 5)];
        let mmco5 = apply_mmco(&mut dpb, &ops, &mut current, 4, 16);
        assert!(!mmco5);
        assert_eq!(dpb[0].marking, RefMarking::ShortTerm); // key 2 untouched
        assert_eq!(dpb[1].marking, RefMarking::LongTerm); // key 3 promoted
        assert_eq!(dpb[1].long_term_frame_idx, 5);
        assert_eq!(dpb[2].marking, RefMarking::Unused); // key 105 evicted
    }

    // §8.2.5.4.4 — MMCO 4 (set MaxLongTermFrameIdx).
    #[test]
    fn mmco_set_max_long_term_idx() {
        let mut dpb = vec![
            lt_frame(0, 0, 10),
            lt_frame(3, 0, 13),
            lt_frame(7, 0, 17),
        ];
        let mut current = st_frame(5, 0, 5);
        // max_long_term_frame_idx_plus1 = 4 ⇒ MaxLTFI = 3.
        // LTs with ltfi > 3 (only ltfi=7) get marked unused.
        let ops = [MmcoOp::SetMaxLongTermIdx(4)];
        apply_mmco(&mut dpb, &ops, &mut current, 5, 16);
        assert_eq!(dpb[0].marking, RefMarking::LongTerm);
        assert_eq!(dpb[1].marking, RefMarking::LongTerm);
        assert_eq!(dpb[2].marking, RefMarking::Unused);
    }

    #[test]
    fn mmco_set_max_long_term_idx_zero_clears_all() {
        let mut dpb = vec![lt_frame(0, 0, 10), lt_frame(3, 0, 13)];
        let mut current = st_frame(5, 0, 5);
        let ops = [MmcoOp::SetMaxLongTermIdx(0)];
        apply_mmco(&mut dpb, &ops, &mut current, 5, 16);
        assert_eq!(dpb[0].marking, RefMarking::Unused);
        assert_eq!(dpb[1].marking, RefMarking::Unused);
    }

    // §8.2.5.4.5 — MMCO 5 (mark all unused).
    #[test]
    fn mmco_mark_all_unused() {
        let mut dpb = vec![st_frame(1, 0, 1), lt_frame(0, 0, 10)];
        let mut current = st_frame(5, 0, 5);
        let ops = [MmcoOp::MarkAllUnused];
        let mmco5 = apply_mmco(&mut dpb, &ops, &mut current, 5, 16);
        assert!(mmco5);
        assert_eq!(dpb[0].marking, RefMarking::Unused);
        assert_eq!(dpb[1].marking, RefMarking::Unused);
    }

    // §8.2.5.4.6 — MMCO 6 (current ⇒ long-term).
    #[test]
    fn mmco_assign_current_long_term() {
        let mut dpb = vec![lt_frame(3, 0, 103)];
        let mut current = st_frame(5, 0, 5);
        // Request ltfi=3 — the existing LT with ltfi=3 must be
        // evicted.
        let ops = [MmcoOp::AssignCurrentLongTerm(3)];
        apply_mmco(&mut dpb, &ops, &mut current, 5, 16);
        assert_eq!(dpb[0].marking, RefMarking::Unused);
        assert_eq!(current.marking, RefMarking::LongTerm);
        assert_eq!(current.long_term_frame_idx, 3);
    }

    // §8.2.5.1 — IDR path.
    #[test]
    fn marking_idr_short_term() {
        let mut dpb = vec![st_frame(1, 0, 1), lt_frame(0, 0, 10)];
        let mut current = st_frame(0, 0, 99);
        let mmco5 = perform_marking(
            &mut dpb,
            &mut current,
            4,
            true,  // is_idr
            false, // long_term_reference_flag (IDR)
            false,
            None,
            0,
            16,
        );
        assert!(!mmco5);
        assert_eq!(dpb[0].marking, RefMarking::Unused);
        assert_eq!(dpb[1].marking, RefMarking::Unused);
        assert_eq!(current.marking, RefMarking::ShortTerm);
    }

    #[test]
    fn marking_idr_long_term() {
        let mut dpb = vec![st_frame(1, 0, 1)];
        let mut current = st_frame(0, 0, 99);
        perform_marking(&mut dpb, &mut current, 4, true, true, false, None, 0, 16);
        assert_eq!(dpb[0].marking, RefMarking::Unused);
        assert_eq!(current.marking, RefMarking::LongTerm);
        assert_eq!(current.long_term_frame_idx, 0);
    }

    // §8.2.5.1 — non-IDR sliding-window path.
    #[test]
    fn marking_non_idr_sliding_window() {
        let mut dpb = vec![
            st_frame(1, 0, 1),
            st_frame(2, 0, 2),
            st_frame(3, 0, 3),
        ];
        let mut current = st_frame(4, 0, 4);
        let mmco5 = perform_marking(
            &mut dpb,
            &mut current,
            2, // max_num_ref_frames
            false, false, false, None, 4, 16,
        );
        assert!(!mmco5);
        // Oldest ST (frame_num=1) evicted.
        assert_eq!(dpb[0].marking, RefMarking::Unused);
        assert_eq!(current.marking, RefMarking::ShortTerm);
    }

    // §8.2.5.1 — non-IDR adaptive path with MMCO 5.
    #[test]
    fn marking_non_idr_adaptive_mmco5() {
        let mut dpb = vec![st_frame(1, 0, 1)];
        let mut current = st_frame(4, 0, 4);
        let ops = [MmcoOp::MarkAllUnused];
        let mmco5 = perform_marking(
            &mut dpb,
            &mut current,
            4,
            false,
            false,
            false,
            Some(&ops),
            4,
            16,
        );
        assert!(mmco5);
        assert_eq!(dpb[0].marking, RefMarking::Unused);
        // Step 3: current is not marked LT ⇒ becomes ST.
        assert_eq!(current.marking, RefMarking::ShortTerm);
    }

    // §8.2.4.3 — after modification the list is exactly num_active long.
    #[test]
    fn modify_pads_and_truncates() {
        let dpb = vec![st_frame(1, 0, 1)];
        let mut list = vec![1];
        // num_active = 3 but list has 1 entry — should pad with sentinel.
        modify_ref_pic_list(&mut list, &[], &dpb, 3, 2, 16, false, false);
        assert_eq!(list.len(), 3);
        assert_eq!(list[0], 1);
        assert_eq!(list[1], u32::MAX);
        assert_eq!(list[2], u32::MAX);
    }
}
