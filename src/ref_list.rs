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
//! is implemented by [`init_ref_pic_list_p_field`] /
//! [`init_ref_pic_lists_b_field`], which build per-field
//! [`RefFieldEntry`] lists via the §8.2.4.2.5 parity-alternation
//! interleave, and [`modify_ref_pic_list_field`] (§8.2.4.3) applies RPLM
//! ops to them using the field forms of PicNum / LongTermPicNum. The
//! frame-only entry points ([`init_ref_pic_list_p`] /
//! [`init_ref_pic_lists_b`] / [`modify_ref_pic_list`]) return `dpb_key`
//! lists and remain the path used for frame-coded streams. Wiring the
//! field lists into pixel reconstruction is gated on the field-coded
//! reconstruction path (`field_pic_flag == 1`), which is a separate
//! phase; these functions are complete and unit-tested against the spec
//! equations independently of that wiring.

#![allow(dead_code)]
// Spec-driven §8.2.4 / §8.2.5 ops legitimately take many parameters
// (DPB + slice + POC + frame_num + MMCO ops + marking flags). Suppress
// clippy's too_many_arguments.
#![allow(clippy::too_many_arguments)]

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

/// Parity of a single reference field. Used by the §8.2.4.2.5
/// parity-alternation interleave to designate *which* field of a stored
/// reference frame occupies a given `RefPicListX` index when the current
/// picture is a coded field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldParity {
    Top,
    Bottom,
}

impl FieldParity {
    /// The opposite parity.
    fn flip(self) -> FieldParity {
        match self {
            FieldParity::Top => FieldParity::Bottom,
            FieldParity::Bottom => FieldParity::Top,
        }
    }
}

/// One entry of a field-coded `RefPicListX` (§8.2.4.2.2 / §8.2.4.2.4).
///
/// When decoding a coded field, each *field* of a stored reference frame
/// is a separate reference picture with its own list index (§8.2.4.2.4),
/// so a list entry must name both the DPB frame (`dpb_key`) and the
/// `parity` of the field within it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefFieldEntry {
    pub dpb_key: u32,
    pub parity: FieldParity,
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

    /// §8.2.4.1 eq. 8-30 / 8-31 — `PicNum` of a *specific parity field*
    /// of this stored reference frame, as seen by a current field of
    /// parity `current_bottom`.
    ///
    /// Unlike [`Self::pic_num`] (which infers the field parity from
    /// `self.structure`), this names the parity explicitly via
    /// `field_parity` — needed by §8.2.4.3.1 field modification, where a
    /// stored *frame* can supply either of its two parity fields and the
    /// PicNum hinges on whether that chosen field matches the current
    /// field's parity.
    fn field_pic_num(
        &self,
        field_parity: FieldParity,
        current_frame_num: u32,
        max_frame_num: u32,
        current_bottom: bool,
    ) -> i32 {
        let frame_num_wrap = if self.frame_num > current_frame_num {
            self.frame_num as i64 - max_frame_num as i64
        } else {
            self.frame_num as i64
        };
        let same_parity = (field_parity == FieldParity::Bottom) == current_bottom;
        if same_parity {
            (2 * frame_num_wrap + 1) as i32
        } else {
            (2 * frame_num_wrap) as i32
        }
    }

    /// §8.2.4.1 eq. 8-32 / 8-33 — `LongTermPicNum` of a *specific parity
    /// field* of this stored long-term reference frame.
    fn field_long_term_pic_num(&self, field_parity: FieldParity, current_bottom: bool) -> i32 {
        let same_parity = (field_parity == FieldParity::Bottom) == current_bottom;
        if same_parity {
            (2 * self.long_term_frame_idx + 1) as i32
        } else {
            (2 * self.long_term_frame_idx) as i32
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
    pub fn is_long_term(&self) -> bool {
        matches!(self.marking, RefMarking::LongTerm)
    }

    /// True if this entry is used for reference at all.
    fn is_ref(&self) -> bool {
        !matches!(self.marking, RefMarking::Unused)
    }

    /// §8.2.4.2.5 — does this stored frame carry a *decoded* field of the
    /// given `parity`?
    ///
    /// A `Frame` or complementary `FieldPair` carries both parities; a
    /// single coded field (`TopField` / `BottomField`) carries only its
    /// own parity. The §8.2.4.2.5 interleave skips ("the missing field is
    /// ignored") any frame whose field of the requested parity was not
    /// decoded or is not marked with the matching reference type.
    fn has_field(&self, parity: FieldParity) -> bool {
        match self.structure {
            PicStructure::Frame | PicStructure::FieldPair => true,
            PicStructure::TopField => parity == FieldParity::Top,
            PicStructure::BottomField => parity == FieldParity::Bottom,
        }
    }

    /// `PicOrderCnt` of the individual field of the given `parity`, per
    /// §8.2.4.1 (`top_field_order_cnt` / `bottom_field_order_cnt`). For a
    /// frame/pair the two may differ; for a single field both members hold
    /// that field's POC.
    fn field_poc(&self, parity: FieldParity) -> i32 {
        match parity {
            FieldParity::Top => self.top_field_order_cnt,
            FieldParity::Bottom => self.bottom_field_order_cnt,
        }
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
    short.sort_by_key(|e| std::cmp::Reverse(e.0));

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
    long.sort_by_key(|a| a.0);

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
    st_less.sort_by_key(|e| std::cmp::Reverse(e.0));
    st_geq.sort_by_key(|a| a.0);

    // RefPicList1: st_greater asc, then st_leq desc.
    st_greater.sort_by_key(|a| a.0);
    st_leq.sort_by_key(|e| std::cmp::Reverse(e.0));

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
    long.sort_by_key(|a| a.0);

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
// §8.2.4.2.2 / §8.2.4.2.4 / §8.2.4.2.5 — Field reference-list init.
//
// When the current picture is a coded field (`field_pic_flag == 1`) the
// initialisation runs in two stages:
//
//   1. An ordered list of reference *frames* is built (§8.2.4.2.2 for
//      P/SP, §8.2.4.2.4 for B) using the same frame-level FrameNumWrap /
//      PicOrderCnt ordering as the frame cases, but treating any frame
//      with at least one referenced field as a list member.
//   2. §8.2.4.2.5 walks that ordered frame list and emits one
//      `RefFieldEntry` per field, alternating parity (starting with the
//      current field's own parity) and skipping frames whose field of the
//      wanted parity is absent. When a parity runs out, the remaining
//      fields of the other parity are appended in frame order.
//
// Each emitted entry names a specific (`dpb_key`, `parity`) field, since
// §8.2.4.2.4 makes every field of a stored frame a separate reference
// picture with its own list index.
// ---------------------------------------------------------------------

/// §8.2.4.2.5 — Initialisation process for reference picture lists in
/// fields. Given an ordered list of reference-frame `dpb_key`s and the
/// `current_parity` of the field being decoded, interleave the per-frame
/// fields by alternating parity.
///
/// `frame_order` is the §8.2.4.2.2 / §8.2.4.2.4 ordered frame list (each
/// a `dpb_key`); `dpb` is consulted via `has_field` to know which
/// parities each frame actually carries. The same helper serves both the
/// short-term and long-term stages (the spec text for the two stages is
/// identical except for the marking type, which the caller already
/// filtered into `frame_order`).
fn interleave_fields(
    frame_order: &[u32],
    dpb: &[DpbEntry],
    current_parity: FieldParity,
) -> Vec<RefFieldEntry> {
    // Index from dpb_key → entry for the `has_field` lookups.
    let lookup = |key: u32| -> Option<&DpbEntry> { dpb.iter().find(|e| e.dpb_key == key) };

    let mut out: Vec<RefFieldEntry> = Vec::with_capacity(frame_order.len() * 2);
    // A per-frame "already consumed" flag per parity so the leftover pass
    // doesn't re-emit a field that the alternating pass already placed.
    let mut used_top = vec![false; frame_order.len()];
    let mut used_bot = vec![false; frame_order.len()];

    // Alternating pass — start with the current field's own parity.
    let mut want = current_parity;
    // Per-parity scan cursors into `frame_order`: the spec advances to
    // "the next available stored reference field of the chosen parity".
    let mut cursor_top = 0usize;
    let mut cursor_bot = 0usize;

    loop {
        let (cursor, used) = match want {
            FieldParity::Top => (&mut cursor_top, &mut used_top),
            FieldParity::Bottom => (&mut cursor_bot, &mut used_bot),
        };
        // Advance the cursor to the next frame carrying a `want`-parity
        // field that has not been consumed yet.
        let mut found: Option<usize> = None;
        while *cursor < frame_order.len() {
            let i = *cursor;
            *cursor += 1;
            if used[i] {
                continue;
            }
            if lookup(frame_order[i])
                .map(|e| e.has_field(want))
                .unwrap_or(false)
            {
                found = Some(i);
                break;
            }
        }
        match found {
            Some(i) => {
                used[i] = true;
                out.push(RefFieldEntry {
                    dpb_key: frame_order[i],
                    parity: want,
                });
                want = want.flip();
            }
            None => {
                // No more fields of the wanted parity. If the *other*
                // parity still has any unconsumed field, the spec switches
                // to appending those in frame order; otherwise we are done.
                let other = want.flip();
                let other_has_more = frame_order.iter().enumerate().any(|(i, &k)| {
                    let used_other = match other {
                        FieldParity::Top => used_top[i],
                        FieldParity::Bottom => used_bot[i],
                    };
                    !used_other && lookup(k).map(|e| e.has_field(other)).unwrap_or(false)
                });
                if other_has_more {
                    want = other;
                } else {
                    break;
                }
            }
        }
    }

    out
}

/// §8.2.4.2.2 — Initialisation process for the reference picture list for
/// P and SP slices in fields.
///
/// Builds `refFrameList0ShortTerm` (short-term frames, descending
/// `FrameNumWrap`) and `refFrameList0LongTerm` (long-term frames,
/// ascending `LongTermFrameIdx`), then invokes §8.2.4.2.5 on each and
/// concatenates short-term fields ahead of long-term fields.
///
/// `current_bottom` selects the starting parity for the §8.2.4.2.5
/// alternation (the current field's own parity).
pub fn init_ref_pic_list_p_field(
    dpb: &[DpbEntry],
    current_frame_num: u32,
    max_frame_num: u32,
    current_bottom: bool,
) -> Vec<RefFieldEntry> {
    let current_parity = if current_bottom {
        FieldParity::Bottom
    } else {
        FieldParity::Top
    };

    // refFrameList0ShortTerm — every frame with ≥1 short-term field,
    // ordered by descending FrameNumWrap (eq. 8-27). `pic_num` with
    // `current_is_field = false` yields FrameNumWrap directly (eq. 8-28),
    // which is the frame-level ordering key the spec asks for here.
    let mut short: Vec<(i32, u32)> = dpb
        .iter()
        .filter(|e| e.is_short_term())
        .map(|e| {
            (
                e.pic_num(current_frame_num, max_frame_num, false, false),
                e.dpb_key,
            )
        })
        .collect();
    short.sort_by_key(|e| std::cmp::Reverse(e.0));
    let short_frames: Vec<u32> = short.into_iter().map(|(_, k)| k).collect();

    // refFrameList0LongTerm — ascending LongTermFrameIdx.
    let mut long: Vec<(u32, u32)> = dpb
        .iter()
        .filter(|e| e.is_long_term())
        .map(|e| (e.long_term_frame_idx, e.dpb_key))
        .collect();
    long.sort_by_key(|a| a.0);
    let long_frames: Vec<u32> = long.into_iter().map(|(_, k)| k).collect();

    let mut out = interleave_fields(&short_frames, dpb, current_parity);
    out.extend(interleave_fields(&long_frames, dpb, current_parity));
    out
}

/// §8.2.4.2.4 — Initialisation process for reference picture lists for B
/// slices in fields.
///
/// Builds three ordered frame lists keyed on `PicOrderCnt`:
///
/// * `refFrameList0ShortTerm` — frames with POC ≤ current first
///   (descending POC), then the rest (ascending POC);
/// * `refFrameList1ShortTerm` — frames with POC > current first
///   (ascending POC), then the rest (descending POC);
/// * `refFrameListLongTerm` — ascending `LongTermFrameIdx`.
///
/// Each is fed through §8.2.4.2.5; short-term fields precede long-term
/// fields. Finally, if `RefPicList1` has >1 entry and equals
/// `RefPicList0`, the first two entries of List1 are swapped.
///
/// The per-frame ordering POC is the frame's `pic_order_cnt`
/// (`min(top,bottom)` per eq. 8-1, already stored on the entry), matching
/// the spec's use of `PicOrderCnt(entryShortTerm)` over a frame.
pub fn init_ref_pic_lists_b_field(
    dpb: &[DpbEntry],
    current_poc: i32,
    current_bottom: bool,
) -> (Vec<RefFieldEntry>, Vec<RefFieldEntry>) {
    let current_parity = if current_bottom {
        FieldParity::Bottom
    } else {
        FieldParity::Top
    };

    // List0 frame order: POC ≤ current desc, then POC > current asc.
    let mut l0_le: Vec<(i32, u32)> = Vec::new();
    let mut l0_gt: Vec<(i32, u32)> = Vec::new();
    // List1 frame order: POC > current asc, then POC ≤ current desc.
    let mut l1_gt: Vec<(i32, u32)> = Vec::new();
    let mut l1_le: Vec<(i32, u32)> = Vec::new();

    for e in dpb.iter().filter(|e| e.is_short_term()) {
        let poc = e.pic_order_cnt;
        if poc <= current_poc {
            l0_le.push((poc, e.dpb_key));
            l1_le.push((poc, e.dpb_key));
        } else {
            l0_gt.push((poc, e.dpb_key));
            l1_gt.push((poc, e.dpb_key));
        }
    }
    l0_le.sort_by_key(|e| std::cmp::Reverse(e.0));
    l0_gt.sort_by_key(|a| a.0);
    l1_gt.sort_by_key(|a| a.0);
    l1_le.sort_by_key(|e| std::cmp::Reverse(e.0));

    let mut l0_frames: Vec<u32> = Vec::new();
    l0_frames.extend(l0_le.iter().map(|(_, k)| *k));
    l0_frames.extend(l0_gt.iter().map(|(_, k)| *k));
    let mut l1_frames: Vec<u32> = Vec::new();
    l1_frames.extend(l1_gt.iter().map(|(_, k)| *k));
    l1_frames.extend(l1_le.iter().map(|(_, k)| *k));

    // refFrameListLongTerm — ascending LongTermFrameIdx (shared).
    let mut long: Vec<(u32, u32)> = dpb
        .iter()
        .filter(|e| e.is_long_term())
        .map(|e| (e.long_term_frame_idx, e.dpb_key))
        .collect();
    long.sort_by_key(|a| a.0);
    let long_frames: Vec<u32> = long.into_iter().map(|(_, k)| k).collect();

    let mut list0 = interleave_fields(&l0_frames, dpb, current_parity);
    list0.extend(interleave_fields(&long_frames, dpb, current_parity));

    let mut list1 = interleave_fields(&l1_frames, dpb, current_parity);
    list1.extend(interleave_fields(&long_frames, dpb, current_parity));

    // §8.2.4.2.4 final rule — when RefPicList1 has more than one entry and
    // is identical to RefPicList0, swap [0] and [1] of List1.
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

                splice_into_list(list, ref_idx_lx, num_active as usize, target_key, |k| {
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
                });
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

/// Sentinel "no reference picture" entry for a field list.
fn no_ref_field() -> RefFieldEntry {
    RefFieldEntry {
        dpb_key: u32::MAX,
        parity: FieldParity::Top,
    }
}

/// §8.2.4.3 — apply RPLM ops to a *field-coded* reference picture list
/// (`field_pic_flag == 1`).
///
/// Operates on the `RefFieldEntry` lists produced by
/// [`init_ref_pic_list_p_field`] / [`init_ref_pic_lists_b_field`]. The
/// short-term ops (idc 0/1) resolve `picNumLX` to a *field* of the DPB —
/// the entry whose field-level PicNum (eq. 8-30/8-31, parity-relative to
/// the current field) equals `picNumLX`; the long-term op (idc 2)
/// resolves a field whose field-level LongTermPicNum (eq. 8-32/8-33)
/// equals `long_term_pic_num`. CurrPicNum / MaxPicNum use the field forms
/// (`2*frame_num+1` / `2*MaxFrameNum`, §7.4.3 / clause 8.2.4.1).
///
/// `current_bottom` is the current field's parity. The candidate field's
/// parity is the one stored in each DPB entry (single fields) or chosen
/// by matching `picNumLX` against both parities of a stored frame/pair.
pub fn modify_ref_pic_list_field(
    list: &mut Vec<RefFieldEntry>,
    ops: &[RplmOp],
    dpb: &[DpbEntry],
    num_active: u32,
    current_frame_num: u32,
    max_frame_num: u32,
    current_bottom: bool,
) {
    // Normalise to exactly `num_active` entries (§8.2.4.2 fall-through).
    let target = num_active as usize;
    if list.len() > target {
        list.truncate(target);
    }
    while list.len() < target {
        list.push(no_ref_field());
    }

    if ops.is_empty() {
        return;
    }

    // §7.4.3 — field forms of CurrPicNum / MaxPicNum.
    let curr_pic_num: i32 = (2 * current_frame_num + 1) as i32;
    let max_pic_num: i32 = (2 * max_frame_num) as i32;

    let mut ref_idx_lx: usize = 0;
    let mut pic_num_lx_pred: i32 = curr_pic_num;

    for op in ops {
        match *op {
            RplmOp::Subtract(abs_diff) | RplmOp::Add(abs_diff) => {
                let delta = (abs_diff + 1) as i32;

                // eq. 8-34 / 8-35 — picNumLXNoWrap.
                let pic_num_lx_no_wrap: i32 = if matches!(op, RplmOp::Subtract(_)) {
                    if pic_num_lx_pred - delta < 0 {
                        pic_num_lx_pred - delta + max_pic_num
                    } else {
                        pic_num_lx_pred - delta
                    }
                } else if pic_num_lx_pred + delta >= max_pic_num {
                    pic_num_lx_pred + delta - max_pic_num
                } else {
                    pic_num_lx_pred + delta
                };

                pic_num_lx_pred = pic_num_lx_no_wrap;

                // eq. 8-36 — picNumLX.
                let pic_num_lx: i32 = if pic_num_lx_no_wrap > curr_pic_num {
                    pic_num_lx_no_wrap - max_pic_num
                } else {
                    pic_num_lx_no_wrap
                };

                // Find the (frame, parity) field whose field-PicNum
                // matches. A stored frame/pair can match on either parity;
                // a single field matches only its own.
                let target = find_field_short_term(
                    dpb,
                    pic_num_lx,
                    current_frame_num,
                    max_frame_num,
                    current_bottom,
                );
                splice_field_into_list(list, ref_idx_lx, num_active as usize, target);
                ref_idx_lx += 1;
            }
            RplmOp::LongTerm(long_term_pic_num) => {
                let target = find_field_long_term(dpb, long_term_pic_num as i32, current_bottom);
                splice_field_into_list(list, ref_idx_lx, num_active as usize, target);
                ref_idx_lx += 1;
            }
        }
    }
}

/// Locate the short-term reference *field* whose field-level PicNum
/// equals `pic_num_lx`. Returns the sentinel "no reference picture" when
/// nothing matches (a non-conforming stream).
fn find_field_short_term(
    dpb: &[DpbEntry],
    pic_num_lx: i32,
    current_frame_num: u32,
    max_frame_num: u32,
    current_bottom: bool,
) -> RefFieldEntry {
    for e in dpb.iter().filter(|e| e.is_short_term()) {
        for parity in [FieldParity::Top, FieldParity::Bottom] {
            if e.has_field(parity)
                && e.field_pic_num(parity, current_frame_num, max_frame_num, current_bottom)
                    == pic_num_lx
            {
                return RefFieldEntry {
                    dpb_key: e.dpb_key,
                    parity,
                };
            }
        }
    }
    no_ref_field()
}

/// Locate the long-term reference *field* whose field-level
/// LongTermPicNum equals `long_term_pic_num`.
fn find_field_long_term(
    dpb: &[DpbEntry],
    long_term_pic_num: i32,
    current_bottom: bool,
) -> RefFieldEntry {
    for e in dpb.iter().filter(|e| e.is_long_term()) {
        for parity in [FieldParity::Top, FieldParity::Bottom] {
            if e.has_field(parity)
                && e.field_long_term_pic_num(parity, current_bottom) == long_term_pic_num
            {
                return RefFieldEntry {
                    dpb_key: e.dpb_key,
                    parity,
                };
            }
        }
    }
    no_ref_field()
}

/// §8.2.4.3.1 eq. 8-37 / §8.2.4.3.2 eq. 8-38 for field lists — shift
/// entries right from `ref_idx_lx`, insert the `target` field, then drop
/// any later occurrence of the identical `(dpb_key, parity)` field. A
/// field is uniquely identified by `(dpb_key, parity)`, so the
/// `PicNumF`/`LongTermPicNumF` "!= picNumLX" compaction reduces to "drop
/// the duplicate of the just-inserted field".
fn splice_field_into_list(
    list: &mut Vec<RefFieldEntry>,
    ref_idx_lx: usize,
    num_active: usize,
    target: RefFieldEntry,
) {
    if ref_idx_lx >= num_active {
        return;
    }
    let mut tmp: Vec<RefFieldEntry> = Vec::with_capacity(num_active + 1);
    tmp.extend_from_slice(&list[..ref_idx_lx]);
    tmp.push(target);
    for &e in &list[ref_idx_lx..] {
        if e != target {
            tmp.push(e);
        }
    }
    while tmp.len() < num_active {
        tmp.push(no_ref_field());
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
    dpb: &mut [DpbEntry],
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
#[allow(clippy::too_many_arguments)]
pub fn perform_marking(
    dpb: &mut [DpbEntry],
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
            let mmco5 = apply_mmco(dpb, ops, current_entry, current_frame_num, max_frame_num);
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

    /// A short-term reference *frame* whose two fields carry distinct
    /// POCs (`top_poc` / `bot_poc`); `pic_order_cnt` is `min` per eq. 8-1.
    fn st_pair(frame_num: u32, top_poc: i32, bot_poc: i32, key: u32) -> DpbEntry {
        DpbEntry {
            frame_num,
            top_field_order_cnt: top_poc,
            bottom_field_order_cnt: bot_poc,
            pic_order_cnt: top_poc.min(bot_poc),
            structure: PicStructure::FieldPair,
            marking: RefMarking::ShortTerm,
            long_term_frame_idx: 0,
            dpb_key: key,
        }
    }

    /// A single short-term reference field of the given parity.
    fn st_single_field(frame_num: u32, parity: FieldParity, poc: i32, key: u32) -> DpbEntry {
        let structure = match parity {
            FieldParity::Top => PicStructure::TopField,
            FieldParity::Bottom => PicStructure::BottomField,
        };
        DpbEntry {
            frame_num,
            top_field_order_cnt: poc,
            bottom_field_order_cnt: poc,
            pic_order_cnt: poc,
            structure,
            marking: RefMarking::ShortTerm,
            long_term_frame_idx: 0,
            dpb_key: key,
        }
    }

    /// A long-term reference frame whose two fields are both available.
    fn lt_pair(ltfi: u32, poc: i32, key: u32) -> DpbEntry {
        DpbEntry {
            frame_num: 0,
            top_field_order_cnt: poc,
            bottom_field_order_cnt: poc,
            pic_order_cnt: poc,
            structure: PicStructure::FieldPair,
            marking: RefMarking::LongTerm,
            long_term_frame_idx: ltfi,
            dpb_key: key,
        }
    }

    fn top(key: u32) -> RefFieldEntry {
        RefFieldEntry {
            dpb_key: key,
            parity: FieldParity::Top,
        }
    }

    fn bot(key: u32) -> RefFieldEntry {
        RefFieldEntry {
            dpb_key: key,
            parity: FieldParity::Bottom,
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
        let mut dpb = vec![st_frame(1, 0, 1), st_frame(2, 0, 2), st_frame(3, 0, 3)];
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
        let mut dpb = vec![lt_frame(0, 0, 10), lt_frame(3, 0, 13), lt_frame(7, 0, 17)];
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
        let mut dpb = vec![st_frame(1, 0, 1), st_frame(2, 0, 2), st_frame(3, 0, 3)];
        let mut current = st_frame(4, 0, 4);
        let mmco5 = perform_marking(
            &mut dpb,
            &mut current,
            2, // max_num_ref_frames
            false,
            false,
            false,
            None,
            4,
            16,
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

    // -----------------------------------------------------------------
    // §8.2.4.2.5 — parity-alternation interleave (the shared core).
    // -----------------------------------------------------------------

    // Two complete reference frames; current field is a TOP field. The
    // alternation starts with the current parity (top) and zig-zags:
    //   frame-order = [A(key=1), B(key=2)]
    //   top: A.top, then bottom: A.bot, then top: B.top, then bottom: B.bot
    // i.e. each frame contributes both fields, same-parity-first.
    #[test]
    fn interleave_two_full_frames_top_first() {
        let dpb = vec![st_pair(1, 0, 1, 1), st_pair(2, 2, 3, 2)];
        let order = vec![1u32, 2];
        let out = interleave_fields(&order, &dpb, FieldParity::Top);
        assert_eq!(out, vec![top(1), bot(1), top(2), bot(2)]);
    }

    // Same DPB, current field is a BOTTOM field — alternation starts with
    // bottom parity: B.bot first? No — the cursor for each parity walks
    // `frame_order` independently, so bottom starts at frame A:
    //   bot: A.bot, top: A.top, bot: B.bot, top: B.top
    #[test]
    fn interleave_two_full_frames_bottom_first() {
        let dpb = vec![st_pair(1, 0, 1, 1), st_pair(2, 2, 3, 2)];
        let order = vec![1u32, 2];
        let out = interleave_fields(&order, &dpb, FieldParity::Bottom);
        assert_eq!(out, vec![bot(1), top(1), bot(2), top(2)]);
    }

    // A non-paired field in the middle: frame B carries only its TOP
    // field. Current parity = top.
    //   want=top  → A.top  (A has top)
    //   want=bot  → A.bot  (A has bot; B has no bot, skipped later)
    //   want=top  → B.top  (cursor_top resumes after A)
    //   want=bot  → none left of bottom parity → switch to leftover top →
    //               but all tops consumed → done.
    #[test]
    fn interleave_skips_missing_parity() {
        let dpb = vec![
            st_pair(1, 0, 1, 1),
            st_single_field(2, FieldParity::Top, 2, 2),
        ];
        let order = vec![1u32, 2];
        let out = interleave_fields(&order, &dpb, FieldParity::Top);
        assert_eq!(out, vec![top(1), bot(1), top(2)]);
    }

    // Parity runs out mid-list: only top fields exist. Current = top.
    //   top: F1.top, (want bot → none) → leftover top: F2.top, F3.top
    #[test]
    fn interleave_one_parity_only_appends_leftovers() {
        let dpb = vec![
            st_single_field(1, FieldParity::Top, 0, 1),
            st_single_field(2, FieldParity::Top, 2, 2),
            st_single_field(3, FieldParity::Top, 4, 3),
        ];
        let order = vec![1u32, 2, 3];
        let out = interleave_fields(&order, &dpb, FieldParity::Top);
        assert_eq!(out, vec![top(1), top(2), top(3)]);
    }

    // Current parity = top but only BOTTOM fields exist: the first wanted
    // (top) parity is empty, so the leftover bottom pass takes over from
    // the start, in frame order.
    #[test]
    fn interleave_opposite_parity_only() {
        let dpb = vec![
            st_single_field(1, FieldParity::Bottom, 0, 1),
            st_single_field(2, FieldParity::Bottom, 2, 2),
        ];
        let order = vec![1u32, 2];
        let out = interleave_fields(&order, &dpb, FieldParity::Top);
        assert_eq!(out, vec![bot(1), bot(2)]);
    }

    #[test]
    fn interleave_empty() {
        let dpb: Vec<DpbEntry> = vec![];
        let out = interleave_fields(&[], &dpb, FieldParity::Top);
        assert!(out.is_empty());
    }

    // -----------------------------------------------------------------
    // §8.2.4.2.2 — P/SP field RefPicList0 init.
    // -----------------------------------------------------------------

    // Two short-term reference frames, frame_num {1,2}, current frame_num
    // 5 (no wrap). refFrameList0ShortTerm descending FrameNumWrap = [2,1].
    // Current field = top ⇒ §8.2.4.2.5 alternates:
    //   top: F2.top, bot: F2.bot, top: F1.top, bot: F1.bot
    #[test]
    fn p_field_two_full_frames() {
        let dpb = vec![st_pair(1, 0, 1, 1), st_pair(2, 2, 3, 2)];
        let out = init_ref_pic_list_p_field(&dpb, 5, 16, false);
        assert_eq!(out, vec![top(2), bot(2), top(1), bot(1)]);
    }

    // Mixed short + long term. ST frame_nums {3,4} (desc FrameNumWrap →
    // [4,3]); LT ltfi {0,2} (asc → [k=100(ltfi0), k=102(ltfi2)]). Current
    // field = bottom. Short-term fields precede long-term fields.
    #[test]
    fn p_field_short_then_long() {
        let dpb = vec![
            st_pair(3, 0, 1, 3),
            st_pair(4, 2, 3, 4),
            lt_pair(2, 10, 102),
            lt_pair(0, 8, 100),
        ];
        let out = init_ref_pic_list_p_field(&dpb, 6, 16, true);
        // ST refFrameList0 = [4,3], current=bottom →
        //   bot:F4.bot, top:F4.top, bot:F3.bot, top:F3.top
        // LT refFrameList0 = [100(ltfi0), 102(ltfi2)], current=bottom →
        //   bot:100.bot, top:100.top, bot:102.bot, top:102.top
        assert_eq!(
            out,
            vec![
                bot(4),
                top(4),
                bot(3),
                top(3),
                bot(100),
                top(100),
                bot(102),
                top(102),
            ]
        );
    }

    // FrameNumWrap ordering with wrap: current frame_num 1, ref
    // frame_nums {1, 15} with max_frame_num 16. FrameNumWrap(15) = -1,
    // FrameNumWrap(1) = 1, so descending = [F1(key=1), F15(key=15)].
    #[test]
    fn p_field_framenumwrap_order() {
        let dpb = vec![st_pair(1, 0, 1, 1), st_pair(15, 4, 5, 15)];
        let out = init_ref_pic_list_p_field(&dpb, 1, 16, false);
        assert_eq!(out, vec![top(1), bot(1), top(15), bot(15)]);
    }

    // -----------------------------------------------------------------
    // §8.2.4.2.4 — B field RefPicList0 / RefPicList1 init.
    // -----------------------------------------------------------------

    // Current field POC = 10 (top field). DPB frames with pic_order_cnt
    // {5, 6, 15, 20}. Per §8.2.4.2.4:
    //   refFrameList0ShortTerm: POC≤10 desc [6,5], then POC>10 asc [15,20]
    //                           ⇒ frames [6,5,15,20]
    //   refFrameList1ShortTerm: POC>10 asc [15,20], then POC≤10 desc [6,5]
    //                           ⇒ frames [15,20,6,5]
    // Then §8.2.4.2.5 interleave (current=top, every frame full pair):
    //   list0 = top6,bot6,top5,bot5,top15,bot15,top20,bot20
    //   list1 = top15,bot15,top20,bot20,top6,bot6,top5,bot5
    #[test]
    fn b_field_lists_spec_rules() {
        let dpb = vec![
            st_pair(0, 5, 5, 5),
            st_pair(1, 6, 6, 6),
            st_pair(2, 15, 15, 15),
            st_pair(3, 20, 20, 20),
        ];
        let (l0, l1) = init_ref_pic_lists_b_field(&dpb, 10, false);
        assert_eq!(
            l0,
            vec![
                top(6),
                bot(6),
                top(5),
                bot(5),
                top(15),
                bot(15),
                top(20),
                bot(20),
            ]
        );
        assert_eq!(
            l1,
            vec![
                top(15),
                bot(15),
                top(20),
                bot(20),
                top(6),
                bot(6),
                top(5),
                bot(5),
            ]
        );
    }

    // §8.2.4.2.4 final rule: when List1 == List0 (>1 entry), swap the
    // first two entries of List1. Force identity by giving a single
    // reference frame split into two fields, all on the same side of the
    // current POC so the two frame orders coincide.
    #[test]
    fn b_field_identical_list_swap() {
        // Two full ref frames, both POC < current(10). List0 short order =
        // [POC8, POC6] desc; List1 short order: POC>10 empty, then leftover
        // POC≤10 desc = [POC8, POC6] — identical frame orders ⇒ identical
        // field lists, so List1[0] and List1[1] swap.
        let dpb = vec![st_pair(0, 6, 6, 6), st_pair(1, 8, 8, 8)];
        let (l0, l1) = init_ref_pic_lists_b_field(&dpb, 10, false);
        assert_eq!(l0, vec![top(8), bot(8), top(6), bot(6)]);
        // List1 identical pre-swap → swap [0],[1].
        assert_eq!(l1, vec![bot(8), top(8), top(6), bot(6)]);
    }

    // Long-term fields trail short-term fields in both lists. ST POC 5
    // (<10), LT ltfi 0. Current top field. Here List1's pre-swap content
    // is identical to List0 (the single ST frame's POC 5 ≤ 10 lands in the
    // same leftover slot for both lists), so the §8.2.4.2.4 final rule
    // swaps List1[0] and List1[1].
    #[test]
    fn b_field_short_then_long() {
        let dpb = vec![st_pair(0, 5, 5, 5), lt_pair(0, 99, 100)];
        let (l0, l1) = init_ref_pic_lists_b_field(&dpb, 10, false);
        assert_eq!(l0, vec![top(5), bot(5), top(100), bot(100)]);
        // List1 == List0 pre-swap ⇒ first two entries swapped.
        assert_eq!(l1, vec![bot(5), top(5), top(100), bot(100)]);
    }

    // Non-paired field handling under B: ref frame carries only a bottom
    // field. Current = top field, ref POC 5 < 10.
    //   List0 ST frames = [F(key=7)]; interleave current=top:
    //     want top → F7 has no top → skip; switch to leftover bottom →
    //     F7.bot. Result = [bot7].
    #[test]
    fn b_field_nonpaired_ref() {
        let dpb = vec![st_single_field(0, FieldParity::Bottom, 5, 7)];
        let (l0, _l1) = init_ref_pic_lists_b_field(&dpb, 10, false);
        assert_eq!(l0, vec![bot(7)]);
    }

    // -----------------------------------------------------------------
    // §8.2.4.3 (field) — RPLM on field-coded lists.
    // -----------------------------------------------------------------

    // field_pic_num sanity (eq. 8-30/8-31): frame_num 3, current frame_num
    // 5, max 16 ⇒ FrameNumWrap 3. Current = top field.
    //   same-parity (top): 2*3+1 = 7; opposite (bottom): 2*3 = 6.
    #[test]
    fn field_pic_num_eqs() {
        let e = st_pair(3, 0, 1, 3);
        assert_eq!(e.field_pic_num(FieldParity::Top, 5, 16, false), 7);
        assert_eq!(e.field_pic_num(FieldParity::Bottom, 5, 16, false), 6);
        // Current = bottom field: parities swap which is "same".
        assert_eq!(e.field_pic_num(FieldParity::Top, 5, 16, true), 6);
        assert_eq!(e.field_pic_num(FieldParity::Bottom, 5, 16, true), 7);
    }

    // Subtract op moves a specific field to the front. DPB frames 3,4
    // (full pairs), current frame_num 5 (top field). Initial P-field list
    // = [top4,bot4,top3,bot3] (FrameNumWrap desc 4,3). num_active 4.
    // CurrPicNum = 2*5+1 = 11. Op Subtract(0) → delta 1 → noWrap 10 →
    // picNumLX 10. field_pic_num for frame4 top (FNW 4, same parity) =
    // 2*4+1 = 9; bottom = 8. Frame3 top = 7, bottom = 6. None == 10?
    // Choose a delta that lands on frame4's bottom field (8): picNumLX 8
    // ⇒ Subtract(2) (delta 3, 11-3=8).
    #[test]
    fn field_modify_subtract_reorders() {
        let dpb = vec![st_pair(3, 0, 1, 3), st_pair(4, 2, 3, 4)];
        let mut list = init_ref_pic_list_p_field(&dpb, 5, 16, false);
        assert_eq!(list, vec![top(4), bot(4), top(3), bot(3)]);
        // picNumLX = 11 - 3 = 8 → frame4 bottom field.
        modify_ref_pic_list_field(&mut list, &[RplmOp::Subtract(2)], &dpb, 4, 5, 16, false);
        // bot(4) hoisted to front; its earlier occurrence removed.
        assert_eq!(list, vec![bot(4), top(4), top(3), bot(3)]);
    }

    // Long-term op (idc 2) splices a long-term field. DPB: one ST pair
    // (frame 4) + one LT pair (ltfi 1). Current = top field.
    // field_long_term_pic_num for LT top (same parity) = 2*1+1 = 3,
    // bottom = 2. Op LongTerm(2) → LT bottom field to front.
    #[test]
    fn field_modify_long_term_splices() {
        let dpb = vec![st_pair(4, 2, 3, 4), lt_pair(1, 9, 100)];
        let mut list = init_ref_pic_list_p_field(&dpb, 5, 16, false);
        // ST then LT: [top4,bot4, top100,bot100].
        assert_eq!(list, vec![top(4), bot(4), top(100), bot(100)]);
        modify_ref_pic_list_field(&mut list, &[RplmOp::LongTerm(2)], &dpb, 4, 5, 16, false);
        // LT bottom (LongTermPicNum 2) hoisted to index 0.
        assert_eq!(list, vec![bot(100), top(4), bot(4), top(100)]);
    }

    // No-op list pads / truncates to num_active with the sentinel field.
    #[test]
    fn field_modify_pads_and_truncates() {
        let dpb = vec![st_pair(1, 0, 1, 1)];
        let mut list = vec![top(1)];
        modify_ref_pic_list_field(&mut list, &[], &dpb, 3, 2, 16, false);
        assert_eq!(list.len(), 3);
        assert_eq!(list[0], top(1));
        assert_eq!(list[1], no_ref_field());
        assert_eq!(list[2], no_ref_field());
    }

    // Two sequential ops: picNumLXPred chains across ops (§8.2.4.3.1).
    // DPB frames 2,3,4 full pairs, current frame_num 5 top field.
    // Initial list = [top4,bot4,top3,bot3,top2,bot2] (FNW desc 4,3,2).
    // CurrPicNum = 11. Op1 Subtract(2): pred 11→noWrap 8 → frame4 bottom
    // (field_pic_num 8) to idx0. Op2 Subtract(0): pred 8→noWrap 7 →
    // frame3 top (field_pic_num 7) to idx1.
    #[test]
    fn field_modify_pred_chains_across_ops() {
        let dpb = vec![
            st_pair(2, 4, 5, 2),
            st_pair(3, 6, 7, 3),
            st_pair(4, 8, 9, 4),
        ];
        let mut list = init_ref_pic_list_p_field(&dpb, 5, 16, false);
        assert_eq!(list, vec![top(4), bot(4), top(3), bot(3), top(2), bot(2)]);
        modify_ref_pic_list_field(
            &mut list,
            &[RplmOp::Subtract(2), RplmOp::Subtract(0)],
            &dpb,
            6,
            5,
            16,
            false,
        );
        assert_eq!(list[0], bot(4));
        assert_eq!(list[1], top(3));
    }
}
