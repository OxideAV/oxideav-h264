//! §8.2.1 — Picture Order Count (POC) derivation.
//!
//! Three algorithms indexed by `pic_order_cnt_type` in the active SPS:
//!
//! * §8.2.1.1 — type 0: POC derived from `pic_order_cnt_lsb` with
//!   `MaxPicOrderCntLsb` wrap-around tracked across pictures.
//! * §8.2.1.2 — type 1: POC built up from a cycle of
//!   `offset_for_ref_frame[]` plus `delta_pic_order_cnt[0..=1]`.
//! * §8.2.1.3 — type 2: POC = 2*(FrameNumOffset + frame_num) for
//!   reference pictures, minus 1 for non-reference.
//!
//! Each branch yields `TopFieldOrderCnt`, `BottomFieldOrderCnt` and the
//! derived `PicOrderCnt` for the current picture (= Min of the two for a
//! coded frame; equal to the active field's POC for a coded field —
//! see the `PicOrderCnt( picX )` function, eq. 8-1).

#![allow(dead_code)]

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PocError {
    #[error("pic_order_cnt_type {0} is out of range (must be 0..=2)")]
    InvalidType(u32),
    #[error("log2_max_pic_order_cnt_lsb_minus4 {0} exceeds 12")]
    LsbWidthOutOfRange(u32),
    #[error("log2_max_frame_num_minus4 {0} exceeds 12")]
    FrameNumWidthOutOfRange(u32),
}

/// Subset of SPS syntax elements that feed into §8.2.1 POC derivation.
#[derive(Debug, Clone)]
pub struct PocSps {
    /// §7.4.2.1.1 `pic_order_cnt_type`, range 0..=2.
    pub pic_order_cnt_type: u32,
    /// §7.4.2.1.1 `log2_max_frame_num_minus4`, range 0..=12.
    ///
    /// `MaxFrameNum = 2^(log2_max_frame_num_minus4 + 4)` (eq. 7-10).
    pub log2_max_frame_num_minus4: u32,
    /// §7.4.2.1.1 `log2_max_pic_order_cnt_lsb_minus4`, range 0..=12.
    ///
    /// `MaxPicOrderCntLsb = 2^(log2_max_pic_order_cnt_lsb_minus4 + 4)`
    /// (eq. 7-11). Only meaningful when `pic_order_cnt_type == 0`.
    pub log2_max_pic_order_cnt_lsb_minus4: u32,
    /// §7.4.2.1.1 `delta_pic_order_always_zero_flag`; when true the type-1
    /// `delta_pic_order_cnt[0]` and `delta_pic_order_cnt[1]` are both
    /// inferred to 0.
    pub delta_pic_order_always_zero_flag: bool,
    /// §7.4.2.1.1 `offset_for_non_ref_pic` (type 1).
    pub offset_for_non_ref_pic: i32,
    /// §7.4.2.1.1 `offset_for_top_to_bottom_field` (type 1).
    pub offset_for_top_to_bottom_field: i32,
    /// §7.4.2.1.1 `num_ref_frames_in_pic_order_cnt_cycle` (type 1).
    pub num_ref_frames_in_pic_order_cnt_cycle: u32,
    /// §7.4.2.1.1 `offset_for_ref_frame[0..num_ref_frames_in_pic_order_cnt_cycle]`
    /// (type 1). `ExpectedDeltaPerPicOrderCntCycle` (eq. 7-12) is the
    /// sum of this list.
    pub offset_for_ref_frame: Vec<i32>,
    /// §7.4.2.1.1 `frame_mbs_only_flag`. A coded frame has both fields;
    /// a coded field or MBAFF frame has only the active parity.
    pub frame_mbs_only_flag: bool,
}

/// Subset of slice-header / NAL inputs that feed into §8.2.1.
#[derive(Debug, Clone, Copy)]
pub struct PocSlice {
    /// NAL header: `nal_ref_idc != 0` marks a reference picture.
    pub is_reference: bool,
    /// NAL header: `IdrPicFlag` — true when `nal_unit_type == 5`.
    pub is_idr: bool,
    /// §7.4.3 `frame_num`.
    pub frame_num: u32,
    /// §7.4.3 `field_pic_flag`.
    pub field_pic_flag: bool,
    /// §7.4.3 `bottom_field_flag` (0 when `field_pic_flag == 0`).
    pub bottom_field_flag: bool,
    /// §7.4.3 `pic_order_cnt_lsb` (type 0 only).
    pub pic_order_cnt_lsb: u32,
    /// §7.4.3 `delta_pic_order_cnt_bottom` (type 0, frame only).
    pub delta_pic_order_cnt_bottom: i32,
    /// §7.4.3 `delta_pic_order_cnt[0]` and `[1]` (type 1 only).
    pub delta_pic_order_cnt: [i32; 2],
    /// True if the previous reference picture in decoding order had
    /// `memory_management_control_operation == 5`. §8.2.1 paragraph on
    /// MMCO-5 mandates that `prevPicOrderCntMsb`/`Lsb` — and, for type 1
    /// and 2, `prevFrameNumOffset` — be overridden.
    pub prev_had_mmco5: bool,
    /// §8.2.1.1: when the previous reference picture included MMCO-5 and
    /// was not a bottom field, `prevPicOrderCntLsb` is set equal to the
    /// previous picture's `TopFieldOrderCnt`. Pass 0 if unused. (The
    /// flag says this is i32 to let the caller carry negative values
    /// around; only the lsb's worth of bits enter the computation.)
    pub prev_reference_top_foc_for_mmco5: i32,
}

/// Persistent POC state carried by the caller between pictures.
///
/// Per §8.2.1 the state is logically reset on IDR and on the picture
/// that follows an MMCO-5; the `prev_had_mmco5` flag on `PocSlice`
/// triggers the latter.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PocState {
    /// §8.2.1.1 `prevPicOrderCntMsb` held over from the previous
    /// reference picture (type 0 only).
    pub prev_pic_order_cnt_msb: i32,
    /// §8.2.1.1 `prevPicOrderCntLsb` held over from the previous
    /// reference picture (type 0 only).
    pub prev_pic_order_cnt_lsb: u32,
    /// §8.2.1.2/§8.2.1.3 `prevFrameNum`.
    pub prev_frame_num: u32,
    /// §8.2.1.2/§8.2.1.3 `prevFrameNumOffset` (input `FrameNumOffset`
    /// carried forward). Held as `i64` so that multiple `MaxFrameNum`
    /// accumulations over a long stream do not overflow `i32`.
    pub prev_frame_num_offset: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PocResult {
    pub top_field_order_cnt: i32,
    pub bottom_field_order_cnt: i32,
    /// `PicOrderCnt( picX )` per eq. 8-1 applied to the current picture:
    /// Min of top/bottom for a frame; = top for a top field; = bottom
    /// for a bottom field.
    pub pic_order_cnt: i32,
}

/// §8.2.1 — run the POC derivation for the current picture.
///
/// Updates `state` with the values that subsequent reference-picture
/// POC derivations will read back. For non-reference pictures in type 0
/// the state is intentionally *not* updated (§8.2.1.1 only treats the
/// previous *reference* picture). For types 1 and 2 the state update is
/// driven by `frame_num` and is independent of reference status, but we
/// still honour the MMCO-5 reset hint.
pub fn derive_poc(
    sps: &PocSps,
    slice: &PocSlice,
    state: &mut PocState,
) -> Result<PocResult, PocError> {
    if sps.log2_max_frame_num_minus4 > 12 {
        return Err(PocError::FrameNumWidthOutOfRange(
            sps.log2_max_frame_num_minus4,
        ));
    }
    if sps.log2_max_pic_order_cnt_lsb_minus4 > 12 {
        return Err(PocError::LsbWidthOutOfRange(
            sps.log2_max_pic_order_cnt_lsb_minus4,
        ));
    }
    match sps.pic_order_cnt_type {
        0 => derive_type0(sps, slice, state),
        1 => derive_type1(sps, slice, state),
        2 => derive_type2(sps, slice, state),
        other => Err(PocError::InvalidType(other)),
    }
}

// -------------------------------------------------------------------
// §8.2.1.1 — pic_order_cnt_type == 0.
// -------------------------------------------------------------------

fn derive_type0(
    sps: &PocSps,
    slice: &PocSlice,
    state: &mut PocState,
) -> Result<PocResult, PocError> {
    // eq. 7-11: MaxPicOrderCntLsb = 2^(log2_max_pic_order_cnt_lsb_minus4 + 4).
    let max_pic_order_cnt_lsb: i32 = 1i32 << (sps.log2_max_pic_order_cnt_lsb_minus4 + 4);

    // §8.2.1.1 — derive prevPicOrderCntMsb and prevPicOrderCntLsb.
    //
    // * If the current picture is an IDR, both are 0.
    // * Else if the previous reference picture had MMCO-5:
    //     — if that previous reference picture was not a bottom field,
    //       prevPicOrderCntMsb = 0, prevPicOrderCntLsb = its top FOC.
    //     — otherwise both are 0.
    // * Otherwise carry prev* forward from `state`.
    let (prev_msb, prev_lsb) = if slice.is_idr {
        (0i32, 0u32)
    } else if slice.prev_had_mmco5 {
        // The caller tells us, via `prev_reference_top_foc_for_mmco5`,
        // what the previous (MMCO-5) reference picture's TopFieldOrderCnt
        // was, assuming it was not a bottom field. A bottom field path
        // collapses to (0, 0). We encode "was bottom field" as the
        // caller setting that parameter to 0 — both arms of the spec
        // arrive at the same numbers in that case, so no extra flag is
        // needed.
        let top_foc = slice.prev_reference_top_foc_for_mmco5;
        let lsb = top_foc.rem_euclid(max_pic_order_cnt_lsb) as u32;
        (0i32, lsb)
    } else {
        (state.prev_pic_order_cnt_msb, state.prev_pic_order_cnt_lsb)
    };

    // eq. 8-3 — PicOrderCntMsb wrap-around logic.
    let cur_lsb = slice.pic_order_cnt_lsb as i32;
    let prev_lsb_i = prev_lsb as i32;
    let half = max_pic_order_cnt_lsb / 2;
    let pic_order_cnt_msb = if cur_lsb < prev_lsb_i && (prev_lsb_i - cur_lsb) >= half {
        prev_msb + max_pic_order_cnt_lsb
    } else if cur_lsb > prev_lsb_i && (cur_lsb - prev_lsb_i) > half {
        prev_msb - max_pic_order_cnt_lsb
    } else {
        prev_msb
    };

    // eq. 8-4 — TopFieldOrderCnt (present unless current picture is a
    // bottom field).
    // eq. 8-5 — BottomFieldOrderCnt (present unless current picture is a
    // top field).
    //
    // "Current picture is a bottom field" ⇔ field_pic_flag && bottom_field_flag.
    // "Current picture is a top field"    ⇔ field_pic_flag && !bottom_field_flag.
    // A frame has both fields.
    let is_bottom_field = slice.field_pic_flag && slice.bottom_field_flag;
    let is_top_field = slice.field_pic_flag && !slice.bottom_field_flag;

    let top = if !is_bottom_field {
        pic_order_cnt_msb + cur_lsb // eq. 8-4
    } else {
        0
    };
    let bottom = if !is_top_field {
        if !slice.field_pic_flag {
            top + slice.delta_pic_order_cnt_bottom // eq. 8-5 (frame branch)
        } else {
            pic_order_cnt_msb + cur_lsb // eq. 8-5 (bottom field branch)
        }
    } else {
        0
    };

    // eq. 8-1 — PicOrderCnt() projection onto current picture.
    let pic = if !slice.field_pic_flag {
        top.min(bottom)
    } else if is_bottom_field {
        bottom
    } else {
        top
    };

    // §8.2.1 — carry state forward only for reference pictures and only
    // when the current picture does *not* include MMCO-5 (MMCO-5 is
    // handled by the caller setting `prev_had_mmco5` on the *next*
    // slice). Non-reference pictures do not update the POC anchor.
    if slice.is_reference {
        state.prev_pic_order_cnt_msb = pic_order_cnt_msb;
        // eq. 8-3 consumes `pic_order_cnt_lsb` directly, so store it.
        state.prev_pic_order_cnt_lsb = slice.pic_order_cnt_lsb;
    }
    state.prev_frame_num = slice.frame_num;

    Ok(PocResult {
        top_field_order_cnt: top,
        bottom_field_order_cnt: bottom,
        pic_order_cnt: pic,
    })
}

// -------------------------------------------------------------------
// §8.2.1.2 — pic_order_cnt_type == 1.
// -------------------------------------------------------------------

fn derive_type1(
    sps: &PocSps,
    slice: &PocSlice,
    state: &mut PocState,
) -> Result<PocResult, PocError> {
    // eq. 7-10: MaxFrameNum = 2^(log2_max_frame_num_minus4 + 4).
    let max_frame_num: i64 = 1i64 << (sps.log2_max_frame_num_minus4 + 4);

    // §8.2.1.2 — prevFrameNumOffset.
    //
    // For non-IDR pictures:
    //   — if the previous picture had MMCO-5, prevFrameNumOffset = 0.
    //   — otherwise it equals FrameNumOffset of the previous picture.
    // (For IDRs, prevFrameNumOffset is irrelevant — FrameNumOffset = 0.)
    let prev_frame_num_offset: i64 = if slice.is_idr {
        0
    } else if slice.prev_had_mmco5 {
        0
    } else {
        state.prev_frame_num_offset
    };

    // eq. 8-6 — FrameNumOffset.
    let frame_num_offset: i64 = if slice.is_idr {
        0
    } else if (state.prev_frame_num as i64) > (slice.frame_num as i64) {
        prev_frame_num_offset + max_frame_num
    } else {
        prev_frame_num_offset
    };

    // eq. 8-7 — absFrameNum.
    let mut abs_frame_num: i64 = if sps.num_ref_frames_in_pic_order_cnt_cycle != 0 {
        frame_num_offset + slice.frame_num as i64
    } else {
        0
    };
    if !slice.is_reference && abs_frame_num > 0 {
        abs_frame_num -= 1;
    }

    // eq. 7-12 — ExpectedDeltaPerPicOrderCntCycle = sum(offset_for_ref_frame[]).
    // We recompute it here rather than caching it in `PocSps`; it is a
    // tiny vector and the crate's POC hot path is already dwarfed by
    // the rest of slice decoding.
    let expected_delta_per_cycle: i64 = sps
        .offset_for_ref_frame
        .iter()
        .take(sps.num_ref_frames_in_pic_order_cnt_cycle as usize)
        .map(|&v| v as i64)
        .sum();

    // eq. 8-8 — picOrderCntCycleCnt / frameNumInPicOrderCntCycle.
    // eq. 8-9 — expectedPicOrderCnt.
    let expected_pic_order_cnt: i64 = if abs_frame_num > 0 {
        let cycle_len = sps.num_ref_frames_in_pic_order_cnt_cycle as i64;
        // cycle_len == 0 ⇒ abs_frame_num == 0 by eq. 8-7, so this branch
        // is only reached when cycle_len > 0.
        let pic_order_cnt_cycle_cnt = (abs_frame_num - 1) / cycle_len;
        let frame_num_in_cycle = ((abs_frame_num - 1) % cycle_len) as usize;
        let mut accum: i64 = pic_order_cnt_cycle_cnt * expected_delta_per_cycle;
        for i in 0..=frame_num_in_cycle {
            accum += sps.offset_for_ref_frame[i] as i64;
        }
        accum
    } else {
        0
    };
    let expected_pic_order_cnt: i64 = if !slice.is_reference {
        expected_pic_order_cnt + sps.offset_for_non_ref_pic as i64
    } else {
        expected_pic_order_cnt
    };

    // eq. 8-10 — TopFieldOrderCnt / BottomFieldOrderCnt.
    //
    // Per §7.4.2.1.1 + §7.4.3, when `delta_pic_order_always_zero_flag`
    // is set, both `delta_pic_order_cnt[0]` and `delta_pic_order_cnt[1]`
    // are inferred to be 0. We honour that by zeroing the deltas here
    // so callers do not have to.
    let (d0, d1) = if sps.delta_pic_order_always_zero_flag {
        (0i64, 0i64)
    } else {
        (slice.delta_pic_order_cnt[0] as i64, slice.delta_pic_order_cnt[1] as i64)
    };
    let offset_top_to_bottom = sps.offset_for_top_to_bottom_field as i64;

    let (top_i64, bot_i64) = if !slice.field_pic_flag {
        // Frame: both fields are derived.
        let top = expected_pic_order_cnt + d0;
        let bot = top + offset_top_to_bottom + d1;
        (top, bot)
    } else if !slice.bottom_field_flag {
        // Top field only.
        let top = expected_pic_order_cnt + d0;
        (top, 0)
    } else {
        // Bottom field only.
        let bot = expected_pic_order_cnt + offset_top_to_bottom + d0;
        (0, bot)
    };

    let top = top_i64 as i32;
    let bottom = bot_i64 as i32;

    let pic = if !slice.field_pic_flag {
        top.min(bottom)
    } else if slice.bottom_field_flag {
        bottom
    } else {
        top
    };

    // §8.2.1 state update: FrameNumOffset is derived from *every*
    // picture's frame_num regardless of reference status.
    state.prev_frame_num = slice.frame_num;
    state.prev_frame_num_offset = frame_num_offset;

    Ok(PocResult {
        top_field_order_cnt: top,
        bottom_field_order_cnt: bottom,
        pic_order_cnt: pic,
    })
}

// -------------------------------------------------------------------
// §8.2.1.3 — pic_order_cnt_type == 2.
// -------------------------------------------------------------------

fn derive_type2(
    sps: &PocSps,
    slice: &PocSlice,
    state: &mut PocState,
) -> Result<PocResult, PocError> {
    // eq. 7-10.
    let max_frame_num: i64 = 1i64 << (sps.log2_max_frame_num_minus4 + 4);

    // §8.2.1.3 — prevFrameNumOffset (same rule as type 1).
    let prev_frame_num_offset: i64 = if slice.is_idr {
        0
    } else if slice.prev_had_mmco5 {
        0
    } else {
        state.prev_frame_num_offset
    };

    // eq. 8-11 — FrameNumOffset.
    let frame_num_offset: i64 = if slice.is_idr {
        0
    } else if (state.prev_frame_num as i64) > (slice.frame_num as i64) {
        prev_frame_num_offset + max_frame_num
    } else {
        prev_frame_num_offset
    };

    // eq. 8-12 — tempPicOrderCnt.
    let temp_pic_order_cnt: i64 = if slice.is_idr {
        0
    } else if !slice.is_reference {
        2 * (frame_num_offset + slice.frame_num as i64) - 1
    } else {
        2 * (frame_num_offset + slice.frame_num as i64)
    };

    // eq. 8-13 — TopFieldOrderCnt / BottomFieldOrderCnt.
    let temp = temp_pic_order_cnt as i32;
    let (top, bottom) = if !slice.field_pic_flag {
        (temp, temp)
    } else if slice.bottom_field_flag {
        (0, temp)
    } else {
        (temp, 0)
    };

    let pic = if !slice.field_pic_flag {
        top.min(bottom)
    } else if slice.bottom_field_flag {
        bottom
    } else {
        top
    };

    state.prev_frame_num = slice.frame_num;
    state.prev_frame_num_offset = frame_num_offset;

    Ok(PocResult {
        top_field_order_cnt: top,
        bottom_field_order_cnt: bottom,
        pic_order_cnt: pic,
    })
}

// -------------------------------------------------------------------
// Tests — hand-computed from §8.2.1.* equations, no external decoder.
// -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sps_type0(lsb_bits: u32, frame_bits: u32) -> PocSps {
        PocSps {
            pic_order_cnt_type: 0,
            log2_max_frame_num_minus4: frame_bits,
            log2_max_pic_order_cnt_lsb_minus4: lsb_bits,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            offset_for_ref_frame: Vec::new(),
            frame_mbs_only_flag: true,
        }
    }

    fn base_slice() -> PocSlice {
        PocSlice {
            is_reference: true,
            is_idr: false,
            frame_num: 0,
            field_pic_flag: false,
            bottom_field_flag: false,
            pic_order_cnt_lsb: 0,
            delta_pic_order_cnt_bottom: 0,
            delta_pic_order_cnt: [0, 0],
            prev_had_mmco5: false,
            prev_reference_top_foc_for_mmco5: 0,
        }
    }

    // -------- §8.2.1.1 (type 0) ------------------------------------

    #[test]
    fn type0_idr_is_zero() {
        // IDR: prevPicOrderCntMsb = prevPicOrderCntLsb = 0 and
        // pic_order_cnt_lsb = 0 → TopFieldOrderCnt = 0,
        // BottomFieldOrderCnt = 0 + delta_pic_order_cnt_bottom = 0.
        let sps = sps_type0(4, 4); // MaxPicOrderCntLsb = 256.
        let mut st = PocState::default();
        let mut sl = base_slice();
        sl.is_idr = true;
        sl.pic_order_cnt_lsb = 0;
        let r = derive_poc(&sps, &sl, &mut st).unwrap();
        assert_eq!(r.top_field_order_cnt, 0);
        assert_eq!(r.bottom_field_order_cnt, 0);
        assert_eq!(r.pic_order_cnt, 0);
    }

    #[test]
    fn type0_second_frame_basic() {
        // MaxPicOrderCntLsb = 256. IDR with lsb=0, then non-IDR with
        // lsb=2, delta_bottom=0.
        // eq. 8-3: |2 - 0| = 2 < 128, so PicOrderCntMsb = 0.
        // eq. 8-4: TopFieldOrderCnt = 0 + 2 = 2.
        // eq. 8-5: BottomFieldOrderCnt = 2 + 0 = 2.
        // eq. 8-1: PicOrderCnt = min(2, 2) = 2.
        let sps = sps_type0(4, 4);
        let mut st = PocState::default();
        let mut sl = base_slice();
        sl.is_idr = true;
        let _ = derive_poc(&sps, &sl, &mut st).unwrap();

        let mut sl2 = base_slice();
        sl2.is_idr = false;
        sl2.frame_num = 1;
        sl2.pic_order_cnt_lsb = 2;
        let r = derive_poc(&sps, &sl2, &mut st).unwrap();
        assert_eq!(r.top_field_order_cnt, 2);
        assert_eq!(r.bottom_field_order_cnt, 2);
        assert_eq!(r.pic_order_cnt, 2);
    }

    #[test]
    fn type0_with_delta_bottom() {
        // Frame with non-zero delta_pic_order_cnt_bottom.
        // MaxPicOrderCntLsb=16. Previous ref had lsb=0 msb=0.
        // Current lsb=4, delta_bottom=-2 (bottom field earlier than top).
        // eq. 8-3: |4-0|=4, half=8, no wrap ⇒ Msb=0.
        // eq. 8-4: Top = 0 + 4 = 4.
        // eq. 8-5 (frame): Bot = 4 + (-2) = 2.
        // eq. 8-1 frame: PicOrderCnt = min(4, 2) = 2.
        let sps = sps_type0(0, 0); // lsb bits = 4, frame bits = 4.
        let mut st = PocState::default();
        let mut sl = base_slice();
        sl.is_idr = true;
        let _ = derive_poc(&sps, &sl, &mut st).unwrap();

        let mut sl2 = base_slice();
        sl2.is_idr = false;
        sl2.frame_num = 1;
        sl2.pic_order_cnt_lsb = 4;
        sl2.delta_pic_order_cnt_bottom = -2;
        let r = derive_poc(&sps, &sl2, &mut st).unwrap();
        assert_eq!(r.top_field_order_cnt, 4);
        assert_eq!(r.bottom_field_order_cnt, 2);
        assert_eq!(r.pic_order_cnt, 2);
    }

    #[test]
    fn type0_wraparound_forward() {
        // MaxPicOrderCntLsb = 16, half = 8.
        // Picture A: IDR lsb=0 → Top=0, Bot=0; state.prev_msb=0, prev_lsb=0.
        // Picture B: non-IDR lsb=14 → |14-0|=14 > 8, so branch:
        //            cur > prev AND diff > half ⇒ Msb = 0 - 16 = -16.
        //            Top = -16 + 14 = -2.  (POC = -2.)
        // Picture C: non-IDR lsb=1. state carried from B: prev_msb=-16,
        //            prev_lsb=14. cur(1) < prev(14) AND diff=13 >= 8,
        //            so Msb = -16 + 16 = 0. Top = 0 + 1 = 1.
        let sps = sps_type0(0, 0);
        let mut st = PocState::default();

        let mut a = base_slice();
        a.is_idr = true;
        let _ = derive_poc(&sps, &a, &mut st).unwrap();

        let mut b = base_slice();
        b.is_idr = false;
        b.frame_num = 1;
        b.pic_order_cnt_lsb = 14;
        let r_b = derive_poc(&sps, &b, &mut st).unwrap();
        assert_eq!(r_b.top_field_order_cnt, -2);
        assert_eq!(r_b.bottom_field_order_cnt, -2);
        assert_eq!(r_b.pic_order_cnt, -2);

        let mut c = base_slice();
        c.is_idr = false;
        c.frame_num = 2;
        c.pic_order_cnt_lsb = 1;
        let r_c = derive_poc(&sps, &c, &mut st).unwrap();
        assert_eq!(r_c.top_field_order_cnt, 1);
        assert_eq!(r_c.bottom_field_order_cnt, 1);
        assert_eq!(r_c.pic_order_cnt, 1);
    }

    #[test]
    fn type0_top_field_only() {
        // field_pic_flag=1, bottom_field_flag=0 (top field).
        // eq. 8-4 applies: TopFieldOrderCnt = msb + lsb.
        // eq. 8-5 bottom branch: field_pic_flag=1 AND is_top_field
        //   (so current is a top field): BottomFieldOrderCnt is *not*
        //   derived — we report 0.
        // eq. 8-1: PicOrderCnt = top for a top field.
        let sps = sps_type0(0, 0);
        let mut st = PocState::default();
        let mut sl = base_slice();
        sl.is_idr = true;
        sl.field_pic_flag = true;
        sl.bottom_field_flag = false;
        sl.pic_order_cnt_lsb = 0;
        let r = derive_poc(&sps, &sl, &mut st).unwrap();
        assert_eq!(r.top_field_order_cnt, 0);
        assert_eq!(r.bottom_field_order_cnt, 0);
        assert_eq!(r.pic_order_cnt, 0);
    }

    #[test]
    fn type0_non_reference_does_not_update_anchor() {
        // A non-reference picture in between IDR and next ref must not
        // shift prev_pic_order_cnt_msb/lsb (§8.2.1.1 only speaks of the
        // previous *reference* picture in decoding order).
        let sps = sps_type0(0, 0); // MaxPicOrderCntLsb = 16.
        let mut st = PocState::default();

        let mut idr = base_slice();
        idr.is_idr = true;
        idr.pic_order_cnt_lsb = 0;
        let _ = derive_poc(&sps, &idr, &mut st).unwrap();
        let anchor_after_idr = st.clone();

        let mut nonref = base_slice();
        nonref.is_idr = false;
        nonref.is_reference = false;
        nonref.frame_num = 1;
        nonref.pic_order_cnt_lsb = 6;
        let _ = derive_poc(&sps, &nonref, &mut st).unwrap();
        assert_eq!(
            st.prev_pic_order_cnt_msb, anchor_after_idr.prev_pic_order_cnt_msb,
            "non-ref picture must not advance prev_pic_order_cnt_msb"
        );
        assert_eq!(
            st.prev_pic_order_cnt_lsb, anchor_after_idr.prev_pic_order_cnt_lsb,
            "non-ref picture must not advance prev_pic_order_cnt_lsb"
        );
    }

    // -------- §8.2.1.2 (type 1) ------------------------------------

    fn sps_type1(
        cycle: Vec<i32>,
        off_nonref: i32,
        off_t2b: i32,
        always_zero: bool,
    ) -> PocSps {
        PocSps {
            pic_order_cnt_type: 1,
            log2_max_frame_num_minus4: 0, // MaxFrameNum = 16.
            log2_max_pic_order_cnt_lsb_minus4: 0,
            delta_pic_order_always_zero_flag: always_zero,
            offset_for_non_ref_pic: off_nonref,
            offset_for_top_to_bottom_field: off_t2b,
            num_ref_frames_in_pic_order_cnt_cycle: cycle.len() as u32,
            offset_for_ref_frame: cycle,
            frame_mbs_only_flag: true,
        }
    }

    #[test]
    fn type1_idr_first_frame() {
        // IDR reference frame, frame_num=0, offsets=[2].
        // eq. 8-6: FrameNumOffset = 0.
        // eq. 8-7: absFrameNum = 0 + 0 = 0.
        // eq. 8-9: expectedPicOrderCnt = 0.
        // eq. 8-10 frame: Top = 0 + 0 = 0, Bot = 0 + off_t2b(0) + 0 = 0.
        let sps = sps_type1(vec![2], 0, 0, false);
        let mut st = PocState::default();
        let mut sl = base_slice();
        sl.is_idr = true;
        let r = derive_poc(&sps, &sl, &mut st).unwrap();
        assert_eq!(r.top_field_order_cnt, 0);
        assert_eq!(r.bottom_field_order_cnt, 0);
        assert_eq!(r.pic_order_cnt, 0);
    }

    #[test]
    fn type1_second_frame_single_entry_cycle() {
        // offset_for_ref_frame = [2]. Ref frame with frame_num=1.
        // eq. 8-6: prevFrameNum(0) <= frame_num(1) ⇒ FrameNumOffset = 0.
        // eq. 8-7: absFrameNum = 0 + 1 = 1 (num_ref_frames_in_cycle=1 ≠ 0).
        //          is_reference → no -1.
        // eq. 8-8: cycleCnt = (1-1)/1 = 0, frameNumInCycle = 0.
        // eq. 8-9: expected = 0*2 + offset_for_ref_frame[0] = 2.
        //          + offset_for_non_ref_pic? No, is reference.
        // eq. 8-10 frame: Top = 2 + 0 = 2. Bot = 2 + 0 + 0 = 2.
        let sps = sps_type1(vec![2], 0, 0, false);
        let mut st = PocState::default();
        let mut sl0 = base_slice();
        sl0.is_idr = true;
        let _ = derive_poc(&sps, &sl0, &mut st).unwrap();

        let mut sl1 = base_slice();
        sl1.is_idr = false;
        sl1.frame_num = 1;
        let r = derive_poc(&sps, &sl1, &mut st).unwrap();
        assert_eq!(r.top_field_order_cnt, 2);
        assert_eq!(r.bottom_field_order_cnt, 2);
        assert_eq!(r.pic_order_cnt, 2);
    }

    #[test]
    fn type1_three_entry_cycle_second_element() {
        // offset_for_ref_frame = [4, -2, 6]. Cycle length = 3.
        // Three ref frames, frame_num=0,1,2; check frame_num=1.
        //
        // For frame_num=0 (IDR): absFrameNum=0, expected=0, POC=0.
        // For frame_num=1 (ref): FrameNumOffset=0,
        //   absFrameNum = 0+1 = 1,
        //   cycleCnt = 0, frameNumInCycle = 0,
        //   expected = 0*(4+(-2)+6) + offset_for_ref_frame[0] = 4,
        //   Top = 4 + 0 = 4, Bot = 4 + 0 + 0 = 4.
        let sps = sps_type1(vec![4, -2, 6], 0, 0, false);
        let mut st = PocState::default();
        let mut sl0 = base_slice();
        sl0.is_idr = true;
        let _ = derive_poc(&sps, &sl0, &mut st).unwrap();

        let mut sl1 = base_slice();
        sl1.frame_num = 1;
        let r = derive_poc(&sps, &sl1, &mut st).unwrap();
        assert_eq!(r.top_field_order_cnt, 4);
        assert_eq!(r.bottom_field_order_cnt, 4);
        assert_eq!(r.pic_order_cnt, 4);
    }

    #[test]
    fn type1_three_entry_cycle_full_cycle() {
        // Same cycle as above: [4, -2, 6], ExpectedDeltaPerCycle = 8.
        // Check frame_num = 3 (one full cycle done).
        //   FrameNumOffset = 0, absFrameNum = 3,
        //   cycleCnt = (3-1)/3 = 0, frameNumInCycle = (3-1)%3 = 2,
        //   expected = 0*8 + offsets[0]+offsets[1]+offsets[2] = 8.
        // Check frame_num = 4:
        //   absFrameNum = 4, cycleCnt = 3/3 = 1, frameNumInCycle = 3%3=0,
        //   expected = 1*8 + offsets[0] = 12.
        let sps = sps_type1(vec![4, -2, 6], 0, 0, false);
        let mut st = PocState::default();
        let mut sl0 = base_slice();
        sl0.is_idr = true;
        let _ = derive_poc(&sps, &sl0, &mut st).unwrap();

        let mut sl = base_slice();
        sl.frame_num = 3;
        let r3 = derive_poc(&sps, &sl, &mut st).unwrap();
        assert_eq!(r3.top_field_order_cnt, 8);

        sl.frame_num = 4;
        let r4 = derive_poc(&sps, &sl, &mut st).unwrap();
        assert_eq!(r4.top_field_order_cnt, 12);
    }

    #[test]
    fn type1_always_zero_flag_zeroes_deltas() {
        // When delta_pic_order_always_zero_flag = true, the slice-header
        // deltas are inferred to be 0 even if the caller passed nonzero.
        let sps = sps_type1(vec![2], 0, 0, true);
        let mut st = PocState::default();
        let mut sl = base_slice();
        sl.is_idr = true;
        sl.delta_pic_order_cnt = [7, -9]; // should be ignored.
        let r = derive_poc(&sps, &sl, &mut st).unwrap();
        assert_eq!(r.top_field_order_cnt, 0);
        assert_eq!(r.bottom_field_order_cnt, 0);
    }

    #[test]
    fn type1_non_reference_offset() {
        // offset_for_non_ref_pic = 3. offsets = [5]. Non-ref frame_num=1.
        // eq. 8-7: absFrameNum = 0+1=1, and !is_reference → absFrameNum -= 1 = 0.
        // eq. 8-9: absFrameNum=0 ⇒ expected=0; then + offset_for_non_ref_pic = 3.
        // Top = 3 + delta[0](0) = 3.
        let sps = sps_type1(vec![5], 3, 0, false);
        let mut st = PocState::default();
        let mut sl0 = base_slice();
        sl0.is_idr = true;
        let _ = derive_poc(&sps, &sl0, &mut st).unwrap();

        let mut sl = base_slice();
        sl.frame_num = 1;
        sl.is_reference = false;
        let r = derive_poc(&sps, &sl, &mut st).unwrap();
        assert_eq!(r.top_field_order_cnt, 3);
    }

    #[test]
    fn type1_bottom_field() {
        // Bottom field of an IDR: field_pic_flag=1, bottom_field_flag=1.
        // offset_for_top_to_bottom_field = 1.
        // eq. 8-10 bottom branch:
        //   BottomFieldOrderCnt = expected + off_t2b + delta[0]
        //                       = 0 + 1 + 0 = 1.
        let sps = sps_type1(vec![2], 0, 1, false);
        let mut st = PocState::default();
        let mut sl = base_slice();
        sl.is_idr = true;
        sl.field_pic_flag = true;
        sl.bottom_field_flag = true;
        let r = derive_poc(&sps, &sl, &mut st).unwrap();
        assert_eq!(r.top_field_order_cnt, 0);
        assert_eq!(r.bottom_field_order_cnt, 1);
        assert_eq!(r.pic_order_cnt, 1);
    }

    #[test]
    fn type1_empty_cycle_uses_abs_frame_num_zero() {
        // When num_ref_frames_in_pic_order_cnt_cycle == 0, eq. 8-7 sets
        // absFrameNum = 0 regardless of frame_num. Hence
        // expectedPicOrderCnt = 0 for every reference picture.
        let sps = sps_type1(Vec::new(), 0, 0, false);
        let mut st = PocState::default();
        let mut sl0 = base_slice();
        sl0.is_idr = true;
        let _ = derive_poc(&sps, &sl0, &mut st).unwrap();

        let mut sl = base_slice();
        sl.frame_num = 5;
        let r = derive_poc(&sps, &sl, &mut st).unwrap();
        assert_eq!(r.top_field_order_cnt, 0);
        assert_eq!(r.bottom_field_order_cnt, 0);
    }

    // -------- §8.2.1.3 (type 2) ------------------------------------

    fn sps_type2() -> PocSps {
        PocSps {
            pic_order_cnt_type: 2,
            log2_max_frame_num_minus4: 0, // MaxFrameNum = 16.
            log2_max_pic_order_cnt_lsb_minus4: 0,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            offset_for_ref_frame: Vec::new(),
            frame_mbs_only_flag: true,
        }
    }

    #[test]
    fn type2_idr_zero() {
        // IDR ⇒ tempPicOrderCnt = 0.
        let sps = sps_type2();
        let mut st = PocState::default();
        let mut sl = base_slice();
        sl.is_idr = true;
        let r = derive_poc(&sps, &sl, &mut st).unwrap();
        assert_eq!(r.top_field_order_cnt, 0);
        assert_eq!(r.bottom_field_order_cnt, 0);
    }

    #[test]
    fn type2_reference_frame() {
        // Reference picture, frame_num=3: tempPicOrderCnt = 2*(0+3) = 6.
        // eq. 8-13 frame branch: Top = Bot = 6. PicOrderCnt = 6.
        let sps = sps_type2();
        let mut st = PocState::default();
        let mut sl0 = base_slice();
        sl0.is_idr = true;
        let _ = derive_poc(&sps, &sl0, &mut st).unwrap();

        let mut sl = base_slice();
        sl.frame_num = 3;
        let r = derive_poc(&sps, &sl, &mut st).unwrap();
        assert_eq!(r.top_field_order_cnt, 6);
        assert_eq!(r.bottom_field_order_cnt, 6);
        assert_eq!(r.pic_order_cnt, 6);
    }

    #[test]
    fn type2_non_reference_frame() {
        // Non-reference picture, frame_num=3:
        // tempPicOrderCnt = 2*(0+3) - 1 = 5. Top = Bot = 5.
        let sps = sps_type2();
        let mut st = PocState::default();
        let mut sl0 = base_slice();
        sl0.is_idr = true;
        let _ = derive_poc(&sps, &sl0, &mut st).unwrap();

        let mut sl = base_slice();
        sl.frame_num = 3;
        sl.is_reference = false;
        let r = derive_poc(&sps, &sl, &mut st).unwrap();
        assert_eq!(r.top_field_order_cnt, 5);
        assert_eq!(r.bottom_field_order_cnt, 5);
        assert_eq!(r.pic_order_cnt, 5);
    }

    #[test]
    fn type2_frame_num_wraparound() {
        // MaxFrameNum = 16. prevFrameNum = 15, current frame_num = 0,
        // non-IDR reference picture.
        // eq. 8-11: 15 > 0 ⇒ FrameNumOffset = 0 + 16 = 16.
        // eq. 8-12: temp = 2*(16 + 0) = 32.
        let sps = sps_type2();
        let mut st = PocState {
            prev_frame_num: 15,
            prev_frame_num_offset: 0,
            prev_pic_order_cnt_msb: 0,
            prev_pic_order_cnt_lsb: 0,
        };
        let mut sl = base_slice();
        sl.frame_num = 0;
        let r = derive_poc(&sps, &sl, &mut st).unwrap();
        assert_eq!(r.top_field_order_cnt, 32);
        assert_eq!(r.bottom_field_order_cnt, 32);
        assert_eq!(st.prev_frame_num_offset, 16);
    }

    #[test]
    fn type2_bottom_field() {
        // Bottom field: Top=0, Bottom=temp, PicOrderCnt=bottom.
        let sps = sps_type2();
        let mut st = PocState::default();
        let mut sl0 = base_slice();
        sl0.is_idr = true;
        let _ = derive_poc(&sps, &sl0, &mut st).unwrap();

        let mut sl = base_slice();
        sl.frame_num = 2;
        sl.field_pic_flag = true;
        sl.bottom_field_flag = true;
        let r = derive_poc(&sps, &sl, &mut st).unwrap();
        // Reference by default; temp = 2*(0+2) = 4.
        assert_eq!(r.top_field_order_cnt, 0);
        assert_eq!(r.bottom_field_order_cnt, 4);
        assert_eq!(r.pic_order_cnt, 4);
    }

    // -------- error paths ------------------------------------------

    #[test]
    fn invalid_pic_order_cnt_type() {
        let mut sps = sps_type0(0, 0);
        sps.pic_order_cnt_type = 3;
        let mut st = PocState::default();
        let mut sl = base_slice();
        sl.is_idr = true;
        let r = derive_poc(&sps, &sl, &mut st);
        assert_eq!(r, Err(PocError::InvalidType(3)));
    }

    #[test]
    fn invalid_lsb_width() {
        let mut sps = sps_type0(0, 0);
        sps.log2_max_pic_order_cnt_lsb_minus4 = 13;
        let mut st = PocState::default();
        let mut sl = base_slice();
        sl.is_idr = true;
        let r = derive_poc(&sps, &sl, &mut st);
        assert_eq!(r, Err(PocError::LsbWidthOutOfRange(13)));
    }

    #[test]
    fn invalid_frame_num_width() {
        let mut sps = sps_type0(0, 0);
        sps.log2_max_frame_num_minus4 = 13;
        let mut st = PocState::default();
        let mut sl = base_slice();
        sl.is_idr = true;
        let r = derive_poc(&sps, &sl, &mut st);
        assert_eq!(r, Err(PocError::FrameNumWidthOutOfRange(13)));
    }
}
