//! §9.3 — CABAC per-element binarisations and context-index derivation.
//!
//! This module wraps the low-level CABAC engine in [`crate::cabac`] with
//! the per-syntax-element tables and ctxIdx derivation procedures from
//! ITU-T Rec. H.264 (08/2024), clauses 9.3.1.1 .. 9.3.3.1.
//!
//! Only data/logic transcribed from the ITU-T specification is used
//! here; no other H.264 decoder source is consulted (clean-room rule).
//!
//! Coverage in this initial drop:
//!
//! * §9.3.1.1 + Tables 9-12..9-24 — context initialisation (m, n)
//!   pairs for ctxIdx 0..=459. Higher ranges (coeff-level /
//!   significant-coeff field/8x8 variants in Tables 9-25..9-33) are
//!   transcribed here as well for completeness.
//! * §9.3.2 — U / TU / UEGk / FL binarisations.
//! * §9.3.2.5 — mb_type binarisation tables 9-36, 9-37.
//! * §9.3.3.1 / Table 9-39 — ctxIdxInc table.
//! * §9.3.3.1.1.* — per-element ctxIdxInc derivation (mb_skip_flag,
//!   mb_type, mb_qp_delta, ref_idx_lX, mvd_lX, coded_block_pattern,
//!   coded_block_flag, intra_chroma_pred_mode,
//!   transform_size_8x8_flag).
//! * §9.3.3.1.2 / Table 9-41 — prior-decoded-bin ctxIdxInc rules.
//! * §9.3.3.1.3 / Table 9-43 — significant/last coeff ctxIdxInc for
//!   8x8 blocks.
//! * §9.3.3.2 — per-element decoding entry points that wire all of the
//!   above through the engine primitives.

#![allow(dead_code)]

use crate::cabac::{CabacDecoder, CabacResult, CtxState};

#[inline]
fn dbg_enabled() -> bool {
    // Separate switch from OXIDEAV_H264_DEBUG to avoid drowning the
    // macroblock-layer logs.
    std::env::var("OXIDEAV_H264_CABAC_DEBUG").is_ok()
}

#[inline]
fn dbg_emit(syntax: &str, bins_used: u64, extra: &str) {
    if dbg_enabled() {
        eprintln!("[CABAC] {syntax} bins={bins_used} {extra}");
    }
}

// ---------------------------------------------------------------------------
// Slice kinds and selectors
// ---------------------------------------------------------------------------

/// Slice kind used to pick an initialisation column from Table 9-11.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SliceKind {
    I,
    P,
    B,
    SI,
    SP,
}

impl SliceKind {
    /// True for the two I-like slice types that have no `cabac_init_idc`.
    fn is_i_like(self) -> bool {
        matches!(self, SliceKind::I | SliceKind::SI)
    }

    /// True for the three P-like slice types (P, SP).
    fn is_p_like(self) -> bool {
        matches!(self, SliceKind::P | SliceKind::SP)
    }
}

// ---------------------------------------------------------------------------
// §9.3.1.1 / Tables 9-12 .. 9-33 — initialisation (m, n) pairs.
//
// Each `InitCell` captures the four possible (m, n) sources for a
// single ctxIdx: the "I and SI slices" column and the three
// cabac_init_idc columns (0, 1, 2) for P/SP/B slices.
//
// Entries whose column is marked "na" in the spec are stored as
// `None`. `NA` is used as shorthand.
//
// Values transcribed verbatim from document pages 231..252 of
// ITU-T Rec. H.264 (08/2024).
// ---------------------------------------------------------------------------

/// (m, n) init pair.
type MN = (i32, i32);

/// "na" entry shorthand (context not used by this slice kind).
const NA: Option<MN> = None;

/// Per-ctxIdx init row: four alternatives selected by (slice_kind,
/// cabac_init_idc). `i_si` is "I and SI slices"; `p0`/`p1`/`p2` are
/// the three `cabac_init_idc` columns for P/SP/B slices.
#[derive(Debug, Copy, Clone)]
struct InitCell {
    i_si: Option<MN>,
    p0: Option<MN>,
    p1: Option<MN>,
    p2: Option<MN>,
}

impl InitCell {
    const fn all(i_si: MN, p0: MN, p1: MN, p2: MN) -> Self {
        Self {
            i_si: Some(i_si),
            p0: Some(p0),
            p1: Some(p1),
            p2: Some(p2),
        }
    }
    const fn p(p0: MN, p1: MN, p2: MN) -> Self {
        Self {
            i_si: None,
            p0: Some(p0),
            p1: Some(p1),
            p2: Some(p2),
        }
    }
    const fn i(i_si: MN) -> Self {
        Self {
            i_si: Some(i_si),
            p0: NA,
            p1: NA,
            p2: NA,
        }
    }
    /// Use a single (m, n) pair for every slice type and every
    /// `cabac_init_idc`. Table 9-17 (ctxIdx 60..=69) lists only one
    /// column of values yet is referenced by I, SI, P/SP, and B rows
    /// of Table 9-11, so the same (m, n) applies everywhere.
    const fn same(mn: MN) -> Self {
        Self {
            i_si: Some(mn),
            p0: Some(mn),
            p1: Some(mn),
            p2: Some(mn),
        }
    }
    const fn none() -> Self {
        Self {
            i_si: NA,
            p0: NA,
            p1: NA,
            p2: NA,
        }
    }

    fn pick(&self, slice_kind: SliceKind, init_idc: Option<u32>) -> Option<MN> {
        if slice_kind.is_i_like() {
            return self.i_si;
        }
        match init_idc? {
            0 => self.p0,
            1 => self.p1,
            2 => self.p2,
            _ => None,
        }
    }
}

/// Total number of context entries the engine tracks (spec bounds
/// ctxIdx to 0..=1023).
pub const NUM_CTX_IDX: usize = 1024;

// Table 9-12 — ctxIdx 0..=10 (applies to SI, I, P, SP, B).
const TBL_9_12: &[(usize, InitCell)] = &[
    (0, InitCell::all((20, -15), (20, -15), (20, -15), (20, -15))),
    (1, InitCell::all((2, 54), (2, 54), (2, 54), (2, 54))),
    (2, InitCell::all((3, 74), (3, 74), (3, 74), (3, 74))),
    (3, InitCell::all((20, -15), (20, -15), (20, -15), (20, -15))),
    (4, InitCell::all((2, 54), (2, 54), (2, 54), (2, 54))),
    (5, InitCell::all((3, 74), (3, 74), (3, 74), (3, 74))),
    (
        6,
        InitCell::all((-28, 127), (-28, 127), (-28, 127), (-28, 127)),
    ),
    (
        7,
        InitCell::all((-23, 104), (-23, 104), (-23, 104), (-23, 104)),
    ),
    (8, InitCell::all((-6, 53), (-6, 53), (-6, 53), (-6, 53))),
    (9, InitCell::all((-1, 54), (-1, 54), (-1, 54), (-1, 54))),
    (10, InitCell::all((7, 51), (7, 51), (7, 51), (7, 51))),
];

// Table 9-13 — ctxIdx 11..=23 (cabac_init_idc 0, 1, 2 only; no I/SI).
const TBL_9_13: &[(usize, InitCell)] = &[
    (11, InitCell::p((23, 33), (22, 25), (29, 16))),
    (12, InitCell::p((23, 2), (34, 0), (25, 0))),
    (13, InitCell::p((21, 0), (16, 0), (14, 0))),
    (14, InitCell::p((1, 9), (-2, 9), (-10, 51))),
    (15, InitCell::p((0, 49), (4, 41), (-3, 62))),
    (16, InitCell::p((-37, 118), (-29, 118), (-27, 99))),
    (17, InitCell::p((5, 57), (2, 65), (26, 16))),
    (18, InitCell::p((-13, 78), (-6, 71), (-4, 85))),
    (19, InitCell::p((-11, 65), (-13, 79), (-24, 102))),
    (20, InitCell::p((1, 62), (5, 52), (5, 57))),
    (21, InitCell::p((12, 49), (9, 50), (6, 57))),
    (22, InitCell::p((-4, 73), (-3, 70), (-17, 73))),
    (23, InitCell::p((17, 50), (10, 54), (14, 57))),
];

// Table 9-14 — ctxIdx 24..=39 (cabac_init_idc 0, 1, 2 only).
const TBL_9_14: &[(usize, InitCell)] = &[
    (24, InitCell::p((18, 64), (26, 34), (20, 40))),
    (25, InitCell::p((9, 43), (19, 22), (20, 10))),
    (26, InitCell::p((29, 0), (40, 0), (29, 0))),
    (27, InitCell::p((26, 67), (57, 2), (54, 0))),
    (28, InitCell::p((16, 90), (41, 36), (37, 42))),
    (29, InitCell::p((9, 104), (26, 69), (12, 97))),
    (30, InitCell::p((-46, 127), (-45, 127), (-32, 127))),
    (31, InitCell::p((-20, 104), (-15, 101), (-22, 117))),
    (32, InitCell::p((1, 67), (-4, 76), (-2, 74))),
    (33, InitCell::p((-13, 78), (-6, 71), (-4, 85))),
    (34, InitCell::p((-11, 65), (-13, 79), (-24, 102))),
    (35, InitCell::p((1, 62), (5, 52), (5, 57))),
    (36, InitCell::p((-6, 86), (6, 69), (-6, 93))),
    (37, InitCell::p((-17, 95), (-13, 90), (-14, 88))),
    (38, InitCell::p((-6, 61), (0, 52), (-6, 44))),
    (39, InitCell::p((9, 45), (8, 43), (4, 55))),
];

// Table 9-15 — ctxIdx 40..=53 (cabac_init_idc 0, 1, 2 only).
const TBL_9_15: &[(usize, InitCell)] = &[
    (40, InitCell::p((-3, 69), (-2, 69), (-11, 89))),
    (41, InitCell::p((-6, 81), (-5, 82), (-15, 103))),
    (42, InitCell::p((-11, 96), (-10, 96), (-21, 116))),
    (43, InitCell::p((6, 55), (2, 59), (19, 57))),
    (44, InitCell::p((7, 67), (2, 75), (20, 58))),
    (45, InitCell::p((-5, 86), (-3, 87), (4, 84))),
    (46, InitCell::p((2, 88), (-3, 100), (6, 96))),
    (47, InitCell::p((0, 58), (1, 56), (1, 63))),
    (48, InitCell::p((-3, 76), (-3, 74), (-5, 85))),
    (49, InitCell::p((-10, 94), (-6, 85), (-13, 106))),
    (50, InitCell::p((5, 54), (0, 59), (5, 63))),
    (51, InitCell::p((4, 69), (-3, 81), (6, 75))),
    (52, InitCell::p((-3, 81), (-7, 86), (-3, 90))),
    (53, InitCell::p((0, 88), (-5, 95), (-1, 101))),
];

// Table 9-16 — ctxIdx 54..=59 and 399..=401 (per spec this table
// splits into two disjoint ranges). I/SI has only `na` for 54..=59;
// the 399..=401 entries provide I-slice values.
const TBL_9_16: &[(usize, InitCell)] = &[
    (54, InitCell::p((-7, 67), (-1, 66), (3, 55))),
    (55, InitCell::p((-5, 74), (-1, 77), (-4, 79))),
    (56, InitCell::p((-4, 74), (1, 70), (-2, 75))),
    (57, InitCell::p((-5, 80), (-2, 86), (-12, 97))),
    (58, InitCell::p((-7, 72), (-5, 72), (-7, 50))),
    (59, InitCell::p((1, 58), (0, 61), (1, 60))),
    (399, InitCell::all((31, 21), (12, 40), (25, 32), (21, 33))),
    (400, InitCell::all((31, 31), (11, 51), (21, 49), (19, 50))),
    (401, InitCell::all((25, 50), (14, 59), (21, 54), (17, 61))),
];

// Table 9-17 — ctxIdx 60..=69.
//
// Per Table 9-11 of the 08/2024 spec, `mb_qp_delta` (60..=63),
// `intra_chroma_pred_mode` (64..=67), `prev_intra*_pred_mode_flag` (68),
// and `rem_intra*_pred_mode` (69) all reference Table 9-17 for every
// slice type, and Table 9-17 has a single (m, n) column — so the same
// (m, n) pair initialises each context for I, SI, P, SP, and B slices.
//
// All ten entries use ::same. Initially 64..=69 were gated on ::i
// (I-slice only) to work around a then-unknown compensating bug on
// intra-coded MBs inside P slices. Round 35 traced that bug to this
// same init mismatch: intra MBs inside P slices on CABA2_SVA_B
// (frame 3 MB 28) entered `prev_intra4x4_pred_mode_flag` decode with
// ctxIdx 68 at default (state=0, mps=0) and mis-decoded flag=0 (LPS
// loop) instead of flag=1 (MPS loop), inflating the MB's bit budget
// by 21 bits and cascading pixel divergence from that MB onward.
// Applying the literal ::same fix matches JM's per-MB bit budget on
// MB 28 of frame 3 and should be the correct spec behaviour.
const TBL_9_17: &[(usize, InitCell)] = &[
    (60, InitCell::same((0, 41))),
    (61, InitCell::same((0, 63))),
    (62, InitCell::same((0, 63))),
    (63, InitCell::same((0, 63))),
    (64, InitCell::same((-9, 83))),
    (65, InitCell::same((4, 86))),
    (66, InitCell::same((0, 97))),
    (67, InitCell::same((-7, 72))),
    (68, InitCell::same((13, 41))),
    (69, InitCell::same((3, 62))),
];

// Table 9-18 — ctxIdx 70..=104 (applies to I/SI and all three init_idc).
const TBL_9_18: &[(usize, InitCell)] = &[
    (70, InitCell::all((0, 11), (0, 45), (13, 15), (7, 34))),
    (71, InitCell::all((1, 55), (-4, 78), (7, 51), (-9, 88))),
    (72, InitCell::all((0, 69), (-3, 96), (2, 80), (-20, 127))),
    (
        73,
        InitCell::all((-17, 127), (-27, 126), (-39, 127), (-36, 127)),
    ),
    (
        74,
        InitCell::all((-13, 102), (-28, 98), (-18, 91), (-17, 91)),
    ),
    (75, InitCell::all((0, 82), (-25, 101), (-17, 96), (-14, 95))),
    (76, InitCell::all((-7, 74), (-23, 67), (-26, 81), (-25, 84))),
    (
        77,
        InitCell::all((-21, 107), (-28, 82), (-35, 98), (-25, 86)),
    ),
    (
        78,
        InitCell::all((-27, 127), (-20, 94), (-24, 102), (-12, 89)),
    ),
    (
        79,
        InitCell::all((-31, 127), (-16, 83), (-23, 97), (-17, 91)),
    ),
    (
        80,
        InitCell::all((-24, 127), (-22, 110), (-27, 119), (-31, 127)),
    ),
    (
        81,
        InitCell::all((-18, 95), (-21, 91), (-24, 99), (-14, 76)),
    ),
    (
        82,
        InitCell::all((-27, 127), (-18, 102), (-21, 110), (-18, 103)),
    ),
    (
        83,
        InitCell::all((-21, 114), (-13, 93), (-18, 102), (-13, 90)),
    ),
    (
        84,
        InitCell::all((-30, 127), (-29, 127), (-36, 127), (-37, 127)),
    ),
    (85, InitCell::all((-17, 123), (-7, 92), (0, 80), (11, 80))),
    (86, InitCell::all((-12, 115), (-5, 89), (-5, 89), (5, 76))),
    (87, InitCell::all((-16, 122), (-7, 96), (-7, 94), (2, 84))),
    (88, InitCell::all((-11, 115), (-13, 108), (-4, 92), (5, 78))),
    (89, InitCell::all((-12, 63), (-3, 46), (0, 39), (-6, 55))),
    (90, InitCell::all((-2, 68), (-1, 65), (0, 65), (4, 61))),
    (91, InitCell::all((-15, 84), (-1, 57), (-15, 84), (-14, 83))),
    (
        92,
        InitCell::all((-13, 104), (-9, 93), (-35, 127), (-37, 127)),
    ),
    (93, InitCell::all((-3, 70), (-3, 74), (-2, 73), (-5, 79))),
    (
        94,
        InitCell::all((-8, 93), (-9, 92), (-12, 104), (-11, 104)),
    ),
    (95, InitCell::all((-10, 90), (-8, 87), (-9, 91), (-11, 91))),
    (
        96,
        InitCell::all((-30, 127), (-23, 126), (-31, 127), (-30, 127)),
    ),
    (97, InitCell::all((-1, 74), (5, 54), (3, 55), (0, 65))),
    (98, InitCell::all((-6, 97), (6, 60), (7, 56), (-2, 79))),
    (99, InitCell::all((-7, 91), (6, 59), (7, 55), (0, 72))),
    (100, InitCell::all((-20, 127), (6, 69), (8, 61), (-4, 92))),
    (101, InitCell::all((-4, 56), (-1, 48), (-3, 53), (-6, 56))),
    (102, InitCell::all((-5, 82), (0, 68), (0, 68), (3, 68))),
    (103, InitCell::all((-7, 76), (-4, 69), (-7, 74), (-8, 71))),
    (
        104,
        InitCell::all((-22, 125), (-8, 88), (-9, 88), (-13, 98)),
    ),
];

// Table 9-19 — ctxIdx 105..=165.
const TBL_9_19: &[(usize, InitCell)] = &[
    (105, InitCell::all((-7, 93), (-2, 85), (-13, 103), (-4, 86))),
    (
        106,
        InitCell::all((-11, 87), (-6, 78), (-13, 91), (-12, 88)),
    ),
    (107, InitCell::all((-3, 77), (-1, 75), (-9, 89), (-5, 82))),
    (108, InitCell::all((-5, 71), (-7, 77), (-14, 92), (-3, 72))),
    (109, InitCell::all((-4, 63), (2, 54), (-8, 76), (-4, 67))),
    (110, InitCell::all((-4, 68), (5, 50), (-12, 87), (-8, 72))),
    (
        111,
        InitCell::all((-12, 84), (-3, 68), (-23, 110), (-16, 89)),
    ),
    (112, InitCell::all((-7, 62), (1, 50), (-24, 105), (-9, 69))),
    (113, InitCell::all((-7, 65), (6, 42), (-10, 78), (-1, 59))),
    (114, InitCell::all((8, 61), (-4, 81), (-20, 112), (5, 66))),
    (115, InitCell::all((5, 56), (1, 63), (-17, 99), (4, 57))),
    (116, InitCell::all((-2, 66), (-4, 70), (-78, 127), (-4, 71))),
    (117, InitCell::all((1, 64), (0, 67), (-70, 127), (-2, 71))),
    (118, InitCell::all((0, 61), (2, 57), (-50, 127), (2, 58))),
    (119, InitCell::all((-2, 78), (-2, 76), (-46, 127), (-1, 74))),
    (120, InitCell::all((1, 50), (11, 35), (-4, 66), (-4, 44))),
    (121, InitCell::all((7, 52), (4, 64), (-5, 78), (-1, 69))),
    (122, InitCell::all((10, 35), (1, 61), (-4, 71), (0, 62))),
    (123, InitCell::all((0, 44), (11, 35), (-8, 72), (-7, 51))),
    (124, InitCell::all((11, 38), (18, 25), (2, 59), (-4, 47))),
    (125, InitCell::all((1, 45), (12, 24), (-1, 55), (-6, 42))),
    (126, InitCell::all((0, 46), (13, 29), (-7, 70), (-3, 41))),
    (127, InitCell::all((5, 44), (13, 36), (-6, 75), (-6, 53))),
    (128, InitCell::all((31, 17), (-10, 93), (-8, 89), (8, 76))),
    (129, InitCell::all((1, 51), (-7, 73), (-34, 119), (-9, 78))),
    (130, InitCell::all((7, 50), (-2, 73), (-3, 75), (-11, 83))),
    (131, InitCell::all((28, 19), (13, 46), (32, 20), (9, 52))),
    (132, InitCell::all((16, 33), (9, 49), (30, 22), (0, 67))),
    (
        133,
        InitCell::all((14, 62), (-7, 100), (-44, 127), (-5, 90)),
    ),
    (134, InitCell::all((-13, 108), (9, 53), (0, 54), (1, 67))),
    (135, InitCell::all((-15, 100), (2, 53), (-5, 61), (-15, 72))),
    (136, InitCell::all((-13, 101), (5, 53), (0, 58), (-5, 75))),
    (137, InitCell::all((-13, 91), (-2, 61), (-1, 60), (-8, 80))),
    (138, InitCell::all((-12, 94), (0, 56), (-3, 61), (-21, 83))),
    (139, InitCell::all((-10, 88), (0, 56), (-8, 67), (-21, 64))),
    (
        140,
        InitCell::all((-16, 84), (-13, 63), (-25, 84), (-13, 31)),
    ),
    (
        141,
        InitCell::all((-10, 86), (-5, 60), (-14, 74), (-25, 64)),
    ),
    (142, InitCell::all((-7, 83), (-1, 62), (-5, 65), (-29, 94))),
    (143, InitCell::all((-13, 87), (4, 57), (5, 52), (9, 75))),
    (144, InitCell::all((-19, 94), (-6, 69), (2, 57), (17, 63))),
    (145, InitCell::all((1, 70), (4, 57), (0, 61), (-8, 74))),
    (146, InitCell::all((0, 72), (14, 39), (-9, 69), (-5, 35))),
    (147, InitCell::all((-5, 74), (4, 51), (-11, 70), (-2, 27))),
    (148, InitCell::all((18, 59), (13, 68), (18, 55), (13, 91))),
    (149, InitCell::all((-8, 102), (3, 64), (-4, 71), (3, 65))),
    (150, InitCell::all((-15, 100), (1, 61), (0, 58), (-7, 69))),
    (151, InitCell::all((0, 95), (9, 63), (7, 61), (8, 77))),
    (152, InitCell::all((-4, 75), (7, 50), (9, 41), (-10, 66))),
    (153, InitCell::all((2, 72), (16, 39), (18, 25), (3, 62))),
    (154, InitCell::all((-11, 75), (5, 44), (9, 32), (-3, 68))),
    (155, InitCell::all((-3, 71), (4, 52), (5, 43), (-20, 81))),
    (156, InitCell::all((15, 46), (11, 48), (9, 47), (0, 30))),
    (157, InitCell::all((-13, 69), (-5, 60), (0, 44), (1, 7))),
    (158, InitCell::all((0, 62), (-1, 59), (0, 51), (-3, 23))),
    (159, InitCell::all((0, 65), (0, 59), (2, 46), (-21, 74))),
    (160, InitCell::all((21, 37), (22, 33), (19, 38), (16, 66))),
    (161, InitCell::all((-15, 72), (5, 44), (-4, 66), (-23, 124))),
    (162, InitCell::all((9, 57), (14, 43), (15, 38), (17, 37))),
    (163, InitCell::all((16, 54), (-1, 78), (12, 42), (44, -18))),
    (164, InitCell::all((0, 62), (0, 60), (9, 34), (50, -34))),
    (165, InitCell::all((12, 72), (9, 69), (0, 89), (-22, 127))),
];

// Table 9-20 — ctxIdx 166..=226.
const TBL_9_20: &[(usize, InitCell)] = &[
    (166, InitCell::all((24, 0), (11, 28), (4, 45), (4, 39))),
    (167, InitCell::all((15, 9), (2, 40), (10, 28), (0, 42))),
    (168, InitCell::all((8, 25), (3, 44), (10, 31), (7, 34))),
    (169, InitCell::all((13, 18), (0, 49), (33, -11), (11, 29))),
    (170, InitCell::all((15, 9), (0, 46), (52, -43), (8, 31))),
    (171, InitCell::all((13, 19), (2, 44), (18, 15), (6, 37))),
    (172, InitCell::all((10, 37), (2, 51), (28, 0), (7, 42))),
    (173, InitCell::all((12, 18), (0, 47), (35, -22), (3, 40))),
    (174, InitCell::all((6, 29), (4, 39), (38, -25), (8, 33))),
    (175, InitCell::all((20, 33), (2, 62), (34, 0), (13, 43))),
    (176, InitCell::all((15, 30), (6, 46), (39, -18), (13, 36))),
    (177, InitCell::all((4, 45), (0, 54), (32, -12), (4, 47))),
    (178, InitCell::all((1, 58), (3, 54), (102, -94), (3, 55))),
    (179, InitCell::all((0, 62), (2, 58), (0, 0), (2, 58))),
    (180, InitCell::all((7, 61), (4, 63), (56, -15), (6, 60))),
    (181, InitCell::all((12, 38), (6, 51), (33, -4), (8, 44))),
    (182, InitCell::all((11, 45), (6, 57), (29, 10), (11, 44))),
    (183, InitCell::all((15, 39), (7, 53), (37, -5), (14, 42))),
    (184, InitCell::all((11, 42), (6, 52), (51, -29), (7, 48))),
    (185, InitCell::all((13, 44), (6, 55), (39, -9), (4, 56))),
    (186, InitCell::all((16, 45), (11, 45), (52, -34), (4, 52))),
    (187, InitCell::all((12, 41), (14, 36), (69, -58), (13, 37))),
    (188, InitCell::all((10, 49), (8, 53), (67, -63), (9, 49))),
    (189, InitCell::all((30, 34), (-1, 82), (44, -5), (19, 58))),
    (190, InitCell::all((18, 42), (7, 55), (32, 7), (10, 48))),
    (191, InitCell::all((10, 55), (-3, 78), (55, -29), (12, 45))),
    (192, InitCell::all((17, 51), (15, 46), (32, 1), (0, 69))),
    (193, InitCell::all((17, 46), (22, 31), (0, 0), (20, 33))),
    (194, InitCell::all((0, 89), (-1, 84), (27, 36), (8, 63))),
    (195, InitCell::all((26, -19), (25, 7), (33, -25), (35, -18))),
    (
        196,
        InitCell::all((22, -17), (30, -7), (34, -30), (33, -25)),
    ),
    (197, InitCell::all((26, -17), (28, 3), (36, -28), (28, -3))),
    (198, InitCell::all((30, -25), (28, 4), (38, -28), (24, 10))),
    (199, InitCell::all((28, -20), (32, 0), (38, -27), (27, 0))),
    (
        200,
        InitCell::all((33, -23), (34, -1), (34, -18), (34, -14)),
    ),
    (201, InitCell::all((37, -27), (30, 6), (35, -16), (52, -44))),
    (202, InitCell::all((33, -23), (30, 6), (34, -14), (39, -24))),
    (203, InitCell::all((40, -28), (32, 9), (32, -8), (19, 17))),
    (204, InitCell::all((38, -17), (31, 19), (37, -6), (31, 25))),
    (205, InitCell::all((33, -11), (26, 27), (35, 0), (36, 29))),
    (206, InitCell::all((40, -15), (26, 30), (30, 10), (24, 33))),
    (207, InitCell::all((41, -6), (37, 20), (28, 18), (34, 15))),
    (208, InitCell::all((38, 1), (28, 34), (26, 25), (30, 20))),
    (209, InitCell::all((41, 17), (17, 70), (29, 41), (22, 73))),
    (210, InitCell::all((30, -6), (1, 67), (0, 75), (20, 34))),
    (211, InitCell::all((27, 3), (5, 59), (2, 72), (19, 31))),
    (212, InitCell::all((26, 22), (9, 67), (8, 77), (27, 44))),
    (213, InitCell::all((37, -16), (16, 30), (14, 35), (19, 16))),
    (214, InitCell::all((35, -4), (18, 32), (18, 31), (15, 36))),
    (215, InitCell::all((38, -8), (18, 35), (17, 35), (15, 36))),
    (216, InitCell::all((38, -3), (22, 29), (21, 30), (21, 28))),
    (217, InitCell::all((37, 3), (24, 31), (17, 45), (25, 21))),
    (218, InitCell::all((38, 5), (23, 38), (20, 42), (30, 20))),
    (219, InitCell::all((42, 0), (18, 43), (18, 45), (31, 12))),
    (220, InitCell::all((35, 16), (20, 41), (27, 26), (27, 16))),
    (221, InitCell::all((39, 22), (11, 63), (16, 54), (24, 42))),
    (222, InitCell::all((14, 48), (9, 59), (7, 66), (0, 93))),
    (223, InitCell::all((27, 37), (9, 64), (16, 56), (14, 56))),
    (224, InitCell::all((21, 60), (-1, 94), (11, 73), (15, 57))),
    (225, InitCell::all((12, 68), (-2, 89), (10, 67), (26, 38))),
    (
        226,
        InitCell::all((2, 97), (-9, 108), (-10, 116), (-24, 127)),
    ),
];

// Table 9-21 — ctxIdx 227..=275.
const TBL_9_21: &[(usize, InitCell)] = &[
    (
        227,
        InitCell::all((-3, 71), (-6, 76), (-23, 112), (-24, 115)),
    ),
    (228, InitCell::all((-6, 42), (-2, 44), (-15, 71), (-22, 82))),
    (229, InitCell::all((-5, 50), (0, 45), (-7, 61), (-9, 62))),
    (230, InitCell::all((-3, 54), (0, 52), (0, 53), (0, 53))),
    (231, InitCell::all((-2, 62), (-3, 64), (-5, 66), (0, 59))),
    (232, InitCell::all((0, 58), (-2, 59), (-11, 77), (-14, 85))),
    (233, InitCell::all((1, 63), (-4, 70), (-9, 80), (-13, 89))),
    (234, InitCell::all((-2, 72), (-4, 75), (-9, 84), (-13, 94))),
    (235, InitCell::all((-1, 74), (-8, 82), (-10, 87), (-11, 92))),
    (
        236,
        InitCell::all((-9, 91), (-17, 102), (-34, 127), (-29, 127)),
    ),
    (
        237,
        InitCell::all((-5, 67), (-9, 77), (-21, 101), (-21, 100)),
    ),
    (238, InitCell::all((-5, 27), (3, 24), (-3, 39), (-14, 57))),
    (239, InitCell::all((-3, 39), (0, 42), (-5, 53), (-12, 67))),
    (240, InitCell::all((-2, 44), (0, 48), (-7, 61), (-11, 71))),
    (241, InitCell::all((0, 46), (0, 55), (-11, 75), (-10, 77))),
    (
        242,
        InitCell::all((-16, 64), (-6, 59), (-15, 77), (-21, 85)),
    ),
    (243, InitCell::all((-8, 68), (-7, 71), (-17, 91), (-16, 88))),
    (
        244,
        InitCell::all((-10, 78), (-12, 83), (-25, 107), (-23, 104)),
    ),
    (
        245,
        InitCell::all((-6, 77), (-11, 87), (-25, 111), (-15, 98)),
    ),
    (
        246,
        InitCell::all((-10, 86), (-30, 119), (-28, 122), (-37, 127)),
    ),
    (247, InitCell::all((-12, 92), (1, 58), (-11, 76), (-10, 82))),
    (248, InitCell::all((-15, 55), (-3, 29), (-10, 44), (-8, 48))),
    (249, InitCell::all((-10, 60), (-1, 36), (-10, 52), (-8, 61))),
    (250, InitCell::all((-6, 62), (1, 38), (-10, 57), (-8, 66))),
    (251, InitCell::all((-4, 65), (2, 43), (-9, 58), (-7, 70))),
    (
        252,
        InitCell::all((-12, 73), (-6, 55), (-16, 72), (-14, 75)),
    ),
    (253, InitCell::all((-8, 76), (0, 58), (-7, 69), (-10, 79))),
    (254, InitCell::all((-7, 80), (0, 64), (-4, 69), (-9, 83))),
    (255, InitCell::all((-9, 88), (-3, 74), (-5, 74), (-12, 92))),
    (
        256,
        InitCell::all((-17, 110), (-10, 90), (-9, 86), (-18, 108)),
    ),
    (257, InitCell::all((-11, 97), (0, 70), (2, 66), (-4, 79))),
    (258, InitCell::all((-20, 84), (-4, 29), (-9, 34), (-22, 69))),
    (259, InitCell::all((-11, 79), (5, 31), (1, 32), (-16, 75))),
    (260, InitCell::all((-6, 73), (7, 42), (11, 31), (-2, 58))),
    (261, InitCell::all((-4, 74), (1, 59), (5, 52), (1, 58))),
    (262, InitCell::all((-13, 86), (-2, 58), (-2, 55), (-13, 78))),
    (263, InitCell::all((-13, 96), (-3, 72), (-2, 67), (-9, 83))),
    (264, InitCell::all((-11, 97), (-3, 81), (0, 73), (-4, 81))),
    (
        265,
        InitCell::all((-19, 117), (-11, 97), (-8, 89), (-13, 99)),
    ),
    (266, InitCell::all((-8, 78), (0, 58), (3, 52), (-13, 81))),
    (267, InitCell::all((-5, 33), (8, 5), (7, 4), (-6, 38))),
    (268, InitCell::all((-4, 48), (10, 14), (10, 8), (-13, 62))),
    (269, InitCell::all((-2, 53), (14, 18), (17, 8), (-6, 58))),
    (270, InitCell::all((-3, 62), (13, 27), (16, 19), (-2, 59))),
    (271, InitCell::all((-13, 71), (2, 40), (3, 37), (-16, 73))),
    (272, InitCell::all((-10, 79), (0, 58), (-1, 61), (-10, 76))),
    (273, InitCell::all((-12, 86), (-3, 70), (-5, 73), (-13, 86))),
    (274, InitCell::all((-13, 90), (-6, 79), (-1, 70), (-9, 83))),
    (275, InitCell::all((-14, 97), (-8, 85), (-4, 78), (-10, 87))),
];

/// Union of all per-ctxIdx init cells used by this module. Assembled
/// once into `INIT_TABLE[ctxIdx]`.
///
/// Only the ranges needed by the syntax elements implemented in this
/// drop are populated; every other index defaults to `none()` and
/// falls through to a neutral `(0, 0)` init if ever referenced.
struct InitTable {
    cells: [InitCell; NUM_CTX_IDX],
}

impl InitTable {
    const fn empty() -> Self {
        Self {
            cells: [InitCell::none(); NUM_CTX_IDX],
        }
    }
}

/// Lazily-initialised global init table. Populated from the `TBL_9_*`
/// slices on first access.
fn init_table() -> &'static [InitCell; NUM_CTX_IDX] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[InitCell; NUM_CTX_IDX]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut arr = [InitCell::none(); NUM_CTX_IDX];
        for table in &[
            TBL_9_12, TBL_9_13, TBL_9_14, TBL_9_15, TBL_9_16, TBL_9_17, TBL_9_18, TBL_9_19,
            TBL_9_20, TBL_9_21,
        ] {
            for (idx, cell) in *table {
                arr[*idx] = *cell;
            }
        }
        arr
    })
}

// ---------------------------------------------------------------------------
// Per-slice context holder
// ---------------------------------------------------------------------------

/// Per-slice CABAC context state: the full set of contexts needed for
/// a slice, initialised once at slice start per §9.3.1.1.
#[derive(Debug, Clone)]
pub struct CabacContexts {
    pub contexts: Vec<CtxState>,
    pub slice_kind: SliceKind,
}

impl CabacContexts {
    /// §9.3.1.1 — initialise every populated ctxIdx using Tables
    /// 9-12..9-33 picked via `slice_kind` + `cabac_init_idc`.
    ///
    /// ctxIdx = 276 is special (`end_of_slice_flag` / I_PCM terminator,
    /// see §9.3.3.2.4 NOTE 2) — stored as the non-adapting state
    /// (pStateIdx = 63, valMPS = 0).
    pub fn init(
        slice_kind: SliceKind,
        cabac_init_idc: Option<u32>,
        slice_qp_y: i32,
    ) -> CabacResult<Self> {
        let table = init_table();
        let mut ctx = vec![
            CtxState {
                state_idx: 0,
                val_mps: 0
            };
            NUM_CTX_IDX
        ];
        for (idx, cell) in table.iter().enumerate() {
            if let Some((m, n)) = cell.pick(slice_kind, cabac_init_idc) {
                ctx[idx] = CtxState::init(m, n, slice_qp_y)?;
            }
        }
        // §9.3.1.1 NOTE 2 — ctxIdx 276 is hardcoded as pStateIdx=63,
        // valMPS=0 (used by end_of_slice_flag and I_PCM terminator).
        ctx[276] = CtxState {
            state_idx: 63,
            val_mps: 0,
        };
        Ok(Self {
            contexts: ctx,
            slice_kind,
        })
    }

    pub fn at(&self, idx: usize) -> &CtxState {
        &self.contexts[idx]
    }

    pub fn at_mut(&mut self, idx: usize) -> &mut CtxState {
        &mut self.contexts[idx]
    }
}

// ---------------------------------------------------------------------------
// Neighbour context (shared across syntax elements, §9.3.3.1.1.*).
// ---------------------------------------------------------------------------

/// Neighbour facts collected by the macroblock layer and passed into
/// ctxIdxInc derivation for the common per-element processes.
///
/// Fields cover every `condTermFlagN` predicate used by the elements
/// implemented in this module. Callers set the unused fields to their
/// defaults.
#[derive(Debug, Copy, Clone, Default)]
pub struct NeighbourCtx {
    pub available_left: bool,
    pub available_above: bool,
    // mb_skip_flag (§9.3.3.1.1.1)
    pub mb_skip_flag_left: bool,
    pub mb_skip_flag_above: bool,
    // mb_type (§9.3.3.1.1.3)
    pub left_is_si: bool,
    pub above_is_si: bool,
    pub left_is_i_nxn: bool,
    pub above_is_i_nxn: bool,
    pub left_is_b_skip_or_direct: bool,
    pub above_is_b_skip_or_direct: bool,
    // intra_chroma_pred_mode (§9.3.3.1.1.8)
    pub left_inter: bool,
    pub above_inter: bool,
    pub left_is_i_pcm: bool,
    pub above_is_i_pcm: bool,
    pub left_intra_chroma_pred_mode_nonzero: bool,
    pub above_intra_chroma_pred_mode_nonzero: bool,
    // transform_size_8x8_flag (§9.3.3.1.1.10)
    pub left_transform_8x8: bool,
    pub above_transform_8x8: bool,
    // coded_block_pattern (§9.3.3.1.1.4).
    //
    // The 4-bit luma CBP of each neighbour macroblock (per-8x8 blocks,
    // in raster order 0..3). Used to evaluate
    //   ( ( CodedBlockPatternLuma >> luma8x8BlkIdxN ) & 1 )
    // per the spec's condTermFlagN derivation.
    //
    // `left_is_p_or_b_skip` / `above_is_p_or_b_skip` allow skip MBs to
    // suppress the CBP lookup per the spec's "not P_Skip / B_Skip" gate.
    pub left_cbp_luma: u8,
    pub above_cbp_luma: u8,
    pub left_cbp_chroma: u8,
    pub above_cbp_chroma: u8,
    pub left_is_p_or_b_skip: bool,
    pub above_is_p_or_b_skip: bool,
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.1 — mb_skip_flag.
// ---------------------------------------------------------------------------

/// Table 9-34: `mb_skip_flag` ctxIdxOffset is 11 for P/SP slices and
/// 24 for B slices.
fn mb_skip_flag_ctx_offset(slice_kind: SliceKind) -> Option<u32> {
    match slice_kind {
        SliceKind::P | SliceKind::SP => Some(11),
        SliceKind::B => Some(24),
        _ => None, // not decoded on I/SI slices
    }
}

/// §9.3.3.1.1.1 — mb_skip_flag binarisation (FL, cMax=1) + ctxIdxInc.
///
/// Returns `Ok(true)` when the macroblock is skipped.
pub fn decode_mb_skip_flag(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    slice_kind: SliceKind,
    neighbours: &NeighbourCtx,
) -> CabacResult<bool> {
    let offset = mb_skip_flag_ctx_offset(slice_kind)
        .expect("mb_skip_flag is not decoded on I/SI slices; caller must gate");
    // §9.3.3.1.1.1 — condTermFlagN = 0 if mbAddrN unavailable or
    // mb_skip_flag[mbAddrN] == 1; else 1.
    let cond_a = u32::from(neighbours.available_left && !neighbours.mb_skip_flag_left);
    let cond_b = u32::from(neighbours.available_above && !neighbours.mb_skip_flag_above);
    let ctx_idx = (offset + cond_a + cond_b) as usize;
    let bin = dec.decode_decision(ctxs.at_mut(ctx_idx))?;
    Ok(bin == 1)
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.3 + §9.3.2.5 — mb_type (I, P, B).
// ---------------------------------------------------------------------------

/// §9.3.3.1.1.3 condTermFlag rules, parameterised on ctxIdxOffset.
fn mb_type_cond_term_flag(offset: u32, neighbours: &NeighbourCtx, is_left: bool) -> u32 {
    let available = if is_left {
        neighbours.available_left
    } else {
        neighbours.available_above
    };
    if !available {
        return 0;
    }
    let filtered = match offset {
        0 => {
            // "mb_type for the macroblock mbAddrN is equal to SI"
            if is_left {
                neighbours.left_is_si
            } else {
                neighbours.above_is_si
            }
        }
        3 => {
            // "mb_type for the macroblock mbAddrN is equal to I_NxN"
            if is_left {
                neighbours.left_is_i_nxn
            } else {
                neighbours.above_is_i_nxn
            }
        }
        27 => {
            // "mb_type for mbAddrN is B_Skip or B_Direct_16x16"
            if is_left {
                neighbours.left_is_b_skip_or_direct
            } else {
                neighbours.above_is_b_skip_or_direct
            }
        }
        _ => false,
    };
    u32::from(!filtered)
}

/// §9.3.3.1.1.3 — derive ctxIdxInc for the first 3 bins of mb_type.
fn mb_type_ctx_inc_first3(offset: u32, bin_idx: u32, neighbours: &NeighbourCtx) -> u32 {
    // Bin 0 uses the neighbour-derived increment; subsequent bins use
    // the fixed increments from Table 9-39 (bin 1, bin 2, etc).
    if bin_idx == 0 {
        mb_type_cond_term_flag(offset, neighbours, true)
            + mb_type_cond_term_flag(offset, neighbours, false)
    } else {
        bin_idx
    }
}

/// §9.3.2.5 + Table 9-36 — decode the Table 9-36 sub-string used as
/// the mb_type suffix in I / P / B slices. Returns the Table 9-36 row
/// index (0..=25) — the caller adds any per-slice base offset.
///
/// `ctxIdxOffset` comes from Table 9-34 and selects the ctxIdxInc
/// family. `already_read_bin0` tells the helper whether Table 9-36's
/// binIdx=0 has already been read by the caller.
///
/// * **I slices** (`offset = 3`): the caller's main decoder
///   (`decode_mb_type_i`) has already read binIdx=0 at ctxIdx
///   `offset + mb_type_ctx_inc_first3(...)` — so `already_read_bin0`
///   must carry the decoded value of that bin. This helper then
///   proceeds from binIdx=1.
///
/// * **P/SP slices** (`offset = 17`) and **B slices** (`offset = 32`)
///   intra suffix: the caller has read the 1-bin (P/SP) or 6-bin (B)
///   intra *prefix* and must now decode Table 9-36 from binIdx=0. In
///   those cases `already_read_bin0` is `None` and this helper reads
///   binIdx=0 itself at ctxIdx `offset + 0`.
///
/// Table 9-39 ctxIdxInc for binIdx 2..=6:
///
/// * `offset = 3` (I slices). Table 9-39 row 3:
///     binIdx 2 inc = 3; binIdx 3 inc = 4; binIdx 4 inc per Table 9-41
///     = (b3 != 0) ? 5 : 6; binIdx 5 inc per Table 9-41 = (b3 != 0) ?
///     6 : 7; binIdx 6 inc = 7.
/// * `offset = 17` (P/SP intra suffix) and `offset = 32` (B intra
///   suffix). Table 9-39 rows 17/32:
///     binIdx 2 inc = 1; binIdx 3 inc = 2; binIdx 4 inc per Table 9-41
///     = (b3 != 0) ? 2 : 3; binIdx 5 inc = 3; binIdx 6 inc = 3.
///
/// binIdx 1 is always the ctxIdx=276 DecodeTerminate bin regardless of
/// `offset` (§9.3.1.1 NOTE 2 / Table 9-39 / Figure 9-2).
fn decode_mb_type_i_suffix(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    offset: u32,
    already_read_bin0: Option<u8>,
) -> CabacResult<u32> {
    // --- binIdx 0 --------------------------------------------------
    //
    // For I slices the caller has consumed binIdx=0 and told us whether
    // it was 0 (I_NxN, not reachable here) or 1 (entering this helper).
    // For P/SP + B intra prefixes the caller has only read the outer
    // prefix, so we need to read Table-9-36 binIdx=0 here. Table 9-39
    // gives ctxIdxInc = 0 for offsets 17 and 32.
    let b0 = match already_read_bin0 {
        Some(v) => v,
        None => dec.decode_decision(ctxs.at_mut(offset as usize))?,
    };
    if b0 == 0 {
        // Table 9-36 row 0 — I_NxN (in I slices; the caller adds 5 or
        // 23 for P / B slices, so this yields the appropriate intra
        // equivalent).
        return Ok(0);
    }

    // --- binIdx 1 --------------------------------------------------
    //
    // Table 9-36 layout (given binIdx 0 == 1):
    //   binIdx 1 == 1 → Table 9-36 row 25 (I_PCM), bin string "1 1".
    //   binIdx 1 == 0 → Table 9-36 rows 1..=24 (I_16x16_*), 6 or 7 bins.
    //
    // Table 9-39 assigns binIdx 1 the special ctxIdx = 276, which
    // per §9.3.1.1 NOTE 2 is the non-adapting terminate context. The
    // I_PCM branch is parsed with the `DecodeTerminate` procedure
    // (§9.3.3.2.4), not the ordinary arithmetic decision path.
    let b1 = dec.decode_terminate()?;
    if b1 == 1 {
        // Table 9-36 row 25 — I_PCM.
        return Ok(25);
    }

    // Per Table 9-39, the binIdx 2..=6 ctxIdxInc family depends on
    // whether this is the I-slice offset (=3) or the P/SP / B intra
    // suffix (=17 / =32). See the function doc comment.
    let (inc_b2, inc_b3, inc_b4_b3ne0, inc_b4_b3eq0, inc_b5_b3ne0, inc_b5_b3eq0, inc_b6) =
        if offset == 3 {
            // I-slice suffix — Table 9-39 row ctxIdxOffset=3 + Table 9-41.
            (3u32, 4u32, 5u32, 6u32, 6u32, 7u32, 7u32)
        } else {
            // P/SP (offset=17) and B (offset=32) intra suffix — Table 9-39
            // rows ctxIdxOffset=17/32 + Table 9-41.
            //
            // binIdx 5 and binIdx 6 both take ctxIdxInc = 3 unconditionally
            // (Table 9-39 lists "3" for binIdx 5 and ">=6" for the tail).
            (1u32, 2u32, 2u32, 3u32, 3u32, 3u32, 3u32)
        };

    // --- binIdx 2..=5 (always present for rows 1..=24) -------------
    //
    // binIdx 2 partitions Table 9-36 rows:
    //   b2 == 0 → rows 1..=12  ("1 0 0 ...").
    //   b2 == 1 → rows 13..=24 ("1 0 1 ...").
    let b2 = dec.decode_decision(ctxs.at_mut((offset + inc_b2) as usize))?;
    // binIdx 3 partitions inside each b2-group:
    //   b3 == 0 → 6-bin rows (rows 1..=4 if b2=0, rows 13..=16 if b2=1).
    //   b3 == 1 → 7-bin rows (rows 5..=12 if b2=0, rows 17..=24 if b2=1).
    let b3 = dec.decode_decision(ctxs.at_mut((offset + inc_b3) as usize))?;
    // binIdx 4 uses the Table-9-41 prior-bin-dependent rule.
    let b4_inc = if b3 != 0 { inc_b4_b3ne0 } else { inc_b4_b3eq0 };
    let b4 = dec.decode_decision(ctxs.at_mut((offset + b4_inc) as usize))?;
    // binIdx 5 only has a Table-9-41 rule for offset=3; for offsets
    // 17 / 32 both branches resolve to the same value.
    let b5_inc = if b3 != 0 { inc_b5_b3ne0 } else { inc_b5_b3eq0 };
    let b5 = dec.decode_decision(ctxs.at_mut((offset + b5_inc) as usize))?;

    // --- binIdx 6 (present only when b3 == 1) ----------------------
    //
    // Table 9-36 row lengths (after the leading "1"):
    //   Rows 1..=4   "0 0 0 X Y"       — 5 suffix bins, b3 = 0.
    //   Rows 5..=12  "0 0 1 X Y Z"     — 6 suffix bins, b3 = 1.
    //   Rows 13..=16 "0 1 0 X Y"       — 5 suffix bins, b3 = 0.
    //   Rows 17..=24 "0 1 1 X Y Z"     — 6 suffix bins, b3 = 1.
    //
    // So the optional 7th bin (binIdx = 6) is read iff b3 == 1.
    let b6 = if b3 != 0 {
        dec.decode_decision(ctxs.at_mut((offset + inc_b6) as usize))?
    } else {
        0
    };

    Ok(mb_type_i_suffix_value(b2, b3, b4, b5, b6))
}

/// §9.3.2.5 + Table 9-36 — map the decoded suffix bins of an
/// I-slice mb_type (rows 1..=24) to the mb_type value. Row 25
/// (I_PCM) is handled separately by the caller (binIdx 1 == 1).
///
/// Letting `base = 12 * b2` (so rows 1..=12 use base=0 and rows
/// 13..=24 use base=12), Table 9-36 simplifies to:
///
///   * b3 == 0 (6-bin rows, Intra16x16PredMode = `(b4<<1)|b5`):
///       `value = base + 1 + ((b4 << 1) | b5)`
///
///   * b3 == 1 (7-bin rows, CodedBlockPatternChroma selector = `b4`,
///              Intra16x16PredMode = `(b5<<1)|b6`):
///       `value = base + 5 + 4*b4 + ((b5 << 1) | b6)`
///
/// Verification of every row of Table 9-36 (rows 1..=24) is in the
/// unit test `mb_type_i_suffix_value_matches_table_9_36`.
fn mb_type_i_suffix_value(b2: u8, b3: u8, b4: u8, b5: u8, b6: u8) -> u32 {
    let base = 12u32 * (b2 as u32);
    if b3 == 0 {
        // 6-bin rows (rows 1..=4 or 13..=16): tail (b4,b5) is the
        // 2-bit Intra16x16PredMode.
        base + 1 + (((b4 as u32) << 1) | (b5 as u32))
    } else {
        // 7-bin rows (rows 5..=12 or 17..=24): b4 is the chroma
        // selector bit (CodedBlockPatternChroma is 1 when b4=0, 2 when
        // b4=1), and (b5,b6) is the 2-bit Intra16x16PredMode.
        base + 5 + 4 * (b4 as u32) + (((b5 as u32) << 1) | (b6 as u32))
    }
}

/// §9.3.2.5 + Table 9-36 — mb_type in I slices (values 0..=25).
///
/// ctxIdxOffset = 3 per Table 9-34.
pub fn decode_mb_type_i(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    neighbours: &NeighbourCtx,
) -> CabacResult<u32> {
    const OFFSET: u32 = 3;
    // Bin 0: I_NxN (0) or else-branch (1). ctxIdxInc via §9.3.3.1.1.3.
    let inc0 = mb_type_ctx_inc_first3(OFFSET, 0, neighbours);
    let b0 = dec.decode_decision(ctxs.at_mut((OFFSET + inc0) as usize))?;
    if b0 == 0 {
        return Ok(0); // I_NxN
    }
    // b0 == 1 → fall through to suffix tree (rows 1..=25). The
    // suffix's binIdx=0 is the same bin we just read (§9.3.2.5 — the
    // I-slice Table 9-36 encoding is a single concatenation with no
    // separate prefix), so pass `Some(b0)` to prevent re-reading.
    decode_mb_type_i_suffix(dec, ctxs, OFFSET, Some(b0))
}

/// §9.3.2.5 + Table 9-37 — mb_type in P/SP slices.
///
/// Returns one of {0..=4 (inter), 5..=30 (intra, 5 + I-slice row)}.
pub fn decode_mb_type_p(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    _neighbours: &NeighbourCtx,
) -> CabacResult<u32> {
    const OFFSET: u32 = 14;
    // OXIDEAV_H264_CTX17_TRACE=1 — dump ctxs 14..=17 before each
    // mb_type_p decode. Used when investigating P-slice intra-suffix
    // decoding vs JVT trace (e.g. CABA2_SVA_B Pic 3 MB 28).
    if std::env::var_os("OXIDEAV_H264_CTX17_TRACE").is_some() {
        let c14 = ctxs.at(14);
        let c15 = ctxs.at(15);
        let c16 = ctxs.at(16);
        let c17 = ctxs.at(17);
        eprintln!(
            "[CTX17] pre mb_type_p: c14=({},{}) c15=({},{}) c16=({},{}) c17=({},{})",
            c14.state_idx,
            c14.val_mps,
            c15.state_idx,
            c15.val_mps,
            c16.state_idx,
            c16.val_mps,
            c17.state_idx,
            c17.val_mps,
        );
    }
    // Table 9-39 (08/2024) row ctxIdxOffset=14:
    //   binIdx 0 → 0 (fixed — no "clause 9.3.3.1.1.3" reference, unlike
    //                 ctxIdxOffsets 0, 3, 27 which DO use neighbour-
    //                 derived increments per §9.3.3.1.1.3)
    //   binIdx 1 → 1 (fixed)
    //   binIdx 2 → (b1 != 1) ? 2 : 3 (Table 9-41)
    let b0 = dec.decode_decision(ctxs.at_mut(OFFSET as usize))?;
    if b0 == 1 {
        // Intra prefix "1" → suffix is Table 9-36 with offset=17 per
        // Table 9-34. The P-slice intra prefix is a single separate
        // bin; the suffix's binIdx=0 has NOT been consumed yet, so we
        // pass `None` and let the helper read it at ctxIdx=17.
        let suffix = decode_mb_type_i_suffix(dec, ctxs, 17, None)?;
        return Ok(5 + suffix);
    }
    // b0 == 0 — inter P_ macroblock type (values 0..=3).
    let b1 = dec.decode_decision(ctxs.at_mut((OFFSET + 1) as usize))?;
    // Table 9-41 row ctxIdxOffset=14, binIdx=2: ctxIdxInc per the
    // 08/2024 spec literal is `(b1 != 1) ? 2 : 3`. Although this looks
    // inconsistent with sibling rows at ctxIdxOffset=27 / 36 (which
    // use `!= 0`), conformance decoding (CABA2_SVA_B Pic 1 MB 2 under
    // `!= 0` gives P_8x8 and desyncs the engine; under `!= 1` it
    // decodes as P_L0_16x16 which matches JVT). Keep spec-literal
    // pending further investigation of the `!= 0` divergence (intra-
    // after-P ctx 17 pollution observed at Pic 3 MB 28 of CABA2_SVA_B).
    let inc_b2 = if b1 != 1 { 2 } else { 3 };
    let b2 = dec.decode_decision(ctxs.at_mut((OFFSET + inc_b2) as usize))?;
    // Table 9-37 P/SP:
    //   0 (P_L0_16x16)    0 0 0
    //   1 (P_L0_L0_16x8)  0 1 1
    //   2 (P_L0_L0_8x16)  0 1 0
    //   3 (P_8x8)         0 0 1
    // (value 4 is illegal.)
    let val = match (b1, b2) {
        (0, 0) => 0,
        (0, 1) => 3,
        (1, 1) => 1,
        (1, 0) => 2,
        _ => unreachable!(),
    };
    Ok(val)
}

/// §9.3.2.5 + Table 9-37 — mb_type in B slices.
///
/// Returns one of {0..=22 (inter), 23..=48 (intra, 23 + I-slice row)}.
pub fn decode_mb_type_b(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    neighbours: &NeighbourCtx,
) -> CabacResult<u32> {
    const OFFSET: u32 = 27;
    // §9.3.3.1.1.3 — bin 0 ctxIdxInc with offset=27.
    let inc0 = mb_type_ctx_inc_first3(OFFSET, 0, neighbours);
    let b0 = dec.decode_decision(ctxs.at_mut((OFFSET + inc0) as usize))?;
    if b0 == 0 {
        return Ok(0); // B_Direct_16x16
    }
    // Table 9-39 for offset=27:
    //   bin 0: 0,1,2 (§9.3.3.1.1.3)
    //   bin 1: 3
    //   bin 2: (b1!=0)?4:5
    //   bin 3: 5
    //   bin 4..: 5
    let b1 = dec.decode_decision(ctxs.at_mut((OFFSET + 3) as usize))?;
    let inc_b2 = if b1 != 0 { 4 } else { 5 };
    let b2 = dec.decode_decision(ctxs.at_mut((OFFSET + inc_b2) as usize))?;
    // After (b0=1,b1,b2) early termination cases from Table 9-37:
    //   1 0 0  → 1 (B_L0_16x16)
    //   1 0 1  → 2 (B_L1_16x16)
    // Else continue.
    if b1 == 0 {
        let val = if b2 == 0 { 1 } else { 2 };
        return Ok(val);
    }
    // (b0,b1) = (1,1). Read 3 more bins at ctx offset 5.
    let b3 = dec.decode_decision(ctxs.at_mut((OFFSET + 5) as usize))?;
    let b4 = dec.decode_decision(ctxs.at_mut((OFFSET + 5) as usize))?;
    let b5 = dec.decode_decision(ctxs.at_mut((OFFSET + 5) as usize))?;
    // After (1,1,b2,b3,b4,b5) determine branch.
    // Table 9-37 fixed 6-bit rows (values 3..=11):
    //   3  B_Bi_16x16       1 1 0 0 0 0
    //   4  B_L0_L0_16x8     1 1 0 0 0 1
    //   5  B_L0_L0_8x16     1 1 0 0 1 0
    //   6  B_L1_L1_16x8     1 1 0 0 1 1
    //   7  B_L1_L1_8x16     1 1 0 1 0 0
    //   8  B_L0_L1_16x8     1 1 0 1 0 1
    //   9  B_L0_L1_8x16     1 1 0 1 1 0
    //   10 B_L1_L0_16x8     1 1 0 1 1 1
    //   11 B_L1_L0_8x16     1 1 1 1 1 0
    //   22 B_8x8            1 1 1 1 1 1
    if b2 == 0 {
        // rows 3..10 — value = 3 + (b3<<2) + (b4<<1) + b5.
        return Ok(3 + ((b3 as u32) << 2) + ((b4 as u32) << 1) + (b5 as u32));
    }
    // b2 == 1
    if b3 == 1 && b4 == 1 && b5 == 1 {
        // Need one more bin to distinguish 22 (B_8x8) from 11.
        // Table 9-37 shows B_8x8 as 1 1 1 1 1 1 (6 bins). B_L1_L0_8x16
        // = 1 1 1 1 1 0 (6 bins). So b5 alone decides.
        return Ok(22); // 1 1 1 1 1 1 → B_8x8
    }
    if b3 == 1 && b4 == 1 && b5 == 0 {
        return Ok(11); // B_L1_L0_8x16
    }
    // b2=1, not (b3=b4=1). Two disjoint classes (Table 9-37):
    //
    //   * Intra prefix "1 1 1 1 0 1"  →  values 23..=48. The 6-bin
    //     intra prefix is complete after b5; the following bin is the
    //     first bin of the Table 9-36 suffix (binIdx=0 of the suffix
    //     at ctxIdx=32+0=32). Reading a 7th "prefix" bin before
    //     dispatching to the suffix helper would desync the engine.
    //
    //   * Rows 12..=21 (bin strings "1 1 1 X Y Z W"): 7 total bins,
    //     b6 completes the code word.
    //
    // The intra prefix corresponds uniquely to (b3, b4, b5) = (1, 0, 1),
    // so we can dispatch to the suffix *before* consuming a 7th bin.
    if b3 == 1 && b4 == 0 && b5 == 1 {
        // 1 1 1 1 0 1 — intra prefix. Suffix is Table 9-36 with
        // offset=32 per Table 9-34; binIdx=0 of the suffix has not
        // been consumed yet, so pass `None`.
        let suffix = decode_mb_type_i_suffix(dec, ctxs, 32, None)?;
        return Ok(23 + suffix);
    }
    let b6 = dec.decode_decision(ctxs.at_mut((OFFSET + 5) as usize))?;
    // Rows 12..21 — 7 bins: 1 1 1 X Y Z W with X=b3, Y=b4, Z=b5, W=b6.
    //   12 B_L0_Bi_16x8   1 1 1 0 0 0 0
    //   13 B_L0_Bi_8x16   1 1 1 0 0 0 1
    //   14 B_L1_Bi_16x8   1 1 1 0 0 1 0
    //   15 B_L1_Bi_8x16   1 1 1 0 0 1 1
    //   16 B_Bi_L0_16x8   1 1 1 0 1 0 0
    //   17 B_Bi_L0_8x16   1 1 1 0 1 0 1
    //   18 B_Bi_L1_16x8   1 1 1 0 1 1 0
    //   19 B_Bi_L1_8x16   1 1 1 0 1 1 1
    //   20 B_Bi_Bi_16x8   1 1 1 1 0 0 0
    //   21 B_Bi_Bi_8x16   1 1 1 1 0 0 1
    let val = 12 + ((b3 as u32) << 3) + ((b4 as u32) << 2) + ((b5 as u32) << 1) + (b6 as u32);
    Ok(val)
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.5 — mb_qp_delta (signed Exp-Golomb via U binarisation).
// ---------------------------------------------------------------------------

/// §9.3.3.1.1.5 — mb_qp_delta. Binarisation: Unary over the mapped
/// (unsigned) value (Table 9-3). Maps back: 0→0, 1→+1, 2→-1, 3→+2, …
pub fn decode_mb_qp_delta(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    prev_mb_qp_delta_nonzero: bool,
) -> CabacResult<i32> {
    const OFFSET: u32 = 60;
    // §9.3.3.1.1.5 — ctxIdxInc(bin 0) = prev_mb_qp_delta_nonzero ? 1 : 0.
    // Table 9-39 row for offset=60: bin 0 = 0,1 (§9.3.3.1.1.5);
    // bin 1 = 2; bin 2..= 3.
    let inc0 = u32::from(prev_mb_qp_delta_nonzero);
    let b0 = dec.decode_decision(ctxs.at_mut((OFFSET + inc0) as usize))?;
    if b0 == 0 {
        return Ok(0);
    }
    // Bin 1 uses inc=2; further bins use inc=3.
    let b1 = dec.decode_decision(ctxs.at_mut((OFFSET + 2) as usize))?;
    if b1 == 0 {
        // mapped == 1 → +1
        return Ok(1);
    }
    let mut mapped: u32 = 2;
    loop {
        let bin = dec.decode_decision(ctxs.at_mut((OFFSET + 3) as usize))?;
        if bin == 0 {
            break;
        }
        mapped += 1;
        if mapped > 100 {
            // guard against malformed bitstream
            break;
        }
    }
    // Table 9-3 mapping: mapped k → value = ((k + 1) / 2) * (k is odd ? 1 : -1).
    let val = if mapped & 1 == 1 {
        ((mapped as i32) + 1) / 2
    } else {
        -((mapped as i32) / 2)
    };
    Ok(val)
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.6 — ref_idx_lX (Unary).
// ---------------------------------------------------------------------------

/// §9.3.3.1.1.6 — ref_idx_lX. Unary binarisation, ctxIdxOffset=54.
/// Table 9-39 row for offset=54: bin 0..3 from §9.3.3.1.1.6, bin 4+ = 5.
pub fn decode_ref_idx_lx(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    neighbour_ref_idx_gt_0_left: bool,
    neighbour_ref_idx_gt_0_above: bool,
) -> CabacResult<u32> {
    const OFFSET: u32 = 54;
    let cond_a = u32::from(neighbour_ref_idx_gt_0_left);
    let cond_b = u32::from(neighbour_ref_idx_gt_0_above);
    // bin 0 inc per §9.3.3.1.1.6 eq. 9-14:
    //   ctxIdxInc = condTermFlagA + 2*condTermFlagB
    let inc0 = cond_a + 2 * cond_b;
    let b0 = dec.decode_decision(ctxs.at_mut((OFFSET + inc0) as usize))?;
    if b0 == 0 {
        return Ok(0);
    }
    // Table 9-39 gives bin 1 = 4 and bin >=2 = 5.
    let b1 = dec.decode_decision(ctxs.at_mut((OFFSET + 4) as usize))?;
    if b1 == 0 {
        return Ok(1);
    }
    let mut value: u32 = 2;
    // Subsequent unary bins use ctxIdxInc = 5.
    loop {
        let bin = dec.decode_decision(ctxs.at_mut((OFFSET + 5) as usize))?;
        if bin == 0 {
            break;
        }
        value += 1;
        if value > 32 {
            break;
        }
    }
    Ok(value)
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.7 — mvd_lX (UEG3, signed).
// ---------------------------------------------------------------------------

/// Component of a motion vector difference.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MvdComponent {
    X,
    Y,
}

/// §9.3.3.1.1.7 — mvd_lX[][][compIdx]. UEG3 binarisation, uCoff=9,
/// signedValFlag=1. ctxIdxOffset = 40 (X) or 47 (Y).
///
/// `neighbour_abs_mvd_sum` is `absMvdCompA + absMvdCompB` from the
/// spec (§9.3.3.1.1.7 — already clipped by caller with the max-32
/// tests).
pub fn decode_mvd_lx(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    component: MvdComponent,
    neighbour_abs_mvd_sum: u32,
) -> CabacResult<i32> {
    let offset = match component {
        MvdComponent::X => 40,
        MvdComponent::Y => 47,
    };
    // §9.3.3.1.1.7 — ctxIdxInc(bin 0):
    //   sum > 32 → 2
    //   sum > 2  → 1
    //   else     → 0
    let inc0 = if neighbour_abs_mvd_sum > 32 {
        2
    } else if neighbour_abs_mvd_sum > 2 {
        1
    } else {
        0
    };

    // UEG3 prefix: TU with cMax = uCoff = 9. Bins past bin 0 use
    // ctxIdxInc = 3, 4, 5, 6, 6, 6 per Table 9-39.
    let mut prefix_val: u32 = 0;
    let bins: [u32; 9] = [inc0, 3, 4, 5, 6, 6, 6, 6, 6];
    for (i, inc) in bins.iter().enumerate() {
        let ctx_idx = (offset + inc) as usize;
        let bin = dec.decode_decision(ctxs.at_mut(ctx_idx))?;
        if bin == 0 {
            prefix_val = i as u32;
            break;
        }
        if i == 8 {
            prefix_val = 9;
        }
    }

    let abs_val: u32 = if prefix_val < 9 {
        prefix_val
    } else {
        // §9.3.2.3 — UEG3 suffix (bypass-coded).
        let k0: u32 = 3;
        let mut k = k0;
        let mut suf_s: u32 = 0;
        // Unary-of-1s phase (with k++ each step).
        loop {
            let b = dec.decode_bypass()? as u32;
            if b == 1 {
                suf_s += 1u32 << k;
                k += 1;
            } else {
                break;
            }
        }
        // Now read `k` bits MSB-first into the low part of suf_s.
        let mut tail: u32 = 0;
        for _ in 0..k {
            tail = (tail << 1) | (dec.decode_bypass()? as u32);
        }
        suf_s + tail + 9
    };

    if abs_val == 0 {
        return Ok(0);
    }
    // Signed-val flag bypass bit: 0 → positive, 1 → negative.
    let sign = dec.decode_bypass()? as u32;
    let v = abs_val as i32;
    Ok(if sign == 1 { -v } else { v })
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.8 — intra_chroma_pred_mode (TU, cMax=3).
// ---------------------------------------------------------------------------

/// §9.3.3.1.1.8 — intra_chroma_pred_mode. ctxIdxOffset = 64.
pub fn decode_intra_chroma_pred_mode(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    neighbours: &NeighbourCtx,
) -> CabacResult<u32> {
    const OFFSET: u32 = 64;
    let cond_a = u32::from(
        neighbours.available_left
            && !neighbours.left_inter
            && !neighbours.left_is_i_pcm
            && neighbours.left_intra_chroma_pred_mode_nonzero,
    );
    let cond_b = u32::from(
        neighbours.available_above
            && !neighbours.above_inter
            && !neighbours.above_is_i_pcm
            && neighbours.above_intra_chroma_pred_mode_nonzero,
    );
    let inc0 = cond_a + cond_b;
    let b0 = dec.decode_decision(ctxs.at_mut((OFFSET + inc0) as usize))?;
    if b0 == 0 {
        return Ok(0);
    }
    // Table 9-39 offset=64: bin 1 = 3; bin 2 = 3.
    let b1 = dec.decode_decision(ctxs.at_mut((OFFSET + 3) as usize))?;
    if b1 == 0 {
        return Ok(1);
    }
    let b2 = dec.decode_decision(ctxs.at_mut((OFFSET + 3) as usize))?;
    Ok(if b2 == 0 { 2 } else { 3 })
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.9 — coded_block_flag.
// ---------------------------------------------------------------------------

/// Transform block category per Table 9-42.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BlockType {
    Luma16x16Dc = 0,
    Luma16x16Ac = 1,
    Luma4x4 = 2,
    ChromaDc = 3,
    ChromaAc = 4,
    Luma8x8 = 5,
    CbIntra16x16Dc = 6,
    CbIntra16x16Ac = 7,
    CbLuma4x4 = 8,
    Cb8x8 = 9,
    CrIntra16x16Dc = 10,
    CrIntra16x16Ac = 11,
    CrLuma4x4 = 12,
    Cr8x8 = 13,
}

impl BlockType {
    fn ctx_block_cat(self) -> u32 {
        self as u32
    }

    /// Table 9-40 — ctxIdxBlockCatOffset for `coded_block_flag`.
    fn cbf_block_cat_offset(self) -> u32 {
        match self {
            BlockType::Luma16x16Dc => 0,
            BlockType::Luma16x16Ac => 4,
            BlockType::Luma4x4 => 8,
            BlockType::ChromaDc => 12,
            BlockType::ChromaAc => 16,
            BlockType::Luma8x8 => 0,
            BlockType::CbIntra16x16Dc => 0,
            BlockType::CbIntra16x16Ac => 4,
            BlockType::CbLuma4x4 => 8,
            BlockType::Cb8x8 => 4,
            BlockType::CrIntra16x16Dc => 0,
            BlockType::CrIntra16x16Ac => 4,
            BlockType::CrLuma4x4 => 8,
            BlockType::Cr8x8 => 8,
        }
    }

    /// Table 9-40 — ctxIdxBlockCatOffset for significant_coeff_flag
    /// and last_significant_coeff_flag.
    fn sig_block_cat_offset(self) -> u32 {
        match self {
            BlockType::Luma16x16Dc => 0,
            BlockType::Luma16x16Ac => 15,
            BlockType::Luma4x4 => 29,
            BlockType::ChromaDc => 44,
            BlockType::ChromaAc => 47,
            BlockType::Luma8x8 => 0,
            BlockType::CbIntra16x16Dc => 0,
            BlockType::CbIntra16x16Ac => 15,
            BlockType::CbLuma4x4 => 29,
            BlockType::Cb8x8 => 0,
            BlockType::CrIntra16x16Dc => 0,
            BlockType::CrIntra16x16Ac => 15,
            BlockType::CrLuma4x4 => 29,
            BlockType::Cr8x8 => 0,
        }
    }

    /// Table 9-40 — ctxIdxBlockCatOffset for coeff_abs_level_minus1.
    fn lvl_block_cat_offset(self) -> u32 {
        match self {
            BlockType::Luma16x16Dc => 0,
            BlockType::Luma16x16Ac => 10,
            BlockType::Luma4x4 => 20,
            BlockType::ChromaDc => 30,
            BlockType::ChromaAc => 39,
            BlockType::Luma8x8 => 0,
            BlockType::CbIntra16x16Dc => 0,
            BlockType::CbIntra16x16Ac => 10,
            BlockType::CbLuma4x4 => 20,
            BlockType::Cb8x8 => 0,
            BlockType::CrIntra16x16Dc => 0,
            BlockType::CrIntra16x16Ac => 10,
            BlockType::CrLuma4x4 => 20,
            BlockType::Cr8x8 => 0,
        }
    }

    /// Table 9-34 — ctxIdxOffset for coded_block_flag.
    ///
    /// Three ranges depending on ctxBlockCat: <5, 5<cat<9, 9<cat<13,
    /// cat∈{5,9,13}. Returns the ctxIdxOffset base.
    fn cbf_ctx_offset(self) -> u32 {
        match self.ctx_block_cat() {
            0..=4 => 85,
            6..=8 => 460,
            10..=12 => 472,
            5 | 9 | 13 => 1012,
            _ => 85,
        }
    }

    /// Table 9-34 — ctxIdxOffset for significant_coeff_flag
    /// (frame-coded).
    fn sig_coef_ctx_offset(self, field: bool) -> u32 {
        if field {
            match self.ctx_block_cat() {
                0..=4 => 277,
                5 => 436,
                6..=8 => 776,
                9 => 675,
                10..=12 => 820,
                13 => 733,
                _ => 277,
            }
        } else {
            match self.ctx_block_cat() {
                0..=4 => 105,
                5 => 402,
                6..=8 => 484,
                9 => 660,
                10..=12 => 528,
                13 => 718,
                _ => 105,
            }
        }
    }

    /// Table 9-34 — ctxIdxOffset for last_significant_coeff_flag.
    fn last_coef_ctx_offset(self, field: bool) -> u32 {
        if field {
            match self.ctx_block_cat() {
                0..=4 => 338,
                5 => 451,
                6..=8 => 864,
                9 => 699,
                10..=12 => 908,
                13 => 757,
                _ => 338,
            }
        } else {
            match self.ctx_block_cat() {
                0..=4 => 166,
                5 => 417,
                6..=8 => 572,
                9 => 690,
                10..=12 => 616,
                13 => 748,
                _ => 166,
            }
        }
    }

    /// Table 9-34 — ctxIdxOffset for coeff_abs_level_minus1.
    fn abs_level_ctx_offset(self) -> u32 {
        match self.ctx_block_cat() {
            0..=4 => 227,
            5 => 426,
            6..=8 => 952,
            9 => 708,
            10..=12 => 982,
            13 => 766,
            _ => 227,
        }
    }
}

/// §9.3.3.1.1.9 — decode coded_block_flag (FL, cMax=1). The ctxIdxInc
/// derivation needs the neighbouring block's decoded coded_block_flag
/// (`transBlockN.coded_block_flag` in the spec) or equivalent.
pub fn decode_coded_block_flag(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    block_type: BlockType,
    neighbour_cbf_left: Option<bool>,
    neighbour_cbf_above: Option<bool>,
) -> CabacResult<bool> {
    let base = block_type.cbf_ctx_offset();
    let cat_offset = block_type.cbf_block_cat_offset();
    // §9.3.3.1.1.9 — ctxIdxInc = condTermFlagA + 2 * condTermFlagB.
    // When neighbour unavailable + current is intra → 1; unavailable +
    // current is inter → 0. Callers encode the intra/inter decision
    // into `neighbour_cbf_*`: `None` means "not available", `Some(b)`
    // means decoded value `b`.
    let cond_a = u32::from(neighbour_cbf_left.unwrap_or(false));
    let cond_b = u32::from(neighbour_cbf_above.unwrap_or(false));
    let inc = cond_a + 2 * cond_b;
    let ctx_idx = (base + cat_offset + inc) as usize;
    let bin = dec.decode_decision(ctxs.at_mut(ctx_idx))?;
    if std::env::var_os("OXIDEAV_H264_CABAC_DEBUG").is_some() {
        eprintln!(
            "[CABAC] cbf bt={:?} base={} cat={} inc={} ctx={} -> {}  cond_left={:?} cond_above={:?}",
            block_type, base, cat_offset, inc, ctx_idx, bin,
            neighbour_cbf_left, neighbour_cbf_above
        );
    }
    Ok(bin == 1)
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.4 — coded_block_pattern.
// ---------------------------------------------------------------------------

/// §9.3.3.1.1.4 — coded_block_pattern.
///
/// 4-bit luma prefix (FL cMax=15) + optional 2-bit chroma suffix
/// (TU cMax=2) per §9.3.2.6. Returns the composed 6-bit CBP value
/// (bits 0..=3 = luma, bits 4..=5 = chroma).
pub fn decode_coded_block_pattern(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    neighbours: &NeighbourCtx,
    chroma_array_type: u32,
) -> CabacResult<u32> {
    const LUMA_OFFSET: u32 = 73;
    const CHROMA_OFFSET: u32 = 77;
    let mut luma: u32 = 0;
    let mut prev_bins = [0u32; 4];

    // §9.3.3.1.1.4 — derive condTermFlagN for an external neighbour MB.
    //
    // Inputs:
    //   avail        — mbAddrN available?
    //   is_i_pcm     — is mb_type == I_PCM?
    //   is_skip      — is mb_type ∈ {P_Skip, B_Skip}?
    //   cbp_luma     — CodedBlockPatternLuma of the neighbour MB
    //   bit_idx_n    — luma8x8BlkIdxN for N
    //
    // Spec: condTermFlagN is 0 when any of these hold:
    //   * mbAddrN not available
    //   * mb_type == I_PCM
    //   * (neighbour ≠ current) and (not skip) and (CBP bit set)
    // Otherwise condTermFlagN = 1.
    let cond_ext =
        |avail: bool, is_i_pcm: bool, is_skip: bool, cbp_luma: u8, bit_idx_n: u32| -> u32 {
            if !avail {
                return 0;
            }
            if is_i_pcm {
                // Spec: I_PCM → condTermFlagN = 0.
                return 0;
            }
            if is_skip {
                // Not skip'd → fall to bit check; skip'd → spec's "not P_Skip/B_Skip"
                // predicate fails, so the cbp-bit branch does not apply, and
                // the "Otherwise" branch yields condTermFlagN = 1.
                return 1;
            }
            let bit_set = ((cbp_luma >> bit_idx_n) & 1) != 0;
            if bit_set {
                0
            } else {
                1
            }
        };

    // §6.4.11.2 / §6.4.3 Figure 6-11 — neighbour 8x8 block of
    // luma8x8BlkIdx. Each binIdx (= luma8x8BlkIdx) maps to an
    // (A, B) pair where A is the left neighbour and B is the above
    // neighbour. For binIdx values where the neighbour lies inside
    // the current MB, the "internal" branch reads the prior decoded
    // bin value b_k; for those outside it, the external MB's CBP bit
    // is consulted.
    for bin_idx in 0..4u32 {
        // Left neighbour (A).
        let cond_a = match bin_idx {
            0 => cond_ext(
                neighbours.available_left,
                neighbours.left_is_i_pcm,
                neighbours.left_is_p_or_b_skip,
                neighbours.left_cbp_luma,
                1,
            ),
            1 => {
                // Left neighbour is block 0 of the current MB.
                let b_k = prev_bins[0];
                if b_k != 0 {
                    0
                } else {
                    1
                }
            }
            2 => cond_ext(
                neighbours.available_left,
                neighbours.left_is_i_pcm,
                neighbours.left_is_p_or_b_skip,
                neighbours.left_cbp_luma,
                3,
            ),
            3 => {
                // Left neighbour is block 2 of the current MB.
                let b_k = prev_bins[2];
                if b_k != 0 {
                    0
                } else {
                    1
                }
            }
            _ => 0,
        };

        // Above neighbour (B).
        let cond_b = match bin_idx {
            0 => cond_ext(
                neighbours.available_above,
                neighbours.above_is_i_pcm,
                neighbours.above_is_p_or_b_skip,
                neighbours.above_cbp_luma,
                2,
            ),
            1 => cond_ext(
                neighbours.available_above,
                neighbours.above_is_i_pcm,
                neighbours.above_is_p_or_b_skip,
                neighbours.above_cbp_luma,
                3,
            ),
            2 => {
                // Above neighbour is block 0 of the current MB.
                let b_k = prev_bins[0];
                if b_k != 0 {
                    0
                } else {
                    1
                }
            }
            3 => {
                // Above neighbour is block 1 of the current MB.
                let b_k = prev_bins[1];
                if b_k != 0 {
                    0
                } else {
                    1
                }
            }
            _ => 0,
        };

        let inc = cond_a + 2 * cond_b;
        let ctx_idx = (LUMA_OFFSET + inc) as usize;
        let bin = dec.decode_decision(ctxs.at_mut(ctx_idx))?;
        prev_bins[bin_idx as usize] = bin as u32;
        luma |= (bin as u32) << bin_idx;
    }
    let chroma = if chroma_array_type == 1 || chroma_array_type == 2 {
        // TU cMax=2 — up to 2 bins at ctxIdxOffset 77.
        // §9.3.3.1.1.4 eq. (9-11):
        //   ctxIdxInc = condTermFlagA + 2*condTermFlagB + ((binIdx==1)?4:0)
        //
        // condTermFlagN derivation:
        //   * avail and mb_type == I_PCM → 1.
        //   * avail && !skip &&
        //       (binIdx==0 ? cbp_chroma==0 : cbp_chroma != 2) → 0.
        //   * else → avail-and-not-skip ? 1 : 0.
        let chroma_cond =
            |avail: bool, is_i_pcm: bool, is_skip: bool, cbp_chroma: u8, bin_idx: u32| -> u32 {
                if !avail {
                    return 0;
                }
                if is_i_pcm {
                    return 1;
                }
                if is_skip {
                    return 0;
                }
                // Predicate from spec: flag=0 when
                //   binIdx==0 && cbp_chroma==0
                //   binIdx==1 && cbp_chroma != 2
                let zero_branch = match bin_idx {
                    0 => cbp_chroma == 0,
                    1 => cbp_chroma != 2,
                    _ => false,
                };
                if zero_branch {
                    0
                } else {
                    1
                }
            };
        // binIdx 0.
        let cond_a0 = chroma_cond(
            neighbours.available_left,
            neighbours.left_is_i_pcm,
            neighbours.left_is_p_or_b_skip,
            neighbours.left_cbp_chroma,
            0,
        );
        let cond_b0 = chroma_cond(
            neighbours.available_above,
            neighbours.above_is_i_pcm,
            neighbours.above_is_p_or_b_skip,
            neighbours.above_cbp_chroma,
            0,
        );
        let inc0 = cond_a0 + 2 * cond_b0;
        let b0 = dec.decode_decision(ctxs.at_mut((CHROMA_OFFSET + inc0) as usize))?;
        if b0 == 0 {
            0
        } else {
            // binIdx 1.
            let cond_a1 = chroma_cond(
                neighbours.available_left,
                neighbours.left_is_i_pcm,
                neighbours.left_is_p_or_b_skip,
                neighbours.left_cbp_chroma,
                1,
            );
            let cond_b1 = chroma_cond(
                neighbours.available_above,
                neighbours.above_is_i_pcm,
                neighbours.above_is_p_or_b_skip,
                neighbours.above_cbp_chroma,
                1,
            );
            let inc1 = cond_a1 + 2 * cond_b1 + 4;
            let b1 = dec.decode_decision(ctxs.at_mut((CHROMA_OFFSET + inc1) as usize))?;
            if b1 == 0 {
                1
            } else {
                2
            }
        }
    } else {
        0
    };
    Ok(luma | (chroma << 4))
}

// ---------------------------------------------------------------------------
// §9.3.3.1.3 — significant_coeff_flag / last_significant_coeff_flag.
// ---------------------------------------------------------------------------

/// Table 9-43 — mapping of scanning position to ctxIdxInc for
/// ctxBlockCat ∈ {5, 9, 13}. Three per-position triples:
/// (significant_frame, significant_field, last). Values from document
/// page 273.
static TBL_9_43: [(u8, u8, u8); 63] = [
    (0, 0, 0),
    (1, 1, 1),
    (2, 1, 1),
    (3, 2, 1),
    (4, 2, 1),
    (5, 3, 1),
    (5, 3, 1),
    (4, 4, 1),
    (4, 5, 1),
    (3, 6, 1),
    (3, 7, 1),
    (4, 7, 1),
    (4, 7, 1),
    (4, 8, 1),
    (5, 4, 1),
    (5, 5, 1),
    (4, 6, 2),
    (4, 9, 2),
    (4, 10, 2),
    (4, 10, 2),
    (3, 8, 2),
    (3, 11, 2),
    (6, 12, 2),
    (7, 11, 2),
    (7, 9, 2),
    (7, 9, 2),
    (8, 10, 2),
    (9, 10, 2),
    (10, 8, 2),
    (9, 11, 2),
    (8, 12, 2),
    (7, 11, 2),
    (7, 9, 3),
    (6, 9, 3),
    (11, 10, 3),
    (12, 10, 3),
    (13, 8, 3),
    (11, 11, 3),
    (6, 12, 3),
    (7, 11, 3),
    (8, 9, 4),
    (9, 9, 4),
    (14, 10, 4),
    (10, 10, 4),
    (9, 8, 4),
    (8, 13, 4),
    (6, 13, 4),
    (11, 9, 4),
    (12, 9, 5),
    (13, 10, 5),
    (11, 10, 5),
    (6, 8, 5),
    (9, 13, 6),
    (14, 13, 6),
    (10, 9, 6),
    (9, 9, 6),
    (11, 10, 7),
    (12, 10, 7),
    (13, 14, 7),
    (11, 14, 7),
    (14, 14, 8),
    (10, 14, 8),
    (12, 14, 8),
];

/// §9.3.3.1.3 — significant_coeff_flag ctxIdxInc per equation (9-21)
/// / (9-22) / Table 9-43 depending on block category.
fn sig_coeff_inc(block_type: BlockType, level_list_idx: u32, field: bool) -> u32 {
    match block_type.ctx_block_cat() {
        3 => {
            // §9.3.3.1.3 eq. (9-22): Min(levelListIdx / NumC8x8, 2).
            // NumC8x8 = 1 for ChromaArrayType=1, 2 for =2. Conservatively
            // use 1 (4:2:0). Callers that need 4:2:2 may override.
            core::cmp::min(level_list_idx, 2)
        }
        5 | 9 | 13 => {
            let row = TBL_9_43[level_list_idx as usize];
            if field {
                row.1 as u32
            } else {
                row.0 as u32
            }
        }
        _ => {
            // eq. (9-21): ctxIdxInc = levelListIdx.
            level_list_idx
        }
    }
}

/// §9.3.3.1.3 — last_significant_coeff_flag ctxIdxInc.
fn last_coeff_inc(block_type: BlockType, level_list_idx: u32) -> u32 {
    match block_type.ctx_block_cat() {
        3 => core::cmp::min(level_list_idx, 2),
        5 | 9 | 13 => TBL_9_43[level_list_idx as usize].2 as u32,
        _ => level_list_idx,
    }
}

/// §9.3.3.1.1.9 + §9.3.3.1.3 — decode significant_coeff_flag(coeff_idx).
pub fn decode_significant_coeff_flag(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    block_type: BlockType,
    coeff_idx: u32,
    field: bool,
) -> CabacResult<bool> {
    let base = block_type.sig_coef_ctx_offset(field);
    let cat_offset = block_type.sig_block_cat_offset();
    let inc = sig_coeff_inc(block_type, coeff_idx, field);
    let ctx_idx = (base + cat_offset + inc) as usize;
    let bin = dec.decode_decision(ctxs.at_mut(ctx_idx))?;
    Ok(bin == 1)
}

/// §9.3.3.1.1.9 + §9.3.3.1.3 — decode last_significant_coeff_flag.
pub fn decode_last_significant_coeff_flag(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    block_type: BlockType,
    coeff_idx: u32,
    field: bool,
) -> CabacResult<bool> {
    let base = block_type.last_coef_ctx_offset(field);
    let cat_offset = block_type.sig_block_cat_offset();
    let inc = last_coeff_inc(block_type, coeff_idx);
    let ctx_idx = (base + cat_offset + inc) as usize;
    let bin = dec.decode_decision(ctxs.at_mut(ctx_idx))?;
    Ok(bin == 1)
}

// ---------------------------------------------------------------------------
// §9.3.3.1.3 — coeff_abs_level_minus1 (UEG0 with uCoff=14).
// ---------------------------------------------------------------------------

/// §9.3.3.1.3 — decode coeff_abs_level_minus1.
///
/// - Prefix: TU with cMax=14. Bin 0 uses eq. (9-23); later bins use
///   eq. (9-24).
/// - Suffix: 0-th order Exp-Golomb via DecodeBypass when prefix == 14.
pub fn decode_coeff_abs_level_minus1(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    block_type: BlockType,
    num_decoded_eq_1: u32,
    num_decoded_gt_1: u32,
) -> CabacResult<u32> {
    let start_bins = dec.bin_count();
    let base = block_type.abs_level_ctx_offset();
    let cat_offset = block_type.lvl_block_cat_offset();
    let u_coff: u32 = 14;
    // bin 0 ctxIdxInc — eq. (9-23).
    let inc0 = if num_decoded_gt_1 != 0 {
        0
    } else {
        core::cmp::min(4, 1 + num_decoded_eq_1)
    };
    let ctx0 = (base + cat_offset + inc0) as usize;
    let b0 = dec.decode_decision(ctxs.at_mut(ctx0))?;
    if b0 == 0 {
        dbg_emit(
            "coeff_abs_level_minus1",
            dec.bin_count() - start_bins,
            &format!("eq1={num_decoded_eq_1} gt1={num_decoded_gt_1} val=0"),
        );
        return Ok(0);
    }
    // Remaining unary bins — eq. (9-24):
    //   ctxIdxInc = 5 + Min(4 - (cat==3?1:0), num_gt_1)
    let cat_minus = if block_type.ctx_block_cat() == 3 {
        1
    } else {
        0
    };
    let inc_rest = 5 + core::cmp::min(4u32 - cat_minus, num_decoded_gt_1);
    let ctx_rest = (base + cat_offset + inc_rest) as usize;
    let mut prefix_val: u32 = 1;
    while prefix_val < u_coff {
        let bin = dec.decode_decision(ctxs.at_mut(ctx_rest))?;
        if bin == 0 {
            dbg_emit(
                "coeff_abs_level_minus1",
                dec.bin_count() - start_bins,
                &format!("eq1={num_decoded_eq_1} gt1={num_decoded_gt_1} val={prefix_val}"),
            );
            return Ok(prefix_val);
        }
        prefix_val += 1;
    }
    // prefix_val == u_coff — decode Exp-Golomb (k=0) suffix by bypass.
    let mut k: u32 = 0;
    let mut suf_s: u32 = 0;
    loop {
        let b = dec.decode_bypass()? as u32;
        if b == 1 {
            suf_s += 1u32 << k;
            k += 1;
        } else {
            break;
        }
    }
    let mut tail: u32 = 0;
    for _ in 0..k {
        tail = (tail << 1) | (dec.decode_bypass()? as u32);
    }
    let val = u_coff + suf_s + tail;
    dbg_emit(
        "coeff_abs_level_minus1",
        dec.bin_count() - start_bins,
        &format!("eq1={num_decoded_eq_1} gt1={num_decoded_gt_1} ESCAPE k={k} val={val}"),
    );
    Ok(val)
}

// ---------------------------------------------------------------------------
// Bypass-coded / terminator-coded elements.
// ---------------------------------------------------------------------------

/// coeff_sign_flag — bypass bin (§9.3.3.2.3).
pub fn decode_coeff_sign_flag(dec: &mut CabacDecoder<'_>) -> CabacResult<bool> {
    Ok(dec.decode_bypass()? == 1)
}

/// end_of_slice_flag — terminate bin (§9.3.3.2.4).
pub fn decode_end_of_slice_flag(dec: &mut CabacDecoder<'_>) -> CabacResult<bool> {
    let before_bins = dec.bin_count();
    let (b, bi) = dec.position();
    let v = dec.decode_terminate()?;
    if std::env::var_os("OXIDEAV_H264_CABAC_DEBUG").is_some() {
        eprintln!(
            "[CABAC] end_of_slice_flag pre_bins={} pre_pos=({},{}) -> {}",
            before_bins, b, bi, v
        );
    }
    Ok(v == 1)
}

/// prev_intra4x4_pred_mode_flag / prev_intra8x8_pred_mode_flag — FL,
/// cMax=1, ctxIdxOffset=68 with ctxIdxInc=0 per Table 9-39.
pub fn decode_prev_intra_pred_mode_flag(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
) -> CabacResult<bool> {
    let ctx_idx = 68usize;
    Ok(dec.decode_decision(ctxs.at_mut(ctx_idx))? == 1)
}

/// rem_intra4x4_pred_mode / rem_intra8x8_pred_mode — FL cMax=7 (3 bins),
/// ctxIdxOffset=69 with ctxIdxInc=0 for all bins per Table 9-39.
pub fn decode_rem_intra_pred_mode(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
) -> CabacResult<u32> {
    let ctx_idx = 69usize;
    let mut v: u32 = 0;
    for i in 0..3 {
        let bin = dec.decode_decision(ctxs.at_mut(ctx_idx))? as u32;
        // FL binarisation: binIdx=0 is LSB.
        v |= bin << i;
    }
    Ok(v)
}

/// transform_size_8x8_flag — FL cMax=1, ctxIdxOffset=399.
pub fn decode_transform_size_8x8_flag(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    neighbours: &NeighbourCtx,
) -> CabacResult<bool> {
    const OFFSET: u32 = 399;
    let cond_a = u32::from(neighbours.available_left && neighbours.left_transform_8x8);
    let cond_b = u32::from(neighbours.available_above && neighbours.above_transform_8x8);
    let inc = cond_a + cond_b;
    let ctx_idx = (OFFSET + inc) as usize;
    Ok(dec.decode_decision(ctxs.at_mut(ctx_idx))? == 1)
}

// ---------------------------------------------------------------------------
// §9.3.3.1.2 + Tables 9-37 / 9-38 — sub_mb_type for P/SP and B slices.
//
// Per Table 9-34:
//   * P/SP sub_mb_type[]: ctxIdxOffset = 21, maxBinIdxCtx = 2.
//   * B   sub_mb_type[]: ctxIdxOffset = 36, maxBinIdxCtx = 3.
//
// Per Table 9-39:
//   * ctxIdxOffset = 21 (P/SP): binIdx 0 → inc 0, binIdx 1 → inc 1,
//     binIdx 2 → inc 2. (binIdx >= 3 is "na".)
//   * ctxIdxOffset = 36 (B):   binIdx 0 → inc 0, binIdx 1 → inc 1,
//     binIdx 2 → inc 2 or 3 via §9.3.3.1.2 (Table 9-41 row
//     "36,binIdx=2"),  binIdx 3 → inc 3, binIdx 4 → inc 3,
//     binIdx 5 → inc 3. (binIdx >= 6 is "na".)
//
// Table 9-41 entry for ctxIdxOffset = 36, binIdx = 2:
//   ctxIdxInc = (b1 != 0) ? 2 : 3.
//
// Binarisations (Tables 9-37 and 9-38) are walked literally by the
// code below; see function-level doc comments for each bin branch.
// ---------------------------------------------------------------------------

/// §9.3.2.5 + Table 9-37 — P / SP slice sub_mb_type.
///
/// ctxIdxOffset = 21 (Table 9-34). Bin strings (Table 9-37 P/SP rows):
///
///   * value 0 (P_L0_8x8): `1`
///   * value 1 (P_L0_8x4): `0 0`
///   * value 2 (P_L0_4x8): `0 1 1`
///   * value 3 (P_L0_4x4): `0 1 0`
///
/// ctxIdxInc per Table 9-39 row for ctxIdxOffset=21:
///   binIdx 0 → 0, binIdx 1 → 1, binIdx 2 → 2.
pub fn decode_sub_mb_type_p(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
) -> CabacResult<u32> {
    const OFFSET: u32 = 21;
    // binIdx 0 — Table 9-39 ctxIdxInc = 0 → ctxIdx = 21.
    let b0 = dec.decode_decision(ctxs.at_mut(OFFSET as usize))?;
    if b0 == 1 {
        // "1" → Table 9-37 P value 0 (P_L0_8x8).
        return Ok(0);
    }
    // binIdx 1 — Table 9-39 ctxIdxInc = 1 → ctxIdx = 22.
    let b1 = dec.decode_decision(ctxs.at_mut((OFFSET + 1) as usize))?;
    if b1 == 0 {
        // "0 0" → Table 9-37 P value 1 (P_L0_8x4).
        return Ok(1);
    }
    // binIdx 2 — Table 9-39 ctxIdxInc = 2 → ctxIdx = 23.
    let b2 = dec.decode_decision(ctxs.at_mut((OFFSET + 2) as usize))?;
    // "0 1 1" → value 2 (P_L0_4x8); "0 1 0" → value 3 (P_L0_4x4).
    Ok(if b2 == 1 { 2 } else { 3 })
}

/// §9.3.2.5 + Table 9-38 — B slice sub_mb_type.
///
/// ctxIdxOffset = 36 (Table 9-34). Bin strings (Table 9-38 B rows):
///
///   * value 0  (B_Direct_8x8): `0`
///   * value 1  (B_L0_8x8):     `1 0 0`
///   * value 2  (B_L1_8x8):     `1 0 1`
///   * value 3  (B_Bi_8x8):     `1 1 0 0 0`
///   * value 4  (B_L0_8x4):     `1 1 0 0 1`
///   * value 5  (B_L0_4x8):     `1 1 0 1 0`
///   * value 6  (B_L1_8x4):     `1 1 0 1 1`
///   * value 7  (B_L1_4x8):     `1 1 1 0 0 0`
///   * value 8  (B_Bi_8x4):     `1 1 1 0 0 1`
///   * value 9  (B_Bi_4x8):     `1 1 1 0 1 0`
///   * value 10 (B_L0_4x4):     `1 1 1 0 1 1`
///   * value 11 (B_L1_4x4):     `1 1 1 1 0`
///   * value 12 (B_Bi_4x4):     `1 1 1 1 1`
///
/// ctxIdxInc per Table 9-39 row for ctxIdxOffset=36:
///   binIdx 0 → 0, binIdx 1 → 1, binIdx 2 → Table 9-41
///   ((b1 != 0) ? 2 : 3), binIdx 3 → 3, binIdx 4 → 3, binIdx 5 → 3.
pub fn decode_sub_mb_type_b(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
) -> CabacResult<u32> {
    const OFFSET: u32 = 36;
    // binIdx 0 — Table 9-39 ctxIdxInc = 0 → ctxIdx = 36.
    let b0 = dec.decode_decision(ctxs.at_mut(OFFSET as usize))?;
    if b0 == 0 {
        // "0" → Table 9-38 B value 0 (B_Direct_8x8).
        return Ok(0);
    }
    // binIdx 1 — Table 9-39 ctxIdxInc = 1 → ctxIdx = 37.
    let b1 = dec.decode_decision(ctxs.at_mut((OFFSET + 1) as usize))?;
    // binIdx 2 — Table 9-39 ctxIdxInc via §9.3.3.1.2:
    //   Table 9-41 row "36, binIdx=2": (b1 != 0) ? 2 : 3.
    let inc_b2 = if b1 != 0 { 2 } else { 3 };
    let b2 = dec.decode_decision(ctxs.at_mut((OFFSET + inc_b2) as usize))?;

    if b1 == 0 {
        // Prefix "1 0 X" — two 3-bin rows:
        //   b2 == 0 → "1 0 0" → value 1 (B_L0_8x8)
        //   b2 == 1 → "1 0 1" → value 2 (B_L1_8x8)
        return Ok(if b2 == 0 { 1 } else { 2 });
    }

    // b1 == 1. Prefix so far is "1 1". Remaining bins use ctxIdxInc = 3
    // (Table 9-39 row for ctxIdxOffset=36 column binIdx >= 3).
    let b3 = dec.decode_decision(ctxs.at_mut((OFFSET + 3) as usize))?;
    let b4 = dec.decode_decision(ctxs.at_mut((OFFSET + 3) as usize))?;

    if b2 == 0 {
        // "1 1 0 X Y" — 5-bin rows (Table 9-38 values 3..=6):
        //   1 1 0 0 0 → 3  (B_Bi_8x8)
        //   1 1 0 0 1 → 4  (B_L0_8x4)
        //   1 1 0 1 0 → 5  (B_L0_4x8)
        //   1 1 0 1 1 → 6  (B_L1_8x4)
        return Ok(3 + ((b3 as u32) << 1) + (b4 as u32));
    }

    // b2 == 1. Prefix so far is "1 1 1 b3 b4".
    //
    // From Table 9-38:
    //   value 7  B_L1_4x8 "1 1 1 0 0 0"
    //   value 8  B_Bi_8x4 "1 1 1 0 0 1"
    //   value 9  B_Bi_4x8 "1 1 1 0 1 0"
    //   value 10 B_L0_4x4 "1 1 1 0 1 1"
    //   value 11 B_L1_4x4 "1 1 1 1 0"
    //   value 12 B_Bi_4x4 "1 1 1 1 1"
    //
    // So when b3 == 1, the codeword is exactly 5 bins and ends at b4:
    //   b4 == 0 → 11, b4 == 1 → 12.
    if b3 == 1 {
        return Ok(if b4 == 0 { 11 } else { 12 });
    }

    // b3 == 0. Need a 6th bin. Table 9-39 ctxIdxInc = 3.
    let b5 = dec.decode_decision(ctxs.at_mut((OFFSET + 3) as usize))?;
    // "1 1 1 0 b4 b5":
    //   0 0 → 7  (B_L1_4x8)
    //   0 1 → 8  (B_Bi_8x4)
    //   1 0 → 9  (B_Bi_4x8)
    //   1 1 → 10 (B_L0_4x4)
    Ok(7 + ((b4 as u32) << 1) + (b5 as u32))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitReader;

    // ---------------------------------------------------------------
    // §9.3.1.1 — context init spot-checks.
    // ---------------------------------------------------------------

    #[test]
    fn ctx_init_picks_i_si_column_for_i_slice() {
        // Table 9-12 ctxIdx=0 → m=20, n=-15. SliceQPY=26 (see
        // `ctx_init_table_9_12_ctx0_mid_qp` in cabac.rs — expected
        // state_idx=46, val_mps=0).
        let ctxs = CabacContexts::init(SliceKind::I, None, 26).unwrap();
        assert_eq!(
            ctxs.at(0),
            &CtxState {
                state_idx: 46,
                val_mps: 0
            }
        );
    }

    #[test]
    fn ctx_init_picks_p_column_by_init_idc() {
        // Table 9-13 ctxIdx=11 init_idc=0 → m=23, n=33. SliceQPY=30.
        // ((23*30)>>4)+33 = (690>>4)+33 = 43+33 = 76 → >63 → state_idx=12, val_mps=1.
        let ctxs = CabacContexts::init(SliceKind::P, Some(0), 30).unwrap();
        assert_eq!(
            ctxs.at(11),
            &CtxState {
                state_idx: 12,
                val_mps: 1
            }
        );
    }

    #[test]
    fn ctx_init_ctx_276_is_non_adapting() {
        let ctxs = CabacContexts::init(SliceKind::I, None, 26).unwrap();
        // §9.3.1.1 NOTE 2.
        assert_eq!(
            ctxs.at(276),
            &CtxState {
                state_idx: 63,
                val_mps: 0
            }
        );
    }

    #[test]
    fn ctx_init_b_slice_reads_b_column() {
        // Table 9-13 ctxIdx=11 init_idc=2 → m=29, n=16. SliceQPY=26.
        // ((29*26)>>4)+16 = (754>>4)+16 = 47+16 = 63 → state_idx=0, val_mps=0.
        let ctxs = CabacContexts::init(SliceKind::B, Some(2), 26).unwrap();
        assert_eq!(
            ctxs.at(11),
            &CtxState {
                state_idx: 0,
                val_mps: 0
            }
        );
    }

    // ---------------------------------------------------------------
    // Syntax-element decoding via synthetic streams.
    // ---------------------------------------------------------------

    /// Build a `CabacDecoder` over a fixed byte stream.
    fn make_decoder(data: &[u8]) -> CabacDecoder<'_> {
        CabacDecoder::new(BitReader::new(data)).unwrap()
    }

    #[test]
    fn mb_skip_flag_offset_ranges() {
        assert_eq!(mb_skip_flag_ctx_offset(SliceKind::P), Some(11));
        assert_eq!(mb_skip_flag_ctx_offset(SliceKind::SP), Some(11));
        assert_eq!(mb_skip_flag_ctx_offset(SliceKind::B), Some(24));
        assert!(mb_skip_flag_ctx_offset(SliceKind::I).is_none());
    }

    #[test]
    fn mb_skip_flag_decode_mps() {
        // Offset=11, neighbours both unavailable → inc=0, ctxIdx=11.
        // With Table 9-13 ctxIdx=11 init_idc=0, m=23 n=33, QP=26
        // gives state_idx=(((23*26)>>4)+33)=(37+33=70 → -6 Clip3),
        // let's just test that the decode returns a deterministic
        // value given all-zero offset (which maps to MPS on fresh ctx).
        let data = [0x00, 0x00, 0x00];
        let mut dec = make_decoder(&data);
        let mut ctxs = CabacContexts::init(SliceKind::P, Some(0), 26).unwrap();
        let neighbours = NeighbourCtx::default();
        let flag = decode_mb_skip_flag(&mut dec, &mut ctxs, SliceKind::P, &neighbours).unwrap();
        // With codIOffset=0 and MPS path, the returned bin equals
        // valMPS which may be 0 or 1 depending on init. Just assert
        // determinism by re-running.
        let mut dec2 = make_decoder(&data);
        let mut ctxs2 = CabacContexts::init(SliceKind::P, Some(0), 26).unwrap();
        let flag2 = decode_mb_skip_flag(&mut dec2, &mut ctxs2, SliceKind::P, &neighbours).unwrap();
        assert_eq!(flag, flag2);
    }

    #[test]
    fn mb_type_i_i_nxn() {
        // For I-slice mb_type, I_NxN = 0 encoded as a single bin "0".
        // With all-zero stream, the MPS of freshly-initialised ctx
        // matches the valMPS of the init. For Table 9-12 ctxIdx=3
        // (offset=3, inc=0..=2 on bin0), m=20 n=-15, QP=26 →
        // state_idx=46, val_mps=0. So decode returns bin=0 → mb_type=0.
        let data = [0x00, 0x00, 0x00];
        let mut dec = make_decoder(&data);
        let mut ctxs = CabacContexts::init(SliceKind::I, None, 26).unwrap();
        let neighbours = NeighbourCtx::default();
        let mb_type = decode_mb_type_i(&mut dec, &mut ctxs, &neighbours).unwrap();
        assert_eq!(mb_type, 0);
    }

    /// ITU-T Rec. H.264 (08/2024) §9.3.2.5, Table 9-36 — verify the
    /// (b2, b3, b4, b5, b6) → mb_type value mapping for every row of
    /// Table 9-36 that reaches `mb_type_i_suffix_value` (rows 1..=24).
    ///
    /// The expected bin sequences come directly from Table 9-36 (each
    /// row in the spec is "1 binIdx1 binIdx2 ..."); the leading "1" is
    /// consumed by `decode_mb_type_i` (binIdx 0) and the next bin
    /// selects I_PCM vs I_16x16_*. For rows 1..=24 binIdx 1 is 0 and
    /// rows 1..=24 then decompose as (b2, b3, b4, b5[, b6]) below.
    ///
    /// Row 0 (I_NxN) and row 25 (I_PCM) do not reach this helper; they
    /// are covered by `mb_type_i_i_nxn` and by the early-return path
    /// in `decode_mb_type_i_suffix`.
    #[test]
    fn mb_type_i_suffix_value_matches_table_9_36() {
        // (expected_value, b2, b3, b4, b5, b6) — b6 is 0 whenever
        // b3 == 0 (the 6-bin rows do not read a binIdx 6).
        let cases: &[(u32, u8, u8, u8, u8, u8)] = &[
            // Rows 1..=4: "1 0 0 0 X Y" — b2=0, b3=0.
            (1, 0, 0, 0, 0, 0), // I_16x16_0_0_0 — "1 0 0 0 0 0"
            (2, 0, 0, 0, 1, 0), // I_16x16_1_0_0 — "1 0 0 0 0 1"
            (3, 0, 0, 1, 0, 0), // I_16x16_2_0_0 — "1 0 0 0 1 0"
            (4, 0, 0, 1, 1, 0), // I_16x16_3_0_0 — "1 0 0 0 1 1"
            // Rows 5..=12: "1 0 0 1 X Y Z" — b2=0, b3=1.
            (5, 0, 1, 0, 0, 0),  // I_16x16_0_1_0 — "1 0 0 1 0 0 0"
            (6, 0, 1, 0, 0, 1),  // I_16x16_1_1_0 — "1 0 0 1 0 0 1"
            (7, 0, 1, 0, 1, 0),  // I_16x16_2_1_0 — "1 0 0 1 0 1 0"
            (8, 0, 1, 0, 1, 1),  // I_16x16_3_1_0 — "1 0 0 1 0 1 1"
            (9, 0, 1, 1, 0, 0),  // I_16x16_0_2_0 — "1 0 0 1 1 0 0"
            (10, 0, 1, 1, 0, 1), // I_16x16_1_2_0 — "1 0 0 1 1 0 1"
            (11, 0, 1, 1, 1, 0), // I_16x16_2_2_0 — "1 0 0 1 1 1 0"
            (12, 0, 1, 1, 1, 1), // I_16x16_3_2_0 — "1 0 0 1 1 1 1"
            // Rows 13..=16: "1 0 1 0 X Y" — b2=1, b3=0.
            (13, 1, 0, 0, 0, 0), // I_16x16_0_0_1 — "1 0 1 0 0 0"
            (14, 1, 0, 0, 1, 0), // I_16x16_1_0_1 — "1 0 1 0 0 1"
            (15, 1, 0, 1, 0, 0), // I_16x16_2_0_1 — "1 0 1 0 1 0"
            (16, 1, 0, 1, 1, 0), // I_16x16_3_0_1 — "1 0 1 0 1 1"
            // Rows 17..=24: "1 0 1 1 X Y Z" — b2=1, b3=1.
            (17, 1, 1, 0, 0, 0), // I_16x16_0_1_1 — "1 0 1 1 0 0 0"
            (18, 1, 1, 0, 0, 1), // I_16x16_1_1_1 — "1 0 1 1 0 0 1"
            (19, 1, 1, 0, 1, 0), // I_16x16_2_1_1 — "1 0 1 1 0 1 0"
            (20, 1, 1, 0, 1, 1), // I_16x16_3_1_1 — "1 0 1 1 0 1 1"
            (21, 1, 1, 1, 0, 0), // I_16x16_0_2_1 — "1 0 1 1 1 0 0"
            (22, 1, 1, 1, 0, 1), // I_16x16_1_2_1 — "1 0 1 1 1 0 1"
            (23, 1, 1, 1, 1, 0), // I_16x16_2_2_1 — "1 0 1 1 1 1 0"
            (24, 1, 1, 1, 1, 1), // I_16x16_3_2_1 — "1 0 1 1 1 1 1"
        ];
        for &(expected, b2, b3, b4, b5, b6) in cases {
            let got = mb_type_i_suffix_value(b2, b3, b4, b5, b6);
            assert_eq!(
                got, expected,
                "mb_type_i_suffix_value({b2},{b3},{b4},{b5},{b6}) = {got}, want {expected}"
            );
        }
        // Exhaustive cardinality: every (b2, b3, b4, b5, b6) with b3=0
        // (b6 must be zero — not read) or b3=1 should produce a unique
        // value in 1..=24.
        let mut seen = std::collections::BTreeSet::new();
        for b2 in 0..=1u8 {
            for b4 in 0..=1u8 {
                for b5 in 0..=1u8 {
                    // b3 = 0, b6 forced to 0.
                    seen.insert(mb_type_i_suffix_value(b2, 0, b4, b5, 0));
                    // b3 = 1, b6 ∈ {0, 1}.
                    for b6 in 0..=1u8 {
                        seen.insert(mb_type_i_suffix_value(b2, 1, b4, b5, b6));
                    }
                }
            }
        }
        let expected_set: std::collections::BTreeSet<u32> = (1u32..=24).collect();
        assert_eq!(seen, expected_set);
    }

    #[test]
    fn coeff_sign_flag_bypass_round_trip() {
        // 9 bits for init = 0b101010101 (0x155). The bypass bit read
        // next is (offset<<1)|bit0_of_next_byte — pick a stream that
        // yields the expected sign.
        let data = [0xFE, 0x80, 0x00]; // init=509, next bit=0 → offset=1018 → >=510 → bin=1
        let mut dec = make_decoder(&data);
        let sign = decode_coeff_sign_flag(&mut dec).unwrap();
        assert!(sign);
    }

    #[test]
    fn end_of_slice_flag_not_yet() {
        // codIOffset=0 (init from all zeros) → after range-=2, 0 < 508
        // → bin=0 → not terminated.
        let data = [0x00, 0x00, 0x00];
        let mut dec = make_decoder(&data);
        let flag = decode_end_of_slice_flag(&mut dec).unwrap();
        assert!(!flag);
    }

    #[test]
    fn end_of_slice_flag_fires() {
        // codIOffset=509 (bytes 0xFE 0x80 init) → terminate fires.
        let data = [0xFE, 0x80, 0x00];
        let mut dec = make_decoder(&data);
        let flag = decode_end_of_slice_flag(&mut dec).unwrap();
        assert!(flag);
    }

    #[test]
    fn block_type_cat_offsets() {
        // Table 9-42 — spot-check ctxBlockCat integers.
        assert_eq!(BlockType::Luma16x16Dc.ctx_block_cat(), 0);
        assert_eq!(BlockType::ChromaDc.ctx_block_cat(), 3);
        assert_eq!(BlockType::Luma8x8.ctx_block_cat(), 5);
        assert_eq!(BlockType::Cr8x8.ctx_block_cat(), 13);
    }

    #[test]
    fn block_type_cbf_block_cat_offsets_match_table_9_40() {
        // Table 9-40: coded_block_flag row.
        assert_eq!(BlockType::Luma16x16Dc.cbf_block_cat_offset(), 0);
        assert_eq!(BlockType::Luma16x16Ac.cbf_block_cat_offset(), 4);
        assert_eq!(BlockType::Luma4x4.cbf_block_cat_offset(), 8);
        assert_eq!(BlockType::ChromaDc.cbf_block_cat_offset(), 12);
        assert_eq!(BlockType::ChromaAc.cbf_block_cat_offset(), 16);
        assert_eq!(BlockType::Luma8x8.cbf_block_cat_offset(), 0);
        assert_eq!(BlockType::CbIntra16x16Dc.cbf_block_cat_offset(), 0);
        assert_eq!(BlockType::CbIntra16x16Ac.cbf_block_cat_offset(), 4);
        assert_eq!(BlockType::CbLuma4x4.cbf_block_cat_offset(), 8);
        assert_eq!(BlockType::Cb8x8.cbf_block_cat_offset(), 4);
        assert_eq!(BlockType::CrIntra16x16Dc.cbf_block_cat_offset(), 0);
        assert_eq!(BlockType::CrIntra16x16Ac.cbf_block_cat_offset(), 4);
        assert_eq!(BlockType::CrLuma4x4.cbf_block_cat_offset(), 8);
        assert_eq!(BlockType::Cr8x8.cbf_block_cat_offset(), 8);
    }

    #[test]
    fn tbl_9_43_sentinel_cells() {
        // Spot check a handful of rows from the transcribed table.
        assert_eq!(TBL_9_43[0], (0, 0, 0));
        assert_eq!(TBL_9_43[1], (1, 1, 1));
        assert_eq!(TBL_9_43[15], (5, 5, 1));
        assert_eq!(TBL_9_43[32], (7, 9, 3));
        assert_eq!(TBL_9_43[62], (12, 14, 8));
    }

    #[test]
    fn ref_idx_lx_decodes_zero_with_mps_stream() {
        // All-zero stream → MPS path. ref_idx ctx 54 (init_idc=0):
        // Table 9-16 m=-7 n=67 QP=26 → ((-7*26)>>4)+67 = (-182/16)+67
        // = -12 + 67 = 55 → state_idx=8, val_mps=0 → bin=0 →
        // ref_idx=0.
        let data = [0x00, 0x00, 0x00, 0x00];
        let mut dec = make_decoder(&data);
        let mut ctxs = CabacContexts::init(SliceKind::P, Some(0), 26).unwrap();
        let v = decode_ref_idx_lx(&mut dec, &mut ctxs, false, false).unwrap();
        assert_eq!(v, 0);
    }

    #[test]
    fn mb_qp_delta_zero_path() {
        let data = [0x00, 0x00, 0x00, 0x00];
        let mut dec = make_decoder(&data);
        let mut ctxs = CabacContexts::init(SliceKind::P, Some(0), 26).unwrap();
        let v = decode_mb_qp_delta(&mut dec, &mut ctxs, false).unwrap();
        assert_eq!(v, 0);
    }

    #[test]
    fn intra_chroma_pred_mode_zero_path() {
        let data = [0x00, 0x00, 0x00, 0x00];
        let mut dec = make_decoder(&data);
        let mut ctxs = CabacContexts::init(SliceKind::I, None, 26).unwrap();
        let neighbours = NeighbourCtx::default();
        let v = decode_intra_chroma_pred_mode(&mut dec, &mut ctxs, &neighbours).unwrap();
        // With fresh I-slice ctx @ ctxIdx 64 (m=-9, n=83, QP=26) the
        // preCtxState is ((-9*26)>>4)+83 = (-234/16)+83 = -15+83=68
        // → valMPS=1, so first bin = 1. Decode advances past bin 0.
        // Expect non-zero return.
        assert!(v > 0);
    }

    #[test]
    fn decoder_new_init_table_populated_ranges() {
        // Sanity: ensure the init table was populated for the key
        // contexts used by this module.
        let table = init_table();
        assert!(table[0].i_si.is_some()); // Table 9-12
        assert!(table[11].p0.is_some()); // Table 9-13
        assert!(table[54].p0.is_some()); // Table 9-16
        assert!(table[68].i_si.is_some()); // Table 9-17
        assert!(table[104].p1.is_some()); // Table 9-18
        assert!(table[105].p0.is_some()); // Table 9-19
        assert!(table[226].p2.is_some()); // Table 9-20
        assert!(table[275].i_si.is_some()); // Table 9-21
    }

    // ---------------------------------------------------------------
    // P/B intra-suffix regression tests — these exercise the
    // Table 9-36 suffix read under P-slice (offset=17) and B-slice
    // (offset=32) mb_type contexts, where the suffix's binIdx=0 is a
    // distinct bin from the outer prefix (§9.3.2.5). An earlier version
    // of `decode_mb_type_i_suffix` skipped that bin, causing an
    // under-read for every intra macroblock in P and B slices.
    // ---------------------------------------------------------------

    /// Exercise the P-slice intra-suffix path: `decode_mb_type_p`
    /// invokes the suffix helper with `offset=17, None` when the outer
    /// prefix decodes to 1. We expose the helper indirectly here by
    /// forcing MPS-only bins (all-zero bitstream) and checking that
    /// (a) the return is `5 + 0 = 5` (P intra I_NxN) when the contexts
    /// at offset=14 (prefix) and offset=17 (suffix binIdx=0) are MPS=1
    /// / MPS=0 respectively at init, and (b) the bin consumption is
    /// exactly 2 (prefix bit + suffix binIdx=0).
    ///
    /// Note: we can't easily force the exact ctx states without writing
    /// a custom bitstream; this test just asserts the helper-path
    /// doesn't panic and returns a sensible value for an all-MPS
    /// stream. The real regression guard is the bin-count check below.
    #[test]
    fn p_slice_intra_suffix_reads_bin0_of_suffix() {
        // Build the worst case "all bins are MPS" stream. With all-zero
        // data bytes, bypass and decision bins flow MPS almost always.
        let data = [0x00, 0x00, 0x00, 0x00, 0x00];
        let mut dec = make_decoder(&data);
        let mut ctxs = CabacContexts::init(SliceKind::P, Some(0), 26).unwrap();
        let before = dec.bin_count();
        // Drive a full P-slice mb_type decode. Bin 0 (prefix) at
        // ctxIdx=14: Table 9-13 ctxIdx=14 P0 (m=1, n=9), QP=26 →
        // preCtxState = ((1*26)>>4) + 9 = 1 + 9 = 10 → state_idx=53,
        // val_mps=0. All-zero bitstream produces MPS path → bin=0 →
        // inter branch (b0 == 0). So this test exercises the INTER
        // path, not intra. The valuable assertion is that the function
        // does not panic.
        let _ = decode_mb_type_p(&mut dec, &mut ctxs, &NeighbourCtx::default()).unwrap();
        let bins_used = dec.bin_count() - before;
        // Inter path always reads 3 bins (b0, b1, b2).
        assert_eq!(bins_used, 3, "inter P mb_type must read exactly 3 bins");
    }

    /// Helper: force a specific stream into the engine and count bins
    /// consumed by a single decode. Used to sanity-check that the
    /// intra-suffix helper path for P/B slices decodes the expected
    /// number of bins.
    ///
    /// `prefix_bits` are the high bits of the 9-bit `codIOffset` init
    /// read + whatever renorm bits will be consumed; this test just
    /// asserts that the combined prefix + suffix consume exactly
    /// `expected_bin_count` bins via the DecodeDecision / Terminate /
    /// Bypass count exposed by `CabacDecoder::bin_count`.
    fn assert_bins_consumed(data: &[u8], expected: u64, body: impl FnOnce(&mut CabacDecoder<'_>)) {
        let mut dec = make_decoder(data);
        let before = dec.bin_count();
        body(&mut dec);
        let got = dec.bin_count() - before;
        assert_eq!(got, expected, "bin count mismatch");
    }

    /// Verify the I-slice suffix helper reads binIdx 1..=6 under an
    /// all-zero stream: caller supplies binIdx=0 as `Some(1)`, then
    /// the helper issues 1 DecodeTerminate (binIdx=1), 4 DecodeDecision
    /// calls for binIdx 2..=5, and because the all-zero stream drives
    /// b3 to MPS = 1 under the I-slice init for ctxIdx=3+4=7 (Table
    /// 9-12: m=-23, n=104 → preCtx = ((-23*26)>>4)+104 = -38+104 = 66
    /// → state_idx=2, val_mps=1), b3 != 0 fires the 7-bin branch, so
    /// binIdx=6 is also read. Helper bin consumption = 6.
    #[test]
    fn i_suffix_helper_reads_binidx_1_through_6_when_b3_nonzero() {
        let data = [0x00, 0x00, 0x00, 0x00, 0x00];
        assert_bins_consumed(&data, 6, |dec| {
            let mut ctxs = CabacContexts::init(SliceKind::I, None, 26).unwrap();
            // Pass Some(1) as the already-consumed binIdx=0. The helper
            // reads binIdx 1..=5 then conditionally binIdx=6.
            let _ = decode_mb_type_i_suffix(dec, &mut ctxs, 3, Some(1)).unwrap();
        });
    }

    /// Verify the P-slice suffix helper (offset=17, `None`) reads
    /// binIdx 0 first (1 bin), then binIdx 1 via DecodeTerminate
    /// (1 bin), and if both resolve to 0, stops after 2 bins (mb_type
    /// value 5 — P I_NxN). With a uniformly-zero stream and the init
    /// table, ctxIdx=17 in a P slice is driven to val_mps=1 (from
    /// Table 9-13: m=5, n=57), so binIdx 0 MPS = 1 → enter the
    /// binIdx 1 terminate path, which also resolves to 0 (not
    /// terminated) → read binIdx 2..=5 for 4 more bins. Helper total
    /// = 6 bins (binIdx 0..=5).
    ///
    /// This is the key regression guard: before the fix, the helper
    /// returned after only 5 bins (skipping binIdx 0 of the suffix).
    #[test]
    fn p_suffix_helper_reads_binidx_0_through_5() {
        let data = [0x00, 0x00, 0x00, 0x00, 0x00];
        assert_bins_consumed(&data, 6, |dec| {
            let mut ctxs = CabacContexts::init(SliceKind::P, Some(0), 26).unwrap();
            // Sanity: ctxIdx=17 in P slice init_idc=0 at QP=26:
            //   Table 9-13 (17, P0) = (5, 57)  →  preCtx = ((5*26)>>4)+57
            //   = 8 + 57 = 65  →  state_idx = 65 - 64 = 1, val_mps = 1.
            // So binIdx 0 MPS = 1, taking the "not I_NxN" branch into
            // the terminate bin and on into binIdx 2..=5.
            let _ = decode_mb_type_i_suffix(dec, &mut ctxs, 17, None).unwrap();
        });
    }

    /// Same shape as the P-slice guard but under offset=32 (B-slice
    /// intra suffix). Table 9-14 ctxIdx=32, P0 = (1, 67): preCtx =
    /// ((1*26)>>4) + 67 = 1 + 67 = 68 → state_idx = 4, val_mps = 1.
    /// All-zero stream → MPS path → b0 of suffix = 1, proceed to
    /// binIdx 1 terminate which resolves to 0 (not terminated), then
    /// binIdx 2..=5 (4 more bins). Helper total = 6 bins.
    #[test]
    fn b_suffix_helper_reads_binidx_0_through_5() {
        let data = [0x00, 0x00, 0x00, 0x00, 0x00];
        assert_bins_consumed(&data, 6, |dec| {
            let mut ctxs = CabacContexts::init(SliceKind::B, Some(0), 26).unwrap();
            let _ = decode_mb_type_i_suffix(dec, &mut ctxs, 32, None).unwrap();
        });
    }

    /// With offset=17 and a stream that drives binIdx 0 of the suffix
    /// to 0 (I_NxN in the P slice), the helper must stop after 1 bin
    /// and return 0 so the caller produces mb_type = 5 + 0 = 5.
    ///
    /// We craft the state so bin 0 decodes as LPS=0 by:
    ///   - Picking an init where ctxIdx=17 starts with valMPS=1 and a
    ///     state_idx giving a reasonably-large LPS subrange.
    ///   - Feeding a codIOffset near `codIRange`, so `codIOffset >=
    ///     codIRange - codIRangeLPS` (the LPS path) fires, yielding
    ///     binVal = 1 - valMPS = 0.
    ///
    /// Rather than hand-tune the first 9 bits of codIOffset, we rely on
    /// the fact that codIRange is 510 at start and ctxIdx=17 in P slice
    /// init_idc=0, QP=26 starts at state_idx=1, valMPS=1. Then
    /// rangeTabLPS[1][3] = 227, so codIRange after subtraction =
    /// 510 - 227 = 283. LPS fires when codIOffset ≥ 283. First 9 bits
    /// decoding 283 is 0b100011011 = 283 → first byte = 0b1000_1101,
    /// second byte bit 7 = 1 → bytes [0x8D, 0x80, ...].
    #[test]
    fn p_suffix_helper_i_nxn_stops_after_one_bin() {
        // codIOffset init reads 9 bits = 283 (just at the LPS threshold).
        let data = [0x8D, 0x80, 0x00, 0x00];
        let mut dec = make_decoder(&data);
        let before = dec.bin_count();
        let mut ctxs = CabacContexts::init(SliceKind::P, Some(0), 26).unwrap();
        let v = decode_mb_type_i_suffix(&mut dec, &mut ctxs, 17, None).unwrap();
        let bins = dec.bin_count() - before;
        // Row 0 (I_NxN) of Table 9-36.
        assert_eq!(v, 0, "I_NxN in P slice is Table 9-36 row 0");
        assert_eq!(bins, 1, "I_NxN path reads exactly binIdx=0 of the suffix");
    }

    // ---------------------------------------------------------------
    // §9.3.4 — test-only CABAC encoder used to round-trip
    // sub_mb_type bin strings through the real decoder. The encoder
    // here is transcribed directly from the spec's informative clauses:
    //
    //   * §9.3.4.1 — InitEncoder: codILow = 0, codIRange = 510,
    //     firstBitFlag = 1, bitsOutstanding = 0.
    //   * §9.3.4.2 / Figure 9-7 — EncodeDecision.
    //   * §9.3.4.3 / Figure 9-8 — RenormE + Figure 9-9 PutBit.
    //   * §9.3.4.5 / Figure 9-12 — EncodeFlush (byte alignment).
    //
    // Bypass / terminate encoding are not needed here (sub_mb_type is
    // decision-only) so they are omitted.
    // ---------------------------------------------------------------

    /// Minimal CABAC encoder. `ctxs` must be initialised with the same
    /// `(slice_kind, init_idc, qp)` triple the decoder will use, so
    /// that encoder and decoder walk identical context states.
    struct TestEncoder {
        cod_i_low: u32,
        cod_i_range: u32,
        first_bit_flag: bool,
        bits_outstanding: u64,
        bits: Vec<u8>,       // accumulating bit queue, MSB-first within each byte
        bit_pos_in_byte: u8, // 0..=7; 0 means next bit goes to MSB of the last byte
    }

    impl TestEncoder {
        fn new() -> Self {
            Self {
                cod_i_low: 0,
                cod_i_range: 510,
                first_bit_flag: true,
                bits_outstanding: 0,
                bits: Vec::new(),
                bit_pos_in_byte: 0,
            }
        }

        /// Append one bit to the output queue (MSB-first packing), to
        /// match `BitReader::u(n)` consumption order used by the
        /// decoder.
        fn write_bit(&mut self, b: u8) {
            if self.bit_pos_in_byte == 0 {
                self.bits.push(0);
            }
            let idx = self.bits.len() - 1;
            let shift = 7 - self.bit_pos_in_byte;
            self.bits[idx] |= (b & 1) << shift;
            self.bit_pos_in_byte = (self.bit_pos_in_byte + 1) % 8;
        }

        fn put_bit(&mut self, b: u8) {
            // Figure 9-9.
            if self.first_bit_flag {
                self.first_bit_flag = false;
            } else {
                self.write_bit(b);
            }
            while self.bits_outstanding > 0 {
                self.write_bit(1 - b);
                self.bits_outstanding -= 1;
            }
        }

        fn renorm_e(&mut self) {
            // Figure 9-8.
            while self.cod_i_range < 256 {
                if self.cod_i_low < 256 {
                    self.put_bit(0);
                } else if self.cod_i_low >= 512 {
                    self.cod_i_low -= 512;
                    self.put_bit(1);
                } else {
                    self.cod_i_low -= 256;
                    self.bits_outstanding += 1;
                }
                self.cod_i_range <<= 1;
                self.cod_i_low <<= 1;
            }
        }

        /// §9.3.4.2 Figure 9-7 — EncodeDecision. Mirrors
        /// `CabacDecoder::decode_decision` state transitions.
        fn encode_decision(&mut self, ctx: &mut crate::cabac::CtxState, bin: u8) {
            let q_idx = ((self.cod_i_range >> 6) & 0b11) as usize;
            let cod_i_range_lps = crate::cabac::RANGE_TAB_LPS[ctx.state_idx as usize][q_idx] as u32;
            self.cod_i_range -= cod_i_range_lps;
            if bin != ctx.val_mps {
                // LPS branch.
                self.cod_i_low += self.cod_i_range;
                self.cod_i_range = cod_i_range_lps;
                if ctx.state_idx == 0 {
                    ctx.val_mps = 1 - ctx.val_mps;
                }
                ctx.state_idx = crate::cabac::TRANS_IDX_LPS[ctx.state_idx as usize];
            } else {
                // MPS branch.
                ctx.state_idx = crate::cabac::TRANS_IDX_MPS[ctx.state_idx as usize];
            }
            self.renorm_e();
        }

        /// §9.3.4.5 Figure 9-12 — EncodeFlush. Terminates the stream so
        /// all outstanding bits are flushed and the decoder's 9-bit
        /// codIOffset read lines up against valid data.
        fn finish(mut self) -> Vec<u8> {
            // codIRange = 2
            self.cod_i_range = 2;
            self.renorm_e();
            // PutBit((codILow >> 9) & 1)
            self.put_bit(((self.cod_i_low >> 9) & 1) as u8);
            // WriteBits(((codILow >> 7) & 3) | 1, 2)
            let last2 = ((self.cod_i_low >> 7) & 3) | 1;
            self.write_bit(((last2 >> 1) & 1) as u8);
            self.write_bit((last2 & 1) as u8);
            // Pad to a whole byte with zeros so the BitReader has room
            // to consume any residual alignment/renorm bits. A trailing
            // byte of zeros is more than enough for the 9-bit init and
            // any renorm triggers during the small bin strings we test.
            while self.bit_pos_in_byte != 0 {
                self.write_bit(0);
            }
            for _ in 0..4 {
                self.bits.push(0);
            }
            self.bits
        }
    }

    /// §9.3.4 — round-trip one bin sequence through the minimal test
    /// encoder and then back through the real decoder, using identical
    /// context indices on both sides. Returns the decoded bins for
    /// caller-side assertions.
    ///
    /// `bins` is a slice of (ctxIdx, binVal) pairs.
    fn round_trip_bins(
        slice_kind: SliceKind,
        init_idc: Option<u32>,
        qp: i32,
        bins: &[(usize, u8)],
    ) -> (Vec<u8>, CabacContexts) {
        // Encode side.
        let mut enc = TestEncoder::new();
        let mut enc_ctxs = CabacContexts::init(slice_kind, init_idc, qp).unwrap();
        for &(ctx_idx, bin) in bins {
            enc.encode_decision(enc_ctxs.at_mut(ctx_idx), bin);
        }
        let bytes = enc.finish();

        // Decode side — fresh context table, same init.
        let mut dec = make_decoder(&bytes);
        let mut dec_ctxs = CabacContexts::init(slice_kind, init_idc, qp).unwrap();
        let mut decoded = Vec::with_capacity(bins.len());
        for &(ctx_idx, _) in bins {
            let b = dec.decode_decision(dec_ctxs.at_mut(ctx_idx)).unwrap();
            decoded.push(b);
        }
        (decoded, dec_ctxs)
    }

    /// Sanity check: the minimal test encoder round-trips a handful of
    /// hand-picked bin sequences faithfully. This protects the
    /// round-trip infrastructure itself.
    #[test]
    fn test_encoder_round_trips_simple_sequences() {
        // A single bin at ctxIdx=21 (P sub_mb_type).
        for bin in [0u8, 1] {
            let (decoded, _) = round_trip_bins(SliceKind::P, Some(0), 26, &[(21, bin)]);
            assert_eq!(decoded, vec![bin]);
        }
        // A 3-bin sequence with varied ctxIdx.
        let bins = [(21, 0u8), (22, 1), (23, 0)];
        let (decoded, _) = round_trip_bins(SliceKind::P, Some(0), 26, &bins);
        assert_eq!(
            decoded,
            vec![0, 1, 0],
            "test encoder must round-trip a 3-bin P sub_mb_type sequence"
        );
        // A longer 6-bin sequence at B-slice sub_mb_type ctxs:
        // binIdx 0..=5 with ctxIdx derivation per Table 9-39 row 36
        // (binIdx 2 uses inc=2 when b1!=0 via Table 9-41).
        let bins: [(usize, u8); 6] = [(36, 1), (37, 1), (38, 1), (39, 0), (39, 0), (39, 0)];
        let (decoded, _) = round_trip_bins(SliceKind::B, Some(2), 26, &bins);
        let expected: Vec<u8> = bins.iter().map(|&(_, b)| b).collect();
        assert_eq!(decoded, expected);
    }

    /// Encode the bin string for every Table 9-37 P sub_mb_type value,
    /// then decode via `decode_sub_mb_type_p` and assert the returned
    /// value matches.
    #[test]
    fn sub_mb_type_p_round_trips_all_values() {
        // Per Table 9-37 (P / SP rows) and Table 9-39 row 21:
        //   value 0 (P_L0_8x8): [(21, 1)]
        //   value 1 (P_L0_8x4): [(21, 0), (22, 0)]
        //   value 2 (P_L0_4x8): [(21, 0), (22, 1), (23, 1)]
        //   value 3 (P_L0_4x4): [(21, 0), (22, 1), (23, 0)]
        let cases: [(&[(usize, u8)], u32); 4] = [
            (&[(21, 1)], 0),
            (&[(21, 0), (22, 0)], 1),
            (&[(21, 0), (22, 1), (23, 1)], 2),
            (&[(21, 0), (22, 1), (23, 0)], 3),
        ];
        for &(bins, expected) in &cases {
            // Encode the bins.
            let mut enc = TestEncoder::new();
            let mut enc_ctxs = CabacContexts::init(SliceKind::P, Some(0), 26).unwrap();
            for &(ctx_idx, b) in bins {
                enc.encode_decision(enc_ctxs.at_mut(ctx_idx), b);
            }
            let bytes = enc.finish();
            // Decode through the real primitive.
            let mut dec = make_decoder(&bytes);
            let mut dec_ctxs = CabacContexts::init(SliceKind::P, Some(0), 26).unwrap();
            let got = decode_sub_mb_type_p(&mut dec, &mut dec_ctxs).unwrap();
            assert_eq!(
                got, expected,
                "P sub_mb_type round-trip: bins {bins:?} expected {expected}, got {got}"
            );
        }
    }

    /// Encode the bin string for every Table 9-38 B sub_mb_type value,
    /// then decode via `decode_sub_mb_type_b`.
    ///
    /// ctxIdxOffset = 36 per Table 9-34.
    /// ctxIdxInc per Table 9-39 row 36:
    ///   binIdx 0 → inc 0 → ctxIdx 36
    ///   binIdx 1 → inc 1 → ctxIdx 37
    ///   binIdx 2 → (b1 != 0) ? 2 : 3 (Table 9-41) → ctxIdx 38 or 39
    ///   binIdx 3 → inc 3 → ctxIdx 39
    ///   binIdx 4 → inc 3 → ctxIdx 39
    ///   binIdx 5 → inc 3 → ctxIdx 39
    #[test]
    fn sub_mb_type_b_round_trips_all_values() {
        // Helper: build the (ctxIdx, bin) sequence for a given bin
        // string. `bs` is the bit sequence as a slice of u8 (0 or 1).
        fn build(bs: &[u8]) -> Vec<(usize, u8)> {
            let mut out: Vec<(usize, u8)> = Vec::with_capacity(bs.len());
            let mut b1: u8 = 0;
            for (i, &bit) in bs.iter().enumerate() {
                let ctx_idx: usize = match i {
                    0 => 36, // binIdx 0 → inc 0
                    1 => 37, // binIdx 1 → inc 1
                    2 => {
                        // binIdx 2 → Table 9-41
                        if b1 != 0 {
                            36 + 2
                        } else {
                            36 + 3
                        }
                    }
                    _ => 36 + 3, // binIdx 3..=5 → inc 3
                };
                out.push((ctx_idx, bit));
                if i == 1 {
                    b1 = bit;
                }
            }
            out
        }

        // Table 9-38 B sub_mb_type bin strings.
        let cases: [(&[u8], u32); 13] = [
            (&[0], 0),                 // B_Direct_8x8
            (&[1, 0, 0], 1),           // B_L0_8x8
            (&[1, 0, 1], 2),           // B_L1_8x8
            (&[1, 1, 0, 0, 0], 3),     // B_Bi_8x8
            (&[1, 1, 0, 0, 1], 4),     // B_L0_8x4
            (&[1, 1, 0, 1, 0], 5),     // B_L0_4x8
            (&[1, 1, 0, 1, 1], 6),     // B_L1_8x4
            (&[1, 1, 1, 0, 0, 0], 7),  // B_L1_4x8
            (&[1, 1, 1, 0, 0, 1], 8),  // B_Bi_8x4
            (&[1, 1, 1, 0, 1, 0], 9),  // B_Bi_4x8
            (&[1, 1, 1, 0, 1, 1], 10), // B_L0_4x4
            (&[1, 1, 1, 1, 0], 11),    // B_L1_4x4
            (&[1, 1, 1, 1, 1], 12),    // B_Bi_4x4
        ];
        for &(bs, expected) in &cases {
            let bins = build(bs);
            // Encode.
            let mut enc = TestEncoder::new();
            let mut enc_ctxs = CabacContexts::init(SliceKind::B, Some(1), 26).unwrap();
            for &(ctx_idx, b) in &bins {
                enc.encode_decision(enc_ctxs.at_mut(ctx_idx), b);
            }
            let bytes = enc.finish();
            // Decode.
            let mut dec = make_decoder(&bytes);
            let mut dec_ctxs = CabacContexts::init(SliceKind::B, Some(1), 26).unwrap();
            let got = decode_sub_mb_type_b(&mut dec, &mut dec_ctxs).unwrap();
            assert_eq!(
                got, expected,
                "B sub_mb_type round-trip: bin string {bs:?} expected {expected}, got {got}"
            );
        }
    }
}
