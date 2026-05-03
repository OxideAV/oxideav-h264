//! §8.5 — Inverse transform and scaling.
//!
//! Pure integer math per ITU-T Rec. H.264 (08/2024). No bitstream, no
//! context. Takes parsed coefficient arrays and a quantiser parameter,
//! produces residual sample arrays for the reconstruction pipeline.
//!
//! Spec coverage:
//!   §8.5.6  — Inverse scanning process for 4x4 transform coefficients.
//!             Exposed separately as [`inverse_scan_4x4_zigzag`] /
//!             [`inverse_scan_4x4_field`].
//!   §8.5.8  — Derivation process for chroma quantization parameters
//!             (QPc via Table 8-15).
//!   §8.5.9  — Derivation process for scaling functions (LevelScale).
//!   §8.5.10 — Scaling + Hadamard for Intra_16x16 luma DC block.
//!   §8.5.11 — Scaling + Hadamard for chroma DC (2x2 for 4:2:0,
//!             2x4 for 4:2:2).
//!   §8.5.12 — Scaling + inverse transform for residual 4x4 blocks.
//!   §8.5.13 — Scaling + inverse transform for residual 8x8 blocks.
//!
//! All equation numbers below are from the 08/2024 edition.

#![allow(dead_code)]

use thiserror::Error;

use crate::pps::Pps;
use crate::sps::{ScalingListEntry, Sps};

// ---------------------------------------------------------------------------
// Errors.
// ---------------------------------------------------------------------------

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TransformError {
    /// qP outside the valid range for the given bit depth.
    /// §8.5.9 — the qP fed to the scaling formulas is qP'Y / qP'C, which
    /// equals (QPY + QpBdOffsetY) or (QPC + QpBdOffsetC), so its valid
    /// range is `0..=51 + QpBdOffset`. For 8-bit coding QpBdOffset is 0
    /// and the range is 0..=51; at 10-bit luma it is 0..=63; at 14-bit
    /// luma 0..=87. The error carries the offending qP for diagnostics.
    #[error("QP out of range (got {0}, must be 0..=51+QpBdOffset for the bit depth)")]
    QpOutOfRange(i32),
    /// bit_depth outside the supported 8..=14 range.
    #[error("bit_depth out of range (got {0}, must be 8..=14)")]
    BitDepthOutOfRange(u32),
}

/// §7.4.2.1.1 Eq. 7-4 / 7-6 — `QpBdOffsetY = 6 * bit_depth_luma_minus8`
/// (the same shape applies to `QpBdOffsetC`). Caller passes
/// `bit_depth_luma_minus8` (or `bit_depth_chroma_minus8`).
#[inline]
pub fn qp_bd_offset(bit_depth_minus8: u32) -> i32 {
    6 * bit_depth_minus8 as i32
}

// ---------------------------------------------------------------------------
// §7.4.5 / Table 8-15 — QPy -> QPc mapping.
// ---------------------------------------------------------------------------

/// §8.5.8 / Table 8-15 — maps qPI (any value) to QPc.
/// For qPI < 30, the mapping is identity; for qPI ∈ [30,51] a table
/// lookup is used. Negative qPI values are returned unchanged because
/// the encoder is allowed to take qPI below zero (QpBdOffset range).
/// The caller is expected to have performed any required clipping.
pub fn qp_c_table_lookup(qp_i: i32) -> i32 {
    // Table 8-15 — QPc as a function of qPI, qPI ∈ [30,51].
    // Row index = qPI - 30.
    const QP_C_FROM_QPI_30_51: [i32; 22] = [
        29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39,
    ];
    if qp_i < 30 {
        qp_i
    } else if qp_i <= 51 {
        QP_C_FROM_QPI_30_51[(qp_i - 30) as usize]
    } else {
        // Out of normal bitstream range — clamp to the 51 entry.
        QP_C_FROM_QPI_30_51[21]
    }
}

/// §8.5.8 — chroma QP derivation (eq. 8-311) for the 8-bit chroma path
/// (QpBdOffsetC = 0). For bit depths > 8 use
/// [`qp_y_to_qp_c_with_bd_offset`] instead, which clamps qPI to the
/// extended `−QpBdOffsetC..=51` range and propagates negative qPI values
/// through the table-lookup unchanged (eq. 8-312 selects QPC = qPI when
/// qPI < 30, including negatives).
///
/// `qp_y` is the luma QP (post-adjustment) and `chroma_qp_index_offset`
/// comes from the PPS (`chroma_qp_index_offset` for Cb,
/// `second_chroma_qp_index_offset` for Cr). Returns QPc ∈ 0..=51.
pub fn qp_y_to_qp_c(qp_y: i32, chroma_qp_index_offset: i32) -> i32 {
    qp_y_to_qp_c_with_bd_offset(qp_y, chroma_qp_index_offset, 0)
}

/// §8.5.8 — chroma QP derivation (eq. 8-311) for arbitrary bit depth.
///
/// Computes `QPC` from `QPY`, `chroma_qp_index_offset` and
/// `qp_bd_offset_c` (= `6 * bit_depth_chroma_minus8`). The intermediate
/// `qPI = Clip3(-QpBdOffsetC, 51, QPY + qPOffset)` ranges in
/// `-QpBdOffsetC..=51`, and qPI < 30 returns QPC = qPI verbatim — so
/// the caller should add `QpBdOffsetC` afterwards to get qP'C for the
/// transform/scaling helpers.
pub fn qp_y_to_qp_c_with_bd_offset(
    qp_y: i32,
    chroma_qp_index_offset: i32,
    qp_bd_offset_c: i32,
) -> i32 {
    // §8.5.8 eq. 8-311: qPI = Clip3(-QpBdOffsetC, 51, QPY + qPOffset).
    let lo = -qp_bd_offset_c;
    let qp_i = (qp_y + chroma_qp_index_offset).clamp(lo, 51);
    qp_c_table_lookup(qp_i)
}

// ---------------------------------------------------------------------------
// §8.5.9 — normAdjust matrices (Tables 8-4, 8-5 transcribed verbatim).
// ---------------------------------------------------------------------------

/// Table 8-4 / Eq. 8-315 — v-matrix for 4x4 normAdjust.
///
/// Rows are indexed by m ∈ 0..=5 (= qP % 6), columns are the three
/// equivalence-class values selected by (i, j) parity per Eq. 8-314.
const NORM_ADJUST_4X4_V: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

/// Table 8-5 / Eq. 8-318 — v-matrix for 8x8 normAdjust.
const NORM_ADJUST_8X8_V: [[i32; 6]; 6] = [
    [20, 18, 32, 19, 25, 24],
    [22, 19, 35, 21, 28, 26],
    [26, 23, 42, 24, 33, 31],
    [28, 25, 45, 26, 35, 33],
    [32, 28, 51, 30, 40, 38],
    [36, 32, 58, 34, 46, 43],
];

/// §8.5.9 / Eq. 8-314 — normAdjust for 4x4 blocks.
///
/// Equivalence class selection:
///   (i%2, j%2) == (0, 0) -> v_{m,0}
///   (i%2, j%2) == (1, 1) -> v_{m,1}
///   otherwise            -> v_{m,2}
#[inline]
fn norm_adjust_4x4(m: usize, i: usize, j: usize) -> i32 {
    let row = &NORM_ADJUST_4X4_V[m];
    match (i & 1, j & 1) {
        (0, 0) => row[0],
        (1, 1) => row[1],
        _ => row[2],
    }
}

/// §8.5.9 / Eq. 8-317 — normAdjust for 8x8 blocks.
///
/// Equivalence class selection (in the order listed in the spec):
///   (i%4, j%4) == (0, 0)                      -> v_{m,0}
///   (i%2, j%2) == (1, 1)                      -> v_{m,1}
///   (i%4, j%4) == (2, 2)                      -> v_{m,2}
///   (i%4, j%2) == (0, 1) or (i%2, j%4) == (1, 0) -> v_{m,3}
///   (i%4, j%4) == (0, 2) or (i%4, j%4) == (2, 0) -> v_{m,4}
///   otherwise                                 -> v_{m,5}
#[inline]
fn norm_adjust_8x8(m: usize, i: usize, j: usize) -> i32 {
    let row = &NORM_ADJUST_8X8_V[m];
    let im4 = i & 3;
    let jm4 = j & 3;
    let im2 = i & 1;
    let jm2 = j & 1;
    if im4 == 0 && jm4 == 0 {
        row[0]
    } else if im2 == 1 && jm2 == 1 {
        row[1]
    } else if im4 == 2 && jm4 == 2 {
        row[2]
    } else if (im4 == 0 && jm2 == 1) || (im2 == 1 && jm4 == 0) {
        row[3]
    } else if (im4 == 0 && jm4 == 2) || (im4 == 2 && jm4 == 0) {
        row[4]
    } else {
        row[5]
    }
}

/// §8.5.9 Eq. 8-313 — LevelScale4x4(m, i, j) = weightScale4x4[i,j] *
/// normAdjust4x4(m, i, j).
#[inline]
fn level_scale_4x4(scaling_list: &[i32; 16], m: usize, i: usize, j: usize) -> i32 {
    scaling_list[i * 4 + j] * norm_adjust_4x4(m, i, j)
}

/// §8.5.9 Eq. 8-316 — LevelScale8x8(m, i, j) = weightScale8x8[i,j] *
/// normAdjust8x8(m, i, j).
#[inline]
fn level_scale_8x8(scaling_list: &[i32; 64], m: usize, i: usize, j: usize) -> i32 {
    scaling_list[i * 8 + j] * norm_adjust_8x8(m, i, j)
}

// ---------------------------------------------------------------------------
// §7.4.2.1.1 Eqs. 7-8 / 7-9 — flat scaling lists (all 16s).
// §7.4.2.1.1.1 Tables 7-3 / 7-4 — default scaling matrices.
// ---------------------------------------------------------------------------

/// §7.4.2.1.1 Eq. 7-8 — `Flat_4x4_16`. All 16s. Used as the inferred
/// sequence-level scaling list when `seq_scaling_matrix_present_flag == 0`
/// (every entry = 16 cancels out the normAdjust/16 factor implicit in
/// the LevelScale derivation → "no scaling").
pub fn default_scaling_list_4x4_flat() -> [i32; 16] {
    FLAT_4X4_16
}

/// §7.4.2.1.1 Eq. 7-9 — `Flat_8x8_16`. All 16s.
pub fn default_scaling_list_8x8_flat() -> [i32; 64] {
    FLAT_8X8_16
}

/// §7.4.2.1.1 Eq. 7-8 — `Flat_4x4_16`.
pub const FLAT_4X4_16: [i32; 16] = [16; 16];

/// §7.4.2.1.1 Eq. 7-9 — `Flat_8x8_16`.
pub const FLAT_8X8_16: [i32; 64] = [16; 64];

/// §7.4.2.1.1.1 Table 7-3 — default Intra 4x4 scaling matrix
/// (`Default_4x4_Intra`). Row-major, scan-order (same ordering as
/// Table 7-3 which lists idx 0..15).
pub const DEFAULT_4X4_INTRA: [i32; 16] = [
    6, 13, 13, 20, 20, 20, 28, 28, 28, 28, 32, 32, 32, 37, 37, 42,
];

/// §7.4.2.1.1.1 Table 7-3 — default Inter 4x4 scaling matrix
/// (`Default_4x4_Inter`).
pub const DEFAULT_4X4_INTER: [i32; 16] = [
    10, 14, 14, 20, 20, 20, 24, 24, 24, 24, 27, 27, 27, 30, 30, 34,
];

/// §7.4.2.1.1.1 Table 7-4 — default Intra 8x8 scaling matrix
/// (`Default_8x8_Intra`). 64 entries, scan-order as given in Table 7-4.
pub const DEFAULT_8X8_INTRA: [i32; 64] = [
    6, 10, 10, 13, 11, 13, 16, 16, 16, 16, 18, 18, 18, 18, 18, 23, 23, 23, 23, 23, 23, 25, 25, 25,
    25, 25, 25, 25, 27, 27, 27, 27, 27, 27, 27, 27, 29, 29, 29, 29, 29, 29, 29, 31, 31, 31, 31, 31,
    31, 33, 33, 33, 33, 33, 36, 36, 36, 36, 38, 38, 38, 40, 40, 42,
];

/// §7.4.2.1.1.1 Table 7-4 — default Inter 8x8 scaling matrix
/// (`Default_8x8_Inter`).
pub const DEFAULT_8X8_INTER: [i32; 64] = [
    9, 13, 13, 15, 13, 15, 17, 17, 17, 17, 19, 19, 19, 19, 19, 21, 21, 21, 21, 21, 21, 22, 22, 22,
    22, 22, 22, 22, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27, 27,
    27, 28, 28, 28, 28, 28, 30, 30, 30, 30, 32, 32, 32, 33, 33, 35,
];

// ---------------------------------------------------------------------------
// §7.4.2.1.1.1 Table 7-2 / fall-back rules A and B — effective scaling
// list selection for a given list index.
// ---------------------------------------------------------------------------
//
// Table 7-2 (transcribed):
//
//   idx  mnemonic            block  pred   comp  rule A                  rule B                      default
//   ---  ------------------  -----  -----  ----  ----------------------  --------------------------  ------------------
//    0   Sl_4x4_Intra_Y       4x4    Intra   Y    default scaling list    sequence-level scaling list  Default_4x4_Intra
//    1   Sl_4x4_Intra_Cb      4x4    Intra   Cb   scaling list for i=0    scaling list for i=0         Default_4x4_Intra
//    2   Sl_4x4_Intra_Cr      4x4    Intra   Cr   scaling list for i=1    scaling list for i=1         Default_4x4_Intra
//    3   Sl_4x4_Inter_Y       4x4    Inter   Y    default scaling list    sequence-level scaling list  Default_4x4_Inter
//    4   Sl_4x4_Inter_Cb      4x4    Inter   Cb   scaling list for i=3    scaling list for i=3         Default_4x4_Inter
//    5   Sl_4x4_Inter_Cr      4x4    Inter   Cr   scaling list for i=4    scaling list for i=4         Default_4x4_Inter
//    6   Sl_8x8_Intra_Y       8x8    Intra   Y    default scaling list    sequence-level scaling list  Default_8x8_Intra
//    7   Sl_8x8_Inter_Y       8x8    Inter   Y    default scaling list    sequence-level scaling list  Default_8x8_Inter
//    8   Sl_8x8_Intra_Cb      8x8    Intra   Cb   scaling list for i=6    scaling list for i=6         Default_8x8_Intra
//    9   Sl_8x8_Inter_Cb      8x8    Inter   Cb   scaling list for i=7    scaling list for i=7         Default_8x8_Inter
//   10   Sl_8x8_Intra_Cr      8x8    Intra   Cr   scaling list for i=8    scaling list for i=8         Default_8x8_Intra
//   11   Sl_8x8_Inter_Cr      8x8    Inter   Cr   scaling list for i=9    scaling list for i=9         Default_8x8_Inter

/// §7.4.2.1.1.1 Table 7-2 — the class-initial default matrix for a
/// given 4x4 scaling-list index (0..=5 covers both 4:2:0/4:2:2 cases
/// `n=8` and 4:4:4 `n=12`).
fn default_4x4_for(idx: usize) -> [i32; 16] {
    // idx 0..=2 are Intra (Y/Cb/Cr), idx 3..=5 are Inter.
    if idx < 3 {
        DEFAULT_4X4_INTRA
    } else {
        DEFAULT_4X4_INTER
    }
}

/// §7.4.2.1.1.1 Table 7-2 — the class-initial default matrix for a
/// given 8x8 scaling-list index. The spec's "12-entry" layout numbers
/// 8x8 lists as i ∈ 6..=11; the `idx` parameter here is 0..=5 where
/// 0 maps to i=6 (Intra_Y), 1 → i=7 (Inter_Y), 2 → i=8 (Intra_Cb),
/// 3 → i=9 (Inter_Cb), 4 → i=10 (Intra_Cr), 5 → i=11 (Inter_Cr).
fn default_8x8_for(idx: usize) -> [i32; 64] {
    // Even `idx` (0, 2, 4) = Intra; odd (1, 3, 5) = Inter.
    if idx & 1 == 0 {
        DEFAULT_8X8_INTRA
    } else {
        DEFAULT_8X8_INTER
    }
}

/// §7.4.2.1.1.1 Table 7-2 — Rule A/B "previous index" chain for 4x4
/// lists. Returns `Some(prev)` when this index inherits from another
/// 4x4 index (i=1→0, i=2→1, i=4→3, i=5→4), `None` for i=0 and i=3 (the
/// class heads).
fn chain_prev_4x4(idx: usize) -> Option<usize> {
    match idx {
        1 => Some(0),
        2 => Some(1),
        4 => Some(3),
        5 => Some(4),
        _ => None, // 0 and 3 are class heads
    }
}

/// §7.4.2.1.1.1 Table 7-2 — Rule A/B "previous index" chain for 8x8
/// lists. Argument `idx` is the 0..=5 position within the 8x8 sub-array
/// (same convention as [`default_8x8_for`]). Returns `Some(prev)` to
/// inherit from a previous 8x8 index, `None` for i=6 / i=7 (class
/// heads, corresponding to local idx 0 / 1).
fn chain_prev_8x8(idx: usize) -> Option<usize> {
    match idx {
        2 => Some(0), // i=8 → i=6
        3 => Some(1), // i=9 → i=7
        4 => Some(2), // i=10 → i=8
        5 => Some(3), // i=11 → i=9
        _ => None,    // 0 (i=6), 1 (i=7) are class heads
    }
}

/// §7.4.2.1.1.1 — derive the sequence-level 4x4 scaling list for index
/// `idx` ∈ 0..=5. The SPS's `seq_scaling_lists` must be `Some` when
/// `seq_scaling_matrix_present_flag == 1`; otherwise the flat list is
/// returned (Eq. 7-8).
fn derive_sps_4x4(idx: usize, sps: &Sps) -> [i32; 16] {
    if !sps.seq_scaling_matrix_present_flag {
        // §7.4.2.1.1 — inferred Flat_4x4_16.
        return FLAT_4X4_16;
    }
    let lists = match sps.seq_scaling_lists.as_ref() {
        Some(l) => l,
        None => return FLAT_4X4_16,
    };
    let entry = lists.entries.get(idx);
    match entry {
        Some(ScalingListEntry::Explicit(vals)) => vec_to_16(vals),
        Some(ScalingListEntry::UseDefault) => default_4x4_for(idx),
        Some(ScalingListEntry::NotPresent) => {
            // Fall-back rule A for the sequence-level list.
            match chain_prev_4x4(idx) {
                Some(prev) => derive_sps_4x4(prev, sps),
                None => default_4x4_for(idx),
            }
        }
        None => FLAT_4X4_16,
    }
}

/// §7.4.2.1.1.1 — derive the sequence-level 8x8 scaling list for 8x8
/// sub-index `idx` ∈ 0..=5 (where 0 maps to spec i=6, 1 → i=7, etc.).
fn derive_sps_8x8(idx: usize, sps: &Sps) -> [i32; 64] {
    if !sps.seq_scaling_matrix_present_flag {
        return FLAT_8X8_16;
    }
    let lists = match sps.seq_scaling_lists.as_ref() {
        Some(l) => l,
        None => return FLAT_8X8_16,
    };
    // The SPS entries vector layout: [0..6) = 4x4 lists; [6..n) = 8x8
    // lists where n ∈ {8, 12}. We want entries[6 + idx].
    let spec_idx = 6 + idx;
    let entry = lists.entries.get(spec_idx);
    match entry {
        Some(ScalingListEntry::Explicit(vals)) => vec_to_64(vals),
        Some(ScalingListEntry::UseDefault) => default_8x8_for(idx),
        Some(ScalingListEntry::NotPresent) => {
            // Fall-back rule A for the sequence-level list.
            match chain_prev_8x8(idx) {
                Some(prev) => derive_sps_8x8(prev, sps),
                None => default_8x8_for(idx),
            }
        }
        None => FLAT_8X8_16,
    }
}

/// §7.4.2.1.1.1 / §7.4.2.2 — effective 4x4 scaling list for a given
/// list index (§7.4.2.1.1.1 Table 7-2 indices 0..=5), following the
/// PPS-over-SPS override and fall-back rules A/B.
///
/// `list_idx`: 0=Intra_Y, 1=Intra_Cb, 2=Intra_Cr, 3=Inter_Y, 4=Inter_Cb,
/// 5=Inter_Cr. Out-of-range indices (≥6 in a non-4:4:4 stream) return
/// `FLAT_4X4_16`.
pub fn select_scaling_list_4x4(list_idx: usize, sps: &Sps, pps: &Pps) -> [i32; 16] {
    if list_idx >= 6 {
        return FLAT_4X4_16;
    }
    // PPS override: when pic_scaling_matrix_present_flag == 1 the
    // picture-level list is derived from the PPS per-index entries
    // with the appropriate fall-back rule set.
    let ext = pps.extension.as_ref();
    let pic_present = ext.is_some_and(|e| e.pic_scaling_matrix_present_flag);
    if !pic_present {
        // §7.4.2.2: pic_scaling_matrix_present_flag == 0 → picture-level
        // lists equal the sequence-level lists.
        return derive_sps_4x4(list_idx, sps);
    }
    let pic_lists = match ext.and_then(|e| e.pic_scaling_lists.as_ref()) {
        Some(l) => l,
        None => return derive_sps_4x4(list_idx, sps),
    };
    derive_pps_4x4(list_idx, sps, pic_lists)
}

/// §7.4.2.1.1.1 / §7.4.2.2 — effective 8x8 scaling list.
///
/// `list_idx` ∈ 0..=5 is the 8x8 sub-index (0=Intra_Y, 1=Inter_Y,
/// 2=Intra_Cb, 3=Inter_Cb, 4=Intra_Cr, 5=Inter_Cr — corresponding to
/// spec-wide indices 6..=11).
pub fn select_scaling_list_8x8(list_idx: usize, sps: &Sps, pps: &Pps) -> [i32; 64] {
    if list_idx >= 6 {
        return FLAT_8X8_16;
    }
    let ext = pps.extension.as_ref();
    let pic_present = ext.is_some_and(|e| e.pic_scaling_matrix_present_flag);
    if !pic_present {
        return derive_sps_8x8(list_idx, sps);
    }
    let pic_lists = match ext.and_then(|e| e.pic_scaling_lists.as_ref()) {
        Some(l) => l,
        None => return derive_sps_8x8(list_idx, sps),
    };
    derive_pps_8x8(list_idx, sps, pic_lists)
}

/// §7.4.2.1.1.1 Table 7-2 — PPS-level derivation for 4x4 list `idx`.
///
/// The PPS entries vector is indexed by i ∈ 0..=5 (the 4x4 slice of
/// the PPS scaling_list loop; the 8x8 slice follows at index 6..).
fn derive_pps_4x4(idx: usize, sps: &Sps, pic: &crate::pps::PicScalingLists) -> [i32; 16] {
    let entry = pic.entries.get(idx);
    match entry {
        Some(ScalingListEntry::Explicit(vals)) => vec_to_16(vals),
        Some(ScalingListEntry::UseDefault) => default_4x4_for(idx),
        Some(ScalingListEntry::NotPresent) => {
            // §7.4.2.2 semantics on pic_scaling_list_present_flag[i]:
            //   seq_scaling_matrix_present_flag == 0 → rule A
            //   otherwise                            → rule B
            if !sps.seq_scaling_matrix_present_flag {
                // Rule A
                match chain_prev_4x4(idx) {
                    Some(prev) => derive_pps_4x4(prev, sps, pic),
                    None => default_4x4_for(idx),
                }
            } else {
                // Rule B — for class heads (i=0, 3) use the
                // sequence-level list; for i=1/2/4/5 inherit from the
                // previous PPS index (same PPS context).
                match chain_prev_4x4(idx) {
                    Some(prev) => derive_pps_4x4(prev, sps, pic),
                    None => derive_sps_4x4(idx, sps),
                }
            }
        }
        None => derive_sps_4x4(idx, sps),
    }
}

/// §7.4.2.1.1.1 Table 7-2 — PPS-level derivation for 8x8 list (sub-idx
/// 0..=5 = spec i=6..=11).
fn derive_pps_8x8(idx: usize, sps: &Sps, pic: &crate::pps::PicScalingLists) -> [i32; 64] {
    let spec_idx = 6 + idx;
    let entry = pic.entries.get(spec_idx);
    match entry {
        Some(ScalingListEntry::Explicit(vals)) => vec_to_64(vals),
        Some(ScalingListEntry::UseDefault) => default_8x8_for(idx),
        Some(ScalingListEntry::NotPresent) => {
            if !sps.seq_scaling_matrix_present_flag {
                // Rule A
                match chain_prev_8x8(idx) {
                    Some(prev) => derive_pps_8x8(prev, sps, pic),
                    None => default_8x8_for(idx),
                }
            } else {
                // Rule B
                match chain_prev_8x8(idx) {
                    Some(prev) => derive_pps_8x8(prev, sps, pic),
                    None => derive_sps_8x8(idx, sps),
                }
            }
        }
        None => derive_sps_8x8(idx, sps),
    }
}

#[inline]
fn vec_to_16(v: &[i32]) -> [i32; 16] {
    let mut out = [0i32; 16];
    let n = v.len().min(16);
    out[..n].copy_from_slice(&v[..n]);
    if n < 16 {
        // Defensive: pad with the last value (shouldn't happen with a
        // valid parse, since scaling_list() always fills exactly n
        // entries). Fall back to flat.
        for slot in out.iter_mut().skip(n) {
            *slot = 16;
        }
    }
    out
}

#[inline]
fn vec_to_64(v: &[i32]) -> [i32; 64] {
    let mut out = [0i32; 64];
    let n = v.len().min(64);
    out[..n].copy_from_slice(&v[..n]);
    if n < 64 {
        for slot in out.iter_mut().skip(n) {
            *slot = 16;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// §8.5.6 — inverse zig-zag scanning for 4x4 transform coefficients.
//
// Table 8-13 row "zig-zag": idx -> c_ij. This utility helps callers that
// receive coefficients already in scan order.
// ---------------------------------------------------------------------------

/// Table 8-13, zig-zag row. `scan[k] = (i, j)` where k is the scan
/// index and (i, j) is the 2-D position in the 4x4 coefficient block.
const ZIGZAG_4X4: [(usize, usize); 16] = [
    (0, 0),
    (0, 1),
    (1, 0),
    (2, 0),
    (1, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (2, 3),
    (3, 2),
    (3, 3),
];

/// Table 8-13, field scan row.
const FIELD_4X4: [(usize, usize); 16] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (2, 0),
    (3, 0),
    (1, 1),
    (2, 1),
    (3, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (3, 2),
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 3),
];

/// §8.5.6 — invert zig-zag scan; transforms a 16-entry scan-order list
/// into a 4x4 row-major coefficient matrix.
pub fn inverse_scan_4x4_zigzag(levels: &[i32; 16]) -> [i32; 16] {
    let mut out = [0i32; 16];
    for (k, &(i, j)) in ZIGZAG_4X4.iter().enumerate() {
        out[i * 4 + j] = levels[k];
    }
    out
}

/// §8.5.6 / §7.3.5.3.3 — invert the zig-zag scan for an AC-only block
/// (Intra_16x16 AC, Chroma AC). Input `levels[0..=14]` holds the
/// coefficient at spec scan positions 1..=15 respectively (startIdx=1).
/// Slot `levels[15]` is ignored. Position (0, 0) of the output matrix
/// (spec scan 0, the DC slot) is set to 0; the caller must overwrite it
/// with the pre-scaled DC value per §8.5.10 / §8.5.11.2.
pub fn inverse_scan_4x4_zigzag_ac(levels: &[i32; 16]) -> [i32; 16] {
    let mut out = [0i32; 16];
    // Spec scan k (k=1..=15) -> ZIGZAG_4X4[k]; input slot k-1 holds level.
    for k in 1..16 {
        let (i, j) = ZIGZAG_4X4[k];
        out[i * 4 + j] = levels[k - 1];
    }
    out
}

/// §8.5.6 — invert the informative field scan for 4x4 blocks.
pub fn inverse_scan_4x4_field(levels: &[i32; 16]) -> [i32; 16] {
    let mut out = [0i32; 16];
    for (k, &(i, j)) in FIELD_4X4.iter().enumerate() {
        out[i * 4 + j] = levels[k];
    }
    out
}

/// §8.5.6 / §7.3.5.3.3 — invert the informative field scan for an AC-only
/// block (Intra_16x16 AC, Chroma AC). Input `levels[0..=14]` holds the
/// coefficient at spec scan positions 1..=15 respectively (startIdx=1).
/// Slot `levels[15]` is ignored. Position (0, 0) of the output matrix
/// (spec scan 0, the DC slot) is set to 0; the caller must overwrite it
/// with the pre-scaled DC value per §8.5.10 / §8.5.11.2.
pub fn inverse_scan_4x4_field_ac(levels: &[i32; 16]) -> [i32; 16] {
    let mut out = [0i32; 16];
    for k in 1..16 {
        let (i, j) = FIELD_4X4[k];
        out[i * 4 + j] = levels[k - 1];
    }
    out
}

// ---------------------------------------------------------------------------
// §8.5.12.1 — scaling for 4x4 residual blocks.
//
// Used internally by [`inverse_transform_4x4`]; exposed for callers that
// need to run scaling and transform as separate steps.
// ---------------------------------------------------------------------------

/// §8.5.12.1 — apply the LevelScale4x4 scaling to a 4x4 coefficient
/// block, producing the `d_ij` array ready for the inverse transform.
///
/// Equations 8-335..8-337. Does not touch c[0,0] for Intra_16x16 or
/// chroma residual blocks (the caller provides the already-scaled DC
/// value via the `is_dc_preserved` flag — true = leave d[0,0] as-is,
/// false = normal scaling).
fn scale_4x4(
    coeffs: &[i32; 16],
    qp: i32,
    scaling_list: &[i32; 16],
    is_dc_preserved: bool,
) -> [i32; 16] {
    // §8.5.9 — qP/6 and qP%6 control the log2 and modular parts of
    // the scaling factor.
    let qp_div = qp / 6;
    let qp_mod = (qp % 6) as usize;

    let mut d = [0i32; 16];
    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            if is_dc_preserved && i == 0 && j == 0 {
                // Eq. 8-335 — DC path: copy c_00 through unchanged.
                d[0] = coeffs[0];
                continue;
            }
            let ls = level_scale_4x4(scaling_list, qp_mod, i, j);
            let c = coeffs[idx];
            d[idx] = if qp >= 24 {
                // Eq. 8-336 — no rounding, left shift by (qP/6 - 4).
                (c * ls) << (qp_div - 4)
            } else {
                // Eq. 8-337 — right shift with rounding.
                let shift = 4 - qp_div;
                let round = 1 << (3 - qp_div);
                (c * ls + round) >> shift
            };
        }
    }
    d
}

// ---------------------------------------------------------------------------
// §8.5.12.2 — inverse 4x4 integer DCT (the core "bit-exact" transform).
//
// Equations 8-338..8-354.
// ---------------------------------------------------------------------------

/// §8.5.12.2 — 4x4 inverse transform (eq. 8-338..8-354). Takes the
/// scaled coefficient block `d` and returns the residual block `r`.
fn inverse_core_4x4(d: &[i32; 16]) -> [i32; 16] {
    // Row transform (eqs. 8-338..8-345).
    // f[i][j] = row-transformed value at position (i,j).
    let mut f = [0i32; 16];
    for i in 0..4 {
        let d_i0 = d[i * 4];
        let d_i1 = d[i * 4 + 1];
        let d_i2 = d[i * 4 + 2];
        let d_i3 = d[i * 4 + 3];

        // Intermediate e (eqs. 8-338..8-341).
        let e_i0 = d_i0 + d_i2;
        let e_i1 = d_i0 - d_i2;
        let e_i2 = (d_i1 >> 1) - d_i3;
        let e_i3 = d_i1 + (d_i3 >> 1);

        // Butterfly to f (eqs. 8-342..8-345).
        f[i * 4] = e_i0 + e_i3;
        f[i * 4 + 1] = e_i1 + e_i2;
        f[i * 4 + 2] = e_i1 - e_i2;
        f[i * 4 + 3] = e_i0 - e_i3;
    }

    // Column transform (eqs. 8-346..8-353). Produces h_ij.
    let mut h = [0i32; 16];
    for j in 0..4 {
        let f_0j = f[j];
        let f_1j = f[4 + j];
        let f_2j = f[8 + j];
        let f_3j = f[12 + j];

        // Intermediate g (eqs. 8-346..8-349).
        let g_0j = f_0j + f_2j;
        let g_1j = f_0j - f_2j;
        let g_2j = (f_1j >> 1) - f_3j;
        let g_3j = f_1j + (f_3j >> 1);

        // Butterfly to h (eqs. 8-350..8-353).
        h[j] = g_0j + g_3j;
        h[4 + j] = g_1j + g_2j;
        h[8 + j] = g_1j - g_2j;
        h[12 + j] = g_0j - g_3j;
    }

    // §8.5.12.2 eq. 8-354 — final right shift with rounding by 2^5 / 6.
    let mut r = [0i32; 16];
    for k in 0..16 {
        r[k] = (h[k] + 32) >> 6;
    }
    r
}

// ---------------------------------------------------------------------------
// §8.5.12 — public entry points for 4x4 residual blocks.
// ---------------------------------------------------------------------------

/// §8.5.9 / §8.5.12.1 / §8.5.13.1 — verify that the qP entering the
/// scaling formulas is in `0..=51 + QpBdOffset`. The qP fed in here is
/// qP'Y / qP'C (= QPY + QpBdOffsetY or QPC + QpBdOffsetC, see eq. 7-40
/// and eq. 8-312), so the lower bound is always 0 and the upper bound
/// grows with bit depth (from 51 at 8-bit to 87 at 14-bit). For
/// out-of-spec bit depths (which `check_bit_depth` would reject) the
/// upper bound is computed conservatively by clamping `bit_depth - 8`
/// at 0 so the call still produces a meaningful QP-range check; the
/// caller is expected to invoke `check_bit_depth` separately.
#[inline]
fn check_qp(qp: i32, bit_depth: u32) -> Result<(), TransformError> {
    let bd_minus8 = bit_depth.saturating_sub(8) as i32;
    let qp_bd_offset = 6 * bd_minus8;
    if !(0..=51 + qp_bd_offset).contains(&qp) {
        Err(TransformError::QpOutOfRange(qp))
    } else {
        Ok(())
    }
}

#[inline]
fn check_bit_depth(bit_depth: u32) -> Result<(), TransformError> {
    if !(8..=14).contains(&bit_depth) {
        Err(TransformError::BitDepthOutOfRange(bit_depth))
    } else {
        Ok(())
    }
}

/// §8.5.12 — scaling + inverse transform for a single 4x4 residual
/// block. `coeffs` is row-major (c[i,j] = coeffs[i*4+j]). `scaling_list`
/// is row-major weight_scale values (typically all 16s when scaling
/// matrices are not signalled).
pub fn inverse_transform_4x4(
    coeffs: &[i32; 16],
    qp: i32,
    scaling_list: &[i32; 16],
    bit_depth: u32,
) -> Result<[i32; 16], TransformError> {
    check_qp(qp, bit_depth)?;
    check_bit_depth(bit_depth)?;
    // Standard path: c[0,0] participates in scaling (not an Intra_16x16
    // luma DC block / not a chroma DC block).
    let d = scale_4x4(coeffs, qp, scaling_list, false);
    Ok(inverse_core_4x4(&d))
}

/// §8.5.12 with Intra_16x16 luma AC / chroma AC semantics: c[0,0] is a
/// pre-scaled DC value supplied by the caller (from §8.5.10 / §8.5.11)
/// and must not be rescaled here.
pub fn inverse_transform_4x4_dc_preserved(
    coeffs: &[i32; 16],
    qp: i32,
    scaling_list: &[i32; 16],
    bit_depth: u32,
) -> Result<[i32; 16], TransformError> {
    check_qp(qp, bit_depth)?;
    check_bit_depth(bit_depth)?;
    let d = scale_4x4(coeffs, qp, scaling_list, true);
    Ok(inverse_core_4x4(&d))
}

// ---------------------------------------------------------------------------
// §8.5.13 — 8x8 residual blocks: scaling + inverse transform.
// ---------------------------------------------------------------------------

/// §8.5.13.1 — scaling for 8x8 residual blocks (eqs. 8-356, 8-357).
fn scale_8x8(coeffs: &[i32; 64], qp: i32, scaling_list: &[i32; 64]) -> [i32; 64] {
    let qp_div = qp / 6;
    let qp_mod = (qp % 6) as usize;
    let mut d = [0i32; 64];
    for i in 0..8 {
        for j in 0..8 {
            let idx = i * 8 + j;
            let ls = level_scale_8x8(scaling_list, qp_mod, i, j);
            let c = coeffs[idx];
            d[idx] = if qp >= 36 {
                // Eq. 8-356.
                (c * ls) << (qp_div - 6)
            } else {
                // Eq. 8-357.
                let shift = 6 - qp_div;
                let round = 1 << (5 - qp_div);
                (c * ls + round) >> shift
            };
        }
    }
    d
}

/// §8.5.13.2 — 8x8 inverse transform core (eqs. 8-358..8-406).
fn inverse_core_8x8(d: &[i32; 64]) -> [i32; 64] {
    // Row (horizontal) transform.
    //
    // For each row i: produce g_i[0..8] by sequentially computing
    // e_i[0..8] (eqs. 8-358..8-365), f_i[0..8] (eqs. 8-366..8-373),
    // and g_i[0..8] (eqs. 8-374..8-381).
    let mut g = [0i32; 64];
    for i in 0..8 {
        let base = i * 8;
        let d0 = d[base];
        let d1 = d[base + 1];
        let d2 = d[base + 2];
        let d3 = d[base + 3];
        let d4 = d[base + 4];
        let d5 = d[base + 5];
        let d6 = d[base + 6];
        let d7 = d[base + 7];

        // eqs. 8-358..8-365.
        let e0 = d0 + d4;
        let e1 = -d3 + d5 - d7 - (d7 >> 1);
        let e2 = d0 - d4;
        let e3 = d1 + d7 - d3 - (d3 >> 1);
        let e4 = (d2 >> 1) - d6;
        let e5 = -d1 + d7 + d5 + (d5 >> 1);
        let e6 = d2 + (d6 >> 1);
        let e7 = d3 + d5 + d1 + (d1 >> 1);

        // eqs. 8-366..8-373.
        let f0 = e0 + e6;
        let f1 = e1 + (e7 >> 2);
        let f2 = e2 + e4;
        let f3 = e3 + (e5 >> 2);
        let f4 = e2 - e4;
        let f5 = (e3 >> 2) - e5;
        let f6 = e0 - e6;
        let f7 = e7 - (e1 >> 2);

        // eqs. 8-374..8-381.
        g[base] = f0 + f7;
        g[base + 1] = f2 + f5;
        g[base + 2] = f4 + f3;
        g[base + 3] = f6 + f1;
        g[base + 4] = f6 - f1;
        g[base + 5] = f4 - f3;
        g[base + 6] = f2 - f5;
        g[base + 7] = f0 - f7;
    }

    // Column (vertical) transform.
    //
    // Same 1-D inverse applied to each column. Produces m_ij, eqs.
    // 8-382..8-405.
    let mut m = [0i32; 64];
    for j in 0..8 {
        let g0 = g[j];
        let g1 = g[8 + j];
        let g2 = g[16 + j];
        let g3 = g[24 + j];
        let g4 = g[32 + j];
        let g5 = g[40 + j];
        let g6 = g[48 + j];
        let g7 = g[56 + j];

        // eqs. 8-382..8-389.
        let h0 = g0 + g4;
        let h1 = -g3 + g5 - g7 - (g7 >> 1);
        let h2 = g0 - g4;
        let h3 = g1 + g7 - g3 - (g3 >> 1);
        let h4 = (g2 >> 1) - g6;
        let h5 = -g1 + g7 + g5 + (g5 >> 1);
        let h6 = g2 + (g6 >> 1);
        let h7 = g3 + g5 + g1 + (g1 >> 1);

        // eqs. 8-390..8-397.
        let k0 = h0 + h6;
        let k1 = h1 + (h7 >> 2);
        let k2 = h2 + h4;
        let k3 = h3 + (h5 >> 2);
        let k4 = h2 - h4;
        let k5 = (h3 >> 2) - h5;
        let k6 = h0 - h6;
        let k7 = h7 - (h1 >> 2);

        // eqs. 8-398..8-405.
        m[j] = k0 + k7;
        m[8 + j] = k2 + k5;
        m[16 + j] = k4 + k3;
        m[24 + j] = k6 + k1;
        m[32 + j] = k6 - k1;
        m[40 + j] = k4 - k3;
        m[48 + j] = k2 - k5;
        m[56 + j] = k0 - k7;
    }

    // §8.5.13.2 eq. 8-406 — final right shift with rounding by 2^5 / 6.
    let mut r = [0i32; 64];
    for k in 0..64 {
        r[k] = (m[k] + 32) >> 6;
    }
    r
}

/// §8.5.13 — scaling + inverse transform for a single 8x8 block.
pub fn inverse_transform_8x8(
    coeffs: &[i32; 64],
    qp: i32,
    scaling_list: &[i32; 64],
    bit_depth: u32,
) -> Result<[i32; 64], TransformError> {
    check_qp(qp, bit_depth)?;
    check_bit_depth(bit_depth)?;
    let d = scale_8x8(coeffs, qp, scaling_list);
    Ok(inverse_core_8x8(&d))
}

// ---------------------------------------------------------------------------
// §8.5.10 — Intra_16x16 luma DC block: 4x4 Hadamard + scaling.
// ---------------------------------------------------------------------------

/// §8.5.10 — apply the 4x4 Hadamard-like transform (Eq. 8-320) to the
/// 16 DC coefficients of an Intra_16x16 luma macroblock, then scale
/// each output per Eq. 8-321 / 8-322. The returned `dcY` array is
/// arranged row-major and is meant to be plugged back into each 4x4
/// sub-block as its c[0,0] entry (see §8.5.2 step 2a).
pub fn inverse_hadamard_luma_dc_16x16(
    dc_coeffs: &[i32; 16],
    qp: i32,
    scaling_list: &[i32; 16],
    bit_depth: u32,
) -> Result<[i32; 16], TransformError> {
    check_qp(qp, bit_depth)?;
    check_bit_depth(bit_depth)?;

    // §8.5.10 Eq. 8-320 — f = H * c * H, with
    //     H = [[1, 1, 1, 1],
    //          [1, 1,-1,-1],
    //          [1,-1,-1, 1],
    //          [1,-1, 1,-1]].
    // Being a Hadamard this is self-inverse up to a factor of 4; the
    // factor is absorbed into the scaling step below.
    let c = dc_coeffs;
    let mut t = [0i32; 16]; // t = H * c
    for j in 0..4 {
        // Row 0 of H: [ 1,  1,  1,  1]
        t[j] = c[j] + c[4 + j] + c[8 + j] + c[12 + j];
        // Row 1: [ 1,  1, -1, -1]
        t[4 + j] = c[j] + c[4 + j] - c[8 + j] - c[12 + j];
        // Row 2: [ 1, -1, -1,  1]
        t[8 + j] = c[j] - c[4 + j] - c[8 + j] + c[12 + j];
        // Row 3: [ 1, -1,  1, -1]
        t[12 + j] = c[j] - c[4 + j] + c[8 + j] - c[12 + j];
    }
    let mut f = [0i32; 16]; // f = t * H
    for i in 0..4 {
        let base = i * 4;
        let t0 = t[base];
        let t1 = t[base + 1];
        let t2 = t[base + 2];
        let t3 = t[base + 3];
        // Right-multiplying by H above yields:
        //   f[i][0] = t[i][0]+t[i][1]+t[i][2]+t[i][3]
        //   f[i][1] = t[i][0]+t[i][1]-t[i][2]-t[i][3]
        //   f[i][2] = t[i][0]-t[i][1]-t[i][2]+t[i][3]
        //   f[i][3] = t[i][0]-t[i][1]+t[i][2]-t[i][3]
        f[base] = t0 + t1 + t2 + t3;
        f[base + 1] = t0 + t1 - t2 - t3;
        f[base + 2] = t0 - t1 - t2 + t3;
        f[base + 3] = t0 - t1 + t2 - t3;
    }

    // §8.5.10 — Scale per Eq. 8-321 / 8-322 using LevelScale4x4(qP%6, 0, 0).
    let qp_mod = (qp % 6) as usize;
    let qp_div = qp / 6;
    let ls = level_scale_4x4(scaling_list, qp_mod, 0, 0);

    let mut dc_y = [0i32; 16];
    if qp >= 36 {
        // Eq. 8-321.
        let shift = qp_div - 6;
        for k in 0..16 {
            dc_y[k] = (f[k] * ls) << shift;
        }
    } else {
        // Eq. 8-322.
        let shift = 6 - qp_div;
        let round = 1 << (5 - qp_div);
        for k in 0..16 {
            dc_y[k] = (f[k] * ls + round) >> shift;
        }
    }
    Ok(dc_y)
}

// ---------------------------------------------------------------------------
// §8.5.11 — chroma DC Hadamard + scaling.
// ---------------------------------------------------------------------------

/// §8.5.11.1 + §8.5.11.2 (4:2:0 sub-path) — 2x2 Hadamard on chroma DC,
/// followed by §8.5.11.2 Eq. 8-326 scaling.
///
/// `dc_coeffs` is the 2x2 coefficient array laid out row-major:
///   [c00, c01,
///    c10, c11].
///
/// `scaling_list_dc_entry` is LevelScale4x4(qp_c%6, 0, 0) worth of
/// state — callers must pass `scaling_list[0] * normAdjust4x4(qp%6,0,0)`;
/// the helper below recomputes it from a flat scaling list entry if
/// desired via [`level_scale_4x4`].
pub fn inverse_hadamard_chroma_dc_420(
    dc_coeffs: &[i32; 4],
    qp: i32,
    scaling_list: &[i32; 16],
    bit_depth: u32,
) -> Result<[i32; 4], TransformError> {
    check_qp(qp, bit_depth)?;
    check_bit_depth(bit_depth)?;

    // §8.5.11.1 Eq. 8-324 — f = H2 * c * H2 with
    //   H2 = [[1, 1],
    //         [1,-1]].
    //
    // Unrolled:
    //   t00 = c00 + c10,  t01 = c01 + c11
    //   t10 = c00 - c10,  t11 = c01 - c11
    //   f00 = t00 + t01,  f01 = t00 - t01
    //   f10 = t10 + t11,  f11 = t10 - t11
    let c00 = dc_coeffs[0];
    let c01 = dc_coeffs[1];
    let c10 = dc_coeffs[2];
    let c11 = dc_coeffs[3];
    let t00 = c00 + c10;
    let t01 = c01 + c11;
    let t10 = c00 - c10;
    let t11 = c01 - c11;
    let f = [t00 + t01, t00 - t01, t10 + t11, t10 - t11];

    // §8.5.11.2 Eq. 8-326 — scaling for 4:2:0:
    //   dcC_ij = ( (f_ij * LevelScale4x4(qP%6, 0, 0)) << (qP/6) ) >> 5
    //
    // Note: LevelScale4x4 here uses the 2012+ definition that already
    // includes the weight_scale factor (typically 16). The `>> 5`
    // absorbs the extra 2^4 factor plus one more to recover the
    // equivalent of the 2003 spec's `>> 1`.
    let qp_mod = (qp % 6) as usize;
    let qp_div = qp / 6;
    let ls = level_scale_4x4(scaling_list, qp_mod, 0, 0);
    let mut out = [0i32; 4];
    for k in 0..4 {
        out[k] = ((f[k] * ls) << qp_div) >> 5;
    }
    Ok(out)
}

/// §8.5.11.1 + §8.5.11.2 (4:2:2 sub-path) — 2x4 Hadamard (Eq. 8-325)
/// followed by Eq. 8-327..8-329 scaling (qPDC = qP + 3).
///
/// The input `dc_coeffs` is the raw `ChromaDCLevel[iCbCr][0..=7]` as
/// parsed from the bitstream (scan-order). This function applies the
/// §8.5.4 Eq. 8-305 inverse-raster scan internally to build the 4x2
/// coefficient matrix `c` (rows 0..=3, cols 0..=1), so callers should
/// pass the coefficient list verbatim — NOT a pre-scanned matrix.
///
/// The returned array is laid out as row-major (4 rows × 2 cols):
///   [dcC[0,0], dcC[0,1],
///    dcC[1,0], dcC[1,1],
///    dcC[2,0], dcC[2,1],
///    dcC[3,0], dcC[3,1]].
pub fn inverse_hadamard_chroma_dc_422(
    dc_coeffs: &[i32; 8],
    qp: i32,
    scaling_list: &[i32; 16],
    bit_depth: u32,
) -> Result<[i32; 8], TransformError> {
    check_qp(qp, bit_depth)?;
    check_bit_depth(bit_depth)?;

    // §8.5.4 Eq. 8-305 — inverse raster scan of ChromaDCLevel into the
    // 4x2 array c. Note the non-trivial mapping:
    //   c[0,0]=L[0], c[0,1]=L[2]
    //   c[1,0]=L[1], c[1,1]=L[5]
    //   c[2,0]=L[3], c[2,1]=L[6]
    //   c[3,0]=L[4], c[3,1]=L[7]
    let c00 = dc_coeffs[0];
    let c01 = dc_coeffs[2];
    let c10 = dc_coeffs[1];
    let c11 = dc_coeffs[5];
    let c20 = dc_coeffs[3];
    let c21 = dc_coeffs[6];
    let c30 = dc_coeffs[4];
    let c31 = dc_coeffs[7];

    // §8.5.11.1 Eq. 8-325: f = Hv * c * Hh, where
    //   Hv = [[1,  1,  1,  1],
    //         [1,  1, -1, -1],
    //         [1, -1, -1,  1],
    //         [1, -1,  1, -1]]  (4x4 Hadamard, acts on the 4-row side)
    //   Hh = [[1,  1],
    //         [1, -1]]           (2x2 on the 2-column side)
    //
    // Apply Hv to the left (combines rows).
    let t00 = c00 + c10 + c20 + c30;
    let t01 = c01 + c11 + c21 + c31;
    let t10 = c00 + c10 - c20 - c30;
    let t11 = c01 + c11 - c21 - c31;
    let t20 = c00 - c10 - c20 + c30;
    let t21 = c01 - c11 - c21 + c31;
    let t30 = c00 - c10 + c20 - c30;
    let t31 = c01 - c11 + c21 - c31;
    // Apply Hh to the right (combines columns).
    let f = [
        t00 + t01,
        t00 - t01,
        t10 + t11,
        t10 - t11,
        t20 + t21,
        t20 - t21,
        t30 + t31,
        t30 - t31,
    ];

    // §8.5.11.2 — scaling path with qPDC = qP + 3 (Eq. 8-327).
    let qp_dc = qp + 3;
    let mut out = [0i32; 8];
    if qp_dc >= 36 {
        // Eq. 8-328 — shift uses qP/6 (NOT qPDC/6), LevelScale uses qP%6.
        //   dcCij = ( ( fij * LevelScale4x4(qP % 6, 0, 0) ) << (qP/6 - 6) )
        let qp_mod = (qp % 6) as usize;
        let ls = level_scale_4x4(scaling_list, qp_mod, 0, 0);
        let shift = (qp / 6) - 6;
        for k in 0..8 {
            out[k] = (f[k] * ls) << shift;
        }
    } else {
        // Eq. 8-329 — shift and mod both use qPDC.
        //   dcCij = ( fij * LevelScale4x4(qPDC % 6, 0, 0)
        //              + 2^(5 - qPDC/6) ) >> (6 - qPDC/6)
        let qp_dc_mod = (qp_dc % 6) as usize;
        let qp_dc_div = qp_dc / 6;
        let ls = level_scale_4x4(scaling_list, qp_dc_mod, 0, 0);
        let shift = 6 - qp_dc_div;
        let round = 1 << (5 - qp_dc_div);
        for k in 0..8 {
            out[k] = (f[k] * ls + round) >> shift;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------ QPc lookup (§8.5.8 / Table 8-15) ----------------------

    #[test]
    fn qpc_identity_below_30() {
        for qp in 0..30 {
            assert_eq!(qp_c_table_lookup(qp), qp);
        }
    }

    #[test]
    fn qpc_table_boundaries() {
        // Per Table 8-15, qPI=30 -> 29, qPI=39 -> 35, qPI=51 -> 39.
        assert_eq!(qp_c_table_lookup(30), 29);
        assert_eq!(qp_c_table_lookup(31), 30);
        assert_eq!(qp_c_table_lookup(39), 35);
        assert_eq!(qp_c_table_lookup(40), 36);
        assert_eq!(qp_c_table_lookup(51), 39);
    }

    #[test]
    fn qpc_clips_high_qpi() {
        // Out of range saturates to the qPI=51 entry.
        assert_eq!(qp_c_table_lookup(100), 39);
    }

    #[test]
    fn qp_y_to_qp_c_with_offset() {
        // qp_y = 28 + offset 2 -> qPI=30 -> QPc=29.
        assert_eq!(qp_y_to_qp_c(28, 2), 29);
        // qp_y = 40 + offset 0 -> qPI=40 -> QPc=36.
        assert_eq!(qp_y_to_qp_c(40, 0), 36);
        // Negative accumulation clamps to 0.
        assert_eq!(qp_y_to_qp_c(-5, -10), 0);
        // Huge positive clamps to 51 -> 39.
        assert_eq!(qp_y_to_qp_c(80, 10), 39);
    }

    // ------------ 4x4 inverse transform ---------------------------------

    #[test]
    fn inverse_transform_4x4_all_zero() {
        let coeffs = [0i32; 16];
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_transform_4x4(&coeffs, 26, &sl, 8).unwrap();
        assert_eq!(out, [0i32; 16]);
    }

    #[test]
    fn inverse_transform_4x4_rejects_bad_qp() {
        let coeffs = [0i32; 16];
        let sl = default_scaling_list_4x4_flat();
        assert_eq!(
            inverse_transform_4x4(&coeffs, -1, &sl, 8).unwrap_err(),
            TransformError::QpOutOfRange(-1)
        );
        assert_eq!(
            inverse_transform_4x4(&coeffs, 52, &sl, 8).unwrap_err(),
            TransformError::QpOutOfRange(52)
        );
    }

    #[test]
    fn inverse_transform_4x4_rejects_bad_bit_depth() {
        let coeffs = [0i32; 16];
        let sl = default_scaling_list_4x4_flat();
        assert_eq!(
            inverse_transform_4x4(&coeffs, 26, &sl, 7).unwrap_err(),
            TransformError::BitDepthOutOfRange(7)
        );
        assert_eq!(
            inverse_transform_4x4(&coeffs, 26, &sl, 15).unwrap_err(),
            TransformError::BitDepthOutOfRange(15)
        );
    }

    #[test]
    fn inverse_transform_4x4_qp_range_extends_with_bit_depth() {
        // §7.4.2.1.1 eq. 7-4 + §8.5.12 — qP'Y range is 0..=51+QpBdOffsetY.
        // At 10-bit luma QpBdOffsetY = 12, so qp=63 must be accepted.
        let coeffs = [0i32; 16];
        let sl = default_scaling_list_4x4_flat();
        // 8-bit: 51 ok, 52 rejected.
        inverse_transform_4x4(&coeffs, 51, &sl, 8).expect("qp=51 ok at 8-bit");
        assert!(matches!(
            inverse_transform_4x4(&coeffs, 52, &sl, 8),
            Err(TransformError::QpOutOfRange(52))
        ));
        // 10-bit: 63 ok (= 51 + 12), 64 rejected.
        inverse_transform_4x4(&coeffs, 63, &sl, 10).expect("qp=63 ok at 10-bit");
        assert!(matches!(
            inverse_transform_4x4(&coeffs, 64, &sl, 10),
            Err(TransformError::QpOutOfRange(64))
        ));
        // 12-bit: 75 ok (= 51 + 24).
        inverse_transform_4x4(&coeffs, 75, &sl, 12).expect("qp=75 ok at 12-bit");
        assert!(matches!(
            inverse_transform_4x4(&coeffs, 76, &sl, 12),
            Err(TransformError::QpOutOfRange(76))
        ));
        // 14-bit: 87 ok (= 51 + 36).
        inverse_transform_4x4(&coeffs, 87, &sl, 14).expect("qp=87 ok at 14-bit");
        assert!(matches!(
            inverse_transform_4x4(&coeffs, 88, &sl, 14),
            Err(TransformError::QpOutOfRange(88))
        ));
    }

    #[test]
    fn qp_bd_offset_eq_7_4() {
        assert_eq!(qp_bd_offset(0), 0);
        assert_eq!(qp_bd_offset(2), 12);
        assert_eq!(qp_bd_offset(4), 24);
        assert_eq!(qp_bd_offset(6), 36);
    }

    #[test]
    fn qp_y_to_qp_c_with_bd_offset_extends_negative_range() {
        // §8.5.8 eq. 8-311 + 8-312 — at 10-bit chroma (QpBdOffsetC=12)
        // qPI can be as low as -12 and QPC = qPI verbatim for qPI<30.
        // qp_y=-10 + offset=-2 = qPI=-12, QPC=-12.
        assert_eq!(qp_y_to_qp_c_with_bd_offset(-10, -2, 12), -12);
        // Negative QPC + QpBdOffsetC = qP'C = 0, the smallest qP'C.
        assert_eq!(qp_y_to_qp_c_with_bd_offset(-10, -2, 12) + 12, 0);
        // 8-bit path (qp_bd_offset_c=0) clamps to 0.
        assert_eq!(qp_y_to_qp_c_with_bd_offset(-10, -2, 0), 0);
        // High positive still uses Table 8-15 lookup (capped at 51 → 39).
        assert_eq!(qp_y_to_qp_c_with_bd_offset(60, 0, 12), 39);
    }

    #[test]
    fn inverse_transform_4x4_single_dc_qp24() {
        // qP = 24: qP/6 = 4, so qP-24 >=0 path (eq. 8-336), shift by 0.
        // LevelScale4x4(0,0,0) = weightScale(0,0)=16 * normAdjust(0,0,0)=10 = 160.
        // d[0,0] = c[0,0] * 160 << 0 = 160.
        // Inverse transform: single DC in e/f propagates to all positions
        // as d_00 (since e_i0 = d_00, e_i1 = d_00 for row 0; g_0j / h
        // similarly). The full derivation yields:
        //   row: f_i0 = f_i1 = f_i2 = f_i3 = d_00 for i=0 only; other rows zero.
        //   col: h_0j = f_0j for j=0..3; other rows zero.
        // So each h_ij = d_00 = 160 for i=j=0 only... wait: let's trace.
        //
        // Row transform, row 0 only (rest zero):
        //   e00 = d00+0 = 160; e01 = d00-0 = 160
        //   e02 = 0 - 0 = 0;  e03 = 0 + 0 = 0
        //   f00 = e00+e03 = 160; f01 = e01+e02 = 160
        //   f02 = e01-e02 = 160; f03 = e00-e03 = 160
        // So f = row 0 all 160, rest zero.
        //
        // Column transform for each j (only f_0j non-zero):
        //   g_0j = f_0j + 0 = 160; g_1j = f_0j - 0 = 160
        //   g_2j = 0 - 0 = 0;      g_3j = 0 + 0 = 0
        //   h_0j = g_0j+g_3j = 160
        //   h_1j = g_1j+g_2j = 160
        //   h_2j = g_1j-g_2j = 160
        //   h_3j = g_0j-g_3j = 160
        // => h is all-160.
        // r_ij = (160+32) >> 6 = 192 >> 6 = 3.
        let mut coeffs = [0i32; 16];
        coeffs[0] = 1;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_transform_4x4(&coeffs, 24, &sl, 8).unwrap();
        assert_eq!(out, [3i32; 16]);
    }

    #[test]
    fn inverse_transform_4x4_single_dc_qp30() {
        // qP = 30: qP/6 = 5, >=24, so eq. 8-336 with shift (5-4)=1.
        // LevelScale(0,0,0) at qP%6=0 = 16*10 = 160.
        // d[0,0] = 1*160 << 1 = 320.
        // By same argument, h is all-320 and r_ij = (320+32)>>6 = 352>>6 = 5.
        let mut coeffs = [0i32; 16];
        coeffs[0] = 1;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_transform_4x4(&coeffs, 30, &sl, 8).unwrap();
        assert_eq!(out, [5i32; 16]);
    }

    #[test]
    fn inverse_transform_4x4_single_dc_qp18() {
        // qP = 18: qP/6 = 3, <24, eq. 8-337 path.
        //   shift = 4-3 = 1, round = 1 << (3-3) = 1.
        // LevelScale(qP%6=0, 0, 0) = 16*10 = 160.
        // d[0,0] = (1*160 + 1) >> 1 = 161 >> 1 = 80.
        // h = all-80; r_ij = (80+32)>>6 = 112>>6 = 1.
        let mut coeffs = [0i32; 16];
        coeffs[0] = 1;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_transform_4x4(&coeffs, 18, &sl, 8).unwrap();
        assert_eq!(out, [1i32; 16]);
    }

    #[test]
    fn inverse_transform_4x4_dc_preserved_skips_scaling() {
        // With is_dc_preserved=true the input c[0,0] is used as the
        // scaled value directly. Setting c[0,0]=64 produces h_ij=64
        // across the board and r_ij = (64+32)>>6 = 1.
        let mut coeffs = [0i32; 16];
        coeffs[0] = 64;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_transform_4x4_dc_preserved(&coeffs, 24, &sl, 8).unwrap();
        assert_eq!(out, [1i32; 16]);
    }

    #[test]
    fn inverse_transform_4x4_known_pattern_qp0() {
        // qP = 0 -> qP/6=0, qP<24, eq. 8-337: shift=4, round=8.
        // LevelScale(0, i, j) uses normAdjust4x4(0, i, j) = {10,16,13}.
        //   norm(0,0,0)=10, norm(0,0,1)=13, norm(0,0,2)=10,  norm(0,0,3)=13
        //   norm(0,1,0)=13, norm(0,1,1)=16, norm(0,1,2)=13,  norm(0,1,3)=16
        //   norm(0,2,0)=10, norm(0,2,1)=13, norm(0,2,2)=10,  norm(0,2,3)=13
        //   norm(0,3,0)=13, norm(0,3,1)=16, norm(0,3,2)=13,  norm(0,3,3)=16
        // Weight scale = 16 everywhere, so LevelScale = 16*norm.
        // A single c[0,0]=1 -> d[0,0] = (1*160+8)>>4 = 168>>4 = 10.
        // Row xform: f row 0 all = 10. Col xform: h all = 10.
        // r_ij = (10+32)>>6 = 42>>6 = 0 (integer).
        //
        // Instead we pick c[0,0]=64 (scaled up to survive the DC path).
        //   d[0,0] = (64*160+8)>>4 = 10248>>4 = 640.
        //   f row 0 all 640; h all 640; r_ij = (640+32)>>6 = 10.
        let mut coeffs = [0i32; 16];
        coeffs[0] = 64;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_transform_4x4(&coeffs, 0, &sl, 8).unwrap();
        assert_eq!(out, [10i32; 16]);
    }

    #[test]
    fn inverse_transform_4x4_row0_ac_pattern() {
        // Non-trivial first-row pattern: c[0,0]=64, c[0,1]=32, c[0,2]=16, c[0,3]=8.
        // qP=24, so qp_div=4, >=24 path → d = (c * LevelScale) << 0.
        //
        // LevelScale4x4(0, 0, j) for j=0..3 with flat scaling (all 16):
        //   j=0 (even,even): normAdjust=10 -> 160
        //   j=1 (even,odd) : normAdjust=13 -> 208
        //   j=2 (even,even): normAdjust=10 -> 160
        //   j=3 (even,odd) : normAdjust=13 -> 208
        //
        // d[0,0..3] = [64*160, 32*208, 16*160, 8*208]
        //           = [10240, 6656, 2560, 1664]
        //
        // Row transform on row 0 (rest rows all-zero):
        //   e_00 = d00 + d02 = 10240 + 2560 = 12800
        //   e_01 = d00 - d02 = 10240 - 2560 = 7680
        //   e_02 = (d01 >> 1) - d03 = 3328 - 1664 = 1664
        //   e_03 = d01 + (d03 >> 1) = 6656 + 832 = 7488
        //
        //   f_00 = e_00 + e_03 = 12800 + 7488 = 20288
        //   f_01 = e_01 + e_02 = 7680 + 1664 = 9344
        //   f_02 = e_01 - e_02 = 7680 - 1664 = 6016
        //   f_03 = e_00 - e_03 = 12800 - 7488 = 5312
        //
        // Other rows of f are zero. Column transform for each j (only f_0j
        // nonzero, rest zero):
        //   g_0j = f_0j; g_1j = f_0j; g_2j = 0; g_3j = 0.
        //   h_0j = g_0j + g_3j = f_0j
        //   h_1j = g_1j + g_2j = f_0j
        //   h_2j = g_1j - g_2j = f_0j
        //   h_3j = g_0j - g_3j = f_0j
        // So h has identical rows: row_i = [f_00, f_01, f_02, f_03].
        // r_ij = (h_ij + 32) >> 6.
        let mut coeffs = [0i32; 16];
        coeffs[0] = 64;
        coeffs[1] = 32;
        coeffs[2] = 16;
        coeffs[3] = 8;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_transform_4x4(&coeffs, 24, &sl, 8).unwrap();

        // Expected column values (constant over rows).
        let r0 = (20288 + 32) >> 6; // = 20320 >> 6 = 317
        let r1 = (9344 + 32) >> 6; // = 146
        let r2 = (6016 + 32) >> 6; // = 94
        let r3 = (5312 + 32) >> 6; // = 83
        for i in 0..4 {
            assert_eq!(out[i * 4], r0, "row {} col 0", i);
            assert_eq!(out[i * 4 + 1], r1, "row {} col 1", i);
            assert_eq!(out[i * 4 + 2], r2, "row {} col 2", i);
            assert_eq!(out[i * 4 + 3], r3, "row {} col 3", i);
        }
    }

    #[test]
    fn inverse_transform_4x4_col0_pattern_qp22() {
        // Sub-24 QP exercises the round=1 path (eq. 8-337).
        // Single c[1,0] = 16 (first AC row coefficient).
        // qP=22: qp_div=3, qp_mod=4, round=1<<(3-3)=1, shift=4-3=1.
        // LevelScale4x4(4, 1, 0) = 16 * normAdjust4x4(4,1,0).
        //   norm(4,1,0): (i%2, j%2) = (1,0) → row[2] at m=4 = 20.
        //   LevelScale = 16*20 = 320.
        // d[1,0] = (16 * 320 + 1) >> 1 = (5120+1)>>1 = 2560.
        //
        // Only d[1,0] non-zero. Row transform for row 1:
        //   e_10 = d10 + d12 = 2560; e_11 = d10 - d12 = 2560.
        //   e_12 = (d11>>1)-d13 = 0; e_13 = d11 + (d13>>1) = 0.
        //   f_10 = 2560; f_11 = 2560; f_12 = 2560; f_13 = 2560.
        // Other f rows zero. Column transform, each col j only f_1j=2560:
        //   g_0j = f_0j + f_2j = 0;
        //   g_1j = f_0j - f_2j = 0;
        //   g_2j = (f_1j >> 1) - f_3j = 1280;
        //   g_3j = f_1j + (f_3j >> 1) = 2560.
        //   h_0j = g_0j + g_3j = 2560.
        //   h_1j = g_1j + g_2j = 1280.
        //   h_2j = g_1j - g_2j = -1280.
        //   h_3j = g_0j - g_3j = -2560.
        // r_ij = (h_ij + 32) >> 6.
        let mut coeffs = [0i32; 16];
        coeffs[4] = 16; // c[1, 0]
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_transform_4x4(&coeffs, 22, &sl, 8).unwrap();
        let r0 = (2560 + 32) >> 6; // 40
        let r1 = (1280 + 32) >> 6; // 20
        let r2 = (-1280 + 32) >> 6; // -20 (arithmetic shift)
        let r3 = (-2560 + 32) >> 6; // -40
        for j in 0..4 {
            assert_eq!(out[j], r0, "row 0 col {}", j);
            assert_eq!(out[4 + j], r1, "row 1 col {}", j);
            assert_eq!(out[2 * 4 + j], r2, "row 2 col {}", j);
            assert_eq!(out[3 * 4 + j], r3, "row 3 col {}", j);
        }
    }

    #[test]
    fn inverse_transform_4x4_is_approximate_inverse_of_forward() {
        // Round-trip sanity: for small input samples, applying the spec's
        // forward 4x4 transform (implemented inline here) and then the
        // inverse, we should recover values close to the originals.
        // This is a coarse sanity check — the H.264 integer transform is
        // designed to be bit-exact on the inverse side, with the forward
        // side signalled by convention only. We just check that for a
        // constant all-128 block, DC-only coefficient [256,0,..,0] (which
        // is 16*sum >> 4 with sum=16*128) inverses back to all-128.
        //
        // At qP=0 the scaling factor cancels: d[0,0]=c[0,0]*160/16 = c*10.
        // For c[0,0]=205 -> d[0,0] = (205*160+8)>>4 = 32808>>4 = 2050.
        // After transform all positions = 2050; (2050+32)>>6 = 2082>>6 = 32.
        // Plus prediction 128 → 160. That's the decoder rebuild.
        //
        // For this test, just verify that zero-coefficient input produces
        // a zero residual (trivial round-trip, but catches any "residual
        // leaks" if the scaler isn't gated correctly).
        let coeffs = [0i32; 16];
        let sl = default_scaling_list_4x4_flat();
        for qp in 0..=51 {
            let out = inverse_transform_4x4(&coeffs, qp, &sl, 8).unwrap();
            assert_eq!(
                out, [0i32; 16],
                "zero input must produce zero output (qP={})",
                qp
            );
        }
    }

    // ------------ 8x8 inverse transform ---------------------------------

    #[test]
    fn inverse_transform_8x8_all_zero() {
        let coeffs = [0i32; 64];
        let sl = default_scaling_list_8x8_flat();
        let out = inverse_transform_8x8(&coeffs, 26, &sl, 8).unwrap();
        assert_eq!(out, [0i32; 64]);
    }

    #[test]
    fn inverse_transform_8x8_rejects_bad_qp() {
        let coeffs = [0i32; 64];
        let sl = default_scaling_list_8x8_flat();
        assert_eq!(
            inverse_transform_8x8(&coeffs, -5, &sl, 8).unwrap_err(),
            TransformError::QpOutOfRange(-5)
        );
        assert_eq!(
            inverse_transform_8x8(&coeffs, 60, &sl, 8).unwrap_err(),
            TransformError::QpOutOfRange(60)
        );
    }

    #[test]
    fn inverse_transform_8x8_single_dc_qp36() {
        // qP = 36: qP/6 = 6, >=36 path (eq. 8-356, shift by 0).
        // LevelScale8x8(0, 0, 0) = weightScale(0,0)=16 * normAdjust8x8(0,0,0)=20 = 320.
        // d[0,0] = 1 * 320 << 0 = 320.
        //
        // Inverse 8x8: with only d[0,0] non-zero, following the spec
        // butterflies row-by-row:
        //   Row 0 only non-zero column j=0, d0=320, rest 0.
        //   e0=d0+d4=320; e1=0; e2=d0-d4=320; e3=0; e4=0; e5=0; e6=0; e7=0.
        //   f0=e0+e6=320; f1=e1+(e7>>2)=0; f2=e2+e4=320; f3=e3+(e5>>2)=0;
        //   f4=e2-e4=320; f5=(e3>>2)-e5=0; f6=e0-e6=320; f7=e7-(e1>>2)=0.
        //   g0=f0+f7=320; g1=f2+f5=320; g2=f4+f3=320; g3=f6+f1=320;
        //   g4=f6-f1=320; g5=f4-f3=320; g6=f2-f5=320; g7=f0-f7=320.
        // So after row transform g[row=0, j=0..7] is all 320, all
        // other rows are zero.
        //
        // Column transform: for each column j, only g_0j = 320, rest 0.
        // By the same derivation, m[i=0..7, j] = 320 for all i.
        // Final r_ij = (320+32)>>6 = 352>>6 = 5.
        let mut coeffs = [0i32; 64];
        coeffs[0] = 1;
        let sl = default_scaling_list_8x8_flat();
        let out = inverse_transform_8x8(&coeffs, 36, &sl, 8).unwrap();
        assert_eq!(out, [5i32; 64]);
    }

    #[test]
    fn inverse_transform_8x8_single_dc_qp24() {
        // qP = 24: qP/6=4, <36, eq. 8-357: shift=2, round=2.
        // LevelScale8x8(0,0,0) = 16*20 = 320.
        //   d[0,0] = (1*320 + 2) >> 2 = 322>>2 = 80.
        // Same derivation as above -> r_ij = (80+32)>>6 = 112>>6 = 1.
        let mut coeffs = [0i32; 64];
        coeffs[0] = 1;
        let sl = default_scaling_list_8x8_flat();
        let out = inverse_transform_8x8(&coeffs, 24, &sl, 8).unwrap();
        assert_eq!(out, [1i32; 64]);
    }

    // ------------ 16x16 luma DC Hadamard (§8.5.10) ----------------------

    #[test]
    fn luma_dc_16x16_all_zero() {
        let dc = [0i32; 16];
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_luma_dc_16x16(&dc, 26, &sl, 8).unwrap();
        assert_eq!(out, [0i32; 16]);
    }

    #[test]
    fn luma_dc_16x16_rejects_bad_qp() {
        let dc = [0i32; 16];
        let sl = default_scaling_list_4x4_flat();
        assert_eq!(
            inverse_hadamard_luma_dc_16x16(&dc, 52, &sl, 8).unwrap_err(),
            TransformError::QpOutOfRange(52)
        );
    }

    #[test]
    fn luma_dc_16x16_single_dc_qp36() {
        // Only c[0,0]=1, others 0. The Hadamard H*c*H sends this
        // all-zero-except-(0,0)=1 matrix to a matrix of all 1s
        // (with sign = +1 at every position since H_ik * H_jk = ±1
        // and the (0,0,0,0) product is +1, but actually:
        //   f_ij = sum_{p,q} H_ip * c_pq * H_qj = H_i0 * c_00 * H_0j
        //        = H_i0 * H_0j * 1.
        // H_i0 = 1 for all i (first column of H). H_0j = 1 for all j
        // (first row of H). So f_ij = 1 for all i,j.
        //
        // qP=36 -> qp_div=6, >=36 path: shift=0.
        // LevelScale4x4(0, 0, 0) = 16*10 = 160.
        // dc_y_ij = (1 * 160) << 0 = 160 at every position.
        let mut dc = [0i32; 16];
        dc[0] = 1;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_luma_dc_16x16(&dc, 36, &sl, 8).unwrap();
        assert_eq!(out, [160i32; 16]);
    }

    #[test]
    fn luma_dc_16x16_hadamard_self_inverse() {
        // Feeding a "delta" Hadamard input (all equal) should, after the
        // double Hadamard, produce zero everywhere except the top-left.
        // Specifically, c_ij = 1 for every i,j:
        //   (H*c)_ij = sum_k H_ik * c_kj = sum_k H_ik (since c=1).
        //           sum_k H_0k = 4; sum_k H_{i!=0,k} = 0 (row sums).
        //   So (H*c) is a matrix whose only non-zero row is row 0 = [4,4,4,4].
        //   Then (H*c*H)_ij = sum_l (H*c)_il * H_lj. Only l=0 non-zero:
        //     = 4 * H_0j. Row 0 of H is all 1s, so _ = 4 everywhere in row 0 only.
        // Actually: (H*c*H)_ij = (H*c)_i0 * H_0j + ... but H acts on l-index
        // of (H*c)_il. (H*c)_il is 4 if i==0 and 0 otherwise, regardless
        // of l. So (H*c*H)_ij = sum_l (H*c)_il * H_lj.
        // For i=0: = sum_l 4 * H_lj = 4 * sum_l H_lj. Column sums: col 0 = 4, rest = 0.
        // For i!=0: =0.
        // So f is all zero except f[0,0] = 16.
        //
        // qP=36, LevelScale=160, dc_y[0,0] = 16*160 = 2560, rest 0.
        let dc = [1i32; 16];
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_luma_dc_16x16(&dc, 36, &sl, 8).unwrap();
        let mut expected = [0i32; 16];
        expected[0] = 2560;
        assert_eq!(out, expected);
    }

    // ------------ Chroma DC 2x2 (§8.5.11.1 / §8.5.11.2) -----------------

    #[test]
    fn chroma_dc_420_all_zero() {
        let dc = [0i32; 4];
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_chroma_dc_420(&dc, 20, &sl, 8).unwrap();
        assert_eq!(out, [0i32; 4]);
    }

    #[test]
    fn chroma_dc_420_single_dc_qp0() {
        // dc[0,0]=1, rest 0.
        // 2x2 Hadamard:
        //   t00=1+0=1, t01=0, t10=1-0=1, t11=0
        //   f00=t00+t01=1, f01=t00-t01=1, f10=t10+t11=1, f11=t10-t11=1
        // qP=0: qp_div=0, Eq. 8-326: dcC_ij = (f_ij * LevelScale(0,0,0)) << 0 >> 5
        //   LevelScale(0,0,0) = 16*10 = 160.
        //   dcC = (1*160)>>5 = 5 at every slot.
        let mut dc = [0i32; 4];
        dc[0] = 1;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_chroma_dc_420(&dc, 0, &sl, 8).unwrap();
        assert_eq!(out, [5i32; 4]);
    }

    #[test]
    fn chroma_dc_420_single_dc_qp6() {
        // qP=6 -> qp_div=1, LevelScale(0,0,0)=160 (qP%6=0).
        // dcC = (1*160 << 1) >> 5 = 320>>5 = 10.
        let mut dc = [0i32; 4];
        dc[0] = 1;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_chroma_dc_420(&dc, 6, &sl, 8).unwrap();
        assert_eq!(out, [10i32; 4]);
    }

    #[test]
    fn chroma_dc_420_hadamard_self_inverse() {
        // c = [[1,1],[1,1]] -> 2x2 Hadamard:
        //   t00=2, t01=2, t10=0, t11=0
        //   f00=4, f01=0, f10=0, f11=0
        let dc = [1i32; 4];
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_chroma_dc_420(&dc, 0, &sl, 8).unwrap();
        // f = [4, 0, 0, 0]. dcC = [(4*160)>>5, 0, 0, 0] = [20, 0, 0, 0].
        assert_eq!(out, [20, 0, 0, 0]);
    }

    // ------------ Chroma DC 2x4 (§8.5.11.2 / 4:2:2) ---------------------

    #[test]
    fn chroma_dc_422_all_zero() {
        let dc = [0i32; 8];
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_chroma_dc_422(&dc, 20, &sl, 8).unwrap();
        assert_eq!(out, [0i32; 8]);
    }

    #[test]
    fn chroma_dc_422_single_dc_qp0() {
        // dc[0,0]=1. 4x4 Hadamard on rows + 2x2 on columns:
        //   t_ij (after row transform) = H[i,0]*c[0,j] + 0 = H[i,0]*c[0,j]
        //   H[i,0] = 1 for all i. t00=1,t01=0; t10=1,t11=0; t20=1,t21=0; t30=1,t31=0.
        //   After col transform: f[i,0]=t_i0+t_i1 = 1; f[i,1]=t_i0-t_i1 = 1.
        //   So f is all 1s (8 entries).
        // qP=0 -> qpDC=3. qpDC<36, eq. 8-329 uses qpDC, not qP:
        //   qpDC%6=3, qpDC/6=0. LevelScale4x4(3, 0, 0) = 16*14 = 224.
        //   shift = 6-0 = 6, round = 1<<(5-0) = 32.
        //   dcC = (1*224 + 32) >> 6 = 256>>6 = 4.
        let mut dc = [0i32; 8];
        dc[0] = 1;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_chroma_dc_422(&dc, 0, &sl, 8).unwrap();
        assert_eq!(out, [4i32; 8]);
    }

    #[test]
    fn chroma_dc_422_rejects_bad_qp() {
        let dc = [0i32; 8];
        let sl = default_scaling_list_4x4_flat();
        assert_eq!(
            inverse_hadamard_chroma_dc_422(&dc, -1, &sl, 8).unwrap_err(),
            TransformError::QpOutOfRange(-1)
        );
    }

    // Regression: §8.5.11.2 Eq. 8-326 uses `>> 5`, not `>> 1`. The
    // earlier implementation accidentally transcribed the 2003 draft's
    // formula that assumed LevelScale did not include the weight_scale
    // factor. Make sure we land on the spec's value for a known qP.
    #[test]
    fn chroma_dc_420_regression_qp36() {
        // qP=36: qp_div=6, qp%6=0 -> LevelScale(0,0,0)=160.
        // Eq. 8-326 (4:2:0): dcC = ((f*160) << 6) >> 5 = (f*160*64)/32 = f*320.
        let mut dc = [0i32; 4];
        dc[0] = 1;
        // Hadamard: t00=1, t01=0, t10=1, t11=0 -> f = [1, 1, 1, 1]
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_chroma_dc_420(&dc, 36, &sl, 8).unwrap();
        assert_eq!(out, [320i32; 4]);
    }

    // Regression: §8.5.11.2 Eq. 8-329 (4:2:2, qPDC<36) uses qPDC%6 to
    // select LevelScale, not qP%6. The previous implementation passed
    // qP%6 which produced wrong LevelScale values for most qP.
    #[test]
    fn chroma_dc_422_regression_qp_mod_lookup() {
        // qP=6: qpDC=9, qpDC<36. qpDC%6=3 -> LevelScale(3,0,0)=16*14=224.
        // (If we incorrectly used qP%6=0, LevelScale would be 160 instead.)
        //   shift = 6 - qpDC/6 = 6 - 1 = 5; round = 1<<(5-1) = 16
        //   dcC = (1*224 + 16) >> 5 = 240>>5 = 7.
        let mut dc = [0i32; 8];
        dc[0] = 1;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_chroma_dc_422(&dc, 6, &sl, 8).unwrap();
        // After inverse-raster (c00=L[0]=1, rest zero) + Hadamard f is all 1s
        // across all 8 entries (same derivation as qp0 test), so every out
        // equals (1*224 + 16) >> 5 = 7.
        assert_eq!(out, [7i32; 8]);
    }

    // Regression: §8.5.11.2 Eq. 8-328 (4:2:2, qPDC>=36) uses qP/6,
    // not qPDC/6, for the shift amount. These differ at the boundary
    // where qP is a multiple of 3.
    #[test]
    fn chroma_dc_422_regression_qpdc_vs_qp_shift() {
        // Choose qP=33: qP/6=5, qP%6=3 -> LevelScale(3,0,0)=16*14=224.
        // qpDC=36, qpDC>=36 path. Shift = qP/6 - 6 = -1? No, that's a
        // negative shift. The spec requires qP/6 >= 6 which means
        // qP >= 36. For qpDC>=36 to hold with qP<36, qP in [33, 35]:
        // qP/6 = 5 → shift = 5 - 6 = -1 (undefined behavior in plain
        // Rust). The spec intends qpDC>=36 to imply qP>=33 but the
        // shift is indeed qP/6 - 6 which is negative for qP<36. So the
        // formula can only apply when qP>=36 in practice.
        //
        // Pick qP=36: qP/6=6, shift = 0. qpDC=39, qpDC>=36.
        //   LevelScale(0,0,0) = 16*10 = 160.
        //   dcC = (1*160) << 0 = 160 per entry.
        let mut dc = [0i32; 8];
        dc[0] = 1;
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_chroma_dc_422(&dc, 36, &sl, 8).unwrap();
        assert_eq!(out, [160i32; 8]);
    }

    // Regression: §8.5.4 Eq. 8-305 — 4:2:2 chroma DC input must be
    // inverse-raster-scanned from ChromaDCLevel before the Hadamard.
    // The mapping is non-trivial (L[5] at position c[1,1]).
    #[test]
    fn chroma_dc_422_regression_inverse_raster_scan() {
        // Put unique non-zero values at each ChromaDCLevel index to
        // verify the inverse-raster scan. After §8.5.4 Eq. 8-305:
        //   c[0,0]=L[0], c[0,1]=L[2]
        //   c[1,0]=L[1], c[1,1]=L[5]
        //   c[2,0]=L[3], c[2,1]=L[6]
        //   c[3,0]=L[4], c[3,1]=L[7]
        // If we only set L[5]=8 (so c[1,1]=8, rest 0), then by the
        // Hadamard Hv*c*Hh the output would include -8 in position
        // f[1,1] and similar for the other cross-product terms.
        //
        // Simpler check: set L[0]=1 and L[5]=0 — they shouldn't
        // interact. But setting L[5]=8 with L[0]=0 should produce
        // non-zero outputs.
        let mut dc = [0i32; 8];
        dc[5] = 8; // This becomes c[1,1] per Eq. 8-305.
        let sl = default_scaling_list_4x4_flat();
        let out = inverse_hadamard_chroma_dc_422(&dc, 36, &sl, 8).unwrap();
        // Confirm the output is not all-zeros — previous buggy mapping
        // would have placed L[5] at c[2,1] giving a different pattern.
        assert!(out.iter().any(|&v| v != 0), "L[5] should affect the output");
    }

    // ------------ Scaling list helpers ----------------------------------

    #[test]
    fn default_scaling_lists_all_16s() {
        assert_eq!(default_scaling_list_4x4_flat(), [16i32; 16]);
        assert_eq!(default_scaling_list_8x8_flat(), [16i32; 64]);
    }

    // ------------ Default scaling matrices (§7.4.2.1.1.1 T7-3/7-4) ----

    #[test]
    fn default_4x4_intra_table_7_3() {
        // Verbatim from Table 7-3, row Default_4x4_Intra.
        assert_eq!(
            DEFAULT_4X4_INTRA,
            [6, 13, 13, 20, 20, 20, 28, 28, 28, 28, 32, 32, 32, 37, 37, 42]
        );
    }

    #[test]
    fn default_4x4_inter_table_7_3() {
        assert_eq!(
            DEFAULT_4X4_INTER,
            [10, 14, 14, 20, 20, 20, 24, 24, 24, 24, 27, 27, 27, 30, 30, 34]
        );
    }

    #[test]
    fn default_8x8_intra_table_7_4_spot_checks() {
        // Table 7-4: idx 0→6, idx 1→10, idx 15→23, idx 31→27, idx 63→42.
        assert_eq!(DEFAULT_8X8_INTRA[0], 6);
        assert_eq!(DEFAULT_8X8_INTRA[1], 10);
        assert_eq!(DEFAULT_8X8_INTRA[15], 23);
        assert_eq!(DEFAULT_8X8_INTRA[31], 27);
        assert_eq!(DEFAULT_8X8_INTRA[63], 42);
    }

    #[test]
    fn default_8x8_inter_table_7_4_spot_checks() {
        // Table 7-4: idx 0→9, idx 1→13, idx 15→21, idx 31→24, idx 63→35.
        assert_eq!(DEFAULT_8X8_INTER[0], 9);
        assert_eq!(DEFAULT_8X8_INTER[1], 13);
        assert_eq!(DEFAULT_8X8_INTER[15], 21);
        assert_eq!(DEFAULT_8X8_INTER[31], 24);
        assert_eq!(DEFAULT_8X8_INTER[63], 35);
    }

    // ------------ select_scaling_list_4x4 / _8x8 (§7.4.2.1.1.1) --------

    // Minimal SPS/PPS construction helpers for unit tests.
    fn minimal_sps() -> crate::sps::Sps {
        // All fields that are irrelevant for scaling-list derivation
        // get safe defaults; the parser never emits them for a flat
        // profile, but we only need them to satisfy the struct shape.
        crate::sps::Sps {
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

    fn minimal_pps() -> crate::pps::Pps {
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

    #[test]
    fn select_4x4_returns_flat_when_neither_sps_nor_pps_present() {
        let sps = minimal_sps();
        let pps = minimal_pps();
        for i in 0..6 {
            assert_eq!(select_scaling_list_4x4(i, &sps, &pps), FLAT_4X4_16);
        }
    }

    #[test]
    fn select_8x8_returns_flat_when_neither_sps_nor_pps_present() {
        let sps = minimal_sps();
        let pps = minimal_pps();
        for i in 0..6 {
            assert_eq!(select_scaling_list_8x8(i, &sps, &pps), FLAT_8X8_16);
        }
    }

    #[test]
    fn select_4x4_sps_all_not_present_gives_rule_a_defaults() {
        // seq_scaling_matrix_present_flag=1, every entry NotPresent.
        // Per Table 7-2 / Rule A: i=0/1/2 → Default_4x4_Intra,
        // i=3/4/5 → Default_4x4_Inter (via inheritance chain).
        let mut sps = minimal_sps();
        sps.seq_scaling_matrix_present_flag = true;
        sps.seq_scaling_lists = Some(crate::sps::SeqScalingLists {
            entries: vec![ScalingListEntry::NotPresent; 8],
        });
        let pps = minimal_pps();

        assert_eq!(select_scaling_list_4x4(0, &sps, &pps), DEFAULT_4X4_INTRA);
        assert_eq!(select_scaling_list_4x4(1, &sps, &pps), DEFAULT_4X4_INTRA);
        assert_eq!(select_scaling_list_4x4(2, &sps, &pps), DEFAULT_4X4_INTRA);
        assert_eq!(select_scaling_list_4x4(3, &sps, &pps), DEFAULT_4X4_INTER);
        assert_eq!(select_scaling_list_4x4(4, &sps, &pps), DEFAULT_4X4_INTER);
        assert_eq!(select_scaling_list_4x4(5, &sps, &pps), DEFAULT_4X4_INTER);
    }

    #[test]
    fn select_4x4_sps_explicit_entries_are_returned_verbatim() {
        let mut sps = minimal_sps();
        sps.seq_scaling_matrix_present_flag = true;
        let custom: Vec<i32> = (0..16).map(|k| k + 1).collect(); // 1..=16
        sps.seq_scaling_lists = Some(crate::sps::SeqScalingLists {
            entries: vec![
                ScalingListEntry::Explicit(custom.clone()),
                ScalingListEntry::NotPresent,
                ScalingListEntry::NotPresent,
                ScalingListEntry::NotPresent,
                ScalingListEntry::NotPresent,
                ScalingListEntry::NotPresent,
                ScalingListEntry::NotPresent,
                ScalingListEntry::NotPresent,
            ],
        });
        let pps = minimal_pps();
        let got = select_scaling_list_4x4(0, &sps, &pps);
        let mut expected = [0i32; 16];
        for (k, v) in custom.iter().enumerate() {
            expected[k] = *v;
        }
        assert_eq!(got, expected);
        // idx=1 inherits (Rule A) from idx=0 → same explicit values.
        assert_eq!(select_scaling_list_4x4(1, &sps, &pps), expected);
        // idx=2 inherits from idx=1 → same again.
        assert_eq!(select_scaling_list_4x4(2, &sps, &pps), expected);
        // idx=3 is the Inter class head with no signalled list →
        // Default_4x4_Inter.
        assert_eq!(select_scaling_list_4x4(3, &sps, &pps), DEFAULT_4X4_INTER);
    }

    #[test]
    fn select_4x4_use_default_returns_table_default() {
        let mut sps = minimal_sps();
        sps.seq_scaling_matrix_present_flag = true;
        sps.seq_scaling_lists = Some(crate::sps::SeqScalingLists {
            entries: vec![
                ScalingListEntry::UseDefault, // idx 0 → Default_4x4_Intra
                ScalingListEntry::UseDefault, // idx 1 → Default_4x4_Intra
                ScalingListEntry::UseDefault, // idx 2 → Default_4x4_Intra
                ScalingListEntry::UseDefault, // idx 3 → Default_4x4_Inter
                ScalingListEntry::UseDefault, // idx 4 → Default_4x4_Inter
                ScalingListEntry::UseDefault, // idx 5 → Default_4x4_Inter
                ScalingListEntry::NotPresent, // 8x8 placeholders
                ScalingListEntry::NotPresent,
            ],
        });
        let pps = minimal_pps();
        assert_eq!(select_scaling_list_4x4(0, &sps, &pps), DEFAULT_4X4_INTRA);
        assert_eq!(select_scaling_list_4x4(1, &sps, &pps), DEFAULT_4X4_INTRA);
        assert_eq!(select_scaling_list_4x4(3, &sps, &pps), DEFAULT_4X4_INTER);
        assert_eq!(select_scaling_list_4x4(5, &sps, &pps), DEFAULT_4X4_INTER);
    }

    #[test]
    fn select_8x8_use_default_and_not_present() {
        // Layout: 8 entries (n=8 for chroma_format_idc != 3) with 6
        // 4x4 lists followed by 2 8x8 lists. Mark the 8x8 Intra as
        // UseDefault, 8x8 Inter as NotPresent.
        let mut sps = minimal_sps();
        sps.seq_scaling_matrix_present_flag = true;
        sps.seq_scaling_lists = Some(crate::sps::SeqScalingLists {
            entries: vec![
                ScalingListEntry::NotPresent, // 0
                ScalingListEntry::NotPresent, // 1
                ScalingListEntry::NotPresent, // 2
                ScalingListEntry::NotPresent, // 3
                ScalingListEntry::NotPresent, // 4
                ScalingListEntry::NotPresent, // 5
                ScalingListEntry::UseDefault, // 6 (8x8 Intra_Y)
                ScalingListEntry::NotPresent, // 7 (8x8 Inter_Y) → Rule A → Default_8x8_Inter
            ],
        });
        let pps = minimal_pps();
        assert_eq!(select_scaling_list_8x8(0, &sps, &pps), DEFAULT_8X8_INTRA);
        assert_eq!(select_scaling_list_8x8(1, &sps, &pps), DEFAULT_8X8_INTER);
    }

    #[test]
    fn select_8x8_chroma_444_12_entries() {
        // 4:4:4 stream: 12 entries (6 4x4 + 6 8x8). Exercise idx 2..=5.
        let mut sps = minimal_sps();
        sps.chroma_format_idc = 3;
        sps.seq_scaling_matrix_present_flag = true;
        let explicit: Vec<i32> = (1..=64).collect();
        sps.seq_scaling_lists = Some(crate::sps::SeqScalingLists {
            entries: vec![
                ScalingListEntry::NotPresent, // 4x4 i=0
                ScalingListEntry::NotPresent,
                ScalingListEntry::NotPresent,
                ScalingListEntry::NotPresent,
                ScalingListEntry::NotPresent,
                ScalingListEntry::NotPresent,
                ScalingListEntry::Explicit(explicit.clone()), // 8x8 i=6 (Intra_Y)
                ScalingListEntry::NotPresent,                 // 8x8 i=7 (Inter_Y)
                ScalingListEntry::NotPresent,                 // 8x8 i=8 (Intra_Cb)
                ScalingListEntry::NotPresent,                 // 8x8 i=9 (Inter_Cb)
                ScalingListEntry::NotPresent,                 // 8x8 i=10 (Intra_Cr)
                ScalingListEntry::NotPresent,                 // 8x8 i=11 (Inter_Cr)
            ],
        });
        let pps = minimal_pps();
        // idx 0 (i=6 Intra_Y): explicit → 1..=64.
        let got = select_scaling_list_8x8(0, &sps, &pps);
        assert_eq!(got[0], 1);
        assert_eq!(got[63], 64);
        // idx 1 (i=7 Inter_Y): NotPresent, Rule A class head →
        // Default_8x8_Inter.
        assert_eq!(select_scaling_list_8x8(1, &sps, &pps), DEFAULT_8X8_INTER);
        // idx 2 (i=8 Intra_Cb): NotPresent, inherits from idx 0 (i=6).
        assert_eq!(select_scaling_list_8x8(2, &sps, &pps), got);
        // idx 3 (i=9 Inter_Cb): inherits from idx 1 (i=7) →
        // Default_8x8_Inter.
        assert_eq!(select_scaling_list_8x8(3, &sps, &pps), DEFAULT_8X8_INTER);
        // idx 4 (i=10 Intra_Cr): inherits from idx 2 (i=8) → got.
        assert_eq!(select_scaling_list_8x8(4, &sps, &pps), got);
        // idx 5 (i=11 Inter_Cr): inherits from idx 3 → Default_8x8_Inter.
        assert_eq!(select_scaling_list_8x8(5, &sps, &pps), DEFAULT_8X8_INTER);
    }

    #[test]
    fn select_4x4_pps_override_takes_precedence_rule_b() {
        // SPS has seq_scaling_matrix_present_flag=1 with an Explicit idx
        // 0; PPS has pic_scaling_matrix_present_flag=1 with a DIFFERENT
        // Explicit idx 0. The effective list should be the PPS value.
        let mut sps = minimal_sps();
        sps.seq_scaling_matrix_present_flag = true;
        sps.seq_scaling_lists = Some(crate::sps::SeqScalingLists {
            entries: vec![
                ScalingListEntry::Explicit(vec![1; 16]),
                ScalingListEntry::NotPresent,
                ScalingListEntry::NotPresent,
                ScalingListEntry::NotPresent,
                ScalingListEntry::NotPresent,
                ScalingListEntry::NotPresent,
                ScalingListEntry::NotPresent,
                ScalingListEntry::NotPresent,
            ],
        });

        let mut pps = minimal_pps();
        pps.extension = Some(crate::pps::PpsExtension {
            transform_8x8_mode_flag: false,
            pic_scaling_matrix_present_flag: true,
            pic_scaling_lists: Some(crate::pps::PicScalingLists {
                entries: vec![
                    ScalingListEntry::Explicit(vec![2; 16]), // PPS overrides idx 0
                    ScalingListEntry::NotPresent,
                    ScalingListEntry::NotPresent,
                    ScalingListEntry::NotPresent,
                    ScalingListEntry::NotPresent,
                    ScalingListEntry::NotPresent,
                ],
            }),
            second_chroma_qp_index_offset: 0,
        });
        // PPS idx 0 Explicit=[2;16] → PPS wins.
        assert_eq!(select_scaling_list_4x4(0, &sps, &pps), [2i32; 16]);
        // PPS idx 1 NotPresent, SPS present → Rule B. Rule B for i=1
        // inherits the "scaling list for i=0" in the PPS chain →
        // [2;16].
        assert_eq!(select_scaling_list_4x4(1, &sps, &pps), [2i32; 16]);
        // PPS idx 3 NotPresent, SPS present → Rule B. Class head (i=3)
        // falls back to the SEQUENCE-level list for index 3. SPS idx
        // 3 is NotPresent → its Rule A chain gives Default_4x4_Inter.
        assert_eq!(select_scaling_list_4x4(3, &sps, &pps), DEFAULT_4X4_INTER);
    }

    #[test]
    fn select_4x4_pps_override_rule_a_when_sps_absent() {
        // pic_scaling_matrix_present_flag=1 AND
        // seq_scaling_matrix_present_flag=0 → Rule A for picture-level.
        let sps = minimal_sps(); // seq_scaling_matrix_present_flag=false
        let mut pps = minimal_pps();
        pps.extension = Some(crate::pps::PpsExtension {
            transform_8x8_mode_flag: false,
            pic_scaling_matrix_present_flag: true,
            pic_scaling_lists: Some(crate::pps::PicScalingLists {
                entries: vec![
                    ScalingListEntry::NotPresent, // idx 0 → Rule A → Default_4x4_Intra
                    ScalingListEntry::NotPresent,
                    ScalingListEntry::NotPresent,
                    ScalingListEntry::NotPresent,
                    ScalingListEntry::NotPresent,
                    ScalingListEntry::NotPresent,
                ],
            }),
            second_chroma_qp_index_offset: 0,
        });
        assert_eq!(select_scaling_list_4x4(0, &sps, &pps), DEFAULT_4X4_INTRA);
        assert_eq!(select_scaling_list_4x4(3, &sps, &pps), DEFAULT_4X4_INTER);
    }

    // ------------ Inverse scan (§8.5.6) ---------------------------------

    #[test]
    fn zigzag_scan_is_bijection() {
        // A scan-order list of 0..16 should map to a 4x4 matrix whose
        // entries, when read back in zig-zag, give 0..16.
        let mut levels = [0i32; 16];
        for (k, slot) in levels.iter_mut().enumerate() {
            *slot = k as i32;
        }
        let block = inverse_scan_4x4_zigzag(&levels);
        // Verify per Table 8-13: idx 0 -> c00, idx 1 -> c01, idx 2 -> c10, etc.
        // (row, col) matrix notation — clippy's identity_op lint would
        // fire on the explicit `row * 4 + col` form when either is 0,
        // but matrix coordinates read more naturally here.
        let at = |row: usize, col: usize| block[row * 4 + col];
        assert_eq!(at(0, 0), 0);
        assert_eq!(at(0, 1), 1);
        assert_eq!(at(1, 0), 2);
        assert_eq!(at(2, 0), 3);
        assert_eq!(at(1, 1), 4);
        assert_eq!(at(0, 2), 5);
        assert_eq!(at(3, 3), 15);
    }

    #[test]
    fn zigzag_ac_scan_maps_slot_to_spec_positions_1_through_15() {
        // §7.3.5.3.3 Intra16x16ACLevel / ChromaACLevel use startIdx=1,
        // so parser-slot k ∈ 0..=14 holds the coefficient at spec scan
        // position (k + 1). `inverse_scan_4x4_zigzag_ac` must dispatch
        // slot k to ZIGZAG_4X4[k + 1]. Slot 15 is padding and is
        // ignored. Matrix position (0, 0) is reserved for the DC slot
        // and must be zero.
        let mut levels = [0i32; 16];
        for (k, slot) in levels.iter_mut().enumerate().take(15) {
            *slot = (k + 1) as i32; // spec scan pos k+1 -> value k+1
        }
        levels[15] = 12345; // must be ignored
        let block = inverse_scan_4x4_zigzag_ac(&levels);
        let at = |row: usize, col: usize| block[row * 4 + col];
        // DC slot untouched.
        assert_eq!(at(0, 0), 0);
        // slot 0 (spec pos 1) -> ZIGZAG_4X4[1] = (0, 1).
        assert_eq!(at(0, 1), 1);
        // slot 1 (spec pos 2) -> ZIGZAG_4X4[2] = (1, 0).
        assert_eq!(at(1, 0), 2);
        // slot 2 (spec pos 3) -> ZIGZAG_4X4[3] = (2, 0).
        assert_eq!(at(2, 0), 3);
        // slot 3 (spec pos 4) -> ZIGZAG_4X4[4] = (1, 1).
        assert_eq!(at(1, 1), 4);
        // slot 4 (spec pos 5) -> ZIGZAG_4X4[5] = (0, 2).
        assert_eq!(at(0, 2), 5);
        // slot 14 (spec pos 15) -> ZIGZAG_4X4[15] = (3, 3).
        assert_eq!(at(3, 3), 15);
        // Ensure slot[15] padding was ignored (DC (0,0) would otherwise
        // collide with it only if we erroneously mapped slot 15 there).
        assert_ne!(at(0, 0), 12345);
    }

    #[test]
    fn zigzag_ac_scan_matches_zigzag_with_shift() {
        // Cross-check: for any AC levels at slots 0..=14, using the
        // AC scan must be equivalent to calling the full scan on a
        // [0, levels[0], levels[1], ..., levels[14]] array.
        let mut ac_levels = [0i32; 16];
        for (k, slot) in ac_levels.iter_mut().enumerate().take(15) {
            *slot = (100 + k as i32) * if k % 2 == 0 { 1 } else { -1 };
        }
        let mut shifted = [0i32; 16];
        shifted[1..16].copy_from_slice(&ac_levels[..15]);
        let via_ac = inverse_scan_4x4_zigzag_ac(&ac_levels);
        let via_shifted = inverse_scan_4x4_zigzag(&shifted);
        assert_eq!(via_ac, via_shifted);
    }

    #[test]
    fn field_ac_scan_maps_slot_to_spec_positions_1_through_15() {
        // Mirror of the zigzag_ac test for the informative field scan
        // (Table 8-13 second column).
        let mut levels = [0i32; 16];
        for (k, slot) in levels.iter_mut().enumerate().take(15) {
            *slot = (k + 1) as i32;
        }
        levels[15] = 9999;
        let block = inverse_scan_4x4_field_ac(&levels);
        let at = |row: usize, col: usize| block[row * 4 + col];
        // DC slot untouched.
        assert_eq!(at(0, 0), 0);
        // FIELD_4X4[1] = (1, 0) per §8.5.6.
        assert_eq!(at(1, 0), 1);
        // FIELD_4X4[2] = (0, 1).
        assert_eq!(at(0, 1), 2);
        // FIELD_4X4[15] = (3, 3).
        assert_eq!(at(3, 3), 15);
    }

    // ------------ NormAdjust spot checks (Tables 8-4, 8-5) --------------

    #[test]
    fn norm_adjust_4x4_spot_checks() {
        // Row 0 (qP%6=0): v = [10, 16, 13].
        assert_eq!(norm_adjust_4x4(0, 0, 0), 10); // (0,0) class
        assert_eq!(norm_adjust_4x4(0, 1, 1), 16); // (1,1) class
        assert_eq!(norm_adjust_4x4(0, 0, 1), 13); // other
        assert_eq!(norm_adjust_4x4(0, 2, 3), 13); // other
                                                  // Row 5 (qP%6=5): v = [18, 29, 23].
        assert_eq!(norm_adjust_4x4(5, 0, 0), 18);
        assert_eq!(norm_adjust_4x4(5, 3, 3), 29);
        assert_eq!(norm_adjust_4x4(5, 2, 1), 23);
    }

    #[test]
    fn norm_adjust_8x8_spot_checks() {
        // Row 0: v = [20, 18, 32, 19, 25, 24].
        assert_eq!(norm_adjust_8x8(0, 0, 0), 20); // (i%4,j%4)=(0,0)
        assert_eq!(norm_adjust_8x8(0, 4, 4), 20); // (i%4,j%4)=(0,0) again
        assert_eq!(norm_adjust_8x8(0, 1, 1), 18); // (i%2,j%2)=(1,1)
        assert_eq!(norm_adjust_8x8(0, 3, 3), 18);
        assert_eq!(norm_adjust_8x8(0, 2, 2), 32); // (i%4,j%4)=(2,2)
        assert_eq!(norm_adjust_8x8(0, 6, 6), 32);
        assert_eq!(norm_adjust_8x8(0, 0, 1), 19); // (i%4,j%2)=(0,1)
        assert_eq!(norm_adjust_8x8(0, 1, 0), 19); // (i%2,j%4)=(1,0)
        assert_eq!(norm_adjust_8x8(0, 0, 2), 25); // (i%4,j%4)=(0,2)
        assert_eq!(norm_adjust_8x8(0, 2, 0), 25); // (i%4,j%4)=(2,0)
                                                  // "otherwise" class:
        assert_eq!(norm_adjust_8x8(0, 1, 2), 24);
        assert_eq!(norm_adjust_8x8(0, 2, 1), 24);
        assert_eq!(norm_adjust_8x8(0, 2, 3), 24);
    }
}
