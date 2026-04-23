//! §7.3.5 / §7.4.5 — macroblock_layer parsing.
//!
//! Spec-driven implementation of `macroblock_layer()` per ITU-T Rec.
//! H.264 (08/2024). Clean-room rewrite — only the ITU-T PDF is
//! consulted.
//!
//! Coverage focus (per the task scope):
//! * §7.3.5  — `macroblock_layer()` syntax: mb_type dispatch, I_PCM
//!   path, mb_pred / sub_mb_pred dispatch, coded_block_pattern (CAVLC
//!   `me(v)` / CABAC), transform_size_8x8_flag, mb_qp_delta, and the
//!   residual() walker.
//! * §7.4.5  — semantics (`MbPartPredMode`, `NumMbPart`,
//!   `CodedBlockPatternLuma` / `CodedBlockPatternChroma` mapping).
//! * Table 7-11 (I-slice mb_type), Table 7-13 (P/SP), Table 7-14 (B).
//! * Table 7-17 / 7-18 (sub_mb_type in P and B).
//! * §9.1.2 `me(v)` + Tables 9-4(a)/(b) for coded_block_pattern in CAVLC.
//! * §7.3.5.3 residual (plus §7.3.5.3.1 residual_block_cavlc — handled
//!   via [`crate::cavlc`] helpers).
//!
//! What is NOT covered and the justified shortcuts:
//!
//! * CABAC residual walker uses per-block sequential decoding with
//!   neighbour-CBF left/above = None. This is enough for bit-accurate
//!   standalone-block tests; full neighbour wiring lives above this
//!   module.
//! * The uncommon SI / Intra_8x8 subset + Intra_8x8 pred-mode stream is
//!   modelled but with simplified derived-value records; exotic variants
//!   fall through to `MbType::Reserved(raw)`.
//! * The chroma AC residual only iterates `4 * NumC8x8` 4x4 blocks for
//!   ChromaArrayType ∈ {1, 2} (4:2:0, 4:2:2). ChromaArrayType == 3
//!   (4:4:4) is not modelled in this pass.
//! * Neighbour-derived `nC` values for CAVLC coeff_token selection are
//!   approximated as `0` (table column 0) unless the caller overrides
//!   — this is spec-consistent when left + above are both unavailable
//!   (eq. 9-3 → 0), which is the case for the top-left MB every slice
//!   starts at.

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use crate::bitstream::{BitError, BitReader};
use crate::cabac::{CabacDecoder, CabacError};
use crate::cabac_ctx::{
    decode_coded_block_flag, decode_coded_block_pattern, decode_coeff_abs_level_minus1,
    decode_coeff_sign_flag, decode_intra_chroma_pred_mode, decode_last_significant_coeff_flag,
    decode_mb_qp_delta, decode_mb_type_b, decode_mb_type_i, decode_mb_type_p,
    decode_mvd_lx, decode_prev_intra_pred_mode_flag, decode_ref_idx_lx,
    decode_rem_intra_pred_mode, decode_significant_coeff_flag,
    decode_transform_size_8x8_flag, BlockType, CabacContexts, MvdComponent, NeighbourCtx,
    SliceKind,
};
use crate::cavlc::{parse_residual_block_cavlc, CavlcError, CoeffTokenContext};
use crate::mv_deriv::{neighbour_4x4_map, NeighbourSource};
use crate::pps::Pps;
use crate::slice_header::{SliceHeader, SliceType};
use crate::sps::Sps;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MacroblockLayerError {
    #[error("bitstream read failed: {0}")]
    Bitstream(#[from] BitError),
    #[error("CABAC decoding failed: {0}")]
    Cabac(#[from] CabacError),
    #[error("CAVLC residual block decode failed: {0}")]
    Cavlc(#[from] CavlcError),
    /// §7.4.5 — mb_type raw value out of Table 7-11/7-13/7-14 range.
    #[error("mb_type raw value {0} out of range for slice type")]
    MbTypeOutOfRange(u32),
    /// §7.3.5 — `coded_block_pattern` only carries values ≤ 47 for
    /// ChromaArrayType ∈ {1,2} and ≤ 15 for ChromaArrayType ∈ {0,3}.
    #[error("coded_block_pattern value {0} out of range")]
    CbpOutOfRange(u32),
    /// §7.4.5 — `mb_qp_delta` ∈ [-(26 + QpBdOffsetY/2), +25 + QpBdOffsetY/2].
    #[error("mb_qp_delta value {0} out of range")]
    MbQpDeltaOutOfRange(i32),
    /// §7.3.5 — `sub_mb_type` raw value out of Table 7-17/7-18 range.
    #[error("sub_mb_type raw value {0} out of range")]
    SubMbTypeOutOfRange(u32),
    /// Unsupported ChromaArrayType for the residual walker.
    #[error("ChromaArrayType {0} not supported in residual walker")]
    UnsupportedChromaArrayType(u32),
}

pub type McblResult<T> = Result<T, MacroblockLayerError>;

// ---------------------------------------------------------------------------
// Macroblock type enumeration (Tables 7-11, 7-13, 7-14).
// ---------------------------------------------------------------------------

/// Derived Intra_16x16 sub-type as encoded in rows 1..=24 of Table 7-11.
///
/// Each row specifies three derived parameters via the mb_type number:
/// - `pred_mode`: 0..=3 — `Intra16x16PredMode`.
/// - `cbp_luma`: 0 or 15 — `CodedBlockPatternLuma`.
/// - `cbp_chroma`: 0, 1 or 2 — `CodedBlockPatternChroma`.
///
/// See Table 7-11 on PDF page ~85 of ITU-T Rec. H.264 (08/2024).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Intra16x16 {
    pub pred_mode: u8,
    pub cbp_luma: u8,
    pub cbp_chroma: u8,
}

/// Macroblock type — picks the relevant slice-type table at parse time.
///
/// The variants we explicitly name are the ones the macroblock-layer
/// dispatch needs to examine (I_NxN for the 4x4/8x8 intra pred path,
/// I_PCM for the aligned-sample path, `P_8x8*` and `B_8x8` for the
/// sub_mb_pred path, the `BDirect16x16` and `*_Skip` variants for the
/// sub-partition mode derivation, and the `Intra16x16` variants since
/// they imply `cbp_chroma/luma` and `pred_mode` directly).
///
/// Any remaining mb_type value that we don't distinguish is stored as
/// `Reserved { raw, .. }` — the dispatch fields (num_mb_part, is_intra,
/// etc.) are precomputed so the caller can still route correctly.
/// §7.4.5 Table 7-13 / 7-14 — MbPartPredMode / SubMbPredMode values
/// used to gate ref_idx / mvd presence in mb_pred / sub_mb_pred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MbPartPredMode {
    /// Inter list-0 prediction.
    PredL0,
    /// Inter list-1 prediction.
    PredL1,
    /// Inter bipred (both list 0 and list 1).
    BiPred,
    /// B_Direct_16x16 / B_Skip / B_Direct_8x8 — no ref_idx / mvd.
    Direct,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MbType {
    // --- I / SI slices (Table 7-11) ---
    /// mb_type == 0 — Intra_4x4 or Intra_8x8 depending on
    /// `transform_size_8x8_flag`.
    INxN,
    /// mb_type ∈ 1..=24 — Intra_16x16 with derived params.
    Intra16x16(Intra16x16),
    /// mb_type == 25 — I_PCM.
    IPcm,

    // --- P / SP slices (Table 7-13) ---
    PL016x16,
    PL0L016x8,
    PL0L08x16,
    P8x8,
    /// `P_8x8ref0` — constrains all four ref_idx to 0 (entropy signal).
    P8x8Ref0,
    /// Implicit from mb_skip_run / mb_skip_flag.
    PSkip,

    // --- B slices (Table 7-14) ---
    BDirect16x16,
    BL016x16,
    BL116x16,
    BBi16x16,
    BL0L016x8,
    BL0L08x16,
    BL1L116x8,
    BL1L18x16,
    BL0L116x8,
    BL0L18x16,
    BL1L016x8,
    BL1L08x16,
    BL0Bi16x8,
    BL0Bi8x16,
    BL1Bi16x8,
    BL1Bi8x16,
    BBiL016x8,
    BBiL08x16,
    BBiL116x8,
    BBiL18x16,
    BBiBi16x8,
    BBiBi8x16,
    B8x8,
    /// Implicit from mb_skip_run / mb_skip_flag.
    BSkip,

    /// Any other legal mb_type (e.g. SI variants) we don't distinguish.
    /// The raw value is retained for post-hoc analysis.
    Reserved(u32),
}

impl MbType {
    /// §7.4.5 — true for I_PCM (byte-aligned raw sample path).
    pub fn is_i_pcm(&self) -> bool {
        matches!(self, MbType::IPcm)
    }

    /// §7.4.5 — true for I_NxN (intra 4x4 / 8x8 per
    /// `transform_size_8x8_flag`).
    pub fn is_i_nxn(&self) -> bool {
        matches!(self, MbType::INxN)
    }

    /// §7.4.5 — true for any Intra_16x16 variant.
    pub fn is_intra_16x16(&self) -> bool {
        matches!(self, MbType::Intra16x16(_))
    }

    /// §7.4.5 — true for any intra macroblock.
    pub fn is_intra(&self) -> bool {
        matches!(self, MbType::INxN | MbType::Intra16x16(_) | MbType::IPcm)
    }

    /// §7.4.5 — true for P_Skip / B_Skip.
    pub fn is_skip(&self) -> bool {
        matches!(self, MbType::PSkip | MbType::BSkip)
    }

    /// §7.4.5 — `NumMbPart(mb_type)` — 1, 2, or 4. Intra mbs return 1.
    pub fn num_mb_part(&self) -> u8 {
        use MbType::*;
        match self {
            // Intra → 1.
            INxN | Intra16x16(_) | IPcm => 1,
            // P inter.
            PL016x16 | PSkip => 1,
            PL0L016x8 | PL0L08x16 => 2,
            P8x8 | P8x8Ref0 => 4,
            // B inter.
            BDirect16x16 | BSkip | BL016x16 | BL116x16 | BBi16x16 => 1,
            BL0L016x8 | BL0L08x16 | BL1L116x8 | BL1L18x16 | BL0L116x8 | BL0L18x16 | BL1L016x8
            | BL1L08x16 | BL0Bi16x8 | BL0Bi8x16 | BL1Bi16x8 | BL1Bi8x16 | BBiL016x8 | BBiL08x16
            | BBiL116x8 | BBiL18x16 | BBiBi16x8 | BBiBi8x16 => 2,
            B8x8 => 4,
            Reserved(_) => 1,
        }
    }

    /// Returns `true` when the sub_mb_pred path applies: `NumMbPart == 4`
    /// and not I_NxN / Intra_16x16 (§7.3.5).
    pub fn is_sub_mb_path(&self) -> bool {
        matches!(self, MbType::P8x8 | MbType::P8x8Ref0 | MbType::B8x8)
    }

    /// §7.4.5 — true when the MB is partitioned as 16x8 (two horizontal
    /// slabs). False for 8x16 (two vertical slabs) and all others.
    /// Used by the CABAC neighbour derivation to know which 8x8 / 4x4
    /// blocks a given mb_part covers.
    pub fn is_partitioned_16x8(&self) -> bool {
        use MbType::*;
        matches!(
            self,
            PL0L016x8
                | BL0L016x8
                | BL1L116x8
                | BL0L116x8
                | BL1L016x8
                | BL0Bi16x8
                | BL1Bi16x8
                | BBiL016x8
                | BBiL116x8
                | BBiBi16x8
        )
    }

    /// §7.4.5 Tables 7-13 / 7-14 — `MbPartPredMode(mb_type, mbPartIdx)`.
    ///
    /// Returns `None` for mbPartIdx ≥ NumMbPart, for sub-mb path MBs
    /// (where the per-partition mode is carried by sub_mb_type), and for
    /// intra MBs that don't have a list-based prediction mode.
    pub fn mb_part_pred_mode(&self, mb_part_idx: usize) -> Option<MbPartPredMode> {
        use MbPartPredMode::*;
        use MbType::*;
        if mb_part_idx >= self.num_mb_part() as usize {
            return None;
        }
        match self {
            // P inter (Table 7-13).
            PL016x16 | PSkip => Some(PredL0),
            PL0L016x8 | PL0L08x16 => Some(PredL0),
            // P_8x8 / P_8x8ref0: sub_mb path uses sub_mb_type; here we
            // return None so the caller falls back to sub_mb_pred.
            P8x8 | P8x8Ref0 => None,
            // B inter (Table 7-14).
            BDirect16x16 | BSkip => Some(Direct),
            BL016x16 => Some(PredL0),
            BL116x16 => Some(PredL1),
            BBi16x16 => Some(BiPred),
            BL0L016x8 | BL0L08x16 => Some(PredL0),
            BL1L116x8 | BL1L18x16 => Some(PredL1),
            BL0L116x8 | BL0L18x16 => {
                if mb_part_idx == 0 {
                    Some(PredL0)
                } else {
                    Some(PredL1)
                }
            }
            BL1L016x8 | BL1L08x16 => {
                if mb_part_idx == 0 {
                    Some(PredL1)
                } else {
                    Some(PredL0)
                }
            }
            BL0Bi16x8 | BL0Bi8x16 => {
                if mb_part_idx == 0 {
                    Some(PredL0)
                } else {
                    Some(BiPred)
                }
            }
            BL1Bi16x8 | BL1Bi8x16 => {
                if mb_part_idx == 0 {
                    Some(PredL1)
                } else {
                    Some(BiPred)
                }
            }
            BBiL016x8 | BBiL08x16 => {
                if mb_part_idx == 0 {
                    Some(BiPred)
                } else {
                    Some(PredL0)
                }
            }
            BBiL116x8 | BBiL18x16 => {
                if mb_part_idx == 0 {
                    Some(BiPred)
                } else {
                    Some(PredL1)
                }
            }
            BBiBi16x8 | BBiBi8x16 => Some(BiPred),
            B8x8 => None,
            INxN | Intra16x16(_) | IPcm | Reserved(_) => None,
        }
    }

    /// §7.4.5 — `CodedBlockPatternLuma` for Intra_16x16 variants.
    pub fn intra_16x16_cbp_luma(&self) -> Option<u8> {
        if let MbType::Intra16x16(i) = self {
            Some(i.cbp_luma)
        } else {
            None
        }
    }

    /// §7.4.5 — `CodedBlockPatternChroma` for Intra_16x16 variants.
    pub fn intra_16x16_cbp_chroma(&self) -> Option<u8> {
        if let MbType::Intra16x16(i) = self {
            Some(i.cbp_chroma)
        } else {
            None
        }
    }

    /// Table 7-11 row → decoded `MbType` for an I slice.
    pub fn from_i_slice(raw: u32) -> McblResult<Self> {
        match raw {
            0 => Ok(MbType::INxN),
            1..=24 => {
                // Rows 1..=24 encode (pred_mode, cbp_luma, cbp_chroma).
                // Rows 1..=4:   cbp_luma=0, cbp_chroma=0, pred_mode=row-1.
                // Rows 5..=8:   cbp_luma=0, cbp_chroma=1, pred_mode=row-5.
                // Rows 9..=12:  cbp_luma=0, cbp_chroma=2, pred_mode=row-9.
                // Rows 13..=16: cbp_luma=15,cbp_chroma=0, pred_mode=row-13.
                // Rows 17..=20: cbp_luma=15,cbp_chroma=1, pred_mode=row-17.
                // Rows 21..=24: cbp_luma=15,cbp_chroma=2, pred_mode=row-21.
                let r = raw - 1;
                let pred_mode = (r % 4) as u8;
                let group = r / 4;
                let (cbp_luma, cbp_chroma) = match group {
                    0 => (0, 0),
                    1 => (0, 1),
                    2 => (0, 2),
                    3 => (15, 0),
                    4 => (15, 1),
                    5 => (15, 2),
                    _ => return Err(MacroblockLayerError::MbTypeOutOfRange(raw)),
                };
                Ok(MbType::Intra16x16(Intra16x16 {
                    pred_mode,
                    cbp_luma,
                    cbp_chroma,
                }))
            }
            25 => Ok(MbType::IPcm),
            _ => Err(MacroblockLayerError::MbTypeOutOfRange(raw)),
        }
    }

    /// Table 7-13 row → decoded `MbType` for a P/SP slice.
    /// Values 0..=4 are inter P types, 5..=30 are intra (remapped I-slice).
    pub fn from_p_slice(raw: u32) -> McblResult<Self> {
        Ok(match raw {
            0 => MbType::PL016x16,
            1 => MbType::PL0L016x8,
            2 => MbType::PL0L08x16,
            3 => MbType::P8x8,
            4 => MbType::P8x8Ref0,
            5..=30 => Self::from_i_slice(raw - 5)?,
            _ => return Err(MacroblockLayerError::MbTypeOutOfRange(raw)),
        })
    }

    /// Table 7-14 row → decoded `MbType` for a B slice.
    /// Values 0..=22 are inter B types, 23..=48 are intra (remapped I-slice).
    pub fn from_b_slice(raw: u32) -> McblResult<Self> {
        Ok(match raw {
            0 => MbType::BDirect16x16,
            1 => MbType::BL016x16,
            2 => MbType::BL116x16,
            3 => MbType::BBi16x16,
            4 => MbType::BL0L016x8,
            5 => MbType::BL0L08x16,
            6 => MbType::BL1L116x8,
            7 => MbType::BL1L18x16,
            8 => MbType::BL0L116x8,
            9 => MbType::BL0L18x16,
            10 => MbType::BL1L016x8,
            11 => MbType::BL1L08x16,
            12 => MbType::BL0Bi16x8,
            13 => MbType::BL0Bi8x16,
            14 => MbType::BL1Bi16x8,
            15 => MbType::BL1Bi8x16,
            16 => MbType::BBiL016x8,
            17 => MbType::BBiL08x16,
            18 => MbType::BBiL116x8,
            19 => MbType::BBiL18x16,
            20 => MbType::BBiBi16x8,
            21 => MbType::BBiBi8x16,
            22 => MbType::B8x8,
            23..=48 => Self::from_i_slice(raw - 23)?,
            _ => return Err(MacroblockLayerError::MbTypeOutOfRange(raw)),
        })
    }
}

// ---------------------------------------------------------------------------
// sub_mb_type enumeration (Tables 7-17, 7-18).
// ---------------------------------------------------------------------------

/// §7.4.5.2 — per-8x8-partition sub-macroblock type.
///
/// Names track Tables 7-17 and 7-18. For exotic values not covered in
/// the enum, we store the raw value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubMbType {
    // Table 7-17 (P slices)
    PL08x8,
    PL08x4,
    PL04x8,
    PL04x4,
    // Table 7-18 (B slices)
    BDirect8x8,
    BL08x8,
    BL18x8,
    BBi8x8,
    BL08x4,
    BL04x8,
    BL18x4,
    BL14x8,
    BBi8x4,
    BBi4x8,
    BL04x4,
    BL14x4,
    BBi4x4,
    Reserved(u32),
}

impl SubMbType {
    /// §7.4.5.2 — `NumSubMbPart(sub_mb_type)`.
    pub fn num_sub_mb_part(&self) -> u8 {
        use SubMbType::*;
        match self {
            PL08x8 | BDirect8x8 | BL08x8 | BL18x8 | BBi8x8 => 1,
            PL08x4 | PL04x8 | BL08x4 | BL04x8 | BL18x4 | BL14x8 | BBi8x4 | BBi4x8 => 2,
            PL04x4 | BL04x4 | BL14x4 | BBi4x4 => 4,
            Reserved(_) => 1,
        }
    }

    /// §7.4.5.2 — true when the sub_mb is partitioned as 8x4 (two
    /// horizontal sub-slabs). False for 4x8 and others.
    pub fn is_8x4(&self) -> bool {
        use SubMbType::*;
        matches!(self, PL08x4 | BL08x4 | BL18x4 | BBi8x4)
    }

    pub fn from_p(raw: u32) -> McblResult<Self> {
        Ok(match raw {
            0 => SubMbType::PL08x8,
            1 => SubMbType::PL08x4,
            2 => SubMbType::PL04x8,
            3 => SubMbType::PL04x4,
            _ => SubMbType::Reserved(raw),
        })
    }

    pub fn from_b(raw: u32) -> McblResult<Self> {
        Ok(match raw {
            0 => SubMbType::BDirect8x8,
            1 => SubMbType::BL08x8,
            2 => SubMbType::BL18x8,
            3 => SubMbType::BBi8x8,
            4 => SubMbType::BL08x4,
            5 => SubMbType::BL04x8,
            6 => SubMbType::BL18x4,
            7 => SubMbType::BL14x8,
            8 => SubMbType::BBi8x4,
            9 => SubMbType::BBi4x8,
            10 => SubMbType::BL04x4,
            11 => SubMbType::BL14x4,
            12 => SubMbType::BBi4x4,
            _ => SubMbType::Reserved(raw),
        })
    }

    /// §7.4.5.2 Tables 7-17 / 7-18 — `SubMbPredMode(sub_mb_type)`.
    pub fn sub_mb_pred_mode(&self) -> Option<MbPartPredMode> {
        use MbPartPredMode::*;
        use SubMbType::*;
        match self {
            // P (Table 7-17) — all Pred_L0.
            PL08x8 | PL08x4 | PL04x8 | PL04x4 => Some(PredL0),
            // B (Table 7-18).
            BDirect8x8 => Some(Direct),
            BL08x8 | BL08x4 | BL04x8 | BL04x4 => Some(PredL0),
            BL18x8 | BL18x4 | BL14x8 | BL14x4 => Some(PredL1),
            BBi8x8 | BBi8x4 | BBi4x8 | BBi4x4 => Some(BiPred),
            Reserved(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Sub-structures of `macroblock_layer()` parse result.
// ---------------------------------------------------------------------------

/// §7.3.5 — `pcm_sample_luma[256]` + `pcm_sample_chroma[...]` payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcmSamples {
    pub luma: Vec<u32>,
    pub chroma_cb: Vec<u32>,
    pub chroma_cr: Vec<u32>,
}

/// §7.3.5.1 / §7.4.5.1 — `mb_pred()` for non-sub_mb path.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MbPred {
    /// Intra_4x4 prev-flag + rem values for the 16 4x4 blocks.
    pub prev_intra4x4_pred_mode_flag: [bool; 16],
    pub rem_intra4x4_pred_mode: [u8; 16],
    /// Intra_8x8 prev-flag + rem values for the 4 8x8 blocks.
    pub prev_intra8x8_pred_mode_flag: [bool; 4],
    pub rem_intra8x8_pred_mode: [u8; 4],
    /// §7.4.5.1 — chroma intra pred mode 0..=3. Only read when
    /// ChromaArrayType ∈ {1, 2} (§7.3.5.1).
    pub intra_chroma_pred_mode: u8,
    /// For inter prediction: ref_idx_l0 / ref_idx_l1 for each of the
    /// `NumMbPart` partitions.
    pub ref_idx_l0: Vec<u32>,
    pub ref_idx_l1: Vec<u32>,
    /// MVD[mb_part][component] per partition (list 0 / list 1).
    pub mvd_l0: Vec<[i32; 2]>,
    pub mvd_l1: Vec<[i32; 2]>,
}

/// §7.3.5.2 / §7.4.5.2 — `sub_mb_pred()` payload.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SubMbPred {
    pub sub_mb_type: [SubMbType; 4],
    pub ref_idx_l0: [u32; 4],
    pub ref_idx_l1: [u32; 4],
    /// `mvd_lX[mbPart][subMbPart]` with mvd stored as `[x, y]`.
    /// Length 4 per partition (max case of 4x4 subdivision); shorter
    /// partitions populate only the first N entries.
    pub mvd_l0: [Vec<[i32; 2]>; 4],
    pub mvd_l1: [Vec<[i32; 2]>; 4],
}

// Required because `SubMbType` doesn't derive Default — but the fixed-
// length array `SubMbType[4]` needs some default. Pick `Reserved(0)`.
impl Default for SubMbType {
    fn default() -> Self {
        SubMbType::Reserved(0)
    }
}

/// Parsed macroblock_layer result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Macroblock {
    pub mb_type: MbType,
    /// Raw mb_type value as parsed from the bitstream (pre slice-table
    /// mapping), kept for diagnostics.
    pub mb_type_raw: u32,
    /// `Some` for the non-sub, non-PCM path.
    pub mb_pred: Option<MbPred>,
    /// `Some` when `NumMbPart == 4` and not an I_NxN / Intra_16x16 MB.
    pub sub_mb_pred: Option<SubMbPred>,
    /// `Some` for I_PCM.
    pub pcm_samples: Option<PcmSamples>,
    /// Derived `CodedBlockPattern` — bits 0..=3 luma, bits 4..=5 chroma.
    /// Equal to the Intra_16x16 derived values when that mb_type, and
    /// equal to 0 on P_Skip / B_Skip.
    pub coded_block_pattern: u32,
    /// §7.3.5 — only read when CodedBlockPatternLuma > 0 and I_NxN is
    /// not the mb_type; defaults to false otherwise.
    pub transform_size_8x8_flag: bool,
    /// §7.3.5 — mb_qp_delta. 0 when absent.
    pub mb_qp_delta: i32,
    /// §7.3.5.3 — luma residual: list of 4x4 blocks (16 for Intra_16x16
    /// AC, 16 for Intra_4x4 / inter 4x4, 4 for 8x8 transform).
    pub residual_luma: Vec<[i32; 16]>,
    /// §7.3.5.3 — Intra_16x16 DC block (16 coefficients in zigzag).
    pub residual_luma_dc: Option<[i32; 16]>,
    /// §7.3.5.3 — chroma DC (length 4 for 4:2:0, 8 for 4:2:2).
    pub residual_chroma_dc_cb: Vec<i32>,
    pub residual_chroma_dc_cr: Vec<i32>,
    /// §7.3.5.3 — chroma AC per 4x4 block.
    pub residual_chroma_ac_cb: Vec<[i32; 16]>,
    pub residual_chroma_ac_cr: Vec<[i32; 16]>,
    /// §7.3.5.3 — Cb plane residual in the 4:4:4 "coded like luma" path
    /// (ChromaArrayType == 3 only). Mirrors `residual_luma`: each entry
    /// is one 4x4 block (or sub-block of an 8x8 transform, padded to
    /// 16). Empty for ChromaArrayType != 3.
    pub residual_cb_luma_like: Vec<[i32; 16]>,
    /// §7.3.5.3 — Cr plane 4:4:4 luma-like residual. See
    /// `residual_cb_luma_like`.
    pub residual_cr_luma_like: Vec<[i32; 16]>,
    /// §7.3.5.3 — Cb Intra16x16 DC block when ChromaArrayType == 3 and
    /// mb_type is Intra_16x16.
    pub residual_cb_16x16_dc: Option<[i32; 16]>,
    /// §7.3.5.3 — Cr Intra16x16 DC block when ChromaArrayType == 3 and
    /// mb_type is Intra_16x16.
    pub residual_cr_16x16_dc: Option<[i32; 16]>,
    /// True if this entry represents a P_Skip / B_Skip MB, in which
    /// case none of the residual / pred fields carry meaningful data.
    pub is_skip: bool,
}

impl Macroblock {
    /// Build a skipped-MB record for the given slice type.
    pub fn new_skip(slice_type: SliceType) -> Self {
        let mb_type = match slice_type {
            SliceType::B => MbType::BSkip,
            SliceType::P | SliceType::SP => MbType::PSkip,
            // I/SI slices never skip — but keep a defined value so the
            // test harness can still represent an implicit skip.
            _ => MbType::PSkip,
        };
        Self {
            mb_type,
            mb_type_raw: 0,
            mb_pred: None,
            sub_mb_pred: None,
            pcm_samples: None,
            coded_block_pattern: 0,
            transform_size_8x8_flag: false,
            mb_qp_delta: 0,
            residual_luma: Vec::new(),
            residual_luma_dc: None,
            residual_chroma_dc_cb: Vec::new(),
            residual_chroma_dc_cr: Vec::new(),
            residual_chroma_ac_cb: Vec::new(),
            residual_chroma_ac_cr: Vec::new(),
            residual_cb_luma_like: Vec::new(),
            residual_cr_luma_like: Vec::new(),
            residual_cb_16x16_dc: None,
            residual_cr_16x16_dc: None,
            is_skip: true,
        }
    }
}

// ---------------------------------------------------------------------------
// me(v) mapping for coded_block_pattern (Tables 9-4(a) / 9-4(b)).
// ---------------------------------------------------------------------------

/// §9.1.2 + Tables 9-4(a)/(b) — `me(v)` mapping of codeNum to CBP.
///
/// Table 9-4(a) — ChromaArrayType ∈ {1, 2} (4:2:0 / 4:2:2).
/// The table has 48 rows (codeNum 0..=47); the mapped value is the
/// (coded_block_pattern_chroma << 4) | coded_block_pattern_luma, where
/// luma ∈ 0..=15 and chroma ∈ 0..=2.
///
/// The table entries are split by `is_intra` (intra mb vs inter mb)
/// since the spec gives two separate mappings. Verbatim from Table 9-4
/// (PDF page ~216).
#[rustfmt::skip]
static ME_INTRA_420_422: [u8; 48] = [
    // codeNum 0..=47 → CBP value for intra mb (ChromaArrayType 1 or 2).
    47, 31, 15,  0, 23, 27, 29, 30,  7, 11, 13, 14, 39, 43, 45, 46,
    16,  3,  5, 10, 12, 19, 21, 26, 28, 35, 37, 42, 44,  1,  2,  4,
     8, 17, 18, 20, 24,  6,  9, 22, 25, 32, 33, 34, 36, 40, 38, 41,
];

#[rustfmt::skip]
static ME_INTER_420_422: [u8; 48] = [
    // codeNum 0..=47 → CBP value for inter mb (ChromaArrayType 1 or 2).
    // Transcribed verbatim from ITU-T H.264 (08/2024) Table 9-4(a),
    // "Inter" column.
     0, 16,  1,  2,  4,  8, 32,  3,  5, 10, 12, 15, 47,  7, 11, 13,
    14,  6,  9, 31, 35, 37, 42, 44, 33, 34, 36, 40, 39, 43, 45, 46,
    17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
];

// Table 9-4(b): ChromaArrayType ∈ {0, 3} — CBP is just 4-bit luma.
#[rustfmt::skip]
static ME_INTRA_0_3: [u8; 16] = [
    15,  0,  7, 11, 13, 14,  3,  5, 10, 12,  1,  2,  4,  8,  6,  9,
];

#[rustfmt::skip]
static ME_INTER_0_3: [u8; 16] = [
     0,  1,  2,  4,  8,  3,  5, 10, 12, 15,  7, 11, 13, 14,  6,  9,
];

/// §9.1.2 — `me(v)` for coded_block_pattern. Reads ue(v), indexes the
/// appropriate table per (ChromaArrayType, is_intra).
fn parse_cbp_me(r: &mut BitReader<'_>, chroma_array_type: u32, is_intra: bool) -> McblResult<u32> {
    let k = r.ue()?;
    let val = if chroma_array_type == 1 || chroma_array_type == 2 {
        let idx = k as usize;
        if idx >= 48 {
            return Err(MacroblockLayerError::CbpOutOfRange(k));
        }
        if is_intra {
            ME_INTRA_420_422[idx]
        } else {
            ME_INTER_420_422[idx]
        }
    } else {
        let idx = k as usize;
        if idx >= 16 {
            return Err(MacroblockLayerError::CbpOutOfRange(k));
        }
        if is_intra {
            ME_INTRA_0_3[idx]
        } else {
            ME_INTER_0_3[idx]
        }
    };
    Ok(val as u32)
}

// ---------------------------------------------------------------------------
// Per-MB CAVLC neighbour state (§9.2.1.1).
// ---------------------------------------------------------------------------

/// Per-MB total_coeff values for CAVLC neighbour lookup (§9.2.1.1).
///
/// For each decoded MB the CAVLC engine records the per-4x4-block
/// `TotalCoeff(coeff_token)` values needed by the `nC` derivation of
/// §9.2.1.1 for the next MB that queries them.
///
/// Indexing:
/// * `luma_total_coeff[i]` — `i` is the luma 4x4 block index per
///   §6.4.3 Figure 6-10 (raster-scan, so 0..=15).
/// * `cb_total_coeff[i]` / `cr_total_coeff[i]` — `i` is the chroma 4x4
///   block index per §6.4.7 (4 blocks for 4:2:0, 8 for 4:2:2).
///
/// `is_available` gates reads: an un-decoded neighbour MB must report
/// `availableFlagN == 0` per §9.2.1.1 step 5.
#[derive(Debug, Clone, Copy)]
pub struct CavlcMbNc {
    /// §6.4.4 / §9.2.1.1 step 5 — true once this MB has been decoded
    /// and may be consulted as a neighbour by subsequent MBs.
    pub is_available: bool,
    /// §9.2.1.1 step 6 — `P_Skip` / `B_Skip` → `nN = 0`.
    pub is_skip: bool,
    /// §9.2.1.1 step 6 — `I_PCM` → `nN = 16`.
    pub is_i_pcm: bool,
    /// §9.2.1.1 step 6 — true for intra MBs (governs the
    /// `constrained_intra_pred_flag` availability override).
    pub is_intra: bool,
    /// Per-luma-4x4-block `TotalCoeff(coeff_token)` values (§9.2.1.1
    /// NOTE 1: AC counts only — DC coefficients of Intra_16x16 blocks
    /// are tracked separately and do NOT contribute).
    pub luma_total_coeff: [u8; 16],
    /// Per-chroma-4x4-block `TotalCoeff(coeff_token)` values for Cb
    /// (sized for 4:2:2; 4:2:0 only uses the first 4 entries).
    pub cb_total_coeff: [u8; 8],
    /// Per-chroma-4x4-block `TotalCoeff(coeff_token)` values for Cr.
    pub cr_total_coeff: [u8; 8],
    /// §7.3.5.3 / §9.2.1.1 — per-4x4 Cb block `TotalCoeff` for
    /// ChromaArrayType == 3 (4:4:4), where chroma is coded like luma.
    /// 16 blocks, same indexing as `luma_total_coeff`.
    pub cb_luma_total_coeff: [u8; 16],
    /// §7.3.5.3 / §9.2.1.1 — per-4x4 Cr block `TotalCoeff` for
    /// ChromaArrayType == 3.
    pub cr_luma_total_coeff: [u8; 16],
}

impl Default for CavlcMbNc {
    fn default() -> Self {
        Self {
            is_available: false,
            is_skip: false,
            is_i_pcm: false,
            is_intra: false,
            luma_total_coeff: [0; 16],
            cb_total_coeff: [0; 8],
            cr_total_coeff: [0; 8],
            cb_luma_total_coeff: [0; 16],
            cr_luma_total_coeff: [0; 16],
        }
    }
}

/// Slice-scoped MB grid for CAVLC neighbour lookups (§9.2.1.1).
///
/// Sized `pic_size_in_mbs`. The caller is responsible for marking each
/// MB's slot `is_available = true` once the residual for that MB has
/// been decoded (§9.2.1.1 relies on `mbAddrN` having already been
/// decoded — our non-MBAFF raster-scan order guarantees that left and
/// above neighbours are always decoded before the current MB).
#[derive(Debug, Clone)]
pub struct CavlcNcGrid {
    pub pic_width_in_mbs: u32,
    pub pic_height_in_mbs: u32,
    pub mbs: Vec<CavlcMbNc>,
}

impl CavlcNcGrid {
    /// Allocate a fresh all-unavailable grid.
    pub fn new(pic_width_in_mbs: u32, pic_height_in_mbs: u32) -> Self {
        let len = (pic_width_in_mbs as usize) * (pic_height_in_mbs as usize);
        Self {
            pic_width_in_mbs,
            pic_height_in_mbs,
            mbs: vec![CavlcMbNc::default(); len],
        }
    }

    /// §6.4.9 — left (mbAddrA) and above (mbAddrB) neighbour MB
    /// addresses in raster scan. `None` when outside the picture.
    #[inline]
    pub fn neighbour_mb_addrs(&self, mb_addr: u32) -> (Option<u32>, Option<u32>) {
        let w = self.pic_width_in_mbs;
        if w == 0 {
            return (None, None);
        }
        let x = mb_addr % w;
        let y = mb_addr / w;
        let a = if x > 0 { Some(mb_addr - 1) } else { None };
        let b = if y > 0 { Some(mb_addr - w) } else { None };
        (a, b)
    }
}

/// §9.2.1.1 — luma kind selector for the nC derivation. `LumaAc` covers
/// `Intra16x16ACLevel` and `LumaLevel4x4` (both are 4x4 AC contexts);
/// `Intra16x16Dc` covers `Intra16x16DCLevel` (step 1 of §9.2.1.1:
/// `luma4x4BlkIdx = 0` for the DC block).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LumaNcKind {
    /// 4x4 AC / LumaLevel4x4 — indexed by the block's luma4x4BlkIdx.
    Ac,
    /// Intra16x16DCLevel — §9.2.1.1 step 1: `luma4x4BlkIdx = 0`.
    Intra16x16Dc,
}

/// §9.2.1.1 ordered steps 4..7 for luma blocks.
///
/// Given the current MB's neighbour grid state, derive `nC` for a 4x4
/// luma block at index `blk_idx` (0..=15). `constrained_intra_pred_flag`
/// is passed through, though our slice-data walker always runs in a
/// single slice and we conservatively treat slice-data-partitioning as
/// absent (so that condition never fires). For Intra16x16DCLevel the
/// spec fixes `luma4x4BlkIdx = 0` (§9.2.1.1 step 1); callers that want
/// that behaviour pass `kind = LumaNcKind::Intra16x16Dc` and the
/// `blk_idx` is ignored.
fn derive_nc_luma(
    grid: &CavlcNcGrid,
    current_mb_addr: u32,
    blk_idx: u8,
    kind: LumaNcKind,
    current_is_intra: bool,
    constrained_intra_pred_flag: bool,
) -> i32 {
    // §9.2.1.1 step 1: for Intra16x16DCLevel the spec fixes
    // luma4x4BlkIdx = 0.
    let blk = match kind {
        LumaNcKind::Intra16x16Dc => 0u8,
        LumaNcKind::Ac => blk_idx,
    };

    // §6.4.11.4 — derive blkA/blkB for this luma4x4BlkIdx.
    let [a_src, b_src, _c, _d] = neighbour_4x4_map(blk);
    let (a_mb_addr, a_blk) = resolve_luma_neighbour(grid, current_mb_addr, a_src);
    let (b_mb_addr, b_blk) = resolve_luma_neighbour(grid, current_mb_addr, b_src);

    // §9.2.1.1 step 5 — availability + step 6 — nN derivation.
    let (avail_a, n_a) = nc_nn_luma(
        grid,
        a_mb_addr,
        a_blk,
        current_is_intra,
        constrained_intra_pred_flag,
    );
    let (avail_b, n_b) = nc_nn_luma(
        grid,
        b_mb_addr,
        b_blk,
        current_is_intra,
        constrained_intra_pred_flag,
    );

    // §9.2.1.1 step 7 — combine.
    combine_nc(avail_a, n_a, avail_b, n_b)
}

/// Turn a `NeighbourSource` (from §6.4.11.4 via `mv_deriv`) into an
/// `(Option<mb_addr>, block_idx)` pair. `None` means the neighbour MB
/// lies outside the picture.
fn resolve_luma_neighbour(
    grid: &CavlcNcGrid,
    current_mb_addr: u32,
    src: NeighbourSource,
) -> (Option<u32>, u8) {
    let w = grid.pic_width_in_mbs;
    let x = if w == 0 { 0 } else { current_mb_addr % w };
    let y = if w == 0 { 0 } else { current_mb_addr / w };
    match src {
        NeighbourSource::InternalBlock(i) => (Some(current_mb_addr), i),
        NeighbourSource::ExternalLeft(i) => {
            if x > 0 {
                (Some(current_mb_addr - 1), i)
            } else {
                (None, i)
            }
        }
        NeighbourSource::ExternalAbove(i) => {
            if y > 0 {
                (Some(current_mb_addr - w), i)
            } else {
                (None, i)
            }
        }
        NeighbourSource::ExternalAboveRight(i) => {
            if y > 0 && x + 1 < w {
                (Some(current_mb_addr - w + 1), i)
            } else {
                (None, i)
            }
        }
        NeighbourSource::ExternalAboveLeft(i) => {
            if y > 0 && x > 0 {
                (Some(current_mb_addr - w - 1), i)
            } else {
                (None, i)
            }
        }
    }
}

/// §9.2.1.1 step 5 + step 6 for a luma 4x4 neighbour. Returns
/// `(availableFlagN, nN)`.
fn nc_nn_luma(
    grid: &CavlcNcGrid,
    mb_addr: Option<u32>,
    blk: u8,
    current_is_intra: bool,
    constrained_intra_pred_flag: bool,
) -> (bool, u8) {
    let Some(addr) = mb_addr else {
        return (false, 0);
    };
    let Some(info) = grid.mbs.get(addr as usize) else {
        return (false, 0);
    };
    if !info.is_available {
        // §9.2.1.1 step 5 — mbAddrN not (yet) available.
        return (false, 0);
    }
    // §9.2.1.1 step 5 second bullet: constrained_intra_pred_flag + slice
    // data partitioning interaction. Our decoder does not yet carry the
    // `nal_unit_type` through to here, and slice_data_partition_b/c NALs
    // are not parsed, so this branch is inactive — spec-consistent for
    // streams without partitioning.
    let _ = (current_is_intra, constrained_intra_pred_flag);
    // §9.2.1.1 step 6.
    if info.is_skip {
        // P_Skip / B_Skip — nN = 0.
        return (true, 0);
    }
    if info.is_i_pcm {
        // I_PCM — nN = 16.
        return (true, 16);
    }
    let idx = blk as usize;
    let n = info.luma_total_coeff.get(idx).copied().unwrap_or(0);
    (true, n)
}

/// §9.2.1.1 step 5 + step 6 for a Cb/Cr 4x4 neighbour in the 4:4:4
/// "coded like luma" path (ChromaArrayType == 3). Mirrors `nc_nn_luma`
/// but reads the per-plane `cb_luma_total_coeff` / `cr_luma_total_coeff`
/// arrays. Returns `(availableFlagN, nN)`.
fn nc_nn_plane_luma_like(
    grid: &CavlcNcGrid,
    mb_addr: Option<u32>,
    blk: u8,
    is_cr: bool,
    current_is_intra: bool,
    constrained_intra_pred_flag: bool,
) -> (bool, u8) {
    let Some(addr) = mb_addr else { return (false, 0) };
    let Some(info) = grid.mbs.get(addr as usize) else { return (false, 0) };
    if !info.is_available {
        return (false, 0);
    }
    let _ = (current_is_intra, constrained_intra_pred_flag);
    if info.is_skip {
        return (true, 0);
    }
    if info.is_i_pcm {
        return (true, 16);
    }
    let arr = if is_cr {
        &info.cr_luma_total_coeff
    } else {
        &info.cb_luma_total_coeff
    };
    let n = arr.get(blk as usize).copied().unwrap_or(0);
    (true, n)
}

/// §9.2.1.1 step 6/7 for a chroma AC 4x4 neighbour. The chroma block
/// layout is 2x1 (4:2:0) or 2x2 (4:2:2) of 4x4 blocks per 8x8 unit —
/// but since the ITU derivation is expressed in terms of spatial
/// (xN, yN) locations and Table 6-3, we implement the same neighbour
/// dispatch directly on the chroma block's (x, y) in the MB's chroma
/// grid (§6.4.7 eq. 6-21/6-22 for the inverse scan).
fn derive_nc_chroma_ac(
    grid: &CavlcNcGrid,
    current_mb_addr: u32,
    blk_idx: u8,
    is_cr: bool,
    chroma_array_type: u32,
    current_is_intra: bool,
    constrained_intra_pred_flag: bool,
) -> i32 {
    let max_w: i32 = 8; // MbWidthC for 4:2:0 / 4:2:2 (§6.4.12 eq. 6-32).
    let max_h: i32 = if chroma_array_type == 2 { 16 } else { 8 }; // eq. 6-33.

    // §6.4.7 eq. 6-21/6-22 — inverse 4x4 chroma block scan.
    // For 4:2:0 (indices 0..=3, layout 2x2):
    //   x = (idx % 2) * 4 ; y = (idx / 2) * 4.
    // For 4:2:2 (indices 0..=7, layout 2x4):
    //   InverseRasterScan(idx, 4, 4, 8, 0) = (idx % 2) * 4
    //   InverseRasterScan(idx, 4, 4, 8, 1) = (idx / 2) * 4
    let x = ((blk_idx as i32) % 2) * 4;
    let y = ((blk_idx as i32) / 2) * 4;

    // A: (xD, yD) = (-1, 0). B: ( 0, -1). (§6.4.11.5 step 1 → Table 6-2.)
    let (xa, ya) = (x - 1, y);
    let (xb, yb) = (x, y - 1);
    let (a_mb_addr, a_blk) =
        resolve_chroma_neighbour(grid, current_mb_addr, xa, ya, max_w, max_h);
    let (b_mb_addr, b_blk) =
        resolve_chroma_neighbour(grid, current_mb_addr, xb, yb, max_w, max_h);

    let (avail_a, n_a) = nc_nn_chroma(
        grid,
        a_mb_addr,
        a_blk,
        is_cr,
        current_is_intra,
        constrained_intra_pred_flag,
    );
    let (avail_b, n_b) = nc_nn_chroma(
        grid,
        b_mb_addr,
        b_blk,
        is_cr,
        current_is_intra,
        constrained_intra_pred_flag,
    );

    combine_nc(avail_a, n_a, avail_b, n_b)
}

/// §6.4.12 Table 6-3 dispatch for chroma locations + §6.4.13.2
/// eq. 6-39 wrap. Returns `(mbAddrN, chroma4x4BlkIdxN)`.
fn resolve_chroma_neighbour(
    grid: &CavlcNcGrid,
    current_mb_addr: u32,
    xn: i32,
    yn: i32,
    max_w: i32,
    max_h: i32,
) -> (Option<u32>, u8) {
    let w = grid.pic_width_in_mbs;
    let x = if w == 0 { 0 } else { current_mb_addr % w };
    let y = if w == 0 { 0 } else { current_mb_addr / w };

    // §6.4.12 Table 6-3.
    let (mb_opt, xw, yw) = if xn < 0 && yn < 0 {
        // mbAddrD — above-left.
        let addr = if y > 0 && x > 0 {
            Some(current_mb_addr - w - 1)
        } else {
            None
        };
        (addr, xn + max_w, yn + max_h)
    } else if xn < 0 && (0..max_h).contains(&yn) {
        // mbAddrA — left.
        let addr = if x > 0 { Some(current_mb_addr - 1) } else { None };
        (addr, xn + max_w, yn)
    } else if (0..max_w).contains(&xn) && yn < 0 {
        // mbAddrB — above.
        let addr = if y > 0 { Some(current_mb_addr - w) } else { None };
        (addr, xn, yn + max_h)
    } else if (0..max_w).contains(&xn) && (0..max_h).contains(&yn) {
        // Internal.
        (Some(current_mb_addr), xn, yn)
    } else if xn > max_w - 1 && yn < 0 {
        // mbAddrC — above-right.
        let addr = if y > 0 && x + 1 < w {
            Some(current_mb_addr - w + 1)
        } else {
            None
        };
        (addr, xn - max_w, yn + max_h)
    } else {
        // Not available — per Table 6-3 (e.g. xN past right edge within
        // the MB). Treat as unavailable.
        (None, 0, 0)
    };
    // §6.4.13.2 eq. 6-39 — chroma4x4BlkIdx from (xW, yW).
    let blk = if mb_opt.is_some() {
        (2 * (yw / 4) + (xw / 4)) as u8
    } else {
        0
    };
    (mb_opt, blk)
}

/// §9.2.1.1 step 5+6 for a chroma AC 4x4 neighbour.
fn nc_nn_chroma(
    grid: &CavlcNcGrid,
    mb_addr: Option<u32>,
    blk: u8,
    is_cr: bool,
    current_is_intra: bool,
    constrained_intra_pred_flag: bool,
) -> (bool, u8) {
    let Some(addr) = mb_addr else {
        return (false, 0);
    };
    let Some(info) = grid.mbs.get(addr as usize) else {
        return (false, 0);
    };
    if !info.is_available {
        return (false, 0);
    }
    let _ = (current_is_intra, constrained_intra_pred_flag);
    if info.is_skip {
        return (true, 0);
    }
    if info.is_i_pcm {
        return (true, 16);
    }
    let arr = if is_cr {
        &info.cr_total_coeff
    } else {
        &info.cb_total_coeff
    };
    let n = arr.get(blk as usize).copied().unwrap_or(0);
    (true, n)
}

/// §9.2.1.1 step 7 — combine the two neighbour nN values.
#[inline]
fn combine_nc(avail_a: bool, n_a: u8, avail_b: bool, n_b: u8) -> i32 {
    match (avail_a, avail_b) {
        (true, true) => (n_a as i32 + n_b as i32 + 1) >> 1,
        (true, false) => n_a as i32,
        (false, true) => n_b as i32,
        (false, false) => 0,
    }
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.6 / .7 / .9 — CABAC per-MB neighbour grid.
// ---------------------------------------------------------------------------

/// Per-MB decoded values needed for CABAC neighbour context lookups
/// (§9.3.3.1.1.6 ref_idx, §9.3.3.1.1.7 mvd, §9.3.3.1.1.9 coded_block_flag).
///
/// Populated by the MB parser at the end of each macroblock; consulted at
/// the start of the NEXT MB when deriving `ctxIdxInc` for inter / residual
/// syntax elements.
#[derive(Debug, Clone, Copy)]
pub struct CabacMbNeighbourInfo {
    /// §9.3.3.1.1.* — `availableFlagN`. The caller sets this `true`
    /// after the MB's parse completes; prior to that the slot reads as
    /// unavailable and all `condTermFlag*` derivations return 0 per spec.
    pub available: bool,
    /// §9.3.3.1.1.9 — `P_Skip` / `B_Skip` → `coded_block_flag[...] = 0`.
    pub is_skip: bool,
    /// §9.3.3.1.1.6 / .9 — intra macroblock (governs the condTerm
    /// initial value for some syntax elements per Table 9-35).
    pub is_intra: bool,
    /// §9.3.3.1.1.9 — I_PCM → all coded_block_flag values are 1.
    pub is_i_pcm: bool,
    /// Per-8x8 partition ref_idx_l0 (§9.3.3.1.1.6 condTerm uses
    /// `ref_idx_lX > 0`). Signed so that `-1` can mark "absent" (e.g.
    /// intra partitions in this MB, or direct 8x8 blocks). The
    /// comparison against `> 0` yields `false` for both `0` and `-1`.
    pub ref_idx_l0: [i8; 4],
    pub ref_idx_l1: [i8; 4],
    /// Per-4x4 block mvd components (§9.3.3.1.1.7 sums `|mvd|` from
    /// neighbour 4x4 blocks). Indexed by luma 4x4 block idx
    /// (§6.4.3 / Figure 6-10 raster order).
    pub mvd_l0_x: [i16; 16],
    pub mvd_l0_y: [i16; 16],
    pub mvd_l1_x: [i16; 16],
    pub mvd_l1_y: [i16; 16],
    /// Per-luma-4x4-block coded_block_flag (§9.3.3.1.1.9 Table 9-42
    /// ctxBlockCat = 2 — Luma4x4).
    pub cbf_luma_4x4: [bool; 16],
    /// §9.3.3.1.1.9 Table 9-42 ctxBlockCat = 3 — ChromaDC.
    pub cbf_cb_dc: bool,
    pub cbf_cr_dc: bool,
    /// §9.3.3.1.1.9 Table 9-42 ctxBlockCat = 4 — ChromaAC. Sized for
    /// 4:2:2 (8 blocks); 4:2:0 uses the first 4 entries.
    pub cbf_cb_ac: [bool; 8],
    pub cbf_cr_ac: [bool; 8],
    /// §9.3.3.1.1.9 Table 9-42 ctxBlockCat = 0 — Luma16x16DCLevel.
    pub cbf_luma_16x16_dc: bool,
    /// §9.3.3.1.1.9 Table 9-42 ctxBlockCat = 1 — Luma16x16ACLevel.
    pub cbf_luma_16x16_ac: [bool; 16],
    /// §9.3.3.1.1.9 Table 9-42 ctxBlockCat = 6 / 7 / 8 — Cb plane in
    /// 4:4:4 (coded like luma): 16x16 DC, 16x16 AC per 4x4, 4x4 and 8x8
    /// CBFs respectively. Populated when ChromaArrayType == 3.
    pub cbf_cb_16x16_dc: bool,
    pub cbf_cb_16x16_ac: [bool; 16],
    pub cbf_cb_luma_4x4: [bool; 16],
    /// §9.3.3.1.1.9 Table 9-42 ctxBlockCat = 10 / 11 / 12 — Cr plane in
    /// 4:4:4.
    pub cbf_cr_16x16_dc: bool,
    pub cbf_cr_16x16_ac: [bool; 16],
    pub cbf_cr_luma_4x4: [bool; 16],
    /// §9.3.3.1.1.4 — CodedBlockPatternLuma (bits 0..=3) for this MB,
    /// consulted by the next MB's CBP ctxIdxInc derivation.
    pub coded_block_pattern_luma: u8,
    /// §9.3.3.1.1.4 — CodedBlockPatternChroma for this MB (0, 1, or 2).
    pub coded_block_pattern_chroma: u8,
    /// §9.3.3.1.1.3 — mb_type == I_NxN (I_4x4 / I_8x8) for this MB,
    /// used by the next MB's mb_type ctxIdxInc derivation.
    pub is_i_nxn: bool,
    /// §9.3.3.1.1.8 — intra_chroma_pred_mode of this MB (0..=3). The
    /// next MB's ctxIdxInc derivation consults whether this is non-zero.
    pub intra_chroma_pred_mode: u8,
    /// §9.3.3.1.1.10 — transform_size_8x8_flag of this MB, used by the
    /// next MB's transform_size_8x8_flag ctxIdxInc derivation.
    pub transform_size_8x8_flag: bool,
}

impl Default for CabacMbNeighbourInfo {
    fn default() -> Self {
        Self {
            available: false,
            is_skip: false,
            is_intra: false,
            is_i_pcm: false,
            ref_idx_l0: [-1; 4],
            ref_idx_l1: [-1; 4],
            mvd_l0_x: [0; 16],
            mvd_l0_y: [0; 16],
            mvd_l1_x: [0; 16],
            mvd_l1_y: [0; 16],
            cbf_luma_4x4: [false; 16],
            cbf_cb_dc: false,
            cbf_cr_dc: false,
            cbf_cb_ac: [false; 8],
            cbf_cr_ac: [false; 8],
            cbf_luma_16x16_dc: false,
            cbf_luma_16x16_ac: [false; 16],
            cbf_cb_16x16_dc: false,
            cbf_cb_16x16_ac: [false; 16],
            cbf_cb_luma_4x4: [false; 16],
            cbf_cr_16x16_dc: false,
            cbf_cr_16x16_ac: [false; 16],
            cbf_cr_luma_4x4: [false; 16],
            coded_block_pattern_luma: 0,
            coded_block_pattern_chroma: 0,
            is_i_nxn: false,
            intra_chroma_pred_mode: 0,
            transform_size_8x8_flag: false,
        }
    }
}

/// Slice-scoped MB grid for CABAC neighbour lookups (§9.3.3.1.1.*).
///
/// Sized `pic_width_in_mbs * pic_height_in_mbs`. Mirrors the CAVLC
/// neighbour grid but carries the per-MB values consumed by CABAC
/// inter + residual syntax elements.
#[derive(Debug, Clone)]
pub struct CabacNeighbourGrid {
    pub width_in_mbs: u32,
    pub height_in_mbs: u32,
    pub mbs: Vec<CabacMbNeighbourInfo>,
}

impl CabacNeighbourGrid {
    /// Allocate a fresh all-unavailable grid of `width * height` MB slots.
    pub fn new(width_in_mbs: u32, height_in_mbs: u32) -> Self {
        let len = (width_in_mbs as usize) * (height_in_mbs as usize);
        Self {
            width_in_mbs,
            height_in_mbs,
            mbs: vec![CabacMbNeighbourInfo::default(); len],
        }
    }

    /// §6.4.9 — raster-scan `(mbAddrA, mbAddrB)` pair. Non-MBAFF only —
    /// our Phase-1 MBAFF falls back to this derivation per the task spec.
    #[inline]
    pub fn neighbour_mb_addrs(&self, mb_addr: u32) -> (Option<u32>, Option<u32>) {
        let w = self.width_in_mbs;
        if w == 0 {
            return (None, None);
        }
        let x = mb_addr % w;
        let y = mb_addr / w;
        let a = if x > 0 { Some(mb_addr - 1) } else { None };
        let b = if y > 0 { Some(mb_addr - w) } else { None };
        (a, b)
    }
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.6 — ref_idx_lX ctxIdxInc helper.
// ---------------------------------------------------------------------------

/// §9.3.3.1.1.6 — derive the (left, above) "`ref_idx_lX > 0`" condTerm
/// pair for 8x8 partition `part8x8` of the current MB.
///
/// Per Table 9-42-ish / eq. 9-14: `ctxIdxInc = condTermA + 2*condTermB`,
/// where `condTermN = 1` iff the neighbour 8x8 partition's `ref_idx_lX`
/// is strictly greater than 0. Unavailable / intra / skip neighbours
/// contribute 0 per the spec's "not available" fallback.
///
/// 8x8 partition layout inside the MB (each 8x8 covers 4 luma 4x4 blocks):
///   part 0 at (x=0, y=0)  — 4x4 blocks 0,1,2,3
///   part 1 at (x=8, y=0)  — 4x4 blocks 4,5,6,7
///   part 2 at (x=0, y=8)  — 4x4 blocks 8,9,10,11
///   part 3 at (x=8, y=8)  — 4x4 blocks 12,13,14,15
///
/// Neighbour 8x8 partition (per §6.4.11.4 for 8x8-sized blocks):
///   part 0: A = left-MB partition 1, B = above-MB partition 2.
///   part 1: A = current MB partition 0, B = above-MB partition 3.
///   part 2: A = left-MB partition 3, B = current MB partition 0.
///   part 3: A = current MB partition 2, B = current MB partition 1.
fn cabac_ref_idx_cond_terms(
    grid: &CabacNeighbourGrid,
    current_mb_addr: u32,
    current_mb: &CabacMbNeighbourInfo,
    list: u8,
    part8x8: u8,
) -> (bool, bool) {
    let w = grid.width_in_mbs;
    let x = if w == 0 { 0 } else { current_mb_addr % w };
    let y = if w == 0 { 0 } else { current_mb_addr / w };
    let left_mb = if x > 0 { grid.mbs.get((current_mb_addr - 1) as usize) } else { None };
    let above_mb = if y > 0 { grid.mbs.get((current_mb_addr - w) as usize) } else { None };

    // Map (part, direction) → (neighbour_mb, neighbour_part).
    // `None` for neighbour_mb means "current MB".
    let (a_mb, a_part): (Option<&CabacMbNeighbourInfo>, u8) = match part8x8 {
        0 => (left_mb, 1),
        1 => (Some(current_mb), 0),
        2 => (left_mb, 3),
        3 => (Some(current_mb), 2),
        _ => (None, 0),
    };
    let (b_mb, b_part): (Option<&CabacMbNeighbourInfo>, u8) = match part8x8 {
        0 => (above_mb, 2),
        1 => (above_mb, 3),
        2 => (Some(current_mb), 0),
        3 => (Some(current_mb), 1),
        _ => (None, 0),
    };

    let cond = |mb: Option<&CabacMbNeighbourInfo>, part: u8| -> bool {
        let Some(info) = mb else { return false };
        if !info.available { return false }
        if info.is_skip || info.is_intra { return false }
        let arr = if list == 0 { info.ref_idx_l0 } else { info.ref_idx_l1 };
        arr[part as usize] > 0
    };
    (cond(a_mb, a_part), cond(b_mb, b_part))
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.7 — mvd_lX ctxIdxInc helper.
// ---------------------------------------------------------------------------

/// §6.4.13.1 eq. 6-38 — 4x4 block index from (x, y) inside an MB.
#[inline]
fn blk4x4_idx(x: i32, y: i32) -> u8 {
    (8 * (y / 8) + 4 * (x / 8) + 2 * ((y % 8) / 4) + ((x % 8) / 4)) as u8
}

/// §6.4.3 Figure 6-10 inverse scan — upper-left (x, y) of 4x4 block idx.
#[inline]
fn blk4x4_xy(idx: u8) -> (i32, i32) {
    let hi = (idx / 4) as i32; // 8x8 block
    let lo = (idx % 4) as i32; // 4x4 inside 8x8
    let x_hi = (hi % 2) * 8;
    let y_hi = (hi / 2) * 8;
    let x_lo = (lo % 2) * 4;
    let y_lo = (lo / 2) * 4;
    (x_hi + x_lo, y_hi + y_lo)
}

/// §9.3.3.1.1.7 — derive `absMvdCompA + absMvdCompB` for the current
/// 4x4 block `blk4x4` of the given list / component. Skip / intra /
/// unavailable neighbours contribute 0 per the spec.
fn cabac_mvd_abs_sum(
    grid: &CabacNeighbourGrid,
    current_mb_addr: u32,
    current_mb: &CabacMbNeighbourInfo,
    list: u8,
    comp: MvdComponent,
    blk4x4: u8,
) -> u32 {
    let w = grid.width_in_mbs;
    let x0 = if w == 0 { 0 } else { current_mb_addr % w };
    let y0 = if w == 0 { 0 } else { current_mb_addr / w };
    let (bx, by) = blk4x4_xy(blk4x4);

    // Neighbour A: (bx-1, by); Neighbour B: (bx, by-1).
    let pick = |xn: i32, yn: i32| -> u32 {
        if xn >= 0 && yn >= 0 {
            // Internal to current MB.
            if !current_mb.available {
                // Current MB not yet marked available → the block has
                // not yet been decoded. For in-MB neighbours the mvd
                // will have been written into current_mb before we
                // query it (the parser fills mvds in raster order and
                // writes back per-block). So `available` doesn't gate
                // in-MB lookups — we still read the slot value.
            }
            let idx = blk4x4_idx(xn, yn) as usize;
            let v = match (list, comp) {
                (0, MvdComponent::X) => current_mb.mvd_l0_x[idx],
                (0, MvdComponent::Y) => current_mb.mvd_l0_y[idx],
                (1, MvdComponent::X) => current_mb.mvd_l1_x[idx],
                _                    => current_mb.mvd_l1_y[idx],
            };
            return v.unsigned_abs() as u32;
        }
        // External neighbour lookup.
        let (nbr_mb_addr, nxn, nyn) = if xn < 0 && yn >= 0 {
            if x0 == 0 { return 0 }
            (current_mb_addr - 1, xn + 16, yn)
        } else if xn >= 0 && yn < 0 {
            if y0 == 0 { return 0 }
            (current_mb_addr - w, xn, yn + 16)
        } else {
            // Corner diagonal (xn<0 && yn<0) — not used by (A, B).
            return 0;
        };
        let Some(info) = grid.mbs.get(nbr_mb_addr as usize) else { return 0 };
        if !info.available { return 0 }
        // §9.3.3.1.1.7: skip / intra / unavailable neighbours contribute 0.
        if info.is_skip || info.is_intra { return 0 }
        let idx = blk4x4_idx(nxn, nyn) as usize;
        let v = match (list, comp) {
            (0, MvdComponent::X) => info.mvd_l0_x[idx],
            (0, MvdComponent::Y) => info.mvd_l0_y[idx],
            (1, MvdComponent::X) => info.mvd_l1_x[idx],
            _                    => info.mvd_l1_y[idx],
        };
        v.unsigned_abs() as u32
    };
    pick(bx - 1, by) + pick(bx, by - 1)
}

// ---------------------------------------------------------------------------
// §9.3.3.1.1.9 — coded_block_flag ctxIdxInc helper.
// ---------------------------------------------------------------------------

/// §9.3.3.1.1.9 — derive the (left, above) coded_block_flag condTerm pair
/// for the given neighbour address + block index. Handles the spec's
/// "not available" fallback (unavailable + current-is-intra → 1;
/// unavailable + current-is-inter → 0), P_Skip / B_Skip → 0, and
/// I_PCM → 1.
///
/// `is_cr` disambiguates the iCbCr axis for `BlockType::ChromaDc` and
/// `BlockType::ChromaAc`: the spec §9.3.3.1.1.9 cat=3 / cat=4 use the
/// neighbour's Cb DC/AC block when iCbCr=0 and the neighbour's Cr
/// DC/AC block when iCbCr=1 — our grid tracks both arrays but the
/// block_type alone is cat-4 generic.
///
/// Returns `(cond_a, cond_b)` such that `ctxIdxInc = cond_a + 2 * cond_b`.
#[allow(clippy::too_many_arguments)]
fn cabac_cbf_cond_terms(
    grid: &CabacNeighbourGrid,
    current_mb_addr: u32,
    current_mb: &CabacMbNeighbourInfo,
    current_is_intra: bool,
    block_type: BlockType,
    blk_idx: u8,
    is_cr: bool,
) -> (bool, bool) {
    let w = grid.width_in_mbs;
    let x = if w == 0 { 0 } else { current_mb_addr % w };
    let y = if w == 0 { 0 } else { current_mb_addr / w };

    // Compute in-MB A / B neighbour block coordinates for this block type.
    // For luma 4x4 / 16x16AC blocks (indexed 0..=15): use §6.4.11.4.
    // For chroma AC (indexed 0..=7 at 4:2:2 or 0..=3 at 4:2:0): use the
    // chroma layout. For chroma DC / luma 16x16 DC: the block is
    // MB-level — there is no "4x4 internal neighbour"; A/B both come
    // from the left/above MB's same category.
    let (a_loc, b_loc) = match block_type {
        BlockType::Luma4x4
        | BlockType::Luma16x16Ac
        | BlockType::CbLuma4x4
        | BlockType::CbIntra16x16Ac
        | BlockType::CrLuma4x4
        | BlockType::CrIntra16x16Ac => {
            let (bx, by) = blk4x4_xy(blk_idx);
            (CbfLoc::Internal4x4(bx - 1, by), CbfLoc::Internal4x4(bx, by - 1))
        }
        BlockType::ChromaAc | BlockType::Cb8x8 | BlockType::Cr8x8 => {
            // Chroma 4x4 layout (4:2:0 = 2x2, 4:2:2 = 2x4).
            let cx = ((blk_idx as i32) % 2) * 4;
            let cy = ((blk_idx as i32) / 2) * 4;
            (CbfLoc::InternalChromaAc(cx - 1, cy, blk_idx), CbfLoc::InternalChromaAc(cx, cy - 1, blk_idx))
        }
        BlockType::ChromaDc
        | BlockType::Luma16x16Dc
        | BlockType::CbIntra16x16Dc
        | BlockType::CrIntra16x16Dc => {
            (CbfLoc::MbLevel, CbfLoc::MbLevel)
        }
        // Luma 8x8 (4:4:4 style) and reserved combos fall back to
        // MB-level (conservative but spec-consistent for the block
        // categories we instantiate from macroblock_layer.rs).
        _ => (CbfLoc::MbLevel, CbfLoc::MbLevel),
    };

    let left_addr = if x > 0 { Some(current_mb_addr - 1) } else { None };
    let above_addr = if y > 0 { Some(current_mb_addr - w) } else { None };
    let cond_a = cbf_cond_for(grid, current_mb, current_is_intra, block_type, a_loc, left_addr, true, is_cr);
    let cond_b = cbf_cond_for(grid, current_mb, current_is_intra, block_type, b_loc, above_addr, false, is_cr);
    (cond_a, cond_b)
}

#[derive(Debug, Clone, Copy)]
enum CbfLoc {
    /// Luma 4x4-style block at (x, y) in the MB grid. Negative coordinates
    /// dispatch to the respective external MB (A for x<0, B for y<0).
    Internal4x4(i32, i32),
    /// Chroma AC block at (x, y) plus its original in-MB blk index (for
    /// fallback when already in this MB — the caller computes x,y from
    /// blk_idx).
    InternalChromaAc(i32, i32, u8),
    /// MB-level block (chroma DC / luma 16x16 DC) — no in-MB neighbour.
    MbLevel,
}

fn cbf_cond_for(
    grid: &CabacNeighbourGrid,
    current_mb: &CabacMbNeighbourInfo,
    current_is_intra: bool,
    block_type: BlockType,
    loc: CbfLoc,
    ext_mb_addr: Option<u32>,
    is_left_dir: bool,
    is_cr: bool,
) -> bool {
    match loc {
        CbfLoc::Internal4x4(xn, yn) => {
            if xn >= 0 && yn >= 0 {
                // Internal — read current MB's progressive state.
                let bi = blk4x4_idx(xn, yn) as usize;
                cbf_luma_for(current_mb, block_type, bi)
            } else {
                // External — xn<0 → left MB, yn<0 → above MB. Wrap coords.
                let Some(addr) = ext_mb_addr else {
                    return unavail_cbf(current_is_intra, block_type);
                };
                let Some(info) = grid.mbs.get(addr as usize) else {
                    return unavail_cbf(current_is_intra, block_type);
                };
                if !info.available { return unavail_cbf(current_is_intra, block_type) }
                if info.is_i_pcm { return true } // I_PCM → cbf = 1 everywhere.
                if info.is_skip { return false }
                // Wrap coordinates into the neighbour MB (eq. 6-34/6-35).
                let (wx, wy) = if is_left_dir { (xn + 16, yn) } else { (xn, yn + 16) };
                let bi = blk4x4_idx(wx, wy) as usize;
                // §9.3.3.1.1.9 — transBlockN availability for Luma4x4 /
                // Luma16x16Ac / CbLuma4x4 / CrLuma4x4 / CbIntra16x16Ac /
                // CrIntra16x16Ac cat 1/2/7/8/11/12: the neighbour's 4x4
                // luma block is "available" only when the 8x8 CBP bit
                // covering `luma4x4BlkIdxN` is set. If the CBP bit is
                // clear, transBlockN is not available → per spec
                // condTermFlagN = 0 ("mbAddrN available AND transBlockN
                // not available AND mb_type != I_PCM"; I_PCM is
                // filtered earlier in this match arm).
                match block_type {
                    BlockType::Luma4x4
                    | BlockType::Luma16x16Ac
                    | BlockType::CbLuma4x4
                    | BlockType::CrLuma4x4
                    | BlockType::CbIntra16x16Ac
                    | BlockType::CrIntra16x16Ac => {
                        let blk8 = (bi >> 2) as u8;
                        let cbp = match block_type {
                            BlockType::CbLuma4x4 | BlockType::CbIntra16x16Ac => {
                                info.coded_block_pattern_luma // 4:4:4 Cb plane reuses luma CBP in spec; conservative
                            }
                            BlockType::CrLuma4x4 | BlockType::CrIntra16x16Ac => {
                                info.coded_block_pattern_luma
                            }
                            _ => info.coded_block_pattern_luma,
                        };
                        if ((cbp >> blk8) & 1) == 0 {
                            // §9.3.3.1.1.9 — "mbAddrN available AND
                            // transBlockN not available AND mb_type !=
                            // I_PCM" → condTermFlagN = 0 (spec rule,
                            // bullet 2 under the "If any of the following
                            // conditions are true, condTermFlagN is set
                            // equal to 0" branch).
                            return trans_block_unavail_cbf();
                        }
                        // transform_size_8x8_flag gate: for ctxBlockCat ∈
                        // {1, 2}, the 4x4 block is available only when
                        // t8x8=0 for the neighbour. For ctxBlockCat == 5
                        // (Luma8x8), the 8x8 block is available only when
                        // t8x8 == 1. We don't hit cat=5 here (Luma8x8 is
                        // MbLevel for CBF in our dispatcher). Same spec
                        // rule: transBlockN not available → condTermFlagN = 0.
                        if info.transform_size_8x8_flag && matches!(block_type, BlockType::Luma4x4 | BlockType::Luma16x16Ac) {
                            return trans_block_unavail_cbf();
                        }
                    }
                    _ => {}
                }
                cbf_luma_for(info, block_type, bi)
            }
        }
        CbfLoc::InternalChromaAc(xn, yn, _blk_idx) => {
            if xn >= 0 && yn >= 0 {
                // Internal — find blk idx for (xn, yn) in chroma layout.
                // 4:2:0 layout: 2x2 (y up to 4); 4:2:2: 2x4 (y up to 12).
                // Use the same inverse from xy to idx: idx = 2*(y/4) + (x/4).
                let bi = (2 * (yn / 4) + (xn / 4)) as usize;
                cbf_chroma_ac_for(current_mb, block_type, bi.min(7), is_cr)
            } else {
                // External — use left or above MB's same category.
                let Some(addr) = ext_mb_addr else {
                    return unavail_cbf(current_is_intra, block_type);
                };
                let Some(info) = grid.mbs.get(addr as usize) else {
                    return unavail_cbf(current_is_intra, block_type);
                };
                if !info.available { return unavail_cbf(current_is_intra, block_type) }
                if info.is_i_pcm { return true }
                if info.is_skip { return false }
                // §9.3.3.1.1.9 ctxBlockCat=4 (Chroma AC) — the 4x4
                // chroma block is available only when
                // CodedBlockPatternChroma == 2 for the neighbour MB.
                if matches!(block_type, BlockType::ChromaAc)
                    && info.coded_block_pattern_chroma != 2
                {
                    // §9.3.3.1.1.9 — transBlockN not available →
                    // condTermFlagN = 0 (spec rule, bullet 2 under the
                    // "If any of the following conditions are true,
                    // condTermFlagN is set equal to 0" branch; I_PCM
                    // filtered above).
                    return trans_block_unavail_cbf();
                }
                // Wrap: for chroma plane of width 8 / height 8 (4:2:0) or
                // 16 (4:2:2). For simplicity always wrap by 8; the blk
                // index we query is in the first 4 slots for 4:2:0.
                let (wx, wy) = if is_left_dir { (xn + 8, yn) } else { (xn, yn + 8) };
                let bi = (2 * (wy / 4) + (wx / 4)) as usize;
                cbf_chroma_ac_for(info, block_type, bi.min(7), is_cr)
            }
        }
        CbfLoc::MbLevel => {
            let Some(addr) = ext_mb_addr else {
                return unavail_cbf(current_is_intra, block_type);
            };
            let Some(info) = grid.mbs.get(addr as usize) else {
                return unavail_cbf(current_is_intra, block_type);
            };
            if !info.available { return unavail_cbf(current_is_intra, block_type) }
            if info.is_i_pcm { return true }
            if info.is_skip { return false }
            // §9.3.3.1.1.9 — transBlockN availability for MB-level DC
            // blocks depends on whether the neighbour MB carries the
            // matching DC plane. If it does not, transBlockN is
            // "not available" and we fall back to unavail_cbf.
            //
            // * ctxBlockCat=0 (Luma16x16Dc) — present only when the
            //   neighbour is coded in Intra_16x16 mode.
            // * ctxBlockCat=3 (ChromaDc) — present whenever
            //   CodedBlockPatternChroma != 0 (i.e., ≥1) per spec. When
            //   chroma_format_idc is 1 or 2, the NOT-P_Skip / NOT-B_Skip
            //   neighbours always have the DC block potentially present;
            //   `info.cbf_cb_dc` / `info.cbf_cr_dc` carry the decoded
            //   coded_block_flag so the fall-through is correct.
            // * ctxBlockCat=6 (CbIntra16x16Dc) / =10 (CrIntra16x16Dc) —
            //   same Intra_16x16 gate as cat=0, in 4:4:4.
            let neighbour_has_transblock = match block_type {
                BlockType::Luma16x16Dc
                | BlockType::CbIntra16x16Dc
                | BlockType::CrIntra16x16Dc => {
                    // Present only when neighbour is Intra_16x16. Our
                    // `cbf_luma_16x16_dc` / `cbf_c?_16x16_dc` already
                    // encode that path: they're only set inside the
                    // residual walker's Intra_16x16 branch. But the
                    // condition we need here is "was this MB Intra_16x16?"
                    // — we don't store that bit directly, so use a proxy:
                    // if the MB's cbp_luma is 15 AND it's intra AND it
                    // committed `cbf_luma_16x16_dc` (true or false
                    // deliberately), transBlockN is available. Simpler
                    // proxy: for non-Intra_16x16 intra MBs, `is_i_nxn`
                    // is true; we treat those as "not available".
                    info.is_intra && !info.is_i_nxn
                }
                BlockType::ChromaDc => info.coded_block_pattern_chroma != 0,
                _ => true,
            };
            if !neighbour_has_transblock {
                // §9.3.3.1.1.9 — mbAddrN available, transBlockN not
                // available (DC block not present for neighbour MB),
                // mb_type != I_PCM (filtered above) → condTermFlagN = 0.
                return trans_block_unavail_cbf();
            }
            cbf_mb_level_for(info, block_type, is_cr)
        }
    }
}

/// §9.3.3.1.1.9 — "mbAddrN not available" fallback per Table 9-42:
/// condTermFlag = 0 when the current MB is inter, 1 when intra.
///
/// Call this ONLY when the neighbouring macroblock itself is unavailable
/// (edge of picture / different slice). Do NOT use this for the case
/// where mbAddrN is available but its transBlockN is not — per spec,
/// that path yields condTermFlag = 0 regardless of current_is_intra
/// (unless mb_type(mbAddrN) = I_PCM, which callers filter earlier). Use
/// `trans_block_unavail_cbf` for the available-but-no-transblock path.
#[inline]
fn unavail_cbf(current_is_intra: bool, _block_type: BlockType) -> bool {
    current_is_intra
}

/// §9.3.3.1.1.9 — "mbAddrN available but transBlockN not available"
/// fallback per the spec condTerm rules:
///   "mbAddrN is available and transBlockN is not available and
///    mb_type for the macroblock mbAddrN is not equal to I_PCM"
///   → condTermFlagN = 0.
/// Callers must have already handled the I_PCM case (→ 1) and the
/// P_Skip / B_Skip case (→ 0) before invoking this helper.
#[inline]
fn trans_block_unavail_cbf() -> bool {
    false
}

#[inline]
fn cbf_luma_for(info: &CabacMbNeighbourInfo, block_type: BlockType, bi: usize) -> bool {
    // §9.3.3.1.1.9 — condTermFlagN is the coded_block_flag of the
    // transBlockN 4x4 block at the neighbour's luma4x4BlkIdxN. The
    // neighbour's cbf is stored in `cbf_luma_4x4` when the neighbour
    // was I_NxN / Inter (the 4x4 block is a Luma4x4 residual) or in
    // `cbf_luma_16x16_ac` when the neighbour was Intra_16x16 (the 4x4
    // block is a Luma16x16ACLevel residual). The CURRENT MB's
    // `block_type` (cat=1 Luma16x16Ac vs cat=2 Luma4x4) selects which
    // ctxIdx-family we're decoding — NOT which array to consult on
    // the neighbour. Always OR both so that an Intra_16x16 MB looking
    // up an I_NxN neighbour (or vice versa) reads the correct cbf.
    //
    // Per-MB residual populates only ONE of the two arrays (the other
    // stays default `false`), so OR-ing yields the decoded cbf
    // regardless of neighbour mb_type.
    match block_type {
        BlockType::Luma4x4 | BlockType::Luma16x16Ac => {
            info.cbf_luma_4x4.get(bi).copied().unwrap_or(false)
                || info.cbf_luma_16x16_ac.get(bi).copied().unwrap_or(false)
        }
        BlockType::CbLuma4x4 | BlockType::CbIntra16x16Ac => {
            info.cbf_cb_luma_4x4.get(bi).copied().unwrap_or(false)
                || info.cbf_cb_16x16_ac.get(bi).copied().unwrap_or(false)
        }
        BlockType::CrLuma4x4 | BlockType::CrIntra16x16Ac => {
            info.cbf_cr_luma_4x4.get(bi).copied().unwrap_or(false)
                || info.cbf_cr_16x16_ac.get(bi).copied().unwrap_or(false)
        }
        _ => false,
    }
}

#[inline]
fn cbf_chroma_ac_for(
    info: &CabacMbNeighbourInfo,
    block_type: BlockType,
    bi: usize,
    is_cr: bool,
) -> bool {
    match block_type {
        BlockType::ChromaAc => {
            // §9.3.3.1.1.9 cat=4 — Cb when iCbCr=0, Cr when iCbCr=1.
            if is_cr {
                info.cbf_cr_ac.get(bi).copied().unwrap_or(false)
            } else {
                info.cbf_cb_ac.get(bi).copied().unwrap_or(false)
            }
        }
        BlockType::Cb8x8 => info.cbf_cb_ac.get(bi).copied().unwrap_or(false),
        BlockType::Cr8x8 => info.cbf_cr_ac.get(bi).copied().unwrap_or(false),
        _ => false,
    }
}

#[inline]
fn cbf_mb_level_for(info: &CabacMbNeighbourInfo, block_type: BlockType, is_cr: bool) -> bool {
    match block_type {
        // §9.3.3.1.1.9 cat=3 — Cb DC when iCbCr=0, Cr DC when iCbCr=1.
        BlockType::ChromaDc => {
            if is_cr {
                info.cbf_cr_dc
            } else {
                info.cbf_cb_dc
            }
        }
        BlockType::Luma16x16Dc => info.cbf_luma_16x16_dc,
        // §9.3.3.1.1.9 — 4:4:4 per-plane 16x16-DC: each plane has its
        // own coded_block_flag bit (ctxBlockCat=6 for Cb, 10 for Cr).
        BlockType::CbIntra16x16Dc => info.cbf_cb_16x16_dc,
        BlockType::CrIntra16x16Dc => info.cbf_cr_16x16_dc,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Entropy-state abstraction for mb_type parsing.
// ---------------------------------------------------------------------------

/// Thin wrapper over `Option<&mut (CabacDecoder, CabacContexts)>` plus
/// per-slice derived state. Mirrors the slice data walker's view.
pub struct EntropyState<'ctx, 'data> {
    /// `None` → CAVLC; `Some` → CABAC with decoder + per-slice contexts.
    pub cabac: Option<(&'ctx mut CabacDecoder<'data>, &'ctx mut CabacContexts)>,
    /// Slice kind — only consulted in the CABAC path.
    pub slice_kind: SliceKind,
    /// Neighbour context (§9.3.3.1.1.*). Conservative default = all
    /// neighbours unavailable.
    pub neighbours: NeighbourCtx,
    /// Rolling flag for mb_qp_delta CABAC bin 0 context (§9.3.3.1.1.5).
    pub prev_mb_qp_delta_nonzero: bool,
    /// ChromaArrayType derived from the active SPS (§6.2).
    pub chroma_array_type: u32,
    /// True when `transform_8x8_mode_flag` is set on the active PPS —
    /// used to gate the `transform_size_8x8_flag` read (§7.3.5).
    pub transform_8x8_mode_flag: bool,
    /// §9.2.1.1 — CAVLC per-MB total_coeff grid for neighbour lookups.
    /// Optional so CABAC / unit-test callers that only exercise a
    /// single MB can leave it unset (in which case nC defaults to 0 —
    /// spec-consistent for the very first MB of a slice).
    pub cavlc_nc: Option<&'ctx mut CavlcNcGrid>,
    /// §9.2.1.1 — MB address of the macroblock currently being parsed.
    /// Required when `cavlc_nc` is set.
    pub current_mb_addr: u32,
    /// §7.4.2.2 — `constrained_intra_pred_flag` from the active PPS.
    /// Threaded in but currently a no-op (the slice-data-partitioning
    /// override in §9.2.1.1 step 5 does not apply to the NAL types our
    /// walker accepts).
    pub constrained_intra_pred_flag: bool,
    /// §7.3.3 / §7.3.5.1 — active `num_ref_idx_l0_active_minus1` for the
    /// current slice (from the slice header, possibly overridden by
    /// `num_ref_idx_active_override_flag`). Used to gate / size ref_idx_l0
    /// parsing per the syntax rule in §7.3.5.1 / §7.3.5.2.
    pub num_ref_idx_l0_active_minus1: u32,
    /// §7.3.3 / §7.3.5.1 — active `num_ref_idx_l1_active_minus1` (B slices).
    pub num_ref_idx_l1_active_minus1: u32,
    /// §7.3.4 / §7.4.4 — `MbaffFrameFlag`. When 0, `mb_field_decoding_flag`
    /// equals `field_pic_flag`, i.e. the "mb_field != field_pic" gate in
    /// §7.3.5.1 / §7.3.5.2 never fires. We don't currently support MBAFF
    /// decoding, so parsers should assume this is false.
    pub mbaff_frame_flag: bool,
    /// §9.3.3.1.1.6 / .7 / .9 — CABAC per-MB neighbour grid. Populated
    /// by the MB parser with each macroblock's decoded inter / residual
    /// state, consulted on subsequent MBs for neighbour-derived
    /// ctxIdxInc. Pass `None` for a conservative approximation (all
    /// neighbour condTerms = 0 — spec-consistent fallback that biases
    /// probabilities but does not desync bitstreams).
    pub cabac_nb: Option<&'ctx mut CabacNeighbourGrid>,
    /// §6.4.9 — picture width in macroblocks, required for the
    /// `address → (x, y)` conversion inside the CABAC neighbour grid.
    /// `0` disables the grid (treats everything as unavailable) even
    /// when `cabac_nb` is `Some`.
    pub pic_width_in_mbs: u32,
}

// ---------------------------------------------------------------------------
// Top-level entry point.
// ---------------------------------------------------------------------------

/// §7.3.5 — parse one `macroblock_layer()`.
///
/// The CAVLC path reads from the bitstream directly. The CABAC path
/// reuses the supplied decoder + contexts (CABAC consumes bytes from
/// its own underlying reader, not from `r`). Callers pass `r` even in
/// CABAC mode for I_PCM (which byte-aligns and reads from the CABAC
/// reader — our CABAC engine doesn't expose that, so CABAC I_PCM is a
/// documented limitation in this pass).
pub fn parse_macroblock(
    r: &mut BitReader<'_>,
    entropy: &mut EntropyState<'_, '_>,
    slice_header: &SliceHeader,
    sps: &Sps,
    _pps: &Pps,
    _current_mb_addr: u32,
) -> McblResult<Macroblock> {
    let slice_type = slice_header.slice_type;
    let dbg = std::env::var("OXIDEAV_H264_DEBUG").is_ok();
    let mb_addr_dbg = entropy.current_mb_addr;
    if dbg {
        let (b, bi) = r.position();
        let (cb, cbi, bins) = if let Some((dec, _)) = entropy.cabac.as_ref() {
            let (pb, pbi) = dec.position();
            (pb as i64, pbi as i64, dec.bin_count() as i64)
        } else {
            (-1, -1, -1)
        };
        eprintln!(
            "[MB {:>4}] enter cursor=({},{}) cabac=({},{}) bins={} slice_type={:?}",
            mb_addr_dbg, b, bi, cb, cbi, bins, slice_type
        );
    }

    // ---------------------------------------------------------------
    // §7.3.5 — mb_type.
    // CAVLC: ue(v). CABAC: per-slice-type decoder in cabac_ctx.
    // ---------------------------------------------------------------
    let mb_type_raw = if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
        match slice_type {
            SliceType::I | SliceType::SI => decode_mb_type_i(dec, ctxs, &entropy.neighbours)?,
            SliceType::P | SliceType::SP => decode_mb_type_p(dec, ctxs, &entropy.neighbours)?,
            SliceType::B => decode_mb_type_b(dec, ctxs, &entropy.neighbours)?,
        }
    } else {
        r.ue()?
    };
    if dbg {
        let (b, bi) = r.position();
        eprintln!(
            "[MB {:>4}]   after mb_type raw={} cursor=({},{})",
            mb_addr_dbg, mb_type_raw, b, bi
        );
    }

    let mb_type = match slice_type {
        SliceType::I | SliceType::SI => MbType::from_i_slice(mb_type_raw)?,
        SliceType::P | SliceType::SP => MbType::from_p_slice(mb_type_raw)?,
        SliceType::B => MbType::from_b_slice(mb_type_raw)?,
    };

    // ---------------------------------------------------------------
    // §7.3.5 — I_PCM path.
    // ---------------------------------------------------------------
    if mb_type.is_i_pcm() {
        // Byte-align in CAVLC mode. CABAC already renormalises to bit
        // boundaries; byte alignment for I_PCM in CABAC uses a different
        // path (§9.3.1.2) not supported in this pass.
        if entropy.cabac.is_some() {
            return Err(MacroblockLayerError::UnsupportedChromaArrayType(999));
        }
        while !r.byte_aligned() {
            // §7.3.5: pcm_alignment_zero_bit shall be 0. We tolerate
            // any value since the spec wording is "shall be" rather
            // than "shall be parsed as".
            let _ = r.u(1)?;
        }
        let bit_depth_y: u32 = 8; // For baseline/main/hi420 profiles we're
                                  // testing, bit_depth_luma_minus8 == 0 → 8.
                                  // A future pass should consult the SPS.
        let bit_depth_c: u32 = 8;
        // ChromaArrayType determines the count of chroma samples per §7.4.5:
        //   0 → none; 1 → 2*MbWidthC*MbHeightC = 2*8*8 = 128 total (64 Cb + 64 Cr);
        //   2 → 2*16*8 = 256 (128 + 128); 3 → 2*16*16 = 512 (256 + 256).
        let (num_cb, num_cr): (usize, usize) = match entropy.chroma_array_type {
            0 => (0, 0),
            1 => (64, 64),
            2 => (128, 128),
            3 => (256, 256),
            _ => {
                return Err(MacroblockLayerError::UnsupportedChromaArrayType(
                    entropy.chroma_array_type,
                ))
            }
        };
        let mut luma = Vec::with_capacity(256);
        for _ in 0..256 {
            luma.push(r.u(bit_depth_y)?);
        }
        let mut cb = Vec::with_capacity(num_cb);
        for _ in 0..num_cb {
            cb.push(r.u(bit_depth_c)?);
        }
        let mut cr = Vec::with_capacity(num_cr);
        for _ in 0..num_cr {
            cr.push(r.u(bit_depth_c)?);
        }
        // §9.3.3.1.1.5 — I_PCM forces the next MB's mb_qp_delta
        // ctxIdxInc to 0. Reset the rolling flag so the slice walker
        // propagates the correct state.
        entropy.prev_mb_qp_delta_nonzero = false;
        return Ok(Macroblock {
            mb_type,
            mb_type_raw,
            mb_pred: None,
            sub_mb_pred: None,
            pcm_samples: Some(PcmSamples {
                luma,
                chroma_cb: cb,
                chroma_cr: cr,
            }),
            coded_block_pattern: 0,
            transform_size_8x8_flag: false,
            mb_qp_delta: 0,
            residual_luma: Vec::new(),
            residual_luma_dc: None,
            residual_chroma_dc_cb: Vec::new(),
            residual_chroma_dc_cr: Vec::new(),
            residual_chroma_ac_cb: Vec::new(),
            residual_chroma_ac_cr: Vec::new(),
            residual_cb_luma_like: Vec::new(),
            residual_cr_luma_like: Vec::new(),
            residual_cb_16x16_dc: None,
            residual_cr_16x16_dc: None,
            is_skip: false,
        });
    }

    // ---------------------------------------------------------------
    // §7.3.5 — non-PCM path: sub_mb_pred OR mb_pred.
    //
    // Per the spec syntax table, when mb_type == I_NxN AND the PPS
    // has transform_8x8_mode_flag == 1, `transform_size_8x8_flag` is
    // read BEFORE `mb_pred(mb_type)` so that the pred-mode loop can
    // switch to 4 Intra_8x8 entries (vs. 16 Intra_4x4 entries). This
    // gate is checked again — for the non-I_NxN case — after
    // coded_block_pattern.
    // ---------------------------------------------------------------
    let mut mb_pred: Option<MbPred> = None;
    let mut sub_mb_pred: Option<SubMbPred> = None;
    let mut transform_size_8x8_flag = false;

    if mb_type.is_sub_mb_path() {
        sub_mb_pred = Some(parse_sub_mb_pred(r, entropy, &mb_type)?);
    } else {
        // §7.3.5 — read transform_size_8x8_flag for I_NxN when allowed.
        if mb_type.is_i_nxn() && entropy.transform_8x8_mode_flag {
            transform_size_8x8_flag = if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
                decode_transform_size_8x8_flag(dec, ctxs, &entropy.neighbours)?
            } else {
                r.u(1)? == 1
            };
        }
        mb_pred = Some(parse_mb_pred(r, entropy, &mb_type, transform_size_8x8_flag)?);
    }
    if dbg {
        let (b, bi) = r.position();
        eprintln!(
            "[MB {:>4}]   after mb_pred/sub_mb_pred mb_type={:?} cursor=({},{})",
            mb_addr_dbg, mb_type, b, bi
        );
    }

    // ---------------------------------------------------------------
    // §7.3.5 — coded_block_pattern when mb_type != Intra_16x16.
    // ---------------------------------------------------------------
    let cbp_luma: u32;
    let cbp_chroma: u32;
    let cbp_total: u32;
    if let Some(i16) = mb_type.intra_16x16_cbp_luma() {
        cbp_luma = i16 as u32;
        cbp_chroma = mb_type.intra_16x16_cbp_chroma().unwrap_or(0) as u32;
        cbp_total = cbp_luma | (cbp_chroma << 4);
    } else {
        cbp_total = if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
            let bb = dec.bin_count();
            let v = decode_coded_block_pattern(dec, ctxs, &entropy.neighbours, entropy.chroma_array_type)?;
            if dbg {
                eprintln!("[MB {:>4}]   CBP bins_consumed={} value={}", mb_addr_dbg, dec.bin_count() - bb, v);
            }
            v
        } else {
            parse_cbp_me(r, entropy.chroma_array_type, mb_type.is_intra())?
        };
        cbp_luma = cbp_total & 0x0F;
        cbp_chroma = (cbp_total >> 4) & 0x03;
    }
    if dbg {
        let (b, bi) = r.position();
        eprintln!(
            "[MB {:>4}]   after CBP total={} luma={} chroma={} cursor=({},{})",
            mb_addr_dbg, cbp_total, cbp_luma, cbp_chroma, b, bi
        );
    }

    // ---------------------------------------------------------------
    // §7.3.5 — transform_size_8x8_flag (second gate, for non-I_NxN).
    //   if (CodedBlockPatternLuma > 0 && transform_8x8_mode_flag &&
    //       mb_type != I_NxN && noSubMbPartSizeLessThan8x8Flag &&
    //       (mb_type != B_Direct_16x16 || direct_8x8_inference_flag))
    // For the I_NxN case the flag was already read before mb_pred.
    //
    // §7.3.5 — `noSubMbPartSizeLessThan8x8Flag` is computed during the
    // sub_mb_pred pass:
    //   1) it starts at 1;
    //   2) for each sub_mb_type != B_Direct_8x8, if NumSubMbPart > 1,
    //      set it to 0;
    //   3) for each sub_mb_type == B_Direct_8x8, if
    //      !direct_8x8_inference_flag, set it to 0.
    // When the MB is on the mb_pred (non-sub_mb) path the flag is
    // always 1 (there are no sub-partitions to invalidate it).
    // ---------------------------------------------------------------
    let direct_8x8_inference_flag = sps.direct_8x8_inference_flag;
    let no_sub_lt_8x8 = if let Some(sub) = sub_mb_pred.as_ref() {
        let mut flag = true;
        for i in 0..4 {
            let sub_t = sub.sub_mb_type[i];
            if matches!(sub_t, SubMbType::BDirect8x8) {
                if !direct_8x8_inference_flag {
                    flag = false;
                }
            } else if sub_t.num_sub_mb_part() > 1 {
                flag = false;
            }
        }
        flag
    } else {
        true
    };
    let t8x8_gate = cbp_luma > 0
        && entropy.transform_8x8_mode_flag
        && !mb_type.is_i_nxn()
        && !mb_type.is_intra_16x16()
        && no_sub_lt_8x8
        && (!matches!(mb_type, MbType::BDirect16x16) || direct_8x8_inference_flag);
    if t8x8_gate {
        transform_size_8x8_flag = if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
            decode_transform_size_8x8_flag(dec, ctxs, &entropy.neighbours)?
        } else {
            r.u(1)? == 1
        };
    }

    // ---------------------------------------------------------------
    // §7.3.5 — mb_qp_delta. Read when any of cbp_luma, cbp_chroma, or
    // Intra_16x16 mb_type are present.
    // ---------------------------------------------------------------
    let needs_qp_delta = cbp_luma > 0 || cbp_chroma > 0 || mb_type.is_intra_16x16();
    let mb_qp_delta = if needs_qp_delta {
        if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
            let bins_before = dec.bin_count();
            let v = decode_mb_qp_delta(dec, ctxs, entropy.prev_mb_qp_delta_nonzero)?;
            let bins_after = dec.bin_count();
            if dbg {
                eprintln!(
                    "[MB {:>4}]   mb_qp_delta decode: prev_nz={} bins_consumed={} value={}",
                    mb_addr_dbg, entropy.prev_mb_qp_delta_nonzero,
                    bins_after - bins_before, v
                );
            }
            v
        } else {
            r.se()?
        }
    } else {
        0
    };
    entropy.prev_mb_qp_delta_nonzero = mb_qp_delta != 0;
    if dbg {
        let (b, bi) = r.position();
        eprintln!(
            "[MB {:>4}]   after mb_qp_delta={} t8x8_flag={} cursor=({},{})",
            mb_addr_dbg, mb_qp_delta, transform_size_8x8_flag, b, bi
        );
    }

    // ---------------------------------------------------------------
    // §7.3.5.3 — residual(). CABAC path is simplified (per-block decode
    // with neighbour CBF set conservatively to None).
    // ---------------------------------------------------------------
    let mut out = Macroblock {
        mb_type: mb_type.clone(),
        mb_type_raw,
        mb_pred,
        sub_mb_pred,
        pcm_samples: None,
        coded_block_pattern: cbp_total,
        transform_size_8x8_flag,
        mb_qp_delta,
        residual_luma: Vec::new(),
        residual_luma_dc: None,
        residual_chroma_dc_cb: Vec::new(),
        residual_chroma_dc_cr: Vec::new(),
        residual_chroma_ac_cb: Vec::new(),
        residual_chroma_ac_cr: Vec::new(),
        residual_cb_luma_like: Vec::new(),
        residual_cr_luma_like: Vec::new(),
        residual_cb_16x16_dc: None,
        residual_cr_16x16_dc: None,
        is_skip: false,
    };

    // §7.3.5.3 — residual block = residual_block_cavlc or
    // residual_block_cabac depending on entropy_coding_mode_flag.
    if entropy.cabac.is_some() {
        // Split the borrow: `cabac` and `cabac_nb` are both fields of
        // `entropy` but we need both mutably in the residual walker.
        let current_mb_addr = entropy.current_mb_addr;
        let chroma_at = entropy.chroma_array_type;
        let current_is_intra = mb_type.is_intra();
        let (cabac, ctxs) = entropy.cabac.as_mut().unwrap();
        let nb_grid = entropy.cabac_nb.as_deref_mut();
        parse_residual_cabac_only(
            cabac,
            ctxs,
            chroma_at,
            &mb_type,
            cbp_luma,
            cbp_chroma,
            transform_size_8x8_flag,
            &mut out,
            nb_grid,
            current_mb_addr,
            current_is_intra,
        )?;
        // §9.3.3.1.1.* — commit the per-MB syntax state consulted by the
        // NEXT MB's CABAC ctxIdxInc derivations that are not covered by
        // the residual walker's cbf_* commit (CBP, intra_chroma_pred_mode,
        // transform_size_8x8_flag, I_NxN classification).
        if let Some(grid) = entropy.cabac_nb.as_deref_mut() {
            if let Some(slot) = grid.mbs.get_mut(entropy.current_mb_addr as usize) {
                slot.coded_block_pattern_luma = (cbp_total & 0x0F) as u8;
                slot.coded_block_pattern_chroma = ((cbp_total >> 4) & 0x03) as u8;
                slot.is_i_nxn = mb_type.is_i_nxn();
                slot.transform_size_8x8_flag = transform_size_8x8_flag;
                if let Some(pred) = out.mb_pred.as_ref() {
                    slot.intra_chroma_pred_mode = pred.intra_chroma_pred_mode;
                }
            }
        }
    } else {
        parse_residual_cavlc_only(
            r,
            entropy,
            &mb_type,
            cbp_luma,
            cbp_chroma,
            transform_size_8x8_flag,
            &mut out,
        )?;
    }

    // §9.2.1.1 — mark this MB's slot "available" in the CAVLC neighbour
    // grid so subsequent MBs see it when computing nC. We only do this
    // on the CAVLC path (CABAC doesn't need the grid) — `cavlc_nc` is
    // `None` in CABAC mode.
    if entropy.cabac.is_none() {
        if let Some(grid) = entropy.cavlc_nc.as_deref_mut() {
            let addr = entropy.current_mb_addr as usize;
            if let Some(slot) = grid.mbs.get_mut(addr) {
                slot.is_available = true;
                slot.is_skip = false;
                slot.is_intra = mb_type.is_intra();
                slot.is_i_pcm = mb_type.is_i_pcm();
                // I_PCM contributes nN = 16 regardless of total_coeff
                // arrays; zero them so the `is_i_pcm` path takes over.
                if mb_type.is_i_pcm() {
                    slot.luma_total_coeff = [0; 16];
                    slot.cb_total_coeff = [0; 8];
                    slot.cr_total_coeff = [0; 8];
                }
            }
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// §7.3.5.1 — mb_pred() parsing.
// ---------------------------------------------------------------------------

fn parse_mb_pred(
    r: &mut BitReader<'_>,
    entropy: &mut EntropyState<'_, '_>,
    mb_type: &MbType,
    transform_size_8x8_flag: bool,
) -> McblResult<MbPred> {
    let mut pred = MbPred::default();
    let dbg_mb = std::env::var("OXIDEAV_H264_MB_TRACE").ok()
        .and_then(|v| v.parse::<u32>().ok());
    let dbg_on = dbg_mb == Some(entropy.current_mb_addr);

    if mb_type.is_i_nxn() {
        // §7.3.5.1 — Intra_4x4 (16 blocks) or Intra_8x8 (4 blocks)
        // pred modes, selected by the PPS `transform_8x8_mode_flag`
        // + per-MB `transform_size_8x8_flag` gate read immediately
        // before `mb_pred` (see caller).
        if transform_size_8x8_flag {
            // Intra_8x8: 4 luma8x8BlkIdx entries.
            for i in 0..4 {
                let flag = if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
                    decode_prev_intra_pred_mode_flag(dec, ctxs)?
                } else {
                    r.u(1)? == 1
                };
                pred.prev_intra8x8_pred_mode_flag[i] = flag;
                if !flag {
                    let rem = if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
                        decode_rem_intra_pred_mode(dec, ctxs)?
                    } else {
                        r.u(3)?
                    };
                    pred.rem_intra8x8_pred_mode[i] = rem as u8;
                }
            }
        } else {
            // Intra_4x4: 16 luma4x4BlkIdx entries.
            for i in 0..16 {
                let flag = if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
                    decode_prev_intra_pred_mode_flag(dec, ctxs)?
                } else {
                    r.u(1)? == 1
                };
                pred.prev_intra4x4_pred_mode_flag[i] = flag;
                if !flag {
                    let rem = if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
                        decode_rem_intra_pred_mode(dec, ctxs)?
                    } else {
                        r.u(3)?
                    };
                    pred.rem_intra4x4_pred_mode[i] = rem as u8;
                }
            }

        }
        // Chroma intra pred mode — read when ChromaArrayType ∈ {1, 2}.
        if matches!(entropy.chroma_array_type, 1 | 2) {
            let mode = if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
                decode_intra_chroma_pred_mode(dec, ctxs, &entropy.neighbours)?
            } else {
                r.ue()?
            };
            pred.intra_chroma_pred_mode = mode as u8;
        }
    } else if mb_type.is_intra_16x16() {
        // Intra_16x16 → only chroma pred mode is read via mb_pred.
        if matches!(entropy.chroma_array_type, 1 | 2) {
            let mode = if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
                decode_intra_chroma_pred_mode(dec, ctxs, &entropy.neighbours)?
            } else {
                r.ue()?
            };
            pred.intra_chroma_pred_mode = mode as u8;
        }
    } else {
        // Inter — §7.3.5.1 `mb_pred()` for non-sub_mb, non-Direct cases.
        //
        // The spec's inner conditions are:
        //   - ref_idx_l0 present when  (num_ref_idx_l0_active_minus1 > 0
        //       || mb_field_decoding_flag != field_pic_flag)
        //       && MbPartPredMode != Pred_L1.
        //   - ref_idx_l1 present when  (num_ref_idx_l1_active_minus1 > 0
        //       || mb_field_decoding_flag != field_pic_flag)
        //       && MbPartPredMode != Pred_L0.
        //   - mvd_l0 present when MbPartPredMode != Pred_L1.
        //   - mvd_l1 present when MbPartPredMode != Pred_L0.
        // Direct is only reachable via B_Direct_16x16 / B_Skip (no
        // mb_pred fields); we don't enter this branch for those.
        //
        // MBAFF is not currently supported in our pipeline
        // (`mbaff_frame_flag == false`), so the "mb_field !=
        // field_pic" condition simplifies away.
        let num_parts = mb_type.num_mb_part() as usize;
        let mbaff_override = entropy.mbaff_frame_flag;
        let ref_l0_present = entropy.num_ref_idx_l0_active_minus1 > 0 || mbaff_override;
        let ref_l1_present = entropy.num_ref_idx_l1_active_minus1 > 0 || mbaff_override;
        let x_l0 = entropy.num_ref_idx_l0_active_minus1;
        let x_l1 = entropy.num_ref_idx_l1_active_minus1;

        // Map MB partition index to the 8x8 partition covering it for
        // ref_idx neighbour lookups. Per §7.4.5 / Table 7-13:
        //   16x16 (num_parts=1): part 0 covers all 4 8x8 → use 8x8 #0.
        //   16x8  (num_parts=2): part 0 covers 8x8 #0/#1, part 1 covers #2/#3.
        //     For (left, above) neighbour derivation at partition 0
        //     A/B come from the 8x8-top-left slot → use 8x8 #0.
        //     For partition 1 use 8x8 #2 (its top-left 8x8).
        //   8x16  (num_parts=2): part 0 uses 8x8 #0, part 1 uses 8x8 #1.
        let part_to_8x8 = |part: usize| -> u8 {
            match (num_parts, part) {
                (1, _) => 0,
                (2, 0) => 0,
                (2, 1) => {
                    // Need to tell 16x8 vs 8x16 apart.
                    if mb_type.is_partitioned_16x8() { 2 } else { 1 }
                }
                (4, p) => p as u8,
                _ => 0,
            }
        };

        // Scratch "current MB" neighbour info populated as we decode —
        // later committed to the grid at MB end. For now we only need
        // this to allow internal 8x8 #1/#2/#3 ref_idx lookups to see
        // their already-decoded siblings, and for in-MB 4x4 mvd
        // neighbour lookups (mvd[0] → mvd[1] etc.).
        let mut curr_nb = CabacMbNeighbourInfo::default();
        curr_nb.available = false; // not yet — still decoding this MB

        // ref_idx_l0. §7.3.5.1 gate: (num_ref_idx_l0_active_minus1 > 0 ||
        //   mb_field_decoding_flag != field_pic_flag) &&
        //   MbPartPredMode(mb_type, mbPartIdx) != Pred_L1.
        // CABAC uses §9.3.3.1.1.6 (decode_ref_idx_lx); CAVLC uses te(v)
        // (§9.1.2 equation 9-3). Under CABAC the gate still decides
        // presence — an absent ref_idx is inferred to 0 per §7.4.5.1.
        for part in 0..num_parts {
            let mode = mb_type.mb_part_pred_mode(part);
            if ref_l0_present && mode != Some(MbPartPredMode::PredL1) {
                let v = if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
                    // §9.3.3.1.1.6 eq. 9-14 — (condTermA, condTermB).
                    let (ca, cb) = neighbour_ref_idx_gt_0(
                        entropy.cabac_nb.as_deref(),
                        entropy.current_mb_addr,
                        &curr_nb,
                        0,
                        part_to_8x8(part),
                    );
                    decode_ref_idx_lx(dec, ctxs, ca, cb)?
                } else {
                    r.te(x_l0)?
                };
                if dbg_on {
                    eprintln!("[MBP {}] ref_idx_l0 part={} gated=true val={}",
                             entropy.current_mb_addr, part, v);
                }
                // Record per-8x8-partition value covering this mb part.
                for p8 in partition_8x8_range(num_parts, part, mb_type.is_partitioned_16x8()) {
                    curr_nb.ref_idx_l0[p8 as usize] = v.min(127) as i8;
                }
                pred.ref_idx_l0.push(v);
            } else {
                pred.ref_idx_l0.push(0);
            }
        }
        // ref_idx_l1. §7.3.5.1 B-slice gate: (num_ref_idx_l1_active_minus1 > 0
        //   || mb_field_decoding_flag != field_pic_flag) &&
        //   MbPartPredMode != Pred_L0 (and not Direct).
        for part in 0..num_parts {
            let mode = mb_type.mb_part_pred_mode(part);
            if ref_l1_present && mode != Some(MbPartPredMode::PredL0)
                && mode != Some(MbPartPredMode::Direct)
                && entropy.slice_kind == SliceKind::B
            {
                let v = if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
                    let (ca, cb) = neighbour_ref_idx_gt_0(
                        entropy.cabac_nb.as_deref(),
                        entropy.current_mb_addr,
                        &curr_nb,
                        1,
                        part_to_8x8(part),
                    );
                    decode_ref_idx_lx(dec, ctxs, ca, cb)?
                } else {
                    r.te(x_l1)?
                };
                for p8 in partition_8x8_range(num_parts, part, mb_type.is_partitioned_16x8()) {
                    curr_nb.ref_idx_l1[p8 as usize] = v.min(127) as i8;
                }
                pred.ref_idx_l1.push(v);
            } else {
                pred.ref_idx_l1.push(0);
            }
        }
        // mvd_l0. §7.3.5.1 gate: MbPartPredMode != Pred_L1 (and not
        // Direct). CABAC: §9.3.3.1.1.7 UEG3; CAVLC: se(v).
        for part in 0..num_parts {
            let mode = mb_type.mb_part_pred_mode(part);
            if mode != Some(MbPartPredMode::PredL1)
                && mode != Some(MbPartPredMode::Direct)
            {
                let (mx, my) = if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
                    // §9.3.3.1.1.7: for MB-level partitions (16x16/16x8/8x16)
                    // only the top-left 4x4 of the partition carries the
                    // mvd; ctxIdxInc uses its neighbours' 4x4 mvd sums.
                    let blk4 = mb_part_top_left_4x4(num_parts, part, mb_type.is_partitioned_16x8());
                    let sum_x = neighbour_mvd_abs_sum(
                        entropy.cabac_nb.as_deref(),
                        entropy.current_mb_addr,
                        &curr_nb,
                        0,
                        MvdComponent::X,
                        blk4,
                    );
                    let sum_y = neighbour_mvd_abs_sum(
                        entropy.cabac_nb.as_deref(),
                        entropy.current_mb_addr,
                        &curr_nb,
                        0,
                        MvdComponent::Y,
                        blk4,
                    );
                    let bb_mvd = dec.bin_count();
                    let mx = decode_mvd_lx(dec, ctxs, MvdComponent::X, sum_x)?;
                    let mb_x = dec.bin_count();
                    let my = decode_mvd_lx(dec, ctxs, MvdComponent::Y, sum_y)?;
                    let mb_y = dec.bin_count();
                    if dbg_on {
                        eprintln!("[MBP {}] mvd_l0 part={} blk4={} sum_x={} sum_y={} mx={} (bins={}) my={} (bins={})",
                                 entropy.current_mb_addr, part, blk4, sum_x, sum_y,
                                 mx, mb_x - bb_mvd, my, mb_y - mb_x);
                    }
                    // Propagate decoded mvd to ALL 4x4 blocks covered by
                    // this partition (so in-MB neighbour lookups see
                    // consistent values).
                    for b4 in mb_part_4x4_range(num_parts, part, mb_type.is_partitioned_16x8()) {
                        curr_nb.mvd_l0_x[b4 as usize] = mx.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                        curr_nb.mvd_l0_y[b4 as usize] = my.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    }
                    (mx, my)
                } else {
                    (r.se()?, r.se()?)
                };
                pred.mvd_l0.push([mx, my]);
            } else {
                pred.mvd_l0.push([0, 0]);
            }
        }
        // mvd_l1. §7.3.5.1 B-slice gate: MbPartPredMode != Pred_L0
        // (and not Direct).
        for part in 0..num_parts {
            let mode = mb_type.mb_part_pred_mode(part);
            if mode != Some(MbPartPredMode::PredL0)
                && mode != Some(MbPartPredMode::Direct)
                && entropy.slice_kind == SliceKind::B
            {
                let (mx, my) = if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
                    let blk4 = mb_part_top_left_4x4(num_parts, part, mb_type.is_partitioned_16x8());
                    let sum_x = neighbour_mvd_abs_sum(
                        entropy.cabac_nb.as_deref(),
                        entropy.current_mb_addr,
                        &curr_nb,
                        1,
                        MvdComponent::X,
                        blk4,
                    );
                    let sum_y = neighbour_mvd_abs_sum(
                        entropy.cabac_nb.as_deref(),
                        entropy.current_mb_addr,
                        &curr_nb,
                        1,
                        MvdComponent::Y,
                        blk4,
                    );
                    let mx = decode_mvd_lx(dec, ctxs, MvdComponent::X, sum_x)?;
                    let my = decode_mvd_lx(dec, ctxs, MvdComponent::Y, sum_y)?;
                    for b4 in mb_part_4x4_range(num_parts, part, mb_type.is_partitioned_16x8()) {
                        curr_nb.mvd_l1_x[b4 as usize] = mx.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                        curr_nb.mvd_l1_y[b4 as usize] = my.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    }
                    (mx, my)
                } else {
                    (r.se()?, r.se()?)
                };
                pred.mvd_l1.push([mx, my]);
            } else {
                pred.mvd_l1.push([0, 0]);
            }
        }

        // Commit this MB's partial state to the grid. `available` stays
        // `false` until parse_macroblock completes and writes the final
        // record with all residual / CBF fields filled.
        if let Some(grid) = entropy.cabac_nb.as_deref_mut() {
            if let Some(slot) = grid.mbs.get_mut(entropy.current_mb_addr as usize) {
                // Preserve any fields already written earlier (none for a
                // fresh MB) but populate inter-prediction state now.
                slot.ref_idx_l0 = curr_nb.ref_idx_l0;
                slot.ref_idx_l1 = curr_nb.ref_idx_l1;
                slot.mvd_l0_x = curr_nb.mvd_l0_x;
                slot.mvd_l0_y = curr_nb.mvd_l0_y;
                slot.mvd_l1_x = curr_nb.mvd_l1_x;
                slot.mvd_l1_y = curr_nb.mvd_l1_y;
            }
        }
    }

    Ok(pred)
}

/// Range of 8x8 partition indices covered by mb partition `part` of
/// `num_parts` total partitions. For 16x16 this is 0..=3; for 16x8 part 0
/// covers 0..=1 and part 1 covers 2..=3; for 8x16 part 0 covers 0,2 and
/// part 1 covers 1,3.
fn partition_8x8_range(num_parts: usize, part: usize, is_16x8: bool) -> Vec<u8> {
    match (num_parts, part, is_16x8) {
        (1, _, _) => vec![0, 1, 2, 3],
        (2, 0, true) => vec![0, 1],   // 16x8 top
        (2, 1, true) => vec![2, 3],   // 16x8 bottom
        (2, 0, false) => vec![0, 2],  // 8x16 left
        (2, 1, false) => vec![1, 3],  // 8x16 right
        (4, p, _) => vec![p as u8],
        _ => vec![0],
    }
}

/// Range of 4x4 block indices covered by mb partition `part`.
fn mb_part_4x4_range(num_parts: usize, part: usize, is_16x8: bool) -> Vec<u8> {
    match (num_parts, part, is_16x8) {
        (1, _, _) => (0..16u8).collect(),
        (2, 0, true) => vec![0, 1, 2, 3, 4, 5, 6, 7],            // 16x8 top half
        (2, 1, true) => vec![8, 9, 10, 11, 12, 13, 14, 15],      // 16x8 bottom half
        (2, 0, false) => vec![0, 1, 2, 3, 8, 9, 10, 11],         // 8x16 left half
        (2, 1, false) => vec![4, 5, 6, 7, 12, 13, 14, 15],       // 8x16 right half
        (4, 0, _) => vec![0, 1, 2, 3],
        (4, 1, _) => vec![4, 5, 6, 7],
        (4, 2, _) => vec![8, 9, 10, 11],
        (4, 3, _) => vec![12, 13, 14, 15],
        _ => vec![0],
    }
}

/// Top-left 4x4 block index of the given mb partition (used for mvd
/// neighbour sum lookups — the mvd syntax element is read once per
/// partition, and the spec's ctxIdxInc references the top-left 4x4
/// of the partition per §9.3.3.1.1.7).
fn mb_part_top_left_4x4(num_parts: usize, part: usize, is_16x8: bool) -> u8 {
    match (num_parts, part, is_16x8) {
        (1, _, _) => 0,
        (2, 0, true) => 0,
        (2, 1, true) => 8,
        (2, 0, false) => 0,
        (2, 1, false) => 4,
        (4, 0, _) => 0,
        (4, 1, _) => 4,
        (4, 2, _) => 8,
        (4, 3, _) => 12,
        _ => 0,
    }
}

/// Thin wrapper on `cabac_ref_idx_cond_terms` that tolerates a missing grid.
fn neighbour_ref_idx_gt_0(
    grid: Option<&CabacNeighbourGrid>,
    current_mb_addr: u32,
    current_mb: &CabacMbNeighbourInfo,
    list: u8,
    part8x8: u8,
) -> (bool, bool) {
    let Some(g) = grid else { return (false, false) };
    if g.width_in_mbs == 0 { return (false, false) }
    cabac_ref_idx_cond_terms(g, current_mb_addr, current_mb, list, part8x8)
}

/// Thin wrapper on `cabac_mvd_abs_sum` tolerant of a missing grid.
fn neighbour_mvd_abs_sum(
    grid: Option<&CabacNeighbourGrid>,
    current_mb_addr: u32,
    current_mb: &CabacMbNeighbourInfo,
    list: u8,
    comp: MvdComponent,
    blk4x4: u8,
) -> u32 {
    let Some(g) = grid else { return 0 };
    if g.width_in_mbs == 0 { return 0 }
    cabac_mvd_abs_sum(g, current_mb_addr, current_mb, list, comp, blk4x4)
}

// ---------------------------------------------------------------------------
// §7.3.5.2 — sub_mb_pred() parsing.
// ---------------------------------------------------------------------------

fn parse_sub_mb_pred(
    r: &mut BitReader<'_>,
    entropy: &mut EntropyState<'_, '_>,
    mb_type: &MbType,
) -> McblResult<SubMbPred> {
    let mut out = SubMbPred::default();
    let is_b = matches!(mb_type, MbType::B8x8);
    let is_p8x8_ref0 = matches!(mb_type, MbType::P8x8Ref0);
    // OXIDEAV_H264_MB_TRACE=N — per-element trace of the specified MB
    // address inside the current slice. Used for CABAC bisection against
    // the JM trace oracle; see the round-30 notes in the commit history.
    let dbg_mb = std::env::var("OXIDEAV_H264_MB_TRACE").ok()
        .and_then(|v| v.parse::<u32>().ok());
    let dbg_on = dbg_mb == Some(entropy.current_mb_addr);

    // 4 sub_mb_type entries. §7.3.5.2.
    //
    // CABAC: §9.3.3.1.2 / Table 9-39 — ctxIdxOffset = 21 (P/SP) or 36 (B).
    // The bin strings are in Tables 9-37 (P/SP) and 9-38 (B). The concrete
    // decoders live in `cabac_ctx::decode_sub_mb_type_{p,b}` and return a
    // raw u32 that maps directly to the `SubMbType::from_{p,b}` tables
    // (Tables 7-17 / 7-18).
    //
    // Dispatch by slice kind (not `is_b`): a P/SP slice with `mb_type ==
    // P_8x8 / P_8x8ref0` uses the P binarisation (Table 9-37), while a B
    // slice with `mb_type == B_8x8` uses the B binarisation (Table 9-38).
    for i in 0..4 {
        let raw = if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
            match entropy.slice_kind {
                crate::cabac_ctx::SliceKind::B => {
                    crate::cabac_ctx::decode_sub_mb_type_b(dec, ctxs)?
                }
                // P / SP slices use the P/SP binarisation per Table 9-37.
                // I / SI slices never reach sub_mb_pred (sub_mb_pred is
                // only called for P_8x8 / P_8x8ref0 / B_8x8 macroblocks),
                // so defaulting them to the P decoder is safe.
                _ => crate::cabac_ctx::decode_sub_mb_type_p(dec, ctxs)?,
            }
        } else {
            r.ue()?
        };
        out.sub_mb_type[i] = if is_b {
            SubMbType::from_b(raw)?
        } else {
            SubMbType::from_p(raw)?
        };
        if dbg_on {
            eprintln!("[SUB {}] sub_mb_type[{}] raw={} -> {:?}",
                     entropy.current_mb_addr, i, raw, out.sub_mb_type[i]);
        }
    }

    // §7.3.5.2 — ref_idx_l0 gate:
    //   (num_ref_idx_l0_active_minus1 > 0 ||
    //     mb_field_decoding_flag != field_pic_flag) &&
    //   mb_type != P_8x8ref0 &&
    //   sub_mb_type[i] != B_Direct_8x8 &&
    //   SubMbPredMode(sub_mb_type[i]) != Pred_L1
    let mbaff_override = entropy.mbaff_frame_flag;
    let ref_l0_present = entropy.num_ref_idx_l0_active_minus1 > 0 || mbaff_override;
    let ref_l1_present = entropy.num_ref_idx_l1_active_minus1 > 0 || mbaff_override;
    let x_l0 = entropy.num_ref_idx_l0_active_minus1;
    let x_l1 = entropy.num_ref_idx_l1_active_minus1;

    // Scratch neighbour info populated as ref_idx/mvd are decoded.
    let mut curr_nb = CabacMbNeighbourInfo::default();

    for i in 0..4 {
        let sub = out.sub_mb_type[i];
        let mode = sub.sub_mb_pred_mode();
        let is_direct_sub = matches!(sub, SubMbType::BDirect8x8);
        let gated = ref_l0_present
            && !is_p8x8_ref0
            && !is_direct_sub
            && mode != Some(MbPartPredMode::PredL1);
        let v = if gated {
            if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
                let (ca, cb) = neighbour_ref_idx_gt_0(
                    entropy.cabac_nb.as_deref(),
                    entropy.current_mb_addr,
                    &curr_nb,
                    0,
                    i as u8,
                );
                decode_ref_idx_lx(dec, ctxs, ca, cb)?
            } else {
                r.te(x_l0)?
            }
        } else {
            0
        };
        out.ref_idx_l0[i] = v;
        curr_nb.ref_idx_l0[i] = v.min(127) as i8;
    }
    // ref_idx_l1 — B-slice only.
    //   (num_ref_idx_l1_active_minus1 > 0 ||
    //     mb_field_decoding_flag != field_pic_flag) &&
    //   sub_mb_type[i] != B_Direct_8x8 &&
    //   SubMbPredMode(sub_mb_type[i]) != Pred_L0
    if is_b {
        for i in 0..4 {
            let sub = out.sub_mb_type[i];
            let mode = sub.sub_mb_pred_mode();
            let is_direct_sub = matches!(sub, SubMbType::BDirect8x8);
            let gated = ref_l1_present
                && !is_direct_sub
                && mode != Some(MbPartPredMode::PredL0);
            let v = if gated {
                if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
                    let (ca, cb) = neighbour_ref_idx_gt_0(
                        entropy.cabac_nb.as_deref(),
                        entropy.current_mb_addr,
                        &curr_nb,
                        1,
                        i as u8,
                    );
                    decode_ref_idx_lx(dec, ctxs, ca, cb)?
                } else {
                    r.te(x_l1)?
                }
            } else {
                0
            };
            out.ref_idx_l1[i] = v;
            curr_nb.ref_idx_l1[i] = v.min(127) as i8;
        }
    }

    // §7.3.5.2 — mvd_l0 gate: sub_mb_type != B_Direct_8x8 &&
    // SubMbPredMode != Pred_L1. One mvd per NumSubMbPart(sub_mb_type).
    for i in 0..4 {
        let sub = out.sub_mb_type[i];
        let mode = sub.sub_mb_pred_mode();
        let is_direct_sub = matches!(sub, SubMbType::BDirect8x8);
        if !is_direct_sub && mode != Some(MbPartPredMode::PredL1) {
            let n = sub.num_sub_mb_part() as usize;
            for sp in 0..n {
                let (mx, my) = if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
                    let blk4 = sub_mb_sub_part_4x4(i, sp, sub);
                    let sum_x = neighbour_mvd_abs_sum(
                        entropy.cabac_nb.as_deref(),
                        entropy.current_mb_addr,
                        &curr_nb,
                        0,
                        MvdComponent::X,
                        blk4,
                    );
                    let sum_y = neighbour_mvd_abs_sum(
                        entropy.cabac_nb.as_deref(),
                        entropy.current_mb_addr,
                        &curr_nb,
                        0,
                        MvdComponent::Y,
                        blk4,
                    );
                    let mx = decode_mvd_lx(dec, ctxs, MvdComponent::X, sum_x)?;
                    let my = decode_mvd_lx(dec, ctxs, MvdComponent::Y, sum_y)?;
                    if dbg_on {
                        eprintln!("[SUB {}] mvd_l0 blk8={} sp={} blk4={} sum_x={} sum_y={} mx={} my={}",
                                 entropy.current_mb_addr, i, sp, blk4, sum_x, sum_y, mx, my);
                    }
                    for b4 in sub_mb_sub_part_4x4_range(i, sp, sub) {
                        curr_nb.mvd_l0_x[b4 as usize] = mx.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                        curr_nb.mvd_l0_y[b4 as usize] = my.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    }
                    (mx, my)
                } else {
                    (r.se()?, r.se()?)
                };
                out.mvd_l0[i].push([mx, my]);
            }
        }
    }
    // mvd_l1 gate: sub_mb_type != B_Direct_8x8 && SubMbPredMode != Pred_L0.
    if is_b {
        for i in 0..4 {
            let sub = out.sub_mb_type[i];
            let mode = sub.sub_mb_pred_mode();
            let is_direct_sub = matches!(sub, SubMbType::BDirect8x8);
            if !is_direct_sub && mode != Some(MbPartPredMode::PredL0) {
                let n = sub.num_sub_mb_part() as usize;
                for sp in 0..n {
                    let (mx, my) = if let Some((dec, ctxs)) = entropy.cabac.as_mut() {
                        let blk4 = sub_mb_sub_part_4x4(i, sp, sub);
                        let sum_x = neighbour_mvd_abs_sum(
                            entropy.cabac_nb.as_deref(),
                            entropy.current_mb_addr,
                            &curr_nb,
                            1,
                            MvdComponent::X,
                            blk4,
                        );
                        let sum_y = neighbour_mvd_abs_sum(
                            entropy.cabac_nb.as_deref(),
                            entropy.current_mb_addr,
                            &curr_nb,
                            1,
                            MvdComponent::Y,
                            blk4,
                        );
                        let mx = decode_mvd_lx(dec, ctxs, MvdComponent::X, sum_x)?;
                        let my = decode_mvd_lx(dec, ctxs, MvdComponent::Y, sum_y)?;
                        for b4 in sub_mb_sub_part_4x4_range(i, sp, sub) {
                            curr_nb.mvd_l1_x[b4 as usize] = mx.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                            curr_nb.mvd_l1_y[b4 as usize] = my.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                        }
                        (mx, my)
                    } else {
                        (r.se()?, r.se()?)
                    };
                    out.mvd_l1[i].push([mx, my]);
                }
            }
        }
    }

    // Commit sub_mb_pred's partial inter-prediction state to the grid.
    if let Some(grid) = entropy.cabac_nb.as_deref_mut() {
        if let Some(slot) = grid.mbs.get_mut(entropy.current_mb_addr as usize) {
            slot.ref_idx_l0 = curr_nb.ref_idx_l0;
            slot.ref_idx_l1 = curr_nb.ref_idx_l1;
            slot.mvd_l0_x = curr_nb.mvd_l0_x;
            slot.mvd_l0_y = curr_nb.mvd_l0_y;
            slot.mvd_l1_x = curr_nb.mvd_l1_x;
            slot.mvd_l1_y = curr_nb.mvd_l1_y;
        }
    }

    Ok(out)
}

/// Top-left 4x4 block index of the given sub-partition within its 8x8
/// block. 8x8 block `i` (0..=3) spans 4x4 blocks `[4i, 4i+1, 4i+2, 4i+3]`
/// per the conventional sub-layout; within each 8x8, the sub-partitions
/// for sub_mb_types are:
///   8x8   → 1 sub-part (all 4 4x4s).
///   8x4   → 2 sub-parts (top 2 + bottom 2 4x4s).
///   4x8   → 2 sub-parts (left 2 + right 2 4x4s).
///   4x4   → 4 sub-parts.
fn sub_mb_sub_part_4x4(i: usize, sp: usize, sub: SubMbType) -> u8 {
    let base = (i as u8) * 4;
    // 4x4 layout inside an 8x8 block: [base+0 base+1; base+2 base+3]
    // (top-left, top-right, bottom-left, bottom-right).
    let offset = match sub.num_sub_mb_part() {
        1 => 0u8,
        2 => {
            // 8x4: sub-part 0 covers [0,1] (top), 1 covers [2,3] (bot).
            // 4x8: sub-part 0 covers [0,2] (left), 1 covers [1,3] (right).
            if sub.is_8x4() {
                if sp == 0 { 0 } else { 2 }
            } else {
                if sp == 0 { 0 } else { 1 }
            }
        }
        4 => sp as u8,
        _ => 0,
    };
    base + offset
}

/// Range of 4x4 block indices covered by sub-partition `sp` of 8x8 `i`.
fn sub_mb_sub_part_4x4_range(i: usize, sp: usize, sub: SubMbType) -> Vec<u8> {
    let base = (i as u8) * 4;
    match sub.num_sub_mb_part() {
        1 => vec![base, base + 1, base + 2, base + 3],
        2 => {
            if sub.is_8x4() {
                if sp == 0 { vec![base, base + 1] } else { vec![base + 2, base + 3] }
            } else {
                if sp == 0 { vec![base, base + 2] } else { vec![base + 1, base + 3] }
            }
        }
        4 => vec![base + sp as u8],
        _ => vec![base],
    }
}

// ---------------------------------------------------------------------------
// §7.3.5.3 — residual() walker (CAVLC-only in this pass).
// ---------------------------------------------------------------------------

fn pad_to_16(v: Vec<i32>) -> [i32; 16] {
    let mut out = [0i32; 16];
    for (i, x) in v.into_iter().enumerate() {
        if i >= 16 {
            break;
        }
        out[i] = x;
    }
    out
}

/// Count non-zero entries in a decoded residual block — this is the
/// `TotalCoeff(coeff_token)` value the CAVLC engine decoded. We do
/// not modify `parse_residual_block_cavlc` to return the token directly
/// (that would touch `cavlc.rs`, which is out of scope for this
/// change), so we recover the value here. Post §9.2.2 / §9.2.4 a
/// non-zero `coeffLevel[i]` always came from a non-zero transform
/// coefficient level; the reverse also holds, because the levelVal
/// prelude only writes to `coeffLevel[startIdx + zerosLeft + ... ]`
/// for indices covered by `TotalCoeff` decoded levels.
#[inline]
fn count_nonzero(buf: &[i32]) -> u8 {
    buf.iter().filter(|&&v| v != 0).count().min(255) as u8
}

/// §9.2.1.1 / §7.3.5.3.1 — chroma array layout: how many 4x4 chroma
/// blocks per plane. 4 for 4:2:0, 8 for 4:2:2.
#[inline]
fn num_chroma_4x4_blocks(chroma_array_type: u32) -> usize {
    match chroma_array_type {
        1 => 4,
        2 => 8,
        _ => 0,
    }
}

fn parse_residual_cavlc_only(
    r: &mut BitReader<'_>,
    entropy: &mut EntropyState<'_, '_>,
    mb_type: &MbType,
    cbp_luma: u32,
    cbp_chroma: u32,
    transform_size_8x8_flag: bool,
    out: &mut Macroblock,
) -> McblResult<()> {
    let chroma_array_type = entropy.chroma_array_type;
    let current_mb_addr = entropy.current_mb_addr;
    let current_is_intra = mb_type.is_intra();
    let cip = entropy.constrained_intra_pred_flag;

    // Snapshot everything the nC derivation needs. We take a short
    // immutable borrow of the grid (if present) to compute nC values,
    // then release the borrow before writing back this MB's state.
    // Working copy of this MB's own luma totals so in-MB block-to-block
    // neighbour lookups see the progressively-filled counts.
    let mut own_luma_totals: [u8; 16] = [0; 16];
    let mut own_cb_totals: [u8; 8] = [0; 8];
    let mut own_cr_totals: [u8; 8] = [0; 8];
    // §7.3.5.3 — 4:4:4 (ChromaArrayType == 3) chroma is coded like
    // luma: 16 per-plane 4x4 blocks in the same indexing as luma. We
    // track their TotalCoeff values here for nC derivation of the
    // remaining blocks within this MB (§9.2.1.1 step 3).
    let mut own_cb_luma_totals: [u8; 16] = [0; 16];
    let mut own_cr_luma_totals: [u8; 16] = [0; 16];

    // Helper: derive nC for a luma 4x4 block using the neighbour grid
    // plus this MB's own-block progressively-filled totals. We model
    // the "current MB as neighbour" branch manually — the grid doesn't
    // have this MB's entry marked available yet, so a NeighbourSource
    // of `InternalBlock(i)` falls back to `own_luma_totals[i]`.
    let derive_luma = |blk_idx: u8,
                       kind: LumaNcKind,
                       own: &[u8; 16],
                       grid: Option<&CavlcNcGrid>|
     -> i32 {
        let blk = match kind {
            LumaNcKind::Intra16x16Dc => 0u8,
            LumaNcKind::Ac => blk_idx,
        };
        let [a_src, b_src, _c, _d] = neighbour_4x4_map(blk);
        // A:
        let (avail_a, n_a) = match a_src {
            NeighbourSource::InternalBlock(i) => (true, own[i as usize]),
            _ => {
                let Some(g) = grid else { return 0 };
                let (addr, bi) = resolve_luma_neighbour(g, current_mb_addr, a_src);
                nc_nn_luma(g, addr, bi, current_is_intra, cip)
            }
        };
        // B:
        let (avail_b, n_b) = match b_src {
            NeighbourSource::InternalBlock(i) => (true, own[i as usize]),
            _ => {
                let Some(g) = grid else { return 0 };
                let (addr, bi) = resolve_luma_neighbour(g, current_mb_addr, b_src);
                nc_nn_luma(g, addr, bi, current_is_intra, cip)
            }
        };
        combine_nc(avail_a, n_a, avail_b, n_b)
    };

    // Helper: derive nC for a chroma AC 4x4 block.
    let derive_chroma = |blk_idx: u8,
                         is_cr: bool,
                         own_cb: &[u8; 8],
                         own_cr: &[u8; 8],
                         grid: Option<&CavlcNcGrid>|
     -> i32 {
        // Geometry of 4x4 chroma block within its plane.
        let x = ((blk_idx as i32) % 2) * 4;
        let y = ((blk_idx as i32) / 2) * 4;
        let max_w: i32 = 8;
        let max_h: i32 = if chroma_array_type == 2 { 16 } else { 8 };
        let (xa, ya) = (x - 1, y);
        let (xb, yb) = (x, y - 1);

        // Classify each neighbour as internal-to-this-MB vs external.
        // Internal iff (0..max_w).contains(&xN) && (0..max_h).contains(&yN).
        let own_arr = if is_cr { own_cr } else { own_cb };
        let (avail_a, n_a) = if xa >= 0 && xa < max_w && ya >= 0 && ya < max_h {
            // Internal: §6.4.13.2 eq. 6-39.
            let bi = (2 * (ya / 4) + (xa / 4)) as usize;
            (true, own_arr[bi])
        } else {
            let Some(g) = grid else { return 0 };
            let (addr, bi) = resolve_chroma_neighbour(g, current_mb_addr, xa, ya, max_w, max_h);
            nc_nn_chroma(g, addr, bi, is_cr, current_is_intra, cip)
        };
        let (avail_b, n_b) = if xb >= 0 && xb < max_w && yb >= 0 && yb < max_h {
            let bi = (2 * (yb / 4) + (xb / 4)) as usize;
            (true, own_arr[bi])
        } else {
            let Some(g) = grid else { return 0 };
            let (addr, bi) = resolve_chroma_neighbour(g, current_mb_addr, xb, yb, max_w, max_h);
            nc_nn_chroma(g, addr, bi, is_cr, current_is_intra, cip)
        };
        combine_nc(avail_a, n_a, avail_b, n_b)
    };

    // §9.2.1.1 — nC derivation for a Cb/Cr 4x4 block in the 4:4:4
    // (ChromaArrayType == 3) coded-like-luma path. The block layout
    // matches luma (16 blocks, same §6.4.11.4 neighbour map), but the
    // TotalCoeff values come from the per-plane arrays in the
    // neighbour grid (`cb_luma_total_coeff` / `cr_luma_total_coeff`).
    let derive_plane_luma_like = |blk_idx: u8,
                                  is_cr: bool,
                                  own_cb_lt: &[u8; 16],
                                  own_cr_lt: &[u8; 16],
                                  grid: Option<&CavlcNcGrid>|
     -> i32 {
        let [a_src, b_src, _c, _d] = neighbour_4x4_map(blk_idx);
        let own = if is_cr { own_cr_lt } else { own_cb_lt };
        let (avail_a, n_a) = match a_src {
            NeighbourSource::InternalBlock(i) => (true, own[i as usize]),
            _ => {
                let Some(g) = grid else { return 0 };
                let (addr, bi) = resolve_luma_neighbour(g, current_mb_addr, a_src);
                nc_nn_plane_luma_like(g, addr, bi, is_cr, current_is_intra, cip)
            }
        };
        let (avail_b, n_b) = match b_src {
            NeighbourSource::InternalBlock(i) => (true, own[i as usize]),
            _ => {
                let Some(g) = grid else { return 0 };
                let (addr, bi) = resolve_luma_neighbour(g, current_mb_addr, b_src);
                nc_nn_plane_luma_like(g, addr, bi, is_cr, current_is_intra, cip)
            }
        };
        combine_nc(avail_a, n_a, avail_b, n_b)
    };

    // Short immutable borrow for the nC lookups.
    let grid_ref: Option<&CavlcNcGrid> = entropy.cavlc_nc.as_deref();
    let dbg = std::env::var("OXIDEAV_H264_DEBUG").is_ok();

    // --- Luma DC (Intra_16x16) ---
    if mb_type.is_intra_16x16() {
        // §7.3.5.3 — residual_block_cavlc(Intra16x16DCLevel, 0, 15, 16).
        // §9.2.1.1 step 1 + NOTE 2: nC for Intra16x16DCLevel uses
        // luma4x4BlkIdx = 0 and is based on *AC* totals of the
        // neighbours, i.e. the standard luma derivation.
        let nc = derive_luma(0, LumaNcKind::Intra16x16Dc, &own_luma_totals, grid_ref);
        let blk = parse_residual_block_cavlc(r, CoeffTokenContext::Numeric(nc), 0, 15, 16)?;
        out.residual_luma_dc = Some(pad_to_16(blk));
        // NOTE 1 of §9.2.1.1: DC counts do NOT contribute to nN — do
        // not update own_luma_totals here.
    }

    // --- Luma AC / 4x4 blocks / 8x8 blocks ---
    if mb_type.is_intra_16x16() {
        if cbp_luma == 15 {
            for blk_idx in 0..16u8 {
                let nc = derive_luma(blk_idx, LumaNcKind::Ac, &own_luma_totals, grid_ref);
                let blk =
                    parse_residual_block_cavlc(r, CoeffTokenContext::Numeric(nc), 0, 14, 15)?;
                let tc = count_nonzero(&blk);
                own_luma_totals[blk_idx as usize] = tc;
                out.residual_luma.push(pad_to_16(blk));
            }
        }
    } else if transform_size_8x8_flag {
        // 8x8 transform: we still emit four 4x4-sized passes per 8x8
        // block in this reader (see function comment). For nC derivation
        // we use the luma4x4BlkIdx of the per-4x4 sub-block in scan
        // order — that matches §7.3.5.3.1 when the per-4x4 CAVLC path
        // is used; for a strict 8x8 decoder the residual would be a
        // single 64-coeff call, but this code path is a known
        // simplification (see doc comment at top of this module).
        for blk8 in 0..4u8 {
            if (cbp_luma >> blk8) & 1 == 1 {
                for sub in 0..4u8 {
                    let blk_idx = blk8 * 4 + sub;
                    let nc = derive_luma(blk_idx, LumaNcKind::Ac, &own_luma_totals, grid_ref);
                    let blk =
                        parse_residual_block_cavlc(r, CoeffTokenContext::Numeric(nc), 0, 15, 16)?;
                    let tc = count_nonzero(&blk);
                    own_luma_totals[blk_idx as usize] = tc;
                    out.residual_luma.push(pad_to_16(blk));
                }
            }
        }
    } else {
        // 16 4x4 blocks — read only where cbp_luma bit is set (grouped
        // as 4-blocks-per-cbp_luma-bit per §7.3.5.3).
        for blk8 in 0..4u8 {
            if (cbp_luma >> blk8) & 1 == 1 {
                for sub in 0..4u8 {
                    let blk_idx = blk8 * 4 + sub;
                    let nc = derive_luma(blk_idx, LumaNcKind::Ac, &own_luma_totals, grid_ref);
                    if dbg {
                        let (b, bi) = r.position();
                        eprintln!(
                            "[MB {:>4}]   luma4x4 blk={} nc={} cursor=({},{})",
                            current_mb_addr, blk_idx, nc, b, bi
                        );
                    }
                    let blk =
                        parse_residual_block_cavlc(r, CoeffTokenContext::Numeric(nc), 0, 15, 16)?;
                    let tc = count_nonzero(&blk);
                    own_luma_totals[blk_idx as usize] = tc;
                    out.residual_luma.push(pad_to_16(blk));
                }
            }
        }
    }

    // --- Chroma residual ---
    if chroma_array_type == 1 || chroma_array_type == 2 {
        let num_c_blk = num_chroma_4x4_blocks(chroma_array_type);

        // Chroma DC blocks (one per plane). §9.2.1.1 special cases:
        // ChromaDCLevel uses nC = -1 (4:2:0) or -2 (4:2:2).
        if cbp_chroma > 0 {
            let dc_ctx = if chroma_array_type == 1 {
                CoeffTokenContext::ChromaDc420
            } else {
                CoeffTokenContext::ChromaDc422
            };
            let max_dc = if chroma_array_type == 1 { 4 } else { 8 };
            if dbg {
                let (b, bi) = r.position();
                eprintln!(
                    "[MB {:>4}]   chromaDc cb cursor=({},{})",
                    current_mb_addr, b, bi
                );
            }
            let cb_dc = parse_residual_block_cavlc(r, dc_ctx, 0, max_dc - 1, max_dc)?;
            out.residual_chroma_dc_cb = cb_dc;
            if dbg {
                let (b, bi) = r.position();
                eprintln!(
                    "[MB {:>4}]   chromaDc cr cursor=({},{})",
                    current_mb_addr, b, bi
                );
            }
            let cr_dc = parse_residual_block_cavlc(r, dc_ctx, 0, max_dc - 1, max_dc)?;
            out.residual_chroma_dc_cr = cr_dc;
            if dbg {
                let (b, bi) = r.position();
                eprintln!(
                    "[MB {:>4}]   chromaDc done cursor=({},{})",
                    current_mb_addr, b, bi
                );
            }
        }

        // Chroma AC 4x4 blocks when cbp_chroma == 2 (per §7.3.5.3).
        if cbp_chroma == 2 {
            if dbg {
                let (b, bi) = r.position();
                eprintln!(
                    "[MB {:>4}]   chromaAc start cursor=({},{})",
                    current_mb_addr, b, bi
                );
            }
            // Cb plane.
            for blk_idx in 0..num_c_blk as u8 {
                let nc = derive_chroma(
                    blk_idx,
                    /*is_cr=*/ false,
                    &own_cb_totals,
                    &own_cr_totals,
                    grid_ref,
                );
                let blk = parse_residual_block_cavlc(
                    r,
                    CoeffTokenContext::Numeric(nc),
                    0,
                    14,
                    15,
                )?;
                let tc = count_nonzero(&blk);
                own_cb_totals[blk_idx as usize] = tc;
                out.residual_chroma_ac_cb.push(pad_to_16(blk));
            }
            // Cr plane.
            for blk_idx in 0..num_c_blk as u8 {
                let nc = derive_chroma(
                    blk_idx,
                    /*is_cr=*/ true,
                    &own_cb_totals,
                    &own_cr_totals,
                    grid_ref,
                );
                let blk = parse_residual_block_cavlc(
                    r,
                    CoeffTokenContext::Numeric(nc),
                    0,
                    14,
                    15,
                )?;
                let tc = count_nonzero(&blk);
                own_cr_totals[blk_idx as usize] = tc;
                out.residual_chroma_ac_cr.push(pad_to_16(blk));
            }
        }
    } else if chroma_array_type == 0 {
        // Monochrome — no chroma residual.
    } else if chroma_array_type == 3 {
        // §7.3.5.3 — ChromaArrayType == 3: Cb and Cr are each coded
        // like luma — `residual_luma(...)` invoked once per plane.
        // Each plane carries its own 16x16 DC (for Intra_16x16), 16
        // 4x4 AC blocks (or 4 8x8 blocks when transform_size_8x8 is
        // set), gated by cbp_luma in the same way as luma.
        //
        // §9.2.1 — coeff_token nC derivation uses per-plane neighbour
        // nC (the spec points to "neighbouring blocks of the same
        // colour component").
        for plane_is_cr in [false, true] {
            // --- Plane 16x16 DC (Intra_16x16 only) ---
            if mb_type.is_intra_16x16() {
                let nc = derive_plane_luma_like(
                    0,
                    plane_is_cr,
                    &own_cb_luma_totals,
                    &own_cr_luma_totals,
                    grid_ref,
                );
                let blk = parse_residual_block_cavlc(
                    r,
                    CoeffTokenContext::Numeric(nc),
                    0,
                    15,
                    16,
                )?;
                if plane_is_cr {
                    out.residual_cr_16x16_dc = Some(pad_to_16(blk));
                } else {
                    out.residual_cb_16x16_dc = Some(pad_to_16(blk));
                }
                // §9.2.1.1 NOTE 1: DC counts do NOT contribute to nN.
            }

            // --- Plane AC / 4x4 / 8x8 ---
            if mb_type.is_intra_16x16() {
                if cbp_luma == 15 {
                    for blk_idx in 0..16u8 {
                        let nc = derive_plane_luma_like(
                            blk_idx,
                            plane_is_cr,
                            &own_cb_luma_totals,
                            &own_cr_luma_totals,
                            grid_ref,
                        );
                        let blk = parse_residual_block_cavlc(
                            r,
                            CoeffTokenContext::Numeric(nc),
                            0,
                            14,
                            15,
                        )?;
                        let tc = count_nonzero(&blk);
                        if plane_is_cr {
                            own_cr_luma_totals[blk_idx as usize] = tc;
                            out.residual_cr_luma_like.push(pad_to_16(blk));
                        } else {
                            own_cb_luma_totals[blk_idx as usize] = tc;
                            out.residual_cb_luma_like.push(pad_to_16(blk));
                        }
                    }
                }
            } else if transform_size_8x8_flag {
                for blk8 in 0..4u8 {
                    if (cbp_luma >> blk8) & 1 == 1 {
                        for sub in 0..4u8 {
                            let blk_idx = blk8 * 4 + sub;
                            let nc = derive_plane_luma_like(
                                blk_idx,
                                plane_is_cr,
                                &own_cb_luma_totals,
                                &own_cr_luma_totals,
                                grid_ref,
                            );
                            let blk = parse_residual_block_cavlc(
                                r,
                                CoeffTokenContext::Numeric(nc),
                                0,
                                15,
                                16,
                            )?;
                            let tc = count_nonzero(&blk);
                            if plane_is_cr {
                                own_cr_luma_totals[blk_idx as usize] = tc;
                                out.residual_cr_luma_like.push(pad_to_16(blk));
                            } else {
                                own_cb_luma_totals[blk_idx as usize] = tc;
                                out.residual_cb_luma_like.push(pad_to_16(blk));
                            }
                        }
                    }
                }
            } else {
                for blk8 in 0..4u8 {
                    if (cbp_luma >> blk8) & 1 == 1 {
                        for sub in 0..4u8 {
                            let blk_idx = blk8 * 4 + sub;
                            let nc = derive_plane_luma_like(
                                blk_idx,
                                plane_is_cr,
                                &own_cb_luma_totals,
                                &own_cr_luma_totals,
                                grid_ref,
                            );
                            let blk = parse_residual_block_cavlc(
                                r,
                                CoeffTokenContext::Numeric(nc),
                                0,
                                15,
                                16,
                            )?;
                            let tc = count_nonzero(&blk);
                            if plane_is_cr {
                                own_cr_luma_totals[blk_idx as usize] = tc;
                                out.residual_cr_luma_like.push(pad_to_16(blk));
                            } else {
                                own_cb_luma_totals[blk_idx as usize] = tc;
                                out.residual_cb_luma_like.push(pad_to_16(blk));
                            }
                        }
                    }
                }
            }
        }
    } else {
        return Err(MacroblockLayerError::UnsupportedChromaArrayType(
            chroma_array_type,
        ));
    }

    // Write the accumulated own-MB totals back to the grid.
    if let Some(grid) = entropy.cavlc_nc.as_deref_mut() {
        if let Some(slot) = grid.mbs.get_mut(current_mb_addr as usize) {
            slot.luma_total_coeff = own_luma_totals;
            slot.cb_total_coeff = own_cb_totals;
            slot.cr_total_coeff = own_cr_totals;
            slot.cb_luma_total_coeff = own_cb_luma_totals;
            slot.cr_luma_total_coeff = own_cr_luma_totals;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// §7.3.5.3.3 / §9.3.3.1.1.9 — CABAC residual walker.
// ---------------------------------------------------------------------------

/// §7.3.5.3.3 — residual_block_cabac(coeffLevel, startIdx, endIdx, maxNumCoeff).
///
/// Drives the four-stage CABAC residual loop:
///   1. coded_block_flag (§9.3.3.1.1.9 / Table 9-42)
///      — skipped when `maxNumCoeff == 64` and ChromaArrayType != 3
///        (8x8 blocks have coded_block_flag inferred from CBP).
///   2. significance map: for scan position `i ∈ startIdx..numCoeff-1`
///      read `significant_coeff_flag[i]` and, if set,
///      `last_significant_coeff_flag[i]`. Hitting a "last" truncates
///      `numCoeff`.
///   3. Reverse-scan abs level + sign decoding: process positions
///      `numCoeff-1 .. startIdx` decrementing, reading
///      `coeff_abs_level_minus1` and `coeff_sign_flag` for each
///      significant coefficient.
///
/// `neighbour_cbf_{left,above}` are the §9.3.3.1.1.9 condition-term
/// flags for the coded_block_flag ctxIdx, passed in by the caller
/// using the `CabacNeighbourGrid` (when available) or defaulting to
/// the spec's "not available" fallback per §9.3.3.1.1.9 Table 9-42.
#[allow(clippy::too_many_arguments)]
fn parse_residual_block_cabac(
    cabac: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    block_type: BlockType,
    start_idx: u32,
    end_idx: u32,
    max_num_coeff: u32,
    chroma_array_type: u32,
    neighbour_cbf_left: Option<bool>,
    neighbour_cbf_above: Option<bool>,
) -> McblResult<(Vec<i32>, bool)> {
    let len = (end_idx - start_idx + 1) as usize;
    let mut out = vec![0i32; len];

    // §7.3.5.3.3 — coded_block_flag is suppressed for 8x8 blocks unless
    // we're in 4:4:4 (where 8x8 CBP semantics differ).
    let coded = if max_num_coeff != 64 || chroma_array_type == 3 {
        decode_coded_block_flag(cabac, ctxs, block_type, neighbour_cbf_left, neighbour_cbf_above)?
    } else {
        true
    };

    if !coded {
        return Ok((out, false));
    }

    // §7.3.5.3.3 — significance map, in forward scan order.
    // `num_coeff` is the count of coefficients still to decode including
    // the "last" one; it starts at `end_idx - start_idx + 1` and shrinks
    // when `last_significant_coeff_flag[i]` fires.
    let mut significant = vec![false; len];
    let span = end_idx - start_idx + 1;
    let mut num_coeff_in_scan = span; // positions 0..num_coeff_in_scan-1 remain
    let field = false; // frame coding; MBAFF field parsing is deferred
    let mut i: u32 = 0;
    while i + 1 < num_coeff_in_scan {
        let sig = decode_significant_coeff_flag(cabac, ctxs, block_type, i, field)?;
        significant[i as usize] = sig;
        if sig {
            let last = decode_last_significant_coeff_flag(cabac, ctxs, block_type, i, field)?;
            if last {
                num_coeff_in_scan = i + 1;
                break;
            }
        }
        i += 1;
    }
    // The final scan position in `num_coeff_in_scan` is implicitly
    // significant (that's what made the while loop terminate, either via
    // `last_significant_coeff_flag == 1` or by hitting the trailing
    // position that's always the block's last non-zero by construction).
    significant[(num_coeff_in_scan - 1) as usize] = true;

    // §7.3.5.3.3 — reverse-scan abs level + sign decode. The ctxIdx for
    // coeff_abs_level_minus1 bins 0 and 1+ depend on the running counts
    // of previously-decoded |level|=1 and |level|>1 values per §9.3.3.1.3.
    let mut num_eq1: u32 = 0;
    let mut num_gt1: u32 = 0;
    for pos in (0..num_coeff_in_scan).rev() {
        if !significant[pos as usize] {
            continue;
        }
        let abs_m1 =
            decode_coeff_abs_level_minus1(cabac, ctxs, block_type, num_eq1, num_gt1)?;
        let sign = decode_coeff_sign_flag(cabac)?;
        let val = ((abs_m1 + 1) as i32) * if sign { -1 } else { 1 };
        out[pos as usize] = val;
        if abs_m1 == 0 {
            num_eq1 += 1;
        } else {
            num_gt1 += 1;
        }
    }
    Ok((out, true))
}

/// CABAC residual walker (§7.3.5.3 with entropy_coding_mode_flag == 1).
/// Same shape as `parse_residual_cavlc_only` but routes every block
/// through the CABAC engine instead of the CAVLC tables.
#[allow(clippy::too_many_arguments)]
fn parse_residual_cabac_only(
    cabac: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    chroma_array_type: u32,
    mb_type: &MbType,
    cbp_luma: u32,
    cbp_chroma: u32,
    transform_size_8x8_flag: bool,
    out: &mut Macroblock,
    mut nb_grid: Option<&mut CabacNeighbourGrid>,
    current_mb_addr: u32,
    current_is_intra: bool,
) -> McblResult<()> {
    // Scratch for this MB's accumulating CBF state; committed back to
    // the grid at the end of the function.
    let mut curr_cbf = CabacMbNeighbourInfo::default();
    curr_cbf.is_intra = current_is_intra;
    curr_cbf.is_i_pcm = mb_type.is_i_pcm();
    curr_cbf.is_skip = mb_type.is_skip();
    // Snapshot the neighbour grid (immutable) for CBF lookups.
    let grid_snap: Option<CabacNeighbourGrid> = nb_grid.as_deref().cloned();
    let grid_ref: Option<&CabacNeighbourGrid> = grid_snap.as_ref();

    // --- Luma DC (Intra_16x16) ---
    if mb_type.is_intra_16x16() {
        let (ca, cb) = if let Some(g) = grid_ref {
            let (a, b) = cabac_cbf_cond_terms(
                g, current_mb_addr, &curr_cbf, current_is_intra,
                BlockType::Luma16x16Dc, 0, false);
            (Some(a), Some(b))
        } else {
            (None, None)
        };
        let (blk, coded) = parse_residual_block_cabac(
            cabac, ctxs, BlockType::Luma16x16Dc, 0, 15, 16, chroma_array_type, ca, cb,
        )?;
        curr_cbf.cbf_luma_16x16_dc = coded;
        out.residual_luma_dc = Some(pad_to_16(blk));
    }

    // --- Luma AC / 4x4 blocks / 8x8 blocks ---
    if mb_type.is_intra_16x16() {
        if cbp_luma == 15 {
            for blk_idx in 0..16u8 {
                let (ca, cb) = if let Some(g) = grid_ref {
                    let (a, b) = cabac_cbf_cond_terms(
                        g, current_mb_addr, &curr_cbf, current_is_intra,
                        BlockType::Luma16x16Ac, blk_idx, false);
                    (Some(a), Some(b))
                } else {
                    (None, None)
                };
                let (blk, coded) = parse_residual_block_cabac(
                    cabac, ctxs, BlockType::Luma16x16Ac, 0, 14, 15, chroma_array_type, ca, cb,
                )?;
                curr_cbf.cbf_luma_16x16_ac[blk_idx as usize] = coded;
                out.residual_luma.push(pad_to_16(blk));
            }
        }
    } else if transform_size_8x8_flag {
        for blk8 in 0..4u8 {
            if (cbp_luma >> blk8) & 1 == 1 {
                // 8x8 block: CBF is inferred from CBP (not decoded), so
                // no neighbour lookup needed for the CBF itself.
                let (blk, coded) = parse_residual_block_cabac(
                    cabac, ctxs, BlockType::Luma8x8, 0, 63, 64, chroma_array_type, None, None,
                )?;
                // Propagate the inferred coded flag to the four 4x4
                // sub-blocks for CBF neighbour tracking.
                for sub in 0..4u8 {
                    let idx = (blk8 * 4 + sub) as usize;
                    curr_cbf.cbf_luma_4x4[idx] = coded;
                }
                // Pad the 64 coefficients into four 16-slot arrays.
                for sub in 0..4 {
                    let mut chunk = [0i32; 16];
                    for k in 0..16 {
                        chunk[k] = blk[sub * 16 + k];
                    }
                    out.residual_luma.push(chunk);
                }
            }
        }
    } else {
        for blk8 in 0..4u8 {
            if (cbp_luma >> blk8) & 1 == 1 {
                for sub in 0..4u8 {
                    let blk_idx = blk8 * 4 + sub;
                    let (ca, cb) = if let Some(g) = grid_ref {
                        let (a, b) = cabac_cbf_cond_terms(
                            g, current_mb_addr, &curr_cbf, current_is_intra,
                            BlockType::Luma4x4, blk_idx, false);
                        (Some(a), Some(b))
                    } else {
                        (None, None)
                    };
                    let (blk, coded) = parse_residual_block_cabac(
                        cabac, ctxs, BlockType::Luma4x4, 0, 15, 16, chroma_array_type, ca, cb,
                    )?;
                    curr_cbf.cbf_luma_4x4[blk_idx as usize] = coded;
                    out.residual_luma.push(pad_to_16(blk));
                }
            }
        }
    }

    // --- Chroma residual ---
    if chroma_array_type == 1 || chroma_array_type == 2 {
        let num_c8x8 = if chroma_array_type == 1 { 1 } else { 2 };

        if cbp_chroma > 0 {
            let max_dc = if chroma_array_type == 1 { 4 } else { 8 };
            // Chroma DC — MB-level block. §9.3.3.1.1.9 cat=3 derives
            // condTermN from the iCbCr-matched neighbour DC block.
            // Compute separately for Cb (iCbCr=0) and Cr (iCbCr=1).
            let (ca_cb, cb_cb) = if let Some(g) = grid_ref {
                let (a, b) = cabac_cbf_cond_terms(
                    g, current_mb_addr, &curr_cbf, current_is_intra,
                    BlockType::ChromaDc, 0, false);
                (Some(a), Some(b))
            } else {
                (None, None)
            };
            let (cb_dc, cb_coded) = parse_residual_block_cabac(
                cabac, ctxs, BlockType::ChromaDc, 0, max_dc - 1, max_dc, chroma_array_type,
                ca_cb, cb_cb,
            )?;
            curr_cbf.cbf_cb_dc = cb_coded;
            out.residual_chroma_dc_cb = cb_dc;

            let (ca_cr, cb_cr) = if let Some(g) = grid_ref {
                let (a, b) = cabac_cbf_cond_terms(
                    g, current_mb_addr, &curr_cbf, current_is_intra,
                    BlockType::ChromaDc, 0, true);
                (Some(a), Some(b))
            } else {
                (None, None)
            };
            let (cr_dc, cr_coded) = parse_residual_block_cabac(
                cabac, ctxs, BlockType::ChromaDc, 0, max_dc - 1, max_dc, chroma_array_type,
                ca_cr, cb_cr,
            )?;
            curr_cbf.cbf_cr_dc = cr_coded;
            out.residual_chroma_dc_cr = cr_dc;
        }

        if cbp_chroma == 2 {
            let num_ac = 4 * num_c8x8 as usize;
            for blk_idx in 0..num_ac as u8 {
                let (ca, cb) = if let Some(g) = grid_ref {
                    let (a, b) = cabac_cbf_cond_terms(
                        g, current_mb_addr, &curr_cbf, current_is_intra,
                        BlockType::ChromaAc, blk_idx, false);
                    (Some(a), Some(b))
                } else {
                    (None, None)
                };
                let (blk, coded) = parse_residual_block_cabac(
                    cabac, ctxs, BlockType::ChromaAc, 0, 14, 15, chroma_array_type, ca, cb,
                )?;
                curr_cbf.cbf_cb_ac[blk_idx as usize] = coded;
                out.residual_chroma_ac_cb.push(pad_to_16(blk));
            }
            for blk_idx in 0..num_ac as u8 {
                let (ca, cb) = if let Some(g) = grid_ref {
                    let (a, b) = cabac_cbf_cond_terms(
                        g, current_mb_addr, &curr_cbf, current_is_intra,
                        BlockType::ChromaAc, blk_idx, true);
                    (Some(a), Some(b))
                } else {
                    (None, None)
                };
                let (blk, coded) = parse_residual_block_cabac(
                    cabac, ctxs, BlockType::ChromaAc, 0, 14, 15, chroma_array_type, ca, cb,
                )?;
                curr_cbf.cbf_cr_ac[blk_idx as usize] = coded;
                out.residual_chroma_ac_cr.push(pad_to_16(blk));
            }
        }
    } else if chroma_array_type == 0 {
        // Monochrome — no chroma residual.
    } else if chroma_array_type == 3 {
        // §7.3.5.3 — ChromaArrayType == 3: each plane is coded like
        // luma. Use the per-plane CABAC block types with per-plane CBF
        // neighbour tracking (§9.3.3.1.1.9 Table 9-42 ctxBlockCat:
        // Cb DC=6, Cb AC=7, Cb 4x4=8, Cb 8x8=9,
        // Cr DC=10, Cr AC=11, Cr 4x4=12, Cr 8x8=13).
        for plane_is_cr in [false, true] {
            let (bt_dc, bt_ac, bt_4x4, bt_8x8) = if plane_is_cr {
                (
                    BlockType::CrIntra16x16Dc,
                    BlockType::CrIntra16x16Ac,
                    BlockType::CrLuma4x4,
                    BlockType::Cr8x8,
                )
            } else {
                (
                    BlockType::CbIntra16x16Dc,
                    BlockType::CbIntra16x16Ac,
                    BlockType::CbLuma4x4,
                    BlockType::Cb8x8,
                )
            };

            // --- Plane 16x16 DC (Intra_16x16 only) ---
            if mb_type.is_intra_16x16() {
                let (ca, cb) = if let Some(g) = grid_ref {
                    let (a, b) = cabac_cbf_cond_terms(
                        g, current_mb_addr, &curr_cbf, current_is_intra,
                        bt_dc, 0, plane_is_cr);
                    (Some(a), Some(b))
                } else {
                    (None, None)
                };
                let (blk, coded) = parse_residual_block_cabac(
                    cabac, ctxs, bt_dc, 0, 15, 16, chroma_array_type, ca, cb,
                )?;
                if plane_is_cr {
                    curr_cbf.cbf_cr_16x16_dc = coded;
                    out.residual_cr_16x16_dc = Some(pad_to_16(blk));
                } else {
                    curr_cbf.cbf_cb_16x16_dc = coded;
                    out.residual_cb_16x16_dc = Some(pad_to_16(blk));
                }
            }

            // --- Plane AC / 4x4 / 8x8 ---
            if mb_type.is_intra_16x16() {
                if cbp_luma == 15 {
                    for blk_idx in 0..16u8 {
                        let (ca, cb) = if let Some(g) = grid_ref {
                            let (a, b) = cabac_cbf_cond_terms(
                                g, current_mb_addr, &curr_cbf, current_is_intra,
                                bt_ac, blk_idx, plane_is_cr);
                            (Some(a), Some(b))
                        } else {
                            (None, None)
                        };
                        let (blk, coded) = parse_residual_block_cabac(
                            cabac, ctxs, bt_ac, 0, 14, 15, chroma_array_type, ca, cb,
                        )?;
                        if plane_is_cr {
                            curr_cbf.cbf_cr_16x16_ac[blk_idx as usize] = coded;
                            out.residual_cr_luma_like.push(pad_to_16(blk));
                        } else {
                            curr_cbf.cbf_cb_16x16_ac[blk_idx as usize] = coded;
                            out.residual_cb_luma_like.push(pad_to_16(blk));
                        }
                    }
                }
            } else if transform_size_8x8_flag {
                for blk8 in 0..4u8 {
                    if (cbp_luma >> blk8) & 1 == 1 {
                        // 8x8 block: CBF is inferred from CBP in 4:4:4
                        // (max_num_coeff == 64 AND chroma_array_type
                        // == 3 → coded_block_flag IS coded per
                        // §7.3.5.3.3).
                        let (blk, coded) = parse_residual_block_cabac(
                            cabac, ctxs, bt_8x8, 0, 63, 64, chroma_array_type,
                            None, None,
                        )?;
                        // Propagate coded flag to the four 4x4 sub-
                        // blocks so subsequent intra-MB neighbour
                        // lookups see it.
                        for sub in 0..4u8 {
                            let idx = (blk8 * 4 + sub) as usize;
                            if plane_is_cr {
                                curr_cbf.cbf_cr_luma_4x4[idx] = coded;
                            } else {
                                curr_cbf.cbf_cb_luma_4x4[idx] = coded;
                            }
                        }
                        for sub in 0..4 {
                            let mut chunk = [0i32; 16];
                            for k in 0..16 {
                                chunk[k] = blk[sub * 16 + k];
                            }
                            if plane_is_cr {
                                out.residual_cr_luma_like.push(chunk);
                            } else {
                                out.residual_cb_luma_like.push(chunk);
                            }
                        }
                    }
                }
            } else {
                for blk8 in 0..4u8 {
                    if (cbp_luma >> blk8) & 1 == 1 {
                        for sub in 0..4u8 {
                            let blk_idx = blk8 * 4 + sub;
                            let (ca, cb) = if let Some(g) = grid_ref {
                                let (a, b) = cabac_cbf_cond_terms(
                                    g, current_mb_addr, &curr_cbf, current_is_intra,
                                    bt_4x4, blk_idx, plane_is_cr);
                                (Some(a), Some(b))
                            } else {
                                (None, None)
                            };
                            let (blk, coded) = parse_residual_block_cabac(
                                cabac, ctxs, bt_4x4, 0, 15, 16, chroma_array_type, ca, cb,
                            )?;
                            if plane_is_cr {
                                curr_cbf.cbf_cr_luma_4x4[blk_idx as usize] = coded;
                                out.residual_cr_luma_like.push(pad_to_16(blk));
                            } else {
                                curr_cbf.cbf_cb_luma_4x4[blk_idx as usize] = coded;
                                out.residual_cb_luma_like.push(pad_to_16(blk));
                            }
                        }
                    }
                }
            }
        }
    } else {
        return Err(MacroblockLayerError::UnsupportedChromaArrayType(
            chroma_array_type,
        ));
    }

    // Commit this MB's final CBF state (and availability) to the grid.
    if let Some(grid) = nb_grid.as_deref_mut() {
        if let Some(slot) = grid.mbs.get_mut(current_mb_addr as usize) {
            slot.available = true;
            slot.is_intra = current_is_intra;
            slot.is_i_pcm = mb_type.is_i_pcm();
            slot.is_skip = mb_type.is_skip();
            slot.cbf_luma_4x4 = curr_cbf.cbf_luma_4x4;
            slot.cbf_cb_dc = curr_cbf.cbf_cb_dc;
            slot.cbf_cr_dc = curr_cbf.cbf_cr_dc;
            slot.cbf_cb_ac = curr_cbf.cbf_cb_ac;
            slot.cbf_cr_ac = curr_cbf.cbf_cr_ac;
            slot.cbf_luma_16x16_dc = curr_cbf.cbf_luma_16x16_dc;
            slot.cbf_luma_16x16_ac = curr_cbf.cbf_luma_16x16_ac;
            slot.cbf_cb_16x16_dc = curr_cbf.cbf_cb_16x16_dc;
            slot.cbf_cb_16x16_ac = curr_cbf.cbf_cb_16x16_ac;
            slot.cbf_cb_luma_4x4 = curr_cbf.cbf_cb_luma_4x4;
            slot.cbf_cr_16x16_dc = curr_cbf.cbf_cr_16x16_dc;
            slot.cbf_cr_16x16_ac = curr_cbf.cbf_cr_16x16_ac;
            slot.cbf_cr_luma_4x4 = curr_cbf.cbf_cr_luma_4x4;
            // is_i_pcm sets all CBF to 1 per §9.3.3.1.1.9.
            if mb_type.is_i_pcm() {
                slot.cbf_luma_4x4 = [true; 16];
                slot.cbf_cb_dc = true;
                slot.cbf_cr_dc = true;
                slot.cbf_cb_ac = [true; 8];
                slot.cbf_cr_ac = [true; 8];
                slot.cbf_luma_16x16_dc = true;
                slot.cbf_luma_16x16_ac = [true; 16];
                slot.cbf_cb_16x16_dc = true;
                slot.cbf_cb_16x16_ac = [true; 16];
                slot.cbf_cb_luma_4x4 = [true; 16];
                slot.cbf_cr_16x16_dc = true;
                slot.cbf_cr_16x16_ac = [true; 16];
                slot.cbf_cr_luma_4x4 = [true; 16];
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal::{NalHeader, NalUnitType};

    /// Hand-rolled MSB-first bit writer (mirrors the helpers in sps.rs
    /// / slice_header.rs tests).
    struct BitWriter {
        bytes: Vec<u8>,
        bit_pos: u8,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                bit_pos: 0,
            }
        }
        fn u(&mut self, bits: u32, value: u32) {
            for i in (0..bits).rev() {
                let bit = ((value >> i) & 1) as u8;
                if self.bit_pos == 0 {
                    self.bytes.push(0);
                }
                let idx = self.bytes.len() - 1;
                self.bytes[idx] |= bit << (7 - self.bit_pos);
                self.bit_pos = (self.bit_pos + 1) % 8;
            }
        }
        fn ue(&mut self, value: u32) {
            let v = value + 1;
            let leading = 31 - v.leading_zeros();
            for _ in 0..leading {
                self.u(1, 0);
            }
            self.u(leading + 1, v);
        }
        fn se(&mut self, value: i32) {
            let k = if value <= 0 {
                (-2 * value) as u32
            } else {
                (2 * value - 1) as u32
            };
            self.ue(k);
        }
        fn trailing(&mut self) {
            self.u(1, 1);
            while self.bit_pos != 0 {
                self.u(1, 0);
            }
        }
        fn into_bytes(self) -> Vec<u8> {
            self.bytes
        }
    }

    pub(super) fn dummy_sps() -> Sps {
        Sps {
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
            pic_width_in_mbs_minus1: 19,
            pic_height_in_map_units_minus1: 14,
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: true,
            frame_cropping: None,
            vui_parameters_present_flag: false,
            vui: None,
        }
    }

    pub(super) fn dummy_pps() -> Pps {
        Pps {
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

    pub(super) fn dummy_slice_header(slice_type: SliceType) -> SliceHeader {
        use crate::slice_header::{DecRefPicMarking, RefPicListModification};
        SliceHeader {
            first_mb_in_slice: 0,
            slice_type_raw: match slice_type {
                SliceType::P => 0,
                SliceType::B => 1,
                SliceType::I => 2,
                SliceType::SP => 3,
                SliceType::SI => 4,
            },
            slice_type,
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
            dec_ref_pic_marking: Some(DecRefPicMarking {
                no_output_of_prior_pics_flag: false,
                long_term_reference_flag: false,
                adaptive_marking: None,
            }),
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

    pub(super) fn _nal(t: NalUnitType, ref_idc: u8) -> NalHeader {
        NalHeader {
            forbidden_zero_bit: 0,
            nal_ref_idc: ref_idc,
            nal_unit_type: t,
        }
    }

    // -----------------------------------------------------------------
    // MbType mapping tests (Tables 7-11 / 7-13 / 7-14).
    // -----------------------------------------------------------------

    #[test]
    fn i_slice_mb_type_0_is_i_nxn() {
        assert_eq!(MbType::from_i_slice(0).unwrap(), MbType::INxN);
    }

    #[test]
    fn i_slice_mb_type_25_is_i_pcm() {
        assert_eq!(MbType::from_i_slice(25).unwrap(), MbType::IPcm);
    }

    #[test]
    fn i_slice_mb_type_intra_16x16_rows_map_derived_fields() {
        // Row 1 → Intra_16x16_0_0_0 (pred=0, cbp_luma=0, cbp_chroma=0).
        assert_eq!(
            MbType::from_i_slice(1).unwrap(),
            MbType::Intra16x16(Intra16x16 {
                pred_mode: 0,
                cbp_luma: 0,
                cbp_chroma: 0,
            })
        );
        // Row 13 → Intra_16x16_0_0_15 (pred=0, cbp_luma=15, cbp_chroma=0).
        assert_eq!(
            MbType::from_i_slice(13).unwrap(),
            MbType::Intra16x16(Intra16x16 {
                pred_mode: 0,
                cbp_luma: 15,
                cbp_chroma: 0,
            })
        );
        // Row 24 → Intra_16x16_3_2_15 (pred=3, cbp_luma=15, cbp_chroma=2).
        assert_eq!(
            MbType::from_i_slice(24).unwrap(),
            MbType::Intra16x16(Intra16x16 {
                pred_mode: 3,
                cbp_luma: 15,
                cbp_chroma: 2,
            })
        );
    }

    #[test]
    fn p_slice_mb_type_maps_to_inter_or_intra() {
        assert_eq!(MbType::from_p_slice(0).unwrap(), MbType::PL016x16);
        assert_eq!(MbType::from_p_slice(3).unwrap(), MbType::P8x8);
        assert_eq!(MbType::from_p_slice(5).unwrap(), MbType::INxN); // 5 + 0 = intra I_NxN
    }

    #[test]
    fn b_slice_mb_type_maps_to_inter_or_intra() {
        assert_eq!(MbType::from_b_slice(0).unwrap(), MbType::BDirect16x16);
        assert_eq!(MbType::from_b_slice(1).unwrap(), MbType::BL016x16);
        assert_eq!(MbType::from_b_slice(22).unwrap(), MbType::B8x8);
        assert_eq!(MbType::from_b_slice(23).unwrap(), MbType::INxN);
    }

    #[test]
    fn mb_type_out_of_range_rejected() {
        assert_eq!(
            MbType::from_i_slice(26).unwrap_err(),
            MacroblockLayerError::MbTypeOutOfRange(26)
        );
    }

    #[test]
    fn num_mb_part_matches_spec() {
        assert_eq!(MbType::PL016x16.num_mb_part(), 1);
        assert_eq!(MbType::PL0L016x8.num_mb_part(), 2);
        assert_eq!(MbType::P8x8.num_mb_part(), 4);
        assert_eq!(MbType::BBiBi16x8.num_mb_part(), 2);
        assert_eq!(MbType::B8x8.num_mb_part(), 4);
        assert!(MbType::P8x8.is_sub_mb_path());
        assert!(!MbType::PL016x16.is_sub_mb_path());
    }

    // -----------------------------------------------------------------
    // me(v) / CBP mapping spot checks.
    // -----------------------------------------------------------------

    #[test]
    fn me_cbp_420_intra_codeword_0_maps_to_47() {
        // Table 9-4(a) row for ChromaArrayType ∈ {1,2} intra codeNum=0
        // is 47 (= 0b10_1111 → cbp_luma=15, cbp_chroma=2).
        assert_eq!(ME_INTRA_420_422[0], 47);
    }

    #[test]
    fn me_cbp_420_inter_codeword_0_maps_to_0() {
        // Table 9-4(a) inter codeNum=0 is 0 (all-zero CBP).
        assert_eq!(ME_INTER_420_422[0], 0);
    }

    // -----------------------------------------------------------------
    // parse_macroblock dispatch tests.
    // -----------------------------------------------------------------

    /// Build a minimal EntropyState for CAVLC tests.
    fn cavlc_entropy() -> EntropyState<'static, 'static> {
        EntropyState {
            cabac: None,
            slice_kind: SliceKind::I,
            neighbours: NeighbourCtx::default(),
            prev_mb_qp_delta_nonzero: false,
            chroma_array_type: 1,
            transform_8x8_mode_flag: false,
            cavlc_nc: None,
            current_mb_addr: 0,
            constrained_intra_pred_flag: false,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            mbaff_frame_flag: false,
            cabac_nb: None,
            pic_width_in_mbs: 0,
        }
    }

    /// §7.3.5 regression: when mb_type == I_NxN and the PPS has
    /// `transform_8x8_mode_flag == 1`, a `transform_size_8x8_flag` bit
    /// must be read BEFORE `mb_pred(mb_type)` so the pred-mode loop can
    /// iterate 4 Intra_8x8 entries (not 16 Intra_4x4 entries). Before
    /// the 2026-04-20 fix this flag was (incorrectly) read AFTER
    /// `mb_pred` and after CBP, which guaranteed a bit-misalignment
    /// for any High/Extended-profile stream with 8x8 transform on.
    #[test]
    fn cavlc_i_nxn_reads_transform_size_8x8_flag_before_mb_pred() {
        // Build a minimal CAVLC bitstream for I_NxN with the 8x8 path:
        //   mb_type = 0 → ue(v) "1"
        //   transform_size_8x8_flag = 1 → u(1) "1"  (BEFORE mb_pred)
        //   4 prev_intra8x8_pred_mode_flag = 1 → "1" × 4  (not 16)
        //   intra_chroma_pred_mode = 0 → ue(v) "1"
        //   coded_block_pattern = 0 (intra codeNum=3) → "00100"
        //   (cbp=0 → no mb_qp_delta, no residual)
        // If the fix is absent, `mb_pred` will read 16 u(1)s and then
        // fail downstream because the cursor is in the wrong place.
        let mut w = BitWriter::new();
        w.ue(0); // mb_type = I_NxN
        w.u(1, 1); // transform_size_8x8_flag = 1
        for _ in 0..4 {
            w.u(1, 1); // prev_intra8x8_pred_mode_flag
        }
        w.ue(0); // intra_chroma_pred_mode
        w.ue(3); // coded_block_pattern codeNum=3 → CBP=0 (intra)
        w.trailing();
        let bytes = w.into_bytes();

        let sps = dummy_sps();
        let mut pps = dummy_pps();
        // §7.3.5 requires transform_8x8_mode_flag == 1 to exercise the
        // new code path. Our dummy PPS has no transform_8x8 control by
        // default; patch via the accessor our tests rely on.
        pps.extension = Some(crate::pps::PpsExtension {
            transform_8x8_mode_flag: true,
            pic_scaling_matrix_present_flag: false,
            pic_scaling_lists: None,
            second_chroma_qp_index_offset: 0,
        });
        let hdr = dummy_slice_header(SliceType::I);
        let mut r = BitReader::new(&bytes);
        let mut entropy = cavlc_entropy();
        entropy.transform_8x8_mode_flag = true;

        let mb = parse_macroblock(&mut r, &mut entropy, &hdr, &sps, &pps, 0).unwrap();
        assert_eq!(mb.mb_type, MbType::INxN);
        assert!(mb.transform_size_8x8_flag);
        assert_eq!(mb.coded_block_pattern, 0);
        // mb_pred should have filled the 8x8 flags, not the 4x4 ones.
        let pred = mb.mb_pred.expect("mb_pred present");
        for i in 0..4 {
            assert!(
                pred.prev_intra8x8_pred_mode_flag[i],
                "prev_intra8x8_pred_mode_flag[{i}] should have been parsed as 1"
            );
        }
    }

    #[test]
    fn cavlc_i_nxn_with_all_zero_residual() {
        // Build a CAVLC bitstream for a minimal I_NxN macroblock:
        //   mb_type = 0              → ue(v) "1"
        //   16 prev_intra4x4 flags   → u(1) "1" each (use predicted mode)
        //   intra_chroma_pred_mode=0 → ue(v) "1"
        //   coded_block_pattern      → ue(v) codeNum=3 (→ intra CBP = 0)
        //       codeNum=3 "00100"
        //   (cbp_luma=0, cbp_chroma=0 → no mb_qp_delta, no residual)
        // NOTE: table row codeNum=3 → 0 for intra (index 3 in ME_INTRA_420_422).
        assert_eq!(ME_INTRA_420_422[3], 0);

        let mut w = BitWriter::new();
        w.ue(0); // mb_type
        for _ in 0..16 {
            w.u(1, 1); // prev_intra4x4_pred_mode_flag
        }
        w.ue(0); // intra_chroma_pred_mode (= 0)
        w.ue(3); // coded_block_pattern codeNum=3 → CBP=0
        w.trailing();
        let bytes = w.into_bytes();

        let sps = dummy_sps();
        let pps = dummy_pps();
        let hdr = dummy_slice_header(SliceType::I);
        let mut r = BitReader::new(&bytes);
        let mut entropy = cavlc_entropy();

        let mb = parse_macroblock(&mut r, &mut entropy, &hdr, &sps, &pps, 0).unwrap();
        assert_eq!(mb.mb_type, MbType::INxN);
        assert_eq!(mb.coded_block_pattern, 0);
        assert!(mb.pcm_samples.is_none());
        assert!(mb.mb_pred.is_some());
        assert!(mb.residual_luma.is_empty());
        assert_eq!(mb.mb_qp_delta, 0);
        assert!(mb.residual_luma_dc.is_none());
    }

    #[test]
    fn cavlc_intra_16x16_dc_cbp_zero() {
        // Intra_16x16 with pred_mode=0, cbp_luma=0, cbp_chroma=0
        // (mb_type = 1 on I slice).
        //   mb_type=1 → ue(v) codeNum=1 "010"
        //   intra_chroma_pred_mode = 0 → ue(v) "1"
        //   (no coded_block_pattern — Intra_16x16 carries it via mb_type)
        //   mb_qp_delta = 0 → se(v) codeNum=0 "1"
        //   Intra_16x16 DC residual: single residual_block_cavlc with
        //     coeff_token (TotalCoeff=0) → codeword "1" in col nC<2.
        //   cbp_luma=0 → no AC blocks; cbp_chroma=0 → no chroma.
        let mut w = BitWriter::new();
        w.ue(1); // mb_type = 1
        w.ue(0); // intra_chroma_pred_mode
        w.se(0); // mb_qp_delta
        w.u(1, 1); // Intra_16x16 DC coeff_token = (TotalCoeff=0, T1=0) "1"
        w.trailing();
        let bytes = w.into_bytes();

        let sps = dummy_sps();
        let pps = dummy_pps();
        let hdr = dummy_slice_header(SliceType::I);
        let mut r = BitReader::new(&bytes);
        let mut entropy = cavlc_entropy();

        let mb = parse_macroblock(&mut r, &mut entropy, &hdr, &sps, &pps, 0).unwrap();
        assert!(matches!(
            mb.mb_type,
            MbType::Intra16x16(Intra16x16 {
                pred_mode: 0,
                cbp_luma: 0,
                cbp_chroma: 0
            })
        ));
        assert_eq!(mb.coded_block_pattern, 0);
        assert!(mb.residual_luma_dc.is_some());
        assert!(mb.residual_luma.is_empty());
        assert_eq!(mb.mb_qp_delta, 0);
    }

    #[test]
    fn cavlc_i_pcm_path_reads_256_plus_chroma_samples() {
        //   mb_type=25 (I_PCM) → ue(v) codeNum=25
        //     25+1=26 (bin 11010) → 4 leading zeros + suffix 1_1010
        //     Bit string: "0000 1_1010"
        //   pcm_alignment_zero_bit until byte-aligned (4 bits after the
        //     9-bit ue above).
        //   256 u(8) luma + 64 u(8) Cb + 64 u(8) Cr.
        let mut w = BitWriter::new();
        w.ue(25); // mb_type
                  // Align to byte.
        while w.bit_pos != 0 {
            w.u(1, 0);
        }
        for i in 0..256 {
            w.u(8, (i & 0xFF) as u32);
        }
        for i in 0..64 {
            w.u(8, (i & 0xFF) as u32);
        }
        for i in 0..64 {
            w.u(8, ((i * 2) & 0xFF) as u32);
        }
        // (Real streams would include rbsp_trailing_bits but we don't
        //  require them in parse_macroblock; the slice_data walker
        //  handles trailing bits separately.)
        let bytes = w.into_bytes();
        let sps = dummy_sps();
        let pps = dummy_pps();
        let hdr = dummy_slice_header(SliceType::I);
        let mut r = BitReader::new(&bytes);
        let mut entropy = cavlc_entropy();

        let mb = parse_macroblock(&mut r, &mut entropy, &hdr, &sps, &pps, 0).unwrap();
        assert!(mb.mb_type.is_i_pcm());
        let pcm = mb.pcm_samples.as_ref().unwrap();
        assert_eq!(pcm.luma.len(), 256);
        assert_eq!(pcm.chroma_cb.len(), 64);
        assert_eq!(pcm.chroma_cr.len(), 64);
        assert_eq!(pcm.luma[1], 1);
        assert_eq!(pcm.chroma_cb[10], 10);
        assert_eq!(pcm.chroma_cr[3], 6);
    }

    // -----------------------------------------------------------------
    // §9.2.1.1 — CAVLC nC neighbour derivation tests.
    // -----------------------------------------------------------------

    /// Populate `grid.mbs[addr]` as a fully-available non-skip, non-PCM
    /// intra MB with the given per-block total_coeff values.
    fn fill_mb(grid: &mut CavlcNcGrid, addr: u32, luma_totals: [u8; 16]) {
        let slot = &mut grid.mbs[addr as usize];
        slot.is_available = true;
        slot.is_skip = false;
        slot.is_i_pcm = false;
        slot.is_intra = true;
        slot.luma_total_coeff = luma_totals;
    }

    #[test]
    fn nc_top_left_mb_both_neighbours_unavailable() {
        // §9.2.1.1 step 7: availableFlagA == 0 && availableFlagB == 0 → nC = 0.
        // First MB of the slice, block 0: neither A (outside left) nor
        // B (outside top) is available.
        let grid = CavlcNcGrid::new(4, 4);
        let nc = derive_nc_luma(&grid, 0, 0, LumaNcKind::Ac, true, false);
        assert_eq!(nc, 0);
    }

    #[test]
    fn nc_mb0_block5_uses_in_mb_neighbours_once_filled() {
        // §6.4.11.4 — block 5's A = block 4 (internal), B = block 1
        // (internal). When both neighbours-in-grid are not yet written
        // (since we only fill mbAddr=0 after the MB is done), the
        // derivation we do for in-MB neighbours is via the "own"
        // totals (passed into the closure) — exercised by the integration
        // path. For the grid-only helper, with nothing filled, these are
        // InternalBlock(4) / InternalBlock(1) → grid_ref doesn't have
        // availability because it's the *current* MB being decoded; the
        // grid entry for mb_addr=0 has is_available=false so the result
        // is nC=0 — which is the expected "nothing decoded yet" state.
        let grid = CavlcNcGrid::new(4, 4);
        let nc = derive_nc_luma(&grid, 0, 5, LumaNcKind::Ac, true, false);
        // Both neighbours are InternalBlock; they resolve to current MB;
        // current MB hasn't been marked available yet → nC = 0.
        assert_eq!(nc, 0);
    }

    #[test]
    fn nc_mb1_block0_uses_left_mb_block_per_6_4_11_4() {
        // §6.4.11.4 — MB 1 block 0 in a 4x4 MB grid: A = left MB
        // block 5 (eq. 6-38 at wrapped (xW=15, yW=0) = 8*0 + 4*1 + 0 +
        // 7/4 = 5), B unavailable (top row).
        let mut grid = CavlcNcGrid::new(4, 4);
        let mut totals = [0u8; 16];
        totals[5] = 5; // block 5 of MB 0 had total_coeff = 5
        fill_mb(&mut grid, 0, totals);
        let nc = derive_nc_luma(&grid, /*mb_addr=*/ 1, /*blk=*/ 0, LumaNcKind::Ac, true, false);
        // availableFlagA=1 (nA=5), availableFlagB=0 → nC = nA = 5.
        assert_eq!(nc, 5);
    }

    #[test]
    fn nc_mb4_block0_uses_above_mb_block10() {
        // MB at (0, 1) in a 4x4 grid = addr 4. Block 0: A unavailable
        // (left edge), B = MB 0's block at bottom row, first column.
        // Figure 6-10: bottom-row-left block indices are 10, 11, 14, 15;
        // eq. 6-38 for (xW=0, yW=15) = 8*(15/8) + 4*0 + 2*((15%8)/4) +
        // (0%8)/4 = 8 + 0 + 2 + 0 = 10.
        let mut grid = CavlcNcGrid::new(4, 4);
        let mut totals = [0u8; 16];
        totals[10] = 7;
        fill_mb(&mut grid, 0, totals);
        let nc = derive_nc_luma(&grid, /*mb_addr=*/ 4, /*blk=*/ 0, LumaNcKind::Ac, true, false);
        // availableFlagA=0, availableFlagB=1 (nB=7) → nC = 7.
        assert_eq!(nc, 7);
    }

    #[test]
    fn nc_mb5_block0_averages_left_and_above() {
        // §6.4.11.4 — MB at (1, 1) = addr 5. Block 0 at (x=0, y=0):
        //   A at (-1, 0) → left MB (MB 4), wrapped (xW=15, yW=0)
        //     = block 5 (per eq. 6-38).
        //   B at ( 0, -1) → above MB (MB 1), wrapped (xW=0, yW=15)
        //     = block 10 (per eq. 6-38).
        let mut grid = CavlcNcGrid::new(4, 4);
        let mut t4 = [0u8; 16];
        t4[5] = 4;
        fill_mb(&mut grid, 4, t4);
        let mut t1 = [0u8; 16];
        t1[10] = 6;
        fill_mb(&mut grid, 1, t1);
        let nc = derive_nc_luma(&grid, /*mb_addr=*/ 5, /*blk=*/ 0, LumaNcKind::Ac, true, false);
        // (nA + nB + 1) >> 1 = (4 + 6 + 1) >> 1 = 5.
        assert_eq!(nc, 5);
    }

    #[test]
    fn nc_skip_neighbour_contributes_zero() {
        // §9.2.1.1 step 6: P_Skip / B_Skip neighbour → nN = 0 even if
        // the block slot numerically holds a non-zero total.
        let mut grid = CavlcNcGrid::new(4, 4);
        let mut totals = [0u8; 16];
        totals[5] = 9;
        fill_mb(&mut grid, 0, totals);
        // Mark MB 0 as a skip — its total_coeff values must be ignored.
        grid.mbs[0].is_skip = true;
        let nc = derive_nc_luma(&grid, 1, 0, LumaNcKind::Ac, true, false);
        // A = MB 0 blk 5 (nN forced to 0), B = unavailable → nC = 0.
        assert_eq!(nc, 0);
    }

    #[test]
    fn nc_i_pcm_neighbour_contributes_sixteen() {
        // §9.2.1.1 step 6: I_PCM neighbour → nN = 16.
        let mut grid = CavlcNcGrid::new(4, 4);
        fill_mb(&mut grid, 0, [0; 16]); // totals irrelevant for I_PCM.
        grid.mbs[0].is_i_pcm = true;
        let nc = derive_nc_luma(&grid, 1, 0, LumaNcKind::Ac, true, false);
        // A = MB 0 blk 5 (nN=16), B = unavailable → nC = nA = 16.
        assert_eq!(nc, 16);
    }

    #[test]
    fn nc_intra16x16_dc_ignores_block_idx() {
        // §9.2.1.1 step 1 for Intra16x16DCLevel: luma4x4BlkIdx = 0
        // regardless of the caller's supplied value. Verify by asking
        // for a "block 15" neighbour on MB 1 — the derivation should
        // still use block 0's neighbours (A = MB 0 block 5, B up).
        let mut grid = CavlcNcGrid::new(4, 4);
        let mut totals = [0u8; 16];
        totals[5] = 8;
        fill_mb(&mut grid, 0, totals);
        let nc = derive_nc_luma(&grid, 1, 15, LumaNcKind::Intra16x16Dc, true, false);
        // Equivalent to asking for block 0: A=MB 0 blk 5 = 8, B unavail.
        assert_eq!(nc, 8);
    }

    #[test]
    fn nc_chroma_ac_420_top_left_block_uses_left_mb() {
        // Chroma 4x4 block 0 in 4:2:0: (x=0, y=0) inside the chroma
        // plane. A = (-1, 0) → left MB's chroma block at (xW=7, yW=0)
        // → chroma4x4BlkIdx = 2*(0/4) + 7/4 = 0 + 1 = 1.
        // B = (0, -1) → above MB's chroma block at (xW=0, yW=7)
        // → chroma4x4BlkIdx = 2*(7/4) + 0/4 = 2.
        let mut grid = CavlcNcGrid::new(4, 4);
        // mb 0: mark available with chroma totals filled.
        grid.mbs[0].is_available = true;
        grid.mbs[0].cb_total_coeff[2] = 4; // block 2 (above neighbour from MB 4's POV)
        let nc = derive_nc_chroma_ac(&grid, /*mb_addr=*/ 4, /*blk=*/ 0, /*is_cr=*/ false, 1, true, false);
        // MB 4 = (0, 1). A unavailable (left edge), B = MB 0 block 2 (cb=4).
        assert_eq!(nc, 4);
    }

    #[test]
    fn nc_grid_neighbour_mb_addrs_corners() {
        // Raster-scan left+above neighbour lookup on a 4x4 MB grid.
        let grid = CavlcNcGrid::new(4, 4);
        assert_eq!(grid.neighbour_mb_addrs(0), (None, None));
        assert_eq!(grid.neighbour_mb_addrs(3), (Some(2), None));
        assert_eq!(grid.neighbour_mb_addrs(4), (None, Some(0)));
        assert_eq!(grid.neighbour_mb_addrs(5), (Some(4), Some(1)));
        assert_eq!(grid.neighbour_mb_addrs(15), (Some(14), Some(11)));
    }

    #[test]
    fn combine_nc_formula_matches_spec() {
        // §9.2.1.1 step 7 arithmetic cross-check.
        assert_eq!(combine_nc(false, 0, false, 0), 0);
        assert_eq!(combine_nc(true, 5, false, 0), 5);
        assert_eq!(combine_nc(false, 0, true, 7), 7);
        assert_eq!(combine_nc(true, 3, true, 4), (3 + 4 + 1) >> 1); // 4
        assert_eq!(combine_nc(true, 0, true, 0), 0);
        assert_eq!(combine_nc(true, 16, true, 16), (16 + 16 + 1) >> 1); // 16
    }

    // -----------------------------------------------------------------
    // §9.3.3.1.1.6 — ref_idx_lX ctxIdxInc tests.
    //
    // Fresh grid, MB 1 (left = MB 0, above = unavailable) unless noted.
    // The `condTerm = (ref_idx > 0)` derivation only consults the
    // neighbour's ref_idx_lX — A lives at left MB's 8x8 #1 for the
    // current MB's 8x8 #0. Setting left MB's 8x8 #1 ref_idx should
    // therefore drive condTermA.
    // -----------------------------------------------------------------

    fn mk_cabac_grid() -> CabacNeighbourGrid {
        CabacNeighbourGrid::new(4, 4)
    }

    /// Populate MB `addr` as a decoded inter MB with ref_idx_l0 set
    /// uniformly across 8x8 partitions to `ref_idx`.
    fn fill_inter_mb(grid: &mut CabacNeighbourGrid, addr: u32, ref_idx: i8) {
        let slot = &mut grid.mbs[addr as usize];
        slot.available = true;
        slot.is_intra = false;
        slot.is_skip = false;
        slot.is_i_pcm = false;
        slot.ref_idx_l0 = [ref_idx; 4];
        slot.ref_idx_l1 = [-1; 4];
    }

    #[test]
    fn ref_idx_ctx_inc_both_neighbours_zero() {
        // MB 5 (row 1, col 1 in a 4x4 MB grid). A = MB 4, B = MB 1.
        let mut grid = mk_cabac_grid();
        fill_inter_mb(&mut grid, 4, 0); // A: ref_idx = 0 → condTermA = 0
        fill_inter_mb(&mut grid, 1, 0); // B: ref_idx = 0 → condTermB = 0
        let curr = CabacMbNeighbourInfo::default();
        let (ca, cb) = cabac_ref_idx_cond_terms(&grid, 5, &curr, 0, 0);
        assert!(!ca);
        assert!(!cb);
        // ctxIdxInc per eq. 9-14 = 0 + 2*0 = 0.
        let inc = u32::from(ca) + 2 * u32::from(cb);
        assert_eq!(inc, 0);
    }

    #[test]
    fn ref_idx_ctx_inc_a_nonzero() {
        let mut grid = mk_cabac_grid();
        fill_inter_mb(&mut grid, 4, 2); // A → condTermA = 1
        fill_inter_mb(&mut grid, 1, 0); // B → condTermB = 0
        let curr = CabacMbNeighbourInfo::default();
        let (ca, cb) = cabac_ref_idx_cond_terms(&grid, 5, &curr, 0, 0);
        assert!(ca);
        assert!(!cb);
        let inc = u32::from(ca) + 2 * u32::from(cb);
        assert_eq!(inc, 1);
    }

    #[test]
    fn ref_idx_ctx_inc_b_nonzero() {
        let mut grid = mk_cabac_grid();
        fill_inter_mb(&mut grid, 4, 0); // A → condTermA = 0
        fill_inter_mb(&mut grid, 1, 1); // B → condTermB = 1
        let curr = CabacMbNeighbourInfo::default();
        let (ca, cb) = cabac_ref_idx_cond_terms(&grid, 5, &curr, 0, 0);
        assert!(!ca);
        assert!(cb);
        let inc = u32::from(ca) + 2 * u32::from(cb);
        assert_eq!(inc, 2);
    }

    #[test]
    fn ref_idx_ctx_inc_both_nonzero() {
        let mut grid = mk_cabac_grid();
        fill_inter_mb(&mut grid, 4, 2); // A
        fill_inter_mb(&mut grid, 1, 3); // B
        let curr = CabacMbNeighbourInfo::default();
        let (ca, cb) = cabac_ref_idx_cond_terms(&grid, 5, &curr, 0, 0);
        assert!(ca);
        assert!(cb);
        let inc = u32::from(ca) + 2 * u32::from(cb);
        assert_eq!(inc, 3);
    }

    #[test]
    fn ref_idx_intra_neighbour_contributes_zero() {
        // §9.3.3.1.1.6 — intra / skip neighbours contribute 0.
        let mut grid = mk_cabac_grid();
        fill_inter_mb(&mut grid, 4, 2);
        grid.mbs[4].is_intra = true; // re-mark as intra
        fill_inter_mb(&mut grid, 1, 3);
        grid.mbs[1].is_skip = true; // re-mark as skip
        let curr = CabacMbNeighbourInfo::default();
        let (ca, cb) = cabac_ref_idx_cond_terms(&grid, 5, &curr, 0, 0);
        assert!(!ca);
        assert!(!cb);
    }

    #[test]
    fn ref_idx_unavailable_neighbours_contribute_zero() {
        // MB 0 — top-left; both A and B unavailable (outside picture).
        let grid = mk_cabac_grid();
        let curr = CabacMbNeighbourInfo::default();
        let (ca, cb) = cabac_ref_idx_cond_terms(&grid, 0, &curr, 0, 0);
        assert!(!ca);
        assert!(!cb);
    }

    // -----------------------------------------------------------------
    // §9.3.3.1.1.7 — mvd_lX ctxIdxInc tests.
    //
    // The derivation `ctxIdxInc = 0/1/2 per (<3, 3..=32, >32)` is mapped
    // by `decode_mvd_lx` from a `neighbour_abs_mvd_sum` value. We test
    // the sum-producing helper directly.
    // -----------------------------------------------------------------

    /// Populate MB `addr` with uniform mvd_l0 (x, y).
    fn fill_inter_mvd(
        grid: &mut CabacNeighbourGrid,
        addr: u32,
        mvd_x: i16,
        mvd_y: i16,
    ) {
        let slot = &mut grid.mbs[addr as usize];
        slot.available = true;
        slot.is_intra = false;
        slot.is_skip = false;
        slot.is_i_pcm = false;
        slot.mvd_l0_x = [mvd_x; 16];
        slot.mvd_l0_y = [mvd_y; 16];
    }

    #[test]
    fn mvd_ctx_inc_small_sum() {
        // sum < 3 → ctxIdxInc = 0. A has mvd=1, B has mvd=1 → sum = 2.
        let mut grid = mk_cabac_grid();
        fill_inter_mvd(&mut grid, 4, 1, 0);
        fill_inter_mvd(&mut grid, 1, 1, 0);
        let curr = CabacMbNeighbourInfo::default();
        let sum = cabac_mvd_abs_sum(&grid, 5, &curr, 0, MvdComponent::X, 0);
        assert_eq!(sum, 2);
        // decode_mvd_lx maps sum>2 → 1, sum>32 → 2, else 0.
        // sum == 2 → 0.
        assert!(sum < 3);
    }

    #[test]
    fn mvd_ctx_inc_medium_sum() {
        // 3 <= sum <= 32 → ctxIdxInc = 1. A has 2, B has 2 → sum = 4.
        let mut grid = mk_cabac_grid();
        fill_inter_mvd(&mut grid, 4, 2, 0);
        fill_inter_mvd(&mut grid, 1, 2, 0);
        let curr = CabacMbNeighbourInfo::default();
        let sum = cabac_mvd_abs_sum(&grid, 5, &curr, 0, MvdComponent::X, 0);
        assert_eq!(sum, 4);
        assert!(sum > 2 && sum <= 32);
    }

    #[test]
    fn mvd_ctx_inc_large_sum() {
        // sum > 32 → ctxIdxInc = 2. A has 20, B has 20 → sum = 40.
        let mut grid = mk_cabac_grid();
        fill_inter_mvd(&mut grid, 4, 20, 0);
        fill_inter_mvd(&mut grid, 1, 20, 0);
        let curr = CabacMbNeighbourInfo::default();
        let sum = cabac_mvd_abs_sum(&grid, 5, &curr, 0, MvdComponent::X, 0);
        assert_eq!(sum, 40);
        assert!(sum > 32);
    }

    #[test]
    fn mvd_skip_or_intra_neighbour_contributes_zero() {
        let mut grid = mk_cabac_grid();
        fill_inter_mvd(&mut grid, 4, 10, 0);
        grid.mbs[4].is_skip = true; // A is skip → 0
        fill_inter_mvd(&mut grid, 1, 10, 0);
        grid.mbs[1].is_intra = true; // B is intra → 0
        let curr = CabacMbNeighbourInfo::default();
        let sum = cabac_mvd_abs_sum(&grid, 5, &curr, 0, MvdComponent::X, 0);
        assert_eq!(sum, 0);
    }

    #[test]
    fn mvd_sign_contributes_via_abs() {
        // §9.3.3.1.1.7 uses |mvd|, so sign doesn't matter.
        let mut grid = mk_cabac_grid();
        fill_inter_mvd(&mut grid, 4, -5, 0); // |A| = 5
        fill_inter_mvd(&mut grid, 1, -10, 0); // |B| = 10
        let curr = CabacMbNeighbourInfo::default();
        let sum = cabac_mvd_abs_sum(&grid, 5, &curr, 0, MvdComponent::X, 0);
        assert_eq!(sum, 15);
    }

    // -----------------------------------------------------------------
    // §9.3.3.1.1.9 — coded_block_flag ctxIdxInc tests.
    // -----------------------------------------------------------------

    #[test]
    fn cbf_ctx_inc_both_unavailable_inter_current() {
        // Top-left MB, inter current: unavailable → both cond 0.
        let grid = mk_cabac_grid();
        let curr = CabacMbNeighbourInfo::default();
        let (ca, cb) = cabac_cbf_cond_terms(
            &grid, 0, &curr, /*current_is_intra=*/ false, BlockType::Luma4x4, 0, false,
        );
        assert!(!ca);
        assert!(!cb);
    }

    #[test]
    fn cbf_ctx_inc_both_unavailable_intra_current() {
        // Top-left MB, intra current: unavailable → both cond 1 per
        // Table 9-42 fallback ("available: no, intra: yes → 1").
        let grid = mk_cabac_grid();
        let curr = CabacMbNeighbourInfo {
            is_intra: true,
            ..Default::default()
        };
        let (ca, cb) = cabac_cbf_cond_terms(
            &grid, 0, &curr, /*current_is_intra=*/ true, BlockType::Luma4x4, 0, false,
        );
        assert!(ca);
        assert!(cb);
    }

    #[test]
    fn cbf_ctx_inc_both_coded() {
        // MB 5: A = MB 4, B = MB 1. Fill both with cbf_luma_4x4 = true
        // across all blocks AND cbp_luma = 15 so that the
        // §9.3.3.1.1.9 cat=1/2 "transBlockN available" gate triggers
        // (cbp bit for the 8x8 containing the 4x4 block is set).
        let mut grid = mk_cabac_grid();
        for addr in [4, 1] {
            let slot = &mut grid.mbs[addr as usize];
            slot.available = true;
            slot.is_intra = true;
            slot.cbf_luma_4x4 = [true; 16];
            slot.coded_block_pattern_luma = 15;
        }
        let curr = CabacMbNeighbourInfo::default();
        // For current MB 5, block 0's neighbour A = MB 4 block at
        // wrapped (xN=-1+16=15, yN=0) → block idx = 5.
        // Neighbour B = MB 1 block at (xN=0, yN=-1+16=15) → idx = 10.
        let (ca, cb) = cabac_cbf_cond_terms(
            &grid, 5, &curr, true, BlockType::Luma4x4, 0, false,
        );
        assert!(ca);
        assert!(cb);
        // ctxIdxInc = 1 + 2*1 = 3.
        let inc = u32::from(ca) + 2 * u32::from(cb);
        assert_eq!(inc, 3);
    }

    /// §9.3.3.1.1.9 cat=3 — if neighbour's CodedBlockPatternChroma is 0
    /// (no chroma DC block), transBlockN not available → condTermFlagN
    /// = 0. Regression for the MB-level Cb/Cr DC path.
    #[test]
    fn cbf_ctx_inc_chroma_dc_neighbour_cbp_chroma_zero_yields_zero() {
        let mut grid = mk_cabac_grid();
        // MB 4 (left of MB 5): available, intra, chroma CBP = 0.
        let slot = &mut grid.mbs[4];
        slot.available = true;
        slot.is_intra = true;
        slot.coded_block_pattern_chroma = 0;
        slot.cbf_cb_dc = true; // numerically non-zero — irrelevant
        let curr = CabacMbNeighbourInfo::default();
        let (ca, _cb) = cabac_cbf_cond_terms(
            &grid, 5, &curr, /*current_is_intra=*/ true, BlockType::ChromaDc, 0, false,
        );
        // Spec §9.3.3.1.1.9 cat=3: mbAddrA available, transBlockA not
        // available (CodedBlockPatternChroma == 0), mb_type != I_PCM
        // → condTermA = 0.
        assert!(!ca, "chroma-CBP=0 neighbour must yield ChromaDC condTermA = 0");
    }

    #[test]
    fn cbf_i_pcm_neighbour_all_ones() {
        // I_PCM MB as neighbour contributes cbf = 1 per §9.3.3.1.1.9.
        let mut grid = mk_cabac_grid();
        grid.mbs[4].available = true;
        grid.mbs[4].is_i_pcm = true;
        grid.mbs[4].is_intra = true; // I_PCM is intra
        let curr = CabacMbNeighbourInfo { is_intra: true, ..Default::default() };
        let (ca, _cb) = cabac_cbf_cond_terms(
            &grid, 1, &curr, true, BlockType::Luma4x4, 0, false,
        );
        assert!(ca);
    }

    /// §9.3.3.1.1.9 cat=1/2 — when neighbour MB is available, NOT I_PCM,
    /// and its cbp_luma bit for the 8x8 containing luma4x4BlkIdxN is 0,
    /// transBlockN is "not available" and condTermFlagN = 0 regardless
    /// of whether the current MB is intra (spec bullet 2 in the
    /// "condTermFlagN = 0" branch).
    #[test]
    fn cbf_luma4x4_neighbour_cbp_bit_zero_should_yield_zero() {
        let mut grid = mk_cabac_grid();
        // MB 4 (left of MB 5): available, intra I_NxN, cbp_luma=0x3
        // (8x8 #0 and #1 coded). Blk 13 is in 8x8 #3 (bit 3 = 0) so its
        // transBlockN should be "not available" for cat 1/2.
        let slot = &mut grid.mbs[4];
        slot.available = true;
        slot.is_intra = true;
        slot.coded_block_pattern_luma = 0x3;
        slot.cbf_luma_4x4[13] = true; // deliberately non-zero — irrelevant
        let curr = CabacMbNeighbourInfo { is_intra: true, ..Default::default() };
        // Blk 8 of MB 5 at (0, 8) — left neighbour maps to MB 4 blk 13.
        let (ca, _cb) = cabac_cbf_cond_terms(
            &grid, 5, &curr, /*current_is_intra=*/ true,
            BlockType::Luma4x4, 8, false,
        );
        // Per spec the condTerm must be 0.
        assert!(!ca, "spec §9.3.3.1.1.9 says condTermA = 0 when neighbour cbp bit is 0");
    }

    /// Same spec rule for cat=4 (ChromaAc): when neighbour
    /// CodedBlockPatternChroma != 2, transBlockN is not available and
    /// condTerm = 0 per spec bullet 2.
    #[test]
    fn cbf_chroma_ac_neighbour_cbp_chroma_neq_2_should_yield_zero() {
        let mut grid = mk_cabac_grid();
        let slot = &mut grid.mbs[4];
        slot.available = true;
        slot.is_intra = true;
        slot.coded_block_pattern_chroma = 1; // != 2 → transBlockN not available
        slot.cbf_cb_ac[0] = true; // deliberately non-zero — irrelevant
        let curr = CabacMbNeighbourInfo { is_intra: true, ..Default::default() };
        let (ca, _cb) = cabac_cbf_cond_terms(
            &grid, 5, &curr, /*current_is_intra=*/ true,
            BlockType::ChromaAc, 0, false,
        );
        assert!(!ca, "spec §9.3.3.1.1.9 cat=4 says condTermA = 0 when neighbour cbp_chroma != 2");
    }

    #[test]
    fn cbf_skip_neighbour_all_zero() {
        let mut grid = mk_cabac_grid();
        grid.mbs[4].available = true;
        grid.mbs[4].is_skip = true;
        grid.mbs[4].cbf_luma_4x4 = [true; 16]; // numerically non-zero
        let curr = CabacMbNeighbourInfo::default();
        let (ca, _cb) = cabac_cbf_cond_terms(
            &grid, 1, &curr, false, BlockType::Luma4x4, 0, false,
        );
        assert!(!ca);
    }

    #[test]
    fn cbf_internal_block_sees_current_mb_progressive_state() {
        // Block 5's A = block 4 (internal), B = block 1 (internal).
        // Simulate "decoded so far": current MB's cbf_luma_4x4[4] = true,
        // cbf_luma_4x4[1] = false.
        let grid = mk_cabac_grid();
        let mut curr = CabacMbNeighbourInfo::default();
        curr.cbf_luma_4x4[4] = true;
        curr.cbf_luma_4x4[1] = false;
        let (ca, cb) = cabac_cbf_cond_terms(
            &grid, 0, &curr, /*current_is_intra=*/ false,
            BlockType::Luma4x4, 5, false,
        );
        assert!(ca, "A (internal block 4) has cbf=true");
        assert!(!cb, "B (internal block 1) has cbf=false");
    }

    // -----------------------------------------------------------------
    // §9.3.3.1.1.* grid write-after-decode integration test.
    // -----------------------------------------------------------------

    #[test]
    fn cabac_neighbour_grid_default_all_unavailable() {
        let grid = mk_cabac_grid();
        assert_eq!(grid.width_in_mbs, 4);
        assert_eq!(grid.height_in_mbs, 4);
        assert_eq!(grid.mbs.len(), 16);
        for info in &grid.mbs {
            assert!(!info.available);
            assert_eq!(info.ref_idx_l0, [-1; 4]);
        }
    }

    #[test]
    fn cabac_neighbour_grid_addr_to_xy() {
        // Raster-scan neighbour lookup on a 4x4 MB grid.
        let grid = mk_cabac_grid();
        assert_eq!(grid.neighbour_mb_addrs(0), (None, None));
        assert_eq!(grid.neighbour_mb_addrs(3), (Some(2), None));
        assert_eq!(grid.neighbour_mb_addrs(4), (None, Some(0)));
        assert_eq!(grid.neighbour_mb_addrs(5), (Some(4), Some(1)));
        assert_eq!(grid.neighbour_mb_addrs(15), (Some(14), Some(11)));
    }

    // -----------------------------------------------------------------
    // §9.3.3.1.1.9 cat=3 / cat=4 — ChromaDC / ChromaAC iCbCr.
    //
    // The spec says `transBlockN` for cat=3 (ChromaDc) is "the chroma DC
    // block of chroma component iCbCr of macroblock mbAddrN" — Cb-DC
    // when iCbCr=0, Cr-DC when iCbCr=1. Same distinction for cat=4.
    // -----------------------------------------------------------------

    #[test]
    fn cbf_chroma_dc_separates_cb_and_cr() {
        // Left MB has Cb-DC coded but Cr-DC uncoded (encoder wrote
        // cbf_cb_dc=1, cbf_cr_dc=0). For the current MB, condTermA for
        // Cb DC should be 1; for Cr DC it should be 0.
        let mut grid = mk_cabac_grid();
        grid.mbs[0].available = true;
        grid.mbs[0].is_intra = true;
        // §9.3.3.1.1.9 — transBlockN for ChromaDc requires the neighbour
        // to have CodedBlockPatternChroma != 0; set it here so the
        // cbf_cb_dc/cbf_cr_dc values are consulted.
        grid.mbs[0].coded_block_pattern_chroma = 1;
        grid.mbs[0].cbf_cb_dc = true;
        grid.mbs[0].cbf_cr_dc = false;
        let curr = CabacMbNeighbourInfo { is_intra: true, ..Default::default() };
        // MB 1 = right neighbour of MB 0. A = MB 0, B = unavailable (row 0).
        let (ca_cb, _cb_cb) = cabac_cbf_cond_terms(
            &grid, 1, &curr, true, BlockType::ChromaDc, 0, /*is_cr=*/false,
        );
        let (ca_cr, _cb_cr) = cabac_cbf_cond_terms(
            &grid, 1, &curr, true, BlockType::ChromaDc, 0, /*is_cr=*/true,
        );
        assert!(ca_cb, "Cb DC should see neighbour cbf_cb_dc=true");
        assert!(!ca_cr, "Cr DC should see neighbour cbf_cr_dc=false");
    }

    #[test]
    fn cbf_chroma_ac_separates_cb_and_cr() {
        // Left MB (MB 0) has Cb-AC coded on block 0 but Cr-AC uncoded.
        let mut grid = mk_cabac_grid();
        grid.mbs[0].available = true;
        grid.mbs[0].is_intra = true;
        // §9.3.3.1.1.9 — transBlockN for ChromaAc requires the neighbour
        // to have CodedBlockPatternChroma == 2 (DC+AC); set it so the
        // cbf_c?_ac values are consulted.
        grid.mbs[0].coded_block_pattern_chroma = 2;
        grid.mbs[0].cbf_cb_ac[0] = true;
        grid.mbs[0].cbf_cr_ac[0] = false;
        // The neighbour wrap for MB 1's chroma AC block 0 at (0, 0)
        // goes to left-MB chroma block index 1 (wrap xn=-1+8=7 → idx 1).
        grid.mbs[0].cbf_cb_ac[1] = true;
        grid.mbs[0].cbf_cr_ac[1] = false;
        let curr = CabacMbNeighbourInfo { is_intra: true, ..Default::default() };
        let (ca_cb, _cb_cb) = cabac_cbf_cond_terms(
            &grid, 1, &curr, true, BlockType::ChromaAc, 0, false,
        );
        let (ca_cr, _cb_cr) = cabac_cbf_cond_terms(
            &grid, 1, &curr, true, BlockType::ChromaAc, 0, true,
        );
        assert!(ca_cb, "Cb AC should see neighbour cbf_cb_ac=true");
        assert!(!ca_cr, "Cr AC should see neighbour cbf_cr_ac=false");
    }

    #[test]
    fn cbf_chroma_ac_internal_separates_cb_and_cr() {
        // Current MB has decoded Cb AC block 0 as coded, Cr AC block 0
        // as uncoded. When decoding Cb AC block 1 (iCbCr=0) we read
        // `curr_cbf.cbf_cb_ac[0]` for the internal A-neighbour. When
        // decoding Cr AC block 1 (iCbCr=1) we should read
        // `curr_cbf.cbf_cr_ac[0]` instead.
        let grid = mk_cabac_grid();
        let mut curr = CabacMbNeighbourInfo::default();
        curr.is_intra = true;
        curr.cbf_cb_ac[0] = true;
        curr.cbf_cr_ac[0] = false;
        // ChromaAc blk_idx=1 is at (4, 0) → A=(3, 0) internal block 0.
        let (ca_cb, _cb_cb) = cabac_cbf_cond_terms(
            &grid, 0, &curr, true, BlockType::ChromaAc, 1, false,
        );
        let (ca_cr, _cb_cr) = cabac_cbf_cond_terms(
            &grid, 0, &curr, true, BlockType::ChromaAc, 1, true,
        );
        assert!(ca_cb, "Cb AC internal should see cbf_cb_ac[0]=true");
        assert!(!ca_cr, "Cr AC internal should see cbf_cr_ac[0]=false");
    }

    // -----------------------------------------------------------------
    // Regression: §7.3.5.1 / §7.3.5.2 ref_idx / mvd gating.
    //
    // Before this round `parse_mb_pred` unconditionally read a 1-bit
    // ref_idx_l0 per partition on every inter MB (even when
    // num_ref_idx_l0_active_minus1 == 0, in which case the syntax
    // element must not be present at all per §7.3.5.1). That produced
    // 1-bit desyncs that showed up downstream as bogus mb_type / CBP
    // values on subsequent MBs.
    // -----------------------------------------------------------------

    #[test]
    fn cavlc_p_l0_16x16_with_single_ref_reads_no_ref_idx() {
        // PL0_16x16 with num_ref_idx_l0_active_minus1 == 0: no
        // ref_idx_l0 bits should be consumed; only mvd_l0 x/y.
        //
        // Build:
        //   mb_type = 0 → ue(v) "1"
        //   (no ref_idx)
        //   mvd_l0[0][0] = 0 → se(v) codeNum=0 "1"
        //   mvd_l0[0][1] = 0 → se(v) codeNum=0 "1"
        //   coded_block_pattern (inter codeNum=0 → CBP=0) → "1"
        //   (cbp=0 → no mb_qp_delta, no residual)
        let mut w = BitWriter::new();
        w.ue(0); // mb_type = P_L0_16x16
        // mvd x, y
        w.ue(0); // mvd x = 0
        w.ue(0); // mvd y = 0
        w.ue(0); // CBP inter codeNum=0 → 0
        w.trailing();
        let bytes = w.into_bytes();

        let sps = dummy_sps();
        let pps = dummy_pps();
        let hdr = dummy_slice_header(SliceType::P);
        let mut r = BitReader::new(&bytes);
        let mut entropy = cavlc_entropy();
        entropy.slice_kind = SliceKind::P;
        entropy.num_ref_idx_l0_active_minus1 = 0;
        let mb = parse_macroblock(&mut r, &mut entropy, &hdr, &sps, &pps, 0).unwrap();
        assert_eq!(mb.mb_type, MbType::PL016x16);
        assert_eq!(mb.coded_block_pattern, 0);
        let pred = mb.mb_pred.expect("mb_pred");
        // ref_idx_l0 should have been filled with the default (0) — no
        // bits consumed.
        assert_eq!(pred.ref_idx_l0.len(), 1);
        assert_eq!(pred.ref_idx_l0[0], 0);
        // MVD was read.
        assert_eq!(pred.mvd_l0.len(), 1);
        assert_eq!(pred.mvd_l0[0], [0, 0]);
    }

    #[test]
    fn cavlc_p_l0_16x16_with_two_refs_reads_ref_idx_te() {
        // PL0_16x16 with num_ref_idx_l0_active_minus1 == 1: one bit
        // (inverted) for ref_idx_l0 per §9.1.2 eq. 9-3.
        //
        // Build:
        //   mb_type = 0 → ue(v) "1"
        //   ref_idx_l0[0] = 1 → te(1) = read 1 bit; codeNum = !b.
        //     write b=0 → value=1.
        //   mvd_l0[0][0] = 0 → "1"
        //   mvd_l0[0][1] = 0 → "1"
        //   CBP inter codeNum=0 → "1"
        let mut w = BitWriter::new();
        w.ue(0); // mb_type = P_L0_16x16
        w.u(1, 0); // ref_idx_l0 te(1), b=0 → value 1.
        w.ue(0); // mvd x
        w.ue(0); // mvd y
        w.ue(0); // CBP inter codeNum=0 → 0
        w.trailing();
        let bytes = w.into_bytes();

        let sps = dummy_sps();
        let pps = dummy_pps();
        let hdr = dummy_slice_header(SliceType::P);
        let mut r = BitReader::new(&bytes);
        let mut entropy = cavlc_entropy();
        entropy.slice_kind = SliceKind::P;
        entropy.num_ref_idx_l0_active_minus1 = 1;
        let mb = parse_macroblock(&mut r, &mut entropy, &hdr, &sps, &pps, 0).unwrap();
        assert_eq!(mb.mb_type, MbType::PL016x16);
        let pred = mb.mb_pred.expect("mb_pred");
        assert_eq!(pred.ref_idx_l0[0], 1);
    }

    #[test]
    fn mb_part_pred_mode_p_slice_cases() {
        // Table 7-13.
        assert_eq!(
            MbType::PL016x16.mb_part_pred_mode(0),
            Some(MbPartPredMode::PredL0)
        );
        assert_eq!(MbType::PL016x16.mb_part_pred_mode(1), None);
        assert_eq!(
            MbType::PL0L016x8.mb_part_pred_mode(0),
            Some(MbPartPredMode::PredL0)
        );
        assert_eq!(
            MbType::PL0L016x8.mb_part_pred_mode(1),
            Some(MbPartPredMode::PredL0)
        );
        // P8x8 / P_8x8ref0 — sub_mb path, so mb-level mode is None.
        assert_eq!(MbType::P8x8.mb_part_pred_mode(0), None);
        assert_eq!(MbType::P8x8Ref0.mb_part_pred_mode(0), None);
    }

    #[test]
    fn mb_part_pred_mode_b_slice_cases() {
        // Table 7-14 — mixed-list partitions hit both Pred_L0 and Pred_L1.
        assert_eq!(
            MbType::BL016x16.mb_part_pred_mode(0),
            Some(MbPartPredMode::PredL0)
        );
        assert_eq!(
            MbType::BL116x16.mb_part_pred_mode(0),
            Some(MbPartPredMode::PredL1)
        );
        assert_eq!(
            MbType::BBi16x16.mb_part_pred_mode(0),
            Some(MbPartPredMode::BiPred)
        );
        assert_eq!(
            MbType::BDirect16x16.mb_part_pred_mode(0),
            Some(MbPartPredMode::Direct)
        );
        // B_L0_L1_16x8 — part0=L0, part1=L1.
        assert_eq!(
            MbType::BL0L116x8.mb_part_pred_mode(0),
            Some(MbPartPredMode::PredL0)
        );
        assert_eq!(
            MbType::BL0L116x8.mb_part_pred_mode(1),
            Some(MbPartPredMode::PredL1)
        );
        // B_Bi_L0_8x16 — part0=BiPred, part1=L0.
        assert_eq!(
            MbType::BBiL08x16.mb_part_pred_mode(0),
            Some(MbPartPredMode::BiPred)
        );
        assert_eq!(
            MbType::BBiL08x16.mb_part_pred_mode(1),
            Some(MbPartPredMode::PredL0)
        );
    }

    #[test]
    fn sub_mb_pred_mode_p_and_b_tables() {
        // Table 7-17 — all P sub_mb_types are Pred_L0.
        assert_eq!(
            SubMbType::PL08x8.sub_mb_pred_mode(),
            Some(MbPartPredMode::PredL0)
        );
        assert_eq!(
            SubMbType::PL04x4.sub_mb_pred_mode(),
            Some(MbPartPredMode::PredL0)
        );
        // Table 7-18.
        assert_eq!(
            SubMbType::BDirect8x8.sub_mb_pred_mode(),
            Some(MbPartPredMode::Direct)
        );
        assert_eq!(
            SubMbType::BL08x8.sub_mb_pred_mode(),
            Some(MbPartPredMode::PredL0)
        );
        assert_eq!(
            SubMbType::BL18x4.sub_mb_pred_mode(),
            Some(MbPartPredMode::PredL1)
        );
        assert_eq!(
            SubMbType::BBi4x4.sub_mb_pred_mode(),
            Some(MbPartPredMode::BiPred)
        );
    }

    // -----------------------------------------------------------------
    // Regression: §9.1 te(v) with x_max == 0 is a no-op.
    //
    // The earlier implementation fell through to ue(v) when x_max ==
    // 0, which could eat extra bits from the bitstream (any 1-bit
    // prefix encoded codeNum=0, so "lucky" cases were hiding the bug).
    // -----------------------------------------------------------------
    #[test]
    fn te_x_max_zero_consumes_no_bits() {
        let bytes = [0xFFu8];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.te(0).unwrap(), 0);
        // Reader should still be at bit 0.
        let (byte, bit) = r.position();
        assert_eq!((byte, bit), (0, 0));
    }

    // -----------------------------------------------------------------
    // Re-audit of Tables 9-4(a)/(b) vs the ITU-T 08/2024 spec.
    //
    // Below we assert the full 48-row Table 9-4(a) (both intra and
    // inter columns) and the full 16-row Table 9-4(b), so any future
    // typo that corrupts a row is caught by the unit tests without
    // having to re-run integration fixtures.
    // -----------------------------------------------------------------
    #[test]
    fn table_9_4a_intra_420_422_full_row_audit() {
        let expected: [u8; 48] = [
            47, 31, 15, 0, 23, 27, 29, 30, 7, 11, 13, 14, 39, 43, 45, 46, 16, 3, 5, 10, 12, 19,
            21, 26, 28, 35, 37, 42, 44, 1, 2, 4, 8, 17, 18, 20, 24, 6, 9, 22, 25, 32, 33, 34, 36,
            40, 38, 41,
        ];
        for i in 0..48 {
            assert_eq!(ME_INTRA_420_422[i], expected[i], "row {i}");
        }
    }

    #[test]
    fn table_9_4a_inter_420_422_full_row_audit() {
        let expected: [u8; 48] = [
            0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37, 42, 44,
            33, 34, 36, 40, 39, 43, 45, 46, 17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22,
            25, 38, 41,
        ];
        for i in 0..48 {
            assert_eq!(ME_INTER_420_422[i], expected[i], "row {i}");
        }
    }

    #[test]
    fn table_9_4b_intra_0_3_full_row_audit() {
        let expected: [u8; 16] = [15, 0, 7, 11, 13, 14, 3, 5, 10, 12, 1, 2, 4, 8, 6, 9];
        for i in 0..16 {
            assert_eq!(ME_INTRA_0_3[i], expected[i], "row {i}");
        }
    }

    #[test]
    fn table_9_4b_inter_0_3_full_row_audit() {
        let expected: [u8; 16] = [0, 1, 2, 4, 8, 3, 5, 10, 12, 15, 7, 11, 13, 14, 6, 9];
        for i in 0..16 {
            assert_eq!(ME_INTER_0_3[i], expected[i], "row {i}");
        }
    }

    // mb_type boundary tests — catch off-by-one in the I/P/B remap.
    #[test]
    fn p_slice_mb_type_boundary_cases() {
        // Table 7-13 rows 0..=4 are P-specific.
        assert_eq!(MbType::from_p_slice(0).unwrap(), MbType::PL016x16);
        assert_eq!(MbType::from_p_slice(4).unwrap(), MbType::P8x8Ref0);
        // Row 5 is remapped to I_NxN (I row 0).
        assert_eq!(MbType::from_p_slice(5).unwrap(), MbType::INxN);
        // Row 30 is the max (remapped to I row 25 = I_PCM).
        assert_eq!(MbType::from_p_slice(30).unwrap(), MbType::IPcm);
        // Row 31 is out of range (31-5=26, and Table 7-11 max is 25).
        assert!(matches!(
            MbType::from_p_slice(31).unwrap_err(),
            MacroblockLayerError::MbTypeOutOfRange(31)
        ));
    }

    #[test]
    fn b_slice_mb_type_boundary_cases() {
        // Table 7-14 rows 0..=22 are B-specific.
        assert_eq!(MbType::from_b_slice(0).unwrap(), MbType::BDirect16x16);
        assert_eq!(MbType::from_b_slice(22).unwrap(), MbType::B8x8);
        // Row 23 is remapped to I_NxN.
        assert_eq!(MbType::from_b_slice(23).unwrap(), MbType::INxN);
        // Row 48 is the max (remapped to I row 25 = I_PCM).
        assert_eq!(MbType::from_b_slice(48).unwrap(), MbType::IPcm);
        // Row 49 is out of range.
        assert!(matches!(
            MbType::from_b_slice(49).unwrap_err(),
            MacroblockLayerError::MbTypeOutOfRange(49)
        ));
    }

    // -----------------------------------------------------------------
    // CABAC inter parsing — ref_idx / mvd via cabac_ctx primitives.
    //
    // Before these tests were added, `parse_mb_pred` returned a zero
    // stub for CABAC ref_idx_lX / mvd_lX; the resulting bitstream
    // desync showed up as nonsense mb_types / CBPs on the next MB.
    // Each test drives `parse_mb_pred` directly and cross-checks the
    // result against a side-by-side invocation of `decode_ref_idx_lx`
    // / `decode_mvd_lx` on an identical stream + context snapshot.
    // §7.3.5.1 gating; §9.3.3.1.1.6 / §9.3.3.1.1.7 bin layouts.
    // -----------------------------------------------------------------

    /// Run `parse_mb_pred` over `data` with CABAC enabled. Returns the
    /// decoded `MbPred` plus the number of bins the CABAC engine
    /// consumed (DecodeDecision + DecodeBypass + DecodeTerminate).
    fn run_cabac_mb_pred(
        data: &[u8],
        slice_kind: SliceKind,
        mb_type: &MbType,
        num_ref_idx_l0_active_minus1: u32,
        num_ref_idx_l1_active_minus1: u32,
    ) -> (MbPred, u64) {
        let mut dec = CabacDecoder::new(BitReader::new(data)).unwrap();
        let cabac_init_idc = match slice_kind {
            SliceKind::I | SliceKind::SI => None,
            _ => Some(0u32),
        };
        let mut ctxs = CabacContexts::init(slice_kind, cabac_init_idc, 26).unwrap();
        let bins_before = dec.bin_count();
        // `r` is only consulted in CAVLC mode / I_PCM; inter CABAC MBs
        // don't touch it, so a dummy reader suffices.
        let dummy = [0u8; 1];
        let mut r = BitReader::new(&dummy);
        let mut entropy = EntropyState {
            cabac: Some((&mut dec, &mut ctxs)),
            slice_kind,
            neighbours: NeighbourCtx::default(),
            prev_mb_qp_delta_nonzero: false,
            chroma_array_type: 1,
            transform_8x8_mode_flag: false,
            cavlc_nc: None,
            current_mb_addr: 0,
            constrained_intra_pred_flag: false,
            num_ref_idx_l0_active_minus1,
            num_ref_idx_l1_active_minus1,
            mbaff_frame_flag: false,
            cabac_nb: None,
            pic_width_in_mbs: 0,
        };
        let pred = parse_mb_pred(&mut r, &mut entropy, mb_type, false).unwrap();
        let bins_used = dec.bin_count() - bins_before;
        (pred, bins_used)
    }

    /// Drive the oracle path: on a fresh decoder + contexts over the
    /// same `data`, replay exactly the `decode_ref_idx_lx` /
    /// `decode_mvd_lx` calls that `parse_mb_pred` should have issued
    /// for the given mb_type / num_ref_idx_lX_active_minus1 settings.
    /// Returns the collected values and the total bin count consumed.
    fn oracle_mb_pred_calls(
        data: &[u8],
        slice_kind: SliceKind,
        mb_type: &MbType,
        num_ref_idx_l0_active_minus1: u32,
        num_ref_idx_l1_active_minus1: u32,
    ) -> (MbPred, u64) {
        let mut dec = CabacDecoder::new(BitReader::new(data)).unwrap();
        let cabac_init_idc = match slice_kind {
            SliceKind::I | SliceKind::SI => None,
            _ => Some(0u32),
        };
        let mut ctxs = CabacContexts::init(slice_kind, cabac_init_idc, 26).unwrap();
        let bins_before = dec.bin_count();
        let num_parts = mb_type.num_mb_part() as usize;
        let ref_l0_present = num_ref_idx_l0_active_minus1 > 0;
        let ref_l1_present = num_ref_idx_l1_active_minus1 > 0;
        let mut pred = MbPred::default();
        // ref_idx_l0.
        for part in 0..num_parts {
            let mode = mb_type.mb_part_pred_mode(part);
            if ref_l0_present && mode != Some(MbPartPredMode::PredL1) {
                let v = decode_ref_idx_lx(&mut dec, &mut ctxs, false, false).unwrap();
                pred.ref_idx_l0.push(v);
            } else {
                pred.ref_idx_l0.push(0);
            }
        }
        // ref_idx_l1.
        for part in 0..num_parts {
            let mode = mb_type.mb_part_pred_mode(part);
            if ref_l1_present
                && mode != Some(MbPartPredMode::PredL0)
                && mode != Some(MbPartPredMode::Direct)
                && slice_kind == SliceKind::B
            {
                let v = decode_ref_idx_lx(&mut dec, &mut ctxs, false, false).unwrap();
                pred.ref_idx_l1.push(v);
            } else {
                pred.ref_idx_l1.push(0);
            }
        }
        // mvd_l0.
        for part in 0..num_parts {
            let mode = mb_type.mb_part_pred_mode(part);
            if mode != Some(MbPartPredMode::PredL1) && mode != Some(MbPartPredMode::Direct) {
                let mx = decode_mvd_lx(&mut dec, &mut ctxs, MvdComponent::X, 0).unwrap();
                let my = decode_mvd_lx(&mut dec, &mut ctxs, MvdComponent::Y, 0).unwrap();
                pred.mvd_l0.push([mx, my]);
            } else {
                pred.mvd_l0.push([0, 0]);
            }
        }
        // mvd_l1.
        for part in 0..num_parts {
            let mode = mb_type.mb_part_pred_mode(part);
            if mode != Some(MbPartPredMode::PredL0)
                && mode != Some(MbPartPredMode::Direct)
                && slice_kind == SliceKind::B
            {
                let mx = decode_mvd_lx(&mut dec, &mut ctxs, MvdComponent::X, 0).unwrap();
                let my = decode_mvd_lx(&mut dec, &mut ctxs, MvdComponent::Y, 0).unwrap();
                pred.mvd_l1.push([mx, my]);
            } else {
                pred.mvd_l1.push([0, 0]);
            }
        }
        let bins_used = dec.bin_count() - bins_before;
        (pred, bins_used)
    }

    /// Standalone oracle for a single `decode_mvd_lx` invocation against a
    /// fresh CABAC stream + context snapshot — used to compute the bin
    /// count / return value that `parse_mb_pred` MUST reproduce.
    fn oracle_single_mvd(
        data: &[u8],
        slice_kind: SliceKind,
        comp: MvdComponent,
    ) -> (i32, u64) {
        let mut dec = CabacDecoder::new(BitReader::new(data)).unwrap();
        let cabac_init_idc = match slice_kind {
            SliceKind::I | SliceKind::SI => None,
            _ => Some(0u32),
        };
        let mut ctxs = CabacContexts::init(slice_kind, cabac_init_idc, 26).unwrap();
        let before = dec.bin_count();
        let v = decode_mvd_lx(&mut dec, &mut ctxs, comp, 0).unwrap();
        (v, dec.bin_count() - before)
    }

    /// §7.3.5.1 gate regression — `num_ref_idx_l0_active_minus1 == 0`
    /// (+ `mb_field == field_pic`) suppresses ref_idx_l0; parse_mb_pred
    /// must then read only the two mvd_l0 components via §9.3.3.1.1.7.
    #[test]
    fn cabac_p_l0_16x16_single_ref_skips_ref_idx() {
        // Any payload works — CABAC just walks the same bit string; we
        // only care that (a) bins were consumed (not a stub), and (b)
        // the result matches the oracle invocation sequence on the same
        // stream with the same gating decisions.
        let data: [u8; 16] = [
            0xA5, 0x5A, 0x3C, 0xC3, 0x81, 0x7E, 0x12, 0xED,
            0x00, 0xFF, 0x80, 0x01, 0x22, 0x44, 0x88, 0x77,
        ];
        let mb_type = MbType::PL016x16;
        let (pred, bins) =
            run_cabac_mb_pred(&data, SliceKind::P, &mb_type, 0, 0);
        let (expected, oracle_bins) =
            oracle_mb_pred_calls(&data, SliceKind::P, &mb_type, 0, 0);
        assert_eq!(pred.ref_idx_l0, expected.ref_idx_l0);
        assert_eq!(pred.mvd_l0, expected.mvd_l0);
        assert_eq!(pred.mvd_l1, expected.mvd_l1);
        assert_eq!(bins, oracle_bins, "bin consumption must match oracle");
        // The ref_idx-gating branch should not have consumed bins; only
        // the two mvd_lx calls should have. Also guard against the old
        // stub behaviour (which returned 0 with 0 bins consumed).
        assert!(bins > 0, "CABAC mvd_lx must consume bins");
        // The default ref_idx_l0 value is 0 (gated out per §7.3.5.1).
        assert_eq!(pred.ref_idx_l0.len(), 1);
        assert_eq!(pred.ref_idx_l0[0], 0);
    }

    /// §7.3.5.1 — `num_ref_idx_l0_active_minus1 > 0` enables ref_idx_l0.
    /// Parse_mb_pred must consume strictly more bins than the single-
    /// ref case because `decode_ref_idx_lx` reads at least bin 0.
    #[test]
    fn cabac_p_l0_16x16_multi_ref_reads_ref_idx() {
        let data: [u8; 16] = [
            0xA5, 0x5A, 0x3C, 0xC3, 0x81, 0x7E, 0x12, 0xED,
            0x00, 0xFF, 0x80, 0x01, 0x22, 0x44, 0x88, 0x77,
        ];
        let mb_type = MbType::PL016x16;
        let (pred, bins) =
            run_cabac_mb_pred(&data, SliceKind::P, &mb_type, 2, 0);
        let (expected, oracle_bins) =
            oracle_mb_pred_calls(&data, SliceKind::P, &mb_type, 2, 0);
        assert_eq!(pred.ref_idx_l0, expected.ref_idx_l0);
        assert_eq!(pred.mvd_l0, expected.mvd_l0);
        assert_eq!(bins, oracle_bins);
        assert!(bins > 0, "CABAC ref_idx + mvd must consume bins");
        // Bin-count regression: this path MUST consume strictly more
        // bins than the single-ref path on the same bitstream — the
        // extra bins are decode_ref_idx_lx's unary prefix.
        let (_, bins_single_ref) =
            run_cabac_mb_pred(&data, SliceKind::P, &mb_type, 0, 0);
        assert!(
            bins > bins_single_ref,
            "multi-ref path must read ref_idx bins (got {bins} vs single-ref {bins_single_ref})",
        );
    }

    /// §7.3.5.1 — B_L0_L1_16x8 has 2 partitions with opposite list
    /// modes. Per the gate:
    ///   part 0 (Pred_L0) → ref_idx_l0 + mvd_l0 only.
    ///   part 1 (Pred_L1) → ref_idx_l1 + mvd_l1 only.
    /// This exercises the per-partition list-mode gating under CABAC.
    #[test]
    fn cabac_b_l0_l1_16x8_per_list_gating() {
        let data: [u8; 16] = [
            0xA5, 0x5A, 0x3C, 0xC3, 0x81, 0x7E, 0x12, 0xED,
            0x00, 0xFF, 0x80, 0x01, 0x22, 0x44, 0x88, 0x77,
        ];
        let mb_type = MbType::BL0L116x8;
        // num_ref_idx_l{0,1}_active_minus1 > 0 → ref_idx is present for
        // each list on its matching partition.
        let (pred, bins) =
            run_cabac_mb_pred(&data, SliceKind::B, &mb_type, 2, 2);
        let (expected, oracle_bins) =
            oracle_mb_pred_calls(&data, SliceKind::B, &mb_type, 2, 2);
        assert_eq!(pred.ref_idx_l0, expected.ref_idx_l0);
        assert_eq!(pred.ref_idx_l1, expected.ref_idx_l1);
        assert_eq!(pred.mvd_l0, expected.mvd_l0);
        assert_eq!(pred.mvd_l1, expected.mvd_l1);
        assert_eq!(bins, oracle_bins);
        // Two partitions with 4 syntax elements each
        // (ref_idx_lX + mvd_x + mvd_y). Under any non-degenerate input
        // the engine must consume at least 4 bins (the bin-0 of each
        // decode path).
        assert!(
            bins >= 4,
            "B_L0_L1_16x8 CABAC must consume ≥ 4 bins across partitions (got {bins})",
        );
    }

    /// Cross-check the mvd_lx wiring by constructing a stream whose
    /// first-call output is independently known (via `decode_mvd_lx`
    /// standalone), then confirming `parse_mb_pred` returns the same
    /// value for mvd_l0[0][0]. This covers mvd=0 (bin 0 LPS, single
    /// bin) as the most common case under a fresh ctx + all-zero
    /// padding. §9.3.3.1.1.7 + Table 9-39 row `ctxIdxOffset=40`.
    ///
    /// We avoid hand-rolling the exact bin encoding for 0/1/-1/… because
    /// CABAC's arithmetic state depends on the fresh context init; the
    /// oracle-comparison pattern gives us bit-accurate regression
    /// coverage without having to reproduce the entire §9.3 decoder
    /// here.
    #[test]
    fn cabac_mvd_matches_standalone_oracle() {
        // Several different streams — the values will differ, but the
        // invariant to check is `parse_mb_pred` and
        // `decode_mvd_lx` (standalone) observe the same stream and so
        // must produce the same numbers.
        let streams: &[[u8; 16]] = &[
            [0x00; 16],
            [0xFF; 16],
            [0xA5, 0x5A, 0x3C, 0xC3, 0x81, 0x7E, 0x12, 0xED,
             0x00, 0xFF, 0x80, 0x01, 0x22, 0x44, 0x88, 0x77],
            [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE,
             0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
        ];
        let mb_type = MbType::PL016x16;
        for (i, data) in streams.iter().enumerate() {
            let (pred, _) =
                run_cabac_mb_pred(data, SliceKind::P, &mb_type, 0, 0);
            // Oracle: decode_mvd_lx called standalone on the same stream
            // (fresh context, fresh engine) gives the first mvd X
            // exactly; chaining two calls gives (X, Y). parse_mb_pred
            // issues the identical pair for part 0 since ref_idx_l0 is
            // gated out (num_ref_idx_l0_active_minus1 == 0).
            let mut dec = CabacDecoder::new(BitReader::new(data)).unwrap();
            let mut ctxs =
                CabacContexts::init(SliceKind::P, Some(0), 26).unwrap();
            let mx = decode_mvd_lx(&mut dec, &mut ctxs, MvdComponent::X, 0)
                .unwrap();
            let my = decode_mvd_lx(&mut dec, &mut ctxs, MvdComponent::Y, 0)
                .unwrap();
            assert_eq!(
                pred.mvd_l0[0],
                [mx, my],
                "stream #{i}: parse_mb_pred mvd_l0[0] must match standalone decode_mvd_lx pair",
            );
            // Also cross-check single-element consumption to prove the
            // engine actually walked through `decode_mvd_lx` and not a
            // zero stub (the stub case would return [0, 0] on every
            // stream — which we'd detect here because at least one of
            // our streams produces a non-zero mvd).
            let (single_mx, bins_mx) =
                oracle_single_mvd(data, SliceKind::P, MvdComponent::X);
            assert!(
                bins_mx > 0,
                "stream #{i}: decode_mvd_lx must consume ≥1 bin",
            );
            // The first component of pred.mvd_l0[0] must equal the
            // standalone decode result.
            assert_eq!(
                pred.mvd_l0[0][0], single_mx,
                "stream #{i}: first mvd X must match",
            );
        }
        // At least one of the streams above must produce a non-zero
        // mvd for some component — otherwise the oracle comparison is
        // a tautology. Confirm the combined harness is discriminating.
        let mut any_nonzero = false;
        for data in streams {
            let (pred, _) =
                run_cabac_mb_pred(data, SliceKind::P, &mb_type, 0, 0);
            if pred.mvd_l0[0][0] != 0 || pred.mvd_l0[0][1] != 0 {
                any_nonzero = true;
                break;
            }
        }
        assert!(
            any_nonzero,
            "no test stream produced a non-zero mvd — tighten inputs",
        );
    }

    // --------------------------------------------------------------
    // sub_mb_type CABAC wiring tests.
    //
    // parse_sub_mb_pred must dispatch to decode_sub_mb_type_p for P/SP
    // slices and decode_sub_mb_type_b for B slices (§9.3.3.1.2 /
    // Table 9-39 — ctxIdxOffset = 21 (P/SP) or 36 (B)). Skipping those
    // calls — which is what the former stub did — leaves the sub_mb_type
    // bins in the stream unconsumed. This test confirms that:
    //   - For a P slice's P_8x8 mb_type, the CABAC engine consumes at
    //     least one bin (vs. zero for the old stub).
    //   - The P-slice path never touches B-slice ctxs (36..=39).
    //   - The B-slice path consumes bins when mb_type is B_8x8.
    // --------------------------------------------------------------

    /// Helper: run parse_sub_mb_pred over a fresh CABAC engine on
    /// `data`, return the resulting SubMbPred and the number of bins
    /// consumed by CABAC.
    fn run_cabac_sub_mb_pred(
        data: &[u8],
        slice_kind: SliceKind,
        mb_type: &MbType,
        num_ref_idx_l0_active_minus1: u32,
        num_ref_idx_l1_active_minus1: u32,
    ) -> (SubMbPred, u64) {
        let mut dec = CabacDecoder::new(BitReader::new(data)).unwrap();
        let cabac_init_idc = match slice_kind {
            SliceKind::I | SliceKind::SI => None,
            _ => Some(0u32),
        };
        let mut ctxs = CabacContexts::init(slice_kind, cabac_init_idc, 26).unwrap();
        let bins_before = dec.bin_count();
        let dummy = [0u8; 1];
        let mut r = BitReader::new(&dummy);
        let mut entropy = EntropyState {
            cabac: Some((&mut dec, &mut ctxs)),
            slice_kind,
            neighbours: NeighbourCtx::default(),
            prev_mb_qp_delta_nonzero: false,
            chroma_array_type: 1,
            transform_8x8_mode_flag: false,
            cavlc_nc: None,
            current_mb_addr: 0,
            constrained_intra_pred_flag: false,
            num_ref_idx_l0_active_minus1,
            num_ref_idx_l1_active_minus1,
            mbaff_frame_flag: false,
            cabac_nb: None,
            pic_width_in_mbs: 0,
        };
        let pred = parse_sub_mb_pred(&mut r, &mut entropy, mb_type).unwrap();
        let bins_used = dec.bin_count() - bins_before;
        (pred, bins_used)
    }

    /// P slice P_8x8 must invoke decode_sub_mb_type_p — i.e. consume at
    /// least 4 bins (one per 8x8 part; minimum 1 bin per part for the
    /// P_L0_8x8 bin string "1" per Table 9-37).
    #[test]
    fn cabac_sub_mb_type_p_consumes_bins() {
        // Stream of all-0xFF bytes forces CABAC's MPS/LPS decisions to
        // march through context states; the important property is that
        // decode_sub_mb_type_p is actually called (so ≥ 4 bins for the
        // 4 sub-MB parts).
        let data: [u8; 16] = [0xFF; 16];
        let mb_type = MbType::P8x8;
        let (_pred, bins) =
            run_cabac_sub_mb_pred(&data, SliceKind::P, &mb_type, 0, 0);
        assert!(
            bins >= 4,
            "P-slice sub_mb_type must consume ≥ 4 bins (one per part) — got {bins}",
        );
    }

    /// P slice sub_mb_type must be decoded by `decode_sub_mb_type_p`, not
    /// `decode_sub_mb_type_b`. Cross-check by comparing to the standalone
    /// oracle.
    #[test]
    fn cabac_sub_mb_type_p_matches_standalone_p_decoder() {
        let data: [u8; 16] = [
            0xA5, 0x5A, 0x3C, 0xC3, 0x81, 0x7E, 0x12, 0xED,
            0x00, 0xFF, 0x80, 0x01, 0x22, 0x44, 0x88, 0x77,
        ];
        let mb_type = MbType::P8x8;
        let (pred, _) =
            run_cabac_sub_mb_pred(&data, SliceKind::P, &mb_type, 0, 0);

        // Oracle: invoke decode_sub_mb_type_p four times on a fresh
        // engine over the same data — must yield the same 4 raw values.
        let mut dec = CabacDecoder::new(BitReader::new(&data)).unwrap();
        let mut ctxs =
            CabacContexts::init(SliceKind::P, Some(0), 26).unwrap();
        let mut expected_raws = [0u32; 4];
        for r in expected_raws.iter_mut() {
            *r = crate::cabac_ctx::decode_sub_mb_type_p(&mut dec, &mut ctxs)
                .unwrap();
        }
        // Map raw to SubMbType via from_p (§7.3.5.2).
        for (i, &raw) in expected_raws.iter().enumerate() {
            let expected = SubMbType::from_p(raw).unwrap();
            assert_eq!(
                pred.sub_mb_type[i], expected,
                "sub_mb_type[{i}] mismatch: expected {expected:?} got {:?}",
                pred.sub_mb_type[i],
            );
        }
    }

    /// B slice B_8x8 must invoke decode_sub_mb_type_b, not _p.
    #[test]
    fn cabac_sub_mb_type_b_matches_standalone_b_decoder() {
        let data: [u8; 16] = [
            0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE,
            0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
        ];
        let mb_type = MbType::B8x8;
        let (pred, bins) =
            run_cabac_sub_mb_pred(&data, SliceKind::B, &mb_type, 0, 0);
        assert!(
            bins >= 4,
            "B-slice sub_mb_type must consume ≥ 4 bins (one per part) — got {bins}",
        );

        // Oracle: invoke decode_sub_mb_type_b four times on a fresh
        // engine over the same data — must yield the same 4 raw values.
        let mut dec = CabacDecoder::new(BitReader::new(&data)).unwrap();
        let mut ctxs =
            CabacContexts::init(SliceKind::B, Some(0), 26).unwrap();
        let mut expected_raws = [0u32; 4];
        for r in expected_raws.iter_mut() {
            *r = crate::cabac_ctx::decode_sub_mb_type_b(&mut dec, &mut ctxs)
                .unwrap();
        }
        for (i, &raw) in expected_raws.iter().enumerate() {
            let expected = SubMbType::from_b(raw).unwrap();
            assert_eq!(
                pred.sub_mb_type[i], expected,
                "sub_mb_type[{i}] mismatch: expected {expected:?} got {:?}",
                pred.sub_mb_type[i],
            );
        }
    }

    /// Regression: the pre-wiring stub returned raw=0 without consuming
    /// any bins. Now the dispatch must produce non-zero bin counts for
    /// both P and B slices.
    #[test]
    fn cabac_sub_mb_type_not_stubbed_zero_bins() {
        let data: [u8; 16] = [
            0x80, 0x00, 0x01, 0xFE, 0x7F, 0x55, 0xAA, 0x33,
            0xCC, 0x11, 0xEE, 0x00, 0xFF, 0x00, 0xFF, 0x00,
        ];
        let (_, p_bins) = run_cabac_sub_mb_pred(
            &data, SliceKind::P, &MbType::P8x8, 0, 0,
        );
        assert!(p_bins > 0, "P sub_mb_type must consume bins (got 0)");
        let (_, b_bins) = run_cabac_sub_mb_pred(
            &data, SliceKind::B, &MbType::B8x8, 0, 0,
        );
        assert!(b_bins > 0, "B sub_mb_type must consume bins (got 0)");
    }

    // -----------------------------------------------------------------
    // §7.3.5.3 — ChromaArrayType == 3 (4:4:4) residual parse tests.
    //
    // In 4:4:4 mode, Cb and Cr are each coded like luma: Intra_16x16
    // MBs have per-plane 16x16 DC + 16 4x4 AC blocks; Intra_NxN / Inter
    // MBs have 16 per-plane 4x4 blocks gated by cbp_luma. None of these
    // paths existed before; the tests verify the parse completes and
    // the coefficients land in the new `residual_*_luma_like` /
    // `residual_*_16x16_dc` storage fields.
    // -----------------------------------------------------------------

    /// Build a minimal EntropyState for CAVLC 4:4:4 tests
    /// (ChromaArrayType == 3).
    fn cavlc_entropy_444() -> EntropyState<'static, 'static> {
        EntropyState {
            cabac: None,
            slice_kind: SliceKind::I,
            neighbours: NeighbourCtx::default(),
            prev_mb_qp_delta_nonzero: false,
            chroma_array_type: 3,
            transform_8x8_mode_flag: false,
            cavlc_nc: None,
            current_mb_addr: 0,
            constrained_intra_pred_flag: false,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            mbaff_frame_flag: false,
            cabac_nb: None,
            pic_width_in_mbs: 0,
        }
    }

    /// §7.3.5.3 / §9.2.1 — CAVLC Intra_16x16 in 4:4:4 with cbp_luma=0,
    /// cbp_chroma=0. The residual consists of exactly three DC blocks
    /// (Y, Cb, Cr), each a single TotalCoeff=0 coeff_token ("1") in the
    /// nC<2 column. No AC blocks follow (cbp_luma=0 gates them off for
    /// all three planes).
    #[test]
    fn cavlc_intra_16x16_444_all_zero_dc() {
        let mut w = BitWriter::new();
        w.ue(1); // mb_type = 1 → Intra_16x16_0_0_0 (pred=0, cbp_luma=0, cbp_chroma=0)
        // §7.3.5.1: intra_chroma_pred_mode is only read when
        // ChromaArrayType == 1 || 2 — NOT for 4:4:4.
        w.se(0); // mb_qp_delta
        w.u(1, 1); // Y DC coeff_token TotalCoeff=0 (nC<2) = "1"
        w.u(1, 1); // Cb DC coeff_token TotalCoeff=0 = "1"
        w.u(1, 1); // Cr DC coeff_token TotalCoeff=0 = "1"
        w.trailing();
        let bytes = w.into_bytes();

        let sps = dummy_sps();
        let pps = dummy_pps();
        let hdr = dummy_slice_header(SliceType::I);
        let mut r = BitReader::new(&bytes);
        let mut entropy = cavlc_entropy_444();

        let mb = parse_macroblock(&mut r, &mut entropy, &hdr, &sps, &pps, 0).unwrap();
        assert!(matches!(
            mb.mb_type,
            MbType::Intra16x16(Intra16x16 {
                pred_mode: 0,
                cbp_luma: 0,
                cbp_chroma: 0
            })
        ));
        assert_eq!(mb.coded_block_pattern, 0);
        // Y DC present, Cb DC + Cr DC present (4:4:4 Intra_16x16 always
        // parses per-plane DC blocks even when cbp_luma/chroma == 0).
        assert!(mb.residual_luma_dc.is_some());
        assert!(mb.residual_cb_16x16_dc.is_some());
        assert!(mb.residual_cr_16x16_dc.is_some());
        // All 16 coefficients of each DC block are zero (TotalCoeff=0).
        assert_eq!(mb.residual_luma_dc.unwrap(), [0; 16]);
        assert_eq!(mb.residual_cb_16x16_dc.unwrap(), [0; 16]);
        assert_eq!(mb.residual_cr_16x16_dc.unwrap(), [0; 16]);
        // No AC blocks (cbp_luma == 0 gates all planes).
        assert!(mb.residual_luma.is_empty());
        assert!(mb.residual_cb_luma_like.is_empty());
        assert!(mb.residual_cr_luma_like.is_empty());
        // No 4:2:0 / 4:2:2 chroma path state.
        assert!(mb.residual_chroma_dc_cb.is_empty());
        assert!(mb.residual_chroma_ac_cb.is_empty());
    }

    /// §7.3.5.3 / §9.2.1 — CAVLC Intra_16x16 in 4:4:4 with cbp_luma=15,
    /// cbp_chroma=0. Each plane (Y, Cb, Cr) emits 1 DC block + 16 AC
    /// blocks. All blocks carry TotalCoeff=0 so each is a single "1"
    /// bit coeff_token. Total: 3 * (1 + 16) = 51 bits.
    #[test]
    fn cavlc_intra_16x16_444_all_zero_with_ac() {
        let mut w = BitWriter::new();
        w.ue(13); // mb_type = 13 → Intra_16x16_0_0_15 (pred=0, cbp_luma=15, cbp_chroma=0)
        w.se(0); // mb_qp_delta
        // Y: DC + 16 AC. Cb: DC + 16 AC. Cr: DC + 16 AC.
        // Each block TotalCoeff=0 → coeff_token = "1" (nC<2 column).
        for _ in 0..(3 * 17) {
            w.u(1, 1);
        }
        w.trailing();
        let bytes = w.into_bytes();

        let sps = dummy_sps();
        let pps = dummy_pps();
        let hdr = dummy_slice_header(SliceType::I);
        let mut r = BitReader::new(&bytes);
        let mut entropy = cavlc_entropy_444();

        let mb = parse_macroblock(&mut r, &mut entropy, &hdr, &sps, &pps, 0).unwrap();
        assert!(matches!(
            mb.mb_type,
            MbType::Intra16x16(Intra16x16 {
                pred_mode: 0,
                cbp_luma: 15,
                cbp_chroma: 0
            })
        ));
        // Y: DC + 16 AC.
        assert!(mb.residual_luma_dc.is_some());
        assert_eq!(mb.residual_luma.len(), 16);
        // Cb: DC + 16 AC (stored in luma-like fields).
        assert!(mb.residual_cb_16x16_dc.is_some());
        assert_eq!(mb.residual_cb_luma_like.len(), 16);
        // Cr: DC + 16 AC.
        assert!(mb.residual_cr_16x16_dc.is_some());
        assert_eq!(mb.residual_cr_luma_like.len(), 16);
        // Every block is zero (TotalCoeff=0 → empty coefficient vector).
        for blk in &mb.residual_luma {
            assert_eq!(*blk, [0; 16]);
        }
        for blk in &mb.residual_cb_luma_like {
            assert_eq!(*blk, [0; 16]);
        }
        for blk in &mb.residual_cr_luma_like {
            assert_eq!(*blk, [0; 16]);
        }
    }

    /// §7.3.5.3 — CAVLC I_NxN in 4:4:4 with all-zero CBP. Verifies the
    /// parser doesn't error when ChromaArrayType == 3 but there's no
    /// residual to read — the Cb/Cr plane loops are gated by cbp_luma
    /// just like luma.
    #[test]
    fn cavlc_i_nxn_444_all_zero_residual() {
        // §9.1.2 / Table 9-4(b): ChromaArrayType ∈ {0, 3} uses
        // `ME_INTRA_0_3`. codeNum=1 → CBP=0 for intra.
        assert_eq!(ME_INTRA_0_3[1], 0);

        let mut w = BitWriter::new();
        w.ue(0); // mb_type = I_NxN
        for _ in 0..16 {
            w.u(1, 1); // prev_intra4x4_pred_mode_flag
        }
        // §7.3.5.1: no intra_chroma_pred_mode for 4:4:4.
        w.ue(1); // coded_block_pattern codeNum=1 → CBP=0 (4:4:4 intra table)
        w.trailing();
        let bytes = w.into_bytes();

        let sps = dummy_sps();
        let pps = dummy_pps();
        let hdr = dummy_slice_header(SliceType::I);
        let mut r = BitReader::new(&bytes);
        let mut entropy = cavlc_entropy_444();

        let mb = parse_macroblock(&mut r, &mut entropy, &hdr, &sps, &pps, 0).unwrap();
        assert_eq!(mb.mb_type, MbType::INxN);
        assert_eq!(mb.coded_block_pattern, 0);
        assert!(mb.residual_luma.is_empty());
        assert!(mb.residual_cb_luma_like.is_empty());
        assert!(mb.residual_cr_luma_like.is_empty());
        assert!(mb.residual_cb_16x16_dc.is_none());
        assert!(mb.residual_cr_16x16_dc.is_none());
    }

    /// §7.3.5.3 — CAVLC Inter P_L0_16x16 in 4:4:4 with cbp_total = 0.
    /// Verifies the parse succeeds and that the 4:4:4 branch is also
    /// reachable in inter macroblocks (not just intra).
    #[test]
    fn cavlc_inter_444_all_zero_residual() {
        // codeNum=0 → CBP 0 in the inter 4:2:0/4:2:2 table.
        assert_eq!(ME_INTER_420_422[0], 0);

        let mut w = BitWriter::new();
        w.ue(0); // mb_type = P_L0_16x16 (inter, one 16x16 partition)
        // mb_pred: since num_ref_idx_l0_active_minus1 == 0 and not MBAFF,
        // no ref_idx_l0 is read. MVD_l0 per component with value 0:
        w.se(0); // mvd_l0[0][0][0]  (x)
        w.se(0); // mvd_l0[0][0][1]  (y)
        w.ue(0); // coded_block_pattern codeNum=0 → CBP 0 (inter)
        // cbp_luma == 0 AND cbp_chroma == 0 AND not Intra_16x16 →
        // mb_qp_delta is NOT read, residual block entirely skipped.
        w.trailing();
        let bytes = w.into_bytes();

        let sps = dummy_sps();
        let pps = dummy_pps();
        let hdr = dummy_slice_header(SliceType::P);
        let mut r = BitReader::new(&bytes);
        let mut entropy = cavlc_entropy_444();
        entropy.slice_kind = SliceKind::P;

        let mb = parse_macroblock(&mut r, &mut entropy, &hdr, &sps, &pps, 0).unwrap();
        assert_eq!(mb.mb_type, MbType::PL016x16);
        assert_eq!(mb.coded_block_pattern, 0);
        assert!(mb.residual_luma.is_empty());
        assert!(mb.residual_cb_luma_like.is_empty());
        assert!(mb.residual_cr_luma_like.is_empty());
        assert!(mb.residual_luma_dc.is_none());
        assert!(mb.residual_cb_16x16_dc.is_none());
        assert!(mb.residual_cr_16x16_dc.is_none());
    }

    /// §9.2.1.1 — CAVLC 4:4:4 nC neighbour derivation uses per-plane
    /// totals. `nc_nn_plane_luma_like` returns the Cb/Cr
    /// `_luma_total_coeff` array entry of an available MB, with the
    /// standard skip/i_pcm fallbacks of §9.2.1.1 step 6.
    #[test]
    fn cavlc_444_nc_plane_reads_per_plane_totals() {
        let mut grid = CavlcNcGrid::new(2, 2);
        let slot = &mut grid.mbs[0];
        slot.is_available = true;
        slot.is_intra = true;
        slot.cb_luma_total_coeff[5] = 7;
        slot.cr_luma_total_coeff[5] = 11;

        // Cb plane: read block index 5 from MB 0.
        let (avail, n) = nc_nn_plane_luma_like(&grid, Some(0), 5, false, true, false);
        assert!(avail);
        assert_eq!(n, 7);

        // Cr plane: same block index, different array.
        let (avail, n) = nc_nn_plane_luma_like(&grid, Some(0), 5, true, true, false);
        assert!(avail);
        assert_eq!(n, 11);

        // Unavailable MB → (false, 0).
        let (avail, _) = nc_nn_plane_luma_like(&grid, Some(99), 5, false, true, false);
        assert!(!avail);

        // Skip neighbour → (true, 0).
        grid.mbs[1].is_available = true;
        grid.mbs[1].is_skip = true;
        grid.mbs[1].cb_luma_total_coeff[5] = 9;
        let (avail, n) = nc_nn_plane_luma_like(&grid, Some(1), 5, false, true, false);
        assert!(avail);
        assert_eq!(n, 0);

        // I_PCM neighbour → (true, 16) per §9.2.1.1 step 6.
        grid.mbs[1].is_skip = false;
        grid.mbs[1].is_i_pcm = true;
        let (avail, n) = nc_nn_plane_luma_like(&grid, Some(1), 5, false, true, false);
        assert!(avail);
        assert_eq!(n, 16);
    }

    /// §7.3.5.3 / §9.3.3.1.1.9 — CABAC 4:4:4 reaches the per-plane
    /// code path when chroma_array_type == 3. This test drives the
    /// parse through `parse_residual_cabac_only` for an Intra_16x16 MB
    /// with cbp_luma=0 / cbp_chroma=0 (smallest residual in 4:4:4:
    /// only Y DC, Cb DC, Cr DC blocks are parsed). We don't construct
    /// a real CABAC arithmetic stream here — instead we verify that
    /// the dispatch path does NOT error out with
    /// `UnsupportedChromaArrayType(3)` when ChromaArrayType == 3.
    #[test]
    fn cabac_444_intra_16x16_residual_path_reachable() {
        // A byte-stream that yields a short sequence of "0" bins in the
        // CABAC engine — sufficient to get `coded_block_flag = 0` for
        // each of the three DC blocks so no further significance-map
        // bins are read. For the Intra_16x16 / cbp=0 case this is the
        // only residual we need to parse.
        // Any short zero-biased payload works: the contexts start with
        // `valMPS=0, pStateIdx=low` in many Intra slots, making the
        // first bin likely LPS=0. The test asserts only that parsing
        // does NOT return `UnsupportedChromaArrayType(3)`.
        let data: [u8; 32] = [
            // CABAC engine state: codIRange=510, codIOffset = first 9 bits.
            // Pad enough for several decode_decision calls.
            0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let mut dec = CabacDecoder::new(BitReader::new(&data)).unwrap();
        let mut ctxs = CabacContexts::init(SliceKind::I, None, 26).unwrap();
        let mut out = Macroblock::new_skip(SliceType::I);
        // Set it up to look like a non-skip Intra_16x16 MB.
        out.is_skip = false;
        let mb_type = MbType::Intra16x16(Intra16x16 {
            pred_mode: 0,
            cbp_luma: 0,
            cbp_chroma: 0,
        });
        let res = parse_residual_cabac_only(
            &mut dec,
            &mut ctxs,
            /*chroma_array_type=*/ 3,
            &mb_type,
            /*cbp_luma=*/ 0,
            /*cbp_chroma=*/ 0,
            /*transform_size_8x8_flag=*/ false,
            &mut out,
            /*nb_grid=*/ None,
            /*current_mb_addr=*/ 0,
            /*current_is_intra=*/ true,
        );
        // The 4:4:4 branch must return Ok (possibly with all-zero
        // residual blocks) — NOT UnsupportedChromaArrayType.
        assert!(
            !matches!(
                res,
                Err(MacroblockLayerError::UnsupportedChromaArrayType(3))
            ),
            "CABAC 4:4:4 must not return UnsupportedChromaArrayType(3)"
        );
        // Whether res is Ok or some other CABAC error depends on the
        // specific bit pattern — we only assert that the 4:4:4 path
        // no longer rejects with UnsupportedChromaArrayType(3).
        if res.is_ok() {
            // When coded_block_flag decodes to 0 for all three planes'
            // DC blocks, the corresponding residuals are present as
            // all-zero vectors (parse_residual_block_cabac pads
            // `out` to `len` = 16 and returns `coded = false`).
            assert!(out.residual_luma_dc.is_some());
            assert!(out.residual_cb_16x16_dc.is_some());
            assert!(out.residual_cr_16x16_dc.is_some());
        }
    }

    /// §9.3.3.1.1.9 — `cbf_luma_for` + `cbf_mb_level_for` route per-
    /// plane CBF queries to the plane-specific neighbour arrays.
    #[test]
    fn cabac_444_cbf_helpers_route_per_plane() {
        let mut info = CabacMbNeighbourInfo::default();
        info.available = true;
        info.cbf_luma_4x4[3] = true;
        info.cbf_cb_luma_4x4[3] = false;
        info.cbf_cr_luma_4x4[3] = true;
        info.cbf_luma_16x16_dc = true;
        info.cbf_cb_16x16_dc = false;
        info.cbf_cr_16x16_dc = true;
        info.cbf_luma_16x16_ac[5] = true;
        info.cbf_cb_16x16_ac[5] = false;
        info.cbf_cr_16x16_ac[5] = true;

        // Luma 4x4: route to `cbf_luma_4x4`.
        assert!(cbf_luma_for(&info, BlockType::Luma4x4, 3));
        // Cb 4x4: route to `cbf_cb_luma_4x4`.
        assert!(!cbf_luma_for(&info, BlockType::CbLuma4x4, 3));
        // Cr 4x4: route to `cbf_cr_luma_4x4`.
        assert!(cbf_luma_for(&info, BlockType::CrLuma4x4, 3));

        // 16x16 AC routing.
        assert!(cbf_luma_for(&info, BlockType::Luma16x16Ac, 5));
        assert!(!cbf_luma_for(&info, BlockType::CbIntra16x16Ac, 5));
        assert!(cbf_luma_for(&info, BlockType::CrIntra16x16Ac, 5));

        // 16x16 DC routing.
        assert!(cbf_mb_level_for(&info, BlockType::Luma16x16Dc, false));
        assert!(!cbf_mb_level_for(&info, BlockType::CbIntra16x16Dc, false));
        assert!(cbf_mb_level_for(&info, BlockType::CrIntra16x16Dc, true));
    }
}
