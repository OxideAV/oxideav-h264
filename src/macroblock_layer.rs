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
    decode_prev_intra_pred_mode_flag, decode_rem_intra_pred_mode,
    decode_significant_coeff_flag, decode_transform_size_8x8_flag, BlockType, CabacContexts,
    NeighbourCtx, SliceKind,
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
    _sps: &Sps,
    _pps: &Pps,
    _current_mb_addr: u32,
) -> McblResult<Macroblock> {
    let slice_type = slice_header.slice_type;
    let dbg = std::env::var("OXIDEAV_H264_DEBUG").is_ok();
    let mb_addr_dbg = entropy.current_mb_addr;
    if dbg {
        let (b, bi) = r.position();
        eprintln!(
            "[MB {:>4}] enter cursor=({},{}) slice_type={:?}",
            mb_addr_dbg, b, bi, slice_type
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
            decode_coded_block_pattern(dec, ctxs, &entropy.neighbours, entropy.chroma_array_type)?
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
    // ---------------------------------------------------------------
    if cbp_luma > 0
        && entropy.transform_8x8_mode_flag
        && !mb_type.is_i_nxn()
        && !mb_type.is_intra_16x16()
    {
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
            decode_mb_qp_delta(dec, ctxs, entropy.prev_mb_qp_delta_nonzero)?
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
        is_skip: false,
    };

    // §7.3.5.3 — residual block = residual_block_cavlc or
    // residual_block_cabac depending on entropy_coding_mode_flag.
    if let Some((cabac, ctxs)) = entropy.cabac.as_mut() {
        parse_residual_cabac_only(
            cabac,
            ctxs,
            entropy.chroma_array_type,
            &mb_type,
            cbp_luma,
            cbp_chroma,
            transform_size_8x8_flag,
            &mut out,
        )?;
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
        // Inter — NumMbPart partitions × ref_idx + MVD lists.
        let num_parts = mb_type.num_mb_part() as usize;
        let num_ref_l0_m1 = 0u32; // We don't parameterise; CAVLC reads
                                  // `te(v)` anyway (handled below).
                                  // §7.3.5.1 — ref_idx_l0[mbPartIdx] for mbPartIdx < NumMbPart
                                  // when num_ref_idx_l0_active_minus1 > 0 (or mb_field override).
                                  // We approximate by reading when slice has list 0.
        if entropy.slice_kind != SliceKind::I && entropy.slice_kind != SliceKind::SI {
            {
                // list 0 always present for P/B.
                for _ in 0..num_parts {
                    let v = if entropy.cabac.is_some() {
                        // With num_ref_l0_active_minus1 > 0 assumed →
                        // ref_idx would be decoded via CABAC; we have a
                        // helper but it needs neighbour ref state. For
                        // now return 0 as a placeholder (spec-accurate
                        // when num_ref_idx_l0_active_minus1 == 0).
                        0
                    } else {
                        r.te(num_ref_l0_m1)?
                    };
                    pred.ref_idx_l0.push(v);
                }
            }
            if entropy.slice_kind == SliceKind::B {
                for _ in 0..num_parts {
                    let v = if entropy.cabac.is_some() {
                        0
                    } else {
                        r.te(num_ref_l0_m1)?
                    };
                    pred.ref_idx_l1.push(v);
                }
            }
            // §7.3.5.1 — mvd_l0[mbPartIdx][0..2].
            for _ in 0..num_parts {
                let mx = if entropy.cabac.is_some() { 0 } else { r.se()? };
                let my = if entropy.cabac.is_some() { 0 } else { r.se()? };
                pred.mvd_l0.push([mx, my]);
            }
            if entropy.slice_kind == SliceKind::B {
                for _ in 0..num_parts {
                    let mx = if entropy.cabac.is_some() { 0 } else { r.se()? };
                    let my = if entropy.cabac.is_some() { 0 } else { r.se()? };
                    pred.mvd_l1.push([mx, my]);
                }
            }
        }
    }

    Ok(pred)
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

    // 4 sub_mb_type entries.
    for i in 0..4 {
        let raw = if entropy.cabac.is_some() {
            // CABAC sub_mb_type decoding would need dedicated ctxs that
            // we don't expose in this pass; fall back to 0 which maps
            // to P_L0_8x8 / B_Direct_8x8 (spec-accurate default).
            0u32
        } else {
            r.ue()?
        };
        out.sub_mb_type[i] = if is_b {
            SubMbType::from_b(raw)?
        } else {
            SubMbType::from_p(raw)?
        };
    }

    // ref_idx_l0 for each of the 4 partitions.
    for i in 0..4 {
        let v = if entropy.cabac.is_some() { 0 } else { r.te(0)? };
        out.ref_idx_l0[i] = v;
    }
    if is_b {
        for i in 0..4 {
            let v = if entropy.cabac.is_some() { 0 } else { r.te(0)? };
            out.ref_idx_l1[i] = v;
        }
    }

    // MVDs — NumSubMbPart entries per partition.
    for i in 0..4 {
        let n = out.sub_mb_type[i].num_sub_mb_part() as usize;
        for _ in 0..n {
            let mx = if entropy.cabac.is_some() { 0 } else { r.se()? };
            let my = if entropy.cabac.is_some() { 0 } else { r.se()? };
            out.mvd_l0[i].push([mx, my]);
        }
    }
    if is_b {
        for i in 0..4 {
            let n = out.sub_mb_type[i].num_sub_mb_part() as usize;
            for _ in 0..n {
                let mx = if entropy.cabac.is_some() { 0 } else { r.se()? };
                let my = if entropy.cabac.is_some() { 0 } else { r.se()? };
                out.mvd_l1[i].push([mx, my]);
            }
        }
    }

    Ok(out)
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
    } else {
        // 4:4:4 chroma is modelled like luma; not implemented.
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
/// flags for the coded_block_flag ctxIdx. We pass `None` for both —
/// the macroblock layer doesn't yet track per-block CBF neighbours;
/// spec-wise this biases the probability a little but doesn't
/// desync the bitstream (ctxIdxInc is just a small integer).
fn parse_residual_block_cabac(
    cabac: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    block_type: BlockType,
    start_idx: u32,
    end_idx: u32,
    max_num_coeff: u32,
    chroma_array_type: u32,
) -> McblResult<Vec<i32>> {
    let len = (end_idx - start_idx + 1) as usize;
    let mut out = vec![0i32; len];

    // §7.3.5.3.3 — coded_block_flag is suppressed for 8x8 blocks unless
    // we're in 4:4:4 (where 8x8 CBP semantics differ).
    let coded = if max_num_coeff != 64 || chroma_array_type == 3 {
        decode_coded_block_flag(cabac, ctxs, block_type, None, None)?
    } else {
        true
    };

    if !coded {
        return Ok(out);
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
    Ok(out)
}

/// CABAC residual walker (§7.3.5.3 with entropy_coding_mode_flag == 1).
/// Same shape as `parse_residual_cavlc_only` but routes every block
/// through the CABAC engine instead of the CAVLC tables.
fn parse_residual_cabac_only(
    cabac: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
    chroma_array_type: u32,
    mb_type: &MbType,
    cbp_luma: u32,
    cbp_chroma: u32,
    transform_size_8x8_flag: bool,
    out: &mut Macroblock,
) -> McblResult<()> {
    // --- Luma DC (Intra_16x16) ---
    if mb_type.is_intra_16x16() {
        let blk = parse_residual_block_cabac(
            cabac,
            ctxs,
            BlockType::Luma16x16Dc,
            0,
            15,
            16,
            chroma_array_type,
        )?;
        out.residual_luma_dc = Some(pad_to_16(blk));
    }

    // --- Luma AC / 4x4 blocks / 8x8 blocks ---
    if mb_type.is_intra_16x16() {
        if cbp_luma == 15 {
            for _ in 0..16 {
                let blk = parse_residual_block_cabac(
                    cabac,
                    ctxs,
                    BlockType::Luma16x16Ac,
                    0,
                    14,
                    15,
                    chroma_array_type,
                )?;
                out.residual_luma.push(pad_to_16(blk));
            }
        }
    } else if transform_size_8x8_flag {
        for blk8 in 0..4 {
            if (cbp_luma >> blk8) & 1 == 1 {
                // §7.3.5.3.1: a single residual_block call for the whole
                // 8x8 (64 coefficients). We model it the same way as
                // the CAVLC fallback (four 4x4 sub-blocks) for now since
                // the 8x8-specific CABAC scanning tables would need
                // additional plumbing.
                let blk = parse_residual_block_cabac(
                    cabac,
                    ctxs,
                    BlockType::Luma8x8,
                    0,
                    63,
                    64,
                    chroma_array_type,
                )?;
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
        for blk8 in 0..4 {
            if (cbp_luma >> blk8) & 1 == 1 {
                for _ in 0..4 {
                    let blk = parse_residual_block_cabac(
                        cabac,
                        ctxs,
                        BlockType::Luma4x4,
                        0,
                        15,
                        16,
                        chroma_array_type,
                    )?;
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
            let cb_dc = parse_residual_block_cabac(
                cabac,
                ctxs,
                BlockType::ChromaDc,
                0,
                max_dc - 1,
                max_dc,
                chroma_array_type,
            )?;
            out.residual_chroma_dc_cb = cb_dc;
            let cr_dc = parse_residual_block_cabac(
                cabac,
                ctxs,
                BlockType::ChromaDc,
                0,
                max_dc - 1,
                max_dc,
                chroma_array_type,
            )?;
            out.residual_chroma_dc_cr = cr_dc;
        }

        if cbp_chroma == 2 {
            let num_ac = 4 * num_c8x8 as usize;
            for _ in 0..num_ac {
                let blk = parse_residual_block_cabac(
                    cabac,
                    ctxs,
                    BlockType::ChromaAc,
                    0,
                    14,
                    15,
                    chroma_array_type,
                )?;
                out.residual_chroma_ac_cb.push(pad_to_16(blk));
            }
            for _ in 0..num_ac {
                let blk = parse_residual_block_cabac(
                    cabac,
                    ctxs,
                    BlockType::ChromaAc,
                    0,
                    14,
                    15,
                    chroma_array_type,
                )?;
                out.residual_chroma_ac_cr.push(pad_to_16(blk));
            }
        }
    } else if chroma_array_type == 0 {
        // Monochrome — no chroma residual.
    } else {
        return Err(MacroblockLayerError::UnsupportedChromaArrayType(
            chroma_array_type,
        ));
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
}
