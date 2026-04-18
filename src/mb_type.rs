//! Macroblock-type bookkeeping for I- and P-slices — ITU-T H.264 §7.4.5.
//!
//! For an I-slice the parser decodes a `mb_type` ue(v). The encoded value
//! `mbType` distinguishes:
//!
//! * `0` → `I_NxN`: macroblock is partitioned into sixteen 4×4 luma blocks
//!   each with its own intra mode (or eight 8×8 if `transform_8x8_mode_flag`
//!   — not handled here).
//! * `1..=24` → `I_16x16`: full-block luma prediction, parameterised by
//!   `(intra16x16_mode, cbp_luma, cbp_chroma)`.
//! * `25` → `I_PCM`: raw uncompressed bytes follow.
//!
//! Table 7-11 maps mb_type → (Intra16x16PredMode, CodedBlockPatternLuma,
//! CodedBlockPatternChroma).
//!
//! For a P-slice (§7.4.5.1 Table 7-13) a `mb_type` of `0..=4` picks a
//! motion-compensated partition layout from list 0; anything `>= 5` is an
//! intra macroblock whose bookkeeping matches Table 7-11 with the offset
//! removed. `P_Skip` is not encoded as a `mb_type` — it's signalled by
//! `mb_skip_run` before the macroblock (CAVLC) or by `mb_skip_flag` (CABAC).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IMbType {
    /// `I_NxN` — 16 separate 4×4 luma intra-mode blocks.
    INxN,
    /// `I_16x16` with parameters baked in.
    I16x16 {
        /// 0..=3 (Vertical / Horizontal / DC / Plane).
        intra16x16_pred_mode: u8,
        /// 0 = no luma AC, 15 = all four 8×8 blocks have AC.
        cbp_luma: u8,
        /// 0 = no chroma, 1 = chroma DC only, 2 = chroma DC + AC.
        cbp_chroma: u8,
    },
    IPcm,
}

/// Decode an I-slice mb_type into the table 7-11 components.
///
/// Mapping follows the spec's Table 7-11: mb_type `n` in `1..=24`
/// decomposes as `n - 1 = Intra16x16PredMode + 4*CBPChromaClass + 12*CBPLumaFlag`
/// where `CBPLumaFlag = 0` for CBP_Y=0 or `1` for CBP_Y=15, and
/// `CBPChromaClass` is one of `0` (no chroma), `1` (chroma DC only),
/// `2` (chroma DC + AC). `Intra16x16PredMode` values `0..=3` map to
/// Vertical / Horizontal / DC / Plane per Table 8-4.
pub fn decode_i_slice_mb_type(mb_type: u32) -> Option<IMbType> {
    match mb_type {
        0 => Some(IMbType::INxN),
        25 => Some(IMbType::IPcm),
        n if (1..=24).contains(&n) => {
            let idx = n - 1;
            let pred = (idx % 4) as u8;
            let cbp_chroma = ((idx / 4) % 3) as u8;
            let cbp_luma = if idx >= 12 { 15 } else { 0 };
            Some(IMbType::I16x16 {
                intra16x16_pred_mode: pred,
                cbp_luma,
                cbp_chroma,
            })
        }
        _ => None,
    }
}

/// Luma partition shape chosen by the P-slice `mb_type` (§7.4.5.1
/// Table 7-13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PPartition {
    /// Whole macroblock = single 16×16 motion-compensated block.
    P16x16,
    /// Two 16×8 partitions — top then bottom.
    P16x8,
    /// Two 8×16 partitions — left then right.
    P8x16,
    /// Four 8×8 partitions, each with its own `sub_mb_type` + MV(s).
    P8x8,
    /// Same as `P8x8` but every sub-MB is forced to ref_idx_l0 = 0, and no
    /// per-partition ref_idx syntax is emitted (`mb_type = 4`).
    P8x8Ref0,
}

impl PPartition {
    /// Number of top-level motion partitions for this layout.
    pub fn num_partitions(self) -> usize {
        match self {
            PPartition::P16x16 => 1,
            PPartition::P16x8 | PPartition::P8x16 => 2,
            PPartition::P8x8 | PPartition::P8x8Ref0 => 4,
        }
    }
    /// Partition rectangle in 4×4-block units, (row, col, h, w) with row/col
    /// and h/w in units of four luma samples (so a 16×16 partition is 4×4
    /// blocks).
    pub fn partition_rect(self, idx: usize) -> (usize, usize, usize, usize) {
        match self {
            PPartition::P16x16 => (0, 0, 4, 4),
            PPartition::P16x8 => match idx {
                0 => (0, 0, 2, 4),
                _ => (2, 0, 2, 4),
            },
            PPartition::P8x16 => match idx {
                0 => (0, 0, 4, 2),
                _ => (0, 2, 4, 2),
            },
            PPartition::P8x8 | PPartition::P8x8Ref0 => match idx {
                0 => (0, 0, 2, 2),
                1 => (0, 2, 2, 2),
                2 => (2, 0, 2, 2),
                _ => (2, 2, 2, 2),
            },
        }
    }
}

/// Decoded P-slice macroblock type (§7.4.5.1 Table 7-13). For `mb_type`
/// values `>= 5`, the value is re-indexed against Table 7-11 and returned
/// as `IntraInP(..)` so the caller can reuse the I-slice decode path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PMbType {
    Inter { partition: PPartition },
    IntraInP(IMbType),
}

/// Decode a P-slice `mb_type` value (before the `P_Skip` short-circuit).
pub fn decode_p_slice_mb_type(mb_type: u32) -> Option<PMbType> {
    match mb_type {
        0 => Some(PMbType::Inter {
            partition: PPartition::P16x16,
        }),
        1 => Some(PMbType::Inter {
            partition: PPartition::P16x8,
        }),
        2 => Some(PMbType::Inter {
            partition: PPartition::P8x16,
        }),
        3 => Some(PMbType::Inter {
            partition: PPartition::P8x8,
        }),
        4 => Some(PMbType::Inter {
            partition: PPartition::P8x8Ref0,
        }),
        n if (5..=30).contains(&n) => decode_i_slice_mb_type(n - 5).map(PMbType::IntraInP),
        _ => None,
    }
}

/// Luma partition shape for `sub_mb_type` inside a `P_8x8` macroblock
/// (§7.4.5.2 Table 7-17 — P entries). Baseline uses list-0 motion only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PSubPartition {
    /// 8×8 — a single sub-partition covering the whole 8×8 block.
    Sub8x8,
    /// 8×4 — two horizontal halves.
    Sub8x4,
    /// 4×8 — two vertical halves.
    Sub4x8,
    /// 4×4 — four 4×4 sub-partitions.
    Sub4x4,
}

impl PSubPartition {
    pub fn num_sub_partitions(self) -> usize {
        match self {
            PSubPartition::Sub8x8 => 1,
            PSubPartition::Sub8x4 | PSubPartition::Sub4x8 => 2,
            PSubPartition::Sub4x4 => 4,
        }
    }
    /// Sub-partition rectangle in 4×4-block units, relative to the parent
    /// 8×8 block's top-left (so max extent is 2×2 blocks).
    pub fn sub_rect(self, idx: usize) -> (usize, usize, usize, usize) {
        match self {
            PSubPartition::Sub8x8 => (0, 0, 2, 2),
            PSubPartition::Sub8x4 => match idx {
                0 => (0, 0, 1, 2),
                _ => (1, 0, 1, 2),
            },
            PSubPartition::Sub4x8 => match idx {
                0 => (0, 0, 2, 1),
                _ => (0, 1, 2, 1),
            },
            PSubPartition::Sub4x4 => match idx {
                0 => (0, 0, 1, 1),
                1 => (0, 1, 1, 1),
                2 => (1, 0, 1, 1),
                _ => (1, 1, 1, 1),
            },
        }
    }
}

/// Decode a P-slice `sub_mb_type` value (Table 7-17, P-entries only).
pub fn decode_p_sub_mb_type(sub_mb_type: u32) -> Option<PSubPartition> {
    match sub_mb_type {
        0 => Some(PSubPartition::Sub8x8),
        1 => Some(PSubPartition::Sub8x4),
        2 => Some(PSubPartition::Sub4x8),
        3 => Some(PSubPartition::Sub4x4),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// B-slice mb_type + sub_mb_type — §7.4.5.1 Table 7-14, §7.4.5.2 Table 7-17.
// ---------------------------------------------------------------------------

/// Which of L0 / L1 a B-slice partition uses for prediction. `BiPred`
/// averages samples from both lists (per §8.4.2.3, default — or weighted
/// via `pred_weight_table` when `weighted_bipred_idc == 1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredDir {
    L0,
    L1,
    BiPred,
}

/// Partition layout + per-partition prediction direction for a B-slice
/// macroblock. Covers Table 7-14 entries; anything not in this enum surfaces
/// `Error::Unsupported` at decode time.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BPartition {
    /// Whole MB is derived from §8.4.1.2 direct-mode MVs, no MVD transmitted.
    Direct16x16,
    /// Whole-MB single-list-0 prediction.
    L0_16x16,
    /// Whole-MB single-list-1 prediction.
    L1_16x16,
    /// Whole-MB bi-prediction (average of L0 + L1 MC).
    Bi_16x16,
    /// Two 16×8 or 8×16 partitions, each with its own prediction direction.
    TwoPart16x8 {
        dirs: [PredDir; 2],
    },
    TwoPart8x16 {
        dirs: [PredDir; 2],
    },
    /// Four 8×8 sub-MBs — sub_mb_type follows (§7.4.5.2 Table 7-17).
    B8x8,
}

/// Decoded B-slice macroblock type. For `mb_type` values that code an intra
/// macroblock (23..48 in spec), the value is re-indexed against Table 7-11
/// and returned as `IntraInB(..)` so the caller reuses the I-slice decode
/// path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BMbType {
    Inter { partition: BPartition },
    IntraInB(IMbType),
}

/// Decode a B-slice `mb_type` per Table 7-14. Returns `None` for values the
/// decoder has not implemented (uncommon transpose or swap variants); the
/// caller surfaces `Error::Unsupported` when that happens.
///
/// Supported entries:
///   0   → B_Direct_16x16
///   1   → B_L0_16x16
///   2   → B_L1_16x16
///   3   → B_Bi_16x16
///   4   → B_L0_L0_16x8
///   5   → B_L0_L0_8x16
///   6   → B_L1_L1_16x8
///   7   → B_L1_L1_8x16
///   8   → B_L0_L1_16x8
///   9   → B_L0_L1_8x16
///   10  → B_L1_L0_16x8
///   11  → B_L1_L0_8x16
///   12  → B_L0_Bi_16x8
///   13  → B_L0_Bi_8x16
///   14  → B_L1_Bi_16x8
///   15  → B_L1_Bi_8x16
///   16  → B_Bi_L0_16x8
///   17  → B_Bi_L0_8x16
///   18  → B_Bi_L1_16x8
///   19  → B_Bi_L1_8x16
///   20  → B_Bi_Bi_16x8
///   21  → B_Bi_Bi_8x16
///   22  → B_8x8
///   23..48 → intra (offset -23 into Table 7-11)
pub fn decode_b_slice_mb_type(mb_type: u32) -> Option<BMbType> {
    use BPartition::*;
    use PredDir::*;
    Some(match mb_type {
        0 => BMbType::Inter {
            partition: Direct16x16,
        },
        1 => BMbType::Inter {
            partition: L0_16x16,
        },
        2 => BMbType::Inter {
            partition: L1_16x16,
        },
        3 => BMbType::Inter {
            partition: Bi_16x16,
        },
        // 16x8 pairs: columns are top then bottom.
        4 => BMbType::Inter {
            partition: TwoPart16x8 { dirs: [L0, L0] },
        },
        5 => BMbType::Inter {
            partition: TwoPart8x16 { dirs: [L0, L0] },
        },
        6 => BMbType::Inter {
            partition: TwoPart16x8 { dirs: [L1, L1] },
        },
        7 => BMbType::Inter {
            partition: TwoPart8x16 { dirs: [L1, L1] },
        },
        8 => BMbType::Inter {
            partition: TwoPart16x8 { dirs: [L0, L1] },
        },
        9 => BMbType::Inter {
            partition: TwoPart8x16 { dirs: [L0, L1] },
        },
        10 => BMbType::Inter {
            partition: TwoPart16x8 { dirs: [L1, L0] },
        },
        11 => BMbType::Inter {
            partition: TwoPart8x16 { dirs: [L1, L0] },
        },
        12 => BMbType::Inter {
            partition: TwoPart16x8 { dirs: [L0, BiPred] },
        },
        13 => BMbType::Inter {
            partition: TwoPart8x16 { dirs: [L0, BiPred] },
        },
        14 => BMbType::Inter {
            partition: TwoPart16x8 { dirs: [L1, BiPred] },
        },
        15 => BMbType::Inter {
            partition: TwoPart8x16 { dirs: [L1, BiPred] },
        },
        16 => BMbType::Inter {
            partition: TwoPart16x8 { dirs: [BiPred, L0] },
        },
        17 => BMbType::Inter {
            partition: TwoPart8x16 { dirs: [BiPred, L0] },
        },
        18 => BMbType::Inter {
            partition: TwoPart16x8 { dirs: [BiPred, L1] },
        },
        19 => BMbType::Inter {
            partition: TwoPart8x16 { dirs: [BiPred, L1] },
        },
        20 => BMbType::Inter {
            partition: TwoPart16x8 {
                dirs: [BiPred, BiPred],
            },
        },
        21 => BMbType::Inter {
            partition: TwoPart8x16 {
                dirs: [BiPred, BiPred],
            },
        },
        22 => BMbType::Inter { partition: B8x8 },
        n if (23..=48).contains(&n) => {
            return decode_i_slice_mb_type(n - 23).map(BMbType::IntraInB);
        }
        _ => return None,
    })
}

/// B-slice sub-partition kind for one of the four 8×8 sub-MBs inside a
/// `B_8x8` macroblock (§7.4.5.2 Table 7-17). Covers entries 0–12.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BSubPartition {
    /// `B_Direct_8x8` — §8.4.1.2 direct-mode MVs; no MVD in the bitstream.
    Direct8x8,
    /// `B_L0_8x8` / `B_L1_8x8` / `B_Bi_8x8` — whole 8×8 coded with one dir.
    Full8x8 {
        dir: PredDir,
    },
    /// 8×4 or 4×8 pair; each half has the same prediction direction.
    TwoPart8x4 {
        dir: PredDir,
    },
    TwoPart4x8 {
        dir: PredDir,
    },
    /// Four 4×4 sub-partitions, all with the same prediction direction.
    Four4x4 {
        dir: PredDir,
    },
}

impl BSubPartition {
    pub fn num_sub_partitions(self) -> usize {
        match self {
            BSubPartition::Direct8x8 | BSubPartition::Full8x8 { .. } => 1,
            BSubPartition::TwoPart8x4 { .. } | BSubPartition::TwoPart4x8 { .. } => 2,
            BSubPartition::Four4x4 { .. } => 4,
        }
    }
    /// Prediction direction, or `None` for `Direct8x8` (callers derive MVs
    /// via direct prediction rather than following a partition-level dir).
    pub fn pred_dir(self) -> Option<PredDir> {
        match self {
            BSubPartition::Direct8x8 => None,
            BSubPartition::Full8x8 { dir }
            | BSubPartition::TwoPart8x4 { dir }
            | BSubPartition::TwoPart4x8 { dir }
            | BSubPartition::Four4x4 { dir } => Some(dir),
        }
    }
    /// Sub-rectangle in 4×4-block units, relative to the parent 8×8 block.
    pub fn sub_rect(self, idx: usize) -> (usize, usize, usize, usize) {
        match self {
            BSubPartition::Direct8x8 | BSubPartition::Full8x8 { .. } => (0, 0, 2, 2),
            BSubPartition::TwoPart8x4 { .. } => match idx {
                0 => (0, 0, 1, 2),
                _ => (1, 0, 1, 2),
            },
            BSubPartition::TwoPart4x8 { .. } => match idx {
                0 => (0, 0, 2, 1),
                _ => (0, 1, 2, 1),
            },
            BSubPartition::Four4x4 { .. } => match idx {
                0 => (0, 0, 1, 1),
                1 => (0, 1, 1, 1),
                2 => (1, 0, 1, 1),
                _ => (1, 1, 1, 1),
            },
        }
    }
}

/// Decode a B-slice `sub_mb_type` per Table 7-17 (B entries: values 0–12).
pub fn decode_b_sub_mb_type(v: u32) -> Option<BSubPartition> {
    use PredDir::*;
    Some(match v {
        0 => BSubPartition::Direct8x8,
        1 => BSubPartition::Full8x8 { dir: L0 },
        2 => BSubPartition::Full8x8 { dir: L1 },
        3 => BSubPartition::Full8x8 { dir: BiPred },
        4 => BSubPartition::TwoPart8x4 { dir: L0 },
        5 => BSubPartition::TwoPart4x8 { dir: L0 },
        6 => BSubPartition::TwoPart8x4 { dir: L1 },
        7 => BSubPartition::TwoPart4x8 { dir: L1 },
        8 => BSubPartition::TwoPart8x4 { dir: BiPred },
        9 => BSubPartition::TwoPart4x8 { dir: BiPred },
        10 => BSubPartition::Four4x4 { dir: L0 },
        11 => BSubPartition::Four4x4 { dir: L1 },
        12 => BSubPartition::Four4x4 { dir: BiPred },
        _ => return None,
    })
}

/// Decode a P-slice `coded_block_pattern` me(v) value (Table 9-4(b) — inter
/// column). Returns `(cbp_luma, cbp_chroma)`.
pub const ME_INTER_4_2_0: [u8; 48] = [
    0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37, 42, 44, 33, 34,
    36, 40, 39, 43, 45, 46, 17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
];

pub fn decode_cbp_inter(me_value: u32) -> Option<(u8, u8)> {
    let idx = me_value as usize;
    if idx >= ME_INTER_4_2_0.len() {
        return None;
    }
    let v = ME_INTER_4_2_0[idx];
    Some((v & 0x0F, v >> 4))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p_mb_type_partitions() {
        assert!(matches!(
            decode_p_slice_mb_type(0),
            Some(PMbType::Inter {
                partition: PPartition::P16x16
            })
        ));
        assert!(matches!(
            decode_p_slice_mb_type(1),
            Some(PMbType::Inter {
                partition: PPartition::P16x8
            })
        ));
        assert!(matches!(
            decode_p_slice_mb_type(2),
            Some(PMbType::Inter {
                partition: PPartition::P8x16
            })
        ));
        assert!(matches!(
            decode_p_slice_mb_type(3),
            Some(PMbType::Inter {
                partition: PPartition::P8x8
            })
        ));
        assert!(matches!(
            decode_p_slice_mb_type(4),
            Some(PMbType::Inter {
                partition: PPartition::P8x8Ref0
            })
        ));
    }

    #[test]
    fn p_mb_type_intra_passthrough() {
        // mb_type == 5 → intra index 0 → I_NxN.
        assert!(matches!(
            decode_p_slice_mb_type(5),
            Some(PMbType::IntraInP(IMbType::INxN))
        ));
        // mb_type == 30 → intra index 25 → I_PCM.
        assert!(matches!(
            decode_p_slice_mb_type(30),
            Some(PMbType::IntraInP(IMbType::IPcm))
        ));
        assert!(decode_p_slice_mb_type(31).is_none());
    }

    #[test]
    fn p_cbp_roundtrip_samples() {
        // me_value 0 → cbp_luma=0, cbp_chroma=0. me_value 1 → 0,1.
        assert_eq!(decode_cbp_inter(0), Some((0, 0)));
        assert_eq!(decode_cbp_inter(1), Some((0, 1)));
        assert_eq!(decode_cbp_inter(2), Some((1, 0)));
    }

    #[test]
    fn mb_type_zero_is_inxn() {
        assert!(matches!(decode_i_slice_mb_type(0), Some(IMbType::INxN)));
    }

    #[test]
    fn mb_type_one() {
        // Per spec Table 7-11: mb_type 1 = I_16x16_0_0_0 — Intra16x16PredMode=0
        // (Vertical), CBPLuma=0, CBPChroma=0.
        match decode_i_slice_mb_type(1).unwrap() {
            IMbType::I16x16 {
                intra16x16_pred_mode,
                cbp_luma,
                cbp_chroma,
            } => {
                assert_eq!(intra16x16_pred_mode, 0);
                assert_eq!(cbp_luma, 0);
                assert_eq!(cbp_chroma, 0);
            }
            _ => panic!("wrong"),
        }
    }

    #[test]
    fn mb_type_dc_cbp_full() {
        // mb_type 23 (spec) = I_16x16_2_2_15: DC pred, CBPLuma=15, CBPChroma=2.
        match decode_i_slice_mb_type(23).unwrap() {
            IMbType::I16x16 {
                intra16x16_pred_mode,
                cbp_luma,
                cbp_chroma,
            } => {
                assert_eq!(intra16x16_pred_mode, 2, "pred should be DC");
                assert_eq!(cbp_luma, 15);
                assert_eq!(cbp_chroma, 2);
            }
            _ => panic!("wrong"),
        }
    }
}
