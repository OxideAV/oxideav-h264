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
/// The mapping table is taken from FFmpeg's `i_mb_type_info` (libavcodec
/// h264data.h, originally derived from H.264 §7.4.5 Table 7-11). The
/// `intra16x16_pred_mode` values use the FFmpeg numbering where
/// `0=Vertical`, `1=Horizontal`, `2=DC`, `3=Plane` — same as ours.
pub fn decode_i_slice_mb_type(mb_type: u32) -> Option<IMbType> {
    match mb_type {
        0 => Some(IMbType::INxN),
        25 => Some(IMbType::IPcm),
        n if (1..=24).contains(&n) => {
            // Per FFmpeg's i_mb_type_info[1..=24] table.
            // (pred_mode, cbp_chroma, cbp_luma) for each entry:
            const TABLE: [(u8, u8, u8); 24] = [
                (2, 0, 0), // mb_type 1: DC, cbp_chroma=0, cbp_luma=0
                (1, 0, 0), // 2: Horizontal
                (0, 0, 0), // 3: Vertical
                (3, 0, 0), // 4: Plane
                (2, 1, 0), // 5: DC, cbp_chroma=1, cbp_luma=0
                (1, 1, 0),
                (0, 1, 0),
                (3, 1, 0),
                (2, 2, 0), // 9: DC, cbp_chroma=2, cbp_luma=0
                (1, 2, 0),
                (0, 2, 0),
                (3, 2, 0),
                (2, 0, 15), // 13: DC, cbp_chroma=0, cbp_luma=15
                (1, 0, 15),
                (0, 0, 15),
                (3, 0, 15),
                (2, 1, 15), // 17: DC, cbp_chroma=1, cbp_luma=15
                (1, 1, 15),
                (0, 1, 15),
                (3, 1, 15),
                (2, 2, 15), // 21: DC, cbp_chroma=2, cbp_luma=15
                (1, 2, 15),
                (0, 2, 15),
                (3, 2, 15),
            ];
            let (pred, cbp_chroma, cbp_luma) = TABLE[(n - 1) as usize];
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
    Inter {
        partition: PPartition,
    },
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
        // mb_type 1 -> I_16x16, DC mode (per FFmpeg's table), cbp=0.
        // (FFmpeg's "pred_mode" numbering treats 0/1/2/3 as Vert/Hor/DC/Plane;
        // the spec mb_type 1 maps to internal pred_mode = 2 = DC.)
        match decode_i_slice_mb_type(1).unwrap() {
            IMbType::I16x16 {
                intra16x16_pred_mode,
                cbp_luma,
                cbp_chroma,
            } => {
                assert_eq!(intra16x16_pred_mode, 2);
                assert_eq!(cbp_luma, 0);
                assert_eq!(cbp_chroma, 0);
            }
            _ => panic!("wrong"),
        }
    }
}
