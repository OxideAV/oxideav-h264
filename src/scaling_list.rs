//! Scaling-list matrix resolution — ITU-T H.264 §7.4.2.1.1.1 /
//! §7.4.2.2.
//!
//! The spec distinguishes **eight** (or twelve for 4:4:4) scaling-list
//! slots, addressed per block category:
//!
//! | index | role                        |
//! |-------|-----------------------------|
//! | 0     | Intra-Y 4×4                 |
//! | 1     | Intra-Cb 4×4                |
//! | 2     | Intra-Cr 4×4                |
//! | 3     | Inter-Y 4×4                 |
//! | 4     | Inter-Cb 4×4                |
//! | 5     | Inter-Cr 4×4                |
//! | 6     | Intra-Y 8×8                 |
//! | 7     | Inter-Y 8×8                 |
//! | 8-11  | Cb / Cr 8×8 for 4:4:4       |
//!
//! Table 7-2 specifies how an absent matrix falls back:
//!
//! * Rule A — "use the appropriate default scaling matrix" (indices 0,
//!   3 for 4×4 and 6, 7 for 8×8 — the "first" entry in each triple).
//! * Rule B — "inherit from matrix `i-1`" (indices 1, 2, 4, 5, and the
//!   4:4:4 chroma-8×8 slots).
//!
//! PPS matrices override SPS matrices; SPS matrices override the
//! built-in defaults. `pic_scaling_matrix_present_flag = 1` means the
//! PPS's matrices are live AND the SPS's matrices are ignored outright
//! (§7.4.2.2).
//!
//! Default matrices (Tables 7-3 / 7-4) are carried in raster (row-
//! major) order.

use crate::pps::Pps;
use crate::sps::Sps;

/// 4×4 zig-zag scan order (frame coding) — Table 8-14. Maps linear
/// scanning position 0..=15 → raster (`row*4 + col`).
#[rustfmt::skip]
pub const ZIGZAG_4X4: [usize; 16] = [
    0,  1,  4,  8,
    5,  2,  3,  6,
    9, 12, 13, 10,
    7, 11, 14, 15,
];

/// Table 7-3 — `Default_4x4_Intra` in raster order.
#[rustfmt::skip]
pub const DEFAULT_4X4_INTRA: [i16; 16] = [
     6, 13, 13, 20,
    13, 20, 20, 28,
    13, 20, 20, 28,
    20, 28, 28, 32,
];

/// Table 7-3 — `Default_4x4_Inter` in raster order.
#[rustfmt::skip]
pub const DEFAULT_4X4_INTER: [i16; 16] = [
    10, 14, 14, 20,
    14, 20, 20, 24,
    14, 20, 20, 24,
    20, 24, 24, 28,
];

/// Table 7-4 — `Default_8x8_Intra` in raster order.
#[rustfmt::skip]
pub const DEFAULT_8X8_INTRA: [i16; 64] = [
     6, 10, 13, 16, 18, 23, 25, 27,
    10, 11, 16, 18, 23, 25, 27, 29,
    13, 16, 18, 23, 25, 27, 29, 31,
    16, 18, 23, 25, 27, 29, 31, 33,
    18, 23, 25, 27, 29, 31, 33, 36,
    23, 25, 27, 29, 31, 33, 36, 38,
    25, 27, 29, 31, 33, 36, 38, 40,
    27, 29, 31, 33, 36, 38, 40, 42,
];

/// Table 7-4 — `Default_8x8_Inter` in raster order.
#[rustfmt::skip]
pub const DEFAULT_8X8_INTER: [i16; 64] = [
     9, 13, 15, 17, 19, 21, 22, 24,
    13, 13, 17, 19, 21, 22, 24, 25,
    15, 17, 19, 21, 22, 24, 25, 27,
    17, 19, 21, 22, 24, 25, 27, 28,
    19, 21, 22, 24, 25, 27, 28, 30,
    21, 22, 24, 25, 27, 28, 30, 32,
    22, 24, 25, 27, 28, 30, 32, 33,
    24, 25, 27, 28, 30, 32, 33, 35,
];

/// Flat (all-16) scaling list. This is the "no scaling" matrix — both
/// SPS and PPS absent means every slot is 16 per §7.4.2.1.1.1.
pub const FLAT_4X4: [i16; 16] = [16; 16];
pub const FLAT_8X8: [i16; 64] = [16; 64];

/// Default 4×4 matrix for a given slot per Table 7-2 rule A.
/// `i` is the 0..=5 index into `ScalingList4x4[]`.
pub fn default_4x4_for_index(i: usize) -> [i16; 16] {
    // §7.4.2.1.1.1 groups the six 4×4 slots into two triples: indices
    // 0..=2 inherit Default_4x4_Intra, indices 3..=5 inherit
    // Default_4x4_Inter.
    if i < 3 {
        DEFAULT_4X4_INTRA
    } else {
        DEFAULT_4X4_INTER
    }
}

/// Default 8×8 matrix for a given slot. `i = 0` → `Default_8x8_Intra`
/// (slots 6, and 4:4:4 chroma intra 8..=10), `i = 1` → `Default_8x8_Inter`.
pub fn default_8x8_for_index(i: usize) -> [i16; 64] {
    if i % 2 == 0 {
        DEFAULT_8X8_INTRA
    } else {
        DEFAULT_8X8_INTER
    }
}

/// Resolved scaling-list matrices for a slice. All six 4×4 and (up to
/// two) 8×8 matrices are pre-materialised in raster order so the
/// dequant path can index them without re-running the Table 7-2
/// fallback logic per block.
#[derive(Clone, Debug)]
pub struct ScalingLists {
    pub list_4x4: [[i16; 16]; 6],
    pub list_8x8: [[i16; 64]; 2],
    /// When `true` every 4×4 and 8×8 matrix equals [`FLAT_4X4`] /
    /// [`FLAT_8X8`]. The dequant path uses this to take a fast-path
    /// that avoids the per-position multiply-then-shift for each
    /// coefficient.
    pub all_flat: bool,
}

impl Default for ScalingLists {
    fn default() -> Self {
        Self {
            list_4x4: [FLAT_4X4; 6],
            list_8x8: [FLAT_8X8; 2],
            all_flat: true,
        }
    }
}

impl ScalingLists {
    /// Flat (identity) scaling lists — the "no scaling list signalled"
    /// baseline.
    pub fn flat() -> Self {
        Self::default()
    }

    /// Resolve the active scaling lists for the current slice per
    /// §7.4.2.2 Table 7-2.
    ///
    /// Precedence:
    /// 1. If `pic_scaling_matrix_present_flag = 1` (High Profile PPS),
    ///    the SPS's scaling-list set is ignored — only the PPS's
    ///    matrices feed the fallback.
    /// 2. Otherwise, the SPS's `seq_scaling_matrix_present_flag = 1`
    ///    matrices feed the fallback.
    /// 3. In either case, a missing matrix at index `i` inherits
    ///    per Table 7-2: rule A (defaults) for indices 0, 3 (4×4) and
    ///    0 (8×8); rule B (copy from `i-1`) for 1, 2, 4, 5 (4×4).
    pub fn resolve(sps: &Sps, pps: &Pps) -> Self {
        // Nothing custom signalled → flat scaling for every slot. This
        // is the overwhelming-common case (most x264 output) and the
        // fast path below relies on `all_flat`.
        if !sps.seq_scaling_matrix_present_flag && !pps.pic_scaling_matrix_present_flag {
            return Self::flat();
        }

        // Pick the source set per §7.4.2.2: PPS overrides SPS.
        let pic_set = if pps.pic_scaling_matrix_present_flag {
            Some((&pps.pic_scaling_list_4x4, &pps.pic_scaling_list_8x8))
        } else {
            None
        };
        let seq_set = if sps.seq_scaling_matrix_present_flag {
            Some((&sps.seq_scaling_list_4x4, &sps.seq_scaling_list_8x8))
        } else {
            None
        };

        let mut list_4x4 = [FLAT_4X4; 6];
        let mut list_8x8 = [FLAT_8X8; 2];
        let mut all_flat = true;

        for i in 0..6usize {
            // §7.4.2.2 — PPS presence takes precedence; if absent,
            // look at SPS. If both absent, apply Table 7-2 fallback.
            let custom: Option<[i16; 16]> = pic_set
                .and_then(|(m4, _)| m4[i])
                .or_else(|| seq_set.and_then(|(m4, _)| m4[i]));
            let m = match custom {
                Some(m) => m,
                None => {
                    // Rule A for i ∈ {0, 3}, rule B for others.
                    if i == 0 || i == 3 {
                        default_4x4_for_index(i)
                    } else {
                        list_4x4[i - 1]
                    }
                }
            };
            if m != FLAT_4X4 {
                all_flat = false;
            }
            list_4x4[i] = m;
        }

        // 8×8 slots 0 (intra-Y) and 1 (inter-Y).
        for i in 0..2usize {
            let custom: Option<[i16; 64]> = pic_set
                .and_then(|(_, m8)| m8[i])
                .or_else(|| seq_set.and_then(|(_, m8)| m8[i]));
            let m = match custom {
                Some(m) => m,
                None => {
                    // Rule A for i == 0 (intra-Y). i == 1 inherits
                    // from i - 1 (= intra-Y 8×8) per Table 7-2.
                    if i == 0 {
                        default_8x8_for_index(0)
                    } else {
                        list_8x8[i - 1]
                    }
                }
            };
            if m != FLAT_8X8 {
                all_flat = false;
            }
            list_8x8[i] = m;
        }

        Self {
            list_4x4,
            list_8x8,
            all_flat,
        }
    }

    /// Pick the active 4×4 matrix for a block at the given category.
    /// `category`:
    /// - 0 = Intra-Y, 1 = Intra-Cb, 2 = Intra-Cr,
    /// - 3 = Inter-Y, 4 = Inter-Cb, 5 = Inter-Cr.
    pub fn matrix_4x4(&self, category: usize) -> &[i16; 16] {
        &self.list_4x4[category.min(5)]
    }

    /// Pick the active 8×8 matrix. `category`: 0 = intra-Y, 1 = inter-Y.
    pub fn matrix_8x8(&self, category: usize) -> &[i16; 64] {
        &self.list_8x8[category.min(1)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// When neither SPS nor PPS signal a matrix, every list is flat
    /// and `all_flat` is `true`.
    #[test]
    fn resolve_flat_when_no_matrices() {
        let sps = make_sps(false, [None; 6], [None; 6]);
        let pps = make_pps(false, [None; 6], [None; 6]);
        let r = ScalingLists::resolve(&sps, &pps);
        assert!(r.all_flat);
        for i in 0..6 {
            assert_eq!(r.matrix_4x4(i), &FLAT_4X4);
        }
    }

    /// Rule A: missing matrix at index 0 falls back to Default_4x4_Intra.
    #[test]
    fn resolve_rule_a_falls_back_to_default() {
        let sps = make_sps(true, [None; 6], [None; 6]);
        let pps = make_pps(false, [None; 6], [None; 6]);
        let r = ScalingLists::resolve(&sps, &pps);
        assert_eq!(r.matrix_4x4(0), &DEFAULT_4X4_INTRA);
        assert_eq!(r.matrix_4x4(3), &DEFAULT_4X4_INTER);
        assert!(!r.all_flat);
    }

    /// Rule B: missing at index 1 inherits from index 0.
    #[test]
    fn resolve_rule_b_inherits_previous() {
        let custom = [17i16; 16];
        let mut sps_m = [None; 6];
        sps_m[0] = Some(custom);
        let sps = make_sps(true, sps_m, [None; 6]);
        let pps = make_pps(false, [None; 6], [None; 6]);
        let r = ScalingLists::resolve(&sps, &pps);
        assert_eq!(r.matrix_4x4(0), &custom);
        // Slot 1 missing → inherit from slot 0 (rule B).
        assert_eq!(r.matrix_4x4(1), &custom);
        // Slot 2 also missing → inherit from slot 1 → still the same matrix.
        assert_eq!(r.matrix_4x4(2), &custom);
        // Slot 3 missing → rule A → Default_4x4_Inter.
        assert_eq!(r.matrix_4x4(3), &DEFAULT_4X4_INTER);
    }

    /// PPS matrices override SPS matrices at the same index.
    #[test]
    fn pps_overrides_sps() {
        let m_sps = [11i16; 16];
        let m_pps = [22i16; 16];
        let mut sps_m = [None; 6];
        sps_m[0] = Some(m_sps);
        let mut pps_m = [None; 6];
        pps_m[0] = Some(m_pps);
        let sps = make_sps(true, sps_m, [None; 6]);
        let pps = make_pps(true, pps_m, [None; 6]);
        let r = ScalingLists::resolve(&sps, &pps);
        assert_eq!(r.matrix_4x4(0), &m_pps);
    }

    fn make_sps(
        present: bool,
        list_4x4: [Option<[i16; 16]>; 6],
        list_8x8: [Option<[i16; 64]>; 6],
    ) -> Sps {
        Sps {
            profile_idc: 100,
            constraint_set0_flag: false,
            constraint_set1_flag: false,
            constraint_set2_flag: false,
            constraint_set3_flag: false,
            constraint_set4_flag: false,
            constraint_set5_flag: false,
            level_idc: 30,
            seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            qpprime_y_zero_transform_bypass_flag: false,
            seq_scaling_matrix_present_flag: present,
            seq_scaling_list_4x4: list_4x4,
            seq_scaling_list_8x8: list_8x8,
            log2_max_frame_num_minus4: 0,
            pic_order_cnt_type: 2,
            log2_max_pic_order_cnt_lsb_minus4: 0,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            offset_for_ref_frame: Vec::new(),
            max_num_ref_frames: 1,
            gaps_in_frame_num_value_allowed_flag: false,
            pic_width_in_mbs_minus1: 7,
            pic_height_in_map_units_minus1: 5,
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: true,
            frame_cropping_flag: false,
            frame_crop_left_offset: 0,
            frame_crop_right_offset: 0,
            frame_crop_top_offset: 0,
            frame_crop_bottom_offset: 0,
            vui_parameters_present_flag: false,
            vui: None,
        }
    }

    fn make_pps(
        present: bool,
        list_4x4: [Option<[i16; 16]>; 6],
        list_8x8: [Option<[i16; 64]>; 6],
    ) -> Pps {
        Pps {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
            entropy_coding_mode_flag: false,
            bottom_field_pic_order_in_frame_present_flag: false,
            num_slice_groups_minus1: 0,
            slice_group_map_type: 0,
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
            transform_8x8_mode_flag: false,
            pic_scaling_matrix_present_flag: present,
            pic_scaling_list_4x4: list_4x4,
            pic_scaling_list_8x8: list_8x8,
            second_chroma_qp_index_offset: 0,
        }
    }
}
