//! §7.3.2.2 — encoder for `pic_parameter_set_rbsp()`.
//!
//! Round-1 minimal PPS for the Baseline I_16x16 path:
//!
//! * `entropy_coding_mode_flag = 0` (CAVLC)
//! * `num_slice_groups_minus1 = 0` (no FMO)
//! * `weighted_pred_flag = 0`, `weighted_bipred_idc = 0`
//! * `deblocking_filter_control_present_flag = 1` (per-slice control)
//! * `constrained_intra_pred_flag = 0`
//! * `redundant_pic_cnt_present_flag = 0`
//! * No optional tail (`transform_8x8_mode_flag` etc. inferred to 0).
//!
//! Round-26: `weighted_pred_flag` and `weighted_bipred_idc` are now
//! caller-configurable. Setting `weighted_bipred_idc = 1` enables the
//! per-slice `pred_weight_table()` syntax (§7.3.3.2) for B-slices and
//! makes the decoder use §8.4.2.3.2 explicit weighted prediction. The
//! encoder pairs this with a least-squares weight selector (see
//! [`crate::encoder::compute_explicit_bipred_weights`]) and a matching
//! merge in `build_b_predictors` to keep the local recon bit-equivalent
//! to what the decoder will produce.

use crate::encoder::bitstream::BitWriter;

/// Configuration for [`build_baseline_pps_rbsp`].
#[derive(Debug, Clone, Default)]
pub struct BaselinePpsConfig {
    pub pic_parameter_set_id: u32,
    pub seq_parameter_set_id: u32,
    /// `pic_init_qp_minus26` per §7.4.2.2. The actual slice-level QP is
    /// further offset by `slice_qp_delta` in the slice header. Default
    /// is 0 → QP = 26.
    pub pic_init_qp_minus26: i32,
    /// `chroma_qp_index_offset` per §7.4.2.2. Range -12..=12.
    pub chroma_qp_index_offset: i32,
    /// §7.4.2.2 — `weighted_pred_flag` (P / SP slices). When `true`,
    /// the encoder ships a `pred_weight_table()` per P slice and the
    /// decoder applies §8.4.2.3.2 explicit weighted prediction. Round-26
    /// only wires explicit weighted bipred (B slices), so this flag is
    /// usually `false`.
    pub weighted_pred_flag: bool,
    /// §7.4.2.2 — `weighted_bipred_idc` ∈ 0..=2. 0 = default weighted
    /// bipred (`(L0 + L1 + 1) >> 1`), 1 = explicit (round-26),
    /// 2 = implicit (POC-distance derivation, not yet wired). Range
    /// validated by the decoder's PPS parser.
    pub weighted_bipred_idc: u32,
    /// Round-30 — §7.4.2.2 `entropy_coding_mode_flag`. `false` (default)
    /// signals CAVLC; `true` signals CABAC. Baseline profile (66) does
    /// NOT permit CABAC per §A.2.1; the encoder must bump
    /// `profile_idc` to Main (77) or higher before enabling this.
    pub entropy_coding_mode_flag: bool,
    /// §7.4.2.2 — `transform_8x8_mode_flag`. When `true`, the PPS carries
    /// the optional trailing group (`transform_8x8_mode_flag = 1`,
    /// `pic_scaling_matrix_present_flag`, `second_chroma_qp_index_offset`)
    /// so the decoder reads a per-MB `transform_size_8x8_flag` for I_NxN
    /// macroblocks and accepts Intra_8x8 / 8x8-transform residuals. Only
    /// permitted in High profile (100) and above per §A.2.4;
    /// `second_chroma_qp_index_offset` mirrors `chroma_qp_index_offset`.
    pub transform_8x8_mode_flag: bool,
    /// §7.3.2.2 / §7.3.2.1.1.1 — when `Some`, the optional PPS tail
    /// carries `pic_scaling_matrix_present_flag = 1` with every list
    /// present (6 lists, + 2 8x8 lists when `transform_8x8_mode_flag`
    /// is set, per the §7.3.2.2 loop bound).
    /// `ScalingListsSpec::Default` codes each list as
    /// UseDefaultScalingMatrixFlag (delta_scale = -8), selecting the
    /// Table 7-3 / Table 7-4 default matrices at picture level;
    /// `ScalingListsSpec::Custom` codes the caller's values explicitly
    /// (round-391).
    pub pic_scaling_lists: Option<super::ScalingListsSpec>,
    /// §7.3.2.2 — the active SPS's `chroma_format_idc`. Only consulted
    /// for the scaling-list loop bound (4:4:4 carries 6 extra 8x8
    /// lists when `transform_8x8_mode_flag = 1`). Default 1 (4:2:0).
    pub chroma_format_idc: u32,
    /// Round-451 — §7.4.2.2 `redundant_pic_cnt_present_flag`. When
    /// `true` every slice header referencing this PPS carries a
    /// §7.3.3 `redundant_pic_cnt` (used by the redundant-coded-picture
    /// gates; ordinary streams leave it `false`).
    pub redundant_pic_cnt_present_flag: bool,
    /// Round-453 — §7.3.2.2 flexible macroblock ordering: `Some(map)`
    /// codes `num_slice_groups_minus1 = num_slice_groups - 1` and the
    /// `slice_group_map_type` 0..=6 parameters (the decoder's own
    /// [`crate::pps::SliceGroupMap`] shape, so the encoder derives the
    /// §8.2.2 map with the decoder's code); `None` = one slice group.
    pub slice_groups: Option<(u32, crate::pps::SliceGroupMap)>,
    /// Round-453 — §7.4.2.2 `constrained_intra_pred_flag`: intra
    /// macroblocks in P slices predict only from intra neighbours.
    pub constrained_intra_pred_flag: bool,
}

/// Build a Baseline PPS RBSP body (§7.3.2.2).
pub fn build_baseline_pps_rbsp(cfg: &BaselinePpsConfig) -> Vec<u8> {
    let mut w = BitWriter::new();

    w.ue(cfg.pic_parameter_set_id);
    w.ue(cfg.seq_parameter_set_id);

    // §7.4.2.2 — entropy_coding_mode_flag. 0 = CAVLC, 1 = CABAC.
    w.u(1, if cfg.entropy_coding_mode_flag { 1 } else { 0 });
    // bottom_field_pic_order_in_frame_present_flag = 0.
    w.u(1, 0);
    // §7.3.2.2 — num_slice_groups_minus1 + slice group map (FMO,
    // round-453); 0 = single slice group.
    match &cfg.slice_groups {
        None => w.ue(0),
        Some((num_groups, map)) => {
            use crate::pps::SliceGroupMap;
            debug_assert!((2..=8).contains(num_groups));
            let n_minus1 = num_groups - 1;
            w.ue(n_minus1);
            match map {
                SliceGroupMap::Interleaved { run_length_minus1 } => {
                    w.ue(0);
                    debug_assert_eq!(run_length_minus1.len() as u32, n_minus1 + 1);
                    for &r in run_length_minus1 {
                        w.ue(r);
                    }
                }
                SliceGroupMap::Dispersed => w.ue(1),
                SliceGroupMap::Foreground {
                    top_left,
                    bottom_right,
                } => {
                    w.ue(2);
                    debug_assert_eq!(top_left.len() as u32, n_minus1);
                    debug_assert_eq!(bottom_right.len() as u32, n_minus1);
                    for (&tl, &br) in top_left.iter().zip(bottom_right.iter()) {
                        w.ue(tl);
                        w.ue(br);
                    }
                }
                SliceGroupMap::Changing {
                    slice_group_map_type,
                    change_direction_flag,
                    change_rate_minus1,
                } => {
                    debug_assert!((3..=5).contains(slice_group_map_type));
                    w.ue(*slice_group_map_type);
                    w.u(1, u32::from(*change_direction_flag));
                    w.ue(*change_rate_minus1);
                }
                SliceGroupMap::Explicit {
                    pic_size_in_map_units_minus1,
                    slice_group_id,
                } => {
                    w.ue(6);
                    w.ue(*pic_size_in_map_units_minus1);
                    debug_assert_eq!(
                        slice_group_id.len() as u32,
                        pic_size_in_map_units_minus1 + 1
                    );
                    // §7.4.2.2 — slice_group_id[i] is u(v) with
                    // v = Ceil(Log2(num_slice_groups_minus1 + 1)).
                    let v = 32 - (n_minus1).leading_zeros();
                    for &id in slice_group_id {
                        debug_assert!(id <= n_minus1);
                        w.u(v, id);
                    }
                }
            }
        }
    }

    // num_ref_idx_l0_default_active_minus1 = 0, ditto l1.
    w.ue(0);
    w.ue(0);

    // §7.4.2.2 — weighted_pred_flag (1 bit) + weighted_bipred_idc (2 bits).
    debug_assert!(cfg.weighted_bipred_idc <= 2);
    w.u(1, if cfg.weighted_pred_flag { 1 } else { 0 });
    w.u(2, cfg.weighted_bipred_idc);

    w.se(cfg.pic_init_qp_minus26);
    w.se(0); // pic_init_qs_minus26
    w.se(cfg.chroma_qp_index_offset);

    // deblocking_filter_control_present_flag = 1 — lets the slice
    // header carry `disable_deblocking_filter_idc` and the offsets.
    w.u(1, 1);
    // §7.4.2.2 — constrained_intra_pred_flag (round-453).
    w.u(1, u32::from(cfg.constrained_intra_pred_flag));
    // §7.4.2.2 — redundant_pic_cnt_present_flag (round-451:
    // caller-selectable for the redundant-coded-picture gates).
    w.u(1, u32::from(cfg.redundant_pic_cnt_present_flag));

    // §7.3.2.2 — optional trailing group. Only emitted when a High-profile
    // feature (8x8 transform / picture scaling matrices) needs it;
    // otherwise the decoder infers transform_8x8_mode_flag = 0 and
    // flat picture lists from the absent tail.
    if cfg.transform_8x8_mode_flag || cfg.pic_scaling_lists.is_some() {
        w.u(1, if cfg.transform_8x8_mode_flag { 1 } else { 0 });
        if let Some(spec) = &cfg.pic_scaling_lists {
            w.u(1, 1); // pic_scaling_matrix_present_flag
                       // §7.3.2.2 loop bound: 6 + [2, 6] * transform_8x8_mode_flag
                       // (4:2:0/4:2:2 → 2 extra 8x8 lists).
                       // §7.3.2.2 loop bound:
                       //   6 + ((chroma_format_idc != 3) ? 2 : 6) * transform_8x8_mode_flag
            let n_lists = 6 + if !cfg.transform_8x8_mode_flag {
                0
            } else if cfg.chroma_format_idc != 3 {
                2
            } else {
                6
            };
            for i in 0..n_lists {
                w.u(1, 1); // pic_scaling_list_present_flag[i]
                crate::encoder::sps::write_scaling_list_slot(&mut w, spec, i);
            }
        } else {
            w.u(1, 0); // pic_scaling_matrix_present_flag = 0
        }
        w.se(cfg.chroma_qp_index_offset); // second_chroma_qp_index_offset
    }

    w.rbsp_trailing_bits();
    w.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pps::Pps;

    #[test]
    fn baseline_pps_round_trips_through_decoder_parser() {
        let cfg = BaselinePpsConfig {
            redundant_pic_cnt_present_flag: false,
            slice_groups: None,
            constrained_intra_pred_flag: false,
            pic_scaling_lists: None,
            chroma_format_idc: 1,
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
            pic_init_qp_minus26: 0,
            chroma_qp_index_offset: 0,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            entropy_coding_mode_flag: false,
            transform_8x8_mode_flag: false,
        };
        let rbsp = build_baseline_pps_rbsp(&cfg);
        let pps = Pps::parse(&rbsp).expect("decoder parses our PPS");
        assert_eq!(pps.pic_parameter_set_id, 0);
        assert_eq!(pps.seq_parameter_set_id, 0);
        assert!(!pps.entropy_coding_mode_flag);
        assert!(!pps.bottom_field_pic_order_in_frame_present_flag);
        assert_eq!(pps.num_slice_groups_minus1, 0);
        assert_eq!(pps.num_ref_idx_l0_default_active_minus1, 0);
        assert!(!pps.weighted_pred_flag);
        assert_eq!(pps.weighted_bipred_idc, 0);
        assert_eq!(pps.pic_init_qp_minus26, 0);
        assert_eq!(pps.chroma_qp_index_offset, 0);
        assert!(pps.deblocking_filter_control_present_flag);
        assert!(!pps.constrained_intra_pred_flag);
        assert!(!pps.redundant_pic_cnt_present_flag);
        assert!(pps.extension.is_none());
    }
}
