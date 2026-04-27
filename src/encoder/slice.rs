//! §7.3.3 — slice_header emit (encoder side).
//!
//! Round-14: in-loop deblocking is enabled by default
//! (`disable_deblocking_filter_idc=0`). The PPS's
//! `deblocking_filter_control_present_flag=1` lets the slice header
//! drive the loop-filter state. Round-1..13 emitted
//! `disable_deblocking_filter_idc=1` to keep the encoder simple; we now
//! mirror what the decoder produces so the encoder's local recon stays
//! aligned with what downstream decoders see.
//!
//! Round-16: adds P-slice header writer
//! ([`write_p_slice_header`]) for non-IDR P-slices. Single-reference
//! L0-only profile: `num_ref_idx_active_override_flag = 0`,
//! `ref_pic_list_modification` empty, `dec_ref_pic_marking` non-IDR
//! "use defaults" path.

use crate::encoder::bitstream::BitWriter;

/// Configuration for [`build_idr_i_slice_header_rbsp`]. Holds only the
/// slice-header fields. Caller appends `slice_data()` and trailing bits.
#[derive(Debug, Clone, Copy)]
pub struct IdrSliceHeaderConfig {
    pub first_mb_in_slice: u32,
    /// Raw `slice_type` value. For an "all-I, all slices same type"
    /// picture use 7 (`slice_type % 5 == 2` and `slice_type >= 5`).
    pub slice_type_raw: u32,
    pub pic_parameter_set_id: u32,
    /// `frame_num` width is `log2_max_frame_num_minus4 + 4`. We supply
    /// the value, the bit width is derived by the caller.
    pub frame_num: u32,
    pub frame_num_bits: u32,
    pub idr_pic_id: u32,
    /// `pic_order_cnt_lsb` width is `log2_max_pic_order_cnt_lsb_minus4 + 4`.
    pub pic_order_cnt_lsb: u32,
    pub poc_lsb_bits: u32,
    /// `slice_qp_delta` per §7.4.3.
    pub slice_qp_delta: i32,
    /// §7.4.3 — `disable_deblocking_filter_idc` ∈ {0, 1, 2}. 0 = filter
    /// across slice boundaries, 1 = no filtering, 2 = filter except on
    /// slice boundaries. Our single-slice encoder uses 0.
    pub disable_deblocking_filter_idc: u32,
    /// §7.4.3 — `slice_alpha_c0_offset_div2` ∈ -6..=6. Only emitted when
    /// `disable_deblocking_filter_idc != 1`.
    pub slice_alpha_c0_offset_div2: i32,
    /// §7.4.3 — `slice_beta_offset_div2` ∈ -6..=6. Only emitted when
    /// `disable_deblocking_filter_idc != 1`.
    pub slice_beta_offset_div2: i32,
}

/// Emit the bits of a single IDR I-slice header into the supplied
/// `BitWriter`. The writer is left at the position where `slice_data()`
/// should begin.
pub fn write_idr_i_slice_header(w: &mut BitWriter, cfg: &IdrSliceHeaderConfig) {
    w.ue(cfg.first_mb_in_slice);
    w.ue(cfg.slice_type_raw);
    w.ue(cfg.pic_parameter_set_id);
    // No `colour_plane_id` (separate_colour_plane_flag == 0 in our SPS).
    w.u(cfg.frame_num_bits, cfg.frame_num);
    // No field_pic_flag / bottom_field_flag — frame_mbs_only_flag == 1.
    // idr_pic_id only on IDR — and we always emit IDR here.
    w.ue(cfg.idr_pic_id);
    // POC type 0 branch: pic_order_cnt_lsb. No delta_pic_order_cnt_bottom
    // (bottom_field_pic_order_in_frame_present_flag == 0 in our PPS).
    w.u(cfg.poc_lsb_bits, cfg.pic_order_cnt_lsb);
    // I-slice: no num_ref_idx_active_override_flag,
    // no ref_pic_list_modification, no pred_weight_table.

    // §7.3.3.3 — dec_ref_pic_marking() (nal_ref_idc != 0 always for IDR).
    // IDR path: no_output_of_prior_pics_flag = 0, long_term_reference_flag = 0.
    w.u(1, 0);
    w.u(1, 0);

    // No cabac_init_idc — entropy_coding_mode_flag == 0 (CAVLC).
    w.se(cfg.slice_qp_delta);
    // No sp_for_switch_flag, slice_qs_delta — not SP/SI.

    // §7.3.3 — deblocking_filter_control_present_flag == 1 in our PPS,
    // so we always emit `disable_deblocking_filter_idc`. When the IDC
    // is not 1 (i.e. some form of filtering is enabled), the alpha-c0
    // and beta offsets follow.
    debug_assert!(cfg.disable_deblocking_filter_idc <= 2);
    debug_assert!((-6..=6).contains(&cfg.slice_alpha_c0_offset_div2));
    debug_assert!((-6..=6).contains(&cfg.slice_beta_offset_div2));
    w.ue(cfg.disable_deblocking_filter_idc);
    if cfg.disable_deblocking_filter_idc != 1 {
        w.se(cfg.slice_alpha_c0_offset_div2);
        w.se(cfg.slice_beta_offset_div2);
    }
}

// ---------------------------------------------------------------------------
// Round-16 — P-slice header (single L0 ref, no list modification).
// ---------------------------------------------------------------------------

/// Configuration for [`write_p_slice_header`]. Holds only the slice-header
/// fields. Caller appends `slice_data()` and trailing bits.
///
/// This writer covers the **simple non-IDR P-slice** path:
///
/// * `nal_unit_type = 1` (Slice non-IDR)
/// * `slice_type` = 5 (P, all slices same type) per Table 7-6
/// * `num_ref_idx_active_override_flag = 0` (use PPS default; for our
///   PPS that's `num_ref_idx_l0_default_active_minus1 = 0` ⇒ one L0 ref)
/// * `ref_pic_list_modification`: list-0 modification flag = 0
///   (no modifications). No list-1 (P-slice).
/// * `dec_ref_pic_marking` with `adaptive_ref_pic_marking_mode_flag = 0`
///   (sliding-window default — works for our 1-ref short-term setup).
/// * No `pred_weight_table` (PPS `weighted_pred_flag = 0`).
#[derive(Debug, Clone, Copy)]
pub struct PSliceHeaderConfig {
    pub first_mb_in_slice: u32,
    /// Raw `slice_type` per Table 7-6. Use 5 for "all slices in picture
    /// are P" (slice_type % 5 == 0 and >= 5).
    pub slice_type_raw: u32,
    pub pic_parameter_set_id: u32,
    pub frame_num: u32,
    pub frame_num_bits: u32,
    pub pic_order_cnt_lsb: u32,
    pub poc_lsb_bits: u32,
    pub slice_qp_delta: i32,
    pub disable_deblocking_filter_idc: u32,
    pub slice_alpha_c0_offset_div2: i32,
    pub slice_beta_offset_div2: i32,
    /// `nal_ref_idc` of the parent NAL — used to gate
    /// `dec_ref_pic_marking()` per §7.3.3 last branch.
    pub nal_ref_idc: u32,
}

/// Emit the bits of a single P-slice header (non-IDR) into the supplied
/// `BitWriter`. The writer is left at the position where `slice_data()`
/// should begin.
pub fn write_p_slice_header(w: &mut BitWriter, cfg: &PSliceHeaderConfig) {
    debug_assert!(cfg.slice_type_raw % 5 == 0); // P-slice
    w.ue(cfg.first_mb_in_slice);
    w.ue(cfg.slice_type_raw);
    w.ue(cfg.pic_parameter_set_id);
    // No `colour_plane_id` (separate_colour_plane_flag == 0).
    w.u(cfg.frame_num_bits, cfg.frame_num);
    // No field_pic_flag — frame_mbs_only_flag == 1.
    // No idr_pic_id — non-IDR.
    // POC type 0 branch.
    w.u(cfg.poc_lsb_bits, cfg.pic_order_cnt_lsb);
    // No delta_pic_order_cnt_bottom (PPS bottom_field_pic_order_in_frame_present_flag == 0).
    // No redundant_pic_cnt (PPS redundant_pic_cnt_present_flag == 0).
    // No direct_spatial_mv_pred_flag — P-slice (B only).

    // §7.3.3 — num_ref_idx_active_override_flag for P/SP/B slices.
    // We use the PPS default (num_ref_idx_l0_default_active_minus1 = 0).
    w.u(1, 0);

    // §7.3.3.1 — ref_pic_list_modification(): list-0 only for P-slice.
    // ref_pic_list_modification_flag_l0 = 0 (no modifications).
    w.u(1, 0);
    // No list-1 loop for P-slice.

    // §7.3.3.2 — pred_weight_table() absent (PPS weighted_pred_flag == 0).

    // §7.3.3.3 — dec_ref_pic_marking() when nal_ref_idc != 0. For the
    // non-IDR path we choose adaptive_ref_pic_marking_mode_flag = 0
    // (sliding window).
    if cfg.nal_ref_idc != 0 {
        w.u(1, 0); // adaptive_ref_pic_marking_mode_flag = 0
    }

    // No cabac_init_idc — entropy_coding_mode_flag == 0 (CAVLC).
    w.se(cfg.slice_qp_delta);
    // No SP/SI fields — slice_type is P.

    // §7.3.3 — deblocking_filter_control_present_flag == 1 in our PPS.
    debug_assert!(cfg.disable_deblocking_filter_idc <= 2);
    debug_assert!((-6..=6).contains(&cfg.slice_alpha_c0_offset_div2));
    debug_assert!((-6..=6).contains(&cfg.slice_beta_offset_div2));
    w.ue(cfg.disable_deblocking_filter_idc);
    if cfg.disable_deblocking_filter_idc != 1 {
        w.se(cfg.slice_alpha_c0_offset_div2);
        w.se(cfg.slice_beta_offset_div2);
    }
}

// ---------------------------------------------------------------------------
// Round-20 — B-slice header (single L0 + single L1 ref, no list mod, spatial
// direct MV pred, non-reference B by default).
// ---------------------------------------------------------------------------

/// Configuration for [`write_b_slice_header`]. Holds only the slice-header
/// fields. Caller appends `slice_data()` and trailing bits.
///
/// This writer covers the **simple non-IDR B-slice** path:
///
/// * `nal_unit_type = 1` (Slice non-IDR)
/// * `slice_type` = 6 (B, all slices same type) per Table 7-6
/// * `direct_spatial_mv_pred_flag = 1` — round 20 only emits explicit
///   B_L0_16x16 / B_L1_16x16 / B_Bi_16x16 macroblocks (no B_Skip /
///   B_Direct yet) so the flag is irrelevant in practice; we set it to
///   1 (spatial direct, the simpler mode) for consistency with §8.4.1.2.2.
/// * `num_ref_idx_active_override_flag = 0` (use PPS defaults; for our
///   round-20 PPS that's `num_ref_idx_l0_default_active_minus1 = 0` and
///   `num_ref_idx_l1_default_active_minus1 = 0` ⇒ one L0 ref + one L1 ref)
/// * `ref_pic_list_modification`: list-0 and list-1 modification flags = 0
/// * `dec_ref_pic_marking` only when `nal_ref_idc != 0` (we emit B as
///   non-reference, so absent)
/// * No `pred_weight_table` (PPS `weighted_bipred_idc = 0` ⇒ default
///   weighted prediction per §8.4.2.3.1, i.e. `(L0 + L1 + 1) >> 1` for
///   bipred and identity for L0-only / L1-only)
#[derive(Debug, Clone, Copy)]
pub struct BSliceHeaderConfig {
    pub first_mb_in_slice: u32,
    /// Raw `slice_type` per Table 7-6. Use 6 for "all slices in picture
    /// are B" (slice_type % 5 == 1 and >= 5).
    pub slice_type_raw: u32,
    pub pic_parameter_set_id: u32,
    pub frame_num: u32,
    pub frame_num_bits: u32,
    pub pic_order_cnt_lsb: u32,
    pub poc_lsb_bits: u32,
    /// §7.4.3 / §8.4.1.2.2 — `direct_spatial_mv_pred_flag`. For our
    /// round-20 explicit-only B path we set 1 (spatial). The value
    /// only matters if the bitstream contains a B_Skip / B_Direct MB.
    pub direct_spatial_mv_pred_flag: bool,
    pub slice_qp_delta: i32,
    pub disable_deblocking_filter_idc: u32,
    pub slice_alpha_c0_offset_div2: i32,
    pub slice_beta_offset_div2: i32,
    /// `nal_ref_idc` of the parent NAL — used to gate
    /// `dec_ref_pic_marking()` per §7.3.3 last branch. For non-reference
    /// B-frames pass 0; the writer omits dec_ref_pic_marking().
    pub nal_ref_idc: u32,
}

/// Emit the bits of a single B-slice header (non-IDR) into the supplied
/// `BitWriter`. The writer is left at the position where `slice_data()`
/// should begin.
pub fn write_b_slice_header(w: &mut BitWriter, cfg: &BSliceHeaderConfig) {
    debug_assert!(cfg.slice_type_raw % 5 == 1); // B-slice
    w.ue(cfg.first_mb_in_slice);
    w.ue(cfg.slice_type_raw);
    w.ue(cfg.pic_parameter_set_id);
    // No `colour_plane_id` (separate_colour_plane_flag == 0).
    w.u(cfg.frame_num_bits, cfg.frame_num);
    // No field_pic_flag — frame_mbs_only_flag == 1.
    // No idr_pic_id — non-IDR.
    // POC type 0 branch.
    w.u(cfg.poc_lsb_bits, cfg.pic_order_cnt_lsb);
    // No delta_pic_order_cnt_bottom (PPS bottom_field_pic_order_in_frame_present_flag == 0).
    // No redundant_pic_cnt (PPS redundant_pic_cnt_present_flag == 0).

    // §7.3.3 — direct_spatial_mv_pred_flag for B slices (always emitted
    // even when no Direct MB is present, per §7.3.3 syntax structure).
    w.u(
        1,
        if cfg.direct_spatial_mv_pred_flag {
            1
        } else {
            0
        },
    );

    // §7.3.3 — num_ref_idx_active_override_flag. We use the PPS default
    // (num_ref_idx_l0_default_active_minus1 = 0, ditto l1), so the
    // override is 0 and the `_minus1` fields are absent.
    w.u(1, 0);

    // §7.3.3.1 — ref_pic_list_modification(): both lists for B-slice.
    // ref_pic_list_modification_flag_l0 = 0, ref_pic_list_modification_flag_l1 = 0.
    w.u(1, 0);
    w.u(1, 0);

    // §7.3.3.2 — pred_weight_table() absent (PPS weighted_pred_flag == 0
    // for P/SP, weighted_bipred_idc == 0 for B → default weighted pred
    // per §8.4.2.3.1).

    // §7.3.3.3 — dec_ref_pic_marking() when nal_ref_idc != 0. For non-
    // reference B-frames (nal_ref_idc == 0) the field is absent.
    if cfg.nal_ref_idc != 0 {
        w.u(1, 0); // adaptive_ref_pic_marking_mode_flag = 0 (sliding window)
    }

    // No cabac_init_idc — entropy_coding_mode_flag == 0 (CAVLC).
    w.se(cfg.slice_qp_delta);
    // No SP/SI fields — slice_type is B.

    // §7.3.3 — deblocking_filter_control_present_flag == 1 in our PPS.
    debug_assert!(cfg.disable_deblocking_filter_idc <= 2);
    debug_assert!((-6..=6).contains(&cfg.slice_alpha_c0_offset_div2));
    debug_assert!((-6..=6).contains(&cfg.slice_beta_offset_div2));
    w.ue(cfg.disable_deblocking_filter_idc);
    if cfg.disable_deblocking_filter_idc != 1 {
        w.se(cfg.slice_alpha_c0_offset_div2);
        w.se(cfg.slice_beta_offset_div2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::nal::build_nal_unit;
    use crate::encoder::pps::{build_baseline_pps_rbsp, BaselinePpsConfig};
    use crate::encoder::sps::{build_baseline_sps_rbsp, BaselineSpsConfig};
    use crate::nal::{parse_nal_unit, AnnexBSplitter, NalHeader, NalUnitType};
    use crate::pps::Pps;
    use crate::slice_header::{SliceHeader, SliceType};
    use crate::sps::Sps;

    #[test]
    fn idr_slice_header_round_trips_through_decoder_parser() {
        let sps_rbsp = build_baseline_sps_rbsp(&BaselineSpsConfig {
            seq_parameter_set_id: 0,
            level_idc: 30,
            width_in_mbs: 4,
            height_in_mbs: 4,
            log2_max_frame_num_minus4: 4,
            log2_max_poc_lsb_minus4: 4,
            max_num_ref_frames: 1,
            profile_idc: 66,
        });
        let sps = Sps::parse(&sps_rbsp).unwrap();
        let pps_rbsp = build_baseline_pps_rbsp(&BaselinePpsConfig::default());
        let pps = Pps::parse(&pps_rbsp).unwrap();

        let mut w = BitWriter::new();
        write_idr_i_slice_header(
            &mut w,
            &IdrSliceHeaderConfig {
                first_mb_in_slice: 0,
                slice_type_raw: 7, // I, all-same-type
                pic_parameter_set_id: 0,
                frame_num: 0,
                frame_num_bits: sps.log2_max_frame_num_minus4 + 4,
                idr_pic_id: 0,
                pic_order_cnt_lsb: 0,
                poc_lsb_bits: sps.log2_max_pic_order_cnt_lsb_minus4 + 4,
                slice_qp_delta: 0,
                disable_deblocking_filter_idc: 1,
                slice_alpha_c0_offset_div2: 0,
                slice_beta_offset_div2: 0,
            },
        );
        // Append a trivial slice_data placeholder + rbsp_trailing_bits so
        // the parser doesn't blow up on EOF mid-header.
        w.u(1, 1); // dummy bit so trailing alignment doesn't degenerate
        w.rbsp_trailing_bits();
        let rbsp = w.into_bytes();

        // Build the IDR NAL unit and feed it back through the existing
        // decoder primitives.
        let nal = build_nal_unit(3, NalUnitType::SliceIdr, &rbsp);
        let mut split = AnnexBSplitter::new(&nal);
        let nal_bytes = split.next().unwrap();
        let nu = parse_nal_unit(nal_bytes).unwrap();
        let header = NalHeader {
            forbidden_zero_bit: 0,
            nal_ref_idc: 3,
            nal_unit_type: NalUnitType::SliceIdr,
        };
        let parsed = SliceHeader::parse(&nu.rbsp, &sps, &pps, &header).expect("parse slice header");
        assert_eq!(parsed.first_mb_in_slice, 0);
        assert_eq!(parsed.slice_type, SliceType::I);
        assert!(parsed.all_slices_same_type);
        assert_eq!(parsed.pic_parameter_set_id, 0);
        assert_eq!(parsed.frame_num, 0);
        assert_eq!(parsed.idr_pic_id, 0);
        assert_eq!(parsed.pic_order_cnt_lsb, 0);
        assert_eq!(parsed.slice_qp_delta, 0);
        assert_eq!(parsed.disable_deblocking_filter_idc, 1);
    }

    #[test]
    fn p_slice_header_round_trips_through_decoder_parser() {
        let sps_rbsp = build_baseline_sps_rbsp(&BaselineSpsConfig {
            seq_parameter_set_id: 0,
            level_idc: 30,
            width_in_mbs: 4,
            height_in_mbs: 4,
            log2_max_frame_num_minus4: 4,
            log2_max_poc_lsb_minus4: 4,
            max_num_ref_frames: 1,
            profile_idc: 66,
        });
        let sps = Sps::parse(&sps_rbsp).unwrap();
        let pps_rbsp = build_baseline_pps_rbsp(&BaselinePpsConfig::default());
        let pps = Pps::parse(&pps_rbsp).unwrap();

        let mut w = BitWriter::new();
        write_p_slice_header(
            &mut w,
            &PSliceHeaderConfig {
                first_mb_in_slice: 0,
                slice_type_raw: 5, // P, all-same-type
                pic_parameter_set_id: 0,
                frame_num: 1,
                frame_num_bits: sps.log2_max_frame_num_minus4 + 4,
                pic_order_cnt_lsb: 2,
                poc_lsb_bits: sps.log2_max_pic_order_cnt_lsb_minus4 + 4,
                slice_qp_delta: 0,
                disable_deblocking_filter_idc: 0,
                slice_alpha_c0_offset_div2: 0,
                slice_beta_offset_div2: 0,
                nal_ref_idc: 2,
            },
        );
        // Append a dummy bit + trailing so the parser doesn't blow up.
        w.u(1, 1);
        w.rbsp_trailing_bits();
        let rbsp = w.into_bytes();

        let nal = build_nal_unit(2, NalUnitType::SliceNonIdr, &rbsp);
        let mut split = AnnexBSplitter::new(&nal);
        let nal_bytes = split.next().unwrap();
        let nu = parse_nal_unit(nal_bytes).unwrap();
        let header = NalHeader {
            forbidden_zero_bit: 0,
            nal_ref_idc: 2,
            nal_unit_type: NalUnitType::SliceNonIdr,
        };
        let parsed =
            SliceHeader::parse(&nu.rbsp, &sps, &pps, &header).expect("parse P slice header");
        assert_eq!(parsed.first_mb_in_slice, 0);
        assert_eq!(parsed.slice_type, SliceType::P);
        assert!(parsed.all_slices_same_type);
        assert_eq!(parsed.frame_num, 1);
        assert_eq!(parsed.pic_order_cnt_lsb, 2);
        assert_eq!(parsed.disable_deblocking_filter_idc, 0);
        assert_eq!(parsed.slice_alpha_c0_offset_div2, 0);
        assert_eq!(parsed.slice_beta_offset_div2, 0);
        assert!(!parsed.num_ref_idx_active_override_flag);
        assert!(parsed.ref_pic_list_modification.modifications_l0.is_empty());
        let drm = parsed.dec_ref_pic_marking.as_ref().unwrap();
        assert!(drm.adaptive_marking.is_none());
    }

    #[test]
    fn b_slice_header_round_trips_through_decoder_parser() {
        // Round-20 — non-reference B slice in a Main-profile stream
        // (B-slices are forbidden under Baseline §A.2.2).
        let sps_rbsp = build_baseline_sps_rbsp(&BaselineSpsConfig {
            seq_parameter_set_id: 0,
            level_idc: 30,
            width_in_mbs: 4,
            height_in_mbs: 4,
            log2_max_frame_num_minus4: 4,
            log2_max_poc_lsb_minus4: 4,
            max_num_ref_frames: 2,
            profile_idc: 77,
        });
        let sps = Sps::parse(&sps_rbsp).unwrap();
        let pps_rbsp = build_baseline_pps_rbsp(&BaselinePpsConfig::default());
        let pps = Pps::parse(&pps_rbsp).unwrap();

        let mut w = BitWriter::new();
        write_b_slice_header(
            &mut w,
            &BSliceHeaderConfig {
                first_mb_in_slice: 0,
                slice_type_raw: 6, // B, all-same-type
                pic_parameter_set_id: 0,
                frame_num: 2,
                frame_num_bits: sps.log2_max_frame_num_minus4 + 4,
                pic_order_cnt_lsb: 1,
                poc_lsb_bits: sps.log2_max_pic_order_cnt_lsb_minus4 + 4,
                direct_spatial_mv_pred_flag: true,
                slice_qp_delta: 0,
                disable_deblocking_filter_idc: 0,
                slice_alpha_c0_offset_div2: 0,
                slice_beta_offset_div2: 0,
                nal_ref_idc: 0, // non-reference B
            },
        );
        // Dummy bit + trailing so the parser doesn't blow up on EOF.
        w.u(1, 1);
        w.rbsp_trailing_bits();
        let rbsp = w.into_bytes();

        let nal = build_nal_unit(0, NalUnitType::SliceNonIdr, &rbsp);
        let mut split = AnnexBSplitter::new(&nal);
        let nal_bytes = split.next().unwrap();
        let nu = parse_nal_unit(nal_bytes).unwrap();
        let header = NalHeader {
            forbidden_zero_bit: 0,
            nal_ref_idc: 0,
            nal_unit_type: NalUnitType::SliceNonIdr,
        };
        let parsed =
            SliceHeader::parse(&nu.rbsp, &sps, &pps, &header).expect("parse B slice header");
        assert_eq!(parsed.first_mb_in_slice, 0);
        assert_eq!(parsed.slice_type, SliceType::B);
        assert!(parsed.all_slices_same_type);
        assert_eq!(parsed.frame_num, 2);
        assert_eq!(parsed.pic_order_cnt_lsb, 1);
        assert!(parsed.direct_spatial_mv_pred_flag);
        assert!(!parsed.num_ref_idx_active_override_flag);
        assert!(parsed.ref_pic_list_modification.modifications_l0.is_empty());
        assert!(parsed.ref_pic_list_modification.modifications_l1.is_empty());
        // nal_ref_idc == 0 → dec_ref_pic_marking absent.
        assert!(parsed.dec_ref_pic_marking.is_none());
        assert_eq!(parsed.disable_deblocking_filter_idc, 0);
    }
}
