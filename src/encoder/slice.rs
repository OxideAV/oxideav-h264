//! §7.3.3 — slice_header emit (encoder side, IDR I-slice path).
//!
//! Round-14: in-loop deblocking is enabled by default
//! (`disable_deblocking_filter_idc=0`). The PPS's
//! `deblocking_filter_control_present_flag=1` lets the slice header
//! drive the loop-filter state. Round-1..13 emitted
//! `disable_deblocking_filter_idc=1` to keep the encoder simple; we now
//! mirror what the decoder produces so the encoder's local recon stays
//! aligned with what downstream decoders see.

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
}
