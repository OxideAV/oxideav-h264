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

use crate::encoder::bitstream::BitWriter;

/// Configuration for [`build_baseline_pps_rbsp`].
#[derive(Debug, Clone, Copy, Default)]
pub struct BaselinePpsConfig {
    pub pic_parameter_set_id: u32,
    pub seq_parameter_set_id: u32,
    /// `pic_init_qp_minus26` per §7.4.2.2. The actual slice-level QP is
    /// further offset by `slice_qp_delta` in the slice header. Default
    /// is 0 → QP = 26.
    pub pic_init_qp_minus26: i32,
    /// `chroma_qp_index_offset` per §7.4.2.2. Range -12..=12.
    pub chroma_qp_index_offset: i32,
}

/// Build a Baseline PPS RBSP body (§7.3.2.2).
pub fn build_baseline_pps_rbsp(cfg: &BaselinePpsConfig) -> Vec<u8> {
    let mut w = BitWriter::new();

    w.ue(cfg.pic_parameter_set_id);
    w.ue(cfg.seq_parameter_set_id);

    // entropy_coding_mode_flag = 0 (CAVLC).
    w.u(1, 0);
    // bottom_field_pic_order_in_frame_present_flag = 0.
    w.u(1, 0);
    // num_slice_groups_minus1 = 0 (no FMO).
    w.ue(0);

    // num_ref_idx_l0_default_active_minus1 = 0, ditto l1.
    w.ue(0);
    w.ue(0);

    // weighted_pred_flag = 0, weighted_bipred_idc = 0 (2 bits).
    w.u(1, 0);
    w.u(2, 0);

    w.se(cfg.pic_init_qp_minus26);
    w.se(0); // pic_init_qs_minus26
    w.se(cfg.chroma_qp_index_offset);

    // deblocking_filter_control_present_flag = 1 — lets the slice
    // header carry `disable_deblocking_filter_idc` and the offsets.
    w.u(1, 1);
    // constrained_intra_pred_flag = 0.
    w.u(1, 0);
    // redundant_pic_cnt_present_flag = 0.
    w.u(1, 0);

    // No optional tail — caller relies on inferred defaults
    // (transform_8x8_mode_flag = 0, etc.).
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
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
            pic_init_qp_minus26: 0,
            chroma_qp_index_offset: 0,
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
