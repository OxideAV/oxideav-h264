//! §7.3.2.1 — encoder for `seq_parameter_set_rbsp()`.
//!
//! Emits the minimum subset needed by the round-1 Baseline encoder:
//!
//! * `profile_idc = 66` (Baseline)
//! * `constraint_set0_flag = 1` (Baseline subset)
//! * `chroma_format_idc = 1` inferred (4:2:0 — not signalled because
//!   profile 66 is not in the chroma-extended group, §7.3.2.1.1)
//! * `pic_order_cnt_type = 0`, `log2_max_pic_order_cnt_lsb_minus4 = 4`
//! * `frame_mbs_only_flag = 1`, no FMO, no VUI, no cropping
//!
//! Width/height are passed in macroblocks (16-sample units). The picture
//! dimensions in samples must be a multiple of 16 — this round-1 path
//! does not emit `frame_cropping`. Callers needing odd-aligned output
//! should pad the input to a 16-multiple before encoding.
//!
//! The returned bytes form the RBSP body that goes after the NAL header.
//! Wrap with [`crate::encoder::nal::build_nal_unit`] using
//! `NalUnitType::Sps` to obtain a complete Annex B NAL unit.

use crate::encoder::bitstream::BitWriter;

/// Configuration for [`build_baseline_sps_rbsp`].
#[derive(Debug, Clone, Copy)]
pub struct BaselineSpsConfig {
    pub seq_parameter_set_id: u32,
    pub level_idc: u8,
    pub width_in_mbs: u32,
    pub height_in_mbs: u32,
    /// `log2_max_frame_num_minus4` per §7.4.2.1.1 (range 0..=12). The
    /// resulting `MaxFrameNum = 2^(this + 4)`. Round-1: 4 → 256 frames.
    pub log2_max_frame_num_minus4: u32,
    /// `log2_max_pic_order_cnt_lsb_minus4` per §7.4.2.1.1. Round-1: 4 →
    /// 256 POC LSB values.
    pub log2_max_poc_lsb_minus4: u32,
    /// `max_num_ref_frames`. For an IDR-only encoder 1 is sufficient.
    pub max_num_ref_frames: u32,
}

impl Default for BaselineSpsConfig {
    fn default() -> Self {
        Self {
            seq_parameter_set_id: 0,
            level_idc: 30,
            width_in_mbs: 0,
            height_in_mbs: 0,
            log2_max_frame_num_minus4: 4,
            log2_max_poc_lsb_minus4: 4,
            max_num_ref_frames: 1,
        }
    }
}

/// Build a Baseline SPS RBSP body (§7.3.2.1.1). Returns the bytes
/// *without* the NAL header byte. Wrap with `build_nal_unit` to get a
/// complete Annex B NAL unit.
pub fn build_baseline_sps_rbsp(cfg: &BaselineSpsConfig) -> Vec<u8> {
    let mut w = BitWriter::new();

    // §7.3.2.1.1 — profile_idc = 66 (Baseline).
    w.u(8, 66);
    // constraint_set0_flag=1 (Baseline subset), constraint_set1..5_flag=0,
    // reserved_zero_2bits=00.
    let cs0 = 1;
    w.u(1, cs0);
    for _ in 0..5 {
        w.u(1, 0);
    }
    w.u(2, 0);
    w.u(8, cfg.level_idc as u32);

    w.ue(cfg.seq_parameter_set_id);
    // No chroma-extended fields for profile 66 — chroma_format_idc=1
    // (4:2:0) is inferred per §7.4.2.1.1.

    w.ue(cfg.log2_max_frame_num_minus4);
    // pic_order_cnt_type = 0.
    w.ue(0);
    w.ue(cfg.log2_max_poc_lsb_minus4);

    w.ue(cfg.max_num_ref_frames);
    // gaps_in_frame_num_value_allowed_flag = 0.
    w.u(1, 0);

    debug_assert!(cfg.width_in_mbs >= 1);
    debug_assert!(cfg.height_in_mbs >= 1);
    w.ue(cfg.width_in_mbs - 1);
    w.ue(cfg.height_in_mbs - 1);

    // frame_mbs_only_flag = 1 (no field/MBAFF).
    w.u(1, 1);
    // direct_8x8_inference_flag = 1 (recommended even for I-only —
    // some decoders verify this is set when frame_mbs_only_flag=1).
    w.u(1, 1);
    // frame_cropping_flag = 0.
    w.u(1, 0);
    // vui_parameters_present_flag = 0.
    w.u(1, 0);

    // §7.3.2.11 — rbsp_trailing_bits().
    w.rbsp_trailing_bits();

    w.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sps::Sps;

    #[test]
    fn baseline_sps_round_trips_through_decoder_parser() {
        let cfg = BaselineSpsConfig {
            seq_parameter_set_id: 0,
            level_idc: 30,
            width_in_mbs: 4, // 64 samples
            height_in_mbs: 4,
            log2_max_frame_num_minus4: 4,
            log2_max_poc_lsb_minus4: 4,
            max_num_ref_frames: 1,
        };
        let rbsp = build_baseline_sps_rbsp(&cfg);
        let sps = Sps::parse(&rbsp).expect("decoder parses our SPS");
        assert_eq!(sps.profile_idc, 66);
        assert_eq!(sps.constraint_set_flags & 1, 1);
        assert_eq!(sps.level_idc, 30);
        assert_eq!(sps.seq_parameter_set_id, 0);
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.log2_max_frame_num_minus4, 4);
        assert_eq!(sps.pic_order_cnt_type, 0);
        assert_eq!(sps.log2_max_pic_order_cnt_lsb_minus4, 4);
        assert_eq!(sps.max_num_ref_frames, 1);
        assert!(!sps.gaps_in_frame_num_value_allowed_flag);
        assert_eq!(sps.pic_width_in_mbs(), 4);
        assert_eq!(sps.frame_height_in_mbs(), 4);
        assert!(sps.frame_mbs_only_flag);
        assert!(sps.frame_cropping.is_none());
        assert!(!sps.vui_parameters_present_flag);
    }
}
