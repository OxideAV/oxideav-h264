//! §7.3.2.1 — encoder for `seq_parameter_set_rbsp()`.
//!
//! Emits the minimum subset needed by the round-1 Baseline encoder:
//!
//! * `profile_idc = 66` (Baseline) by default; round 20 lets callers
//!   bump to Main (77) via [`BaselineSpsConfig::profile_idc`] so that
//!   B-slices are permitted (§A.2.2 — Baseline forbids B-slices).
//! * `constraint_set0_flag = 1` for Baseline; cleared when the caller
//!   bumps the profile (Main/High don't satisfy the Baseline subset).
//! * `chroma_format_idc = 1` inferred (4:2:0 — not signalled because
//!   neither profile 66 nor 77 is in the chroma-extended group,
//!   §7.3.2.1.1)
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
//!
//! Round-27: 4:2:2 (`chroma_format_idc = 2`, profile_idc = 122 — High
//! 4:2:2). When the caller selects profile 122, the writer emits the
//! §7.3.2.1.1 chroma-extended group (chroma_format_idc / bit_depth_*
//! / qpprime / seq_scaling_matrix_present), pinning bit_depth_luma /
//! bit_depth_chroma at 8 (10/12/14-bit are out of round-27 scope) and
//! seq_scaling_matrix_present_flag to 0 (flat scaling).
//!
//! Round-28: 4:4:4 (`chroma_format_idc = 3`, profile_idc = 244 — High
//! 4:4:4 Predictive). Same chroma-extended group emit, but
//! `chroma_format_idc=3` adds a `separate_colour_plane_flag` bit (held
//! at 0 — separate planes are out of round-28 scope). The decoder maps
//! `chroma_format_idc=3 + separate_colour_plane_flag=0` back to
//! `ChromaArrayType=3`, the "chroma coded like luma" path of §7.3.5.3.

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
    /// Round-20 B-slice tests need 2 (the IDR's L0 ref and the prior
    /// P-frame's L1 ref).
    pub max_num_ref_frames: u32,
    /// §7.4.2.1 — `profile_idc`. Defaults to 66 (Baseline). Round-20
    /// allows 77 (Main) so that B-slices are permitted (§A.2.2).
    /// Round-27 allows 122 (High 4:2:2). Round-28 adds 244 (High 4:4:4
    /// Predictive). All three trigger the §7.3.2.1.1 chroma-extended
    /// group emission. Other chroma-extended profiles (100, 110, …)
    /// and bit depths > 8 remain out of scope.
    pub profile_idc: u8,
    /// §7.4.2.1.1 — `chroma_format_idc`. Default 1 (4:2:0). Set to 2
    /// (4:2:2) when emitting a High 4:2:2 (122) SPS or 3 (4:4:4) when
    /// emitting a High 4:4:4 Predictive (244) SPS. The writer asserts
    /// the (profile_idc, chroma_format_idc) pairing is one it understands.
    pub chroma_format_idc: u32,
    /// §7.3.2.1.1 / §7.3.2.1.1.1 — when `Some`, emit
    /// `seq_scaling_matrix_present_flag = 1` with every list present.
    /// `ScalingListsSpec::Default` codes each list as
    /// **UseDefaultScalingMatrixFlag** (a single `delta_scale = -8`
    /// drives nextScale to 0 at j == 0), selecting the Table 7-3 /
    /// Table 7-4 default matrices; `ScalingListsSpec::Custom` codes
    /// the caller's values explicitly through the §7.3.2.1.1.1
    /// delta_scale chain (round-391). Requires a chroma-extended
    /// profile_idc (the flag lives in the §7.3.2.1.1 optional group).
    pub seq_scaling_lists: Option<super::ScalingListsSpec>,
    /// Round-416 — PAFF: when `true` the writer emits
    /// `frame_mbs_only_flag = 0` + `mb_adaptive_frame_field_flag = 0`
    /// (field pictures, no MBAFF) and interprets `height_in_mbs` as
    /// **FrameHeightInMbs** — `pic_height_in_map_units_minus1` is then
    /// coded as `height_in_mbs / 2 - 1` per §7.4.2.1.1 eq. 7-15/7-16
    /// (a map unit is a field-MB row pair when `frame_mbs_only_flag`
    /// is 0). `height_in_mbs` must be even. Requires `profile_idc !=
    /// 66` — §A.2.1 pins `frame_mbs_only_flag = 1` in Baseline.
    pub interlaced_fields: bool,
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
            profile_idc: 66,
            chroma_format_idc: 1,
            seq_scaling_lists: None,
            interlaced_fields: false,
        }
    }
}

/// §7.3.2.1.1.1 — emit one `scaling_list()` structure carrying
/// explicit values (each in `1..=255`, given in the `j`-loop scan
/// order). Per the spec's derivation
/// `nextScale = (lastScale + delta_scale + 256) % 256`, so the
/// delta between consecutive values is wrapped into the mandated
/// `-128..=127` `delta_scale` range.
pub(crate) fn write_scaling_list_values(w: &mut BitWriter, values: &[i32]) {
    let mut last = 8i32;
    for &v in values {
        debug_assert!(
            (1..=255).contains(&v),
            "scaling-list value {v} out of the 1..=255 range"
        );
        let mut delta = v - last;
        if delta > 127 {
            delta -= 256;
        }
        if delta < -128 {
            delta += 256;
        }
        w.se(delta);
        last = v;
    }
}

/// Emit the body of one SPS/PPS scaling-list slot per the caller's
/// [`super::ScalingListsSpec`]. `list_idx` follows the §7.3.2.1.1 /
/// §7.3.2.2 loop: 0..=2 are the 4x4 intra lists (Y/Cb/Cr), 3..=5 the
/// 4x4 inter lists, and 6.. alternate 8x8 intra / 8x8 inter (the
/// Y/Cb/Cr repetition at 4:4:4 keeps the same parity rule).
pub(crate) fn write_scaling_list_slot(
    w: &mut BitWriter,
    spec: &super::ScalingListsSpec,
    list_idx: usize,
) {
    match spec {
        super::ScalingListsSpec::Default => {
            // delta_scale = -8 → nextScale 0 at j == 0 →
            // UseDefaultScalingMatrixFlag (Table 7-3 / 7-4).
            w.se(-8);
        }
        super::ScalingListsSpec::Custom(l) => {
            if list_idx < 6 {
                let vals = if list_idx < 3 { &l.intra4 } else { &l.inter4 };
                write_scaling_list_values(w, vals);
            } else if (list_idx - 6) % 2 == 0 {
                write_scaling_list_values(w, &l.intra8);
            } else {
                write_scaling_list_values(w, &l.inter8);
            }
        }
    }
}

/// Build a Baseline SPS RBSP body (§7.3.2.1.1). Returns the bytes
/// *without* the NAL header byte. Wrap with `build_nal_unit` to get a
/// complete Annex B NAL unit.
pub fn build_baseline_sps_rbsp(cfg: &BaselineSpsConfig) -> Vec<u8> {
    let mut w = BitWriter::new();

    // §7.3.2.1.1 — profile_idc.
    debug_assert!(
        matches!(cfg.profile_idc, 66 | 77 | 88 | 100 | 122 | 244),
        "this writer only emits SPS bodies for profile_idc ∈ {{66, 77, 88, 100, 122, 244}} \
         (Baseline / Main / Extended / High / High 4:2:2 / High 4:4:4 Predictive). Profile {} \
         would require additional bit_depth_* / scaling-matrix wiring per §7.3.2.1.1.",
        cfg.profile_idc,
    );
    // Round-27/28: chroma_format_idc accepted values.
    debug_assert!(
        matches!(cfg.chroma_format_idc, 1..=3),
        "chroma_format_idc {} not in writer scope (1=4:2:0, 2=4:2:2, 3=4:4:4)",
        cfg.chroma_format_idc,
    );
    // Round-27: 4:2:2 only with profile 122. Round-28: 4:4:4 only with
    // profile 244. The other chroma-extended-group profiles aren't
    // emitted by this writer.
    debug_assert!(
        match cfg.chroma_format_idc {
            1 => true,
            2 => cfg.profile_idc == 122,
            3 => cfg.profile_idc == 244,
            _ => false,
        },
        "chroma_format_idc={} not paired with the expected profile_idc; got profile {}",
        cfg.chroma_format_idc,
        cfg.profile_idc,
    );
    w.u(8, cfg.profile_idc as u32);
    // §A.2 — constraint_set flags. constraint_set0_flag is the Baseline
    // subset gate (only meaningful when the bitstream conforms to
    // profile 66's constraints). Setting it on a Main / Extended SPS
    // would force decoders to also verify the Baseline restrictions
    // (no B-slices, no weighted pred, no CABAC) which is incompatible
    // with what we want for B-slice support.
    let cs0 = if cfg.profile_idc == 66 { 1 } else { 0 };
    w.u(1, cs0);
    for _ in 0..5 {
        w.u(1, 0);
    }
    w.u(2, 0);
    w.u(8, cfg.level_idc as u32);

    w.ue(cfg.seq_parameter_set_id);
    // §7.3.2.1.1 — chroma-extended group. Profile 122 (High 4:2:2)
    // triggers the chroma_format_idc / bit_depth_* / qpprime /
    // seq_scaling_matrix_present_flag fields. Round 27 pins bit depth
    // to 8 and clears scaling-matrix-present (flat scaling lists).
    if matches!(
        cfg.profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        w.ue(cfg.chroma_format_idc);
        if cfg.chroma_format_idc == 3 {
            // separate_colour_plane_flag — 4:4:4 only, out of scope.
            w.u(1, 0);
        }
        // bit_depth_luma_minus8 = 0 (8-bit luma).
        w.ue(0);
        // bit_depth_chroma_minus8 = 0 (8-bit chroma).
        w.ue(0);
        // qpprime_y_zero_transform_bypass_flag = 0.
        w.u(1, 0);
        // §7.3.2.1.1.1 — seq_scaling_matrix_present_flag. When default
        // matrices are requested, each of the 8 (12 at 4:4:4) lists is
        // present and coded as UseDefaultScalingMatrixFlag: one
        // delta_scale = -8 makes nextScale 0 at j == 0, selecting the
        // Table 7-3 / Table 7-4 defaults for the whole list.
        if let Some(spec) = &cfg.seq_scaling_lists {
            w.u(1, 1);
            let n_lists = if cfg.chroma_format_idc == 3 { 12 } else { 8 };
            for i in 0..n_lists {
                w.u(1, 1); // seq_scaling_list_present_flag[i]
                write_scaling_list_slot(&mut w, spec, i);
            }
        } else {
            w.u(1, 0);
        }
    }

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
    if cfg.interlaced_fields {
        // §A.2.1 — Baseline requires frame_mbs_only_flag = 1.
        debug_assert!(
            cfg.profile_idc != 66,
            "interlaced_fields requires a non-Baseline profile (§A.2.1)",
        );
        debug_assert!(
            cfg.height_in_mbs % 2 == 0,
            "FrameHeightInMbs must be even for field coding",
        );
        // §7.4.2.1.1 eq. 7-15/7-16 — with frame_mbs_only_flag == 0 a
        // pic height map unit covers two frame MB rows:
        // FrameHeightInMbs = 2 * PicHeightInMapUnits.
        w.ue(cfg.height_in_mbs / 2 - 1);
        // frame_mbs_only_flag = 0 (field pictures allowed).
        w.u(1, 0);
        // mb_adaptive_frame_field_flag = 0 (PAFF, no MBAFF).
        w.u(1, 0);
    } else {
        w.ue(cfg.height_in_mbs - 1);
        // frame_mbs_only_flag = 1 (no field/MBAFF).
        w.u(1, 1);
    }
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
            seq_scaling_lists: None,
            interlaced_fields: false,
            seq_parameter_set_id: 0,
            level_idc: 30,
            width_in_mbs: 4, // 64 samples
            height_in_mbs: 4,
            log2_max_frame_num_minus4: 4,
            log2_max_poc_lsb_minus4: 4,
            max_num_ref_frames: 1,
            profile_idc: 66,
            chroma_format_idc: 1,
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

    #[test]
    fn high_444_sps_emits_chroma_extended_group_with_separate_colour_plane_flag() {
        // §7.3.2.1.1 / §7.4.2.1.1 — profile_idc=244 (High 4:4:4
        // Predictive) triggers the chroma-extended group, and
        // chroma_format_idc=3 additionally requires the
        // separate_colour_plane_flag bit (held at 0 in round-28 since
        // separate planes are out of scope; the decoder maps the pair
        // (chroma_format_idc=3, separate_colour_plane_flag=0) back to
        // ChromaArrayType=3).
        let cfg = BaselineSpsConfig {
            seq_scaling_lists: None,
            interlaced_fields: false,
            seq_parameter_set_id: 0,
            level_idc: 30,
            width_in_mbs: 4,
            height_in_mbs: 4,
            log2_max_frame_num_minus4: 4,
            log2_max_poc_lsb_minus4: 4,
            max_num_ref_frames: 1,
            profile_idc: 244,
            chroma_format_idc: 3,
        };
        let rbsp = build_baseline_sps_rbsp(&cfg);
        let sps = Sps::parse(&rbsp).expect("decoder parses our 4:4:4 SPS");
        assert_eq!(sps.profile_idc, 244);
        assert_eq!(sps.constraint_set_flags, 0);
        assert_eq!(sps.chroma_format_idc, 3);
        assert!(!sps.separate_colour_plane_flag);
        assert_eq!(sps.bit_depth_luma_minus8, 0);
        assert_eq!(sps.bit_depth_chroma_minus8, 0);
        assert!(!sps.qpprime_y_zero_transform_bypass_flag);
        assert!(!sps.seq_scaling_matrix_present_flag);
        // §6.2 — ChromaArrayType == chroma_format_idc when
        // separate_colour_plane_flag == 0 → 3 (4:4:4 unified-plane path).
        assert_eq!(sps.chroma_array_type(), 3);
        assert_eq!(sps.pic_width_in_mbs(), 4);
        assert_eq!(sps.frame_height_in_mbs(), 4);
    }

    #[test]
    fn high_422_sps_emits_chroma_extended_group() {
        // §7.3.2.1.1 / §7.4.2.1.1 — profile_idc=122 triggers the
        // chroma_format_idc / bit_depth_*_minus8 / qpprime /
        // seq_scaling_matrix_present_flag tail. Round-27 emits
        // chroma_format_idc=2 (4:2:2), 8-bit depth, no scaling matrix.
        let cfg = BaselineSpsConfig {
            seq_scaling_lists: None,
            interlaced_fields: false,
            seq_parameter_set_id: 0,
            level_idc: 30,
            width_in_mbs: 4,
            height_in_mbs: 4,
            log2_max_frame_num_minus4: 4,
            log2_max_poc_lsb_minus4: 4,
            max_num_ref_frames: 1,
            profile_idc: 122,
            chroma_format_idc: 2,
        };
        let rbsp = build_baseline_sps_rbsp(&cfg);
        let sps = Sps::parse(&rbsp).expect("decoder parses our 4:2:2 SPS");
        assert_eq!(sps.profile_idc, 122);
        // Profile 122 → constraint_set0_flag is 0 (Baseline-only gate).
        assert_eq!(sps.constraint_set_flags, 0);
        assert_eq!(sps.chroma_format_idc, 2);
        assert!(!sps.separate_colour_plane_flag);
        assert_eq!(sps.bit_depth_luma_minus8, 0);
        assert_eq!(sps.bit_depth_chroma_minus8, 0);
        assert!(!sps.qpprime_y_zero_transform_bypass_flag);
        assert!(!sps.seq_scaling_matrix_present_flag);
        // §6.2 — ChromaArrayType == chroma_format_idc when
        // separate_colour_plane_flag == 0.
        assert_eq!(sps.chroma_array_type(), 2);
        assert_eq!(sps.pic_width_in_mbs(), 4);
        assert_eq!(sps.frame_height_in_mbs(), 4);
    }
}
