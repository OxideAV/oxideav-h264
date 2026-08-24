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
use crate::vui::{HrdParameters, VuiParameters};

/// Configuration for [`build_baseline_sps_rbsp`].
#[derive(Debug, Clone)]
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
    /// Round-448 — §7.3.2.1.1 `separate_colour_plane_flag`. When
    /// `true` the SPS codes `chroma_format_idc = 3` followed by
    /// `separate_colour_plane_flag = 1` (the coded planes each use the
    /// monochrome syntax — ChromaArrayType 0 — per §7.4.2.1.1), and
    /// `cfg.chroma_format_idc` must be 0 (the caller's per-plane
    /// encoder runs the monochrome pipeline). Requires
    /// `profile_idc = 244` (§A.2.7: 4:4:4 syntax) and flat scaling
    /// lists (`seq_scaling_lists = None`).
    pub separate_colour_plane: bool,
    /// Round-451 — §7.3.2.1.1 `bit_depth_luma_minus8` /
    /// `bit_depth_chroma_minus8` (0..=6, i.e. 8..=14-bit). Non-zero
    /// values require a chroma-extended profile that admits the depth
    /// (§A.2: High 10 up to 10-bit, High 4:4:4 Predictive up to
    /// 14-bit); the writer only emits the fields inside the
    /// chroma-extended group.
    pub bit_depth_luma_minus8: u32,
    /// See [`Self::bit_depth_luma_minus8`].
    pub bit_depth_chroma_minus8: u32,
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
    /// Round-430 — §E.1.1 `vui_parameters()` to append (with
    /// `vui_parameters_present_flag = 1`). `None` keeps the historical
    /// `vui_parameters_present_flag = 0` emission. Used by the
    /// rate-controlled sessions to annotate CBR streams with §E.1.2
    /// NAL HRD parameters + timing info so the Annex C buffering model
    /// is formally declared in-band.
    pub vui: Option<VuiParameters>,
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
            separate_colour_plane: false,
            seq_scaling_lists: None,
            interlaced_fields: false,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            vui: None,
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
    // Round-27/28: chroma_format_idc accepted values. Round-448 adds
    // 0 (monochrome 4:0:0).
    debug_assert!(
        matches!(cfg.chroma_format_idc, 0..=3),
        "chroma_format_idc {} not in writer scope (0=4:0:0, 1=4:2:0, 2=4:2:2, 3=4:4:4)",
        cfg.chroma_format_idc,
    );
    // Round-27: 4:2:2 only with profile 122. Round-28: 4:4:4 only with
    // profile 244. Round-448: 4:0:0 is a High-family tool — §A.2.4
    // (High) lists chroma_format_idc ∈ {0, 1}, and the higher-fidelity
    // High profiles (§A.2.5..§A.2.7) include monochrome in their
    // supported chroma range too. The other chroma-extended-group
    // profiles aren't emitted by this writer.
    debug_assert!(
        !cfg.separate_colour_plane
            || (cfg.profile_idc == 244
                && cfg.chroma_format_idc == 0
                && cfg.seq_scaling_lists.is_none()),
        "separate_colour_plane requires profile 244, an internally-monochrome \
         (chroma_format_idc = 0) per-plane encoder, and flat scaling lists",
    );
    debug_assert!(
        match cfg.chroma_format_idc {
            0 => matches!(cfg.profile_idc, 100 | 110 | 122 | 244),
            1 => true,
            2 => cfg.profile_idc == 122,
            3 => cfg.profile_idc == 244,
            _ => false,
        },
        "chroma_format_idc={} not paired with the expected profile_idc; got profile {}",
        cfg.chroma_format_idc,
        cfg.profile_idc,
    );
    debug_assert!(
        (cfg.bit_depth_luma_minus8 == 0 && cfg.bit_depth_chroma_minus8 == 0)
            || matches!(cfg.profile_idc, 100 | 110 | 122 | 244),
        ">8-bit depths require a chroma-extended profile (§A.2)",
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
        // Round-448 — a separate-colour-plane SPS codes the on-wire
        // pair (chroma_format_idc = 3, separate_colour_plane_flag = 1)
        // while the per-plane coding stays monochrome (§7.4.2.1.1:
        // ChromaArrayType is 0 when separate_colour_plane_flag is 1).
        let wire_chroma_format_idc = if cfg.separate_colour_plane {
            3
        } else {
            cfg.chroma_format_idc
        };
        w.ue(wire_chroma_format_idc);
        if wire_chroma_format_idc == 3 {
            // §7.3.2.1.1 — separate_colour_plane_flag (4:4:4 only).
            w.u(1, u32::from(cfg.separate_colour_plane));
        }
        // §7.3.2.1.1 — bit_depth_luma_minus8 / bit_depth_chroma_minus8
        // (round-451: caller-selectable, 0..=6).
        debug_assert!(cfg.bit_depth_luma_minus8 <= 6 && cfg.bit_depth_chroma_minus8 <= 6);
        w.ue(cfg.bit_depth_luma_minus8);
        w.ue(cfg.bit_depth_chroma_minus8);
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
    // §E.1.1 — vui_parameters_present_flag + vui_parameters().
    if let Some(vui) = &cfg.vui {
        w.u(1, 1);
        write_vui_parameters(&mut w, vui);
    } else {
        w.u(1, 0);
    }

    // §7.3.2.11 — rbsp_trailing_bits().
    w.rbsp_trailing_bits();

    w.into_bytes()
}

/// §E.1.1 — emit a `vui_parameters()` structure. Field-for-field
/// mirror of [`crate::vui::VuiParameters::parse`]; every `Option`
/// drives the corresponding `*_present_flag`.
pub fn write_vui_parameters(w: &mut BitWriter, vui: &VuiParameters) {
    // aspect_ratio_info_present_flag + aspect_ratio_info.
    if let Some(ar) = &vui.aspect_ratio {
        w.u(1, 1);
        w.u(8, ar.aspect_ratio_idc as u32);
        // §E.2.1 Table E-1 — Extended_SAR (255) carries explicit
        // sar_width / sar_height u(16) fields.
        if ar.aspect_ratio_idc == 255 {
            let (sw_, sh) = ar
                .extended_sar
                .expect("aspect_ratio_idc == Extended_SAR requires extended_sar");
            w.u(16, sw_ as u32);
            w.u(16, sh as u32);
        } else {
            debug_assert!(
                ar.extended_sar.is_none(),
                "extended_sar is only writable with aspect_ratio_idc == 255"
            );
        }
    } else {
        w.u(1, 0);
    }
    // overscan_info_present_flag + overscan_appropriate_flag.
    if let Some(f) = vui.overscan_appropriate_flag {
        w.u(1, 1);
        w.u(1, u32::from(f));
    } else {
        w.u(1, 0);
    }
    // video_signal_type_present_flag + block.
    if let Some(vst) = &vui.video_signal_type {
        w.u(1, 1);
        w.u(3, vst.video_format as u32);
        w.u(1, u32::from(vst.video_full_range_flag));
        if let Some(cd) = &vst.colour_description {
            w.u(1, 1);
            w.u(8, cd.colour_primaries as u32);
            w.u(8, cd.transfer_characteristics as u32);
            w.u(8, cd.matrix_coefficients as u32);
        } else {
            w.u(1, 0);
        }
    } else {
        w.u(1, 0);
    }
    // chroma_loc_info_present_flag + block.
    if let Some(cl) = &vui.chroma_loc_info {
        w.u(1, 1);
        w.ue(cl.chroma_sample_loc_type_top_field);
        w.ue(cl.chroma_sample_loc_type_bottom_field);
    } else {
        w.u(1, 0);
    }
    // timing_info_present_flag + block.
    if let Some(t) = &vui.timing_info {
        w.u(1, 1);
        w.u(32, t.num_units_in_tick);
        w.u(32, t.time_scale);
        w.u(1, u32::from(t.fixed_frame_rate_flag));
    } else {
        w.u(1, 0);
    }
    // nal_hrd_parameters_present_flag + hrd_parameters().
    if let Some(h) = &vui.nal_hrd_parameters {
        w.u(1, 1);
        write_hrd_parameters(w, h);
    } else {
        w.u(1, 0);
    }
    // vcl_hrd_parameters_present_flag + hrd_parameters().
    if let Some(h) = &vui.vcl_hrd_parameters {
        w.u(1, 1);
        write_hrd_parameters(w, h);
    } else {
        w.u(1, 0);
    }
    // §E.1.1 — low_delay_hrd_flag only when either HRD block present.
    if vui.nal_hrd_parameters.is_some() || vui.vcl_hrd_parameters.is_some() {
        w.u(1, u32::from(vui.low_delay_hrd_flag.unwrap_or(false)));
    } else {
        debug_assert!(
            vui.low_delay_hrd_flag.is_none(),
            "low_delay_hrd_flag is only writable when an HRD block is present"
        );
    }
    // pic_struct_present_flag.
    w.u(1, u32::from(vui.pic_struct_present_flag));
    // bitstream_restriction_flag + block.
    if let Some(br) = &vui.bitstream_restriction {
        w.u(1, 1);
        w.u(1, u32::from(br.motion_vectors_over_pic_boundaries_flag));
        w.ue(br.max_bytes_per_pic_denom);
        w.ue(br.max_bits_per_mb_denom);
        w.ue(br.log2_max_mv_length_horizontal);
        w.ue(br.log2_max_mv_length_vertical);
        w.ue(br.max_num_reorder_frames);
        w.ue(br.max_dec_frame_buffering);
    } else {
        w.u(1, 0);
    }
}

/// §E.1.2 — emit an `hrd_parameters()` structure. Field-for-field
/// mirror of [`crate::vui::HrdParameters::parse`].
pub fn write_hrd_parameters(w: &mut BitWriter, h: &HrdParameters) {
    debug_assert!(
        h.cpb_cnt_minus1 <= 31,
        "cpb_cnt_minus1 range 0..=31 (§E.2.2)"
    );
    let count = h.cpb_cnt_minus1 as usize + 1;
    debug_assert_eq!(h.bit_rate_value_minus1.len(), count);
    debug_assert_eq!(h.cpb_size_value_minus1.len(), count);
    debug_assert_eq!(h.cbr_flag.len(), count);
    w.ue(h.cpb_cnt_minus1);
    w.u(4, h.bit_rate_scale as u32);
    w.u(4, h.cpb_size_scale as u32);
    for i in 0..count {
        w.ue(h.bit_rate_value_minus1[i]);
        w.ue(h.cpb_size_value_minus1[i]);
        w.u(1, u32::from(h.cbr_flag[i]));
    }
    w.u(5, h.initial_cpb_removal_delay_length_minus1 as u32);
    w.u(5, h.cpb_removal_delay_length_minus1 as u32);
    w.u(5, h.dpb_output_delay_length_minus1 as u32);
    w.u(5, h.time_offset_length as u32);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sps::Sps;
    use crate::vui::TimingInfo;

    /// Round-430 — an SPS carrying §E.1.1 VUI timing + §E.1.2 NAL HRD
    /// round-trips through the decoder's parser field-exactly.
    #[test]
    fn sps_with_vui_timing_and_nal_hrd_round_trips() {
        let hrd = HrdParameters {
            cpb_cnt_minus1: 0,
            bit_rate_scale: 0,
            cpb_size_scale: 0,
            bit_rate_value_minus1: vec![1874],
            cpb_size_value_minus1: vec![7499],
            cbr_flag: vec![true],
            initial_cpb_removal_delay_length_minus1: 23,
            cpb_removal_delay_length_minus1: 23,
            dpb_output_delay_length_minus1: 23,
            time_offset_length: 24,
        };
        let vui = VuiParameters {
            timing_info: Some(TimingInfo {
                num_units_in_tick: 1,
                time_scale: 60,
                fixed_frame_rate_flag: true,
            }),
            nal_hrd_parameters: Some(hrd),
            low_delay_hrd_flag: Some(false),
            ..VuiParameters::default()
        };
        let cfg = BaselineSpsConfig {
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            width_in_mbs: 5,
            height_in_mbs: 4,
            vui: Some(vui.clone()),
            ..BaselineSpsConfig::default()
        };
        let rbsp = build_baseline_sps_rbsp(&cfg);
        let sps = Sps::parse(&rbsp).expect("decoder parses our SPS");
        assert!(sps.vui_parameters_present_flag);
        assert_eq!(sps.vui.as_ref(), Some(&vui), "VUI must round-trip exactly");
        let parsed_hrd = sps
            .vui
            .as_ref()
            .and_then(|v| v.nal_hrd_parameters.as_ref())
            .expect("nal hrd present");
        // BitRate[0] = (bit_rate_value_minus1 + 1) << (6 + bit_rate_scale)
        assert_eq!((parsed_hrd.bit_rate_value_minus1[0] + 1) << 6, 120_000);
        // CpbSize[0] = (cpb_size_value_minus1 + 1) << (4 + cpb_size_scale)
        assert_eq!((parsed_hrd.cpb_size_value_minus1[0] + 1) << 4, 120_000);
        assert!(parsed_hrd.cbr_flag[0]);
    }

    #[test]
    fn baseline_sps_round_trips_through_decoder_parser() {
        let cfg = BaselineSpsConfig {
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            seq_scaling_lists: None,
            interlaced_fields: false,
            vui: None,
            seq_parameter_set_id: 0,
            level_idc: 30,
            width_in_mbs: 4, // 64 samples
            height_in_mbs: 4,
            log2_max_frame_num_minus4: 4,
            log2_max_poc_lsb_minus4: 4,
            max_num_ref_frames: 1,
            profile_idc: 66,
            chroma_format_idc: 1,
            separate_colour_plane: false,
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
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            seq_scaling_lists: None,
            interlaced_fields: false,
            vui: None,
            seq_parameter_set_id: 0,
            level_idc: 30,
            width_in_mbs: 4,
            height_in_mbs: 4,
            log2_max_frame_num_minus4: 4,
            log2_max_poc_lsb_minus4: 4,
            max_num_ref_frames: 1,
            profile_idc: 244,
            chroma_format_idc: 3,
            separate_colour_plane: false,
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
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            seq_scaling_lists: None,
            interlaced_fields: false,
            vui: None,
            seq_parameter_set_id: 0,
            level_idc: 30,
            width_in_mbs: 4,
            height_in_mbs: 4,
            log2_max_frame_num_minus4: 4,
            log2_max_poc_lsb_minus4: 4,
            max_num_ref_frames: 1,
            profile_idc: 122,
            chroma_format_idc: 2,
            separate_colour_plane: false,
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
