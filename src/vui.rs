//! §E.1 — VUI (Video Usability Information) parameters parsing.
//!
//! VUI parameters ride in the SPS when `vui_parameters_present_flag` is
//! set. The structure is a chain of `if(flag)` blocks, each of which
//! only appears when an earlier flag bit is 1, so the parser walks in
//! lock-step with the spec's syntax table from §E.1.1. The embedded
//! `hrd_parameters()` structure (§E.1.2) lives in a nested module.
//!
//! Semantics references used for range checks and defaults:
//!   * §E.2.1 — VUI parameters semantics.
//!   * §E.2.2 — HRD parameters semantics (in particular
//!     `cpb_cnt_minus1 ∈ 0..=31`).

use crate::bitstream::{BitError, BitReader};

/// Value of `aspect_ratio_idc` that signals the Extended_SAR encoding
/// where `sar_width` / `sar_height` follow as 16-bit fields. §E.2.1
/// Table E-1.
pub const EXTENDED_SAR: u8 = 255;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum VuiError {
    #[error("bitstream read failed: {0}")]
    Bitstream(#[from] BitError),
    #[error(
        "aspect_ratio_idc=255 (Extended_SAR) requires 16+16 trailing bits — got truncated input"
    )]
    ExtendedSarTruncated,
    #[error("invalid HRD cpb_cnt_minus1 (got {0}, max 31)")]
    HrdCpbCountOutOfRange(u32),
}

// `thiserror`'s `#[from]` above generates `impl From<BitError> for VuiError`
// automatically — no manual impl is needed.

/// §E.2.1 aspect_ratio_info — conditional block starting at
/// `aspect_ratio_info_present_flag`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AspectRatioInfo {
    pub aspect_ratio_idc: u8,
    /// Present only when `aspect_ratio_idc == EXTENDED_SAR` (255).
    pub extended_sar: Option<(u16, u16)>, // (sar_width, sar_height)
}

/// §E.2.1 video_signal_type block (when `video_signal_type_present_flag` == 1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoSignalType {
    /// `video_format` — u(3). 5 is the "unspecified" default when the
    /// whole block is absent.
    pub video_format: u8,
    pub video_full_range_flag: bool,
    pub colour_description: Option<ColourDescription>,
}

/// §E.2.1 colour_description block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColourDescription {
    pub colour_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
}

/// §E.2.1 chroma_loc_info block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChromaLocInfo {
    /// `chroma_sample_loc_type_top_field` — ue(v), range 0..=5 (§E.2.1).
    pub chroma_sample_loc_type_top_field: u32,
    pub chroma_sample_loc_type_bottom_field: u32,
}

/// §E.2.1 timing_info block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimingInfo {
    pub num_units_in_tick: u32,
    pub time_scale: u32,
    pub fixed_frame_rate_flag: bool,
}

/// §E.1.2 hrd_parameters() — common NAL and VCL HRD block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HrdParameters {
    /// `cpb_cnt_minus1` — ue(v), §E.2.2 range 0..=31.
    pub cpb_cnt_minus1: u32,
    pub bit_rate_scale: u8,
    pub cpb_size_scale: u8,
    /// One entry per SchedSelIdx in 0..=cpb_cnt_minus1.
    pub bit_rate_value_minus1: Vec<u32>,
    pub cpb_size_value_minus1: Vec<u32>,
    pub cbr_flag: Vec<bool>,
    pub initial_cpb_removal_delay_length_minus1: u8,
    pub cpb_removal_delay_length_minus1: u8,
    pub dpb_output_delay_length_minus1: u8,
    pub time_offset_length: u8,
}

/// §E.2.1 bitstream_restriction block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitstreamRestriction {
    pub motion_vectors_over_pic_boundaries_flag: bool,
    pub max_bytes_per_pic_denom: u32,
    pub max_bits_per_mb_denom: u32,
    pub log2_max_mv_length_horizontal: u32,
    pub log2_max_mv_length_vertical: u32,
    pub max_num_reorder_frames: u32,
    pub max_dec_frame_buffering: u32,
}

/// §E.1.1 vui_parameters(). Each `Option` maps to the
/// corresponding `*_present_flag` in the syntax table.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VuiParameters {
    pub aspect_ratio: Option<AspectRatioInfo>,
    /// Present only when `overscan_info_present_flag` == 1.
    pub overscan_appropriate_flag: Option<bool>,
    pub video_signal_type: Option<VideoSignalType>,
    pub chroma_loc_info: Option<ChromaLocInfo>,
    pub timing_info: Option<TimingInfo>,
    pub nal_hrd_parameters: Option<HrdParameters>,
    pub vcl_hrd_parameters: Option<HrdParameters>,
    /// `low_delay_hrd_flag` — only present when one of the HRD blocks is.
    pub low_delay_hrd_flag: Option<bool>,
    pub pic_struct_present_flag: bool,
    pub bitstream_restriction: Option<BitstreamRestriction>,
}

impl VuiParameters {
    /// §E.1.1 — parse a `vui_parameters()` structure from the current
    /// cursor position. Does NOT consume rbsp_trailing_bits — the
    /// caller (SPS parser) handles that.
    pub fn parse(r: &mut BitReader<'_>) -> Result<Self, VuiError> {
        let mut out = VuiParameters::default();

        // §E.1.1 aspect_ratio_info_present_flag u(1)
        let aspect_ratio_info_present_flag = r.u(1)? == 1;
        if aspect_ratio_info_present_flag {
            // aspect_ratio_idc u(8)
            let aspect_ratio_idc = r.u(8)? as u8;
            let extended_sar = if aspect_ratio_idc == EXTENDED_SAR {
                // sar_width u(16), sar_height u(16)
                if r.bits_remaining() < 32 {
                    return Err(VuiError::ExtendedSarTruncated);
                }
                let sar_width = r.u(16)? as u16;
                let sar_height = r.u(16)? as u16;
                Some((sar_width, sar_height))
            } else {
                None
            };
            out.aspect_ratio = Some(AspectRatioInfo {
                aspect_ratio_idc,
                extended_sar,
            });
        }

        // §E.1.1 overscan_info_present_flag u(1)
        let overscan_info_present_flag = r.u(1)? == 1;
        if overscan_info_present_flag {
            // overscan_appropriate_flag u(1)
            out.overscan_appropriate_flag = Some(r.u(1)? == 1);
        }

        // §E.1.1 video_signal_type_present_flag u(1)
        let video_signal_type_present_flag = r.u(1)? == 1;
        if video_signal_type_present_flag {
            let video_format = r.u(3)? as u8;
            let video_full_range_flag = r.u(1)? == 1;
            let colour_description_present_flag = r.u(1)? == 1;
            let colour_description = if colour_description_present_flag {
                Some(ColourDescription {
                    colour_primaries: r.u(8)? as u8,
                    transfer_characteristics: r.u(8)? as u8,
                    matrix_coefficients: r.u(8)? as u8,
                })
            } else {
                None
            };
            out.video_signal_type = Some(VideoSignalType {
                video_format,
                video_full_range_flag,
                colour_description,
            });
        }

        // §E.1.1 chroma_loc_info_present_flag u(1)
        let chroma_loc_info_present_flag = r.u(1)? == 1;
        if chroma_loc_info_present_flag {
            out.chroma_loc_info = Some(ChromaLocInfo {
                chroma_sample_loc_type_top_field: r.ue()?,
                chroma_sample_loc_type_bottom_field: r.ue()?,
            });
        }

        // §E.1.1 timing_info_present_flag u(1)
        let timing_info_present_flag = r.u(1)? == 1;
        if timing_info_present_flag {
            out.timing_info = Some(TimingInfo {
                num_units_in_tick: r.u(32)?,
                time_scale: r.u(32)?,
                fixed_frame_rate_flag: r.u(1)? == 1,
            });
        }

        // §E.1.1 nal_hrd_parameters_present_flag u(1)
        let nal_hrd_parameters_present_flag = r.u(1)? == 1;
        if nal_hrd_parameters_present_flag {
            out.nal_hrd_parameters = Some(HrdParameters::parse(r)?);
        }

        // §E.1.1 vcl_hrd_parameters_present_flag u(1)
        let vcl_hrd_parameters_present_flag = r.u(1)? == 1;
        if vcl_hrd_parameters_present_flag {
            out.vcl_hrd_parameters = Some(HrdParameters::parse(r)?);
        }

        // §E.1.1: low_delay_hrd_flag u(1) only when either HRD block present.
        if nal_hrd_parameters_present_flag || vcl_hrd_parameters_present_flag {
            out.low_delay_hrd_flag = Some(r.u(1)? == 1);
        }

        // §E.1.1 pic_struct_present_flag u(1)
        out.pic_struct_present_flag = r.u(1)? == 1;

        // §E.1.1 bitstream_restriction_flag u(1)
        let bitstream_restriction_flag = r.u(1)? == 1;
        if bitstream_restriction_flag {
            out.bitstream_restriction = Some(BitstreamRestriction {
                motion_vectors_over_pic_boundaries_flag: r.u(1)? == 1,
                max_bytes_per_pic_denom: r.ue()?,
                max_bits_per_mb_denom: r.ue()?,
                log2_max_mv_length_horizontal: r.ue()?,
                log2_max_mv_length_vertical: r.ue()?,
                max_num_reorder_frames: r.ue()?,
                max_dec_frame_buffering: r.ue()?,
            });
        }

        Ok(out)
    }
}

impl HrdParameters {
    /// §E.1.2 — parse an `hrd_parameters()` structure from the current
    /// cursor position.
    pub fn parse(r: &mut BitReader<'_>) -> Result<Self, VuiError> {
        // §E.1.2 cpb_cnt_minus1 ue(v). §E.2.2 restricts range to 0..=31.
        let cpb_cnt_minus1 = r.ue()?;
        if cpb_cnt_minus1 > 31 {
            return Err(VuiError::HrdCpbCountOutOfRange(cpb_cnt_minus1));
        }
        // bit_rate_scale u(4), cpb_size_scale u(4)
        let bit_rate_scale = r.u(4)? as u8;
        let cpb_size_scale = r.u(4)? as u8;

        let count = (cpb_cnt_minus1 as usize) + 1;
        let mut bit_rate_value_minus1 = Vec::with_capacity(count);
        let mut cpb_size_value_minus1 = Vec::with_capacity(count);
        let mut cbr_flag = Vec::with_capacity(count);
        for _ in 0..count {
            // bit_rate_value_minus1[SchedSelIdx] ue(v)
            bit_rate_value_minus1.push(r.ue()?);
            // cpb_size_value_minus1[SchedSelIdx] ue(v)
            cpb_size_value_minus1.push(r.ue()?);
            // cbr_flag[SchedSelIdx] u(1)
            cbr_flag.push(r.u(1)? == 1);
        }

        // u(5) each, §E.1.2.
        let initial_cpb_removal_delay_length_minus1 = r.u(5)? as u8;
        let cpb_removal_delay_length_minus1 = r.u(5)? as u8;
        let dpb_output_delay_length_minus1 = r.u(5)? as u8;
        let time_offset_length = r.u(5)? as u8;

        Ok(HrdParameters {
            cpb_cnt_minus1,
            bit_rate_scale,
            cpb_size_scale,
            bit_rate_value_minus1,
            cpb_size_value_minus1,
            cbr_flag,
            initial_cpb_removal_delay_length_minus1,
            cpb_removal_delay_length_minus1,
            dpb_output_delay_length_minus1,
            time_offset_length,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack a list of `(value, bit_width)` pairs into an MSB-first
    /// byte buffer. Used to assemble crafted bitstreams for tests.
    fn pack_bits(bits: &[(u32, u32)]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        let mut acc: u64 = 0;
        let mut nbits: u32 = 0;
        for &(val, w) in bits {
            assert!(w <= 32);
            let masked = if w == 32 {
                val
            } else {
                val & ((1u32 << w) - 1)
            };
            acc = (acc << w) | masked as u64;
            nbits += w;
            while nbits >= 8 {
                nbits -= 8;
                let byte = ((acc >> nbits) & 0xFF) as u8;
                out.push(byte);
            }
        }
        if nbits > 0 {
            let byte = ((acc << (8 - nbits)) & 0xFF) as u8;
            out.push(byte);
        }
        out
    }

    /// Encode ue(v) for codenum k as (bits, width).
    /// §9.1: k → binary (k+1) prefixed with floor(log2(k+1)) zeros.
    fn ue_bits(k: u32) -> (u32, u32) {
        let m = k + 1;
        let leading = 31 - m.leading_zeros();
        let total = 2 * leading + 1;
        (m, total)
    }

    // -----------------------------------------------------------------
    // VUI
    // -----------------------------------------------------------------

    #[test]
    fn empty_vui_all_flags_zero() {
        // 7 flags = 0 before pic_struct_present_flag, then
        // pic_struct_present_flag = 0, then bitstream_restriction_flag = 0.
        // Order of flags (u(1) each):
        //   aspect_ratio_info_present_flag
        //   overscan_info_present_flag
        //   video_signal_type_present_flag
        //   chroma_loc_info_present_flag
        //   timing_info_present_flag
        //   nal_hrd_parameters_present_flag
        //   vcl_hrd_parameters_present_flag
        //   (low_delay_hrd_flag only when either HRD present — skipped here)
        //   pic_struct_present_flag
        //   bitstream_restriction_flag
        // Total 9 zero bits → one byte 0x00.
        let bytes = pack_bits(&[(0, 9)]);
        let mut r = BitReader::new(&bytes);
        let vui = VuiParameters::parse(&mut r).unwrap();
        assert_eq!(vui, VuiParameters::default());
    }

    #[test]
    fn timing_info_parsed() {
        // Flags: aspect=0 overscan=0 video=0 chroma_loc=0 timing=1 → bits 00001
        //   num_units_in_tick u(32) = 1
        //   time_scale u(32) = 50
        //   fixed_frame_rate_flag u(1) = 1
        // After timing block: nal_hrd=0 vcl_hrd=0 pic_struct=0 br=0 → 4 zeros.
        let bits = vec![
            (0, 1), // aspect
            (0, 1), // overscan
            (0, 1), // video
            (0, 1), // chroma
            (1, 1), // timing present
            (1, 32),
            (50, 32),
            (1, 1), // fixed_frame_rate_flag
            (0, 1), // nal_hrd
            (0, 1), // vcl_hrd
            (0, 1), // pic_struct
            (0, 1), // bitstream_restriction
        ];
        let bytes = pack_bits(&bits);
        let mut r = BitReader::new(&bytes);
        let vui = VuiParameters::parse(&mut r).unwrap();
        let t = vui.timing_info.as_ref().unwrap();
        assert_eq!(t.num_units_in_tick, 1);
        assert_eq!(t.time_scale, 50);
        assert!(t.fixed_frame_rate_flag);
        assert_eq!(vui.aspect_ratio, None);
        assert!(!vui.pic_struct_present_flag);
        assert_eq!(vui.low_delay_hrd_flag, None);
    }

    #[test]
    fn aspect_ratio_idc_1_no_extended() {
        // aspect=1, aspect_ratio_idc=1 (1:1). Remaining 6 flags = 0.
        let bits = vec![
            (1, 1), // aspect_ratio_info_present_flag
            (1, 8), // aspect_ratio_idc = 1
            (0, 1), // overscan
            (0, 1), // video
            (0, 1), // chroma
            (0, 1), // timing
            (0, 1), // nal_hrd
            (0, 1), // vcl_hrd
            (0, 1), // pic_struct
            (0, 1), // bitstream_restriction
        ];
        let bytes = pack_bits(&bits);
        let mut r = BitReader::new(&bytes);
        let vui = VuiParameters::parse(&mut r).unwrap();
        let ar = vui.aspect_ratio.as_ref().unwrap();
        assert_eq!(ar.aspect_ratio_idc, 1);
        assert_eq!(ar.extended_sar, None);
    }

    #[test]
    fn aspect_ratio_extended_sar() {
        // aspect=1, aspect_ratio_idc=255, sar=(16,9). Remaining 6 flags = 0.
        let bits = vec![
            (1, 1),
            (255, 8),
            (16, 16),
            (9, 16),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
        ];
        let bytes = pack_bits(&bits);
        let mut r = BitReader::new(&bytes);
        let vui = VuiParameters::parse(&mut r).unwrap();
        let ar = vui.aspect_ratio.as_ref().unwrap();
        assert_eq!(ar.aspect_ratio_idc, 255);
        assert_eq!(ar.extended_sar, Some((16, 9)));
    }

    #[test]
    fn bitstream_restriction_parsed() {
        // Flip bitstream_restriction_flag and fill its block with
        // known ue(v) values.
        let mut bits: Vec<(u32, u32)> = vec![
            (0, 1), // aspect
            (0, 1), // overscan
            (0, 1), // video
            (0, 1), // chroma
            (0, 1), // timing
            (0, 1), // nal_hrd
            (0, 1), // vcl_hrd
            (0, 1), // pic_struct
            (1, 1), // bitstream_restriction_flag
            (1, 1), // motion_vectors_over_pic_boundaries_flag
        ];
        // ue(v) values: max_bytes_per_pic_denom=2,
        //   max_bits_per_mb_denom=1, log2_max_mv_length_horizontal=16,
        //   log2_max_mv_length_vertical=16, max_num_reorder_frames=0,
        //   max_dec_frame_buffering=4.
        for k in [2u32, 1, 16, 16, 0, 4] {
            bits.push(ue_bits(k));
        }
        let bytes = pack_bits(&bits);
        let mut r = BitReader::new(&bytes);
        let vui = VuiParameters::parse(&mut r).unwrap();
        let br = vui.bitstream_restriction.as_ref().unwrap();
        assert!(br.motion_vectors_over_pic_boundaries_flag);
        assert_eq!(br.max_bytes_per_pic_denom, 2);
        assert_eq!(br.max_bits_per_mb_denom, 1);
        assert_eq!(br.log2_max_mv_length_horizontal, 16);
        assert_eq!(br.log2_max_mv_length_vertical, 16);
        assert_eq!(br.max_num_reorder_frames, 0);
        assert_eq!(br.max_dec_frame_buffering, 4);
    }

    #[test]
    fn vui_with_nal_hrd_then_low_delay_flag() {
        // Minimal HRD: cpb_cnt_minus1 = 0 (single CPB). Build the full
        // bitstream for VUI containing only a NAL HRD block.
        // Flag layout up to nal_hrd_present:
        //   aspect=0 overscan=0 video=0 chroma=0 timing=0 nal_hrd=1 → 000001
        let mut bits: Vec<(u32, u32)> = vec![
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (1, 1), // nal_hrd present
        ];
        // hrd_parameters():
        //   cpb_cnt_minus1 = 0 → ue code "1"
        bits.push(ue_bits(0));
        //   bit_rate_scale u(4) = 4
        bits.push((4, 4));
        //   cpb_size_scale u(4) = 5
        bits.push((5, 4));
        //   one SchedSelIdx (=0):
        //     bit_rate_value_minus1 ue(v) = 3    → ue_bits(3)
        //     cpb_size_value_minus1 ue(v) = 10   → ue_bits(10)
        //     cbr_flag u(1) = 1
        bits.push(ue_bits(3));
        bits.push(ue_bits(10));
        bits.push((1, 1));
        //   initial_cpb_removal_delay_length_minus1 u(5) = 23
        //   cpb_removal_delay_length_minus1         u(5) = 23
        //   dpb_output_delay_length_minus1          u(5) = 23
        //   time_offset_length                      u(5) = 24
        bits.push((23, 5));
        bits.push((23, 5));
        bits.push((23, 5));
        bits.push((24, 5));
        // After nal HRD: vcl_hrd_present=0, low_delay_hrd_flag=1,
        // pic_struct_present_flag=0, bitstream_restriction_flag=0
        bits.push((0, 1)); // vcl_hrd
        bits.push((1, 1)); // low_delay_hrd_flag
        bits.push((0, 1)); // pic_struct
        bits.push((0, 1)); // bitstream_restriction

        let bytes = pack_bits(&bits);
        let mut r = BitReader::new(&bytes);
        let vui = VuiParameters::parse(&mut r).unwrap();
        let hrd = vui.nal_hrd_parameters.as_ref().unwrap();
        assert_eq!(hrd.cpb_cnt_minus1, 0);
        assert_eq!(hrd.bit_rate_scale, 4);
        assert_eq!(hrd.cpb_size_scale, 5);
        assert_eq!(hrd.bit_rate_value_minus1, vec![3]);
        assert_eq!(hrd.cpb_size_value_minus1, vec![10]);
        assert_eq!(hrd.cbr_flag, vec![true]);
        assert_eq!(hrd.initial_cpb_removal_delay_length_minus1, 23);
        assert_eq!(hrd.cpb_removal_delay_length_minus1, 23);
        assert_eq!(hrd.dpb_output_delay_length_minus1, 23);
        assert_eq!(hrd.time_offset_length, 24);
        assert!(vui.vcl_hrd_parameters.is_none());
        assert_eq!(vui.low_delay_hrd_flag, Some(true));
    }

    // -----------------------------------------------------------------
    // HRD standalone
    // -----------------------------------------------------------------

    #[test]
    fn hrd_parameters_single_cpb() {
        // cpb_cnt_minus1 = 0; bit_rate_scale = 2, cpb_size_scale = 3.
        // One SchedSelIdx entry: bit_rate_value_minus1=7, cpb_size_value_minus1=15, cbr=0.
        // Lengths: 23,23,23,24.
        let mut bits: Vec<(u32, u32)> = Vec::new();
        bits.push(ue_bits(0)); // cpb_cnt_minus1
        bits.push((2, 4)); // bit_rate_scale
        bits.push((3, 4)); // cpb_size_scale
        bits.push(ue_bits(7)); // bit_rate_value_minus1
        bits.push(ue_bits(15)); // cpb_size_value_minus1
        bits.push((0, 1)); // cbr_flag
        bits.push((23, 5));
        bits.push((23, 5));
        bits.push((23, 5));
        bits.push((24, 5));
        let bytes = pack_bits(&bits);
        let mut r = BitReader::new(&bytes);
        let hrd = HrdParameters::parse(&mut r).unwrap();
        assert_eq!(hrd.cpb_cnt_minus1, 0);
        assert_eq!(hrd.bit_rate_scale, 2);
        assert_eq!(hrd.cpb_size_scale, 3);
        assert_eq!(hrd.bit_rate_value_minus1, vec![7]);
        assert_eq!(hrd.cpb_size_value_minus1, vec![15]);
        assert_eq!(hrd.cbr_flag, vec![false]);
        assert_eq!(hrd.initial_cpb_removal_delay_length_minus1, 23);
        assert_eq!(hrd.cpb_removal_delay_length_minus1, 23);
        assert_eq!(hrd.dpb_output_delay_length_minus1, 23);
        assert_eq!(hrd.time_offset_length, 24);
    }

    #[test]
    fn hrd_cpb_cnt_out_of_range_rejected() {
        // cpb_cnt_minus1 = 32 → out-of-range. ue_bits(32) is a valid ue code.
        let mut bits: Vec<(u32, u32)> = Vec::new();
        bits.push(ue_bits(32));
        let bytes = pack_bits(&bits);
        let mut r = BitReader::new(&bytes);
        let err = HrdParameters::parse(&mut r).unwrap_err();
        assert_eq!(err, VuiError::HrdCpbCountOutOfRange(32));
    }

    #[test]
    fn hrd_two_sched_sel_idx() {
        // cpb_cnt_minus1 = 1 → two CPBs.
        let mut bits: Vec<(u32, u32)> = Vec::new();
        bits.push(ue_bits(1)); // cpb_cnt_minus1 = 1
        bits.push((0, 4));
        bits.push((0, 4));
        // idx 0
        bits.push(ue_bits(1));
        bits.push(ue_bits(2));
        bits.push((1, 1));
        // idx 1
        bits.push(ue_bits(3));
        bits.push(ue_bits(4));
        bits.push((0, 1));
        // lengths
        bits.push((20, 5));
        bits.push((20, 5));
        bits.push((20, 5));
        bits.push((20, 5));
        let bytes = pack_bits(&bits);
        let mut r = BitReader::new(&bytes);
        let hrd = HrdParameters::parse(&mut r).unwrap();
        assert_eq!(hrd.cpb_cnt_minus1, 1);
        assert_eq!(hrd.bit_rate_value_minus1, vec![1, 3]);
        assert_eq!(hrd.cpb_size_value_minus1, vec![2, 4]);
        assert_eq!(hrd.cbr_flag, vec![true, false]);
    }

    #[test]
    fn aspect_ratio_extended_sar_truncated() {
        // aspect=1, idc=255, only 10 bits of SAR data → truncated.
        let bits = vec![
            (1, 1),
            (255, 8),
            (0, 10), // partial
        ];
        let bytes = pack_bits(&bits);
        let mut r = BitReader::new(&bytes);
        let err = VuiParameters::parse(&mut r).unwrap_err();
        assert_eq!(err, VuiError::ExtendedSarTruncated);
    }
}
