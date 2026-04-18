//! Sequence Parameter Set (SPS) parsing — ITU-T H.264 §7.3.2.1.1.
//!
//! Only the fields needed to drive the decoder and report stream metadata
//! are stored. The VUI syntax is walked so the `bitstream_restriction`
//! block at its tail can be reached; within VUI, aspect ratio / video
//! signal / HRD / timing are parsed only far enough to advance the bit
//! position, while [`VuiParameters::max_num_reorder_frames`] and
//! [`VuiParameters::max_dec_frame_buffering`] are extracted.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::nal::NalHeader;

/// Parsed SPS. Field names follow the spec exactly.
#[derive(Clone, Debug)]
pub struct Sps {
    pub profile_idc: u8,
    pub constraint_set0_flag: bool,
    pub constraint_set1_flag: bool,
    pub constraint_set2_flag: bool,
    pub constraint_set3_flag: bool,
    pub constraint_set4_flag: bool,
    pub constraint_set5_flag: bool,
    pub level_idc: u8,
    pub seq_parameter_set_id: u32,

    // Profile-specific (present only for high profiles).
    pub chroma_format_idc: u32,
    pub separate_colour_plane_flag: bool,
    pub bit_depth_luma_minus8: u32,
    pub bit_depth_chroma_minus8: u32,
    pub qpprime_y_zero_transform_bypass_flag: bool,
    /// Set when the SPS carries custom scaling-list matrices (§7.4.2.1.1).
    pub seq_scaling_matrix_present_flag: bool,
    /// 4×4 scaling lists captured from the SPS. Each entry is `Some`
    /// when `seq_scaling_list_present_flag[i] = 1` AND a non-default
    /// matrix was parsed; `None` means "inherit per Table 7-2 fallback
    /// rule A" (fall back to the default matrix). The "use default
    /// matrix" flag is distinguished from "not present" by carrying
    /// the per-i16-positions default matrix literally (see
    /// [`crate::scaling_list::DEFAULT_4X4_INTRA`] /
    /// [`crate::scaling_list::DEFAULT_4X4_INTER`]).
    pub seq_scaling_list_4x4: [Option<[i16; 16]>; 6],
    /// 8×8 scaling lists from the SPS. Indexed 0 = intra-Y, 1 = inter-Y
    /// for 4:2:0 (chroma 8×8 is only present for 4:4:4 — the extra
    /// slots are captured but unused in this decoder).
    pub seq_scaling_list_8x8: [Option<[i16; 64]>; 6],

    pub log2_max_frame_num_minus4: u32,
    pub pic_order_cnt_type: u32,
    pub log2_max_pic_order_cnt_lsb_minus4: u32,
    // pic_order_cnt_type == 1
    pub delta_pic_order_always_zero_flag: bool,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub num_ref_frames_in_pic_order_cnt_cycle: u32,
    pub offset_for_ref_frame: Vec<i32>,

    pub max_num_ref_frames: u32,
    pub gaps_in_frame_num_value_allowed_flag: bool,
    pub pic_width_in_mbs_minus1: u32,
    pub pic_height_in_map_units_minus1: u32,
    pub frame_mbs_only_flag: bool,
    pub mb_adaptive_frame_field_flag: bool,
    pub direct_8x8_inference_flag: bool,

    pub frame_cropping_flag: bool,
    pub frame_crop_left_offset: u32,
    pub frame_crop_right_offset: u32,
    pub frame_crop_top_offset: u32,
    pub frame_crop_bottom_offset: u32,

    pub vui_parameters_present_flag: bool,
    /// Parsed VUI parameters (§E.1.1) when `vui_parameters_present_flag`
    /// is set. `None` otherwise. Currently only the fields the decoder
    /// actually uses (bitstream restriction) are captured; other VUI
    /// sub-structures are parsed just enough to reach the tail.
    pub vui: Option<VuiParameters>,
}

/// Selected VUI parameters (§E.1.1). Most of VUI is discarded during
/// parsing; this struct carries only the fields the decoder needs.
#[derive(Clone, Debug, Default)]
pub struct VuiParameters {
    /// `bitstream_restriction_flag` — true when the block beyond it
    /// (including the reorder / DPB sizing fields below) is present.
    pub bitstream_restriction_flag: bool,
    /// `max_num_reorder_frames` (§E.2.1). Drives the DPB output reorder
    /// window so the decoder can emit pictures in POC order without
    /// over-buffering when the encoder signals a tighter bound than
    /// `max_num_ref_frames`.
    pub max_num_reorder_frames: Option<u32>,
    /// `max_dec_frame_buffering` (§E.2.1). Upper bound on DPB size the
    /// encoder promises.
    pub max_dec_frame_buffering: Option<u32>,
}

impl Sps {
    /// Width in macroblocks.
    pub fn pic_width_in_mbs(&self) -> u32 {
        self.pic_width_in_mbs_minus1 + 1
    }

    /// Height in map units.
    pub fn pic_height_in_map_units(&self) -> u32 {
        self.pic_height_in_map_units_minus1 + 1
    }

    /// Width in macroblocks * 16.
    pub fn coded_width(&self) -> u32 {
        self.pic_width_in_mbs() * 16
    }

    /// Coded height in luma samples (accounts for `frame_mbs_only_flag`).
    pub fn coded_height(&self) -> u32 {
        let map_units = self.pic_height_in_map_units();
        let factor = if self.frame_mbs_only_flag { 1 } else { 2 };
        map_units * 16 * factor
    }

    /// Final visible width / height after `frame_cropping`. Returns `(w, h)`
    /// in luma samples (`§6.4.1`). Honours `chroma_format_idc` for the
    /// crop unit derivation.
    pub fn visible_size(&self) -> (u32, u32) {
        let (sub_w, sub_h) = match self.chroma_format_idc {
            0 => (1, 1),
            1 => (2, 2), // 4:2:0
            2 => (2, 1), // 4:2:2
            _ => (1, 1), // 4:4:4
        };
        let crop_x = sub_w
            * (self
                .frame_crop_left_offset
                .saturating_add(self.frame_crop_right_offset));
        let frame_mbs = if self.frame_mbs_only_flag { 1 } else { 2 };
        let crop_y = sub_h
            * frame_mbs
            * (self
                .frame_crop_top_offset
                .saturating_add(self.frame_crop_bottom_offset));
        (
            self.coded_width().saturating_sub(crop_x),
            self.coded_height().saturating_sub(crop_y),
        )
    }
}

/// Parse a `seq_parameter_set_rbsp()` body. `rbsp` must already have had
/// emulation-prevention bytes stripped, and `header` is the NAL header
/// preceding the RBSP.
pub fn parse_sps(header: &NalHeader, rbsp: &[u8]) -> Result<Sps> {
    if header.nal_unit_type != crate::nal::NalUnitType::Sps {
        return Err(Error::invalid("h264 sps: NAL header is not SPS"));
    }
    if rbsp.len() < 3 {
        return Err(Error::invalid("h264 sps: RBSP too short"));
    }
    let profile_idc = rbsp[0];
    let cs = rbsp[1];
    let level_idc = rbsp[2];
    let mut br = BitReader::new(&rbsp[3..]);
    let seq_parameter_set_id = br.read_ue()?;
    if seq_parameter_set_id > 31 {
        return Err(Error::invalid(format!(
            "h264 sps: seq_parameter_set_id {seq_parameter_set_id} > 31"
        )));
    }

    let mut chroma_format_idc = 1; // 4:2:0 default
    let mut separate_colour_plane_flag = false;
    let mut bit_depth_luma_minus8 = 0;
    let mut bit_depth_chroma_minus8 = 0;
    let mut qpprime_y_zero_transform_bypass_flag = false;
    let mut seq_scaling_matrix_present_flag = false;
    let mut seq_scaling_list_4x4: [Option<[i16; 16]>; 6] = [None; 6];
    let mut seq_scaling_list_8x8: [Option<[i16; 64]>; 6] = [None; 6];

    let high_profile = matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    );
    if high_profile {
        chroma_format_idc = br.read_ue()?;
        if chroma_format_idc > 3 {
            return Err(Error::invalid("h264 sps: chroma_format_idc > 3"));
        }
        if chroma_format_idc == 3 {
            separate_colour_plane_flag = br.read_flag()?;
        }
        bit_depth_luma_minus8 = br.read_ue()?;
        bit_depth_chroma_minus8 = br.read_ue()?;
        qpprime_y_zero_transform_bypass_flag = br.read_flag()?;
        seq_scaling_matrix_present_flag = br.read_flag()?;
        if seq_scaling_matrix_present_flag {
            let count = if chroma_format_idc != 3 { 8 } else { 12 };
            for i in 0..count {
                let present = br.read_flag()?;
                if present {
                    if i < 6 {
                        let (mat, use_default) = parse_scaling_list_4x4(&mut br)?;
                        // §7.4.2.1.1.1: `useDefaultScalingMatrixFlag`
                        // means "use the spec's default matrix (Table
                        // 7-3)". Resolving that default here lets the
                        // downstream fallback-rule logic treat the
                        // matrix uniformly.
                        seq_scaling_list_4x4[i] = Some(if use_default {
                            crate::scaling_list::default_4x4_for_index(i)
                        } else {
                            mat
                        });
                    } else {
                        let (mat, use_default) = parse_scaling_list_8x8(&mut br)?;
                        let slot = i - 6;
                        seq_scaling_list_8x8[slot] = Some(if use_default {
                            crate::scaling_list::default_8x8_for_index(slot)
                        } else {
                            mat
                        });
                    }
                }
            }
        }
    }

    let log2_max_frame_num_minus4 = br.read_ue()?;
    if log2_max_frame_num_minus4 > 12 {
        return Err(Error::invalid("h264 sps: log2_max_frame_num_minus4 > 12"));
    }
    let pic_order_cnt_type = br.read_ue()?;
    if pic_order_cnt_type > 2 {
        return Err(Error::invalid("h264 sps: pic_order_cnt_type > 2"));
    }

    let mut log2_max_pic_order_cnt_lsb_minus4 = 0;
    let mut delta_pic_order_always_zero_flag = false;
    let mut offset_for_non_ref_pic = 0;
    let mut offset_for_top_to_bottom_field = 0;
    let mut num_ref_frames_in_pic_order_cnt_cycle = 0;
    let mut offset_for_ref_frame = Vec::new();

    match pic_order_cnt_type {
        0 => {
            log2_max_pic_order_cnt_lsb_minus4 = br.read_ue()?;
            if log2_max_pic_order_cnt_lsb_minus4 > 12 {
                return Err(Error::invalid(
                    "h264 sps: log2_max_pic_order_cnt_lsb_minus4 > 12",
                ));
            }
        }
        1 => {
            delta_pic_order_always_zero_flag = br.read_flag()?;
            offset_for_non_ref_pic = br.read_se()?;
            offset_for_top_to_bottom_field = br.read_se()?;
            num_ref_frames_in_pic_order_cnt_cycle = br.read_ue()?;
            if num_ref_frames_in_pic_order_cnt_cycle > 255 {
                return Err(Error::invalid(
                    "h264 sps: num_ref_frames_in_pic_order_cnt_cycle > 255",
                ));
            }
            offset_for_ref_frame.reserve_exact(num_ref_frames_in_pic_order_cnt_cycle as usize);
            for _ in 0..num_ref_frames_in_pic_order_cnt_cycle {
                offset_for_ref_frame.push(br.read_se()?);
            }
        }
        _ => {}
    }

    let max_num_ref_frames = br.read_ue()?;
    if max_num_ref_frames > 16 {
        return Err(Error::invalid("h264 sps: max_num_ref_frames > 16"));
    }
    let gaps_in_frame_num_value_allowed_flag = br.read_flag()?;
    let pic_width_in_mbs_minus1 = br.read_ue()?;
    let pic_height_in_map_units_minus1 = br.read_ue()?;
    let frame_mbs_only_flag = br.read_flag()?;
    let mb_adaptive_frame_field_flag = if !frame_mbs_only_flag {
        br.read_flag()?
    } else {
        false
    };
    let direct_8x8_inference_flag = br.read_flag()?;
    let frame_cropping_flag = br.read_flag()?;
    let (
        frame_crop_left_offset,
        frame_crop_right_offset,
        frame_crop_top_offset,
        frame_crop_bottom_offset,
    ) = if frame_cropping_flag {
        (br.read_ue()?, br.read_ue()?, br.read_ue()?, br.read_ue()?)
    } else {
        (0, 0, 0, 0)
    };
    let vui_parameters_present_flag = br.read_flag()?;
    let mut vui: Option<VuiParameters> = None;
    if vui_parameters_present_flag {
        // §E.1.1 — walk the VUI grammar so the `bitstream_restriction`
        // block at its tail can be reached. `parse_vui` is resilient: if
        // any sub-block truncates unexpectedly we just stop and surface
        // whatever was captured.
        vui = Some(parse_vui(&mut br).unwrap_or_default());
    }

    // We deliberately do not enforce rbsp_trailing_bits: we may have stopped
    // mid-VUI. The fields we extracted are sufficient for the decoder.
    let _ = constraints(cs);

    let cs_bits = constraints(cs);
    Ok(Sps {
        profile_idc,
        constraint_set0_flag: cs_bits[0],
        constraint_set1_flag: cs_bits[1],
        constraint_set2_flag: cs_bits[2],
        constraint_set3_flag: cs_bits[3],
        constraint_set4_flag: cs_bits[4],
        constraint_set5_flag: cs_bits[5],
        level_idc,
        seq_parameter_set_id,
        chroma_format_idc,
        separate_colour_plane_flag,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        qpprime_y_zero_transform_bypass_flag,
        seq_scaling_matrix_present_flag,
        seq_scaling_list_4x4,
        seq_scaling_list_8x8,
        log2_max_frame_num_minus4,
        pic_order_cnt_type,
        log2_max_pic_order_cnt_lsb_minus4,
        delta_pic_order_always_zero_flag,
        offset_for_non_ref_pic,
        offset_for_top_to_bottom_field,
        num_ref_frames_in_pic_order_cnt_cycle,
        offset_for_ref_frame,
        max_num_ref_frames,
        gaps_in_frame_num_value_allowed_flag,
        pic_width_in_mbs_minus1,
        pic_height_in_map_units_minus1,
        frame_mbs_only_flag,
        mb_adaptive_frame_field_flag,
        direct_8x8_inference_flag,
        frame_cropping_flag,
        frame_crop_left_offset,
        frame_crop_right_offset,
        frame_crop_top_offset,
        frame_crop_bottom_offset,
        vui_parameters_present_flag,
        vui,
    })
}

fn constraints(byte: u8) -> [bool; 8] {
    let mut out = [false; 8];
    for i in 0..8u8 {
        out[i as usize] = (byte >> (7 - i)) & 1 == 1;
    }
    out
}

/// Walk `vui_parameters()` (§E.1.1) just far enough to capture the
/// `bitstream_restriction` block. The reader's cursor may end up past
/// the end of the true VUI syntax — this is acceptable because the
/// parent SPS parser does not require `rbsp_trailing_bits` alignment,
/// and we deliberately return partial data on truncation.
fn parse_vui(br: &mut BitReader<'_>) -> Result<VuiParameters> {
    let mut out = VuiParameters::default();
    if br.read_flag()? {
        // aspect_ratio_info_present_flag
        let aspect_ratio_idc = br.read_u32(8)?;
        // §E.1.1 Table E-1 — idc == 255 means Extended_SAR.
        if aspect_ratio_idc == 255 {
            let _sar_w = br.read_u32(16)?;
            let _sar_h = br.read_u32(16)?;
        }
    }
    if br.read_flag()? {
        // overscan_info_present_flag
        let _overscan_appropriate_flag = br.read_flag()?;
    }
    if br.read_flag()? {
        // video_signal_type_present_flag
        let _video_format = br.read_u32(3)?;
        let _video_full_range_flag = br.read_flag()?;
        if br.read_flag()? {
            let _colour_primaries = br.read_u32(8)?;
            let _transfer_characteristics = br.read_u32(8)?;
            let _matrix_coefficients = br.read_u32(8)?;
        }
    }
    if br.read_flag()? {
        // chroma_loc_info_present_flag
        let _top = br.read_ue()?;
        let _bot = br.read_ue()?;
    }
    if br.read_flag()? {
        // timing_info_present_flag
        let _num_units_in_tick = br.read_u32(32)?;
        let _time_scale = br.read_u32(32)?;
        let _fixed_frame_rate_flag = br.read_flag()?;
    }
    let nal_hrd_present = br.read_flag()?;
    if nal_hrd_present {
        skip_hrd_parameters(br)?;
    }
    let vcl_hrd_present = br.read_flag()?;
    if vcl_hrd_present {
        skip_hrd_parameters(br)?;
    }
    if nal_hrd_present || vcl_hrd_present {
        let _low_delay_hrd_flag = br.read_flag()?;
    }
    let _pic_struct_present_flag = br.read_flag()?;
    let bitstream_restriction_flag = br.read_flag()?;
    out.bitstream_restriction_flag = bitstream_restriction_flag;
    if bitstream_restriction_flag {
        let _motion_vectors_over_pic_boundaries_flag = br.read_flag()?;
        let _max_bytes_per_pic_denom = br.read_ue()?;
        let _max_bits_per_mb_denom = br.read_ue()?;
        let _log2_max_mv_length_horizontal = br.read_ue()?;
        let _log2_max_mv_length_vertical = br.read_ue()?;
        out.max_num_reorder_frames = Some(br.read_ue()?);
        out.max_dec_frame_buffering = Some(br.read_ue()?);
    }
    Ok(out)
}

fn skip_hrd_parameters(br: &mut BitReader<'_>) -> Result<()> {
    // §E.1.2 `hrd_parameters()`.
    let cpb_cnt_minus1 = br.read_ue()?;
    let _bit_rate_scale = br.read_u32(4)?;
    let _cpb_size_scale = br.read_u32(4)?;
    for _ in 0..=cpb_cnt_minus1 {
        let _bit_rate_value_minus1 = br.read_ue()?;
        let _cpb_size_value_minus1 = br.read_ue()?;
        let _cbr_flag = br.read_flag()?;
    }
    let _initial_cpb_removal_delay_length_minus1 = br.read_u32(5)?;
    let _cpb_removal_delay_length_minus1 = br.read_u32(5)?;
    let _dpb_output_delay_length_minus1 = br.read_u32(5)?;
    let _time_offset_length = br.read_u32(5)?;
    Ok(())
}

/// §7.3.2.1.1.1 `scaling_list(scalingList, sizeOfScalingList,
/// useDefaultScalingMatrixFlag)` for the 4×4 case. Returns `(matrix,
/// use_default)`. The spec stores the list in zig-zag scan order; the
/// output matrix is in raster (row-major) order so downstream dequant
/// can index `row*4 + col` directly.
fn parse_scaling_list_4x4(br: &mut BitReader<'_>) -> Result<([i16; 16], bool)> {
    let mut list = [8i16; 16];
    let mut last_scale: i32 = 8;
    let mut next_scale: i32 = 8;
    let mut use_default = false;
    for j in 0..16u32 {
        if next_scale != 0 {
            let delta = br.read_se()?;
            next_scale = (last_scale + delta + 256) & 0xFF;
            if j == 0 && next_scale == 0 {
                use_default = true;
            }
        }
        let val = if next_scale == 0 { last_scale } else { next_scale };
        // §7.3.2.1.1.1 stores deltas in zig-zag scan order.
        let raster = crate::scaling_list::ZIGZAG_4X4[j as usize];
        list[raster] = val as i16;
        if next_scale != 0 {
            last_scale = next_scale;
        }
    }
    Ok((list, use_default))
}

/// §7.3.2.1.1.1 for the 8×8 case.
fn parse_scaling_list_8x8(br: &mut BitReader<'_>) -> Result<([i16; 64], bool)> {
    let mut list = [8i16; 64];
    let mut last_scale: i32 = 8;
    let mut next_scale: i32 = 8;
    let mut use_default = false;
    for j in 0..64u32 {
        if next_scale != 0 {
            let delta = br.read_se()?;
            next_scale = (last_scale + delta + 256) & 0xFF;
            if j == 0 && next_scale == 0 {
                use_default = true;
            }
        }
        let val = if next_scale == 0 { last_scale } else { next_scale };
        let raster = crate::transform::ZIGZAG_8X8[j as usize];
        list[raster] = val as i16;
        if next_scale != 0 {
            last_scale = next_scale;
        }
    }
    Ok((list, use_default))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal::{extract_rbsp, NalHeader, NalUnitType};

    #[test]
    fn parse_baseline_sps() {
        // SPS from a 128x96 baseline clip (encoded with x264).
        // Header byte: 0x67 (forbidden=0, nal_ref_idc=3, type=7).
        // Body: profile=0x42 (66=baseline), constraints=0xC0, level=0x0A.
        // Then exp-golomb fields.
        // For this unit test we synthesise a minimal SPS body that the parser
        // must accept without VUI.
        // Build:
        //   profile_idc = 66
        //   constraint_set_flags = 11000000 (cs0=cs1=1)
        //   level_idc = 10  (level 1.0)
        //   ue: seq_parameter_set_id = 0     -> "1"
        //   ue: log2_max_frame_num_minus4=0  -> "1"
        //   ue: pic_order_cnt_type=2         -> "011"  (= ue(2))
        //   ue: max_num_ref_frames=1         -> "010"
        //   gaps_in_frame_num=0              -> "0"
        //   ue: pic_width_in_mbs_minus1=7    -> "0001000"  (= ue(7))
        //   ue: pic_height_in_map_units_minus1=5 -> "00110" (= ue(5))
        //   frame_mbs_only=1                 -> "1"
        //   direct_8x8_inference=1           -> "1"
        //   frame_cropping=0                 -> "0"
        //   vui_parameters_present=0         -> "0"
        //   trailing rbsp stop bit + zero alignment
        // Concatenate bits (after the 3-byte preamble):
        //   1 1 011 010 0 0001000 00110 1 1 0 0 1 [00..]
        // = 1 1 0 1 1 0 1 0  0 0 0 0 1 0 0 0  0 0 1 1 0 1 1 0  0 1 0 0
        // Group into bytes:
        //   1101 1010 = 0xDA
        //   0000 1000 = 0x08
        //   0011 0110 = 0x36
        //   0100 ____ = need stop bit + alignment
        //   With trailing 1 then zeros: 0100 1000 = 0x48
        let mut rbsp = vec![66u8, 0xC0, 10];
        rbsp.extend_from_slice(&[0xDA, 0x08, 0x36, 0x48]);
        let h = NalHeader::parse(0x67).unwrap();
        let sps = parse_sps(&h, &rbsp).expect("parse");
        assert_eq!(sps.profile_idc, 66);
        assert_eq!(sps.level_idc, 10);
        assert_eq!(sps.pic_order_cnt_type, 2);
        assert!(sps.frame_mbs_only_flag);
        assert_eq!(sps.coded_width(), 128);
        assert_eq!(sps.coded_height(), 96);
    }

    /// Build a VUI body with only `bitstream_restriction` set and walk it
    /// through [`parse_vui`]. Confirms that `max_num_reorder_frames` /
    /// `max_dec_frame_buffering` are captured and the skip-all path
    /// through the rest of VUI doesn't mis-align the bit cursor.
    /// Round-trip a High Profile SPS with a custom 4×4 scaling list
    /// signalled for slot 0 (Intra-Y) — use the `useDefaultScalingMatrixFlag`
    /// short-circuit (delta_0 = -8 → next_scale = 0) so only one
    /// se(v) needs writing for that slot. The resolver should then
    /// install `Default_4x4_Intra` at slot 0.
    #[test]
    fn parse_sps_with_custom_4x4_scaling_list() {
        use crate::bitwriter::BitWriter;
        let mut bw = BitWriter::new();
        // seq_parameter_set_id = 0
        bw.write_ue(0);
        // chroma_format_idc = 1
        bw.write_ue(1);
        // bit_depth_luma_minus8
        bw.write_ue(0);
        // bit_depth_chroma_minus8
        bw.write_ue(0);
        // qpprime_y_zero_transform_bypass_flag
        bw.write_flag(false);
        // seq_scaling_matrix_present_flag = 1
        bw.write_flag(true);
        // Slot 0 (Intra-Y 4×4) present, delta_0 = -8 → next_scale=0,
        // useDefaultScalingMatrixFlag = true. Parser halts the delta
        // loop there per §7.3.2.1.1.1 (no more deltas read when
        // next_scale == 0).
        bw.write_flag(true);
        bw.write_se(-8);
        // Slots 1..=7 flag = 0 (not signalled).
        for _ in 1..8 {
            bw.write_flag(false);
        }
        // Rest of the SPS — minimal fields.
        bw.write_ue(0); // log2_max_frame_num_minus4
        bw.write_ue(2); // pic_order_cnt_type = 2
        bw.write_ue(1); // max_num_ref_frames
        bw.write_flag(false); // gaps_in_frame_num_value_allowed_flag
        bw.write_ue(7); // pic_width_in_mbs_minus1
        bw.write_ue(5); // pic_height_in_map_units_minus1
        bw.write_flag(true); // frame_mbs_only_flag
        bw.write_flag(true); // direct_8x8_inference
        bw.write_flag(false); // frame_cropping_flag
        bw.write_flag(false); // vui_parameters_present_flag
        bw.write_rbsp_trailing_bits();
        let body = bw.finish();

        // Build a full SPS RBSP with profile=100 (High).
        let mut rbsp = vec![100u8, 0, 30];
        rbsp.extend_from_slice(&body);
        let h = NalHeader::parse(0x67).unwrap();
        let sps = parse_sps(&h, &rbsp).expect("parse");
        assert!(sps.seq_scaling_matrix_present_flag);
        // useDefaultScalingMatrixFlag resolved to Default_4x4_Intra.
        assert_eq!(
            sps.seq_scaling_list_4x4[0],
            Some(crate::scaling_list::DEFAULT_4X4_INTRA)
        );
        for i in 1..6 {
            assert_eq!(sps.seq_scaling_list_4x4[i], None);
        }
    }

    #[test]
    fn parse_vui_bitstream_restriction_only() {
        use crate::bitwriter::BitWriter;
        let mut bw = BitWriter::new();
        // aspect_ratio_info_present_flag = 0
        bw.write_flag(false);
        // overscan_info_present_flag = 0
        bw.write_flag(false);
        // video_signal_type_present_flag = 0
        bw.write_flag(false);
        // chroma_loc_info_present_flag = 0
        bw.write_flag(false);
        // timing_info_present_flag = 0
        bw.write_flag(false);
        // nal_hrd_parameters_present_flag = 0
        bw.write_flag(false);
        // vcl_hrd_parameters_present_flag = 0
        bw.write_flag(false);
        // pic_struct_present_flag = 0
        bw.write_flag(false);
        // bitstream_restriction_flag = 1
        bw.write_flag(true);
        // motion_vectors_over_pic_boundaries_flag
        bw.write_flag(true);
        // max_bytes_per_pic_denom
        bw.write_ue(2);
        // max_bits_per_mb_denom
        bw.write_ue(1);
        // log2_max_mv_length_horizontal
        bw.write_ue(10);
        // log2_max_mv_length_vertical
        bw.write_ue(10);
        // max_num_reorder_frames
        bw.write_ue(2);
        // max_dec_frame_buffering
        bw.write_ue(4);
        let buf = bw.finish();

        let mut br = BitReader::new(&buf);
        let v = parse_vui(&mut br).expect("parse_vui");
        assert!(v.bitstream_restriction_flag);
        assert_eq!(v.max_num_reorder_frames, Some(2));
        assert_eq!(v.max_dec_frame_buffering, Some(4));
    }

    #[test]
    fn rbsp_extract_then_parse() {
        // Round-trip: any NAL payload without 0x00,0x00,0x00..0x03 sequences
        // passes through extract_rbsp unchanged.
        let body = [66u8, 0xC0, 10, 0xDA, 0x08, 0x36, 0x48];
        let rbsp = extract_rbsp(&body);
        assert_eq!(rbsp, body);
        let h = NalHeader::parse(0x67).unwrap();
        assert_eq!(h.nal_unit_type, NalUnitType::Sps);
        let _sps = parse_sps(&h, &rbsp).unwrap();
    }
}
