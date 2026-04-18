//! Picture Parameter Set (PPS) parsing — ITU-T H.264 §7.3.2.2.
//!
//! We extract the fields the decoder needs (entropy coding, slice groups,
//! reference index defaults, weighted prediction, deblocking control,
//! constrained_intra and redundant_pic_cnt). Slice-group maps for type 0/2/3/4/5
//! are parsed enough to advance the bit position correctly.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::nal::NalHeader;
use crate::sps::Sps;

/// Parsed PPS.
#[derive(Clone, Debug)]
pub struct Pps {
    pub pic_parameter_set_id: u32,
    pub seq_parameter_set_id: u32,
    pub entropy_coding_mode_flag: bool,
    pub bottom_field_pic_order_in_frame_present_flag: bool,
    pub num_slice_groups_minus1: u32,
    pub slice_group_map_type: u32,
    pub num_ref_idx_l0_default_active_minus1: u32,
    pub num_ref_idx_l1_default_active_minus1: u32,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_idc: u32,
    pub pic_init_qp_minus26: i32,
    pub pic_init_qs_minus26: i32,
    pub chroma_qp_index_offset: i32,
    pub deblocking_filter_control_present_flag: bool,
    pub constrained_intra_pred_flag: bool,
    pub redundant_pic_cnt_present_flag: bool,

    // Optional high-profile extensions (transform_8x8 etc.). Parsed only if
    // present in the bitstream after the basic fields.
    pub transform_8x8_mode_flag: bool,
    pub pic_scaling_matrix_present_flag: bool,
    /// 4×4 scaling lists from the PPS (§7.4.2.2). `Some` slot means the
    /// PPS signals a (custom or default) matrix for that index; `None`
    /// means "fall back per Table 7-2" — either to the SPS matrix at
    /// the same index or, when that is also absent, to the built-in
    /// default for the block category.
    pub pic_scaling_list_4x4: [Option<[i16; 16]>; 6],
    /// 8×8 scaling lists from the PPS. Same fallback semantics as
    /// [`Self::pic_scaling_list_4x4`].
    pub pic_scaling_list_8x8: [Option<[i16; 64]>; 6],
    pub second_chroma_qp_index_offset: i32,
}

pub fn parse_pps(header: &NalHeader, rbsp: &[u8], _sps: Option<&Sps>) -> Result<Pps> {
    if header.nal_unit_type != crate::nal::NalUnitType::Pps {
        return Err(Error::invalid("h264 pps: NAL header is not PPS"));
    }
    let mut br = BitReader::new(rbsp);

    let pic_parameter_set_id = br.read_ue()?;
    if pic_parameter_set_id > 255 {
        return Err(Error::invalid("h264 pps: pic_parameter_set_id > 255"));
    }
    let seq_parameter_set_id = br.read_ue()?;
    let entropy_coding_mode_flag = br.read_flag()?;
    let bottom_field_pic_order_in_frame_present_flag = br.read_flag()?;
    let num_slice_groups_minus1 = br.read_ue()?;

    let mut slice_group_map_type = 0;
    if num_slice_groups_minus1 > 0 {
        slice_group_map_type = br.read_ue()?;
        match slice_group_map_type {
            0 => {
                for _ in 0..=num_slice_groups_minus1 {
                    let _run_length_minus1 = br.read_ue()?;
                }
            }
            2 => {
                for _ in 0..num_slice_groups_minus1 {
                    let _top_left = br.read_ue()?;
                    let _bottom_right = br.read_ue()?;
                }
            }
            3..=5 => {
                let _slice_group_change_direction_flag = br.read_flag()?;
                let _slice_group_change_rate_minus1 = br.read_ue()?;
            }
            6 => {
                let pic_size_in_map_units_minus1 = br.read_ue()?;
                let bits = bits_for_max(num_slice_groups_minus1 + 1);
                for _ in 0..=pic_size_in_map_units_minus1 {
                    let _slice_group_id = br.read_u32(bits)?;
                }
            }
            _ => {
                // 1 = dispersed; no extra params.
            }
        }
    }

    let num_ref_idx_l0_default_active_minus1 = br.read_ue()?;
    let num_ref_idx_l1_default_active_minus1 = br.read_ue()?;
    let weighted_pred_flag = br.read_flag()?;
    let weighted_bipred_idc = br.read_u32(2)?;
    let pic_init_qp_minus26 = br.read_se()?;
    let pic_init_qs_minus26 = br.read_se()?;
    let chroma_qp_index_offset = br.read_se()?;
    let deblocking_filter_control_present_flag = br.read_flag()?;
    let constrained_intra_pred_flag = br.read_flag()?;
    let redundant_pic_cnt_present_flag = br.read_flag()?;

    // Optional high-profile extensions if any RBSP data is left.
    let mut transform_8x8_mode_flag = false;
    let mut pic_scaling_matrix_present_flag = false;
    let mut pic_scaling_list_4x4: [Option<[i16; 16]>; 6] = [None; 6];
    let mut pic_scaling_list_8x8: [Option<[i16; 64]>; 6] = [None; 6];
    let mut second_chroma_qp_index_offset = chroma_qp_index_offset;
    if br.more_rbsp_data().unwrap_or(false) {
        transform_8x8_mode_flag = br.read_flag()?;
        pic_scaling_matrix_present_flag = br.read_flag()?;
        if pic_scaling_matrix_present_flag {
            let count = 6 + if transform_8x8_mode_flag { 2 } else { 0 };
            for i in 0..count {
                let present = br.read_flag()?;
                if present {
                    if i < 6 {
                        let (mat, use_default) = parse_scaling_list_4x4(&mut br)?;
                        pic_scaling_list_4x4[i] = Some(if use_default {
                            crate::scaling_list::default_4x4_for_index(i)
                        } else {
                            mat
                        });
                    } else {
                        let (mat, use_default) = parse_scaling_list_8x8(&mut br)?;
                        let slot = i - 6;
                        pic_scaling_list_8x8[slot] = Some(if use_default {
                            crate::scaling_list::default_8x8_for_index(slot)
                        } else {
                            mat
                        });
                    }
                }
            }
        }
        second_chroma_qp_index_offset = br.read_se()?;
    }

    Ok(Pps {
        pic_parameter_set_id,
        seq_parameter_set_id,
        entropy_coding_mode_flag,
        bottom_field_pic_order_in_frame_present_flag,
        num_slice_groups_minus1,
        slice_group_map_type,
        num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1,
        weighted_pred_flag,
        weighted_bipred_idc,
        pic_init_qp_minus26,
        pic_init_qs_minus26,
        chroma_qp_index_offset,
        deblocking_filter_control_present_flag,
        constrained_intra_pred_flag,
        redundant_pic_cnt_present_flag,
        transform_8x8_mode_flag,
        pic_scaling_matrix_present_flag,
        pic_scaling_list_4x4,
        pic_scaling_list_8x8,
        second_chroma_qp_index_offset,
    })
}

/// ceil(log2(n)) bits required to encode values < n.
fn bits_for_max(mut n: u32) -> u32 {
    if n <= 1 {
        return 0;
    }
    let mut bits = 0;
    n -= 1;
    while n > 0 {
        bits += 1;
        n >>= 1;
    }
    bits
}

/// §7.3.2.1.1.1 — 4×4 scaling list, PPS flavour. Identical to the SPS
/// helper in [`crate::sps`]; duplicated locally so [`parse_pps`] does
/// not take a cross-module dependency on a private `pub(crate)` item.
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
        let raster = crate::scaling_list::ZIGZAG_4X4[j as usize];
        list[raster] = val as i16;
        if next_scale != 0 {
            last_scale = next_scale;
        }
    }
    Ok((list, use_default))
}

/// §7.3.2.1.1.1 — 8×8 scaling list, PPS flavour.
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
    use crate::nal::NalHeader;

    #[test]
    fn parse_minimal_pps() {
        // Build a baseline PPS body that exercises:
        //   pps_id=0   -> "1"
        //   sps_id=0   -> "1"
        //   entropy_coding_mode=0 (CAVLC) -> "0"
        //   bottom_field_pic_order=0 -> "0"
        //   num_slice_groups_minus1=0 -> "1"
        //   num_ref_idx_l0_default_active_minus1=0 -> "1"
        //   num_ref_idx_l1_default_active_minus1=0 -> "1"
        //   weighted_pred_flag=0 -> "0"
        //   weighted_bipred_idc=0 -> "00"
        //   pic_init_qp_minus26=0 -> se(0) = "1"
        //   pic_init_qs_minus26=0 -> "1"
        //   chroma_qp_index_offset=0 -> "1"
        //   deblocking_filter_control_present=1 -> "1"
        //   constrained_intra_pred=0 -> "0"
        //   redundant_pic_cnt_present=0 -> "0"
        //   trailing bits: "1 0 0 0 0 0 ..."
        // Bit string:
        //   1 1 0 0 1 1 1 0 0 0 1 1 1 1 0 0 1
        // Pad to byte boundary with stop bit + zeros (already includes the stop bit
        // at end? No — append "1" stop + zeros).
        // Append "1 0..." giving: 1 1 0 0 1 1 1 0 0 0 1 1 1 1 0 0 1 1 0 0 0 0 0 0
        //   = 1100 1110 0011 1100 1000 0000
        //   = 0xCE 0x3C 0x80
        let rbsp = [0xCE, 0x3C, 0x80];
        let h = NalHeader::parse(0x68).unwrap();
        let pps = parse_pps(&h, &rbsp, None).expect("parse");
        assert_eq!(pps.pic_parameter_set_id, 0);
        assert_eq!(pps.seq_parameter_set_id, 0);
        assert!(!pps.entropy_coding_mode_flag);
        assert_eq!(pps.num_slice_groups_minus1, 0);
        assert!(pps.deblocking_filter_control_present_flag);
    }
}
