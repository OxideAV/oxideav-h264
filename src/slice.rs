//! Slice header parsing — ITU-T H.264 §7.3.3.
//!
//! Covers the fields needed to identify a slice and reach the start of
//! `slice_data()` for a baseline-profile decoder. Reference picture list
//! modification (§7.3.3.1) is fully parsed into [`SliceHeader::rplm_l0`] /
//! [`SliceHeader::rplm_l1`] and applied by the decoder via
//! [`crate::dpb::apply_rplm`]. Decoded reference picture marking is parsed
//! into [`SliceHeader::mmco_commands`].
//!
//! Explicit weighted prediction (§7.3.3.2) for P-slices is parsed and the
//! weight table is returned on the [`SliceHeader`]. B-slice bi-prediction is
//! out of scope; the L1 side of the weight table is parsed but left unused.
//! Implicit weighted bi-prediction (`weighted_bipred_idc == 2`) is also out
//! of scope.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::dpb::{MmcoCommand, RplmCommand};
use crate::nal::{NalHeader, NalUnitType};
use crate::pps::Pps;
use crate::sps::Sps;

/// H.264 slice type (§7.4.3 Table 7-6). Includes the `% 5` and the
/// `>= 5` (single-slice-type-per-picture) variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceType {
    P = 0,
    B = 1,
    I = 2,
    SP = 3,
    SI = 4,
}

impl SliceType {
    pub fn from_raw(t: u32) -> Result<Self> {
        match t % 5 {
            0 => Ok(SliceType::P),
            1 => Ok(SliceType::B),
            2 => Ok(SliceType::I),
            3 => Ok(SliceType::SP),
            4 => Ok(SliceType::SI),
            _ => unreachable!(),
        }
    }
}

/// Per-reference luma weight/offset triple from the `pred_weight_table`
/// syntax (§7.3.3.2). Values are stored in the range the spec gives them:
/// `weight` and `offset` are signed ed(v), `log2_denom` comes from the
/// `luma_log2_weight_denom` read once for the whole table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LumaWeight {
    /// `luma_weight_lX[i]` — signed multiplier in units of `1 << log2_denom`.
    pub weight: i32,
    /// `luma_offset_lX[i]` — signed additive offset in sample units (pre-clip).
    pub offset: i32,
    /// `luma_log2_weight_denom` for the whole list (§7.4.3.2).
    pub log2_denom: u32,
    /// True if `luma_weight_lX_flag` was set for this reference. When false,
    /// the default-weight rule in §8.4.2.3.2 applies (weight = 1<<denom,
    /// offset = 0).
    pub present: bool,
}

/// Per-reference chroma weight/offset pair (one per chroma plane, Cb then Cr).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChromaWeight {
    pub weight: [i32; 2],
    pub offset: [i32; 2],
    pub log2_denom: u32,
    pub present: bool,
}

/// Parsed `pred_weight_table` (§7.3.3.2). Entry index is `ref_idx`.
/// Only the L0 side is applied by the decoder today (P-slices); L1 entries
/// are parsed for bitstream conformance but remain unused until B-slice
/// support lands.
#[derive(Clone, Debug, Default)]
pub struct PredWeightTable {
    pub luma_log2_weight_denom: u32,
    pub chroma_log2_weight_denom: u32,
    pub luma_l0: Vec<LumaWeight>,
    pub chroma_l0: Vec<ChromaWeight>,
    pub luma_l1: Vec<LumaWeight>,
    pub chroma_l1: Vec<ChromaWeight>,
}

/// Parsed slice header (subset).
#[derive(Clone, Debug)]
pub struct SliceHeader {
    pub first_mb_in_slice: u32,
    pub slice_type_raw: u32,
    pub slice_type: SliceType,
    pub pic_parameter_set_id: u32,
    pub colour_plane_id: u8,
    pub frame_num: u32,
    pub field_pic_flag: bool,
    pub bottom_field_flag: bool,
    pub idr_pic_id: u32,
    pub pic_order_cnt_lsb: u32,
    pub delta_pic_order_cnt_bottom: i32,
    pub delta_pic_order_cnt: [i32; 2],
    pub redundant_pic_cnt: u32,
    pub direct_spatial_mv_pred_flag: bool,
    pub num_ref_idx_active_override_flag: bool,
    pub num_ref_idx_l0_active_minus1: u32,
    pub num_ref_idx_l1_active_minus1: u32,
    pub cabac_init_idc: u32,
    pub slice_qp_delta: i32,
    pub sp_for_switch_flag: bool,
    pub slice_qs_delta: i32,
    pub disable_deblocking_filter_idc: u32,
    pub slice_alpha_c0_offset_div2: i32,
    pub slice_beta_offset_div2: i32,
    /// Total bit position at which `slice_data()` begins, relative to the
    /// start of the RBSP.
    pub slice_data_bit_offset: u64,
    /// True if this slice belongs to an IDR picture.
    pub is_idr: bool,
    /// Parsed `pred_weight_table` (§7.3.3.2). `None` unless explicit
    /// weighted prediction is enabled for this slice's type + PPS flags.
    pub pred_weight_table: Option<PredWeightTable>,
    /// `no_output_of_prior_pics_flag` from `dec_ref_pic_marking()` on an
    /// IDR slice (§7.3.3.3).
    pub idr_no_output_of_prior_pics_flag: bool,
    /// `long_term_reference_flag` from `dec_ref_pic_marking()` on an IDR.
    pub idr_long_term_reference_flag: bool,
    /// `adaptive_ref_pic_marking_mode_flag` on a non-IDR reference slice.
    pub adaptive_ref_pic_marking_mode_flag: bool,
    /// Parsed MMCO command list (§8.2.5.4), populated only when
    /// `adaptive_ref_pic_marking_mode_flag == true`.
    pub mmco_commands: Vec<MmcoCommand>,
    /// Parsed `ref_pic_list_modification` commands for list 0 (§7.3.3.1).
    /// Empty when `ref_pic_list_modification_flag_l0 == 0` or the slice
    /// type has no list 0 (I / SI). Applied via
    /// [`crate::dpb::apply_rplm`] on top of the default-built list.
    pub rplm_l0: Vec<RplmCommand>,
    /// Parsed `ref_pic_list_modification` commands for list 1 (B-slices).
    pub rplm_l1: Vec<RplmCommand>,
}

/// Parse slice header. The active SPS and PPS must already be available.
pub fn parse_slice_header(
    header: &NalHeader,
    rbsp: &[u8],
    sps: &Sps,
    pps: &Pps,
) -> Result<SliceHeader> {
    if !header.nal_unit_type.is_slice() {
        return Err(Error::invalid("h264 slice: NAL header is not a slice"));
    }
    let is_idr = header.nal_unit_type == NalUnitType::SliceIdr;
    let mut br = BitReader::new(rbsp);
    let first_mb_in_slice = br.read_ue()?;
    let slice_type_raw = br.read_ue()?;
    let slice_type = SliceType::from_raw(slice_type_raw)?;
    let pic_parameter_set_id = br.read_ue()?;
    if pic_parameter_set_id != pps.pic_parameter_set_id {
        return Err(Error::invalid(format!(
            "h264 slice: pic_parameter_set_id={pic_parameter_set_id} doesn't match PPS={}",
            pps.pic_parameter_set_id
        )));
    }
    let colour_plane_id = if sps.separate_colour_plane_flag {
        br.read_u32(2)? as u8
    } else {
        0
    };
    let frame_num_bits = sps.log2_max_frame_num_minus4 + 4;
    let frame_num = br.read_u32(frame_num_bits)?;

    let mut field_pic_flag = false;
    let mut bottom_field_flag = false;
    if !sps.frame_mbs_only_flag {
        field_pic_flag = br.read_flag()?;
        if field_pic_flag {
            bottom_field_flag = br.read_flag()?;
        }
    }

    let idr_pic_id = if is_idr { br.read_ue()? } else { 0 };

    let mut pic_order_cnt_lsb = 0;
    let mut delta_pic_order_cnt_bottom = 0;
    let mut delta_pic_order_cnt = [0i32; 2];
    if sps.pic_order_cnt_type == 0 {
        let bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4;
        pic_order_cnt_lsb = br.read_u32(bits)?;
        if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
            delta_pic_order_cnt_bottom = br.read_se()?;
        }
    } else if sps.pic_order_cnt_type == 1 && !sps.delta_pic_order_always_zero_flag {
        delta_pic_order_cnt[0] = br.read_se()?;
        if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
            delta_pic_order_cnt[1] = br.read_se()?;
        }
    }

    let redundant_pic_cnt = if pps.redundant_pic_cnt_present_flag {
        br.read_ue()?
    } else {
        0
    };

    let mut direct_spatial_mv_pred_flag = false;
    let mut num_ref_idx_active_override_flag = false;
    let mut num_ref_idx_l0_active_minus1 = pps.num_ref_idx_l0_default_active_minus1;
    let mut num_ref_idx_l1_active_minus1 = pps.num_ref_idx_l1_default_active_minus1;

    if slice_type == SliceType::B {
        direct_spatial_mv_pred_flag = br.read_flag()?;
    }
    if matches!(slice_type, SliceType::P | SliceType::SP | SliceType::B) {
        num_ref_idx_active_override_flag = br.read_flag()?;
        if num_ref_idx_active_override_flag {
            num_ref_idx_l0_active_minus1 = br.read_ue()?;
            if slice_type == SliceType::B {
                num_ref_idx_l1_active_minus1 = br.read_ue()?;
            }
        }
    }

    // Reference picture list modification (§7.3.3.1). Commands are stored
    // on the slice header and applied by the decoder over the default-
    // built RefPicList0 / RefPicList1 via `dpb::apply_rplm`.
    let mut rplm_l0: Vec<RplmCommand> = Vec::new();
    let mut rplm_l1: Vec<RplmCommand> = Vec::new();
    if header.nal_unit_type != NalUnitType::SliceExtension {
        parse_ref_pic_list_modification(&mut br, slice_type, &mut rplm_l0, &mut rplm_l1)?;
    }

    // Prediction weight table (§7.3.3.2). Present when explicit weighted
    // prediction is enabled by the PPS for this slice type. Implicit
    // weighted bi-prediction (`weighted_bipred_idc == 2`) is gated by the
    // PPS flag being `== 1` and therefore does not trigger parsing here.
    let weighted_explicit = (matches!(slice_type, SliceType::P | SliceType::SP)
        && pps.weighted_pred_flag)
        || (slice_type == SliceType::B && pps.weighted_bipred_idc == 1);
    let mut pred_weight_table: Option<PredWeightTable> = None;
    if weighted_explicit {
        pred_weight_table = Some(parse_pred_weight_table(
            &mut br,
            sps.chroma_format_idc,
            num_ref_idx_l0_active_minus1,
            num_ref_idx_l1_active_minus1,
            slice_type == SliceType::B,
        )?);
    }

    // Decoded reference picture marking (§7.3.3.3) — parse and retain
    // the commands so the decoder can replay them against its DPB.
    let mut idr_no_output_of_prior_pics_flag = false;
    let mut idr_long_term_reference_flag = false;
    let mut adaptive_ref_pic_marking_mode_flag = false;
    let mut mmco_commands: Vec<MmcoCommand> = Vec::new();
    if header.nal_ref_idc != 0 {
        parse_dec_ref_pic_marking(
            &mut br,
            is_idr,
            &mut idr_no_output_of_prior_pics_flag,
            &mut idr_long_term_reference_flag,
            &mut adaptive_ref_pic_marking_mode_flag,
            &mut mmco_commands,
        )?;
    }

    let mut cabac_init_idc = 0;
    if pps.entropy_coding_mode_flag && !matches!(slice_type, SliceType::I | SliceType::SI) {
        cabac_init_idc = br.read_ue()?;
    }

    let slice_qp_delta = br.read_se()?;
    let mut sp_for_switch_flag = false;
    let mut slice_qs_delta = 0;
    if matches!(slice_type, SliceType::SP | SliceType::SI) {
        if slice_type == SliceType::SP {
            sp_for_switch_flag = br.read_flag()?;
        }
        slice_qs_delta = br.read_se()?;
    }

    let mut disable_deblocking_filter_idc = 0;
    let mut slice_alpha_c0_offset_div2 = 0;
    let mut slice_beta_offset_div2 = 0;
    if pps.deblocking_filter_control_present_flag {
        disable_deblocking_filter_idc = br.read_ue()?;
        if disable_deblocking_filter_idc != 1 {
            slice_alpha_c0_offset_div2 = br.read_se()?;
            slice_beta_offset_div2 = br.read_se()?;
        }
    }

    let slice_data_bit_offset = br.bit_position();

    Ok(SliceHeader {
        first_mb_in_slice,
        slice_type_raw,
        slice_type,
        pic_parameter_set_id,
        colour_plane_id,
        frame_num,
        field_pic_flag,
        bottom_field_flag,
        idr_pic_id,
        pic_order_cnt_lsb,
        delta_pic_order_cnt_bottom,
        delta_pic_order_cnt,
        redundant_pic_cnt,
        direct_spatial_mv_pred_flag,
        num_ref_idx_active_override_flag,
        num_ref_idx_l0_active_minus1,
        num_ref_idx_l1_active_minus1,
        cabac_init_idc,
        slice_qp_delta,
        sp_for_switch_flag,
        slice_qs_delta,
        disable_deblocking_filter_idc,
        slice_alpha_c0_offset_div2,
        slice_beta_offset_div2,
        slice_data_bit_offset,
        is_idr,
        pred_weight_table,
        idr_no_output_of_prior_pics_flag,
        idr_long_term_reference_flag,
        adaptive_ref_pic_marking_mode_flag,
        mmco_commands,
        rplm_l0,
        rplm_l1,
    })
}

/// Parse `ref_pic_list_modification` syntax (§7.3.3.1). The grammar is a
/// per-list pair of `ref_pic_list_modification_flag_lX` followed by a loop
/// of `(modification_of_pic_nums_idc, operand)` pairs terminated by
/// `modification_of_pic_nums_idc == 3`. List 0 is parsed for every slice
/// type except I / SI; list 1 only for B-slices.
fn parse_ref_pic_list_modification(
    br: &mut BitReader<'_>,
    st: SliceType,
    rplm_l0: &mut Vec<RplmCommand>,
    rplm_l1: &mut Vec<RplmCommand>,
) -> Result<()> {
    let do_l0 = !matches!(st, SliceType::I | SliceType::SI);
    let do_l1 = matches!(st, SliceType::B);
    if do_l0 {
        let flag = br.read_flag()?;
        if flag {
            parse_rplm_loop(br, rplm_l0)?;
        }
    }
    if do_l1 {
        let flag = br.read_flag()?;
        if flag {
            parse_rplm_loop(br, rplm_l1)?;
        }
    }
    Ok(())
}

/// Inner loop of `ref_pic_list_modification` — reads commands until the
/// sentinel `modification_of_pic_nums_idc == 3` (§7.3.3.1 Table 7-9).
fn parse_rplm_loop(br: &mut BitReader<'_>, out: &mut Vec<RplmCommand>) -> Result<()> {
    loop {
        let idc = br.read_ue()?;
        match idc {
            0 => {
                let d = br.read_ue()?;
                out.push(RplmCommand::ShortTermSubtract {
                    abs_diff_pic_num_minus1: d,
                });
            }
            1 => {
                let d = br.read_ue()?;
                out.push(RplmCommand::ShortTermAdd {
                    abs_diff_pic_num_minus1: d,
                });
            }
            2 => {
                let n = br.read_ue()?;
                out.push(RplmCommand::LongTerm {
                    long_term_pic_num: n,
                });
            }
            3 => break,
            _ => {
                return Err(Error::invalid(format!(
                    "h264 slice: bad modification_of_pic_nums_idc {idc}"
                )));
            }
        }
    }
    Ok(())
}

fn parse_pred_weight_table(
    br: &mut BitReader<'_>,
    chroma_format_idc: u32,
    num_l0: u32,
    num_l1: u32,
    has_l1: bool,
) -> Result<PredWeightTable> {
    let luma_log2_weight_denom = br.read_ue()?;
    let chroma_present = chroma_format_idc != 0;
    let chroma_log2_weight_denom = if chroma_present { br.read_ue()? } else { 0 };

    let (luma_l0, chroma_l0) = parse_weight_list(
        br,
        num_l0,
        chroma_present,
        luma_log2_weight_denom,
        chroma_log2_weight_denom,
    )?;
    let (luma_l1, chroma_l1) = if has_l1 {
        parse_weight_list(
            br,
            num_l1,
            chroma_present,
            luma_log2_weight_denom,
            chroma_log2_weight_denom,
        )?
    } else {
        (Vec::new(), Vec::new())
    };

    Ok(PredWeightTable {
        luma_log2_weight_denom,
        chroma_log2_weight_denom,
        luma_l0,
        chroma_l0,
        luma_l1,
        chroma_l1,
    })
}

fn parse_weight_list(
    br: &mut BitReader<'_>,
    num: u32,
    chroma: bool,
    luma_log2_denom: u32,
    chroma_log2_denom: u32,
) -> Result<(Vec<LumaWeight>, Vec<ChromaWeight>)> {
    let count = num as usize + 1;
    let mut luma = Vec::with_capacity(count);
    let mut cw = Vec::with_capacity(if chroma { count } else { 0 });
    for _ in 0..count {
        let luma_flag = br.read_flag()?;
        let mut lw = LumaWeight {
            weight: 1 << luma_log2_denom,
            offset: 0,
            log2_denom: luma_log2_denom,
            present: luma_flag,
        };
        if luma_flag {
            lw.weight = br.read_se()?;
            lw.offset = br.read_se()?;
        }
        luma.push(lw);
        if chroma {
            let c_flag = br.read_flag()?;
            let mut c = ChromaWeight {
                weight: [1 << chroma_log2_denom, 1 << chroma_log2_denom],
                offset: [0, 0],
                log2_denom: chroma_log2_denom,
                present: c_flag,
            };
            if c_flag {
                for j in 0..2 {
                    c.weight[j] = br.read_se()?;
                    c.offset[j] = br.read_se()?;
                }
            }
            cw.push(c);
        }
    }
    Ok((luma, cw))
}

fn parse_dec_ref_pic_marking(
    br: &mut BitReader<'_>,
    is_idr: bool,
    idr_no_output_of_prior_pics_flag: &mut bool,
    idr_long_term_reference_flag: &mut bool,
    adaptive_ref_pic_marking_mode_flag: &mut bool,
    mmco_commands: &mut Vec<MmcoCommand>,
) -> Result<()> {
    if is_idr {
        *idr_no_output_of_prior_pics_flag = br.read_flag()?;
        *idr_long_term_reference_flag = br.read_flag()?;
    } else {
        let adaptive = br.read_flag()?;
        *adaptive_ref_pic_marking_mode_flag = adaptive;
        if adaptive {
            loop {
                let op = br.read_ue()?;
                if op == 0 {
                    break;
                }
                match op {
                    1 => {
                        let d = br.read_ue()?;
                        mmco_commands.push(MmcoCommand::MarkShortTermUnused {
                            difference_of_pic_nums_minus1: d,
                        });
                    }
                    2 => {
                        let n = br.read_ue()?;
                        mmco_commands.push(MmcoCommand::MarkLongTermUnused {
                            long_term_pic_num: n,
                        });
                    }
                    3 => {
                        let d = br.read_ue()?;
                        let idx = br.read_ue()?;
                        mmco_commands.push(MmcoCommand::AssignLongTerm {
                            difference_of_pic_nums_minus1: d,
                            long_term_frame_idx: idx,
                        });
                    }
                    4 => {
                        let m = br.read_ue()?;
                        mmco_commands.push(MmcoCommand::SetMaxLongTerm {
                            max_long_term_frame_idx_plus1: m,
                        });
                    }
                    5 => {
                        mmco_commands.push(MmcoCommand::MarkAllUnused);
                    }
                    6 => {
                        let idx = br.read_ue()?;
                        mmco_commands.push(MmcoCommand::AssignCurrentLongTerm {
                            long_term_frame_idx: idx,
                        });
                    }
                    _ => {
                        return Err(Error::invalid(format!(
                            "h264 slice: bad memory_management_control_operation {op}"
                        )));
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitwriter::BitWriter;

    /// Round-trip a pred_weight_table with a handful of entries through the
    /// BitWriter → BitReader pipeline and assert the parser recovers the
    /// exact weight/offset/log2_denom values, including the "flag not set"
    /// path that defaults to weight = 1<<denom / offset = 0.
    #[test]
    fn pred_weight_table_roundtrip() {
        let mut bw = BitWriter::new();
        // luma_log2_weight_denom = 5, chroma_log2_weight_denom = 4.
        bw.write_ue(5);
        bw.write_ue(4);
        // L0 has two entries (num_l0 = 1). Entry 0: luma present w=48 o=-16,
        // chroma present w=[20, 18] o=[-4, 2]. Entry 1: all flags off.
        bw.write_flag(true);
        bw.write_se(48);
        bw.write_se(-16);
        bw.write_flag(true);
        bw.write_se(20);
        bw.write_se(-4);
        bw.write_se(18);
        bw.write_se(2);

        bw.write_flag(false);
        bw.write_flag(false);
        bw.write_rbsp_trailing_bits();
        let buf = bw.finish();

        let mut br = BitReader::new(&buf);
        let t = parse_pred_weight_table(&mut br, 1, 1, 0, false).expect("parse");
        assert_eq!(t.luma_log2_weight_denom, 5);
        assert_eq!(t.chroma_log2_weight_denom, 4);
        assert_eq!(t.luma_l0.len(), 2);
        assert_eq!(t.chroma_l0.len(), 2);

        assert!(t.luma_l0[0].present);
        assert_eq!(t.luma_l0[0].weight, 48);
        assert_eq!(t.luma_l0[0].offset, -16);
        assert_eq!(t.luma_l0[0].log2_denom, 5);

        assert!(t.chroma_l0[0].present);
        assert_eq!(t.chroma_l0[0].weight, [20, 18]);
        assert_eq!(t.chroma_l0[0].offset, [-4, 2]);
        assert_eq!(t.chroma_l0[0].log2_denom, 4);

        // Entry 1: flag cleared → default weight = 1<<denom, offset = 0.
        assert!(!t.luma_l0[1].present);
        assert_eq!(t.luma_l0[1].weight, 1 << 5);
        assert_eq!(t.luma_l0[1].offset, 0);
        assert!(!t.chroma_l0[1].present);
        assert_eq!(t.chroma_l0[1].weight, [1 << 4, 1 << 4]);
        assert_eq!(t.chroma_l0[1].offset, [0, 0]);
    }

    /// A P-slice with `ref_pic_list_modification_flag_l0 = 1` and a mix of
    /// idc=0 / idc=1 / idc=2 commands terminated by idc=3 round-trips
    /// through the parser producing the matching [`RplmCommand`] list.
    #[test]
    fn rplm_parse_p_slice_all_idcs() {
        let mut bw = BitWriter::new();
        // ref_pic_list_modification_flag_l0 = 1.
        bw.write_flag(true);
        // idc=0 (subtract), delta=2.
        bw.write_ue(0);
        bw.write_ue(2);
        // idc=1 (add), delta=0.
        bw.write_ue(1);
        bw.write_ue(0);
        // idc=2 (long-term), long_term_pic_num = 4.
        bw.write_ue(2);
        bw.write_ue(4);
        // idc=3 terminator.
        bw.write_ue(3);
        bw.write_rbsp_trailing_bits();
        let buf = bw.finish();

        let mut br = BitReader::new(&buf);
        let mut l0 = Vec::new();
        let mut l1 = Vec::new();
        parse_ref_pic_list_modification(&mut br, SliceType::P, &mut l0, &mut l1).expect("parse");
        assert_eq!(l0.len(), 3);
        assert!(l1.is_empty());
        assert_eq!(
            l0[0],
            RplmCommand::ShortTermSubtract {
                abs_diff_pic_num_minus1: 2
            }
        );
        assert_eq!(
            l0[1],
            RplmCommand::ShortTermAdd {
                abs_diff_pic_num_minus1: 0
            }
        );
        assert_eq!(
            l0[2],
            RplmCommand::LongTerm {
                long_term_pic_num: 4
            }
        );
    }

    /// A B-slice parses both list flags + loops; if only `flag_l1` is set
    /// the L0 output is empty and the L1 output carries commands.
    #[test]
    fn rplm_parse_b_slice_only_l1() {
        let mut bw = BitWriter::new();
        // ref_pic_list_modification_flag_l0 = 0.
        bw.write_flag(false);
        // ref_pic_list_modification_flag_l1 = 1.
        bw.write_flag(true);
        // idc=0 delta=0.
        bw.write_ue(0);
        bw.write_ue(0);
        // idc=3.
        bw.write_ue(3);
        bw.write_rbsp_trailing_bits();
        let buf = bw.finish();

        let mut br = BitReader::new(&buf);
        let mut l0 = Vec::new();
        let mut l1 = Vec::new();
        parse_ref_pic_list_modification(&mut br, SliceType::B, &mut l0, &mut l1).expect("parse");
        assert!(l0.is_empty());
        assert_eq!(l1.len(), 1);
        assert_eq!(
            l1[0],
            RplmCommand::ShortTermSubtract {
                abs_diff_pic_num_minus1: 0
            }
        );
    }

    /// I-slices have no RPLM syntax; the parser must not read any bits.
    #[test]
    fn rplm_parse_i_slice_is_noop() {
        let buf = [0u8; 1];
        let mut br = BitReader::new(&buf);
        let before = br.bit_position();
        let mut l0 = Vec::new();
        let mut l1 = Vec::new();
        parse_ref_pic_list_modification(&mut br, SliceType::I, &mut l0, &mut l1).expect("parse");
        assert_eq!(br.bit_position(), before);
        assert!(l0.is_empty());
        assert!(l1.is_empty());
    }

    #[test]
    fn pred_weight_table_skip_chroma_when_monochrome() {
        // chroma_format_idc = 0 → chroma syntax fully skipped.
        let mut bw = BitWriter::new();
        bw.write_ue(2); // luma_log2_weight_denom
        bw.write_flag(true);
        bw.write_se(7);
        bw.write_se(-3);
        bw.write_rbsp_trailing_bits();
        let buf = bw.finish();

        let mut br = BitReader::new(&buf);
        let t = parse_pred_weight_table(&mut br, 0, 0, 0, false).expect("parse");
        assert_eq!(t.luma_log2_weight_denom, 2);
        assert_eq!(t.chroma_log2_weight_denom, 0);
        assert_eq!(t.luma_l0.len(), 1);
        assert!(t.chroma_l0.is_empty());
        assert_eq!(t.luma_l0[0].weight, 7);
        assert_eq!(t.luma_l0[0].offset, -3);
    }
}
