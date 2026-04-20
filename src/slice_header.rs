//! §7.3.3 / §7.4.3 — Slice header parsing.
//!
//! Spec-driven implementation of `slice_header()` per ITU-T Rec. H.264
//! (08/2024). Stops at the end of the slice header — `slice_data()` is
//! decoded separately.
//!
//! Sub-structures covered:
//! * §7.3.3.1 — `ref_pic_list_modification()`
//! * §7.3.3.2 — `pred_weight_table()`
//! * §7.3.3.3 — `dec_ref_pic_marking()`
//!
//! The parser takes an already-de-emulated RBSP body (emulation
//! prevention bytes stripped by [`crate::nal::parse_nal_unit`]), plus
//! references to the active SPS, active PPS, and the NAL header. The
//! PPS is referenced by `pic_parameter_set_id` inside the slice header;
//! callers resolve the PPS from their parameter-set table before the
//! call.
//!
//! Annex F/G/J MVC/SVC/3D-AVC extensions are *not* parsed here. Callers
//! are expected to route those NAL types elsewhere; if we see
//! `nal_unit_type == 20 || 21` the spec calls
//! `ref_pic_list_mvc_modification()` instead of the plain
//! `ref_pic_list_modification()` — we reject those NAL types.

use crate::bitstream::{BitError, BitReader};
use crate::nal::{NalHeader, NalUnitType};
use crate::pps::Pps;
use crate::sps::Sps;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SliceHeaderError {
    #[error("bitstream read failed: {0}")]
    Bitstream(#[from] BitError),
    /// §7.4.3 Table 7-6 — `slice_type` is in the range 0..=9.
    #[error("slice_type out of range (got {0}, max 9)")]
    SliceTypeOutOfRange(u32),
    /// §7.4.3 — `pic_parameter_set_id` range 0..=255.
    #[error("pic_parameter_set_id out of range (got {0}, max 255)")]
    PpsIdOutOfRange(u32),
    /// §7.4.3 — slice_type shall be 2,4,7,9 (I/SI) when nal_unit_type==5 (IDR).
    #[error("IDR slice must have slice_type I or SI (got raw {0})")]
    IdrSliceTypeNotI(u32),
    /// §7.4.3 — `colour_plane_id` in range 0..=2.
    #[error("colour_plane_id out of range (got {0}, max 2)")]
    ColourPlaneIdOutOfRange(u32),
    /// §7.4.3 — `idr_pic_id` in range 0..=65535.
    #[error("idr_pic_id out of range (got {0}, max 65535)")]
    IdrPicIdOutOfRange(u32),
    /// §7.4.3 — `redundant_pic_cnt` in range 0..=127.
    #[error("redundant_pic_cnt out of range (got {0}, max 127)")]
    RedundantPicCntOutOfRange(u32),
    /// §7.4.3 — `cabac_init_idc` in range 0..=2.
    #[error("cabac_init_idc out of range (got {0}, max 2)")]
    CabacInitIdcOutOfRange(u32),
    /// §7.4.3 — `disable_deblocking_filter_idc` in range 0..=2.
    #[error("disable_deblocking_filter_idc out of range (got {0}, max 2)")]
    DisableDeblockingFilterIdcOutOfRange(u32),
    /// §7.4.3 — `slice_alpha_c0_offset_div2` in range -6..=6.
    #[error("slice_alpha_c0_offset_div2 out of range (got {0}, must be -6..=6)")]
    SliceAlphaC0OffsetDiv2OutOfRange(i32),
    /// §7.4.3 — `slice_beta_offset_div2` in range -6..=6.
    #[error("slice_beta_offset_div2 out of range (got {0}, must be -6..=6)")]
    SliceBetaOffsetDiv2OutOfRange(i32),
    /// §7.4.3.2 — `luma_log2_weight_denom` in range 0..=7.
    #[error("luma_log2_weight_denom out of range (got {0}, max 7)")]
    LumaLog2WeightDenomOutOfRange(u32),
    /// §7.4.3.2 — `chroma_log2_weight_denom` in range 0..=7.
    #[error("chroma_log2_weight_denom out of range (got {0}, max 7)")]
    ChromaLog2WeightDenomOutOfRange(u32),
    /// §7.4.3.3 — `memory_management_control_operation` in range 0..=6.
    #[error("memory_management_control_operation out of range (got {0}, max 6)")]
    MmcoOutOfRange(u32),
    /// §7.3.3.1 — `modification_of_pic_nums_idc` in range 0..=3 in the
    /// base spec (4/5 are MVC-only, Annex G); we reject those values
    /// because this module doesn't target Annex G.
    #[error("modification_of_pic_nums_idc out of range (got {0}, max 3)")]
    ModOfPicNumsIdcOutOfRange(u32),
    /// §7.3.3 — PPS lookup failed. The slice named a PPS id the caller
    /// does not have stored.
    #[error("referenced PPS id {0} not available")]
    PpsLookupFailed(u32),
    /// §7.3.3 — MVC/SVC/3D-AVC slice extension NAL types are not
    /// handled by this parser.
    #[error("slice layer extension (nal_unit_type {0}) not supported")]
    SliceExtensionNotSupported(u8),
}

/// §7.4.3 Table 7-6 — decoded slice type. `slice_type` values 5..=9
/// indicate "all slices of the picture have this type"; we track that
/// as a separate flag and collapse them to the underlying P/B/I/SP/SI.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SliceType {
    P,
    B,
    I,
    SP,
    SI,
}

impl SliceType {
    /// §7.4.3 — `slice_type` values 5..=9 set "all slices in the picture
    /// have this type". Otherwise the type is picture-wise independent.
    pub fn all_in_picture(slice_type_raw: u32) -> bool {
        slice_type_raw >= 5
    }

    /// §7.4.3 Table 7-6 — decode a raw 0..=9 slice_type value.
    pub fn from_raw(slice_type_raw: u32) -> Result<Self, SliceHeaderError> {
        Ok(match slice_type_raw % 5 {
            0 => SliceType::P,
            1 => SliceType::B,
            2 => SliceType::I,
            3 => SliceType::SP,
            4 => SliceType::SI,
            _ => return Err(SliceHeaderError::SliceTypeOutOfRange(slice_type_raw)),
        })
    }

    /// True if the slice type has a list 0 (P, SP, B).
    pub fn has_list_0(self) -> bool {
        matches!(self, SliceType::P | SliceType::SP | SliceType::B)
    }

    /// True if the slice type has a list 1 (B only).
    pub fn has_list_1(self) -> bool {
        matches!(self, SliceType::B)
    }

    /// True for I/SI slices (no inter prediction).
    pub fn is_intra(self) -> bool {
        matches!(self, SliceType::I | SliceType::SI)
    }
}

/// §7.3.3.1 / §7.4.3.1 — one modification op within the
/// `ref_pic_list_modification()` loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefPicListModificationOp {
    /// `modification_of_pic_nums_idc == 0` — `abs_diff_pic_num_minus1`
    /// subtracts from the running picNum prediction value.
    Subtract(u32),
    /// `modification_of_pic_nums_idc == 1` — `abs_diff_pic_num_minus1`
    /// adds to the running picNum prediction value.
    Add(u32),
    /// `modification_of_pic_nums_idc == 2` — `long_term_pic_num`.
    LongTerm(u32),
}

/// §7.3.3.1 — reference picture list modification for the slice.
/// Empty vectors mean the flag was 0 (no modification loop).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RefPicListModification {
    pub modifications_l0: Vec<RefPicListModificationOp>,
    pub modifications_l1: Vec<RefPicListModificationOp>,
}

/// §7.3.3.2 — prediction weight table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredWeightTable {
    pub luma_log2_weight_denom: u32,
    /// `chroma_log2_weight_denom` — only present when
    /// `ChromaArrayType != 0`; stored as 0 otherwise.
    pub chroma_log2_weight_denom: u32,
    /// Per-entry luma weighting for list 0: `Some((weight, offset))`
    /// when `luma_weight_l0_flag == 1`, else `None` (inferred to
    /// `2^luma_log2_weight_denom` per §7.4.3.2).
    pub luma_weights_l0: Vec<Option<(i32, i32)>>,
    /// Per-entry chroma weighting for list 0: `[(weight_cb, offset_cb),
    /// (weight_cr, offset_cr)]`. Empty if `ChromaArrayType == 0`.
    pub chroma_weights_l0: Vec<Option<[(i32, i32); 2]>>,
    /// List 1 equivalents (B slices only).
    pub luma_weights_l1: Vec<Option<(i32, i32)>>,
    pub chroma_weights_l1: Vec<Option<[(i32, i32); 2]>>,
}

/// §7.3.3.3 Table 7-9 — `memory_management_control_operation` (MMCO).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MmcoOp {
    /// MMCO 1 — mark a short-term reference as unused:
    /// `difference_of_pic_nums_minus1`.
    MarkShortTermUnused(u32),
    /// MMCO 2 — mark a long-term reference as unused: `long_term_pic_num`.
    MarkLongTermUnused(u32),
    /// MMCO 3 — mark a short-term reference as used for long-term and
    /// assign a long-term frame index: `(difference_of_pic_nums_minus1,
    /// long_term_frame_idx)`.
    AssignLongTerm(u32, u32),
    /// MMCO 4 — `max_long_term_frame_idx_plus1`.
    SetMaxLongTermIdx(u32),
    /// MMCO 5 — mark all reference pictures as unused.
    MarkAllUnused,
    /// MMCO 6 — mark the current picture as used for long-term:
    /// `long_term_frame_idx`.
    AssignCurrentLongTerm(u32),
}

/// §7.3.3.3 — decoded reference picture marking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecRefPicMarking {
    /// §7.3.3.3 — present only on IDR slices.
    pub no_output_of_prior_pics_flag: bool,
    /// §7.3.3.3 — present only on IDR slices.
    pub long_term_reference_flag: bool,
    /// `None` on IDR slices. On non-IDR slices: `None` when
    /// `adaptive_ref_pic_marking_mode_flag == 0` (sliding window),
    /// `Some(ops)` otherwise (the MMCO loop, without the terminator).
    pub adaptive_marking: Option<Vec<MmcoOp>>,
}

/// §7.3.3 — parsed `slice_header()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceHeader {
    pub first_mb_in_slice: u32,
    /// Raw `slice_type` value as signalled in the bitstream (0..=9).
    pub slice_type_raw: u32,
    pub slice_type: SliceType,
    /// True when `slice_type_raw >= 5` — every slice of the picture
    /// shall carry this same slice_type.
    pub all_slices_same_type: bool,
    pub pic_parameter_set_id: u32,
    /// §7.3.3 — only present when `separate_colour_plane_flag == 1`,
    /// 0 otherwise.
    pub colour_plane_id: u8,
    pub frame_num: u32,
    pub field_pic_flag: bool,
    /// §7.3.3 — only present when `field_pic_flag == 1`, inferred to 0
    /// otherwise (§7.4.3).
    pub bottom_field_flag: bool,
    /// §7.3.3 — only present on IDR slices; 0 otherwise.
    pub idr_pic_id: u32,
    /// `pic_order_cnt_type == 0` branch.
    pub pic_order_cnt_lsb: u32,
    /// `pic_order_cnt_type == 0` branch, optional — 0 if absent.
    pub delta_pic_order_cnt_bottom: i32,
    /// `pic_order_cnt_type == 1 && !delta_pic_order_always_zero_flag`
    /// branch. `[0]` is always present when branch taken. `[1]` is only
    /// present under `bottom_field_pic_order_in_frame_present_flag &&
    /// !field_pic_flag`; otherwise 0.
    pub delta_pic_order_cnt: [i32; 2],
    pub redundant_pic_cnt: u32,
    /// B-slice only.
    pub direct_spatial_mv_pred_flag: bool,
    pub num_ref_idx_active_override_flag: bool,
    /// After applying override / PPS defaults per §7.4.3.
    pub num_ref_idx_l0_active_minus1: u32,
    pub num_ref_idx_l1_active_minus1: u32,
    pub ref_pic_list_modification: RefPicListModification,
    pub pred_weight_table: Option<PredWeightTable>,
    /// §7.3.3 — only present when `nal_ref_idc != 0`; `None` otherwise.
    pub dec_ref_pic_marking: Option<DecRefPicMarking>,
    /// §7.3.3 — present when `entropy_coding_mode_flag == 1` and slice
    /// is not I/SI; 0 otherwise.
    pub cabac_init_idc: u32,
    pub slice_qp_delta: i32,
    /// SP slice only.
    pub sp_for_switch_flag: bool,
    /// SP / SI slice only; 0 otherwise.
    pub slice_qs_delta: i32,
    /// §7.3.3 — present under `deblocking_filter_control_present_flag`;
    /// inferred to 0 otherwise (§7.4.3).
    pub disable_deblocking_filter_idc: u32,
    /// §7.3.3 — present when the previous flag is present *and*
    /// `disable_deblocking_filter_idc != 1`; 0 otherwise.
    pub slice_alpha_c0_offset_div2: i32,
    pub slice_beta_offset_div2: i32,
    /// §7.3.3 — present when `num_slice_groups_minus1 > 0` and FMO map
    /// type is 3..=5; 0 otherwise. Bit width per §7.4.3 eq. (7-37):
    /// `Ceil(Log2(PicSizeInMapUnits / SliceGroupChangeRate + 1))`.
    pub slice_group_change_cycle: u32,
}

impl SliceHeader {
    /// §7.3.3 — parse a `slice_header()` from the de-emulated RBSP body
    /// starting just after the NAL header byte.
    ///
    /// Stops parsing at the end of the slice_header — the remainder of
    /// the RBSP is the slice_data() payload which is handled separately.
    pub fn parse(
        rbsp: &[u8],
        sps: &Sps,
        pps: &Pps,
        nal_header: &NalHeader,
    ) -> Result<Self, SliceHeaderError> {
        // Annex F/G/J — MVC/SVC/3D-AVC slice extensions use a
        // different sub-header prefix. We don't parse those.
        let nut = nal_header.nal_unit_type;
        if matches!(
            nut,
            NalUnitType::SliceExtension | NalUnitType::SliceExtensionDepth
        ) {
            return Err(SliceHeaderError::SliceExtensionNotSupported(nut.as_u8()));
        }
        let idr_pic_flag = nut.is_idr();

        let mut r = BitReader::new(rbsp);

        // §7.3.3 — first_mb_in_slice ue(v), slice_type ue(v), pic_parameter_set_id ue(v).
        let first_mb_in_slice = r.ue()?;
        let slice_type_raw = r.ue()?;
        if slice_type_raw > 9 {
            return Err(SliceHeaderError::SliceTypeOutOfRange(slice_type_raw));
        }
        let slice_type = SliceType::from_raw(slice_type_raw)?;
        let all_slices_same_type = SliceType::all_in_picture(slice_type_raw);

        // §7.4.3 — on IDR (nal_unit_type == 5) slice_type must be in
        // {2,4,7,9} (I / SI).
        if idr_pic_flag && !slice_type.is_intra() {
            return Err(SliceHeaderError::IdrSliceTypeNotI(slice_type_raw));
        }

        let pic_parameter_set_id = r.ue()?;
        if pic_parameter_set_id > 255 {
            return Err(SliceHeaderError::PpsIdOutOfRange(pic_parameter_set_id));
        }

        // §7.3.3 — colour_plane_id u(2) when separate_colour_plane_flag == 1.
        let colour_plane_id = if sps.separate_colour_plane_flag {
            let v = r.u(2)?;
            if v > 2 {
                return Err(SliceHeaderError::ColourPlaneIdOutOfRange(v));
            }
            v as u8
        } else {
            0
        };

        // §7.4.3 — frame_num u(v), v = log2_max_frame_num_minus4 + 4.
        let frame_num = r.u(sps.log2_max_frame_num_minus4 + 4)?;

        // §7.3.3 — field_pic_flag / bottom_field_flag only when
        // !frame_mbs_only_flag.
        let (field_pic_flag, bottom_field_flag) = if !sps.frame_mbs_only_flag {
            let fpf = r.u(1)? == 1;
            let bff = if fpf { r.u(1)? == 1 } else { false };
            (fpf, bff)
        } else {
            (false, false)
        };

        // §7.3.3 — idr_pic_id ue(v) when IdrPicFlag.
        let idr_pic_id = if idr_pic_flag {
            let v = r.ue()?;
            if v > 65535 {
                return Err(SliceHeaderError::IdrPicIdOutOfRange(v));
            }
            v
        } else {
            0
        };

        // §7.3.3 — POC type 0 fields.
        let mut pic_order_cnt_lsb: u32 = 0;
        let mut delta_pic_order_cnt_bottom: i32 = 0;
        // §7.3.3 — POC type 1 fields (present only when !delta_pic_order_always_zero_flag).
        let mut delta_pic_order_cnt = [0i32; 2];

        if sps.pic_order_cnt_type == 0 {
            // u(v), v = log2_max_pic_order_cnt_lsb_minus4 + 4.
            pic_order_cnt_lsb = r.u(sps.log2_max_pic_order_cnt_lsb_minus4 + 4)?;
            if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
                delta_pic_order_cnt_bottom = r.se()?;
            }
        } else if sps.pic_order_cnt_type == 1 && !sps.delta_pic_order_always_zero_flag {
            delta_pic_order_cnt[0] = r.se()?;
            if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
                delta_pic_order_cnt[1] = r.se()?;
            }
        }
        // §7.3.3 — POC type 2 has no per-slice POC fields.

        // §7.3.3 — redundant_pic_cnt under redundant_pic_cnt_present_flag.
        let redundant_pic_cnt = if pps.redundant_pic_cnt_present_flag {
            let v = r.ue()?;
            if v > 127 {
                return Err(SliceHeaderError::RedundantPicCntOutOfRange(v));
            }
            v
        } else {
            0
        };

        // §7.3.3 — direct_spatial_mv_pred_flag for B slices.
        let direct_spatial_mv_pred_flag = if matches!(slice_type, SliceType::B) {
            r.u(1)? == 1
        } else {
            false
        };

        // §7.3.3 — num_ref_idx_active_override_flag group for P/SP/B slices.
        let mut num_ref_idx_active_override_flag = false;
        let mut num_ref_idx_l0_active_minus1 = pps.num_ref_idx_l0_default_active_minus1;
        let mut num_ref_idx_l1_active_minus1 = pps.num_ref_idx_l1_default_active_minus1;
        if matches!(slice_type, SliceType::P | SliceType::SP | SliceType::B) {
            num_ref_idx_active_override_flag = r.u(1)? == 1;
            if num_ref_idx_active_override_flag {
                num_ref_idx_l0_active_minus1 = r.ue()?;
                if matches!(slice_type, SliceType::B) {
                    num_ref_idx_l1_active_minus1 = r.ue()?;
                }
            }
        }

        // §7.3.3 — ref_pic_list_modification() (or ref_pic_list_mvc_modification()
        // for nal_unit_type 20/21, which we rejected above).
        let ref_pic_list_modification = parse_ref_pic_list_modification(&mut r, &slice_type)?;

        // §7.3.3 — pred_weight_table() when explicit weighted pred is active.
        let use_pred_weight = (pps.weighted_pred_flag
            && matches!(slice_type, SliceType::P | SliceType::SP))
            || (pps.weighted_bipred_idc == 1 && matches!(slice_type, SliceType::B));
        let pred_weight_table = if use_pred_weight {
            Some(parse_pred_weight_table(
                &mut r,
                &slice_type,
                num_ref_idx_l0_active_minus1,
                num_ref_idx_l1_active_minus1,
                sps.chroma_array_type(),
            )?)
        } else {
            None
        };

        // §7.3.3 — dec_ref_pic_marking() when nal_ref_idc != 0.
        let dec_ref_pic_marking = if nal_header.nal_ref_idc != 0 {
            Some(parse_dec_ref_pic_marking(&mut r, idr_pic_flag)?)
        } else {
            None
        };

        // §7.3.3 — cabac_init_idc when CABAC and slice is not I/SI.
        let cabac_init_idc = if pps.entropy_coding_mode_flag && !slice_type.is_intra() {
            let v = r.ue()?;
            if v > 2 {
                return Err(SliceHeaderError::CabacInitIdcOutOfRange(v));
            }
            v
        } else {
            0
        };

        // §7.3.3 — slice_qp_delta se(v).
        let slice_qp_delta = r.se()?;

        // §7.3.3 — SP / SI fields.
        let mut sp_for_switch_flag = false;
        let mut slice_qs_delta = 0i32;
        if matches!(slice_type, SliceType::SP | SliceType::SI) {
            if matches!(slice_type, SliceType::SP) {
                sp_for_switch_flag = r.u(1)? == 1;
            }
            slice_qs_delta = r.se()?;
        }

        // §7.3.3 — deblocking filter group under
        // deblocking_filter_control_present_flag.
        let mut disable_deblocking_filter_idc: u32 = 0;
        let mut slice_alpha_c0_offset_div2: i32 = 0;
        let mut slice_beta_offset_div2: i32 = 0;
        if pps.deblocking_filter_control_present_flag {
            disable_deblocking_filter_idc = r.ue()?;
            if disable_deblocking_filter_idc > 2 {
                return Err(SliceHeaderError::DisableDeblockingFilterIdcOutOfRange(
                    disable_deblocking_filter_idc,
                ));
            }
            if disable_deblocking_filter_idc != 1 {
                slice_alpha_c0_offset_div2 = r.se()?;
                if !(-6..=6).contains(&slice_alpha_c0_offset_div2) {
                    return Err(SliceHeaderError::SliceAlphaC0OffsetDiv2OutOfRange(
                        slice_alpha_c0_offset_div2,
                    ));
                }
                slice_beta_offset_div2 = r.se()?;
                if !(-6..=6).contains(&slice_beta_offset_div2) {
                    return Err(SliceHeaderError::SliceBetaOffsetDiv2OutOfRange(
                        slice_beta_offset_div2,
                    ));
                }
            }
        }

        // §7.3.3 — slice_group_change_cycle u(v) when FMO map types 3..=5.
        let mut slice_group_change_cycle: u32 = 0;
        if pps.num_slice_groups_minus1 > 0
            && is_changing_map_type(pps)
        {
            // §7.4.3 eq. (7-37): bits = Ceil(Log2(PicSizeInMapUnits /
            // SliceGroupChangeRate + 1)).
            let pic_size_in_map_units = sps.pic_width_in_mbs() * sps.pic_height_in_map_units();
            let change_rate = slice_group_change_rate(pps).max(1);
            // Division is integer division, matching the spec's "/".
            let bits = ceil_log2(pic_size_in_map_units / change_rate + 1);
            slice_group_change_cycle = if bits == 0 { 0 } else { r.u(bits)? };
        }

        Ok(SliceHeader {
            first_mb_in_slice,
            slice_type_raw,
            slice_type,
            all_slices_same_type,
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
            ref_pic_list_modification,
            pred_weight_table,
            dec_ref_pic_marking,
            cabac_init_idc,
            slice_qp_delta,
            sp_for_switch_flag,
            slice_qs_delta,
            disable_deblocking_filter_idc,
            slice_alpha_c0_offset_div2,
            slice_beta_offset_div2,
            slice_group_change_cycle,
        })
    }

    /// §7.4.3 — `MbaffFrameFlag = mb_adaptive_frame_field_flag && !field_pic_flag`.
    pub fn mbaff_frame_flag(&self, sps: &Sps) -> bool {
        sps.mb_adaptive_frame_field_flag && !self.field_pic_flag
    }
}

/// §7.3.3.1 — parse the `ref_pic_list_modification()` sub-structure.
fn parse_ref_pic_list_modification(
    r: &mut BitReader<'_>,
    slice_type: &SliceType,
) -> Result<RefPicListModification, SliceHeaderError> {
    let mut result = RefPicListModification::default();
    // §7.3.3.1 — list 0 loop when slice_type % 5 != 2 && != 4
    // (i.e. slice has list 0 prediction: P, SP, B).
    if slice_type.has_list_0() {
        let flag_l0 = r.u(1)? == 1;
        if flag_l0 {
            result.modifications_l0 = parse_mod_loop(r)?;
        }
    }
    // §7.3.3.1 — list 1 loop for B slices (slice_type % 5 == 1).
    if slice_type.has_list_1() {
        let flag_l1 = r.u(1)? == 1;
        if flag_l1 {
            result.modifications_l1 = parse_mod_loop(r)?;
        }
    }
    Ok(result)
}

/// §7.3.3.1 — the inner `do { modification_of_pic_nums_idc ... } while
/// (idc != 3)` loop. Returns the decoded ops, *not* including the
/// terminator (idc == 3).
fn parse_mod_loop(
    r: &mut BitReader<'_>,
) -> Result<Vec<RefPicListModificationOp>, SliceHeaderError> {
    let mut ops = Vec::new();
    loop {
        let idc = r.ue()?;
        match idc {
            0 => {
                let abs_diff = r.ue()?;
                ops.push(RefPicListModificationOp::Subtract(abs_diff));
            }
            1 => {
                let abs_diff = r.ue()?;
                ops.push(RefPicListModificationOp::Add(abs_diff));
            }
            2 => {
                let long_term_pic_num = r.ue()?;
                ops.push(RefPicListModificationOp::LongTerm(long_term_pic_num));
            }
            3 => break,
            _ => return Err(SliceHeaderError::ModOfPicNumsIdcOutOfRange(idc)),
        }
    }
    Ok(ops)
}

/// §7.3.3.2 — parse the `pred_weight_table()` sub-structure.
fn parse_pred_weight_table(
    r: &mut BitReader<'_>,
    slice_type: &SliceType,
    num_ref_idx_l0_active_minus1: u32,
    num_ref_idx_l1_active_minus1: u32,
    chroma_array_type: u32,
) -> Result<PredWeightTable, SliceHeaderError> {
    let luma_log2_weight_denom = r.ue()?;
    if luma_log2_weight_denom > 7 {
        return Err(SliceHeaderError::LumaLog2WeightDenomOutOfRange(
            luma_log2_weight_denom,
        ));
    }
    let chroma_log2_weight_denom = if chroma_array_type != 0 {
        let v = r.ue()?;
        if v > 7 {
            return Err(SliceHeaderError::ChromaLog2WeightDenomOutOfRange(v));
        }
        v
    } else {
        0
    };

    let (luma_weights_l0, chroma_weights_l0) = parse_weight_list(
        r,
        num_ref_idx_l0_active_minus1,
        chroma_array_type,
    )?;

    // §7.3.3.2 — list 1 branch only for B slices (slice_type % 5 == 1).
    let (luma_weights_l1, chroma_weights_l1) = if slice_type.has_list_1() {
        parse_weight_list(r, num_ref_idx_l1_active_minus1, chroma_array_type)?
    } else {
        (Vec::new(), Vec::new())
    };

    Ok(PredWeightTable {
        luma_log2_weight_denom,
        chroma_log2_weight_denom,
        luma_weights_l0,
        chroma_weights_l0,
        luma_weights_l1,
        chroma_weights_l1,
    })
}

/// §7.3.3.2 — one pass of the `for (i = 0; i <= num_ref_idx_lX_active_minus1; i++)`
/// loop, covering both the luma and chroma per-entry arrays.
#[allow(clippy::type_complexity)]
fn parse_weight_list(
    r: &mut BitReader<'_>,
    num_ref_idx_minus1: u32,
    chroma_array_type: u32,
) -> Result<
    (
        Vec<Option<(i32, i32)>>,
        Vec<Option<[(i32, i32); 2]>>,
    ),
    SliceHeaderError,
> {
    let count = num_ref_idx_minus1 as usize + 1;
    let mut luma = Vec::with_capacity(count);
    let mut chroma = Vec::with_capacity(if chroma_array_type != 0 { count } else { 0 });
    for _ in 0..count {
        let luma_flag = r.u(1)? == 1;
        luma.push(if luma_flag {
            let w = r.se()?;
            let o = r.se()?;
            Some((w, o))
        } else {
            None
        });
        if chroma_array_type != 0 {
            let chroma_flag = r.u(1)? == 1;
            if chroma_flag {
                let wcb = r.se()?;
                let ocb = r.se()?;
                let wcr = r.se()?;
                let ocr = r.se()?;
                chroma.push(Some([(wcb, ocb), (wcr, ocr)]));
            } else {
                chroma.push(None);
            }
        }
    }
    Ok((luma, chroma))
}

/// §7.3.3.3 — parse `dec_ref_pic_marking()`.
fn parse_dec_ref_pic_marking(
    r: &mut BitReader<'_>,
    idr_pic_flag: bool,
) -> Result<DecRefPicMarking, SliceHeaderError> {
    if idr_pic_flag {
        let no_output_of_prior_pics_flag = r.u(1)? == 1;
        let long_term_reference_flag = r.u(1)? == 1;
        Ok(DecRefPicMarking {
            no_output_of_prior_pics_flag,
            long_term_reference_flag,
            adaptive_marking: None,
        })
    } else {
        let adaptive = r.u(1)? == 1;
        let adaptive_marking = if adaptive {
            let mut ops = Vec::new();
            loop {
                // §7.3.3.3 — memory_management_control_operation ue(v).
                let mmco = r.ue()?;
                match mmco {
                    0 => break,
                    1 => {
                        let diff = r.ue()?;
                        ops.push(MmcoOp::MarkShortTermUnused(diff));
                    }
                    2 => {
                        let ltpn = r.ue()?;
                        ops.push(MmcoOp::MarkLongTermUnused(ltpn));
                    }
                    3 => {
                        let diff = r.ue()?;
                        let ltfi = r.ue()?;
                        ops.push(MmcoOp::AssignLongTerm(diff, ltfi));
                    }
                    4 => {
                        let max = r.ue()?;
                        ops.push(MmcoOp::SetMaxLongTermIdx(max));
                    }
                    5 => {
                        ops.push(MmcoOp::MarkAllUnused);
                    }
                    6 => {
                        let ltfi = r.ue()?;
                        ops.push(MmcoOp::AssignCurrentLongTerm(ltfi));
                    }
                    _ => return Err(SliceHeaderError::MmcoOutOfRange(mmco)),
                }
            }
            Some(ops)
        } else {
            None
        };
        Ok(DecRefPicMarking {
            no_output_of_prior_pics_flag: false,
            long_term_reference_flag: false,
            adaptive_marking,
        })
    }
}

/// §7.4.3 — true when the active PPS uses slice_group_map_type 3, 4, or 5.
fn is_changing_map_type(pps: &Pps) -> bool {
    matches!(
        pps.slice_group_map,
        Some(crate::pps::SliceGroupMap::Changing { .. })
    )
}

/// §7.4.2.2 — `SliceGroupChangeRate = slice_group_change_rate_minus1 + 1`
/// when the PPS uses a Changing FMO map; returns 1 as a safe default
/// otherwise (the bit-width formula uses 1 as the divisor floor, and
/// callers only invoke this when `is_changing_map_type` is true).
fn slice_group_change_rate(pps: &Pps) -> u32 {
    match &pps.slice_group_map {
        Some(crate::pps::SliceGroupMap::Changing {
            change_rate_minus1, ..
        }) => *change_rate_minus1 + 1,
        _ => 1,
    }
}

/// `Ceil(Log2(n))` for `n >= 1`. Same helper as in pps.rs — duplicated
/// here to avoid exposing a crate-wide util module.
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal::{NalHeader, NalUnitType};

    /// Hand-rolled MSB-first bit writer for fixtures.
    struct BitWriter {
        bytes: Vec<u8>,
        bit_pos: u8,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                bit_pos: 0,
            }
        }

        fn u(&mut self, bits: u32, value: u32) {
            for i in (0..bits).rev() {
                let bit = ((value >> i) & 1) as u8;
                if self.bit_pos == 0 {
                    self.bytes.push(0);
                }
                let idx = self.bytes.len() - 1;
                self.bytes[idx] |= bit << (7 - self.bit_pos);
                self.bit_pos = (self.bit_pos + 1) % 8;
            }
        }

        fn ue(&mut self, value: u32) {
            let v = value + 1;
            let leading = 31 - v.leading_zeros();
            for _ in 0..leading {
                self.u(1, 0);
            }
            self.u(leading + 1, v);
        }

        fn se(&mut self, value: i32) {
            let k = if value <= 0 {
                (-2 * value) as u32
            } else {
                (2 * value - 1) as u32
            };
            self.ue(k);
        }

        fn into_bytes(self) -> Vec<u8> {
            self.bytes
        }
    }

    /// Build a minimal Sps — all fields inferred to Baseline-profile
    /// defaults. Callers override what they need by mutating the returned
    /// struct.
    fn dummy_sps() -> Sps {
        Sps {
            profile_idc: 66,
            constraint_set_flags: 0,
            level_idc: 30,
            seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            qpprime_y_zero_transform_bypass_flag: false,
            seq_scaling_matrix_present_flag: false,
            seq_scaling_lists: None,
            log2_max_frame_num_minus4: 0, // MaxFrameNum = 16, frame_num = 4 bits
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb_minus4: 0, // pic_order_cnt_lsb = 4 bits
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            offset_for_ref_frame: Vec::new(),
            max_num_ref_frames: 1,
            gaps_in_frame_num_value_allowed_flag: false,
            pic_width_in_mbs_minus1: 19, // 320px / 16
            pic_height_in_map_units_minus1: 14, // 240px / 16
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: true,
            frame_cropping: None,
            vui_parameters_present_flag: false,
            vui: None,
        }
    }

    /// Build a minimal Pps with CAVLC, no FMO, no weighted pred.
    fn dummy_pps() -> Pps {
        Pps {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
            entropy_coding_mode_flag: false,
            bottom_field_pic_order_in_frame_present_flag: false,
            num_slice_groups_minus1: 0,
            slice_group_map: None,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            pic_init_qp_minus26: 0,
            pic_init_qs_minus26: 0,
            chroma_qp_index_offset: 0,
            deblocking_filter_control_present_flag: false,
            constrained_intra_pred_flag: false,
            redundant_pic_cnt_present_flag: false,
            extension: None,
        }
    }

    /// NAL header with the given type + ref_idc.
    fn nal(nut: NalUnitType, nal_ref_idc: u8) -> NalHeader {
        NalHeader {
            forbidden_zero_bit: 0,
            nal_ref_idc,
            nal_unit_type: nut,
        }
    }

    // -------- Tests --------

    #[test]
    fn slice_type_raw_decoding() {
        // Table 7-6: 0=P, 1=B, 2=I, 3=SP, 4=SI, and 5..=9 repeat with "all slices same".
        assert_eq!(SliceType::from_raw(0).unwrap(), SliceType::P);
        assert_eq!(SliceType::from_raw(1).unwrap(), SliceType::B);
        assert_eq!(SliceType::from_raw(2).unwrap(), SliceType::I);
        assert_eq!(SliceType::from_raw(3).unwrap(), SliceType::SP);
        assert_eq!(SliceType::from_raw(4).unwrap(), SliceType::SI);
        assert_eq!(SliceType::from_raw(5).unwrap(), SliceType::P);
        assert_eq!(SliceType::from_raw(7).unwrap(), SliceType::I);
        assert_eq!(SliceType::from_raw(9).unwrap(), SliceType::SI);
        assert!(!SliceType::all_in_picture(4));
        assert!(SliceType::all_in_picture(5));
        assert!(SliceType::all_in_picture(9));
    }

    #[test]
    fn simple_i_slice_header() {
        // Non-IDR I-slice of a coded frame: first_mb=0, slice_type=7
        // (I, all slices same type), pps=0, frame_num=0, no POC in this
        // branch since poc_type=0 still requires pic_order_cnt_lsb.
        let sps = dummy_sps();
        let pps = dummy_pps();
        let nh = nal(NalUnitType::SliceNonIdr, 3);

        let mut w = BitWriter::new();
        w.ue(0); // first_mb_in_slice
        w.ue(7); // slice_type = 7 (I, all same)
        w.ue(0); // pic_parameter_set_id
        // no colour_plane_id (separate_colour_plane_flag=0)
        w.u(4, 0); // frame_num (4 bits)
        // frame_mbs_only_flag=1 → no field/bottom field flags
        // IdrPicFlag = 0 → no idr_pic_id
        // poc_type=0 → pic_order_cnt_lsb (4 bits)
        w.u(4, 0);
        // bottom_field_pic_order_in_frame_present_flag=0 → no delta bottom
        // redundant_pic_cnt_present_flag=0 → none
        // not B → no direct_spatial_mv_pred_flag
        // I slice → no num_ref_idx_active_override_flag
        // ref_pic_list_modification: slice_type % 5 = 2 (I) → no list 0 loop.
        //                           slice_type % 5 != 1 → no list 1 loop.
        // pred_weight_table not invoked (weighted_pred_flag=0, idc=0).
        // nal_ref_idc != 0 → dec_ref_pic_marking() present.
        //   non-IDR: adaptive_ref_pic_marking_mode_flag=0 → sliding-window.
        w.u(1, 0);
        // entropy_coding_mode=0 AND slice is I → no cabac_init_idc.
        w.se(0); // slice_qp_delta
        // not SP/SI → no sp_for_switch/qs_delta.
        // deblocking_filter_control_present_flag=0 → no deblocking fields.
        // num_slice_groups_minus1=0 → no slice_group_change_cycle.

        let bytes = w.into_bytes();
        let hdr = SliceHeader::parse(&bytes, &sps, &pps, &nh).unwrap();
        assert_eq!(hdr.first_mb_in_slice, 0);
        assert_eq!(hdr.slice_type_raw, 7);
        assert_eq!(hdr.slice_type, SliceType::I);
        assert!(hdr.all_slices_same_type);
        assert_eq!(hdr.frame_num, 0);
        assert!(!hdr.field_pic_flag);
        assert_eq!(hdr.pic_order_cnt_lsb, 0);
        assert_eq!(hdr.slice_qp_delta, 0);
        let drm = hdr.dec_ref_pic_marking.as_ref().unwrap();
        assert!(!drm.no_output_of_prior_pics_flag);
        assert!(drm.adaptive_marking.is_none());
        assert!(hdr.ref_pic_list_modification.modifications_l0.is_empty());
        assert!(hdr.ref_pic_list_modification.modifications_l1.is_empty());
    }

    #[test]
    fn idr_slice_header() {
        // IDR slice: NalUnitType::SliceIdr → IdrPicFlag=1; idr_pic_id
        // and (no_output_of_prior_pics_flag, long_term_reference_flag)
        // present.
        let sps = dummy_sps();
        let pps = dummy_pps();
        let nh = nal(NalUnitType::SliceIdr, 3);

        let mut w = BitWriter::new();
        w.ue(0); // first_mb_in_slice
        w.ue(7); // slice_type = 7 (I, all same)
        w.ue(0); // pps_id
        w.u(4, 0); // frame_num
        w.ue(42); // idr_pic_id = 42
        w.u(4, 12); // pic_order_cnt_lsb = 12
        w.u(1, 1); // no_output_of_prior_pics_flag
        w.u(1, 0); // long_term_reference_flag
        w.se(-4); // slice_qp_delta = -4

        let bytes = w.into_bytes();
        let hdr = SliceHeader::parse(&bytes, &sps, &pps, &nh).unwrap();
        assert_eq!(hdr.slice_type, SliceType::I);
        assert_eq!(hdr.idr_pic_id, 42);
        assert_eq!(hdr.pic_order_cnt_lsb, 12);
        assert_eq!(hdr.slice_qp_delta, -4);
        let drm = hdr.dec_ref_pic_marking.as_ref().unwrap();
        assert!(drm.no_output_of_prior_pics_flag);
        assert!(!drm.long_term_reference_flag);
        assert!(drm.adaptive_marking.is_none());
    }

    #[test]
    fn idr_with_non_intra_type_is_rejected() {
        let sps = dummy_sps();
        let pps = dummy_pps();
        let nh = nal(NalUnitType::SliceIdr, 3);

        let mut w = BitWriter::new();
        w.ue(0); // first_mb_in_slice
        w.ue(0); // slice_type = 0 (P) — illegal for IDR
        w.ue(0); // pps_id
        let bytes = w.into_bytes();
        let err = SliceHeader::parse(&bytes, &sps, &pps, &nh).unwrap_err();
        assert_eq!(err, SliceHeaderError::IdrSliceTypeNotI(0));
    }

    #[test]
    fn p_slice_with_num_ref_override() {
        // Non-IDR P-slice overriding num_ref_idx_l0_active_minus1.
        let sps = dummy_sps();
        let mut pps = dummy_pps();
        pps.num_ref_idx_l0_default_active_minus1 = 0; // default is 0
        let nh = nal(NalUnitType::SliceNonIdr, 2);

        let mut w = BitWriter::new();
        w.ue(0); // first_mb_in_slice
        w.ue(0); // slice_type = 0 (P)
        w.ue(0); // pps_id
        w.u(4, 3); // frame_num = 3
        w.u(4, 6); // pic_order_cnt_lsb = 6
        // redundant_pic_cnt_present_flag=0
        // P slice → num_ref_idx_active_override_flag
        w.u(1, 1); // override = 1
        w.ue(2); // num_ref_idx_l0_active_minus1 = 2
        // no l1
        // ref_pic_list_modification for P (has list 0, no list 1):
        w.u(1, 0); // ref_pic_list_modification_flag_l0 = 0
        // weighted_pred_flag=0 AND slice_type=P: no pred_weight_table.
        // nal_ref_idc != 0 → dec_ref_pic_marking():
        w.u(1, 0); // adaptive_ref_pic_marking_mode_flag
        // entropy_coding_mode=0 → no cabac_init_idc
        w.se(1); // slice_qp_delta

        let bytes = w.into_bytes();
        let hdr = SliceHeader::parse(&bytes, &sps, &pps, &nh).unwrap();
        assert_eq!(hdr.slice_type, SliceType::P);
        assert_eq!(hdr.frame_num, 3);
        assert_eq!(hdr.pic_order_cnt_lsb, 6);
        assert!(hdr.num_ref_idx_active_override_flag);
        assert_eq!(hdr.num_ref_idx_l0_active_minus1, 2);
        assert_eq!(hdr.num_ref_idx_l1_active_minus1, 0);
        assert_eq!(hdr.slice_qp_delta, 1);
    }

    #[test]
    fn b_slice_with_pred_weight_table_implicit_bipred() {
        // B slice, weighted_bipred_idc=1 → explicit pred_weight_table.
        let sps = dummy_sps();
        let mut pps = dummy_pps();
        pps.weighted_bipred_idc = 1;
        pps.num_ref_idx_l0_default_active_minus1 = 0;
        pps.num_ref_idx_l1_default_active_minus1 = 0;
        let nh = nal(NalUnitType::SliceNonIdr, 2);

        let mut w = BitWriter::new();
        w.ue(0); // first_mb_in_slice
        w.ue(1); // slice_type = 1 (B)
        w.ue(0); // pps_id
        w.u(4, 5); // frame_num
        w.u(4, 10); // pic_order_cnt_lsb
        // redundant_pic_cnt_present_flag=0
        // B → direct_spatial_mv_pred_flag
        w.u(1, 1);
        // P/SP/B → num_ref_idx_active_override_flag
        w.u(1, 0); // no override
        // ref_pic_list_modification:
        w.u(1, 0); // list 0 flag
        w.u(1, 0); // list 1 flag
        // pred_weight_table (ChromaArrayType=1 for 4:2:0):
        w.ue(1); // luma_log2_weight_denom = 1
        w.ue(0); // chroma_log2_weight_denom = 0
        // list 0: one entry
        w.u(1, 1); // luma_weight_l0_flag
        w.se(2); // luma_weight_l0
        w.se(-1); // luma_offset_l0
        w.u(1, 0); // chroma_weight_l0_flag = 0
        // list 1: one entry (B has list 1)
        w.u(1, 0); // luma_weight_l1_flag = 0
        w.u(1, 1); // chroma_weight_l1_flag = 1
        w.se(3); // chroma_weight_l1[0][0] (Cb weight)
        w.se(0); // chroma_offset_l1[0][0]
        w.se(-2); // chroma_weight_l1[0][1] (Cr weight)
        w.se(1); // chroma_offset_l1[0][1]
        // nal_ref_idc != 0 → dec_ref_pic_marking:
        w.u(1, 0); // sliding window
        // entropy=0 and B (not I/SI): cabac_init_idc present only with CABAC, so skip.
        w.se(0); // slice_qp_delta

        let bytes = w.into_bytes();
        let hdr = SliceHeader::parse(&bytes, &sps, &pps, &nh).unwrap();
        assert_eq!(hdr.slice_type, SliceType::B);
        assert!(hdr.direct_spatial_mv_pred_flag);
        let pwt = hdr.pred_weight_table.as_ref().unwrap();
        assert_eq!(pwt.luma_log2_weight_denom, 1);
        assert_eq!(pwt.chroma_log2_weight_denom, 0);
        assert_eq!(pwt.luma_weights_l0, vec![Some((2, -1))]);
        assert_eq!(pwt.chroma_weights_l0, vec![None]);
        assert_eq!(pwt.luma_weights_l1, vec![None]);
        assert_eq!(pwt.chroma_weights_l1, vec![Some([(3, 0), (-2, 1)])]);
    }

    #[test]
    fn non_idr_p_slice_with_mmco_ops() {
        // Non-IDR P slice with adaptive ref marking containing MMCO 1, 5, 6.
        let sps = dummy_sps();
        let pps = dummy_pps();
        let nh = nal(NalUnitType::SliceNonIdr, 3);

        let mut w = BitWriter::new();
        w.ue(0); // first_mb
        w.ue(0); // slice_type P
        w.ue(0); // pps_id
        w.u(4, 1); // frame_num
        w.u(4, 4); // pic_order_cnt_lsb
        // no redundant_pic_cnt
        // P: num_ref_idx_active_override_flag
        w.u(1, 0);
        // ref_pic_list_modification:
        w.u(1, 0); // flag_l0 = 0
        // dec_ref_pic_marking (non-IDR):
        w.u(1, 1); // adaptive_ref_pic_marking_mode_flag = 1
        w.ue(1); // MMCO 1
        w.ue(5); // difference_of_pic_nums_minus1
        w.ue(5); // MMCO 5
        w.ue(6); // MMCO 6
        w.ue(2); // long_term_frame_idx
        w.ue(0); // MMCO 0 terminator
        w.se(-1); // slice_qp_delta

        let bytes = w.into_bytes();
        let hdr = SliceHeader::parse(&bytes, &sps, &pps, &nh).unwrap();
        let drm = hdr.dec_ref_pic_marking.as_ref().unwrap();
        let ops = drm.adaptive_marking.as_ref().unwrap();
        assert_eq!(
            ops,
            &vec![
                MmcoOp::MarkShortTermUnused(5),
                MmcoOp::MarkAllUnused,
                MmcoOp::AssignCurrentLongTerm(2),
            ]
        );
    }

    #[test]
    fn p_slice_with_ref_pic_list_modification_ops() {
        // P slice with ref_pic_list_modification containing
        // Subtract(1), Add(2), LongTerm(4), then terminator.
        let sps = dummy_sps();
        let pps = dummy_pps();
        let nh = nal(NalUnitType::SliceNonIdr, 2);

        let mut w = BitWriter::new();
        w.ue(0); // first_mb
        w.ue(0); // P
        w.ue(0); // pps_id
        w.u(4, 2); // frame_num
        w.u(4, 2); // pic_order_cnt_lsb
        // no redundant
        w.u(1, 0); // num_ref_idx_active_override_flag = 0
        // ref_pic_list_modification:
        w.u(1, 1); // flag_l0 = 1
        w.ue(0); // idc = 0 Subtract
        w.ue(1); // abs_diff - 1 = 1
        w.ue(1); // idc = 1 Add
        w.ue(2);
        w.ue(2); // idc = 2 LongTerm
        w.ue(4);
        w.ue(3); // idc = 3 terminate
        // nal_ref_idc != 0 → dec_ref_pic_marking (non-IDR, sliding window):
        w.u(1, 0);
        w.se(0); // slice_qp_delta

        let bytes = w.into_bytes();
        let hdr = SliceHeader::parse(&bytes, &sps, &pps, &nh).unwrap();
        let mods = &hdr.ref_pic_list_modification.modifications_l0;
        assert_eq!(mods.len(), 3);
        assert_eq!(mods[0], RefPicListModificationOp::Subtract(1));
        assert_eq!(mods[1], RefPicListModificationOp::Add(2));
        assert_eq!(mods[2], RefPicListModificationOp::LongTerm(4));
        assert!(hdr.ref_pic_list_modification.modifications_l1.is_empty());
    }

    #[test]
    fn i_slice_with_deblocking_control_fields() {
        // PPS has deblocking_filter_control_present_flag=1, so the
        // slice header carries disable_deblocking_filter_idc and, when
        // idc != 1, alpha/beta offsets.
        let sps = dummy_sps();
        let mut pps = dummy_pps();
        pps.deblocking_filter_control_present_flag = true;
        let nh = nal(NalUnitType::SliceNonIdr, 3);

        let mut w = BitWriter::new();
        w.ue(0); // first_mb
        w.ue(7); // slice_type = 7 (I all same)
        w.ue(0); // pps_id
        w.u(4, 0); // frame_num
        w.u(4, 0); // pic_order_cnt_lsb
        // no redundant
        // I → no override / no direct_spatial
        // I → no ref_pic_list_modification for list 0, no list 1.
        // nal_ref_idc != 0: dec_ref_pic_marking (non-IDR sliding window).
        w.u(1, 0);
        // I + CAVLC: no cabac_init_idc.
        w.se(0); // slice_qp_delta
        // no SP/SI fields
        // deblocking group present:
        w.ue(0); // disable_deblocking_filter_idc = 0
        w.se(-2); // slice_alpha_c0_offset_div2
        w.se(3); // slice_beta_offset_div2

        let bytes = w.into_bytes();
        let hdr = SliceHeader::parse(&bytes, &sps, &pps, &nh).unwrap();
        assert_eq!(hdr.disable_deblocking_filter_idc, 0);
        assert_eq!(hdr.slice_alpha_c0_offset_div2, -2);
        assert_eq!(hdr.slice_beta_offset_div2, 3);
    }

    #[test]
    fn deblocking_idc_1_skips_offsets() {
        // When disable_deblocking_filter_idc == 1 the offsets are absent.
        let sps = dummy_sps();
        let mut pps = dummy_pps();
        pps.deblocking_filter_control_present_flag = true;
        let nh = nal(NalUnitType::SliceNonIdr, 3);

        let mut w = BitWriter::new();
        w.ue(0); // first_mb
        w.ue(7); // I all same
        w.ue(0);
        w.u(4, 0);
        w.u(4, 0);
        // dec_ref_pic_marking non-IDR
        w.u(1, 0);
        w.se(0); // slice_qp_delta
        w.ue(1); // disable_deblocking_filter_idc = 1 → no offsets

        let bytes = w.into_bytes();
        let hdr = SliceHeader::parse(&bytes, &sps, &pps, &nh).unwrap();
        assert_eq!(hdr.disable_deblocking_filter_idc, 1);
        assert_eq!(hdr.slice_alpha_c0_offset_div2, 0);
        assert_eq!(hdr.slice_beta_offset_div2, 0);
    }

    #[test]
    fn field_pic_and_bottom_field_flags_read_when_not_frame_mbs_only() {
        let mut sps = dummy_sps();
        sps.frame_mbs_only_flag = false;
        sps.mb_adaptive_frame_field_flag = false;
        let pps = dummy_pps();
        let nh = nal(NalUnitType::SliceNonIdr, 3);

        let mut w = BitWriter::new();
        w.ue(0); // first_mb
        w.ue(7); // I all same
        w.ue(0); // pps_id
        w.u(4, 0); // frame_num
        // frame_mbs_only=0 → field_pic_flag, then bottom_field_flag if set.
        w.u(1, 1); // field_pic_flag = 1
        w.u(1, 1); // bottom_field_flag = 1
        w.u(4, 0); // pic_order_cnt_lsb
        // dec_ref_pic_marking:
        w.u(1, 0);
        w.se(0);

        let bytes = w.into_bytes();
        let hdr = SliceHeader::parse(&bytes, &sps, &pps, &nh).unwrap();
        assert!(hdr.field_pic_flag);
        assert!(hdr.bottom_field_flag);
    }

    #[test]
    fn cabac_init_idc_present_under_cabac_and_non_intra() {
        let sps = dummy_sps();
        let mut pps = dummy_pps();
        pps.entropy_coding_mode_flag = true;
        let nh = nal(NalUnitType::SliceNonIdr, 2);

        let mut w = BitWriter::new();
        w.ue(0); // first_mb
        w.ue(0); // P
        w.ue(0); // pps_id
        w.u(4, 1); // frame_num
        w.u(4, 0); // pic_order_cnt_lsb
        // no redundant
        // P → num_ref_idx_active_override_flag
        w.u(1, 0);
        // ref_pic_list_modification (list 0 only):
        w.u(1, 0);
        // weighted_pred=0 → no pwt.
        // dec_ref_pic_marking:
        w.u(1, 0);
        // CABAC + non-intra: cabac_init_idc
        w.ue(2);
        w.se(0); // slice_qp_delta

        let bytes = w.into_bytes();
        let hdr = SliceHeader::parse(&bytes, &sps, &pps, &nh).unwrap();
        assert_eq!(hdr.cabac_init_idc, 2);
    }

    #[test]
    fn no_dec_ref_pic_marking_when_ref_idc_zero() {
        // nal_ref_idc=0 ⇒ dec_ref_pic_marking() is absent.
        let sps = dummy_sps();
        let pps = dummy_pps();
        let nh = nal(NalUnitType::SliceNonIdr, 0);

        let mut w = BitWriter::new();
        w.ue(0); // first_mb
        w.ue(7); // I all same
        w.ue(0);
        w.u(4, 0);
        w.u(4, 0);
        w.se(0); // slice_qp_delta

        let bytes = w.into_bytes();
        let hdr = SliceHeader::parse(&bytes, &sps, &pps, &nh).unwrap();
        assert!(hdr.dec_ref_pic_marking.is_none());
    }

    #[test]
    fn slice_extension_nal_type_is_rejected() {
        let sps = dummy_sps();
        let pps = dummy_pps();
        let nh = nal(NalUnitType::SliceExtension, 3);
        let err = SliceHeader::parse(&[], &sps, &pps, &nh).unwrap_err();
        assert_eq!(err, SliceHeaderError::SliceExtensionNotSupported(20));
    }

    #[test]
    fn invalid_slice_type_rejected() {
        let sps = dummy_sps();
        let pps = dummy_pps();
        let nh = nal(NalUnitType::SliceNonIdr, 3);
        let mut w = BitWriter::new();
        w.ue(0); // first_mb
        w.ue(10); // slice_type = 10 → out of range
        let bytes = w.into_bytes();
        let err = SliceHeader::parse(&bytes, &sps, &pps, &nh).unwrap_err();
        assert_eq!(err, SliceHeaderError::SliceTypeOutOfRange(10));
    }

    #[test]
    fn ceil_log2_matches_pps_helper() {
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
    }

    #[test]
    fn slice_group_change_cycle_read_when_fmo_changing() {
        // num_slice_groups_minus1=1 + changing map (type 3) → slice header
        // carries slice_group_change_cycle with bit width per eq. (7-37).
        let sps = dummy_sps();
        let mut pps = dummy_pps();
        pps.num_slice_groups_minus1 = 1;
        pps.slice_group_map = Some(crate::pps::SliceGroupMap::Changing {
            slice_group_map_type: 3,
            change_direction_flag: false,
            change_rate_minus1: 0, // SliceGroupChangeRate = 1
        });
        // PicSizeInMapUnits = 20 * 15 = 300. SliceGroupChangeRate = 1.
        // bits = Ceil(Log2(300 / 1 + 1)) = Ceil(Log2(301)) = 9.
        let nh = nal(NalUnitType::SliceNonIdr, 3);

        let mut w = BitWriter::new();
        w.ue(0); // first_mb
        w.ue(7); // I all same
        w.ue(0);
        w.u(4, 0);
        w.u(4, 0);
        // dec_ref_pic_marking non-IDR
        w.u(1, 0);
        w.se(0); // slice_qp_delta
        // num_slice_groups_minus1 > 0 AND map type is Changing → read cycle.
        w.u(9, 123); // slice_group_change_cycle = 123

        let bytes = w.into_bytes();
        let hdr = SliceHeader::parse(&bytes, &sps, &pps, &nh).unwrap();
        assert_eq!(hdr.slice_group_change_cycle, 123);
    }
}
