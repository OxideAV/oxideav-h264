//! §7.3.2.1.3 — Subset sequence parameter set RBSP parsing.
//!
//! A subset SPS (NAL unit type 15) wraps an ordinary
//! `seq_parameter_set_data()` body (§7.3.2.1.1, shared with
//! [`crate::sps::Sps`]) followed by a per-profile extension structure:
//!
//! * `profile_idc ∈ {83, 86}` — `seq_parameter_set_svc_extension()`
//!   (Annex F). Not parsed yet; surfaced as
//!   [`SubsetSpsExtension::SvcNotParsed`].
//! * `profile_idc ∈ {118, 128, 134}` —
//!   `seq_parameter_set_mvc_extension()` (§G.7.3.2.1.4) and the
//!   optional `mvc_vui_parameters_extension()` (§G.14.1). Fully
//!   parsed here.
//! * `profile_idc ∈ {138, 135}` — `seq_parameter_set_mvcd_extension()`
//!   (Annex H). Not parsed yet; surfaced as
//!   [`SubsetSpsExtension::MvcdNotParsed`].
//! * `profile_idc == 139` — MVCD + `seq_parameter_set_3davc_extension()`
//!   (Annex I). Not parsed yet; surfaced as
//!   [`SubsetSpsExtension::Mvcd3dNotParsed`].
//!
//! Per §7.4.1.2.1, subset sequence parameter sets use a
//! `seq_parameter_set_id` value space *separate* from that of ordinary
//! sequence parameter sets — [`crate::decoder::Decoder`] stores them in
//! their own table.
//!
//! Semantics references used for range checks:
//!   * §7.4.2.1.3 — subset SPS RBSP semantics (`bit_equal_to_one`,
//!     `additional_extension2_flag`).
//!   * §G.7.4.2.1.4 — MVC extension semantics (all `0..=1023` view-id
//!     bounds, the `Min(15, num_views_minus1)` inter-view reference
//!     count cap, `num_level_values_signalled_minus1 ≤ 63`, the
//!     `{4, 8, 12}` grid-position constraint, and the
//!     `default_grid_position_flag == 1` inference rules).
//!   * §G.14.2 — MVC VUI extension semantics (`0..=1023` bounds on
//!     operation-point counts and view ids).

use crate::bitstream::{BitError, BitReader};
use crate::sps::{Sps, SpsError};
use crate::vui::{HrdParameters, VuiError};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SubsetSpsError {
    #[error("bitstream read failed: {0}")]
    Bitstream(#[from] BitError),
    #[error("seq_parameter_set_data parse failed: {0}")]
    Sps(#[from] SpsError),
    /// §G.14.1 — the embedded `hrd_parameters()` structures reuse the
    /// §E.1.2 parser.
    #[error("embedded hrd_parameters parse failed: {0}")]
    Hrd(#[from] VuiError),
    /// §7.4.2.1.3 — "bit_equal_to_one shall be equal to 1".
    #[error("bit_equal_to_one is 0")]
    BitEqualToOneViolated,
    /// §G.7.4.2.1.4 — `num_views_minus1` in 0..=1023. Checked before
    /// allocating any per-view storage (anti-OOM gate).
    #[error("num_views_minus1 out of range (got {0}, max 1023)")]
    NumViewsOutOfRange(u32),
    /// §G.7.4.2.1.4 — `view_id[i]` in 0..=1023.
    #[error("view_id[{i}] out of range (got {value}, max 1023)")]
    ViewIdOutOfRange { i: usize, value: u32 },
    /// §G.7.4.2.1.4 — `num_anchor_refs_lX[i]` /
    /// `num_non_anchor_refs_lX[i]` "shall not be greater than
    /// Min( 15, num_views_minus1 )". Checked before allocating the
    /// reference list.
    #[error(
        "num_{kind}_refs_l{list}[{i}] out of range (got {value}, max {max} = Min(15, num_views_minus1))"
    )]
    NumInterViewRefsOutOfRange {
        /// `"anchor"` or `"non_anchor"`.
        kind: &'static str,
        list: u8,
        i: usize,
        value: u32,
        max: u32,
    },
    /// §G.7.4.2.1.4 — `anchor_ref_lX[i][j]` / `non_anchor_ref_lX[i][j]`
    /// in 0..=1023.
    #[error("{kind}_ref_l{list}[{i}][{j}] out of range (got {value}, max 1023)")]
    InterViewRefOutOfRange {
        /// `"anchor"` or `"non_anchor"`.
        kind: &'static str,
        list: u8,
        i: usize,
        j: usize,
        value: u32,
    },
    /// §G.7.4.2.1.4 — `num_level_values_signalled_minus1` in 0..=63.
    #[error("num_level_values_signalled_minus1 out of range (got {0}, max 63)")]
    NumLevelValuesOutOfRange(u32),
    /// §G.7.4.2.1.4 — `num_applicable_ops_minus1[i]` in 0..=1023.
    #[error("num_applicable_ops_minus1[{i}] out of range (got {value}, max 1023)")]
    NumApplicableOpsOutOfRange { i: usize, value: u32 },
    /// §G.7.4.2.1.4 — `applicable_op_num_target_views_minus1[i][j]` in
    /// 0..=1023.
    #[error(
        "applicable_op_num_target_views_minus1[{i}][{j}] out of range (got {value}, max 1023)"
    )]
    NumTargetViewsOutOfRange { i: usize, j: usize, value: u32 },
    /// §G.7.4.2.1.4 — `applicable_op_target_view_id[i][j][k]` in
    /// 0..=1023.
    #[error("applicable_op_target_view_id[{i}][{j}][{k}] out of range (got {value}, max 1023)")]
    TargetViewIdOutOfRange {
        i: usize,
        j: usize,
        k: usize,
        value: u32,
    },
    /// §G.7.4.2.1.4 — `applicable_op_num_views_minus1[i][j]` in 0..=1023.
    #[error("applicable_op_num_views_minus1[{i}][{j}] out of range (got {value}, max 1023)")]
    OpNumViewsOutOfRange { i: usize, j: usize, value: u32 },
    /// §G.7.4.2.1.4 — each of `view0_grid_position_x` /
    /// `view0_grid_position_y` / `view1_grid_position_x` /
    /// `view1_grid_position_y` "shall be equal to 4, 8 or 12".
    #[error("{field} must be 4, 8 or 12 (got {value})")]
    GridPositionInvalid { field: &'static str, value: u8 },
    /// §G.14.2 — `vui_mvc_num_ops_minus1` in 0..=1023.
    #[error("vui_mvc_num_ops_minus1 out of range (got {0}, max 1023)")]
    VuiNumOpsOutOfRange(u32),
    /// §G.14.2 — `vui_mvc_num_target_output_views_minus1[i]` in 0..=1023.
    #[error("vui_mvc_num_target_output_views_minus1[{i}] out of range (got {value}, max 1023)")]
    VuiNumTargetOutputViewsOutOfRange { i: usize, value: u32 },
    /// §G.14.2 — `vui_mvc_view_id[i][j]` in 0..=1023.
    #[error("vui_mvc_view_id[{i}][{j}] out of range (got {value}, max 1023)")]
    VuiViewIdOutOfRange { i: usize, j: usize, value: u32 },
}

/// §G.7.3.2.1.4 — the per-view inter-view dependency lists, indexed by
/// view order index (VOIdx). The syntax only signals entries for
/// `i ∈ 1..=num_views_minus1`; VOIdx 0 (the base view) carries no
/// inter-view references — §G.7.4.2.1.4 pins
/// `num_anchor_refs_lX[0] == 0` and `num_non_anchor_refs_lX[0] == 0` —
/// so its entry here is always four empty lists.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MvcViewDependency {
    /// `anchor_ref_l0[i][..]` — view_id values (0..=1023) of the view
    /// components available for inter-view prediction in the initial
    /// RefPicList0 when decoding *anchor* view components with this
    /// VOIdx (§G.8.2.1).
    pub anchor_refs_l0: Vec<u16>,
    /// `anchor_ref_l1[i][..]` — RefPicList1 counterpart.
    pub anchor_refs_l1: Vec<u16>,
    /// `non_anchor_ref_l0[i][..]` — RefPicList0 inter-view references
    /// when decoding *non-anchor* view components with this VOIdx.
    pub non_anchor_refs_l0: Vec<u16>,
    /// `non_anchor_ref_l1[i][..]` — RefPicList1 counterpart.
    pub non_anchor_refs_l1: Vec<u16>,
}

/// §G.7.3.2.1.4 — one operation point to which a signalled
/// `level_idc[i]` applies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvcApplicableOp {
    /// `applicable_op_temporal_id[i][j]` — u(3).
    pub temporal_id: u8,
    /// `applicable_op_target_view_id[i][j][..]` — the target output
    /// views of this operation point (each 0..=1023).
    pub target_view_ids: Vec<u16>,
    /// `applicable_op_num_views_minus1[i][j]` — plus 1 is the number of
    /// views required to decode the target output views (including the
    /// views they transitively depend on, per the §G.8.5 sub-bitstream
    /// extraction process).
    pub num_views_minus1: u16,
}

/// §G.7.3.2.1.4 — one signalled level value and the operation points it
/// applies to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvcLevelValue {
    /// `level_idc[i]` — u(8).
    pub level_idc: u8,
    /// One entry per `j ∈ 0..=num_applicable_ops_minus1[i]`.
    pub applicable_ops: Vec<MvcApplicableOp>,
}

/// §G.7.3.2.1.4 — the trailing MFC (frame-compatible) block, present
/// only when `profile_idc == 134` (MFC High).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MfcExtension {
    /// `mfc_format_idc` — u(6). Table G-1: 0 = base view side-by-side
    /// (non-base view inferred top-bottom), 1 = base view top-bottom
    /// (non-base view inferred side-by-side). Values 2..=63 are
    /// reserved; §G.7.4.2.1.4 directs decoders to ignore the coded
    /// video sequence when the value is greater than 1, so the parser
    /// stores the raw value rather than rejecting.
    pub mfc_format_idc: u8,
    /// `default_grid_position_flag` — present only when
    /// `mfc_format_idc ∈ {0, 1}`.
    pub default_grid_position_flag: Option<bool>,
    /// `(view0_grid_position_x, view0_grid_position_y,
    /// view1_grid_position_x, view1_grid_position_y)` — present only
    /// when `default_grid_position_flag == 0`; each constrained to
    /// {4, 8, 12} by §G.7.4.2.1.4.
    pub explicit_grid_position: Option<(u8, u8, u8, u8)>,
    /// `rpu_filter_enabled_flag` — 1: down/upsampling filter processes
    /// generate inter-view prediction references; 0: all their sample
    /// values are set equal to 128.
    pub rpu_filter_enabled_flag: bool,
    /// `rpu_field_processing_flag` — present only when
    /// `frame_mbs_only_flag == 0`. Use
    /// [`MfcExtension::rpu_field_processing`] for the §G.7.4.2.1.4
    /// inferred semantic value.
    pub rpu_field_processing_flag: Option<bool>,
}

impl MfcExtension {
    /// §G.7.4.2.1.4 — the effective `(view0_grid_position_x,
    /// view0_grid_position_y, view1_grid_position_x,
    /// view1_grid_position_y)` quadruple: the explicit values when
    /// `default_grid_position_flag == 0`, otherwise the inferred set
    /// (`mfc_format_idc == 0` → (4, 8, 12, 8); `mfc_format_idc == 1` →
    /// (8, 4, 8, 12)). `None` when `mfc_format_idc` is a reserved value
    /// (2..=63), for which no grid position is defined.
    pub fn grid_positions(&self) -> Option<(u8, u8, u8, u8)> {
        if let Some(explicit) = self.explicit_grid_position {
            return Some(explicit);
        }
        match (self.default_grid_position_flag, self.mfc_format_idc) {
            (Some(true), 0) => Some((4, 8, 12, 8)),
            (Some(true), 1) => Some((8, 4, 8, 12)),
            _ => None,
        }
    }

    /// §G.7.4.2.1.4 — "When not present, the value of
    /// rpu_field_processing_flag is inferred to be equal to 0."
    pub fn rpu_field_processing(&self) -> bool {
        self.rpu_field_processing_flag.unwrap_or(false)
    }
}

/// §G.7.3.2.1.4 — `seq_parameter_set_mvc_extension()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpsMvcExtension {
    /// `view_id[i]` for `i ∈ 0..=num_views_minus1` — the view_id of the
    /// view with VOIdx equal to `i` (each 0..=1023). The vector length
    /// is `num_views_minus1 + 1`.
    pub view_ids: Vec<u16>,
    /// Inter-view dependency lists, one entry per VOIdx (same length as
    /// `view_ids`; index 0 is always empty — see
    /// [`MvcViewDependency`]).
    pub view_dependencies: Vec<MvcViewDependency>,
    /// `level_idc[i]` + operation points, one entry per
    /// `i ∈ 0..=num_level_values_signalled_minus1`.
    pub level_values: Vec<MvcLevelValue>,
    /// The `profile_idc == 134` MFC tail; `None` for profiles 118/128.
    pub mfc: Option<MfcExtension>,
}

impl SpsMvcExtension {
    /// `num_views_minus1 + 1` — the maximum number of coded views in
    /// the coded video sequence (§G.7.4.2.1.4 NOTE 2: the actual number
    /// of views present may be less).
    pub fn num_views(&self) -> usize {
        self.view_ids.len()
    }

    /// §G.7.3.2.1.4 — parse from the current cursor position.
    /// `profile_idc` selects the trailing MFC block (134 only) and
    /// `frame_mbs_only_flag` gates `rpu_field_processing_flag` inside
    /// it.
    pub fn parse(
        r: &mut BitReader<'_>,
        profile_idc: u8,
        frame_mbs_only_flag: bool,
    ) -> Result<Self, SubsetSpsError> {
        // §G.7.4.2.1.4 — num_views_minus1 in 0..=1023; checked before
        // any allocation sized from it.
        let num_views_minus1 = r.ue()?;
        if num_views_minus1 > 1023 {
            return Err(SubsetSpsError::NumViewsOutOfRange(num_views_minus1));
        }
        let num_views = num_views_minus1 as usize + 1;

        let mut view_ids = Vec::with_capacity(num_views);
        for i in 0..num_views {
            let value = r.ue()?;
            if value > 1023 {
                return Err(SubsetSpsError::ViewIdOutOfRange { i, value });
            }
            view_ids.push(value as u16);
        }

        // §G.7.4.2.1.4 — the per-list reference count cap
        // Min( 15, num_views_minus1 ).
        let max_refs = num_views_minus1.min(15);
        let mut view_dependencies = vec![MvcViewDependency::default(); num_views];
        // First loop: anchor references for i ∈ 1..=num_views_minus1.
        for (i, dep) in view_dependencies.iter_mut().enumerate().skip(1) {
            dep.anchor_refs_l0 = parse_inter_view_ref_list(r, "anchor", 0, i, max_refs)?;
            dep.anchor_refs_l1 = parse_inter_view_ref_list(r, "anchor", 1, i, max_refs)?;
        }
        // Second loop: non-anchor references for i ∈ 1..=num_views_minus1.
        for (i, dep) in view_dependencies.iter_mut().enumerate().skip(1) {
            dep.non_anchor_refs_l0 = parse_inter_view_ref_list(r, "non_anchor", 0, i, max_refs)?;
            dep.non_anchor_refs_l1 = parse_inter_view_ref_list(r, "non_anchor", 1, i, max_refs)?;
        }

        // §G.7.4.2.1.4 — num_level_values_signalled_minus1 in 0..=63.
        let num_level_values_minus1 = r.ue()?;
        if num_level_values_minus1 > 63 {
            return Err(SubsetSpsError::NumLevelValuesOutOfRange(
                num_level_values_minus1,
            ));
        }
        let mut level_values = Vec::with_capacity(num_level_values_minus1 as usize + 1);
        for i in 0..=num_level_values_minus1 as usize {
            let level_idc = r.u(8)? as u8;
            // §G.7.4.2.1.4 — num_applicable_ops_minus1[i] in 0..=1023.
            let num_ops_minus1 = r.ue()?;
            if num_ops_minus1 > 1023 {
                return Err(SubsetSpsError::NumApplicableOpsOutOfRange {
                    i,
                    value: num_ops_minus1,
                });
            }
            let mut applicable_ops = Vec::with_capacity(num_ops_minus1 as usize + 1);
            for j in 0..=num_ops_minus1 as usize {
                let temporal_id = r.u(3)? as u8;
                // §G.7.4.2.1.4 —
                // applicable_op_num_target_views_minus1[i][j] in 0..=1023.
                let num_target_views_minus1 = r.ue()?;
                if num_target_views_minus1 > 1023 {
                    return Err(SubsetSpsError::NumTargetViewsOutOfRange {
                        i,
                        j,
                        value: num_target_views_minus1,
                    });
                }
                let mut target_view_ids = Vec::with_capacity(num_target_views_minus1 as usize + 1);
                for k in 0..=num_target_views_minus1 as usize {
                    let value = r.ue()?;
                    if value > 1023 {
                        return Err(SubsetSpsError::TargetViewIdOutOfRange { i, j, k, value });
                    }
                    target_view_ids.push(value as u16);
                }
                // §G.7.4.2.1.4 — applicable_op_num_views_minus1[i][j]
                // in 0..=1023.
                let op_num_views_minus1 = r.ue()?;
                if op_num_views_minus1 > 1023 {
                    return Err(SubsetSpsError::OpNumViewsOutOfRange {
                        i,
                        j,
                        value: op_num_views_minus1,
                    });
                }
                applicable_ops.push(MvcApplicableOp {
                    temporal_id,
                    target_view_ids,
                    num_views_minus1: op_num_views_minus1 as u16,
                });
            }
            level_values.push(MvcLevelValue {
                level_idc,
                applicable_ops,
            });
        }

        // §G.7.3.2.1.4 — the MFC tail rides only on profile_idc 134.
        let mfc = if profile_idc == 134 {
            let mfc_format_idc = r.u(6)? as u8;
            let mut default_grid_position_flag = None;
            let mut explicit_grid_position = None;
            if mfc_format_idc == 0 || mfc_format_idc == 1 {
                let flag = r.u(1)? == 1;
                default_grid_position_flag = Some(flag);
                if !flag {
                    let v0x = read_grid_position(r, "view0_grid_position_x")?;
                    let v0y = read_grid_position(r, "view0_grid_position_y")?;
                    let v1x = read_grid_position(r, "view1_grid_position_x")?;
                    let v1y = read_grid_position(r, "view1_grid_position_y")?;
                    explicit_grid_position = Some((v0x, v0y, v1x, v1y));
                }
            }
            let rpu_filter_enabled_flag = r.u(1)? == 1;
            let rpu_field_processing_flag = if !frame_mbs_only_flag {
                Some(r.u(1)? == 1)
            } else {
                None
            };
            Some(MfcExtension {
                mfc_format_idc,
                default_grid_position_flag,
                explicit_grid_position,
                rpu_filter_enabled_flag,
                rpu_field_processing_flag,
            })
        } else {
            None
        };

        Ok(SpsMvcExtension {
            view_ids,
            view_dependencies,
            level_values,
            mfc,
        })
    }
}

/// §G.7.3.2.1.4 — one `num_*_refs_lX[i]` count + its `*_ref_lX[i][j]`
/// list, with the §G.7.4.2.1.4 `Min(15, num_views_minus1)` count cap
/// and the 0..=1023 view-id range check.
fn parse_inter_view_ref_list(
    r: &mut BitReader<'_>,
    kind: &'static str,
    list: u8,
    i: usize,
    max_refs: u32,
) -> Result<Vec<u16>, SubsetSpsError> {
    let count = r.ue()?;
    if count > max_refs {
        return Err(SubsetSpsError::NumInterViewRefsOutOfRange {
            kind,
            list,
            i,
            value: count,
            max: max_refs,
        });
    }
    let mut refs = Vec::with_capacity(count as usize);
    for j in 0..count as usize {
        let value = r.ue()?;
        if value > 1023 {
            return Err(SubsetSpsError::InterViewRefOutOfRange {
                kind,
                list,
                i,
                j,
                value,
            });
        }
        refs.push(value as u16);
    }
    Ok(refs)
}

/// §G.7.4.2.1.4 — read one u(4) grid-position field and enforce the
/// "shall be equal to 4, 8 or 12" constraint.
fn read_grid_position(r: &mut BitReader<'_>, field: &'static str) -> Result<u8, SubsetSpsError> {
    let value = r.u(4)? as u8;
    if !matches!(value, 4 | 8 | 12) {
        return Err(SubsetSpsError::GridPositionInvalid { field, value });
    }
    Ok(value)
}

/// §G.14.1 — the per-operation-point timing triple, mirroring the
/// §E.2.1 `timing_info` semantics for the sub-bitstream extracted at
/// this operation point (§G.14.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvcVuiTimingInfo {
    /// `vui_mvc_num_units_in_tick[i]` — u(32).
    pub num_units_in_tick: u32,
    /// `vui_mvc_time_scale[i]` — u(32).
    pub time_scale: u32,
    /// `vui_mvc_fixed_frame_rate_flag[i]` — u(1).
    pub fixed_frame_rate_flag: bool,
}

/// §G.14.1 — one operation point inside
/// `mvc_vui_parameters_extension()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvcVuiOp {
    /// `vui_mvc_temporal_id[i]` — u(3); the maximum temporal_id of all
    /// VCL NAL units in this operation point's representation.
    pub temporal_id: u8,
    /// `vui_mvc_view_id[i][..]` — the target output views (each
    /// 0..=1023).
    pub target_output_view_ids: Vec<u16>,
    /// Present when `vui_mvc_timing_info_present_flag[i] == 1`.
    pub timing_info: Option<MvcVuiTimingInfo>,
    /// Present when `vui_mvc_nal_hrd_parameters_present_flag[i] == 1`.
    pub nal_hrd_parameters: Option<HrdParameters>,
    /// Present when `vui_mvc_vcl_hrd_parameters_present_flag[i] == 1`.
    pub vcl_hrd_parameters: Option<HrdParameters>,
    /// `vui_mvc_low_delay_hrd_flag[i]` — present only when one of the
    /// two HRD blocks is.
    pub low_delay_hrd_flag: Option<bool>,
    /// `vui_mvc_pic_struct_present_flag[i]` — u(1).
    pub pic_struct_present_flag: bool,
}

/// §G.14.1 — `mvc_vui_parameters_extension()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvcVuiParametersExtension {
    /// One entry per `i ∈ 0..=vui_mvc_num_ops_minus1`.
    pub ops: Vec<MvcVuiOp>,
}

impl MvcVuiParametersExtension {
    /// §G.14.1 — parse from the current cursor position.
    pub fn parse(r: &mut BitReader<'_>) -> Result<Self, SubsetSpsError> {
        // §G.14.2 — vui_mvc_num_ops_minus1 in 0..=1023; checked before
        // allocation.
        let num_ops_minus1 = r.ue()?;
        if num_ops_minus1 > 1023 {
            return Err(SubsetSpsError::VuiNumOpsOutOfRange(num_ops_minus1));
        }
        let mut ops = Vec::with_capacity(num_ops_minus1 as usize + 1);
        for i in 0..=num_ops_minus1 as usize {
            let temporal_id = r.u(3)? as u8;
            // §G.14.2 — vui_mvc_num_target_output_views_minus1[i] in
            // 0..=1023.
            let num_target_views_minus1 = r.ue()?;
            if num_target_views_minus1 > 1023 {
                return Err(SubsetSpsError::VuiNumTargetOutputViewsOutOfRange {
                    i,
                    value: num_target_views_minus1,
                });
            }
            let mut target_output_view_ids =
                Vec::with_capacity(num_target_views_minus1 as usize + 1);
            for j in 0..=num_target_views_minus1 as usize {
                let value = r.ue()?;
                if value > 1023 {
                    return Err(SubsetSpsError::VuiViewIdOutOfRange { i, j, value });
                }
                target_output_view_ids.push(value as u16);
            }
            let timing_info = if r.u(1)? == 1 {
                Some(MvcVuiTimingInfo {
                    num_units_in_tick: r.u(32)?,
                    time_scale: r.u(32)?,
                    fixed_frame_rate_flag: r.u(1)? == 1,
                })
            } else {
                None
            };
            let nal_hrd_parameters = if r.u(1)? == 1 {
                Some(HrdParameters::parse(r)?)
            } else {
                None
            };
            let vcl_hrd_parameters = if r.u(1)? == 1 {
                Some(HrdParameters::parse(r)?)
            } else {
                None
            };
            let low_delay_hrd_flag = if nal_hrd_parameters.is_some() || vcl_hrd_parameters.is_some()
            {
                Some(r.u(1)? == 1)
            } else {
                None
            };
            let pic_struct_present_flag = r.u(1)? == 1;
            ops.push(MvcVuiOp {
                temporal_id,
                target_output_view_ids,
                timing_info,
                nal_hrd_parameters,
                vcl_hrd_parameters,
                low_delay_hrd_flag,
                pic_struct_present_flag,
            });
        }
        Ok(MvcVuiParametersExtension { ops })
    }
}

/// §7.3.2.1.3 — the per-profile extension branch of a subset SPS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubsetSpsExtension {
    /// `profile_idc ∈ {118, 128, 134}` — the Annex G MVC extension and
    /// its optional MVC VUI.
    Mvc {
        extension: SpsMvcExtension,
        /// `Some(..)` when `mvc_vui_parameters_present_flag == 1`.
        vui: Option<MvcVuiParametersExtension>,
    },
    /// `profile_idc ∈ {83, 86}` — Annex F
    /// `seq_parameter_set_svc_extension()` is not parsed yet; the base
    /// `seq_parameter_set_data()` is still surfaced.
    SvcNotParsed,
    /// `profile_idc ∈ {138, 135}` — Annex H
    /// `seq_parameter_set_mvcd_extension()` is not parsed yet.
    MvcdNotParsed,
    /// `profile_idc == 139` — Annex H MVCD + Annex I 3D-AVC extensions
    /// are not parsed yet.
    Mvcd3dNotParsed,
    /// Any other (legal §A.2) `profile_idc` — the §7.3.2.1.3 syntax
    /// table has no extension branch for it; only
    /// `additional_extension2_flag` follows the base
    /// `seq_parameter_set_data()`.
    None,
}

/// §7.3.2.1.3 — `subset_seq_parameter_set_rbsp()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubsetSps {
    /// The embedded `seq_parameter_set_data()` body (§7.3.2.1.1).
    /// NOTE — per §7.4.1.2.1 its `seq_parameter_set_id` lives in a
    /// value space separate from ordinary SPSs.
    pub sps: Sps,
    /// The per-profile extension branch.
    pub extension: SubsetSpsExtension,
    /// §7.4.2.1.3 — "shall be equal to 0 in bitstreams conforming to
    /// this Recommendation | International Standard"; when 1, decoders
    /// ignore all data that follow it (this parser consumes and
    /// discards the `additional_extension2_data_flag` run). `None`
    /// when the extension branch was not parsed
    /// ([`SubsetSpsExtension::SvcNotParsed`] /
    /// [`SubsetSpsExtension::MvcdNotParsed`] /
    /// [`SubsetSpsExtension::Mvcd3dNotParsed`]) — the flag sits after
    /// the unparsed structure and cannot be located.
    pub additional_extension2_flag: Option<bool>,
}

impl SubsetSps {
    /// §7.3.2.1.3 — parse a `subset_seq_parameter_set_rbsp()` from a
    /// slice that holds the RBSP body (NAL header byte stripped,
    /// emulation prevention bytes removed — normally obtained from
    /// [`crate::nal::parse_nal_unit`] on a NAL unit of type 15).
    pub fn parse(rbsp: &[u8]) -> Result<Self, SubsetSpsError> {
        let mut r = BitReader::new(rbsp);
        let sps = Sps::parse_seq_parameter_set_data(&mut r)?;

        match sps.profile_idc {
            // Annex F SVC profiles — extension body not parsed yet.
            83 | 86 => Ok(SubsetSps {
                sps,
                extension: SubsetSpsExtension::SvcNotParsed,
                additional_extension2_flag: None,
            }),
            // Annex G MVC profiles (118 Multiview High, 128 Stereo
            // High, 134 MFC High).
            118 | 128 | 134 => {
                // §7.4.2.1.3 — bit_equal_to_one shall be equal to 1.
                if r.f(1)? != 1 {
                    return Err(SubsetSpsError::BitEqualToOneViolated);
                }
                let extension =
                    SpsMvcExtension::parse(&mut r, sps.profile_idc, sps.frame_mbs_only_flag)?;
                let vui = if r.u(1)? == 1 {
                    Some(MvcVuiParametersExtension::parse(&mut r)?)
                } else {
                    None
                };
                let additional_extension2_flag = Self::finish_rbsp(&mut r)?;
                Ok(SubsetSps {
                    sps,
                    extension: SubsetSpsExtension::Mvc { extension, vui },
                    additional_extension2_flag: Some(additional_extension2_flag),
                })
            }
            // Annex H MVCD profiles — extension body not parsed yet.
            138 | 135 => Ok(SubsetSps {
                sps,
                extension: SubsetSpsExtension::MvcdNotParsed,
                additional_extension2_flag: None,
            }),
            // Annex H MVCD + Annex I 3D-AVC — not parsed yet.
            139 => Ok(SubsetSps {
                sps,
                extension: SubsetSpsExtension::Mvcd3dNotParsed,
                additional_extension2_flag: None,
            }),
            // Every other legal §A.2 profile: the §7.3.2.1.3 if/else
            // chain has no branch, so additional_extension2_flag
            // directly follows seq_parameter_set_data().
            _ => {
                let additional_extension2_flag = Self::finish_rbsp(&mut r)?;
                Ok(SubsetSps {
                    sps,
                    extension: SubsetSpsExtension::None,
                    additional_extension2_flag: Some(additional_extension2_flag),
                })
            }
        }
    }

    /// §7.3.2.1.3 tail — `additional_extension2_flag`, the reserved
    /// `additional_extension2_data_flag` run when it is 1 (consumed and
    /// discarded per §7.4.2.1.3 "decoders shall ignore all data that
    /// follow"), and the rbsp_trailing_bits (lenient, mirroring
    /// [`Sps::parse`]).
    fn finish_rbsp(r: &mut BitReader<'_>) -> Result<bool, SubsetSpsError> {
        let additional_extension2_flag = r.u(1)? == 1;
        if additional_extension2_flag {
            while r.more_rbsp_data() {
                let _ = r.u(1)?;
            }
        }
        let _ = r.rbsp_trailing_bits();
        Ok(additional_extension2_flag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::{Decoder, Event};

    /// Tiny MSB-first bit writer — only for constructing test fixtures.
    /// Not part of the crate API (mirrors the helper in `sps::tests`).
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

        fn trailing(&mut self) {
            self.u(1, 1);
            while self.bit_pos != 0 {
                self.u(1, 0);
            }
        }

        fn into_bytes(self) -> Vec<u8> {
            self.bytes
        }
    }

    /// §7.3.2.1.1 — write a minimal `seq_parameter_set_data()` with the
    /// given `profile_idc` (4x4 MBs, 4:2:0 when the chroma-extended
    /// gate applies, poc type 0, no VUI).
    fn write_sps_data(w: &mut BitWriter, profile_idc: u8, frame_mbs_only: bool) {
        w.u(8, profile_idc as u32);
        w.u(8, 0); // constraint_set0..5_flag + reserved_zero_2bits
        w.u(8, 30); // level_idc
        w.ue(0); // seq_parameter_set_id
        if matches!(
            profile_idc,
            100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
        ) {
            w.ue(1); // chroma_format_idc — 4:2:0
            w.ue(0); // bit_depth_luma_minus8
            w.ue(0); // bit_depth_chroma_minus8
            w.u(1, 0); // qpprime_y_zero_transform_bypass_flag
            w.u(1, 0); // seq_scaling_matrix_present_flag
        }
        w.ue(0); // log2_max_frame_num_minus4
        w.ue(0); // pic_order_cnt_type
        w.ue(0); // log2_max_pic_order_cnt_lsb_minus4
        w.ue(1); // max_num_ref_frames
        w.u(1, 0); // gaps_in_frame_num_value_allowed_flag
        w.ue(3); // pic_width_in_mbs_minus1
        w.ue(3); // pic_height_in_map_units_minus1
        w.u(1, frame_mbs_only as u32); // frame_mbs_only_flag
        if !frame_mbs_only {
            w.u(1, 0); // mb_adaptive_frame_field_flag
        }
        w.u(1, 1); // direct_8x8_inference_flag
        w.u(1, 0); // frame_cropping_flag
        w.u(1, 0); // vui_parameters_present_flag
    }

    /// §G.7.3.2.1.4 — write the two-view MVC extension used by several
    /// tests: view_ids [0, 2], VOIdx-1 anchor refs l0/l1 = [0], non-
    /// anchor refs l0 = [0] / l1 = [], one level value (40) with one
    /// operation point.
    fn write_two_view_mvc_extension(w: &mut BitWriter) {
        w.ue(1); // num_views_minus1
        w.ue(0); // view_id[0]
        w.ue(2); // view_id[1]
        w.ue(1); // num_anchor_refs_l0[1]
        w.ue(0); // anchor_ref_l0[1][0]
        w.ue(1); // num_anchor_refs_l1[1]
        w.ue(0); // anchor_ref_l1[1][0]
        w.ue(1); // num_non_anchor_refs_l0[1]
        w.ue(0); // non_anchor_ref_l0[1][0]
        w.ue(0); // num_non_anchor_refs_l1[1]
        w.ue(0); // num_level_values_signalled_minus1
        w.u(8, 40); // level_idc[0]
        w.ue(0); // num_applicable_ops_minus1[0]
        w.u(3, 1); // applicable_op_temporal_id[0][0]
        w.ue(1); // applicable_op_num_target_views_minus1[0][0]
        w.ue(0); // applicable_op_target_view_id[0][0][0]
        w.ue(2); // applicable_op_target_view_id[0][0][1]
        w.ue(1); // applicable_op_num_views_minus1[0][0]
    }

    #[test]
    fn mvc_two_views_full_round_trip() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 118, true);
        w.u(1, 1); // bit_equal_to_one
        write_two_view_mvc_extension(&mut w);
        w.u(1, 0); // mvc_vui_parameters_present_flag
        w.u(1, 0); // additional_extension2_flag
        w.trailing();

        let subset = SubsetSps::parse(&w.into_bytes()).expect("two-view MVC subset SPS");
        assert_eq!(subset.sps.profile_idc, 118);
        assert_eq!(subset.sps.seq_parameter_set_id, 0);
        assert_eq!(subset.additional_extension2_flag, Some(false));
        let SubsetSpsExtension::Mvc { extension, vui } = &subset.extension else {
            panic!(
                "expected the Mvc extension variant, got {:?}",
                subset.extension
            );
        };
        assert!(vui.is_none());
        assert_eq!(extension.num_views(), 2);
        assert_eq!(extension.view_ids, vec![0, 2]);
        // §G.7.4.2.1.4 — VOIdx 0 carries no inter-view references.
        assert_eq!(extension.view_dependencies[0], MvcViewDependency::default());
        assert_eq!(extension.view_dependencies[1].anchor_refs_l0, vec![0]);
        assert_eq!(extension.view_dependencies[1].anchor_refs_l1, vec![0]);
        assert_eq!(extension.view_dependencies[1].non_anchor_refs_l0, vec![0]);
        assert!(extension.view_dependencies[1].non_anchor_refs_l1.is_empty());
        assert_eq!(extension.level_values.len(), 1);
        let lv = &extension.level_values[0];
        assert_eq!(lv.level_idc, 40);
        assert_eq!(lv.applicable_ops.len(), 1);
        let op = &lv.applicable_ops[0];
        assert_eq!(op.temporal_id, 1);
        assert_eq!(op.target_view_ids, vec![0, 2]);
        assert_eq!(op.num_views_minus1, 1);
        assert!(extension.mfc.is_none()); // profile 118 has no MFC tail
    }

    #[test]
    fn stereo_high_minimal_single_view() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 128, true);
        w.u(1, 1); // bit_equal_to_one
        w.ue(0); // num_views_minus1 — both i ∈ 1.. loops are empty
        w.ue(5); // view_id[0]
        w.ue(0); // num_level_values_signalled_minus1
        w.u(8, 9); // level_idc[0]
        w.ue(0); // num_applicable_ops_minus1[0]
        w.u(3, 0); // applicable_op_temporal_id[0][0]
        w.ue(0); // applicable_op_num_target_views_minus1[0][0]
        w.ue(5); // applicable_op_target_view_id[0][0][0]
        w.ue(0); // applicable_op_num_views_minus1[0][0]
        w.u(1, 0); // mvc_vui_parameters_present_flag
        w.u(1, 0); // additional_extension2_flag
        w.trailing();

        let subset = SubsetSps::parse(&w.into_bytes()).expect("single-view subset SPS");
        let SubsetSpsExtension::Mvc { extension, .. } = &subset.extension else {
            panic!("expected Mvc variant");
        };
        assert_eq!(extension.num_views(), 1);
        assert_eq!(extension.view_ids, vec![5]);
        assert_eq!(extension.view_dependencies.len(), 1);
        assert_eq!(extension.view_dependencies[0], MvcViewDependency::default());
    }

    /// §G.7.3.2.1.4 / §G.7.4.2.1.4 — profile 134 MFC tail with
    /// `default_grid_position_flag == 1` and `mfc_format_idc == 0`:
    /// grid positions are inferred as (4, 8, 12, 8).
    #[test]
    fn mfc_default_grid_position_inference() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 134, true);
        w.u(1, 1); // bit_equal_to_one
        write_two_view_mvc_extension(&mut w);
        w.u(6, 0); // mfc_format_idc — side-by-side base view
        w.u(1, 1); // default_grid_position_flag
        w.u(1, 1); // rpu_filter_enabled_flag
                   // frame_mbs_only_flag == 1 ⇒ no rpu_field_processing_flag
        w.u(1, 0); // mvc_vui_parameters_present_flag
        w.u(1, 0); // additional_extension2_flag
        w.trailing();

        let subset = SubsetSps::parse(&w.into_bytes()).expect("MFC subset SPS");
        let SubsetSpsExtension::Mvc { extension, .. } = &subset.extension else {
            panic!("expected Mvc variant");
        };
        let mfc = extension.mfc.as_ref().expect("profile 134 carries MFC");
        assert_eq!(mfc.mfc_format_idc, 0);
        assert_eq!(mfc.default_grid_position_flag, Some(true));
        assert!(mfc.explicit_grid_position.is_none());
        assert_eq!(mfc.grid_positions(), Some((4, 8, 12, 8)));
        assert!(mfc.rpu_filter_enabled_flag);
        assert_eq!(mfc.rpu_field_processing_flag, None);
        // §G.7.4.2.1.4 — inferred 0 when not present.
        assert!(!mfc.rpu_field_processing());
    }

    /// Profile 134 with explicit grid positions and
    /// `frame_mbs_only_flag == 0` (so `rpu_field_processing_flag` is
    /// present).
    #[test]
    fn mfc_explicit_grid_position_and_field_flag() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 134, false);
        w.u(1, 1); // bit_equal_to_one
        write_two_view_mvc_extension(&mut w);
        w.u(6, 1); // mfc_format_idc — top-bottom base view
        w.u(1, 0); // default_grid_position_flag
        w.u(4, 4); // view0_grid_position_x
        w.u(4, 8); // view0_grid_position_y
        w.u(4, 12); // view1_grid_position_x
        w.u(4, 4); // view1_grid_position_y
        w.u(1, 0); // rpu_filter_enabled_flag
        w.u(1, 1); // rpu_field_processing_flag (frame_mbs_only == 0)
        w.u(1, 0); // mvc_vui_parameters_present_flag
        w.u(1, 0); // additional_extension2_flag
        w.trailing();

        let subset = SubsetSps::parse(&w.into_bytes()).expect("MFC explicit-grid subset SPS");
        let SubsetSpsExtension::Mvc { extension, .. } = &subset.extension else {
            panic!("expected Mvc variant");
        };
        let mfc = extension.mfc.as_ref().expect("MFC tail");
        assert_eq!(mfc.mfc_format_idc, 1);
        assert_eq!(mfc.default_grid_position_flag, Some(false));
        assert_eq!(mfc.explicit_grid_position, Some((4, 8, 12, 4)));
        assert_eq!(mfc.grid_positions(), Some((4, 8, 12, 4)));
        assert!(!mfc.rpu_filter_enabled_flag);
        assert_eq!(mfc.rpu_field_processing_flag, Some(true));
        assert!(mfc.rpu_field_processing());
    }

    /// §G.7.4.2.1.4 — reserved `mfc_format_idc` values (2..=63) carry
    /// no grid syntax at all; `grid_positions()` has no defined value.
    #[test]
    fn mfc_reserved_format_idc_skips_grid() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 134, true);
        w.u(1, 1); // bit_equal_to_one
        write_two_view_mvc_extension(&mut w);
        w.u(6, 2); // mfc_format_idc — reserved
        w.u(1, 1); // rpu_filter_enabled_flag (grid block absent)
        w.u(1, 0); // mvc_vui_parameters_present_flag
        w.u(1, 0); // additional_extension2_flag
        w.trailing();

        let subset = SubsetSps::parse(&w.into_bytes()).expect("reserved-idc subset SPS");
        let SubsetSpsExtension::Mvc { extension, .. } = &subset.extension else {
            panic!("expected Mvc variant");
        };
        let mfc = extension.mfc.as_ref().expect("MFC tail");
        assert_eq!(mfc.mfc_format_idc, 2);
        assert_eq!(mfc.default_grid_position_flag, None);
        assert_eq!(mfc.grid_positions(), None);
        assert!(mfc.rpu_filter_enabled_flag);
    }

    /// §G.7.4.2.1.4 — grid positions "shall be equal to 4, 8 or 12".
    #[test]
    fn mfc_grid_position_value_rejected() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 134, true);
        w.u(1, 1); // bit_equal_to_one
        write_two_view_mvc_extension(&mut w);
        w.u(6, 0); // mfc_format_idc
        w.u(1, 0); // default_grid_position_flag — explicit grid follows
        w.u(4, 5); // view0_grid_position_x — INVALID (not 4/8/12)
        w.u(4, 8);
        w.u(4, 12);
        w.u(4, 4);
        w.u(1, 0);
        w.u(1, 0);
        w.u(1, 0);
        w.trailing();

        let err = SubsetSps::parse(&w.into_bytes()).unwrap_err();
        assert_eq!(
            err,
            SubsetSpsError::GridPositionInvalid {
                field: "view0_grid_position_x",
                value: 5
            }
        );
    }

    /// §7.4.2.1.3 — "bit_equal_to_one shall be equal to 1".
    #[test]
    fn bit_equal_to_one_zero_rejected() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 118, true);
        w.u(1, 0); // bit_equal_to_one — VIOLATION
        write_two_view_mvc_extension(&mut w);
        w.u(1, 0);
        w.u(1, 0);
        w.trailing();

        let err = SubsetSps::parse(&w.into_bytes()).unwrap_err();
        assert_eq!(err, SubsetSpsError::BitEqualToOneViolated);
    }

    /// §G.7.4.2.1.4 — `num_views_minus1 ≤ 1023`, checked before the
    /// per-view allocation (anti-OOM gate).
    #[test]
    fn num_views_out_of_range_rejected() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 118, true);
        w.u(1, 1); // bit_equal_to_one
        w.ue(1024); // num_views_minus1 — out of range
        w.trailing();

        let err = SubsetSps::parse(&w.into_bytes()).unwrap_err();
        assert_eq!(err, SubsetSpsError::NumViewsOutOfRange(1024));
    }

    /// §G.7.4.2.1.4 — `view_id[i] ≤ 1023`.
    #[test]
    fn view_id_out_of_range_rejected() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 118, true);
        w.u(1, 1); // bit_equal_to_one
        w.ue(0); // num_views_minus1
        w.ue(1024); // view_id[0] — out of range
        w.trailing();

        let err = SubsetSps::parse(&w.into_bytes()).unwrap_err();
        assert_eq!(err, SubsetSpsError::ViewIdOutOfRange { i: 0, value: 1024 });
    }

    /// §G.7.4.2.1.4 — `num_anchor_refs_l0[i]` shall not be greater than
    /// `Min(15, num_views_minus1)`. With 2 views the cap is 1.
    #[test]
    fn anchor_ref_count_over_cap_rejected() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 118, true);
        w.u(1, 1); // bit_equal_to_one
        w.ue(1); // num_views_minus1
        w.ue(0); // view_id[0]
        w.ue(1); // view_id[1]
        w.ue(2); // num_anchor_refs_l0[1] — exceeds Min(15, 1) = 1
        w.trailing();

        let err = SubsetSps::parse(&w.into_bytes()).unwrap_err();
        assert_eq!(
            err,
            SubsetSpsError::NumInterViewRefsOutOfRange {
                kind: "anchor",
                list: 0,
                i: 1,
                value: 2,
                max: 1
            }
        );
    }

    /// §G.14.1 — MVC VUI with one operation point carrying timing info
    /// and NAL HRD parameters.
    #[test]
    fn mvc_vui_with_timing_and_hrd() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 118, true);
        w.u(1, 1); // bit_equal_to_one
        write_two_view_mvc_extension(&mut w);
        w.u(1, 1); // mvc_vui_parameters_present_flag
        w.ue(0); // vui_mvc_num_ops_minus1
        w.u(3, 2); // vui_mvc_temporal_id[0]
        w.ue(1); // vui_mvc_num_target_output_views_minus1[0]
        w.ue(0); // vui_mvc_view_id[0][0]
        w.ue(2); // vui_mvc_view_id[0][1]
        w.u(1, 1); // vui_mvc_timing_info_present_flag[0]
        w.u(32, 1000); // vui_mvc_num_units_in_tick[0]
        w.u(32, 60000); // vui_mvc_time_scale[0]
        w.u(1, 1); // vui_mvc_fixed_frame_rate_flag[0]
        w.u(1, 1); // vui_mvc_nal_hrd_parameters_present_flag[0]
                   // §E.1.2 hrd_parameters():
        w.ue(0); // cpb_cnt_minus1
        w.u(4, 1); // bit_rate_scale
        w.u(4, 2); // cpb_size_scale
        w.ue(999); // bit_rate_value_minus1[0]
        w.ue(499); // cpb_size_value_minus1[0]
        w.u(1, 1); // cbr_flag[0]
        w.u(5, 23); // initial_cpb_removal_delay_length_minus1
        w.u(5, 23); // cpb_removal_delay_length_minus1
        w.u(5, 23); // dpb_output_delay_length_minus1
        w.u(5, 24); // time_offset_length
        w.u(1, 0); // vui_mvc_vcl_hrd_parameters_present_flag[0]
        w.u(1, 0); // vui_mvc_low_delay_hrd_flag[0] (NAL HRD present)
        w.u(1, 1); // vui_mvc_pic_struct_present_flag[0]
        w.u(1, 0); // additional_extension2_flag
        w.trailing();

        let subset = SubsetSps::parse(&w.into_bytes()).expect("subset SPS with MVC VUI");
        let SubsetSpsExtension::Mvc { vui, .. } = &subset.extension else {
            panic!("expected Mvc variant");
        };
        let vui = vui.as_ref().expect("MVC VUI present");
        assert_eq!(vui.ops.len(), 1);
        let op = &vui.ops[0];
        assert_eq!(op.temporal_id, 2);
        assert_eq!(op.target_output_view_ids, vec![0, 2]);
        let timing = op.timing_info.as_ref().expect("timing info");
        assert_eq!(timing.num_units_in_tick, 1000);
        assert_eq!(timing.time_scale, 60000);
        assert!(timing.fixed_frame_rate_flag);
        let hrd = op.nal_hrd_parameters.as_ref().expect("NAL HRD");
        assert_eq!(hrd.cpb_cnt_minus1, 0);
        assert_eq!(hrd.bit_rate_scale, 1);
        assert_eq!(hrd.cpb_size_scale, 2);
        assert_eq!(hrd.bit_rate_value_minus1, vec![999]);
        assert_eq!(hrd.cpb_size_value_minus1, vec![499]);
        assert_eq!(hrd.cbr_flag, vec![true]);
        assert_eq!(hrd.time_offset_length, 24);
        assert!(op.vcl_hrd_parameters.is_none());
        assert_eq!(op.low_delay_hrd_flag, Some(false));
        assert!(op.pic_struct_present_flag);
    }

    /// Annex F SVC profiles (83/86): the extension body is not parsed
    /// yet but the embedded `seq_parameter_set_data()` is surfaced.
    #[test]
    fn svc_profile_not_parsed() {
        for profile in [83u8, 86] {
            let mut w = BitWriter::new();
            write_sps_data(&mut w, profile, true);
            // Arbitrary unread extension bytes follow.
            w.u(8, 0xAA);
            w.trailing();

            let subset = SubsetSps::parse(&w.into_bytes()).expect("SVC subset SPS");
            assert_eq!(subset.sps.profile_idc, profile);
            assert_eq!(subset.sps.pic_width_in_mbs(), 4);
            assert_eq!(subset.extension, SubsetSpsExtension::SvcNotParsed);
            assert_eq!(subset.additional_extension2_flag, None);
        }
    }

    /// Annex H MVCD profiles (138/135) and the Annex H + I profile
    /// (139): extension bodies not parsed yet.
    #[test]
    fn mvcd_profiles_not_parsed() {
        for (profile, expected) in [
            (138u8, SubsetSpsExtension::MvcdNotParsed),
            (135, SubsetSpsExtension::MvcdNotParsed),
            (139, SubsetSpsExtension::Mvcd3dNotParsed),
        ] {
            let mut w = BitWriter::new();
            write_sps_data(&mut w, profile, true);
            w.u(8, 0xAA);
            w.trailing();

            let subset = SubsetSps::parse(&w.into_bytes()).expect("MVCD subset SPS");
            assert_eq!(subset.sps.profile_idc, profile);
            assert_eq!(subset.extension, expected);
            assert_eq!(subset.additional_extension2_flag, None);
        }
    }

    /// §7.3.2.1.3 — a profile outside every extension branch reads
    /// `additional_extension2_flag` directly after
    /// `seq_parameter_set_data()`.
    #[test]
    fn non_extension_profile_reads_additional_flag() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 66, true);
        w.u(1, 0); // additional_extension2_flag
        w.trailing();

        let subset = SubsetSps::parse(&w.into_bytes()).expect("plain-profile subset SPS");
        assert_eq!(subset.extension, SubsetSpsExtension::None);
        assert_eq!(subset.additional_extension2_flag, Some(false));
    }

    /// §7.4.2.1.3 — `additional_extension2_flag == 1`: "decoders shall
    /// ignore all data that follow"; the reserved
    /// `additional_extension2_data_flag` run is consumed and discarded.
    #[test]
    fn additional_extension2_flag_one_ignores_tail() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 118, true);
        w.u(1, 1); // bit_equal_to_one
        write_two_view_mvc_extension(&mut w);
        w.u(1, 0); // mvc_vui_parameters_present_flag
        w.u(1, 1); // additional_extension2_flag — reserved tail follows
        w.u(8, 0b1010_1010); // additional_extension2_data_flag run
        w.u(8, 0b0101_0101);
        w.trailing();

        let subset = SubsetSps::parse(&w.into_bytes()).expect("reserved-tail subset SPS");
        assert_eq!(subset.additional_extension2_flag, Some(true));
        assert!(matches!(subset.extension, SubsetSpsExtension::Mvc { .. }));
    }

    /// End-to-end through [`crate::decoder::Decoder`]: a NAL unit of
    /// type 15 produces [`Event::SubsetSpsStored`] and lands in the
    /// subset-SPS table — NOT the ordinary SPS table (§7.4.1.2.1
    /// separate value spaces).
    #[test]
    fn decoder_stores_subset_sps_in_separate_space() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 118, true);
        w.u(1, 1); // bit_equal_to_one
        write_two_view_mvc_extension(&mut w);
        w.u(1, 0); // mvc_vui_parameters_present_flag
        w.u(1, 0); // additional_extension2_flag
        w.trailing();
        let mut nal = vec![0x6F]; // nal_ref_idc = 3, nal_unit_type = 15
        nal.extend_from_slice(&w.into_bytes());

        let mut dec = Decoder::new();
        let ev = dec.process_nal(&nal).expect("subset SPS NAL");
        assert!(matches!(ev, Event::SubsetSpsStored(0)));
        let stored = dec.subset_sps(0).expect("stored subset SPS");
        assert!(matches!(stored.extension, SubsetSpsExtension::Mvc { .. }));
        // Separate id space: no ordinary SPS was stored.
        assert!(dec.sps(0).is_none());
    }
}
