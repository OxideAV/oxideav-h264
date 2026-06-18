//! §7.3.2.1.3 — Subset sequence parameter set RBSP parsing.
//!
//! A subset SPS (NAL unit type 15) wraps an ordinary
//! `seq_parameter_set_data()` body (§7.3.2.1.1, shared with
//! [`crate::sps::Sps`]) followed by a per-profile extension structure:
//!
//! * `profile_idc ∈ {83, 86}` — `seq_parameter_set_svc_extension()`
//!   (§F.7.3.2.1.4) and the optional `svc_vui_parameters_extension()`
//!   (§F.14.1). Fully parsed here, surfaced as
//!   [`SubsetSpsExtension::Svc`].
//! * `profile_idc ∈ {118, 128, 134}` —
//!   `seq_parameter_set_mvc_extension()` (§G.7.3.2.1.4) and the
//!   optional `mvc_vui_parameters_extension()` (§G.14.1). Fully
//!   parsed here.
//! * `profile_idc ∈ {138, 135}` — `seq_parameter_set_mvcd_extension()`
//!   (§H.7.3.2.1.4) and the optional `mvcd_vui_parameters_extension()`
//!   (§H.14.1) + texture `mvc_vui_parameters_extension()` (§G.14.1).
//!   Fully parsed here, surfaced as [`SubsetSpsExtension::Mvcd`].
//! * `profile_idc == 139` — MVCD + `seq_parameter_set_3davc_extension()`
//!   (Annex I). The trailing 3D-AVC body is not parsed yet, so the
//!   leading MVCD half is left unparsed too (the RBSP tail can't be
//!   located); surfaced as [`SubsetSpsExtension::Mvcd3dNotParsed`].
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
    /// §F.7.4.2.1.4 — `extended_spatial_scalability_idc` "shall be in the
    /// range of 0 to 2, inclusive". The u(2) read can yield 3.
    #[error("extended_spatial_scalability_idc out of range (got {0}, max 2)")]
    ExtendedSpatialScalabilityIdcOutOfRange(u32),
    /// §F.7.4.2.1.4 — `chroma_phase_y_plus1` /
    /// `seq_ref_layer_chroma_phase_y_plus1` "shall be in the range of 0
    /// to 2, inclusive". The u(2) read can yield 3.
    #[error("{field} out of range (got {value}, max 2)")]
    ChromaPhaseYOutOfRange {
        /// `"chroma_phase_y_plus1"` or
        /// `"seq_ref_layer_chroma_phase_y_plus1"`.
        field: &'static str,
        value: u32,
    },
    /// §F.14.2 — `vui_ext_num_entries_minus1` in 0..=1023. Checked
    /// before allocating any per-entry storage (anti-OOM gate).
    #[error("vui_ext_num_entries_minus1 out of range (got {0}, max 1023)")]
    VuiExtNumEntriesOutOfRange(u32),
    /// §H.7.4.2.1.4 — `applicable_op_num_texture_views_minus1[i][j]` in
    /// 0..=1023.
    #[error(
        "applicable_op_num_texture_views_minus1[{i}][{j}] out of range (got {value}, max 1023)"
    )]
    OpNumTextureViewsOutOfRange { i: usize, j: usize, value: u32 },
    /// §H.7.4.2.1.4 — `applicable_op_num_depth_views[i][j]` in 0..=1023
    /// (the semantics name it `applicable_op_num_depth_views_minus1`, but
    /// the §H.7.3.2.1.4 syntax reads `applicable_op_num_depth_views`
    /// directly as `ue(v)`).
    #[error("applicable_op_num_depth_views[{i}][{j}] out of range (got {value}, max 1023)")]
    OpNumDepthViewsOutOfRange { i: usize, j: usize, value: u32 },
    /// §H.14.2 — `vui_mvcd_num_ops_minus1` in 0..=1023.
    #[error("vui_mvcd_num_ops_minus1 out of range (got {0}, max 1023)")]
    VuiMvcdNumOpsOutOfRange(u32),
    /// §H.14.2 — `vui_mvcd_num_target_output_views_minus1[i]` in 0..=1023.
    #[error("vui_mvcd_num_target_output_views_minus1[{i}] out of range (got {value}, max 1023)")]
    VuiMvcdNumTargetOutputViewsOutOfRange { i: usize, value: u32 },
    /// §H.14.2 — `vui_mvcd_view_id[i][j]` in 0..=1023.
    #[error("vui_mvcd_view_id[{i}][{j}] out of range (got {value}, max 1023)")]
    VuiMvcdViewIdOutOfRange { i: usize, j: usize, value: u32 },
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

/// §H.7.3.2.1.4 — one entry of the per-view information loop of
/// `seq_parameter_set_mvcd_extension()`. Each VOIdx `i ∈ 0..=num_views_minus1`
/// carries a `view_id` plus two presence flags distinguishing whether a
/// texture view and/or a depth view exists for that VOIdx.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvcdViewInfo {
    /// `view_id[i]` (0..=1023).
    pub view_id: u16,
    /// `depth_view_present_flag[i]` — when set, a depth view exists for
    /// this VOIdx and contributes one entry to `DepthViewId[]`
    /// (§H.7.4.2.1.4).
    pub depth_view_present_flag: bool,
    /// `texture_view_present_flag[i]`. §H.7.4.2.1.4 — when
    /// `depth_view_present_flag[i] == 0`, this shall be 1.
    pub texture_view_present_flag: bool,
}

/// §H.7.3.2.1.4 — one `applicable_op_target_view_id[i][j][k]` plus its
/// per-view depth / texture inclusion flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvcdOpTargetView {
    /// `applicable_op_target_view_id[i][j][k]` (0..=1023).
    pub target_view_id: u16,
    /// `applicable_op_depth_flag[i][j][k]` — whether the depth view with
    /// this view_id is included in the j-th operation point.
    pub depth_flag: bool,
    /// `applicable_op_texture_flag[i][j][k]`. §H.7.4.2.1.4 — when
    /// `applicable_op_depth_flag` is 0, this shall be 1.
    pub texture_flag: bool,
}

/// §H.7.3.2.1.4 — one operation point to which a signalled
/// `level_idc[i]` applies (the MVCD counterpart of [`MvcApplicableOp`],
/// extended with depth/texture view counts).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvcdApplicableOp {
    /// `applicable_op_temporal_id[i][j]` — u(3).
    pub temporal_id: u8,
    /// `applicable_op_target_view_id[i][j][..]` + per-view depth/texture
    /// flags (length `applicable_op_num_target_views_minus1[i][j] + 1`).
    pub target_views: Vec<MvcdOpTargetView>,
    /// `applicable_op_num_texture_views_minus1[i][j]` — plus 1 is the
    /// number of texture views required to decode the target output
    /// views of this operation point.
    pub num_texture_views_minus1: u16,
    /// `applicable_op_num_depth_views[i][j]` — the number of depth views
    /// required to decode the target output views.
    pub num_depth_views: u16,
}

/// §H.7.3.2.1.4 — one signalled level value and the MVCD operation
/// points it applies to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvcdLevelValue {
    /// `level_idc[i]` — u(8).
    pub level_idc: u8,
    /// One entry per `j ∈ 0..=num_applicable_ops_minus1[i]`.
    pub applicable_ops: Vec<MvcdApplicableOp>,
}

/// §H.7.3.2.1.4 — `seq_parameter_set_mvcd_extension()`, the Annex H
/// (MVC + depth) tail of a subset SPS for MVCD profiles
/// (`profile_idc ∈ {138, 135}`; also the leading half of the
/// `profile_idc == 139` 3D-AVC body).
///
/// §H.7.4.2.1.4 inherits the §G.7.4.2.1.4 semantics by reference, so the
/// view-id / count range bounds and the `Min(15, num_views_minus1)`
/// reference-count cap match the MVC extension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpsMvcdExtension {
    /// `view_id[i]` + presence flags, one entry per VOIdx
    /// `i ∈ 0..=num_views_minus1`.
    pub views: Vec<MvcdViewInfo>,
    /// Inter-view dependency lists, one entry per VOIdx (same length as
    /// `views`). Entries are signalled only for `i ∈ 1..=num_views_minus1`
    /// with `depth_view_present_flag[i] == 1`; every other VOIdx keeps
    /// four empty lists (§H.7.3.2.1.4 — the loops are gated on
    /// `depth_view_present_flag[i]`).
    pub view_dependencies: Vec<MvcViewDependency>,
    /// `level_idc[i]` + operation points, one entry per
    /// `i ∈ 0..=num_level_values_signalled_minus1`.
    pub level_values: Vec<MvcdLevelValue>,
}

impl SpsMvcdExtension {
    /// `num_views_minus1 + 1` — the maximum number of coded views (texture
    /// + depth) in the coded video sequence.
    pub fn num_views(&self) -> usize {
        self.views.len()
    }

    /// §H.7.4.2.1.4 — `NumDepthViews`, the count of VOIdx values carrying
    /// a depth view (the number of entries pushed into `DepthViewId[]`).
    pub fn num_depth_views(&self) -> usize {
        self.views
            .iter()
            .filter(|v| v.depth_view_present_flag)
            .count()
    }

    /// §H.7.3.2.1.4 — parse from the current cursor position.
    pub fn parse(r: &mut BitReader<'_>) -> Result<Self, SubsetSpsError> {
        // §G.7.4.2.1.4 (by reference from §H.7.4.2.1.4) —
        // num_views_minus1 in 0..=1023; checked before any allocation.
        let num_views_minus1 = r.ue()?;
        if num_views_minus1 > 1023 {
            return Err(SubsetSpsError::NumViewsOutOfRange(num_views_minus1));
        }
        let num_views = num_views_minus1 as usize + 1;

        // First loop: view_id + depth/texture presence flags per VOIdx.
        let mut views = Vec::with_capacity(num_views);
        for i in 0..num_views {
            let view_id = r.ue()?;
            if view_id > 1023 {
                return Err(SubsetSpsError::ViewIdOutOfRange { i, value: view_id });
            }
            let depth_view_present_flag = r.u(1)? == 1;
            let texture_view_present_flag = r.u(1)? == 1;
            views.push(MvcdViewInfo {
                view_id: view_id as u16,
                depth_view_present_flag,
                texture_view_present_flag,
            });
        }

        // §G.7.4.2.1.4 — the per-list reference count cap
        // Min( 15, num_views_minus1 ).
        let max_refs = num_views_minus1.min(15);
        let mut view_dependencies = vec![MvcViewDependency::default(); num_views];
        // §H.7.3.2.1.4 — the anchor + non-anchor reference loops run for
        // i ∈ 1..=num_views_minus1 but only when depth_view_present_flag[i]
        // is set (unlike §G's MVC body, which signals them for every i).
        for (i, dep) in view_dependencies.iter_mut().enumerate().skip(1) {
            if views[i].depth_view_present_flag {
                dep.anchor_refs_l0 = parse_inter_view_ref_list(r, "anchor", 0, i, max_refs)?;
                dep.anchor_refs_l1 = parse_inter_view_ref_list(r, "anchor", 1, i, max_refs)?;
            }
        }
        for (i, dep) in view_dependencies.iter_mut().enumerate().skip(1) {
            if views[i].depth_view_present_flag {
                dep.non_anchor_refs_l0 =
                    parse_inter_view_ref_list(r, "non_anchor", 0, i, max_refs)?;
                dep.non_anchor_refs_l1 =
                    parse_inter_view_ref_list(r, "non_anchor", 1, i, max_refs)?;
            }
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
                let mut target_views = Vec::with_capacity(num_target_views_minus1 as usize + 1);
                for k in 0..=num_target_views_minus1 as usize {
                    let target_view_id = r.ue()?;
                    if target_view_id > 1023 {
                        return Err(SubsetSpsError::TargetViewIdOutOfRange {
                            i,
                            j,
                            k,
                            value: target_view_id,
                        });
                    }
                    let depth_flag = r.u(1)? == 1;
                    let texture_flag = r.u(1)? == 1;
                    target_views.push(MvcdOpTargetView {
                        target_view_id: target_view_id as u16,
                        depth_flag,
                        texture_flag,
                    });
                }
                // §H.7.4.2.1.4 —
                // applicable_op_num_texture_views_minus1[i][j] in 0..=1023.
                let num_texture_views_minus1 = r.ue()?;
                if num_texture_views_minus1 > 1023 {
                    return Err(SubsetSpsError::OpNumTextureViewsOutOfRange {
                        i,
                        j,
                        value: num_texture_views_minus1,
                    });
                }
                // §H.7.4.2.1.4 — applicable_op_num_depth_views[i][j] in
                // 0..=1023.
                let num_depth_views = r.ue()?;
                if num_depth_views > 1023 {
                    return Err(SubsetSpsError::OpNumDepthViewsOutOfRange {
                        i,
                        j,
                        value: num_depth_views,
                    });
                }
                applicable_ops.push(MvcdApplicableOp {
                    temporal_id,
                    target_views,
                    num_texture_views_minus1: num_texture_views_minus1 as u16,
                    num_depth_views: num_depth_views as u16,
                });
            }
            level_values.push(MvcdLevelValue {
                level_idc,
                applicable_ops,
            });
        }

        Ok(SpsMvcdExtension {
            views,
            view_dependencies,
            level_values,
        })
    }
}

/// §H.14.1 — one operation point inside `mvcd_vui_parameters_extension()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvcdVuiOp {
    /// `vui_mvcd_temporal_id[i]` — u(3).
    pub temporal_id: u8,
    /// `vui_mvcd_view_id[i][..]` + per-view depth/texture flags
    /// (length `vui_mvcd_num_target_output_views_minus1[i] + 1`).
    pub target_output_views: Vec<MvcdVuiTargetView>,
    /// Present when `vui_mvcd_timing_info_present_flag[i] == 1`.
    pub timing_info: Option<MvcVuiTimingInfo>,
    /// Present when `vui_mvcd_nal_hrd_parameters_present_flag[i] == 1`.
    pub nal_hrd_parameters: Option<HrdParameters>,
    /// Present when `vui_mvcd_vcl_hrd_parameters_present_flag[i] == 1`.
    pub vcl_hrd_parameters: Option<HrdParameters>,
    /// `vui_mvcd_low_delay_hrd_flag[i]` — present only when one of the
    /// two HRD blocks is.
    pub low_delay_hrd_flag: Option<bool>,
    /// `vui_mvcd_pic_struct_present_flag[i]` — u(1).
    pub pic_struct_present_flag: bool,
}

/// §H.14.1 — one `vui_mvcd_view_id[i][j]` plus its depth / texture
/// inclusion flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MvcdVuiTargetView {
    /// `vui_mvcd_view_id[i][j]` (0..=1023).
    pub view_id: u16,
    /// `vui_mvcd_depth_flag[i][j]`.
    pub depth_flag: bool,
    /// `vui_mvcd_texture_flag[i][j]`. §H.14.2 — when
    /// `vui_mvcd_depth_flag[i][j]` is 0, this shall be 1.
    pub texture_flag: bool,
}

/// §H.14.1 — `mvcd_vui_parameters_extension()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MvcdVuiParametersExtension {
    /// One entry per `i ∈ 0..=vui_mvcd_num_ops_minus1`.
    pub ops: Vec<MvcdVuiOp>,
}

impl MvcdVuiParametersExtension {
    /// §H.14.1 — parse from the current cursor position.
    pub fn parse(r: &mut BitReader<'_>) -> Result<Self, SubsetSpsError> {
        // §H.14.2 — vui_mvcd_num_ops_minus1 in 0..=1023; checked before
        // allocation.
        let num_ops_minus1 = r.ue()?;
        if num_ops_minus1 > 1023 {
            return Err(SubsetSpsError::VuiMvcdNumOpsOutOfRange(num_ops_minus1));
        }
        let mut ops = Vec::with_capacity(num_ops_minus1 as usize + 1);
        for i in 0..=num_ops_minus1 as usize {
            let temporal_id = r.u(3)? as u8;
            // §H.14.2 — vui_mvcd_num_target_output_views_minus1[i] in
            // 0..=1023.
            let num_target_views_minus1 = r.ue()?;
            if num_target_views_minus1 > 1023 {
                return Err(SubsetSpsError::VuiMvcdNumTargetOutputViewsOutOfRange {
                    i,
                    value: num_target_views_minus1,
                });
            }
            let mut target_output_views = Vec::with_capacity(num_target_views_minus1 as usize + 1);
            for j in 0..=num_target_views_minus1 as usize {
                let view_id = r.ue()?;
                if view_id > 1023 {
                    return Err(SubsetSpsError::VuiMvcdViewIdOutOfRange {
                        i,
                        j,
                        value: view_id,
                    });
                }
                let depth_flag = r.u(1)? == 1;
                let texture_flag = r.u(1)? == 1;
                target_output_views.push(MvcdVuiTargetView {
                    view_id: view_id as u16,
                    depth_flag,
                    texture_flag,
                });
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
            ops.push(MvcdVuiOp {
                temporal_id,
                target_output_views,
                timing_info,
                nal_hrd_parameters,
                vcl_hrd_parameters,
                low_delay_hrd_flag,
                pic_struct_present_flag,
            });
        }
        Ok(MvcdVuiParametersExtension { ops })
    }
}

/// §F.7.3.2.1.4 — geometrical resampling parameters, present only when
/// `extended_spatial_scalability_idc == 1`. Each offset is `se(v)` in
/// the range `−2^15 .. 2^15 − 1`; absent fields infer to 0 per
/// §F.7.4.2.1.4.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SvcSeqRefLayerOffsets {
    /// `seq_ref_layer_chroma_phase_x_plus1_flag` — present only when
    /// `ChromaArrayType > 0`; otherwise inferred equal to
    /// `chroma_phase_x_plus1_flag` per §F.7.4.2.1.4.
    pub ref_layer_chroma_phase_x_plus1_flag: bool,
    /// `seq_ref_layer_chroma_phase_y_plus1` (0..=2) — present only when
    /// `ChromaArrayType > 0`; otherwise inferred equal to
    /// `chroma_phase_y_plus1`.
    pub ref_layer_chroma_phase_y_plus1: u8,
    /// `seq_scaled_ref_layer_left_offset` (units of two luma samples).
    pub scaled_ref_layer_left_offset: i32,
    /// `seq_scaled_ref_layer_top_offset` (units of two/four luma samples
    /// per `frame_mbs_only_flag`).
    pub scaled_ref_layer_top_offset: i32,
    /// `seq_scaled_ref_layer_right_offset` (units of two luma samples).
    pub scaled_ref_layer_right_offset: i32,
    /// `seq_scaled_ref_layer_bottom_offset` (units of two/four luma
    /// samples per `frame_mbs_only_flag`).
    pub scaled_ref_layer_bottom_offset: i32,
}

/// §F.7.3.2.1.4 — `seq_parameter_set_svc_extension()`, the Annex F
/// (scalable extension) tail of a subset SPS for SVC profiles
/// (`profile_idc ∈ {83, 86}`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpsSvcExtension {
    /// `inter_layer_deblocking_filter_control_present_flag` — u(1).
    pub inter_layer_deblocking_filter_control_present_flag: bool,
    /// `extended_spatial_scalability_idc` (0..=2) — u(2).
    pub extended_spatial_scalability_idc: u8,
    /// `chroma_phase_x_plus1_flag`; present only when
    /// `ChromaArrayType ∈ {1, 2}`, else inferred to `true` (value 1) per
    /// §F.7.4.2.1.4.
    pub chroma_phase_x_plus1_flag: bool,
    /// `chroma_phase_y_plus1` (0..=2); present only when
    /// `ChromaArrayType == 1`, else inferred to 1.
    pub chroma_phase_y_plus1: u8,
    /// Geometrical parameters — `Some` only when
    /// `extended_spatial_scalability_idc == 1`.
    pub seq_ref_layer_offsets: Option<SvcSeqRefLayerOffsets>,
    /// `seq_tcoeff_level_prediction_flag` — u(1).
    pub seq_tcoeff_level_prediction_flag: bool,
    /// `adaptive_tcoeff_level_prediction_flag`; present only when
    /// `seq_tcoeff_level_prediction_flag == 1`, else inferred to `false`
    /// (0) per §F.7.4.2.1.4.
    pub adaptive_tcoeff_level_prediction_flag: bool,
    /// `slice_header_restriction_flag` — u(1).
    pub slice_header_restriction_flag: bool,
}

impl SpsSvcExtension {
    /// §F.7.3.2.1.4 — parse from the current cursor. `chroma_array_type`
    /// is `ChromaArrayType` (§6.2) derived from the embedded
    /// `seq_parameter_set_data()`; it gates the chroma-phase and
    /// reference-layer-chroma-phase reads. `frame_mbs_only_flag` is
    /// carried in the returned offsets' implicit units but is not needed
    /// for parsing (the `se(v)` reads are unit-agnostic).
    pub fn parse(r: &mut BitReader<'_>, chroma_array_type: u32) -> Result<Self, SubsetSpsError> {
        let inter_layer_deblocking_filter_control_present_flag = r.u(1)? == 1;
        let extended_spatial_scalability_idc = r.u(2)?;
        if extended_spatial_scalability_idc > 2 {
            return Err(SubsetSpsError::ExtendedSpatialScalabilityIdcOutOfRange(
                extended_spatial_scalability_idc,
            ));
        }
        let extended_spatial_scalability_idc = extended_spatial_scalability_idc as u8;

        // §F.7.4.2.1.4 — chroma_phase_x_plus1_flag inferred to 1 when
        // absent; chroma_phase_y_plus1 inferred to 1 when absent.
        let chroma_phase_x_plus1_flag = if chroma_array_type == 1 || chroma_array_type == 2 {
            r.u(1)? == 1
        } else {
            true
        };
        let chroma_phase_y_plus1 = if chroma_array_type == 1 {
            let v = r.u(2)?;
            if v > 2 {
                return Err(SubsetSpsError::ChromaPhaseYOutOfRange {
                    field: "chroma_phase_y_plus1",
                    value: v,
                });
            }
            v as u8
        } else {
            1
        };

        let seq_ref_layer_offsets = if extended_spatial_scalability_idc == 1 {
            // §F.7.4.2.1.4 — when ChromaArrayType == 0 these chroma-phase
            // fields are absent and inferred from the layer's own
            // chroma_phase_* values.
            let (ref_layer_chroma_phase_x_plus1_flag, ref_layer_chroma_phase_y_plus1) =
                if chroma_array_type > 0 {
                    let x = r.u(1)? == 1;
                    let y = r.u(2)?;
                    if y > 2 {
                        return Err(SubsetSpsError::ChromaPhaseYOutOfRange {
                            field: "seq_ref_layer_chroma_phase_y_plus1",
                            value: y,
                        });
                    }
                    (x, y as u8)
                } else {
                    (chroma_phase_x_plus1_flag, chroma_phase_y_plus1)
                };
            Some(SvcSeqRefLayerOffsets {
                ref_layer_chroma_phase_x_plus1_flag,
                ref_layer_chroma_phase_y_plus1,
                scaled_ref_layer_left_offset: r.se()?,
                scaled_ref_layer_top_offset: r.se()?,
                scaled_ref_layer_right_offset: r.se()?,
                scaled_ref_layer_bottom_offset: r.se()?,
            })
        } else {
            None
        };

        let seq_tcoeff_level_prediction_flag = r.u(1)? == 1;
        // §F.7.4.2.1.4 — adaptive_tcoeff_level_prediction_flag inferred
        // to 0 when seq_tcoeff_level_prediction_flag == 0.
        let adaptive_tcoeff_level_prediction_flag = if seq_tcoeff_level_prediction_flag {
            r.u(1)? == 1
        } else {
            false
        };
        let slice_header_restriction_flag = r.u(1)? == 1;

        Ok(SpsSvcExtension {
            inter_layer_deblocking_filter_control_present_flag,
            extended_spatial_scalability_idc,
            chroma_phase_x_plus1_flag,
            chroma_phase_y_plus1,
            seq_ref_layer_offsets,
            seq_tcoeff_level_prediction_flag,
            adaptive_tcoeff_level_prediction_flag,
            slice_header_restriction_flag,
        })
    }
}

/// §F.14.1 — one information entry inside
/// `svc_vui_parameters_extension()`. Mirrors the §E.2.1 timing /
/// §E.1.2 HRD reuse pattern (cf. [`MvcVuiOp`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvcVuiEntry {
    /// `vui_ext_dependency_id[i]` — u(3).
    pub dependency_id: u8,
    /// `vui_ext_quality_id[i]` — u(4).
    pub quality_id: u8,
    /// `vui_ext_temporal_id[i]` — u(3).
    pub temporal_id: u8,
    /// Present when `vui_ext_timing_info_present_flag[i] == 1`. Reuses
    /// the §E.2.1 timing triple ([`MvcVuiTimingInfo`]).
    pub timing_info: Option<MvcVuiTimingInfo>,
    /// Present when `vui_ext_nal_hrd_parameters_present_flag[i] == 1`.
    pub nal_hrd_parameters: Option<HrdParameters>,
    /// Present when `vui_ext_vcl_hrd_parameters_present_flag[i] == 1`.
    pub vcl_hrd_parameters: Option<HrdParameters>,
    /// `vui_ext_low_delay_hrd_flag[i]` — present only when one of the
    /// two HRD blocks is.
    pub low_delay_hrd_flag: Option<bool>,
    /// `vui_ext_pic_struct_present_flag[i]` — u(1).
    pub pic_struct_present_flag: bool,
}

/// §F.14.1 — `svc_vui_parameters_extension()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvcVuiParametersExtension {
    /// One entry per `i ∈ 0..=vui_ext_num_entries_minus1`.
    pub entries: Vec<SvcVuiEntry>,
}

impl SvcVuiParametersExtension {
    /// §F.14.1 — parse from the current cursor position.
    pub fn parse(r: &mut BitReader<'_>) -> Result<Self, SubsetSpsError> {
        // §F.14.2 — vui_ext_num_entries_minus1 in 0..=1023; checked
        // before allocation.
        let num_entries_minus1 = r.ue()?;
        if num_entries_minus1 > 1023 {
            return Err(SubsetSpsError::VuiExtNumEntriesOutOfRange(
                num_entries_minus1,
            ));
        }
        let mut entries = Vec::with_capacity(num_entries_minus1 as usize + 1);
        for _ in 0..=num_entries_minus1 {
            let dependency_id = r.u(3)? as u8;
            let quality_id = r.u(4)? as u8;
            let temporal_id = r.u(3)? as u8;
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
            entries.push(SvcVuiEntry {
                dependency_id,
                quality_id,
                temporal_id,
                timing_info,
                nal_hrd_parameters,
                vcl_hrd_parameters,
                low_delay_hrd_flag,
                pic_struct_present_flag,
            });
        }
        Ok(SvcVuiParametersExtension { entries })
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
    /// `profile_idc ∈ {83, 86}` — the Annex F SVC extension and its
    /// optional SVC VUI.
    Svc {
        extension: SpsSvcExtension,
        /// `Some(..)` when `svc_vui_parameters_present_flag == 1`.
        vui: Option<SvcVuiParametersExtension>,
    },
    /// `profile_idc ∈ {138, 135}` — the Annex H
    /// `seq_parameter_set_mvcd_extension()` and its optional MVCD VUI.
    Mvcd {
        extension: SpsMvcdExtension,
        /// `Some(..)` when `mvcd_vui_parameters_present_flag == 1`.
        vui: Option<MvcdVuiParametersExtension>,
        /// `Some(..)` when `texture_vui_parameters_present_flag == 1` —
        /// §H.7.3.2.1.4 reuses the §G.14.1 `mvc_vui_parameters_extension()`
        /// for the texture-view VUI tail.
        texture_vui: Option<MvcVuiParametersExtension>,
    },
    /// `profile_idc == 139` — Annex H MVCD + Annex I 3D-AVC. The leading
    /// `seq_parameter_set_mvcd_extension()` half is not parsed because the
    /// trailing `seq_parameter_set_3davc_extension()` body (Annex I) is
    /// not yet implemented, so `additional_extension2_flag` cannot be
    /// located.
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
    /// ([`SubsetSpsExtension::Mvcd3dNotParsed`]) — the flag sits after
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
            // Annex F SVC profiles (83 Scalable Baseline, 86 Scalable
            // High). §7.3.2.1.3 — no bit_equal_to_one precedes the SVC
            // extension (unlike the MVC/MVCD branches).
            83 | 86 => {
                let chroma_array_type = sps.chroma_array_type();
                let extension = SpsSvcExtension::parse(&mut r, chroma_array_type)?;
                let vui = if r.u(1)? == 1 {
                    Some(SvcVuiParametersExtension::parse(&mut r)?)
                } else {
                    None
                };
                let additional_extension2_flag = Self::finish_rbsp(&mut r)?;
                Ok(SubsetSps {
                    sps,
                    extension: SubsetSpsExtension::Svc { extension, vui },
                    additional_extension2_flag: Some(additional_extension2_flag),
                })
            }
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
            // Annex H MVCD profiles (138 Multiview Depth High, 135 MVCD
            // Multiview Depth Stereo).
            138 | 135 => {
                // §7.4.2.1.3 — bit_equal_to_one shall be equal to 1.
                if r.f(1)? != 1 {
                    return Err(SubsetSpsError::BitEqualToOneViolated);
                }
                let extension = SpsMvcdExtension::parse(&mut r)?;
                // §H.7.3.2.1.4 tail — mvcd_vui_parameters_present_flag then
                // texture_vui_parameters_present_flag sit inside the MVCD
                // extension structure.
                let vui = if r.u(1)? == 1 {
                    Some(MvcdVuiParametersExtension::parse(&mut r)?)
                } else {
                    None
                };
                let texture_vui = if r.u(1)? == 1 {
                    Some(MvcVuiParametersExtension::parse(&mut r)?)
                } else {
                    None
                };
                let additional_extension2_flag = Self::finish_rbsp(&mut r)?;
                Ok(SubsetSps {
                    sps,
                    extension: SubsetSpsExtension::Mvcd {
                        extension,
                        vui,
                        texture_vui,
                    },
                    additional_extension2_flag: Some(additional_extension2_flag),
                })
            }
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

        /// §9.1.1 inverse mapping — `v > 0 ⇒ 2v − 1`, `v ≤ 0 ⇒ −2v`.
        fn se(&mut self, value: i32) {
            let code_num = if value > 0 {
                (value as u32) * 2 - 1
            } else {
                (value.unsigned_abs()) * 2
            };
            self.ue(code_num);
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

    /// §H.7.3.2.1.4 — write a minimal two-view MVCD extension body: VOIdx
    /// 0 carries texture only (`depth_view_present_flag[0] == 0`), VOIdx 1
    /// carries both texture and depth (`depth_view_present_flag[1] == 1`)
    /// so its anchor / non-anchor reference loops are signalled. One level
    /// value with one operation point targeting both views.
    fn write_two_view_mvcd_extension(w: &mut BitWriter) {
        w.ue(1); // num_views_minus1
                 // view info loop
        w.ue(0); // view_id[0]
        w.u(1, 0); // depth_view_present_flag[0] = 0
        w.u(1, 1); // texture_view_present_flag[0] = 1
        w.ue(2); // view_id[1]
        w.u(1, 1); // depth_view_present_flag[1] = 1
        w.u(1, 1); // texture_view_present_flag[1] = 1
                   // anchor refs: only i==1 (depth_view_present_flag[1]==1)
        w.ue(1); // num_anchor_refs_l0[1]
        w.ue(0); // anchor_ref_l0[1][0]
        w.ue(0); // num_anchor_refs_l1[1]
                 // non-anchor refs: only i==1
        w.ue(0); // num_non_anchor_refs_l0[1]
        w.ue(1); // num_non_anchor_refs_l1[1]
        w.ue(0); // non_anchor_ref_l1[1][0]
                 // level values
        w.ue(0); // num_level_values_signalled_minus1
        w.u(8, 40); // level_idc[0]
        w.ue(0); // num_applicable_ops_minus1[0]
        w.u(3, 1); // applicable_op_temporal_id[0][0]
        w.ue(1); // applicable_op_num_target_views_minus1[0][0]
        w.ue(0); // applicable_op_target_view_id[0][0][0]
        w.u(1, 0); // applicable_op_depth_flag[0][0][0] = 0
        w.u(1, 1); // applicable_op_texture_flag[0][0][0] = 1
        w.ue(2); // applicable_op_target_view_id[0][0][1]
        w.u(1, 1); // applicable_op_depth_flag[0][0][1] = 1
        w.u(1, 1); // applicable_op_texture_flag[0][0][1] = 1
        w.ue(1); // applicable_op_num_texture_views_minus1[0][0]
        w.ue(1); // applicable_op_num_depth_views[0][0]
    }

    /// §H.7.3.2.1.4 — a two-view Multiview Depth High (profile 138) subset
    /// SPS round-trips through every MVCD field.
    #[test]
    fn mvcd_two_views_full_round_trip() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 138, true);
        w.u(1, 1); // bit_equal_to_one
        write_two_view_mvcd_extension(&mut w);
        w.u(1, 0); // mvcd_vui_parameters_present_flag
        w.u(1, 0); // texture_vui_parameters_present_flag
        w.u(1, 0); // additional_extension2_flag
        w.trailing();

        let subset = SubsetSps::parse(&w.into_bytes()).expect("two-view MVCD subset SPS");
        assert_eq!(subset.sps.profile_idc, 138);
        assert_eq!(subset.additional_extension2_flag, Some(false));
        let SubsetSpsExtension::Mvcd {
            extension,
            vui,
            texture_vui,
        } = &subset.extension
        else {
            panic!(
                "expected the Mvcd extension variant, got {:?}",
                subset.extension
            );
        };
        assert!(vui.is_none());
        assert!(texture_vui.is_none());
        assert_eq!(extension.num_views(), 2);
        // §H.7.4.2.1.4 — NumDepthViews counts VOIdx with a depth view.
        assert_eq!(extension.num_depth_views(), 1);
        assert_eq!(
            extension.views[0],
            MvcdViewInfo {
                view_id: 0,
                depth_view_present_flag: false,
                texture_view_present_flag: true,
            }
        );
        assert_eq!(
            extension.views[1],
            MvcdViewInfo {
                view_id: 2,
                depth_view_present_flag: true,
                texture_view_present_flag: true,
            }
        );
        // VOIdx 0 has no inter-view references; VOIdx 1's depth loops fired.
        assert_eq!(extension.view_dependencies[0], MvcViewDependency::default());
        assert_eq!(extension.view_dependencies[1].anchor_refs_l0, vec![0]);
        assert!(extension.view_dependencies[1].anchor_refs_l1.is_empty());
        assert!(extension.view_dependencies[1].non_anchor_refs_l0.is_empty());
        assert_eq!(extension.view_dependencies[1].non_anchor_refs_l1, vec![0]);
        assert_eq!(extension.level_values.len(), 1);
        let lv = &extension.level_values[0];
        assert_eq!(lv.level_idc, 40);
        assert_eq!(lv.applicable_ops.len(), 1);
        let op = &lv.applicable_ops[0];
        assert_eq!(op.temporal_id, 1);
        assert_eq!(op.target_views.len(), 2);
        assert_eq!(
            op.target_views[0],
            MvcdOpTargetView {
                target_view_id: 0,
                depth_flag: false,
                texture_flag: true,
            }
        );
        assert_eq!(
            op.target_views[1],
            MvcdOpTargetView {
                target_view_id: 2,
                depth_flag: true,
                texture_flag: true,
            }
        );
        assert_eq!(op.num_texture_views_minus1, 1);
        assert_eq!(op.num_depth_views, 1);
    }

    /// §H.7.3.2.1.4 — a VOIdx 1 with `depth_view_present_flag == 0` skips
    /// its anchor / non-anchor reference loops entirely (the gating differs
    /// from the §G MVC body, which signals them unconditionally).
    #[test]
    fn mvcd_no_depth_view_skips_ref_loops() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 135, true);
        w.u(1, 1); // bit_equal_to_one
        w.ue(1); // num_views_minus1
        w.ue(0); // view_id[0]
        w.u(1, 0); // depth_view_present_flag[0] = 0
        w.u(1, 1); // texture_view_present_flag[0] = 1
        w.ue(7); // view_id[1]
        w.u(1, 0); // depth_view_present_flag[1] = 0 ⇒ ref loops skipped
        w.u(1, 1); // texture_view_present_flag[1] = 1
                   // (no anchor / non-anchor reference syntax)
        w.ue(0); // num_level_values_signalled_minus1
        w.u(8, 30); // level_idc[0]
        w.ue(0); // num_applicable_ops_minus1[0]
        w.u(3, 0); // applicable_op_temporal_id[0][0]
        w.ue(0); // applicable_op_num_target_views_minus1[0][0]
        w.ue(7); // applicable_op_target_view_id[0][0][0]
        w.u(1, 0); // applicable_op_depth_flag[0][0][0] = 0
        w.u(1, 1); // applicable_op_texture_flag[0][0][0] = 1
        w.ue(0); // applicable_op_num_texture_views_minus1[0][0]
        w.ue(0); // applicable_op_num_depth_views[0][0]
        w.u(1, 0); // mvcd_vui_parameters_present_flag
        w.u(1, 0); // texture_vui_parameters_present_flag
        w.u(1, 0); // additional_extension2_flag
        w.trailing();

        let subset = SubsetSps::parse(&w.into_bytes()).expect("depthless MVCD subset SPS");
        assert_eq!(subset.sps.profile_idc, 135);
        let SubsetSpsExtension::Mvcd { extension, .. } = &subset.extension else {
            panic!("expected Mvcd variant");
        };
        assert_eq!(extension.num_views(), 2);
        assert_eq!(extension.num_depth_views(), 0);
        // No depth view anywhere ⇒ every dependency entry stays empty.
        assert_eq!(extension.view_dependencies[0], MvcViewDependency::default());
        assert_eq!(extension.view_dependencies[1], MvcViewDependency::default());
    }

    /// §H.14.1 — the `mvcd_vui_parameters_extension()` tail parses, with a
    /// timing-info-present operation point and no HRD blocks.
    #[test]
    fn mvcd_vui_parameters_extension_round_trip() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 138, true);
        w.u(1, 1); // bit_equal_to_one
        write_two_view_mvcd_extension(&mut w);
        w.u(1, 1); // mvcd_vui_parameters_present_flag
                   // mvcd_vui_parameters_extension()
        w.ue(0); // vui_mvcd_num_ops_minus1
        w.u(3, 2); // vui_mvcd_temporal_id[0]
        w.ue(0); // vui_mvcd_num_target_output_views_minus1[0]
        w.ue(2); // vui_mvcd_view_id[0][0]
        w.u(1, 1); // vui_mvcd_depth_flag[0][0]
        w.u(1, 1); // vui_mvcd_texture_flag[0][0]
        w.u(1, 1); // vui_mvcd_timing_info_present_flag[0]
        w.u(32, 1001); // vui_mvcd_num_units_in_tick[0]
        w.u(32, 60000); // vui_mvcd_time_scale[0]
        w.u(1, 1); // vui_mvcd_fixed_frame_rate_flag[0]
        w.u(1, 0); // vui_mvcd_nal_hrd_parameters_present_flag[0]
        w.u(1, 0); // vui_mvcd_vcl_hrd_parameters_present_flag[0]
                   // no low_delay_hrd_flag (both HRD flags 0)
        w.u(1, 1); // vui_mvcd_pic_struct_present_flag[0]
        w.u(1, 0); // texture_vui_parameters_present_flag
        w.u(1, 0); // additional_extension2_flag
        w.trailing();

        let subset = SubsetSps::parse(&w.into_bytes()).expect("MVCD subset SPS with VUI");
        let SubsetSpsExtension::Mvcd { vui, .. } = &subset.extension else {
            panic!("expected Mvcd variant");
        };
        let vui = vui.as_ref().expect("mvcd_vui_parameters present");
        assert_eq!(vui.ops.len(), 1);
        let op = &vui.ops[0];
        assert_eq!(op.temporal_id, 2);
        assert_eq!(op.target_output_views.len(), 1);
        assert_eq!(
            op.target_output_views[0],
            MvcdVuiTargetView {
                view_id: 2,
                depth_flag: true,
                texture_flag: true,
            }
        );
        let timing = op.timing_info.as_ref().expect("timing info present");
        assert_eq!(timing.num_units_in_tick, 1001);
        assert_eq!(timing.time_scale, 60000);
        assert!(timing.fixed_frame_rate_flag);
        assert!(op.nal_hrd_parameters.is_none());
        assert!(op.vcl_hrd_parameters.is_none());
        assert!(op.low_delay_hrd_flag.is_none());
        assert!(op.pic_struct_present_flag);
    }

    /// §H.7.3.2.1.4 — a `texture_vui_parameters_present_flag == 1` tail
    /// reuses the §G.14.1 `mvc_vui_parameters_extension()` parser.
    #[test]
    fn mvcd_texture_vui_uses_mvc_parser() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 138, true);
        w.u(1, 1); // bit_equal_to_one
        write_two_view_mvcd_extension(&mut w);
        w.u(1, 0); // mvcd_vui_parameters_present_flag
        w.u(1, 1); // texture_vui_parameters_present_flag
                   // mvc_vui_parameters_extension() — one op, no timing/HRD
        w.ue(0); // vui_mvc_num_ops_minus1
        w.u(3, 0); // vui_mvc_temporal_id[0]
        w.ue(0); // vui_mvc_num_target_output_views_minus1[0]
        w.ue(0); // vui_mvc_view_id[0][0]
        w.u(1, 0); // vui_mvc_timing_info_present_flag[0]
        w.u(1, 0); // vui_mvc_nal_hrd_parameters_present_flag[0]
        w.u(1, 0); // vui_mvc_vcl_hrd_parameters_present_flag[0]
        w.u(1, 0); // vui_mvc_pic_struct_present_flag[0]
        w.u(1, 0); // additional_extension2_flag
        w.trailing();

        let subset = SubsetSps::parse(&w.into_bytes()).expect("MVCD subset SPS with texture VUI");
        let SubsetSpsExtension::Mvcd {
            vui, texture_vui, ..
        } = &subset.extension
        else {
            panic!("expected Mvcd variant");
        };
        assert!(vui.is_none());
        let texture_vui = texture_vui.as_ref().expect("texture VUI present");
        assert_eq!(texture_vui.ops.len(), 1);
        assert_eq!(texture_vui.ops[0].target_output_view_ids, vec![0]);
    }

    /// §H.7.4.2.1.4 (by reference to §G.7.4.2.1.4) — an out-of-range
    /// `num_views_minus1` is rejected before any allocation.
    #[test]
    fn mvcd_num_views_out_of_range_rejected() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 138, true);
        w.u(1, 1); // bit_equal_to_one
        w.ue(1024); // num_views_minus1 — VIOLATION (max 1023)
        w.trailing();

        let err = SubsetSps::parse(&w.into_bytes()).unwrap_err();
        assert_eq!(err, SubsetSpsError::NumViewsOutOfRange(1024));
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

    /// §F.7.3.2.1.4 — a minimal SVC extension body for both SVC
    /// profiles (83 Scalable Baseline, 86 Scalable High) with
    /// `extended_spatial_scalability_idc == 0` (no geometrical params),
    /// ChromaArrayType == 1 (so both chroma-phase fields are present),
    /// `seq_tcoeff_level_prediction_flag == 0`, no SVC VUI.
    #[test]
    fn svc_profile_basic_round_trip() {
        for profile in [83u8, 86] {
            let mut w = BitWriter::new();
            write_sps_data(&mut w, profile, true);
            // seq_parameter_set_svc_extension():
            w.u(1, 1); // inter_layer_deblocking_filter_control_present_flag
            w.u(2, 0); // extended_spatial_scalability_idc
            w.u(1, 1); // chroma_phase_x_plus1_flag (ChromaArrayType==1)
            w.u(2, 2); // chroma_phase_y_plus1 (ChromaArrayType==1)
            w.u(1, 0); // seq_tcoeff_level_prediction_flag
            w.u(1, 1); // slice_header_restriction_flag
            w.u(1, 0); // svc_vui_parameters_present_flag
            w.u(1, 0); // additional_extension2_flag
            w.trailing();

            let subset = SubsetSps::parse(&w.into_bytes()).expect("SVC subset SPS");
            assert_eq!(subset.sps.profile_idc, profile);
            assert_eq!(subset.sps.pic_width_in_mbs(), 4);
            assert_eq!(subset.additional_extension2_flag, Some(false));
            let SubsetSpsExtension::Svc { extension, vui } = &subset.extension else {
                panic!(
                    "expected the Svc extension variant, got {:?}",
                    subset.extension
                );
            };
            assert!(vui.is_none());
            assert!(extension.inter_layer_deblocking_filter_control_present_flag);
            assert_eq!(extension.extended_spatial_scalability_idc, 0);
            assert!(extension.chroma_phase_x_plus1_flag);
            assert_eq!(extension.chroma_phase_y_plus1, 2);
            assert!(extension.seq_ref_layer_offsets.is_none());
            assert!(!extension.seq_tcoeff_level_prediction_flag);
            assert!(!extension.adaptive_tcoeff_level_prediction_flag);
            assert!(extension.slice_header_restriction_flag);
        }
    }

    /// §F.7.3.2.1.4 — `extended_spatial_scalability_idc == 1` makes the
    /// `seq_ref_layer_*` geometrical parameters present (ChromaArrayType
    /// nonzero ⇒ the reference-layer chroma-phase fields are also
    /// coded), and `seq_tcoeff_level_prediction_flag == 1` makes
    /// `adaptive_tcoeff_level_prediction_flag` present.
    #[test]
    fn svc_extended_spatial_scalability_round_trip() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 86, true);
        w.u(1, 0); // inter_layer_deblocking_filter_control_present_flag
        w.u(2, 1); // extended_spatial_scalability_idc == 1
        w.u(1, 0); // chroma_phase_x_plus1_flag (ChromaArrayType==1)
        w.u(2, 1); // chroma_phase_y_plus1 (ChromaArrayType==1)
                   // ChromaArrayType > 0 ⇒ ref-layer chroma-phase fields present:
        w.u(1, 1); // seq_ref_layer_chroma_phase_x_plus1_flag
        w.u(2, 2); // seq_ref_layer_chroma_phase_y_plus1
        w.se(-3); // seq_scaled_ref_layer_left_offset
        w.se(5); // seq_scaled_ref_layer_top_offset
        w.se(0); // seq_scaled_ref_layer_right_offset
        w.se(-1); // seq_scaled_ref_layer_bottom_offset
        w.u(1, 1); // seq_tcoeff_level_prediction_flag
        w.u(1, 1); // adaptive_tcoeff_level_prediction_flag
        w.u(1, 0); // slice_header_restriction_flag
        w.u(1, 0); // svc_vui_parameters_present_flag
        w.u(1, 0); // additional_extension2_flag
        w.trailing();

        let subset = SubsetSps::parse(&w.into_bytes()).expect("SVC ESS=1 subset SPS");
        let SubsetSpsExtension::Svc { extension, vui } = &subset.extension else {
            panic!("expected Svc, got {:?}", subset.extension);
        };
        assert!(vui.is_none());
        assert_eq!(extension.extended_spatial_scalability_idc, 1);
        let offs = extension
            .seq_ref_layer_offsets
            .as_ref()
            .expect("ESS=1 ⇒ geometrical params present");
        assert!(offs.ref_layer_chroma_phase_x_plus1_flag);
        assert_eq!(offs.ref_layer_chroma_phase_y_plus1, 2);
        assert_eq!(offs.scaled_ref_layer_left_offset, -3);
        assert_eq!(offs.scaled_ref_layer_top_offset, 5);
        assert_eq!(offs.scaled_ref_layer_right_offset, 0);
        assert_eq!(offs.scaled_ref_layer_bottom_offset, -1);
        assert!(extension.seq_tcoeff_level_prediction_flag);
        assert!(extension.adaptive_tcoeff_level_prediction_flag);
    }

    /// §F.7.4.2.1.4 — when `ChromaArrayType == 0` (monochrome) neither
    /// `chroma_phase_x_plus1_flag` nor `chroma_phase_y_plus1` is coded;
    /// they infer to 1. With `extended_spatial_scalability_idc == 1` the
    /// reference-layer chroma-phase fields are likewise absent and infer
    /// from the layer's own chroma_phase_* values.
    #[test]
    fn svc_monochrome_chroma_phase_inferred() {
        let mut w = BitWriter::new();
        // write_sps_data writes chroma_format_idc=1 for the SVC gate; we
        // instead build a monochrome (chroma_format_idc=0) SPS by hand.
        w.u(8, 86); // profile_idc
        w.u(8, 0); // constraint flags + reserved
        w.u(8, 30); // level_idc
        w.ue(0); // seq_parameter_set_id
        w.ue(0); // chroma_format_idc == 0 (monochrome)
        w.ue(0); // bit_depth_luma_minus8
        w.ue(0); // bit_depth_chroma_minus8
        w.u(1, 0); // qpprime_y_zero_transform_bypass_flag
        w.u(1, 0); // seq_scaling_matrix_present_flag
        w.ue(0); // log2_max_frame_num_minus4
        w.ue(0); // pic_order_cnt_type
        w.ue(0); // log2_max_pic_order_cnt_lsb_minus4
        w.ue(1); // max_num_ref_frames
        w.u(1, 0); // gaps_in_frame_num_value_allowed_flag
        w.ue(3); // pic_width_in_mbs_minus1
        w.ue(3); // pic_height_in_map_units_minus1
        w.u(1, 1); // frame_mbs_only_flag
        w.u(1, 1); // direct_8x8_inference_flag
        w.u(1, 0); // frame_cropping_flag
        w.u(1, 0); // vui_parameters_present_flag
                   // seq_parameter_set_svc_extension() — ChromaArrayType == 0:
        w.u(1, 0); // inter_layer_deblocking_filter_control_present_flag
        w.u(2, 1); // extended_spatial_scalability_idc == 1
                   // no chroma_phase_x_plus1_flag / chroma_phase_y_plus1 coded.
                   // ChromaArrayType == 0 ⇒ no ref-layer chroma-phase fields:
        w.se(2); // seq_scaled_ref_layer_left_offset
        w.se(0); // seq_scaled_ref_layer_top_offset
        w.se(0); // seq_scaled_ref_layer_right_offset
        w.se(0); // seq_scaled_ref_layer_bottom_offset
        w.u(1, 0); // seq_tcoeff_level_prediction_flag
        w.u(1, 0); // slice_header_restriction_flag
        w.u(1, 0); // svc_vui_parameters_present_flag
        w.u(1, 0); // additional_extension2_flag
        w.trailing();

        let subset = SubsetSps::parse(&w.into_bytes()).expect("monochrome SVC subset SPS");
        assert_eq!(subset.sps.chroma_array_type(), 0);
        let SubsetSpsExtension::Svc { extension, .. } = &subset.extension else {
            panic!("expected Svc, got {:?}", subset.extension);
        };
        // §F.7.4.2.1.4 inference defaults.
        assert!(extension.chroma_phase_x_plus1_flag);
        assert_eq!(extension.chroma_phase_y_plus1, 1);
        let offs = extension.seq_ref_layer_offsets.as_ref().unwrap();
        // Inferred from the (default) layer chroma_phase_* values.
        assert!(offs.ref_layer_chroma_phase_x_plus1_flag);
        assert_eq!(offs.ref_layer_chroma_phase_y_plus1, 1);
        assert_eq!(offs.scaled_ref_layer_left_offset, 2);
    }

    /// §F.14.1 — an SVC subset SPS carrying a two-entry
    /// `svc_vui_parameters_extension()`, the second entry with NAL HRD
    /// parameters present.
    #[test]
    fn svc_vui_parameters_round_trip() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 83, true);
        // seq_parameter_set_svc_extension() (minimal, ESS=0):
        w.u(1, 0); // inter_layer_deblocking_filter_control_present_flag
        w.u(2, 0); // extended_spatial_scalability_idc
        w.u(1, 1); // chroma_phase_x_plus1_flag
        w.u(2, 1); // chroma_phase_y_plus1
        w.u(1, 0); // seq_tcoeff_level_prediction_flag
        w.u(1, 0); // slice_header_restriction_flag
        w.u(1, 1); // svc_vui_parameters_present_flag
                   // svc_vui_parameters_extension():
        w.ue(1); // vui_ext_num_entries_minus1 == 1 (two entries)
                 // entry 0:
        w.u(3, 0); // vui_ext_dependency_id
        w.u(4, 0); // vui_ext_quality_id
        w.u(3, 0); // vui_ext_temporal_id
        w.u(1, 0); // vui_ext_timing_info_present_flag
        w.u(1, 0); // vui_ext_nal_hrd_parameters_present_flag
        w.u(1, 0); // vui_ext_vcl_hrd_parameters_present_flag
        w.u(1, 1); // vui_ext_pic_struct_present_flag
                   // entry 1 (with NAL HRD parameters):
        w.u(3, 1); // vui_ext_dependency_id
        w.u(4, 2); // vui_ext_quality_id
        w.u(3, 3); // vui_ext_temporal_id
        w.u(1, 0); // vui_ext_timing_info_present_flag
        w.u(1, 1); // vui_ext_nal_hrd_parameters_present_flag
                   // hrd_parameters() — single CPB, simplest legal body:
        w.ue(0); // cpb_cnt_minus1
        w.u(4, 1); // bit_rate_scale
        w.u(4, 1); // cpb_size_scale
        w.ue(999); // bit_rate_value_minus1[0]
        w.ue(499); // cpb_size_value_minus1[0]
        w.u(1, 1); // cbr_flag[0]
        w.u(5, 23); // initial_cpb_removal_delay_length_minus1
        w.u(5, 23); // cpb_removal_delay_length_minus1
        w.u(5, 23); // dpb_output_delay_length_minus1
        w.u(5, 23); // time_offset_length
        w.u(1, 0); // vui_ext_vcl_hrd_parameters_present_flag
        w.u(1, 0); // vui_ext_low_delay_hrd_flag
        w.u(1, 0); // vui_ext_pic_struct_present_flag
        w.u(1, 0); // additional_extension2_flag
        w.trailing();

        let subset = SubsetSps::parse(&w.into_bytes()).expect("SVC subset SPS with VUI");
        let SubsetSpsExtension::Svc { vui, .. } = &subset.extension else {
            panic!("expected Svc, got {:?}", subset.extension);
        };
        let vui = vui.as_ref().expect("svc_vui_parameters_present_flag == 1");
        assert_eq!(vui.entries.len(), 2);
        assert_eq!(vui.entries[0].dependency_id, 0);
        assert!(vui.entries[0].pic_struct_present_flag);
        assert!(vui.entries[0].nal_hrd_parameters.is_none());
        let e1 = &vui.entries[1];
        assert_eq!(e1.dependency_id, 1);
        assert_eq!(e1.quality_id, 2);
        assert_eq!(e1.temporal_id, 3);
        let hrd = e1
            .nal_hrd_parameters
            .as_ref()
            .expect("NAL HRD present on entry 1");
        assert_eq!(hrd.bit_rate_value_minus1, vec![999]);
        assert_eq!(e1.low_delay_hrd_flag, Some(false));
        assert!(!e1.pic_struct_present_flag);
    }

    /// §F.7.4.2.1.4 — `extended_spatial_scalability_idc` "shall be in
    /// the range of 0 to 2"; a u(2) value of 3 is rejected.
    #[test]
    fn svc_ess_idc_3_rejected() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 86, true);
        w.u(1, 0); // inter_layer_deblocking_filter_control_present_flag
        w.u(2, 3); // extended_spatial_scalability_idc == 3 (illegal)
        w.trailing();

        let err = SubsetSps::parse(&w.into_bytes()).expect_err("ESS idc 3 must be rejected");
        assert_eq!(
            err,
            SubsetSpsError::ExtendedSpatialScalabilityIdcOutOfRange(3)
        );
    }

    /// §7.3.2.1.3 — the Annex H MVCD + Annex I 3D-AVC profile (139): the
    /// trailing `seq_parameter_set_3davc_extension()` body is not parsed,
    /// so the leading MVCD half is left unparsed too (the RBSP tail can't
    /// be located). Profiles 138/135 are now fully parsed and covered by
    /// `mvcd_two_views_full_round_trip` et al.
    #[test]
    fn mvcd_3davc_profile_not_parsed() {
        let mut w = BitWriter::new();
        write_sps_data(&mut w, 139, true);
        w.u(8, 0xAA);
        w.trailing();

        let subset = SubsetSps::parse(&w.into_bytes()).expect("3D-AVC subset SPS");
        assert_eq!(subset.sps.profile_idc, 139);
        assert_eq!(subset.extension, SubsetSpsExtension::Mvcd3dNotParsed);
        assert_eq!(subset.additional_extension2_flag, None);
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
