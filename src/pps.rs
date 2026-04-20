//! §7.3.2.2 — Picture Parameter Set parsing.
//!
//! Spec-driven implementation of `pic_parameter_set_rbsp()` per ITU-T
//! Rec. H.264 (08/2024). Flexible Macroblock Ordering (FMO) fields are
//! parsed in full — the `slice_group_map_type` branching is bounded.
//! The `scaling_list()` body (§7.3.2.1.1.1) is also parsed in full; it
//! delegates to [`crate::scaling_list::parse_scaling_list`].
//!
//! The parser takes an already-de-emulated RBSP (`emulation_prevention_three_byte`
//! stripped by [`crate::nal::parse_nal_unit`]). The caller peels off the
//! NAL header byte before handing us the slice.
//!
//! Because the PPS scaling-matrix loop length depends on
//! `chroma_format_idc` from the active SPS (4:4:4 carries 6 8x8
//! lists; everything else carries 2), [`Pps::parse`] takes
//! `chroma_format_idc` as an explicit parameter. For streams with
//! `pic_scaling_matrix_present_flag == 0` (the common case) the value
//! is ignored.

use crate::bitstream::{BitError, BitReader};
use crate::scaling_list::{parse_scaling_list, ScalingListError};
use crate::sps::ScalingListEntry;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PpsError {
    #[error("bitstream read failed: {0}")]
    Bitstream(#[from] BitError),
    /// §7.4.2.2 — `pic_parameter_set_id` in range 0..=255.
    #[error("pic_parameter_set_id out of range (got {0}, max 255)")]
    PpsIdOutOfRange(u32),
    /// §7.4.2.2 — `seq_parameter_set_id` in range 0..=31.
    #[error("seq_parameter_set_id out of range (got {0}, max 31)")]
    SpsIdOutOfRange(u32),
    /// §7.4.2.2 — `num_slice_groups_minus1` bounded by Annex A level
    /// limits. We accept up to 7 here (max across all levels is 7).
    #[error("num_slice_groups_minus1 out of range (got {0}, max 7)")]
    NumSliceGroupsOutOfRange(u32),
    /// §7.4.2.2 — `slice_group_map_type` in range 0..=6.
    #[error("slice_group_map_type out of range (got {0}, max 6)")]
    SliceGroupMapTypeOutOfRange(u32),
    /// §7.4.2.2 — `weighted_bipred_idc` in range 0..=2.
    #[error("weighted_bipred_idc out of range (got {0}, max 2)")]
    WeightedBipredIdcOutOfRange(u32),
    /// §7.4.2.2 — `chroma_qp_index_offset` in range -12..=12.
    #[error("chroma_qp_index_offset out of range (got {0}, must be -12..=12)")]
    ChromaQpIndexOffsetOutOfRange(i32),
    /// §7.4.2.2 — `second_chroma_qp_index_offset` in range -12..=12.
    #[error("second_chroma_qp_index_offset out of range (got {0}, must be -12..=12)")]
    SecondChromaQpIndexOffsetOutOfRange(i32),
    /// §7.3.2.1.1.1 — scaling_list body parse error.
    #[error("scaling_list: {0}")]
    ScalingList(#[from] ScalingListError),
}

/// §7.3.2.2 — PPS scaling matrices.
///
/// First 6 entries are 4x4 lists. When `transform_8x8_mode_flag == 1`
/// the tail carries 8x8 lists — 2 entries when `chroma_format_idc != 3`
/// (total length 8), 6 entries when `chroma_format_idc == 3` (total 12).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PicScalingLists {
    pub entries: Vec<ScalingListEntry>,
}

/// §7.4.2.2 — FMO (Flexible Macroblock Ordering) map description.
/// The shape depends on `slice_group_map_type`:
///
/// * 0: interleaved, one `run_length_minus1` per slice group,
/// * 1: dispersed, no extra fields,
/// * 2: foreground/leftover, top-left/bottom-right per foreground group,
/// * 3/4/5: changing, direction flag + change-rate,
/// * 6: explicit per-map-unit assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SliceGroupMap {
    /// Type 0 — `run_length_minus1[i]` for `i ∈ 0..=num_slice_groups_minus1`.
    Interleaved { run_length_minus1: Vec<u32> },
    /// Type 1 — dispersed. The mapping is derived arithmetically.
    Dispersed,
    /// Type 2 — foreground rectangles with a leftover group.
    Foreground {
        top_left: Vec<u32>,
        bottom_right: Vec<u32>,
    },
    /// Types 3/4/5 — changing slice groups.
    Changing {
        slice_group_map_type: u32, // 3, 4, or 5
        change_direction_flag: bool,
        change_rate_minus1: u32,
    },
    /// Type 6 — explicit `slice_group_id[i]` for `i ∈ 0..=pic_size_in_map_units_minus1`.
    Explicit {
        pic_size_in_map_units_minus1: u32,
        slice_group_id: Vec<u32>,
    },
}

/// §7.4.2.2 — extra fields at the tail of a PPS when `more_rbsp_data()`
/// indicates they are present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PpsExtension {
    pub transform_8x8_mode_flag: bool,
    pub pic_scaling_matrix_present_flag: bool,
    /// Parsed scaling lists when `pic_scaling_matrix_present_flag == 1`.
    pub pic_scaling_lists: Option<PicScalingLists>,
    pub second_chroma_qp_index_offset: i32,
}

/// §7.3.2.2 / §7.4.2.2 — parsed `pic_parameter_set_rbsp()`.
///
/// When `num_slice_groups_minus1 == 0`, `slice_group_map` is `None`.
/// When the optional trailing fields are absent (short PPS), `extension`
/// is `None` — callers should apply the §7.4.2.2 "not present" inferred
/// values (`transform_8x8_mode_flag = 0`, `pic_scaling_matrix_present_flag = 0`,
/// `second_chroma_qp_index_offset = chroma_qp_index_offset`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pps {
    pub pic_parameter_set_id: u32,
    pub seq_parameter_set_id: u32,
    pub entropy_coding_mode_flag: bool,
    pub bottom_field_pic_order_in_frame_present_flag: bool,
    pub num_slice_groups_minus1: u32,
    pub slice_group_map: Option<SliceGroupMap>,
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
    pub extension: Option<PpsExtension>,
}

impl Pps {
    /// §7.3.2.2 — parse a `pic_parameter_set_rbsp()` from a slice that
    /// holds the RBSP body (the bytes *after* the NAL header byte,
    /// emulation-prevention stripped — typically from
    /// [`crate::nal::parse_nal_unit`]).
    /// Parse with a default `chroma_format_idc == 1` (4:2:0). Equivalent
    /// to [`Pps::parse_with_chroma_format(rbsp, 1)`]. Safe for the common
    /// case; for 4:4:4 streams where `pic_scaling_matrix_present_flag`
    /// might be set, prefer [`Pps::parse_with_chroma_format`].
    pub fn parse(rbsp: &[u8]) -> Result<Self, PpsError> {
        Self::parse_with_chroma_format(rbsp, 1)
    }

    /// §7.3.2.2 — parse a `pic_parameter_set_rbsp()` with the
    /// `chroma_format_idc` of the active SPS. The value is only
    /// consulted when `pic_scaling_matrix_present_flag == 1` and
    /// `transform_8x8_mode_flag == 1`: it determines whether the
    /// 8x8 tail carries 2 lists (4:2:0 / 4:2:2) or 6 lists (4:4:4).
    pub fn parse_with_chroma_format(
        rbsp: &[u8],
        chroma_format_idc: u32,
    ) -> Result<Self, PpsError> {
        let mut r = BitReader::new(rbsp);

        // §7.4.2.2 — pic_parameter_set_id ue(v), 0..=255.
        let pic_parameter_set_id = r.ue()?;
        if pic_parameter_set_id > 255 {
            return Err(PpsError::PpsIdOutOfRange(pic_parameter_set_id));
        }

        // §7.4.2.2 — seq_parameter_set_id ue(v), 0..=31.
        let seq_parameter_set_id = r.ue()?;
        if seq_parameter_set_id > 31 {
            return Err(PpsError::SpsIdOutOfRange(seq_parameter_set_id));
        }

        let entropy_coding_mode_flag = r.u(1)? == 1;
        let bottom_field_pic_order_in_frame_present_flag = r.u(1)? == 1;

        // §7.4.2.2 — num_slice_groups_minus1. Annex A caps this at 7.
        let num_slice_groups_minus1 = r.ue()?;
        if num_slice_groups_minus1 > 7 {
            return Err(PpsError::NumSliceGroupsOutOfRange(num_slice_groups_minus1));
        }

        // §7.3.2.2 — FMO description, when more than one slice group.
        let slice_group_map = if num_slice_groups_minus1 > 0 {
            let slice_group_map_type = r.ue()?;
            if slice_group_map_type > 6 {
                return Err(PpsError::SliceGroupMapTypeOutOfRange(slice_group_map_type));
            }
            let map = match slice_group_map_type {
                0 => {
                    // §7.3.2.2: run_length_minus1[i] for i in 0..=num_slice_groups_minus1.
                    let mut run_length_minus1 =
                        Vec::with_capacity(num_slice_groups_minus1 as usize + 1);
                    for _ in 0..=num_slice_groups_minus1 {
                        run_length_minus1.push(r.ue()?);
                    }
                    SliceGroupMap::Interleaved { run_length_minus1 }
                }
                1 => SliceGroupMap::Dispersed,
                2 => {
                    // §7.3.2.2: top_left / bottom_right for i in 0..num_slice_groups_minus1.
                    // That's `num_slice_groups_minus1` iterations (i.e. the
                    // "background" group is implicit), per the loop's `<`.
                    let mut top_left = Vec::with_capacity(num_slice_groups_minus1 as usize);
                    let mut bottom_right = Vec::with_capacity(num_slice_groups_minus1 as usize);
                    for _ in 0..num_slice_groups_minus1 {
                        top_left.push(r.ue()?);
                        bottom_right.push(r.ue()?);
                    }
                    SliceGroupMap::Foreground {
                        top_left,
                        bottom_right,
                    }
                }
                3 | 4 | 5 => {
                    let change_direction_flag = r.u(1)? == 1;
                    let change_rate_minus1 = r.ue()?;
                    SliceGroupMap::Changing {
                        slice_group_map_type,
                        change_direction_flag,
                        change_rate_minus1,
                    }
                }
                6 => {
                    // §7.3.2.2: pic_size_in_map_units_minus1, then
                    // slice_group_id[i] for i in 0..=pic_size_in_map_units_minus1.
                    // §7.4.2.2: each slice_group_id[i] is a u(v) with
                    // v = Ceil(Log2(num_slice_groups_minus1 + 1)) bits.
                    let pic_size_in_map_units_minus1 = r.ue()?;
                    let v = ceil_log2(num_slice_groups_minus1 + 1);
                    let count = pic_size_in_map_units_minus1 as usize + 1;
                    let mut slice_group_id = Vec::with_capacity(count);
                    for _ in 0..count {
                        // v==0 means "at most one slice group", which is
                        // only possible for num_slice_groups_minus1==0;
                        // but that branch is excluded by the outer `if`.
                        // Defensive: if v somehow is 0 we read nothing.
                        let id = if v == 0 { 0 } else { r.u(v)? };
                        slice_group_id.push(id);
                    }
                    SliceGroupMap::Explicit {
                        pic_size_in_map_units_minus1,
                        slice_group_id,
                    }
                }
                _ => unreachable!("slice_group_map_type bound-checked above"),
            };
            Some(map)
        } else {
            None
        };

        // §7.3.2.2 — remaining "always present" fields.
        let num_ref_idx_l0_default_active_minus1 = r.ue()?;
        let num_ref_idx_l1_default_active_minus1 = r.ue()?;
        let weighted_pred_flag = r.u(1)? == 1;
        let weighted_bipred_idc = r.u(2)?;
        if weighted_bipred_idc > 2 {
            return Err(PpsError::WeightedBipredIdcOutOfRange(weighted_bipred_idc));
        }
        let pic_init_qp_minus26 = r.se()?;
        let pic_init_qs_minus26 = r.se()?;
        let chroma_qp_index_offset = r.se()?;
        if !(-12..=12).contains(&chroma_qp_index_offset) {
            return Err(PpsError::ChromaQpIndexOffsetOutOfRange(chroma_qp_index_offset));
        }
        let deblocking_filter_control_present_flag = r.u(1)? == 1;
        let constrained_intra_pred_flag = r.u(1)? == 1;
        let redundant_pic_cnt_present_flag = r.u(1)? == 1;

        // §7.3.2.2 — optional trailing group gated on more_rbsp_data().
        let extension = if r.more_rbsp_data() {
            let transform_8x8_mode_flag = r.u(1)? == 1;
            let pic_scaling_matrix_present_flag = r.u(1)? == 1;
            let pic_scaling_lists = if pic_scaling_matrix_present_flag {
                // §7.3.2.2 loop bound:
                //   6 + ((chroma_format_idc != 3) ? 2 : 6) * transform_8x8_mode_flag
                let extra_8x8 = if !transform_8x8_mode_flag {
                    0
                } else if chroma_format_idc != 3 {
                    2
                } else {
                    6
                };
                let n = 6 + extra_8x8;
                let mut entries = Vec::with_capacity(n);
                for i in 0..n {
                    let present = r.u(1)? == 1;
                    if !present {
                        entries.push(ScalingListEntry::NotPresent);
                    } else {
                        let size = if i < 6 { 16 } else { 64 };
                        let result = parse_scaling_list(&mut r, size)?;
                        entries.push(if result.use_default {
                            ScalingListEntry::UseDefault
                        } else {
                            ScalingListEntry::Explicit(result.scaling_list)
                        });
                    }
                }
                Some(PicScalingLists { entries })
            } else {
                None
            };
            let second_chroma_qp_index_offset = r.se()?;
            if !(-12..=12).contains(&second_chroma_qp_index_offset) {
                return Err(PpsError::SecondChromaQpIndexOffsetOutOfRange(
                    second_chroma_qp_index_offset,
                ));
            }
            // §7.3.2.2 closes with rbsp_trailing_bits(); be lenient:
            // extra padding bytes are common, so ignore errors.
            let _ = r.rbsp_trailing_bits();
            Some(PpsExtension {
                transform_8x8_mode_flag,
                pic_scaling_matrix_present_flag,
                pic_scaling_lists,
                second_chroma_qp_index_offset,
            })
        } else {
            // §7.4.2.2 — when the optional tail is absent,
            // `second_chroma_qp_index_offset` is inferred to equal
            // `chroma_qp_index_offset` and the two flags are 0. We
            // represent that by `extension: None` and let callers apply
            // the inferred defaults.
            None
        };

        Ok(Pps {
            pic_parameter_set_id,
            seq_parameter_set_id,
            entropy_coding_mode_flag,
            bottom_field_pic_order_in_frame_present_flag,
            num_slice_groups_minus1,
            slice_group_map,
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
            extension,
        })
    }

    /// §7.4.2.2 — `second_chroma_qp_index_offset` defaults to
    /// `chroma_qp_index_offset` when the optional tail is absent.
    pub fn second_chroma_qp_index_offset(&self) -> i32 {
        self.extension
            .as_ref()
            .map(|e| e.second_chroma_qp_index_offset)
            .unwrap_or(self.chroma_qp_index_offset)
    }

    /// §7.4.2.2 — `transform_8x8_mode_flag` defaults to 0 when absent.
    pub fn transform_8x8_mode_flag(&self) -> bool {
        self.extension
            .as_ref()
            .is_some_and(|e| e.transform_8x8_mode_flag)
    }
}

/// `Ceil(Log2(n))` for `n >= 1`. Used for the `u(v)` width of
/// `slice_group_id[i]` per §7.4.2.2.
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        // floor(log2(n-1)) + 1
        32 - (n - 1).leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn ceil_log2_matches_spec() {
        // §7.4.2.2 slice_group_id[i] bit count v = Ceil(Log2(N+1)).
        // For N+1 = 1 → 0 bits, 2 → 1, 3 → 2, 4 → 2, 5 → 3, 8 → 3, 9 → 4.
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(8), 3);
        assert_eq!(ceil_log2(9), 4);
    }

    #[test]
    fn minimal_cavlc_pps() {
        // entropy_coding_mode_flag=0, num_slice_groups_minus1=0,
        // no optional tail.
        let mut w = BitWriter::new();
        w.ue(0); // pic_parameter_set_id
        w.ue(0); // seq_parameter_set_id
        w.u(1, 0); // entropy_coding_mode_flag = CAVLC
        w.u(1, 0); // bottom_field_pic_order_in_frame_present_flag
        w.ue(0); // num_slice_groups_minus1
        w.ue(0); // num_ref_idx_l0_default_active_minus1
        w.ue(0); // num_ref_idx_l1_default_active_minus1
        w.u(1, 0); // weighted_pred_flag
        w.u(2, 0); // weighted_bipred_idc
        w.se(0); // pic_init_qp_minus26
        w.se(0); // pic_init_qs_minus26
        w.se(0); // chroma_qp_index_offset
        w.u(1, 1); // deblocking_filter_control_present_flag
        w.u(1, 0); // constrained_intra_pred_flag
        w.u(1, 0); // redundant_pic_cnt_present_flag
        w.trailing();
        let bytes = w.into_bytes();

        let pps = Pps::parse(&bytes).unwrap();
        assert_eq!(pps.pic_parameter_set_id, 0);
        assert_eq!(pps.seq_parameter_set_id, 0);
        assert!(!pps.entropy_coding_mode_flag);
        assert!(!pps.bottom_field_pic_order_in_frame_present_flag);
        assert_eq!(pps.num_slice_groups_minus1, 0);
        assert!(pps.slice_group_map.is_none());
        assert_eq!(pps.num_ref_idx_l0_default_active_minus1, 0);
        assert_eq!(pps.num_ref_idx_l1_default_active_minus1, 0);
        assert!(!pps.weighted_pred_flag);
        assert_eq!(pps.weighted_bipred_idc, 0);
        assert_eq!(pps.pic_init_qp_minus26, 0);
        assert_eq!(pps.pic_init_qs_minus26, 0);
        assert_eq!(pps.chroma_qp_index_offset, 0);
        assert!(pps.deblocking_filter_control_present_flag);
        assert!(!pps.constrained_intra_pred_flag);
        assert!(!pps.redundant_pic_cnt_present_flag);
        assert!(pps.extension.is_none());
        // §7.4.2.2 inferred defaults:
        assert_eq!(pps.second_chroma_qp_index_offset(), 0);
        assert!(!pps.transform_8x8_mode_flag());
    }

    #[test]
    fn cabac_pps_with_trailing_group() {
        // entropy_coding_mode_flag=1 + tail with transform_8x8=1,
        // pic_scaling_matrix=0, second_chroma_qp_index_offset=-3.
        let mut w = BitWriter::new();
        w.ue(1); // pic_parameter_set_id
        w.ue(2); // seq_parameter_set_id
        w.u(1, 1); // entropy_coding_mode_flag = CABAC
        w.u(1, 1); // bottom_field_pic_order_in_frame_present_flag
        w.ue(0); // num_slice_groups_minus1
        w.ue(2); // num_ref_idx_l0_default_active_minus1
        w.ue(0); // num_ref_idx_l1_default_active_minus1
        w.u(1, 1); // weighted_pred_flag
        w.u(2, 2); // weighted_bipred_idc = 2 (implicit)
        w.se(-3); // pic_init_qp_minus26
        w.se(0); // pic_init_qs_minus26
        w.se(-2); // chroma_qp_index_offset
        w.u(1, 1); // deblocking_filter_control_present_flag
        w.u(1, 0); // constrained_intra_pred_flag
        w.u(1, 1); // redundant_pic_cnt_present_flag
        // more_rbsp_data() is true → tail follows.
        w.u(1, 1); // transform_8x8_mode_flag
        w.u(1, 0); // pic_scaling_matrix_present_flag
        w.se(-3); // second_chroma_qp_index_offset
        w.trailing();
        let bytes = w.into_bytes();

        let pps = Pps::parse(&bytes).unwrap();
        assert_eq!(pps.pic_parameter_set_id, 1);
        assert_eq!(pps.seq_parameter_set_id, 2);
        assert!(pps.entropy_coding_mode_flag);
        assert!(pps.bottom_field_pic_order_in_frame_present_flag);
        assert_eq!(pps.num_ref_idx_l0_default_active_minus1, 2);
        assert!(pps.weighted_pred_flag);
        assert_eq!(pps.weighted_bipred_idc, 2);
        assert_eq!(pps.pic_init_qp_minus26, -3);
        assert_eq!(pps.chroma_qp_index_offset, -2);
        assert!(pps.redundant_pic_cnt_present_flag);
        let ext = pps.extension.clone().expect("tail present");
        assert!(ext.transform_8x8_mode_flag);
        assert!(!ext.pic_scaling_matrix_present_flag);
        assert_eq!(ext.second_chroma_qp_index_offset, -3);
        assert_eq!(pps.second_chroma_qp_index_offset(), -3);
        assert!(pps.transform_8x8_mode_flag());
    }

    #[test]
    fn pps_with_scaling_matrix_all_not_present() {
        // transform_8x8_mode_flag=1 + pic_scaling_matrix_present_flag=1,
        // all 8 per-list flags 0 (NotPresent). chroma_format_idc=1.
        let mut w = BitWriter::new();
        w.ue(0); // pps_id
        w.ue(0); // sps_id
        w.u(1, 0); // entropy_coding
        w.u(1, 0); // bottom_field_pic_order
        w.ue(0); // num_slice_groups_minus1
        w.ue(0); // num_ref_idx_l0_default_active_minus1
        w.ue(0); // num_ref_idx_l1_default_active_minus1
        w.u(1, 0); // weighted_pred
        w.u(2, 0); // weighted_bipred_idc
        w.se(0); // pic_init_qp_minus26
        w.se(0); // pic_init_qs_minus26
        w.se(0); // chroma_qp_index_offset
        w.u(1, 0); // deblocking_filter_control_present
        w.u(1, 0); // constrained_intra_pred
        w.u(1, 0); // redundant_pic_cnt_present
        // Tail:
        w.u(1, 1); // transform_8x8_mode_flag
        w.u(1, 1); // pic_scaling_matrix_present_flag
        // Loop: 6 + 2 = 8 entries, all flags 0.
        for _ in 0..8 {
            w.u(1, 0);
        }
        w.se(0); // second_chroma_qp_index_offset
        w.trailing();
        let bytes = w.into_bytes();

        let pps = Pps::parse(&bytes).unwrap();
        let ext = pps.extension.as_ref().expect("tail present");
        assert!(ext.transform_8x8_mode_flag);
        assert!(ext.pic_scaling_matrix_present_flag);
        let lists = ext.pic_scaling_lists.as_ref().unwrap();
        assert_eq!(lists.entries.len(), 8);
        for e in &lists.entries {
            assert_eq!(*e, ScalingListEntry::NotPresent);
        }
    }

    #[test]
    fn pps_with_scaling_matrix_4x4x4_loop_count() {
        // chroma_format_idc=3 (4:4:4) + transform_8x8=1 → loop count is
        // 6 + 6 = 12 entries. Verify Pps::parse_with_chroma_format picks
        // up the extra 4 entries compared to the default parser.
        let mut w = BitWriter::new();
        w.ue(0);
        w.ue(0);
        w.u(1, 0);
        w.u(1, 0);
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.u(1, 0);
        w.u(2, 0);
        w.se(0);
        w.se(0);
        w.se(0);
        w.u(1, 0);
        w.u(1, 0);
        w.u(1, 0);
        w.u(1, 1); // transform_8x8_mode_flag
        w.u(1, 1); // pic_scaling_matrix_present_flag
        for _ in 0..12 {
            w.u(1, 0);
        }
        w.se(0);
        w.trailing();
        let bytes = w.into_bytes();

        let pps = Pps::parse_with_chroma_format(&bytes, 3).unwrap();
        let ext = pps.extension.as_ref().unwrap();
        let lists = ext.pic_scaling_lists.as_ref().unwrap();
        assert_eq!(lists.entries.len(), 12);
    }

    #[test]
    fn pps_with_fmo_type_0_interleaved() {
        // num_slice_groups_minus1=2 (3 groups), map type 0 with three run_lengths.
        let mut w = BitWriter::new();
        w.ue(0);
        w.ue(0);
        w.u(1, 0);
        w.u(1, 0);
        w.ue(2); // num_slice_groups_minus1 = 2
        w.ue(0); // slice_group_map_type = 0 (interleaved)
        w.ue(10); // run_length_minus1[0]
        w.ue(20); // run_length_minus1[1]
        w.ue(30); // run_length_minus1[2]
        w.ue(0);
        w.ue(0);
        w.u(1, 0);
        w.u(2, 0);
        w.se(0);
        w.se(0);
        w.se(0);
        w.u(1, 0);
        w.u(1, 0);
        w.u(1, 0);
        w.trailing();
        let bytes = w.into_bytes();

        let pps = Pps::parse(&bytes).unwrap();
        assert_eq!(pps.num_slice_groups_minus1, 2);
        match pps.slice_group_map {
            Some(SliceGroupMap::Interleaved { run_length_minus1 }) => {
                assert_eq!(run_length_minus1, vec![10, 20, 30]);
            }
            other => panic!("unexpected slice_group_map: {:?}", other),
        }
    }

    #[test]
    fn pps_with_fmo_type_6_explicit() {
        // num_slice_groups_minus1 = 2 → three groups → slice_group_id is u(2).
        // pic_size_in_map_units_minus1 = 3 → four map units: [0, 1, 2, 1].
        let mut w = BitWriter::new();
        w.ue(0);
        w.ue(0);
        w.u(1, 0);
        w.u(1, 0);
        w.ue(2); // num_slice_groups_minus1 = 2
        w.ue(6); // slice_group_map_type = 6 (explicit)
        w.ue(3); // pic_size_in_map_units_minus1 = 3
        // ceil_log2(3) = 2 bits per slice_group_id.
        w.u(2, 0);
        w.u(2, 1);
        w.u(2, 2);
        w.u(2, 1);
        w.ue(0);
        w.ue(0);
        w.u(1, 0);
        w.u(2, 0);
        w.se(0);
        w.se(0);
        w.se(0);
        w.u(1, 0);
        w.u(1, 0);
        w.u(1, 0);
        w.trailing();
        let bytes = w.into_bytes();

        let pps = Pps::parse(&bytes).unwrap();
        match pps.slice_group_map {
            Some(SliceGroupMap::Explicit {
                pic_size_in_map_units_minus1,
                slice_group_id,
            }) => {
                assert_eq!(pic_size_in_map_units_minus1, 3);
                assert_eq!(slice_group_id, vec![0, 1, 2, 1]);
            }
            other => panic!("unexpected slice_group_map: {:?}", other),
        }
    }

    #[test]
    fn pps_with_fmo_type_3_changing() {
        let mut w = BitWriter::new();
        w.ue(0);
        w.ue(0);
        w.u(1, 0);
        w.u(1, 0);
        w.ue(1); // num_slice_groups_minus1 = 1
        w.ue(3); // slice_group_map_type = 3
        w.u(1, 1); // slice_group_change_direction_flag
        w.ue(7); // slice_group_change_rate_minus1
        w.ue(0);
        w.ue(0);
        w.u(1, 0);
        w.u(2, 0);
        w.se(0);
        w.se(0);
        w.se(0);
        w.u(1, 0);
        w.u(1, 0);
        w.u(1, 0);
        w.trailing();
        let bytes = w.into_bytes();

        let pps = Pps::parse(&bytes).unwrap();
        match pps.slice_group_map {
            Some(SliceGroupMap::Changing {
                slice_group_map_type,
                change_direction_flag,
                change_rate_minus1,
            }) => {
                assert_eq!(slice_group_map_type, 3);
                assert!(change_direction_flag);
                assert_eq!(change_rate_minus1, 7);
            }
            other => panic!("unexpected slice_group_map: {:?}", other),
        }
    }

    #[test]
    fn pps_with_fmo_type_2_foreground() {
        // num_slice_groups_minus1=2 → two (top_left, bottom_right) pairs.
        let mut w = BitWriter::new();
        w.ue(0);
        w.ue(0);
        w.u(1, 0);
        w.u(1, 0);
        w.ue(2);
        w.ue(2); // slice_group_map_type = 2
        w.ue(0);
        w.ue(99);
        w.ue(5);
        w.ue(77);
        w.ue(0);
        w.ue(0);
        w.u(1, 0);
        w.u(2, 0);
        w.se(0);
        w.se(0);
        w.se(0);
        w.u(1, 0);
        w.u(1, 0);
        w.u(1, 0);
        w.trailing();
        let bytes = w.into_bytes();

        let pps = Pps::parse(&bytes).unwrap();
        match pps.slice_group_map {
            Some(SliceGroupMap::Foreground {
                top_left,
                bottom_right,
            }) => {
                assert_eq!(top_left, vec![0, 5]);
                assert_eq!(bottom_right, vec![99, 77]);
            }
            other => panic!("unexpected slice_group_map: {:?}", other),
        }
    }

    #[test]
    fn pps_rejects_out_of_range_chroma_qp_offset() {
        let mut w = BitWriter::new();
        w.ue(0);
        w.ue(0);
        w.u(1, 0);
        w.u(1, 0);
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.u(1, 0);
        w.u(2, 0);
        w.se(0);
        w.se(0);
        w.se(13); // out of range (-12..=12)
        let bytes = w.into_bytes();
        assert_eq!(
            Pps::parse(&bytes).unwrap_err(),
            PpsError::ChromaQpIndexOffsetOutOfRange(13)
        );
    }

    #[test]
    fn pps_rejects_weighted_bipred_idc_3() {
        let mut w = BitWriter::new();
        w.ue(0);
        w.ue(0);
        w.u(1, 0);
        w.u(1, 0);
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.u(1, 0);
        w.u(2, 3); // weighted_bipred_idc = 3 → invalid
        let bytes = w.into_bytes();
        assert_eq!(
            Pps::parse(&bytes).unwrap_err(),
            PpsError::WeightedBipredIdcOutOfRange(3)
        );
    }
}
