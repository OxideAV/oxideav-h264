//! §8.2.2 + §8.2.3 — FMO macroblock address derivation.
//!
//! Spec-driven implementation per ITU-T Rec. H.264 (08/2024). Covers:
//!
//! * §8.2.2 — `MapUnitToSliceGroupMap` derivation for all 7
//!   `slice_group_map_type` values (0..=6).
//! * §8.2.2.8 — conversion from the map-unit map to the macroblock
//!   map, handling frame-only / MBAFF / non-MBAFF-interlaced cases.
//! * §8.2.3 `NextMbAddress` — scan-order walk within a slice group.
//!
//! No external decoder source was consulted; only the ITU-T spec PDF.

#![allow(dead_code)]

use crate::pps::{Pps, SliceGroupMap};

/// §8.2.2 — errors surfaced while deriving the FMO maps.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum FmoError {
    /// §7.3.3 — `slice_group_change_cycle` is required for map types
    /// 3/4/5 but was not supplied.
    #[error("slice_group_change_cycle not supplied but slice_group_map_type is {0}")]
    MissingChangeCycle(u32),
    /// §7.4.2.1.1 eq. (7-17) — `PicSizeInMapUnits == 0` would make the
    /// derivation ill-defined.
    #[error("PicSizeInMapUnits is 0 (degenerate stream)")]
    EmptyPicture,
    /// §7.4.2.2 — `slice_group_id[i]` array length mismatches
    /// `pic_size_in_map_units_minus1 + 1` vs. `PicSizeInMapUnits`
    /// derived from the SPS.
    #[error("explicit slice_group_id length {got} != PicSizeInMapUnits {expected}")]
    ExplicitMapSizeMismatch { got: usize, expected: u32 },
    /// §8.2.2.3 — a foreground rectangle index exceeds
    /// `PicSizeInMapUnits`.
    #[error("foreground rectangle corner {0} out of range for PicSizeInMapUnits {1}")]
    ForegroundOutOfRange(u32, u32),
    /// Eq. (8-17) — a zero-length run would loop forever.
    #[error("interleaved run_length_minus1 array is empty")]
    EmptyRunLengths,
}

/// §8.2.2 — derive `MapUnitToSliceGroupMap[ 0..PicSizeInMapUnits-1 ]`.
///
/// `pic_size_in_map_units` is the §7.4.2.1.1 (7-17) product
/// `PicWidthInMbs * PicHeightInMapUnits`.
/// `slice_group_change_cycle` comes from the slice header (§7.3.3);
/// it is only consulted for map types 3/4/5. For the other types the
/// caller can pass 0.
///
/// `_mbaff_frame_flag` is not consulted here — §8.2.2 is purely a
/// map-unit derivation — but is accepted for symmetry with
/// [`mb_to_slice_group_map`].
pub fn map_unit_to_slice_group_map(
    pps: &Pps,
    pic_size_in_map_units: u32,
    pic_width_in_mbs: u32,
    slice_group_change_cycle: u32,
    _mbaff_frame_flag: bool,
) -> Result<Vec<u32>, FmoError> {
    if pic_size_in_map_units == 0 {
        return Err(FmoError::EmptyPicture);
    }
    let n = pic_size_in_map_units as usize;
    let num_slice_groups_minus1 = pps.num_slice_groups_minus1;

    // §8.2.2 eq. (8-15) — single slice group.
    if num_slice_groups_minus1 == 0 {
        return Ok(vec![0u32; n]);
    }
    let num_slice_groups = num_slice_groups_minus1 + 1;

    // The PPS must carry a SliceGroupMap in this branch; if missing,
    // fall back to all-zero (defensive).
    let Some(map) = pps.slice_group_map.as_ref() else {
        return Ok(vec![0u32; n]);
    };

    let mut out = vec![0u32; n];

    match map {
        // §8.2.2.1 — interleaved slice group map type (type 0).
        SliceGroupMap::Interleaved { run_length_minus1 } => {
            // eq. (8-17):
            //   i = 0
            //   do
            //     for( iGroup = 0; iGroup <= num_slice_groups_minus1 &&
            //                      i < PicSizeInMapUnits;
            //          i += run_length_minus1[iGroup++] + 1 )
            //       for( j = 0; j <= run_length_minus1[iGroup] &&
            //                   i + j < PicSizeInMapUnits; j++ )
            //         mapUnitToSliceGroupMap[i + j] = iGroup
            //   while( i < PicSizeInMapUnits )
            if run_length_minus1.is_empty() {
                return Err(FmoError::EmptyRunLengths);
            }
            let mut i: usize = 0;
            loop {
                let mut i_group: u32 = 0;
                while i_group <= num_slice_groups_minus1 && i < n {
                    let rl = run_length_minus1[i_group as usize] as usize;
                    let mut j: usize = 0;
                    while j <= rl && i + j < n {
                        out[i + j] = i_group;
                        j += 1;
                    }
                    i += rl + 1;
                    i_group += 1;
                }
                if i >= n {
                    break;
                }
            }
        }

        // §8.2.2.2 — dispersed slice group map type (type 1).
        SliceGroupMap::Dispersed => {
            // eq. (8-18):
            //   for( i = 0; i < PicSizeInMapUnits; i++ )
            //     mapUnitToSliceGroupMap[i] =
            //        ( ( i % PicWidthInMbs ) +
            //          ( ( ( i / PicWidthInMbs ) *
            //              ( num_slice_groups_minus1 + 1 ) ) / 2 ) )
            //        % ( num_slice_groups_minus1 + 1 )
            for i in 0..n {
                let i_u = i as u32;
                let x = i_u % pic_width_in_mbs;
                let y = i_u / pic_width_in_mbs;
                out[i] = (x + (y * num_slice_groups) / 2) % num_slice_groups;
            }
        }

        // §8.2.2.3 — foreground with left-over (type 2).
        SliceGroupMap::Foreground {
            top_left,
            bottom_right,
        } => {
            // eq. (8-19). Initialise every map unit to num_slice_groups_minus1,
            // then paint foreground rectangles in decreasing iGroup order.
            for i in 0..n {
                out[i] = num_slice_groups_minus1;
            }
            // iGroup goes from num_slice_groups_minus1 - 1 down to 0.
            // num_slice_groups_minus1 >= 1 here because we early-returned
            // when it was 0.
            for i_group in (0..num_slice_groups_minus1).rev() {
                let idx = i_group as usize;
                let tl = top_left[idx];
                let br = bottom_right[idx];
                if tl >= pic_size_in_map_units || br >= pic_size_in_map_units {
                    // Out-of-range corner — spec forbids it, but we surface
                    // rather than panic.
                    return Err(FmoError::ForegroundOutOfRange(
                        tl.max(br),
                        pic_size_in_map_units,
                    ));
                }
                let y_top_left = tl / pic_width_in_mbs;
                let x_top_left = tl % pic_width_in_mbs;
                let y_bottom_right = br / pic_width_in_mbs;
                let x_bottom_right = br % pic_width_in_mbs;
                let mut y = y_top_left;
                while y <= y_bottom_right {
                    let mut x = x_top_left;
                    while x <= x_bottom_right {
                        let idx = (y * pic_width_in_mbs + x) as usize;
                        if idx < n {
                            out[idx] = i_group;
                        }
                        x += 1;
                    }
                    y += 1;
                }
            }
        }

        // §8.2.2.4/5/6 — changing slice group map types (3/4/5).
        SliceGroupMap::Changing {
            slice_group_map_type,
            change_direction_flag,
            change_rate_minus1,
        } => {
            // MapUnitsInSliceGroup0 = Min( slice_group_change_rate *
            //   slice_group_change_cycle, PicSizeInMapUnits ) — per §7.4.3
            // (semantics of slice_group_change_cycle).
            let slice_group_change_rate = change_rate_minus1 + 1;
            let mapunits_in_slice_group0 = (slice_group_change_rate as u64)
                .saturating_mul(slice_group_change_cycle as u64)
                .min(pic_size_in_map_units as u64)
                as u32;

            match *slice_group_map_type {
                3 => {
                    // §8.2.2.4 box-out — eq. (8-20). num_slice_groups_minus1
                    // is required to be 1 for types 3/4/5.
                    let dir = *change_direction_flag as i32; // 0 or 1
                    // Initialise: mapUnitToSliceGroupMap[i] = 1 for all i.
                    for i in 0..n {
                        out[i] = 1;
                    }
                    let pic_width = pic_width_in_mbs as i32;
                    let pic_height =
                        (pic_size_in_map_units / pic_width_in_mbs) as i32;
                    let mut x: i32 = (pic_width - dir) / 2;
                    let mut y: i32 = (pic_height - dir) / 2;
                    let mut left_bound: i32 = x;
                    let mut top_bound: i32 = y;
                    let mut right_bound: i32 = x;
                    let mut bottom_bound: i32 = y;
                    let mut x_dir: i32 = dir - 1;
                    let mut y_dir: i32 = dir;

                    let mapunits_in_g0 = mapunits_in_slice_group0 as usize;
                    let mut k: usize = 0;
                    while k < mapunits_in_g0 {
                        let lin = (y * pic_width + x) as usize;
                        let map_unit_vacant = if lin < n { out[lin] == 1 } else { false };
                        if map_unit_vacant && lin < n {
                            out[lin] = 0;
                        }
                        if x_dir == -1 && x == left_bound {
                            left_bound = (left_bound - 1).max(0);
                            x = left_bound;
                            x_dir = 0;
                            y_dir = 2 * dir - 1;
                        } else if x_dir == 1 && x == right_bound {
                            right_bound = (right_bound + 1).min(pic_width - 1);
                            x = right_bound;
                            x_dir = 0;
                            y_dir = 1 - 2 * dir;
                        } else if y_dir == -1 && y == top_bound {
                            top_bound = (top_bound - 1).max(0);
                            y = top_bound;
                            x_dir = 1 - 2 * dir;
                            y_dir = 0;
                        } else if y_dir == 1 && y == bottom_bound {
                            bottom_bound = (bottom_bound + 1).min(pic_height - 1);
                            y = bottom_bound;
                            x_dir = 2 * dir - 1;
                            y_dir = 0;
                        } else {
                            x += x_dir;
                            y += y_dir;
                        }
                        k += 1;
                    }
                }
                4 => {
                    // §8.2.2.5 raster scan — eq. (8-21).
                    // sizeOfUpperLeftGroup — eq. (8-14):
                    //   slice_group_change_direction_flag
                    //     ? PicSizeInMapUnits - MapUnitsInSliceGroup0
                    //     : MapUnitsInSliceGroup0
                    let size_of_upper_left = if *change_direction_flag {
                        pic_size_in_map_units - mapunits_in_slice_group0
                    } else {
                        mapunits_in_slice_group0
                    };
                    let dir = *change_direction_flag as u32;
                    for i in 0..n {
                        out[i] = if (i as u32) < size_of_upper_left {
                            dir
                        } else {
                            1 - dir
                        };
                    }
                }
                5 => {
                    // §8.2.2.6 wipe — eq. (8-22).
                    let size_of_upper_left = if *change_direction_flag {
                        pic_size_in_map_units - mapunits_in_slice_group0
                    } else {
                        mapunits_in_slice_group0
                    };
                    let dir = *change_direction_flag as u32;
                    let pic_height =
                        pic_size_in_map_units / pic_width_in_mbs;
                    let mut k: u32 = 0;
                    for j in 0..pic_width_in_mbs {
                        for i in 0..pic_height {
                            let lin = (i * pic_width_in_mbs + j) as usize;
                            if lin < n {
                                out[lin] = if k < size_of_upper_left {
                                    dir
                                } else {
                                    1 - dir
                                };
                            }
                            k += 1;
                        }
                    }
                }
                other => {
                    // Shouldn't happen — PPS parse already rejects > 6
                    // and SliceGroupMap::Changing only carries 3/4/5.
                    return Err(FmoError::MissingChangeCycle(other));
                }
            }
        }

        // §8.2.2.7 — explicit map (type 6). Eq. (8-23).
        SliceGroupMap::Explicit {
            pic_size_in_map_units_minus1,
            slice_group_id,
        } => {
            let expected = *pic_size_in_map_units_minus1 + 1;
            if slice_group_id.len() != expected as usize {
                return Err(FmoError::ExplicitMapSizeMismatch {
                    got: slice_group_id.len(),
                    expected,
                });
            }
            // Honour the PPS-declared length: copy up to min(n, expected).
            // Note: the bitstream's pic_size_in_map_units_minus1 must
            // match the one derived from the SPS; if they differ we still
            // copy what we have and pad with 0 rather than panic.
            let take = n.min(slice_group_id.len());
            out[..take].copy_from_slice(&slice_group_id[..take]);
        }
    }

    Ok(out)
}

/// §8.2.2.8 — derive `MbToSliceGroupMap` from the map-unit map.
///
/// Per the spec:
/// * If `frame_mbs_only_flag == 1` or `field_pic_flag == 1` → eq. (8-24):
///   passthrough, `MbToSliceGroupMap[i] = mapUnitToSliceGroupMap[i]`.
/// * Else if `MbaffFrameFlag == 1` → eq. (8-25):
///   `MbToSliceGroupMap[i] = mapUnitToSliceGroupMap[i / 2]`.
/// * Else (non-MBAFF interlaced frame, i.e. `frame_mbs_only_flag == 0
///   && mb_adaptive_frame_field_flag == 0 && field_pic_flag == 0`) → eq. (8-26):
///   `MbToSliceGroupMap[i] = mapUnitToSliceGroupMap[
///      (i / (2*PicWidthInMbs)) * PicWidthInMbs + (i % PicWidthInMbs) ]`.
///
/// Returned vector length is `PicSizeInMbs` — which equals
/// `PicWidthInMbs * FrameHeightInMbs` for frame-coded pictures, or
/// `PicWidthInMbs * PicHeightInMapUnits` for field-coded pictures.
/// Callers supply the MB count directly via the output-length inference:
/// the frame-only case yields `map_unit_map.len()` entries; the MBAFF
/// case doubles it; the non-MBAFF interlaced case doubles it as well.
pub fn mb_to_slice_group_map(
    map_unit_map: &[u32],
    frame_mbs_only_flag: bool,
    mbaff_frame_flag: bool,
    field_pic_flag: bool,
    pic_width_in_mbs: u32,
) -> Vec<u32> {
    // §8.2.2.8 passthrough case — eq. (8-24).
    if frame_mbs_only_flag || field_pic_flag {
        return map_unit_map.to_vec();
    }

    let n_map_units = map_unit_map.len();
    let pic_size_in_mbs = n_map_units * 2; // interlaced frame

    if mbaff_frame_flag {
        // §8.2.2.8 eq. (8-25): i / 2.
        let mut out = vec![0u32; pic_size_in_mbs];
        for i in 0..pic_size_in_mbs {
            let mu = i / 2;
            if mu < n_map_units {
                out[i] = map_unit_map[mu];
            }
        }
        out
    } else {
        // §8.2.2.8 eq. (8-26): non-MBAFF interlaced frame.
        // mapUnitToSliceGroupMap[ (i / (2 * PicWidthInMbs)) * PicWidthInMbs
        //                         + (i % PicWidthInMbs) ]
        let w = pic_width_in_mbs as usize;
        let two_w = 2 * w;
        let mut out = vec![0u32; pic_size_in_mbs];
        for i in 0..pic_size_in_mbs {
            let mu = (i / two_w) * w + (i % w);
            if mu < n_map_units {
                out[i] = map_unit_map[mu];
            }
        }
        out
    }
}

/// §8.2.3 eq. (8-16) — `NextMbAddress( n )`.
///
/// Walk MB addresses in scan order starting at `current_mb_addr + 1`
/// and return the first address whose slice group matches `slice_group`.
/// Returns `pic_size_in_mbs` as a "no more MBs in this slice group"
/// sentinel.
///
/// The canonical spec form compares against
/// `MbToSliceGroupMap[ n ]` directly; we accept the slice group
/// explicitly so callers already know which group they're walking.
pub fn next_mb_address(
    current_mb_addr: u32,
    mb_to_slice_group_map: &[u32],
    slice_group: u32,
    pic_size_in_mbs: u32,
) -> u32 {
    // eq. (8-16):
    //   i = n + 1
    //   while( i < PicSizeInMbs && MbToSliceGroupMap[i] != MbToSliceGroupMap[n] )
    //     i++
    //   nextMbAddress = i
    let mut i = current_mb_addr.saturating_add(1);
    let len = mb_to_slice_group_map.len() as u32;
    let end = pic_size_in_mbs.min(len);
    while i < end && mb_to_slice_group_map[i as usize] != slice_group {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pps::{Pps, SliceGroupMap};

    /// Helper: build a minimal PPS with a given slice_group_map and
    /// num_slice_groups_minus1. All other fields are zeroed/defaulted.
    fn make_pps(num_slice_groups_minus1: u32, map: Option<SliceGroupMap>) -> Pps {
        Pps {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
            entropy_coding_mode_flag: false,
            bottom_field_pic_order_in_frame_present_flag: false,
            num_slice_groups_minus1,
            slice_group_map: map,
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

    // --- §8.2.2 eq. (8-15) — single slice group ---

    #[test]
    fn no_fmo_all_zeros() {
        let pps = make_pps(0, None);
        let map = map_unit_to_slice_group_map(&pps, 16, 4, 0, false).unwrap();
        assert_eq!(map, vec![0u32; 16]);
    }

    // --- §8.2.2.1 Type 0 interleaved — eq. (8-17) ---

    #[test]
    fn type0_interleaved_two_groups() {
        // num_slice_groups = 2, run_length_minus1 = [2, 4] →
        // run_length = [3, 5]. PicSize = 16.
        // Per eq. (8-17): the outer do-while keeps cycling iGroup=0,1
        // until PicSize is exhausted.
        //   i=0: iGroup=0, j=0..3 → mb 0..2 = 0, i=3
        //         iGroup=1, j=0..5 → mb 3..7 = 1, i=8
        //   i=8: iGroup=0, j=0..3 → mb 8..10 = 0, i=11
        //         iGroup=1, j=0..5 → mb 11..15 = 1, i=16 → exit.
        let pps = make_pps(
            1,
            Some(SliceGroupMap::Interleaved {
                run_length_minus1: vec![2, 4],
            }),
        );
        let map = map_unit_to_slice_group_map(&pps, 16, 4, 0, false).unwrap();
        assert_eq!(map, vec![0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn type0_interleaved_runs_of_one_each() {
        // run_length_minus1 = [0, 0] → run = 1 each. Pattern alternates
        // 0,1,0,1,...
        let pps = make_pps(
            1,
            Some(SliceGroupMap::Interleaved {
                run_length_minus1: vec![0, 0],
            }),
        );
        let map = map_unit_to_slice_group_map(&pps, 8, 4, 0, false).unwrap();
        assert_eq!(map, vec![0, 1, 0, 1, 0, 1, 0, 1]);
    }

    // --- §8.2.2.2 Type 1 dispersed — eq. (8-18) ---

    #[test]
    fn type1_dispersed_two_groups_4x4() {
        // num_slice_groups=2, PicWidth=4, PicSize=16.
        // sliceGroup[i] = (x + (y * 2)/2) % 2 = (x + y) % 2
        //   y=0: 0,1,0,1
        //   y=1: 1,0,1,0
        //   y=2: 0,1,0,1
        //   y=3: 1,0,1,0
        let pps = make_pps(1, Some(SliceGroupMap::Dispersed));
        let map = map_unit_to_slice_group_map(&pps, 16, 4, 0, false).unwrap();
        assert_eq!(map, vec![0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]);
    }

    #[test]
    fn type1_dispersed_three_groups_3x3() {
        // num_slice_groups=3, PicWidth=3, PicSize=9.
        // sliceGroup[i] = (x + (y * 3)/2) % 3
        //   i=0: x=0,y=0 → (0+0)%3 = 0
        //   i=1: x=1,y=0 → (1+0)%3 = 1
        //   i=2: x=2,y=0 → (2+0)%3 = 2
        //   i=3: x=0,y=1 → (0+1)%3 = 1   (3/2=1)
        //   i=4: x=1,y=1 → (1+1)%3 = 2
        //   i=5: x=2,y=1 → (2+1)%3 = 0
        //   i=6: x=0,y=2 → (0+3)%3 = 0   (6/2=3)
        //   i=7: x=1,y=2 → (1+3)%3 = 1
        //   i=8: x=2,y=2 → (2+3)%3 = 2
        let pps = make_pps(2, Some(SliceGroupMap::Dispersed));
        let map = map_unit_to_slice_group_map(&pps, 9, 3, 0, false).unwrap();
        assert_eq!(map, vec![0, 1, 2, 1, 2, 0, 0, 1, 2]);
    }

    // --- §8.2.2.3 Type 2 foreground + leftover — eq. (8-19) ---

    #[test]
    fn type2_foreground_one_rectangle() {
        // num_slice_groups=2 → num_slice_groups_minus1=1 → 1 foreground
        // rectangle, plus leftover group 1.
        // top_left=[5], bottom_right=[10], PicWidth=4, PicSize=16.
        //   yTopLeft = 5/4 = 1, xTopLeft = 5%4 = 1
        //   yBottomRight = 10/4 = 2, xBottomRight = 10%4 = 2
        //   Rectangle: rows 1..=2, cols 1..=2.
        //   Covered: idx 5, 6, 9, 10 ← group 0. Everything else = 1.
        let pps = make_pps(
            1,
            Some(SliceGroupMap::Foreground {
                top_left: vec![5],
                bottom_right: vec![10],
            }),
        );
        let map = map_unit_to_slice_group_map(&pps, 16, 4, 0, false).unwrap();
        let mut expected = vec![1u32; 16];
        expected[5] = 0;
        expected[6] = 0;
        expected[9] = 0;
        expected[10] = 0;
        assert_eq!(map, expected);
    }

    #[test]
    fn type2_foreground_two_rectangles_priority() {
        // Two foreground rectangles: group 0 covers rows 0..=1, cols
        // 0..=3 (idx 0..8); group 1 covers rows 1..=2, cols 0..=3
        // (idx 4..12). Leftover group 2 fills the rest. Because painting
        // goes iGroup = num_slice_groups_minus1 - 1 (=1) down to 0, the
        // group-0 rectangle ends up on top: indices in both rectangles
        // go to group 0.
        let pps = make_pps(
            2,
            Some(SliceGroupMap::Foreground {
                top_left: vec![0, 4],
                bottom_right: vec![7, 11],
            }),
        );
        let map = map_unit_to_slice_group_map(&pps, 16, 4, 0, false).unwrap();
        // 0..=7  → group 0 (both rectangles, 0 wins).
        // 8..=11 → group 1.
        // 12..=15 → group 2 (leftover).
        let expected = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
        ];
        assert_eq!(map, expected);
    }

    // --- §8.2.2.4 Type 3 box-out — eq. (8-20) ---

    #[test]
    fn type3_box_out_clockwise_small() {
        // Small 4x4 picture, change_direction_flag=0 (box-out CW).
        // change_rate=1, change_cycle=3 → MapUnitsInSliceGroup0=3.
        // Initial (x,y) = ((4-0)/2, (4-0)/2) = (2,2). xDir=-1, yDir=0.
        // Trace:
        //  k=0: vacant, mark (2,2)=0. xDir==-1 && x==leftBound(2):
        //        leftBound=1, x=1, xDir=0, yDir=-1.
        //  k=1: vacant, mark (1,2)=0. yDir==-1 && y==topBound(2):
        //        topBound=1, y=1, xDir=1, yDir=0.
        //  k=2: vacant, mark (1,1)=0. xDir==1 && x==rightBound(2):
        //        rightBound=3, x=3, xDir=0, yDir=1. — but we only do 3 iterations.
        //   k increments to 3 → loop exits.
        // Output: all 1s except indices {2*4+2=10, 2*4+1=9, 1*4+1=5} = 0.
        let pps = make_pps(
            1,
            Some(SliceGroupMap::Changing {
                slice_group_map_type: 3,
                change_direction_flag: false,
                change_rate_minus1: 0, // rate = 1
            }),
        );
        let map = map_unit_to_slice_group_map(&pps, 16, 4, 3, false).unwrap();
        let mut expected = vec![1u32; 16];
        expected[10] = 0;
        expected[9] = 0;
        expected[5] = 0;
        assert_eq!(map, expected);
    }

    #[test]
    fn type3_box_out_size_zero() {
        // With MapUnitsInSliceGroup0=0 (change_cycle=0), nothing is
        // marked — all entries stay at 1.
        let pps = make_pps(
            1,
            Some(SliceGroupMap::Changing {
                slice_group_map_type: 3,
                change_direction_flag: false,
                change_rate_minus1: 0,
            }),
        );
        let map = map_unit_to_slice_group_map(&pps, 16, 4, 0, false).unwrap();
        assert_eq!(map, vec![1u32; 16]);
    }

    // --- §8.2.2.5 Type 4 raster scan — eq. (8-21) ---

    #[test]
    fn type4_raster_direction_0() {
        // change_direction_flag=0 → sizeOfUpperLeftGroup=MapUnitsInSliceGroup0=6.
        //   i < 6 → slice_group_change_direction_flag = 0
        //   i >= 6 → 1 - 0 = 1
        let pps = make_pps(
            1,
            Some(SliceGroupMap::Changing {
                slice_group_map_type: 4,
                change_direction_flag: false,
                change_rate_minus1: 1, // rate = 2
            }),
        );
        // PicSize=16, rate=2, cycle=3 → MapUnitsInSliceGroup0=6.
        let map = map_unit_to_slice_group_map(&pps, 16, 4, 3, false).unwrap();
        let expected = vec![
            0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];
        assert_eq!(map, expected);
    }

    #[test]
    fn type4_raster_direction_1() {
        // change_direction_flag=1 → sizeOfUpperLeftGroup = PicSize - mapUnits0
        //   = 16 - 6 = 10. First 10 → 1; rest → 0.
        let pps = make_pps(
            1,
            Some(SliceGroupMap::Changing {
                slice_group_map_type: 4,
                change_direction_flag: true,
                change_rate_minus1: 1,
            }),
        );
        let map = map_unit_to_slice_group_map(&pps, 16, 4, 3, false).unwrap();
        let expected = vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(map, expected);
    }

    // --- §8.2.2.6 Type 5 wipe — eq. (8-22) ---

    #[test]
    fn type5_wipe_direction_0() {
        // change_direction_flag=0 → first MapUnitsInSliceGroup0 in
        // column-major scan get group 0; rest get group 1.
        // PicSize=16, PicWidth=4, PicHeight=4. rate=1, cycle=3 → size0=3.
        // Column-major k order:
        //   k=0..3 are (j=0, i=0..3) → idx 0,4,8,12 → group 0 for k<3 → 0,4,8=0; 12=1.
        //   k=4..7 (j=1) idx 1,5,9,13 → all group 1.
        //   (etc.)
        let pps = make_pps(
            1,
            Some(SliceGroupMap::Changing {
                slice_group_map_type: 5,
                change_direction_flag: false,
                change_rate_minus1: 0,
            }),
        );
        let map = map_unit_to_slice_group_map(&pps, 16, 4, 3, false).unwrap();
        let mut expected = vec![1u32; 16];
        expected[0] = 0;
        expected[4] = 0;
        expected[8] = 0;
        assert_eq!(map, expected);
    }

    // --- §8.2.2.7 Type 6 explicit — eq. (8-23) ---

    #[test]
    fn type6_explicit_passthrough() {
        let pps = make_pps(
            1,
            Some(SliceGroupMap::Explicit {
                pic_size_in_map_units_minus1: 7,
                slice_group_id: vec![0, 0, 1, 1, 0, 0, 1, 1],
            }),
        );
        let map = map_unit_to_slice_group_map(&pps, 8, 4, 0, false).unwrap();
        assert_eq!(map, vec![0, 0, 1, 1, 0, 0, 1, 1]);
    }

    // --- §8.2.2.8 — MbToSliceGroupMap conversion ---

    #[test]
    fn mb_map_passthrough_frame_only() {
        // eq. (8-24): frame_mbs_only_flag=1 → direct copy.
        let mu_map = vec![0, 1, 0, 1, 2, 2];
        let mb_map = mb_to_slice_group_map(&mu_map, true, false, false, 3);
        assert_eq!(mb_map, mu_map);
    }

    #[test]
    fn mb_map_passthrough_field_pic() {
        // eq. (8-24): field_pic_flag=1 → direct copy.
        let mu_map = vec![3, 3, 2, 1];
        let mb_map = mb_to_slice_group_map(&mu_map, false, false, true, 2);
        assert_eq!(mb_map, mu_map);
    }

    #[test]
    fn mb_map_mbaff_i_over_2() {
        // eq. (8-25): MBAFF frame → MbToSliceGroupMap[i] = mapUnitMap[i/2].
        let mu_map = vec![0, 1, 2, 3];
        let mb_map = mb_to_slice_group_map(&mu_map, false, true, false, 2);
        assert_eq!(mb_map, vec![0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn mb_map_non_mbaff_interlaced_frame() {
        // eq. (8-26): non-MBAFF interlaced frame — frame_mbs_only=0,
        // mbaff=0, field_pic=0.
        // PicWidth=2, map-unit map is 2 wide x 2 tall (4 entries).
        // PicSize in MBs = 8. For each i:
        //   mu = (i / (2*2)) * 2 + (i % 2) = (i/4)*2 + (i%2)
        //   i=0: (0)*2 + 0 = 0
        //   i=1: (0)*2 + 1 = 1
        //   i=2: (0)*2 + 0 = 0
        //   i=3: (0)*2 + 1 = 1
        //   i=4: (1)*2 + 0 = 2
        //   i=5: (1)*2 + 1 = 3
        //   i=6: (1)*2 + 0 = 2
        //   i=7: (1)*2 + 1 = 3
        let mu_map = vec![10, 20, 30, 40];
        let mb_map = mb_to_slice_group_map(&mu_map, false, false, false, 2);
        assert_eq!(mb_map, vec![10, 20, 10, 20, 30, 40, 30, 40]);
    }

    // --- §8.2.3 eq. (8-16) NextMbAddress ---

    #[test]
    fn next_mb_skips_other_slice_group() {
        // Slice group layout: [0,1,1,0,1,0,0,1]. Walking slice group 0
        // from addr 0 → next should be 3.
        let mb_map = vec![0u32, 1, 1, 0, 1, 0, 0, 1];
        let next = next_mb_address(0, &mb_map, 0, 8);
        assert_eq!(next, 3);
    }

    #[test]
    fn next_mb_same_slice_group_adjacent() {
        let mb_map = vec![0u32, 0, 1, 0, 1, 0, 0, 1];
        // From addr 0, slice group 0 → next is 1.
        assert_eq!(next_mb_address(0, &mb_map, 0, 8), 1);
        // From addr 2 (which is group 1), walking group 1 → next is 4.
        assert_eq!(next_mb_address(2, &mb_map, 1, 8), 4);
    }

    #[test]
    fn next_mb_returns_pic_size_when_done() {
        let mb_map = vec![0u32, 0, 0, 1];
        // Walking group 0 starting at 2 → next = 3 is group 1 → 4 = pic_size.
        let next = next_mb_address(2, &mb_map, 0, 4);
        assert_eq!(next, 4); // sentinel
    }

    #[test]
    fn next_mb_current_at_end() {
        // current == pic_size_in_mbs - 1, no next → sentinel.
        let mb_map = vec![0u32, 0, 0];
        let next = next_mb_address(2, &mb_map, 0, 3);
        assert_eq!(next, 3);
    }

    // --- Defensive tests ---

    #[test]
    fn empty_picture_rejected() {
        let pps = make_pps(0, None);
        let err = map_unit_to_slice_group_map(&pps, 0, 4, 0, false).unwrap_err();
        assert_eq!(err, FmoError::EmptyPicture);
    }

    #[test]
    fn explicit_length_mismatch_detected() {
        let pps = make_pps(
            1,
            Some(SliceGroupMap::Explicit {
                pic_size_in_map_units_minus1: 3,
                slice_group_id: vec![0, 0], // too short
            }),
        );
        let err = map_unit_to_slice_group_map(&pps, 4, 2, 0, false).unwrap_err();
        assert!(matches!(err, FmoError::ExplicitMapSizeMismatch { .. }));
    }

    #[test]
    fn type0_bounded_by_pic_size() {
        // If PicSize < sum of runs, loop should clip correctly.
        // run_length_minus1 = [99, 99] (rl=100). PicSize=3.
        //   i=0: iGroup=0, j=0..100 but +i<3 → sets idx 0,1,2 = 0; i=100.
        //   100 >= 3 → exit while in outer.
        let pps = make_pps(
            1,
            Some(SliceGroupMap::Interleaved {
                run_length_minus1: vec![99, 99],
            }),
        );
        let map = map_unit_to_slice_group_map(&pps, 3, 3, 0, false).unwrap();
        assert_eq!(map, vec![0, 0, 0]);
    }
}
