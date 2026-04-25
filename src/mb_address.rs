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
            #[allow(clippy::needless_range_loop)] // spec §8.2.2.2 eq. (8-18)
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
            for slot in out.iter_mut().take(n) {
                *slot = num_slice_groups_minus1;
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
                    for slot in out.iter_mut().take(n) {
                        *slot = 1;
                    }
                    let pic_width = pic_width_in_mbs as i32;
                    let pic_height = (pic_size_in_map_units / pic_width_in_mbs) as i32;
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
                    for (i, slot) in out.iter_mut().enumerate().take(n) {
                        *slot = if (i as u32) < size_of_upper_left {
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
                    let pic_height = pic_size_in_map_units / pic_width_in_mbs;
                    let mut k: u32 = 0;
                    for j in 0..pic_width_in_mbs {
                        for i in 0..pic_height {
                            let lin = (i * pic_width_in_mbs + j) as usize;
                            if lin < n {
                                out[lin] = if k < size_of_upper_left { dir } else { 1 - dir };
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
        for (i, slot) in out.iter_mut().enumerate() {
            let mu = i / 2;
            if mu < n_map_units {
                *slot = map_unit_map[mu];
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
        for (i, slot) in out.iter_mut().enumerate() {
            let mu = (i / two_w) * w + (i % w);
            if mu < n_map_units {
                *slot = map_unit_map[mu];
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

// =========================================================================
// §6.4.1 / §6.4.10 / §6.4.11.1 — MBAFF addressing helpers
// =========================================================================
//
// When MbaffFrameFlag == 1 (§7.4.2.1.1 — i.e. `mb_adaptive_frame_field_flag`
// in the SPS is 1 and the slice header's `field_pic_flag` is 0) macroblocks
// are arranged in vertical pairs. Given CurrMbAddr:
//   * currMbFrameFlag   = !mb_field_decoding_flag  (frame-coded MB when true)
//   * mbIsTopMbFlag     = (CurrMbAddr % 2 == 0)    (top MB of its pair)
//   * pairIdx           = CurrMbAddr / 2           (pair raster index)
// The pair origin in the frame is `(xO, yO)` from §6.4.1 eqs. (6-5)/(6-6):
//   xO = (pairIdx % PicWidthInMbs) * 16
//   yO = (pairIdx / PicWidthInMbs) * 32
// Within the pair (§6.4.1 eqs. 6-7..6-10):
//   * Frame MB: x = xO, y = yO + (CurrMbAddr % 2) * 16.
//   * Field MB: x = xO, y = yO + (CurrMbAddr % 2). (Field MB samples are
//     then placed every 2 rows starting from this y — i.e. interleaved
//     with the pair partner.)
//
// §6.4.10 specifies the neighbour pair addresses (A/B/C/D) relative to
// the top MB of the current pair; these are used downstream by §6.4.11.1
// to derive the per-MB neighbour addresses accounting for the
// current+neighbour `mb_field_decoding_flag` combination.

/// §6.4.10 — derive `(currMbFrameFlag, mbIsTopMbFlag)` for the MB at
/// `curr_mb_addr` in an MBAFF frame picture.
///
/// * `currMbFrameFlag` == `true` when the MB is frame-coded
///   (`mb_field_decoding_flag` is 0).
/// * `mbIsTopMbFlag` == `true` when the MB is the top of its pair
///   (`CurrMbAddr % 2 == 0`).
pub fn mb_pair_flags(curr_mb_addr: u32, mb_field_decoding_flag: bool) -> (bool, bool) {
    let mb_is_top = curr_mb_addr % 2 == 0;
    let curr_frame = !mb_field_decoding_flag;
    (curr_frame, mb_is_top)
}

/// §6.4.1 — inverse macroblock-to-sample mapping for MBAFF frame
/// pictures. Returns `(x, y)` — the upper-left luma-sample coordinate
/// of `curr_mb_addr` within the frame.
///
/// For field-coded MBs, the returned `y` is the starting row; the MB's
/// samples occupy rows `y, y+2, y+4, ...` (every 2 rows) per eqs.
/// (6-9)/(6-10) — the caller is responsible for applying the
/// `y_step = 2` stride.
///
/// For frame-coded MBs, `y` is the MB origin and the samples are
/// contiguous (`y_step = 1`).
pub fn mbaff_mb_to_sample_xy(
    curr_mb_addr: u32,
    pic_width_in_mbs: u32,
    mb_field_decoding_flag: bool,
) -> (u32, u32) {
    if pic_width_in_mbs == 0 {
        return (0, 0);
    }
    let pair_idx = curr_mb_addr / 2;
    // §6.4.1 eqs. (6-5)/(6-6) via `InverseRasterScan(a, 16, 32, PicW, e)`
    // which expands to x = (a % (PicW/16)) * 16, y = (a / (PicW/16)) * 32.
    // Here `pic_width_in_mbs = PicW / 16`.
    let x_o = (pair_idx % pic_width_in_mbs) * 16;
    let y_o = (pair_idx / pic_width_in_mbs) * 32;
    let bot = curr_mb_addr % 2; // 0 for top, 1 for bottom.
    if mb_field_decoding_flag {
        // §6.4.1 eqs. (6-9)/(6-10) — field MB starting row offset by 0 or 1.
        (x_o, y_o + bot)
    } else {
        // §6.4.1 eqs. (6-7)/(6-8) — frame MB stacked at 0 or 16.
        (x_o, y_o + bot * 16)
    }
}

/// §6.4.1 — y-stride used to step between rows within an MBAFF MB.
/// * Frame-coded MB → stride of 1 (samples at y, y+1, y+2, ...).
/// * Field-coded MB → stride of 2 (samples at y, y+2, y+4, ..., i.e.
///   interleaved with the pair partner — §6.4.1 note under eq. 6-10).
#[inline]
pub fn mbaff_y_step(mb_field_decoding_flag: bool) -> u32 {
    if mb_field_decoding_flag {
        2
    } else {
        1
    }
}

/// §6.4.10 — neighbour macroblock pair addresses for the pair
/// containing `curr_mb_addr` in an MBAFF frame picture.
///
/// Returns `[A, B, C, D]` where each entry is the `mbAddr` of the
/// **top macroblock** of the neighbouring pair (to the left, above,
/// above-right, above-left respectively), or `None` when the neighbour
/// is outside the picture.
///
/// The spec formulas (for the pair-level addresses) are:
/// * `mbAddrA = 2 * ( CurrMbAddr / 2 - 1 )`
///   — invalid when `( CurrMbAddr / 2 ) % PicWidthInMbs == 0`.
/// * `mbAddrB = 2 * ( CurrMbAddr / 2 - PicWidthInMbs )`
/// * `mbAddrC = 2 * ( CurrMbAddr / 2 - PicWidthInMbs + 1 )`
///   — invalid when `( CurrMbAddr / 2 + 1 ) % PicWidthInMbs == 0`.
/// * `mbAddrD = 2 * ( CurrMbAddr / 2 - PicWidthInMbs - 1 )`
///   — invalid when `( CurrMbAddr / 2 ) % PicWidthInMbs == 0`.
///
/// Each addition is also guarded against "address > CurrMbAddr" per
/// §6.4.8 ("the macroblock is marked as not available" when mbAddr is
/// less than 0 or greater than CurrMbAddr). We enforce that by using
/// integer underflow guards.
///
/// This function does NOT further disambiguate the per-MB neighbour
/// within the pair — that is a §6.4.11.1 (MBAFF case) follow-up using
/// Tables 6-4/6-5 depending on current+neighbour mb_field_decoding_flag
/// and mbIsTopMbFlag. The pair-level base addresses are all a Phase-2
/// consumer needs to build a best-effort lookup.
pub fn mbaff_pair_neighbour_addrs(curr_mb_addr: u32, pic_width_in_mbs: u32) -> [Option<u32>; 4] {
    if pic_width_in_mbs == 0 {
        return [None; 4];
    }
    let pair_idx = curr_mb_addr / 2;
    let pair_col = pair_idx % pic_width_in_mbs;
    let pair_row = pair_idx / pic_width_in_mbs;

    // A: left pair. Not available when pair_col == 0.
    let a = if pair_col > 0 {
        Some(2 * (pair_idx - 1))
    } else {
        None
    };
    // B: above pair. Not available when pair_row == 0.
    let b = if pair_row > 0 {
        Some(2 * (pair_idx - pic_width_in_mbs))
    } else {
        None
    };
    // C: above-right pair. Not available when pair_row == 0 or
    // (pair_idx / 2 + 1) % PicWidthInMbs == 0 (i.e. at right edge).
    let c = if pair_row > 0 && pair_col + 1 < pic_width_in_mbs {
        Some(2 * (pair_idx - pic_width_in_mbs + 1))
    } else {
        None
    };
    // D: above-left pair.
    let d = if pair_row > 0 && pair_col > 0 {
        Some(2 * (pair_idx - pic_width_in_mbs - 1))
    } else {
        None
    };
    [a, b, c, d]
}

/// §6.4.11.1 (MBAFF case) — best-effort neighbour MB address derivation
/// for the current MB in an MBAFF frame picture. Consumers use this
/// for intra-pred neighbour lookup (§8.3.1.1 / §8.3.2.1).
///
/// The spec's full Table 6-4 disambiguates the per-MB neighbour (top or
/// bottom of the neighbour pair) based on the 4-way combination of
/// (current frame-or-field, current top-or-bottom, neighbour frame-or-
/// field, neighbour top-or-bottom). For this phase we use a common-case
/// approximation:
///
/// * **Left neighbour A**: same-pair role of the left pair's MB —
///   take mb_addr mapped across pair boundary at the same within-pair
///   index (top<->top, bottom<->bottom). For frame/frame and field/field
///   pairings this matches Table 6-4; for mixed pairings the result
///   falls back to the pair's top MB (mb_addr even), which is the Table
///   6-4 fallback for the common cases.
/// * **Above neighbour B**: the bottom MB of the above pair
///   (mb_addr + 1) when current is the top of its pair; otherwise
///   (current is bottom) the current MB's own top. This matches the
///   common case in Table 6-4 where the above-pair's bottom MB is the
///   row immediately above the current pair.
///
/// Cases where this approximation is wrong (e.g. frame<->field mixed
/// pairs near the top of the picture) should be treated as "neighbour
/// unavailable" by the caller if correctness matters — this is
/// documented as a Phase-3 refinement.
///
/// `mb_field_decoding_flag_grid` — per-MB flag, length
/// `pic_size_in_mbs`. Pass the grid from the parsed SliceData.
pub fn mbaff_neighbour_addrs(
    curr_mb_addr: u32,
    pic_width_in_mbs: u32,
    mb_field_decoding_flag_grid: &[bool],
) -> [Option<u32>; 4] {
    let pair_neighbours = mbaff_pair_neighbour_addrs(curr_mb_addr, pic_width_in_mbs);
    let mb_is_top = curr_mb_addr % 2 == 0;
    let curr_field = mb_field_decoding_flag_grid
        .get(curr_mb_addr as usize)
        .copied()
        .unwrap_or(false);

    // Return Some(addr) only when within grid bounds. Helper closure.
    let in_bounds = |addr: u32| -> Option<u32> {
        if (addr as usize) < mb_field_decoding_flag_grid.len() {
            Some(addr)
        } else {
            None
        }
    };

    // §6.4.11.1 (MBAFF) approximation, per doc-comment:
    //
    //   A (left): map to the neighbour pair's top MB when current is
    //   top, or the bottom MB when current is bottom. For mixed-mode
    //   pairs this is a Phase-2 approximation.
    //   B (above): bottom MB of above pair when current is top;
    //   else current-pair top MB when current is bottom.
    //   C (above-right): top MB of the above-right pair — conservative.
    //   D (above-left): top MB of the above-left pair — conservative.
    let pair_a = pair_neighbours[0];
    let pair_b = pair_neighbours[1];
    let pair_c = pair_neighbours[2];
    let pair_d = pair_neighbours[3];

    let a = pair_a.and_then(|base| {
        let neigh_field = mb_field_decoding_flag_grid
            .get(base as usize)
            .copied()
            .unwrap_or(false);
        // Same-pair-role mapping when the two pairs share
        // mb_field_decoding_flag; otherwise fall back to the pair's top
        // MB (the Table 6-4 common-case fallback).
        if neigh_field == curr_field {
            let off = if mb_is_top { 0 } else { 1 };
            in_bounds(base + off)
        } else {
            in_bounds(base)
        }
    });

    let b = if mb_is_top {
        // Current top → B is bottom of above pair (base + 1).
        pair_b.and_then(|base| in_bounds(base + 1))
    } else {
        // Current bottom → B is current-pair top = curr_mb_addr - 1.
        if curr_mb_addr > 0 {
            in_bounds(curr_mb_addr - 1)
        } else {
            None
        }
    };

    let c = if mb_is_top {
        // Current top → C is bottom of above-right pair.
        pair_c.and_then(|base| in_bounds(base + 1))
    } else {
        // Current bottom → C is top of current-right pair (if present).
        // This differs from §6.4.11.1 but is conservative: unavailable
        // C lets the caller fall back to D per Table 6-3.
        None
    };

    let d = if mb_is_top {
        // Current top → D is bottom of above-left pair.
        pair_d.and_then(|base| in_bounds(base + 1))
    } else {
        // Current bottom → D is top of left pair (same pair column).
        pair_a
    };

    [a, b, c, d]
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
        let expected = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
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
        let expected = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
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
        let expected = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0];
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

    // =====================================================================
    // §6.4.1 / §6.4.10 — MBAFF addressing helpers
    // =====================================================================

    // --- §6.4.10 `mb_pair_flags` ---

    #[test]
    fn mb_pair_flags_top_frame() {
        // CurrMbAddr 0 — top of pair 0, frame-coded.
        let (frame, top) = mb_pair_flags(0, false);
        assert!(frame, "mb_field_decoding_flag=0 → currMbFrameFlag=1");
        assert!(top, "CurrMbAddr % 2 == 0 → mbIsTopMbFlag=1");
    }

    #[test]
    fn mb_pair_flags_bottom_field() {
        // CurrMbAddr 1 — bottom of pair 0, field-coded.
        let (frame, top) = mb_pair_flags(1, true);
        assert!(!frame, "mb_field_decoding_flag=1 → currMbFrameFlag=0");
        assert!(!top, "CurrMbAddr % 2 == 1 → mbIsTopMbFlag=0");
    }

    #[test]
    fn mb_pair_flags_pair_three_bottom_frame() {
        // CurrMbAddr 7 — bottom of pair 3, frame-coded.
        let (frame, top) = mb_pair_flags(7, false);
        assert!(frame);
        assert!(!top);
    }

    // --- §6.4.1 `mbaff_mb_to_sample_xy` ---

    #[test]
    fn mbaff_xy_frame_top_of_first_pair() {
        // PicW = 4 MBs. Pair 0 top: (0, 0). Frame-coded.
        let (x, y) = mbaff_mb_to_sample_xy(0, 4, false);
        assert_eq!((x, y), (0, 0));
    }

    #[test]
    fn mbaff_xy_frame_bottom_of_first_pair() {
        // Pair 0 bottom (frame): y = 0 + 1*16 = 16.
        let (x, y) = mbaff_mb_to_sample_xy(1, 4, false);
        assert_eq!((x, y), (0, 16));
    }

    #[test]
    fn mbaff_xy_field_top_of_first_pair() {
        // Pair 0 top (field): y = 0 + 0 = 0.
        let (x, y) = mbaff_mb_to_sample_xy(0, 4, true);
        assert_eq!((x, y), (0, 0));
    }

    #[test]
    fn mbaff_xy_field_bottom_of_first_pair() {
        // Pair 0 bottom (field): y = 0 + 1 = 1 (interleaved — next row).
        let (x, y) = mbaff_mb_to_sample_xy(1, 4, true);
        assert_eq!((x, y), (0, 1));
    }

    #[test]
    fn mbaff_xy_second_pair_top_frame() {
        // PicW = 4. Pair 1 is at column 1 of row 0 → xO=16, yO=0.
        // MB 2 = top of pair 1, frame.
        let (x, y) = mbaff_mb_to_sample_xy(2, 4, false);
        assert_eq!((x, y), (16, 0));
    }

    #[test]
    fn mbaff_xy_second_row_pair_top_frame() {
        // PicW = 4, so row 1 starts at pair 4 = mb_addr 8.
        let (x, y) = mbaff_mb_to_sample_xy(8, 4, false);
        assert_eq!((x, y), (0, 32));
    }

    #[test]
    fn mbaff_xy_y_step_helper() {
        assert_eq!(mbaff_y_step(false), 1, "frame MB → stride 1");
        assert_eq!(mbaff_y_step(true), 2, "field MB → stride 2");
    }

    #[test]
    fn mbaff_xy_zero_width_is_safe() {
        // Degenerate input — should not panic; returns (0, 0).
        let (x, y) = mbaff_mb_to_sample_xy(5, 0, false);
        assert_eq!((x, y), (0, 0));
    }

    // --- §6.4.10 `mbaff_pair_neighbour_addrs` + `mbaff_neighbour_addrs` ---

    #[test]
    fn pair_neighbours_top_left_corner_all_none() {
        // CurrMbAddr=0, pair 0, row=0 col=0. No neighbours.
        let ns = mbaff_pair_neighbour_addrs(0, 4);
        assert_eq!(ns, [None, None, None, None]);
    }

    #[test]
    fn pair_neighbours_interior_has_all_four() {
        // PicW=4, row_pair=1, col=2 → pair_idx = 6 → mb_addr top = 12.
        let ns = mbaff_pair_neighbour_addrs(12, 4);
        // A = 2*(6-1) = 10 ; B = 2*(6-4) = 4 ; C = 2*(6-4+1) = 6 ;
        // D = 2*(6-4-1) = 2.
        assert_eq!(ns, [Some(10), Some(4), Some(6), Some(2)]);
    }

    #[test]
    fn pair_neighbours_same_for_top_and_bottom_of_pair() {
        // §6.4.10: "mbAddrA..D have identical values regardless whether
        // the current macroblock is the top or bottom macroblock of the
        // pair".
        let top = mbaff_pair_neighbour_addrs(12, 4);
        let bot = mbaff_pair_neighbour_addrs(13, 4);
        assert_eq!(top, bot);
    }

    #[test]
    fn pair_neighbours_right_edge_no_c() {
        // PicW=4, pair at col=3 → pair_idx = 7 (row 1) → mb_addr top=14.
        // A = 2*(7-1) = 12 ; B = 2*(7-4) = 6 ;
        // C invalid (col+1 == PicW) ;
        // D = 2*(7-4-1) = 4.
        let ns = mbaff_pair_neighbour_addrs(14, 4);
        assert_eq!(ns, [Some(12), Some(6), None, Some(4)]);
    }

    #[test]
    fn pair_neighbours_left_edge_no_a_d() {
        // PicW=4, pair at col=0 → pair_idx = 4 → mb_addr top=8.
        // A invalid; B = 2*(4-4) = 0 ; C = 2*(4-4+1) = 2 ; D invalid.
        let ns = mbaff_pair_neighbour_addrs(8, 4);
        assert_eq!(ns, [None, Some(0), Some(2), None]);
    }

    #[test]
    fn neighbour_addrs_uniform_field_pair_at_interior() {
        // PicW=4, flags = all frame. MB at addr=12 (top of pair 6,
        // row 1 col 2). Expect A→10 (same top of left pair),
        // B→5 (bottom of above pair 2), C→7 (bottom of above-right),
        // D→3 (bottom of above-left).
        let flags = vec![false; 16];
        let n = mbaff_neighbour_addrs(12, 4, &flags);
        assert_eq!(n, [Some(10), Some(5), Some(7), Some(3)]);
    }

    #[test]
    fn neighbour_addrs_top_left_corner_all_none() {
        let flags = vec![false; 16];
        let n = mbaff_neighbour_addrs(0, 4, &flags);
        assert_eq!(n, [None, None, None, None]);
    }

    #[test]
    fn neighbour_addrs_bottom_within_same_pair() {
        // Bottom MB of an interior pair. B must be the top MB of the
        // same pair (curr - 1). A must be left pair's bottom (same
        // within-pair role).
        let flags = vec![false; 16];
        let n = mbaff_neighbour_addrs(13, 4, &flags);
        // A: pair A = 10. curr is bottom, same flags → 10+1 = 11.
        // B: curr bottom → within-pair top = 12.
        // C: currently returns None for bottom (conservative).
        // D: currently returns pair A top (10).
        assert_eq!(n, [Some(11), Some(12), None, Some(10)]);
    }

    #[test]
    fn neighbour_addrs_left_edge_has_only_above() {
        // PicW=4, row 1 col 0 → mb_addr 8 top. A/D should be None.
        // B = bottom of pair above = addr 1. C = top of above-right
        // pair (flag-dependent); uniform-frame, current top → C is
        // bottom of above-right pair = addr 3.
        let flags = vec![false; 16];
        let n = mbaff_neighbour_addrs(8, 4, &flags);
        assert_eq!(n, [None, Some(1), Some(3), None]);
    }
}
