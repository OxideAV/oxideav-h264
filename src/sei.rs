//! Annex D — SEI payload parsing.
//!
//! Wraps the envelope output from `crate::non_vcl::parse_sei_rbsp`,
//! parsing the raw payload bytes into typed structs for the common
//! payload types. Each parser consumes exactly `payload_size` bytes.
//!
//! Spec cross-reference (Rec. ITU-T H.264 (08/2024)):
//!
//! | payload_type | Syntax      | Semantics   | Meaning                            |
//! | ------------ | ----------- | ----------- | ---------------------------------- |
//! | 0            | §D.1.2      | §D.2.2      | buffering_period                   |
//! | 1            | §D.1.3      | §D.2.3      | pic_timing                         |
//! | 2            | §D.1.4      | §D.2.4      | pan_scan_rect                      |
//! | 3            | §D.1.5      | §D.2.5      | filler_payload                     |
//! | 4            | §D.1.6      | §D.2.6      | user_data_registered_itu_t_t35     |
//! | 5            | §D.1.7      | §D.2.7      | user_data_unregistered             |
//! | 6            | §D.1.8      | §D.2.8      | recovery_point                     |
//! | 19           | §D.1.21     | §D.2.21     | film_grain_characteristics         |
//! | 22           | §D.1.24     | §D.2.24     | post_filter_hint                   |
//! | 23           | §D.1.25     | §D.2.25     | tone_mapping_info                  |
//! | 45           | §D.1.26     | §D.2.26     | frame_packing_arrangement          |
//! | 47           | §D.1.27     | §D.2.27     | display_orientation                |
//! | 137          | §D.1.29     | §D.2.29     | mastering_display_colour_volume    |
//! | 144          | §D.1.31     | §D.2.31     | content_light_level_info           |
//! | 147          | §D.1.32     | §D.2.32     | alternative_transfer_characteristics |

#![allow(dead_code)]

use crate::bitstream::{BitError, BitReader};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SeiError {
    #[error("bitstream read failed: {0}")]
    Bitstream(#[from] BitError),
    #[error("SEI payload type {0} is not implemented — caller should keep the raw bytes")]
    UnsupportedPayloadType(u32),
    #[error("user_data_unregistered payload must be at least 16 bytes for the UUID (got {0})")]
    UserDataTooShort(usize),
    #[error("filler_payload contains a byte that is not 0xFF at position {0}")]
    FillerNonFfAt(usize),
    #[error("user_data_registered_itu_t_t35 payload must contain at least the country code byte")]
    RegisteredUserDataMissingCountryCode,
    #[error("pan_scan_rect_cnt_minus1 must be < 32 per §D.2.4 (got {0})")]
    PanScanCountTooLarge(u32),
    #[error(
        "film_grain num_model_values_minus1 must be in 0..=5 per §D.2.21 (got {got} for component {component})"
    )]
    FilmGrainNumModelValuesOutOfRange { component: usize, got: u8 },
    #[error(
        "post_filter_hint filter_hint_size_x / filter_hint_size_y must be in 1..=15 per §D.2.24 (got x={x}, y={y})"
    )]
    PostFilterHintSizeOutOfRange { x: u32, y: u32 },
    #[error("post_filter_hint filter_hint_type 3 is reserved per §D.2.24")]
    PostFilterHintTypeReserved,
    #[error(
        "post_filter_hint filter_hint_type 1 (two 1D FIRs) requires filter_hint_size_y == 2 per §D.2.24 (got {0})"
    )]
    PostFilterHintType1WrongHeight(u32),
}

/// §D.2.2 — buffering_period.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferingPeriod {
    pub seq_parameter_set_id: u32,
    pub nal_hrd: Option<Vec<CpbRemovalDelay>>, // per SchedSelIdx
    pub vcl_hrd: Option<Vec<CpbRemovalDelay>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpbRemovalDelay {
    pub initial_cpb_removal_delay: u32,
    pub initial_cpb_removal_delay_offset: u32,
}

/// §D.2.3 — pic_timing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PicTiming {
    pub cpb_removal_delay: u32,
    pub dpb_output_delay: u32,
    pub pic_struct: Option<PicStructInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PicStructInfo {
    pub pic_struct: u8, // 0..=8
    pub clock_timestamps: Vec<Option<ClockTimestamp>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClockTimestamp {
    pub ct_type: u8,
    pub nuit_field_based_flag: bool,
    pub counting_type: u8,
    pub full_timestamp_flag: bool,
    pub discontinuity_flag: bool,
    pub cnt_dropped_flag: bool,
    pub n_frames: u8,
    pub seconds: u8,
    pub minutes: u8,
    pub hours: u8,
    pub time_offset: i32,
}

/// §D.2.7 — user_data_unregistered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserDataUnregistered {
    pub uuid: [u8; 16],
    pub user_data: Vec<u8>,
}

/// §D.2.6 — user_data_registered_itu_t_t35.
///
/// The single most common SEI in real-world H.264 streams: ATSC A/53
/// closed captions, AFD, ATSC bar data, HDR10+ dynamic metadata, and
/// Dolby Vision RPU all ride on this payload, distinguished by their
/// `(country_code, country_code_extension)` pair plus the leading bytes
/// of `payload_bytes`. We surface the raw triple — the per-vendor
/// payload format is out of scope for this layer (callers can switch
/// on `country_code` / `country_code_extension` and parse `payload_bytes`
/// against the relevant T.35 terminal-provider spec).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserDataRegisteredItuTT35 {
    /// itu_t_t35_country_code (u(8)). 0xB5 = USA, 0x26 = China, etc.
    pub country_code: u8,
    /// itu_t_t35_country_code_extension_byte (u(8)). Only present when
    /// `country_code == 0xFF` (per §D.1.6); `None` otherwise.
    pub country_code_extension: Option<u8>,
    /// Remaining `itu_t_t35_payload_byte`s after the country code(s).
    pub payload_bytes: Vec<u8>,
}

/// §D.2.4 — pan_scan_rect.
///
/// Carries one or more rectangles signalling the producer's preferred
/// display crop (commonly used to letterbox/pillarbox 4:3 content
/// inside a 16:9 frame, or vice-versa). Each rectangle's offsets are
/// expressed in units of 1/16 sample (per §D.2.4) relative to the
/// coded picture rectangle, and a positive offset shrinks the rectangle
/// from that edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanScanRect {
    pub pan_scan_rect_id: u32,
    pub cancel_flag: bool,
    /// Present only when `cancel_flag == false`.
    pub body: Option<PanScanRectBody>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanScanRectBody {
    /// One entry per rectangle (length = pan_scan_cnt_minus1 + 1).
    pub rects: Vec<PanScanRectOffsets>,
    pub repetition_period: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PanScanRectOffsets {
    pub left_offset: i32,
    pub right_offset: i32,
    pub top_offset: i32,
    pub bottom_offset: i32,
}

/// §D.2.8 — recovery_point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoveryPoint {
    pub recovery_frame_cnt: u32,
    pub exact_match_flag: bool,
    pub broken_link_flag: bool,
    pub changing_slice_group_idc: u8,
}

/// §D.2.29 — mastering_display_colour_volume (HDR10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MasteringDisplayColourVolume {
    /// 3 primaries, each (x, y) in 0..=50000 range.
    pub display_primaries: [(u16, u16); 3],
    pub white_point: (u16, u16),
    pub max_display_mastering_luminance: u32,
    pub min_display_mastering_luminance: u32,
}

/// §D.2.31 — content_light_level_info (HDR10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContentLightLevelInfo {
    pub max_content_light_level: u16,
    pub max_pic_average_light_level: u16,
}

/// §D.2.26 — frame_packing_arrangement. Stereoscopic 3D packing info.
///
/// When `cancel_flag` is true the message only carries the
/// identifier plus the trailing `extension_flag`; all other fields
/// default to zero and should be ignored by consumers.
///
/// When `quincunx_sampling_flag == 1` or `ty == 5` (temporal
/// interleaving) the four grid-position nibbles are not transmitted
/// and remain zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FramePackingArrangement {
    pub id: u32,
    pub cancel_flag: bool,
    pub ty: u8, // 0..=7 packing arrangement type (u(7))
    pub quincunx_sampling_flag: bool,
    pub content_interpretation_type: u8,
    pub spatial_flipping_flag: bool,
    pub frame0_flipped_flag: bool,
    pub field_views_flag: bool,
    pub current_frame_is_frame0_flag: bool,
    pub frame0_self_contained_flag: bool,
    pub frame1_self_contained_flag: bool,
    pub frame0_grid_position_x: u8,
    pub frame0_grid_position_y: u8,
    pub frame1_grid_position_x: u8,
    pub frame1_grid_position_y: u8,
    pub reserved_byte: u8,
    pub repetition_period: u32,
    pub extension_flag: bool,
}

/// §D.2.27 — display_orientation.
///
/// When `cancel_flag` is true all other fields are omitted from the
/// bitstream and default to zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayOrientation {
    pub cancel_flag: bool,
    pub hor_flip: bool,
    pub ver_flip: bool,
    /// u(16). Interpreted as counter-clockwise rotation in
    /// units of 360 / 2^16 degrees (per §D.2.27).
    pub anticlockwise_rotation: u16,
    pub repetition_period: u32,
    pub extension_flag: bool,
}

/// §D.2.24 — post_filter_hint.
///
/// Carries either FIR filter coefficients (1-D pair or 2-D) or a
/// cross-correlation matrix between the original and the decoded
/// signal, intended as a post-processing step after the picture is
/// decoded, cropped, and output. One coefficient set per colour
/// component (Y / Cb / Cr).
///
/// Per §D.2.24 the constraints policed here are:
/// * `filter_hint_size_x`, `filter_hint_size_y` shall be in 1..=15;
/// * `filter_hint_type` shall be in 0..=2 (3 is reserved);
/// * `filter_hint_type == 1` (two 1-D FIRs) implies
///   `filter_hint_size_y == 2`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostFilterHint {
    pub filter_hint_size_x: u32,
    pub filter_hint_size_y: u32,
    pub filter_hint_type: u8,
    /// `filter_hint_value[cIdx][cy][cx]` flattened in spec order:
    /// outer = component (0..3), middle = `cy` (0..size_y), inner =
    /// `cx` (0..size_x). Length = `3 * size_y * size_x`.
    pub filter_hint_value: Vec<i32>,
    pub additional_extension_flag: bool,
}

/// §D.2.25 — tone_mapping_info. HDR→SDR mapping parameters.
///
/// When `cancel_flag` is true no further fields are transmitted
/// (the body is absent). Otherwise the `model` enum captures the
/// model-specific payload keyed by `tone_map_model_id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToneMappingInfo {
    pub tone_map_id: u32,
    pub cancel_flag: bool,
    pub body: Option<ToneMappingBody>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToneMappingBody {
    pub repetition_period: u32,
    pub coded_data_bit_depth: u8,
    pub target_bit_depth: u8,
    pub model_id: u32,
    pub model: ToneMappingModel,
}

/// §D.2.25 — per-`tone_map_model_id` payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToneMappingModel {
    /// model_id = 0 — linear mapping in `[min_value, max_value]`.
    Linear { min_value: u32, max_value: u32 },
    /// model_id = 1 — sigmoid mapping.
    Sigmoid {
        sigmoid_midpoint: u32,
        sigmoid_width: u32,
    },
    /// model_id = 2 — piecewise mapping via interval start values.
    ///
    /// Per §D.2.25 each entry is
    /// `((coded_data_bit_depth + 7) >> 3) << 3` bits wide and the
    /// array has `1 << target_bit_depth` entries.
    StartOfCodedInterval { start_of_coded_interval: Vec<u32> },
    /// model_id = 3 — piecewise linear mapping via pivot pairs.
    ///
    /// Pivot values use
    /// `((coded_data_bit_depth + 7) >> 3) << 3` bits for
    /// `coded_pivot_value` and
    /// `((target_bit_depth + 7) >> 3) << 3` bits for
    /// `target_pivot_value`.
    PiecewisePivots {
        num_pivots: u16,
        coded_pivot_value: Vec<u32>,
        target_pivot_value: Vec<u32>,
    },
    /// Reserved for future use (model_id >= 4 — e.g. model_id 4 is
    /// the ISO-speed / exposure parameter set which this first pass
    /// does not decode).
    Reserved {
        model_id: u32,
        remaining_bytes: Vec<u8>,
    },
}

/// §D.2.21 — film_grain_characteristics.
///
/// Fully decodes the §D.1.21 / §D.2.21 syntax including the per-
/// component intensity-interval body (model_id 0 — frequency filtering;
/// model_id 1 — auto-regression). For each component flagged in
/// `comp_model_present_flag`, the resulting `FilmGrainComponent` carries
/// the `num_model_values_minus1`, the per-interval `(lower_bound,
/// upper_bound)` plus the `comp_model_value[i][j]` matrix as signed
/// integers, and the trailing
/// `film_grain_characteristics_repetition_period` is exposed on the
/// body for callers that need to know whether the message persists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilmGrainCharacteristics {
    pub cancel_flag: bool,
    pub body: Option<FilmGrainBody>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilmGrainBody {
    pub model_id: u8, // u(2)
    pub separate_colour_description_present_flag: bool,
    pub separate_colour_description: Option<FilmGrainSeparateColourDescription>,
    pub blending_mode_id: u8,  // u(2)
    pub log2_scale_factor: u8, // u(4)
    /// One flag per colour component (c = 0..=2). When the flag is set
    /// the corresponding entry in [`Self::components`] is `Some(_)`.
    pub comp_model_present_flag: [bool; 3],
    /// Per-component model parameters, in the same order as
    /// `comp_model_present_flag` (Y, Cb, Cr). An entry is `Some(_)`
    /// iff the matching present-flag is `true`.
    pub components: [Option<FilmGrainComponent>; 3],
    /// `film_grain_characteristics_repetition_period`. 0 = applies
    /// only to the current decoded picture; 1 = persists until the
    /// end of the coded video sequence (or until cancelled / replaced);
    /// values > 1 mean expect the next message within this many
    /// output-order pictures (see §D.2.21).
    pub repetition_period: u32,
}

/// §D.1.21 / §D.2.21 — per-component film-grain model parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilmGrainComponent {
    /// `num_model_values_minus1[c]`. The number of `comp_model_value`
    /// entries per intensity interval is this value plus one. Per
    /// §D.2.21 must be in 0..=5.
    pub num_model_values_minus1: u8,
    /// One entry per intensity interval (length =
    /// `num_intensity_intervals_minus1[c] + 1`).
    pub intensity_intervals: Vec<FilmGrainIntensityInterval>,
}

/// §D.1.21 — one intensity interval's bounds plus its model values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilmGrainIntensityInterval {
    pub lower_bound: u8,
    pub upper_bound: u8,
    /// `comp_model_value[c][i][j]` for j = 0..=num_model_values_minus1.
    pub model_values: Vec<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilmGrainSeparateColourDescription {
    pub bit_depth_luma_minus8: u8,   // u(3)
    pub bit_depth_chroma_minus8: u8, // u(3)
    pub full_range_flag: bool,
    pub colour_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
}

/// Inputs a SEI parser needs beyond the raw payload bytes. Many parsers
/// depend on VUI fields (CpbDpbDelaysPresentFlag, pic_struct_present_flag,
/// time_offset_length, etc.).
#[derive(Debug, Copy, Clone, Default)]
pub struct SeiContext {
    /// From VUI.nal_hrd: initial_cpb_removal_delay_length_minus1 and
    /// cpb_removal_delay_length_minus1, needed by buffering_period and
    /// pic_timing.
    pub initial_cpb_removal_delay_length_minus1: u8,
    pub cpb_removal_delay_length_minus1: u8,
    pub dpb_output_delay_length_minus1: u8,
    pub time_offset_length: u8,
    /// From VUI: cpb_cnt_minus1 of NAL/VCL HRD blocks (if present).
    pub nal_hrd_cpb_cnt_minus1: Option<u32>,
    pub vcl_hrd_cpb_cnt_minus1: Option<u32>,
    /// From VUI.pic_struct_present_flag.
    pub pic_struct_present_flag: bool,
    /// CpbDpbDelaysPresentFlag: derived from VUI (NAL or VCL HRD present).
    pub cpb_dpb_delays_present_flag: bool,
}

/// §D.2.2 — buffering_period.
///
/// Syntax (§D.1.2):
/// ```text
/// buffering_period( payloadSize ) {
///   seq_parameter_set_id                                  ue(v)
///   if( NalHrdBpPresentFlag )
///     for( SchedSelIdx = 0; SchedSelIdx <= cpb_cnt_minus1; SchedSelIdx++ ) {
///       initial_cpb_removal_delay[ SchedSelIdx ]          u(v)
///       initial_cpb_removal_delay_offset[ SchedSelIdx ]   u(v)
///     }
///   if( VclHrdBpPresentFlag )
///     for( SchedSelIdx = 0; SchedSelIdx <= cpb_cnt_minus1; SchedSelIdx++ ) {
///       initial_cpb_removal_delay[ SchedSelIdx ]          u(v)
///       initial_cpb_removal_delay_offset[ SchedSelIdx ]   u(v)
///     }
/// }
/// ```
///
/// The length in bits of the u(v) elements is
/// `initial_cpb_removal_delay_length_minus1 + 1` (§D.2.2).
pub fn parse_buffering_period(
    payload: &[u8],
    ctx: &SeiContext,
) -> Result<BufferingPeriod, SeiError> {
    let mut r = BitReader::new(payload);
    let seq_parameter_set_id = r.ue()?;
    let delay_bits = ctx.initial_cpb_removal_delay_length_minus1 as u32 + 1;

    let nal_hrd = if let Some(cnt_minus1) = ctx.nal_hrd_cpb_cnt_minus1 {
        let count = (cnt_minus1 as usize).saturating_add(1);
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let d = r.u(delay_bits)?;
            let o = r.u(delay_bits)?;
            out.push(CpbRemovalDelay {
                initial_cpb_removal_delay: d,
                initial_cpb_removal_delay_offset: o,
            });
        }
        Some(out)
    } else {
        None
    };

    let vcl_hrd = if let Some(cnt_minus1) = ctx.vcl_hrd_cpb_cnt_minus1 {
        let count = (cnt_minus1 as usize).saturating_add(1);
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let d = r.u(delay_bits)?;
            let o = r.u(delay_bits)?;
            out.push(CpbRemovalDelay {
                initial_cpb_removal_delay: d,
                initial_cpb_removal_delay_offset: o,
            });
        }
        Some(out)
    } else {
        None
    };

    Ok(BufferingPeriod {
        seq_parameter_set_id,
        nal_hrd,
        vcl_hrd,
    })
}

/// §D.2.3 — pic_timing.
///
/// Syntax (§D.1.3). Depends on `CpbDpbDelaysPresentFlag` and
/// `pic_struct_present_flag` from VUI. When `pic_struct_present_flag`
/// is set, the number of clockTimestamp slots `NumClockTS` is
/// determined by Table D-1 (§D.2.3).
///
/// NumClockTS per pic_struct:
/// | pic_struct | meaning                               | NumClockTS |
/// | ---------- | ------------------------------------- | ---------- |
/// | 0          | progressive frame                     | 1          |
/// | 1          | top field                             | 1          |
/// | 2          | bottom field                          | 1          |
/// | 3          | top+bottom                            | 2          |
/// | 4          | bottom+top                            | 2          |
/// | 5          | top+bottom+top                        | 3          |
/// | 6          | bottom+top+bottom                     | 3          |
/// | 7          | frame doubling                        | 2          |
/// | 8          | frame tripling                        | 3          |
pub fn parse_pic_timing(payload: &[u8], ctx: &SeiContext) -> Result<PicTiming, SeiError> {
    let mut r = BitReader::new(payload);

    let (cpb_removal_delay, dpb_output_delay) = if ctx.cpb_dpb_delays_present_flag {
        let cpb_bits = ctx.cpb_removal_delay_length_minus1 as u32 + 1;
        let dpb_bits = ctx.dpb_output_delay_length_minus1 as u32 + 1;
        (r.u(cpb_bits)?, r.u(dpb_bits)?)
    } else {
        (0, 0)
    };

    let pic_struct_info = if ctx.pic_struct_present_flag {
        // §D.1.3 — pic_struct u(4), then NumClockTS clockTimestamps.
        let pic_struct_val = r.u(4)? as u8;
        let num_clock_ts = num_clock_ts_from_pic_struct(pic_struct_val);

        let mut ts_slots: Vec<Option<ClockTimestamp>> = Vec::with_capacity(num_clock_ts as usize);
        for _ in 0..num_clock_ts {
            let clock_timestamp_flag = r.u(1)? == 1;
            if !clock_timestamp_flag {
                ts_slots.push(None);
                continue;
            }
            // §D.1.3 clockTimestamp inner body + §D.2.3 semantics.
            let ct_type = r.u(2)? as u8;
            let nuit_field_based_flag = r.u(1)? == 1;
            let counting_type = r.u(5)? as u8;
            let full_timestamp_flag = r.u(1)? == 1;
            let discontinuity_flag = r.u(1)? == 1;
            let cnt_dropped_flag = r.u(1)? == 1;
            let n_frames = r.u(8)? as u8;

            let (seconds, minutes, hours) = if full_timestamp_flag {
                let s = r.u(6)? as u8;
                let m = r.u(6)? as u8;
                let h = r.u(5)? as u8;
                (s, m, h)
            } else {
                let seconds_flag = r.u(1)? == 1;
                let (mut s, mut m, mut h) = (0u8, 0u8, 0u8);
                if seconds_flag {
                    s = r.u(6)? as u8;
                    let minutes_flag = r.u(1)? == 1;
                    if minutes_flag {
                        m = r.u(6)? as u8;
                        let hours_flag = r.u(1)? == 1;
                        if hours_flag {
                            h = r.u(5)? as u8;
                        }
                    }
                }
                (s, m, h)
            };

            let time_offset = if ctx.time_offset_length > 0 {
                r.i(ctx.time_offset_length as u32)?
            } else {
                0
            };

            ts_slots.push(Some(ClockTimestamp {
                ct_type,
                nuit_field_based_flag,
                counting_type,
                full_timestamp_flag,
                discontinuity_flag,
                cnt_dropped_flag,
                n_frames,
                seconds,
                minutes,
                hours,
                time_offset,
            }));
        }

        Some(PicStructInfo {
            pic_struct: pic_struct_val,
            clock_timestamps: ts_slots,
        })
    } else {
        None
    };

    Ok(PicTiming {
        cpb_removal_delay,
        dpb_output_delay,
        pic_struct: pic_struct_info,
    })
}

/// Number of clockTimestamp entries per pic_struct, per §D.2.3 Table D-1.
/// For reserved values (9..=15) the table is unspecified; returning 0
/// is the safest choice: no clockTimestamp flags are read.
fn num_clock_ts_from_pic_struct(pic_struct: u8) -> u8 {
    match pic_struct {
        0..=2 => 1,
        3 | 4 | 7 => 2,
        5 | 6 | 8 => 3,
        _ => 0,
    }
}

/// §D.2.5 — filler_payload.
///
/// Syntax (§D.1.5): `payloadSize` occurrences of `ff_byte` equal to
/// `0xFF`. This parser verifies every byte matches and returns `()`;
/// no state is kept because the payload carries no signal.
pub fn parse_filler_payload(payload: &[u8]) -> Result<(), SeiError> {
    for (i, &b) in payload.iter().enumerate() {
        if b != 0xFF {
            return Err(SeiError::FillerNonFfAt(i));
        }
    }
    Ok(())
}

/// §D.2.6 — user_data_registered_itu_t_t35.
///
/// Syntax (§D.1.6):
/// ```text
/// user_data_registered_itu_t_t35( payloadSize ) {
///   itu_t_t35_country_code                          b(8)
///   if( itu_t_t35_country_code == 0xFF )
///     itu_t_t35_country_code_extension_byte         b(8)
///   for( i = ...; i < payloadSize; i++ )
///     itu_t_t35_payload_byte                        b(8)
/// }
/// ```
///
/// All fields are byte-aligned per §D.1.6, so we read straight from
/// the slice without going through the bit-reader. The extension byte
/// is consumed only when `country_code == 0xFF` per Recommendation
/// ITU-T T.35 §3.1.
pub fn parse_user_data_registered_itu_t_t35(
    payload: &[u8],
) -> Result<UserDataRegisteredItuTT35, SeiError> {
    let Some((&country_code, rest)) = payload.split_first() else {
        return Err(SeiError::RegisteredUserDataMissingCountryCode);
    };
    let (country_code_extension, payload_bytes) = if country_code == 0xFF {
        match rest.split_first() {
            Some((&ext, body)) => (Some(ext), body.to_vec()),
            // §D.1.6 / T.35 §3.1: when country_code is the escape value
            // 0xFF, the extension byte is required. A truncated payload
            // here is malformed.
            None => return Err(SeiError::RegisteredUserDataMissingCountryCode),
        }
    } else {
        (None, rest.to_vec())
    };
    Ok(UserDataRegisteredItuTT35 {
        country_code,
        country_code_extension,
        payload_bytes,
    })
}

/// §D.2.4 — pan_scan_rect.
///
/// Syntax (§D.1.4):
/// ```text
/// pan_scan_rect( payloadSize ) {
///   pan_scan_rect_id                                ue(v)
///   pan_scan_rect_cancel_flag                       u(1)
///   if( !pan_scan_rect_cancel_flag ) {
///     pan_scan_cnt_minus1                           ue(v)
///     for( i = 0; i <= pan_scan_cnt_minus1; i++ ) {
///       pan_scan_rect_left_offset[ i ]              se(v)
///       pan_scan_rect_right_offset[ i ]             se(v)
///       pan_scan_rect_top_offset[ i ]               se(v)
///       pan_scan_rect_bottom_offset[ i ]            se(v)
///     }
///     pan_scan_rect_repetition_period               ue(v)
///   }
/// }
/// ```
///
/// Per §D.2.4 `pan_scan_cnt_minus1` is constrained to `0..=2` (so the
/// payload carries at most three rectangles); we accept up to 31 to
/// stay forgiving of streams that exceed the constraint, but reject
/// counts that are obviously out of range to avoid pathological
/// allocations.
pub fn parse_pan_scan_rect(payload: &[u8]) -> Result<PanScanRect, SeiError> {
    let mut r = BitReader::new(payload);
    let pan_scan_rect_id = r.ue()?;
    let cancel_flag = r.u(1)? == 1;
    if cancel_flag {
        return Ok(PanScanRect {
            pan_scan_rect_id,
            cancel_flag,
            body: None,
        });
    }
    let cnt_minus1 = r.ue()?;
    if cnt_minus1 >= 32 {
        return Err(SeiError::PanScanCountTooLarge(cnt_minus1));
    }
    let count = (cnt_minus1 as usize).saturating_add(1);
    let mut rects = Vec::with_capacity(count);
    for _ in 0..count {
        let left = r.se()?;
        let right = r.se()?;
        let top = r.se()?;
        let bottom = r.se()?;
        rects.push(PanScanRectOffsets {
            left_offset: left,
            right_offset: right,
            top_offset: top,
            bottom_offset: bottom,
        });
    }
    let repetition_period = r.ue()?;
    Ok(PanScanRect {
        pan_scan_rect_id,
        cancel_flag,
        body: Some(PanScanRectBody {
            rects,
            repetition_period,
        }),
    })
}

/// §D.2.7 — user_data_unregistered.
///
/// Syntax (§D.1.7): `uuid_iso_iec_11578` (u(128)) followed by
/// `payloadSize - 16` bytes of `user_data_payload_byte`.
pub fn parse_user_data_unregistered(payload: &[u8]) -> Result<UserDataUnregistered, SeiError> {
    if payload.len() < 16 {
        return Err(SeiError::UserDataTooShort(payload.len()));
    }
    let mut uuid = [0u8; 16];
    uuid.copy_from_slice(&payload[..16]);
    let user_data = payload[16..].to_vec();
    Ok(UserDataUnregistered { uuid, user_data })
}

/// §D.2.8 — recovery_point.
///
/// Syntax (§D.1.8):
/// ```text
/// recovery_point( payloadSize ) {
///   recovery_frame_cnt       ue(v)
///   exact_match_flag          u(1)
///   broken_link_flag          u(1)
///   changing_slice_group_idc  u(2)
/// }
/// ```
pub fn parse_recovery_point(payload: &[u8]) -> Result<RecoveryPoint, SeiError> {
    let mut r = BitReader::new(payload);
    let recovery_frame_cnt = r.ue()?;
    let exact_match_flag = r.u(1)? == 1;
    let broken_link_flag = r.u(1)? == 1;
    let changing_slice_group_idc = r.u(2)? as u8;
    Ok(RecoveryPoint {
        recovery_frame_cnt,
        exact_match_flag,
        broken_link_flag,
        changing_slice_group_idc,
    })
}

/// §D.2.29 — mastering_display_colour_volume (HDR10 metadata).
///
/// Syntax (§D.1.29):
/// ```text
/// mastering_display_colour_volume( payloadSize ) {
///   for( c = 0; c < 3; c++ ) {
///     display_primaries_x[ c ]         u(16)
///     display_primaries_y[ c ]         u(16)
///   }
///   white_point_x                       u(16)
///   white_point_y                       u(16)
///   max_display_mastering_luminance     u(32)
///   min_display_mastering_luminance     u(32)
/// }
/// ```
///
/// Per §D.2.29 the primaries and white point are on a 0..=50000 scale
/// (CIE 1931 chromaticity * 50000); luminance is in units of 0.0001
/// candela per square metre.
pub fn parse_mastering_display(payload: &[u8]) -> Result<MasteringDisplayColourVolume, SeiError> {
    let mut r = BitReader::new(payload);
    let mut primaries = [(0u16, 0u16); 3];
    for slot in primaries.iter_mut() {
        let x = r.u(16)? as u16;
        let y = r.u(16)? as u16;
        *slot = (x, y);
    }
    let wp_x = r.u(16)? as u16;
    let wp_y = r.u(16)? as u16;
    let max_l = r.u(32)?;
    let min_l = r.u(32)?;
    Ok(MasteringDisplayColourVolume {
        display_primaries: primaries,
        white_point: (wp_x, wp_y),
        max_display_mastering_luminance: max_l,
        min_display_mastering_luminance: min_l,
    })
}

/// §D.2.31 — content_light_level_info (HDR10 metadata).
///
/// Syntax (§D.1.31):
/// ```text
/// content_light_level_info( payloadSize ) {
///   max_content_light_level       u(16)
///   max_pic_average_light_level   u(16)
/// }
/// ```
pub fn parse_content_light_level(payload: &[u8]) -> Result<ContentLightLevelInfo, SeiError> {
    let mut r = BitReader::new(payload);
    let max_content_light_level = r.u(16)? as u16;
    let max_pic_average_light_level = r.u(16)? as u16;
    Ok(ContentLightLevelInfo {
        max_content_light_level,
        max_pic_average_light_level,
    })
}

/// §D.2.26 — frame_packing_arrangement (payload type 45).
///
/// Syntax (§D.1.26):
/// ```text
/// frame_packing_arrangement( payloadSize ) {
///   frame_packing_arrangement_id                       ue(v)
///   frame_packing_arrangement_cancel_flag              u(1)
///   if( !frame_packing_arrangement_cancel_flag ) {
///     frame_packing_arrangement_type                   u(7)
///     quincunx_sampling_flag                           u(1)
///     content_interpretation_type                      u(6)
///     spatial_flipping_flag                            u(1)
///     frame0_flipped_flag                              u(1)
///     field_views_flag                                 u(1)
///     current_frame_is_frame0_flag                     u(1)
///     frame0_self_contained_flag                       u(1)
///     frame1_self_contained_flag                       u(1)
///     if( !quincunx_sampling_flag &&
///         frame_packing_arrangement_type != 5 ) {
///       frame0_grid_position_x                         u(4)
///       frame0_grid_position_y                         u(4)
///       frame1_grid_position_x                         u(4)
///       frame1_grid_position_y                         u(4)
///     }
///     frame_packing_arrangement_reserved_byte          u(8)
///     frame_packing_arrangement_repetition_period      ue(v)
///   }
///   frame_packing_arrangement_extension_flag           u(1)
/// }
/// ```
pub fn parse_frame_packing_arrangement(
    payload: &[u8],
) -> Result<FramePackingArrangement, SeiError> {
    let mut r = BitReader::new(payload);
    let id = r.ue()?;
    let cancel_flag = r.u(1)? == 1;

    let mut out = FramePackingArrangement {
        id,
        cancel_flag,
        ty: 0,
        quincunx_sampling_flag: false,
        content_interpretation_type: 0,
        spatial_flipping_flag: false,
        frame0_flipped_flag: false,
        field_views_flag: false,
        current_frame_is_frame0_flag: false,
        frame0_self_contained_flag: false,
        frame1_self_contained_flag: false,
        frame0_grid_position_x: 0,
        frame0_grid_position_y: 0,
        frame1_grid_position_x: 0,
        frame1_grid_position_y: 0,
        reserved_byte: 0,
        repetition_period: 0,
        extension_flag: false,
    };

    if !cancel_flag {
        out.ty = r.u(7)? as u8;
        out.quincunx_sampling_flag = r.u(1)? == 1;
        out.content_interpretation_type = r.u(6)? as u8;
        out.spatial_flipping_flag = r.u(1)? == 1;
        out.frame0_flipped_flag = r.u(1)? == 1;
        out.field_views_flag = r.u(1)? == 1;
        out.current_frame_is_frame0_flag = r.u(1)? == 1;
        out.frame0_self_contained_flag = r.u(1)? == 1;
        out.frame1_self_contained_flag = r.u(1)? == 1;

        // §D.1.26: the grid positions are transmitted only when the
        // planes are non-quincunx AND the arrangement is not temporal
        // interleaving (type 5).
        if !out.quincunx_sampling_flag && out.ty != 5 {
            out.frame0_grid_position_x = r.u(4)? as u8;
            out.frame0_grid_position_y = r.u(4)? as u8;
            out.frame1_grid_position_x = r.u(4)? as u8;
            out.frame1_grid_position_y = r.u(4)? as u8;
        }

        out.reserved_byte = r.u(8)? as u8;
        out.repetition_period = r.ue()?;
    }

    out.extension_flag = r.u(1)? == 1;
    Ok(out)
}

/// §D.2.27 — display_orientation (payload type 47).
///
/// Syntax (§D.1.27):
/// ```text
/// display_orientation( payloadSize ) {
///   display_orientation_cancel_flag               u(1)
///   if( !display_orientation_cancel_flag ) {
///     hor_flip                                    u(1)
///     ver_flip                                    u(1)
///     anticlockwise_rotation                      u(16)
///     display_orientation_repetition_period       ue(v)
///     display_orientation_extension_flag          u(1)
///   }
/// }
/// ```
pub fn parse_display_orientation(payload: &[u8]) -> Result<DisplayOrientation, SeiError> {
    let mut r = BitReader::new(payload);
    let cancel_flag = r.u(1)? == 1;
    let mut out = DisplayOrientation {
        cancel_flag,
        hor_flip: false,
        ver_flip: false,
        anticlockwise_rotation: 0,
        repetition_period: 0,
        extension_flag: false,
    };
    if !cancel_flag {
        out.hor_flip = r.u(1)? == 1;
        out.ver_flip = r.u(1)? == 1;
        out.anticlockwise_rotation = r.u(16)? as u16;
        out.repetition_period = r.ue()?;
        out.extension_flag = r.u(1)? == 1;
    }
    Ok(out)
}

/// §D.2.32 — alternative_transfer_characteristics (payload type 147).
///
/// Syntax (§D.1.32):
/// ```text
/// alternative_transfer_characteristics( payloadSize ) {
///   preferred_transfer_characteristics    u(8)
/// }
/// ```
///
/// Per §D.2.32 the single field overrides the VUI
/// transfer_characteristics for display purposes (for example to
/// signal HLG or PQ in a stream whose VUI says BT.709).
pub fn parse_alternative_transfer_characteristics(payload: &[u8]) -> Result<u8, SeiError> {
    let mut r = BitReader::new(payload);
    Ok(r.u(8)? as u8)
}

/// §D.2.24 — post_filter_hint (payload type 22).
///
/// Syntax (§D.1.24):
/// ```text
/// post_filter_hint( payloadSize ) {
///   filter_hint_size_y                                    ue(v)
///   filter_hint_size_x                                    ue(v)
///   filter_hint_type                                      u(2)
///   for( cIdx = 0; cIdx < 3; cIdx++ )
///     for( cy = 0; cy < filter_hint_size_y; cy++ )
///       for( cx = 0; cx < filter_hint_size_x; cx++ )
///         filter_hint_value[ cIdx ][ cx ][ cy ]           se(v)
///   additional_extension_flag                             u(1)
/// }
/// ```
///
/// Returns [`SeiError::PostFilterHintSizeOutOfRange`] if either size
/// falls outside the §D.2.24 1..=15 range,
/// [`SeiError::PostFilterHintTypeReserved`] if `filter_hint_type == 3`
/// (reserved by ITU-T | ISO/IEC), and
/// [`SeiError::PostFilterHintType1WrongHeight`] if
/// `filter_hint_type == 1` but `filter_hint_size_y != 2`.
pub fn parse_post_filter_hint(payload: &[u8]) -> Result<PostFilterHint, SeiError> {
    let mut r = BitReader::new(payload);
    let filter_hint_size_y = r.ue()?;
    let filter_hint_size_x = r.ue()?;
    if !(1..=15).contains(&filter_hint_size_y) || !(1..=15).contains(&filter_hint_size_x) {
        return Err(SeiError::PostFilterHintSizeOutOfRange {
            x: filter_hint_size_x,
            y: filter_hint_size_y,
        });
    }
    let filter_hint_type = r.u(2)? as u8;
    if filter_hint_type == 3 {
        return Err(SeiError::PostFilterHintTypeReserved);
    }
    if filter_hint_type == 1 && filter_hint_size_y != 2 {
        return Err(SeiError::PostFilterHintType1WrongHeight(filter_hint_size_y));
    }
    let total = 3usize * filter_hint_size_y as usize * filter_hint_size_x as usize;
    let mut filter_hint_value = Vec::with_capacity(total);
    for _ in 0..total {
        filter_hint_value.push(r.se()?);
    }
    let additional_extension_flag = r.u(1)? == 1;
    Ok(PostFilterHint {
        filter_hint_size_x,
        filter_hint_size_y,
        filter_hint_type,
        filter_hint_value,
        additional_extension_flag,
    })
}

/// §D.2.25 — tone_mapping_info (payload type 23).
///
/// Syntax (§D.1.25):
/// ```text
/// tone_mapping_info( payloadSize ) {
///   tone_map_id                                   ue(v)
///   tone_map_cancel_flag                          u(1)
///   if( !tone_map_cancel_flag ) {
///     tone_map_repetition_period                  ue(v)
///     coded_data_bit_depth                        u(8)
///     target_bit_depth                            u(8)
///     tone_map_model_id                           ue(v)
///     if( tone_map_model_id == 0 ) {
///       min_value                                 u(32)
///       max_value                                 u(32)
///     }
///     if( tone_map_model_id == 1 ) {
///       sigmoid_midpoint                          u(32)
///       sigmoid_width                             u(32)
///     }
///     if( tone_map_model_id == 2 )
///       for( i = 0; i < ( 1 << target_bit_depth ); i++ )
///         start_of_coded_interval[ i ]            u(v)
///     if( tone_map_model_id == 3 ) {
///       num_pivots                                u(16)
///       for( i = 0; i < num_pivots; i++ ) {
///         coded_pivot_value[ i ]                  u(v)
///         target_pivot_value[ i ]                 u(v)
///       }
///     }
///     if( tone_map_model_id == 4 ) { ... }
///   }
/// }
/// ```
///
/// Per §D.2.25:
/// - `start_of_coded_interval` bit width is
///   `((coded_data_bit_depth + 7) >> 3) << 3`.
/// - `coded_pivot_value` bit width is
///   `((coded_data_bit_depth + 7) >> 3) << 3`.
/// - `target_pivot_value` bit width is
///   `((target_bit_depth + 7) >> 3) << 3`.
///
/// This implementation decodes model_id 0..=3. model_id 4 (and any
/// future value) is captured as `ToneMappingModel::Reserved` with
/// the remaining payload bytes preserved so callers can keep it.
pub fn parse_tone_mapping_info(payload: &[u8]) -> Result<ToneMappingInfo, SeiError> {
    let mut r = BitReader::new(payload);
    let tone_map_id = r.ue()?;
    let cancel_flag = r.u(1)? == 1;
    if cancel_flag {
        return Ok(ToneMappingInfo {
            tone_map_id,
            cancel_flag,
            body: None,
        });
    }

    let repetition_period = r.ue()?;
    let coded_data_bit_depth = r.u(8)? as u8;
    let target_bit_depth = r.u(8)? as u8;
    let model_id = r.ue()?;

    let coded_width = ((coded_data_bit_depth as u32 + 7) >> 3) << 3;
    let target_width = ((target_bit_depth as u32 + 7) >> 3) << 3;

    let model = match model_id {
        0 => {
            let min_value = r.u(32)?;
            let max_value = r.u(32)?;
            ToneMappingModel::Linear {
                min_value,
                max_value,
            }
        }
        1 => {
            let sigmoid_midpoint = r.u(32)?;
            let sigmoid_width = r.u(32)?;
            ToneMappingModel::Sigmoid {
                sigmoid_midpoint,
                sigmoid_width,
            }
        }
        2 => {
            // Per §D.2.25 there are (1 << target_bit_depth) entries.
            // Guard against pathological values so we don't allocate
            // several GB on a corrupted stream.
            if target_bit_depth > 16 || coded_width == 0 {
                return Err(SeiError::UnsupportedPayloadType(23));
            }
            let count = 1usize << target_bit_depth as usize;
            let mut vals = Vec::with_capacity(count);
            for _ in 0..count {
                vals.push(r.u(coded_width)?);
            }
            ToneMappingModel::StartOfCodedInterval {
                start_of_coded_interval: vals,
            }
        }
        3 => {
            let num_pivots = r.u(16)? as u16;
            if coded_width == 0 || target_width == 0 {
                return Err(SeiError::UnsupportedPayloadType(23));
            }
            let mut coded_pivot_value = Vec::with_capacity(num_pivots as usize);
            let mut target_pivot_value = Vec::with_capacity(num_pivots as usize);
            for _ in 0..num_pivots {
                coded_pivot_value.push(r.u(coded_width)?);
                target_pivot_value.push(r.u(target_width)?);
            }
            ToneMappingModel::PiecewisePivots {
                num_pivots,
                coded_pivot_value,
                target_pivot_value,
            }
        }
        _ => {
            // model_id >= 4: preserve remaining bytes so callers can
            // keep the payload for later forwarding or custom parsing.
            let (byte_pos, bit_pos) = r.position();
            let remaining_bytes = if bit_pos == 0 {
                payload.get(byte_pos..).unwrap_or(&[]).to_vec()
            } else {
                // Skip to next byte boundary before copying; bits
                // within the partial byte are discarded in this
                // first pass.
                payload.get(byte_pos + 1..).unwrap_or(&[]).to_vec()
            };
            ToneMappingModel::Reserved {
                model_id,
                remaining_bytes,
            }
        }
    };

    Ok(ToneMappingInfo {
        tone_map_id,
        cancel_flag,
        body: Some(ToneMappingBody {
            repetition_period,
            coded_data_bit_depth,
            target_bit_depth,
            model_id,
            model,
        }),
    })
}

/// §D.2.21 — film_grain_characteristics (payload type 19).
///
/// Syntax (§D.1.21):
/// ```text
/// film_grain_characteristics( payloadSize ) {
///   film_grain_characteristics_cancel_flag                  u(1)
///   if( !film_grain_characteristics_cancel_flag ) {
///     film_grain_model_id                                   u(2)
///     separate_colour_description_present_flag              u(1)
///     if( separate_colour_description_present_flag ) {
///       film_grain_bit_depth_luma_minus8                    u(3)
///       film_grain_bit_depth_chroma_minus8                  u(3)
///       film_grain_full_range_flag                          u(1)
///       film_grain_colour_primaries                         u(8)
///       film_grain_transfer_characteristics                 u(8)
///       film_grain_matrix_coefficients                      u(8)
///     }
///     blending_mode_id                                      u(2)
///     log2_scale_factor                                     u(4)
///     for( c = 0; c < 3; c++ )
///       comp_model_present_flag[ c ]                        u(1)
///     for( c = 0; c < 3; c++ )
///       if( comp_model_present_flag[ c ] ) {
///         num_intensity_intervals_minus1[ c ]               u(8)
///         num_model_values_minus1[ c ]                      u(3)
///         for( i = 0; i <= num_intensity_intervals_minus1[ c ]; i++ ) {
///           intensity_interval_lower_bound[ c ][ i ]        u(8)
///           intensity_interval_upper_bound[ c ][ i ]        u(8)
///           for( j = 0; j <= num_model_values_minus1[ c ]; j++ )
///             comp_model_value[ c ][ i ][ j ]               se(v)
///         }
///       }
///     film_grain_characteristics_repetition_period          ue(v)
///   }
/// }
/// ```
///
/// Fully decodes the per-component intensity-interval body. Per
/// §D.2.21 `num_model_values_minus1[c]` must be in 0..=5; values
/// outside that range produce
/// [`SeiError::FilmGrainNumModelValuesOutOfRange`]. The
/// `num_intensity_intervals_minus1[c] + 1` interval count uses the
/// full u(8) range (0..=256 entries per component); pathological
/// streams therefore allocate at most ~1.5 KiB per component for the
/// interval-bounds list and 6 × 4 B × 256 ≈ 6 KiB per component for
/// `comp_model_value`, both bounded.
pub fn parse_film_grain_characteristics(
    payload: &[u8],
) -> Result<FilmGrainCharacteristics, SeiError> {
    let mut r = BitReader::new(payload);
    let cancel_flag = r.u(1)? == 1;
    if cancel_flag {
        return Ok(FilmGrainCharacteristics {
            cancel_flag,
            body: None,
        });
    }

    let model_id = r.u(2)? as u8;
    let separate_flag = r.u(1)? == 1;
    let separate_colour_description = if separate_flag {
        let bit_depth_luma_minus8 = r.u(3)? as u8;
        let bit_depth_chroma_minus8 = r.u(3)? as u8;
        let full_range_flag = r.u(1)? == 1;
        let colour_primaries = r.u(8)? as u8;
        let transfer_characteristics = r.u(8)? as u8;
        let matrix_coefficients = r.u(8)? as u8;
        Some(FilmGrainSeparateColourDescription {
            bit_depth_luma_minus8,
            bit_depth_chroma_minus8,
            full_range_flag,
            colour_primaries,
            transfer_characteristics,
            matrix_coefficients,
        })
    } else {
        None
    };

    let blending_mode_id = r.u(2)? as u8;
    let log2_scale_factor = r.u(4)? as u8;

    let mut comp_model_present_flag = [false; 3];
    for flag in comp_model_present_flag.iter_mut() {
        *flag = r.u(1)? == 1;
    }

    let mut components: [Option<FilmGrainComponent>; 3] = [None, None, None];
    for (c, present) in comp_model_present_flag.iter().enumerate() {
        if !*present {
            continue;
        }
        let num_intensity_intervals_minus1 = r.u(8)? as u8;
        let num_model_values_minus1 = r.u(3)? as u8;
        if num_model_values_minus1 > 5 {
            return Err(SeiError::FilmGrainNumModelValuesOutOfRange {
                component: c,
                got: num_model_values_minus1,
            });
        }
        let interval_count = num_intensity_intervals_minus1 as usize + 1;
        let mvalue_count = num_model_values_minus1 as usize + 1;
        let mut intensity_intervals = Vec::with_capacity(interval_count);
        for _ in 0..interval_count {
            let lower_bound = r.u(8)? as u8;
            let upper_bound = r.u(8)? as u8;
            let mut model_values = Vec::with_capacity(mvalue_count);
            for _ in 0..mvalue_count {
                model_values.push(r.se()?);
            }
            intensity_intervals.push(FilmGrainIntensityInterval {
                lower_bound,
                upper_bound,
                model_values,
            });
        }
        components[c] = Some(FilmGrainComponent {
            num_model_values_minus1,
            intensity_intervals,
        });
    }

    let repetition_period = r.ue()?;

    Ok(FilmGrainCharacteristics {
        cancel_flag,
        body: Some(FilmGrainBody {
            model_id,
            separate_colour_description_present_flag: separate_flag,
            separate_colour_description,
            blending_mode_id,
            log2_scale_factor,
            comp_model_present_flag,
            components,
            repetition_period,
        }),
    })
}

/// Dispatch helper: given a payload_type and bytes, produce the typed
/// payload if it's one we recognise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeiPayload {
    BufferingPeriod(BufferingPeriod),
    PicTiming(PicTiming),
    FillerPayload,
    UserDataRegisteredItuTT35(UserDataRegisteredItuTT35),
    UserDataUnregistered(UserDataUnregistered),
    PanScanRect(PanScanRect),
    RecoveryPoint(RecoveryPoint),
    FilmGrainCharacteristics(FilmGrainCharacteristics),
    PostFilterHint(PostFilterHint),
    ToneMappingInfo(ToneMappingInfo),
    FramePackingArrangement(FramePackingArrangement),
    DisplayOrientation(DisplayOrientation),
    MasteringDisplay(MasteringDisplayColourVolume),
    ContentLightLevel(ContentLightLevelInfo),
    /// preferred_transfer_characteristics — single u(8) field.
    AlternativeTransferCharacteristics(u8),
    /// Not parsed — caller keeps the raw payload for later interpretation.
    Unknown {
        payload_type: u32,
        payload: Vec<u8>,
    },
}

/// §D.1.1 sei_payload() dispatch for the subset of payload types we
/// parse. Unknown types are returned as `SeiPayload::Unknown` with
/// their raw bytes preserved.
pub fn parse_payload(
    payload_type: u32,
    payload: &[u8],
    ctx: &SeiContext,
) -> Result<SeiPayload, SeiError> {
    match payload_type {
        0 => Ok(SeiPayload::BufferingPeriod(parse_buffering_period(
            payload, ctx,
        )?)),
        1 => Ok(SeiPayload::PicTiming(parse_pic_timing(payload, ctx)?)),
        2 => Ok(SeiPayload::PanScanRect(parse_pan_scan_rect(payload)?)),
        3 => {
            parse_filler_payload(payload)?;
            Ok(SeiPayload::FillerPayload)
        }
        4 => Ok(SeiPayload::UserDataRegisteredItuTT35(
            parse_user_data_registered_itu_t_t35(payload)?,
        )),
        5 => Ok(SeiPayload::UserDataUnregistered(
            parse_user_data_unregistered(payload)?,
        )),
        6 => Ok(SeiPayload::RecoveryPoint(parse_recovery_point(payload)?)),
        19 => Ok(SeiPayload::FilmGrainCharacteristics(
            parse_film_grain_characteristics(payload)?,
        )),
        22 => Ok(SeiPayload::PostFilterHint(parse_post_filter_hint(payload)?)),
        23 => Ok(SeiPayload::ToneMappingInfo(parse_tone_mapping_info(
            payload,
        )?)),
        45 => Ok(SeiPayload::FramePackingArrangement(
            parse_frame_packing_arrangement(payload)?,
        )),
        47 => Ok(SeiPayload::DisplayOrientation(parse_display_orientation(
            payload,
        )?)),
        137 => Ok(SeiPayload::MasteringDisplay(parse_mastering_display(
            payload,
        )?)),
        144 => Ok(SeiPayload::ContentLightLevel(parse_content_light_level(
            payload,
        )?)),
        147 => Ok(SeiPayload::AlternativeTransferCharacteristics(
            parse_alternative_transfer_characteristics(payload)?,
        )),
        other => Ok(SeiPayload::Unknown {
            payload_type: other,
            payload: payload.to_vec(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: pack a sequence of (value, bit_width) fields MSB-first
    // into a byte vector, zero-padding the final byte.
    fn pack_bits(fields: &[(u64, u32)]) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();
        for &(value, width) in fields {
            for i in (0..width).rev() {
                bits.push(((value >> i) & 1) as u8);
            }
        }
        // Pad to a whole byte boundary.
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        bits.chunks(8)
            .map(|c| c.iter().fold(0u8, |acc, &b| (acc << 1) | b))
            .collect()
    }

    #[test]
    fn buffering_period_nal_only_single_sched() {
        // ue(0) → bit "1" encodes seq_parameter_set_id = 0.
        // initial_cpb_removal_delay_length_minus1 = 23 → 24-bit fields.
        // Pick delay = 0x012345, offset = 0x0ABCDE.
        // Bit layout: "1" (ue=0), then 24 bits delay, then 24 bits offset
        // → 1 + 24 + 24 = 49 bits, 7 bytes after padding.
        let fields = [(1, 1), (0x012345, 24), (0x0ABCDE, 24)];
        let payload = pack_bits(&fields);

        let ctx = SeiContext {
            initial_cpb_removal_delay_length_minus1: 23,
            nal_hrd_cpb_cnt_minus1: Some(0), // one SchedSelIdx
            ..Default::default()
        };

        let bp = parse_buffering_period(&payload, &ctx).unwrap();
        assert_eq!(bp.seq_parameter_set_id, 0);
        assert!(bp.vcl_hrd.is_none());
        let nal = bp.nal_hrd.expect("nal hrd present");
        assert_eq!(nal.len(), 1);
        assert_eq!(nal[0].initial_cpb_removal_delay, 0x012345);
        assert_eq!(nal[0].initial_cpb_removal_delay_offset, 0x0ABCDE);
    }

    #[test]
    fn pic_timing_with_delays_and_pic_struct_zero() {
        // CpbDpbDelaysPresentFlag=true, lengths=24 each.
        // cpb_delay=5, dpb_delay=10, pic_struct=0 (1 clockTimestamp).
        // clock_timestamp_flag=0 → empty slot.
        // Fields: cpb(24)=5, dpb(24)=10, pic_struct(4)=0, ct_flag(1)=0 = 53 bits.
        let fields = [(5, 24), (10, 24), (0, 4), (0, 1)];
        let payload = pack_bits(&fields);

        let ctx = SeiContext {
            cpb_removal_delay_length_minus1: 23,
            dpb_output_delay_length_minus1: 23,
            cpb_dpb_delays_present_flag: true,
            pic_struct_present_flag: true,
            time_offset_length: 24,
            ..Default::default()
        };
        let pt = parse_pic_timing(&payload, &ctx).unwrap();
        assert_eq!(pt.cpb_removal_delay, 5);
        assert_eq!(pt.dpb_output_delay, 10);
        let info = pt.pic_struct.expect("pic_struct present");
        assert_eq!(info.pic_struct, 0);
        assert_eq!(info.clock_timestamps.len(), 1);
        assert!(info.clock_timestamps[0].is_none());
    }

    #[test]
    fn pic_timing_without_delays_or_pic_struct() {
        // Both flags false → parser reads nothing; any payload accepted.
        let ctx = SeiContext::default();
        let pt = parse_pic_timing(&[], &ctx).unwrap();
        assert_eq!(pt.cpb_removal_delay, 0);
        assert_eq!(pt.dpb_output_delay, 0);
        assert!(pt.pic_struct.is_none());
    }

    #[test]
    fn recovery_point_basic() {
        // recovery_frame_cnt=8 → ue codeNum=8.
        // Exp-Golomb of 8: leadingZeros=3 ⇒ "0001001" (7 bits).
        // Then exact_match=1, broken_link=0, changing_slice_group_idc=0 (2 bits).
        // Total = 7 + 1 + 1 + 2 = 11 bits.
        //
        // ue(8) bit pattern: codeNum=8 → bit string "0001001".
        //   leadingZeros=3 zeros, a 1, then suffix = 8 - (2^3 - 1) = 1 in 3 bits → "001".
        //   → "000 1 001" = "0001001".
        let fields = [
            (0b0001001, 7), // ue=8
            (1, 1),         // exact_match_flag
            (0, 1),         // broken_link_flag
            (0, 2),         // changing_slice_group_idc
        ];
        let payload = pack_bits(&fields);
        let rp = parse_recovery_point(&payload).unwrap();
        assert_eq!(rp.recovery_frame_cnt, 8);
        assert!(rp.exact_match_flag);
        assert!(!rp.broken_link_flag);
        assert_eq!(rp.changing_slice_group_idc, 0);
    }

    #[test]
    fn user_data_unregistered_roundtrip() {
        let uuid: [u8; 16] = [
            0xDC, 0x45, 0xE9, 0xBD, 0xE6, 0xD9, 0x48, 0xB7, 0x96, 0x2C, 0xD8, 0x20, 0xD9, 0x23,
            0xEE, 0xEF,
        ];
        let mut payload = Vec::new();
        payload.extend_from_slice(&uuid);
        payload.extend_from_slice(b"hello");
        let udu = parse_user_data_unregistered(&payload).unwrap();
        assert_eq!(udu.uuid, uuid);
        assert_eq!(udu.user_data, b"hello");
    }

    #[test]
    fn user_data_unregistered_too_short() {
        let err = parse_user_data_unregistered(&[0u8; 15]).unwrap_err();
        assert_eq!(err, SeiError::UserDataTooShort(15));
    }

    #[test]
    fn filler_payload_all_ff() {
        parse_filler_payload(&[0xFF; 5]).unwrap();
    }

    #[test]
    fn filler_payload_bad_byte_reports_position() {
        let mut bytes = [0xFFu8; 5];
        bytes[2] = 0xFE;
        let err = parse_filler_payload(&bytes).unwrap_err();
        assert_eq!(err, SeiError::FillerNonFfAt(2));
    }

    #[test]
    fn mastering_display_decodes_fields() {
        // Use recognisable values:
        //   primaries: (10000,20000), (30000,40000), (45000,5000)
        //   white_point: (31270,32900) (D65-ish)
        //   max_lum: 10_000_000 (= 1000 cd/m^2), min_lum: 500 (= 0.05 cd/m^2)
        let mut p: Vec<u8> = Vec::new();
        fn push_u16(v: &mut Vec<u8>, x: u16) {
            v.push((x >> 8) as u8);
            v.push((x & 0xFF) as u8);
        }
        fn push_u32(v: &mut Vec<u8>, x: u32) {
            v.extend_from_slice(&x.to_be_bytes());
        }
        for &(x, y) in &[(10000u16, 20000u16), (30000, 40000), (45000, 5000)] {
            push_u16(&mut p, x);
            push_u16(&mut p, y);
        }
        push_u16(&mut p, 31270);
        push_u16(&mut p, 32900);
        push_u32(&mut p, 10_000_000);
        push_u32(&mut p, 500);

        let m = parse_mastering_display(&p).unwrap();
        assert_eq!(m.display_primaries[0], (10000, 20000));
        assert_eq!(m.display_primaries[1], (30000, 40000));
        assert_eq!(m.display_primaries[2], (45000, 5000));
        assert_eq!(m.white_point, (31270, 32900));
        assert_eq!(m.max_display_mastering_luminance, 10_000_000);
        assert_eq!(m.min_display_mastering_luminance, 500);
    }

    #[test]
    fn content_light_level_decodes_fields() {
        // max_content=1000, max_pic_avg=400.
        let payload = [0x03, 0xE8, 0x01, 0x90]; // 1000, 400
        let c = parse_content_light_level(&payload).unwrap();
        assert_eq!(c.max_content_light_level, 1000);
        assert_eq!(c.max_pic_average_light_level, 400);
    }

    #[test]
    fn parse_payload_dispatches_unknown() {
        let ctx = SeiContext::default();
        let raw = vec![0x01, 0x02, 0x03];
        let got = parse_payload(999, &raw, &ctx).unwrap();
        match got {
            SeiPayload::Unknown {
                payload_type,
                payload,
            } => {
                assert_eq!(payload_type, 999);
                assert_eq!(payload, raw);
            }
            other => panic!("expected Unknown, got {:?}", other),
        }
    }

    #[test]
    fn parse_payload_dispatches_known_content_light_level() {
        let ctx = SeiContext::default();
        let payload = [0x03, 0xE8, 0x01, 0x90];
        let got = parse_payload(144, &payload, &ctx).unwrap();
        match got {
            SeiPayload::ContentLightLevel(c) => {
                assert_eq!(c.max_content_light_level, 1000);
                assert_eq!(c.max_pic_average_light_level, 400);
            }
            other => panic!("expected ContentLightLevel, got {:?}", other),
        }
    }

    #[test]
    fn parse_payload_dispatches_filler() {
        let ctx = SeiContext::default();
        let got = parse_payload(3, &[0xFF; 4], &ctx).unwrap();
        assert_eq!(got, SeiPayload::FillerPayload);
    }

    // §D.2.6 — user_data_registered_itu_t_t35.
    //
    // Country code 0xB5 (USA) is the headline real-world case: A/53
    // closed-caption SEI uses (country_code=0xB5, provider_code=0x0031,
    // provider_user_id="GA94"); HDR10+ uses (0xB5, 0x003C); Dolby
    // Vision uses (country_code=0x4D, no extension).
    #[test]
    fn registered_user_data_us_country_no_extension() {
        // 0xB5 (USA) + payload {0x00, 0x31, b"GA94"}
        let payload = [0xB5, 0x00, 0x31, b'G', b'A', b'9', b'4'];
        let got = parse_user_data_registered_itu_t_t35(&payload).unwrap();
        assert_eq!(got.country_code, 0xB5);
        assert_eq!(got.country_code_extension, None);
        assert_eq!(got.payload_bytes, &[0x00, 0x31, b'G', b'A', b'9', b'4']);
    }

    #[test]
    fn registered_user_data_escape_country_takes_extension_byte() {
        // 0xFF (escape) + extension 0x42 + payload {0x00, 0x01}
        let payload = [0xFF, 0x42, 0x00, 0x01];
        let got = parse_user_data_registered_itu_t_t35(&payload).unwrap();
        assert_eq!(got.country_code, 0xFF);
        assert_eq!(got.country_code_extension, Some(0x42));
        assert_eq!(got.payload_bytes, &[0x00, 0x01]);
    }

    #[test]
    fn registered_user_data_empty_payload_after_country_code_is_ok() {
        let payload = [0xB5];
        let got = parse_user_data_registered_itu_t_t35(&payload).unwrap();
        assert_eq!(got.country_code, 0xB5);
        assert_eq!(got.country_code_extension, None);
        assert!(got.payload_bytes.is_empty());
    }

    #[test]
    fn registered_user_data_empty_payload_rejected() {
        let err = parse_user_data_registered_itu_t_t35(&[]).unwrap_err();
        assert_eq!(err, SeiError::RegisteredUserDataMissingCountryCode);
    }

    #[test]
    fn registered_user_data_escape_without_extension_rejected() {
        // country_code=0xFF requires the extension byte to follow.
        let err = parse_user_data_registered_itu_t_t35(&[0xFF]).unwrap_err();
        assert_eq!(err, SeiError::RegisteredUserDataMissingCountryCode);
    }

    #[test]
    fn parse_payload_dispatches_registered_user_data() {
        let ctx = SeiContext::default();
        let payload = [0xB5, 0x00, 0x31, b'G', b'A', b'9', b'4'];
        let got = parse_payload(4, &payload, &ctx).unwrap();
        match got {
            SeiPayload::UserDataRegisteredItuTT35(u) => {
                assert_eq!(u.country_code, 0xB5);
                assert_eq!(u.payload_bytes, &[0x00, 0x31, b'G', b'A', b'9', b'4']);
            }
            other => panic!("expected UserDataRegisteredItuTT35, got {:?}", other),
        }
    }

    // §D.2.4 — pan_scan_rect.
    #[test]
    fn pan_scan_rect_cancel_flag_only() {
        // id = 0 → ue "1"; cancel_flag = 1 → 1 bit. Total 2 bits.
        let payload = pack_bits(&[(1, 1), (1, 1)]);
        let got = parse_pan_scan_rect(&payload).unwrap();
        assert_eq!(got.pan_scan_rect_id, 0);
        assert!(got.cancel_flag);
        assert!(got.body.is_none());
    }

    #[test]
    fn pan_scan_rect_single_rectangle_with_offsets() {
        // id = 0 (ue "1"), cancel_flag = 0, cnt_minus1 = 0 (ue "1"),
        // four se() offsets (left=2, right=-2, top=4, bottom=-4),
        // repetition_period = 1 (ue "010").
        //
        // se(2)  → ue(3) → 5 bits "00100"
        // se(-2) → ue(4) → 5 bits "00101"
        // se(4)  → ue(7) → 7 bits "0001000"
        // se(-4) → ue(8) → 7 bits "0001001"
        // ue(1)  → 3 bits "010"
        let fields = [
            (1, 1), // id = 0
            (0, 1), // cancel = 0
            (1, 1), // cnt_minus1 = 0
            (0b00100, 5),
            (0b00101, 5),
            (0b0001000, 7),
            (0b0001001, 7),
            (0b010, 3),
        ];
        let payload = pack_bits(&fields);
        let got = parse_pan_scan_rect(&payload).unwrap();
        assert_eq!(got.pan_scan_rect_id, 0);
        assert!(!got.cancel_flag);
        let body = got.body.expect("body present when cancel_flag = 0");
        assert_eq!(body.rects.len(), 1);
        assert_eq!(body.rects[0].left_offset, 2);
        assert_eq!(body.rects[0].right_offset, -2);
        assert_eq!(body.rects[0].top_offset, 4);
        assert_eq!(body.rects[0].bottom_offset, -4);
        assert_eq!(body.repetition_period, 1);
    }

    #[test]
    fn pan_scan_rect_pathological_count_rejected() {
        // id = 0 (ue "1"), cancel = 0, cnt_minus1 = 32 (ue "0000010_0001",
        // 11 bits) — beyond our 32-rectangle ceiling.
        let fields = [
            (1, 1),
            (0, 1),
            (0b00000_100001, 11), // ue(32) = 11 bits "00000100001"
        ];
        let payload = pack_bits(&fields);
        let err = parse_pan_scan_rect(&payload).unwrap_err();
        assert_eq!(err, SeiError::PanScanCountTooLarge(32));
    }

    #[test]
    fn parse_payload_dispatches_pan_scan_rect() {
        let ctx = SeiContext::default();
        let payload = pack_bits(&[(1, 1), (1, 1)]); // cancel-flag only
        let got = parse_payload(2, &payload, &ctx).unwrap();
        match got {
            SeiPayload::PanScanRect(p) => {
                assert_eq!(p.pan_scan_rect_id, 0);
                assert!(p.cancel_flag);
                assert!(p.body.is_none());
            }
            other => panic!("expected PanScanRect, got {:?}", other),
        }
    }

    #[test]
    fn num_clock_ts_table_d1() {
        // Per Table D-1.
        assert_eq!(num_clock_ts_from_pic_struct(0), 1);
        assert_eq!(num_clock_ts_from_pic_struct(1), 1);
        assert_eq!(num_clock_ts_from_pic_struct(2), 1);
        assert_eq!(num_clock_ts_from_pic_struct(3), 2);
        assert_eq!(num_clock_ts_from_pic_struct(4), 2);
        assert_eq!(num_clock_ts_from_pic_struct(5), 3);
        assert_eq!(num_clock_ts_from_pic_struct(6), 3);
        assert_eq!(num_clock_ts_from_pic_struct(7), 2);
        assert_eq!(num_clock_ts_from_pic_struct(8), 3);
        assert_eq!(num_clock_ts_from_pic_struct(9), 0); // reserved
    }

    // §D.2.26 — frame_packing_arrangement. Build a payload that
    // signals side-by-side (type 3), non-quincunx, so the four grid
    // nibbles are present.
    //
    // Field layout (bits):
    //   id = 0 → ue "1"                              (1)
    //   cancel_flag = 0                              (1)
    //   type = 3                                     (7)
    //   quincunx_sampling_flag = 0                   (1)
    //   content_interpretation_type = 1              (6)
    //   spatial_flipping_flag = 0                    (1)
    //   frame0_flipped_flag = 0                      (1)
    //   field_views_flag = 0                         (1)
    //   current_frame_is_frame0_flag = 1             (1)
    //   frame0_self_contained_flag = 1               (1)
    //   frame1_self_contained_flag = 0               (1)
    //   frame0_grid_position_x = 0                   (4)
    //   frame0_grid_position_y = 0                   (4)
    //   frame1_grid_position_x = 0                   (4)
    //   frame1_grid_position_y = 0                   (4)
    //   reserved_byte = 0                            (8)
    //   repetition_period = 0 → ue "1"               (1)
    //   extension_flag = 0                           (1)
    #[test]
    fn frame_packing_arrangement_side_by_side() {
        let fields = [
            (1, 1), // ue id=0
            (0, 1), // cancel_flag
            (3, 7), // type=3 (side-by-side)
            (0, 1), // quincunx_sampling_flag
            (1, 6), // content_interpretation_type
            (0, 1), // spatial_flipping_flag
            (0, 1), // frame0_flipped_flag
            (0, 1), // field_views_flag
            (1, 1), // current_frame_is_frame0_flag
            (1, 1), // frame0_self_contained_flag
            (0, 1), // frame1_self_contained_flag
            (0, 4), // frame0_grid_x
            (0, 4), // frame0_grid_y
            (0, 4), // frame1_grid_x
            (0, 4), // frame1_grid_y
            (0, 8), // reserved_byte
            (1, 1), // ue repetition_period=0
            (0, 1), // extension_flag
        ];
        let payload = pack_bits(&fields);
        let fpa = parse_frame_packing_arrangement(&payload).unwrap();
        assert_eq!(fpa.id, 0);
        assert!(!fpa.cancel_flag);
        assert_eq!(fpa.ty, 3);
        assert!(!fpa.quincunx_sampling_flag);
        assert_eq!(fpa.content_interpretation_type, 1);
        assert!(fpa.current_frame_is_frame0_flag);
        assert!(fpa.frame0_self_contained_flag);
        assert!(!fpa.frame1_self_contained_flag);
        assert_eq!(fpa.frame0_grid_position_x, 0);
        assert_eq!(fpa.frame1_grid_position_y, 0);
        assert_eq!(fpa.repetition_period, 0);
        assert!(!fpa.extension_flag);
    }

    // §D.2.26 — cancel_flag path skips the body but still reads
    // extension_flag at the end.
    #[test]
    fn frame_packing_arrangement_cancel_path() {
        let fields = [
            (1, 1), // ue id=0
            (1, 1), // cancel_flag = 1
            (1, 1), // extension_flag = 1
        ];
        let payload = pack_bits(&fields);
        let fpa = parse_frame_packing_arrangement(&payload).unwrap();
        assert!(fpa.cancel_flag);
        assert!(fpa.extension_flag);
        // Body fields default to zero when cancel_flag is set.
        assert_eq!(fpa.ty, 0);
        assert_eq!(fpa.repetition_period, 0);
    }

    // §D.2.26 — type 5 (temporal interleaving) and quincunx=1 cases
    // both skip the grid positions per the syntax.
    #[test]
    fn frame_packing_arrangement_type5_skips_grid() {
        let fields = [
            (1, 1), // ue id=0
            (0, 1), // cancel_flag
            (5, 7), // type=5 (temporal interleaving)
            (0, 1), // quincunx
            (0, 6), // content_interp
            (0, 1), // spatial_flip
            (0, 1), // f0_flipped
            (0, 1), // field_views
            (0, 1), // current_is_f0
            (0, 1), // f0_self_contained
            (0, 1), // f1_self_contained
            // no grid positions because type == 5
            (0xAB, 8), // reserved_byte
            (1, 1),    // ue repetition=0
            (0, 1),    // extension_flag
        ];
        let payload = pack_bits(&fields);
        let fpa = parse_frame_packing_arrangement(&payload).unwrap();
        assert_eq!(fpa.ty, 5);
        assert_eq!(fpa.reserved_byte, 0xAB);
    }

    // §D.2.27 — display_orientation. Full body: cancel=0, then a
    // 180-degree rotation (anticlockwise_rotation = 0x8000 =
    // 2^15 units of 360/2^16 degrees).
    #[test]
    fn display_orientation_full_body() {
        let fields = [
            (0, 1),       // cancel_flag = 0
            (1, 1),       // hor_flip
            (0, 1),       // ver_flip
            (0x8000, 16), // anticlockwise_rotation
            (1, 1),       // ue repetition_period = 0
            (1, 1),       // extension_flag = 1
        ];
        let payload = pack_bits(&fields);
        let dor = parse_display_orientation(&payload).unwrap();
        assert!(!dor.cancel_flag);
        assert!(dor.hor_flip);
        assert!(!dor.ver_flip);
        assert_eq!(dor.anticlockwise_rotation, 0x8000);
        assert_eq!(dor.repetition_period, 0);
        assert!(dor.extension_flag);
    }

    // §D.2.27 — cancel_flag path reads nothing else.
    #[test]
    fn display_orientation_cancel_path() {
        let payload = pack_bits(&[(1, 1)]);
        let dor = parse_display_orientation(&payload).unwrap();
        assert!(dor.cancel_flag);
        assert!(!dor.hor_flip);
        assert_eq!(dor.anticlockwise_rotation, 0);
    }

    // §D.2.32 — single u(8).
    #[test]
    fn alternative_transfer_characteristics_pq() {
        // preferred_transfer_characteristics = 16 (PQ / SMPTE ST 2084).
        let payload = [16u8];
        let got = parse_alternative_transfer_characteristics(&payload).unwrap();
        assert_eq!(got, 16);
    }

    // §D.2.25 — tone_mapping_info with model_id = 0 (linear).
    //
    // tone_map_id = 0 → "1"
    // cancel_flag = 0
    // repetition_period = 0 → "1"
    // coded_data_bit_depth = 10
    // target_bit_depth = 8
    // model_id = 0 → "1"
    // min_value = 0x0000_0100 (u32)
    // max_value = 0x0000_03FF (u32)
    #[test]
    fn tone_mapping_info_linear() {
        let fields: [(u64, u32); 8] = [
            (1, 1),            // ue tone_map_id=0
            (0, 1),            // cancel_flag=0
            (1, 1),            // ue repetition_period=0
            (10, 8),           // coded_data_bit_depth
            (8, 8),            // target_bit_depth
            (1, 1),            // ue model_id=0
            (0x0000_0100, 32), // min_value
            (0x0000_03FF, 32), // max_value
        ];
        let packed = pack_bits(&fields);
        let tmi = parse_tone_mapping_info(&packed).unwrap();
        assert_eq!(tmi.tone_map_id, 0);
        assert!(!tmi.cancel_flag);
        let body = tmi.body.as_ref().expect("body present");
        assert_eq!(body.repetition_period, 0);
        assert_eq!(body.coded_data_bit_depth, 10);
        assert_eq!(body.target_bit_depth, 8);
        assert_eq!(body.model_id, 0);
        match &body.model {
            ToneMappingModel::Linear {
                min_value,
                max_value,
            } => {
                assert_eq!(*min_value, 0x0000_0100);
                assert_eq!(*max_value, 0x0000_03FF);
            }
            other => panic!("expected Linear, got {:?}", other),
        }
    }

    // §D.2.25 — tone_mapping_info cancel_flag path: nothing after
    // the cancel bit is read.
    #[test]
    fn tone_mapping_info_cancel_path() {
        // tone_map_id = 7 → ue codeword "0001000" (7 bits).
        // cancel_flag = 1.
        let fields = [
            (0b0001000, 7), // ue(7)
            (1, 1),         // cancel_flag
        ];
        let payload = pack_bits(&fields);
        let tmi = parse_tone_mapping_info(&payload).unwrap();
        assert_eq!(tmi.tone_map_id, 7);
        assert!(tmi.cancel_flag);
        assert!(tmi.body.is_none());
    }

    // §D.2.21 — film_grain_characteristics.
    //
    // cancel_flag = 0
    // model_id = 1
    // separate_colour_description_present_flag = 0
    // blending_mode_id = 0
    // log2_scale_factor = 4
    // comp_model_present_flag = [true, false, false]
    // luma component:
    //   num_intensity_intervals_minus1 = 0  → 1 interval
    //   num_model_values_minus1 = 0         → 1 model value per interval
    //   interval[0]: lower=16, upper=235, comp_model_value[0]=0 (se "1")
    // repetition_period = 0 (ue "1")
    #[test]
    fn film_grain_characteristics_basic() {
        let fields = [
            (0, 1), // cancel_flag
            (1, 2), // model_id
            (0, 1), // separate_colour_description_present_flag
            (0, 2), // blending_mode_id
            (4, 4), // log2_scale_factor
            (1, 1), // comp_model_present_flag[0]
            (0, 1), // comp_model_present_flag[1]
            (0, 1), // comp_model_present_flag[2]
            // c=0 body
            (0, 8),   // num_intensity_intervals_minus1
            (0, 3),   // num_model_values_minus1
            (16, 8),  // interval[0].lower_bound
            (235, 8), // interval[0].upper_bound
            (1, 1),   // comp_model_value[0][0] = se(0) = 0
            (1, 1),   // repetition_period = ue(0) = 0
        ];
        let payload = pack_bits(&fields);
        let fg = parse_film_grain_characteristics(&payload).unwrap();
        assert!(!fg.cancel_flag);
        let body = fg.body.as_ref().expect("body present");
        assert_eq!(body.model_id, 1);
        assert!(!body.separate_colour_description_present_flag);
        assert!(body.separate_colour_description.is_none());
        assert_eq!(body.blending_mode_id, 0);
        assert_eq!(body.log2_scale_factor, 4);
        assert_eq!(body.comp_model_present_flag, [true, false, false]);
        assert_eq!(body.repetition_period, 0);

        let luma = body.components[0].as_ref().expect("luma component present");
        assert_eq!(luma.num_model_values_minus1, 0);
        assert_eq!(luma.intensity_intervals.len(), 1);
        assert_eq!(luma.intensity_intervals[0].lower_bound, 16);
        assert_eq!(luma.intensity_intervals[0].upper_bound, 235);
        assert_eq!(luma.intensity_intervals[0].model_values, vec![0]);

        assert!(body.components[1].is_none());
        assert!(body.components[2].is_none());
    }

    // §D.2.21 — cancel flag leaves body absent.
    #[test]
    fn film_grain_characteristics_cancel_path() {
        let payload = pack_bits(&[(1, 1)]);
        let fg = parse_film_grain_characteristics(&payload).unwrap();
        assert!(fg.cancel_flag);
        assert!(fg.body.is_none());
    }

    // §D.2.21 — with separate colour description the inner block is
    // populated.
    //
    // Both luma and chroma flagged. Each component carries 1 interval
    // with 1 model value. repetition_period = 0.
    #[test]
    fn film_grain_characteristics_with_separate_colour() {
        let fields = [
            (0, 1),  // cancel_flag
            (2, 2),  // model_id
            (1, 1),  // separate_colour_description_present_flag
            (2, 3),  // bit_depth_luma_minus8 (= 10 bit luma)
            (2, 3),  // bit_depth_chroma_minus8
            (1, 1),  // full_range_flag
            (1, 8),  // colour_primaries = 1 (BT.709)
            (13, 8), // transfer_characteristics = 13
            (1, 8),  // matrix_coefficients = 1
            (0, 2),  // blending_mode_id
            (3, 4),  // log2_scale_factor
            (1, 1),  // comp_model_present_flag[0]
            (1, 1),  // comp_model_present_flag[1]
            (0, 1),  // comp_model_present_flag[2]
            // c=0 body
            (0, 8),   // num_intensity_intervals_minus1
            (0, 3),   // num_model_values_minus1
            (0, 8),   // interval[0].lower_bound
            (255, 8), // interval[0].upper_bound
            (1, 1),   // comp_model_value[0][0] = se(0) = 0
            // c=1 body
            (0, 8),   // num_intensity_intervals_minus1
            (0, 3),   // num_model_values_minus1
            (0, 8),   // interval[0].lower_bound
            (255, 8), // interval[0].upper_bound
            (1, 1),   // comp_model_value[0][0] = se(0) = 0
            (1, 1),   // repetition_period = ue(0) = 0
        ];
        let payload = pack_bits(&fields);
        let fg = parse_film_grain_characteristics(&payload).unwrap();
        let body = fg.body.expect("body present");
        assert_eq!(body.model_id, 2);
        let scd = body.separate_colour_description.expect("scd present");
        assert_eq!(scd.bit_depth_luma_minus8, 2);
        assert_eq!(scd.bit_depth_chroma_minus8, 2);
        assert!(scd.full_range_flag);
        assert_eq!(scd.colour_primaries, 1);
        assert_eq!(scd.transfer_characteristics, 13);
        assert_eq!(scd.matrix_coefficients, 1);
        assert_eq!(body.blending_mode_id, 0);
        assert_eq!(body.log2_scale_factor, 3);
        assert_eq!(body.comp_model_present_flag, [true, true, false]);
        assert!(body.components[0].is_some());
        assert!(body.components[1].is_some());
        assert!(body.components[2].is_none());
        assert_eq!(body.repetition_period, 0);
    }

    // §D.2.21 — multi-interval / multi-model-value path with non-trivial
    // se(v) values exercising the full per-c body. Three intervals, two
    // model values per interval; values include +/- non-zero entries.
    //
    // se codewords:
    //   se(0)  = ue(0) = "1"        (1 bit)
    //   se(+1) = ue(1) = "010"      (3 bits)
    //   se(-1) = ue(2) = "011"      (3 bits)
    //   se(+2) = ue(3) = "00100"    (5 bits)
    //   se(-2) = ue(4) = "00101"    (5 bits)
    #[test]
    fn film_grain_characteristics_multi_interval_multi_value() {
        let fields = [
            (0, 1), // cancel_flag
            (1, 2), // model_id (auto-regression)
            (0, 1), // separate_colour_description_present_flag
            (0, 2), // blending_mode_id
            (5, 4), // log2_scale_factor
            (0, 1), // comp_model_present_flag[0] = false
            (1, 1), // comp_model_present_flag[1] = true
            (0, 1), // comp_model_present_flag[2] = false
            // c=1 body
            (2, 8), // num_intensity_intervals_minus1 = 2 → 3 intervals
            (1, 3), // num_model_values_minus1 = 1 → 2 model values per interval
            // interval 0
            (0, 8),     // lower
            (84, 8),    // upper
            (0b010, 3), // se(+1)
            (0b011, 3), // se(-1)
            // interval 1
            (85, 8),
            (170, 8),
            (0b00100, 5), // se(+2)
            (1, 1),       // se(0)
            // interval 2
            (171, 8),
            (255, 8),
            (1, 1),       // se(0)
            (0b00101, 5), // se(-2)
            (0b010, 3),   // repetition_period = ue(1) = 1
        ];
        let payload = pack_bits(&fields);
        let fg = parse_film_grain_characteristics(&payload).unwrap();
        let body = fg.body.expect("body present");
        assert_eq!(body.repetition_period, 1);
        assert!(body.components[0].is_none());
        assert!(body.components[2].is_none());
        let chroma = body.components[1].as_ref().expect("Cb component present");
        assert_eq!(chroma.num_model_values_minus1, 1);
        assert_eq!(chroma.intensity_intervals.len(), 3);
        assert_eq!(chroma.intensity_intervals[0].lower_bound, 0);
        assert_eq!(chroma.intensity_intervals[0].upper_bound, 84);
        assert_eq!(chroma.intensity_intervals[0].model_values, vec![1, -1]);
        assert_eq!(chroma.intensity_intervals[1].lower_bound, 85);
        assert_eq!(chroma.intensity_intervals[1].upper_bound, 170);
        assert_eq!(chroma.intensity_intervals[1].model_values, vec![2, 0]);
        assert_eq!(chroma.intensity_intervals[2].lower_bound, 171);
        assert_eq!(chroma.intensity_intervals[2].upper_bound, 255);
        assert_eq!(chroma.intensity_intervals[2].model_values, vec![0, -2]);
    }

    // §D.2.21 — num_model_values_minus1 > 5 must be rejected.
    #[test]
    fn film_grain_characteristics_rejects_num_model_values_out_of_range() {
        let fields = [
            (0, 1), // cancel_flag
            (0, 2), // model_id
            (0, 1), // separate_colour_description_present_flag
            (0, 2), // blending_mode_id
            (0, 4), // log2_scale_factor
            (1, 1), // comp_model_present_flag[0] = true
            (0, 1),
            (0, 1),
            (0, 8), // num_intensity_intervals_minus1 = 0
            (6, 3), // num_model_values_minus1 = 6 — out of range
        ];
        let payload = pack_bits(&fields);
        let err = parse_film_grain_characteristics(&payload).unwrap_err();
        assert_eq!(
            err,
            SeiError::FilmGrainNumModelValuesOutOfRange {
                component: 0,
                got: 6
            }
        );
    }

    // Dispatcher wiring for the new payload types.
    #[test]
    fn parse_payload_dispatches_display_orientation() {
        let ctx = SeiContext::default();
        let payload = pack_bits(&[(1, 1)]); // cancel_flag=1 only
        let got = parse_payload(47, &payload, &ctx).unwrap();
        match got {
            SeiPayload::DisplayOrientation(d) => assert!(d.cancel_flag),
            other => panic!("expected DisplayOrientation, got {:?}", other),
        }
    }

    #[test]
    fn parse_payload_dispatches_alt_transfer() {
        let ctx = SeiContext::default();
        let got = parse_payload(147, &[18u8], &ctx).unwrap();
        match got {
            SeiPayload::AlternativeTransferCharacteristics(v) => assert_eq!(v, 18),
            other => panic!(
                "expected AlternativeTransferCharacteristics, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn parse_payload_dispatches_frame_packing() {
        let ctx = SeiContext::default();
        let payload = pack_bits(&[
            (1, 1), // ue id=0
            (1, 1), // cancel_flag=1
            (0, 1), // extension_flag=0
        ]);
        let got = parse_payload(45, &payload, &ctx).unwrap();
        match got {
            SeiPayload::FramePackingArrangement(f) => assert!(f.cancel_flag),
            other => panic!("expected FramePackingArrangement, got {:?}", other),
        }
    }

    // §D.2.24 — post_filter_hint round-trip. 1x1 cross-correlation,
    // each component carrying a single coefficient.
    //
    // size_y = 1, size_x = 1   → ue codeword "010" each
    // type   = 2 (cross-corr)  → u(2) "10"
    // c=0 coef = +5  → ue(9) = "0001010"  → se = 5  (codeword "0001010")
    // c=1 coef = -3  → ue(6) = "00111"    → se = -3
    // c=2 coef = 0   → "1"
    // additional_extension_flag = 0
    #[test]
    fn post_filter_hint_basic_cross_correlation_1x1() {
        let fields = [
            (0b010, 3),     // ue(1) = 1 (size_y)
            (0b010, 3),     // ue(1) = 1 (size_x)
            (0b10, 2),      // type = 2
            (0b0001010, 7), // se(+5)
            (0b00111, 5),   // se(-3)
            (1, 1),         // se(0)
            (0, 1),         // additional_extension_flag = 0
        ];
        let payload = pack_bits(&fields);
        let pfh = parse_post_filter_hint(&payload).unwrap();
        assert_eq!(pfh.filter_hint_size_x, 1);
        assert_eq!(pfh.filter_hint_size_y, 1);
        assert_eq!(pfh.filter_hint_type, 2);
        assert_eq!(pfh.filter_hint_value, vec![5, -3, 0]);
        assert!(!pfh.additional_extension_flag);
    }

    // §D.2.24 — type=0 2-D FIR with size_y = 2, size_x = 3.
    // Each component carries 6 coefficients. Use se(0) = "1" (1 bit
    // each) to keep the fixture short. additional_extension_flag = 1.
    #[test]
    fn post_filter_hint_2d_fir_2x3() {
        let mut fields = vec![
            (0b011, 3),   // ue(2) = 2 (size_y)
            (0b00100, 5), // ue(3) = 3 (size_x)
            (0b00, 2),    // type = 0
        ];
        // 3 components × 2 rows × 3 cols = 18 se(0) coefficients.
        for _ in 0..18 {
            fields.push((1, 1));
        }
        fields.push((1, 1)); // additional_extension_flag = 1
        let payload = pack_bits(&fields);
        let pfh = parse_post_filter_hint(&payload).unwrap();
        assert_eq!(pfh.filter_hint_size_y, 2);
        assert_eq!(pfh.filter_hint_size_x, 3);
        assert_eq!(pfh.filter_hint_type, 0);
        assert_eq!(pfh.filter_hint_value.len(), 18);
        assert!(pfh.filter_hint_value.iter().all(|&v| v == 0));
        assert!(pfh.additional_extension_flag);
    }

    // §D.2.24 — type 1 (two 1-D FIRs) requires size_y == 2.
    #[test]
    fn post_filter_hint_type1_requires_height_two() {
        let mut fields = vec![
            (0b00100, 5), // ue(3) = 3 (size_y) — wrong (should be 2)
            (0b010, 3),   // ue(1) = 1 (size_x)
            (0b01, 2),    // type = 1
        ];
        for _ in 0..9 {
            fields.push((1, 1));
        }
        fields.push((0, 1));
        let payload = pack_bits(&fields);
        let err = parse_post_filter_hint(&payload).unwrap_err();
        assert_eq!(err, SeiError::PostFilterHintType1WrongHeight(3));
    }

    // §D.2.24 — filter_hint_size_x must be in 1..=15. ue(15) = "0001111",
    // ue(16) = "000010001" (codeNum 16). Use size_x=16 to trip the
    // out-of-range guard.
    #[test]
    fn post_filter_hint_rejects_size_out_of_range() {
        let fields = [
            (0b010, 3),       // ue(1) = 1 (size_y)
            (0b000010001, 9), // ue(16) = 16 (size_x — out of range)
        ];
        let payload = pack_bits(&fields);
        let err = parse_post_filter_hint(&payload).unwrap_err();
        assert_eq!(err, SeiError::PostFilterHintSizeOutOfRange { x: 16, y: 1 });
    }

    // §D.2.24 — filter_hint_type 3 is reserved.
    #[test]
    fn post_filter_hint_rejects_reserved_type() {
        let fields = [
            (0b010, 3), // ue(1) = 1 (size_y)
            (0b010, 3), // ue(1) = 1 (size_x)
            (0b11, 2),  // type = 3 (reserved)
        ];
        let payload = pack_bits(&fields);
        let err = parse_post_filter_hint(&payload).unwrap_err();
        assert_eq!(err, SeiError::PostFilterHintTypeReserved);
    }

    #[test]
    fn parse_payload_dispatches_post_filter_hint() {
        let ctx = SeiContext::default();
        let fields = [
            (0b010, 3), // ue(1) = 1 (size_y)
            (0b010, 3), // ue(1) = 1 (size_x)
            (0b10, 2),  // type = 2
            (1, 1),     // se(0)
            (1, 1),     // se(0)
            (1, 1),     // se(0)
            (0, 1),     // additional_extension_flag
        ];
        let payload = pack_bits(&fields);
        let got = parse_payload(22, &payload, &ctx).unwrap();
        match got {
            SeiPayload::PostFilterHint(p) => {
                assert_eq!(p.filter_hint_size_x, 1);
                assert_eq!(p.filter_hint_size_y, 1);
                assert_eq!(p.filter_hint_type, 2);
                assert_eq!(p.filter_hint_value, vec![0, 0, 0]);
            }
            other => panic!("expected PostFilterHint, got {:?}", other),
        }
    }
}
