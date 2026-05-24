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
//! | 7            | §D.1.9      | §D.2.9      | dec_ref_pic_marking_repetition     |
//! | 8            | §D.1.10     | §D.2.10     | spare_pic                          |
//! | 9            | §D.1.11     | §D.2.11     | scene_info                         |
//! | 10           | §D.1.11†    | §D.2.11†    | sub_seq_info                       |
//! | 11           | §D.1.12†    | §D.2.12†    | sub_seq_layer_characteristics      |
//! | 12           | §D.1.13†    | §D.2.13†    | sub_seq_characteristics            |
//! | 13           | §D.1.15     | §D.2.15     | full_frame_freeze                  |
//! | 14           | §D.1.16     | §D.2.16     | full_frame_freeze_release          |
//! | 15           | §D.1.17     | §D.2.17     | full_frame_snapshot                |
//! | 16           | §D.1.18     | §D.2.18     | progressive_refinement_segment_start |
//! | 17           | §D.1.19     | §D.2.19     | progressive_refinement_segment_end |
//! | 18           | §D.1.20     | §D.2.20     | motion_constrained_slice_group_set |
//! | 19           | §D.1.21     | §D.2.21     | film_grain_characteristics         |
//! | 20           | §D.1.22     | §D.2.22     | deblocking_filter_display_preference |
//! | 21           | §D.1.23     | §D.2.23     | stereo_video_info                  |
//! | 22           | §D.1.24     | §D.2.24     | post_filter_hint                   |
//! | 23           | §D.1.25     | §D.2.25     | tone_mapping_info                  |
//! | 45           | §D.1.26     | §D.2.26     | frame_packing_arrangement          |
//! | 47           | §D.1.27     | §D.2.27     | display_orientation                |
//! | 137          | §D.1.29     | §D.2.29     | mastering_display_colour_volume    |
//! | 144          | §D.1.31     | §D.2.31     | content_light_level_info           |
//! | 147          | §D.1.32     | §D.2.32     | alternative_transfer_characteristics |
//! | 148          | §D.1.34     | §D.2.34     | ambient_viewing_environment        |
//! | 150          | §D.1.35.1   | §D.2.35.1   | equirectangular_projection         |
//! | 151          | §D.1.35.2   | §D.2.35.2   | cubemap_projection                 |
//! | 154          | §D.1.35.3   | §D.2.35.3   | sphere_rotation                    |
//! | 155          | §D.1.35.4   | §D.2.35.4   | regionwise_packing                 |
//! | 156          | §D.1.35.5   | §D.2.35.5   | omni_viewport                      |
//! | 205          | §D.1.38     | §D.2.38     | shutter_interval_info              |
//!
//! † The sub-sequence SEI family (types 10/11/12) is specified in the
//!   2003 draft (`docs/video/h264/ISO_IEC_14496-10-AVC-2003-draft.pdf`,
//!   §D.1.11–§D.1.13 / §D.2.11–§D.2.13). It was removed from later
//!   editions; the section numbers above are the 2003-draft numbering.

#![allow(dead_code)]

use crate::bitstream::{BitError, BitReader};
use crate::slice_header::{DecRefPicMarking, MmcoOp};

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
    #[error("ambient_viewing_environment ambient_illuminance shall not be 0 per §D.2.34")]
    AmbientIlluminanceZero,
    #[error(
        "ambient_viewing_environment ambient_light_x / ambient_light_y shall be in 0..=50000 per §D.2.34 (got x={x}, y={y})"
    )]
    AmbientLightOutOfRange { x: u32, y: u32 },
    #[error(
        "sphere_rotation yaw_rotation shall be in [-180*2^16, 180*2^16 - 1] per §D.2.35.3 (got {0})"
    )]
    SphereRotationYawOutOfRange(i32),
    #[error(
        "sphere_rotation pitch_rotation shall be in [-90*2^16, 90*2^16] per §D.2.35.3 (got {0})"
    )]
    SphereRotationPitchOutOfRange(i32),
    #[error(
        "sphere_rotation roll_rotation shall be in [-180*2^16, 180*2^16 - 1] per §D.2.35.3 (got {0})"
    )]
    SphereRotationRollOutOfRange(i32),
    #[error(
        "omni_viewport azimuth_centre shall be in [-180*2^16, 180*2^16 - 1] per §D.2.35.5 (got {0})"
    )]
    OmniViewportAzimuthOutOfRange(i32),
    #[error(
        "omni_viewport elevation_centre shall be in [-90*2^16, 90*2^16] per §D.2.35.5 (got {0})"
    )]
    OmniViewportElevationOutOfRange(i32),
    #[error(
        "omni_viewport tilt_centre shall be in [-180*2^16, 180*2^16 - 1] per §D.2.35.5 (got {0})"
    )]
    OmniViewportTiltOutOfRange(i32),
    #[error("omni_viewport hor_range shall be in 1..=360*2^16 per §D.2.35.5 (got {0})")]
    OmniViewportHorRangeOutOfRange(u32),
    #[error("omni_viewport ver_range shall be in 1..=180*2^16 per §D.2.35.5 (got {0})")]
    OmniViewportVerRangeOutOfRange(u32),
    #[error("shutter_interval_info sii_time_scale shall not be 0 per §D.2.38")]
    ShutterIntervalTimeScaleZero,
    #[error("sub-sequence sub_seq_layer_num shall be less than 256 (got {0})")]
    SubSeqLayerNumTooLarge(u32),
    #[error("sub-sequence sub_seq_id shall be less than 65536 (got {0})")]
    SubSeqIdTooLarge(u32),
    #[error("sub_seq_layer_characteristics num_sub_seq_layers_minus1 shall be <= 255 (got {0})")]
    SubSeqLayersTooLarge(u32),
    #[error("sub_seq_characteristics num_referenced_subseqs shall be less than 256 (got {0})")]
    SubSeqReferencedTooLarge(u32),
    #[error("spare_pic num_spare_pics_minus1 shall be in 0..=15 per §D.2.10 (got {0})")]
    SparePicCountTooLarge(u32),
    #[error(
        "spare_pic spare_area_idc shall be in 0..=2 per §D.2.10 (got {idc} for spare picture {i})"
    )]
    SparePicAreaIdcOutOfRange { i: usize, idc: u32 },
    #[error(
        "spare_pic spare_area_idc {idc} (spare picture {i}) needs PicSizeInMapUnits from the active SPS, but the SeiContext carried 0 — wire pic_size_in_map_units in"
    )]
    SparePicMapUnitsUnknown { i: usize, idc: u32 },
    #[error(
        "content_colour_volume requires at least one of the four present flags (primaries / min / max / avg) to be set per §D.2.33"
    )]
    ContentColourVolumeNoComponents,
    #[error(
        "content_colour_volume ccv_primaries_x/y[{c}] shall be in [-5000000, 5000000] per §D.2.33 (got x={x}, y={y})"
    )]
    ContentColourVolumePrimaryOutOfRange { c: usize, x: i32, y: i32 },
    #[error(
        "content_colour_volume luminance ordering violated per §D.2.33: ccv_min_luminance_value ({min}) <= ccv_avg_luminance_value ({avg}) <= ccv_max_luminance_value ({max}) must hold for whichever values are present"
    )]
    ContentColourVolumeLuminanceOrder { min: u64, avg: u64, max: u64 },
    #[error(
        "dec_ref_pic_marking_repetition memory_management_control_operation shall be in 0..=6 per §7.4.3.3 / Table 7-9 (got {0})"
    )]
    DecRefPicMarkingRepetitionMmcoOutOfRange(u32),
    #[error("regionwise_packing num_packed_regions shall be greater than 0 per §D.2.35.4")]
    RegionWisePackingNoRegions,
    #[error(
        "regionwise_packing proj_picture_width / proj_picture_height shall both be greater than 0 per §D.2.35.4 (got w={w}, h={h})"
    )]
    RegionWisePackingProjPictureZero { w: u32, h: u32 },
    #[error(
        "regionwise_packing packed_picture_width / packed_picture_height shall both be greater than 0 per §D.2.35.4 (got w={w}, h={h})"
    )]
    RegionWisePackingPackedPictureZero { w: u32, h: u32 },
    #[error(
        "regionwise_packing packed region {0} sets guard_band_flag but all four guard-band dimensions are 0 — at least one shall be greater than 0 per §D.2.35.4"
    )]
    RegionWisePackingGuardBandAllZero(usize),
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

/// §D.2.15 — full_frame_freeze.
///
/// Asks the display layer to freeze the most recent output picture
/// for `full_frame_freeze_repetition_period` decoded pictures (in
/// output order); see §D.2.15 for the persistence semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FullFrameFreeze {
    pub full_frame_freeze_repetition_period: u32,
}

/// §D.2.17 — full_frame_snapshot.
///
/// Tags the current decoded picture as a still snapshot identified
/// by `snapshot_id` (typically used to mark a frame for
/// thumbnail / KLV-style external indexing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FullFrameSnapshot {
    pub snapshot_id: u32,
}

/// §D.2.22 — deblocking_filter_display_preference.
///
/// Producer hint for the display layer about whether to display the
/// pre-deblocking-filter or post-deblocking-filter samples. When
/// `cancel_flag` is true the body is absent (see §D.2.22).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeblockingFilterDisplayPreference {
    pub cancel_flag: bool,
    pub display_prior_to_deblocking_preferred_flag: bool,
    pub dec_frame_buffering_constraint_flag: bool,
    pub repetition_period: u32,
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
#[derive(Debug, Copy, Clone)]
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
    /// From the active PPS — `num_slice_groups_minus1 + 1`. Needed by
    /// §D.1.20 motion_constrained_slice_group_set to compute the
    /// `u(v)` width of each slice_group_id entry. Defaults to 1 (single
    /// slice group), under which the §D.1.20 field becomes 0 bits wide.
    pub num_slice_groups: u32,
    /// From the active SPS — `PicSizeInMapUnits` =
    /// `(pic_width_in_mbs_minus1 + 1) * (pic_height_in_map_units_minus1 + 1)`
    /// (§7.4.2.1.1, Eq. 7-25). Needed by §D.1.10 spare_pic to bound the
    /// per-map-unit `spare_unit_flag` / `zero_run_length` loops. Defaults
    /// to 0; a spare_pic message that reaches a per-map-unit branch
    /// (`spare_area_idc == 1` or `2`) with a 0 value is rejected with
    /// [`SeiError::SparePicMapUnitsUnknown`].
    pub pic_size_in_map_units: u32,
    /// From the active SPS — `frame_mbs_only_flag` (§7.3.2.1.1). Needed by
    /// §D.1.9 dec_ref_pic_marking_repetition: the `original_field_pic_flag`
    /// / `original_bottom_field_flag` fields are present only when
    /// `frame_mbs_only_flag == 0`. Defaults to `true` (frame-only coding,
    /// the common case), under which those two fields are absent.
    pub frame_mbs_only_flag: bool,
}

impl Default for SeiContext {
    fn default() -> Self {
        Self {
            initial_cpb_removal_delay_length_minus1: 0,
            cpb_removal_delay_length_minus1: 0,
            dpb_output_delay_length_minus1: 0,
            time_offset_length: 0,
            nal_hrd_cpb_cnt_minus1: None,
            vcl_hrd_cpb_cnt_minus1: None,
            pic_struct_present_flag: false,
            cpb_dpb_delays_present_flag: false,
            // §D.1.20 — 0 bits per slice_group_id field when num_slice_groups = 1.
            num_slice_groups: 1,
            // §D.1.10 — unknown until the active SPS dimensions are wired in.
            pic_size_in_map_units: 0,
            // §D.1.9 — assume frame-only coding unless the active SPS says
            // otherwise; under frame_mbs_only_flag the original_field_pic_flag
            // / original_bottom_field_flag fields are absent.
            frame_mbs_only_flag: true,
        }
    }
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

/// §D.2.15 — full_frame_freeze (payload type 13).
///
/// Syntax (§D.1.15):
/// ```text
/// full_frame_freeze( payloadSize ) {
///   full_frame_freeze_repetition_period   ue(v)
/// }
/// ```
pub fn parse_full_frame_freeze(payload: &[u8]) -> Result<FullFrameFreeze, SeiError> {
    let mut r = BitReader::new(payload);
    let full_frame_freeze_repetition_period = r.ue()?;
    Ok(FullFrameFreeze {
        full_frame_freeze_repetition_period,
    })
}

/// §D.2.17 — full_frame_snapshot (payload type 15).
///
/// Syntax (§D.1.17):
/// ```text
/// full_frame_snapshot( payloadSize ) {
///   snapshot_id   ue(v)
/// }
/// ```
pub fn parse_full_frame_snapshot(payload: &[u8]) -> Result<FullFrameSnapshot, SeiError> {
    let mut r = BitReader::new(payload);
    let snapshot_id = r.ue()?;
    Ok(FullFrameSnapshot { snapshot_id })
}

/// §D.2.22 — deblocking_filter_display_preference (payload type 20).
///
/// Syntax (§D.1.22):
/// ```text
/// deblocking_filter_display_preference( payloadSize ) {
///   deblocking_display_preference_cancel_flag                 u(1)
///   if( !deblocking_display_preference_cancel_flag ) {
///     display_prior_to_deblocking_preferred_flag              u(1)
///     dec_frame_buffering_constraint_flag                     u(1)
///     deblocking_display_preference_repetition_period         ue(v)
///   }
/// }
/// ```
pub fn parse_deblocking_filter_display_preference(
    payload: &[u8],
) -> Result<DeblockingFilterDisplayPreference, SeiError> {
    let mut r = BitReader::new(payload);
    let cancel_flag = r.u(1)? == 1;
    if cancel_flag {
        return Ok(DeblockingFilterDisplayPreference {
            cancel_flag,
            display_prior_to_deblocking_preferred_flag: false,
            dec_frame_buffering_constraint_flag: false,
            repetition_period: 0,
        });
    }
    let display_prior_to_deblocking_preferred_flag = r.u(1)? == 1;
    let dec_frame_buffering_constraint_flag = r.u(1)? == 1;
    let repetition_period = r.ue()?;
    Ok(DeblockingFilterDisplayPreference {
        cancel_flag,
        display_prior_to_deblocking_preferred_flag,
        dec_frame_buffering_constraint_flag,
        repetition_period,
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

// ---------------------------------------------------------------------------
// Round 73 — additional SEI payload types.
//
// All five additions are syntactically simple Exp-Golomb / fixed-width
// flag payloads from Annex D and add transparent parse coverage without
// touching the pixel pipeline. They each implement EXACTLY the §D.1.x
// syntax tables, with semantic-range validation deferred to §D.2.x
// callers (per the existing crate convention of letting the syntax
// layer accept anything the bitstream encodes).
// ---------------------------------------------------------------------------

/// §D.2.10 — spare_pic, the i-th spare picture entry (payload type 8).
///
/// One entry per spare picture referenced by the message. The spare area
/// is encoded with one of three methods selected by `spare_area_idc`:
///
/// * `spare_area_idc == 0` — every slice-group map unit in the spare
///   picture is a spare unit; no per-unit data follows. `spare_units`
///   is `None`.
/// * `spare_area_idc == 1` — one `spare_unit_flag` per map unit in
///   raster scan order. `spare_units` carries those flags
///   (`true` = spare, mapped from `spare_unit_flag == 0` per §D.2.10).
/// * `spare_area_idc == 2` — run-length coded in counter-clockwise
///   box-out order (§8.2.2.4). `zero_run_lengths` carries the raw
///   `zero_run_length[ ]` values; deriving
///   `spareUnitFlagInBoxOutOrder[ ]` (Eq. D-4 / D-5) needs the
///   box-out scan and is left to a caller that has the §8.2.2.4 walk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparePicEntry {
    /// `delta_spare_frame_num[ i ]` — identifies the i-th spare picture
    /// relative to the running candidate frame_num (Eq. D-3). In the
    /// range `0..=MaxFrameNum − 2 + spare_field_flag` per §D.2.10.
    pub delta_spare_frame_num: u32,
    /// `spare_bottom_field_flag[ i ]` — present only when the message's
    /// `spare_field_flag` is set. `true` ⇒ the i-th spare picture is a
    /// bottom field.
    pub spare_bottom_field_flag: Option<bool>,
    /// `spare_area_idc[ i ]` — selects the spare-area coding method
    /// (0, 1, or 2 per §D.2.10).
    pub spare_area_idc: u32,
    /// Present only when `spare_area_idc == 1`. One entry per slice-group
    /// map unit in raster scan order; `true` marks a spare unit
    /// (`spare_unit_flag == 0` per §D.2.10).
    pub spare_units: Option<Vec<bool>>,
    /// Present only when `spare_area_idc == 2`. The raw `zero_run_length[ ]`
    /// runs read until the box-out map-unit count reaches
    /// PicSizeInMapUnits.
    pub zero_run_lengths: Option<Vec<u32>>,
}

/// §D.2.9 — dec_ref_pic_marking_repetition (payload type 7).
///
/// Repeats the `dec_ref_pic_marking()` syntax structure (§7.3.3.3) that
/// originally appeared in the slice header(s) of an earlier picture in
/// the same coded video sequence, so a decoder that lost that earlier
/// picture can still apply its reference-picture-marking operations.
///
/// Syntax (§D.1.9):
/// ```text
/// dec_ref_pic_marking_repetition( payloadSize ) {
///   original_idr_flag                                  u(1)
///   original_frame_num                                 ue(v)
///   if( !frame_mbs_only_flag ) {
///     original_field_pic_flag                          u(1)
///     if( original_field_pic_flag )
///       original_bottom_field_flag                     u(1)
///   }
///   dec_ref_pic_marking( )
/// }
/// ```
///
/// Per §D.2.9, the `IdrPicFlag` used to interpret the embedded
/// `dec_ref_pic_marking()` (clause 7.3.3.3) is `original_idr_flag` rather
/// than the IdrPicFlag of the SEI NAL unit itself: when `original_idr_flag`
/// is set, the embedded structure carries `no_output_of_prior_pics_flag`
/// and `long_term_reference_flag`; otherwise it carries the adaptive /
/// sliding-window MMCO loop.
///
/// `original_field_pic_flag` / `original_bottom_field_flag` are present only
/// when the active SPS has `frame_mbs_only_flag == 0`; the parser reads
/// that flag from [`SeiContext::frame_mbs_only_flag`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecRefPicMarkingRepetition {
    /// `original_idr_flag` — `true` when the repeated structure originally
    /// occurred in an IDR picture (§D.2.9). Also serves as the `IdrPicFlag`
    /// used to interpret the embedded `dec_ref_pic_marking()`.
    pub original_idr_flag: bool,
    /// `original_frame_num` — the frame_num of the picture where the
    /// repeated structure originally occurred (§D.2.9).
    pub original_frame_num: u32,
    /// `original_field_pic_flag` — present only when the active SPS has
    /// `frame_mbs_only_flag == 0`; `None` under frame-only coding (§D.1.9).
    pub original_field_pic_flag: Option<bool>,
    /// `original_bottom_field_flag` — present only when
    /// `original_field_pic_flag == Some(true)` (§D.1.9).
    pub original_bottom_field_flag: Option<bool>,
    /// The repeated `dec_ref_pic_marking()` structure (§7.3.3.3),
    /// interpreted with `IdrPicFlag = original_idr_flag` per §D.2.9.
    pub dec_ref_pic_marking: DecRefPicMarking,
}

/// §D.1.9 / §D.2.9 — parse dec_ref_pic_marking_repetition (payload type 7).
///
/// `frame_mbs_only_flag` of the active SPS comes from
/// [`SeiContext::frame_mbs_only_flag`].
pub fn parse_dec_ref_pic_marking_repetition(
    payload: &[u8],
    ctx: &SeiContext,
) -> Result<DecRefPicMarkingRepetition, SeiError> {
    let mut r = BitReader::new(payload);
    // §D.1.9 — original_idr_flag u(1).
    let original_idr_flag = r.u(1)? == 1;
    // §D.1.9 — original_frame_num ue(v).
    let original_frame_num = r.ue()?;
    // §D.1.9 — original_field_pic_flag / original_bottom_field_flag present
    // only when !frame_mbs_only_flag.
    let (original_field_pic_flag, original_bottom_field_flag) = if ctx.frame_mbs_only_flag {
        (None, None)
    } else {
        let fpf = r.u(1)? == 1;
        let bff = if fpf { Some(r.u(1)? == 1) } else { None };
        (Some(fpf), bff)
    };
    // §D.1.9 / §7.3.3.3 — dec_ref_pic_marking(), interpreted with
    // IdrPicFlag = original_idr_flag per §D.2.9.
    let dec_ref_pic_marking = parse_repeated_dec_ref_pic_marking(&mut r, original_idr_flag)?;
    Ok(DecRefPicMarkingRepetition {
        original_idr_flag,
        original_frame_num,
        original_field_pic_flag,
        original_bottom_field_flag,
        dec_ref_pic_marking,
    })
}

/// §7.3.3.3 — read a `dec_ref_pic_marking()` structure from `r`, using
/// `idr_pic_flag` as the `IdrPicFlag` that selects the IDR branch.
///
/// This is a self-contained spec-direct reader for the SEI repetition
/// payload; it mirrors the §7.3.3.3 syntax table (the same structure that
/// also appears in the slice header) and yields the shared
/// [`DecRefPicMarking`] / [`MmcoOp`] data model.
fn parse_repeated_dec_ref_pic_marking(
    r: &mut BitReader<'_>,
    idr_pic_flag: bool,
) -> Result<DecRefPicMarking, SeiError> {
    if idr_pic_flag {
        // §7.3.3.3 IDR branch.
        let no_output_of_prior_pics_flag = r.u(1)? == 1;
        let long_term_reference_flag = r.u(1)? == 1;
        Ok(DecRefPicMarking {
            no_output_of_prior_pics_flag,
            long_term_reference_flag,
            adaptive_marking: None,
        })
    } else {
        // §7.3.3.3 non-IDR branch.
        let adaptive_ref_pic_marking_mode_flag = r.u(1)? == 1;
        let adaptive_marking = if adaptive_ref_pic_marking_mode_flag {
            let mut ops = Vec::new();
            // do { ... } while( memory_management_control_operation != 0 )
            loop {
                let mmco = r.ue()?;
                match mmco {
                    0 => break,
                    1 => {
                        // difference_of_pic_nums_minus1 ue(v)
                        let diff = r.ue()?;
                        ops.push(MmcoOp::MarkShortTermUnused(diff));
                    }
                    2 => {
                        // long_term_pic_num ue(v)
                        let ltpn = r.ue()?;
                        ops.push(MmcoOp::MarkLongTermUnused(ltpn));
                    }
                    3 => {
                        // difference_of_pic_nums_minus1 ue(v), long_term_frame_idx ue(v)
                        let diff = r.ue()?;
                        let ltfi = r.ue()?;
                        ops.push(MmcoOp::AssignLongTerm(diff, ltfi));
                    }
                    4 => {
                        // max_long_term_frame_idx_plus1 ue(v)
                        let max = r.ue()?;
                        ops.push(MmcoOp::SetMaxLongTermIdx(max));
                    }
                    5 => ops.push(MmcoOp::MarkAllUnused),
                    6 => {
                        // long_term_frame_idx ue(v)
                        let ltfi = r.ue()?;
                        ops.push(MmcoOp::AssignCurrentLongTerm(ltfi));
                    }
                    other => {
                        return Err(SeiError::DecRefPicMarkingRepetitionMmcoOutOfRange(other));
                    }
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

/// §D.2.10 — spare_pic (payload type 8).
///
/// Marks slice-group map units of one or more decoded reference pictures
/// (the spare pictures) as resembling the co-located map units of a
/// specified target picture, so a decoder concealing a lost target-picture
/// region can substitute the spare picture's co-located samples.
///
/// Syntax (§D.1.10):
/// ```text
/// spare_pic( payloadSize ) {
///   target_frame_num                                  ue(v)
///   spare_field_flag                                  u(1)
///   if( spare_field_flag )
///     target_bottom_field_flag                        u(1)
///   num_spare_pics_minus1                             ue(v)
///   for( i = 0; i < num_spare_pics_minus1 + 1; i++ ) {
///     delta_spare_frame_num[ i ]                      ue(v)
///     if( spare_field_flag )
///       spare_bottom_field_flag[ i ]                  u(1)
///     spare_area_idc[ i ]                             ue(v)
///     if( spare_area_idc[ i ] == 1 )
///       for( j = 0; j < PicSizeInMapUnits; j++ )
///         spare_unit_flag[ i ][ j ]                   u(1)
///     else if( spare_area_idc[ i ] == 2 ) {
///       mapUnitCnt = 0
///       for( j = 0; mapUnitCnt < PicSizeInMapUnits; j++ ) {
///         zero_run_length[ i ][ j ]                   ue(v)
///         mapUnitCnt += zero_run_length[ i ][ j ] + 1
///       }
///     }
///   }
/// }
/// ```
///
/// The per-map-unit branches (`spare_area_idc` 1 and 2) iterate over
/// `PicSizeInMapUnits` of the active SPS, which the bitstream does not
/// repeat — the parser reads it from [`SeiContext::pic_size_in_map_units`]
/// and rejects a message that reaches one of those branches with a 0 value
/// ([`SeiError::SparePicMapUnitsUnknown`]). A message that only uses
/// `spare_area_idc == 0` for every spare picture parses without needing
/// PicSizeInMapUnits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparePic {
    /// `target_frame_num` — the frame_num of the target picture (§D.2.10).
    pub target_frame_num: u32,
    /// `spare_field_flag` — `true` ⇒ target and spare pictures are decoded
    /// fields; `false` ⇒ decoded frames.
    pub spare_field_flag: bool,
    /// `target_bottom_field_flag` — present only when `spare_field_flag`.
    /// `true` ⇒ the target picture is a bottom field.
    pub target_bottom_field_flag: Option<bool>,
    /// One entry per spare picture; `entries.len() == num_spare_pics_minus1 + 1`
    /// (1..=16 per §D.2.10).
    pub entries: Vec<SparePicEntry>,
}

pub fn parse_spare_pic(payload: &[u8], ctx: &SeiContext) -> Result<SparePic, SeiError> {
    let mut r = BitReader::new(payload);
    let target_frame_num = r.ue()?;
    let spare_field_flag = r.u(1)? == 1;
    let target_bottom_field_flag = if spare_field_flag {
        Some(r.u(1)? == 1)
    } else {
        None
    };
    let num_spare_pics_minus1 = r.ue()?;
    // §D.2.10 — num_spare_pics_minus1 shall be in the range of 0 to 15.
    if num_spare_pics_minus1 > 15 {
        return Err(SeiError::SparePicCountTooLarge(num_spare_pics_minus1));
    }
    let count = (num_spare_pics_minus1 as usize) + 1;
    let pic_size_in_map_units = ctx.pic_size_in_map_units;
    let mut entries = Vec::with_capacity(count);
    for i in 0..count {
        let delta_spare_frame_num = r.ue()?;
        let spare_bottom_field_flag = if spare_field_flag {
            Some(r.u(1)? == 1)
        } else {
            None
        };
        let spare_area_idc = r.ue()?;
        // §D.2.10 — spare_area_idc shall be in the range of 0 to 2.
        if spare_area_idc > 2 {
            return Err(SeiError::SparePicAreaIdcOutOfRange {
                i,
                idc: spare_area_idc,
            });
        }
        let (spare_units, zero_run_lengths) = match spare_area_idc {
            0 => (None, None),
            1 => {
                // §D.1.10 — spare_unit_flag[ i ][ j ] for j in 0..PicSizeInMapUnits.
                if pic_size_in_map_units == 0 {
                    return Err(SeiError::SparePicMapUnitsUnknown {
                        i,
                        idc: spare_area_idc,
                    });
                }
                let mut units = Vec::with_capacity(pic_size_in_map_units as usize);
                for _ in 0..pic_size_in_map_units {
                    // §D.2.10 — spare_unit_flag == 0 marks a spare unit.
                    units.push(r.u(1)? == 0);
                }
                (Some(units), None)
            }
            // spare_area_idc == 2.
            _ => {
                // §D.1.10 — zero_run_length[ i ][ j ] read until the box-out
                // map-unit count reaches PicSizeInMapUnits.
                if pic_size_in_map_units == 0 {
                    return Err(SeiError::SparePicMapUnitsUnknown {
                        i,
                        idc: spare_area_idc,
                    });
                }
                let mut runs = Vec::new();
                let mut map_unit_cnt: u64 = 0;
                let target = pic_size_in_map_units as u64;
                // §D.1.10 — the read loop continues while mapUnitCnt is below
                // PicSizeInMapUnits; the final run may push mapUnitCnt past it
                // (each run consumes zero_run_length + 1 box-out slots), so the
                // post-loop count is allowed to exceed the target.
                while map_unit_cnt < target {
                    let zero_run_length = r.ue()?;
                    runs.push(zero_run_length);
                    // mapUnitCnt += zero_run_length[ i ][ j ] + 1.
                    map_unit_cnt += zero_run_length as u64 + 1;
                }
                (None, Some(runs))
            }
        };
        entries.push(SparePicEntry {
            delta_spare_frame_num,
            spare_bottom_field_flag,
            spare_area_idc,
            spare_units,
            zero_run_lengths,
        });
    }
    Ok(SparePic {
        target_frame_num,
        spare_field_flag,
        target_bottom_field_flag,
        entries,
    })
}

/// §D.2.11 — scene_info (payload type 9).
///
/// Communicates "the current picture is part of scene N" plus an
/// optional transition descriptor. Used by editors / players that need
/// to know where a scene cut occurred without re-analysing the pixels.
///
/// Syntax (§D.1.11):
/// ```text
/// scene_info( payloadSize ) {
///   scene_info_present_flag                u(1)
///   if( scene_info_present_flag ) {
///     scene_id                             ue(v)
///     scene_transition_type                ue(v)
///     if( scene_transition_type > 3 )
///       second_scene_id                    ue(v)
///   }
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SceneInfo {
    pub scene_info_present_flag: bool,
    /// Present only when `scene_info_present_flag` is true.
    pub scene_id: Option<u32>,
    /// Present only when `scene_info_present_flag` is true. Per §D.2.11:
    /// 0 = no transition, 1 = fade-to-black, 2 = fade-from-black,
    /// 3 = unspecified transition, 4 = dissolve, 5 = wipe, 6 = unspecified,
    /// values > 6 are reserved.
    pub scene_transition_type: Option<u32>,
    /// Present only when `scene_transition_type > 3` — identifies the
    /// "second" scene the current picture transitions to.
    pub second_scene_id: Option<u32>,
}

pub fn parse_scene_info(payload: &[u8]) -> Result<SceneInfo, SeiError> {
    let mut r = BitReader::new(payload);
    let scene_info_present_flag = r.u(1)? == 1;
    if !scene_info_present_flag {
        return Ok(SceneInfo {
            scene_info_present_flag,
            scene_id: None,
            scene_transition_type: None,
            second_scene_id: None,
        });
    }
    let scene_id = r.ue()?;
    let scene_transition_type = r.ue()?;
    let second_scene_id = if scene_transition_type > 3 {
        Some(r.ue()?)
    } else {
        None
    };
    Ok(SceneInfo {
        scene_info_present_flag,
        scene_id: Some(scene_id),
        scene_transition_type: Some(scene_transition_type),
        second_scene_id,
    })
}

/// §D.2.11 — sub_seq_info (payload type 10).
///
/// Locates the current picture within the sub-sequence layer / sub-sequence
/// data-dependency hierarchy. Layer 0 is independently decodable; a layer
/// `N` may be predicted only from layers `0..N`. A sub-sequence is a set of
/// coded pictures within one layer that does not depend on any other
/// sub-sequence in the same or a higher layer.
///
/// Syntax (§D.1.11):
/// ```text
/// sub_seq_info( payloadSize ) {
///   sub_seq_layer_num             ue(v)
///   sub_seq_id                    ue(v)
///   first_ref_pic_flag            u(1)
///   leading_non_ref_pic_flag      u(1)
///   last_pic_flag                 u(1)
///   sub_seq_frame_num_flag        u(1)
///   if( sub_seq_frame_num_flag )
///     sub_seq_frame_num           ue(v)
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubSeqInfo {
    /// Sub-sequence layer number of the current picture (< 256 per §D.2.11).
    pub sub_seq_layer_num: u32,
    /// Identifies the sub-sequence within its layer (< 65536 per §D.2.11).
    pub sub_seq_id: u32,
    /// `true` ⇒ the current picture is the first reference picture of the
    /// sub-sequence in decoding order.
    pub first_ref_pic_flag: bool,
    /// `true` ⇒ the current picture is a non-reference picture preceding any
    /// reference picture in decoding order within the sub-sequence (or the
    /// sub-sequence contains no reference pictures).
    pub leading_non_ref_pic_flag: bool,
    /// `true` ⇒ the current picture is the last picture of the sub-sequence
    /// in decoding order.
    pub last_pic_flag: bool,
    /// Present only when `sub_seq_frame_num_flag` is 1. Counts coded pictures
    /// within the sub-sequence modulo `MaxFrameNum` (< `MaxFrameNum`).
    pub sub_seq_frame_num: Option<u32>,
}

pub fn parse_sub_seq_info(payload: &[u8]) -> Result<SubSeqInfo, SeiError> {
    let mut r = BitReader::new(payload);
    let sub_seq_layer_num = r.ue()?;
    if sub_seq_layer_num >= 256 {
        return Err(SeiError::SubSeqLayerNumTooLarge(sub_seq_layer_num));
    }
    let sub_seq_id = r.ue()?;
    if sub_seq_id >= 65536 {
        return Err(SeiError::SubSeqIdTooLarge(sub_seq_id));
    }
    let first_ref_pic_flag = r.u(1)? == 1;
    let leading_non_ref_pic_flag = r.u(1)? == 1;
    let last_pic_flag = r.u(1)? == 1;
    let sub_seq_frame_num_flag = r.u(1)? == 1;
    let sub_seq_frame_num = if sub_seq_frame_num_flag {
        Some(r.ue()?)
    } else {
        None
    };
    Ok(SubSeqInfo {
        sub_seq_layer_num,
        sub_seq_id,
        first_ref_pic_flag,
        leading_non_ref_pic_flag,
        last_pic_flag,
        sub_seq_frame_num,
    })
}

/// §D.2.12 — one `(accurate_statistics_flag, average_bit_rate,
/// average_frame_rate)` triple, characterising the joint range of
/// sub-sequence layers `0..=layer` (the loop counter index).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubSeqLayerCharacteristic {
    /// `true` ⇒ `average_bit_rate` / `average_frame_rate` are rounded from
    /// statistically correct values; `false` ⇒ they are estimates.
    pub accurate_statistics_flag: bool,
    /// Average bit rate in units of 1000 bits/second (`u(16)`). 0 ⇒
    /// unspecified.
    pub average_bit_rate: u16,
    /// Average frame rate in units of frames/(256 seconds) (`u(16)`). 0 ⇒
    /// unspecified.
    pub average_frame_rate: u16,
}

/// §D.2.12 — sub_seq_layer_characteristics (payload type 11).
///
/// Specifies bit-rate / frame-rate characteristics of sub-sequence layers.
/// The `layer`-th entry characterises the joint range of layers `0..=layer`.
///
/// Syntax (§D.1.12):
/// ```text
/// sub_seq_layer_characteristics( payloadSize ) {
///   num_sub_seq_layers_minus1                 ue(v)
///   for( layer = 0; layer <= num_sub_seq_layers_minus1; layer++ ) {
///     accurate_statistics_flag                u(1)
///     average_bit_rate                        u(16)
///     average_frame_rate                      u(16)
///   }
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubSeqLayerCharacteristics {
    /// One entry per sub-sequence layer; length == `num_sub_seq_layers_minus1
    /// + 1` (1..=256 entries).
    pub layers: Vec<SubSeqLayerCharacteristic>,
}

pub fn parse_sub_seq_layer_characteristics(
    payload: &[u8],
) -> Result<SubSeqLayerCharacteristics, SeiError> {
    let mut r = BitReader::new(payload);
    let num_sub_seq_layers_minus1 = r.ue()?;
    if num_sub_seq_layers_minus1 > 255 {
        return Err(SeiError::SubSeqLayersTooLarge(num_sub_seq_layers_minus1));
    }
    let count = num_sub_seq_layers_minus1 as usize + 1;
    let mut layers = Vec::with_capacity(count);
    for _ in 0..count {
        let accurate_statistics_flag = r.u(1)? == 1;
        let average_bit_rate = r.u(16)? as u16;
        let average_frame_rate = r.u(16)? as u16;
        layers.push(SubSeqLayerCharacteristic {
            accurate_statistics_flag,
            average_bit_rate,
            average_frame_rate,
        });
    }
    Ok(SubSeqLayerCharacteristics { layers })
}

/// §D.2.13 — one referenced sub-sequence identifier triple.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubSeqReference {
    /// Layer number of the referenced sub-sequence.
    pub ref_sub_seq_layer_num: u32,
    /// Identifier of the referenced sub-sequence within its layer.
    pub ref_sub_seq_id: u32,
    /// `false` ⇒ the referenced sub-sequence's first picture precedes the
    /// target's first picture in decoding order; `true` ⇒ it succeeds it.
    pub ref_sub_seq_direction: bool,
}

/// §D.2.13 — sub_seq_characteristics (payload type 12).
///
/// Describes the duration / rate of a target sub-sequence (the next
/// sub-sequence in decoding order with the given layer + id) and which
/// other sub-sequences it inter-predicts from.
///
/// Syntax (§D.1.13):
/// ```text
/// sub_seq_characteristics( payloadSize ) {
///   sub_seq_layer_num                         ue(v)
///   sub_seq_id                                ue(v)
///   duration_flag                             u(1)
///   if( duration_flag )
///     sub_seq_duration                        u(32)
///   average_rate_flag                         u(1)
///   if( average_rate_flag ) {
///     accurate_statistics_flag                u(1)
///     average_bit_rate                        u(16)
///     average_frame_rate                      u(16)
///   }
///   num_referenced_subseqs                    ue(v)
///   for( n = 0; n < num_referenced_subseqs; n++ ) {
///     ref_sub_seq_layer_num                   ue(v)
///     ref_sub_seq_id                          ue(v)
///     ref_sub_seq_direction                   u(1)
///   }
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubSeqCharacteristics {
    /// Layer the message applies to (< 256 per §D.2.13).
    pub sub_seq_layer_num: u32,
    /// Sub-sequence within the layer the message applies to (< 65536).
    pub sub_seq_id: u32,
    /// Duration of the target sub-sequence in 90-kHz clock ticks. Present
    /// only when `duration_flag` is 1.
    pub sub_seq_duration: Option<u32>,
    /// Average bit/frame-rate statistics. Present only when
    /// `average_rate_flag` is 1.
    pub average_rate: Option<SubSeqLayerCharacteristic>,
    /// Sub-sequences whose pictures are used as references for inter
    /// prediction by the target sub-sequence (length < 256).
    pub referenced: Vec<SubSeqReference>,
}

pub fn parse_sub_seq_characteristics(payload: &[u8]) -> Result<SubSeqCharacteristics, SeiError> {
    let mut r = BitReader::new(payload);
    let sub_seq_layer_num = r.ue()?;
    if sub_seq_layer_num >= 256 {
        return Err(SeiError::SubSeqLayerNumTooLarge(sub_seq_layer_num));
    }
    let sub_seq_id = r.ue()?;
    if sub_seq_id >= 65536 {
        return Err(SeiError::SubSeqIdTooLarge(sub_seq_id));
    }
    let duration_flag = r.u(1)? == 1;
    let sub_seq_duration = if duration_flag { Some(r.u(32)?) } else { None };
    let average_rate_flag = r.u(1)? == 1;
    let average_rate = if average_rate_flag {
        let accurate_statistics_flag = r.u(1)? == 1;
        let average_bit_rate = r.u(16)? as u16;
        let average_frame_rate = r.u(16)? as u16;
        Some(SubSeqLayerCharacteristic {
            accurate_statistics_flag,
            average_bit_rate,
            average_frame_rate,
        })
    } else {
        None
    };
    let num_referenced_subseqs = r.ue()?;
    if num_referenced_subseqs >= 256 {
        return Err(SeiError::SubSeqReferencedTooLarge(num_referenced_subseqs));
    }
    let mut referenced = Vec::with_capacity(num_referenced_subseqs as usize);
    for _ in 0..num_referenced_subseqs {
        let ref_sub_seq_layer_num = r.ue()?;
        let ref_sub_seq_id = r.ue()?;
        let ref_sub_seq_direction = r.u(1)? == 1;
        referenced.push(SubSeqReference {
            ref_sub_seq_layer_num,
            ref_sub_seq_id,
            ref_sub_seq_direction,
        });
    }
    Ok(SubSeqCharacteristics {
        sub_seq_layer_num,
        sub_seq_id,
        sub_seq_duration,
        average_rate,
        referenced,
    })
}

/// §D.2.18 — progressive_refinement_segment_start (payload type 16).
///
/// Marks the first picture of a sequence whose subsequent pictures
/// refine the same scene (e.g. a still photo progressively encoded).
///
/// Syntax (§D.1.18):
/// ```text
/// progressive_refinement_segment_start( payloadSize ) {
///   progressive_refinement_id              ue(v)
///   num_refinement_steps_minus1            ue(v)
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressiveRefinementSegmentStart {
    pub progressive_refinement_id: u32,
    pub num_refinement_steps_minus1: u32,
}

pub fn parse_progressive_refinement_segment_start(
    payload: &[u8],
) -> Result<ProgressiveRefinementSegmentStart, SeiError> {
    let mut r = BitReader::new(payload);
    let progressive_refinement_id = r.ue()?;
    let num_refinement_steps_minus1 = r.ue()?;
    Ok(ProgressiveRefinementSegmentStart {
        progressive_refinement_id,
        num_refinement_steps_minus1,
    })
}

/// §D.2.19 — progressive_refinement_segment_end (payload type 17).
///
/// Closes a refinement segment opened by payload type 16, identified
/// by the matching `progressive_refinement_id`.
///
/// Syntax (§D.1.19):
/// ```text
/// progressive_refinement_segment_end( payloadSize ) {
///   progressive_refinement_id              ue(v)
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressiveRefinementSegmentEnd {
    pub progressive_refinement_id: u32,
}

pub fn parse_progressive_refinement_segment_end(
    payload: &[u8],
) -> Result<ProgressiveRefinementSegmentEnd, SeiError> {
    let mut r = BitReader::new(payload);
    let progressive_refinement_id = r.ue()?;
    Ok(ProgressiveRefinementSegmentEnd {
        progressive_refinement_id,
    })
}

/// §D.2.23 — stereo_video_info (payload type 21).
///
/// Pre-dates the more capable frame_packing_arrangement (payload 45);
/// still used by some legacy stereoscopic streams. Identifies which
/// field / frame carries the left vs right view.
///
/// Syntax (§D.1.23):
/// ```text
/// stereo_video_info( payloadSize ) {
///   field_views_flag                       u(1)
///   if( field_views_flag )
///     top_field_is_left_view_flag          u(1)
///   else {
///     current_frame_is_left_view_flag      u(1)
///     next_frame_is_second_view_flag       u(1)
///   }
///   left_view_self_contained_flag          u(1)
///   right_view_self_contained_flag         u(1)
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StereoVideoInfo {
    pub field_views_flag: bool,
    /// Present only when `field_views_flag` is true.
    pub top_field_is_left_view_flag: Option<bool>,
    /// Present only when `field_views_flag` is false.
    pub current_frame_is_left_view_flag: Option<bool>,
    /// Present only when `field_views_flag` is false.
    pub next_frame_is_second_view_flag: Option<bool>,
    pub left_view_self_contained_flag: bool,
    pub right_view_self_contained_flag: bool,
}

pub fn parse_stereo_video_info(payload: &[u8]) -> Result<StereoVideoInfo, SeiError> {
    let mut r = BitReader::new(payload);
    let field_views_flag = r.u(1)? == 1;
    let (
        top_field_is_left_view_flag,
        current_frame_is_left_view_flag,
        next_frame_is_second_view_flag,
    ) = if field_views_flag {
        (Some(r.u(1)? == 1), None, None)
    } else {
        let cur = r.u(1)? == 1;
        let nxt = r.u(1)? == 1;
        (None, Some(cur), Some(nxt))
    };
    let left_view_self_contained_flag = r.u(1)? == 1;
    let right_view_self_contained_flag = r.u(1)? == 1;
    Ok(StereoVideoInfo {
        field_views_flag,
        top_field_is_left_view_flag,
        current_frame_is_left_view_flag,
        next_frame_is_second_view_flag,
        left_view_self_contained_flag,
        right_view_self_contained_flag,
    })
}

/// §D.2.20 — motion_constrained_slice_group_set (payload type 18).
///
/// Identifies a set of slice groups that the encoder promises to
/// motion-compensate only from samples inside the same set. Used by
/// ROI / sub-picture decoding workflows.
///
/// Syntax (§D.1.20):
/// ```text
/// motion_constrained_slice_group_set( payloadSize ) {
///   num_slice_groups_in_set_minus1         ue(v)
///   for( i = 0; i <= num_slice_groups_in_set_minus1; i++ )
///     slice_group_id[ i ]                  u(v)   // v = Ceil(Log2(num_slice_groups))
///   exact_sample_value_match_flag          u(1)
///   pan_scan_rect_flag                     u(1)
///   if( pan_scan_rect_flag )
///     pan_scan_rect_id                     ue(v)
/// }
/// ```
///
/// Note on slice_group_id width: §D.1.20 specifies `u(v)` for
/// slice_group_id[i] with `v = Ceil(Log2(num_slice_groups_minus1 + 1))`
/// per the active PPS. We accept the width hint via [`SeiContext`]
/// (`num_slice_groups`); when the field is missing or `<= 1` the §D.2.20
/// width collapses to 0 bits and no slice_group_id is read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MotionConstrainedSliceGroupSet {
    pub slice_group_ids: Vec<u32>,
    pub exact_sample_value_match_flag: bool,
    pub pan_scan_rect_flag: bool,
    /// Present only when `pan_scan_rect_flag` is true.
    pub pan_scan_rect_id: Option<u32>,
}

pub fn parse_motion_constrained_slice_group_set(
    payload: &[u8],
    ctx: &SeiContext,
) -> Result<MotionConstrainedSliceGroupSet, SeiError> {
    let mut r = BitReader::new(payload);
    let num_slice_groups_in_set_minus1 = r.ue()?;
    // §D.1.20 — slice_group_id[ i ] u(v). The width is Ceil(Log2(num_slice_groups))
    // taken from the active PPS via `SeiContext::num_slice_groups`. When the
    // PPS only declares a single slice group, the encoded width is 0 bits
    // and we read zero slice_group_id values.
    let num_slice_groups = ctx.num_slice_groups.max(1);
    let width = if num_slice_groups <= 1 {
        0
    } else {
        // Ceil(Log2(num_slice_groups)).
        32 - (num_slice_groups - 1).leading_zeros()
    };
    let count = (num_slice_groups_in_set_minus1 as usize).saturating_add(1);
    let mut slice_group_ids = Vec::with_capacity(count.min(8));
    for _ in 0..count {
        let v = if width == 0 { 0 } else { r.u(width)? };
        slice_group_ids.push(v);
    }
    let exact_sample_value_match_flag = r.u(1)? == 1;
    let pan_scan_rect_flag = r.u(1)? == 1;
    let pan_scan_rect_id = if pan_scan_rect_flag {
        Some(r.ue()?)
    } else {
        None
    };
    Ok(MotionConstrainedSliceGroupSet {
        slice_group_ids,
        exact_sample_value_match_flag,
        pan_scan_rect_flag,
        pan_scan_rect_id,
    })
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

// ---------------------------------------------------------------------------
// Round 107 — content_colour_volume (payload type 149).
//
// HDR colour-volume metadata, parallel in shape to mastering_display
// (137) / content_light_level (144). cancel/persistence gate plus four
// independent "present" flags select which sub-blocks (primaries, min /
// max / avg luminance) follow. Range + ordering validation per §D.2.33.
// ---------------------------------------------------------------------------

/// One CIE-1931 chromaticity coordinate pair for a content colour
/// primary, in normalized increments of 0.00002. Per §D.2.33 each
/// component is signed `i(32)` constrained to [-5 000 000, 5 000 000].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContentColourVolumePrimary {
    /// ccv_primaries_x[ c ] — CIE 1931 x, units of 0.00002.
    pub x: i32,
    /// ccv_primaries_y[ c ] — CIE 1931 y, units of 0.00002.
    pub y: i32,
}

/// §D.2.33 — content_colour_volume (payload type 149).
///
/// Describes the nominal colour volume (display gamut + luminance
/// range) of the associated pictures. Complements mastering_display
/// (137, the *mastering* display's volume) and content_light_level
/// (144, MaxCLL / MaxFALL): this message records the volume actually
/// *present in the content* rather than the mastering display's
/// capability.
///
/// Syntax (§D.1.33):
/// ```text
/// content_colour_volume( payloadSize ) {
///   ccv_cancel_flag                              u(1)
///   if( !ccv_cancel_flag ) {
///     ccv_persistence_flag                       u(1)
///     ccv_primaries_present_flag                 u(1)
///     ccv_min_luminance_value_present_flag       u(1)
///     ccv_max_luminance_value_present_flag       u(1)
///     ccv_avg_luminance_value_present_flag       u(1)
///     ccv_reserved_zero_2bits                    u(2)
///     if( ccv_primaries_present_flag )
///       for( c = 0; c < 3; c++ ) {
///         ccv_primaries_x[ c ]                   i(32)
///         ccv_primaries_y[ c ]                   i(32)
///       }
///     if( ccv_min_luminance_value_present_flag )
///       ccv_min_luminance_value                  u(32)
///     if( ccv_max_luminance_value_present_flag )
///       ccv_max_luminance_value                  u(32)
///     if( ccv_avg_luminance_value_present_flag )
///       ccv_avg_luminance_value                  u(32)
///   }
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentColourVolume {
    /// ccv_cancel_flag. When `true`, cancels persistence of any prior
    /// content colour volume SEI message and no further fields follow.
    pub cancel_flag: bool,
    /// Present only when `cancel_flag == false`.
    pub body: Option<ContentColourVolumeBody>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentColourVolumeBody {
    /// ccv_persistence_flag.
    pub persistence_flag: bool,
    /// ccv_primaries_x/y[ 0..3 ]. Present only when
    /// `ccv_primaries_present_flag == 1`. Per §D.2.33 the suggested
    /// ordering is green (0), blue (1), red (2).
    pub primaries: Option<[ContentColourVolumePrimary; 3]>,
    /// ccv_min_luminance_value, units of 0.0000001. Present only when
    /// `ccv_min_luminance_value_present_flag == 1`.
    pub min_luminance_value: Option<u32>,
    /// ccv_max_luminance_value, units of 0.0000001. Present only when
    /// `ccv_max_luminance_value_present_flag == 1`.
    pub max_luminance_value: Option<u32>,
    /// ccv_avg_luminance_value, units of 0.0000001. Present only when
    /// `ccv_avg_luminance_value_present_flag == 1`.
    pub avg_luminance_value: Option<u32>,
}

/// §D.2.33 — parse content_colour_volume (payload type 149).
pub fn parse_content_colour_volume(payload: &[u8]) -> Result<ContentColourVolume, SeiError> {
    let mut r = BitReader::new(payload);
    let cancel_flag = r.u(1)? == 1;
    if cancel_flag {
        return Ok(ContentColourVolume {
            cancel_flag,
            body: None,
        });
    }
    let persistence_flag = r.u(1)? == 1;
    let primaries_present = r.u(1)? == 1;
    let min_present = r.u(1)? == 1;
    let max_present = r.u(1)? == 1;
    let avg_present = r.u(1)? == 1;
    // §D.2.33: ccv_reserved_zero_2bits — "Decoders shall ignore the
    // value", but the two bits are still consumed.
    let _reserved = r.u(2)?;

    // §D.2.33 bitstream conformance: the four present flags shall not
    // all be 0.
    if !primaries_present && !min_present && !max_present && !avg_present {
        return Err(SeiError::ContentColourVolumeNoComponents);
    }

    let primaries = if primaries_present {
        let mut prims = [ContentColourVolumePrimary { x: 0, y: 0 }; 3];
        for (c, prim) in prims.iter_mut().enumerate() {
            let x = r.i(32)?;
            let y = r.i(32)?;
            // §D.2.33: values shall be in [-5 000 000, 5 000 000].
            const PRIM_LO: i32 = -5_000_000;
            const PRIM_HI: i32 = 5_000_000;
            if !(PRIM_LO..=PRIM_HI).contains(&x) || !(PRIM_LO..=PRIM_HI).contains(&y) {
                return Err(SeiError::ContentColourVolumePrimaryOutOfRange { c, x, y });
            }
            *prim = ContentColourVolumePrimary { x, y };
        }
        Some(prims)
    } else {
        None
    };

    let min_luminance_value = if min_present { Some(r.u(32)?) } else { None };
    let max_luminance_value = if max_present { Some(r.u(32)?) } else { None };
    let avg_luminance_value = if avg_present { Some(r.u(32)?) } else { None };

    // §D.2.33 ordering: min <= avg <= max, min <= max, for whichever
    // values are present. Each pairwise constraint applies only when
    // both members of the pair are present.
    if let (Some(min), Some(avg)) = (min_luminance_value, avg_luminance_value) {
        if min > avg {
            return Err(SeiError::ContentColourVolumeLuminanceOrder {
                min: min as u64,
                avg: avg as u64,
                max: max_luminance_value.map_or(u64::MAX, |m| m as u64),
            });
        }
    }
    if let (Some(avg), Some(max)) = (avg_luminance_value, max_luminance_value) {
        if avg > max {
            return Err(SeiError::ContentColourVolumeLuminanceOrder {
                min: min_luminance_value.map_or(0, |m| m as u64),
                avg: avg as u64,
                max: max as u64,
            });
        }
    }
    if let (Some(min), Some(max)) = (min_luminance_value, max_luminance_value) {
        if min > max {
            return Err(SeiError::ContentColourVolumeLuminanceOrder {
                min: min as u64,
                avg: avg_luminance_value.map_or(min as u64, |a| a as u64),
                max: max as u64,
            });
        }
    }

    Ok(ContentColourVolume {
        cancel_flag,
        body: Some(ContentColourVolumeBody {
            persistence_flag,
            primaries,
            min_luminance_value,
            max_luminance_value,
            avg_luminance_value,
        }),
    })
}

// ---------------------------------------------------------------------------
// Round 78 — four additional Annex D payload parsers (HDR + 360 family).
//
// All four are syntactically fixed-width u(n) / i(n) reads, parallel to the
// existing mastering_display / content_light_level / display_orientation
// shape. Range validation per §D.2.34 / §D.2.35.* is policed at parse time.
// ---------------------------------------------------------------------------

/// §D.2.34 — ambient_viewing_environment (payload type 148).
///
/// Describes the nominal ambient viewing environment (illuminance +
/// background chromaticity) the receiving system should assume when
/// adapting the decoded picture for display.
///
/// Syntax (§D.1.34):
/// ```text
/// ambient_viewing_environment( payloadSize ) {
///   ambient_illuminance      u(32)   // units of 0.0001 lux, nonzero
///   ambient_light_x          u(16)   // 0..=50000, CIE 1931 x * 50000
///   ambient_light_y          u(16)   // 0..=50000, CIE 1931 y * 50000
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmbientViewingEnvironment {
    pub ambient_illuminance: u32,
    pub ambient_light_x: u16,
    pub ambient_light_y: u16,
}

pub fn parse_ambient_viewing_environment(
    payload: &[u8],
) -> Result<AmbientViewingEnvironment, SeiError> {
    let mut r = BitReader::new(payload);
    let ambient_illuminance = r.u(32)?;
    let ambient_light_x = r.u(16)?;
    let ambient_light_y = r.u(16)?;
    // §D.2.34: "ambient_illuminance shall not be equal to 0".
    if ambient_illuminance == 0 {
        return Err(SeiError::AmbientIlluminanceZero);
    }
    // §D.2.34: "The values of ambient_light_x and ambient_light_y shall
    // be in the range of 0 to 50 000, inclusive."
    if ambient_light_x > 50_000 || ambient_light_y > 50_000 {
        return Err(SeiError::AmbientLightOutOfRange {
            x: ambient_light_x,
            y: ambient_light_y,
        });
    }
    Ok(AmbientViewingEnvironment {
        ambient_illuminance,
        ambient_light_x: ambient_light_x as u16,
        ambient_light_y: ambient_light_y as u16,
    })
}

/// §D.2.35.1 — equirectangular_projection (payload type 150).
///
/// Signals that the decoded picture is the equirectangular projection
/// of an omnidirectional video, plus optional guard-band geometry on
/// the left / right edges.
///
/// Syntax (§D.1.35.1):
/// ```text
/// equirectangular_projection( payloadSize ) {
///   erp_cancel_flag                              u(1)
///   if( !erp_cancel_flag ) {
///     erp_persistence_flag                       u(1)
///     erp_padding_flag                           u(1)
///     erp_reserved_zero_2bits                    u(2)
///     if( erp_padding_flag == 1 ) {
///       gb_erp_type                              u(3)
///       left_gb_erp_width                        u(8)
///       right_gb_erp_width                       u(8)
///     }
///   }
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EquirectangularProjection {
    pub cancel_flag: bool,
    /// Present only when `cancel_flag` is false.
    pub persistence_flag: Option<bool>,
    /// Present only when `cancel_flag` is false.
    pub padding_flag: Option<bool>,
    /// Present only when `padding_flag` is true. §D.2.35.1 lists
    /// values 0..=3 with meaningful interpretations; values > 3 are
    /// reserved and should be ignored.
    pub gb_erp_type: Option<u8>,
    /// Present only when `padding_flag` is true. Width of the left
    /// guard band, in luma samples.
    pub left_gb_erp_width: Option<u8>,
    /// Present only when `padding_flag` is true. Width of the right
    /// guard band, in luma samples.
    pub right_gb_erp_width: Option<u8>,
}

pub fn parse_equirectangular_projection(
    payload: &[u8],
) -> Result<EquirectangularProjection, SeiError> {
    let mut r = BitReader::new(payload);
    let cancel_flag = r.u(1)? == 1;
    if cancel_flag {
        return Ok(EquirectangularProjection {
            cancel_flag,
            persistence_flag: None,
            padding_flag: None,
            gb_erp_type: None,
            left_gb_erp_width: None,
            right_gb_erp_width: None,
        });
    }
    let persistence_flag = r.u(1)? == 1;
    let padding_flag = r.u(1)? == 1;
    // §D.2.35.1: erp_reserved_zero_2bits — decoders ignore the value
    // but the two bits are still consumed.
    let _reserved = r.u(2)?;
    let (gb_erp_type, left_gb_erp_width, right_gb_erp_width) = if padding_flag {
        let ty = r.u(3)? as u8;
        let left = r.u(8)? as u8;
        let right = r.u(8)? as u8;
        (Some(ty), Some(left), Some(right))
    } else {
        (None, None, None)
    };
    Ok(EquirectangularProjection {
        cancel_flag,
        persistence_flag: Some(persistence_flag),
        padding_flag: Some(padding_flag),
        gb_erp_type,
        left_gb_erp_width,
        right_gb_erp_width,
    })
}

/// §D.2.35.2 — cubemap_projection (payload type 151).
///
/// Marks the decoded picture as the cubemap projection of an
/// omnidirectional video. The spec body carries only a persistence
/// flag — the cube faces' layout is signalled via the frame-packing
/// arrangement SEI message.
///
/// Syntax (§D.1.35.2):
/// ```text
/// cubemap_projection( payloadSize ) {
///   cmp_cancel_flag             u(1)
///   if( !cmp_cancel_flag )
///     cmp_persistence_flag      u(1)
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CubemapProjection {
    pub cancel_flag: bool,
    /// Present only when `cancel_flag` is false.
    pub persistence_flag: Option<bool>,
}

pub fn parse_cubemap_projection(payload: &[u8]) -> Result<CubemapProjection, SeiError> {
    let mut r = BitReader::new(payload);
    let cancel_flag = r.u(1)? == 1;
    let persistence_flag = if cancel_flag {
        None
    } else {
        Some(r.u(1)? == 1)
    };
    Ok(CubemapProjection {
        cancel_flag,
        persistence_flag,
    })
}

/// §D.2.35.3 — sphere_rotation (payload type 154).
///
/// Yaw / pitch / roll rotation between the global and local axes, in
/// units of 2^-16 degrees. Applies after the omnidirectional
/// projection (equirectangular or cubemap) is reconstructed.
///
/// Syntax (§D.1.35.3):
/// ```text
/// sphere_rotation( payloadSize ) {
///   sphere_rotation_cancel_flag           u(1)
///   if( !sphere_rotation_cancel_flag ) {
///     sphere_rotation_persistence_flag    u(1)
///     sphere_rotation_reserved_zero_6bits u(6)
///     yaw_rotation                        i(32)
///     pitch_rotation                      i(32)
///     roll_rotation                       i(32)
///   }
/// }
/// ```
///
/// Per §D.2.35.3 the angles are clamped to:
/// * yaw, roll  ∈ [−180·2^16, 180·2^16 − 1]   (i.e. −11_796_480..=11_796_479)
/// * pitch     ∈ [−90·2^16, 90·2^16]          (i.e. −5_898_240..=5_898_240)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SphereRotation {
    pub cancel_flag: bool,
    /// Present only when `cancel_flag` is false.
    pub persistence_flag: Option<bool>,
    /// Present only when `cancel_flag` is false. Yaw (α) in 2^-16 deg.
    pub yaw_rotation: Option<i32>,
    /// Present only when `cancel_flag` is false. Pitch (β) in 2^-16 deg.
    pub pitch_rotation: Option<i32>,
    /// Present only when `cancel_flag` is false. Roll (γ) in 2^-16 deg.
    pub roll_rotation: Option<i32>,
}

pub fn parse_sphere_rotation(payload: &[u8]) -> Result<SphereRotation, SeiError> {
    let mut r = BitReader::new(payload);
    let cancel_flag = r.u(1)? == 1;
    if cancel_flag {
        return Ok(SphereRotation {
            cancel_flag,
            persistence_flag: None,
            yaw_rotation: None,
            pitch_rotation: None,
            roll_rotation: None,
        });
    }
    let persistence_flag = r.u(1)? == 1;
    // §D.2.35.3: reserved_zero_6bits — decoders ignore the value, but
    // the six bits are still consumed.
    let _reserved = r.u(6)?;
    let yaw = r.i(32)?;
    let pitch = r.i(32)?;
    let roll = r.i(32)?;
    // §D.2.35.3 range checks.
    const YAW_ROLL_LO: i32 = -180 * (1 << 16);
    const YAW_ROLL_HI: i32 = 180 * (1 << 16) - 1;
    const PITCH_LO: i32 = -90 * (1 << 16);
    const PITCH_HI: i32 = 90 * (1 << 16);
    if !(YAW_ROLL_LO..=YAW_ROLL_HI).contains(&yaw) {
        return Err(SeiError::SphereRotationYawOutOfRange(yaw));
    }
    if !(PITCH_LO..=PITCH_HI).contains(&pitch) {
        return Err(SeiError::SphereRotationPitchOutOfRange(pitch));
    }
    if !(YAW_ROLL_LO..=YAW_ROLL_HI).contains(&roll) {
        return Err(SeiError::SphereRotationRollOutOfRange(roll));
    }
    Ok(SphereRotation {
        cancel_flag,
        persistence_flag: Some(persistence_flag),
        yaw_rotation: Some(yaw),
        pitch_rotation: Some(pitch),
        roll_rotation: Some(roll),
    })
}

/// §D.2.35.4 — regionwise_packing (payload type 155).
///
/// Describes the region-wise packing transformation that maps regions
/// of the cropped decoded ("packed") picture onto a projected picture,
/// together with optional guard-band information. Pairs with the
/// equirectangular-projection (150) / cubemap-projection (151) SEIs:
/// per §D.2.35.4 a region-wise packing message with `rwp_cancel_flag`
/// equal to 0 may only follow one of those projection messages in
/// decoding order. We parse and surface the structure; the actual
/// sphere remapping is left to the application.
///
/// Syntax (§D.1.35.4):
/// ```text
/// regionwise_packing( payloadSize ) {
///   rwp_cancel_flag                          u(1)
///   if( !rwp_cancel_flag ) {
///     rwp_persistence_flag                   u(1)
///     constituent_picture_matching_flag      u(1)
///     rwp_reserved_zero_5bits                u(5)
///     num_packed_regions                     u(8)
///     proj_picture_width                     u(32)
///     proj_picture_height                    u(32)
///     packed_picture_width                   u(16)
///     packed_picture_height                  u(16)
///     for( i = 0; i < num_packed_regions; i++ ) {
///       rwp_reserved_zero_4bits[ i ]         u(4)
///       transform_type[ i ]                  u(3)
///       guard_band_flag[ i ]                 u(1)
///       proj_region_width[ i ]               u(32)
///       proj_region_height[ i ]              u(32)
///       proj_region_top[ i ]                 u(32)
///       proj_region_left[ i ]                u(32)
///       packed_region_width[ i ]             u(16)
///       packed_region_height[ i ]            u(16)
///       packed_region_top[ i ]               u(16)
///       packed_region_left[ i ]              u(16)
///       if( guard_band_flag[ i ] ) {
///         left_gb_width[ i ]                 u(8)
///         right_gb_width[ i ]                u(8)
///         top_gb_height[ i ]                 u(8)
///         bottom_gb_height[ i ]              u(8)
///         gb_not_used_for_pred_flag[ i ]     u(1)
///         for( j = 0; j < 4; j++ )
///           gb_type[ i ][ j ]                u(3)
///         rwp_gb_reserved_zero_3bits[ i ]    u(3)
///       }
///     }
///   }
/// }
/// ```
///
/// Per §D.2.35.4 conformance: `num_packed_regions` shall be greater than
/// 0; `proj_picture_width`/`proj_picture_height` and
/// `packed_picture_width`/`packed_picture_height` shall all be greater
/// than 0; and when `guard_band_flag[ i ]` is 1 at least one of the four
/// guard-band dimensions shall be greater than 0. The `rwp_reserved_*`
/// and `gb_type` reserved values are read and ignored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionWisePacking {
    pub cancel_flag: bool,
    /// Present only when `cancel_flag` is false.
    pub persistence_flag: Option<bool>,
    /// Present only when `cancel_flag` is false. When true, the loop's
    /// per-region information applies individually to each constituent
    /// picture of a frame-packed stereoscopic picture (§D.2.35.4).
    pub constituent_picture_matching_flag: Option<bool>,
    /// Width / height of the projected picture, in relative projected
    /// picture sample units. Present only when `cancel_flag` is false.
    pub proj_picture_width: Option<u32>,
    pub proj_picture_height: Option<u32>,
    /// Width / height of the packed picture, in relative packed picture
    /// sample units. Present only when `cancel_flag` is false.
    pub packed_picture_width: Option<u16>,
    pub packed_picture_height: Option<u16>,
    /// One entry per packed region (length = `num_packed_regions`).
    /// Empty when `cancel_flag` is true.
    pub regions: Vec<PackedRegion>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedRegion {
    /// Rotation/mirroring mapping packed → projected (Table D-11, 0..=7).
    pub transform_type: u8,
    /// Projected region geometry, in relative projected picture units.
    pub proj_region_width: u32,
    pub proj_region_height: u32,
    pub proj_region_top: u32,
    pub proj_region_left: u32,
    /// Packed region geometry, in relative packed picture units.
    pub packed_region_width: u16,
    pub packed_region_height: u16,
    pub packed_region_top: u16,
    pub packed_region_left: u16,
    /// Guard-band description, present iff `guard_band_flag[ i ]` was 1.
    pub guard_band: Option<GuardBand>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuardBand {
    pub left_gb_width: u8,
    pub right_gb_width: u8,
    pub top_gb_height: u8,
    pub bottom_gb_height: u8,
    /// `gb_not_used_for_pred_flag`: when true the guard-band samples are
    /// not used in inter prediction (§D.2.35.4).
    pub gb_not_used_for_pred_flag: bool,
    /// `gb_type[ i ][ 0..4 ]` for the left / right / top / bottom edges.
    /// Values greater than 3 are reserved and surfaced verbatim.
    pub gb_type: [u8; 4],
}

pub fn parse_regionwise_packing(payload: &[u8]) -> Result<RegionWisePacking, SeiError> {
    let mut r = BitReader::new(payload);
    let cancel_flag = r.u(1)? == 1;
    if cancel_flag {
        return Ok(RegionWisePacking {
            cancel_flag,
            persistence_flag: None,
            constituent_picture_matching_flag: None,
            proj_picture_width: None,
            proj_picture_height: None,
            packed_picture_width: None,
            packed_picture_height: None,
            regions: Vec::new(),
        });
    }
    let persistence_flag = r.u(1)? == 1;
    let constituent_picture_matching_flag = r.u(1)? == 1;
    let _rwp_reserved_zero_5bits = r.u(5)?; // read and ignored per §D.2.35.4
    let num_packed_regions = r.u(8)?;
    if num_packed_regions == 0 {
        return Err(SeiError::RegionWisePackingNoRegions);
    }
    let proj_picture_width = r.u(32)?;
    let proj_picture_height = r.u(32)?;
    if proj_picture_width == 0 || proj_picture_height == 0 {
        return Err(SeiError::RegionWisePackingProjPictureZero {
            w: proj_picture_width,
            h: proj_picture_height,
        });
    }
    let packed_picture_width = r.u(16)? as u16;
    let packed_picture_height = r.u(16)? as u16;
    if packed_picture_width == 0 || packed_picture_height == 0 {
        return Err(SeiError::RegionWisePackingPackedPictureZero {
            w: packed_picture_width as u32,
            h: packed_picture_height as u32,
        });
    }
    let mut regions = Vec::with_capacity(num_packed_regions as usize);
    for i in 0..num_packed_regions as usize {
        let _rwp_reserved_zero_4bits = r.u(4)?; // read and ignored per §D.2.35.4
        let transform_type = r.u(3)? as u8;
        let guard_band_flag = r.u(1)? == 1;
        let proj_region_width = r.u(32)?;
        let proj_region_height = r.u(32)?;
        let proj_region_top = r.u(32)?;
        let proj_region_left = r.u(32)?;
        let packed_region_width = r.u(16)? as u16;
        let packed_region_height = r.u(16)? as u16;
        let packed_region_top = r.u(16)? as u16;
        let packed_region_left = r.u(16)? as u16;
        let guard_band = if guard_band_flag {
            let left_gb_width = r.u(8)? as u8;
            let right_gb_width = r.u(8)? as u8;
            let top_gb_height = r.u(8)? as u8;
            let bottom_gb_height = r.u(8)? as u8;
            if left_gb_width == 0
                && right_gb_width == 0
                && top_gb_height == 0
                && bottom_gb_height == 0
            {
                return Err(SeiError::RegionWisePackingGuardBandAllZero(i));
            }
            let gb_not_used_for_pred_flag = r.u(1)? == 1;
            let mut gb_type = [0u8; 4];
            for entry in gb_type.iter_mut() {
                *entry = r.u(3)? as u8;
            }
            let _rwp_gb_reserved_zero_3bits = r.u(3)?; // read and ignored
            Some(GuardBand {
                left_gb_width,
                right_gb_width,
                top_gb_height,
                bottom_gb_height,
                gb_not_used_for_pred_flag,
                gb_type,
            })
        } else {
            None
        };
        regions.push(PackedRegion {
            transform_type,
            proj_region_width,
            proj_region_height,
            proj_region_top,
            proj_region_left,
            packed_region_width,
            packed_region_height,
            packed_region_top,
            packed_region_left,
            guard_band,
        });
    }
    Ok(RegionWisePacking {
        cancel_flag,
        persistence_flag: Some(persistence_flag),
        constituent_picture_matching_flag: Some(constituent_picture_matching_flag),
        proj_picture_width: Some(proj_picture_width),
        proj_picture_height: Some(proj_picture_height),
        packed_picture_width: Some(packed_picture_width),
        packed_picture_height: Some(packed_picture_height),
        regions,
    })
}

/// §D.2.35.5 — omni_viewport (payload type 156).
///
/// Specifies the coordinates of one or more spherical-coordinate
/// viewport regions (great-circle bounded) recommended for display
/// when the user has no control of viewing orientation. Pairs with
/// the equirectangular-projection / cubemap-projection SEIs already
/// parsed at payload types 150 / 151.
///
/// Syntax (§D.1.35.5):
/// ```text
/// omni_viewport( payloadSize ) {
///   omni_viewport_id                       u(10)
///   omni_viewport_cancel_flag              u(1)
///   if( !omni_viewport_cancel_flag ) {
///     omni_viewport_persistence_flag       u(1)
///     omni_viewport_cnt_minus1             u(4)
///     for( i = 0; i <= omni_viewport_cnt_minus1; i++ ) {
///       omni_viewport_azimuth_centre[ i ]    i(32)
///       omni_viewport_elevation_centre[ i ]  i(32)
///       omni_viewport_tilt_centre[ i ]       i(32)
///       omni_viewport_hor_range[ i ]         u(32)
///       omni_viewport_ver_range[ i ]         u(32)
///     }
///   }
/// }
/// ```
///
/// Per §D.2.35.5 each viewport carries five values in units of 2^-16
/// degrees, with the clamps:
/// * `azimuth_centre`, `tilt_centre` ∈ [−180·2^16, 180·2^16 − 1]
/// * `elevation_centre` ∈ [−90·2^16, 90·2^16]
/// * `hor_range` ∈ [1, 360·2^16]
/// * `ver_range` ∈ [1, 180·2^16]
///
/// Per the same clause, `omni_viewport_id` values 512..=1023 are
/// reserved; we surface them verbatim and leave handling to callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OmniViewport {
    pub viewport_id: u16,
    pub cancel_flag: bool,
    /// Present only when `cancel_flag` is false.
    pub persistence_flag: Option<bool>,
    /// One entry per recommended viewport (length =
    /// `omni_viewport_cnt_minus1 + 1`). Present only when
    /// `cancel_flag` is false.
    pub viewports: Vec<OmniViewportEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OmniViewportEntry {
    pub azimuth_centre: i32,
    pub elevation_centre: i32,
    pub tilt_centre: i32,
    pub hor_range: u32,
    pub ver_range: u32,
}

pub fn parse_omni_viewport(payload: &[u8]) -> Result<OmniViewport, SeiError> {
    let mut r = BitReader::new(payload);
    // §D.1.35.5 — u(10) viewport_id.
    let viewport_id = r.u(10)? as u16;
    let cancel_flag = r.u(1)? == 1;
    if cancel_flag {
        return Ok(OmniViewport {
            viewport_id,
            cancel_flag,
            persistence_flag: None,
            viewports: Vec::new(),
        });
    }
    let persistence_flag = r.u(1)? == 1;
    let cnt_minus1 = r.u(4)?;
    // §D.2.35.5 range constants.
    const AZ_TILT_LO: i32 = -180 * (1 << 16);
    const AZ_TILT_HI: i32 = 180 * (1 << 16) - 1;
    const EL_LO: i32 = -90 * (1 << 16);
    const EL_HI: i32 = 90 * (1 << 16);
    const HOR_HI: u32 = 360 * (1 << 16);
    const VER_HI: u32 = 180 * (1 << 16);
    let count = (cnt_minus1 as usize) + 1;
    let mut viewports = Vec::with_capacity(count);
    for _ in 0..count {
        let azimuth_centre = r.i(32)?;
        let elevation_centre = r.i(32)?;
        let tilt_centre = r.i(32)?;
        let hor_range = r.u(32)?;
        let ver_range = r.u(32)?;
        if !(AZ_TILT_LO..=AZ_TILT_HI).contains(&azimuth_centre) {
            return Err(SeiError::OmniViewportAzimuthOutOfRange(azimuth_centre));
        }
        if !(EL_LO..=EL_HI).contains(&elevation_centre) {
            return Err(SeiError::OmniViewportElevationOutOfRange(elevation_centre));
        }
        if !(AZ_TILT_LO..=AZ_TILT_HI).contains(&tilt_centre) {
            return Err(SeiError::OmniViewportTiltOutOfRange(tilt_centre));
        }
        if hor_range == 0 || hor_range > HOR_HI {
            return Err(SeiError::OmniViewportHorRangeOutOfRange(hor_range));
        }
        if ver_range == 0 || ver_range > VER_HI {
            return Err(SeiError::OmniViewportVerRangeOutOfRange(ver_range));
        }
        viewports.push(OmniViewportEntry {
            azimuth_centre,
            elevation_centre,
            tilt_centre,
            hor_range,
            ver_range,
        });
    }
    Ok(OmniViewport {
        viewport_id,
        cancel_flag,
        persistence_flag: Some(persistence_flag),
        viewports,
    })
}

/// §D.2.38 — shutter_interval_info (payload type 205).
///
/// Indicates the camera shutter interval (image-sensor exposure time)
/// for the associated source pictures, either as a single CVS-wide
/// constant or as a per-sub-layer schedule keyed on `sii_sub_layer_idx`.
///
/// Syntax (§D.1.38):
/// ```text
/// shutter_interval_info( payloadSize ) {
///   sii_sub_layer_idx                                ue(v)
///   if( sii_sub_layer_idx == 0 ) {
///     shutter_interval_info_present_flag             u(1)
///     if( shutter_interval_info_present_flag ) {
///       sii_time_scale                               u(32)
///       fixed_shutter_interval_within_cvs_flag       u(1)
///       if( fixed_shutter_interval_within_cvs_flag )
///         sii_num_units_in_shutter_interval          u(32)
///       else {
///         sii_max_sub_layers_minus1                  u(3)
///         for( i = 0; i <= sii_max_sub_layers_minus1; i++ )
///           sub_layer_num_units_in_shutter_interval[ i ]  u(32)
///       }
///     }
///   }
/// }
/// ```
///
/// Per §D.2.38: `sii_time_scale` shall be greater than 0; when
/// `fixed_shutter_interval_within_cvs_flag` is 1 the picture's shutter
/// interval, in seconds, is `sii_num_units_in_shutter_interval /
/// sii_time_scale`; otherwise it is
/// `sub_layer_num_units_in_shutter_interval[sii_sub_layer_idx] /
/// sii_time_scale`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShutterIntervalInfo {
    /// Shutter-interval temporal sub-layer index of the current picture.
    pub sub_layer_idx: u32,
    /// Body fields. Present only when `sub_layer_idx == 0`.
    pub body: Option<ShutterIntervalInfoBody>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShutterIntervalInfoBody {
    /// True when the syntax elements `sii_time_scale`,
    /// `fixed_shutter_interval_within_cvs_flag`, and the trailing
    /// schedule are all present.
    pub info_present_flag: bool,
    /// Present only when `info_present_flag` is true.
    pub schedule: Option<ShutterIntervalSchedule>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShutterIntervalSchedule {
    /// Time-base for all `*_num_units_in_shutter_interval` values.
    /// Per §D.2.38 this is the count of time units in one second; e.g.
    /// 27_000_000 for the canonical 27 MHz reference clock. Strictly
    /// positive.
    pub time_scale: u32,
    /// CVS-wide vs. per-sub-layer body.
    pub kind: ShutterIntervalKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShutterIntervalKind {
    /// `fixed_shutter_interval_within_cvs_flag == 1` — one constant
    /// shutter interval for every picture in the CVS. The picture's
    /// shutter interval, in seconds, is `num_units / time_scale`.
    Fixed { num_units: u32 },
    /// `fixed_shutter_interval_within_cvs_flag == 0` — per-sub-layer
    /// schedule. `max_sub_layers_minus1` plus 1 entries; element `i`
    /// gives the shutter interval for pictures with
    /// `sii_sub_layer_idx == i`.
    PerSubLayer {
        max_sub_layers_minus1: u8,
        num_units_per_sub_layer: Vec<u32>,
    },
}

pub fn parse_shutter_interval_info(payload: &[u8]) -> Result<ShutterIntervalInfo, SeiError> {
    let mut r = BitReader::new(payload);
    // §D.1.38: ue(v) `sii_sub_layer_idx`. The bitstream layer's
    // BitReader::ue caps the leading-zero run at 31 so a hostile
    // input can't blow the shifter (see r91 fuzz fix on bitstream.rs).
    let sub_layer_idx = r.ue()?;
    if sub_layer_idx != 0 {
        // Per §D.2.38 the rest of the payload is absent when
        // sub_layer_idx != 0. We preserve the value so callers can
        // map this picture against an earlier IDR-borne schedule.
        return Ok(ShutterIntervalInfo {
            sub_layer_idx,
            body: None,
        });
    }
    let info_present_flag = r.u(1)? == 1;
    if !info_present_flag {
        return Ok(ShutterIntervalInfo {
            sub_layer_idx,
            body: Some(ShutterIntervalInfoBody {
                info_present_flag,
                schedule: None,
            }),
        });
    }
    let time_scale = r.u(32)?;
    // §D.2.38: "The value of sii_time_scale shall be greater than 0."
    if time_scale == 0 {
        return Err(SeiError::ShutterIntervalTimeScaleZero);
    }
    let fixed = r.u(1)? == 1;
    let kind = if fixed {
        let num_units = r.u(32)?;
        ShutterIntervalKind::Fixed { num_units }
    } else {
        let max_sub_layers_minus1 = r.u(3)? as u8;
        let count = (max_sub_layers_minus1 as usize) + 1;
        let mut num_units_per_sub_layer = Vec::with_capacity(count);
        for _ in 0..count {
            num_units_per_sub_layer.push(r.u(32)?);
        }
        ShutterIntervalKind::PerSubLayer {
            max_sub_layers_minus1,
            num_units_per_sub_layer,
        }
    };
    Ok(ShutterIntervalInfo {
        sub_layer_idx,
        body: Some(ShutterIntervalInfoBody {
            info_present_flag,
            schedule: Some(ShutterIntervalSchedule { time_scale, kind }),
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
    /// §D.2.9 — dec_ref_pic_marking_repetition (payload type 7). Round 110.
    DecRefPicMarkingRepetition(DecRefPicMarkingRepetition),
    /// §D.2.10 — spare_pic (payload type 8). Round 103.
    SparePic(SparePic),
    FullFrameFreeze(FullFrameFreeze),
    /// §D.2.16 — full_frame_freeze_release. Empty payload.
    FullFrameFreezeRelease,
    FullFrameSnapshot(FullFrameSnapshot),
    FilmGrainCharacteristics(FilmGrainCharacteristics),
    DeblockingFilterDisplayPreference(DeblockingFilterDisplayPreference),
    PostFilterHint(PostFilterHint),
    ToneMappingInfo(ToneMappingInfo),
    FramePackingArrangement(FramePackingArrangement),
    DisplayOrientation(DisplayOrientation),
    MasteringDisplay(MasteringDisplayColourVolume),
    ContentLightLevel(ContentLightLevelInfo),
    /// preferred_transfer_characteristics — single u(8) field.
    AlternativeTransferCharacteristics(u8),
    /// §D.2.11 — scene_info (payload type 9). Round 73.
    SceneInfo(SceneInfo),
    /// §D.2.11 (2003 draft) — sub_seq_info (payload type 10). Round 99.
    SubSeqInfo(SubSeqInfo),
    /// §D.2.12 (2003 draft) — sub_seq_layer_characteristics (payload type 11). Round 99.
    SubSeqLayerCharacteristics(SubSeqLayerCharacteristics),
    /// §D.2.13 (2003 draft) — sub_seq_characteristics (payload type 12). Round 99.
    SubSeqCharacteristics(SubSeqCharacteristics),
    /// §D.2.18 — progressive_refinement_segment_start (payload type 16). Round 73.
    ProgressiveRefinementSegmentStart(ProgressiveRefinementSegmentStart),
    /// §D.2.19 — progressive_refinement_segment_end (payload type 17). Round 73.
    ProgressiveRefinementSegmentEnd(ProgressiveRefinementSegmentEnd),
    /// §D.2.20 — motion_constrained_slice_group_set (payload type 18). Round 73.
    MotionConstrainedSliceGroupSet(MotionConstrainedSliceGroupSet),
    /// §D.2.23 — stereo_video_info (payload type 21). Round 73.
    StereoVideoInfo(StereoVideoInfo),
    /// §D.2.33 — content_colour_volume (payload type 149). Round 107.
    ContentColourVolume(ContentColourVolume),
    /// §D.2.34 — ambient_viewing_environment (payload type 148). Round 78.
    AmbientViewingEnvironment(AmbientViewingEnvironment),
    /// §D.2.35.1 — equirectangular_projection (payload type 150). Round 78.
    EquirectangularProjection(EquirectangularProjection),
    /// §D.2.35.2 — cubemap_projection (payload type 151). Round 78.
    CubemapProjection(CubemapProjection),
    /// §D.2.35.3 — sphere_rotation (payload type 154). Round 78.
    SphereRotation(SphereRotation),
    /// §D.2.35.4 — regionwise_packing (payload type 155). Round 113.
    RegionWisePacking(RegionWisePacking),
    /// §D.2.35.5 — omni_viewport (payload type 156). Round 95.
    OmniViewport(OmniViewport),
    /// §D.2.38 — shutter_interval_info (payload type 205). Round 95.
    ShutterIntervalInfo(ShutterIntervalInfo),
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
        7 => Ok(SeiPayload::DecRefPicMarkingRepetition(
            parse_dec_ref_pic_marking_repetition(payload, ctx)?,
        )),
        8 => Ok(SeiPayload::SparePic(parse_spare_pic(payload, ctx)?)),
        9 => Ok(SeiPayload::SceneInfo(parse_scene_info(payload)?)),
        10 => Ok(SeiPayload::SubSeqInfo(parse_sub_seq_info(payload)?)),
        11 => Ok(SeiPayload::SubSeqLayerCharacteristics(
            parse_sub_seq_layer_characteristics(payload)?,
        )),
        12 => Ok(SeiPayload::SubSeqCharacteristics(
            parse_sub_seq_characteristics(payload)?,
        )),
        13 => Ok(SeiPayload::FullFrameFreeze(parse_full_frame_freeze(
            payload,
        )?)),
        14 => Ok(SeiPayload::FullFrameFreezeRelease),
        15 => Ok(SeiPayload::FullFrameSnapshot(parse_full_frame_snapshot(
            payload,
        )?)),
        16 => Ok(SeiPayload::ProgressiveRefinementSegmentStart(
            parse_progressive_refinement_segment_start(payload)?,
        )),
        17 => Ok(SeiPayload::ProgressiveRefinementSegmentEnd(
            parse_progressive_refinement_segment_end(payload)?,
        )),
        18 => Ok(SeiPayload::MotionConstrainedSliceGroupSet(
            parse_motion_constrained_slice_group_set(payload, ctx)?,
        )),
        19 => Ok(SeiPayload::FilmGrainCharacteristics(
            parse_film_grain_characteristics(payload)?,
        )),
        20 => Ok(SeiPayload::DeblockingFilterDisplayPreference(
            parse_deblocking_filter_display_preference(payload)?,
        )),
        21 => Ok(SeiPayload::StereoVideoInfo(parse_stereo_video_info(
            payload,
        )?)),
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
        148 => Ok(SeiPayload::AmbientViewingEnvironment(
            parse_ambient_viewing_environment(payload)?,
        )),
        149 => Ok(SeiPayload::ContentColourVolume(
            parse_content_colour_volume(payload)?,
        )),
        150 => Ok(SeiPayload::EquirectangularProjection(
            parse_equirectangular_projection(payload)?,
        )),
        151 => Ok(SeiPayload::CubemapProjection(parse_cubemap_projection(
            payload,
        )?)),
        154 => Ok(SeiPayload::SphereRotation(parse_sphere_rotation(payload)?)),
        155 => Ok(SeiPayload::RegionWisePacking(parse_regionwise_packing(
            payload,
        )?)),
        156 => Ok(SeiPayload::OmniViewport(parse_omni_viewport(payload)?)),
        205 => Ok(SeiPayload::ShutterIntervalInfo(
            parse_shutter_interval_info(payload)?,
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

    // §D.2.15 — full_frame_freeze: single ue(v) repetition_period.
    #[test]
    fn full_frame_freeze_basic() {
        // ue(7) = "0001000" (7 bits).
        let payload = pack_bits(&[(0b0001000, 7)]);
        let fff = parse_full_frame_freeze(&payload).unwrap();
        assert_eq!(fff.full_frame_freeze_repetition_period, 7);
    }

    #[test]
    fn parse_payload_dispatches_full_frame_freeze() {
        let ctx = SeiContext::default();
        let payload = pack_bits(&[(0b010, 3)]); // ue(1) = 1
        let got = parse_payload(13, &payload, &ctx).unwrap();
        match got {
            SeiPayload::FullFrameFreeze(f) => {
                assert_eq!(f.full_frame_freeze_repetition_period, 1);
            }
            other => panic!("expected FullFrameFreeze, got {:?}", other),
        }
    }

    // §D.2.16 — full_frame_freeze_release: empty payload, dispatch
    // returns the unit variant.
    #[test]
    fn parse_payload_dispatches_full_frame_freeze_release() {
        let ctx = SeiContext::default();
        let got = parse_payload(14, &[], &ctx).unwrap();
        assert_eq!(got, SeiPayload::FullFrameFreezeRelease);
    }

    // §D.2.17 — full_frame_snapshot: single ue(v) snapshot_id.
    #[test]
    fn full_frame_snapshot_basic() {
        // ue(42) = codeNum 42, 42+1 = 43 = 0b101011 (6 bits), so k=5
        // → "00000 1 01011" = "00000101011" (11 bits).
        let payload = pack_bits(&[(0b00000101011, 11)]);
        let snap = parse_full_frame_snapshot(&payload).unwrap();
        assert_eq!(snap.snapshot_id, 42);
    }

    #[test]
    fn parse_payload_dispatches_full_frame_snapshot() {
        let ctx = SeiContext::default();
        let payload = pack_bits(&[(0b010, 3)]); // ue(1) = 1
        let got = parse_payload(15, &payload, &ctx).unwrap();
        match got {
            SeiPayload::FullFrameSnapshot(s) => assert_eq!(s.snapshot_id, 1),
            other => panic!("expected FullFrameSnapshot, got {:?}", other),
        }
    }

    // §D.2.22 — deblocking_filter_display_preference: cancel_flag path.
    #[test]
    fn deblocking_filter_display_preference_cancel() {
        let payload = pack_bits(&[(1, 1)]); // cancel_flag=1
        let p = parse_deblocking_filter_display_preference(&payload).unwrap();
        assert!(p.cancel_flag);
        assert!(!p.display_prior_to_deblocking_preferred_flag);
        assert!(!p.dec_frame_buffering_constraint_flag);
        assert_eq!(p.repetition_period, 0);
    }

    // §D.2.22 — deblocking_filter_display_preference: full body.
    #[test]
    fn deblocking_filter_display_preference_full_body() {
        let payload = pack_bits(&[
            (0, 1),     // cancel_flag = 0
            (1, 1),     // display_prior_to_deblocking_preferred_flag = 1
            (0, 1),     // dec_frame_buffering_constraint_flag = 0
            (0b011, 3), // ue(2) = 2 (repetition_period)
        ]);
        let p = parse_deblocking_filter_display_preference(&payload).unwrap();
        assert!(!p.cancel_flag);
        assert!(p.display_prior_to_deblocking_preferred_flag);
        assert!(!p.dec_frame_buffering_constraint_flag);
        assert_eq!(p.repetition_period, 2);
    }

    #[test]
    fn parse_payload_dispatches_deblocking_filter_display_preference() {
        let ctx = SeiContext::default();
        let payload = pack_bits(&[(1, 1)]); // cancel_flag=1
        let got = parse_payload(20, &payload, &ctx).unwrap();
        match got {
            SeiPayload::DeblockingFilterDisplayPreference(p) => assert!(p.cancel_flag),
            other => panic!(
                "expected DeblockingFilterDisplayPreference, got {:?}",
                other
            ),
        }
    }

    // -----------------------------------------------------------------
    // Round 73 — scene_info / progressive_refinement / stereo_video /
    // motion_constrained_slice_group_set.
    // -----------------------------------------------------------------

    #[test]
    fn scene_info_present_flag_zero_yields_empty_struct() {
        // scene_info_present_flag = 0 → struct is otherwise all-None.
        let payload = pack_bits(&[(0, 1)]);
        let got = parse_scene_info(&payload).unwrap();
        assert!(!got.scene_info_present_flag);
        assert!(got.scene_id.is_none());
        assert!(got.scene_transition_type.is_none());
        assert!(got.second_scene_id.is_none());
    }

    #[test]
    fn scene_info_transition_type_le_3_omits_second_scene_id() {
        // present=1, scene_id=5 (ue=5 → "00110"), transition=2 (ue=2 → "011"),
        // no second_scene_id because transition <= 3.
        let payload = pack_bits(&[
            (1, 1),       // present_flag
            (0b00110, 5), // ue(5) → scene_id=5
            (0b011, 3),   // ue(2) → transition=2
        ]);
        let got = parse_scene_info(&payload).unwrap();
        assert!(got.scene_info_present_flag);
        assert_eq!(got.scene_id, Some(5));
        assert_eq!(got.scene_transition_type, Some(2));
        assert!(got.second_scene_id.is_none());
    }

    #[test]
    fn scene_info_transition_type_gt_3_reads_second_scene_id() {
        // present=1, scene_id=1 (ue=1 → "010"), transition=4 (ue=4 → "00101"),
        // second_scene_id=2 (ue=2 → "011").
        let payload = pack_bits(&[
            (1, 1),
            (0b010, 3),   // scene_id=1
            (0b00101, 5), // transition=4
            (0b011, 3),   // second_scene_id=2
        ]);
        let got = parse_scene_info(&payload).unwrap();
        assert_eq!(got.scene_id, Some(1));
        assert_eq!(got.scene_transition_type, Some(4));
        assert_eq!(got.second_scene_id, Some(2));
    }

    #[test]
    fn parse_payload_dispatches_scene_info() {
        let ctx = SeiContext::default();
        let payload = pack_bits(&[(0, 1)]);
        let got = parse_payload(9, &payload, &ctx).unwrap();
        match got {
            SeiPayload::SceneInfo(s) => assert!(!s.scene_info_present_flag),
            other => panic!("expected SceneInfo, got {other:?}"),
        }
    }

    // --- §D.1.10 spare_pic (payload type 8) ---------------------------

    #[test]
    fn spare_pic_area_idc_0_needs_no_map_units() {
        // target_frame_num=3 (ue → "00100"), spare_field_flag=0,
        // num_spare_pics_minus1=0 (ue → "1") → one spare picture,
        // delta_spare_frame_num=2 (ue → "011"), spare_area_idc=0 (ue → "1").
        // PicSizeInMapUnits is irrelevant for spare_area_idc == 0.
        let payload = pack_bits(&[
            (0b00100, 5), // target_frame_num=3
            (0, 1),       // spare_field_flag=0
            (1, 1),       // num_spare_pics_minus1=0
            (0b011, 3),   // delta_spare_frame_num=2
            (1, 1),       // spare_area_idc=0
        ]);
        let ctx = SeiContext::default(); // pic_size_in_map_units = 0
        let got = parse_spare_pic(&payload, &ctx).unwrap();
        assert_eq!(got.target_frame_num, 3);
        assert!(!got.spare_field_flag);
        assert!(got.target_bottom_field_flag.is_none());
        assert_eq!(got.entries.len(), 1);
        let e = &got.entries[0];
        assert_eq!(e.delta_spare_frame_num, 2);
        assert_eq!(e.spare_area_idc, 0);
        assert!(e.spare_units.is_none());
        assert!(e.zero_run_lengths.is_none());
        assert!(e.spare_bottom_field_flag.is_none());
    }

    #[test]
    fn spare_pic_area_idc_1_reads_one_flag_per_map_unit() {
        // PicSizeInMapUnits=4. target_frame_num=0 (ue → "1"),
        // spare_field_flag=0, num_spare_pics_minus1=0 (ue → "1"),
        // delta_spare_frame_num=0 (ue → "1"), spare_area_idc=1 (ue → "010"),
        // spare_unit_flag[0..4] = 0,1,0,1 → spare,not,spare,not.
        let payload = pack_bits(&[
            (1, 1),     // target_frame_num=0
            (0, 1),     // spare_field_flag=0
            (1, 1),     // num_spare_pics_minus1=0
            (1, 1),     // delta_spare_frame_num=0
            (0b010, 3), // spare_area_idc=1
            (0, 1),     // spare_unit_flag[0]=0 → spare
            (1, 1),     // spare_unit_flag[1]=1 → not spare
            (0, 1),     // spare_unit_flag[2]=0 → spare
            (1, 1),     // spare_unit_flag[3]=1 → not spare
        ]);
        let ctx = SeiContext {
            pic_size_in_map_units: 4,
            ..SeiContext::default()
        };
        let got = parse_spare_pic(&payload, &ctx).unwrap();
        let e = &got.entries[0];
        assert_eq!(e.spare_area_idc, 1);
        assert_eq!(e.spare_units, Some(vec![true, false, true, false]));
        assert!(e.zero_run_lengths.is_none());
    }

    #[test]
    fn spare_pic_area_idc_2_reads_zero_runs_until_map_units() {
        // PicSizeInMapUnits=6. target_frame_num=0 (ue → "1"),
        // spare_field_flag=0, num_spare_pics_minus1=0 (ue → "1"),
        // delta_spare_frame_num=0 (ue → "1"), spare_area_idc=2 (ue → "011").
        // zero_run_length runs: 2 (ue → "011"), 1 (ue → "010").
        //   run 0: mapUnitCnt = 0 + 2 + 1 = 3 (< 6, continue)
        //   run 1: mapUnitCnt = 3 + 1 + 1 = 5 (< 6, continue)
        //   run 2: mapUnitCnt = 5 + 0 + 1 = 6 (>= 6, stop) — zero_run_length=0 (ue → "1")
        let payload = pack_bits(&[
            (1, 1),     // target_frame_num=0
            (0, 1),     // spare_field_flag=0
            (1, 1),     // num_spare_pics_minus1=0
            (1, 1),     // delta_spare_frame_num=0
            (0b011, 3), // spare_area_idc=2
            (0b011, 3), // zero_run_length[0]=2
            (0b010, 3), // zero_run_length[1]=1
            (1, 1),     // zero_run_length[2]=0
        ]);
        let ctx = SeiContext {
            pic_size_in_map_units: 6,
            ..SeiContext::default()
        };
        let got = parse_spare_pic(&payload, &ctx).unwrap();
        let e = &got.entries[0];
        assert_eq!(e.spare_area_idc, 2);
        assert!(e.spare_units.is_none());
        assert_eq!(e.zero_run_lengths, Some(vec![2, 1, 0]));
    }

    #[test]
    fn spare_pic_field_message_reads_bottom_field_flags() {
        // spare_field_flag=1 pulls in target_bottom_field_flag and a
        // per-spare spare_bottom_field_flag. target_frame_num=1 (ue → "010"),
        // spare_field_flag=1, target_bottom_field_flag=1,
        // num_spare_pics_minus1=0 (ue → "1"), delta_spare_frame_num=0 (ue → "1"),
        // spare_bottom_field_flag[0]=0, spare_area_idc=0 (ue → "1").
        let payload = pack_bits(&[
            (0b010, 3), // target_frame_num=1
            (1, 1),     // spare_field_flag=1
            (1, 1),     // target_bottom_field_flag=1
            (1, 1),     // num_spare_pics_minus1=0
            (1, 1),     // delta_spare_frame_num=0
            (0, 1),     // spare_bottom_field_flag[0]=0
            (1, 1),     // spare_area_idc=0
        ]);
        let ctx = SeiContext::default();
        let got = parse_spare_pic(&payload, &ctx).unwrap();
        assert_eq!(got.target_frame_num, 1);
        assert!(got.spare_field_flag);
        assert_eq!(got.target_bottom_field_flag, Some(true));
        assert_eq!(got.entries[0].spare_bottom_field_flag, Some(false));
        assert_eq!(got.entries[0].spare_area_idc, 0);
    }

    #[test]
    fn spare_pic_rejects_num_spare_pics_minus1_over_15() {
        // target_frame_num=0 (ue → "1"), spare_field_flag=0,
        // num_spare_pics_minus1=16 (ue(16) → "000010001", 9 bits) > 15.
        let payload = pack_bits(&[
            (1, 1),           // target_frame_num=0
            (0, 1),           // spare_field_flag=0
            (0b000010001, 9), // num_spare_pics_minus1=16
        ]);
        let ctx = SeiContext::default();
        assert_eq!(
            parse_spare_pic(&payload, &ctx),
            Err(SeiError::SparePicCountTooLarge(16))
        );
    }

    #[test]
    fn spare_pic_rejects_area_idc_over_2() {
        // delta_spare_frame_num=0 then spare_area_idc=3 (ue(3) → "00100").
        let payload = pack_bits(&[
            (1, 1),       // target_frame_num=0
            (0, 1),       // spare_field_flag=0
            (1, 1),       // num_spare_pics_minus1=0
            (1, 1),       // delta_spare_frame_num=0
            (0b00100, 5), // spare_area_idc=3
        ]);
        let ctx = SeiContext::default();
        assert_eq!(
            parse_spare_pic(&payload, &ctx),
            Err(SeiError::SparePicAreaIdcOutOfRange { i: 0, idc: 3 })
        );
    }

    #[test]
    fn spare_pic_area_idc_1_without_map_units_is_rejected() {
        // spare_area_idc=1 but the SeiContext carried PicSizeInMapUnits=0.
        let payload = pack_bits(&[
            (1, 1),     // target_frame_num=0
            (0, 1),     // spare_field_flag=0
            (1, 1),     // num_spare_pics_minus1=0
            (1, 1),     // delta_spare_frame_num=0
            (0b010, 3), // spare_area_idc=1
        ]);
        let ctx = SeiContext::default(); // pic_size_in_map_units = 0
        assert_eq!(
            parse_spare_pic(&payload, &ctx),
            Err(SeiError::SparePicMapUnitsUnknown { i: 0, idc: 1 })
        );
    }

    #[test]
    fn spare_pic_two_entries_mix_methods() {
        // num_spare_pics_minus1=1 → two spare pictures.
        // entry 0: delta=0, spare_area_idc=0 (no map data).
        // entry 1: delta=1 (ue → "010"), spare_area_idc=1, 2 flags (1,0).
        let payload = pack_bits(&[
            (1, 1),     // target_frame_num=0
            (0, 1),     // spare_field_flag=0
            (0b010, 3), // num_spare_pics_minus1=1
            (1, 1),     // delta_spare_frame_num[0]=0
            (1, 1),     // spare_area_idc[0]=0
            (0b010, 3), // delta_spare_frame_num[1]=1
            (0b010, 3), // spare_area_idc[1]=1
            (1, 1),     // spare_unit_flag[1][0]=1 → not spare
            (0, 1),     // spare_unit_flag[1][1]=0 → spare
        ]);
        let ctx = SeiContext {
            pic_size_in_map_units: 2,
            ..SeiContext::default()
        };
        let got = parse_spare_pic(&payload, &ctx).unwrap();
        assert_eq!(got.entries.len(), 2);
        assert_eq!(got.entries[0].spare_area_idc, 0);
        assert!(got.entries[0].spare_units.is_none());
        assert_eq!(got.entries[1].delta_spare_frame_num, 1);
        assert_eq!(got.entries[1].spare_units, Some(vec![false, true]));
    }

    #[test]
    fn parse_payload_dispatches_spare_pic() {
        // Same as spare_pic_area_idc_0_needs_no_map_units, via parse_payload(8).
        let payload = pack_bits(&[
            (0b00100, 5), // target_frame_num=3
            (0, 1),       // spare_field_flag=0
            (1, 1),       // num_spare_pics_minus1=0
            (0b011, 3),   // delta_spare_frame_num=2
            (1, 1),       // spare_area_idc=0
        ]);
        let ctx = SeiContext::default();
        match parse_payload(8, &payload, &ctx).unwrap() {
            SeiPayload::SparePic(s) => {
                assert_eq!(s.target_frame_num, 3);
                assert_eq!(s.entries.len(), 1);
                assert_eq!(s.entries[0].spare_area_idc, 0);
            }
            other => panic!("expected SparePic, got {other:?}"),
        }
    }

    // --- §D.1.9 dec_ref_pic_marking_repetition (payload type 7) -------

    #[test]
    fn dec_ref_pic_marking_repetition_idr_frame_only() {
        // frame_mbs_only_flag = true ⇒ no field flags.
        // original_idr_flag = 1, original_frame_num = 0 (ue → "1"),
        // IDR branch: no_output_of_prior_pics_flag = 1, long_term_reference_flag = 0.
        let payload = pack_bits(&[
            (1, 1), // original_idr_flag
            (1, 1), // original_frame_num = 0 (ue)
            (1, 1), // no_output_of_prior_pics_flag
            (0, 1), // long_term_reference_flag
        ]);
        let ctx = SeiContext::default(); // frame_mbs_only_flag = true
        let got = parse_dec_ref_pic_marking_repetition(&payload, &ctx).unwrap();
        assert!(got.original_idr_flag);
        assert_eq!(got.original_frame_num, 0);
        assert_eq!(got.original_field_pic_flag, None);
        assert_eq!(got.original_bottom_field_flag, None);
        assert!(got.dec_ref_pic_marking.no_output_of_prior_pics_flag);
        assert!(!got.dec_ref_pic_marking.long_term_reference_flag);
        assert_eq!(got.dec_ref_pic_marking.adaptive_marking, None);
    }

    #[test]
    fn dec_ref_pic_marking_repetition_non_idr_adaptive_mmco() {
        // original_idr_flag = 0, original_frame_num = 5 (ue → "00110"),
        // non-IDR branch: adaptive_ref_pic_marking_mode_flag = 1, then
        //   mmco = 1 (ue → "010"), difference_of_pic_nums_minus1 = 3 (ue → "00100"),
        //   mmco = 3 (ue → "00100"), difference = 0 (ue → "1"),
        //                            long_term_frame_idx = 2 (ue → "011"),
        //   mmco = 0 (ue → "1")  [terminator].
        let payload = pack_bits(&[
            (0, 1),       // original_idr_flag
            (0b00110, 5), // original_frame_num = 5
            (1, 1),       // adaptive_ref_pic_marking_mode_flag
            (0b010, 3),   // mmco = 1
            (0b00100, 5), // difference_of_pic_nums_minus1 = 3
            (0b00100, 5), // mmco = 3
            (1, 1),       // difference_of_pic_nums_minus1 = 0
            (0b011, 3),   // long_term_frame_idx = 2
            (1, 1),       // mmco = 0 (terminator)
        ]);
        let ctx = SeiContext::default();
        let got = parse_dec_ref_pic_marking_repetition(&payload, &ctx).unwrap();
        assert!(!got.original_idr_flag);
        assert_eq!(got.original_frame_num, 5);
        let ops = got.dec_ref_pic_marking.adaptive_marking.unwrap();
        assert_eq!(
            ops,
            vec![MmcoOp::MarkShortTermUnused(3), MmcoOp::AssignLongTerm(0, 2)]
        );
    }

    #[test]
    fn dec_ref_pic_marking_repetition_non_idr_sliding_window() {
        // original_idr_flag = 0, original_frame_num = 2 (ue → "011"),
        // adaptive_ref_pic_marking_mode_flag = 0 ⇒ sliding window (no ops).
        let payload = pack_bits(&[
            (0, 1),     // original_idr_flag
            (0b011, 3), // original_frame_num = 2
            (0, 1),     // adaptive_ref_pic_marking_mode_flag
        ]);
        let ctx = SeiContext::default();
        let got = parse_dec_ref_pic_marking_repetition(&payload, &ctx).unwrap();
        assert!(!got.original_idr_flag);
        assert_eq!(got.original_frame_num, 2);
        assert_eq!(got.dec_ref_pic_marking.adaptive_marking, None);
    }

    #[test]
    fn dec_ref_pic_marking_repetition_field_with_bottom_flag() {
        // frame_mbs_only_flag = false ⇒ field flags present.
        // original_idr_flag = 1, original_frame_num = 0 (ue → "1"),
        // original_field_pic_flag = 1, original_bottom_field_flag = 1,
        // IDR branch: no_output = 0, long_term_reference_flag = 1.
        let payload = pack_bits(&[
            (1, 1), // original_idr_flag
            (1, 1), // original_frame_num = 0
            (1, 1), // original_field_pic_flag
            (1, 1), // original_bottom_field_flag
            (0, 1), // no_output_of_prior_pics_flag
            (1, 1), // long_term_reference_flag
        ]);
        let ctx = SeiContext {
            frame_mbs_only_flag: false,
            ..SeiContext::default()
        };
        let got = parse_dec_ref_pic_marking_repetition(&payload, &ctx).unwrap();
        assert_eq!(got.original_field_pic_flag, Some(true));
        assert_eq!(got.original_bottom_field_flag, Some(true));
        assert!(!got.dec_ref_pic_marking.no_output_of_prior_pics_flag);
        assert!(got.dec_ref_pic_marking.long_term_reference_flag);
    }

    #[test]
    fn dec_ref_pic_marking_repetition_field_no_bottom_flag() {
        // frame_mbs_only_flag = false, original_field_pic_flag = 0 ⇒
        // original_bottom_field_flag absent.
        // original_idr_flag = 0, original_frame_num = 1 (ue → "010"),
        // original_field_pic_flag = 0, adaptive = 0.
        let payload = pack_bits(&[
            (0, 1),     // original_idr_flag
            (0b010, 3), // original_frame_num = 1
            (0, 1),     // original_field_pic_flag
            (0, 1),     // adaptive_ref_pic_marking_mode_flag
        ]);
        let ctx = SeiContext {
            frame_mbs_only_flag: false,
            ..SeiContext::default()
        };
        let got = parse_dec_ref_pic_marking_repetition(&payload, &ctx).unwrap();
        assert_eq!(got.original_field_pic_flag, Some(false));
        assert_eq!(got.original_bottom_field_flag, None);
        assert_eq!(got.dec_ref_pic_marking.adaptive_marking, None);
    }

    #[test]
    fn dec_ref_pic_marking_repetition_all_mmco_ops() {
        // Exercise every MMCO 1..=6 in one adaptive loop, frame-only.
        // original_idr_flag = 0, original_frame_num = 0 (ue → "1"),
        // adaptive = 1, then:
        //   mmco 1 (ue "010") + diff 0 (ue "1")
        //   mmco 2 (ue "011") + long_term_pic_num 1 (ue "010")
        //   mmco 3 (ue "00100") + diff 1 (ue "010") + ltfi 0 (ue "1")
        //   mmco 4 (ue "00101") + max_lt_idx_plus1 2 (ue "011")
        //   mmco 5 (ue "00110")
        //   mmco 6 (ue "00111") + ltfi 3 (ue "00100")
        //   mmco 0 (ue "1")
        let payload = pack_bits(&[
            (0, 1),       // original_idr_flag
            (1, 1),       // original_frame_num = 0
            (1, 1),       // adaptive_ref_pic_marking_mode_flag
            (0b010, 3),   // mmco 1
            (1, 1),       // difference_of_pic_nums_minus1 = 0
            (0b011, 3),   // mmco 2
            (0b010, 3),   // long_term_pic_num = 1
            (0b00100, 5), // mmco 3
            (0b010, 3),   // difference_of_pic_nums_minus1 = 1
            (1, 1),       // long_term_frame_idx = 0
            (0b00101, 5), // mmco 4
            (0b011, 3),   // max_long_term_frame_idx_plus1 = 2
            (0b00110, 5), // mmco 5
            (0b00111, 5), // mmco 6
            (0b00100, 5), // long_term_frame_idx = 3
            (1, 1),       // mmco 0 (terminator)
        ]);
        let ctx = SeiContext::default();
        let got = parse_dec_ref_pic_marking_repetition(&payload, &ctx).unwrap();
        let ops = got.dec_ref_pic_marking.adaptive_marking.unwrap();
        assert_eq!(
            ops,
            vec![
                MmcoOp::MarkShortTermUnused(0),
                MmcoOp::MarkLongTermUnused(1),
                MmcoOp::AssignLongTerm(1, 0),
                MmcoOp::SetMaxLongTermIdx(2),
                MmcoOp::MarkAllUnused,
                MmcoOp::AssignCurrentLongTerm(3),
            ]
        );
    }

    #[test]
    fn dec_ref_pic_marking_repetition_mmco_out_of_range_rejected() {
        // mmco = 7 is out of range (Table 7-9 defines 0..=6).
        // original_idr_flag = 0, original_frame_num = 0 (ue "1"),
        // adaptive = 1, mmco = 7 (ue → "0001000").
        let payload = pack_bits(&[
            (0, 1),         // original_idr_flag
            (1, 1),         // original_frame_num = 0
            (1, 1),         // adaptive_ref_pic_marking_mode_flag
            (0b0001000, 7), // mmco = 7
        ]);
        let ctx = SeiContext::default();
        let err = parse_dec_ref_pic_marking_repetition(&payload, &ctx).unwrap_err();
        assert_eq!(err, SeiError::DecRefPicMarkingRepetitionMmcoOutOfRange(7));
    }

    #[test]
    fn parse_payload_dispatches_dec_ref_pic_marking_repetition() {
        // Same body as dec_ref_pic_marking_repetition_idr_frame_only,
        // routed via parse_payload(7).
        let payload = pack_bits(&[(1, 1), (1, 1), (1, 1), (0, 1)]);
        let ctx = SeiContext::default();
        match parse_payload(7, &payload, &ctx).unwrap() {
            SeiPayload::DecRefPicMarkingRepetition(d) => {
                assert!(d.original_idr_flag);
                assert_eq!(d.original_frame_num, 0);
                assert!(d.dec_ref_pic_marking.no_output_of_prior_pics_flag);
            }
            other => panic!("expected DecRefPicMarkingRepetition, got {other:?}"),
        }
    }

    // --- §D.1.11 sub_seq_info (payload type 10) -----------------------

    #[test]
    fn sub_seq_info_without_frame_num() {
        // layer=2 (ue → "011"), id=5 (ue → "00110"), first=1, leading=0,
        // last=1, frame_num_flag=0 → no sub_seq_frame_num.
        let payload = pack_bits(&[
            (0b011, 3),   // sub_seq_layer_num = 2
            (0b00110, 5), // sub_seq_id = 5
            (1, 1),       // first_ref_pic_flag
            (0, 1),       // leading_non_ref_pic_flag
            (1, 1),       // last_pic_flag
            (0, 1),       // sub_seq_frame_num_flag
        ]);
        let got = parse_sub_seq_info(&payload).unwrap();
        assert_eq!(got.sub_seq_layer_num, 2);
        assert_eq!(got.sub_seq_id, 5);
        assert!(got.first_ref_pic_flag);
        assert!(!got.leading_non_ref_pic_flag);
        assert!(got.last_pic_flag);
        assert!(got.sub_seq_frame_num.is_none());
    }

    #[test]
    fn sub_seq_info_with_frame_num() {
        // layer=0 (ue → "1"), id=0 (ue → "1"), first=0, leading=1, last=0,
        // frame_num_flag=1, frame_num=3 (ue → "00100").
        let payload = pack_bits(&[
            (1, 1),       // sub_seq_layer_num = 0
            (1, 1),       // sub_seq_id = 0
            (0, 1),       // first_ref_pic_flag
            (1, 1),       // leading_non_ref_pic_flag
            (0, 1),       // last_pic_flag
            (1, 1),       // sub_seq_frame_num_flag
            (0b00100, 5), // sub_seq_frame_num = 3
        ]);
        let got = parse_sub_seq_info(&payload).unwrap();
        assert_eq!(got.sub_seq_layer_num, 0);
        assert_eq!(got.sub_seq_id, 0);
        assert!(!got.first_ref_pic_flag);
        assert!(got.leading_non_ref_pic_flag);
        assert!(!got.last_pic_flag);
        assert_eq!(got.sub_seq_frame_num, Some(3));
    }

    #[test]
    fn sub_seq_info_layer_num_out_of_range_rejected() {
        // sub_seq_layer_num = 256 is out of range (must be < 256).
        // ue(256) = 8 leading zeros + "1" + 8-bit (256-255=... actually
        // 256+1 = 257 = 0b1_0000_0001 → 8 leading zeros, then 9-bit value).
        let payload = pack_bits(&[(0b1_0000_0001, 17)]);
        let err = parse_sub_seq_info(&payload).unwrap_err();
        assert_eq!(err, SeiError::SubSeqLayerNumTooLarge(256));
    }

    #[test]
    fn parse_payload_dispatches_sub_seq_info() {
        let ctx = SeiContext::default();
        let payload = pack_bits(&[(1, 1), (1, 1), (1, 1), (0, 1), (0, 1), (0, 1)]);
        match parse_payload(10, &payload, &ctx).unwrap() {
            SeiPayload::SubSeqInfo(s) => {
                assert_eq!(s.sub_seq_layer_num, 0);
                assert!(s.first_ref_pic_flag);
            }
            other => panic!("expected SubSeqInfo, got {other:?}"),
        }
    }

    // --- §D.1.12 sub_seq_layer_characteristics (payload type 11) ------

    #[test]
    fn sub_seq_layer_characteristics_two_layers() {
        // num_sub_seq_layers_minus1 = 1 (ue → "010") → 2 entries.
        // layer 0: accurate=1, bit_rate=0x1234, frame_rate=0x0100
        // layer 1: accurate=0, bit_rate=0x0000, frame_rate=0x00FF
        let payload = pack_bits(&[
            (0b010, 3),   // num_sub_seq_layers_minus1 = 1
            (1, 1),       // accurate_statistics_flag[0]
            (0x1234, 16), // average_bit_rate[0]
            (0x0100, 16), // average_frame_rate[0]
            (0, 1),       // accurate_statistics_flag[1]
            (0x0000, 16), // average_bit_rate[1]
            (0x00FF, 16), // average_frame_rate[1]
        ]);
        let got = parse_sub_seq_layer_characteristics(&payload).unwrap();
        assert_eq!(got.layers.len(), 2);
        assert!(got.layers[0].accurate_statistics_flag);
        assert_eq!(got.layers[0].average_bit_rate, 0x1234);
        assert_eq!(got.layers[0].average_frame_rate, 0x0100);
        assert!(!got.layers[1].accurate_statistics_flag);
        assert_eq!(got.layers[1].average_bit_rate, 0);
        assert_eq!(got.layers[1].average_frame_rate, 0x00FF);
    }

    #[test]
    fn parse_payload_dispatches_sub_seq_layer_characteristics() {
        let ctx = SeiContext::default();
        // single layer: minus1=0 (ue → "1"), accurate=0, rates 0.
        let payload = pack_bits(&[(1, 1), (0, 1), (0, 16), (0, 16)]);
        match parse_payload(11, &payload, &ctx).unwrap() {
            SeiPayload::SubSeqLayerCharacteristics(s) => {
                assert_eq!(s.layers.len(), 1);
            }
            other => panic!("expected SubSeqLayerCharacteristics, got {other:?}"),
        }
    }

    // --- §D.1.13 sub_seq_characteristics (payload type 12) ------------

    #[test]
    fn sub_seq_characteristics_full() {
        // layer=1 (ue → "010"), id=2 (ue → "011"),
        // duration_flag=1, duration=0xDEADBEEF,
        // average_rate_flag=1, accurate=1, bit_rate=0xAAAA, frame_rate=0x5555,
        // num_referenced_subseqs=1 (ue → "010"),
        //   ref_layer=0 (ue → "1"), ref_id=3 (ue → "00100"), direction=1.
        let payload = pack_bits(&[
            (0b010, 3),       // sub_seq_layer_num = 1
            (0b011, 3),       // sub_seq_id = 2
            (1, 1),           // duration_flag
            (0xDEADBEEF, 32), // sub_seq_duration
            (1, 1),           // average_rate_flag
            (1, 1),           // accurate_statistics_flag
            (0xAAAA, 16),     // average_bit_rate
            (0x5555, 16),     // average_frame_rate
            (0b010, 3),       // num_referenced_subseqs = 1
            (1, 1),           // ref_sub_seq_layer_num = 0
            (0b00100, 5),     // ref_sub_seq_id = 3
            (1, 1),           // ref_sub_seq_direction = 1
        ]);
        let got = parse_sub_seq_characteristics(&payload).unwrap();
        assert_eq!(got.sub_seq_layer_num, 1);
        assert_eq!(got.sub_seq_id, 2);
        assert_eq!(got.sub_seq_duration, Some(0xDEADBEEF));
        let rate = got.average_rate.expect("average_rate present");
        assert!(rate.accurate_statistics_flag);
        assert_eq!(rate.average_bit_rate, 0xAAAA);
        assert_eq!(rate.average_frame_rate, 0x5555);
        assert_eq!(got.referenced.len(), 1);
        assert_eq!(got.referenced[0].ref_sub_seq_layer_num, 0);
        assert_eq!(got.referenced[0].ref_sub_seq_id, 3);
        assert!(got.referenced[0].ref_sub_seq_direction);
    }

    #[test]
    fn sub_seq_characteristics_minimal() {
        // layer=0, id=0, duration_flag=0, average_rate_flag=0,
        // num_referenced_subseqs=0.
        let payload = pack_bits(&[
            (1, 1), // sub_seq_layer_num = 0
            (1, 1), // sub_seq_id = 0
            (0, 1), // duration_flag
            (0, 1), // average_rate_flag
            (1, 1), // num_referenced_subseqs = 0
        ]);
        let got = parse_sub_seq_characteristics(&payload).unwrap();
        assert_eq!(got.sub_seq_layer_num, 0);
        assert_eq!(got.sub_seq_id, 0);
        assert!(got.sub_seq_duration.is_none());
        assert!(got.average_rate.is_none());
        assert!(got.referenced.is_empty());
    }

    #[test]
    fn parse_payload_dispatches_sub_seq_characteristics() {
        let ctx = SeiContext::default();
        let payload = pack_bits(&[(1, 1), (1, 1), (0, 1), (0, 1), (1, 1)]);
        match parse_payload(12, &payload, &ctx).unwrap() {
            SeiPayload::SubSeqCharacteristics(s) => {
                assert_eq!(s.sub_seq_layer_num, 0);
                assert!(s.referenced.is_empty());
            }
            other => panic!("expected SubSeqCharacteristics, got {other:?}"),
        }
    }

    #[test]
    fn progressive_refinement_segment_start_basic() {
        // id=2 (ue=2 → "011"), steps_minus1=0 (ue=0 → "1").
        let payload = pack_bits(&[(0b011, 3), (1, 1)]);
        let got = parse_progressive_refinement_segment_start(&payload).unwrap();
        assert_eq!(got.progressive_refinement_id, 2);
        assert_eq!(got.num_refinement_steps_minus1, 0);
    }

    #[test]
    fn progressive_refinement_segment_end_basic() {
        // id=3 (ue=3 → "00100").
        let payload = pack_bits(&[(0b00100, 5)]);
        let got = parse_progressive_refinement_segment_end(&payload).unwrap();
        assert_eq!(got.progressive_refinement_id, 3);
    }

    #[test]
    fn parse_payload_dispatches_progressive_refinement() {
        let ctx = SeiContext::default();
        let payload_start = pack_bits(&[(1, 1), (1, 1)]); // id=0, steps=0
        match parse_payload(16, &payload_start, &ctx).unwrap() {
            SeiPayload::ProgressiveRefinementSegmentStart(p) => {
                assert_eq!(p.progressive_refinement_id, 0);
                assert_eq!(p.num_refinement_steps_minus1, 0);
            }
            other => panic!("expected ProgressiveRefinementSegmentStart, got {other:?}"),
        }
        let payload_end = pack_bits(&[(1, 1)]); // id=0
        match parse_payload(17, &payload_end, &ctx).unwrap() {
            SeiPayload::ProgressiveRefinementSegmentEnd(p) => {
                assert_eq!(p.progressive_refinement_id, 0)
            }
            other => panic!("expected ProgressiveRefinementSegmentEnd, got {other:?}"),
        }
    }

    #[test]
    fn stereo_video_info_field_views_path() {
        // field_views=1 → reads top_field_is_left_view_flag (1) then the
        // two self_contained flags (1, 0). 5 bits, padded to 1 byte.
        let payload = pack_bits(&[(1, 1), (1, 1), (1, 1), (0, 1)]);
        let got = parse_stereo_video_info(&payload).unwrap();
        assert!(got.field_views_flag);
        assert_eq!(got.top_field_is_left_view_flag, Some(true));
        assert!(got.current_frame_is_left_view_flag.is_none());
        assert!(got.next_frame_is_second_view_flag.is_none());
        assert!(got.left_view_self_contained_flag);
        assert!(!got.right_view_self_contained_flag);
    }

    #[test]
    fn stereo_video_info_frame_views_path() {
        // field_views=0 → reads current_frame_is_left=1, next_frame_is_second=0,
        // then left_self=1, right_self=1.
        let payload = pack_bits(&[(0, 1), (1, 1), (0, 1), (1, 1), (1, 1)]);
        let got = parse_stereo_video_info(&payload).unwrap();
        assert!(!got.field_views_flag);
        assert!(got.top_field_is_left_view_flag.is_none());
        assert_eq!(got.current_frame_is_left_view_flag, Some(true));
        assert_eq!(got.next_frame_is_second_view_flag, Some(false));
        assert!(got.left_view_self_contained_flag);
        assert!(got.right_view_self_contained_flag);
    }

    #[test]
    fn parse_payload_dispatches_stereo_video_info() {
        let ctx = SeiContext::default();
        let payload = pack_bits(&[(1, 1), (0, 1), (1, 1), (1, 1)]);
        match parse_payload(21, &payload, &ctx).unwrap() {
            SeiPayload::StereoVideoInfo(s) => {
                assert!(s.field_views_flag);
                assert_eq!(s.top_field_is_left_view_flag, Some(false));
            }
            other => panic!("expected StereoVideoInfo, got {other:?}"),
        }
    }

    #[test]
    fn motion_constrained_slice_group_set_single_group_collapses_width() {
        // num_slice_groups = 1 → slice_group_id is 0 bits wide, so we
        // only consume:
        //   num_slice_groups_in_set_minus1 = 0 (ue=0 → "1") = 1 bit
        //   exact_sample_value_match_flag = 1
        //   pan_scan_rect_flag = 0
        // Total = 3 bits.
        let payload = pack_bits(&[(1, 1), (1, 1), (0, 1)]);
        let ctx = SeiContext {
            num_slice_groups: 1,
            ..SeiContext::default()
        };
        let got = parse_motion_constrained_slice_group_set(&payload, &ctx).unwrap();
        assert_eq!(got.slice_group_ids, vec![0]);
        assert!(got.exact_sample_value_match_flag);
        assert!(!got.pan_scan_rect_flag);
        assert!(got.pan_scan_rect_id.is_none());
    }

    #[test]
    fn motion_constrained_slice_group_set_multi_group_reads_u_v() {
        // num_slice_groups = 4 → width = Ceil(Log2(4)) = 2 bits per id.
        //   num_slice_groups_in_set_minus1 = 1 (ue=1 → "010") = 3 bits.
        //   slice_group_id[0] = 1 (2 bits), slice_group_id[1] = 3 (2 bits).
        //   exact_sample_value_match_flag = 0.
        //   pan_scan_rect_flag = 1.
        //   pan_scan_rect_id = 5 (ue=5 → "00110") = 5 bits.
        let payload = pack_bits(&[
            (0b010, 3),   // count_minus1 = 1
            (0b01, 2),    // id[0] = 1
            (0b11, 2),    // id[1] = 3
            (0, 1),       // exact_match = 0
            (1, 1),       // pan_scan_flag = 1
            (0b00110, 5), // pan_scan_id = 5
        ]);
        let ctx = SeiContext {
            num_slice_groups: 4,
            ..SeiContext::default()
        };
        let got = parse_motion_constrained_slice_group_set(&payload, &ctx).unwrap();
        assert_eq!(got.slice_group_ids, vec![1, 3]);
        assert!(!got.exact_sample_value_match_flag);
        assert!(got.pan_scan_rect_flag);
        assert_eq!(got.pan_scan_rect_id, Some(5));
    }

    #[test]
    fn parse_payload_dispatches_motion_constrained_slice_group_set() {
        let payload = pack_bits(&[(1, 1), (0, 1), (0, 1)]);
        let ctx = SeiContext::default();
        match parse_payload(18, &payload, &ctx).unwrap() {
            SeiPayload::MotionConstrainedSliceGroupSet(s) => {
                assert_eq!(s.slice_group_ids, vec![0]);
                assert!(!s.exact_sample_value_match_flag);
            }
            other => panic!("expected MotionConstrainedSliceGroupSet, got {other:?}"),
        }
    }

    // ----- Round 78 — ambient_viewing_environment / equirectangular_projection /
    // cubemap_projection / sphere_rotation -----

    // Helper: encode a u32 / u16 big-endian to a Vec<u8>.
    fn u32_be(v: u32) -> [u8; 4] {
        v.to_be_bytes()
    }
    fn u16_be(v: u16) -> [u8; 2] {
        v.to_be_bytes()
    }
    fn i32_be(v: i32) -> [u8; 4] {
        v.to_be_bytes()
    }

    #[test]
    fn ambient_viewing_environment_bt2035_d65() {
        // §D.2.34 BT.2035 example: illuminance=100_000, D65 chromaticity
        // (15_635, 16_450).
        let mut payload = Vec::new();
        payload.extend_from_slice(&u32_be(100_000));
        payload.extend_from_slice(&u16_be(15_635));
        payload.extend_from_slice(&u16_be(16_450));
        let got = parse_ambient_viewing_environment(&payload).unwrap();
        assert_eq!(got.ambient_illuminance, 100_000);
        assert_eq!(got.ambient_light_x, 15_635);
        assert_eq!(got.ambient_light_y, 16_450);
    }

    #[test]
    fn ambient_viewing_environment_zero_illuminance_rejected() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&u32_be(0));
        payload.extend_from_slice(&u16_be(0));
        payload.extend_from_slice(&u16_be(0));
        let err = parse_ambient_viewing_environment(&payload).unwrap_err();
        assert_eq!(err, SeiError::AmbientIlluminanceZero);
    }

    #[test]
    fn ambient_viewing_environment_chromaticity_out_of_range_rejected() {
        // ambient_light_x = 50_001 (just outside 0..=50_000).
        let mut payload = Vec::new();
        payload.extend_from_slice(&u32_be(1));
        payload.extend_from_slice(&u16_be(50_001));
        payload.extend_from_slice(&u16_be(0));
        let err = parse_ambient_viewing_environment(&payload).unwrap_err();
        assert_eq!(err, SeiError::AmbientLightOutOfRange { x: 50_001, y: 0 });
    }

    #[test]
    fn parse_payload_dispatches_ambient_viewing_environment() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&u32_be(50_000));
        payload.extend_from_slice(&u16_be(14_155));
        payload.extend_from_slice(&u16_be(14_855));
        let ctx = SeiContext::default();
        match parse_payload(148, &payload, &ctx).unwrap() {
            SeiPayload::AmbientViewingEnvironment(a) => {
                assert_eq!(a.ambient_illuminance, 50_000);
                assert_eq!(a.ambient_light_x, 14_155);
                assert_eq!(a.ambient_light_y, 14_855);
            }
            other => panic!("expected AmbientViewingEnvironment, got {other:?}"),
        }
    }

    // ----- Round 107 — content_colour_volume (payload type 149) -----

    // Helper: build the one-byte content_colour_volume header given the
    // cancel flag, persistence flag, and the four "present" flags. The
    // reserved 2 bits are written as 0. The header is exactly 8 bits, so
    // any subsequent i(32) / u(32) fields stay byte-aligned and can be
    // appended big-endian.
    fn ccv_header(cancel: u8, persistence: u8, prim: u8, min: u8, max: u8, avg: u8) -> Vec<u8> {
        pack_bits(&[
            (cancel as u64, 1),
            (persistence as u64, 1),
            (prim as u64, 1),
            (min as u64, 1),
            (max as u64, 1),
            (avg as u64, 1),
            (0, 2), // ccv_reserved_zero_2bits
        ])
    }

    #[test]
    fn content_colour_volume_cancel() {
        // cancel_flag = 1 → no body; only the first bit is consumed but
        // the header byte is fully padded.
        let payload = pack_bits(&[(1, 1)]);
        let got = parse_content_colour_volume(&payload).unwrap();
        assert!(got.cancel_flag);
        assert!(got.body.is_none());
    }

    #[test]
    fn content_colour_volume_all_present_bt2020() {
        // cancel=0, persistence=1, all four present. Primaries are the
        // suggested green/blue/red ordering using BT.2020-style coords
        // scaled to 0.00002 increments; luminance min<=avg<=max.
        let mut payload = ccv_header(0, 1, 1, 1, 1, 1);
        // green (0): x=8500, y=39850 ; blue (1): x=6550, y=2300 ;
        // red (2): x=35400, y=14600  (illustrative, within range).
        for &(x, y) in &[(8_500i32, 39_850i32), (6_550, 2_300), (35_400, 14_600)] {
            payload.extend_from_slice(&i32_be(x));
            payload.extend_from_slice(&i32_be(y));
        }
        payload.extend_from_slice(&u32_be(1)); // min
        payload.extend_from_slice(&u32_be(100_000_000)); // max
        payload.extend_from_slice(&u32_be(50_000_000)); // avg
        let got = parse_content_colour_volume(&payload).unwrap();
        assert!(!got.cancel_flag);
        let body = got.body.expect("body present");
        assert!(body.persistence_flag);
        let prims = body.primaries.expect("primaries present");
        assert_eq!(
            prims[0],
            ContentColourVolumePrimary {
                x: 8_500,
                y: 39_850
            }
        );
        assert_eq!(prims[1], ContentColourVolumePrimary { x: 6_550, y: 2_300 });
        assert_eq!(
            prims[2],
            ContentColourVolumePrimary {
                x: 35_400,
                y: 14_600
            }
        );
        assert_eq!(body.min_luminance_value, Some(1));
        assert_eq!(body.max_luminance_value, Some(100_000_000));
        assert_eq!(body.avg_luminance_value, Some(50_000_000));
    }

    #[test]
    fn content_colour_volume_luminance_only() {
        // No primaries; only max luminance present. Exercises the
        // "some present flags 0" branch and a single-field body.
        let mut payload = ccv_header(0, 0, 0, 0, 1, 0);
        payload.extend_from_slice(&u32_be(42_000_000));
        let got = parse_content_colour_volume(&payload).unwrap();
        let body = got.body.expect("body present");
        assert!(!body.persistence_flag);
        assert!(body.primaries.is_none());
        assert!(body.min_luminance_value.is_none());
        assert_eq!(body.max_luminance_value, Some(42_000_000));
        assert!(body.avg_luminance_value.is_none());
    }

    #[test]
    fn content_colour_volume_negative_primary_coord() {
        // A negative primary coordinate inside [-5_000_000, 5_000_000]
        // round-trips through i(32) two's-complement decoding.
        let mut payload = ccv_header(0, 0, 1, 0, 0, 0);
        for &(x, y) in &[
            (-1_000_000i32, 2_000_000i32),
            (0, 0),
            (5_000_000, -5_000_000),
        ] {
            payload.extend_from_slice(&i32_be(x));
            payload.extend_from_slice(&i32_be(y));
        }
        let got = parse_content_colour_volume(&payload).unwrap();
        let prims = got.body.unwrap().primaries.unwrap();
        assert_eq!(
            prims[0],
            ContentColourVolumePrimary {
                x: -1_000_000,
                y: 2_000_000
            }
        );
        assert_eq!(
            prims[2],
            ContentColourVolumePrimary {
                x: 5_000_000,
                y: -5_000_000
            }
        );
    }

    #[test]
    fn content_colour_volume_no_components_rejected() {
        // §D.2.33: all four present flags equal 0 is a conformance
        // violation.
        let payload = ccv_header(0, 1, 0, 0, 0, 0);
        let err = parse_content_colour_volume(&payload).unwrap_err();
        assert_eq!(err, SeiError::ContentColourVolumeNoComponents);
    }

    #[test]
    fn content_colour_volume_primary_out_of_range_rejected() {
        // ccv_primaries_x[0] = 5_000_001, just past the upper bound.
        let mut payload = ccv_header(0, 0, 1, 0, 0, 0);
        payload.extend_from_slice(&i32_be(5_000_001));
        payload.extend_from_slice(&i32_be(0));
        // Remaining primary coords to keep the read in-bounds (the
        // error fires on c == 0 before these matter, but provide them
        // anyway).
        for _ in 0..4 {
            payload.extend_from_slice(&i32_be(0));
        }
        let err = parse_content_colour_volume(&payload).unwrap_err();
        assert_eq!(
            err,
            SeiError::ContentColourVolumePrimaryOutOfRange {
                c: 0,
                x: 5_000_001,
                y: 0
            }
        );
    }

    #[test]
    fn content_colour_volume_luminance_order_min_gt_max_rejected() {
        // min present (100), max present (50): min > max violates §D.2.33.
        let mut payload = ccv_header(0, 0, 0, 1, 1, 0);
        payload.extend_from_slice(&u32_be(100)); // min
        payload.extend_from_slice(&u32_be(50)); // max
        let err = parse_content_colour_volume(&payload).unwrap_err();
        assert_eq!(
            err,
            SeiError::ContentColourVolumeLuminanceOrder {
                min: 100,
                avg: 100,
                max: 50
            }
        );
    }

    #[test]
    fn content_colour_volume_luminance_order_min_gt_avg_rejected() {
        // min (10) > avg (5): violates §D.2.33 even with no max.
        let mut payload = ccv_header(0, 0, 0, 1, 0, 1);
        payload.extend_from_slice(&u32_be(10)); // min
        payload.extend_from_slice(&u32_be(5)); // avg
        let err = parse_content_colour_volume(&payload).unwrap_err();
        assert_eq!(
            err,
            SeiError::ContentColourVolumeLuminanceOrder {
                min: 10,
                avg: 5,
                max: u64::MAX
            }
        );
    }

    #[test]
    fn content_colour_volume_luminance_order_avg_gt_max_rejected() {
        // avg (80) > max (60): violates §D.2.33 with no min present.
        let mut payload = ccv_header(0, 0, 0, 0, 1, 1);
        payload.extend_from_slice(&u32_be(60)); // max
        payload.extend_from_slice(&u32_be(80)); // avg
        let err = parse_content_colour_volume(&payload).unwrap_err();
        assert_eq!(
            err,
            SeiError::ContentColourVolumeLuminanceOrder {
                min: 0,
                avg: 80,
                max: 60
            }
        );
    }

    #[test]
    fn parse_payload_dispatches_content_colour_volume() {
        let mut payload = ccv_header(0, 1, 0, 0, 1, 0);
        payload.extend_from_slice(&u32_be(10_000_000));
        let ctx = SeiContext::default();
        match parse_payload(149, &payload, &ctx).unwrap() {
            SeiPayload::ContentColourVolume(c) => {
                assert!(!c.cancel_flag);
                let body = c.body.unwrap();
                assert!(body.persistence_flag);
                assert_eq!(body.max_luminance_value, Some(10_000_000));
            }
            other => panic!("expected ContentColourVolume, got {other:?}"),
        }
    }

    #[test]
    fn equirectangular_projection_cancel() {
        // cancel_flag = 1 → body skipped.
        let payload = pack_bits(&[(1, 1)]);
        let got = parse_equirectangular_projection(&payload).unwrap();
        assert!(got.cancel_flag);
        assert!(got.persistence_flag.is_none());
        assert!(got.padding_flag.is_none());
        assert!(got.gb_erp_type.is_none());
    }

    #[test]
    fn equirectangular_projection_no_padding() {
        // cancel=0, persistence=1, padding=0, reserved_zero_2bits=0.
        let payload = pack_bits(&[
            (0, 1), // cancel
            (1, 1), // persistence
            (0, 1), // padding
            (0, 2), // reserved
        ]);
        let got = parse_equirectangular_projection(&payload).unwrap();
        assert!(!got.cancel_flag);
        assert_eq!(got.persistence_flag, Some(true));
        assert_eq!(got.padding_flag, Some(false));
        assert!(got.gb_erp_type.is_none());
        assert!(got.left_gb_erp_width.is_none());
        assert!(got.right_gb_erp_width.is_none());
    }

    #[test]
    fn equirectangular_projection_with_padding() {
        // cancel=0, persistence=0, padding=1, reserved_zero_2bits=0,
        // gb_erp_type=2, left=8, right=4.
        let payload = pack_bits(&[
            (0, 1), // cancel
            (0, 1), // persistence
            (1, 1), // padding
            (0, 2), // reserved
            (2, 3), // gb_erp_type
            (8, 8), // left
            (4, 8), // right
        ]);
        let got = parse_equirectangular_projection(&payload).unwrap();
        assert!(!got.cancel_flag);
        assert_eq!(got.persistence_flag, Some(false));
        assert_eq!(got.padding_flag, Some(true));
        assert_eq!(got.gb_erp_type, Some(2));
        assert_eq!(got.left_gb_erp_width, Some(8));
        assert_eq!(got.right_gb_erp_width, Some(4));
    }

    #[test]
    fn parse_payload_dispatches_equirectangular_projection() {
        let payload = pack_bits(&[(1, 1)]);
        let ctx = SeiContext::default();
        match parse_payload(150, &payload, &ctx).unwrap() {
            SeiPayload::EquirectangularProjection(e) => assert!(e.cancel_flag),
            other => panic!("expected EquirectangularProjection, got {other:?}"),
        }
    }

    #[test]
    fn cubemap_projection_cancel() {
        let payload = pack_bits(&[(1, 1)]);
        let got = parse_cubemap_projection(&payload).unwrap();
        assert!(got.cancel_flag);
        assert!(got.persistence_flag.is_none());
    }

    #[test]
    fn cubemap_projection_persist() {
        let payload = pack_bits(&[(0, 1), (1, 1)]);
        let got = parse_cubemap_projection(&payload).unwrap();
        assert!(!got.cancel_flag);
        assert_eq!(got.persistence_flag, Some(true));
    }

    #[test]
    fn parse_payload_dispatches_cubemap_projection() {
        let payload = pack_bits(&[(0, 1), (0, 1)]);
        let ctx = SeiContext::default();
        match parse_payload(151, &payload, &ctx).unwrap() {
            SeiPayload::CubemapProjection(c) => {
                assert!(!c.cancel_flag);
                assert_eq!(c.persistence_flag, Some(false));
            }
            other => panic!("expected CubemapProjection, got {other:?}"),
        }
    }

    #[test]
    fn sphere_rotation_cancel() {
        let payload = pack_bits(&[(1, 1)]);
        let got = parse_sphere_rotation(&payload).unwrap();
        assert!(got.cancel_flag);
        assert!(got.yaw_rotation.is_none());
        assert!(got.pitch_rotation.is_none());
        assert!(got.roll_rotation.is_none());
    }

    #[test]
    fn sphere_rotation_zero_angles() {
        // cancel=0, persistence=1, reserved=0, then three i32(0) big-endian.
        // First byte: 0_1_000000 = 0x40. Then 12 zero bytes.
        let mut payload = vec![0x40];
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&i32_be(0));
        let got = parse_sphere_rotation(&payload).unwrap();
        assert!(!got.cancel_flag);
        assert_eq!(got.persistence_flag, Some(true));
        assert_eq!(got.yaw_rotation, Some(0));
        assert_eq!(got.pitch_rotation, Some(0));
        assert_eq!(got.roll_rotation, Some(0));
    }

    #[test]
    fn sphere_rotation_min_max_angles() {
        // Test boundary values per §D.2.35.3:
        //   yaw=-180*2^16 (-11_796_480), pitch=90*2^16 (5_898_240),
        //   roll=180*2^16-1 (11_796_479).
        // cancel=0, persistence=0, reserved=0.
        let mut payload = vec![0x00];
        payload.extend_from_slice(&i32_be(-11_796_480));
        payload.extend_from_slice(&i32_be(5_898_240));
        payload.extend_from_slice(&i32_be(11_796_479));
        let got = parse_sphere_rotation(&payload).unwrap();
        assert_eq!(got.yaw_rotation, Some(-11_796_480));
        assert_eq!(got.pitch_rotation, Some(5_898_240));
        assert_eq!(got.roll_rotation, Some(11_796_479));
    }

    #[test]
    fn sphere_rotation_yaw_out_of_range_rejected() {
        // yaw=180*2^16 (11_796_480) — one past the high limit.
        let mut payload = vec![0x00];
        payload.extend_from_slice(&i32_be(11_796_480));
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&i32_be(0));
        let err = parse_sphere_rotation(&payload).unwrap_err();
        assert_eq!(err, SeiError::SphereRotationYawOutOfRange(11_796_480));
    }

    #[test]
    fn sphere_rotation_pitch_out_of_range_rejected() {
        // pitch=-90*2^16 - 1 (-5_898_241) — one past the low limit.
        let mut payload = vec![0x00];
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&i32_be(-5_898_241));
        payload.extend_from_slice(&i32_be(0));
        let err = parse_sphere_rotation(&payload).unwrap_err();
        assert_eq!(err, SeiError::SphereRotationPitchOutOfRange(-5_898_241));
    }

    #[test]
    fn sphere_rotation_roll_out_of_range_rejected() {
        let mut payload = vec![0x00];
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&i32_be(11_796_480));
        let err = parse_sphere_rotation(&payload).unwrap_err();
        assert_eq!(err, SeiError::SphereRotationRollOutOfRange(11_796_480));
    }

    #[test]
    fn parse_payload_dispatches_sphere_rotation() {
        let payload = pack_bits(&[(1, 1)]);
        let ctx = SeiContext::default();
        match parse_payload(154, &payload, &ctx).unwrap() {
            SeiPayload::SphereRotation(s) => assert!(s.cancel_flag),
            other => panic!("expected SphereRotation, got {other:?}"),
        }
    }

    // ----- Round 95 — omni_viewport / shutter_interval_info -----

    #[test]
    fn omni_viewport_cancel() {
        // viewport_id=42, cancel=1 → 10+1 bits = (42<<1)|1 = 85
        // bits MSB-first, padded.
        let payload = pack_bits(&[(42, 10), (1, 1)]);
        let got = parse_omni_viewport(&payload).unwrap();
        assert_eq!(got.viewport_id, 42);
        assert!(got.cancel_flag);
        assert!(got.persistence_flag.is_none());
        assert!(got.viewports.is_empty());
    }

    #[test]
    fn omni_viewport_single_directors_cut() {
        // viewport_id=0 (director's cut), cancel=0, persistence=1, cnt_minus1=0.
        // Header bits: u(10)=0, u(1)=0, u(1)=1, u(4)=0 → 16 bits = 2 bytes.
        // bytes 0..2: 0b00000000_00010000 → 0x00, 0x10.
        let mut payload = vec![0x00, 0x10];
        // One viewport entry: azimuth=0, elevation=0, tilt=0, hor=1, ver=1.
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&u32_be(1));
        payload.extend_from_slice(&u32_be(1));
        let got = parse_omni_viewport(&payload).unwrap();
        assert_eq!(got.viewport_id, 0);
        assert!(!got.cancel_flag);
        assert_eq!(got.persistence_flag, Some(true));
        assert_eq!(got.viewports.len(), 1);
        let v = got.viewports[0];
        assert_eq!(v.azimuth_centre, 0);
        assert_eq!(v.elevation_centre, 0);
        assert_eq!(v.tilt_centre, 0);
        assert_eq!(v.hor_range, 1);
        assert_eq!(v.ver_range, 1);
    }

    #[test]
    fn omni_viewport_max_range_boundary_accepted() {
        // viewport_id=1, cancel=0, persistence=0, cnt_minus1=0 — two entries
        // requires cnt_minus1=1; here use one entry with max-allowed values.
        // header: u(10)=1, u(1)=0, u(1)=0, u(4)=0 = 16 bits total =
        //   0b00000000_01_0_0_0000 = 0x00 0x40
        let mut payload = vec![0x00, 0x40];
        payload.extend_from_slice(&i32_be(180 * (1 << 16) - 1));
        payload.extend_from_slice(&i32_be(90 * (1 << 16)));
        payload.extend_from_slice(&i32_be(-180 * (1 << 16)));
        payload.extend_from_slice(&u32_be(360 * (1 << 16)));
        payload.extend_from_slice(&u32_be(180 * (1 << 16)));
        let got = parse_omni_viewport(&payload).unwrap();
        assert_eq!(got.viewport_id, 1);
        assert_eq!(got.viewports.len(), 1);
        let v = got.viewports[0];
        assert_eq!(v.azimuth_centre, 180 * (1 << 16) - 1);
        assert_eq!(v.elevation_centre, 90 * (1 << 16));
        assert_eq!(v.tilt_centre, -180 * (1 << 16));
        assert_eq!(v.hor_range, 360 * (1 << 16));
        assert_eq!(v.ver_range, 180 * (1 << 16));
    }

    #[test]
    fn omni_viewport_two_entries() {
        // viewport_id=2, cancel=0, persistence=1, cnt_minus1=1 (so 2 entries).
        // 10 + 1 + 1 + 4 = 16 bits: u(10)=2, u(1)=0, u(1)=1, u(4)=1
        //   = 0b00000000_10_0_1_0001 = 0x00, 0x91.
        let mut payload = vec![0x00, 0x91];
        // entry 0
        payload.extend_from_slice(&i32_be(100));
        payload.extend_from_slice(&i32_be(200));
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&u32_be(1024));
        payload.extend_from_slice(&u32_be(512));
        // entry 1
        payload.extend_from_slice(&i32_be(-300));
        payload.extend_from_slice(&i32_be(-400));
        payload.extend_from_slice(&i32_be(50));
        payload.extend_from_slice(&u32_be(2048));
        payload.extend_from_slice(&u32_be(1024));
        let got = parse_omni_viewport(&payload).unwrap();
        assert_eq!(got.viewport_id, 2);
        assert_eq!(got.viewports.len(), 2);
        assert_eq!(got.viewports[0].azimuth_centre, 100);
        assert_eq!(got.viewports[1].azimuth_centre, -300);
        assert_eq!(got.viewports[1].tilt_centre, 50);
    }

    #[test]
    fn omni_viewport_hor_range_zero_rejected() {
        // header for one entry, then hor_range=0 (forbidden by §D.2.35.5).
        let mut payload = vec![0x00, 0x10];
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&u32_be(0));
        payload.extend_from_slice(&u32_be(1));
        let err = parse_omni_viewport(&payload).unwrap_err();
        assert_eq!(err, SeiError::OmniViewportHorRangeOutOfRange(0));
    }

    #[test]
    fn omni_viewport_ver_range_overflow_rejected() {
        let mut payload = vec![0x00, 0x10];
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&u32_be(1));
        payload.extend_from_slice(&u32_be(180 * (1 << 16) + 1));
        let err = parse_omni_viewport(&payload).unwrap_err();
        assert_eq!(
            err,
            SeiError::OmniViewportVerRangeOutOfRange(180 * (1 << 16) + 1)
        );
    }

    #[test]
    fn omni_viewport_azimuth_overflow_rejected() {
        let mut payload = vec![0x00, 0x10];
        payload.extend_from_slice(&i32_be(180 * (1 << 16))); // one past high
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&u32_be(1));
        payload.extend_from_slice(&u32_be(1));
        let err = parse_omni_viewport(&payload).unwrap_err();
        assert_eq!(
            err,
            SeiError::OmniViewportAzimuthOutOfRange(180 * (1 << 16))
        );
    }

    #[test]
    fn omni_viewport_elevation_underflow_rejected() {
        let mut payload = vec![0x00, 0x10];
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&i32_be(-90 * (1 << 16) - 1));
        payload.extend_from_slice(&i32_be(0));
        payload.extend_from_slice(&u32_be(1));
        payload.extend_from_slice(&u32_be(1));
        let err = parse_omni_viewport(&payload).unwrap_err();
        assert_eq!(
            err,
            SeiError::OmniViewportElevationOutOfRange(-90 * (1 << 16) - 1)
        );
    }

    #[test]
    fn parse_payload_dispatches_omni_viewport() {
        // viewport_id=0, cancel=1.
        let payload = pack_bits(&[(0, 10), (1, 1)]);
        let ctx = SeiContext::default();
        match parse_payload(156, &payload, &ctx).unwrap() {
            SeiPayload::OmniViewport(v) => {
                assert!(v.cancel_flag);
                assert_eq!(v.viewport_id, 0);
            }
            other => panic!("expected OmniViewport, got {other:?}"),
        }
    }

    #[test]
    fn regionwise_packing_cancel_skips_body() {
        // rwp_cancel_flag = 1 → body absent (§D.1.35.4).
        let payload = pack_bits(&[(1, 1)]);
        let got = parse_regionwise_packing(&payload).unwrap();
        assert!(got.cancel_flag);
        assert!(got.persistence_flag.is_none());
        assert!(got.constituent_picture_matching_flag.is_none());
        assert!(got.proj_picture_width.is_none());
        assert!(got.regions.is_empty());
    }

    #[test]
    fn regionwise_packing_single_region_no_guard_band() {
        // One packed region, no guard band.
        //   rwp_cancel_flag                   = 0   (1)
        //   rwp_persistence_flag              = 1   (1)
        //   constituent_picture_matching_flag = 0   (1)
        //   rwp_reserved_zero_5bits           = 0   (5)
        //   num_packed_regions               = 1   u(8)
        //   proj_picture_width               = 3840 u(32)
        //   proj_picture_height              = 1920 u(32)
        //   packed_picture_width             = 2560 u(16)
        //   packed_picture_height            = 1280 u(16)
        //   region 0:
        //     rwp_reserved_zero_4bits        = 0   (4)
        //     transform_type                 = 5   (3)   (90° anticlockwise)
        //     guard_band_flag                = 0   (1)
        //     proj_region_width              = 1920 u(32)
        //     proj_region_height             = 1920 u(32)
        //     proj_region_top                = 0   u(32)
        //     proj_region_left               = 0   u(32)
        //     packed_region_width            = 1280 u(16)
        //     packed_region_height           = 1280 u(16)
        //     packed_region_top              = 0   u(16)
        //     packed_region_left             = 0   u(16)
        let fields = [
            (0u64, 1),
            (1, 1),
            (0, 1),
            (0, 5),
            (1, 8),
            (3840, 32),
            (1920, 32),
            (2560, 16),
            (1280, 16),
            (0, 4),
            (5, 3),
            (0, 1),
            (1920, 32),
            (1920, 32),
            (0, 32),
            (0, 32),
            (1280, 16),
            (1280, 16),
            (0, 16),
            (0, 16),
        ];
        let payload = pack_bits(&fields);
        let got = parse_regionwise_packing(&payload).unwrap();
        assert!(!got.cancel_flag);
        assert_eq!(got.persistence_flag, Some(true));
        assert_eq!(got.constituent_picture_matching_flag, Some(false));
        assert_eq!(got.proj_picture_width, Some(3840));
        assert_eq!(got.proj_picture_height, Some(1920));
        assert_eq!(got.packed_picture_width, Some(2560));
        assert_eq!(got.packed_picture_height, Some(1280));
        assert_eq!(got.regions.len(), 1);
        let reg = got.regions[0];
        assert_eq!(reg.transform_type, 5);
        assert_eq!(reg.proj_region_width, 1920);
        assert_eq!(reg.proj_region_height, 1920);
        assert_eq!(reg.packed_region_width, 1280);
        assert_eq!(reg.packed_region_height, 1280);
        assert!(reg.guard_band.is_none());
    }

    #[test]
    fn regionwise_packing_region_with_guard_band() {
        // One region carrying a guard band on all four edges.
        let fields = [
            (0u64, 1), // cancel
            (0, 1),    // persistence
            (1, 1),    // constituent_picture_matching_flag
            (0, 5),    // reserved
            (1, 8),    // num_packed_regions
            (1000, 32),
            (500, 32),
            (1000, 16),
            (500, 16),
            // region 0
            (0, 4), // reserved
            (0, 3), // transform_type = 0 (no transform)
            (1, 1), // guard_band_flag = 1
            (100, 32),
            (100, 32),
            (0, 32),
            (0, 32),
            (100, 16),
            (100, 16),
            (0, 16),
            (0, 16),
            // guard band
            (4, 8), // left_gb_width
            (4, 8), // right_gb_width
            (2, 8), // top_gb_height
            (2, 8), // bottom_gb_height
            (1, 1), // gb_not_used_for_pred_flag
            (1, 3), // gb_type[0]
            (2, 3), // gb_type[1]
            (3, 3), // gb_type[2]
            (1, 3), // gb_type[3]
            (0, 3), // rwp_gb_reserved_zero_3bits
        ];
        let payload = pack_bits(&fields);
        let got = parse_regionwise_packing(&payload).unwrap();
        assert_eq!(got.constituent_picture_matching_flag, Some(true));
        assert_eq!(got.regions.len(), 1);
        let gb = got.regions[0].guard_band.expect("guard band present");
        assert_eq!(gb.left_gb_width, 4);
        assert_eq!(gb.right_gb_width, 4);
        assert_eq!(gb.top_gb_height, 2);
        assert_eq!(gb.bottom_gb_height, 2);
        assert!(gb.gb_not_used_for_pred_flag);
        assert_eq!(gb.gb_type, [1, 2, 3, 1]);
    }

    #[test]
    fn regionwise_packing_two_regions_mixed_guard_band() {
        // Two regions: region 0 with guard band, region 1 without.
        let fields = [
            (0u64, 1), // cancel
            (1, 1),    // persistence
            (0, 1),    // constituent_picture_matching_flag
            (0, 5),    // reserved
            (2, 8),    // num_packed_regions = 2
            (2000, 32),
            (1000, 32),
            (2000, 16),
            (1000, 16),
            // region 0 (guard band)
            (0, 4),
            (2, 3), // transform_type
            (1, 1), // guard_band_flag
            (1000, 32),
            (1000, 32),
            (0, 32),
            (0, 32),
            (1000, 16),
            (1000, 16),
            (0, 16),
            (0, 16),
            (8, 8),
            (0, 8),
            (0, 8),
            (0, 8),
            (0, 1), // gb_not_used_for_pred_flag
            (0, 3),
            (1, 3),
            (1, 3),
            (1, 3),
            (0, 3),
            // region 1 (no guard band)
            (0, 4),
            (7, 3), // transform_type = 7
            (0, 1), // guard_band_flag = 0
            (1000, 32),
            (1000, 32),
            (0, 32),
            (1000, 32),
            (1000, 16),
            (1000, 16),
            (0, 16),
            (1000, 16),
        ];
        let payload = pack_bits(&fields);
        let got = parse_regionwise_packing(&payload).unwrap();
        assert_eq!(got.regions.len(), 2);
        assert_eq!(got.regions[0].transform_type, 2);
        let gb = got.regions[0].guard_band.expect("region 0 guard band");
        assert_eq!(gb.left_gb_width, 8);
        assert!(!gb.gb_not_used_for_pred_flag);
        assert_eq!(got.regions[1].transform_type, 7);
        assert_eq!(got.regions[1].proj_region_left, 1000);
        assert_eq!(got.regions[1].packed_region_left, 1000);
        assert!(got.regions[1].guard_band.is_none());
    }

    #[test]
    fn regionwise_packing_zero_regions_rejected() {
        // num_packed_regions = 0 → §D.2.35.4 violation.
        let fields = [(0u64, 1), (0, 1), (0, 1), (0, 5), (0, 8)];
        let payload = pack_bits(&fields);
        assert_eq!(
            parse_regionwise_packing(&payload),
            Err(SeiError::RegionWisePackingNoRegions)
        );
    }

    #[test]
    fn regionwise_packing_zero_proj_picture_rejected() {
        // proj_picture_width = 0 → §D.2.35.4 violation.
        let fields = [
            (0u64, 1),
            (0, 1),
            (0, 1),
            (0, 5),
            (1, 8),
            (0, 32),
            (100, 32),
        ];
        let payload = pack_bits(&fields);
        assert_eq!(
            parse_regionwise_packing(&payload),
            Err(SeiError::RegionWisePackingProjPictureZero { w: 0, h: 100 })
        );
    }

    #[test]
    fn regionwise_packing_zero_packed_picture_rejected() {
        // packed_picture_height = 0 → §D.2.35.4 violation.
        let fields = [
            (0u64, 1),
            (0, 1),
            (0, 1),
            (0, 5),
            (1, 8),
            (100, 32),
            (100, 32),
            (100, 16),
            (0, 16),
        ];
        let payload = pack_bits(&fields);
        assert_eq!(
            parse_regionwise_packing(&payload),
            Err(SeiError::RegionWisePackingPackedPictureZero { w: 100, h: 0 })
        );
    }

    #[test]
    fn regionwise_packing_guard_band_all_zero_rejected() {
        // guard_band_flag = 1 but all four dimensions 0 → §D.2.35.4.
        let fields = [
            (0u64, 1),
            (0, 1),
            (0, 1),
            (0, 5),
            (1, 8),
            (100, 32),
            (100, 32),
            (100, 16),
            (100, 16),
            (0, 4),
            (0, 3),
            (1, 1), // guard_band_flag = 1
            (50, 32),
            (50, 32),
            (0, 32),
            (0, 32),
            (50, 16),
            (50, 16),
            (0, 16),
            (0, 16),
            (0, 8), // all guard-band dims 0
            (0, 8),
            (0, 8),
            (0, 8),
        ];
        let payload = pack_bits(&fields);
        assert_eq!(
            parse_regionwise_packing(&payload),
            Err(SeiError::RegionWisePackingGuardBandAllZero(0))
        );
    }

    #[test]
    fn parse_payload_dispatches_regionwise_packing() {
        // Dispatch wiring for payload type 155: cancelled message.
        let payload = pack_bits(&[(1, 1)]);
        let ctx = SeiContext::default();
        match parse_payload(155, &payload, &ctx).unwrap() {
            SeiPayload::RegionWisePacking(r) => {
                assert!(r.cancel_flag);
                assert!(r.regions.is_empty());
            }
            other => panic!("expected RegionWisePacking, got {other:?}"),
        }
    }

    #[test]
    fn shutter_interval_info_sub_layer_idx_nonzero_skips_body() {
        // sii_sub_layer_idx = 1, no further bytes (per §D.2.38 the
        // body fields are absent when sub_layer_idx != 0).
        // ue(v) for 1 = "010" → 3 bits → padded byte 0b01000000 = 0x40.
        let payload = vec![0x40];
        let got = parse_shutter_interval_info(&payload).unwrap();
        assert_eq!(got.sub_layer_idx, 1);
        assert!(got.body.is_none());
    }

    #[test]
    fn shutter_interval_info_present_false() {
        // sii_sub_layer_idx = 0 → ue codeword "1" (1 bit).
        // info_present_flag = 0 → 1 more bit.
        // Two bits total → 0b10_000000 = 0x80.
        let payload = vec![0x80];
        let got = parse_shutter_interval_info(&payload).unwrap();
        assert_eq!(got.sub_layer_idx, 0);
        let body = got.body.expect("body present when sub_layer_idx==0");
        assert!(!body.info_present_flag);
        assert!(body.schedule.is_none());
    }

    #[test]
    fn shutter_interval_info_fixed_27mhz_1_080_000() {
        // Per §D.2.38 worked example: 27 MHz time-scale, 1_080_000 units
        // → 0.04 s shutter interval.
        // Header bits (MSB-first, packed):
        //   ue(0)             = "1"                    (1 bit)
        //   info_present_flag = 1                       (1 bit)
        //   time_scale        = 27_000_000   u(32)
        //   fixed_flag        = 1                       (1 bit)
        //   num_units         = 1_080_000    u(32)
        // Total bits before u(32) values: 3 → "11 1" → 0b11100000 first byte
        // but with the two u(32) values interleaved bit-by-bit. Build via
        // pack_bits to avoid hand-aligning.
        let payload = pack_bits(&[
            (1, 1),           // ue("1") = 0
            (1, 1),           // info_present
            (27_000_000, 32), // time_scale
            (1, 1),           // fixed_flag
            (1_080_000, 32),  // num_units
        ]);
        let got = parse_shutter_interval_info(&payload).unwrap();
        assert_eq!(got.sub_layer_idx, 0);
        let body = got.body.expect("body present");
        assert!(body.info_present_flag);
        let sched = body.schedule.expect("schedule present");
        assert_eq!(sched.time_scale, 27_000_000);
        match sched.kind {
            ShutterIntervalKind::Fixed { num_units } => assert_eq!(num_units, 1_080_000),
            other => panic!("expected Fixed, got {other:?}"),
        }
    }

    #[test]
    fn shutter_interval_info_per_sub_layer() {
        // 3 sub-layers (max_sub_layers_minus1 = 2 → 3 entries).
        let payload = pack_bits(&[
            (1, 1),       // ue("1") = sii_sub_layer_idx = 0
            (1, 1),       // info_present
            (90_000, 32), // time_scale = 90 kHz
            (0, 1),       // fixed_flag = 0
            (2, 3),       // max_sub_layers_minus1 = 2 → 3 entries
            (1_800, 32),  // sub_layer[0]
            (900, 32),    // sub_layer[1]
            (450, 32),    // sub_layer[2]
        ]);
        let got = parse_shutter_interval_info(&payload).unwrap();
        let body = got.body.expect("body present");
        let sched = body.schedule.expect("schedule present");
        assert_eq!(sched.time_scale, 90_000);
        match sched.kind {
            ShutterIntervalKind::PerSubLayer {
                max_sub_layers_minus1,
                num_units_per_sub_layer,
            } => {
                assert_eq!(max_sub_layers_minus1, 2);
                assert_eq!(num_units_per_sub_layer, vec![1_800, 900, 450]);
            }
            other => panic!("expected PerSubLayer, got {other:?}"),
        }
    }

    #[test]
    fn shutter_interval_info_time_scale_zero_rejected() {
        let payload = pack_bits(&[
            (1, 1),  // ue("1") = 0
            (1, 1),  // info_present
            (0, 32), // time_scale = 0 (forbidden)
            (1, 1),  // fixed_flag
            (1, 32), // num_units
        ]);
        let err = parse_shutter_interval_info(&payload).unwrap_err();
        assert_eq!(err, SeiError::ShutterIntervalTimeScaleZero);
    }

    #[test]
    fn parse_payload_dispatches_shutter_interval_info() {
        // info_present_flag = 0 → just two bits 0b10 = 0x80.
        let payload = vec![0x80];
        let ctx = SeiContext::default();
        match parse_payload(205, &payload, &ctx).unwrap() {
            SeiPayload::ShutterIntervalInfo(s) => {
                assert_eq!(s.sub_layer_idx, 0);
                assert!(s.body.is_some());
            }
            other => panic!("expected ShutterIntervalInfo, got {other:?}"),
        }
    }
}
