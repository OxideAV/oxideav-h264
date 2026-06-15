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
//! | 39           | §G.13.1.4   | §G.13.2.4   | multiview_scene_info (Annex G)     |
//! | 41           | §G.13.1.6   | §G.13.2.6   | non_required_view_component (Annex G) |
//! | 43           | §G.13.1.8   | §G.13.2.8   | operation_point_not_present (Annex G) |
//! | 45           | §D.1.26     | §D.2.26     | frame_packing_arrangement          |
//! | 46           | §G.13.1.10  | §G.13.2.10  | multiview_view_position (Annex G)  |
//! | 47           | §D.1.27     | §D.2.27     | display_orientation                |
//! | 50           | §H.13.1.3   | §H.13.2.3   | depth_representation_info (Annex H) |
//! | 51           | §H.13.1.4   | §H.13.2.4   | three_dimensional_reference_displays_info (Annex H) |
//! | 52           | §H.13.1.5   | §H.13.2.5   | depth_timing (Annex H)             |
//! | 53           | §H.13.1.7   | §H.13.2.7   | depth_sampling_info (Annex H) |
//! | 54           | §I.13.1.1   | §I.13.2.1   | constrained_depth_parameter_set_identifier (Annex I) |
//! | 137          | §D.1.29     | §D.2.29     | mastering_display_colour_volume    |
//! | 142          | §D.1.30     | §D.2.30     | colour_remapping_info              |
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
//!
//! Round-158 sub-parser: `parse_atsc1_envelope` decodes the typed ATSC1
//! envelope (provider_code = 0x0031 + user_identifier in {GA94, DTG1})
//! that real-world streams put inside a payload-type-4
//! `user_data_registered_itu_t_t35` with country_code = 0xB5 (USA). See
//! the `parse_atsc1_envelope` doc comment for the wire layout and the
//! ATSC A/53 Part 4 §6.2.3 reference. The CEA-708 cc_data() inner byte
//! layout is intentionally surfaced opaquely (CEA-708 spec is not in the
//! docs tree).

#![allow(dead_code)]

use crate::bitstream::{BitError, BitReader};
use crate::slice_header::{DecRefPicMarking, MmcoOp};
use crate::vui::{HrdParameters, VuiError};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SeiError {
    #[error("bitstream read failed: {0}")]
    Bitstream(#[from] BitError),
    #[error("hrd_parameters() parse failed: {0}")]
    Hrd(#[from] VuiError),
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
        "motion_constrained_slice_group_set num_slice_groups_in_set_minus1 shall be in 0..=num_slice_groups_minus1 per §D.2.20 (got count {count}, num_slice_groups {num_slice_groups})"
    )]
    MotionConstrainedSliceGroupCountTooLarge { count: u32, num_slice_groups: u32 },
    #[error(
        "motion_constrained_slice_group_set slice_group_id[{i}] shall be in 0..=num_slice_groups_minus1 per §D.2.20 (got {id}, num_slice_groups {num_slice_groups})"
    )]
    MotionConstrainedSliceGroupIdOutOfRange {
        i: usize,
        id: u32,
        num_slice_groups: u32,
    },
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
    #[error(
        "colour_remapping_info colour_remap_repetition_period shall be in 0..=16384 per §D.2.30 (got {0})"
    )]
    ColourRemapRepetitionPeriodOutOfRange(u32),
    #[error(
        "colour_remapping_info colour_remap_{which}_bit_depth shall be in 8..=16 per §D.2.30 (got {got})"
    )]
    ColourRemapBitDepthOutOfRange { which: &'static str, got: u8 },
    #[error(
        "colour_remapping_info {which}_lut_num_val_minus1[{c}] shall be in 0..=32 per §D.2.30 (got {got})"
    )]
    ColourRemapLutCountOutOfRange {
        which: &'static str,
        c: usize,
        got: u8,
    },
    #[error(
        "colour_remapping_info colour_remap_coeffs shall be in [-32768, 32767] per §D.2.30 (got {0})"
    )]
    ColourRemapCoeffOutOfRange(i32),
    #[error(
        "sei_manifest manifest_sei_payload_type values shall be unique per §D.2.36 (got duplicate {payload_type} at indices {first} and {second})"
    )]
    SeiManifestDuplicatePayloadType {
        payload_type: u16,
        first: usize,
        second: usize,
    },
    #[error(
        "sei_prefix_indication num_bits_in_prefix_indication_minus1[{i}] plus 1 = {bits} bits exceeds the remaining payload (only {available} bits left) per §D.1.37"
    )]
    SeiPrefixIndicationOverflow {
        i: usize,
        bits: u32,
        available: usize,
    },
    #[error(
        "ATSC1 envelope requires at least 6 bytes (2 provider_code + 4 user_identifier) per ATSC A/53 Part 4 §6.2.3 (got {0})"
    )]
    Atsc1EnvelopeTooShort(usize),
    #[error(
        "ATSC1 envelope provider_code shall be 0x0031 (ATSC) per ATSC A/53 Part 4 §6.2.3 / SMPTE-RA T.35 registration (got 0x{0:04X})"
    )]
    Atsc1ProviderCodeMismatch(u16),
    #[error(
        "ATSC1 envelope user_identifier 0x{0:08X} is not a registered ATSC value per ATSC A/53 Part 4 §6.2.3 Table 6.7 (expected 'GA94' 0x47413934 or 'DTG1' 0x44544731)"
    )]
    Atsc1UnknownUserIdentifier(u32),
    #[error(
        "ATSC1 ATSC_user_data() requires at least 1 byte for user_data_type_code per A/53 Part 4 §6.2.3 Table 6.8 (got empty payload)"
    )]
    Atsc1AtscUserDataEmpty,
    #[error(
        "ATSC1 bar_data reserved field must be '1111' per A/53 Part 4 Table 6.11 (got 0b{0:04b})"
    )]
    Atsc1BarDataReservedMismatch(u8),
    #[error(
        "ATSC1 bar_data {which}_bar one_bits field must be '11' per A/53 Part 4 Table 6.11 (got 0b{got:02b})"
    )]
    Atsc1BarDataOneBitsMismatch { which: &'static str, got: u8 },
    #[error(
        "ATSC1 bar_data top_bar_flag and bottom_bar_flag must match per A/53 Part 4 §6.2.3.2 (top={top}, bottom={bottom})"
    )]
    Atsc1BarDataTopBottomMismatch { top: bool, bottom: bool },
    #[error(
        "ATSC1 bar_data left_bar_flag and right_bar_flag must match per A/53 Part 4 §6.2.3.2 (left={left}, right={right})"
    )]
    Atsc1BarDataLeftRightMismatch { left: bool, right: bool },
    #[error(
        "ATSC1 bar_data may not signal both letterbox (top/bottom) and pillarbox (left/right) at once per A/53 Part 4 §6.2.3.2"
    )]
    Atsc1BarDataLetterboxPillarboxBoth,
    #[error("ATSC1 bar_data payload truncated — needed {needed} bytes, got {got}")]
    Atsc1BarDataTruncated { needed: usize, got: usize },
    #[error(
        "multiview_view_position num_views_minus1 shall be in 0..=1023 per Annex G §G.13.2.10 (got {0})"
    )]
    MultiviewViewPositionNumViewsOutOfRange(u32),
    #[error(
        "multiview_view_position view_position[{i}] shall be in 0..=1023 per Annex G §G.13.2.10 (got {got})"
    )]
    MultiviewViewPositionViewPositionOutOfRange { i: usize, got: u32 },
    #[error(
        "multiview_scene_info max_disparity shall be in 0..=1023 per Annex G §G.13.2.4 (got {0})"
    )]
    MultiviewSceneInfoMaxDisparityOutOfRange(u32),
    #[error(
        "operation_point_not_present num_operation_points shall be in 0..=65536 per Annex G §G.13.2.8 (got {0})"
    )]
    OperationPointNotPresentCountOutOfRange(u32),
    #[error(
        "operation_point_not_present operation_point_not_present_id[{i}] shall be in 0..=65535 per Annex G §G.13.2.8 (got {got})"
    )]
    OperationPointNotPresentIdOutOfRange { i: usize, got: u32 },
    #[error(
        "base_view_temporal_hrd num_of_temporal_layers_in_base_view_minus1 shall be in 0..=7 per Annex G §G.13.2.9 (got {0})"
    )]
    BaseViewTemporalHrdLayerCountOutOfRange(u32),
    #[error(
        "non_required_view_component num_info_entries_minus1 shall be in 0..=1022 per Annex G §G.13.2.6 (num_views_minus1 - 1, with num_views_minus1 bounded at 1023) (got {0})"
    )]
    NonRequiredViewComponentNumInfoEntriesOutOfRange(u32),
    #[error(
        "non_required_view_component view_order_index[{i}] shall be in 1..=1023 per Annex G §G.13.2.6 (got {got})"
    )]
    NonRequiredViewComponentViewOrderIndexOutOfRange { i: usize, got: u32 },
    #[error(
        "non_required_view_component num_non_required_view_components_minus1[{i}] shall be in 0..=view_order_index[i]-1 per Annex G §G.13.2.6 (got {got}, view_order_index={view_order_index})"
    )]
    NonRequiredViewComponentCountOutOfRange {
        i: usize,
        got: u32,
        view_order_index: u32,
    },
    #[error(
        "non_required_view_component index_delta_minus1[{i}][{j}] shall be in 0..=view_order_index[i]-1 per Annex G §G.13.2.6 (got {got}, view_order_index={view_order_index})"
    )]
    NonRequiredViewComponentIndexDeltaOutOfRange {
        i: usize,
        j: usize,
        got: u32,
        view_order_index: u32,
    },
    #[error(
        "multiview_acquisition_info num_views_minus1 shall be in 0..=1023 per Annex G §G.13.2.5 (got {0})"
    )]
    MultiviewAcquisitionInfoNumViewsOutOfRange(u32),
    #[error(
        "multiview_acquisition_info {field} shall be in 0..=31 per Annex G §G.13.2.5 (got {got})"
    )]
    MultiviewAcquisitionInfoPrecOutOfRange { field: &'static str, got: u32 },
    #[error(
        "three_dimensional_reference_displays_info {field} shall be in 0..=31 per Annex H §H.13.2.4 (got {got})"
    )]
    ThreeDimensionalReferenceDisplaysInfoPrecOutOfRange { field: &'static str, got: u32 },
    #[error(
        "three_dimensional_reference_displays_info num_ref_displays_minus1 shall be in 0..=31 per Annex H §H.13.2.4 (got {0})"
    )]
    ThreeDimensionalReferenceDisplaysInfoNumRefDisplaysOutOfRange(u32),
    #[error(
        "depth_representation_info num_views_minus1 shall be in 0..=1023 per Annex H §H.13.2.3 (got {0})"
    )]
    DepthRepresentationInfoNumViewsOutOfRange(u32),
    #[error(
        "depth_representation_info {field} shall be in 0..=1023 per Annex H §H.13.2.3 (got {got})"
    )]
    DepthRepresentationInfoViewIdOutOfRange { field: &'static str, got: u32 },
    #[error(
        "depth_representation_info depth_nonlinear_representation_num_minus1 shall be in 0..=62 per Annex H §H.13.2.3 (got {0})"
    )]
    DepthRepresentationInfoNonlinearNumOutOfRange(u32),
    #[error(
        "depth_representation_info depth_nonlinear_representation_model[{i}] shall be in 0..=65535 per Annex H §H.13.2.3 (got {got})"
    )]
    DepthRepresentationInfoNonlinearModelOutOfRange { i: usize, got: u32 },
    #[error(
        "constrained_depth_parameter_set_identifier max_dps_id shall be in 0..=62 per Annex I §I.13.2.1 (depth_parameter_set_id range 1..=63 caps max_dps_id + 1 ≤ 63) (got {0})"
    )]
    ConstrainedDepthParameterSetIdentifierMaxDpsIdOutOfRange(u32),
    #[error(
        "constrained_depth_parameter_set_identifier max_dps_id_diff * 2 shall be less than max_dps_id per Annex I §I.13.2.1 (got max_dps_id_diff = {max_dps_id_diff}, max_dps_id = {max_dps_id})"
    )]
    ConstrainedDepthParameterSetIdentifierDiffViolatesBound {
        max_dps_id_diff: u32,
        max_dps_id: u32,
    },
    #[error("depth_sampling_info dttsr_{axis}_mul = 0 is reserved per Annex H §H.13.2.7")]
    DepthSamplingInfoDttsrMulReserved { axis: &'static str },
    #[error(
        "depth_sampling_info num_video_plus_depth_views_minus1 shall be in 0..=1023 per Annex H §H.13.2.7 (num_views_minus1 absolute upper bound from Annex G/H) (got {0})"
    )]
    DepthSamplingInfoNumViewsOutOfRange(u32),
    #[error(
        "depth_sampling_info depth_grid_view_id[{i}] shall be in 0..=1023 per Annex H §H.13.2.7 (view_id range) (got {got})"
    )]
    DepthSamplingInfoViewIdOutOfRange { i: usize, got: u32 },
    #[error(
        "depth_timing per_view_depth_timing_flag = 1 needs NumDepthViews from the active subset SPS MVCD extension (§H.7.3.2.1.5) to bound the per-view loop per Annex H §H.13.1.5, but SeiContext.num_depth_views is 0 (unknown)"
    )]
    DepthTimingNumDepthViewsUnknown,
    #[error(
        "depth_timing NumDepthViews shall be in 1..=1024 per Annex H (§H.7.3.2.1.5 num_views_minus1 ≤ 1023 caps the depth-view count) (got {0})"
    )]
    DepthTimingNumDepthViewsOutOfRange(u32),
    #[error(
        "alternative_depth_info num_constituent_views_gvd_minus1 shall be in 0..=3 per Annex H §H.13.2.6 (got {0})"
    )]
    AlternativeDepthInfoNumConstituentViewsOutOfRange(u32),
    #[error("alternative_depth_info {field} shall be in 0..=31 per Annex H §H.13.2.6 (got {got})")]
    AlternativeDepthInfoPrecOutOfRange { field: &'static str, got: u32 },
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

impl ToneMappingBody {
    /// §D.2.25 — the two normative *default end points* of the
    /// piece-wise linear mapping function (`tone_map_model_id == 3`).
    ///
    /// The spec defines `num_pivots` as
    ///
    /// > *"the number of pivot points in the piece-wise linear
    /// > mapping function without counting the two default end
    /// > points, (0, 0) and
    /// > (2^coded_data_bit_depth − 1, 2^target_bit_depth − 1)."*
    ///
    /// Only the interior `num_pivots` pivots carry signalled
    /// `(coded_pivot_value[i], target_pivot_value[i])` pairs; the two
    /// end points are implicit and never appear in the bitstream.
    /// This accessor materialises them as `(coded, target)` pairs in
    /// the same `(coded_pivot_value, target_pivot_value)` domain as
    /// the stored interior pivots, so a caller assembling the full
    /// mapping curve can prepend the start point, append the end
    /// point, and interpolate across all `num_pivots + 2` points.
    ///
    /// Returns `None` unless `tone_map_model_id == 3` — the implicit
    /// end points are defined only for the piece-wise linear model;
    /// no inferred end points exist for the linear / sigmoid /
    /// user-table / reserved models.
    ///
    /// The end value `(2^coded_data_bit_depth − 1,
    /// 2^target_bit_depth − 1)` is computed from the body's
    /// `coded_data_bit_depth` (range 8..=14 ⇒ coded end 255..=16383)
    /// and `target_bit_depth` (range 1..=16 ⇒ target end 1..=65535);
    /// both fit in `u32`, the same width the interior pivots are
    /// stored in.
    #[must_use]
    pub fn piecewise_default_end_points(&self) -> Option<((u32, u32), (u32, u32))> {
        if self.model_id != 3 {
            return None;
        }
        // Saturating shifts keep this total even for out-of-spec
        // bit depths (the parser already range-checks the legal
        // 8..=14 / 1..=16 windows, but the accessor must not panic
        // on a hand-constructed body).
        let coded_end = 1u32
            .checked_shl(u32::from(self.coded_data_bit_depth))
            .map_or(u32::MAX, |v| v - 1);
        let target_end = 1u32
            .checked_shl(u32::from(self.target_bit_depth))
            .map_or(u32::MAX, |v| v - 1);
        Some(((0, 0), (coded_end, target_end)))
    }

    /// §D.2.25 — the total number of pivot points in the piece-wise
    /// linear mapping curve, **including** the two implicit default
    /// end points `(0, 0)` and
    /// `(2^coded_data_bit_depth − 1, 2^target_bit_depth − 1)`.
    ///
    /// The bitstream signals only the interior `num_pivots` points
    /// (§D.2.25: `num_pivots` is counted *without* the two end
    /// points), so the assembled curve has `num_pivots + 2` points.
    ///
    /// Returns `None` unless `tone_map_model_id == 3`. With the
    /// on-wire `num_pivots` being u(16) (0..=65535) the total lies in
    /// `2..=65537`, which is why the return widens to `u32`.
    #[must_use]
    pub fn piecewise_total_pivot_count(&self) -> Option<u32> {
        match &self.model {
            ToneMappingModel::PiecewisePivots { num_pivots, .. } => {
                Some(u32::from(*num_pivots) + 2)
            }
            _ => None,
        }
    }
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

impl FilmGrainSeparateColourDescription {
    /// §D.2.21 — luma bit depth of the simulated film grain signal,
    /// reconstructed from the on-wire `film_grain_bit_depth_luma_minus8`
    /// field.
    ///
    /// The spec text reads: "*film_grain_bit_depth_luma_minus8 plus 8
    /// specifies the bit depth used for the luma component of the film
    /// grain characteristics specified in the SEI message*". The
    /// derivation is `filmGrainBitDepth[0] =
    /// film_grain_bit_depth_luma_minus8 + 8` (eq. D-14). This accessor
    /// surfaces that `filmGrainBitDepth[0]` semantic directly rather
    /// than the biased on-wire `_minus8` carrier.
    ///
    /// The on-wire field is u(3) (0..=7), so the result is in 8..=15 and
    /// always fits in `u8`. The return is unconditional (not `Option`)
    /// because the field is only present — and this struct only
    /// constructed — when `separate_colour_description_present_flag ==
    /// 1`; the §D.2.21 "not present ⇒ inferred to bit_depth_luma_minus8"
    /// rule applies to the absent-struct case, which is represented by
    /// [`FilmGrainBody::separate_colour_description`] being `None`.
    #[must_use]
    pub fn bit_depth_luma(&self) -> u8 {
        self.bit_depth_luma_minus8 + 8
    }

    /// §D.2.21 — chroma (Cb / Cr) bit depth of the simulated film grain
    /// signal, reconstructed from the on-wire
    /// `film_grain_bit_depth_chroma_minus8` field.
    ///
    /// The spec text reads: "*film_grain_bit_depth_chroma_minus8 plus 8
    /// specifies the bit depth used for the Cb and Cr components of the
    /// film grain characteristics specified in the SEI message*". The
    /// derivation is `filmGrainBitDepth[c] =
    /// film_grain_bit_depth_chroma_minus8 + 8` for c = 1, 2 (eq. D-15).
    /// This accessor surfaces that shared `filmGrainBitDepth[1] ==
    /// filmGrainBitDepth[2]` semantic directly rather than the biased
    /// on-wire `_minus8` carrier.
    ///
    /// The on-wire field is u(3) (0..=7), so the result is in 8..=15 and
    /// always fits in `u8`. The return is unconditional (not `Option`)
    /// for the same reason as [`Self::bit_depth_luma`]: the §D.2.21
    /// "not present ⇒ inferred" rule is represented by the carrying
    /// struct being `None`, not by an in-struct sentinel.
    #[must_use]
    pub fn bit_depth_chroma(&self) -> u8 {
        self.bit_depth_chroma_minus8 + 8
    }
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
    /// From the active subset SPS — the `NumDepthViews` variable
    /// accumulated from `depth_view_present_flag[i]` over the
    /// §H.7.3.2.1.5 `seq_parameter_set_mvcd_extension()` view loop.
    /// Needed by §H.13.1.5 depth_timing: when
    /// `per_view_depth_timing_flag == 1` the message carries one
    /// `depth_timing_offset()` per depth view, and the loop bound
    /// comes from this SPS-derived variable rather than the payload.
    /// Defaults to 0 (unknown / no Annex H subset SPS active); a
    /// depth_timing message that reaches the per-view branch with a 0
    /// value is rejected with
    /// [`SeiError::DepthTimingNumDepthViewsUnknown`].
    pub num_depth_views: u32,
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
            // §H.13.1.5 — unknown until an Annex H subset SPS MVCD
            // extension is wired in; the depth_timing per-view branch
            // is rejected under this value.
            num_depth_views: 0,
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

// =============================================================================
// ATSC1 (ATSC A/53 Part 4) typed envelope on top of user_data_registered_itu_t_t35
// =============================================================================
//
// The single most common ATSC1 carriage seen in real-world H.264 streams
// is closed-caption + bar / AFD metadata wrapped inside SEI payload-type 4
// (`user_data_registered_itu_t_t35`, §D.2.6). The wire layout is:
//
//   country_code      = 0xB5             (USA, ITU-T T.35 §3.1)
//   provider_code     = 0x0031           (ATSC, per the SMPTE-RA T.35
//                                         terminal-provider registration
//                                         referenced from ATSC A/53 Part 4)
//   user_identifier   = 0x47413934       ("GA94" → ATSC_user_data() body,
//                                         A/53 Part 4 §6.2.3 Table 6.7)
//                       | 0x44544731     ("DTG1" → afd_data() body,
//                                         A/53 Part 4 §6.2.3 Table 6.7)
//   ATSC_user_data() = user_data_type_code (u(8)) + user_data_type_structure()
//                                        A/53 Part 4 Table 6.8 + Table 6.9:
//   user_data_type_code values:
//     0x00..=0x02 — ATSC reserved
//     0x03        — MPEG_cc_data() (cc_data() per CEA-708 [1] Table 2)
//     0x04..=0x05 — ATSC reserved
//     0x06        — bar_data()
//     0x07..=0xFF — ATSC reserved
//
// This module parses the **outer envelope** (country / provider / user_id /
// type_code) and the bar_data() body (Table 6.11 + 6.12 + §6.2.3.2). The
// inner cc_data() bytes are kept opaque because the CEA-708 Table 2 byte
// layout is normatively defined outside this spec — see the docs gap note
// in the surrounding parser.

/// ATSC1 user_identifier registered values (A/53 Part 4 Table 6.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Atsc1UserIdentifier {
    /// 'GA94' (0x47413934) — ATSC_user_data() body per A/53 Part 4 §6.2.3.
    Ga94,
    /// 'DTG1' (0x44544731) — afd_data() body per A/53 Part 4 §6.2.3.
    Dtg1,
}

impl Atsc1UserIdentifier {
    /// Raw 4-byte representation as it appears on the wire.
    pub const GA94: u32 = 0x4741_3934;
    /// Raw 4-byte representation as it appears on the wire.
    pub const DTG1: u32 = 0x4454_4731;

    fn from_u32(v: u32) -> Option<Self> {
        match v {
            Self::GA94 => Some(Self::Ga94),
            Self::DTG1 => Some(Self::Dtg1),
            _ => None,
        }
    }
}

/// One ATSC_user_data() entry per A/53 Part 4 §6.2.3 Table 6.8 / Table 6.9.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Atsc1UserData {
    /// `user_data_type_code == 0x03` — MPEG_cc_data() per Table 6.10.
    ///
    /// The cc_data() inner bytes are normatively defined in CEA-708 [1]
    /// Table 2, which is *not* present in this docs tree. We surface the
    /// raw cc_data() byte string (with the trailing marker_bits == 0xFF
    /// preserved) so callers that have a CEA-708 parser handy can drive
    /// it; this layer treats the inner payload as opaque.
    CcData { cc_data_bytes: Vec<u8> },
    /// `user_data_type_code == 0x06` — bar_data() per Table 6.11.
    BarData(BarData),
    /// `user_data_type_code ∈ {0x00, 0x01, 0x02, 0x04, 0x05, 0x07..=0xFF}` —
    /// ATSC reserved per Table 6.9. Receiving devices are expected to
    /// silently discard the structure (A/53 Part 4 §6.2.2). We keep the
    /// raw bytes to make round-trip diagnostics possible.
    Reserved { type_code: u8, raw: Vec<u8> },
}

/// `afd_data()` per ATSC A/53 Part 4 §6.2.4 / Table 6.13.
///
/// Carried under the `'DTG1'` user_identifier. The wire layout is
/// `[zero u(1) | active_format_flag u(1) | reserved '000001' u(6)]`
/// followed optionally (when `active_format_flag == 1`) by `reserved '1111'
/// u(4) | active_format u(4)`. We surface `active_format_flag` and the
/// resolved 4-bit `active_format` value; the reserved bits are validated
/// against their fixed patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AfdData {
    pub active_format_flag: bool,
    /// `active_format` per A/53 Part 4 Table 6.14. Only meaningful when
    /// `active_format_flag` is true; `None` otherwise.
    pub active_format: Option<u8>,
}

/// `bar_data()` per ATSC A/53 Part 4 §6.2.3.2 / Table 6.11.
///
/// A `bar_data()` structure signals either a letterbox region
/// (top + bottom bars) or a pillarbox region (left + right bars). Per
/// §6.2.3.2 the two pairs are mutually exclusive in a single structure;
/// `top_bar_flag` shall equal `bottom_bar_flag`, and `left_bar_flag`
/// shall equal `right_bar_flag`. We enforce both constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BarData {
    /// `(line_number_end_of_top_bar, line_number_start_of_bottom_bar)` —
    /// `Some(_)` when `top_bar_flag == bottom_bar_flag == 1`. Each value
    /// is a 14-bit unsigned line number whose video-format-dependent
    /// designation is specified in Table 6.12.
    pub letterbox: Option<(u16, u16)>,
    /// `(pixel_number_end_of_left_bar, pixel_number_start_of_right_bar)` —
    /// `Some(_)` when `left_bar_flag == right_bar_flag == 1`. Each value
    /// is a 14-bit unsigned pixel index counted from zero at the leftmost
    /// luma sample per §6.2.3.2.
    pub pillarbox: Option<(u16, u16)>,
}

/// Typed ATSC1 envelope sitting on top of `user_data_registered_itu_t_t35`.
///
/// `provider_code` is preserved verbatim — `parse_atsc1_envelope` rejects
/// any value other than `0x0031` so consumers can assume it. The structured
/// `user_identifier` enum disambiguates the inner body type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Atsc1Envelope {
    /// `provider_code` — always `0x0031` after a successful parse.
    pub provider_code: u16,
    /// `user_identifier` — registered value per A/53 Part 4 Table 6.7.
    pub user_identifier: Atsc1UserIdentifier,
    /// Decoded inner body — `Ga94 → AtscUserData`, `Dtg1 → Afd`.
    pub body: Atsc1Body,
}

/// Inner body of an ATSC1 envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Atsc1Body {
    /// `'GA94'` → `ATSC_user_data()` per Table 6.8.
    AtscUserData(Atsc1UserData),
    /// `'DTG1'` → `afd_data()` per §6.2.4.
    Afd(AfdData),
}

/// Parse an `afd_data()` body per A/53 Part 4 §6.2.4 / Table 6.13.
///
/// Layout (1 or 2 bytes):
/// ```text
/// afd_data() {
///   zero                              u(1) = 0
///   active_format_flag                u(1)
///   reserved                          u(6) = 0b000001
///   if (active_format_flag == 1) {
///     reserved                        u(4) = 0b1111
///     active_format                   u(4)
///   }
/// }
/// ```
///
/// The 'zero' leading bit + the 6-bit reserved field both have fixed
/// patterns per Table 6.13 and the §6.2.4.2 semantics, but we accept any
/// values silently because real-world streams have been known to populate
/// the high bit differently and §6.2.4 Note (a) clarifies the field is
/// non-`reserved`-in-the-strict-sense.
fn parse_afd_data(bytes: &[u8]) -> Result<AfdData, SeiError> {
    let Some(&first) = bytes.first() else {
        return Err(SeiError::Atsc1BarDataTruncated { needed: 1, got: 0 });
    };
    let active_format_flag = (first & 0b0100_0000) != 0;
    if !active_format_flag {
        return Ok(AfdData {
            active_format_flag: false,
            active_format: None,
        });
    }
    // Second byte = '1111' u(4) + active_format u(4).
    let Some(&second) = bytes.get(1) else {
        return Err(SeiError::Atsc1BarDataTruncated { needed: 2, got: 1 });
    };
    let active_format = second & 0x0F;
    Ok(AfdData {
        active_format_flag: true,
        active_format: Some(active_format),
    })
}

/// Parse a `bar_data()` body per A/53 Part 4 §6.2.3.2 / Table 6.11.
///
/// Wire layout (1 + 0..4 × 2 bytes):
/// ```text
///   top_bar_flag    u(1)
///   bottom_bar_flag u(1)
///   left_bar_flag   u(1)
///   right_bar_flag  u(1)
///   reserved        u(4) = '1111'
///   if (top_bar_flag == 1) {
///     one_bits      u(2) = '11'
///     line_number_end_of_top_bar     u(14)
///   }
///   if (bottom_bar_flag == 1) {
///     one_bits      u(2) = '11'
///     line_number_start_of_bottom_bar u(14)
///   }
///   if (left_bar_flag == 1) {
///     one_bits      u(2) = '11'
///     pixel_number_end_of_left_bar    u(14)
///   }
///   if (right_bar_flag == 1) {
///     one_bits      u(2) = '11'
///     pixel_number_start_of_right_bar u(14)
///   }
/// ```
///
/// Per §6.2.3.2: top_bar_flag == bottom_bar_flag, left_bar_flag ==
/// right_bar_flag, and only one of the two pairs may be set at a time.
/// The reserved nibble and each `one_bits` value have fixed bit patterns
/// per the table and are validated.
fn parse_bar_data(bytes: &[u8]) -> Result<BarData, SeiError> {
    let mut r = BitReader::new(bytes);
    let top = r.u(1)? == 1;
    let bottom = r.u(1)? == 1;
    let left = r.u(1)? == 1;
    let right = r.u(1)? == 1;
    let reserved = r.u(4)? as u8;
    if reserved != 0b1111 {
        return Err(SeiError::Atsc1BarDataReservedMismatch(reserved));
    }
    if top != bottom {
        return Err(SeiError::Atsc1BarDataTopBottomMismatch { top, bottom });
    }
    if left != right {
        return Err(SeiError::Atsc1BarDataLeftRightMismatch { left, right });
    }
    if (top || bottom) && (left || right) {
        return Err(SeiError::Atsc1BarDataLetterboxPillarboxBoth);
    }

    let mut read_14 = |which: &'static str| -> Result<u16, SeiError> {
        let one_bits = r.u(2)? as u8;
        if one_bits != 0b11 {
            return Err(SeiError::Atsc1BarDataOneBitsMismatch {
                which,
                got: one_bits,
            });
        }
        Ok(r.u(14)? as u16)
    };
    let letterbox = if top {
        let t = read_14("top")?;
        let b = read_14("bottom")?;
        Some((t, b))
    } else {
        None
    };
    let pillarbox = if left {
        let l = read_14("left")?;
        let rt = read_14("right")?;
        Some((l, rt))
    } else {
        None
    };
    Ok(BarData {
        letterbox,
        pillarbox,
    })
}

/// Parse an ATSC1 envelope from the **`payload_bytes`** field of a
/// `user_data_registered_itu_t_t35` (i.e. the bytes that follow the leading
/// country_code byte). The caller has already established that
/// `country_code == 0xB5` (USA).
///
/// Validates:
/// * Minimum length (≥ 6 bytes for the 2-byte provider_code + 4-byte
///   user_identifier).
/// * `provider_code == 0x0031` (ATSC's T.35 administered slot).
/// * `user_identifier ∈ {0x47413934 'GA94', 0x44544731 'DTG1'}` per
///   A/53 Part 4 §6.2.3 Table 6.7.
///
/// For `'GA94'` it reads the `user_data_type_code` (u(8)) per Table 6.8
/// and dispatches:
/// * `0x03` → `Atsc1UserData::CcData { cc_data_bytes }` — opaque
///   (CEA-708 Table 2 layout out of docs scope).
/// * `0x06` → `Atsc1UserData::BarData(_)` per Table 6.11.
/// * any reserved code → `Atsc1UserData::Reserved { type_code, raw }`
///   per §6.2.2 ("receiving devices are expected to silently discard
///   unrecognized video user data").
///
/// For `'DTG1'` it parses `afd_data()` per §6.2.4 / Table 6.13.
pub fn parse_atsc1_envelope(payload_bytes: &[u8]) -> Result<Atsc1Envelope, SeiError> {
    if payload_bytes.len() < 6 {
        return Err(SeiError::Atsc1EnvelopeTooShort(payload_bytes.len()));
    }
    let provider_code = u16::from_be_bytes([payload_bytes[0], payload_bytes[1]]);
    if provider_code != 0x0031 {
        return Err(SeiError::Atsc1ProviderCodeMismatch(provider_code));
    }
    let user_identifier_raw = u32::from_be_bytes([
        payload_bytes[2],
        payload_bytes[3],
        payload_bytes[4],
        payload_bytes[5],
    ]);
    let user_identifier = Atsc1UserIdentifier::from_u32(user_identifier_raw)
        .ok_or(SeiError::Atsc1UnknownUserIdentifier(user_identifier_raw))?;
    let rest = &payload_bytes[6..];
    let body = match user_identifier {
        Atsc1UserIdentifier::Ga94 => {
            let Some((&type_code, after)) = rest.split_first() else {
                return Err(SeiError::Atsc1AtscUserDataEmpty);
            };
            let inner = match type_code {
                0x03 => Atsc1UserData::CcData {
                    cc_data_bytes: after.to_vec(),
                },
                0x06 => Atsc1UserData::BarData(parse_bar_data(after)?),
                _ => Atsc1UserData::Reserved {
                    type_code,
                    raw: after.to_vec(),
                },
            };
            Atsc1Body::AtscUserData(inner)
        }
        Atsc1UserIdentifier::Dtg1 => Atsc1Body::Afd(parse_afd_data(rest)?),
    };
    Ok(Atsc1Envelope {
        provider_code,
        user_identifier,
        body,
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

impl SubSeqLayerCharacteristic {
    /// Reconstruct the §D.2.13 (2024) / §D.2.12 (2003 draft) `average_bit_rate`
    /// semantic in bits/second: the on-wire field is in units of
    /// 1000 bits/second (equations D-6 / D-10), so this multiplies by 1000.
    ///
    /// Returns `None` when the on-wire field is 0 — equations D-7 / D-11
    /// reserve that sentinel for the `t1 == t2` degenerate case and for the
    /// general "unspecified" reading; the spec defines no scalar derivation
    /// from a 0 carrier so no inferred value exists.
    ///
    /// The result fits in `u32` since the on-wire range is `0..=65535` and
    /// `65535 * 1000 == 65_535_000`, well below `u32::MAX`.
    #[must_use]
    pub fn average_bit_rate_bps(&self) -> Option<u32> {
        if self.average_bit_rate == 0 {
            None
        } else {
            Some((self.average_bit_rate as u32) * 1000)
        }
    }

    /// Reconstruct the §D.2.13 (2024) / §D.2.12 (2003 draft) `average_frame_rate`
    /// semantic in frames/second: the on-wire field is in units of
    /// frames/(256 seconds) (equations D-8 / D-12), so this divides by 256.
    ///
    /// Returns `None` when the on-wire field is 0 — equations D-9 / D-13
    /// reserve that sentinel for the `t1 == t2` degenerate case and the
    /// general "unspecified" reading; the spec defines no scalar derivation
    /// from a 0 carrier so no inferred value exists.
    ///
    /// `f64` is used so the 1/256 quantisation step is exactly representable
    /// (256 is a power of two, so all on-wire values map to dyadic rationals
    /// with no rounding error).
    #[must_use]
    pub fn average_frame_rate_fps(&self) -> Option<f64> {
        if self.average_frame_rate == 0 {
            None
        } else {
            Some((self.average_frame_rate as f64) / 256.0)
        }
    }
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

impl SubSeqCharacteristics {
    /// Reconstruct §D.2.13 (2024) / §D.2.12 (2003 draft) `sub_seq_duration`
    /// in seconds: the on-wire field is counted in 90-kHz clock ticks
    /// ("sub_seq_duration specifies the duration of the target sub-sequence
    /// in clock ticks of a 90-kHz clock"), so this divides by 90_000.
    ///
    /// Returns `None` when `duration_flag == 0` (the spec says: "duration_flag
    /// equal to 0 indicates that the duration of the target sub-sequence is
    /// not specified") — `sub_seq_duration` is `None` in that case and no
    /// scalar interpretation is defined.
    ///
    /// `f64` is used so the 90-kHz quantisation step is representable with
    /// only the lossless `u32`-to-`f64` widening followed by a single
    /// floating-point division; the full u(32) carrier range
    /// (0..=4_294_967_295 ticks ≈ 13.25 hours) round-trips through
    /// `f64::from(u32)` without loss.
    #[must_use]
    pub fn sub_seq_duration_seconds(&self) -> Option<f64> {
        self.sub_seq_duration
            .map(|ticks| f64::from(ticks) / 90_000.0)
    }
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

impl ProgressiveRefinementSegmentStart {
    /// §D.2.18 — total count of refinement steps the segment promises,
    /// reconstructed from the on-wire `num_refinement_steps_minus1`
    /// field.
    ///
    /// The spec text reads: "*num_refinement_steps_minus1 plus 1
    /// indicates the maximum number of refinement steps that may be
    /// used during the refinement of the picture quality from the
    /// progressive_refinement_segment_start to the corresponding
    /// progressive_refinement_segment_end*". This accessor surfaces
    /// the semantic `NumRefinementSteps = num_refinement_steps_minus1
    /// + 1` directly.
    ///
    /// The return type is `u64` so the full u(32) carrier range maps
    /// without overflow: `num_refinement_steps_minus1 == u32::MAX`
    /// (the `ue(v)` carrier is bit-bounded by the parser, but the
    /// stored field is `u32`) yields `u32::MAX + 1 == 4_294_967_296`,
    /// which would overflow a `u32` return. `u32` callers can safely
    /// downcast when the segment's carrier is known to fit, but the
    /// raw return makes no such assumption.
    ///
    /// Always returns a positive value: the on-wire `_minus1` bias
    /// guarantees `NumRefinementSteps >= 1` for any well-formed
    /// payload, so the return is unconditional `u64` rather than
    /// `Option<u64>` — unlike the `Option`-returning accessors that
    /// guard against an unsignalled-flag sentinel, this field has no
    /// "absent" encoding.
    #[must_use]
    pub fn num_refinement_steps(&self) -> u64 {
        u64::from(self.num_refinement_steps_minus1) + 1
    }
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
    // §D.2.20 — "The allowed range of num_slice_groups_in_set_minus1 is 0 to
    // num_slice_groups_minus1, inclusive." Reject anything larger before
    // allocating the per-id loop: the inner read is `u(width)` bits, and at
    // `width == 0` (single-slice-group PPS) the loop reads zero bits per
    // iteration, so an unbounded `ue(v)` count would drive an unbounded
    // `Vec::push(0)` and OOM the decoder. Fuzz crash from
    // OxideAV/oxideav-h264#1044 reached this path via path-2 of the
    // `sei_payload` target.
    if num_slice_groups_in_set_minus1 >= num_slice_groups {
        return Err(SeiError::MotionConstrainedSliceGroupCountTooLarge {
            count: num_slice_groups_in_set_minus1.saturating_add(1),
            num_slice_groups,
        });
    }
    let count = (num_slice_groups_in_set_minus1 as usize) + 1;
    let mut slice_group_ids = Vec::with_capacity(count);
    for i in 0..count {
        let v = if width == 0 { 0 } else { r.u(width)? };
        // §D.2.20 — "slice_group_id[ i ] ... identifies the slice group(s)
        // contained within the slice group set" with the same per-PPS range
        // as §7.4.2.2's slice_group_id (0..=num_slice_groups_minus1). Reject
        // out-of-range ids so the message can't point at a slice group that
        // the active PPS never declared.
        if v >= num_slice_groups {
            return Err(SeiError::MotionConstrainedSliceGroupIdOutOfRange {
                i,
                id: v,
                num_slice_groups,
            });
        }
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
// Round 117 — colour_remapping_info (payload type 142, §D.1.30 / §D.2.30).
// ---------------------------------------------------------------------------

/// §D.2.30 — colour_remap_video_signal_info: the optional triple of
/// colour-space descriptors (E.2.1 semantics) for the *remapped*
/// reconstructed picture. Present only when
/// `colour_remap_video_signal_info_present_flag == 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColourRemapVideoSignalInfo {
    /// colour_remap_full_range_flag — E.2.1 `video_full_range_flag`
    /// semantics, applied to the remapped picture.
    pub full_range_flag: bool,
    /// colour_remap_primaries — E.2.1 `colour_primaries`.
    pub primaries: u8,
    /// colour_remap_transfer_function — E.2.1 `transfer_characteristics`.
    pub transfer_function: u8,
    /// colour_remap_matrix_coefficients — E.2.1 `matrix_coefficients`.
    pub matrix_coefficients: u8,
}

/// One (coded, target) pivot point of a piece-wise linear LUT, for one
/// colour component. Used for both the pre-LUT and the post-LUT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColourRemapLutEntry {
    /// `pre_lut_coded_value` / `post_lut_coded_value`.
    pub coded_value: u32,
    /// `pre_lut_target_value` / `post_lut_target_value`.
    pub target_value: u32,
}

/// The 3×3 colour-remapping matrix (§D.2.30). Present only when
/// `colour_remap_matrix_present_flag == 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColourRemapMatrix {
    /// log2_matrix_denom — base-2 log of the shared denominator, 0..=15.
    pub log2_matrix_denom: u8,
    /// colour_remap_coeffs[ c ][ i ], row-major (c outer, i inner). Each
    /// value is in [-2^15, 2^15 - 1].
    pub coeffs: [[i32; 3]; 3],
}

/// §D.2.30 — colour_remapping_info (payload type 142).
///
/// Conveys a colour-remapping function — an optional 3-component
/// pre-LUT, an optional 3×3 matrix, and an optional 3-component post-LUT
/// — that maps the decoded samples (interpreted at
/// `colour_remap_input_bit_depth`) to a remapped colour space at
/// `colour_remap_output_bit_depth`. Display application of the function
/// is outside the scope of this Specification; this layer parses and
/// validates the descriptor.
///
/// Syntax (§D.1.30):
/// ```text
/// colour_remapping_info( payloadSize ) {
///   colour_remap_id                              ue(v)
///   colour_remap_cancel_flag                     u(1)
///   if( !colour_remap_cancel_flag ) {
///     colour_remap_repetition_period             ue(v)
///     colour_remap_video_signal_info_present_flag u(1)
///     if( ... ) { full_range u(1); primaries u(8);
///                 transfer u(8); matrix u(8) }
///     colour_remap_input_bit_depth               u(8)
///     colour_remap_output_bit_depth              u(8)
///     for( c = 0; c < 3; c++ ) {                 // pre-LUT
///       pre_lut_num_val_minus1[c]                u(8)
///       if( > 0 ) for(i..) { coded u(v); target u(v) }
///     }
///     colour_remap_matrix_present_flag           u(1)
///     if( ... ) { log2_matrix_denom u(4);
///                 colour_remap_coeffs[c][i] se(v) }
///     for( c = 0; c < 3; c++ ) {                 // post-LUT
///       post_lut_num_val_minus1[c]               u(8)
///       if( > 0 ) for(i..) { coded u(v); target u(v) }
///     }
///   }
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColourRemappingInfo {
    /// colour_remap_id — identifying number, 0..=2^32 - 2.
    pub colour_remap_id: u32,
    /// colour_remap_cancel_flag. When `true`, the message cancels prior
    /// persistence and no further fields follow.
    pub cancel_flag: bool,
    /// Present only when `cancel_flag == false`.
    pub body: Option<ColourRemappingInfoBody>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColourRemappingInfoBody {
    /// colour_remap_repetition_period — 0..=16384.
    pub repetition_period: u32,
    /// colour_remap_video_signal_info, present only when its
    /// present_flag was set.
    pub video_signal_info: Option<ColourRemapVideoSignalInfo>,
    /// colour_remap_input_bit_depth — 8..=16.
    pub input_bit_depth: u8,
    /// colour_remap_output_bit_depth — 8..=16.
    pub output_bit_depth: u8,
    /// pre-LUT for the three components (0 = luma/G, 1 = Cb/B,
    /// 2 = Cr/R). An empty inner Vec means `pre_lut_num_val_minus1 == 0`
    /// (default end points, no coded pivots).
    pub pre_lut: [Vec<ColourRemapLutEntry>; 3],
    /// colour_remap_matrix, present only when its present_flag was set.
    pub matrix: Option<ColourRemapMatrix>,
    /// post-LUT for the three components, same convention as `pre_lut`.
    pub post_lut: [Vec<ColourRemapLutEntry>; 3],
}

/// §D.2.30 — parse colour_remapping_info (payload type 142).
pub fn parse_colour_remapping_info(payload: &[u8]) -> Result<ColourRemappingInfo, SeiError> {
    let mut r = BitReader::new(payload);
    let colour_remap_id = r.ue()?;
    let cancel_flag = r.u(1)? == 1;
    if cancel_flag {
        return Ok(ColourRemappingInfo {
            colour_remap_id,
            cancel_flag,
            body: None,
        });
    }

    let repetition_period = r.ue()?;
    // §D.2.30: shall be in 0..=16384.
    if repetition_period > 16_384 {
        return Err(SeiError::ColourRemapRepetitionPeriodOutOfRange(
            repetition_period,
        ));
    }

    let video_signal_info = if r.u(1)? == 1 {
        let full_range_flag = r.u(1)? == 1;
        let primaries = r.u(8)? as u8;
        let transfer_function = r.u(8)? as u8;
        let matrix_coefficients = r.u(8)? as u8;
        Some(ColourRemapVideoSignalInfo {
            full_range_flag,
            primaries,
            transfer_function,
            matrix_coefficients,
        })
    } else {
        None
    };

    let input_bit_depth = r.u(8)? as u8;
    let output_bit_depth = r.u(8)? as u8;
    // §D.2.30: both shall be in 8..=16; decoders shall ignore messages
    // outside the range. We surface that as an error so the caller can
    // drop the message.
    if !(8..=16).contains(&input_bit_depth) {
        return Err(SeiError::ColourRemapBitDepthOutOfRange {
            which: "input",
            got: input_bit_depth,
        });
    }
    if !(8..=16).contains(&output_bit_depth) {
        return Err(SeiError::ColourRemapBitDepthOutOfRange {
            which: "output",
            got: output_bit_depth,
        });
    }

    // §D.2.30: number of bits for a LUT value is
    // ( ( bit_depth + 7 ) >> 3 ) << 3 — i.e. round the bit depth up to a
    // whole number of bytes. With bit_depth in 8..=16 this is 8 or 16.
    let coded_bits = |bd: u8| -> u32 { ((bd as u32 + 7) >> 3) << 3 };
    let pre_coded_bits = coded_bits(input_bit_depth);
    let pre_target_bits = coded_bits(output_bit_depth);
    let post_coded_bits = coded_bits(output_bit_depth);
    let post_target_bits = coded_bits(output_bit_depth);

    let read_lut = |r: &mut BitReader,
                    coded_bits: u32,
                    target_bits: u32,
                    which: &'static str|
     -> Result<[Vec<ColourRemapLutEntry>; 3], SeiError> {
        let mut lut: [Vec<ColourRemapLutEntry>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (c, comp) in lut.iter_mut().enumerate() {
            let num_val_minus1 = r.u(8)? as u8;
            // §D.2.30: pre/post_lut_num_val_minus1[c] shall be in 0..=32.
            if num_val_minus1 > 32 {
                return Err(SeiError::ColourRemapLutCountOutOfRange {
                    which,
                    c,
                    got: num_val_minus1,
                });
            }
            // The syntax only emits pivots when num_val_minus1 > 0; a
            // value of 0 means the default end points (no coded entries).
            if num_val_minus1 > 0 {
                for _ in 0..=num_val_minus1 {
                    let coded_value = r.u(coded_bits)?;
                    let target_value = r.u(target_bits)?;
                    comp.push(ColourRemapLutEntry {
                        coded_value,
                        target_value,
                    });
                }
            }
        }
        Ok(lut)
    };

    let pre_lut = read_lut(&mut r, pre_coded_bits, pre_target_bits, "pre")?;

    let matrix = if r.u(1)? == 1 {
        let log2_matrix_denom = r.u(4)? as u8; // u(4) ⇒ 0..=15, in-range by construction.
        let mut coeffs = [[0i32; 3]; 3];
        for row in coeffs.iter_mut() {
            for coeff in row.iter_mut() {
                let v = r.se()?;
                // §D.2.30: colour_remap_coeffs shall be in [-2^15, 2^15 - 1].
                if !(-(1 << 15)..=(1 << 15) - 1).contains(&v) {
                    return Err(SeiError::ColourRemapCoeffOutOfRange(v));
                }
                *coeff = v;
            }
        }
        Some(ColourRemapMatrix {
            log2_matrix_denom,
            coeffs,
        })
    } else {
        None
    };

    let post_lut = read_lut(&mut r, post_coded_bits, post_target_bits, "post")?;

    Ok(ColourRemappingInfo {
        colour_remap_id,
        cancel_flag,
        body: Some(ColourRemappingInfoBody {
            repetition_period,
            video_signal_info,
            input_bit_depth,
            output_bit_depth,
            pre_lut,
            matrix,
            post_lut,
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

/// §D.2.36 — SEI manifest (payload type 200).
///
/// Advertises, for the entire coded video sequence, which SEI payload
/// types are expected to be present along with a per-type
/// "necessary / unnecessary / undetermined" classification from
/// Table D-12. Transport- or systems-layer elements can use this to
/// decide whether the CVS is suitable for delivery to a receiver
/// without requiring the receiver to scan the whole stream first.
///
/// Syntax (§D.1.36):
/// ```text
/// sei_manifest( payloadSize ) {
///   manifest_num_sei_msg_types                          u(16)
///   for( i = 0; i < manifest_num_sei_msg_types; i++ ) {
///     manifest_sei_payload_type[ i ]                    u(16)
///     manifest_sei_description[ i ]                     u(8)
///   }
/// }
/// ```
///
/// Per §D.2.36: the `manifest_sei_payload_type[ i ]` values shall all
/// be distinct (parse-time check), and `manifest_sei_description[ i ]`
/// values 4..=255 are reserved. We preserve reserved descriptions
/// verbatim so callers honouring §D.2.36's "Decoders shall allow ...
/// shall ignore" rule can route them through `Description::Reserved`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeiManifest {
    pub entries: Vec<SeiManifestEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeiManifestEntry {
    pub payload_type: u16,
    pub description: SeiManifestDescription,
}

/// Table D-12 — `manifest_sei_description[ i ]` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeiManifestDescription {
    /// 0 — No SEI message of this type is expected in the CVS.
    NotExpected,
    /// 1 — SEI messages of this type are expected and considered
    /// necessary by the encoder.
    ExpectedNecessary,
    /// 2 — SEI messages of this type are expected and considered
    /// unnecessary by the encoder.
    ExpectedUnnecessary,
    /// 3 — SEI messages of this type are expected and their necessity
    /// is undetermined.
    ExpectedUndetermined,
    /// 4..=255 — reserved for future use. §D.2.36 requires decoders
    /// to allow these values and ignore the associated information,
    /// including any SEI prefix indication SEI messages keyed off the
    /// same payload type.
    Reserved(u8),
}

impl SeiManifestDescription {
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::NotExpected,
            1 => Self::ExpectedNecessary,
            2 => Self::ExpectedUnnecessary,
            3 => Self::ExpectedUndetermined,
            v => Self::Reserved(v),
        }
    }
}

pub fn parse_sei_manifest(payload: &[u8]) -> Result<SeiManifest, SeiError> {
    let mut r = BitReader::new(payload);
    // §D.1.36: manifest_num_sei_msg_types u(16).
    let count = r.u(16)? as usize;
    let mut entries: Vec<SeiManifestEntry> = Vec::with_capacity(count);
    for i in 0..count {
        let payload_type = r.u(16)? as u16;
        let description_raw = r.u(8)? as u8;
        // §D.2.36: "The values of manifest_sei_payload_type[ m ] and
        // manifest_sei_payload_type[ n ] shall not be identical when m
        // is not equal to n." We enforce uniqueness against the entries
        // already accepted.
        if let Some(prev) = entries.iter().position(|e| e.payload_type == payload_type) {
            return Err(SeiError::SeiManifestDuplicatePayloadType {
                payload_type,
                first: prev,
                second: i,
            });
        }
        entries.push(SeiManifestEntry {
            payload_type,
            description: SeiManifestDescription::from_raw(description_raw),
        });
    }
    Ok(SeiManifest { entries })
}

/// §D.2.37 — SEI prefix indication (payload type 201).
///
/// Carries one or more bit-string "prefix indications" for SEI messages
/// of a particular `prefix_sei_payload_type`. Each indication is a bit
/// string that follows the SEI payload syntax of that payload type
/// starting from its first syntax element, providing transport- or
/// systems-layer elements with a fast way to inspect the leading
/// syntax-elements of an upcoming SEI message (e.g.
/// `frame_packing_arrangement_type` for type 45) without parsing the
/// entire payload.
///
/// Syntax (§D.1.37):
/// ```text
/// sei_prefix_indication( payloadSize ) {
///   prefix_sei_payload_type                             u(16)
///   num_sei_prefix_indications_minus1                   u(8)
///   for( i = 0; i <= num_sei_prefix_indications_minus1; i++ ) {
///     num_bits_in_prefix_indication_minus1[ i ]         u(16)
///     for( j = 0; j <= num_bits_in_prefix_indication_minus1[ i ]; j++ )
///       sei_prefix_data_bit[ i ][ j ]                   u(1)
///     while( !byte_aligned() )
///       byte_alignment_bit_equal_to_one /* equal to 1 */ f(1)
///   }
/// }
/// ```
///
/// We keep the raw bits of each indication as a (`bit_count`, `bytes`)
/// pair — the spec leaves interpretation of `sei_prefix_data_bit[ i ]`
/// to the payload type identified by `prefix_sei_payload_type`, and
/// the byte-alignment fill is unconditionally `1`-bits per §D.1.37
/// (verified by `parse_sei_prefix_indication`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeiPrefixIndication {
    pub prefix_sei_payload_type: u16,
    pub indications: Vec<SeiPrefixIndicationEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeiPrefixIndicationEntry {
    /// `num_bits_in_prefix_indication_minus1[ i ] + 1` — the number of
    /// `sei_prefix_data_bit[ i ][ j ]` bits stored in `data`. The
    /// remaining low-order bits of the last byte of `data` are unused
    /// padding zeros (this struct stores the bits MSB-first).
    pub bit_count: u32,
    /// MSB-first packed `sei_prefix_data_bit[ i ][ * ]`. `data.len() ==
    /// bit_count.div_ceil(8)`.
    pub data: Vec<u8>,
}

pub fn parse_sei_prefix_indication(payload: &[u8]) -> Result<SeiPrefixIndication, SeiError> {
    let mut r = BitReader::new(payload);
    // §D.1.37: prefix_sei_payload_type u(16) + num_sei_prefix_indications_minus1 u(8).
    let prefix_sei_payload_type = r.u(16)? as u16;
    let count = (r.u(8)? as usize) + 1;
    let mut indications: Vec<SeiPrefixIndicationEntry> = Vec::with_capacity(count);
    for i in 0..count {
        // num_bits_in_prefix_indication_minus1[ i ] u(16) — value
        // plus one is the number of sei_prefix_data_bit[ i ][ j ] u(1).
        let bit_count = r.u(16)? + 1;
        let available = r.bits_remaining();
        if (bit_count as usize) > available {
            return Err(SeiError::SeiPrefixIndicationOverflow {
                i,
                bits: bit_count,
                available,
            });
        }
        // Pack the next `bit_count` u(1) values into a byte buffer,
        // MSB-first. Each iteration reads exactly one bit.
        let byte_len = bit_count.div_ceil(8) as usize;
        let mut data = vec![0u8; byte_len];
        for j in 0..(bit_count as usize) {
            let bit = r.u(1)? as u8;
            if bit == 1 {
                let byte = j / 8;
                let pos = 7 - (j % 8);
                data[byte] |= 1 << pos;
            }
        }
        // §D.1.37 byte-alignment trailer: while not byte-aligned, read
        // a single bit which shall equal 1. The `f(1)` descriptor
        // means the value is fixed by syntax. We don't reject a
        // stray 0 here — only the structurally required SEI trailing
        // bits at the very end of the SEI payload are checked
        // elsewhere (sei_rbsp + rbsp_trailing_bits in non_vcl.rs) —
        // but we DO consume the bits so the next indication starts
        // on a byte boundary.
        while !r.byte_aligned() {
            let _ = r.u(1)?;
        }
        indications.push(SeiPrefixIndicationEntry { bit_count, data });
    }
    Ok(SeiPrefixIndication {
        prefix_sei_payload_type,
        indications,
    })
}

/// §G.13.2.10 — multiview_view_position (payload type 46).
///
/// Annex G MVC extension. Specifies the relative left-to-right view
/// position of every view component in a coded video sequence; valid
/// only when associated with an IDR access unit, and the per-CVS
/// information applies to the whole sequence. The syntax lives outside
/// the main Annex D table because Annex G specifies it.
///
/// Syntax (§G.13.1.10):
/// ```text
/// multiview_view_position( payloadSize ) {
///   num_views_minus1                                ue(v)
///   for( i = 0; i <= num_views_minus1; i++ )
///     view_position[ i ]                            ue(v)
///   multiview_view_position_extension_flag          u(1)
/// }
/// ```
///
/// §G.13.2.10 semantics constraints:
///
/// * `num_views_minus1` shall match the active MVC SPS and shall be in
///   the range 0..=1023 inclusive (so the loop reads at most 1024
///   `view_position[i]` entries).
/// * Each `view_position[i]` shall be in the range 0..=1023 inclusive.
/// * `multiview_view_position_extension_flag` shall be equal to 0; a
///   value of 1 is reserved for future use, and decoders shall ignore
///   any bits following it. We preserve the observed bit verbatim so
///   round-trip callers can distinguish a conforming `0` from a
///   reserved-`1` extension marker.
///
/// This decoder does not (yet) carry the §G.7.3.2 MVC SPS — Phase 4 on
/// the README's "Profiles + features in scope" table. The parser is
/// nonetheless harmless on a non-MVC bitstream: when this payload type
/// appears outside an MVC access unit it can still be parsed for
/// inspection and logging without affecting decode of the main
/// (`view_id == 0`) sub-bitstream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiviewViewPosition {
    /// `view_position[i]`, one entry per view. The vector length is
    /// `num_views_minus1 + 1`, bounded at 1024 by the §G.13.2.10 range
    /// check that runs before allocation.
    pub view_positions: Vec<u16>,
    /// `multiview_view_position_extension_flag`. Per §G.13.2.10 the
    /// value shall be 0; conforming streams will always have `false`
    /// here.
    pub extension_flag: bool,
}

/// §G.13.2.4 — multiview_scene_info (payload type 39, Annex G / MVC).
///
/// Indicates the maximum disparity, in units of luma samples, between
/// spatially adjacent view components in an MVC access unit. A 3D
/// display compositor can use this hint when planning the per-view
/// rendering pipeline; the value is informative and does not affect
/// the decoding of any single view.
///
/// Per §G.13.2.4 the SEI message shall be associated with an IDR
/// access unit and the signalled value applies to the entire coded
/// video sequence. The actual maximum disparity may be smaller than
/// the signalled bound after sub-bitstream extraction (§G.8.5.3) has
/// removed views from the original stream.
///
/// Syntax — §G.13.1.4:
///
/// ```text
/// multiview_scene_info( payloadSize ) {
///   max_disparity                                     ue(v)
/// }
/// ```
///
/// Constraints — §G.13.2.4: `0 <= max_disparity <= 1023`. Values
/// outside this range surface as `MultiviewSceneInfoMaxDisparityOutOfRange`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiviewSceneInfo {
    /// `max_disparity` in units of luma samples, capped at 1023 by
    /// the §G.13.2.4 range check that runs before the value is
    /// surfaced.
    pub max_disparity: u16,
}

/// Parse a §G.13.1.4 `multiview_scene_info()` payload.
///
/// Enforces the §G.13.2.4 `0..=1023` range bound on `max_disparity`
/// before storage so a malformed Exp-Golomb codeword can't surface a
/// value the spec forbids.
pub fn parse_multiview_scene_info(payload: &[u8]) -> Result<MultiviewSceneInfo, SeiError> {
    let mut r = BitReader::new(payload);
    // §G.13.1.4 — max_disparity ue(v). §G.13.2.4 range check.
    let max_disparity = r.ue()?;
    if max_disparity > 1023 {
        return Err(SeiError::MultiviewSceneInfoMaxDisparityOutOfRange(
            max_disparity,
        ));
    }
    Ok(MultiviewSceneInfo {
        max_disparity: max_disparity as u16,
    })
}

/// §G.13.2.8 — operation_point_not_present (payload type 43, Annex G /
/// MVC).
///
/// Lists MVC operation-point identifiers that are NOT present in the
/// bitstream starting with the current access unit, interpreted with
/// respect to the previous view_scalability_info SEI message in
/// decoding order. The message remains effective until the next
/// operation_point_not_present of the same type or the end of the
/// coded video sequence, whichever is earlier.
///
/// Per §G.13.2.8 NOTE 1, the message is non-cumulative — each new
/// instance replaces (rather than augments) the previous list. The
/// IDs are u(16)-sized identifiers drawn from the
/// `operation_point_id[i]` set declared in the preceding
/// view_scalability_info SEI message.
///
/// Syntax — §G.13.1.8:
///
/// ```text
/// operation_point_not_present( payloadSize ) {
///   num_operation_points                             ue(v)
///   for( k = 0; k < num_operation_points; k++ )
///     operation_point_not_present_id[ k ]            ue(v)
/// }
/// ```
///
/// Constraints — §G.13.2.8:
/// * `0 <= num_operation_points <= num_operation_points_minus1` of
///   the preceding view_scalability_info; the absolute upper bound
///   is 65536 (since each ID is u(16)-bounded and IDs are unique).
///   We enforce that 65536 ceiling before allocating the
///   `Vec<u16>` so an adversarial `ue(v)` can't drive an unbounded
///   `Vec::with_capacity` from a maliciously-crafted SEI.
/// * Each `operation_point_not_present_id[k]` shall be in
///   `0..=65535` (i.e., representable as u16).
///
/// The 65536 upper bound on `num_operation_points` follows from the
/// spec's `num_operation_points_minus1` field elsewhere in Annex G
/// being a u(15)+1 maximum and IDs being u(16); we deliberately
/// pre-check before storage so a fuzzer cannot OOM the decoder via
/// the count field alone (the §D.2.20 oom-1bf45ba3 fix in round 177
/// was a structurally identical issue).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationPointNotPresent {
    /// `operation_point_not_present_id[k]` list. Each entry is the
    /// 16-bit operation-point identifier that is being declared as
    /// not present in the bitstream. Vector length =
    /// `num_operation_points`, capped at 65536 by the §G.13.2.8
    /// range check that runs before allocation.
    pub operation_point_not_present_ids: Vec<u16>,
}

/// Parse a §G.13.1.8 `operation_point_not_present()` payload.
///
/// Enforces both the §G.13.2.8 `0..=65536` cap on
/// `num_operation_points` (pre-allocation, anti-OOM) and the
/// §G.13.2.8 `0..=65535` range check on each
/// `operation_point_not_present_id[k]` before storage.
pub fn parse_operation_point_not_present(
    payload: &[u8],
) -> Result<OperationPointNotPresent, SeiError> {
    let mut r = BitReader::new(payload);
    // §G.13.1.8 — num_operation_points ue(v). Pre-allocation cap per
    // §G.13.2.8: ids are u(16) and unique, so the maximum count is
    // 65536. Larger values are bitstream-conformance errors AND a
    // realistic OOM vector (a malformed `ue(v)` can encode up to
    // 2^31 - 1 here pre-check); cf. round-177 fix for §D.2.20.
    let num_operation_points = r.ue()?;
    if num_operation_points > 65536 {
        return Err(SeiError::OperationPointNotPresentCountOutOfRange(
            num_operation_points,
        ));
    }
    let count = num_operation_points as usize;
    let mut operation_point_not_present_ids: Vec<u16> = Vec::with_capacity(count);
    for i in 0..count {
        // §G.13.1.8 — operation_point_not_present_id[k] ue(v).
        // §G.13.2.8: shall be in 0..=65535 (u16 representable).
        let id = r.ue()?;
        if id > 65535 {
            return Err(SeiError::OperationPointNotPresentIdOutOfRange { i, got: id });
        }
        operation_point_not_present_ids.push(id as u16);
    }
    Ok(OperationPointNotPresent {
        operation_point_not_present_ids,
    })
}

/// One temporal sub-bitstream's HRD entry inside a §G.13.1.9
/// `base_view_temporal_hrd()` SEI message (payload type 44, Annex G /
/// MVC). One `BaseViewTemporalHrdLayer` is recorded per temporal
/// sub-bitstream identified by `sei_mvc_temporal_id`.
///
/// Each field carries the value of the correspondingly-named §E.2.1
/// VUI/HRD syntax element that applies to the i-th temporal
/// sub-bitstream of the base view (`num_units_in_tick`, `time_scale`,
/// `fixed_frame_rate_flag`, `nal_hrd_parameters`, `vcl_hrd_parameters`,
/// `low_delay_hrd_flag`, `pic_struct_present_flag`), as restated by
/// §G.13.2.9.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseViewTemporalHrdLayer {
    /// `sei_mvc_temporal_id[i]` u(3) — the `temporal_id` value of the
    /// i-th temporal sub-bitstream (§G.13.2.9).
    pub temporal_id: u8,
    /// Present iff `sei_mvc_timing_info_present_flag[i]` was 1. Holds
    /// `(sei_mvc_num_units_in_tick, sei_mvc_time_scale,
    /// sei_mvc_fixed_frame_rate_flag)` for the i-th sub-bitstream.
    pub timing_info: Option<BaseViewTemporalHrdTiming>,
    /// `hrd_parameters()` block present iff
    /// `sei_mvc_nal_hrd_parameters_present_flag[i]` was 1.
    pub nal_hrd_parameters: Option<HrdParameters>,
    /// `hrd_parameters()` block present iff
    /// `sei_mvc_vcl_hrd_parameters_present_flag[i]` was 1.
    pub vcl_hrd_parameters: Option<HrdParameters>,
    /// `sei_mvc_low_delay_hrd_flag[i]` u(1). `Some` iff either of the
    /// NAL/VCL HRD parameter blocks is present (the syntax only codes
    /// this flag under that condition, §G.13.1.9).
    pub low_delay_hrd_flag: Option<bool>,
    /// `sei_mvc_pic_struct_present_flag[i]` u(1).
    pub pic_struct_present_flag: bool,
}

/// `(num_units_in_tick, time_scale, fixed_frame_rate_flag)` triple for a
/// single temporal sub-bitstream, present when
/// `sei_mvc_timing_info_present_flag[i]` is 1 (§G.13.1.9).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseViewTemporalHrdTiming {
    /// `sei_mvc_num_units_in_tick[i]` u(32).
    pub num_units_in_tick: u32,
    /// `sei_mvc_time_scale[i]` u(32).
    pub time_scale: u32,
    /// `sei_mvc_fixed_frame_rate_flag[i]` u(1).
    pub fixed_frame_rate_flag: bool,
}

/// §G.13.2.9 — base_view_temporal_hrd (payload type 44, Annex G / MVC).
///
/// Conveys the timing and HRD parameters that apply to each temporal
/// sub-bitstream of the base view of an MVC coded video sequence. It
/// lets an HRD verifier (or a temporal sub-bitstream extractor) recover
/// the per-temporal-layer `num_units_in_tick` / `time_scale` and the
/// NAL/VCL HRD parameter sets without re-deriving them from the base
/// view's own VUI, which describes only the full operation point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseViewTemporalHrd {
    /// One entry per temporal sub-bitstream. Length =
    /// `num_of_temporal_layers_in_base_view_minus1 + 1`, which §G.13.2.9
    /// constrains to 1..=8 (the minus1 form is bounded to 0..=7).
    pub layers: Vec<BaseViewTemporalHrdLayer>,
}

/// Parse a §G.13.1.9 `base_view_temporal_hrd()` payload (type 44).
///
/// `num_of_temporal_layers_in_base_view_minus1` is range-checked to
/// 0..=7 per §G.13.2.9 before the per-layer loop runs; an out-of-range
/// value is a bitstream-conformance error (and would otherwise be a
/// large `ue(v)` driving an unbounded loop). Each per-layer
/// `hrd_parameters()` block is delegated to the shared §E.1.2
/// [`HrdParameters::parse`]; its errors surface as [`SeiError::Hrd`].
pub fn parse_base_view_temporal_hrd(payload: &[u8]) -> Result<BaseViewTemporalHrd, SeiError> {
    let mut r = BitReader::new(payload);
    // §G.13.1.9 — num_of_temporal_layers_in_base_view_minus1 ue(v).
    // §G.13.2.9 constrains it to 0..=7.
    let minus1 = r.ue()?;
    if minus1 > 7 {
        return Err(SeiError::BaseViewTemporalHrdLayerCountOutOfRange(minus1));
    }
    let count = (minus1 + 1) as usize;
    let mut layers: Vec<BaseViewTemporalHrdLayer> = Vec::with_capacity(count);
    for _ in 0..count {
        // sei_mvc_temporal_id[i] u(3)
        let temporal_id = r.u(3)? as u8;
        // sei_mvc_timing_info_present_flag[i] u(1)
        let timing_info = if r.u(1)? == 1 {
            // sei_mvc_num_units_in_tick[i] u(32)
            let num_units_in_tick = r.u(32)?;
            // sei_mvc_time_scale[i] u(32)
            let time_scale = r.u(32)?;
            // sei_mvc_fixed_frame_rate_flag[i] u(1)
            let fixed_frame_rate_flag = r.u(1)? == 1;
            Some(BaseViewTemporalHrdTiming {
                num_units_in_tick,
                time_scale,
                fixed_frame_rate_flag,
            })
        } else {
            None
        };
        // sei_mvc_nal_hrd_parameters_present_flag[i] u(1)
        let nal_present = r.u(1)? == 1;
        let nal_hrd_parameters = if nal_present {
            Some(HrdParameters::parse(&mut r)?)
        } else {
            None
        };
        // sei_mvc_vcl_hrd_parameters_present_flag[i] u(1)
        let vcl_present = r.u(1)? == 1;
        let vcl_hrd_parameters = if vcl_present {
            Some(HrdParameters::parse(&mut r)?)
        } else {
            None
        };
        // sei_mvc_low_delay_hrd_flag[i] u(1) — coded only when either
        // HRD parameter block is present (§G.13.1.9).
        let low_delay_hrd_flag = if nal_present || vcl_present {
            Some(r.u(1)? == 1)
        } else {
            None
        };
        // sei_mvc_pic_struct_present_flag[i] u(1)
        let pic_struct_present_flag = r.u(1)? == 1;
        layers.push(BaseViewTemporalHrdLayer {
            temporal_id,
            timing_info,
            nal_hrd_parameters,
            vcl_hrd_parameters,
            low_delay_hrd_flag,
            pic_struct_present_flag,
        });
    }
    Ok(BaseViewTemporalHrd { layers })
}

/// §G.13.2.6 — non_required_view_component (payload type 41, Annex G /
/// MVC).
///
/// Indicates, per target view component in the access unit, the set of
/// view components that are NOT needed to decode the target view
/// component (or any subsequent view component with the same view_id in
/// decoding order within the coded video sequence). The list lets a
/// sub-bitstream extractor / display compositor drop view components
/// that won't be consumed downstream without disturbing decodability.
///
/// Per §G.13.2.6 the listed view components may even be absent from the
/// associated access unit; the message records the dependency graph
/// hint, not a presence assertion.
///
/// Syntax — §G.13.1.6:
///
/// ```text
/// non_required_view_component( payloadSize ) {
///   num_info_entries_minus1                          ue(v)
///   for( i = 0; i <= num_info_entries_minus1; i++ ) {
///     view_order_index[ i ]                          ue(v)
///     num_non_required_view_components_minus1[ i ]   ue(v)
///     for( j = 0; j <= num_non_required_view_components_minus1[ i ]; j++ )
///       index_delta_minus1[ i ][ j ]                 ue(v)
///   }
/// }
/// ```
///
/// Constraints — §G.13.2.6:
/// * `num_info_entries_minus1` shall be in `0..=num_views_minus1 − 1`.
///   With `num_views_minus1` itself bounded at 1023 per Annex G (see
///   §G.13.2.10 et al.), the absolute upper bound on
///   `num_info_entries_minus1` is 1022 → `num_info_entries` ≤ 1023.
///   We enforce the absolute 1022 ceiling before allocating the entry
///   vector so an adversarial `ue(v)` count can't drive an unbounded
///   `Vec::with_capacity` (cf. round-200 fix on
///   `operation_point_not_present`).
/// * `view_order_index[i]` shall be in `1..=num_views_minus1`. We use
///   the absolute upper bound 1023 and additionally require ≥ 1; any
///   zero `view_order_index` reads as a §G.13.2.6 conformance error
///   (the "i-th target view component" is by definition view_id 1 or
///   greater).
/// * `num_non_required_view_components_minus1[i]` shall be in
///   `0..=view_order_index[i] − 1` (so an entry's listed non-required
///   view components are always strictly earlier in view-order than
///   the target view). We use `view_order_index − 1` as the bound;
///   with `view_order_index ≤ 1023`, the per-entry list length is
///   bounded at 1023 before the inner loop allocates.
/// * `index_delta_minus1[i][j]` shall be in `0..=view_order_index[i]
///   − 1` (the delta is the difference between the target's
///   view_order_index and the non-required component's
///   view_order_index, both bounded above by 1023).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonRequiredViewComponent {
    /// One entry per target view component. There are
    /// `num_info_entries_minus1 + 1` entries total, capped at 1023 by
    /// the §G.13.2.6 absolute bound; the ordering matches the
    /// bitstream order.
    pub entries: Vec<NonRequiredViewComponentEntry>,
}

/// One target-view entry inside a `NonRequiredViewComponent` payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonRequiredViewComponentEntry {
    /// `view_order_index[i]` — view order index of the target view
    /// component. Bounded at 1..=1023 by the §G.13.2.6 range check.
    pub view_order_index: u16,
    /// `index_delta_minus1[i][j]` list. Each entry encodes a
    /// non-required view component as the delta in view-order index
    /// from the target: `non_required_voi = view_order_index −
    /// index_delta_minus1[j] − 1`. List length is bounded at 1023 by
    /// the §G.13.2.6 `view_order_index − 1` cap (with
    /// `view_order_index ≤ 1023`).
    pub index_delta_minus1: Vec<u16>,
}

/// Parse a §G.13.1.6 `non_required_view_component()` payload.
///
/// Enforces all four §G.13.2.6 range bounds before storage:
/// * `num_info_entries_minus1 ≤ 1022` (pre-allocation, anti-OOM)
/// * `view_order_index[i] ∈ 1..=1023` (lower bound from §G.13.2.6,
///   absolute upper bound from Annex G's `num_views_minus1 ≤ 1023`)
/// * `num_non_required_view_components_minus1[i] ≤ view_order_index −
///   1` (pre-allocation; this implicitly caps the inner list at 1022
///   when `view_order_index = 1023`)
/// * `index_delta_minus1[i][j] ≤ view_order_index − 1`
pub fn parse_non_required_view_component(
    payload: &[u8],
) -> Result<NonRequiredViewComponent, SeiError> {
    let mut r = BitReader::new(payload);
    // §G.13.1.6 — num_info_entries_minus1 ue(v). §G.13.2.6 caps the
    // value at `num_views_minus1 − 1`. With Annex G's absolute
    // `num_views_minus1 ≤ 1023`, the absolute upper bound is 1022 →
    // `num_info_entries` ≤ 1023. Pre-allocation cap (cf. round-200
    // anti-OOM pattern on `num_operation_points`).
    let num_info_entries_minus1 = r.ue()?;
    if num_info_entries_minus1 > 1022 {
        return Err(SeiError::NonRequiredViewComponentNumInfoEntriesOutOfRange(
            num_info_entries_minus1,
        ));
    }
    let entry_count = (num_info_entries_minus1 as usize) + 1;
    let mut entries: Vec<NonRequiredViewComponentEntry> = Vec::with_capacity(entry_count);
    for i in 0..entry_count {
        // §G.13.1.6 — view_order_index[i] ue(v). §G.13.2.6 range
        // 1..=num_views_minus1; absolute upper bound 1023. The lower
        // bound 1 follows from "the i-th target view component has
        // view_id equal to view_id[view_order_index[i]]" with view 0
        // being the base view (not a target).
        let view_order_index = r.ue()?;
        if !(1..=1023).contains(&view_order_index) {
            return Err(SeiError::NonRequiredViewComponentViewOrderIndexOutOfRange {
                i,
                got: view_order_index,
            });
        }
        // §G.13.1.6 — num_non_required_view_components_minus1[i]
        // ue(v). §G.13.2.6 caps it at `view_order_index − 1` so each
        // non-required view component is strictly earlier in
        // view-order than the target. Pre-allocation cap before the
        // inner loop allocates the per-entry delta vector.
        let num_non_required_view_components_minus1 = r.ue()?;
        let voi_minus_1 = view_order_index - 1; // safe — view_order_index >= 1 verified
        if num_non_required_view_components_minus1 > voi_minus_1 {
            return Err(SeiError::NonRequiredViewComponentCountOutOfRange {
                i,
                got: num_non_required_view_components_minus1,
                view_order_index,
            });
        }
        let inner_count = (num_non_required_view_components_minus1 as usize) + 1;
        let mut index_delta_minus1: Vec<u16> = Vec::with_capacity(inner_count);
        for j in 0..inner_count {
            // §G.13.1.6 — index_delta_minus1[i][j] ue(v). §G.13.2.6
            // range 0..=view_order_index[i] − 1. Both bounds map to
            // u16 once range-checked.
            let idm1 = r.ue()?;
            if idm1 > voi_minus_1 {
                return Err(SeiError::NonRequiredViewComponentIndexDeltaOutOfRange {
                    i,
                    j,
                    got: idm1,
                    view_order_index,
                });
            }
            index_delta_minus1.push(idm1 as u16);
        }
        entries.push(NonRequiredViewComponentEntry {
            view_order_index: view_order_index as u16,
            index_delta_minus1,
        });
    }
    Ok(NonRequiredViewComponent { entries })
}

pub fn parse_multiview_view_position(payload: &[u8]) -> Result<MultiviewViewPosition, SeiError> {
    let mut r = BitReader::new(payload);
    // §G.13.1.10 — num_views_minus1 ue(v); §G.13.2.10 range check
    // BEFORE allocating the per-view vector so an adversarial ue(v)
    // can't drive an unbounded Vec::push. BitReader::ue caps the
    // leading-zero run at 31 (per the r91 fuzz fix on bitstream.rs),
    // so the raw value is already bounded to fit in a u32; the
    // 0..=1023 range check below pins it to a 1024-element ceiling.
    let num_views_minus1 = r.ue()?;
    if num_views_minus1 > 1023 {
        return Err(SeiError::MultiviewViewPositionNumViewsOutOfRange(
            num_views_minus1,
        ));
    }
    let count = (num_views_minus1 as usize) + 1;
    let mut view_positions: Vec<u16> = Vec::with_capacity(count);
    for i in 0..count {
        // §G.13.1.10 — view_position[ i ] ue(v); §G.13.2.10 range
        // 0..=1023 inclusive. ue() returns u32 so the cast to u16
        // is safe only after the range check.
        let vp = r.ue()?;
        if vp > 1023 {
            return Err(SeiError::MultiviewViewPositionViewPositionOutOfRange { i, got: vp });
        }
        view_positions.push(vp as u16);
    }
    // §G.13.1.10 — multiview_view_position_extension_flag u(1).
    // §G.13.2.10: shall equal 0. We preserve the observed bit so
    // callers can audit a non-conforming stream rather than silently
    // collapsing it; we do NOT consume any trailing bits beyond
    // this flag (per §G.13.2.10 a `1` value means decoders shall
    // ignore everything after).
    let extension_flag = r.u(1)? == 1;
    Ok(MultiviewViewPosition {
        view_positions,
        extension_flag,
    })
}

/// §G.13.2.5 — multiview_acquisition_info (payload type 40, Annex G /
/// MVC).
///
/// Specifies the intrinsic and extrinsic camera parameters of every
/// view in an MVC coded video sequence. A 3D display compositor or a
/// post-render geometry pipeline can use these parameters to project
/// the per-view content according to the camera model defined in
/// §G.13.2.5 equation G-85
/// (`s * cP[i] = A[i] * R^{-1}[i] * (wP − T[i])`), where `A[i]` is the
/// intrinsic matrix (focal length / principal point / skew),
/// `R[i]` the rotation matrix, and `T[i]` the translation vector.
///
/// Per §G.13.2.5, when present as a non-nested SEI message the
/// payload shall be associated with an IDR access unit and the
/// information applies to the entire coded video sequence; when
/// nested inside an MVCD scalable nesting SEI message the loop
/// index `i` refers to `sei_view_id[i]` of the nesting message and
/// the application scope shrinks to the nesting message lifetime.
///
/// Each scalar component is signalled in a sign-exponent-mantissa
/// floating-point form (see [`FloatComponent`]) — `s` is a single
/// sign bit (`u(1)`); `e` is a 6-bit exponent (`u(6)`); the
/// mantissa is variable-width and computed from the per-block
/// precision exponent `prec_*`:
///
/// * If `e == 0`, the mantissa width is `max(0, prec − 30)`.
/// * If `0 < e < 63`, the mantissa width is `max(0, e + prec − 31)`.
/// * If `e == 63`, the spec reserves the value (decoders treat
///   it as "unspecified") and does not specify a mantissa width;
///   we treat the mantissa as zero-width in that case so the
///   bitstream cursor advances consistently regardless of which
///   reserved-future encoder produced the stream.
///
/// The decoded scalar value `x` reconstructs from a stored
/// `FloatComponent { sign, exponent: e, mantissa: n, mantissa_width: v }`
/// per §G.13.2.5 (Table G-3 below the syntax table) as:
///
/// * `e == 0`         → `x = (-1)^sign * 2^(−30) * (n / 2^v)`     (denormal)
/// * `0 < e < 63`     → `x = (-1)^sign * 2^(e − 31) * (1 + n / 2^v)` (normal)
/// * `e == 63`        → "unspecified" — callers should treat the
///   component as informational only.
///
/// Syntax — §G.13.1.5:
///
/// ```text
/// multiview_acquisition_info( payloadSize ) {
///   num_views_minus1                                 ue(v)
///   intrinsic_param_flag                             u(1)
///   extrinsic_param_flag                             u(1)
///   if( intrinsic_param_flag ) {
///     intrinsic_params_equal_flag                    u(1)
///     prec_focal_length                              ue(v)
///     prec_principal_point                           ue(v)
///     prec_skew_factor                               ue(v)
///     for( i = 0; i <= ( intrinsic_params_equal_flag ? 0 : num_views_minus1 ); i++ ) {
///       focal_length_x[ i ]                          /* sign u(1) + exponent u(6) + mantissa u(v) */
///       focal_length_y[ i ]
///       principal_point_x[ i ]
///       principal_point_y[ i ]
///       skew_factor[ i ]
///     }
///   }
///   if( extrinsic_param_flag ) {
///     prec_rotation_param                            ue(v)
///     prec_translation_param                         ue(v)
///     for( i = 0; i <= num_views_minus1; i++ ) {
///       for( j = 1; j <= 3; j++ ) {        /* row */
///         for( k = 1; k <= 3; k++ ) {      /* column */
///           r[ i ][ j ][ k ]               /* sign + exponent + mantissa */
///         }
///         t[ i ][ j ]                       /* sign + exponent + mantissa */
///       }
///     }
///   }
/// }
/// ```
///
/// Constraints — §G.13.2.5:
///
/// * `0 ≤ num_views_minus1 ≤ 1023` (Annex G absolute). Enforced
///   pre-allocation so an adversarial `ue(v)` cannot drive an
///   unbounded `Vec` (same anti-OOM rationale as
///   round-177 §D.2.20 and round-200 §G.13.2.8).
/// * Each of `prec_focal_length`, `prec_principal_point`,
///   `prec_skew_factor`, `prec_rotation_param`,
///   `prec_translation_param` shall be in `0..=31` inclusive.
/// * `exponent_*` is a `u(6)` so its raw range is `0..=63`. The
///   spec's range `0..=62` is informative — value 63 is reserved
///   for future use and decoders shall treat it as "unspecified";
///   we preserve the observed value without rejecting it so a
///   future-compatible bitstream remains parsable.
///
/// Note: this decoder does not (yet) carry the §G.7.3.2 MVC SPS —
/// Phase 4 on the README's "Profiles + features in scope" table.
/// The parser is nonetheless harmless on a non-MVC bitstream: when
/// this payload type appears outside an MVC access unit it can
/// still be parsed for inspection and logging without affecting
/// decode of the main (`view_id == 0`) sub-bitstream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiviewAcquisitionInfo {
    /// `num_views_minus1` — number of views in the CVS (minus one).
    /// Capped at 1023 (Annex G absolute) by the §G.13.2.5 range
    /// check that runs before allocation.
    pub num_views_minus1: u16,
    /// `intrinsic_param_flag == 1` populates `intrinsic`.
    pub intrinsic: Option<IntrinsicCameraParams>,
    /// `extrinsic_param_flag == 1` populates `extrinsic`.
    pub extrinsic: Option<ExtrinsicCameraParams>,
}

/// §G.13.2.5 — intrinsic camera parameters block (focal lengths,
/// principal point, skew factor). When
/// `intrinsic_params_equal_flag == 1` the block carries a single
/// per-CVS camera entry; otherwise it carries one entry per view
/// (`num_views_minus1 + 1` total).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntrinsicCameraParams {
    /// `intrinsic_params_equal_flag` — `true` if a single intrinsic
    /// set applies to every camera; `false` if a distinct set is
    /// signalled per camera.
    pub intrinsic_params_equal_flag: bool,
    /// `prec_focal_length` (0..=31) — sets the truncation precision
    /// for `focal_length_x[i]` and `focal_length_y[i]` via the
    /// mantissa-width formula in §G.13.2.5.
    pub prec_focal_length: u8,
    /// `prec_principal_point` (0..=31) — sets the truncation
    /// precision for `principal_point_x[i]` and `principal_point_y[i]`.
    pub prec_principal_point: u8,
    /// `prec_skew_factor` (0..=31) — sets the truncation precision
    /// for `skew_factor[i]`.
    pub prec_skew_factor: u8,
    /// Per-camera intrinsic entries. Length is either 1 (when
    /// `intrinsic_params_equal_flag == true`) or
    /// `num_views_minus1 + 1`.
    pub cameras: Vec<IntrinsicCamera>,
}

/// §G.13.2.5 — per-camera intrinsic parameters: focal lengths
/// (x, y), principal point (x, y), and skew factor. Each component
/// is a [`FloatComponent`] following the §G.13.2.5 floating-point
/// encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntrinsicCamera {
    pub focal_length_x: FloatComponent,
    pub focal_length_y: FloatComponent,
    pub principal_point_x: FloatComponent,
    pub principal_point_y: FloatComponent,
    pub skew_factor: FloatComponent,
}

/// §G.13.2.5 — extrinsic camera parameters block. Carries the 3×3
/// rotation matrix and 3-vector translation per camera.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtrinsicCameraParams {
    /// `prec_rotation_param` (0..=31) — sets the truncation
    /// precision for `r[i][j][k]`.
    pub prec_rotation_param: u8,
    /// `prec_translation_param` (0..=31) — sets the truncation
    /// precision for `t[i][j]`.
    pub prec_translation_param: u8,
    /// Per-camera extrinsic entries. Length is `num_views_minus1 + 1`.
    pub cameras: Vec<ExtrinsicCamera>,
}

/// §G.13.2.5 — per-camera extrinsic parameters: 3×3 rotation matrix
/// `r[j][k]` (rows and columns are 1-indexed in the spec; we store
/// in 0-indexed form so `r[j_spec − 1][k_spec − 1]` is the
/// (j_spec, k_spec) entry) and 3-vector translation `t[j]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtrinsicCamera {
    /// Rotation matrix entries. Spec 1-indexes `j, k ∈ {1, 2, 3}`;
    /// we store with `r[j_spec − 1][k_spec − 1]`.
    pub r: [[FloatComponent; 3]; 3],
    /// Translation vector entries. Spec 1-indexes `j ∈ {1, 2, 3}`;
    /// we store with `t[j_spec − 1]`.
    pub t: [FloatComponent; 3],
}

/// §G.13.2.5 — sign-exponent-mantissa floating-point component used
/// for every scalar in the multiview acquisition info SEI. The
/// stored value is the raw bitstream value; the
/// [`FloatComponent::to_f64`] helper reconstructs the IEEE-style
/// scalar per Table G-3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FloatComponent {
    /// `sign` — `false` for the positive sign-bit value (`u(1) == 0`),
    /// `true` for the negative (`u(1) == 1`).
    pub sign: bool,
    /// `exponent` — `u(6)`, raw range `0..=63`. Value 63 is reserved
    /// (decoders treat as "unspecified" per §G.13.2.5).
    pub exponent: u8,
    /// `mantissa` — variable-width `u(v)` raw bits; width is
    /// recorded in `mantissa_width`. The width is at most 62 bits so
    /// it fits in `u64` (62 = 62 + 31 − 31, the §G.13.2.5 maximum).
    pub mantissa: u64,
    /// `mantissa_width` — number of bits in `mantissa`, derived per
    /// §G.13.2.5 from the per-block precision and the exponent.
    /// Bounded by 62.
    pub mantissa_width: u8,
}

impl FloatComponent {
    /// Reconstruct the §G.13.2.5 floating-point scalar value:
    ///
    /// * `e == 0`     → `(-1)^s * 2^(−30) * (n / 2^v)`         (denormal)
    /// * `0 < e < 63` → `(-1)^s * 2^(e − 31) * (1 + n / 2^v)`  (normal)
    /// * `e == 63`    → `f64::NAN` ("unspecified" per the spec).
    ///
    /// `v == 0` is handled implicitly: `n / 2^0 = n = 0` (parser
    /// guarantees the stored mantissa is 0 when width is 0).
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        if self.exponent == 63 {
            return f64::NAN;
        }
        let sign: f64 = if self.sign { -1.0 } else { 1.0 };
        let v = self.mantissa_width as i32;
        let n_over_2v = if v == 0 {
            0.0
        } else {
            (self.mantissa as f64) / (2f64.powi(v))
        };
        if self.exponent == 0 {
            sign * 2f64.powi(-30) * n_over_2v
        } else {
            sign * 2f64.powi((self.exponent as i32) - 31) * (1.0 + n_over_2v)
        }
    }
}

/// §G.13.2.5 — derive the mantissa bit-width from a precision
/// `prec` (0..=31) and an exponent `e` (0..=63).
///
/// * `e == 0`     → `max(0, prec − 30)`
/// * `0 < e < 63` → `max(0, e + prec − 31)`
/// * `e == 63`    → `0` (spec reserves the value; no mantissa)
///
/// The maximum is `62 + 31 − 31 = 62` bits, comfortably within
/// `u64`.
fn mantissa_width_g1325(prec: u8, e: u8) -> u8 {
    debug_assert!(prec <= 31);
    if e == 0 {
        prec.saturating_sub(30)
    } else if e == 63 {
        0
    } else {
        // 0 < e < 63 → e + prec − 31, clamped at 0.
        let sum = e as i32 + prec as i32 - 31;
        if sum <= 0 {
            0
        } else {
            sum as u8
        }
    }
}

/// Read a §G.13.2.5 sign-exponent-mantissa component from `r`,
/// using `prec` to derive the mantissa width. Returns the raw
/// bitstream value; see [`FloatComponent::to_f64`] for the
/// reconstructed scalar.
fn read_float_component(r: &mut BitReader<'_>, prec: u8) -> Result<FloatComponent, SeiError> {
    // sign u(1).
    let sign = r.u(1)? == 1;
    // exponent u(6).
    let exponent = r.u(6)? as u8;
    // mantissa u(v) where v is derived per §G.13.2.5.
    let v = mantissa_width_g1325(prec, exponent);
    let mantissa: u64 = if v == 0 {
        0
    } else if v <= 32 {
        r.u(v as u32)? as u64
    } else {
        // 32 < v <= 62 — split into hi (v − 32 bits) + lo (32 bits)
        // so each individual r.u call stays within the
        // u(<=32) primitive contract.
        let hi_bits = (v - 32) as u32;
        let hi = r.u(hi_bits)? as u64;
        let lo = r.u(32)? as u64;
        (hi << 32) | lo
    };
    Ok(FloatComponent {
        sign,
        exponent,
        mantissa,
        mantissa_width: v,
    })
}

/// Parse a §G.13.1.5 `multiview_acquisition_info()` payload.
///
/// Enforces:
///
/// * `num_views_minus1 ≤ 1023` (Annex G absolute, anti-OOM
///   pre-allocation cap — cf. round-200 §G.13.2.8).
/// * Each of `prec_focal_length`, `prec_principal_point`,
///   `prec_skew_factor`, `prec_rotation_param`,
///   `prec_translation_param` shall be in `0..=31` inclusive.
///
/// Floating-point components are stored verbatim in
/// [`FloatComponent`] (no IEEE conversion at parse time — callers
/// can call [`FloatComponent::to_f64`] when they need a scalar).
pub fn parse_multiview_acquisition_info(
    payload: &[u8],
) -> Result<MultiviewAcquisitionInfo, SeiError> {
    let mut r = BitReader::new(payload);

    // §G.13.1.5 — num_views_minus1 ue(v).
    // §G.13.2.5 / Annex G absolute: 0..=1023.
    let num_views_minus1 = r.ue()?;
    if num_views_minus1 > 1023 {
        return Err(SeiError::MultiviewAcquisitionInfoNumViewsOutOfRange(
            num_views_minus1,
        ));
    }
    let num_views = (num_views_minus1 as usize) + 1;

    // §G.13.1.5 — intrinsic_param_flag u(1), extrinsic_param_flag u(1).
    let intrinsic_param_flag = r.u(1)? == 1;
    let extrinsic_param_flag = r.u(1)? == 1;

    let intrinsic = if intrinsic_param_flag {
        // §G.13.1.5 — intrinsic_params_equal_flag u(1).
        let intrinsic_params_equal_flag = r.u(1)? == 1;
        // §G.13.1.5 — prec_focal_length, prec_principal_point,
        // prec_skew_factor ue(v). §G.13.2.5: each shall be in
        // 0..=31 inclusive.
        let prec_focal_length = read_prec_ue(&mut r, "prec_focal_length")?;
        let prec_principal_point = read_prec_ue(&mut r, "prec_principal_point")?;
        let prec_skew_factor = read_prec_ue(&mut r, "prec_skew_factor")?;

        // §G.13.1.5 — when intrinsic_params_equal_flag == 1, only
        // the i = 0 entry is signalled; otherwise the loop covers
        // i = 0..=num_views_minus1.
        let cam_count = if intrinsic_params_equal_flag {
            1
        } else {
            num_views
        };
        let mut cameras: Vec<IntrinsicCamera> = Vec::with_capacity(cam_count);
        for _ in 0..cam_count {
            let focal_length_x = read_float_component(&mut r, prec_focal_length)?;
            let focal_length_y = read_float_component(&mut r, prec_focal_length)?;
            let principal_point_x = read_float_component(&mut r, prec_principal_point)?;
            let principal_point_y = read_float_component(&mut r, prec_principal_point)?;
            let skew_factor = read_float_component(&mut r, prec_skew_factor)?;
            cameras.push(IntrinsicCamera {
                focal_length_x,
                focal_length_y,
                principal_point_x,
                principal_point_y,
                skew_factor,
            });
        }

        Some(IntrinsicCameraParams {
            intrinsic_params_equal_flag,
            prec_focal_length,
            prec_principal_point,
            prec_skew_factor,
            cameras,
        })
    } else {
        None
    };

    let extrinsic = if extrinsic_param_flag {
        // §G.13.1.5 — prec_rotation_param, prec_translation_param
        // ue(v). §G.13.2.5: each shall be in 0..=31 inclusive.
        let prec_rotation_param = read_prec_ue(&mut r, "prec_rotation_param")?;
        let prec_translation_param = read_prec_ue(&mut r, "prec_translation_param")?;

        let mut cameras: Vec<ExtrinsicCamera> = Vec::with_capacity(num_views);
        // The spec's outer loop runs i = 0..=num_views_minus1
        // (the §G.13.1.5 loop comment differs from the intrinsic
        // block in that there is no intrinsic_params_equal_flag
        // shortcut — extrinsics are always per-camera).
        for _ in 0..num_views {
            // Initialise with a zero placeholder we'll overwrite —
            // `FloatComponent` is Copy so this is cheap.
            let zero = FloatComponent {
                sign: false,
                exponent: 0,
                mantissa: 0,
                mantissa_width: 0,
            };
            let mut r_mat: [[FloatComponent; 3]; 3] = [[zero; 3]; 3];
            let mut t_vec: [FloatComponent; 3] = [zero; 3];
            // §G.13.1.5 — for( j = 1; j <= 3; j++ ) { for( k = 1; k <= 3; k++ ) r[i][j][k]; t[i][j]; }
            // i.e., row j carries three r entries (k=1..3) followed
            // by a single t entry. We store with 0-indexed j, k.
            for (j, row) in r_mat.iter_mut().enumerate() {
                for cell in row.iter_mut() {
                    *cell = read_float_component(&mut r, prec_rotation_param)?;
                }
                t_vec[j] = read_float_component(&mut r, prec_translation_param)?;
            }
            cameras.push(ExtrinsicCamera { r: r_mat, t: t_vec });
        }

        Some(ExtrinsicCameraParams {
            prec_rotation_param,
            prec_translation_param,
            cameras,
        })
    } else {
        None
    };

    Ok(MultiviewAcquisitionInfo {
        num_views_minus1: num_views_minus1 as u16,
        intrinsic,
        extrinsic,
    })
}

/// Helper: read a `prec_*` ue(v) and enforce the §G.13.2.5
/// `0..=31` range bound. The `field` argument is the spec name
/// surfaced in the error.
fn read_prec_ue(r: &mut BitReader<'_>, field: &'static str) -> Result<u8, SeiError> {
    let v = r.ue()?;
    if v > 31 {
        return Err(SeiError::MultiviewAcquisitionInfoPrecOutOfRange { field, got: v });
    }
    Ok(v as u8)
}

/// §H.13.2.4 — sign-less floating-point component used by the 3D
/// reference displays info SEI. Same exponent / mantissa-width
/// formula as [`FloatComponent`] (§G.13.2.5), but the spec's Table
/// H-3 fixes the sign column at 0 for every variable, so no sign
/// bit is read from the bitstream.
///
/// The stored value is the raw bitstream value; the
/// [`UnsignedFloatComponent::to_f64`] helper reconstructs the
/// IEEE-style scalar per Table H-3:
///
/// * `0 < e < 63` → `2^(e − 31) * (1 + n ÷ 2^v)`         (normal)
/// * `e == 0`     → `2^−(30 + v) * n`                    (denormal)
/// * `e == 63`    → unspecified; we return `f64::NAN`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsignedFloatComponent {
    /// `exponent` — `u(6)`, raw range `0..=63`. Value 63 is
    /// reserved (treated as "unspecified" per §H.13.2.4).
    pub exponent: u8,
    /// `mantissa` — variable-width `u(v)` raw bits; width is
    /// recorded in `mantissa_width` and is at most 62.
    pub mantissa: u64,
    /// `mantissa_width` — bit-width of `mantissa`, derived per
    /// §H.13.2.4 (same formula as §G.13.2.5) from the per-block
    /// precision and the exponent.
    pub mantissa_width: u8,
}

impl UnsignedFloatComponent {
    /// Reconstruct the §H.13.2.4 scalar value per Table H-3 with
    /// `s = 0`. Mirrors [`FloatComponent::to_f64`] minus the sign.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        if self.exponent == 63 {
            return f64::NAN;
        }
        let v = self.mantissa_width as i32;
        let n_over_2v = if v == 0 {
            0.0
        } else {
            (self.mantissa as f64) / (2f64.powi(v))
        };
        if self.exponent == 0 {
            2f64.powi(-30) * n_over_2v
        } else {
            2f64.powi((self.exponent as i32) - 31) * (1.0 + n_over_2v)
        }
    }
}

/// Read a §H.13.2.4 unsigned floating-point component from `r`,
/// using `prec` to derive the mantissa width. The §H.13.2.4
/// mantissa-width formula is identical to the §G.13.2.5 one used
/// by [`mantissa_width_g1325`].
fn read_unsigned_float_component(
    r: &mut BitReader<'_>,
    prec: u8,
) -> Result<UnsignedFloatComponent, SeiError> {
    // exponent u(6).
    let exponent = r.u(6)? as u8;
    let v = mantissa_width_g1325(prec, exponent);
    let mantissa: u64 = if v == 0 {
        0
    } else if v <= 32 {
        r.u(v as u32)? as u64
    } else {
        // 32 < v <= 62 — split into hi (v − 32 bits) + lo (32 bits)
        // so each individual r.u call stays within the
        // u(<=32) primitive contract.
        let hi_bits = (v - 32) as u32;
        let hi = r.u(hi_bits)? as u64;
        let lo = r.u(32)? as u64;
        (hi << 32) | lo
    };
    Ok(UnsignedFloatComponent {
        exponent,
        mantissa,
        mantissa_width: v,
    })
}

/// §H.13.2.4 — 3D reference displays information SEI message.
///
/// Carries the reference display widths, reference viewing
/// distances (optional), and the corresponding baseline distances
/// and additional horizontal shifts that form a stereo pair for
/// each reference display. Used by view renderers to compute a
/// proper stereo pair for the target screen width and viewing
/// distance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreeDimensionalReferenceDisplaysInfo {
    /// `prec_ref_baseline` (0..=31) — truncation exponent for
    /// `ref_baseline[i]`.
    pub prec_ref_baseline: u8,
    /// `prec_ref_display_width` (0..=31) — truncation exponent for
    /// `ref_display_width[i]`.
    pub prec_ref_display_width: u8,
    /// `prec_ref_viewing_dist` (0..=31) — truncation exponent for
    /// `ref_viewing_distance[i]`. Present iff
    /// `ref_viewing_distance_flag == 1`.
    pub prec_ref_viewing_dist: Option<u8>,
    /// One entry per reference display
    /// (`num_ref_displays_minus1 + 1`, capped at 32).
    pub displays: Vec<ReferenceDisplay>,
    /// `three_dimensional_reference_displays_extension_flag` — the
    /// spec mandates 0 in this edition; non-zero values are
    /// reserved for future use and a decoder shall ignore any
    /// trailing data.
    pub extension_flag: bool,
}

/// §H.13.2.4 — per-reference-display entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceDisplay {
    /// `ref_baseline[i]` — reference baseline distance, in the same
    /// units as the x component of the translation vector in the
    /// associated multiview_acquisition_info SEI.
    pub ref_baseline: UnsignedFloatComponent,
    /// `ref_display_width[i]` — reference display width in
    /// centimetres.
    pub ref_display_width: UnsignedFloatComponent,
    /// `ref_viewing_distance[i]` — reference viewing distance in
    /// centimetres. Present iff `ref_viewing_distance_flag == 1`.
    pub ref_viewing_distance: Option<UnsignedFloatComponent>,
    /// `additional_shift_present_flag[i]` — whether `num_sample_shift_plus512`
    /// is signalled for this display.
    pub additional_shift_present_flag: bool,
    /// `num_sample_shift_plus512[i]` — recommended additional
    /// horizontal shift in samples between the left and right views
    /// (biased by 512). Present iff
    /// `additional_shift_present_flag == 1`. Range `0..=1023`.
    pub num_sample_shift_plus512: Option<u16>,
}

impl ReferenceDisplay {
    /// §H.13.2.4 — recommended additional horizontal shift between
    /// the left and right views, decoded from the biased
    /// `num_sample_shift_plus512[i]` field into a signed sample
    /// count.
    ///
    /// The bitstream stores the shift as `num_sample_shift_plus512`
    /// with the bias `+ 512`, giving a raw range of `0..=1023`. The
    /// semantic shift `NumSampleShift[i]` defined by §H.13.2.4 is
    ///
    /// ```text
    /// NumSampleShift[i] = num_sample_shift_plus512[i] − 512
    /// ```
    ///
    /// which spans `−512..=511`. The sign convention from §H.13.2.4
    /// is:
    ///
    /// * `< 0` — recommend shifting the **left** view to the left by
    ///   `512 − num_sample_shift_plus512[i]` samples (i.e. by
    ///   `|NumSampleShift|`) relative to the right view.
    /// * `= 0` — recommend that no shift be applied.
    /// * `> 0` — recommend shifting the **left** view to the right
    ///   by `num_sample_shift_plus512[i] − 512` samples (i.e. by
    ///   `NumSampleShift`) relative to the right view.
    ///
    /// Returns `None` when `additional_shift_present_flag == 0` (the
    /// shift is unsignalled and §H.13.2.4 does not define an
    /// inferred value); returns `Some(NumSampleShift[i])` otherwise.
    ///
    /// The return type is `i16` since the value fits in
    /// `−512..=511`.
    #[must_use]
    pub fn num_sample_shift(&self) -> Option<i16> {
        self.num_sample_shift_plus512
            .map(|biased| (biased as i32 - 512) as i16)
    }
}

/// Parse a §H.13.1.4 `three_dimensional_reference_displays_info()`
/// payload.
///
/// Enforces:
///
/// * `prec_ref_baseline`, `prec_ref_display_width`,
///   `prec_ref_viewing_dist` shall be in `0..=31` (§H.13.2.4).
/// * `num_ref_displays_minus1` shall be in `0..=31` (§H.13.2.4) —
///   serves as the pre-allocation cap for the per-display loop.
///
/// `num_sample_shift_plus512` is read as `u(10)`, naturally
/// constrained to `0..=1023` by the bit width.
///
/// `three_dimensional_reference_displays_extension_flag` is read
/// but no further data is parsed when it is 1 — the spec reserves
/// the trailing region for future extensions and decoders shall
/// ignore everything after this flag.
pub fn parse_three_dimensional_reference_displays_info(
    payload: &[u8],
) -> Result<ThreeDimensionalReferenceDisplaysInfo, SeiError> {
    let mut r = BitReader::new(payload);

    // prec_ref_baseline ue(v), 0..=31.
    let prec_ref_baseline = {
        let v = r.ue()?;
        if v > 31 {
            return Err(
                SeiError::ThreeDimensionalReferenceDisplaysInfoPrecOutOfRange {
                    field: "prec_ref_baseline",
                    got: v,
                },
            );
        }
        v as u8
    };
    // prec_ref_display_width ue(v), 0..=31.
    let prec_ref_display_width = {
        let v = r.ue()?;
        if v > 31 {
            return Err(
                SeiError::ThreeDimensionalReferenceDisplaysInfoPrecOutOfRange {
                    field: "prec_ref_display_width",
                    got: v,
                },
            );
        }
        v as u8
    };
    // ref_viewing_distance_flag u(1).
    let ref_viewing_distance_flag = r.u(1)? == 1;
    // prec_ref_viewing_dist ue(v), 0..=31. Present iff
    // ref_viewing_distance_flag == 1.
    let prec_ref_viewing_dist = if ref_viewing_distance_flag {
        let v = r.ue()?;
        if v > 31 {
            return Err(
                SeiError::ThreeDimensionalReferenceDisplaysInfoPrecOutOfRange {
                    field: "prec_ref_viewing_dist",
                    got: v,
                },
            );
        }
        Some(v as u8)
    } else {
        None
    };
    // num_ref_displays_minus1 ue(v), 0..=31. Anti-OOM cap before
    // the per-display loop allocation.
    let num_ref_displays_minus1 = r.ue()?;
    if num_ref_displays_minus1 > 31 {
        return Err(
            SeiError::ThreeDimensionalReferenceDisplaysInfoNumRefDisplaysOutOfRange(
                num_ref_displays_minus1,
            ),
        );
    }
    let num_ref_displays = (num_ref_displays_minus1 as usize) + 1;
    let mut displays = Vec::with_capacity(num_ref_displays);
    for _ in 0..num_ref_displays {
        // exponent_ref_baseline u(6) + mantissa_ref_baseline u(v).
        let ref_baseline = read_unsigned_float_component(&mut r, prec_ref_baseline)?;
        // exponent_ref_display_width u(6) + mantissa_ref_display_width u(v).
        let ref_display_width = read_unsigned_float_component(&mut r, prec_ref_display_width)?;
        // exponent_ref_viewing_distance u(6) + mantissa u(v), iff
        // ref_viewing_distance_flag == 1.
        let ref_viewing_distance = if let Some(prec) = prec_ref_viewing_dist {
            Some(read_unsigned_float_component(&mut r, prec)?)
        } else {
            None
        };
        // additional_shift_present_flag u(1).
        let additional_shift_present_flag = r.u(1)? == 1;
        // num_sample_shift_plus512 u(10) iff additional_shift_present_flag == 1.
        let num_sample_shift_plus512 = if additional_shift_present_flag {
            Some(r.u(10)? as u16)
        } else {
            None
        };
        displays.push(ReferenceDisplay {
            ref_baseline,
            ref_display_width,
            ref_viewing_distance,
            additional_shift_present_flag,
            num_sample_shift_plus512,
        });
    }
    // three_dimensional_reference_displays_extension_flag u(1).
    let extension_flag = r.u(1)? == 1;

    Ok(ThreeDimensionalReferenceDisplaysInfo {
        prec_ref_baseline,
        prec_ref_display_width,
        prec_ref_viewing_dist,
        displays,
        extension_flag,
    })
}

/// §H.13.2.3.1 — depth representation SEI element.
///
/// Carries the raw `(s, e, n, v)` quadruple for one floating-point
/// scalar in the depth representation information message. The
/// scalar uses a different syntax shape than the
/// §G.13.2.5 / §H.13.2.4 floats handled by [`FloatComponent`] /
/// [`UnsignedFloatComponent`]:
///
/// * sign — `u(1)` (`da_sign_flag`)
/// * exponent — `u(7)` (`da_exponent`, 0..=126; 127 is reserved for
///   future use and shall be treated as unspecified)
/// * mantissa width — `u(5) + 1` (`da_mantissa_len_minus1 + 1`,
///   1..=32)
/// * mantissa — `u(v)` (`da_mantissa`)
///
/// The §H.13.2.3 reconstruction formula in Table H-2 is:
///
/// * `0 < e < 127` → `x = (−1)^s * 2^(e − 31) * (1 + n / 2^v)`
/// * `e == 0`      → `x = (−1)^s * 2^(−(30 + v)) * n`
///
/// The stored value is the raw bitstream value; [`DepthFloatComponent::to_f64`]
/// reconstructs the scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthFloatComponent {
    /// `da_sign_flag` — `false` for positive (`u(1) == 0`), `true` for
    /// negative (`u(1) == 1`).
    pub sign: bool,
    /// `da_exponent` — `u(7)`, raw range `0..=127`. Value 127 is
    /// reserved (decoders shall treat as "unspecified" per
    /// §H.13.2.3.1).
    pub exponent: u8,
    /// `da_mantissa` — variable-width `u(v)` raw bits.
    /// `da_mantissa_len_minus1` is `u(5)` so the width is in
    /// `1..=32`; the mantissa fits in `u32`, stored here as `u64` for
    /// uniformity with the other float components.
    pub mantissa: u64,
    /// `da_mantissa_len_minus1 + 1` — bit-width of `mantissa`,
    /// in `1..=32`.
    pub mantissa_width: u8,
}

impl DepthFloatComponent {
    /// Reconstruct the §H.13.2.3 floating-point scalar value per the
    /// Table H-2 formula.
    ///
    /// * `e == 0`      → `(-1)^s * 2^(-(30 + v)) * n`        (denormal)
    /// * `0 < e < 127` → `(-1)^s * 2^(e - 31) * (1 + n / 2^v)` (normal)
    /// * `e == 127`    → `f64::NAN` ("unspecified" per the spec).
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        if self.exponent == 127 {
            return f64::NAN;
        }
        let sign: f64 = if self.sign { -1.0 } else { 1.0 };
        let v = self.mantissa_width as i32;
        let n_over_2v = if v == 0 {
            0.0
        } else {
            (self.mantissa as f64) / (2f64.powi(v))
        };
        if self.exponent == 0 {
            // 2^(-(30 + v)) * n  =  2^(-30) * (n / 2^v).
            sign * 2f64.powi(-30) * n_over_2v
        } else {
            sign * 2f64.powi((self.exponent as i32) - 31) * (1.0 + n_over_2v)
        }
    }
}

/// Read a §H.13.1.3.1 `depth_representation_sei_element` from `r`.
fn read_depth_float_component(r: &mut BitReader<'_>) -> Result<DepthFloatComponent, SeiError> {
    // da_sign_flag u(1).
    let sign = r.u(1)? == 1;
    // da_exponent u(7).
    let exponent = r.u(7)? as u8;
    // da_mantissa_len_minus1 u(5). Width is 1..=32.
    let mantissa_len_minus1 = r.u(5)? as u8;
    let mantissa_width = mantissa_len_minus1 + 1;
    // da_mantissa u(v). v is in 1..=32 so a single u(<=32) read
    // satisfies the BitReader primitive contract.
    let mantissa = r.u(mantissa_width as u32)? as u64;
    Ok(DepthFloatComponent {
        sign,
        exponent,
        mantissa,
        mantissa_width,
    })
}

/// §H.13.2.3 — depth representation information SEI message.
///
/// Carries depth / disparity range parameters per view used by a
/// renderer prior to display (e.g. for view synthesis on a 3D
/// display). When `all_views_equal_flag == 1`, the message describes
/// one shared view (so `views` has length 1); otherwise it describes
/// `num_views_minus1 + 1` views.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepthRepresentationInfo {
    /// `all_views_equal_flag` (u(1)) — `true` when the parameters
    /// apply equally to every target view (`views` carries one shared
    /// entry); `false` otherwise.
    pub all_views_equal_flag: bool,
    /// `z_near_flag` (u(1)) — whether `z_near` is signalled in each
    /// view entry.
    pub z_near_flag: bool,
    /// `z_far_flag` (u(1)) — whether `z_far` is signalled in each
    /// view entry.
    pub z_far_flag: bool,
    /// `z_axis_equal_flag` (u(1)) — present iff `z_near_flag ||
    /// z_far_flag`; `None` otherwise. `true` means all per-view
    /// `z_near` / `z_far` values share the same Z-axis (signalled
    /// once in `common_z_axis_reference_view`); `false` means each
    /// view carries its own `z_axis_reference_view[i]`.
    pub z_axis_equal_flag: Option<bool>,
    /// `common_z_axis_reference_view` (ue(v), 0..=1023) — present iff
    /// `z_axis_equal_flag == 1`.
    pub common_z_axis_reference_view: Option<u16>,
    /// `d_min_flag` (u(1)) — whether `d_min` is signalled in each
    /// view entry.
    pub d_min_flag: bool,
    /// `d_max_flag` (u(1)) — whether `d_max` is signalled in each
    /// view entry.
    pub d_max_flag: bool,
    /// `depth_representation_type` (ue(v)) — Table H-1 interpretation
    /// of the decoded depth-view luma samples. 0..=3 are defined by
    /// the spec; 4..=15 are reserved (and the spec requires decoders
    /// to ignore any data that follows a reserved value, so the
    /// nonlinear-representation tail below is skipped in that case).
    pub depth_representation_type: u32,
    /// Per-view entries (length 1 if `all_views_equal_flag`,
    /// otherwise `num_views_minus1 + 1`).
    pub views: Vec<DepthRepresentationView>,
    /// `depth_nonlinear_representation_num_minus1` (ue(v), 0..=62) —
    /// present iff `depth_representation_type == 3`.
    pub depth_nonlinear_representation_num_minus1: Option<u8>,
    /// `depth_nonlinear_representation_model[i]` for `i` in
    /// `1..=depth_nonlinear_representation_num_minus1 + 1`. Each
    /// entry is `ue(v)` in `0..=65535`. Present iff
    /// `depth_representation_type == 3`. The trailing-coverage `i=0`
    /// / `i = num_minus1 + 2` slots are pre-defined to 0 by the spec
    /// (§H.13.2.3 DepthLUT construction) and are not signalled.
    pub depth_nonlinear_representation_model: Vec<u16>,
}

impl DepthRepresentationInfo {
    /// §H.13.2.3 — the spec semantic value
    /// `DepthNonlinearRepresentationNumSegments`: the number of
    /// piecewise linear segments used to map decoded depth-view luma
    /// samples to a scale uniformly quantised in disparity.
    ///
    /// The bitstream stores the count as the biased field
    /// `depth_nonlinear_representation_num_minus1` with the bias
    /// `+ 2` (per the spec sentence:
    /// *"depth_nonlinear_representation_num_minus1 plus 2 specifies
    /// the number of piecewise linear segments…"*). Combined with
    /// the on-wire range `0..=62`, the semantic value lies in
    /// `2..=64`, which always fits in `u8`.
    ///
    /// Returns `None` when the field is absent — i.e. when
    /// `depth_representation_type != 3` and the nonlinear-model
    /// tail is therefore not signalled. The §H.13.2.3 piecewise
    /// model is only defined for `depth_representation_type == 3`,
    /// so no inferred segment count exists for other types.
    ///
    /// Note that the signalled model carries
    /// `num_minus1 + 1` entries (see
    /// [`Self::depth_nonlinear_representation_model_len`]); the
    /// segment count is `num_minus1 + 2` because the §H.13.2.3
    /// DepthLUT construction loops the piecewise linear interpolant
    /// over indices `k = 0..=num_minus1 + 1`, yielding
    /// `num_minus1 + 2` segments delimited by the `num_minus1 + 3`
    /// model points `model[0..=num_minus1 + 2]` (with
    /// `model[0]` and `model[num_minus1 + 2]` pre-defined to `0`
    /// and not signalled).
    #[must_use]
    pub fn depth_nonlinear_representation_num_segments(&self) -> Option<u8> {
        self.depth_nonlinear_representation_num_minus1
            .map(|num_minus1| num_minus1 + 2)
    }

    /// §H.13.2.3 — the number of *signalled*
    /// `depth_nonlinear_representation_model[i]` entries in the
    /// bitstream, equal to `depth_nonlinear_representation_num_minus1
    /// + 1`.
    ///
    /// The spec's syntax loop in §H.13.1.3 runs
    ///
    /// ```text
    /// for( i = 1; i <= depth_nonlinear_representation_num_minus1 + 1; i++ )
    ///     depth_nonlinear_representation_model[ i ]
    /// ```
    ///
    /// so the on-wire count is `num_minus1 + 1`. The two
    /// trailing-coverage sentinels `model[0]` and
    /// `model[num_minus1 + 2]` defined by the §H.13.2.3 DepthLUT
    /// construction are pre-defined to `0` and are not part of the
    /// bitstream — this accessor counts only the entries the
    /// decoder actually parsed (and that are stored in
    /// [`Self::depth_nonlinear_representation_model`]).
    ///
    /// Returns `None` when the field is absent. With the on-wire
    /// range `num_minus1 ∈ 0..=62`, the signalled count lies in
    /// `1..=63` and always fits in `u8`.
    #[must_use]
    pub fn depth_nonlinear_representation_model_len(&self) -> Option<u8> {
        self.depth_nonlinear_representation_num_minus1
            .map(|num_minus1| num_minus1 + 1)
    }
}

/// §H.13.2.3 — per-view depth representation parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthRepresentationView {
    /// `depth_info_view_id[i]` (ue(v), 0..=1023) — the `view_id` to
    /// which the rest of the entry applies. When the outer struct's
    /// `all_views_equal_flag` is 1, this carries the spec's view_id
    /// for the single shared entry.
    pub depth_info_view_id: u16,
    /// `z_axis_reference_view[i]` (ue(v), 0..=1023) — present iff
    /// `(z_near_flag || z_far_flag) && !z_axis_equal_flag`.
    pub z_axis_reference_view: Option<u16>,
    /// `disparity_reference_view[i]` (ue(v), 0..=1023) — present iff
    /// `d_min_flag || d_max_flag`.
    pub disparity_reference_view: Option<u16>,
    /// `ZNear[i]` quadruple — present iff `z_near_flag`.
    pub z_near: Option<DepthFloatComponent>,
    /// `ZFar[i]` quadruple — present iff `z_far_flag`.
    pub z_far: Option<DepthFloatComponent>,
    /// `DMin[i]` quadruple — present iff `d_min_flag`.
    pub d_min: Option<DepthFloatComponent>,
    /// `DMax[i]` quadruple — present iff `d_max_flag`.
    pub d_max: Option<DepthFloatComponent>,
}

/// Parse a §H.13.1.3 `depth_representation_info()` payload.
///
/// Enforces the §H.13.2.3 range bounds before allocation:
///
/// * `num_views_minus1 ≤ 1023` (pre-allocation cap on the per-view
///   loop).
/// * `common_z_axis_reference_view` ∈ `0..=1023`.
/// * `depth_info_view_id[i]`, `z_axis_reference_view[i]`,
///   `disparity_reference_view[i]` ∈ `0..=1023`.
/// * `depth_nonlinear_representation_num_minus1 ≤ 62`
///   (pre-allocation cap on the model loop; spec range
///   `0..=62`).
/// * `depth_nonlinear_representation_model[i]` ∈ `0..=65535`.
///
/// `depth_representation_type` in the reserved range `4..=15` is
/// accepted (per §H.13.2.3 the decoder shall ignore trailing data);
/// the nonlinear-model tail is parsed only when the value is exactly
/// 3.
pub fn parse_depth_representation_info(
    payload: &[u8],
) -> Result<DepthRepresentationInfo, SeiError> {
    let mut r = BitReader::new(payload);

    // all_views_equal_flag u(1).
    let all_views_equal_flag = r.u(1)? == 1;

    // num_views_minus1 ue(v), 0..=1023 (anti-OOM cap before the
    // per-view loop allocation). Present iff all_views_equal_flag
    // == 0; when 1 the loop runs once over a shared entry.
    let num_views = if all_views_equal_flag {
        1usize
    } else {
        let v = r.ue()?;
        if v > 1023 {
            return Err(SeiError::DepthRepresentationInfoNumViewsOutOfRange(v));
        }
        (v as usize) + 1
    };

    // z_near_flag u(1), z_far_flag u(1).
    let z_near_flag = r.u(1)? == 1;
    let z_far_flag = r.u(1)? == 1;

    // z_axis_equal_flag u(1) + common_z_axis_reference_view ue(v)
    // gated by (z_near_flag || z_far_flag).
    let (z_axis_equal_flag, common_z_axis_reference_view) = if z_near_flag || z_far_flag {
        let z_axis_equal_flag = r.u(1)? == 1;
        let common = if z_axis_equal_flag {
            let v = r.ue()?;
            if v > 1023 {
                return Err(SeiError::DepthRepresentationInfoViewIdOutOfRange {
                    field: "common_z_axis_reference_view",
                    got: v,
                });
            }
            Some(v as u16)
        } else {
            None
        };
        (Some(z_axis_equal_flag), common)
    } else {
        (None, None)
    };

    // d_min_flag u(1), d_max_flag u(1).
    let d_min_flag = r.u(1)? == 1;
    let d_max_flag = r.u(1)? == 1;

    // depth_representation_type ue(v). 0..=3 defined, 4..=15
    // reserved (per §H.13.2.3 the decoder shall ignore the tail when
    // the value is reserved). Values outside 0..=15 are not
    // explicitly bounded by §H.13.2.3 (the prose says "0 to 3,
    // inclusive, in bitstreams conforming to this version" + "4 to
    // 15, inclusive" reserved), so we store the raw ue(v) value
    // without an upper bound: §H.13.2.3 instructs the decoder to
    // ignore the tail for any non-{0..=3} value, which we honour by
    // only parsing the §H.13.2.3 nonlinear tail when the value is
    // exactly 3.
    let depth_representation_type = r.ue()?;

    let mut views = Vec::with_capacity(num_views);
    for _ in 0..num_views {
        // depth_info_view_id[i] ue(v), 0..=1023.
        let depth_info_view_id = {
            let v = r.ue()?;
            if v > 1023 {
                return Err(SeiError::DepthRepresentationInfoViewIdOutOfRange {
                    field: "depth_info_view_id",
                    got: v,
                });
            }
            v as u16
        };
        // z_axis_reference_view[i] ue(v), 0..=1023, iff
        // (z_near_flag || z_far_flag) && !z_axis_equal_flag.
        let z_axis_reference_view =
            if (z_near_flag || z_far_flag) && z_axis_equal_flag == Some(false) {
                let v = r.ue()?;
                if v > 1023 {
                    return Err(SeiError::DepthRepresentationInfoViewIdOutOfRange {
                        field: "z_axis_reference_view",
                        got: v,
                    });
                }
                Some(v as u16)
            } else {
                None
            };
        // disparity_reference_view[i] ue(v), 0..=1023, iff
        // d_min_flag || d_max_flag.
        let disparity_reference_view = if d_min_flag || d_max_flag {
            let v = r.ue()?;
            if v > 1023 {
                return Err(SeiError::DepthRepresentationInfoViewIdOutOfRange {
                    field: "disparity_reference_view",
                    got: v,
                });
            }
            Some(v as u16)
        } else {
            None
        };
        // Optional ZNear[i] depth_representation_sei_element.
        let z_near = if z_near_flag {
            Some(read_depth_float_component(&mut r)?)
        } else {
            None
        };
        // Optional ZFar[i].
        let z_far = if z_far_flag {
            Some(read_depth_float_component(&mut r)?)
        } else {
            None
        };
        // Optional DMin[i].
        let d_min = if d_min_flag {
            Some(read_depth_float_component(&mut r)?)
        } else {
            None
        };
        // Optional DMax[i].
        let d_max = if d_max_flag {
            Some(read_depth_float_component(&mut r)?)
        } else {
            None
        };

        views.push(DepthRepresentationView {
            depth_info_view_id,
            z_axis_reference_view,
            disparity_reference_view,
            z_near,
            z_far,
            d_min,
            d_max,
        });
    }

    // depth_nonlinear_representation_num_minus1 + per-segment
    // depth_nonlinear_representation_model[i], iff
    // depth_representation_type == 3.
    let (depth_nonlinear_representation_num_minus1, depth_nonlinear_representation_model) =
        if depth_representation_type == 3 {
            let num_minus1 = r.ue()?;
            if num_minus1 > 62 {
                return Err(SeiError::DepthRepresentationInfoNonlinearNumOutOfRange(
                    num_minus1,
                ));
            }
            // The spec loop runs i = 1..=depth_nonlinear_representation_num_minus1 + 1,
            // so the signalled count is num_minus1 + 1 (the i=0 and
            // i=num_minus1+2 sentinels are pre-defined to 0 by the
            // spec — they're not in the bitstream).
            let count = (num_minus1 as usize) + 1;
            let mut model = Vec::with_capacity(count);
            for i in 0..count {
                let v = r.ue()?;
                if v > 65535 {
                    return Err(SeiError::DepthRepresentationInfoNonlinearModelOutOfRange {
                        i,
                        got: v,
                    });
                }
                model.push(v as u16);
            }
            (Some(num_minus1 as u8), model)
        } else {
            (None, Vec::new())
        };

    Ok(DepthRepresentationInfo {
        all_views_equal_flag,
        z_near_flag,
        z_far_flag,
        z_axis_equal_flag,
        common_z_axis_reference_view,
        d_min_flag,
        d_max_flag,
        depth_representation_type,
        views,
        depth_nonlinear_representation_num_minus1,
        depth_nonlinear_representation_model,
    })
}

/// Annex I §I.13.1.1 / §I.13.2.1 — `constrained_depth_parameter_set_identifier`
/// (payload type 54, Annex I 3D-AVC depth coding).
///
/// When present, this SEI message is associated with an IDR access unit
/// and indicates that the `depth_parameter_set_id` and `dps_id` values
/// in the coded video sequence are constrained to the windowed range
/// described by the (`max_dps_id`, `max_dps_id_diff`) pair. Decoders
/// use this signal to conclude losses of depth parameter set NAL units
/// (NOTE 1 in §I.13.2.1) and to maintain the running `MaxUsedDpsId` /
/// `UsedDpsIdSet` state per slice.
///
/// Syntax — §I.13.1.1:
///
/// ```text
/// constrained_depth_parameter_set_identifier( payloadSize ) {
///     max_dps_id          ue(v)
///     max_dps_id_diff     ue(v)
/// }
/// ```
///
/// Constraints — §I.13.2.1:
///
/// * `max_dps_id` plus 1 specifies the maximum allowed
///   `depth_range_parameter_set_id` value. Per §7.4.2.16, the
///   `depth_parameter_set_id` field itself is in the range
///   `1..=63` (i.e. 0..=63 storage with 0 reserved for the
///   active-SPS-bound default), so the maximum meaningful value of
///   `max_dps_id` is 62 (encoding `max_dps_id + 1 ≤ 63`). Values
///   greater than 62 are rejected before storage.
/// * `max_dps_id_diff * 2` shall be less than `max_dps_id`. This is
///   the normative window-width constraint for the §I.13.2.1 sliding
///   used-DPS-id range computation; a value that violates it would
///   make the `prevMinUsedDpsId` / `minUsedDpsId` distance walk
///   around §I.13.2.1 eq. (I-86) ambiguous (the window would overlap
///   itself), so the parser rejects it before storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstrainedDepthParameterSetIdentifier {
    /// `max_dps_id` plus 1 specifies the maximum allowed
    /// `depth_range_parameter_set_id` value. Raw ue(v) value, bounded
    /// 0..=62 per the constraint above.
    pub max_dps_id: u8,
    /// `max_dps_id_diff` specifies the value range of
    /// `depth_range_parameter_set_id` values marked as "used". Raw
    /// ue(v) value, bounded 0..=30 (since `max_dps_id_diff * 2 <
    /// max_dps_id` and `max_dps_id ≤ 62`).
    pub max_dps_id_diff: u8,
}

/// Parse a §I.13.1.1 `constrained_depth_parameter_set_identifier()`
/// payload.
///
/// Enforces both §I.13.2.1 normative constraints before storage:
/// * `max_dps_id ≤ 62` — derived from the `depth_parameter_set_id`
///   range `1..=63` (§7.4.2.16).
/// * `max_dps_id_diff * 2 < max_dps_id` — the §I.13.2.1 sliding-window
///   integrity constraint for the §I.13.2.1 eq. (I-86) used-DPS-id
///   computation.
pub fn parse_constrained_depth_parameter_set_identifier(
    payload: &[u8],
) -> Result<ConstrainedDepthParameterSetIdentifier, SeiError> {
    let mut r = BitReader::new(payload);

    // §I.13.1.1 — max_dps_id ue(v).
    let max_dps_id = r.ue()?;
    if max_dps_id > 62 {
        return Err(SeiError::ConstrainedDepthParameterSetIdentifierMaxDpsIdOutOfRange(max_dps_id));
    }

    // §I.13.1.1 — max_dps_id_diff ue(v).
    let max_dps_id_diff = r.ue()?;
    // §I.13.2.1: `max_dps_id_diff * 2 < max_dps_id`. Use u64 widening
    // so the multiplication never overflows even at the ue(v) ceiling.
    if (max_dps_id_diff as u64) * 2 >= max_dps_id as u64 {
        return Err(
            SeiError::ConstrainedDepthParameterSetIdentifierDiffViolatesBound {
                max_dps_id_diff,
                max_dps_id,
            },
        );
    }

    Ok(ConstrainedDepthParameterSetIdentifier {
        max_dps_id: max_dps_id as u8,
        max_dps_id_diff: max_dps_id_diff as u8,
    })
}

// -------------------------------------------------------------------------
// Annex H §H.13.1.6 / §H.13.2.6 — alternative_depth_info (SEI payload 181)
// -------------------------------------------------------------------------

/// §H.13.2.6 — global view and depth (GVD) per-camera parameter block,
/// present in an [`AlternativeDepthInfo`] when `depth_type == 0`.
///
/// The `i`-index of every per-camera vector runs `0..=num_constituent_
/// views_gvd_minus1 + 1` (that is, `num_constituent_views_gvd_minus1 + 2`
/// cameras: i == 0 is the base texture view, i > 0 the constituent
/// views packed per Table H-4). Each optional sub-block is gated by its
/// corresponding presence flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlternativeDepthGvd {
    /// `num_constituent_views_gvd_minus1` (ue(v), 0..=3) — the number
    /// of constituent texture pictures packed into each non-base view
    /// texture component, minus one.
    pub num_constituent_views_gvd_minus1: u8,
    /// `depth_present_gvd_flag` (u(1)).
    pub depth_present_gvd_flag: bool,
    /// `z_gvd_flag` (u(1)) — gates the per-camera near/far depth block.
    pub z_gvd_flag: bool,
    /// `intrinsic_param_gvd_flag` (u(1)) — gates the focal-length +
    /// principal-point block.
    pub intrinsic_param_gvd_flag: bool,
    /// `rotation_gvd_flag` (u(1)) — gates the 3×3 rotation block. When
    /// `false` a unit rotation matrix is inferred per §H.13.2.6.
    pub rotation_gvd_flag: bool,
    /// `translation_gvd_flag` (u(1)) — gates the horizontal translation
    /// block.
    pub translation_gvd_flag: bool,
    /// Per-camera near/far depth values. Present (length =
    /// `num_constituent_views_gvd_minus1 + 2`) iff `z_gvd_flag`; empty
    /// otherwise.
    pub z_values: Vec<AlternativeDepthZ>,
    /// `prec_gvd_focal_length` (ue(v), 0..=31). `None` iff
    /// `!intrinsic_param_gvd_flag`.
    pub prec_gvd_focal_length: Option<u8>,
    /// `prec_gvd_principal_point` (ue(v), 0..=31). `None` iff
    /// `!intrinsic_param_gvd_flag`.
    pub prec_gvd_principal_point: Option<u8>,
    /// `prec_gvd_rotation_param` (ue(v), 0..=31). `None` iff
    /// `!rotation_gvd_flag`.
    pub prec_gvd_rotation_param: Option<u8>,
    /// `prec_gvd_translation_param` (ue(v), 0..=31). `None` iff
    /// `!translation_gvd_flag`.
    pub prec_gvd_translation_param: Option<u8>,
    /// Per-camera intrinsic + extrinsic float entries, one per camera
    /// (`num_constituent_views_gvd_minus1 + 2`). Each entry's optional
    /// members mirror the per-block presence flags above.
    pub cameras: Vec<AlternativeDepthCamera>,
}

/// §H.13.2.6 — per-camera nearest/farthest depth pair. Each value uses
/// the §H.13.2.3 depth float encoding (sign u(1), exponent u(7) with
/// reserved value 127, explicit mantissa length `man_len_minus1 + 1`),
/// so [`DepthFloatComponent`] is reused verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlternativeDepthZ {
    /// `zNear[i]` per eq. H-1 (`man_gvd_z_near` mantissa width is
    /// `man_len_gvd_z_near_minus1 + 1`).
    pub z_near: DepthFloatComponent,
    /// `zFar[i]` per eq. H-1.
    pub z_far: DepthFloatComponent,
}

/// §H.13.2.6 — per-camera intrinsic + extrinsic float parameters. Each
/// scalar uses the §G.13.2.5-style float encoding (sign u(1), exponent
/// u(6) with reserved value 63, prec-derived mantissa width per eq.
/// in §H.13.2.6), so [`FloatComponent`] is reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlternativeDepthCamera {
    /// `(focalLengthX, focalLengthY, principalPointX, principalPointY)`
    /// — `Some` iff `intrinsic_param_gvd_flag`.
    pub intrinsic: Option<AlternativeDepthIntrinsic>,
    /// 3×3 rotation matrix `r[j][k]` (0-indexed) — `Some` iff
    /// `rotation_gvd_flag`.
    pub rotation: Option<[[FloatComponent; 3]; 3]>,
    /// Horizontal translation `tX` — `Some` iff `translation_gvd_flag`.
    pub translation_x: Option<FloatComponent>,
}

/// §H.13.2.6 — per-camera intrinsic float quadruple.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlternativeDepthIntrinsic {
    pub focal_length_x: FloatComponent,
    pub focal_length_y: FloatComponent,
    pub principal_point_x: FloatComponent,
    pub principal_point_y: FloatComponent,
}

/// §H.13.2.6 — alternative depth information SEI message (payload type
/// 181, Annex H). Indicates that one output view's view components are
/// a spatial packing of multiple distinct constituent pictures (global
/// view + depth, GVD), so the view is not suitable for direct display.
///
/// `depth_type` selects the body: only `depth_type == 0` carries the
/// GVD parameter block (`gvd`). Other values are reserved; per
/// §H.13.2.6 "Decoders shall ignore alternative depth information SEI
/// messages in which such other values are present", so for a non-zero
/// type we record the type and leave `gvd` as `None` (graceful ignore,
/// not an error).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlternativeDepthInfo {
    /// `depth_type` (ue(v)). Conforming bitstreams use 0.
    pub depth_type: u32,
    /// GVD parameter block, present iff `depth_type == 0`.
    pub gvd: Option<AlternativeDepthGvd>,
}

/// Helper: read a `prec_gvd_*` ue(v) and enforce the §H.13.2.6 `0..=31`
/// range bound, surfacing an alternative-depth-specific error.
fn read_alt_depth_prec(r: &mut BitReader<'_>, field: &'static str) -> Result<u8, SeiError> {
    let v = r.ue()?;
    if v > 31 {
        return Err(SeiError::AlternativeDepthInfoPrecOutOfRange { field, got: v });
    }
    Ok(v as u8)
}

/// Parse an Annex H §H.13.1.6 `alternative_depth_info()` payload.
///
/// Enforces:
///
/// * `num_constituent_views_gvd_minus1 ≤ 3` (§H.13.2.6).
/// * each `prec_gvd_*` ∈ `0..=31` (§H.13.2.6).
///
/// Float components are stored verbatim:
///
/// * near/far depth pairs use [`DepthFloatComponent`] (exponent u(7),
///   reserved 127, explicit mantissa length).
/// * intrinsic / extrinsic scalars use [`FloatComponent`] (exponent
///   u(6), reserved 63, prec-derived mantissa width via
///   [`mantissa_width_g1325`], shared with §G.13.2.5).
///
/// Non-zero `depth_type` values are reserved; the body is not parsed
/// (graceful ignore per §H.13.2.6).
pub fn parse_alternative_depth_info(payload: &[u8]) -> Result<AlternativeDepthInfo, SeiError> {
    let mut r = BitReader::new(payload);

    // §H.13.1.6 — depth_type ue(v).
    let depth_type = r.ue()?;
    if depth_type != 0 {
        // Reserved value — decoders shall ignore the message body.
        return Ok(AlternativeDepthInfo {
            depth_type,
            gvd: None,
        });
    }

    // §H.13.1.6 — num_constituent_views_gvd_minus1 ue(v), 0..=3.
    let num_constituent_views_gvd_minus1 = r.ue()?;
    if num_constituent_views_gvd_minus1 > 3 {
        return Err(SeiError::AlternativeDepthInfoNumConstituentViewsOutOfRange(
            num_constituent_views_gvd_minus1,
        ));
    }
    // i runs 0..=num_constituent_views_gvd_minus1 + 1 → that many + 2.
    let cam_count = (num_constituent_views_gvd_minus1 as usize) + 2;

    // §H.13.1.6 — five u(1) presence flags.
    let depth_present_gvd_flag = r.u(1)? == 1;
    let z_gvd_flag = r.u(1)? == 1;
    let intrinsic_param_gvd_flag = r.u(1)? == 1;
    let rotation_gvd_flag = r.u(1)? == 1;
    let translation_gvd_flag = r.u(1)? == 1;

    // §H.13.1.6 — when z_gvd_flag, a per-camera near/far depth pair.
    let z_values = if z_gvd_flag {
        let mut v = Vec::with_capacity(cam_count);
        for _ in 0..cam_count {
            // §H.13.1.6: sign u(1), exp u(7), man_len_minus1 u(5),
            // man u(man_len_minus1 + 1) — exactly the §H.13.2.3
            // depth-float layout, so reuse read_depth_float_component.
            let z_near = read_depth_float_component(&mut r)?;
            let z_far = read_depth_float_component(&mut r)?;
            v.push(AlternativeDepthZ { z_near, z_far });
        }
        v
    } else {
        Vec::new()
    };

    // §H.13.1.6 — precision ue(v) words, each gated by its block flag.
    let prec_gvd_focal_length = if intrinsic_param_gvd_flag {
        Some(read_alt_depth_prec(&mut r, "prec_gvd_focal_length")?)
    } else {
        None
    };
    let prec_gvd_principal_point = if intrinsic_param_gvd_flag {
        Some(read_alt_depth_prec(&mut r, "prec_gvd_principal_point")?)
    } else {
        None
    };
    let prec_gvd_rotation_param = if rotation_gvd_flag {
        Some(read_alt_depth_prec(&mut r, "prec_gvd_rotation_param")?)
    } else {
        None
    };
    let prec_gvd_translation_param = if translation_gvd_flag {
        Some(read_alt_depth_prec(&mut r, "prec_gvd_translation_param")?)
    } else {
        None
    };

    // §H.13.1.6 — per-camera intrinsic / rotation / translation floats.
    let mut cameras: Vec<AlternativeDepthCamera> = Vec::with_capacity(cam_count);
    for _ in 0..cam_count {
        let intrinsic = if intrinsic_param_gvd_flag {
            let prec_fl = prec_gvd_focal_length.unwrap();
            let prec_pp = prec_gvd_principal_point.unwrap();
            let focal_length_x = read_float_component(&mut r, prec_fl)?;
            let focal_length_y = read_float_component(&mut r, prec_fl)?;
            let principal_point_x = read_float_component(&mut r, prec_pp)?;
            let principal_point_y = read_float_component(&mut r, prec_pp)?;
            Some(AlternativeDepthIntrinsic {
                focal_length_x,
                focal_length_y,
                principal_point_x,
                principal_point_y,
            })
        } else {
            None
        };

        let rotation = if rotation_gvd_flag {
            let prec_rot = prec_gvd_rotation_param.unwrap();
            let zero = FloatComponent {
                sign: false,
                exponent: 0,
                mantissa: 0,
                mantissa_width: 0,
            };
            // §H.13.1.6: for( j = 0; j < 3; j++ ) for( k = 0; k < 3; k++ ).
            let mut mat: [[FloatComponent; 3]; 3] = [[zero; 3]; 3];
            for row in mat.iter_mut() {
                for cell in row.iter_mut() {
                    *cell = read_float_component(&mut r, prec_rot)?;
                }
            }
            Some(mat)
        } else {
            None
        };

        let translation_x = if translation_gvd_flag {
            let prec_tr = prec_gvd_translation_param.unwrap();
            Some(read_float_component(&mut r, prec_tr)?)
        } else {
            None
        };

        cameras.push(AlternativeDepthCamera {
            intrinsic,
            rotation,
            translation_x,
        });
    }

    Ok(AlternativeDepthInfo {
        depth_type,
        gvd: Some(AlternativeDepthGvd {
            num_constituent_views_gvd_minus1: num_constituent_views_gvd_minus1 as u8,
            depth_present_gvd_flag,
            z_gvd_flag,
            intrinsic_param_gvd_flag,
            rotation_gvd_flag,
            translation_gvd_flag,
            z_values,
            prec_gvd_focal_length,
            prec_gvd_principal_point,
            prec_gvd_rotation_param,
            prec_gvd_translation_param,
            cameras,
        }),
    })
}

/// Annex H §H.13.1.7.1 — `depth_grid_position()` sub-structure used by
/// the §H.13.1.7 `depth_sampling_info` SEI message.
///
/// Carries the raw bitstream values for the horizontal + vertical
/// top-left-sample location of a depth view's sampling grid relative
/// to the same-`view_id` texture view. The §H.13.2.7.1 reconstruction
/// formulas are:
///
/// * horizontal location = `(1 − 2 * depth_grid_pos_x_sign_flag) *
///   (depth_grid_pos_x_fp ÷ 2^depth_grid_pos_x_dp)`
/// * vertical   location = `(1 − 2 * depth_grid_pos_y_sign_flag) *
///   (depth_grid_pos_y_fp ÷ 2^depth_grid_pos_y_dp)`
///
/// All six fields are u(n) reads so their range follows directly from
/// the bit width:
///
/// * `depth_grid_pos_{x,y}_fp` — u(20), 0..=1_048_575
/// * `depth_grid_pos_{x,y}_dp` — u(4), 0..=15
/// * `depth_grid_pos_{x,y}_sign_flag` — u(1), 0..=1
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthGridPosition {
    /// `depth_grid_pos_x_fp` — u(20). Raw fixed-point magnitude of
    /// the horizontal grid position.
    pub pos_x_fp: u32,
    /// `depth_grid_pos_x_dp` — u(4). Number of fractional bits in
    /// `pos_x_fp` (the divisor is `2^pos_x_dp`).
    pub pos_x_dp: u8,
    /// `depth_grid_pos_x_sign_flag` — u(1). `false` for positive,
    /// `true` for negative.
    pub pos_x_sign: bool,
    /// `depth_grid_pos_y_fp` — u(20). Raw fixed-point magnitude of
    /// the vertical grid position.
    pub pos_y_fp: u32,
    /// `depth_grid_pos_y_dp` — u(4). Number of fractional bits in
    /// `pos_y_fp` (the divisor is `2^pos_y_dp`).
    pub pos_y_dp: u8,
    /// `depth_grid_pos_y_sign_flag` — u(1). `false` for positive,
    /// `true` for negative.
    pub pos_y_sign: bool,
}

impl DepthGridPosition {
    /// Reconstruct the §H.13.2.7.1 horizontal position scalar
    /// `(1 − 2 * pos_x_sign_flag) * (pos_x_fp ÷ 2^pos_x_dp)`.
    pub fn x_to_f64(&self) -> f64 {
        let mag = (self.pos_x_fp as f64) / (1u64 << self.pos_x_dp) as f64;
        if self.pos_x_sign {
            -mag
        } else {
            mag
        }
    }
    /// Reconstruct the §H.13.2.7.1 vertical position scalar
    /// `(1 − 2 * pos_y_sign_flag) * (pos_y_fp ÷ 2^pos_y_dp)`.
    pub fn y_to_f64(&self) -> f64 {
        let mag = (self.pos_y_fp as f64) / (1u64 << self.pos_y_dp) as f64;
        if self.pos_y_sign {
            -mag
        } else {
            mag
        }
    }
}

fn read_depth_grid_position(r: &mut BitReader) -> Result<DepthGridPosition, SeiError> {
    // §H.13.1.7.1 — depth_grid_pos_x_fp u(20); depth_grid_pos_x_dp u(4);
    // depth_grid_pos_x_sign_flag u(1); same triple repeated for y.
    let pos_x_fp = r.u(20)?;
    let pos_x_dp = r.u(4)? as u8;
    let pos_x_sign = r.u(1)? == 1;
    let pos_y_fp = r.u(20)?;
    let pos_y_dp = r.u(4)? as u8;
    let pos_y_sign = r.u(1)? == 1;
    Ok(DepthGridPosition {
        pos_x_fp,
        pos_x_dp,
        pos_x_sign,
        pos_y_fp,
        pos_y_dp,
        pos_y_sign,
    })
}

/// Per-view depth grid position entry used by §H.13.1.7
/// `depth_sampling_info` when `per_view_depth_grid_pos_flag == 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthSamplingViewEntry {
    /// `depth_grid_view_id[i]` — ue(v). The `view_id` value for which
    /// the following `depth_grid_position()` applies. Bounded 0..=1023
    /// per the Annex G/H view_id range.
    pub depth_grid_view_id: u16,
    /// `depth_grid_position()` for this `view_id`.
    pub position: DepthGridPosition,
}

/// Annex H §H.13.1.7 / §H.13.2.7 — `depth_sampling_info` (payload
/// type 53).
///
/// Specifies the depth-sample size relative to the luma-texture-sample
/// size and the depth-sampling grid position for one or more depth
/// view components relative to the texture view component with the
/// same `view_id` value. When present, this SEI message shall be
/// associated with an IDR access unit; the semantics apply to the
/// current coded video sequence.
///
/// Syntax — §H.13.1.7:
///
/// ```text
/// depth_sampling_info( payloadSize ) {
///     dttsr_x_mul                                u(16)
///     dttsr_x_dp                                 u(4)
///     dttsr_y_mul                                u(16)
///     dttsr_y_dp                                 u(4)
///     per_view_depth_grid_pos_flag               u(1)
///     if( per_view_depth_grid_pos_flag ) {
///         num_video_plus_depth_views_minus1      ue(v)
///         for( i = 0; i <= num_video_plus_depth_views_minus1; i++ ) {
///             depth_grid_view_id[i]              ue(v)
///             depth_grid_position()
///         }
///     } else
///         depth_grid_position()
/// }
/// ```
///
/// Sample-size semantics (§H.13.2.7):
///
/// * The width of a depth sample relative to the width of a luma
///   texture sample is approximately
///   `dttsr_x_mul ÷ 2^dttsr_x_dp`. The value 0 for `dttsr_x_mul` is
///   reserved (rejected with
///   [`SeiError::DepthSamplingInfoDttsrMulReserved`]).
/// * The height of a depth sample relative to the height of a luma
///   texture sample is approximately
///   `dttsr_y_mul ÷ 2^dttsr_y_dp`. The value 0 for `dttsr_y_mul` is
///   reserved (rejected with
///   [`SeiError::DepthSamplingInfoDttsrMulReserved`]).
///
/// Grid-position semantics (§H.13.2.7):
///
/// * `per_view_depth_grid_pos_flag == 0` — a single
///   `depth_grid_position()` describes the same grid for all depth
///   views in the access unit (stored as a single-entry
///   `views` vector with a sentinel `depth_grid_view_id = 0`).
/// * `per_view_depth_grid_pos_flag == 1` — one
///   `depth_grid_position()` per declared `view_id`. The Annex G/H
///   absolute upper bound on `num_views_minus1` (1023, derived from
///   `num_views_minus1 ≤ 1023` in the Annex G/H SPS extensions) is
///   used as the anti-OOM cap before allocating the per-view vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepthSamplingInfo {
    /// `dttsr_x_mul` — u(16), the horizontal depth/texture sample
    /// ratio numerator. The value 0 is reserved.
    pub dttsr_x_mul: u16,
    /// `dttsr_x_dp` — u(4), the horizontal ratio's fractional bit
    /// width (divisor is `2^dttsr_x_dp`).
    pub dttsr_x_dp: u8,
    /// `dttsr_y_mul` — u(16), the vertical depth/texture sample ratio
    /// numerator. The value 0 is reserved.
    pub dttsr_y_mul: u16,
    /// `dttsr_y_dp` — u(4), the vertical ratio's fractional bit width
    /// (divisor is `2^dttsr_y_dp`).
    pub dttsr_y_dp: u8,
    /// `per_view_depth_grid_pos_flag` — u(1). When `true`, `views`
    /// carries one entry per signalled `view_id`. When `false`,
    /// `views` carries a single shared entry.
    pub per_view_depth_grid_pos_flag: bool,
    /// `num_video_plus_depth_views_minus1` — only present when
    /// `per_view_depth_grid_pos_flag == 1`. Plus one gives the count
    /// of per-view entries that follow. Bounded 0..=1023 per the
    /// Annex G/H absolute `num_views_minus1 ≤ 1023`.
    pub num_video_plus_depth_views_minus1: Option<u16>,
    /// Either a single shared `depth_grid_position()` entry
    /// (`per_view_depth_grid_pos_flag == 0`, length 1, sentinel
    /// `depth_grid_view_id = 0`) or one entry per signalled `view_id`.
    pub views: Vec<DepthSamplingViewEntry>,
}

impl DepthSamplingInfo {
    /// Reconstruct the §H.13.2.7 horizontal depth/texture sample-size
    /// ratio `dttsr_x_mul ÷ 2^dttsr_x_dp`. Always > 0 since the parser
    /// rejects `dttsr_x_mul == 0` (Annex H §H.13.2.7 reserved value).
    pub fn dttsr_x_to_f64(&self) -> f64 {
        (self.dttsr_x_mul as f64) / (1u64 << self.dttsr_x_dp) as f64
    }
    /// Reconstruct the §H.13.2.7 vertical depth/texture sample-size
    /// ratio `dttsr_y_mul ÷ 2^dttsr_y_dp`. Always > 0 since the parser
    /// rejects `dttsr_y_mul == 0` (Annex H §H.13.2.7 reserved value).
    pub fn dttsr_y_to_f64(&self) -> f64 {
        (self.dttsr_y_mul as f64) / (1u64 << self.dttsr_y_dp) as f64
    }
}

/// Parse a §H.13.1.7 `depth_sampling_info()` payload (payload type 53).
///
/// Enforces the §H.13.2.7 normative range bounds before any
/// allocation:
/// * `dttsr_x_mul != 0` and `dttsr_y_mul != 0` (the value 0 is
///   reserved per §H.13.2.7).
/// * `num_video_plus_depth_views_minus1 ≤ 1023` (Annex G/H absolute
///   `num_views_minus1` cap). Pre-allocation gate against the
///   ue(v)-driven OOM lever closed in round 177 for §D.1.20 and
///   round 200 for §G.13.2.8.
/// * `depth_grid_view_id[i] ≤ 1023` (Annex G/H view_id range).
pub fn parse_depth_sampling_info(payload: &[u8]) -> Result<DepthSamplingInfo, SeiError> {
    let mut r = BitReader::new(payload);

    // §H.13.1.7 — dttsr_x_mul u(16). §H.13.2.7: 0 is reserved.
    let dttsr_x_mul = r.u(16)? as u16;
    if dttsr_x_mul == 0 {
        return Err(SeiError::DepthSamplingInfoDttsrMulReserved { axis: "x" });
    }
    // §H.13.1.7 — dttsr_x_dp u(4).
    let dttsr_x_dp = r.u(4)? as u8;
    // §H.13.1.7 — dttsr_y_mul u(16). §H.13.2.7: 0 is reserved.
    let dttsr_y_mul = r.u(16)? as u16;
    if dttsr_y_mul == 0 {
        return Err(SeiError::DepthSamplingInfoDttsrMulReserved { axis: "y" });
    }
    // §H.13.1.7 — dttsr_y_dp u(4).
    let dttsr_y_dp = r.u(4)? as u8;
    // §H.13.1.7 — per_view_depth_grid_pos_flag u(1).
    let per_view_depth_grid_pos_flag = r.u(1)? == 1;

    let (num_video_plus_depth_views_minus1, views) = if per_view_depth_grid_pos_flag {
        // §H.13.1.7 — num_video_plus_depth_views_minus1 ue(v).
        // §H.13.2.7: views_minus1 + 1 is the number of declared
        // view_ids. Anti-OOM cap derived from the Annex G/H absolute
        // `num_views_minus1 ≤ 1023`.
        let num_minus1 = r.ue()?;
        if num_minus1 > 1023 {
            return Err(SeiError::DepthSamplingInfoNumViewsOutOfRange(num_minus1));
        }
        let count = (num_minus1 as usize) + 1;
        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            // §H.13.1.7 — depth_grid_view_id[i] ue(v). §H.13.2.7
            // refers to the same view_id 0..=1023 range used across
            // Annex G/H view-signalling messages; reject anything
            // beyond.
            let view_id = r.ue()?;
            if view_id > 1023 {
                return Err(SeiError::DepthSamplingInfoViewIdOutOfRange { i, got: view_id });
            }
            let position = read_depth_grid_position(&mut r)?;
            entries.push(DepthSamplingViewEntry {
                depth_grid_view_id: view_id as u16,
                position,
            });
        }
        (Some(num_minus1 as u16), entries)
    } else {
        // §H.13.1.7 else branch — single shared depth_grid_position().
        // We surface it as a single-entry views vector with a
        // sentinel depth_grid_view_id = 0 so downstream consumers
        // have a uniform shape.
        let position = read_depth_grid_position(&mut r)?;
        (
            None,
            vec![DepthSamplingViewEntry {
                depth_grid_view_id: 0,
                position,
            }],
        )
    };

    Ok(DepthSamplingInfo {
        dttsr_x_mul,
        dttsr_x_dp,
        dttsr_y_mul,
        dttsr_y_dp,
        per_view_depth_grid_pos_flag,
        num_video_plus_depth_views_minus1,
        views,
    })
}

/// Annex H §H.13.1.5.1 / §H.13.2.5.1 — `depth_timing_offset()`
/// sub-structure used by the §H.13.1.5 `depth_timing` SEI message.
///
/// Specifies the acquisition offset of the respective depth view
/// component(s) relative to the DPB output time of the access unit
/// containing them, equal to
/// `depth_disp_delay_offset_fp ÷ 2^depth_disp_delay_offset_dp` in
/// units of clock ticks as specified in Annex C (§H.13.2.5.1).
///
/// Field ranges follow directly from the bit widths:
///
/// * `offset_len_minus1` — u(5), 0..=31; plus 1 gives the bit width
///   of `depth_disp_delay_offset_fp` (§H.13.2.5.1: "The length of
///   depth_disp_delay_offset_fp syntax element is equal to
///   offset_len_minus1 + 1"), so the fp field spans 1..=32 bits.
/// * `depth_disp_delay_offset_fp` — u(offset_len_minus1 + 1),
///   0..=`2^(offset_len_minus1 + 1) − 1` (up to the full u32 range).
/// * `depth_disp_delay_offset_dp` — u(6), 0..=63.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthTimingOffset {
    /// `offset_len_minus1` — u(5). Plus 1 gives the bit width of
    /// `depth_disp_delay_offset_fp`.
    pub offset_len_minus1: u8,
    /// `depth_disp_delay_offset_fp` — u(offset_len_minus1 + 1). The
    /// fixed-point numerator of the acquisition offset.
    pub depth_disp_delay_offset_fp: u32,
    /// `depth_disp_delay_offset_dp` — u(6). The number of fractional
    /// bits in `depth_disp_delay_offset_fp` (the divisor is
    /// `2^depth_disp_delay_offset_dp`).
    pub depth_disp_delay_offset_dp: u8,
}

impl DepthTimingOffset {
    /// Reconstruct the §H.13.2.5.1 acquisition-offset scalar
    /// `depth_disp_delay_offset_fp ÷ 2^depth_disp_delay_offset_dp`,
    /// in units of clock ticks as specified in Annex C. Always
    /// non-negative — the §H.13.1.5.1 syntax has no sign field.
    pub fn offset_clock_ticks(&self) -> f64 {
        f64::from(self.depth_disp_delay_offset_fp)
            / 2f64.powi(i32::from(self.depth_disp_delay_offset_dp))
    }
}

fn read_depth_timing_offset(r: &mut BitReader) -> Result<DepthTimingOffset, SeiError> {
    // §H.13.1.5.1 — offset_len_minus1 u(5);
    // depth_disp_delay_offset_fp u(offset_len_minus1 + 1);
    // depth_disp_delay_offset_dp u(6).
    let offset_len_minus1 = r.u(5)? as u8;
    let depth_disp_delay_offset_fp = r.u(u32::from(offset_len_minus1) + 1)?;
    let depth_disp_delay_offset_dp = r.u(6)? as u8;
    Ok(DepthTimingOffset {
        offset_len_minus1,
        depth_disp_delay_offset_fp,
        depth_disp_delay_offset_dp,
    })
}

/// Annex H §H.13.1.5 / §H.13.2.5 — `depth_timing` (payload type 52).
///
/// Indicates the acquisition time of the depth view components of one
/// or more access units relative to the DPB output time of the same
/// access units. The message may be present in any access unit and
/// pertains until the end of the coded video sequence or until the
/// next depth timing SEI message, whichever is earlier in decoding
/// order (the access units it pertains to are the *target access unit
/// set*).
///
/// Syntax — §H.13.1.5:
///
/// ```text
/// depth_timing( payloadSize ) {
///     per_view_depth_timing_flag                 u(1)
///     if( per_view_depth_timing_flag )
///         for( i = 0; i < NumDepthViews; i++ )
///             depth_timing_offset( )
///     else
///         depth_timing_offset( )
/// }
/// ```
///
/// Semantics (§H.13.2.5):
///
/// * `per_view_depth_timing_flag == 0` — all the depth view
///   components within the target access unit set have the same
///   acquisition time offset; the single `depth_timing_offset()`
///   occurrence specifies it (stored as a single-entry `offsets`
///   vector).
/// * `per_view_depth_timing_flag == 1` — one `depth_timing_offset()`
///   per depth view, in ascending order of view order index values
///   for the depth views. The loop bound is the SPS-derived
///   `NumDepthViews` variable (§H.7.3.2.1.5), supplied through
///   [`SeiContext::num_depth_views`]; a message reaching this branch
///   under an unknown (0) value is rejected with
///   [`SeiError::DepthTimingNumDepthViewsUnknown`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepthTiming {
    /// `per_view_depth_timing_flag` — u(1). When `true`, `offsets`
    /// carries one entry per depth view in ascending view order index
    /// order. When `false`, `offsets` carries a single shared entry.
    pub per_view_depth_timing_flag: bool,
    /// Either a single shared `depth_timing_offset()` entry
    /// (`per_view_depth_timing_flag == 0`, length 1) or one entry per
    /// depth view (`per_view_depth_timing_flag == 1`, length
    /// `NumDepthViews`).
    pub offsets: Vec<DepthTimingOffset>,
}

/// Parse a §H.13.1.5 `depth_timing()` payload (payload type 52).
///
/// The per-view loop bound is NOT in the payload — §H.13.1.5 loops
/// over the `NumDepthViews` variable accumulated from
/// `depth_view_present_flag[i]` in the active subset SPS MVCD
/// extension (§H.7.3.2.1.5), supplied here through
/// [`SeiContext::num_depth_views`]. Two gates run before the
/// per-view allocation:
///
/// * `ctx.num_depth_views == 0` (unknown / no Annex H subset SPS
///   active) — rejected with
///   [`SeiError::DepthTimingNumDepthViewsUnknown`], mirroring the
///   §D.1.10 spare_pic treatment of an unknown `PicSizeInMapUnits`.
/// * `ctx.num_depth_views > 1024` — rejected with
///   [`SeiError::DepthTimingNumDepthViewsOutOfRange`]; the
///   §H.7.3.2.1.5 view loop runs `num_views_minus1 + 1 ≤ 1024` times
///   and increments `NumDepthViews` at most once per iteration, so
///   1024 is the absolute ceiling. Pre-allocation gate in the same
///   anti-OOM spirit as the round-177 / round-200 ue(v) caps (here
///   the count is caller-supplied rather than bitstream-driven, but
///   the `Vec::with_capacity` cost is bounded the same way).
pub fn parse_depth_timing(payload: &[u8], ctx: &SeiContext) -> Result<DepthTiming, SeiError> {
    let mut r = BitReader::new(payload);

    // §H.13.1.5 — per_view_depth_timing_flag u(1).
    let per_view_depth_timing_flag = r.u(1)? == 1;

    let offsets = if per_view_depth_timing_flag {
        // §H.13.1.5 — for( i = 0; i < NumDepthViews; i++ )
        //                 depth_timing_offset( )
        if ctx.num_depth_views == 0 {
            return Err(SeiError::DepthTimingNumDepthViewsUnknown);
        }
        if ctx.num_depth_views > 1024 {
            return Err(SeiError::DepthTimingNumDepthViewsOutOfRange(
                ctx.num_depth_views,
            ));
        }
        let count = ctx.num_depth_views as usize;
        let mut offsets = Vec::with_capacity(count);
        for _ in 0..count {
            offsets.push(read_depth_timing_offset(&mut r)?);
        }
        offsets
    } else {
        // §H.13.1.5 else branch — single shared depth_timing_offset().
        vec![read_depth_timing_offset(&mut r)?]
    };

    Ok(DepthTiming {
        per_view_depth_timing_flag,
        offsets,
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
    /// §D.2.30 — colour_remapping_info (payload type 142). Round 117.
    ColourRemappingInfo(ColourRemappingInfo),
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
    /// §G.13.2.4 — multiview_scene_info (payload type 39, Annex G). Round 200.
    MultiviewSceneInfo(MultiviewSceneInfo),
    /// §G.13.2.5 — multiview_acquisition_info (payload type 40, Annex G). Round 213.
    MultiviewAcquisitionInfo(MultiviewAcquisitionInfo),
    /// §G.13.2.6 — non_required_view_component (payload type 41, Annex G). Round 207.
    NonRequiredViewComponent(NonRequiredViewComponent),
    /// §G.13.2.8 — operation_point_not_present (payload type 43, Annex G). Round 200.
    OperationPointNotPresent(OperationPointNotPresent),
    /// §G.13.2.9 — base_view_temporal_hrd (payload type 44, Annex G / MVC). Round 293.
    BaseViewTemporalHrd(BaseViewTemporalHrd),
    /// §G.13.2.10 — multiview_view_position (payload type 46). Round 183.
    MultiviewViewPosition(MultiviewViewPosition),
    /// §H.13.2.3 — depth_representation_info (payload type 50,
    /// Annex H). Round 231.
    DepthRepresentationInfo(DepthRepresentationInfo),
    /// §H.13.2.4 — three_dimensional_reference_displays_info
    /// (payload type 51, Annex H). Round 226.
    ThreeDimensionalReferenceDisplaysInfo(ThreeDimensionalReferenceDisplaysInfo),
    /// §H.13.2.5 — depth_timing (payload type 52, Annex H 3D-AVC).
    /// Round 278.
    DepthTiming(DepthTiming),
    /// §H.13.2.7 — depth_sampling_info (payload type 53, Annex H 3D-AVC).
    /// Round 247.
    DepthSamplingInfo(DepthSamplingInfo),
    /// §I.13.2.1 — constrained_depth_parameter_set_identifier (payload
    /// type 54, Annex I 3D-AVC depth coding). Round 237.
    ConstrainedDepthParameterSetIdentifier(ConstrainedDepthParameterSetIdentifier),
    /// §H.13.2.6 — alternative_depth_info (payload type 181, Annex H
    /// 3D-AVC). Round 318.
    AlternativeDepthInfo(AlternativeDepthInfo),
    /// §D.2.36 — sei_manifest (payload type 200). Round 120.
    SeiManifest(SeiManifest),
    /// §D.2.37 — sei_prefix_indication (payload type 201). Round 120.
    SeiPrefixIndication(SeiPrefixIndication),
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
        39 => Ok(SeiPayload::MultiviewSceneInfo(parse_multiview_scene_info(
            payload,
        )?)),
        40 => Ok(SeiPayload::MultiviewAcquisitionInfo(
            parse_multiview_acquisition_info(payload)?,
        )),
        41 => Ok(SeiPayload::NonRequiredViewComponent(
            parse_non_required_view_component(payload)?,
        )),
        43 => Ok(SeiPayload::OperationPointNotPresent(
            parse_operation_point_not_present(payload)?,
        )),
        44 => Ok(SeiPayload::BaseViewTemporalHrd(
            parse_base_view_temporal_hrd(payload)?,
        )),
        45 => Ok(SeiPayload::FramePackingArrangement(
            parse_frame_packing_arrangement(payload)?,
        )),
        46 => Ok(SeiPayload::MultiviewViewPosition(
            parse_multiview_view_position(payload)?,
        )),
        47 => Ok(SeiPayload::DisplayOrientation(parse_display_orientation(
            payload,
        )?)),
        50 => Ok(SeiPayload::DepthRepresentationInfo(
            parse_depth_representation_info(payload)?,
        )),
        51 => Ok(SeiPayload::ThreeDimensionalReferenceDisplaysInfo(
            parse_three_dimensional_reference_displays_info(payload)?,
        )),
        52 => Ok(SeiPayload::DepthTiming(parse_depth_timing(payload, ctx)?)),
        53 => Ok(SeiPayload::DepthSamplingInfo(parse_depth_sampling_info(
            payload,
        )?)),
        54 => Ok(SeiPayload::ConstrainedDepthParameterSetIdentifier(
            parse_constrained_depth_parameter_set_identifier(payload)?,
        )),
        181 => Ok(SeiPayload::AlternativeDepthInfo(
            parse_alternative_depth_info(payload)?,
        )),
        137 => Ok(SeiPayload::MasteringDisplay(parse_mastering_display(
            payload,
        )?)),
        142 => Ok(SeiPayload::ColourRemappingInfo(
            parse_colour_remapping_info(payload)?,
        )),
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
        200 => Ok(SeiPayload::SeiManifest(parse_sei_manifest(payload)?)),
        201 => Ok(SeiPayload::SeiPrefixIndication(
            parse_sei_prefix_indication(payload)?,
        )),
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

    // §D.2.25 — the implicit-end-point accessors only fire for
    // tone_map_model_id == 3. For the linear (model_id 0) body
    // exercised by `tone_mapping_info_linear`, both
    // `piecewise_default_end_points` and `piecewise_total_pivot_count`
    // return None.
    #[test]
    fn tone_mapping_piecewise_accessors_none_for_linear() {
        let fields: [(u64, u32); 8] = [
            (1, 1),            // ue tone_map_id=0
            (0, 1),            // cancel_flag=0
            (1, 1),            // ue repetition_period=0
            (10, 8),           // coded_data_bit_depth
            (8, 8),            // target_bit_depth
            (1, 1),            // ue model_id=0 (linear)
            (0x0000_0100, 32), // min_value
            (0x0000_03FF, 32), // max_value
        ];
        let packed = pack_bits(&fields);
        let body = parse_tone_mapping_info(&packed).unwrap().body.unwrap();
        assert_eq!(body.model_id, 0);
        assert!(body.piecewise_default_end_points().is_none());
        assert!(body.piecewise_total_pivot_count().is_none());
    }

    // §D.2.25 — piecewise-linear (model_id 3) implicit default end
    // points. num_pivots = 2 interior pivots; coded_data_bit_depth =
    // 10, target_bit_depth = 8. The two implicit end points are
    // (0, 0) and (2^10 − 1, 2^8 − 1) = (1023, 255). The assembled
    // curve therefore has num_pivots + 2 = 4 points.
    //
    // coded_pivot_value width = ((10 + 7) >> 3) << 3 = 16 bits.
    // target_pivot_value width = ((8 + 7) >> 3) << 3 = 8 bits.
    #[test]
    fn tone_mapping_piecewise_default_end_points_and_count() {
        let fields: [(u64, u32); 11] = [
            (1, 1),       // ue tone_map_id=0
            (0, 1),       // cancel_flag=0
            (1, 1),       // ue repetition_period=0
            (10, 8),      // coded_data_bit_depth = 10
            (8, 8),       // target_bit_depth = 8
            (0b00100, 5), // ue model_id=3 (5-bit Exp-Golomb codeword)
            (2, 16),      // num_pivots = 2
            (300, 16),    // coded_pivot_value[0] (16-bit)
            (100, 8),     // target_pivot_value[0] (8-bit)
            (700, 16),    // coded_pivot_value[1] (16-bit)
            (200, 8),     // target_pivot_value[1] (8-bit)
        ];
        let packed = pack_bits(&fields);
        let body = parse_tone_mapping_info(&packed).unwrap().body.unwrap();
        assert_eq!(body.model_id, 3);
        match &body.model {
            ToneMappingModel::PiecewisePivots {
                num_pivots,
                coded_pivot_value,
                target_pivot_value,
            } => {
                assert_eq!(*num_pivots, 2);
                assert_eq!(coded_pivot_value, &vec![300, 700]);
                assert_eq!(target_pivot_value, &vec![100, 200]);
            }
            other => panic!("expected PiecewisePivots, got {:?}", other),
        }

        // Implicit end points: start (0, 0) and end (1023, 255).
        let (start, end) = body.piecewise_default_end_points().unwrap();
        assert_eq!(start, (0, 0));
        assert_eq!(end, (1023, 255));

        // Total = interior num_pivots (2) + 2 default end points = 4.
        assert_eq!(body.piecewise_total_pivot_count(), Some(4));
    }

    // §D.2.25 — the implicit end value tracks coded_data_bit_depth /
    // target_bit_depth independently. Here coded = 14 (max legal),
    // target = 16 (max legal): end = (2^14 − 1, 2^16 − 1) =
    // (16383, 65535) with zero interior pivots → total count 2.
    //
    // coded_pivot_value width = ((14 + 7) >> 3) << 3 = 16.
    // target_pivot_value width = ((16 + 7) >> 3) << 3 = 16.
    #[test]
    fn tone_mapping_piecewise_max_bit_depths_zero_pivots() {
        let fields = vec![
            (1u64, 1u32), // ue tone_map_id=0
            (0, 1),       // cancel_flag=0
            (1, 1),       // ue repetition_period=0
            (14, 8),      // coded_data_bit_depth = 14
            (16, 8),      // target_bit_depth = 16
            (0b00100, 5), // ue model_id = 3
            (0, 16),      // num_pivots = 0
        ];
        let packed = pack_bits(&fields);
        let body = parse_tone_mapping_info(&packed).unwrap().body.unwrap();
        assert_eq!(body.model_id, 3);
        let (start, end) = body.piecewise_default_end_points().unwrap();
        assert_eq!(start, (0, 0));
        assert_eq!(end, (16_383, 65_535));
        assert_eq!(body.piecewise_total_pivot_count(), Some(2));
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

    // §D.2.21 eq. D-14 / D-15 — typed bit-depth accessors on the
    // separate-colour-description body. The on-wire `_minus8` carriers in
    // the round-trip fixture above are both 2 (10-bit luma + chroma), so
    // the accessors must surface filmGrainBitDepth = 10.
    #[test]
    fn film_grain_bit_depth_typical_10bit() {
        let scd = FilmGrainSeparateColourDescription {
            bit_depth_luma_minus8: 2,
            bit_depth_chroma_minus8: 2,
            full_range_flag: false,
            colour_primaries: 1,
            transfer_characteristics: 13,
            matrix_coefficients: 1,
        };
        // eq. D-14: filmGrainBitDepth[0] = 2 + 8 = 10.
        assert_eq!(scd.bit_depth_luma(), 10);
        // eq. D-15: filmGrainBitDepth[1] = filmGrainBitDepth[2] = 2 + 8 = 10.
        assert_eq!(scd.bit_depth_chroma(), 10);
    }

    // §D.2.21 — minimum on-wire carrier (0 ⇒ 8-bit), the most common case.
    #[test]
    fn film_grain_bit_depth_minimum_8bit() {
        let scd = FilmGrainSeparateColourDescription {
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            full_range_flag: false,
            colour_primaries: 2,
            transfer_characteristics: 2,
            matrix_coefficients: 2,
        };
        assert_eq!(scd.bit_depth_luma(), 8);
        assert_eq!(scd.bit_depth_chroma(), 8);
    }

    // §D.2.21 — u(3) ceiling (7 ⇒ 15-bit), and an asymmetric luma/chroma
    // pair to prove the two accessors read independent carriers.
    #[test]
    fn film_grain_bit_depth_u3_ceiling_and_asymmetric() {
        let scd = FilmGrainSeparateColourDescription {
            bit_depth_luma_minus8: 7,
            bit_depth_chroma_minus8: 4,
            full_range_flag: true,
            colour_primaries: 1,
            transfer_characteristics: 1,
            matrix_coefficients: 1,
        };
        // u(3) max carrier 7 ⇒ 15.
        assert_eq!(scd.bit_depth_luma(), 15);
        // Distinct chroma carrier 4 ⇒ 12 (not coupled to luma).
        assert_eq!(scd.bit_depth_chroma(), 12);
    }

    // §D.2.21 — end-to-end through the parser: the round-trip fixture's
    // separate-colour body carries 2 / 2, and the accessors must fire on
    // the parsed struct (not just a hand-built one).
    #[test]
    fn film_grain_bit_depth_parse_round_trip() {
        let fields = [
            (0, 1),  // cancel_flag
            (2, 2),  // model_id
            (1, 1),  // separate_colour_description_present_flag
            (2, 3),  // bit_depth_luma_minus8 (= 10-bit luma)
            (2, 3),  // bit_depth_chroma_minus8 (= 10-bit chroma)
            (1, 1),  // full_range_flag
            (1, 8),  // colour_primaries
            (13, 8), // transfer_characteristics
            (1, 8),  // matrix_coefficients
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
            (1, 1),   // comp_model_value[0][0] = se(0)
            // c=1 body
            (0, 8),   // num_intensity_intervals_minus1
            (0, 3),   // num_model_values_minus1
            (0, 8),   // interval[0].lower_bound
            (255, 8), // interval[0].upper_bound
            (1, 1),   // comp_model_value[0][0] = se(0)
            (1, 1),   // repetition_period = ue(0) = 0
        ];
        let payload = pack_bits(&fields);
        let fg = parse_film_grain_characteristics(&payload).unwrap();
        let scd = fg
            .body
            .expect("body present")
            .separate_colour_description
            .expect("scd present");
        assert_eq!(scd.bit_depth_luma_minus8, 2);
        assert_eq!(scd.bit_depth_chroma_minus8, 2);
        assert_eq!(scd.bit_depth_luma(), 10);
        assert_eq!(scd.bit_depth_chroma(), 10);
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

    // §G.13.1.10 / §G.13.2.10 — multiview_view_position (payload 46).
    //
    // Two-view case (num_views_minus1 = 1):
    //   num_views_minus1 = 1 → ue codeword "010" (3 bits)
    //   view_position[0] = 0 → ue codeword "1" (1 bit)
    //   view_position[1] = 1 → ue codeword "010" (3 bits)
    //   multiview_view_position_extension_flag = 0 (1 bit)
    // Total: 8 bits → one byte.
    #[test]
    fn multiview_view_position_two_views_left_then_right() {
        let payload = pack_bits(&[
            (0b010, 3), // num_views_minus1 = 1
            (1, 1),     // view_position[0] = 0
            (0b010, 3), // view_position[1] = 1
            (0, 1),     // extension_flag = 0
        ]);
        let got = parse_multiview_view_position(&payload).unwrap();
        assert_eq!(got.view_positions, vec![0u16, 1u16]);
        assert!(!got.extension_flag);
    }

    #[test]
    fn multiview_view_position_single_view_zero_position() {
        // num_views_minus1 = 0 → ue codeword "1" (1 bit)
        // view_position[0] = 0 → ue "1" (1 bit)
        // extension_flag = 0 (1 bit)
        let payload = pack_bits(&[(1, 1), (1, 1), (0, 1)]);
        let got = parse_multiview_view_position(&payload).unwrap();
        assert_eq!(got.view_positions, vec![0u16]);
        assert!(!got.extension_flag);
    }

    #[test]
    fn multiview_view_position_preserves_extension_flag_when_set() {
        // Single-view case with extension_flag = 1. §G.13.2.10 says
        // conforming encoders write 0; the parser still surfaces the
        // bit so callers can audit a non-conforming stream.
        let payload = pack_bits(&[(1, 1), (1, 1), (1, 1)]);
        let got = parse_multiview_view_position(&payload).unwrap();
        assert_eq!(got.view_positions, vec![0u16]);
        assert!(got.extension_flag);
    }

    #[test]
    fn multiview_view_position_rejects_num_views_above_1023() {
        // num_views_minus1 = 1024 → ue codeword "00000000001 0000000001"
        // (10 leading zeros, leading 1, then 10-bit suffix = 0b00000000001).
        // §G.13.2.10 caps it at 1023. We don't bother walking the
        // suffix bits — the parser must reject before allocating.
        let mut bits = Vec::new();
        // ue(1024) = 11 bits prefix (10 zeros + a 1) + 10 bits suffix
        // = "00000000001 0000000001".
        for _ in 0..10 {
            bits.push((0, 1));
        }
        bits.push((1, 1));
        for _ in 0..9 {
            bits.push((0, 1));
        }
        bits.push((1, 1));
        let payload = pack_bits(&bits);
        let err = parse_multiview_view_position(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::MultiviewViewPositionNumViewsOutOfRange(1024)
        ));
    }

    #[test]
    fn multiview_view_position_rejects_view_position_above_1023() {
        // num_views_minus1 = 0 → ue "1"
        // view_position[0] = 1024 → ue 11+10 bits as above.
        let mut bits = vec![(1u64, 1u32)];
        for _ in 0..10 {
            bits.push((0, 1));
        }
        bits.push((1, 1));
        for _ in 0..9 {
            bits.push((0, 1));
        }
        bits.push((1, 1));
        let payload = pack_bits(&bits);
        let err = parse_multiview_view_position(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::MultiviewViewPositionViewPositionOutOfRange { i: 0, got: 1024 }
        ));
    }

    #[test]
    fn parse_payload_dispatches_multiview_view_position() {
        let ctx = SeiContext::default();
        let payload = pack_bits(&[
            (0b010, 3), // num_views_minus1 = 1
            (1, 1),     // view_position[0] = 0
            (0b010, 3), // view_position[1] = 1
            (0, 1),     // extension_flag = 0
        ]);
        let got = parse_payload(46, &payload, &ctx).unwrap();
        match got {
            SeiPayload::MultiviewViewPosition(m) => {
                assert_eq!(m.view_positions, vec![0u16, 1u16]);
                assert!(!m.extension_flag);
            }
            other => panic!("expected MultiviewViewPosition, got {:?}", other),
        }
    }

    // §G.13.1.4 / §G.13.2.4 — multiview_scene_info (payload 39, Annex G).
    // Single ue(v) `max_disparity`, range 0..=1023.
    #[test]
    fn multiview_scene_info_zero_disparity() {
        // max_disparity = 0 → ue codeword "1" (1 bit).
        let payload = pack_bits(&[(1, 1)]);
        let got = parse_multiview_scene_info(&payload).unwrap();
        assert_eq!(got.max_disparity, 0);
    }

    #[test]
    fn multiview_scene_info_picks_up_typical_value() {
        // max_disparity = 16 → ue codeword "000010001" (4 leading zeros
        // + leading 1 + 4-bit suffix 0001).
        let payload = pack_bits(&[(0b000010001, 9)]);
        let got = parse_multiview_scene_info(&payload).unwrap();
        assert_eq!(got.max_disparity, 16);
    }

    #[test]
    fn multiview_scene_info_accepts_boundary_1023() {
        // max_disparity = 1023 → codeNum+1 = 1024 = 2^10 → 10 leading
        // zeros + 1 + 10-bit suffix 0b0000000000 = 21 bits. Per
        // §9.1 the ue(v) codeword for value v has Floor(log2(v+1))
        // leading zeros.
        let mut bits: Vec<(u64, u32)> = Vec::new();
        for _ in 0..10 {
            bits.push((0, 1));
        }
        bits.push((1, 1));
        for _ in 0..10 {
            bits.push((0, 1));
        }
        let payload = pack_bits(&bits);
        let got = parse_multiview_scene_info(&payload).unwrap();
        assert_eq!(got.max_disparity, 1023);
    }

    #[test]
    fn multiview_scene_info_rejects_max_disparity_above_1023() {
        // max_disparity = 1024 → ue codeword "00000000001 0000000001"
        // (10 leading zeros + 1 + 10-bit suffix 0b00000_00001).
        let mut bits: Vec<(u64, u32)> = Vec::new();
        for _ in 0..10 {
            bits.push((0, 1));
        }
        bits.push((1, 1));
        for _ in 0..9 {
            bits.push((0, 1));
        }
        bits.push((1, 1));
        let payload = pack_bits(&bits);
        let err = parse_multiview_scene_info(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::MultiviewSceneInfoMaxDisparityOutOfRange(1024)
        ));
    }

    #[test]
    fn parse_payload_dispatches_multiview_scene_info() {
        let ctx = SeiContext::default();
        // max_disparity = 7 → ue codeword "0001000" (3 leading zeros +
        // 1 + 3-bit suffix 0b000).
        let payload = pack_bits(&[(0b0001000, 7)]);
        let got = parse_payload(39, &payload, &ctx).unwrap();
        match got {
            SeiPayload::MultiviewSceneInfo(m) => assert_eq!(m.max_disparity, 7),
            other => panic!("expected MultiviewSceneInfo, got {:?}", other),
        }
    }

    // §G.13.1.8 / §G.13.2.8 — operation_point_not_present (payload 43).
    #[test]
    fn operation_point_not_present_empty_list() {
        // num_operation_points = 0 → ue codeword "1" (1 bit).
        // Per §G.13.2.8 num_operation_points = 0 means "all
        // operation points declared in the prior view_scalability_info
        // SEI message are present" — a valid degenerate case we must
        // accept rather than refuse.
        let payload = pack_bits(&[(1, 1)]);
        let got = parse_operation_point_not_present(&payload).unwrap();
        assert!(got.operation_point_not_present_ids.is_empty());
    }

    #[test]
    fn operation_point_not_present_three_ids() {
        // num_operation_points = 3 → ue codeword "00100" (5 bits).
        // ids = [0, 1, 65535]:
        //   0 → ue "1" (1 bit)
        //   1 → ue "010" (3 bits)
        //   65535 → codeNum+1 = 65536 = 2^16 → 16 leading zeros + 1 +
        //           16-bit suffix 0b0000_0000_0000_0000 = 33 bits.
        let mut bits: Vec<(u64, u32)> = vec![(0b00100, 5), (1, 1), (0b010, 3)];
        for _ in 0..16 {
            bits.push((0, 1));
        }
        bits.push((1, 1));
        for _ in 0..16 {
            bits.push((0, 1));
        }
        let payload = pack_bits(&bits);
        let got = parse_operation_point_not_present(&payload).unwrap();
        assert_eq!(
            got.operation_point_not_present_ids,
            vec![0u16, 1u16, 65535u16]
        );
    }

    #[test]
    fn operation_point_not_present_rejects_id_above_65535() {
        // num_operation_points = 1, id = 65536 → codeNum+1 = 65537 =
        // 2^16 + 1 → 16 leading zeros + 1 + 16-bit suffix
        // 0b0000_0000_0000_0001 = 33 bits.
        let mut bits: Vec<(u64, u32)> = vec![(0b010, 3)];
        for _ in 0..16 {
            bits.push((0, 1));
        }
        bits.push((1, 1));
        for _ in 0..15 {
            bits.push((0, 1));
        }
        bits.push((1, 1));
        let payload = pack_bits(&bits);
        let err = parse_operation_point_not_present(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::OperationPointNotPresentIdOutOfRange { i: 0, got: 65536 }
        ));
    }

    #[test]
    fn operation_point_not_present_rejects_count_above_65536() {
        // num_operation_points = 65537 → codeNum+1 = 65538 = 2^16 + 2
        // → 16 leading zeros + 1 + 16-bit suffix
        // 0b0000_0000_0000_0010 = 33 bits. The §G.13.2.8 65536 cap
        // must reject BEFORE the per-id loop allocates.
        let mut bits: Vec<(u64, u32)> = Vec::new();
        for _ in 0..16 {
            bits.push((0, 1));
        }
        bits.push((1, 1));
        for _ in 0..14 {
            bits.push((0, 1));
        }
        bits.push((1, 1));
        bits.push((0, 1));
        let payload = pack_bits(&bits);
        let err = parse_operation_point_not_present(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::OperationPointNotPresentCountOutOfRange(65537)
        ));
    }

    #[test]
    fn parse_payload_dispatches_operation_point_not_present() {
        let ctx = SeiContext::default();
        // num_operation_points = 2, ids = [10, 200]:
        //   2 → ue "011" (3 bits)
        //   10 → ue "0001011" (3 zeros + 1 + 3-bit suffix 0b011 = 7 bits)
        //   200 → ue: 7 leading zeros + 1 + 7-bit suffix
        //               0b1001001 = 15 bits.
        let mut bits: Vec<(u64, u32)> = vec![(0b011, 3), (0b0001011, 7)];
        for _ in 0..7 {
            bits.push((0, 1));
        }
        bits.push((1, 1));
        bits.push((1, 1));
        bits.push((0, 1));
        bits.push((0, 1));
        bits.push((1, 1));
        bits.push((0, 1));
        bits.push((0, 1));
        bits.push((1, 1));
        let payload = pack_bits(&bits);
        let got = parse_payload(43, &payload, &ctx).unwrap();
        match got {
            SeiPayload::OperationPointNotPresent(o) => {
                assert_eq!(o.operation_point_not_present_ids, vec![10u16, 200u16]);
            }
            other => panic!("expected OperationPointNotPresent, got {:?}", other),
        }
    }

    // §G.13.1.9 / §G.13.2.9 — base_view_temporal_hrd (payload 44).

    /// Build the bits of one §E.1.2 `hrd_parameters()` block with a
    /// single SchedSelIdx (cpb_cnt_minus1 = 0) and all-zero values, so
    /// tests can append/prepend other fields around it.
    fn minimal_hrd_bits() -> Vec<(u64, u32)> {
        vec![
            (1, 1), // cpb_cnt_minus1 = 0 → ue "1"
            (0, 4), // bit_rate_scale
            (0, 4), // cpb_size_scale
            (1, 1), // bit_rate_value_minus1[0] = 0 → ue "1"
            (1, 1), // cpb_size_value_minus1[0] = 0 → ue "1"
            (0, 1), // cbr_flag[0]
            (0, 5), // initial_cpb_removal_delay_length_minus1
            (0, 5), // cpb_removal_delay_length_minus1
            (0, 5), // dpb_output_delay_length_minus1
            (0, 5), // time_offset_length
        ]
    }

    #[test]
    fn base_view_temporal_hrd_single_layer_no_timing_no_hrd() {
        // num_of_temporal_layers_in_base_view_minus1 = 0 → ue "1"
        // layer[0]: temporal_id u(3) = 5
        //           sei_mvc_timing_info_present_flag u(1) = 0
        //           sei_mvc_nal_hrd_parameters_present_flag u(1) = 0
        //           sei_mvc_vcl_hrd_parameters_present_flag u(1) = 0
        //           (low_delay flag NOT coded — neither HRD present)
        //           sei_mvc_pic_struct_present_flag u(1) = 1
        let payload = pack_bits(&[(1, 1), (5, 3), (0, 1), (0, 1), (0, 1), (1, 1)]);
        let got = parse_base_view_temporal_hrd(&payload).unwrap();
        assert_eq!(got.layers.len(), 1);
        let l = &got.layers[0];
        assert_eq!(l.temporal_id, 5);
        assert!(l.timing_info.is_none());
        assert!(l.nal_hrd_parameters.is_none());
        assert!(l.vcl_hrd_parameters.is_none());
        assert_eq!(l.low_delay_hrd_flag, None);
        assert!(l.pic_struct_present_flag);
    }

    #[test]
    fn base_view_temporal_hrd_layer_with_timing_and_nal_hrd() {
        // num_of_temporal_layers_in_base_view_minus1 = 0 → ue "1"
        // layer[0]: temporal_id u(3) = 2
        //           timing_info_present u(1) = 1
        //             num_units_in_tick u(32) = 1001
        //             time_scale u(32) = 60000
        //             fixed_frame_rate_flag u(1) = 1
        //           nal_hrd_present u(1) = 1 → minimal hrd_parameters()
        //           vcl_hrd_present u(1) = 0
        //           low_delay_hrd_flag u(1) = 1 (coded: nal present)
        //           pic_struct_present_flag u(1) = 0
        let mut bits: Vec<(u64, u32)> = vec![
            (1, 1),
            (2, 3),
            (1, 1),
            (1001, 32),
            (60000, 32),
            (1, 1),
            (1, 1), // nal_hrd_parameters_present_flag
        ];
        bits.extend(minimal_hrd_bits());
        bits.push((0, 1)); // vcl_hrd_parameters_present_flag
        bits.push((1, 1)); // low_delay_hrd_flag
        bits.push((0, 1)); // pic_struct_present_flag
        let payload = pack_bits(&bits);
        let got = parse_base_view_temporal_hrd(&payload).unwrap();
        assert_eq!(got.layers.len(), 1);
        let l = &got.layers[0];
        assert_eq!(l.temporal_id, 2);
        let t = l.timing_info.as_ref().expect("timing present");
        assert_eq!(t.num_units_in_tick, 1001);
        assert_eq!(t.time_scale, 60000);
        assert!(t.fixed_frame_rate_flag);
        let nal = l.nal_hrd_parameters.as_ref().expect("nal hrd present");
        assert_eq!(nal.cpb_cnt_minus1, 0);
        assert!(l.vcl_hrd_parameters.is_none());
        assert_eq!(l.low_delay_hrd_flag, Some(true));
        assert!(!l.pic_struct_present_flag);
    }

    #[test]
    fn base_view_temporal_hrd_two_layers_via_dispatch() {
        // num_of_temporal_layers_in_base_view_minus1 = 1 → ue "010"
        // layer[0]: tid=0, no timing, no hrd, pic_struct=0
        // layer[1]: tid=1, no timing, no hrd, pic_struct=1
        let payload = pack_bits(&[
            (0b010, 3),
            (0, 3),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1), // layer 0
            (1, 3),
            (0, 1),
            (0, 1),
            (0, 1),
            (1, 1), // layer 1
        ]);
        let ctx = SeiContext::default();
        let got = parse_payload(44, &payload, &ctx).unwrap();
        match got {
            SeiPayload::BaseViewTemporalHrd(b) => {
                assert_eq!(b.layers.len(), 2);
                assert_eq!(b.layers[0].temporal_id, 0);
                assert!(!b.layers[0].pic_struct_present_flag);
                assert_eq!(b.layers[1].temporal_id, 1);
                assert!(b.layers[1].pic_struct_present_flag);
            }
            other => panic!("expected BaseViewTemporalHrd, got {:?}", other),
        }
    }

    #[test]
    fn base_view_temporal_hrd_rejects_layer_count_above_8() {
        // num_of_temporal_layers_in_base_view_minus1 = 8 → out of range
        // (§G.13.2.9 caps the minus1 form at 7). ue(8) = codeNum 8
        // → 3 leading zeros + 1 + 3-bit suffix 0b001 = "0001001".
        let payload = pack_bits(&[(0b0001001, 7)]);
        let err = parse_base_view_temporal_hrd(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::BaseViewTemporalHrdLayerCountOutOfRange(8)
        ));
    }

    // §G.13.1.6 / §G.13.2.6 — non_required_view_component (payload 41).
    #[test]
    fn non_required_view_component_single_entry_single_component() {
        // num_info_entries_minus1 = 0 → ue "1"
        // entry[0]:
        //   view_order_index = 1   → ue "010" (codeNum=1)
        //   num_non_required_view_components_minus1 = 0 → ue "1"
        //   index_delta_minus1[0] = 0 → ue "1"
        // Total: 1 + 3 + 1 + 1 = 6 bits.
        let payload = pack_bits(&[(1, 1), (0b010, 3), (1, 1), (1, 1)]);
        let got = parse_non_required_view_component(&payload).unwrap();
        assert_eq!(got.entries.len(), 1);
        assert_eq!(got.entries[0].view_order_index, 1);
        assert_eq!(got.entries[0].index_delta_minus1, vec![0u16]);
    }

    #[test]
    fn non_required_view_component_two_entries_with_multiple_deltas() {
        // Two target views, with multiple non-required deltas each.
        //
        // num_info_entries_minus1 = 1 → ue "010"
        // entry[0]:
        //   view_order_index = 2 → ue "011" (codeNum=2)
        //   num_non_required_view_components_minus1 = 1 → ue "010"
        //   index_delta_minus1[0] = 0 → ue "1"
        //   index_delta_minus1[1] = 1 → ue "010"
        // entry[1]:
        //   view_order_index = 3 → ue "00100" (codeNum=3)
        //   num_non_required_view_components_minus1 = 0 → ue "1"
        //   index_delta_minus1[0] = 2 → ue "011"
        // Total: 3 + 3 + 3 + 1 + 3 + 5 + 1 + 3 = 22 bits.
        let payload = pack_bits(&[
            (0b010, 3),
            (0b011, 3),
            (0b010, 3),
            (1, 1),
            (0b010, 3),
            (0b00100, 5),
            (1, 1),
            (0b011, 3),
        ]);
        let got = parse_non_required_view_component(&payload).unwrap();
        assert_eq!(got.entries.len(), 2);
        assert_eq!(got.entries[0].view_order_index, 2);
        assert_eq!(got.entries[0].index_delta_minus1, vec![0u16, 1u16]);
        assert_eq!(got.entries[1].view_order_index, 3);
        assert_eq!(got.entries[1].index_delta_minus1, vec![2u16]);
    }

    #[test]
    fn non_required_view_component_rejects_view_order_index_zero() {
        // §G.13.2.6 — view_order_index shall be in 1..=num_views_minus1;
        // a 0 value is below the lower bound (view 0 is the base view,
        // not a target).
        //
        // num_info_entries_minus1 = 0 → ue "1"
        // entry[0].view_order_index = 0 → ue "1"
        let payload = pack_bits(&[(1, 1), (1, 1)]);
        let err = parse_non_required_view_component(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::NonRequiredViewComponentViewOrderIndexOutOfRange { i: 0, got: 0 }
        ));
    }

    #[test]
    fn non_required_view_component_rejects_num_info_entries_above_bound() {
        // §G.13.2.6 — num_info_entries_minus1 ≤ num_views_minus1 − 1.
        // With Annex G's absolute num_views_minus1 ≤ 1023, the cap is
        // 1022. We test the boundary: num_info_entries_minus1 = 1023.
        //
        // codeNum = 1023 → codeNum + 1 = 1024 = 2^10 → 10 leading
        // zeros + 1 + 10-bit suffix 0b0000000000 = 21 bits.
        let mut bits: Vec<(u64, u32)> = Vec::new();
        for _ in 0..10 {
            bits.push((0, 1));
        }
        bits.push((1, 1));
        for _ in 0..10 {
            bits.push((0, 1));
        }
        let payload = pack_bits(&bits);
        let err = parse_non_required_view_component(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::NonRequiredViewComponentNumInfoEntriesOutOfRange(1023)
        ));
    }

    #[test]
    fn non_required_view_component_rejects_inner_count_above_voi_minus_one() {
        // §G.13.2.6 — num_non_required_view_components_minus1[i] ≤
        // view_order_index − 1. With view_order_index = 1, the cap is
        // 0; setting the count to 1 must reject before allocation.
        //
        // num_info_entries_minus1 = 0 → ue "1"
        // entry[0]:
        //   view_order_index = 1 → ue "010"
        //   num_non_required_view_components_minus1 = 1 → ue "010"
        let payload = pack_bits(&[(1, 1), (0b010, 3), (0b010, 3)]);
        let err = parse_non_required_view_component(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::NonRequiredViewComponentCountOutOfRange {
                i: 0,
                got: 1,
                view_order_index: 1
            }
        ));
    }

    #[test]
    fn non_required_view_component_rejects_index_delta_above_voi_minus_one() {
        // §G.13.2.6 — index_delta_minus1[i][j] ≤ view_order_index − 1.
        // With view_order_index = 2, the cap is 1; setting the delta
        // to 2 must reject. (One inner component so the per-entry
        // allocation succeeds and we reach the inner range check.)
        //
        // num_info_entries_minus1 = 0 → ue "1"
        // entry[0]:
        //   view_order_index = 2 → ue "011"
        //   num_non_required_view_components_minus1 = 0 → ue "1"
        //   index_delta_minus1[0] = 2 → ue "011"
        let payload = pack_bits(&[(1, 1), (0b011, 3), (1, 1), (0b011, 3)]);
        let err = parse_non_required_view_component(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::NonRequiredViewComponentIndexDeltaOutOfRange {
                i: 0,
                j: 0,
                got: 2,
                view_order_index: 2
            }
        ));
    }

    #[test]
    fn parse_payload_dispatches_non_required_view_component() {
        // Same single-entry single-component shape as the smoke test
        // above, but routed via `parse_payload`'s dispatch arm.
        let ctx = SeiContext::default();
        let payload = pack_bits(&[(1, 1), (0b010, 3), (1, 1), (1, 1)]);
        let got = parse_payload(41, &payload, &ctx).unwrap();
        match got {
            SeiPayload::NonRequiredViewComponent(n) => {
                assert_eq!(n.entries.len(), 1);
                assert_eq!(n.entries[0].view_order_index, 1);
                assert_eq!(n.entries[0].index_delta_minus1, vec![0u16]);
            }
            other => panic!("expected NonRequiredViewComponent, got {:?}", other),
        }
    }

    // Round-213 — §G.13.2.5 multiview_acquisition_info (payload
    // type 40). Tests cover the four parse paths:
    //
    //   1. Both flags off — only num_views + the two flag bits are
    //      consumed.
    //   2. Intrinsic only, intrinsic_params_equal_flag = 1 — single
    //      shared camera entry.
    //   3. Extrinsic only — full 3×3 R + 3-vector T per camera.
    //   4. Range-check rejections on num_views_minus1 + each prec_*.
    //
    // A separate test exercises the §G.13.2.5 mantissa-width formula
    // (denormal e == 0, normal 0 < e < 63, reserved e == 63) and
    // the FloatComponent::to_f64 reconstruction.

    #[test]
    fn multiview_acquisition_info_no_flags_set() {
        // num_views_minus1 = 0 → ue "1"
        // intrinsic_param_flag = 0
        // extrinsic_param_flag = 0
        // Total: 1 + 1 + 1 = 3 bits.
        let payload = pack_bits(&[(1, 1), (0, 1), (0, 1)]);
        let info = parse_multiview_acquisition_info(&payload).unwrap();
        assert_eq!(info.num_views_minus1, 0);
        assert!(info.intrinsic.is_none());
        assert!(info.extrinsic.is_none());
    }

    #[test]
    fn multiview_acquisition_info_intrinsic_equal_flag_single_camera() {
        // Two views (num_views_minus1 = 1) but
        // intrinsic_params_equal_flag = 1 so only one intrinsic
        // entry is signalled.
        //
        // num_views_minus1 = 1                          → ue "010"      (3 bits)
        // intrinsic_param_flag = 1                       → u(1) "1"      (1 bit)
        // extrinsic_param_flag = 0                       → u(1) "0"      (1 bit)
        // intrinsic_params_equal_flag = 1                → u(1) "1"      (1 bit)
        // prec_focal_length = 1                          → ue "010"      (3 bits)
        // prec_principal_point = 1                       → ue "010"      (3 bits)
        // prec_skew_factor = 1                           → ue "010"      (3 bits)
        //
        // Five FloatComponents, each (sign u(1) + exponent u(6) +
        // mantissa u(v)) where v = max(0, e + prec − 31) for
        // 0 < e < 63. With prec = 1 and e = 30 (well below 63):
        // v = max(0, 30 + 1 − 31) = 0 → mantissa is zero-width.
        // Pick sign = 0, exponent = 30, mantissa = (absent).
        //
        // Per FloatComponent: u(1)=0, u(6)=30 (= 0b011110) → 7 bits.
        // Five components × 7 bits = 35 bits.
        let mut fields: Vec<(u64, u32)> = vec![
            (0b010, 3), // num_views_minus1 = 1
            (1, 1),     // intrinsic_param_flag = 1
            (0, 1),     // extrinsic_param_flag = 0
            (1, 1),     // intrinsic_params_equal_flag = 1
            (0b010, 3), // prec_focal_length = 1
            (0b010, 3), // prec_principal_point = 1
            (0b010, 3), // prec_skew_factor = 1
        ];
        // Five FloatComponents each with sign=0, exponent=30,
        // mantissa absent (width 0 since 0 < e < 63 → v = e + prec − 31
        // = 30 + 1 − 31 = 0).
        for _ in 0..5 {
            fields.push((0, 1)); // sign
            fields.push((30, 6)); // exponent
        }
        let payload = pack_bits(&fields);

        let info = parse_multiview_acquisition_info(&payload).unwrap();
        assert_eq!(info.num_views_minus1, 1);
        let intr = info.intrinsic.expect("intrinsic block populated");
        assert!(intr.intrinsic_params_equal_flag);
        assert_eq!(intr.prec_focal_length, 1);
        assert_eq!(intr.prec_principal_point, 1);
        assert_eq!(intr.prec_skew_factor, 1);
        // Only one camera entry shared across both views.
        assert_eq!(intr.cameras.len(), 1);
        let cam = intr.cameras[0];
        assert_eq!(cam.focal_length_x.exponent, 30);
        assert_eq!(cam.focal_length_x.mantissa_width, 0);
        assert_eq!(cam.focal_length_x.mantissa, 0);
        assert!(!cam.focal_length_x.sign);
        assert!(info.extrinsic.is_none());
    }

    #[test]
    fn multiview_acquisition_info_extrinsic_only_single_view() {
        // Single view (num_views_minus1 = 0), no intrinsic, full
        // extrinsic block (3×3 R + 3 t entries = 12 FloatComponents).
        // Pick prec_rotation_param = 0, prec_translation_param = 0
        // and each exponent = 0 → mantissa width = max(0, 0 − 30) = 0,
        // so each FloatComponent is 7 bits (sign + exponent).
        //
        // num_views_minus1 = 0                  → ue "1"  (1 bit)
        // intrinsic_param_flag = 0              → u(1) "0"
        // extrinsic_param_flag = 1              → u(1) "1"
        // prec_rotation_param = 0               → ue "1"  (1 bit)
        // prec_translation_param = 0            → ue "1"  (1 bit)
        // 12 FloatComponents × 7 bits each      = 84 bits
        // Total: 89 bits → 12 bytes (5 bits padding).
        let mut fields: Vec<(u64, u32)> = vec![
            (1, 1), // num_views_minus1 = 0
            (0, 1), // intrinsic_param_flag = 0
            (1, 1), // extrinsic_param_flag = 1
            (1, 1), // prec_rotation_param = 0
            (1, 1), // prec_translation_param = 0
        ];
        // Per row j ∈ {1..3}: 3 r entries + 1 t entry = 4 components.
        // Pick exponents in 0..=31 so the §G.13.2.5 mantissa-width
        // formula yields width 0 for both prec_rotation_param == 0
        // and prec_translation_param == 0 (otherwise an exponent in
        // 32..62 introduces a 1+ bit mantissa that re-frames every
        // subsequent component in this fixed-bit fixture).
        let pattern = [
            // (sign, exponent) — 9 r entries
            (0u64, 11u64),
            (1, 12),
            (0, 13),
            (1, 21),
            (0, 22),
            (1, 23),
            (0, 14),
            (1, 24),
            (0, 31),
        ];
        let t_pattern = [(1u64, 1u64), (0, 2), (1, 3)];
        // §G.13.1.5 order: for j { for k r[j][k]; t[j] }
        for j in 0..3 {
            for k in 0..3 {
                let (s, e) = pattern[j * 3 + k];
                fields.push((s, 1));
                fields.push((e, 6));
            }
            let (s, e) = t_pattern[j];
            fields.push((s, 1));
            fields.push((e, 6));
        }
        let payload = pack_bits(&fields);

        let info = parse_multiview_acquisition_info(&payload).unwrap();
        assert_eq!(info.num_views_minus1, 0);
        assert!(info.intrinsic.is_none());
        let extr = info.extrinsic.expect("extrinsic block populated");
        assert_eq!(extr.prec_rotation_param, 0);
        assert_eq!(extr.prec_translation_param, 0);
        assert_eq!(extr.cameras.len(), 1);
        let cam = extr.cameras[0];
        // Spot-check a few R entries.
        assert_eq!(cam.r[0][0].exponent, 11);
        assert!(!cam.r[0][0].sign);
        assert_eq!(cam.r[2][2].exponent, 31);
        assert_eq!(cam.r[2][2].mantissa_width, 0);
        // Spot-check the translation.
        assert_eq!(cam.t[0].exponent, 1);
        assert!(cam.t[0].sign);
        assert_eq!(cam.t[2].exponent, 3);
    }

    #[test]
    fn multiview_acquisition_info_rejects_num_views_above_1023() {
        // Encode num_views_minus1 = 1024 as ue(1024):
        // codeword length = 21 bits (10 leading zeros + 1 + 10
        // info bits for the value 1024 = 0b10000000001 with the
        // leading 1 acting as the marker), and Exp-Golomb decoder
        // returns ue = 2^10 + bits − 1 = 1024.
        //
        // 10 zeros + 1 + "0000000001" → codeNum = 0b10000000001 −1 = 1024.
        // Total bits: 21 → 3 bytes.
        let payload = pack_bits(&[(0, 10), (1, 1), (0b0000000001, 10)]);
        let err = parse_multiview_acquisition_info(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::MultiviewAcquisitionInfoNumViewsOutOfRange(1024)
        ));
    }

    #[test]
    fn multiview_acquisition_info_rejects_prec_focal_length_above_31() {
        // num_views_minus1 = 0, intrinsic_param_flag = 1,
        // extrinsic_param_flag = 0, intrinsic_params_equal_flag = 1,
        // prec_focal_length = 32 → ue("0000010001") (10 bits encode
        // codeNum 33 → ue value 32).
        //
        // Wait: codeNum k → ue value = k. The ue codeword for value
        // 32 has codeword "000001000001" → 12 bits (5 leading zeros
        // + 1 + 6 info bits = 0b100001 - 1 = 32). Let me just feed
        // a value > 31 via the marker-only form:
        // ue(32) = "0000001000001" (6 zeros + 1 + 0b000001 = ... hmm).
        //
        // Simpler: encode value via the 0 + marker form:
        // For value v, codeword is the (k+1)-bit binary
        // representation of (v+1) with the leading 1 implicit-marker:
        //   v = 32  →  v+1 = 33 = 0b100001 (6 bits)
        //   codeword = 5 leading zeros + 0b100001  (11 bits total).
        let payload = pack_bits(&[
            (1, 1), // num_views_minus1 = 0
            (1, 1), // intrinsic_param_flag = 1
            (0, 1), // extrinsic_param_flag = 0
            (1, 1), // intrinsic_params_equal_flag = 1
            // prec_focal_length = 32 → ue codeword "000001000001"
            // (5 zeros + 0b100001 = 11 bits).
            (0, 5),
            (0b100001, 6),
        ]);
        let err = parse_multiview_acquisition_info(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::MultiviewAcquisitionInfoPrecOutOfRange {
                field: "prec_focal_length",
                got: 32
            }
        ));
    }

    #[test]
    fn multiview_acquisition_info_mantissa_width_formula() {
        // §G.13.2.5: cover the three branches of the
        // mantissa-width formula.
        //
        // e == 0       → width = max(0, prec − 30)
        // 0 < e < 63   → width = max(0, e + prec − 31)
        // e == 63      → reserved; we store width = 0
        assert_eq!(mantissa_width_g1325(0, 0), 0); // prec=0, e=0 → 0
        assert_eq!(mantissa_width_g1325(31, 0), 1); // prec=31, e=0 → max(0, 1) = 1
        assert_eq!(mantissa_width_g1325(30, 0), 0); // prec=30, e=0 → max(0, 0) = 0
        assert_eq!(mantissa_width_g1325(0, 1), 0); // prec=0, e=1 → max(0, −30) = 0
        assert_eq!(mantissa_width_g1325(31, 1), 1); // prec=31, e=1 → 1+31−31 = 1
        assert_eq!(mantissa_width_g1325(31, 62), 62); // prec=31, e=62 → 62 (max)
        assert_eq!(mantissa_width_g1325(31, 63), 0); // e=63 reserved → 0
        assert_eq!(mantissa_width_g1325(0, 63), 0); // e=63 reserved → 0
    }

    #[test]
    fn float_component_to_f64_branches() {
        // Denormal: e == 0, n == 1, v == 1
        //   x = (-1)^0 * 2^(-30) * (1 / 2) = 2^(-31)
        let denormal = FloatComponent {
            sign: false,
            exponent: 0,
            mantissa: 1,
            mantissa_width: 1,
        };
        let expected_denormal = 2f64.powi(-31);
        assert!((denormal.to_f64() - expected_denormal).abs() < 1e-300);

        // Normal: e == 31, prec == 0 → v = 0, mantissa absent
        //   x = (-1)^0 * 2^(31 − 31) * (1 + 0) = 1.0
        let unity = FloatComponent {
            sign: false,
            exponent: 31,
            mantissa: 0,
            mantissa_width: 0,
        };
        assert_eq!(unity.to_f64(), 1.0);

        // Normal negative with a mantissa:
        //   e = 32, prec = 31 → v = 32 + 31 − 31 = 32
        //   mantissa = 0x80000000 (top bit set) → n / 2^v = 0.5
        //   x = (-1)^1 * 2^(32 − 31) * (1 + 0.5) = -3.0
        let neg_three = FloatComponent {
            sign: true,
            exponent: 32,
            mantissa: 0x8000_0000,
            mantissa_width: 32,
        };
        assert_eq!(neg_three.to_f64(), -3.0);

        // Reserved (e == 63) → NaN.
        let unspec = FloatComponent {
            sign: false,
            exponent: 63,
            mantissa: 0,
            mantissa_width: 0,
        };
        assert!(unspec.to_f64().is_nan());
    }

    #[test]
    fn parse_payload_dispatches_multiview_acquisition_info() {
        // Both flags off — three-bit payload routed via dispatcher.
        let ctx = SeiContext::default();
        let payload = pack_bits(&[(1, 1), (0, 1), (0, 1)]);
        let got = parse_payload(40, &payload, &ctx).unwrap();
        match got {
            SeiPayload::MultiviewAcquisitionInfo(m) => {
                assert_eq!(m.num_views_minus1, 0);
                assert!(m.intrinsic.is_none());
                assert!(m.extrinsic.is_none());
            }
            other => panic!("expected MultiviewAcquisitionInfo, got {other:?}"),
        }
    }

    // Round-226 — §H.13.2.4 three_dimensional_reference_displays_info
    // (payload type 51, Annex H). Tests cover:
    //
    //   1. Single reference display, ref_viewing_distance_flag = 0,
    //      additional_shift_present_flag = 0 — minimum-syntax path.
    //   2. Two displays, ref_viewing_distance_flag = 1,
    //      additional_shift_present_flag = 1 — full per-display
    //      block exercised.
    //   3. UnsignedFloatComponent::to_f64 branches (normal /
    //      denormal / reserved).
    //   4. Range-check rejections on the three prec_* fields and
    //      num_ref_displays_minus1.
    //   5. parse_payload dispatcher round-trip.

    #[test]
    fn three_dimensional_reference_displays_info_minimum_single_display() {
        // prec_ref_baseline = 0                  → ue "1"        (1 bit)
        // prec_ref_display_width = 0             → ue "1"        (1 bit)
        // ref_viewing_distance_flag = 0          → u(1)          (1 bit)
        // num_ref_displays_minus1 = 0            → ue "1"        (1 bit)
        // One reference display, ref_viewing_distance_flag = 0:
        //   exponent_ref_baseline u(6) = 31      → (6 bits)
        //   mantissa_ref_baseline u(0) = absent  (prec=0, 0<e<63
        //                                          → v = 0+0-31 = 0)
        //   exponent_ref_display_width u(6) = 31 → (6 bits)
        //   mantissa_ref_display_width u(0)      = absent
        //   additional_shift_present_flag = 0    → u(1)          (1 bit)
        //   no num_sample_shift_plus512
        // three_dimensional_reference_displays_extension_flag = 0
        //                                        → u(1)          (1 bit)
        let payload = pack_bits(&[
            (1, 1),  // prec_ref_baseline = 0
            (1, 1),  // prec_ref_display_width = 0
            (0, 1),  // ref_viewing_distance_flag = 0
            (1, 1),  // num_ref_displays_minus1 = 0
            (31, 6), // exponent_ref_baseline
            (31, 6), // exponent_ref_display_width
            (0, 1),  // additional_shift_present_flag = 0
            (0, 1),  // extension_flag = 0
        ]);
        let info = parse_three_dimensional_reference_displays_info(&payload).unwrap();
        assert_eq!(info.prec_ref_baseline, 0);
        assert_eq!(info.prec_ref_display_width, 0);
        assert!(info.prec_ref_viewing_dist.is_none());
        assert_eq!(info.displays.len(), 1);
        let d = info.displays[0];
        assert_eq!(d.ref_baseline.exponent, 31);
        assert_eq!(d.ref_baseline.mantissa_width, 0);
        assert_eq!(d.ref_display_width.exponent, 31);
        assert!(d.ref_viewing_distance.is_none());
        assert!(!d.additional_shift_present_flag);
        assert!(d.num_sample_shift_plus512.is_none());
        assert!(!info.extension_flag);
    }

    #[test]
    fn three_dimensional_reference_displays_info_two_displays_with_viewing_distance() {
        // num_ref_displays_minus1 = 1 (two displays),
        // ref_viewing_distance_flag = 1, additional_shift_present_flag
        // = 1 on the first display only — exercises every conditional
        // branch in the per-display loop.
        let mut fields: Vec<(u64, u32)> = vec![
            (1, 1),     // prec_ref_baseline = 0
            (1, 1),     // prec_ref_display_width = 0
            (1, 1),     // ref_viewing_distance_flag = 1
            (1, 1),     // prec_ref_viewing_dist = 0
            (0b010, 3), // num_ref_displays_minus1 = 1
        ];
        // Display 0: every component uses exponent in 0..=31, prec=0
        // → mantissa width = 0 for both formulas.
        fields.push((20, 6)); // exponent_ref_baseline
        fields.push((21, 6)); // exponent_ref_display_width
        fields.push((22, 6)); // exponent_ref_viewing_distance
        fields.push((1, 1)); // additional_shift_present_flag = 1
                             // num_sample_shift_plus512 = 768 → centre + 256 samples.
        fields.push((768, 10));
        // Display 1: no additional shift signalled.
        fields.push((10, 6)); // exponent_ref_baseline
        fields.push((11, 6)); // exponent_ref_display_width
        fields.push((12, 6)); // exponent_ref_viewing_distance
        fields.push((0, 1)); // additional_shift_present_flag = 0
                             // extension_flag = 0
        fields.push((0, 1));
        let payload = pack_bits(&fields);
        let info = parse_three_dimensional_reference_displays_info(&payload).unwrap();
        assert_eq!(info.prec_ref_viewing_dist, Some(0));
        assert_eq!(info.displays.len(), 2);
        // Display 0.
        let d0 = info.displays[0];
        assert_eq!(d0.ref_baseline.exponent, 20);
        assert_eq!(d0.ref_display_width.exponent, 21);
        assert_eq!(d0.ref_viewing_distance.expect("rvd present").exponent, 22);
        assert!(d0.additional_shift_present_flag);
        assert_eq!(d0.num_sample_shift_plus512, Some(768));
        // Display 1.
        let d1 = info.displays[1];
        assert_eq!(d1.ref_baseline.exponent, 10);
        assert_eq!(d1.ref_display_width.exponent, 11);
        assert_eq!(d1.ref_viewing_distance.expect("rvd present").exponent, 12);
        assert!(!d1.additional_shift_present_flag);
        assert!(d1.num_sample_shift_plus512.is_none());
        assert!(!info.extension_flag);
    }

    #[test]
    fn three_dimensional_reference_displays_info_extension_flag_propagates() {
        // Single display, extension_flag = 1 — verify the trailing
        // bit is surfaced (per §H.13.2.4 the trailing region is
        // reserved; we ignore any subsequent data).
        let payload = pack_bits(&[
            (1, 1),  // prec_ref_baseline = 0
            (1, 1),  // prec_ref_display_width = 0
            (0, 1),  // ref_viewing_distance_flag = 0
            (1, 1),  // num_ref_displays_minus1 = 0
            (31, 6), // exponent_ref_baseline
            (31, 6), // exponent_ref_display_width
            (0, 1),  // additional_shift_present_flag = 0
            (1, 1),  // extension_flag = 1 (reserved, ignored)
        ]);
        let info = parse_three_dimensional_reference_displays_info(&payload).unwrap();
        assert!(info.extension_flag);
    }

    // Round-250 — §H.13.2.4 typed accessor on the already-parsed
    // `three_dimensional_reference_displays_info` body: surface the
    // signed `NumSampleShift[i] = num_sample_shift_plus512[i] − 512`
    // from the biased u(10) field stored on each `ReferenceDisplay`.

    #[test]
    fn reference_display_num_sample_shift_returns_none_when_absent() {
        // §H.13.2.4 — additional_shift_present_flag == 0 means the
        // shift was not signalled. The spec does not define an
        // inferred value, so the accessor returns None.
        let d = ReferenceDisplay {
            ref_baseline: UnsignedFloatComponent {
                exponent: 31,
                mantissa: 0,
                mantissa_width: 0,
            },
            ref_display_width: UnsignedFloatComponent {
                exponent: 31,
                mantissa: 0,
                mantissa_width: 0,
            },
            ref_viewing_distance: None,
            additional_shift_present_flag: false,
            num_sample_shift_plus512: None,
        };
        assert_eq!(d.num_sample_shift(), None);
    }

    #[test]
    fn reference_display_num_sample_shift_zero_at_bias() {
        // §H.13.2.4 — num_sample_shift_plus512[i] == 512 is the
        // "recommend that shifting is not applied" centre point; the
        // semantic shift is 0.
        let d = ReferenceDisplay {
            ref_baseline: UnsignedFloatComponent {
                exponent: 31,
                mantissa: 0,
                mantissa_width: 0,
            },
            ref_display_width: UnsignedFloatComponent {
                exponent: 31,
                mantissa: 0,
                mantissa_width: 0,
            },
            ref_viewing_distance: None,
            additional_shift_present_flag: true,
            num_sample_shift_plus512: Some(512),
        };
        assert_eq!(d.num_sample_shift(), Some(0));
    }

    #[test]
    fn reference_display_num_sample_shift_positive_branch() {
        // §H.13.2.4 — values above the 512 centre map to a positive
        // shift of (num_sample_shift_plus512 − 512) samples to the
        // right of the left view. Use the test fixture's value 768
        // → semantic +256.
        let d = ReferenceDisplay {
            ref_baseline: UnsignedFloatComponent {
                exponent: 31,
                mantissa: 0,
                mantissa_width: 0,
            },
            ref_display_width: UnsignedFloatComponent {
                exponent: 31,
                mantissa: 0,
                mantissa_width: 0,
            },
            ref_viewing_distance: None,
            additional_shift_present_flag: true,
            num_sample_shift_plus512: Some(768),
        };
        assert_eq!(d.num_sample_shift(), Some(256));
    }

    #[test]
    fn reference_display_num_sample_shift_negative_branch() {
        // §H.13.2.4 — values below the 512 centre map to a negative
        // shift; |NumSampleShift| samples to the left of the right
        // view. 256 → semantic −256.
        let d = ReferenceDisplay {
            ref_baseline: UnsignedFloatComponent {
                exponent: 31,
                mantissa: 0,
                mantissa_width: 0,
            },
            ref_display_width: UnsignedFloatComponent {
                exponent: 31,
                mantissa: 0,
                mantissa_width: 0,
            },
            ref_viewing_distance: None,
            additional_shift_present_flag: true,
            num_sample_shift_plus512: Some(256),
        };
        assert_eq!(d.num_sample_shift(), Some(-256));
    }

    #[test]
    fn reference_display_num_sample_shift_extreme_endpoints() {
        // §H.13.2.4 — full bitstream range 0..=1023 (u(10)) maps to
        // semantic −512..=511. Cover both endpoints to confirm no
        // off-by-one in the bias subtraction.
        let template = ReferenceDisplay {
            ref_baseline: UnsignedFloatComponent {
                exponent: 31,
                mantissa: 0,
                mantissa_width: 0,
            },
            ref_display_width: UnsignedFloatComponent {
                exponent: 31,
                mantissa: 0,
                mantissa_width: 0,
            },
            ref_viewing_distance: None,
            additional_shift_present_flag: true,
            num_sample_shift_plus512: Some(0),
        };
        // Bias subtraction floor: 0 − 512 = −512.
        assert_eq!(template.num_sample_shift(), Some(-512));
        let max = ReferenceDisplay {
            num_sample_shift_plus512: Some(1023),
            ..template
        };
        // Bias subtraction ceiling: 1023 − 512 = 511.
        assert_eq!(max.num_sample_shift(), Some(511));
    }

    #[test]
    fn reference_display_num_sample_shift_round_trip_via_parser() {
        // End-to-end: parse a §H.13.1.4 payload with two displays
        // (one with shift signalled, one without) and confirm the
        // typed accessor returns the §H.13.2.4 semantic value on
        // both. The first display uses num_sample_shift_plus512 =
        // 768, identical to the existing parser fixture, so the
        // expected accessor reading is +256.
        let mut fields: Vec<(u64, u32)> = vec![
            (1, 1),     // prec_ref_baseline = 0
            (1, 1),     // prec_ref_display_width = 0
            (1, 1),     // ref_viewing_distance_flag = 1
            (1, 1),     // prec_ref_viewing_dist = 0
            (0b010, 3), // num_ref_displays_minus1 = 1
        ];
        // Display 0 — additional_shift_present_flag = 1, biased value 768.
        fields.push((20, 6)); // exponent_ref_baseline
        fields.push((21, 6)); // exponent_ref_display_width
        fields.push((22, 6)); // exponent_ref_viewing_distance
        fields.push((1, 1)); // additional_shift_present_flag = 1
        fields.push((768, 10));
        // Display 1 — additional_shift_present_flag = 0.
        fields.push((10, 6));
        fields.push((11, 6));
        fields.push((12, 6));
        fields.push((0, 1));
        fields.push((0, 1)); // extension_flag = 0
        let payload = pack_bits(&fields);
        let info = parse_three_dimensional_reference_displays_info(&payload).unwrap();
        assert_eq!(info.displays.len(), 2);
        assert_eq!(info.displays[0].num_sample_shift(), Some(256));
        assert_eq!(info.displays[1].num_sample_shift(), None);
    }

    #[test]
    fn unsigned_float_component_to_f64_branches() {
        // Normal: e = 31, prec = 0 → v = 0 → x = 2^0 * 1 = 1.0
        let unity = UnsignedFloatComponent {
            exponent: 31,
            mantissa: 0,
            mantissa_width: 0,
        };
        assert_eq!(unity.to_f64(), 1.0);
        // Normal positive with a mantissa:
        //   e = 32, prec = 31 → v = 32 + 31 − 31 = 32
        //   mantissa = 0x8000_0000 (top bit) → n / 2^v = 0.5
        //   x = 2^1 * (1 + 0.5) = 3.0  (sign-less per H.13.2.4)
        let three = UnsignedFloatComponent {
            exponent: 32,
            mantissa: 0x8000_0000,
            mantissa_width: 32,
        };
        assert_eq!(three.to_f64(), 3.0);
        // Denormal: e = 0, n = 1, v = 1 → x = 2^(-30) * (1/2) = 2^-31
        let denormal = UnsignedFloatComponent {
            exponent: 0,
            mantissa: 1,
            mantissa_width: 1,
        };
        assert!((denormal.to_f64() - 2f64.powi(-31)).abs() < 1e-300);
        // Reserved: e = 63 → NaN.
        let unspec = UnsignedFloatComponent {
            exponent: 63,
            mantissa: 0,
            mantissa_width: 0,
        };
        assert!(unspec.to_f64().is_nan());
    }

    #[test]
    fn three_dimensional_reference_displays_info_rejects_prec_ref_baseline_above_31() {
        // prec_ref_baseline = 32 → ue codeword "000001000001"
        // (5 zeros + 0b100001, 11 bits total). The first field
        // checked is prec_ref_baseline, so no earlier bits.
        let payload = pack_bits(&[(0, 5), (0b100001, 6)]);
        let err = parse_three_dimensional_reference_displays_info(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::ThreeDimensionalReferenceDisplaysInfoPrecOutOfRange {
                field: "prec_ref_baseline",
                got: 32
            }
        ));
    }

    #[test]
    fn three_dimensional_reference_displays_info_rejects_prec_ref_viewing_dist_above_31() {
        // prec_ref_baseline = 0 (1 bit), prec_ref_display_width = 0
        // (1 bit), ref_viewing_distance_flag = 1 (1 bit), then
        // prec_ref_viewing_dist = 32 (11 bits).
        let payload = pack_bits(&[
            (1, 1), // prec_ref_baseline = 0
            (1, 1), // prec_ref_display_width = 0
            (1, 1), // ref_viewing_distance_flag = 1
            (0, 5), // prec_ref_viewing_dist = 32 — codeword start
            (0b100001, 6),
        ]);
        let err = parse_three_dimensional_reference_displays_info(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::ThreeDimensionalReferenceDisplaysInfoPrecOutOfRange {
                field: "prec_ref_viewing_dist",
                got: 32
            }
        ));
    }

    #[test]
    fn three_dimensional_reference_displays_info_rejects_num_ref_displays_above_31() {
        // prec_ref_baseline = 0 (1), prec_ref_display_width = 0 (1),
        // ref_viewing_distance_flag = 0 (1), num_ref_displays_minus1
        // = 32 (11 bits codeword "000001000001").
        let payload = pack_bits(&[(1, 1), (1, 1), (0, 1), (0, 5), (0b100001, 6)]);
        let err = parse_three_dimensional_reference_displays_info(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::ThreeDimensionalReferenceDisplaysInfoNumRefDisplaysOutOfRange(32)
        ));
    }

    #[test]
    fn parse_payload_dispatches_three_dimensional_reference_displays_info() {
        // Minimum-syntax single-display payload routed via dispatcher.
        let ctx = SeiContext::default();
        let payload = pack_bits(&[
            (1, 1),  // prec_ref_baseline = 0
            (1, 1),  // prec_ref_display_width = 0
            (0, 1),  // ref_viewing_distance_flag = 0
            (1, 1),  // num_ref_displays_minus1 = 0
            (31, 6), // exponent_ref_baseline
            (31, 6), // exponent_ref_display_width
            (0, 1),  // additional_shift_present_flag = 0
            (0, 1),  // extension_flag = 0
        ]);
        let got = parse_payload(51, &payload, &ctx).unwrap();
        match got {
            SeiPayload::ThreeDimensionalReferenceDisplaysInfo(info) => {
                assert_eq!(info.displays.len(), 1);
                assert!(!info.extension_flag);
            }
            other => panic!("expected ThreeDimensionalReferenceDisplaysInfo, got {other:?}"),
        }
    }

    // §H.13.2.3 — depth_representation_info, minimum-syntax
    // single-view path with all flags off.
    //
    // all_views_equal_flag = 1  →  numViews = 1, num_views_minus1
    //                              absent.
    // z_near_flag = z_far_flag = 0  →  no z_axis_equal_flag.
    // d_min_flag = d_max_flag = 0   →  no disparity_reference_view.
    // depth_representation_type = 0 →  no nonlinear tail.
    // depth_info_view_id[0] = 0     →  ue codeword "1".
    #[test]
    fn depth_representation_info_minimum_syntax_all_views_equal() {
        let payload = pack_bits(&[
            (1, 1), // all_views_equal_flag = 1
            (0, 1), // z_near_flag = 0
            (0, 1), // z_far_flag = 0
            (0, 1), // d_min_flag = 0
            (0, 1), // d_max_flag = 0
            (1, 1), // depth_representation_type ue(0) = "1"
            (1, 1), // depth_info_view_id[0] ue(0) = "1"
        ]);
        let info = parse_depth_representation_info(&payload).unwrap();
        assert!(info.all_views_equal_flag);
        assert!(!info.z_near_flag);
        assert!(!info.z_far_flag);
        assert!(!info.d_min_flag);
        assert!(!info.d_max_flag);
        assert_eq!(info.z_axis_equal_flag, None);
        assert_eq!(info.common_z_axis_reference_view, None);
        assert_eq!(info.depth_representation_type, 0);
        assert_eq!(info.views.len(), 1);
        let v0 = info.views[0];
        assert_eq!(v0.depth_info_view_id, 0);
        assert!(v0.z_near.is_none());
        assert!(v0.z_far.is_none());
        assert!(v0.d_min.is_none());
        assert!(v0.d_max.is_none());
        assert!(v0.z_axis_reference_view.is_none());
        assert!(v0.disparity_reference_view.is_none());
        assert!(info.depth_nonlinear_representation_num_minus1.is_none());
        assert!(info.depth_nonlinear_representation_model.is_empty());
    }

    // §H.13.2.3 — two views with z_axis_equal_flag = 0 (so
    // z_axis_reference_view[i] is signalled per view), z_near + d_min
    // both signalled. Verifies the float-element wiring and the
    // per-view branch selectors.
    #[test]
    fn depth_representation_info_two_views_per_view_z_axis() {
        // num_views_minus1 = 1 → ue codeword "010" (3 bits).
        // depth_representation_type = 2 → ue codeword "011" (3 bits).
        // Each view: depth_info_view_id = 0 → "1";
        //            z_axis_reference_view = 0 → "1";
        //            disparity_reference_view = 0 → "1";
        //            ZNear float: sign=0, exp=63, mlen_minus1=0, mantissa=0
        //              (1 + 7 + 5 + 1 = 14 bits)
        //            DMin float : sign=1, exp=31, mlen_minus1=0, mantissa=1
        //              (1 + 7 + 5 + 1 = 14 bits)
        let view = [
            (1, 1),  // depth_info_view_id[i] = 0
            (1, 1),  // z_axis_reference_view[i] = 0
            (1, 1),  // disparity_reference_view[i] = 0
            (0, 1),  // ZNear sign
            (63, 7), // ZNear exponent
            (0, 5),  // ZNear mantissa_len_minus1 → width 1
            (0, 1),  // ZNear mantissa
            (1, 1),  // DMin sign (negative)
            (31, 7), // DMin exponent
            (0, 5),  // DMin mantissa_len_minus1 → width 1
            (1, 1),  // DMin mantissa
        ];
        let mut fields: Vec<(u64, u32)> = vec![
            (0, 1),     // all_views_equal_flag = 0
            (0b010, 3), // num_views_minus1 = 1
            (1, 1),     // z_near_flag = 1
            (0, 1),     // z_far_flag = 0
            (0, 1),     // z_axis_equal_flag = 0
            (1, 1),     // d_min_flag = 1
            (0, 1),     // d_max_flag = 0
            (0b011, 3), // depth_representation_type = 2
        ];
        for _ in 0..2 {
            fields.extend_from_slice(&view);
        }
        let payload = pack_bits(&fields);
        let info = parse_depth_representation_info(&payload).unwrap();
        assert!(!info.all_views_equal_flag);
        assert_eq!(info.views.len(), 2);
        assert_eq!(info.z_axis_equal_flag, Some(false));
        assert!(info.common_z_axis_reference_view.is_none());
        assert_eq!(info.depth_representation_type, 2);
        for v in &info.views {
            assert_eq!(v.depth_info_view_id, 0);
            assert_eq!(v.z_axis_reference_view, Some(0));
            assert_eq!(v.disparity_reference_view, Some(0));
            let zn = v.z_near.expect("ZNear present when z_near_flag = 1");
            assert!(!zn.sign);
            assert_eq!(zn.exponent, 63);
            assert_eq!(zn.mantissa_width, 1);
            assert!(v.z_far.is_none());
            let dn = v.d_min.expect("DMin present when d_min_flag = 1");
            assert!(dn.sign);
            assert_eq!(dn.exponent, 31);
            assert_eq!(dn.mantissa_width, 1);
            assert_eq!(dn.mantissa, 1);
            assert!(v.d_max.is_none());
        }
        // Type 2 ≠ 3 — no nonlinear tail.
        assert!(info.depth_nonlinear_representation_num_minus1.is_none());
    }

    // §H.13.2.3 — depth_representation_type == 3 triggers the
    // nonlinear-model tail. num_minus1 = 0 ⇒ one entry signalled.
    #[test]
    fn depth_representation_info_nonlinear_type_3_tail() {
        let payload = pack_bits(&[
            (1, 1),       // all_views_equal_flag = 1
            (1, 1),       // z_near_flag = 1
            (1, 1),       // z_far_flag = 1
            (1, 1),       // z_axis_equal_flag = 1
            (1, 1),       // common_z_axis_reference_view = 0 → "1"
            (0, 1),       // d_min_flag = 0
            (0, 1),       // d_max_flag = 0
            (0b00100, 5), // depth_representation_type = 3
            (1, 1),       // depth_info_view_id[0] = 0
            // ZNear float: sign=0, exp=1, mlen_minus1=0, mantissa=0
            (0, 1),
            (1, 7),
            (0, 5),
            (0, 1),
            // ZFar float: sign=0, exp=2, mlen_minus1=0, mantissa=0
            (0, 1),
            (2, 7),
            (0, 5),
            (0, 1),
            // depth_nonlinear_representation_num_minus1 = 0 → "1"
            (1, 1),
            // depth_nonlinear_representation_model[0] = 4 → "00101"
            (0b00101, 5),
        ]);
        let info = parse_depth_representation_info(&payload).unwrap();
        assert!(info.all_views_equal_flag);
        assert_eq!(info.depth_representation_type, 3);
        assert_eq!(info.z_axis_equal_flag, Some(true));
        assert_eq!(info.common_z_axis_reference_view, Some(0));
        assert_eq!(info.views.len(), 1);
        assert_eq!(info.depth_nonlinear_representation_num_minus1, Some(0));
        assert_eq!(info.depth_nonlinear_representation_model, vec![4]);
    }

    // §H.13.2.3 — typed accessor for the
    // DepthNonlinearRepresentationNumSegments semantic value: when
    // the field is unsignalled (depth_representation_type != 3) the
    // accessor returns None — the §H.13.2.3 piecewise model is not
    // defined for non-3 types so there is no inferred value.
    #[test]
    fn depth_representation_info_num_segments_returns_none_when_absent() {
        let info = DepthRepresentationInfo {
            all_views_equal_flag: true,
            z_near_flag: false,
            z_far_flag: false,
            z_axis_equal_flag: None,
            common_z_axis_reference_view: None,
            d_min_flag: false,
            d_max_flag: false,
            depth_representation_type: 2,
            views: Vec::new(),
            depth_nonlinear_representation_num_minus1: None,
            depth_nonlinear_representation_model: Vec::new(),
        };
        assert_eq!(info.depth_nonlinear_representation_num_segments(), None);
        assert_eq!(info.depth_nonlinear_representation_model_len(), None);
    }

    // §H.13.2.3 — segment-count accessor decodes the `+ 2` bias on
    // a num_minus1 = 0 carrier (the minimum permitted value). The
    // signalled-model-len accessor reports 1 entry for the same
    // carrier.
    #[test]
    fn depth_representation_info_num_segments_minimum() {
        let info = DepthRepresentationInfo {
            all_views_equal_flag: true,
            z_near_flag: false,
            z_far_flag: false,
            z_axis_equal_flag: None,
            common_z_axis_reference_view: None,
            d_min_flag: false,
            d_max_flag: false,
            depth_representation_type: 3,
            views: Vec::new(),
            depth_nonlinear_representation_num_minus1: Some(0),
            depth_nonlinear_representation_model: vec![0],
        };
        assert_eq!(info.depth_nonlinear_representation_num_segments(), Some(2));
        assert_eq!(info.depth_nonlinear_representation_model_len(), Some(1));
    }

    // §H.13.2.3 — segment-count accessor decodes the `+ 2` bias on
    // an interior carrier (num_minus1 = 3 ⇒ 5 segments, 4 signalled
    // model entries).
    #[test]
    fn depth_representation_info_num_segments_interior() {
        let info = DepthRepresentationInfo {
            all_views_equal_flag: true,
            z_near_flag: false,
            z_far_flag: false,
            z_axis_equal_flag: None,
            common_z_axis_reference_view: None,
            d_min_flag: false,
            d_max_flag: false,
            depth_representation_type: 3,
            views: Vec::new(),
            depth_nonlinear_representation_num_minus1: Some(3),
            depth_nonlinear_representation_model: vec![10, 20, 30, 40],
        };
        assert_eq!(info.depth_nonlinear_representation_num_segments(), Some(5));
        assert_eq!(info.depth_nonlinear_representation_model_len(), Some(4));
    }

    // §H.13.2.3 — segment-count accessor decodes the `+ 2` bias at
    // the maximum on-wire value (num_minus1 = 62 ⇒ 64 segments,
    // 63 signalled model entries). Confirms the u8 return type
    // accommodates the upper endpoint without overflow.
    #[test]
    fn depth_representation_info_num_segments_maximum() {
        let info = DepthRepresentationInfo {
            all_views_equal_flag: true,
            z_near_flag: false,
            z_far_flag: false,
            z_axis_equal_flag: None,
            common_z_axis_reference_view: None,
            d_min_flag: false,
            d_max_flag: false,
            depth_representation_type: 3,
            views: Vec::new(),
            depth_nonlinear_representation_num_minus1: Some(62),
            depth_nonlinear_representation_model: vec![0; 63],
        };
        assert_eq!(info.depth_nonlinear_representation_num_segments(), Some(64));
        assert_eq!(info.depth_nonlinear_representation_model_len(), Some(63));
    }

    // §H.13.2.3 — segment count and signalled-model length must be
    // consistent with the stored `depth_nonlinear_representation_model`
    // length: the spec relation
    //
    //   model_len() == depth_nonlinear_representation_model.len()
    //   num_segments() == model_len() + 1
    //
    // holds end-to-end through the parser on the existing
    // type-3 fixture (num_minus1 = 0 ⇒ 1 model entry, 2 segments).
    #[test]
    fn depth_representation_info_num_segments_round_trip_via_parser() {
        let payload = pack_bits(&[
            (1, 1),       // all_views_equal_flag = 1
            (1, 1),       // z_near_flag = 1
            (1, 1),       // z_far_flag = 1
            (1, 1),       // z_axis_equal_flag = 1
            (1, 1),       // common_z_axis_reference_view = 0 → "1"
            (0, 1),       // d_min_flag = 0
            (0, 1),       // d_max_flag = 0
            (0b00100, 5), // depth_representation_type = 3
            (1, 1),       // depth_info_view_id[0] = 0
            // ZNear float: sign=0, exp=1, mlen_minus1=0, mantissa=0
            (0, 1),
            (1, 7),
            (0, 5),
            (0, 1),
            // ZFar float: sign=0, exp=2, mlen_minus1=0, mantissa=0
            (0, 1),
            (2, 7),
            (0, 5),
            (0, 1),
            // depth_nonlinear_representation_num_minus1 = 0 → "1"
            (1, 1),
            // depth_nonlinear_representation_model[0] = 4 → "00101"
            (0b00101, 5),
        ]);
        let info = parse_depth_representation_info(&payload).unwrap();
        let model_len = info
            .depth_nonlinear_representation_model_len()
            .expect("type-3 payload signals model_len");
        let num_segments = info
            .depth_nonlinear_representation_num_segments()
            .expect("type-3 payload signals num_segments");
        assert_eq!(
            usize::from(model_len),
            info.depth_nonlinear_representation_model.len()
        );
        assert_eq!(num_segments, model_len + 1);
        assert_eq!(model_len, 1);
        assert_eq!(num_segments, 2);
    }

    // §H.13.2.3 — values 4..=15 of depth_representation_type are
    // reserved; the parser shall accept them and emit no nonlinear
    // tail.
    #[test]
    fn depth_representation_info_reserved_type_skips_nonlinear_tail() {
        let payload = pack_bits(&[
            (1, 1),       // all_views_equal_flag = 1
            (0, 1),       // z_near_flag = 0
            (0, 1),       // z_far_flag = 0
            (0, 1),       // d_min_flag = 0
            (0, 1),       // d_max_flag = 0
            (0b00101, 5), // depth_representation_type = 4 (reserved)
            (1, 1),       // depth_info_view_id[0] = 0
        ]);
        let info = parse_depth_representation_info(&payload).unwrap();
        assert_eq!(info.depth_representation_type, 4);
        assert!(info.depth_nonlinear_representation_num_minus1.is_none());
    }

    // §H.13.2.3 — num_views_minus1 > 1023 must be rejected before
    // allocation.
    #[test]
    fn depth_representation_info_rejects_num_views_above_1023() {
        // ue codeword for 1024 = "00000000001 0000000001" =
        // 10 zero prefix + 11-bit binary 0b10000000001 = 21 bits.
        // all_views_equal_flag = 0 + that ue codeword.
        // codeNum 1024 = ue codeword: 10 leading zeros + binary
        // representation of (1024 + 1) = 0b10000000001 (11 bits).
        let payload = pack_bits(&[
            (0, 1), // all_views_equal_flag = 0
            (0, 10),
            (0b10000000001, 11),
        ]);
        let err = parse_depth_representation_info(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::DepthRepresentationInfoNumViewsOutOfRange(1024)
        ));
    }

    // §H.13.2.3 — common_z_axis_reference_view > 1023 rejected.
    #[test]
    fn depth_representation_info_rejects_common_z_axis_above_1023() {
        let payload = pack_bits(&[
            (1, 1), // all_views_equal_flag = 1
            (1, 1), // z_near_flag = 1
            (0, 1), // z_far_flag = 0
            (1, 1), // z_axis_equal_flag = 1
            // common_z_axis_reference_view = 1024 → ue codeword as above.
            (0, 10),
            (0b10000000001, 11),
        ]);
        let err = parse_depth_representation_info(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::DepthRepresentationInfoViewIdOutOfRange {
                field: "common_z_axis_reference_view",
                got: 1024
            }
        ));
    }

    // §H.13.2.3 — depth_nonlinear_representation_num_minus1 > 62
    // rejected before allocation.
    #[test]
    fn depth_representation_info_rejects_nonlinear_num_above_62() {
        // Build a minimal type-3 prelude, then a num_minus1 = 63
        // codeword. ue codeword for 63 = "0000010000000" — 6 zeros +
        // 7-bit binary 0b1000000 (= codeNum+1 = 64).
        let payload = pack_bits(&[
            (1, 1),       // all_views_equal_flag = 1
            (0, 1),       // z_near_flag = 0
            (0, 1),       // z_far_flag = 0
            (0, 1),       // d_min_flag = 0
            (0, 1),       // d_max_flag = 0
            (0b00100, 5), // depth_representation_type = 3
            (1, 1),       // depth_info_view_id[0] = 0
            (0, 6),
            (0b1000000, 7),
        ]);
        let err = parse_depth_representation_info(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::DepthRepresentationInfoNonlinearNumOutOfRange(63)
        ));
    }

    // §H.13.1.3.1 / Table H-2 — verify the DepthFloatComponent::to_f64
    // arithmetic.
    #[test]
    fn depth_float_component_to_f64_branches() {
        // Normal with e = 31, v = 1, n = 0 → 2^0 * 1 = 1.0
        let one = DepthFloatComponent {
            sign: false,
            exponent: 31,
            mantissa: 0,
            mantissa_width: 1,
        };
        assert_eq!(one.to_f64(), 1.0);
        // Normal with e = 32, v = 1, n = 1 → 2^1 * (1 + 0.5) = 3.0
        let three = DepthFloatComponent {
            sign: false,
            exponent: 32,
            mantissa: 1,
            mantissa_width: 1,
        };
        assert_eq!(three.to_f64(), 3.0);
        // Negative-sign sentinel: same magnitude, opposite sign.
        let minus_three = DepthFloatComponent {
            sign: true,
            exponent: 32,
            mantissa: 1,
            mantissa_width: 1,
        };
        assert_eq!(minus_three.to_f64(), -3.0);
        // Denormal e = 0, n = 1, v = 1 → 2^-30 * (1/2) = 2^-31
        let dn = DepthFloatComponent {
            sign: false,
            exponent: 0,
            mantissa: 1,
            mantissa_width: 1,
        };
        assert!((dn.to_f64() - 2f64.powi(-31)).abs() < 1e-300);
        // Reserved e = 127 → NaN.
        let res = DepthFloatComponent {
            sign: false,
            exponent: 127,
            mantissa: 0,
            mantissa_width: 1,
        };
        assert!(res.to_f64().is_nan());
    }

    // §H.13.2.3 — dispatch through the public parse_payload entry
    // point. Same fixture as the minimum-syntax test above; we just
    // verify the match arm selects the right variant.
    #[test]
    fn parse_payload_dispatches_depth_representation_info() {
        let ctx = SeiContext::default();
        let payload = pack_bits(&[
            (1, 1), // all_views_equal_flag = 1
            (0, 1), // z_near_flag = 0
            (0, 1), // z_far_flag = 0
            (0, 1), // d_min_flag = 0
            (0, 1), // d_max_flag = 0
            (1, 1), // depth_representation_type = 0
            (1, 1), // depth_info_view_id[0] = 0
        ]);
        let got = parse_payload(50, &payload, &ctx).unwrap();
        match got {
            SeiPayload::DepthRepresentationInfo(info) => {
                assert_eq!(info.views.len(), 1);
                assert_eq!(info.depth_representation_type, 0);
            }
            other => panic!("expected DepthRepresentationInfo, got {other:?}"),
        }
    }

    // ==============================================================
    // §I.13.2.1 — constrained_depth_parameter_set_identifier (payload
    // type 54, Annex I 3D-AVC depth coding). Round 237.
    // ==============================================================

    // Smallest legal fixture: max_dps_id = 1 (ue codeword "010"),
    // max_dps_id_diff = 0 (ue codeword "1"). 0 * 2 < 1 holds.
    #[test]
    fn constrained_depth_parameter_set_identifier_min_legal() {
        let payload = pack_bits(&[
            (0b010, 3), // ue(1) = 1 → max_dps_id
            (1, 1),     // ue(0) = 0 → max_dps_id_diff
        ]);
        let got = parse_constrained_depth_parameter_set_identifier(&payload).unwrap();
        assert_eq!(got.max_dps_id, 1);
        assert_eq!(got.max_dps_id_diff, 0);
    }

    // Mid-range fixture: max_dps_id = 7 (ue codeword "0001000"),
    // max_dps_id_diff = 3 (ue codeword "00100"). 3 * 2 = 6 < 7 holds.
    #[test]
    fn constrained_depth_parameter_set_identifier_mid_range() {
        let payload = pack_bits(&[
            (0b0001000, 7), // ue(7) = 7 → max_dps_id
            (0b00100, 5),   // ue(3) = 3 → max_dps_id_diff
        ]);
        let got = parse_constrained_depth_parameter_set_identifier(&payload).unwrap();
        assert_eq!(got.max_dps_id, 7);
        assert_eq!(got.max_dps_id_diff, 3);
    }

    // Maximum legal fixture: max_dps_id = 62, max_dps_id_diff = 30.
    // 30 * 2 = 60 < 62 holds. ue(62) = leadingZeros 5, suffix = 62 -
    // (2^5 - 1) = 31 in 5 bits → "000001" + "11111" = "00000111111"
    // (11 bits). ue(30) = leadingZeros 4, suffix = 30 - (2^4 - 1) =
    // 15 in 4 bits → "00001" + "1111" = "000011111" (9 bits).
    #[test]
    fn constrained_depth_parameter_set_identifier_max_legal() {
        let payload = pack_bits(&[
            (0b00000111111, 11), // ue(62)
            (0b000011111, 9),    // ue(30)
        ]);
        let got = parse_constrained_depth_parameter_set_identifier(&payload).unwrap();
        assert_eq!(got.max_dps_id, 62);
        assert_eq!(got.max_dps_id_diff, 30);
    }

    // max_dps_id = 63 (ue codeword "0000001000000") is out of range:
    // the §I.13.2.1 derivation `max_dps_id + 1 ≤ 63` requires
    // max_dps_id ≤ 62 (depth_parameter_set_id range 1..=63 per
    // §7.4.2.16). ue(63): leadingZeros 6, suffix = 63 - (2^6 - 1) = 0
    // in 6 bits → "0000001" + "000000" = "0000001000000" (13 bits).
    #[test]
    fn constrained_depth_parameter_set_identifier_rejects_max_dps_id_above_62() {
        let payload = pack_bits(&[(0b0000001000000, 13)]);
        let err = parse_constrained_depth_parameter_set_identifier(&payload).unwrap_err();
        assert_eq!(
            err,
            SeiError::ConstrainedDepthParameterSetIdentifierMaxDpsIdOutOfRange(63)
        );
    }

    // max_dps_id = 4, max_dps_id_diff = 2 → 2 * 2 = 4 NOT < 4.
    // Violates the §I.13.2.1 sliding-window-integrity constraint.
    #[test]
    fn constrained_depth_parameter_set_identifier_rejects_diff_equals_half_max() {
        let payload = pack_bits(&[
            (0b00101, 5), // ue(4) = 4 → max_dps_id
            (0b011, 3),   // ue(2) = 2 → max_dps_id_diff
        ]);
        let err = parse_constrained_depth_parameter_set_identifier(&payload).unwrap_err();
        assert_eq!(
            err,
            SeiError::ConstrainedDepthParameterSetIdentifierDiffViolatesBound {
                max_dps_id_diff: 2,
                max_dps_id: 4,
            }
        );
    }

    // max_dps_id = 0, max_dps_id_diff = 0 → 0 * 2 = 0 NOT < 0.
    // Edge case: the spec's strict-less-than relation rules out
    // even a "no diff" pair when max_dps_id is also zero.
    #[test]
    fn constrained_depth_parameter_set_identifier_rejects_zero_pair() {
        let payload = pack_bits(&[
            (1, 1), // ue(0) = 0 → max_dps_id
            (1, 1), // ue(0) = 0 → max_dps_id_diff
        ]);
        let err = parse_constrained_depth_parameter_set_identifier(&payload).unwrap_err();
        assert_eq!(
            err,
            SeiError::ConstrainedDepthParameterSetIdentifierDiffViolatesBound {
                max_dps_id_diff: 0,
                max_dps_id: 0,
            }
        );
    }

    // §I.13.2.1 — dispatch through the public parse_payload entry
    // point. Same fixture as the min-legal test above; the match arm
    // selects the right variant.
    #[test]
    fn parse_payload_dispatches_constrained_depth_parameter_set_identifier() {
        let ctx = SeiContext::default();
        let payload = pack_bits(&[
            (0b010, 3), // ue(1) = 1 → max_dps_id
            (1, 1),     // ue(0) = 0 → max_dps_id_diff
        ]);
        let got = parse_payload(54, &payload, &ctx).unwrap();
        match got {
            SeiPayload::ConstrainedDepthParameterSetIdentifier(info) => {
                assert_eq!(info.max_dps_id, 1);
                assert_eq!(info.max_dps_id_diff, 0);
            }
            other => panic!("expected ConstrainedDepthParameterSetIdentifier, got {other:?}"),
        }
    }

    // §H.13.2.7 — minimal depth_sampling_info: shared
    // depth_grid_position() (per_view_depth_grid_pos_flag == 0).
    // dttsr_{x,y}_mul = 1 / dp = 0 means a 1:1 sample-size ratio.
    // The shared depth_grid_position() carries a zero position with
    // both fractional widths set to 0.
    #[test]
    fn depth_sampling_info_min_shared_position() {
        let payload = pack_bits(&[
            (1, 16), // dttsr_x_mul = 1
            (0, 4),  // dttsr_x_dp = 0
            (1, 16), // dttsr_y_mul = 1
            (0, 4),  // dttsr_y_dp = 0
            (0, 1),  // per_view_depth_grid_pos_flag = 0
            (0, 20), // depth_grid_pos_x_fp = 0
            (0, 4),  // depth_grid_pos_x_dp = 0
            (0, 1),  // depth_grid_pos_x_sign_flag = 0
            (0, 20), // depth_grid_pos_y_fp = 0
            (0, 4),  // depth_grid_pos_y_dp = 0
            (0, 1),  // depth_grid_pos_y_sign_flag = 0
        ]);
        let info = parse_depth_sampling_info(&payload).unwrap();
        assert_eq!(info.dttsr_x_mul, 1);
        assert_eq!(info.dttsr_x_dp, 0);
        assert_eq!(info.dttsr_y_mul, 1);
        assert_eq!(info.dttsr_y_dp, 0);
        assert!(!info.per_view_depth_grid_pos_flag);
        assert!(info.num_video_plus_depth_views_minus1.is_none());
        assert_eq!(info.views.len(), 1);
        assert_eq!(info.views[0].depth_grid_view_id, 0);
        assert_eq!(info.views[0].position.pos_x_fp, 0);
        assert_eq!(info.views[0].position.pos_y_fp, 0);
        assert!((info.dttsr_x_to_f64() - 1.0).abs() < 1e-9);
        assert!((info.dttsr_y_to_f64() - 1.0).abs() < 1e-9);
        assert!((info.views[0].position.x_to_f64() - 0.0).abs() < 1e-9);
    }

    // §H.13.2.7 — per-view depth_sampling_info with two views. Each
    // view's depth_grid_position() carries a distinct fp / dp /
    // sign-bit triple so we can confirm the per-view storage isn't
    // accidentally shared.
    #[test]
    fn depth_sampling_info_per_view_two_views() {
        let payload = pack_bits(&[
            (3, 16),     // dttsr_x_mul = 3
            (0b0001, 4), // dttsr_x_dp = 1
            (3, 16),     // dttsr_y_mul = 3
            (0b0001, 4), // dttsr_y_dp = 1
            (1, 1),      // per_view_depth_grid_pos_flag = 1
            (0b010, 3),  // ue(1) = 1 → num_video_plus_depth_views_minus1
            // view 0: depth_grid_view_id = 0 (ue "1")
            (1, 1),
            (0b0000_0000_0000_0001_0000u32 as u64, 20), // pos_x_fp = 16
            (0b0011, 4),                                // pos_x_dp = 3
            (0, 1),                                     // pos_x_sign = 0
            (0b0000_0000_0000_0000_1000u32 as u64, 20), // pos_y_fp = 8
            (0b0010, 4),                                // pos_y_dp = 2
            (1, 1),                                     // pos_y_sign = 1
            // view 1: depth_grid_view_id = 5 (ue codenum 5 = "00110")
            (0b00110, 5),
            (0b0000_0000_0000_0010_0000u32 as u64, 20), // pos_x_fp = 32
            (0b0100, 4),                                // pos_x_dp = 4
            (1, 1),                                     // pos_x_sign = 1
            (0b0000_0000_0000_0001_0000u32 as u64, 20), // pos_y_fp = 16
            (0b0011, 4),                                // pos_y_dp = 3
            (0, 1),                                     // pos_y_sign = 0
        ]);
        let info = parse_depth_sampling_info(&payload).unwrap();
        assert!(info.per_view_depth_grid_pos_flag);
        assert_eq!(info.num_video_plus_depth_views_minus1, Some(1));
        assert_eq!(info.views.len(), 2);
        assert_eq!(info.views[0].depth_grid_view_id, 0);
        assert_eq!(info.views[0].position.pos_x_fp, 16);
        assert_eq!(info.views[0].position.pos_x_dp, 3);
        assert!(!info.views[0].position.pos_x_sign);
        assert_eq!(info.views[0].position.pos_y_fp, 8);
        assert!(info.views[0].position.pos_y_sign);
        assert_eq!(info.views[1].depth_grid_view_id, 5);
        assert_eq!(info.views[1].position.pos_x_fp, 32);
        assert!(info.views[1].position.pos_x_sign);
        assert_eq!(info.views[1].position.pos_y_dp, 3);
        // Reconstructed scalar: view 0 y = -(8/4) = -2.0
        assert!((info.views[0].position.y_to_f64() + 2.0).abs() < 1e-9);
        // view 1 x = -(32/16) = -2.0
        assert!((info.views[1].position.x_to_f64() + 2.0).abs() < 1e-9);
        // dttsr_x = 3/2 = 1.5
        assert!((info.dttsr_x_to_f64() - 1.5).abs() < 1e-9);
    }

    // §H.13.2.7 — dttsr_x_mul = 0 is reserved; reject.
    #[test]
    fn depth_sampling_info_rejects_dttsr_x_mul_zero() {
        let payload = pack_bits(&[
            (0, 16), // dttsr_x_mul = 0 (reserved)
            (0, 4),
            (1, 16),
            (0, 4),
            (0, 1),
        ]);
        let err = parse_depth_sampling_info(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::DepthSamplingInfoDttsrMulReserved { axis: "x" }
        ));
    }

    // §H.13.2.7 — dttsr_y_mul = 0 is reserved; reject.
    #[test]
    fn depth_sampling_info_rejects_dttsr_y_mul_zero() {
        let payload = pack_bits(&[
            (1, 16), // dttsr_x_mul = 1
            (0, 4),
            (0, 16), // dttsr_y_mul = 0 (reserved)
            (0, 4),
            (0, 1),
        ]);
        let err = parse_depth_sampling_info(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::DepthSamplingInfoDttsrMulReserved { axis: "y" }
        ));
    }

    // §H.13.2.7 — num_video_plus_depth_views_minus1 > 1023 rejected
    // before allocation.
    #[test]
    fn depth_sampling_info_rejects_num_views_above_1023() {
        // ue codeNum 1024 = 10 leading zeros + 11-bit binary
        // (1024 + 1) = 0b10000000001.
        let payload = pack_bits(&[
            (1, 16), // dttsr_x_mul = 1
            (0, 4),
            (1, 16), // dttsr_y_mul = 1
            (0, 4),
            (1, 1), // per_view_depth_grid_pos_flag = 1
            (0, 10),
            (0b10000000001, 11),
        ]);
        let err = parse_depth_sampling_info(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::DepthSamplingInfoNumViewsOutOfRange(1024)
        ));
    }

    // §H.13.2.7 — depth_grid_view_id[i] > 1023 rejected; the i index
    // is preserved in the error so a regression points at the failed
    // entry without re-running the parser.
    #[test]
    fn depth_sampling_info_rejects_view_id_above_1023() {
        let payload = pack_bits(&[
            (1, 16), // dttsr_x_mul = 1
            (0, 4),
            (1, 16), // dttsr_y_mul = 1
            (0, 4),
            (1, 1),     // per_view_depth_grid_pos_flag = 1
            (0b010, 3), // ue(1) = 1 → num_views_minus1 = 1
            // view 0: view_id = 1024 (out of range).
            (0, 10),
            (0b10000000001, 11),
        ]);
        let err = parse_depth_sampling_info(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::DepthSamplingInfoViewIdOutOfRange { i: 0, got: 1024 }
        ));
    }

    // §H.13.1.7 dispatch — parse_payload(53, ..) must route to
    // parse_depth_sampling_info.
    #[test]
    fn parse_payload_dispatches_depth_sampling_info() {
        let ctx = SeiContext::default();
        let payload = pack_bits(&[
            (2, 16), // dttsr_x_mul = 2
            (0, 4),
            (2, 16), // dttsr_y_mul = 2
            (0, 4),
            (0, 1),  // per_view_depth_grid_pos_flag = 0
            (0, 20), // depth_grid_pos_x_fp
            (0, 4),
            (0, 1),
            (0, 20), // depth_grid_pos_y_fp
            (0, 4),
            (0, 1),
        ]);
        let got = parse_payload(53, &payload, &ctx).unwrap();
        match got {
            SeiPayload::DepthSamplingInfo(info) => {
                assert_eq!(info.dttsr_x_mul, 2);
                assert_eq!(info.dttsr_y_mul, 2);
                assert!(!info.per_view_depth_grid_pos_flag);
                assert_eq!(info.views.len(), 1);
            }
            other => panic!("expected DepthSamplingInfo, got {other:?}"),
        }
    }

    // §H.13.1.6 — alternative_depth_info with depth_type == 0 and all
    // four GVD presence flags clear: the message reduces to depth_type +
    // num_constituent_views_gvd_minus1 + the five flags, then one empty
    // per-camera entry per camera (no intrinsic / rotation / translation
    // sub-blocks, no z block). num_constituent_views_gvd_minus1 = 1 →
    // 1 + 2 = 3 cameras.
    #[test]
    fn alternative_depth_info_minimal_flags_clear() {
        let payload = pack_bits(&[
            (0b1, 1),   // depth_type ue(0) = 0
            (0b010, 3), // num_constituent_views_gvd_minus1 ue(1) = 1
            (0, 1),     // depth_present_gvd_flag = 0
            (0, 1),     // z_gvd_flag = 0
            (0, 1),     // intrinsic_param_gvd_flag = 0
            (0, 1),     // rotation_gvd_flag = 0
            (0, 1),     // translation_gvd_flag = 0
        ]);
        let info = parse_alternative_depth_info(&payload).unwrap();
        assert_eq!(info.depth_type, 0);
        let gvd = info.gvd.expect("depth_type 0 carries a gvd block");
        assert_eq!(gvd.num_constituent_views_gvd_minus1, 1);
        assert!(!gvd.depth_present_gvd_flag);
        assert!(!gvd.z_gvd_flag);
        assert!(gvd.z_values.is_empty());
        assert_eq!(gvd.prec_gvd_focal_length, None);
        assert_eq!(gvd.prec_gvd_rotation_param, None);
        // 1 + 2 = 3 cameras, each with no sub-blocks.
        assert_eq!(gvd.cameras.len(), 3);
        for cam in &gvd.cameras {
            assert!(cam.intrinsic.is_none());
            assert!(cam.rotation.is_none());
            assert!(cam.translation_x.is_none());
        }
    }

    // §H.13.2.6 — a non-zero depth_type is reserved; the decoder records
    // the type and ignores the body (no error, gvd == None).
    #[test]
    fn alternative_depth_info_nonzero_type_ignored() {
        // depth_type = 2 → ue(2) = 0b011 (3 bits). Trailing garbage is
        // not read because the body is skipped.
        let payload = pack_bits(&[(0b011, 3), (0xFF, 8)]);
        let info = parse_alternative_depth_info(&payload).unwrap();
        assert_eq!(info.depth_type, 2);
        assert!(info.gvd.is_none());
    }

    // §H.13.2.6 — num_constituent_views_gvd_minus1 must be in 0..=3.
    #[test]
    fn alternative_depth_info_num_views_out_of_range() {
        // depth_type = 0, num_constituent_views_gvd_minus1 = 4 →
        // ue(4) = 0b00101 (5 bits). 4 > 3 must error.
        let payload = pack_bits(&[(0b1, 1), (0b00101, 5)]);
        let err = parse_alternative_depth_info(&payload).unwrap_err();
        assert!(matches!(
            err,
            SeiError::AlternativeDepthInfoNumConstituentViewsOutOfRange(4)
        ));
    }

    // §H.13.2.6 — z_gvd_flag set: a near/far DepthFloatComponent pair per
    // camera (sign u(1), exp u(7), man_len_minus1 u(5), man u(v)). One
    // constituent view (minus1 = 0 → 2 cameras). exp == 0 with a small
    // mantissa exercises the denormal eq. H-1 path; the second camera
    // uses exp == 127 (reserved → NaN).
    #[test]
    fn alternative_depth_info_z_block_two_cameras() {
        let payload = pack_bits(&[
            (0b1, 1), // depth_type = 0
            (0b1, 1), // num_constituent_views_gvd_minus1 ue(0) = 0 → 2 cams
            (0, 1),   // depth_present_gvd_flag = 0
            (1, 1),   // z_gvd_flag = 1
            (0, 1),   // intrinsic_param_gvd_flag = 0
            (0, 1),   // rotation_gvd_flag = 0
            (0, 1),   // translation_gvd_flag = 0
            // camera 0 z_near: sign 0, exp 0 (denormal), man_len_minus1 3
            // (→ width 4), mantissa 0b1000 = 8.
            (0, 1),
            (0, 7),
            (3, 5),
            (0b1000, 4),
            // camera 0 z_far: sign 1, exp 64 (normal), man_len_minus1 0
            // (→ width 1), mantissa 1.
            (1, 1),
            (64, 7),
            (0, 5),
            (1, 1),
            // camera 1 z_near: sign 0, exp 127 (reserved), man_len_minus1
            // 0 (→ width 1), mantissa 0.
            (0, 1),
            (127, 7),
            (0, 5),
            (0, 1),
            // camera 1 z_far: sign 0, exp 1, man_len_minus1 1 (width 2),
            // mantissa 0b10 = 2.
            (0, 1),
            (1, 7),
            (1, 5),
            (0b10, 2),
        ]);
        let info = parse_alternative_depth_info(&payload).unwrap();
        let gvd = info.gvd.unwrap();
        assert!(gvd.z_gvd_flag);
        assert_eq!(gvd.z_values.len(), 2);

        let z0 = gvd.z_values[0];
        // z_near denormal: (-1)^0 * 2^-30 * (8 / 2^4) = 2^-30 * 0.5.
        assert!((z0.z_near.to_f64() - 2f64.powi(-30) * 0.5).abs() < 1e-18);
        // z_far normal: (-1)^1 * 2^(64-31) * (1 + 1/2^1) = -2^33 * 1.5.
        assert!((z0.z_far.to_f64() - (-(2f64.powi(33)) * 1.5)).abs() < 1.0);

        let z1 = gvd.z_values[1];
        // z_near exp == 127 → unspecified (NaN).
        assert!(z1.z_near.to_f64().is_nan());
        // z_far normal: 2^(1-31) * (1 + 2/2^2) = 2^-30 * 1.5.
        assert!((z1.z_far.to_f64() - 2f64.powi(-30) * 1.5).abs() < 1e-18);
    }

    // §H.13.2.6 — intrinsic + rotation + translation blocks for a single
    // camera (num_constituent_views_gvd_minus1 = 0 → 2 cameras). Uses
    // the §G.13.2.5 FloatComponent layout (exp u(6), reserved 63,
    // prec-derived mantissa width). prec = 31 with exp = 31 gives a
    // mantissa width of 31 + 31 - 31 = 31 bits.
    #[test]
    fn alternative_depth_info_intrinsic_rotation_translation() {
        // Build the per-camera float as sign u(1) + exp u(6) +
        // man u(width). With prec = 0 and exp in 1..=31, width =
        // max(0, exp + 0 - 31) = 0 for exp <= 31, so no mantissa bits.
        // That keeps the test compact: every float is just sign + exp.
        let mut fields: Vec<(u64, u32)> = vec![
            (0b1, 1), // depth_type = 0
            (0b1, 1), // num_constituent_views_gvd_minus1 ue(0) = 0 → 2 cams
            (0, 1),   // depth_present_gvd_flag = 0
            (0, 1),   // z_gvd_flag = 0
            (1, 1),   // intrinsic_param_gvd_flag = 1
            (1, 1),   // rotation_gvd_flag = 1
            (1, 1),   // translation_gvd_flag = 1
            // prec_gvd_focal_length ue(0) = 0
            (0b1, 1),
            // prec_gvd_principal_point ue(0) = 0
            (0b1, 1),
            // prec_gvd_rotation_param ue(0) = 0
            (0b1, 1),
            // prec_gvd_translation_param ue(0) = 0
            (0b1, 1),
        ];
        // For each of the 2 cameras: 4 intrinsic floats + 9 rotation
        // floats + 1 translation float = 14 floats, each sign u(1) +
        // exp u(6), no mantissa (width 0 because prec = 0, exp <= 62
        // small).
        for cam in 0..2u64 {
            for f in 0..14u64 {
                let exp = (cam * 14 + f) % 30 + 1; // 1..=30
                fields.push((0, 1)); // sign
                fields.push((exp, 6)); // exp u(6)
            }
        }
        let payload = pack_bits(&fields);
        let info = parse_alternative_depth_info(&payload).unwrap();
        let gvd = info.gvd.unwrap();
        assert_eq!(gvd.prec_gvd_focal_length, Some(0));
        assert_eq!(gvd.prec_gvd_rotation_param, Some(0));
        assert_eq!(gvd.prec_gvd_translation_param, Some(0));
        assert_eq!(gvd.cameras.len(), 2);
        for cam in &gvd.cameras {
            let intr = cam.intrinsic.expect("intrinsic present");
            // exp values were 1..=30, width 0 → mantissa 0, normal float.
            assert_eq!(intr.focal_length_x.mantissa_width, 0);
            let rot = cam.rotation.expect("rotation present");
            assert_eq!(rot.len(), 3);
            assert_eq!(rot[0].len(), 3);
            assert!(cam.translation_x.is_some());
        }
    }

    // §H.13.1.6 dispatch — parse_payload(181, ..) routes to
    // parse_alternative_depth_info.
    #[test]
    fn parse_payload_dispatches_alternative_depth_info() {
        let ctx = SeiContext::default();
        let payload = pack_bits(&[
            (0b1, 1), // depth_type = 0
            (0b1, 1), // num_constituent_views_gvd_minus1 = 0
            (0, 1),   // depth_present_gvd_flag
            (0, 1),   // z_gvd_flag
            (0, 1),   // intrinsic_param_gvd_flag
            (0, 1),   // rotation_gvd_flag
            (0, 1),   // translation_gvd_flag
        ]);
        let got = parse_payload(181, &payload, &ctx).unwrap();
        match got {
            SeiPayload::AlternativeDepthInfo(info) => {
                assert_eq!(info.depth_type, 0);
                let gvd = info.gvd.unwrap();
                assert_eq!(gvd.cameras.len(), 2);
            }
            other => panic!("expected AlternativeDepthInfo, got {other:?}"),
        }
    }

    // §H.13.2.5 — minimal depth_timing: per_view_depth_timing_flag == 0
    // so a single shared depth_timing_offset() follows. offset_len_minus1
    // = 7 makes the fp field u(8); fp = 64 with dp = 6 reconstructs to
    // 64 / 2^6 = 1.0 clock ticks.
    #[test]
    fn depth_timing_shared_single_offset() {
        let payload = pack_bits(&[
            (0, 1),  // per_view_depth_timing_flag = 0
            (7, 5),  // offset_len_minus1 = 7 → fp is u(8)
            (64, 8), // depth_disp_delay_offset_fp = 64
            (6, 6),  // depth_disp_delay_offset_dp = 6
        ]);
        // The shared branch never reads ctx.num_depth_views, so the
        // default (0 = unknown) context must succeed.
        let got = parse_depth_timing(&payload, &SeiContext::default()).unwrap();
        assert!(!got.per_view_depth_timing_flag);
        assert_eq!(got.offsets.len(), 1);
        let off = got.offsets[0];
        assert_eq!(off.offset_len_minus1, 7);
        assert_eq!(off.depth_disp_delay_offset_fp, 64);
        assert_eq!(off.depth_disp_delay_offset_dp, 6);
        assert!((off.offset_clock_ticks() - 1.0).abs() < 1e-12);
    }

    // §H.13.2.5 — per-view depth_timing with NumDepthViews = 2 supplied
    // through the context. The two offsets use the extreme fp widths:
    // view 0 the minimum (offset_len_minus1 = 0 → u(1)), view 1 the
    // maximum (offset_len_minus1 = 31 → u(32), all bits set).
    #[test]
    fn depth_timing_per_view_two_views() {
        let payload = pack_bits(&[
            (1, 1), // per_view_depth_timing_flag = 1
            // view 0: offset_len_minus1 = 0 → fp u(1) = 1, dp = 0
            // → 1 / 2^0 = 1.0 clock ticks.
            (0, 5),
            (1, 1),
            (0, 6),
            // view 1: offset_len_minus1 = 31 → fp u(32) = u32::MAX,
            // dp = 32 → (2^32 − 1) / 2^32 just under 1.0.
            (31, 5),
            (u32::MAX as u64, 32),
            (32, 6),
        ]);
        let ctx = SeiContext {
            num_depth_views: 2,
            ..SeiContext::default()
        };
        let got = parse_depth_timing(&payload, &ctx).unwrap();
        assert!(got.per_view_depth_timing_flag);
        assert_eq!(got.offsets.len(), 2);
        assert_eq!(got.offsets[0].offset_len_minus1, 0);
        assert_eq!(got.offsets[0].depth_disp_delay_offset_fp, 1);
        assert_eq!(got.offsets[0].depth_disp_delay_offset_dp, 0);
        assert!((got.offsets[0].offset_clock_ticks() - 1.0).abs() < 1e-12);
        assert_eq!(got.offsets[1].offset_len_minus1, 31);
        assert_eq!(got.offsets[1].depth_disp_delay_offset_fp, u32::MAX);
        assert_eq!(got.offsets[1].depth_disp_delay_offset_dp, 32);
        let expected = f64::from(u32::MAX) / 2f64.powi(32);
        assert!((got.offsets[1].offset_clock_ticks() - expected).abs() < 1e-12);
    }

    // §H.13.1.5 — the per-view loop bound is the SPS-derived
    // NumDepthViews (§H.7.3.2.1.5), not a payload field. Reaching the
    // per-view branch with the context still at the unknown default
    // (0) must be rejected, mirroring the §D.1.10 spare_pic treatment
    // of an unknown PicSizeInMapUnits.
    #[test]
    fn depth_timing_per_view_unknown_num_depth_views_rejected() {
        let payload = pack_bits(&[
            (1, 1), // per_view_depth_timing_flag = 1
            (0, 5),
            (1, 1),
            (0, 6),
        ]);
        let err = parse_depth_timing(&payload, &SeiContext::default()).unwrap_err();
        assert!(matches!(err, SeiError::DepthTimingNumDepthViewsUnknown));
    }

    // §H.7.3.2.1.5 — the MVCD-extension view loop runs at most
    // num_views_minus1 + 1 ≤ 1024 times and increments NumDepthViews
    // at most once per iteration, so 1024 is the absolute ceiling. A
    // caller-supplied context above it is rejected before allocation.
    #[test]
    fn depth_timing_rejects_num_depth_views_above_1024() {
        let payload = pack_bits(&[
            (1, 1), // per_view_depth_timing_flag = 1
            (0, 5),
            (1, 1),
            (0, 6),
        ]);
        let ctx = SeiContext {
            num_depth_views: 1025,
            ..SeiContext::default()
        };
        let err = parse_depth_timing(&payload, &ctx).unwrap_err();
        assert!(matches!(
            err,
            SeiError::DepthTimingNumDepthViewsOutOfRange(1025)
        ));
    }

    // §H.13.2.5.1 — fractional reconstruction: fp = 3 with dp = 1 is
    // 1.5 clock ticks; the dp ceiling (u(6) max = 63) divides by 2^63
    // without overflowing the intermediate.
    #[test]
    fn depth_timing_offset_clock_ticks_fractional_and_dp_ceiling() {
        let half_step = DepthTimingOffset {
            offset_len_minus1: 1,
            depth_disp_delay_offset_fp: 3,
            depth_disp_delay_offset_dp: 1,
        };
        assert!((half_step.offset_clock_ticks() - 1.5).abs() < 1e-12);

        let dp_ceiling = DepthTimingOffset {
            offset_len_minus1: 0,
            depth_disp_delay_offset_fp: 1,
            depth_disp_delay_offset_dp: 63,
        };
        let expected = 1.0 / 2f64.powi(63);
        assert!((dp_ceiling.offset_clock_ticks() - expected).abs() < 1e-300);
        assert!(dp_ceiling.offset_clock_ticks() > 0.0);
    }

    // §H.13.1.5.1 — a payload that ends inside the offset triple is a
    // bitstream error, not a panic. Five bits of offset_len_minus1 = 31
    // promise a 32-bit fp that the 1-byte payload cannot deliver.
    #[test]
    fn depth_timing_truncated_offset_rejected() {
        let payload = pack_bits(&[
            (0, 1),  // per_view_depth_timing_flag = 0
            (31, 5), // offset_len_minus1 = 31 → fp u(32) expected
        ]);
        let err = parse_depth_timing(&payload, &SeiContext::default()).unwrap_err();
        assert!(matches!(err, SeiError::Bitstream(_)));
    }

    // §H.13.1.5 dispatch — parse_payload(52, ..) must route to
    // parse_depth_timing and thread the context through.
    #[test]
    fn parse_payload_dispatches_depth_timing() {
        let payload = pack_bits(&[
            (1, 1), // per_view_depth_timing_flag = 1
            // view 0: fp u(3) = 5, dp = 2 → 1.25 clock ticks.
            (2, 5),
            (5, 3),
            (2, 6),
        ]);
        let ctx = SeiContext {
            num_depth_views: 1,
            ..SeiContext::default()
        };
        let got = parse_payload(52, &payload, &ctx).unwrap();
        match got {
            SeiPayload::DepthTiming(dt) => {
                assert!(dt.per_view_depth_timing_flag);
                assert_eq!(dt.offsets.len(), 1);
                assert_eq!(dt.offsets[0].depth_disp_delay_offset_fp, 5);
                assert!((dt.offsets[0].offset_clock_ticks() - 1.25).abs() < 1e-12);
            }
            other => panic!("expected DepthTiming, got {other:?}"),
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

    // --- typed accessors on SubSeqLayerCharacteristic -----------------
    //
    // §D.2.13 (2024) / §D.2.12 (2003 draft) carrier semantics:
    // * average_bit_rate is in units of 1000 bits/second; 0 ⇒ unspecified.
    // * average_frame_rate is in units of frames/(256 seconds);
    //   0 ⇒ unspecified.

    #[test]
    fn sub_seq_layer_avg_bit_rate_bps_zero_is_none() {
        let s = SubSeqLayerCharacteristic {
            accurate_statistics_flag: false,
            average_bit_rate: 0,
            average_frame_rate: 0,
        };
        assert_eq!(s.average_bit_rate_bps(), None);
        assert_eq!(s.average_frame_rate_fps(), None);
    }

    #[test]
    fn sub_seq_layer_avg_bit_rate_bps_minimum_nonzero() {
        // Smallest valid signalled value: on-wire 1 ⇒ 1000 bits/second.
        let s = SubSeqLayerCharacteristic {
            accurate_statistics_flag: true,
            average_bit_rate: 1,
            average_frame_rate: 0,
        };
        assert_eq!(s.average_bit_rate_bps(), Some(1_000));
    }

    #[test]
    fn sub_seq_layer_avg_bit_rate_bps_typical() {
        // 5_000 * 1000 = 5_000_000 bits/second (5 Mbit/s).
        let s = SubSeqLayerCharacteristic {
            accurate_statistics_flag: true,
            average_bit_rate: 5_000,
            average_frame_rate: 0,
        };
        assert_eq!(s.average_bit_rate_bps(), Some(5_000_000));
    }

    #[test]
    fn sub_seq_layer_avg_bit_rate_bps_maximum() {
        // On-wire u(16) ceiling: 65535 * 1000 = 65_535_000 bits/second.
        let s = SubSeqLayerCharacteristic {
            accurate_statistics_flag: true,
            average_bit_rate: u16::MAX,
            average_frame_rate: 0,
        };
        assert_eq!(s.average_bit_rate_bps(), Some(65_535_000));
    }

    #[test]
    fn sub_seq_layer_avg_frame_rate_fps_quarter_step() {
        // 1/256 ⇒ smallest representable step.
        let s = SubSeqLayerCharacteristic {
            accurate_statistics_flag: true,
            average_bit_rate: 0,
            average_frame_rate: 1,
        };
        let fps = s.average_frame_rate_fps().expect("non-zero ⇒ Some");
        assert!((fps - (1.0 / 256.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn sub_seq_layer_avg_frame_rate_fps_exact_integer_rates() {
        // 24, 25, 30 fps map to 24*256, 25*256, 30*256 on-wire.
        for &(carrier, expected_fps) in
            &[(24u16 * 256, 24.0_f64), (25 * 256, 25.0), (30 * 256, 30.0)]
        {
            let s = SubSeqLayerCharacteristic {
                accurate_statistics_flag: true,
                average_bit_rate: 0,
                average_frame_rate: carrier,
            };
            let fps = s.average_frame_rate_fps().expect("non-zero ⇒ Some");
            assert!(
                (fps - expected_fps).abs() < f64::EPSILON,
                "carrier {carrier} ⇒ expected {expected_fps} got {fps}"
            );
        }
    }

    #[test]
    fn sub_seq_layer_avg_frame_rate_fps_maximum() {
        // On-wire u(16) ceiling: 65535 / 256 ≈ 255.996 fps.
        let s = SubSeqLayerCharacteristic {
            accurate_statistics_flag: true,
            average_bit_rate: 0,
            average_frame_rate: u16::MAX,
        };
        let fps = s.average_frame_rate_fps().expect("non-zero ⇒ Some");
        assert!((fps - (65535.0_f64 / 256.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn sub_seq_layer_accessors_via_parse_round_trip() {
        // End-to-end: parse_sub_seq_layer_characteristics → accessors.
        // num_sub_seq_layers_minus1 = 0 (ue → "1") → 1 entry.
        // accurate=1, bit_rate=2500 (2.5 Mbit/s), frame_rate=24*256 (24 fps).
        let payload = pack_bits(&[
            (1, 1),            // num_sub_seq_layers_minus1 = 0
            (1, 1),            // accurate_statistics_flag
            (2_500, 16),       // average_bit_rate
            (24u64 * 256, 16), // average_frame_rate
        ]);
        let got = parse_sub_seq_layer_characteristics(&payload).unwrap();
        assert_eq!(got.layers.len(), 1);
        assert_eq!(got.layers[0].average_bit_rate_bps(), Some(2_500_000));
        let fps = got.layers[0]
            .average_frame_rate_fps()
            .expect("non-zero ⇒ Some");
        assert!((fps - 24.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sub_seq_characteristics_avg_rate_accessors_round_trip() {
        // End-to-end via parse_sub_seq_characteristics, exercising the
        // §D.2.14 carrier (payload type 12) that re-uses
        // SubSeqLayerCharacteristic under `average_rate`.
        // layer=0 (ue → "1"), id=0 (ue → "1"),
        // duration_flag=0, average_rate_flag=1,
        // accurate=0, bit_rate=10000 (10 Mbit/s), frame_rate=60*256,
        // num_referenced_subseqs=0 (ue → "1").
        let payload = pack_bits(&[
            (1, 1),            // sub_seq_layer_num = 0
            (1, 1),            // sub_seq_id = 0
            (0, 1),            // duration_flag = 0
            (1, 1),            // average_rate_flag = 1
            (0, 1),            // accurate_statistics_flag = 0
            (10_000, 16),      // average_bit_rate
            (60u64 * 256, 16), // average_frame_rate
            (1, 1),            // num_referenced_subseqs = 0
        ]);
        let got = parse_sub_seq_characteristics(&payload).unwrap();
        let rate = got.average_rate.expect("average_rate present");
        assert_eq!(rate.average_bit_rate_bps(), Some(10_000_000));
        let fps = rate.average_frame_rate_fps().expect("non-zero ⇒ Some");
        assert!((fps - 60.0).abs() < f64::EPSILON);
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

    // --- §D.2.13 sub_seq_duration_seconds typed accessor ---------------

    #[test]
    fn sub_seq_duration_seconds_none_when_unspecified() {
        // duration_flag == 0 ⇒ sub_seq_duration is None per spec
        // ("duration_flag equal to 0 indicates that the duration of the
        // target sub-sequence is not specified") — no scalar derivation.
        let c = SubSeqCharacteristics {
            sub_seq_layer_num: 0,
            sub_seq_id: 0,
            sub_seq_duration: None,
            average_rate: None,
            referenced: Vec::new(),
        };
        assert!(c.sub_seq_duration_seconds().is_none());
    }

    #[test]
    fn sub_seq_duration_seconds_minimum_nonzero() {
        // Smallest non-zero carrier: 1 tick of a 90-kHz clock is exactly
        // 1/90_000 second; with denominator a power-of-five times a power
        // of two the value is non-dyadic, so check via the exact integer
        // identity ticks * 90_000 == reconstructed * 90_000² (i.e. the
        // accessor's f64 result times 90_000 returns to 1.0 exactly).
        let c = SubSeqCharacteristics {
            sub_seq_layer_num: 0,
            sub_seq_id: 0,
            sub_seq_duration: Some(1),
            average_rate: None,
            referenced: Vec::new(),
        };
        let s = c.sub_seq_duration_seconds().unwrap();
        // 1 / 90_000 ≈ 1.111…e-5; verify the round-trip pins to 1 tick.
        assert!((s * 90_000.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sub_seq_duration_seconds_one_second() {
        // 90_000 ticks == 1 second exactly (the 90-kHz tick rate is
        // defined so an integer second is an integer tick count).
        let c = SubSeqCharacteristics {
            sub_seq_layer_num: 0,
            sub_seq_id: 0,
            sub_seq_duration: Some(90_000),
            average_rate: None,
            referenced: Vec::new(),
        };
        assert_eq!(c.sub_seq_duration_seconds(), Some(1.0));
    }

    #[test]
    fn sub_seq_duration_seconds_one_frame_at_30fps() {
        // 3_000 ticks == 1 frame at 30 fps (30 * 3_000 == 90_000); a
        // common GOP-edge unit and exactly representable in f64.
        let c = SubSeqCharacteristics {
            sub_seq_layer_num: 0,
            sub_seq_id: 0,
            sub_seq_duration: Some(3_000),
            average_rate: None,
            referenced: Vec::new(),
        };
        let s = c.sub_seq_duration_seconds().unwrap();
        assert!((s - 1.0 / 30.0).abs() < 1e-12);
    }

    #[test]
    fn sub_seq_duration_seconds_u32_ceiling() {
        // Carrier ceiling u32::MAX == 4_294_967_295 ticks ≈ 47_721.86 s
        // ≈ 13 h 15 min 21.86 s — pin the exact f64 quotient.
        let c = SubSeqCharacteristics {
            sub_seq_layer_num: 0,
            sub_seq_id: 0,
            sub_seq_duration: Some(u32::MAX),
            average_rate: None,
            referenced: Vec::new(),
        };
        let expected = f64::from(u32::MAX) / 90_000.0;
        assert_eq!(c.sub_seq_duration_seconds(), Some(expected));
    }

    #[test]
    fn sub_seq_duration_seconds_roundtrip_via_parse() {
        // End-to-end via parse_sub_seq_characteristics: pack a payload
        // with duration_flag=1 and sub_seq_duration=180_000 (== 2 s).
        let payload = pack_bits(&[
            (1, 1),        // sub_seq_layer_num = 0
            (1, 1),        // sub_seq_id = 0
            (1, 1),        // duration_flag = 1
            (180_000, 32), // sub_seq_duration = 180_000 ticks == 2 s
            (0, 1),        // average_rate_flag = 0
            (1, 1),        // num_referenced_subseqs = 0
        ]);
        let got = parse_sub_seq_characteristics(&payload).unwrap();
        assert_eq!(got.sub_seq_duration, Some(180_000));
        assert_eq!(got.sub_seq_duration_seconds(), Some(2.0));
    }

    #[test]
    fn sub_seq_duration_seconds_roundtrip_unspecified() {
        // End-to-end via parse_sub_seq_characteristics with
        // duration_flag=0: typed accessor must mirror the on-wire None.
        let payload = pack_bits(&[
            (1, 1), // sub_seq_layer_num = 0
            (1, 1), // sub_seq_id = 0
            (0, 1), // duration_flag = 0
            (0, 1), // average_rate_flag = 0
            (1, 1), // num_referenced_subseqs = 0
        ]);
        let got = parse_sub_seq_characteristics(&payload).unwrap();
        assert!(got.sub_seq_duration.is_none());
        assert!(got.sub_seq_duration_seconds().is_none());
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
    fn progressive_refinement_num_refinement_steps_minimum() {
        // num_refinement_steps_minus1 == 0 ⇒ NumRefinementSteps == 1.
        let r = ProgressiveRefinementSegmentStart {
            progressive_refinement_id: 0,
            num_refinement_steps_minus1: 0,
        };
        assert_eq!(r.num_refinement_steps(), 1);
    }

    #[test]
    fn progressive_refinement_num_refinement_steps_typical() {
        // A typical small refinement segment with three steps:
        // num_refinement_steps_minus1 == 2 ⇒ NumRefinementSteps == 3.
        let r = ProgressiveRefinementSegmentStart {
            progressive_refinement_id: 7,
            num_refinement_steps_minus1: 2,
        };
        assert_eq!(r.num_refinement_steps(), 3);
    }

    #[test]
    fn progressive_refinement_num_refinement_steps_u32_ceiling() {
        // num_refinement_steps_minus1 == u32::MAX must not overflow
        // the u64 return: u32::MAX + 1 == 4_294_967_296.
        let r = ProgressiveRefinementSegmentStart {
            progressive_refinement_id: 0,
            num_refinement_steps_minus1: u32::MAX,
        };
        assert_eq!(r.num_refinement_steps(), u64::from(u32::MAX) + 1);
        assert_eq!(r.num_refinement_steps(), 4_294_967_296);
    }

    #[test]
    fn progressive_refinement_num_refinement_steps_parse_round_trip() {
        // id=5 (ue=5 → "00110"), steps_minus1=9 (ue=9 → "0001010").
        // After parsing, NumRefinementSteps should be 9 + 1 = 10.
        let payload = pack_bits(&[(0b00110, 5), (0b0001010, 7)]);
        let got = parse_progressive_refinement_segment_start(&payload).unwrap();
        assert_eq!(got.progressive_refinement_id, 5);
        assert_eq!(got.num_refinement_steps_minus1, 9);
        assert_eq!(got.num_refinement_steps(), 10);
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

    // -- Round 117: colour_remapping_info (payload type 142, §D.2.30) --

    #[test]
    fn colour_remapping_info_cancel() {
        // colour_remap_id = ue(2) → codeNum 2 = "011"; cancel_flag = 1.
        let payload = pack_bits(&[(0b011, 3), (1, 1)]);
        let got = parse_colour_remapping_info(&payload).unwrap();
        assert_eq!(got.colour_remap_id, 2);
        assert!(got.cancel_flag);
        assert!(got.body.is_none());
    }

    #[test]
    fn colour_remapping_info_minimal_no_optionals() {
        // id = ue(0) = "1"; cancel = 0; repetition_period = ue(0) = "1";
        // video_signal_info_present = 0; input_bd = 8; output_bd = 8;
        // pre-LUT: three num_val_minus1 = 0; matrix_present = 0;
        // post-LUT: three num_val_minus1 = 0.
        let payload = pack_bits(&[
            (1, 1), // id ue=0
            (0, 1), // cancel
            (1, 1), // repetition_period ue=0
            (0, 1), // video_signal_info_present
            (8, 8), // input_bit_depth
            (8, 8), // output_bit_depth
            (0, 8), // pre num_val_minus1[0]
            (0, 8), // pre num_val_minus1[1]
            (0, 8), // pre num_val_minus1[2]
            (0, 1), // matrix_present
            (0, 8), // post num_val_minus1[0]
            (0, 8), // post num_val_minus1[1]
            (0, 8), // post num_val_minus1[2]
        ]);
        let got = parse_colour_remapping_info(&payload).unwrap();
        assert_eq!(got.colour_remap_id, 0);
        assert!(!got.cancel_flag);
        let body = got.body.unwrap();
        assert_eq!(body.repetition_period, 0);
        assert!(body.video_signal_info.is_none());
        assert_eq!(body.input_bit_depth, 8);
        assert_eq!(body.output_bit_depth, 8);
        assert!(body.pre_lut.iter().all(|c| c.is_empty()));
        assert!(body.matrix.is_none());
        assert!(body.post_lut.iter().all(|c| c.is_empty()));
    }

    #[test]
    fn colour_remapping_info_full_8bit() {
        // Full message at 8-bit input/output (LUT values are 8 bits).
        // video signal info present (BT.709-ish: prim=1, tf=1, mc=1),
        // a 2-pivot pre-LUT on component 0 only, an identity-ish matrix
        // with log2_matrix_denom = 5, and a 2-pivot post-LUT on comp 2.
        let payload = pack_bits(&[
            (0b010, 3), // id ue=1 ("010")
            (0, 1),     // cancel
            (1, 1),     // repetition_period ue=0
            (1, 1),     // video_signal_info_present
            (1, 1),     // full_range_flag
            (1, 8),     // primaries
            (1, 8),     // transfer_function
            (1, 8),     // matrix_coefficients
            (8, 8),     // input_bit_depth = 8 → 8-bit LUT codes
            (8, 8),     // output_bit_depth = 8 → 8-bit LUT targets
            // pre-LUT comp 0: num_val_minus1 = 1 → 2 entries
            (1, 8),
            (16, 8),  // coded[0]
            (32, 8),  // target[0]
            (200, 8), // coded[1]
            (220, 8), // target[1]
            (0, 8),   // pre comp 1: none
            (0, 8),   // pre comp 2: none
            // matrix present
            (1, 1), // matrix_present
            (5, 4), // log2_matrix_denom
            // colour_remap_coeffs[c][i], se(v): 32,0,0 / 0,32,0 / 0,0,32
            // se(32) ⇒ codeNum 63 ⇒ ue value n = 64 in 13 bits
            // (6 leading zeros + 7-bit "1000000"); se(0) ⇒ "1".
            (64, 13), // se(32)
            (1, 1),   // se(0)
            (1, 1),   // se(0)
            (1, 1),   // se(0)
            (64, 13), // se(32)
            (1, 1),   // se(0)
            (1, 1),   // se(0)
            (1, 1),   // se(0)
            (64, 13), // se(32)
            // post-LUT comp 0: none
            (0, 8),
            (0, 8), // post comp 1: none
            // post comp 2: num_val_minus1 = 1 → 2 entries
            (1, 8),
            (5, 8),   // coded[0]
            (9, 8),   // target[0]
            (250, 8), // coded[1]
            (255, 8), // target[1]
        ]);
        let got = parse_colour_remapping_info(&payload).unwrap();
        assert_eq!(got.colour_remap_id, 1);
        let body = got.body.unwrap();
        let vsi = body.video_signal_info.unwrap();
        assert!(vsi.full_range_flag);
        assert_eq!(vsi.primaries, 1);
        assert_eq!(vsi.transfer_function, 1);
        assert_eq!(vsi.matrix_coefficients, 1);
        assert_eq!(body.pre_lut[0].len(), 2);
        assert_eq!(body.pre_lut[0][0].coded_value, 16);
        assert_eq!(body.pre_lut[0][0].target_value, 32);
        assert_eq!(body.pre_lut[0][1].coded_value, 200);
        assert_eq!(body.pre_lut[0][1].target_value, 220);
        assert!(body.pre_lut[1].is_empty());
        assert!(body.pre_lut[2].is_empty());
        let m = body.matrix.unwrap();
        assert_eq!(m.log2_matrix_denom, 5);
        assert_eq!(m.coeffs, [[32, 0, 0], [0, 32, 0], [0, 0, 32]]);
        assert!(body.post_lut[0].is_empty());
        assert!(body.post_lut[1].is_empty());
        assert_eq!(body.post_lut[2].len(), 2);
        assert_eq!(body.post_lut[2][1].coded_value, 250);
        assert_eq!(body.post_lut[2][1].target_value, 255);
    }

    #[test]
    fn colour_remapping_info_10bit_uses_16bit_lut_values() {
        // input_bit_depth = 10 → ((10+7)>>3)<<3 = 16-bit LUT codes;
        // output_bit_depth = 12 → 16-bit targets. One pivot on comp 1.
        let payload = pack_bits(&[
            (1, 1),  // id ue=0
            (0, 1),  // cancel
            (1, 1),  // repetition_period ue=0
            (0, 1),  // video_signal_info_present
            (10, 8), // input_bit_depth = 10
            (12, 8), // output_bit_depth = 12
            (0, 8),  // pre comp 0: none
            // pre comp 1: num_val_minus1 = 1 → 2 entries, 16-bit each
            (1, 8),
            (500, 16),  // coded[0]
            (1000, 16), // target[0]
            (1000, 16), // coded[1]
            (3000, 16), // target[1]
            (0, 8),     // pre comp 2: none
            (0, 1),     // matrix_present = 0
            (0, 8),     // post comp 0
            (0, 8),     // post comp 1
            (0, 8),     // post comp 2
        ]);
        let got = parse_colour_remapping_info(&payload).unwrap();
        let body = got.body.unwrap();
        assert_eq!(body.input_bit_depth, 10);
        assert_eq!(body.output_bit_depth, 12);
        assert_eq!(body.pre_lut[1].len(), 2);
        assert_eq!(body.pre_lut[1][0].coded_value, 500);
        assert_eq!(body.pre_lut[1][0].target_value, 1000);
        assert_eq!(body.pre_lut[1][1].coded_value, 1000);
        assert_eq!(body.pre_lut[1][1].target_value, 3000);
    }

    #[test]
    fn colour_remapping_info_negative_matrix_coeff() {
        // se(-1) = codeNum 2 = "011". Verify signed coeff round-trips.
        let payload = pack_bits(&[
            (1, 1), // id ue=0
            (0, 1), // cancel
            (1, 1), // repetition_period ue=0
            (0, 1), // video_signal_info_present
            (8, 8), // input
            (8, 8), // output
            (0, 8), // pre 0
            (0, 8), // pre 1
            (0, 8), // pre 2
            (1, 1), // matrix_present
            (0, 4), // log2_matrix_denom = 0
            // coeffs: all se(-1) = "011"
            (0b011, 3),
            (0b011, 3),
            (0b011, 3),
            (0b011, 3),
            (0b011, 3),
            (0b011, 3),
            (0b011, 3),
            (0b011, 3),
            (0b011, 3),
            (0, 8), // post 0
            (0, 8), // post 1
            (0, 8), // post 2
        ]);
        let got = parse_colour_remapping_info(&payload).unwrap();
        let m = got.body.unwrap().matrix.unwrap();
        assert_eq!(m.log2_matrix_denom, 0);
        assert_eq!(m.coeffs, [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]);
    }

    #[test]
    fn colour_remapping_info_input_bit_depth_too_low_rejected() {
        // input_bit_depth = 7 (reserved) → rejected per §D.2.30.
        let payload = pack_bits(&[
            (1, 1), // id ue=0
            (0, 1), // cancel
            (1, 1), // repetition_period ue=0
            (0, 1), // video_signal_info_present
            (7, 8), // input_bit_depth = 7 (out of 8..=16)
            (8, 8), // output_bit_depth
        ]);
        let err = parse_colour_remapping_info(&payload).unwrap_err();
        assert_eq!(
            err,
            SeiError::ColourRemapBitDepthOutOfRange {
                which: "input",
                got: 7
            }
        );
    }

    #[test]
    fn colour_remapping_info_repetition_period_out_of_range_rejected() {
        // repetition_period = 16385 → ue codeNum 16385.
        // ue(16385): 16385+1 = 16386 = 0b100000000000010, 15 bits ⇒
        // 14 leading zeros, then the 15-bit value.
        let payload = pack_bits(&[
            (1, 1),                  // id ue=0
            (0, 1),                  // cancel
            (0, 14),                 // ue prefix: 14 leading zeros
            (0b100000000000010, 15), // 15-bit suffix → codeNum 16385
        ]);
        let err = parse_colour_remapping_info(&payload).unwrap_err();
        assert_eq!(err, SeiError::ColourRemapRepetitionPeriodOutOfRange(16385));
    }

    #[test]
    fn colour_remapping_info_lut_count_out_of_range_rejected() {
        // pre_lut_num_val_minus1[0] = 33 (> 32) → rejected.
        let payload = pack_bits(&[
            (1, 1),  // id ue=0
            (0, 1),  // cancel
            (1, 1),  // repetition_period ue=0
            (0, 1),  // video_signal_info_present
            (8, 8),  // input
            (8, 8),  // output
            (33, 8), // pre num_val_minus1[0] = 33
        ]);
        let err = parse_colour_remapping_info(&payload).unwrap_err();
        assert_eq!(
            err,
            SeiError::ColourRemapLutCountOutOfRange {
                which: "pre",
                c: 0,
                got: 33
            }
        );
    }

    #[test]
    fn parse_payload_dispatches_colour_remapping_info() {
        // Minimal cancel message routed through parse_payload(142, ..).
        let payload = pack_bits(&[(1, 1), (1, 1)]); // id ue=0, cancel=1
        let ctx = SeiContext::default();
        match parse_payload(142, &payload, &ctx).unwrap() {
            SeiPayload::ColourRemappingInfo(c) => {
                assert_eq!(c.colour_remap_id, 0);
                assert!(c.cancel_flag);
                assert!(c.body.is_none());
            }
            other => panic!("expected ColourRemappingInfo, got {other:?}"),
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

    // ----- §D.1.36 / §D.2.36 SEI manifest (payload type 200) -----

    #[test]
    fn sei_manifest_empty() {
        // manifest_num_sei_msg_types = 0 → exactly 16 zero bits.
        let payload = vec![0x00, 0x00];
        let got = parse_sei_manifest(&payload).unwrap();
        assert!(got.entries.is_empty());
    }

    #[test]
    fn sei_manifest_single_entry_necessary() {
        // count = 1, payload_type = 45 (frame_packing_arrangement),
        // description = 1 (ExpectedNecessary).
        let payload = vec![
            0x00, 0x01, // count = 1
            0x00, 0x2D, // payload_type = 45
            0x01, // description = 1
        ];
        let got = parse_sei_manifest(&payload).unwrap();
        assert_eq!(got.entries.len(), 1);
        assert_eq!(got.entries[0].payload_type, 45);
        assert_eq!(
            got.entries[0].description,
            SeiManifestDescription::ExpectedNecessary
        );
    }

    #[test]
    fn sei_manifest_three_entries_mixed_descriptions() {
        let payload = vec![
            0x00, 0x03, // count = 3
            0x00, 0x00, // payload_type = 0 (buffering_period)
            0x01, // description = 1 (necessary)
            0x00, 0x90, // payload_type = 144 (content_light_level_info)
            0x02, // description = 2 (unnecessary)
            0x00, 0x97, // payload_type = 151 (cubemap_projection)
            0x03, // description = 3 (undetermined)
        ];
        let got = parse_sei_manifest(&payload).unwrap();
        assert_eq!(got.entries.len(), 3);
        assert_eq!(got.entries[0].payload_type, 0);
        assert_eq!(
            got.entries[0].description,
            SeiManifestDescription::ExpectedNecessary
        );
        assert_eq!(got.entries[1].payload_type, 144);
        assert_eq!(
            got.entries[1].description,
            SeiManifestDescription::ExpectedUnnecessary
        );
        assert_eq!(got.entries[2].payload_type, 151);
        assert_eq!(
            got.entries[2].description,
            SeiManifestDescription::ExpectedUndetermined
        );
    }

    #[test]
    fn sei_manifest_reserved_description_preserved() {
        // §D.2.36: values 4..=255 are reserved. Decoders shall allow
        // them and ignore the associated info. We surface the raw byte
        // via SeiManifestDescription::Reserved so callers can implement
        // the "shall ignore" requirement.
        let payload = vec![
            0x00, 0x01, // count = 1
            0x00, 0x05, // payload_type = 5
            0x42, // description = 0x42 (reserved)
        ];
        let got = parse_sei_manifest(&payload).unwrap();
        assert_eq!(got.entries.len(), 1);
        assert_eq!(
            got.entries[0].description,
            SeiManifestDescription::Reserved(0x42)
        );
    }

    #[test]
    fn sei_manifest_duplicate_payload_type_rejected() {
        // §D.2.36: manifest_sei_payload_type values shall be distinct.
        let payload = vec![
            0x00, 0x02, // count = 2
            0x00, 0x2D, 0x01, // (45, necessary)
            0x00, 0x2D, 0x02, // (45, unnecessary) — duplicate
        ];
        let err = parse_sei_manifest(&payload).unwrap_err();
        assert_eq!(
            err,
            SeiError::SeiManifestDuplicatePayloadType {
                payload_type: 45,
                first: 0,
                second: 1
            }
        );
    }

    #[test]
    fn sei_manifest_truncated_input_rejects() {
        // count promises 1 entry but payload runs out partway through.
        let payload = vec![0x00, 0x01, 0x00, 0x2D]; // missing description byte
        let err = parse_sei_manifest(&payload).unwrap_err();
        assert!(matches!(err, SeiError::Bitstream(_)));
    }

    #[test]
    fn parse_payload_dispatches_sei_manifest() {
        let payload = vec![
            0x00, 0x01, // count = 1
            0x00, 0x00, // payload_type = 0
            0x00, // description = 0 (NotExpected)
        ];
        let ctx = SeiContext::default();
        match parse_payload(200, &payload, &ctx).unwrap() {
            SeiPayload::SeiManifest(m) => {
                assert_eq!(m.entries.len(), 1);
                assert_eq!(m.entries[0].payload_type, 0);
                assert_eq!(
                    m.entries[0].description,
                    SeiManifestDescription::NotExpected
                );
            }
            other => panic!("expected SeiManifest, got {other:?}"),
        }
    }

    // ----- §D.1.37 / §D.2.37 SEI prefix indication (payload type 201) -----

    #[test]
    fn sei_prefix_indication_single_byte_aligned() {
        // prefix_sei_payload_type = 45, num_indications = 1 (minus1=0).
        // Indication 0: num_bits_minus1 = 7 → 8 bits of data = 0xA5.
        let payload = vec![
            0x00, 0x2D, // prefix_sei_payload_type = 45
            0x00, // num_sei_prefix_indications_minus1 = 0
            0x00, 0x07, // num_bits_in_prefix_indication_minus1[0] = 7
            0xA5, // 8 sei_prefix_data_bit values
        ];
        let got = parse_sei_prefix_indication(&payload).unwrap();
        assert_eq!(got.prefix_sei_payload_type, 45);
        assert_eq!(got.indications.len(), 1);
        assert_eq!(got.indications[0].bit_count, 8);
        assert_eq!(got.indications[0].data, vec![0xA5]);
    }

    #[test]
    fn sei_prefix_indication_short_bitstring_with_alignment_one_bits() {
        // 3-bit indication "101" followed by five `1`-fill alignment bits
        // (per §D.1.37 byte_alignment_bit_equal_to_one) → 0xBF.
        let payload = vec![
            0x00, 0x2D, // prefix_sei_payload_type
            0x00, // num_indications_minus1 = 0
            0x00, 0x02, // num_bits_minus1 = 2 → 3 bits
            0xBF, // 101 + five filler `1`s
        ];
        let got = parse_sei_prefix_indication(&payload).unwrap();
        assert_eq!(got.indications.len(), 1);
        assert_eq!(got.indications[0].bit_count, 3);
        // Bits "101" packed MSB-first into one byte → 0b1010_0000 = 0xA0.
        assert_eq!(got.indications[0].data, vec![0xA0]);
    }

    #[test]
    fn sei_prefix_indication_two_entries() {
        // Two indications: 4 bits "1100" + 12 bits "1010_1100_0011".
        // First indication occupies 4 bits + 4 fill = 1 byte = 0b1100_1111 = 0xCF.
        // Second indication occupies 12 bits + 4 fill = 2 bytes.
        //   bits: 1010 1100 0011 + 1111 → 0xAC 0x3F.
        let payload = vec![
            0x00, 0x2D, // prefix_sei_payload_type
            0x01, // num_indications_minus1 = 1
            0x00, 0x03, // bits_minus1 = 3 → 4 bits
            0xCF, // "1100" + "1111" filler
            0x00, 0x0B, // bits_minus1 = 11 → 12 bits
            0xAC, 0x3F, // "1010_1100_0011" + "1111" filler
        ];
        let got = parse_sei_prefix_indication(&payload).unwrap();
        assert_eq!(got.indications.len(), 2);
        assert_eq!(got.indications[0].bit_count, 4);
        // 4 bits "1100" packed MSB-first → 0b1100_0000 = 0xC0.
        assert_eq!(got.indications[0].data, vec![0xC0]);
        assert_eq!(got.indications[1].bit_count, 12);
        // 12 bits "1010_1100_0011" → byte0=0xAC, byte1=0b0011_0000=0x30.
        assert_eq!(got.indications[1].data, vec![0xAC, 0x30]);
    }

    #[test]
    fn sei_prefix_indication_oversized_bit_count_rejected() {
        // num_bits_minus1 + 1 = 9 bits but only 1 byte of data follows.
        let payload = vec![
            0x00, 0x2D, // prefix_sei_payload_type
            0x00, // num_indications_minus1 = 0
            0x00, 0x08, // bits_minus1 = 8 → 9 bits required
            0xFF, // only 8 bits available
        ];
        let err = parse_sei_prefix_indication(&payload).unwrap_err();
        match err {
            SeiError::SeiPrefixIndicationOverflow { i, bits, available } => {
                assert_eq!(i, 0);
                assert_eq!(bits, 9);
                assert_eq!(available, 8);
            }
            other => panic!("expected SeiPrefixIndicationOverflow, got {other:?}"),
        }
    }

    #[test]
    fn sei_prefix_indication_truncated_header_rejected() {
        // Header truncated before num_sei_prefix_indications_minus1.
        let payload = vec![0x00, 0x2D]; // only prefix_sei_payload_type
        let err = parse_sei_prefix_indication(&payload).unwrap_err();
        assert!(matches!(err, SeiError::Bitstream(_)));
    }

    #[test]
    fn parse_payload_dispatches_sei_prefix_indication() {
        let payload = vec![
            0x00, 0x2D, // prefix_sei_payload_type = 45
            0x00, // num_indications_minus1 = 0
            0x00, 0x07, // bits_minus1 = 7 → 8 bits
            0xA5,
        ];
        let ctx = SeiContext::default();
        match parse_payload(201, &payload, &ctx).unwrap() {
            SeiPayload::SeiPrefixIndication(p) => {
                assert_eq!(p.prefix_sei_payload_type, 45);
                assert_eq!(p.indications.len(), 1);
                assert_eq!(p.indications[0].bit_count, 8);
                assert_eq!(p.indications[0].data, vec![0xA5]);
            }
            other => panic!("expected SeiPrefixIndication, got {other:?}"),
        }
    }

    // =========================================================================
    // ATSC1 envelope (A/53 Part 4 §6.2.3) — typed view on top of
    // user_data_registered_itu_t_t35 (round 158).
    // =========================================================================

    /// Build a full user_data_registered_itu_t_t35 payload that wraps an
    /// ATSC1 envelope (country=0xB5 USA + provider 0x0031 + user_identifier
    /// + body). Used to exercise the chained `parse_user_data_registered…`
    ///   → `parse_atsc1_envelope` flow.
    fn build_t35_atsc1(user_identifier: u32, body: &[u8]) -> Vec<u8> {
        let mut v = Vec::with_capacity(7 + body.len());
        v.push(0xB5);
        v.extend_from_slice(&[0x00, 0x31]); // provider_code = 0x0031
        v.extend_from_slice(&user_identifier.to_be_bytes());
        v.extend_from_slice(body);
        v
    }

    #[test]
    fn atsc1_envelope_ga94_cc_data_round_trip() {
        // 'GA94' + user_data_type_code 0x03 (MPEG_cc_data) + 5 opaque cc bytes
        // ending in the 0xFF marker the cc_data() trailer requires.
        let cc = [0xFC, 0x94, 0x2C, 0xFE, 0xFF];
        let mut body = vec![0x03u8];
        body.extend_from_slice(&cc);
        let raw = build_t35_atsc1(Atsc1UserIdentifier::GA94, &body);

        let outer = parse_user_data_registered_itu_t_t35(&raw).unwrap();
        assert_eq!(outer.country_code, 0xB5);
        assert!(outer.country_code_extension.is_none());

        let env = parse_atsc1_envelope(&outer.payload_bytes).unwrap();
        assert_eq!(env.provider_code, 0x0031);
        assert_eq!(env.user_identifier, Atsc1UserIdentifier::Ga94);
        match env.body {
            Atsc1Body::AtscUserData(Atsc1UserData::CcData { cc_data_bytes }) => {
                // CEA-708 inner layout intentionally opaque — only the byte
                // count is asserted (we hand the bytes through unchanged for
                // a downstream CEA-708 parser).
                assert_eq!(cc_data_bytes, cc);
            }
            other => panic!("expected GA94 → CcData, got {other:?}"),
        }
    }

    #[test]
    fn atsc1_envelope_ga94_bar_data_letterbox() {
        // top_bar_flag=1, bottom_bar_flag=1, left=0, right=0, reserved=1111,
        // (one_bits=11, 14-bit line_number_end_of_top_bar = 0x0040 = 64),
        // (one_bits=11, 14-bit line_number_start_of_bottom_bar = 0x03BF
        //  = 959).
        //
        // First byte:  1 1 0 0  | 1 1 1 1 = 0b1100_1111 = 0xCF
        // Then: '11' u(2) + u(14) = 16 bits = 2 bytes per value.
        //   64  = 0x0040 → with 11 prefix → 0xC040
        //   959 = 0x03BF → with 11 prefix → 0xC3BF
        let body = [0x06, 0xCF, 0xC0, 0x40, 0xC3, 0xBF];
        let raw = build_t35_atsc1(Atsc1UserIdentifier::GA94, &body);
        let outer = parse_user_data_registered_itu_t_t35(&raw).unwrap();
        let env = parse_atsc1_envelope(&outer.payload_bytes).unwrap();
        match env.body {
            Atsc1Body::AtscUserData(Atsc1UserData::BarData(bar)) => {
                assert_eq!(bar.letterbox, Some((64, 959)));
                assert_eq!(bar.pillarbox, None);
            }
            other => panic!("expected BarData letterbox, got {other:?}"),
        }
    }

    #[test]
    fn atsc1_envelope_ga94_bar_data_pillarbox() {
        // top=0, bottom=0, left=1, right=1, reserved=1111, then two 16-bit
        // (11 prefix + 14-bit) values: pixel_number_end_of_left_bar = 100,
        // pixel_number_start_of_right_bar = 1820.
        //   100  = 0x0064 → 0xC064
        //   1820 = 0x071C → 0xC71C
        // First byte: 0 0 1 1 | 1 1 1 1 = 0x3F
        let body = [0x06, 0x3F, 0xC0, 0x64, 0xC7, 0x1C];
        let raw = build_t35_atsc1(Atsc1UserIdentifier::GA94, &body);
        let outer = parse_user_data_registered_itu_t_t35(&raw).unwrap();
        let env = parse_atsc1_envelope(&outer.payload_bytes).unwrap();
        match env.body {
            Atsc1Body::AtscUserData(Atsc1UserData::BarData(bar)) => {
                assert_eq!(bar.letterbox, None);
                assert_eq!(bar.pillarbox, Some((100, 1820)));
            }
            other => panic!("expected BarData pillarbox, got {other:?}"),
        }
    }

    #[test]
    fn atsc1_envelope_bar_data_rejects_letterbox_and_pillarbox_simultaneously() {
        // top=1, bottom=1, left=1, right=1 — explicitly forbidden by
        // §6.2.3.2 ("either top and bottom bars or left and right bars,
        // but not both pairs at once").
        let body = [0x06, 0xFF];
        let raw = build_t35_atsc1(Atsc1UserIdentifier::GA94, &body);
        let outer = parse_user_data_registered_itu_t_t35(&raw).unwrap();
        let err = parse_atsc1_envelope(&outer.payload_bytes).unwrap_err();
        assert_eq!(err, SeiError::Atsc1BarDataLetterboxPillarboxBoth);
    }

    #[test]
    fn atsc1_envelope_bar_data_rejects_top_bottom_mismatch() {
        // top=1, bottom=0 — §6.2.3.2 requires equality.
        // First byte: 1 0 0 0 | 1 1 1 1 = 0x8F. Then a single (11+14) value.
        let body = [0x06, 0x8F, 0xC0, 0x10];
        let raw = build_t35_atsc1(Atsc1UserIdentifier::GA94, &body);
        let outer = parse_user_data_registered_itu_t_t35(&raw).unwrap();
        let err = parse_atsc1_envelope(&outer.payload_bytes).unwrap_err();
        assert_eq!(
            err,
            SeiError::Atsc1BarDataTopBottomMismatch {
                top: true,
                bottom: false
            }
        );
    }

    #[test]
    fn atsc1_envelope_bar_data_rejects_left_right_mismatch() {
        // top=0, bottom=0, left=1, right=0 — §6.2.3.2 requires equality.
        // First byte: 0 0 1 0 | 1 1 1 1 = 0x2F.
        let body = [0x06, 0x2F, 0xC0, 0x10];
        let raw = build_t35_atsc1(Atsc1UserIdentifier::GA94, &body);
        let outer = parse_user_data_registered_itu_t_t35(&raw).unwrap();
        let err = parse_atsc1_envelope(&outer.payload_bytes).unwrap_err();
        assert_eq!(
            err,
            SeiError::Atsc1BarDataLeftRightMismatch {
                left: true,
                right: false,
            }
        );
    }

    #[test]
    fn atsc1_envelope_bar_data_rejects_wrong_reserved_nibble() {
        // top=0, bottom=0, left=0, right=0, reserved=0000 (must be 1111).
        let body = [0x06, 0x00];
        let raw = build_t35_atsc1(Atsc1UserIdentifier::GA94, &body);
        let outer = parse_user_data_registered_itu_t_t35(&raw).unwrap();
        let err = parse_atsc1_envelope(&outer.payload_bytes).unwrap_err();
        assert_eq!(err, SeiError::Atsc1BarDataReservedMismatch(0b0000));
    }

    #[test]
    fn atsc1_envelope_bar_data_rejects_wrong_one_bits_prefix() {
        // Letterbox flags set, reserved correct, but the per-value '11'
        // prefix bits before line_number_end_of_top_bar are 0b01.
        // First byte: 1 1 0 0 | 1 1 1 1 = 0xCF.
        // Then: prefix 01 + 14-bit 0 → 0x4000.
        let body = [0x06, 0xCF, 0x40, 0x00, 0xC0, 0x00];
        let raw = build_t35_atsc1(Atsc1UserIdentifier::GA94, &body);
        let outer = parse_user_data_registered_itu_t_t35(&raw).unwrap();
        let err = parse_atsc1_envelope(&outer.payload_bytes).unwrap_err();
        assert_eq!(
            err,
            SeiError::Atsc1BarDataOneBitsMismatch {
                which: "top",
                got: 0b01,
            }
        );
    }

    #[test]
    fn atsc1_envelope_ga94_reserved_type_code_keeps_raw_bytes() {
        // Table 6.9: 0x01 is ATSC reserved. §6.2.2 requires receivers
        // to silently discard; we surface the raw bytes so a diagnostic
        // tool can still inspect them.
        let body = [0x01, 0xDE, 0xAD, 0xBE, 0xEF];
        let raw = build_t35_atsc1(Atsc1UserIdentifier::GA94, &body);
        let outer = parse_user_data_registered_itu_t_t35(&raw).unwrap();
        let env = parse_atsc1_envelope(&outer.payload_bytes).unwrap();
        match env.body {
            Atsc1Body::AtscUserData(Atsc1UserData::Reserved { type_code, raw }) => {
                assert_eq!(type_code, 0x01);
                assert_eq!(raw, vec![0xDE, 0xAD, 0xBE, 0xEF]);
            }
            other => panic!("expected reserved type_code, got {other:?}"),
        }
    }

    #[test]
    fn atsc1_envelope_ga94_empty_atsc_user_data_rejected() {
        // 'GA94' with no following user_data_type_code byte — A/53 Part 4
        // Table 6.8 requires the 8-bit type code field.
        let body: [u8; 0] = [];
        let raw = build_t35_atsc1(Atsc1UserIdentifier::GA94, &body);
        let outer = parse_user_data_registered_itu_t_t35(&raw).unwrap();
        let err = parse_atsc1_envelope(&outer.payload_bytes).unwrap_err();
        assert_eq!(err, SeiError::Atsc1AtscUserDataEmpty);
    }

    #[test]
    fn atsc1_envelope_dtg1_afd_flag_off() {
        // 'DTG1' + 1-byte afd_data() with active_format_flag = 0.
        // First (and only) byte = 0 0 000001 = 0x01 (zero=0,
        // active_format_flag=0, reserved=000001).
        let body = [0x01u8];
        let raw = build_t35_atsc1(Atsc1UserIdentifier::DTG1, &body);
        let outer = parse_user_data_registered_itu_t_t35(&raw).unwrap();
        let env = parse_atsc1_envelope(&outer.payload_bytes).unwrap();
        assert_eq!(env.user_identifier, Atsc1UserIdentifier::Dtg1);
        match env.body {
            Atsc1Body::Afd(afd) => {
                assert!(!afd.active_format_flag);
                assert_eq!(afd.active_format, None);
            }
            other => panic!("expected DTG1 → AFD, got {other:?}"),
        }
    }

    #[test]
    fn atsc1_envelope_dtg1_afd_flag_on_carries_active_format() {
        // 'DTG1' + 2-byte afd_data(): byte0 = 0 1 000001 = 0x41,
        // byte1 = '1111' u(4) + active_format u(4) = 0xFA (active_format =
        // 0b1010 = 10 → "16:9 letterbox image" per Table 6.14).
        let body = [0x41u8, 0xFA];
        let raw = build_t35_atsc1(Atsc1UserIdentifier::DTG1, &body);
        let outer = parse_user_data_registered_itu_t_t35(&raw).unwrap();
        let env = parse_atsc1_envelope(&outer.payload_bytes).unwrap();
        match env.body {
            Atsc1Body::Afd(afd) => {
                assert!(afd.active_format_flag);
                assert_eq!(afd.active_format, Some(0b1010));
            }
            other => panic!("expected DTG1 → AFD with active_format, got {other:?}"),
        }
    }

    #[test]
    fn atsc1_envelope_rejects_short_payload() {
        // Six bytes are the minimum (provider 2 + user_id 4).
        let err = parse_atsc1_envelope(&[0x00, 0x31, 0x47, 0x41, 0x39]).unwrap_err();
        assert_eq!(err, SeiError::Atsc1EnvelopeTooShort(5));
    }

    #[test]
    fn atsc1_envelope_rejects_wrong_provider_code() {
        // 0x003C is the HDR10+ provider slot (also under USA 0xB5) — *not*
        // the ATSC slot. The typed ATSC1 parser refuses to decode it as
        // ATSC1; callers can fall back to the raw `payload_bytes` view.
        let mut raw = vec![0x00, 0x3C];
        raw.extend_from_slice(&Atsc1UserIdentifier::GA94.to_be_bytes());
        let err = parse_atsc1_envelope(&raw).unwrap_err();
        assert_eq!(err, SeiError::Atsc1ProviderCodeMismatch(0x003C));
    }

    #[test]
    fn atsc1_envelope_rejects_unknown_user_identifier() {
        // provider_code OK, but user_identifier 'ZZZZ' is unregistered.
        let mut raw = vec![0x00, 0x31];
        raw.extend_from_slice(b"ZZZZ");
        let err = parse_atsc1_envelope(&raw).unwrap_err();
        assert_eq!(
            err,
            SeiError::Atsc1UnknownUserIdentifier(u32::from_be_bytes(*b"ZZZZ"))
        );
    }

    #[test]
    fn atsc1_envelope_chain_through_parse_payload_dispatch() {
        // End-to-end: parse_payload(4, …) → UserDataRegisteredItuTT35 →
        // parse_atsc1_envelope. Mirrors what a slice-layer SEI consumer
        // would do.
        let cc = [0xFC, 0xC4, 0x91];
        let mut body = vec![0x03u8];
        body.extend_from_slice(&cc);
        let raw = build_t35_atsc1(Atsc1UserIdentifier::GA94, &body);
        let ctx = SeiContext::default();
        let outer = match parse_payload(4, &raw, &ctx).unwrap() {
            SeiPayload::UserDataRegisteredItuTT35(o) => o,
            other => panic!("expected UserDataRegisteredItuTT35, got {other:?}"),
        };
        let env = parse_atsc1_envelope(&outer.payload_bytes).unwrap();
        assert!(matches!(
            env.body,
            Atsc1Body::AtscUserData(Atsc1UserData::CcData { .. })
        ));
    }
}
