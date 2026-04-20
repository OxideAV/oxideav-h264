//! Annex D — SEI payload parsing.
//!
//! Wraps the envelope output from `crate::non_vcl::parse_sei_rbsp`,
//! parsing the raw payload bytes into typed structs for the common
//! payload types. Each parser consumes exactly `payload_size` bytes.
//!
//! Spec cross-reference (Rec. ITU-T H.264 (08/2024)):
//!
//! | payload_type | Syntax      | Semantics   | Meaning                          |
//! | ------------ | ----------- | ----------- | -------------------------------- |
//! | 0            | §D.1.2      | §D.2.2      | buffering_period                 |
//! | 1            | §D.1.3      | §D.2.3      | pic_timing                       |
//! | 3            | §D.1.5      | §D.2.5      | filler_payload                   |
//! | 5            | §D.1.7      | §D.2.7      | user_data_unregistered           |
//! | 6            | §D.1.8      | §D.2.8      | recovery_point                   |
//! | 137          | §D.1.29     | §D.2.29     | mastering_display_colour_volume  |
//! | 144          | §D.1.31     | §D.2.35     | content_light_level_info         |

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

/// §D.2.35 — content_light_level_info (HDR10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContentLightLevelInfo {
    pub max_content_light_level: u16,
    pub max_pic_average_light_level: u16,
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

/// §D.2.35 — content_light_level_info (HDR10 metadata).
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

/// Dispatch helper: given a payload_type and bytes, produce the typed
/// payload if it's one we recognise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeiPayload {
    BufferingPeriod(BufferingPeriod),
    PicTiming(PicTiming),
    FillerPayload,
    UserDataUnregistered(UserDataUnregistered),
    RecoveryPoint(RecoveryPoint),
    MasteringDisplay(MasteringDisplayColourVolume),
    ContentLightLevel(ContentLightLevelInfo),
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
        3 => {
            parse_filler_payload(payload)?;
            Ok(SeiPayload::FillerPayload)
        }
        5 => Ok(SeiPayload::UserDataUnregistered(
            parse_user_data_unregistered(payload)?,
        )),
        6 => Ok(SeiPayload::RecoveryPoint(parse_recovery_point(payload)?)),
        137 => Ok(SeiPayload::MasteringDisplay(parse_mastering_display(
            payload,
        )?)),
        144 => Ok(SeiPayload::ContentLightLevel(parse_content_light_level(
            payload,
        )?)),
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
}
