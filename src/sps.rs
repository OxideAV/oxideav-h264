//! §7.3.2.1 — Sequence Parameter Set parsing.
//!
//! Spec-driven implementation of `seq_parameter_set_rbsp()` per ITU-T
//! Rec. H.264 (08/2024). The VUI body (§E.1) and `scaling_list()` body
//! (§7.3.2.1.1.1) are deferred to follow-up work — we record whether
//! they were signalled but don't walk their contents.
//!
//! The parser takes an already-de-emulated RBSP (`emulation_prevention_three_byte`
//! stripped by [`crate::nal::parse_nal_unit`]). The caller peels off the
//! NAL header byte before handing us the slice.

use crate::bitstream::{BitError, BitReader};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SpsError {
    #[error("bitstream read failed: {0}")]
    Bitstream(#[from] BitError),
    /// §7.3.2.1.1 — `reserved_zero_2bits` shall be equal to 0.
    #[error("reserved_zero_2bits is not 0 (got {0})")]
    ReservedNotZero(u32),
    /// §7.4.2.1.1 — `chroma_format_idc` in range 0..=3.
    #[error("chroma_format_idc out of range (got {0}, max 3)")]
    ChromaFormatIdcOutOfRange(u32),
    /// §7.4.2.1.1 — `bit_depth_luma_minus8` in range 0..=6.
    #[error("bit_depth_luma_minus8 out of range (got {0}, max 6)")]
    BitDepthLumaOutOfRange(u32),
    /// §7.4.2.1.1 — `bit_depth_chroma_minus8` in range 0..=6.
    #[error("bit_depth_chroma_minus8 out of range (got {0}, max 6)")]
    BitDepthChromaOutOfRange(u32),
    /// §7.4.2.1.1 — `log2_max_frame_num_minus4` in range 0..=12.
    #[error("log2_max_frame_num_minus4 out of range (got {0}, max 12)")]
    Log2MaxFrameNumOutOfRange(u32),
    /// §7.4.2.1.1 — `pic_order_cnt_type` in range 0..=2.
    #[error("pic_order_cnt_type out of range (got {0}, max 2)")]
    PicOrderCntTypeOutOfRange(u32),
    /// §7.4.2.1.1 — `log2_max_pic_order_cnt_lsb_minus4` in range 0..=12.
    #[error("log2_max_pic_order_cnt_lsb_minus4 out of range (got {0}, max 12)")]
    Log2MaxPocLsbOutOfRange(u32),
    /// §7.4.2.1.1 — `num_ref_frames_in_pic_order_cnt_cycle` in range 0..=255.
    #[error("num_ref_frames_in_pic_order_cnt_cycle out of range (got {0}, max 255)")]
    NumRefFramesInPocCycleOutOfRange(u32),
    /// §7.3.2.1.1 — scaling_list body is not yet implemented; SPS with
    /// `seq_scaling_matrix_present_flag == 1` cannot be fully parsed.
    #[error("seq_scaling_matrix_present_flag == 1 (scaling_list body not yet implemented)")]
    ScalingMatrixNotSupported,
    /// §7.3.2.1.1 — `seq_parameter_set_id` range 0..=31.
    #[error("seq_parameter_set_id out of range (got {0}, max 31)")]
    SpsIdOutOfRange(u32),
}

/// §7.4.2.1.1 — frame cropping offsets. Present when
/// `frame_cropping_flag == 1`. Derived `CropUnitX` / `CropUnitY` depend
/// on `ChromaArrayType` (§7.4.2.1.1 eq. 7-19 .. 7-22).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameCropping {
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
}

/// §7.3.2.1.1 — parsed `seq_parameter_set_data()`.
///
/// Fields that the spec infers (not present in the bitstream for the
/// current profile path) are stored at their inferred values, as
/// documented in §7.4.2.1.1:
/// * `chroma_format_idc` ← 1 (4:2:0) when not present,
/// * `separate_colour_plane_flag` ← 0,
/// * `bit_depth_luma_minus8` / `bit_depth_chroma_minus8` ← 0,
/// * `qpprime_y_zero_transform_bypass_flag` ← 0,
/// * `seq_scaling_matrix_present_flag` ← 0,
/// * `mb_adaptive_frame_field_flag` ← 0 when `frame_mbs_only_flag == 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sps {
    // §7.3.2.1.1
    pub profile_idc: u8,
    /// Packed bits: bit 0 = constraint_set0_flag, …, bit 5 = constraint_set5_flag.
    /// Bits 6..=7 are always 0 (they hold `reserved_zero_2bits`, which
    /// we validate to be zero during parsing).
    pub constraint_set_flags: u8,
    pub level_idc: u8,
    pub seq_parameter_set_id: u32,

    // Only present when profile_idc is one of the "chroma-extended"
    // profiles listed in §7.3.2.1.1 (100/110/122/244/44/83/86/118/128/138/139/134/135).
    pub chroma_format_idc: u32,
    pub separate_colour_plane_flag: bool,
    pub bit_depth_luma_minus8: u32,
    pub bit_depth_chroma_minus8: u32,
    pub qpprime_y_zero_transform_bypass_flag: bool,
    pub seq_scaling_matrix_present_flag: bool,

    pub log2_max_frame_num_minus4: u32,
    pub pic_order_cnt_type: u32,

    // pic_order_cnt_type == 0 branch:
    pub log2_max_pic_order_cnt_lsb_minus4: u32,

    // pic_order_cnt_type == 1 branch:
    pub delta_pic_order_always_zero_flag: bool,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub num_ref_frames_in_pic_order_cnt_cycle: u32,
    pub offset_for_ref_frame: Vec<i32>,

    // Common tail:
    pub max_num_ref_frames: u32,
    pub gaps_in_frame_num_value_allowed_flag: bool,
    pub pic_width_in_mbs_minus1: u32,
    pub pic_height_in_map_units_minus1: u32,
    pub frame_mbs_only_flag: bool,
    pub mb_adaptive_frame_field_flag: bool,
    pub direct_8x8_inference_flag: bool,
    pub frame_cropping: Option<FrameCropping>,
    pub vui_parameters_present_flag: bool,
}

impl Sps {
    /// §7.3.2.1 — parse a `seq_parameter_set_rbsp()` from a slice that
    /// holds the RBSP body (i.e. the bytes *after* the NAL header byte,
    /// with emulation prevention bytes already stripped — normally
    /// obtained from [`crate::nal::parse_nal_unit`]).
    ///
    /// On `vui_parameters_present_flag == 1` the parser stops after
    /// recording the flag; VUI parsing is deferred. Similarly the
    /// scaling_list body is not yet implemented: an SPS with
    /// `seq_scaling_matrix_present_flag == 1` returns
    /// [`SpsError::ScalingMatrixNotSupported`].
    pub fn parse(rbsp: &[u8]) -> Result<Self, SpsError> {
        let mut r = BitReader::new(rbsp);

        // §7.3.2.1.1 — profile_idc / constraint_setN_flag / reserved_zero_2bits / level_idc.
        let profile_idc = r.u(8)? as u8;
        let cs0 = r.u(1)? as u8;
        let cs1 = r.u(1)? as u8;
        let cs2 = r.u(1)? as u8;
        let cs3 = r.u(1)? as u8;
        let cs4 = r.u(1)? as u8;
        let cs5 = r.u(1)? as u8;
        // §7.3.2.1.1 — reserved_zero_2bits /* equal to 0 */.
        let reserved = r.u(2)?;
        if reserved != 0 {
            return Err(SpsError::ReservedNotZero(reserved));
        }
        let level_idc = r.u(8)? as u8;

        let constraint_set_flags = cs0
            | (cs1 << 1)
            | (cs2 << 2)
            | (cs3 << 3)
            | (cs4 << 4)
            | (cs5 << 5);

        // §7.3.2.1.1 — seq_parameter_set_id ue(v); semantics clamp to 0..=31.
        let seq_parameter_set_id = r.ue()?;
        if seq_parameter_set_id > 31 {
            return Err(SpsError::SpsIdOutOfRange(seq_parameter_set_id));
        }

        // §7.3.2.1.1 — the "chroma-extended" profile gate.
        let mut chroma_format_idc: u32 = 1; // §7.4.2.1.1 default when absent
        let mut separate_colour_plane_flag = false;
        let mut bit_depth_luma_minus8: u32 = 0;
        let mut bit_depth_chroma_minus8: u32 = 0;
        let mut qpprime_y_zero_transform_bypass_flag = false;
        let mut seq_scaling_matrix_present_flag = false;

        if matches!(
            profile_idc,
            100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
        ) {
            chroma_format_idc = r.ue()?;
            if chroma_format_idc > 3 {
                return Err(SpsError::ChromaFormatIdcOutOfRange(chroma_format_idc));
            }
            if chroma_format_idc == 3 {
                separate_colour_plane_flag = r.u(1)? == 1;
            }
            bit_depth_luma_minus8 = r.ue()?;
            if bit_depth_luma_minus8 > 6 {
                return Err(SpsError::BitDepthLumaOutOfRange(bit_depth_luma_minus8));
            }
            bit_depth_chroma_minus8 = r.ue()?;
            if bit_depth_chroma_minus8 > 6 {
                return Err(SpsError::BitDepthChromaOutOfRange(bit_depth_chroma_minus8));
            }
            qpprime_y_zero_transform_bypass_flag = r.u(1)? == 1;
            seq_scaling_matrix_present_flag = r.u(1)? == 1;
            if seq_scaling_matrix_present_flag {
                // TODO(§7.3.2.1.1.1): parse the scaling_list() body
                // (for i in 0..((chroma_format_idc != 3) ? 8 : 12)).
                // Deferred to follow-up work.
                return Err(SpsError::ScalingMatrixNotSupported);
            }
        }

        // §7.3.2.1.1 — log2_max_frame_num_minus4 ue(v); range 0..=12.
        let log2_max_frame_num_minus4 = r.ue()?;
        if log2_max_frame_num_minus4 > 12 {
            return Err(SpsError::Log2MaxFrameNumOutOfRange(log2_max_frame_num_minus4));
        }

        // §7.3.2.1.1 — pic_order_cnt_type ue(v); range 0..=2.
        let pic_order_cnt_type = r.ue()?;
        if pic_order_cnt_type > 2 {
            return Err(SpsError::PicOrderCntTypeOutOfRange(pic_order_cnt_type));
        }

        let mut log2_max_pic_order_cnt_lsb_minus4: u32 = 0;
        let mut delta_pic_order_always_zero_flag = false;
        let mut offset_for_non_ref_pic: i32 = 0;
        let mut offset_for_top_to_bottom_field: i32 = 0;
        let mut num_ref_frames_in_pic_order_cnt_cycle: u32 = 0;
        let mut offset_for_ref_frame: Vec<i32> = Vec::new();

        if pic_order_cnt_type == 0 {
            log2_max_pic_order_cnt_lsb_minus4 = r.ue()?;
            if log2_max_pic_order_cnt_lsb_minus4 > 12 {
                return Err(SpsError::Log2MaxPocLsbOutOfRange(
                    log2_max_pic_order_cnt_lsb_minus4,
                ));
            }
        } else if pic_order_cnt_type == 1 {
            delta_pic_order_always_zero_flag = r.u(1)? == 1;
            offset_for_non_ref_pic = r.se()?;
            offset_for_top_to_bottom_field = r.se()?;
            num_ref_frames_in_pic_order_cnt_cycle = r.ue()?;
            if num_ref_frames_in_pic_order_cnt_cycle > 255 {
                return Err(SpsError::NumRefFramesInPocCycleOutOfRange(
                    num_ref_frames_in_pic_order_cnt_cycle,
                ));
            }
            offset_for_ref_frame.reserve_exact(num_ref_frames_in_pic_order_cnt_cycle as usize);
            for _ in 0..num_ref_frames_in_pic_order_cnt_cycle {
                offset_for_ref_frame.push(r.se()?);
            }
        }
        // pic_order_cnt_type == 2: no extra fields.

        // §7.3.2.1.1 — common tail.
        let max_num_ref_frames = r.ue()?;
        let gaps_in_frame_num_value_allowed_flag = r.u(1)? == 1;
        let pic_width_in_mbs_minus1 = r.ue()?;
        let pic_height_in_map_units_minus1 = r.ue()?;
        let frame_mbs_only_flag = r.u(1)? == 1;
        let mb_adaptive_frame_field_flag = if !frame_mbs_only_flag {
            r.u(1)? == 1
        } else {
            false
        };
        let direct_8x8_inference_flag = r.u(1)? == 1;

        // §7.3.2.1.1 — frame_cropping_flag + four offsets.
        let frame_cropping_flag = r.u(1)? == 1;
        let frame_cropping = if frame_cropping_flag {
            let left = r.ue()?;
            let right = r.ue()?;
            let top = r.ue()?;
            let bottom = r.ue()?;
            Some(FrameCropping {
                left,
                right,
                top,
                bottom,
            })
        } else {
            None
        };

        // §7.3.2.1.1 — vui_parameters_present_flag. VUI body is deferred.
        let vui_parameters_present_flag = r.u(1)? == 1;
        // TODO(§E.1): parse vui_parameters(). Until then, if the flag
        // is set we simply stop without validating trailing bits — the
        // remainder of the RBSP may contain VUI + rbsp_trailing_bits
        // that we don't know how to walk yet.
        if !vui_parameters_present_flag {
            // §7.3.2.11 — only when we've consumed every preceding
            // syntax element can we check the trailing bits.
            // We intentionally don't require this strictly: some
            // producers pad with extra zero bytes. If the stop bit
            // isn't where we expect, don't reject — just stop.
            let _ = r.rbsp_trailing_bits();
        }

        Ok(Sps {
            profile_idc,
            constraint_set_flags,
            level_idc,
            seq_parameter_set_id,
            chroma_format_idc,
            separate_colour_plane_flag,
            bit_depth_luma_minus8,
            bit_depth_chroma_minus8,
            qpprime_y_zero_transform_bypass_flag,
            seq_scaling_matrix_present_flag,
            log2_max_frame_num_minus4,
            pic_order_cnt_type,
            log2_max_pic_order_cnt_lsb_minus4,
            delta_pic_order_always_zero_flag,
            offset_for_non_ref_pic,
            offset_for_top_to_bottom_field,
            num_ref_frames_in_pic_order_cnt_cycle,
            offset_for_ref_frame,
            max_num_ref_frames,
            gaps_in_frame_num_value_allowed_flag,
            pic_width_in_mbs_minus1,
            pic_height_in_map_units_minus1,
            frame_mbs_only_flag,
            mb_adaptive_frame_field_flag,
            direct_8x8_inference_flag,
            frame_cropping,
            vui_parameters_present_flag,
        })
    }

    /// §6.2 — `ChromaArrayType`:
    /// * `separate_colour_plane_flag == 0`: equal to `chroma_format_idc`
    /// * `separate_colour_plane_flag == 1`: equal to 0 (each plane is
    ///   coded as an independent monochrome picture).
    pub fn chroma_array_type(&self) -> u32 {
        if self.separate_colour_plane_flag {
            0
        } else {
            self.chroma_format_idc
        }
    }

    /// §7.4.2.1.1 eq. (7-13) — `PicWidthInMbs = pic_width_in_mbs_minus1 + 1`.
    pub fn pic_width_in_mbs(&self) -> u32 {
        self.pic_width_in_mbs_minus1 + 1
    }

    /// §7.4.2.1.1 eq. (7-16) — `PicHeightInMapUnits = pic_height_in_map_units_minus1 + 1`.
    pub fn pic_height_in_map_units(&self) -> u32 {
        self.pic_height_in_map_units_minus1 + 1
    }

    /// §7.4.2.1.1 eq. (7-18) — `FrameHeightInMbs = (2 - frame_mbs_only_flag) * PicHeightInMapUnits`.
    pub fn frame_height_in_mbs(&self) -> u32 {
        let factor = if self.frame_mbs_only_flag { 1 } else { 2 };
        factor * self.pic_height_in_map_units()
    }

    /// §7.4.2.1.1 eq. (7-10) — `MaxFrameNum = 2^(log2_max_frame_num_minus4 + 4)`.
    pub fn max_frame_num(&self) -> u32 {
        1u32 << (self.log2_max_frame_num_minus4 + 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny MSB-first bit writer — only for constructing test fixtures.
    /// Not part of the crate API.
    struct BitWriter {
        bytes: Vec<u8>,
        bit_pos: u8, // 0..=7, 0 = next bit goes into MSB of a fresh byte
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
            // §9.1: codeNum = value ⇒ leadingZeros = floor(log2(value + 1)),
            // followed by (value + 1) written in `leadingZeros + 1` bits.
            let v = value + 1;
            let leading = 31 - v.leading_zeros();
            for _ in 0..leading {
                self.u(1, 0);
            }
            self.u(leading + 1, v);
        }

        fn se(&mut self, value: i32) {
            // §9.1.1: k=0→0, k=1→1, k=2→-1, k=3→2, …
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

    /// Build a minimal Baseline (profile_idc=66) SPS byte-by-byte.
    /// Matches the §7.3.2.1.1 syntax table.
    fn build_baseline_sps(
        level_idc: u8,
        width_mbs: u32,
        height_mbs: u32,
        with_vui: bool,
    ) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.u(8, 66); // profile_idc — Baseline
        // constraint_set0..5_flag = 0, reserved_zero_2bits = 00
        w.u(8, 0);
        w.u(8, level_idc as u32); // level_idc
        w.ue(0); // seq_parameter_set_id
        // no chroma-extended fields for profile 66
        w.ue(0); // log2_max_frame_num_minus4
        w.ue(0); // pic_order_cnt_type
        w.ue(0); // log2_max_pic_order_cnt_lsb_minus4 (since poc_type==0)
        w.ue(1); // max_num_ref_frames
        w.u(1, 0); // gaps_in_frame_num_value_allowed_flag
        w.ue(width_mbs - 1); // pic_width_in_mbs_minus1
        w.ue(height_mbs - 1); // pic_height_in_map_units_minus1
        w.u(1, 1); // frame_mbs_only_flag
        w.u(1, 1); // direct_8x8_inference_flag
        w.u(1, 0); // frame_cropping_flag
        w.u(1, if with_vui { 1 } else { 0 }); // vui_parameters_present_flag
        if !with_vui {
            w.trailing();
        }
        w.into_bytes()
    }

    #[test]
    fn baseline_sps_320x240_level_30() {
        // 320x240 → 20 MBs wide, 15 MBs tall.
        let bytes = build_baseline_sps(30, 20, 15, /*with_vui=*/ false);
        let sps = Sps::parse(&bytes).unwrap();
        assert_eq!(sps.profile_idc, 66);
        assert_eq!(sps.constraint_set_flags, 0);
        assert_eq!(sps.level_idc, 30);
        assert_eq!(sps.seq_parameter_set_id, 0);
        // Profile 66 is not in the chroma-extended set, so these are
        // the §7.4.2.1.1 inferred defaults.
        assert_eq!(sps.chroma_format_idc, 1);
        assert!(!sps.separate_colour_plane_flag);
        assert_eq!(sps.bit_depth_luma_minus8, 0);
        assert_eq!(sps.bit_depth_chroma_minus8, 0);
        assert!(!sps.seq_scaling_matrix_present_flag);
        assert_eq!(sps.log2_max_frame_num_minus4, 0);
        assert_eq!(sps.pic_order_cnt_type, 0);
        assert_eq!(sps.log2_max_pic_order_cnt_lsb_minus4, 0);
        assert_eq!(sps.max_num_ref_frames, 1);
        assert!(!sps.gaps_in_frame_num_value_allowed_flag);
        assert_eq!(sps.pic_width_in_mbs_minus1, 19);
        assert_eq!(sps.pic_height_in_map_units_minus1, 14);
        assert!(sps.frame_mbs_only_flag);
        assert!(!sps.mb_adaptive_frame_field_flag);
        assert!(sps.direct_8x8_inference_flag);
        assert!(sps.frame_cropping.is_none());
        assert!(!sps.vui_parameters_present_flag);

        // Derived helpers.
        assert_eq!(sps.pic_width_in_mbs(), 20);
        assert_eq!(sps.pic_height_in_map_units(), 15);
        assert_eq!(sps.frame_height_in_mbs(), 15);
        assert_eq!(sps.max_frame_num(), 16);
        assert_eq!(sps.chroma_array_type(), 1);
    }

    #[test]
    fn main_profile_sps_poc_type_1() {
        // profile_idc=77 (Main); pic_order_cnt_type=1 with a short cycle.
        let mut w = BitWriter::new();
        w.u(8, 77); // profile_idc
        // constraint_set0_flag=1, rest=0; reserved_zero_2bits=0.
        w.u(1, 1);
        for _ in 0..5 {
            w.u(1, 0);
        }
        w.u(2, 0);
        w.u(8, 40); // level_idc=40
        w.ue(3); // seq_parameter_set_id=3
        w.ue(4); // log2_max_frame_num_minus4=4
        w.ue(1); // pic_order_cnt_type=1
        w.u(1, 0); // delta_pic_order_always_zero_flag=0
        w.se(-2); // offset_for_non_ref_pic
        w.se(1); // offset_for_top_to_bottom_field
        w.ue(2); // num_ref_frames_in_pic_order_cnt_cycle
        w.se(5);
        w.se(-7);
        w.ue(4); // max_num_ref_frames
        w.u(1, 1); // gaps_in_frame_num_value_allowed_flag
        w.ue(39); // 640px → 40 MBs wide → minus1 = 39
        w.ue(29); // 480px → 30 MBs tall (frame) → minus1 = 29
        w.u(1, 1); // frame_mbs_only_flag
        w.u(1, 0); // direct_8x8_inference_flag
        w.u(1, 0); // frame_cropping_flag
        w.u(1, 0); // vui_parameters_present_flag
        w.trailing();
        let bytes = w.into_bytes();

        let sps = Sps::parse(&bytes).unwrap();
        assert_eq!(sps.profile_idc, 77);
        assert_eq!(sps.constraint_set_flags, 0b0000_0001);
        assert_eq!(sps.level_idc, 40);
        assert_eq!(sps.seq_parameter_set_id, 3);
        // Profile 77 not in chroma-extended set.
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.log2_max_frame_num_minus4, 4);
        assert_eq!(sps.pic_order_cnt_type, 1);
        assert!(!sps.delta_pic_order_always_zero_flag);
        assert_eq!(sps.offset_for_non_ref_pic, -2);
        assert_eq!(sps.offset_for_top_to_bottom_field, 1);
        assert_eq!(sps.num_ref_frames_in_pic_order_cnt_cycle, 2);
        assert_eq!(sps.offset_for_ref_frame, vec![5, -7]);
        assert_eq!(sps.max_num_ref_frames, 4);
        assert!(sps.gaps_in_frame_num_value_allowed_flag);
        assert_eq!(sps.pic_width_in_mbs_minus1, 39);
        assert_eq!(sps.pic_height_in_map_units_minus1, 29);
        assert!(sps.frame_mbs_only_flag);
        assert!(!sps.direct_8x8_inference_flag);
        assert!(sps.frame_cropping.is_none());
        assert!(!sps.vui_parameters_present_flag);
    }

    #[test]
    fn high_profile_sps_includes_chroma_fields() {
        // profile_idc=100 (High) triggers the chroma_format_idc /
        // bit_depth_*_minus8 / qpprime / seq_scaling_matrix_present_flag
        // group. Use 4:2:0 (idc=1), 8-bit, no scaling matrix.
        let mut w = BitWriter::new();
        w.u(8, 100); // profile_idc=High
        // Baseline had cs* flags 0; here set cs3_flag to 1.
        w.u(1, 0);
        w.u(1, 0);
        w.u(1, 0);
        w.u(1, 1);
        w.u(1, 0);
        w.u(1, 0);
        w.u(2, 0); // reserved
        w.u(8, 41); // level_idc
        w.ue(0); // seq_parameter_set_id
        w.ue(1); // chroma_format_idc=1 (4:2:0)
        // chroma_format_idc != 3, so no separate_colour_plane_flag.
        w.ue(2); // bit_depth_luma_minus8=2 → 10-bit
        w.ue(2); // bit_depth_chroma_minus8=2
        w.u(1, 0); // qpprime_y_zero_transform_bypass_flag
        w.u(1, 0); // seq_scaling_matrix_present_flag=0
        w.ue(6); // log2_max_frame_num_minus4
        w.ue(2); // pic_order_cnt_type=2 (no extra fields)
        w.ue(5); // max_num_ref_frames
        w.u(1, 0); // gaps_in_frame_num_value_allowed_flag
        w.ue(119); // pic_width_in_mbs_minus1 → 1920px
        w.ue(67); // pic_height_in_map_units_minus1 → 1088px/16=68
        w.u(1, 1); // frame_mbs_only_flag
        w.u(1, 1); // direct_8x8_inference_flag
        // Frame cropping: crop 0/0/0/8 samples. CropUnitY=2 for 4:2:0 frame_mbs_only=1,
        // so crop_bottom_offset=4 → 8-sample crop.
        w.u(1, 1); // frame_cropping_flag
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.ue(4);
        w.u(1, 0); // vui_parameters_present_flag
        w.trailing();
        let bytes = w.into_bytes();

        let sps = Sps::parse(&bytes).unwrap();
        assert_eq!(sps.profile_idc, 100);
        assert_eq!(sps.constraint_set_flags, 0b0000_1000);
        assert_eq!(sps.level_idc, 41);
        assert_eq!(sps.chroma_format_idc, 1);
        assert!(!sps.separate_colour_plane_flag);
        assert_eq!(sps.bit_depth_luma_minus8, 2);
        assert_eq!(sps.bit_depth_chroma_minus8, 2);
        assert!(!sps.qpprime_y_zero_transform_bypass_flag);
        assert!(!sps.seq_scaling_matrix_present_flag);
        assert_eq!(sps.log2_max_frame_num_minus4, 6);
        assert_eq!(sps.pic_order_cnt_type, 2);
        // POC type 2 → no offset_for_* arrays.
        assert!(sps.offset_for_ref_frame.is_empty());
        assert_eq!(sps.max_num_ref_frames, 5);
        assert_eq!(sps.pic_width_in_mbs_minus1, 119);
        assert_eq!(sps.pic_height_in_map_units_minus1, 67);
        assert!(sps.direct_8x8_inference_flag);
        let crop = sps.frame_cropping.clone().expect("frame cropping");
        assert_eq!(crop.left, 0);
        assert_eq!(crop.right, 0);
        assert_eq!(crop.top, 0);
        assert_eq!(crop.bottom, 4);
        assert!(!sps.vui_parameters_present_flag);
    }

    #[test]
    fn vui_present_stops_parsing_gracefully() {
        let mut w = BitWriter::new();
        w.u(8, 66);
        w.u(8, 0);
        w.u(8, 30);
        w.ue(0); // sps_id
        w.ue(0); // log2_max_frame_num_minus4
        w.ue(0); // pic_order_cnt_type
        w.ue(0); // log2_max_pic_order_cnt_lsb_minus4
        w.ue(1); // max_num_ref_frames
        w.u(1, 0); // gaps
        w.ue(10); // width
        w.ue(10); // height
        w.u(1, 1); // frame_mbs_only
        w.u(1, 1); // direct_8x8
        w.u(1, 0); // no crop
        w.u(1, 1); // vui_parameters_present_flag = 1
        // Deliberately *no* trailing bits / VUI body — parser should
        // stop without touching the rest.
        let bytes = w.into_bytes();

        let sps = Sps::parse(&bytes).unwrap();
        assert!(sps.vui_parameters_present_flag);
        assert_eq!(sps.pic_width_in_mbs_minus1, 10);
    }

    #[test]
    fn reserved_zero_2bits_nonzero_is_rejected() {
        let mut w = BitWriter::new();
        w.u(8, 66); // profile_idc
        w.u(6, 0); // constraint_set0..5_flag
        w.u(2, 0b01); // reserved_zero_2bits *not* zero
        w.u(8, 30);
        let bytes = w.into_bytes();
        assert_eq!(
            Sps::parse(&bytes).unwrap_err(),
            SpsError::ReservedNotZero(0b01)
        );
    }

    #[test]
    fn scaling_matrix_present_is_rejected() {
        let mut w = BitWriter::new();
        w.u(8, 100); // High profile
        w.u(8, 0);
        w.u(8, 40);
        w.ue(0); // sps_id
        w.ue(1); // chroma_format_idc=1
        w.ue(0);
        w.ue(0);
        w.u(1, 0);
        w.u(1, 1); // seq_scaling_matrix_present_flag=1 → not yet supported
        let bytes = w.into_bytes();
        assert_eq!(
            Sps::parse(&bytes).unwrap_err(),
            SpsError::ScalingMatrixNotSupported
        );
    }

    #[test]
    fn interlaced_sps_reads_mbaff_flag() {
        let mut w = BitWriter::new();
        w.u(8, 77); // Main
        w.u(8, 0);
        w.u(8, 30);
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.ue(1);
        w.u(1, 0);
        w.ue(10);
        w.ue(10);
        w.u(1, 0); // frame_mbs_only_flag=0
        w.u(1, 1); // mb_adaptive_frame_field_flag=1
        w.u(1, 1); // direct_8x8_inference_flag
        w.u(1, 0); // frame_cropping_flag
        w.u(1, 0); // vui
        w.trailing();
        let bytes = w.into_bytes();

        let sps = Sps::parse(&bytes).unwrap();
        assert!(!sps.frame_mbs_only_flag);
        assert!(sps.mb_adaptive_frame_field_flag);
        // FrameHeightInMbs = 2 * (pic_height_in_map_units_minus1 + 1).
        assert_eq!(sps.frame_height_in_mbs(), 22);
    }

    #[test]
    fn separate_colour_plane_sets_chroma_array_type_zero() {
        let mut w = BitWriter::new();
        w.u(8, 244); // High 4:4:4 Intra — in chroma-extended set
        w.u(8, 0);
        w.u(8, 40);
        w.ue(0);
        w.ue(3); // chroma_format_idc=3 (4:4:4)
        w.u(1, 1); // separate_colour_plane_flag=1
        w.ue(0); // bit_depth_luma_minus8
        w.ue(0); // bit_depth_chroma_minus8
        w.u(1, 0); // qpprime
        w.u(1, 0); // no scaling matrix
        w.ue(0); // log2_max_frame_num_minus4
        w.ue(2); // pic_order_cnt_type=2
        w.ue(1);
        w.u(1, 0);
        w.ue(9);
        w.ue(9);
        w.u(1, 1);
        w.u(1, 1);
        w.u(1, 0);
        w.u(1, 0);
        w.trailing();
        let bytes = w.into_bytes();

        let sps = Sps::parse(&bytes).unwrap();
        assert_eq!(sps.chroma_format_idc, 3);
        assert!(sps.separate_colour_plane_flag);
        assert_eq!(sps.chroma_array_type(), 0);
    }
}
