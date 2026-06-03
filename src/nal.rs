//! §7.3.1 + §7.4.1 + §B.1 — NAL unit parsing and byte-stream framing.
//!
//! H.264 wraps every syntactic structure (parameter sets, slices,
//! SEI, …) in a *NAL unit* — a 1-byte header plus an RBSP body. This
//! module provides:
//!
//! * [`split_annex_b`] — Annex B byte-stream splitter (looks for the
//!   `0x000001` / `0x00000001` start code prefix).
//! * [`split_avcc`] — length-prefixed AVCC framing (used inside
//!   `avcC` boxes / MKV `V_MPEG4/ISO/AVC` track private data).
//! * [`parse_nal_unit`] — parse the 1-byte NAL header (§7.4.1) and
//!   strip the `emulation_prevention_three_byte`s (§7.4.1.1) to expose
//!   the RBSP payload.
//!
//! The splitters yield raw NAL bytes — the leading byte is still the
//! NAL header. Pass that slice to [`parse_nal_unit`] to get a typed
//! [`NalHeader`] plus a `Cow<[u8]>` RBSP. The Cow is borrowed when no
//! emulation prevention bytes were present (the common case), and
//! owned `Vec<u8>` when at least one had to be stripped.

use std::borrow::Cow;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum NalError {
    #[error("NAL unit is empty (no header byte)")]
    EmptyNal,
    #[error("forbidden_zero_bit is not 0 (got {0})")]
    ForbiddenZeroBitNotZero(u8),
    #[error("AVCC length prefix is truncated")]
    TruncatedAvccLength,
    #[error("AVCC NAL payload is shorter than the declared length")]
    TruncatedAvccPayload,
    #[error("AVCC nalu_length_size must be 1, 2, or 4 (got {0})")]
    InvalidAvccLengthSize(u8),
    #[error("NAL extension header is truncated (need {need} bytes after type {nut}, got {got})")]
    TruncatedExtensionHeader { nut: u8, need: usize, got: usize },
}

/// Per §7.4.1 + Table 7-1. The variants cover every value in the spec
/// table; the `Reserved`/`Unspecified` variants carry the raw 5-bit
/// type so callers can still log/log-and-skip what they got.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NalUnitType {
    /// 0 — Unspecified.
    Unspecified(u8),
    /// 1 — Coded slice of a non-IDR picture (`slice_layer_without_partitioning_rbsp`).
    SliceNonIdr,
    /// 2 — Coded slice data partition A.
    SliceDataPartitionA,
    /// 3 — Coded slice data partition B.
    SliceDataPartitionB,
    /// 4 — Coded slice data partition C.
    SliceDataPartitionC,
    /// 5 — Coded slice of an IDR picture.
    SliceIdr,
    /// 6 — Supplemental enhancement information (`sei_rbsp`).
    Sei,
    /// 7 — Sequence parameter set (`seq_parameter_set_rbsp`).
    Sps,
    /// 8 — Picture parameter set (`pic_parameter_set_rbsp`).
    Pps,
    /// 9 — Access unit delimiter (`access_unit_delimiter_rbsp`).
    AccessUnitDelimiter,
    /// 10 — End of sequence (`end_of_seq_rbsp`).
    EndOfSequence,
    /// 11 — End of stream (`end_of_stream_rbsp`).
    EndOfStream,
    /// 12 — Filler data (`filler_data_rbsp`).
    FillerData,
    /// 13 — Sequence parameter set extension.
    SpsExtension,
    /// 14 — Prefix NAL unit (Annex F SVC / G MVC).
    PrefixNalUnit,
    /// 15 — Subset sequence parameter set.
    SubsetSps,
    /// 16 — Depth parameter set (Annex H/I).
    DepthParameterSet,
    /// 17..18, 22..23 — reserved values.
    Reserved(u8),
    /// 19 — Coded slice of an auxiliary coded picture without partitioning.
    SliceAuxiliary,
    /// 20 — Coded slice extension (Annex F SVC / G MVC).
    SliceExtension,
    /// 21 — Coded slice extension for a depth view component or a 3D-AVC texture view.
    SliceExtensionDepth,
    /// 24..31 — Unspecified.
    UnspecifiedRange(u8),
}

impl NalUnitType {
    pub fn from_u8(v: u8) -> Self {
        match v & 0x1F {
            0 => NalUnitType::Unspecified(0),
            1 => NalUnitType::SliceNonIdr,
            2 => NalUnitType::SliceDataPartitionA,
            3 => NalUnitType::SliceDataPartitionB,
            4 => NalUnitType::SliceDataPartitionC,
            5 => NalUnitType::SliceIdr,
            6 => NalUnitType::Sei,
            7 => NalUnitType::Sps,
            8 => NalUnitType::Pps,
            9 => NalUnitType::AccessUnitDelimiter,
            10 => NalUnitType::EndOfSequence,
            11 => NalUnitType::EndOfStream,
            12 => NalUnitType::FillerData,
            13 => NalUnitType::SpsExtension,
            14 => NalUnitType::PrefixNalUnit,
            15 => NalUnitType::SubsetSps,
            16 => NalUnitType::DepthParameterSet,
            17 | 18 | 22 | 23 => NalUnitType::Reserved(v & 0x1F),
            19 => NalUnitType::SliceAuxiliary,
            20 => NalUnitType::SliceExtension,
            21 => NalUnitType::SliceExtensionDepth,
            24..=31 => NalUnitType::UnspecifiedRange(v & 0x1F),
            _ => unreachable!(),
        }
    }

    pub fn as_u8(self) -> u8 {
        match self {
            NalUnitType::Unspecified(v) => v,
            NalUnitType::SliceNonIdr => 1,
            NalUnitType::SliceDataPartitionA => 2,
            NalUnitType::SliceDataPartitionB => 3,
            NalUnitType::SliceDataPartitionC => 4,
            NalUnitType::SliceIdr => 5,
            NalUnitType::Sei => 6,
            NalUnitType::Sps => 7,
            NalUnitType::Pps => 8,
            NalUnitType::AccessUnitDelimiter => 9,
            NalUnitType::EndOfSequence => 10,
            NalUnitType::EndOfStream => 11,
            NalUnitType::FillerData => 12,
            NalUnitType::SpsExtension => 13,
            NalUnitType::PrefixNalUnit => 14,
            NalUnitType::SubsetSps => 15,
            NalUnitType::DepthParameterSet => 16,
            NalUnitType::Reserved(v) => v,
            NalUnitType::SliceAuxiliary => 19,
            NalUnitType::SliceExtension => 20,
            NalUnitType::SliceExtensionDepth => 21,
            NalUnitType::UnspecifiedRange(v) => v,
        }
    }

    /// VCL NAL unit per Annex A column of Table 7-1.
    pub fn is_vcl(self) -> bool {
        matches!(
            self,
            NalUnitType::SliceNonIdr
                | NalUnitType::SliceDataPartitionA
                | NalUnitType::SliceDataPartitionB
                | NalUnitType::SliceDataPartitionC
                | NalUnitType::SliceIdr
        )
    }

    /// IDR picture flag (eq. 7-1: `IdrPicFlag = (nal_unit_type == 5) ? 1 : 0`).
    pub fn is_idr(self) -> bool {
        matches!(self, NalUnitType::SliceIdr)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NalHeader {
    pub forbidden_zero_bit: u8,
    pub nal_ref_idc: u8,
    pub nal_unit_type: NalUnitType,
}

/// §G.7.3.1.1 — MVC NAL unit header extension.
///
/// Three bytes laid out (MSB-first inside each byte) as:
///
/// * `svc_extension_flag` (u(1)) — leading bit of byte 1, equal to 0
///   when the trailing 23 bits encode the MVC extension; conceptually
///   belongs to the §7.3.1 dispatch but is stored here so the raw
///   three-byte slice round-trips.
/// * `non_idr_flag` u(1)
/// * `priority_id`  u(6)
/// * `view_id`      u(10)
/// * `temporal_id`  u(3)
/// * `anchor_pic_flag`   u(1)
/// * `inter_view_flag`   u(1)
/// * `reserved_one_bit`  u(1)
///
/// The MVC view-id field is 10 bits per §G.7.4.1.1, so the parsed
/// struct widens it to `u16`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NalUnitHeaderMvcExtension {
    /// `non_idr_flag` — 0 ⇒ this NAL belongs to an IDR view component.
    pub non_idr_flag: u8,
    /// `priority_id` — bitstream extraction priority (0 = highest).
    pub priority_id: u8,
    /// `view_id` — distinguishes view components in an MVC bitstream.
    pub view_id: u16,
    /// `temporal_id` — temporal layer identifier (0 = base layer).
    pub temporal_id: u8,
    /// `anchor_pic_flag` — 1 ⇒ part of an anchor access unit.
    pub anchor_pic_flag: u8,
    /// `inter_view_flag` — 1 ⇒ may be used for inter-view prediction.
    pub inter_view_flag: u8,
    /// `reserved_one_bit` — fixed to 1 by §G.7.4.1.1.
    pub reserved_one_bit: u8,
}

/// §F.7.3.1.1 — SVC NAL unit header extension. Three bytes total
/// (one of which is the leading `svc_extension_flag = 1`).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NalUnitHeaderSvcExtension {
    /// `idr_flag` — 1 ⇒ IDR layer representation.
    pub idr_flag: u8,
    /// `priority_id` u(6).
    pub priority_id: u8,
    /// `no_inter_layer_pred_flag` u(1).
    pub no_inter_layer_pred_flag: u8,
    /// `dependency_id` u(3).
    pub dependency_id: u8,
    /// `quality_id` u(4).
    pub quality_id: u8,
    /// `temporal_id` u(3).
    pub temporal_id: u8,
    /// `use_ref_base_pic_flag` u(1).
    pub use_ref_base_pic_flag: u8,
    /// `discardable_flag` u(1).
    pub discardable_flag: u8,
    /// `output_flag` u(1).
    pub output_flag: u8,
    /// `reserved_three_2bits` u(2) — fixed to 3 by §F.7.4.1.1.
    pub reserved_three_2bits: u8,
}

/// §I.7.3.1.1 — 3D-AVC NAL unit header extension. Two bytes total
/// (one of which is the leading `avc_3d_extension_flag = 1`).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NalUnitHeader3dAvcExtension {
    /// `view_idx` u(8).
    pub view_idx: u8,
    /// `depth_flag` u(1).
    pub depth_flag: u8,
    /// `non_idr_flag` u(1).
    pub non_idr_flag: u8,
    /// `temporal_id` u(3).
    pub temporal_id: u8,
    /// `anchor_pic_flag` u(1).
    pub anchor_pic_flag: u8,
    /// `inter_view_flag` u(1).
    pub inter_view_flag: u8,
}

/// Discriminated extension header for §7.3.1 `nal_unit_type ∈ {14, 20, 21}`.
///
/// Per §7.3.1: for type 14 or 20, the next bit is `svc_extension_flag`
/// — 1 ⇒ SVC (Annex F), 0 ⇒ MVC (Annex G). For type 21 the same bit
/// is renamed `avc_3d_extension_flag` — 1 ⇒ 3D-AVC (Annex I), 0 ⇒ MVC.
/// The MVC/SVC branches consume 3 extension bytes; the 3D-AVC branch
/// consumes 2.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NalUnitHeaderExtension {
    Mvc(NalUnitHeaderMvcExtension),
    Svc(NalUnitHeaderSvcExtension),
    Avc3d(NalUnitHeader3dAvcExtension),
}

#[derive(Debug, Clone)]
pub struct NalUnit<'a> {
    pub header: NalHeader,
    /// Parsed §G/F/I.7.3.1.1 extension header for NAL types 14/20/21,
    /// `None` for all other types. The extension bytes are stripped
    /// from `rbsp` — only the post-extension RBSP body is returned.
    pub extension: Option<NalUnitHeaderExtension>,
    pub rbsp: Cow<'a, [u8]>,
}

/// Parse §7.3.1 `nal_unit()` from the raw NAL bytes (no start code, no
/// length prefix — just the NAL header byte followed by the encoded
/// payload). Strips emulation prevention bytes per §7.4.1.1.
///
/// For extension NAL types (14/20/21) the §7.3.1 dispatch reads the
/// `svc_extension_flag` / `avc_3d_extension_flag` bit and then either
/// 3 bytes (SVC §F.7.3.1.1 or MVC §G.7.3.1.1) or 2 bytes (3D-AVC
/// §I.7.3.1.1) of extension header. Per §7.3.1 the emulation-prevention
/// scan starts AFTER `nalUnitHeaderBytes`, so the extension bytes are
/// not subject to it. The returned `rbsp` field excludes the extension
/// bytes, and the parsed structure is exposed via [`NalUnit::extension`].
pub fn parse_nal_unit(payload: &[u8]) -> Result<NalUnit<'_>, NalError> {
    if payload.is_empty() {
        return Err(NalError::EmptyNal);
    }
    let header_byte = payload[0];
    let forbidden_zero_bit = (header_byte >> 7) & 1;
    if forbidden_zero_bit != 0 {
        return Err(NalError::ForbiddenZeroBitNotZero(forbidden_zero_bit));
    }
    let header = NalHeader {
        forbidden_zero_bit,
        nal_ref_idc: (header_byte >> 5) & 3,
        nal_unit_type: NalUnitType::from_u8(header_byte),
    };
    let raw_after_header = &payload[1..];
    let (extension, ext_bytes) = match header.nal_unit_type {
        NalUnitType::PrefixNalUnit | NalUnitType::SliceExtension => {
            // §7.3.1: read svc_extension_flag, then 3 extension bytes
            // (SVC if 1, MVC if 0).
            parse_extension_14_or_20(raw_after_header, header.nal_unit_type.as_u8())?
        }
        NalUnitType::SliceExtensionDepth => {
            // §7.3.1: read avc_3d_extension_flag, then either 2 bytes
            // (3D-AVC) or 3 bytes (MVC).
            parse_extension_21(raw_after_header)?
        }
        _ => (None, 0),
    };
    let rbsp = rbsp_from_nal_payload(&raw_after_header[ext_bytes..]);
    Ok(NalUnit {
        header,
        extension,
        rbsp,
    })
}

/// §7.3.1 + §F.7.3.1.1 + §G.7.3.1.1 — read the leading
/// `svc_extension_flag` and then 3 extension bytes for NAL types
/// 14 (PrefixNalUnit) or 20 (SliceExtension).
fn parse_extension_14_or_20(
    raw: &[u8],
    nut: u8,
) -> Result<(Option<NalUnitHeaderExtension>, usize), NalError> {
    if raw.len() < 3 {
        return Err(NalError::TruncatedExtensionHeader {
            nut,
            need: 3,
            got: raw.len(),
        });
    }
    // Concatenate the three header bytes into a single 24-bit big-endian
    // integer so the bit-packed fields can be sliced by shift+mask.
    let b0 = raw[0] as u32;
    let b1 = raw[1] as u32;
    let b2 = raw[2] as u32;
    let bits24 = (b0 << 16) | (b1 << 8) | b2;
    // Top bit of byte 0 is svc_extension_flag (§7.3.1).
    let svc_extension_flag = (bits24 >> 23) & 0x1;
    let ext = if svc_extension_flag == 1 {
        // §F.7.3.1.1 — 23-bit packed body after the flag.
        NalUnitHeaderExtension::Svc(parse_svc_body(bits24))
    } else {
        // §G.7.3.1.1 — 23-bit packed body after the flag.
        NalUnitHeaderExtension::Mvc(parse_mvc_body(bits24))
    };
    Ok((Some(ext), 3))
}

/// §7.3.1 + §I.7.3.1.1 — read `avc_3d_extension_flag`; on 1 consume 2
/// bytes (3D-AVC), on 0 consume 3 bytes (MVC). NAL type 21
/// (SliceExtensionDepth) is the only caller.
fn parse_extension_21(raw: &[u8]) -> Result<(Option<NalUnitHeaderExtension>, usize), NalError> {
    if raw.is_empty() {
        return Err(NalError::TruncatedExtensionHeader {
            nut: 21,
            need: 2,
            got: 0,
        });
    }
    let flag = (raw[0] >> 7) & 1;
    if flag == 1 {
        // §I.7.3.1.1 — 2-byte 3D-AVC extension.
        if raw.len() < 2 {
            return Err(NalError::TruncatedExtensionHeader {
                nut: 21,
                need: 2,
                got: raw.len(),
            });
        }
        let bits16 = ((raw[0] as u32) << 8) | raw[1] as u32;
        let ext = NalUnitHeaderExtension::Avc3d(parse_3davc_body(bits16));
        Ok((Some(ext), 2))
    } else {
        // §G.7.3.1.1 — MVC body, same shape as type 14/20 branch.
        if raw.len() < 3 {
            return Err(NalError::TruncatedExtensionHeader {
                nut: 21,
                need: 3,
                got: raw.len(),
            });
        }
        let bits24 = ((raw[0] as u32) << 16) | ((raw[1] as u32) << 8) | raw[2] as u32;
        let ext = NalUnitHeaderExtension::Mvc(parse_mvc_body(bits24));
        Ok((Some(ext), 3))
    }
}

/// §G.7.3.1.1 — decode the 23-bit MVC body from the top of a 24-bit
/// word whose MSB is the leading `svc_extension_flag` (already known
/// to be 0). Field layout (MSB-first): `non_idr_flag(1) | priority_id(6)
/// | view_id(10) | temporal_id(3) | anchor_pic_flag(1) |
/// inter_view_flag(1) | reserved_one_bit(1)`.
fn parse_mvc_body(bits24: u32) -> NalUnitHeaderMvcExtension {
    NalUnitHeaderMvcExtension {
        non_idr_flag: ((bits24 >> 22) & 0x1) as u8,
        priority_id: ((bits24 >> 16) & 0x3F) as u8,
        view_id: ((bits24 >> 6) & 0x3FF) as u16,
        temporal_id: ((bits24 >> 3) & 0x7) as u8,
        anchor_pic_flag: ((bits24 >> 2) & 0x1) as u8,
        inter_view_flag: ((bits24 >> 1) & 0x1) as u8,
        reserved_one_bit: (bits24 & 0x1) as u8,
    }
}

/// §F.7.3.1.1 — decode the 23-bit SVC body from the top of a 24-bit
/// word whose MSB is the leading `svc_extension_flag` (already known
/// to be 1). Field layout (MSB-first): `idr_flag(1) | priority_id(6) |
/// no_inter_layer_pred_flag(1) | dependency_id(3) | quality_id(4) |
/// temporal_id(3) | use_ref_base_pic_flag(1) | discardable_flag(1) |
/// output_flag(1) | reserved_three_2bits(2)`.
fn parse_svc_body(bits24: u32) -> NalUnitHeaderSvcExtension {
    NalUnitHeaderSvcExtension {
        idr_flag: ((bits24 >> 22) & 0x1) as u8,
        priority_id: ((bits24 >> 16) & 0x3F) as u8,
        no_inter_layer_pred_flag: ((bits24 >> 15) & 0x1) as u8,
        dependency_id: ((bits24 >> 12) & 0x7) as u8,
        quality_id: ((bits24 >> 8) & 0xF) as u8,
        temporal_id: ((bits24 >> 5) & 0x7) as u8,
        use_ref_base_pic_flag: ((bits24 >> 4) & 0x1) as u8,
        discardable_flag: ((bits24 >> 3) & 0x1) as u8,
        output_flag: ((bits24 >> 2) & 0x1) as u8,
        reserved_three_2bits: (bits24 & 0x3) as u8,
    }
}

/// §I.7.3.1.1 — decode the 15-bit 3D-AVC body from the top of a 16-bit
/// word whose MSB is the leading `avc_3d_extension_flag` (already
/// known to be 1). Field layout (MSB-first): `view_idx(8) |
/// depth_flag(1) | non_idr_flag(1) | temporal_id(3) | anchor_pic_flag(1)
/// | inter_view_flag(1)`.
fn parse_3davc_body(bits16: u32) -> NalUnitHeader3dAvcExtension {
    NalUnitHeader3dAvcExtension {
        view_idx: ((bits16 >> 7) & 0xFF) as u8,
        depth_flag: ((bits16 >> 6) & 0x1) as u8,
        non_idr_flag: ((bits16 >> 5) & 0x1) as u8,
        temporal_id: ((bits16 >> 2) & 0x7) as u8,
        anchor_pic_flag: ((bits16 >> 1) & 0x1) as u8,
        inter_view_flag: (bits16 & 0x1) as u8,
    }
}

/// Strip `emulation_prevention_three_byte` (0x03) per §7.4.1.1. The
/// rule: anywhere a `00 00 03` triplet appears with at least one byte
/// remaining, output the `00 00` and drop the `03`.
///
/// Hot-path: scans once for any `00 00 03`; if absent, returns the
/// input slice borrowed (no allocation). Otherwise copies into a new
/// `Vec` with the 0x03 bytes stripped.
pub fn rbsp_from_nal_payload(nal_payload: &[u8]) -> Cow<'_, [u8]> {
    if !contains_emulation_byte(nal_payload) {
        return Cow::Borrowed(nal_payload);
    }
    let mut out = Vec::with_capacity(nal_payload.len());
    let mut i = 0;
    while i < nal_payload.len() {
        // §7.3.1: `i + 2 < N && next_bits(24) == 0x000003` ⇒ keep the
        // two leading zeros, skip the 0x03 emulation byte.
        if i + 2 < nal_payload.len()
            && nal_payload[i] == 0
            && nal_payload[i + 1] == 0
            && nal_payload[i + 2] == 0x03
        {
            out.push(0);
            out.push(0);
            i += 3;
        } else {
            out.push(nal_payload[i]);
            i += 1;
        }
    }
    Cow::Owned(out)
}

fn contains_emulation_byte(bytes: &[u8]) -> bool {
    let mut i = 0;
    while i + 2 < bytes.len() {
        if bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 0x03 {
            return true;
        }
        i += 1;
    }
    false
}

/// §B.1 — Annex B byte-stream splitter.
///
/// Yields each NAL unit's bytes (excluding the start-code prefix and
/// any trailing-zero padding bytes). Handles both 3-byte (`0x00 00 01`)
/// and 4-byte (`0x00 00 00 01`) start code prefixes; the 4-byte form
/// is just `leading_zero_8bits` (§B.1) followed by the 3-byte prefix.
///
/// Tolerant of leading zeros and trailing zeros around each NAL.
/// Returns no error on malformed input — anything that doesn't contain
/// a start code yields nothing.
pub struct AnnexBSplitter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> AnnexBSplitter<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl<'a> Iterator for AnnexBSplitter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        // Find the next start-code prefix from `self.pos`.
        let start_offset = find_start_code_prefix(&self.data[self.pos..])?;
        let prefix_pos = self.pos + start_offset;
        let after_prefix = prefix_pos + 3;
        // Find the start of the next prefix to bound this NAL.
        let next_prefix = find_start_code_prefix(&self.data[after_prefix..])
            .map(|off| after_prefix + off)
            .unwrap_or(self.data.len());
        // §B.1 trailing_zero_8bits — strip trailing zeros so the NAL
        // ends at its real last byte. Also strip the optional zero byte
        // that turns a 3-byte prefix into the canonical 4-byte form
        // for the *next* NAL (we'll find that prefix again next time).
        let mut nal_end = next_prefix;
        while nal_end > after_prefix && self.data[nal_end - 1] == 0 {
            nal_end -= 1;
        }
        self.pos = next_prefix;
        Some(&self.data[after_prefix..nal_end])
    }
}

fn find_start_code_prefix(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 3 {
        return None;
    }
    let mut i = 0;
    while i + 2 < bytes.len() {
        if bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 1 {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// AVCC / `avcC` length-prefixed framing. `nalu_length_size` is the
/// number of bytes per length field (1, 2, or 4) — taken from
/// `lengthSizeMinusOne + 1` in the AVCDecoderConfigurationRecord.
///
/// Each iteration returns one NAL unit's bytes, or an error if the
/// stream is truncated.
#[derive(Debug)]
pub struct AvccSplitter<'a> {
    data: &'a [u8],
    pos: usize,
    nalu_length_size: u8,
}

impl<'a> AvccSplitter<'a> {
    pub fn new(data: &'a [u8], nalu_length_size: u8) -> Result<Self, NalError> {
        if !matches!(nalu_length_size, 1 | 2 | 4) {
            return Err(NalError::InvalidAvccLengthSize(nalu_length_size));
        }
        Ok(Self {
            data,
            pos: 0,
            nalu_length_size,
        })
    }
}

impl<'a> Iterator for AvccSplitter<'a> {
    type Item = Result<&'a [u8], NalError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }
        let n = self.nalu_length_size as usize;
        if self.pos + n > self.data.len() {
            return Some(Err(NalError::TruncatedAvccLength));
        }
        let mut len: usize = 0;
        for i in 0..n {
            len = (len << 8) | self.data[self.pos + i] as usize;
        }
        self.pos += n;
        if self.pos + len > self.data.len() {
            return Some(Err(NalError::TruncatedAvccPayload));
        }
        let payload = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Some(Ok(payload))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nal_header_parses_sps_byte() {
        // 0x67 = 0b 0 11 00111 → forbidden=0, nal_ref_idc=3, type=7 (SPS)
        let nu = parse_nal_unit(&[0x67, 0x42, 0x00, 0x1e]).unwrap();
        assert_eq!(nu.header.forbidden_zero_bit, 0);
        assert_eq!(nu.header.nal_ref_idc, 3);
        assert_eq!(nu.header.nal_unit_type, NalUnitType::Sps);
        assert_eq!(&*nu.rbsp, &[0x42, 0x00, 0x1e]);
    }

    #[test]
    fn nal_header_parses_idr_slice() {
        // 0x65 = 0b 0 11 00101 → forbidden=0, nal_ref_idc=3, type=5 (IDR)
        let nu = parse_nal_unit(&[0x65, 0x88]).unwrap();
        assert_eq!(nu.header.nal_unit_type, NalUnitType::SliceIdr);
        assert!(nu.header.nal_unit_type.is_vcl());
        assert!(nu.header.nal_unit_type.is_idr());
    }

    #[test]
    fn nal_header_parses_non_idr_slice() {
        // 0x41 = 0b 0 10 00001 → forbidden=0, nal_ref_idc=2, type=1 (non-IDR slice)
        let nu = parse_nal_unit(&[0x41, 0x9a]).unwrap();
        assert_eq!(nu.header.nal_ref_idc, 2);
        assert_eq!(nu.header.nal_unit_type, NalUnitType::SliceNonIdr);
        assert!(nu.header.nal_unit_type.is_vcl());
        assert!(!nu.header.nal_unit_type.is_idr());
    }

    #[test]
    fn nal_header_rejects_forbidden_zero_bit_set() {
        // 0xE7 = 0b 1 11 00111 → forbidden_zero_bit = 1
        assert_eq!(
            parse_nal_unit(&[0xe7, 0x00]).unwrap_err(),
            NalError::ForbiddenZeroBitNotZero(1)
        );
    }

    #[test]
    fn nal_header_rejects_empty_input() {
        assert_eq!(parse_nal_unit(&[]).unwrap_err(), NalError::EmptyNal);
    }

    #[test]
    fn nal_unit_type_round_trip_for_table_7_1() {
        // Every value 0..=31 must round-trip through from_u8 / as_u8.
        for v in 0u8..=31 {
            let t = NalUnitType::from_u8(v);
            assert_eq!(t.as_u8(), v, "round-trip failed for nal_unit_type {}", v);
        }
    }

    #[test]
    fn rbsp_no_emulation_borrows() {
        let input: &[u8] = &[0x00, 0x01, 0x02, 0x04];
        let cow = rbsp_from_nal_payload(input);
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(&*cow, input);
    }

    #[test]
    fn rbsp_strips_emulation_byte_in_middle() {
        // 00 00 03 04 → 00 00 04
        let cow = rbsp_from_nal_payload(&[0x00, 0x00, 0x03, 0x04]);
        assert!(matches!(cow, Cow::Owned(_)));
        assert_eq!(&*cow, &[0x00, 0x00, 0x04]);
    }

    #[test]
    fn rbsp_strips_multiple_emulation_bytes() {
        // 00 00 03 00 00 03 01 → 00 00 00 00 01
        let cow = rbsp_from_nal_payload(&[0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x01]);
        assert_eq!(&*cow, &[0x00, 0x00, 0x00, 0x00, 0x01]);
    }

    #[test]
    fn rbsp_keeps_unrelated_03_bytes() {
        // 03 alone or after non-zero is preserved.
        let cow = rbsp_from_nal_payload(&[0x03, 0x00, 0x03, 0x05]);
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(&*cow, &[0x03, 0x00, 0x03, 0x05]);
    }

    #[test]
    fn annex_b_splits_three_byte_start_codes() {
        // [00 00 01] AA BB [00 00 01] CC DD
        let stream = [0x00, 0x00, 0x01, 0xaa, 0xbb, 0x00, 0x00, 0x01, 0xcc, 0xdd];
        let nals: Vec<_> = AnnexBSplitter::new(&stream).collect();
        assert_eq!(nals, vec![&[0xaa, 0xbb][..], &[0xcc, 0xdd][..]]);
    }

    #[test]
    fn annex_b_splits_four_byte_start_codes() {
        // 00 [00 00 01] AA BB 00 [00 00 01] CC DD
        let stream = [
            0x00, 0x00, 0x00, 0x01, 0xaa, 0xbb, 0x00, 0x00, 0x00, 0x01, 0xcc, 0xdd,
        ];
        let nals: Vec<_> = AnnexBSplitter::new(&stream).collect();
        assert_eq!(nals, vec![&[0xaa, 0xbb][..], &[0xcc, 0xdd][..]]);
    }

    #[test]
    fn annex_b_handles_leading_zeros_then_three_byte() {
        // 00 00 00 00 [00 00 01] AA BB
        let stream = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0xaa, 0xbb];
        let nals: Vec<_> = AnnexBSplitter::new(&stream).collect();
        assert_eq!(nals, vec![&[0xaa, 0xbb][..]]);
    }

    #[test]
    fn annex_b_strips_trailing_zeros() {
        // [00 00 01] AA BB 00 00 [00 00 01] CC
        let stream = [
            0x00, 0x00, 0x01, 0xaa, 0xbb, 0x00, 0x00, 0x00, 0x00, 0x01, 0xcc,
        ];
        let nals: Vec<_> = AnnexBSplitter::new(&stream).collect();
        assert_eq!(nals, vec![&[0xaa, 0xbb][..], &[0xcc][..]]);
    }

    #[test]
    fn annex_b_empty_yields_nothing() {
        let nals: Vec<_> = AnnexBSplitter::new(&[]).collect();
        assert!(nals.is_empty());
        let nals: Vec<_> = AnnexBSplitter::new(&[0x00, 0x00, 0x00]).collect();
        assert!(nals.is_empty());
    }

    #[test]
    fn avcc_splits_with_4_byte_length() {
        // 4-byte length prefix: 0x0000_0002 then [AA BB], 0x0000_0003 then [CC DD EE]
        let stream = [
            0x00, 0x00, 0x00, 0x02, 0xaa, 0xbb, 0x00, 0x00, 0x00, 0x03, 0xcc, 0xdd, 0xee,
        ];
        let nals: Vec<_> = AvccSplitter::new(&stream, 4)
            .unwrap()
            .map(|r| r.unwrap().to_vec())
            .collect();
        assert_eq!(nals, vec![vec![0xaa, 0xbb], vec![0xcc, 0xdd, 0xee]]);
    }

    #[test]
    fn avcc_truncated_payload_errors() {
        // Length 0x05 but only 2 bytes of payload.
        let stream = [0x00, 0x00, 0x00, 0x05, 0xaa, 0xbb];
        let mut split = AvccSplitter::new(&stream, 4).unwrap();
        assert_eq!(
            split.next().unwrap().unwrap_err(),
            NalError::TruncatedAvccPayload
        );
    }

    #[test]
    fn avcc_truncated_length_errors() {
        // Less than 4 bytes left for the length.
        let stream = [0x00, 0x00];
        let mut split = AvccSplitter::new(&stream, 4).unwrap();
        assert_eq!(
            split.next().unwrap().unwrap_err(),
            NalError::TruncatedAvccLength
        );
    }

    #[test]
    fn avcc_rejects_invalid_length_size() {
        assert_eq!(
            AvccSplitter::new(&[], 3).unwrap_err(),
            NalError::InvalidAvccLengthSize(3)
        );
    }

    #[test]
    fn round_trip_annex_b_then_parse_with_emulation() {
        // SPS NAL containing an emulation prevention byte.
        // header 0x67, then RBSP 0x00 0x00 0x05 (would otherwise look
        // like a start code prefix tail) → encoded as 0x00 0x00 0x03 0x05.
        let stream = [0x00, 0x00, 0x01, 0x67, 0x00, 0x00, 0x03, 0x05];
        let mut split = AnnexBSplitter::new(&stream);
        let nal_bytes = split.next().unwrap();
        assert_eq!(nal_bytes, &[0x67, 0x00, 0x00, 0x03, 0x05]);
        let nu = parse_nal_unit(nal_bytes).unwrap();
        assert_eq!(nu.header.nal_unit_type, NalUnitType::Sps);
        assert_eq!(&*nu.rbsp, &[0x00, 0x00, 0x05]);
        // Non-extension NALs surface no extension header.
        assert!(nu.extension.is_none());
    }

    // -------- §G.7.3.1.1 MVC NAL extension header --------

    /// Helper: pack a 24-bit MVC extension header (leading
    /// svc_extension_flag = 0) into three bytes.
    fn pack_mvc_ext(
        non_idr: u8,
        priority: u8,
        view: u16,
        temporal: u8,
        anchor: u8,
        inter_view: u8,
        reserved: u8,
    ) -> [u8; 3] {
        // svc_extension_flag = 0 sits in bit 23 (MSB of byte 0) — left
        // out of the OR so clippy's identity_op rule is satisfied.
        let bits24: u32 = ((non_idr as u32 & 0x1) << 22)
            | ((priority as u32 & 0x3F) << 16)
            | ((view as u32 & 0x3FF) << 6)
            | ((temporal as u32 & 0x7) << 3)
            | ((anchor as u32 & 0x1) << 2)
            | ((inter_view as u32 & 0x1) << 1)
            | (reserved as u32 & 0x1);
        [
            ((bits24 >> 16) & 0xFF) as u8,
            ((bits24 >> 8) & 0xFF) as u8,
            (bits24 & 0xFF) as u8,
        ]
    }

    #[test]
    fn mvc_extension_parses_via_prefix_nal_14() {
        // §G.7.3.1.1 — prefix NAL with MVC extension. view_id widens
        // to u16 since it is 10 bits wide.
        let ext = pack_mvc_ext(1, 0, 0, 0, 1, 1, 1);
        let mut nal = vec![0x0E]; // type = 14, nal_ref_idc = 0
        nal.extend_from_slice(&ext);
        nal.extend_from_slice(&[0xAA, 0xBB]); // RBSP body
        let nu = parse_nal_unit(&nal).unwrap();
        assert_eq!(nu.header.nal_unit_type, NalUnitType::PrefixNalUnit);
        match nu.extension {
            Some(NalUnitHeaderExtension::Mvc(m)) => {
                assert_eq!(m.non_idr_flag, 1);
                assert_eq!(m.priority_id, 0);
                assert_eq!(m.view_id, 0);
                assert_eq!(m.temporal_id, 0);
                assert_eq!(m.anchor_pic_flag, 1);
                assert_eq!(m.inter_view_flag, 1);
                assert_eq!(m.reserved_one_bit, 1);
            }
            other => panic!("expected MVC extension, got {:?}", other),
        }
        // The extension bytes are stripped from the RBSP — only the
        // body bytes that follow are surfaced.
        assert_eq!(&*nu.rbsp, &[0xAA, 0xBB]);
    }

    #[test]
    fn mvc_extension_handles_max_field_values() {
        // priority_id = 63 (all 6 bits set), view_id = 1023
        // (all 10 bits set), temporal_id = 7 (all 3 bits set).
        let ext = pack_mvc_ext(0, 63, 1023, 7, 0, 0, 0);
        let mut nal = vec![0x14]; // type = 20 (slice extension)
        nal.extend_from_slice(&ext);
        let nu = parse_nal_unit(&nal).unwrap();
        assert_eq!(nu.header.nal_unit_type, NalUnitType::SliceExtension);
        match nu.extension {
            Some(NalUnitHeaderExtension::Mvc(m)) => {
                assert_eq!(m.priority_id, 63);
                assert_eq!(m.view_id, 1023);
                assert_eq!(m.temporal_id, 7);
            }
            other => panic!("expected MVC extension, got {:?}", other),
        }
    }

    // -------- §F.7.3.1.1 SVC NAL extension header --------

    /// Helper: pack a 24-bit SVC extension header (leading
    /// svc_extension_flag = 1) into three bytes.
    #[allow(clippy::too_many_arguments)]
    fn pack_svc_ext(
        idr: u8,
        priority: u8,
        no_inter: u8,
        dependency: u8,
        quality: u8,
        temporal: u8,
        use_ref: u8,
        discardable: u8,
        output_flag: u8,
        reserved: u8,
    ) -> [u8; 3] {
        let bits24: u32 = (1u32 << 23) // svc_extension_flag = 1
            | ((idr as u32 & 0x1) << 22)
            | ((priority as u32 & 0x3F) << 16)
            | ((no_inter as u32 & 0x1) << 15)
            | ((dependency as u32 & 0x7) << 12)
            | ((quality as u32 & 0xF) << 8)
            | ((temporal as u32 & 0x7) << 5)
            | ((use_ref as u32 & 0x1) << 4)
            | ((discardable as u32 & 0x1) << 3)
            | ((output_flag as u32 & 0x1) << 2)
            | (reserved as u32 & 0x3);
        [
            ((bits24 >> 16) & 0xFF) as u8,
            ((bits24 >> 8) & 0xFF) as u8,
            (bits24 & 0xFF) as u8,
        ]
    }

    #[test]
    fn svc_extension_parses_via_prefix_nal_14() {
        let ext = pack_svc_ext(1, 5, 0, 2, 3, 4, 1, 0, 1, 3);
        let mut nal = vec![0x0E]; // type 14
        nal.extend_from_slice(&ext);
        let nu = parse_nal_unit(&nal).unwrap();
        match nu.extension {
            Some(NalUnitHeaderExtension::Svc(s)) => {
                assert_eq!(s.idr_flag, 1);
                assert_eq!(s.priority_id, 5);
                assert_eq!(s.no_inter_layer_pred_flag, 0);
                assert_eq!(s.dependency_id, 2);
                assert_eq!(s.quality_id, 3);
                assert_eq!(s.temporal_id, 4);
                assert_eq!(s.use_ref_base_pic_flag, 1);
                assert_eq!(s.discardable_flag, 0);
                assert_eq!(s.output_flag, 1);
                assert_eq!(s.reserved_three_2bits, 3);
            }
            other => panic!("expected SVC extension, got {:?}", other),
        }
    }

    #[test]
    fn svc_extension_handles_max_field_values() {
        // priority_id 63, dependency 7, quality 15, temporal 7.
        let ext = pack_svc_ext(0, 63, 1, 7, 15, 7, 0, 1, 0, 3);
        let mut nal = vec![0x14];
        nal.extend_from_slice(&ext);
        let nu = parse_nal_unit(&nal).unwrap();
        match nu.extension {
            Some(NalUnitHeaderExtension::Svc(s)) => {
                assert_eq!(s.priority_id, 63);
                assert_eq!(s.dependency_id, 7);
                assert_eq!(s.quality_id, 15);
                assert_eq!(s.temporal_id, 7);
            }
            other => panic!("expected SVC extension, got {:?}", other),
        }
    }

    // -------- §I.7.3.1.1 3D-AVC NAL extension header --------

    /// Helper: pack a 16-bit 3D-AVC extension header (leading
    /// avc_3d_extension_flag = 1) into two bytes.
    fn pack_3davc_ext(
        view_idx: u8,
        depth: u8,
        non_idr: u8,
        temporal: u8,
        anchor: u8,
        inter_view: u8,
    ) -> [u8; 2] {
        let bits16: u32 = (1u32 << 15) // avc_3d_extension_flag = 1
            | ((view_idx as u32 & 0xFF) << 7)
            | ((depth as u32 & 0x1) << 6)
            | ((non_idr as u32 & 0x1) << 5)
            | ((temporal as u32 & 0x7) << 2)
            | ((anchor as u32 & 0x1) << 1)
            | (inter_view as u32 & 0x1);
        [((bits16 >> 8) & 0xFF) as u8, (bits16 & 0xFF) as u8]
    }

    #[test]
    fn avc_3d_extension_parses_via_slice_extension_depth() {
        let ext = pack_3davc_ext(5, 1, 0, 3, 1, 1);
        let mut nal = vec![0x15]; // type = 21
        nal.extend_from_slice(&ext);
        nal.extend_from_slice(&[0xAA, 0xBB]);
        let nu = parse_nal_unit(&nal).unwrap();
        assert_eq!(nu.header.nal_unit_type, NalUnitType::SliceExtensionDepth);
        match nu.extension {
            Some(NalUnitHeaderExtension::Avc3d(a)) => {
                assert_eq!(a.view_idx, 5);
                assert_eq!(a.depth_flag, 1);
                assert_eq!(a.non_idr_flag, 0);
                assert_eq!(a.temporal_id, 3);
                assert_eq!(a.anchor_pic_flag, 1);
                assert_eq!(a.inter_view_flag, 1);
            }
            other => panic!("expected 3D-AVC extension, got {:?}", other),
        }
        // 3D-AVC consumes only 2 bytes after the NAL header byte.
        assert_eq!(&*nu.rbsp, &[0xAA, 0xBB]);
    }

    #[test]
    fn slice_extension_depth_with_avc3d_flag_zero_falls_through_to_mvc() {
        // §7.3.1: type 21 + avc_3d_extension_flag = 0 selects the MVC
        // 3-byte body. The leading bit of the first extension byte is
        // 0, the remaining 23 bits encode the §G.7.3.1.1 fields.
        let ext = pack_mvc_ext(1, 1, 7, 2, 0, 1, 1);
        let mut nal = vec![0x15];
        nal.extend_from_slice(&ext);
        let nu = parse_nal_unit(&nal).unwrap();
        match nu.extension {
            Some(NalUnitHeaderExtension::Mvc(m)) => {
                assert_eq!(m.non_idr_flag, 1);
                assert_eq!(m.priority_id, 1);
                assert_eq!(m.view_id, 7);
                assert_eq!(m.temporal_id, 2);
                assert_eq!(m.anchor_pic_flag, 0);
                assert_eq!(m.inter_view_flag, 1);
                assert_eq!(m.reserved_one_bit, 1);
            }
            other => panic!("expected MVC extension fallthrough, got {:?}", other),
        }
    }

    #[test]
    fn extension_header_truncated_three_byte_prefix() {
        // Type 14 with only 2 body bytes — extension parser needs 3.
        let nal = [0x0E, 0xAB, 0xCD];
        let err = parse_nal_unit(&nal).unwrap_err();
        assert_eq!(
            err,
            NalError::TruncatedExtensionHeader {
                nut: 14,
                need: 3,
                got: 2,
            }
        );
    }

    #[test]
    fn extension_header_truncated_two_byte_3davc() {
        // Type 21 with only 1 body byte (avc_3d_extension_flag = 1
        // requires 2 total).
        let nal = [0x15, 0x80];
        let err = parse_nal_unit(&nal).unwrap_err();
        assert_eq!(
            err,
            NalError::TruncatedExtensionHeader {
                nut: 21,
                need: 2,
                got: 1,
            }
        );
    }

    #[test]
    fn extension_header_truncated_zero_after_header() {
        // Type 21 with no extension bytes at all.
        let nal = [0x15];
        let err = parse_nal_unit(&nal).unwrap_err();
        assert_eq!(
            err,
            NalError::TruncatedExtensionHeader {
                nut: 21,
                need: 2,
                got: 0,
            }
        );
        // Same for type 14.
        let nal = [0x0E];
        let err = parse_nal_unit(&nal).unwrap_err();
        assert_eq!(
            err,
            NalError::TruncatedExtensionHeader {
                nut: 14,
                need: 3,
                got: 0,
            }
        );
    }

    #[test]
    fn extension_bytes_are_not_emulation_prevention_stripped() {
        // §7.3.1: the emulation-prevention scan starts at i =
        // nalUnitHeaderBytes, so a `00 00 03` triple inside the
        // extension bytes is preserved verbatim. The leading
        // svc_extension_flag = 0 sits in the MSB of byte 0; here we
        // use 0x00 to verify that a fully-zero header still parses
        // as an MVC extension (all-zero fields).
        let nal = [
            0x0E, // type 14, nal_ref_idc = 0
            0x00, 0x00, 0x03, // raw MVC ext bytes — `03` is NOT stripped here
            0xAA, 0xBB,
        ];
        let nu = parse_nal_unit(&nal).unwrap();
        match nu.extension {
            Some(NalUnitHeaderExtension::Mvc(m)) => {
                // The three raw extension bytes 00 00 03 decode as:
                //   non_idr_flag = 0, priority_id = 0, view_id = 0,
                //   temporal_id = 0, anchor_pic_flag = 0,
                //   inter_view_flag = 1, reserved_one_bit = 1
                // (the low byte 0x03 = 0b00000011).
                assert_eq!(m.non_idr_flag, 0);
                assert_eq!(m.priority_id, 0);
                assert_eq!(m.view_id, 0);
                assert_eq!(m.temporal_id, 0);
                assert_eq!(m.anchor_pic_flag, 0);
                assert_eq!(m.inter_view_flag, 1);
                assert_eq!(m.reserved_one_bit, 1);
            }
            other => panic!("expected MVC extension, got {:?}", other),
        }
        // The RBSP body starts AFTER the extension bytes.
        assert_eq!(&*nu.rbsp, &[0xAA, 0xBB]);
    }

    #[test]
    fn extension_followed_by_emulation_prevention_in_rbsp() {
        // §7.3.1 emulation-prevention starts past the extension bytes:
        // the `03` inside `00 00 03 04` in the RBSP body IS stripped.
        let mvc = pack_mvc_ext(0, 0, 0, 0, 0, 0, 0);
        let mut nal = vec![0x0E];
        nal.extend_from_slice(&mvc);
        nal.extend_from_slice(&[0x00, 0x00, 0x03, 0x04]);
        let nu = parse_nal_unit(&nal).unwrap();
        assert!(matches!(nu.extension, Some(NalUnitHeaderExtension::Mvc(_))));
        // RBSP body has the emulation-prevention byte stripped.
        assert_eq!(&*nu.rbsp, &[0x00, 0x00, 0x04]);
    }

    #[test]
    fn non_extension_nal_types_carry_no_extension_field() {
        // Type 7 (SPS) — no extension, full payload becomes RBSP.
        let nu = parse_nal_unit(&[0x67, 0x42, 0x00, 0x1e]).unwrap();
        assert!(nu.extension.is_none());
        // Type 1 (non-IDR slice) — also no extension.
        let nu = parse_nal_unit(&[0x41, 0x9a]).unwrap();
        assert!(nu.extension.is_none());
    }
}
