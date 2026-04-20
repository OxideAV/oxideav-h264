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

#[derive(Debug, Clone)]
pub struct NalUnit<'a> {
    pub header: NalHeader,
    pub rbsp: Cow<'a, [u8]>,
}

/// Parse §7.3.1 `nal_unit()` from the raw NAL bytes (no start code, no
/// length prefix — just the NAL header byte followed by the encoded
/// payload). Strips emulation prevention bytes per §7.4.1.1.
///
/// For extension NAL types (14/20/21) the extra header bytes are
/// included in the returned RBSP slice — this module doesn't parse
/// SVC/MVC/3D-AVC headers. Callers can detect those types via
/// [`NalHeader::nal_unit_type`] and skip them.
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
    let rbsp = rbsp_from_nal_payload(&payload[1..]);
    Ok(NalUnit { header, rbsp })
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
    }
}
