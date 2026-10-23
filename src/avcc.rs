//! ISO/IEC 14496-15 §5.3.2.1 — `AVCDecoderConfigurationRecord`
//! (`avcC`) builder + parser, and the Annex B → length-prefixed sample
//! framing that goes with it.
//!
//! The record carries the parameter sets out of band plus the size of
//! the per-NAL length field used inside samples (§5.3.2.1.3
//! `lengthSizeMinusOne` ∈ {0, 1, 3}). For every `AVCProfileIndication`
//! other than 66 / 77 / 88 (§5.3.2.1.2 syntax) the record continues
//! with the chroma format / bit-depth trailer and the SPS-extension
//! array. §5.3.2.1.3 bounds `bit_depth_luma_minus8` /
//! `bit_depth_chroma_minus8` to 0..=4 — a 14-bit stream (`minus8 = 6`,
//! legal under §7.4.2.1.1 of H.264) has no conforming `avcC`.
//!
//! Field derivation (§5.3.2.1.1 / §5.3.2.1.3): `AVCProfileIndication`
//! is the first SPS's `profile_idc`; `profile_compatibility` is "the
//! byte that occurs between the profile_IDC and level_IDC in a
//! sequence parameter set", ANDed over every SPS ("each profile
//! compatibility flag may only be set if all the included parameter
//! sets set that flag"); `AVCLevelIndication` is the highest
//! `level_idc` over the SPSs ("equal to or greater than the highest
//! level indicated"); `chroma_format` / bit depths come from the SPS
//! and "shall be identical in all sequence parameter sets in a single
//! AVC configuration record".
//!
//! This is codec-configuration work (the record is built from the
//! codec's own parameter sets), so it lives here rather than in a
//! container crate; ISO BMFF / HEIF muxers wrap the bytes in their
//! `avcC` box.

use crate::nal::{AnnexBSplitter, parse_nal_unit};
use crate::sps::Sps;

/// Errors from building or parsing an `avcC` record.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AvccError {
    /// No SPS NAL unit to derive the profile / level / layout from.
    #[error("avcC: no sequence parameter set")]
    NoSps,
    /// A parameter-set NAL unit failed to parse.
    #[error("avcC: {0}")]
    ParameterSet(String),
    /// §5.3.2.1.3 — `bit_depth_*_minus8` shall be in 0..=4.
    #[error(
        "avcC: bit_depth_minus8 = {0} exceeds the §5.3.2.1.3 range 0..=4 (no conforming record for >12-bit streams)"
    )]
    BitDepthUnrepresentable(u32),
    /// §5.3.2.1.1 — chroma format / bit depths differ between SPSs.
    #[error(
        "avcC: sequence parameter sets disagree on chroma_format_idc / bit depth (§5.3.2.1.1 needs one record per layout)"
    )]
    MismatchedSps,
    /// `numOfSequenceParameterSets` is a 5-bit field.
    #[error("avcC: {0} SPS entries exceed the 5-bit count")]
    TooManySps(usize),
    /// `numOfPictureParameterSets` / `numOfSequenceParameterSetExt` are 8-bit fields.
    #[error("avcC: {0} parameter-set entries exceed the 8-bit count")]
    TooManyEntries(usize),
    /// A NAL unit longer than the 16-bit length field (or the sample
    /// length prefix) can carry.
    #[error("avcC: NAL unit of {0} bytes exceeds the length field")]
    NalTooLong(usize),
    /// The record ended inside a field.
    #[error("avcC: truncated at {0}")]
    Truncated(&'static str),
    /// `configurationVersion` other than 1.
    #[error("avcC: configurationVersion {0} (only 1 is defined)")]
    BadVersion(u8),
    /// `lengthSizeMinusOne == 2` (3-byte prefixes) is forbidden.
    #[error("avcC: lengthSizeMinusOne = 2 (3-byte NAL length prefix) is forbidden by §5.3.2.1.3")]
    BadLengthSize,
}

/// A parsed / buildable `AVCDecoderConfigurationRecord`. NAL units are
/// stored raw (header byte + emulation-prevented payload, no start
/// code, no length prefix).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvcDecoderConfigurationRecord {
    /// `AVCProfileIndication` (§A.2 `profile_idc`).
    pub profile_idc: u8,
    /// `profile_compatibility` — the SPS byte between `profile_idc`
    /// and `level_idc` (constraint_set0..5 flags, MSB first, then
    /// `reserved_zero_2bits`).
    pub profile_compatibility: u8,
    /// `AVCLevelIndication` (§A.3 `level_idc`).
    pub level_idc: u8,
    /// `lengthSizeMinusOne + 1`: 1, 2 or 4 bytes per sample NAL length.
    pub length_size: u8,
    /// `sequenceParameterSetNALUnit` array.
    pub sps: Vec<Vec<u8>>,
    /// `pictureParameterSetNALUnit` array.
    pub pps: Vec<Vec<u8>>,
    /// Trailer `chroma_format` (present for profiles other than
    /// 66 / 77 / 88; `None` when a parsed record lacked the trailer).
    pub chroma_format: Option<u8>,
    /// Trailer `bit_depth_luma_minus8`.
    pub bit_depth_luma_minus8: Option<u8>,
    /// Trailer `bit_depth_chroma_minus8`.
    pub bit_depth_chroma_minus8: Option<u8>,
    /// `sequenceParameterSetExtNALUnit` array (trailer).
    pub sps_ext: Vec<Vec<u8>>,
}

impl AvcDecoderConfigurationRecord {
    /// §5.3.2.1.2 — whether a record for `profile_idc` carries the
    /// chroma-format / bit-depth / SPS-extension trailer: every
    /// profile other than Baseline (66), Main (77) and Extended (88).
    pub fn has_extension_trailer(profile_idc: u8) -> bool {
        !matches!(profile_idc, 66 | 77 | 88)
    }

    /// Build a record from raw SPS / PPS NAL units (and optional SPS
    /// extension NAL units). Profile / compatibility / level / layout
    /// are derived from the SPSs as the module docs describe; the
    /// length size defaults to 4 ([`Self::with_length_size`]).
    pub fn from_parameter_sets(
        sps: &[&[u8]],
        pps: &[&[u8]],
        sps_ext: &[&[u8]],
    ) -> Result<Self, AvccError> {
        let first = sps.first().ok_or(AvccError::NoSps)?;
        let (profile_idc, mut compat, mut level_idc) = sps_header_bytes(first)?;
        let parsed_first = parse_sps(first)?;
        let layout = (
            parsed_first.chroma_format_idc,
            parsed_first.bit_depth_luma_minus8,
            parsed_first.bit_depth_chroma_minus8,
        );
        for s in &sps[1..] {
            let (_, c, l) = sps_header_bytes(s)?;
            compat &= c;
            level_idc = level_idc.max(l);
            let p = parse_sps(s)?;
            if (
                p.chroma_format_idc,
                p.bit_depth_luma_minus8,
                p.bit_depth_chroma_minus8,
            ) != layout
            {
                return Err(AvccError::MismatchedSps);
            }
        }
        let trailer = Self::has_extension_trailer(profile_idc);
        if trailer {
            for bd in [layout.1, layout.2] {
                if bd > 4 {
                    return Err(AvccError::BitDepthUnrepresentable(bd));
                }
            }
        }
        Ok(Self {
            profile_idc,
            profile_compatibility: compat,
            level_idc,
            length_size: 4,
            sps: sps.iter().map(|s| s.to_vec()).collect(),
            pps: pps.iter().map(|p| p.to_vec()).collect(),
            chroma_format: trailer.then_some(layout.0 as u8),
            bit_depth_luma_minus8: trailer.then_some(layout.1 as u8),
            bit_depth_chroma_minus8: trailer.then_some(layout.2 as u8),
            sps_ext: sps_ext.iter().map(|s| s.to_vec()).collect(),
        })
    }

    /// Build a record from the parameter sets found in an Annex B byte
    /// stream (SPS = type 7, PPS = type 8, SPS extension = type 13, in
    /// stream order). Typically the first IDR access unit of an
    /// encoder that repeats its parameter sets in band.
    pub fn from_annex_b(annex_b: &[u8]) -> Result<Self, AvccError> {
        let mut sps: Vec<&[u8]> = Vec::new();
        let mut pps: Vec<&[u8]> = Vec::new();
        let mut ext: Vec<&[u8]> = Vec::new();
        for nal in AnnexBSplitter::new(annex_b) {
            match nal.first().map(|h| h & 0x1f) {
                Some(7) => sps.push(nal),
                Some(8) => pps.push(nal),
                Some(13) => ext.push(nal),
                _ => {}
            }
        }
        Self::from_parameter_sets(&sps, &pps, &ext)
    }

    /// Set the sample NAL length-prefix size (1, 2 or 4 bytes).
    pub fn with_length_size(mut self, length_size: u8) -> Self {
        assert!(
            matches!(length_size, 1 | 2 | 4),
            "lengthSizeMinusOne ∈ {{0, 1, 3}} — length_size must be 1, 2 or 4"
        );
        self.length_size = length_size;
        self
    }

    /// The first SPS, parsed (§7.3.2.1.1).
    pub fn parsed_sps(&self) -> Result<Sps, AvccError> {
        let first = self.sps.first().ok_or(AvccError::NoSps)?;
        parse_sps(first)
    }

    /// Serialise per §5.3.2.1.2. For a trailer-carrying profile whose
    /// trailer fields are unset (a parsed record that lacked them),
    /// the values are recovered from the first SPS.
    pub fn to_bytes(&self) -> Result<Vec<u8>, AvccError> {
        if self.sps.len() > 31 {
            return Err(AvccError::TooManySps(self.sps.len()));
        }
        if self.pps.len() > 255 {
            return Err(AvccError::TooManyEntries(self.pps.len()));
        }
        if self.sps_ext.len() > 255 {
            return Err(AvccError::TooManyEntries(self.sps_ext.len()));
        }
        assert!(
            matches!(self.length_size, 1 | 2 | 4),
            "length_size must be 1, 2 or 4"
        );
        let mut out = Vec::with_capacity(16 + self.sps.len() * 32 + self.pps.len() * 8);
        out.push(1);
        out.push(self.profile_idc);
        out.push(self.profile_compatibility);
        out.push(self.level_idc);
        out.push(0xFC | (self.length_size - 1));
        out.push(0xE0 | self.sps.len() as u8);
        for s in &self.sps {
            push_array_entry(&mut out, s)?;
        }
        out.push(self.pps.len() as u8);
        for p in &self.pps {
            push_array_entry(&mut out, p)?;
        }
        if Self::has_extension_trailer(self.profile_idc) {
            let (cf, bdl, bdc) = match (
                self.chroma_format,
                self.bit_depth_luma_minus8,
                self.bit_depth_chroma_minus8,
            ) {
                (Some(c), Some(l), Some(k)) => (c, l, k),
                _ => {
                    let p = self.parsed_sps()?;
                    (
                        p.chroma_format_idc as u8,
                        p.bit_depth_luma_minus8 as u8,
                        p.bit_depth_chroma_minus8 as u8,
                    )
                }
            };
            for bd in [bdl, bdc] {
                if bd > 4 {
                    return Err(AvccError::BitDepthUnrepresentable(u32::from(bd)));
                }
            }
            out.push(0xFC | (cf & 0x03));
            out.push(0xF8 | (bdl & 0x07));
            out.push(0xF8 | (bdc & 0x07));
            out.push(self.sps_ext.len() as u8);
            for e in &self.sps_ext {
                push_array_entry(&mut out, e)?;
            }
        }
        Ok(out)
    }

    /// Parse a record per §5.3.2.1.2. Tolerates a trailer-carrying
    /// profile whose record stops after the PPS array (some muxers
    /// truncate it) — the trailer fields are then `None`.
    pub fn parse(bytes: &[u8]) -> Result<Self, AvccError> {
        if bytes.len() < 6 {
            return Err(AvccError::Truncated("header"));
        }
        if bytes[0] != 1 {
            return Err(AvccError::BadVersion(bytes[0]));
        }
        let profile_idc = bytes[1];
        let profile_compatibility = bytes[2];
        let level_idc = bytes[3];
        let lsm1 = bytes[4] & 0x03;
        if lsm1 == 2 {
            return Err(AvccError::BadLengthSize);
        }
        let mut pos = 6;
        let num_sps = (bytes[5] & 0x1f) as usize;
        let mut sps = Vec::with_capacity(num_sps);
        for _ in 0..num_sps {
            sps.push(read_array_entry(bytes, &mut pos, "SPS")?);
        }
        if pos >= bytes.len() {
            return Err(AvccError::Truncated("PPS count"));
        }
        let num_pps = bytes[pos] as usize;
        pos += 1;
        let mut pps = Vec::with_capacity(num_pps);
        for _ in 0..num_pps {
            pps.push(read_array_entry(bytes, &mut pos, "PPS")?);
        }
        let mut rec = Self {
            profile_idc,
            profile_compatibility,
            level_idc,
            length_size: lsm1 + 1,
            sps,
            pps,
            chroma_format: None,
            bit_depth_luma_minus8: None,
            bit_depth_chroma_minus8: None,
            sps_ext: Vec::new(),
        };
        if Self::has_extension_trailer(profile_idc) && pos < bytes.len() {
            if pos + 4 > bytes.len() {
                return Err(AvccError::Truncated("extension trailer"));
            }
            rec.chroma_format = Some(bytes[pos] & 0x03);
            rec.bit_depth_luma_minus8 = Some(bytes[pos + 1] & 0x07);
            rec.bit_depth_chroma_minus8 = Some(bytes[pos + 2] & 0x07);
            let n = bytes[pos + 3] as usize;
            pos += 4;
            for _ in 0..n {
                rec.sps_ext
                    .push(read_array_entry(bytes, &mut pos, "SPS extension")?);
            }
        }
        Ok(rec)
    }
}

fn push_array_entry(out: &mut Vec<u8>, nal: &[u8]) -> Result<(), AvccError> {
    if nal.len() > 0xFFFF {
        return Err(AvccError::NalTooLong(nal.len()));
    }
    out.extend_from_slice(&(nal.len() as u16).to_be_bytes());
    out.extend_from_slice(nal);
    Ok(())
}

fn read_array_entry(
    bytes: &[u8],
    pos: &mut usize,
    what: &'static str,
) -> Result<Vec<u8>, AvccError> {
    if *pos + 2 > bytes.len() {
        return Err(AvccError::Truncated(what));
    }
    let len = u16::from_be_bytes([bytes[*pos], bytes[*pos + 1]]) as usize;
    *pos += 2;
    if *pos + len > bytes.len() {
        return Err(AvccError::Truncated(what));
    }
    let v = bytes[*pos..*pos + len].to_vec();
    *pos += len;
    Ok(v)
}

/// `(profile_idc, profile_compatibility byte, level_idc)` — the first
/// three RBSP bytes of an SPS NAL unit (§7.3.2.1.1).
fn sps_header_bytes(nal: &[u8]) -> Result<(u8, u8, u8), AvccError> {
    let parsed =
        parse_nal_unit(nal).map_err(|e| AvccError::ParameterSet(format!("SPS NAL: {e}")))?;
    if parsed.rbsp.len() < 3 {
        return Err(AvccError::ParameterSet("SPS shorter than 3 bytes".into()));
    }
    Ok((parsed.rbsp[0], parsed.rbsp[1], parsed.rbsp[2]))
}

fn parse_sps(nal: &[u8]) -> Result<Sps, AvccError> {
    let parsed =
        parse_nal_unit(nal).map_err(|e| AvccError::ParameterSet(format!("SPS NAL: {e}")))?;
    Sps::parse(&parsed.rbsp).map_err(|e| AvccError::ParameterSet(format!("SPS: {e}")))
}

/// Re-frame an Annex B access unit as a length-prefixed sample
/// (`length_size` ∈ {1, 2, 4} big-endian bytes per NAL unit — the
/// record's [`AvcDecoderConfigurationRecord::length_size`]). With
/// `strip_out_of_band`, parameter sets (SPS 7 / PPS 8 / SPS extension
/// 13) and filler data (12) are dropped: the parameter sets ride in the
/// `avcC` record and §5.3.1 forbids filler NAL units in samples whose
/// sample entry does not permit in-band parameter sets.
pub fn annex_b_to_length_prefixed(
    annex_b: &[u8],
    length_size: u8,
    strip_out_of_band: bool,
) -> Result<Vec<u8>, AvccError> {
    assert!(
        matches!(length_size, 1 | 2 | 4),
        "length_size must be 1, 2 or 4"
    );
    let mut out = Vec::with_capacity(annex_b.len());
    for nal in AnnexBSplitter::new(annex_b) {
        if nal.is_empty() {
            continue;
        }
        if strip_out_of_band && matches!(nal[0] & 0x1f, 7 | 8 | 12 | 13) {
            continue;
        }
        let max = if length_size == 4 {
            usize::MAX
        } else {
            (1usize << (8 * length_size as usize)) - 1
        };
        if nal.len() > max || nal.len() > u32::MAX as usize {
            return Err(AvccError::NalTooLong(nal.len()));
        }
        let len = nal.len() as u32;
        out.extend_from_slice(&len.to_be_bytes()[4 - length_size as usize..]);
        out.extend_from_slice(nal);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny SPS NAL: Baseline 66, compat 0xC0
    /// (constraint_set0/1), level 30, then a minimal body.
    fn sps_nal(profile: u8, compat: u8, level: u8) -> Vec<u8> {
        use crate::encoder::nal::build_nal_unit;
        use crate::encoder::sps::{BaselineSpsConfig, build_baseline_sps_rbsp};
        use crate::nal::NalUnitType;
        let rbsp = build_baseline_sps_rbsp(&BaselineSpsConfig {
            seq_parameter_set_id: 0,
            level_idc: level,
            width_in_mbs: 4,
            height_in_mbs: 3,
            log2_max_frame_num_minus4: 4,
            log2_max_poc_lsb_minus4: 4,
            max_num_ref_frames: 1,
            profile_idc: profile,
            chroma_format_idc: 1,
            separate_colour_plane: false,
            seq_scaling_lists: None,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            interlaced_fields: false,
            mbaff: false,
            vui: None,
        });
        let mut rbsp = rbsp;
        rbsp[1] = compat;
        // `build_nal_unit` prepends the Annex B start code; the record
        // stores bare NAL units.
        build_nal_unit(3, NalUnitType::Sps, &rbsp)[4..].to_vec()
    }

    fn pps_nal() -> Vec<u8> {
        use crate::encoder::nal::build_nal_unit;
        use crate::encoder::pps::{BaselinePpsConfig, build_baseline_pps_rbsp};
        use crate::nal::NalUnitType;
        let rbsp = build_baseline_pps_rbsp(&BaselinePpsConfig {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
            pic_init_qp_minus26: 0,
            chroma_qp_index_offset: 0,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            entropy_coding_mode_flag: false,
            transform_8x8_mode_flag: false,
            redundant_pic_cnt_present_flag: false,
            slice_groups: None,
            constrained_intra_pred_flag: false,
            pic_scaling_lists: None,
            chroma_format_idc: 1,
        });
        build_nal_unit(3, NalUnitType::Pps, &rbsp)[4..].to_vec()
    }

    #[test]
    fn baseline_record_has_no_trailer_and_round_trips() {
        let sps = sps_nal(66, 0xC0, 30);
        let pps = pps_nal();
        let rec =
            AvcDecoderConfigurationRecord::from_parameter_sets(&[&sps], &[&pps], &[]).unwrap();
        assert_eq!(rec.profile_idc, 66);
        assert_eq!(rec.profile_compatibility, 0xC0);
        assert_eq!(rec.level_idc, 30);
        assert_eq!(rec.chroma_format, None);
        let bytes = rec.to_bytes().unwrap();
        assert_eq!(&bytes[..6], &[1, 66, 0xC0, 30, 0xFF, 0xE1]);
        assert_eq!(bytes.len(), 6 + 2 + sps.len() + 1 + 2 + pps.len());
        let back = AvcDecoderConfigurationRecord::parse(&bytes).unwrap();
        assert_eq!(back, rec);
    }

    #[test]
    fn high_record_carries_trailer_and_level_max_compat_and() {
        let s1 = sps_nal(100, 0xC0, 30);
        let s2 = sps_nal(100, 0x80, 31);
        let pps = pps_nal();
        let rec = AvcDecoderConfigurationRecord::from_parameter_sets(&[&s1, &s2], &[&pps], &[])
            .unwrap()
            .with_length_size(2);
        assert_eq!(rec.level_idc, 31, "highest level over the SPSs");
        assert_eq!(rec.profile_compatibility, 0x80, "flags ANDed over SPSs");
        assert_eq!(rec.chroma_format, Some(1));
        assert_eq!(rec.bit_depth_luma_minus8, Some(0));
        let bytes = rec.to_bytes().unwrap();
        assert_eq!(bytes[4], 0xFC | 1, "lengthSizeMinusOne = 1");
        assert_eq!(bytes[5], 0xE0 | 2);
        let tail = &bytes[bytes.len() - 4..];
        assert_eq!(tail, &[0xFC | 1, 0xF8, 0xF8, 0]);
        let back = AvcDecoderConfigurationRecord::parse(&bytes).unwrap();
        assert_eq!(back, rec);
    }

    #[test]
    fn truncated_high_record_parses_without_trailer() {
        let sps = sps_nal(100, 0, 30);
        let pps = pps_nal();
        let full = AvcDecoderConfigurationRecord::from_parameter_sets(&[&sps], &[&pps], &[])
            .unwrap()
            .to_bytes()
            .unwrap();
        let cut = &full[..full.len() - 4];
        let rec = AvcDecoderConfigurationRecord::parse(cut).unwrap();
        assert_eq!(rec.chroma_format, None);
        // Re-serialising recovers the trailer from the SPS.
        assert_eq!(rec.to_bytes().unwrap(), full);
    }

    #[test]
    fn rejects_bad_version_and_three_byte_lengths() {
        let sps = sps_nal(66, 0, 30);
        let pps = pps_nal();
        let mut bytes = AvcDecoderConfigurationRecord::from_parameter_sets(&[&sps], &[&pps], &[])
            .unwrap()
            .to_bytes()
            .unwrap();
        bytes[0] = 2;
        assert_eq!(
            AvcDecoderConfigurationRecord::parse(&bytes),
            Err(AvccError::BadVersion(2))
        );
        bytes[0] = 1;
        bytes[4] = 0xFE;
        assert_eq!(
            AvcDecoderConfigurationRecord::parse(&bytes),
            Err(AvccError::BadLengthSize)
        );
        assert_eq!(
            AvcDecoderConfigurationRecord::parse(&bytes[..5]),
            Err(AvccError::Truncated("header"))
        );
        assert_eq!(
            AvcDecoderConfigurationRecord::from_parameter_sets(&[], &[&pps], &[]),
            Err(AvccError::NoSps)
        );
    }

    #[test]
    fn length_prefix_framing_strips_parameter_sets_and_filler() {
        let sps = sps_nal(66, 0, 30);
        let pps = pps_nal();
        let mut annex_b = Vec::new();
        annex_b.extend_from_slice(&[0, 0, 0, 1]);
        annex_b.extend_from_slice(&sps);
        annex_b.extend_from_slice(&[0, 0, 0, 1]);
        annex_b.extend_from_slice(&pps);
        annex_b.extend_from_slice(&[0, 0, 0, 1, 0x0C, 0xFF, 0xFF, 0x80]); // filler
        annex_b.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x88, 0x84, 0x21]); // IDR slice
        let out = annex_b_to_length_prefixed(&annex_b, 4, true).unwrap();
        assert_eq!(out, vec![0, 0, 0, 4, 0x65, 0x88, 0x84, 0x21]);
        let keep = annex_b_to_length_prefixed(&annex_b, 2, false).unwrap();
        assert_eq!(&keep[..2], &(sps.len() as u16).to_be_bytes());
        // A NAL longer than a 1-byte prefix can carry is refused.
        let long = annex_b_to_length_prefixed(&annex_b, 1, false);
        assert!(long.is_ok() || matches!(long, Err(AvccError::NalTooLong(_))));
        let mut big = vec![0, 0, 0, 1, 0x65];
        big.extend(std::iter::repeat_n(0x11u8, 300));
        assert_eq!(
            annex_b_to_length_prefixed(&big, 1, false),
            Err(AvccError::NalTooLong(301))
        );
    }
}
