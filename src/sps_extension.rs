//! §7.3.2.1.2 — Sequence parameter set extension RBSP parsing.
//!
//! Spec-driven implementation of `seq_parameter_set_extension_rbsp()` per
//! ITU-T Rec. H.264 | ISO/IEC 14496-10 (08/2024). This RBSP rides in a NAL
//! unit of type 13 (Table 7-1) and carries the parameters that describe an
//! *auxiliary coded picture* — the alpha / opacity plane used for the
//! alpha-blending composition example in §7.4.2.1.2.
//!
//! The syntax (§7.3.2.1.2) is:
//!
//! ```text
//! seq_parameter_set_extension_rbsp( ) {
//!     seq_parameter_set_id                          ue(v)
//!     aux_format_idc                                ue(v)
//!     if( aux_format_idc != 0 ) {
//!         bit_depth_aux_minus8                      ue(v)
//!         alpha_incr_flag                           u(1)
//!         alpha_opaque_value                        u(v)
//!         alpha_transparent_value                   u(v)
//!     }
//!     additional_extension_flag                     u(1)
//!     rbsp_trailing_bits( )
//! }
//! ```
//!
//! where the `u(v)` width of `alpha_opaque_value` / `alpha_transparent_value`
//! is `bit_depth_aux_minus8 + 9` bits (§7.4.2.1.2).
//!
//! Per §7.4.2.1.2 "the decoding of the sequence parameter set extension and
//! the decoding of auxiliary coded pictures is not required for conformance",
//! so this parser only *surfaces* the fields; no auxiliary reconstruction is
//! wired. The parser takes an already-de-emulated RBSP (the NAL header byte
//! peeled off, `emulation_prevention_three_byte` stripped by
//! [`crate::nal::parse_nal_unit`]).

use crate::bitstream::{BitError, BitReader};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SpsExtensionError {
    #[error("bitstream read failed: {0}")]
    Bitstream(#[from] BitError),
    /// §7.4.2.1.1 — `seq_parameter_set_id` range 0..=31.
    #[error("seq_parameter_set_id out of range (got {0}, max 31)")]
    SpsIdOutOfRange(u32),
    /// §7.4.2.1.2 — `aux_format_idc` shall be in the range 0..=3. Values
    /// greater than 3 are "reserved to indicate the presence of exactly one
    /// auxiliary coded picture … for purposes to be specified in the future".
    /// Reject them rather than mis-derive the auxiliary plane's bit depth.
    #[error("aux_format_idc out of range (got {0}, max 3)")]
    AuxFormatIdcOutOfRange(u32),
    /// §7.4.2.1.2 — `bit_depth_aux_minus8` shall be in the range 0..=4.
    #[error("bit_depth_aux_minus8 out of range (got {0}, max 4)")]
    BitDepthAuxOutOfRange(u32),
}

pub type SpsExtensionResult<T> = Result<T, SpsExtensionError>;

/// §7.3.2.1.2 — the auxiliary-coded-picture parameters present only when
/// `aux_format_idc != 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuxFormat {
    /// §7.4.2.1.2 — bit depth of the auxiliary picture samples is
    /// `bit_depth_aux_minus8 + 8`. Stored as the raw `minus8` value
    /// (0..=4); use [`AuxFormat::bit_depth_aux`] for the resolved depth.
    pub bit_depth_aux_minus8: u32,
    /// §7.4.2.1.2 — when set, auxiliary sample values strictly greater than
    /// `Min(alpha_opaque_value, alpha_transparent_value)` are incremented by
    /// one to form the interpretation sample value.
    pub alpha_incr_flag: bool,
    /// §7.4.2.1.2 — interpretation sample value treated as opaque. Coded in
    /// `bit_depth_aux_minus8 + 9` bits.
    pub alpha_opaque_value: u32,
    /// §7.4.2.1.2 — interpretation sample value treated as transparent.
    /// Coded in `bit_depth_aux_minus8 + 9` bits.
    pub alpha_transparent_value: u32,
}

impl AuxFormat {
    /// §7.4.2.1.2 — resolved bit depth of the auxiliary picture samples
    /// (`bit_depth_aux_minus8 + 8`, range 8..=12).
    pub fn bit_depth_aux(&self) -> u32 {
        self.bit_depth_aux_minus8 + 8
    }

    /// §7.4.2.1.2 — width in bits of the `alpha_opaque_value` /
    /// `alpha_transparent_value` `u(v)` syntax elements
    /// (`bit_depth_aux_minus8 + 9`, range 9..=13).
    pub fn alpha_value_bits(&self) -> u32 {
        self.bit_depth_aux_minus8 + 9
    }
}

/// §7.3.2.1.2 — parsed sequence parameter set extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeqParameterSetExtension {
    /// §7.4.2.1.2 — identifies the sequence parameter set this extension
    /// supplements. The corresponding `seq_parameter_set_rbsp()` with the
    /// same id describes the *primary* coded picture; this extension
    /// describes the associated *auxiliary* coded picture. Range 0..=31.
    pub seq_parameter_set_id: u32,
    /// §7.4.2.1.2 — `aux_format_idc`. 0 = no auxiliary coded pictures;
    /// 1/2 = present with / without primary-sample pre-multiplication for
    /// alpha blending; 3 = present, usage unspecified.
    pub aux_format_idc: u32,
    /// §7.3.2.1.2 — the auxiliary-format block, present iff
    /// `aux_format_idc != 0`.
    pub aux_format: Option<AuxFormat>,
    /// §7.4.2.1.2 — `additional_extension_flag`. The spec mandates 0; a 1
    /// reserves trailing data for future use, which decoders "shall ignore".
    /// Surfaced verbatim so callers can detect the reserved case.
    pub additional_extension_flag: bool,
}

impl SeqParameterSetExtension {
    /// Parse a `seq_parameter_set_extension_rbsp()` from a de-emulated RBSP
    /// body (NAL header byte already removed).
    pub fn parse(rbsp: &[u8]) -> SpsExtensionResult<Self> {
        let mut r = BitReader::new(rbsp);
        Self::parse_from(&mut r)
    }

    /// Parse the body from an existing [`BitReader`] positioned at the first
    /// bit of `seq_parameter_set_id`.
    pub fn parse_from(r: &mut BitReader<'_>) -> SpsExtensionResult<Self> {
        let seq_parameter_set_id = r.ue()?;
        if seq_parameter_set_id > 31 {
            return Err(SpsExtensionError::SpsIdOutOfRange(seq_parameter_set_id));
        }

        let aux_format_idc = r.ue()?;
        if aux_format_idc > 3 {
            return Err(SpsExtensionError::AuxFormatIdcOutOfRange(aux_format_idc));
        }

        let aux_format = if aux_format_idc != 0 {
            let bit_depth_aux_minus8 = r.ue()?;
            if bit_depth_aux_minus8 > 4 {
                return Err(SpsExtensionError::BitDepthAuxOutOfRange(
                    bit_depth_aux_minus8,
                ));
            }
            let alpha_incr_flag = r.u(1)? == 1;
            // §7.4.2.1.2 — `bit_depth_aux_minus8 + 9` bits (range 9..=13).
            let alpha_value_bits = bit_depth_aux_minus8 + 9;
            let alpha_opaque_value = r.u(alpha_value_bits)?;
            let alpha_transparent_value = r.u(alpha_value_bits)?;
            Some(AuxFormat {
                bit_depth_aux_minus8,
                alpha_incr_flag,
                alpha_opaque_value,
                alpha_transparent_value,
            })
        } else {
            None
        };

        let additional_extension_flag = r.u(1)? == 1;
        // Per §7.4.2.1.2, when additional_extension_flag == 1 the decoder
        // "shall ignore all data that follows"; we stop reading the
        // extension body here either way and let the caller discard the rest.
        // The mandated rbsp_trailing_bits() are not consumed (they carry no
        // information and the body is fully parsed above).

        Ok(Self {
            seq_parameter_set_id,
            aux_format_idc,
            aux_format,
            additional_extension_flag,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-writer helper mirroring the spec's MSB-first packing so the tests
    /// can hand-assemble `seq_parameter_set_extension_rbsp()` payloads.
    struct BitWriter {
        bits: Vec<u8>,
    }

    impl BitWriter {
        fn new() -> Self {
            Self { bits: Vec::new() }
        }

        fn u(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                self.bits.push(((value >> i) & 1) as u8);
            }
        }

        fn ue(&mut self, value: u32) {
            // §9.1 — leadingZeroBits zeros, then a 1, then the info bits.
            let code_num = value + 1;
            let n = 32 - code_num.leading_zeros();
            let leading = n - 1;
            for _ in 0..leading {
                self.bits.push(0);
            }
            self.u(code_num, n);
        }

        fn into_bytes(mut self) -> Vec<u8> {
            // §7.3.2.11 rbsp_trailing_bits(): stop-one-bit then zero-pad.
            self.bits.push(1);
            while self.bits.len() % 8 != 0 {
                self.bits.push(0);
            }
            let mut out = vec![0u8; self.bits.len() / 8];
            for (i, &b) in self.bits.iter().enumerate() {
                if b == 1 {
                    out[i / 8] |= 1 << (7 - (i % 8));
                }
            }
            out
        }
    }

    #[test]
    fn aux_format_idc_zero_skips_alpha_block() {
        let mut w = BitWriter::new();
        w.ue(7); // seq_parameter_set_id
        w.ue(0); // aux_format_idc == 0 → no aux block
        w.u(0, 1); // additional_extension_flag
        let bytes = w.into_bytes();

        let ext = SeqParameterSetExtension::parse(&bytes).unwrap();
        assert_eq!(ext.seq_parameter_set_id, 7);
        assert_eq!(ext.aux_format_idc, 0);
        assert_eq!(ext.aux_format, None);
        assert!(!ext.additional_extension_flag);
    }

    #[test]
    fn aux_format_present_full_block() {
        let mut w = BitWriter::new();
        w.ue(0); // seq_parameter_set_id
        w.ue(1); // aux_format_idc == 1
        w.ue(2); // bit_depth_aux_minus8 == 2 → 10-bit aux, 11-bit alpha vals
        w.u(1, 1); // alpha_incr_flag
        w.u(0x3FF, 11); // alpha_opaque_value
        w.u(0x000, 11); // alpha_transparent_value
        w.u(0, 1); // additional_extension_flag
        let bytes = w.into_bytes();

        let ext = SeqParameterSetExtension::parse(&bytes).unwrap();
        assert_eq!(ext.seq_parameter_set_id, 0);
        assert_eq!(ext.aux_format_idc, 1);
        let aux = ext.aux_format.expect("aux block present");
        assert_eq!(aux.bit_depth_aux_minus8, 2);
        assert_eq!(aux.bit_depth_aux(), 10);
        assert_eq!(aux.alpha_value_bits(), 11);
        assert!(aux.alpha_incr_flag);
        assert_eq!(aux.alpha_opaque_value, 0x3FF);
        assert_eq!(aux.alpha_transparent_value, 0x000);
        assert!(!ext.additional_extension_flag);
    }

    #[test]
    fn aux_value_width_tracks_bit_depth_minus8_zero() {
        // bit_depth_aux_minus8 == 0 → 8-bit aux, 9-bit alpha values.
        let mut w = BitWriter::new();
        w.ue(3); // seq_parameter_set_id
        w.ue(3); // aux_format_idc == 3 (usage unspecified)
        w.ue(0); // bit_depth_aux_minus8 == 0
        w.u(0, 1); // alpha_incr_flag
        w.u(0x1FF, 9); // alpha_opaque_value (max 9-bit)
        w.u(0x055, 9); // alpha_transparent_value
        w.u(0, 1); // additional_extension_flag
        let bytes = w.into_bytes();

        let ext = SeqParameterSetExtension::parse(&bytes).unwrap();
        let aux = ext.aux_format.unwrap();
        assert_eq!(aux.bit_depth_aux(), 8);
        assert_eq!(aux.alpha_value_bits(), 9);
        assert_eq!(aux.alpha_opaque_value, 0x1FF);
        assert_eq!(aux.alpha_transparent_value, 0x055);
    }

    #[test]
    fn additional_extension_flag_surfaced() {
        let mut w = BitWriter::new();
        w.ue(1);
        w.ue(0);
        w.u(1, 1); // additional_extension_flag == 1 (reserved)
        let bytes = w.into_bytes();

        let ext = SeqParameterSetExtension::parse(&bytes).unwrap();
        assert!(ext.additional_extension_flag);
    }

    #[test]
    fn rejects_sps_id_out_of_range() {
        let mut w = BitWriter::new();
        w.ue(32); // seq_parameter_set_id == 32 (> 31)
        let bytes = w.into_bytes();
        assert_eq!(
            SeqParameterSetExtension::parse(&bytes),
            Err(SpsExtensionError::SpsIdOutOfRange(32))
        );
    }

    #[test]
    fn rejects_aux_format_idc_out_of_range() {
        let mut w = BitWriter::new();
        w.ue(0);
        w.ue(4); // aux_format_idc == 4 (> 3, reserved)
        let bytes = w.into_bytes();
        assert_eq!(
            SeqParameterSetExtension::parse(&bytes),
            Err(SpsExtensionError::AuxFormatIdcOutOfRange(4))
        );
    }

    #[test]
    fn rejects_bit_depth_aux_out_of_range() {
        let mut w = BitWriter::new();
        w.ue(0);
        w.ue(1); // aux_format_idc == 1
        w.ue(5); // bit_depth_aux_minus8 == 5 (> 4)
        let bytes = w.into_bytes();
        assert_eq!(
            SeqParameterSetExtension::parse(&bytes),
            Err(SpsExtensionError::BitDepthAuxOutOfRange(5))
        );
    }

    #[test]
    fn truncated_input_surfaces_eof() {
        // Hand-pack a body whose alpha_value reads run past the end of the
        // buffer: seq_parameter_set_id=0 (`1`), aux_format_idc=1 (`010`),
        // bit_depth_aux_minus8=0 (`1`), alpha_incr_flag=0 (`0`) occupies the
        // first 6 bits, then the parser needs 9 bits for alpha_opaque_value
        // but only 2 bits remain in this single byte. Packed MSB-first:
        // 1 010 1 0 | 00  =  0b1010_1000.
        let bytes = [0b1010_1000u8];
        let res = SeqParameterSetExtension::parse(&bytes);
        assert!(matches!(
            res,
            Err(SpsExtensionError::Bitstream(BitError::Eof))
        ));
    }
}
