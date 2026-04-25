//! §7.3.1 + §7.4.1.1 + §B.1 — NAL emit + Annex B framing (encoder side).
//!
//! Mirrors [`crate::nal`] in reverse:
//! * [`build_nal_unit`] takes a NAL header byte and an RBSP payload,
//!   inserts `emulation_prevention_three_byte` (§7.4.1.1) where needed,
//!   and prepends a four-byte Annex B start-code prefix (§B.1).
//!
//! The resulting byte run can be concatenated with other NAL units to
//! form a valid Annex B byte stream. We always emit the four-byte
//! start-code prefix (`00 00 00 01`) — any decoder accepts it because
//! the leading `00` is just a `leading_zero_8bits` (§B.1).

#![allow(dead_code)]

use crate::nal::NalUnitType;

/// Build the NAL header byte per §7.3.1: forbidden_zero_bit (always 0)
/// + nal_ref_idc (2 bits) + nal_unit_type (5 bits).
pub fn nal_header_byte(nal_ref_idc: u8, nal_unit_type: NalUnitType) -> u8 {
    debug_assert!(nal_ref_idc <= 3);
    let nut = nal_unit_type.as_u8() & 0x1F;
    ((nal_ref_idc & 0x3) << 5) | nut
}

/// Insert `emulation_prevention_three_byte` (0x03) per §7.4.1.1 into an
/// RBSP byte run. The rule: anywhere `00 00 00`, `00 00 01`, `00 00 02`,
/// or `00 00 03` would appear, an `0x03` is inserted between the two
/// leading zeros and the third byte.
///
/// In practice the spec only requires insertion before bytes that would
/// otherwise look like a start code prefix or stop the de-emulation
/// machine; this implementation matches the §7.4.1.1 informative form
/// exactly: scan for `00 00` followed by a byte in `{00, 01, 02, 03}`.
pub fn insert_emulation_prevention(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + rbsp.len() / 16);
    let mut i = 0;
    while i < rbsp.len() {
        out.push(rbsp[i]);
        // Look ahead two bytes; emit emulation byte before any in-place
        // 00 00 0x sequence with x ≤ 3.
        if i + 2 < rbsp.len() && rbsp[i] == 0 && rbsp[i + 1] == 0 && rbsp[i + 2] <= 3 {
            out.push(0);
            out.push(0x03);
            i += 2;
        } else {
            i += 1;
        }
    }
    out
}

/// Build a complete Annex B NAL unit: 4-byte start-code prefix +
/// header byte + RBSP with emulation prevention bytes inserted.
///
/// `rbsp` is the de-emulated body (the bytes the decoder sees after
/// `parse_nal_unit` strips the header and 0x03s).
pub fn build_nal_unit(nal_ref_idc: u8, nal_unit_type: NalUnitType, rbsp: &[u8]) -> Vec<u8> {
    let header_byte = nal_header_byte(nal_ref_idc, nal_unit_type);
    let escaped = insert_emulation_prevention(rbsp);
    let mut out = Vec::with_capacity(4 + 1 + escaped.len());
    out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    out.push(header_byte);
    out.extend_from_slice(&escaped);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nal::{parse_nal_unit, AnnexBSplitter, NalUnitType};

    #[test]
    fn header_byte_encodes_sps() {
        // SPS with nal_ref_idc=3 → 0b 0 11 00111 = 0x67.
        assert_eq!(nal_header_byte(3, NalUnitType::Sps), 0x67);
    }

    #[test]
    fn header_byte_encodes_idr() {
        // IDR with nal_ref_idc=3 → 0b 0 11 00101 = 0x65.
        assert_eq!(nal_header_byte(3, NalUnitType::SliceIdr), 0x65);
    }

    #[test]
    fn no_emulation_byte_when_no_run() {
        let rbsp = [0x00, 0x01, 0x02, 0x05];
        assert_eq!(insert_emulation_prevention(&rbsp), rbsp);
    }

    #[test]
    fn inserts_emulation_byte_before_00_00_01() {
        // 00 00 01 → 00 00 03 01
        let rbsp = [0x00, 0x00, 0x01];
        assert_eq!(
            insert_emulation_prevention(&rbsp),
            vec![0x00, 0x00, 0x03, 0x01]
        );
    }

    #[test]
    fn inserts_emulation_byte_before_00_00_03() {
        // 00 00 03 must be escaped to 00 00 03 03 so the third byte
        // (the original 0x03) is preserved by de-emulation.
        let rbsp = [0x00, 0x00, 0x03];
        assert_eq!(
            insert_emulation_prevention(&rbsp),
            vec![0x00, 0x00, 0x03, 0x03]
        );
    }

    #[test]
    fn round_trip_through_annex_b_splitter_and_de_emulation() {
        let rbsp_in: &[u8] = &[0x42, 0x00, 0x00, 0x01, 0xAA];
        let nal = build_nal_unit(3, NalUnitType::Sps, rbsp_in);
        // Start-code prefix:
        assert_eq!(&nal[0..4], &[0x00, 0x00, 0x00, 0x01]);
        // Header byte:
        assert_eq!(nal[4], 0x67);
        // Decode it back through the existing splitter and de-emulator.
        let nals: Vec<_> = AnnexBSplitter::new(&nal).collect();
        assert_eq!(nals.len(), 1);
        let nu = parse_nal_unit(nals[0]).unwrap();
        assert_eq!(nu.header.nal_unit_type, NalUnitType::Sps);
        assert_eq!(nu.header.nal_ref_idc, 3);
        assert_eq!(&*nu.rbsp, rbsp_in);
    }
}
