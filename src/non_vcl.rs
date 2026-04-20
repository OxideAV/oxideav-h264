//! §7.3.2.{3..7} — Non-VCL NAL unit parsers.
//!
//! AUD, SEI envelope, filler, and the end-of-sequence/stream markers.
//! SEI payload parsing is deferred (Annex D — many payload types).

use crate::bitstream::{BitError, BitReader};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum NonVclError {
    #[error("bitstream read failed: {0}")]
    Bitstream(#[from] BitError),
    #[error("AUD primary_pic_type {0} exceeds 7 (must be 0..=7)")]
    AudPicTypeOutOfRange(u32),
    #[error("SEI RBSP is truncated (ff_byte run did not terminate)")]
    SeiTruncated,
    #[error("filler RBSP contains a byte that is not 0xFF (got 0x{0:02X})")]
    FillerNonFfByte(u8),
}

/// §7.4.2.4 — Table 7-5 primary_pic_type.
/// Values 0..=7 where each row is the set of slice_type values allowed
/// in the access unit. This enum wraps the raw value; callers should
/// decode using Table 7-5 semantics as needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimaryPicType(pub u8); // 0..=7

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessUnitDelimiter {
    pub primary_pic_type: PrimaryPicType,
}

impl AccessUnitDelimiter {
    /// §7.3.2.4 — parse access_unit_delimiter_rbsp. The RBSP body is
    /// 3 bits of primary_pic_type + rbsp_trailing_bits.
    pub fn parse(rbsp: &[u8]) -> Result<Self, NonVclError> {
        let mut r = BitReader::new(rbsp);
        // §7.3.2.4 — primary_pic_type u(3)
        let ppt = r.u(3)?;
        // u(3) can only yield 0..=7, but keep the check for the API
        // contract (NonVclError::AudPicTypeOutOfRange is part of the
        // public error set).
        if ppt > 7 {
            return Err(NonVclError::AudPicTypeOutOfRange(ppt));
        }
        // §7.3.2.4 — rbsp_trailing_bits()
        r.rbsp_trailing_bits()?;
        Ok(Self {
            primary_pic_type: PrimaryPicType(ppt as u8),
        })
    }
}

/// §7.3.2.3.1 — one SEI message: payload_type + payload_size + raw payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeiMessage {
    pub payload_type: u32,
    pub payload_size: u32,
    /// Raw payload bytes (caller interprets per Annex D for known types).
    pub payload: Vec<u8>,
}

/// §7.3.2.3 — parse sei_rbsp into a sequence of messages. Each message
/// consumes: ff_byte-run for payload_type, last_payload_type_byte,
/// ff_byte-run for payload_size, last_payload_size_byte, then
/// `payload_size` bytes of payload, ending when more_rbsp_data()
/// returns false.
pub fn parse_sei_rbsp(rbsp: &[u8]) -> Result<Vec<SeiMessage>, NonVclError> {
    let mut r = BitReader::new(rbsp);
    let mut messages = Vec::new();

    // §7.3.2.3 — do { sei_message() } while (more_rbsp_data())
    // Empty SEI RBSPs are legal in practice (a lone rbsp_stop_one_bit
    // with no messages); guard against them before the first iteration.
    if !r.more_rbsp_data() {
        r.rbsp_trailing_bits()?;
        return Ok(messages);
    }

    loop {
        messages.push(parse_sei_message(&mut r)?);
        if !r.more_rbsp_data() {
            break;
        }
    }

    // §7.3.2.3 — rbsp_trailing_bits() closes the RBSP after the loop.
    r.rbsp_trailing_bits()?;
    Ok(messages)
}

/// §7.3.2.3.1 — single sei_message. The reader must be byte aligned on
/// entry (SEI messages are specified as sequences of f(8) bytes).
fn parse_sei_message(r: &mut BitReader<'_>) -> Result<SeiMessage, NonVclError> {
    // §7.3.2.3.1 — payload_type = sum of all ff_byte (each 0xFF) +
    // last_payload_type_byte (u(8)).
    let payload_type = read_variable_length_u8_sum(r)?;
    // §7.3.2.3.1 — payload_size computed the same way.
    let payload_size = read_variable_length_u8_sum(r)?;

    // §7.3.2.3.1 — sei_payload(payloadType, payloadSize): `payload_size`
    // raw bytes. We do NOT interpret the payload (Annex D territory).
    let mut payload = Vec::with_capacity(payload_size as usize);
    for _ in 0..payload_size {
        payload.push(r.u(8)? as u8);
    }

    Ok(SeiMessage {
        payload_type,
        payload_size,
        payload,
    })
}

/// §7.3.2.3.1 — read a run of `ff_byte`s (0xFF) followed by one final
/// `last_*_byte` (u(8) < 0xFF in normal encodings, but the spec accepts
/// any final u(8)); returns the sum. The loop must terminate before the
/// reader runs out of bytes.
fn read_variable_length_u8_sum(r: &mut BitReader<'_>) -> Result<u32, NonVclError> {
    let mut total: u32 = 0;
    loop {
        if r.bits_remaining() < 8 {
            return Err(NonVclError::SeiTruncated);
        }
        let byte = r.u(8)?;
        total = total.saturating_add(byte);
        if byte != 0xFF {
            // This was last_payload_{type,size}_byte.
            return Ok(total);
        }
        // Otherwise it was an ff_byte; keep accumulating.
    }
}

/// §7.3.2.5 — end_of_seq_rbsp. Body is empty (no trailing bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndOfSequence;

impl EndOfSequence {
    /// §7.3.2.5 — end_of_seq_rbsp() carries no syntax elements.
    pub fn parse(_rbsp: &[u8]) -> Result<Self, NonVclError> {
        Ok(Self)
    }
}

/// §7.3.2.6 — end_of_stream_rbsp. Body is empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndOfStream;

impl EndOfStream {
    /// §7.3.2.6 — end_of_stream_rbsp() carries no syntax elements.
    pub fn parse(_rbsp: &[u8]) -> Result<Self, NonVclError> {
        Ok(Self)
    }
}

/// §7.3.2.7 — filler_data_rbsp. Returns the count of ff_byte's consumed.
pub fn parse_filler_data(rbsp: &[u8]) -> Result<u32, NonVclError> {
    let mut r = BitReader::new(rbsp);
    let mut count: u32 = 0;

    // §7.3.2.7 — while (next_bits(8) == 0xFF) ff_byte f(8).
    // The loop stops as soon as the next byte is not 0xFF; that byte
    // must be the first byte of rbsp_trailing_bits() (i.e. contain the
    // stop_one_bit as its MSB and zero-pad to the byte boundary).
    while r.bits_remaining() >= 8 {
        // Peek the next 8 bits without consuming.
        let (byte_pos, bit_pos) = r.position();
        let next = r.u(8)?;
        if next == 0xFF {
            count += 1;
            continue;
        }
        // Not an ff_byte: this must be rbsp_trailing_bits(). The only
        // legal first byte of rbsp_trailing_bits at byte-alignment is
        // 0x80 (stop_one_bit followed by seven zero pad bits).
        // Anything else is a malformed filler NAL.
        if next != 0x80 {
            return Err(NonVclError::FillerNonFfByte(next as u8));
        }
        // Put the cursor back before the 0x80 byte so
        // rbsp_trailing_bits() can consume it via its normal routine.
        // (BitReader has no seek, but since we know the only legal
        // pre-trailing state is byte-aligned, we just verify the byte
        // was the stop byte and return. Using the peeked position is
        // only for clarity — the byte has already been consumed and
        // validated.)
        let _ = (byte_pos, bit_pos);
        // Verify no unexpected trailing data after the stop byte.
        if r.bits_remaining() != 0 {
            // §7.3.2.11 rbsp_trailing_bits must end exactly at EOF once
            // byte aligned; any leftover bytes are malformed.
            let leftover = r.u(8)? as u8;
            return Err(NonVclError::FillerNonFfByte(leftover));
        }
        return Ok(count);
    }

    // Ran out of bytes without seeing rbsp_trailing_bits.
    Err(NonVclError::SeiTruncated)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: pack a raw `bits` bit-string (MSB-first) into bytes,
    // padding the final byte with zeros. Used by AUD trailing-bits
    // tests to construct exact bit patterns.
    fn pack_bits(bits: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity((bits.len() + 7) / 8);
        let mut cur: u8 = 0;
        let mut n: u8 = 0;
        for &b in bits {
            cur = (cur << 1) | (b & 1);
            n += 1;
            if n == 8 {
                out.push(cur);
                cur = 0;
                n = 0;
            }
        }
        if n != 0 {
            cur <<= 8 - n;
            out.push(cur);
        }
        out
    }

    // ---- AUD ----

    #[test]
    fn aud_primary_pic_type_zero() {
        // primary_pic_type = 000, stop_one_bit = 1, four zero pad bits.
        // Bits: 0 0 0 1 0 0 0 0 → 0x10.
        let aud = AccessUnitDelimiter::parse(&[0x10]).unwrap();
        assert_eq!(aud.primary_pic_type, PrimaryPicType(0));
    }

    #[test]
    fn aud_primary_pic_type_seven() {
        // primary_pic_type = 111, stop_one_bit = 1, four zero pad bits.
        // Bits: 1 1 1 1 0 0 0 0 → 0xF0.
        let aud = AccessUnitDelimiter::parse(&[0xF0]).unwrap();
        assert_eq!(aud.primary_pic_type, PrimaryPicType(7));
    }

    #[test]
    fn aud_all_values_roundtrip() {
        for v in 0u8..=7 {
            // 3 bits of v, then 1 stop bit, then zero pad to byte boundary.
            let bits: Vec<u8> = (0..3)
                .map(|i| (v >> (2 - i)) & 1)
                .chain(std::iter::once(1))
                .chain(std::iter::repeat(0).take(4))
                .collect();
            let bytes = pack_bits(&bits);
            let aud = AccessUnitDelimiter::parse(&bytes).unwrap();
            assert_eq!(aud.primary_pic_type, PrimaryPicType(v));
        }
    }

    #[test]
    fn aud_rejects_missing_trailing_bits() {
        // 000 then zero padding (no stop bit).
        let err = AccessUnitDelimiter::parse(&[0x00]).unwrap_err();
        assert_eq!(err, NonVclError::Bitstream(BitError::BadTrailingBits));
    }

    // ---- SEI ----

    #[test]
    fn sei_single_message() {
        // payload_type=1 (picture_timing), payload_size=4, payload=[0xAA, 0xBB, 0xCC, 0xDD].
        // Bytes: 0x01 0x04 AA BB CC DD | 0x80 (rbsp_trailing_bits: stop_bit + 7 zeros)
        let rbsp = [0x01, 0x04, 0xAA, 0xBB, 0xCC, 0xDD, 0x80];
        let msgs = parse_sei_rbsp(&rbsp).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].payload_type, 1);
        assert_eq!(msgs[0].payload_size, 4);
        assert_eq!(msgs[0].payload, vec![0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn sei_multibyte_payload_type() {
        // payload_type = 300 = 255 + 45 → one ff_byte (0xFF) + last_byte (0x2D).
        // payload_size = 2, payload = [0x11, 0x22].
        // Bytes: 0xFF 0x2D 0x02 0x11 0x22 | 0x80
        let rbsp = [0xFF, 0x2D, 0x02, 0x11, 0x22, 0x80];
        let msgs = parse_sei_rbsp(&rbsp).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].payload_type, 300);
        assert_eq!(msgs[0].payload_size, 2);
        assert_eq!(msgs[0].payload, vec![0x11, 0x22]);
    }

    #[test]
    fn sei_multibyte_payload_size() {
        // payload_type=5 (user_data_unregistered), payload_size=260 = 255+5
        // → one ff_byte + 0x05. Payload = 260 bytes of 0x77.
        let mut rbsp = vec![0x05, 0xFF, 0x05];
        rbsp.extend(std::iter::repeat(0x77).take(260));
        rbsp.push(0x80);
        let msgs = parse_sei_rbsp(&rbsp).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].payload_type, 5);
        assert_eq!(msgs[0].payload_size, 260);
        assert_eq!(msgs[0].payload.len(), 260);
        assert!(msgs[0].payload.iter().all(|&b| b == 0x77));
    }

    #[test]
    fn sei_two_messages() {
        // msg1: type=1, size=2, payload=[0xAA, 0xBB]
        // msg2: type=6 (recovery_point), size=1, payload=[0xCC]
        // Bytes: 01 02 AA BB 06 01 CC 80
        let rbsp = [0x01, 0x02, 0xAA, 0xBB, 0x06, 0x01, 0xCC, 0x80];
        let msgs = parse_sei_rbsp(&rbsp).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].payload_type, 1);
        assert_eq!(msgs[0].payload_size, 2);
        assert_eq!(msgs[0].payload, vec![0xAA, 0xBB]);
        assert_eq!(msgs[1].payload_type, 6);
        assert_eq!(msgs[1].payload_size, 1);
        assert_eq!(msgs[1].payload, vec![0xCC]);
    }

    #[test]
    fn sei_empty_rbsp() {
        // Just rbsp_trailing_bits (0x80 = stop_one_bit + zero pad).
        let msgs = parse_sei_rbsp(&[0x80]).unwrap();
        assert!(msgs.is_empty());
    }

    #[test]
    fn sei_zero_size_payload() {
        // type=0 (buffering_period), size=0, no payload bytes.
        // Bytes: 00 00 80
        let rbsp = [0x00, 0x00, 0x80];
        let msgs = parse_sei_rbsp(&rbsp).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].payload_type, 0);
        assert_eq!(msgs[0].payload_size, 0);
        assert!(msgs[0].payload.is_empty());
    }

    // ---- Filler ----

    #[test]
    fn filler_three_ff_bytes() {
        // FF FF FF | 80 (rbsp_trailing_bits).
        let count = parse_filler_data(&[0xFF, 0xFF, 0xFF, 0x80]).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn filler_zero_ff_bytes() {
        // Just rbsp_trailing_bits.
        let count = parse_filler_data(&[0x80]).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn filler_rejects_non_ff_byte() {
        // FF 00 80 → 0x00 is neither ff_byte nor a valid stop byte.
        let err = parse_filler_data(&[0xFF, 0x00, 0x80]).unwrap_err();
        assert_eq!(err, NonVclError::FillerNonFfByte(0x00));
    }

    #[test]
    fn filler_rejects_garbage_after_stop_byte() {
        // Stop byte looks legal, but trailing garbage follows.
        let err = parse_filler_data(&[0xFF, 0x80, 0x42]).unwrap_err();
        assert_eq!(err, NonVclError::FillerNonFfByte(0x42));
    }

    // ---- End of sequence / End of stream ----

    #[test]
    fn end_of_sequence_empty() {
        assert_eq!(EndOfSequence::parse(&[]).unwrap(), EndOfSequence);
    }

    #[test]
    fn end_of_stream_empty() {
        assert_eq!(EndOfStream::parse(&[]).unwrap(), EndOfStream);
    }

    #[test]
    fn end_of_sequence_ignores_body() {
        // Per §7.3.2.5 the RBSP is empty; our parser doesn't inspect it.
        assert_eq!(EndOfSequence::parse(&[0x00, 0x11]).unwrap(), EndOfSequence);
    }

    #[test]
    fn end_of_stream_ignores_body() {
        assert_eq!(EndOfStream::parse(&[0xDE, 0xAD]).unwrap(), EndOfStream);
    }
}
