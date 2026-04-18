//! MSB-first bit writer for H.264 RBSP emission with Exp-Golomb helpers.
//!
//! Mirror of [`crate::bitreader`] on the emit side. Bytes are accumulated in
//! an internal buffer, always MSB-first. The writer is used by the encoder
//! to build SPS / PPS / slice RBSP payloads; a separate
//! [`rbsp_to_ebsp`] pass then converts the RBSP to an EBSP (Encapsulated
//! Byte Sequence Payload) by re-inserting §7.4.1.1 emulation-prevention
//! bytes before the NAL is framed with a 4-byte Annex B start code.
//!
//! References: ITU-T H.264 (07/2019) §7.2 / §9.1 (Exp-Golomb),
//! §7.3.2.11 (`rbsp_trailing_bits`), §7.4.1.1 (emulation prevention).

/// MSB-first bit writer producing a byte stream. `finish()` pads out the
/// final byte with zeros — callers that care about bit-exact alignment
/// should emit an `rbsp_trailing_bits` syntax element (a `1` bit then
/// zero-padding to the next byte boundary) before `finish`.
#[derive(Default)]
pub struct BitWriter {
    buf: Vec<u8>,
    /// Number of bits already emitted in the "current" (last, partial)
    /// byte. `0` means the buffer is byte-aligned; `1..=7` means the last
    /// byte is partial and the next bit goes into position `7 - cur_bit`
    /// within it.
    cur_bit: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            cur_bit: 0,
        }
    }

    /// Number of bits written so far.
    pub fn bits_written(&self) -> u64 {
        (self.buf.len() as u64) * 8 - if self.cur_bit == 0 { 0 } else { 8 - self.cur_bit as u64 }
    }

    pub fn is_byte_aligned(&self) -> bool {
        self.cur_bit == 0
    }

    /// Write the low `n` bits of `v`, MSB-first. `n` must be <= 32.
    pub fn write_bits(&mut self, v: u32, n: u32) {
        debug_assert!(n <= 32);
        if n == 0 {
            return;
        }
        let mut remaining = n;
        while remaining > 0 {
            if self.cur_bit == 0 {
                self.buf.push(0);
            }
            let byte_ix = self.buf.len() - 1;
            let free_in_byte = 8 - self.cur_bit;
            let take = remaining.min(free_in_byte);
            // Extract the next `take` bits (the top `take` of the `remaining`-bit window).
            let shift_down = remaining - take;
            let bits = (v >> shift_down) & ((1u32 << take) - 1);
            // Pack them into the current byte.
            let byte_shift = free_in_byte - take;
            self.buf[byte_ix] |= (bits as u8) << byte_shift;
            self.cur_bit = (self.cur_bit + take) % 8;
            remaining -= take;
        }
    }

    pub fn write_bit(&mut self, b: u32) {
        self.write_bits(b & 1, 1);
    }

    pub fn write_flag(&mut self, b: bool) {
        self.write_bit(if b { 1 } else { 0 });
    }

    /// Unsigned Exp-Golomb (`ue(v)`), §9.1.
    pub fn write_ue(&mut self, v: u32) {
        // code_num = v; leading_zeros = floor(log2(v+1)); code = v + 1 with
        // `leading_zeros` leading zeros prepended.
        if v == u32::MAX {
            // v+1 overflows. H.264 restricts ue(v) to < 2^32 - 1, so this is
            // rejected here to keep the encoder honest.
            panic!("bitwriter: ue(v) overflow (v == u32::MAX)");
        }
        let x = v + 1;
        let leading_zeros = 31 - x.leading_zeros();
        for _ in 0..leading_zeros {
            self.write_bit(0);
        }
        // Write the x itself as (leading_zeros+1) bits — its MSB is the 1 terminator.
        self.write_bits(x, leading_zeros + 1);
    }

    /// Signed Exp-Golomb (`se(v)`), §9.1.1.
    pub fn write_se(&mut self, v: i32) {
        // mapping: 0 -> 0, k>0 -> 2k-1, k<0 -> -2k
        let code = if v == 0 {
            0u32
        } else if v > 0 {
            (v as u32) * 2 - 1
        } else {
            (-v) as u32 * 2
        };
        self.write_ue(code);
    }

    /// Truncated Exp-Golomb (`te(v)`), §9.1.2 — `x` is the declared upper
    /// bound of the syntax element.
    pub fn write_te(&mut self, v: u32, x: u32) {
        if x == 1 {
            // 1-bit, reverse polarity: 0 -> 1, 1 -> 0
            self.write_bit(if v == 0 { 1 } else { 0 });
        } else {
            self.write_ue(v);
        }
    }

    /// Emit `rbsp_trailing_bits()` (§7.3.2.11) — a stop bit `1` followed by
    /// zero-padding to the next byte boundary. Leaves the buffer byte-aligned.
    pub fn write_rbsp_trailing_bits(&mut self) {
        self.write_bit(1);
        while !self.is_byte_aligned() {
            self.write_bit(0);
        }
    }

    /// Finish writing. Remaining bits in the final partial byte have already
    /// been zero-padded by [`write_bits`]. Returns the underlying byte buffer.
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }

    /// Borrow the in-progress buffer (tests + helpers).
    pub fn bytes(&self) -> &[u8] {
        &self.buf
    }
}

/// Convert a raw RBSP byte stream into an EBSP (Encapsulated Byte Sequence
/// Payload) — i.e. insert §7.4.1.1 emulation-prevention `0x03` bytes so the
/// resulting stream cannot contain an Annex B start-code sequence or a
/// `0x00 0x00 0x00`/`0x00 0x00 0x01`/`0x00 0x00 0x02`/`0x00 0x00 0x03`
/// triplet.
///
/// Per the spec, for every triplet `0x00 0x00 0xAB` where `0xAB <= 0x03`,
/// an `0x03` byte is inserted before the third byte. Also, if the final
/// two bytes are both `0x00`, an extra `0x03` is appended as cabac_zero_word
/// protection. We keep the common case only — the extra-trailing-zero
/// condition is handled by never emitting trailing zeros.
pub fn rbsp_to_ebsp(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + rbsp.len() / 100);
    let mut i = 0;
    while i < rbsp.len() {
        let b = rbsp[i];
        if out.len() >= 2
            && out[out.len() - 1] == 0
            && out[out.len() - 2] == 0
            && b <= 0x03
        {
            out.push(0x03);
        }
        out.push(b);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;

    #[test]
    fn ue_roundtrip_small() {
        let mut w = BitWriter::new();
        for v in 0..32u32 {
            w.write_ue(v);
        }
        w.write_rbsp_trailing_bits();
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for v in 0..32u32 {
            assert_eq!(r.read_ue().unwrap(), v, "ue mismatch at {v}");
        }
    }

    #[test]
    fn se_roundtrip_small() {
        let mut w = BitWriter::new();
        for v in -16..=16i32 {
            w.write_se(v);
        }
        w.write_rbsp_trailing_bits();
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for v in -16..=16i32 {
            assert_eq!(r.read_se().unwrap(), v, "se mismatch at {v}");
        }
    }

    #[test]
    fn bits_msb_first() {
        let mut w = BitWriter::new();
        w.write_bits(0xA, 4);
        w.write_bits(0xBC, 8);
        w.write_bits(0xD, 4);
        let bytes = w.finish();
        assert_eq!(bytes, vec![0xAB, 0xCD]);
    }

    #[test]
    fn rbsp_to_ebsp_inserts_03() {
        // 00 00 00 -> 00 00 03 00
        // 00 00 01 -> 00 00 03 01
        // 00 00 02 -> 00 00 03 02
        // 00 00 03 -> 00 00 03 03
        // 00 00 04 -> 00 00 04 (no insertion)
        let rbsp = [0x00u8, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x04];
        let ebsp = rbsp_to_ebsp(&rbsp);
        assert_eq!(
            ebsp,
            vec![0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x01, 0x00, 0x00, 0x04]
        );
    }

    #[test]
    fn rbsp_trailing_bits_aligns() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        w.write_rbsp_trailing_bits();
        let b = w.finish();
        assert_eq!(b.len(), 1);
        // 101 then stop bit 1 then 4 zeros -> 1011 0000 = 0xB0
        assert_eq!(b[0], 0xB0);
    }
}
