//! §7.2 — bitstream reader and Exp-Golomb decoding.
//!
//! H.264 reads bits MSB-first within each byte (spec §7.2: u(n) "with
//! most significant bit written first"). This reader operates on a
//! borrowed `&[u8]` and tracks the cursor as `(byte_index, bit_index)`
//! where `bit_index` is the offset of the next bit to consume within
//! the current byte (0 = MSB, 7 = LSB; once it reaches 8 the byte
//! index advances).
//!
//! Callers feed RBSP bytes (with emulation-prevention bytes already
//! stripped) for syntax inside a parameter-set or slice body. The NAL
//! unit header — which is parsed before stripping — is just one byte,
//! so a temporary one-byte reader handles it.
//!
//! Descriptors implemented here:
//!
//! | Spec | Function           | Notes                                    |
//! | ---- | ------------------ | ---------------------------------------- |
//! | §7.2 | `u(n)`             | unsigned, n ≤ 32                         |
//! | §7.2 | `i(n)`             | signed two's complement, n ≤ 32          |
//! | §7.2 | `f(n)`             | fixed pattern — same parsing as `u(n)`   |
//! | §7.2 | `ue(v)`            | unsigned Exp-Golomb (§9.1)               |
//! | §7.2 | `se(v)`            | signed Exp-Golomb (§9.1.1)               |
//! | §7.2 | `te(v)`            | truncated Exp-Golomb (§9.1.2)            |
//! | §7.2 | `byte_aligned()`   | cursor is at start of a byte             |
//! | §7.2 | `more_rbsp_data()` | data exists before `rbsp_trailing_bits()`|
//! | §7.3.2.11 | `rbsp_trailing_bits()` | consume stop bit + zero-pad      |
//!
//! `me(v)` (mapped Exp-Golomb, §9.1.2) and `ce(v)` / `ae(v)` (CAVLC /
//! CABAC entropy decoding, §9.2 / §9.3) live with their respective
//! decoders rather than in this module.

#![allow(dead_code)]
// Each method is exercised by tests in this module and will be called
// from the NAL/SPS/PPS parsers as they land — silence dead-code while
// the rest of the crate is still empty.

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BitError {
    #[error("attempted to read past end of bitstream")]
    Eof,
    #[error("Exp-Golomb leading-zero run exceeded 31 bits")]
    ExpGolombOverflow,
    #[error("u(n)/i(n) requested {0} bits, max supported is 32")]
    TooManyBits(u32),
    #[error("malformed rbsp_trailing_bits(): expected stop_one_bit then zeros")]
    BadTrailingBits,
}

pub type BitResult<T> = Result<T, BitError>;

pub struct BitReader<'a> {
    data: &'a [u8],
    /// Index of the byte holding the next bit to consume.
    byte_pos: usize,
    /// Offset of the next bit within `data[byte_pos]`. 0 = MSB.
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// `(byte_index, bit_index)`. `bit_index` is 0..=7, MSB-first.
    pub fn position(&self) -> (usize, u8) {
        (self.byte_pos, self.bit_pos)
    }

    pub fn bits_remaining(&self) -> usize {
        if self.byte_pos >= self.data.len() {
            0
        } else {
            (self.data.len() - self.byte_pos) * 8 - self.bit_pos as usize
        }
    }

    /// §7.2 `byte_aligned()`.
    pub fn byte_aligned(&self) -> bool {
        self.bit_pos == 0
    }

    /// §7.2 `u(n)`. `n` must be ≤ 32.
    pub fn u(&mut self, bits: u32) -> BitResult<u32> {
        if bits > 32 {
            return Err(BitError::TooManyBits(bits));
        }
        if self.bits_remaining() < bits as usize {
            return Err(BitError::Eof);
        }
        let mut value: u32 = 0;
        for _ in 0..bits {
            value = (value << 1) | self.read_bit_unchecked() as u32;
        }
        Ok(value)
    }

    /// §7.2 `i(n)`. Two's complement, `n` ≤ 32.
    pub fn i(&mut self, bits: u32) -> BitResult<i32> {
        let raw = self.u(bits)?;
        if bits == 0 {
            return Ok(0);
        }
        let sign_bit = 1u32 << (bits - 1);
        Ok(if bits == 32 {
            raw as i32
        } else if raw & sign_bit != 0 {
            (raw | !((1u32 << bits) - 1)) as i32
        } else {
            raw as i32
        })
    }

    /// §7.2 `f(n)`. Same parsing as `u(n)`; the distinction in the
    /// spec is editorial (the value is fixed by syntax, not data).
    pub fn f(&mut self, bits: u32) -> BitResult<u32> {
        self.u(bits)
    }

    /// §7.2 `ue(v)`. Unsigned Exp-Golomb (§9.1).
    pub fn ue(&mut self) -> BitResult<u32> {
        self.read_codenum()
    }

    /// §7.2 `se(v)`. Signed Exp-Golomb mapping (§9.1.1):
    /// `value = (-1)^(k+1) * ceil(k / 2)` for codeNum `k`.
    pub fn se(&mut self) -> BitResult<i32> {
        let k = self.read_codenum()?;
        Ok(if k & 1 == 1 {
            k.div_ceil(2) as i32
        } else {
            -((k / 2) as i32)
        })
    }

    /// §7.2 `te(v)`. Truncated Exp-Golomb (§9.1.2, eq. 9-3).
    ///
    /// - If `x_max == 0`, no bits are read — the value is always 0.
    /// - If `x_max == 1`, a single inverted bit is parsed
    ///   (`codeNum = !read_bits(1)`).
    /// - Otherwise (`x_max > 1`), it falls through to `ue(v)`.
    pub fn te(&mut self, x_max: u32) -> BitResult<u32> {
        if x_max == 0 {
            Ok(0)
        } else if x_max == 1 {
            Ok(if self.u(1)? == 0 { 1 } else { 0 })
        } else {
            self.ue()
        }
    }

    /// §7.2 `more_rbsp_data()`. Locate the rbsp_stop_one_bit (the
    /// last `1` bit in the RBSP) and report whether the cursor is
    /// strictly before it.
    pub fn more_rbsp_data(&self) -> bool {
        if self.bits_remaining() == 0 {
            return false;
        }
        let mut last_one: Option<(usize, u8)> = None;
        for byte_idx in 0..self.data.len() {
            let b = self.data[byte_idx];
            if b == 0 {
                continue;
            }
            for bit_idx in 0..8u8 {
                if (b >> (7 - bit_idx)) & 1 == 1 {
                    last_one = Some((byte_idx, bit_idx));
                }
            }
        }
        match last_one {
            None => false,
            Some((b, bi)) => match self.byte_pos.cmp(&b) {
                core::cmp::Ordering::Less => true,
                core::cmp::Ordering::Equal => self.bit_pos < bi,
                core::cmp::Ordering::Greater => false,
            },
        }
    }

    /// §7.3.2.11 `rbsp_trailing_bits()` — consume `rbsp_stop_one_bit`
    /// then `rbsp_alignment_zero_bit` until byte aligned.
    pub fn rbsp_trailing_bits(&mut self) -> BitResult<()> {
        if self.u(1)? != 1 {
            return Err(BitError::BadTrailingBits);
        }
        while !self.byte_aligned() {
            if self.u(1)? != 0 {
                return Err(BitError::BadTrailingBits);
            }
        }
        Ok(())
    }

    /// §9.1 — read the leading-zero count, then the `leadingZeroBits`
    /// suffix; return `codeNum = 2^leadingZeroBits − 1 + suffix`.
    ///
    /// `leadingZeroBits` is bounded at 31: the formula `2^k − 1 + suffix`
    /// already saturates `u32::MAX` at `k = 32` (since `suffix` itself
    /// can be `2^32 − 1`), and `1u32 << 32` is UB in Rust. Anything past
    /// 31 zeros indicates malformed input — the spec syntax elements
    /// (slice headers, MB types, ref indices, etc.) all top out well
    /// below this, so a stream that asks for `k ≥ 32` is corrupt by
    /// definition. Return `ExpGolombOverflow` rather than panic.
    fn read_codenum(&mut self) -> BitResult<u32> {
        let mut leading_zeros: u32 = 0;
        loop {
            if self.bits_remaining() == 0 {
                return Err(BitError::Eof);
            }
            if self.read_bit_unchecked() == 1 {
                break;
            }
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err(BitError::ExpGolombOverflow);
            }
        }
        let suffix = if leading_zeros == 0 {
            0
        } else {
            self.u(leading_zeros)?
        };
        Ok((1u32 << leading_zeros) - 1 + suffix)
    }

    fn read_bit_unchecked(&mut self) -> u8 {
        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        bit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u_reads_msb_first() {
        let mut r = BitReader::new(&[0b1010_1100, 0b1111_0000]);
        assert_eq!(r.u(1).unwrap(), 1);
        assert_eq!(r.u(3).unwrap(), 0b010);
        assert_eq!(r.u(4).unwrap(), 0b1100);
        assert_eq!(r.u(8).unwrap(), 0b1111_0000);
        assert_eq!(r.u(1).unwrap_err(), BitError::Eof);
    }

    #[test]
    fn u_zero_bits_yields_zero() {
        let mut r = BitReader::new(&[0xff]);
        assert_eq!(r.u(0).unwrap(), 0);
        assert_eq!(r.position(), (0, 0));
    }

    #[test]
    fn u_full_32_bits() {
        let mut r = BitReader::new(&[0x12, 0x34, 0x56, 0x78]);
        assert_eq!(r.u(32).unwrap(), 0x1234_5678);
    }

    #[test]
    fn i_sign_extends() {
        let mut r = BitReader::new(&[0b1111_0000]);
        assert_eq!(r.i(4).unwrap(), -1);
        assert_eq!(r.i(4).unwrap(), 0);

        let mut r = BitReader::new(&[0b1000_0000]);
        assert_eq!(r.i(8).unwrap(), -128);
    }

    #[test]
    fn ue_table_9_1_values() {
        // Spec §9.1, Table 9-1: codeword bit string → codeNum.
        // 1                      → 0
        // 010                    → 1
        // 011                    → 2
        // 00100                  → 3
        // 00101                  → 4
        // 00110                  → 5
        // 00111                  → 6
        // 0001000                → 7
        let cases: &[(&[u8], &[u32])] = &[
            // bit-string "1" packed in MSB position, padded with 0s
            (&[0b1000_0000], &[0]),
            (&[0b0100_0000], &[1]),
            (&[0b0110_0000], &[2]),
            (&[0b0010_0000], &[3]),
            (&[0b0010_1000], &[4]),
            (&[0b0011_0000], &[5]),
            (&[0b0011_1000], &[6]),
            (&[0b0001_0000], &[7]),
            // Concatenated: "1 010 011" = 1, 1, 2
            (&[0b1010_0110], &[0, 1, 2]),
        ];
        for (bytes, want) in cases {
            let mut r = BitReader::new(bytes);
            for &v in *want {
                assert_eq!(r.ue().unwrap(), v, "input bytes {:08b}", bytes[0]);
            }
        }
    }

    #[test]
    fn se_table_9_3_mapping() {
        // §9.1.1 / Table 9-3: codeNum → signed value.
        // 0→0, 1→1, 2→-1, 3→2, 4→-2, 5→3, 6→-3
        // We synthesise ue codewords and run them through se().
        let cases: &[(u8, i32)] = &[
            (0b1000_0000, 0),
            (0b0100_0000, 1),
            (0b0110_0000, -1),
            (0b0010_0000, 2),
            (0b0010_1000, -2),
            (0b0011_0000, 3),
            (0b0011_1000, -3),
        ];
        for (byte, want) in cases {
            let bytes = [*byte];
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.se().unwrap(), *want);
        }
    }

    #[test]
    fn te_x_max_one_inverts_bit() {
        let mut r = BitReader::new(&[0b1000_0000]);
        assert_eq!(r.te(1).unwrap(), 0); // bit was 1
        assert_eq!(r.te(1).unwrap(), 1); // bit was 0

        // x_max > 1 falls through to ue().
        let mut r = BitReader::new(&[0b0100_0000]);
        assert_eq!(r.te(7).unwrap(), 1);
    }

    #[test]
    fn te_x_max_zero_consumes_no_bits() {
        // §9.1.2 eq. 9-3 degenerate case: when the range is {0} (i.e.
        // x_max == 0), no bits are read and the value is 0. The
        // pre-fix behaviour was to fall through to ue() and eat bits.
        let mut r = BitReader::new(&[0xFFu8, 0xFFu8]);
        assert_eq!(r.te(0).unwrap(), 0);
        // Reader should still be at (0, 0) since nothing was consumed.
        let (byte, bit) = r.position();
        assert_eq!((byte, bit), (0, 0));
    }

    #[test]
    fn more_rbsp_data_locates_stop_bit() {
        // RBSP: ue=0 (bit "1") then rbsp_trailing_bits.
        // After the ue, byte = "1 1 0000000" = 0xC0, byte aligned at bit 1.
        // The stop_one_bit is the second bit. After consuming the ue,
        // cursor is at bit 1 == position of stop bit → no more data.
        let mut r = BitReader::new(&[0b1100_0000]);
        assert_eq!(r.ue().unwrap(), 0);
        assert!(!r.more_rbsp_data());

        // RBSP: ue=0, ue=0, then trailing bits.
        // byte = "1 1 1 00000" = 0xE0. After two ue reads cursor at bit 2,
        // stop_one_bit at bit 2 → no more data.
        let mut r = BitReader::new(&[0b1110_0000]);
        assert_eq!(r.ue().unwrap(), 0);
        assert!(r.more_rbsp_data());
        assert_eq!(r.ue().unwrap(), 0);
        assert!(!r.more_rbsp_data());
    }

    #[test]
    fn rbsp_trailing_bits_consumes_stop_and_pad() {
        // "11" followed by zeros, then trailing bits = "1 000000".
        // Combined byte = "11 1 00000" = 0xE0.
        let mut r = BitReader::new(&[0b1110_0000]);
        r.u(2).unwrap(); // skip the "11"
        r.rbsp_trailing_bits().unwrap();
        assert_eq!(r.bits_remaining(), 0);
    }

    #[test]
    fn rbsp_trailing_bits_rejects_missing_stop_bit() {
        let mut r = BitReader::new(&[0b0000_0000]);
        assert_eq!(
            r.rbsp_trailing_bits().unwrap_err(),
            BitError::BadTrailingBits
        );
    }

    #[test]
    fn ue_reports_eof() {
        let mut r = BitReader::new(&[0x00]);
        assert_eq!(r.ue().unwrap_err(), BitError::Eof);
    }

    #[test]
    fn ue_rejects_oversize_leading_zero_run() {
        // §9.1: 32 leading zeros followed by a `1` would require
        // `1u32 << 32` to compute the codeNum, which is undefined and
        // panics in debug builds. Reject with `ExpGolombOverflow`
        // before the shift. Regression for fuzz crash
        // panic_free_decode/crash-5c9d8c8642cdb8ffc371d3dbb45c3a1604477676
        // (PPS parser propagated an unbounded `ue(v)` decode).
        // 32 zero bits = 4 bytes of 0x00, then a `1` bit.
        let bytes = [0x00, 0x00, 0x00, 0x00, 0b1000_0000];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.ue().unwrap_err(), BitError::ExpGolombOverflow);

        // Even a much longer run of zeros (well past 32) must not panic
        // — the reader bails as soon as it counts 32 zero bits.
        let bytes = [0u8; 16];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.ue().unwrap_err(), BitError::ExpGolombOverflow);
    }

    #[test]
    fn ue_accepts_31_leading_zeros_max_codenum() {
        // 31 leading zeros + `1` + 31-bit suffix is the largest legal
        // codeNum that fits in u32: `(1 << 31) - 1 + (2^31 - 1)`
        // = `2^32 - 2`. Make sure the boundary case decodes without
        // panicking.
        // Layout: 31 zeros, then `1`, then 31 suffix bits all `1`.
        // Total = 63 bits; pack into 8 bytes (last bit padding).
        let mut bits: Vec<u8> = Vec::new();
        bits.extend(std::iter::repeat_n(0, 31));
        bits.push(1); // the terminating one
        bits.extend(std::iter::repeat_n(1, 31)); // suffix = 2^31 - 1
        bits.push(0); // pad to 64
        let mut bytes = [0u8; 8];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                bytes[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.ue().unwrap(), u32::MAX - 1);
    }
}
