//! §7.2 — bitstream writer (encoder side).
//!
//! Mirror of [`crate::bitstream::BitReader`] for the encoder. Writes bits
//! MSB-first within each byte, supports the same descriptors used by the
//! parser:
//!
//! | Spec | Function | Notes |
//! | ---- | -------- | ----- |
//! | §7.2 | `u(n)`   | unsigned, n ≤ 32 |
//! | §7.2 | `i(n)`   | signed two's complement, n ≤ 32 |
//! | §7.2 | `ue(v)`  | unsigned Exp-Golomb (§9.1) |
//! | §7.2 | `se(v)`  | signed Exp-Golomb (§9.1.1) |
//! | §7.3.2.11 | `rbsp_trailing_bits()` | stop bit + zero pad |
//!
//! No external crates; self-contained.

#![allow(dead_code)]

/// MSB-first bit writer used by every encoder syntax module.
#[derive(Clone)]
pub struct BitWriter {
    bytes: Vec<u8>,
    /// Position of the next bit to write within the current byte
    /// (0 = MSB, 7 = LSB). When `bit_pos == 0` and `bytes` is non-empty,
    /// the previous byte is fully written; the next `write_bit` will
    /// push a new zero byte and start at MSB.
    bit_pos: u8,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_pos: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(cap),
            bit_pos: 0,
        }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn byte_aligned(&self) -> bool {
        self.bit_pos == 0
    }

    /// Total number of bits written so far.
    ///
    /// Useful for **rate measurement** in Lagrangian RDO: encode a
    /// candidate residual to a scratch [`BitWriter`], read the delta in
    /// `bits_emitted()` against the writer's pre-encode state, and use
    /// the result as the rate term `R` in `J = D + λ · R`.
    pub fn bits_emitted(&self) -> usize {
        let full_bytes = if self.bit_pos == 0 {
            self.bytes.len()
        } else {
            self.bytes.len() - 1
        };
        full_bytes * 8 + self.bit_pos as usize
    }

    /// §7.2 `u(n)`. `bits` ≤ 32. Lower-order `bits` of `value` are
    /// written, MSB of that field first.
    pub fn u(&mut self, bits: u32, value: u32) {
        debug_assert!(bits <= 32);
        for i in (0..bits).rev() {
            let bit = ((value >> i) & 1) as u8;
            self.write_bit(bit);
        }
    }

    /// §7.2 `i(n)`. Two's complement. `bits` ≤ 32.
    pub fn i(&mut self, bits: u32, value: i32) {
        debug_assert!(bits <= 32);
        let mask = if bits == 32 {
            u32::MAX
        } else {
            (1u32 << bits) - 1
        };
        self.u(bits, (value as u32) & mask);
    }

    /// §7.2 `f(n)` — fixed pattern; same encoding as `u(n)`.
    pub fn f(&mut self, bits: u32, value: u32) {
        self.u(bits, value);
    }

    /// §9.1 — `ue(v)` codeNum encoding.
    ///
    /// codeNum `k` → `leadingZeros = floor(log2(k+1))` zero bits followed
    /// by `(k+1)` written in `leadingZeros + 1` bits.
    pub fn ue(&mut self, value: u32) {
        let v = value
            .checked_add(1)
            .expect("ue(v) overflow: value+1 must fit in u32");
        // floor(log2(v)) for v >= 1.
        let leading = 31 - v.leading_zeros();
        for _ in 0..leading {
            self.write_bit(0);
        }
        // Total field width is `leading + 1` bits, holding `v`.
        self.u(leading + 1, v);
    }

    /// §9.1.1 — `se(v)` mapping. `value > 0` → codeNum `2*value-1`,
    /// `value <= 0` → codeNum `-2*value`.
    pub fn se(&mut self, value: i32) {
        let k = if value <= 0 {
            (value as i64).unsigned_abs().checked_mul(2).unwrap() as u32
        } else {
            ((value as u32).checked_mul(2).unwrap())
                .checked_sub(1)
                .unwrap()
        };
        self.ue(k);
    }

    /// §7.2 `te(v)` — truncated Exp-Golomb (§9.1.2 eq. 9-3). When `x_max
    /// == 1` we write a single inverted bit; for `x_max > 1` we delegate
    /// to `ue(v)`. `x_max == 0` writes nothing.
    pub fn te(&mut self, x_max: u32, value: u32) {
        if x_max == 0 {
            return;
        }
        if x_max == 1 {
            self.u(1, if value == 0 { 1 } else { 0 });
        } else {
            self.ue(value);
        }
    }

    /// §7.3.2.11 — `rbsp_trailing_bits()`. Writes a stop_one_bit
    /// followed by zero alignment bits up to the next byte boundary.
    pub fn rbsp_trailing_bits(&mut self) {
        self.u(1, 1);
        while !self.byte_aligned() {
            self.u(1, 0);
        }
    }

    /// Pad the cursor with zero bits to the next byte boundary without
    /// emitting any stop bit. Useful for syntax that ends on a byte
    /// boundary by construction.
    pub fn align_to_byte_zero(&mut self) {
        while !self.byte_aligned() {
            self.u(1, 0);
        }
    }

    fn write_bit(&mut self, bit: u8) {
        if self.bit_pos == 0 {
            self.bytes.push(0);
        }
        let idx = self.bytes.len() - 1;
        self.bytes[idx] |= bit << (7 - self.bit_pos);
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
        }
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitReader;

    #[test]
    fn u_roundtrips() {
        let mut w = BitWriter::new();
        w.u(1, 1);
        w.u(3, 0b010);
        w.u(4, 0b1100);
        w.u(8, 0b1111_0000);
        let bytes = w.into_bytes();
        assert_eq!(bytes, vec![0b1010_1100, 0b1111_0000]);

        let mut r = BitReader::new(&bytes);
        assert_eq!(r.u(1).unwrap(), 1);
        assert_eq!(r.u(3).unwrap(), 0b010);
        assert_eq!(r.u(4).unwrap(), 0b1100);
        assert_eq!(r.u(8).unwrap(), 0b1111_0000);
    }

    #[test]
    fn ue_roundtrips_table_9_1() {
        // §9.1 Table 9-1 reference values.
        let cases = [0u32, 1, 2, 3, 4, 5, 6, 7, 100, 1000];
        for &v in &cases {
            let mut w = BitWriter::new();
            w.ue(v);
            w.rbsp_trailing_bits();
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.ue().unwrap(), v, "value {v}");
        }
    }

    #[test]
    fn se_roundtrips_table_9_3() {
        // §9.1.1 Table 9-3 mapping.
        let cases = [0i32, 1, -1, 2, -2, 3, -3, 100, -100];
        for &v in &cases {
            let mut w = BitWriter::new();
            w.se(v);
            w.rbsp_trailing_bits();
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            assert_eq!(r.se().unwrap(), v, "value {v}");
        }
    }

    #[test]
    fn rbsp_trailing_bits_pads_to_byte() {
        let mut w = BitWriter::new();
        w.u(2, 0b11);
        w.rbsp_trailing_bits();
        // "11 1 00000" = 0xE0
        assert_eq!(w.into_bytes(), vec![0b1110_0000]);
    }

    #[test]
    fn ue_zero_writes_single_one_bit() {
        let mut w = BitWriter::new();
        w.ue(0);
        w.rbsp_trailing_bits();
        // ue(0) = "1", trailing produces nothing extra (already pads).
        // Combined byte = "1 1 000000" = 0xC0
        assert_eq!(w.into_bytes(), vec![0b1100_0000]);
    }

    #[test]
    fn bits_emitted_tracks_writes() {
        let mut w = BitWriter::new();
        assert_eq!(w.bits_emitted(), 0);
        w.u(1, 1);
        assert_eq!(w.bits_emitted(), 1);
        w.u(3, 0b010);
        assert_eq!(w.bits_emitted(), 4);
        w.u(4, 0b1100);
        assert_eq!(w.bits_emitted(), 8);
        w.u(8, 0b1111_0000);
        assert_eq!(w.bits_emitted(), 16);
        // Mid-byte position must be tracked.
        let mut w2 = BitWriter::new();
        w2.u(13, 0);
        assert_eq!(w2.bits_emitted(), 13);
    }

    #[test]
    fn i_negative_round_trips() {
        let mut w = BitWriter::new();
        w.i(8, -5);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.i(8).unwrap(), -5);
    }
}
