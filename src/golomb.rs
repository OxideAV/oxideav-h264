//! H.264-specific bit-reader / bit-writer extensions.
//!
//! Exp-Golomb codes, trailing-bits framing, and the `more_rbsp_data()`
//! sentinel from ITU-T H.264 §7 / §9.1 live here as extension traits
//! over [`oxideav_core::bits::BitReader`] and
//! [`oxideav_core::bits::BitWriter`]. The shared readers/writers stay
//! codec-agnostic; H.264 pulls these in via
//! `use oxideav_h264::golomb::{BitReaderExt, BitWriterExt};`.

use oxideav_core::{
    bits::{BitReader, BitWriter},
    Error, Result,
};

pub trait BitReaderExt {
    /// Read a `u(1)` flag as bool.
    fn read_flag(&mut self) -> Result<bool>;

    /// Peek up to `n` bits, padding with zeros if fewer are available.
    /// Used by the CAVLC total-zeros / run-before tables where the
    /// lookup is done against a possibly-truncated suffix.
    fn peek_u32_lax(&mut self, n: u32) -> u32;

    /// Unsigned Exp-Golomb `ue(v)` — §9.1.
    fn read_ue(&mut self) -> Result<u32>;

    /// Signed Exp-Golomb `se(v)` — §9.1.1.
    fn read_se(&mut self) -> Result<i32>;

    /// Truncated Exp-Golomb `te(v)` — §9.1.2. `x` is the declared
    /// upper bound of the syntax element.
    fn read_te(&mut self, x: u32) -> Result<u32>;

    /// Consume `rbsp_trailing_bits()` (§7.3.2.11): one `1` stop bit
    /// followed by zero-padding to the next byte boundary.
    fn read_rbsp_trailing_bits(&mut self) -> Result<()>;

    /// Return true iff more RBSP data remains beyond the current
    /// position — i.e. there is at least one `1` bit strictly before
    /// the final `1` (the RBSP stop bit). §7.2.
    fn more_rbsp_data(&mut self) -> Result<bool>;
}

pub trait BitWriterExt {
    fn write_flag(&mut self, b: bool);
    fn write_ue(&mut self, v: u32);
    fn write_se(&mut self, v: i32);
    fn write_te(&mut self, v: u32, x: u32);
    fn write_rbsp_trailing_bits(&mut self);
}

// ==================== Reader impl ====================

impl BitReaderExt for BitReader<'_> {
    fn read_flag(&mut self) -> Result<bool> {
        Ok(self.read_u1()? != 0)
    }

    fn peek_u32_lax(&mut self, n: u32) -> u32 {
        debug_assert!(n <= 32);
        match self.peek_u32(n) {
            Ok(v) => v,
            Err(_) => {
                // Fall back to whatever's available, zero-padded on the
                // low end. The shared reader exposes `bits_remaining()`
                // so we can compute the shortfall.
                let avail = self.bits_remaining().min(n as u64) as u32;
                if avail == 0 {
                    return 0;
                }
                // Peek the available prefix — guaranteed to succeed
                // since we just measured bits_remaining().
                let hi = self.peek_u32(avail).unwrap_or(0);
                hi << (n - avail)
            }
        }
    }

    fn read_ue(&mut self) -> Result<u32> {
        let mut leading_zeros = 0u32;
        loop {
            if leading_zeros > 32 {
                return Err(Error::invalid("h264 ue(v): >32 leading zeros"));
            }
            let bit = self.read_u1()?;
            if bit == 1 {
                break;
            }
            leading_zeros += 1;
        }
        if leading_zeros == 0 {
            return Ok(0);
        }
        let suffix = self.read_u32(leading_zeros)?;
        let base = (1u64 << leading_zeros) - 1;
        Ok((base + suffix as u64) as u32)
    }

    fn read_se(&mut self) -> Result<i32> {
        let code = self.read_ue()?;
        if code & 1 == 1 {
            Ok(code.div_ceil(2) as i32)
        } else {
            Ok(-((code / 2) as i32))
        }
    }

    fn read_te(&mut self, x: u32) -> Result<u32> {
        if x == 1 {
            Ok(1 - self.read_u1()?)
        } else {
            self.read_ue()
        }
    }

    fn read_rbsp_trailing_bits(&mut self) -> Result<()> {
        let one = self.read_u1()?;
        if one != 1 {
            return Err(Error::invalid("h264 rbsp_trailing_bits: missing stop bit"));
        }
        while !self.is_byte_aligned() {
            let z = self.read_u1()?;
            if z != 0 {
                return Err(Error::invalid(
                    "h264 rbsp_trailing_bits: non-zero alignment bit",
                ));
            }
        }
        Ok(())
    }

    fn more_rbsp_data(&mut self) -> Result<bool> {
        // The shared BitReader doesn't expose its backing slice, so the
        // "locate the last one bit in the input" check is re-expressed
        // as "peek increasing widths until we hit the stop bit's
        // position". Simpler: walk bit-by-bit from the current cursor
        // looking for a `1` strictly before the final one.
        //
        // H.264 RBSP has <= 1 trailing-bits byte after the last payload
        // byte, so the scan is bounded by bits_remaining().
        let mut probe = *self; // BitReader is Copy.
        let remaining = probe.bits_remaining();
        if remaining == 0 {
            return Ok(false);
        }
        // Find the position of the very last 1 bit (the RBSP stop bit)
        // by scanning backwards. We do this by consuming all remaining
        // bits into a buffer and locating the last set bit.
        let mut ones: Vec<u64> = Vec::with_capacity((remaining as usize).div_ceil(64));
        let mut left = remaining;
        while left > 0 {
            let take = left.min(32) as u32;
            let v = probe.read_u32(take)? as u64;
            ones.push(v << (64 - take));
            left -= take as u64;
        }
        // Find the bit index of the very last `1` across the `ones` runs.
        let mut last_one_bit: Option<u64> = None;
        let mut bit_off: u64 = 0;
        for &chunk in &ones {
            if chunk != 0 {
                // `chunk` is left-aligned in 64 bits with `take` valid
                // bits at the top. The last `1` is at position
                // `63 - chunk.trailing_zeros()` (counting from the
                // high bit).
                let within = 63 - chunk.trailing_zeros() as u64;
                last_one_bit = Some(bit_off + within);
            }
            bit_off += 32;
        }
        let Some(stop_bit_abs) = last_one_bit else {
            return Ok(false);
        };
        // If any `1` appears strictly before `stop_bit_abs` in the
        // unconsumed stream, payload remains. Re-scan:
        let mut probe2 = *self;
        let mut consumed: u64 = 0;
        while consumed < stop_bit_abs {
            let take = (stop_bit_abs - consumed).min(32) as u32;
            let chunk = probe2.read_u32(take)?;
            if chunk != 0 {
                return Ok(true);
            }
            consumed += take as u64;
        }
        Ok(false)
    }
}

// ==================== Writer impl ====================

impl BitWriterExt for BitWriter {
    fn write_flag(&mut self, b: bool) {
        self.write_bit(b);
    }

    fn write_ue(&mut self, v: u32) {
        if v == u32::MAX {
            panic!("bitwriter: ue(v) overflow (v == u32::MAX)");
        }
        let x = v + 1;
        let leading_zeros = 31 - x.leading_zeros();
        for _ in 0..leading_zeros {
            self.write_bit(false);
        }
        self.write_u32(x, leading_zeros + 1);
    }

    fn write_se(&mut self, v: i32) {
        let code = if v == 0 {
            0u32
        } else if v > 0 {
            (v as u32) * 2 - 1
        } else {
            (-v) as u32 * 2
        };
        self.write_ue(code);
    }

    fn write_te(&mut self, v: u32, x: u32) {
        if x == 1 {
            self.write_bit(v == 0);
        } else {
            self.write_ue(v);
        }
    }

    fn write_rbsp_trailing_bits(&mut self) {
        self.write_bit(true);
        while !self.is_byte_aligned() {
            self.write_bit(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn se_roundtrip() {
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
    fn rbsp_trailing_bits_aligns() {
        let mut w = BitWriter::new();
        w.write_u32(0b101, 3);
        w.write_rbsp_trailing_bits();
        let b = w.finish();
        // 101 then stop 1 then 4 zeros -> 1011_0000 = 0xB0.
        assert_eq!(b, vec![0xB0]);
    }
}
