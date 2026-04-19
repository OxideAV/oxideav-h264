//! CABAC **encoder** arithmetic engine — ITU-T H.264 (07/2019) §9.3.4.
//!
//! The encoder mirrors the decoder state: `codILow` + `codIRange`, plus
//! an "outstanding bits" counter that implements the §9.3.4.2 put-bit
//! with follow mechanism (classic arithmetic coder carry propagation).
//!
//! Symmetry with [`crate::cabac::engine::CabacDecoder`]:
//!
//! | Encode                         | Decode                             |
//! |--------------------------------|------------------------------------|
//! | [`CabacEncoder::encode_bin`]   | `decode_bin` (§9.3.3.2.1)          |
//! | [`CabacEncoder::encode_bypass`]| `decode_bypass` (§9.3.3.2.2)       |
//! | [`CabacEncoder::encode_terminate`] | `decode_terminate` (§9.3.3.2.4)|
//!
//! Both sides share the same `RANGE_TAB_LPS`, `TRANS_IDX_MPS`, and
//! `TRANS_IDX_LPS` ROM — imported from `crate::cabac::engine` so there's
//! no drift risk.

use crate::cabac::context::CabacContext;
use crate::cabac::engine::{RANGE_TAB_LPS, TRANS_IDX_LPS, TRANS_IDX_MPS};

/// Binary arithmetic encoder state (§9.3.4).
///
/// State invariants while encoding:
/// * `cod_i_range` is in `[256, 510]` after every renormalisation.
/// * `cod_i_low` holds the low end of the current arithmetic interval.
/// * `outstanding` counts undecided "follow" bits that will be emitted
///   when the next definite bit (0 or 1) is known.
/// * `first` is true before the very first put-bit; it suppresses that
///   initial bit (§9.3.4.2 "FirstBitFlag" trick — the decoder's 9-bit
///   init read compensates for the skipped lead bit).
pub struct CabacEncoder {
    cod_i_low: u32,
    cod_i_range: u32,
    outstanding: u32,
    first: bool,
    /// Emitted bits, MSB-first; packed into bytes by [`Self::finish`].
    bits: Vec<u8>,
}

impl Default for CabacEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl CabacEncoder {
    /// Fresh encoder per §9.3.1.2: `codIRange = 0x01FE`, `codILow = 0`.
    pub fn new() -> Self {
        Self {
            cod_i_low: 0,
            cod_i_range: 0x01FE,
            outstanding: 0,
            first: true,
            bits: Vec::new(),
        }
    }

    /// Number of bits written so far (pending byte alignment). Useful for
    /// tests that want to compare two encodes symbol-by-symbol.
    pub fn bit_len(&self) -> usize {
        self.bits.len()
    }

    /// §9.3.3.2.1 / §9.3.4.4 — encode one regular-mode bin against `ctx`.
    /// Updates `ctx.p_state_idx` and (when `ctx.p_state_idx == 0` and an
    /// LPS is emitted) `ctx.val_mps`, exactly like the decoder.
    pub fn encode_bin(&mut self, ctx: &mut CabacContext, bin: u8) {
        let rlps_idx = ((self.cod_i_range >> 6) & 3) as usize;
        let p = ctx.p_state_idx as usize;
        let rlps = RANGE_TAB_LPS[p][rlps_idx] as u32;
        self.cod_i_range -= rlps;
        if bin != ctx.val_mps {
            // LPS path — low advances by the MPS range, range shrinks to rLPS.
            self.cod_i_low += self.cod_i_range;
            self.cod_i_range = rlps;
            if ctx.p_state_idx == 0 {
                ctx.val_mps = 1 - ctx.val_mps;
            }
            ctx.p_state_idx = TRANS_IDX_LPS[p];
        } else {
            ctx.p_state_idx = TRANS_IDX_MPS[p];
        }
        self.renormalize();
    }

    /// §9.3.4.2 — bypass (equiprobable) encode. The bypass path doubles
    /// `cod_i_low` and conditionally adds `cod_i_range` instead of going
    /// through the `rangeTabLPS` lookup.
    pub fn encode_bypass(&mut self, bin: u8) {
        self.cod_i_low <<= 1;
        if bin != 0 {
            self.cod_i_low += self.cod_i_range;
        }
        // After the doubling we need a single put-bit-or-stash cycle rather
        // than a full renorm loop (bypass always emits exactly one bit of
        // precision per call; §9.3.4.2 "EncodeBypass" in the spec).
        if self.cod_i_low >= 0x0400 {
            self.cod_i_low -= 0x0400;
            self.put_bit(1);
        } else if self.cod_i_low < 0x0200 {
            self.put_bit(0);
        } else {
            self.cod_i_low -= 0x0200;
            self.outstanding += 1;
        }
    }

    /// §9.3.4.5 — end-of-slice terminate. Passing `bin = 0` means "keep
    /// going" (the usual per-MB terminate bit); `bin = 1` means "this was
    /// the last macroblock" and the encoder should be followed by
    /// [`Self::finish`].
    ///
    /// `bin = 1` matches the decoder's `decode_terminate` = 1 return
    /// without renormalising — the spec's §9.3.4.5 mirror. On `bin = 0`
    /// we renormalise so the decoder's non-terminate branch stays aligned.
    pub fn encode_terminate(&mut self, bin: u8) {
        if bin != 0 {
            // §9.3.4.5: subtract 2 from codIRange, then set codIRange = 2
            // and codILow += (codIRange - 2) so the decoder's "offset >=
            // codIRange - 2" test succeeds. No renorm in the terminate
            // path (§9.3.4.5 says the next data bit is written straight
            // into the stream after encoding the terminate).
            self.cod_i_low += self.cod_i_range - 2;
            self.cod_i_range = 2;
            self.flush_on_terminate();
        } else {
            self.cod_i_range -= 2;
            self.renormalize();
        }
    }

    /// §9.3.4.5 / §9.3.4.6 terminate flush — after a terminate(1), write
    /// out `codILow`'s remaining precision + stop bit so the byte-aligned
    /// stream's tail decodes cleanly. Deliberately oversamples (10 bits
    /// rather than the minimal 3) to absorb any trailing read-ahead the
    /// arithmetic decoder performs during its 9-bit init / renorm loop
    /// on the *next* slice if anything consumes this stream further.
    /// Mirrors the proven recipe from `cabac::residual::tests::CabacEncoder`.
    fn flush_on_terminate(&mut self) {
        for _ in 0..10 {
            let b = ((self.cod_i_low >> 9) & 1) as u8;
            self.put_bit(b);
            self.cod_i_low = (self.cod_i_low << 1) & 0x3FF;
        }
        // §7.3.2.11 slice trailer: append a '1' stop bit then align to the
        // next byte with zeros.
        self.bits.push(1);
    }

    /// §9.3.4.4 renormalisation — while `cod_i_range < 256` emit one bit
    /// (or stash an outstanding-follow bit) and double both state variables.
    fn renormalize(&mut self) {
        while self.cod_i_range < 0x0100 {
            if self.cod_i_low < 0x0100 {
                self.put_bit(0);
            } else if self.cod_i_low >= 0x0200 {
                self.cod_i_low -= 0x0200;
                self.put_bit(1);
            } else {
                self.cod_i_low -= 0x0100;
                self.outstanding += 1;
            }
            self.cod_i_low <<= 1;
            self.cod_i_range <<= 1;
        }
    }

    /// §9.3.4.2 put-bit: write `bit`, then `outstanding` copies of
    /// `1 - bit` (the deferred-follow bits). The very first invocation
    /// writes no lead bit per §9.3.4.3 `FirstBitFlag = 1`, since the
    /// decoder's 9-bit init read spans the same ambiguity window.
    fn put_bit(&mut self, bit: u8) {
        if self.first {
            self.first = false;
        } else {
            self.bits.push(bit);
        }
        for _ in 0..self.outstanding {
            self.bits.push(1 - bit);
        }
        self.outstanding = 0;
    }

    /// Take the emitted bits and pack them MSB-first into a byte vector.
    /// Callers are responsible for ensuring [`Self::encode_terminate`]
    /// has been called first with `bin = 1` so the final interval is
    /// flushed.
    ///
    /// Pads the trailing partial byte with zero bits (§7.3.2.11) and
    /// appends 32 extra zero bytes so any over-eager read-ahead by the
    /// decoder during its 9-bit init and renorm loop doesn't run off the
    /// end of the buffer — the arithmetic decoder is guaranteed to stop
    /// reading real data before those padding bytes (otherwise the
    /// `decode_terminate` bin would never have returned 1 in the first
    /// place).
    pub fn finish(mut self) -> Vec<u8> {
        while self.bits.len() % 8 != 0 {
            self.bits.push(0);
        }
        for _ in 0..32 {
            self.bits.push(0);
        }
        let mut out = Vec::with_capacity(self.bits.len() / 8);
        for chunk in self.bits.chunks(8) {
            let mut b = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                b |= (bit & 1) << (7 - i);
            }
            out.push(b);
        }
        out
    }

    /// Like [`Self::finish`] but returns bits whose length is rounded up
    /// to the next byte boundary **without** the 32-byte trailing slack.
    /// Used by the encoder-proper to build a real RBSP: the NAL layer
    /// caps the size and any read-past-end must be impossible by
    /// construction (the decoder never looks past `bit_pos` after
    /// `decode_terminate = 1`). Tests consume `finish()` which adds slack.
    pub fn finish_rbsp(mut self) -> Vec<u8> {
        while self.bits.len() % 8 != 0 {
            self.bits.push(0);
        }
        let mut out = Vec::with_capacity(self.bits.len() / 8);
        for chunk in self.bits.chunks(8) {
            let mut b = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                b |= (bit & 1) << (7 - i);
            }
            out.push(b);
        }
        out
    }
}

// ==========================================================================
// Tests — arithmetic round-trip against the decoder.
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::engine::CabacDecoder;

    #[test]
    fn regular_bin_roundtrip_random_sequence() {
        // Encode a fixed pseudo-random sequence of 256 bins against a
        // single context, then decode and compare. This is the core
        // correctness property of the arithmetic coder.
        let seq: Vec<u8> = (0..256u32)
            .map(|i| {
                let v = i.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((v >> 16) & 1) as u8
            })
            .collect();

        let mut enc = CabacEncoder::new();
        let mut enc_ctx = CabacContext::default();
        for &b in &seq {
            enc.encode_bin(&mut enc_ctx, b);
        }
        enc.encode_terminate(1);
        let bytes = enc.finish();

        let mut dec = CabacDecoder::new(&bytes, 0).unwrap();
        let mut dec_ctx = CabacContext::default();
        for (i, &expected) in seq.iter().enumerate() {
            let got = dec.decode_bin(&mut dec_ctx).unwrap();
            assert_eq!(got, expected, "bin {i} mismatch");
        }
        // Final terminate should read 1.
        assert_eq!(dec.decode_terminate().unwrap(), 1);
    }

    #[test]
    fn bypass_bin_roundtrip() {
        let seq: &[u8] = &[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0];
        let mut enc = CabacEncoder::new();
        for &b in seq {
            enc.encode_bypass(b);
        }
        enc.encode_terminate(1);
        let bytes = enc.finish();
        let mut dec = CabacDecoder::new(&bytes, 0).unwrap();
        for (i, &expected) in seq.iter().enumerate() {
            let got = dec.decode_bypass().unwrap();
            assert_eq!(got, expected, "bypass bin {i} mismatch");
        }
    }

    #[test]
    fn mixed_regular_and_bypass_roundtrip() {
        // Interleave regular + bypass + terminate(0) intermediates, just
        // like a real residual block encode. The decoder must replay the
        // same sequence without drift.
        let mut enc = CabacEncoder::new();
        let mut ctxs = [CabacContext::default(); 8];

        // Sequence: binarise "coded_block_flag=1 / sig map with one coeff /
        // level value with bypass sign" using ad-hoc context picks.
        let plan: &[(&str, u8)] = &[
            ("reg", 1),    // coded_block_flag = 1
            ("reg", 0),    // sig[0] = 0
            ("reg", 1),    // sig[1] = 1
            ("reg", 1),    // last[1] = 1
            ("reg", 1),    // level prefix bin0 = 1 (value >= 1)
            ("reg", 0),    // level prefix bin1 = 0 (value = 1 exactly, i.e. abs_level-1 = 0)
            ("bypass", 0), // sign = 0 (positive)
            ("term", 0),   // non-terminate between MBs
            ("bypass", 1),
            ("bypass", 0),
            ("reg", 0),
            ("reg", 1),
        ];
        for &(kind, v) in plan {
            match kind {
                "reg" => enc.encode_bin(&mut ctxs[0], v),
                "bypass" => enc.encode_bypass(v),
                "term" => enc.encode_terminate(v),
                _ => unreachable!(),
            }
        }
        enc.encode_terminate(1);
        let bytes = enc.finish();

        let mut dec = CabacDecoder::new(&bytes, 0).unwrap();
        let mut dec_ctxs = [CabacContext::default(); 8];
        for &(kind, v) in plan {
            let got = match kind {
                "reg" => dec.decode_bin(&mut dec_ctxs[0]).unwrap(),
                "bypass" => dec.decode_bypass().unwrap(),
                "term" => dec.decode_terminate().unwrap(),
                _ => unreachable!(),
            };
            assert_eq!(got, v, "mismatch for kind={kind}");
        }
        assert_eq!(dec.decode_terminate().unwrap(), 1);
    }
}
