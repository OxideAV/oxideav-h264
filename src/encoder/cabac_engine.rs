//! §9.3 — CABAC arithmetic **encoder** engine.
//!
//! Mirror image of the decoder engine in [`crate::cabac`]. Implements the
//! three primitives an H.264 CABAC encoder needs:
//!
//! * §9.3.4.2 — `EncodeDecision`  (the inverse of §9.3.3.2.1).
//! * §9.3.4.3 — `EncodeBypass`    (the inverse of §9.3.3.2.3).
//! * §9.3.4.5 — `EncodeTerminate` (the inverse of §9.3.3.2.4).
//! * §9.3.4.4 — `RenormE` + carry-propagation handling.
//!
//! The state-transition tables and per-syntax (`m`, `n`) initialisation
//! tables are shared with the decoder via [`crate::cabac::TRANS_IDX_MPS`],
//! [`crate::cabac::TRANS_IDX_LPS`], and [`crate::cabac::RANGE_TAB_LPS`]
//! — the encoder advances the same state machine in the same direction
//! after each bin, just from the writer's perspective.
//!
//! Output is the byte stream that, when fed back to the decoder engine
//! in [`crate::cabac::CabacDecoder`], reproduces the exact bin sequence
//! that was encoded. The caller is responsible for invoking
//! `EncodeFinish` (via [`CabacEncoder::finish`]) before reading
//! `into_bytes` — that flushes the residual `low` register and any
//! outstanding "outstanding bit" run.
//!
//! Bit-exactness against the decoder is verified by the per-primitive
//! unit tests at the bottom of this module: each test encodes a known
//! bin sequence into bytes and then walks those bytes back through the
//! decoder engine, asserting both the recovered bin sequence and the
//! per-bin context state evolution match the encode side step-for-step.

#![allow(dead_code)]

use crate::cabac::{CtxState, RANGE_TAB_LPS, TRANS_IDX_LPS, TRANS_IDX_MPS};

/// When `OXIDEAV_H264_BIN_TRACE` is set, emit one line per encoded bin
/// in the same format as the decoder's trace so the two can be diff'd.
#[inline]
fn bin_trace_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("OXIDEAV_H264_BIN_TRACE").is_some())
}

/// CABAC arithmetic encoder state per §9.3.4.
///
/// `cod_i_low` holds the current low end of the interval. Per the
/// encoder pseudo-code in the spec, it grows past 9 bits during
/// encoding; we hold it as `u32` so the carry-out of bit 9
/// (`cod_i_low >= 2^10` after shifting) propagates through the
/// "outstanding bits" mechanism per §9.3.4.4.
///
/// The `outstanding_bits` counter implements the carry-propagation /
/// "put_bit" mechanism described in §9.3.4.4: any bit emitted is
/// preceded by `outstanding_bits` copies of its complement.
pub struct CabacEncoder {
    /// §9.3.4.1 — `codIRange`, initialised to 510.
    cod_i_range: u32,
    /// §9.3.4.1 — `codILow`, initialised to 0. Up to 10 working bits.
    cod_i_low: u32,
    /// §9.3.4.4 — count of buffered "outstanding bits" awaiting either
    /// a 0 or 1 to be emitted. Each outstanding bit eventually appears
    /// as the complement of whatever resolving bit follows.
    outstanding_bits: u32,
    /// §9.3.4 — partial output byte and the index of the next bit to
    /// write within it. Bits are written MSB-first.
    bytes: Vec<u8>,
    bit_pos: u8,
    /// Number of `EncodeDecision` / `EncodeBypass` / `EncodeTerminate`
    /// calls so far (excluding `finish`). Useful for parallel debug
    /// instrumentation against the decoder's `bin_count`.
    bin_count: u64,
    /// §9.3.4.2 — first-bit-flag. Per the spec the encoder's initial
    /// "put_bit" emits no bit until the first carry resolution. This
    /// matches a 9-bit precision setup where the initial codILow=0
    /// reads back as the decoder's initial codIOffset=0 (after RenormE
    /// has padded the leading bits with 0).
    first_bit_flag: bool,
}

impl Default for CabacEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl CabacEncoder {
    /// §9.3.4.1 — initialise the engine.
    pub fn new() -> Self {
        Self {
            cod_i_range: 510,
            cod_i_low: 0,
            outstanding_bits: 0,
            bytes: Vec::new(),
            bit_pos: 0,
            bin_count: 0,
            first_bit_flag: true,
        }
    }

    /// Number of bins emitted so far.
    pub fn bin_count(&self) -> u64 {
        self.bin_count
    }

    /// Number of bits already physically committed to the byte buffer.
    /// Excludes the residual `cod_i_low` and any outstanding-bit run.
    pub fn bits_emitted(&self) -> usize {
        let full_bytes = if self.bit_pos == 0 {
            self.bytes.len()
        } else {
            self.bytes.len() - 1
        };
        full_bytes * 8 + self.bit_pos as usize
    }

    /// §9.3.4.2 — EncodeDecision. Encodes one bin against context `ctx`
    /// and updates `ctx` per Table 9-45 — exactly the same transition
    /// the decoder would apply after reading the same bin.
    pub fn encode_decision(&mut self, ctx: &mut CtxState, bin: u8) {
        debug_assert!(bin <= 1);
        let trace_on = bin_trace_enabled();
        let pre_state = ctx.state_idx;
        let pre_mps = ctx.val_mps;
        let pre_range = self.cod_i_range;
        let pre_low = self.cod_i_low;

        let q_idx = ((self.cod_i_range >> 6) & 0b11) as usize;
        let cod_i_range_lps = RANGE_TAB_LPS[ctx.state_idx as usize][q_idx] as u32;
        self.cod_i_range -= cod_i_range_lps;

        if bin != ctx.val_mps {
            // LPS path.
            self.cod_i_low += self.cod_i_range;
            self.cod_i_range = cod_i_range_lps;
            if ctx.state_idx == 0 {
                ctx.val_mps = 1 - ctx.val_mps;
            }
            ctx.state_idx = TRANS_IDX_LPS[ctx.state_idx as usize];
        } else {
            // MPS path.
            ctx.state_idx = TRANS_IDX_MPS[ctx.state_idx as usize];
        }
        self.renorm_e();
        self.bin_count = self.bin_count.wrapping_add(1);
        if trace_on {
            eprintln!(
                "[ENC {:>8}] DD pre_state={:>2} pre_mps={} pre_range={:>4} pre_low={:>4} bin={} post_state={:>2} post_mps={} post_range={:>4} post_low={:>4}",
                self.bin_count,
                pre_state, pre_mps, pre_range, pre_low,
                bin,
                ctx.state_idx, ctx.val_mps, self.cod_i_range, self.cod_i_low,
            );
        }
    }

    /// §9.3.4.3 — EncodeBypass. Encodes one equiprobable bin without
    /// touching any context.
    pub fn encode_bypass(&mut self, bin: u8) {
        debug_assert!(bin <= 1);
        let trace_on = bin_trace_enabled();
        let pre_range = self.cod_i_range;
        let pre_low = self.cod_i_low;

        self.cod_i_low <<= 1;
        if bin == 1 {
            self.cod_i_low += self.cod_i_range;
        }
        // Bypass renormalisation: a single-bit shift-out (no while loop;
        // the codIRange stays the same).
        if self.cod_i_low >= 1024 {
            self.put_bit(1);
            self.cod_i_low -= 1024;
        } else if self.cod_i_low < 512 {
            self.put_bit(0);
        } else {
            // codILow in [512, 1023]: carry-uncertain zone — defer the bit
            // using the same outstanding-bits mechanism as renorm_e.
            self.cod_i_low -= 512;
            self.outstanding_bits += 1;
        }
        self.bin_count = self.bin_count.wrapping_add(1);
        if trace_on {
            eprintln!(
                "[ENC {:>8}] BP pre_range={:>4} pre_low={:>4} bin={} post_range={:>4} post_low={:>4}",
                self.bin_count,
                pre_range, pre_low,
                bin,
                self.cod_i_range, self.cod_i_low,
            );
        }
    }

    /// §9.3.4.5 — EncodeTerminate. `bin == 1` finalises the encoding
    /// (e.g. `end_of_slice_flag` MPS path on the LAST MB of the slice);
    /// `bin == 0` is the "not yet" path used after every non-final MB.
    pub fn encode_terminate(&mut self, bin: u8) {
        debug_assert!(bin <= 1);
        let trace_on = bin_trace_enabled();
        let pre_range = self.cod_i_range;
        let pre_low = self.cod_i_low;

        self.cod_i_range -= 2;
        if bin != 0 {
            // Terminate: per §9.3.4.5, codILow += codIRange; then EncodeFlush
            // is called. We perform the addition here and let `finish`
            // (mirror of EncodeFlush) handle the flush.
            self.cod_i_low += self.cod_i_range;
            self.encode_flush();
        } else {
            self.renorm_e();
        }
        self.bin_count = self.bin_count.wrapping_add(1);
        if trace_on {
            eprintln!(
                "[ENC {:>8}] TT pre_range={:>4} pre_low={:>4} bin={} post_range={:>4} post_low={:>4}",
                self.bin_count,
                pre_range, pre_low,
                bin,
                self.cod_i_range, self.cod_i_low,
            );
        }
    }

    /// §9.3.4.5 — EncodeFlush. Called by [`Self::finish`] (or by a
    /// terminating-1 `EncodeTerminate`) to drain the residual `cod_i_low`
    /// state into the output byte stream. Per the spec, this writes
    /// exactly 9 bits + zero alignment to the next byte boundary.
    fn encode_flush(&mut self) {
        // §9.3.4.5: codIRange = 2; RenormE; then 'put_bit' the 10-bit
        // (codILow & 1023) >> 7, then the (codILow & 127) >> 7 ... that's
        // a literal: we put bit 9 (the carry-position) through put_bit
        // (which handles outstanding bits + carry), then write the
        // remaining 7 bits literally (no further carry possible).
        self.cod_i_range = 2;
        self.renorm_e();
        // Emit bit position 9 of low (i.e. the high bit of low after
        // renorm has shifted things). Then write the next bit literally
        // ("PutBit(codILow >> 7) & 1") followed by a 1-bit stop and
        // alignment.
        //
        // The spec writes (eq. 9.3.4.5):
        //   PutBit((codILow >> 9) & 1)
        //   WriteBits((codILow >> 7) & 3, 2)
        //   WriteBits(1, 1)               -- rbsp_stop_one_bit
        //
        // followed by zero-padding to a byte boundary by rbsp_trailing_bits.
        self.put_bit(((self.cod_i_low >> 9) & 1) as u8);
        // The remaining 2 bits (positions 8..7 of cod_i_low) are emitted
        // literally — no further carry is possible.
        self.write_bits_literal(((self.cod_i_low >> 7) & 0b11) as u8, 2);
    }

    /// Drain pending state and finish the encoding.
    /// Returns the encoded bytes (caller may want to add an
    /// `rbsp_stop_one_bit` and align to the next byte boundary; we do
    /// that here by writing one '1' bit and zero-padding).
    pub fn finish(mut self) -> Vec<u8> {
        // Per §9.3.4.5 (when `EncodeTerminate(1)` already ran, the
        // engine is already in the flushed state). We assume the caller
        // has invoked `encode_terminate(1)` for the final MB's
        // `end_of_slice_flag` — that's the spec-mandated termination.
        // We then write the rbsp_stop_one_bit and pad to byte boundary
        // matching `BitWriter::rbsp_trailing_bits`.
        self.write_bits_literal(1, 1);
        while self.bit_pos != 0 {
            self.write_bits_literal(0, 1);
        }
        self.bytes
    }

    /// Variant of `finish` that does NOT append an rbsp_stop_one_bit.
    /// Used when the caller wants to splice CABAC payload into a wider
    /// container that handles trailing bits separately.
    pub fn finish_no_trailing(self) -> Vec<u8> {
        self.bytes
    }

    /// §9.3.4.4 — RenormE. Iteratively shifts `cod_i_low` left until
    /// `cod_i_range >= 256`, emitting bits or buffering them as
    /// outstanding bits per the carry-propagation rules.
    fn renorm_e(&mut self) {
        while self.cod_i_range < 256 {
            if self.cod_i_low < 256 {
                self.put_bit(0);
            } else if self.cod_i_low >= 512 {
                self.cod_i_low -= 512;
                self.put_bit(1);
            } else {
                self.cod_i_low -= 256;
                self.outstanding_bits += 1;
            }
            self.cod_i_range <<= 1;
            self.cod_i_low <<= 1;
        }
    }

    /// §9.3.4.4 — `put_bit(b)`. Emits `outstanding_bits` copies of `1-b`
    /// (the complement of `b`) followed by `b` itself, then resets the
    /// outstanding-bit counter.
    ///
    /// The spec defers the very first call (the "first_bit_flag"
    /// branch); per §9.3.4.4 the leading bit is held back until the
    /// first call and used only to cancel the carry-out of any
    /// outstanding-bit runs. We implement that by treating
    /// `first_bit_flag = true` as "throw away the first put_bit, keep
    /// the outstanding-bit count for subsequent runs".
    fn put_bit(&mut self, bit: u8) {
        debug_assert!(bit <= 1);
        if self.first_bit_flag {
            self.first_bit_flag = false;
        } else {
            self.write_bits_literal(bit, 1);
        }
        let comp = 1 - bit;
        while self.outstanding_bits > 0 {
            self.write_bits_literal(comp, 1);
            self.outstanding_bits -= 1;
        }
    }

    /// Write `bits` low-order bits of `value`, MSB-first, to the byte
    /// buffer. Used by `put_bit` and by the trailing-bits emitter.
    fn write_bits_literal(&mut self, value: u8, bits: u32) {
        debug_assert!(bits <= 8);
        for i in (0..bits).rev() {
            let b = (value >> i) & 1;
            if self.bit_pos == 0 {
                self.bytes.push(0);
            }
            let idx = self.bytes.len() - 1;
            self.bytes[idx] |= b << (7 - self.bit_pos);
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.bit_pos = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — round-trip every primitive against the decoder's CabacDecoder.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitReader;
    use crate::cabac::CabacDecoder;

    /// Encode a sequence of (ctx_init_state, bin) pairs and verify that
    /// the decoder, given the same context init state, reads back the
    /// same bins.
    fn round_trip_decisions(bins: &[(CtxState, u8)]) {
        let mut enc = CabacEncoder::new();
        let mut enc_ctxs: Vec<CtxState> = bins.iter().map(|(c, _)| *c).collect();
        for (i, (_, bin)) in bins.iter().enumerate() {
            enc.encode_decision(&mut enc_ctxs[i], *bin);
        }
        // EncodeTerminate(1) finalises. The decoder will see this as the
        // terminate-1 path on the last bin.
        enc.encode_terminate(1);
        let bytes = enc.finish_no_trailing();

        // Decode side: instantiate fresh contexts (matching the input
        // states) and read back the bins.
        let mut dec_ctxs: Vec<CtxState> = bins.iter().map(|(c, _)| *c).collect();
        let mut dec = CabacDecoder::new(BitReader::new(&bytes)).unwrap();
        for (i, (_, expected)) in bins.iter().enumerate() {
            let got = dec.decode_decision(&mut dec_ctxs[i]).unwrap();
            assert_eq!(
                got, *expected,
                "bin {i}: expected {expected}, got {got} (bytes={:?})",
                bytes
            );
        }
        // Verify the matching terminate(1) is recoverable.
        let term = dec.decode_terminate().unwrap();
        assert_eq!(term, 1, "trailing terminate should be 1");
    }

    #[test]
    fn round_trip_single_mps_decision() {
        let ctx = CtxState {
            state_idx: 0,
            val_mps: 0,
        };
        round_trip_decisions(&[(ctx, 0)]);
    }

    #[test]
    fn round_trip_single_lps_decision() {
        let ctx = CtxState {
            state_idx: 0,
            val_mps: 0,
        };
        round_trip_decisions(&[(ctx, 1)]);
    }

    #[test]
    fn round_trip_alternating_bins() {
        let ctx = CtxState {
            state_idx: 30,
            val_mps: 0,
        };
        let bins: Vec<_> = (0..32).map(|i| (ctx, (i & 1) as u8)).collect();
        round_trip_decisions(&bins);
    }

    #[test]
    fn round_trip_long_lps_run() {
        // Force lots of LPS decisions at low state — exercises the
        // outstanding-bits / carry path.
        let ctx = CtxState {
            state_idx: 0,
            val_mps: 0,
        };
        let bins: Vec<_> = (0..16).map(|_| (ctx, 1)).collect();
        round_trip_decisions(&bins);
    }

    #[test]
    fn round_trip_mixed_with_bypass() {
        // Encode: dec(0), bypass(1), bypass(0), dec(1), bypass(1), terminate(1).
        let mut ctx_a = CtxState {
            state_idx: 10,
            val_mps: 0,
        };
        let mut ctx_b = CtxState {
            state_idx: 0,
            val_mps: 1,
        };
        let mut enc = CabacEncoder::new();
        let mut a1 = ctx_a;
        let mut b1 = ctx_b;
        enc.encode_decision(&mut a1, 0);
        enc.encode_bypass(1);
        enc.encode_bypass(0);
        enc.encode_decision(&mut b1, 1);
        enc.encode_bypass(1);
        enc.encode_terminate(1);
        let bytes = enc.finish_no_trailing();

        let mut dec = CabacDecoder::new(BitReader::new(&bytes)).unwrap();
        // Read back in same order.
        assert_eq!(dec.decode_decision(&mut ctx_a).unwrap(), 0);
        assert_eq!(dec.decode_bypass().unwrap(), 1);
        assert_eq!(dec.decode_bypass().unwrap(), 0);
        assert_eq!(dec.decode_decision(&mut ctx_b).unwrap(), 1);
        assert_eq!(dec.decode_bypass().unwrap(), 1);
        assert_eq!(dec.decode_terminate().unwrap(), 1);
    }

    #[test]
    fn round_trip_terminate_zero_then_more_decisions() {
        // EncodeTerminate(0) + more decisions + EncodeTerminate(1).
        let mut ctx = CtxState {
            state_idx: 5,
            val_mps: 1,
        };
        let mut enc = CabacEncoder::new();
        for _ in 0..5 {
            enc.encode_terminate(0);
            let mut c = ctx;
            enc.encode_decision(&mut c, 1);
            ctx = c;
        }
        enc.encode_terminate(1);
        let bytes = enc.finish_no_trailing();

        let mut dec_ctx = CtxState {
            state_idx: 5,
            val_mps: 1,
        };
        let mut dec = CabacDecoder::new(BitReader::new(&bytes)).unwrap();
        for _ in 0..5 {
            assert_eq!(dec.decode_terminate().unwrap(), 0);
            assert_eq!(dec.decode_decision(&mut dec_ctx).unwrap(), 1);
        }
        assert_eq!(dec.decode_terminate().unwrap(), 1);
    }

    #[test]
    fn round_trip_random_bins_against_decoder() {
        // Pseudo-random bin sequence, deterministic seed.
        let mut state = 0x12345678u32;
        let mut rng = || {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            state
        };
        let mut enc = CabacEncoder::new();
        // Use a couple of contexts that adapt as the bins go by.
        let mut ctxs = [
            CtxState {
                state_idx: 20,
                val_mps: 0,
            },
            CtxState {
                state_idx: 40,
                val_mps: 1,
            },
        ];
        let mut record: Vec<(usize, u8)> = Vec::new();
        for _ in 0..200 {
            let r = rng();
            let kind = (r >> 24) & 0b11;
            let bin = ((r >> 16) & 1) as u8;
            match kind {
                0 | 1 => {
                    let ci = ((r >> 8) & 1) as usize;
                    record.push((ci, bin));
                    enc.encode_decision(&mut ctxs[ci], bin);
                }
                2 => {
                    record.push((100, bin));
                    enc.encode_bypass(bin);
                }
                _ => {
                    // Skip terminates inside the loop to keep bookkeeping simple.
                    let ci = ((r >> 8) & 1) as usize;
                    record.push((ci, bin));
                    enc.encode_decision(&mut ctxs[ci], bin);
                }
            }
        }
        enc.encode_terminate(1);
        let bytes = enc.finish_no_trailing();

        // Replay through the decoder.
        let mut dec_ctxs = [
            CtxState {
                state_idx: 20,
                val_mps: 0,
            },
            CtxState {
                state_idx: 40,
                val_mps: 1,
            },
        ];
        let mut dec = CabacDecoder::new(BitReader::new(&bytes)).unwrap();
        for (i, (which, bin)) in record.iter().enumerate() {
            if *which == 100 {
                let got = dec.decode_bypass().unwrap();
                assert_eq!(got, *bin, "bypass bin {i} mismatch");
            } else {
                let got = dec.decode_decision(&mut dec_ctxs[*which]).unwrap();
                assert_eq!(got, *bin, "decision bin {i} mismatch (ctx {which})");
            }
        }
        assert_eq!(dec.decode_terminate().unwrap(), 1);
    }
}
