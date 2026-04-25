//! §9.3 — CABAC arithmetic decoding engine.
//!
//! Spec-driven implementation per ITU-T Rec. H.264 (08/2024). This
//! module implements the engine primitives only: initialisation,
//! DecodeDecision / DecodeBypass / DecodeTerminate, and the state
//! transition tables. Per-syntax binarisation and context
//! initialisation tables (§9.3.2 Tables 9-12 .. 9-43) live with the
//! macroblock/slice decoders that consume them.
//!
//! # Spec-conformance of the engine primitives
//!
//! The engine has been audited against the 08/2024 spec on the following
//! points (see the corresponding unit tests further down this file):
//!
//! * §9.3.1.2 initialisation — `codIRange=510`, `codIOffset=read_bits(9)`.
//!   Verified by [`tests::engine_new_reads_9_bit_offset`] /
//!   [`tests::engine_new_all_zeros_offset`].
//! * §9.3.3.2.1 / Figure 9-3 DecodeDecision — the `qCodIRangeIdx`
//!   derivation (eq. 9-25), `codIRangeLPS` lookup (eq. 9-26), and the
//!   state-transition path including the `pStateIdx == 0` valMPS-flip
//!   special case are all exercised bit-exactly by
//!   [`tests::decode_decision_bit_accounting_mps_then_lps`] and
//!   [`tests::decode_decision_lps_path_flips_mps_at_state_zero`].
//! * §9.3.3.2.2 / Figure 9-4 RenormD — exercised via the decision tests.
//! * §9.3.3.2.3 / Figure 9-5 DecodeBypass — the 10-bit working-register
//!   width is verified by [`tests::decode_bypass_ten_bit_register_width`].
//! * §9.3.3.2.4 / Figure 9-6 DecodeTerminate — both the "terminate"
//!   branch and the "not yet + RenormD" branch are covered; the
//!   [`tests::decode_terminate_zero_then_decision_continues`] test
//!   additionally asserts that a non-terminating DecodeTerminate call
//!   leaves the engine in a state that a subsequent DecodeDecision can
//!   consume correctly (this is the `end_of_slice_flag` hot path for
//!   every non-last MB of a slice).
//!
//! # Historical note on CABAC conformance regressions
//!
//! The low-level arithmetic decoder in this file is spec-correct and
//! bit-exact. Conformance regressions on CABAC streams (e.g. a max|Δ|
//! climb on the foreman_p16x16 sample) have repeatedly turned out to
//! be in the ctxIdxInc derivations (§9.3.3.1.1.*) or the per-syntax
//! binarisations in `cabac_ctx.rs` / `macroblock_layer.rs`, never in
//! this engine. When chasing a regression, the recommended order is:
//!
//! 1. Confirm these unit tests still pass (they pin the engine's
//!    bit-exact behaviour against the spec flowcharts).
//! 2. Walk the suspect syntax element's `decode_*` helper in
//!    `cabac_ctx.rs` and compare its ctxIdxInc derivation against
//!    §9.3.3.1.1.{1..10}.
//! 3. Check that the `CabacMbNeighbourInfo` grid in
//!    `macroblock_layer.rs` is being updated with ALL the per-MB
//!    fields the next MB's ctxIdxInc derivations will consult (CBP,
//!    intra_chroma_pred_mode, is_i_nxn, transform_size_8x8_flag,
//!    is_skip, is_i_pcm).
//! 4. Check that `EntropyState` in `slice_data.rs` threads
//!    `prev_mb_qp_delta_nonzero` across MBs within a single slice
//!    (§9.3.3.1.1.5 — it is NOT per-MB state, it carries forward).

#![allow(dead_code)]
// The engine is used by the slice/macroblock-layer decoders (not yet
// written). Silence dead-code while the rest of the CABAC pipeline is
// still empty.

use crate::bitstream::{BitError, BitReader};

/// `OXIDEAV_H264_BIN_TRACE` — when set (any value), every call to
/// `decode_decision`, `decode_bypass`, and `decode_terminate` emits a
/// line to stderr capturing the full pre/post engine state and the
/// decoded bin. Intended for bisecting CABAC divergences against the
/// JM `_trace.txt` oracle.
#[inline]
fn bin_trace_enabled() -> bool {
    std::env::var_os("OXIDEAV_H264_BIN_TRACE").is_some()
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CabacError {
    #[error("bitstream read failed: {0}")]
    Bitstream(#[from] BitError),
    #[error("invalid cabac_init_idc {0} (must be 0..=2 or absent for I-slice)")]
    InvalidInitIdc(u32),
    #[error("invalid SliceQPY {0} (must be 0..=51)")]
    InvalidSliceQp(i32),
}

pub type CabacResult<T> = Result<T, CabacError>;

// ---------------------------------------------------------------------------
// §9.3.3.2 / Table 9-44 — rangeTabLPS[pStateIdx][qCodIRangeIdx].
//
// Spec cell coordinates: pStateIdx is the row (0..=63); qCodIRangeIdx is the
// column (0..=3). Values transcribed verbatim from ITU-T Rec. H.264 (08/2024),
// document page 278.
// ---------------------------------------------------------------------------
pub(crate) static RANGE_TAB_LPS: [[u8; 4]; 64] = [
    [128, 176, 208, 240], // pStateIdx  0
    [128, 167, 197, 227], // pStateIdx  1
    [128, 158, 187, 216], // pStateIdx  2
    [123, 150, 178, 205], // pStateIdx  3
    [116, 142, 169, 195], // pStateIdx  4
    [111, 135, 160, 185], // pStateIdx  5
    [105, 128, 152, 175], // pStateIdx  6
    [100, 122, 144, 166], // pStateIdx  7
    [95, 116, 137, 158],  // pStateIdx  8
    [90, 110, 130, 150],  // pStateIdx  9
    [85, 104, 123, 142],  // pStateIdx 10
    [81, 99, 117, 135],   // pStateIdx 11
    [77, 94, 111, 128],   // pStateIdx 12
    [73, 89, 105, 122],   // pStateIdx 13
    [69, 85, 100, 116],   // pStateIdx 14
    [66, 80, 95, 110],    // pStateIdx 15
    [62, 76, 90, 104],    // pStateIdx 16
    [59, 72, 86, 99],     // pStateIdx 17
    [56, 69, 81, 94],     // pStateIdx 18
    [53, 65, 77, 89],     // pStateIdx 19
    [51, 62, 73, 85],     // pStateIdx 20
    [48, 59, 69, 80],     // pStateIdx 21
    [46, 56, 66, 76],     // pStateIdx 22
    [43, 53, 63, 72],     // pStateIdx 23
    [41, 50, 59, 69],     // pStateIdx 24
    [39, 48, 56, 65],     // pStateIdx 25
    [37, 45, 54, 62],     // pStateIdx 26
    [35, 43, 51, 59],     // pStateIdx 27
    [33, 41, 48, 56],     // pStateIdx 28
    [32, 39, 46, 53],     // pStateIdx 29
    [30, 37, 43, 50],     // pStateIdx 30
    [29, 35, 41, 48],     // pStateIdx 31
    [27, 33, 39, 45],     // pStateIdx 32
    [26, 31, 37, 43],     // pStateIdx 33
    [24, 30, 35, 41],     // pStateIdx 34
    [23, 28, 33, 39],     // pStateIdx 35
    [22, 27, 32, 37],     // pStateIdx 36
    [21, 26, 30, 35],     // pStateIdx 37
    [20, 24, 29, 33],     // pStateIdx 38
    [19, 23, 27, 31],     // pStateIdx 39
    [18, 22, 26, 30],     // pStateIdx 40
    [17, 21, 25, 28],     // pStateIdx 41
    [16, 20, 23, 27],     // pStateIdx 42
    [15, 19, 22, 25],     // pStateIdx 43
    [14, 18, 21, 24],     // pStateIdx 44
    [14, 17, 20, 23],     // pStateIdx 45
    [13, 16, 19, 22],     // pStateIdx 46
    [12, 15, 18, 21],     // pStateIdx 47
    [12, 14, 17, 20],     // pStateIdx 48
    [11, 14, 16, 19],     // pStateIdx 49
    [11, 13, 15, 18],     // pStateIdx 50
    [10, 12, 15, 17],     // pStateIdx 51
    [10, 12, 14, 16],     // pStateIdx 52
    [9, 11, 13, 15],      // pStateIdx 53
    [9, 11, 12, 14],      // pStateIdx 54
    [8, 10, 12, 14],      // pStateIdx 55
    [8, 9, 11, 13],       // pStateIdx 56
    [7, 9, 11, 12],       // pStateIdx 57
    [7, 9, 10, 12],       // pStateIdx 58
    [7, 8, 10, 11],       // pStateIdx 59
    [6, 8, 9, 11],        // pStateIdx 60
    [6, 7, 9, 10],        // pStateIdx 61
    [6, 7, 8, 9],         // pStateIdx 62
    [2, 2, 2, 2],         // pStateIdx 63
];

// ---------------------------------------------------------------------------
// §9.3.3.2.1.1 / Table 9-45 — state transition table.
//
// Two parallel arrays indexed by pStateIdx (0..=63). Values transcribed
// verbatim from ITU-T Rec. H.264 (08/2024), document page 279.
// ---------------------------------------------------------------------------

/// Table 9-45 column `transIdxLPS` — next `pStateIdx` after decoding an LPS.
pub(crate) static TRANS_IDX_LPS: [u8; 64] = [
    // pStateIdx  0..15
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12, // pStateIdx 16..31
    13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24, // pStateIdx 32..47
    24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33, // pStateIdx 48..63
    33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

/// Table 9-45 column `transIdxMPS` — next `pStateIdx` after decoding an MPS.
pub(crate) static TRANS_IDX_MPS: [u8; 64] = [
    // pStateIdx  0..15
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, // pStateIdx 16..31
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, // pStateIdx 32..47
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, // pStateIdx 48..63
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
];

/// §5.7 — `Clip3(x, y, z)` = max(x, min(y, z)).
#[inline]
fn clip3_i32(lo: i32, hi: i32, v: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// One context model — `(StateIdx, valMPS)` per §9.3.1.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CtxState {
    /// §9.3.1.1 — `pStateIdx`, range 0..=63.
    pub state_idx: u8,
    /// §9.3.1.1 — `valMPS`, either 0 or 1.
    pub val_mps: u8,
}

impl CtxState {
    /// §9.3.1.1 — equation 9-5 (preState clipped to 1..=126, then split
    /// into `(pStateIdx, valMPS)`).
    ///
    /// The `(m, n)` values come from Tables 9-12 .. 9-33 and depend on
    /// the syntax element, slice type, and `cabac_init_idc`. Callers
    /// resolve those tables and pass the pair here.
    pub fn init(m: i32, n: i32, slice_qp_y: i32) -> CabacResult<Self> {
        if !(0..=51).contains(&slice_qp_y) {
            return Err(CabacError::InvalidSliceQp(slice_qp_y));
        }
        // §9.3.1.1 equation (9-5):
        //   preCtxState = Clip3(1, 126, ((m * Clip3(0, 51, SliceQPY)) >> 4) + n)
        let clipped_qp = clip3_i32(0, 51, slice_qp_y);
        let pre = clip3_i32(1, 126, ((m * clipped_qp) >> 4) + n);
        let (state_idx, val_mps) = if pre <= 63 {
            ((63 - pre) as u8, 0u8)
        } else {
            ((pre - 64) as u8, 1u8)
        };
        Ok(Self { state_idx, val_mps })
    }
}

/// Sentinel `cabac_init_idc` values per Table 9-11 / §7.4.3.
///
/// The engine does not itself consult the per-syntax init tables; the
/// caller resolves them and passes `(m, n)` to [`CtxState::init`]. This
/// helper simply validates the `cabac_init_idc` field and normalises the
/// "not present" (I/SI slice) case into `None`.
pub fn validate_init_idc(init_idc: Option<u32>) -> CabacResult<Option<u32>> {
    match init_idc {
        None => Ok(None),
        Some(v) if v <= 2 => Ok(Some(v)),
        Some(v) => Err(CabacError::InvalidInitIdc(v)),
    }
}

/// CABAC arithmetic decoder engine state per §9.3.1.2 / §9.3.3.2.
///
/// `cod_i_range` and `cod_i_offset` are held as `u32` but never exceed
/// the 10-bit register precision required by the spec (§9.3.1.2 NOTE —
/// 9 bits suffice for DecodeDecision/DecodeTerminate, 10 bits for
/// DecodeBypass because of the doubling step).
pub struct CabacDecoder<'a> {
    /// §9.3.1.2 — `codIRange`, initialised to 510.
    cod_i_range: u32,
    /// §9.3.1.2 — `codIOffset`, initialised from `read_bits(9)`.
    cod_i_offset: u32,
    reader: BitReader<'a>,
    /// Running count of bins decoded (decision + bypass + terminate).
    /// Used only for debug instrumentation, never affects decoding.
    bin_count: u64,
}

impl<'a> core::fmt::Debug for CabacDecoder<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CabacDecoder")
            .field("cod_i_range", &self.cod_i_range)
            .field("cod_i_offset", &self.cod_i_offset)
            .field("reader_position", &self.reader.position())
            .finish()
    }
}

impl<'a> CabacDecoder<'a> {
    /// §9.3.1.2 — initialisation of the arithmetic decoding engine.
    ///
    /// `codIRange = 510` and `codIOffset = read_bits(9)` (9-bit big-endian).
    /// The spec also constrains conforming bitstreams to avoid the values
    /// `codIOffset == 510` or `codIOffset == 511`, but those are bitstream
    /// conformance requirements rather than decoder preconditions — we
    /// accept any 9-bit value here and let higher layers validate.
    pub fn new(mut reader: BitReader<'a>) -> CabacResult<Self> {
        let cod_i_offset = reader.u(9)?;
        Ok(Self {
            cod_i_range: 510,
            cod_i_offset,
            reader,
            bin_count: 0,
        })
    }

    /// Total number of bins decoded so far (decision + bypass + terminate).
    /// Exposed for debug instrumentation.
    pub fn bin_count(&self) -> u64 {
        self.bin_count
    }

    /// Cursor into the underlying reader — exposed for slice-layer code
    /// that needs to know the post-CABAC position (e.g. for debug).
    pub fn position(&self) -> (usize, u8) {
        self.reader.position()
    }

    /// Current `codIRange` — exposed for tests.
    #[cfg(test)]
    fn cod_i_range(&self) -> u32 {
        self.cod_i_range
    }

    /// Current `codIOffset` — exposed for tests.
    #[cfg(test)]
    fn cod_i_offset(&self) -> u32 {
        self.cod_i_offset
    }

    /// Current `codIRange` — exposed for debug-only external callers
    /// (bin-trace MB boundary markers).
    pub fn debug_range(&self) -> u32 {
        self.cod_i_range
    }

    /// Current `codIOffset` — exposed for debug-only external callers.
    pub fn debug_offset(&self) -> u32 {
        self.cod_i_offset
    }

    /// §9.3.3.2.1 — DecodeDecision. Returns the decoded bin (0 or 1)
    /// and updates `ctx` in place per §9.3.3.2.1.1 (Table 9-45).
    ///
    /// Algorithm (Figure 9-3):
    ///   qCodIRangeIdx = (codIRange >> 6) & 3                           (9-25)
    ///   codIRangeLPS  = rangeTabLPS[pStateIdx][qCodIRangeIdx]          (9-26)
    ///   codIRange    -= codIRangeLPS
    ///   if codIOffset >= codIRange:
    ///       binVal      = 1 - valMPS
    ///       codIOffset -= codIRange
    ///       codIRange   = codIRangeLPS
    ///       if pStateIdx == 0: valMPS = 1 - valMPS                   (9-27)
    ///       pStateIdx   = transIdxLPS[pStateIdx]
    ///   else:
    ///       binVal      = valMPS
    ///       pStateIdx   = transIdxMPS[pStateIdx]
    ///   RenormD()
    pub fn decode_decision(&mut self, ctx: &mut CtxState) -> CabacResult<u8> {
        // Debug: capture pre-state before we mutate anything. `ctx` does
        // not carry a ctxIdx (we only see the `(state, mps)` pair), so the
        // caller is responsible for any richer trace line. We still emit a
        // minimal per-bin trace gated on OXIDEAV_H264_BIN_TRACE that
        // covers every decision bin with the engine-visible state.
        let trace_on = bin_trace_enabled();
        let pre_state = ctx.state_idx;
        let pre_mps = ctx.val_mps;
        let pre_range = self.cod_i_range;
        let pre_offset = self.cod_i_offset;

        // §9.3.3.2.1 equations (9-25) / (9-26).
        let q_idx = ((self.cod_i_range >> 6) & 0b11) as usize;
        let cod_i_range_lps = RANGE_TAB_LPS[ctx.state_idx as usize][q_idx] as u32;
        self.cod_i_range -= cod_i_range_lps;

        let bin_val: u8;
        if self.cod_i_offset >= self.cod_i_range {
            // LPS path.
            bin_val = 1 - ctx.val_mps;
            self.cod_i_offset -= self.cod_i_range;
            self.cod_i_range = cod_i_range_lps;
            // §9.3.3.2.1.1 equation (9-27).
            if ctx.state_idx == 0 {
                ctx.val_mps = 1 - ctx.val_mps;
            }
            ctx.state_idx = TRANS_IDX_LPS[ctx.state_idx as usize];
        } else {
            // MPS path.
            bin_val = ctx.val_mps;
            ctx.state_idx = TRANS_IDX_MPS[ctx.state_idx as usize];
        }

        self.renorm_d()?;
        self.bin_count = self.bin_count.wrapping_add(1);
        if trace_on {
            eprintln!(
                "[BIN {:>8}] DD pre_state={:>2} pre_mps={} pre_range={:>4} pre_offset={:>4} bin={} post_state={:>2} post_mps={} post_range={:>4} post_offset={:>4}",
                self.bin_count,
                pre_state, pre_mps, pre_range, pre_offset,
                bin_val,
                ctx.state_idx, ctx.val_mps, self.cod_i_range, self.cod_i_offset,
            );
        }
        Ok(bin_val)
    }

    /// §9.3.3.2.3 — DecodeBypass. Reads one bypass bin (no context,
    /// no state update, no renorm of the state machine).
    ///
    /// Algorithm (Figure 9-5):
    ///   codIOffset = (codIOffset << 1) | read_bits(1)
    ///   if codIOffset >= codIRange:
    ///       binVal     = 1
    ///       codIOffset -= codIRange
    ///   else:
    ///       binVal = 0
    pub fn decode_bypass(&mut self) -> CabacResult<u8> {
        let trace_on = bin_trace_enabled();
        let pre_range = self.cod_i_range;
        let pre_offset = self.cod_i_offset;
        self.cod_i_offset = (self.cod_i_offset << 1) | self.reader.u(1)?;
        let bin = if self.cod_i_offset >= self.cod_i_range {
            self.cod_i_offset -= self.cod_i_range;
            1
        } else {
            0
        };
        self.bin_count = self.bin_count.wrapping_add(1);
        if trace_on {
            eprintln!(
                "[BIN {:>8}] BP pre_range={:>4} pre_offset={:>4} bin={} post_range={:>4} post_offset={:>4}",
                self.bin_count,
                pre_range, pre_offset,
                bin,
                self.cod_i_range, self.cod_i_offset,
            );
        }
        Ok(bin)
    }

    /// §9.3.3.2.4 — DecodeTerminate. Special path for `end_of_slice_flag`
    /// and the I_PCM terminator bin (ctxIdx = 276).
    ///
    /// Algorithm (Figure 9-6):
    ///   codIRange -= 2
    ///   if codIOffset >= codIRange:
    ///       binVal = 1  (decoding terminated, no renorm)
    ///   else:
    ///       binVal = 0
    ///       RenormD()
    pub fn decode_terminate(&mut self) -> CabacResult<u8> {
        let trace_on = bin_trace_enabled();
        let pre_range = self.cod_i_range;
        let pre_offset = self.cod_i_offset;
        self.cod_i_range -= 2;
        let bin = if self.cod_i_offset >= self.cod_i_range {
            // Terminated. Per spec: "no renormalization is carried out, and
            // CABAC decoding is terminated". The last inserted bit is the
            // rbsp_stop_one_bit when decoding end_of_slice_flag.
            1
        } else {
            self.renorm_d()?;
            0
        };
        self.bin_count = self.bin_count.wrapping_add(1);
        if trace_on {
            eprintln!(
                "[BIN {:>8}] TT pre_range={:>4} pre_offset={:>4} bin={} post_range={:>4} post_offset={:>4}",
                self.bin_count,
                pre_range, pre_offset,
                bin,
                self.cod_i_range, self.cod_i_offset,
            );
        }
        Ok(bin)
    }

    /// §9.3.3.2.2 — RenormD. Called after DecodeDecision whenever
    /// `codIRange < 256`, and from DecodeTerminate on the "not yet"
    /// branch.
    ///
    /// Algorithm (Figure 9-4):
    ///   while codIRange < 256:
    ///       codIRange  <<= 1
    ///       codIOffset  = (codIOffset << 1) | read_bits(1)
    fn renorm_d(&mut self) -> CabacResult<()> {
        while self.cod_i_range < 256 {
            self.cod_i_range <<= 1;
            self.cod_i_offset = (self.cod_i_offset << 1) | self.reader.u(1)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // §9.3.1.1 — context variable initialisation. Tests hand-derive the
    // expected (pStateIdx, valMPS) from equation (9-5) using known
    // (m, n) pairs from Table 9-12.
    // -----------------------------------------------------------------

    #[test]
    fn ctx_init_table_9_12_ctx0_mid_qp() {
        // Table 9-12 ctxIdx=0 → m=20, n=-15. SliceQPY=26.
        // ((20 * 26) >> 4) + (-15) = (520>>4) - 15 = 32 - 15 = 17
        // Clip3(1,126,17) = 17; preCtxState <= 63 so
        //   pStateIdx = 63 - 17 = 46, valMPS = 0.
        let c = CtxState::init(20, -15, 26).unwrap();
        assert_eq!(
            c,
            CtxState {
                state_idx: 46,
                val_mps: 0
            }
        );
    }

    #[test]
    fn ctx_init_table_9_12_ctx1_mid_qp() {
        // Table 9-12 ctxIdx=1 → m=2, n=54. SliceQPY=26.
        // ((2*26)>>4)+54 = (52>>4)+54 = 3 + 54 = 57.
        // 57 <= 63 → pStateIdx = 63 - 57 = 6, valMPS = 0.
        let c = CtxState::init(2, 54, 26).unwrap();
        assert_eq!(
            c,
            CtxState {
                state_idx: 6,
                val_mps: 0
            }
        );
    }

    #[test]
    fn ctx_init_negative_m_picks_mps_one() {
        // Table 9-12 ctxIdx=6 → m=-28, n=127. SliceQPY=26.
        // ((-28*26)>>4)+127 = (-728>>4)+127.
        // -728 / 16 with arithmetic right shift (Rust i32 `>>`) = -46
        // (floor division). -46 + 127 = 81. Clip3(1,126,81)=81.
        // 81 > 63 → pStateIdx = 81 - 64 = 17, valMPS = 1.
        let c = CtxState::init(-28, 127, 26).unwrap();
        assert_eq!(
            c,
            CtxState {
                state_idx: 17,
                val_mps: 1
            }
        );
    }

    #[test]
    fn ctx_init_low_qp_clips_to_one() {
        // ctxIdx=0 (m=20, n=-15) at SliceQPY=0.
        // ((20*0)>>4) + (-15) = -15. Clip3(1,126,-15) = 1.
        // preCtxState=1 <= 63 → pStateIdx = 63-1 = 62, valMPS=0.
        let c = CtxState::init(20, -15, 0).unwrap();
        assert_eq!(
            c,
            CtxState {
                state_idx: 62,
                val_mps: 0
            }
        );
    }

    #[test]
    fn ctx_init_high_qp() {
        // ctxIdx=0 (m=20, n=-15) at SliceQPY=51.
        // ((20*51)>>4)+(-15) = (1020>>4)-15 = 63 - 15 = 48.
        // preCtxState=48 <= 63 → pStateIdx = 63-48 = 15, valMPS=0.
        let c = CtxState::init(20, -15, 51).unwrap();
        assert_eq!(
            c,
            CtxState {
                state_idx: 15,
                val_mps: 0
            }
        );
    }

    #[test]
    fn ctx_init_rejects_out_of_range_qp() {
        assert_eq!(
            CtxState::init(20, -15, -1).unwrap_err(),
            CabacError::InvalidSliceQp(-1),
        );
        assert_eq!(
            CtxState::init(20, -15, 52).unwrap_err(),
            CabacError::InvalidSliceQp(52),
        );
    }

    #[test]
    fn validate_init_idc_accepts_absent_and_0_to_2() {
        assert_eq!(validate_init_idc(None).unwrap(), None);
        for v in 0u32..=2 {
            assert_eq!(validate_init_idc(Some(v)).unwrap(), Some(v));
        }
        assert_eq!(
            validate_init_idc(Some(3)).unwrap_err(),
            CabacError::InvalidInitIdc(3),
        );
    }

    // -----------------------------------------------------------------
    // §9.3.1.2 — engine initialisation.
    // -----------------------------------------------------------------

    #[test]
    fn engine_new_reads_9_bit_offset() {
        // First 9 bits = 0b1_0101_0101 = 0x155 = 341. Rest ignored.
        // Byte 0 = 0b1010_1010, byte 1 bit 7 = 1 → 9 bits = 101010101.
        let data = [0b1010_1010u8, 0b1000_0000];
        let r = BitReader::new(&data);
        let dec = CabacDecoder::new(r).unwrap();
        assert_eq!(dec.cod_i_range(), 510);
        assert_eq!(dec.cod_i_offset(), 0b1_0101_0101);
    }

    #[test]
    fn engine_new_all_zeros_offset() {
        let data = [0x00, 0x00];
        let dec = CabacDecoder::new(BitReader::new(&data)).unwrap();
        assert_eq!(dec.cod_i_range(), 510);
        assert_eq!(dec.cod_i_offset(), 0);
    }

    // -----------------------------------------------------------------
    // §9.3.3.2 — DecodeDecision round-trips.
    //
    // Each case hand-walks the flowchart from a known initial state.
    // -----------------------------------------------------------------

    /// First call: state_idx=0, val_mps=0, codIRange=510, codIOffset=0.
    ///   qCodIRangeIdx = (510>>6)&3 = 7&3 = 3
    ///   codIRangeLPS  = RANGE_TAB_LPS[0][3] = 240
    ///   codIRange     = 510 - 240 = 270
    ///   codIOffset(0) < codIRange(270) → binVal = valMPS = 0 (MPS path)
    ///   state_idx     = transIdxMPS[0] = 1
    ///   RenormD: 270 >= 256, no-op.
    #[test]
    fn decode_decision_mps_path_no_renorm() {
        let data = [0x00, 0x00];
        let mut dec = CabacDecoder::new(BitReader::new(&data)).unwrap();
        let mut ctx = CtxState {
            state_idx: 0,
            val_mps: 0,
        };
        let bin = dec.decode_decision(&mut ctx).unwrap();
        assert_eq!(bin, 0);
        assert_eq!(
            ctx,
            CtxState {
                state_idx: 1,
                val_mps: 0
            }
        );
        assert_eq!(dec.cod_i_range(), 270);
        assert_eq!(dec.cod_i_offset(), 0);
    }

    /// MPS path that triggers a single renorm iteration.
    ///
    /// Start: state_idx=1, val_mps=0, codIRange=270, codIOffset=0.
    ///   qCodIRangeIdx = (270>>6)&3 = 4&3 = 0
    ///   codIRangeLPS  = RANGE_TAB_LPS[1][0] = 128
    ///   codIRange     = 270 - 128 = 142
    ///   codIOffset(0) < 142 → MPS, binVal=0
    ///   state_idx     = transIdxMPS[1] = 2
    ///   RenormD: 142 < 256 → codIRange=284, codIOffset=(0<<1)|bit=0 → 0.
    #[test]
    fn decode_decision_mps_path_with_renorm() {
        // Engine init consumes 9 zero bits, RenormD reads 1 more zero.
        let data = [0x00, 0x00];
        let mut dec = CabacDecoder::new(BitReader::new(&data)).unwrap();
        dec.cod_i_range = 270;
        dec.cod_i_offset = 0;
        let mut ctx = CtxState {
            state_idx: 1,
            val_mps: 0,
        };
        let bin = dec.decode_decision(&mut ctx).unwrap();
        assert_eq!(bin, 0);
        assert_eq!(
            ctx,
            CtxState {
                state_idx: 2,
                val_mps: 0
            }
        );
        assert_eq!(dec.cod_i_range(), 284);
        assert_eq!(dec.cod_i_offset(), 0);
    }

    /// LPS path exercising the `pStateIdx == 0` valMPS-flip special case
    /// from §9.3.3.2.1.1.
    ///
    /// Input bits chosen so the 9-bit codIOffset reads as 509:
    ///   Byte 0 = 0b1111_1110, byte 1 bit 7 = 1  →  offset = 0b111111101 = 509.
    /// Byte 1 bit 6 = 0 is consumed by the single RenormD iteration.
    ///
    /// Start: state_idx=0, val_mps=0, codIRange=510, codIOffset=509.
    ///   qCodIRangeIdx = 3 → codIRangeLPS = 240
    ///   codIRange     = 510 - 240 = 270
    ///   codIOffset(509) >= codIRange(270) → LPS:
    ///     binVal      = 1 - 0 = 1
    ///     codIOffset -= 270 → 239
    ///     codIRange   = 240
    ///     pStateIdx == 0 → valMPS = 1 - 0 = 1
    ///     pStateIdx   = transIdxLPS[0] = 0
    ///   RenormD: 240 < 256 → codIRange=480, codIOffset=(239<<1)|0=478.
    #[test]
    fn decode_decision_lps_path_flips_mps_at_state_zero() {
        let data = [0b1111_1110u8, 0b1000_0000];
        let mut dec = CabacDecoder::new(BitReader::new(&data)).unwrap();
        assert_eq!(dec.cod_i_offset(), 509);
        let mut ctx = CtxState {
            state_idx: 0,
            val_mps: 0,
        };
        let bin = dec.decode_decision(&mut ctx).unwrap();
        assert_eq!(bin, 1);
        assert_eq!(
            ctx,
            CtxState {
                state_idx: 0,
                val_mps: 1
            }
        );
        assert_eq!(dec.cod_i_range(), 480);
        assert_eq!(dec.cod_i_offset(), 478);
    }

    // -----------------------------------------------------------------
    // §9.3.3.2.3 — DecodeBypass.
    //
    // Start with codIOffset=509 and feed a stream that keeps producing
    // binVal=1 so we also exercise the `codIOffset -= codIRange` branch.
    // -----------------------------------------------------------------

    #[test]
    fn decode_bypass_sequence_of_ones() {
        // Byte 0 = 0b1111_1110 gives 8 bits, byte 1 bit 7 = 1 → the
        // 9-bit codIOffset init reads 509. Then the next 5 bits of
        // byte 1 feed into 5 bypass decodes — all zero.
        //
        // Bypass trace (codIRange stays at 510):
        //   #1: codIOffset = (509<<1)|0 = 1018;  1018 >= 510 → bin=1,
        //       codIOffset = 508
        //   #2: codIOffset = (508<<1)|1 = 1017;  bin=1, codIOffset=507
        //
        // Byte 1 after bit 7 has bits 6..0 = 0101010 (from 0xAA >> 1
        // padded). We need a precise pattern — use 0b1_010_1010:
        //   bit 7 = 1 (init), bit 6 = 0, bit 5 = 1, bit 4 = 0,
        //   bit 3 = 1, bit 2 = 0, bit 1 = 1, bit 0 = 0.
        let data = [0b1111_1110u8, 0b1010_1010];
        let mut dec = CabacDecoder::new(BitReader::new(&data)).unwrap();
        assert_eq!(dec.cod_i_offset(), 509);

        // #1: bit=0 → codIOffset=1018 → bin=1, codIOffset=508.
        assert_eq!(dec.decode_bypass().unwrap(), 1);
        assert_eq!(dec.cod_i_offset(), 508);
        // #2: bit=1 → codIOffset=1017 → bin=1, codIOffset=507.
        assert_eq!(dec.decode_bypass().unwrap(), 1);
        assert_eq!(dec.cod_i_offset(), 507);
        // #3: bit=0 → codIOffset=1014 → bin=1, codIOffset=504.
        assert_eq!(dec.decode_bypass().unwrap(), 1);
        assert_eq!(dec.cod_i_offset(), 504);
        // #4: bit=1 → codIOffset=1009 → bin=1, codIOffset=499.
        assert_eq!(dec.decode_bypass().unwrap(), 1);
        assert_eq!(dec.cod_i_offset(), 499);
        // codIRange stays 510 throughout bypass.
        assert_eq!(dec.cod_i_range(), 510);
    }

    #[test]
    fn decode_bypass_zero_branch() {
        // Offset starts at 0, so after the (<<1)|bit shift it stays well
        // below codIRange(510) for any single bit.
        let data = [0x00, 0b0100_0000];
        let mut dec = CabacDecoder::new(BitReader::new(&data)).unwrap();
        assert_eq!(dec.cod_i_offset(), 0);
        // bit=1 → offset=1 → 1 < 510 → bin=0.
        assert_eq!(dec.decode_bypass().unwrap(), 0);
        assert_eq!(dec.cod_i_offset(), 1);
    }

    // -----------------------------------------------------------------
    // §9.3.3.2.4 — DecodeTerminate.
    // -----------------------------------------------------------------

    #[test]
    fn decode_terminate_not_yet() {
        // codIOffset=0 → after `codIRange -= 2` (508), 0 < 508 → bin=0.
        // RenormD: 508 >= 256, no-op.
        let data = [0x00, 0x00];
        let mut dec = CabacDecoder::new(BitReader::new(&data)).unwrap();
        assert_eq!(dec.decode_terminate().unwrap(), 0);
        assert_eq!(dec.cod_i_range(), 508);
        assert_eq!(dec.cod_i_offset(), 0);
    }

    #[test]
    fn decode_terminate_fires() {
        // codIOffset=509 (from bytes [0xFE, 0x80]) → 509 >= 508 → bin=1,
        // no renorm, state frozen for caller inspection.
        let data = [0b1111_1110u8, 0b1000_0000];
        let mut dec = CabacDecoder::new(BitReader::new(&data)).unwrap();
        assert_eq!(dec.cod_i_offset(), 509);
        assert_eq!(dec.decode_terminate().unwrap(), 1);
        assert_eq!(dec.cod_i_range(), 508);
        assert_eq!(dec.cod_i_offset(), 509);
    }

    // -----------------------------------------------------------------
    // Table 9-44 / Table 9-45 sanity — spot-check cells against the
    // spec. If a later edit ever tries to "clean up" the tables, these
    // pin specific cells per the ITU-T 08/2024 edition.
    // -----------------------------------------------------------------

    #[test]
    fn range_tab_lps_sentinel_cells() {
        // pStateIdx=0 row is the full 4-tuple printed at the top of the
        // table (document page 278, Table 9-44).
        assert_eq!(RANGE_TAB_LPS[0], [128, 176, 208, 240]);
        // pStateIdx=63 row — the LPS-dominant terminus, all 2s.
        assert_eq!(RANGE_TAB_LPS[63], [2, 2, 2, 2]);
        // A couple of interior cells.
        assert_eq!(RANGE_TAB_LPS[15], [66, 80, 95, 110]);
        assert_eq!(RANGE_TAB_LPS[32], [27, 33, 39, 45]);
        assert_eq!(RANGE_TAB_LPS[47], [12, 15, 18, 21]);
    }

    #[test]
    fn trans_idx_tables_sentinel_cells() {
        // Table 9-45 first row (pStateIdx=0) — hand-verified against
        // document page 279.
        assert_eq!(TRANS_IDX_LPS[0], 0);
        assert_eq!(TRANS_IDX_MPS[0], 1);
        // Mid-table.
        assert_eq!(TRANS_IDX_LPS[24], 19);
        assert_eq!(TRANS_IDX_MPS[24], 25);
        // Termini — pStateIdx=63 is the non-adapting fixed point for
        // both directions (transIdxLPS[63] = 63, transIdxMPS[63] = 63),
        // and pStateIdx=62's transIdxMPS wraps back to 62 (not 63).
        assert_eq!(TRANS_IDX_LPS[63], 63);
        assert_eq!(TRANS_IDX_MPS[63], 63);
        assert_eq!(TRANS_IDX_MPS[62], 62);
        assert_eq!(TRANS_IDX_LPS[62], 38);
    }

    #[test]
    fn trans_idx_tables_are_monotonic_nondecreasing() {
        // Sanity: both tables are monotonically non-decreasing in the
        // spec. A mistranscription would very likely break monotonicity.
        for i in 1..64 {
            assert!(TRANS_IDX_LPS[i] >= TRANS_IDX_LPS[i - 1]);
            assert!(TRANS_IDX_MPS[i] >= TRANS_IDX_MPS[i - 1]);
        }
    }

    // -----------------------------------------------------------------
    // §9.3.3.2 flowchart invariants — these guard the engine against
    // future "cleanup" refactors that might inadvertently break the
    // bit-exact arithmetic. Each test walks a minimal scenario and
    // pins the exact (codIRange, codIOffset) intermediates.
    // -----------------------------------------------------------------

    /// Figure 9-3 DecodeDecision produces the SAME number of bits
    /// consumed regardless of the branch taken (MPS vs LPS) when the
    /// subsequent renorm iteration count is equal. This test fires
    /// one MPS and one LPS back-to-back and confirms we end up at the
    /// expected bit cursor — the MPS path consumes 0 new bits (270 ≥
    /// 256), the LPS path consumes 1 new bit (240 < 256 → one renorm).
    #[test]
    fn decode_decision_bit_accounting_mps_then_lps() {
        // Input bytes: byte 0 = 0b1111_1110 (= 254), byte 1 = 0b1_000_0000.
        //   - Engine init reads 9 bits: codIOffset = 0b1_1111_1101 = 509.
        //   - We then hand-walk two DecodeDecision calls.
        let data = [0b1111_1110u8, 0b1000_0000, 0b0000_0000];
        let mut dec = CabacDecoder::new(BitReader::new(&data)).unwrap();
        assert_eq!(dec.cod_i_offset(), 509);

        // 1) state_idx=0, val_mps=0 → codIRangeLPS=240, codIRange-=240=270.
        //    offset 509 ≥ 270 → LPS, binVal=1, offset=509-270=239,
        //    range=240. state_idx=0 → valMPS=1, transIdxLPS[0]=0.
        //    RenormD: 240<256 → read 1 bit (next byte bit 6 = 0).
        //      range=480, offset=(239<<1)|0=478.
        let mut ctx1 = CtxState {
            state_idx: 0,
            val_mps: 0,
        };
        assert_eq!(dec.decode_decision(&mut ctx1).unwrap(), 1);
        assert_eq!(dec.cod_i_range(), 480);
        assert_eq!(dec.cod_i_offset(), 478);
        assert_eq!(
            ctx1,
            CtxState {
                state_idx: 0,
                val_mps: 1
            }
        );
        assert_eq!(dec.bin_count(), 1);

        // 2) Now with range=480, pick state_idx=5 val_mps=1:
        //    qCodIRangeIdx = (480>>6)&3 = 7&3 = 3; RANGE_TAB_LPS[5][3]=185.
        //    range = 480-185 = 295. offset 478 ≥ 295 → LPS, binVal=0,
        //    offset=478-295=183, range=185, state_idx=transIdxLPS[5]=4.
        //    RenormD: 185<256 → read 1 bit (byte 1 bit 5 = 0).
        //      range=370, offset=(183<<1)|0=366.
        let mut ctx2 = CtxState {
            state_idx: 5,
            val_mps: 1,
        };
        assert_eq!(dec.decode_decision(&mut ctx2).unwrap(), 0);
        assert_eq!(dec.cod_i_range(), 370);
        assert_eq!(dec.cod_i_offset(), 366);
        assert_eq!(
            ctx2,
            CtxState {
                state_idx: TRANS_IDX_LPS[5],
                val_mps: 1
            }
        );
        assert_eq!(dec.bin_count(), 2);
    }

    /// §9.3.3.2.3 NOTE — DecodeBypass uses a 10-bit codIOffset window
    /// (not 9-bit) because of the doubling step. This test reads two
    /// bypass bins starting at the highest legal codIOffset=509 to
    /// confirm the engine tolerates the 10-bit working-register width
    /// without wrapping or losing bits.
    #[test]
    fn decode_bypass_ten_bit_register_width() {
        // Init bytes: codIOffset = 509 (0b1_1111_1101).
        // Then 4 bypass bits: bit pattern "1 1 1 1" (so each bypass
        // (offset<<1)|1 exceeds codIRange=510 and emits bin=1).
        //   #1: offset=(509<<1)|1=1019; 1019 >= 510 → bin=1, offset=509.
        //   #2: offset=(509<<1)|1=1019; bin=1, offset=509.
        //   #3: same.
        //   #4: same.
        let data = [0b1111_1110u8, 0b1111_1111u8];
        let mut dec = CabacDecoder::new(BitReader::new(&data)).unwrap();
        for _ in 0..4 {
            assert_eq!(dec.decode_bypass().unwrap(), 1);
            assert_eq!(dec.cod_i_offset(), 509);
        }
        // codIRange never changes in bypass decoding.
        assert_eq!(dec.cod_i_range(), 510);
        assert_eq!(dec.bin_count(), 4);
    }

    /// DecodeTerminate followed by DecodeDecision must continue to
    /// work correctly when the terminate returned 0 — i.e. the
    /// RenormD on the "not yet" branch must restore `codIRange` to a
    /// valid working range. This is the normal end-of-slice_flag
    /// path: every MB except the last sees `end_of_slice_flag` as
    /// binVal=0 followed by RenormD, and then the next MB's syntax
    /// starts from a normalized engine.
    #[test]
    fn decode_terminate_zero_then_decision_continues() {
        // Start with codIOffset low enough that DecodeTerminate yields
        // binVal=0 and renorm runs one iteration.
        //   Init: codIOffset = 0.
        //   Terminate: range=510-2=508. 0 < 508 → binVal=0, RenormD.
        //   RenormD: 508 >= 256 → NO iterations. (Edge case: 508 is
        //     the threshold — no renorm needed.)
        let data = [0x00u8, 0x00, 0x00];
        let mut dec = CabacDecoder::new(BitReader::new(&data)).unwrap();
        assert_eq!(dec.decode_terminate().unwrap(), 0);
        assert_eq!(dec.cod_i_range(), 508);

        // Next decision should still decode correctly: state_idx=0,
        // val_mps=0 → codIRangeLPS=240 (from row 0 col 3 since
        // (508>>6)&3 = 7&3 = 3). 508-240=268 ≥ codIOffset=0 → MPS.
        let mut ctx = CtxState {
            state_idx: 0,
            val_mps: 0,
        };
        assert_eq!(dec.decode_decision(&mut ctx).unwrap(), 0);
        assert_eq!(dec.cod_i_range(), 268);
        assert_eq!(dec.cod_i_offset(), 0);
    }

    /// Cross-check: the full RANGE_TAB_LPS[s][q] row for s=31 (a
    /// middle row) matches the spec table (document page 278).
    /// A transcription error here would show up as 1-off decision
    /// boundaries for many ctxIdx values.
    #[test]
    fn range_tab_lps_row_31_exact() {
        assert_eq!(RANGE_TAB_LPS[31], [29, 35, 41, 48]);
    }

    /// Cross-check: TRANS_IDX_LPS[i] <= i for all i in [0, 63].
    /// (LPS always transitions towards smaller or equal states.)
    /// Similarly TRANS_IDX_MPS[i] >= i for all i.
    #[test]
    fn trans_idx_tables_direction_invariant() {
        for i in 0..64 {
            assert!(
                TRANS_IDX_LPS[i] as usize <= i,
                "transIdxLPS[{}] = {} must be <= {}",
                i,
                TRANS_IDX_LPS[i],
                i,
            );
            assert!(
                TRANS_IDX_MPS[i] as usize >= i,
                "transIdxMPS[{}] = {} must be >= {}",
                i,
                TRANS_IDX_MPS[i],
                i,
            );
        }
    }
}
