//! §9.2 — CAVLC entropy decoding.
//!
//! Spec-driven implementation per ITU-T Rec. H.264 (08/2024), §9.2 and
//! Tables 9-5 through 9-10. Self-contained: tables below are transcribed
//! directly from the PDF, one line per spec entry, with codeword bit
//! strings preserved verbatim so each row can be audited against the
//! table it came from. No external decoder source was consulted.
//!
//! The top-level entry point is [`parse_residual_block_cavlc`], which
//! implements §9.2 (the four sub-clauses) + §7.3.5.3.1 (`residual_block_cavlc()`
//! syntax) to parse one block of transform coefficient levels.
//!
//! # Decode strategy
//!
//! Every VLC table below is encoded as `&[(u32 bits, u8 length, T value)]`,
//! sorted by length ascending (matching the spec tables). To decode, we
//! read bits one at a time into an accumulator and at each step check
//! whether the accumulator (as a `length`-bit value) matches any row.
//! Codewords in all CAVLC tables are prefix-free, so the first match is
//! the answer. This mirrors the spec's "read bits until a codeword
//! matches" phrasing and keeps the transcription unambiguous.

#![allow(dead_code)]

use crate::bitstream::{BitError, BitReader};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CavlcError {
    #[error("bitstream read failed: {0}")]
    Bitstream(#[from] BitError),
    #[error("no matching coeff_token codeword (nC={nc})")]
    UnknownCoeffToken { nc: i32 },
    #[error("level_prefix exceeded 32 leading zeros")]
    LevelPrefixOverflow,
    #[error("no matching total_zeros codeword (total_coeff={tc})")]
    UnknownTotalZeros { tc: u32 },
    #[error("no matching run_before codeword (zeros_left={zl})")]
    UnknownRunBefore { zl: u32 },
    #[error("invalid total_coeff={tc} for CAVLC (max {max})")]
    InvalidTotalCoeff { tc: u32, max: u32 },
    #[error("invalid trailing_ones={t1} for total_coeff={tc}")]
    InvalidTrailingOnes { t1: u32, tc: u32 },
}

pub type CavlcResult<T> = Result<T, CavlcError>;

// ------------------------------------------------------------------
// Context selector for coeff_token (§9.2.1.1)
// ------------------------------------------------------------------

/// `nC` selector for the coeff_token tables (§9.2.1.1).
///
/// The spec derives `nC` from neighbouring blocks — see §9.2.1.1 ordered
/// steps 1..7. The derivation lives above CAVLC (it needs macroblock /
/// neighbour context); this module just accepts the final value.
#[derive(Debug, Clone, Copy)]
pub enum CoeffTokenContext {
    /// nC value derived from neighbouring blocks (§9.2.1.1 step 7).
    /// Valid range: `0..=16`, though 8..=16 all use the 4th column.
    Numeric(i32),
    /// Use the `nC == -1` column — ChromaDCLevel for ChromaArrayType=1
    /// (i.e. 4:2:0 chroma sampling), per §9.2.1.1.
    ChromaDc420,
    /// Use the `nC == -2` column — ChromaDCLevel for ChromaArrayType=2
    /// (i.e. 4:2:2 chroma sampling), per §9.2.1.1.
    ChromaDc422,
}

impl CoeffTokenContext {
    /// Map the context to the coeff_token table to use. See Table 9-5.
    fn select_table(self) -> &'static [CoeffTokenRow] {
        match self {
            CoeffTokenContext::Numeric(nc) => {
                if nc < 2 {
                    &TABLE_9_5_COL0
                } else if nc < 4 {
                    &TABLE_9_5_COL1
                } else if nc < 8 {
                    &TABLE_9_5_COL2
                } else {
                    &TABLE_9_5_COL3
                }
            }
            CoeffTokenContext::ChromaDc420 => &TABLE_9_5_COL_NC_M1,
            CoeffTokenContext::ChromaDc422 => &TABLE_9_5_COL_NC_M2,
        }
    }

    /// For error reporting.
    fn nc_for_error(self) -> i32 {
        match self {
            CoeffTokenContext::Numeric(nc) => nc,
            CoeffTokenContext::ChromaDc420 => -1,
            CoeffTokenContext::ChromaDc422 => -2,
        }
    }
}

// ------------------------------------------------------------------
// Common VLC decoder
// ------------------------------------------------------------------

/// Generic VLC lookup: reads bits one at a time and returns the value
/// associated with the first matching codeword. `rows` must be sorted
/// by `length` ascending (codewords of equal length may appear in any
/// order since they are mutually exclusive).
fn decode_vlc<T: Copy>(
    r: &mut BitReader<'_>,
    rows: &[(u32, u8, T)],
) -> Result<Option<T>, BitError> {
    let max_len = rows.iter().map(|(_, l, _)| *l).max().unwrap_or(0);
    if max_len == 0 {
        return Ok(None);
    }
    let mut acc: u32 = 0;
    let mut len: u8 = 0;
    while len < max_len {
        acc = (acc << 1) | r.u(1)?;
        len += 1;
        for (bits, rlen, value) in rows {
            if *rlen == len && *bits == acc {
                return Ok(Some(*value));
            }
        }
    }
    Ok(None)
}

// ------------------------------------------------------------------
// §9.2.1 — Table 9-5 rows
// ------------------------------------------------------------------

/// (total_coeff, trailing_ones). TotalCoeff spans 0..=16; TrailingOnes 0..=3.
type CoeffTokenRow = (u32, u8, (u32, u32));

// §9.2.1 — Table 9-5, column `0 <= nC < 2`. 62 rows.
// Each tuple is (codeword bits, codeword length, (TotalCoeff, TrailingOnes))
// transcribed verbatim from the PDF rows (p. 219-220).
#[rustfmt::skip]
static TABLE_9_5_COL0: [CoeffTokenRow; 62] = [
    // TrailingOnes=0, TotalCoeff=0 : "1"
    (0b1,                  1, (0, 0)),
    // TrailingOnes=0, TotalCoeff=1 : "0001 01"
    (0b000101,             6, (1, 0)),
    // TrailingOnes=1, TotalCoeff=1 : "01"
    (0b01,                 2, (1, 1)),
    // TrailingOnes=0, TotalCoeff=2 : "0000 0111"
    (0b00000111,           8, (2, 0)),
    // TrailingOnes=1, TotalCoeff=2 : "0001 00"
    (0b000100,             6, (2, 1)),
    // TrailingOnes=2, TotalCoeff=2 : "001"
    (0b001,                3, (2, 2)),
    // TrailingOnes=0, TotalCoeff=3 : "0000 0011 1"
    (0b000000111,          9, (3, 0)),
    // TrailingOnes=1, TotalCoeff=3 : "0000 0110"
    (0b00000110,           8, (3, 1)),
    // TrailingOnes=2, TotalCoeff=3 : "0000 101"
    (0b0000101,            7, (3, 2)),
    // TrailingOnes=3, TotalCoeff=3 : "0001 1"
    (0b00011,              5, (3, 3)),
    // TrailingOnes=0, TotalCoeff=4 : "0000 0001 11"
    (0b0000000111,         10, (4, 0)),
    // TrailingOnes=1, TotalCoeff=4 : "0000 0011 0"
    (0b000000110,          9, (4, 1)),
    // TrailingOnes=2, TotalCoeff=4 : "0000 0101"
    (0b00000101,           8, (4, 2)),
    // TrailingOnes=3, TotalCoeff=4 : "0000 11"
    (0b000011,             6, (4, 3)),
    // TrailingOnes=0, TotalCoeff=5 : "0000 0000 111"
    (0b00000000111,        11, (5, 0)),
    // TrailingOnes=1, TotalCoeff=5 : "0000 0001 10"
    (0b0000000110,         10, (5, 1)),
    // TrailingOnes=2, TotalCoeff=5 : "0000 0010 1"
    (0b000000101,          9, (5, 2)),
    // TrailingOnes=3, TotalCoeff=5 : "0000 100"
    (0b0000100,            7, (5, 3)),
    // TrailingOnes=0, TotalCoeff=6 : "0000 0000 0111 1"
    (0b0000000001111,      13, (6, 0)),
    // TrailingOnes=1, TotalCoeff=6 : "0000 0000 110"
    (0b00000000110,        11, (6, 1)),
    // TrailingOnes=2, TotalCoeff=6 : "0000 0001 01"
    (0b0000000101,         10, (6, 2)),
    // TrailingOnes=3, TotalCoeff=6 : "0000 0100"
    (0b00000100,           8, (6, 3)),
    // TrailingOnes=0, TotalCoeff=7 : "0000 0000 0101 1"
    (0b0000000001011,      13, (7, 0)),
    // TrailingOnes=1, TotalCoeff=7 : "0000 0000 0111 0"
    (0b0000000001110,      13, (7, 1)),
    // TrailingOnes=2, TotalCoeff=7 : "0000 0000 101"
    (0b00000000101,        11, (7, 2)),
    // TrailingOnes=3, TotalCoeff=7 : "0000 0010 0"
    (0b000000100,          9, (7, 3)),
    // TrailingOnes=0, TotalCoeff=8 : "0000 0000 0100 0"
    (0b0000000001000,      13, (8, 0)),
    // TrailingOnes=1, TotalCoeff=8 : "0000 0000 0101 0"
    (0b0000000001010,      13, (8, 1)),
    // TrailingOnes=2, TotalCoeff=8 : "0000 0000 0110 1"
    (0b0000000001101,      13, (8, 2)),
    // TrailingOnes=3, TotalCoeff=8 : "0000 0001 00"
    (0b0000000100,         10, (8, 3)),
    // TrailingOnes=0, TotalCoeff=9 : "0000 0000 0011 11"
    (0b00000000001111,     14, (9, 0)),
    // TrailingOnes=1, TotalCoeff=9 : "0000 0000 0011 10"
    (0b00000000001110,     14, (9, 1)),
    // TrailingOnes=2, TotalCoeff=9 : "0000 0000 0100 1"
    (0b0000000001001,      13, (9, 2)),
    // TrailingOnes=3, TotalCoeff=9 : "0000 0000 100"
    (0b00000000100,        11, (9, 3)),
    // TrailingOnes=0, TotalCoeff=10: "0000 0000 0010 11"
    (0b00000000001011,     14, (10, 0)),
    // TrailingOnes=1, TotalCoeff=10: "0000 0000 0010 10"
    (0b00000000001010,     14, (10, 1)),
    // TrailingOnes=2, TotalCoeff=10: "0000 0000 0011 01"
    (0b00000000001101,     14, (10, 2)),
    // TrailingOnes=3, TotalCoeff=10: "0000 0000 0110 0"
    (0b0000000001100,      13, (10, 3)),
    // TrailingOnes=0, TotalCoeff=11: "0000 0000 0001 111"
    (0b000000000001111,    15, (11, 0)),
    // TrailingOnes=1, TotalCoeff=11: "0000 0000 0001 110"
    (0b000000000001110,    15, (11, 1)),
    // TrailingOnes=2, TotalCoeff=11: "0000 0000 0010 01"
    (0b00000000001001,     14, (11, 2)),
    // TrailingOnes=3, TotalCoeff=11: "0000 0000 0011 00"
    (0b00000000001100,     14, (11, 3)),
    // TrailingOnes=0, TotalCoeff=12: "0000 0000 0001 011"
    (0b000000000001011,    15, (12, 0)),
    // TrailingOnes=1, TotalCoeff=12: "0000 0000 0001 010"
    (0b000000000001010,    15, (12, 1)),
    // TrailingOnes=2, TotalCoeff=12: "0000 0000 0001 101"
    (0b000000000001101,    15, (12, 2)),
    // TrailingOnes=3, TotalCoeff=12: "0000 0000 0010 00"
    (0b00000000001000,     14, (12, 3)),
    // TrailingOnes=0, TotalCoeff=13: "0000 0000 0000 1111"
    (0b0000000000001111,   16, (13, 0)),
    // TrailingOnes=1, TotalCoeff=13: "0000 0000 0000 001"
    (0b000000000000001,    15, (13, 1)),
    // TrailingOnes=2, TotalCoeff=13: "0000 0000 0001 001"
    (0b000000000001001,    15, (13, 2)),
    // TrailingOnes=3, TotalCoeff=13: "0000 0000 0001 100"
    (0b000000000001100,    15, (13, 3)),
    // TrailingOnes=0, TotalCoeff=14: "0000 0000 0000 1011"
    (0b0000000000001011,   16, (14, 0)),
    // TrailingOnes=1, TotalCoeff=14: "0000 0000 0000 1110"
    (0b0000000000001110,   16, (14, 1)),
    // TrailingOnes=2, TotalCoeff=14: "0000 0000 0000 1101"
    (0b0000000000001101,   16, (14, 2)),
    // TrailingOnes=3, TotalCoeff=14: "0000 0000 0001 000"
    (0b000000000001000,    15, (14, 3)),
    // TrailingOnes=0, TotalCoeff=15: "0000 0000 0000 0111"
    (0b0000000000000111,   16, (15, 0)),
    // TrailingOnes=1, TotalCoeff=15: "0000 0000 0000 1010"
    (0b0000000000001010,   16, (15, 1)),
    // TrailingOnes=2, TotalCoeff=15: "0000 0000 0000 1001"
    (0b0000000000001001,   16, (15, 2)),
    // TrailingOnes=3, TotalCoeff=15: "0000 0000 0000 1100"
    (0b0000000000001100,   16, (15, 3)),
    // TrailingOnes=0, TotalCoeff=16: "0000 0000 0000 0100"
    (0b0000000000000100,   16, (16, 0)),
    // TrailingOnes=1, TotalCoeff=16: "0000 0000 0000 0110"
    (0b0000000000000110,   16, (16, 1)),
    // TrailingOnes=2, TotalCoeff=16: "0000 0000 0000 0101"
    (0b0000000000000101,   16, (16, 2)),
    // TrailingOnes=3, TotalCoeff=16: "0000 0000 0000 1000"
    (0b0000000000001000,   16, (16, 3)),
];

// §9.2.1 — Table 9-5, column `2 <= nC < 4`. 62 rows.
#[rustfmt::skip]
static TABLE_9_5_COL1: [CoeffTokenRow; 62] = [
    // T1=0, TC=0 : "11"
    (0b11,                 2, (0, 0)),
    // T1=0, TC=1 : "0010 11"
    (0b001011,             6, (1, 0)),
    // T1=1, TC=1 : "10"
    (0b10,                 2, (1, 1)),
    // T1=0, TC=2 : "0001 11"
    (0b000111,             6, (2, 0)),
    // T1=1, TC=2 : "0011 1"
    (0b00111,              5, (2, 1)),
    // T1=2, TC=2 : "011"
    (0b011,                3, (2, 2)),
    // T1=0, TC=3 : "0000 111"
    (0b0000111,            7, (3, 0)),
    // T1=1, TC=3 : "0010 10"
    (0b001010,             6, (3, 1)),
    // T1=2, TC=3 : "0010 01"
    (0b001001,             6, (3, 2)),
    // T1=3, TC=3 : "0101"
    (0b0101,               4, (3, 3)),
    // T1=0, TC=4 : "0000 0111"
    (0b00000111,           8, (4, 0)),
    // T1=1, TC=4 : "0001 10"
    (0b000110,             6, (4, 1)),
    // T1=2, TC=4 : "0001 01"
    (0b000101,             6, (4, 2)),
    // T1=3, TC=4 : "0100"
    (0b0100,               4, (4, 3)),
    // T1=0, TC=5 : "0000 0100"
    (0b00000100,           8, (5, 0)),
    // T1=1, TC=5 : "0000 110"
    (0b0000110,            7, (5, 1)),
    // T1=2, TC=5 : "0000 101"
    (0b0000101,            7, (5, 2)),
    // T1=3, TC=5 : "0011 0"
    (0b00110,              5, (5, 3)),
    // T1=0, TC=6 : "0000 0011 1"
    (0b000000111,          9, (6, 0)),
    // T1=1, TC=6 : "0000 0110"
    (0b00000110,           8, (6, 1)),
    // T1=2, TC=6 : "0000 0101"
    (0b00000101,           8, (6, 2)),
    // T1=3, TC=6 : "0010 00"
    (0b001000,             6, (6, 3)),
    // T1=0, TC=7 : "0000 0001 111" — 11 bits.
    // BUGFIX 2026-04-20: length was previously 10, which is one phantom
    // leading zero — desync for nC in [2,4) whose first block was (TC=7,T1=0).
    (0b00000001111,        11, (7, 0)),
    // T1=1, TC=7 : "0000 0011 0"
    (0b000000110,          9, (7, 1)),
    // T1=2, TC=7 : "0000 0010 1"
    (0b000000101,          9, (7, 2)),
    // T1=3, TC=7 : "0001 00"
    (0b000100,             6, (7, 3)),
    // T1=0, TC=8 : "0000 0001 011" — 11 bits.
    // BUGFIX 2026-04-20: length was previously 10 (see §9.2.1 Table 9-5 col "2 <= nC < 4").
    (0b00000001011,        11, (8, 0)),
    // T1=1, TC=8 : "0000 0001 110" — 11 bits.
    // BUGFIX 2026-04-20: length was previously 10.
    (0b00000001110,        11, (8, 1)),
    // T1=2, TC=8 : "0000 0001 101" — 11 bits.
    // BUGFIX 2026-04-20: length was previously 10.
    (0b00000001101,        11, (8, 2)),
    // T1=3, TC=8 : "0000 100"
    (0b0000100,            7, (8, 3)),
    // T1=0, TC=9 : "0000 0000 1111"
    (0b000000001111,       12, (9, 0)),
    // T1=1, TC=9 : "0000 0001 010" — 11 bits.
    // BUGFIX 2026-04-20: length was previously 10.
    (0b00000001010,        11, (9, 1)),
    // T1=2, TC=9 : "0000 0001 001" — 11 bits.
    // BUGFIX 2026-04-20: length was previously 10.
    (0b00000001001,        11, (9, 2)),
    // T1=3, TC=9 : "0000 0010 0"
    (0b000000100,          9, (9, 3)),
    // T1=0, TC=10: "0000 0000 1011"
    (0b000000001011,       12, (10, 0)),
    // T1=1, TC=10: "0000 0000 1110"
    (0b000000001110,       12, (10, 1)),
    // T1=2, TC=10: "0000 0000 1101"
    (0b000000001101,       12, (10, 2)),
    // T1=3, TC=10: "0000 0001 100" — 11 bits.
    // BUGFIX 2026-04-20: length was previously 10.
    (0b00000001100,        11, (10, 3)),
    // T1=0, TC=11: "0000 0000 1000"
    (0b000000001000,       12, (11, 0)),
    // T1=1, TC=11: "0000 0000 1010"
    (0b000000001010,       12, (11, 1)),
    // T1=2, TC=11: "0000 0000 1001"
    (0b000000001001,       12, (11, 2)),
    // T1=3, TC=11: "0000 0001 000" — 11 bits.
    // BUGFIX 2026-04-20: length was previously 10.
    (0b00000001000,        11, (11, 3)),
    // T1=0, TC=12: "0000 0000 0111 1"
    (0b0000000001111,      13, (12, 0)),
    // T1=1, TC=12: "0000 0000 0111 0"
    (0b0000000001110,      13, (12, 1)),
    // T1=2, TC=12: "0000 0000 0110 1"
    (0b0000000001101,      13, (12, 2)),
    // T1=3, TC=12: "0000 0000 1100"
    (0b000000001100,       12, (12, 3)),
    // T1=0, TC=13: "0000 0000 0101 1"
    (0b0000000001011,      13, (13, 0)),
    // T1=1, TC=13: "0000 0000 0101 0"
    (0b0000000001010,      13, (13, 1)),
    // T1=2, TC=13: "0000 0000 0100 1"
    (0b0000000001001,      13, (13, 2)),
    // T1=3, TC=13: "0000 0000 0110 0"
    (0b0000000001100,      13, (13, 3)),
    // T1=0, TC=14: "0000 0000 0011 1"
    (0b0000000000111,      13, (14, 0)),
    // T1=1, TC=14: "0000 0000 0010 11"
    (0b00000000001011,     14, (14, 1)),
    // T1=2, TC=14: "0000 0000 0011 0"
    (0b0000000000110,      13, (14, 2)),
    // T1=3, TC=14: "0000 0000 0100 0"
    (0b0000000001000,      13, (14, 3)),
    // T1=0, TC=15: "0000 0000 0010 01"
    (0b00000000001001,     14, (15, 0)),
    // T1=1, TC=15: "0000 0000 0010 00"
    (0b00000000001000,     14, (15, 1)),
    // T1=2, TC=15: "0000 0000 0010 10"
    (0b00000000001010,     14, (15, 2)),
    // T1=3, TC=15: "0000 0000 0000 1" — 13 bits.
    // BUGFIX 2026-04-20: the length was previously 15, which is two
    // extra phantom leading zeros — the VLC decoder would then never
    // match this codeword and instead consume bits for other 13-bit
    // entries, desyncing the stream for any MB whose nC was in [2,4)
    // and whose first block was (TC=15, T1=3).
    (0b0000000000001,      13, (15, 3)),
    // T1=0, TC=16: "0000 0000 0001 11"
    (0b00000000000111,     14, (16, 0)),
    // T1=1, TC=16: "0000 0000 0001 10"
    (0b00000000000110,     14, (16, 1)),
    // T1=2, TC=16: "0000 0000 0001 01"
    (0b00000000000101,     14, (16, 2)),
    // T1=3, TC=16: "0000 0000 0001 00"
    (0b00000000000100,     14, (16, 3)),
];

// §9.2.1 — Table 9-5, column `4 <= nC < 8`. 62 rows.
#[rustfmt::skip]
static TABLE_9_5_COL2: [CoeffTokenRow; 62] = [
    // T1=0, TC=0 : "1111"
    (0b1111,               4, (0, 0)),
    // T1=0, TC=1 : "0011 11"
    (0b001111,             6, (1, 0)),
    // T1=1, TC=1 : "1110"
    (0b1110,               4, (1, 1)),
    // T1=0, TC=2 : "0010 11"
    (0b001011,             6, (2, 0)),
    // T1=1, TC=2 : "0111 1"
    (0b01111,              5, (2, 1)),
    // T1=2, TC=2 : "1101"
    (0b1101,               4, (2, 2)),
    // T1=0, TC=3 : "0010 00"
    (0b001000,             6, (3, 0)),
    // T1=1, TC=3 : "0110 0"
    (0b01100,              5, (3, 1)),
    // T1=2, TC=3 : "0111 0"
    (0b01110,              5, (3, 2)),
    // T1=3, TC=3 : "1100"
    (0b1100,               4, (3, 3)),
    // T1=0, TC=4 : "0001 111"
    (0b0001111,            7, (4, 0)),
    // T1=1, TC=4 : "0101 0"
    (0b01010,              5, (4, 1)),
    // T1=2, TC=4 : "0101 1"
    (0b01011,              5, (4, 2)),
    // T1=3, TC=4 : "1011"
    (0b1011,               4, (4, 3)),
    // T1=0, TC=5 : "0001 011"
    (0b0001011,            7, (5, 0)),
    // T1=1, TC=5 : "0100 0"
    (0b01000,              5, (5, 1)),
    // T1=2, TC=5 : "0100 1"
    (0b01001,              5, (5, 2)),
    // T1=3, TC=5 : "1010"
    (0b1010,               4, (5, 3)),
    // T1=0, TC=6 : "0001 001"
    (0b0001001,            7, (6, 0)),
    // T1=1, TC=6 : "0011 10"
    (0b001110,             6, (6, 1)),
    // T1=2, TC=6 : "0011 01"
    (0b001101,             6, (6, 2)),
    // T1=3, TC=6 : "1001"
    (0b1001,               4, (6, 3)),
    // T1=0, TC=7 : "0001 000"
    (0b0001000,            7, (7, 0)),
    // T1=1, TC=7 : "0010 10"
    (0b001010,             6, (7, 1)),
    // T1=2, TC=7 : "0010 01"
    (0b001001,             6, (7, 2)),
    // T1=3, TC=7 : "1000"
    (0b1000,               4, (7, 3)),
    // T1=0, TC=8 : "0000 1111"
    (0b00001111,           8, (8, 0)),
    // T1=1, TC=8 : "0001 110"
    (0b0001110,            7, (8, 1)),
    // T1=2, TC=8 : "0001 101"
    (0b0001101,            7, (8, 2)),
    // T1=3, TC=8 : "0110 1"
    (0b01101,              5, (8, 3)),
    // T1=0, TC=9 : "0000 1011"
    (0b00001011,           8, (9, 0)),
    // T1=1, TC=9 : "0000 1110"
    (0b00001110,           8, (9, 1)),
    // T1=2, TC=9 : "0001 010"
    (0b0001010,            7, (9, 2)),
    // T1=3, TC=9 : "0011 00"
    (0b001100,             6, (9, 3)),
    // T1=0, TC=10: "0000 0111 1"
    (0b000001111,          9, (10, 0)),
    // T1=1, TC=10: "0000 1010"
    (0b00001010,           8, (10, 1)),
    // T1=2, TC=10: "0000 1101"
    (0b00001101,           8, (10, 2)),
    // T1=3, TC=10: "0001 100"
    (0b0001100,            7, (10, 3)),
    // T1=0, TC=11: "0000 0101 1"
    (0b000001011,          9, (11, 0)),
    // T1=1, TC=11: "0000 0111 0"
    (0b000001110,          9, (11, 1)),
    // T1=2, TC=11: "0000 1001"
    (0b00001001,           8, (11, 2)),
    // T1=3, TC=11: "0000 1100"
    (0b00001100,           8, (11, 3)),
    // T1=0, TC=12: "0000 0100 0"
    (0b000001000,          9, (12, 0)),
    // T1=1, TC=12: "0000 0101 0"
    (0b000001010,          9, (12, 1)),
    // T1=2, TC=12: "0000 0110 1"
    (0b000001101,          9, (12, 2)),
    // T1=3, TC=12: "0000 1000"
    (0b00001000,           8, (12, 3)),
    // T1=0, TC=13: "0000 0011 01"
    (0b0000001101,         10, (13, 0)),
    // T1=1, TC=13: "0000 0011 1"
    (0b000000111,          9, (13, 1)),
    // T1=2, TC=13: "0000 0100 1"
    (0b000001001,          9, (13, 2)),
    // T1=3, TC=13: "0000 0110 0"
    (0b000001100,          9, (13, 3)),
    // T1=0, TC=14: "0000 0010 01"
    (0b0000001001,         10, (14, 0)),
    // T1=1, TC=14: "0000 0011 00"
    (0b0000001100,         10, (14, 1)),
    // T1=2, TC=14: "0000 0010 11"
    (0b0000001011,         10, (14, 2)),
    // T1=3, TC=14: "0000 0010 10"
    (0b0000001010,         10, (14, 3)),
    // T1=0, TC=15: "0000 0001 01"
    (0b0000000101,         10, (15, 0)),
    // T1=1, TC=15: "0000 0010 00"
    (0b0000001000,         10, (15, 1)),
    // T1=2, TC=15: "0000 0001 11"
    (0b0000000111,         10, (15, 2)),
    // T1=3, TC=15: "0000 0001 10"
    (0b0000000110,         10, (15, 3)),
    // T1=0, TC=16: "0000 0000 01"
    (0b0000000001,         10, (16, 0)),
    // T1=1, TC=16: "0000 0001 00"
    (0b0000000100,         10, (16, 1)),
    // T1=2, TC=16: "0000 0000 11"
    (0b0000000011,         10, (16, 2)),
    // T1=3, TC=16: "0000 0000 10"
    (0b0000000010,         10, (16, 3)),
];

// §9.2.1 — Table 9-5, column `8 <= nC`.
// This column is a 6-bit fixed-length code: bits 5..2 = TotalCoeff (4 bits)
// and bits 1..0 = TrailingOnes (2 bits). Offsets shown in the PDF:
//   (0,0)=0000 11, (1,1)=0000 00, ... (16,3)=1111 11
// Concretely, codeword = (TotalCoeff << 2) | TrailingOnes + an offset
// that puts (0,0) at 0b000011 and walks up sequentially.
//
// Reading the PDF values:
//   (0,0)=00 0011, (0,1)=00 0000, (1,1)=00 0001, (0,2)=00 0100,
//   (1,2)=00 0101, (2,2)=00 0110, (0,3)=00 1000, (1,3)=00 1001,
//   (2,3)=00 1010, (3,3)=00 1011, (0,4)=00 1100, (1,4)=00 1101,
//   (2,4)=00 1110, (3,4)=00 1111, (0,5)=01 0000, ..., (3,16)=1111 11
//
// Verification: the numeric column in Table 9-5 is a dense 6-bit
// enumeration with TotalCoeff=0,TrailingOnes=0 anchored at 000011 and
// all subsequent (TotalCoeff, TrailingOnes) pairs increasing by 1 in
// row-major order with TrailingOnes=0..=3 within TotalCoeff groups,
// EXCEPT that (0,0) sits at 000011 and (0,1)..(0,3), (1,0)..(1,3) slots
// are rearranged per the PDF.
//
// Rather than encode the subtle offset math, we emit one row per entry
// from Table 9-5 column "8 <= nC" verbatim.
//
// BUGFIX 2026-04-20: the prior transcription of this column stored value
// tuples as `(TrailingOnes, TotalCoeff)` — i.e. the two columns were
// swapped relative to all other columns (COL0/COL1/COL2 store
// `(TotalCoeff, TrailingOnes)`). That broke any CAVLC block with
// neighbour `nC >= 8` — e.g. once a sufficiently dense MB had been
// decoded, subsequent blocks that crossed into the COL3 selector
// produced bogus `(total_coeff, trailing_ones)` pairs like (1, 3) or
// (0, 1) that fail the `trailing_ones ≤ total_coeff` invariant. The
// tuples below are now stored as `(TotalCoeff, TrailingOnes)` per the
// `CoeffTokenRow` contract.
#[rustfmt::skip]
static TABLE_9_5_COL3: [CoeffTokenRow; 62] = [
    // Codeword | (TotalCoeff, TrailingOnes) per Table 9-5 "8 <= nC".
    (0b000011, 6, (0, 0)),
    (0b000000, 6, (1, 0)),
    (0b000001, 6, (1, 1)),
    (0b000100, 6, (2, 0)),
    (0b000101, 6, (2, 1)),
    (0b000110, 6, (2, 2)),
    (0b001000, 6, (3, 0)),
    (0b001001, 6, (3, 1)),
    (0b001010, 6, (3, 2)),
    (0b001011, 6, (3, 3)),
    (0b001100, 6, (4, 0)),
    (0b001101, 6, (4, 1)),
    (0b001110, 6, (4, 2)),
    (0b001111, 6, (4, 3)),
    (0b010000, 6, (5, 0)),
    (0b010001, 6, (5, 1)),
    (0b010010, 6, (5, 2)),
    (0b010011, 6, (5, 3)),
    (0b010100, 6, (6, 0)),
    (0b010101, 6, (6, 1)),
    (0b010110, 6, (6, 2)),
    (0b010111, 6, (6, 3)),
    (0b011000, 6, (7, 0)),
    (0b011001, 6, (7, 1)),
    (0b011010, 6, (7, 2)),
    (0b011011, 6, (7, 3)),
    (0b011100, 6, (8, 0)),
    (0b011101, 6, (8, 1)),
    (0b011110, 6, (8, 2)),
    (0b011111, 6, (8, 3)),
    (0b100000, 6, (9, 0)),
    (0b100001, 6, (9, 1)),
    (0b100010, 6, (9, 2)),
    (0b100011, 6, (9, 3)),
    (0b100100, 6, (10, 0)),
    (0b100101, 6, (10, 1)),
    (0b100110, 6, (10, 2)),
    (0b100111, 6, (10, 3)),
    (0b101000, 6, (11, 0)),
    (0b101001, 6, (11, 1)),
    (0b101010, 6, (11, 2)),
    (0b101011, 6, (11, 3)),
    (0b101100, 6, (12, 0)),
    (0b101101, 6, (12, 1)),
    (0b101110, 6, (12, 2)),
    (0b101111, 6, (12, 3)),
    (0b110000, 6, (13, 0)),
    (0b110001, 6, (13, 1)),
    (0b110010, 6, (13, 2)),
    (0b110011, 6, (13, 3)),
    (0b110100, 6, (14, 0)),
    (0b110101, 6, (14, 1)),
    (0b110110, 6, (14, 2)),
    (0b110111, 6, (14, 3)),
    (0b111000, 6, (15, 0)),
    (0b111001, 6, (15, 1)),
    (0b111010, 6, (15, 2)),
    (0b111011, 6, (15, 3)),
    (0b111100, 6, (16, 0)),
    (0b111101, 6, (16, 1)),
    (0b111110, 6, (16, 2)),
    (0b111111, 6, (16, 3)),
];

// §9.2.1 — Table 9-5, column `nC == -1` (ChromaDCLevel, ChromaArrayType=1).
// Max TotalCoeff is 4 (ChromaDC 2x2 has 4 coefficients).
#[rustfmt::skip]
static TABLE_9_5_COL_NC_M1: [CoeffTokenRow; 14] = [
    // T1=0, TC=0 : "01"
    (0b01,        2, (0, 0)),
    // T1=0, TC=1 : "0001 11"
    (0b000111,    6, (1, 0)),
    // T1=1, TC=1 : "1"
    (0b1,         1, (1, 1)),
    // T1=0, TC=2 : "0001 00"
    (0b000100,    6, (2, 0)),
    // T1=1, TC=2 : "0001 10"
    (0b000110,    6, (2, 1)),
    // T1=2, TC=2 : "001"
    (0b001,       3, (2, 2)),
    // T1=0, TC=3 : "0000 11"
    (0b000011,    6, (3, 0)),
    // T1=1, TC=3 : "0000 011"
    (0b0000011,   7, (3, 1)),
    // T1=2, TC=3 : "0000 010"
    (0b0000010,   7, (3, 2)),
    // T1=3, TC=3 : "0001 01"
    (0b000101,    6, (3, 3)),
    // T1=0, TC=4 : "0000 10"
    (0b000010,    6, (4, 0)),
    // T1=1, TC=4 : "0000 0011"
    (0b00000011,  8, (4, 1)),
    // T1=2, TC=4 : "0000 0010"
    (0b00000010,  8, (4, 2)),
    // T1=3, TC=4 : "0000 000"
    (0b0000000,   7, (4, 3)),
];

// §9.2.1 — Table 9-5, column `nC == -2` (ChromaDCLevel, ChromaArrayType=2).
// Max TotalCoeff is 8 (ChromaDC 2x4 has 8 coefficients).
#[rustfmt::skip]
static TABLE_9_5_COL_NC_M2: [CoeffTokenRow; 30] = [
    // T1=0, TC=0 : "1"
    (0b1,             1, (0, 0)),
    // T1=0, TC=1 : "0001 111"
    (0b0001111,       7, (1, 0)),
    // T1=1, TC=1 : "01"
    (0b01,            2, (1, 1)),
    // T1=0, TC=2 : "0001 110"
    (0b0001110,       7, (2, 0)),
    // T1=1, TC=2 : "0001 101"
    (0b0001101,       7, (2, 1)),
    // T1=2, TC=2 : "001"
    (0b001,           3, (2, 2)),
    // T1=0, TC=3 : "0000 0011 1"
    (0b000000111,     9, (3, 0)),
    // T1=1, TC=3 : "0001 100"
    (0b0001100,       7, (3, 1)),
    // T1=2, TC=3 : "0001 011"
    (0b0001011,       7, (3, 2)),
    // T1=3, TC=3 : "0000 1"
    (0b00001,         5, (3, 3)),
    // T1=0, TC=4 : "0000 0011 0"
    (0b000000110,     9, (4, 0)),
    // T1=1, TC=4 : "0000 0010 1"
    (0b000000101,     9, (4, 1)),
    // T1=2, TC=4 : "0001 010"
    (0b0001010,       7, (4, 2)),
    // T1=3, TC=4 : "0000 01"
    (0b000001,        6, (4, 3)),
    // T1=0, TC=5 : "0000 0001 11"
    (0b0000000111,    10, (5, 0)),
    // T1=1, TC=5 : "0000 0001 10"
    (0b0000000110,    10, (5, 1)),
    // T1=2, TC=5 : "0000 0010 0"
    (0b000000100,     9, (5, 2)),
    // T1=3, TC=5 : "0001 001"
    (0b0001001,       7, (5, 3)),
    // T1=0, TC=6 : "0000 0000 111"
    (0b00000000111,   11, (6, 0)),
    // T1=1, TC=6 : "0000 0000 110"
    (0b00000000110,   11, (6, 1)),
    // T1=2, TC=6 : "0000 0001 01"
    (0b0000000101,    10, (6, 2)),
    // T1=3, TC=6 : "0001 000"
    (0b0001000,       7, (6, 3)),
    // T1=0, TC=7 : "0000 0000 0111"
    (0b000000000111,  12, (7, 0)),
    // T1=1, TC=7 : "0000 0000 0110"
    (0b000000000110,  12, (7, 1)),
    // T1=2, TC=7 : "0000 0000 101"
    (0b00000000101,   11, (7, 2)),
    // T1=3, TC=7 : "0000 0001 00"
    (0b0000000100,    10, (7, 3)),
    // T1=0, TC=8 : "0000 0000 0011 1"
    (0b0000000000111, 13, (8, 0)),
    // T1=1, TC=8 : "0000 0000 0101"
    (0b000000000101,  12, (8, 1)),
    // T1=2, TC=8 : "0000 0000 0100"
    (0b000000000100,  12, (8, 2)),
    // T1=3, TC=8 : "0000 0000 100"
    (0b00000000100,   11, (8, 3)),
];

// ------------------------------------------------------------------
// §9.2.3 — Table 9-7, 9-8, 9-9 rows (total_zeros)
// ------------------------------------------------------------------

/// Table type selector for total_zeros (§9.2.3).
#[derive(Debug, Clone, Copy)]
pub enum TotalZerosTable {
    /// Tables 9-7 and 9-8, for 4x4 blocks (maxNumCoeff ∉ {4, 8}).
    Luma,
    /// Table 9-9(a), for chroma DC 2x2 (maxNumCoeff = 4).
    ChromaDc420,
    /// Table 9-9(b), for chroma DC 2x4 (maxNumCoeff = 8).
    ChromaDc422,
}

// §9.2.3 — Tables 9-7 and 9-8 combined.
// Each inner slice is indexed by `tzVlcIndex = total_coeff`, 1..=15.
// Entries are (bits, length, total_zeros).

// tzVlcIndex = 1
#[rustfmt::skip]
static TZ_LUMA_TZVLC_1: [(u32, u8, u32); 16] = [
    (0b1,          1,  0),
    (0b011,        3,  1),
    (0b010,        3,  2),
    (0b0011,       4,  3),
    (0b0010,       4,  4),
    (0b00011,      5,  5),
    (0b00010,      5,  6),
    (0b000011,     6,  7),
    (0b000010,     6,  8),
    (0b0000011,    7,  9),
    (0b0000010,    7, 10),
    (0b00000011,   8, 11),
    (0b00000010,   8, 12),
    (0b000000011,  9, 13),
    (0b000000010,  9, 14),
    (0b000000001,  9, 15),
];

// tzVlcIndex = 2
#[rustfmt::skip]
static TZ_LUMA_TZVLC_2: [(u32, u8, u32); 15] = [
    (0b111,      3,  0),
    (0b110,      3,  1),
    (0b101,      3,  2),
    (0b100,      3,  3),
    (0b011,      3,  4),
    (0b0101,     4,  5),
    (0b0100,     4,  6),
    (0b0011,     4,  7),
    (0b0010,     4,  8),
    (0b00011,    5,  9),
    (0b00010,    5, 10),
    (0b000011,   6, 11),
    (0b000010,   6, 12),
    (0b000001,   6, 13),
    (0b000000,   6, 14),
];

// tzVlcIndex = 3
#[rustfmt::skip]
static TZ_LUMA_TZVLC_3: [(u32, u8, u32); 14] = [
    (0b0101,   4,  0),
    (0b111,    3,  1),
    (0b110,    3,  2),
    (0b101,    3,  3),
    (0b0100,   4,  4),
    (0b0011,   4,  5),
    (0b100,    3,  6),
    (0b011,    3,  7),
    (0b0010,   4,  8),
    (0b00011,  5,  9),
    (0b00010,  5, 10),
    (0b000001, 6, 11),
    (0b00001,  5, 12),
    (0b000000, 6, 13),
];

// tzVlcIndex = 4
#[rustfmt::skip]
static TZ_LUMA_TZVLC_4: [(u32, u8, u32); 13] = [
    (0b00011, 5,  0),
    (0b111,   3,  1),
    (0b0101,  4,  2),
    (0b0100,  4,  3),
    (0b110,   3,  4),
    (0b101,   3,  5),
    (0b100,   3,  6),
    (0b0011,  4,  7),
    (0b011,   3,  8),
    (0b0010,  4,  9),
    (0b00010, 5, 10),
    (0b00001, 5, 11),
    (0b00000, 5, 12),
];

// tzVlcIndex = 5
#[rustfmt::skip]
static TZ_LUMA_TZVLC_5: [(u32, u8, u32); 12] = [
    (0b0101,  4,  0),
    (0b0100,  4,  1),
    (0b0011,  4,  2),
    (0b111,   3,  3),
    (0b110,   3,  4),
    (0b101,   3,  5),
    (0b100,   3,  6),
    (0b011,   3,  7),
    (0b0010,  4,  8),
    (0b00001, 5,  9),
    (0b0001,  4, 10),
    (0b00000, 5, 11),
];

// tzVlcIndex = 6
#[rustfmt::skip]
static TZ_LUMA_TZVLC_6: [(u32, u8, u32); 11] = [
    (0b000001, 6,  0),
    (0b00001,  5,  1),
    (0b111,    3,  2),
    (0b110,    3,  3),
    (0b101,    3,  4),
    (0b100,    3,  5),
    (0b011,    3,  6),
    (0b010,    3,  7),
    (0b0001,   4,  8),
    (0b001,    3,  9),
    (0b000000, 6, 10),
];

// tzVlcIndex = 7
#[rustfmt::skip]
static TZ_LUMA_TZVLC_7: [(u32, u8, u32); 10] = [
    (0b000001,  6, 0),
    (0b00001,   5, 1),
    (0b101,     3, 2),
    (0b100,     3, 3),
    (0b011,     3, 4),
    (0b11,      2, 5),
    (0b010,     3, 6),
    (0b0001,    4, 7),
    (0b001,     3, 8),
    (0b000000,  6, 9),
];

// tzVlcIndex = 8
#[rustfmt::skip]
static TZ_LUMA_TZVLC_8: [(u32, u8, u32); 9] = [
    (0b000001, 6, 0),
    (0b0001,   4, 1),
    (0b00001,  5, 2),
    (0b011,    3, 3),
    (0b11,     2, 4),
    (0b10,     2, 5),
    (0b010,    3, 6),
    (0b001,    3, 7),
    (0b000000, 6, 8),
];

// tzVlcIndex = 9
#[rustfmt::skip]
static TZ_LUMA_TZVLC_9: [(u32, u8, u32); 8] = [
    (0b000001, 6, 0),
    (0b000000, 6, 1),
    (0b0001,   4, 2),
    (0b11,     2, 3),
    (0b10,     2, 4),
    (0b001,    3, 5),
    (0b01,     2, 6),
    (0b00001,  5, 7),
];

// tzVlcIndex = 10
#[rustfmt::skip]
static TZ_LUMA_TZVLC_10: [(u32, u8, u32); 7] = [
    (0b00001, 5, 0),
    (0b00000, 5, 1),
    (0b001,   3, 2),
    (0b11,    2, 3),
    (0b10,    2, 4),
    (0b01,    2, 5),
    (0b0001,  4, 6),
];

// tzVlcIndex = 11
#[rustfmt::skip]
static TZ_LUMA_TZVLC_11: [(u32, u8, u32); 6] = [
    (0b0000, 4, 0),
    (0b0001, 4, 1),
    (0b001,  3, 2),
    (0b010,  3, 3),
    (0b1,    1, 4),
    (0b011,  3, 5),
];

// tzVlcIndex = 12
#[rustfmt::skip]
static TZ_LUMA_TZVLC_12: [(u32, u8, u32); 5] = [
    (0b0000, 4, 0),
    (0b0001, 4, 1),
    (0b01,   2, 2),
    (0b1,    1, 3),
    (0b001,  3, 4),
];

// tzVlcIndex = 13
#[rustfmt::skip]
static TZ_LUMA_TZVLC_13: [(u32, u8, u32); 4] = [
    (0b000, 3, 0),
    (0b001, 3, 1),
    (0b1,   1, 2),
    (0b01,  2, 3),
];

// tzVlcIndex = 14
#[rustfmt::skip]
static TZ_LUMA_TZVLC_14: [(u32, u8, u32); 3] = [
    (0b00, 2, 0),
    (0b01, 2, 1),
    (0b1,  1, 2),
];

// tzVlcIndex = 15
#[rustfmt::skip]
static TZ_LUMA_TZVLC_15: [(u32, u8, u32); 2] = [
    (0b0, 1, 0),
    (0b1, 1, 1),
];

// §9.2.3 — Table 9-9(a). Chroma DC 2x2. tzVlcIndex = total_coeff, 1..=3.
#[rustfmt::skip]
static TZ_CHROMA_420_TZVLC_1: [(u32, u8, u32); 4] = [
    (0b1,   1, 0),
    (0b01,  2, 1),
    (0b001, 3, 2),
    (0b000, 3, 3),
];

#[rustfmt::skip]
static TZ_CHROMA_420_TZVLC_2: [(u32, u8, u32); 3] = [
    (0b1,  1, 0),
    (0b01, 2, 1),
    (0b00, 2, 2),
];

#[rustfmt::skip]
static TZ_CHROMA_420_TZVLC_3: [(u32, u8, u32); 2] = [
    (0b1, 1, 0),
    (0b0, 1, 1),
];

// §9.2.3 — Table 9-9(b). Chroma DC 2x4. tzVlcIndex = total_coeff, 1..=7.
#[rustfmt::skip]
static TZ_CHROMA_422_TZVLC_1: [(u32, u8, u32); 8] = [
    (0b1,      1, 0),
    (0b010,    3, 1),
    (0b011,    3, 2),
    (0b0010,   4, 3),
    (0b0011,   4, 4),
    (0b0001,   4, 5),
    (0b00001,  5, 6),
    (0b00000,  5, 7),
];

#[rustfmt::skip]
static TZ_CHROMA_422_TZVLC_2: [(u32, u8, u32); 7] = [
    (0b000, 3, 0),
    (0b01,  2, 1),
    (0b001, 3, 2),
    (0b100, 3, 3),
    (0b101, 3, 4),
    (0b110, 3, 5),
    (0b111, 3, 6),
];

#[rustfmt::skip]
static TZ_CHROMA_422_TZVLC_3: [(u32, u8, u32); 6] = [
    (0b000, 3, 0),
    (0b001, 3, 1),
    (0b01,  2, 2),
    (0b10,  2, 3),
    (0b110, 3, 4),
    (0b111, 3, 5),
];

#[rustfmt::skip]
static TZ_CHROMA_422_TZVLC_4: [(u32, u8, u32); 5] = [
    (0b110, 3, 0),
    (0b00,  2, 1),
    (0b01,  2, 2),
    (0b10,  2, 3),
    (0b111, 3, 4),
];

#[rustfmt::skip]
static TZ_CHROMA_422_TZVLC_5: [(u32, u8, u32); 4] = [
    (0b00, 2, 0),
    (0b01, 2, 1),
    (0b10, 2, 2),
    (0b11, 2, 3),
];

#[rustfmt::skip]
static TZ_CHROMA_422_TZVLC_6: [(u32, u8, u32); 3] = [
    (0b00, 2, 0),
    (0b01, 2, 1),
    (0b1,  1, 2),
];

#[rustfmt::skip]
static TZ_CHROMA_422_TZVLC_7: [(u32, u8, u32); 2] = [
    (0b0, 1, 0),
    (0b1, 1, 1),
];

fn tz_luma_table(tzvlc_index: u32) -> Option<&'static [(u32, u8, u32)]> {
    Some(match tzvlc_index {
        1 => &TZ_LUMA_TZVLC_1,
        2 => &TZ_LUMA_TZVLC_2,
        3 => &TZ_LUMA_TZVLC_3,
        4 => &TZ_LUMA_TZVLC_4,
        5 => &TZ_LUMA_TZVLC_5,
        6 => &TZ_LUMA_TZVLC_6,
        7 => &TZ_LUMA_TZVLC_7,
        8 => &TZ_LUMA_TZVLC_8,
        9 => &TZ_LUMA_TZVLC_9,
        10 => &TZ_LUMA_TZVLC_10,
        11 => &TZ_LUMA_TZVLC_11,
        12 => &TZ_LUMA_TZVLC_12,
        13 => &TZ_LUMA_TZVLC_13,
        14 => &TZ_LUMA_TZVLC_14,
        15 => &TZ_LUMA_TZVLC_15,
        _ => return None,
    })
}

fn tz_chroma_420_table(tzvlc_index: u32) -> Option<&'static [(u32, u8, u32)]> {
    Some(match tzvlc_index {
        1 => &TZ_CHROMA_420_TZVLC_1,
        2 => &TZ_CHROMA_420_TZVLC_2,
        3 => &TZ_CHROMA_420_TZVLC_3,
        _ => return None,
    })
}

fn tz_chroma_422_table(tzvlc_index: u32) -> Option<&'static [(u32, u8, u32)]> {
    Some(match tzvlc_index {
        1 => &TZ_CHROMA_422_TZVLC_1,
        2 => &TZ_CHROMA_422_TZVLC_2,
        3 => &TZ_CHROMA_422_TZVLC_3,
        4 => &TZ_CHROMA_422_TZVLC_4,
        5 => &TZ_CHROMA_422_TZVLC_5,
        6 => &TZ_CHROMA_422_TZVLC_6,
        7 => &TZ_CHROMA_422_TZVLC_7,
        _ => return None,
    })
}

// ------------------------------------------------------------------
// §9.2.3 — Table 9-10 rows (run_before)
// ------------------------------------------------------------------

// §9.2.3 Table 9-10 column "zerosLeft = 1". 1-bit unary.
#[rustfmt::skip]
static RB_ZL_1: [(u32, u8, u32); 2] = [
    (0b1, 1, 0),  // §9.2.3 Table 9-10(zl=1): run_before=0 → "1"
    (0b0, 1, 1),  // §9.2.3 Table 9-10(zl=1): run_before=1 → "0"
];

// §9.2.3 Table 9-10 column "zerosLeft = 2".
#[rustfmt::skip]
static RB_ZL_2: [(u32, u8, u32); 3] = [
    (0b1,  1, 0),  // §9.2.3 Table 9-10(zl=2): run_before=0 → "1"
    (0b01, 2, 1),  // §9.2.3 Table 9-10(zl=2): run_before=1 → "01"
    (0b00, 2, 2),  // §9.2.3 Table 9-10(zl=2): run_before=2 → "00"
];

// §9.2.3 Table 9-10 column "zerosLeft = 3".
#[rustfmt::skip]
static RB_ZL_3: [(u32, u8, u32); 4] = [
    (0b11, 2, 0),  // §9.2.3 Table 9-10(zl=3): run_before=0 → "11"
    (0b10, 2, 1),  // §9.2.3 Table 9-10(zl=3): run_before=1 → "10"
    (0b01, 2, 2),  // §9.2.3 Table 9-10(zl=3): run_before=2 → "01"
    (0b00, 2, 3),  // §9.2.3 Table 9-10(zl=3): run_before=3 → "00"
];

// §9.2.3 Table 9-10 column "zerosLeft = 4".
#[rustfmt::skip]
static RB_ZL_4: [(u32, u8, u32); 5] = [
    (0b11,   2, 0),  // §9.2.3 Table 9-10(zl=4): run_before=0 → "11"
    (0b10,   2, 1),  // §9.2.3 Table 9-10(zl=4): run_before=1 → "10"
    (0b01,   2, 2),  // §9.2.3 Table 9-10(zl=4): run_before=2 → "01"
    (0b001,  3, 3),  // §9.2.3 Table 9-10(zl=4): run_before=3 → "001"
    (0b000,  3, 4),  // §9.2.3 Table 9-10(zl=4): run_before=4 → "000"
];

// §9.2.3 Table 9-10 column "zerosLeft = 5".
#[rustfmt::skip]
static RB_ZL_5: [(u32, u8, u32); 6] = [
    (0b11,   2, 0),  // §9.2.3 Table 9-10(zl=5): run_before=0 → "11"
    (0b10,   2, 1),  // §9.2.3 Table 9-10(zl=5): run_before=1 → "10"
    (0b011,  3, 2),  // §9.2.3 Table 9-10(zl=5): run_before=2 → "011"
    (0b010,  3, 3),  // §9.2.3 Table 9-10(zl=5): run_before=3 → "010"
    (0b001,  3, 4),  // §9.2.3 Table 9-10(zl=5): run_before=4 → "001"
    (0b000,  3, 5),  // §9.2.3 Table 9-10(zl=5): run_before=5 → "000"
];

// §9.2.3 Table 9-10 column "zerosLeft = 6".
// Note the non-monotone assignment for run_before=1..=6: the spec
// explicitly orders these as 1→000, 2→001, 3→011, 4→010, 5→101, 6→100.
#[rustfmt::skip]
static RB_ZL_6: [(u32, u8, u32); 7] = [
    (0b11,   2, 0),  // §9.2.3 Table 9-10(zl=6): run_before=0 → "11"
    (0b000,  3, 1),  // §9.2.3 Table 9-10(zl=6): run_before=1 → "000"
    (0b001,  3, 2),  // §9.2.3 Table 9-10(zl=6): run_before=2 → "001"
    (0b011,  3, 3),  // §9.2.3 Table 9-10(zl=6): run_before=3 → "011"
    (0b010,  3, 4),  // §9.2.3 Table 9-10(zl=6): run_before=4 → "010"
    (0b101,  3, 5),  // §9.2.3 Table 9-10(zl=6): run_before=5 → "101"
    (0b100,  3, 6),  // §9.2.3 Table 9-10(zl=6): run_before=6 → "100"
];

// §9.2.3 Table 9-10 column "zerosLeft > 6" (the long-form table).
// Transcribed verbatim from the spec (Rec. ITU-T H.264 (08/2024), p. 226).
//
//   run_before  codeword
//   0           111
//   1           110
//   2           101
//   3           100
//   4           011
//   5           010
//   6           001
//   7           0001
//   8           0000 1            (4 zeros, then 1)
//   9           0000 01           (4 zeros, then 01)
//   10          0000 001          (4 zeros, then 001)
//   11          0000 0001         (7 zeros, then 1)
//   12          0000 0000 1       (8 zeros, then 1)
//   13          0000 0000 01      (9 zeros, then 1)
//   14          0000 0000 001     (10 zeros, then 1)
//
// NOTE on run_before ∈ 8..=10: the spec writes these as "0000 1",
// "0000 01", "0000 001" — i.e. a fixed 4-zero prefix followed by an
// ever-lengthening suffix that re-enters unary territory. That's
// equivalent to the continuous-unary forms "00001", "000001", "0000001"
// for lengths 5/6/7 — the total bit-count and bit pattern are the same,
// so we store them as contiguous integer literals.
#[rustfmt::skip]
static RB_ZL_GT_6: [(u32, u8, u32); 15] = [
    (0b111,              3,  0),  // §9.2.3 Table 9-10(>6): run_before=0 → "111"
    (0b110,              3,  1),  // §9.2.3 Table 9-10(>6): run_before=1 → "110"
    (0b101,              3,  2),  // §9.2.3 Table 9-10(>6): run_before=2 → "101"
    (0b100,              3,  3),  // §9.2.3 Table 9-10(>6): run_before=3 → "100"
    (0b011,              3,  4),  // §9.2.3 Table 9-10(>6): run_before=4 → "011"
    (0b010,              3,  5),  // §9.2.3 Table 9-10(>6): run_before=5 → "010"
    (0b001,              3,  6),  // §9.2.3 Table 9-10(>6): run_before=6 → "001"
    (0b0001,             4,  7),  // §9.2.3 Table 9-10(>6): run_before=7 → "0001"
    (0b00001,            5,  8),  // §9.2.3 Table 9-10(>6): run_before=8 → "0000 1"
    (0b000001,           6,  9),  // §9.2.3 Table 9-10(>6): run_before=9 → "0000 01"
    (0b0000001,          7, 10),  // §9.2.3 Table 9-10(>6): run_before=10 → "0000 001"
    (0b00000001,         8, 11),  // §9.2.3 Table 9-10(>6): run_before=11 → "0000 0001"
    (0b000000001,        9, 12),  // §9.2.3 Table 9-10(>6): run_before=12 → "0000 0000 1"
    (0b0000000001,      10, 13),  // §9.2.3 Table 9-10(>6): run_before=13 → "0000 0000 01"
    (0b00000000001,     11, 14),  // §9.2.3 Table 9-10(>6): run_before=14 → "0000 0000 001"
];

fn rb_table(zeros_left: u32) -> &'static [(u32, u8, u32)] {
    match zeros_left {
        1 => &RB_ZL_1,
        2 => &RB_ZL_2,
        3 => &RB_ZL_3,
        4 => &RB_ZL_4,
        5 => &RB_ZL_5,
        6 => &RB_ZL_6,
        _ => &RB_ZL_GT_6, // zeros_left >= 7 uses the long-code column
    }
}

// ------------------------------------------------------------------
// Public API
// ------------------------------------------------------------------

/// §9.2.1 — Decode one `coeff_token` from the bitstream.
/// Returns `(total_coeff, trailing_ones)`.
pub fn decode_coeff_token(
    r: &mut BitReader<'_>,
    ctx: CoeffTokenContext,
) -> CavlcResult<(u32, u32)> {
    let table = ctx.select_table();
    match decode_vlc(r, table)? {
        Some(v) => Ok(v),
        None => Err(CavlcError::UnknownCoeffToken {
            nc: ctx.nc_for_error(),
        }),
    }
}

/// §9.2.2 — Decode one non-trailing-ones level.
///
/// Implements steps 1..11 of the "remaining non-zero level values" loop
/// at §9.2.2: reads `level_prefix` (§9.2.2.1, Eq. 9-4), conditionally
/// reads `level_suffix` (u(v)), computes `levelCode`, and returns the
/// signed `levelVal[i]`.
///
/// `suffix_length` is the caller's current suffixLength (updated after
/// the call per §9.2.2 step 9–10). `is_first_level_after_t1_lt_3` is
/// true only when the index `i` matches `TrailingOnes(coeff_token)` AND
/// `TrailingOnes(coeff_token) < 3` (step 7).
///
/// Returns `(level_val, next_suffix_length)` so the caller doesn't have
/// to re-run the update rules.
pub fn decode_level(
    r: &mut BitReader<'_>,
    suffix_length: u32,
    is_first_level_after_t1_lt_3: bool,
) -> CavlcResult<(i32, u32)> {
    // Step 1: read level_prefix per Eq. 9-4.
    let level_prefix = read_level_prefix(r)?;

    // Step 2: compute levelSuffixSize (§9.2.2, step 2).
    let level_suffix_size: u32 = if level_prefix == 14 && suffix_length == 0 {
        4
    } else if level_prefix >= 15 {
        level_prefix - 3
    } else {
        suffix_length
    };

    // Step 3: read level_suffix if non-zero size, else inferred 0.
    let level_suffix: u32 = if level_suffix_size > 0 {
        r.u(level_suffix_size)?
    } else {
        0
    };

    // Step 4: levelCode = (Min(15, level_prefix) << suffixLength) + level_suffix.
    let mut level_code: i64 =
        ((core::cmp::min(15, level_prefix) as i64) << suffix_length) + level_suffix as i64;

    // Step 5: if level_prefix >= 15 and suffixLength == 0, add 15.
    if level_prefix >= 15 && suffix_length == 0 {
        level_code += 15;
    }
    // Step 6: if level_prefix >= 16, add (1 << (level_prefix - 3)) - 4096.
    if level_prefix >= 16 {
        level_code += (1_i64 << (level_prefix - 3)) - 4096;
    }
    // Step 7: if i == TrailingOnes and TrailingOnes < 3, increment levelCode by 2.
    if is_first_level_after_t1_lt_3 {
        level_code += 2;
    }

    // Step 8: derive levelVal[i].
    let level_val: i32 = if level_code & 1 == 0 {
        ((level_code + 2) >> 1) as i32
    } else {
        ((-level_code - 1) >> 1) as i32
    };

    // Steps 9–10: update suffix_length for the next iteration.
    let mut next_sl = suffix_length;
    if next_sl == 0 {
        next_sl = 1;
    }
    let abs_level = level_val.unsigned_abs();
    if abs_level > (3u32 << (next_sl - 1)) && next_sl < 6 {
        next_sl += 1;
    }

    Ok((level_val, next_sl))
}

/// §9.2.2.1 — Decode `level_prefix` (Eq. 9-4): count leading zeros then
/// consume the terminating `1`.
fn read_level_prefix(r: &mut BitReader<'_>) -> CavlcResult<u32> {
    let mut leading: u32 = 0;
    loop {
        if leading > 32 {
            return Err(CavlcError::LevelPrefixOverflow);
        }
        if r.u(1)? == 1 {
            return Ok(leading);
        }
        leading += 1;
    }
}

/// §9.2.3 — Decode one `total_zeros` syntax element. `total_coeff` is
/// the caller's `TotalCoeff(coeff_token)` (i.e. `tzVlcIndex`). `table`
/// selects among Tables 9-7/9-8/9-9(a)/9-9(b).
pub fn decode_total_zeros(
    r: &mut BitReader<'_>,
    total_coeff: u32,
    table: TotalZerosTable,
) -> CavlcResult<u32> {
    let rows = match table {
        TotalZerosTable::Luma => tz_luma_table(total_coeff),
        TotalZerosTable::ChromaDc420 => tz_chroma_420_table(total_coeff),
        TotalZerosTable::ChromaDc422 => tz_chroma_422_table(total_coeff),
    }
    .ok_or(CavlcError::UnknownTotalZeros { tc: total_coeff })?;
    match decode_vlc(r, rows)? {
        Some(v) => Ok(v),
        None => Err(CavlcError::UnknownTotalZeros { tc: total_coeff }),
    }
}

/// §9.2.3 — Decode one `run_before` syntax element.
pub fn decode_run_before(r: &mut BitReader<'_>, zeros_left: u32) -> CavlcResult<u32> {
    if zeros_left == 0 {
        // Caller should not invoke decode_run_before when zerosLeft == 0
        // (§9.2.3 handles that case without reading bits), but return 0
        // rather than an error to be forgiving.
        return Ok(0);
    }
    let rows = rb_table(zeros_left);
    match decode_vlc(r, rows)? {
        Some(v) => Ok(v),
        None => Err(CavlcError::UnknownRunBefore { zl: zeros_left }),
    }
}

/// §9.2 + §7.3.5.3.1 — Parse one `residual_block_cavlc(coeffLevel,
/// startIdx, endIdx, maxNumCoeff)` block. Returns the level values for
/// indices `startIdx..=endIdx` in zigzag order.
///
/// Implements the four sub-clauses in order:
///  1. §9.2.1 — read coeff_token → (TotalCoeff, TrailingOnes).
///  2. §9.2.2 — read `trailing_ones_sign_flag`, then per §9.2.2 steps
///     1..11 the remaining non-zero levels, producing `levelVal[]`.
///  3. §9.2.3 — read `total_zeros` (unless TotalCoeff == maxNumCoeff),
///     then `run_before` loop producing `runVal[]`.
///  4. §9.2.4 — combine levelVal + runVal into coeffLevel[startIdx..].
///
/// The chosen total_zeros sub-table is derived from `max_num_coeff`
/// per §9.2.3 (`== 4` ⇒ Table 9-9(a), `== 8` ⇒ Table 9-9(b), else
/// Tables 9-7/9-8). The coeff_token sub-table comes from the caller via
/// `ctx` (since the neighbour-based derivation of `nC` is outside this
/// module's scope — it needs macroblock context).
pub fn parse_residual_block_cavlc(
    r: &mut BitReader<'_>,
    ctx: CoeffTokenContext,
    start_idx: u32,
    end_idx: u32,
    max_num_coeff: u32,
) -> CavlcResult<Vec<i32>> {
    // §7.3.5.3.1 — coeffLevel is sized (endIdx - startIdx + 1).
    let span = (end_idx - start_idx + 1) as usize;
    let mut coeff_level = vec![0i32; span];

    // §9.2.1
    let (total_coeff, trailing_ones) = decode_coeff_token(r, ctx)?;
    if total_coeff == 0 {
        // §9.2 step 3a: all zeros, done.
        return Ok(coeff_level);
    }
    if total_coeff > max_num_coeff {
        return Err(CavlcError::InvalidTotalCoeff {
            tc: total_coeff,
            max: max_num_coeff,
        });
    }
    if trailing_ones > 3 || trailing_ones > total_coeff {
        return Err(CavlcError::InvalidTrailingOnes {
            t1: trailing_ones,
            tc: total_coeff,
        });
    }

    // §9.2.2 — build levelVal[0..total_coeff].
    let mut level_val = vec![0i32; total_coeff as usize];
    for i in 0..trailing_ones {
        let sign = r.u(1)?;
        level_val[i as usize] = if sign == 0 { 1 } else { -1 };
    }
    // Initialise suffixLength per §9.2.2.
    let mut suffix_length: u32 = if total_coeff > 10 && trailing_ones < 3 {
        1
    } else {
        0
    };
    // Remaining non-zero levels.
    for i in trailing_ones..total_coeff {
        let is_first_after = i == trailing_ones && trailing_ones < 3;
        let (lv, next_sl) = decode_level(r, suffix_length, is_first_after)?;
        level_val[i as usize] = lv;
        suffix_length = next_sl;
    }

    // §9.2.3 — runs.
    // Select the total_zeros table from max_num_coeff.
    let tz_table = match max_num_coeff {
        4 => TotalZerosTable::ChromaDc420,
        8 => TotalZerosTable::ChromaDc422,
        _ => TotalZerosTable::Luma,
    };
    let mut run_val = vec![0u32; total_coeff as usize];
    let mut zeros_left: u32 = if total_coeff < max_num_coeff {
        decode_total_zeros(r, total_coeff, tz_table)?
    } else {
        0
    };
    // TotalCoeff - 1 iterations, each reads run_before when zerosLeft > 0.
    for i in 0..total_coeff.saturating_sub(1) {
        let run_before = if zeros_left > 0 {
            decode_run_before(r, zeros_left)?
        } else {
            0
        };
        run_val[i as usize] = run_before;
        // §9.2.3: zerosLeft -= runVal[i]. Must stay ≥ 0.
        if run_before > zeros_left {
            return Err(CavlcError::UnknownRunBefore { zl: zeros_left });
        }
        zeros_left -= run_before;
    }
    // The last runVal entry absorbs whatever zerosLeft remains.
    run_val[(total_coeff - 1) as usize] = zeros_left;

    // §9.2.4 — combine. coeffNum starts at −1 and walks up; we collect
    // (index, value) pairs and place into coeff_level after translating
    // from (startIdx + coeffNum) indexing.
    let mut coeff_num: i32 = -1;
    // Iterate i = total_coeff - 1 down to 0 (§9.2.4).
    for j in 0..total_coeff {
        let i = (total_coeff - 1 - j) as usize;
        coeff_num += (run_val[i] as i32) + 1;
        let dst_idx = start_idx as i32 + coeff_num;
        let rel = (dst_idx - start_idx as i32) as usize;
        if rel >= span {
            return Err(CavlcError::InvalidTotalCoeff {
                tc: total_coeff,
                max: max_num_coeff,
            });
        }
        coeff_level[rel] = level_val[i];
    }
    Ok(coeff_level)
}

// ==================================================================
// Tests
// ==================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a BitReader over a byte slice and optionally skip
    /// `n` leading bits (to position the reader at a custom offset).
    fn reader(bytes: &'static [u8]) -> BitReader<'static> {
        BitReader::new(bytes)
    }

    /// Helper: pack a sequence of bit-strings (as `&str` of '0'/'1')
    /// into a `Vec<u8>` for test input construction. MSB-first within
    /// each byte.
    fn pack_bits(bits: &[&str]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut cur: u8 = 0;
        let mut n: u8 = 0;
        for s in bits {
            for c in s.chars() {
                let bit = match c {
                    '0' => 0,
                    '1' => 1,
                    ' ' | '_' => continue, // allow "0001 11"-style spacing
                    other => panic!("bad bit char: {other:?}"),
                };
                cur = (cur << 1) | bit;
                n += 1;
                if n == 8 {
                    out.push(cur);
                    cur = 0;
                    n = 0;
                }
            }
        }
        if n > 0 {
            cur <<= 8 - n;
            out.push(cur);
        }
        out
    }

    #[test]
    fn pack_bits_matches_expected() {
        // "1010 0110" → 0xA6
        assert_eq!(pack_bits(&["1010 0110"]), vec![0xA6]);
        // "1" padded with 0s → 0x80
        assert_eq!(pack_bits(&["1"]), vec![0x80]);
        // Two tokens concatenated: "01" + "001" = "01001" → 0x48
        assert_eq!(pack_bits(&["01", "001"]), vec![0x48]);
    }

    // ---- coeff_token tests ----

    #[test]
    fn coeff_token_col0_nc_lt_2_samples() {
        // nC = 0 → column 0. Spot-check five distinct rows.
        let cases: &[(&str, (u32, u32))] = &[
            ("1", (0, 0)),
            ("01", (1, 1)),
            ("001", (2, 2)),
            ("0001 1", (3, 3)),
            ("0000 0001 11", (4, 0)),
        ];
        for (bits, want) in cases {
            let bytes = pack_bits(&[bits]);
            let mut r = BitReader::new(&bytes);
            let got = decode_coeff_token(&mut r, CoeffTokenContext::Numeric(0)).unwrap();
            assert_eq!(got, *want, "bits={bits}");
        }
    }

    #[test]
    fn coeff_token_col1_2_le_nc_lt_4_samples() {
        let cases: &[(&str, (u32, u32))] = &[
            ("11", (0, 0)),
            ("10", (1, 1)),
            ("011", (2, 2)),
            ("0101", (3, 3)),
            ("0010 11", (1, 0)),
        ];
        for (bits, want) in cases {
            let bytes = pack_bits(&[bits]);
            let mut r = BitReader::new(&bytes);
            let got = decode_coeff_token(&mut r, CoeffTokenContext::Numeric(2)).unwrap();
            assert_eq!(got, *want, "bits={bits}");
        }
    }

    #[test]
    fn coeff_token_col2_4_le_nc_lt_8_samples() {
        let cases: &[(&str, (u32, u32))] = &[
            ("1111", (0, 0)),
            ("1110", (1, 1)),
            ("1101", (2, 2)),
            ("1100", (3, 3)),
            ("0111 0", (3, 2)),
        ];
        for (bits, want) in cases {
            let bytes = pack_bits(&[bits]);
            let mut r = BitReader::new(&bytes);
            let got = decode_coeff_token(&mut r, CoeffTokenContext::Numeric(5)).unwrap();
            assert_eq!(got, *want, "bits={bits}");
        }
    }

    #[test]
    fn coeff_token_col3_8_le_nc_samples() {
        // Column "8 <= nC" is the 6-bit fixed-length column. Per Table
        // 9-5, the decoded pair is (TotalCoeff, TrailingOnes).
        let cases: &[(&str, (u32, u32))] = &[
            ("0000 11", (0, 0)),  // T1=0, TC=0
            ("0000 00", (1, 0)),  // T1=0, TC=1
            ("0000 01", (1, 1)),  // T1=1, TC=1
            ("0001 00", (2, 0)),  // T1=0, TC=2
            ("1111 11", (16, 3)), // T1=3, TC=16
        ];
        for (bits, want) in cases {
            let bytes = pack_bits(&[bits]);
            let mut r = BitReader::new(&bytes);
            let got = decode_coeff_token(&mut r, CoeffTokenContext::Numeric(10)).unwrap();
            assert_eq!(got, *want, "bits={bits}");
        }
    }

    #[test]
    fn coeff_token_col1_tc10_t1_2_from_spec() {
        // Table 9-5 2<=nC<4 row TrailingOnes=2, TotalCoeff=10: "0000 0000 1101".
        let bytes = pack_bits(&["0000 0000 1101"]);
        let mut r = BitReader::new(&bytes);
        let got = decode_coeff_token(&mut r, CoeffTokenContext::Numeric(3)).unwrap();
        assert_eq!(got, (10, 2));
    }

    /// Regression test: decoding a length-13 codeword from COL1
    /// (T1=3, TC=15, "0000 0000 0000 1"). Prior to the 2026-04-20 fix
    /// the COL1 row had length 15 (two phantom leading zeros), and the
    /// decoder would never match this codeword.
    #[test]
    fn coeff_token_col1_t1_3_tc_15_length_fix() {
        // Pack 13-bit codeword then pad to byte boundary.
        let bytes = pack_bits(&["0000 0000 0000 1"]);
        let mut r = BitReader::new(&bytes);
        let got = decode_coeff_token(&mut r, CoeffTokenContext::Numeric(3)).unwrap();
        assert_eq!(got, (15, 3));
        // Cursor should have advanced exactly 13 bits.
        let (b, bi) = r.position();
        assert_eq!((b, bi), (1, 5));
    }

    /// Regression tests for the 2026-04-20 COL1 11-bit length bug:
    /// prior to this fix, every 11-bit codeword in Table 9-5 column
    /// "2 <= nC < 4" was stored with length=10, which means the VLC
    /// decoder matched a 10-bit prefix too early and desynced the stream.
    /// The entries affected: (T1=0,TC=7), (T1=0..2,TC=8), (T1=1..2,TC=9),
    /// (T1=3,TC=10), (T1=3,TC=11).
    ///
    /// Every case below is transcribed directly from ITU-T H.264 (08/2024)
    /// Table 9-5 `2 <= nC < 4` column.
    #[test]
    fn coeff_token_col1_11bit_length_fix() {
        let cases: &[(&str, (u32, u32))] = &[
            ("0000 0001 111", (7, 0)), // §9.2.1 Table 9-5 col "2<=nC<4"
            ("0000 0001 011", (8, 0)),
            ("0000 0001 110", (8, 1)),
            ("0000 0001 101", (8, 2)),
            ("0000 0001 010", (9, 1)),
            ("0000 0001 001", (9, 2)),
            ("0000 0001 100", (10, 3)),
            ("0000 0001 000", (11, 3)),
        ];
        for (bits, want) in cases {
            let bytes = pack_bits(&[bits]);
            let mut r = BitReader::new(&bytes);
            let got = decode_coeff_token(&mut r, CoeffTokenContext::Numeric(3)).unwrap();
            assert_eq!(got, *want, "bits={bits}");
            // Cursor must have advanced exactly 11 bits.
            let (b, bi) = r.position();
            let consumed = b * 8 + bi as usize;
            assert_eq!(consumed, 11, "wrong bit count consumed for bits={bits}");
        }
    }

    /// Exhaustive check: every COL1 entry decodes round-trip to the
    /// expected (TotalCoeff, TrailingOnes) with exactly the stored length
    /// of bits consumed. Also verifies distinct codewords are all
    /// prefix-free (implied by the VLC decoder always finding a match).
    #[test]
    fn coeff_token_col1_all_entries_roundtrip() {
        for (bits, length, (tc, t1)) in TABLE_9_5_COL1.iter() {
            // Build a packed bit string of exactly `length` bits.
            let s: String = (0..*length)
                .rev()
                .map(|k| if (bits >> k) & 1 == 1 { '1' } else { '0' })
                .collect();
            let bytes = pack_bits(&[&s]);
            let mut r = BitReader::new(&bytes);
            let got = decode_coeff_token(&mut r, CoeffTokenContext::Numeric(2)).unwrap();
            assert_eq!(got, (*tc, *t1), "entry bits={s:?}");
            let (b, bi) = r.position();
            let consumed = b * 8 + bi as usize;
            assert_eq!(consumed, *length as usize, "wrong bit count for bits={s:?}");
        }
    }

    /// Regression test for the 2026-04-20 COL3 swap bug: prior to the
    /// fix the COL3 table stored `(TrailingOnes, TotalCoeff)` instead of
    /// `(TotalCoeff, TrailingOnes)`. These are rows where the two
    /// fields differ so a swap would be visible.
    #[test]
    fn coeff_token_col3_bugfix_regression() {
        // (codeword, expected (TotalCoeff, TrailingOnes))
        let cases: &[(&str, (u32, u32))] = &[
            ("0000 00", (1, 0)), // would be (0,1) if swapped
            ("0001 00", (2, 0)), // would be (0,2) if swapped
            ("0010 00", (3, 0)), // would be (0,3) if swapped
            ("0010 01", (3, 1)), // would be (1,3) if swapped — trigger for "trailing_ones=3 for total_coeff=1"
            ("0011 00", (4, 0)),
            ("1000 00", (9, 0)),
            ("1111 00", (16, 0)),
        ];
        for (bits, want) in cases {
            let bytes = pack_bits(&[bits]);
            let mut r = BitReader::new(&bytes);
            let got = decode_coeff_token(&mut r, CoeffTokenContext::Numeric(10)).unwrap();
            assert_eq!(got, *want, "bits={bits}");
            // Invariant check: TrailingOnes <= TotalCoeff.
            assert!(
                got.1 <= got.0,
                "invariant TrailingOnes<=TotalCoeff violated for bits={bits}"
            );
        }
    }

    #[test]
    fn coeff_token_chroma_dc_420_samples() {
        // Column nC == -1. Table 9-5 rightmost-but-one.
        let cases: &[(&str, (u32, u32))] = &[
            ("01", (0, 0)),
            ("1", (1, 1)),
            ("001", (2, 2)),
            ("0001 11", (1, 0)),
            ("0000 10", (4, 0)),
        ];
        for (bits, want) in cases {
            let bytes = pack_bits(&[bits]);
            let mut r = BitReader::new(&bytes);
            let got = decode_coeff_token(&mut r, CoeffTokenContext::ChromaDc420).unwrap();
            assert_eq!(got, *want, "bits={bits}");
        }
    }

    #[test]
    fn coeff_token_chroma_dc_422_samples() {
        // Column nC == -2.
        let cases: &[(&str, (u32, u32))] = &[
            ("1", (0, 0)),
            ("01", (1, 1)),
            ("001", (2, 2)),
            ("0000 1", (3, 3)),
            ("0001 111", (1, 0)),
        ];
        for (bits, want) in cases {
            let bytes = pack_bits(&[bits]);
            let mut r = BitReader::new(&bytes);
            let got = decode_coeff_token(&mut r, CoeffTokenContext::ChromaDc422).unwrap();
            assert_eq!(got, *want, "bits={bits}");
        }
    }

    // ---- level_prefix + level decode ----

    #[test]
    fn level_prefix_reads_leading_zeros_then_one() {
        // "1"        → 0
        // "01"       → 1
        // "001"      → 2
        // "0001"     → 3
        let cases: &[(&str, u32)] = &[
            ("1", 0),
            ("01", 1),
            ("001", 2),
            ("0001", 3),
            ("00000000 1", 8),
        ];
        for (bits, want) in cases {
            let bytes = pack_bits(&[bits]);
            let mut r = BitReader::new(&bytes);
            assert_eq!(read_level_prefix(&mut r).unwrap(), *want, "bits={bits}");
        }
    }

    #[test]
    fn decode_level_basic_values() {
        // With suffix_length = 0 and the standard level mapping:
        //   level_prefix=1 ("01"), suffixSize=0, levelCode=1 (before
        //   step 7). If is_first_after_t1_lt_3 = true, levelCode = 3
        //   (odd) → levelVal = (-3-1)>>1 = -2. We instead test the
        //   is_first_after_t1_lt_3=false path and observe level=−1:
        //
        //     level_prefix=1, suffix_length=0 → Min(15,1)<<0 = 1 = levelCode
        //     levelCode=1 odd ⇒ (-1-1)>>1 = -1.
        let bytes = pack_bits(&["01"]);
        let mut r = BitReader::new(&bytes);
        let (lv, _next) = decode_level(&mut r, 0, false).unwrap();
        assert_eq!(lv, -1);

        // Same bits but is_first_after=true → levelCode = 1 + 2 = 3 (odd)
        // ⇒ (-3-1)>>1 = -2.
        let bytes = pack_bits(&["01"]);
        let mut r = BitReader::new(&bytes);
        let (lv, _next) = decode_level(&mut r, 0, true).unwrap();
        assert_eq!(lv, -2);

        // level_prefix=0 ("1"), suffix_length=0, is_first_after=true
        // → levelCode = 0 + 2 = 2 (even) ⇒ (2+2)>>1 = 2.
        let bytes = pack_bits(&["1"]);
        let mut r = BitReader::new(&bytes);
        let (lv, next) = decode_level(&mut r, 0, true).unwrap();
        assert_eq!(lv, 2);
        assert_eq!(next, 1); // §9.2.2 step 9 promotes 0 → 1.

        // Positive level=1 path: level_prefix=0, is_first_after=false
        // → levelCode = 0 (even) ⇒ (0+2)>>1 = 1.
        let bytes = pack_bits(&["1"]);
        let mut r = BitReader::new(&bytes);
        let (lv, _next) = decode_level(&mut r, 0, false).unwrap();
        assert_eq!(lv, 1);
    }

    #[test]
    fn decode_level_with_suffix_bits() {
        // suffix_length = 1, level_prefix = 1 ("01"), level_suffix = 1 bit,
        // level_suffix = 1 → levelCode = (Min(15,1) << 1) + 1 = 3, odd
        // ⇒ levelVal = (-3-1)>>1 = -2.
        let bytes = pack_bits(&["01", "1"]);
        let mut r = BitReader::new(&bytes);
        let (lv, _) = decode_level(&mut r, 1, false).unwrap();
        assert_eq!(lv, -2);

        // suffix_length = 1, level_prefix = 0 ("1"), level_suffix = 0 bit,
        // level_suffix = 0 → levelCode = 0 (even) ⇒ +2/2 = 1.
        let bytes = pack_bits(&["1", "0"]);
        let mut r = BitReader::new(&bytes);
        let (lv, _) = decode_level(&mut r, 1, false).unwrap();
        assert_eq!(lv, 1);
    }

    /// Edge-case: suffix_length == 0, level_prefix == 14. §9.2.2 step 2
    /// requires levelSuffixSize = 4 in this case (not suffix_length=0),
    /// so level_suffix is a 4-bit u(v).
    #[test]
    fn decode_level_prefix14_suffix0_reads_4_bit_suffix() {
        // level_prefix = 14 ⇒ 14 zeros then a 1 ⇒ 15-bit unary "0000 0000 0000 001"
        // followed by a 4-bit suffix, say 0b1010 = 10.
        // levelCode = (Min(15,14) << 0) + 10 = 14 + 10 = 24 (even).
        // is_first_after_t1_lt_3 = false → no +2.
        // levelVal = (24 + 2) >> 1 = 13.
        let bits = "00000000000000 1 1010"; // 14 zeros + 1 + 4-bit suffix
        let bytes = pack_bits(&[bits]);
        let mut r = BitReader::new(&bytes);
        let (lv, next) = decode_level(&mut r, 0, false).unwrap();
        assert_eq!(lv, 13);
        // After this: step 9 promotes 0→1, step 10 checks abs(13) > (3<<0)=3, true and 1<6 → 2.
        assert_eq!(next, 2);
    }

    /// §9.2.2 step 10: when `|levelVal| > (3 << (suffixLength − 1))` and
    /// suffixLength < 6, suffixLength must be incremented. Verify the
    /// boundary: start at suffixLength = 1, decode a small level that
    /// does NOT trigger the bump, then one that does.
    #[test]
    fn decode_level_step10_triggers_suffix_length_bump() {
        // suffix_length = 1, level_prefix = 0 ("1"), level_suffix=0 (1 bit).
        // levelCode = 0, levelVal = (0+2)>>1 = 1. |1| > (3<<0)=3? No.
        // Expected: next_sl stays 1.
        let bytes = pack_bits(&["1", "0"]);
        let mut r = BitReader::new(&bytes);
        let (lv, next) = decode_level(&mut r, 1, false).unwrap();
        assert_eq!(lv, 1);
        assert_eq!(next, 1);

        // suffix_length = 1, level_prefix = 2 ("001"), level_suffix = 0 (1 bit).
        // levelCode = (Min(15,2)<<1) + 0 = 4. Even, levelVal = (4+2)>>1 = 3.
        // |3| > (3<<0)=3? No (strictly greater).
        let bytes = pack_bits(&["001", "0"]);
        let mut r = BitReader::new(&bytes);
        let (lv, next) = decode_level(&mut r, 1, false).unwrap();
        assert_eq!(lv, 3);
        assert_eq!(next, 1);

        // suffix_length = 1, level_prefix = 2 ("001"), level_suffix = 1 (1 bit).
        // levelCode = (2<<1)+1 = 5. Odd, levelVal = (-5-1)>>1 = -3. |-3|=3, not >3.
        let bytes = pack_bits(&["001", "1"]);
        let mut r = BitReader::new(&bytes);
        let (lv, next) = decode_level(&mut r, 1, false).unwrap();
        assert_eq!(lv, -3);
        assert_eq!(next, 1);

        // suffix_length = 1, level_prefix = 3 ("0001"), level_suffix = 0.
        // levelCode = (3<<1) + 0 = 6 even, levelVal = 4. |4| > 3 ⇒ next_sl = 2.
        let bytes = pack_bits(&["0001", "0"]);
        let mut r = BitReader::new(&bytes);
        let (lv, next) = decode_level(&mut r, 1, false).unwrap();
        assert_eq!(lv, 4);
        assert_eq!(next, 2);
    }

    /// §9.2.2 step 10: suffixLength caps at 6. Starting at 5, a big level
    /// should bump to 6. Starting at 6, it must stay 6.
    #[test]
    fn decode_level_step10_cap_at_6() {
        // suffix_length = 5. level_prefix=0, level_suffix=0 (5 bits).
        // levelCode = 0, levelVal = 1. |1| > (3<<4)=48? No.
        // Stays at 5.
        let bytes = pack_bits(&["1", "00000"]);
        let mut r = BitReader::new(&bytes);
        let (lv, next) = decode_level(&mut r, 5, false).unwrap();
        assert_eq!(lv, 1);
        assert_eq!(next, 5);

        // suffix_length = 5, level_prefix=3 ("0001"), level_suffix=11111 (=31).
        // levelCode = (3<<5)+31 = 96+31 = 127, odd. levelVal = (-127-1)>>1 = -64.
        // |-64| > (3<<4)=48? Yes. next_sl++ → 6.
        let bytes = pack_bits(&["0001", "11111"]);
        let mut r = BitReader::new(&bytes);
        let (lv, next) = decode_level(&mut r, 5, false).unwrap();
        assert_eq!(lv, -64);
        assert_eq!(next, 6);

        // suffix_length = 6, level_prefix=3 ("0001"), level_suffix = 111111 (63).
        // levelCode = (3<<6) + 63 = 192+63 = 255, odd. |-(−255-1)>>1| = 128.
        // |128| > (3<<5)=96? Yes. But suffixLength already 6 → no bump.
        let bytes = pack_bits(&["0001", "111111"]);
        let mut r = BitReader::new(&bytes);
        let (lv, next) = decode_level(&mut r, 6, false).unwrap();
        assert_eq!(lv, -128);
        assert_eq!(next, 6);
    }

    /// §9.2.2 step 5: when level_prefix >= 15 AND suffixLength == 0,
    /// levelCode gets +15. With level_prefix = 15 and suffix = 12 bits,
    /// levelCode = (15<<0) + level_suffix + 15 = 30 + level_suffix.
    /// (Baseline/Extended profile: level_prefix maxes at 15, so this
    /// is the escape path for large levels on those profiles.)
    #[test]
    fn decode_level_prefix15_suffix0_escape() {
        // level_prefix = 15 ⇒ 15 zeros + 1 = 16 bits unary.
        // level_suffix = 12 bits of 000000000010 (=2).
        // levelCode = (Min(15,15)<<0) + 2 = 17. Step 5 adds 15 → 32. Even.
        // levelVal = (32+2)>>1 = 17.
        let bits = "0000000000000001 000000000010";
        let bytes = pack_bits(&[bits]);
        let mut r = BitReader::new(&bytes);
        let (lv, next) = decode_level(&mut r, 0, false).unwrap();
        assert_eq!(lv, 17);
        // Step 9 promotes 0→1. |17| > 3? Yes → bump to 2.
        assert_eq!(next, 2);
    }

    /// §9.2.2 step 6: level_prefix >= 16. This is not valid for
    /// Baseline/Constrained/Main/Extended profiles but is allowed for
    /// High. Verify the (1<<(lp-3))-4096 correction.
    #[test]
    fn decode_level_prefix16_step6_correction() {
        // level_prefix = 16 ⇒ 16 zeros + 1 = 17 bits unary.
        // levelSuffixSize = level_prefix - 3 = 13. level_suffix = 0.
        // levelCode = (Min(15,16)<<0) + 0 = 15.
        // Step 5 (lp>=15 && sl==0): +15 → 30.
        // Step 6 (lp>=16): +(1<<(16-3))-4096 = 8192-4096 = 4096 → 4126.
        // Even, levelVal = (4126+2)>>1 = 2064.
        let bits = "0000000000000000 1 0000000000000"; // 16 zeros + 1 + 13-bit 0 suffix
        let bytes = pack_bits(&[bits]);
        let mut r = BitReader::new(&bytes);
        let (lv, _next) = decode_level(&mut r, 0, false).unwrap();
        assert_eq!(lv, 2064);
    }

    /// §9.2.2 step 7: when `i == TrailingOnes && TrailingOnes < 3`,
    /// the levelCode is incremented by 2 BEFORE the even/odd mapping.
    /// This biases the first non-trailing-one level away from ±1
    /// (already used by trailing ones). Verify the bias against hand
    /// computation.
    #[test]
    fn decode_level_step7_first_after_bias() {
        // suffix_length = 0, level_prefix = 0 ("1"), not first_after.
        // levelCode=0, levelVal=1.
        let bytes = pack_bits(&["1"]);
        let mut r = BitReader::new(&bytes);
        let (lv, _) = decode_level(&mut r, 0, false).unwrap();
        assert_eq!(lv, 1);

        // Same with first_after=true. levelCode=0+2=2, levelVal=(2+2)>>1=2.
        let bytes = pack_bits(&["1"]);
        let mut r = BitReader::new(&bytes);
        let (lv, _) = decode_level(&mut r, 0, true).unwrap();
        assert_eq!(lv, 2);

        // level_prefix=1 ("01"), not first_after: levelCode=1, odd, levelVal=-1.
        let bytes = pack_bits(&["01"]);
        let mut r = BitReader::new(&bytes);
        let (lv, _) = decode_level(&mut r, 0, false).unwrap();
        assert_eq!(lv, -1);

        // Same with first_after=true. levelCode=1+2=3, odd, levelVal=(-3-1)>>1=-2.
        let bytes = pack_bits(&["01"]);
        let mut r = BitReader::new(&bytes);
        let (lv, _) = decode_level(&mut r, 0, true).unwrap();
        assert_eq!(lv, -2);
    }

    /// The "suffix_length bumps from 0 to 1 on entry" corner (§9.2.2
    /// step 9). Verify return value.
    #[test]
    fn decode_level_step9_always_promotes_0_to_1() {
        // Small level encoded with suffix_length = 0. step 9 must push
        // next_sl to 1. Step 10 then considers abs(lv) > 3.
        let bytes = pack_bits(&["1"]);
        let mut r = BitReader::new(&bytes);
        let (_, next) = decode_level(&mut r, 0, false).unwrap();
        assert_eq!(next, 1);
    }

    // ---- total_zeros ----

    #[test]
    fn total_zeros_luma_tzvlc_1_samples() {
        // Table 9-7 column tzVlcIndex=1.
        let cases: &[(&str, u32)] = &[("1", 0), ("011", 1), ("010", 2), ("0011", 3)];
        for (bits, want) in cases {
            let bytes = pack_bits(&[bits]);
            let mut r = BitReader::new(&bytes);
            let got = decode_total_zeros(&mut r, 1, TotalZerosTable::Luma).unwrap();
            assert_eq!(got, *want, "bits={bits}");
        }
    }

    #[test]
    fn total_zeros_luma_tzvlc_7_samples() {
        // Table 9-7 column tzVlcIndex=7.
        let cases: &[(&str, u32)] = &[
            ("0000 01", 0),
            ("0000 1", 1),
            ("101", 2),
            ("11", 5),
            ("0000 00", 9),
        ];
        for (bits, want) in cases {
            let bytes = pack_bits(&[bits]);
            let mut r = BitReader::new(&bytes);
            let got = decode_total_zeros(&mut r, 7, TotalZerosTable::Luma).unwrap();
            assert_eq!(got, *want, "bits={bits}");
        }
    }

    #[test]
    fn total_zeros_luma_tzvlc_15_samples() {
        // Table 9-8 column tzVlcIndex=15: just 1 bit — 0 or 1.
        let cases: &[(&str, u32)] = &[("0", 0), ("1", 1)];
        for (bits, want) in cases {
            let bytes = pack_bits(&[bits]);
            let mut r = BitReader::new(&bytes);
            let got = decode_total_zeros(&mut r, 15, TotalZerosTable::Luma).unwrap();
            assert_eq!(got, *want, "bits={bits}");
        }
    }

    #[test]
    fn total_zeros_chroma_420_samples() {
        // Table 9-9(a).
        let cases: &[(u32, &str, u32)] = &[
            (1, "1", 0),
            (1, "01", 1),
            (1, "001", 2),
            (1, "000", 3),
            (2, "1", 0),
            (2, "00", 2),
            (3, "1", 0),
            (3, "0", 1),
        ];
        for (tc, bits, want) in cases {
            let bytes = pack_bits(&[bits]);
            let mut r = BitReader::new(&bytes);
            let got = decode_total_zeros(&mut r, *tc, TotalZerosTable::ChromaDc420).unwrap();
            assert_eq!(got, *want, "tc={tc} bits={bits}");
        }
    }

    #[test]
    fn total_zeros_chroma_422_samples() {
        // Table 9-9(b).
        let cases: &[(u32, &str, u32)] = &[
            (1, "1", 0),
            (1, "010", 1),
            (1, "0001", 5),
            (4, "110", 0),
            (7, "0", 0),
            (7, "1", 1),
        ];
        for (tc, bits, want) in cases {
            let bytes = pack_bits(&[bits]);
            let mut r = BitReader::new(&bytes);
            let got = decode_total_zeros(&mut r, *tc, TotalZerosTable::ChromaDc422).unwrap();
            assert_eq!(got, *want, "tc={tc} bits={bits}");
        }
    }

    // ---- run_before ----
    //
    // All cases below are derived from ITU-T H.264 (08/2024) §9.2.3
    // Table 9-10, transcribed one entry per row against the spec PDF.

    /// Run one decode case: feed `bits` into a BitReader, decode with
    /// `zeros_left`, and assert the result equals `want`.
    fn check_rb(bits: &str, zeros_left: u32, want: u32) {
        let bytes = pack_bits(&[bits]);
        let mut r = BitReader::new(&bytes);
        let got = decode_run_before(&mut r, zeros_left)
            .unwrap_or_else(|e| panic!("decode failed zl={zeros_left} bits={bits:?}: {e}"));
        assert_eq!(
            got, want,
            "zl={zeros_left} bits={bits:?} expected run_before={want} got {got}"
        );
    }

    #[test]
    fn run_before_zl_1_all_entries() {
        // §9.2.3 Table 9-10 column zerosLeft=1.
        check_rb("1", 1, 0);
        check_rb("0", 1, 1);
    }

    #[test]
    fn run_before_zl_2_all_entries() {
        // §9.2.3 Table 9-10 column zerosLeft=2.
        check_rb("1", 2, 0);
        check_rb("01", 2, 1);
        check_rb("00", 2, 2);
    }

    #[test]
    fn run_before_zl_3_all_entries() {
        // §9.2.3 Table 9-10 column zerosLeft=3.
        check_rb("11", 3, 0);
        check_rb("10", 3, 1);
        check_rb("01", 3, 2);
        check_rb("00", 3, 3);
    }

    #[test]
    fn run_before_zl_4_all_entries() {
        // §9.2.3 Table 9-10 column zerosLeft=4.
        check_rb("11", 4, 0);
        check_rb("10", 4, 1);
        check_rb("01", 4, 2);
        check_rb("001", 4, 3);
        check_rb("000", 4, 4);
    }

    #[test]
    fn run_before_zl_5_all_entries() {
        // §9.2.3 Table 9-10 column zerosLeft=5.
        check_rb("11", 5, 0);
        check_rb("10", 5, 1);
        check_rb("011", 5, 2);
        check_rb("010", 5, 3);
        check_rb("001", 5, 4);
        check_rb("000", 5, 5);
    }

    #[test]
    fn run_before_zl_6_all_entries() {
        // §9.2.3 Table 9-10 column zerosLeft=6 — non-monotone ordering.
        check_rb("11", 6, 0);
        check_rb("000", 6, 1);
        check_rb("001", 6, 2);
        check_rb("011", 6, 3);
        check_rb("010", 6, 4);
        check_rb("101", 6, 5);
        check_rb("100", 6, 6);
    }

    /// Every entry of Table 9-10 column "zerosLeft > 6" decoded with
    /// zeros_left=7 (the smallest value that uses the long-form table
    /// — this is the primary regression covering the
    /// "no matching run_before codeword (zeros_left=7)" error).
    ///
    /// Note: for zeros_left=7, run_before is bounded to 0..=7 by
    /// bitstream conformance (§9.2.3 step 2 requires the running
    /// subtraction to remain ≥ 0), but the VLC table itself is shared
    /// with all zeros_left ≥ 7 and accepts every codeword. We decode
    /// all 15 rows here to exercise every path through the VLC.
    #[test]
    fn run_before_zl_7_decodes_every_long_form_row() {
        check_rb("111", 7, 0);
        check_rb("110", 7, 1);
        check_rb("101", 7, 2);
        check_rb("100", 7, 3);
        check_rb("011", 7, 4);
        check_rb("010", 7, 5);
        check_rb("001", 7, 6);
        check_rb("0001", 7, 7);
        check_rb("00001", 7, 8);
        check_rb("000001", 7, 9);
        check_rb("0000001", 7, 10);
        check_rb("00000001", 7, 11);
        check_rb("000000001", 7, 12);
        check_rb("0000000001", 7, 13);
        check_rb("00000000001", 7, 14);
    }

    /// Spot-check: zeros_left ∈ {8,9,10,11,12,13,14} share the same VLC
    /// as zeros_left=7 ("zerosLeft > 6" column of Table 9-10). Verify
    /// a representative long codeword decodes identically across the
    /// whole range.
    #[test]
    fn run_before_long_form_shared_across_zl_7_to_14() {
        for zl in 7u32..=14 {
            check_rb("111", zl, 0); // shortest
            check_rb("0001", zl, 7); // just across the unary boundary
            check_rb("00000001", zl, 11);
            check_rb("00000000001", zl, 14); // longest (11 bits)
        }
    }

    /// The two extreme endpoints called out by the spec: run_before=7
    /// with zeros_left=7, and run_before=14 with zeros_left=14.
    #[test]
    fn run_before_endpoint_cases() {
        check_rb("0001", 7, 7); // zl=7, rb=7 (tight max)
        check_rb("00000000001", 14, 14); // zl=14, rb=14 (table max)
    }

    /// Unknown codewords must surface as CavlcError::UnknownRunBefore.
    #[test]
    fn run_before_errors_on_exhausted_stream() {
        // Feed only zeros — the long-form VLC consumes up to 11 bits
        // looking for a terminating 1. With fewer bits available the
        // reader should return a bitstream error (EOF), not a silent
        // wrong answer.
        let bytes: [u8; 1] = [0x00];
        let mut r = BitReader::new(&bytes);
        let err = decode_run_before(&mut r, 7).unwrap_err();
        // Either UnknownRunBefore or a Bitstream(Eof) is acceptable;
        // both indicate the VLC did not match.
        match err {
            CavlcError::UnknownRunBefore { zl } => assert_eq!(zl, 7),
            CavlcError::Bitstream(_) => {}
            other => panic!("expected UnknownRunBefore or Bitstream, got {other:?}"),
        }
    }

    /// zeros_left=0 is a caller-safety path: §9.2.3 says runVal[i]=0
    /// without reading bits. Verify we honour that and don't consume
    /// from the reader.
    #[test]
    fn run_before_zl_zero_returns_zero_without_reading() {
        let bytes: [u8; 1] = [0xFF];
        let mut r = BitReader::new(&bytes);
        assert_eq!(decode_run_before(&mut r, 0).unwrap(), 0);
        // Reader untouched — we can still pull the 8 bits back out.
        assert_eq!(r.u(8).unwrap(), 0xFF);
    }

    // ---- full block roundtrip ----

    #[test]
    fn full_block_small_example() {
        // Construct coeffLevel = [3, 0, 1, -1, 0, 0, 0, 0, ... padded to 16]
        // in zigzag order:
        //   coeffLevel[0] = 3
        //   coeffLevel[1] = 0
        //   coeffLevel[2] = 1
        //   coeffLevel[3] = -1
        //   rest = 0
        //
        // TotalCoeff = 3, TrailingOnes = 2 (the 1 and -1), trailing-ones
        // are the LAST non-zero coeffs before zeros — per the §9.2.4
        // combining rule, iteration walks from i=TotalCoeff-1 down to
        // i=0. The "trailing ones" are the up-to-3 highest-index ±1s.
        //
        // Coefficients in forward scan:  3  0  1 -1 ...
        // In §9.2 levelVal[] is indexed from high frequency inward:
        //   levelVal[0] = trailing_one at highest index = -1 (coeff[3])
        //   levelVal[1] = trailing_one at next index     =  1 (coeff[2])
        //   levelVal[2] = non-TO level                   =  3 (coeff[0])
        // runVal[] (high→low): gaps of zeros between non-zero coeffs
        // starting from the highest:
        //   After coeffLevel[3]=-1 nothing remains higher → total_zeros
        //     counts zeros BEFORE highest non-zero in forward scan, i.e.
        //     the number of zero coeffs interleaved with non-zero ones
        //     over indices 0..=3. There's 1 zero (at index 1).
        //   runVal[0] (gap before top) = 0 (no zeros between coeff[3] and end of block
        //     … but total_zeros = 1 counts interior zeros)
        //   runVal[1] (gap before coeff[2]) = 0 (adjacent to coeff[3])
        //   runVal[2] (gap before coeff[0]) = 1 (the zero at index 1)
        //
        // Verify with §9.2.4: coeffNum starts at -1, i walks 2→1→0.
        //   i=2: coeffNum += runVal[2]+1 = 2, coeffLevel[2]=levelVal[2]=3
        //   i=1: coeffNum += runVal[1]+1 = 3, coeffLevel[3]=1
        //   i=0: coeffNum += runVal[0]+1 = 4, coeffLevel[4]=-1
        // Hmm — that gives [0,0,3,1,-1,...] not [3,0,1,-1,...].
        //
        // So we redefine our target: coeffLevel[startIdx..] = [3, 0, 1, -1] at
        // indices 2..5 (start_idx=0, end_idx=15, so absolute indices
        // 2,3,4 will hold 3,1,-1 if runVal = {0, 0, 1} and levelVal =
        // {-1, 1, 3}).
        //
        // Result coeff_level (len 16): zeros except at indices 2, 3, 4 →
        //   coeff_level[2] = 3
        //   coeff_level[3] = 1
        //   coeff_level[4] = -1
        //
        // Encoding:
        //   coeff_token: nC=0 → TotalCoeff=3, TrailingOnes=2 → column0, (2,3)
        //     → "001"
        //   Actually wait: look up in TABLE_9_5_COL0 for (3, 2):
        //     (3,2) → "0000 101" (7 bits).
        //   trailing_ones_sign_flag for each trailing one (2 of them),
        //     MSB-first = first-in-scan-order. levelVal[0] = -1 → sign=1.
        //     levelVal[1] =  1 → sign=0. So "1" "0".
        //   Remaining level: levelVal[2] = 3 with suffixLength=0 and
        //     is_first_after_t1_lt_3 = (i==trailing_ones && T1<3) = true.
        //     To encode level=3: solve levelCode.
        //       If +3 and is_first_after, we reverse step 7: subtract 2
        //       from levelCode. levelVal=3 → levelCode=4 (since 3=
        //       (4+2)>>1). After reversal, levelCode' = 4-2 = 2.
        //       suffixLength=0 ⇒ Min(15, lp)<<0 + suf = lp + suf = 2.
        //       Smallest level_prefix+level_suffix combination: lp=2,
        //       suf=0 (suffix size = 0 since lp<15 and lp!=14 with sl=0)
        //       ⇒ level_prefix = 2 → bit string "001".
        //
        //   total_zeros: TotalCoeff=3, zeros-between = 1 (the zero at
        //     index 3 in absolute terms, which is index 1 within the
        //     highest 4 non-zero cells). Actually total_zeros is the
        //     count of zeros between the first non-zero and the last.
        //     With coeff_level[2]=3, [3]=1, [4]=-1, scanning 0..=15:
        //     last non-zero = index 4. First = index 2. But §9.2.3
        //     defines total_zeros as the sum of zeros between non-zero
        //     coefficients in forward scan order from startIdx to the
        //     LAST non-zero, NOT including zeros above the last.
        //
        //     Let me re-read §9.2.3 and §9.2.4.
        //
        //     §9.2.4 rewrites coeffLevel using coeffNum starting at −1.
        //     coeffNum tracks the position in the forward scan of the
        //     LAST non-zero coefficient backwards. After all iterations,
        //     coeffNum = TotalCoeff + sum(runVal) = index of the LAST
        //     non-zero in forward scan. total_zeros = sum of run_before
        //     values, so the last non-zero is at coeffNum = TotalCoeff +
        //     total_zeros - 1.
        //
        //     For our coeff array with non-zeros at indices 2, 3, 4:
        //       last non-zero index = 4, TotalCoeff = 3.
        //       4 = 3 + total_zeros - 1 ⇒ total_zeros = 2.
        //
        //     So coeff_level[0..=1] are zero (before first non-zero),
        //     coeff_level[5..=15] are zero (after last non-zero).
        //     The 2 zeros counted by total_zeros are BOTH before the
        //     first non-zero (at indices 0 and 1). There are 0 zeros
        //     between the non-zeros (indices 2,3,4 are all non-zero).
        //
        //     runVal[i] for i from TotalCoeff-1 down to 0:
        //       i=2 (highest-frequency levelVal, = 3 at position 2):
        //         iteration 1 of total_coeff-1 = 2 iterations.
        //         zerosLeft starts = 2.
        //         runVal[i=0 in spec ordering]: spec walks i=0..TotalCoeff-1
        //         in §9.2.3. Let me re-read.
        //
        // §9.2.3: "Initially, an index i is set equal to 0. [...]
        //  The following ordered steps are then performed
        //  TotalCoeff(coeff_token)-1 times:
        //   1. runVal[i] = run_before (if zerosLeft > 0) else 0
        //   2. zerosLeft -= runVal[i]
        //   3. i ++
        //  Finally, runVal[i] = zerosLeft."
        //
        // So runVal[0..TotalCoeff-2] from run_before, runVal[TotalCoeff-1]
        // = leftover zerosLeft.
        //
        // §9.2.4: "coeffNum = -1; i = TotalCoeff-1;
        //  TotalCoeff times:
        //   1. coeffNum += runVal[i] + 1
        //   2. coeffLevel[coeffNum] = levelVal[i]
        //   3. i --"
        //
        // So coeffLevel gets indices = cumulative sum of (runVal[T-1],
        // runVal[T-2], ..., runVal[0]) + iteration count.
        //
        // Desired final coeff_level: non-zero at indices 2,3,4 with
        // values 3, 1, -1.
        //
        // Work backwards: iteration j=0 (i=2): coeffNum = runVal[2]+0 =
        //   runVal[2]. That must equal 2 (first written index). So
        //   runVal[2] = 2.
        // iteration j=1 (i=1): coeffNum += runVal[1]+1. Must equal 3.
        //   So runVal[1]+1 = 1 ⇒ runVal[1] = 0.
        // iteration j=2 (i=0): coeffNum += runVal[0]+1. Must equal 4.
        //   runVal[0]+1 = 1 ⇒ runVal[0] = 0.
        //
        // levelVal[0..=2]: after iteration j=0 we wrote levelVal[2] = 3
        // at coeff_level[2]. After j=1: levelVal[1] = 1 at
        // coeff_level[3]. After j=2: levelVal[0] = -1 at
        // coeff_level[4].
        //
        // Per §9.2.2 build order: trailing_ones_sign_flags are read for
        // i=0..TrailingOnes-1. Those set levelVal[0..=T-1] to ±1.
        // Remaining levels come after. We have trailing_ones = 2:
        //   levelVal[0] = -1 → sign=1
        //   levelVal[1] =  1 → sign=0
        //   levelVal[2] = 3  (non-trailing)
        // Good.
        //
        // runVal derivation in §9.2.3 order:
        //   runVal[0] reads run_before with zerosLeft starting at
        //   total_zeros = 2. We need runVal[0] = 0.
        //   After: zerosLeft = 2 - 0 = 2.
        //   runVal[1] reads run_before with zerosLeft = 2. We need
        //   runVal[1] = 0.
        //   After: zerosLeft = 2 - 0 = 2.
        //   Last: runVal[2] = zerosLeft = 2. (Matches the required 2.)
        //
        // Encoding of run_before values:
        //   runVal[0] = 0 with zerosLeft = 2 → Table 9-10 zl=2, rb=0 →
        //     "1" (1 bit).
        //   runVal[1] = 0 with zerosLeft = 2 → "1" (1 bit).
        //
        // total_zeros = 2 encoding with tzVlcIndex = 3 (TotalCoeff):
        //   Table 9-7 col tzVlcIndex=3, total_zeros=2 → "110" (3 bits).
        //
        // Full bit stream:
        //   coeff_token (3,2)   = "0000 101"  (7 bits)   [TABLE_9_5_COL0]
        //   trailing_ones signs = "1" "0"     (2 bits)
        //   remaining level, levelVal[2]=3:
        //     is_first_after_t1_lt_3 = true (i=2 = T1=2, and T1=2 < 3)
        //     suffixLength = 0 (TotalCoeff=3 ≤ 10).
        //     level_prefix = 2 → "001" (3 bits)
        //     levelSuffixSize = 0, no level_suffix bits.
        //   total_zeros = "110" (3 bits)
        //   run_before[0] zl=2 rb=0 → "1" (1 bit)
        //   run_before[1] zl=2 rb=0 → "1" (1 bit)
        //
        //   Total = 7+2+3+3+1+1 = 17 bits.

        let bytes = pack_bits(&[
            "0000101", // coeff_token (3,2) column 0
            "1",       // trailing-one sign for levelVal[0] = -1
            "0",       // trailing-one sign for levelVal[1] = 1
            "001",     // level_prefix = 2 (for levelVal[2] = 3)
            "110",     // total_zeros = 2, tzVlc=3
            "1",       // run_before[0] = 0 (zl=2)
            "1",       // run_before[1] = 0 (zl=2)
        ]);
        let mut r = BitReader::new(&bytes);
        let coeff =
            parse_residual_block_cavlc(&mut r, CoeffTokenContext::Numeric(0), 0, 15, 16).unwrap();

        let mut expected = vec![0i32; 16];
        expected[2] = 3;
        expected[3] = 1;
        expected[4] = -1;
        assert_eq!(coeff, expected);
    }

    #[test]
    fn full_block_all_zero() {
        // coeff_token with TotalCoeff=0 → return all-zero vec, no more
        // bits consumed. Column 0, TotalCoeff=0, TrailingOnes=0 → "1".
        let bytes = pack_bits(&["1"]);
        let mut r = BitReader::new(&bytes);
        let coeff =
            parse_residual_block_cavlc(&mut r, CoeffTokenContext::Numeric(0), 0, 15, 16).unwrap();
        assert_eq!(coeff, vec![0; 16]);
    }

    /// Regression test: prior to the 2026-04-20 COL1 11-bit-length fix,
    /// an Intra_16x16 AC block with neighbour nC==3 whose coeff_token
    /// was (TC=7, T1=0) — "0000 0001 111" in the spec's 11-bit form —
    /// would be mis-decoded as a 10-bit codeword, consuming only 10 of
    /// the 11 bits the spec requires. This test exercises parse end-
    /// to-end and verifies the residual stops at the expected bit
    /// cursor.
    #[test]
    fn col1_11bit_codeword_full_parse_regression() {
        // Build a minimal valid 4x4 residual block for nC=3 with
        // TotalCoeff=7, TrailingOnes=0:
        //   coeff_token (7, 0) col1 "0000 0001 111" (11 bits)
        //   followed by 7 non-trailing-one level values and
        //   total_zeros + run_before encoding. To keep the test simple
        //   we use level_prefix=0 and appropriate suffix choices so
        //   every level decodes to ±1 or ±small values. With T1=0 the
        //   first non-trailing is a "first-after" with T1 < 3, so
        //   levelCode = level_prefix + 2 on the first level.
        //
        // suffixLength starts at 0 (TotalCoeff=7 ≤ 10).
        // Level 0 (is_first_after = true): level_prefix=0 ("1") ⇒
        //   levelCode=0+2=2 (even) ⇒ levelVal = (2+2)/2 = 2.
        //   step 9 → next_sl=1. abs(2) > 3? No. next_sl stays 1.
        // Level 1: level_prefix=0 ("1"), suffix=0 (1 bit) ⇒
        //   levelCode=(0<<1)+0=0 even ⇒ levelVal = 1. step 10: |1|>3? No.
        // Levels 2..6: same as level 1, all +1.
        //
        // total_zeros with tzVlc=7, want 0 ("0000 01", 6 bits).
        // All runVal are 0.
        //
        // Expected: coeff_level = [2, 1, 1, 1, 1, 1, 1, 0..=0]
        //
        // Bits:
        //   coeff_token "00000001111" (11)
        //   level[0] lp=0 "1" (1)         → levelVal 2
        //   level[1] lp=0 "1" + suffix 0 (1 bit) "0" → levelVal 1
        //   (same for 2..6) : 5 more "10" pairs = 10 bits
        //   total_zeros "000001" (6)
        //   No run_before reads (all zerosLeft=0).
        let bytes = pack_bits(&[
            "00000001111", // coeff_token (7, 0) col1 11-bit
            "1",           // level_prefix=0 for levelVal[0]=2 (first-after)
            "1",
            "0", // level_prefix=0, suffix=0 for levelVal[1]=1
            "1",
            "0", // levelVal[2]=1
            "1",
            "0", // levelVal[3]=1
            "1",
            "0", // levelVal[4]=1
            "1",
            "0", // levelVal[5]=1
            "1",
            "0",      // levelVal[6]=1
            "000001", // total_zeros=0 tzVlc=7
        ]);
        let mut r = BitReader::new(&bytes);
        let coeff =
            parse_residual_block_cavlc(&mut r, CoeffTokenContext::Numeric(3), 0, 15, 16).unwrap();
        let mut expected = vec![0i32; 16];
        // §9.2.4 walk: i=T-1 down to 0, coeffNum starts at -1.
        //   i=6 (levelVal=1): coeffNum += runVal[6]+1 = 0, coeff[0]=1
        //   i=5 (1):           coeffNum += 1 = 1, coeff[1]=1
        //   i=4 (1):           coeff[2]=1
        //   i=3 (1):           coeff[3]=1
        //   i=2 (1):           coeff[4]=1
        //   i=1 (1):           coeff[5]=1
        //   i=0 (2):           coeff[6]=2
        expected[0] = 1;
        expected[1] = 1;
        expected[2] = 1;
        expected[3] = 1;
        expected[4] = 1;
        expected[5] = 1;
        expected[6] = 2;
        assert_eq!(coeff, expected);
        // Cursor should have advanced exactly 11 + 1 + 2*6 + 6 = 30 bits.
        let (b, bi) = r.position();
        assert_eq!(b * 8 + bi as usize, 30);
    }

    #[test]
    fn full_block_single_trailing_one() {
        // TotalCoeff=1, TrailingOnes=1, levelVal[0] = ±1, runVal[0] =
        // total_zeros = whichever position the coefficient ends up at.
        //
        // Put the single coefficient value = -1 at coeff_level[0] of a
        // 16-coefficient block.
        //   coeff_token (1,1) col 0: "01" (2 bits)
        //   trailing-one sign: "1" (negative → -1)
        //   TotalCoeff != maxNumCoeff(=16) so we read total_zeros.
        //   total_zeros for TotalCoeff=1 (tzVlc=1): want 0 → "1"
        //   Only TotalCoeff-1 = 0 run_before reads.
        //   runVal[0] = leftover zerosLeft = 0.
        //   §9.2.4: coeffNum = -1 + 0 + 1 = 0 → coeff_level[0] = -1. Good.
        let bytes = pack_bits(&["01", "1", "1"]);
        let mut r = BitReader::new(&bytes);
        let coeff =
            parse_residual_block_cavlc(&mut r, CoeffTokenContext::Numeric(0), 0, 15, 16).unwrap();
        let mut expected = vec![0i32; 16];
        expected[0] = -1;
        assert_eq!(coeff, expected);
    }
}
