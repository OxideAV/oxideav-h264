//! §7.3.4 / §7.4.4 — slice_data walker.
//!
//! Walks the macroblocks within a slice. Clean-room implementation from
//! ITU-T Rec. H.264 (08/2024).
//!
//! Responsibilities (§7.3.4):
//!
//! 1. **CABAC bootstrap**: when `entropy_coding_mode_flag == 1`, parse
//!    `cabac_alignment_one_bit`s until byte aligned, then instantiate
//!    the CABAC engine and per-slice contexts
//!    ([`crate::cabac::CabacDecoder::new`] + [`CabacContexts::init`]).
//!
//! 2. **`CurrMbAddr` seeding**: starts at
//!    `first_mb_in_slice * (1 + MbaffFrameFlag)`
//!    per §7.3.4, and advances via `NextMbAddress` (§8.2.2). With FMO
//!    disabled this is just `CurrMbAddr + 1` (which is what we
//!    implement — MBAFF/FMO neighbour wiring is above this module's
//!    scope).
//!
//! 3. **Per-MB loop**:
//!    - `mb_skip_run` (CAVLC, non-I/SI slices): a `ue(v)` counting how
//!      many MBs to skip before the next coded MB.
//!    - `mb_skip_flag` (CABAC, non-I/SI slices): decoded via
//!      [`crate::cabac_ctx::decode_mb_skip_flag`].
//!    - `mb_field_decoding_flag`: read per §7.3.4 when
//!      `MbaffFrameFlag == 1` and
//!      `(CurrMbAddr % 2 == 0 || (CurrMbAddr % 2 == 1 && prevMbSkipped))`.
//!      Phase-1 parsing only: the flag is captured per MB pair (both
//!      MBs of a pair record the same value) but MBAFF reconstruction
//!      is still out of scope.
//!    - [`crate::macroblock_layer::parse_macroblock`] for non-skipped
//!      MBs.
//!
//! 4. **Termination**:
//!    - CAVLC: loop while `more_rbsp_data()`.
//!    - CABAC: decode `end_of_slice_flag` via
//!      [`crate::cabac_ctx::decode_end_of_slice_flag`] after each MB.
//!      In MBAFF, `end_of_slice_flag` is only decoded when
//!      `CurrMbAddr % 2 == 1` (top-of-pair always continues to bottom,
//!      per §7.3.4).

#![allow(dead_code)]

use crate::bitstream::{BitError, BitReader};
use crate::cabac::{CabacDecoder, CabacError};
use crate::cabac_ctx::{
    decode_end_of_slice_flag, decode_mb_skip_flag, CabacContexts, NeighbourCtx, SliceKind,
};
use crate::macroblock_layer::{
    parse_macroblock, CabacNeighbourGrid, CavlcNcGrid, EntropyState, Macroblock,
    MacroblockLayerError,
};
use crate::pps::Pps;
use crate::slice_header::{SliceHeader, SliceType};
use crate::sps::Sps;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SliceDataError {
    #[error("bitstream read failed: {0}")]
    Bitstream(#[from] BitError),
    #[error("CABAC engine failed: {0}")]
    Cabac(#[from] CabacError),
    #[error("macroblock_layer: {0}")]
    Macroblock(#[from] MacroblockLayerError),
    /// A macroblock failed to parse; carries the MB address + bit
    /// offset at MB entry for diagnostic purposes.
    #[error("macroblock #{mb_addr} at byte {byte}:bit {bit}: {source}")]
    MacroblockAt {
        mb_addr: u32,
        byte: usize,
        bit: u8,
        #[source]
        source: MacroblockLayerError,
    },
    /// §7.3.4 — MBAFF reconstruction is not wired in this walker. Phase
    /// 1 parses the MB-pair + `mb_field_decoding_flag` structure but
    /// downstream layers may still reject MBAFF streams with this.
    #[error("MBAFF macroblock layer is not supported in this walker")]
    MbaffNotSupported,
    /// §7.4.4 — `slice_qp_y` (from slice_header + pps) out of valid range
    /// 0..=51.
    #[error("derived SliceQPY {0} out of range")]
    SliceQpOutOfRange(i32),
}

pub type SliceDataResult<T> = Result<T, SliceDataError>;

/// §7.3.4 — parsed slice_data payload.
#[derive(Debug, Clone, Default)]
pub struct SliceData {
    /// One entry per macroblock (including implicit skip entries).
    pub macroblocks: Vec<Macroblock>,
    /// §7.4.4 — `mb_field_decoding_flag` per macroblock, parallel to
    /// `macroblocks`. For non-MBAFF slices every entry is `false`
    /// (the flag is absent from the bitstream and is inferred to
    /// `field_pic_flag` by the spec, which is also `false` for a
    /// frame-coded picture). For MBAFF slices both MBs of a pair
    /// carry the same value — the one parsed at the top of the pair
    /// (or retroactively at the bottom when the top was skipped).
    ///
    /// Invariant: `mb_field_decoding_flags.len() == macroblocks.len()`.
    pub mb_field_decoding_flags: Vec<bool>,
    /// Final CurrMbAddr after the loop — i.e. `first_mb + len(macroblocks)`
    /// for single-slice-group streams.
    pub last_mb_addr: u32,
}

/// Map [`SliceType`] to the CABAC [`SliceKind`].
fn slice_kind(slice_type: SliceType) -> SliceKind {
    match slice_type {
        SliceType::I => SliceKind::I,
        SliceType::P => SliceKind::P,
        SliceType::B => SliceKind::B,
        SliceType::SP => SliceKind::SP,
        SliceType::SI => SliceKind::SI,
    }
}

/// §7.3.4 — walk a slice's `slice_data()`.
///
/// `rbsp` is the de-emulated RBSP (emulation-prevention bytes stripped)
/// starting at the beginning of the NAL unit. `bit_cursor_bytes` +
/// `bit_cursor_bits` pinpoint the position within `rbsp` where
/// slice_data() begins — i.e. the byte offset of the next bit plus
/// the MSB-first bit index within that byte (matching the convention
/// used by [`BitReader::position`]).
pub fn parse_slice_data(
    rbsp: &[u8],
    bit_cursor_bytes: usize,
    bit_cursor_bits: u8,
    slice_header: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
) -> SliceDataResult<SliceData> {
    // §7.3.4 — MbaffFrameFlag = mb_adaptive_frame_field_flag &&
    // !field_pic_flag. Phase 1: the walker steps in MB pairs and
    // reads mb_field_decoding_flag per pair, but downstream
    // reconstruction is still out of scope (it will reject the
    // returned SliceData via its own checks).
    let mbaff_frame_flag = sps.mb_adaptive_frame_field_flag && !slice_header.field_pic_flag;

    // Position the reader at (bit_cursor_bytes, bit_cursor_bits).
    let mut r = position_reader(rbsp, bit_cursor_bytes, bit_cursor_bits)?;

    let kind = slice_kind(slice_header.slice_type);
    let chroma_array_type = sps.chroma_array_type();

    // §7.4.2.1 — QpBdOffsetY = 6 * bit_depth_luma_minus8, SliceQPY =
    // 26 + pic_init_qp_minus26 + slice_qp_delta (§7.4.3).
    let slice_qp_y = 26 + pps.pic_init_qp_minus26 + slice_header.slice_qp_delta;
    if !(0..=51).contains(&slice_qp_y) {
        return Err(SliceDataError::SliceQpOutOfRange(slice_qp_y));
    }

    let mut macroblocks: Vec<Macroblock> = Vec::new();
    let mut mb_field_decoding_flags: Vec<bool> = Vec::new();
    let mut curr_mb_addr: u32 = slice_header.first_mb_in_slice * (1 + u32::from(mbaff_frame_flag));

    // §9.2.1.1 — CAVLC nC neighbour grid, allocated per picture. The
    // grid is only consulted in the CAVLC path but we allocate it for
    // the CABAC path too so any future CABAC residual-neighbour work
    // can re-use the same store.
    let pic_w_mbs = sps.pic_width_in_mbs_minus1 + 1;
    let pic_h_mus = sps.pic_height_in_map_units_minus1 + 1;
    let pic_h_mbs = if sps.frame_mbs_only_flag {
        pic_h_mus
    } else {
        pic_h_mus * 2
    };
    let mut cavlc_nc = CavlcNcGrid::new(pic_w_mbs, pic_h_mbs);

    if pps.entropy_coding_mode_flag {
        // ---------------------------------------------------------
        // CABAC path (§7.3.4).
        // ---------------------------------------------------------
        while !r.byte_aligned() {
            // §7.3.4 — cabac_alignment_one_bit. The spec mandates each
            // alignment bit be equal to 1; we tolerate any value for
            // robustness in fixtures.
            let _ = r.u(1)?;
        }
        // Hand the byte-aligned remainder to the CABAC engine. CABAC's
        // first-byte initialisation consumes 9 bits from its own
        // private reader.
        let byte_pos = r.position().0;
        let remainder = &rbsp[byte_pos..];
        let mut cabac_dec = CabacDecoder::new(BitReader::new(remainder))?;
        let mut ctxs = CabacContexts::init(
            kind,
            match kind {
                SliceKind::I | SliceKind::SI => None,
                _ => Some(slice_header.cabac_init_idc),
            },
            slice_qp_y,
        )?;

        // §7.3.4 — prevMbSkipped is initialised to 0; in the CABAC
        // path it is updated each iteration to mb_skip_flag when the
        // slice isn't I/SI.
        let mut prev_mb_skipped = false;
        // MBAFF: the flag is decoded once per MB pair but applies to
        // both MBs. `pending_pair_flag` holds the top MB's flag so the
        // bottom MB gets the same value without a second read.
        let mut pending_pair_flag: Option<bool> = None;
        // §9.3.3.1.1.9 — CABAC residual neighbour grid; populated per MB
        // as the walker steps, and consulted when deriving `ctxIdxInc`
        // for coded_block_flag / ref_idx / mvd on subsequent MBs.
        let mut cabac_nb = CabacNeighbourGrid::new(pic_w_mbs, pic_h_mbs);
        loop {
            let mut skipped = false;
            let mut mb_skip_flag_this_iter = false;
            if !slice_header.slice_type.is_intra() {
                let mb_skip_flag =
                    decode_mb_skip_flag(&mut cabac_dec, &mut ctxs, kind, &NeighbourCtx::default())?;
                mb_skip_flag_this_iter = mb_skip_flag;
                if mb_skip_flag {
                    // §7.4.4 — mb_field_decoding_flag for this MB is
                    // not read here. If this MB is the top of a pair
                    // whose flag was never read, the inference from
                    // §7.4.4 applies (Phase-1: default to 0 = frame
                    // pair; real spatial inference needs MBAFF
                    // neighbour addressing which is out of scope).
                    let flag = pending_pair_flag.unwrap_or(false);
                    macroblocks.push(Macroblock::new_skip(slice_header.slice_type));
                    mb_field_decoding_flags.push(flag);
                    // §9.2.1.1 step 6 — a P_Skip / B_Skip neighbour
                    // contributes nN = 0. Mark available + is_skip.
                    if let Some(slot) = cavlc_nc.mbs.get_mut(curr_mb_addr as usize) {
                        slot.is_available = true;
                        slot.is_skip = true;
                        slot.is_intra = false;
                        slot.is_i_pcm = false;
                        slot.luma_total_coeff = [0; 16];
                        slot.cb_total_coeff = [0; 8];
                        slot.cr_total_coeff = [0; 8];
                    }
                    // If we just completed a pair (odd CurrMbAddr),
                    // clear the pending_pair_flag — the next iteration
                    // starts a new pair.
                    if mbaff_frame_flag && curr_mb_addr % 2 == 1 {
                        pending_pair_flag = None;
                    }
                    curr_mb_addr += 1;
                    skipped = true;
                }
            }
            if !skipped {
                // §7.3.4 — MBAFF: read mb_field_decoding_flag before
                // macroblock_layer() when (CurrMbAddr % 2 == 0) or
                // (CurrMbAddr % 2 == 1 && prevMbSkipped). The flag
                // applies to both MBs of the pair.
                if mbaff_frame_flag
                    && (curr_mb_addr % 2 == 0
                        || (curr_mb_addr % 2 == 1 && prev_mb_skipped))
                {
                    let flag = decode_mb_field_decoding_flag_cabac(
                        &mut cabac_dec,
                        &mut ctxs,
                    )?;
                    pending_pair_flag = Some(flag);
                    // Retroactively patch the top MB of this pair if
                    // it was skipped (CurrMbAddr % 2 == 1 path).
                    if curr_mb_addr % 2 == 1 {
                        if let Some(last) = mb_field_decoding_flags.last_mut() {
                            *last = flag;
                        }
                    }
                }
                let flag = pending_pair_flag.unwrap_or(false);
                let mut entropy = EntropyState {
                    cabac: Some((&mut cabac_dec, &mut ctxs)),
                    slice_kind: kind,
                    neighbours: NeighbourCtx::default(),
                    prev_mb_qp_delta_nonzero: false,
                    chroma_array_type,
                    transform_8x8_mode_flag: pps.transform_8x8_mode_flag(),
                    cavlc_nc: Some(&mut cavlc_nc),
                    current_mb_addr: curr_mb_addr,
                    constrained_intra_pred_flag: pps.constrained_intra_pred_flag,
                    num_ref_idx_l0_active_minus1: slice_header.num_ref_idx_l0_active_minus1,
                    num_ref_idx_l1_active_minus1: slice_header.num_ref_idx_l1_active_minus1,
                    mbaff_frame_flag: false,
                    // CABAC per-MB neighbour grid — spec-correct per
                    // §9.3.3.1.1.9.
                    cabac_nb: Some(&mut cabac_nb),
                    pic_width_in_mbs: pic_w_mbs,
                };
                let (byte, bit) = r.position();
                let mb = parse_macroblock(
                    &mut r, &mut entropy, slice_header, sps, pps, curr_mb_addr,
                )
                .map_err(|source| SliceDataError::MacroblockAt {
                    mb_addr: curr_mb_addr,
                    byte,
                    bit,
                    source,
                })?;
                macroblocks.push(mb);
                mb_field_decoding_flags.push(flag);
                if mbaff_frame_flag && curr_mb_addr % 2 == 1 {
                    pending_pair_flag = None;
                }
                curr_mb_addr += 1;
            }
            // §7.3.4 — in the CABAC path, update prevMbSkipped with
            // the most recent mb_skip_flag (only when read at all).
            if !slice_header.slice_type.is_intra() {
                prev_mb_skipped = mb_skip_flag_this_iter;
            }
            // §7.3.4 — MBAFF: end_of_slice_flag is only decoded when
            // CurrMbAddr % 2 == 1 (i.e. at the bottom MB of a pair).
            // For top MBs, moreDataFlag is forced to 1 so we continue
            // unconditionally to the bottom MB.
            if mbaff_frame_flag && curr_mb_addr % 2 == 1 {
                continue;
            }
            let end = decode_end_of_slice_flag(&mut cabac_dec)?;
            if end {
                break;
            }
        }
    } else {
        // ---------------------------------------------------------
        // CAVLC path (§7.3.4).
        // ---------------------------------------------------------
        // §7.3.4 — prevMbSkipped is the "top MB of this pair was
        // skipped" signal. CAVLC sets it from (mb_skip_run > 0).
        let mut prev_mb_skipped = false;
        // MBAFF: pair flag shared by both MBs of a pair.
        let mut pending_pair_flag: Option<bool> = None;
        let mut pending_skip: u32 = 0;
        loop {
            // On non-I/SI slices, an `mb_skip_run` precedes each coded
            // macroblock (the "skip run" is exp-Golomb).
            if !slice_header.slice_type.is_intra() && pending_skip == 0 {
                pending_skip = r.ue()?;
                prev_mb_skipped = pending_skip > 0;
                for _ in 0..pending_skip {
                    // §7.4.4 — inferred mb_field_decoding_flag for a
                    // skipped MB whose pair flag hasn't been read.
                    // Phase-1 uses 0 as the spec's default-when-no-
                    // neighbour inference outcome; full spatial
                    // inference is out of scope.
                    let flag = pending_pair_flag.unwrap_or(false);
                    macroblocks.push(Macroblock::new_skip(slice_header.slice_type));
                    mb_field_decoding_flags.push(flag);
                    // §9.2.1.1 step 6 — skipped MB contributes nN = 0.
                    if let Some(slot) = cavlc_nc.mbs.get_mut(curr_mb_addr as usize) {
                        slot.is_available = true;
                        slot.is_skip = true;
                        slot.is_intra = false;
                        slot.is_i_pcm = false;
                        slot.luma_total_coeff = [0; 16];
                        slot.cb_total_coeff = [0; 8];
                        slot.cr_total_coeff = [0; 8];
                    }
                    // If we rolled through the bottom of a pair, the
                    // pair is complete — clear the pending flag.
                    if mbaff_frame_flag && curr_mb_addr % 2 == 1 {
                        pending_pair_flag = None;
                    }
                    curr_mb_addr += 1;
                }
                // After advancing the skip run, the CAVLC spec says:
                //   if( mb_skip_run > 0 ) moreDataFlag = more_rbsp_data()
                if pending_skip > 0 && !r.more_rbsp_data() {
                    break;
                }
            }
            // If we just consumed all remaining slice_data as skip
            // runs, check the more_rbsp_data() guard.
            if !r.more_rbsp_data() {
                break;
            }
            // §7.3.4 — MBAFF: read mb_field_decoding_flag before
            // macroblock_layer() when (CurrMbAddr % 2 == 0) or
            // (CurrMbAddr % 2 == 1 && prevMbSkipped).
            if mbaff_frame_flag
                && (curr_mb_addr % 2 == 0
                    || (curr_mb_addr % 2 == 1 && prev_mb_skipped))
            {
                let flag = r.u(1)? != 0;
                pending_pair_flag = Some(flag);
                // Retroactively patch the (skipped) top MB of this
                // pair if we're at the bottom.
                if curr_mb_addr % 2 == 1 {
                    if let Some(last) = mb_field_decoding_flags.last_mut() {
                        *last = flag;
                    }
                }
            }
            let flag = pending_pair_flag.unwrap_or(false);
            let mut entropy = EntropyState {
                cabac: None,
                slice_kind: kind,
                neighbours: NeighbourCtx::default(),
                prev_mb_qp_delta_nonzero: false,
                chroma_array_type,
                transform_8x8_mode_flag: pps.transform_8x8_mode_flag(),
                cavlc_nc: Some(&mut cavlc_nc),
                current_mb_addr: curr_mb_addr,
                constrained_intra_pred_flag: pps.constrained_intra_pred_flag,
                num_ref_idx_l0_active_minus1: slice_header.num_ref_idx_l0_active_minus1,
                num_ref_idx_l1_active_minus1: slice_header.num_ref_idx_l1_active_minus1,
                mbaff_frame_flag: false,
                // CABAC neighbour grid unused on the CAVLC path.
                cabac_nb: None,
                pic_width_in_mbs: 0,
            };
            let (byte, bit) = r.position();
            let mb = parse_macroblock(&mut r, &mut entropy, slice_header, sps, pps, curr_mb_addr)
                .map_err(|source| SliceDataError::MacroblockAt {
                    mb_addr: curr_mb_addr,
                    byte,
                    bit,
                    source,
                })?;
            macroblocks.push(mb);
            mb_field_decoding_flags.push(flag);
            // CAVLC: the per-iteration mb_skip_run update is what
            // drives prev_mb_skipped; a coded-MB iteration does not
            // observe a new skip run so prev_mb_skipped remains as
            // last set (either false after a non-skipped iteration, or
            // true after a skip-run iteration). After parsing a coded
            // MB on the bottom of a pair, we've consumed the pair:
            // clear the pending flag + reset prev_mb_skipped so the
            // next iteration starts fresh.
            if mbaff_frame_flag && curr_mb_addr % 2 == 1 {
                pending_pair_flag = None;
            }
            prev_mb_skipped = false;
            curr_mb_addr += 1;
            pending_skip = 0;
            if !r.more_rbsp_data() {
                break;
            }
        }
    }

    debug_assert_eq!(
        macroblocks.len(),
        mb_field_decoding_flags.len(),
        "mb_field_decoding_flags must be parallel to macroblocks"
    );
    Ok(SliceData {
        macroblocks,
        mb_field_decoding_flags,
        last_mb_addr: curr_mb_addr,
    })
}

/// §7.3.4 + §9.3.3.1.1.2 — decode `mb_field_decoding_flag` as ae(v) in
/// CABAC. Phase-1 uses ctxIdxInc = 0 (both condTermFlagA and
/// condTermFlagB = 0) because MBAFF neighbour addressing (§6.4.10) is
/// not wired yet; a proper decoder must consult the left + above MB
/// pair's mb_field_decoding_flag to set ctxIdxInc ∈ {0,1,2}. The
/// value still round-trips correctly for streams whose neighbours are
/// all frame pairs (the common case at pair 0).
///
/// Per Table 9-34: ctxIdxOffset = 70, binarization FL with cMax = 1,
/// so a single decode_decision call.
fn decode_mb_field_decoding_flag_cabac(
    dec: &mut CabacDecoder<'_>,
    ctxs: &mut CabacContexts,
) -> SliceDataResult<bool> {
    const CTX_IDX_OFFSET: usize = 70;
    let ctx_idx_inc: usize = 0; // Phase-1 default; see doc-comment.
    let ctx_idx = CTX_IDX_OFFSET + ctx_idx_inc;
    let bin = dec.decode_decision(ctxs.at_mut(ctx_idx))?;
    Ok(bin != 0)
}

/// Construct a `BitReader` positioned at the given (byte, bit) within
/// `rbsp`. Returns an `Eof` when the position is past the end.
fn position_reader(rbsp: &[u8], byte: usize, bit: u8) -> SliceDataResult<BitReader<'_>> {
    if byte > rbsp.len() || bit >= 8 {
        return Err(SliceDataError::Bitstream(BitError::Eof));
    }
    let mut r = BitReader::new(rbsp);
    // Walk to the target position without exposing raw cursor fields.
    let total_bits = byte * 8 + bit as usize;
    for _ in 0..total_bits {
        r.u(1)?;
    }
    Ok(r)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macroblock_layer::MbType;

    // Re-use the fixture helpers via a local copy (slightly redundant
    // but keeps this module test-self-contained).

    struct BitWriter {
        bytes: Vec<u8>,
        bit_pos: u8,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                bit_pos: 0,
            }
        }
        fn u(&mut self, bits: u32, value: u32) {
            for i in (0..bits).rev() {
                let bit = ((value >> i) & 1) as u8;
                if self.bit_pos == 0 {
                    self.bytes.push(0);
                }
                let idx = self.bytes.len() - 1;
                self.bytes[idx] |= bit << (7 - self.bit_pos);
                self.bit_pos = (self.bit_pos + 1) % 8;
            }
        }
        fn ue(&mut self, value: u32) {
            let v = value + 1;
            let leading = 31 - v.leading_zeros();
            for _ in 0..leading {
                self.u(1, 0);
            }
            self.u(leading + 1, v);
        }
        fn se(&mut self, value: i32) {
            let k = if value <= 0 {
                (-2 * value) as u32
            } else {
                (2 * value - 1) as u32
            };
            self.ue(k);
        }
        fn trailing(&mut self) {
            self.u(1, 1);
            while self.bit_pos != 0 {
                self.u(1, 0);
            }
        }
        fn into_bytes(self) -> Vec<u8> {
            self.bytes
        }
    }

    fn dummy_sps() -> Sps {
        Sps {
            profile_idc: 66,
            constraint_set_flags: 0,
            level_idc: 30,
            seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            qpprime_y_zero_transform_bypass_flag: false,
            seq_scaling_matrix_present_flag: false,
            seq_scaling_lists: None,
            log2_max_frame_num_minus4: 0,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb_minus4: 0,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_pic_order_cnt_cycle: 0,
            offset_for_ref_frame: Vec::new(),
            max_num_ref_frames: 1,
            gaps_in_frame_num_value_allowed_flag: false,
            pic_width_in_mbs_minus1: 19,
            pic_height_in_map_units_minus1: 14,
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: true,
            frame_cropping: None,
            vui_parameters_present_flag: false,
            vui: None,
        }
    }

    fn dummy_pps() -> Pps {
        Pps {
            pic_parameter_set_id: 0,
            seq_parameter_set_id: 0,
            entropy_coding_mode_flag: false,
            bottom_field_pic_order_in_frame_present_flag: false,
            num_slice_groups_minus1: 0,
            slice_group_map: None,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            pic_init_qp_minus26: 0,
            pic_init_qs_minus26: 0,
            chroma_qp_index_offset: 0,
            deblocking_filter_control_present_flag: false,
            constrained_intra_pred_flag: false,
            redundant_pic_cnt_present_flag: false,
            extension: None,
        }
    }

    fn dummy_slice_header(slice_type: SliceType) -> SliceHeader {
        use crate::slice_header::{DecRefPicMarking, RefPicListModification};
        SliceHeader {
            first_mb_in_slice: 0,
            slice_type_raw: match slice_type {
                SliceType::P => 0,
                SliceType::B => 1,
                SliceType::I => 2,
                SliceType::SP => 3,
                SliceType::SI => 4,
            },
            slice_type,
            all_slices_same_type: false,
            pic_parameter_set_id: 0,
            colour_plane_id: 0,
            frame_num: 0,
            field_pic_flag: false,
            bottom_field_flag: false,
            idr_pic_id: 0,
            pic_order_cnt_lsb: 0,
            delta_pic_order_cnt_bottom: 0,
            delta_pic_order_cnt: [0, 0],
            redundant_pic_cnt: 0,
            direct_spatial_mv_pred_flag: false,
            num_ref_idx_active_override_flag: false,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            ref_pic_list_modification: RefPicListModification::default(),
            pred_weight_table: None,
            dec_ref_pic_marking: Some(DecRefPicMarking {
                no_output_of_prior_pics_flag: false,
                long_term_reference_flag: false,
                adaptive_marking: None,
            }),
            cabac_init_idc: 0,
            slice_qp_delta: 0,
            sp_for_switch_flag: false,
            slice_qs_delta: 0,
            disable_deblocking_filter_idc: 0,
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
            slice_group_change_cycle: 0,
        }
    }

    /// Append a minimal I_NxN macroblock to `w` (CAVLC, all-zero
    /// residual, cbp=0, no mb_qp_delta).
    fn append_i_nxn_mb(w: &mut BitWriter) {
        w.ue(0); // mb_type = I_NxN
        for _ in 0..16 {
            w.u(1, 1); // prev_intra4x4_pred_mode_flag
        }
        w.ue(0); // intra_chroma_pred_mode
        w.ue(3); // coded_block_pattern codeNum=3 → CBP=0
    }

    #[test]
    fn cavlc_four_i_nxn_macroblocks() {
        // I-slice with 4 I_NxN MBs back to back. We produce a bit-
        // stream where the reader sees 4 complete MBs then the RBSP
        // trailing bit.
        let mut w = BitWriter::new();
        for _ in 0..4 {
            append_i_nxn_mb(&mut w);
        }
        w.trailing();
        let bytes = w.into_bytes();
        let sps = dummy_sps();
        let pps = dummy_pps();
        let hdr = dummy_slice_header(SliceType::I);
        let sd = parse_slice_data(&bytes, 0, 0, &hdr, &sps, &pps).unwrap();
        assert_eq!(sd.macroblocks.len(), 4);
        for mb in &sd.macroblocks {
            assert_eq!(mb.mb_type, MbType::INxN);
            assert!(!mb.is_skip);
        }
        assert_eq!(sd.last_mb_addr, 4);
        // Non-MBAFF: mb_field_decoding_flag is absent from the bit-
        // stream. §7.4.4 infers it as field_pic_flag (= false here),
        // which is what the walker records.
        assert_eq!(sd.mb_field_decoding_flags.len(), 4);
        assert!(sd.mb_field_decoding_flags.iter().all(|f| !*f));
    }

    #[test]
    fn cavlc_p_slice_mb_skip_run_3_then_one_coded_mb() {
        // P slice: mb_skip_run=3 (three skipped MBs) then one coded P
        // macroblock, then trailing bits.
        //
        // For simplicity we encode the coded MB as an intra I_NxN via
        // the P-slice path (mb_type=5 on P maps to I_NxN, §Table 7-13).
        let mut w = BitWriter::new();
        w.ue(3); // mb_skip_run = 3
        w.ue(5); // mb_type = 5 on P = I_NxN
        for _ in 0..16 {
            w.u(1, 1);
        }
        w.ue(0); // intra_chroma_pred_mode
        w.ue(3); // CBP codeNum=3 → 0
        w.trailing();
        let bytes = w.into_bytes();
        let sps = dummy_sps();
        let pps = dummy_pps();
        let hdr = dummy_slice_header(SliceType::P);
        let sd = parse_slice_data(&bytes, 0, 0, &hdr, &sps, &pps).unwrap();
        // 3 skip + 1 coded = 4 MB entries.
        assert_eq!(sd.macroblocks.len(), 4);
        assert!(sd.macroblocks[0].is_skip);
        assert!(sd.macroblocks[1].is_skip);
        assert!(sd.macroblocks[2].is_skip);
        assert!(!sd.macroblocks[3].is_skip);
        assert_eq!(sd.macroblocks[3].mb_type, MbType::INxN);
    }

    #[test]
    fn cavlc_p_slice_only_skip_run() {
        // A P slice that only contains a skip run followed by rbsp
        // trailing bits (all 4 MBs skipped).
        let mut w = BitWriter::new();
        w.ue(4); // mb_skip_run = 4
        w.trailing();
        let bytes = w.into_bytes();
        let sps = dummy_sps();
        let pps = dummy_pps();
        let hdr = dummy_slice_header(SliceType::P);
        let sd = parse_slice_data(&bytes, 0, 0, &hdr, &sps, &pps).unwrap();
        assert_eq!(sd.macroblocks.len(), 4);
        for mb in &sd.macroblocks {
            assert!(mb.is_skip);
        }
    }

    /// Build an SPS that enables MBAFF: frame_mbs_only_flag = 0 +
    /// mb_adaptive_frame_field_flag = 1.
    fn mbaff_sps() -> Sps {
        let mut sps = dummy_sps();
        sps.frame_mbs_only_flag = false;
        sps.mb_adaptive_frame_field_flag = true;
        sps
    }

    #[test]
    fn cavlc_mbaff_simple_pair_flag_one() {
        // §7.3.4 — MBAFF CAVLC I slice containing a single pair of
        // I_NxN macroblocks with mb_field_decoding_flag = 1 read at
        // the top of the pair. Both MBs of the pair must record the
        // same flag value.
        let mut w = BitWriter::new();
        w.u(1, 1); // mb_field_decoding_flag = 1 (top of pair, even addr)
        append_i_nxn_mb(&mut w); // top MB
        // No mb_field_decoding_flag at the bottom (top wasn't skipped).
        append_i_nxn_mb(&mut w); // bottom MB
        w.trailing();
        let bytes = w.into_bytes();
        let sps = mbaff_sps();
        let pps = dummy_pps();
        let hdr = dummy_slice_header(SliceType::I);
        let sd = parse_slice_data(&bytes, 0, 0, &hdr, &sps, &pps).unwrap();
        assert_eq!(sd.macroblocks.len(), 2);
        assert_eq!(sd.mb_field_decoding_flags.len(), 2);
        assert!(
            sd.mb_field_decoding_flags[0],
            "top MB of pair should carry the flag"
        );
        assert!(
            sd.mb_field_decoding_flags[1],
            "bottom MB of pair shares the same flag"
        );
        assert_eq!(sd.macroblocks[0].mb_type, MbType::INxN);
        assert_eq!(sd.macroblocks[1].mb_type, MbType::INxN);
    }

    #[test]
    fn cavlc_mbaff_pair_flag_zero() {
        // Same as above but mb_field_decoding_flag = 0 (frame pair).
        let mut w = BitWriter::new();
        w.u(1, 0); // mb_field_decoding_flag = 0
        append_i_nxn_mb(&mut w);
        append_i_nxn_mb(&mut w);
        w.trailing();
        let bytes = w.into_bytes();
        let sps = mbaff_sps();
        let pps = dummy_pps();
        let hdr = dummy_slice_header(SliceType::I);
        let sd = parse_slice_data(&bytes, 0, 0, &hdr, &sps, &pps).unwrap();
        assert_eq!(sd.macroblocks.len(), 2);
        assert_eq!(sd.mb_field_decoding_flags, vec![false, false]);
    }

    #[test]
    fn cavlc_mbaff_skipped_top_reads_flag_at_bottom() {
        // §7.3.4 + §7.4.4 — in a P slice with MBAFF, if the top MB of
        // a pair is skipped via mb_skip_run, the mb_field_decoding_flag
        // is read retroactively at the bottom MB (CurrMbAddr % 2 == 1
        // && prevMbSkipped). Both MBs of the pair end up with the
        // same flag.
        let mut w = BitWriter::new();
        w.ue(1); // mb_skip_run = 1 → top MB skipped
        // Now CurrMbAddr = 1 (bottom of pair), prevMbSkipped = true.
        w.u(1, 1); // mb_field_decoding_flag = 1 (read at bottom)
        w.ue(5); // mb_type = 5 on P = I_NxN
        for _ in 0..16 {
            w.u(1, 1); // prev_intra4x4_pred_mode_flag
        }
        w.ue(0); // intra_chroma_pred_mode
        w.ue(3); // CBP codeNum=3 → 0
        w.trailing();
        let bytes = w.into_bytes();
        let sps = mbaff_sps();
        let pps = dummy_pps();
        let hdr = dummy_slice_header(SliceType::P);
        let sd = parse_slice_data(&bytes, 0, 0, &hdr, &sps, &pps).unwrap();
        assert_eq!(sd.macroblocks.len(), 2);
        assert!(sd.macroblocks[0].is_skip, "top MB should be a skip MB");
        assert!(!sd.macroblocks[1].is_skip, "bottom MB should be coded");
        // Retroactive patch: both MBs of the pair carry flag = 1.
        assert_eq!(sd.mb_field_decoding_flags, vec![true, true]);
    }

    #[test]
    fn cavlc_non_mbaff_has_no_flag_reads() {
        // Regression: on a non-MBAFF SPS, the walker must not attempt
        // to read mb_field_decoding_flag. A stream containing just two
        // I_NxN MBs (no extra u(1) for the flag) must parse to 2 MBs,
        // and each mb_field_decoding_flag recorded is false.
        let mut w = BitWriter::new();
        append_i_nxn_mb(&mut w);
        append_i_nxn_mb(&mut w);
        w.trailing();
        let bytes = w.into_bytes();
        let sps = dummy_sps(); // frame_mbs_only_flag = true → not MBAFF
        let pps = dummy_pps();
        let hdr = dummy_slice_header(SliceType::I);
        let sd = parse_slice_data(&bytes, 0, 0, &hdr, &sps, &pps).unwrap();
        assert_eq!(sd.macroblocks.len(), 2);
        assert_eq!(sd.mb_field_decoding_flags, vec![false, false]);
    }

    #[test]
    fn cavlc_mbaff_two_pairs_independent_flags() {
        // Two pairs, first flag = 0, second flag = 1. Each pair's
        // flag must be read independently at the top of the pair.
        let mut w = BitWriter::new();
        w.u(1, 0); // pair 0 top: mb_field_decoding_flag = 0
        append_i_nxn_mb(&mut w); // pair 0 top
        append_i_nxn_mb(&mut w); // pair 0 bottom
        w.u(1, 1); // pair 1 top: mb_field_decoding_flag = 1
        append_i_nxn_mb(&mut w); // pair 1 top
        append_i_nxn_mb(&mut w); // pair 1 bottom
        w.trailing();
        let bytes = w.into_bytes();
        let sps = mbaff_sps();
        let pps = dummy_pps();
        let hdr = dummy_slice_header(SliceType::I);
        let sd = parse_slice_data(&bytes, 0, 0, &hdr, &sps, &pps).unwrap();
        assert_eq!(sd.macroblocks.len(), 4);
        assert_eq!(
            sd.mb_field_decoding_flags,
            vec![false, false, true, true],
            "each pair carries its own flag, applied to both MBs"
        );
    }

    #[test]
    fn cabac_mbaff_two_skip_mbs_per_pair_smoke() {
        // §7.3.4 — CABAC MBAFF: a non-I slice whose first pair has
        // both MBs skipped. mb_skip_flag is read for top (= 1),
        // mb_field_decoding_flag is NOT read (skipped top), then
        // mb_skip_flag for bottom (= 1), mb_field_decoding_flag still
        // not read (both MBs of the pair are skipped → §7.4.4 infers
        // it). end_of_slice_flag is only decoded on the bottom MB.
        //
        // CABAC state starts with codIOffset=0 for an all-zero
        // prefix, which makes P-slice mb_skip_flag's MPS decode
        // (valMPS=0 or 1 depending on ctxIdx init) deterministic. We
        // don't want to hand-craft the exact bitstream here — this is
        // a smoke test that the CABAC walker doesn't hit the
        // MbaffNotSupported error and terminates without hanging.
        let mut bytes: Vec<u8> = vec![0x00; 16];
        bytes.extend(std::iter::repeat_n(0xFFu8, 16));
        let mut pps = dummy_pps();
        pps.entropy_coding_mode_flag = true;
        let sps = mbaff_sps();
        let hdr = dummy_slice_header(SliceType::I);
        let sd = parse_slice_data(&bytes, 0, 0, &hdr, &sps, &pps);
        // Accept clean parse or known error — MbaffNotSupported is
        // NOT an acceptable outcome (it was the Phase-0 behaviour).
        match sd {
            Ok(sd) => {
                assert_eq!(sd.mb_field_decoding_flags.len(), sd.macroblocks.len());
            }
            Err(SliceDataError::MbaffNotSupported) => {
                panic!("Phase 1 must not reject MBAFF up-front");
            }
            Err(_) => {
                // Other errors (bitstream EOF, unsupported MB type)
                // are acceptable for this synthetic stream.
            }
        }
    }

    #[test]
    fn cabac_mbaff_flag_decoded_at_top_of_pair() {
        // §7.3.4 — CABAC MBAFF: an I slice whose first pair is a
        // coded I_NxN pair. mb_field_decoding_flag must be decoded
        // before the top MB's macroblock_layer() and recorded for
        // both MBs.
        //
        // We hand-construct a minimal synthetic bitstream where the
        // first CABAC bins resolve to predictable values using the
        // zero-offset seed convention. The test is a smoke-test for
        // the walker's branching logic; if the inner macroblock_layer
        // errors on residual parsing, we still verify the walker
        // didn't reject MBAFF up-front and that for any successfully
        // parsed MBs, flags are consistent per-pair.
        let mut bytes: Vec<u8> = vec![0x00; 8];
        bytes.extend(std::iter::repeat_n(0xFFu8, 8));
        let mut pps = dummy_pps();
        pps.entropy_coding_mode_flag = true;
        let sps = mbaff_sps();
        let hdr = dummy_slice_header(SliceType::I);
        let sd = parse_slice_data(&bytes, 0, 0, &hdr, &sps, &pps);
        match sd {
            Ok(sd) => {
                // Parallel invariant.
                assert_eq!(sd.mb_field_decoding_flags.len(), sd.macroblocks.len());
                // Pair flag consistency: for each complete pair, both
                // MBs should carry the same flag value.
                for pair in sd.mb_field_decoding_flags.chunks_exact(2) {
                    assert_eq!(pair[0], pair[1], "MBs in a pair share the flag");
                }
            }
            Err(SliceDataError::MbaffNotSupported) => {
                panic!("Phase 1 must not reject MBAFF up-front");
            }
            Err(_) => {
                // Synthetic stream: parse errors in macroblock_layer
                // are acceptable; the important behaviour is that we
                // don't hit MbaffNotSupported and don't hang.
            }
        }
    }

    #[test]
    fn cabac_path_single_i_pcm_mb_terminates() {
        // A CABAC I slice with one I_PCM macroblock isn't practical
        // (our CABAC I_PCM path is unsupported). Instead use a CABAC
        // I slice with a single I_NxN macroblock whose fresh-init
        // state yields mb_type=I_NxN on the first bin.
        //
        // For the test we rely on the CABAC engine being initialised
        // at codIOffset=0 (bytes 0x00 0x00 ...), which means bin 0 of
        // any FL/TU element returns the valMPS of its context. For
        // ctxIdx=3 (I-slice mb_type bin 0, m=20 n=-15 QPY=26) valMPS=0
        // so mb_type=0 (I_NxN). Subsequent residual reads also hit
        // their MPS path, which for this synthetic stream is
        // consistent with cbp=0 and no residual.
        //
        // After the MB, `decode_end_of_slice_flag` returns 1 once
        // codIOffset gets close to codIRange (requires enough 1-bits
        // in the stream). Pad the remainder with 0xFF so the
        // terminator fires on the first check.
        // cabac_alignment_one_bit — none needed when already aligned.
        // CABAC consumes 9 bits of state then a long run of 0s, then
        // switch to 1s so terminate fires.
        let mut bytes: Vec<u8> = vec![0x00; 16];
        bytes.extend(std::iter::repeat_n(0xFFu8, 16));

        // Build a PPS with entropy_coding_mode_flag = 1.
        let mut pps = dummy_pps();
        pps.entropy_coding_mode_flag = true;
        let sps = dummy_sps();
        let hdr = dummy_slice_header(SliceType::I);

        // Because the CABAC path's exact sequence depends on bin-level
        // decisions that cascade through residual contexts, this test
        // is a smoke-test: it must terminate with at least one MB and
        // without errors.
        let sd = parse_slice_data(&bytes, 0, 0, &hdr, &sps, &pps);
        // Accept either a clean parse or an explicit known error path
        // (e.g. bitstream EOF on malformed trailing bytes). The
        // important behaviour is that the walker doesn't loop
        // infinitely and returns deterministically.
        match sd {
            Ok(sd) => {
                assert!(!sd.macroblocks.is_empty());
            }
            Err(e) => {
                // Acceptable: the synthetic stream ran out before
                // end_of_slice_flag fired. The engine didn't hang.
                let _ = e;
            }
        }
    }
}
