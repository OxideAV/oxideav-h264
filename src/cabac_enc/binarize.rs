//! Per-syntax CABAC **encoders** — mirrors of `crate::cabac::binarize::decode_*`.
//!
//! For every syntax element the encoder emits the bin-string derived by
//! §9.3.2 (binarisation) through [`CabacEncoder::encode_bin`] (regular
//! mode) / [`CabacEncoder::encode_bypass`] (bypass) in the exact order
//! the decoder expects to read them. Context selection (ctxIdxInc) is
//! passed in as a number by the caller (same convention as the decoder),
//! so the shared `crate::cabac::mb` helpers drive both directions.

use oxideav_core::{Error, Result};

use crate::cabac::context::CabacContext;

use super::engine::CabacEncoder;

/// §9.3.3.1.1.6 — `prev_intra4x4_pred_mode_flag` / `prev_intra8x8_pred_mode_flag`.
/// Single regular-mode bin with fixed ctxIdxInc = 0.
pub fn encode_prev_intra4x4_pred_mode_flag(
    enc: &mut CabacEncoder,
    ctxs: &mut [CabacContext],
    flag: bool,
) -> Result<()> {
    if ctxs.is_empty() {
        return Err(Error::invalid(
            "cabac_enc::binarize::encode_prev_intra4x4_pred_mode_flag: empty ctxs",
        ));
    }
    enc.encode_bin(&mut ctxs[0], flag as u8);
    Ok(())
}

/// §9.3.3.1.1.7 — `rem_intra4x4_pred_mode` (FL(cMax=7) in REGULAR mode).
/// All three bins share the same context (§9.3.3.1.1.7 Table 9-34, slot
/// ctxIdxOffset 276 + ctxIdxInc 0 = absolute ctxIdx 69). Bit 0 is emitted
/// first per FFmpeg's `mode += (1<<binIdx) * bin` accumulator — i.e. LSB
/// first of the 3-bit FL, matching the decoder.
pub fn encode_rem_intra4x4_pred_mode(
    enc: &mut CabacEncoder,
    ctxs: &mut [CabacContext],
    rem: u32,
) -> Result<()> {
    if ctxs.is_empty() {
        return Err(Error::invalid(
            "cabac_enc::binarize::encode_rem_intra4x4_pred_mode: empty ctxs",
        ));
    }
    if rem >= 8 {
        return Err(Error::invalid(format!(
            "cabac_enc::binarize::encode_rem_intra4x4_pred_mode: rem {rem} >= 8"
        )));
    }
    for i in 0..3u32 {
        let bit = ((rem >> i) & 1) as u8;
        enc.encode_bin(&mut ctxs[0], bit);
    }
    Ok(())
}

/// §9.3.3.1.1.8 — `intra_chroma_pred_mode` (truncated unary, cMax = 3).
/// Bin 0 uses `ctxs[ctx_idx_inc_a_b]`; bins 1..=2 share `ctxs[shared]`
/// where `shared = ctxs.len() - 1` (the last slot in the 4-wide bank).
pub fn encode_intra_chroma_pred_mode(
    enc: &mut CabacEncoder,
    ctxs: &mut [CabacContext],
    ctx_idx_inc_a_b: u8,
    mode: u32,
) -> Result<()> {
    if ctxs.len() < 2 {
        return Err(Error::invalid(
            "cabac_enc::binarize::encode_intra_chroma_pred_mode: need >=2 ctxs",
        ));
    }
    if mode > 3 {
        return Err(Error::invalid(format!(
            "cabac_enc::binarize::encode_intra_chroma_pred_mode: mode {mode} > 3"
        )));
    }
    let inc0 = ctx_idx_inc_a_b as usize;
    if inc0 >= ctxs.len() {
        return Err(Error::invalid(
            "cabac_enc::binarize::encode_intra_chroma_pred_mode: ctx_idx_inc out of range",
        ));
    }
    let shared = ctxs.len() - 1;
    match mode {
        0 => {
            enc.encode_bin(&mut ctxs[inc0], 0);
        }
        1 => {
            enc.encode_bin(&mut ctxs[inc0], 1);
            enc.encode_bin(&mut ctxs[shared], 0);
        }
        2 => {
            enc.encode_bin(&mut ctxs[inc0], 1);
            enc.encode_bin(&mut ctxs[shared], 1);
            enc.encode_bin(&mut ctxs[shared], 0);
        }
        3 => {
            enc.encode_bin(&mut ctxs[inc0], 1);
            enc.encode_bin(&mut ctxs[shared], 1);
            enc.encode_bin(&mut ctxs[shared], 1);
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// §9.3.3.1.1.3 + Table 9-36 — I-slice `mb_type` encoder. Emits the bin
/// tree matching `decode_mb_type_i`:
///
/// * `mb_type == 0` → `I_NxN` → emit one 0 bin at `ctxs[inc0]`.
/// * `mb_type == 25` → `I_PCM` → emit 1 then a terminate(1) bin.
/// * 1..=24 → I_16x16 with fields derived from `idx = mb_type - 1`:
///   emit prefix = 1, terminate(0), then 5 regular bins at contexts
///   3/4/5/6/7 per Table 9-31.
pub fn encode_mb_type_i(
    enc: &mut CabacEncoder,
    ctxs: &mut [CabacContext],
    ctx_idx_inc_a_b: u8,
    mb_type: u32,
) -> Result<()> {
    if ctxs.len() < 8 {
        return Err(Error::invalid(
            "cabac_enc::binarize::encode_mb_type_i: need >=8 I-slice mb_type ctxs",
        ));
    }
    let inc0 = ctx_idx_inc_a_b as usize;
    if inc0 >= ctxs.len() {
        return Err(Error::invalid(
            "cabac_enc::binarize::encode_mb_type_i: ctx_idx_inc out of range",
        ));
    }
    if mb_type == 0 {
        enc.encode_bin(&mut ctxs[inc0], 0);
        return Ok(());
    }
    enc.encode_bin(&mut ctxs[inc0], 1);
    if mb_type == 25 {
        // I_PCM path — terminate(1) plays the role of bin 1.
        enc.encode_terminate(1);
        return Ok(());
    }
    if !(1..=24).contains(&mb_type) {
        return Err(Error::invalid(format!(
            "cabac_enc::binarize::encode_mb_type_i: bad mb_type {mb_type}"
        )));
    }
    // Non-terminate for the I_16×16 path.
    enc.encode_terminate(0);
    // Decompose mb_type - 1:
    //   pred = idx % 4          (Intra16x16PredMode, 0..=3)
    //   cbp_chroma = (idx/4) % 3
    //   cbp_luma = 0 if idx < 12 else 15
    let idx = mb_type - 1;
    let pred = idx % 4;
    let cbp_chroma_class = (idx / 4) % 3;
    let cbp_luma_flag = if idx >= 12 { 1u32 } else { 0 };
    // bin 2 (ctx 3): cbp_luma bit.
    enc.encode_bin(&mut ctxs[3], cbp_luma_flag as u8);
    // bin 3 (ctx 4): cbp_chroma bit 0.
    let ch0 = if cbp_chroma_class == 0 { 0u8 } else { 1 };
    enc.encode_bin(&mut ctxs[4], ch0);
    if ch0 == 1 {
        // bin 4 (ctx 5): cbp_chroma bit 1 (chroma class 1 vs 2).
        let ch1 = if cbp_chroma_class == 1 { 0u8 } else { 1 };
        enc.encode_bin(&mut ctxs[5], ch1);
    }
    // bins 5/6 (ctx 6/7): pred mode MSB then LSB.
    let p_hi = ((pred >> 1) & 1) as u8;
    let p_lo = (pred & 1) as u8;
    enc.encode_bin(&mut ctxs[6], p_hi);
    enc.encode_bin(&mut ctxs[7], p_lo);
    Ok(())
}

/// §9.3.3.1.1.5 Table 9-40 — `mb_qp_delta` encoder. Mirrors the decoder's
/// unary remapping `0, 1, -1, 2, -2, 3, -3, …`.
pub fn encode_mb_qp_delta(
    enc: &mut CabacEncoder,
    ctxs: &mut [CabacContext],
    ctx_idx_inc_prev_nonzero: u8,
    dqp: i32,
) -> Result<()> {
    if ctxs.len() < 4 {
        return Err(Error::invalid(
            "cabac_enc::binarize::encode_mb_qp_delta: need >=4 ctxs",
        ));
    }
    let inc0 = ctx_idx_inc_prev_nonzero as usize;
    if inc0 >= ctxs.len() {
        return Err(Error::invalid(
            "cabac_enc::binarize::encode_mb_qp_delta: ctx_idx_inc out of range",
        ));
    }
    if dqp == 0 {
        enc.encode_bin(&mut ctxs[inc0], 0);
        return Ok(());
    }
    enc.encode_bin(&mut ctxs[inc0], 1);
    // Invert the decoder's mapping:
    //   k = prefix length. dqp > 0 → k odd, magnitude = (k+1)/2.
    //   dqp < 0 → k even, magnitude = k/2.
    // Equivalent:
    //   k = 2*|dqp| - (dqp > 0 ? 1 : 0).
    let k = if dqp > 0 {
        (2 * dqp - 1) as u32
    } else {
        (-2 * dqp) as u32
    };
    // Prefix is `k` ones then a 0. The first bin (= bin 1 of the whole
    // mb_qp_delta bin string) is on ctxs[2]; subsequent bins saturate at
    // ctxs[3] (§9.3.3.1.1.5).
    if k == 1 {
        enc.encode_bin(&mut ctxs[2], 0);
        return Ok(());
    }
    enc.encode_bin(&mut ctxs[2], 1);
    // k >= 2: emit (k-2) ones then a zero on ctxs[3].
    for _ in 1..(k - 1) {
        enc.encode_bin(&mut ctxs[3], 1);
    }
    enc.encode_bin(&mut ctxs[3], 0);
    Ok(())
}

/// §9.3.3.1.1.9 Table 9-41 — emit `coded_block_flag`.
pub fn encode_coded_block_flag(enc: &mut CabacEncoder, ctx: &mut CabacContext, cbf: bool) {
    enc.encode_bin(ctx, cbf as u8);
}

/// §9.3.3.1.1.9 — emit one `significant_coeff_flag` bin.
pub fn encode_significant_coeff_flag(enc: &mut CabacEncoder, ctx: &mut CabacContext, sig: bool) {
    enc.encode_bin(ctx, sig as u8);
}

/// §9.3.3.1.1.9 — emit one `last_significant_coeff_flag` bin.
pub fn encode_last_significant_coeff_flag(
    enc: &mut CabacEncoder,
    ctx: &mut CabacContext,
    last: bool,
) {
    enc.encode_bin(ctx, last as u8);
}

/// §9.3.3.1.1.9 / §9.3.2.3 — emit `coeff_abs_level_minus1` (UEGk with
/// `k = 0, uCoff = 14`). Mirror of
/// `cabac::residual::decode_coeff_abs_level_minus1`.
///
/// `value` is `abs_level - 1` (so `abs_level = 1` ⇒ `value = 0`).
/// `ctxs` is the same 10-slot abs-level bank the decoder sees at
/// `ctxs[33..=42]` — caller must supply it with the correct `num_eq1` /
/// `num_gt1` driven ctxIdxInc selection applied externally.
///
/// The function returns `(delta_eq1, delta_gt1)` so callers can maintain
/// the Table 9-43 stats counters exactly like the decoder does.
pub fn encode_coeff_abs_level_minus1(
    enc: &mut CabacEncoder,
    abs_ctxs: &mut [CabacContext],
    num_eq1: u32,
    num_gt1: u32,
    value: u32,
) -> Result<(u32, u32)> {
    if abs_ctxs.len() < 10 {
        return Err(Error::invalid(
            "cabac_enc::binarize::encode_coeff_abs_level_minus1: need >=10 abs-level ctxs",
        ));
    }
    let bin0_inc = if num_gt1 != 0 {
        0u32
    } else {
        core::cmp::min(4, 1 + num_eq1)
    };
    let binge1_inc = 5 + core::cmp::min(4u32, num_gt1);

    if value == 0 {
        enc.encode_bin(&mut abs_ctxs[bin0_inc as usize], 0);
        return Ok((1, 0));
    }
    enc.encode_bin(&mut abs_ctxs[bin0_inc as usize], 1);
    // Prefix: TU(14). After bin0 (already a 1), emit (prefix_len - 1) more
    // ones then a terminating zero IF prefix_len < 14. Saturated prefix
    // (prefix_len == 14) emits 13 more ones then no terminator.
    let prefix_len = value.min(14);
    for _ in 1..prefix_len {
        enc.encode_bin(&mut abs_ctxs[binge1_inc as usize], 1);
    }
    if prefix_len < 14 {
        enc.encode_bin(&mut abs_ctxs[binge1_inc as usize], 0);
        return Ok((0, 1));
    }
    // Saturated path — emit EG0 suffix in bypass for `suffix = value - 14`.
    let suffix = value - 14;
    let mut k: u32 = 0;
    while (1u32 << (k + 1)) - 1 <= suffix {
        k = k.checked_add(1).ok_or_else(|| {
            Error::invalid("cabac_enc::binarize::encode_coeff_abs_level_minus1: k overflow")
        })?;
        if k > 31 {
            return Err(Error::invalid(
                "cabac_enc::binarize::encode_coeff_abs_level_minus1: suffix too large",
            ));
        }
    }
    // EG0 prefix: `k` one-bins then a zero — the decoder's bypass loop
    // reads ones while the next bin is 1, breaking on the first 0.
    for _ in 0..k {
        enc.encode_bypass(1);
    }
    enc.encode_bypass(0);
    // EG0 tail: k bypass bits of `extra`.
    let extra = suffix - ((1u32 << k) - 1);
    for i in (0..k).rev() {
        enc.encode_bypass(((extra >> i) & 1) as u8);
    }
    Ok((0, 1))
}

/// §9.3.3.1.1.9 — emit `coeff_sign_flag` (bypass bin, 0 positive, 1 negative).
pub fn encode_coeff_sign_flag(enc: &mut CabacEncoder, sign_negative: bool) {
    enc.encode_bypass(sign_negative as u8);
}

/// §9.3.3.1.1.1 — emit `mb_skip_flag` for a P/SP slice. Single regular
/// bin at `ctxs[ctx_idx_inc]` (ctx_idx_inc derived from the neighbour
/// skipped state).
pub fn encode_mb_skip_flag_p(
    enc: &mut CabacEncoder,
    ctxs: &mut [CabacContext],
    ctx_idx_inc: u8,
    skip: bool,
) -> Result<()> {
    if ctxs.len() < 3 {
        return Err(Error::invalid(
            "cabac_enc::binarize::encode_mb_skip_flag_p: need >=3 ctxs",
        ));
    }
    let inc = (ctx_idx_inc as usize).min(2);
    enc.encode_bin(&mut ctxs[inc], skip as u8);
    Ok(())
}

/// §9.3.3.1.1.4 Table 9-39 — `coded_block_pattern` encoder.
///
/// Shape mirrors `decode_coded_block_pattern`: 4 luma sub-block bins
/// (each with its own ctxIdxInc from the caller-supplied callback), then
/// the chroma 2-bin tree (0 = none; 10 = DC only; 11 = DC + AC).
///
/// `chroma_format_idc == 0` skips chroma entirely.
pub fn encode_coded_block_pattern(
    enc: &mut CabacEncoder,
    ctxs: &mut [CabacContext],
    chroma_format_idc: u8,
    cbp_luma: u8,
    cbp_chroma: u8,
    mut luma_inc_of: impl FnMut(usize, u8) -> u8,
    mut chroma_inc_of: impl FnMut(usize) -> u8,
) -> Result<()> {
    if ctxs.len() < 12 {
        return Err(Error::invalid(
            "cabac_enc::binarize::encode_coded_block_pattern: need >=12 CBP ctxs",
        ));
    }
    if cbp_luma > 0x0F {
        return Err(Error::invalid(
            "cabac_enc::binarize::encode_coded_block_pattern: cbp_luma > 0x0F",
        ));
    }
    if cbp_chroma > 2 {
        return Err(Error::invalid(
            "cabac_enc::binarize::encode_coded_block_pattern: cbp_chroma > 2",
        ));
    }

    let mut running: u8 = 0;
    for i in 0..4usize {
        let inc = luma_inc_of(i, running) as usize;
        if inc >= 4 {
            return Err(Error::invalid(
                "cabac_enc::binarize::encode_coded_block_pattern: luma ctx inc out of range",
            ));
        }
        let bit = (cbp_luma >> i) & 1;
        enc.encode_bin(&mut ctxs[inc], bit);
        running |= bit << i;
    }
    if chroma_format_idc != 0 {
        let inc0 = chroma_inc_of(0) as usize;
        if inc0 >= 4 {
            return Err(Error::invalid(
                "cabac_enc::binarize::encode_coded_block_pattern: chroma ctx inc[0] out of range",
            ));
        }
        let any = (cbp_chroma != 0) as u8;
        enc.encode_bin(&mut ctxs[4 + inc0], any);
        if any == 1 {
            let inc1 = chroma_inc_of(1) as usize;
            if inc1 >= 4 {
                return Err(Error::invalid(
                    "cabac_enc::binarize::encode_coded_block_pattern: chroma ctx inc[1] out of range",
                ));
            }
            let ac = (cbp_chroma == 2) as u8;
            enc.encode_bin(&mut ctxs[8 + inc1], ac);
        }
    }
    Ok(())
}

// ==========================================================================
// Tests — encode via the CABAC encoder + decode via the existing decoder
// and assert the symbol is recovered identically.
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::binarize as dec_bin;
    use crate::cabac::engine::CabacDecoder;

    fn fresh_ctxs(n: usize) -> Vec<CabacContext> {
        vec![CabacContext::default(); n]
    }

    #[test]
    fn mb_type_i_roundtrip_all_values() {
        // Encode every legal I-slice mb_type (0..=25) separately and
        // decode it back through the existing decoder — symbol identity
        // is the core property the scope asks for.
        for mb_type in 0..=25u32 {
            let mut enc = CabacEncoder::new();
            let mut ctxs = fresh_ctxs(8);
            encode_mb_type_i(&mut enc, &mut ctxs, 0, mb_type).unwrap();
            enc.encode_terminate(1);
            let bytes = enc.finish();

            let mut dec = CabacDecoder::new(&bytes, 0).unwrap();
            let mut dec_ctxs = fresh_ctxs(8);
            let got = dec_bin::decode_mb_type_i(&mut dec, &mut dec_ctxs, 0).unwrap();
            assert_eq!(got, mb_type, "mb_type {mb_type} round-trip mismatch");
        }
    }

    #[test]
    fn intra_chroma_pred_mode_roundtrip_all_values() {
        for mode in 0..=3u32 {
            let mut enc = CabacEncoder::new();
            let mut ctxs = fresh_ctxs(4);
            encode_intra_chroma_pred_mode(&mut enc, &mut ctxs, 0, mode).unwrap();
            enc.encode_terminate(1);
            let bytes = enc.finish();

            let mut dec = CabacDecoder::new(&bytes, 0).unwrap();
            let mut dec_ctxs = fresh_ctxs(4);
            let got = dec_bin::decode_intra_chroma_pred_mode(&mut dec, &mut dec_ctxs, 0).unwrap();
            assert_eq!(got, mode, "chroma mode {mode} round-trip mismatch");
        }
    }

    #[test]
    fn mb_qp_delta_roundtrip_small_range() {
        for dqp in -5..=5i32 {
            let mut enc = CabacEncoder::new();
            let mut ctxs = fresh_ctxs(4);
            encode_mb_qp_delta(&mut enc, &mut ctxs, 0, dqp).unwrap();
            enc.encode_terminate(1);
            let bytes = enc.finish();

            let mut dec = CabacDecoder::new(&bytes, 0).unwrap();
            let mut dec_ctxs = fresh_ctxs(4);
            let got = dec_bin::decode_mb_qp_delta(&mut dec, &mut dec_ctxs, 0).unwrap();
            assert_eq!(got, dqp, "dqp {dqp} round-trip mismatch");
        }
    }

    #[test]
    fn coeff_abs_level_minus1_roundtrip_small_values() {
        // Values 0..=20 cover both the non-saturated TU branch and the
        // EGk tail (value >= 14).
        for value in 0..=20u32 {
            let mut enc = CabacEncoder::new();
            let mut ctxs = fresh_ctxs(64);
            // Base ctxs starts at slot 33 in the decoder's "ctxs" layout;
            // here we pass a slice starting from 33 directly — encode &
            // decode must agree on the offset.
            encode_coeff_abs_level_minus1(&mut enc, &mut ctxs[33..43], 0, 0, value).unwrap();
            enc.encode_terminate(1);
            let bytes = enc.finish();

            // The decoder's `decode_coeff_abs_level_minus1` is private, so
            // we invoke the public `decode_residual_block_cabac` via a
            // stub: set cbf = 1, sig[0] = 1, last[0] = 1, then the level.
            //
            // But the decoder API we have expects those bins to come
            // first. So for a clean binarize-only round-trip we re-derive
            // `value` from the bytes manually using the arithmetic decoder
            // against the same context indices.
            //
            // Simpler: decode the residual block's UEGk stream using the
            // public binarize helpers we already have — but they're for
            // OTHER syntax elements. Use UEGk wrapper via `decode_unary`
            // cannot emit mixed regular/bypass.
            //
            // Instead: drive the arithmetic decoder manually to undo the
            // exact bin sequence we emitted.
            let mut dec = CabacDecoder::new(&bytes, 0).unwrap();
            let mut dec_ctxs = fresh_ctxs(64);
            // Bin 0 at ctx[33 + bin0_inc]; we used num_eq1 = num_gt1 = 0
            // ⇒ bin0_inc = min(4, 1) = 1 → index 34. Bins 1..prefix_len
            // at ctx[33 + binge1_inc = 33 + 5] = 38.
            let b0 = dec.decode_bin(&mut dec_ctxs[34]).unwrap();
            if value == 0 {
                assert_eq!(b0, 0);
                continue;
            }
            assert_eq!(b0, 1, "value {value}: expected prefix start = 1");
            let mut got: u32 = 1;
            let prefix_cap = 14u32;
            let mut saturated = true;
            for _ in 1..prefix_cap {
                let b = dec.decode_bin(&mut dec_ctxs[38]).unwrap();
                if b == 0 {
                    saturated = false;
                    break;
                }
                got += 1;
            }
            let decoded_value = if saturated {
                // EG0 tail.
                let mut k: u32 = 0;
                loop {
                    let b = dec.decode_bypass().unwrap();
                    if b == 0 {
                        break;
                    }
                    k += 1;
                }
                let mut extra: u32 = 0;
                for _ in 0..k {
                    extra = (extra << 1) | dec.decode_bypass().unwrap() as u32;
                }
                14 + (1u32 << k) - 1 + extra
            } else {
                got
            };
            assert_eq!(decoded_value, value, "coeff abs value {value} round-trip");
        }
    }
}
