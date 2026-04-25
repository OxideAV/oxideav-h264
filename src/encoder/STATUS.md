# H.264 encoder — status

This module hosts the **clean-room H.264 encoder** rebuilt from scratch
against ITU-T Rec. H.264 (08/2024). No external implementation
(libavcodec / openh264 / x264 / JM) was consulted while writing it.
Same strict policy as the decoder.

## Round 1 — landed

Minimal Baseline I-only IDR encoder, end-to-end:

- **SPS / PPS emit** (Baseline, profile_idc=66, 4:2:0, no FMO, no VUI,
  CAVLC, single slice per picture).
- **Slice header emit** for IDR I-slice (POC type 0,
  `disable_deblocking_filter_idc=1`).
- **Macroblock**: every MB is `I_16x16` with `pred_mode=2` (DC luma)
  and `intra_chroma_pred_mode=0` (DC chroma). `cbp_luma=0`,
  `cbp_chroma=0` — only the luma DC block is transmitted (every
  Intra_16x16 MB carries it unconditionally per §7.3.5.3.1).
- **Forward 4x4 integer transform** (`Cf * X * Cf^T`) + Hadamard for
  luma DC + matched quantizer (JM-reference `MF` table, intra rounding
  offset = `2^qBits / 3`).
- **CAVLC encode** for the luma DC residual block. Re-uses the
  decoder's transcribed §9.2 tables (every table is a slice of
  `(bits, length, value)` rows, looked up by `value` on the encoder
  side). Round-trip-tested against `crate::cavlc::decode_*` for
  `coeff_token` / `level` / `total_zeros` / `run_before`.
- **Reconstruction loop** mirrors the decoder bit-exactly: same DC
  prediction (§8.3.3.3 / §8.3.4.1), same inverse transform, same
  `Clip1Y`/`Clip1C`. The encoder publishes the recon planes alongside
  the bitstream so tests can assert that
  `decode(encode(picture)) == encoder_recon`.

### Validation

- `cargo test -p oxideav-h264 --lib` — 749 passed.
- `cargo test -p oxideav-h264` — 851+ passed (29 new encoder unit
  tests + 2 integration tests).
- **Self-roundtrip**: encode a 64x64 testsrc → decode with our own
  decoder → exact match against encoder's local recon.
- **ffmpeg interop**: encode → write `.h264` → run `ffmpeg -i .h264 -f
  rawvideo -pix_fmt yuv420p .yuv` → exact match against encoder's
  local recon. Verified with ffmpeg 8.1 (`--enable-libx264`).
- **Self-roundtrip PSNR vs. original**: ~38.85 dB on the 64x64
  diagonal-gradient testsrc at QP 26. Every Intra_16x16 MB transmits
  one DC value per 4x4 sub-block, no AC.

### Status caveats

- I-only / IDR-only — no P / B slices.
- Single mb_type (I_16x16 with DC luma + DC chroma).
- No AC residual — `cbp_luma == 0` always.
- No chroma residual — `cbp_chroma == 0` always.
- No in-loop deblocking — `disable_deblocking_filter_idc == 1`.
- No CABAC — CAVLC only.
- No rate control — fixed QP from `EncoderConfig::qp`.

## Round 2 — landed

Lift encoder coverage from DC-only luma to multi-mode I_16x16 + chroma
residual transmit.

- **All four Intra_16x16 modes** (V / H / DC / Plane) per
  [`encoder::intra_pred`], with a per-MB SAD-driven mode decision.
  Modes whose required neighbours are missing (top edge → V/Plane,
  left edge → H/Plane, top-left corner → Plane) are skipped during
  the search; the encoder always falls back to DC, never signals an
  unavailable mode.
- **All four Intra_Chroma modes** (DC / H / V / Plane) selected per
  MB by joint-plane SAD (Cb + Cr).
- **Chroma residual transmit**: `cbp_chroma == 1` (DC only) when the
  chroma DC has any non-zero levels; promoted to `cbp_chroma == 2`
  (DC + AC) when any chroma 4x4 block has non-zero AC. Encoder
  computes the residual against the chosen predictor, runs forward
  4x4 → 2x2 chroma-DC Hadamard → quantize, and reconstructs through
  the same inverse path the decoder uses.
- **`zigzag_scan_4x4_ac`** added to the encoder transform module to
  pack 4x4 AC coefficients into the AC-only scan layout the decoder
  reads via `inverse_scan_4x4_zigzag_ac`.
- **`quantize_4x4_ac`**: AC-only quantizer (clears DC slot 0).

### Validation

- `cargo test -p oxideav-h264 --lib` — 758 passed.
- `cargo test -p oxideav-h264` — all integration tests pass:
  - `encode_decode_roundtrip_matches_local_recon`
  - `encode_decode_roundtrip_chroma_residual` (new)
  - `encode_mixed_picture_picks_non_dc_modes` (new — 128x128 mixed
    bars + gradient)
  - `ffmpeg_decode_matches_encoder_recon`
  - `ffmpeg_decode_chroma_residual_matches` (new)
- **PSNR (QP 26)**:
  - 64x64 diagonal-gradient testsrc, luma: **38.85 → 42.25 dB**
    (round-1 → round-2). Stream **148 → 88 bytes** (smaller because
    V/H modes need fewer non-zero DC coefficients than DC mode does).
  - 64x64 mixed chroma-gradient: Y 42.25 dB, Cb/Cr 47.05 dB.
  - 128x128 vertical+horizontal bars + gradient: **50.11 dB Y**.
- **ffmpeg interop**: bit-exact match against ffmpeg 8.1's libavcodec
  H.264 decoder on every test fixture.

### Status caveats (unchanged from round 1 unless noted)

- Still I-only / IDR-only.
- `cbp_luma` still 0 — luma AC residual not yet emitted (we send only
  the Intra_16x16 DC block per MB). This caps luma quality at
  whatever the per-MB DC + intra-prediction can resolve; with V/H/
  Plane modes that's already excellent on smooth + structured
  content, but detail at QP 26 will still soften.
- No in-loop deblocking — `disable_deblocking_filter_idc == 1`. This
  is mostly invisible on the test pictures because we reconstruct
  every MB from per-MB DCs, so MB-boundary discontinuities are tiny.
- No CABAC — CAVLC only.
- No rate control — fixed QP from `EncoderConfig::qp`.
- nC for chroma AC blocks is hard-coded to 0 (sub-optimal bitrate but
  correct).

## Round 3 — landed

Wire **luma AC residual transmit** (`cbp_luma == 15`) on the
Intra_16x16 path — the encoder can now resolve high-frequency detail
in luma instead of capping the per-MB recon at the DC term.

- **`cbp_luma` decision**: per-MB the encoder runs `quantize_4x4_ac`
  on every 4x4 sub-block; if any AC level is non-zero, `cbp_luma` is
  promoted from 0 to 15 (per §7.4.5.1 Intra_16x16 only allows those
  two values). The mb_type emitted in the bitstream then jumps to the
  matching Table 7-11 entry (groups 3..=5 instead of 0..=2).
- **AC block emit**: 16 `Intra16x16ACLevel` blocks per MB in §6.4.3
  raster-Z order, each a 15-coefficient stream
  (`startIdx=1, endIdx=15, maxNumCoeff=15`). The `nC` for each block
  is derived per §9.2.1.1: the encoder maintains its own
  `CavlcNcGrid`, sets `is_available = true` on the current MB before
  enumerating its blocks, and progressively writes
  `luma_total_coeff[blk]` so subsequent internal-to-MB neighbour
  lookups see the right TotalCoeff. External neighbours pick up the
  fully-populated values from prior MBs.
- **Reconstruction**: per-block inverse 4x4 DC-preserved transform now
  consumes the quantized AC levels alongside the inverse-Hadamard DC,
  so the encoder's recon matches what the decoder will produce
  bit-for-bit even when AC is non-zero.

### Validation

- `cargo test -p oxideav-h264 --lib` — 758 passed.
- `cargo test -p oxideav-h264` — all integration tests pass:
  - all 5 round-1/round-2 tests still pass with identical PSNR (the
    smooth gradients used there genuinely quantize all AC to zero, so
    the recon path is unchanged for them).
  - `round3_ac_residual_lifts_psnr_on_noisy_picture` (new): a 64x64
    high-frequency hash pattern (DC-only encoder gives ~14 dB PSNR
    Y) reaches **46.36 dB Y at QP=18** and **37.63 dB Y at QP=26**.
  - `round3_ffmpeg_decode_noisy_picture_matches` (new): bit-exact
    against ffmpeg 8.1's libavcodec H.264 decoder on the same noisy
    picture.

### Status caveats (unchanged from round 2 unless noted)

- Still I-only / IDR-only.
- Still single mb_type — `I_16x16` only. `I_NxN` (Intra_4x4) is the
  next big quality lift on detailed content.
- No in-loop deblocking — `disable_deblocking_filter_idc == 1`.
- No CABAC — CAVLC only.
- No rate control — fixed QP from `EncoderConfig::qp`.
- Chroma AC `nC` is still hard-coded to 0 (sub-optimal bitrate but
  correct).

## Round 4 (planned)

- **`I_NxN` (Intra_4x4)** with all 9 modes — biggest quality lift on
  highly-detailed content where one prediction mode per 16x16 isn't
  enough to track texture.
- **In-loop deblocking** on the local recon (mirror the decoder's
  §8.7) — required for high PSNR across MB boundaries once we send
  enough residual to make boundaries visible.
- **Chroma AC nC** per §9.2.1.1 (encoder tracks chroma TotalCoeff in
  the `CavlcNcGrid`).

## Round 5+

- P-slice support (single L0, integer MV search).
- Multiple slices per picture.
- B-slices.
- 8x8 transform / High-profile features.
- CABAC.
- Rate control.

## Entry points

- [`Encoder::encode_idr`] — top-level: takes one `YuvFrame` and emits
  an Annex B byte stream + the matching reconstruction.
- Lower-level helpers under [`bitstream`], [`nal`], [`sps`], [`pps`],
  [`slice`], [`macroblock`], [`cavlc`], [`transform`] are usable
  independently — round 2's higher-mode encoder can compose them
  without touching the round-1 dispatcher.
