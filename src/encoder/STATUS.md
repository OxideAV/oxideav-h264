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

## Round 2 (planned)

- **Multiple Intra_16x16 modes** (V/H/Plane) per MB based on a simple
  SAD selection.
- **`I_NxN` (Intra_4x4)** with all 9 modes — same forward+inverse 4x4
  with full per-block prediction.
- **Chroma DC residual** transmitted (cbp_chroma >= 1) so chroma
  reconstruction tracks the source instead of staying at the
  predictor.
- **In-loop deblocking** on the local recon — required for high PSNR
  across MB boundaries.

## Round 3+

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
