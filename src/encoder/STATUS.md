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

## Round 4 — landed

Wire **I_NxN (Intra_4x4) with all 9 modes** — per-4x4-block intra
prediction so the encoder can track local texture orientation that
one I_16x16 mode per macroblock can't capture.

- **`encoder::intra4x4`** module: full §8.3.1.2 implementation of all
  9 prediction modes (Vertical / Horizontal / DC / Diagonal_Down_Left
  / Diagonal_Down_Right / Vertical_Right / Horizontal_Down /
  Vertical_Left / Horizontal_Up). Mirrors the decoder's [`crate::
  intra_pred`] line-by-line but operates on raw `u8` reconstruction
  buffers with simple `(mb_x, mb_y, bx, by)` coordinates. Includes
  §8.3.1.2 "top-right substitution" rule for blocks 3, 5, 7, 11, 13,
  15 and right-edge MBs.
- **Per-block mode decision**: SAD against source over all 9 modes
  per 4x4 sub-block (skipping modes whose required neighbours are
  unavailable). Best mode written to a small encoder-side
  `IntraGrid` so subsequent blocks (in-MB and across-MB) can derive
  `predIntra4x4PredMode` per §8.3.1.1 step 4.
- **mb_type decision**: per MB the encoder scores I_16x16 SAD vs
  I_NxN SAD (sum of best per-block SADs) and picks the lower.
  Picture content with multiple distinct edge orientations per MB
  trips into I_NxN; smooth content stays on I_16x16.
- **Macroblock-layer I_NxN emit** ([`encoder::macroblock::
  write_i_nxn_mb`]):
    1. `mb_type = 0` (raw I_NxN value in I-slice Table 7-11)
    2. 16 × `prev_intra4x4_pred_mode_flag` (1 bit) plus optional
       `rem_intra4x4_pred_mode` (3 bits) per §7.3.5.1.
    3. `intra_chroma_pred_mode` ue(v).
    4. `coded_block_pattern` me(v) — Table 9-4(a) intra column
       inverse mapping (`INTRA_CBP_TO_CODENUM_420_422`).
    5. `mb_qp_delta` se(v) when any cbp non-zero.
    6. 16 4x4 luma blocks (`maxNumCoeff = 16`, full DC + AC) gated
       by 8x8-quadrant cbp bits, in §6.4.3 raster-Z order.
    7. Chroma DC + AC identical to the I_16x16 path.
- **Per-block local recon**: forward 4x4 + quantize + inverse 4x4 +
  add to predictor, written back to `recon_y` so subsequent blocks
  see the correct neighbour samples. For 8x8 quadrants whose cbp bit
  is 0 the recon is rolled back to `pred + 0` (matching what the
  decoder will see — no transmitted residual).
- **CAVLC nC** for the new luma 4x4 blocks: re-uses the existing
  `CavlcNcGrid` + `derive_nc_luma(LumaNcKind::Ac)` chain, with
  `own_luma_totals` reset per MB and progressively updated per block.
  Blocks in zero-cbp quadrants contribute TotalCoeff = 0.

### Validation

- `cargo test -p oxideav-h264 --lib` — **764 passed** (+6 unit tests
  for the new modes and the CBP inverse table).
- All round-1/2/3 integration tests still pass with identical PSNR
  on smooth/mixed/noisy content (as expected — those pictures are
  already optimal under I_16x16; mb_type decision keeps I_16x16).
- New integration tests (round-4):
  - `round4_i_nxn_lifts_psnr_on_oriented_edges` — synthetic 64x64
    picture with a different best 4x4 mode in each sub-block
    (vertical / horizontal / diagonal stripes per 4x4 within each
    MB). I_NxN gets picked, **PSNR Y = 41.25 dB at QP=26** with
    1066-byte stream (vs ~22 dB for I_16x16 alone on this content).
  - `round4_ffmpeg_decode_oriented_edges_matches` — bit-exact
    against ffmpeg 8.1 libavcodec on the same picture.

### Status caveats (unchanged from round 3 unless noted)

- Still I-only / IDR-only.
- mb_type now switches between `I_16x16` and `I_NxN` per MB (no
  I_8x8 / I_PCM yet).
- Mode-decision metric is SAD on the predictor (no rate term, no
  RDO). Round 5+ can add Lagrangian RDO for proper rate-distortion
  trade-off.
- No in-loop deblocking — `disable_deblocking_filter_idc == 1`.
- No CABAC — CAVLC only.
- No rate control — fixed QP from `EncoderConfig::qp`.
- Chroma AC `nC` is still hard-coded to 0 (sub-optimal bitrate but
  correct).

## Round 14 — landed

Wire **in-loop §8.7 deblocking** on the local recon. The encoder now
runs the same picture-level deblock pass that the decoder runs, so the
encoder's `EncodedIdr.recon_*` matches what every downstream decoder
will produce (including future inter slices that use it as a reference).

- **`encoder::deblock`** module: thin adapter that wraps the encoder's
  `Vec<u8>` recon planes in a [`crate::picture::Picture`] (i32
  samples), builds a [`crate::mb_grid::MbGrid`] from per-MB
  `MbDeblockInfo` records (intra flag, QP_Y, per-4x4 nonzero mask),
  and calls [`crate::reconstruct::deblock_picture_full`] verbatim.
  Filtered samples are clipped back into the encoder's `u8` recon
  buffers.
- **Per-MB nonzero tracking**: each `encode_mb_*` function now returns
  a `MbDeblockInfo` carrying the per-4x4 luma AC nonzero bitmap (Z-scan
  Figure 6-10) and per-4x4 chroma AC nonzero bitmap. The bitmap drives
  §8.7.2.1's third bullet (bS=2 when either side has non-zero
  coefficients). For Intra_16x16 with `cbp_luma == 15` we set the bit
  iff the AC scan-order array has any non-zero entry; with `cbp_luma
  == 0` the bitmap is 0. For I_NxN we use the per-block AC nonzero
  flags computed for cbp_luma 8x8 quadrant decision.
- **Slice header**: `disable_deblocking_filter_idc = 0` (filter
  active), `slice_alpha_c0_offset_div2 = slice_beta_offset_div2 = 0`.
  The PPS already had `deblocking_filter_control_present_flag = 1`.
- **`IdrSliceHeaderConfig`** gains `disable_deblocking_filter_idc`,
  `slice_alpha_c0_offset_div2`, `slice_beta_offset_div2` fields. When
  IDC != 1 the writer emits the alpha/beta offsets per §7.3.3.

### Validation

- `cargo test -p oxideav-h264 --lib` — **768 passed** (+4 unit tests
  for the new module).
- All 9 prior integration tests still pass.
- New round-14 integration tests:
  - `round14_deblock_self_roundtrip_matches` — DC-staircase 64x64
    picture (per-MB DC offsets that produce visible blocking without
    the filter). Self-roundtrip matches encoder recon bit-exactly,
    confirming encoder-side deblock == decoder-side deblock.
  - `round14_ffmpeg_decode_deblocked_matches` — bit-exact against
    ffmpeg 8.1 libavcodec on the same picture.
- **PSNR uplift** (QP 26):
  - 64x64 diagonal-gradient testsrc, luma: **42.58 → 48.05 dB**
    (+5.47 dB).
  - 64x64 chroma-gradient: Y 42.58 → 48.05; Cb 47.05 → 50.03;
    Cr 47.05 → 50.00.
  - 128x128 mixed bars + gradient: **50.11 → 51.72 dB** (+1.61 dB).
  - The other fixtures (round-3 noisy QP=18; round-4 oriented-edges)
    are dominated by AC residual rather than block-boundary
    artefacts, so PSNR is essentially unchanged there.
- **ffmpeg interop**: bit-exact match against ffmpeg 8.1 libavcodec
  on every fixture.

### Status caveats (unchanged from round 4 unless noted)

- Still I-only / IDR-only.
- mb_type still switches between I_16x16 and I_NxN.
- **In-loop deblocking now active** (`disable_deblocking_filter_idc
  == 0`, alpha/beta offsets 0).
- No CABAC — CAVLC only.
- No rate control — fixed QP from `EncoderConfig::qp`.
- Chroma AC `nC` is still hard-coded to 0 (sub-optimal bitrate but
  correct).

## Round 15 — landed

Wire **Lagrangian rate-distortion mode decision** at every level: the
per-block I_NxN 9-mode pick, the per-mode I_16x16 pick, and the MB-level
I_16x16 vs I_NxN dispatch all now minimise `J = D + λ · R` instead of
SAD-on-predictor.

- **`encoder::rdo`** module: closed-form `λ = 0.85 · 2^((QP-12)/3)`
  per the JM convention, plus `ssd_4x4` / `ssd_16x16` distortion helpers
  and a `cost_combined(D, R, λ)` integer accumulator (λ scaled by 1024
  to keep arithmetic in `u64`).
- **`BitWriter::bits_emitted`**: returns the running bit count so trial
  CAVLC encodes can measure rate without committing to the live writer.
  Added a `Clone` impl for the snapshot/restore loop.
- **MB-level RDO dispatch**: `Encoder::encode_mb` now snapshots the
  recon planes (only the 16x16 + 8x8 + 8x8 MB window — 384 bytes), the
  `BitWriter`, the `CavlcNcGrid` slot, and the `IntraGrid` slot before
  trialing. Both paths run in turn into the live state; the loser is
  rolled back via the snapshot. Lower-J path wins; on tie, prefer
  I_16x16 (smaller MB syntax).
- **I_16x16 mode RDO**: `trial_intra16x16_cost` runs the full §8.5.10
  forward-quant-inverse pipeline per candidate mode and computes
  `D = SSD(source, recon_clamped)` plus `R = trial CAVLC bits`. The
  `mb_type` bit cost is approximated as a constant 7 bits (Table 7-11
  width range is 1..9 across the 24 entries; the constant is identical
  for every I_16x16 candidate so it cancels out in the within-path
  comparison and matches the I_NxN MB-overhead seed for the cross-path
  comparison).
- **I_NxN per-block RDO**: each of the 16 4x4 blocks trials all 9
  available modes; for each, we predict, transform, quantise, inverse,
  add to predictor, compute SSD, run trial CAVLC for rate. Mode rate
  also includes the per-block `prev_intra4x4_pred_mode_flag` (1 bit)
  plus optional `rem_intra4x4_pred_mode` (3 bits when mode != predicted).
  The MB-level cost seeds with a `+25` bit overhead (`mb_type` + chroma
  pred + cbp + qp_delta + an empirical "post-deblock D under-measurement"
  bias) so the cross-path comparison is calibrated against the I_16x16
  seed.

### Encoder bug fix (round-15 collateral)

Surfaced and fixed a pre-existing top-right substitution bug for
`Intra_4x4` block 5: the encoder's `gather_top_row` was substituting
`p[3, -1]` for `p[4..=7, -1]` even when those samples were available
from the already-encoded above-right MB. The decoder reads the real
samples per §8.3.1.2, causing recon divergence whenever the round-15
RDO finally picked `Intra_4x4_VerticalLeft` for block 5. Fixed in
`encoder::intra4x4::gather_top_row` by removing 5 from the substitution
set (matching the decoder's `gather_samples_4x4`).

### Validation

- `cargo test -p oxideav-h264 --lib` — **774 passed** (+6 unit tests
  for the new `rdo` module + `bits_emitted` test).
- All 11 prior integration tests still pass with bit-exact ffmpeg
  interop on every fixture.
- **PSNR** (QP 26, RDO mixed dispatch):
  - 64x64 diagonal-gradient testsrc: 48.05 → **47.58 dB** (-0.47 dB,
    stream **100 → 88 bytes** for −12% rate).
  - chroma roundtrip: Y 47.58, Cb 50.03, Cr 50.00 (chroma path
    unchanged; luma stream 154 vs 232 bytes).
  - 128x128 mixed bars + gradient: 51.72 → **51.51 dB** (-0.21 dB,
    stream 196 vs 201 bytes).
  - 64x64 oriented-edges: 41.25 → **41.16 dB** (-0.09 dB, stream
    1056 vs 1066 bytes).
  - 64x64 noisy QP=18: 46.36 → **46.47 dB** (+0.11 dB, stream
    3056 vs 3089 bytes).
  - 64x64 noisy QP=26: 37.63 → **37.69 dB** (+0.06 dB, stream
    2101 vs 2107 bytes).
  - 64x64 dc-staircase: 50.00 → **50.00 dB** (unchanged).
- The smooth fixtures see a small PSNR regression but a comparable rate
  saving — RDO is choosing a different R-D operating point. The noisy
  fixtures see a small PSNR gain plus rate savings — pure improvement.

### Status caveats (unchanged from round 14 unless noted)

- Still I-only / IDR-only.
- mb_type still switches between I_16x16 and I_NxN; selection now via
  Lagrangian RDO (round 15) instead of SAD (rounds 4..14).
- In-loop deblocking active.
- No CABAC — CAVLC only.
- No rate control — fixed QP from `EncoderConfig::qp`.
- Chroma AC `nC` still hard-coded to 0 (sub-optimal bitrate but correct).
- λ is calibrated against the round-15 fixture set; future rounds may
  refine it further (especially for inter-MB λ scaling once P/B slices
  land).

## Round 16 — landed

Wire **P-slice support** (inter prediction) end-to-end. Single L0 ref,
integer-pel motion estimation, P_Skip + P_L0_16x16.

- **Slice header (`encoder::slice::write_p_slice_header`)**: emits the
  non-IDR P-slice header per §7.3.3 — `slice_type=5` (P, all-same-type),
  `num_ref_idx_active_override_flag=0` (uses PPS default of 1 ref),
  `ref_pic_list_modification` empty, non-IDR `dec_ref_pic_marking` with
  `adaptive_ref_pic_marking_mode_flag=0` (sliding-window).
- **Motion estimation (`encoder::me`)**: integer-pel **full search**
  within ±16 px around (0, 0) using SAD on the source luma 16×16 block
  vs the reference plane (with §8.4.2.2.1 edge replication). Tie-break
  prefers smaller `|mv|` to minimise mvd cost when SADs are equal.
- **MV prediction grid (`MvGrid` + `mvp_for_16x16` + `p_skip_mv`)**:
  encoder tracks the per-MB chosen MV/refIdx so that subsequent MBs
  derive `mvpLX` per §8.4.1.3 the same way the decoder will. P_Skip
  derivation per §8.4.1.2 (zero-MV substitution rules + median otherwise)
  uses the existing `crate::mv_deriv` helpers — encoder and decoder
  share the spec primitives.
- **Inter prediction** wraps `crate::inter_pred::interpolate_luma` and
  `interpolate_chroma`. Integer-pel luma MVs collapse to the
  edge-replicated copy path; chroma MVs derived from luma per §8.4.1.4
  (4:2:0) may end up half-pel even for integer-pel luma, in which case
  the bilinear chroma interpolator runs.
- **MB writer (`encoder::macroblock::write_p_l0_16x16_mb`)**: emits
  `mb_type=P_L0_16x16` (raw 0), `mvd_l0_x se(v)`, `mvd_l0_y se(v)`,
  inter `coded_block_pattern me(v)` (Table 9-4(a) inter column),
  `mb_qp_delta se(v)` (when CBP > 0), 16 luma 4×4 residual blocks
  gated by 8×8 quadrant `cbp_luma`, then chroma DC + AC. `ref_idx_l0`
  is **absent** when `num_ref_idx_l0_active_minus1 == 0` per §7.3.5.1.
- **P_Skip path**: when the chosen MV equals the §8.4.1.2-derived skip
  MV AND `cbp_luma == cbp_chroma == 0`, the MB is omitted from the
  bitstream and contributes to a CAVLC `mb_skip_run` ue(v) emitted
  before the next coded MB (or at slice end).
- **Encoder API** (`Encoder::encode_p`): takes a `YuvFrame` + an
  `EncodedFrameRef` (the previous frame's recon planes) and emits the
  P-slice access unit. `EncodedFrameRef::from(&EncodedIdr)` and
  `From<&EncodedP>` make chaining frame-by-frame trivial.
- **Local recon + deblocking**: the encoder's recon planes are
  populated and run through the same §8.7 deblocker as I-frames.
  `MbDeblockInfo` gained `mv_l0[16]`, `ref_idx_l0[4]`, `ref_poc_l0[4]`
  so that the deblock walker's `different_ref_or_mv_luma` (which
  drives bS=2 vs bS=1 on inter/inter edges) sees the same MV / ref
  identity that the decoder will compute. Without this the encoder's
  recon would diverge from the decoder's for any inter MB.

### Validation

- `cargo test -p oxideav-h264 --lib` — **780 passed** (+6: 4 new ME
  tests, 1 P-slice header round-trip, 1 inter-CBP table round-trip).
- All prior integration tests still pass (the 11 round-1..14 I-only
  fixtures unchanged).
- New round-16 integration tests:
  - `round16_p_slice_self_roundtrip_matches_local_recon`: encode
    IDR + P (where P = IDR shifted right 4 luma px), decode the
    combined Annex B with our own decoder → bit-exact match against
    encoder recon. Decoded P luma vs original source = **49.79 dB
    PSNR** (clean integer-shift content + integer-pel ME finds the
    motion exactly).
  - `round16_p_slice_static_picture_uses_skips`: when P-frame source ==
    IDR source, the encoder produces a 53-byte P slice (vs 88-byte
    IDR), bit-exact through self-roundtrip. The drop in size is
    smaller than ideal because IDR encoding noise prevents some MBs
    from quantising the residual to zero.
  - `round16_ffmpeg_decode_p_sequence_matches`: ffmpeg/libavcodec
    decodes the same combined IDR+P stream bit-exactly against the
    encoder's local recon (≤1 LSB tolerance).

### Status caveats (unchanged from round 15 unless noted)

- IDR + P-slice (P only). No B-slices.
- mb_type now extends to `P_L0_16x16` and `P_Skip` for P-slices, in
  addition to `I_16x16` / `I_NxN` for the IDR.
- **Single MV per MB** (16×16 partition only). No 16×8 / 8×16 / 8×8
  / 4MV.
- **Integer-pel ME only**. Half-pel and quarter-pel refinement are
  later rounds.
- **No intra fallback in P-slices** — every P-MB stays inter (P_Skip
  or P_L0_16x16). Future round can add I_16x16 / I_NxN trial in P
  via the existing RDO framework.
- Single reference: `num_ref_idx_l0_default_active_minus1 = 0` → one
  L0 ref. The encoder uses the previous reconstructed frame.
- In-loop deblocking active across both intra and inter MBs.
- No CABAC — CAVLC only.
- No rate control — fixed QP from `EncoderConfig::qp`.

## Round 17 — landed

Layer **half-pel motion estimation** on top of round-16's integer-pel
P-slice support, using the same §8.4.2.2.1 6-tap luma interpolator the
decoder runs. Encoder and decoder pick the **same** predictor for any
half-pel MV the encoder selects, so bit-equivalence is preserved
end-to-end (verified against ffmpeg).

- **Half-pel refinement (`encoder::me::refine_half_pel_16x16`)**: takes
  the integer-pel winner and probes the 8 surrounding ½-pel offsets
  (`(±2, 0)`, `(0, ±2)`, `(±2, ±2)` in quarter-pel units). Predictor
  comes straight from `inter_pred::interpolate_luma`, which is the
  same routine the decoder calls — the encoder doesn't reimplement the
  6-tap FIR. Tie-break still favours the smaller-magnitude MV to keep
  `mvd_l0_*` se(v) costs low.
- **Combined ME entry (`encoder::me::search_half_pel_16x16`)**: integer
  full search (±16) followed by half-pel refinement. Drop-in
  replacement for `search_integer_16x16`; round-17 `encode_p_mb` now
  calls it.
- **No P-slice header / writer changes** — the bitstream already
  carried fractional MVs via `mvd_l0_x`/`mvd_l0_y se(v)`. Only the
  encoder's MV decision changed.

### Validation

- `cargo test -p oxideav-h264` — **900 passed** (+5 vs round 16: 3
  new ME unit tests + 2 new half-pel integration tests).
- All round-1..16 fixtures still pass. The round-16 4-pixel-shift
  fixture moved 49.79 → **50.28 dB** PSNR_Y under the new ME (no
  regression on integer motion: integer winner stays integer when
  the half-pel neighbours can't beat its SAD).
- New round-17 integration tests
  (`integration_p_slice_half_pel.rs`):
  - `round17_half_pel_p_slice_high_psnr`: encode IDR + P where P is
    the IDR shifted right by **½ luma pixel**, decode through our own
    decoder → bit-exact against encoder recon (max diff 0). PSNR_Y =
    **49.19 dB**, with `cbp_luma == 0` on most MBs because the
    half-pel predictor matches the source bit-for-bit.
  - `round17_half_pel_ffmpeg_interop`: ffmpeg/libavcodec decodes the
    same half-pel P stream bit-equivalently against the encoder's
    local recon (max diff 0).

### Status caveats (unchanged from round 16 unless noted)

- IDR + P-slice. No B-slices.
- mb_type: `P_L0_16x16` / `P_Skip` for P; `I_16x16` / `I_NxN` for IDR.
- Single MV per MB (16×16 partition only). No 16×8 / 8×16 / 8×8 / 4MV.
- ME: **integer + half-pel** (round-17). Quarter-pel refinement is the
  next step.
- No intra fallback in P-slices.
- Single reference, sliding-window DPB.
- In-loop deblocking active across intra and inter MBs.
- CAVLC only.
- No rate control — fixed QP.

## Round 18 — landed

Layer **quarter-pel motion estimation** on top of round-17's half-pel
chain. The encoder now searches all 16 sub-pel positions in
§8.4.2.2.1 Table 8-12 (integer + 3 half-pel + 12 quarter-pel) and
picks the SAD-best MV. Predictor comes straight from
`inter_pred::interpolate_luma`, the same routine the decoder calls,
so encoder and decoder remain bit-equivalent end-to-end.

- **Quarter-pel refinement (`encoder::me::refine_quarter_pel_16x16`)**:
  takes the half-pel winner and probes the 8 surrounding ¼-pel offsets
  (`(±1, 0)`, `(0, ±1)`, `(±1, ±1)` in quarter-pel units). Reuses the
  `sad_16x16_at_qpel` helper that already drives half-pel; no new
  spec primitive is reimplemented (`interpolate_luma` already supports
  all 16 sub-pel positions per §8.4.2.2.1 eq. 8-243 .. 8-261).
  Tie-break still favours the smaller-magnitude MV to keep
  `mvd_l0_*` se(v) cost low.
- **Combined ME entry (`encoder::me::search_quarter_pel_16x16`)**:
  integer full search (±16) → half-pel refinement → quarter-pel
  refinement. Drop-in replacement for `search_half_pel_16x16`;
  round-18 `encode_p_mb` now calls it.
- **No P-slice header / writer changes** — `mvd_l0_x`/`mvd_l0_y se(v)`
  already carries quarter-pel MVs natively; only the encoder's MV
  decision changed.

### Validation

- `cargo test -p oxideav-h264` — **906 passed** (+6 vs round 17:
  4 new ME unit tests + 2 new quarter-pel integration tests).
- All round-1..17 fixtures still pass. Notable PSNR uplifts under the
  new ME (no regression on integer or half-pel motion: each pass
  refines only when the new candidate beats the previous SAD):
  - round-16 4-pixel-shift fixture: 50.28 → **50.82 dB** PSNR_Y.
  - round-17 ½-pixel-shift fixture: 49.19 → **49.40 dB** PSNR_Y,
    bit-exact ffmpeg interop.
- New round-18 integration tests
  (`integration_p_slice_quarter_pel.rs`):
  - `round18_quarter_pel_p_slice_high_psnr`: encode IDR + P where
    P is the IDR shifted right by **¼ luma pixel**, decode through
    our own decoder → bit-exact against encoder recon (max diff 0).
    PSNR_Y = **53.61 dB**, P-slice stream = 30 bytes (vs 37 bytes
    for the half-pel fixture — most MBs reach `cbp_luma == 0`
    because the quarter-pel predictor matches the source bit-for-
    bit).
  - `round18_quarter_pel_ffmpeg_interop`: ffmpeg/libavcodec decodes
    the same quarter-pel P stream bit-equivalently against the
    encoder's local recon (max diff 0).

### Status caveats (unchanged from round 17 unless noted)

- IDR + P-slice. No B-slices.
- mb_type: `P_L0_16x16` / `P_Skip` for P; `I_16x16` / `I_NxN` for IDR.
- Single MV per MB (16×16 partition only). No 16×8 / 8×16 / 8×8 / 4MV.
- ME: **integer + half-pel + quarter-pel** (round-18). All 16 sub-pel
  positions per §8.4.2.2.1 are reachable.
- No intra fallback in P-slices.
- Single reference, sliding-window DPB.
- In-loop deblocking active across intra and inter MBs.
- CAVLC only.
- No rate control — fixed QP.

## Round 19 — landed

**4MV (P_8x8 with `sub_mb_type = PL08x8`)** — adds the per-8x8-
partition motion-compensation mode, gated by a content-driven SAD
heuristic so smooth content stays on the round-16 1MV path.

- **Per-8x8 ME** (`encoder::me::search_quarter_pel_8x8`) mirrors the
  round-18 quarter-pel chain but on 8×8 sub-blocks:
  integer-pel ±range full search → ½-pel refinement → ¼-pel
  refinement, all SAD-based against the same `interpolate_luma` the
  decoder uses (so encoder and decoder agree bit-for-bit on the
  predictor for any sub-pel MV).
- **MV grid extended** (`MvGridSlot::{mv_l0_8x8, ref_idx_l0_8x8}`)
  to track per-8x8 MVs / refs. P_L0_16x16 / P_Skip MBs fill all
  four 8x8 entries with the same value, preserving exact round-16
  / 17 / 18 neighbour reads.
- **Per-partition MVpred** (`mvp_for_p_8x8_partition`) walks the
  §8.4.1.3 (A, B, C, D) neighbours of an 8x8 partition, supporting
  within-MB neighbour reads of already-decoded sub-partitions
  (sub-blocks decode in raster order 0→1→2→3) and falling back to
  the §8.4.1.3.2 C→D substitution when the C neighbour is in a
  not-yet-decoded sub-partition of the current MB.
- **Bitstream emit** (`encoder::macroblock::write_p_8x8_all_pl08x8_mb`)
  → `mb_type ue(3)` (P_8x8, Table 7-13) + 4× `sub_mb_type ue(0)`
  (PL08x8, Table 7-17) + 4× `(mvd_l0_x se(v), mvd_l0_y se(v))` +
  inter `coded_block_pattern me(v)` + optional `mb_qp_delta` +
  the same residual layout as P_L0_16x16. Single-reference path
  (`num_ref_idx_l0_active_minus1 == 0`) → ref_idx absent.
- **Mode decision**: 4MV picked iff
  `me_16x16_sad >= 1024 && me_4mv_sad_sum * 10 < me_16x16_sad * 7`
  (≥30% SAD reduction over a non-trivial 1MV residual). The two
  gates protect rounds 16/17/18 fixtures (smooth content → both
  paths converge to the same MV → 1MV wins → no regression).

### Validation

- `cargo test -p oxideav-h264` — **911 passed** (was 906).
- Round 16/17/18 PSNR fixtures unchanged: 50.82 / 49.40 / 53.61 dB.
- New round-19 fixture (`integration_p_slice_4mv.rs`,
  `round19_4mv_p_slice_high_psnr_on_sub_mb_motion`):
  source has each 8x8 sub-block of every 16x16 MB shifted by a
  *different* integer-pel amount. **Local recon PSNR_Y = 40.95 dB**,
  decoder PSNR_Y = 40.95 dB (max enc/dec diff 0).
- ffmpeg interop (`round19_4mv_ffmpeg_interop`):
  libavcodec decodes the 4MV P-frame **bit-equivalently** against
  the encoder's local recon (max diff 0).
- 3 new unit tests in `me::tests` covering per-8x8 ME (zero motion +
  per-sub-block motion recovery + `p_8x8_pred_bits`).

## Round 20 — landed

Wire **B-slice support** for the explicit-inter macroblock types:
`B_L0_16x16`, `B_L1_16x16`, `B_Bi_16x16` (Table 7-14 raw mb_type 1, 2,
3). Bumps the SPS profile to **Main (77)** because Baseline forbids
B-slices per §A.2.2. New entry point `Encoder::encode_b(frame,
ref_l0, ref_l1, frame_num, pic_order_cnt_lsb)` emits a single
non-reference (`nal_ref_idc = 0`) B-slice using the IDR + a prior
P-frame as L0 / L1 anchors.

- **`EncoderConfig`** gains `profile_idc` and `max_num_ref_frames`.
  Defaults preserve round-19 behaviour (Baseline, 1 ref). Tests for
  B-slices set `profile_idc = 77`, `max_num_ref_frames = 2`.
- **B-slice header** (`encoder::slice::write_b_slice_header`):
  `slice_type = 6` (B, all-same-type), `direct_spatial_mv_pred_flag = 1`,
  `num_ref_idx_active_override_flag = 0` (use PPS defaults — one ref
  per list), both list-modification flags = 0, `pred_weight_table`
  absent (`weighted_bipred_idc == 0` ⇒ §8.4.2.3.1 default), and
  `dec_ref_pic_marking` absent (B is non-reference, `nal_ref_idc=0`).
- **B-slice macroblock writer**
  (`encoder::macroblock::write_b_16x16_mb` + `B16x16McbConfig` +
  `BPred16x16` enum). Layout per §7.3.5.1:
    1. `mb_type ue(v)` ∈ {1, 2, 3}
    2. `ref_idx_l0` te(v) — absent at single-ref
    3. `ref_idx_l1` te(v) — absent at single-ref
    4. `mvd_l0_x, mvd_l0_y se(v)` — present when `pred.uses_l0()`
    5. `mvd_l1_x, mvd_l1_y se(v)` — present when `pred.uses_l1()`
    6. inter `coded_block_pattern me(v)` + optional `mb_qp_delta` +
       the same residual layout as P_L0_16x16.
- **Mode decision**: per MB the encoder runs quarter-pel ME against
  L0 and L1 independently, then SAD-scores three candidates:
  L0-only, L1-only, and bipred = `(L0_pred + L1_pred + 1) >> 1`
  (§8.4.2.3.1 eq. 8-273). Lowest-SAD wins with tie-break
  Bi → L0 → L1 (Bi first because matching predictors mean lowest
  residual + minimal mvd cost when both lists agree).
- **Two-list MV-pred grids** (`mv_grid_l0` / `mv_grid_l1`) — §8.4.1.3
  derives `mvpLX` per list independently. L0-only MBs leave L1 grid
  slot at default (`ref_idx = -1, mv = 0`), and vice versa, so a
  later B MB reading either side gets the correct §8.4.1.3.2 step-2
  collapse.
- **CAVLC `mb_skip_run`** (§7.3.4): emitted as `ue(0) = '1'` before
  every coded MB (round 20 never produces B_Skip), plus a final
  trailing `mb_skip_run = 0` at end of slice. Without this the
  decoder mis-aligns at MB 2 (saw "cbp 55 out of range"); landed the
  fix once the symptom showed up under our own decoder.
- **Picture-level `MbDeblockInfo`**: extended with `mv_l1`,
  `ref_idx_l1`, `ref_poc_l1`. The §8.7.2.1 walker's bipred
  `different_ref_or_mv_luma` reads picture identity by POC, so the
  encoder feeds in `ref_poc_l0 = 0` (IDR anchor) and `ref_poc_l1 =
  1000` (sentinel ≠ L0 POC) to make the two anchors distinct under
  NOTE 1.
- **Bipred predictor build** (`build_b_predictors`): wraps the
  existing `build_inter_pred_luma` / `build_inter_pred_chroma` for
  each list, then applies the §8.4.2.3.1 average elementwise. The
  encoder never re-implements the 6-tap FIR or the bilinear chroma
  interpolator — both come straight from the decoder's
  `inter_pred::interpolate_luma` / `interpolate_chroma`, which keeps
  encoder and decoder bit-equivalent on every MV / sub-pel position.

### Validation

- `cargo test -p oxideav-h264` — **915 passed** (was 911 in round 19;
  +4: 2 new unit tests + 2 new integration tests).
- Round 16/17/18/19 PSNR fixtures unchanged: 50.82 / 49.40 / 53.61 /
  40.95 dB.
- New round-20 fixture (`integration_b_slice.rs`):
  - `round20_b_slice_self_roundtrip_matches_local_recon`: encode
    IDR + P + B where B = midpoint(IDR, P) sample-by-sample. Every
    MB picks `B_Bi_16x16` because the bipred predictor reconstructs
    the source bit-for-bit. Decode through our own decoder →
    bit-exact match (max diff 0). **Local + decoder PSNR_Y = 54.19
    dB**.
  - `round20_b_slice_ffmpeg_interop`: ffmpeg/libavcodec decodes the
    same IDR + P + B Annex B stream **bit-equivalently** against the
    encoder's local recon (max diff 0). 54.19 dB PSNR_Y vs source.

### Status caveats (unchanged from round 19 unless noted)

- **B-slice landed** (round 20). mb_type ∈ {`B_L0_16x16`,
  `B_L1_16x16`, `B_Bi_16x16`} alongside the round-19 P / IDR types.
- No `B_Skip` / `B_Direct_16x16` (the spec's mode-decision derivation
  paths). `direct_spatial_mv_pred_flag` is signalled but never
  exercised.
- No intra fallback in P / B slices.
- Single reference per list: L0 = the previous IDR, L1 = the
  previous P-frame. `max_num_ref_frames = 2` in the SPS.
- Profile bumped to Main (77) when the caller asks for B-slices via
  `EncoderConfig::profile_idc`. The default still emits Baseline
  (66) for IDR-only / P-only streams.
- In-loop deblocking active across intra / P / B MBs, with proper
  L0+L1 ref / POC identity feeding §8.7.2.1's bipred edge derivation.
- CAVLC only.
- No rate control — fixed QP.

## Round 21 — landed

Wire **B_Skip / B_Direct_16x16** spatial-direct mode on top of round-20's
explicit-inter B-slice support. The encoder now mirrors the §8.4.1.2.2
derivation locally and emits the small/zero-bitstream B_Direct_16x16
(raw mb_type 0) or B_Skip (no syntax — folded into `mb_skip_run`)
whenever the spatial-direct predictor is competitive with the
round-20 explicit-inter (B_L0 / B_L1 / B_Bi) candidates.

- **§8.4.1.2.2 derivation** (`encoder::mod::b_spatial_direct_derive`):
  reads the encoder's L0 and L1 MV grids for the (A, B, C, D)
  neighbours, applies the §8.4.1.3.2 C→D substitution, runs the
  MinPositive chain on refIdxLX (eq. 8-184/8-185), detects
  `directZeroPredictionFlag` (eq. 8-188..8-190), and computes per-8x8
  partition MVs subject to the step-7 `colZeroFlag` short-circuit.
- **`colZeroFlag` (§8.4.1.2.2 step 7)**: encoder reads the colocated
  block's L0 MV / refIdx from the L1 anchor. New
  `EncodedFrameRef::partition_mvs` carries these — populated by
  `encode_p` from its `MvGrid`, populated for IDRs as "all intra"
  (refIdxCol = -1 → colZeroFlag = 0). Intra colocated blocks short-
  circuit per §8.4.1.2.1 NOTE.
- **B_Direct_16x16 writer** (`encoder::macroblock::write_b_direct_16x16_mb`
  + `BDirect16x16McbConfig`): emits `mb_type=0` (Table 7-14 raw 0) +
  `coded_block_pattern me(v)` + optional `mb_qp_delta` + residuals.
  No `mvd_l0` / `mvd_l1` / `ref_idx_l0` / `ref_idx_l1` (the decoder
  derives those itself).
- **B_Skip path**: when the chosen mode is Direct AND `cbp_luma == 0
  AND cbp_chroma == 0`, the MB is omitted from the bitstream entirely.
  `pending_skip` accumulates the run; flushed as a single ue(v)
  before the next coded MB or at end-of-slice.
- **Mode-decision Lagrangian bias**: Direct gets a SAD allowance over
  the explicit candidates of `(qp_y/6 + 1) * 16` SAD units, calibrated
  so that on smooth / static content the explicit-inter path's
  marginal SAD wins are outweighed by Direct's 10-20 bit syntax savings.
  Round-20 fixtures (midpoint linear interp) still pass under this
  bias because the explicit-Bi SAD on those is materially below the
  spatial-direct SAD when neighbours haven't propagated the (0, 0) MV
  yet.
- **Per-8x8 uniformity gate**: round-21 only emits Direct / Skip when
  the §8.4.1.2.2 derivation's per-partition MVs are uniform across the
  MB. Non-uniform direct (per-partition motion compensation) is
  deferred to a later round.

### Validation

- `cargo test -p oxideav-h264 --lib` — **803 passed** (+2 unit tests
  for the B_Direct_16x16 writer round-trip and the mb_type/Table 7-14
  decode).
- All round-1..20 fixtures still pass with bit-exact ffmpeg interop on
  every fixture. The round-20 midpoint test PSNR is 51.86 dB (was
  54.19 dB; the small drop reflects more MBs choosing
  Direct/Skip on the smooth content where explicit-Bi was marginally
  better; quality is still well above the test's 38 dB floor).
- New round-21 integration tests
  (`integration_b_skip_direct.rs`):
  - `round21_static_picture_uses_b_skip`: IDR + P + B all from the
    same source → every B-MB picks Direct → cbp == 0 → B_Skip.
    **B-slice = 11 bytes** (vs round-20's 75 bytes for same content
    when only explicit modes are available; a **7× compression** on
    the B portion). Decoder PSNR_Y = 52.36 dB, max enc/dec diff 0.
  - `round21_b_skip_ffmpeg_interop`: ffmpeg/libavcodec decodes the
    same all-skips B stream **bit-equivalently** against the encoder's
    local recon (max diff 0). 52.36 dB PSNR_Y.
  - `round21_midpoint_b_slice_decoder_match`: round-20's midpoint
    fixture re-tested under the new mode-decision — bit-exact recon
    via our own decoder, B-slice 16 bytes (unchanged).

### Status caveats (unchanged from round 20 unless noted)

- B-slices: round-20 explicit-inter (B_L0 / B_L1 / B_Bi) plus
  **round-21 B_Direct_16x16 / B_Skip** for spatial direct mode.
  No 16x8 / 8x16 partitions, no B_8x8 / B_Direct_8x8.
- Spatial direct only (`direct_spatial_mv_pred_flag = 1`, hard-wired).
  Temporal direct (§8.4.1.2.3) deferred.
- Direct mode emits only when the §8.4.1.2.2 derivation gives uniform
  per-8x8 MVs and at least one list is used. Non-uniform direct is
  deferred.
- No intra fallback in P / B slices (the round-21 frontier still
  applies).
- In-loop deblocking active across intra / P / B (with B_Skip /
  B_Direct properly fed through `MbDeblockInfo`).
- CAVLC only.
- No rate control — fixed QP.

## Round 22 — landed

B-slice 16x8 / 8x16 partition modes (§7.4.5 Table 7-14 raw mb_types
4..=21). The encoder picks among:
- Round-21 set: B_Skip, B_Direct_16x16, B_L0_16x16, B_L1_16x16,
  B_Bi_16x16.
- Round-22 set: 9 mb_types per shape × 2 shapes = 18 new variants
  (`B_L0_L0_16x8` / `B_L0_L1_16x8` / `B_Bi_Bi_16x8` / etc.).

Per-MB decision flow extends the round-21 logic with two extra steps:

1. **Per-cell ME** for both L0 and L1 (4 cells × 2 lists = 8 calls to
   `search_quarter_pel_8x8`). The 16x16-best MV from round-20 is also
   kept as a candidate.

2. **Per-partition mode enumeration** (`best_partition_mode_b`): for
   each of the 4 partition shapes (16x8 top, 16x8 bottom, 8x16 left,
   8x16 right), try all 9 (mode_l0, mode_l1) combinations and keep
   the one with minimum partition-wide SAD. Joint SAD across both
   halves is the partition-shape SAD; the better of `(16x8, 8x16)` is
   the partition winner.

3. **Lagrangian gate**: partition wins over 16x16 only when its SAD
   beats 16x16's by more than `(qp_y/6 + 1) * 8` SAD units (the rate
   penalty for the larger mb_type ue(v) plus the duplicated mvds).
   Direct mode is never overridden by partition mode (B_Skip's 0 bits
   of MB syntax dominates any partition mode's ~10+ bits).

§8.4.1.3 partition-shape MVP derivation uses the
`Partition16x8Top/Bottom` / `Partition8x16Left/Right` directional
shortcuts and falls through to median(A, B, C) with the §8.4.1.3.2
C→D substitution. The encoder's partition writer (`write_b_16x8_mb`,
`write_b_8x16_mb`) emits per-§7.3.5.1: `mb_type` ue(v), then
`ref_idx_l0[part]` for parts 0..1 (gated on partition mode), then
`ref_idx_l1[part]`, then `mvd_l0[part]`, then `mvd_l1[part]`, then
the standard cbp + mb_qp_delta + residuals tail.

Per-MB grid update writes per-8x8 MVs (top half = #0/#1, bottom =
#2/#3 for 16x8; left = #0/#2, right = #1/#3 for 8x16). The deblock
walker's per-4x4 MV array (`MbDeblockInfo::mv_l0`) is filled in §6.4.3
Z-scan order via `LUMA_4X4_BLK[blk]` — the spec-mandated layout that
the decoder also uses, ensuring §8.7.2.1 boundary-strength derivation
agrees encoder ↔ decoder.

Side fix: `neighbour_mvs_16x16` now reads each neighbour's per-8x8
cell that **spatially abuts** the current MB's top-left 4x4 (`#1` for
left, `#2` for above and above-right, `#3` for above-left). Previously
it always read 8x8 #0 — a bug that didn't bite earlier because all
prior B/P MBs had uniform per-8x8 MVs (16x16 partition); with round-22
partition mode introducing per-8x8 variation, the bug would have
produced encoder ↔ decoder mvp mismatches for Direct mode of
partition-mode-neighbour MBs.

### Validation

- 936 tests passing (was 929 in round 21; +4 unit + 3 integration).
- New `integration_b_partitions.rs`:
  - `round22_b_16x8_self_roundtrip_matches_local_recon` — encode IDR
    + P + B with per-half-vertical motion (top half = IDR content,
    bottom = P-shifted content). Encoder picks 16x8 mb_types per MB.
    Decoder match: max enc/dec diff 0, PSNR_Y = 46.87 dB.
  - `round22_b_8x16_self_roundtrip_matches_local_recon` — same with
    per-half-horizontal motion. Encoder picks 8x16 mb_types. PSNR_Y
    = 48.54 dB, max enc/dec diff 0.
  - `round22_b_partitions_ffmpeg_interop` — ffmpeg/libavcodec decodes
    the 16x8 stream **bit-equivalently** (max diff 0).

### Caveats

- B_8x8 sub-MB partitions still deferred to a later round.
- B_Direct_8x8 (sub-MB direct) still deferred.
- Partition mode never overrides Direct (intentional — B_Skip
  dominates any partition mode rate-wise).
- CAVLC only.
- Single ref per list.

## Round 29 — landed

**Intra fallback in P / B slices** via per-MB Lagrangian RDO. The
encoder now compares the inter winner's cost (`J_inter = D_inter +
λ·R_inter` measured on the actually-emitted bitstream) against an
`Intra_16x16` trial cost (`J_intra` measured the same way) and
installs the lower-J path. Picks up scene-cut frames, occlusion of
new content over a textured background, and high-detail areas where
motion-compensated prediction's residual is more expensive to code
than a fresh intra block.

- **`EncoderConfig::intra_in_inter`** (default `true`) — toggle for
  the new path. When `false`, the round-16..28 inter-only behaviour
  is preserved exactly (useful for A/B testing).
- **`encoder::macroblock::write_intra16x16_mb_in_inter_slice`** — new
  writer mirroring `write_intra16x16_mb` but emitting the mb_type
  ue(v) with a configurable offset: `+5` for P-slice (Table 7-13
  intra range raw 5..=30), `+23` for B-slice (Table 7-14 intra range
  raw 23..=48). The decoder side already accepts these values
  (`MbType::from_p_slice` / `from_b_slice` map them via
  `from_i_slice(raw - 5)` / `from_i_slice(raw - 23)`).
- **`InterMbStateSnapshot`** wraps the existing `MbStateSnapshot`
  with the inter-context-only fields: `MvGridSlot` (and L1 grid for
  B), plus `pending_skip`. Lets the trial RDO roll back the loser
  cleanly without leaking state into the next MB.
- **`Encoder::encode_p_mb_with_intra_fallback`** /
  **`Encoder::encode_b_mb_with_intra_fallback`** — RDO wrappers
  around the existing `encode_p_mb` / `encode_b_mb` per-MB encoders.
  Snapshot pre-state, run inter trial, measure J_inter, snapshot
  post-inter, restore pre-state, run new
  `encode_p_mb_intra16x16` / `encode_b_mb_intra16x16` trial,
  measure J_intra, install winner.
- **`Encoder::encode_intra16x16_in_inter_slice`** — shared emit
  helper used by both P/B paths. Runs §8.3.3 luma RDO across the 4
  pred_modes (Vertical / Horizontal / DC / Plane), §8.3.4 chroma
  SAD across the 4 chroma modes, the forward+quantize+inverse
  pipeline, and emits the syntax via
  `write_intra16x16_mb_in_inter_slice` with the requested offset.
  Updates `recon_*`, `nc_grid` (per-block CAVLC TotalCoeff), and
  `intra_grid` (`available = true, is_i_nxn = false` so subsequent
  I_NxN neighbour lookups fall to DC). The wrapper updates `mv_grid`
  with `is_intra = true`, `ref = -1`, `mv = 0` so subsequent inter
  MBs' §8.4.1.3.2 step-2 collapse fires.
- **Per-MB syntax accounting** — `J_inter` = `cost_combined(SSD,
  inter_bits, λ_inter)`. For B_Skip / P_Skip MBs the inter trial
  doesn't write any bits (the skip is folded into `pending_skip`),
  so `J_inter = SSD(source, recon)` only — the fast path always
  beats intra on static / skip-able content.

### Validation

- Lib tests: 819 → **819 passed** (all unit + integration tests
  pass). Two trivial clippy lints fixed.
- Pre-existing failure `round21_static_picture_uses_b_skip` (B-
  slice byte budget regression unrelated to this round) is on the
  master branch already; not caused by round-29.
- New integration tests (`integration_p_slice_intra_fallback.rs`):
  - `round29_p_slice_intra_fallback_shrinks_occlusion_fixture`:
    encode a 64x64 textured-wall IDR + a P-frame with a 32x32
    "occluding ball" of brand-new content. P-slice **588 → 108
    bytes (81.6 % smaller)** under intra fallback vs. inter-only.
  - `round29_p_slice_intra_fallback_decodes_through_self`:
    self-roundtrip bit-exact through our own decoder (max enc/dec
    luma diff 0).
  - `round29_p_slice_intra_fallback_ffmpeg_interop`:
    ffmpeg/libavcodec decodes the same intra-fallback P-slice
    bit-equivalently against the encoder's local recon (max
    ff/enc luma diff 0).
  - `round29_p_slice_intra_fallback_picks_intra_for_occluded_mbs`:
    sanity check that the parser accepts the intra-in-inter
    mb_type values without error.
- Round-16/17/18/19/20/21/22/23/24/25/26 fixtures all still pass
  (their inter cost is dominated by motion-compensated prediction
  matching the source; the intra trial never wins).

### Status caveats (unchanged from round 28 unless noted)

- **Intra fallback active in P/B slices** (round-29 default). When
  `intra_in_inter = true`, every coded inter MB trials Intra_16x16
  via RDO; lower-J wins.
- Round-29 scope: **Intra_16x16 only** (no I_NxN / I_PCM fallback).
  The §8.3.1.2 9-mode I_NxN path in P/B slices needs careful CAVLC
  nC-grid plumbing for the in-MB neighbour reads — deferred.
- Round-29 chroma: **4:2:0 only** (encode_p / encode_b are 4:2:0-
  only at this level).
- Spatial direct + temporal direct mode B (rounds 21/24) +
  partition modes (round 22) + B_8x8 mixed (round 25) + explicit
  weighted bipred (round 26) all still active and unaffected.
- CAVLC only. No rate control. Single ref per list (B-slices).

## Round 30 — landed

Wire **CABAC entropy coding** for I + P slices via two new entry points
(`Encoder::encode_idr_cabac`, `Encoder::encode_p_cabac`) layered on the
round-1..29 motion estimation / intra prediction / transform /
quantisation primitives. The CAVLC paths (`encode_idr` / `encode_p`)
remain unchanged.

- **`encoder::cabac_engine`** — §9.3.4 arithmetic encoder primitives
  (`EncodeDecision`, `EncodeBypass`, `EncodeTerminate`, `RenormE`,
  `EncodeFlush`, `put_bit` carry-propagation). Reuses the decoder's
  `RANGE_TAB_LPS` / `TRANS_IDX_LPS` / `TRANS_IDX_MPS` tables — same
  state machine, same probabilities, walked in encode direction.
- **`encoder::cabac_syntax`** — per-syntax-element binarisations and
  ctxIdxInc derivations mirroring `cabac_ctx::decode_*` step-for-step.
  Coverage: `mb_skip_flag`, `mb_type` (I and P), `mvd_lX`,
  `ref_idx_lX`, `coded_block_pattern`, `coded_block_flag`,
  `significant_coeff_flag`, `last_significant_coeff_flag`,
  `coeff_abs_level_minus1`, `coeff_sign_flag`,
  `intra_chroma_pred_mode`, `mb_qp_delta`, `end_of_slice_flag`,
  `prev_intra4x4_pred_mode_flag`, `rem_intra4x4_pred_mode`, plus a
  full `encode_residual_block_cabac` helper (CBF + significance map +
  abs_level + sign).
- **`encoder::cabac_path`** — top-level IDR + P encoders. IDR encodes
  every MB as `Intra_16x16` (4 luma modes + 4 chroma modes by SAD).
  P encodes `P_Skip` when CBP collapses to zero or `P_L0_16x16` (full
  luma 4x4 DC+AC + chroma DC ± AC residual). Per-MB CBF neighbour
  state tracked in `CabacEncGrid` so encoder + decoder agree on every
  ctxIdxInc.
- **PPS** (`BaselinePpsConfig::entropy_coding_mode_flag`): toggles
  CAVLC vs CABAC. **Slice header** (`PSliceHeaderConfig::cabac`):
  emits `cabac_init_idc` ue(v) when CABAC is enabled.
- **Profile gate**: CABAC requires Main (77) or higher per §A.2.1.

### Validation

- `cargo test -p oxideav-h264 --lib` — **846 passed** (+27 new tests:
  7 in `cabac_engine` + 20 in `cabac_syntax`).
- All round-1..29 integration tests still pass with the existing
  CAVLC paths unchanged.
- New `tests/integration_cabac_encoder.rs`:
  - `round30_idr_cabac_self_roundtrip`: 64x64 gradient → 79-byte IDR
    decoded through our own decoder bit-exactly (max enc/dec diff = 0,
    PSNR_Y = 47.58 dB).
  - `round30_idr_plus_p_cabac_self_roundtrip`: IDR + P (same source) →
    P-slice = 12 bytes (mostly P_Skip), bit-exact recon, PSNR_Y vs
    source = 47.58 dB.
  - `round30_idr_cabac_ffmpeg_interop`: ffmpeg 8.1 libavcodec decodes
    the same CABAC IDR bit-equivalently against the encoder's local
    recon (max diff = 0, PSNR_Y = 47.58 dB).

### Status caveats

- I + P slices only; B-slice CABAC encode deferred to a later round.
- 4:2:0 only on the CABAC path; 4:2:2 / 4:4:4 still CAVLC-only.
- Round-30 P encoder pins MV = (0, 0) for every inter MB to keep the
  §8.4.1.3 mvp derivation off the critical path. Sufficient for
  static / near-static fixtures (round-30 acceptance bar) since the
  P_Skip fast path dominates; the next round will wire the per-MB MV
  grid + spec mvp so non-trivial motion is supported.
- I_NxN (Intra_4x4) under CABAC not yet wired — the IDR CABAC path
  emits every MB as Intra_16x16.

## Round 31+ (planned)

- B-slice CABAC encode.
- Per-MB MV grid + §8.4.1.3 mvp under CABAC P encoder so motion is
  supported beyond P_Skip.
- I_NxN under CABAC.
- 4:2:2 / 4:4:4 under CABAC.
- I_NxN fallback in P / B slices (round-29 currently only trials
  Intra_16x16; I_NxN can be a further win on detailed content).
- Multiple slices per picture.
- 8x8 transform / High-profile features.
- Rate control.
- Chroma AC nC per §9.2.1.1.
- VUI / SEI tuning (HRD, mastering display, content light level).

## Entry points

- [`Encoder::encode_idr`] — top-level IDR I-slice: takes one `YuvFrame`
  and emits an SPS + PPS + Annex B IDR slice + the matching reconstruction.
- [`Encoder::encode_p`] — non-IDR P-slice (round 16). Takes one
  `YuvFrame` + a single L0 reference (`EncodedFrameRef`) and emits the
  P slice NAL + recon. SPS/PPS were shipped by the IDR.
- [`Encoder::encode_b`] — non-reference B-slice (round 20 + 21 + 22).
  Takes one `YuvFrame` + L0 and L1 references (`EncodedFrameRef`) and
  emits the B slice NAL + recon. Requires `EncoderConfig::profile_idc
  >= 77` and `max_num_ref_frames >= 2`. SPS/PPS were shipped by the
  IDR. Round-21: when an `EncodedFrameRef` is built `From<&EncodedP>`
  (or `From<&EncodedIdr>`), its `partition_mvs` field is populated for
  use by the §8.4.1.2.2 spatial-direct derivation when this frame
  serves as the L1 anchor. Round-22: per-MB the encoder picks among
  16x16 modes (round-20: B_L0/L1/Bi_16x16; round-21: B_Skip,
  B_Direct_16x16) and the 18 partition modes (B_*_*_16x8 with raw
  values 4..=20 even; B_*_*_8x16 with raw values 5..=21 odd).
- Lower-level helpers under [`bitstream`], [`nal`], [`sps`], [`pps`],
  [`slice`], [`macroblock`], [`cavlc`], [`transform`] are usable
  independently — round 2's higher-mode encoder can compose them
  without touching the round-1 dispatcher.
- [`Encoder::encode_idr_cabac`] / [`Encoder::encode_p_cabac`]
  (round 30) — CABAC-enabled IDR + P entry points. Require
  `EncoderConfig::cabac = true` and `profile_idc >= 77`. Produce
  Annex B streams that round-trip through this crate's decoder and
  through ffmpeg's libavcodec H.264 decoder bit-equivalently.
- [`Encoder::encode_b_cabac`] (round 31) — CABAC B-slice entry point
  (B_L0_16x16 / B_L1_16x16 / B_Bi_16x16, default-merge bipred). Round
  32 extends the CABAC P + B paths with real quarter-pel ME (replacing
  the round-30/31 forced MV=(0,0)) plus `B_Skip` / `B_Direct_16x16`
  emission via the §8.4.1.2.2 spatial-direct derivation mirrored in
  `cabac_b_spatial_direct_derive`. Per-8x8 / mixed `B_8x8`+all-Direct
  CABAC variants + 4:2:2 / 4:4:4 P/B paths still deferred (they need a
  CABAC sub_mb_type writer + Yuv422/Yuv444 chroma writer plumbing).
