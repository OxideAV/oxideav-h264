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

## Round 17+ (planned)

- Half-pel + quarter-pel ME (luma 6-tap + chroma bilinear refinement).
- 4MV (8×8 and smaller P sub-partitions).
- Intra fallback in P-slices via RDO.
- B-slices.
- Multiple slices per picture.
- 8x8 transform / High-profile features.
- CABAC.
- Rate control.
- Chroma AC nC per §9.2.1.1.

## Entry points

- [`Encoder::encode_idr`] — top-level: takes one `YuvFrame` and emits
  an Annex B byte stream + the matching reconstruction.
- Lower-level helpers under [`bitstream`], [`nal`], [`sps`], [`pps`],
  [`slice`], [`macroblock`], [`cavlc`], [`transform`] are usable
  independently — round 2's higher-mode encoder can compose them
  without touching the round-1 dispatcher.
