# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **slice_data: bound `mb_skip_run` and the macroblock walker against
  malformed streams.** §7.4.4 implicitly bounds `mb_skip_run` by
  `PicSizeInMbs - CurrMbAddr`, but the parser used to accept any
  ue(v) and grew the `macroblocks` Vec until the process OOM'd
  (~2.7 GB realloc reachable in CAVLC). A hard cap of 2^18 MBs (well
  above any Annex A level) on both `mb_skip_run` and `CurrMbAddr`
  rejects pathological streams with `SliceDataError::MbSkipRunOverflow`
  / `MbAddrOverflow` instead of panicking.
- **slice_header: cap `num_ref_idx_lX_active_minus1` at 63.** §7.4.3
  bounds the field at 31 (frame) / 63 (field); without enforcement an
  oversized value funneled into `parse_pred_weight_table` /
  `parse_ref_pic_list_modification` and drove a multi-gigabyte
  allocation. Defensive re-check covers PPS-inherited defaults too.
- **pps: cap `pic_size_in_map_units_minus1` (FMO type 6) at 2^22 − 2.**
  Previously fed unbounded into `Vec::with_capacity`, exposing a
  decoder-side OOM via a single PPS NAL (~3.2 GB realloc reachable).
- **cabac: bound the bypass Exp-Golomb escape suffix at `k = 30`.**
  Both `coeff_abs_level_minus1` (UEG0, §9.3.3.1.3) and `mvd_lX`
  (UEG3, §9.3.3.1.1.7) accumulated `k` without limit before computing
  `1u32 << k`, panicking with `attempt to shift left with overflow`
  in CABAC residual decode for any stream that fed ≥ 32 leading-one
  bins in a row. Now returns `CabacError::EgEscapeSuffixOverflow`.
- **scaling_list: enforce §7.4.2.1.1.1 `delta_scale ∈ −128..=127`
  bound.** Previously the parser quietly accepted any se(v) value and
  evaluated `last_scale + delta_scale + 256` in i32, which panicked
  with `attempt to add with overflow` for values fed via oversized
  se(v) codewords. Out-of-range deltas now return
  `ScalingListError::DeltaScaleOutOfRange`. Surfaced by fuzz CI run
  25642717757 once the bitstream-overflow gate landed.
- **sps: cap `pic_width_in_mbs_minus1` and
  `pic_height_in_map_units_minus1` at 510.** §A.3 level limits bound
  real streams well below this — Level 6.2 needs ~1056 mbs / axis,
  4K-class needs ~240 — but malformed inputs could previously feed
  arbitrarily large dimensions into `sps.pic_width_in_mbs() * 16`
  (panic: multiply with overflow), `pic_width_in_mbs() *
  frame_height_in_mbs()` (panic on mb_count), and `Picture::new`
  (multi-GB calloc → fuzz OOM at 7.5 GB). The chosen ceiling
  guarantees every downstream multiplication stays in u32 *and* the
  worst-case `Picture` allocation (3 × width × height at 4:4:4)
  stays under ~200 MB.
- **Exp-Golomb decoder no longer panics on 32+ leading zeros.**
  `BitReader::read_codenum` previously accepted up to 32 leading zero
  bits, then computed `1u32 << leading_zeros` — which is undefined
  behaviour in Rust at `leading_zeros == 32` and panicked with
  `attempt to shift left with overflow` in debug builds. The bound is
  now 31 (the largest `k` for which `2^k − 1 + suffix` still fits in
  u32), returning `BitError::ExpGolombOverflow` for malformed streams
  that ask for more. The crash was reachable via PPS parsing on
  fuzzer input (panic_free_decode crash
  `5c9d8c8642cdb8ffc371d3dbb45c3a1604477676`). Regression unit tests
  added in `src/bitstream.rs` cover both the overflow rejection path
  and the 31-zeros boundary that yields the largest legal codeNum.

## [0.1.4](https://github.com/OxideAV/oxideav-h264/compare/v0.1.3...v0.1.4) - 2026-05-06

### Other

- drop stale REGISTRARS / with_all_features intra-doc links
- drop dead `linkme` dep
- CABAC 4:4:4 IDR encode (round 33) + fix ctxIdx 460-1023 init
- registry calls: rename make_decoder/make_encoder → first_decoder/first_encoder
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-h264/pull/502))

### Fixed

- **CABAC context init Tables 9-25..9-33 (ctxIdx 460..=1023) added.**
  All nine missing tables — coded_block_flag, significant_coeff_flag,
  last_significant_coeff_flag, and coeff_abs_level_minus1 for 4:4:4
  chroma blockCat 6..=13 (Cb/Cr planes) plus 8x8 field/frame variants —
  were absent from `init_table()`. Every context in the 460..=1023 range
  fell back to the neutral (0,0) init (pStateIdx=62, valMPS=0), causing
  ffmpeg to fail decoding multi-MB 4:4:4 IDR CABAC streams with "top
  block unavailable for requested intra mode". All five round-33 4:4:4
  CABAC tests now pass including ffmpeg interop for 32×16 and 64×64.

## [0.1.3](https://github.com/OxideAV/oxideav-h264/compare/v0.1.2...v0.1.3) - 2026-05-05

### Other

- CABAC per-cell mixed B_8x8 (task #475)
- CABAC B 16x8 / 8x16 partition modes + B_8x8 all-Direct (task #475)
- CABAC real ME + B_Skip / B_Direct_16x16 (round 32)
- B-slice CABAC encode (round 31)

### Added

- **Encoder: CABAC per-cell mixed B_8x8 (task #475).** `encode_b_cabac`
  now picks per-quadrant `sub_mb_type ∈ {B_Direct_8x8, B_L0_8x8,
  B_L1_8x8, B_Bi_8x8}` for `mb_type=22` MBs, mirroring the CAVLC
  round-25 selector. Per-cell ME (`pick_best_b8x8_cell`) builds an
  L0 / L1 / Bi candidate per 8x8 quadrant and picks the lowest-
  Lagrangian-cost variant; mixed mode wins over `B_Direct_16x16` /
  `B_8x8`-all-Direct / explicit-16x16 / partition modes when its
  total cell SAD beats the current best by more than the
  +12-bit MB-header bias. Writer emits `mb_type=22` + per-cell
  `sub_mb_type` raw {0, 1, 2, 3} via `encode_sub_mb_type_b`, then
  per-cell mvd_l0[i] (cell order 0..3) + mvd_l1[i] (cell order 0..3)
  per §7.3.5.2 sub_mb_pred() field ordering.

  New `cabac_mvp_for_b_8x8_partition` derives §8.4.1.3 mvp for one
  8x8 sub-MB partition with `MvpredShape::Default` (no directional
  shortcut for 8x8 cells) using a per-cell inflight tracker
  `cabac_neighbour_partition_mv_b8x8` that distinguishes
  "cell decoded but list X unused" (returns `intra_but_mb_available`,
  partition_available=true) from "cell not yet decoded" (returns
  `partition_not_yet_decoded_same_mb`, partition_available=false) —
  the distinction matters for the §8.4.1.3.2 C→D substitution gate.
  Per-cell inflight is updated in raster order, mirroring the
  decoder's per-cell `process_partition` grid update; Direct cells
  contribute their §8.4.1.2.2-derived MVs / refs to the inflight even
  though they emit no mvd. The L1 mvd loop walks raster order with a
  fresh per-cell inflight to re-mirror the decoder's
  cells-0..(c-1)-only state at cell c's L1 mvp derivation point.

  Predictor build stitches per-cell 8x8 luma + 4x4 chroma predictors
  via `build_b_explicit_8x8_cell_luma` /
  `build_b_explicit_4x4_cell_chroma` (now `pub(crate)`); Direct
  cells build per-cell predictors from `direct.mv_l0_per_8x8[c]` /
  `direct.mv_l1_per_8x8[c]` directly (independent of the
  `direct_8x8_competitive` build, which only fires when
  `!direct_uniform`). Per-MB grid update writes per-cell `ref_idx_l*`
  + `mv_l*` + `abs_mvd_l*` so neighbour MBs reading our slot get the
  right §8.4.1.3 / §9.3.3.1.1.7 inputs. Per-4x4 deblock state
  (`MbDeblockInfo.mv_l*` / `ref_idx_l*` / `ref_poc_l*`) reflects the
  per-cell layout so the §8.7.2.1 bS derivation matches the decoder.

  ffmpeg cross-decode bit-exact (max diff 0) on the round-25 mixed-
  motion fixture (32×16 px, IDR + P + B, per-quadrant content
  offsets exercising all four cell kinds) at PSNR_Y = 44.20 dB.
  B-slice shrinks 99 → 73 bytes (~26 %) on the 64×64 round-475
  per-half-motion fixture vs the prior partition-only path. Encoder
  exits at ~72 % H.264 syntax coverage estimated by the task
  yardstick.

- **Encoder: CABAC B 16x8 / 8x16 partition modes + B_8x8 all-Direct
  (task #475).** `encode_b_cabac` now emits Table 7-14 mb_type ∈ 4..=21
  (the nine 16x8 + nine 8x16 partition variants) by mirroring the CAVLC
  partition-decision pipeline: per-cell `search_quarter_pel_8x8` ME on
  each list × four 8x8 cells, per-partition `best_partition_mode_b`
  pick across {L0, L1, Bi}, Lagrangian SAD compare against the 16x16
  candidates, and stitched composite predictor via the existing
  `build_b_partition_pred_luma_16x8` / `_8x16` (now `pub(crate)`).
  Per-partition mvd emission follows §7.3.5.1 ordering with within-MB
  inflight tracking for partition 1's MVP. New `cabac_enc_mvd_abs_sum`
  helper mirrors the decoder's `cabac_mvd_abs_sum` byte-for-byte so
  the §9.3.3.1.1.7 ctxIdxInc lands on the same context regardless of
  partition shape; `cabac_mvp_for_b_16x8_partition` /
  `cabac_mvp_for_b_8x16_partition` reuse the shared
  `derive_mvpred_with_d` machinery with the directional shortcuts
  (`Partition16x8Top` / `Bottom`, `Partition8x16Left` / `Right`). The
  per-MB deblock state (`MbDeblockInfo.mv_l0` / `mv_l1` / `ref_idx_*` /
  `ref_poc_*`) is now built per-4x4 from the partition layout so the
  internal partition boundary's bS derivation matches what the decoder
  computes (without this the decode/encode recon drifts ~2 LSB at the
  partition seam).

  Also adds **B_8x8 + 4× B_Direct_8x8** (mb_type=22) as a fallback
  whenever the §8.4.1.2.2 spatial-direct derivation produces non-uniform
  per-8x8 MVs: composite predictor stitched per 8x8 quadrant via
  `build_inter_pred_luma_8x8` / `build_inter_pred_chroma_4x4` (now
  `pub(crate)`); writer emits `mb_type=22` + 4× `sub_mb_type=0` (no
  ref_idx / mvd, decoder re-derives all four (mvL0, mvL1, refIdxL0,
  refIdxL1) per §8.4.1.2.2). New `encode_sub_mb_type_p` /
  `encode_sub_mb_type_b` helpers in `cabac_syntax.rs` mirror Table 9-37
  / 9-38 binarisations bin-for-bin against `decode_sub_mb_type_p` /
  `_b`. Per-cell mixed B_8x8 (per-quadrant L0 / L1 / Bi) deferred —
  the all-Direct path establishes the syntax scaffolding for the next
  round.

  ffmpeg cross-decode bit-exact (≤ 1 LSB) on per-half-motion fixtures
  forcing B_L0_L1_16x8 / B_L0_L1_8x16 (PSNR_Y ≥ 38 dB) and on the
  bipred-on-midpoint fixture exercising B_8x8 + all-Direct. Encoder
  now exits at ~62 → ~70 % H.264 syntax coverage estimated by the
  task's progress yardstick.

- **Encoder: CABAC real ME + `B_Skip` / `B_Direct_16x16` (round 32).**
  P/B CABAC paths replace the round-30/31 forced MV=(0, 0) with real
  quarter-pel motion estimation (`search_quarter_pel_16x16`) and emit
  proper §8.4.1.3 mvds against a median MV predictor. New
  `cabac_mvp_for_16x16` / `cabac_p_skip_mv` /
  `cabac_b_spatial_direct_derive` helpers track per-8x8 MVs / refIdxs
  in `CabacEncMbInfo` (`mv_l0_8x8` + `ref_idx_l0_8x8` + L1 mirrors) and
  read the §6.4.11.7 (A, B, C, D) neighbour cells through the existing
  `derive_mvpred_with_d` / `derive_p_skip_mv_with_d` /
  `derive_b_spatial_direct_with_d` infrastructure. P-CABAC now requires
  `chosen_mv == skip_mv` for P_Skip eligibility (no longer trivially
  always-skip). B-CABAC additionally implements `B_Skip` (mb_skip_flag
  with no further MB syntax) and `B_Direct_16x16` (mb_type=0 with no
  mvd / ref_idx); the §8.4.1.2.2 derivation is mirrored encoder-side
  and the direct candidate is picked over the explicit-inter rows
  (B_L0/L1/Bi_16x16) on a Lagrangian rate-bias-adjusted SAD. Per-8x8 /
  mixed `B_8x8`+all-Direct CABAC variants + 4:2:2 / 4:4:4 P/B still
  deferred. ffmpeg cross-decode bit-exact at QP 22 on translation-
  motion fixtures (PSNR_Y ≈ 99 dB on flat-object content,
  ≥ 46 dB on textured-background motion via
  `round32_b_cabac_textured_motion_ffmpeg_interop`); the smooth-content
  16-frame B-pyramid GOP shrinks from 555 → 487 bytes (~12 %) thanks
  to B_Skip / B_Direct gradually replacing the explicit B_L0_16x16
  emissions.

## [0.1.2](https://github.com/OxideAV/oxideav-h264/compare/v0.1.1...v0.1.2) - 2026-05-04

### Other

- add full_frame_freeze/release/snapshot + deblocking_filter_display_preference (§D.2.15-17, §D.2.22)
- add post_filter_hint payload type 22 (§D.1.24 / §D.2.24)
- rustfmt the film_grain test fixtures (CI green-up after 716fa7b)
- complete film_grain_characteristics per-component body (§D.2.21)
- derive level_idc from picture size instead of hard-coding 30
- rustfmt fixes from 4e0e6b4 CI red
- distinguish Level 1b from Level 1.1 in MaxDpbMbs lookup
- add user_data_registered_itu_t_t35 ([#4](https://github.com/OxideAV/oxideav-h264/pull/4)) + pan_scan_rect ([#2](https://github.com/OxideAV/oxideav-h264/pull/2))
- document why luma indexA clamp stays at 0 (no -QpBdOffsetY)
- thread bit_depth_chroma_minus8 through QPC derivation
- switch chroma QPC derivation to extended-range qPI for High10+
- rustfmt fixes from CI red on previous commit
- extend dequant + QP range for High10 / 4:2:2 / 4:4:4
- rustfmt docs_corpus.rs videoframe_to_packed pack() calls
- emit High10/High422/High444-Predictive 10/12-bit planes (task #259)
- clippy fixes for docs_corpus.rs
- rustfmt docs_corpus.rs
- wire docs/video/h264/ fixture corpus into integration test
- replace never-match regex with semver_check = false
- drop enable_miri input (miri now manual-only via workflow_dispatch)
- fix red-CI items from round 30
- grant release-plz shim contents+pull-requests write
- migrate to OxideAV/.github reusable workflows
- round 30 — encoder CABAC entropy coding for I + P slices
- round 29 — encoder intra fallback in P / B slices (Intra_16x16)
- round 28 — encoder 4:4:4 chroma (IDR Intra_16x16-only path)
- round 27 — encoder 4:2:2 chroma (IDR-only path)
- oxideav-core ^0.2 -> ^0.1 (0.2.0 was yanked)
- round 26 — encoder explicit weighted bipred (§7.4.2.2 + §8.4.2.3.2)
- round 24 — encoder B-slice §8.4.1.2.3 temporal direct mode
- round 23 — encoder B_8x8 + 4× B_Direct_8x8 (§8.4.1.2.2 spatial direct, Table 7-14 raw 22 / Table 7-18 raw 0)
- round 22 — encoder B-slice 16x8 / 8x16 partitions (Table 7-14 raw 4..=21)
- round 21 — encoder B_Skip / B_Direct_16x16 (§8.4.1.2.2 spatial direct)
- decoder round 6 — fix DPB output ordering on under-reported max_num_reorder_frames
- decoder round 5 — speed pass (2.01x on solana-ad.mp4)
- decoder round 4 — fix CABAC drift on High-profile B-slice content
- round 20 — encoder B-slice (B_L0 / B_L1 / B_Bi 16x16)
- round 19 — encoder 4MV (P_8x8 / sub_mb_type=PL08x8)
- round 18 — encoder quarter-pel ME (§8.4.2.2.1)
- half-pel ME + 6-tap interpolation (round 17)
- adopt slim VideoFrame shape
- P-slice support — single L0, integer-pel ME (round 16)
- Lagrangian RDO mode decision + intra4x4 blk-5 fix (round 15)
- in-loop §8.7 deblocking on local recon (round 14)
- I_NxN (Intra_4x4) all 9 modes + per-MB mb_type decision (round 4)
- I_16x16 luma AC residual transmit (round 3)
- I_16x16 V/H/DC/Plane modes + chroma residual (round 2)
- Baseline I_16x16 IDR scaffold (round 1)
- pin release-plz to patch-only bumps

### Added

- **Encoder: B-slice CABAC encode (round 31).** New
  `Encoder::encode_b_cabac` entry point covering non-reference B-slices
  with CABAC entropy coding (mirror of round-30's
  `encode_idr_cabac` / `encode_p_cabac`). Per-MB pick of {`B_L0_16x16`,
  `B_L1_16x16`, `B_Bi_16x16`} on minimum-luma-SAD against MV=(0, 0)
  predictors; bipred uses the §8.4.2.3.1 default merge `(L0 + L1 + 1)
  >> 1`. New `encode_mb_type_b` syntax helper (mirror of
  `decode_mb_type_b`) covers Table 9-37 B-slice rows 0..=22 + intra
  suffix (rows 23..=48). `BSliceHeaderConfig` gains `cabac:
  Option<CabacSliceParams>` to drive `cabac_init_idc` emission per
  §7.3.3, parallel to `PSliceHeaderConfig`. Validated by
  `round31_b_cabac_self_roundtrip` (decoder ↔ encoder recon match,
  PSNR_Y ≥ 45 dB) and `round31_b_cabac_pyramid_ffmpeg_interop`
  (16-frame B-pyramid GOP — 1 IDR + 7 P + 7 B — decoded by ffmpeg with
  PSNR_Y ≈ 51.5 dB on smooth content). `B_Skip` / `B_Direct_16x16` /
  4:2:2 / 4:4:4 P/B paths still deferred. New
  `CabacEncMbInfo::is_b_skip_or_direct` field threads the spec's
  §9.3.3.1.1.3 mb_type ctxIdxOffset=27 bin-0 condTerm correctly through
  the CABAC encoder grid.

- **SEI: `full_frame_freeze` (type 13), `full_frame_freeze_release`
  (type 14), `full_frame_snapshot` (type 15), and
  `deblocking_filter_display_preference` (type 20)**, covering §D.1.15 /
  §D.1.16 / §D.1.17 / §D.1.22 and the corresponding semantics clauses.
  All four are short single-element-or-flag-gated payloads:
  * `FullFrameFreeze { full_frame_freeze_repetition_period: u32 }` —
    one ue(v) field.
  * `FullFrameFreezeRelease` — empty payload, surfaced as a unit
    `SeiPayload::FullFrameFreezeRelease` variant.
  * `FullFrameSnapshot { snapshot_id: u32 }` — one ue(v) field.
  * `DeblockingFilterDisplayPreference { cancel_flag,
    display_prior_to_deblocking_preferred_flag,
    dec_frame_buffering_constraint_flag, repetition_period }` —
    cancel-flag-gated body. When `cancel_flag` is true the body is
    absent and the other fields default to zero / false per §D.2.22.
  Wired into `parse_payload` and into `SeiPayload`. 8 new unit tests
  including dispatcher round-trips and the empty-payload release path.

- **SEI: `post_filter_hint` (payload type 22, §D.1.24 / §D.2.24)**.
  Decodes the `filter_hint_size_y` / `filter_hint_size_x` ue(v)
  dimensions, the `filter_hint_type` u(2) (0 = 2-D FIR, 1 = pair of
  1-D FIRs, 2 = cross-correlation matrix), the `filter_hint_value
  [cIdx][cy][cx]` se(v) coefficients (`3 * size_y * size_x` entries
  flattened in spec order), and the trailing
  `additional_extension_flag` u(1). Spec constraints policed:
  * `filter_hint_size_y`, `filter_hint_size_x` ∈ 1..=15 →
    `SeiError::PostFilterHintSizeOutOfRange`;
  * `filter_hint_type == 3` (reserved) →
    `SeiError::PostFilterHintTypeReserved`;
  * `filter_hint_type == 1` ⇒ `filter_hint_size_y == 2` →
    `SeiError::PostFilterHintType1WrongHeight`.
  Wired into `parse_payload` and into `SeiPayload::PostFilterHint`.
  6 new unit tests (1x1 cross-correlation round-trip, 2x3 type-0 2-D
  FIR with 18 coefficients, type=1 wrong-height rejection,
  size-out-of-range rejection, reserved-type rejection, dispatcher
  wiring).

- **SEI: `film_grain_characteristics` per-component intensity-interval body
  (§D.1.21 / §D.2.21).** Previously `parse_film_grain_characteristics`
  stopped after the three `comp_model_present_flag` bits and dropped the
  per-component (`num_intensity_intervals_minus1`,
  `num_model_values_minus1`, `intensity_interval_lower_bound`,
  `intensity_interval_upper_bound`, `comp_model_value`) data plus the
  trailing `film_grain_characteristics_repetition_period`. The full body
  is now decoded:
  * New `FilmGrainBody::components: [Option<FilmGrainComponent>; 3]`
    populated in lockstep with `comp_model_present_flag` (Y, Cb, Cr).
  * New `FilmGrainComponent { num_model_values_minus1, intensity_intervals }`
    + new `FilmGrainIntensityInterval { lower_bound, upper_bound,
    model_values: Vec<i32> }`. `comp_model_value[c][i][j]` is read as
    se(v) per the spec.
  * New `FilmGrainBody::repetition_period: u32`
    (`film_grain_characteristics_repetition_period`, ue(v)).
  * Per §D.2.21 `num_model_values_minus1[c]` shall be in 0..=5; out-of-
    range values produce
    `SeiError::FilmGrainNumModelValuesOutOfRange { component, got }`.
  * Two new round-trip tests (multi-interval / multi-model-value with
    signed `comp_model_value` entries; out-of-range rejection); the two
    pre-existing tests now extend through the new body.

- **SEI: `user_data_registered_itu_t_t35` (payload type 4) + `pan_scan_rect`
  (payload type 2)** (§D.1.4 / §D.2.4, §D.1.6 / §D.2.6). Both are
  upgraded from `SeiPayload::Unknown { payload, .. }` round-tripping
  to fully-typed payloads in the `parse_payload` dispatch.
  * `parse_user_data_registered_itu_t_t35` returns
    `(country_code, country_code_extension, payload_bytes)`. Country
    code 0xFF triggers the §D.1.6 / Recommendation ITU-T T.35 §3.1
    extension-byte read; an empty payload — or a 0xFF country code
    with no extension byte to follow — is rejected with
    `SeiError::RegisteredUserDataMissingCountryCode`. Real-world
    consumers (ATSC A/53 closed captions, AFD, ATSC bar data,
    HDR10+ dynamic metadata, Dolby Vision RPU) all ride on this
    payload, distinguished by the country / provider tuple plus the
    leading bytes of `payload_bytes`.
  * `parse_pan_scan_rect` returns `pan_scan_rect_id`, `cancel_flag`,
    and (when not cancelled) a `Vec<PanScanRectOffsets>` plus a
    `repetition_period`. The §D.2.4 `pan_scan_cnt_minus1 ∈ 0..=2`
    constraint is policed lazily — counts up to 31 are accepted but
    32+ rejected with `SeiError::PanScanCountTooLarge` to avoid
    pathological allocations on malformed streams.
  * 11 new unit tests (`registered_user_data_*` ×5, `pan_scan_rect_*`
    ×3, `parse_payload_dispatches_*` ×3) including hand-derived
    Exp-Golomb encodings of the offsets test vector.

- **Decoder honours SPS `bit_depth_luma_minus8` / `bit_depth_chroma_minus8`
  in `picture_to_video_frame`** (task #259). When a High10 / High 4:2:2
  / High 4:4:4 Predictive stream declares 9..=14-bit samples, the
  reconstructed planes are emitted as little-endian u16 (two bytes per
  sample) clamped to `(1 << bit_depth) - 1`, matching the
  `oxideav_core::PixelFormat::{Yuv420P10Le, Yuv422P10Le, Yuv444P10Le,
  Yuv420P12Le, Yuv422P12Le, Yuv444P12Le}` plane layouts. 8-bit streams
  are unchanged (one byte per sample, clamped to `0..=255`). The
  previously-`#[ignore]`d `corpus_10_bit_high10` integration test now
  runs and scores against the 16-bit-per-sample `expected.yuv`
  reference; it stays `Tier::ReportOnly` until the §8.5.10 inverse
  transform / dequant path widens to >8-bit arithmetic. Spec refs:
  §7.4.2.1.1, §6.4.2.1.

- **Round 30 — encoder CABAC entropy coding for I + P slices**.
  Adds the H.264 §9.3 CABAC arithmetic encoder (engine + per-syntax
  binarisations + ctxIdx derivation) and wires up two new entry points
  on `Encoder`: `encode_idr_cabac` and `encode_p_cabac`. PPS now carries
  `entropy_coding_mode_flag = 1` when [`EncoderConfig::cabac`] is enabled
  (Main profile or higher — Baseline forbids CABAC per §A.2.1).
  Implementation:
  * New module `encoder::cabac_engine`: §9.3.4 `EncodeDecision`,
    `EncodeBypass`, `EncodeTerminate`, `RenormE`, plus the
    carry-propagation `put_bit` / outstanding-bit mechanism. Reuses the
    decoder's `RANGE_TAB_LPS`, `TRANS_IDX_LPS`, `TRANS_IDX_MPS` tables —
    same probabilities, just walked in encode direction.
  * New module `encoder::cabac_syntax`: per-element encoders mirroring
    the decoder's `cabac_ctx::decode_*` step-for-step
    (`encode_mb_skip_flag`, `encode_mb_type_i` / `_p`,
    `encode_mvd_lx`, `encode_ref_idx_lx`, `encode_coded_block_pattern`,
    `encode_coded_block_flag`, `encode_significant_coeff_flag`,
    `encode_last_significant_coeff_flag`,
    `encode_coeff_abs_level_minus1`, `encode_coeff_sign_flag`,
    `encode_intra_chroma_pred_mode`, `encode_mb_qp_delta`,
    `encode_end_of_slice_flag`, `encode_residual_block_cabac`).
  * New module `encoder::cabac_path`: top-level IDR + P encoders
    layered on existing motion estimation / intra prediction /
    transform / quantisation primitives. IDR encodes every MB as
    Intra_16x16 with all four §8.3.3 modes trialled by SAD + chroma DC
    / V / H / Plane. P encodes P_Skip when CBP collapses to zero or
    P_L0_16x16 (single-ref, MV pinned to 0 for round-30 to keep the
    §8.4.1.3 mvp derivation off the critical path; sufficient for the
    P_Skip-dominated round-30 fixtures).
  * `BaselinePpsConfig::entropy_coding_mode_flag`: PPS gains a CABAC
    toggle. The slice header writer (`write_p_slice_header`) gains a
    `cabac` parameter that emits `cabac_init_idc` ue(v) when present
    (P/SP/B only).
  * Per-MB CBF neighbour tracking (`CabacEncGrid` +
    `cbf_neighbour_mb_level` / `cbf_neighbour_luma_ac` /
    `cbf_neighbour_chroma_ac`) mirrors the decoder's
    `CabacNeighbourGrid` so encoder + decoder agree on every
    `coded_block_flag` ctxIdxInc derivation. Internal-to-MB neighbour
    reads consult the running CBF state of the current MB; external
    neighbour reads consult the previously-encoded MB's grid slot,
    with the §9.3.3.1.1.9 "transBlockN unavailable" rule firing on
    the cbp_luma 8x8-quadrant gate for `Luma4x4` / `Luma16x16Ac`
    block types.
  Validation:
  * 27 new unit tests in `encoder::cabac_engine` (7) and
    `encoder::cabac_syntax` (20) — every primitive round-trips through
    the decoder bin-for-bin, including LPS-runs that exercise the
    `outstanding_bits` carry path, alternating bins, mixed
    decision/bypass/terminate sequences, and pseudo-random 200-bin
    streams.
  * `tests/integration_cabac_encoder.rs` — 3 integration tests:
    `round30_idr_cabac_self_roundtrip` (47.58 dB Y on 64x64 gradient,
    max enc/dec diff = 0), `round30_idr_plus_p_cabac_self_roundtrip`
    (IDR 79 bytes + P 12 bytes, 47.58 dB Y, max enc/dec diff = 0),
    `round30_idr_cabac_ffmpeg_interop` (ffmpeg 8.1 libavcodec decodes
    bit-equivalently against encoder local recon, max diff = 0).
  Status caveats:
  * I + P slices only; B slices not yet wired to the CABAC path.
  * 4:2:0 only; 4:2:2 / 4:4:4 still CAVLC-only.
  * Round-30 P encoder forces MV = (0, 0) for every inter MB to
    sidestep the §8.4.1.3 mvp derivation. Sufficient for the round-30
    acceptance fixture (≥45 dB on synthetic gradient) since static /
    near-static content hits the P_Skip fast path; future rounds will
    wire the per-MB MV grid + spec mvp derivation for moving content.
  * I_NxN (Intra_4x4) under CABAC deferred — the IDR CABAC path emits
    every MB as Intra_16x16.

- **Round 29 — encoder intra fallback in P/B slices (Intra_16x16)**.
  When [`EncoderConfig::intra_in_inter`] is true (default), the per-MB
  RDO loop in `encode_p` and `encode_b` now compares the inter cost
  `J_inter = D_inter + λ·R_inter` against an `Intra_16x16` trial cost
  `J_intra` and installs the lower-J winner. This covers occlusion,
  scene-cut frames, and high-detail areas where motion-compensated
  prediction's residual is more expensive to code than a fresh intra
  block. Per H.264 §7.3.5 the chosen MB is emitted with the intra
  mb_type values shifted by **+5** (P-slice Table 7-13 raw mb_type ∈
  6..=29 for Intra_16x16) or **+23** (B-slice Table 7-14 raw mb_type ∈
  24..=47); the existing decoder already handles intra MBs in P/B
  slices so no decoder-side change is required.
  Implementation:
  * New writer `write_intra16x16_mb_in_inter_slice` in
    `encoder::macroblock` — identical layout to `write_intra16x16_mb`
    but with a configurable `mb_type_offset` (5 or 23).
  * New `InterMbStateSnapshot` type wraps the existing `MbStateSnapshot`
    with the inter-context-only `MvGridSlot` (and optional L1 grid for
    B-slice) plus `pending_skip` counter so the trial RDO can roll
    back the loser cleanly.
  * New methods `Encoder::encode_p_mb_with_intra_fallback` /
    `encode_b_mb_with_intra_fallback` snapshot pre-state, run the
    inter trial (existing `encode_p_mb` / `encode_b_mb`), measure
    `J_inter` from `D_inter = SSD(source, recon)` and `R_inter =
    bits_emitted` delta, snapshot post-inter state, restore pre-state,
    run the new `encode_*_mb_intra16x16` trial, measure `J_intra`,
    and install the winner.
  * Shared emit helper `Encoder::encode_intra16x16_in_inter_slice`
    runs §8.3.3 luma mode RDO + §8.3.4 chroma mode SAD, builds the
    forward + quantize + inverse pipeline, writes the MB syntax, and
    updates `recon_*` / `nc_grid` / `intra_grid`. Caller updates
    `mv_grid` (intra MB → `is_intra = true`) so subsequent inter MBs'
    §8.4.1.3.2 step-2 collapse fires (mv=0, ref=-1).
  Round-29 scope:
  * `Intra_16x16` only (no `I_NxN` / `I_PCM` fallback). The §8.3.1.2
    9-mode I_NxN path in P/B slices needs careful CAVLC nC-grid
    plumbing for the in-MB neighbour reads — deferred.
  * 4:2:0 chroma only (encode_p / encode_b are 4:2:0-only).
  * `P_Skip` / `B_Skip` fast paths still trigger normally — when the
    inter trial decides the MB is a skip, `J_inter` is essentially
    zero (no syntax + zero residual SSD) and intra never wins.
  Acceptance fixture: `integration_p_slice_intra_fallback.rs`
  encodes a 64x64 textured-wall IDR + a P-frame with a 32x32
  "occluding ball" of brand-new content. Result: P-slice **588 → 108
  bytes (81.6% smaller)** under intra fallback, ffmpeg/libavcodec
  cross-decode bit-equivalent (max diff 0), self-roundtrip bit-exact
  through our own decoder. New `EncoderConfig::intra_in_inter`
  toggle (default `true`) lets callers A/B test against the inter-
  only path. P-only / B-only fixtures from rounds 16..28 are
  unaffected (their inter cost is well below any intra trial cost).

- **Round 28 — encoder 4:4:4 chroma support (IDR Intra_16x16-only path)**.
  `EncoderConfig::chroma_format_idc` now also accepts 3 (4:4:4); when
  paired with `profile_idc=244` (High 4:4:4 Predictive) the SPS writer
  emits the §7.3.2.1.1 chroma-extended group with `chroma_format_idc=3`
  and `separate_colour_plane_flag=0` (the decoder maps that pair to
  `ChromaArrayType=3`). Per-MB chroma planes are encoded "like luma"
  per §7.3.5.3 / §8.3.4.1: each plane has its own 16x16 DC Hadamard
  block + 16 4x4 AC blocks (§6.4.3 raster-Z), shares the luma
  Intra_16x16 prediction mode, and the `intra_chroma_pred_mode` u(v)
  field is omitted from the bitstream. New helper
  `encode_intra16x16_chroma_plane_luma_like` runs the luma transform
  pipeline on a chroma plane and writes its decoded-bit-exact recon
  back to `recon_u`/`recon_v`. Per §7.3.5.3 the per-plane chroma AC
  emit is gated by `cbp_luma == 15`, so the encoder promotes
  `cbp_luma` to 15 whenever any chroma plane has nonzero AC. Round-28
  scope:
  * IDR Intra_16x16 only (I_NxN for 4:4:4 is rejected).
  * `disable_deblocking_filter_idc=1` is forced in the slice header
    (§8.7 chroma deblock for ChromaArrayType==3 uses the LUMA filter
    ladder applied per-plane — separate plumbing exercise).
  * `separate_colour_plane_flag=0` only (separate-plane coding is
    out of scope).
  * P/B-slice paths still pin `chroma_format_idc ∈ {1}`.
  Decoder side: lifted the blanket `chroma_array_type==3` rejection in
  `reconstruct_slice_no_deblock`; new helper
  `reconstruct_chroma_intra_444` reconstructs each chroma plane via
  the §8.3.3 16x16 luma predictor + §8.5.10 Hadamard-DC + 16 AC
  blocks. New writer variant `ChromaWriteKind::Yuv444` carries the
  per-plane DC/AC arrays + their CAVLC nC contexts. New tests:
  * `sps::tests::high_444_sps_emits_chroma_extended_group_with_separate_colour_plane_flag`
  * Integration: `round28_444_encode_decode_self_roundtrip_matches_local_recon`
    (self-roundtrip + bit-exact decoder match, PSNR Y=42.25 / U=V=50.85
    on a 64x64 gradient).
  * Integration: `round28_444_ffmpeg_decode_matches_encoder_recon`
    (ffmpeg cross-decode bit-exact for all three planes,
    Y PSNR ≥ 40 dB acceptance threshold).

- **Round 27 — encoder 4:2:2 chroma support (IDR-only path)**.
  `EncoderConfig::chroma_format_idc` selects between 1 (4:2:0,
  default) and 2 (4:2:2). When set to 2, the SPS writer emits a High
  4:2:2 (`profile_idc=122`) sequence parameter set with the
  §7.3.2.1.1 chroma-extended group fields (chroma_format_idc=2,
  bit_depth_luma/chroma_minus8=0, no scaling matrix). The per-MB
  chroma path now dispatches to either:
  * 4:2:0 (`encode_chroma_residual`) — 4 4x4 sub-blocks per plane,
    2x2 Hadamard DC, `ChromaDc420` CAVLC context, 8x8 chroma tile;
    OR
  * 4:2:2 (`encode_chroma_residual_422`) — 8 4x4 sub-blocks per
    plane, 4x2 Hadamard DC (§8.5.11.2 / eq. 8-325), `ChromaDc422`
    CAVLC context, 8x16 chroma tile.
  Chroma intra prediction adds `predict_chroma_8x16` covering DC, H,
  V, Plane modes for the 4:2:2 8x16 block (§8.3.4 with `yCF=4` /
  `c=5*V` per eq. 8-147 / 8-149). The §8.7 deblock walker now uses
  `Picture::chroma_array_type` derived from the encoder config so
  chroma 4x4 edges land in the right pixel positions for both
  subsampling modes. New tests:
  * `transform::tests::chroma_dc_422_uniform_residual_round_trips_within_tolerance`
  * `transform::tests::chroma_dc_422_scan_matches_decoder_inverse`
  * `sps::tests::high_422_sps_emits_chroma_extended_group`
  * Integration: `round27_422_encode_decode_self_roundtrip_matches_local_recon`
    (self-roundtrip + bit-exact decoder match, luma PSNR ≥ 40 dB
    on a 64x64 gradient).
  * Integration: `round27_422_ffmpeg_decode_matches_encoder_recon`
    (ffmpeg cross-decode bit-exact for both planes).
  P-slice / B-slice paths still pin `chroma_format_idc=1` (4:2:0).

### Fixed

- **Encoder no longer emits `level_idc = 30` for HD/4K streams**
  (§7.4.2.1.1 / Annex A Table A-1). The pre-fix encoder hard-coded
  `level_idc = 30` (Level 3, MaxFS = 1620 mb = 720x576 SD) for
  every emitted SPS regardless of picture size, producing
  spec-non-conformant streams when encoding at 720p (3600 mb,
  needs 3.1) / 1080p (8160 mb, needs 4.0) / 4K UHD (32400 mb,
  needs 5.1) / 8K (129600 mb, needs 6.0). Strict decoders are
  entitled to reject those streams.
  * New `EncoderConfig::level_idc` field (u8). Default for new
    configs is derived from `(width, height)` via the new
    `min_level_idc_for_picture_size` helper, which walks Annex A
    Table A-1 low-to-high and picks the first level whose `MaxFS`
    covers the picture size in macroblocks. Includes Levels 1
    through 6.2 (the full 2024-08 set).
  * Hard-coded `level_idc: 30` removed from the two production
    SPS-emit sites: `Encoder::encode_idr` (CAVLC path,
    `src/encoder/mod.rs`) and `Encoder::encode_idr_cabac` (CABAC
    path, `src/encoder/cabac_path.rs`). Test fixtures using small
    synthetic dimensions (64x64) now correctly signal `level_idc = 10`.
  * 3 new unit tests
    (`encoder::level_derivation_tests::min_level_idc_picks_smallest_covering_level`,
    `encoder_config_default_level_matches_picture_size`,
    `non_aligned_dimensions_round_up_to_whole_mbs`) covering the
    full Table A-1 walk plus dimension-rounding edge cases.

- **Annex A Table A-1 — Level 1b vs Level 1.1 disambiguation**
  (§A.3.4.1). `max_dpb_mbs_for_level` now takes
  `constraint_set3_flag` and returns 396 (Level 1b) instead of 900
  (Level 1.1) when `level_idc == 11 && constraint_set3_flag == 1`.
  The shorthand `level_idc == 9` (Baseline / Constrained Baseline
  Level 1b form) also returns 396. Fixes a real over-allocation in
  the §A.3.1 / §C.4 DPB reorder-window floor: a 176x144 QCIF
  Level 1b stream (real PicSize = 99 MBs) now floors at 4 reorder
  frames, not 9, matching the spec cap. Two new regression tests
  (`max_dpb_mbs_level_1b_versus_level_1_1`,
  `level_1b_qcif_stream_sized_at_level_1b_cap_not_level_1_1`).

- **DPB output ordering when a real-world encoder under-reports
  `max_num_reorder_frames` in SPS VUI** (§E.2.1 / §A.3.1).
  `output_dpb_sizing` now floors the reorder window at the
  level-derived MaxDpbMbs/PicSize cap (§A.3.1 item h, Annex A
  Table A-1) so a stream whose actual reorder depth exceeds the
  encoder's claim is no longer bumped out of POC order. The
  textbook reproducer is solana-ad.mp4 (High@L3.1 720p, 4-deep
  B-pyramid): VUI claims `reorder=2` but the stream needs 5,
  and the §C.4 bumping process emits frames in display order
  only when the queue can hold all five. Pre-fix `oxideplay`
  saw frames arrive in something close to decode order with one
  trailing B-frame per 4 anchors; post-fix every frame on the
  3960-frame clip arrives with strictly monotonically-increasing
  pts. Two new tests (`dpb_sizing_raises_undersized_vui_to_level_cap`,
  `b_pyramid_emits_in_poc_order_at_level_31_720p`) pin the new
  behaviour.

### Performance

- decoder (round 5 — speed pass): **2.01× end-to-end on solana-ad.mp4**
  (1280×720 yuv420p High@3.1, 3960 frames, ~158s of content) —
  wall-time **89.83s → 44.76s** via `oxideplay --vo hash --ao null`,
  hash digest unchanged (`da9c18e4008cfd37`, regression-checked).
  Three independent wins, smallest-to-biggest:
  1. **`src/simd/` skeleton + `chunked` interpolate_luma /
     interpolate_chroma** matching the `oxideav-mpeg4video` pattern
     (scalar / chunked / nightly portable). The 6-tap luma FIR is
     rewritten so the H-FIR row strip is computed exactly once and
     fed to the V-FIR for the diagonal "j" position; the per-pixel
     edge-clip is hoisted out of the FIR. Bench numbers (1000 ×
     16×16 blocks): scalar 222 µs → chunked 36 µs (integer copy,
     **6.16×**); 798 µs → 109 µs (H-half-pel, **7.32×**); 805 µs →
     98 µs (V-half-pel, **8.21×**); **3645 µs → 161 µs (diagonal-j,
     22.64×)**. End-to-end contribution on solana-ad: ~4 % (~3.6 s)
     because the luma interpolator wasn't the actual hotspot at
     720p — but the kernel is now SIMD-shaped for the round-6
     widening to 1080p / 4K.
  2. **Cached debug-env-var lookups in CABAC + macroblock-layer +
     reconstruct hot paths.** Several `std::env::var(...)` calls
     (`OXIDEAV_H264_BIN_TRACE`, `OXIDEAV_H264_CABAC_DEBUG`,
     `OXIDEAV_H264_MB_TRACE`, `OXIDEAV_H264_RECON_DEBUG`,
     `OXIDEAV_H264_NO_DEBLOCK`, etc.) were resolving via a `getenv()`
     syscall on **every CABAC bin decode and every per-MB parse**.
     With ~140 M bin decisions in this 3960-frame clip that
     dominated decoder wall-time. Each var is now resolved once on
     first access and cached behind a `OnceLock<bool>` /
     `OnceLock<Option<u32>>`. End-to-end contribution: ~21 %
     (85.89 s → 68.12 s).
  3. **Eliminated per-MB clone of the CABAC neighbour grid in
     `parse_residual_cabac_only`.** The function previously cloned
     the entire `CabacNeighbourGrid` (~720 KiB on 720p) at the top
     of every macroblock parse to dodge the borrow checker between
     read-only neighbour CBF lookups and the end-of-MB writeback.
     Replaced with `nb_grid.as_deref()` for a shared re-borrow that
     NLL drops before the writeback. **Eliminates ~10 TB of
     `_platform_memmove` traffic** across the full file.
     End-to-end contribution: ~35 % (68.12 s → 44.76 s).
- decoder: deblocking-filter fast paths in `filter_vertical_edge_luma`
  / `filter_horizontal_edge_luma` / `filter_chroma_vertical_rows` /
  `filter_chroma_horizontal_cols` — when the 8-sample window across an
  edge is fully inside the picture (the common case for picture-
  internal edges), bypass the clipped `Picture::luma_at` /
  `set_luma` accessor pair and read/write the contiguous slice
  directly. Slow-path clipping retained for edges that straddle the
  picture boundary so semantics are unchanged.

### Fixed

- decoder (round 4): CABAC parse failures on real-world High-profile
  B-slice content with `transform_size_8x8_flag=1`. Two related bugs:
  (1) Table 9-24 (ctxIdx 402..=459 — Luma8x8 frame/field
  significant_coeff_flag, last_significant_coeff_flag, and
  coeff_abs_level_minus1) was missing from the CABAC context init
  pipeline, leaving every cat-5 context at the neutral
  `(m=0, n=0)` → `pStateIdx=62, valMPS=0` collapse; the arithmetic
  decoder drifted out of sync within a few hundred macroblocks. (2)
  §9.3.3.1.1.9 cat=1/2 transBlockN derivation: when the neighbour MB
  has `transform_size_8x8_flag=1` and the cbp bit covering
  `luma4x4BlkIdxN` is set, the spec assigns the neighbour's 8x8 luma
  block as transBlockN — its `coded_block_flag` is inferred to 1 from
  the CBP for ChromaArrayType < 3. Our `cabac_cbf_cond_terms` was
  short-circuiting to "transBlockN unavailable" instead. On
  `solana-ad.mp4` (1280×720 yuv420p High@3.1, 3960 frames, all B/P),
  slice-skip errors went from ~1900 to **0** end-to-end. Adds 4
  regression tests pinning Table 9-24 init values via §9.3.1.1
  eq. 9-5 hand-derivation.

### Added

- encoder: **B-slice §8.4.1.2.3 temporal direct mode (round 24)** —
  alongside the round-21/23 spatial direct path, the encoder can now
  emit B_Skip / B_Direct_16x16 / B_8x8+all-B_Direct_8x8 macroblocks
  driven by **temporal direct** MV derivation: per-8x8 (mvL0, mvL1)
  computed by POC-distance scaling of the L1 anchor's per-block
  colocated L0 motion vectors (eq. 8-191..8-202), with refIdxL0 /
  refIdxL1 collapsed to 0 for our single-ref-per-list setup
  (MapColToList0 collapses to identity since the L1 anchor's only L0
  reference is the IDR — same as the current B's L0 reference). Slice
  header now signals `direct_spatial_mv_pred_flag = 0`. New
  `EncoderConfig::direct_temporal_mv_pred` toggle (default `false` =
  spatial); new `EncodedFrameRef::pic_order_cnt` field carrying each
  reference's POC (auto-populated for `EncodedIdr` / `EncodedP` /
  `EncodedB` via the existing `From` impls). The encoder mirrors the
  decoder's §8.4.1.2.3 derivation locally (`b_temporal_direct_derive`)
  and re-uses the existing BMode chooser + `B_Direct_16x16` / `B_8x8`
  writers — only the MV/refIdx derivation differs from spatial. Five
  new integration tests (`integration_b_temporal_direct.rs`) pin the
  new behaviour: slice-header signalling check, 5-frame I-P-B-P-B
  self-roundtrip (max enc/dec diff ≤ 1 LSB on both B-frames),
  ffmpeg-interop on the same fixture (max diff ≤ 1, ≥ 38 dB PSNR_Y on
  every B), plus three boundary checks pinning the
  `derive_b_temporal_direct` helper at zero-mvCol / td-zero passthrough
  / symmetric-midpoint inputs. Round-24 caveats: still single ref per
  list (multi-ref temporal direct needs full §8.2.4 RefPicList
  construction + the MapColToList0 POC search — deferred), CAVLC only,
  no field/MBAFF, no long-term refs.
- encoder: **B_8x8 with all four `sub_mb_type = B_Direct_8x8` (round
  23)** — Table 7-14 raw mb_type 22 (`B_8x8`) + Table 7-18 raw
  sub_mb_type 0 (`B_Direct_8x8`) ×4. Zero MV / refIdx fields in the
  bitstream — the decoder re-derives all four per-8x8 (mvL0, mvL1,
  refIdxL0, refIdxL1) per §8.4.1.2.2 (spatial direct, since
  `direct_spatial_mv_pred_flag = 1` in our slice header). With
  `direct_8x8_inference_flag = 1` (hard-wired in our SPS), each 8x8
  sub-MB carries one (mvL0, mvL1) pair shared across its four 4x4
  sub-partitions. The encoder mirrors the §8.4.1.2.2 derivation locally
  (already present from round 21), now exercises the per-8x8 MV grid
  it produces by stitching four independent 8x8 luma + 4x4 chroma
  predictors into a 16x16 / 8x8 buffer and trial-coding it as a third
  Direct candidate (uniform `B_Direct_16x16` from round 21 + per-8x8
  `B_8x8`+all-Direct from round 23 + explicit/partition modes from
  round 22). Lifts the round-21 "uniform-only" gate that previously
  rejected Direct when the §8.4.1.2.2 step-7 colZeroFlag drove
  per-partition MVs to disagree. Lagrangian rate bias `(qp_y/6+1)*12`
  SAD units favours the smaller-syntax `B_Direct_16x16` over per-8x8
  Direct when the per-8x8 SAD doesn't materially beat the uniform one
  (per-8x8 syntax cost: mb_type=22 ue(9) + 4× sub_mb_type=0 ue(1) = 13
  bits vs B_Direct_16x16's 1-bit mb_type=0). New writer
  `write_b_8x8_all_direct_mb` + config `B8x8AllDirectMcbConfig`. The
  encoder's per-8x8 grid update + per-4x4 deblock-walker MV array
  honour the §8.4.1.2.2 derivation's per-8x8 MVs so the decoder's bs
  derivation sees the same MVs as the encoder. Three new integration
  fixtures (`integration_b_8x8_direct.rs`) verify bit-exact ffmpeg /
  libavcodec interop on a synthetic IPB midpoint fixture, plus two new
  unit tests for the writer (Table 7-14 / 7-18 round-trip + 14-bit
  zero-residual emit count). Round-23 caveats: spatial direct only
  (no temporal direct §8.4.1.2.3), CAVLC only, no per-sub-MB-partition
  granularity (B_8x8 paths with mixed sub_mb_type ∈ {B_Direct_8x8,
  B_L0_8x8, B_L1_8x8, B_Bi_8x8, …} are out of scope).
- encoder: **B-slice 16x8 / 8x16 partition modes (round 22)** — Table
  7-14 raw mb_types 4..=21 (`B_L0_L0_16x8` through `B_Bi_Bi_8x16`). Per-
  partition refIdx, MV, prediction mode (L0 / L1 / Bi), and §8.4.1.3
  partition-shape MVP derivation (`Partition16x8Top/Bottom`,
  `Partition8x16Left/Right`). Per-partition ME candidates pulled from the
  3-way set `{16x16-best-MV, cell-A-best-MV, cell-B-best-MV}`; mode
  decision enumerates the 9 `(mode_a, mode_b)` combos per shape and
  picks the minimum-SAD pair. Partition mode wins over 16x16 only when
  its joint SAD beats 16x16 by more than `(qp_y/6 + 1) * 8` SAD units —
  the Lagrangian rate penalty for the larger mb_type ue(v) plus the
  duplicated mvds. Bipred uses the same §8.4.2.3.1 default weighted
  average `(L0+L1+1)>>1` as round-20. The encoder's per-MB
  `MvGridSlot` now carries genuinely per-8x8 MVs (top half = #0/#1,
  bottom = #2/#3 for 16x8; left = #0/#2, right = #1/#3 for 8x16); the
  §8.4.1.3.2 C→D substitution and the per-partition `LUMA_4X4_BLK`
  Z-scan indexing of the deblock-walker's MV array are honoured so the
  decoder's bs derivation sees the same MVs as the encoder. Two new
  integration fixtures (`integration_b_partitions.rs`) verify
  bit-exact ffmpeg/libavcodec interop on a per-half-motion B-frame
  (max diff 0, 46.87 dB PSNR_Y on 16x8 split, 48.54 dB on 8x16). The
  encoder's `neighbour_mvs_16x16` now reads each neighbour's per-8x8
  cell that spatially abuts the current MB's top-left 4x4 (`#1` for
  left, `#2` for above and above-right, `#3` for above-left), matching
  the decoder's per-partition MV grid for partition-mode neighbours.
  Round-22 caveats: partition mode never overrides Direct (B_Skip
  beats partition by 0 vs ~10+ bits of mb_type / mvd cost), CAVLC only,
  single ref per list.
- encoder: **B_Skip / B_Direct_16x16 spatial-direct mode (round 21)**
  on top of round-20 explicit-inter. The encoder now mirrors the
  §8.4.1.2.2 derivation locally using its own L0 / L1 MV grids plus
  the L1 anchor's per-block colocated MVs (new
  `EncodedFrameRef::partition_mvs`, populated by `encode_p`). When the
  derived predictor's SAD is competitive with the explicit-inter
  candidates (under a Lagrangian rate bias of `(qp_y/6 + 1) * 16` SAD
  units), the encoder emits **B_Direct_16x16** (raw mb_type 0, no mvds
  / ref_idx in the bitstream) — or **B_Skip** if the residual quantises
  to zero (folded into a `mb_skip_run` ue(v)). On a 16-MB static-content
  B-frame (IDR + P + B all from the same source), the new path
  collapses the B-slice from 75 → 11 bytes (~85% reduction) with
  identical PSNR (52.36 dB Y, max enc/dec diff 0, bit-equivalent
  against ffmpeg/libavcodec). Implements §8.4.1.2.1 colocated-block
  selection, §8.4.1.2.2 step 4 (MinPositive on refIdxLX), step 5
  (directZeroPredictionFlag), step 7 (colZeroFlag short-circuit), and
  §8.4.1.3 median MV prediction. Round-21 caveats: spatial direct only
  (no temporal direct), 16x16 partitions only (no B_Direct_8x8 / B_8x8),
  per-8x8 uniformity gate skips Direct when the derivation's per-
  partition MVs disagree.
- encoder: B-slice support (round 20). Explicit B_L0_16x16 / B_L1_16x16
  / B_Bi_16x16 macroblock types via the new `Encoder::encode_b` entry
  point. SPS bumps to Main profile (profile_idc=77) since §A.2.2
  forbids B-slices in Baseline; `EncoderConfig::profile_idc` and
  `max_num_ref_frames` are now part of the public configuration.
  Per-MB SAD-driven L0/L1/Bi mode decision; bipred uses §8.4.2.3.1
  default weighted average `(L0+L1+1)>>1`. Two-list MV-pred grids
  (independent §8.4.1.3 mvpLX derivation per list). `MbDeblockInfo`
  extended with `mv_l1` / `ref_idx_l1` / `ref_poc_l1` so the §8.7.2.1
  bipred edge-strength derivation sees the right per-list identity.
  ffmpeg/libavcodec decodes the B-slice **bit-equivalently** (max diff
  0) at 54.19 dB PSNR_Y on a synthetic IPB fixture; our own decoder
  agrees bit-for-bit. Round-20 caveats: no B_Skip, no B_Direct, no
  intra fallback in B, single ref per list, no sub-MB partitions.
- encoder: 4MV / P_8x8 with all sub_mb_type = PL08x8 (round 19).
  Per-8x8 quarter-pel ME, content-driven SAD heuristic gate (≥30%
  SAD reduction over 1MV), §8.4.1.3 within-MB-aware mvp derivation.
  ffmpeg interop bit-equivalent on a per-sub-block-shifted fixture;
  40.95 dB PSNR_Y.
- encoder: P-slice support (round 16). Single L0 reference, integer-pel
  motion estimation, P_Skip + P_L0_16x16 macroblock types. Self-roundtrip
  + ffmpeg interop bit-exact on a 2-frame I+P fixture; PSNR vs source
  ~50 dB on integer-shift content.

## [0.1.1](https://github.com/OxideAV/oxideav-h264/compare/v0.1.0...v0.1.1) - 2026-04-25

### Other

- release v0.1.0 ([#3](https://github.com/OxideAV/oxideav-h264/pull/3))

## [0.1.0](https://github.com/OxideAV/oxideav-h264/compare/v0.0.4...v0.1.0) - 2026-04-25

### Other

- fix clippy 1.95 lints
- promote to 0.1.0
- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- bump thiserror 1 → 2
- sync Picture identity with DPB entry on MMCO-5
- honour §7.4.1.2.4 MMCO-5 prevFrameNum reset in POC types 1 & 2
- fix MMCO-5 POC reset for output ordering and DPB identity
- insert non-existing refs on frame_num gaps (§8.2.5.2)
- fix §8.3.1.2 / §8.3.2.2 / §8.3.3.1 / §8.3.4.1 constrained_intra_pred
- Add 15 more JVT vectors — 5 fully pixel-exact, inc. LS_SVA_D 1700/1700
- Add 14 JVT vectors — 13 pixel-exact on first run, 1307 more exact frames
- snapshot activated PPS/SPS on Slice event (§7.4.1.2.1)
- Add 14 JVT vectors — 12 pixel-exact on first run, 2034 new exact frames
- fix Table 8-17 tC0 values for indexA 17..=23 (spec transcription)
- Add 12 JVT CAVLC conformance vectors — 10 pixel-exact on first run
- derive mb_field_decoding_flag ctxIdxInc per §9.3.3.1.1.2
- implement I_PCM macroblock handling in CABAC mode (§7.3.5 / §9.3.1.2)
- implement spatial direct MV derivation (§8.4.1.2.2) for B_Skip / B_Direct
- Add JVT vectors: HCMP1 (hierarchical GOP + ref reorder), CAMA1 (MBAFF), CAPA1 (PAFF)
- resolve deblocking ref-picture identity per-slice not per-picture
- Add JVT CABACI3_Sony_B + CABAST3_Sony_E: multi-slice CABAC coverage
- honour slice boundaries in neighbour availability for multi-slice pictures
- Add 3 JVT conformance vectors: CABA3 (CABAC IPB), SVA_Base_B + SL1_SVA_B (multi-slice)
- fix bS for single-pred B-slice edges with swapped lists
- fix bS derivation with per-POC picture-identity compare
- fix MVpred same-MB partition-availability for single-list predictions
- proper §8.4.1.2.3 temporal-direct with MapColToList0 and B_Direct_8x8
- fix B-slice ref_idx_l0 gate + mb_type condTerm on B_Skip/B_Direct_16x16
- implement §8.4.1.2.3 temporal direct MV derivation
- fix Table 9-17 init for ctxIdx 64..=69 (intra-pred-mode contexts)
- add CTX17 debug trace for P-slice mb_type investigation
- fix ref_idx condTermFlag for in-MB 8x8 partition neighbours
- fix P-slice mb_type binIdx=0 ctxIdxInc + Table 9-17 mb_qp_delta init
- add bin-level trace instrumentation + Round 31 CABA2 analysis
- add MB-level trace instrumentation + InitCell::same helper
- add P_Skip filter to P-slice mb_type ctxIdxInc (§9.3.3.1.1.3)
- use '(b1 != 0) ? 2 : 3' for P-slice mb_type binIdx 2
- Add JVT CABA2_SVA_B: CABAC P-slice conformance with trace file
- §9.3.3.1.1.1 / .1.1.3 / .1.1.5: thread CABAC per-slice neighbour state
- Remove accidental [workspace] marker from Cargo.toml
- chroma 4:2:0 per-sub-segment bS derivation
- per-4x4 nonzero-coef mask + chroma bS inherit path
- wire inter-edge bS=1 MV-delta / ref-mismatch test
- median step-1 replacement is partition-level, not MVpred-level
- C→D substitution is partition-level, not MVpred-level
- P_Skip intra neighbour must not zero-force MVpred
- Deblock per-MB iteration order fix (§8.7.1)
- distinguish I-slice Intra_16x16 raw=5 from I_NxN neighbour
- Fix AC scan off-by-one for Intra_16x16 & Chroma AC blocks
- CABAC §9.3.3.1.1.9: fix cbf_cond_for cbp-bit branch + unify cbf lookup
- Round 17: investigate §9.3.3.1.1.9 CBF cbp-bit gate (known gap)
- CABAC coded_block_flag MB-level DC availability + engine spec audit
- CABAC CBP + coded_block_flag ctxIdxInc fixes per §9.3.3.1.1.4/9
- Add JVT CABA1_SVA_B: CABAC conformance WITH bit-level trace file
- Add JVT CABAC frame-coded conformance: camp_mot_frm0 (Mobile 720x480)
- Add JVT conformance harness: AUD_MW_E (Baseline/CAVLC, foreman QCIF)
- Round 14: P/B MVpred fixes + deblock Table 8-17 + 4:4:4 chroma parse
- Latent CABAC Cb/Cr mixup + chroma DC transform bugs (§8.5.11.2)
- Pixel-accuracy fixes + enhanced conformance diagnostics + sub_mb_type CABAC
- CABAC sub_mb_type + neighbour grid + MBAFF field stride + conformance harness
- CABAC inter syntax + MBAFF Phase 2 + multi-slice picture assembly
- Inter CAVLC fixes (100% parse on 5 streams) + CABAC suffix + MBAFF phase 1 + intra neighbour pred
- CAVLC Table 9-5 Col1 length fix + DpbOutput reorder + scaling lists wired
- CAVLC table + transform_8x8 ordering fixes + DPB wiring + recon polish
- CAVLC nC + run_before tables + weighted pred + more SEI + C→D MVpred
- Wire receive_frame + fix CABAC mb_type_i binarisation
- Add CABAC residual walker (§7.3.5.3.3) — foreman decodes end-to-end
- Fix ME_INTER_420_422 tail + surface MB index in slice_data errors
- Register h264 codec decoder in the registry
- rustfmt + clippy pass: auto-fixes + scoped too_many_arguments allows
- Expose slice_data bit cursor + wire end-to-end in integration test
- Add integration test: parse foreman_p16x16.264 (ffmpeg sample)
- Add inter MC reconstruction path (P + B slices)
- Pre-stub ref_store module for inter MC reference picture access
- Add I-slice reconstruction pipeline + POC-ordered output queue
- Pre-stub picture / mb_grid / reconstruct / dpb_output modules
- Add slice_data, macroblock_layer, FMO MB address, top-level Decoder
- point spec link at new video/h264/ path
- Pre-stub slice_data / macroblock_layer / mb_address / decoder modules
- Wire VUI + scaling_list into SPS/PPS
- Add inter prediction, MV derivation, CABAC binarisations, SEI payloads
- Pre-stub inter_pred / mv_deriv / cabac_ctx / sei modules
- aim for complete H.264, widening from the common subset
- Add intra prediction, inverse transform, deblocking, ref list + marking
- Pre-stub intra_pred / transform / deblock / ref_list modules
- Add slice_header, scaling_list, VUI, non-VCL parsers, POC derivation
- Pre-stub slice_header / scaling_list / vui / non_vcl / poc modules
- Add §7.3.2.1 SPS, §7.3.2.2 PPS, §9.2 CAVLC, §9.3 CABAC
- Pre-stub sps/pps/cavlc/cabac modules
- Add §7.3.1/§B.1 NAL unit parsing + Annex B / AVCC framing
- Add §7.2 bitstream reader (u/i/f/ue/se/te + RBSP helpers)
- Reset to empty skeleton for spec-driven rewrite

### Removed

- **Entire previous implementation.** The pre-rewrite codebase decodes
  many H.264 streams but accumulates spec-deviations on real-world
  content that bug-by-bug fixing stopped paying off; archived to the
  [`old`](https://github.com/OxideAV/oxideav-h264/tree/old) branch
  for reference.

### Added

- Empty crate skeleton with the standard CI / release-plz / LICENSE
  layout. `register()` is a no-op so the workspace aggregator keeps
  building.
- `README.md` is now a spec coverage matrix that grows with the
  rewrite.

### Notes

The rewrite is driven by ITU-T Rec. H.264 | ISO/IEC 14496-10 (2024-08
edition) as the single authoritative source. No external decoder code
is consulted while writing the implementation; conformance is verified
by behavioural diff against an external reference decoder run
separately.
