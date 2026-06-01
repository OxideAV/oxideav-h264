# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Other

- round 207 — new Annex G (MVC) SEI payload type implemented in
  `sei.rs`: payload type 41 `non_required_view_component` (§G.13.1.6 /
  §G.13.2.6 — per-target-view list of `(view_order_index[i],
  index_delta_minus1[i][])` pairs, each pair recording the
  view-order-index of the target view plus the deltas locating the
  view components that are not needed to decode it). All four §G.13.2.6
  range bounds are enforced before storage:
  `num_info_entries_minus1 ≤ 1022` (pre-allocation cap derived from
  Annex G's absolute `num_views_minus1 ≤ 1023` per §G.13.2.10);
  `view_order_index[i] ∈ 1..=1023` (incl. the spec's lower bound of 1
  since view 0 is the base view, not a target);
  `num_non_required_view_components_minus1[i] ≤ view_order_index − 1`
  (pre-allocation, anti-OOM); and `index_delta_minus1[i][j] ≤
  view_order_index − 1`. Wired into `parse_payload` dispatch; the
  payload-type table at the top of `sei.rs` grows one row. New
  `SeiError` variants `NonRequiredViewComponentNumInfoEntriesOutOfRange`,
  `NonRequiredViewComponentViewOrderIndexOutOfRange`,
  `NonRequiredViewComponentCountOutOfRange`, and
  `NonRequiredViewComponentIndexDeltaOutOfRange`. The
  `tests/integration_sei_malformed.rs` `KNOWN_PAYLOAD_TYPES` table
  moves 41 out of the `Unknown`-fallback set, lifting the
  sweep-cardinality lock from 2904 → 2948 panic-free invocations. 7
  new lib unit tests cover the single-entry / two-entry happy path,
  rejection of `view_order_index == 0`, rejection of
  `num_info_entries_minus1 == 1023` (1 above the absolute cap),
  rejection of the per-entry inner count exceeding `view_order_index
  − 1`, rejection of an index delta exceeding `view_order_index − 1`,
  and `parse_payload` dispatch round-trip. Harmless on non-MVC streams
  (Phase 4 on the README "Profiles + features in scope" table).
- round 200 — two new Annex G (MVC) SEI payload types implemented in
  `sei.rs`: payload type 39 `multiview_scene_info` (§G.13.1.4 /
  §G.13.2.4 — single `max_disparity` ue(v), 0..=1023 range bound
  enforced before storage) and payload type 43
  `operation_point_not_present` (§G.13.1.8 / §G.13.2.8 — list of
  16-bit operation-point identifiers declared absent in the bitstream,
  with the §G.13.2.8 65536-id pre-allocation cap closing the same
  shape of OOM lever that the round-177 §D.2.20 fix removed for
  motion_constrained_slice_group_set, plus the 0..=65535 per-id range
  check). Both arms are wired into `parse_payload` dispatch; the
  payload-type table at the top of `sei.rs` grows two rows. New
  `SeiError` variants `MultiviewSceneInfoMaxDisparityOutOfRange`,
  `OperationPointNotPresentCountOutOfRange`, and
  `OperationPointNotPresentIdOutOfRange`. The
  `tests/integration_sei_malformed.rs` `KNOWN_PAYLOAD_TYPES` table
  moves 39 and 43 out of the `Unknown`-fallback set (which also lifts
  the sweep-cardinality lock from 2816 → 2904 panic-free invocations).
  10 new lib unit tests (5 per payload type) cover boundary 0,
  typical value, boundary 1023 / boundary 65535, count-overflow
  rejection, id-overflow rejection, and `parse_payload` dispatch
  round-trip. Harmless on non-MVC streams (Phase 4 on the README
  "Profiles + features in scope" table).
- round 194 — `parse_residual_block_cavlc` rejects malformed §7.3.5.3.1
  call-contract triples (`startIdx > endIdx`, `maxNumCoeff ∉ {4, 8, 15, 16}`,
  span exceeding `maxNumCoeff`) before any bit is read; new `CavlcError`
  variants `InvalidBlockSpan` / `InvalidMaxNumCoeff` / `BlockSpanExceedsMax`
- round 194 — sweep `fuzz/artifacts/panic_free_decode/` (4 local oom-*
  artifacts cleared by round-177 §D.2.20 + round-187 §8.2.1 i64 staging;
  re-fed through `panic_free_decode -runs=1`, all exit 0 in 0 ms within
  libFuzzer's default 2 GiB RSS budget — the artifacts directory is
  git-ignored so the prune is a local-disk operation) + new
  `fuzz/README.md` documenting the harness layout, the round-194 sweep,
  and the local-rerun recipe

## [0.1.5](https://github.com/OxideAV/oxideav-h264/compare/v0.1.4...v0.1.5) - 2026-05-30

### Other

- round 192 — strict avcC parser + ISO/IEC 14496-15 §5.2.4.1.1 High-family extension
- round 187 — i64-stage §8.2.1 derivation + PocError::Overflow (fuzz crash 1743a9ce)
- round 183 — Annex G §G.13.2.10 multiview_view_position (payload type 46)
- round 177 — fix sei_payload OOM via §D.2.20 num_slice_groups_in_set_minus1 bound
- round 164 — fuzz target + malformed-input property sweep
- round 158 — typed ATSC1 envelope sub-parser (A/53 Part 4 §6.2.3 / §D.2.6)
- round-151 — opt-in CABAC IDR Intra_16x16 chroma AC trellis (§8.5.11.1 + §7.3.5.3 + §9.3.3.1.3)
- round-148 — opt-in CABAC IDR Intra_16x16 luma AC trellis (§8.5.10 + §9.3.3.1.3)
- scrub pre-existing JM / libavcodec decorative-attribution prose
- round 145: end-to-end Criterion encode benchmark
- round 139 — Criterion decode bench over IDR / P-chain / IPB / 4:2:2
- CAVLC I_PCM honours BitDepth{Y,C} from the active SPS (§7.4.5)
- reject PPS that arrives before its backing SPS (task #1044 / round 120 fuzz failure)
- round-120 — sei_manifest + sei_prefix_indication (payload types 200, 201)
- parse colour_remapping_info (payload type 142, §D.1.30/§D.2.30)
- regionwise_packing Annex D SEI payload (type 155)
- dec_ref_pic_marking_repetition Annex D SEI payload (type 7)
- round-107 — content_colour_volume Annex D SEI payload (type 149)
- round 103 — spare_pic Annex D SEI payload (type 8)
- round 99 — sub-sequence Annex D SEI family (types 10/11/12)
- round-95 — omni_viewport + shutter_interval_info parsers
- drop incomplete pictures whose MbGrid leaves MBs uncovered
- validate frame_cropping offsets stay within the coded picture
- reject first_mb_in_slice != 0 for new coded picture
- round-78 — four additional Annex D HDR + 360 payload parsers
- round-73 — five additional Annex D payload parsers
- CABAC inter trellis quantisation (round 49 RDOQ-lite)
- bound max_num_ref_frames at 16 (§7.4.2.1.1 / Annex A.3.1)
- reconstruct + inter_pred: reject inter MC against zero-dim reference (§8.4.2)
- bound first_mb_in_slice by PicSizeInMbs (§7.4.3)
- drop unfinished picture when every slice failed
- sps + transform: reject reserved profile_idc + wrap chroma DC math
- reject FMO PPS activation per §A.2 (fuzz oracle parity)
- transform + cavlc + cabac: keep §8.5.12 inverse-quant multiply within i32
- tighten picture-dim cap to 510 mbs / axis (was 32766)
- cabac + slice_data + slice_header + pps: bound parse-time growth & shifts
- sps + scaling_list: bound parse-time fields to prevent arithmetic overflow
- bound Exp-Golomb leading-zero count at 31 (fix shl overflow)
- add cargo-fuzz harness + libavcodec oracle for H.264 decoder

### Added

- **round 192 — ISO/IEC 14496-15 §5.2.4.1.1 strict avcC parser +
  High-profile extension trailer.** `H264CodecDecoder::consume_extradata`
  now (a) rejects `lengthSizeMinusOne == 2` (3-byte length prefix) at
  the avcC header, as required by §5.2.4.1.1 — previously the illegal
  value silently flowed into `AvccSplitter`, which rejected it later
  with a less specific error; (b) parses the §5.2.4.1.1 High-family
  extension (`chroma_format`, `bit_depth_luma_minus8`,
  `bit_depth_chroma_minus8`, `numOfSequenceParameterSetExt` + SPS-Ext
  NAL list) when `AVCProfileIndication ∈ {100, 110, 122, 144, 244}`,
  rather than leaving up to 4+ trailing bytes silently dropped; (c)
  enforces the §7.4.2.1.1 `bit_depth_*_minus8 ≤ 6` cap before the
  values reach any consumer. New diagnostic accessors
  `avcc_profile_idc()` / `avcc_level_idc()` / `avcc_chroma_format()` /
  `avcc_bit_depth_luma()` / `avcc_bit_depth_chroma()` surface what the
  avcC header advertised — useful for cross-checking against the
  in-band SPS on streams where the two disagree (the SPS still wins).
  Records that elide the High-family extension trailer are still
  accepted (some real-world muxers truncate it); the accessors stay
  `None` in that case. 11 new tests in `src/h264_decoder.rs` pin: the
  baseline + 1-byte + 2-byte prefix accept paths, the
  `lengthSizeMinusOne == 2` reject, wrong configurationVersion reject,
  short-header reject, profile_idc 100 + 244 extension parse, missing
  extension toleration, and the §7.4.2.1.1 bit-depth overflow / SPS
  truncation rejections.

### Fixed

- **round 187 — §8.2.1 POC derivation overflow now structured, not
  a panic.** Nightly `panic_free_decode` libfuzzer run #26632302094
  surfaced a debug-build `attempt to add with overflow` at
  `src/poc.rs:225` inside `derive_type0`'s eq. 8-5 frame branch
  (`TopFieldOrderCnt + delta_pic_order_cnt_bottom`, i32+i32). The
  `Decoder` trait contract is that adversarial bytes surface as
  `Err(...)` from `send_packet` — never a process-killing panic —
  so the i32 arithmetic in §8.2.1.1 eq. 8-3 / 8-4 / 8-5 was lifted
  into i64 staging, the §8.2.1.2 / §8.2.1.3 silent `as i32` tail
  truncations were replaced with `i32::try_from`, and a new
  `PocError::Overflow` variant carries the §C.4-mandated
  i32-PicOrderCnt invariant violation back to the caller. The
  decoder driver in `h264_decoder.rs` already maps `PocError` to
  `Error::invalid(...)`, so the affected slice is now skipped (and
  logged) instead of aborting the process.

  Four new unit tests in `src/poc.rs` pin the structured-error
  contract per algorithm (type 0 eq. 8-5 frame branch overflow,
  type 0 eq. 8-3 negative msb-wrap overflow, type 1 eq. 8-10
  frame-branch overflow against an i32::MAX × 10 cycle, type 2
  eq. 8-12 doubling overflow against `prev_frame_num_offset` ≈
  i32::MAX / 2). A new integration regression
  (`fuzz_crash_1743a9ce_poc_eq_8_5_frame_branch_overflow_does_not_panic`)
  embeds the 596-byte libfuzzer artefact verbatim so CI exercises
  the previously panicking path on every run; surviving the
  `send_packet` / `flush` / `receive_frame` round trip without a
  panic IS the assertion. All 22 pre-existing POC unit tests plus
  the round-187 additions keep passing.

### Added

- **round 183 — Annex G §G.13.2.10 multiview_view_position SEI
  (payload type 46).** Wires the per-CVS view-position SEI from the
  MVC family into the SEI dispatch table at payload type 46. The
  syntax (§G.13.1.10) is three fields: `num_views_minus1` ue(v),
  `view_position[i]` ue(v) for `i = 0..=num_views_minus1`, and a
  trailing `multiview_view_position_extension_flag` u(1). Each ue(v)
  is bounded at 1023 inclusive per §G.13.2.10 before allocation, so
  an adversarial input can never drive the per-view vector past 1024
  entries — the `num_views_minus1` check fires before
  `Vec::with_capacity` runs. The extension flag is preserved
  verbatim so callers auditing a non-conforming stream (§G.13.2.10
  requires the flag equal 0) can distinguish a clean stream from a
  reserved-extension marker. Two new `SeiError` variants
  (`MultiviewViewPositionNumViewsOutOfRange`,
  `MultiviewViewPositionViewPositionOutOfRange`) surface the bound
  failures with the offending value + index.

  New unit tests cover: two-view left/right ordering, single-view
  zero-position degenerate case, extension-flag-preserved-when-set,
  rejection of `num_views_minus1 = 1024` before allocation,
  rejection of `view_position[i] = 1024`, and the `parse_payload`
  dispatcher routing payload type 46 to the new variant. The
  malformed-input integration sweep (`integration_sei_malformed.rs`)
  pulls payload type 46 into `KNOWN_PAYLOAD_TYPES` so the existing
  panic-free contract automatically extends to the new parser
  across all 11 adversarial shapes × 4 contexts.

  SEI dispatch arm count goes 40 → 41. This is the first §G.13.2
  (Annex G — MVC) SEI parser to land; the rest of Annex G/H/I
  (24–44, 48–54, 181) remains deferred to Phase 4 alongside the
  matching `nal_unit_type` 14/15/20/21 prefix/subset-SPS handling.

### Fixed

- **round 177 — `sei_payload` fuzz target OOM (artifact
  `oom-1bf45ba3a116aa21631ccf3b3cec29fed395ef0e`, 11 bytes, reported
  by oxideav-h264 fuzz CI run 26569743256).** §D.2.20
  motion_constrained_slice_group_set's
  `parse_motion_constrained_slice_group_set` read
  `num_slice_groups_in_set_minus1` as a raw `ue(v)` and pushed
  `count = num_slice_groups_in_set_minus1 + 1` entries into the
  slice_group_id vector before checking the §D.2.20 range bound.
  When the active PPS only declares a single slice group, the
  encoded width of each slice_group_id field collapses to 0 bits
  per §D.1.20 (`Ceil(Log2(num_slice_groups))`), so each iteration
  consumed zero payload bits — letting an adversarial `ue(v)` value
  near 2^28 drive a ~1 GiB allocation and trip libfuzzer's
  rss_limit. The fix reads the same `ue(v)` and immediately
  enforces the §D.2.20 normative bound
  `num_slice_groups_in_set_minus1 ≤ num_slice_groups_minus1` (where
  `num_slice_groups_minus1` arrives via `SeiContext::num_slice_groups`),
  returning the new
  `SeiError::MotionConstrainedSliceGroupCountTooLarge` before the
  loop allocates. Each `slice_group_id[i]` then gets the §7.4.2.2 /
  §D.2.20 per-id range check (`< num_slice_groups`), surfaced as
  `SeiError::MotionConstrainedSliceGroupIdOutOfRange`.

  New `tests/integration_sei_malformed.rs` test
  `motion_constrained_slice_group_set_unbounded_count_is_rejected`
  replays the exact 11-byte fuzz artifact's path-2 payload tail
  (first byte 0x90 used for the §D.2.20 `SeiContext` derivation;
  remaining 10 bytes as the raw payload) and asserts the parser
  returns a deterministic `Err` rather than allocating its way to
  OOM. The two existing motion_constrained_slice_group_set tests
  (`single_group_collapses_width` /
  `multi_group_reads_u_v`) continue to pass — the bound is
  consistent with their `num_slice_groups_in_set_minus1` values
  (0 < 1 and 1 < 4 respectively). All 1033 lib tests + 5
  `integration_sei_malformed` tests pass.

### Added

- **round 164 — SEI parser hardening (new cargo-fuzz target +
  malformed-input property sweep).** Two complementary additions
  exercise the Annex D SEI surface that the existing whole-decoder
  fuzz targets reach only when libFuzzer happens to synthesise a
  well-formed Annex-B NAL→RBSP→sei_payload framing — i.e. almost
  never. Together they bring the 40-payload SEI dispatcher under the
  same panic-free contract the rest of the public Decoder surface
  has carried since round 120.

  `fuzz/fuzz_targets/sei_payload.rs` — new cargo-fuzz binary that
  feeds raw libFuzzer bytes through three SEI-parsing paths: (1) the
  full envelope (`parse_sei_rbsp`) and then `parse_payload` on every
  extracted message; (2) `parse_payload` directly with a payload type
  chosen from a 63-entry fixed pool (every dispatcher arm + several
  `Unknown`-fallback sentinels covering Annex F/G/H/I numeric
  ranges); (3) `parse_payload` on the full input under a second
  payload-type selector. A `context_from_byte` helper derives a
  varied-but-bounded `SeiContext` from the input so the VUI-gated
  branches (`buffering_period`, `pic_timing`,
  `dec_ref_pic_marking_repetition`, `spare_pic`, the §D.1.20
  `num_slice_groups`-dependent u(v) width) are reached without
  requiring libFuzzer to discover them. The new binary sits adjacent
  to the existing `panic_free_decode` / `ffmpeg_oracle_decode`
  binaries.

  `tests/integration_sei_malformed.rs` — deterministic property
  sweep covering every (payload_type, shape, context) triple in the
  matrix the fuzz target explores, so CI catches a regression even
  without libFuzzer. The grid is 40 known + 23 fallback payload
  types × 11 adversarial shapes (empty / one_zero / eight_zeros /
  sixtyfour_zeros / one_ff / eight_ffs / sixtyfour_ffs /
  max_ueg_prefix / alt_55_aa / alt_aa_55 / long_zeros_then_stop) ×
  4 `SeiContext` cells (`default` / `nal_hrd_present_24bit` /
  `vcl_hrd_present_4sched` / `max_lengths_8sg`) = 2772 invocations
  of `parse_payload`, asserted to never panic. Three companion
  tests cover the envelope-only path
  (`sei_parse_rbsp_never_panics`), the envelope-then-dispatch
  composition (`sei_envelope_then_dispatch_never_panics`), and the
  `SeiPayload::Unknown` catch-all contract
  (`sei_unknown_payload_type_is_ok`). The cardinality is explicitly
  pinned (2772) so a future change that accidentally drops a
  payload-type row or shape entry fails loudly instead of silently
  shrinking coverage.

  No behaviour change — `parse_payload` and `parse_sei_rbsp` are
  unchanged. This round is hardening only: it locks in the
  panic-free contract on the SEI surface and gives a defined surface
  for future libFuzzer runs to target without re-deriving the
  framing.

- **round 158 — typed ATSC1 envelope sub-parser on top of
  `user_data_registered_itu_t_t35` (§D.2.6).** Adds
  `parse_atsc1_envelope(payload_bytes)` consuming the post-country-code
  bytes of a country=0xB5 (USA) T.35 payload and decoding the ATSC1
  envelope per ATSC A/53 Part 4 (2009) §6.2.3:
  * `provider_code` (u(16) BE) — validated to be `0x0031` (the ATSC slot
    in the T.35 administered provider registration).
  * `user_identifier` (u(32) BE) — validated to be one of the two
    registered ATSC values per Table 6.7: `'GA94'` (0x47413934 →
    `ATSC_user_data()`) or `'DTG1'` (0x44544731 → `afd_data()`).
  * For `'GA94'`: reads `user_data_type_code` (u(8)) per Table 6.8 and
    dispatches per Table 6.9 — `0x03` surfaces the inner `MPEG_cc_data()`
    bytes opaquely (the inner cc_data() byte layout is normatively in
    CEA-708 [1] Table 2 which is **not present in this docs tree** — see
    docs-gap note below), `0x06` parses `bar_data()` per Table 6.11
    (top/bottom letterbox + left/right pillarbox pairs, reserved nibble
    + per-value '11' one_bits validation + §6.2.3.2 mutual-exclusion
    enforcement), any other code surfaces as `Atsc1UserData::Reserved`
    with raw bytes preserved per §6.2.2 silent-discard rule.
  * For `'DTG1'`: parses `afd_data()` per Table 6.13 / §6.2.4.2 (1 byte
    when `active_format_flag = 0`, 2 bytes when `1` carrying the 4-bit
    `active_format` value from Table 6.14).

  New public surface in `crate::sei`: `Atsc1Envelope`, `Atsc1Body`,
  `Atsc1UserData`, `Atsc1UserIdentifier` (with `GA94` / `DTG1`
  constants), `BarData`, `AfdData`, and `parse_atsc1_envelope`. Ten new
  `SeiError` variants pin the validation failures
  (`Atsc1EnvelopeTooShort`, `Atsc1ProviderCodeMismatch`,
  `Atsc1UnknownUserIdentifier`, `Atsc1AtscUserDataEmpty`,
  `Atsc1BarDataReservedMismatch`, `Atsc1BarDataOneBitsMismatch`,
  `Atsc1BarDataTopBottomMismatch`, `Atsc1BarDataLeftRightMismatch`,
  `Atsc1BarDataLetterboxPillarboxBoth`, `Atsc1BarDataTruncated`).

  Sixteen new `sei::tests::atsc1_*` unit tests cover: GA94+cc_data
  round-trip with the raw cc_data bytes passed through verbatim;
  GA94+bar_data letterbox + pillarbox decode at line/pixel values from
  the worked example in §6.2.3.2; rejections for top≠bottom, left≠right,
  reserved≠'1111', one_bits≠'11', and simultaneous letterbox+pillarbox;
  GA94+reserved type-code preserving raw bytes for diagnostics; GA94
  empty body rejected; DTG1+AFD flag-off (1-byte) and flag-on (2-byte
  with `active_format = '1010'` 16:9 letterbox); 6-byte minimum size
  enforcement; wrong provider_code (0x003C HDR10+) rejected; unknown
  user_identifier rejected; and an end-to-end chain through
  `parse_payload(4, …)` → `UserDataRegisteredItuTT35` →
  `parse_atsc1_envelope`. 1033 lib tests + 1243 total tests pass with
  the new module wired in (was 1017 + 1227).

  Spec basis: ATSC A/53 Part 4 (2009) §6.2.3 Tables 6.7 / 6.8 / 6.9 /
  6.10 / 6.11 / 6.13 / 6.14 (`docs/broadcast/atsc/A53-Part-4-2009.pdf`);
  Rec. ITU-T H.264 (08/2024) §D.1.6 / §D.2.6 for the outer
  `user_data_registered_itu_t_t35` carriage; ITU-T T.35 §3.1 for the
  country_code / extension semantics.

  **Docs-gap** — CEA-708 (the spec carrying the cc_data() Table 2 byte
  layout: `process_cc_data_flag` + `cc_count` + per-triple `cc_valid` /
  `cc_type` / `cc_data_1` / `cc_data_2` + DTVCC packet framing) is not
  present in `docs/`. Until it lands, ATSC1 `MPEG_cc_data` inner bytes
  stay opaque (the round-158 parser hands them through unchanged for a
  downstream CEA-708-aware consumer). Adding CEA-708-D to
  `docs/broadcast/cea/` would unblock a typed `CcData` decoder returning
  the per-triple (cc_type, cc_data_1, cc_data_2) array.

- **round 151 — opt-in CABAC IDR Intra_16x16 chroma AC trellis
  quantisation.** Extends the round-148 luma-AC trellis to the chroma
  Intra_16x16 paths. For 4:2:0 the lever refines the 4 per-block 4×4
  AC arrays per chroma plane (Cb / Cr) inside
  `encode_chroma_intra16x16_420`; for 4:4:4 the lever refines the 16
  per-block 4×4 AC arrays per chroma plane inside
  `encode_chroma_intra16x16_444` (§7.3.5.3 "chroma coded like luma").
  Each block is passed through
  `crate::encoder::transform::trellis_refine_4x4_ac` with
  `skip_dc=true` so the §8.5.11.1 2×2 chroma-DC Hadamard chain (4:2:0)
  or the §8.5.10 luma-style 4×4 DC Hadamard chain on the 4:4:4 chroma
  plane stays bit-exact. New `EncoderConfig::trellis_quant_intra_chroma`
  toggle (default `false`) gates the refinement. Bitstream syntax is
  unchanged; the decoder is unaffected; `cbp_chroma` keeps the §7.4.5.1
  Table 7-15 value (`0`/`1`/`2` based purely on whether DC and / or AC
  remains non-zero across the MB). The default stays off for the same
  reason as `trellis_quant_intra`: the round-49 rate model is
  calibrated for inter blockCat 2 (Luma4x4) and is roughly net-zero on
  dense intra chroma AC at typical QP — re-calibrating for chroma
  blockCats 4 (ChromaACLevel @ 4:2:0) and 1 (Intra16x16AC @ 4:4:4,
  treated like luma) so the lever can ship default-on is the obvious
  follow-up. New `integration_trellis_quant_intra_chroma.rs` pins six
  guarantees: 4:2:0 and 4:4:4 default-off byte-for-byte determinism;
  4:2:0 decoder self-roundtrip bit-exact on all three planes at QP=22;
  4:4:4 decoder self-roundtrip bit-exact on all three planes at QP=22;
  QP {18..42} sweep stays decodable + plane-wise bit-exact; luma PSNR
  invariant under the chroma lever; and a measurement-only byte-savings
  print across QP {18..38}. Spec basis: §8.5.10, §8.5.11.1, §7.3.5.3,
  §7.4.5.1, §9.3.3.1.3.

- **round 148 — opt-in CABAC IDR Intra_16x16 luma AC trellis
  quantisation.** Extends the round-49 `D + λ·R` refinement
  (`crate::encoder::transform::trellis_refine_4x4_ac`) to the 16 per-
  block 4x4 AC arrays inside an Intra_16x16 luma MB on the
  `encode_idr_cabac` path. Each block is passed through the trellis
  with `skip_dc=true` so the §8.5.10 inverse-Hadamard DC chain stays
  bit-exact while higher-frequency AC levels can shrink toward zero.
  New `EncoderConfig::trellis_quant_intra` toggle (default `false`)
  gates the refinement. Bitstream syntax is unchanged; the decoder
  is unaffected; `cbp_luma` keeps the §7.4.5.1 Table 7-11 value
  (`15` whenever any AC remains non-zero across the MB). The default
  stays off because the round-49 rate model is calibrated for inter
  blockCat 2 (Luma4x4) and is net-negative on dense intra
  Intra_16x16 AC at low QP — re-calibrating for blockCat 1
  (Intra16x16AC) so the lever can ship default-on is the obvious
  follow-up. New `integration_trellis_quant_intra.rs` pins four
  guarantees: default-off byte-for-byte determinism, decoder self-
  roundtrip bit-exact at QP=22, PSNR drift inside a 3 dB envelope
  across QP {22, 28, 34}, and QP {18..42} sweep stays decodable +
  bit-exact (max enc/dec luma diff = 0). Spec basis: §8.5.10, §8.5.12,
  §7.4.5.1, §9.3.3.1.3.

- **round 145 — end-to-end Criterion `encode` benchmark.** Adds
  `benches/encode.rs`, sibling to `benches/decode.rs`, exercising
  the public encoder entry points across seven in-process variants
  with no on-disk fixtures: 64x64 + 128x96 Baseline CAVLC IDR;
  Baseline IDR + 4×P CAVLC chain (real P-slice ME); Main IDR + P +
  B (§8.4.1.2.2 spatial-direct + bipred); 64x64 CABAC IDR
  (`encode_idr_cabac`); 64x64 CABAC IDR + P
  (`encode_idr_cabac` + `encode_p_cabac`, exercises the round-49
  trellis-quant refinement that's on by default); 64x64 High 4:2:2
  IDR (§8.5.11.2 4×2 chroma DC Hadamard). Criterion reports both
  per-frame (`Throughput::Elements`, "Kelem/s" ≈ frames-encoded
  per second) and per-source-byte (`Throughput::Bytes`,
  MiB/s of raw planar input absorbed) views per variant. Measured
  headline numbers on an Apple M-series host (Criterion `--quick`,
  release profile): CABAC IDR ~18.3 Kelem/s (107 MiB/s of source);
  CAVLC Baseline IDR 64x64 ~2.7 Kelem/s (17 MiB/s); CAVLC IDR + 4×P
  chain 207 elem/s (1.22 MiB/s — ME-dominated); Main IPB GOP
  176 elem/s; CABAC IDR + P 685 elem/s. Complements the existing
  `inter_pred_bench`, `transform_bench`, and `decode` benches —
  the encode hot path was previously unmeasured.

- **round 139 — end-to-end Criterion `decode` benchmark.** Adds
  `benches/decode.rs` driving the public `H264CodecDecoder`
  (`send_packet` + `flush` + `receive_frame`) across five
  in-process corpora generated by `oxideav-h264`'s own encoder so
  the bench is self-contained and needs no on-disk fixtures:
  64x64 Baseline IDR, 128x96 Baseline IDR, IDR + 4×P Baseline
  chain, IDR + P + B Main GOP (spatial-direct B), and a 64x64
  High 4:2:2 IDR. Criterion reports both per-frame
  (`Throughput::Elements`) and per-byte (`Throughput::Bytes`)
  numbers per variant. Headline decode rates on an Apple M-series
  host range from ~6 Kelem/s (128x96 Baseline IDR) to ~21 Kelem/s
  (IDR + 4×P chain, amortised over 5 frames per access unit).
  Complements the existing `inter_pred_bench` (motion-compensation
  kernels) and `transform_bench` (inverse-transform kernels),
  which only measured isolated hot paths.

### Fixed

- **round 128 — CAVLC I_PCM macroblock now honours `BitDepthY` /
  `BitDepthC` from the active SPS (§7.4.5 / §7.4.2.1.1).** Before this
  round, the CAVLC `I_PCM` path in `macroblock_layer.rs` hard-coded
  `bit_depth_y = bit_depth_c = 8`, so `pcm_sample_luma[i]` and
  `pcm_sample_chroma[i]` were always read as 8-bit values regardless
  of `bit_depth_luma_minus8` / `bit_depth_chroma_minus8` on the SPS.
  For a High10 (BitDepthY = 10) sequence with an `I_PCM` MB this
  would truncate every sample to its low 8 bits and leave the bit
  reader off by `(256 + 2 * MbWidthC * MbHeightC) * (BitDepth - 8)`
  bits — corrupting all subsequent macroblocks in the slice.

  Fix threads `bit_depth_luma_minus8` / `bit_depth_chroma_minus8`
  through `EntropyState` (new fields, defaulting to 0 for parser
  unit-test call sites) and reads them in the CAVLC PCM loop. The
  CABAC PCM path in `slice_data.rs` already consulted the SPS
  directly, so no behavioural change there beyond forwarding the
  same fields into its `EntropyState`. New regression
  `slice_data::tests::cavlc_i_pcm_macroblock_10bit_high10_round_trips`
  emits a single 10-bit `I_PCM` MB with samples that explicitly
  require bits 8..=9 and asserts byte-for-byte round-trip through
  `parse_slice_data`.

- **fuzz: round-125 / task #1044 — reject PPS that arrives BEFORE its
  referenced SPS (§7.4.1.2.1 / §7.4.2.2).** Closes the
  `ffmpeg_oracle_decode` divergence flagged on
  `crash-065436ed80e17520598536327afceed7133b3122` (359 B) and the
  resulting scheduled-Fuzz-job redness on master since round 120. The
  crash input delivered a PPS NAL ahead of its backing SPS; our
  `process_nal` PPS branch used to fall back to a guessed
  `chroma_format_idc = 1` in that case, store the PPS anyway, and let
  the IDR slices that arrived later activate it and emit two 16x16
  frames. libavcodec rejects the PPS at NAL arrival time
  ("sps_id 0 out of range", referring to the unstored backing SPS) and
  produces zero frames.

  Per §7.4.2.2 the PPS scaling-matrix tail loop length depends on
  the referenced SPS's `chroma_format_idc` (4:4:4 carries 6 8x8 lists;
  everything else carries 2 when `transform_8x8_mode_flag` is set),
  so the PPS cannot be parsed unambiguously when the SPS hasn't been
  seen yet — the previous "parse with default, replace if SPS turns
  up later" code path was correct only for 4:2:0 streams without a
  PPS scaling matrix.

  Fix sits in `Decoder::process_nal` (PPS branch): peek the
  `seq_parameter_set_id`, look up the SPS, and return
  `DecoderError::UnknownSps(sps_id)` when it isn't stored. The
  permissive 4:2:0 fallback is removed; only the SPS-driven
  `chroma_format_idc` reaches `Pps::parse_with_chroma_format`.

  Regression: `tests/integration_fuzz_oracle_regression.rs::
  fuzz_crash_065436ed_pps_before_sps_rejected` pins the exact
  359-byte payload and asserts the decoder reports `Error::Eof`
  with zero frames. The pre-existing
  `slice_referencing_pps_with_unknown_sps_is_rejected` test in
  `src/decoder.rs` is renamed to
  `pps_referencing_unknown_sps_is_rejected_at_storage` and now
  asserts the rejection happens at PPS arrival (not at slice
  activation), with neither PPS storage nor active-pps state
  committed.

### Added

- **sei: round-120 — sei_manifest + sei_prefix_indication Annex D SEI
  payloads (types 200 and 201).** Adds `sei_manifest` (§D.1.36 / §D.2.36)
  and `sei_prefix_indication` (§D.1.37 / §D.2.37) of
  `docs/video/h264/T-REC-H.264-202408-I.pdf`, bringing the typed-SEI
  surface from 38 to 40 recognised payloads. These are transport- and
  systems-layer hint messages — `sei_manifest` advertises, for the
  entire coded video sequence, which SEI payload types are expected to
  be present along with a per-type "necessary / unnecessary /
  undetermined / not-expected" classification from Table D-12, and
  `sei_prefix_indication` carries one or more leading bit-strings from
  the start of an upcoming SEI payload of a given
  `prefix_sei_payload_type` so a transport element can inspect the
  first few syntax elements (e.g. `frame_packing_arrangement_type` for
  type 45) without parsing the full SEI message.

  New `parse_sei_manifest` reads `manifest_num_sei_msg_types u(16)`
  then per entry `manifest_sei_payload_type[ i ] u(16)` and
  `manifest_sei_description[ i ] u(8)`. Descriptions 0..=3 are
  surfaced as `SeiManifestDescription::{NotExpected,
  ExpectedNecessary, ExpectedUnnecessary, ExpectedUndetermined}`; per
  §D.2.36 values 4..=255 are reserved and must be passed through (so
  callers can honour "Decoders shall allow ... shall ignore" for both
  the manifest entry and any SEI prefix indication keyed off the same
  payload type) — these land as `Reserved(raw)`. The §D.2.36
  uniqueness constraint (`manifest_sei_payload_type[ m ] !=
  manifest_sei_payload_type[ n ] for m != n`) is enforced at parse
  time via `SeiManifestDuplicatePayloadType { payload_type, first,
  second }`.

  New `parse_sei_prefix_indication` reads
  `prefix_sei_payload_type u(16)`, `num_sei_prefix_indications_minus1
  u(8)`, then for each indication `num_bits_in_prefix_indication_minus1
  u(16)` (the data field's bit count is value + 1), packs that many
  `sei_prefix_data_bit u(1)` values MSB-first into a `Vec<u8>` (length
  `bit_count.div_ceil(8)`), and consumes the §D.1.37
  byte-alignment trailer (`while !byte_aligned() { f(1) }`). A
  `SeiPrefixIndicationOverflow { i, bits, available }` rejects an
  indication whose declared `bit_count` exceeds the remaining payload
  bits — the existing fuzzed bitstream EOF check already guards the
  individual `u(1)` reads but the upfront check shortens the failure
  path on hostile inputs.

  Wired into `parse_payload(200, …)` → `SeiPayload::SeiManifest` and
  `parse_payload(201, …)` → `SeiPayload::SeiPrefixIndication`.
  Thirteen new unit tests in `src/sei.rs`: SEI manifest empty / single
  entry / three-entry mixed / reserved-description / duplicate
  rejection / truncated input / `parse_payload` dispatch; SEI prefix
  indication single byte-aligned / short bit-string with `1`-fill
  byte-alignment trailer / two entries with mixed bit-widths /
  oversized-bit-count rejection / truncated-header rejection /
  `parse_payload` dispatch. Lib-test count 1003 → 1016 (+13).

- **sei: round-113 — regionwise_packing Annex D SEI payload (type 155).**
  Adds `regionwise_packing` (§D.1.35.4 / §D.2.35.4 of
  `docs/video/h264/T-REC-H.264-202408-I.pdf`), bringing the typed-SEI
  surface from 36 to 37 recognised payloads and completing the
  omnidirectional projection family alongside equirectangular (150),
  cubemap (151), sphere_rotation (154), and omni_viewport (156). The
  message describes the region-wise packing transformation that maps
  regions of the cropped decoded ("packed") picture onto a projected
  picture, plus optional per-region guard bands. New
  `parse_regionwise_packing` reads `rwp_cancel_flag u(1)`; when not
  cancelled it reads `rwp_persistence_flag u(1)`,
  `constituent_picture_matching_flag u(1)`, `rwp_reserved_zero_5bits
  u(5)` (read and ignored), `num_packed_regions u(8)`,
  `proj_picture_width`/`proj_picture_height u(32)`, and
  `packed_picture_width`/`packed_picture_height u(16)`, then one
  `PackedRegion` per region: `rwp_reserved_zero_4bits u(4)` (ignored),
  `transform_type u(3)` (Table D-11, 0..=7), `guard_band_flag u(1)`,
  the four `proj_region_*` `u(32)` and four `packed_region_*` `u(16)`
  geometry fields, and — when `guard_band_flag` is set — a `GuardBand`
  carrying the four `*_gb_*` `u(8)` widths/heights,
  `gb_not_used_for_pred_flag u(1)`, the four `gb_type[j] u(3)` edge
  types, and `rwp_gb_reserved_zero_3bits u(3)` (ignored). Parse-time
  §D.2.35.4 conformance checks reject `num_packed_regions == 0`
  (`RegionWisePackingNoRegions`), a zero `proj_picture_*`
  (`RegionWisePackingProjPictureZero`) or `packed_picture_*`
  (`RegionWisePackingPackedPictureZero`) dimension, and a guard band
  with all four dimensions zero (`RegionWisePackingGuardBandAllZero`).
  Wired into `parse_payload(155, …)` →
  `SeiPayload::RegionWisePacking`. Nine unit tests cover the cancel
  path, a single region without a guard band, a region with a
  four-edge guard band, a two-region mixed-guard-band message, the
  four §D.2.35.4 error paths, and parse_payload dispatch.

- **sei: round-110 — dec_ref_pic_marking_repetition Annex D SEI payload
  (type 7).** Adds `dec_ref_pic_marking_repetition` (§D.1.9 / §D.2.9 of
  `docs/video/h264/T-REC-H.264-202408-I.pdf`), bringing the typed-SEI
  surface from 35 to 36 recognised payloads. The message repeats the
  `dec_ref_pic_marking()` syntax structure (§7.3.3.3) that originally
  appeared in an earlier picture's slice header(s), so a decoder that
  lost that picture can still apply its reference-marking operations.
  New `parse_dec_ref_pic_marking_repetition` reads `original_idr_flag
  u(1)`, `original_frame_num ue(v)`, and — only when the active SPS has
  `frame_mbs_only_flag == 0` — `original_field_pic_flag u(1)` plus
  (when set) `original_bottom_field_flag u(1)`, then the embedded
  `dec_ref_pic_marking()`. Per §D.2.9 the `IdrPicFlag` selecting the
  embedded structure's IDR vs. non-IDR branch is `original_idr_flag`,
  not the SEI NAL's own flag: the IDR branch yields
  `no_output_of_prior_pics_flag` + `long_term_reference_flag`, the
  non-IDR branch the `adaptive_ref_pic_marking_mode_flag` sliding-window
  / MMCO loop. The embedded reader reuses the shared
  `slice_header::{DecRefPicMarking, MmcoOp}` data model and validates
  `memory_management_control_operation` to 0..=6 per Table 7-9
  (`SeiError::DecRefPicMarkingRepetitionMmcoOutOfRange`). The active
  SPS `frame_mbs_only_flag` is newly carried on
  `SeiContext::frame_mbs_only_flag` (default `true`). Wired into
  `parse_payload(7, …)` → `SeiPayload::DecRefPicMarkingRepetition`.
  Eight unit tests cover the IDR frame-only path, a two-op adaptive
  MMCO loop, sliding-window, field coding with and without a bottom-
  field flag, an all-six-MMCO-ops sweep, the out-of-range MMCO error,
  and parse_payload dispatch.

- **sei: round-107 — content_colour_volume Annex D SEI payload (type 149).**
  Adds `content_colour_volume` (§D.1.33 / §D.2.33 of
  `docs/video/h264/T-REC-H.264-202408-I.pdf`), bringing the typed-SEI
  surface from 34 to 35 recognised payloads. The message records the
  colour volume (display gamut + luminance range) actually present in
  the content, complementing the mastering-display volume (type 137)
  and MaxCLL/MaxFALL (type 144). New `parse_content_colour_volume`
  reads `ccv_cancel_flag u(1)`; when not cancelled it reads
  `ccv_persistence_flag u(1)` + the four present flags
  (`ccv_primaries_present_flag`, `ccv_min_/max_/avg_luminance_value_
  present_flag`, each `u(1)`) + `ccv_reserved_zero_2bits u(2)` (read
  and ignored per §D.2.33), then the gated sub-blocks: three
  `(ccv_primaries_x[c], ccv_primaries_y[c])` `i(32)` pairs (green /
  blue / red ordering per §D.2.33), and `ccv_min_/max_/avg_luminance_
  value u(32)`. Parse-time §D.2.33 conformance checks reject the
  all-present-flags-zero case (`ContentColourVolumeNoComponents`),
  primary coordinates outside [-5 000 000, 5 000 000]
  (`ContentColourVolumePrimaryOutOfRange`), and any luminance-ordering
  violation among the present values — `min <= avg`, `avg <= max`,
  `min <= max` (`ContentColourVolumeLuminanceOrder`). Wired into
  `parse_payload(149, …)` → `SeiPayload::ContentColourVolume`. Ten
  unit tests cover the cancel path, an all-fields BT.2020-style
  message, a single-luminance-only body, negative primary coordinates
  round-tripping through `i(32)`, and the four §D.2.33 error paths.

- **sei: round-103 — spare_pic Annex D SEI payload (type 8).**
  Adds `spare_pic` (§D.1.10 / §D.2.10 of
  `docs/video/h264/T-REC-H.264-202408-I.pdf`), bringing the typed-SEI
  surface from 33 to 34 recognised payloads. The message flags
  slice-group map units of one or more decoded reference pictures (the
  spare pictures) as resembling the co-located map units of a target
  picture, so an error-concealing decoder can substitute spare-picture
  samples. New `parse_spare_pic` reads `target_frame_num ue(v)`,
  `spare_field_flag u(1)` (+ optional `target_bottom_field_flag`),
  `num_spare_pics_minus1 ue(v)` (validated to 0..=15 per §D.2.10), then
  one `SparePicEntry` per spare picture: `delta_spare_frame_num ue(v)`,
  optional `spare_bottom_field_flag` (field messages only), and
  `spare_area_idc ue(v)` (validated to 0..=2) selecting the spare-area
  coding — `0` = whole picture spare (no per-unit data), `1` = one
  `spare_unit_flag u(1)` per map unit in raster order (stored as
  `true` = spare per §D.2.10), `2` = `zero_run_length ue(v)` runs in
  counter-clockwise box-out order read until the running map-unit count
  reaches `PicSizeInMapUnits`. The `1`/`2` branches need
  `PicSizeInMapUnits`, newly carried on `SeiContext::pic_size_in_map_units`
  (default 0; a message reaching a per-unit branch with 0 is rejected
  with `SeiError::SparePicMapUnitsUnknown` rather than guessed). Wired
  into `parse_payload(8, …)` → `SeiPayload::SparePic`. Nine unit tests
  cover all three `spare_area_idc` methods, a field message with bottom
  -field flags, a two-entry mixed-method message, and the three range/
  context error paths.

- **sei: round-99 — sub-sequence Annex D SEI payload family (2003 draft).**
  Three new payload types land, bringing the typed-SEI surface from
  30 to 33 recognised payloads. All three are specified in the 2003
  draft (`docs/video/h264/ISO_IEC_14496-10-AVC-2003-draft.pdf`,
  §D.1.11–§D.1.13 / §D.2.11–§D.2.13) and describe the sub-sequence
  data-dependency hierarchy (layer 0 independently decodable, higher
  layers predicted only from lower ones):
  - **10** `sub_seq_info` (§D.1.11 / §D.2.11) — places the current
    picture in the hierarchy: `sub_seq_layer_num ue(v)` (< 256) +
    `sub_seq_id ue(v)` (< 65536) + `first_ref_pic_flag` +
    `leading_non_ref_pic_flag` + `last_pic_flag` +
    `sub_seq_frame_num_flag`, the optional `sub_seq_frame_num ue(v)`
    counting pictures within the sub-sequence modulo `MaxFrameNum`.
  - **11** `sub_seq_layer_characteristics` (§D.1.12 / §D.2.12) —
    `num_sub_seq_layers_minus1 ue(v)` (≤ 255) followed by one
    `(accurate_statistics_flag u(1), average_bit_rate u(16),
    average_frame_rate u(16))` triple per layer; the layer-`N` entry
    jointly characterises layers `0..=N`.
  - **12** `sub_seq_characteristics` (§D.1.13 / §D.2.13) — target
    `sub_seq_layer_num` / `sub_seq_id`, optional `sub_seq_duration
    u(32)` (90-kHz ticks) gated on `duration_flag`, an optional
    average-rate triple gated on `average_rate_flag`, and a list of
    `num_referenced_subseqs ue(v)` (< 256) inter-prediction-reference
    sub-sequences `(ref_sub_seq_layer_num, ref_sub_seq_id,
    ref_sub_seq_direction)`.
  Parse-time §D.2.11–§D.2.13 range checks reject out-of-range
  `sub_seq_layer_num` / `sub_seq_id` / `num_sub_seq_layers_minus1` /
  `num_referenced_subseqs`. Nine new unit tests cover both the
  with/without optional-field branches and the dispatch wiring for
  payload types 10/11/12.

- **sei: round-95 — two additional Annex D SEI payload parsers.**
  Two new payload types land, bringing the typed-SEI surface from
  28 to 30 recognised payloads:
  - **156** `omni_viewport` (§D.1.35.5 / §D.2.35.5) — completes
    the 360-video SEI family started in round 78 (alongside
    `equirectangular_projection` (150), `cubemap_projection`
    (151), and `sphere_rotation` (154)). `omni_viewport_id u(10)`
    + `omni_viewport_cancel_flag u(1)` + (when not cancelled)
    `omni_viewport_persistence_flag u(1)` + `omni_viewport_cnt_minus1
    u(4)` + the per-viewport quintuple `(azimuth_centre i(32),
    elevation_centre i(32), tilt_centre i(32), hor_range u(32),
    ver_range u(32))`. Parse-time §D.2.35.5 range checks enforce:
    `azimuth_centre`, `tilt_centre` ∈ [−180·2^16, 180·2^16 − 1];
    `elevation_centre` ∈ [−90·2^16, 90·2^16]; `hor_range` ∈
    [1, 360·2^16]; `ver_range` ∈ [1, 180·2^16].
  - **205** `shutter_interval_info` (§D.1.38 / §D.2.38) — indicates
    the camera shutter interval (image-sensor exposure time) for
    the associated source pictures, either as a single CVS-wide
    constant or as a per-sub-layer schedule. `sii_sub_layer_idx
    ue(v)` gates the body; when the index is 0 the body adds
    `shutter_interval_info_present_flag u(1)` and (when the
    present-flag is 1) `sii_time_scale u(32)` +
    `fixed_shutter_interval_within_cvs_flag u(1)` + either
    `sii_num_units_in_shutter_interval u(32)` (fixed-mode) or
    `sii_max_sub_layers_minus1 u(3)` followed by
    `sub_layer_num_units_in_shutter_interval[i] u(32)`
    (per-sub-layer). Parse-time §D.2.38 range check enforces
    `sii_time_scale > 0`. The shutter interval, in seconds, is
    derivable as `num_units / time_scale` (e.g. the §D.2.38
    worked example: 27 MHz time-scale + 1_080_000 units →
    0.04 s).

  Each new parser ships unit tests (16 new: 9 for omni_viewport
  covering cancel / single director's-cut / two-entry / max-range
  boundary plus four range-rejection paths and a parse_payload
  dispatch; 7 for shutter_interval_info covering present-false /
  non-zero sub-layer-idx / fixed 27 MHz worked example /
  per-sub-layer schedule / time_scale=0 reject and a parse_payload
  dispatch). Six new `SeiError` variants police the §D.2.35.5
  / §D.2.38 range checks. The SEI table header comment in
  `src/sei.rs` is updated; coverage bumps from 28 → 30 recognised
  SEI types.

### Fixed

- **h264_decoder: drop in-progress pictures whose MbGrid is
  incomplete at finalize time (round 91 fuzz-oracle triage,
  third hardening).** Per §7.4.2.1 / Annex A, a coded picture's
  slices shall collectively cover macroblock addresses
  `0..PicSizeInMbs`. Previously we emitted any picture for which
  at least one slice succeeded (`any_slice_succeeded` gate) even
  when the slice walk stopped well short of the picture's last
  MB — the un-walked tail came out zero. A 420 B fuzz crash
  (`crash-b20f4127…`) hit this on a 48x2048 (PicSizeInMbs = 384)
  picture whose two slices each walked ~4 MBs before
  `end_of_slice_flag` or a CABAC failure; we emitted two
  mostly-zero `Frame::Video`, libavcodec emitted none. The fix
  scans `MbGrid::info[]` for any `available == false` entry at
  finalize time and drops the in-progress picture when found
  (the `MbInfo::available` flag was already set per-MB by
  `reconstruct_slice_no_deblock`).
- **sps: reject `frame_cropping` offsets that overshoot the coded
  picture (round 91 fuzz-oracle triage, follow-up).** Per §7.4.2.1.1
  the cropping rectangle must satisfy
  `frame_crop_left_offset ≤ (PicWidthInSamplesL / CropUnitX) − (frame_crop_right_offset + 1)`
  and the mirror inequality on the vertical axis. Previously our
  SPS parser accepted any `ue(v)`-decoded offsets without checking
  against the picture extent; a 506 B input
  (`crash-8e8b38ae…`) carried `frame_crop_top_offset = 2148` against
  a `FrameHeightInSamplesL = 16` picture, so `CropUnitY * (top +
  bottom + 1) = 2 * 2151 = 4302 > 16` — but the SPS still activated,
  slices reconstructed against MB 0 OK, and we emitted two
  `Frame::Video` while libavcodec rejected the SPS outright with
  "crop values invalid 0 5 2148 2 / 2848 16". The fix derives
  CropUnitX / CropUnitY per §7.4.2.1.1 eqs. 7-19..7-22 (4:2:0 / 4:2:2
  / 4:4:4 SubWidthC × SubHeightC table, mbs_only_flag scaling) and
  raises `SpsError::FrameCroppingOutOfRange` when either inequality
  fails. Two new unit tests pin the boundary (max-allowed offsets
  accepted) and the reject path (overshoot rejected); a third
  integration test
  `fuzz_crash_8e8b38ae_sps_crop_out_of_range_rejected` embeds the
  506 B crash verbatim.
- **slice_header: reject `first_mb_in_slice != 0` when opening a new
  coded picture (round 91 fuzz-oracle triage).** Per §7.4.3 + Annex A
  the first VCL slice of a coded picture must cover MB 0 when
  arbitrary slice order (ASO) is not allowed; ASO is gated on the
  same Baseline/Extended-with-FMO carve-out we already reject at PPS
  activation (`DecoderError::FmoNotSupported` — see `decoder.rs`).
  Previously, an input whose every "first" slice carried
  `first_mb_in_slice > 0` opened a fresh `PictureInProgress` whose
  MBs `0..first_mb_in_slice` were never coded; the slice walk did
  return Ok (it just covered the wrong MBs), so
  `any_slice_succeeded` flipped to true and `finalize_in_progress_picture`
  emitted a `Frame::Video` whose luma + chroma planes were the
  zero-initialised allocation buffer. libavcodec rejects the access
  unit outright; we now match. Caught by `ffmpeg_oracle_decode` on
  `crash-957ac808…` (440 B, four IDR slices, all with
  `first_mb_in_slice == 2`, 1×6-MB picture). New regression test
  `fuzz_crash_957ac808_first_mb_nonzero_for_new_picture_rejected`
  pins the libavcodec parity.

### Added

- **sei: round-78 — four additional Annex D HDR + 360 payload parsers.**
  The decoder now recognises four more SEI payload types it previously
  kept as `SeiPayload::Unknown`:
  - **148** `ambient_viewing_environment` (§D.1.34 / §D.2.34) —
    `ambient_illuminance` (`u(32)`, nonzero per §D.2.34) +
    `ambient_light_x` / `ambient_light_y` (`u(16)` each, 0..=50000 per
    §D.2.34 CIE 1931 chromaticity scaled by 50000). Parse-time range
    check rejects illuminance=0 and chromaticity > 50000.
  - **150** `equirectangular_projection` (§D.1.35.1 / §D.2.35.1) —
    `erp_cancel_flag` + (when not cancelled) `erp_persistence_flag` +
    `erp_padding_flag` + `erp_reserved_zero_2bits` (consumed but
    ignored per §D.2.35.1) + the §D.1.35.1 padding-conditional triple
    (`gb_erp_type u(3)`, `left_gb_erp_width u(8)`,
    `right_gb_erp_width u(8)`).
  - **151** `cubemap_projection` (§D.1.35.2 / §D.2.35.2) — the
    shortest body in the family: `cmp_cancel_flag u(1)` plus, when
    not cancelled, `cmp_persistence_flag u(1)`.
  - **154** `sphere_rotation` (§D.1.35.3 / §D.2.35.3) —
    `sphere_rotation_cancel_flag` + (when not cancelled) the
    `sphere_rotation_persistence_flag u(1)`, `reserved_zero_6bits
    u(6)` (consumed but ignored), and three `i(32)` rotation
    angles (yaw, pitch, roll). Parse-time §D.2.35.3 range checks
    enforce yaw / roll ∈ [−180·2^16, 180·2^16 − 1] and pitch ∈
    [−90·2^16, 90·2^16].

  Each new parser ships unit tests (15 new) exercising both
  `parse_*` directly and the `parse_payload` dispatcher for the
  matching `payload_type`. Six new `SeiError` variants police the
  §D.2.34 / §D.2.35.3 range checks. Coverage bumps from 24 → 28
  recognised SEI types; the SEI table header comment in
  `src/sei.rs` is updated.

- **sei: round-73 — five additional Annex D payload parsers.** The
  decoder now recognises five SEI payload types it previously kept as
  `SeiPayload::Unknown`:
  - **9** `scene_info` (§D.1.11 / §D.2.11) — `scene_id` +
    `scene_transition_type` with the §D.1.11 `> 3` branch reading
    `second_scene_id`.
  - **16** `progressive_refinement_segment_start` (§D.1.18 / §D.2.18)
    — `progressive_refinement_id` + `num_refinement_steps_minus1`.
  - **17** `progressive_refinement_segment_end` (§D.1.19 / §D.2.19)
    — pairs with type 16 by `progressive_refinement_id`.
  - **18** `motion_constrained_slice_group_set` (§D.1.20 / §D.2.20)
    — variable-width `slice_group_id[i]` (`u(v)` with
    `v = Ceil(Log2(num_slice_groups))`) drawn from the active PPS via
    a new `SeiContext::num_slice_groups` field; collapses to 0 bits
    when `num_slice_groups <= 1`, with optional `pan_scan_rect_id`.
  - **21** `stereo_video_info` (§D.1.23 / §D.2.23) — both
    `field_views_flag = 1` (top_field_is_left_view_flag) and the
    frame-views branch (`current_frame_is_left_view_flag` +
    `next_frame_is_second_view_flag`) plus the two
    `*_self_contained_flag` bits.

  Each new parser ships unit tests in `sei::tests` (12 new tests)
  exercising both `parse_*` directly and the `parse_payload`
  dispatcher for the corresponding `payload_type`. `SeiContext` loses
  its derived `Default` for a manual impl so the new
  `num_slice_groups` field defaults to `1`. Coverage bumps from
  19 → 24 recognised SEI types; the SEI table header comment in
  `src/sei.rs` is updated to list all 24.

- **encoder: CABAC inter trellis quantisation (round 49 — Lever C
  RDOQ-lite).** A new `EncoderConfig::trellis_quant` toggle (default
  `true`) wires a single-pass greedy Viterbi over the open-loop
  quantised levels in the P-CABAC + B-CABAC inter luma paths.
  `crate::encoder::transform::trellis_refine_4x4_ac` revisits each
  non-zero quantised coefficient at each scan position and considers
  one alternative (`|level|` → `|level| − 1` toward zero); the
  alternative is accepted iff the spatial-domain `D + λ·R` cost
  strictly drops, where the rate term is approximated from §9.3.3.1.3
  (significant_coeff_flag + last_significant_coeff_flag + truncated-
  unary prefix + sign), λ follows the JM `0.85 · 2^((QP-12)/3) · 2/3`
  RDOQ relaxation, and D is the SSD between source and reconstructed
  residual after the actual §8.5.12 inverse transform. The refinement
  is informative — the decoder is unchanged, the bitstream syntax is
  identical — so existing fixtures pin bit-equivalent ffmpeg cross-
  decode (max diff 0) with the trellis enabled. On a 64×64 textured-
  motion P-slice at QP=22 the encoded length drops from 178 → 167
  bytes (-6.2 %) while local-recon PSNR_Y moves from 45.10 → 44.94
  dB (-0.16 dB, well inside quantisation noise); the new
  `integration_trellis_quant.rs` pins this number plus a QP=18..38
  sweep that asserts the trellis never enlarges the inter slice.
  Spec basis: §9.3.3.1.3 (CABAC residual binarisation), §8.5.12
  (inverse 4×4 transform), informative reference JVT-D015. No
  external encoder source consulted.

### Fixed

- **sps: bound `max_num_ref_frames` at 16 (§7.4.2.1.1 / Annex A.3.1).**
  The spec says `max_num_ref_frames` shall be in the range `0..MaxDpbFrames`
  where MaxDpbFrames = `Min(MaxDpbMbs / (PicWidthInMbs * FrameHeightInMbs), 16)`.
  The cap at 16 is absolute — independent of level — and any value above
  it implies a malformed bitstream that libavcodec rejects with
  "too many reference frames". Closes a fuzz-oracle strictness divergence
  on `crash-f9387b9697e640efbd689a9909ff59deaaf32864`, where a stream
  with `max_num_ref_frames=32` was accepted by oxideav-h264 (producing
  partially-decoded pictures from the slices that survived MB-level
  CABAC parse) while libavcodec rejected the SPS outright.
- **reconstruct + inter_pred: reject inter MC against zero-dim
  reference (§8.4.2).** §8.4.2 motion compensation addresses reference
  samples via `Clip3(0, src_width-1, x)` / `Clip3(0, src_height-1, y)`.
  With `src_width` or `src_height == 0` the clip range becomes
  `[0, -1]` (lo > hi), and our `clip3` implementation returns
  `hi == -1`. `-1 as usize == usize::MAX` then indexes the empty
  `ref_pic.cb` slice and panics. A malformed bitstream whose active
  reference list resolves to an uninitialised DPB slot (chroma
  monochrome / placeholder) triggers this in the slow edge-replicated
  path of `interpolate_chroma`. New `ReconstructError::InvalidRefDims`
  guards both `mc_luma_partition` (luma dims) and `mc_chroma_partition`
  (chroma dims) at the MC entry. New `InterPredError::ZeroDimRef`
  guards `interpolate_luma` / `interpolate_chroma` in `inter_pred`,
  `simd::chunked` (and via delegation `simd::scalar`, `simd::portable`)
  for defence-in-depth. Regression for fuzz crash
  `crash-a6f950699905bc0f4d353474b94d4801044c8091`.
- **slice_header: bound `first_mb_in_slice` by PicSizeInMbs
  (§7.4.3).** The spec constrains `first_mb_in_slice` to
  `0..PicSizeInMbs - 1` (non-MBAFF) or `0..PicSizeInMbs/2 - 1` (MBAFF,
  where the parsed value is in macroblock-pair units and the underlying
  raw MB address is `first_mb_in_slice * 2`). With no upper bound
  applied at parse time, a multi-million MB address propagated to
  `reconstruct::mb_sample_origin`, where the `(mb_addr / PicWidthInMbs)
  * 16` i32 multiply overflowed in debug / sanitizer builds. New
  `SliceHeaderError::FirstMbInSliceOutOfRange` carries the parsed
  value + max + PicSizeInMbs + mbaff flag for diagnostics. PicSizeInMbs
  is derived per eq. (7-31) `= PicWidthInMbs * PicHeightInMbs` with
  PicHeightInMbs = FrameHeightInMbs / (1 + field_pic_flag) per eq.
  (7-28), so the check runs after `field_pic_flag` is parsed.
  Regression for fuzz crash
  `crash-80e231ea0edb920c618dc6ea3f9d04626a5b2b16` (175-byte input
  triggering the i32 multiply overflow path through `reconstruct.rs:254`).
- **h264_decoder: drop in-progress picture when every slice failed
  parse / reconstruction (§7.4.1.2.4).** `H264CodecDecoder` collects
  per-slice CABAC/CAVLC failures via an `eprintln!` log line and
  keeps assembling the in-progress picture; the previous behaviour
  then ran §8.7 deblocking on the never-painted picture and pushed
  it through the §C.4 output bumping process, so `receive_frame`
  emitted a zeroed `Frame::Video` for an access unit whose every
  slice was malformed. libavcodec rejects the access unit outright
  in that case (no frame produced), which the fuzz oracle flagged
  as a strictness divergence on a 175-byte input
  (`crash-2ad9589faea57c09b55364bf7ac4af0373e66c18`: PPS + SPS +
  three non-IDR slices, every slice's CABAC stream runs past EOF).
  `PictureInProgress` now carries an `any_slice_succeeded` flag set
  on the OK return of `reconstruct_slice_into_in_progress`; finalize
  drops the picture (no DPB insert, no `VideoFrame` emit) when the
  flag is still false. Multi-slice pictures where some slices
  succeed and others fail are unchanged — the partially-painted
  picture still finalizes, matching the existing best-effort
  recovery for the JVT SVA_Base_B / SL1_SVA_B conformance streams.
- **sps: reject reserved profile_idc values (§A.1 / §A.2).** Only the
  16 profile_idc values enumerated in clause A.2 (66, 77, 88, 100,
  110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135) are
  defined; §A.1 says "all other values of profile_idc and level_idc
  are reserved for future use". `Sps::parse` now returns
  `SpsError::ProfileIdcReserved` for any other value, matching
  libavcodec's "sps_id 0 out of range" rejection of fuzz inputs
  carrying profile_idc=243 / 251 (regression for fuzz crash
  `crash-1b2aaf6f54f92df08bf67802c970555c02de3b4e`).
- **transform: use wrapping arithmetic in §8.5.11.1 / §8.5.11.2
  chroma DC Hadamard + scaling for both 4:2:0 and 4:2:2.** Same
  class of bug as commit `dc09485` (4x4 / 8x8 transforms): a
  malformed CAVLC/CABAC stream can carry residual coefficients near
  i32::MAX, the Hadamard combine sums up to 4 of them, and `f * ls
  << shift` then panicked on debug builds with "attempt to shift
  left with overflow" (regression for fuzz crash
  `crash-5e4c895f03eb99238772ecd84738327f07b4cf75`). Wrapping
  arithmetic absorbs the overflow; the §8.5.13 Clip3 stage in
  reconstruct clamps wrapped pixels back to bit-depth range.
- **decoder: reject FMO PPS activation (§A.2).** A slice that
  activates a PPS with `num_slice_groups_minus1 > 0` (Flexible
  Macroblock Ordering) is now rejected with
  `DecoderError::FmoNotSupported`. Per §A.2.1 / §A.2.3, FMO is
  constrained to the Baseline (profile_idc=66) and Extended
  (profile_idc=88) profiles, and our §8.4 reconstruction path walks
  raster (slice-group-0) order — it does not honour the §8.2.2
  MbToSliceGroupMap. Without this gate the decoder emitted a
  silently-mis-decoded picture for FMO streams that the H.264
  reference decoder (libavcodec) refuses outright with "FMO is not
  implemented", which the fuzz oracle flagged as a strictness
  divergence (`crash-181e9dea7dfa8fcb2c9721a3f9f044214af6167b`).
- **transform: use wrapping arithmetic in §8.5.12 / §8.5.13
  inverse-quant multiplications.** Spec-conformant streams keep `c *
  ls` in i32; only malformed input could drive the product past
  i32::MAX (panic: multiply with overflow). The downstream §8.5.13
  Clip3 stage clamps everything back into bit-depth range, so
  silently wrapping bad input is preferable to panicking on it.
- **cavlc: tighten `level_prefix` cap from 32 to 25 (§9.2.2 note).**
  The spec mandates `level_prefix ≤ 25`. The previous cap of 32 let
  malformed CAVLC streams synthesise `levelCode` values whose
  downstream §8.5.12 multiplication overflowed.
- **cabac: tighten UEG0 escape suffix cap from k=30 to k=14
  (§7.4.5.3.2).** Real coefficients fit in 16 bits; the previous
  k=30 cap let malformed streams produce `coeff_abs_level_minus1`
  values up to ~2^31, which then panicked in the §8.5.12 inverse
  quant multiplication. Cap aligns with the actual spec residual
  range.
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
