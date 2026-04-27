# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
