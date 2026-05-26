# oxideav-h264

Pure-Rust **H.264 / AVC** (ITU-T Rec. H.264 | ISO/IEC 14496-10) codec
for the [`oxideav`](https://github.com/OxideAV/oxideav-workspace)
framework.

## Status

**Empty crate, under spec-driven rewrite.** A previous implementation
lives on the [`old`](https://github.com/OxideAV/oxideav-h264/tree/old)
branch — it decodes I-slices and most P/B-slices on synthetic content,
but accumulated enough small spec-deviations on real-world streams
that bug-by-bug fixing stopped paying off. We're rebuilding from a
blank slate with the ITU-T Rec. H.264 | ISO/IEC 14496-10 specification
(2024-08 edition) as the **single authoritative source**.

No external decoder source (libavcodec, openh264, JM reference, x264,
…) is consulted while writing this implementation — only the spec
PDF. Conformance is verified by behavioural diff (run our decoder
against synthetic and public-domain real-world streams, compare YUV
output against an external reference decoder run separately).

The spec PDF itself can't be redistributed (ITU-T copyright) so it
lives in the private OxideAV/docs repo at
[`video/h264/T-REC-H.264-202408-I.pdf`](https://github.com/OxideAV/docs/blob/master/video/h264/T-REC-H.264-202408-I.pdf).
Clone that repo next to your `oxideav` checkout if you're an internal
contributor; otherwise the document is freely downloadable from
https://www.itu.int/rec/T-REC-H.264 (pick the 2024-08 edition).

## Specification coverage

This section will grow as the rewrite lands. **Today: nothing
decodes.** The crate registers a no-op against the workspace's
codec registry so other crates that depend on it keep compiling, but
no packet is decoded.

| Spec area | Clause | Status |
| --------- | ------ | ------ |
| NAL parsing (Annex B + AVCC) | §7.3.1, §7.4.1, §B.1 | parsed (no extension headers) |
| Sequence Parameter Set | §7.3.2.1 | parsed (VUI + scaling_list fully integrated) |
| Picture Parameter Set | §7.3.2.2 | parsed (FMO + scaling_list fully integrated; 4:4:4 via `parse_with_chroma_format`) |
| scaling_list body | §7.3.2.1.1.1 | integrated into SPS + PPS |
| VUI parameters (+ HRD) | §E.1 | integrated into SPS |
| AUD + SEI framing + filler + end-of-seq/stream | §7.3.2.3..7 | parsed (SEI payloads per Annex D deferred) |
| Slice header | §7.3.3 | parsed (incl. RPLM, pred_weight_table, dec_ref_pic_marking) |
| Slice data + macroblock layer | §7.3.4, §7.3.5 | parsed (I/P/B; 4:2:0 + 4:2:2 + 4:4:4; I_PCM honours `BitDepth{Y,C}` from the SPS on both CAVLC and CABAC paths, incl. §9.3.1.2 post-PCM arithmetic-engine re-init; MBAFF parse + 4:4:4 P/B inter still deferred) |
| FMO MB address derivation | §8.2.2, §8.2.3 | implemented (all 7 slice_group_map_types + NextMbAddress) |
| Top-level decoder driver | §7.4.1.2.1 | implemented (events: SPS/PPS stored, AUD, Slice, SEI, end markers, Ignored) |
| I-slice reconstruction (Picture + MB grid + intra + deblock) | §8 / §6.4 | implemented (I_PCM, Intra_4x4, Intra_8x8, Intra_16x16, chroma) |
| P/B-slice reconstruction (MC + ref store + MV wiring) | §8.4 | implemented (P_Skip, P_L0_*, P_8x8; B_Skip, B_Direct, B_L0/L1/Bi_16x16/16x8/8x16; flat scaling) |
| DPB output ordering (POC-ordered delivery + bumping) | §C.4 | implemented |
| Reference picture marking (sliding window + MMCO) | §7.3.3.3, §8.2.5 | implemented |
| Reference picture list construction | §8.2.4 | implemented (frame-only; field-pair interleaving partial) |
| Reference picture list modification (RPLM) | §7.3.3.1, §8.2.4.3 | implemented |
| Picture order count derivation (types 0/1/2) | §8.2.1 | implemented |
| Intra prediction (4x4 / 8x8 / 16x16 / chroma) | §8.3 | implemented (all modes) |
| Inter prediction — fractional interpolation + weighted pred | §8.4.2 | implemented (luma 6-tap, chroma bilinear, default + explicit weighted) |
| Inter prediction — MV derivation (MVpred, P/B skip, spatial + temporal direct) | §8.4.1 | implemented |
| Transform decode + reconstruction | §8.5 | implemented (4x4 / 8x8 / Hadamard DC / chroma DC) |
| Deblocking filter | §8.7 | implemented (edge-level filter + bS derivation) |
| CAVLC entropy decode | §9.2 | tables + primitives (residual_block_cavlc loop in place; nC derivation deferred to slice layer) |
| CABAC entropy decode — engine | §9.3.1 / §9.3.3.2 | implemented (init + DecodeDecision/Bypass/Terminate) |
| CABAC entropy decode — per-element binarisations + ctxIdx | §9.3.2 / §9.3.3.1 | implemented (mb_skip/mb_type/mvd/ref_idx/mb_qp_delta/cbp/cbf/sig_coeff/coeff_abs/sign/end_of_slice + intra pred mode flags; sub_mb_type and a few high-index init rows deferred) |
| SEI payloads (buffering_period, pic_timing, pan_scan_rect, filler, user_data_registered_itu_t_t35, user_data_unreg, recovery_point, dec_ref_pic_marking_repetition, spare_pic, scene_info, sub_seq_info, sub_seq_layer_characteristics, sub_seq_characteristics, full_frame_freeze, full_frame_freeze_release, full_frame_snapshot, progressive_refinement_segment_start, progressive_refinement_segment_end, motion_constrained_slice_group_set, film_grain_characteristics, deblocking_filter_display_preference, stereo_video_info, post_filter_hint, tone_mapping_info, frame_packing_arrangement, display_orientation, colour_remapping_info, mastering_display, content_light_level, alternative_transfer_characteristics, ambient_viewing_environment, content_colour_volume, equirectangular_projection, cubemap_projection, sphere_rotation, regionwise_packing, omni_viewport, sei_manifest, sei_prefix_indication, shutter_interval_info) | §D.2 | implemented (40 types incl. dec_ref_pic_marking_repetition + colour_remapping_info (pre/post LUTs + 3x3 matrix, §D.2.30) + content_colour_volume + spare_pic error-concealment hint + the 2003-draft sub-sequence family + HDR + the complete 360 projection family (equirect / cubemap / sphere / region-wise packing / omni viewport) + sei_manifest / sei_prefix_indication transport-layer hints + shutter interval; rest deferred. **Round 158** — typed ATSC1 envelope sub-parser (`parse_atsc1_envelope`) on top of the §D.2.6 `user_data_registered_itu_t_t35` (country=0xB5 / provider=0x0031): decodes `'GA94'` → `MPEG_cc_data()` (cc_data bytes opaque, CEA-708 layout deferred — see docs-gap below) + `bar_data()` (Table 6.11 letterbox / pillarbox pairs incl. §6.2.3.2 mutual-exclusion constraint) + reserved type codes, and `'DTG1'` → `afd_data()` (Table 6.13 / 6.14 active_format)) |

### Encoder coverage

Round-by-round encoder progress:

| Round | Scope | Notes |
| ----- | ----- | ----- |
| 16 | P-slice support | Single L0 reference, integer-pel ME, P_Skip + P_L0_16x16 |
| 19 | 4MV / P_8x8 | All sub_mb_type=PL08x8, per-8x8 quarter-pel ME |
| 20 | B-slice support | Explicit B_L0_16x16 / B_L1_16x16 / B_Bi_16x16, default weighted bipred |
| 21 | B_Skip / B_Direct_16x16 | §8.4.1.2.2 spatial direct, uniform-MV gate |
| 22 | B 16x8 / 8x16 partitions | Table 7-14 mb_types 4..=21 (all per-partition L0/L1/Bi combos) |
| 23 | B_8x8 + 4× B_Direct_8x8 | Per-8x8 spatial direct (lifts uniform-MV gate via per-partition MVs) |
| 24 | B_Direct §8.4.1.2.3 temporal | POC-distance-scaled colocated MVs, `direct_temporal_mv_pred` toggle |
| 26 | Explicit weighted bipred | §7.4.2.2 `weighted_bipred_idc=1`, per-slice `pred_weight_table()` (luma, log2_wd=5), least-squares (w0,w1,off) selector, `explicit_weighted_bipred` toggle. Fade-in fixture: B-slice 1148 → 21 bytes (98 % smaller); ffmpeg cross-decode bit-equivalent. |
| 27 | 4:2:2 chroma (IDR-only) | §7.3.2.1.1 High 4:2:2 SPS (`profile_idc=122`, `chroma_format_idc=2`), §8.5.11.2 4x2 chroma DC Hadamard, §8.3.4 8x16 chroma intra (DC/H/V/Plane with `yCF=4` and `c=5*V`), `ChromaDc422` CAVLC context, 8 4x4 chroma blocks per plane. ffmpeg cross-decode bit-exact at QP 26; luma PSNR ≥ 40 dB on synthetic gradient. P / B-slice 4:2:2 deferred. |
| 28 | 4:4:4 chroma (IDR Intra_16x16-only) | §7.3.2.1.1 High 4:4:4 Predictive SPS (`profile_idc=244`, `chroma_format_idc=3`, `separate_colour_plane_flag=0` → ChromaArrayType=3). §7.3.5.3 / §8.3.4.1 chroma "coded like luma": each plane gets its own 16x16 DC Hadamard + 16 4x4 AC blocks (§6.4.3 raster-Z), shares the luma Intra_16x16 pred mode, `intra_chroma_pred_mode` omitted from bitstream. Per-plane chroma AC gated by `cbp_luma == 15` (encoder promotes when chroma has nonzero AC). New `ChromaWriteKind::Yuv444` writer variant + decoder `reconstruct_chroma_intra_444` helper. `disable_deblocking_filter_idc=1` (the §8.7 chroma deblock for ChromaArrayType==3 — luma filter applied per-plane — is a separate plumbing exercise). ffmpeg cross-decode bit-exact at QP 26; PSNR Y=42.25 / U=V=50.85 dB on a 64x64 synthetic gradient. I_NxN / P / B 4:4:4 + `separate_colour_plane_flag=1` deferred. |
| 29 | Intra fallback in P / B slices (Intra_16x16) | §7.3.5 Tables 7-13 / 7-14 — per-MB Lagrangian RDO trials Intra_16x16 against the inter winner inside `encode_p` / `encode_b`, picking the lower-J path. Intra mb_types emitted with `+5` (P-slice) / `+23` (B-slice) offset; existing decoder accepts them unchanged. New `EncoderConfig::intra_in_inter` toggle (default `true`) for A/B testing. New `write_intra16x16_mb_in_inter_slice` writer + `InterMbStateSnapshot` snapshot/restore helper. Occlusion fixture (64x64 textured wall + 32x32 fresh-content ball) shrinks P-slice from **588 → 108 bytes (81.6 % smaller)**; ffmpeg cross-decode bit-exact (max diff 0); self-roundtrip bit-exact. I_NxN / I_PCM fallback + 4:2:2 / 4:4:4 P/B paths still deferred. |
| 30 | CABAC encode — IDR + P slices | §9.3 + Tables 9-12..9-26 — `Encoder::encode_idr_cabac` + `encode_p_cabac` entry points wire the CABAC arithmetic encoder (`encoder::cabac_engine`) + per-syntax binarisations (`encoder::cabac_syntax`). PPS `entropy_coding_mode_flag = 1`, slice header `cabac_init_idc = 0`. P-slice forces MV = (0, 0) so encoder/decoder mvp agree; per-MB pick of P_Skip / P_L0_16x16. ffmpeg cross-decode bit-exact at QP 26 on 64x64 gradient (PSNR_Y ≈ 49 dB). |
| 31 | CABAC encode — B slices | §9.3 + §9.3.3.1.1.3 (mb_type ctxIdxOffset = 27) + §9.3.3.1.1.1 (mb_skip ctxIdxOffset = 24) — new `Encoder::encode_b_cabac` entry point + `encode_mb_type_b` syntax helper covering Table 9-37 B-slice rows (B_Direct_16x16, B_L0/L1/Bi_16x16, partition modes, intra suffix). Per-MB pick of {B_L0_16x16, B_L1_16x16, B_Bi_16x16} on minimum-luma-SAD against MV=(0,0) predictors; `BSliceHeaderConfig` gains `cabac` field for `cabac_init_idc` emission. ffmpeg decodes a 16-frame B-pyramid GOP (1 IDR + 7 P + 7 B) at PSNR_Y ≈ 51.5 dB on smooth content. B_Skip / B_Direct_16x16 / 4:2:2 + 4:4:4 P/B deferred. |
| 32 | CABAC encode — real ME + B_Skip / B_Direct_16x16 | §8.4.1.3 (mvp median + C→D substitution) + §8.4.1.2 (P_Skip MV) + §8.4.1.2.2 (B-slice spatial direct, uniform-MV gate) — `cabac_path` learns to track per-8x8 MVs/refIdxs in `CabacEncMbInfo` (`mv_l0_8x8` + `ref_idx_l0_8x8` + L1 mirrors) and ships `cabac_mvp_for_16x16` / `cabac_p_skip_mv` / `cabac_b_spatial_direct_derive` helpers reading neighbours via §6.4.11.7 8x8-cell selection. P-CABAC: real qpel ME (`search_quarter_pel_16x16`) replaces the round-30 forced MV=(0,0) — `mvd = mv - mvp` against the median predictor, P_Skip eligibility now requires `chosen_mv == skip_mv`. B-CABAC: per-list qpel ME, explicit-inter rows emit real mvds; the §8.4.1.2.2 spatial-direct derivation is mirrored on the encoder side and emits `B_Direct_16x16` when its SAD is competitive (Lagrangian QP-scaled bias, ~16-128 SAD units) or `B_Skip` when the residual additionally quantises to all-zero (mb_skip_flag = 1 with no further MB syntax). Per-8x8 / mixed `B_8x8`+all-Direct under CABAC + 4:2:2 / 4:4:4 P/B still deferred. ffmpeg cross-decode bit-exact on translation-motion fixtures (`round32_p_cabac_motion_ffmpeg_interop` + `round32_b_cabac_motion_ffmpeg_interop` both achieve PSNR_Y ≈ 99 dB at QP 22) and ≥ 46 dB on a textured-background motion fixture; B-pyramid GOP shrinks 555 → 487 bytes (12 %) on smooth content. |
| 475 | CABAC encode — B 16x8 / 8x16 partition modes + B_8x8 all-Direct | §7.3.5.1 (per-partition mb_pred order: ref_idx_l0[..] / ref_idx_l1[..] / mvd_l0[..] / mvd_l1[..]), §7.4.5 Table 7-14 (mb_type 4..21 = nine 16x8 + nine 8x16 (mode_a, mode_b) variants), Table 7-18 (sub_mb_type 0 = B_Direct_8x8), §8.4.1.3 directional shortcuts (`Partition16x8Top` / `Partition16x8Bottom` / `Partition8x16Left` / `Partition8x16Right`), §9.3.3.1.1.7 within-MB mvd-sum tracking. New `encode_sub_mb_type_p` + `encode_sub_mb_type_b` (Tables 9-37 / 9-38) in `cabac_syntax`; `cabac_enc_mvd_abs_sum` mirrors the decoder's per-4x4 absMvdComp lookup so the §9.3.3.1.1.7 ctxIdxInc lands on the same context regardless of partition shape; `cabac_mvp_for_b_16x8_partition` / `cabac_mvp_for_b_8x16_partition` reuse `derive_mvpred_with_d` with within-MB inflight neighbour reads. Per-MB deblock state (`MbDeblockInfo.mv_l0` / `mv_l1` / `ref_idx_*` / `ref_poc_*`) is now built per-4x4 from the partition layout so the internal partition boundary's bS derivation matches the decoder. B_8x8 + 4× B_Direct_8x8 (mb_type=22) gates on `!direct.is_uniform()` (per-quadrant MVs differ from the §8.4.1.2.2 colZeroFlag step-7 override) and stitches a per-quadrant predictor via `build_inter_pred_luma_8x8` / `build_inter_pred_chroma_4x4`; emits no ref_idx / mvd (decoder re-derives all four (mvL0, mvL1, refIdxL0, refIdxL1) per §8.4.1.2.2). ffmpeg cross-decode bit-exact (≤ 1 LSB) on the per-half-motion fixtures forcing B_L0_L1_16x8 / B_L0_L1_8x16 (PSNR_Y ≥ 38 dB) and on the bipred-on-midpoint fixture exercising B_8x8 + all-Direct. Per-cell mixed B_8x8 (per-quadrant L0 / L1 / Bi) still deferred. |
| 475 (cont.) | CABAC encode — per-cell mixed B_8x8 | §7.3.5.2 sub_mb_pred() per-cell mvd_l0[i] / mvd_l1[i] in cell-raster order (0, 1, 2, 3 = TL, TR, BL, BR), Table 7-18 (sub_mb_type ∈ {0, 1, 2, 3} = B_Direct_8x8 / B_L0_8x8 / B_L1_8x8 / B_Bi_8x8), §8.4.1.3 with `MvpredShape::Default` for 8x8 sub-MB partitions. New `cabac_mvp_for_b_8x8_partition` derives per-cell mvp using a `cell_decoded[4]` flag separate from per-list inflight, distinguishing "cell decoded but list X unused" (returns `intra_but_mb_available`) from "cell not yet decoded" (returns `partition_not_yet_decoded_same_mb`) — the §8.4.1.3.2 C→D substitution gate depends on the difference. Per-cell selector reuses CAVLC `pick_best_b8x8_cell` (Lagrangian SAD with per-cell rate cost approximations) and stitches per-cell luma + chroma predictors via `build_b_explicit_8x8_cell_luma` / `build_b_explicit_4x4_cell_chroma`. Direct cells contribute their §8.4.1.2.2-derived MVs / refs to the inflight even though they emit no mvd, mirroring the decoder's per-cell `process_partition` grid update. Per-MB grid + per-4x4 deblock state populated from the per-cell layout. ffmpeg cross-decode bit-exact (max diff 0) at PSNR_Y = 44.20 dB on the round-25 mixed-motion fixture; B-slice shrinks 99 → 73 bytes (~26 %) on the 64×64 round-475 per-half-motion fixture. B-CABAC intra fallback wiring + 4:2:2 / 4:4:4 P/B CABAC still deferred. |
| 49 | CABAC inter trellis quantisation (RDOQ-lite) | §9.3.3.1.3 (CABAC residual binarisation) + §8.5.12 (inverse 4×4 transform). New `EncoderConfig::trellis_quant` toggle (default `true`) wires `crate::encoder::transform::trellis_refine_4x4_ac` into the P-CABAC + B-CABAC inter luma 4×4 paths. After the open-loop `quantize_4x4` step, the trellis revisits each non-zero quantised level in reverse-raster order and accepts a one-step move toward zero whenever the spatial-domain `D + λ·R` cost strictly drops. D uses the actual §8.5.12 inverse transform (no Parseval approximation); R is approximated from the CABAC binarisation (sig + last-sig + truncated-unary prefix + sign); λ follows the `0.85 · 2^((QP-12)/3) · 2/3` RDOQ relaxation. The trellis is informative — bitstream syntax is unchanged, the decoder is unaffected, and ffmpeg cross-decodes the trellis-encoded stream bit-equivalently. On a 64×64 textured-motion P-slice at QP=22 the encoded length drops 178 → 167 bytes (**-6.2 %**) with PSNR_Y moving 45.10 → 44.94 dB (-0.16 dB). New `integration_trellis_quant.rs` pins the QP=22 6 % saving + a QP=18..38 sweep asserting trellis never enlarges any inter slice. Intra paths (chroma AC/DC) deferred to a later round. |
| 148 | CABAC IDR Intra_16x16 luma AC trellis (opt-in) | §8.5.10 (luma DC Hadamard round trip) + §9.3.3.1.3 (CABAC residual binarisation). Extends the round-49 refinement to the 16 per-block 4×4 AC arrays inside an Intra_16x16 luma MB, passing `skip_dc=true` so the §8.5.10 inverse-Hadamard DC chain stays bit-exact. New `EncoderConfig::trellis_quant_intra` toggle (default `false`) wires `quantize_intra16x16_luma` through `trellis_refine_4x4_ac`. The flag defaults to off because the round-49 rate model is calibrated for inter blockCat 2 and is net-negative on dense intra Intra_16x16 AC at low QP (it picks up the obvious savings only on high-noise content). Bitstream syntax is unchanged; the decoder is unaffected; `cbp_luma` keeps the §7.4.5.1 Table 7-11 value (`15` whenever any AC remains). New `integration_trellis_quant_intra.rs` (4 tests) pins: (1) default-off byte-for-byte determinism; (2) decoder self-roundtrip bit-exact (max-luma-diff = 0) at QP=22; (3) PSNR drift inside a 3 dB envelope across QP {22, 28, 34}; (4) QP {18..42} sweep stays decodable + bit-exact. Re-calibrating the rate model for blockCat 1 (Intra16x16AC) so the lever can ship default-on is the obvious follow-up. |
| 151 | CABAC IDR Intra_16x16 chroma AC trellis (opt-in) | §8.5.10 (luma-style DC Hadamard, applied to 4:4:4 chroma) + §8.5.11.1 (4:2:0 2×2 chroma-DC Hadamard) + §7.3.5.3 (chroma "coded like luma" at 4:4:4) + §9.3.3.1.3 (CABAC residual binarisation). Extends the round-148 luma trellis to the chroma planes inside `encode_chroma_intra16x16_420` (4 per-plane 4×4 AC arrays) and `encode_chroma_intra16x16_444` (16 per-plane 4×4 AC arrays). Each block runs through `trellis_refine_4x4_ac` with `skip_dc=true` so the chroma-DC Hadamard chain stays bit-exact. New `EncoderConfig::trellis_quant_intra_chroma` toggle (default `false`); bitstream syntax is unchanged; the decoder is unaffected; `cbp_chroma` keeps the §7.4.5.1 Table 7-15 value (`0`/`1`/`2`). New `integration_trellis_quant_intra_chroma.rs` (6 tests) pins: 4:2:0 + 4:4:4 default-off byte-for-byte determinism; 4:2:0 + 4:4:4 decoder self-roundtrip plane-wise bit-exact at QP=22 (max diff Y/Cb/Cr = 0); QP {18..42} sweep stays decodable + plane-wise bit-exact; luma PSNR invariant under the chroma lever (|drift| ≤ 0.05 dB across QP {22, 28, 34}); plus a measurement-only byte-savings print across QP {18..38} (effect on round-151 textured fixture is roughly net-zero at this QP range, consistent with the inter-calibrated rate model). Re-calibrating the rate model for chroma blockCats 4 / 1 so the lever can ship default-on remains the obvious follow-up. |

## Goals

* Spec-faithful: every clause of §7, §8, §9 implemented from the spec
  text. Comments cite the clause number being implemented.
* Pure Rust: no `unsafe` outside what the std-lib already requires; no
  C dependencies.
* No external decoder code: nothing copied from libavcodec / openh264
  / JM / etc. Conformance is verified by output diff, not by code
  reading.
* Pluggable into the existing oxideav `Decoder` / `Encoder` trait
  surface once functional.

## Benchmarks

Criterion benches live under `benches/`:

* `inter_pred_bench` — §8.4.2 luma + chroma fractional interpolation
  (scalar / chunked / default 6-tap kernels, 4x4 / 8x8 / 16x16 blocks).
* `transform_bench` — §8.5 inverse 4x4 and 8x8 transforms at QP 26.
* `decode` — end-to-end `H264CodecDecoder` over five in-process
  corpora: 64x64 + 128x96 Baseline IDR, IDR+4×P Baseline chain,
  IDR+P+B Main GOP, and a 64x64 High 4:2:2 IDR. All streams are
  synthesised from `oxideav-h264`'s own encoder so the bench needs
  no on-disk fixtures.
* `encode` — sibling to `decode`, driving the public encoder entry
  points across seven in-process variants: Baseline CAVLC IDR
  (64x64 + 128x96), Baseline IDR + 4×P chain (real per-MB integer ME),
  Main IDR + P + B GOP (§8.4.1.2.2 spatial-direct + bipred),
  CABAC IDR + IDR + P (`encode_idr_cabac` / `encode_p_cabac` with the
  round-49 trellis-quant refinement on), and High 4:2:2 IDR. Reports
  both per-frame (`Throughput::Elements`, ≈ frames-encoded per
  second) and per-source-byte (`Throughput::Bytes`, MiB/s of raw
  planar input absorbed). The encoded-stream byte count is a
  compression-efficiency metric and stays in the integration tests
  rather than the bench.

Run with `cargo bench --bench decode -- --quick`
or `cargo bench --bench encode -- --quick` for fast smokes,
or omit `--quick` for the full Criterion report.

## Profiles + features in scope

The long-term target is a **complete** implementation of ITU-T Rec.
H.264 | ISO/IEC 14496-10 — every profile and every coding tool in the
spec. We're starting from the subset in widespread use and widening
out from there:

| Phase | Profiles | Bit depths | Chroma | Coding |
| ----- | -------- | ---------- | ------ | ------ |
| 1 (now) | Baseline (66), Main (77), High (100) | 8 | 4:2:0 | Frame only |
| 2 | + High10 (110), High 4:2:2 (122) | + 10-bit | + 4:2:2 | + Field / MBAFF / PAFF |
| 3 | + High 4:4:4 Predictive (244), Extended (88) | + 12 / 14-bit | + 4:4:4 | Full coverage |
| 4 | + Annex F SVC, Annex G MVC, Annex H 3D-AVC, Annex I 3D-AVC depth | all | all | all |

Intermediate profiles (High 10 Intra, CAVLC 4:4:4 Intra, etc.) land
opportunistically as the pieces they need go in. Nothing in the spec
is considered out-of-scope forever — the table is a phasing plan, not
a scope fence.

## License

MIT — see `LICENSE`.
