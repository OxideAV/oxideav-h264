# oxideav-h264

Pure-Rust **H.264 / AVC** (ITU-T Rec. H.264 | ISO/IEC 14496-10) codec
for the [`oxideav`](https://github.com/OxideAV/oxideav-workspace)
framework.

## Status

Active spec-driven implementation built solely from the ITU-T Rec.
H.264 | ISO/IEC 14496-10 specification (2024-08 edition) as the
**single authoritative source**. The decoder reconstructs I / P / B
slices (4:2:0, 4:2:2 and 4:4:4 chroma) through full intra / inter
prediction, transform, deblocking and DPB output ordering on both the
CAVLC and CABAC entropy paths; the encoder emits CAVLC and CABAC
streams across Baseline / Main / High profiles. Field / MBAFF coding,
higher bit depths, and the Annex F/G/H/I scalable / multiview / 3D
extensions are in progress (see the coverage matrix below).

No external decoder source is consulted while writing this
implementation — only the spec PDF. Conformance is verified by
behavioural diff: our decoder is run against synthetic and
public-domain real-world streams and the YUV output is compared
against an independent reference decoder run separately.

The spec PDF itself can't be redistributed (ITU-T copyright) so it
lives in the private OxideAV/docs repo at
[`video/h264/T-REC-H.264-202408-I.pdf`](https://github.com/OxideAV/docs/blob/master/video/h264/T-REC-H.264-202408-I.pdf).
Clone that repo next to your `oxideav` checkout if you're an internal
contributor; otherwise the document is freely downloadable from
https://www.itu.int/rec/T-REC-H.264 (pick the 2024-08 edition).

## Specification coverage

| Spec area | Clause | Status |
| --------- | ------ | ------ |
| NAL parsing (Annex B + AVCC) | §7.3.1, §7.4.1, §B.1 + ISO/IEC 14496-15 §5.2.4.1.1 | parsed (strict §7.3.1 dispatch on `nal_unit_type ∈ {14, 20, 21}`: `parse_nal_unit` reads the leading `svc_extension_flag` / `avc_3d_extension_flag` and parses the matching extension header per §F.7.3.1.1 (SVC, 3 bytes), §G.7.3.1.1 (MVC, 3 bytes — surfaces `non_idr_flag`, `priority_id`, `view_id`, `temporal_id`, `anchor_pic_flag`, `inter_view_flag`, `reserved_one_bit`), or §I.7.3.1.1 (3D-AVC, 2 bytes); emulation-prevention scoping honoured (extension bytes verbatim, only the post-extension RBSP fed through the `00 00 03` stripper). The avcC parser parses the §5.2.4.1.1 High-family extension trailer (`chroma_format`, `bit_depth_luma_minus8`, `bit_depth_chroma_minus8`, SPS-Ext NAL list) for `profile_idc ∈ {100, 110, 122, 144, 244}`, surfaced via `avcc_*` accessors) |
| Sequence Parameter Set | §7.3.2.1 | parsed (VUI + scaling_list fully integrated) |
| Subset Sequence Parameter Set (Annex G MVC) | §7.3.2.1.3, §G.7.3.2.1.4, §G.14.1 | parsed (`subset_seq_parameter_set_rbsp()` for NAL unit type 15: the embedded `seq_parameter_set_data()` reuses the §7.3.2.1.1 parser, then the per-profile branch dispatches on `profile_idc`. Profiles 118 / 128 / 134 get the full §G.7.3.2.1.4 `seq_parameter_set_mvc_extension()` — per-VOIdx `view_id[]`, anchor + non-anchor inter-view dependency lists, per-level operation points, and the profile-134 MFC tail — plus the optional §G.14.1 `mvc_vui_parameters_extension()`. SVC (83/86), MVCD (138/135) and MVCD+3D-AVC (139) bodies surface as typed `*NotParsed` variants with the base SPS still parsed. Subset SPSs are stored in a table separate from ordinary SPSs per §7.4.1.2.1) |
| Coded slice MVC extension (Annex G, NAL 20) | §G.7.3.2.13, §G.7.4.1.2.1, §G.7.4.1.1 | header parsed (NAL unit type 20 with `svc_extension_flag == 0` decodes as a §G.7.3.2.13 `slice_layer_extension_rbsp()` MVC body, reusing the base §7.3.3 `slice_header()` parser with the §G.7.4.1.1 `IdrPicFlag` adaptation and the §G.7.4.1.2.1 subset-SPS id-space resolution. `Event::SliceExtension` surfaces the parsed header + MVC fields. §G.8 inter-view-predicted reconstruction not yet wired) |
| Picture Parameter Set | §7.3.2.2 | parsed (FMO + scaling_list fully integrated; 4:4:4 via `parse_with_chroma_format`) |
| scaling_list body | §7.3.2.1.1.1 | integrated into SPS + PPS |
| VUI parameters (+ HRD) | §E.1 | integrated into SPS |
| AUD + SEI framing + filler + end-of-seq/stream | §7.3.2.3..7 | parsed |
| Slice header | §7.3.3 | parsed (incl. RPLM, pred_weight_table, dec_ref_pic_marking) |
| Slice data + macroblock layer | §7.3.4, §7.3.5 | parsed (I/P/B; 4:2:0 + 4:2:2 + 4:4:4; I_PCM honours `BitDepth{Y,C}` from the SPS on both CAVLC and CABAC paths, incl. §9.3.1.2 post-PCM arithmetic-engine re-init; 4:4:4 P/B inter reconstruction now wired — MBAFF parse still deferred) |
| FMO MB address derivation | §8.2.2, §8.2.3 | implemented (all 7 slice_group_map_types + NextMbAddress) |
| Top-level decoder driver | §7.4.1.2.1 | implemented (events: SPS/PPS stored, AUD, Slice, SEI, end markers, Ignored) |
| I-slice reconstruction (Picture + MB grid + intra + deblock) | §8 / §6.4 | implemented (I_PCM, Intra_4x4, Intra_8x8, Intra_16x16, chroma; §8.3.4.5 4:4:4 I_NxN chroma reconstruction at ChromaArrayType==3 — Cb/Cr "coded like luma" per §7.3.5.3, reusing the per-block luma pred modes with neighbour samples at BitDepthC) |
| P/B-slice reconstruction (MC + ref store + MV wiring) | §8.4 | implemented (P_Skip, P_L0_*, P_8x8; B_Skip, B_Direct, B_L0/L1/Bi_16x16/16x8/8x16; flat scaling; §8.4.2.2 ChromaArrayType==3 4:4:4 inter — the Cb/Cr planes are motion-compensated identically to luma per eq. 8-221/8-222 (`mvCLX == mvLX`) + eq. 8-235..8-238 (full-resolution integer position, quarter-sample fracs, §8.4.2.2.1 luma 6-tap kernel on each chroma plane via `mc_chroma_partition_444`), with the §8.4.2.3 default / explicit / implicit weighted-sample combine reusing the chroma weight tables; the §8.5.5 "coded like luma" inter chroma residual is added by `reconstruct_inter_chroma_residual_444` (cbp_luma-gated 4x4 / 8x8 blocks, inter chroma scaling lists Table 7-2 i=4/5 + i=9/11, no chroma DC Hadamard)) |
| DPB output ordering (POC-ordered delivery + bumping) | §C.4 | implemented |
| Reference picture marking (sliding window + MMCO) | §7.3.3.3, §8.2.5 | implemented |
| Reference picture list construction | §8.2.4 | implemented (frame-only; field-pair interleaving partial) |
| Reference picture list modification (RPLM) | §7.3.3.1, §8.2.4.3 | implemented |
| Picture order count derivation (types 0/1/2) | §8.2.1 | implemented (eq. 8-3 / 8-4 / 8-5 / 8-12 in i64 with `PocError::Overflow` surfaced through `try_from`) |
| Intra prediction (4x4 / 8x8 / 16x16 / chroma) | §8.3 | implemented (all modes) |
| Inter prediction — fractional interpolation + weighted pred | §8.4.2 | implemented (luma 6-tap, chroma bilinear for 4:2:0 / 4:2:2; at ChromaArrayType==3 the chroma planes use the luma 6-tap kernel per §8.4.2.2 eq. 8-235..8-238; default + explicit + implicit weighted) |
| Inter prediction — MV derivation (MVpred, P/B skip, spatial + temporal direct) | §8.4.1 | implemented |
| Transform decode + reconstruction | §8.5 | implemented (4x4 / 8x8 / Hadamard DC / chroma DC) |
| Deblocking filter | §8.7 | implemented (edge-level filter + bS derivation; §8.7.2 eq. (8-450) 4:4:4 chroma deblock at ChromaArrayType==3 filters the Cb/Cr planes with the *luma* filtering process, full-resolution, sharing the luma MB edge geometry + bS with the chroma QPC per side) |
| CAVLC entropy decode | §9.2 | tables + primitives (`parse_residual_block_cavlc` validates the §7.3.5.3.1 call-contract triples before reading bits) |
| CABAC entropy decode — engine | §9.3.1 / §9.3.3.2 | implemented (init + DecodeDecision/Bypass/Terminate) |
| CABAC entropy decode — per-element binarisations + ctxIdx | §9.3.2 / §9.3.3.1 | implemented (mb_skip/mb_type/mvd/ref_idx/mb_qp_delta/cbp/cbf/sig_coeff/coeff_abs/sign/end_of_slice + intra pred mode flags; sub_mb_type and a few high-index init rows deferred) |
| SEI payloads | §D.2 + §G.13.2 + §H.13.2 + §I.13.2 | 52+ payload types parsed, covering buffering_period / pic_timing / pan_scan_rect / filler / user_data (registered ITU-T T.35 + unregistered, incl. a typed ATSC1 `'GA94'` / `'DTG1'` envelope sub-parser) / recovery_point / dec_ref_pic_marking_repetition / spare_pic / the 2003-draft sub-sequence family / film_grain_characteristics (incl. separate-colour-description bit-depth accessors) / frame_packing_arrangement / display_orientation / the HDR family (mastering_display, content_light_level, alternative_transfer_characteristics, ambient_viewing_environment, content_colour_volume, colour_remapping_info, tone_mapping_info) / the full 360 projection family (equirect / cubemap / sphere / region-wise packing / omni viewport) / sei_manifest / sei_prefix_indication / shutter_interval_info / and the Annex G/H/I multiview + 3D-AVC family (multiview_scene_info, multiview_acquisition_info with sign-exponent-mantissa camera params, multiview_view_position, non_required_view_component, operation_point_not_present, base_view_temporal_hrd, three_dimensional_reference_displays_info, depth_representation_info, depth_timing, depth_sampling_info, constrained_depth_parameter_set_identifier, alternative_depth_info with global-view-and-depth camera params). Every parser enforces the spec's range bounds + anti-OOM pre-allocation caps; a deterministic malformed-input sweep + a `sei_payload` cargo-fuzz target pin panic-freedom across all dispatch arms. Remaining payload types fall through to `SeiPayload::Unknown` |

### Encoder coverage

The encoder emits both CAVLC and CABAC elementary streams; every coded
stream is cross-checked bit-exact against an independent reference
decoder run separately.

| Coding tool | Scope | Notes |
| ----------- | ----- | ----- |
| P slices | CAVLC + CABAC | Single + multi L0, integer + quarter-pel ME, P_Skip / P_L0_16x16 / P_8x8 (per-8x8 4MV) |
| B slices | CAVLC + CABAC | Explicit B_L0/L1/Bi_16x16 / 16x8 / 8x16, per-cell mixed B_8x8, B_Skip, B_Direct_16x16 / 8x8 (§8.4.1.2.2 spatial + §8.4.1.2.3 temporal direct), default + explicit weighted bipred |
| MV derivation | §8.4.1.3 | mvp median + C→D substitution, P_Skip MV, spatial/temporal direct mirrored on the encoder side |
| 4:2:2 chroma | High 4:2:2 (`profile_idc=122`), IDR-only | §8.5.11.2 4x2 chroma DC Hadamard, §8.3.4 8x16 chroma intra; P / B-slice 4:2:2 deferred |
| 4:4:4 chroma | High 4:4:4 Predictive (`profile_idc=244`), IDR Intra_16x16 | §7.3.5.3 / §8.3.4.1 chroma "coded like luma" (per-plane 16x16 DC Hadamard + 4x4 AC); I_NxN / P / B 4:4:4 + `separate_colour_plane_flag=1` deferred |
| Intra fallback in P / B | Intra_16x16 | Per-MB Lagrangian RDO against the inter winner (`EncoderConfig::intra_in_inter`); I_NxN / I_PCM fallback + 4:2:2 / 4:4:4 P/B paths deferred |
| Trellis quantisation (RDOQ-lite) | §9.3.3.1.3 + §8.5.12 | Spatial-domain `D + λ·R` one-step-toward-zero refinement on inter luma 4×4 (`EncoderConfig::trellis_quant`, default on) and opt-in on Intra_16x16 luma + chroma AC; informative only — bitstream syntax is unchanged and the decoder is unaffected |

## Goals

* Spec-faithful: every clause of §7, §8, §9 implemented from the spec
  text. Comments cite the clause number being implemented.
* Pure Rust: no `unsafe` outside what the std-lib already requires; no
  C dependencies.
* No external decoder code: nothing copied from any reference
  implementation. Conformance is verified by output diff against an
  independent reference decoder, not by code reading.
* Pluggable into the existing oxideav `Decoder` / `Encoder` trait
  surface (the decoder is registered with the codec registry).

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
  trellis-quant refinement on), and High 4:2:2 IDR. Reports
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
