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
| Slice data + macroblock layer | §7.3.4, §7.3.5 | parsed (I/P/B; 4:2:0 + 4:2:2; MBAFF + 4:4:4 + CABAC I_PCM termination deferred) |
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
| SEI payloads (buffering_period, pic_timing, recovery_point, user_data_unreg, filler, mastering_display, content_light_level) | §D.2 | implemented (7 common types; rest deferred) |

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
