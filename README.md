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
[`h264/T-REC-H.264-202408-I.pdf`](https://github.com/OxideAV/docs/blob/master/h264/T-REC-H.264-202408-I.pdf).
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
| Sequence Parameter Set | §7.3.2.1 | parsed (VUI body + scaling_list deferred) |
| Picture Parameter Set | §7.3.2.2 | parsed (FMO included; scaling_list deferred) |
| scaling_list body | §7.3.2.1.1.1 | parsed (standalone — not yet wired into SPS/PPS) |
| VUI parameters (+ HRD) | §E.1 | parsed (standalone — not yet wired into SPS) |
| AUD + SEI framing + filler + end-of-seq/stream | §7.3.2.3..7 | parsed (SEI payloads per Annex D deferred) |
| Slice header | §7.3.3 | parsed (incl. RPLM, pred_weight_table, dec_ref_pic_marking) |
| Picture Order Count derivation (types 0/1/2) | §8.2.1 | implemented |
| Slice data + macroblock layer | §7.3.4, §7.3.5 | not implemented |
| Reference picture marking (sliding window + MMCO) | §7.3.3.3, §8.2.5 | implemented |
| Reference picture list construction | §8.2.4 | implemented (frame-only; field-pair interleaving partial) |
| Reference picture list modification (RPLM) | §7.3.3.1, §8.2.4.3 | implemented |
| Picture order count derivation | §8.2.1 | not implemented |
| Intra prediction (4x4 / 8x8 / 16x16 / chroma) | §8.3 | implemented (all modes) |
| Inter prediction — fractional interpolation + weighted pred | §8.4.2 | implemented (luma 6-tap, chroma bilinear, default + explicit weighted) |
| Inter prediction — MV derivation (MVpred, P/B skip, spatial + temporal direct) | §8.4.1 | implemented |
| Transform decode + reconstruction | §8.5 | implemented (4x4 / 8x8 / Hadamard DC / chroma DC) |
| Deblocking filter | §8.7 | implemented (edge-level filter + bS derivation) |
| CAVLC entropy decode | §9.2 | tables + primitives (residual_block_cavlc loop in place; nC derivation deferred to slice layer) |
| CABAC entropy decode — engine | §9.3.1 / §9.3.3.2 | implemented (init + DecodeDecision/Bypass/Terminate) |
| CABAC entropy decode — per-element binarisations + ctxIdx | §9.3.2 / §9.3.3.1 | implemented (mb_skip/mb_type/mvd/ref_idx/mb_qp_delta/cbp/cbf/sig_coeff/coeff_abs/sign/end_of_slice + intra pred mode flags; sub_mb_type and a few high-index init rows deferred) |
| SEI payloads (buffering_period, pic_timing, recovery_point, user_data_unreg, filler, mastering_display, content_light_level) | §D.2 | implemented (7 common types; rest deferred) |

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
