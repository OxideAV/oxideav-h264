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
| Slice header | §7.3.3 | not implemented |
| Slice data + macroblock layer | §7.3.4, §7.3.5 | not implemented |
| Reference picture marking (sliding window + MMCO) | §7.3.3.3, §8.2.5 | not implemented |
| Reference picture list construction | §8.2.4 | not implemented |
| Reference picture list modification (RPLM) | §7.3.3.1, §8.2.4.3 | not implemented |
| Picture order count derivation | §8.2.1 | not implemented |
| Intra prediction | §8.3 | not implemented |
| Inter prediction (MC, MV derivation, weighted pred) | §8.4 | not implemented |
| Transform decode + reconstruction | §8.5 | not implemented |
| Deblocking filter | §8.7 | not implemented |
| CAVLC entropy decode | §9.2 | tables + primitives (residual_block_cavlc loop in place; nC derivation deferred to slice layer) |
| CABAC entropy decode | §9.3 | engine (init + DecodeDecision/Bypass/Terminate); per-element binarisations deferred to slice layer |

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

## Profiles + features in scope (eventually)

Targeting the subset of H.264 in widespread use:

* Profiles: Baseline (66), Main (77), High (100). Extended /
  High10 / High 4:2:2 / High 4:4:4 left for later.
* Bit depths: 8 (Baseline / Main / High). 10-bit may follow.
* Chroma: 4:2:0 first, then 4:2:2 / 4:4:4.
* Frame coding only first; field / MBAFF / PAFF later if at all.

## License

MIT — see `LICENSE`.
