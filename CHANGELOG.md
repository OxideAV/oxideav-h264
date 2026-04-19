# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.4](https://github.com/OxideAV/oxideav-h264/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- migrate register() to CodecInfo builder
- reset last_mb_qp_delta tracker across CABAC skip MBs
- bump oxideav-core + oxideav-codec deps to "0.1"
- fix CABAC P ref_idx_l0 drift via 8x8 luma CBF + DPB padding
- fix CABAC P sub_mb_type bin polarity (Table 9-37)
- fix doc_list_item_without_indentation on apply_rplm
- publish ref_idx_l0/mvd before same-MB ctxIdxInc reads (§9.3.3.1.1.6 / .7)
- refresh README to match decoder coverage
- normalise l0_weight_for signature lines
- silence doc_quote_line on temporal_scale_mv docs
- silence remaining clippy warnings so -D warnings passes clean
- direct-cache ctxIdxInc + drop drift-recovery tail
- pick up cargo fmt deltas after wt/enc-p merge
- migrate to oxideav_core::bits + golomb extension trait
- Merge wt/enc-p: CAVLC P-slice encoder (P_L0_16x16 + P_Skip, integer ME)
- Merge wt/sep-plane: separate_colour_plane_flag = 1 CAVLC I full decode
- close CABAC B-slice inter decode on 4:4:4
- Merge wt/sep-colour: PAFF P-field roundtrip + separate_colour_plane narrowing
- Merge wt/yuv444-cabac-full: Table 9-42 Cb/Cr extension banks + CABAC I/P 4:4:4
- stage B-inter scaffolding; re-ignore B test
- wire CABAC P inter partitions
- wire Table 9-42 Cb/Cr extension banks
- Merge wt/si-sp: SI/SP primary slice decode + hand-crafted fixtures
- Merge wt/mbaff-pb: MBAFF P + B CAVLC slice data + widen PAFF P/B gate
- MBAFF P + B CAVLC slice data loops
- Merge wt/yuv444-cabac: 4:4:4 CABAC scaffolding (entry points, fixtures, gate + ignored tests)
- Merge wt/yuv422-cabac: 4:2:2 CABAC I+P + 4:2:2 CAVLC B
- 4:2:2 CABAC I+P + 4:2:2 CAVLC B decode
- Merge wt/12bit: 12-bit + 14-bit CAVLC I-slice decode
- accept MBAFF with chroma_format_idc ∈ {2, 3}
- wire §8.7.1.1 MBAFF in-loop deblocker
- Merge wt/yuv444-ext: 4:4:4 CAVLC P + B slices
- Merge wt/yuv422-ext: 4:2:2 CAVLC P-slice
- 10-bit CAVLC B-slice — bit-exact 64×64 6-frame clip
- 10-bit I-slice CABAC — bit-exact 64×64 at 100%
- Merge wt/mbaff: MBAFF I-slice CAVLC MVP (§7.3.4 + §6.4.9.4)
- Merge wt/paff: PAFF field-picture CAVLC I-slice decode (§7.3.3 field_pic_flag)
- 10-bit (High 10) CAVLC P-slice decode — bit-exact
- 10-bit §8.7 deblocker (u16 samples) — bit-exact I-slice
- Merge wt/yuv422: 4:2:2 CAVLC I-slice decode (§8.5.11.2 + §8.3.4 + §9.2.1)
- h264 docs: document High 10 CAVLC I-slice decode
- h264 10bit: wire CAVLC I-slice decode path (bit-exact 64x64 fixture)
- h264 picture: add u16 plane scaffolding + 10-bit fixture
- Merge wt/followups: Intra_8x8 e[] resize (4 bugs) + CABAC CBP per-8x8 context
- h264 cabac: CBP luma/chroma ctxIdxInc per §9.3.3.1.1.4 Table 9-39
- h264 intra_pred: Intra_8×8 HD/VR/DDR e[] resize to 17 + HD formula fixes
- Merge wt/cabac-8x8: CABAC transform_size_8x8_flag + Luma8×8 residual (§9.3.3.1.1.10, cat 5)
- Merge wt/cabac-b: CABAC B-slices (§9.3.3.1, Table 9-37, 9-38)
- h264 cabac-p test: hard-assert IDR frame (bit-exact after CBF fix)
- h264 cabac: spec-correct CBF neighbour tracking → 100% I-slice match
- h264 cabac: update 64x64 test comment with post-EG0-fix status
- h264 cabac: flip EG0 suffix prefix polarity for coeff_abs_level (§9.3.2.3)
- close three small gaps — POC type 1, VUI reorder, scaling lists
- Merge wt/deblock: full §8.7 deblocker with chroma + spec-correct bS derivation
- spec-accurate §8.7 in-loop deblocking filter
- Merge wt/implicit-bipred: §8.4.2.3.3 implicit weighted bi-prediction
- CABAC INIT_MN_DATA transcription fix + §8.5.6 inverse scan (§9.3.1.1)
- Merge wt/temporal-direct: §8.4.1.2.3 temporal direct MV prediction for B-slices
- CABAC I-slice residual-decode partial fixes (§9.3.3.1)
