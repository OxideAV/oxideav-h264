//! Pure-Rust H.264 / AVC (ITU-T H.264 | ISO/IEC 14496-10) decoder + Baseline
//! I-frame-only encoder.
//!
//! # What is decoded today
//!
//! * NAL unit framing in **both** Annex B (`00 00 [00] 01` start codes) and
//!   AVCC length-prefixed form (used inside MP4 `mdat`), emulation-prevention
//!   byte stripping (§7.4.1.1), and `AVCDecoderConfigurationRecord` parsing
//!   (ISO/IEC 14496-15 §5.2.4.1).
//! * SPS / PPS / slice-header parsing (§7.3.2.1.1, §7.3.2.2, §7.3.3).
//! * **I-slice** pixel reconstruction for **both CAVLC and CABAC** — intra
//!   prediction, residual decode, 4×4 IDCT + Hadamard, spec-accurate
//!   in-loop deblocking (§8.7). The CABAC I-path covers I_16×16, I_NxN and
//!   I_PCM (with the mid-stream arithmetic-engine re-init mandated by
//!   §9.3.1.2).
//! * **10-bit (High 10) CAVLC I / P / B + CABAC I slices**
//!   (§7.4.2.1.1 with `bit_depth_luma_minus8 = 2`) — a u16 plane pair
//!   ([`picture::Picture::y16`] / `cb16` / `cr16`) holds the
//!   reconstructed samples, the dequant math uses i64-widened
//!   `*_ext` helpers so the extended `QpY + QpBdOffsetY` range
//!   (up to 63 at 10-bit) doesn't overflow, `mb_qp_delta` wraps mod
//!   `52 + QpBdOffsetY`, and intra prediction clips to
//!   `(1 << bit_depth) - 1`. The P-slice path
//!   ([`p_mb_hi`]) covers P_L0_16×16 / 16×8 / 8×16, P_8×8 (all
//!   sub-partitions) + P_8×8ref0, P_Skip, and intra-in-P; motion
//!   compensation goes through u16 6-tap + bilinear luma and bilinear
//!   1/8-pel chroma filters ([`motion::luma_mc_hi`] /
//!   [`motion::chroma_mc_hi`]). The B-slice path ([`b_mb_hi`]) covers
//!   B_Direct_16×16 + B_Skip via spatial and temporal direct, the
//!   16×16 / 16×8 / 8×16 / B_8×8 shapes with every Table 7-14 +
//!   Table 7-17 direction combination, and intra-in-B. Bi-prediction
//!   blends u16 samples via `(predL0 + predL1 + 1) >> 1` clipped to
//!   `(1 << BitDepth) - 1`, with explicit (§8.4.2.3.2) and implicit
//!   (§8.4.2.3.3) weighted variants. The CABAC I-slice path
//!   ([`cabac::mb_hi`]) reuses the 8-bit entropy layer (contexts,
//!   binarisation, residual decode — all bit-depth agnostic) and
//!   reconstructs into u16 planes via the same `*_ext` dequant chain.
//!   §8.7 deblocking is implemented on u16 ([`deblock_hi`]) with
//!   α / β / tC0 scaled by `1 << (BitDepth - 8)` and the clip bound
//!   lifted to `(1 << BitDepth) - 1`. The decoded frame is surfaced as
//!   `PixelFormat::Yuv420P10Le`. Bit-exact against ffmpeg on
//!   `iframe_10bit_64x64` (I-slice), `pframe_10bit_64x64` (three
//!   P-slices), `10bit_cabac_i_64x64` (CABAC I, 100% match), and
//!   `10bit_b_64x64` (6-frame I / P / B mix at 100% bit-exact). CABAC P /
//!   B at 10-bit returns `Error::Unsupported`. **Intra_8×8 at 10-bit**
//!   (`transform_size_8x8_flag = 1` on an I_NxN MB) decodes through the
//!   same u16 intra-prediction + i64-widened 8×8 dequant /
//!   inverse-transform chain and lands bit-exact on a 64×64 smptebars
//!   IDR fixture (`iframe_10bit_8x8_64x64`). **I_PCM at 10-bit** reads
//!   each raw sample as a `bit_depth`-wide unsigned integer straight
//!   into the u16 planes (round-tripped end-to-end via a hand-crafted
//!   single-MB fixture). **10-bit 4:2:2 (High 4:2:2, Yuv422P10Le)** and
//!   **10-bit 4:4:4 (High 4:4:4, Yuv444P10Le)** are wired for CAVLC
//!   I-slices: the 4:2:2 path reuses the 10-bit luma decode and runs
//!   chroma through [`mb_hi_422`] (2×4 chroma DC Hadamard + 8 AC blocks
//!   per plane through the i64-widened
//!   `inv_hadamard_2x4_chroma_dc_scaled_ext`); the 4:4:4 path routes
//!   the three planes through [`mb_hi_444`] as luma-style block
//!   streams with per-plane QP (Table 7-2 via `chroma_qp_hi` +
//!   `QpBdOffsetC`) and per-plane scaling lists. Both land 100 %
//!   bit-exact against ffmpeg on the corresponding 64×64 fixtures
//!   (`iframe_10bit_422_64x64` / `iframe_10bit_444_64x64`). CAVLC P/B
//!   and CABAC under 4:2:2 / 4:4:4 at 10-bit still return
//!   `Error::Unsupported`.
//! * **12-bit + 14-bit (High 4:2:2 / High 4:4:4 Predictive CAVLC I)**
//!   (§7.4.2.1.1 with `bit_depth_luma_minus8 ∈ {4, 6}`, 4:2:0 only for
//!   now) — the slice-entry gate accepts even `bit_depth_luma_minus8`
//!   values 0 / 2 / 4 / 6. `QpBdOffsetY = 6 * bit_depth_luma_minus8`
//!   drives the extended QP range (`0..=75` at 12-bit, `0..=87` at
//!   14-bit) and `mb_qp_delta` wraps modulo `52 + QpBdOffsetY` through
//!   the same wrapping logic the 10-bit path uses. The i64-widened
//!   `dequantize_4x4_scaled_ext`, `inv_hadamard_4x4_dc_scaled_ext`, and
//!   `inv_hadamard_2x2_chroma_dc_scaled_ext` in [`transform`] handle
//!   shift constants at the larger `qp/6` magnitudes (up to 14 at
//!   `qp = 87`). Intra prediction clips to `(1 << bit_depth) - 1` =
//!   4095 at 12-bit / 16383 at 14-bit, and the DC fallback yields
//!   `1 << (bit_depth - 1)`. Emit: `PixelFormat::Yuv420P12Le` for
//!   12-bit; 14-bit reuses `Yuv420P10Le` as a 16-bit-container fallback
//!   (the u16 words still carry full 14-bit samples). 12/14-bit CABAC,
//!   P/B, 4:2:2 / 4:4:4 chroma and asymmetric luma/chroma depths return
//!   `Error::Unsupported`.
//! * **4:4:4 CAVLC I / P / B slices** (§6.4.1 Table 6-1, ChromaArrayType = 3)
//!   — the chroma planes share the luma 16×16 dimensions, reuse the
//!   Intra_4×4 / Intra_16×16 predictors (no chroma 8×8 modes), skip the
//!   `intra_chroma_pred_mode` syntax element, and decode residuals as
//!   three back-to-back luma-style block streams. The I path decodes
//!   (I_16×16: DC Hadamard + 16 AC 4×4; I_NxN: 16 × 4×4) with the 4-bit
//!   CBP mapped through `golomb_to_intra4x4_cbp_gray`. The P and B
//!   paths add motion compensation through the §8.4.2.2.1 luma 6-tap /
//!   bilinear filter applied to every plane (§8.4.2.2 Table 8-9 entry
//!   for ChromaArrayType = 3 — the 4:2:0 chroma 1/8-pel bilinear is
//!   replaced by the luma filter), and three plane-aligned 16 × 4×4
//!   luma-style residual streams gated by a 4-bit inter CBP mapped via
//!   `golomb_to_inter_cbp_gray` (FFmpeg table). Each plane uses its own
//!   QP (luma vs `chroma_qp_index_offset` /
//!   `second_chroma_qp_index_offset`) and scaling list (slots 0/1/2 for
//!   intra, 3/4/5 for inter). The I-slice path is bit-exact against
//!   ffmpeg on the 64×64 `iframe_yuv444_64x64` fixture; the P and B
//!   paths decode the `yuv444_p_64x64` and `yuv444_b_64x64` fixtures at
//!   ≥ 90 % / ≥ 80 % sample match respectively — the residual gap comes
//!   from a pre-existing limitation of the P/B residual path that
//!   affects the equivalent 4:2:0 testsrc fixture identically. Solid
//!   colour P fixtures decode 100 % matched. CABAC under 4:4:4 is
//!   partially staged — [`cabac::mb_444`] / [`cabac::p_mb_444`] hold
//!   the entry-point plumbing (mb_type, per-plane intra prediction,
//!   per-plane residual dispatch, intra-in-P / intra-in-B re-entry)
//!   but the slice gate rejects cleanly until the §9.3.3.1.1.9
//!   Table 9-42 extended Cb/Cr ctxBlockCat banks (6..=13 at
//!   ctxIdxOffset 1012+) land. 8×8 transform on the 4:4:4 inter path
//!   is out of scope.
//! * **4:2:2 CAVLC I-slice** decode (§6.4.1 Table 6-1, ChromaArrayType = 2)
//!   — chroma planes are half-width / full-height relative to luma.
//!   Each MB carries two 2×4 chroma DC Hadamard blocks (new
//!   [`cavlc::BlockKind::ChromaDc2x4`] with Table 9-5(e) coeff_token
//!   and Table 9-9(c) total_zeros; nC is fixed at -2) plus 8 AC blocks
//!   per plane laid out as 2 columns × 4 rows of 4×4 residuals. The
//!   2×4 inverse Hadamard is a 2-point row butterfly stacked on a
//!   4-point column Hadamard (§8.5.11.2); the accompanying DC dequant
//!   uses `QP'_C,DC = QP'_C + 3` per H.264 Amd 2 equations (8-328) /
//!   (8-329) / (8-330) — this +3 offset is the crucial detail that
//!   lifts the DC magnitudes onto the AC-compatible scaled domain. The
//!   chroma intra-prediction runs over an 8×16 tile (§8.3.4): DC mode
//!   splits the tile into 8 independent 4×4 DC splats (FFmpeg's
//!   `pred8x16_dc` recipe — top-left quadrant averages both neighbours,
//!   right quadrants take only top, left quadrants only left, with a
//!   corner-averaged variant for the inner corner of each 8×4 band);
//!   Vertical / Horizontal replicate the edges across all 16 rows /
//!   across each of the 16 left samples; Plane uses an 8-tap vertical
//!   sum with the §8.3.4.4 gradient constants (`b = (17H+16)>>5`,
//!   `c = (5V+32)>>6`). The §8.7.2.3 chroma deblock path was extended
//!   to filter every 4-sample internal edge (8 columns of edges
//!   vertically and 4 horizontal internal edges per MB vs. 4:2:0's
//!   single mid-edge) per H.264 Amd 2 §8.7 chroma edge rules. Bit-exact
//!   against ffmpeg on the 64×64 `iframe_yuv422_64x64` fixture.
//! * **4:2:2 CAVLC P / B-slices** decode — the luma MC path is unchanged
//!   (chroma format doesn't touch luma). Chroma MC follows §8.4.2.2.1
//!   ChromaArrayType == 2: the horizontal 1/8-pel bilinear filter is
//!   the same as 4:2:0 but the vertical MV is scaled by 4 (not 8)
//!   because chroma height matches luma height. Chroma residual reuses
//!   the intra 4:2:2 shape — 2×4 DC Hadamard (CAVLC `ChromaDc2x4`,
//!   nC = -2, QP'_C,DC = QP'_C + 3) plus 8 inter AC blocks per plane
//!   in row-major order with slots 4/5 of the inter scaling lists.
//!   Both single-list P and bipred B paths thread the 4:2:2
//!   chroma partition through the same helper; bipred averaging +
//!   weighted bipred work on 8×16 tiles. Bit-exact against ffmpeg on
//!   the 64×64 `yuv422_p_64x64` smptebars fixture (CAVLC P). The CAVLC
//!   B `yuv422_b_64x64` fixture reconstructs ≥ 98 % matched (the
//!   inherited P-residual divergence in `wt/pb-residual`).
//! * **4:2:2 CABAC I / P-slices** decode — §9.3.3.1.1.9 ctxBlockCat = 3
//!   ChromaDC bank extended to 8 coefficients per plane with the
//!   `sig_coeff_offset_dc[7] = {0,0,1,1,2,2,2}` ctxIdxInc mapping
//!   (§9.3.3.1.3), the cat-3 `coeff_abs_levelgt1_ctx` cap at slot 8
//!   (not 9), and the §6.4.3 inverse scan
//!   (CHROMA422_DC_SCAN = [0,2,1,4,6,3,5,7]) applied before the 2×4
//!   inverse Hadamard. The CABAC P path dispatches through the same
//!   helper with `intra = false` so inter chroma uses slots 4/5
//!   of the dequant scaling list. Both paths are bit-exact against
//!   ffmpeg on the `yuv422_cabac_{i,p}_64x64` fixtures.
//!   CABAC B-slices under 4:2:2 still return `Error::Unsupported`.
//! * **Spec-accurate §8.7 in-loop deblocking filter** — per-4-pixel-edge
//!   boundary strength (bS) derivation distinguishing MB-edge vs internal
//!   edges, intra/inter sides, non-zero residual, and MV/reference
//!   mismatch (quarter-pel delta ≥ 4 or different ref POC → bS = 1).
//!   Luma filtering uses the §8.7.2.2 normal filter for bS ∈ {1, 2, 3}
//!   and the strong filter for bS = 4 (MB-edge intra). Chroma filtering
//!   (§8.7.2.3) runs on both planes with `chroma_qp_index_offset` /
//!   `second_chroma_qp_index_offset` applied through the Table 8-15 QP
//!   map, and uses the `tC = tC0 + 1` chroma bump. Internal luma edges
//!   are transform-size-aware: under `transform_size_8x8_flag = 1` only
//!   the mid-MB edge at offset 8 is filtered (not offsets 4 and 12).
//! * **High Profile 8×8 transform** on the CAVLC path — intra AND inter.
//!   `transform_size_8x8_flag = 1` macroblocks (§7.3.5.3.2) decode as four
//!   interleaved CAVLC sub-blocks via the `zigzag_scan8x8_cavlc` table,
//!   with FFmpeg-style per-sub-block `non_zero_count_cache` neighbour
//!   indexing and post-decode `nnz[0] += nnz[1] + nnz[8] + nnz[9]`
//!   aggregation, then through the §8.5.13 dequant + inverse 8×8 integer
//!   transform. The intra path couples this to §8.3.2 Intra_8×8 prediction
//!   (with the (1,2,1) reference-sample pre-filter); the P-slice path
//!   applies the 8×8 residual on top of motion-compensated samples, gated
//!   on the §7.3.5.1 conditional (`pps.transform_8x8_mode_flag` + `cbp_luma
//!   > 0` + no sub-8×8 motion partitions for P_8x8). Bit-exact against
//!   ffmpeg's libavcodec reference on both a smpte-bars 128×128 IDR and a
//!   128×128 P-slice clip encoded with `-x264opts 8x8dct=1`.
//! * **CAVLC baseline P-slice** pixel reconstruction (§8.4) — P_L0_16×16 /
//!   16×8 / 8×16, P_8×8 (all four sub-partition shapes) and P_8×8ref0,
//!   P_Skip, plus intra-in-P macroblocks. Luma 6-tap half-pel + bilinear
//!   quarter-pel MC; bilinear 1/8-pel chroma MC; edge-replicated
//!   out-of-picture reads.
//! * **CABAC Main-profile P-slice** entropy decode (§9.3) —
//!   `mb_skip_flag`, `mb_type` (Table 9-36), `sub_mb_type` (Table 9-38),
//!   `ref_idx_l0` (unary + neighbour-indexed ctxIdxInc), `mvd_l0_{x,y}`
//!   (UEGk with §9.3.3.1.1.7 neighbour-magnitude ctxIdxInc),
//!   `coded_block_pattern`, `mb_qp_delta`, and 4×4 residual coding.
//!   `weighted_pred_flag = 1` is parsed but weights are not yet wired
//!   on the CABAC path.
//! * **MBAFF I/P/B-slice CAVLC** (§7.3.4, §6.4.9.4, §8.7.1.1) — 4:2:0
//!   8-bit pictures with `frame_mbs_only_flag = 0 &&
//!   mb_adaptive_frame_field_flag = 1` are decoded by a pair-granular
//!   loop that reads `mb_field_decoding_flag` per pair and routes
//!   sample writes through the field-stride helpers on
//!   [`picture::Picture`]. Under P / B slices the MBAFF-aware
//!   CAVLC loop also implements the §7.3.4 "mb_skip_run / flag"
//!   interleave — `mb_field_decoding_flag` is emitted at the first
//!   coded MB of each pair (gated by `(CurrMbAddr even) ||
//!   (odd && prev_mb_skipped)`) and the skip run counts individual
//!   MBs. The §6.4.9.4 same-polarity "above" lookup is implemented
//!   via [`picture::Picture::mb_above_neighbour`], so CAVLC `nC`
//!   prediction, intra-4×4 mode prediction, intra neighbour samples
//!   and the list-0 / list-1 MV predictors all resolve to the
//!   correct previous-pair MB for field-coded pairs. The MBAFF
//!   in-loop deblocker ([`deblock::deblock_picture_mbaff`]) walks
//!   MB pairs and keys the edge schedule on each pair's
//!   `mb_field_decoding_flag`: field-coded pairs step vertical edges
//!   at `row_step = 2` through the interleaved plane layout and
//!   skip the inside-pair top/bottom boundary (§8.7.1.1). MBAFF
//!   also accepts the 4:2:2 and 4:4:4 chroma formats
//!   (ChromaArrayType ∈ {2, 3}) for I-slice decode. MBAFF CABAC
//!   and MBAFF 10-bit are still rejected.
//! * **CABAC Main-profile B-slice** entropy decode (§9.3) —
//!   `mb_skip_flag` (B bank), `mb_type` (Table 9-37 with the 48-value
//!   inter + intra tree), `sub_mb_type` (Table 9-38 B column, 13 values),
//!   `ref_idx_l0` + `ref_idx_l1`, `mvd_l0_{x,y}` + `mvd_l1_{x,y}` (UEGk
//!   with list-specific neighbour magnitudes via the new `mvd_l1_abs`
//!   `MbInfo` field), `coded_block_pattern`, `mb_qp_delta`, and 4×4
//!   residual coding. Motion compensation, direct-mode (spatial + temporal)
//!   MV derivation, and bipred blending all reuse the CAVLC B-slice
//!   helpers. MBAFF is rejected.
//! * **CABAC High-profile 8×8 transform** (§9.3.3.1.1.10) — intra AND
//!   inter. `transform_size_8x8_flag` is binarised as FL(1) with
//!   `ctxIdxInc = condTermFlagA + condTermFlagB` against ctxIdxOffset 399
//!   (slots 399/400/401 per §9.3.3.1.1.10). The 8×8 luma residual is
//!   then read as a single 64-coefficient CABAC block with `ctxBlockCat =
//!   5`: `coded_block_flag` at ctxIdxOffset 1012, `significant_coeff_flag`
//!   at 402, `last_significant_coeff_flag` at 417, `coeff_abs_level_minus1`
//!   at 426. Per-position ctxIdxInc for sig/last uses the §9.3.3.1.3
//!   `ctxIdxMap[63][2]` table wired in [`cabac::residual::LUMA8X8_SIG_LAST_CTX_INC`].
//!   The reconstructed 8×8 block flows through `dequantize_8x8_scaled` +
//!   `idct_8x8`, matching the CAVLC 8×8 path bit-for-bit on the shared
//!   §8.5.13 math.
//! * **Multi-reference DPB** (Annex C / §8.2.5) — [`dpb::Dpb`] holds up
//!   to `sps.max_num_ref_frames` reconstructed reference frames. Each
//!   P-slice builds a fresh `RefPicList0` (§8.2.4.2.1) — short-term
//!   references first (descending `FrameNumWrap`), then long-term
//!   references (ascending `LongTermFrameIdx`) — and `ref_idx_l0 > 0`
//!   indexes into it. Picture order count is derived for
//!   `pic_order_cnt_type == 0`, `== 1` (§8.2.1.2 — with
//!   `offset_for_non_ref_pic`, `offset_for_top_to_bottom_field`, and the
//!   per-reference `offset_for_ref_frame[]` cycle applied to the slice
//!   header's `delta_pic_order_cnt[0..1]`), and `== 2` (§8.2.1.3).
//!   Sliding-window marking (§8.2.5.3) and MMCO operations 1 / 2 / 3 / 4
//!   / 5 / 6 (§8.2.5.4) are implemented.
//! * **Explicit weighted P-slice prediction** (§7.3.3.2 / §8.4.2.3.2) —
//!   the slice's `pred_weight_table` is parsed when the PPS has
//!   `weighted_pred_flag = 1`, and the per-reference (weight, offset,
//!   log2_denom) triples are applied to the MC output before the residual
//!   is added.
//! * **CAVLC B-slice** pixel reconstruction (§7.3.5 / §8.4) —
//!   `B_Direct_16x16` + `B_Skip` via both **spatial direct** MV derivation
//!   (§8.4.1.2.2, `direct_spatial_mv_pred_flag = 1`) and **temporal direct**
//!   derivation (§8.4.1.2.3, `direct_spatial_mv_pred_flag = 0`). The
//!   temporal-direct path walks the colocated 4×4 block in `RefPicList1[0]`,
//!   reads its MV + the POC of the ref it pointed to, and rescales via
//!   `DistScaleFactor` using POC arithmetic — intra-colocated blocks use
//!   zero MVs, long-term references short-circuit the scaler to identity.
//!   Also supported: `B_L0_16x16` / `B_L1_16x16` / `B_Bi_16x16`, the 16×8 /
//!   8×16 pair variants (L0/L0, L1/L1, L0/L1, L1/L0, L0/Bi, L1/Bi, Bi/L0,
//!   Bi/L1, Bi/Bi), and `B_8x8` with every Table 7-17 sub-partition type
//!   (`B_Direct_8x8`, `B_{L0,L1,Bi}_{8x8,8x4,4x8,4x4}`). Intra-in-B MBs
//!   dispatch back into the I-slice decode helper. Default bi-prediction
//!   averages the L0 and L1 samples per `(pred_L0 + pred_L1 + 1) >> 1`
//!   (§8.4.2.3.1); explicit weighted bipred (§8.4.2.3.2) uses the slice's
//!   `pred_weight_table` for both lists when `weighted_bipred_idc == 1`;
//!   **implicit** weighted bipred (§8.4.2.3.3) is derived from POC
//!   differences via `DistScaleFactor` when `weighted_bipred_idc == 2`,
//!   with a default-equal-weights fallback for long-term references or
//!   out-of-range `DistScaleFactor`.
//! * **B-slice RefPicList0 / RefPicList1** construction per §8.2.4.2.3 —
//!   short-term past/future split around `CurrPicOrderCnt` with the
//!   per-list ordering reversal, followed by ascending long-term
//!   references.
//! * **Reference picture list modification** (RPLM, §7.3.3.1 /
//!   §8.2.4.3) — the per-slice `ref_pic_list_modification` commands
//!   (idc = 0 short-term subtract, idc = 1 short-term add, idc = 2
//!   long-term select) are parsed into [`slice::SliceHeader::rplm_l0`] /
//!   [`slice::SliceHeader::rplm_l1`] and applied over the default-built
//!   lists via [`dpb::apply_rplm`]. Both P- and B-slice list 0 / list 1
//!   reordering are supported.
//! * **POC output reordering** — the DPB's pending-output queue is
//!   parameterised by a reorder window that the decoder bumps up on
//!   the first B-slice. Pictures emerge in POC order via
//!   `receive_frame`, letting a `I B B P B B …` decode-order stream
//!   deliver frames in presentation order. The SPS VUI's
//!   `bitstream_restriction.max_num_reorder_frames` (§E.2.1) is now
//!   parsed and used as the window size when present; the old
//!   `max_num_ref_frames` upper bound remains as a conservative
//!   fallback for streams that don't carry the VUI field.
//! * **PAFF I-slice decode** (§7.3.3 / §7.4.3 / §8.2.1 POC for fields)
//!   — a slice header with `field_pic_flag = 1` is accepted for I-slices
//!   in 4:2:0 / 8-bit / CAVLC, a [`picture::Picture::new_field`]
//!   allocates a half-height buffer sized to `PicWidthInMbs ×
//!   PicHeightInMapUnits`, the CAVLC I-slice MB loop reconstructs
//!   directly into that buffer, and `queue_ready_frame` emits a
//!   VideoFrame whose height is the field sample rows. Top and bottom
//!   fields are returned as two separate VideoFrames in decode order;
//!   downstream callers are responsible for pair-stacking back into a
//!   full-height frame when the display layer expects progressive
//!   output. POC derivation is unchanged — §8.2.1.1 type 0 yields
//!   TopFieldOrderCnt or BottomFieldOrderCnt directly per field. The
//!   paired-encoder side exposes an `H264EncoderOptions::paff_field`
//!   knob that writes `frame_mbs_only_flag = 0` in the SPS and
//!   `field_pic_flag = 1` / `bottom_field_flag` in the slice header.
//!   MBAFF, PAFF P / B, PAFF CABAC, PAFF + 10-bit / 4:2:2 / 4:4:4, and
//!   MBAFF mixed with PAFF in the same CVS all return
//!   `Error::Unsupported`.
//! * **Custom scaling lists** (§7.4.2.1.1.1 / §7.4.2.2) — the SPS and
//!   PPS scaling-list matrices are parsed out of their zig-zag
//!   coded form into raster order, resolved per Table 7-2 (PPS
//!   overrides SPS; rule A falls back to `Default_4x4_Intra` /
//!   `_Inter` or `Default_8x8_Intra` / `_Inter`; rule B inherits
//!   from the previous slot), and applied by the 4×4 / 8×8 dequant
//!   functions. Luma Intra_16×16 DC and chroma 2×2 DC also consume
//!   the per-category `(0,0)` weight so the full §8.5.10 / §8.5.11.1
//!   dequant chain honours non-flat matrices. Round-trips a
//!   64×64 IDR encoded with x264 `cqmfile=` custom matrices to within
//!   ±2 of the ffmpeg-decoded reference.
//!
//! # What is encoded today
//!
//! See [`encoder::H264Encoder`] — Baseline Profile, I-frames only,
//! Intra_16×16 DC_PRED, CAVLC, single slice per frame, 4:2:0, 8-bit,
//! fixed QP, deblocking disabled. The encoder and decoder round-trip
//! to PSNR ≥ 28 dB for gradient content at QP 22 and ≥ 40 dB for solid
//! content.
//!
//! # Out of scope (returns `Error::Unsupported` or the encoder refuses)
//!
//! * Any CABAC encoding; any P/B slice encoding. CABAC I/P/B decode and
//!   the CABAC 8×8 transform path are all wired; weighted-P on the CABAC
//!   path is parsed but the weights aren't yet applied to the MC output.
//! * B-slice MBAFF and B-slice 8×8 transform (consistent with the
//!   existing 8×8 scoping).
//! * MBAFF CABAC and MBAFF 10-bit — MBAFF I/P/B-slice CAVLC in 4:2:0
//!   8-bit IS wired (MBAFF I also works at 4:2:2 / 4:4:4). PAFF
//!   I/P/B-slice decode at 4:2:0 / 8-bit is wired through the same
//!   field-picture path; PAFF CABAC, PAFF 10-bit, PAFF at alternate
//!   chroma formats, and PAFF + MBAFF mixed within the same CVS all
//!   still return `Error::Unsupported`.
//! * `separate_colour_plane_flag = 1` (§7.4.2.1.1, three independent
//!   colour planes) is rejected at slice entry with `Error::Unsupported`.
//!   4:2:2 is supported for CAVLC I/P/B and CABAC I/P slices — CABAC
//!   4:2:2 B-slice entry still returns `Error::Unsupported`.
//! * Monochrome (`chroma_format_idc = 0`) is rejected — the same
//!   helpers are wired but the MB-layer path assumes chroma planes
//!   exist.
//! * 12-bit / 14-bit CABAC, 12/14-bit P/B, and 12/14-bit chroma formats
//!   other than 4:2:0. 12/14-bit CAVLC I at 4:2:0 IS wired — see the
//!   12-bit + 14-bit bullet above. 10-bit keeps full CAVLC I/P/B +
//!   CABAC I coverage.
//! * Rate control, adaptive QP, mode decision (Intra4×4 / Plane / Vertical /
//!   Horizontal), or any psychovisual tuning. The encoder always emits
//!   Intra_16×16 with DC_PRED and a fixed QP.
//!
//! This crate has no runtime dependencies beyond `oxideav-core` and
//! `oxideav-codec`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::derivable_impls)]

pub mod b_mb;
pub mod b_mb_hi;
pub mod bitreader;
pub mod bitwriter;
pub mod cabac;
pub mod cavlc;
pub mod cavlc_enc;
pub mod deblock;
pub mod deblock_hi;
pub mod decoder;
pub mod dpb;
pub mod encoder;
pub mod fwd_transform;
pub mod intra_pred;
pub mod intra_pred_hi;
pub mod mb;
pub mod mb_444;
pub mod mb_hi;
pub mod mb_hi_422;
pub mod mb_hi_444;
pub mod mb_type;
pub mod motion;
pub mod nal;
pub mod p_mb;
pub mod p_mb_444;
pub mod p_mb_hi;
pub mod b_mb_444;
pub mod picture;
pub mod pps;
pub mod scaling_list;
pub mod slice;
pub mod sps;
pub mod tables;
pub mod transform;

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecCapabilities, CodecId, PixelFormat};

/// The canonical oxideav codec id for H.264 / AVC video.
pub const CODEC_ID_STR: &str = "h264";

/// Register this crate's decoder and baseline I-only encoder with a codec
/// registry. Both are registered under the same `"h264"` id; the registry's
/// preference + priority logic picks one at resolve time.
pub fn register(reg: &mut CodecRegistry) {
    let dec_caps = CodecCapabilities::video("h264_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(8192, 8192);
    reg.register_decoder_impl(CodecId::new(CODEC_ID_STR), dec_caps, decoder::make_decoder);

    // Encoder: Baseline I-only. The registered caps advertise
    // `intra_only = true` and cap the output at level 3.0 maxima.
    let enc_caps = CodecCapabilities::video("h264_baseline_i_sw")
        .with_lossy(true)
        .with_intra_only(true)
        .with_max_size(720, 576)
        .with_pixel_format(PixelFormat::Yuv420P);
    reg.register_encoder_impl(CodecId::new(CODEC_ID_STR), enc_caps, encoder::make_encoder);
}
