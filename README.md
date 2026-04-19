# oxideav-h264

Pure-Rust **H.264 / AVC** (ITU-T H.264 | ISO/IEC 14496-10) codec for
oxideav. Decodes CAVLC + CABAC I-slices, CAVLC + CABAC P-slices, and
CAVLC + CABAC B-slices (spatial + temporal direct + explicit weighted
bipred + implicit weighted bipred); primary **SI / SP slices** decode
through the CAVLC 4:2:0 / 8-bit path (`slice_qs_delta ==
slice_qp_delta` — the §8.6 identity case); encodes a **Baseline-Profile
CAVLC** output stream with all four **Intra_16×16** modes (Vertical /
Horizontal / DC / Plane) selected per-MB by SAD, all four chroma intra
modes (DC / Horizontal / Vertical / Plane), and CAVLC
**P_L0_16×16 / P_Skip** inter macroblocks driven by ±16 integer-pel
motion estimation with 3×3 half-pel refinement through the §8.4.2.2.1
6-tap / bilinear filter. **High-Profile 8×8 transform** is wired
end-to-end on both the CAVLC path (§7.3.5.3.2, §8.5.13, bit-exact
against libavcodec) and the CABAC path (§9.3.3.1.1.10 for the transform
size flag + §9.3.3.1.1.9 ctxBlockCat = 5 for the 64-coefficient
residual). Zero C dependencies.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.0"
oxideav-codec = "0.0"
oxideav-h264 = "0.0"
```

## Quick use

H.264 is typically delivered inside a container (MP4 / MOV / Matroska) or
as an Annex B elementary stream. The crate exposes the front-end
`H264Decoder` directly and registers itself with a `CodecRegistry` under
the id `"h264"`.

```rust
use oxideav_codec::{CodecRegistry, Decoder};
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};

let mut codecs = CodecRegistry::new();
oxideav_h264::register(&mut codecs);

// When driving from MP4: populate `extradata` with the avcC box body.
// When driving from Annex B: leave extradata empty; NAL units are framed
// by 00 00 [00] 01 start codes.
let mut params = CodecParameters::video(CodecId::new("h264"));
// params.extradata = avcc_box_bytes;

let mut dec = codecs.make_decoder(&params)?;
let pkt = Packet::new(0, TimeBase::new(1, 90_000), annex_b_or_sample_bytes);
dec.send_packet(&pkt)?;
while let Ok(Frame::Video(vf)) = dec.receive_frame() {
    // vf.format == PixelFormat::Yuv420P; vf.planes[0] is Y, [1] is Cb, [2] is Cr.
}
# Ok::<(), oxideav_core::Error>(())
```

## What is decoded today

### Bitstream framing and parameter sets

- NAL unit splitting for both Annex B (`00 00 [00] 01` start codes) and
  length-prefixed AVCC (used inside MP4 `mdat`).
- Emulation-prevention byte stripping (§7.4.1.1).
- `AVCDecoderConfigurationRecord` parsing for MP4 `avcC` boxes.
- Sequence Parameter Set (§7.3.2.1.1) — profile, level, chroma format,
  bit depth, frame size, picture order count type, frame cropping, VUI
  presence flag.
- Picture Parameter Set (§7.3.2.2) — entropy coding mode, slice group
  map, reference index defaults, weighted prediction, deblocking
  control, 8×8 transform flag.
- Slice header (§7.3.3) — slice type, frame number, POC, reference list
  modification syntax (parsed and skipped), prediction weight table
  (parsed and skipped when present), decoded reference picture marking.

### I-slice pixel reconstruction

- CAVLC entropy coding (§9.2): `coeff_token` with `nC` neighbour
  prediction, level + run decoding, zig-zag to raster mapping, 4×4
  blocks + Intra16×16 DC + chroma DC/AC.
- CABAC entropy coding (§9.3) for the full I-slice set — I_16×16, I_NxN
  and I_PCM (with the byte-align + arithmetic-engine re-init mandated by
  §9.3.1.2). Residual decode is complete for intra blocks.
- Intra prediction: Intra4×4 (all nine modes), Intra16×16 (Vertical /
  Horizontal / DC / Plane), chroma 8×8 intra.
- Inverse transforms: 4×4 integer IDCT, 4×4 Hadamard for Intra16×16 DC,
  2×2 Hadamard for chroma DC, spec-accurate dequantisation tables.
- Spec-accurate in-loop deblocking (§8.7) on luma + both chroma planes:
  per-4-pixel-edge `bS` derivation per §8.7.2 (MB-edge intra → 4,
  internal intra → 3, non-zero residual → 2, MV/ref mismatch → 1);
  §8.7.2.2 normal + strong luma filters, §8.7.2.3 chroma filter with
  `chroma_qp_index_offset` applied through the Table 8-15 QP map and
  the `tC = tC0 + 1` chroma bump. Internal edges are transform-size
  aware (§7.3.5.1 `transform_size_8x8_flag = 1` → only the mid-MB edge
  at offset 8 is filtered). Per-slice `slice_alpha_c0_offset_div2` and
  `slice_beta_offset_div2` are honoured. The §8.7.1.1 MBAFF
  extra-edge pass is wired as a sibling entry point
  (`deblock::deblock_picture_mbaff`) — see the MBAFF section below.

### MBAFF I/P/B CAVLC (§7.3.4, §6.4.9.4, §8.7.1.1)

- `frame_mbs_only_flag = 0` AND `mb_adaptive_frame_field_flag = 1`
  pictures are accepted for **CAVLC I, P, and B slices in 4:2:0 /
  4:2:2 / 4:4:4 at 8-bit**. MBAFF + CABAC and MBAFF + 10-bit still
  return `Error::Unsupported`.
- §7.3.4 MB-pair decode loop reads `mb_field_decoding_flag` once per
  pair; per-MB [`Picture::luma_off`] / [`Picture::chroma_off`] and
  [`Picture::luma_row_stride_for`] honour the field-interleaved sample
  layout. P/B dispatch scales `first_mb_in_slice` by 2 and emits
  `mb_field_decoding_flag` at every pair's first coded MB; inside an
  `mb_skip_run` the flag is inferred from the left-neighbour pair.
- §6.4.9.4 same-polarity "above" neighbour lookup for field-coded MBs
  skips the sibling MB of the current pair — the `mb_above_neighbour`
  helper points at `mb_y - 2` rather than `mb_y - 1` so CAVLC `nC`
  prediction, intra-4×4 mode prediction, intra prediction sample
  fetches, and MV prediction (§8.4.1.1) all resolve to the correct
  previous-pair MB.
- §8.7.1.1 MBAFF deblock pass is wired
  (`deblock::deblock_picture_mbaff`). The driver walks MB pairs and
  keys the edge schedule on each pair's `mb_field_decoding_flag`:
  field-coded pairs step vertical edges at `row_step = 2` through
  the interleaved plane layout and skip the inside-pair top/bottom
  boundary. Decoded luma matches ffmpeg bit-exact (100% ±0 LSB) on
  the in-tree `mbaff_128x128` fixture; `mbaff_422_128x128` and
  `mbaff_444_128x128` are bit-exact on luma and within ±4 LSB on
  chroma.
- **Known gaps**: mixed-mode MBAFF (some pairs frame-coded, some
  field-coded) can decode successfully on the sample-read path but
  §6.4.9.4 neighbour lookups that span a frame↔field pair boundary
  still use the "uniform-pair-mode" fallback; a Table 6-5 lookup
  table would make cross-mode neighbour resolution spec-accurate.
  MBAFF + CABAC is deferred.

### CAVLC B-slice decode (§8.4 + §8.2.4.2.3)

- Macroblock types (§7.4.5 Table 7-14): `B_Direct_16x16`, `B_L0_16x16`,
  `B_L1_16x16`, `B_Bi_16x16`, and the 16×8 / 8×16 pair variants —
  L0/L0, L1/L1, L0/L1, L1/L0, L0/Bi, L1/Bi, Bi/L0, Bi/L1, Bi/Bi.
- `B_8x8` sub-macroblock types (§7.4.5.2 Table 7-17 B entries):
  `B_Direct_8x8` and `B_{L0,L1,Bi}_{8x8,8x4,4x8,4x4}`.
- `B_Skip` handling driven by `mb_skip_run` before each coded MB —
  MVs derived via spatial or temporal direct depending on
  `direct_spatial_mv_pred_flag`.
- **Spatial direct** MV prediction for `B_Direct_*` (§8.4.1.2.2):
  per-list `ref_idx` = minimum non-negative over the A/B/C neighbours,
  median MV predictor over neighbours matching that `ref_idx`.
- **Temporal direct** MV prediction (§8.4.1.2.3, the
  `direct_spatial_mv_pred_flag = 0` path): for each 4×4 block the
  colocated block in `RefPicList1[0]` is located, its MV + the POC of
  its reference picture are read from the decoded picture's per-MB
  snapshot, and `DistScaleFactor` (POC arithmetic) produces
  `mvL0 = (DSF·mvCol + 128) >> 8` and `mvL1 = mvL0 − mvCol`.
  Intra-colocated blocks use zero MVs; long-term references in either
  slot short-circuit the scaler to identity.
- Default **bi-prediction** averages the two list predictions as
  `(pred_L0 + pred_L1 + 1) >> 1` (§8.4.2.3.1). **Explicit weighted
  bi-prediction** (§8.4.2.3.2) uses the slice's `pred_weight_table` for
  both lists when `weighted_bipred_idc == 1`. **Implicit weighted
  bi-prediction** (§8.4.2.3.3) derives per-block `(w0, w1)` from POC
  distances via `DistScaleFactor` when `weighted_bipred_idc == 2`;
  long-term references or an out-of-range `DistScaleFactor` fall back
  to the default unweighted average.
- **RefPicList0 / RefPicList1** built per §8.2.4.2.3: short-term
  entries split by POC relative to `CurrPicOrderCnt` (past-desc +
  future-asc for list 0; future-asc + past-desc for list 1), then
  long-term ascending by `LongTermFrameIdx`.
- **POC output reordering** — on the first B-slice the DPB's
  pending-output queue is sized to `max_num_ref_frames` so a
  `I B B P …` decode-order stream emerges as `I P B B …` in
  presentation order. The VUI's
  `bitstream_restriction.max_num_reorder_frames` is not yet parsed.

### CAVLC baseline P-slice decode (§8.4)

- Macroblock types (§7.4.5 Table 7-13): P_L0_16×16, P_L0_16×8,
  P_L0_8×16, P_8×8 and P_8×8ref0. Intra-in-P macroblocks dispatch back
  through the I-slice path.
- Sub-macroblock types for P_8×8 (Table 7-17, P entries): 8×8, 8×4,
  4×8, 4×4.
- P_Skip handling driven by `mb_skip_run` ue(v) before each coded
  macroblock; MV derivation per §8.4.1.1.
- Motion-vector prediction — median-of-ABC with the partition-specific
  early-exit rules from §8.4.1.3 (P16×8 top/bottom, P8×16 left/right);
  falls back to D when C is unavailable.
- Luma motion compensation: integer-pel, 6-tap half-pel
  (`(1, -5, 20, 20, -5, 1) / 32`) and bilinear quarter-pel (§8.4.2.2.1).
- Chroma motion compensation: bilinear 1/8-pel (§8.4.2.2.2) with
  edge-replicated out-of-picture reads.
- Inter residual decode: CAVLC 4×4 luma blocks and 4×4 chroma AC /
  2×2 chroma DC. CBP uses the inter-column of Table 9-4(b).
- Reference picture handling: a Decoded Picture Buffer (Annex C /
  §8.2.5) holding up to `sps.max_num_ref_frames` reconstructed
  reference frames. Each P-slice builds a fresh `RefPicList0`
  (§8.2.4.2.1) and `ref_idx_l0 > 0` indexes into it. Picture order
  count derivation covers `pic_order_cnt_type == 0` (LSB + MSB
  tracking, §8.2.1.1) and `pic_order_cnt_type == 2`
  (`2·frame_num` for references, `2·frame_num - 1` otherwise —
  §8.2.1.3). Reference marking: sliding-window default (§8.2.5.3)
  plus explicit MMCO operations 1 (mark short-term unused),
  2 (mark long-term unused), 3 (assign long-term index),
  4 (`MaxLongTermFrameIdx`), 5 (mark all unused — IDR-like),
  and 6 (tag current picture as long-term) — §8.2.5.4. Reference
  picture list modification (RPLM, §7.3.3.1 / §8.2.4.3) is parsed
  into the slice header and applied over the default-built
  `RefPicList0` / `RefPicList1`: idc=0 (short-term subtract),
  idc=1 (short-term add), and idc=2 (long-term select) all run
  the §8.2.4.3.1 in-place reorder + truncate rule.

## What is encoded today

A **minimal Baseline Profile encoder** — see
[`encoder::H264Encoder`](src/encoder.rs). Each input frame is written as
a self-contained Annex B packet:

```text
IDR:     [start code] SPS  [start code] PPS  [start code] IDR slice
P-frame: [start code] SPS  [start code] PPS  [start code] non-IDR slice
```

Fixed encoder policy (no rate control, no tuning):

- **Baseline Profile** (`profile_idc = 66`), level 3.0.
- **IDR + CAVLC P-slice cadence** — every frame is an IDR by default
  (`p_slice_interval = 0`). Setting `H264EncoderOptions::p_slice_interval
  = N` emits one IDR every `N` frames and CAVLC P-slices in between.
- **CAVLC** entropy coding (`entropy_coding_mode_flag = 0`).
- **Single slice per picture** (`num_slice_groups_minus1 = 0`).
- **4:2:0** chroma (`chroma_format_idc = 1`), **8-bit** luma + chroma.
- **I-macroblocks**: **Intra_16×16 with mode decision** across all four
  Table 8-4 modes (Vertical / Horizontal / DC / Plane), picked
  per-MB by 16×16 SAD against source. Chroma runs a parallel SAD pick
  over Table 8-5 (DC / Horizontal / Vertical / Plane). At picture
  edges where the required neighbour is unavailable, the predictor
  falls back to DC per §8.3.3.
- **P-macroblocks** (opt-in): `P_L0_16×16` with a single 16×16 partition
  and `ref_idx = 0` (single-slot L0 referencing the previous
  reconstructed frame); ±16 integer-pel motion estimation via SAD,
  followed by a **3×3 half-pel refinement** that evaluates the 8
  half-pel neighbours of the integer best through `luma_mc_plane`
  (the §8.4.2.2.1 6-tap / bilinear filter); MVD = mv − §8.4.1.3
  median predictor; inter CBP via me(v); `P_Skip` when the MV
  predictor matches and the quantised residual is all-zero (§8.4.1.1
  `P_Skip` derivation).
- **Fixed QP**, configurable via `H264EncoderOptions::qp`. Default 26.
- **Deblocking filter disabled** on emit
  (`disable_deblocking_filter_idc = 1`).
- **Annex B** framing with 4-byte start codes. An
  `AVCDecoderConfigurationRecord` blob is exposed in
  `output_params().extradata` for MP4 wrapping.

Round-trip PSNR (against this crate's decoder):

- **≥ 44 dB** on a 64×48 gradient IDR at QP 22
  (`tests/encode_roundtrip.rs`).
- **≥ 30 dB** on an IDR + 2 P-frame low-motion gradient at QP 26
  (`tests/encode_pframe_roundtrip.rs`).
- **≥ 30 dB** on a 10-frame 64×64 synthetic-testsrc clip (1 IDR + 9
  P-slices) at QP 26 — average ≈ 49 dB
  (`tests/encode_10frame_testsrc.rs`).

The Annex B stream produced by the encoder is also self-decodable
through the concatenated-NAL entry point: `tests/encode_ffmpeg_decode.rs`
feeds a multi-frame Annex B blob back through the decoder's NAL
splitter without external ffmpeg. Validating against `ffmpeg -i
out.264 -c:v copy out.mp4` is still a useful acceptance check when
ffmpeg is available locally:

```shell
# Dump one frame of Annex B from the integration test harness, then
# re-wrap it into an MP4. ffmpeg should accept the stream without
# complaint (no `-err_detect strict` needed).
cargo test --release --test encode_ffmpeg_decode -- --nocapture
ffmpeg -i /tmp/oxideav_h264_encoder_test.264 -c:v copy /tmp/out.mp4
```

```rust
use oxideav_codec::Encoder;
use oxideav_core::{CodecId, Frame, PixelFormat, TimeBase, VideoFrame};
use oxideav_core::frame::VideoPlane;
use oxideav_h264::encoder::{H264Encoder, H264EncoderOptions};

let mut enc = H264Encoder::new(
    CodecId::new("h264"),
    64,
    48,
    H264EncoderOptions { qp: 22, ..Default::default() },
)?;
let frame = /* YUV420P VideoFrame of the right size */;
# let frame = VideoFrame {
#     format: PixelFormat::Yuv420P, width: 64, height: 48, pts: Some(0),
#     time_base: TimeBase::new(1, 30),
#     planes: vec![
#         VideoPlane { stride: 64, data: vec![128; 64*48] },
#         VideoPlane { stride: 32, data: vec![128; 32*24] },
#         VideoPlane { stride: 32, data: vec![128; 32*24] },
#     ],
# };
enc.send_frame(&Frame::Video(frame))?;
enc.flush()?;
let pkt = enc.receive_packet()?;   // Annex B: SPS+PPS+IDR
# Ok::<(), oxideav_core::Error>(())
```

## High Profile 8×8 transform — CAVLC and CABAC wired

The 8×8 transform path is fully wired on **both** the CAVLC and CABAC
entropy paths, for intra and P-slice inter macroblocks.

Shared §8 math — bit-exact against ffmpeg's libavcodec reference:

- 8×8 inverse integer transform (`transform::idct_8x8`) and its 6-class
  dequantisation (`transform::dequantize_8x8`) over the default flat-16
  scaling list — §8.5.13.
- Nine Intra_8×8 prediction modes with reference-sample low-pass
  filtering per §8.3.2.2.2 (`intra_pred::predict_intra_8x8`) — §8.3.2.
- `transform_size_8x8_flag` at both spec positions: before the
  intra-mode syntax for I_NxN (§7.3.5.1 intra branch) and after
  `coded_block_pattern` / before `mb_qp_delta` for the P-slice inter
  branch, with the §7.3.5.1 conditional (`cbp_luma > 0` and no sub-8×8
  motion partitions on P_8x8).

CAVLC 8×8 residual:

- Per-sub-block CAVLC residual (`cavlc::decode_residual_8x8_sub`) —
  four 16-coefficient blocks spliced via `cavlc::ZIGZAG_8X8_CAVLC`
  (§7.3.5.3.2, FFmpeg's `zigzag_scan8x8_cavlc` untransposed).
- Per-sub-block `non_zero_count_cache` neighbour nC prediction +
  post-decode `nnz[0] += nnz[1] + nnz[8] + nnz[9]` aggregation so
  neighbour MBs see a single representative count for the whole 8×8
  block (mirrors FFmpeg's `scan8[]`-indexed cache). Inter 8×8 reuses
  the same cache layout via `p_mb::predict_inter_nc_luma`.

CABAC 8×8 residual (§9.3.3.1.1.9 / §9.3.3.1.1.10 — new):

- `transform_size_8x8_flag` binarised FL(1) at ctxIdxOffset 399, with
  `ctxIdxInc = condTermFlagA + condTermFlagB` (spec positions 399 / 400
  / 401). Neighbour derivation uses each MB's recorded `transform_8x8`.
- Luma 8×8 residual decodes as a single 64-coefficient CABAC block
  (`ctxBlockCat = 5`): `coded_block_flag` at ctxIdxOffset **1012**,
  `significant_coeff_flag` at **402**, `last_significant_coeff_flag` at
  **417**, `coeff_abs_level_minus1` at **426**. Per-position ctxIdxInc
  for sig/last uses the 63-entry map wired in
  `cabac::residual::LUMA8X8_SIG_LAST_CTX_INC` (§9.3.3.1.3). The
  UEGk(k=0, uCoff=14) abs-level uses the same `num_eq1` / `num_gt1`
  counters as the 4×4 path.
- Custom scaling-list matrices are parsed but not applied (only
  flat-16 is used for 8×8 dequant).

## Not supported

Decoder surfaces `Error::Unsupported` (or `Error::InvalidData` when the
bitstream claims a feature that isn't wired); encoder outright refuses:

- **Decoder**: CABAC P- and B-slices are wired for the CAVLC-equivalent
  coverage (P_Skip / all P Table 7-13 inter partitions + sub-partitions /
  intra-in-P; B_Skip / all B Table 7-14 16×16, 16×8, 8×16 variants +
  B_8×8 with every Table 7-17 sub-partition / intra-in-B), including
  `transform_size_8x8_flag = 1` on both intra and inter CABAC paths.
  Weighted prediction on the CABAC P path is parsed but weights are
  not applied; encoders should disable weighted P for CABAC bitstreams
  fed into this decoder, or tolerate unweighted MC output.
- **Encoder**: B-slices, CABAC, Intra_4×4, Intra_8×8, rate control,
  adaptive QP, in-loop deblocking on emit (the encoder disables §8.7
  via `disable_deblocking_filter_idc = 1` in the slice header, so its
  local reconstruction matches the decoder's bit-exactly without
  running a deblock pass). P-slices: only `P_L0_16×16` / `P_Skip` are
  emitted; no 16×8 / 8×16 / 8×8 partitions, no quarter-pel motion
  refinement (half-pel is wired; quarter-pel refinement runs through
  the same `luma_mc_plane` filter but is not yet tied into a search),
  no multi-reference, no weighted prediction. **Intra_16×16 mode
  decision** covers all four Table 8-4 modes (Vertical / Horizontal /
  DC / Plane) picked by SAD; Intra_4×4 and Intra_8×8 remain out of
  scope.
- `pic_order_cnt_type == 1` — only types 0 and 2 are implemented.
- PAFF CABAC, PAFF chroma > 4:2:0, PAFF 10-bit, and PAFF + MBAFF mixed
  within a coded video sequence remain out of scope.
  **PAFF I / P / B-slice CAVLC** at 4:2:0 / 8-bit IS supported (§7.3.3
  `field_pic_flag = 1`): top and bottom fields decode into half-height
  `Picture`s allocated via `Picture::new_field`, POC derivation reuses
  §8.2.1.1, and each field is emitted as its own `VideoFrame`. The
  encoder exposes a `H264EncoderOptions::paff_field` flag; the first
  emitted PAFF frame is an IDR I-field and every subsequent PAFF frame
  is an all-skip P-field (`mb_skip_run = TotalMbs`, no coded MB,
  §8.4.1.1 `P_Skip` zero-MV) so round-trip tests can drive the PAFF P
  path without shipping a full inter-prediction writer.
- **MBAFF I/P/B CAVLC** is supported for 4:2:0 / 4:2:2 / 4:4:4 8-bit
  (§7.3.4 MB-pair loop + §6.4.9.4 field-stride neighbour lookups +
  §8.7.1.1 MBAFF deblock). MBAFF + CABAC and MBAFF + 10-bit still
  return `Error::Unsupported`.
- Inter 8×8 transform on B-slice MBs — CAVLC P-slice inter 8×8 is wired
  (see above); B-slice inter 8×8 is not yet.
- 4:2:2 CAVLC I- and P-slice decode IS supported (§6.4.1
  ChromaArrayType == 2) — half-width / full-height chroma planes, 2×4
  chroma DC Hadamard (§8.5.11.2 with the `QP'_C,DC = QP'_C + 3` dequant
  offset per Amd 2 equations 8-328..8-330), 8 chroma AC blocks per
  plane arranged in a 2×4 grid, 8×16 chroma intra prediction (§8.3.4),
  and the extra internal chroma horizontal / vertical deblock edges
  (§8.7). The P-slice path reuses the luma 6-tap + bilinear MC and
  adds §8.4.2.2.1 ChromaArrayType == 2 vertical-MV scaling (divide by
  4 instead of 8, since chroma height matches luma height) for chroma
  MC. B-slices and CABAC on 4:2:2 remain out of scope.
- 4:4:4 CAVLC I / P / B decode IS supported (§6.4.1 ChromaArrayType == 3)
  — chroma planes share luma dimensions, reuse the luma Intra_4×4 /
  Intra_16×16 predictors, and decode three back-to-back luma-style
  residual streams per MB. Inter MBs (P / B) run the luma 6-tap +
  bilinear quarter-pel MC on every plane (§8.4.2.2 Table 8-9 entry for
  ChromaArrayType = 3); the 4:2:0 chroma 1/8-pel bilinear is skipped.
  Inter CBP is 4 bits (luma-only), mapped via the FFmpeg
  `golomb_to_inter_cbp_gray` table. The I path is bit-exact against
  ffmpeg on `iframe_yuv444_64x64`; the P / B paths match ≥ 90 % / ≥ 80 %
  of samples on `yuv444_p_64x64` / `yuv444_b_64x64` — the residual gap
  mirrors a pre-existing limitation of the P/B residual path that hits
  the equivalent 4:2:0 testsrc fixture identically (solid-colour 4:4:4
  P decodes 100 % matched). 8×8 transform on 4:4:4 inter is out of
  scope. CABAC on 4:4:4 is supported for I-slices bit-exact and for
  P-slice inter partitions (P16×16 / P16×8 / P8×16 + intra-in-P + P_Skip)
  at ≥ 90 % sample match on the `yuv444_cabac_p_64x64` testsrc fixture.
  The §9.3.3.1.1.9 Table 9-42 extended ctxBlockCat banks (cats 6..=13 at
  spec ctxIdx 460..=483, 484..=674, 708..=775, 952..=1011, 1016..=1023)
  are transcribed from FFmpeg's `libavcodec/h264_cabac.c`
  `cabac_context_init_I / PB` arrays into `src/cabac/tables.rs`; the
  residual helpers take a `ResidualPlane` selector that routes Y/Cb/Cr
  to their own banks. CABAC B-slice inter (Direct / L0 / L1 / Bi 16×16,
  TwoPart 16×8 / 8×16, B_8×8) is still staged behind a specific
  `Error::Unsupported` message — intra-in-B and B_Skip are wired but a
  first cut at B inter exhibits context-state drift during Cb/Cr
  residual decode at MB 7+ of the testsrc fixture; the test is
  `#[ignore]`d with that note and awaits an MVD/ref_idx ctxIdxInc audit.
- Bit depth above 14 or asymmetric luma/chroma bit depths. 12-bit /
  14-bit are supported for **CAVLC I-slices in 4:2:0 only** (see the
  12/14-bit section below). 10-bit (High 10) supports CAVLC I / P / B
  + CABAC I / P / B at 4:2:0, plus CAVLC I at 4:2:2 / 4:4:4,
  Intra_8×8 (`transform_size_8x8_flag = 1`), and I_PCM. Luma must
  equal chroma bit depth.
- **Separate colour planes** (`separate_colour_plane_flag = 1`,
  §7.4.2.1.1) decode end-to-end for **CAVLC I-slices at 8-bit
  progressive**. Each slice decodes as an independent monochrome
  picture keyed by `colour_plane_id ∈ {0, 1, 2}`; the decoder
  accumulates the per-plane buffers by `(frame_num, colour_plane_id)`
  and emits a merged `Yuv444P` `VideoFrame` once all three planes have
  arrived. Sep-plane P/B slices, CABAC entropy, 10-bit samples, and
  PAFF/MBAFF sep-plane still return `Error::Unsupported`. Coverage:
  `tests/separate_colour_plane.rs` (SPS parse, per-plane slice header
  parse, 1×1-MB + 2×2-MB end-to-end decode).
- **Primary SI / SP slices** (§7.3.5 / §7.4.5 Table 7-12 / Table 7-13
  SP entries / §8.6) are **supported** on the CAVLC 4:2:0 / 8-bit
  path when `slice_qs_delta == slice_qp_delta` (the §8.6.1 / §8.6.2
  dequant+requant pair is the identity in that case). Secondary SI /
  SP (the `sp_for_switch_flag = 1` or `QS != QP` switching targets)
  return `Error::Unsupported`; CABAC SI / SP and high-bit-depth /
  4:2:2 / 4:4:4 SI / SP are rejected at the slice-entry gate.

## 10-bit (High 10) decode

The decoder handles 10-bit content end-to-end across **CAVLC I / P / B
slices**, **CABAC I / P / B slices**, **4:2:0 / 4:2:2 / 4:4:4** chroma
formats (4:2:2 and 4:4:4 currently at I-slice only), **Intra_8×8**
(`transform_size_8x8_flag = 1`), and **I_PCM**, emitting
`PixelFormat::Yuv420P10Le` / `Yuv422P10Le` / `Yuv444P10Le` frames (u16
samples packed little-endian). Bit-exact against ffmpeg on every
`iframe_10bit_*` / `pframe_10bit_*` / `10bit_*` / `b_10bit_*` fixture
shipped under `tests/fixtures/`:

- `iframe_10bit_64x64` (CAVLC I 4:2:0): 99% within ±4 LSB.
- `pframe_10bit_64x64` (CAVLC P 4:2:0): 100% bit-exact across three
  consecutive P-slices.
- `10bit_cabac_i_64x64` (CABAC I 4:2:0): 100% bit-exact.
- `10bit_b_64x64` (CAVLC B 4:2:0): 100% bit-exact across 6 frames
  (I / P / B mix with `bf=2`, `no-weightb`).
- `10bit_cabac_p_64x64` (CABAC P 4:2:0): 100% bit-exact.
- `10bit_cabac_b_64x64` (CABAC B 4:2:0): 100% bit-exact.
- `iframe_10bit_422_64x64` (CAVLC I 4:2:2): 100% bit-exact.
- `iframe_10bit_444_64x64` (CAVLC I 4:4:4): 100% bit-exact.
- `iframe_10bit_8x8_64x64` (CAVLC I with `transform_size_8x8_flag = 1`):
  100% bit-exact.
- Hand-crafted 10-bit I_PCM single-MB fixture: 100% bit-exact.

The high-bit-depth path runs alongside — not in place of — the 8-bit
pipeline:

- [`picture::Picture`] gains companion u16 planes (`y16` / `cb16` /
  `cr16`) and `bit_depth_y` / `bit_depth_c` fields. At 8 bits only the
  u8 planes are allocated; above 8 only the u16 planes are populated.
- Intra prediction ([`intra_pred_hi`]) clips to `(1 << bit_depth) - 1`
  and uses `1 << (bit_depth - 1)` as the DC no-neighbour fallback
  (§8.3.1.2.3 / §8.3.3.3 / §8.3.4.2). Includes full Intra_8×8 mode set.
- The residual pipeline ([`mb_hi`] / [`p_mb_hi`] / [`b_mb_hi`]) adds
  `QpBdOffsetY` to the decoded `QpY` before dispatching through the
  i64-widened `dequantize_*_ext` variants in [`transform`], and wraps
  `mb_qp_delta` modulo `52 + QpBdOffsetY` per §7.4.5.
- Chroma dispatch in [`mb_hi`] routes `chroma_format_idc == 2` into
  [`mb_hi_422`] (2×4 chroma DC Hadamard on u16 + 8 AC blocks per plane)
  and `== 3` into [`mb_hi_444`] (three luma-style residual streams at
  u16 with per-plane QP + scaling list).
- CABAC at high bit depth ([`cabac::mb_hi`] / [`cabac::p_mb_hi`] /
  [`cabac::b_mb_hi`]) reuses the 8-bit entropy layer (contexts,
  binarisation, residual decode — all bit-depth agnostic) and
  reconstructs into u16 planes via the shared `*_ext` dequant chain.
- P-slice motion compensation ([`motion::luma_mc_hi`] /
  [`motion::chroma_mc_hi`]) operates on the u16 planes with the same
  6-tap half-pel + bilinear quarter-pel luma filter and bilinear 1/8-pel
  chroma filter as the 8-bit path; the final Clip1 uses the per-plane
  bit-depth max. Explicit and implicit weighted bipred both wire on the
  10-bit B path.
- In-loop deblocking on the 10-bit path ([`deblock_hi`]) reads / writes
  the u16 planes, scales `α` / `β` / `tC0` by `1 << (BitDepth - 8)` per
  §8.7.2.1, and clips to the widened sample range. The §8.7.2 bS
  derivation and edge schedule are shared with the 8-bit deblocker via
  `deblock::derive_bs_for_edge`.
- I_PCM at 10-bit reads `bit_depth_luma` / `bit_depth_chroma` raw bits
  per sample post byte-alignment, matching the 8-bit path parameterised
  on depth.
- [`picture::Picture::to_video_frame`] packs the u16 planes as
  little-endian bytes — `Yuv420P10Le` at 4:2:0, `Yuv422P10Le` at 4:2:2,
  `Yuv444P10Le` at 4:4:4.

Out of scope at 10-bit (return `Error::Unsupported`): CABAC at 4:2:2 /
4:4:4; P / B at 4:2:2 / 4:4:4 (only I is wired for those chroma formats
at 10-bit); MBAFF / PAFF 10-bit.

## 12-bit + 14-bit (High 4:2:2 / High 4:4:4 Predictive) CAVLC I

The slice-entry gate accepts `bit_depth_luma_minus8 ∈ {0, 2, 4, 6}`
(8 / 10 / 12 / 14-bit). 12-bit and 14-bit decode reuses the entire
10-bit high-bit-depth pipeline — u16 `Picture` planes, `bit_depth`-
parameterised intra prediction, and the i64-widened `*_ext` dequant
helpers in [`transform`] — the extensions are:

- `QpBdOffsetY = 6 * bit_depth_luma_minus8` (24 at 12-bit, 36 at
  14-bit). QpY' = `QpY + QpBdOffsetY` reaches 75 / 87 respectively,
  which fits comfortably inside the i64-widened dequant shifts
  (`qp6 = QpY' / 6` is at most 14, well below any overflow bound).
- `mb_qp_delta` wraps modulo `52 + QpBdOffsetY` per §7.4.5 — the same
  wrapping the 10-bit path already does (just with a larger modulus).
- Chroma QP Table 7-2 (`chroma_qp_hi` in [`transform`]) is an identity
  function outside `0..=51`, so the larger `qpi = QpY +
  chroma_qp_index_offset` range passes through untouched before
  `+ QpBdOffsetC`.
- Intra prediction clip bound is `(1 << bit_depth) - 1` = 4095 at
  12-bit / 16383 at 14-bit; DC no-neighbour fallback is
  `1 << (bit_depth - 1)` = 2048 / 8192 (all already parameterised in
  [`intra_pred_hi`]).
- Emit: 12-bit → `PixelFormat::Yuv420P12Le`; 14-bit reuses
  `Yuv420P10Le` as a 16-bit-container fallback because oxideav-core
  doesn't carry a `Yuv420P14Le` variant. The u16 words still hold
  full 14-bit samples (`0..=16383`).

Out of scope at 12/14-bit (return `Error::Unsupported`):

- CABAC at any slice type.
- P / B slices at 12 or 14-bit — only I-slices are wired.
- 4:2:2 or 4:4:4 chroma at 12 / 14-bit (4:2:0 only for now).
- Asymmetric luma / chroma bit depths
  (`bit_depth_luma_minus8 != bit_depth_chroma_minus8`).

A synthetic single-MB fixture (`tests/decode_iframe_12_14bit.rs`)
hand-crafts a 16×16 IDR at `bit_depth_luma_minus8 = 4` and `= 6` to
exercise both the I_NxN DC-fallback path and the I_16x16 Hadamard DC
dequant chain without needing an external 12/14-bit libx264 build
(most distro packages ship a 10-bit-only libx264).

## Codec id

Registered under `CodecId::new("h264")` for both decoder (`h264_sw`)
and encoder (`h264_baseline_sw`). Call `oxideav_h264::register(&mut
reg)` to wire both into a `CodecRegistry`. The decoder and encoder
both operate on `PixelFormat::Yuv420P`.

## Test fixtures

Integration tests under `tests/` skip (printing a note) when their
fixture is missing so CI without ffmpeg still passes. Regenerate the
fixtures locally with:

```shell
# 64×64 baseline, every-frame I — exercises CAVLC + CABAC I-slice paths.
ffmpeg -y -f lavfi -i "testsrc=size=64x64:rate=24:duration=0.1" \
    -pix_fmt yuv420p -c:v libx264 -profile:v baseline -preset:v slow \
    -g 1 -bf 0 -refs 1 /tmp/h264_iframe.mp4
ffmpeg -y -i /tmp/h264_iframe.mp4 -c:v copy -bsf:v h264_mp4toannexb \
    -f h264 /tmp/h264.es
ffmpeg -y -i /tmp/h264_iframe.mp4 -f rawvideo -pix_fmt yuv420p \
    /tmp/h264_iframe.yuv

# Solid-grey I-slice (acceptance bar for Intra16×16 + deblocking).
ffmpeg -y -f lavfi -i "color=c=gray:size=64x64:rate=24:duration=0.04" \
    -pix_fmt yuv420p -c:v libx264 -profile:v baseline -preset:v slow \
    -g 1 -bf 0 -refs 1 /tmp/h264_solid.mp4
ffmpeg -y -i /tmp/h264_solid.mp4 -c:v copy -bsf:v h264_mp4toannexb \
    -f h264 /tmp/h264_solid.es
ffmpeg -y -i /tmp/h264_solid.mp4 -f rawvideo -pix_fmt yuv420p \
    /tmp/h264_solid.yuv

# CAVLC P-slice acceptance clip — six frames (1 I + 5 P), mostly P_Skip.
ffmpeg -y -f lavfi -i "color=c=gray:size=64x64:rate=24:duration=0.25,hue=s=1" \
    -pix_fmt yuv420p -c:v libx264 -profile:v baseline -preset:v slow \
    -g 6 -bf 0 -refs 1 -coder 0 /tmp/h264_pslice_simple.mp4
ffmpeg -y -i /tmp/h264_pslice_simple.mp4 -c:v copy -bsf:v h264_mp4toannexb \
    -f h264 /tmp/h264_pslice_simple.es
ffmpeg -y -i /tmp/h264_pslice_simple.mp4 -f rawvideo -pix_fmt yuv420p \
    /tmp/h264_pslice_simple.yuv

# CAVLC B-slice acceptance clip — 12 frames I/B/B/P/B/B/... solid grey,
# bundled in tests/fixtures/ so this test works without ffmpeg.
ffmpeg -y -f lavfi -i "color=c=gray:size=64x64:rate=24:duration=0.5" \
    -pix_fmt yuv420p -c:v libx264 -profile:v main -g 6 -bf 2 \
    -x264opts no-scenecut:no-weightb:weightp=0:no-mbtree -coder 0 \
    /tmp/oxideav_bslice.mp4
ffmpeg -y -i /tmp/oxideav_bslice.mp4 -c:v copy -bsf:v h264_mp4toannexb \
    -f h264 tests/fixtures/bslice_64x64.es
ffmpeg -y -i /tmp/oxideav_bslice.mp4 -f rawvideo -pix_fmt yuv420p \
    tests/fixtures/bslice_64x64.yuv

# CABAC B-slice acceptance clip — 6 frames (IDR + 2 P + 3 B),
# Main profile with CABAC, bframes=2, no weighted bipred.
ffmpeg -y -f lavfi -i testsrc=duration=1:size=64x64:rate=12 \
    -vframes 6 -pix_fmt yuv420p -c:v libx264 -profile:v main \
    -preset ultrafast -coder 1 -bf 2 \
    -x264opts no-scenecut:no-weightb:weightp=0:no-mbtree:bframes=2:keyint=6:min-keyint=6 \
    /tmp/cabac_b.mp4
ffmpeg -y -i /tmp/cabac_b.mp4 -c:v copy -bsf:v h264_mp4toannexb \
    -f h264 tests/fixtures/cabac_b_64x64.es
ffmpeg -y -i /tmp/cabac_b.mp4 -f rawvideo -pix_fmt yuv420p \
    tests/fixtures/cabac_b_64x64.yuv

# CAVLC B-slice implicit weighted-bipred clip (§8.4.2.3.3) — a 64×64
# `smptebars` pan at QP 20 so the decoder sees real bi-predicted MBs
# combined via POC-derived weights (PPS sets `weighted_bipred_idc = 2`).
ffmpeg -y -f lavfi -i "smptebars=size=128x64:rate=24:duration=0.5" \
    -vf "crop=64:64:t*16:0" -pix_fmt yuv420p -c:v libx264 -profile:v main \
    -qp 20 -preset slow -g 6 -bf 3 -b_strategy 0 -coder 0 \
    -x264opts "no-scenecut:no-mbtree:no-8x8dct:weightb:weightp=0:bframes=3:ref=3" \
    /tmp/bframe_implicit.mp4
ffmpeg -y -i /tmp/bframe_implicit.mp4 -c:v copy -bsf:v h264_mp4toannexb \
    -f h264 tests/fixtures/bframe_implicit_64x64.es
ffmpeg -y -i /tmp/bframe_implicit.mp4 -f rawvideo -pix_fmt yuv420p \
    tests/fixtures/bframe_implicit_64x64.yuv

# 10-bit (High 10) CAVLC I-slice — 64×64 yuv420p10le testsrc IDR.
ffmpeg -y -f lavfi -i testsrc=duration=1:size=64x64:rate=1 \
    -vframes 1 -pix_fmt yuv420p10le -c:v libx264 -profile:v high10 \
    -preset ultrafast /tmp/10bit.mp4
ffmpeg -y -i /tmp/10bit.mp4 -c copy -bsf:v h264_mp4toannexb -f h264 \
    tests/fixtures/iframe_10bit_64x64.es
ffmpeg -y -i /tmp/10bit.mp4 -f rawvideo -pix_fmt yuv420p10le \
    tests/fixtures/iframe_10bit_64x64.yuv

# 10-bit (High 10) CAVLC P-slice — 3-frame yuv420p10le clip (1 I + 2 P).
ffmpeg -y -f lavfi -i testsrc=duration=1:size=64x64:rate=12 -vframes 3 \
    -pix_fmt yuv420p10le -c:v libx264 -profile:v high10 -preset ultrafast \
    -x264-params "bframes=0:keyint=3:no-scenecut=1" /tmp/p_10bit.mp4
ffmpeg -y -i /tmp/p_10bit.mp4 -c copy -bsf:v h264_mp4toannexb -f h264 \
    tests/fixtures/pframe_10bit_64x64.es
ffmpeg -y -i /tmp/p_10bit.mp4 -f rawvideo -pix_fmt yuv420p10le \
    tests/fixtures/pframe_10bit_64x64.yuv

# 10-bit (High 10) CABAC I-slice — 64×64 yuv420p10le IDR with -coder 1.
ffmpeg -y -f lavfi -i testsrc=duration=1:size=64x64:rate=1 -vframes 1 \
    -profile:v high10 -preset ultrafast -coder 1 -pix_fmt yuv420p10le \
    /tmp/10bit_cabac_i.mp4
ffmpeg -y -i /tmp/10bit_cabac_i.mp4 -c copy -bsf:v h264_mp4toannexb \
    -f h264 tests/fixtures/10bit_cabac_i_64x64.es
ffmpeg -y -i /tmp/10bit_cabac_i.mp4 -f rawvideo -pix_fmt yuv420p10le \
    tests/fixtures/10bit_cabac_i_64x64.yuv

# 10-bit (High 10) CAVLC B-slice — 6-frame yuv420p10le mix (bf=2, no-weightb).
ffmpeg -y -f lavfi -i testsrc=duration=2:size=64x64:rate=6 -vframes 6 \
    -profile:v high10 -preset ultrafast -coder 0 -bf 2 -pix_fmt yuv420p10le \
    -x264-params "bframes=2:no-weightb=1" /tmp/10bit_b.mp4
ffmpeg -y -i /tmp/10bit_b.mp4 -c copy -bsf:v h264_mp4toannexb \
    -f h264 tests/fixtures/10bit_b_64x64.es
ffmpeg -y -i /tmp/10bit_b.mp4 -f rawvideo -pix_fmt yuv420p10le \
    tests/fixtures/10bit_b_64x64.yuv
```

## License

MIT — see [LICENSE](LICENSE).
