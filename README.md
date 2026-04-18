# oxideav-h264

Pure-Rust **H.264 / AVC** (ITU-T H.264 | ISO/IEC 14496-10) codec for
oxideav. Decodes CAVLC + CABAC I-slices, CAVLC + CABAC P-slices, and
CAVLC B-slices (spatial direct + explicit weighted bipred); encodes a minimal
Baseline-Profile, I-frame-only, Intra_16×16 DC_PRED output stream. The
High-Profile 8×8 transform + Intra_8×8 prediction math primitives are
unit-tested but **not** wired into the macroblock decode loop —
bitstreams that set `transform_size_8x8_flag = 1` surface as
`Error::Unsupported`. Zero C dependencies.

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
- Optional in-loop deblocking (§8.7) honouring
  `disable_deblocking_filter_idc` and the per-slice alpha / beta
  offsets.

### CAVLC B-slice decode (§8.4 + §8.2.4.2.3)

- Macroblock types (§7.4.5 Table 7-14): `B_Direct_16x16`, `B_L0_16x16`,
  `B_L1_16x16`, `B_Bi_16x16`, and the 16×8 / 8×16 pair variants —
  L0/L0, L1/L1, L0/L1, L1/L0, L0/Bi, L1/Bi, Bi/L0, Bi/L1, Bi/Bi.
- `B_8x8` sub-macroblock types (§7.4.5.2 Table 7-17 B entries):
  `B_Direct_8x8` and `B_{L0,L1,Bi}_{8x8,8x4,4x8,4x4}`.
- `B_Skip` handling driven by `mb_skip_run` before each coded MB —
  MVs derived via spatial direct (§8.4.1.2.2).
- **Spatial direct** MV prediction for `B_Direct_*` (§8.4.1.2.2):
  per-list `ref_idx` = minimum non-negative over the A/B/C neighbours,
  median MV predictor over neighbours matching that `ref_idx`.
- Default **bi-prediction** averages the two list predictions as
  `(pred_L0 + pred_L1 + 1) >> 1` (§8.4.2.3.1). **Explicit weighted
  bi-prediction** (§8.4.2.3.2) uses the slice's `pred_weight_table` for
  both lists when `weighted_bipred_idc == 1`.
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
  picture list modification (RPLM) is parsed but non-identity
  commands surface `Error::Unsupported` (planned follow-up).

## What is encoded today

A **minimal Baseline Profile, I-frame-only encoder** — see
[`encoder::H264Encoder`](src/encoder.rs). Each input frame is written as
a self-contained Annex B packet:

```text
[start code] SPS  [start code] PPS  [start code] IDR slice
```

Fixed encoder policy (no rate control, no mode decision, no tuning):

- **Baseline Profile** (`profile_idc = 66`), level 3.0.
- **I-frames only** — every frame is an IDR.
- **CAVLC** entropy coding (`entropy_coding_mode_flag = 0`).
- **Single slice per picture** (`num_slice_groups_minus1 = 0`).
- **4:2:0** chroma (`chroma_format_idc = 1`), **8-bit** luma + chroma.
- **Intra_16×16** with `Intra16x16PredMode = DC` (Table 8-4 value 2) for
  every luma macroblock; **chroma DC** (Table 8-5 value 0) for chroma.
- **Fixed QP**, configurable via `H264EncoderOptions::qp`. Default 26.
- **Deblocking filter disabled** on emit
  (`disable_deblocking_filter_idc = 1`).
- **Annex B** framing with 4-byte start codes. An
  `AVCDecoderConfigurationRecord` blob is exposed in
  `output_params().extradata` for MP4 wrapping.

Round-trip PSNR at QP 22 on a 64×48 gradient is ≥ 44 dB (tested
against both this crate's own decoder and ffmpeg's `libavcodec`).

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

## High Profile 8×8 transform — primitives only, not wired

The math primitives are implemented and unit-tested but the High-Profile
8×8 residual path is **not** wired into the macroblock decode loop:

- 8×8 inverse integer transform (`transform::idct_8x8`) and its 6-class
  dequantisation (`transform::dequantize_8x8`) over the default flat-16
  scaling list — §8.5.13.
- Nine Intra_8×8 prediction modes (§8.3.2) with reference-sample low-pass
  filtering per §8.3.2.2.2 (`intra_pred::predict_intra_8x8`).

When the PPS advertises `transform_8x8_mode_flag = 1` and a macroblock
sets `transform_size_8x8_flag = 1`, both the CAVLC and CABAC I-slice
paths surface `Error::Unsupported`. The CAVLC §7.3.5.3.2 residual
decode and the CABAC 8×8 context coding (§9.3.3.1.1.10) are not
implemented — a prior attempt hit a bit-accounting desync against real
x264 `8x8dct=1:cabac=0` output and was rolled back rather than left as
a half-finished code path in the decode loop.

## Not supported

Decoder surfaces `Error::Unsupported` (or `Error::InvalidData` when the
bitstream claims a feature that isn't wired); encoder outright refuses:

- **Decoder**: CABAC B-slices (CAVLC B works). CABAC P-slices are wired
  for the CAVLC-equivalent coverage (P_Skip, P_L0_16×16 / 16×8 / 8×16,
  P_8×8 with all four sub-partitions, intra-in-P) with two caveats that
  surface `Error::Unsupported` on this first pass:
  - `transform_size_8x8_flag = 1` is not supported (same scoping as the
    CAVLC I / P paths).
  - Weighted prediction on the CABAC P path is parsed but weights are
    not applied; encoders should disable weighted P for CABAC bitstreams
    fed into this decoder, or tolerate unweighted MC output.
  Any I-slice macroblock with `transform_size_8x8_flag = 1` (CAVLC and
  CABAC).
- **Encoder**: P / B slices, CABAC, Intra_4×4, Intra_8×8, Intra_16×16
  modes other than DC, rate control, adaptive QP, mode decision,
  deblocking.
- **B-slice temporal direct** MV prediction (§8.4.1.2.3,
  `direct_spatial_mv_pred_flag = 0`) — only spatial direct is wired.
- **Implicit weighted bi-prediction** (`weighted_bipred_idc == 2`,
  §8.4.2.3.3) — explicit bipred and P-slice weighted prediction are
  supported.
- Reference picture list modification (RPLM) — the identity form
  (flag = 0) is accepted; non-trivial modification commands surface
  `Error::Unsupported`.
- `pic_order_cnt_type == 1` — only types 0 and 2 are implemented.
- Interlaced coding, MBAFF, PAFF, frame/field picture mixes.
- Inter 8×8 transform (`transform_size_8x8_flag = 1` on P/B MBs).
- 4:2:2 / 4:4:4 chroma, bit depths above 8, separate colour planes.
- SI / SP slices.

## Codec id

Registered under `CodecId::new("h264")` for both decoder (`h264_sw`)
and encoder (`h264_baseline_i_sw`). Call `oxideav_h264::register(&mut
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
```

## License

MIT — see [LICENSE](LICENSE).
