# oxideav-h264

Pure-Rust **H.264 / AVC** (ITU-T H.264 | ISO/IEC 14496-10) decoder — NAL
framing, SPS / PPS / slice-header parsing, CAVLC + CABAC I-slice pixel
reconstruction, CAVLC baseline P-slice pixel reconstruction with motion
compensation. Zero C dependencies.

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
- CABAC entropy coding (§9.3) for the common I-slice subset — I_16×16
  and I_NxN with full residual decode. I_PCM inside CABAC is not yet
  wired.
- Intra prediction: Intra4×4 (all nine modes), Intra16×16 (Vertical /
  Horizontal / DC / Plane), chroma 8×8 intra.
- Inverse transforms: 4×4 integer IDCT, 4×4 Hadamard for Intra16×16 DC,
  2×2 Hadamard for chroma DC, spec-accurate dequantisation tables.
- Optional in-loop deblocking (§8.7) honouring
  `disable_deblocking_filter_idc` and the per-slice alpha / beta
  offsets.

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
- Reference picture handling: exactly one L0 reference — the
  previously-decoded output frame. Multi-reference prediction
  (`ref_idx_l0 > 0`), reference list modification, and reference
  picture list reordering are not supported and surface as
  `Error::Unsupported` when the bitstream requests them.

## Not supported

Calls surface `Error::Unsupported` (or `Error::InvalidData` when the
bitstream claims a feature that isn't wired):

- CABAC P-slices (baseline CAVLC P works — set
  `entropy_coding_mode_flag = 0` at the encoder).
- B-slices, bi-prediction, weighted prediction.
- Multi-reference DPB / `ref_idx_l0 > 0`, reference list modification
  operations.
- Interlaced coding, MBAFF, PAFF, frame/field picture mixes.
- 8×8 transform (`transform_8x8_mode_flag = 1`).
- 4:2:2 / 4:4:4 chroma, bit depths above 8, separate colour planes.
- SI / SP slices.
- Encoder — this crate is decoder-only today.

## Codec id

Registered under `CodecId::new("h264")`. The decoder always emits
`PixelFormat::Yuv420P`.

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
```

## License

MIT — see [LICENSE](LICENSE).
