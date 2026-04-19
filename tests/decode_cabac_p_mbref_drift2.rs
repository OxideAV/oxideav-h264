//! Regression fixtures for the two real-world `ref_idx_l0 N out of range`
//! CABAC P-slice failures that were folded into `wt/mbref-drift2`.
//!
//! 1. `p_cabac_refpad_128x128.{es,yuv}` — 3-frame mandelbrot IDR + P + P
//!    at `ref=2`. x264 writes `num_ref_idx_l0_active_minus1 = 2` in the
//!    second P-frame's slice header (want = 3) even though only 2
//!    pictures are in the DPB at that point. The previous decoder
//!    truncated the default list to the natural 2 entries, so CABAC bins
//!    that decoded to `ref_idx_l0 = 2` (legitimate per
//!    `num_ref_idx_l0_active - 1 = 2`) hit the array end. Fix duplicates
//!    entry 0 into the padding slot, mirroring FFmpeg's
//!    `ff_h264_build_ref_list` error-concealment
//!    (libavcodec/h264_refs.c:400-412).
//!
//! 2. `p_cabac_8x8_96x96.{es,yuv}` — 3-frame mandelbrot IDR + P + P at
//!    High-Profile, `ref=2` / `8x8dct=1` / `partitions=p8x8`. Exercises
//!    the CABAC luma-8×8 residual path. Before the fix, the 8×8 decode
//!    read an extraneous `coded_block_flag` bin for `ctxBlockCat = 5`
//!    (luma-8×8 in 4:2:0) contrary to §7.3.5.3.1 — the flag is only
//!    present when `maxNumCoeff != 64` OR `ChromaArrayType == 3`. The
//!    drift shifted the CABAC state by one bin per 8×8 luma block,
//!    cascading through every subsequent `mb_skip_flag` / `mb_type` /
//!    `ref_idx_l0` decode of the same slice and surfacing as a
//!    spurious `ref_idx_l0 out of range` error many MBs later. FFmpeg
//!    enforces the gate as `cat != 5 || CHROMA444(h)` in
//!    `decode_cabac_residual_nondc` (libavcodec/h264_cabac.c:1912).
//!
//! Fixture regeneration:
//!
//! ```bash
//! # refpad:
//! ffmpeg -y -f lavfi -i mandelbrot=size=128x128:rate=24 -vframes 3 \
//!   -pix_fmt yuv420p -c:v libx264 -profile:v high -preset medium \
//!   -coder 1 -x264-params ref=2:bframes=0:keyint=3:partitions=all:8x8dct=0 \
//!   /tmp/p_cabac_refpad_128x128.mp4
//!
//! # 8x8 P:
//! ffmpeg -y -f lavfi -i mandelbrot=size=96x96:rate=24 -vframes 3 \
//!   -pix_fmt yuv420p -c:v libx264 -profile:v high -preset ultrafast \
//!   -coder 1 -g 3 -keyint_min 3 \
//!   -x264-params ref=2:bframes=0:8x8dct=1:partitions=p8x8 \
//!   /tmp/p_cabac_8x8_96x96.mp4
//!
//! # common:
//! ffmpeg -y -i <in>.mp4 -c:v copy -bsf:v h264_mp4toannexb \
//!   -f h264 tests/fixtures/<name>.es
//! ffmpeg -y -i <in>.mp4 -f rawvideo -pix_fmt yuv420p \
//!   tests/fixtures/<name>.yuv
//! ```

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::decoder::H264Decoder;
use oxideav_h264::nal::{split_annex_b, NalHeader, NalUnitType};

fn split_frames(es: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<Vec<u8>>) {
    let nalus = split_annex_b(es);
    let mut sps = None;
    let mut pps = None;
    let mut frames: Vec<Vec<u8>> = Vec::new();
    for nalu in &nalus {
        let h = NalHeader::parse(nalu[0]).unwrap();
        match h.nal_unit_type {
            NalUnitType::Sps => sps = Some(nalu.to_vec()),
            NalUnitType::Pps => pps = Some(nalu.to_vec()),
            NalUnitType::SliceIdr | NalUnitType::SliceNonIdr => {
                let mut frame = Vec::new();
                frame.extend_from_slice(&[0, 0, 0, 1]);
                frame.extend_from_slice(nalu);
                frames.push(frame);
            }
            _ => {}
        }
    }
    (sps.unwrap(), pps.unwrap(), frames)
}

fn decode_stream(es: &[u8]) -> Result<usize, String> {
    let (sps, pps, frames) = split_frames(es);
    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let mut primer = Vec::new();
    primer.extend_from_slice(&[0, 0, 0, 1]);
    primer.extend_from_slice(&sps);
    primer.extend_from_slice(&[0, 0, 0, 1]);
    primer.extend_from_slice(&pps);
    let tb = TimeBase::new(1, 90_000);
    dec.send_packet(&Packet::new(0, tb, primer))
        .map_err(|e| format!("primer: {e:?}"))?;
    while dec.receive_frame().is_ok() {}

    let mut decoded = 0usize;
    for (idx, data) in frames.iter().enumerate() {
        let pkt = Packet::new(0, tb, data.clone())
            .with_pts(idx as i64)
            .with_keyframe(idx == 0);
        dec.send_packet(&pkt)
            .map_err(|e| format!("frame {idx} send_packet: {e:?}"))?;
        match dec.receive_frame() {
            Ok(Frame::Video(_)) => decoded += 1,
            Ok(_) => return Err("non-video frame".into()),
            Err(e) => return Err(format!("frame {idx} receive_frame: {e:?}")),
        }
    }
    Ok(decoded)
}

#[test]
fn decode_cabac_p_refpad_does_not_error() {
    // Ref-list padding: DPB has 2 entries at frame-2 decode time but the
    // slice header sets `num_ref_idx_l0_active_minus1 = 2` (want = 3).
    // §7.4.5.1 allows `ref_idx_l0 ∈ [0, num_ref_idx_l0_active - 1] = [0,2]`;
    // without the fix, `ref_idx_l0 = 2` aborts with
    //     `ref_idx_l0 2 out of range (len=2)`.
    let es = include_bytes!("fixtures/p_cabac_refpad_128x128.es");
    let frames = decode_stream(es).expect("refpad fixture must decode without error");
    assert_eq!(frames, 3, "expected 3 frames (IDR + P + P)");
}

#[test]
fn decode_cabac_p_8x8_no_cbf_drift() {
    // 8×8 luma residual CBF gating: §7.3.5.3.1 — for `ctxBlockCat = 5`
    // (4:2:0 / 4:2:2 luma 8×8) the `coded_block_flag` bin is NOT
    // present. Reading it drifts the CABAC state by ≥ 1 bin per 8×8
    // block, which cascades into bogus `ref_idx_l0` / `mb_skip_flag`
    // decodes many MBs later.
    let es = include_bytes!("fixtures/p_cabac_8x8_96x96.es");
    let frames = decode_stream(es).expect("8×8 P fixture must decode without error");
    assert_eq!(frames, 3, "expected 3 frames (IDR + P + P)");
}
