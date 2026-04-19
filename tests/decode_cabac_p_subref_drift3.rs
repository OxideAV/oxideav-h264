//! Regression fixture for the `sub_ref_idx_l0 N out of range` CABAC P-slice
//! failure tracked in `wt/subref-drift3`.
//!
//! `p_cabac_skip_qp_drift_80x80.{es,yuv}` — 24-frame mandelbrot IDR + 23 P
//! at High-Profile, `ref=2`, `partitions=p8x8`, `8x8dct=1`, `preset=veryslow`.
//! x264's veryslow pass generates long P_Skip runs interspersed with
//! MBs carrying a non-zero `mb_qp_delta`.
//!
//! §9.3.3.1.1.5 specifies `ctxIdxInc` for `mb_qp_delta` bin 0 as
//! `condTermFlag = !(prevMbQpDelta == 0)`. A skipped MB has no
//! `mb_qp_delta`, so `prevMbQpDelta` resets to 0 across every skip.
//! Before the fix, the CABAC P-skip dispatch (`cabac::p_mb::decode_p_skip`)
//! left `pic.last_mb_qp_delta_was_nonzero` untouched, so a P_Skip that
//! followed a non-zero-delta MB kept `ctx_idx_inc = 1` for the next
//! coded MB's `mb_qp_delta` bin 0 — landing on CABAC context 61 instead
//! of 60. The wrong context flipped bin polarity, cascading through the
//! unary binarisation and surfacing many MBs later as either
//! `sub_ref_idx_l0 N out of range` (no `(len=…)` suffix — the sub-MB
//! bounds check at `cabac::p_mb::decode_p_8x8`) or
//! `end_of_slice_flag did not fire before last MB`.
//!
//! Fix (`cabac::p_mb::decode_p_skip` + sibling B / 10-bit / 4:4:4
//! skip paths) sets `pic.last_mb_qp_delta_was_nonzero = false` right
//! after the P_Skip / B_Skip MC call — mirroring FFmpeg's
//! `sl->last_qscale_diff = 0` after `decode_mb_skip`
//! (libavcodec/h264_cabac.c:1952).
//!
//! Fixture regeneration:
//!
//! ```bash
//! ffmpeg -y -f lavfi -i 'mandelbrot=size=80x80:rate=24' -t 1 \
//!   -pix_fmt yuv420p -c:v libx264 -profile:v high -preset veryslow \
//!   -coder 1 -x264-params ref=2:bframes=0:keyint=30:partitions=p8x8:8x8dct=1 \
//!   tests/fixtures/p_cabac_skip_qp_drift_80x80.es
//! ffmpeg -y -i tests/fixtures/p_cabac_skip_qp_drift_80x80.es \
//!   -f rawvideo -pix_fmt yuv420p tests/fixtures/p_cabac_skip_qp_drift_80x80.yuv
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
fn decode_cabac_p_skip_qp_delta_drift_does_not_error() {
    // Long P_Skip runs interleaved with MBs that carry a non-zero
    // `mb_qp_delta`. §9.3.3.1.1.5 requires the "previous non-zero
    // mb_qp_delta" tracker to reset to 0 across every skipped MB.
    // Before the fix the tracker kept stale state, selected CABAC ctx
    // 61 instead of 60 for the next coded MB's `mb_qp_delta` bin 0,
    // and drifted the bin stream until a sub-MB `ref_idx_l0 = 3`
    // surfaced against a `num_ref_idx_l0_active = 3` slice header —
    // aborting with `sub_ref_idx_l0 3 out of range` (note: no
    // `(len=…)` suffix; it's the sub-MB bounds check in
    // `cabac::p_mb::decode_p_8x8`, not the ref-list look-up in
    // `lookup_ref`).
    let es = include_bytes!("fixtures/p_cabac_skip_qp_drift_80x80.es");
    let frames = decode_stream(es).expect("skip_qp fixture must decode cleanly");
    assert_eq!(frames, 24, "expected 24 frames (IDR + 23 P)");
}
