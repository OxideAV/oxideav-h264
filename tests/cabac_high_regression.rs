//! Regression guard for CABAC High-profile streams with mixed I / P / B
//! slices at realistic sizes.
//!
//! The Main-profile CABAC path (see `decode_cabac_b.rs`) already
//! exercises I + P + B slices at 64×64. At any High-profile size ≥
//! ~200×200 with the same GOP shape, an earlier bug caused the
//! arithmetic decoder to read past end of stream a few frames into
//! the first B-slice — the test now bit-exact verifies every decoded
//! display against the libavcodec reference YUV so that regression
//! cannot come back.
//!
//! ## Fixture regeneration
//!
//! ```sh
//! ffmpeg -y -f lavfi -i testsrc=duration=1:size=200x200:rate=24 \
//!   -vframes 12 -pix_fmt yuv420p -c:v libx264 \
//!   -profile:v high -level 4.0 -preset medium \
//!   -coder 1 -bf 2 -refs 1 -g 12 /tmp/repro.mp4
//! ffmpeg -y -i /tmp/repro.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 \
//!   tests/fixtures/cabac_high_200x200.es
//! ffmpeg -y -i /tmp/repro.mp4 -vframes 12 -f rawvideo -pix_fmt yuv420p \
//!   tests/fixtures/cabac_high_200x200.yuv
//! ```

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::nal::{split_annex_b, NalHeader, NalUnitType};

const ES: &[u8] = include_bytes!("fixtures/cabac_high_200x200.es");
const REF_YUV: &[u8] = include_bytes!("fixtures/cabac_high_200x200.yuv");
const WIDTH: usize = 200;
const HEIGHT: usize = 200;
const FRAME_BYTES: usize = WIDTH * HEIGHT * 3 / 2;
const EXPECTED_FRAMES: usize = 12;

/// Split an Annex-B byte stream into (sps, pps, frame_bodies). Each
/// frame body is a single VCL NALU with a 4-byte start code prefix —
/// good enough for `testsrc`-derived clips where one slice == one
/// frame.
fn split_frames(es: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<Vec<u8>>) {
    let nalus = split_annex_b(es);
    let mut sps = None;
    let mut pps = None;
    let mut frames: Vec<Vec<u8>> = Vec::new();
    for nalu in &nalus {
        if nalu.is_empty() {
            continue;
        }
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

/// Flatten one decoded video frame into a yuv420p byte vector, trimming
/// each plane to its logical size (strides may pad for alignment).
fn flatten_yuv420p(vf: &VideoFrame) -> Vec<u8> {
    let mut out = Vec::with_capacity(FRAME_BYTES);
    for (pi, plane) in vf.planes.iter().enumerate() {
        let (pw, ph) = if pi == 0 {
            (WIDTH, HEIGHT)
        } else {
            (WIDTH / 2, HEIGHT / 2)
        };
        for row in 0..ph {
            let off = row * plane.stride;
            out.extend_from_slice(&plane.data[off..off + pw]);
        }
    }
    out
}

/// Byte diff + max absolute deviation between two equal-sized buffers.
/// Used for a friendlier failure message than a straight `assert_eq`.
fn summarise_diff(ours: &[u8], theirs: &[u8]) -> (usize, u8) {
    debug_assert_eq!(ours.len(), theirs.len());
    let diff = ours.iter().zip(theirs).filter(|(a, b)| a != b).count();
    let max = ours
        .iter()
        .zip(theirs)
        .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    (diff, max)
}

/// Cheap sanity check that the fixture contents haven't drifted. Runs
/// in the default `cargo test` suite so regenerating the fixture with
/// a different `testsrc`/x264 config immediately fails here.
#[test]
fn fixture_contains_expected_frame_count() {
    let (_, _, frames) = split_frames(ES);
    assert_eq!(frames.len(), EXPECTED_FRAMES, "fixture frame count drift");
    assert_eq!(
        REF_YUV.len(),
        FRAME_BYTES * EXPECTED_FRAMES,
        "reference YUV size drift"
    );
}

/// Hard bit-exact check of `oxideav-h264` against libavcodec on a
/// High-profile CABAC fixture with I + P + B slices. Covers the
/// CABAC B-slice intra-in-B context bank (ctx 32..=35) — mis-routing
/// this to the I-slice bank desyncs the arithmetic decoder inside
/// the first B-slice of the GOP.
#[test]
fn cabac_high_profile_decodes_bit_exact_vs_ffmpeg() {
    let (sps, pps, frames) = split_frames(ES);
    assert_eq!(frames.len(), EXPECTED_FRAMES);

    let mut params = CodecParameters::video(CodecId::new("h264"));
    params.extradata.clear();
    let mut dec = oxideav_h264::decoder::make_decoder(&params).expect("make_decoder");

    let mut header = Vec::new();
    header.extend_from_slice(&[0, 0, 0, 1]);
    header.extend_from_slice(&sps);
    header.extend_from_slice(&[0, 0, 0, 1]);
    header.extend_from_slice(&pps);
    let pkt = Packet::new(0, TimeBase::new(1, 24), header);
    let _ = dec.send_packet(&pkt);

    let mut display_idx = 0usize;
    let check = |vf: &VideoFrame, display_idx: usize, decode_idx: Option<usize>| {
        let ours = flatten_yuv420p(vf);
        let off = display_idx * FRAME_BYTES;
        assert!(
            off + FRAME_BYTES <= REF_YUV.len(),
            "decoded past end of reference yuv at display #{display_idx}"
        );
        let theirs = &REF_YUV[off..off + FRAME_BYTES];
        if ours != theirs {
            let (diff, max) = summarise_diff(&ours, theirs);
            match decode_idx {
                Some(i) => panic!(
                    "display #{display_idx} (from decode frame {i}): \
                     {diff}/{total} bytes differ, max |Δ|={max} — \
                     libavcodec-bit-exact decode regressed (or the \
                     known CABAC I-slice bug is still present).",
                    total = FRAME_BYTES
                ),
                None => panic!(
                    "display #{display_idx} (drained after flush): \
                     {diff}/{total} bytes differ, max |Δ|={max} — \
                     libavcodec-bit-exact decode regressed.",
                    total = FRAME_BYTES
                ),
            }
        }
    };
    for (decode_idx, frame_bytes) in frames.iter().enumerate() {
        let pkt = Packet::new(0, TimeBase::new(1, 24), frame_bytes.clone());
        dec.send_packet(&pkt)
            .unwrap_or_else(|e| panic!("send_packet on decode frame {decode_idx} failed: {e}"));
        while let Ok(Frame::Video(vf)) = dec.receive_frame() {
            check(&vf, display_idx, Some(decode_idx));
            display_idx += 1;
        }
    }
    // Final flush — POC reorder may hold the last 1-2 pictures back
    // until EOS. Drain them all so the full display sequence is
    // verified end-to-end.
    dec.flush().expect("flush");
    while let Ok(Frame::Video(vf)) = dec.receive_frame() {
        check(&vf, display_idx, None);
        display_idx += 1;
    }
    assert_eq!(
        display_idx, EXPECTED_FRAMES,
        "decoder emitted {display_idx} frames, expected {EXPECTED_FRAMES}"
    );
}
