//! Regression fixture for the known CABAC I-slice correctness gap on
//! **High-profile** 4:2:0 8-bit streams at realistic sizes.
//!
//! The Main-profile CABAC path (see `decode_cabac_b.rs`) already
//! exercises I + P + B slices at 64×64 and documents that the I-slice
//! produces non-bit-exact pixels. At that size the decoder stays in
//! sync — the residual drift only shows up in pixel values. At any
//! High-profile size ≥ ~200×200, the same underlying desync starts
//! leaking out of the I-slice into the CABAC engine state: a few
//! frames in, the arithmetic decoder reads past the end of a slice
//! and every subsequent slice errors out. That's what failed on the
//! user-reported Slayer/Repentless 1920×1080 MKV.
//!
//! This test captures the failure on a tiny public-domain synthesised
//! fixture so the h264 author has a 5 KB repro that's committable as
//! CI-safe regression. It intentionally asserts the **current broken
//! behaviour** so it stays green on HEAD today. Once the CABAC I-slice
//! residual bug is fixed upstream and frames 8+ stop erroring, flip
//! the assertion: the test will then demand a clean decode.
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
//! ```

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_h264::nal::{split_annex_b, NalHeader, NalUnitType};

const ES: &[u8] = include_bytes!("fixtures/cabac_high_200x200.es");

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

/// Decode all frames and return per-frame outcomes. `Ok(())` when the
/// frame went through without an error; `Err(msg)` on the first error
/// reported by `send_packet`.
fn decode_frames(es: &[u8]) -> Vec<Result<(), String>> {
    let (sps, pps, frames) = split_frames(es);
    let mut params = CodecParameters::video(CodecId::new("h264"));
    params.extradata.clear();
    let mut dec = oxideav_h264::decoder::make_decoder(&params).expect("make_decoder");

    // Prime the SPS + PPS as one combined Annex-B packet.
    let mut header = Vec::new();
    header.extend_from_slice(&[0, 0, 0, 1]);
    header.extend_from_slice(&sps);
    header.extend_from_slice(&[0, 0, 0, 1]);
    header.extend_from_slice(&pps);
    let pkt = Packet::new(0, TimeBase::new(1, 24), header);
    let _ = dec.send_packet(&pkt);

    let mut out = Vec::with_capacity(frames.len());
    for frame_bytes in &frames {
        let pkt = Packet::new(0, TimeBase::new(1, 24), frame_bytes.clone());
        let res = dec.send_packet(&pkt).map_err(|e| format!("{e}"));
        // Drain any frames produced so the decoder state advances
        // correctly between sends. Errors from receive_frame are the
        // "need-more packet" / "EOF" signals, not fatal; fatal errors
        // already come from send_packet.
        while let Ok(Frame::Video(_)) = dec.receive_frame() {}
        out.push(res);
    }
    out
}

/// Sanity check: the fixture contains exactly 12 frames (IDR + 11
/// referencing frames) — matches the `ffmpeg` invocation in the
/// regeneration docstring at the top of this file. If fixture
/// regeneration drifts this will fail loudly.
#[test]
fn fixture_contains_expected_frame_count() {
    let (_, _, frames) = split_frames(ES);
    assert_eq!(frames.len(), 12, "fixture should contain 12 VCL frames");
}

/// Known-broken: decoding 12 frames of High-profile CABAC at 200×200
/// today desyncs the CABAC arithmetic engine partway through and
/// starts returning `cabac: read past end of stream`. This test
/// asserts the **current broken behaviour** so it stays green on
/// HEAD. The expected transition is:
///
/// * **Today (broken):** at least one frame past the first returns a
///   CABAC error. Test passes because the expected-broken assertion
///   holds.
/// * **After fix:** every frame returns `Ok(())`. The assertion below
///   will then *fail* — flip the test to `results.iter().all(Result::is_ok)`
///   or split into the strict pixel-equality variant next to this one.
///
/// See `examples/probe.rs` for a standalone harness that also
/// byte-diffs the decoded YUV against an external reference.
#[test]
fn cabac_high_profile_engine_currently_desyncs_past_idr() {
    let results = decode_frames(ES);
    let errored = results
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.as_ref().err().map(|e| (i, e.clone())))
        .collect::<Vec<_>>();

    eprintln!("decode_frames results:");
    for (i, r) in results.iter().enumerate() {
        match r {
            Ok(()) => eprintln!("  frame {i:>2}: ok"),
            Err(e) => eprintln!("  frame {i:>2}: ERR {e}"),
        }
    }

    // Frame 0 (IDR) is decoded without a CABAC-engine-level error
    // today — the I-slice pixel-residual drift documented in
    // `decode_cabac_b.rs` corrupts the output bytes but leaves the
    // bitstream read cursor in a valid place. The desync only starts
    // cascading once later slices build on the corrupted IDR state.
    assert!(
        results[0].is_ok(),
        "frame 0 (IDR) should not error today: got {:?}",
        results[0]
    );

    // At least one subsequent frame must surface a `cabac: read past
    // end of stream`. If *none* do today, the CABAC I-slice fix has
    // landed — celebrate and flip this assertion to require all-ok.
    assert!(
        errored
            .iter()
            .any(|(_, msg)| msg.contains("cabac: read past end")),
        "expected the known CABAC desync to surface on HEAD; got errors = {:?}",
        errored
    );
}
