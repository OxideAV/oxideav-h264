//! §7.4.1.2.4 multi-slice assembly tests.
//!
//! These tests exercise `H264CodecDecoder`'s primary-coded-picture
//! assembly logic — verifying that two slices sharing the §7.4.1.2.4
//! "same picture" signature produce a single `VideoFrame`, and that
//! streams with known slice-per-picture structure produce the right
//! frame count.
//!
//! Spec references:
//! * §7.4.1.2 — access unit
//! * §7.4.1.2.3 — AUD marks access unit boundary
//! * §7.4.1.2.4 — detection of first VCL NAL unit of a primary coded
//!   picture (the list of conditions we compare)

use oxideav_codec::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264::nal::AnnexBSplitter;
use std::path::PathBuf;

fn foreman_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("OXIDEAV_SAMPLES_H264_FOREMAN") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    let default = PathBuf::from(
        "/home/magicaltux/projects/oxideav/samples/samples.ffmpeg.org/V-codecs/h264/foreman_p16x16.264",
    );
    if default.exists() { Some(default) } else { None }
}

fn moonlight_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("OXIDEAV_SAMPLES_H264_MOONLIGHT") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    let default = PathBuf::from(
        "/home/magicaltux/projects/oxideav/samples/samples.ffmpeg.org/V-codecs/h264/moonlight.264",
    );
    if default.exists() { Some(default) } else { None }
}

/// Drive `bytes` as a single Annex B packet through the decoder, flush,
/// and collect all frames. Panics on decoder errors (caller wants the
/// test to fail loud).
fn decode_all_frames(bytes: &[u8]) -> Vec<Frame> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let packet = Packet::new(0, TimeBase::new(1, 25), bytes.to_vec()).with_pts(0);
    dec.send_packet(&packet).expect("send_packet");
    dec.flush().expect("flush");
    let mut out = Vec::new();
    while let Ok(f) = dec.receive_frame() {
        out.push(f);
    }
    out
}

/// §7.4.1.2.4 — `foreman_p16x16.264` has two slices with different
/// `frame_num` (0 and 1) and different POC (0 and 2). Each is its own
/// primary coded picture, so we still expect two `VideoFrame`s out.
#[test]
fn foreman_p16x16_emits_one_frame_per_coded_picture() {
    let Some(path) = foreman_path() else {
        eprintln!("skip: foreman sample missing");
        return;
    };
    let bytes = std::fs::read(&path).expect("read foreman");
    let frames = decode_all_frames(&bytes);
    // ffmpeg -i foreman_p16x16.264 reports 2 frames for this clip, which
    // matches the 2 slices (each with unique frame_num/POC).
    assert_eq!(
        frames.len(),
        2,
        "expected 2 frames from foreman_p16x16.264, got {}",
        frames.len()
    );
}

/// §7.4.1.2.4 — `moonlight.264` is the multi-slice canary: every
/// picture is split into two slices (first_mb_in_slice = 0 and 384)
/// that share `frame_num`, `pic_parameter_set_id`, POC, IDR flag, and
/// nal_ref_idc zero-ness. Before the multi-slice fix the decoder
/// emitted 252 frames (one per slice); after the fix it should emit
/// ~126 (one per coded picture).
#[test]
fn moonlight_two_slices_per_picture_emits_half_the_frame_count() {
    let Some(path) = moonlight_path() else {
        eprintln!("skip: moonlight sample missing");
        return;
    };
    let bytes = std::fs::read(&path).expect("read moonlight");

    // First count the slices for a sanity baseline.
    let slice_count = {
        use oxideav_h264::decoder::{Decoder, Event};
        let mut d = Decoder::new();
        d.process_annex_b(&bytes)
            .filter(|ev| matches!(ev, Ok(Event::Slice { .. })))
            .count()
    };

    let frames = decode_all_frames(&bytes);

    eprintln!(
        "moonlight: {} slices produced {} frames",
        slice_count,
        frames.len()
    );

    // Every picture has exactly 2 slices, so frame count should be
    // slice_count / 2. Allow ±1 in case the stream tail has an odd
    // last access unit (it doesn't, but robustness).
    let expected = slice_count / 2;
    let diff = (frames.len() as isize - expected as isize).abs();
    assert!(
        diff <= 1,
        "expected ~{} frames from moonlight (slice_count/2), got {}",
        expected,
        frames.len()
    );

    // And critically: the frame count must be STRICTLY less than the
    // slice count — otherwise multi-slice assembly isn't firing.
    assert!(
        frames.len() < slice_count,
        "multi-slice assembly didn't fire: {} frames >= {} slices",
        frames.len(),
        slice_count
    );
}

/// Wrap a raw NAL unit (no start-code prefix) as an Annex B NAL by
/// prepending the 4-byte start-code prefix.
fn annex_b_wrap(nal: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(4 + nal.len());
    v.extend_from_slice(&[0, 0, 0, 1]);
    v.extend_from_slice(nal);
    v
}

/// Build an Access Unit Delimiter (nal_unit_type == 9) with
/// `primary_pic_type == 0`. RBSP: 3 bits of type + stop_one_bit + zero
/// pad → 0x10. NAL header: nal_ref_idc=0, nal_unit_type=9 → 0x09.
/// Together: [0x09, 0x10].
fn aud_nal_bytes() -> Vec<u8> {
    vec![0x09, 0x10]
}

/// Collect Annex B NAL units from moonlight: `(sps, pps, slice1, slice2)`.
type MoonlightNals = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);
fn collect_moonlight_nals() -> Option<MoonlightNals> {
    let path = moonlight_path()?;
    let bytes = std::fs::read(&path).ok()?;
    let mut sps = None;
    let mut pps = None;
    let mut slices: Vec<Vec<u8>> = Vec::new();
    for nal in AnnexBSplitter::new(&bytes) {
        if nal.is_empty() {
            continue;
        }
        let nut = nal[0] & 0x1f;
        match nut {
            7 => sps = Some(nal.to_vec()),
            8 => pps = Some(nal.to_vec()),
            1 | 5 if slices.len() < 2 => slices.push(nal.to_vec()),
            _ => {}
        }
        if sps.is_some() && pps.is_some() && slices.len() >= 2 {
            break;
        }
    }
    Some((sps?, pps?, slices.first()?.clone(), slices.get(1)?.clone()))
}

/// §7.4.1.2.3 — an AUD between two slices of the same coded picture
/// forces a picture boundary, so 2 frames should come out even though
/// the two slices would otherwise assemble into one.
///
/// Baseline: without the AUD moonlight's first two slices produce 1
/// frame (they share frame_num/POC/etc.). With an AUD injected between
/// them we should see 2 frames.
#[test]
fn aud_between_slices_splits_into_two_pictures() {
    let Some((sps, pps, s1, s2)) = collect_moonlight_nals() else {
        eprintln!("skip: moonlight sample missing");
        return;
    };

    // Baseline: SPS + PPS + slice1 + slice2 → 1 frame (they're halves
    // of the same primary coded picture).
    let mut baseline: Vec<u8> = Vec::new();
    baseline.extend_from_slice(&annex_b_wrap(&sps));
    baseline.extend_from_slice(&annex_b_wrap(&pps));
    baseline.extend_from_slice(&annex_b_wrap(&s1));
    baseline.extend_from_slice(&annex_b_wrap(&s2));
    let baseline_frames = decode_all_frames(&baseline).len();
    assert_eq!(
        baseline_frames, 1,
        "moonlight slice1+slice2 should assemble into 1 picture"
    );

    // With AUD: SPS + PPS + slice1 + AUD + slice2 → 2 frames.
    let aud = aud_nal_bytes();
    let mut with_aud: Vec<u8> = Vec::new();
    with_aud.extend_from_slice(&annex_b_wrap(&sps));
    with_aud.extend_from_slice(&annex_b_wrap(&pps));
    with_aud.extend_from_slice(&annex_b_wrap(&s1));
    with_aud.extend_from_slice(&annex_b_wrap(&aud));
    with_aud.extend_from_slice(&annex_b_wrap(&s2));
    let with_aud_frames = decode_all_frames(&with_aud).len();
    assert_eq!(
        with_aud_frames, 2,
        "AUD between slices should yield 2 frames (got {})",
        with_aud_frames
    );
}

/// §7.4.1.2.4 — two moonlight slices sharing all the listed conditions
/// produce exactly one primary coded picture (multi-slice assembly).
///
/// This is a slimmed-down, focused version of the moonlight whole-stream
/// test that only looks at the first picture. The big test is a broad
/// sanity check against a real encoder; this one pins down the exact
/// "2 slices → 1 frame" behaviour on a minimal synthetic stream.
#[test]
fn two_moonlight_slices_assemble_into_one_frame() {
    let Some((sps, pps, s1, s2)) = collect_moonlight_nals() else {
        eprintln!("skip: moonlight sample missing");
        return;
    };

    let mut stream: Vec<u8> = Vec::new();
    stream.extend_from_slice(&annex_b_wrap(&sps));
    stream.extend_from_slice(&annex_b_wrap(&pps));
    stream.extend_from_slice(&annex_b_wrap(&s1));
    stream.extend_from_slice(&annex_b_wrap(&s2));

    let frames = decode_all_frames(&stream);
    assert_eq!(
        frames.len(),
        1,
        "two slices with matching §7.4.1.2.4 signature should yield 1 frame, got {}",
        frames.len()
    );
}
