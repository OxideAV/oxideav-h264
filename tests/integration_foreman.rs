//! Integration smoke test against `foreman_p16x16.264` — a 3.7 KB
//! Annex B H.264 clip from the ffmpeg samples archive.
//!
//! This test exercises the end-to-end parser pipeline: Annex B split
//! → NAL parse → SPS/PPS storage → slice header parsing. It does NOT
//! currently drive reconstruction — that requires exposing the bit
//! cursor where slice_data starts, which is a follow-up on the
//! Decoder API.
//!
//! The test looks for the clip at the path in `OXIDEAV_SAMPLES_H264`
//! env var, or at a hardcoded default path that matches the author's
//! workstation. If neither is present, the test is skipped with a
//! printed reason.

use oxideav_h264::decoder::{Decoder, Event};
use std::path::PathBuf;

fn sample_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("OXIDEAV_SAMPLES_H264_FOREMAN") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    // Author default.
    let default = PathBuf::from(
        "/home/magicaltux/projects/oxideav/samples/samples.ffmpeg.org/V-codecs/h264/foreman_p16x16.264",
    );
    if default.exists() {
        Some(default)
    } else {
        None
    }
}

#[test]
fn parse_foreman_p16x16() {
    let Some(path) = sample_path() else {
        eprintln!(
            "skip: set OXIDEAV_SAMPLES_H264_FOREMAN or place the file at the default path"
        );
        return;
    };

    let bytes = std::fs::read(&path).expect("read test clip");
    assert!(!bytes.is_empty(), "empty test clip");

    let mut dec = Decoder::new();
    let mut sps_stored = 0usize;
    let mut pps_stored = 0usize;
    let mut slices = 0usize;
    let mut idr_slices = 0usize;
    let mut sei_msgs = 0usize;
    let mut auds = 0usize;
    let mut ignored = 0usize;
    let mut errors: Vec<String> = Vec::new();

    for result in dec.process_annex_b(&bytes) {
        match result {
            Ok(ev) => match ev {
                Event::SpsStored(_) => sps_stored += 1,
                Event::PpsStored(_) => pps_stored += 1,
                Event::Slice { nal_unit_type, .. } => {
                    slices += 1;
                    if nal_unit_type == 5 {
                        idr_slices += 1;
                    }
                }
                Event::Sei(msgs) => sei_msgs += msgs.len(),
                Event::AccessUnitDelimiter(_) => auds += 1,
                Event::Ignored { .. } => ignored += 1,
                _ => {}
            },
            Err(e) => errors.push(format!("{e}")),
        }
    }

    eprintln!(
        "foreman_p16x16.264 → SPS:{sps_stored} PPS:{pps_stored} IDR:{idr_slices} \
         slices:{slices} SEI:{sei_msgs} AUD:{auds} ignored:{ignored} errors:{}",
        errors.len()
    );
    for e in &errors {
        eprintln!("  err: {e}");
    }

    // We expect at least one SPS, one PPS, and one IDR slice.
    assert!(sps_stored >= 1, "no SPS parsed");
    assert!(pps_stored >= 1, "no PPS parsed");
    assert!(idr_slices >= 1, "no IDR slice parsed");
    assert_eq!(errors, Vec::<String>::new(), "no parse errors expected");

    // Sanity-check the active SPS was plausible.
    let sps = dec.active_sps().expect("active SPS");
    eprintln!(
        "  SPS: profile={} level={} {}x{}mb chroma={} {}-bit",
        sps.profile_idc,
        sps.level_idc,
        sps.pic_width_in_mbs_minus1 + 1,
        sps.pic_height_in_map_units_minus1 + 1,
        sps.chroma_format_idc,
        sps.bit_depth_luma_minus8 + 8
    );
}
