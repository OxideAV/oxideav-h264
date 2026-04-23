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
use oxideav_h264::mb_grid::MbGrid;
use oxideav_h264::picture::Picture;
use oxideav_h264::ref_store::NoRefs;
use oxideav_h264::{reconstruct, slice_data};
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
        eprintln!("skip: set OXIDEAV_SAMPLES_H264_FOREMAN or place the file at the default path");
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
    let mut slice_data_parsed = 0usize;
    let mut slice_data_errors: Vec<String> = Vec::new();
    let mut reconstructed = 0usize;
    let mut reconstruct_errors: Vec<String> = Vec::new();
    let mut first_idr_luma_sample: Option<i32> = None;

    // Drive the full pipeline. Collect owned data first to detach from
    // the Decoder borrow before reconstruction (so we can pull SPS/PPS
    // back out).
    let mut pending_slices: Vec<(
        u8,
        u8,
        oxideav_h264::slice_header::SliceHeader,
        Vec<u8>,
        (usize, u8),
    )> = Vec::new();
    for result in dec.process_annex_b(&bytes) {
        match result {
            Ok(ev) => match ev {
                Event::SpsStored(_) => sps_stored += 1,
                Event::PpsStored(_) => pps_stored += 1,
                Event::Slice {
                    nal_unit_type,
                    nal_ref_idc,
                    header,
                    rbsp,
                    slice_data_cursor,
                    ..
                } => {
                    slices += 1;
                    if nal_unit_type == 5 {
                        idr_slices += 1;
                    }
                    pending_slices.push((
                        nal_unit_type,
                        nal_ref_idc,
                        header,
                        rbsp,
                        slice_data_cursor,
                    ));
                }
                Event::Sei(msgs) => sei_msgs += msgs.len(),
                Event::AccessUnitDelimiter(_) => auds += 1,
                Event::Ignored { .. } => ignored += 1,
                _ => {}
            },
            Err(e) => errors.push(format!("{e}")),
        }
    }

    // Now drive slice_data parsing + reconstruction for each slice.
    let sps_snapshot = dec.active_sps().cloned();
    let pps_snapshot = dec.active_pps().cloned();
    if let (Some(sps), Some(pps)) = (sps_snapshot, pps_snapshot) {
        eprintln!(
            "  PPS: entropy_coding={} num_ref_idx_l0={} weighted_pred={} \
             deblocking_ctrl={} constrained_intra={} \
             transform_8x8={} chroma_qp_offset={}",
            pps.entropy_coding_mode_flag,
            pps.num_ref_idx_l0_default_active_minus1,
            pps.weighted_pred_flag,
            pps.deblocking_filter_control_present_flag,
            pps.constrained_intra_pred_flag,
            pps.transform_8x8_mode_flag(),
            pps.chroma_qp_index_offset,
        );
        if let Some((nut, _, hdr, _, (byte, bit))) = pending_slices.first() {
            eprintln!(
                "  first slice: type={} cursor=({},{}) slice_type={:?} \
                 slice_qp_delta={} disable_dbf={}",
                nut,
                byte,
                bit,
                hdr.slice_type,
                hdr.slice_qp_delta,
                hdr.disable_deblocking_filter_idc,
            );
        }
        for (nal_unit_type, _nal_ref_idc, hdr, rbsp, (byte, bit)) in &pending_slices {
            let want_idr = *nal_unit_type == 5;
            match slice_data::parse_slice_data(rbsp, *byte, *bit, hdr, &sps, &pps) {
                Ok(sd) => {
                    slice_data_parsed += 1;
                    if !want_idr {
                        continue; // skip reconstruction for P slices (need refs)
                    }
                    // Attempt IDR reconstruction.
                    let width_samples = sps.pic_width_in_mbs() * 16;
                    let height_samples = sps.frame_height_in_mbs() * 16;
                    let mut pic = Picture::new(
                        width_samples,
                        height_samples,
                        sps.chroma_array_type(),
                        sps.bit_depth_luma_minus8 + 8,
                        sps.bit_depth_chroma_minus8 + 8,
                    );
                    let mut grid = MbGrid::new(sps.pic_width_in_mbs(), sps.frame_height_in_mbs());
                    match reconstruct::reconstruct_slice(
                        &sd, hdr, &sps, &pps, &NoRefs, &mut pic, &mut grid,
                    ) {
                        Ok(()) => {
                            reconstructed += 1;
                            if first_idr_luma_sample.is_none() {
                                // Grab the top-left luma sample as a liveness
                                // signal — confirms the pipeline wrote pixels.
                                first_idr_luma_sample = Some(pic.luma_at(0, 0));
                            }
                        }
                        Err(e) => reconstruct_errors.push(format!("{e}")),
                    }
                }
                Err(e) => slice_data_errors.push(format!("{e}")),
            }
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
    eprintln!(
        "  slice_data parsed: {slice_data_parsed}/{slices} \
         (errors:{})",
        slice_data_errors.len()
    );
    for e in slice_data_errors.iter().take(3) {
        eprintln!("    slice_data err: {e}");
    }
    eprintln!(
        "  reconstructed IDR slices: {reconstructed} (errors:{})",
        reconstruct_errors.len()
    );
    for e in reconstruct_errors.iter().take(3) {
        eprintln!("    reconstruct err: {e}");
    }
    if let Some(s) = first_idr_luma_sample {
        eprintln!("  first IDR top-left luma sample: {s}");
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
