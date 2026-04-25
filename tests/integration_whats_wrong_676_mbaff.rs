//! Integration test for `whats_wrong_676.mp4` — an MBAFF H.264 clip
//! from the ffmpeg samples archive.
//!
//! The test verifies that Phase-1 MBAFF slice_data parsing no longer
//! rejects the stream with `MbaffNotSupported`. Reconstruction is
//! still out of scope — slice_data parsing is allowed to succeed or
//! fail with a more-specific error (e.g. a macroblock-layer residual
//! path that isn't wired), as long as the MBAFF walker itself
//! advanced past the old blanket rejection.
//!
//! The file is an ISO BMFF MP4. We use a minimal hand-rolled box
//! walker to extract the `avcC` record (SPS + PPS + length_size)
//! and the `mdat` payload with its first sample offset, then scan
//! the first few length-prefixed NAL units and drive them through
//! `Decoder::process_annex_b` to parse SPS/PPS + slice headers.

use oxideav_h264::slice_data;
use std::path::PathBuf;

fn sample_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("OXIDEAV_SAMPLES_H264_WHATS_WRONG") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    let default = PathBuf::from(
        "/home/magicaltux/projects/oxideav/samples/samples.ffmpeg.org/archive/all/mov+h264+++whats_wrong_676.mp4",
    );
    if default.exists() {
        Some(default)
    } else {
        None
    }
}

/// Walk the boxes under `buf` looking for the named box, descending
/// into known containers. Returns the slice pointing to the box's
/// body (after the 8-byte header). Supports size=1 (largesize) and
/// size=0 (to EOF).
fn find_box<'a>(buf: &'a [u8], path: &[&[u8; 4]]) -> Option<&'a [u8]> {
    find_box_inner(buf, path)
}

fn find_box_inner<'a>(buf: &'a [u8], path: &[&[u8; 4]]) -> Option<&'a [u8]> {
    if path.is_empty() {
        return Some(buf);
    }
    let mut i = 0usize;
    while i + 8 <= buf.len() {
        let size = u32::from_be_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]) as usize;
        let ty: &[u8; 4] = buf[i + 4..i + 8].try_into().unwrap();
        let (hdr_len, box_len) = if size == 1 && i + 16 <= buf.len() {
            let ls = u64::from_be_bytes([
                buf[i + 8],
                buf[i + 9],
                buf[i + 10],
                buf[i + 11],
                buf[i + 12],
                buf[i + 13],
                buf[i + 14],
                buf[i + 15],
            ]) as usize;
            (16, ls)
        } else if size == 0 {
            (8, buf.len() - i)
        } else {
            (8, size)
        };
        if box_len < hdr_len || i + box_len > buf.len() {
            break;
        }
        if ty == path[0] {
            let body = &buf[i + hdr_len..i + box_len];
            // Some box types need to skip a fixed preamble before
            // their children start. Handle the two we traverse here.
            let child_body: &[u8] = match ty {
                // `trak`, `mdia`, `minf`, `stbl`, `moov` are bare
                // containers (full box with 0 overhead? no — these
                // are NOT full boxes, so they have 0 preamble).
                b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" => body,
                // stsd: FullBox (1 version + 3 flags = 4 bytes) +
                // u32 entry_count, then child sample-entry boxes.
                b"stsd" if body.len() >= 8 => &body[8..],
                // Leaf boxes — return the body as-is.
                _ => body,
            };
            if path.len() == 1 {
                return Some(body);
            }
            // Descend. For sample-entry boxes (avc1, avc3, …) there's
            // a fixed 78-byte VisualSampleEntry preamble before avcC.
            if ty == b"stsd" {
                // The first child entry is at the start of child_body.
                // Descend into ANY sample entry: parse it as a box,
                // then inside that box skip the 78-byte preamble.
                if child_body.len() >= 8 {
                    let entry_size = u32::from_be_bytes([
                        child_body[0],
                        child_body[1],
                        child_body[2],
                        child_body[3],
                    ]) as usize;
                    if entry_size <= child_body.len() && entry_size >= 8 + 78 {
                        let inner = &child_body[8 + 78..entry_size];
                        if let Some(found) = find_box_inner(inner, &path[1..]) {
                            return Some(found);
                        }
                    }
                }
            }
            if let Some(found) = find_box_inner(child_body, &path[1..]) {
                return Some(found);
            }
        }
        i += box_len;
    }
    None
}

/// Find the first `stco` (or `co64`) entry — the first chunk offset.
fn first_chunk_offset(buf: &[u8]) -> Option<u64> {
    if let Some(stco) = find_box(buf, &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stco"]) {
        if stco.len() >= 16 {
            // FullBox (4) + entry_count (4) + first offset (4)
            let off = u32::from_be_bytes([stco[8], stco[9], stco[10], stco[11]]);
            return Some(off as u64);
        }
    }
    if let Some(co64) = find_box(buf, &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"co64"]) {
        if co64.len() >= 20 {
            let off = u64::from_be_bytes([
                co64[8], co64[9], co64[10], co64[11], co64[12], co64[13], co64[14], co64[15],
            ]);
            return Some(off);
        }
    }
    None
}

/// avcC record (ISO/IEC 14496-15 §5.2.4.1):
/// `length_size`, SPS NAL units, PPS NAL units.
type AvccRecord = (usize, Vec<Vec<u8>>, Vec<Vec<u8>>);

/// Parse the avcC record (ISO/IEC 14496-15 §5.2.4.1) returning:
/// (length_size_minus_one + 1, SPS NAL units, PPS NAL units).
fn parse_avcc(avcc: &[u8]) -> Option<AvccRecord> {
    if avcc.len() < 7 {
        return None;
    }
    let length_size = (avcc[4] & 0x03) as usize + 1;
    let num_sps = (avcc[5] & 0x1f) as usize;
    let mut i = 6usize;
    let mut sps_list = Vec::new();
    for _ in 0..num_sps {
        if i + 2 > avcc.len() {
            return None;
        }
        let len = u16::from_be_bytes([avcc[i], avcc[i + 1]]) as usize;
        i += 2;
        if i + len > avcc.len() {
            return None;
        }
        sps_list.push(avcc[i..i + len].to_vec());
        i += len;
    }
    if i >= avcc.len() {
        return None;
    }
    let num_pps = avcc[i] as usize;
    i += 1;
    let mut pps_list = Vec::new();
    for _ in 0..num_pps {
        if i + 2 > avcc.len() {
            return None;
        }
        let len = u16::from_be_bytes([avcc[i], avcc[i + 1]]) as usize;
        i += 2;
        if i + len > avcc.len() {
            return None;
        }
        pps_list.push(avcc[i..i + len].to_vec());
        i += len;
    }
    Some((length_size, sps_list, pps_list))
}

#[test]
fn whats_wrong_676_mbaff_slice_data_no_longer_rejected() {
    let Some(path) = sample_path() else {
        eprintln!("skip: set OXIDEAV_SAMPLES_H264_WHATS_WRONG or place file at the default path");
        return;
    };

    let bytes = std::fs::read(&path).expect("read mp4 file");
    assert!(!bytes.is_empty(), "empty mp4");

    // Find avcC
    let avcc = find_box(
        &bytes,
        &[
            b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd", b"avcC",
        ],
    )
    .expect("avcC box");
    let (length_size, sps_list, pps_list) = parse_avcc(avcc).expect("parse avcC");
    eprintln!(
        "avcC: length_size={} SPS_count={} PPS_count={}",
        length_size,
        sps_list.len(),
        pps_list.len()
    );
    assert!(!sps_list.is_empty() && !pps_list.is_empty());

    // Find mdat box start to know where the payload region is. We
    // also use the stco first chunk offset as the absolute file
    // offset of the first sample (usually a few bytes inside mdat).
    // Compute mdat range by walking the top-level boxes.
    let mut mdat_start_file = 0usize;
    let mut mdat_end_file = 0usize;
    {
        let mut i = 0usize;
        while i + 8 <= bytes.len() {
            let size =
                u32::from_be_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize;
            let ty = &bytes[i + 4..i + 8];
            let (hdr_len, box_len) = if size == 1 && i + 16 <= bytes.len() {
                let ls = u64::from_be_bytes([
                    bytes[i + 8],
                    bytes[i + 9],
                    bytes[i + 10],
                    bytes[i + 11],
                    bytes[i + 12],
                    bytes[i + 13],
                    bytes[i + 14],
                    bytes[i + 15],
                ]) as usize;
                (16, ls)
            } else if size == 0 {
                (8, bytes.len() - i)
            } else {
                (8, size)
            };
            if ty == b"mdat" {
                mdat_start_file = i + hdr_len;
                mdat_end_file = i + box_len;
                break;
            }
            i += box_len;
        }
    }
    assert!(mdat_end_file > mdat_start_file, "no mdat box found");

    // First sample offset (absolute file offset from stco). If we
    // can't find stco, assume samples start right after mdat header.
    let first_sample_off = first_chunk_offset(&bytes)
        .map(|o| o as usize)
        .unwrap_or(mdat_start_file);
    eprintln!(
        "mdat: {:#x}..{:#x}, first_sample={:#x}",
        mdat_start_file, mdat_end_file, first_sample_off
    );
    assert!(first_sample_off >= mdat_start_file && first_sample_off < mdat_end_file);

    // Build an Annex B byte-stream: first the SPS + PPS extracted from
    // avcC, then the first few length-prefixed NALs starting at the
    // first sample offset.
    let mut annex_b: Vec<u8> = Vec::new();
    for s in &sps_list {
        annex_b.extend_from_slice(&[0, 0, 0, 1]);
        annex_b.extend_from_slice(s);
    }
    for p in &pps_list {
        annex_b.extend_from_slice(&[0, 0, 0, 1]);
        annex_b.extend_from_slice(p);
    }
    let mut i = first_sample_off;
    let mut nals_emitted = 0usize;
    while i + length_size <= mdat_end_file && nals_emitted < 8 {
        let mut len = 0usize;
        for b in &bytes[i..i + length_size] {
            len = (len << 8) | (*b as usize);
        }
        i += length_size;
        if len == 0 || i + len > mdat_end_file {
            break;
        }
        annex_b.extend_from_slice(&[0, 0, 0, 1]);
        annex_b.extend_from_slice(&bytes[i..i + len]);
        i += len;
        nals_emitted += 1;
    }
    eprintln!("emitted {} NALs + SPS/PPS into Annex B", nals_emitted);
    assert!(nals_emitted > 0, "no NALs extracted from mdat payload");

    use oxideav_h264::decoder::{Decoder, Event};
    let mut dec = Decoder::new();
    let mut slices: Vec<(
        oxideav_h264::slice_header::SliceHeader,
        Vec<u8>,
        (usize, u8),
    )> = Vec::new();
    for ev in dec.process_annex_b(&annex_b) {
        match ev {
            Ok(Event::Slice {
                header,
                rbsp,
                slice_data_cursor,
                ..
            }) => {
                slices.push((header, rbsp, slice_data_cursor));
                if slices.len() >= 4 {
                    break;
                }
            }
            Ok(_) => {}
            Err(e) => eprintln!("decoder event err (non-fatal): {e}"),
        }
    }

    let sps = dec.active_sps().cloned().expect("active SPS");
    let pps = dec.active_pps().cloned().expect("active PPS");
    eprintln!(
        "SPS profile={} level={} mbaff={} frame_mbs_only={}",
        sps.profile_idc, sps.level_idc, sps.mb_adaptive_frame_field_flag, sps.frame_mbs_only_flag,
    );
    assert!(sps.mb_adaptive_frame_field_flag, "fixture must be MBAFF");
    assert!(!sps.frame_mbs_only_flag, "fixture must allow fields");

    assert!(!slices.is_empty(), "no slices extracted");

    let mut mbaff_rejection_count = 0usize;
    let mut other_errors = 0usize;
    let mut successes = 0usize;
    for (hdr, rbsp, (byte, bit)) in &slices {
        match slice_data::parse_slice_data(rbsp, *byte, *bit, hdr, &sps, &pps) {
            Ok(sd) => {
                successes += 1;
                assert_eq!(sd.macroblocks.len(), sd.mb_field_decoding_flags.len());
                eprintln!(
                    "slice parsed: {} MBs, {} flags",
                    sd.macroblocks.len(),
                    sd.mb_field_decoding_flags.len()
                );
            }
            Err(slice_data::SliceDataError::MbaffNotSupported) => {
                mbaff_rejection_count += 1;
            }
            Err(e) => {
                other_errors += 1;
                eprintln!("slice parse error (acceptable downstream): {e}");
            }
        }
    }

    eprintln!(
        "summary: successes={} other_errors={} mbaff_rejections={}",
        successes, other_errors, mbaff_rejection_count
    );

    assert_eq!(
        mbaff_rejection_count, 0,
        "Phase 1 must no longer reject this MBAFF clip with MbaffNotSupported"
    );
}
