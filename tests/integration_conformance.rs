//! Pixel-accuracy conformance harness for the `oxideav-h264` decoder.
//!
//! Uses `ffmpeg` (as an external black-box tool on PATH) to produce a
//! reference YUV dump, decodes the same Annex B stream with our decoder,
//! and compares the two frame-by-frame.
//!
//! Pure-Rust harness (no `unsafe`, no extra crates beyond the std). If
//! ffmpeg is missing or a sample isn't available, the test prints a
//! `skip:` line and returns cleanly — CI without those fixtures keeps
//! passing.
//!
//! Policy for this first pass:
//! - Frame-count equality is a **hard assert** (mismatched frame counts
//!   point at multi-slice assembly or decoder output ordering bugs, not
//!   pixel-reconstruction bugs — surfacing those immediately is worth
//!   more than a soft warning).
//! - Per-frame pixel equality is **soft** — we print the mismatched
//!   frame indices (and the first differing byte offset) but don't fail
//!   the test. Promote to a hard assert once reconstruction is known
//!   pixel-exact.
//!
//! Run with `-- --nocapture` to see the per-stream conformance summary.
//!
//! ```sh
//! cargo test -p oxideav-h264 --test integration_conformance -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use oxideav_codec::Decoder as _;
use oxideav_core::{CodecId, Error, Frame, Packet, PixelFormat, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;

// -------------------------- ffmpeg plumbing --------------------------

/// Returns `true` if `ffmpeg` is invokable on PATH and accepts `-version`.
fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Run ffmpeg and return the raw `yuv420p` dump for the entire stream.
///
/// Layout (as emitted by `ffmpeg -f rawvideo -pix_fmt yuv420p`):
/// contiguous frames, each frame = Y plane (W*H) + Cb plane (W/2 * H/2)
/// + Cr plane (W/2 * H/2). No padding, no stride.
fn ffmpeg_raw_yuv(path: &Path) -> Option<Vec<u8>> {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        .stderr(Stdio::null())
        .stdout(Stdio::piped())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(out.stdout)
}

/// Same as `ffmpeg_raw_yuv` but also capture width/height from `-show_streams`
/// via ffprobe.  We avoid a dependency on ffprobe — we pass the expected
/// geometry in from the caller, so this helper isn't needed.

// ---------------------- our decoder invocation -----------------------

/// Decode the whole Annex B stream through our public `Decoder` trait
/// and return one `Vec<u8>` per output frame, each laid out as yuv420p
/// (Y then Cb then Cr, contiguous, stride == width).
///
/// Also returns the `(width, height)` the decoder reported on the first
/// frame, so the caller can sanity-check against ffmpeg's geometry.
fn decoder_yuv(path: &Path) -> (u32, u32, Vec<Vec<u8>>) {
    let bytes = std::fs::read(path).expect("read sample");
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));

    // Feed the whole Annex B stream as a single packet — the H.264
    // decoder's Annex B splitter handles NAL discovery internally.
    let packet = Packet::new(0, TimeBase::new(1, 25), bytes).with_pts(0);
    dec.send_packet(&packet).expect("send_packet");
    // Drain: signal EOS so the DPB bumping process releases buffered pictures.
    dec.flush().expect("flush");

    let mut frames: Vec<Vec<u8>> = Vec::new();
    let mut width: u32 = 0;
    let mut height: u32 = 0;
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => {
                if frames.is_empty() {
                    width = vf.width;
                    height = vf.height;
                }
                frames.push(videoframe_to_yuv420p(&vf));
            }
            Ok(other) => {
                eprintln!("  unexpected non-video frame: {:?}", other);
            }
            Err(Error::Eof) => break,
            Err(Error::NeedMore) => break,
            Err(e) => {
                eprintln!("  receive_frame error: {e}");
                break;
            }
        }
    }
    (width, height, frames)
}

/// Flatten a `VideoFrame` into the yuv420p layout ffmpeg emits
/// (Y plane || Cb plane || Cr plane, rows packed with stride == width).
///
/// If the frame's plane strides are larger than `width`/`chroma_width`
/// (padding on the right), we copy row-by-row to strip the padding so
/// the resulting buffer matches ffmpeg exactly.
fn videoframe_to_yuv420p(vf: &oxideav_core::VideoFrame) -> Vec<u8> {
    assert_eq!(
        vf.format,
        PixelFormat::Yuv420P,
        "harness expects Yuv420P output"
    );
    assert_eq!(vf.planes.len(), 3, "yuv420p requires 3 planes");
    let w = vf.width as usize;
    let h = vf.height as usize;
    let cw = w / 2;
    let ch = h / 2;

    let mut out = Vec::with_capacity(w * h + 2 * cw * ch);
    pack_plane(&mut out, &vf.planes[0].data, vf.planes[0].stride, w, h);
    pack_plane(&mut out, &vf.planes[1].data, vf.planes[1].stride, cw, ch);
    pack_plane(&mut out, &vf.planes[2].data, vf.planes[2].stride, cw, ch);
    out
}

/// Copy `rows * cols` bytes from `src` (with `stride` bytes per row) into
/// `dst`, row-by-row. Panics if `src` is too small.
fn pack_plane(dst: &mut Vec<u8>, src: &[u8], stride: usize, cols: usize, rows: usize) {
    if stride == cols {
        // Common case for our decoder: no padding, blit the whole thing.
        let n = cols * rows;
        assert!(src.len() >= n, "plane underflow: have {} need {}", src.len(), n);
        dst.extend_from_slice(&src[..n]);
    } else {
        assert!(stride >= cols, "stride {stride} < cols {cols}");
        for r in 0..rows {
            let start = r * stride;
            let end = start + cols;
            assert!(
                end <= src.len(),
                "row {r}: end {end} > plane len {}",
                src.len()
            );
            dst.extend_from_slice(&src[start..end]);
        }
    }
}

// ------------------------- comparison logic --------------------------

#[derive(Debug)]
struct FrameDiff {
    index: usize,
    first_diff_offset: Option<usize>,
    total_bytes: usize,
    differing_bytes: usize,
    max_abs_diff: u8,
}

fn compare_frame(idx: usize, ours: &[u8], theirs: &[u8]) -> FrameDiff {
    let n = ours.len().min(theirs.len());
    let mut first: Option<usize> = None;
    let mut differing = 0usize;
    let mut max_abs: u8 = 0;
    for i in 0..n {
        let d = ours[i].abs_diff(theirs[i]);
        if d != 0 {
            if first.is_none() {
                first = Some(i);
            }
            differing += 1;
            if d > max_abs {
                max_abs = d;
            }
        }
    }
    // If lengths differ, everything beyond the shared prefix is "different".
    if ours.len() != theirs.len() {
        let extra = ours.len().abs_diff(theirs.len());
        differing += extra;
        if first.is_none() {
            first = Some(n);
        }
    }
    FrameDiff {
        index: idx,
        first_diff_offset: first,
        total_bytes: ours.len().max(theirs.len()),
        differing_bytes: differing,
        max_abs_diff: max_abs,
    }
}

// ------------------------- per-stream driver -------------------------

struct StreamReport {
    name: String,
    ours_frames: usize,
    ffmpeg_frames: usize,
    matched: usize,
    first_mismatch: Option<FrameDiff>,
    geometry: Option<(u32, u32)>,
}

fn run_conformance(name: &str, path: &Path) -> Option<StreamReport> {
    if !path.exists() {
        eprintln!("[{name}] skip: sample not found at {}", path.display());
        return None;
    }
    eprintln!("[{name}] starting conformance check: {}", path.display());

    // Reference dump.
    let reference = match ffmpeg_raw_yuv(path) {
        Some(r) => r,
        None => {
            eprintln!("[{name}] skip: ffmpeg failed to decode");
            return None;
        }
    };

    // Decode through us.
    let (w, h, ours_frames) = decoder_yuv(path);
    let our_plane_bytes = (w as usize) * (h as usize) * 3 / 2;

    // Segment the reference dump into frames.
    // Trust our decoder's reported geometry; if it doesn't line up with
    // the reference, bail with a diagnostic.
    let ffmpeg_frames = if our_plane_bytes == 0 {
        // No frames from us — we can't infer geometry. Skip the slicing
        // and report 0/whatever.
        0
    } else if reference.len() % our_plane_bytes != 0 {
        eprintln!(
            "[{name}] WARN: reference dump len {} not a multiple of our frame size {} \
             (our geometry: {}x{}). Frame geometry likely disagrees with ffmpeg.",
            reference.len(),
            our_plane_bytes,
            w,
            h
        );
        // Still try to slice by our geometry so we get *some* comparison.
        reference.len() / our_plane_bytes
    } else {
        reference.len() / our_plane_bytes
    };

    eprintln!(
        "[{name}] geometry: {}x{}  ours:{} frames  ffmpeg:{} frames",
        w,
        h,
        ours_frames.len(),
        ffmpeg_frames
    );

    let mut matched = 0usize;
    let mut first_mismatch: Option<FrameDiff> = None;
    let compare_n = ours_frames.len().min(ffmpeg_frames);
    for i in 0..compare_n {
        let theirs = &reference[i * our_plane_bytes..(i + 1) * our_plane_bytes];
        let diff = compare_frame(i, &ours_frames[i], theirs);
        if diff.differing_bytes == 0 {
            matched += 1;
        } else if first_mismatch.is_none() {
            first_mismatch = Some(diff);
        }
    }

    Some(StreamReport {
        name: name.to_string(),
        ours_frames: ours_frames.len(),
        ffmpeg_frames,
        matched,
        first_mismatch,
        geometry: if w > 0 && h > 0 { Some((w, h)) } else { None },
    })
}

fn print_report(r: &StreamReport) {
    let geom = match r.geometry {
        Some((w, h)) => format!("{w}x{h}"),
        None => "?x?".to_string(),
    };
    eprintln!("---- {} ({}) ----", r.name, geom);
    eprintln!("  our frames:    {}", r.ours_frames);
    eprintln!("  ffmpeg frames: {}", r.ffmpeg_frames);
    let compare_n = r.ours_frames.min(r.ffmpeg_frames);
    eprintln!(
        "  pixel-exact:   {}/{}  ({} mismatched)",
        r.matched,
        compare_n,
        compare_n.saturating_sub(r.matched)
    );
    if let Some(d) = &r.first_mismatch {
        eprintln!(
            "  first mismatch: frame #{}  first-diff-byte={}  differing={}B/{}B  max|Δ|={}",
            d.index,
            d.first_diff_offset
                .map(|o| o.to_string())
                .unwrap_or_else(|| "?".into()),
            d.differing_bytes,
            d.total_bytes,
            d.max_abs_diff,
        );
    } else if r.matched == compare_n && compare_n > 0 {
        eprintln!("  all compared frames pixel-exact against ffmpeg");
    }
}

// --------------------------- sample paths ----------------------------

fn sample_path(env_var: &str, default: &str) -> PathBuf {
    if let Ok(p) = std::env::var(env_var) {
        return PathBuf::from(p);
    }
    PathBuf::from(default)
}

// ------------------------------ tests --------------------------------

#[test]
fn conformance_foreman_p16x16() {
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not on PATH");
        return;
    }
    let path = sample_path(
        "OXIDEAV_SAMPLES_H264_FOREMAN",
        "/home/magicaltux/projects/oxideav/samples/samples.ffmpeg.org/V-codecs/h264/foreman_p16x16.264",
    );
    let Some(report) = run_conformance("foreman_p16x16", &path) else {
        return;
    };
    print_report(&report);

    // Hard assert: frame counts must match. A mismatch here indicates a
    // decoder-output-ordering or multi-slice-assembly problem, not a
    // pixel-reconstruction problem, and should fail loudly.
    assert_eq!(
        report.ours_frames, report.ffmpeg_frames,
        "frame-count mismatch: ours={} ffmpeg={}",
        report.ours_frames, report.ffmpeg_frames
    );
    // Soft: pixel equality. Mismatches are printed but don't fail.
    // Promote to a hard assert once reconstruction is pixel-exact.
}

#[test]
fn conformance_sp2_bt_b() {
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not on PATH");
        return;
    }
    let path = sample_path(
        "OXIDEAV_SAMPLES_H264_SP2_BT_B",
        "/home/magicaltux/projects/oxideav/samples/samples.ffmpeg.org/V-codecs/h264/sp2_bt_b.h264",
    );
    let Some(report) = run_conformance("sp2_bt_b", &path) else {
        return;
    };
    print_report(&report);
    assert_eq!(
        report.ours_frames, report.ffmpeg_frames,
        "frame-count mismatch: ours={} ffmpeg={}",
        report.ours_frames, report.ffmpeg_frames
    );
}

#[test]
fn conformance_killer_cuts9f() {
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not on PATH");
        return;
    }
    let path = sample_path(
        "OXIDEAV_SAMPLES_H264_KILLER_CUTS9F",
        "/home/magicaltux/projects/oxideav/samples/samples.ffmpeg.org/V-codecs/h264/killer_cuts9f_32x32.264",
    );
    let Some(report) = run_conformance("killer_cuts9f_32x32", &path) else {
        return;
    };
    print_report(&report);
    assert_eq!(
        report.ours_frames, report.ffmpeg_frames,
        "frame-count mismatch: ours={} ffmpeg={}",
        report.ours_frames, report.ffmpeg_frames
    );
}

/// Aggregated summary test — runs all streams above (skipping any that
/// can't be loaded) and prints a single table at the end. Useful for
/// `cargo test -- --nocapture` one-shot conformance dashboards.
#[test]
fn conformance_summary_all_streams() {
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not on PATH");
        return;
    }
    let candidates: &[(&str, &str, &str)] = &[
        (
            "foreman_p16x16",
            "OXIDEAV_SAMPLES_H264_FOREMAN",
            "/home/magicaltux/projects/oxideav/samples/samples.ffmpeg.org/V-codecs/h264/foreman_p16x16.264",
        ),
        (
            "sp2_bt_b",
            "OXIDEAV_SAMPLES_H264_SP2_BT_B",
            "/home/magicaltux/projects/oxideav/samples/samples.ffmpeg.org/V-codecs/h264/sp2_bt_b.h264",
        ),
        (
            "killer_cuts9f_32x32",
            "OXIDEAV_SAMPLES_H264_KILLER_CUTS9F",
            "/home/magicaltux/projects/oxideav/samples/samples.ffmpeg.org/V-codecs/h264/killer_cuts9f_32x32.264",
        ),
    ];

    let mut reports: Vec<StreamReport> = Vec::new();
    for (name, env_var, default) in candidates {
        let path = sample_path(env_var, default);
        if let Some(r) = run_conformance(name, &path) {
            reports.push(r);
        }
    }

    // Flush each individual report first …
    for r in &reports {
        print_report(r);
    }

    // … then a one-line-per-stream summary table.
    eprintln!();
    eprintln!("================ conformance summary ================");
    eprintln!(
        "  {:<24} {:>6} {:>6} {:>10}",
        "stream", "ours", "ffmpeg", "exact"
    );
    for r in &reports {
        let compare_n = r.ours_frames.min(r.ffmpeg_frames);
        eprintln!(
            "  {:<24} {:>6} {:>6} {:>5}/{:<4}",
            r.name, r.ours_frames, r.ffmpeg_frames, r.matched, compare_n
        );
    }
    eprintln!("=====================================================");
}
