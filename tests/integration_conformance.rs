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
//! ## Diagnostic output
//!
//! When a frame mismatches, the harness converts the raw byte offset
//! into actionable coordinates: plane, (x, y) within the plane, then
//! macroblock address + intra-MB offset using H.264 Rec. T-REC-H.264
//! (08/2024) terminology:
//!
//! - **Luma (Y)**: `mb_x = x / 16`, `mb_y = y / 16`,
//!   `mb_addr = mb_y * PicWidthInMbs + mb_x` (spec §6.4.1 "Inverse
//!   macroblock scanning process"), `(x_in_mb, y_in_mb) = (x % 16, y % 16)`.
//! - **Chroma (Cb/Cr) for ChromaArrayType=1 (4:2:0, MbWidthC=8,
//!   MbHeightC=8, spec §6.2 Table 6-1)**: chroma is half-resolution, so
//!   `mb_x = x / 8`, `mb_y = y / 8`, and the spec-equivalent intra-MB
//!   luma-space position is `(x_in_mb, y_in_mb) = ((x%8)*2, (y%8)*2)`
//!   (i.e. the top-left luma sample of the 2x2 block that chroma sample
//!   covers under SubWidthC=SubHeightC=2).
//!
//! The "first diverging MB dump" (gated on `OXIDEAV_H264_DUMP_FIRST_DIFF_MB=1`)
//! prints 16×16 grids of luma for the expected vs. our reconstruction,
//! which makes Agent A's pixel-bisection substantially easier.
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

// -------------------------- coordinate math --------------------------

/// Which plane a sample lives in (4:2:0 three-plane layout).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Plane {
    Y,
    Cb,
    Cr,
}

impl Plane {
    fn name(&self) -> &'static str {
        match self {
            Plane::Y => "Y",
            Plane::Cb => "Cb",
            Plane::Cr => "Cr",
        }
    }
}

/// Convert a byte offset within a `yuv420p` frame buffer to
/// `(plane, x, y)` within that plane, given the picture's luma geometry.
///
/// Layout assumption matches `videoframe_to_yuv420p` / ffmpeg
/// `-pix_fmt yuv420p`: contiguous Y (W*H) || Cb (W/2 * H/2) || Cr (W/2 * H/2),
/// no padding.
fn byte_offset_to_plane_xy(offset: usize, width: u32, height: u32) -> Option<(Plane, u32, u32)> {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let y_size = w * h;
    let c_size = cw * ch;

    if offset < y_size {
        let y = (offset / w) as u32;
        let x = (offset % w) as u32;
        Some((Plane::Y, x, y))
    } else if offset < y_size + c_size {
        let o = offset - y_size;
        let y = (o / cw) as u32;
        let x = (o % cw) as u32;
        Some((Plane::Cb, x, y))
    } else if offset < y_size + 2 * c_size {
        let o = offset - y_size - c_size;
        let y = (o / cw) as u32;
        let x = (o % cw) as u32;
        Some((Plane::Cr, x, y))
    } else {
        None
    }
}

/// Given a `(plane, x, y)` position and the picture's `PicWidthInMbs`
/// (spec §7.4.2.1.1), return `(mb_addr, mb_x_in_mb, mb_y_in_mb)` where
/// `(mb_x_in_mb, mb_y_in_mb)` is expressed in luma-sample units (i.e.
/// 0..16) so that Y/Cb/Cr intra-MB positions are directly comparable.
///
/// For Y (spec §6.4.1): `mb_x = x/16, mb_y = y/16, mb_addr = mb_y *
/// PicWidthInMbs + mb_x`, and the intra-MB offset is `(x%16, y%16)`.
///
/// For Cb/Cr with ChromaArrayType=1 (4:2:0, spec Table 6-1): chroma
/// samples are on a half-resolution grid (`MbWidthC=8, MbHeightC=8`), so
/// we divide by 8 to find the MB, and expand `(x%8, y%8)` back to
/// luma-sample coordinates by multiplying by `SubWidthC=SubHeightC=2`.
/// The resulting intra-MB luma-space position identifies the 2×2 luma
/// block that the chroma sample is co-sited with.
fn plane_xy_to_mb_coords(
    plane: Plane,
    x: u32,
    y: u32,
    pic_width_in_mbs: u32,
) -> (u32, u32, u32, u32, u32) {
    // Returns (mb_x, mb_y, mb_addr, x_in_mb, y_in_mb) in luma-sample units.
    match plane {
        Plane::Y => {
            let mb_x = x / 16;
            let mb_y = y / 16;
            let mb_addr = mb_y * pic_width_in_mbs + mb_x;
            (mb_x, mb_y, mb_addr, x % 16, y % 16)
        }
        Plane::Cb | Plane::Cr => {
            // ChromaArrayType=1 (4:2:0): chroma is half the luma resolution.
            let mb_x = x / 8;
            let mb_y = y / 8;
            let mb_addr = mb_y * pic_width_in_mbs + mb_x;
            // Expand intra-chroma-MB offset (0..8) back to luma-space (0..16).
            ((mb_x), (mb_y), mb_addr, (x % 8) * 2, (y % 8) * 2)
        }
    }
}

// ------------------------- comparison logic --------------------------

/// Per-plane divergence stats.
#[derive(Debug, Default, Clone)]
struct PlaneStats {
    /// Total samples inspected in this plane.
    total: usize,
    /// Samples that differed between ours and ffmpeg.
    differing: usize,
    /// Maximum absolute sample delta in this plane.
    max_abs: u8,
}

#[derive(Debug, Clone)]
struct FrameDiff {
    index: usize,
    /// Byte offset of the first differing sample, if any.
    first_diff_offset: Option<usize>,
    /// First diff expressed as (plane, x, y) — matches `first_diff_offset`.
    first_diff_plane_xy: Option<(Plane, u32, u32)>,
    /// First diff expressed as (mb_addr, x_in_mb, y_in_mb) in luma-space units.
    first_diff_mb: Option<(u32, u32, u32)>,
    total_bytes: usize,
    differing_bytes: usize,
    max_abs_diff: u8,
    /// Per-plane breakdown (Y, Cb, Cr).
    per_plane: [PlaneStats; 3],
}

/// Index helper for the per-plane array on `FrameDiff`.
fn plane_idx(p: Plane) -> usize {
    match p {
        Plane::Y => 0,
        Plane::Cb => 1,
        Plane::Cr => 2,
    }
}

fn compare_frame(
    idx: usize,
    ours: &[u8],
    theirs: &[u8],
    width: u32,
    height: u32,
) -> FrameDiff {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let y_size = w * h;
    let c_size = cw * ch;

    let mut per_plane: [PlaneStats; 3] = Default::default();
    per_plane[0].total = y_size;
    per_plane[1].total = c_size;
    per_plane[2].total = c_size;

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
            // Per-plane accounting.
            let pi = if i < y_size {
                0
            } else if i < y_size + c_size {
                1
            } else if i < y_size + 2 * c_size {
                2
            } else {
                // Out-of-layout byte; count it against Y so we don't lose it.
                0
            };
            per_plane[pi].differing += 1;
            if d > per_plane[pi].max_abs {
                per_plane[pi].max_abs = d;
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

    let first_plane_xy = first.and_then(|o| byte_offset_to_plane_xy(o, width, height));
    let first_mb = first_plane_xy.map(|(plane, x, y)| {
        let pic_width_in_mbs = (width + 15) / 16;
        let (_mb_x, _mb_y, mb_addr, xi, yi) =
            plane_xy_to_mb_coords(plane, x, y, pic_width_in_mbs);
        (mb_addr, xi, yi)
    });

    FrameDiff {
        index: idx,
        first_diff_offset: first,
        first_diff_plane_xy: first_plane_xy,
        first_diff_mb: first_mb,
        total_bytes: ours.len().max(theirs.len()),
        differing_bytes: differing,
        max_abs_diff: max_abs,
        per_plane,
    }
}

/// Dump a 16×16 luma-sample grid for MB `mb_addr` from a yuv420p frame
/// buffer. The MB origin in the Y plane is `(mb_x * 16, mb_y * 16)` and
/// we clip to the actual picture dimensions so edge MBs don't over-read.
fn dump_luma_mb(label: &str, buf: &[u8], mb_addr: u32, width: u32, height: u32) {
    let pic_width_in_mbs = (width + 15) / 16;
    let mb_y = mb_addr / pic_width_in_mbs;
    let mb_x = mb_addr % pic_width_in_mbs;
    let x0 = (mb_x * 16) as usize;
    let y0 = (mb_y * 16) as usize;
    let w = width as usize;
    let h = height as usize;
    eprintln!(
        "    {label} luma MB #{mb_addr} origin=({}, {}) 16x16:",
        x0, y0
    );
    for dy in 0..16usize {
        let y = y0 + dy;
        if y >= h {
            eprintln!("      <row clipped>");
            continue;
        }
        let mut line = String::with_capacity(16 * 4);
        for dx in 0..16usize {
            let x = x0 + dx;
            if x >= w {
                line.push_str(" ..");
            } else {
                let v = buf[y * w + x];
                line.push_str(&format!(" {:3}", v));
            }
        }
        eprintln!("     {}", line);
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
    /// Aggregate stats across all compared frames.
    total_differing_bytes: u64,
    #[allow(dead_code)]
    total_compared_bytes: u64,
    worst_max_abs_delta: u8,
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

    run_conformance_with_reference(name, path, &reference)
}

/// Same as `run_conformance` but takes a precomputed reference YUV
/// (raw 4:2:0 planar) rather than shelling out to ffmpeg. Used by
/// JVT-AI conformance vectors that ship their own reference output.
fn run_conformance_with_ref_file(
    name: &str,
    bitstream: &Path,
    ref_yuv: &Path,
) -> Option<StreamReport> {
    if !bitstream.exists() {
        eprintln!("[{name}] skip: bitstream not found at {}", bitstream.display());
        return None;
    }
    if !ref_yuv.exists() {
        eprintln!("[{name}] skip: reference YUV not found at {}", ref_yuv.display());
        return None;
    }
    eprintln!(
        "[{name}] starting conformance check: {} (ref: {})",
        bitstream.display(),
        ref_yuv.display()
    );
    let reference = match std::fs::read(ref_yuv) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[{name}] skip: failed to read reference YUV: {e}");
            return None;
        }
    };
    run_conformance_with_reference(name, bitstream, &reference)
}

fn run_conformance_with_reference(
    name: &str,
    path: &Path,
    reference: &[u8],
) -> Option<StreamReport> {

    // Decode through us.
    let (w, h, ours_frames) = decoder_yuv(path);
    let our_plane_bytes = (w as usize) * (h as usize) * 3 / 2;

    // Segment the reference dump into frames.
    let ffmpeg_frames = if our_plane_bytes == 0 {
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

    let dump_first_diff_mb = std::env::var("OXIDEAV_H264_DUMP_FIRST_DIFF_MB")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let mut matched = 0usize;
    let mut first_mismatch: Option<FrameDiff> = None;
    let mut dumped_mb = false;
    let mut total_differing_bytes: u64 = 0;
    let mut total_compared_bytes: u64 = 0;
    let mut worst_max_abs: u8 = 0;
    let compare_n = ours_frames.len().min(ffmpeg_frames);
    for i in 0..compare_n {
        let theirs = &reference[i * our_plane_bytes..(i + 1) * our_plane_bytes];
        let diff = compare_frame(i, &ours_frames[i], theirs, w, h);
        total_differing_bytes += diff.differing_bytes as u64;
        total_compared_bytes += diff.total_bytes as u64;
        if diff.max_abs_diff > worst_max_abs {
            worst_max_abs = diff.max_abs_diff;
        }
        if diff.differing_bytes == 0 {
            matched += 1;
        } else {
            // Enhanced per-frame diagnostic (emitted for every mismatching frame).
            print_frame_diff(name, &diff, w);
            if first_mismatch.is_none() {
                first_mismatch = Some(diff.clone());
                // Optional 16×16 luma dump for the first mismatching MB.
                if dump_first_diff_mb && !dumped_mb {
                    if let Some((mb_addr, _, _)) = diff.first_diff_mb {
                        eprintln!(
                            "[{name}] OXIDEAV_H264_DUMP_FIRST_DIFF_MB: dumping luma MB #{mb_addr} for frame {}",
                            i
                        );
                        dump_luma_mb("expected (ffmpeg)", theirs, mb_addr, w, h);
                        dump_luma_mb("ours", &ours_frames[i], mb_addr, w, h);
                        dumped_mb = true;
                    }
                }
            }
        }
    }

    Some(StreamReport {
        name: name.to_string(),
        ours_frames: ours_frames.len(),
        ffmpeg_frames,
        matched,
        first_mismatch,
        geometry: if w > 0 && h > 0 { Some((w, h)) } else { None },
        total_differing_bytes,
        total_compared_bytes,
        worst_max_abs_delta: worst_max_abs,
    })
}

/// Print the enhanced per-frame diagnostic: plane, (x, y), MB address,
/// per-plane stats. Example:
///
/// ```text
///   frame 0: 37963 of 38016 bytes differ, max |Δ|=252
///     first diff: Y plane at (0, 0) = MB #0 sample (0, 0)
///     per-plane: Y 24782/25344 max Δ 252 | Cb 6615/6336 max Δ 215 | Cr 6566/6336 max Δ 214
/// ```
fn print_frame_diff(name: &str, d: &FrameDiff, width: u32) {
    eprintln!(
        "[{name}] frame {}: {} of {} bytes differ, max |Δ|={}",
        d.index, d.differing_bytes, d.total_bytes, d.max_abs_diff
    );
    if let (Some((plane, x, y)), Some((mb_addr, xi, yi))) =
        (d.first_diff_plane_xy, d.first_diff_mb)
    {
        let pic_width_in_mbs = (width + 15) / 16;
        let mb_y = mb_addr / pic_width_in_mbs;
        let mb_x = mb_addr % pic_width_in_mbs;
        eprintln!(
            "  first diff: {} plane at ({}, {}) = MB #{} ({}, {}) sample ({}, {})",
            plane.name(),
            x,
            y,
            mb_addr,
            mb_x,
            mb_y,
            xi,
            yi
        );
    } else if let Some(off) = d.first_diff_offset {
        eprintln!("  first diff: byte offset {}", off);
    }
    eprintln!(
        "  per-plane: Y {}/{} max Δ {} | Cb {}/{} max Δ {} | Cr {}/{} max Δ {}",
        d.per_plane[0].differing,
        d.per_plane[0].total,
        d.per_plane[0].max_abs,
        d.per_plane[1].differing,
        d.per_plane[1].total,
        d.per_plane[1].max_abs,
        d.per_plane[2].differing,
        d.per_plane[2].total,
        d.per_plane[2].max_abs,
    );
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
        let off_str = d
            .first_diff_offset
            .map(|o| o.to_string())
            .unwrap_or_else(|| "?".into());
        eprintln!(
            "  first mismatch: frame #{}  first-diff-byte={}  differing={}B/{}B  max|Δ|={}",
            d.index, off_str, d.differing_bytes, d.total_bytes, d.max_abs_diff,
        );
        if let (Some((plane, x, y)), Some((mb_addr, xi, yi))) =
            (d.first_diff_plane_xy, d.first_diff_mb)
        {
            let pic_width_in_mbs = r.geometry.map(|(w, _)| (w + 15) / 16).unwrap_or(0);
            let mb_y = if pic_width_in_mbs > 0 {
                mb_addr / pic_width_in_mbs
            } else {
                0
            };
            let mb_x = if pic_width_in_mbs > 0 {
                mb_addr % pic_width_in_mbs
            } else {
                0
            };
            eprintln!(
                "    first diff at: {} plane ({}, {}) = MB #{} ({}, {}) sample ({}, {})",
                plane.name(),
                x,
                y,
                mb_addr,
                mb_x,
                mb_y,
                xi,
                yi
            );
        }
        eprintln!(
            "    per-plane: Y {}/{} max Δ {} | Cb {}/{} max Δ {} | Cr {}/{} max Δ {}",
            d.per_plane[0].differing,
            d.per_plane[0].total,
            d.per_plane[0].max_abs,
            d.per_plane[1].differing,
            d.per_plane[1].total,
            d.per_plane[1].max_abs,
            d.per_plane[2].differing,
            d.per_plane[2].total,
            d.per_plane[2].max_abs,
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
    eprintln!("======================================= conformance summary =======================================");
    eprintln!(
        "  {:<22} {:>6} {:>6} {:>7} {:>12} {:>8}  {:<30}",
        "stream", "ours", "ffmpeg", "exact", "avg-diff-B/f", "max|Δ|", "first-diff (plane@x,y MB#n)"
    );
    for r in &reports {
        let compare_n = r.ours_frames.min(r.ffmpeg_frames);
        let avg = if compare_n > 0 {
            r.total_differing_bytes / compare_n as u64
        } else {
            0
        };
        let first_str = match (&r.first_mismatch, r.geometry) {
            (Some(d), Some((w, _))) => {
                let pic_width_in_mbs = (w + 15) / 16;
                if let (Some((plane, x, y)), Some((mb_addr, _, _))) =
                    (d.first_diff_plane_xy, d.first_diff_mb)
                {
                    let mb_x = mb_addr % pic_width_in_mbs;
                    let mb_y = mb_addr / pic_width_in_mbs;
                    format!(
                        "f#{} {}@({},{}) MB#{}({},{})",
                        d.index,
                        plane.name(),
                        x,
                        y,
                        mb_addr,
                        mb_x,
                        mb_y
                    )
                } else {
                    format!("f#{}", d.index)
                }
            }
            _ => "—".to_string(),
        };
        eprintln!(
            "  {:<22} {:>6} {:>6} {:>3}/{:<3} {:>12} {:>8}  {:<30}",
            r.name,
            r.ours_frames,
            r.ffmpeg_frames,
            r.matched,
            compare_n,
            avg,
            r.worst_max_abs_delta,
            first_str,
        );
    }
    eprintln!("===================================================================================================");
}

// ------------------------------ JVT conformance vectors ----------------

/// JVT AVCv1 `AUD_MW_E` conformance vector (Mcubeworks, Baseline / CAVLC,
/// QCIF 176x144, 100 frames of foreman, IDR-only, Access Unit Delimiters).
/// Ships with a reference YUV in the zip under `AUD_MW_E_rec.qcif`.
/// After extracting the zip to /tmp/aud_mw_e/ (or setting the env vars),
/// this test compares our decode against the reference byte-by-byte.
#[test]
fn conformance_jvt_aud_mw_e() {
    let bs = std::env::var("OXIDEAV_JVT_AUD_MW_E_BS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/aud_mw_e/AUD_MW_E.264"));
    let ref_yuv = std::env::var("OXIDEAV_JVT_AUD_MW_E_REF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/aud_mw_e/AUD_MW_E_rec.qcif"));
    let Some(report) = run_conformance_with_ref_file("jvt_AUD_MW_E", &bs, &ref_yuv) else {
        return;
    };
    print_report(&report);
    assert_eq!(
        report.ours_frames, report.ffmpeg_frames,
        "frame-count mismatch: ours={} ref={}",
        report.ours_frames, report.ffmpeg_frames
    );
}

/// JVT AVCv1 `camp_mot_frm0_full` (Motorola, Main / CABAC, 720x480
/// 4:2:0, 30 frames of Mobile, IBBP, frame-coded). Reference YUV at
/// `camp_mot_frm0_full_rec.yuv`. Only IDR + frame-coded so should
/// exercise the CABAC decode path without PAFF / MBAFF complications.
#[test]
fn conformance_jvt_camp_mot_frm0_full() {
    let bs = std::env::var("OXIDEAV_JVT_CAMP_MOT_FRM0_BS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/cabac_mot_frm0/camp_mot_frm0_full.26l"));
    let ref_yuv = std::env::var("OXIDEAV_JVT_CAMP_MOT_FRM0_REF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/cabac_mot_frm0/camp_mot_frm0_full_rec.yuv"));
    let Some(report) = run_conformance_with_ref_file("jvt_camp_mot_frm0", &bs, &ref_yuv) else {
        return;
    };
    print_report(&report);
    // Soft: frame counts may mismatch while the decoder's B-frame
    // reordering / output path matures.
}

// ------------------------------ helpers tests --------------------------

#[cfg(test)]
mod helpers_tests {
    use super::*;

    // ------- byte_offset_to_plane_xy -------

    #[test]
    fn byte_offset_zero_is_y_origin() {
        let (p, x, y) = byte_offset_to_plane_xy(0, 176, 144).unwrap();
        assert_eq!(p, Plane::Y);
        assert_eq!((x, y), (0, 0));
    }

    #[test]
    fn byte_offset_one_row_into_y() {
        // Byte offset = width => (0, 1) in Y plane.
        let (p, x, y) = byte_offset_to_plane_xy(176, 176, 144).unwrap();
        assert_eq!(p, Plane::Y);
        assert_eq!((x, y), (0, 1));
    }

    #[test]
    fn byte_offset_last_y_sample() {
        // Last sample of Y plane = (175, 143) for 176x144.
        let off = 176 * 144 - 1;
        let (p, x, y) = byte_offset_to_plane_xy(off, 176, 144).unwrap();
        assert_eq!(p, Plane::Y);
        assert_eq!((x, y), (175, 143));
    }

    #[test]
    fn byte_offset_cb_origin() {
        // First byte after Y plane = Cb origin.
        let y_size = 176 * 144;
        let (p, x, y) = byte_offset_to_plane_xy(y_size, 176, 144).unwrap();
        assert_eq!(p, Plane::Cb);
        assert_eq!((x, y), (0, 0));
    }

    #[test]
    fn byte_offset_cr_origin() {
        // First byte after Y+Cb = Cr origin.
        let w = 176usize;
        let h = 144usize;
        let y_size = w * h;
        let c_size = (w / 2) * (h / 2);
        let (p, x, y) = byte_offset_to_plane_xy(y_size + c_size, 176, 144).unwrap();
        assert_eq!(p, Plane::Cr);
        assert_eq!((x, y), (0, 0));
    }

    #[test]
    fn byte_offset_last_cr_sample() {
        let w = 176usize;
        let h = 144usize;
        let cw = w / 2;
        let ch = h / 2;
        let last = w * h + 2 * cw * ch - 1;
        let (p, x, y) = byte_offset_to_plane_xy(last, 176, 144).unwrap();
        assert_eq!(p, Plane::Cr);
        assert_eq!((x, y), ((cw - 1) as u32, (ch - 1) as u32));
    }

    #[test]
    fn byte_offset_out_of_bounds_returns_none() {
        let w = 176usize;
        let h = 144usize;
        let total = w * h + 2 * (w / 2) * (h / 2);
        assert!(byte_offset_to_plane_xy(total, 176, 144).is_none());
    }

    // ------- plane_xy_to_mb_coords -------

    #[test]
    fn luma_origin_is_mb0() {
        // (Y, 0, 0) in 176x144 (PicWidthInMbs=11) => MB #0, intra (0,0).
        let (mb_x, mb_y, addr, xi, yi) = plane_xy_to_mb_coords(Plane::Y, 0, 0, 11);
        assert_eq!((mb_x, mb_y, addr, xi, yi), (0, 0, 0, 0, 0));
    }

    #[test]
    fn luma_next_mb_right() {
        // x=16, y=0 => MB #1 (one MB to the right), intra (0,0).
        let (mb_x, mb_y, addr, xi, yi) = plane_xy_to_mb_coords(Plane::Y, 16, 0, 11);
        assert_eq!((mb_x, mb_y, addr), (1, 0, 1));
        assert_eq!((xi, yi), (0, 0));
    }

    #[test]
    fn luma_second_row_of_mbs() {
        // x=0, y=16 => MB #PicWidthInMbs (first MB of second MB row).
        let (mb_x, mb_y, addr, xi, yi) = plane_xy_to_mb_coords(Plane::Y, 0, 16, 11);
        assert_eq!((mb_x, mb_y, addr), (0, 1, 11));
        assert_eq!((xi, yi), (0, 0));
    }

    #[test]
    fn luma_inside_mb_offset() {
        // x=17, y=18 => MB (1, 1) = addr 12, intra (1, 2).
        let (mb_x, mb_y, addr, xi, yi) = plane_xy_to_mb_coords(Plane::Y, 17, 18, 11);
        assert_eq!((mb_x, mb_y, addr), (1, 1, 12));
        assert_eq!((xi, yi), (1, 2));
    }

    #[test]
    fn chroma_origin_is_mb0_luma_space() {
        // (Cb, 0, 0) => MB #0, intra (0, 0) in luma-space.
        let (mb_x, mb_y, addr, xi, yi) = plane_xy_to_mb_coords(Plane::Cb, 0, 0, 11);
        assert_eq!((mb_x, mb_y, addr), (0, 0, 0));
        assert_eq!((xi, yi), (0, 0));
    }

    #[test]
    fn chroma_intra_mb_offset_scaled() {
        // 4:2:0: chroma (3, 2) is at chroma-space (3, 2) inside MB0's
        // 8x8 chroma block, expanded to luma-space (6, 4).
        let (mb_x, mb_y, addr, xi, yi) = plane_xy_to_mb_coords(Plane::Cb, 3, 2, 11);
        assert_eq!((mb_x, mb_y, addr), (0, 0, 0));
        assert_eq!((xi, yi), (6, 4));
    }

    #[test]
    fn chroma_second_mb_column() {
        // (Cr, 8, 0) => MB #1, intra (0, 0).
        let (mb_x, mb_y, addr, xi, yi) = plane_xy_to_mb_coords(Plane::Cr, 8, 0, 11);
        assert_eq!((mb_x, mb_y, addr), (1, 0, 1));
        assert_eq!((xi, yi), (0, 0));
    }

    #[test]
    fn chroma_at_edge_of_mb0_block() {
        // (Cb, 7, 7) is last chroma sample inside MB0's 8×8 chroma block
        // — luma-space intra = (14, 14) (co-sited with the top-left of
        // the bottom-right 2×2 luma block).
        let (mb_x, mb_y, addr, xi, yi) = plane_xy_to_mb_coords(Plane::Cb, 7, 7, 11);
        assert_eq!((mb_x, mb_y, addr), (0, 0, 0));
        assert_eq!((xi, yi), (14, 14));
    }

    // ------- combined: offset -> plane_xy -> mb_coords -------

    #[test]
    fn offset_in_second_mb_row_maps_correctly() {
        // For 176x144 (PicWidthInMbs=11), byte offset 176 * 16 (first
        // sample of the 2nd MB row) => Y plane (0, 16) => MB #11.
        let off = 176 * 16;
        let (p, x, y) = byte_offset_to_plane_xy(off, 176, 144).unwrap();
        assert_eq!(p, Plane::Y);
        let (_mx, _my, addr, xi, yi) = plane_xy_to_mb_coords(p, x, y, 11);
        assert_eq!(addr, 11);
        assert_eq!((xi, yi), (0, 0));
    }

    #[test]
    fn compare_frame_per_plane_stats() {
        // 8x4 luma, 4x2 chroma each. Construct ours and theirs with
        // deliberate differences in each plane and verify per-plane
        // accounting.
        let w: u32 = 8;
        let h: u32 = 4;
        let y_size = (w * h) as usize;
        let c_size = ((w / 2) * (h / 2)) as usize;
        let total = y_size + 2 * c_size;

        let mut ours = vec![0u8; total];
        let mut theirs = vec![0u8; total];

        // Y diffs: 2 samples, max 10.
        ours[0] = 10;
        theirs[0] = 0; // Δ=10
        ours[5] = 7;
        theirs[5] = 2; // Δ=5

        // Cb diffs: 1 sample, Δ=3.
        ours[y_size + 1] = 3;

        // Cr diffs: 3 samples, max 20.
        ours[y_size + c_size + 0] = 1;
        ours[y_size + c_size + 2] = 20;
        ours[y_size + c_size + 3] = 4;

        let d = compare_frame(0, &ours, &theirs, w, h);
        assert_eq!(d.differing_bytes, 6);
        assert_eq!(d.max_abs_diff, 20);
        assert_eq!(d.per_plane[0].total, y_size);
        assert_eq!(d.per_plane[1].total, c_size);
        assert_eq!(d.per_plane[2].total, c_size);
        assert_eq!(d.per_plane[0].differing, 2);
        assert_eq!(d.per_plane[0].max_abs, 10);
        assert_eq!(d.per_plane[1].differing, 1);
        assert_eq!(d.per_plane[1].max_abs, 3);
        assert_eq!(d.per_plane[2].differing, 3);
        assert_eq!(d.per_plane[2].max_abs, 20);
        // First diff must be at offset 0 (Y plane).
        assert_eq!(d.first_diff_offset, Some(0));
        assert_eq!(d.first_diff_plane_xy, Some((Plane::Y, 0, 0)));
        assert_eq!(d.first_diff_mb, Some((0, 0, 0)));
    }

    #[test]
    fn plane_idx_maps_correctly() {
        assert_eq!(plane_idx(Plane::Y), 0);
        assert_eq!(plane_idx(Plane::Cb), 1);
        assert_eq!(plane_idx(Plane::Cr), 2);
    }
}
