//! Integration tests against the docs/video/h264/ fixture corpus.
//!
//! Each fixture under `../../docs/video/h264/fixtures/<name>/` carries
//! an `input.h264` (Annex-B byte stream — every fixture has one) plus a
//! per-fixture `expected.yuv` byte-for-byte ground truth produced by
//! libavcodec's H.264 decoder. One fixture
//! (`iso-mp4-vs-annexb`) additionally ships an `input.mp4` whose
//! length-prefixed elementary stream is byte-identical to the .h264
//! Annex-B variant — both framing branches in
//! [`H264CodecDecoder::send_packet`] are exercised.
//!
//! For every fixture we drive [`H264CodecDecoder`], drain all output
//! frames, repack them into the contiguous yuv420p / yuv444p layout
//! that ffmpeg used for `expected.yuv`, and report per-plane match
//! statistics.
//!
//! For the MP4 fixture (`iso-mp4-vs-annexb`) we additionally run the
//! length-prefixed AVC path, parsing the container with an inline
//! single-purpose ISO BMFF walker (avoids a dev-dep on `oxideav-mp4`,
//! which would couple this crate's published version to a crate it
//! does not need at runtime — see the comment by `parse_minimal_mp4_avc`).
//!
//! Per-fixture classification:
//! * [`Tier::BitExact`]   — must round-trip exactly. Failure = CI red.
//! * [`Tier::ReportOnly`] — currently divergent (or not yet decodable);
//!   logged but not asserted. Each carries an inline `TODO(h264-corpus)`
//!   tag explaining why so the underlying decoder bug stays grep-able.
//!
//! Per-fixture `bytes_per_sample` selects how the comparison reads
//! the reference: `1` for 8-bit yuv420p / yuv444p, `2` for 10/12-bit
//! yuv420p10le / yuv422p10le / yuv444p10le (LE u16 per sample).
//! `picture_to_video_frame` emits matching layouts based on the SPS
//! `bit_depth_luma` (clamped to u8 when `bit_depth_luma == 8`,
//! else packed as little-endian u16 clamped to `(1 << bit_depth) - 1`).
//!
//! The trace.txt files under each fixture directory are not consumed by
//! this driver; they exist so anyone bisecting a divergence can `diff`
//! their decoder's per-NAL / per-MB events against ffmpeg's. The trace
//! event vocabulary is documented at
//! `docs/video/h264/h264-fixtures-and-traces.md`.
//!
//! Spec references for the test logic:
//! * Annex B — Byte stream format. NALs delimited by `00 00 01` or
//!   `00 00 00 01` start codes; emulation-prevention byte `0x03`
//!   stripped during NAL parse.
//! * ISO/IEC 14496-15 — `AVCDecoderConfigurationRecord` (`avcC`) carries
//!   `lengthSizeMinusOne` for length-prefix framing in MP4.

use std::fs;
use std::path::PathBuf;

use oxideav_core::{CodecId, Decoder, Error, Frame, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;

/// Locate `docs/video/h264/fixtures/<name>/`. Tests run with CWD set
/// to the crate root, so we walk two levels up to the workspace root
/// and then into `docs/`.
fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from("../../docs/video/h264/fixtures").join(name)
}

/// Aggregated diff stats for one frame's planes. `pct` is the
/// fraction of bytes that match exactly; `max` is the largest abs
/// difference observed.
#[derive(Clone, Copy, Debug, Default)]
struct FrameDiff {
    y_total: usize,
    y_exact: usize,
    y_max: i32,
    uv_total: usize,
    uv_exact: usize,
    uv_max: i32,
}

impl FrameDiff {
    fn pct(&self) -> f64 {
        let exact = self.y_exact + self.uv_exact;
        let total = self.y_total + self.uv_total;
        if total == 0 {
            0.0
        } else {
            exact as f64 / total as f64 * 100.0
        }
    }
    fn merge(&mut self, other: &FrameDiff) {
        self.y_total += other.y_total;
        self.y_exact += other.y_exact;
        self.y_max = self.y_max.max(other.y_max);
        self.uv_total += other.uv_total;
        self.uv_exact += other.uv_exact;
        self.uv_max = self.uv_max.max(other.uv_max);
    }
}

/// Per-byte compare of two equal-length slices, returning
/// `(total, exact_matches, max_abs_diff)`.
fn cmp_bytes(a: &[u8], b: &[u8]) -> (usize, usize, i32) {
    let n = a.len().min(b.len());
    let mut ex = 0usize;
    let mut max = 0i32;
    for i in 0..n {
        let d = (a[i] as i32 - b[i] as i32).abs();
        if d == 0 {
            ex += 1;
        }
        if d > max {
            max = d;
        }
    }
    (n, ex, max)
}

/// Diff three planes (Y, U, V) of a decoded frame against the
/// per-frame slice of `expected.yuv`.
fn diff_planes(our: (&[u8], &[u8], &[u8]), refp: (&[u8], &[u8], &[u8])) -> FrameDiff {
    let (yt, ye, ym) = cmp_bytes(our.0, refp.0);
    let (ut, ue, um) = cmp_bytes(our.1, refp.1);
    let (vt, ve, vm) = cmp_bytes(our.2, refp.2);
    FrameDiff {
        y_total: yt,
        y_exact: ye,
        y_max: ym,
        uv_total: ut + vt,
        uv_exact: ue + ve,
        uv_max: um.max(vm),
    }
}

#[derive(Clone, Copy, Debug)]
enum ChromaFmt {
    /// 4:2:0 planar (Y full-res, U/V half each axis).
    Yuv420,
    /// 4:2:2 planar (U/V half width, full height).
    Yuv422,
    /// 4:4:4 planar (Y/U/V all same size).
    Yuv444,
}

impl ChromaFmt {
    /// Chroma plane geometry (per axis), in samples.
    fn chroma_dims(&self, w: usize, h: usize) -> (usize, usize) {
        match self {
            ChromaFmt::Yuv420 => (w / 2, h / 2),
            ChromaFmt::Yuv422 => (w / 2, h),
            ChromaFmt::Yuv444 => (w, h),
        }
    }
    /// Frame size in bytes given luma dimensions and bytes-per-sample
    /// (1 for 8-bit, 2 for 9..=14-bit P10Le / P12Le).
    fn frame_bytes(&self, w: usize, h: usize, bps: usize) -> usize {
        let (cw, ch) = self.chroma_dims(w, h);
        bps * (w * h + 2 * cw * ch)
    }
}

#[derive(Clone, Copy, Debug)]
enum Tier {
    /// Must decode bit-exactly. Test fails on any divergence. Round 343
    /// promoted the ten fixtures confirmed to match `expected.yuv`
    /// byte-for-byte (the all-intra + CAVLC/CABAC entropy + transform/QP
    /// extremes + explicit-weighted + Annex-B/MP4-framing cases); the
    /// remaining fixtures stay `ReportOnly` pending the residual decode
    /// gaps documented next to each.
    BitExact,
    /// Decode is permitted to diverge from reference; we log the deltas
    /// but do not gate CI on it (the underlying bug is queued for
    /// follow-up).
    ReportOnly,
}

struct CorpusCase {
    name: &'static str,
    width: usize,
    height: usize,
    chroma: ChromaFmt,
    n_frames: usize,
    tier: Tier,
    /// Bytes per sample in the reference `expected.yuv` and in the
    /// decoder's output planes. 1 for 8-bit (yuv420p / yuv444p),
    /// 2 for 10/12-bit (yuv420p10le / yuv422p10le / yuv444p10le, etc.) —
    /// matches the `Yuv*P10Le` / `Yuv*P12Le` PixelFormat layout (LE u16
    /// per sample). Default is 1.
    bytes_per_sample: usize,
}

/// Repack a `VideoFrame` into the row-major plane layout libavcodec
/// emits when dumping `-f rawvideo`: Y plane (W*H*bps) || U plane
/// (Cw*Ch*bps) || V plane (Cw*Ch*bps). Strips any per-row padding so the
/// resulting buffer matches `expected.yuv` byte-for-byte.
///
/// `bps` is bytes-per-sample: 1 for 8-bit, 2 for 10/12-bit `Yuv*P10Le`
/// / `Yuv*P12Le` (LE u16 per sample).
fn videoframe_to_packed(
    vf: &oxideav_core::VideoFrame,
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    bps: usize,
) -> Option<Vec<u8>> {
    if vf.planes.len() < 3 {
        // Monochrome (Y-only) or unexpected shape — fail soft.
        return None;
    }
    // `cols_bytes` is the actual byte width of the visible region per row.
    fn pack(dst: &mut Vec<u8>, src: &[u8], stride: usize, cols_bytes: usize, rows: usize) -> bool {
        if stride < cols_bytes {
            return false;
        }
        if stride == cols_bytes {
            let n = cols_bytes * rows;
            if src.len() < n {
                return false;
            }
            dst.extend_from_slice(&src[..n]);
            return true;
        }
        for r in 0..rows {
            let start = r * stride;
            let end = start + cols_bytes;
            if end > src.len() {
                return false;
            }
            dst.extend_from_slice(&src[start..end]);
        }
        true
    }
    let mut out = Vec::with_capacity(bps * (w * h + 2 * cw * ch));
    if !pack(
        &mut out,
        &vf.planes[0].data,
        vf.planes[0].stride,
        w * bps,
        h,
    ) {
        return None;
    }
    if !pack(
        &mut out,
        &vf.planes[1].data,
        vf.planes[1].stride,
        cw * bps,
        ch,
    ) {
        return None;
    }
    if !pack(
        &mut out,
        &vf.planes[2].data,
        vf.planes[2].stride,
        cw * bps,
        ch,
    ) {
        return None;
    }
    Some(out)
}

/// One frame's decode outcome — either the diff stats against the
/// reference, or a string describing what went wrong.
type FrameResult = Result<FrameDiff, String>;

/// Decode an Annex-B byte stream end-to-end and score each output
/// frame against the corresponding slice of `expected.yuv`.
///
/// Returns `None` if any of the fixture files are missing on disk.
fn decode_annex_b_fixture(case: &CorpusCase) -> Option<Vec<FrameResult>> {
    let dir = fixture_dir(case.name);
    let h264_path = dir.join("input.h264");
    let yuv_path = dir.join("expected.yuv");
    let h264 = match fs::read(&h264_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip {}: missing {} ({e})", case.name, h264_path.display());
            return None;
        }
    };
    let yuv_ref = match fs::read(&yuv_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip {}: missing {} ({e})", case.name, yuv_path.display());
            return None;
        }
    };
    decode_with_decoder(case, &yuv_ref, |dec| {
        // Feed the whole Annex B stream as one packet — the H.264
        // decoder's Annex B splitter walks NAL units internally.
        let pkt = Packet::new(0, TimeBase::new(1, 25), h264).with_pts(0);
        dec.send_packet(&pkt)
            .map_err(|e| format!("send_packet: {e}"))?;
        dec.flush().map_err(|e| format!("flush: {e}"))?;
        Ok(())
    })
}

/// Drive a fresh `H264CodecDecoder` through the closure (which feeds
/// packets), then drain `receive_frame` until EOF / NeedMore and score
/// each emitted frame.
fn decode_with_decoder<F>(case: &CorpusCase, yuv_ref: &[u8], feed: F) -> Option<Vec<FrameResult>>
where
    F: FnOnce(&mut H264CodecDecoder) -> Result<(), String>,
{
    let bps = case.bytes_per_sample.max(1);
    let frame_size = case.chroma.frame_bytes(case.width, case.height, bps);
    if yuv_ref.len() != case.n_frames * frame_size {
        eprintln!(
            "skip {}: expected.yuv size mismatch (have {} bytes, expected {} = {} frames * {})",
            case.name,
            yuv_ref.len(),
            case.n_frames * frame_size,
            case.n_frames,
            frame_size
        );
        return None;
    }

    let (cw, ch) = case.chroma.chroma_dims(case.width, case.height);
    let y_size = case.width * case.height * bps;
    let uv_size = cw * ch * bps;

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    if let Err(e) = feed(&mut dec) {
        return Some(vec![Err(e)]);
    }

    let mut visible_idx = 0usize;
    let mut results: Vec<FrameResult> = Vec::with_capacity(case.n_frames);
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => {
                if visible_idx >= case.n_frames {
                    visible_idx += 1;
                    continue;
                }
                let our = match videoframe_to_packed(&vf, case.width, case.height, cw, ch, bps) {
                    Some(b) => b,
                    None => {
                        results.push(Err(format!(
                            "visible {visible_idx}: unexpected frame shape (planes={}, p0.stride={})",
                            vf.planes.len(),
                            vf.planes.first().map(|p| p.stride).unwrap_or(0)
                        )));
                        visible_idx += 1;
                        continue;
                    }
                };
                if our.len() != frame_size {
                    results.push(Err(format!(
                        "visible {visible_idx}: packed frame size {} != expected {} (W={} H={} {:?})",
                        our.len(), frame_size, case.width, case.height, case.chroma
                    )));
                    visible_idx += 1;
                    continue;
                }
                // Optional: dump our packed frame for offline diff against
                // `expected.yuv` (e.g. region-by-region bisection of a
                // partially-defective fixture). One file per visible frame:
                // `${OXIDEAV_H264_DUMP_OURS_YUV}.<idx>`.
                if let Ok(base) = std::env::var("OXIDEAV_H264_DUMP_OURS_YUV") {
                    let _ = fs::write(format!("{base}.{visible_idx}"), &our);
                }
                let ref_off = visible_idx * frame_size;
                let ref_y = &yuv_ref[ref_off..ref_off + y_size];
                let ref_u = &yuv_ref[ref_off + y_size..ref_off + y_size + uv_size];
                let ref_v = &yuv_ref[ref_off + y_size + uv_size..ref_off + frame_size];
                let our_y = &our[..y_size];
                let our_u = &our[y_size..y_size + uv_size];
                let our_v = &our[y_size + uv_size..frame_size];
                let d = diff_planes((our_y, our_u, our_v), (ref_y, ref_u, ref_v));
                results.push(Ok(d));
                visible_idx += 1;
            }
            Ok(_) => continue,
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => {
                results.push(Err(format!("visible {visible_idx}: receive_frame: {e:?}")));
                break;
            }
        }
    }

    if visible_idx < case.n_frames {
        results.push(Err(format!(
            "decoder produced {} visible frames, expected {} — short by {}",
            visible_idx,
            case.n_frames,
            case.n_frames - visible_idx
        )));
    } else if visible_idx > case.n_frames {
        eprintln!(
            "[note] {}: decoder produced {} visible frames, expected {} (extras dropped)",
            case.name, visible_idx, case.n_frames
        );
    }

    Some(results)
}

/// Pretty-print the per-frame results then enforce the tier policy.
fn evaluate_results(case: &CorpusCase, results: Vec<FrameResult>) {
    let mut agg = FrameDiff::default();
    let mut errors: Vec<String> = Vec::new();
    let mut decoded_frames = 0usize;
    for (i, r) in results.iter().enumerate() {
        match r {
            Ok(d) => {
                eprintln!(
                    "  frame {i}: Y {}/{} exact (max diff {}), UV {}/{} exact (max diff {}), pct={:.2}%",
                    d.y_exact, d.y_total, d.y_max,
                    d.uv_exact, d.uv_total, d.uv_max,
                    d.pct()
                );
                agg.merge(d);
                decoded_frames += 1;
            }
            Err(e) => {
                eprintln!("  frame {i}: ERROR {e}");
                errors.push(format!("frame {i}: {e}"));
            }
        }
    }
    let pct = agg.pct();
    eprintln!(
        "[{:?}] {}: {} frames OK / {} reported, aggregate {}/{} exact ({pct:.2}%), Y max diff {}, UV max diff {}",
        case.tier,
        case.name,
        decoded_frames,
        results.len(),
        agg.y_exact + agg.uv_exact,
        agg.y_total + agg.uv_total,
        agg.y_max,
        agg.uv_max,
    );

    match case.tier {
        Tier::BitExact => {
            assert!(
                errors.is_empty(),
                "{}: {} frame errors prevented bit-exact comparison: {:?}",
                case.name,
                errors.len(),
                errors
            );
            assert_eq!(
                agg.y_exact + agg.uv_exact,
                agg.y_total + agg.uv_total,
                "{}: not bit-exact (Y max diff {}, UV max diff {}; {:.4}% match)",
                case.name,
                agg.y_max,
                agg.uv_max,
                pct
            );
        }
        Tier::ReportOnly => {
            // Don't fail. The eprintln output above is the deliverable.
            let _ = pct;
        }
    }
}

/// Convenience driver for the Annex-B path.
fn evaluate_annex_b(case: &CorpusCase) {
    eprintln!(
        "=== fixture {} ({}x{} {:?}, {} frames, tier {:?}) — Annex B",
        case.name, case.width, case.height, case.chroma, case.n_frames, case.tier
    );
    let results = match decode_annex_b_fixture(case) {
        Some(r) => r,
        None => return,
    };
    evaluate_results(case, results);
}

// ---------------------------------------------------------------------------
// Per-fixture tests
// ---------------------------------------------------------------------------
//
// Round 410: the staged corpus was re-baselined. Thirteen fixtures'
// `expected.yuv` files were regenerated from a stock black-box
// reference decode of the unchanged bitstreams (the original captures
// were truncated to the first 1-2 of N frames and/or contained
// error-concealment `0x80` fill in place of real macroblocks — see
// each fixture's `notes.md` "Regeneration" section). Against the
// full-stream references, every fixture except `mbaff-interlaced`
// decodes byte-exact end-to-end, so those are all pinned at
// `Tier::BitExact` with the FULL frame count. See each fixture's
// `notes.md` for the bitstream feature dimensions exercised;
// `trace.txt` lives next to it for divergence bisection (vocabulary in
// docs/video/h264/h264-fixtures-and-traces.md).

// --- Baseline / I-only -----------------------------------------------------

#[test]
fn corpus_tiny_i_only_16x16_baseline() {
    // Smallest possible H.264: Baseline / 1.0, 1×1 MB, single intra
    // 16×16 Plane-mode block. trace: 12 lines.
    // Round 343: confirmed bit-exact (Y + UV all-exact, max diff 0).
    evaluate_annex_b(&CorpusCase {
        name: "tiny-i-only-16x16-baseline",
        width: 16,
        height: 16,
        chroma: ChromaFmt::Yuv420,
        n_frames: 1,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_i_only_64x64_main() {
    // Main / 1.0 CABAC I-only with mixed I_4x4 + I_16x16 MBs across
    // a 4×4 MB grid. trace: 28 lines.
    //
    // Round 343 found the original `expected.yuv` was 100% mid-gray
    // conceal-fill (not a real decode); round 410 regenerated it from a
    // black-box reference decode of the unchanged bitstream and our
    // output matches byte-for-byte.
    evaluate_annex_b(&CorpusCase {
        name: "i-only-64x64-main",
        width: 64,
        height: 64,
        chroma: ChromaFmt::Yuv420,
        n_frames: 1,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

// --- Inter (P / B) ---------------------------------------------------------

#[test]
fn corpus_i_frame_then_p_frame_baseline() {
    // Baseline / 1.0, single IDR + single P slice. trace: 19 lines.
    // Round 343: confirmed bit-exact on the (then 1-frame) reference;
    // round 410: regenerated full 2-frame reference, both bit-exact.
    evaluate_annex_b(&CorpusCase {
        name: "i-frame-then-p-frame-baseline",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 2,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_bipred_with_b_frames_main() {
    // Main / 1.1, refs=2, 5 coded pictures. expected.yuv is 5 packed
    // 32×32 frames in DISPLAY order (CTS-sorted), so RPL + DPB ordering
    // must match to score. trace: 32 lines. (The fixture name says
    // "b-frames" but the stream's coded pictures are I + P only.)
    //
    // Round 346 proved the original 2-frame expected.yuv was internally
    // inconsistent with its own trace (a bottom-MB band no legal §8.4.2
    // / §8.7.2.1 decode could produce); round 410 regenerated the full
    // 5-frame reference from a black-box decode of the unchanged
    // bitstream and all 5 frames are bit-exact.
    evaluate_annex_b(&CorpusCase {
        name: "bipred-with-b-frames-main",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 5,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_multi_ref_frames() {
    // Main / 1.1, refs=3, ref_frame_count=3, 5 coded pictures.
    // Exercises the §8.2.4 RPL sliding-window when more than one
    // short-term ref is live. trace: 31 lines.
    //
    // Round 346 proved the original 2-frame expected.yuv was frame-
    // misaligned (it held the IDR + the 5th coded picture only); round
    // 410 regenerated the full consecutive 5-frame reference from a
    // black-box decode of the unchanged bitstream and all 5 frames are
    // bit-exact — the "real accumulated inter error" hypothesised in
    // r346 was an artefact of the misaligned reference.
    evaluate_annex_b(&CorpusCase {
        name: "multi-ref-frames",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 5,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_weighted_prediction_explicit() {
    // Main / 1.1 with `weighted_pred=1`, 4 coded pictures. Exercises
    // §8.4.2.3.2 explicit weighted sample prediction. trace: 21 lines.
    // Round 343: confirmed bit-exact on the (then 1-frame) reference;
    // round 410: regenerated full 4-frame reference, all bit-exact.
    evaluate_annex_b(&CorpusCase {
        name: "weighted-prediction-explicit",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 4,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

// --- Slice layout / entropy coding ----------------------------------------

#[test]
fn corpus_multi_slice_per_frame() {
    // Main / 1.0, 4 slices per IDR, one MB row each. Exercises
    // §7.4.1.2.4 multi-slice picture assembly + per-slice CABAC re-init
    // (§9.3.1.2). trace: 55 lines.
    //
    // Round 343 proved the original 2-frame expected.yuv did not match
    // the bitstream's single-picture structure (the four slice headers
    // describe ONE 4×4-MB coded picture); round 410 regenerated the
    // reference from a black-box decode of the unchanged bitstream —
    // one 64×64 frame, bit-exact.
    evaluate_annex_b(&CorpusCase {
        name: "multi-slice-per-frame",
        width: 64,
        height: 64,
        chroma: ChromaFmt::Yuv420,
        n_frames: 1,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_cavlc_mode() {
    // Main / 1.1 with `entropy_coding_mode=0` (CAVLC). trace: 19 lines.
    // Round 343: confirmed bit-exact on the (then 1-frame) reference;
    // round 410: regenerated full 2-frame reference, both bit-exact.
    evaluate_annex_b(&CorpusCase {
        name: "cavlc-mode",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 2,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_cabac_mode() {
    // Main / 1.1 with `entropy_coding_mode=1` (CABAC). trace: 19 lines.
    // Round 343: confirmed bit-exact on the (then 1-frame) reference;
    // round 410: regenerated full 2-frame reference, both bit-exact.
    evaluate_annex_b(&CorpusCase {
        name: "cabac-mode",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 2,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

// --- Transform / quantizer extremes ---------------------------------------

#[test]
fn corpus_transform_8x8_on() {
    // High / 1.1 with `transform_8x8_mode=1`. Exercises §8.5.10
    // 8×8 inverse integer transform path. trace: 19 lines.
    // Round 343: confirmed bit-exact on the (then 1-frame) reference;
    // round 410: regenerated full 2-frame reference, both bit-exact.
    evaluate_annex_b(&CorpusCase {
        name: "transform-8x8-on",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 2,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_qp_low_cqp() {
    // Main / 1.1 with qp=1 (near lossless). trace: 18 lines.
    // Round 343: confirmed bit-exact (Y + UV all-exact, max diff 0).
    evaluate_annex_b(&CorpusCase {
        name: "qp-low-cqp",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 1,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_qp_high_cqp() {
    // Main / 1.1 with qp=51 (maximum). trace: 18 lines.
    // Round 343: confirmed bit-exact (Y + UV all-exact, max diff 0).
    evaluate_annex_b(&CorpusCase {
        name: "qp-high-cqp",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 1,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

// --- High / 4:4:4 / 10-bit profiles ---------------------------------------

#[test]
fn corpus_4_4_4_high() {
    // High 4:4:4 Predictive / 1.0, 8-bit. expected.yuv is packed
    // 32×32×3 bytes (4:4:4 planar). trace: 18 lines.
    //
    // Round 346 found the original expected.yuv luma was 100% conceal-
    // fill (chroma partially); round 410 regenerated it from a
    // black-box decode of the unchanged bitstream — bit-exact.
    evaluate_annex_b(&CorpusCase {
        name: "4-4-4-high",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv444,
        n_frames: 1,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_intra_only_high444() {
    // High 4:4:4 Predictive / 1.1, all-intra, 2 frames. trace: 35 lines.
    //
    // Round 346 proved frame 1 bit-exact and frame 0's reference to be
    // conceal-fill; round 410 regenerated the reference from a
    // black-box decode of the unchanged bitstream — both frames
    // bit-exact (luma + chroma "coded like luma" + intra-pred-mode
    // derivation all exact at low AND high QP).
    evaluate_annex_b(&CorpusCase {
        name: "intra-only-high444",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv444,
        n_frames: 2,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

// 10-bit High10 fixture (yuv420p10le): expected.yuv is 16-bit per
// sample, little-endian (3072 B = 32×32 × 1.5 × 2). The decoder emits
// u16-LE planes for `bit_depth_luma > 8`; we score them byte-by-byte
// against the reference.
//
// History: round 349 fixed a real §8.5.8 eq. 8-309 QP_Y bug (the
// modulo addend was `52 + QpBdOffsetY` instead of `52 + 2*QpBdOffsetY`,
// which shifted every 10-bit MB's QP_Y down by 12 and collapsed the
// dequant scale; see `reconstruct::next_qp_y`), but the fixture could
// not gate: the original `expected.yuv` left MB column was 0x80
// conceal-fill (LE-u16 0x8080 = 32896 — not even a legal 10-bit
// value), and the residual "~7% right-column divergence" turned out to
// be the concealed reference contaminating its own right column, not a
// decoder bug. Round 410 regenerated the reference from a black-box
// decode of the unchanged bitstream: the full frame is bit-exact, so
// the High10 path (predicted DC default `1<<(10-1)=512`,
// `QpBdOffsetY=12` extended QP, §8.5.12 dequant at qP'=qP+12, 10-bit
// deblock) now gates end-to-end.
#[test]
fn corpus_10_bit_high10() {
    evaluate_annex_b(&CorpusCase {
        name: "10-bit-high10",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 1,
        tier: Tier::BitExact,
        // 10-bit samples — 2 bytes per sample, LE u16 storage.
        bytes_per_sample: 2,
    });
}

/// Round-349 milestone guard for the High10 (10-bit) decode path.
///
/// The `corpus_10_bit_high10` scoring stays ReportOnly because the
/// fixture's left MB column is concealed (see that test's comment), so
/// it cannot gate bit-exactness. This guard instead asserts the parts of
/// the 10-bit path that ARE objectively checkable from the bitstream +
/// spec alone, with no dependence on the concealed reference samples:
///
/// 1. The single all-intra frame decodes without error and at the
///    declared 32×32 geometry, packed as 2-bytes-per-sample LE-u16
///    `yuv420p10le` (3072 B).
/// 2. Every emitted luma/chroma sample is a legal 10-bit value
///    (`0..=1023`), i.e. the high byte of each LE-u16 is `<= 3`. A
///    decoder that mistakenly clamped to 8 bits or mis-packed the u16
///    would fail this.
/// 3. The luma plane actually exercises the >8-bit range: a substantial
///    fraction of samples exceed 255. An 8-bit-collapsed reconstruction
///    (the failure mode if `QpBdOffsetY`/`bit_depth` were dropped on the
///    floor) would pin every sample to `<= 255`.
/// 4. The right MB column (luma x∈16..32) matches `expected.yuv`
///    byte-for-byte. This is the regression guard for the round-349
///    §8.5.8 eq. 8-309 QP_Y fix: with the pre-fix addend the right
///    column was 0/512 exact (every MB's QP_Y was 12 too low,
///    collapsing the dequant scale). Since the round-410 reference
///    regeneration the whole fixture also gates BitExact in
///    `corpus_10_bit_high10`; this check remains as the targeted,
///    self-describing QP_Y guard.
#[test]
fn corpus_10_bit_high10_emits_full_range() {
    const W: usize = 32;
    const H: usize = 32;
    let dir = fixture_dir("10-bit-high10");
    let h264_path = dir.join("input.h264");
    let h264 = match fs::read(&h264_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "skip corpus_10_bit_high10_emits_full_range: missing {} ({e})",
                h264_path.display()
            );
            return;
        }
    };

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), h264).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");

    // Drain the single visible frame.
    let vf = loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => break vf,
            Ok(_) => continue,
            Err(Error::NeedMore) | Err(Error::Eof) => {
                panic!("High10 fixture produced no video frame");
            }
            Err(e) => panic!("High10 decode error: {e:?}"),
        }
    };

    // (1) Geometry + 2-bytes-per-sample LE-u16 packing.
    let (cw, ch) = (W / 2, H / 2);
    let packed = videoframe_to_packed(&vf, W, H, cw, ch, 2)
        .expect("High10 frame must repack as yuv420p10le (3 planes, 2 bytes/sample)");
    let frame_bytes = 2 * (W * H + 2 * cw * ch);
    assert_eq!(
        packed.len(),
        frame_bytes,
        "High10 packed frame must be {frame_bytes} bytes (32x32 yuv420p10le)"
    );

    // Read every sample as LE-u16.
    let sample = |i: usize| -> u16 { u16::from_le_bytes([packed[i * 2], packed[i * 2 + 1]]) };
    let n_samples = packed.len() / 2;

    // (2) Every sample is a legal 10-bit value.
    let mut max_val: u16 = 0;
    let mut luma_over_8bit = 0usize;
    for i in 0..n_samples {
        let v = sample(i);
        assert!(
            v <= 1023,
            "sample {i} = {v} exceeds the 10-bit max (1023); 10-bit output is mis-packed or unclamped"
        );
        max_val = max_val.max(v);
        if i < W * H && v > 255 {
            luma_over_8bit += 1;
        }
    }

    // (3) The reconstruction genuinely uses the >8-bit range.
    assert!(
        max_val > 255,
        "High10 output never exceeds 255 — the >8-bit transform/dequant path collapsed to 8-bit (max={max_val})"
    );
    assert!(
        luma_over_8bit >= W * H / 4,
        "only {luma_over_8bit}/{} luma samples exceed 8-bit range — High10 reconstruction looks 8-bit-collapsed",
        W * H
    );

    // (4) Right MB column (x∈16..32) regression guard for the eq. 8-309
    // QP_Y fix.
    let yuv_ref = match fs::read(dir.join("expected.yuv")) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip right-column check: missing expected.yuv ({e})");
            return;
        }
    };
    assert_eq!(
        yuv_ref.len(),
        frame_bytes,
        "reference must be one yuv420p10le frame"
    );
    let ref_sample = |i: usize| -> u16 { u16::from_le_bytes([yuv_ref[i * 2], yuv_ref[i * 2 + 1]]) };

    // The top-right MB's first luma row (y=0, x∈16..32): its 4×4
    // sub-block DC/predicted values reconstruct exactly under the
    // corrected QP_Y. A regression in eq. 8-309 immediately breaks it.
    for x in 16..32usize {
        let i = x; // row 0
        assert_eq!(
            sample(i),
            ref_sample(i),
            "top-right MB row-0 luma at x={x}: ours {} != ref {} — eq. 8-309 QP_Y regression",
            sample(i),
            ref_sample(i),
        );
    }

    // Over the whole right MB column, a large majority must be exact.
    let (mut rc_total, mut rc_exact) = (0usize, 0usize);
    for y in 0..H {
        for x in 16..32usize {
            let i = y * W + x;
            rc_total += 1;
            if sample(i) == ref_sample(i) {
                rc_exact += 1;
            }
        }
    }
    // Post-regeneration the reference is a true full-frame decode, so
    // the right column must be 512/512 exact (pre-fix eq. 8-309 was
    // 0/512).
    assert_eq!(
        rc_exact, rc_total,
        "right MB column only {rc_exact}/{rc_total} luma exact — the >8-bit QP_Y / dequant path regressed"
    );
}

#[test]
fn corpus_cavlc_444_intra8x8() {
    // High 4:4:4 Predictive / 1.1, CAVLC (`entropy_coding_mode=0`),
    // `transform_8x8_mode=1`, 64×64, 2 all-intra frames dominated by
    // Intra_8x8 MBs. The only staged stream pinning the §7.4.5.3.3
    // CAVLC residual-block / chroma-plane interleave on the 8×8 path at
    // ChromaArrayType==3 (both chroma planes "coded like luma" with
    // full-resolution 8×8 residuals). Added to the corpus driver in
    // round 410; bit-exact on first run.
    evaluate_annex_b(&CorpusCase {
        name: "cavlc-444-intra8x8",
        width: 64,
        height: 64,
        chroma: ChromaFmt::Yuv444,
        n_frames: 2,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

// --- Round-410 corpus extension (fixtures 19-29) ---------------------------
//
// Freshly generated black-box conformance streams filling the corpus
// matrix's largest holes (see each fixture's notes.md for the encoder
// command). All BitExact except the two decode-gap pins at the end.

#[test]
fn corpus_4_2_2_cabac() {
    // High 4:2:2 / CABAC, 5-frame I/P/B GOP. Byte-exact after the
    // round-410 §8.5.11.2 eq. 8-328 chroma-DC dequant fix (the inter
    // frames' high-QP chroma DC crosses into the qPDC>=36 leg).
    evaluate_annex_b(&CorpusCase {
        name: "4-2-2-cabac",
        width: 64,
        height: 64,
        chroma: ChromaFmt::Yuv422,
        n_frames: 5,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_4_2_2_cavlc() {
    // High 4:2:2 / CAVLC twin of the above (§9.2.1.1 4:2:2 chroma nC).
    evaluate_annex_b(&CorpusCase {
        name: "4-2-2-cavlc",
        width: 64,
        height: 64,
        chroma: ChromaFmt::Yuv422,
        n_frames: 5,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_10_bit_high10_inter() {
    // High 10 inter: 5-frame 10-bit 4:2:0 I/P/B GOP — the inter
    // complement to the all-intra `10-bit-high10` fixture.
    evaluate_annex_b(&CorpusCase {
        name: "10-bit-high10-inter",
        width: 64,
        height: 64,
        chroma: ChromaFmt::Yuv420,
        n_frames: 5,
        tier: Tier::BitExact,
        bytes_per_sample: 2,
    });
}

#[test]
fn corpus_10_bit_high422() {
    // 10-bit 4:2:2 I/P/B — the decisive eq. 8-328 stream: QpBdOffsetC
    // pushes qPDC past 36 at ordinary QPs, so EVERY chroma-DC block
    // takes the leg the round-410 fix corrected. Whole-picture chroma
    // was ~55% wrong before the fix.
    evaluate_annex_b(&CorpusCase {
        name: "10-bit-high422",
        width: 64,
        height: 64,
        chroma: ChromaFmt::Yuv422,
        n_frames: 5,
        tier: Tier::BitExact,
        bytes_per_sample: 2,
    });
}

#[test]
fn corpus_10_bit_high444() {
    // 10-bit 4:4:4 Predictive I/P/B — chroma coded like luma at
    // BitDepthC=10 (full-res 6-tap chroma MC, luma-style deblock).
    evaluate_annex_b(&CorpusCase {
        name: "10-bit-high444",
        width: 64,
        height: 64,
        chroma: ChromaFmt::Yuv444,
        n_frames: 5,
        tier: Tier::BitExact,
        bytes_per_sample: 2,
    });
}

#[test]
fn corpus_temporal_direct_b() {
    // Main / CABAC, direct=temporal — §8.4.1.2.3 co-located MV scaling
    // (DistScaleFactor) against a real encoder's mode decisions.
    evaluate_annex_b(&CorpusCase {
        name: "temporal-direct-b",
        width: 64,
        height: 64,
        chroma: ChromaFmt::Yuv420,
        n_frames: 6,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_weighted_bipred_implicit() {
    // Main / CABAC, weighted_bipred_idc=2 — §8.4.2.3.2 implicit
    // POC-distance weights (the explicit-weights fixture's complement).
    evaluate_annex_b(&CorpusCase {
        name: "weighted-bipred-implicit",
        width: 64,
        height: 64,
        chroma: ChromaFmt::Yuv420,
        n_frames: 6,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_scaling_matrix_jvt() {
    // High / CABAC, cqm=jvt + transform_8x8 on an I/P/B GOP — non-flat
    // Table 7-3/7-4 matrices through §8.5.9 on intra AND inter, 4x4
    // AND 8x8, in one stream.
    evaluate_annex_b(&CorpusCase {
        name: "scaling-matrix-jvt",
        width: 64,
        height: 64,
        chroma: ChromaFmt::Yuv420,
        n_frames: 6,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_cavlc_b_slices() {
    // Main / CAVLC with real B-slices — previously B-slices existed in
    // the corpus only under CABAC.
    evaluate_annex_b(&CorpusCase {
        name: "cavlc-b-slices",
        width: 64,
        height: 64,
        chroma: ChromaFmt::Yuv420,
        n_frames: 6,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_mbaff_field_pairs() {
    // MBAFF with REAL field-coded MB pairs (the `mbaff-interlaced`
    // fixture's pairs are all frame-coded). Frame 0 (all-frame-pair
    // MBAFF I) became byte-exact with the round-410 §8.7
    // pair-interleaved deblock-order fix; round 413 landed the full
    // MBAFF field-pair INTER subsystem and all three frames now gate
    // BitExact: §8.4.2.1 per-field reference selection (refIdx / 2 +
    // parity), zero-copy field-view MC on parity-interleaved rows,
    // §8.4.1.4 Table 8-10 chroma MV offset for opposite-parity refs,
    // §7.4.4 skipped-pair mb_field_decoding_flag inference, the
    // §9.3.3.1.x FIELD CABAC ctx column (Tables 9-22/9-23 inits +
    // field ctxIdxOffsets), §8.5.6/§8.5.7 field inverse scans, exact
    // Table 6-4 block-level neighbour probes, and the §8.7 per-MB
    // MBAFF deblock walker with the §8.7.2.1 NOTE 3 field MV
    // threshold.
    evaluate_annex_b(&CorpusCase {
        name: "mbaff-field-pairs",
        width: 64,
        height: 128,
        chroma: ChromaFmt::Yuv420,
        n_frames: 3,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_lossless_transform_bypass() {
    // High 4:4:4 Predictive, qpprime_y_zero_transform_bypass_flag=1 at
    // QP'Y == 0 — the §8.5 transform/scaling stages are bypassed
    // (identity legs of eqs. 8-319/8-323/8-334/8-355) and §8.5.15
    // intra residual DPCM applies for horizontal/vertical intra
    // modes. Round 413 implemented the bypass reconstruction path;
    // both frames (all-intra IDR + P) decode bit-exact — i.e.
    // LOSSLESS: the output equals the encoder input sample-for-sample.
    evaluate_annex_b(&CorpusCase {
        name: "lossless-transform-bypass",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv444,
        n_frames: 2,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

// --- Interlaced / MBAFF ---------------------------------------------------

// MBAFF: as of round 339 the §7.3.4/§7.4.4 MB-pair walker + §6.4.12.2
// Table 6-4 neighbour derivation parse this MBAFF clip through to
// completion without CABAC desync, and the MBAFF-frame reconstruct
// path (§6.4.1 field/frame y-stride) produces both visible frames.
// Round 410 regenerated `expected.yuv` from a clean black-box decode
// (the original capture contained MB concealment), making the fixture
// a strict pixel-conformance target, and closed the remaining
// 809/12288-byte divergence with three real decoder fixes: (1) the
// deblock walker's `pixel_to_mb_addr` computed raster-style MBAFF
// addresses while the grid is §6.4.1 pair-interleaved, so per-MB
// deblock metadata (QP / intra / coeffs) came from the wrong MB;
// (2) §8.7.2.1 bS=4 first-bullet handling — a horizontal intra MB
// edge between two FRAME-coded MBAFF pairs takes bS=4 ("p0 and q0
// both in frame macroblocks"), not the vertical-only bS=3 downgrade;
// (3) the inter MV-prediction neighbour fetch used raster addressing
// on the pair-interleaved grid — it now routes through the Table 6-4
// probe with the §8.4.1.3.2 eq. 8-217..8-220 field/frame MV/refIdx
// adjustment. Both frames (MBAFF I with all-frame pairs + MBAFF P
// with a 16x8 partition) now gate BitExact.
#[test]
fn corpus_mbaff_interlaced() {
    evaluate_annex_b(&CorpusCase {
        name: "mbaff-interlaced",
        width: 64,
        height: 64,
        chroma: ChromaFmt::Yuv420,
        n_frames: 2,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

/// Round-339 milestone guard: the MBAFF clip must parse + reconstruct
/// both frames without erroring. This is the regression boundary for
/// the MBAFF slice_data walker + Table 6-4 neighbour derivation.
#[test]
fn corpus_mbaff_interlaced_parses() {
    let case = CorpusCase {
        name: "mbaff-interlaced",
        width: 64,
        height: 64,
        chroma: ChromaFmt::Yuv420,
        n_frames: 2,
        tier: Tier::ReportOnly,
        bytes_per_sample: 1,
    };
    let Some(results) = decode_annex_b_fixture(&case) else {
        eprintln!("skip corpus_mbaff_interlaced_parses: fixture missing");
        return;
    };
    assert_eq!(
        results.len(),
        2,
        "MBAFF clip must yield 2 decoded frames, got {}",
        results.len()
    );
    for (i, r) in results.iter().enumerate() {
        assert!(
            r.is_ok(),
            "MBAFF frame {i} must decode without error, got: {:?}",
            r.as_ref().err()
        );
    }
}

// --- Container framing dispatch -------------------------------------------
//
// `iso-mp4-vs-annexb` ships *both* `input.h264` (Annex B) and
// `input.mp4` (length-prefixed AVC inside ISO BMFF). Per the fixture
// notes, both produce identical YUV. We exercise both paths so the
// `H264CodecDecoder::consume_extradata` (avcC) + AVCC framing branch
// in `send_packet` get a real stream test, not just a synthetic NAL.

#[test]
fn corpus_iso_mp4_vs_annexb_annexb() {
    // Round 343: confirmed bit-exact via the Annex B framing path
    // (both 32×32 frames Y + UV all-exact, max diff 0).
    evaluate_annex_b(&CorpusCase {
        name: "iso-mp4-vs-annexb",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 2,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_iso_mp4_vs_annexb_mp4() {
    // Round 343: confirmed bit-exact via the length-prefixed AVC (avcC)
    // framing path (both 32×32 frames Y + UV all-exact, max diff 0).
    let case = CorpusCase {
        name: "iso-mp4-vs-annexb",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 2,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    };
    eprintln!(
        "=== fixture {} ({}x{} {:?}, {} frames, tier {:?}) — MP4 (length-prefixed AVC)",
        case.name, case.width, case.height, case.chroma, case.n_frames, case.tier
    );

    let dir = fixture_dir(case.name);
    let mp4_path = dir.join("input.mp4");
    let yuv_path = dir.join("expected.yuv");
    let mp4 = match fs::read(&mp4_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip {}: missing {} ({e})", case.name, mp4_path.display());
            return;
        }
    };
    let yuv_ref = match fs::read(&yuv_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip {}: missing {} ({e})", case.name, yuv_path.display());
            return;
        }
    };

    // Inline ISO BMFF walker (avoids a dev-dep on oxideav-mp4): pull
    // avcC out of the avc1 sample entry and reassemble per-sample byte
    // ranges from stsz + stco + stsc. Fixture is single-track,
    // single-chunk faststart, so the layout is simple enough to walk
    // directly. See ISO/IEC 14496-12 §8.x for the box semantics.
    let mp4_data = match parse_minimal_mp4_avc(&mp4) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip {}: MP4 walk failed: {e}", case.name);
            return;
        }
    };
    eprintln!(
        "  MP4 walker: {} samples, avcC extradata: {} bytes (length_size={})",
        mp4_data.samples.len(),
        mp4_data.avcc.len(),
        mp4_data.length_size,
    );

    let results = decode_with_decoder(&case, &yuv_ref, |dec| {
        dec.consume_extradata(&mp4_data.avcc)
            .map_err(|e| format!("consume_extradata: {e}"))?;
        for (i, sample) in mp4_data.samples.iter().enumerate() {
            let pkt = Packet::new(0, TimeBase::new(1, 25), sample.clone()).with_pts(i as i64);
            dec.send_packet(&pkt)
                .map_err(|e| format!("send_packet[{i}]: {e}"))?;
        }
        dec.flush().map_err(|e| format!("flush: {e}"))?;
        Ok(())
    });
    let results = match results {
        Some(r) => r,
        None => return,
    };
    evaluate_results(&case, results);
}

// ---------------------------------------------------------------------------
// Minimal ISO BMFF (MP4) walker — tailored to the iso-mp4-vs-annexb
// fixture. Single video track, single chunk, faststart. Extracts:
//   * `avcC` payload (the `AVCDecoderConfigurationRecord`),
//   * `lengthSizeMinusOne + 1` (= `length_size`),
//   * one byte vec per video sample, length-prefixed AVC NALs intact.
//
// We only support what this one fixture needs (one trak, one chunk per
// track, stsz with per-sample sizes, stco with one offset). Anything
// more elaborate returns an Err so the caller skips the test rather
// than reporting bogus results.
// ---------------------------------------------------------------------------

struct Mp4AvcData {
    avcc: Vec<u8>,
    length_size: u8,
    samples: Vec<Vec<u8>>,
}

fn parse_minimal_mp4_avc(buf: &[u8]) -> Result<Mp4AvcData, String> {
    // Walk top-level boxes to find moov + remember mdat range.
    let mut moov: Option<&[u8]> = None;
    let mut off = 0usize;
    while off + 8 <= buf.len() {
        let (body, ty, hdr_len) = read_box(buf, off)?;
        if &ty == b"moov" {
            moov = Some(body);
        }
        off += hdr_len + body.len();
    }
    let moov = moov.ok_or_else(|| "no moov box".to_string())?;

    // Drill down: moov / trak / mdia / minf / stbl. Pick the first
    // trak with handler 'vide'.
    let mut avcc: Option<Vec<u8>> = None;
    let mut length_size: u8 = 4;
    let mut stsz_sizes: Vec<u32> = Vec::new();
    let mut stco_offsets: Vec<u64> = Vec::new();
    let mut stsc: Vec<(u32, u32, u32)> = Vec::new(); // (first_chunk, samples_per_chunk, sample_description_index)

    let mut moov_off = 0usize;
    while moov_off + 8 <= moov.len() {
        let (body, ty, hdr) = read_box(moov, moov_off)?;
        if &ty == b"trak" {
            // Walk trak / mdia / hdlr to confirm it's video.
            let (mdia, _) = find_box(body, b"mdia")?;
            let (hdlr, _) = find_box(mdia, b"hdlr")?;
            // hdlr: u32 version+flags, u32 pre_defined, 4-byte handler_type
            if hdlr.len() < 12 || &hdlr[8..12] != b"vide" {
                moov_off += hdr + body.len();
                continue;
            }
            let (minf, _) = find_box(mdia, b"minf")?;
            let (stbl, _) = find_box(minf, b"stbl")?;

            // stsd → first sample entry → look for avcC inside.
            let (stsd, _) = find_box(stbl, b"stsd")?;
            // stsd full box: 4 bytes (version+flags) + 4 bytes (entry_count)
            // + sample entries.
            if stsd.len() < 8 {
                return Err("stsd truncated".into());
            }
            // VisualSampleEntry layout per ISO/IEC 14496-12 §8.5.2:
            //   8  bytes  Box header  (size + 4-byte type)
            //   6  bytes  reserved
            //   2  bytes  data_reference_index
            //   16 bytes  pre_defined + reserved
            //   4  bytes  width + height
            //   8  bytes  horizresolution + vertresolution
            //   4  bytes  reserved
            //   2  bytes  frame_count
            //   32 bytes  compressorname
            //   2  bytes  depth
            //   2  bytes  pre_defined (-1)
            //   ----
            //   86 bytes  total before child boxes (e.g. avcC, btrt).
            // After the size+type header (8 bytes) the remainder is 78
            // bytes; child boxes start at `entry_start + 8 + 78`.
            let so = 8usize; // skip stsd full-box version+flags + entry_count
            if so + 8 > stsd.len() {
                return Err("stsd entry truncated".into());
            }
            let entry_size =
                u32::from_be_bytes([stsd[so], stsd[so + 1], stsd[so + 2], stsd[so + 3]]) as usize;
            let entry_ty = &stsd[so + 4..so + 8];
            if entry_ty != b"avc1" && entry_ty != b"avc3" {
                return Err(format!("unsupported video sample entry: {:?}", entry_ty));
            }
            let entry_end = so + entry_size;
            if entry_end > stsd.len() {
                return Err("stsd entry overruns".into());
            }
            // VisualSampleEntry header is 78 bytes after size+type:
            //  6 reserved + 2 data_ref_index + 16 + 4 (w/h) + 8 + 4 + 2
            //  + 32 + 2 + 2 = 78
            let mut child_off = so + 8 + 78;
            while child_off + 8 <= entry_end {
                let (cbody, cty, chdr) = read_box(stsd, child_off)?;
                if &cty == b"avcC" {
                    avcc = Some(cbody.to_vec());
                    if cbody.len() >= 5 {
                        length_size = (cbody[4] & 0x03) + 1;
                    }
                }
                child_off += chdr + cbody.len();
            }

            // stsz: u8 version + 24 flags + u32 sample_size + u32 sample_count + per-sample sizes
            let (stsz, _) = find_box(stbl, b"stsz")?;
            if stsz.len() < 12 {
                return Err("stsz truncated".into());
            }
            let sample_size = u32::from_be_bytes([stsz[4], stsz[5], stsz[6], stsz[7]]);
            let sample_count = u32::from_be_bytes([stsz[8], stsz[9], stsz[10], stsz[11]]) as usize;
            if sample_size != 0 {
                stsz_sizes = vec![sample_size; sample_count];
            } else {
                if stsz.len() < 12 + sample_count * 4 {
                    return Err("stsz per-sample table truncated".into());
                }
                for i in 0..sample_count {
                    let o = 12 + i * 4;
                    stsz_sizes.push(u32::from_be_bytes([
                        stsz[o],
                        stsz[o + 1],
                        stsz[o + 2],
                        stsz[o + 3],
                    ]));
                }
            }

            // stsc: u32 version+flags, u32 entry_count, then 12-byte entries.
            let (stsc_box, _) = find_box(stbl, b"stsc")?;
            if stsc_box.len() < 8 {
                return Err("stsc truncated".into());
            }
            let stsc_count =
                u32::from_be_bytes([stsc_box[4], stsc_box[5], stsc_box[6], stsc_box[7]]) as usize;
            if stsc_box.len() < 8 + stsc_count * 12 {
                return Err("stsc table truncated".into());
            }
            for i in 0..stsc_count {
                let o = 8 + i * 12;
                let fc = u32::from_be_bytes([
                    stsc_box[o],
                    stsc_box[o + 1],
                    stsc_box[o + 2],
                    stsc_box[o + 3],
                ]);
                let spc = u32::from_be_bytes([
                    stsc_box[o + 4],
                    stsc_box[o + 5],
                    stsc_box[o + 6],
                    stsc_box[o + 7],
                ]);
                let sdi = u32::from_be_bytes([
                    stsc_box[o + 8],
                    stsc_box[o + 9],
                    stsc_box[o + 10],
                    stsc_box[o + 11],
                ]);
                stsc.push((fc, spc, sdi));
            }

            // stco or co64.
            if let Ok((stco_box, _)) = find_box(stbl, b"stco") {
                if stco_box.len() < 8 {
                    return Err("stco truncated".into());
                }
                let n = u32::from_be_bytes([stco_box[4], stco_box[5], stco_box[6], stco_box[7]])
                    as usize;
                if stco_box.len() < 8 + n * 4 {
                    return Err("stco table truncated".into());
                }
                for i in 0..n {
                    let o = 8 + i * 4;
                    stco_offsets.push(u32::from_be_bytes([
                        stco_box[o],
                        stco_box[o + 1],
                        stco_box[o + 2],
                        stco_box[o + 3],
                    ]) as u64);
                }
            } else if let Ok((co64_box, _)) = find_box(stbl, b"co64") {
                if co64_box.len() < 8 {
                    return Err("co64 truncated".into());
                }
                let n = u32::from_be_bytes([co64_box[4], co64_box[5], co64_box[6], co64_box[7]])
                    as usize;
                if co64_box.len() < 8 + n * 8 {
                    return Err("co64 table truncated".into());
                }
                for i in 0..n {
                    let o = 8 + i * 8;
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&co64_box[o..o + 8]);
                    stco_offsets.push(u64::from_be_bytes(bytes));
                }
            } else {
                return Err("no stco / co64".into());
            }

            break; // first video trak only
        }
        moov_off += hdr + body.len();
    }

    let avcc = avcc.ok_or_else(|| "no avcC in stsd".to_string())?;

    // Use stsc + stco to walk per-sample byte ranges.
    // For each sample i (0-indexed): figure out which chunk it lives in
    // and its index within that chunk; the chunk's file offset is in
    // stco; samples are laid out contiguously in the chunk in stsz order.
    let n_samples = stsz_sizes.len();
    if stsc.is_empty() || stco_offsets.is_empty() {
        return Err("empty stsc / stco".into());
    }
    let mut chunk_of_sample: Vec<u32> = Vec::with_capacity(n_samples);
    let mut sample_i = 0usize;
    let n_chunks = stco_offsets.len() as u32;
    for entry_i in 0..stsc.len() {
        let (fc, spc, _sdi) = stsc[entry_i];
        let next_fc = stsc.get(entry_i + 1).map(|e| e.0).unwrap_or(n_chunks + 1);
        let mut ch = fc.max(1);
        while ch < next_fc && sample_i < n_samples {
            for _ in 0..spc {
                if sample_i >= n_samples {
                    break;
                }
                chunk_of_sample.push(ch);
                sample_i += 1;
            }
            ch += 1;
        }
    }
    while chunk_of_sample.len() < n_samples {
        // Fallback: stuff into the last chunk.
        chunk_of_sample.push(n_chunks);
    }

    let mut samples: Vec<Vec<u8>> = Vec::with_capacity(n_samples);
    let mut cur_chunk: i32 = -1;
    let mut chunk_cursor: u64 = 0;
    for i in 0..n_samples {
        let ch = chunk_of_sample[i] as i32;
        let size = stsz_sizes[i] as u64;
        if ch != cur_chunk {
            let cidx = (ch - 1) as usize;
            if cidx >= stco_offsets.len() {
                return Err(format!("chunk index {ch} out of range"));
            }
            chunk_cursor = stco_offsets[cidx];
            cur_chunk = ch;
        }
        let start = chunk_cursor as usize;
        let end = start + size as usize;
        if end > buf.len() {
            return Err(format!("sample {i} overruns mp4"));
        }
        samples.push(buf[start..end].to_vec());
        chunk_cursor += size;
    }

    Ok(Mp4AvcData {
        avcc,
        length_size,
        samples,
    })
}

/// Read a single ISO BMFF box at offset `off` in `buf`. Returns
/// `(body_slice, four_cc, header_len)` where `header_len` is 8 (32-bit
/// size) or 16 (64-bit size). Does NOT support box-size==0
/// (extends-to-EOF), since the fixture doesn't use it.
fn read_box(buf: &[u8], off: usize) -> Result<(&[u8], [u8; 4], usize), String> {
    if off + 8 > buf.len() {
        return Err(format!("box header truncated at {off}"));
    }
    let size32 = u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
    let mut ty = [0u8; 4];
    ty.copy_from_slice(&buf[off + 4..off + 8]);
    let (header_len, total): (usize, usize) = if size32 == 1 {
        if off + 16 > buf.len() {
            return Err("largesize box header truncated".into());
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&buf[off + 8..off + 16]);
        (16, u64::from_be_bytes(bytes) as usize)
    } else if size32 < 8 {
        return Err(format!("box {ty:?} size {size32} < 8 at {off}"));
    } else {
        (8, size32 as usize)
    };
    if off + total > buf.len() {
        return Err(format!(
            "box {ty:?} body overruns ({total} bytes from {off})"
        ));
    }
    Ok((&buf[off + header_len..off + total], ty, header_len))
}

/// Walk container's child boxes looking for one with the given fourcc.
/// `'a` is named explicitly because two input references appear and the
/// returned slice borrows from `container`, not `target`; clippy's
/// `needless_lifetimes` does not fire here.
fn find_box<'a>(container: &'a [u8], target: &[u8; 4]) -> Result<(&'a [u8], usize), String> {
    let mut off = 0usize;
    while off + 8 <= container.len() {
        let (body, ty, hdr) = read_box(container, off)?;
        if &ty == target {
            return Ok((body, off));
        }
        off += hdr + body.len();
    }
    Err(format!("box {target:?} not found"))
}

// --- Round-413 corpus extension (fixtures 30-31) ---------------------------

#[test]
fn corpus_mbaff_field_8x8t() {
    // High profile MBAFF with FIELD-coded pairs AND transform_8x8_mode:
    // pins the §8.5.7 Table 8-14 8x8 FIELD inverse scan and the
    // ctxBlockCat=5 field significance contexts (Table 9-24 field
    // column) on real field-coded 8x8 residual blocks. Bit-exact on
    // first run after the round-413 MBAFF field-pair subsystem landed.
    evaluate_annex_b(&CorpusCase {
        name: "mbaff-field-8x8t",
        width: 64,
        height: 128,
        chroma: ChromaFmt::Yuv420,
        n_frames: 3,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_mbaff_field_cavlc() {
    // Main profile CAVLC MBAFF with FIELD-coded pairs — pins the
    // §9.2.1 CAVLC decode path under MBAFF (the corpus previously
    // exercised MBAFF only under CABAC). Round 413 routed the
    // §9.2.1.1 nC neighbour probes through the exact §6.4.12.2
    // Table 6-4 process, doubled the te(v) ref_idx range for field
    // MBs, and wired §7.4.4 inference + field flags into the CAVLC
    // walker. Round 416 closed the remaining desync: the residual
    // walker's own luma/plane nC closures still resolved EXTERNAL
    // neighbours through the raster-scan map, so on the
    // pair-interleaved MBAFF grid the bottom MB of the leftmost pair
    // read its own pair-top as "left" (a dense I MB → bogus nC=9).
    // The closures now route through the same Table 6-4 probe as the
    // chroma AC path — all three frames gate BitExact.
    evaluate_annex_b(&CorpusCase {
        name: "mbaff-field-cavlc",
        width: 64,
        height: 128,
        chroma: ChromaFmt::Yuv420,
        n_frames: 3,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

// --- Round-416 corpus extension (fixtures 32-34): PAFF -------------------
//
// Field-coded pictures (`field_pic_flag = 1`) under an interlaced SPS
// (`frame_mbs_only_flag = 0`, `mb_adaptive_frame_field_flag = 0`). The
// bitstreams come from this crate's clean-room PAFF field encoder
// (`encoder::field`) because no stock encoder binary in the toolchain
// emits PAFF; `expected.yuv` is a BLACK-BOX decode by a stock FFmpeg
// 8.1 binary, so the reference binary — not our own decoder — defines
// the conformance contract these tests gate against.

#[test]
fn corpus_paff_i_fields() {
    // 6 intra field pictures (IDR top + non-IDR I bottom sharing
    // frame_num=0 per §7.4.3, then I/I reference pairs). Pins the
    // §7.3.3 field slice header, §7.4.2.1.1 eq. 7-26 half-height field
    // decode, §8.5.6 FIELD coefficient scan, §8.2.1 field POC, field
    // §8.7 deblock (horizontal intra MB edges bS=3) and the §C.4.4
    // complementary-pair output interleave.
    evaluate_annex_b(&CorpusCase {
        name: "paff-i-fields",
        width: 64,
        height: 96,
        chroma: ChromaFmt::Yuv420,
        n_frames: 3,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_paff_p_fields() {
    // P/P reference field pairs after the IDR pair, single active L0
    // ref. Pins the §8.2.4.2.2 + §8.2.4.2.5 field reference-list
    // initialisation — RefPicList0[0] is the SAME-parity field of the
    // previous frame for both the first and the second field of a
    // frame (round 416 fixed the driver, which previously sorted by
    // per-field PicNum only and handed every second field its own
    // frame's complementary field) — plus half-height field MC and
    // per-field inter deblock with per-FIELD reference POCs.
    evaluate_annex_b(&CorpusCase {
        name: "paff-p-fields",
        width: 64,
        height: 96,
        chroma: ChromaFmt::Yuv420,
        n_frames: 4,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_paff_p_crossparity() {
    // Frame 0's bottom field is a P field referencing the IDR TOP
    // field — the §8.2.4.2.5 second-field fall-through when the
    // same-parity sub-list is empty — pinning cross-parity field MC
    // and the §8.4.1.4 Table 8-10 vertical chroma-MV adjustment
    // (mvCLX[1] = mvLX[1] + 2 for a top reference from a bottom
    // field).
    evaluate_annex_b(&CorpusCase {
        name: "paff-p-crossparity",
        width: 64,
        height: 96,
        chroma: ChromaFmt::Yuv420,
        n_frames: 3,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}

#[test]
fn corpus_paff_mixed_frame_field() {
    // Field pair, then a full-height I FRAME picture
    // (field_pic_flag=0), then another field pair — the
    // picture-adaptive axis proper: PicHeightInMbs, the §8.5.6
    // zig-zag/field scan selection and the §8.7 frame/field bS rules
    // all toggle per access unit under one interlaced SPS.
    evaluate_annex_b(&CorpusCase {
        name: "paff-mixed-frame-field",
        width: 64,
        height: 96,
        chroma: ChromaFmt::Yuv420,
        n_frames: 3,
        tier: Tier::BitExact,
        bytes_per_sample: 1,
    });
}
