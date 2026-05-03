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
//! A handful of fixtures use `#[ignore]` instead of a separate tier
//! when the comparison itself is structurally invalid (e.g. 10-bit
//! `expected.yuv` is 16-bit per sample but our decoder clamps to u8 in
//! `picture_to_video_frame`, so a byte-by-byte score is meaningless
//! until the decoder grows a >8-bit output path).
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
    /// 4:2:0 planar 8-bit (Y full-res, U/V half each axis).
    Yuv420,
    /// 4:4:4 planar 8-bit (Y/U/V all same size).
    Yuv444,
}

impl ChromaFmt {
    /// Frame size in bytes given luma dimensions.
    fn frame_bytes(&self, w: usize, h: usize) -> usize {
        match self {
            ChromaFmt::Yuv420 => w * h + 2 * (w / 2) * (h / 2),
            ChromaFmt::Yuv444 => 3 * w * h,
        }
    }
    /// Chroma plane geometry (per axis).
    fn chroma_dims(&self, w: usize, h: usize) -> (usize, usize) {
        match self {
            ChromaFmt::Yuv420 => (w / 2, h / 2),
            ChromaFmt::Yuv444 => (w, h),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Tier {
    /// Must decode bit-exactly. Test fails on any divergence.
    /// (Unused on day one — every fixture starts ReportOnly per task
    /// brief; promote individual entries here as follow-up rounds
    /// confirm bit-exact decode.)
    #[allow(dead_code)]
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
}

/// Repack a `VideoFrame` into the row-major plane layout libavcodec
/// emits when dumping `-f rawvideo`: Y plane (W*H) || U plane
/// (Cw*Ch) || V plane (Cw*Ch). Strips any per-row padding so the
/// resulting buffer matches `expected.yuv` byte-for-byte.
fn videoframe_to_packed(
    vf: &oxideav_core::VideoFrame,
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
) -> Option<Vec<u8>> {
    if vf.planes.len() < 3 {
        // Monochrome (Y-only) or unexpected shape — fail soft.
        return None;
    }
    fn pack(dst: &mut Vec<u8>, src: &[u8], stride: usize, cols: usize, rows: usize) -> bool {
        if stride < cols {
            return false;
        }
        if stride == cols {
            let n = cols * rows;
            if src.len() < n {
                return false;
            }
            dst.extend_from_slice(&src[..n]);
            return true;
        }
        for r in 0..rows {
            let start = r * stride;
            let end = start + cols;
            if end > src.len() {
                return false;
            }
            dst.extend_from_slice(&src[start..end]);
        }
        true
    }
    let mut out = Vec::with_capacity(w * h + 2 * cw * ch);
    if !pack(&mut out, &vf.planes[0].data, vf.planes[0].stride, w, h) {
        return None;
    }
    if !pack(&mut out, &vf.planes[1].data, vf.planes[1].stride, cw, ch) {
        return None;
    }
    if !pack(&mut out, &vf.planes[2].data, vf.planes[2].stride, cw, ch) {
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
    let frame_size = case.chroma.frame_bytes(case.width, case.height);
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
    let y_size = case.width * case.height;
    let uv_size = cw * ch;

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
                let our = match videoframe_to_packed(&vf, case.width, case.height, cw, ch) {
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
// Every fixture starts in `Tier::ReportOnly` per task brief: this is a
// pure test-integration round (the decoder is not modified). Promote
// individual cases to `Tier::BitExact` once a follow-up round confirms
// they match. See each fixture's `notes.md` for the bitstream feature
// dimensions exercised; `trace.txt` lives next to it for divergence
// bisection (vocabulary in docs/video/h264/h264-fixtures-and-traces.md).

// --- Baseline / I-only -----------------------------------------------------

#[test]
fn corpus_tiny_i_only_16x16_baseline() {
    // Smallest possible H.264: Baseline / 1.0, 1×1 MB, single intra
    // 16×16 Plane-mode block. trace: 12 lines.
    evaluate_annex_b(&CorpusCase {
        name: "tiny-i-only-16x16-baseline",
        width: 16,
        height: 16,
        chroma: ChromaFmt::Yuv420,
        n_frames: 1,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_i_only_64x64_main() {
    // Main / 1.0 CABAC I-only with mixed I_4x4 + I_16x16 MBs across
    // a 4×4 MB grid. trace: 28 lines.
    evaluate_annex_b(&CorpusCase {
        name: "i-only-64x64-main",
        width: 64,
        height: 64,
        chroma: ChromaFmt::Yuv420,
        n_frames: 1,
        tier: Tier::ReportOnly,
    });
}

// --- Inter (P / B) ---------------------------------------------------------

#[test]
fn corpus_i_frame_then_p_frame_baseline() {
    // Baseline / 1.0, single IDR + single P slice. expected.yuv carries
    // exactly one packed frame (the IDR; the second visible frame
    // hashes the same byte-for-byte). trace: 19 lines.
    evaluate_annex_b(&CorpusCase {
        name: "i-frame-then-p-frame-baseline",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 1,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_bipred_with_b_frames_main() {
    // Main / 1.1, B-frames + refs=2. expected.yuv is 2 packed 32×32
    // frames in DISPLAY order (CTS-sorted), so RPL + DPB ordering must
    // match libavcodec to score. trace: 32 lines.
    evaluate_annex_b(&CorpusCase {
        name: "bipred-with-b-frames-main",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 2,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_multi_ref_frames() {
    // Main / 1.1, refs=3, ref_frame_count=3. Exercises the §8.2.4 RPL
    // sliding-window when more than one short-term ref is live. trace:
    // 31 lines.
    evaluate_annex_b(&CorpusCase {
        name: "multi-ref-frames",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 2,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_weighted_prediction_explicit() {
    // Main / 1.1 with `weighted_pred=1`. Exercises §8.4.2.3.2 explicit
    // weighted sample prediction. trace: 21 lines.
    evaluate_annex_b(&CorpusCase {
        name: "weighted-prediction-explicit",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 1,
        tier: Tier::ReportOnly,
    });
}

// --- Slice layout / entropy coding ----------------------------------------

#[test]
fn corpus_multi_slice_per_frame() {
    // Main / 1.0, 4 slices per IDR, one MB row each. Exercises
    // §7.4.1.2.4 multi-slice picture assembly + per-slice CABAC re-init
    // (§9.3.1.2). trace: 55 lines.
    evaluate_annex_b(&CorpusCase {
        name: "multi-slice-per-frame",
        width: 64,
        height: 64,
        chroma: ChromaFmt::Yuv420,
        n_frames: 2,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_cavlc_mode() {
    // Main / 1.1 with `entropy_coding_mode=0` (CAVLC). trace: 19 lines.
    evaluate_annex_b(&CorpusCase {
        name: "cavlc-mode",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 1,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_cabac_mode() {
    // Main / 1.1 with `entropy_coding_mode=1` (CABAC). trace: 19 lines.
    evaluate_annex_b(&CorpusCase {
        name: "cabac-mode",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 1,
        tier: Tier::ReportOnly,
    });
}

// --- Transform / quantizer extremes ---------------------------------------

#[test]
fn corpus_transform_8x8_on() {
    // High / 1.1 with `transform_8x8_mode=1`. Exercises §8.5.10
    // 8×8 inverse integer transform path. trace: 19 lines.
    evaluate_annex_b(&CorpusCase {
        name: "transform-8x8-on",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 1,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_qp_low_cqp() {
    // Main / 1.1 with qp=1 (near lossless). trace: 18 lines.
    evaluate_annex_b(&CorpusCase {
        name: "qp-low-cqp",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 1,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_qp_high_cqp() {
    // Main / 1.1 with qp=51 (maximum). trace: 18 lines.
    evaluate_annex_b(&CorpusCase {
        name: "qp-high-cqp",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 1,
        tier: Tier::ReportOnly,
    });
}

// --- High / 4:4:4 / 10-bit profiles ---------------------------------------

#[test]
fn corpus_4_4_4_high() {
    // High 4:4:4 Predictive / 1.0, 8-bit. expected.yuv is packed
    // 32×32×3 bytes (4:4:4 planar). trace: 18 lines.
    evaluate_annex_b(&CorpusCase {
        name: "4-4-4-high",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv444,
        n_frames: 1,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_intra_only_high444() {
    // High 4:4:4 Predictive / 1.1, all-intra. trace: 35 lines.
    evaluate_annex_b(&CorpusCase {
        name: "intra-only-high444",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv444,
        n_frames: 2,
        tier: Tier::ReportOnly,
    });
}

// 10-bit fixtures: expected.yuv is 16-bit per sample (little-endian),
// our decoder clamps to 8-bit in `picture_to_video_frame`. A direct
// byte-by-byte comparison is meaningless until the decoder grows a
// >8-bit output path, so this stays #[ignore].
#[test]
#[ignore = "10-bit High10: expected.yuv is 16-bit per sample but H264CodecDecoder \
            clamps reconstructed samples to u8 in picture_to_video_frame. \
            Track in a follow-up: add a wider-than-8-bit pixel format / per-frame \
            bit_depth metadata so this fixture becomes scoreable."]
fn corpus_10_bit_high10() {
    evaluate_annex_b(&CorpusCase {
        name: "10-bit-high10",
        width: 32,
        height: 32,
        // Geometry only; the chroma_fmt/frame_bytes() math wouldn't
        // give meaningful diffs at u8 vs u16 anyway — kept here so the
        // structure mirrors the other fixtures when the test eventually
        // un-ignores.
        chroma: ChromaFmt::Yuv420,
        n_frames: 1,
        tier: Tier::ReportOnly,
    });
}

// --- Interlaced / MBAFF ---------------------------------------------------

// MBAFF: H264CodecDecoder rejects field-coded slices in `reconstruct`
// (`field_pic_flag == 1` or `mb_adaptive_frame_field_flag == 1` are
// known simplifications per the module docstring). The decoder won't
// produce visible frames here — kept as ReportOnly so the eprintln
// output documents the failure mode but doesn't gate CI.
#[test]
fn corpus_mbaff_interlaced() {
    evaluate_annex_b(&CorpusCase {
        name: "mbaff-interlaced",
        width: 64,
        height: 64,
        chroma: ChromaFmt::Yuv420,
        n_frames: 2,
        tier: Tier::ReportOnly,
    });
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
    evaluate_annex_b(&CorpusCase {
        name: "iso-mp4-vs-annexb",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 2,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_iso_mp4_vs_annexb_mp4() {
    let case = CorpusCase {
        name: "iso-mp4-vs-annexb",
        width: 32,
        height: 32,
        chroma: ChromaFmt::Yuv420,
        n_frames: 2,
        tier: Tier::ReportOnly,
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
        match &ty {
            b"moov" => {
                moov = Some(body);
            }
            _ => {}
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
fn read_box<'a>(buf: &'a [u8], off: usize) -> Result<(&'a [u8], [u8; 4], usize), String> {
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
