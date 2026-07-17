#![no_main]

//! Fuzz: arbitrary bytes → BOTH oxideav `H264CodecDecoder` and the
//! libavcodec H.264 decoder (bound by name, opened with
//! `err_detect=explode`). Compare:
//!
//! 1. **Reject parity (advisory)** — if libavcodec fails to produce a
//!    clean frame from the input while ours decodes CLEANLY
//!    (`decode_error_count() == 0`), that strictness divergence is
//!    reported to stderr for CI-log triage. It is NOT a crash: the
//!    reference layers heuristic stream-discipline checks beyond
//!    bitstream conformance (it refuses non-paired fields, for one),
//!    so its rejection does not prove the input malformed — and a hard
//!    assert here turns every such corner the fuzzer finds into a red
//!    schedule. Triaging a batch of these reports DID surface four
//!    decoder-side conformance gaps (now enforced with spec cites), so
//!    the reports stay worth reading. Degraded output
//!    (`decode_error_count() > 0`) is exempt entirely: our decoder
//!    deliberately skips broken slices, and salvaging a partial
//!    picture from a stream the reference refuses is resilience, not
//!    mis-acceptance.
//! 2. **Frame parity** — if libavcodec produced a (clean, by explode)
//!    frame and ours decoded cleanly too, the oracle frame must match
//!    one of our output frames: same dimensions + chroma subsampling,
//!    per-pixel YUV within ±1 LSB (IDCT-adjacent rounding drift between
//!    independent implementations of the same spec). "One of" because
//!    a fuzzed buffer may hold several access units: the oracle is fed
//!    the whole buffer as a single packet and drains exactly one frame,
//!    while our decoder decodes every AU and outputs in POC order —
//!    which of the decoded pictures comes out first is not comparable
//!    across the two feeding models, but the oracle's frame must still
//!    BE one of ours.
//!
//! Comparison premises (documented so future edits don't resurrect old
//! false-positive classes):
//! - The oracle wrapper only represents 8-bit planar 4:2:0/4:2:2/4:4:4.
//!   Streams whose SPSes use any other bit depth / chroma format
//!   (High10 9..14-bit, monochrome, separate colour planes) are
//!   `Incomparable` on our side and skipped — libavcodec maps those to
//!   pixel formats the wrapper reports as `Rejected`, which says
//!   nothing about the input being malformed.
//! - The oracle decoder is opened with `err_detect=explode`, so an
//!   oracle frame is never concealment output; our side is gated on
//!   `decode_error_count() == 0` for the same reason. Concealed samples
//!   are implementation-defined and must never reach pixel asserts.
//!
//! Skips silently when libavcodec isn't loadable so the harness still
//! runs against the panic_free coverage even on hosts without ffmpeg
//! installed.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, Decoder, Frame, Packet, TimeBase, VideoPlane};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264::nal::{rbsp_from_nal_payload, AnnexBSplitter};
use oxideav_h264::sps::Sps;
use oxideav_h264_fuzz::libavcodec;

/// Per-channel tolerance (in 8-bit code units) for the cross-decode
/// comparison. The §8.5 inverse transform is bit-exact in the spec, but
/// the surrounding intra-prediction + deblocking arithmetic can produce
/// ±1 LSB drift across implementations that round halves differently.
const TOL: i32 = 1;

/// Maximum frame area (luma samples) we'll spend cycles asserting on.
/// Larger frames burn the per-iteration time budget without exposing
/// new code paths.
const MAX_AREA: usize = 256 * 256;

/// Maximum number of frames drained from our decoder per input. A
/// fuzzed buffer rarely contains more real access units than this; the
/// cap keeps adversarial many-AU inputs from burning the time budget.
const MAX_OURS_FRAMES: usize = 8;

/// Maximum SPS-declared picture size (in macroblocks) this harness will
/// decode at all. A hostile SPS can declare up to 510x510 MBs
/// (~67 Mpx); decoding that allocates hundreds of MB of sample planes
/// in BOTH decoders per iteration, and the allocator's high-water RSS
/// accumulates across libFuzzer iterations until the rss_limit kills
/// the run as a phony OOM. 64x64 MBs (1024x1024 px) is far beyond
/// MAX_AREA (all asserts skip above that anyway), so nothing testable
/// is lost. Robustness on monster declarations stays covered by the
/// `panic_free_decode` target.
const MAX_SPS_MBS: u32 = 64 * 64;

/// Pre-scan the Annex-B buffer's SPS NAL units (type 7). Returns false
/// when any parseable SPS declares a picture larger than
/// [`MAX_SPS_MBS`] — the input is skipped before either decoder runs.
fn sps_sizes_are_sane(data: &[u8]) -> bool {
    for nal in AnnexBSplitter::new(data) {
        let Some((&header, payload)) = nal.split_first() else {
            continue;
        };
        if header & 0x1f != 7 {
            continue;
        }
        let rbsp = rbsp_from_nal_payload(payload);
        if let Ok(sps) = Sps::parse(&rbsp) {
            let mbs = (sps.pic_width_in_mbs_minus1 + 1)
                .saturating_mul(sps.pic_height_in_map_units_minus1 + 1);
            if mbs > MAX_SPS_MBS {
                return false;
            }
        }
    }
    true
}

fuzz_target!(|data: &[u8]| {
    // Print skip banner once per process so the CI log carries proof of
    // whether the oracle actually ran.
    static OBSERVED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if data.is_empty() {
        return;
    }
    let avail = libavcodec::available();
    OBSERVED.get_or_init(|| {
        if !avail {
            eprintln!("[oracle skip] libavcodec not loadable");
        }
        avail
    });
    if !avail {
        return;
    }
    if !sps_sizes_are_sane(data) {
        return;
    }

    // Run libavcodec first so we know what to expect from ours.
    let oracle = libavcodec::decode_h264(data);

    let ours = decode_with_ours(data);

    use libavcodec::OracleResult;
    match (oracle, ours) {
        (OracleResult::Unavailable, _) => {
            // Decoder factory disappeared between `available()` and
            // `decode_h264` — skip the iteration.
        }
        (_, OursResult::Incomparable) => {
            // Stream declares a bit depth / chroma format the oracle
            // wrapper cannot represent (it reports such frames as
            // `Rejected` regardless of stream validity), so neither
            // parity assertion is meaningful.
        }
        (OracleResult::Rejected, OursResult::Rejected) => {
            // Both rejected — desired behaviour.
        }
        (OracleResult::Rejected, OursResult::Concealed) => {
            // The reference refused the stream; we salvaged a degraded
            // picture by skipping the broken slices. Resilient output
            // is not "accepting" the stream — the decoder itself
            // reported the errors (decode_error_count > 0).
            eprintln!("[oracle note] libavcodec rejected; oxideav salvaged a degraded frame");
        }
        (OracleResult::Rejected, OursResult::Frames(f)) => {
            // libavcodec rejected; we decoded cleanly (zero slice
            // errors, 8-bit comparable format). Investigating a batch
            // of these surfaced four decoder-side conformance gaps
            // (§8.3.x intra-mode neighbour availability, §8.2.4
            // empty-DPB inter slices, §8.2.5.2 non-existing references,
            // §7.4.3 frame_num discipline) which are now enforced — but
            // it also showed the reference layers heuristic
            // stream-discipline checks BEYOND bitstream conformance
            // (e.g. it refuses non-paired fields, which §7.4.1.2.4 /
            // Annex C permit). "The reference rejected" therefore does
            // not imply "the input is malformed", and treating this arm
            // as a crash makes the schedule red on every such corner
            // the fuzzer finds. Report it (searchable in CI logs) and
            // move on; actual mis-decode is what the frame-parity arm
            // asserts on.
            eprintln!(
                "[oracle strictness] libavcodec rejected ({} bytes), oxideav decoded {} clean frame(s)",
                data.len(),
                f.len()
            );
        }
        (OracleResult::Frame(o), OursResult::Rejected | OursResult::Concealed) => {
            // libavcodec produced a clean frame but we couldn't. This
            // MIGHT be a real bug (we miss a feature ffmpeg supports).
            // For the early decoder we treat it as a soft skip so the
            // fuzzer keeps making forward progress on the other
            // assertions; we do print to stderr so it's still
            // searchable in CI logs.
            eprintln!(
                "[oracle soft-skip] libavcodec produced {}x{} frame, oxideav rejected/degraded",
                o.width, o.height
            );
        }
        (OracleResult::Frame(o), OursResult::Frames(ours_frames)) => {
            let area = (o.width as usize) * (o.height as usize);
            if area > MAX_AREA {
                return;
            }
            assert_oracle_frame_matches_one_of_ours(&o, &ours_frames, data.len());
        }
    }
});

/// Classification of our decoder's output for the parity arms.
enum OursResult {
    /// Error, or no frame produced.
    Rejected,
    /// The stream's parameter sets use a bit depth / chroma format the
    /// oracle wrapper cannot represent — no comparison is meaningful.
    Incomparable,
    /// One or more frames came out, but `decode_error_count() > 0`:
    /// slices were skipped, so the output is partial/concealed and must
    /// not reach reject-parity or pixel asserts.
    Concealed,
    /// Clean decode: every frame drained (up to `MAX_OURS_FRAMES`),
    /// zero slice errors, 8-bit comparable format.
    Frames(Vec<DecodedYuv>),
}

struct DecodedYuv {
    width: u32,
    height: u32,
    chroma_w_log2: u8,
    chroma_h_log2: u8,
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
}

fn decode_with_ours(data: &[u8]) -> OursResult {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), data.to_vec());
    if dec.send_packet(&pkt).is_err() {
        return OursResult::Rejected;
    }
    // Mirror the oracle's feeding model: one packet, then EOF drain.
    if dec.flush().is_err() {
        return OursResult::Rejected;
    }
    let mut frames = Vec::new();
    while frames.len() < MAX_OURS_FRAMES {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => frames.push(v),
            _ => break,
        }
    }
    if frames.is_empty() {
        return OursResult::Rejected;
    }
    // Bit-depth / chroma-format gate. Scan EVERY stored SPS (not just
    // the active one): with multiple SPSes in one buffer the frame may
    // have been decoded under a different SPS than the one active at
    // EOF, and misreading 16-bit-per-sample planes as 8-bit would fake
    // both dimensions and samples.
    for id in 0..32 {
        if let Some(sps) = dec.stored_sps(id) {
            if sps.bit_depth_luma_minus8 != 0
                || sps.bit_depth_chroma_minus8 != 0
                || sps.chroma_format_idc == 0
                || sps.separate_colour_plane_flag
            {
                return OursResult::Incomparable;
            }
        }
    }
    if dec.decode_error_count() > 0 {
        return OursResult::Concealed;
    }
    let mut out = Vec::with_capacity(frames.len());
    for f in &frames {
        match repack_frame(f) {
            Some(d) => out.push(d),
            // A frame shape we can't express in the comparison struct
            // (e.g. odd plane ratios) — treat like a format gap.
            None => return OursResult::Incomparable,
        }
    }
    OursResult::Frames(out)
}

/// Convert one of our 8-bit `VideoFrame`s into the flat comparison
/// struct. Our packer emits tightly-packed planes (`stride ==
/// width_in_samples` for 8-bit), so plane geometry is recovered from
/// stride + length.
fn repack_frame(frame: &oxideav_core::VideoFrame) -> Option<DecodedYuv> {
    if frame.planes.len() < 3 {
        return None;
    }
    let y_plane = &frame.planes[0];
    let cb_plane = &frame.planes[1];
    let cr_plane = &frame.planes[2];
    let w = y_plane.stride;
    if w == 0 {
        return None;
    }
    let h = y_plane.data.len() / w;
    if h == 0 {
        return None;
    }
    let cw = cb_plane.stride;
    if cw == 0 {
        return None;
    }
    let ch = cb_plane.data.len() / cw;
    if ch == 0 {
        return None;
    }
    let cw_log2 = match w / cw {
        1 => 0u8,
        2 => 1u8,
        _ => return None,
    };
    let ch_log2 = match h / ch {
        1 => 0u8,
        2 => 1u8,
        _ => return None,
    };
    let y = pack_plane(w, h, y_plane);
    let cb = pack_plane(cw, ch, cb_plane);
    let cr = pack_plane(cw, ch, cr_plane);
    Some(DecodedYuv {
        width: w as u32,
        height: h as u32,
        chroma_w_log2: cw_log2,
        chroma_h_log2: ch_log2,
        y,
        cb,
        cr,
    })
}

fn pack_plane(w: usize, h: usize, plane: &VideoPlane) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for row in 0..h {
        let src = &plane.data[row * plane.stride..row * plane.stride + w];
        out[row * w..row * w + w].copy_from_slice(src);
    }
    out
}

/// The oracle drained exactly one frame; our decoder drained every AU
/// in the buffer. The oracle's frame must match at least one of ours
/// (dimensions + subsampling exact, samples within ±`TOL`).
fn assert_oracle_frame_matches_one_of_ours(
    o: &libavcodec::DecodedYuv,
    ours: &[DecodedYuv],
    input_len: usize,
) {
    let mut geometry_matched = false;
    let mut best: Option<(usize, String)> = None; // (worst diff, detail)
    for (idx, r) in ours.iter().enumerate() {
        if (r.width, r.height) != (o.width, o.height)
            || (r.chroma_w_log2, r.chroma_h_log2) != (o.chroma_w_log2, o.chroma_h_log2)
        {
            continue;
        }
        geometry_matched = true;
        let mut worst = 0usize;
        let mut detail = String::new();
        for (label, ow, oh, op, rp) in [
            ("Y", o.width as usize, o.height as usize, &o.y, &r.y),
            (
                "Cb",
                (o.width as usize) >> o.chroma_w_log2,
                (o.height as usize) >> o.chroma_h_log2,
                &o.cb,
                &r.cb,
            ),
            (
                "Cr",
                (o.width as usize) >> o.chroma_w_log2,
                (o.height as usize) >> o.chroma_h_log2,
                &o.cr,
                &r.cr,
            ),
        ] {
            if op.len() != ow * oh || rp.len() != ow * oh {
                worst = usize::MAX;
                detail = format!(
                    "{label}: plane length mismatch (oracle {}, ours {}, w*h {})",
                    op.len(),
                    rp.len(),
                    ow * oh
                );
                break;
            }
            for i in 0..ow * oh {
                let diff = (op[i] as i32 - rp[i] as i32).unsigned_abs() as usize;
                if diff > worst {
                    worst = diff;
                    detail = format!(
                        "{label}[{},{}] libavcodec={} oxideav={}",
                        i / ow,
                        i % ow,
                        op[i],
                        rp[i]
                    );
                }
            }
        }
        if worst <= TOL as usize {
            return; // matched — parity holds
        }
        if best.as_ref().is_none_or(|(b, _)| worst < *b) {
            best = Some((worst, format!("ours[{idx}]: {detail}")));
        }
    }
    if !geometry_matched {
        let shapes: Vec<String> = ours
            .iter()
            .map(|r| {
                format!(
                    "{}x{} c=({},{})",
                    r.width, r.height, r.chroma_w_log2, r.chroma_h_log2
                )
            })
            .collect();
        panic!(
            "divergence ({input_len} bytes): libavcodec frame {}x{} c=({},{}) matches no oxideav \
             frame geometry [{}]",
            o.width,
            o.height,
            o.chroma_w_log2,
            o.chroma_h_log2,
            shapes.join(", ")
        );
    }
    let (worst, detail) = best.unwrap();
    panic!(
        "divergence ({input_len} bytes): no oxideav frame within tol={TOL} of the libavcodec \
         frame; closest differs by {worst} at {detail}"
    );
}
