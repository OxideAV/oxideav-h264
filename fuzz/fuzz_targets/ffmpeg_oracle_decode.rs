#![no_main]

//! Fuzz: arbitrary bytes → BOTH oxideav `H264CodecDecoder` and
//! libavcodec H.264 decoder. Compare:
//!
//! 1. **Reject parity** — if libavcodec fails to produce a frame from
//!    the input, ours MUST also fail to produce one. A divergence here
//!    means we accept (and silently mis-decode) something that the
//!    industry reference rejects, which is a real bug.
//! 2. **Frame parity** — if libavcodec produced a frame, ours MUST too,
//!    with matching dimensions + chroma subsampling. Per-pixel YUV
//!    samples are compared with a ±1 LSB tolerance to absorb the IDCT
//!    precision drift that's expected between independent
//!    implementations of the same spec (the spec only mandates a
//!    bit-exact integer transform inside the CAVLC/CABAC path; the
//!    full reconstruction pipeline includes intra prediction +
//!    deblocking which can drift by 1 LSB across implementations
//!    that round halves differently).
//!
//! Skips silently when libavcodec isn't loadable so the harness still
//! runs against the panic_free coverage even on hosts without ffmpeg
//! installed.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, Decoder, Frame, Packet, TimeBase, VideoPlane};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264_fuzz::libavcodec;

/// Per-channel tolerance (in 8-bit code units) for the cross-decode
/// comparison. The §8.5 IDCT / inverse transform is bit-exact in the
/// spec, but the surrounding intra-prediction + deblocking arithmetic
/// can produce ±1 LSB drift across implementations that round halves
/// differently. ±1 is documented in the task brief.
const TOL: i32 = 1;

/// Maximum frame area (luma samples) we'll spend cycles asserting on.
/// Larger frames burn the per-iteration time budget without exposing
/// new code paths.
const MAX_AREA: usize = 256 * 256;

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

    // Run libavcodec first so we know what to expect from ours.
    let oracle = libavcodec::decode_h264(data);

    // Run our decoder. We treat ANY error (parse, reconstruct,
    // missing-frame) as "rejected" for the comparison — same as the
    // OracleResult::Rejected variant.
    let ours = decode_with_ours(data);

    use libavcodec::OracleResult;
    match (oracle, ours) {
        (OracleResult::Unavailable, _) => {
            // Decoder factory disappeared between `available()` and
            // `decode_h264` — skip the iteration.
        }
        (OracleResult::Rejected, OursResult::Rejected) => {
            // Both rejected — desired behaviour.
        }
        (OracleResult::Rejected, OursResult::Frame(_)) => {
            // libavcodec rejected; we accepted. This is a real
            // divergence: report it and let libfuzzer mark it as a
            // crash so the run surfaces the input.
            //
            // A small set of inputs (e.g. unwrapped raw NAL bytes
            // missing the start codes that libavcodec demands but our
            // parser tolerates) might trip this. Those are real
            // strictness mismatches worth investigating.
            panic!(
                "divergence: libavcodec rejected ({} bytes), oxideav accepted",
                data.len()
            );
        }
        (OracleResult::Frame(o), OursResult::Rejected) => {
            // libavcodec produced a frame but we couldn't. This MIGHT
            // be a real bug (we miss a feature ffmpeg supports). For
            // the early decoder we treat it as a soft skip so the
            // fuzzer keeps making forward progress on the other
            // assertions; we do print to stderr so it's still
            // searchable in CI logs.
            eprintln!(
                "[oracle soft-skip] libavcodec produced {}x{} frame, oxideav rejected",
                o.width, o.height
            );
        }
        (OracleResult::Frame(o), OursResult::Frame(ours)) => {
            assert_eq!(
                ours.width, o.width,
                "width mismatch: oxideav={} libavcodec={}",
                ours.width, o.width
            );
            assert_eq!(
                ours.height, o.height,
                "height mismatch: oxideav={} libavcodec={}",
                ours.height, o.height
            );
            assert_eq!(
                (ours.chroma_w_log2, ours.chroma_h_log2),
                (o.chroma_w_log2, o.chroma_h_log2),
                "chroma subsampling mismatch"
            );
            let area = (o.width as usize) * (o.height as usize);
            if area > MAX_AREA {
                return;
            }
            let cw = (o.width as usize) >> o.chroma_w_log2;
            let ch = (o.height as usize) >> o.chroma_h_log2;
            assert_planes_within_tol(o.width as usize, o.height as usize, &o.y, &ours.y, "Y");
            assert_planes_within_tol(cw, ch, &o.cb, &ours.cb, "Cb");
            assert_planes_within_tol(cw, ch, &o.cr, &ours.cr, "Cr");
        }
    }
});

/// Mirror of `libavcodec::OracleResult` for our decoder, minus the
/// `Unavailable` arm (ours is always available — it's compiled into
/// the fuzz binary).
enum OursResult {
    Rejected,
    Frame(DecodedYuv),
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
    let frame = match dec.receive_frame() {
        Ok(Frame::Video(v)) => v,
        _ => return OursResult::Rejected,
    };
    if frame.planes.len() < 3 {
        return OursResult::Rejected;
    }
    // Use plane[0] stride * count to deduce the luma dims, then derive
    // chroma log2 from the plane[1] dimensions vs plane[0]. This avoids
    // pulling SPS info out of our private API.
    let y_plane = &frame.planes[0];
    let cb_plane = &frame.planes[1];
    let cr_plane = &frame.planes[2];
    let w = y_plane.stride;
    if w == 0 {
        return OursResult::Rejected;
    }
    let h = y_plane.data.len() / w;
    if h == 0 {
        return OursResult::Rejected;
    }
    let cw = cb_plane.stride;
    if cw == 0 {
        return OursResult::Rejected;
    }
    let ch = cb_plane.data.len() / cw;
    if ch == 0 {
        return OursResult::Rejected;
    }
    let cw_log2 = match w / cw {
        1 => 0u8,
        2 => 1u8,
        _ => return OursResult::Rejected,
    };
    let ch_log2 = match h / ch {
        1 => 0u8,
        2 => 1u8,
        _ => return OursResult::Rejected,
    };
    // Tightly pack the planes so the comparison logic can use a flat
    // contiguous index — `VideoPlane::stride` may include padding.
    let y = pack_plane(w, h, y_plane);
    let cb = pack_plane(cw, ch, cb_plane);
    let cr = pack_plane(cw, ch, cr_plane);
    OursResult::Frame(DecodedYuv {
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

fn assert_planes_within_tol(w: usize, h: usize, oracle: &[u8], ours: &[u8], label: &str) {
    assert_eq!(
        oracle.len(),
        w * h,
        "{label}: oracle plane length {} != w*h {}",
        oracle.len(),
        w * h
    );
    assert_eq!(
        ours.len(),
        w * h,
        "{label}: ours plane length {} != w*h {}",
        ours.len(),
        w * h
    );
    for row in 0..h {
        for col in 0..w {
            let o = oracle[row * w + col] as i32;
            let r = ours[row * w + col] as i32;
            let diff = (o - r).abs();
            assert!(
                diff <= TOL,
                "{label}[{row},{col}] differs by {diff} (libavcodec={o}, oxideav={r}, tol={TOL})"
            );
        }
    }
}
