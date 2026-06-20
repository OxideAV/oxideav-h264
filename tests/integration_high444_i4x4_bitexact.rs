//! Bit-exact gate for the High 4:4:4 Predictive I_4x4 decode path.
//!
//! The `intra-only-high444` corpus fixture codes two all-intra 4:4:4
//! (ChromaArrayType == 3) pictures. Its `expected.yuv` frame 0 is a
//! known-defective dump (uniform-128 luma, see `docs_corpus.rs`), so the
//! aggregate corpus case stays `ReportOnly`. Frame 1, however, is a
//! valid high-QP picture whose reference YUV is correct, and our decoder
//! reproduces it byte-for-byte. This test isolates frame 1 and asserts
//! bit-exactness, giving the 4:4:4 I_NxN path (luma Intra_4x4 + the
//! §7.3.5.3 / §8.3.4.5 "chroma coded like luma" Cb/Cr reconstruction
//! that reuses the per-block luma pred modes) an enforced CI gate that
//! the multi-frame corpus case cannot provide while frame 0's reference
//! is bogus.
//!
//! Spec references:
//! * §7.3.5.3 — `residual_luma()` / 4:4:4 chroma residual.
//! * §8.3.4.5 — Intra prediction for 4:4:4 chroma "coded like luma".
//! * §6.2 Table 6-1 — ChromaArrayType == 3 (luma and chroma planes are
//!   the same width/height).
//!
//! No external decoder source is consulted; the reference YUV is an
//! opaque byte vector produced separately by a reference H.264 decoder
//! binary and staged in the private docs repo.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use std::path::PathBuf;

const W: usize = 32;
const H: usize = 32;
// 4:4:4: luma and both chroma planes are full resolution.
const FRAME_BYTES: usize = W * H * 3;

fn fixture_dir() -> PathBuf {
    PathBuf::from("../../docs/video/h264/fixtures/intra-only-high444")
}

/// Repack a 4:4:4 `VideoFrame` into planar Y||U||V row-major bytes,
/// stripping per-row stride padding so it can be compared against the
/// reference dump byte-for-byte.
fn pack_444(vf: &VideoFrame) -> Option<Vec<u8>> {
    if vf.planes.len() < 3 {
        return None;
    }
    let mut out = Vec::with_capacity(FRAME_BYTES);
    for p in 0..3 {
        let plane = &vf.planes[p];
        if plane.stride < W {
            return None;
        }
        for r in 0..H {
            let start = r * plane.stride;
            let end = start + W;
            if end > plane.data.len() {
                return None;
            }
            out.extend_from_slice(&plane.data[start..end]);
        }
    }
    Some(out)
}

#[test]
fn high444_i4x4_frame1_is_bit_exact() {
    let dir = fixture_dir();
    let h264_path = dir.join("input.h264");
    let yuv_path = dir.join("expected.yuv");
    let h264 = match std::fs::read(&h264_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip: missing {} ({e})", h264_path.display());
            return;
        }
    };
    let yuv = match std::fs::read(&yuv_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip: missing {} ({e})", yuv_path.display());
            return;
        }
    };
    // The fixture must hold exactly two 4:4:4 frames.
    assert_eq!(
        yuv.len(),
        2 * FRAME_BYTES,
        "unexpected expected.yuv size {} (want {})",
        yuv.len(),
        2 * FRAME_BYTES
    );

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), h264).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");

    let mut frames: Vec<Vec<u8>> = Vec::new();
    while let Ok(f) = dec.receive_frame() {
        if let Frame::Video(vf) = f {
            frames.push(pack_444(&vf).expect("4:4:4 frame repack"));
        }
    }

    assert!(
        frames.len() >= 2,
        "expected at least 2 decoded frames, got {}",
        frames.len()
    );

    // Frame 1 (decode order == display order, bf=0) is the valid
    // high-QP picture. Compare it byte-for-byte against expected[1].
    let our = &frames[1];
    let reference = &yuv[FRAME_BYTES..2 * FRAME_BYTES];
    assert_eq!(our.len(), FRAME_BYTES, "packed frame size mismatch");

    // Locate the first divergence (if any) for a useful failure message.
    if let Some((idx, (&a, &b))) = our
        .iter()
        .zip(reference.iter())
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        let (plane, off) = if idx < W * H {
            ("Y", idx)
        } else if idx < 2 * W * H {
            ("U", idx - W * H)
        } else {
            ("V", idx - 2 * W * H)
        };
        let (px, py) = (off % W, off / W);
        panic!(
            "4:4:4 I_4x4 frame 1 not bit-exact: first diff at {plane} ({px},{py}) \
             ours={a} ref={b} (byte index {idx})"
        );
    }
}
