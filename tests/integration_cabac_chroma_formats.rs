//! Round-388 — decoder-side CABAC 4:2:2 / 4:4:4 validation against a
//! black-box reference encoder + decoder.
//!
//! The staged docs fixtures for 4:4:4 have partially-defective
//! `expected.yuv` planes (see `tests/docs_corpus.rs` round-346 notes)
//! and none of them contains 8x8-transform macroblocks, so the
//! §9.3.3.1.1.9 cat-5/9/13 coded_block_flag neighbour derivation and
//! the §9.3.3.1.3 eq. (9-22) NumC8x8=2 chroma-DC contexts were
//! previously unreachable from the test suite. These tests generate
//! real conformant streams with the reference encoder binary at run
//! time (skipping when the binary is absent), decode them with our
//! decoder, and require byte-exact agreement with the reference
//! decoder's planar YUV output.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use std::path::PathBuf;
use std::process::Command;

const FFMPEG: &str = "/opt/homebrew/bin/ffmpeg";

fn tmp_dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("oxideav-h264-r388-{}", std::process::id()));
    std::fs::create_dir_all(&d).expect("create temp dir");
    d
}

/// Encode `lavfi_src` with the black-box encoder, returning
/// (annex_b_bytes, reference_yuv_bytes) or `None` when the binary is
/// missing or lacks the requested encoder (test skips).
fn make_reference_stream(
    tag: &str,
    lavfi_src: &str,
    profile: &str,
    pix_fmt: &str,
    x264_params: &str,
    preset: Option<&str>,
) -> Option<(Vec<u8>, Vec<u8>)> {
    if !std::path::Path::new(FFMPEG).exists() {
        eprintln!("skip: reference encoder binary not present");
        return None;
    }
    let dir = tmp_dir();
    let h264 = dir.join(format!("in-{tag}.h264"));
    let yuv = dir.join(format!("ref-{tag}.yuv"));
    let mut cmd = Command::new(FFMPEG);
    cmd.args(["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg(lavfi_src)
        .args(["-frames:v", "1", "-c:v", "libx264", "-profile:v", profile]);
    if let Some(p) = preset {
        cmd.args(["-preset", p]);
    }
    cmd.args(["-pix_fmt", pix_fmt, "-coder", "1", "-x264-params"])
        .arg(x264_params)
        .args(["-f", "h264", "-y"])
        .arg(&h264);
    let status = cmd.status().expect("spawn reference encoder");
    if !status.success() {
        eprintln!("skip: reference encoder unavailable for {profile}/{pix_fmt}");
        return None;
    }
    let status = Command::new(FFMPEG)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", pix_fmt, "-y"])
        .arg(&yuv)
        .status()
        .expect("spawn reference decoder");
    assert!(
        status.success(),
        "reference decoder rejected its own stream"
    );
    Some((
        std::fs::read(&h264).expect("read stream"),
        std::fs::read(&yuv).expect("read reference yuv"),
    ))
}

/// Decode `stream` with our decoder; compare all three planes
/// byte-exactly against `reference` (planar, `cw`/`ch` chroma dims).
fn assert_bit_exact(stream: Vec<u8>, reference: &[u8], w: usize, h: usize, cw: usize, ch: usize) {
    let mut dec = oxideav_h264::h264_decoder::H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let vf = loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => break vf,
            Ok(_) => continue,
            Err(e) => panic!("no video frame decoded: {e:?}"),
        }
    };
    let sizes = [(w, h), (cw, ch), (cw, ch)];
    let mut off = 0usize;
    for (p, &(pw, ph)) in sizes.iter().enumerate() {
        let plane = &vf.planes[p];
        for r in 0..ph {
            for c in 0..pw {
                let a = plane.data[r * plane.stride + c];
                let b = reference[off + r * pw + c];
                assert_eq!(
                    a, b,
                    "plane {p} mismatch at ({c}, {r}): ours {a} vs reference {b}"
                );
            }
        }
        off += pw * ph;
    }
    assert_eq!(off, reference.len(), "reference YUV length mismatch");
}

/// High 4:2:2 CABAC intra — exercises the §9.3.3.1.3 eq. (9-22)
/// chroma-DC significance contexts with NumC8x8 = 2 (8-coefficient DC
/// blocks) and the §9.3.3.1.1.9 cat-4 2x4 chroma-AC neighbour wrap.
#[test]
fn cabac_422_reference_stream_decodes_bit_exact() {
    let Some((stream, reference)) = make_reference_stream(
        "422-testsrc",
        "testsrc=size=64x64:rate=25:duration=0.2",
        "high422",
        "yuv422p",
        "keyint=1:8x8dct=0",
        None,
    ) else {
        return;
    };
    assert_bit_exact(stream, &reference, 64, 64, 32, 64);
}

/// High 4:4:4 CABAC intra with 8x8-transform MBs — exercises the
/// §9.3.3.1.1.9 cat-5/9/13 coded_block_flag neighbour derivation
/// (explicitly-coded CBF at ChromaArrayType == 3 per §7.3.5.3.3).
#[test]
fn cabac_444_8x8_reference_stream_decodes_bit_exact() {
    let Some((stream, reference)) = make_reference_stream(
        "444-8x8",
        "mandelbrot=size=64x64:rate=25",
        "high444",
        "yuv444p",
        "keyint=1:qp=20",
        Some("slower"),
    ) else {
        return;
    };
    assert_bit_exact(stream, &reference, 64, 64, 64, 64);
}

/// High 4:2:2 CABAC at low QP with textured content — denser chroma
/// DC + AC blocks walk more eq. (9-22) context transitions.
#[test]
fn cabac_422_textured_low_qp_reference_stream_decodes_bit_exact() {
    let Some((stream, reference)) = make_reference_stream(
        "422-mandelbrot",
        "mandelbrot=size=64x64:rate=25",
        "high422",
        "yuv422p",
        "keyint=1:qp=18",
        Some("slower"),
    ) else {
        return;
    };
    assert_bit_exact(stream, &reference, 64, 64, 32, 64);
}

/// High 4:2:0 CABAC intra with the JVT default quantisation matrices
/// (Table 7-3 / 7-4 signalled via scaling_list()) and the 8x8
/// transform — exercises the §8.5.9 weightScale inverse-scan
/// derivation on both the 4x4 and 8x8 LevelScale paths.
#[test]
fn cabac_420_default_scaling_matrices_decode_bit_exact() {
    let Some((stream, reference)) = make_reference_stream(
        "420-cqm-jvt",
        "mandelbrot=size=64x64:rate=25",
        "high",
        "yuv420p",
        "keyint=1:qp=20:cqm=jvt",
        Some("slower"),
    ) else {
        return;
    };
    assert_bit_exact(stream, &reference, 64, 64, 32, 32);
}

/// CAVLC variant of the default-matrix stream (coder=0) — the
/// scaling-list path is entropy-mode independent but the residual
/// parse is not.
#[test]
fn cavlc_420_default_scaling_matrices_decode_bit_exact() {
    let Some((stream, reference)) = make_reference_stream(
        "420-cqm-jvt-cavlc",
        "mandelbrot=size=64x64:rate=25",
        "high",
        "yuv420p",
        "keyint=1:qp=20:cqm=jvt:cabac=0",
        Some("slower"),
    ) else {
        return;
    };
    assert_bit_exact(stream, &reference, 64, 64, 32, 32);
}

/// Round-391 — High 4:4:4 CABAC **P-slices** (inter blockCat-8/12
/// residual). Pins the twelve §9.3.1.1 Table 9-30/9-31/9-32 P/B-column
/// (m, n) init values (ctxIdx 805/849, 975..980, 1005..1010) that were
/// mis-transcribed until this round: the I-slice columns were correct
/// (all prior 4:4:4 pins were intra-only), so the first inter Cb/Cr
/// coeff_abs_level_minus1 bin desynchronised every CABAC 4:4:4 P
/// decode. Three frames (IDR + 2 P) must decode byte-exact.
#[test]
fn cabac_444_p_slices_reference_stream_decodes_bit_exact() {
    if !std::path::Path::new(FFMPEG).exists() {
        eprintln!("skip: reference encoder binary not present");
        return;
    }
    let dir = tmp_dir();
    let h264 = dir.join("in-444p.h264");
    let yuv = dir.join("ref-444p.yuv");
    let status = Command::new(FFMPEG)
        .args(["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg("testsrc2=size=64x64:rate=25,format=yuv444p")
        .args(["-frames:v", "3", "-c:v", "libx264", "-profile:v", "high444"])
        .args(["-pix_fmt", "yuv444p", "-coder", "1", "-x264-params"])
        .arg(
            "keyint=100:min-keyint=100:bframes=0:ref=1:8x8dct=0:cabac=1:no-deblock=1:\
             subme=1:me=dia:partitions=none:weightp=0:scenecut=0",
        )
        .args(["-f", "h264", "-y"])
        .arg(&h264)
        .status()
        .expect("spawn reference encoder");
    if !status.success() {
        eprintln!("skip: reference encoder unavailable for high444");
        return;
    }
    let status = Command::new(FFMPEG)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv444p", "-y"])
        .arg(&yuv)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success(), "reference decoder rejected the stream");
    let stream = std::fs::read(&h264).expect("read stream");
    let reference = std::fs::read(&yuv).expect("read reference yuv");

    const W: usize = 64;
    const H: usize = 64;
    let frame_len = 3 * W * H; // 4:4:4 planar
    assert_eq!(reference.len(), 3 * frame_len, "expected 3 frames");

    let mut dec = oxideav_h264::h264_decoder::H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut fi = 0usize;
    while let Ok(frame) = dec.receive_frame() {
        let Frame::Video(vf) = frame else { continue };
        let rf = &reference[fi * frame_len..(fi + 1) * frame_len];
        for p in 0..3usize {
            let plane = &vf.planes[p];
            for r in 0..H {
                for c in 0..W {
                    assert_eq!(
                        plane.data[r * plane.stride + c],
                        rf[p * W * H + r * W + c],
                        "frame {fi} plane {p} mismatch at ({c}, {r})"
                    );
                }
            }
        }
        fi += 1;
    }
    assert_eq!(fi, 3, "expected 3 decoded frames");
}
