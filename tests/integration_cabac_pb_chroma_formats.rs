//! Round-391 — **CABAC P-slices at 4:2:2 and 4:4:4** (inter residual
//! under the blockCat-9/13 contexts landed in r388/r390).
//!
//! `encode_p_cabac` now runs at every chroma format:
//!   * 4:2:2 — half-width full-height chroma MC (§8.4.1.4:
//!     `mvC.y = mvL.y · 2` in 1/8-pel), the §8.5.11.2 2x4-Hadamard
//!     inter chroma chain, and the r388 NumC8x8 = 2 CABAC contexts
//!     (8-coefficient ChromaDCLevel + the 2x4 chroma-AC grid).
//!   * 4:4:4 — chroma motion-compensated like luma (§8.4.2.2.1 6-tap
//!     on each chroma plane, mvC = mvL) and coded like luma
//!     (§7.3.5.3): shared CBP quadrant bits across the three planes,
//!     blockCat-8/12 full 4x4 residuals — or blockCat-5/9/13 8x8
//!     residuals with explicitly coded CBF under `transform_8x8`.
//!
//! Gates per shape: CABAC IDR+P GOP bit-exact self-roundtrip on every
//! plane AND byte-exact black-box reference-decoder cross-decode.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 64;
const H: usize = 64;

/// Chroma plane dims per chroma_format_idc.
fn chroma_dims(fmt: u32) -> (usize, usize) {
    match fmt {
        3 => (W, H),
        2 => (W / 2, H),
        _ => (W / 2, H / 2),
    }
}

/// Textured luma + non-flat chroma at the requested format.
fn make_frame0(fmt: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let base = 40 + ((i * 2 + j) as u32 % 160);
            let tex = if i >= 32 && j >= 32 {
                (i as u32 * 37 + j as u32 * 101 + (i as u32 ^ j as u32) * 13) % 61
            } else {
                0
            };
            y[j * W + i] = (base + tex).min(235) as u8;
        }
    }
    let (cw, ch) = chroma_dims(fmt);
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            let tex = ((i * 29 + j * 53) % 33) as u32;
            u[j * cw + i] = (60 + (i * 160 / cw) as u32 + tex / 2).min(235) as u8;
            v[j * cw + i] = (200u32.saturating_sub((j * 160 / ch) as u32) + tex / 3).min(235) as u8;
        }
    }
    (y, u, v)
}

/// P target: brightness ramp on the left luma half + a chroma drift so
/// coded P MBs carry both luma and chroma residual.
fn make_frame_p(fmt: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (mut y, mut u, mut v) = make_frame0(fmt);
    for j in 0..H {
        for i in 0..W / 2 {
            let bump = 10 + (i + j) / 4;
            let s = y[j * W + i] as usize + bump;
            y[j * W + i] = s.min(235) as u8;
        }
    }
    let (cw, ch) = chroma_dims(fmt);
    for j in 0..ch {
        for i in 0..cw {
            u[j * cw + i] = u[j * cw + i].saturating_add((4 + (i + j) % 7) as u8);
            v[j * cw + i] = v[j * cw + i].saturating_sub((3 + i % 5) as u8);
        }
    }
    (y, u, v)
}

struct Gop {
    idr: oxideav_h264::encoder::EncodedIdr,
    p: oxideav_h264::encoder::EncodedP,
    combined: Vec<u8>,
}

fn encode_gop(fmt: u32, transform_8x8: bool) -> Gop {
    let (y0, u0, v0) = make_frame0(fmt);
    let (y1, u1, v1) = make_frame_p(fmt);
    let f0 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y0,
        u: &u0,
        v: &v0,
    };
    let f1 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y1,
        u: &u1,
        v: &v1,
    };
    let cfg = EncoderConfig {
        cabac: true,
        chroma_format_idc: fmt,
        profile_idc: match fmt {
            3 => 244,
            2 => 122,
            _ => 100,
        },
        transform_8x8,
        qp: if transform_8x8 { 20 } else { 26 },
        ..EncoderConfig::new(W as u32, H as u32)
    };
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr_cabac(&f0);
    let p = enc.encode_p_cabac(&f1, &EncodedFrameRef::from(&idr), 1, 2);
    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p.annex_b);
    Gop { idr, p, combined }
}

fn plane_max_diff(vf: &VideoFrame, plane: usize, recon: &[u8], w: usize, h: usize) -> u32 {
    let p = &vf.planes[plane];
    let mut max = 0u32;
    for r in 0..h {
        for c in 0..w {
            let a = p.data[r * p.stride + c] as i32;
            let b = recon[r * w + c] as i32;
            max = max.max(a.abs_diff(b));
        }
    }
    max
}

fn assert_gop_self_roundtrip(gop: &Gop, fmt: u32, tag: &str) {
    let (cw, ch) = chroma_dims(fmt);
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), gop.combined.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    while let Ok(Frame::Video(vf)) = dec.receive_frame() {
        frames.push(vf);
    }
    assert_eq!(frames.len(), 2, "{tag}: expected IDR + P frames");
    for (idx, (ry, ru, rv, name)) in [
        (&gop.idr.recon_y, &gop.idr.recon_u, &gop.idr.recon_v, "IDR"),
        (&gop.p.recon_y, &gop.p.recon_u, &gop.p.recon_v, "P"),
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(
            plane_max_diff(&frames[idx], 0, ry, W, H),
            0,
            "{tag}: {name} luma"
        );
        assert_eq!(
            plane_max_diff(&frames[idx], 1, ru, cw, ch),
            0,
            "{tag}: {name} Cb"
        );
        assert_eq!(
            plane_max_diff(&frames[idx], 2, rv, cw, ch),
            0,
            "{tag}: {name} Cr"
        );
    }
}

fn assert_gop_ffmpeg_interop(gop: &Gop, fmt: u32, tag: &str) {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: reference decoder binary not present");
        return;
    }
    let (cw, ch) = chroma_dims(fmt);
    let pix_fmt = match fmt {
        3 => "yuv444p",
        2 => "yuv422p",
        _ => "yuv420p",
    };
    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!(
        "oxideav_h264_r391_cabpb_{tag}_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_r391_cabpb_{tag}_{}.yuv",
        std::process::id()
    ));
    std::fs::write(&h264_path, &gop.combined).expect("write h264");
    let status = std::process::Command::new(ffmpeg)
        .args(["-loglevel", "error", "-y", "-f", "h264", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", pix_fmt])
        .arg(&yuv_path)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success(), "reference decoder rejected {tag}");
    let yuv = std::fs::read(&yuv_path).expect("read decoded yuv");
    let _ = std::fs::remove_file(&h264_path);
    let _ = std::fs::remove_file(&yuv_path);
    let frame_len = W * H + 2 * cw * ch;
    assert_eq!(yuv.len(), 2 * frame_len, "{tag}: expected 2 frames");
    for (idx, (ry, ru, rv, name)) in [
        (&gop.idr.recon_y, &gop.idr.recon_u, &gop.idr.recon_v, "IDR"),
        (&gop.p.recon_y, &gop.p.recon_u, &gop.p.recon_v, "P"),
    ]
    .iter()
    .enumerate()
    {
        let f = &yuv[idx * frame_len..(idx + 1) * frame_len];
        assert_eq!(&f[..W * H], &ry[..], "{tag}: {name} luma vs reference");
        assert_eq!(
            &f[W * H..W * H + cw * ch],
            &ru[..],
            "{tag}: {name} Cb vs reference"
        );
        assert_eq!(
            &f[W * H + cw * ch..],
            &rv[..],
            "{tag}: {name} Cr vs reference"
        );
    }
}

#[test]
fn cabac_p_422_gop_bit_exact() {
    let gop = encode_gop(2, false);
    assert!(gop.p.annex_b.len() > 100, "P too small — likely all-skip");
    assert_gop_self_roundtrip(&gop, 2, "cabac-p-422");
    assert_gop_ffmpeg_interop(&gop, 2, "cabac-p-422");
}

#[test]
fn cabac_p_444_gop_bit_exact() {
    let gop = encode_gop(3, false);
    assert!(gop.p.annex_b.len() > 100, "P too small — likely all-skip");
    assert_gop_self_roundtrip(&gop, 3, "cabac-p-444");
    assert_gop_ffmpeg_interop(&gop, 3, "cabac-p-444");
}

#[test]
fn cabac_p_422_transform_8x8_gop_bit_exact() {
    let gop = encode_gop(2, true);
    assert_gop_self_roundtrip(&gop, 2, "cabac-p-422-8x8");
    assert_gop_ffmpeg_interop(&gop, 2, "cabac-p-422-8x8");
}

/// 4:4:4 + transform_8x8: coded P MBs that pick the 8x8 transform emit
/// blockCat-5/9/13 inter residuals with explicitly coded CBF.
#[test]
fn cabac_p_444_transform_8x8_gop_bit_exact() {
    let gop = encode_gop(3, true);
    assert!(
        gop.p.i8x8_mb_count > 0,
        "P RDO never picked the 8x8 transform — blockCat-9/13 inter path unexercised"
    );
    assert_gop_self_roundtrip(&gop, 3, "cabac-p-444-8x8");
    assert_gop_ffmpeg_interop(&gop, 3, "cabac-p-444-8x8");
}

/// 4:2:0 regression pin — the multi-format dispatch must keep the
/// legacy path bit-exact end-to-end.
#[test]
fn cabac_p_420_still_bit_exact() {
    let gop = encode_gop(1, false);
    assert_gop_self_roundtrip(&gop, 1, "cabac-p-420");
    assert_gop_ffmpeg_interop(&gop, 1, "cabac-p-420");
}
