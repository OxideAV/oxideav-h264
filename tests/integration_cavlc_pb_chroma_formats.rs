//! Round-397 — **CAVLC P-slices at 4:2:2 and 4:4:4** (the CABAC legs
//! landed in r391; this closes the CAVLC half of the inter
//! chroma-format matrix).
//!
//! `encode_p` now runs at every chroma format:
//!   * 4:2:2 — half-width full-height chroma MC (§8.4.1.4:
//!     `mvC.y = mvL.y · 2` in 1/8-pel), the §8.5.11.2 2x4-Hadamard
//!     inter chroma chain, `ChromaDc422` coeff_token context for the
//!     8-coefficient ChromaDCLevel, and per-block §9.2.1.1 chroma-AC
//!     nC on the 2x4 grid.
//!   * 4:4:4 — chroma motion-compensated like luma (§8.4.2.2.1 6-tap
//!     on each chroma plane, mvC = mvL) and coded like luma
//!     (§7.3.5.3): Table 9-4(b) inter CBP (luma-only; the quadrant
//!     bits are shared across the three planes), full 16-coefficient
//!     4x4 residual blocks per plane — or the §7.4.5.3.3
//!     de-interleaved 8x8 splits under `transform_8x8` — with the
//!     per-plane `derive_nc_plane_luma_like` nC chain.
//!
//! Gates per shape: CAVLC IDR+P GOP bit-exact self-roundtrip on every
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
        chroma_format_idc: fmt,
        profile_idc: match fmt {
            3 => 244,
            2 => 122,
            _ => 66,
        },
        transform_8x8,
        qp: if transform_8x8 { 20 } else { 26 },
        ..EncoderConfig::new(W as u32, H as u32)
    };
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr(&f0);
    let p = enc.encode_p(&f1, &EncodedFrameRef::from(&idr), 1, 2);
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
        "oxideav_h264_r397_cavpb_{tag}_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_r397_cavpb_{tag}_{}.yuv",
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
fn cavlc_p_422_gop_bit_exact() {
    let gop = encode_gop(2, false);
    assert!(gop.p.annex_b.len() > 100, "P too small — likely all-skip");
    assert_gop_self_roundtrip(&gop, 2, "cavlc-p-422");
    assert_gop_ffmpeg_interop(&gop, 2, "cavlc-p-422");
}

#[test]
fn cavlc_p_444_gop_bit_exact() {
    let gop = encode_gop(3, false);
    assert!(gop.p.annex_b.len() > 100, "P too small — likely all-skip");
    assert_gop_self_roundtrip(&gop, 3, "cavlc-p-444");
    assert_gop_ffmpeg_interop(&gop, 3, "cavlc-p-444");
}

#[test]
fn cavlc_p_422_transform_8x8_gop_bit_exact() {
    let gop = encode_gop(2, true);
    assert_gop_self_roundtrip(&gop, 2, "cavlc-p-422-8x8");
    assert_gop_ffmpeg_interop(&gop, 2, "cavlc-p-422-8x8");
}

/// 4:4:4 + transform_8x8: coded P MBs that pick the 8x8 transform emit
/// each chroma plane's §7.4.5.3.3 de-interleaved splits after luma.
#[test]
fn cavlc_p_444_transform_8x8_gop_bit_exact() {
    let gop = encode_gop(3, true);
    assert!(
        gop.p.i8x8_mb_count > 0,
        "P RDO never picked the 8x8 transform — the 4:4:4 CAVLC 8x8 inter path is unexercised"
    );
    assert_gop_self_roundtrip(&gop, 3, "cavlc-p-444-8x8");
    assert_gop_ffmpeg_interop(&gop, 3, "cavlc-p-444-8x8");
}

/// 4:2:0 regression pin — the multi-format dispatch must keep the
/// legacy path bit-exact end-to-end.
#[test]
fn cavlc_p_420_still_bit_exact() {
    let gop = encode_gop(1, false);
    assert_gop_self_roundtrip(&gop, 1, "cavlc-p-420");
    assert_gop_ffmpeg_interop(&gop, 1, "cavlc-p-420");
}

// ---------------------------------------------------------------------------
// Round-397 — CAVLC B-slices at 4:2:2 / 4:4:4 (B_Skip / B_Direct_16x16
// + the explicit 16x16 mode set).
// ---------------------------------------------------------------------------

/// B target: midpoint blend of frame0 / frameP plus hash texture so
/// coded B MBs carry residual on every plane.
fn make_frame_b(fmt: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (y0, u0, v0) = make_frame0(fmt);
    let (y2, u2, v2) = make_frame_p(fmt);
    let blend = |a: &[u8], b: &[u8]| -> Vec<u8> {
        a.iter()
            .zip(b.iter())
            .enumerate()
            .map(|(k, (&x, &y))| {
                let m = (x as u32 + y as u32) / 2;
                let n = ((k as u32).wrapping_mul(2654435761) >> 27) & 7;
                (m + n).min(235) as u8
            })
            .collect()
    };
    (blend(&y0, &y2), blend(&u0, &u2), blend(&v0, &v2))
}

struct GopB {
    idr: oxideav_h264::encoder::EncodedIdr,
    p: oxideav_h264::encoder::EncodedP,
    b: oxideav_h264::encoder::EncodedB,
    combined: Vec<u8>,
}

fn encode_gop_b(fmt: u32, transform_8x8: bool) -> GopB {
    let (y0, u0, v0) = make_frame0(fmt);
    let (y2, u2, v2) = make_frame_p(fmt);
    let (y1, u1, v1) = make_frame_b(fmt);
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
    let f2 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y2,
        u: &u2,
        v: &v2,
    };
    let cfg = EncoderConfig {
        chroma_format_idc: fmt,
        profile_idc: match fmt {
            3 => 244,
            2 => 122,
            _ => 77,
        },
        max_num_ref_frames: 2,
        transform_8x8,
        qp: if transform_8x8 { 20 } else { 26 },
        ..EncoderConfig::new(W as u32, H as u32)
    };
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr(&f0);
    let p = enc.encode_p(&f2, &EncodedFrameRef::from(&idr), 1, 4);
    let b = enc.encode_b(
        &f1,
        &EncodedFrameRef::from(&idr),
        &EncodedFrameRef::from(&p),
        2,
        2,
    );
    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p.annex_b);
    combined.extend_from_slice(&b.annex_b);
    GopB {
        idr,
        p,
        b,
        combined,
    }
}

fn assert_gop_b_self_roundtrip(gop: &GopB, fmt: u32, tag: &str) {
    let (cw, ch) = chroma_dims(fmt);
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), gop.combined.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    while let Ok(Frame::Video(vf)) = dec.receive_frame() {
        frames.push(vf);
    }
    assert_eq!(frames.len(), 3, "{tag}: expected IDR + B + P frames");
    for (idx, (ry, ru, rv, name)) in [
        (&gop.idr.recon_y, &gop.idr.recon_u, &gop.idr.recon_v, "IDR"),
        (&gop.b.recon_y, &gop.b.recon_u, &gop.b.recon_v, "B"),
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

fn assert_gop_b_ffmpeg_interop(gop: &GopB, fmt: u32, tag: &str) {
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
        "oxideav_h264_r397_cavb_{tag}_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_r397_cavb_{tag}_{}.yuv",
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
    assert_eq!(yuv.len(), 3 * frame_len, "{tag}: expected 3 frames");
    for (idx, (ry, ru, rv, name)) in [
        (&gop.idr.recon_y, &gop.idr.recon_u, &gop.idr.recon_v, "IDR"),
        (&gop.b.recon_y, &gop.b.recon_u, &gop.b.recon_v, "B"),
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
fn cavlc_b_422_gop_bit_exact() {
    let gop = encode_gop_b(2, false);
    assert!(gop.b.annex_b.len() > 100, "B too small — likely all-zero");
    assert_gop_b_self_roundtrip(&gop, 2, "cavlc-b-422");
    assert_gop_b_ffmpeg_interop(&gop, 2, "cavlc-b-422");
}

#[test]
fn cavlc_b_444_gop_bit_exact() {
    let gop = encode_gop_b(3, false);
    assert!(gop.b.annex_b.len() > 100, "B too small — likely all-zero");
    assert_gop_b_self_roundtrip(&gop, 3, "cavlc-b-444");
    assert_gop_b_ffmpeg_interop(&gop, 3, "cavlc-b-444");
}

#[test]
fn cavlc_b_422_transform_8x8_gop_bit_exact() {
    let gop = encode_gop_b(2, true);
    assert_gop_b_self_roundtrip(&gop, 2, "cavlc-b-422-8x8");
    assert_gop_b_ffmpeg_interop(&gop, 2, "cavlc-b-422-8x8");
}

#[test]
fn cavlc_b_444_transform_8x8_gop_bit_exact() {
    let gop = encode_gop_b(3, true);
    assert_gop_b_self_roundtrip(&gop, 3, "cavlc-b-444-8x8");
    assert_gop_b_ffmpeg_interop(&gop, 3, "cavlc-b-444-8x8");
}

/// Static B at 4:2:2 / 4:4:4 — B_Skip (spatial direct, cbp == 0) must
/// carry through the §7.3.4 mb_skip_run walker at every chroma format.
#[test]
fn cavlc_b_skip_all_formats() {
    for fmt in [1u32, 2, 3] {
        let (cw, ch) = chroma_dims(fmt);
        let y0 = vec![120u8; W * H];
        let u0 = vec![90u8; cw * ch];
        let v0 = vec![160u8; cw * ch];
        let f0 = YuvFrame {
            width: W as u32,
            height: H as u32,
            y: &y0,
            u: &u0,
            v: &v0,
        };
        let cfg = EncoderConfig {
            chroma_format_idc: fmt,
            profile_idc: match fmt {
                3 => 244,
                2 => 122,
                _ => 77,
            },
            max_num_ref_frames: 2,
            ..EncoderConfig::new(W as u32, H as u32)
        };
        let enc = Encoder::new(cfg);
        let idr = enc.encode_idr(&f0);
        let p = enc.encode_p(&f0, &EncodedFrameRef::from(&idr), 1, 4);
        let b = enc.encode_b(
            &f0,
            &EncodedFrameRef::from(&idr),
            &EncodedFrameRef::from(&p),
            2,
            2,
        );
        assert!(
            b.annex_b.len() < 32,
            "fmt {fmt}: static-flat B should collapse to skip runs \
             (B = {} bytes)",
            b.annex_b.len()
        );
        let mut combined = Vec::new();
        combined.extend_from_slice(&idr.annex_b);
        combined.extend_from_slice(&p.annex_b);
        combined.extend_from_slice(&b.annex_b);
        let gop = GopB {
            idr,
            p,
            b,
            combined,
        };
        assert_gop_b_self_roundtrip(&gop, fmt, &format!("cavlc-b-skip-{fmt}"));
        assert_gop_b_ffmpeg_interop(&gop, fmt, &format!("cavlc-b-skip-{fmt}"));
    }
}

// ---------------------------------------------------------------------------
// Round-397 — B 16x8 / 8x16 partitions at 4:2:2 / 4:4:4 (CAVLC).
// ---------------------------------------------------------------------------

/// IDR: smooth ramp + textured chroma; P: ramp shifted right by 4 px
/// (integer-pel motion on every plane); B: per-half motion — one half
/// of every MB matches the IDR content, the other the P content —
/// which trips the round-22 16x8 / 8x16 partition trial. The chroma
/// planes carry gradients so the per-format partition chroma-rect MC
/// (§8.4.1.4 at 4:2:2 / §8.4.2.2.1 at 4:4:4) is exercised with real
/// residual on both halves.
fn partition_fixture(fmt: u32, vertical_split: bool) -> GopB {
    let (cw, ch) = chroma_dims(fmt);
    let mut y0 = vec![0u8; W * H];
    let mut u0 = vec![0u8; cw * ch];
    let mut v0 = vec![0u8; cw * ch];
    for j in 0..H {
        for i in 0..W {
            y0[j * W + i] = (60 + i + j) as u8;
        }
    }
    for j in 0..ch {
        for i in 0..cw {
            u0[j * cw + i] = (70 + i * 120 / cw + j * 30 / ch) as u8;
            v0[j * cw + i] = (180 - (i * 90 / cw) - j * 20 / ch) as u8;
        }
    }
    // P: every plane shifted right by 4 LUMA pixels (chroma shift = 4
    // at full-width 4:4:4, 2 at half-width 4:2:0/4:2:2).
    let cshift = if fmt == 3 { 4usize } else { 2 };
    let mut y2 = vec![0u8; W * H];
    let mut u2 = vec![0u8; cw * ch];
    let mut v2 = vec![0u8; cw * ch];
    for j in 0..H {
        for i in 0..W {
            y2[j * W + i] = y0[j * W + i.saturating_sub(4)];
        }
    }
    for j in 0..ch {
        for i in 0..cw {
            u2[j * cw + i] = u0[j * cw + i.saturating_sub(cshift)];
            v2[j * cw + i] = v0[j * cw + i.saturating_sub(cshift)];
        }
    }
    // B: per-MB-half selection between the two anchors.
    let mut y1 = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let from_idr = if vertical_split {
                i % 16 < 8
            } else {
                j % 16 < 8
            };
            y1[j * W + i] = if from_idr {
                y0[j * W + i]
            } else {
                y2[j * W + i]
            };
        }
    }
    // Chroma halves follow the same split in chroma coordinates.
    let (mcw, mch) = match fmt {
        3 => (16usize, 16usize),
        2 => (8, 16),
        _ => (8, 8),
    };
    let mut u1 = vec![0u8; cw * ch];
    let mut v1 = vec![0u8; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            let from_idr = if vertical_split {
                i % mcw < mcw / 2
            } else {
                j % mch < mch / 2
            };
            u1[j * cw + i] = if from_idr {
                u0[j * cw + i]
            } else {
                u2[j * cw + i]
            };
            v1[j * cw + i] = if from_idr {
                v0[j * cw + i]
            } else {
                v2[j * cw + i]
            };
        }
    }
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
    let f2 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y2,
        u: &u2,
        v: &v2,
    };
    let cfg = EncoderConfig {
        chroma_format_idc: fmt,
        profile_idc: match fmt {
            3 => 244,
            2 => 122,
            _ => 77,
        },
        max_num_ref_frames: 2,
        ..EncoderConfig::new(W as u32, H as u32)
    };
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr(&f0);
    let p = enc.encode_p(&f2, &EncodedFrameRef::from(&idr), 1, 4);
    let b = enc.encode_b(
        &f1,
        &EncodedFrameRef::from(&idr),
        &EncodedFrameRef::from(&p),
        2,
        2,
    );
    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p.annex_b);
    combined.extend_from_slice(&b.annex_b);
    GopB {
        idr,
        p,
        b,
        combined,
    }
}

#[test]
fn cavlc_b_16x8_partitions_422_444() {
    for fmt in [2u32, 3] {
        let gop = partition_fixture(fmt, false);
        assert_gop_b_self_roundtrip(&gop, fmt, &format!("cavlc-b-16x8-{fmt}"));
        assert_gop_b_ffmpeg_interop(&gop, fmt, &format!("cavlc-b-16x8-{fmt}"));
    }
}

#[test]
fn cavlc_b_8x16_partitions_422_444() {
    for fmt in [2u32, 3] {
        let gop = partition_fixture(fmt, true);
        assert_gop_b_self_roundtrip(&gop, fmt, &format!("cavlc-b-8x16-{fmt}"));
        assert_gop_b_ffmpeg_interop(&gop, fmt, &format!("cavlc-b-8x16-{fmt}"));
    }
}

/// Static P at 4:2:2 / 4:4:4 — the §7.3.4 mb_skip_run walker must
/// carry P_Skip MBs at every chroma format (the skip recon copies the
/// format-sized chroma predictor tile).
#[test]
fn cavlc_p_skip_all_formats() {
    for fmt in [1u32, 2, 3] {
        // Per-plane flat source: DC intra prediction reconstructs it
        // exactly, so re-encoding the same picture as a P-slice must
        // collapse every MB to P_Skip (one long §7.3.4 mb_skip_run).
        let (cw, ch) = chroma_dims(fmt);
        let y0 = vec![120u8; W * H];
        let u0 = vec![90u8; cw * ch];
        let v0 = vec![160u8; cw * ch];
        let f0 = YuvFrame {
            width: W as u32,
            height: H as u32,
            y: &y0,
            u: &u0,
            v: &v0,
        };
        let cfg = EncoderConfig {
            chroma_format_idc: fmt,
            profile_idc: match fmt {
                3 => 244,
                2 => 122,
                _ => 66,
            },
            ..EncoderConfig::new(W as u32, H as u32)
        };
        let enc = Encoder::new(cfg);
        let idr = enc.encode_idr(&f0);
        // Same source again — MBs collapse to P_Skip (4:2:0 / 4:4:4
        // reach a single 10-byte skip run; 4:2:2 keeps a few coded
        // chroma-DC MBs from the 2x4-Hadamard quantisation noise).
        let p = enc.encode_p(&f0, &EncodedFrameRef::from(&idr), 1, 2);
        assert!(
            p.annex_b.len() < idr.annex_b.len() / 2,
            "fmt {fmt}: static-flat P should be dominated by skip runs \
             (P = {} bytes, IDR = {})",
            p.annex_b.len(),
            idr.annex_b.len()
        );
        let mut combined = Vec::new();
        combined.extend_from_slice(&idr.annex_b);
        combined.extend_from_slice(&p.annex_b);
        let gop = Gop { idr, p, combined };
        assert_gop_self_roundtrip(&gop, fmt, &format!("cavlc-p-skip-{fmt}"));
        assert_gop_ffmpeg_interop(&gop, fmt, &format!("cavlc-p-skip-{fmt}"));
    }
}
