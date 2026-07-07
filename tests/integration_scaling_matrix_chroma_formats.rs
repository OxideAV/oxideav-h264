//! Round-397 — **non-flat scaling matrices at 4:2:2 and 4:4:4**
//! (every entropy coder, I / P / B).
//!
//! r388/r391 wired the §7.3.2.1.1.1 scaling-list emit + §8.5.9
//! weightScale quantiser chain at 4:2:0 only; 4:2:2 / 4:4:4 asserted
//! `Flat`. This round lifts that:
//!   * 4:2:2 — the §8.5.11.2 chroma chain quantises / reconstructs
//!     under the chroma weightScale: DC via
//!     `LevelScale4x4(qP % 6, 0, 0)` (eq. 8-328/8-329), AC via the
//!     full 4x4 list; intra (I_16x16 chroma) and inter (P/B) both.
//!   * 4:4:4 — the coded-like-luma chroma planes run the §8.5.10
//!     luma-DC + AC chain (I_16x16), the §8.3.2 8x8 chain (I_8x8) and
//!     the inter forward path under the intra/inter weightScale.
//!   * SPS emits the 12-list loop at 4:4:4 (already in place); the
//!     §7.3.2.2 PPS tail now emits
//!     `6 + ((chroma_format_idc != 3) ? 2 : 6) * transform_8x8` lists.
//!
//! Gates per (format × mode × entropy coder): IDR+P+B GOP bit-exact
//! self-roundtrip AND byte-exact black-box reference-decoder
//! cross-decode, plus a flat-vs-non-flat stream discriminator.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::{
    CustomScalingLists, EncodedFrameRef, Encoder, EncoderConfig, ScalingMatrixMode, YuvFrame,
};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 64;
const H: usize = 64;

fn chroma_dims(fmt: u32) -> (usize, usize) {
    match fmt {
        3 => (W, H),
        2 => (W / 2, H),
        _ => (W / 2, H / 2),
    }
}

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

struct Gop {
    idr: oxideav_h264::encoder::EncodedIdr,
    p: oxideav_h264::encoder::EncodedP,
    b: oxideav_h264::encoder::EncodedB,
    combined: Vec<u8>,
}

fn encode_gop(fmt: u32, cabac: bool, mode: ScalingMatrixMode, transform_8x8: bool) -> Gop {
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
        cabac,
        chroma_format_idc: fmt,
        profile_idc: match fmt {
            3 => 244,
            2 => 122,
            _ => 100,
        },
        max_num_ref_frames: 2,
        transform_8x8,
        scaling_matrix: mode,
        qp: 24,
        ..EncoderConfig::new(W as u32, H as u32)
    };
    let enc = Encoder::new(cfg);
    let (idr, p, b) = if cabac {
        let idr = enc.encode_idr_cabac(&f0);
        let p = enc.encode_p_cabac(&f2, &EncodedFrameRef::from(&idr), 1, 4);
        let b = enc.encode_b_cabac(
            &f1,
            &EncodedFrameRef::from(&idr),
            &EncodedFrameRef::from(&p),
            2,
            2,
        );
        (idr, p, b)
    } else {
        let idr = enc.encode_idr(&f0);
        let p = enc.encode_p(&f2, &EncodedFrameRef::from(&idr), 1, 4);
        let b = enc.encode_b(
            &f1,
            &EncodedFrameRef::from(&idr),
            &EncodedFrameRef::from(&p),
            2,
            2,
        );
        (idr, p, b)
    };
    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p.annex_b);
    combined.extend_from_slice(&b.annex_b);
    Gop {
        idr,
        p,
        b,
        combined,
    }
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
        "oxideav_h264_r397_sm_{tag}_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_r397_sm_{tag}_{}.yuv",
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

/// The non-flat encode must genuinely differ from the flat encode of
/// the same GOP (the weightScale really shapes the coded residual).
fn assert_differs_from_flat(gop: &Gop, fmt: u32, cabac: bool, transform_8x8: bool, tag: &str) {
    let flat = encode_gop(fmt, cabac, ScalingMatrixMode::Flat, transform_8x8);
    assert_ne!(
        flat.combined, gop.combined,
        "{tag}: non-flat stream identical to the flat encode"
    );
}

fn custom_lists() -> CustomScalingLists {
    // Mildly sloped lists (values near 16 so quality stays reasonable),
    // distinct between intra/inter and 4x4/8x8.
    let mut intra4 = [0i32; 16];
    let mut inter4 = [0i32; 16];
    for k in 0..16 {
        intra4[k] = 12 + (k as i32) / 2;
        inter4[k] = 20 - (k as i32) / 2;
    }
    let mut intra8 = [0i32; 64];
    let mut inter8 = [0i32; 64];
    for k in 0..64 {
        intra8[k] = 12 + (k as i32) / 8;
        inter8[k] = 20 - (k as i32) / 8;
    }
    CustomScalingLists {
        intra4,
        inter4,
        intra8,
        inter8,
    }
}

#[test]
fn cavlc_422_seq_default_gop() {
    let gop = encode_gop(2, false, ScalingMatrixMode::SeqDefault, false);
    assert_gop_self_roundtrip(&gop, 2, "sm-cavlc-422-seqdef");
    assert_gop_ffmpeg_interop(&gop, 2, "sm-cavlc-422-seqdef");
    assert_differs_from_flat(&gop, 2, false, false, "sm-cavlc-422-seqdef");
}

#[test]
fn cavlc_444_seq_default_gop() {
    let gop = encode_gop(3, false, ScalingMatrixMode::SeqDefault, false);
    assert_gop_self_roundtrip(&gop, 3, "sm-cavlc-444-seqdef");
    assert_gop_ffmpeg_interop(&gop, 3, "sm-cavlc-444-seqdef");
    assert_differs_from_flat(&gop, 3, false, false, "sm-cavlc-444-seqdef");
}

/// 4:4:4 + PicDefault + transform_8x8 — exercises the §7.3.2.2 PPS
/// tail with the 12-list loop (6 + 6 extra 8x8 lists at 4:4:4).
#[test]
fn cavlc_444_pic_default_transform_8x8_gop() {
    let gop = encode_gop(3, false, ScalingMatrixMode::PicDefault, true);
    assert_gop_self_roundtrip(&gop, 3, "sm-cavlc-444-picdef-8x8");
    assert_gop_ffmpeg_interop(&gop, 3, "sm-cavlc-444-picdef-8x8");
}

#[test]
fn cavlc_422_seq_custom_gop() {
    let gop = encode_gop(
        2,
        false,
        ScalingMatrixMode::SeqCustom(custom_lists()),
        false,
    );
    assert_gop_self_roundtrip(&gop, 2, "sm-cavlc-422-seqcust");
    assert_gop_ffmpeg_interop(&gop, 2, "sm-cavlc-422-seqcust");
}

#[test]
fn cabac_422_seq_default_gop() {
    let gop = encode_gop(2, true, ScalingMatrixMode::SeqDefault, false);
    assert_gop_self_roundtrip(&gop, 2, "sm-cabac-422-seqdef");
    assert_gop_ffmpeg_interop(&gop, 2, "sm-cabac-422-seqdef");
    assert_differs_from_flat(&gop, 2, true, false, "sm-cabac-422-seqdef");
}

#[test]
fn cabac_444_seq_default_gop() {
    let gop = encode_gop(3, true, ScalingMatrixMode::SeqDefault, false);
    assert_gop_self_roundtrip(&gop, 3, "sm-cabac-444-seqdef");
    assert_gop_ffmpeg_interop(&gop, 3, "sm-cabac-444-seqdef");
    assert_differs_from_flat(&gop, 3, true, false, "sm-cabac-444-seqdef");
}

/// CABAC 4:4:4 + SeqCustom + transform_8x8 — the blockCat-5/9/13 8x8
/// residuals quantise under the custom 8x8 lists (12-list SPS loop).
#[test]
fn cabac_444_seq_custom_transform_8x8_gop() {
    let gop = encode_gop(3, true, ScalingMatrixMode::SeqCustom(custom_lists()), true);
    assert_gop_self_roundtrip(&gop, 3, "sm-cabac-444-seqcust-8x8");
    assert_gop_ffmpeg_interop(&gop, 3, "sm-cabac-444-seqcust-8x8");
}

/// CABAC 4:2:2 + PicDefault + transform_8x8 — §7.3.2.2 PPS tail with
/// the 8-list loop at 4:2:2.
#[test]
fn cabac_422_pic_default_transform_8x8_gop() {
    let gop = encode_gop(2, true, ScalingMatrixMode::PicDefault, true);
    assert_gop_self_roundtrip(&gop, 2, "sm-cabac-422-picdef-8x8");
    assert_gop_ffmpeg_interop(&gop, 2, "sm-cabac-422-picdef-8x8");
}
