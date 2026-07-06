//! Round-391 — **non-flat scaling matrices on P / B slices** (CAVLC).
//!
//! Round-388 wired the §7.3.2.1.1.1 scaling-list emit + the §8.5.9
//! weightScale quantiser chain on the CAVLC IDR path only; P / B
//! slices asserted `Flat`. This round lifts that: the inter luma
//! residual (4x4 and §8.6.4 8x8), the inter chroma residual, and the
//! P/B **intra-fallback** MBs all quantise / dequantise under the
//! matching lists — Default_4x4/8x8_Inter (or the custom inter lists)
//! for inter MBs, the intra lists for I_16x16-in-P/B MBs — mirroring
//! the decoder's per-MB `select_scaling_list_*` dispatch.
//!
//! Gates per mode (SeqDefault / SeqCustom / PicDefault / PicCustom,
//! ± transform_8x8):
//!   * IDR + P + B GOP self-roundtrip bit-exact on all planes.
//!   * Black-box reference-decoder cross-decode byte-exact.
//!   * A discriminator pin: the non-flat P stream must differ from
//!     the flat encode of the same GOP (the inter weightScale really
//!     shapes the coded residual).

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::{
    CustomScalingLists, EncodedFrameRef, Encoder, EncoderConfig, ScalingMatrixMode, YuvFrame,
};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 64;
const H: usize = 64;

/// Textured base frame (gradient + texture quadrant + oriented bars)
/// with non-flat chroma.
fn make_frame0() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
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
    let cw = W / 2;
    let ch = H / 2;
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            u[j * cw + i] = (80 + i as u32 * 2) as u8;
            v[j * cw + i] = (190u32.saturating_sub(j as u32 * 3)).max(30) as u8;
        }
    }
    (y, u, v)
}

/// P target: frame0 with a brightness ramp over the left half plus a
/// small chroma drift — enough inter residual that coded (non-skip)
/// P MBs flow through the weighted inter quantiser.
fn make_frame_p() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (mut y, mut u, v) = make_frame0();
    for j in 0..H {
        for i in 0..W / 2 {
            let bump = 10 + (i + j) / 4;
            let s = y[j * W + i] as usize + bump;
            y[j * W + i] = s.min(235) as u8;
        }
    }
    for uv in u.iter_mut() {
        *uv = uv.saturating_add(6);
    }
    (y, u, v)
}

/// B target: midpoint blend of frame0 / frameP plus hash texture so
/// coded B MBs carry residual.
fn make_frame_b(y0: &[u8], y2: &[u8]) -> Vec<u8> {
    let mut y1 = vec![0u8; W * H];
    for k in 0..W * H {
        let blend = (y0[k] as u32 + y2[k] as u32) / 2;
        let n = ((k as u32).wrapping_mul(2654435761) >> 27) & 7;
        y1[k] = (blend + n).min(235) as u8;
    }
    y1
}

fn custom_lists() -> CustomScalingLists {
    let mut intra4 = [0i32; 16];
    let mut inter4 = [0i32; 16];
    for j in 0..16 {
        intra4[j] = 12 + (j as i32) * 3;
        inter4[j] = 14 + (j as i32) * 2;
    }
    let mut intra8 = [0i32; 64];
    let mut inter8 = [0i32; 64];
    for j in 0..64 {
        intra8[j] = 10 + (j as i32);
        inter8[j] = 12 + (j as i32) / 2;
    }
    CustomScalingLists {
        intra4,
        inter4,
        intra8,
        inter8,
    }
}

struct Gop {
    idr: oxideav_h264::encoder::EncodedIdr,
    p: oxideav_h264::encoder::EncodedP,
    b: oxideav_h264::encoder::EncodedB,
    combined: Vec<u8>,
}

fn encode_gop(mode: ScalingMatrixMode, transform_8x8: bool) -> Gop {
    let (y0, u0, v0) = make_frame0();
    let (y2, u2, v2) = make_frame_p();
    let y1 = make_frame_b(&y0, &y2);
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
        u: &u0,
        v: &v0,
    };
    let f2 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y2,
        u: &u2,
        v: &v2,
    };
    let cfg = EncoderConfig {
        scaling_matrix: mode,
        transform_8x8,
        profile_idc: 100,
        max_num_ref_frames: 2,
        qp: if transform_8x8 { 18 } else { 26 },
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

/// Decode the GOP through our decoder and assert every frame (display
/// order IDR, B, P) reproduces the encoder recon bit-exactly on all
/// three planes.
fn assert_gop_self_roundtrip(gop: &Gop, tag: &str) {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), gop.combined.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    while let Ok(Frame::Video(vf)) = dec.receive_frame() {
        frames.push(vf);
    }
    assert_eq!(frames.len(), 3, "{tag}: expected IDR + B + P frames");
    let cw = W / 2;
    let ch = H / 2;
    for (idx, (recon_y, recon_u, recon_v, name)) in [
        (&gop.idr.recon_y, &gop.idr.recon_u, &gop.idr.recon_v, "IDR"),
        (&gop.b.recon_y, &gop.b.recon_u, &gop.b.recon_v, "B"),
        (&gop.p.recon_y, &gop.p.recon_u, &gop.p.recon_v, "P"),
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(
            plane_max_diff(&frames[idx], 0, recon_y, W, H),
            0,
            "{tag}: {name} luma"
        );
        assert_eq!(
            plane_max_diff(&frames[idx], 1, recon_u, cw, ch),
            0,
            "{tag}: {name} Cb"
        );
        assert_eq!(
            plane_max_diff(&frames[idx], 2, recon_v, cw, ch),
            0,
            "{tag}: {name} Cr"
        );
    }
}

/// Cross-decode the GOP through the black-box reference decoder and
/// assert byte-exactness against the encoder recons (display order).
fn assert_gop_ffmpeg_interop(gop: &Gop, tag: &str) {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: reference decoder binary not present");
        return;
    }
    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!(
        "oxideav_h264_r391_pbsm_{tag}_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_r391_pbsm_{tag}_{}.yuv",
        std::process::id()
    ));
    std::fs::write(&h264_path, &gop.combined).expect("write h264");
    let status = std::process::Command::new(ffmpeg)
        .args(["-loglevel", "error", "-y", "-f", "h264", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success(), "reference decoder rejected {tag}");
    let yuv = std::fs::read(&yuv_path).expect("read decoded yuv");
    let _ = std::fs::remove_file(&h264_path);
    let _ = std::fs::remove_file(&yuv_path);
    let frame_len = W * H + 2 * (W / 2) * (H / 2);
    assert_eq!(yuv.len(), 3 * frame_len, "{tag}: expected 3 frames");
    for (idx, (recon_y, recon_u, recon_v, name)) in [
        (&gop.idr.recon_y, &gop.idr.recon_u, &gop.idr.recon_v, "IDR"),
        (&gop.b.recon_y, &gop.b.recon_u, &gop.b.recon_v, "B"),
        (&gop.p.recon_y, &gop.p.recon_u, &gop.p.recon_v, "P"),
    ]
    .iter()
    .enumerate()
    {
        let f = &yuv[idx * frame_len..(idx + 1) * frame_len];
        assert_eq!(&f[..W * H], &recon_y[..], "{tag}: {name} luma vs reference");
        assert_eq!(
            &f[W * H..W * H + (W / 2) * (H / 2)],
            &recon_u[..],
            "{tag}: {name} Cb vs reference"
        );
        assert_eq!(
            &f[W * H + (W / 2) * (H / 2)..],
            &recon_v[..],
            "{tag}: {name} Cr vs reference"
        );
    }
}

#[test]
fn seq_default_pb_gop_bit_exact() {
    let gop = encode_gop(ScalingMatrixMode::SeqDefault, false);
    // Vacuity guard: the P/B slices must carry real coded residual.
    assert!(gop.p.annex_b.len() > 100, "P too small — likely all-skip");
    assert!(gop.b.annex_b.len() > 100, "B too small — likely all-skip");
    assert_gop_self_roundtrip(&gop, "seq-default");
    assert_gop_ffmpeg_interop(&gop, "seq-default");
}

#[test]
fn seq_custom_pb_gop_bit_exact() {
    let gop = encode_gop(ScalingMatrixMode::SeqCustom(custom_lists()), false);
    assert!(gop.p.annex_b.len() > 100, "P too small — likely all-skip");
    assert!(gop.b.annex_b.len() > 100, "B too small — likely all-skip");
    assert_gop_self_roundtrip(&gop, "seq-custom");
    assert_gop_ffmpeg_interop(&gop, "seq-custom");
}

#[test]
fn pic_default_transform_8x8_pb_gop_bit_exact() {
    let gop = encode_gop(ScalingMatrixMode::PicDefault, true);
    // The §8.6.4 inter 8x8 alternative must win somewhere so the
    // weighted 8x8 inter quantiser (Default_8x8_Inter) is exercised.
    assert!(
        gop.p.i8x8_mb_count + gop.b.i8x8_mb_count > 0,
        "inter RDO never picked the 8x8 transform under the scaled quantiser"
    );
    assert_gop_self_roundtrip(&gop, "pic-default-8x8");
    assert_gop_ffmpeg_interop(&gop, "pic-default-8x8");
}

#[test]
fn pic_custom_transform_8x8_pb_gop_bit_exact() {
    let gop = encode_gop(ScalingMatrixMode::PicCustom(custom_lists()), true);
    assert_gop_self_roundtrip(&gop, "pic-custom-8x8");
    assert_gop_ffmpeg_interop(&gop, "pic-custom-8x8");
}

/// Discriminator: the inter weightScale must actually shape the P/B
/// coding — a non-flat GOP's P stream differs from the flat encode of
/// the identical input.
#[test]
fn non_flat_inter_quantiser_changes_the_stream() {
    let flat = encode_gop(ScalingMatrixMode::Flat, false);
    let seq = encode_gop(ScalingMatrixMode::SeqDefault, false);
    assert_ne!(
        flat.p.annex_b, seq.p.annex_b,
        "P slice unchanged — inter weightScale not applied"
    );
    assert_ne!(
        flat.b.annex_b, seq.b.annex_b,
        "B slice unchanged — inter weightScale not applied"
    );
}
