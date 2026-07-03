//! Round-388 — **chroma-AC CAVLC nC** regression tests.
//!
//! The encoder historically hard-coded `nC = 0` for every chroma AC
//! coeff_token on the assumption it was "only rate-suboptimal". That
//! is wrong: §9.2.1.1 selects the coeff_token VLC *table* from nC, so
//! once any neighbouring chroma AC block carries TotalCoeff ≥ 2 (dense
//! chroma texture) a spec decoder picks a different table than the
//! encoder wrote with and the slice desynchronises — our own decoder
//! failed with `invalid total_coeff` on the first MB of this content.
//!
//! The encoder now derives per-block nC from the `CavlcNcGrid`
//! neighbours (with the in-MB progressive TotalCoeff updates the
//! decoder also sees) and commits its per-block chroma totals for
//! subsequent MBs, across the I / P / B CAVLC writers.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::{Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 64;
const H: usize = 64;

/// Dense-chroma source: hash-textured Cb and striped Cr so most MBs
/// reach `cbp_chroma == 2` with per-block TotalCoeff well above 2,
/// exercising the coeff_token table-1/2 selections that the old
/// hard-coded `nC = 0` could never produce.
fn make_dense_chroma_source(shift: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let base = 40 + ((i * 2 + j + shift) as u32 % 160);
            y[j * W + i] = base.min(235) as u8;
        }
    }
    let cw = W / 2;
    let ch = H / 2;
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            let tex = ((i + shift) as u32 * 19 + j as u32 * 31) % 23;
            u[j * cw + i] = (80 + (i + shift) as u32 * 2 + tex * 3).min(230) as u8;
            v[j * cw + i] =
                (190u32.saturating_sub(j as u32 * 3 + ((i + shift) as u32 % 5) * 11)).max(20) as u8;
        }
    }
    (y, u, v)
}

fn decode_frames(stream: &[u8], n: usize) -> Vec<VideoFrame> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream.to_vec()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut out = Vec::new();
    while out.len() < n {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => out.push(vf),
            Ok(_) => continue,
            Err(e) => panic!("decoded {} of {n} frames: {e:?}", out.len()),
        }
    }
    out
}

fn plane_max_diff(vf: &VideoFrame, plane: usize, recon: &[u8], w: usize, h: usize) -> u32 {
    let p = &vf.planes[plane];
    let mut max = 0u32;
    for r in 0..h {
        for c in 0..w {
            max = max.max((p.data[r * p.stride + c] as i32).abs_diff(recon[r * w + c] as i32));
        }
    }
    max
}

#[test]
fn dense_chroma_idr_self_roundtrip_bit_exact() {
    let (y, u, v) = make_dense_chroma_source(0);
    let frame = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y,
        u: &u,
        v: &v,
    };
    let idr = Encoder::new(EncoderConfig::new(W as u32, H as u32)).encode_idr(&frame);
    let vf = &decode_frames(&idr.annex_b, 1)[0];
    assert_eq!(plane_max_diff(vf, 0, &idr.recon_y, W, H), 0, "luma");
    assert_eq!(plane_max_diff(vf, 1, &idr.recon_u, W / 2, H / 2), 0, "Cb");
    assert_eq!(plane_max_diff(vf, 2, &idr.recon_v, W / 2, H / 2), 0, "Cr");
}

#[test]
fn dense_chroma_gop_self_roundtrip_bit_exact() {
    // IDR + P + B — the P/B chroma writers share the same nC fix.
    let (y0, u0, v0) = make_dense_chroma_source(0);
    let (y1, u1, v1) = make_dense_chroma_source(4);
    let (y2, u2, v2) = make_dense_chroma_source(2);
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
    let mut cfg = EncoderConfig::new(W as u32, H as u32);
    cfg.profile_idc = 77;
    cfg.max_num_ref_frames = 2;
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr(&f0);
    let idr_ref = oxideav_h264::encoder::EncodedFrameRef::from(&idr);
    let p = enc.encode_p(&f1, &idr_ref, 1, 2);
    let p_ref = oxideav_h264::encoder::EncodedFrameRef::from(&p);
    let b = enc.encode_b(&f2, &idr_ref, &p_ref, 2, 1);

    let mut stream = idr.annex_b.clone();
    stream.extend_from_slice(&p.annex_b);
    stream.extend_from_slice(&b.annex_b);
    let frames = decode_frames(&stream, 3);
    // The decoder outputs in POC display order: IDR (poc 0), B (poc 1),
    // P (poc 2) — compare each against its encoder recon.
    let recons: [(&[u8], &[u8], &[u8]); 3] = [
        (&idr.recon_y, &idr.recon_u, &idr.recon_v),
        (&b.recon_y, &b.recon_u, &b.recon_v),
        (&p.recon_y, &p.recon_u, &p.recon_v),
    ];
    for (k, (ry, ru, rv)) in recons.iter().enumerate() {
        assert_eq!(plane_max_diff(&frames[k], 0, ry, W, H), 0, "frame {k} Y");
        assert_eq!(
            plane_max_diff(&frames[k], 1, ru, W / 2, H / 2),
            0,
            "frame {k} Cb"
        );
        assert_eq!(
            plane_max_diff(&frames[k], 2, rv, W / 2, H / 2),
            0,
            "frame {k} Cr"
        );
    }
}

#[test]
fn dense_chroma_idr_reference_decoder_interop_bit_exact() {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: reference decoder binary not present");
        return;
    }
    let (y, u, v) = make_dense_chroma_source(0);
    let frame = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y,
        u: &u,
        v: &v,
    };
    let idr = Encoder::new(EncoderConfig::new(W as u32, H as u32)).encode_idr(&frame);
    let dir = std::env::temp_dir().join(format!("oxideav-h264-cnc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let h264 = dir.join("dense-chroma.h264");
    let yuv = dir.join("dense-chroma.yuv");
    std::fs::write(&h264, &idr.annex_b).expect("write stream");
    let status = std::process::Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-y"])
        .arg(&yuv)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success(), "reference decoder rejected the stream");
    let data = std::fs::read(&yuv).expect("read yuv");
    let (cw, ch) = (W / 2, H / 2);
    assert_eq!(&data[..W * H], &idr.recon_y[..], "luma mismatch");
    assert_eq!(
        &data[W * H..W * H + cw * ch],
        &idr.recon_u[..],
        "Cb mismatch"
    );
    assert_eq!(&data[W * H + cw * ch..], &idr.recon_v[..], "Cr mismatch");
}
