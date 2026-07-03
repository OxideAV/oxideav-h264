//! Round-388 — **CABAC 4:4:4 Intra_8x8 (blockCat-9/13)** integration
//! tests.
//!
//! Exercises `encode_idr_cabac` at `chroma_format_idc = 3` with
//! `transform_8x8`:
//!  * Every MB codes I_NxN + `transform_size_8x8_flag = 1`; chroma is
//!    coded like luma per §8.3.4.5 (per-quadrant luma-mode reuse, 8x8
//!    transform at the chroma QP, shared CBP across the three planes).
//!  * §7.3.5.3.3 at ChromaArrayType == 3: the 64-coefficient blocks
//!    code an explicit `coded_block_flag` — blockCat 5 (luma), 9 (Cb)
//!    and 13 (Cr) with the §9.3.3.1.1.9 neighbouring-8x8-block cond
//!    terms.
//!  * Bit-exact self-roundtrip AND bit-exact black-box reference
//!    decoder interop.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::{Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264::nal::{parse_nal_unit, AnnexBSplitter, NalUnitType};

const W: usize = 64;
const H: usize = 64;

/// Textured 4:4:4 source: oriented luma texture + independently
/// textured full-resolution chroma so that all three planes produce
/// non-zero 8x8 residuals (walking the cat-9/13 significance maps),
/// while a smooth quadrant keeps some per-plane CBFs at 0.
fn make_source_444() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    let mut u = vec![0u8; W * H];
    let mut v = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let base = 40 + ((i * 2 + j) as u32 % 160);
            let tex = if i >= 32 || j >= 32 {
                (i as u32 * 37 + j as u32 * 101 + (i as u32 ^ j as u32) * 13) % 61
            } else {
                0
            };
            y[j * W + i] = (base + tex).min(235) as u8;
            let tex_u = if j >= 16 {
                (i as u32 * 29 + j as u32 * 53) % 41
            } else {
                0
            };
            u[j * W + i] = (60 + (i as u32) + tex_u).min(230) as u8;
            v[j * W + i] = (210u32.saturating_sub(j as u32 * 2 + (i as u32 % 9) * 5)).max(20) as u8;
        }
    }
    (y, u, v)
}

fn encode_cabac_444_8x8() -> oxideav_h264::encoder::EncodedIdr {
    let (y, u, v) = make_source_444();
    let frame = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y,
        u: &u,
        v: &v,
    };
    let cfg = EncoderConfig {
        cabac: true,
        profile_idc: 244,
        chroma_format_idc: 3,
        transform_8x8: true,
        ..EncoderConfig::new(W as u32, H as u32)
    };
    Encoder::new(cfg).encode_idr_cabac(&frame)
}

fn decode_own(stream: &[u8]) -> VideoFrame {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream.to_vec()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => return vf,
            Ok(_) => continue,
            Err(e) => panic!("no video frame decoded: {e:?}"),
        }
    }
}

fn plane_max_diff(vf: &VideoFrame, plane: usize, recon: &[u8]) -> u32 {
    let p = &vf.planes[plane];
    let mut max = 0u32;
    for r in 0..H {
        for c in 0..W {
            let a = p.data[r * p.stride + c] as i32;
            let b = recon[r * W + c] as i32;
            max = max.max(a.abs_diff(b));
        }
    }
    max
}

#[test]
fn cabac_444_8x8_sps_shape_and_mixed_selection() {
    let idr = encode_cabac_444_8x8();
    // Round-388 — the per-MB I_16x16-vs-Intra_8x8 RDO runs at 4:4:4:
    // the textured quadrants pick Intra_8x8, the smooth ones I_16x16,
    // so the stream carries BOTH shapes (a stronger gate than the
    // former all-I_8x8 pin — every mixed-neighbour context path runs).
    let total = ((W / 16) * (H / 16)) as u32;
    assert!(
        idr.i8x8_mb_count > 0 && idr.i8x8_mb_count < total,
        "expected mixed I_16x16/Intra_8x8 selection, got {}/{total}",
        idr.i8x8_mb_count
    );
    let mut saw_sps = false;
    for nal in AnnexBSplitter::new(&idr.annex_b) {
        let unit = parse_nal_unit(nal).expect("parse NAL");
        if unit.header.nal_unit_type == NalUnitType::Sps {
            let sps = oxideav_h264::sps::Sps::parse(&unit.rbsp).expect("parse SPS");
            assert_eq!(sps.profile_idc, 244, "High 4:4:4 Predictive");
            assert_eq!(sps.chroma_format_idc, 3);
            saw_sps = true;
        }
    }
    assert!(saw_sps);
}

#[test]
fn cabac_444_8x8_self_roundtrip_bit_exact() {
    let idr = encode_cabac_444_8x8();
    let vf = decode_own(&idr.annex_b);
    assert_eq!(vf.planes[1].data.len(), W * H, "full-resolution chroma");
    assert_eq!(plane_max_diff(&vf, 0, &idr.recon_y), 0, "luma");
    assert_eq!(plane_max_diff(&vf, 1, &idr.recon_u), 0, "Cb");
    assert_eq!(plane_max_diff(&vf, 2, &idr.recon_v), 0, "Cr");
}

#[test]
fn cabac_444_8x8_reference_decoder_interop_bit_exact() {
    let idr = encode_cabac_444_8x8();
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: reference decoder binary not present");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h264-c444-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let h264 = dir.join("cabac-444-8x8.h264");
    let yuv = dir.join("cabac-444-8x8.yuv");
    std::fs::write(&h264, &idr.annex_b).expect("write stream");
    let status = std::process::Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv444p", "-y"])
        .arg(&yuv)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success(), "reference decoder rejected the stream");
    let data = std::fs::read(&yuv).expect("read yuv");
    assert_eq!(data.len(), 3 * W * H);
    assert_eq!(&data[..W * H], &idr.recon_y[..], "luma mismatch");
    assert_eq!(&data[W * H..2 * W * H], &idr.recon_u[..], "Cb mismatch");
    assert_eq!(&data[2 * W * H..], &idr.recon_v[..], "Cr mismatch");
}
