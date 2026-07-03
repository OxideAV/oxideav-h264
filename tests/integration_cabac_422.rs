//! Round-388 — **CABAC 4:2:2 IDR encode** integration tests.
//!
//! Exercises `encode_idr_cabac` at `chroma_format_idc = 2`:
//!  * High 4:2:2 SPS (profile_idc = 122) + `entropy_coding_mode_flag=1`.
//!  * §7.3.5.3.3 ChromaDCLevel with 8 coefficients (2x4 chroma-DC
//!    Hadamard, NumC8x8 = 2 → the §9.3.3.1.3 eq. (9-22) contexts) and
//!    8 ChromaACLevel blocks per plane (2x4 block grid with the
//!    §9.3.3.1.1.9 cat-4 wrap-by-16 above-neighbour derivation).
//!  * Optional 8x8 luma transform (`transform_8x8`) — the blockCat-5
//!    trial runs unchanged next to the 4:2:2 chroma path.
//!  * Bit-exact self-roundtrip AND bit-exact black-box reference
//!    decoder interop on both wire shapes.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::{Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264::nal::{parse_nal_unit, AnnexBSplitter, NalUnitType};

const W: usize = 64;
const H: usize = 64;

/// Textured 4:2:2 source (chroma plane 32x64): gradient + hash pattern
/// so that chroma DC *and* AC blocks go non-zero (cbp_chroma == 2 on
/// most MBs) and several MBs keep cbp_chroma < 2 for the CBF gating.
fn make_source_422() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let base = 40 + ((i * 2 + j) as u32 % 160);
            let texture = if i >= 32 {
                (i as u32 * 37 + j as u32 * 101 + (i as u32 ^ j as u32) * 13) % 61
            } else {
                0
            };
            y[j * W + i] = (base + texture).min(235) as u8;
        }
    }
    let cw = W / 2;
    let mut u = vec![0u8; cw * H];
    let mut v = vec![0u8; cw * H];
    for j in 0..H {
        for i in 0..cw {
            let tex_u = if j >= 32 {
                (i as u32 * 29 + j as u32 * 53) % 37
            } else {
                0
            };
            u[j * cw + i] = (60 + i as u32 * 2 + tex_u).min(230) as u8;
            v[j * cw + i] =
                (200u32.saturating_sub(j as u32 * 2 + (i as u32 % 7) * 3)).max(20) as u8;
        }
    }
    (y, u, v)
}

fn encode_cabac_422(transform_8x8: bool) -> oxideav_h264::encoder::EncodedIdr {
    let (y, u, v) = make_source_422();
    let frame = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y,
        u: &u,
        v: &v,
    };
    let cfg = EncoderConfig {
        cabac: true,
        profile_idc: 122,
        chroma_format_idc: 2,
        transform_8x8,
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

#[test]
fn cabac_422_sps_pps_shape() {
    let idr = encode_cabac_422(false);
    let mut saw_sps = false;
    let mut saw_pps = false;
    for nal in AnnexBSplitter::new(&idr.annex_b) {
        let unit = parse_nal_unit(nal).expect("parse NAL");
        match unit.header.nal_unit_type {
            NalUnitType::Sps => {
                let sps = oxideav_h264::sps::Sps::parse(&unit.rbsp).expect("parse SPS");
                assert_eq!(sps.profile_idc, 122, "SPS must signal High 4:2:2");
                assert_eq!(sps.chroma_format_idc, 2);
                saw_sps = true;
            }
            NalUnitType::Pps => {
                let pps = oxideav_h264::pps::Pps::parse(&unit.rbsp).expect("parse PPS");
                assert!(pps.entropy_coding_mode_flag, "PPS must signal CABAC");
                saw_pps = true;
            }
            _ => {}
        }
    }
    assert!(saw_sps && saw_pps);
}

#[test]
fn cabac_422_self_roundtrip_bit_exact() {
    let idr = encode_cabac_422(false);
    assert_eq!(idr.recon_u.len(), (W / 2) * H, "Cb plane must be 32x64");
    let vf = decode_own(&idr.annex_b);
    assert_eq!(vf.planes[1].data.len(), (W / 2) * H);
    assert_eq!(plane_max_diff(&vf, 0, &idr.recon_y, W, H), 0, "luma");
    assert_eq!(plane_max_diff(&vf, 1, &idr.recon_u, W / 2, H), 0, "Cb");
    assert_eq!(plane_max_diff(&vf, 2, &idr.recon_v, W / 2, H), 0, "Cr");
}

#[test]
fn cabac_422_transform_8x8_self_roundtrip_bit_exact() {
    let idr = encode_cabac_422(true);
    assert!(
        idr.i8x8_mb_count > 0,
        "the smooth-gradient MBs must pick Intra_8x8 under the trial"
    );
    let vf = decode_own(&idr.annex_b);
    assert_eq!(plane_max_diff(&vf, 0, &idr.recon_y, W, H), 0, "luma");
    assert_eq!(plane_max_diff(&vf, 1, &idr.recon_u, W / 2, H), 0, "Cb");
    assert_eq!(plane_max_diff(&vf, 2, &idr.recon_v, W / 2, H), 0, "Cr");
}

fn ffmpeg_interop(idr: &oxideav_h264::encoder::EncodedIdr, tag: &str) {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: reference decoder binary not present");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h264-c422-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let h264 = dir.join(format!("{tag}.h264"));
    let yuv = dir.join(format!("{tag}.yuv"));
    std::fs::write(&h264, &idr.annex_b).expect("write stream");
    let status = std::process::Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv422p", "-y"])
        .arg(&yuv)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success(), "reference decoder rejected the stream");
    let data = std::fs::read(&yuv).expect("read yuv");
    let (cw, ch) = (W / 2, H);
    assert_eq!(data.len(), W * H + 2 * cw * ch);
    assert_eq!(&data[..W * H], &idr.recon_y[..], "{tag}: luma mismatch");
    assert_eq!(
        &data[W * H..W * H + cw * ch],
        &idr.recon_u[..],
        "{tag}: Cb mismatch"
    );
    assert_eq!(
        &data[W * H + cw * ch..],
        &idr.recon_v[..],
        "{tag}: Cr mismatch"
    );
}

#[test]
fn cabac_422_reference_decoder_interop_bit_exact() {
    let idr = encode_cabac_422(false);
    ffmpeg_interop(&idr, "cabac-422");
}

#[test]
fn cabac_422_transform_8x8_reference_decoder_interop_bit_exact() {
    let idr = encode_cabac_422(true);
    ffmpeg_interop(&idr, "cabac-422-8x8");
}
