//! Round-382 — High-profile **Intra_8x8 / 8x8-transform** encoder
//! integration tests.
//!
//! Verifies `EncoderConfig::transform_8x8` on the `encode_idr` CAVLC
//! path end-to-end:
//!
//!   * The emitted SPS is High profile (`profile_idc = 100`, §A.2.4)
//!     and the PPS carries the §7.3.2.2 optional tail with
//!     `transform_8x8_mode_flag = 1`.
//!   * The per-MB RDO trials Intra_8x8 (I_NxN with
//!     `transform_size_8x8_flag = 1`, §7.3.5) alongside I_16x16 and
//!     I_4x4: §8.3.2 Intra_8x8 prediction, the §7.4.5.3.3 four-4x4
//!     CAVLC residual split, and the §8.5.13 inverse 8x8 transform.
//!     I_4x4 MBs in the same stream code `transform_size_8x8_flag = 0`.
//!   * **Self-roundtrip**: our own decoder reproduces the encoder's
//!     local recon bit-exactly (including the §8.7 deblock pass with
//!     the 8x8-transform internal-edge skip).
//!   * **Black-box interop**: an independent reference decoder binary
//!     reproduces the same recon bit-exactly.
//!   * Fidelity: recon PSNR-Y vs the source is sane for QP 26.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::{Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264::nal::{parse_nal_unit, AnnexBSplitter, NalUnitType};

const W: usize = 64;
const H: usize = 64;

/// Mixed-content 64x64 source: smooth diagonal gradient (favours the
/// directional / DC 8x8 modes with small residual) plus a textured
/// quadrant (forces non-zero 8x8 residual across several coefficients)
/// and non-flat chroma.
fn make_source() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let base = 40 + ((i * 2 + j) as u32 % 160);
            let texture = if i >= 32 && j >= 32 {
                (i as u32 * 37 + j as u32 * 101 + (i as u32 ^ j as u32) * 13) % 61
            } else {
                0
            };
            y[j * W + i] = (base + texture).min(235) as u8;
        }
    }
    let cw = W / 2;
    let ch = H / 2;
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            u[j * cw + i] = (96 + i * 2) as u8;
            v[j * cw + i] = (160u32.saturating_sub(j as u32 * 2)) as u8;
        }
    }
    (y, u, v)
}

fn encode_i8x8() -> oxideav_h264::encoder::EncodedIdr {
    let (y, u, v) = make_source();
    let frame = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y,
        u: &u,
        v: &v,
    };
    let cfg = EncoderConfig {
        transform_8x8: true,
        ..EncoderConfig::new(W as u32, H as u32)
    };
    Encoder::new(cfg).encode_idr(&frame)
}

/// Decode an Annex B stream through this crate's decoder and return the
/// first video frame.
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

/// Compare a decoded plane against the encoder's recon buffer,
/// returning the maximum absolute per-sample difference.
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
fn i8x8_idr_emits_high_profile_sps_and_pps_transform_flag() {
    let idr = encode_i8x8();
    let mut saw_sps = false;
    let mut saw_pps = false;
    for nal in AnnexBSplitter::new(&idr.annex_b) {
        let unit = parse_nal_unit(nal).expect("parse NAL");
        match unit.header.nal_unit_type {
            NalUnitType::Sps => {
                let sps = oxideav_h264::sps::Sps::parse(&unit.rbsp).expect("parse SPS");
                // §A.2.4 — the 8x8 transform requires High profile.
                assert_eq!(sps.profile_idc, 100, "SPS must signal High profile");
                assert_eq!(sps.chroma_format_idc, 1);
                saw_sps = true;
            }
            NalUnitType::Pps => {
                let pps = oxideav_h264::pps::Pps::parse(&unit.rbsp).expect("parse PPS");
                // §7.4.2.2 — the optional tail must be present with
                // transform_8x8_mode_flag = 1.
                assert!(
                    pps.transform_8x8_mode_flag(),
                    "PPS must carry transform_8x8_mode_flag = 1"
                );
                saw_pps = true;
            }
            _ => {}
        }
    }
    assert!(saw_sps, "stream carries an SPS");
    assert!(saw_pps, "stream carries a PPS");
}

#[test]
fn i8x8_idr_self_roundtrip_bit_exact() {
    let idr = encode_i8x8();
    let vf = decode_own(&idr.annex_b);
    assert!(vf.planes.len() >= 3, "4:2:0 frame has 3 planes");

    let dy = plane_max_diff(&vf, 0, &idr.recon_y, W, H);
    let du = plane_max_diff(&vf, 1, &idr.recon_u, W / 2, H / 2);
    let dv = plane_max_diff(&vf, 2, &idr.recon_v, W / 2, H / 2);
    assert_eq!(dy, 0, "luma decode differs from encoder recon");
    assert_eq!(du, 0, "Cb decode differs from encoder recon");
    assert_eq!(dv, 0, "Cr decode differs from encoder recon");
}

#[test]
fn i8x8_idr_recon_psnr_is_sane() {
    let (y, _, _) = make_source();
    let idr = encode_i8x8();
    let mut sse = 0u64;
    for (a, b) in y.iter().zip(idr.recon_y.iter()) {
        let d = (*a as i64) - (*b as i64);
        sse += (d * d) as u64;
    }
    let mse = sse as f64 / (W * H) as f64;
    let psnr = 10.0 * (255.0f64 * 255.0 / mse).log10();
    assert!(
        psnr > 34.0,
        "PSNR-Y {psnr:.2} dB too low for QP 26 Intra_8x8"
    );
}

#[test]
fn i8x8_idr_reference_decoder_interop_bit_exact() {
    // Black-box validator: an independent reference decoder binary
    // (opaque I/O only) must reproduce the encoder's recon bit-exactly.
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: reference decoder binary not present");
        return;
    }
    let idr = encode_i8x8();

    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!(
        "oxideav_h264_r382_i8x8_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!("oxideav_h264_r382_i8x8_{}.yuv", std::process::id()));
    std::fs::write(&h264_path, &idr.annex_b).expect("write h264");

    let status = std::process::Command::new(ffmpeg)
        .args(["-loglevel", "error", "-y", "-f", "h264", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&yuv_path)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success(), "reference decoder rejected the stream");

    let yuv = std::fs::read(&yuv_path).expect("read decoded yuv");
    let _ = std::fs::remove_file(&h264_path);
    let _ = std::fs::remove_file(&yuv_path);
    let frame_bytes = W * H + 2 * (W / 2) * (H / 2);
    assert_eq!(yuv.len(), frame_bytes, "expected exactly one 4:2:0 frame");

    let (ry, rest) = yuv.split_at(W * H);
    let (ru, rv) = rest.split_at((W / 2) * (H / 2));
    assert_eq!(ry, &idr.recon_y[..], "luma mismatch vs reference decoder");
    assert_eq!(ru, &idr.recon_u[..], "Cb mismatch vs reference decoder");
    assert_eq!(rv, &idr.recon_v[..], "Cr mismatch vs reference decoder");
}

#[test]
fn i8x8_adaptive_rdo_selects_8x8_on_textured_content_and_roundtrips() {
    // The per-MB RDO (I_16x16 vs I_4x4 vs I_8x8) must actually select
    // the 8x8 transform somewhere on the mixed fixture (the textured
    // quadrant favours 8x8: finer than 16x16 prediction, fewer mode
    // bits than 4x4), and must NOT select it everywhere (the smooth
    // gradient regions favour I_16x16 Plane/DC).
    let idr = encode_i8x8();
    let total_mbs = (W / 16) * (H / 16);
    assert!(
        idr.i8x8_mb_count > 0,
        "RDO never picked Intra_8x8 on the textured quadrant"
    );
    assert!(
        (idr.i8x8_mb_count as usize) < total_mbs,
        "RDO picked Intra_8x8 for every MB — the 3-way trial is degenerate"
    );

    // The mixed-mb_type stream must still decode bit-exactly through
    // our own decoder (exercises the §7.3.5 transform_size_8x8_flag=0
    // emission on I_NxN 4x4 MBs inside a transform_8x8 stream).
    let vf = decode_own(&idr.annex_b);
    let dy = plane_max_diff(&vf, 0, &idr.recon_y, W, H);
    assert_eq!(dy, 0, "adaptive-stream luma decode differs from recon");
}

#[test]
fn i8x8_disabled_encoder_reports_zero_8x8_mbs() {
    let (y, u, v) = make_source();
    let frame = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y,
        u: &u,
        v: &v,
    };
    let idr = Encoder::new(EncoderConfig::new(W as u32, H as u32)).encode_idr(&frame);
    assert_eq!(idr.i8x8_mb_count, 0);
}
