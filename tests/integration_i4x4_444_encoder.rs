//! Round-391 — High 4:4:4 Predictive **Intra_4x4** encoder leg.
//!
//! Exercises the new `encode_idr` CAVLC 4:4:4 I_NxN (4x4-transform)
//! path end-to-end:
//!
//!   * Per-MB the 4:4:4 dispatch runs a Lagrangian RDO between
//!     Intra_16x16 (§7.3.5.3 chroma coded like luma) and **Intra_4x4**
//!     (§8.3.4.5: per-4x4-block luma mode reuse on the Cb / Cr planes,
//!     full 16-coefficient residual blocks per plane gated by the
//!     shared §7.4.5.1 CBP quadrant bits, per-plane §9.2.1.1 nC).
//!     With `transform_8x8` the trial is three-way (+ Intra_8x8) and
//!     the I_4x4 MBs code `transform_size_8x8_flag = 0` per §7.3.5.
//!   * **Self-roundtrip**: our own decoder
//!     (`reconstruct_chroma_intra_nxn_444` + the per-plane nC walk)
//!     reproduces the encoder's local recon bit-exactly.
//!   * **Black-box interop**: an independent reference decoder binary
//!     reproduces the same recon byte-exactly (opaque I/O only).
//!
//! Spec references: §7.3.5 / §7.3.5.1 / §7.3.5.3 (I_NxN syntax at
//! ChromaArrayType == 3), §8.3.1 (Intra_4x4 prediction), §8.3.4.5
//! (chroma coded like luma), §9.2.1.1 (per-plane nC).

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::{Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264::nal::{parse_nal_unit, AnnexBSplitter, NalUnitType};

const W: usize = 64;
const H: usize = 64;

/// 4:4:4 source engineered for a **mixed** I_16x16 / I_4x4 pick:
///   * Left half: per-4x4-block alternating stripe orientation
///     (vertical stripes in even Z-blocks, horizontal in odd) — the
///     orientation flips every 4 samples, which no single 16x16 (or
///     8x8) mode can track, so the per-block I_4x4 trial wins.
///   * Right half: smooth diagonal gradient — I_16x16 wins on rate.
///
/// Chroma planes carry the same structure at lower amplitude plus a
/// gradient so all three planes produce non-zero residual (exercising
/// the per-plane nC + shared-CBP walks).
fn make_source_444() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    let mut u = vec![0u8; W * H];
    let mut v = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let idx = j * W + i;
            if i < 32 {
                // Per-4x4-sub-block oriented ramps (round-4 fixture
                // family): each sub-block position inside the parent
                // MB gets a different best §8.3.1.2 mode.
                let blk_x = (i / 4) % 4;
                let blk_y = (j / 4) % 4;
                let local_x = i % 4;
                let local_y = j % 4;
                let lum = match (blk_x + 4 * blk_y) % 4 {
                    0 => 32 + (local_x * 50) as i32,             // vertical-ish
                    1 => 32 + (local_y * 50) as i32,             // horizontal-ish
                    2 => 32 + ((local_x + local_y) * 25) as i32, // diagonal
                    _ => 32 + ((3 - local_x) * 30 + local_y * 20) as i32,
                };
                y[idx] = lum.clamp(0, 255) as u8;
                // Chroma follows the same oriented structure at lower
                // amplitude so all three planes carry residual.
                u[idx] = (64 + lum / 2).clamp(0, 255) as u8;
                v[idx] = (200 - lum / 2).clamp(0, 255) as u8;
            } else {
                y[idx] = (16 + (i + j) * 2).min(235) as u8;
                u[idx] = (64 + i * 2).min(235) as u8;
                v[idx] = (200usize.saturating_sub(j * 2)).max(16) as u8;
            }
        }
    }
    (y, u, v)
}

fn encode_444(transform_8x8: bool) -> oxideav_h264::encoder::EncodedIdr {
    let (y, u, v) = make_source_444();
    let frame = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y,
        u: &u,
        v: &v,
    };
    let cfg = EncoderConfig {
        profile_idc: 244, // High 4:4:4 Predictive.
        chroma_format_idc: 3,
        transform_8x8,
        ..EncoderConfig::new(W as u32, H as u32)
    };
    Encoder::new(cfg).encode_idr(&frame)
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

fn psnr_y(orig: &[u8], recon: &[u8]) -> f64 {
    let mut s = 0u64;
    for (&o, &r) in orig.iter().zip(recon.iter()) {
        let d = o as i32 - r as i32;
        s += (d * d) as u64;
    }
    let mse = s as f64 / orig.len() as f64;
    if mse == 0.0 {
        99.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

/// Two-way (I_16x16 / I_4x4) RDO at 4:4:4: mixed selection +
/// bit-exact self-roundtrip on all three full-resolution planes.
#[test]
fn i4x4_444_idr_mixed_rdo_roundtrips_bit_exact() {
    let idr = encode_444(false);
    let total = ((W / 16) * (H / 16)) as u32;
    assert!(
        idr.i4x4_mb_count > 0 && idr.i4x4_mb_count < total,
        "expected mixed I_16x16/Intra_4x4 selection, got {}/{total}",
        idr.i4x4_mb_count
    );
    assert_eq!(idr.i8x8_mb_count, 0, "no 8x8 transform without the flag");

    let mut saw = (false, false);
    for nal in AnnexBSplitter::new(&idr.annex_b) {
        let unit = parse_nal_unit(nal).expect("parse NAL");
        match unit.header.nal_unit_type {
            NalUnitType::Sps => {
                let sps = oxideav_h264::sps::Sps::parse(&unit.rbsp).expect("parse SPS");
                assert_eq!(sps.profile_idc, 244);
                assert_eq!(sps.chroma_format_idc, 3);
                saw.0 = true;
            }
            NalUnitType::Pps => {
                let pps = oxideav_h264::pps::Pps::parse(&unit.rbsp).expect("parse PPS");
                assert!(!pps.transform_8x8_mode_flag());
                saw.1 = true;
            }
            _ => {}
        }
    }
    assert!(saw.0 && saw.1);

    let vf = decode_own(&idr.annex_b);
    assert_eq!(plane_max_diff(&vf, 0, &idr.recon_y, W, H), 0, "luma");
    assert_eq!(plane_max_diff(&vf, 1, &idr.recon_u, W, H), 0, "Cb");
    assert_eq!(plane_max_diff(&vf, 2, &idr.recon_v, W, H), 0, "Cr");

    let (y, _, _) = make_source_444();
    let ps = psnr_y(&y, &idr.recon_y);
    eprintln!(
        "round-391 4:4:4 I_4x4 two-way RDO: stream={} bytes, i4x4={}/{total}, PSNR_Y={ps:.2}",
        idr.annex_b.len(),
        idr.i4x4_mb_count,
    );
    assert!(ps >= 38.0, "PSNR_Y {ps:.2} below floor");
}

/// Three-way (I_16x16 / I_4x4 / I_8x8) RDO under `transform_8x8`:
/// I_4x4 MBs code `transform_size_8x8_flag = 0`, I_8x8 MBs code 1 —
/// both shapes in one stream, bit-exact through our decoder.
#[test]
fn i4x4_444_three_way_rdo_with_transform_8x8_roundtrips() {
    let idr = encode_444(true);
    assert!(
        idr.i4x4_mb_count > 0,
        "expected some Intra_4x4 picks in the three-way trial",
    );

    let vf = decode_own(&idr.annex_b);
    assert_eq!(plane_max_diff(&vf, 0, &idr.recon_y, W, H), 0, "luma");
    assert_eq!(plane_max_diff(&vf, 1, &idr.recon_u, W, H), 0, "Cb");
    assert_eq!(plane_max_diff(&vf, 2, &idr.recon_v, W, H), 0, "Cr");
    eprintln!(
        "round-391 4:4:4 three-way RDO: stream={} bytes, i4x4={} i8x8={}",
        idr.annex_b.len(),
        idr.i4x4_mb_count,
        idr.i8x8_mb_count,
    );
}

/// Black-box cross-decode of both stream shapes through an independent
/// reference decoder binary (opaque I/O only) — byte-exact vs the
/// encoder's local recon.
#[test]
fn i4x4_444_reference_decoder_interop() {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: reference decoder binary not present");
        return;
    }
    for (tag, transform_8x8) in [("4x4only", false), ("mixed8x8", true)] {
        let idr = encode_444(transform_8x8);
        assert!(idr.i4x4_mb_count > 0);

        let tmpdir = std::env::temp_dir();
        let h264_path = tmpdir.join(format!(
            "oxideav_h264_r391_i4x4_444_{tag}_{}.h264",
            std::process::id()
        ));
        let yuv_path = tmpdir.join(format!(
            "oxideav_h264_r391_i4x4_444_{tag}_{}.yuv",
            std::process::id()
        ));
        std::fs::write(&h264_path, &idr.annex_b).expect("write h264");
        let status = std::process::Command::new(ffmpeg)
            .args(["-loglevel", "error", "-y", "-f", "h264", "-i"])
            .arg(&h264_path)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv444p"])
            .arg(&yuv_path)
            .status()
            .expect("spawn reference decoder");
        assert!(status.success(), "reference decoder rejected {tag}");
        let yuv = std::fs::read(&yuv_path).expect("read decoded yuv");
        let _ = std::fs::remove_file(&h264_path);
        let _ = std::fs::remove_file(&yuv_path);
        assert_eq!(yuv.len(), 3 * W * H, "expected one 4:4:4 frame ({tag})");
        assert_eq!(&yuv[..W * H], &idr.recon_y[..], "luma mismatch ({tag})");
        assert_eq!(
            &yuv[W * H..2 * W * H],
            &idr.recon_u[..],
            "Cb mismatch ({tag})"
        );
        assert_eq!(&yuv[2 * W * H..], &idr.recon_v[..], "Cr mismatch ({tag})");
    }
}
