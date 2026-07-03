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

/// P-frame source for the inter 8x8 tests: the IDR source plus a
/// smooth low-frequency brightness ramp over the left half. The
/// residual after motion compensation is a broad gradient — exactly the
/// content the 8x8 transform codes more compactly than sixteen 4x4s
/// (fewer DC-adjacent coefficients per area).
fn make_p_source() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (mut y, u, v) = make_source();
    for j in 0..H {
        for i in 0..W / 2 {
            let bump = 10 + (i + j) / 4;
            let s = y[j * W + i] as usize + bump;
            y[j * W + i] = s.min(235) as u8;
        }
    }
    (y, u, v)
}

#[test]
fn i8x8_p_slice_inter_8x8_transform_roundtrips_bit_exact() {
    use oxideav_h264::encoder::EncodedFrameRef;

    let (y0, u0, v0) = make_source();
    let (y1, u1, v1) = make_p_source();
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
        transform_8x8: true,
        // Keep the P MBs on the inter path so the §7.3.5 second-gate
        // transform_size_8x8_flag coding is what's exercised.
        intra_in_inter: false,
        ..EncoderConfig::new(W as u32, H as u32)
    };
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr(&f0);
    let p = enc.encode_p(&f1, &EncodedFrameRef::from(&idr), 1, 2);

    // The low-frequency inter residual must make the RDO pick the 8x8
    // transform on at least one P MB.
    assert!(
        p.i8x8_mb_count > 0,
        "inter RDO never picked the 8x8 transform on the ramp residual"
    );

    // Self-roundtrip: both frames decode bit-exactly against the
    // encoder recons.
    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p.annex_b);

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), combined.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    while let Ok(Frame::Video(vf)) = dec.receive_frame() {
        frames.push(vf);
    }
    assert_eq!(frames.len(), 2, "expected IDR + P decoded frames");
    assert_eq!(plane_max_diff(&frames[0], 0, &idr.recon_y, W, H), 0);
    assert_eq!(plane_max_diff(&frames[1], 0, &p.recon_y, W, H), 0);
    assert_eq!(plane_max_diff(&frames[1], 1, &p.recon_u, W / 2, H / 2), 0);
    assert_eq!(plane_max_diff(&frames[1], 2, &p.recon_v, W / 2, H / 2), 0);

    // Black-box reference decoder interop on the 2-frame stream.
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: reference decoder binary not present");
        return;
    }
    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!(
        "oxideav_h264_r382_p8x8_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!("oxideav_h264_r382_p8x8_{}.yuv", std::process::id()));
    std::fs::write(&h264_path, &combined).expect("write h264");
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
    assert_eq!(yuv.len(), 2 * frame_bytes, "expected two 4:2:0 frames");
    let f1_bytes = &yuv[frame_bytes..];
    assert_eq!(
        &f1_bytes[..W * H],
        &p.recon_y[..],
        "P-frame luma mismatch vs reference decoder"
    );
    assert_eq!(
        &f1_bytes[W * H..W * H + (W / 2) * (H / 2)],
        &p.recon_u[..],
        "P-frame Cb mismatch vs reference decoder"
    );
}

#[test]
fn i8x8_b_slice_gop_codes_second_gate_flag_and_roundtrips() {
    use oxideav_h264::encoder::EncodedFrameRef;

    // IDR + P + B GOP inside a transform_8x8 stream. The B writers must
    // code the mandatory §7.3.5 second-gate transform_size_8x8_flag = 0
    // on every coded B MB with cbp_luma > 0, or the decoders desync.
    let (y0, u0, v0) = make_source();
    let (y2, u2, v2) = make_p_source();
    // The B frame sits between: halfway blend, plus a small texture so
    // B MBs carry a non-zero luma residual (forcing the flag coding).
    let mut y1 = vec![0u8; W * H];
    for k in 0..W * H {
        let blend = (y0[k] as u32 + y2[k] as u32) / 2;
        let n = ((k as u32).wrapping_mul(2654435761) >> 27) & 7;
        y1[k] = (blend + n).min(235) as u8;
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
        transform_8x8: true,
        profile_idc: 100,
        max_num_ref_frames: 2,
        intra_in_inter: false,
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

    // Vacuity guard: an all-skip B slice (~15 bytes) would never code
    // cbp_luma > 0, leaving the second-gate flag path unexercised. The
    // blended+textured B source must produce genuinely coded B MBs.
    assert!(
        b.annex_b.len() > 100,
        "B slice too small ({} bytes) — likely all-skip, the \
         transform_size_8x8_flag path is not being exercised",
        b.annex_b.len()
    );
    // The B-MB 8x8-residual RDO trial must actually win somewhere on
    // this fixture, so the stream carries both flag values on B MBs
    // (transform_size_8x8_flag = 1 where the RDO picked 8x8, 0 on the
    // 4x4-coded B MBs).
    assert!(
        b.i8x8_mb_count > 0,
        "B-slice RDO never picked the 8x8 transform"
    );

    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p.annex_b);
    combined.extend_from_slice(&b.annex_b);

    // Self-roundtrip: all three frames decode bit-exactly (display
    // order: IDR, B, P).
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), combined.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    while let Ok(Frame::Video(vf)) = dec.receive_frame() {
        frames.push(vf);
    }
    assert_eq!(frames.len(), 3, "expected IDR + B + P decoded frames");
    assert_eq!(plane_max_diff(&frames[0], 0, &idr.recon_y, W, H), 0);
    assert_eq!(
        plane_max_diff(&frames[1], 0, &b.recon_y, W, H),
        0,
        "B-frame luma decode differs from encoder recon"
    );
    assert_eq!(plane_max_diff(&frames[2], 0, &p.recon_y, W, H), 0);

    // Black-box reference decoder interop over the whole GOP.
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: reference decoder binary not present");
        return;
    }
    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!(
        "oxideav_h264_r382_b8x8_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!("oxideav_h264_r382_b8x8_{}.yuv", std::process::id()));
    std::fs::write(&h264_path, &combined).expect("write h264");
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
    assert_eq!(yuv.len(), 3 * frame_bytes, "expected three 4:2:0 frames");
    // Display order: IDR, B, P.
    assert_eq!(
        &yuv[frame_bytes..frame_bytes + W * H],
        &b.recon_y[..],
        "B-frame luma mismatch vs reference decoder"
    );
    assert_eq!(
        &yuv[2 * frame_bytes..2 * frame_bytes + W * H],
        &p.recon_y[..],
        "P-frame luma mismatch vs reference decoder"
    );
}

/// Round-385 — 4:2:2 source for the High 4:2:2 (profile 122) + 8x8
/// transform IDR test: the same mixed luma as `make_source` plus
/// full-height 4:2:2 chroma gradients.
fn make_source_422() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (y, _, _) = make_source();
    let cw = W / 2;
    let ch = H; // 4:2:2 — full-height chroma.
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            u[j * cw + i] = (96 + i * 2 + j / 4) as u8;
            v[j * cw + i] = (170u32.saturating_sub(j as u32) + (i as u32) / 2).min(235) as u8;
        }
    }
    (y, u, v)
}

#[test]
fn i8x8_422_idr_selects_8x8_and_roundtrips_bit_exact() {
    let (y, u, v) = make_source_422();
    let frame = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y,
        u: &u,
        v: &v,
    };
    let cfg = EncoderConfig {
        profile_idc: 122, // High 4:2:2 — required for chroma_format_idc = 2.
        chroma_format_idc: 2,
        transform_8x8: true,
        ..EncoderConfig::new(W as u32, H as u32)
    };
    let idr = Encoder::new(cfg).encode_idr(&frame);
    assert!(
        idr.i8x8_mb_count > 0,
        "the 4:2:2 RDO never picked Intra_8x8 on the mixed fixture"
    );

    // The SPS must keep High 4:2:2 (no down-promotion to 100) and the
    // PPS must carry the 8x8 tool.
    let mut saw = (false, false);
    for nal in AnnexBSplitter::new(&idr.annex_b) {
        let unit = parse_nal_unit(nal).expect("parse NAL");
        match unit.header.nal_unit_type {
            NalUnitType::Sps => {
                let sps = oxideav_h264::sps::Sps::parse(&unit.rbsp).expect("parse SPS");
                assert_eq!(sps.profile_idc, 122);
                assert_eq!(sps.chroma_format_idc, 2);
                saw.0 = true;
            }
            NalUnitType::Pps => {
                let pps = oxideav_h264::pps::Pps::parse(&unit.rbsp).expect("parse PPS");
                assert!(pps.transform_8x8_mode_flag());
                saw.1 = true;
            }
            _ => {}
        }
    }
    assert!(saw.0 && saw.1);

    // Self-roundtrip: our decoder reproduces the encoder recon
    // bit-exactly (luma + full-height 4:2:2 chroma).
    let vf = decode_own(&idr.annex_b);
    assert_eq!(plane_max_diff(&vf, 0, &idr.recon_y, W, H), 0, "luma");
    assert_eq!(plane_max_diff(&vf, 1, &idr.recon_u, W / 2, H), 0, "Cb");
    assert_eq!(plane_max_diff(&vf, 2, &idr.recon_v, W / 2, H), 0, "Cr");
}

#[test]
fn i8x8_422_idr_reference_decoder_interop() {
    let (y, u, v) = make_source_422();
    let frame = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y,
        u: &u,
        v: &v,
    };
    let cfg = EncoderConfig {
        profile_idc: 122,
        chroma_format_idc: 2,
        transform_8x8: true,
        ..EncoderConfig::new(W as u32, H as u32)
    };
    let idr = Encoder::new(cfg).encode_idr(&frame);
    assert!(idr.i8x8_mb_count > 0);

    // Black-box validator: an independent reference decoder binary
    // (opaque I/O only).
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: reference decoder binary not present");
        return;
    }
    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!(
        "oxideav_h264_r385_i8x8_422_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_r385_i8x8_422_{}.yuv",
        std::process::id()
    ));
    std::fs::write(&h264_path, &idr.annex_b).expect("write h264");
    let status = std::process::Command::new(ffmpeg)
        .args(["-loglevel", "error", "-y", "-f", "h264", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv422p"])
        .arg(&yuv_path)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success(), "reference decoder rejected the stream");
    let yuv = std::fs::read(&yuv_path).expect("read decoded yuv");
    let _ = std::fs::remove_file(&h264_path);
    let _ = std::fs::remove_file(&yuv_path);
    let frame_bytes = W * H + 2 * (W / 2) * H;
    assert_eq!(yuv.len(), frame_bytes, "expected one 4:2:2 frame");
    assert_eq!(
        &yuv[..W * H],
        &idr.recon_y[..],
        "luma mismatch vs reference decoder"
    );
    assert_eq!(
        &yuv[W * H..W * H + (W / 2) * H],
        &idr.recon_u[..],
        "Cb mismatch vs reference decoder"
    );
    assert_eq!(
        &yuv[W * H + (W / 2) * H..],
        &idr.recon_v[..],
        "Cr mismatch vs reference decoder"
    );
}
