//! Round-385 — High-profile **8x8 transform under CABAC** integration
//! tests (IDR / Intra_8x8 path).
//!
//! Verifies `EncoderConfig::transform_8x8` on the `encode_idr_cabac`
//! entry point end-to-end:
//!
//!   * SPS auto-promotes to High profile (`profile_idc = 100`, §A.2.4)
//!     and the PPS codes the §7.3.2.2 optional tail with
//!     `transform_8x8_mode_flag = 1` alongside
//!     `entropy_coding_mode_flag = 1`.
//!   * The per-MB trial selects Intra_8x8 (I_NxN with the §9.3.3.1.1.10
//!     `transform_size_8x8_flag = 1` coded at ctxIdxOffset 399) on the
//!     smooth-content MBs: §8.3.2 prediction, the §7.3.5.3.3 blockCat-5
//!     64-coefficient CABAC residual (coded_block_flag suppressed at
//!     4:2:0, Table 9-43 significance maps), the §8.5.13 inverse.
//!   * **Self-roundtrip**: our own decoder reproduces the encoder's
//!     local recon bit-exactly (including the §8.7 deblock pass with
//!     the 8x8-transform internal-edge skip).
//!   * **Black-box interop**: an independent reference decoder binary
//!     reproduces the same recon bit-exactly.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::{Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264::nal::{parse_nal_unit, AnnexBSplitter, NalUnitType};

const W: usize = 64;
const H: usize = 64;

/// Mixed-content 64x64 source: smooth diagonal gradient (favours the
/// directional / DC 8x8 modes with a compact 8x8 residual) plus a
/// textured quadrant and non-flat chroma.
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

fn encode_i8x8_cabac() -> oxideav_h264::encoder::EncodedIdr {
    let (y, u, v) = make_source();
    let frame = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y,
        u: &u,
        v: &v,
    };
    let cfg = EncoderConfig {
        cabac: true,
        profile_idc: 100,
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
fn i8x8_cabac_idr_emits_high_profile_and_both_pps_flags() {
    let idr = encode_i8x8_cabac();
    let mut saw_sps = false;
    let mut saw_pps = false;
    for nal in AnnexBSplitter::new(&idr.annex_b) {
        let unit = parse_nal_unit(nal).expect("parse NAL");
        match unit.header.nal_unit_type {
            NalUnitType::Sps => {
                let sps = oxideav_h264::sps::Sps::parse(&unit.rbsp).expect("parse SPS");
                assert_eq!(sps.profile_idc, 100, "SPS must signal High profile");
                assert_eq!(sps.chroma_format_idc, 1);
                saw_sps = true;
            }
            NalUnitType::Pps => {
                let pps = oxideav_h264::pps::Pps::parse(&unit.rbsp).expect("parse PPS");
                assert!(
                    pps.entropy_coding_mode_flag,
                    "PPS must signal CABAC entropy coding"
                );
                assert!(
                    pps.transform_8x8_mode_flag(),
                    "PPS must carry transform_8x8_mode_flag = 1"
                );
                saw_pps = true;
            }
            _ => {}
        }
    }
    assert!(saw_sps && saw_pps);
}

#[test]
fn i8x8_cabac_idr_selects_8x8_and_roundtrips_bit_exact() {
    let idr = encode_i8x8_cabac();
    assert!(
        idr.i8x8_mb_count > 0,
        "the trial never picked Intra_8x8 on the mixed fixture"
    );
    assert!(
        idr.i8x8_mb_count < ((W / 16) * (H / 16)) as u32,
        "expected a mixed I_16x16 / I_8x8 picture, got all-8x8"
    );

    let vf = decode_own(&idr.annex_b);
    assert_eq!(plane_max_diff(&vf, 0, &idr.recon_y, W, H), 0, "luma");
    assert_eq!(plane_max_diff(&vf, 1, &idr.recon_u, W / 2, H / 2), 0, "Cb");
    assert_eq!(plane_max_diff(&vf, 2, &idr.recon_v, W / 2, H / 2), 0, "Cr");

    // Fidelity floor at QP 26.
    let (y, _, _) = make_source();
    let mut sse = 0u64;
    for (src, rec) in y.iter().zip(idr.recon_y.iter()) {
        let e = (*src as i64) - (*rec as i64);
        sse += (e * e) as u64;
    }
    let mse = sse as f64 / (W * H) as f64;
    let psnr = 10.0 * (255.0f64 * 255.0 / mse).log10();
    assert!(psnr > 32.0, "PSNR-Y {psnr:.2} dB below the QP-26 floor");
}

#[test]
fn i8x8_cabac_idr_without_flag_never_codes_8x8() {
    let (y, u, v) = make_source();
    let frame = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y,
        u: &u,
        v: &v,
    };
    let cfg = EncoderConfig {
        cabac: true,
        profile_idc: 100,
        ..EncoderConfig::new(W as u32, H as u32)
    };
    let idr = Encoder::new(cfg).encode_idr_cabac(&frame);
    assert_eq!(idr.i8x8_mb_count, 0);
}

/// P-frame source: the IDR source plus a smooth low-frequency
/// brightness ramp over the left half — a broad-gradient residual the
/// 8x8 transform codes more compactly than sixteen 4x4s.
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
fn i8x8_cabac_p_slice_second_gate_flag_roundtrips_bit_exact() {
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
        cabac: true,
        profile_idc: 100,
        transform_8x8: true,
        max_num_ref_frames: 1,
        ..EncoderConfig::new(W as u32, H as u32)
    };
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr_cabac(&f0);
    let p = enc.encode_p_cabac(&f1, &EncodedFrameRef::from(&idr), 1, 2);

    // The low-frequency inter residual must make the trial pick the
    // 8x8 transform on at least one P MB (the §7.3.5 second-gate flag
    // gets coded with both values in one stream).
    assert!(
        p.i8x8_mb_count > 0,
        "inter trial never picked the 8x8 transform on the ramp residual"
    );

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
        "oxideav_h264_r385_p8x8_cabac_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_r385_p8x8_cabac_{}.yuv",
        std::process::id()
    ));
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
    assert_eq!(
        &f1_bytes[W * H + (W / 2) * (H / 2)..],
        &p.recon_v[..],
        "P-frame Cr mismatch vs reference decoder"
    );
}

#[test]
fn i8x8_cabac_b_slice_gop_second_gate_flag_roundtrips_bit_exact() {
    use oxideav_h264::encoder::EncodedFrameRef;

    // IDR + P + B CABAC GOP inside a transform_8x8 stream. Every coded
    // B MB with cbp_luma > 0 must code the §7.3.5 second-gate flag or
    // the decoders desync; the halfway blend + low-frequency ramp makes
    // the 8x8 trial win on some B MBs too.
    let (y0, u0, v0) = make_source();
    let (y2, u2, v2) = make_p_source();
    let mut y1 = vec![0u8; W * H];
    for (k, out) in y1.iter_mut().enumerate() {
        let blend = (y0[k] as u32 + y2[k] as u32) / 2;
        let ramp = ((k % W) / 4) as u32;
        *out = (blend + ramp).min(235) as u8;
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
        cabac: true,
        profile_idc: 100,
        transform_8x8: true,
        max_num_ref_frames: 2,
        ..EncoderConfig::new(W as u32, H as u32)
    };
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr_cabac(&f0);
    let p = enc.encode_p_cabac(&f2, &EncodedFrameRef::from(&idr), 1, 4);
    let b = enc.encode_b_cabac(
        &f1,
        &EncodedFrameRef::from(&idr),
        &EncodedFrameRef::from(&p),
        2,
        2,
    );

    // Vacuity guards: the B slice genuinely codes MBs and the 8x8
    // trial genuinely fires on the B residual.
    assert!(
        b.i8x8_mb_count > 0,
        "B-slice trial never picked the 8x8 transform on the blend+ramp residual"
    );

    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p.annex_b);
    combined.extend_from_slice(&b.annex_b);

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), combined.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    while let Ok(Frame::Video(vf)) = dec.receive_frame() {
        frames.push(vf);
    }
    assert_eq!(frames.len(), 3, "expected IDR + B + P in output order");
    // Output order: IDR (poc 0), B (poc 2), P (poc 4).
    assert_eq!(plane_max_diff(&frames[0], 0, &idr.recon_y, W, H), 0);
    assert_eq!(plane_max_diff(&frames[1], 0, &b.recon_y, W, H), 0, "B Y");
    assert_eq!(
        plane_max_diff(&frames[1], 1, &b.recon_u, W / 2, H / 2),
        0,
        "B Cb"
    );
    assert_eq!(
        plane_max_diff(&frames[1], 2, &b.recon_v, W / 2, H / 2),
        0,
        "B Cr"
    );
    assert_eq!(plane_max_diff(&frames[2], 0, &p.recon_y, W, H), 0, "P Y");

    // Black-box reference decoder interop on the 3-frame stream.
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: reference decoder binary not present");
        return;
    }
    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!(
        "oxideav_h264_r385_b8x8_cabac_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_r385_b8x8_cabac_{}.yuv",
        std::process::id()
    ));
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
    let fb = &yuv[frame_bytes..2 * frame_bytes];
    assert_eq!(
        &fb[..W * H],
        &b.recon_y[..],
        "B-frame luma mismatch vs reference decoder"
    );
    assert_eq!(
        &fb[W * H..W * H + (W / 2) * (H / 2)],
        &b.recon_u[..],
        "B-frame Cb mismatch vs reference decoder"
    );
    assert_eq!(
        &fb[W * H + (W / 2) * (H / 2)..],
        &b.recon_v[..],
        "B-frame Cr mismatch vs reference decoder"
    );
}

#[test]
fn i8x8_cabac_idr_reference_decoder_interop() {
    let idr = encode_i8x8_cabac();
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
        "oxideav_h264_r385_i8x8_cabac_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_r385_i8x8_cabac_{}.yuv",
        std::process::id()
    ));
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
    assert_eq!(yuv.len(), frame_bytes, "expected one 4:2:0 frame");
    let ry = &yuv[..W * H];
    let ru = &yuv[W * H..W * H + (W / 2) * (H / 2)];
    let rv = &yuv[W * H + (W / 2) * (H / 2)..];
    assert_eq!(ry, &idr.recon_y[..], "luma mismatch vs reference decoder");
    assert_eq!(ru, &idr.recon_u[..], "Cb mismatch vs reference decoder");
    assert_eq!(rv, &idr.recon_v[..], "Cr mismatch vs reference decoder");
}
