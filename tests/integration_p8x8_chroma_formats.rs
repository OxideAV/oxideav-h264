//! P_8x8 (four P_L0_8x8 partitions): CAVLC at 4:0:0, 4:2:2 and 4:4:4,
//! CABAC at every chroma format, with and without the 8x8 transform;
//! and CABAC B pictures whose co-located picture carries P_8x8 motion.
//!
//! The second picture moves each 8x8 quadrant of every macroblock its
//! own way, so no single 16x16 vector predicts a macroblock and the
//! encoder has to split. Each stream must (a) actually carry P_8x8
//! macroblocks, (b) decode in this crate's decoder to exactly the
//! encoder's reconstruction on every plane, and (c) decode to the same
//! samples in a black-box decoder binary when one is installed.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::decoder::{Decoder, Event};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264::macroblock_layer::MbType;
use oxideav_h264::slice_data::parse_slice_data;

const W: usize = 64;
const H: usize = 48;

fn chroma_dims(fmt: u32) -> (usize, usize) {
    match fmt {
        0 => (0, 0),
        3 => (W, H),
        2 => (W / 2, H),
        _ => (W / 2, H / 2),
    }
}

/// A plane with structure at every scale, so a shifted copy is only
/// matched by the right vector.
fn texture(w: usize, h: usize, seed: usize) -> Vec<u8> {
    let mut p = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            let a = (i * 13 + j * 7 + seed * 31) % 97;
            let b = ((i / 3) * 29 + (j / 2) * 17 + (i ^ j) * 5 + seed) % 89;
            p[j * w + i] = (30 + a + b) as u8;
        }
    }
    p
}

/// `src` with each 8x8 quadrant of every macroblock displaced by its
/// own vector.
fn quadrant_motion(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let shifts: [(i32, i32); 4] = [(2, 0), (-2, 0), (0, 2), (0, -2)];
    let mut out = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            let (dx, dy) = shifts[2 * ((j % 16) / 8) + (i % 16) / 8];
            let x = (i as i32 - dx).clamp(0, w as i32 - 1) as usize;
            let y = (j as i32 - dy).clamp(0, h as i32 - 1) as usize;
            out[j * w + i] = src[y * w + x];
        }
    }
    out
}

struct Gop {
    idr: oxideav_h264::encoder::EncodedIdr,
    p: oxideav_h264::encoder::EncodedP,
    combined: Vec<u8>,
}

fn encode_gop(fmt: u32, cabac: bool, transform_8x8: bool, qp: i32) -> Gop {
    let (cw, ch) = chroma_dims(fmt);
    let y0 = texture(W, H, 1);
    let u0 = texture(cw, ch, 2);
    let v0 = texture(cw, ch, 3);
    let y1 = quadrant_motion(&y0, W, H);
    // The chroma planes keep the still picture's content: the luma
    // vectors then leave chroma residual to code.
    let (u1, v1) = (u0.clone(), v0.clone());
    let frame = |y, u, v| YuvFrame {
        width: W as u32,
        height: H as u32,
        y,
        u,
        v,
    };
    let cfg = EncoderConfig::new(W as u32, H as u32)
        .with_chroma_format_idc(fmt)
        .with_profile_idc(match fmt {
            3 => 244,
            2 => 122,
            _ => 100,
        })
        .with_cabac(cabac)
        .with_transform_8x8(transform_8x8)
        .with_qp(qp);
    let enc = Encoder::new(cfg);
    let (idr, p);
    if cabac {
        idr = enc.encode_idr_cabac(&frame(&y0, &u0, &v0));
        p = enc.encode_p_cabac(&frame(&y1, &u1, &v1), &EncodedFrameRef::from(&idr), 1, 2);
    } else {
        idr = enc.encode_idr(&frame(&y0, &u0, &v0));
        p = enc.encode_p(&frame(&y1, &u1, &v1), &EncodedFrameRef::from(&idr), 1, 2);
    }
    let mut combined = idr.annex_b.clone();
    combined.extend_from_slice(&p.annex_b);
    Gop { idr, p, combined }
}

/// Number of P_8x8 macroblocks in the stream's P slices.
fn count_p8x8(stream: &[u8]) -> usize {
    let mut dec = Decoder::new();
    let mut n = 0usize;
    for ev in dec.process_annex_b(stream) {
        if let Ok(Event::Slice {
            header,
            rbsp,
            slice_data_cursor,
            pps,
            sps,
            ..
        }) = ev
        {
            let sd = parse_slice_data(
                &rbsp,
                slice_data_cursor.0,
                slice_data_cursor.1,
                &header,
                &sps,
                &pps,
            )
            .expect("slice data parses");
            n += sd
                .macroblocks
                .iter()
                .filter(|mb| matches!(mb.mb_type, MbType::P8x8 | MbType::P8x8Ref0))
                .count();
        }
    }
    n
}

fn plane_eq(vf: &VideoFrame, plane: usize, recon: &[u8], w: usize, h: usize) -> bool {
    let p = &vf.planes[plane];
    (0..h).all(|r| p.data[r * p.stride..][..w] == recon[r * w..][..w])
}

fn assert_self_roundtrip(gop: &Gop, fmt: u32, tag: &str) {
    let (cw, ch) = chroma_dims(fmt);
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), gop.combined.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => frames.push(vf),
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    assert_eq!(frames.len(), 2, "{tag}: expected IDR + P frames");
    for (vf, (ry, ru, rv, name)) in frames.iter().zip([
        (&gop.idr.recon_y, &gop.idr.recon_u, &gop.idr.recon_v, "IDR"),
        (&gop.p.recon_y, &gop.p.recon_u, &gop.p.recon_v, "P"),
    ]) {
        assert!(plane_eq(vf, 0, ry, W, H), "{tag}: {name} luma");
        if fmt != 0 {
            assert!(plane_eq(vf, 1, ru, cw, ch), "{tag}: {name} Cb");
            assert!(plane_eq(vf, 2, rv, cw, ch), "{tag}: {name} Cr");
        }
    }
}

fn assert_black_box(gop: &Gop, fmt: u32, tag: &str) {
    black_box_frames(
        &gop.combined,
        fmt,
        tag,
        &[
            (&gop.idr.recon_y, &gop.idr.recon_u, &gop.idr.recon_v),
            (&gop.p.recon_y, &gop.p.recon_u, &gop.p.recon_v),
        ],
    );
}

/// Decode `stream` with the black-box decoder binary (skipped when it
/// is not installed) and compare its output frames with `recon`.
fn black_box_frames(stream: &[u8], fmt: u32, tag: &str, recon: &[(&Vec<u8>, &Vec<u8>, &Vec<u8>)]) {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: black-box decoder binary not present");
        return;
    }
    let (cw, ch) = chroma_dims(fmt);
    // 4:0:0 is written in the decoder's native layout (asking for a
    // grey output format would bring a range conversion in).
    let pix_fmt = match fmt {
        0 => None,
        3 => Some("yuv444p"),
        2 => Some("yuv422p"),
        _ => Some("yuv420p"),
    };
    let dir = std::env::temp_dir();
    let stem = format!("oxideav_h264_p8x8_{tag}_{}", std::process::id());
    let (h264_path, yuv_path) = (
        dir.join(format!("{stem}.h264")),
        dir.join(format!("{stem}.yuv")),
    );
    std::fs::write(&h264_path, stream).expect("write h264");
    let mut cmd = std::process::Command::new(ffmpeg);
    cmd.args(["-loglevel", "error", "-y", "-f", "h264", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo"]);
    if let Some(pf) = pix_fmt {
        cmd.args(["-pix_fmt", pf]);
    }
    let status = cmd
        .arg(&yuv_path)
        .status()
        .expect("spawn black-box decoder");
    assert!(status.success(), "black-box decoder rejected {tag}");
    let yuv = std::fs::read(&yuv_path).expect("read decoded yuv");
    let _ = std::fs::remove_file(&h264_path);
    let _ = std::fs::remove_file(&yuv_path);
    let n = recon.len();
    let frame_len = if fmt == 0 {
        yuv.len() / n
    } else {
        W * H + 2 * cw * ch
    };
    assert!(
        frame_len >= W * H && yuv.len() == n * frame_len,
        "{tag}: expected {n} frames, got {} bytes",
        yuv.len()
    );
    for (k, (f, (ry, ru, rv))) in yuv.chunks_exact(frame_len).zip(recon).enumerate() {
        assert!(f[..W * H] == ry[..], "{tag}: frame {k} luma vs black box");
        if fmt != 0 {
            assert!(f[W * H..W * H + cw * ch] == ru[..], "{tag}: frame {k} Cb");
            assert!(f[W * H + cw * ch..] == rv[..], "{tag}: frame {k} Cr");
        }
    }
}

fn run(fmt: u32, cabac: bool, transform_8x8: bool, qp: i32) {
    let tag = format!(
        "fmt{fmt}-{}-{}-qp{qp}",
        if cabac { "cabac" } else { "cavlc" },
        if transform_8x8 { "t8" } else { "t4" }
    );
    let gop = encode_gop(fmt, cabac, transform_8x8, qp);
    let n = count_p8x8(&gop.combined);
    assert!(n > 0, "{tag}: no P_8x8 macroblock was coded");
    assert_self_roundtrip(&gop, fmt, &tag);
    assert_black_box(&gop, fmt, &tag);
}

#[test]
fn cavlc_p8x8_400() {
    for qp in [18, 26, 34] {
        run(0, false, false, qp);
        run(0, false, true, qp);
    }
}

#[test]
fn cavlc_p8x8_422() {
    for qp in [18, 26, 34] {
        run(2, false, false, qp);
        run(2, false, true, qp);
    }
}

#[test]
fn cavlc_p8x8_444() {
    for qp in [18, 26, 34] {
        run(3, false, false, qp);
        run(3, false, true, qp);
    }
}

#[test]
fn cabac_p8x8_every_chroma_format() {
    for fmt in [0u32, 1, 2, 3] {
        for qp in [18, 26, 34] {
            run(fmt, true, false, qp);
            run(fmt, true, true, qp);
        }
    }
}

/// `src` with the right half of every macroblock displaced by `(d, 0)`
/// and the left half kept.
fn half_motion(src: &[u8], w: usize, h: usize, d: i32) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            let dx = if i % 16 >= 8 { d } else { 0 };
            let x = (i as i32 - dx).clamp(0, w as i32 - 1) as usize;
            out[j * w + i] = src[j * w + x];
        }
    }
    out
}

/// A B picture between an IDR and a P picture whose macroblocks are
/// split (still left half, moving right half): the co-located motion
/// differs per 8x8, so the §8.4.1.2.2 direct prediction differs per
/// 8x8 too (colZeroFlag holds for the still half only) and the B
/// picture codes B_8x8 macroblocks with direct sub-macroblocks. This
/// combination used to end in a panic in the CABAC B mode decision.
#[test]
fn cabac_b_after_p8x8_anchor() {
    for fmt in [1u32, 0, 2, 3] {
        for qp in [20, 30] {
            let (cw, ch) = chroma_dims(fmt);
            let y0 = texture(W, H, 1);
            let (u0, v0) = (texture(cw, ch, 2), texture(cw, ch, 3));
            let y2 = half_motion(&y0, W, H, 4);
            let y1 = half_motion(&y0, W, H, 2);
            let frame = |y, u, v| YuvFrame {
                width: W as u32,
                height: H as u32,
                y,
                u,
                v,
            };
            let cfg = EncoderConfig::new(W as u32, H as u32)
                .with_chroma_format_idc(fmt)
                .with_profile_idc(match fmt {
                    3 => 244,
                    2 => 122,
                    _ => 100,
                })
                .with_max_num_ref_frames(2)
                .with_cabac(true)
                .with_qp(qp);
            let enc = Encoder::new(cfg);
            let idr = enc.encode_idr_cabac(&frame(&y0, &u0, &v0));
            let p = enc.encode_p_cabac(&frame(&y2, &u0, &v0), &EncodedFrameRef::from(&idr), 1, 4);
            let b = enc.encode_b_cabac(
                &frame(&y1, &u0, &v0),
                &EncodedFrameRef::from(&idr),
                &EncodedFrameRef::from(&p),
                2,
                2,
            );
            let tag = format!("b-after-p8x8-fmt{fmt}-qp{qp}");
            let mut stream = idr.annex_b.clone();
            stream.extend_from_slice(&p.annex_b);
            stream.extend_from_slice(&b.annex_b);
            assert!(
                count_p8x8(&p_only(&idr.annex_b, &p.annex_b)) > 0,
                "{tag}: no P_8x8"
            );
            // Output order: IDR, B, P.
            let recon = [
                (&idr.recon_y, &idr.recon_u, &idr.recon_v),
                (&b.recon_y, &b.recon_u, &b.recon_v),
                (&p.recon_y, &p.recon_u, &p.recon_v),
            ];
            let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
            let pkt = Packet::new(0, TimeBase::new(1, 25), stream.clone()).with_pts(0);
            dec.send_packet(&pkt).expect("send_packet");
            dec.flush().expect("flush");
            let mut frames = Vec::new();
            loop {
                match dec.receive_frame() {
                    Ok(Frame::Video(vf)) => frames.push(vf),
                    Ok(_) => continue,
                    Err(_) => break,
                }
            }
            assert_eq!(frames.len(), 3, "{tag}: frames");
            for (k, (vf, (ry, ru, rv))) in frames.iter().zip(recon).enumerate() {
                assert!(plane_eq(vf, 0, ry, W, H), "{tag}: frame {k} luma");
                if fmt != 0 {
                    assert!(plane_eq(vf, 1, ru, cw, ch), "{tag}: frame {k} Cb");
                    assert!(plane_eq(vf, 2, rv, cw, ch), "{tag}: frame {k} Cr");
                }
            }
            black_box_frames(&stream, fmt, &tag, &recon);
        }
    }
}

fn p_only(idr: &[u8], p: &[u8]) -> Vec<u8> {
    let mut s = idr.to_vec();
    s.extend_from_slice(p);
    s
}
