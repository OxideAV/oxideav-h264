//! Intra_16x16 macroblocks inside CAVLC P and B pictures at 4:0:0,
//! 4:2:2 and 4:4:4 (the fallback the mode decision takes when no
//! reference predicts a macroblock).
//!
//! The references are dense texture; the P and the B picture each
//! replace a block of macroblocks with a smooth ramp that exists in
//! neither reference, which intra prediction codes far cheaper than
//! any motion vector. Each stream must carry intra macroblocks in its
//! P and its B slice, decode in this crate's decoder to exactly the
//! encoder's reconstruction on every plane, and decode to the same
//! samples in a black-box decoder binary when one is installed.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::decoder::{Decoder, Event};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264::macroblock_layer::MbType;
use oxideav_h264::slice_data::parse_slice_data;
use oxideav_h264::slice_header::SliceType;

const W: usize = 64;
const H: usize = 64;

fn chroma_dims(fmt: u32) -> (usize, usize) {
    match fmt {
        0 => (0, 0),
        3 => (W, H),
        2 => (W / 2, H),
        _ => (W / 2, H / 2),
    }
}

fn texture(w: usize, h: usize, seed: usize) -> Vec<u8> {
    let mut p = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            let a = (i * 37 + j * 101 + (i ^ j) * 13 + seed * 7) % 113;
            let b = ((i * i + j * 3 + seed) % 61) * 2;
            p[j * w + i] = (20 + a + b).min(250) as u8;
        }
    }
    p
}

/// Overwrite the rectangle `[x0, x0 + rw) x [y0, y0 + rh)` (given as
/// fractions of the plane in 1/4 units) with a smooth ramp.
fn ramp(p: &mut [u8], w: usize, h: usize, qx: usize, qy: usize, level: usize) {
    if w == 0 {
        return;
    }
    let (x0, y0, rw, rh) = (qx * w / 4, qy * h / 4, w / 2, h / 2);
    for j in 0..rh {
        for i in 0..rw {
            p[(y0 + j) * w + x0 + i] = (level + (i * 40) / rw + (j * 24) / rh) as u8;
        }
    }
}

struct Pic {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Pic {
    fn frame(&self) -> YuvFrame<'_> {
        YuvFrame {
            width: W as u32,
            height: H as u32,
            y: &self.y,
            u: &self.u,
            v: &self.v,
        }
    }
}

struct Gop {
    /// Reconstructions in output order: IDR, B, P.
    recon: [(Vec<u8>, Vec<u8>, Vec<u8>); 3],
    stream: Vec<u8>,
}

fn encode_gop(fmt: u32, transform_8x8: bool, qp: i32) -> Gop {
    let (cw, ch) = chroma_dims(fmt);
    let base = Pic {
        y: texture(W, H, 1),
        u: texture(cw, ch, 2),
        v: texture(cw, ch, 3),
    };
    let with_ramp = |qx: usize, qy: usize, level: usize| {
        let mut p = Pic {
            y: base.y.clone(),
            u: base.u.clone(),
            v: base.v.clone(),
        };
        ramp(&mut p.y, W, H, qx, qy, level);
        ramp(&mut p.u, cw, ch, qx, qy, level + 30);
        ramp(&mut p.v, cw, ch, qx, qy, level + 60);
        p
    };
    let p_pic = with_ramp(0, 0, 60);
    let b_pic = with_ramp(2, 2, 120);
    let cfg = EncoderConfig::new(W as u32, H as u32)
        .with_chroma_format_idc(fmt)
        .with_profile_idc(match fmt {
            3 => 244,
            2 => 122,
            _ => 100,
        })
        .with_max_num_ref_frames(2)
        .with_transform_8x8(transform_8x8)
        .with_qp(qp);
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr(&base.frame());
    let p = enc.encode_p(&p_pic.frame(), &EncodedFrameRef::from(&idr), 1, 4);
    let b = enc.encode_b(
        &b_pic.frame(),
        &EncodedFrameRef::from(&idr),
        &EncodedFrameRef::from(&p),
        2,
        2,
    );
    let mut stream = idr.annex_b.clone();
    stream.extend_from_slice(&p.annex_b);
    stream.extend_from_slice(&b.annex_b);
    Gop {
        recon: [
            (idr.recon_y, idr.recon_u, idr.recon_v),
            (b.recon_y, b.recon_u, b.recon_v),
            (p.recon_y, p.recon_u, p.recon_v),
        ],
        stream,
    }
}

/// Intra macroblocks in the stream's (P slices, B slices).
fn count_intra(stream: &[u8]) -> (usize, usize) {
    let mut dec = Decoder::new();
    let (mut in_p, mut in_b) = (0usize, 0usize);
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
            let n = sd
                .macroblocks
                .iter()
                .filter(|mb| matches!(mb.mb_type, MbType::Intra16x16(_) | MbType::INxN))
                .count();
            match header.slice_type {
                SliceType::P => in_p += n,
                SliceType::B => in_b += n,
                _ => {}
            }
        }
    }
    (in_p, in_b)
}

fn plane_eq(vf: &VideoFrame, plane: usize, recon: &[u8], w: usize, h: usize) -> bool {
    let p = &vf.planes[plane];
    (0..h).all(|r| p.data[r * p.stride..][..w] == recon[r * w..][..w])
}

fn assert_self_roundtrip(gop: &Gop, fmt: u32, tag: &str) {
    let (cw, ch) = chroma_dims(fmt);
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), gop.stream.clone()).with_pts(0);
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
    assert_eq!(frames.len(), 3, "{tag}: expected three frames");
    for (k, (vf, (ry, ru, rv))) in frames.iter().zip(&gop.recon).enumerate() {
        assert!(plane_eq(vf, 0, ry, W, H), "{tag}: frame {k} luma");
        if fmt != 0 {
            assert!(plane_eq(vf, 1, ru, cw, ch), "{tag}: frame {k} Cb");
            assert!(plane_eq(vf, 2, rv, cw, ch), "{tag}: frame {k} Cr");
        }
    }
}

fn assert_black_box(gop: &Gop, fmt: u32, tag: &str) {
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
    let stem = format!("oxideav_h264_intrafb_{tag}_{}", std::process::id());
    let (h264_path, yuv_path) = (
        dir.join(format!("{stem}.h264")),
        dir.join(format!("{stem}.yuv")),
    );
    std::fs::write(&h264_path, &gop.stream).expect("write h264");
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
    let frame_len = if fmt == 0 {
        yuv.len() / 3
    } else {
        W * H + 2 * cw * ch
    };
    assert!(
        frame_len >= W * H && yuv.len() == 3 * frame_len,
        "{tag}: expected 3 frames, got {} bytes",
        yuv.len()
    );
    for (k, (f, (ry, ru, rv))) in yuv.chunks_exact(frame_len).zip(&gop.recon).enumerate() {
        assert!(f[..W * H] == ry[..], "{tag}: frame {k} luma vs black box");
        if fmt != 0 {
            assert!(f[W * H..W * H + cw * ch] == ru[..], "{tag}: frame {k} Cb");
            assert!(f[W * H + cw * ch..] == rv[..], "{tag}: frame {k} Cr");
        }
    }
}

fn run(fmt: u32, transform_8x8: bool, qp: i32) {
    let tag = format!(
        "fmt{fmt}-{}-qp{qp}",
        if transform_8x8 { "t8" } else { "t4" }
    );
    let gop = encode_gop(fmt, transform_8x8, qp);
    let (in_p, in_b) = count_intra(&gop.stream);
    assert!(in_p > 0, "{tag}: no intra macroblock in the P slice");
    assert!(in_b > 0, "{tag}: no intra macroblock in the B slice");
    assert_self_roundtrip(&gop, fmt, &tag);
    assert_black_box(&gop, fmt, &tag);
}

#[test]
fn cavlc_intra_in_inter_400() {
    for qp in [20, 28, 36] {
        run(0, false, qp);
        run(0, true, qp);
    }
}

#[test]
fn cavlc_intra_in_inter_422() {
    for qp in [20, 28, 36] {
        run(2, false, qp);
        run(2, true, qp);
    }
}

#[test]
fn cavlc_intra_in_inter_444() {
    for qp in [20, 28, 36] {
        run(3, false, qp);
        run(3, true, qp);
    }
}
