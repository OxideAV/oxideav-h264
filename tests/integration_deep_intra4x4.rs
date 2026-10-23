//! Round-475 — Intra_4x4 (I_NxN) in the high-bit-depth encoder
//! (`DeepSession::with_intra_4x4`): §8.3.1 block-sequential prediction
//! at 8 / 10 / 12 / 14 bit, §8.3.1.1 predicted-mode coding, the
//! §8.5.15 V / H DPCM under transform bypass, CAVLC and CABAC. Every
//! stream decodes bit-exactly in our decoder (against the session's
//! own reconstruction) and byte-exactly in a black-box reference
//! decoder; I_NxN must actually be chosen, and at equal QP it must
//! not cost more bytes at lower quality than Intra_16x16 alone.

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_h264::encoder::deep::{DeepConfig, DeepPictureKind, DeepSession};

const W: usize = 64;
const H: usize = 48;

fn chroma_dims(cf: u32) -> (usize, usize) {
    match cf {
        1 => (W / 2, H / 2),
        2 => (W / 2, H),
        _ => (W, H),
    }
}

/// Detailed synthetic picture: diagonal edges + texture, so directional
/// 4x4 modes matter.
fn source(cf: u32, bd: u32, n: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let max = (1u32 << bd) - 1;
    let sh = bd - 8;
    let mut y = vec![0u16; W * H];
    for j in 0..H {
        for i in 0..W {
            let diag = ((i + j + n) / 6).is_multiple_of(2);
            let edge = if (i * 2 + j) % 17 < 3 { 90 } else { 0 };
            let base = if diag { 40 } else { 120 } + edge + ((i * 5 + j * 3) % 23);
            let fine = ((i * 7 + j * 11) % 3) as u32;
            y[j * W + i] = (((base as u32) << sh) + fine * ((1 << sh) / 2 + 1)).min(max) as u16;
        }
    }
    let (cw, ch) = chroma_dims(cf);
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            u[j * cw + i] = (((80 + ((i * 3 + j + n) % 70)) as u32) << sh).min(max) as u16;
            v[j * cw + i] = (((130 + ((i + j * 2) % 60)) as u32) << sh).min(max) as u16;
        }
    }
    (y, u, v)
}

fn le_bytes(s: &[u16]) -> Vec<u8> {
    s.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn decode_ours(stream: &[u8], bd: u32) -> Vec<Vec<u8>> {
    let p = CodecParameters::video(CodecId::new("h264"));
    let mut dec = oxideav_h264::h264_decoder::make_decoder(&p).expect("decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 30), stream.to_vec());
    dec.send_packet(&pkt).expect("send");
    let mut out = Vec::new();
    let mut push = |vf: oxideav_core::VideoFrame| {
        let mut b = Vec::new();
        for pl in vf.image_planes() {
            b.extend_from_slice(&pl.data);
        }
        out.push(b);
    };
    while let Ok(Frame::Video(vf)) = dec.receive_frame() {
        push(vf);
    }
    dec.flush().expect("flush");
    while let Ok(Frame::Video(vf)) = dec.receive_frame() {
        push(vf);
    }
    let _ = bd;
    out
}

fn reference_decode(stream: &[u8], tag: &str) -> Option<Vec<u8>> {
    let refbin = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !refbin.exists() {
        eprintln!("skip {tag}: reference decoder binary not present");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h264-i4x4-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bs = dir.join("in.h264");
    let out = dir.join("out.yuv");
    std::fs::write(&bs, stream).unwrap();
    let status = std::process::Command::new(refbin)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&bs)
        .args(["-f", "rawvideo"])
        .arg(&out)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success(), "{tag}: reference decoder failed");
    let raw = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some(raw)
}

/// Encode two I pictures (IDR + I) with / without I_NxN; returns
/// `(bytes, min luma PSNR, i4x4 MBs)` for the I_NxN session and
/// `(bytes, min luma PSNR)` for the Intra_16x16-only one, after the
/// bit-exact / byte-exact gates on the I_NxN stream.
fn run(
    cf: u32,
    bd: u32,
    cabac: bool,
    lossless: bool,
    qp: i32,
) -> ((usize, f64, usize), (usize, f64)) {
    // The tag names the reference decoder's scratch directory, so it
    // must be unique across the concurrently running tests (QP too).
    let tag = format!(
        "cf{cf}-bd{bd}-{}-{}-qp{qp}",
        if cabac { "cabac" } else { "cavlc" },
        if lossless { "ll" } else { "lossy" }
    );
    let cfg = DeepConfig {
        width: W as u32,
        height: H as u32,
        bit_depth_luma: bd,
        bit_depth_chroma: bd,
        chroma_format_idc: cf,
        qp: if lossless { -6 * (bd as i32 - 8) } else { qp },
        p_frames: false,
        intra_in_p: false,
        deblock: true,
        lossless,
        lossless_interop: true,
        cabac,
        transform_8x8: false,
    };
    let frames: Vec<_> = (0..2).map(|n| source(cf, bd, n)).collect();
    let mut results = Vec::new();
    for intra_4x4 in [true, false] {
        let mut s = DeepSession::new(cfg.clone()).with_intra_4x4(intra_4x4);
        let mut stream = s.parameter_sets_annex_b();
        let mut recons = Vec::new();
        for (k, (y, u, v)) in frames.iter().enumerate() {
            let kind = if k == 0 {
                DeepPictureKind::Idr
            } else {
                DeepPictureKind::I
            };
            let pic = s.encode_picture(y, u, v, kind, k as u32, 2 * k as i32, 0);
            stream.extend_from_slice(&pic.nal);
            recons.push(pic.recon);
        }
        let max = ((1u32 << bd) - 1) as f64;
        let mut min_psnr = f64::MAX;
        for (k, (ry, _, _)) in recons.iter().enumerate() {
            let sy = &frames[k].0;
            let mse: f64 = sy
                .iter()
                .zip(ry.iter())
                .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                .sum::<f64>()
                / sy.len() as f64;
            let p = if mse == 0.0 {
                99.0
            } else {
                10.0 * (max * max / mse).log10()
            };
            min_psnr = min_psnr.min(p);
            if lossless {
                assert_eq!(
                    (&frames[k].0, &frames[k].1, &frames[k].2),
                    (&recons[k].0, &recons[k].1, &recons[k].2),
                    "{tag}: lossless recon"
                );
            }
        }
        if intra_4x4 {
            assert!(s.i4x4_mbs() > 0, "{tag}: no I_NxN macroblock chosen");
            // Ours: bit-exact against the session's reconstruction.
            let ours = decode_ours(&stream, bd);
            assert_eq!(ours.len(), 2, "{tag}: picture count");
            for (k, (ry, ru, rv)) in recons.iter().enumerate() {
                let mut exp = Vec::new();
                if bd > 8 {
                    exp.extend(le_bytes(ry));
                    exp.extend(le_bytes(ru));
                    exp.extend(le_bytes(rv));
                } else {
                    exp.extend(ry.iter().map(|&x| x as u8));
                    exp.extend(ru.iter().map(|&x| x as u8));
                    exp.extend(rv.iter().map(|&x| x as u8));
                }
                assert_eq!(
                    ours[k], exp,
                    "{tag}: picture {k} not bit-exact in our decoder"
                );
            }
            if let Some(raw) = reference_decode(&stream, &tag) {
                let len = ours[0].len();
                assert_eq!(raw.len(), 2 * len, "{tag}: reference picture count");
                for (k, o) in ours.iter().enumerate() {
                    let mism = o
                        .iter()
                        .zip(&raw[k * len..(k + 1) * len])
                        .filter(|(a, b)| a != b)
                        .count();
                    assert_eq!(
                        mism, 0,
                        "{tag}: picture {k}: {mism} bytes differ vs reference decoder"
                    );
                }
            }
            results.push((stream.len(), min_psnr, s.i4x4_mbs()));
        } else {
            results.push((stream.len(), min_psnr, 0));
        }
    }
    let (a, b) = (results[0], results[1]);
    println!(
        "I4X4 {tag} qp={qp}: with I_NxN {} B / {:.2} dB ({} I_NxN MBs of {}), Intra_16x16 only {} B / {:.2} dB",
        a.0,
        a.1,
        a.2,
        2 * (W / 16) * (H / 16),
        b.0,
        b.1
    );
    ((a.0, a.1, a.2), (b.0, b.1))
}

#[test]
fn intra_4x4_lossy_matrix_bit_exact() {
    for (cf, bd) in [
        (1u32, 8u32),
        (1, 10),
        (2, 10),
        (3, 8),
        (2, 12),
        (3, 12),
        (1, 14),
        (3, 14),
    ] {
        for cabac in [false, true] {
            let ((b4, p4, _), (b16, p16)) = run(cf, bd, cabac, false, 26);
            // The I_NxN session must not be worse on both axes.
            assert!(
                b4 <= b16 || p4 >= p16,
                "cf{cf} bd{bd} cabac={cabac}: I_NxN {b4} B / {p4:.2} dB vs I16 {b16} B / {p16:.2} dB"
            );
        }
    }
}

#[test]
fn intra_4x4_lossless_exact_with_dpcm() {
    for (cf, bd) in [(1u32, 8u32), (2, 10), (3, 12), (1, 14)] {
        for cabac in [false, true] {
            let ((_, p4, _), _) = run(cf, bd, cabac, true, 0);
            assert_eq!(p4, 99.0);
        }
    }
}

#[test]
fn intra_4x4_qp_sweep_prints_table() {
    for qp in [18, 30, 40] {
        run(1, 10, false, false, qp);
        run(3, 8, true, false, qp);
    }
}
