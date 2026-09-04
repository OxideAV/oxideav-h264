//! Round-456 — high-bit-depth (10 / 12 / 14-bit) encoder gates.
//!
//! `encoder::deep` codes `u16` sources at every chroma format under
//! the High 10 / High 4:2:2 / High 4:4:4 Predictive profiles. Each
//! stream must (1) decode bit-exactly to the encoder's own
//! reconstruction in our decoder, (2) decode byte-identically in a
//! stock black-box reference decoder, and (3) reconstruct the source
//! at a PSNR consistent with the QP at that depth — so the deep
//! quantiser scale (`QP′ = QP + QpBdOffset`) is genuinely right, not
//! merely self-consistent.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::deep::{encode_deep_sequence, DeepConfig, DeepEncoded, DeepPlanes};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 64;
const H: usize = 48;

fn chroma_dims(cf: u32) -> (usize, usize) {
    match cf {
        1 => (W / 2, H / 2),
        2 => (W / 2, H),
        _ => (W, H),
    }
}

/// Moving textured gradient spanning the full sample range at `bd`.
fn source(k: usize, bd: u32, cf: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let max = (1u32 << bd) - 1;
    let (cw, ch) = chroma_dims(cf);
    let mut y = vec![0u16; W * H];
    for j in 0..H {
        for i in 0..W {
            let base =
                ((i + 3 * k) * max as usize / (W + 8)) as u32 + (j as u32 * max / (2 * H as u32));
            let tex = (((i * 7 + j * 13) % 11) as u32 * max) / 200;
            y[j * W + i] = (base + tex).min(max) as u16;
        }
    }
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            let t = (i + 2 * k) as u32;
            u[j * cw + i] = ((max / 4 + (t * 37 + j as u32 * 5) % (max / 2)).min(max)) as u16;
            v[j * cw + i] = ((max / 3 + (t * 23 + j as u32 * 11) % (max / 3)).min(max)) as u16;
        }
    }
    (y, u, v)
}

fn encode(bd: u32, cf: u32, qp: i32, n: usize, deblock: bool) -> (DeepEncoded, Vec<DeepPlanes>) {
    let frames: Vec<_> = (0..n).map(|k| source(k, bd, cf)).collect();
    let refs: Vec<(&[u16], &[u16], &[u16])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    let enc = encode_deep_sequence(
        &DeepConfig {
            width: W as u32,
            height: H as u32,
            bit_depth_luma: bd,
            bit_depth_chroma: bd,
            chroma_format_idc: cf,
            qp,
            p_frames: true,
            intra_in_p: true,
            deblock,
            lossless: false,
            lossless_interop: false,
        },
        &refs,
    );
    (enc, frames)
}

fn encode_lossless(bd: u32, cf: u32, n: usize, interop: bool) -> (DeepEncoded, Vec<DeepPlanes>) {
    let frames: Vec<_> = (0..n).map(|k| source(k, bd, cf)).collect();
    let refs: Vec<(&[u16], &[u16], &[u16])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    let enc = encode_deep_sequence(
        &DeepConfig {
            width: W as u32,
            height: H as u32,
            bit_depth_luma: bd,
            bit_depth_chroma: bd,
            chroma_format_idc: cf,
            qp: -(6 * (bd as i32 - 8)),
            p_frames: true,
            intra_in_p: true,
            deblock: true,
            lossless: true,
            lossless_interop: interop,
        },
        &refs,
    );
    (enc, frames)
}

fn le_bytes(src: &[u16]) -> Vec<u8> {
    src.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn decode_ours(stream: &[u8]) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream.to_vec()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut out = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => {
                assert_eq!(vf.planes.len(), 3);
                out.push((
                    vf.planes[0].data.to_vec(),
                    vf.planes[1].data.to_vec(),
                    vf.planes[2].data.to_vec(),
                ));
            }
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    out
}

fn assert_ours_bit_exact(enc: &DeepEncoded, tag: &str) {
    let decoded = decode_ours(&enc.annex_b);
    assert_eq!(decoded.len(), enc.recon_frames.len(), "{tag}: frame count");
    for (i, ((dy, du, dv), (ry, ru, rv))) in decoded.iter().zip(enc.recon_frames.iter()).enumerate()
    {
        for (name, d, r) in [("Y", dy, ry), ("Cb", du, ru), ("Cr", dv, rv)] {
            let exp = le_bytes(r);
            let mism = d.iter().zip(exp.iter()).filter(|(a, b)| a != b).count();
            assert_eq!(d.len(), exp.len(), "{tag}: frame {i} {name} size");
            assert_eq!(
                mism, 0,
                "{tag}: frame {i} plane {name}: {mism} bytes differ"
            );
        }
    }
}

fn reference_decoder_check(enc: &DeepEncoded, cf: u32, tag: &str) {
    let refbin = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !refbin.exists() {
        eprintln!("skip {tag}: reference decoder binary not present");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h264-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bs = dir.join("input.h264");
    let out = dir.join("out.yuv");
    std::fs::write(&bs, &enc.annex_b).unwrap();
    let status = std::process::Command::new(refbin)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&bs)
        .args(["-f", "rawvideo"])
        .arg(&out)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success(), "{tag}: reference decoder failed");
    let raw = std::fs::read(&out).unwrap();
    let (cw, ch) = chroma_dims(cf);
    let frame_bytes = (W * H + 2 * cw * ch) * 2;
    assert_eq!(
        raw.len(),
        frame_bytes * enc.recon_frames.len(),
        "{tag}: frame count"
    );
    for (i, (ry, ru, rv)) in enc.recon_frames.iter().enumerate() {
        let f = &raw[i * frame_bytes..(i + 1) * frame_bytes];
        let c = cw * ch * 2;
        for (name, ours, theirs) in [
            ("Y", le_bytes(ry), &f[..W * H * 2]),
            ("Cb", le_bytes(ru), &f[W * H * 2..W * H * 2 + c]),
            ("Cr", le_bytes(rv), &f[W * H * 2 + c..]),
        ] {
            let mism = ours
                .iter()
                .zip(theirs.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                mism, 0,
                "{tag}: frame {i} plane {name}: {mism} bytes differ vs reference decoder"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

fn psnr(a: &[u16], b: &[u16], bd: u32) -> f64 {
    let max = ((1u32 << bd) - 1) as f64;
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        return 99.0;
    }
    10.0 * (max * max / mse).log10()
}

fn run_matrix(bd: u32, cf: u32, qp: i32, min_psnr: f64) {
    let tag = format!("deep{bd}-cf{cf}");
    let (enc, src) = encode(bd, cf, qp, 3, true);
    assert!(
        enc.intra_mbs_in_p + enc.skipped_mbs < W * H / 256 * 2,
        "{tag}: P pictures should carry inter MBs"
    );
    for (i, ((ry, ru, rv), (sy, su, sv))) in enc.recon_frames.iter().zip(src.iter()).enumerate() {
        let (py, pu, pv) = (psnr(ry, sy, bd), psnr(ru, su, bd), psnr(rv, sv, bd));
        eprintln!("{tag}: frame {i} PSNR Y {py:.2} Cb {pu:.2} Cr {pv:.2}");
        assert!(
            py > min_psnr && pu > min_psnr && pv > min_psnr,
            "{tag}: frame {i} PSNR too low"
        );
        // The recon must genuinely use the deep range.
        assert!(
            ry.iter().any(|&s| s > (1 << (bd - 1))),
            "{tag}: recon never exceeds half range"
        );
    }
    assert_ours_bit_exact(&enc, &tag);
    reference_decoder_check(&enc, cf, &tag);
}

#[test]
fn deep_10bit_420_bit_exact() {
    run_matrix(10, 1, 22, 40.0);
}

#[test]
fn deep_12bit_420_bit_exact() {
    run_matrix(12, 1, 22, 40.0);
}

#[test]
fn deep_14bit_420_bit_exact() {
    run_matrix(14, 1, 22, 40.0);
}

#[test]
fn deep_12bit_422_bit_exact() {
    run_matrix(12, 2, 22, 40.0);
}

#[test]
fn deep_14bit_422_bit_exact() {
    run_matrix(14, 2, 22, 40.0);
}

#[test]
fn deep_12bit_444_bit_exact() {
    run_matrix(12, 3, 22, 40.0);
}

#[test]
fn deep_14bit_444_bit_exact() {
    run_matrix(14, 3, 22, 40.0);
}

#[test]
fn deep_14bit_negative_qp_bit_exact() {
    // §7.4.3 — QP_Y below 0 (down to −QpBdOffsetY = −36) is legal at
    // 14-bit: QP′Y = 6, near-lossless quantisation.
    let tag = "deep14-negqp";
    let (enc, src) = encode(14, 1, -30, 2, true);
    for ((ry, _, _), (sy, _, _)) in enc.recon_frames.iter().zip(src.iter()) {
        assert!(psnr(ry, sy, 14) > 70.0, "{tag}: near-lossless PSNR");
    }
    assert_ours_bit_exact(&enc, tag);
    reference_decoder_check(&enc, 1, tag);
}

#[test]
fn deep_12bit_no_deblock_bit_exact() {
    let tag = "deep12-nodeblock";
    let (enc, _) = encode(12, 1, 30, 3, false);
    assert_ours_bit_exact(&enc, tag);
    reference_decoder_check(&enc, 1, tag);
}

/// §7.4.2.1.1 `qpprime_y_zero_transform_bypass_flag` + QP′Y == 0:
/// the reconstruction must equal the source SAMPLE-FOR-SAMPLE (I
/// pictures through the §8.5.15 V/H DPCM on Intra_16x16 luma and
/// chroma, P pictures through the eq. 8-334 identity on inter
/// residuals) at every depth and chroma format, and our decoder must
/// reproduce it bit-exactly. The spec-literal streams are NOT run
/// through the black-box reference decoder: it skips the §8.5.2 step
/// 3 / §8.5.4 step 3 DPCM (Intra_16x16 + chroma), so its output
/// differs from the spec's on exactly those macroblocks (verified by
/// forcing each intra mode in turn: DC / Plane agree, V / H disagree,
/// and V / H agree again once the DPCM is withheld) — see
/// `deep_lossless_interop_reference_decoder_exact`.
#[test]
fn deep_lossless_bypass_exact_reconstruction() {
    for (bd, cf) in [(8u32, 1u32), (10, 1), (12, 2), (14, 3), (14, 1)] {
        let tag = format!("lossless{bd}-cf{cf}");
        let (enc, src) = encode_lossless(bd, cf, 3, false);
        assert_eq!(enc.recon_frames.len(), src.len());
        for (i, (r, s)) in enc.recon_frames.iter().zip(src.iter()).enumerate() {
            assert_eq!(r.0, s.0, "{tag}: frame {i} luma not lossless");
            assert_eq!(r.1, s.1, "{tag}: frame {i} Cb not lossless");
            assert_eq!(r.2, s.2, "{tag}: frame {i} Cr not lossless");
        }
        if bd > 8 {
            assert_ours_bit_exact(&enc, &tag);
        }
    }
}

/// `DeepConfig::lossless_interop` — DC / Plane intra modes only, so no
/// §8.5.15 DPCM is ever invoked: still lossless, and now byte-exact in
/// the black-box reference decoder as well.
#[test]
fn deep_lossless_interop_reference_decoder_exact() {
    for (bd, cf) in [(10u32, 1u32), (12, 2), (14, 3)] {
        let tag = format!("lossless-interop{bd}-cf{cf}");
        let (enc, src) = encode_lossless(bd, cf, 3, true);
        for (i, (r, s)) in enc.recon_frames.iter().zip(src.iter()).enumerate() {
            assert!(
                r.0 == s.0 && r.1 == s.1 && r.2 == s.2,
                "{tag}: frame {i} not lossless"
            );
        }
        assert_ours_bit_exact(&enc, &tag);
        reference_decoder_check(&enc, cf, &tag);
    }
}
