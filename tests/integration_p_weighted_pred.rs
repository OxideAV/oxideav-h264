//! Round-453 — explicit weighted prediction on **P slices** (PPS
//! `weighted_pred_flag = 1`, per-slice §7.3.3.2 `pred_weight_table()`,
//! §8.4.2.3.2 eq. 8-274 single-list weighting) with fade detection.
//!
//! Fixture: a textured IDR followed by P pictures that are global
//! brightness fades of the same texture (`src = a · ref + b`). With
//! default prediction every sample carries the fade as residual; the
//! encoder's slice-wide least-squares fit elects a luma
//! `(weight, offset)` pair at `logWD = 5` that collapses the residual.
//!
//! Gates:
//!   * the weighted stream's P NALs are materially smaller than the
//!     unweighted encoder's at the same QP, with PSNR not worse;
//!   * a static (non-fade) P picture still ships the mandatory table
//!     but elects the identity (`luma_weight_l0_flag = 0`);
//!   * every stream decodes bit-exactly in our own decoder against the
//!     encoder recon and byte-exactly in the black-box reference
//!     decoder (P_Skip, P_L0_16x16, P_8x8 and Intra_16x16 fallback all
//!     take the weighted predictor).

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::{EncodedFrameRef, EncodedP, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 96;
const H: usize = 64;

/// Textured base picture (mid brightness, diagonal + block texture).
fn make_base() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let tex = ((i * 3 + j * 5) % 41) as i32 + (((i / 8 + j / 8) % 2) as i32) * 20;
            y[j * W + i] = (90 + tex).clamp(0, 255) as u8;
        }
    }
    let cw = W / 2;
    let ch = H / 2;
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            u[j * cw + i] = (110 + ((i + j) % 17)) as u8;
            v[j * cw + i] = (140 + ((i * 2 + j) % 13)) as u8;
        }
    }
    (y, u, v)
}

/// Global luma fade of `base`: `a · y + b`, clipped.
fn fade(base: &(Vec<u8>, Vec<u8>, Vec<u8>), a: f64, b: f64) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y = base
        .0
        .iter()
        .map(|&p| ((p as f64) * a + b).round().clamp(0.0, 255.0) as u8)
        .collect();
    (y, base.1.clone(), base.2.clone())
}

fn mk(f: &(Vec<u8>, Vec<u8>, Vec<u8>)) -> YuvFrame<'_> {
    YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &f.0,
        u: &f.1,
        v: &f.2,
    }
}

fn psnr(orig: &[u8], recon: &[u8]) -> f64 {
    let n = orig.len() as f64;
    let s: f64 = orig
        .iter()
        .zip(recon.iter())
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum();
    let m = s / n;
    if m <= 0.0 {
        99.0
    } else {
        10.0 * (255.0_f64 * 255.0 / m).log10()
    }
}

struct Run {
    annex_b: Vec<u8>,
    recon: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
    p_bytes: Vec<usize>,
    /// Per P picture: `(table present, non-identity luma weights)`.
    tables: Vec<(bool, bool)>,
    p_psnr: Vec<f64>,
}

fn encode_sequence(frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)], weighted: bool, qp: i32) -> Run {
    let mut cfg = EncoderConfig::new(W as u32, H as u32);
    cfg.qp = qp;
    cfg.explicit_weighted_pred = weighted;
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr(&mk(&frames[0]));
    let mut annex_b = idr.annex_b.clone();
    let mut recon = vec![(
        idr.recon_y.clone(),
        idr.recon_u.clone(),
        idr.recon_v.clone(),
    )];
    let mut coded_p: Vec<EncodedP> = Vec::new();
    let mut p_bytes = Vec::new();
    let mut tables = Vec::new();
    let mut p_psnr = Vec::new();
    for (k, f) in frames.iter().enumerate().skip(1) {
        let p = {
            let prev_ref = match coded_p.last() {
                Some(prev) => EncodedFrameRef::from(prev),
                None => EncodedFrameRef::from(&idr),
            };
            enc.encode_p(&mk(f), &prev_ref, k as u32, (2 * k) as u32)
        };
        annex_b.extend_from_slice(&p.annex_b);
        p_bytes.push(p.annex_b.len());
        tables.push((
            p.pred_weight_table.is_some(),
            p.pred_weight_table.is_some_and(|t| t.luma.is_some()),
        ));
        p_psnr.push(psnr(&f.0, &p.recon_y));
        eprintln!(
            "P{k}: table {:?}, bytes {}, intra parts {}/{}",
            p.pred_weight_table,
            p.annex_b.len(),
            p.partition_mvs.iter().filter(|m| m.is_intra).count(),
            p.partition_mvs.len(),
        );
        recon.push((p.recon_y.clone(), p.recon_u.clone(), p.recon_v.clone()));
        coded_p.push(p);
    }
    Run {
        annex_b,
        recon,
        p_bytes,
        tables,
        p_psnr,
    }
}

fn decode_ours(annex_b: &[u8]) -> Vec<oxideav_core::VideoFrame> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), annex_b.to_vec()).with_pts(0);
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
    frames
}

fn assert_bit_exact(run: &Run, tag: &str) {
    let decoded = decode_ours(&run.annex_b);
    assert_eq!(decoded.len(), run.recon.len(), "{tag}: frame count");
    for (i, (vf, (ry, ru, rv))) in decoded.iter().zip(run.recon.iter()).enumerate() {
        for (plane, (exp, name)) in vf
            .planes
            .iter()
            .zip([(ry, "Y"), (ru, "Cb"), (rv, "Cr")].iter())
        {
            let mism = plane
                .data
                .iter()
                .zip(exp.iter())
                .filter(|(&a, &b)| a != b)
                .count();
            assert_eq!(
                mism, 0,
                "{tag}: frame {i} plane {name}: {mism} samples differ"
            );
        }
    }
}

/// Black-box reference decoder cross-check (skips when ffmpeg is not
/// installed at the well-known Homebrew path).
fn ffmpeg_check(run: &Run, tag: &str) {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip {tag}: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h264-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bs = dir.join("input.h264");
    let out = dir.join("out.yuv");
    std::fs::write(&bs, &run.annex_b).unwrap();
    let status = std::process::Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&bs)
        .args(["-f", "rawvideo"])
        .arg(&out)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "{tag}: ffmpeg failed");
    let yuv = std::fs::read(&out).unwrap();
    let fb = W * H * 3 / 2;
    assert_eq!(yuv.len(), fb * run.recon.len(), "{tag}: ffmpeg output size");
    for (i, (ry, ru, rv)) in run.recon.iter().enumerate() {
        let base = i * fb;
        let planes = [
            ("Y", ry.as_slice(), &yuv[base..base + W * H]),
            (
                "Cb",
                ru.as_slice(),
                &yuv[base + W * H..base + W * H * 5 / 4],
            ),
            ("Cr", rv.as_slice(), &yuv[base + W * H * 5 / 4..base + fb]),
        ];
        for (name, ours, ff) in planes {
            let mism = ours.iter().zip(ff.iter()).filter(|(&a, &b)| a != b).count();
            assert_eq!(
                mism, 0,
                "{tag}: frame {i} plane {name}: {mism} samples differ vs reference decoder",
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

fn fade_sequence() -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let base = make_base();
    vec![
        base.clone(),
        fade(&base, 1.25, 10.0),  // brighten + offset
        fade(&base, 1.55, 20.0),  // keep brightening
        fade(&base, 1.20, -30.0), // fade down with negative offset
        fade(&base, 0.70, -5.0),  // dark
    ]
}

#[test]
fn p_weighted_pred_fade_elects_weights_and_shrinks_stream() {
    let frames = fade_sequence();
    let qp = 26;
    let plain = encode_sequence(&frames, false, qp);
    let weighted = encode_sequence(&frames, true, qp);
    assert!(plain.tables.iter().all(|&(present, _)| !present));
    assert!(
        weighted
            .tables
            .iter()
            .all(|&(present, non_id)| present && non_id),
        "every fade P picture must elect a non-identity table: {:?}",
        weighted.tables,
    );
    let plain_total: usize = plain.p_bytes.iter().sum();
    let weighted_total: usize = weighted.p_bytes.iter().sum();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    eprintln!(
        "P bytes plain {plain_total} → weighted {weighted_total} ({:.1}%); PSNR plain {:.2} → weighted {:.2} dB",
        100.0 * weighted_total as f64 / plain_total as f64,
        mean(&plain.p_psnr),
        mean(&weighted.p_psnr),
    );
    assert!(
        weighted_total * 100 <= plain_total * 80,
        "weighted P pictures must be ≥ 20% smaller ({weighted_total} vs {plain_total})",
    );
    assert!(
        mean(&weighted.p_psnr) + 0.1 >= mean(&plain.p_psnr),
        "weighted PSNR must not drop",
    );
    assert_bit_exact(&plain, "p-wp-plain");
    assert_bit_exact(&weighted, "p-wp-weighted");
    ffmpeg_check(&weighted, "p-wp-weighted");
}

#[test]
fn p_weighted_pred_static_content_elects_identity_table() {
    let base = make_base();
    let frames = vec![base.clone(), base.clone(), fade(&base, 1.0, 1.0)];
    let weighted = encode_sequence(&frames, true, 26);
    assert_eq!(
        weighted.tables[0],
        (true, false),
        "a repeated picture must ship the mandatory table with luma_weight_l0_flag = 0",
    );
    assert_bit_exact(&weighted, "p-wp-static");
    ffmpeg_check(&weighted, "p-wp-static");
}

#[test]
fn p_weighted_pred_high_qp_skip_runs_use_weighted_predictor() {
    // High QP makes most MBs P_Skip: the decoder weights P_Skip
    // predictions too, so the skip decision must have been taken on
    // the weighted predictor.
    let frames = fade_sequence();
    let weighted = encode_sequence(&frames, true, 44);
    assert!(weighted.tables.iter().all(|&(present, _)| present));
    assert_bit_exact(&weighted, "p-wp-qp44");
    ffmpeg_check(&weighted, "p-wp-qp44");
}
