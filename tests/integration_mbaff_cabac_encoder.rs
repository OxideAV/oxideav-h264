//! Round-456 — MBAFF **CABAC** encoder end-to-end tests.
//!
//! The `encoder::mbaff_cabac` driver codes interlaced frames as MBAFF
//! frame pictures under `entropy_coding_mode_flag = 1`: §9.3.3.1.1.2
//! `mb_field_decoding_flag` contexts, §9.3.3.1.1.1 `mb_skip_flag`
//! contexts under the §7.4.4 pair-flag inference, the eq. 9-12 /
//! 9-15 / 9-16 field/frame ref_idx and |mvd| scaling, Table 6-4
//! `coded_block_pattern` / `coded_block_flag` neighbour probes across
//! pair boundaries, the Table 9-34 FIELD residual context families and
//! the field coefficient scan on field MBs, `end_of_slice_flag` only
//! after bottom MBs.
//!
//! Each test decodes the emitted Annex B stream with our own
//! `H264CodecDecoder` (the corpus-validated MBAFF decode path is the
//! oracle) and requires the output frames to match the encoder's local
//! recon **bit-exactly**; `ffmpeg_check` re-verifies against a stock
//! black-box reference decoder binary.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::mbaff::PairMode;
use oxideav_h264::encoder::mbaff_cabac::{
    encode_mbaff_cabac_sequence, MbaffCabacConfig, MbaffCabacEncoded,
};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 80;
const H: usize = 96; // 5x6 MBs → 15 MB pairs per frame

/// Synthesize a genuinely interlaced 4:2:0 frame: each field samples a
/// moving diagonal gradient at its own time instant, so adjacent rows
/// carry real inter-field motion in the moving region while the top
/// third of the picture stays static (frame pairs win there under the
/// adaptive decision).
fn make_interlaced_frame(k: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for row in 0..H {
        let moving = row >= H / 3;
        let t = if moving { 2 * k + (row & 1) } else { 0 };
        let shift = 5 * t;
        for col in 0..W {
            let v = 16 + ((col + shift) % W + row / 2) * (235 - 16) / (W + H / 2);
            y[row * W + col] = v.clamp(0, 255) as u8;
        }
    }
    let cw = W / 2;
    let ch = H / 2;
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for row in 0..ch {
        let t = 2 * k + (row & 1);
        for col in 0..cw {
            u[row * cw + col] = (96 + ((col + 2 * t) % 32) * 2) as u8;
            v[row * cw + col] = (160u32.wrapping_sub(((col + t) % 24) as u32 * 2)) as u8;
        }
    }
    (y, u, v)
}

fn encode(pair_mode: PairMode, p_frames: bool, n_frames: usize, qp: i32) -> MbaffCabacEncoded {
    encode_with(pair_mode, p_frames, n_frames, qp, false)
}

fn encode_with(
    pair_mode: PairMode,
    p_frames: bool,
    n_frames: usize,
    qp: i32,
    intra_in_p: bool,
) -> MbaffCabacEncoded {
    let frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> =
        (0..n_frames).map(make_interlaced_frame).collect();
    let refs: Vec<(&[u8], &[u8], &[u8])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    encode_mbaff_cabac_sequence(
        &MbaffCabacConfig {
            width: W as u32,
            frame_height: H as u32,
            qp,
            pair_mode,
            p_frames,
            intra_in_p,
        },
        &refs,
    )
}

/// Decode with our own decoder (the corpus-validated MBAFF oracle).
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

fn assert_frames_match_recon(
    enc: &MbaffCabacEncoded,
    decoded: &[oxideav_core::VideoFrame],
    tag: &str,
) {
    assert_eq!(
        decoded.len(),
        enc.recon_frames.len(),
        "{tag}: decoded frame count",
    );
    for (i, (vf, (ry, ru, rv))) in decoded.iter().zip(enc.recon_frames.iter()).enumerate() {
        assert_eq!(vf.planes.len(), 3, "{tag}: frame {i} plane count");
        for (plane, (exp, name)) in vf
            .planes
            .iter()
            .zip([(ry, "Y"), (ru, "Cb"), (rv, "Cr")].iter())
        {
            let mismatches = plane
                .data
                .iter()
                .zip(exp.iter())
                .filter(|(&a, &b)| a != b)
                .count();
            if mismatches != 0 {
                // Diagnostic: distinct diverging MBs (frame coords).
                let w = plane.stride;
                let mut mbs: Vec<(usize, usize, i32)> = Vec::new();
                for (off, (&a, &b)) in plane.data.iter().zip(exp.iter()).enumerate() {
                    if a != b {
                        let (x, y) = (off % w, off / w);
                        let key = (x / 16, y / 16);
                        match mbs.iter_mut().find(|(mx, my, _)| (*mx, *my) == key) {
                            Some((_, _, dmax)) => *dmax = (*dmax).max((a as i32 - b as i32).abs()),
                            None => mbs.push((key.0, key.1, (a as i32 - b as i32).abs())),
                        }
                    }
                }
                eprintln!("{tag}: frame {i} plane {name} diverging MBs {mbs:?}");
            }
            assert_eq!(
                mismatches,
                0,
                "{tag}: frame {i} plane {name}: {mismatches}/{} samples differ",
                exp.len(),
            );
        }
    }
}

/// Cross-decoder check: a stock ffmpeg binary (black-box validator)
/// must reconstruct the exact same planes. Skips when ffmpeg is not
/// installed at the well-known Homebrew path.
fn ffmpeg_check(enc: &MbaffCabacEncoded, tag: &str) {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip {tag}: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h264-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bs = dir.join("input.h264");
    let out = dir.join("out.yuv");
    std::fs::write(&bs, &enc.annex_b).unwrap();
    let status = std::process::Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&bs)
        .args(["-f", "rawvideo"])
        .arg(&out)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "{tag}: ffmpeg failed");
    let yuv = std::fs::read(&out).unwrap();
    let frame_bytes = W * H * 3 / 2;
    assert_eq!(
        yuv.len(),
        frame_bytes * enc.recon_frames.len(),
        "{tag}: ffmpeg output size",
    );
    for (i, (ry, ru, rv)) in enc.recon_frames.iter().enumerate() {
        let base = i * frame_bytes;
        let ff_y = &yuv[base..base + W * H];
        let ff_u = &yuv[base + W * H..base + W * H + W * H / 4];
        let ff_v = &yuv[base + W * H + W * H / 4..base + frame_bytes];
        for (name, ours, ff) in [
            ("Y", ry.as_slice(), ff_y),
            ("Cb", ru.as_slice(), ff_u),
            ("Cr", rv.as_slice(), ff_v),
        ] {
            let mism = ours.iter().zip(ff.iter()).filter(|(&a, &b)| a != b).count();
            assert_eq!(
                mism,
                0,
                "{tag}: frame {i} plane {name}: {mism}/{} samples differ vs reference decoder",
                ours.len(),
            );
        }
    }
    if std::env::var("OXIDEAV_KEEP_STREAM").is_err() {
        let _ = std::fs::remove_dir_all(&dir);
    } else {
        eprintln!("kept stream at {}", bs.display());
    }
}

#[test]
fn mbaff_cabac_i_all_frame_pairs_bit_exact() {
    let enc = encode(PairMode::AllFrame, false, 2, 26);
    assert_eq!(enc.field_pairs, 0);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-cabac-i-allframe");
    ffmpeg_check(&enc, "mbaff-cabac-i-allframe");
}

#[test]
fn mbaff_cabac_i_all_field_pairs_bit_exact() {
    // Every MB a field MB: the Table 9-34 FIELD significance /
    // last-significance ctxIdxOffset families (277 / 338) and the
    // §8.5.6 field scan carry every residual block.
    let enc = encode(PairMode::AllField, false, 2, 26);
    assert_eq!(enc.frame_pairs, 0);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-cabac-i-allfield");
    ffmpeg_check(&enc, "mbaff-cabac-i-allfield");
}

#[test]
fn mbaff_cabac_i_checker_pairs_bit_exact() {
    // Checkerboard: every §6.4.12.2 current/neighbour flag pairing for
    // the mb_field_decoding_flag, mb_type, intra_chroma_pred_mode and
    // coded_block_flag contexts occurs in one picture.
    let enc = encode(PairMode::Checker, false, 2, 26);
    assert!(enc.field_pairs > 0 && enc.frame_pairs > 0);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-cabac-i-checker");
    ffmpeg_check(&enc, "mbaff-cabac-i-checker");
}

#[test]
fn mbaff_cabac_i_adaptive_pairs_bit_exact() {
    let enc = encode(PairMode::Adaptive, false, 2, 26);
    assert!(enc.field_pairs > 0 && enc.frame_pairs > 0);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-cabac-i-adaptive");
    ffmpeg_check(&enc, "mbaff-cabac-i-adaptive");
}

#[test]
fn mbaff_cabac_p_all_frame_pairs_bit_exact() {
    let enc = encode(PairMode::AllFrame, true, 4, 26);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-cabac-p-allframe");
    ffmpeg_check(&enc, "mbaff-cabac-p-allframe");
}

#[test]
fn mbaff_cabac_p_all_field_pairs_bit_exact() {
    // Field-MB P coding under CABAC: §9.3.3.1.1.6 ref_idx_l0 coded on
    // every field MB, field-context residuals, P_Skip in field geometry.
    let enc = encode(PairMode::AllField, true, 4, 26);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-cabac-p-allfield");
    ffmpeg_check(&enc, "mbaff-cabac-p-allfield");
}

#[test]
fn mbaff_cabac_p_checker_pairs_bit_exact() {
    // Mixed-geometry P: eq. 9-12 refIdxZeroFlag thresholds, eq. 9-15 /
    // 9-16 vertical |mvd| scaling, cross-pair CBP / CBF probes, and
    // skip contexts derived under the §7.4.4 inferred flag.
    let enc = encode(PairMode::Checker, true, 4, 26);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-cabac-p-checker");
    ffmpeg_check(&enc, "mbaff-cabac-p-checker");
}

#[test]
fn mbaff_cabac_p_adaptive_pairs_bit_exact() {
    let enc = encode(PairMode::Adaptive, true, 4, 26);
    assert!(enc.field_pairs > 0 && enc.frame_pairs > 0);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-cabac-p-adaptive");
    ffmpeg_check(&enc, "mbaff-cabac-p-adaptive");
}

#[test]
fn mbaff_cabac_p_intra_fallback_bit_exact() {
    // Intra_16x16 MBs inside P pictures with MBAFF neighbours: the
    // P-slice intra mb_type prefix / Table 9-36 suffix contexts,
    // intra_chroma_pred_mode and the Intra_16x16 DC / AC coded_block_flag
    // cond terms resolved against inter (and skipped) field/frame
    // neighbours.
    let enc = encode_with(PairMode::Checker, true, 4, 26, true);
    assert!(
        enc.intra_mbs_in_p > 0,
        "expected Intra_16x16 MBs in P pictures"
    );
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-cabac-p-intra");
    ffmpeg_check(&enc, "mbaff-cabac-p-intra");
}

#[test]
fn mbaff_cabac_p_high_qp_skip_pairs_bit_exact() {
    // High QP + static content maximises P_Skip runs: fully-skipped
    // pairs code no mb_field_decoding_flag, so the §7.4.4 inference
    // (and the encoder's inferred-flag re-encode) governs the stream,
    // and the mb_skip_flag contexts of skipped-top / coded-bottom pairs
    // are derived under the inferred flag.
    let frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> =
        (0..4).map(|_| make_interlaced_frame(0)).collect();
    let refs: Vec<(&[u8], &[u8], &[u8])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    let enc = encode_mbaff_cabac_sequence(
        &MbaffCabacConfig {
            width: W as u32,
            frame_height: H as u32,
            qp: 40,
            pair_mode: PairMode::Checker,
            p_frames: true,
            intra_in_p: false,
        },
        &refs,
    );
    assert!(enc.skipped_mbs > 0, "expected P_Skip macroblocks");
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-cabac-p-skip");
    ffmpeg_check(&enc, "mbaff-cabac-p-skip");
}
