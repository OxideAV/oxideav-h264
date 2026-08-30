//! Round-453 — MBAFF (macroblock-adaptive frame/field) encoder
//! end-to-end tests.
//!
//! The `encoder::mbaff` driver codes interlaced frames as MBAFF frame
//! pictures (`frame_mbs_only_flag = 0`,
//! `mb_adaptive_frame_field_flag = 1`, `field_pic_flag = 0`): per-pair
//! §7.3.4 `mb_field_decoding_flag`, §6.4.12.2 Table 6-4 neighbour
//! derivation, §8.4.1.3.2 eq. 8-217..8-220 MV/refIdx frame↔field
//! scaling, §8.4.2.1 field-of-reference-frame MC for field MBs, the
//! §7.4.4 skipped-pair flag inference, and the §8.7 MBAFF deblock.
//!
//! Each test decodes the emitted Annex B stream with our own
//! `H264CodecDecoder` (the corpus-validated MBAFF decode path is the
//! oracle) and requires the output frames to match the encoder's local
//! recon **bit-exactly**; `ffmpeg_check` re-verifies against a stock
//! black-box reference decoder binary.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::mbaff::{encode_mbaff_sequence, MbaffConfig, MbaffEncoded, PairMode};
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

fn encode(pair_mode: PairMode, p_frames: bool, n_frames: usize, qp: i32) -> MbaffEncoded {
    let frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> =
        (0..n_frames).map(make_interlaced_frame).collect();
    let refs: Vec<(&[u8], &[u8], &[u8])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    encode_mbaff_sequence(
        &MbaffConfig {
            width: W as u32,
            frame_height: H as u32,
            qp,
            pair_mode,
            p_frames,
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

fn assert_frames_match_recon(enc: &MbaffEncoded, decoded: &[oxideav_core::VideoFrame], tag: &str) {
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
fn ffmpeg_check(enc: &MbaffEncoded, tag: &str) {
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
fn mbaff_i_all_frame_pairs_bit_exact() {
    // Degenerate baseline: every pair frame-coded — pins the MBAFF
    // address interleave + flag emission with pure frame content.
    let enc = encode(PairMode::AllFrame, false, 2, 26);
    assert_eq!(enc.field_pairs, 0);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-i-allframe");
    ffmpeg_check(&enc, "mbaff-i-allframe");
}

#[test]
fn mbaff_i_all_field_pairs_bit_exact() {
    // Every pair field-coded: §6.4.1 eq. 6-9/6-10 interleave, the
    // §8.5.6 field coefficient scan on every MB, field-geometry intra
    // neighbours throughout.
    let enc = encode(PairMode::AllField, false, 2, 26);
    assert_eq!(enc.frame_pairs, 0);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-i-allfield");
    ffmpeg_check(&enc, "mbaff-i-allfield");
}

#[test]
fn mbaff_i_checker_pairs_bit_exact() {
    // Checkerboard frame/field pairs: every Table 6-4 current/
    // neighbour flag combination (frame beside field, field above
    // frame, ...) occurs in one picture.
    let enc = encode(PairMode::Checker, false, 2, 26);
    assert!(enc.field_pairs > 0 && enc.frame_pairs > 0);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-i-checker");
    ffmpeg_check(&enc, "mbaff-i-checker");
}

#[test]
fn mbaff_i_adaptive_pairs_bit_exact() {
    // Content-driven decision: the static top third picks frame pairs,
    // the moving region picks field pairs.
    let enc = encode(PairMode::Adaptive, false, 2, 26);
    assert!(
        enc.field_pairs > 0 && enc.frame_pairs > 0,
        "adaptive decision must produce a mixed picture (field {}, frame {})",
        enc.field_pairs,
        enc.frame_pairs,
    );
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-i-adaptive");
    ffmpeg_check(&enc, "mbaff-i-adaptive");
}

#[test]
fn mbaff_p_all_frame_pairs_bit_exact() {
    let enc = encode(PairMode::AllFrame, true, 4, 26);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-p-allframe");
    ffmpeg_check(&enc, "mbaff-p-allframe");
}

#[test]
fn mbaff_p_all_field_pairs_bit_exact() {
    // Field-MB P coding: §8.4.2.1 same-parity field-of-frame
    // references, §7.3.5.1 ref_idx te(v) with the doubled field list,
    // field-unit MVs + P_Skip in field geometry.
    let enc = encode(PairMode::AllField, true, 4, 26);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-p-allfield");
    ffmpeg_check(&enc, "mbaff-p-allfield");
}

#[test]
fn mbaff_p_checker_pairs_bit_exact() {
    // Mixed-geometry P: eq. 8-217..8-220 MV/refIdx scaling fires on
    // every frame↔field neighbour read; skipped pairs exercise the
    // §7.4.4 inference (checkerboard decisions differ from the
    // inferred left-pair flag, forcing re-encodes).
    let enc = encode(PairMode::Checker, true, 4, 26);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-p-checker");
    ffmpeg_check(&enc, "mbaff-p-checker");
}

#[test]
fn mbaff_p_adaptive_pairs_bit_exact() {
    let enc = encode(PairMode::Adaptive, true, 4, 26);
    assert!(enc.field_pairs > 0 && enc.frame_pairs > 0);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-p-adaptive");
    ffmpeg_check(&enc, "mbaff-p-adaptive");
}

#[test]
fn mbaff_p_high_qp_skip_pairs_bit_exact() {
    // High QP + static content maximises P_Skip runs: fully-skipped
    // pairs code no mb_field_decoding_flag, so the §7.4.4 inference
    // and the driver's inferred-flag re-encode govern the stream.
    let frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> =
        (0..4).map(|_| make_interlaced_frame(0)).collect();
    let refs: Vec<(&[u8], &[u8], &[u8])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    let enc = encode_mbaff_sequence(
        &MbaffConfig {
            width: W as u32,
            frame_height: H as u32,
            qp: 40,
            pair_mode: PairMode::Checker,
            p_frames: true,
        },
        &refs,
    );
    assert!(
        enc.skipped_mbs > 0,
        "static content at QP 40 must produce P_Skip macroblocks",
    );
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "mbaff-p-skip");
    ffmpeg_check(&enc, "mbaff-p-skip");
}
