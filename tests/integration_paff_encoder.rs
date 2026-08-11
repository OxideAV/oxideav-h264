//! Round-416 — PAFF (picture-adaptive frame/field) end-to-end tests.
//!
//! The `encoder::field` driver splits interlaced frames into
//! half-height field pictures (`field_pic_flag == 1`, top-field-first)
//! and codes them per §7.4.3 — frame 0 as IDR top + non-IDR I bottom,
//! later frames as I/I or P/P field pairs (single L0 reference = the
//! same-parity field of the previous frame per §8.2.4.2.5), optionally
//! interleaving full-height I FRAME pictures (`field_pic_flag == 0`)
//! for the picture-adaptive axis.
//!
//! Each test decodes the emitted Annex B stream with our own
//! `H264CodecDecoder` and requires the re-interleaved full-height
//! output frames to match the encoder's local recon **bit-exactly** —
//! this pins the decoder's PAFF path (field slice headers, §8.2.1
//! field POC, §C.4.4 complementary-pair output interleave, §8.7 field
//! deblock, §8.2.4.2.5 field reference lists + field MC) against the
//! §8-conformant reconstruction the encoder mirrors.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::field::{encode_paff_sequence, PaffConfig, PaffEncoded};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 64;
const H: usize = 96; // frame height; fields are 64x48

/// Synthesize a genuinely interlaced 4:2:0 frame: each field samples a
/// moving diagonal gradient at its own time instant (top field = time
/// 2k, bottom field = 2k + 1), so adjacent rows carry real inter-field
/// motion — the content PAFF exists for.
fn make_interlaced_frame(k: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for row in 0..H {
        let t = 2 * k + (row & 1); // field time instant
        let shift = 3 * t;
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

fn encode_cfg2(
    p_fields: bool,
    frame_pictures: Vec<usize>,
    n_frames: usize,
    cross_parity: bool,
    idr_frame_first: bool,
) -> PaffEncoded {
    let frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> =
        (0..n_frames).map(make_interlaced_frame).collect();
    let refs: Vec<(&[u8], &[u8], &[u8])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    encode_paff_sequence(
        &PaffConfig {
            width: W as u32,
            frame_height: H as u32,
            qp: 26,
            p_fields,
            frame_picture_indices: frame_pictures,
            cross_parity_first_bottom: cross_parity,
            idr_frame_first,
            b_fields: false,
            b_temporal_direct: false,
            transform_8x8: false,
            long_term_anchor: false,
            mmco_unpair_first_top: false,
            b_implicit_weight: false,
            b_reference_fields: false,
        },
        &refs,
    )
}

/// Round-436 — B-field sequences: anchors at even display indices,
/// non-reference B/B field pairs at odd ones (coding order 0, 2, 1,
/// 4, 3, …), with the direct-mode derivation selected by
/// `b_temporal_direct`.
fn encode_b_fields(n_frames: usize, b_temporal_direct: bool) -> PaffEncoded {
    let frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> =
        (0..n_frames).map(make_interlaced_frame).collect();
    let refs: Vec<(&[u8], &[u8], &[u8])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    encode_paff_sequence(
        &PaffConfig {
            width: W as u32,
            frame_height: H as u32,
            qp: 26,
            p_fields: true,
            frame_picture_indices: Vec::new(),
            cross_parity_first_bottom: false,
            idr_frame_first: false,
            b_fields: true,
            b_temporal_direct,
            transform_8x8: false,
            long_term_anchor: false,
            mmco_unpair_first_top: false,
            b_implicit_weight: false,
            b_reference_fields: false,
        },
        &refs,
    )
}

fn encode_cfg(
    p_fields: bool,
    frame_pictures: Vec<usize>,
    n_frames: usize,
    cross_parity: bool,
) -> PaffEncoded {
    encode_cfg2(p_fields, frame_pictures, n_frames, cross_parity, false)
}

fn encode(p_fields: bool, frame_pictures: Vec<usize>, n_frames: usize) -> PaffEncoded {
    encode_cfg(p_fields, frame_pictures, n_frames, false)
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

fn assert_frames_match_recon(enc: &PaffEncoded, decoded: &[oxideav_core::VideoFrame], tag: &str) {
    if let Ok(dir) = std::env::var("OXIDEAV_DUMP_RECON") {
        for (i, (ry, ru, rv)) in enc.recon_frames.iter().enumerate() {
            let mut buf = ry.clone();
            buf.extend_from_slice(ru);
            buf.extend_from_slice(rv);
            let _ = std::fs::write(format!("{dir}/{tag}-recon-{i}.yuv"), &buf);
        }
        for (i, vf) in decoded.iter().enumerate() {
            let mut buf = Vec::new();
            for p in &vf.planes {
                buf.extend_from_slice(&p.data);
            }
            let _ = std::fs::write(format!("{dir}/{tag}-dec-{i}.yuv"), &buf);
        }
    }
    assert_eq!(
        decoded.len(),
        enc.recon_frames.len(),
        "{tag}: decoded frame count",
    );
    for (i, (vf, (ry, ru, rv))) in decoded.iter().zip(enc.recon_frames.iter()).enumerate() {
        assert_eq!(vf.planes.len(), 3, "{tag}: frame {i} plane count");
        assert_eq!(vf.planes[0].data.len(), ry.len(), "{tag}: frame {i} Y len");
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

#[test]
fn paff_i_fields_self_roundtrip_bit_exact() {
    let enc = encode(false, Vec::new(), 3);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "paff-i-fields");
}

#[test]
fn paff_p_fields_self_roundtrip_bit_exact() {
    let enc = encode(true, Vec::new(), 4);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "paff-p-fields");
}

#[test]
fn paff_mixed_frame_and_field_pictures_self_roundtrip_bit_exact() {
    // Frame 1 is a full-height I FRAME picture between two field pairs
    // — the picture-adaptive axis proper.
    let enc = encode(false, vec![1], 3);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "paff-mixed");
}

/// Cross-decoder check: a stock ffmpeg binary (black-box validator)
/// must reconstruct the exact same planes. Skips when ffmpeg is not
/// installed at the well-known Homebrew path.
fn ffmpeg_check(enc: &PaffEncoded, tag: &str) {
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
            let mismatches = ours.iter().zip(ff.iter()).filter(|(&a, &b)| a != b).count();
            assert_eq!(
                mismatches,
                0,
                "{tag}: frame {i} plane {name}: {mismatches}/{} samples differ vs ffmpeg",
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
fn paff_p_cross_parity_self_roundtrip_bit_exact() {
    // Frame 0's bottom field is a P field referencing the IDR TOP
    // field (opposite parity) — pins the §8.2.4.2.5 second-field init
    // when the same-parity sub-list is empty AND the §8.4.1.4
    // Table 8-10 chroma-MV adjustment (mvCLX[1] = mvLX[1] + 2).
    let enc = encode_cfg(true, Vec::new(), 3, true);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "paff-p-crossparity");
}

#[test]
fn paff_p_cross_parity_ffmpeg_bit_exact() {
    let enc = encode_cfg(true, Vec::new(), 3, true);
    ffmpeg_check(&enc, "paff-xpar-ffmpeg");
}

#[test]
fn paff_idr_frame_then_p_fields_self_roundtrip_bit_exact() {
    // IDR full-height FRAME picture, then P field pairs whose frame-1
    // references are the parity FIELDS of that stored frame — pins the
    // §8.2.4.2.5 frame-supplies-fields rule and the decoder's
    // field-view materialisation of a frame reference.
    let enc = encode_cfg2(true, Vec::new(), 3, false, true);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "paff-idr-frame-p-fields");
}

#[test]
fn paff_idr_frame_then_p_fields_ffmpeg_bit_exact() {
    let enc = encode_cfg2(true, Vec::new(), 3, false, true);
    ffmpeg_check(&enc, "paff-idrframe-ffmpeg");
}

#[test]
fn paff_i_fields_ffmpeg_bit_exact() {
    let enc = encode(false, Vec::new(), 3);
    ffmpeg_check(&enc, "paff-i-ffmpeg");
}

#[test]
fn paff_p_fields_ffmpeg_bit_exact() {
    let enc = encode(true, Vec::new(), 4);
    ffmpeg_check(&enc, "paff-p-ffmpeg");
}

#[test]
fn paff_mixed_ffmpeg_bit_exact() {
    let enc = encode(false, vec![1], 3);
    ffmpeg_check(&enc, "paff-mixed-ffmpeg");
}

#[test]
fn paff_b_fields_spatial_direct_self_roundtrip_bit_exact() {
    // Display 0 (I/I), 2 (P/P), 4 (P/P) anchors + non-reference B/B
    // pairs at displays 1 and 3, coded AFTER their following anchor —
    // pins the §8.2.4.2.4 + §8.2.4.2.5 B-field list initialisation
    // (L0[0] / L1[0] = same-parity fields of the enclosing anchors),
    // the §8.4.1.2.2 spatial direct derivation on field pictures, the
    // field CAVLC scan on every B residual, the §8.7 field deblock on
    // B fields, and the §C.4.4 POC-ordered output of non-reference
    // field pairs between their anchors.
    let enc = encode_b_fields(5, false);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "paff-b-fields-spatial");
}

#[test]
fn paff_b_fields_spatial_direct_ffmpeg_bit_exact() {
    let enc = encode_b_fields(5, false);
    ffmpeg_check(&enc, "paff-b-spatial-ffmpeg");
}

#[test]
fn paff_b_fields_temporal_direct_self_roundtrip_bit_exact() {
    // Same layout with `direct_spatial_mv_pred_flag = 0`: B_Skip /
    // B_Direct MVs come from §8.4.1.2.3 temporal direct — colPic is
    // the same-parity field of the following anchor (§8.4.1.2.1), its
    // motion grid is read in field coordinates, and the eq. 8-201/
    // 8-202 tb/td distances run on per-FIELD order counts.
    let enc = encode_b_fields(5, true);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "paff-b-fields-temporal");
}

#[test]
fn paff_b_fields_temporal_direct_ffmpeg_bit_exact() {
    let enc = encode_b_fields(5, true);
    ffmpeg_check(&enc, "paff-b-temporal-ffmpeg");
}

/// Round-436 — sequences with `EncoderConfig::transform_8x8` in the
/// field pictures: the CAVLC 8x8 luma coefficients must be emitted in
/// the §8.5.7 Table 8-14 FIELD scan (via the split-pipeline
/// pre-composed scan).
fn encode_8x8_fields(n_frames: usize, p_fields: bool, b_fields: bool) -> PaffEncoded {
    let frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> =
        (0..n_frames).map(make_interlaced_frame).collect();
    let refs: Vec<(&[u8], &[u8], &[u8])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    encode_paff_sequence(
        &PaffConfig {
            width: W as u32,
            frame_height: H as u32,
            // Low QP so the 8x8 trial genuinely wins on part of the
            // gradient content (denser coefficients).
            qp: 22,
            p_fields,
            frame_picture_indices: Vec::new(),
            cross_parity_first_bottom: false,
            idr_frame_first: false,
            b_fields,
            b_temporal_direct: false,
            transform_8x8: true,
            long_term_anchor: false,
            mmco_unpair_first_top: false,
            b_implicit_weight: false,
            b_reference_fields: false,
        },
        &refs,
    )
}

#[test]
fn paff_i_fields_transform_8x8_self_roundtrip_bit_exact() {
    // All-I field pairs under High profile with transform_8x8: the
    // 3-way intra RDO can pick I_8x8, whose CAVLC coefficients must
    // ride the §8.5.7 Table 8-14 FIELD scan (every MB of a field
    // picture is a field MB).
    let enc = encode_8x8_fields(3, false, false);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "paff-i-8x8t");
}

#[test]
fn paff_i_fields_transform_8x8_ffmpeg_bit_exact() {
    let enc = encode_8x8_fields(3, false, false);
    ffmpeg_check(&enc, "paff-i-8x8t-ffmpeg");
}

#[test]
fn paff_p_fields_transform_8x8_self_roundtrip_bit_exact() {
    // P field pairs with the inter 8x8-vs-4x4 residual trial active —
    // the §7.3.5 second-gate transform_size_8x8_flag rides the field
    // CAVLC path and 8x8 winners emit field-scanned coefficients.
    let enc = encode_8x8_fields(4, true, false);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "paff-p-8x8t");
}

#[test]
fn paff_p_fields_transform_8x8_ffmpeg_bit_exact() {
    let enc = encode_8x8_fields(4, true, false);
    ffmpeg_check(&enc, "paff-p-8x8t-ffmpeg");
}

#[test]
fn paff_b_fields_transform_8x8_self_roundtrip_bit_exact() {
    // B field pairs + transform_8x8: every coded B shape passes the
    // §7.3.5 second gate; 8x8 winners in B fields also emit the
    // Table 8-14 FIELD scan.
    let enc = encode_8x8_fields(5, true, true);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "paff-b-8x8t");
}

#[test]
fn paff_b_fields_transform_8x8_ffmpeg_bit_exact() {
    let enc = encode_8x8_fields(5, true, true);
    ffmpeg_check(&enc, "paff-b-8x8t-ffmpeg");
}

/// Round-436 — §8.2.5 field-marking axes.
fn encode_marking_axis(n_frames: usize, long_term: bool, unpair: bool) -> PaffEncoded {
    let frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> =
        (0..n_frames).map(make_interlaced_frame).collect();
    let refs: Vec<(&[u8], &[u8], &[u8])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    encode_paff_sequence(
        &PaffConfig {
            width: W as u32,
            frame_height: H as u32,
            qp: 26,
            p_fields: true,
            frame_picture_indices: Vec::new(),
            cross_parity_first_bottom: false,
            idr_frame_first: false,
            b_fields: false,
            b_temporal_direct: false,
            transform_8x8: false,
            long_term_anchor: long_term,
            mmco_unpair_first_top: unpair,
            b_implicit_weight: false,
            b_reference_fields: false,
        },
        &refs,
    )
}

#[test]
fn paff_long_term_anchor_fields_self_roundtrip_bit_exact() {
    // IDR top field long-term (long_term_reference_flag), bottom I
    // field completes the pair via MMCO 6, and every later P field
    // references the same-parity long-term anchor field through an
    // RPLM long_term_pic_num splice — while the short-term P pairs
    // keep sliding through the §8.2.5.3 window. 5 frames make the
    // long-term pair outlive several sliding-window evictions.
    let enc = encode_marking_axis(5, true, false);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "paff-lt-anchor");
}

#[test]
fn paff_long_term_anchor_fields_ffmpeg_bit_exact() {
    let enc = encode_marking_axis(5, true, false);
    ffmpeg_check(&enc, "paff-lt-anchor-ffmpeg");
}

#[test]
fn paff_mmco1_field_unmark_self_roundtrip_bit_exact() {
    // Frame 1's bottom P field carries a §8.2.5.4.1 field MMCO 1 that
    // unmarks the frame-1 TOP field; frame 2's top field must then
    // resolve RefPicList0[0] to FRAME 0's top field (the §8.2.4.2.5
    // "missing field is ignored" rule). A decoder that ignores the
    // per-field unmarking predicts frame 2's top field from the wrong
    // picture.
    let enc = encode_marking_axis(3, false, true);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "paff-mmco1-field");
}

#[test]
fn paff_mmco1_field_unmark_ffmpeg_bit_exact() {
    let enc = encode_marking_axis(3, false, true);
    ffmpeg_check(&enc, "paff-mmco1-ffmpeg");
}

#[test]
fn paff_b_fields_trailing_p_pair_self_roundtrip_bit_exact() {
    // Even frame count: the last display frame has no following anchor
    // and codes as a trailing P/P pair after the final B pair.
    let enc = encode_b_fields(4, false);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "paff-b-fields-trailing-p");
}

/// Diagnostic (env-gated): write the three PAFF streams + our-decoder /
/// encoder-recon YUV dumps to `OXIDEAV_PAFF_DUMP_DIR` for external
/// black-box comparison during bring-up.
#[test]
fn paff_dump_streams_for_diag() {
    let Some(dir) = std::env::var_os("OXIDEAV_PAFF_DUMP_DIR") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let dump = |name: &str, enc: &PaffEncoded| {
        std::fs::write(dir.join(format!("{name}.h264")), &enc.annex_b).unwrap();
        let mut recon = Vec::new();
        for (y, u, v) in &enc.recon_frames {
            recon.extend_from_slice(y);
            recon.extend_from_slice(u);
            recon.extend_from_slice(v);
        }
        std::fs::write(dir.join(format!("{name}-encrecon.yuv")), recon).unwrap();
        let mut ours = Vec::new();
        for vf in decode_ours(&enc.annex_b) {
            for p in &vf.planes {
                ours.extend_from_slice(&p.data);
            }
        }
        std::fs::write(dir.join(format!("{name}-oursdec.yuv")), ours).unwrap();
    };
    for (name, p_fields, frame_pics, n, xpar, idr_frame) in [
        ("paff-i", false, Vec::new(), 3usize, false, false),
        ("paff-p", true, Vec::new(), 4, false, false),
        ("paff-mixed", false, vec![1], 3, false, false),
        ("paff-p-crossparity", true, Vec::new(), 3, true, false),
        ("paff-idr-frame-p-fields", true, Vec::new(), 3, false, true),
    ] {
        dump(name, &encode_cfg2(p_fields, frame_pics, n, xpar, idr_frame));
    }
    // Round-436 stream classes.
    dump("paff-b-fields-spatial", &encode_b_fields(5, false));
    dump("paff-b-fields-temporal", &encode_b_fields(5, true));
    dump("paff-i-8x8t", &encode_8x8_fields(3, false, false));
    dump("paff-p-8x8t", &encode_8x8_fields(4, true, false));
    dump("paff-b-8x8t", &encode_8x8_fields(5, true, true));
    dump("paff-lt-anchor", &encode_marking_axis(5, true, false));
    dump("paff-mmco1-field", &encode_marking_axis(3, false, true));
}

/// Round-440 — the two new B-field stream axes: §8.4.2.3.3 implicit
/// weighted prediction (stride-3 anchors, unequal POC distances) and
/// B REFERENCE fields (stride-4 anchors with a reference B pair
/// midway).
fn encode_b_ext(n_frames: usize, temporal: bool, implicit: bool, ref_b: bool) -> PaffEncoded {
    let frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> =
        (0..n_frames).map(make_interlaced_frame).collect();
    let refs: Vec<(&[u8], &[u8], &[u8])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    encode_paff_sequence(
        &PaffConfig {
            width: W as u32,
            frame_height: H as u32,
            qp: 26,
            p_fields: true,
            frame_picture_indices: Vec::new(),
            cross_parity_first_bottom: false,
            idr_frame_first: false,
            b_fields: true,
            b_temporal_direct: temporal,
            transform_8x8: false,
            long_term_anchor: false,
            mmco_unpair_first_top: false,
            b_implicit_weight: implicit,
            b_reference_fields: ref_b,
        },
        &refs,
    )
}

#[test]
fn paff_b_fields_implicit_weight_self_roundtrip_bit_exact() {
    // Stride-3 layout (anchors at displays 0, 3, 6): the PPS signals
    // `weighted_bipred_idc = 2` and the two non-reference B pairs
    // between anchors sit at UNEQUAL per-field POC distances, so every
    // bipred / direct-Bi macroblock combines its predictions with the
    // §8.4.2.3.3 POC-derived weights — (w0, w1) = (43, 21) for the
    // first pair, (22, 42) for the second, at logWD = 5 on the
    // fields' own §8.2.1 order counts — on luma AND chroma. A decoder
    // that ignored implicit weighting (or ran it on frame-level POCs)
    // would mispredict every bipred macroblock.
    let enc = encode_b_ext(7, false, true, false);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "paff-b-implicit");
}

#[test]
fn paff_b_fields_implicit_weight_temporal_self_roundtrip_bit_exact() {
    // Same layout with §8.4.1.2.3 temporal direct: the direct-derived
    // bipred macroblocks ALSO combine through the implicit weights.
    let enc = encode_b_ext(7, true, true, false);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "paff-b-implicit-temporal");
}

#[test]
fn paff_b_fields_implicit_weight_ffmpeg_bit_exact() {
    let enc = encode_b_ext(7, false, true, false);
    ffmpeg_check(&enc, "paff-b-implicit-ffmpeg");
}

#[test]
fn paff_b_fields_implicit_weight_temporal_ffmpeg_bit_exact() {
    let enc = encode_b_ext(7, true, true, false);
    ffmpeg_check(&enc, "paff-b-implicit-temporal-ffmpeg");
}

#[test]
fn paff_b_reference_fields_self_roundtrip_bit_exact() {
    // Stride-4 layout 0, 4, 2ref, 1, 3, 8, 6ref, 5, 7: the midway B
    // pair is COD as a reference (`nal_ref_idc = 2`) and stored
    // through the §8.2.5.3 sliding window as a complementary
    // reference field pair; the non-reference B pairs then find the
    // B REFERENCE fields at `RefPicList1[0]` (display d−3 — making a
    // coded B field the §8.4.1.2.1 colPic of the direct derivation)
    // and `RefPicList0[0]` (display d−1) per the §8.2.4.2.4 +
    // §8.2.4.2.5 per-field POC ordering.
    let enc = encode_b_ext(9, false, false, true);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "paff-b-ref-fields");
}

#[test]
fn paff_b_reference_fields_temporal_self_roundtrip_bit_exact() {
    // Temporal-direct variant: the B pair before the reference-B runs
    // §8.4.1.2.3 with colPic = a coded B FIELD (reading its stored L0
    // motion — the reference-B's mode decision is restricted to
    // L0/Bi/intra so the co-located read matches the encoder model).
    let enc = encode_b_ext(9, true, false, true);
    let decoded = decode_ours(&enc.annex_b);
    assert_frames_match_recon(&enc, &decoded, "paff-b-ref-fields-temporal");
}

#[test]
fn paff_b_reference_fields_ffmpeg_bit_exact() {
    let enc = encode_b_ext(9, false, false, true);
    ffmpeg_check(&enc, "paff-b-ref-fields-ffmpeg");
}

#[test]
fn paff_b_reference_fields_temporal_ffmpeg_bit_exact() {
    let enc = encode_b_ext(9, true, false, true);
    ffmpeg_check(&enc, "paff-b-ref-fields-temporal-ffmpeg");
}
