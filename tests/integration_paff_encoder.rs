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

fn encode_cfg(
    p_fields: bool,
    frame_pictures: Vec<usize>,
    n_frames: usize,
    cross_parity: bool,
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
        },
        &refs,
    )
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
    let _ = std::fs::remove_dir_all(&dir);
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
    for (name, p_fields, frame_pics, n, xpar) in [
        ("paff-i", false, Vec::new(), 3usize, false),
        ("paff-p", true, Vec::new(), 4, false),
        ("paff-mixed", false, vec![1], 3, false),
        ("paff-p-crossparity", true, Vec::new(), 3, true),
    ] {
        let enc = encode_cfg(p_fields, frame_pics, n, xpar);
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
    }
}
