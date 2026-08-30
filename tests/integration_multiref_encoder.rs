//! Round-453 — multi-reference P coding with long-term references,
//! MMCO and RPLM (`encoder::multiref`).
//!
//! Every stream decodes bit-exactly in our own decoder against the
//! encoder's recon and byte-exactly in the black-box reference
//! decoder; the content is built so that older / long-term references
//! win the per-MB election, which pins the §7.3.5.1 `ref_idx_l0`
//! te(v) coding, the §8.4.1.3 refIdx-aware MV prediction, the
//! §8.2.4.2.1 list order, §8.2.5 marking and §8.2.4.3 modification.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::multiref::{encode_multiref_sequence, MultiRefConfig, MultiRefEncoded};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 96;
const H: usize = 64;

fn base_texture(phase: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let x = (i + phase) % W;
            let tex = ((x * 7 + j * 3) % 53) as i32 + (((x / 8 + j / 8) % 2) as i32) * 30;
            y[j * W + i] = (70 + tex).clamp(0, 255) as u8;
        }
    }
    let cw = W / 2;
    let ch = H / 2;
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            u[j * cw + i] = (100 + ((i + phase / 2 + j) % 19)) as u8;
            v[j * cw + i] = (150 + ((i * 2 + j + phase / 2) % 11)) as u8;
        }
    }
    (y, u, v)
}

/// Overlay a bright moving block on `f`.
fn with_blob(mut f: (Vec<u8>, Vec<u8>, Vec<u8>), k: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let x0 = (k * 13) % (W - 24);
    let y0 = (k * 7) % (H - 20);
    for j in y0..y0 + 20 {
        for i in x0..x0 + 24 {
            f.0[j * W + i] = 235 - ((i + j) % 9) as u8;
        }
    }
    f
}

fn run(cfg: MultiRefConfig, frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)]) -> MultiRefEncoded {
    let refs: Vec<(&[u8], &[u8], &[u8])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    encode_multiref_sequence(&cfg, &refs)
}

fn cfg(num_ref_frames: u32) -> MultiRefConfig {
    MultiRefConfig {
        width: W as u32,
        height: H as u32,
        qp: 26,
        num_ref_frames,
        long_term_idr: false,
        mmco_long_term_at: None,
        mmco_unmark_at: None,
        rplm_long_term_first_at: None,
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

fn assert_bit_exact(enc: &MultiRefEncoded, tag: &str) {
    let decoded = decode_ours(&enc.annex_b);
    assert_eq!(decoded.len(), enc.recon_frames.len(), "{tag}: frame count");
    for (i, (vf, (ry, ru, rv))) in decoded.iter().zip(enc.recon_frames.iter()).enumerate() {
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
fn ffmpeg_check(enc: &MultiRefEncoded, tag: &str) {
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
    let fb = W * H * 3 / 2;
    assert_eq!(
        yuv.len(),
        fb * enc.recon_frames.len(),
        "{tag}: ffmpeg output size"
    );
    for (i, (ry, ru, rv)) in enc.recon_frames.iter().enumerate() {
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

#[test]
fn multiref_three_active_refs_periodic_content_elects_older_reference() {
    // Frame k repeats frame k-3 exactly (three-phase texture), so with
    // three active references ref_idx 2 predicts losslessly and wins
    // over the closer, mismatched references.
    let frames: Vec<_> = (0..7).map(|k| base_texture((k % 3) * 11)).collect();
    let enc = run(cfg(3), &frames);
    eprintln!("hist {:?} lists {:?}", enc.ref_idx_hist, enc.lists);
    assert_eq!(enc.lists[2].len(), 3);
    assert_eq!(
        enc.lists[2],
        vec![(2, false), (1, false), (0, false)],
        "§8.2.4.2.1: short-term refs by descending PicNum",
    );
    assert!(
        enc.ref_idx_hist[2] > 0,
        "ref_idx 2 never elected: {:?}",
        enc.ref_idx_hist
    );
    assert!(
        enc.ref_idx_hist[1] > 0,
        "ref_idx 1 never elected: {:?}",
        enc.ref_idx_hist
    );
    assert_bit_exact(&enc, "multiref-3");
    ffmpeg_check(&enc, "multiref-3");
}

#[test]
fn multiref_long_term_idr_anchor_survives_sliding_window() {
    // The IDR is marked long-term (§7.4.3.3); every later frame carries
    // a moving blob over the IDR's texture, so the long-term anchor —
    // which the sliding window never evicts — is the best reference
    // for the uncovered background.
    let base = base_texture(0);
    let frames: Vec<_> = (0..6)
        .map(|k| {
            if k == 0 {
                base.clone()
            } else {
                with_blob(base.clone(), k)
            }
        })
        .collect();
    let mut c = cfg(2);
    c.long_term_idr = true;
    let enc = run(c, &frames);
    eprintln!("hist {:?} lists {:?}", enc.ref_idx_hist, enc.lists);
    // From frame 2 on: [previous short-term, long-term IDR].
    assert_eq!(enc.lists[4], vec![(4, false), (0, true)]);
    assert!(enc.long_term_hits > 0, "no MB referenced the long-term IDR");
    assert_bit_exact(&enc, "multiref-lt-idr");
    ffmpeg_check(&enc, "multiref-lt-idr");
}

#[test]
fn multiref_mmco_marks_and_unmarks_long_term_picture() {
    // Frame 1 becomes long-term via MMCO 4 + MMCO 6 (with an MMCO 1
    // eviction keeping the buffer within max_num_ref_frames); frame 5
    // unmarks it via MMCO 2. Frames 2..=4 see [short, long(1)], frame
    // 5 falls back to short-term only.
    let base = base_texture(5);
    let frames: Vec<_> = (0..7)
        .map(|k| {
            if k == 1 {
                base.clone()
            } else {
                with_blob(base.clone(), k + 3)
            }
        })
        .collect();
    let mut c = cfg(2);
    c.mmco_long_term_at = Some(1);
    c.mmco_unmark_at = Some(5);
    let enc = run(c, &frames);
    eprintln!("hist {:?} lists {:?}", enc.ref_idx_hist, enc.lists);
    assert_eq!(enc.lists[0], vec![(0, false)]);
    // Frame 1's MMCO applies after frame 1 is decoded: frames 2..=5
    // list [previous short-term, long-term frame 1] — including
    // frame 5 itself, whose MMCO 2 only takes effect afterwards.
    assert_eq!(enc.lists[3], vec![(3, false), (1, true)]);
    assert_eq!(enc.lists[4], vec![(4, false), (1, true)]);
    assert_eq!(
        enc.lists[5],
        vec![(5, false), (4, false)],
        "after MMCO 2 the list holds no long-term entry",
    );
    assert!(enc.long_term_hits > 0);
    assert_bit_exact(&enc, "multiref-mmco");
    ffmpeg_check(&enc, "multiref-mmco");
}

#[test]
fn multiref_rplm_moves_long_term_to_index_zero() {
    // Long-term IDR; frame 3's slice reorders the long-term picture to
    // RefPicList0[0] (§7.3.3.1 idc 2), so P_Skip and ref_idx 0 hit the
    // anchor there.
    let base = base_texture(2);
    let frames: Vec<_> = (0..5)
        .map(|k| {
            if k == 0 {
                base.clone()
            } else {
                with_blob(base.clone(), k)
            }
        })
        .collect();
    let mut c = cfg(2);
    c.long_term_idr = true;
    c.rplm_long_term_first_at = Some(3);
    let enc = run(c, &frames);
    eprintln!("hist {:?} lists {:?}", enc.ref_idx_hist, enc.lists);
    assert_eq!(enc.lists[2], vec![(0, true), (2, false)]);
    assert_eq!(enc.lists[3], vec![(3, false), (0, true)]);
    assert_bit_exact(&enc, "multiref-rplm");
    ffmpeg_check(&enc, "multiref-rplm");
}
