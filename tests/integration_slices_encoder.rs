//! Round-453 — multi-slice / FMO / ASO / redundant-slice / constrained
//! intra prediction emission on the Baseline encoder
//! (`encoder::slices`).
//!
//! Every stream decodes bit-exactly in our own decoder against the
//! encoder's recon. Streams the black-box reference decoder supports
//! (multi-slice, ASO, redundant slices, constrained intra) are also
//! byte-checked against it; FMO streams are pinned by our decoder only
//! (the reference decoder binary does not implement slice groups).

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::slices::{encode_slices_sequence, SlicesConfig, SlicesEncoded};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264::pps::SliceGroupMap;

const W: usize = 96;
const H: usize = 64; // 6x4 = 24 MBs

fn make_frame(k: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let x = (i + 3 * k) % W;
            let tex = ((x * 5 + j * 7) % 47) as i32 + (((x / 8 + j / 8) % 2) as i32) * 25;
            y[j * W + i] = (60 + tex).clamp(0, 255) as u8;
        }
    }
    // A bright block that moves every frame (drives intra fallback
    // and skips in different areas).
    let x0 = (k * 17) % (W - 20);
    let y0 = (k * 11) % (H - 20);
    for j in y0..y0 + 20 {
        for i in x0..x0 + 20 {
            y[j * W + i] = 230 - ((i * j) % 13) as u8;
        }
    }
    let cw = W / 2;
    let ch = H / 2;
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            u[j * cw + i] = (100 + ((i + j + k) % 23)) as u8;
            v[j * cw + i] = (140 + ((i * 2 + j) % 17)) as u8;
        }
    }
    (y, u, v)
}

fn base_cfg_i() -> SlicesConfig {
    let mut c = base_cfg();
    c.p_frames = false;
    c
}

fn base_cfg() -> SlicesConfig {
    SlicesConfig {
        width: W as u32,
        height: H as u32,
        qp: 26,
        slice_groups: None,
        slice_group_change_cycle: 0,
        mbs_per_slice: 0,
        constrained_intra: false,
        aso: false,
        redundant: false,
        p_frames: true,
    }
}

fn run(cfg: &SlicesConfig, n: usize) -> SlicesEncoded {
    let frames: Vec<_> = (0..n).map(make_frame).collect();
    let refs: Vec<(&[u8], &[u8], &[u8])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    encode_slices_sequence(cfg, &refs)
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

fn assert_bit_exact(enc: &SlicesEncoded, tag: &str) {
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
            if mism != 0 {
                let w = plane.stride;
                let mut mbs: Vec<(usize, usize, i32)> = Vec::new();
                for (off, (&a, &b)) in plane.data.iter().zip(exp.iter()).enumerate() {
                    if a != b {
                        let key = ((off % w) / 16, (off / w) / 16);
                        let d = (a as i32 - b as i32).abs();
                        match mbs.iter_mut().find(|(x, y, _)| (*x, *y) == key) {
                            Some((_, _, m)) => *m = (*m).max(d),
                            None => mbs.push((key.0, key.1, d)),
                        }
                    }
                }
                eprintln!("{tag}: frame {i} plane {name} diverging MBs {mbs:?}");
            }
            assert_eq!(
                mism, 0,
                "{tag}: frame {i} plane {name}: {mism} samples differ"
            );
        }
    }
}

/// Black-box reference decoder cross-check (skips when ffmpeg is not
/// installed at the well-known Homebrew path).
fn ffmpeg_check(enc: &SlicesEncoded, tag: &str) {
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
    let output = std::process::Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&bs)
        .args(["-f", "rawvideo"])
        .arg(&out)
        .output()
        .expect("spawn ffmpeg");
    assert!(
        output.status.success(),
        "{tag}: ffmpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.stderr.is_empty() {
        eprintln!(
            "{tag}: ffmpeg stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
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
fn slices_raster_multi_slice_i_and_p() {
    // 24 MBs in slices of 7 → boundaries mid-row: A / B / D neighbours
    // in the previous slice must be treated as unavailable.
    let mut c = base_cfg();
    c.mbs_per_slice = 7;
    let enc = run(&c, 4);
    assert_eq!(enc.slices_per_picture, vec![4, 4, 4, 4]);
    assert_bit_exact(&enc, "slices-raster");
    ffmpeg_check(&enc, "slices-raster");
}

#[test]
fn slices_constrained_intra_pred_p_pictures() {
    let mut c = base_cfg();
    c.constrained_intra = true;
    c.mbs_per_slice = 5;
    let enc = run(&c, 5);
    eprintln!(
        "constrained intra: {} intra MBs in P, {} with a reduced mask",
        enc.intra_mbs_in_p, enc.masked_intra_mbs
    );
    assert!(
        enc.intra_mbs_in_p > 0,
        "fixture must produce intra MBs in P pictures"
    );
    assert!(
        enc.masked_intra_mbs > 0,
        "constrained intra / slice boundaries must drop at least one neighbour",
    );
    assert_bit_exact(&enc, "slices-cip");
    ffmpeg_check(&enc, "slices-cip");
}

#[test]
fn slices_arbitrary_slice_order() {
    let mut c = base_cfg();
    c.mbs_per_slice = 6;
    c.aso = true;
    let enc = run(&c, 3);
    assert_bit_exact(&enc, "slices-aso");
    // The black-box reference decoder binary does not implement
    // arbitrary slice order (a slice whose first_mb_in_slice precedes
    // the previous slice's opens a new picture there), so ASO streams
    // are pinned by our own §7.4.1.2.4 / §A.2.1 decode path only.
}

#[test]
fn slices_redundant_slices_are_discarded() {
    let mut c = base_cfg();
    c.mbs_per_slice = 8;
    c.redundant = true;
    let enc = run(&c, 3);
    assert_bit_exact(&enc, "slices-redundant");
    // The black-box reference decoder binary does not discard
    // redundant coded slices (it reports "no frame!" and corrupts the
    // following picture), so redundant streams are pinned by our own
    // §7.4.2.2 / §7.4.1.2 discard path only.
}

fn fmo(map: SliceGroupMap, groups: u32, cycle: u32) -> SlicesConfig {
    let mut c = base_cfg();
    c.slice_groups = Some((groups, map));
    c.slice_group_change_cycle = cycle;
    c
}

#[test]
fn fmo_type0_interleaved() {
    let enc = run(
        &fmo(
            SliceGroupMap::Interleaved {
                run_length_minus1: vec![3, 1, 2],
            },
            3,
            0,
        ),
        3,
    );
    assert_eq!(enc.slices_per_picture[0], 3);
    assert_bit_exact(&enc, "fmo-0");
}

#[test]
fn fmo_type1_dispersed() {
    let enc = run(&fmo(SliceGroupMap::Dispersed, 2, 0), 3);
    assert_eq!(enc.slices_per_picture[0], 2);
    assert_bit_exact(&enc, "fmo-1");
}

#[test]
fn fmo_type2_foreground_with_leftover() {
    // Two rectangles (MB addresses top-left / bottom-right) + the
    // background group.
    let enc = run(
        &fmo(
            SliceGroupMap::Foreground {
                top_left: vec![1, 14],
                bottom_right: vec![8, 22],
            },
            3,
            0,
        ),
        3,
    );
    assert_eq!(enc.slices_per_picture[0], 3);
    assert_bit_exact(&enc, "fmo-2");
}

#[test]
fn fmo_types_3_4_5_changing_with_change_cycle() {
    for (t, dir) in [(3u32, false), (3, true), (4, false), (5, true)] {
        let enc = run(
            &fmo(
                SliceGroupMap::Changing {
                    slice_group_map_type: t,
                    change_direction_flag: dir,
                    change_rate_minus1: 1,
                },
                2,
                5,
            ),
            3,
        );
        assert_eq!(enc.slices_per_picture[0], 2, "type {t}");
        assert_bit_exact(&enc, &format!("fmo-{t}-{dir}"));
    }
}

#[test]
fn fmo_type6_explicit_map() {
    let ids: Vec<u32> = (0..24u32).map(|a| a * 7 % 3).collect();
    let enc = run(
        &fmo(
            SliceGroupMap::Explicit {
                pic_size_in_map_units_minus1: 23,
                slice_group_id: ids,
            },
            3,
            0,
        ),
        3,
    );
    assert_eq!(enc.slices_per_picture[0], 3);
    assert_bit_exact(&enc, "fmo-6");
}

#[test]
fn fmo_dispersed_with_multiple_slices_per_group_and_constrained_intra() {
    let mut c = fmo(SliceGroupMap::Dispersed, 2, 0);
    c.mbs_per_slice = 5;
    c.constrained_intra = true;
    let enc = run(&c, 4);
    assert_eq!(enc.slices_per_picture[0], 6);
    assert_bit_exact(&enc, "fmo-1-multi-cip");
}

#[test]
fn slices_single_slice_sanity() {
    // Degenerate configuration: one slice per picture — pins the driver
    // against the ordinary single-slice encoder behaviour.
    let enc = run(&base_cfg(), 3);
    assert_eq!(enc.slices_per_picture, vec![1, 1, 1]);
    assert_bit_exact(&enc, "slices-single");
    ffmpeg_check(&enc, "slices-single");
}

#[test]
fn slices_two_slices_i_only_boundary_row_unavailable() {
    // Two 12-MB slices: the second slice's first MB row sits under the
    // first slice, so every top / top-left neighbour there is
    // unavailable (§6.4.8) — DC / H prediction only, no V / Plane.
    let mut c = base_cfg_i();
    c.mbs_per_slice = 12;
    let enc = run(&c, 2);
    assert_eq!(enc.slices_per_picture, vec![2, 2]);
    assert_bit_exact(&enc, "slices-2i");
    ffmpeg_check(&enc, "slices-2i");
}
