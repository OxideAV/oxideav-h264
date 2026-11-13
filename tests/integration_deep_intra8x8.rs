//! Round-488 — Intra_8x8 in the high-bit-depth encoder
//! (`DeepSession::with_intra_8x8`): §8.3.2 prediction from the
//! §8.3.2.2.1 filtered reference samples at 8 / 10 / 12 / 14 bit,
//! §8.3.2.1 predicted-mode coding (against Intra_4x4, Intra_8x8,
//! Intra_16x16 and inter neighbours), the 8x8 residual at QP′ or the
//! §8.5.15 V / H DPCM under transform bypass, CAVLC and CABAC, at
//! 4:0:0 / 4:2:0 / 4:2:2 — in I pictures and as an intra alternative
//! inside P pictures (where Intra_4x4 is offered too).
//!
//! Every stream decodes bit-exactly in our decoder (against the
//! session's own reconstruction) and byte-exactly in a black-box
//! decoder; Intra_8x8 must actually be chosen.

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_h264::encoder::deep::{DeepConfig, DeepPictureKind, DeepSession};

fn chroma_dims(cf: u32, w: usize, h: usize) -> (usize, usize) {
    match cf {
        0 => (0, 0),
        1 => (w / 2, h / 2),
        _ => (w / 2, h),
    }
}

/// Picture with regions that favour each intra size: a smooth ramp
/// (Intra_16x16), 8-sample diagonal structure (Intra_8x8) and fine
/// texture (Intra_4x4); frame `n` shifts it and adds new content at
/// the right edge so P pictures need intra macroblocks.
fn source(cf: u32, bd: u32, w: usize, h: usize, n: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let max = (1u32 << bd) - 1;
    let sh = bd - 8;
    let mut y = vec![0u16; w * h];
    for j in 0..h {
        for i in 0..w {
            let x = i + 3 * n;
            let v = match (x / 24) % 3 {
                0 => 60 + (x + 2 * j) % 64,
                1 => {
                    if ((x + j) / 5).is_multiple_of(2) {
                        50
                    } else {
                        170
                    }
                }
                _ => 90 + ((x * 7 + j * 13) % 11) * 9 + if (x ^ j) & 4 == 0 { 30 } else { 0 },
            };
            // Uncovered content in later frames.
            let v = if n > 0 && i + 20 > w && (j / 6 + n).is_multiple_of(2) {
                230 - v / 2
            } else {
                v
            };
            y[j * w + i] = ((v as u32) << sh).min(max) as u16;
        }
    }
    let (cw, ch) = chroma_dims(cf, w, h);
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

fn bytes(s: &[u16], bd: u32) -> Vec<u8> {
    if bd > 8 {
        s.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else {
        s.iter().map(|&v| v as u8).collect()
    }
}

fn decode_ours(stream: &[u8]) -> Vec<Vec<u8>> {
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
    out
}

/// Black-box decode to raw planes in the stream's native layout
/// (`None` when the binary is absent).
fn reference_decode(stream: &[u8], tag: &str) -> Option<Vec<u8>> {
    let refbin = [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    .iter()
    .map(std::path::PathBuf::from)
    .find(|p| p.exists());
    let Some(refbin) = refbin else {
        eprintln!("skip {tag}: black-box decoder binary not present");
        return None;
    };
    let dir = std::env::temp_dir().join(format!("oxideav-h264-i8x8-{tag}-{}", std::process::id()));
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
        .expect("spawn black-box decoder");
    assert!(status.success(), "{tag}: black-box decoder failed");
    let raw = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some(raw)
}

struct Case {
    cf: u32,
    bd: u32,
    w: usize,
    h: usize,
    cabac: bool,
    lossless: bool,
    qp: i32,
    /// Pictures after the IDR are P pictures (else I pictures).
    p_frames: bool,
    intra_4x4: bool,
    /// Require at least one Intra_8x8 macroblock (content / QP where
    /// the 8x8 size is expected to win somewhere).
    expect_8x8: bool,
}

struct Outcome {
    bytes: usize,
    /// Luma PSNR over the three pictures (99 when exact).
    psnr: f64,
    i8x8_mbs: usize,
    i4x4_mbs: usize,
    intra_mbs_in_p: usize,
}

/// Encode three pictures with Intra_8x8 offered; gate the stream in
/// both decoders.
fn run(c: &Case) -> Outcome {
    run_with(c, true)
}

/// [`run`] with the Intra_8x8 candidate on or off (the PPS carries
/// `transform_8x8_mode_flag` either way).
fn run_with(c: &Case, intra_8x8: bool) -> Outcome {
    let tag = format!(
        "cf{}-bd{}-{}x{}-{}-{}-qp{}-{}-i4{}-i8{intra_8x8}",
        c.cf,
        c.bd,
        c.w,
        c.h,
        if c.cabac { "cabac" } else { "cavlc" },
        if c.lossless { "ll" } else { "lossy" },
        c.qp,
        if c.p_frames { "ipp" } else { "iii" },
        c.intra_4x4,
    );
    let cfg = DeepConfig::new(
        c.w as u32,
        c.h as u32,
        c.bd,
        c.bd,
        c.cf,
        if c.lossless {
            -6 * (c.bd as i32 - 8)
        } else {
            c.qp
        },
    )
    .with_p_frames(c.p_frames)
    .with_intra_in_p(true)
    .with_deblock(true)
    .with_lossless(c.lossless)
    .with_lossless_interop(true)
    .with_cabac(c.cabac)
    .with_transform_8x8(true);
    let frames: Vec<_> = (0..3).map(|n| source(c.cf, c.bd, c.w, c.h, n)).collect();
    let mut s = DeepSession::new(cfg)
        .with_intra_4x4(c.intra_4x4)
        .with_intra_8x8(intra_8x8);
    let mut stream = s.parameter_sets_annex_b();
    let mut recons = Vec::new();
    for (k, (y, u, v)) in frames.iter().enumerate() {
        let kind = if k == 0 {
            DeepPictureKind::Idr
        } else if c.p_frames {
            DeepPictureKind::P
        } else {
            DeepPictureKind::I
        };
        let pic = s.encode_picture(y, u, v, kind, k as u32, 2 * k as i32, 0);
        stream.extend_from_slice(&pic.nal);
        if c.lossless {
            assert!(
                pic.recon.0 == *y && pic.recon.1 == *u && pic.recon.2 == *v,
                "{tag}: picture {k} lossless reconstruction"
            );
        }
        recons.push(pic.recon);
    }
    if c.expect_8x8 && intra_8x8 {
        assert!(s.i8x8_mbs() > 0, "{tag}: no Intra_8x8 macroblock chosen");
    }
    let max = ((1u64 << c.bd) - 1) as f64;
    let (mut sse, mut count) = (0f64, 0usize);
    for ((sy, _, _), (ry, _, _)) in frames.iter().zip(&recons) {
        sse += sy
            .iter()
            .zip(ry)
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum::<f64>();
        count += sy.len();
    }
    let psnr = if sse == 0.0 {
        99.0
    } else {
        10.0 * (max * max * count as f64 / sse).log10()
    };

    let ours = decode_ours(&stream);
    assert_eq!(ours.len(), 3, "{tag}: picture count");
    for (k, (ry, ru, rv)) in recons.iter().enumerate() {
        let exp = [bytes(ry, c.bd), bytes(ru, c.bd), bytes(rv, c.bd)].concat();
        assert!(
            ours[k] == exp,
            "{tag}: picture {k} not bit-exact in our decoder"
        );
    }
    if let Some(raw) = reference_decode(&stream, &tag) {
        let bps = if c.bd > 8 { 2 } else { 1 };
        // A 4:0:0 stream comes back padded out to 4:2:0.
        let extra = if c.cf == 0 {
            2 * c.w.div_ceil(2) * c.h.div_ceil(2) * bps
        } else {
            0
        };
        let len = ours[0].len();
        assert_eq!(raw.len(), 3 * (len + extra), "{tag}: black-box output size");
        for (k, o) in ours.iter().enumerate() {
            let got = &raw[k * (len + extra)..][..len];
            assert!(
                got == &o[..],
                "{tag}: picture {k} differs in the black-box decoder"
            );
        }
    }
    Outcome {
        bytes: stream.len(),
        psnr,
        i8x8_mbs: s.i8x8_mbs(),
        i4x4_mbs: s.i4x4_mbs(),
        intra_mbs_in_p: s.intra_mbs_in_p(),
    }
}

const LAYOUTS: &[(u32, u32)] = &[
    (1, 8),
    (1, 10),
    (2, 8),
    (2, 10),
    (0, 8),
    (0, 10),
    (1, 12),
    (2, 12),
    (0, 12),
    (1, 14),
    (0, 14),
];

#[test]
fn intra_8x8_i_pictures_lossy_matrix() {
    for &(cf, bd) in LAYOUTS {
        for cabac in [false, true] {
            for intra_4x4 in [false, true] {
                let o = run(&Case {
                    cf,
                    bd,
                    w: 96,
                    h: 48,
                    cabac,
                    lossless: false,
                    qp: 24,
                    p_frames: false,
                    intra_4x4,
                    expect_8x8: true,
                });
                if intra_4x4 {
                    // The three intra sizes coexist in one picture, so
                    // every §8.3.1.1 / §8.3.2.1 neighbour pairing occurs.
                    assert!(
                        o.i4x4_mbs > 0,
                        "cf{cf} bd{bd}: no Intra_4x4 next to Intra_8x8"
                    );
                }
            }
        }
    }
}

#[test]
fn intra_8x8_inside_p_pictures() {
    for &(cf, bd) in LAYOUTS {
        for cabac in [false, true] {
            let o = run(&Case {
                cf,
                bd,
                w: 96,
                h: 48,
                cabac,
                lossless: false,
                qp: 24,
                p_frames: true,
                intra_4x4: true,
                expect_8x8: true,
            });
            assert!(
                o.intra_mbs_in_p > 0,
                "cf{cf} bd{bd} cabac={cabac}: no intra macroblock inside the P pictures"
            );
        }
    }
}

#[test]
fn intra_8x8_lossless_exact_with_dpcm() {
    for &(cf, bd) in &[(1u32, 8u32), (2, 10), (0, 12), (1, 14), (0, 8)] {
        for cabac in [false, true] {
            for p_frames in [false, true] {
                run(&Case {
                    cf,
                    bd,
                    w: 64,
                    h: 48,
                    cabac,
                    lossless: true,
                    qp: 0,
                    p_frames,
                    intra_4x4: true,
                    expect_8x8: false,
                });
            }
        }
    }
}

#[test]
fn intra_8x8_cropped_sizes_and_qp_range() {
    for &(cf, bd, w, h) in &[
        (1u32, 10u32, 70usize, 50usize),
        (2, 8, 66, 49),
        (0, 12, 49, 35),
    ] {
        for qp in [10, 34, 46] {
            for cabac in [false, true] {
                run(&Case {
                    cf,
                    bd,
                    w,
                    h,
                    cabac,
                    lossless: false,
                    qp,
                    p_frames: true,
                    intra_4x4: true,
                    expect_8x8: false,
                });
            }
        }
    }
}

/// Rate / quality with and without the Intra_8x8 candidate
/// (informational table with `--nocapture`).
#[test]
fn intra_8x8_rate_table() {
    for &(cf, bd) in &[(1u32, 8u32), (1, 10), (0, 8)] {
        for cabac in [false, true] {
            for qp in [22, 30, 38] {
                let case = Case {
                    cf,
                    bd,
                    w: 96,
                    h: 48,
                    cabac,
                    lossless: false,
                    qp,
                    p_frames: false,
                    intra_4x4: true,
                    expect_8x8: false,
                };
                let with = run_with(&case, true);
                let without = run_with(&case, false);
                eprintln!(
                    "I8X8 cf{cf} bd{bd} {} QP {qp}: {} B / {:.2} dB with ({} Intra_8x8 + {} Intra_4x4 of {} MBs), {} B / {:.2} dB without",
                    if cabac { "CABAC" } else { "CAVLC" },
                    with.bytes,
                    with.psnr,
                    with.i8x8_mbs,
                    with.i4x4_mbs,
                    3 * 6 * 3,
                    without.bytes,
                    without.psnr,
                );
            }
        }
    }
}
