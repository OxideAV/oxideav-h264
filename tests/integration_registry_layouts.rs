//! Round-475 — the registry encoder codes every input layout the
//! High-family encoder (`encoder::deep`) can: 8-bit 4:2:2 / 4:4:4,
//! 10- / 12-bit 4:2:0 / 4:2:2 / 4:4:4, grey (as 4:2:0 with constant
//! chroma), under the §A.2.4 – §A.2.7 minimum profile, with CAVLC or
//! CABAC, lossy or lossless, Annex B or `avcC`-framed. Both directions
//! are proven per layout: our decoder reconstructs the stream within
//! the lossy bound (exactly for `lossless=true`), and a black-box
//! reference decoder run as an opaque process decodes the same bytes
//! byte-identically to ours.

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational, VideoFrame, VideoPlane,
};
use oxideav_h264::avcc::AvcDecoderConfigurationRecord;
use oxideav_h264::nal::{AnnexBSplitter, NalUnitType, parse_nal_unit};
use oxideav_h264::sps::Sps;

const W: usize = 64;
const H: usize = 48;

type Opts<'a> = &'a [(&'a str, &'a str)];
/// `(layout, options, expected profile_idc, trailer expected)`.
type AvccCase<'a> = (Layout, Opts<'a>, u8, bool);
/// `(layout, options, why it must be refused)`.
type RejectCase<'a> = (Layout, Vec<(&'a str, &'a str)>, &'a str);

#[derive(Clone, Copy)]
struct Layout {
    pf: PixelFormat,
    cf: u32,
    bd: u32,
    gray: bool,
}

const LAYOUTS: &[Layout] = &[
    Layout {
        pf: PixelFormat::Yuv422P,
        cf: 2,
        bd: 8,
        gray: false,
    },
    Layout {
        pf: PixelFormat::Yuv444P,
        cf: 3,
        bd: 8,
        gray: false,
    },
    Layout {
        pf: PixelFormat::Gray8,
        cf: 1,
        bd: 8,
        gray: true,
    },
    Layout {
        pf: PixelFormat::Yuv420P10Le,
        cf: 1,
        bd: 10,
        gray: false,
    },
    Layout {
        pf: PixelFormat::Yuv422P10Le,
        cf: 2,
        bd: 10,
        gray: false,
    },
    Layout {
        pf: PixelFormat::Yuv444P10Le,
        cf: 3,
        bd: 10,
        gray: false,
    },
    Layout {
        pf: PixelFormat::Gray10Le,
        cf: 1,
        bd: 10,
        gray: true,
    },
    Layout {
        pf: PixelFormat::Yuv420P12Le,
        cf: 1,
        bd: 12,
        gray: false,
    },
    Layout {
        pf: PixelFormat::Yuv422P12Le,
        cf: 2,
        bd: 12,
        gray: false,
    },
    Layout {
        pf: PixelFormat::Yuv444P12Le,
        cf: 3,
        bd: 12,
        gray: false,
    },
    Layout {
        pf: PixelFormat::Gray12Le,
        cf: 1,
        bd: 12,
        gray: true,
    },
];

fn chroma_dims(cf: u32) -> (usize, usize) {
    match cf {
        1 => (W / 2, H / 2),
        2 => (W / 2, H),
        _ => (W, H),
    }
}

/// Coded pixel format (grey → the 4:2:0 layout the stream carries).
fn coded_pf(l: Layout) -> PixelFormat {
    match (l.cf, l.bd) {
        (1, 8) => PixelFormat::Yuv420P,
        (2, 8) => PixelFormat::Yuv422P,
        (3, 8) => PixelFormat::Yuv444P,
        (1, 10) => PixelFormat::Yuv420P10Le,
        (2, 10) => PixelFormat::Yuv422P10Le,
        (3, 10) => PixelFormat::Yuv444P10Le,
        (1, _) => PixelFormat::Yuv420P12Le,
        (2, _) => PixelFormat::Yuv422P12Le,
        _ => PixelFormat::Yuv444P12Le,
    }
}

/// Expected `profile_idc`: 8-bit 4:2:0 input (grey included) rides the
/// Baseline / Main session; everything else the §A.2.4 – §A.2.7
/// minimum High-family profile.
fn expected_profile(l: Layout, cabac: bool, lossless: bool) -> u8 {
    if l.cf == 1 && l.bd == 8 && !lossless {
        if cabac { 77 } else { 66 }
    } else if lossless || l.cf == 3 || l.bd > 10 {
        244
    } else if l.cf == 2 {
        122
    } else {
        110
    }
}

/// Synthetic source planes (`u16` samples at `bd`): a gradient, a
/// block pattern and a frame-dependent texture, moving per frame.
fn source(l: Layout, n: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let max = (1u32 << l.bd) - 1;
    let sh = l.bd - 8;
    let mut y = vec![0u16; W * H];
    for j in 0..H {
        for i in 0..W {
            let block = if ((i + n) / 8 + j / 8).is_multiple_of(3) {
                70
            } else {
                0
            };
            let g = 20 + block + ((i * 3 + j * 2 + n * 5) % 140);
            let fine = ((i * 13 + j * 7) % 5) as u32;
            y[j * W + i] = (((g as u32) << sh) + fine * (1 << sh) / 4).min(max) as u16;
        }
    }
    let (cw, ch) = chroma_dims(l.cf);
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            u[j * cw + i] = (((90 + ((i + j + n) % 60)) as u32) << sh).min(max) as u16;
            v[j * cw + i] = (((120 + ((i * 2 + j * 3) % 50)) as u32) << sh).min(max) as u16;
        }
    }
    (y, u, v)
}

fn plane_bytes(samples: &[u16], wide: bool, stride_samples: usize, pad: usize) -> VideoPlane {
    let rows = samples.len() / stride_samples;
    let bps = if wide { 2 } else { 1 };
    let stride = (stride_samples + pad) * bps;
    let mut data = vec![0u8; stride * rows];
    for r in 0..rows {
        for c in 0..stride_samples {
            let s = samples[r * stride_samples + c];
            if wide {
                data[r * stride + 2 * c..r * stride + 2 * c + 2].copy_from_slice(&s.to_le_bytes());
            } else {
                data[r * stride + c] = s as u8;
            }
        }
    }
    VideoPlane { stride, data }
}

fn frame(l: Layout, n: usize, pad: usize) -> Frame {
    let (y, u, v) = source(l, n);
    let wide = l.bd > 8;
    let (cw, _) = chroma_dims(l.cf);
    let mut planes = vec![plane_bytes(&y, wide, W, pad)];
    if !l.gray {
        planes.push(plane_bytes(&u, wide, cw, pad));
        planes.push(plane_bytes(&v, wide, cw, pad));
    }
    Frame::Video(VideoFrame {
        pts: Some(n as i64),
        planes,
    })
}

fn params(l: Layout, opts: &[(&str, &str)]) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("h264"));
    p.width = Some(W as u32);
    p.height = Some(H as u32);
    p.pixel_format = Some(l.pf);
    p.frame_rate = Some(Rational::new(30, 1));
    let mut o = oxideav_core::CodecOptions::new();
    for (k, v) in opts {
        o = o.set(*k, *v);
    }
    p.options = o;
    p
}

fn encode(
    l: Layout,
    opts: &[(&str, &str)],
    frames: usize,
    pad: usize,
) -> (Vec<Packet>, CodecParameters) {
    let ctx = {
        let mut c = oxideav_core::RuntimeContext::new();
        oxideav_h264::register(&mut c);
        c
    };
    let mut enc = ctx
        .codecs
        .first_encoder(&params(l, opts))
        .unwrap_or_else(|e| panic!("{:?}: encoder factory: {e}", l.pf));
    let mut packets = Vec::new();
    for n in 0..frames {
        enc.send_frame(&frame(l, n, pad)).expect("send");
        loop {
            match enc.receive_packet() {
                Ok(p) => packets.push(p),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("receive_packet: {e}"),
            }
        }
    }
    enc.flush().expect("flush");
    while let Ok(p) = enc.receive_packet() {
        packets.push(p);
    }
    (packets, enc.output_params().clone())
}

/// Decode with the registry decoder; planes concatenated per frame.
fn decode_ours(packets: &[Packet], extradata: &[u8]) -> Vec<Vec<u8>> {
    let mut p = CodecParameters::video(CodecId::new("h264"));
    p.extradata = extradata.to_vec();
    let mut dec = oxideav_h264::h264_decoder::make_decoder(&p).expect("decoder");
    let mut out = Vec::new();
    let mut push = |vf: VideoFrame| {
        let mut bytes = Vec::new();
        for pl in vf.image_planes() {
            bytes.extend_from_slice(&pl.data);
        }
        out.push(bytes);
    };
    for pk in packets {
        dec.send_packet(pk).expect("send_packet");
        while let Ok(Frame::Video(vf)) = dec.receive_frame() {
            push(vf);
        }
    }
    dec.flush().expect("flush");
    while let Ok(Frame::Video(vf)) = dec.receive_frame() {
        push(vf);
    }
    out
}

fn first_sps(packets: &[Packet]) -> Sps {
    for nal in AnnexBSplitter::new(&packets[0].data) {
        let parsed = parse_nal_unit(nal).expect("nal");
        if parsed.header.nal_unit_type == NalUnitType::Sps {
            return Sps::parse(&parsed.rbsp).expect("sps");
        }
    }
    panic!("no SPS in the first access unit");
}

/// Source frame as the decoder lays it out (coded chroma included).
fn source_bytes(l: Layout, n: usize) -> Vec<u8> {
    let (y, u, v) = source(l, n);
    let wide = l.bd > 8;
    let (cw, ch) = chroma_dims(l.cf);
    let mut out = Vec::new();
    let mut put = |s: &[u16]| {
        for &x in s {
            if wide {
                out.extend_from_slice(&x.to_le_bytes());
            } else {
                out.push(x as u8);
            }
        }
    };
    put(&y);
    if l.gray {
        let mid = vec![1u16 << (l.bd - 1); cw * ch];
        put(&mid);
        put(&mid);
    } else {
        put(&u);
        put(&v);
    }
    out
}

fn psnr_luma(l: Layout, decoded: &[u8], n: usize) -> f64 {
    let (y, _, _) = source(l, n);
    let wide = l.bd > 8;
    let max = ((1u32 << l.bd) - 1) as f64;
    let mut sse = 0f64;
    for (i, &s) in y.iter().enumerate() {
        let d = if wide {
            u16::from_le_bytes([decoded[2 * i], decoded[2 * i + 1]]) as f64
        } else {
            decoded[i] as f64
        };
        sse += (s as f64 - d) * (s as f64 - d);
    }
    let mse = sse / y.len() as f64;
    if mse == 0.0 {
        99.0
    } else {
        10.0 * (max * max / mse).log10()
    }
}

/// Black-box reference decode of an Annex B stream → per-frame planes
/// (its raw output is the stream's native layout). `None` when the
/// binary is absent.
fn reference_decode(annex_b: &[u8], tag: &str) -> Option<Vec<u8>> {
    let refbin = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !refbin.exists() {
        eprintln!("skip {tag}: reference decoder binary not present");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h264-r475-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bs = dir.join("in.h264");
    let out = dir.join("out.yuv");
    std::fs::write(&bs, annex_b).unwrap();
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

fn annex_b_of(packets: &[Packet]) -> Vec<u8> {
    let mut v = Vec::new();
    for p in packets {
        v.extend_from_slice(&p.data);
    }
    v
}

/// One layout, lossy: SPS truthful, output params truthful, our
/// decode within the bound, reference decode byte-identical to ours.
fn check_lossy(l: Layout, cabac: bool, qp: u32) -> (usize, f64) {
    let tag = format!(
        "{:?}-{}-qp{qp}",
        l.pf,
        if cabac { "cabac" } else { "cavlc" }
    );
    let qp_s = qp.to_string();
    let (packets, out_params) = encode(
        l,
        &[
            ("rc", "cqp"),
            ("qp", &qp_s),
            ("gop", "2"),
            ("cabac", if cabac { "true" } else { "false" }),
        ],
        3,
        3,
    );
    assert_eq!(packets.len(), 3, "{tag}: one access unit per picture");
    assert!(packets[0].flags.keyframe && !packets[1].flags.keyframe && packets[2].flags.keyframe);
    assert_eq!(
        out_params.pixel_format,
        Some(coded_pf(l)),
        "{tag}: coded pixel format"
    );
    assert!(
        out_params.extradata.is_empty(),
        "{tag}: Annex B keeps extradata empty"
    );
    let sps = first_sps(&packets);
    assert_eq!(
        sps.profile_idc,
        expected_profile(l, cabac, false),
        "{tag}: §A.2 profile"
    );
    assert_eq!(sps.chroma_format_idc, l.cf, "{tag}: chroma_format_idc");
    assert_eq!(sps.bit_depth_luma_minus8 + 8, l.bd, "{tag}: bit_depth_luma");
    assert_eq!(
        sps.bit_depth_chroma_minus8 + 8,
        l.bd,
        "{tag}: bit_depth_chroma"
    );
    assert!(!sps.qpprime_y_zero_transform_bypass_flag);

    let ours = decode_ours(&packets, &[]);
    assert_eq!(ours.len(), 3, "{tag}: decoded picture count");
    let expect_len = source_bytes(l, 0).len();
    let mut min_psnr = f64::MAX;
    for (n, f) in ours.iter().enumerate() {
        assert_eq!(f.len(), expect_len, "{tag}: frame {n} size");
        let p = psnr_luma(l, f, n);
        min_psnr = min_psnr.min(p);
        assert!(
            p > 30.0,
            "{tag}: frame {n} luma PSNR {p:.2} dB below the lossy bound"
        );
    }
    if let Some(raw) = reference_decode(&annex_b_of(&packets), &tag) {
        assert_eq!(raw.len(), expect_len * 3, "{tag}: reference picture count");
        for (n, f) in ours.iter().enumerate() {
            let theirs = &raw[n * expect_len..(n + 1) * expect_len];
            let mism = f.iter().zip(theirs).filter(|(a, b)| a != b).count();
            assert_eq!(
                mism, 0,
                "{tag}: frame {n}: {mism} bytes differ vs reference decoder"
            );
        }
    }
    (annex_b_of(&packets).len(), min_psnr)
}

#[test]
fn every_layout_cavlc_lossy_both_directions() {
    for &l in LAYOUTS {
        check_lossy(l, false, 24);
    }
}

#[test]
fn every_layout_cabac_lossy_both_directions() {
    for &l in LAYOUTS {
        check_lossy(l, true, 24);
    }
}

/// `lossless=true`: profile 244 with the bypass flag, reconstruction
/// equals the source exactly in our decoder and in the reference.
#[test]
fn lossless_layouts_reconstruct_exactly() {
    for &l in &[
        LAYOUTS[0], // 4:2:2 8-bit
        LAYOUTS[1], // 4:4:4 8-bit
        LAYOUTS[3], // 4:2:0 10-bit
        LAYOUTS[6], // grey 10-bit
        LAYOUTS[8], // 4:2:2 12-bit
        LAYOUTS[9], // 4:4:4 12-bit
    ] {
        for cabac in [false, true] {
            let tag = format!("lossless-{:?}-{cabac}", l.pf);
            let (packets, out_params) = encode(
                l,
                &[
                    ("lossless", "true"),
                    ("gop", "1"),
                    ("cabac", if cabac { "true" } else { "false" }),
                ],
                2,
                0,
            );
            assert_eq!(out_params.pixel_format, Some(coded_pf(l)));
            let sps = first_sps(&packets);
            assert_eq!(
                sps.profile_idc, 244,
                "{tag}: transform bypass needs 244 (§A.2.7)"
            );
            assert!(sps.qpprime_y_zero_transform_bypass_flag, "{tag}");
            let ours = decode_ours(&packets, &[]);
            assert_eq!(ours.len(), 2);
            for (n, f) in ours.iter().enumerate() {
                assert_eq!(f, &source_bytes(l, n), "{tag}: frame {n} not lossless");
            }
            if let Some(raw) = reference_decode(&annex_b_of(&packets), &tag) {
                let len = ours[0].len();
                assert_eq!(raw.len(), 2 * len);
                for (n, f) in ours.iter().enumerate() {
                    assert_eq!(
                        f,
                        &raw[n * len..(n + 1) * len],
                        "{tag}: reference frame {n}"
                    );
                }
            }
        }
    }
}

/// 8-bit 4:2:0 input under an explicit High-family profile routes to
/// the High-family encoder (High 100, 8x8 transform, deblock off).
#[test]
fn explicit_high_profile_on_8bit_420() {
    let l = Layout {
        pf: PixelFormat::Yuv420P,
        cf: 1,
        bd: 8,
        gray: false,
    };
    for (opts, profile) in [
        (vec![("profile", "high"), ("transform_8x8", "true")], 100u8),
        (vec![("profile", "high10")], 110),
        (vec![("profile", "high422"), ("cabac", "true")], 122),
        (vec![("profile", "high444"), ("deblock", "false")], 244),
        (vec![("transform_8x8", "true")], 100),
        (vec![("deblock", "false")], 100),
    ] {
        let (packets, _) = encode(l, &opts, 3, 0);
        let sps = first_sps(&packets);
        assert_eq!(sps.profile_idc, profile, "{opts:?}");
        let ours = decode_ours(&packets, &[]);
        assert_eq!(ours.len(), 3);
        for (n, f) in ours.iter().enumerate() {
            assert!(psnr_luma(l, f, n) > 30.0, "{opts:?}: frame {n}");
        }
        if let Some(raw) = reference_decode(&annex_b_of(&packets), "high-420") {
            let len = ours[0].len();
            for (n, f) in ours.iter().enumerate() {
                assert_eq!(
                    f,
                    &raw[n * len..(n + 1) * len],
                    "{opts:?}: reference frame {n}"
                );
            }
        }
    }
}

/// `format=avcc`: extradata is a §5.3.2.1 record built from the
/// stream's own SPS / PPS (trailer for the High family, none for
/// Baseline), packets are 4-byte length-prefixed without parameter
/// sets, and the decoder reconstructs the same pictures from
/// extradata + samples as from the Annex B stream.
#[test]
fn avcc_framing_round_trips_through_extradata() {
    let cases: &[AvccCase] = &[
        (
            Layout {
                pf: PixelFormat::Yuv420P,
                cf: 1,
                bd: 8,
                gray: false,
            },
            &[("qp", "28")],
            66,
            false,
        ),
        (
            Layout {
                pf: PixelFormat::Yuv420P,
                cf: 1,
                bd: 8,
                gray: false,
            },
            &[("qp", "28"), ("cabac", "true"), ("bframes", "1")],
            77,
            false,
        ),
        (LAYOUTS[3], &[("qp", "28")], 110, true),
        (LAYOUTS[9], &[("qp", "28"), ("cabac", "true")], 244, true),
        (LAYOUTS[0], &[("lossless", "true")], 244, true),
    ];
    for &(l, base, profile, trailer) in cases {
        let mut annexb_opts: Vec<(&str, &str)> = base.to_vec();
        annexb_opts.push(("gop", "2"));
        let mut avcc_opts = annexb_opts.clone();
        avcc_opts.push(("format", "avcc"));
        let (ab_packets, ab_params) = encode(l, &annexb_opts, 4, 0);
        let (av_packets, av_params) = encode(l, &avcc_opts, 4, 0);
        let tag = format!("avcc-{:?}-{profile}", l.pf);
        assert!(ab_params.extradata.is_empty());
        assert_eq!(av_params.pixel_format, ab_params.pixel_format);

        let rec = AvcDecoderConfigurationRecord::parse(&av_params.extradata)
            .unwrap_or_else(|e| panic!("{tag}: {e}"));
        assert_eq!(rec.profile_idc, profile, "{tag}");
        assert_eq!(rec.length_size, 4);
        assert_eq!(rec.sps.len(), 1);
        assert_eq!(rec.pps.len(), 1);
        let sps = first_sps(&ab_packets);
        assert_eq!(rec.level_idc, sps.level_idc, "{tag}");
        assert_eq!(
            rec.chroma_format.is_some(),
            trailer,
            "{tag}: trailer presence"
        );
        if trailer {
            assert_eq!(rec.chroma_format, Some(l.cf as u8));
            assert_eq!(rec.bit_depth_luma_minus8, Some((l.bd - 8) as u8));
            assert_eq!(rec.bit_depth_chroma_minus8, Some((l.bd - 8) as u8));
        }
        // The SPS in the record is the SPS the Annex B stream carries.
        let in_band: Vec<&[u8]> = AnnexBSplitter::new(&ab_packets[0].data)
            .filter(|n| n[0] & 0x1f == 7)
            .collect();
        assert_eq!(rec.sps[0], in_band[0], "{tag}: record SPS == in-band SPS");
        assert_eq!(
            rec.to_bytes().unwrap(),
            av_params.extradata,
            "{tag}: re-serialise"
        );

        // Samples: 4-byte lengths, no SPS / PPS, same count and flags.
        assert_eq!(av_packets.len(), ab_packets.len());
        for (a, b) in av_packets.iter().zip(&ab_packets) {
            assert_eq!(a.flags.keyframe, b.flags.keyframe);
            assert_eq!((a.pts, a.dts), (b.pts, b.dts));
            let mut pos = 0;
            while pos < a.data.len() {
                let len = u32::from_be_bytes(a.data[pos..pos + 4].try_into().unwrap()) as usize;
                let t = a.data[pos + 4] & 0x1f;
                assert!(
                    !matches!(t, 7 | 8 | 12 | 13),
                    "{tag}: parameter set / filler in sample"
                );
                pos += 4 + len;
            }
            assert_eq!(pos, a.data.len(), "{tag}: sample framing");
        }
        let ours_ab = decode_ours(&ab_packets, &[]);
        let ours_av = decode_ours(&av_packets, &av_params.extradata);
        assert_eq!(ours_ab.len(), 4, "{tag}");
        assert_eq!(
            ours_ab, ours_av,
            "{tag}: extradata + samples decode == Annex B decode"
        );
    }
}

#[test]
fn rejections_are_clean_errors() {
    let l10 = LAYOUTS[3];
    let l444 = LAYOUTS[1];
    let l420 = Layout {
        pf: PixelFormat::Yuv420P,
        cf: 1,
        bd: 8,
        gray: false,
    };
    let cases: Vec<RejectCase> = vec![
        (l444, vec![("profile", "high")], "High does not admit 4:4:4"),
        (
            l10,
            vec![("profile", "high"), ("qp", "20")],
            "High does not admit 10-bit",
        ),
        (
            l10,
            vec![("lossless", "true"), ("profile", "high10")],
            "bypass needs 244",
        ),
        (l10, vec![("bframes", "1")], "no B on the High-family path"),
        (
            l10,
            vec![("rc", "cbr"), ("bitrate", "100000")],
            "no rate control on the High-family path",
        ),
        (
            l420,
            vec![("profile", "baseline"), ("cabac", "true")],
            "Baseline forbids CABAC",
        ),
        (
            l420,
            vec![("profile", "main")],
            "Main needs cabac or bframes",
        ),
        (
            l10,
            vec![("profile", "baseline")],
            "Baseline is 8-bit 4:2:0 only",
        ),
        (l420, vec![("format", "mp4")], "unknown format"),
    ];
    for (l, opts, why) in cases {
        let r = oxideav_h264::h264_encoder::make_encoder(&params(l, &opts));
        assert!(r.is_err(), "{why}: {opts:?} should be refused");
    }
    let mut p = params(l420, &[]);
    p.pixel_format = Some(PixelFormat::Yuv420P16Le);
    assert!(
        oxideav_h264::h264_encoder::make_encoder(&p).is_err(),
        "16-bit is not accepted"
    );
    // rc=auto with a known bitrate is still cqp on the High-family path.
    let mut p = params(l10, &[]);
    p.bit_rate = Some(200_000);
    let enc = oxideav_h264::h264_encoder::make_encoder(&p).expect("auto → cqp");
    assert_eq!(enc.output_params().bit_rate, None);
}

/// The registry descriptor declares the accepted input formats.
#[test]
fn descriptor_lists_accepted_pixel_formats() {
    let mut ctx = oxideav_core::RuntimeContext::new();
    oxideav_h264::register(&mut ctx);
    for &l in LAYOUTS {
        let p = params(l, &[]);
        assert!(
            ctx.codecs.first_encoder(&p).is_ok(),
            "{:?} resolvable",
            l.pf
        );
    }
    let listed = oxideav_h264::h264_encoder::accepted_pixel_formats();
    assert!(listed.contains(&PixelFormat::Yuv420P));
    for &l in LAYOUTS {
        assert!(listed.contains(&l.pf), "{:?} listed", l.pf);
    }
}

/// Measurement table (bytes / min luma PSNR over 3 pictures at QPs
/// 20 / 28 / 36 per layout, CAVLC) — informational, printed with
/// `--nocapture`.
#[test]
fn measure_rate_distortion_table() {
    for &l in LAYOUTS {
        for qp in [20u32, 28, 36] {
            let (bytes, psnr) = check_lossy(l, false, qp);
            println!(
                "MEASURE {:?} qp={qp} bytes={bytes} min_luma_psnr={psnr:.2}",
                l.pf
            );
        }
    }
}
