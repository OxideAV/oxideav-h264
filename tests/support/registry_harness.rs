//! Shared harness for the registry-encoder integration tests: encode a
//! synthetic sequence at a size / layout / option set, check the SPS
//! geometry, decode with our decoder (lossy bound, or sample-exact for
//! `lossless=true`) and require a black-box decoder run as an opaque
//! process to output the same bytes.

#![allow(dead_code)]

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Rational, VideoFrame, VideoPlane,
};
use oxideav_h264::nal::{AnnexBSplitter, NalUnitType, parse_nal_unit};
use oxideav_h264::sps::Sps;

#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub pf: PixelFormat,
    /// Chroma format of the coded stream (0 = 4:0:0: one plane).
    pub cf: u32,
    pub bd: u32,
}

pub const YUV420: Layout = Layout {
    pf: PixelFormat::Yuv420P,
    cf: 1,
    bd: 8,
};
pub const YUV422: Layout = Layout {
    pf: PixelFormat::Yuv422P,
    cf: 2,
    bd: 8,
};
pub const YUV444: Layout = Layout {
    pf: PixelFormat::Yuv444P,
    cf: 3,
    bd: 8,
};
pub const YUV420_10: Layout = Layout {
    pf: PixelFormat::Yuv420P10Le,
    cf: 1,
    bd: 10,
};
pub const YUV422_10: Layout = Layout {
    pf: PixelFormat::Yuv422P10Le,
    cf: 2,
    bd: 10,
};
pub const YUV444_12: Layout = Layout {
    pf: PixelFormat::Yuv444P12Le,
    cf: 3,
    bd: 12,
};

pub fn sub(cf: u32) -> (usize, usize) {
    match cf {
        1 => (2, 2),
        2 => (2, 1),
        _ => (1, 1),
    }
}

/// Synthetic source planes at `bd` bits: gradient + blocks + texture,
/// moving per frame so P pictures carry motion across the padded edge.
pub fn source(l: Layout, w: usize, h: usize, n: usize) -> Vec<Vec<u16>> {
    let max = (1u32 << l.bd) - 1;
    let sh = l.bd - 8;
    let mut y = vec![0u16; w * h];
    for j in 0..h {
        for i in 0..w {
            let block = if ((i + 2 * n) / 8 + j / 8).is_multiple_of(3) {
                70
            } else {
                0
            };
            let g = 20 + block + ((i * 3 + j * 2 + n * 5) % 140);
            y[j * w + i] = ((g as u32) << sh).min(max) as u16;
        }
    }
    if l.cf == 0 {
        return vec![y];
    }
    let (sw, shh) = sub(l.cf);
    let (cw, ch) = (w / sw, h / shh);
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            u[j * cw + i] = (((90 + ((i + j + n) % 60)) as u32) << sh).min(max) as u16;
            v[j * cw + i] = (((120 + ((i * 2 + j * 3) % 50)) as u32) << sh).min(max) as u16;
        }
    }
    vec![y, u, v]
}

pub fn to_bytes(samples: &[u16], wide: bool) -> Vec<u8> {
    if wide {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    } else {
        samples.iter().map(|&s| s as u8).collect()
    }
}

pub fn frame(l: Layout, w: usize, h: usize, n: usize) -> Frame {
    let wide = l.bd > 8;
    let bps = if wide { 2 } else { 1 };
    let (sw, _) = sub(l.cf);
    let planes = source(l, w, h, n)
        .iter()
        .zip([w, w / sw, w / sw])
        .map(|(p, pw)| VideoPlane {
            stride: pw * bps,
            data: to_bytes(p, wide),
        })
        .collect();
    Frame::Video(VideoFrame {
        pts: Some(n as i64),
        planes,
    })
}

pub fn source_bytes(l: Layout, w: usize, h: usize, n: usize) -> Vec<u8> {
    source(l, w, h, n)
        .iter()
        .flat_map(|p| to_bytes(p, l.bd > 8))
        .collect()
}

pub fn params(l: Layout, w: usize, h: usize, opts: &[(&str, &str)]) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("h264"));
    p.width = Some(w as u32);
    p.height = Some(h as u32);
    p.pixel_format = Some(l.pf);
    p.frame_rate = Some(Rational::new(30, 1));
    let mut o = oxideav_core::CodecOptions::new();
    for (k, v) in opts {
        o = o.set(*k, *v);
    }
    p.options = o;
    p
}

pub fn encode(
    l: Layout,
    w: usize,
    h: usize,
    opts: &[(&str, &str)],
    frames: usize,
) -> (Vec<Packet>, CodecParameters) {
    let mut enc = oxideav_h264::h264_encoder::make_encoder(&params(l, w, h, opts))
        .unwrap_or_else(|e| panic!("{l:?} {w}x{h}: encoder: {e}"));
    let mut packets = Vec::new();
    for n in 0..frames {
        enc.send_frame(&frame(l, w, h, n)).expect("send");
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

/// Decode with the registry decoder → `(per-frame planes, sizes)`.
pub fn decode_ours(packets: &[Packet]) -> Vec<Vec<u8>> {
    let p = CodecParameters::video(CodecId::new("h264"));
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

pub fn first_sps(packets: &[Packet]) -> Sps {
    for nal in AnnexBSplitter::new(&packets[0].data) {
        let parsed = parse_nal_unit(nal).expect("nal");
        if parsed.header.nal_unit_type == NalUnitType::Sps {
            return Sps::parse(&parsed.rbsp).expect("sps");
        }
    }
    panic!("no SPS in the first access unit");
}

pub fn black_box_decoder() -> Option<std::path::PathBuf> {
    [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    .iter()
    .map(std::path::PathBuf::from)
    .find(|p| p.exists())
}

/// Black-box decode of an Annex B stream to raw planes in the stream's
/// native layout. `None` when the binary is absent.
pub fn reference_decode(annex_b: &[u8], tag: &str) -> Option<Vec<u8>> {
    let Some(refbin) = black_box_decoder() else {
        eprintln!("skip {tag}: black-box decoder binary not present");
        return None;
    };
    let dir = std::env::temp_dir().join(format!("oxideav-h264-crop-{tag}-{}", std::process::id()));
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
        .expect("spawn black-box decoder");
    assert!(status.success(), "{tag}: black-box decoder failed");
    let raw = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Some(raw)
}

pub fn psnr(l: Layout, a: &[u8], b: &[u8]) -> f64 {
    let wide = l.bd > 8;
    let max = ((1u32 << l.bd) - 1) as f64;
    let get = |s: &[u8], i: usize| {
        if wide {
            u16::from_le_bytes([s[2 * i], s[2 * i + 1]]) as f64
        } else {
            s[i] as f64
        }
    };
    let n = if wide { a.len() / 2 } else { a.len() };
    let sse: f64 = (0..n).map(|i| (get(a, i) - get(b, i)).powi(2)).sum();
    if sse == 0.0 {
        99.0
    } else {
        10.0 * (max * max * n as f64 / sse).log10()
    }
}

/// Encode `frames` pictures of `w` x `h`, check the SPS, both decoders.
/// Returns the first SPS and the total stream size in bytes.
pub fn check(l: Layout, w: usize, h: usize, opts: &[(&str, &str)], frames: usize) -> (Sps, usize) {
    let tag = format!(
        "{:?}-{w}x{h}-{}",
        l.pf,
        opts.iter()
            .map(|(k, v)| format!("{k}{v}"))
            .collect::<Vec<_>>()
            .join("-")
    );
    let lossless = opts.contains(&("lossless", "true"));
    let (packets, out_params) = encode(l, w, h, opts, frames);
    assert_eq!(out_params.width, Some(w as u32), "{tag}: output width");
    assert_eq!(out_params.height, Some(h as u32), "{tag}: output height");

    let sps = first_sps(&packets);
    assert_eq!(sps.chroma_format_idc, l.cf, "{tag}: chroma_format_idc");
    assert_eq!(sps.bit_depth_luma_minus8 + 8, l.bd, "{tag}: luma bit depth");
    assert_eq!(out_params.pixel_format, Some(l.pf), "{tag}: output layout");
    let (sw, shh) = sub(l.cf);
    let (cw, ch) = (w.next_multiple_of(16), h.next_multiple_of(16));
    assert_eq!(sps.pic_width_in_mbs_minus1 as usize + 1, cw / 16, "{tag}");
    assert_eq!(
        sps.pic_height_in_map_units_minus1 as usize + 1,
        ch / 16,
        "{tag}"
    );
    assert_eq!(sps.crop_units(), (sw as u32, shh as u32), "{tag}");
    match &sps.frame_cropping {
        None => assert_eq!((cw, ch), (w, h), "{tag}: cropping missing"),
        Some(c) => {
            assert_eq!((c.left, c.top), (0, 0), "{tag}");
            assert_eq!(c.right as usize * sw, cw - w, "{tag}: right offset");
            assert_eq!(c.bottom as usize * shh, ch - h, "{tag}: bottom offset");
        }
    }
    assert_eq!(
        sps.cropped_dimensions(),
        (w as u32, h as u32),
        "{tag}: cropped size"
    );

    let ours = decode_ours(&packets);
    assert_eq!(ours.len(), frames, "{tag}: frame count");
    for (n, f) in ours.iter().enumerate() {
        let src = source_bytes(l, w, h, n);
        assert_eq!(f.len(), src.len(), "{tag}: frame {n} size");
        if lossless {
            assert!(f == &src, "{tag}: frame {n} not lossless");
        } else {
            let p = psnr(l, f, &src);
            assert!(p > 30.0, "{tag}: frame {n} PSNR {p:.2} dB");
        }
    }
    // Reordered streams come out in display order from both decoders.
    let mut annex_b = Vec::new();
    for p in &packets {
        annex_b.extend_from_slice(&p.data);
    }
    if let Some(mut reference) = reference_decode(&annex_b, &tag) {
        if l.cf == 0 {
            // That decoder returns a 4:0:0 stream padded out to 4:2:0
            // with constant chroma planes (asking it for a grey layout
            // would rescale the samples): keep the luma plane of each
            // frame.
            let bps = if l.bd > 8 { 2 } else { 1 };
            let luma = w * h * bps;
            let frame = luma + 2 * w.div_ceil(2) * h.div_ceil(2) * bps;
            assert_eq!(
                reference.len(),
                frame * frames,
                "{tag}: black-box output size"
            );
            reference = reference
                .chunks_exact(frame)
                .flat_map(|f| f[..luma].iter().copied())
                .collect();
        }
        let all: Vec<u8> = ours.concat();
        assert_eq!(reference.len(), all.len(), "{tag}: black-box output size");
        assert!(reference == all, "{tag}: black-box decoder output differs");
    }
    (sps, annex_b.len())
}

pub const GRAY8: Layout = Layout {
    pf: PixelFormat::Gray8,
    cf: 0,
    bd: 8,
};
pub const GRAY10: Layout = Layout {
    pf: PixelFormat::Gray10Le,
    cf: 0,
    bd: 10,
};
pub const GRAY12: Layout = Layout {
    pf: PixelFormat::Gray12Le,
    cf: 0,
    bd: 12,
};
