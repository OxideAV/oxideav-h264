//! Round-420 — registry encoder (`oxideav_core::Encoder` surface).
//!
//! The crate now registers an encoder factory alongside the decoder
//! (dual-API: `h264_encoder::make_encoder` is also callable
//! directly). These tests drive the factory through
//! `RuntimeContext`/`CodecRegistry` exactly like a pipeline would:
//! options arrive in the `CodecParameters::options` bag, frames as
//! `Frame::Video`, packets come back Annex B with in-band SPS/PPS,
//! and the whole product must round-trip through the registry
//! *decoder*.

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, Rational, VideoFrame, VideoPlane,
};
use std::collections::VecDeque;

const W: u32 = 64;
const H: u32 = 64;

fn registry() -> oxideav_core::RuntimeContext {
    let mut ctx = oxideav_core::RuntimeContext::new();
    oxideav_h264::register(&mut ctx);
    ctx
}

fn base_params() -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("h264"));
    p.width = Some(W);
    p.height = Some(H);
    p.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
    p.frame_rate = Some(Rational::new(30, 1));
    p
}

fn make_video_frame(n: usize, pad: usize) -> Frame {
    let (w, h) = (W as usize, H as usize);
    let mut y = vec![0u8; (w + pad) * h];
    let mut u = vec![0u8; (w / 2 + pad) * (h / 2)];
    let mut v = vec![0u8; (w / 2 + pad) * (h / 2)];
    for j in 0..h {
        for i in 0..w {
            y[j * (w + pad) + i] = (30 + ((i * 3 + j * 2 + n * 7) % 190)) as u8;
        }
    }
    for j in 0..h / 2 {
        for i in 0..w / 2 {
            u[j * (w / 2 + pad) + i] = (100 + ((i + n) % 50)) as u8;
            v[j * (w / 2 + pad) + i] = (130 + ((j + n * 2) % 40)) as u8;
        }
    }
    Frame::Video(VideoFrame {
        pts: Some(n as i64),
        planes: vec![
            VideoPlane {
                stride: w + pad,
                data: y,
            },
            VideoPlane {
                stride: w / 2 + pad,
                data: u,
            },
            VideoPlane {
                stride: w / 2 + pad,
                data: v,
            },
        ],
    })
}

fn encode_n(params: &CodecParameters, frames: usize, pad: usize) -> Vec<Packet> {
    let ctx = registry();
    let mut enc = ctx.codecs.first_encoder(params).expect("encoder factory");
    assert_eq!(enc.codec_id().as_str(), "h264");
    let mut packets = Vec::new();
    for n in 0..frames {
        enc.send_frame(&make_video_frame(n, pad)).expect("send");
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
    packets
}

fn decode_all(packets: &[Packet]) -> Vec<VideoFrame> {
    let ctx = registry();
    let params = base_params();
    let mut dec = ctx.codecs.first_decoder(&params).expect("decoder factory");
    let mut out = VecDeque::new();
    for p in packets {
        dec.send_packet(p).expect("send_packet");
        while let Ok(Frame::Video(vf)) = dec.receive_frame() {
            out.push_back(vf);
        }
    }
    dec.flush().expect("flush");
    while let Ok(f) = dec.receive_frame() {
        if let Frame::Video(vf) = f {
            out.push_back(vf);
        }
    }
    out.into()
}

#[test]
fn cqp_registry_roundtrip_with_padded_strides() {
    let mut params = base_params();
    params.options.insert("rc", "cqp");
    params.options.insert("qp", "30");
    // pad = 16: strided input exercises the repack path.
    let packets = encode_n(&params, 8, 16);
    assert_eq!(packets.len(), 8, "one packet per frame");
    assert!(packets[0].flags.keyframe, "frame 0 must be IDR");
    assert!(
        !packets[1].flags.keyframe,
        "frame 1 must be P (gop default 30)"
    );
    for (n, p) in packets.iter().enumerate() {
        assert_eq!(p.pts, Some(n as i64), "pts passthrough");
        assert_eq!(p.dts, Some(n as i64), "dts == pts (no reordering)");
        assert!(!p.data.is_empty());
    }
    // Annex B: starts with a start code + SPS NAL.
    assert_eq!(&packets[0].data[..4], &[0, 0, 0, 1]);
    assert_eq!(packets[0].data[4] & 0x1F, 7, "first NAL is SPS");

    let frames = decode_all(&packets);
    assert_eq!(frames.len(), 8, "registry decoder frame count");
}

#[test]
fn cbr_via_bit_rate_param_auto_mode() {
    let mut params = base_params();
    params.bit_rate = Some(150_000); // rc=auto → CBR from bit_rate
    let mut params_gop = params.clone();
    params_gop.options.insert("gop", "15");
    let packets = encode_n(&params_gop, 45, 0);
    assert_eq!(packets.len(), 45);
    // gop=15 cadence.
    for (n, p) in packets.iter().enumerate() {
        assert_eq!(p.flags.keyframe, n % 15 == 0, "IDR cadence at {n}");
    }
    // Rate: post-warmup payload within 15% of target (tiny frames,
    // coarse content — the deep accuracy matrix lives in the session
    // tests; here we prove the plumbing).
    let bits: u64 = packets[15..].iter().map(|p| 8 * p.data.len() as u64).sum();
    let rate = bits as f64 / 1.0; // 30 frames @ 30 fps
    let err = (rate - 150_000.0).abs() / 150_000.0;
    eprintln!("registry CBR rate {rate:.0} bps (err {:.2}%)", err * 100.0);
    assert!(err < 0.15, "registry CBR err {err:.4}");

    let frames = decode_all(&packets);
    assert_eq!(frames.len(), 45);
}

#[test]
fn vbr_options_and_output_params() {
    let ctx = registry();
    let mut params = base_params();
    for (k, v) in [
        ("rc", "vbr"),
        ("bitrate", "120000"),
        ("max_bitrate", "240000"),
        ("buffer_size", "240000"),
        ("gop", "30"),
        ("cabac", "true"),
    ] {
        params.options.insert(k, v);
    }
    let enc = ctx.codecs.first_encoder(&params).expect("encoder factory");
    let out = enc.output_params();
    assert_eq!(out.codec_id.as_str(), "h264");
    assert_eq!(out.width, Some(W));
    assert_eq!(out.height, Some(H));
    assert_eq!(out.bit_rate, Some(120_000));
    assert!(out.extradata.is_empty(), "Annex B: in-band SPS/PPS");
}

#[test]
fn invalid_configs_error_cleanly() {
    let ctx = registry();

    // Unknown option key.
    let mut p = base_params();
    p.options.insert("nonsense", "1");
    assert!(ctx.codecs.first_encoder(&p).is_err());

    // vbr without bitrate.
    let mut p = base_params();
    p.options.insert("rc", "vbr");
    assert!(ctx.codecs.first_encoder(&p).is_err());

    // max_bitrate below bitrate.
    let mut p = base_params();
    p.options.insert("rc", "vbr");
    p.options.insert("bitrate", "200000");
    p.options.insert("max_bitrate", "100000");
    assert!(ctx.codecs.first_encoder(&p).is_err());

    // Non-mod-16 geometry.
    let mut p = base_params();
    p.width = Some(100);
    assert!(ctx.codecs.first_encoder(&p).is_err());

    // Bad QP.
    let mut p = base_params();
    p.options.insert("rc", "cqp");
    p.options.insert("qp", "60");
    assert!(ctx.codecs.first_encoder(&p).is_err());

    // gop 0.
    let mut p = base_params();
    p.options.insert("gop", "0");
    assert!(ctx.codecs.first_encoder(&p).is_err());
}

#[test]
fn direct_factory_matches_registry_output() {
    // Dual-API check: calling the factory directly must behave the
    // same as the registry path.
    let mut params = base_params();
    params.options.insert("rc", "cqp");
    params.options.insert("qp", "28");
    let mut direct = oxideav_h264::h264_encoder::make_encoder(&params).expect("direct factory");
    direct.send_frame(&make_video_frame(0, 0)).expect("send");
    let direct_packet = direct.receive_packet().expect("packet");

    let registry_packets = encode_n(&params, 1, 0);
    assert_eq!(direct_packet.data, registry_packets[0].data);
}
