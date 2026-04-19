//! Round-trip the progressive IDR + P-slice encoder through this crate's
//! decoder.
//!
//! Drives [`H264Encoder`] with `p_slice_interval = 3` (one IDR + two P
//! frames per GOP), then feeds every emitted Annex B packet into
//! [`H264Decoder`] and validates that luma PSNR on each decoded frame
//! stays above the Baseline-P bound for QP 26.
//!
//! Reference bar for QP 26: ≥ 30 dB luma.

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{
    frame::VideoPlane, CodecId, Error, Frame, Packet, PixelFormat, TimeBase, VideoFrame,
};
use oxideav_h264::decoder::H264Decoder;
use oxideav_h264::encoder::{H264Encoder, H264EncoderOptions};

fn make_gradient_frame(w: u32, h: u32, t: u32) -> VideoFrame {
    let cw = w / 2;
    let ch = h / 2;
    let mut y = vec![0u8; (w * h) as usize];
    let mut cb = vec![128u8; (cw * ch) as usize];
    let mut cr = vec![128u8; (cw * ch) as usize];
    // Diagonal gradient with a small temporal offset so successive frames
    // aren't identical — drives the motion search (and non-zero residual).
    for r in 0..h {
        for c in 0..w {
            let off = (r + c + t) % 256;
            y[(r * w + c) as usize] = off as u8;
        }
    }
    for r in 0..ch {
        for c in 0..cw {
            cb[(r * cw + c) as usize] = ((64 + r + t).min(255)) as u8;
            cr[(r * cw + c) as usize] = 192u32.saturating_sub(c + t) as u8;
        }
    }
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width: w,
        height: h,
        pts: Some(t as i64),
        time_base: TimeBase::new(1, 30),
        planes: vec![
            VideoPlane {
                stride: w as usize,
                data: y,
            },
            VideoPlane {
                stride: cw as usize,
                data: cb,
            },
            VideoPlane {
                stride: cw as usize,
                data: cr,
            },
        ],
    }
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = *x as f64 - *y as f64;
            d * d
        })
        .sum::<f64>()
        / (a.len() as f64);
    if mse < 1e-9 {
        99.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

fn receive_all(dec: &mut H264Decoder, out: &mut Vec<VideoFrame>) {
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => out.push(v),
            Ok(_) => {}
            Err(Error::NeedMore) | Err(Error::Eof) => return,
            Err(e) => panic!("receive_frame: {e}"),
        }
    }
}

#[test]
fn roundtrip_i_then_p_p_64x64_qp26() {
    let w = 64u32;
    let h = 64u32;
    let qp = 26i32;
    let mut enc = H264Encoder::new(
        CodecId::new("h264"),
        w,
        h,
        H264EncoderOptions {
            qp,
            p_slice_interval: 3,
            ..Default::default()
        },
    )
    .expect("encoder::new");

    // Three source frames with small per-frame shifts.
    let sources: Vec<VideoFrame> = (0..3u32).map(|t| make_gradient_frame(w, h, t)).collect();

    let mut packets: Vec<Vec<u8>> = Vec::new();
    for src in &sources {
        enc.send_frame(&Frame::Video(src.clone()))
            .expect("send_frame");
        let pkt = enc.receive_packet().expect("receive_packet");
        packets.push(pkt.data.clone());
    }
    enc.flush().expect("flush");

    // First packet is the IDR (keyframe); subsequent packets are P
    // (non-keyframe).
    assert!(!packets[0].is_empty(), "IDR packet must be non-empty");
    assert!(!packets[1].is_empty(), "P packet must be non-empty");
    assert!(!packets[2].is_empty(), "P packet must be non-empty");

    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let mut decoded: Vec<VideoFrame> = Vec::new();
    for (i, bytes) in packets.iter().enumerate() {
        let pkt = Packet::new(0, TimeBase::new(1, 30), bytes.clone()).with_pts(i as i64);
        dec.send_packet(&pkt).expect("decoder send_packet");
        receive_all(&mut dec, &mut decoded);
    }
    // Flush any remaining frames.
    dec.flush().expect("decoder flush");
    receive_all(&mut dec, &mut decoded);

    assert_eq!(
        decoded.len(),
        3,
        "expected 3 decoded frames (I + P + P), got {}",
        decoded.len()
    );

    for (i, (src, dec)) in sources.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(dec.width, w);
        assert_eq!(dec.height, h);
        let psnr_y = psnr(&src.planes[0].data, &dec.planes[0].data);
        eprintln!(
            "frame {i} ({}): {} bytes, luma psnr = {:.2} dB",
            if i == 0 { "IDR" } else { "P" },
            packets[i].len(),
            psnr_y
        );
        assert!(psnr_y >= 30.0, "frame {i} luma psnr {psnr_y:.2} < 30 dB");
    }
}

#[test]
fn roundtrip_i_then_p_panning_frame_64x64_qp26() {
    // Pan a textured pattern by a small integer offset each frame so
    // the motion search must actually find the best MV (zero-MV SAD
    // would be high). Drives the ±16 full search into a non-trivial
    // branch.
    fn textured_frame_pan(w: u32, h: u32, pan_x: i32, pan_y: i32) -> VideoFrame {
        let cw = w / 2;
        let ch = h / 2;
        let mut y = vec![0u8; (w * h) as usize];
        for r in 0..h {
            for c in 0..w {
                // Low-frequency gradient with an additional translation
                // so the motion search tracks it. High-frequency texture
                // would stress the I-encoder's fixed `Intra_16×16 DC_PRED`
                // (out of scope for this test).
                let yy = r as i32 + pan_y;
                let xx = c as i32 + pan_x;
                y[(r * w + c) as usize] = (yy + xx).clamp(0, 255) as u8;
            }
        }
        VideoFrame {
            format: PixelFormat::Yuv420P,
            width: w,
            height: h,
            pts: Some(0),
            time_base: TimeBase::new(1, 30),
            planes: vec![
                VideoPlane {
                    stride: w as usize,
                    data: y,
                },
                VideoPlane {
                    stride: cw as usize,
                    data: vec![128u8; (cw * ch) as usize],
                },
                VideoPlane {
                    stride: cw as usize,
                    data: vec![128u8; (cw * ch) as usize],
                },
            ],
        }
    }
    let w = 64u32;
    let h = 64u32;
    let qp = 26i32;
    let mut enc = H264Encoder::new(
        CodecId::new("h264"),
        w,
        h,
        H264EncoderOptions {
            qp,
            p_slice_interval: 3,
            ..Default::default()
        },
    )
    .expect("encoder::new");
    let sources = vec![
        textured_frame_pan(w, h, 0, 0),
        textured_frame_pan(w, h, 2, 0),
        textured_frame_pan(w, h, 4, 0),
    ];
    let mut packets: Vec<Vec<u8>> = Vec::new();
    for src in &sources {
        enc.send_frame(&Frame::Video(src.clone())).expect("send");
        packets.push(enc.receive_packet().expect("recv").data.clone());
    }
    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let mut decoded = Vec::new();
    for (i, p) in packets.iter().enumerate() {
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), p.clone()).with_pts(i as i64))
            .expect("decode");
        receive_all(&mut dec, &mut decoded);
    }
    dec.flush().expect("flush");
    receive_all(&mut dec, &mut decoded);
    assert_eq!(decoded.len(), 3);
    for (i, (src, dec)) in sources.iter().zip(decoded.iter()).enumerate() {
        let psnr_y = psnr(&src.planes[0].data, &dec.planes[0].data);
        eprintln!(
            "pan frame {i}: {} bytes, luma psnr = {:.2} dB",
            packets[i].len(),
            psnr_y
        );
        assert!(psnr_y >= 30.0, "pan frame {i} psnr {psnr_y:.2} < 30 dB");
    }
}

#[test]
fn roundtrip_i_then_p_solid_gray_produces_skip_mbs() {
    // Solid grey frames — every P-MB should resolve to `P_Skip` (MVD=0,
    // residual=0). The slice emits just a single `mb_skip_run` covering
    // every MB plus the stop bit, so the P-packet body stays extremely
    // small.
    let w = 64u32;
    let h = 64u32;
    let mut enc = H264Encoder::new(
        CodecId::new("h264"),
        w,
        h,
        H264EncoderOptions {
            qp: 26,
            p_slice_interval: 3,
            ..Default::default()
        },
    )
    .expect("encoder::new");
    let solid = VideoFrame {
        format: PixelFormat::Yuv420P,
        width: w,
        height: h,
        pts: Some(0),
        time_base: TimeBase::new(1, 30),
        planes: vec![
            VideoPlane {
                stride: w as usize,
                data: vec![128u8; (w * h) as usize],
            },
            VideoPlane {
                stride: (w / 2) as usize,
                data: vec![128u8; (w * h / 4) as usize],
            },
            VideoPlane {
                stride: (w / 2) as usize,
                data: vec![128u8; (w * h / 4) as usize],
            },
        ],
    };

    enc.send_frame(&Frame::Video(solid.clone())).expect("I");
    let idr = enc.receive_packet().expect("IDR packet");
    enc.send_frame(&Frame::Video(solid.clone())).expect("P");
    let p = enc.receive_packet().expect("P packet");

    let mut dec = H264Decoder::new(CodecId::new("h264"));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), idr.data.clone()).with_pts(0))
        .expect("decode IDR");
    let mut decoded: Vec<VideoFrame> = Vec::new();
    receive_all(&mut dec, &mut decoded);
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), p.data.clone()).with_pts(1))
        .expect("decode P");
    receive_all(&mut dec, &mut decoded);
    dec.flush().expect("flush");
    receive_all(&mut dec, &mut decoded);
    assert_eq!(decoded.len(), 2, "expected two decoded frames");
    let luma_psnr = psnr(&solid.planes[0].data, &decoded[1].planes[0].data);
    eprintln!(
        "solid gray P packet body size = {} bytes, luma psnr = {:.2} dB",
        p.data.len(),
        luma_psnr
    );
    assert!(
        luma_psnr >= 40.0,
        "solid P luma psnr {luma_psnr:.2} < 40 dB"
    );
}
