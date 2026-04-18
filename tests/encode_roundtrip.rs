//! Round-trip the baseline I-only encoder through this crate's decoder.
//!
//! Each test:
//!   1. Synthesises a small YUV420P frame (gradient / solid / checker).
//!   2. Runs it through `H264Encoder` at a fixed QP.
//!   3. Feeds the resulting Annex B packet into `H264Decoder`.
//!   4. Checks that the decoded frame matches the input within a PSNR bound
//!      appropriate to the QP.
//!
//! The bound for the gradient fixture at QP=22 is PSNR ≥ 28 dB (luma).

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{
    frame::VideoPlane, CodecId, Frame, Packet, PixelFormat, TimeBase, VideoFrame,
};
use oxideav_h264::decoder::H264Decoder;
use oxideav_h264::encoder::{H264Encoder, H264EncoderOptions};

fn make_gradient_frame(w: u32, h: u32) -> VideoFrame {
    let mut y = vec![0u8; (w * h) as usize];
    let cw = w / 2;
    let ch = h / 2;
    let mut cb = vec![0u8; (cw * ch) as usize];
    let mut cr = vec![0u8; (cw * ch) as usize];
    for r in 0..h {
        for c in 0..w {
            // Diagonal gradient in luma.
            y[(r * w + c) as usize] = ((r + c).min(255)) as u8;
        }
    }
    for r in 0..ch {
        for c in 0..cw {
            cb[(r * cw + c) as usize] = (64 + r as u8).min(255);
            cr[(r * cw + c) as usize] = (192u32.saturating_sub(c) as u8).min(255);
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
                data: cb,
            },
            VideoPlane {
                stride: cw as usize,
                data: cr,
            },
        ],
    }
}

fn make_solid_frame(w: u32, h: u32, yv: u8, cbv: u8, crv: u8) -> VideoFrame {
    let cw = w / 2;
    let ch = h / 2;
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width: w,
        height: h,
        pts: Some(0),
        time_base: TimeBase::new(1, 30),
        planes: vec![
            VideoPlane {
                stride: w as usize,
                data: vec![yv; (w * h) as usize],
            },
            VideoPlane {
                stride: cw as usize,
                data: vec![cbv; (cw * ch) as usize],
            },
            VideoPlane {
                stride: cw as usize,
                data: vec![crv; (cw * ch) as usize],
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

fn encode_decode(frame: &VideoFrame, qp: i32) -> (Vec<u8>, VideoFrame) {
    let mut enc = H264Encoder::new(
        CodecId::new("h264"),
        frame.width,
        frame.height,
        H264EncoderOptions {
            qp,
            ..Default::default()
        },
    )
    .expect("encoder::new");
    enc.send_frame(&Frame::Video(frame.clone())).expect("send_frame");
    enc.flush().expect("flush");
    let pkt = enc.receive_packet().expect("receive_packet");
    let bytes = pkt.data.clone();

    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let dec_pkt = Packet::new(0, TimeBase::new(1, 30), bytes.clone()).with_pts(0);
    dec.send_packet(&dec_pkt).expect("decoder send_packet");
    let f = match dec.receive_frame().expect("receive_frame") {
        Frame::Video(v) => v,
        _ => panic!("expected video frame"),
    };
    (bytes, f)
}

#[test]
fn roundtrip_solid_mid_gray_64x48_qp22() {
    let src = make_solid_frame(64, 48, 128, 128, 128);
    let (bytes, dec) = encode_decode(&src, 22);
    assert_eq!(dec.width, 64);
    assert_eq!(dec.height, 48);
    let psnr_y = psnr(&src.planes[0].data, &dec.planes[0].data);
    eprintln!(
        "solid 64×48 qp22: {} bytes, luma psnr = {:.2} dB",
        bytes.len(),
        psnr_y
    );
    // A solid gray block has DC-only residuals — should encode losslessly
    // at mid-range QPs (bar rounding).
    assert!(psnr_y >= 40.0, "solid luma psnr {:.2} < 40 dB", psnr_y);
}

#[test]
fn roundtrip_gradient_64x48_qp22() {
    let src = make_gradient_frame(64, 48);
    let (bytes, dec) = encode_decode(&src, 22);
    assert_eq!(dec.width, 64);
    assert_eq!(dec.height, 48);
    let psnr_y = psnr(&src.planes[0].data, &dec.planes[0].data);
    let psnr_cb = psnr(&src.planes[1].data, &dec.planes[1].data);
    let psnr_cr = psnr(&src.planes[2].data, &dec.planes[2].data);
    eprintln!(
        "gradient 64×48 qp22: {} bytes, psnr Y={:.2} Cb={:.2} Cr={:.2} dB",
        bytes.len(),
        psnr_y,
        psnr_cb,
        psnr_cr
    );
    assert!(psnr_y >= 28.0, "gradient luma psnr {:.2} < 28 dB", psnr_y);
}

#[test]
fn roundtrip_gradient_128x96_qp28() {
    let src = make_gradient_frame(128, 96);
    let (bytes, dec) = encode_decode(&src, 28);
    assert_eq!(dec.width, 128);
    assert_eq!(dec.height, 96);
    let psnr_y = psnr(&src.planes[0].data, &dec.planes[0].data);
    eprintln!(
        "gradient 128×96 qp28: {} bytes, luma psnr = {:.2} dB",
        bytes.len(),
        psnr_y
    );
    // Larger gradient at higher QP — bound a bit lower.
    assert!(psnr_y >= 24.0, "gradient128 luma psnr {:.2} < 24 dB", psnr_y);
}

#[test]
fn non_mb_aligned_size_80x60_qp22() {
    // Requires SPS frame_cropping; width=80 is MB-aligned (80/16=5) but
    // height=60 needs crop bottom 2*2 = 4 samples.
    let src = make_gradient_frame(80, 60);
    let (bytes, dec) = encode_decode(&src, 22);
    assert_eq!(dec.width, 80);
    assert_eq!(dec.height, 60);
    let psnr_y = psnr(&src.planes[0].data, &dec.planes[0].data);
    eprintln!(
        "gradient 80×60 qp22 (cropped): {} bytes, luma psnr = {:.2} dB",
        bytes.len(),
        psnr_y
    );
    assert!(psnr_y >= 26.0, "cropped luma psnr {:.2} < 26 dB", psnr_y);
}
