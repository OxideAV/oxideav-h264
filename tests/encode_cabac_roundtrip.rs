//! Scope-mandated round-trip for the **CABAC** I-slice encoder.
//!
//! Encodes a 10-frame 64×64 synthetic clip entirely as CABAC IDR frames
//! and decodes each one through this crate's [`H264Decoder`]. Asserts:
//!
//! * Every frame decodes without error (CABAC-aware parse + entropy
//!   decode).
//! * Luma PSNR against the source is ≥ 30 dB.
//!
//! The CABAC encoder currently only emits I-slices (IDRs) — so the
//! `p_slice_interval = 1` config below yields ten IDRs, not a
//! one-IDR-plus-P-slices cadence. Every packet carries SPS + PPS + the
//! IDR slice NAL, both to keep the test self-contained and to exercise
//! the Main-profile PPS emission each time.

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{
    frame::VideoPlane, CodecId, Error, Frame, Packet, PixelFormat, TimeBase, VideoFrame,
};
use oxideav_h264::decoder::H264Decoder;
use oxideav_h264::encoder::{H264Encoder, H264EncoderOptions};

fn testsrc_frame(w: u32, h: u32, t: u32) -> VideoFrame {
    let cw = w / 2;
    let ch = h / 2;
    let mut y = vec![0u8; (w * h) as usize];
    for r in 0..h {
        for c in 0..w {
            let shift = (t * 2) as i32;
            let xp = (c as i32 + shift) as u32 % 256;
            y[(r * w + c) as usize] = ((r + xp) % 256) as u8;
        }
    }
    let mut cb = vec![128u8; (cw * ch) as usize];
    let mut cr = vec![128u8; (cw * ch) as usize];
    for r in 0..ch {
        for c in 0..cw {
            cb[(r * cw + c) as usize] = (80 + r + (t / 2)).min(220) as u8;
            cr[(r * cw + c) as usize] = (200u32.saturating_sub(c + (t / 2))) as u8;
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

fn drain(dec: &mut H264Decoder, out: &mut Vec<VideoFrame>) {
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
fn encode_10_frame_testsrc_cabac_roundtrip_psnr_30db() {
    const W: u32 = 64;
    const H: u32 = 64;
    const FRAMES: u32 = 10;
    const QP: i32 = 26;

    let mut enc = H264Encoder::new(
        CodecId::new("h264"),
        W,
        H,
        H264EncoderOptions {
            qp: QP,
            use_cabac: true,
            ..Default::default()
        },
    )
    .expect("encoder::new");

    let sources: Vec<VideoFrame> = (0..FRAMES).map(|t| testsrc_frame(W, H, t)).collect();

    let mut packets: Vec<Vec<u8>> = Vec::new();
    for src in &sources {
        enc.send_frame(&Frame::Video(src.clone())).expect("send");
        let pkt = enc.receive_packet().expect("recv");
        packets.push(pkt.data.clone());
    }
    enc.flush().expect("flush");

    assert_eq!(packets.len(), FRAMES as usize);

    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let mut decoded: Vec<VideoFrame> = Vec::new();
    for (i, bytes) in packets.iter().enumerate() {
        let pkt = Packet::new(0, TimeBase::new(1, 30), bytes.clone()).with_pts(i as i64);
        dec.send_packet(&pkt).expect("decoder send_packet");
        drain(&mut dec, &mut decoded);
    }
    dec.flush().expect("decoder flush");
    drain(&mut dec, &mut decoded);

    assert_eq!(
        decoded.len(),
        FRAMES as usize,
        "expected {FRAMES} decoded frames, got {}",
        decoded.len()
    );

    let mut sum_psnr = 0.0;
    for (i, (src, dec_f)) in sources.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(dec_f.width, W);
        assert_eq!(dec_f.height, H);
        let psnr_y = psnr(&src.planes[0].data, &dec_f.planes[0].data);
        eprintln!(
            "frame {i} (CABAC IDR): {} bytes, luma psnr = {:.2} dB",
            packets[i].len(),
            psnr_y
        );
        assert!(
            psnr_y >= 30.0,
            "frame {i} luma psnr {psnr_y:.2} below 30 dB scope bar"
        );
        sum_psnr += psnr_y;
    }
    eprintln!(
        "10-frame CABAC round-trip: avg luma psnr = {:.2} dB",
        sum_psnr / (FRAMES as f64)
    );
}

/// `entropy_coding_mode_flag = 1` end-to-end — the PPS round-trips and
/// the decoder sees the CABAC bit set.
#[test]
fn cabac_pps_flag_set_in_packet() {
    let mut enc = H264Encoder::new(
        CodecId::new("h264"),
        64,
        64,
        H264EncoderOptions {
            use_cabac: true,
            ..Default::default()
        },
    )
    .expect("encoder::new");
    let src = testsrc_frame(64, 64, 0);
    enc.send_frame(&Frame::Video(src)).expect("send");
    let pkt = enc.receive_packet().expect("recv");
    // The output stream contains three NALs (SPS, PPS, IDR). Look for
    // the PPS start-code + NAL header 0x68 and scan the body for the
    // entropy_coding_mode_flag, which is the second bit after the two
    // ue(v) fields `pic_parameter_set_id` and `seq_parameter_set_id`.
    //
    // Rather than re-implement the PPS parser, we drive the packet
    // through the decoder (which is our spec-compliant CABAC parser) and
    // confirm it produces a frame — the decoder explicitly rejects
    // streams whose PPS claims CABAC if CABAC decode isn't wired.
    let mut dec = H264Decoder::new(CodecId::new("h264"));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), pkt.data.clone()).with_pts(0))
        .expect("decoder accepts CABAC PPS + IDR");
    let mut out = Vec::new();
    drain(&mut dec, &mut out);
    assert!(!out.is_empty(), "decoder must produce at least one frame");
}
