// Encode-speed harness: read tightly packed 8-bit 4:2:0 frames, push
// them through the registry encoder and report wall time / fps and the
// stream size.
//
// Usage: encode_bench <input.yuv> <width> <height> [key=value ...] [out.h264]
// e.g.   encode_bench clip.yuv 1920 1080 qp=26 cabac=true bframes=2
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, PixelFormat, Rational, VideoFrame, VideoPlane,
};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: encode_bench <input.yuv> <width> <height> [key=value ...] [out.h264]";
    let input = args.next().expect(usage);
    let w: usize = args.next().expect(usage).parse().expect("width");
    let h: usize = args.next().expect(usage).parse().expect("height");
    let mut params = CodecParameters::video(CodecId::new("h264"));
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.frame_rate = Some(Rational::new(30, 1));
    let mut options = oxideav_core::CodecOptions::new();
    let mut out = None;
    for a in args {
        match a.split_once('=') {
            Some((k, v)) => options = options.set(k, v),
            None => out = Some(a),
        }
    }
    params.options = options;
    let data = std::fs::read(&input).expect("read input");
    let frame_len = w * h * 3 / 2;
    let mut enc = oxideav_h264::h264_encoder::make_encoder(&params).expect("encoder");
    let mut stream = Vec::new();
    let drain = |enc: &mut dyn oxideav_core::Encoder, stream: &mut Vec<u8>| loop {
        match enc.receive_packet() {
            Ok(p) => stream.extend_from_slice(&p.data),
            Err(Error::NeedMore) => break,
            Err(e) => panic!("receive_packet: {e}"),
        }
    };
    let frames = data.len() / frame_len;
    let t0 = Instant::now();
    for (n, f) in data.chunks_exact(frame_len).enumerate() {
        let (y, c) = f.split_at(w * h);
        let (u, v) = c.split_at(w * h / 4);
        let plane = |d: &[u8], stride: usize| VideoPlane {
            stride,
            data: d.to_vec(),
        };
        enc.send_frame(&Frame::Video(VideoFrame {
            pts: Some(n as i64),
            planes: vec![plane(y, w), plane(u, w / 2), plane(v, w / 2)],
        }))
        .expect("send_frame");
        drain(&mut *enc, &mut stream);
    }
    enc.flush().expect("flush");
    drain(&mut *enc, &mut stream);
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "{frames} frames in {dt:.3} s = {:.2} fps, {} bytes",
        frames as f64 / dt,
        stream.len()
    );
    if let Some(p) = out {
        std::fs::write(p, &stream).expect("write output");
    }
}
