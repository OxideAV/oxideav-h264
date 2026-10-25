// Decode-speed harness: feed an Annex B stream to the decoder one NAL
// unit per packet (the way a demuxer hands over data), drain frames as
// they become ready, and report wall time / fps. Optionally dump every
// output frame as tightly packed planar YUV for a framemd5 comparison
// against a black-box reference decoder.
//
// Usage: decode_bench <input.h264> [threads] [out.yuv]
use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, ExecutionContext, Frame, Packet, TimeBase};
use std::io::Write as _;
use std::time::Instant;

fn split_nals(data: &[u8]) -> Vec<&[u8]> {
    // Return each NAL unit together with its leading start code.
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            let s = if i > 0 && data[i - 1] == 0 { i - 1 } else { i };
            starts.push(s);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for (k, &s) in starts.iter().enumerate() {
        let e = starts.get(k + 1).copied().unwrap_or(data.len());
        out.push(&data[s..e]);
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let input = args
        .next()
        .expect("usage: decode_bench <input.h264> [threads] [out.yuv]");
    let threads: usize = args.next().map_or(1, |t| t.parse().expect("threads"));
    let out = args.next();
    let stream = std::fs::read(&input).expect("read input");
    let nals = split_nals(&stream);
    let mut sink = out.map(|p| std::io::BufWriter::new(std::fs::File::create(p).unwrap()));
    let mut dec = oxideav_h264::h264_decoder::H264CodecDecoder::new(CodecId::new("h264"));
    dec.set_execution_context(&ExecutionContext::with_threads(threads));
    let tb = TimeBase::new(1, 30);
    let mut frames = 0usize;
    let mut emit = |f: Frame, frames: &mut usize| {
        if let Frame::Video(v) = f {
            if let Some(s) = sink.as_mut() {
                for p in &v.planes {
                    s.write_all(&p.data).unwrap();
                }
            }
            *frames += 1;
        }
    };
    let t0 = Instant::now();
    for (i, nal) in nals.iter().enumerate() {
        let pkt = Packet::new(0, tb, nal.to_vec()).with_pts(i as i64);
        dec.send_packet(&pkt).expect("send_packet");
        while let Ok(f) = dec.receive_frame() {
            emit(f, &mut frames);
        }
    }
    dec.flush().expect("flush");
    while let Ok(f) = dec.receive_frame() {
        emit(f, &mut frames);
    }
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "threads={threads} frames={frames} wall={dt:.3}s fps={:.1}",
        frames as f64 / dt
    );
}
