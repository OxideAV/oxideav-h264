// Scratch harness: decode an Annex B stream with our decoder and dump
// every visible frame as tightly-packed planar YUV (Y || U || V, any
// per-row stride padding stripped), concatenated in output order —
// the same layout a black-box reference decoder emits with
// `-f rawvideo`, so the two files can be diffed byte-for-byte.
//
// Usage: dump_yuv <input.h264> <out.yuv>
use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use std::io::Write as _;

fn main() {
    let mut args = std::env::args().skip(1);
    let h264 = args.next().expect("usage: dump_yuv <input.h264> <out.yuv>");
    let out = args.next().expect("usage: dump_yuv <input.h264> <out.yuv>");
    let stream = std::fs::read(&h264).unwrap();
    let mut dec = oxideav_h264::h264_decoder::H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    let mut n_frames = 0usize;
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => {
                for (pi, plane) in vf.planes.iter().enumerate() {
                    let rows = plane.data.len() / plane.stride.max(1);
                    // The visible byte-width per row: our decoder packs
                    // planes tightly (stride == visible width in bytes),
                    // so dump entire rows.
                    for r in 0..rows {
                        f.write_all(&plane.data[r * plane.stride..(r + 1) * plane.stride])
                            .unwrap();
                    }
                    eprintln!(
                        "frame {n_frames} plane {pi}: stride={} rows={rows}",
                        plane.stride
                    );
                }
                n_frames += 1;
            }
            Ok(_) => continue,
            Err(oxideav_core::Error::NeedMore) | Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_frame after {n_frames} frames: {e:?}"),
        }
    }
    eprintln!("dumped {n_frames} frames to {out}");
}
