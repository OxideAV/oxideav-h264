// Scratch harness: decode an Annex B stream with our decoder and
// compare byte-exact against an ffmpeg-produced planar YUV.
use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};

fn main() {
    let mut args = std::env::args().skip(1);
    let h264 = args.next().unwrap();
    let yuv = args.next().unwrap();
    let w: usize = args.next().unwrap().parse().unwrap();
    let h: usize = args.next().unwrap().parse().unwrap();
    let fmt = args.next().unwrap(); // "422" or "444"
    let (cw, ch) = match fmt.as_str() {
        "422" => (w / 2, h),
        "444" => (w, h),
        _ => (w / 2, h / 2),
    };
    let stream = std::fs::read(&h264).unwrap();
    let reference = std::fs::read(&yuv).unwrap();
    let mut dec = oxideav_h264::h264_decoder::H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let vf = loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => break vf,
            Ok(_) => continue,
            Err(e) => panic!("no frame: {e:?}"),
        }
    };
    let sizes = [(w, h), (cw, ch), (cw, ch)];
    let mut off = 0usize;
    let mut max_diff = 0u32;
    let mut n_diff = 0usize;
    for (p, &(pw, ph)) in sizes.iter().enumerate() {
        let plane = &vf.planes[p];
        for r in 0..ph {
            for c in 0..pw {
                let a = plane.data[r * plane.stride + c] as i32;
                let b = reference[off + r * pw + c] as i32;
                let d = a.abs_diff(b);
                if d > 0 {
                    n_diff += 1;
                }
                max_diff = max_diff.max(d);
            }
        }
        off += pw * ph;
    }
    println!("max_diff={max_diff} n_diff={n_diff} total={off}");
    assert_eq!(max_diff, 0, "mismatch vs reference decoder");
}
