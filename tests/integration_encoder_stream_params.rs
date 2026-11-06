//! The registry encoder's answer to `Frame::StreamParams` (oxideav-core
//! 0.1.43): with Annex B output it re-configures for the new input
//! shape — the old configuration's buffered tail is drained, the next
//! frame opens a new coded video sequence (IDR + in-band SPS / PPS,
//! `idr_pic_id` continuing the old sequence's), `pts` / `dts` keep
//! counting, and `output_params()` describes the new output. Shapes no
//! coding path takes and `format=avcc` (parameter sets fixed in the
//! extradata record) are `Error::Unsupported`, leaving the encoder as
//! it was.
//!
//! The spliced product is decoded by the crate's decoder, which must
//! announce each change with its own event at the same picture.

use oxideav_core::{
    CodecId, CodecParameters, Decoder, Encoder, Error, Frame, Packet, PixelFormat, Rational,
    VideoFrame, VideoPlane,
};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264::h264_encoder::make_encoder;

fn params(w: u32, h: u32, pf: PixelFormat) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("h264"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(pf);
    p.frame_rate = Some(Rational::new(25, 1));
    p
}

fn event(w: u32, h: u32, pf: PixelFormat) -> Frame {
    Frame::StreamParams(Box::new(params(w, h, pf)))
}

/// A synthetic picture `n` of `w` x `h` in `pf` (4:2:0, 8 or 10 bit).
fn picture(n: usize, w: u32, h: u32, pf: PixelFormat) -> Frame {
    let (w, h) = (w as usize, h as usize);
    let wide = pf == PixelFormat::Yuv420P10Le;
    let plane = |pw: usize, ph: usize, seed: usize| -> VideoPlane {
        let mut data = Vec::new();
        for j in 0..ph {
            for i in 0..pw {
                let v = 40 + (i * 3 + j * 2 + n * 5 + seed) % 160;
                if wide {
                    data.extend_from_slice(&((v * 4) as u16).to_le_bytes());
                } else {
                    data.push(v as u8);
                }
            }
        }
        VideoPlane {
            stride: pw * if wide { 2 } else { 1 },
            data,
        }
    };
    Frame::Video(VideoFrame {
        pts: Some(n as i64),
        planes: vec![
            plane(w, h, 0),
            plane(w / 2, h / 2, 30),
            plane(w / 2, h / 2, 70),
        ],
    })
}

fn drain(enc: &mut dyn Encoder, out: &mut Vec<Packet>) {
    loop {
        match enc.receive_packet() {
            Ok(p) => out.push(p),
            Err(Error::NeedMore) => return,
            Err(e) => panic!("receive_packet: {e:?}"),
        }
    }
}

/// `(shape, pictures)` of a decode: one entry per event / picture.
#[derive(Debug, PartialEq)]
enum Out {
    Event(Option<u32>, Option<u32>, Option<PixelFormat>),
    Picture(Vec<Vec<u8>>),
}

fn decode(packets: &[Packet]) -> Vec<Out> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let mut out = Vec::new();
    let pull = |dec: &mut H264CodecDecoder, out: &mut Vec<Out>| {
        while let Ok(f) = dec.receive_frame() {
            match f {
                Frame::StreamParams(p) => out.push(Out::Event(p.width, p.height, p.pixel_format)),
                Frame::Video(v) => {
                    out.push(Out::Picture(v.planes.into_iter().map(|p| p.data).collect()))
                }
                _ => panic!("unexpected frame"),
            }
        }
    };
    for p in packets {
        dec.send_packet(p).expect("send");
        pull(&mut dec, &mut out);
    }
    dec.flush().expect("flush");
    pull(&mut dec, &mut out);
    out
}

/// Packets that open a coded video sequence (IDR access units).
fn idr_count(packets: &[Packet]) -> usize {
    packets.iter().filter(|p| p.flags.keyframe).count()
}

/// Encode `segments` (shape, frame count) with one event between
/// consecutive shapes.
fn encode_spliced(
    mut base: CodecParameters,
    opts: &[(&str, &str)],
    segments: &[(u32, u32, PixelFormat, usize)],
) -> (Vec<Packet>, Vec<CodecParameters>) {
    for (k, v) in opts {
        base.options.insert(*k, *v);
    }
    let mut enc = make_encoder(&base).expect("encoder");
    let mut packets = Vec::new();
    let mut seen = Vec::new();
    let mut n = 0;
    for (k, &(w, h, pf, count)) in segments.iter().enumerate() {
        if k > 0 {
            enc.send_frame(&event(w, h, pf)).expect("re-configure");
        }
        seen.push(enc.output_params().clone());
        for _ in 0..count {
            enc.send_frame(&picture(n, w, h, pf)).expect("send");
            drain(&mut *enc, &mut packets);
            n += 1;
        }
    }
    enc.flush().expect("flush");
    drain(&mut *enc, &mut packets);
    (packets, seen)
}

/// Reference: each segment encoded by a fresh encoder, then decoded.
fn reference(opts: &[(&str, &str)], segments: &[(u32, u32, PixelFormat, usize)]) -> Vec<Out> {
    let mut out = Vec::new();
    let mut n0 = 0;
    for (k, &(w, h, pf, count)) in segments.iter().enumerate() {
        let mut p = params(w, h, pf);
        for (key, v) in opts {
            p.options.insert(*key, *v);
        }
        let mut enc = make_encoder(&p).expect("encoder");
        let mut packets = Vec::new();
        for n in n0..n0 + count {
            enc.send_frame(&picture(n, w, h, pf)).expect("send");
            drain(&mut *enc, &mut packets);
        }
        enc.flush().expect("flush");
        drain(&mut *enc, &mut packets);
        if k > 0 {
            out.push(Out::Event(Some(w), Some(h), Some(pf)));
        }
        out.extend(decode(&packets));
        n0 += count;
    }
    out
}

#[test]
fn reconfigures_across_size_and_bit_depth() {
    let segs = [
        (64, 48, PixelFormat::Yuv420P, 5),
        (96, 64, PixelFormat::Yuv420P10Le, 5),
        (48, 32, PixelFormat::Yuv420P, 5),
    ];
    let (packets, seen) = encode_spliced(params(64, 48, PixelFormat::Yuv420P), &[], &segs);
    for (p, &(w, h, pf, _)) in seen.iter().zip(&segs) {
        assert_eq!(
            (p.width, p.height, p.pixel_format),
            (Some(w), Some(h), Some(pf))
        );
    }
    assert_eq!(packets.len(), 15);
    // A new coded video sequence at each change.
    assert!(packets[0].flags.keyframe && packets[5].flags.keyframe && packets[10].flags.keyframe);
    let pts: Vec<i64> = packets.iter().map(|p| p.pts.unwrap()).collect();
    assert_eq!(pts, (0..15).collect::<Vec<_>>());
    // Decoded: one event per change at the same picture, and every
    // picture identical to a fresh encode of its segment.
    let got = decode(&packets);
    assert_eq!(got, reference(&[], &segs));
    assert_eq!(
        got.iter().filter(|o| matches!(o, Out::Event(..))).count(),
        2
    );
}

#[test]
fn b_frame_tail_drains_before_the_new_sequence() {
    let opts = [("bframes", "2"), ("gop", "4")];
    let segs = [
        (64, 48, PixelFormat::Yuv420P, 7),
        (80, 48, PixelFormat::Yuv420P, 7),
    ];
    let mut base = params(64, 48, PixelFormat::Yuv420P);
    base.frame_rate = Some(Rational::new(25, 1));
    let (packets, _) = encode_spliced(base, &opts, &segs);
    assert_eq!(packets.len(), 14);
    let dts: Vec<i64> = packets.iter().map(|p| p.dts.unwrap()).collect();
    assert_eq!(dts, (0..14).collect::<Vec<_>>(), "dts keeps counting");
    let mut pts: Vec<i64> = packets.iter().map(|p| p.pts.unwrap()).collect();
    assert!(packets.iter().all(|p| p.pts >= p.dts));
    // The first 7 packets are the whole first configuration (its tail
    // included); the 8th opens the new sequence.
    assert!(packets[7].flags.keyframe);
    assert!(packets[..7].iter().all(|p| p.pts.unwrap() <= 7));
    pts.sort();
    assert_eq!(pts, (1..=14).collect::<Vec<_>>(), "pts keeps counting");
    assert_eq!(decode(&packets), reference(&opts, &segs));
    // gop=4: IDRs at frames 0 and 4 of each configuration.
    assert_eq!(idr_count(&packets), 4);
}

#[test]
fn unsupported_shapes_leave_the_encoder_unchanged() {
    let mut p = params(64, 48, PixelFormat::Yuv420P);
    p.options.insert("bframes", "2");
    let mut enc = make_encoder(&p).expect("encoder");
    let mut packets = Vec::new();
    for n in 0..3 {
        enc.send_frame(&picture(n, 64, 48, PixelFormat::Yuv420P))
            .unwrap();
        drain(&mut *enc, &mut packets);
    }
    // B pictures are an 8-bit 4:2:0 tool: no 10-bit configuration.
    assert!(matches!(
        enc.send_frame(&event(64, 48, PixelFormat::Yuv420P10Le)),
        Err(Error::Unsupported(_))
    ));
    // Not 16-aligned.
    assert!(matches!(
        enc.send_frame(&event(70, 46, PixelFormat::Yuv420P)),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(enc.output_params().width, Some(64));
    for n in 3..6 {
        enc.send_frame(&picture(n, 64, 48, PixelFormat::Yuv420P))
            .unwrap();
        drain(&mut *enc, &mut packets);
    }
    enc.flush().unwrap();
    drain(&mut *enc, &mut packets);
    let out = decode(&packets);
    assert_eq!(out.len(), 6, "one sequence, no event");
    assert!(out.iter().all(|o| matches!(o, Out::Picture(_))));
}

#[test]
fn avcc_rejects_and_an_unchanged_shape_is_a_no_op() {
    let mut p = params(64, 48, PixelFormat::Yuv420P);
    p.options.insert("format", "avcc");
    let mut enc = make_encoder(&p).expect("encoder");
    let extradata = enc.output_params().extradata.clone();
    assert!(matches!(
        enc.send_frame(&event(96, 64, PixelFormat::Yuv420P)),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(enc.output_params().extradata, extradata);
    // The current shape: nothing to do, even for avcC.
    enc.send_frame(&event(64, 48, PixelFormat::Yuv420P))
        .expect("no-op");

    let mut enc = make_encoder(&params(64, 48, PixelFormat::Yuv420P)).expect("encoder");
    let mut packets = Vec::new();
    enc.send_frame(&picture(0, 64, 48, PixelFormat::Yuv420P))
        .unwrap();
    enc.send_frame(&event(64, 48, PixelFormat::Yuv420P))
        .unwrap();
    enc.send_frame(&picture(1, 64, 48, PixelFormat::Yuv420P))
        .unwrap();
    drain(&mut *enc, &mut packets);
    assert_eq!(packets.len(), 2);
    assert!(!packets[1].flags.keyframe, "no new sequence for no change");
}

#[test]
fn all_intra_frame_rate_change_continues_idr_pic_id() {
    // gop=1: every access unit is an IDR. A frame-rate-only change
    // still opens a new sequence; its first IDR must not repeat the
    // idr_pic_id of the IDR just before it (§7.4.3), or the two would
    // read as one picture (§7.4.1.2.4).
    let mut p = params(64, 48, PixelFormat::Yuv420P);
    p.options.insert("gop", "1");
    let mut enc = make_encoder(&p).expect("encoder");
    let mut packets = Vec::new();
    // One IDR (idr_pic_id 0) before the change: a restarted sequence
    // would open with 0 again.
    enc.send_frame(&picture(0, 64, 48, PixelFormat::Yuv420P))
        .unwrap();
    drain(&mut *enc, &mut packets);
    let mut ev = params(64, 48, PixelFormat::Yuv420P);
    ev.frame_rate = Some(Rational::new(50, 1));
    enc.send_frame(&Frame::StreamParams(Box::new(ev))).unwrap();
    assert_eq!(enc.output_params().frame_rate, Some(Rational::new(50, 1)));
    for n in 1..4 {
        enc.send_frame(&picture(n, 64, 48, PixelFormat::Yuv420P))
            .unwrap();
        drain(&mut *enc, &mut packets);
    }
    assert_eq!(idr_count(&packets), 4);
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let (mut pictures, mut events) = (0, 0);
    // One packet for all, so only §7.4.1.2.4 separates the pictures.
    let mut all = Vec::new();
    for pk in &packets {
        all.extend_from_slice(&pk.data);
    }
    dec.send_packet(&Packet::new(0, packets[0].time_base, all))
        .unwrap();
    dec.flush().unwrap();
    while let Ok(f) = dec.receive_frame() {
        if f.is_stream_params() {
            events += 1;
        } else {
            pictures += 1;
        }
    }
    // The encoder's SPS carries no VUI timing, so the decoded shape —
    // and with it the decoder's answer — does not change: no event.
    assert_eq!((pictures, events), (4, 0));
}
