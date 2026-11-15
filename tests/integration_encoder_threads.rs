//! The encoder's worker budget never changes the stream: every packet
//! of every option set is byte-identical at 1, 2, 3 and 8 threads, and
//! the budget survives a mid-stream re-configuration.

use oxideav_core::{
    CodecId, CodecParameters, Error, ExecutionContext, Frame, Packet, PixelFormat, Rational,
    VideoFrame, VideoPlane,
};
use oxideav_h264::encoder::{Encoder, EncoderConfig, YuvFrame};

fn planes(w: usize, h: usize, n: usize, gray: bool) -> Vec<VideoPlane> {
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            // A textured field that translates by (2n, n) with a block
            // that moves the other way: real motion for the search.
            let (sx, sy) = (i + 2 * n, j + n);
            let base = (sx * 5 + sy * 3 + (sx / 7) * (sy / 5) * 11) % 160;
            let blk = if (i + 64 - 3 * n) % 64 < 20 && j % 48 < 20 {
                70
            } else {
                0
            };
            y[j * w + i] = (20 + base + blk) as u8;
        }
    }
    let mut out = vec![VideoPlane { stride: w, data: y }];
    if !gray {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        for k in 0..2usize {
            let mut c = vec![0u8; cw * ch];
            for j in 0..ch {
                for i in 0..cw {
                    c[j * cw + i] = (90 + ((i + n * (k + 1) + j * (2 - k)) % 70)) as u8;
                }
            }
            out.push(VideoPlane {
                stride: cw,
                data: c,
            });
        }
    }
    out
}

fn encode(
    w: u32,
    h: u32,
    pf: PixelFormat,
    opts: &[(&str, &str)],
    frames: usize,
    threads: usize,
) -> Vec<Packet> {
    let mut p = CodecParameters::video(CodecId::new("h264"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(pf);
    p.frame_rate = Some(Rational::new(30, 1));
    let mut o = oxideav_core::CodecOptions::new();
    for (k, v) in opts {
        o = o.set(*k, *v);
    }
    p.options = o;
    let mut enc = oxideav_h264::h264_encoder::make_encoder(&p).unwrap();
    enc.set_execution_context(&ExecutionContext::with_threads(threads));
    let mut packets = Vec::new();
    let drain = |enc: &mut dyn oxideav_core::Encoder, packets: &mut Vec<Packet>| loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(p),
            Err(Error::NeedMore) => break,
            Err(e) => panic!("receive_packet: {e}"),
        }
    };
    for n in 0..frames {
        enc.send_frame(&Frame::Video(VideoFrame {
            pts: Some(n as i64),
            planes: planes(w as usize, h as usize, n, pf == PixelFormat::Gray8),
        }))
        .unwrap();
        drain(&mut *enc, &mut packets);
    }
    enc.flush().unwrap();
    drain(&mut *enc, &mut packets);
    packets
}

fn assert_same(a: &[Packet], b: &[Packet], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: packet count");
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.data, y.data, "{what}: packet {k} bytes");
        assert_eq!((x.pts, x.dts), (y.pts, y.dts), "{what}: packet {k} times");
    }
}

#[test]
fn registry_streams_are_identical_at_every_budget() {
    type Case = (
        u32,
        u32,
        PixelFormat,
        &'static [(&'static str, &'static str)],
        usize,
    );
    let cases: &[Case] = &[
        (96, 80, PixelFormat::Yuv420P, &[("qp", "24")], 4),
        (
            96,
            80,
            PixelFormat::Yuv420P,
            &[("qp", "24"), ("cabac", "true")],
            4,
        ),
        (
            96,
            80,
            PixelFormat::Yuv420P,
            &[("qp", "28"), ("bframes", "2")],
            8,
        ),
        (
            96,
            80,
            PixelFormat::Yuv420P,
            &[
                ("qp", "28"),
                ("cabac", "true"),
                ("bframes", "3"),
                ("gop", "6"),
            ],
            11,
        ),
        // A size off the macroblock grid (padded + cropped).
        (
            90,
            70,
            PixelFormat::Yuv420P,
            &[("qp", "30"), ("cabac", "true"), ("bframes", "2")],
            6,
        ),
        // 4:0:0 through the same session.
        (
            96,
            80,
            PixelFormat::Gray8,
            &[("qp", "26"), ("cabac", "true"), ("bframes", "2")],
            6,
        ),
        // Rate control: B pictures stay in coding order, the motion
        // analysis still spreads.
        (
            96,
            80,
            PixelFormat::Yuv420P,
            &[
                ("rc", "cbr"),
                ("bitrate", "200000"),
                ("cabac", "true"),
                ("bframes", "2"),
            ],
            8,
        ),
    ];
    for (w, h, pf, opts, frames) in cases {
        let serial = encode(*w, *h, *pf, opts, *frames, 1);
        assert_eq!(serial.len(), *frames);
        for threads in [2usize, 3, 8] {
            let par = encode(*w, *h, *pf, opts, *frames, threads);
            assert_same(
                &serial,
                &par,
                &format!("{w}x{h} {pf:?} {opts:?} threads={threads}"),
            );
        }
    }
}

/// The stateless picture encoders under `EncoderConfig::threads`.
#[test]
fn picture_encoders_are_identical_at_every_budget() {
    let (w, h) = (96u32, 64u32);
    let pic = |n: usize| planes(w as usize, h as usize, n, false);
    let (p0, p1, p2) = (pic(0), pic(2), pic(1));
    let yuv = |p: &'_ [VideoPlane]| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        (p[0].data.clone(), p[1].data.clone(), p[2].data.clone())
    };
    let (f0, f1, f2) = (yuv(&p0), yuv(&p1), yuv(&p2));
    fn frame(f: &(Vec<u8>, Vec<u8>, Vec<u8>), w: u32, h: u32) -> YuvFrame<'_> {
        YuvFrame {
            width: w,
            height: h,
            y: &f.0,
            u: &f.1,
            v: &f.2,
        }
    }
    let frame = |f| frame(f, w, h);
    for cabac in [false, true] {
        let run = |threads: usize| -> Vec<Vec<u8>> {
            let cfg = EncoderConfig::new(w, h)
                .with_profile_idc(77)
                .with_max_num_ref_frames(2)
                .with_cabac(cabac)
                .with_threads(threads);
            let enc = Encoder::new(cfg);
            let (idr, p, b);
            if cabac {
                idr = enc.encode_idr_cabac(&frame(&f0));
                p = enc.encode_p_cabac(&frame(&f1), &(&idr).into(), 1, 4);
                b = enc.encode_b_cabac(&frame(&f2), &(&idr).into(), &(&p).into(), 2, 2);
            } else {
                idr = enc.encode_idr(&frame(&f0));
                p = enc.encode_p(&frame(&f1), &(&idr).into(), 1, 4);
                b = enc.encode_b(&frame(&f2), &(&idr).into(), &(&p).into(), 2, 2);
            }
            vec![idr.annex_b, p.annex_b, b.annex_b]
        };
        let serial = run(1);
        for threads in [0usize, 2, 5, 16] {
            assert_eq!(run(threads), serial, "cabac={cabac} threads={threads}");
        }
    }
}
