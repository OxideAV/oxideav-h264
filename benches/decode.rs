//! End-to-end H.264 decode benchmark.
//!
//! Existing benches (`inter_pred_bench`, `transform_bench`) measure
//! isolated kernels; this one drives the full
//! [`H264CodecDecoder`](oxideav_h264::h264_decoder::H264CodecDecoder)
//! through `send_packet` + `flush` + `receive_frame` so we get a
//! representative wall-clock cost for the public decode path. It also
//! lets us notice whole-pipeline regressions (slice header parsing,
//! DPB output ordering, deblocking sweep, plane materialisation) that
//! the kernel benches don't surface.
//!
//! The bench corpus is built **in-process** from our own encoder so
//! the bench is self-contained — no on-disk fixtures, no test-data
//! env vars, runs in CI without extra plumbing.
//!
//! Variants (each is a small but representative slice of the surface):
//!
//! | label              | frames    | profile        | exercises                                                   |
//! | ------------------ | --------- | -------------- | ----------------------------------------------------------- |
//! | `idr_only_64x64`   | 1 IDR     | Baseline 4:2:0 | SPS+PPS+IDR parse, intra recon, deblocking, plane materialise |
//! | `idr_only_128x96`  | 1 IDR     | Baseline 4:2:0 | larger picture so MB-grid overhead is visible                |
//! | `idr_p4_64x64`     | IDR+4×P   | Baseline 4:2:0 | inter MC (§8.4.2) on a chain of P references                 |
//! | `idr_p_b_64x64`    | IDR+P+B   | Main 4:2:0     | spatial-direct B (§8.4.1.2.2) on top of a P anchor           |
//! | `idr_only_422`     | 1 IDR     | High 4:2:2     | chroma-array-type 2 plane sizing + deblock                   |
//!
//! Each variant is benched at the picture-throughput element rate
//! (frames per iteration) so Criterion reports `elem/s` directly as
//! decoded-frames-per-second. Per-byte throughput is also available
//! in the Criterion report (`Throughput::Bytes`).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

/// Synthesize a smooth-gradient 4:2:0 picture of the given size.
///
/// Same shape as the existing integration-test sources: luma is a
/// diagonal gradient, chroma is mid-grey. Bench inputs that have any
/// content (vs. a constant plane) keep the entropy coder honest.
fn make_yuv420(w: usize, h: usize, phase: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    assert!(w % 16 == 0 && h % 16 == 0);
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            let g = 16 + (i + j + phase) * (240 - 16) / (w + h + 16);
            y[j * w + i] = g.clamp(0, 255) as u8;
        }
    }
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];
    (y, u, v)
}

/// Bipred-friendly midpoint of two reference luma planes. Same idea
/// as `integration_b_slice.rs::make_b` — every MB then prefers the
/// bipred candidate and we exercise the §8.4.2.3.1 average path.
fn make_b_yuv420(
    w: usize,
    h: usize,
    a: &(Vec<u8>, Vec<u8>, Vec<u8>),
    b: &(Vec<u8>, Vec<u8>, Vec<u8>),
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; w * h];
    for (k, yk) in y.iter_mut().enumerate() {
        let av = a.0[k] as u32;
        let bv = b.0[k] as u32;
        *yk = ((av + bv + 1) >> 1) as u8;
    }
    let u = vec![128u8; (w / 2) * (h / 2)];
    let v = vec![128u8; (w / 2) * (h / 2)];
    (y, u, v)
}

/// 4:2:2 (`chroma_format_idc=2`) picture — chroma is `(W/2, H)`.
fn make_yuv422(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    assert!(w % 16 == 0 && h % 16 == 0);
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; (w / 2) * h];
    let mut v = vec![0u8; (w / 2) * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = (16 + (i + j) * (240 - 16) / (w + h)).clamp(0, 255) as u8;
        }
    }
    for j in 0..h {
        for i in 0..(w / 2) {
            u[j * (w / 2) + i] = (64 + i * 128 / (w / 2 - 1)) as u8;
            v[j * (w / 2) + i] = (64 + j * 128 / (h - 1)) as u8;
        }
    }
    (y, u, v)
}

/// Concatenate one IDR's worth of Annex B bytes.
fn build_idr_only(w: u32, h: u32) -> (Vec<u8>, usize) {
    let (y, u, v) = make_yuv420(w as usize, h as usize, 0);
    let cfg = EncoderConfig::new(w, h);
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr(&YuvFrame {
        width: w,
        height: h,
        y: &y,
        u: &u,
        v: &v,
    });
    (idr.annex_b, 1)
}

/// IDR + n P-frames, each P encoded against the previous picture so
/// the bench drives a real reference chain.
fn build_idr_plus_p(w: u32, h: u32, n_p: u32) -> (Vec<u8>, usize) {
    let mut cfg = EncoderConfig::new(w, h);
    cfg.max_num_ref_frames = 1;
    let enc = Encoder::new(cfg);
    let mut stream = Vec::new();
    let (y0, u0, v0) = make_yuv420(w as usize, h as usize, 0);
    let idr = enc.encode_idr(&YuvFrame {
        width: w,
        height: h,
        y: &y0,
        u: &u0,
        v: &v0,
    });
    stream.extend_from_slice(&idr.annex_b);
    // Each P references the prior frame's recon implicitly via
    // `EncodedFrameRef::from(prev)` — we walk the gradient phase so
    // every P actually has motion to encode.
    let mut prev_ref = EncodedFrameRef::from(&idr);
    let mut keep_alive: Vec<oxideav_h264::encoder::EncodedP> = Vec::new();
    for k in 0..n_p {
        let phase = (k as usize + 1) * 2;
        let (yi, ui, vi) = make_yuv420(w as usize, h as usize, phase);
        let p = enc.encode_p(
            &YuvFrame {
                width: w,
                height: h,
                y: &yi,
                u: &ui,
                v: &vi,
            },
            &prev_ref,
            k + 1,
            (k + 1) * 2,
        );
        stream.extend_from_slice(&p.annex_b);
        keep_alive.push(p);
        // SAFETY of lifetime: `EncodedFrameRef` borrows from the
        // `EncodedP` we just pushed into `keep_alive`. The Vec stays
        // alive for the rest of the function, so the borrow is valid
        // through the loop.
        let last = keep_alive.last().unwrap();
        prev_ref = EncodedFrameRef::from(last);
    }
    let n_frames = 1 + n_p as usize;
    (stream, n_frames)
}

/// IDR + P + B (canonical IPB GOP). Requires Main profile.
fn build_idr_p_b(w: u32, h: u32) -> (Vec<u8>, usize) {
    let mut cfg = EncoderConfig::new(w, h);
    cfg.profile_idc = 77; // Main — required for B-slices (§A.2.2).
    cfg.max_num_ref_frames = 2;
    let enc = Encoder::new(cfg);

    let (y0, u0, v0) = make_yuv420(w as usize, h as usize, 0);
    let (y1, u1, v1) = make_yuv420(w as usize, h as usize, 4);
    let f0 = (y0.clone(), u0.clone(), v0.clone());
    let f1 = (y1.clone(), u1.clone(), v1.clone());
    let (y2, u2, v2) = make_b_yuv420(w as usize, h as usize, &f0, &f1);

    let idr = enc.encode_idr(&YuvFrame {
        width: w,
        height: h,
        y: &y0,
        u: &u0,
        v: &v0,
    });
    let p = enc.encode_p(
        &YuvFrame {
            width: w,
            height: h,
            y: &y1,
            u: &u1,
            v: &v1,
        },
        &EncodedFrameRef::from(&idr),
        1,
        4,
    );
    let b = enc.encode_b(
        &YuvFrame {
            width: w,
            height: h,
            y: &y2,
            u: &u2,
            v: &v2,
        },
        &EncodedFrameRef::from(&idr),
        &EncodedFrameRef::from(&p),
        1,
        2,
    );

    let mut stream = Vec::new();
    stream.extend_from_slice(&idr.annex_b);
    stream.extend_from_slice(&p.annex_b);
    stream.extend_from_slice(&b.annex_b);
    (stream, 3)
}

fn build_idr_422(w: u32, h: u32) -> (Vec<u8>, usize) {
    let (y, u, v) = make_yuv422(w as usize, h as usize);
    let mut cfg = EncoderConfig::new(w, h);
    cfg.profile_idc = 122; // High 4:2:2.
    cfg.chroma_format_idc = 2;
    let enc = Encoder::new(cfg);
    let idr = enc.encode_idr(&YuvFrame {
        width: w,
        height: h,
        y: &y,
        u: &u,
        v: &v,
    });
    (idr.annex_b, 1)
}

/// Run one decode cycle: fresh decoder, one packet, flush, drain. We
/// always recreate the decoder per iteration so DPB state from a
/// prior iteration can't leak into the measurement.
fn decode_once(stream: &[u8]) -> usize {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream.to_vec()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut n = 0usize;
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(_)) => n += 1,
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    n
}

fn bench_decode(c: &mut Criterion) {
    // Build every corpus once; share across iterations.
    let corpora: &[(&str, (Vec<u8>, usize))] = &[
        ("idr_only_64x64", build_idr_only(64, 64)),
        ("idr_only_128x96", build_idr_only(128, 96)),
        ("idr_p4_64x64", build_idr_plus_p(64, 64, 4)),
        ("idr_p_b_64x64", build_idr_p_b(64, 64)),
        ("idr_only_422_64x64", build_idr_422(64, 64)),
    ];

    let mut group = c.benchmark_group("h264_decode");
    for (label, (stream, n_frames)) in corpora {
        // Element throughput == frames decoded per iteration (so
        // Criterion reports `Kelem/s` ≈ frames per second).
        group.throughput(Throughput::Elements(*n_frames as u64));
        group.bench_with_input(BenchmarkId::new("frames_per_sec", label), stream, |b, s| {
            b.iter(|| decode_once(s))
        });
        // Byte throughput on the same input set; reading the same
        // bench twice with a different Throughput is cheap and gives
        // us both views in one report.
        group.throughput(Throughput::Bytes(stream.len() as u64));
        group.bench_with_input(BenchmarkId::new("bytes_per_sec", label), stream, |b, s| {
            b.iter(|| decode_once(s))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
