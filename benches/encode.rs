//! End-to-end H.264 encode benchmark.
//!
//! Sibling to `benches/decode.rs`: where that one drives the public
//! [`H264CodecDecoder`] across in-process corpora, this one drives the
//! encoder ([`Encoder::encode_idr`] / [`encode_p`] / [`encode_b`] +
//! the CABAC variants) so we get a representative wall-clock cost for
//! the public encode path. Bench inputs are plain `YuvFrame`s
//! synthesised in-process — no on-disk fixtures, runs in CI without
//! extra plumbing.
//!
//! Variants (each picks a small but representative slice of the
//! encoder surface):
//!
//! | label                       | what's measured                                                         |
//! | --------------------------- | ----------------------------------------------------------------------- |
//! | `idr_only_64x64_baseline`   | §7.3.5 / §8 CAVLC IDR — SPS+PPS+IDR write path                          |
//! | `idr_only_128x96_baseline`  | same kernel, larger picture (MB-grid overhead visible)                  |
//! | `idr_p4_64x64_baseline`     | IDR + 4×P chain — §8.4 inter ME + §9.2 CAVLC P                          |
//! | `idr_p_b_64x64_main`        | IDR + P + B Main — §8.4.1.2.2 spatial direct + bipred                   |
//! | `idr_only_64x64_cabac`      | §9.3 CABAC IDR — `encode_idr_cabac` arithmetic write path               |
//! | `idr_p_64x64_cabac`         | §9.3 CABAC P — `encode_p_cabac` + §9.3.3.1.3 trellis quant (default on) |
//! | `idr_only_422_64x64`        | High 4:2:2 IDR — §8.5.11.2 4×2 chroma DC Hadamard, plane sizing         |
//!
//! Throughput is reported in two views per variant: per-frame
//! (`Throughput::Elements`, so Criterion prints `Kelem/s` ≈ frames per
//! second encoded) and per-source-byte
//! (`Throughput::Bytes`, so Criterion prints byte rate over the raw
//! source planes). The encoded-stream byte count is *not* a throughput
//! metric — it's a compression-efficiency metric and belongs to the
//! integration tests / cross-decode oracle, not the bench.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};

/// Synthesize a smooth-gradient 4:2:0 picture of the given size. Same
/// shape as `benches/decode.rs::make_yuv420` — diagonal-gradient luma,
/// mid-grey chroma. The gradient gives the entropy coder a non-zero
/// residual to work on; a flat plane would collapse the bench.
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

/// Bipred-friendly midpoint of two reference luma planes — mirrors
/// `benches/decode.rs::make_b_yuv420`.
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

/// 4:2:2 source — chroma is `(W/2, H)`. Mirrors decode bench.
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

/// Total source-plane byte count for a 4:2:0 input. Used for the
/// `Throughput::Bytes` view.
fn yuv420_src_bytes(w: usize, h: usize) -> u64 {
    (w * h + 2 * (w / 2) * (h / 2)) as u64
}

/// Total source-plane byte count for a 4:2:2 input.
fn yuv422_src_bytes(w: usize, h: usize) -> u64 {
    (w * h + 2 * (w / 2) * h) as u64
}

// ---- per-variant runners ---------------------------------------------------
//
// Each runner builds a fresh Encoder inside the iterated closure so DPB
// / per-encoder state from a prior iteration can't leak into the
// measurement. The Encoder itself is cheap to construct — no codec
// tables are allocated lazily.

fn encode_idr_once(w: u32, h: u32, src: &(Vec<u8>, Vec<u8>, Vec<u8>)) -> usize {
    let cfg = EncoderConfig::new(w, h);
    let enc = Encoder::new(cfg);
    let out = enc.encode_idr(&YuvFrame {
        width: w,
        height: h,
        y: &src.0,
        u: &src.1,
        v: &src.2,
    });
    out.annex_b.len()
}

fn encode_idr_plus_p_once(
    w: u32,
    h: u32,
    idr_src: &(Vec<u8>, Vec<u8>, Vec<u8>),
    p_srcs: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
) -> usize {
    let mut cfg = EncoderConfig::new(w, h);
    cfg.max_num_ref_frames = 1;
    let enc = Encoder::new(cfg);

    let idr = enc.encode_idr(&YuvFrame {
        width: w,
        height: h,
        y: &idr_src.0,
        u: &idr_src.1,
        v: &idr_src.2,
    });
    let mut bytes = idr.annex_b.len();

    // Keep prior `EncodedP`s alive so their borrows are valid across
    // the loop — `EncodedFrameRef` borrows the recon planes of the
    // previous frame. Same lifetime pattern as `benches/decode.rs`.
    let mut prev_ref = EncodedFrameRef::from(&idr);
    let mut keep_alive: Vec<oxideav_h264::encoder::EncodedP> = Vec::new();
    for (k, src) in p_srcs.iter().enumerate() {
        let p = enc.encode_p(
            &YuvFrame {
                width: w,
                height: h,
                y: &src.0,
                u: &src.1,
                v: &src.2,
            },
            &prev_ref,
            (k + 1) as u32,
            ((k + 1) * 2) as u32,
        );
        bytes += p.annex_b.len();
        keep_alive.push(p);
        // SAFETY: see decode bench — `keep_alive` outlives this loop
        // body so the borrow into the most-recent push is valid.
        let last = keep_alive.last().unwrap();
        prev_ref = EncodedFrameRef::from(last);
    }
    bytes
}

fn encode_idr_p_b_once(
    w: u32,
    h: u32,
    src_i: &(Vec<u8>, Vec<u8>, Vec<u8>),
    src_p: &(Vec<u8>, Vec<u8>, Vec<u8>),
    src_b: &(Vec<u8>, Vec<u8>, Vec<u8>),
) -> usize {
    let mut cfg = EncoderConfig::new(w, h);
    cfg.profile_idc = 77; // Main — required for B (§A.2.2).
    cfg.max_num_ref_frames = 2;
    let enc = Encoder::new(cfg);

    let idr = enc.encode_idr(&YuvFrame {
        width: w,
        height: h,
        y: &src_i.0,
        u: &src_i.1,
        v: &src_i.2,
    });
    let p = enc.encode_p(
        &YuvFrame {
            width: w,
            height: h,
            y: &src_p.0,
            u: &src_p.1,
            v: &src_p.2,
        },
        &EncodedFrameRef::from(&idr),
        1,
        4,
    );
    let b = enc.encode_b(
        &YuvFrame {
            width: w,
            height: h,
            y: &src_b.0,
            u: &src_b.1,
            v: &src_b.2,
        },
        &EncodedFrameRef::from(&idr),
        &EncodedFrameRef::from(&p),
        1,
        2,
    );
    idr.annex_b.len() + p.annex_b.len() + b.annex_b.len()
}

fn encode_idr_cabac_once(w: u32, h: u32, src: &(Vec<u8>, Vec<u8>, Vec<u8>)) -> usize {
    let mut cfg = EncoderConfig::new(w, h);
    cfg.profile_idc = 77; // Main — CABAC forbidden by Baseline (§A.2.1).
    cfg.cabac = true;
    let enc = Encoder::new(cfg);
    let out = enc.encode_idr_cabac(&YuvFrame {
        width: w,
        height: h,
        y: &src.0,
        u: &src.1,
        v: &src.2,
    });
    out.annex_b.len()
}

fn encode_idr_plus_p_cabac_once(
    w: u32,
    h: u32,
    idr_src: &(Vec<u8>, Vec<u8>, Vec<u8>),
    p_src: &(Vec<u8>, Vec<u8>, Vec<u8>),
) -> usize {
    let mut cfg = EncoderConfig::new(w, h);
    cfg.profile_idc = 77;
    cfg.cabac = true;
    cfg.max_num_ref_frames = 1;
    let enc = Encoder::new(cfg);

    let idr = enc.encode_idr_cabac(&YuvFrame {
        width: w,
        height: h,
        y: &idr_src.0,
        u: &idr_src.1,
        v: &idr_src.2,
    });
    let p = enc.encode_p_cabac(
        &YuvFrame {
            width: w,
            height: h,
            y: &p_src.0,
            u: &p_src.1,
            v: &p_src.2,
        },
        &EncodedFrameRef::from(&idr),
        1,
        2,
    );
    idr.annex_b.len() + p.annex_b.len()
}

fn encode_idr_422_once(w: u32, h: u32, src: &(Vec<u8>, Vec<u8>, Vec<u8>)) -> usize {
    let mut cfg = EncoderConfig::new(w, h);
    cfg.profile_idc = 122; // High 4:2:2.
    cfg.chroma_format_idc = 2;
    let enc = Encoder::new(cfg);
    let out = enc.encode_idr(&YuvFrame {
        width: w,
        height: h,
        y: &src.0,
        u: &src.1,
        v: &src.2,
    });
    out.annex_b.len()
}

// ---- bench harness ---------------------------------------------------------

fn bench_encode(c: &mut Criterion) {
    // Pre-build every source frame once so the bench's hot loop
    // doesn't measure gradient synthesis. The closures capture
    // references into these slabs.
    let s64_p0 = make_yuv420(64, 64, 0);
    let s64_p1 = make_yuv420(64, 64, 4);
    let s64_p2 = make_yuv420(64, 64, 6);
    let s64_p3 = make_yuv420(64, 64, 8);
    let s64_p4 = make_yuv420(64, 64, 10);
    let s64_b = make_b_yuv420(64, 64, &s64_p0, &s64_p1);
    let s128 = make_yuv420(128, 96, 0);
    let s422 = make_yuv422(64, 64);

    let mut group = c.benchmark_group("h264_encode");

    // 1) Baseline IDR — 64x64 + 128x96.
    for (label, w, h, src) in &[
        ("idr_only_64x64_baseline", 64u32, 64u32, &s64_p0),
        ("idr_only_128x96_baseline", 128u32, 96u32, &s128),
    ] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("frames_per_sec", label), src, |b, src| {
            b.iter(|| encode_idr_once(*w, *h, src))
        });
        group.throughput(Throughput::Bytes(yuv420_src_bytes(
            *w as usize,
            *h as usize,
        )));
        group.bench_with_input(
            BenchmarkId::new("src_bytes_per_sec", label),
            src,
            |b, src| b.iter(|| encode_idr_once(*w, *h, src)),
        );
    }

    // 2) Baseline IDR + 4×P (5 frames).
    {
        let label = "idr_p4_64x64_baseline";
        let p_srcs = vec![
            s64_p1.clone(),
            s64_p2.clone(),
            s64_p3.clone(),
            s64_p4.clone(),
        ];
        group.throughput(Throughput::Elements(5));
        group.bench_function(BenchmarkId::new("frames_per_sec", label), |b| {
            b.iter(|| encode_idr_plus_p_once(64, 64, &s64_p0, &p_srcs))
        });
        group.throughput(Throughput::Bytes(yuv420_src_bytes(64, 64) * 5));
        group.bench_function(BenchmarkId::new("src_bytes_per_sec", label), |b| {
            b.iter(|| encode_idr_plus_p_once(64, 64, &s64_p0, &p_srcs))
        });
    }

    // 3) Main IDR + P + B (3 frames).
    {
        let label = "idr_p_b_64x64_main";
        group.throughput(Throughput::Elements(3));
        group.bench_function(BenchmarkId::new("frames_per_sec", label), |b| {
            b.iter(|| encode_idr_p_b_once(64, 64, &s64_p0, &s64_p1, &s64_b))
        });
        group.throughput(Throughput::Bytes(yuv420_src_bytes(64, 64) * 3));
        group.bench_function(BenchmarkId::new("src_bytes_per_sec", label), |b| {
            b.iter(|| encode_idr_p_b_once(64, 64, &s64_p0, &s64_p1, &s64_b))
        });
    }

    // 4) CABAC IDR.
    {
        let label = "idr_only_64x64_cabac";
        group.throughput(Throughput::Elements(1));
        group.bench_function(BenchmarkId::new("frames_per_sec", label), |b| {
            b.iter(|| encode_idr_cabac_once(64, 64, &s64_p0))
        });
        group.throughput(Throughput::Bytes(yuv420_src_bytes(64, 64)));
        group.bench_function(BenchmarkId::new("src_bytes_per_sec", label), |b| {
            b.iter(|| encode_idr_cabac_once(64, 64, &s64_p0))
        });
    }

    // 5) CABAC IDR + P. `EncoderConfig::trellis_quant` defaults to
    // `true` so this bench exercises the §9.3.3.1.3 inter trellis
    // refinement on top of the open-loop quantizer (round 49).
    {
        let label = "idr_p_64x64_cabac";
        group.throughput(Throughput::Elements(2));
        group.bench_function(BenchmarkId::new("frames_per_sec", label), |b| {
            b.iter(|| encode_idr_plus_p_cabac_once(64, 64, &s64_p0, &s64_p1))
        });
        group.throughput(Throughput::Bytes(yuv420_src_bytes(64, 64) * 2));
        group.bench_function(BenchmarkId::new("src_bytes_per_sec", label), |b| {
            b.iter(|| encode_idr_plus_p_cabac_once(64, 64, &s64_p0, &s64_p1))
        });
    }

    // 6) High 4:2:2 IDR.
    {
        let label = "idr_only_422_64x64";
        group.throughput(Throughput::Elements(1));
        group.bench_function(BenchmarkId::new("frames_per_sec", label), |b| {
            b.iter(|| encode_idr_422_once(64, 64, &s422))
        });
        group.throughput(Throughput::Bytes(yuv422_src_bytes(64, 64)));
        group.bench_function(BenchmarkId::new("src_bytes_per_sec", label), |b| {
            b.iter(|| encode_idr_422_once(64, 64, &s422))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
