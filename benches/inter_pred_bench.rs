//! Micro-benchmarks for the H.264 inter-prediction kernels.
//!
//! `interpolate_luma` is the canonical H.264 hotspot — the 6-tap FIR
//! is applied per-output-pixel, with the diagonal "j" position costing
//! a 6-tap H-FIR per intermediate sample plus a 6-tap V-FIR for the
//! final value. We bench representative block sizes (4x4, 8x8, 16x16)
//! across all 16 luma fractional positions, batched 1000 blocks per
//! iteration so loop overhead doesn't dominate.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxideav_h264::simd::{self, chunked, scalar};

fn make_plane(w: usize, h: usize) -> Vec<i32> {
    let mut v = vec![0i32; w * h];
    for y in 0..h {
        for x in 0..w {
            // Mid-grey + low-frequency variation so the FIR doesn't
            // run on a constant plane (which would clip away the work).
            v[y * w + x] = 128 + (((x * 7 + y * 11) ^ (x * y)) & 0x3f) as i32;
        }
    }
    v
}

fn bench_luma(c: &mut Criterion) {
    let src = make_plane(64, 64);
    // 1000 blocks × 16 fractions × 3 sizes makes the bench noisy; pick
    // four representative (xFrac, yFrac) combinations covering all
    // §8.4.2.2.1 code paths: integer, H-half, V-half, diag-j.
    let fracs: [(u8, u8); 4] = [(0, 0), (2, 0), (0, 2), (2, 2)];
    let sizes: [(u32, u32, &str); 3] = [(4, 4, "4x4"), (8, 8, "8x8"), (16, 16, "16x16")];

    for &(w, h, label) in &sizes {
        let mut group = c.benchmark_group(format!("interpolate_luma_{label}"));
        let n = 1000usize;
        group.throughput(Throughput::Elements(n as u64));
        for &(xf, yf) in &fracs {
            let tag = format!("xf{xf}_yf{yf}");
            group.bench_function(BenchmarkId::new(format!("scalar_{tag}"), n), |b| {
                let mut dst = vec![0i32; (w * h) as usize];
                b.iter(|| {
                    for k in 0..n {
                        scalar::interpolate_luma(
                            &src,
                            64,
                            64,
                            64,
                            (k % 30) as i32,
                            (k % 30) as i32,
                            xf,
                            yf,
                            w,
                            h,
                            8,
                            &mut dst,
                            w as usize,
                        )
                        .unwrap();
                    }
                });
            });
            group.bench_function(BenchmarkId::new(format!("chunked_{tag}"), n), |b| {
                let mut dst = vec![0i32; (w * h) as usize];
                b.iter(|| {
                    for k in 0..n {
                        chunked::interpolate_luma(
                            &src,
                            64,
                            64,
                            64,
                            (k % 30) as i32,
                            (k % 30) as i32,
                            xf,
                            yf,
                            w,
                            h,
                            8,
                            &mut dst,
                            w as usize,
                        )
                        .unwrap();
                    }
                });
            });
            group.bench_function(BenchmarkId::new(format!("default_{tag}"), n), |b| {
                let mut dst = vec![0i32; (w * h) as usize];
                b.iter(|| {
                    for k in 0..n {
                        simd::interpolate_luma(
                            &src,
                            64,
                            64,
                            64,
                            (k % 30) as i32,
                            (k % 30) as i32,
                            xf,
                            yf,
                            w,
                            h,
                            8,
                            &mut dst,
                            w as usize,
                        )
                        .unwrap();
                    }
                });
            });
        }
        group.finish();
    }
}

fn bench_chroma(c: &mut Criterion) {
    let src = make_plane(64, 64);
    let mut group = c.benchmark_group("interpolate_chroma_4x4");
    let n = 1000usize;
    group.throughput(Throughput::Elements(n as u64));

    for &(xf, yf) in &[(0u8, 0u8), (4, 4), (7, 7)] {
        let tag = format!("xf{xf}_yf{yf}");
        group.bench_function(BenchmarkId::new(format!("scalar_{tag}"), n), |b| {
            let mut dst = vec![0i32; 4 * 4];
            b.iter(|| {
                for k in 0..n {
                    scalar::interpolate_chroma(
                        &src,
                        64,
                        64,
                        64,
                        (k % 50) as i32,
                        (k % 50) as i32,
                        xf,
                        yf,
                        4,
                        4,
                        8,
                        &mut dst,
                        4,
                    )
                    .unwrap();
                }
            });
        });
        group.bench_function(BenchmarkId::new(format!("chunked_{tag}"), n), |b| {
            let mut dst = vec![0i32; 4 * 4];
            b.iter(|| {
                for k in 0..n {
                    chunked::interpolate_chroma(
                        &src,
                        64,
                        64,
                        64,
                        (k % 50) as i32,
                        (k % 50) as i32,
                        xf,
                        yf,
                        4,
                        4,
                        8,
                        &mut dst,
                        4,
                    )
                    .unwrap();
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_luma, bench_chroma);
criterion_main!(benches);
