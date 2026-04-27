//! Micro-benchmarks for the H.264 inverse transform kernels.
//!
//! `inverse_transform_4x4` is invoked once per 4×4 sub-block (16× per
//! macroblock) and `inverse_transform_8x8` is used for High-profile
//! 8×8 transform mode. We bench batches of 1000 blocks per iteration
//! so the per-iteration cost matches the scale of a real frame
//! (~4500 4x4 blocks for 720p in 4:2:0).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxideav_h264::transform::{
    default_scaling_list_4x4_flat, default_scaling_list_8x8_flat, inverse_transform_4x4,
    inverse_transform_8x8,
};

fn mk_block_4x4(seed: u32) -> [i32; 16] {
    let mut b = [0i32; 16];
    b[0] = ((seed % 200) as i32) - 100;
    for i in 1..6 {
        b[(i * 3) % 16] = (((seed.wrapping_mul(i as u32 + 1)) % 41) as i32) - 20;
    }
    b
}

fn mk_block_8x8(seed: u32) -> [i32; 64] {
    let mut b = [0i32; 64];
    b[0] = ((seed % 200) as i32) - 100;
    for i in 1..16 {
        b[(i * 5) % 64] = (((seed.wrapping_mul(i as u32 + 1)) % 41) as i32) - 20;
    }
    b
}

fn bench_idct4x4(c: &mut Criterion) {
    let n = 1000usize;
    let blocks: Vec<[i32; 16]> = (0..n)
        .map(|i| mk_block_4x4(i as u32 * 2654435761))
        .collect();
    let sl = default_scaling_list_4x4_flat();
    let mut group = c.benchmark_group("inverse_transform_4x4");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function(BenchmarkId::new("scalar_qp26", n), |b| {
        b.iter(|| {
            for blk in &blocks {
                let _ = inverse_transform_4x4(blk, 26, &sl, 8).unwrap();
            }
        });
    });
    group.finish();
}

fn bench_idct8x8(c: &mut Criterion) {
    let n = 1000usize;
    let blocks: Vec<[i32; 64]> = (0..n)
        .map(|i| mk_block_8x8(i as u32 * 2654435761))
        .collect();
    let sl = default_scaling_list_8x8_flat();
    let mut group = c.benchmark_group("inverse_transform_8x8");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function(BenchmarkId::new("scalar_qp26", n), |b| {
        b.iter(|| {
            for blk in &blocks {
                let _ = inverse_transform_8x8(blk, 26, &sl, 8).unwrap();
            }
        });
    });
    group.finish();
}

criterion_group!(benches, bench_idct4x4, bench_idct8x8);
criterion_main!(benches);
