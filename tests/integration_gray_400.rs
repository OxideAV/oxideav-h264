//! 4:0:0 (monochrome) encoding through the registry encoder: grey
//! input (`Gray8` / `Gray10Le` / `Gray12Le`) is coded with
//! `chroma_format_idc = 0` under the §A.2.4 – §A.2.7 minimum
//! High-family profile — luma only, no chroma syntax at all — instead
//! of 4:2:0 with constant chroma planes.
//!
//! Per layout x size x option set: the SPS states `chroma_format_idc =
//! 0` and the bit depth, the output parameters name the grey layout,
//! our decoder returns one plane within the lossy bound (sample-exact
//! for `lossless=true`), and a black-box decoder run as an opaque
//! process outputs the same bytes.

#[path = "support/registry_harness.rs"]
mod harness;

use harness::{GRAY8, GRAY10, GRAY12, check, encode, first_sps, params};
use oxideav_core::PixelFormat;

/// Unit crop grid at 4:0:0: odd sizes included.
const SIZES: &[(usize, usize)] = &[(64, 48), (65, 49), (33, 17), (16, 16), (1, 1), (50, 38)];

#[test]
fn gray8_is_coded_400_high_cavlc_and_cabac() {
    for &(w, h) in SIZES {
        let (sps, _) = check(GRAY8, w, h, &[("qp", "22")], 4);
        assert_eq!(sps.profile_idc, 100, "High is the lowest 4:0:0 profile");
        let (sps, _) = check(GRAY8, w, h, &[("qp", "22"), ("cabac", "true")], 4);
        assert_eq!(sps.profile_idc, 100);
    }
}

#[test]
fn gray8_b_pictures_and_rate_control() {
    check(GRAY8, 64, 48, &[("qp", "22"), ("bframes", "2")], 7);
    check(
        GRAY8,
        65,
        49,
        &[("qp", "22"), ("bframes", "2"), ("cabac", "true")],
        7,
    );
    check(GRAY8, 64, 48, &[("rc", "cbr"), ("bitrate", "300000")], 6);
    check(
        GRAY8,
        50,
        38,
        &[("rc", "vbr"), ("bitrate", "200000"), ("cabac", "true")],
        6,
    );
    check(
        GRAY8,
        64,
        48,
        &[("rc", "cbr"), ("bitrate", "300000"), ("bframes", "2")],
        7,
    );
}

#[test]
fn gray8_high_family_tools() {
    for &(w, h) in SIZES {
        for cabac in ["false", "true"] {
            check(
                GRAY8,
                w,
                h,
                &[("qp", "22"), ("profile", "high"), ("cabac", cabac)],
                3,
            );
            check(
                GRAY8,
                w,
                h,
                &[("qp", "22"), ("transform_8x8", "true"), ("cabac", cabac)],
                3,
            );
            check(
                GRAY8,
                w,
                h,
                &[("qp", "22"), ("deblock", "false"), ("cabac", cabac)],
                3,
            );
            check(
                GRAY8,
                w,
                h,
                &[
                    ("qp", "22"),
                    ("profile", "high"),
                    ("intra_4x4", "false"),
                    ("cabac", cabac),
                ],
                3,
            );
        }
    }
}

#[test]
fn gray_high_bit_depth() {
    for (l, profile) in [(GRAY10, 110u8), (GRAY12, 244)] {
        for &(w, h) in SIZES {
            for cabac in ["false", "true"] {
                let (sps, _) = check(l, w, h, &[("qp", "22"), ("cabac", cabac)], 3);
                assert_eq!(sps.profile_idc, profile, "{l:?}");
                check(
                    l,
                    w,
                    h,
                    &[("qp", "30"), ("transform_8x8", "true"), ("cabac", cabac)],
                    3,
                );
            }
        }
    }
}

#[test]
fn gray_lossless_reconstructs_exactly() {
    for l in [GRAY8, GRAY10, GRAY12] {
        for &(w, h) in &[(64usize, 48usize), (33, 17)] {
            for cabac in ["false", "true"] {
                let (sps, _) = check(l, w, h, &[("lossless", "true"), ("cabac", cabac)], 3);
                assert_eq!(sps.profile_idc, 244);
                assert!(sps.qpprime_y_zero_transform_bypass_flag);
            }
        }
    }
}

/// `profile=baseline` / `main` have no 4:0:0 format: grey input rides
/// 4:2:0 with constant chroma there (the pre-existing behaviour).
#[test]
fn gray8_under_baseline_or_main_stays_420() {
    for (opts, profile) in [
        (vec![("profile", "baseline")], 66u8),
        (vec![("profile", "main"), ("cabac", "true")], 77),
    ] {
        let (packets, out) = encode(GRAY8, 64, 48, &opts, 3);
        let sps = first_sps(&packets);
        assert_eq!(sps.profile_idc, profile);
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(out.pixel_format, Some(PixelFormat::Yuv420P));
    }
    // 10-bit grey has no Baseline / Main route at all.
    assert!(
        oxideav_h264::h264_encoder::make_encoder(&params(
            GRAY10,
            64,
            48,
            &[("profile", "baseline")]
        ))
        .is_err()
    );
}

/// Dropping the two constant chroma planes saves their per-macroblock
/// syntax (`intra_chroma_pred_mode`, the chroma half of the Intra_16x16
/// `mb_type` / `coded_block_pattern`): intra pictures — the still-image
/// case — are smaller at 4:0:0 than at 4:2:0 with constant chroma.
#[test]
fn gray_400_intra_is_smaller_than_420_with_constant_chroma() {
    let (w, h, frames) = (160, 128, 3);
    let size = |opts: &[(&str, &str)]| {
        encode(GRAY8, w, h, opts, frames)
            .0
            .iter()
            .map(|p| p.data.len())
            .sum::<usize>()
    };
    for (name, mono, yuv) in [
        (
            "CAVLC",
            size(&[("qp", "26"), ("gop", "1")]),
            size(&[("qp", "26"), ("gop", "1"), ("profile", "baseline")]),
        ),
        (
            "CABAC",
            size(&[("qp", "26"), ("gop", "1"), ("cabac", "true")]),
            size(&[
                ("qp", "26"),
                ("gop", "1"),
                ("cabac", "true"),
                ("profile", "main"),
            ]),
        ),
    ] {
        eprintln!(
            "grey {w}x{h} x{frames} intra {name} QP 26: 4:0:0 High {mono} B, 4:2:0 {yuv} B ({:+.2} %)",
            100.0 * (mono as f64 - yuv as f64) / yuv as f64
        );
        assert!(mono < yuv, "{name}: 4:0:0 {mono} B vs 4:2:0 {yuv} B");
    }
}
