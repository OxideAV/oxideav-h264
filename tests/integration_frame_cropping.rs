//! §7.4.2.1.1 frame cropping on the encoder side: pictures whose size
//! is not a whole number of macroblocks are coded on a padded grid and
//! the SPS cropping rectangle restores the size at the decoder.
//!
//! Per size x layout x entropy coder the gates are: the SPS carries the
//! exact cropping offsets (in `CropUnitX` / `CropUnitY` for the
//! layout), our decoder outputs pictures of the requested size within
//! the lossy bound (sample-exact for `lossless=true`), and a black-box
//! decoder run as an opaque process outputs the same bytes.

#[path = "support/registry_harness.rs"]
mod harness;

use harness::{
    YUV420, YUV420_10, YUV422, YUV422_10, YUV444, YUV444_12, check, decode_ours, first_sps, params,
    reference_decode,
};
use oxideav_core::Error;

/// 4:2:0 sizes: `CropUnitX = CropUnitY = 2`.
const SIZES_420: &[(usize, usize)] = &[
    (66, 50),
    (18, 34),
    (2, 2),
    (130, 46),
    (64, 50),
    (62, 48),
    (30, 14),
];
/// 4:2:2 sizes: `CropUnitX = 2`, `CropUnitY = 1` (odd heights).
const SIZES_422: &[(usize, usize)] = &[(66, 49), (34, 17), (2, 1), (64, 33), (46, 48)];
/// 4:4:4 sizes: unit crop (odd widths and heights).
const SIZES_444: &[(usize, usize)] = &[(65, 49), (33, 17), (1, 1), (64, 47), (47, 64), (17, 3)];

#[test]
fn yuv420_8bit_sizes_cavlc_and_cabac() {
    for &(w, h) in SIZES_420 {
        check(YUV420, w, h, &[("qp", "22")], 3);
        check(YUV420, w, h, &[("qp", "22"), ("cabac", "true")], 3);
    }
}

#[test]
fn yuv420_8bit_b_pictures_and_rate_control() {
    check(YUV420, 66, 50, &[("qp", "22"), ("bframes", "2")], 7);
    check(
        YUV420,
        50,
        38,
        &[("qp", "22"), ("bframes", "1"), ("cabac", "true")],
        6,
    );
    check(YUV420, 70, 46, &[("rc", "cbr"), ("bitrate", "400000")], 5);
}

#[test]
fn yuv420_high_profile_tools_and_10bit() {
    for &(w, h) in SIZES_420 {
        check(YUV420, w, h, &[("qp", "22"), ("profile", "high")], 3);
        check(
            YUV420,
            w,
            h,
            &[("qp", "22"), ("transform_8x8", "true"), ("cabac", "true")],
            3,
        );
        check(YUV420_10, w, h, &[("qp", "22")], 3);
        check(YUV420_10, w, h, &[("qp", "22"), ("cabac", "true")], 3);
    }
}

#[test]
fn yuv422_sizes() {
    for &(w, h) in SIZES_422 {
        check(YUV422, w, h, &[("qp", "22")], 3);
        check(YUV422, w, h, &[("qp", "22"), ("cabac", "true")], 3);
        check(YUV422_10, w, h, &[("qp", "22")], 3);
    }
}

#[test]
fn yuv444_sizes() {
    for &(w, h) in SIZES_444 {
        check(YUV444, w, h, &[("qp", "22")], 3);
        check(YUV444, w, h, &[("qp", "22"), ("cabac", "true")], 3);
        check(YUV444_12, w, h, &[("qp", "22")], 3);
    }
}

#[test]
fn lossless_cropped_pictures_reconstruct_exactly() {
    for (l, w, h) in [
        (YUV444, 65, 49),
        (YUV444_12, 33, 17),
        (YUV420, 66, 50),
        (YUV422_10, 34, 17),
    ] {
        check(l, w, h, &[("lossless", "true")], 3);
        check(l, w, h, &[("lossless", "true"), ("cabac", "true")], 3);
    }
}

#[test]
fn sizes_off_the_chroma_grid_are_refused() {
    for (l, w, h) in [
        (YUV420, 65, 48),
        (YUV420, 64, 49),
        (YUV422, 65, 48),
        (YUV420_10, 63, 47),
        (YUV444, 0, 16),
    ] {
        assert!(
            matches!(
                oxideav_h264::h264_encoder::make_encoder(&params(l, w, h, &[])),
                Err(Error::InvalidData(_))
            ),
            "{l:?} {w}x{h} must be refused"
        );
    }
}

// ---------------------------------------------------------------------
// Field coding (`frame_mbs_only_flag = 0`): `CropUnitY` doubles (4 luma
// rows at 4:2:0) and the coded height is a whole number of macroblock
// PAIRS.
// ---------------------------------------------------------------------

mod field_coding {
    use super::{decode_ours, first_sps, reference_decode};
    use oxideav_core::{Packet, TimeBase};
    use oxideav_h264::encoder::field::{PaffConfig, encode_paff_sequence};
    use oxideav_h264::encoder::mbaff::{MbaffConfig, PairMode, encode_mbaff_sequence};
    use oxideav_h264::encoder::mbaff_cabac::{MbaffCabacConfig, encode_mbaff_cabac_sequence};

    type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

    /// Interlaced content: each field samples a moving gradient at its
    /// own instant.
    fn interlaced(w: usize, h: usize, k: usize) -> Planes {
        let mut y = vec![0u8; w * h];
        for row in 0..h {
            let t = 2 * k + (row & 1);
            for col in 0..w {
                y[row * w + col] = (16 + ((col + 3 * t) % w + row / 2) * 219 / (w + h / 2)) as u8;
            }
        }
        let (cw, ch) = (w / 2, h / 2);
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];
        for row in 0..ch {
            let t = 2 * k + (row & 1);
            for col in 0..cw {
                u[row * cw + col] = (96 + ((col + 2 * t) % 32) * 2) as u8;
                v[row * cw + col] = (160 - ((col + t) % 24) * 2) as u8;
            }
        }
        (y, u, v)
    }

    /// `(width, frame height)`: multiples of (2, 4).
    const SIZES: &[(usize, usize)] = &[(66, 52), (50, 44), (64, 40), (18, 20), (2, 4), (48, 68)];

    fn gate(tag: &str, w: usize, h: usize, annex_b: &[u8], recon: &[Planes]) {
        let pkt = Packet::new(0, TimeBase::new(1, 25), annex_b.to_vec()).with_pts(0);
        let sps = first_sps(std::slice::from_ref(&pkt));
        assert!(!sps.frame_mbs_only_flag, "{tag}: field coding");
        assert_eq!(sps.crop_units(), (2, 4), "{tag}: crop units");
        assert_eq!(
            sps.cropped_dimensions(),
            (w as u32, h as u32),
            "{tag}: cropped size"
        );
        assert_eq!(
            (sps.pic_height_in_map_units_minus1 as usize + 1) * 32,
            h.next_multiple_of(32),
            "{tag}: coded height in macroblock pairs"
        );
        let ours = decode_ours(std::slice::from_ref(&pkt));
        assert_eq!(ours.len(), recon.len(), "{tag}: frame count");
        for (n, (f, r)) in ours.iter().zip(recon).enumerate() {
            let expect = [r.0.as_slice(), r.1.as_slice(), r.2.as_slice()].concat();
            assert_eq!(expect.len(), w * h * 3 / 2, "{tag}: recon size");
            assert!(
                f == &expect,
                "{tag}: frame {n} differs from the encoder recon"
            );
        }
        if let Some(reference) = reference_decode(annex_b, tag) {
            assert!(
                reference == ours.concat(),
                "{tag}: black-box decoder output differs"
            );
        }
    }

    fn frames(w: usize, h: usize, n: usize) -> Vec<Planes> {
        (0..n).map(|k| interlaced(w, h, k)).collect()
    }

    fn refs(f: &[Planes]) -> Vec<(&[u8], &[u8], &[u8])> {
        f.iter()
            .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
            .collect()
    }

    #[test]
    fn paff_field_pictures_cropped() {
        for &(w, h) in SIZES {
            for (p_fields, b_fields) in [(false, false), (true, false), (true, true)] {
                let src = frames(w, h, 4);
                let enc = encode_paff_sequence(
                    &PaffConfig {
                        width: w as u32,
                        frame_height: h as u32,
                        qp: 24,
                        p_fields,
                        frame_picture_indices: Vec::new(),
                        cross_parity_first_bottom: false,
                        idr_frame_first: false,
                        b_fields,
                        b_temporal_direct: false,
                        transform_8x8: false,
                        long_term_anchor: false,
                        mmco_unpair_first_top: false,
                        b_implicit_weight: false,
                        b_reference_fields: false,
                    },
                    &refs(&src),
                );
                gate(
                    &format!("paff-{w}x{h}-p{p_fields}-b{b_fields}"),
                    w,
                    h,
                    &enc.annex_b,
                    &enc.recon_frames,
                );
            }
        }
    }

    #[test]
    fn mbaff_frames_cropped_cavlc_and_cabac() {
        for &(w, h) in SIZES {
            for pair_mode in [PairMode::AllField, PairMode::Adaptive, PairMode::Checker] {
                let src = frames(w, h, 3);
                let enc = encode_mbaff_sequence(
                    &MbaffConfig {
                        width: w as u32,
                        frame_height: h as u32,
                        qp: 24,
                        pair_mode,
                        p_frames: true,
                    },
                    &refs(&src),
                );
                gate(
                    &format!("mbaff-{w}x{h}-{pair_mode:?}"),
                    w,
                    h,
                    &enc.annex_b,
                    &enc.recon_frames,
                );
                let enc = encode_mbaff_cabac_sequence(
                    &MbaffCabacConfig {
                        width: w as u32,
                        frame_height: h as u32,
                        qp: 24,
                        pair_mode,
                        p_frames: true,
                        intra_in_p: true,
                        transform_8x8: false,
                    },
                    &refs(&src),
                );
                gate(
                    &format!("mbaff-cabac-{w}x{h}-{pair_mode:?}"),
                    w,
                    h,
                    &enc.annex_b,
                    &enc.recon_frames,
                );
            }
        }
    }
}
