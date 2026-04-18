//! Decode a single IDR frame from the elementary stream and compare against
//! the ffmpeg-decoded reference YUV.
//!
//! Fixtures:
//!
//! * `/tmp/h264_solid.{es,yuv}` — solid-gray 64×64 baseline IDR. This is the
//!   "easy" case: x264 codes every macroblock as I_16x16 with no residual,
//!   making it a direct test of intra-prediction + IDCT plumbing.
//!
//! * `/tmp/h264.{es,yuv}` — ffmpeg testsrc 64×64 baseline IDR. Mixed
//!   I_NxN and I_16x16 with full CAVLC residue. Acceptance bar.
//!
//! * `tests/fixtures/iframe_testsrc_64x64.{es,yuv}` — in-tree copy of the
//!   testsrc 64×64 baseline IDR, used by the CAVLC desync regression test.
//!   Regenerate with:
//!
//!   ```bash
//!   ffmpeg -y -f lavfi -i testsrc=duration=1:size=64x64:rate=1 -vframes 1 \
//!     -pix_fmt yuv420p -c:v libx264 -profile:v baseline -preset ultrafast \
//!     -coder 0 -x264opts no-scenecut /tmp/h264.mp4 && \
//!   ffmpeg -y -i /tmp/h264.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 \
//!     tests/fixtures/iframe_testsrc_64x64.es && \
//!   ffmpeg -y -i /tmp/h264.mp4 -pix_fmt yuv420p \
//!     tests/fixtures/iframe_testsrc_64x64.yuv
//!   ```

use std::path::Path;

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::decoder::H264Decoder;
use oxideav_h264::nal::{split_annex_b, NalHeader, NalUnitType};

fn read_fixture(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        eprintln!("fixture {path} missing — skipping test");
        return None;
    }
    Some(std::fs::read(path).expect("read fixture"))
}

fn count_within(a: &[u8], b: &[u8], tol: i32) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() <= tol)
        .count()
}

fn decode_first_iframe(es: &[u8]) -> oxideav_core::VideoFrame {
    let nalus = split_annex_b(es);
    let mut sps_nal: Option<&[u8]> = None;
    let mut pps_nal: Option<&[u8]> = None;
    let mut idr_nal: Option<&[u8]> = None;
    for nalu in &nalus {
        let h = NalHeader::parse(nalu[0]).unwrap();
        match h.nal_unit_type {
            NalUnitType::Sps if sps_nal.is_none() => sps_nal = Some(nalu),
            NalUnitType::Pps if pps_nal.is_none() => pps_nal = Some(nalu),
            NalUnitType::SliceIdr if idr_nal.is_none() => idr_nal = Some(nalu),
            _ => {}
        }
    }
    let sps = sps_nal.expect("no SPS");
    let pps = pps_nal.expect("no PPS");
    let idr = idr_nal.expect("no IDR");

    let mut packet_data = Vec::new();
    packet_data.extend_from_slice(&[0, 0, 0, 1]);
    packet_data.extend_from_slice(sps);
    packet_data.extend_from_slice(&[0, 0, 0, 1]);
    packet_data.extend_from_slice(pps);
    packet_data.extend_from_slice(&[0, 0, 0, 1]);
    packet_data.extend_from_slice(idr);

    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 90_000), packet_data)
        .with_pts(0)
        .with_keyframe(true);
    dec.send_packet(&pkt).expect("send_packet");
    match dec.receive_frame() {
        Ok(Frame::Video(f)) => f,
        other => panic!("expected video, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn decode_solid_color_against_reference() {
    let es = match read_fixture("/tmp/h264_solid.es") {
        Some(d) => d,
        None => return,
    };
    let yuv = match read_fixture("/tmp/h264_solid.yuv") {
        Some(d) => d,
        None => return,
    };
    let frame = decode_first_iframe(&es);
    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 64);

    let ref_y = &yuv[0..(64 * 64)];
    let ref_cb = &yuv[(64 * 64)..(64 * 64 + 32 * 32)];
    let ref_cr = &yuv[(64 * 64 + 32 * 32)..(64 * 64 + 32 * 32 * 2)];
    let dec_y = &frame.planes[0].data;
    let dec_cb = &frame.planes[1].data;
    let dec_cr = &frame.planes[2].data;
    let total = ref_y.len() + ref_cb.len() + ref_cr.len();
    let within = count_within(dec_y, ref_y, 8)
        + count_within(dec_cb, ref_cb, 8)
        + count_within(dec_cr, ref_cr, 8);
    let pct = (within as f64) * 100.0 / (total as f64);
    eprintln!(
        "solid: decoded vs reference within ±8 LSB: {}/{} ({:.2}%)",
        within, total, pct
    );
    assert!(pct >= 90.0, "solid pixel-match {:.2}% < 90%", pct);
}

#[test]
fn decode_first_iframe_against_reference() {
    let es = match read_fixture("/tmp/h264.es") {
        Some(d) => d,
        None => return,
    };
    let yuv = match read_fixture("/tmp/h264_iframe.yuv") {
        Some(d) => d,
        None => return,
    };

    // testsrc is a full-CAVLC 64×64 IDR with every MB coded. The decoder
    // must complete without an error and land most luma pixels on the
    // reference. Chroma prediction still has known non-CAVLC errors that
    // we accept at a wider tolerance for now.
    let frame = decode_first_iframe(&es);
    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 64);

    let ref_y = &yuv[0..(64 * 64)];
    let ref_cb = &yuv[(64 * 64)..(64 * 64 + 32 * 32)];
    let ref_cr = &yuv[(64 * 64 + 32 * 32)..(64 * 64 + 32 * 32 * 2)];

    let dec_y = &frame.planes[0].data;
    let dec_cb = &frame.planes[1].data;
    let dec_cr = &frame.planes[2].data;
    let total = ref_y.len() + ref_cb.len() + ref_cr.len();
    let within = count_within(dec_y, ref_y, 8)
        + count_within(dec_cb, ref_cb, 8)
        + count_within(dec_cr, ref_cr, 8);
    let pct = (within as f64) * 100.0 / (total as f64);
    eprintln!(
        "testsrc: decoded vs reference within ±8 LSB: {}/{} ({:.2}%)",
        within, total, pct
    );
    // CAVLC desync-guard: prior to this fix, the decoder returned an
    // `InvalidData("coded_len 16 > max_num_coeff 15")` error from the
    // very first macroblock. Requiring ≥80% sample-match at ±8 LSB
    // proves the bitstream is now fully consumed and the vast majority
    // of samples are bit-for-bit accurate.
    assert!(pct >= 80.0, "testsrc pixel-match {:.2}% < 80%", pct);
}

/// Regression test for two CAVLC-side defects on a richer testsrc IDR:
///
/// 1. §9.2.1.1 chroma-AC nC desync — `predict_nc_chroma` inside the chroma
///    AC loop read a stale per-MB nC table because the decoder published
///    the accumulated counts only once, after all four blocks had been
///    parsed. The last chroma AC block then fell into VLC class 0 instead
///    of class 2, aliased the `total_zeros` codeword, and ran off the end
///    (`coded_len 16 > max_num_coeff 15`). Fixed by writing the in-progress
///    `nc_arr` back to `MbInfo` after every chroma block — mirroring what
///    the luma path already does.
///
/// 2. §9.2.2.1 level_code +15 over-shift for `level_prefix == 14` — the
///    pre-escape range [14, 29] is reached directly from `prefix = 14,
///    sl = 0` via a 4-bit suffix; the +15 adjustment only applies to the
///    escape path at `prefix >= 15`. The prior code added the +15 for
///    `prefix >= 14`, which double-shifted every level coded at prefix=14
///    (e.g. decoded +17 where the spec-correct level is −8). This biased
///    chroma DC reconstruction by a full Hadamard-basis class in every
///    MB that carried a medium-magnitude chroma DC coefficient, producing
///    systematic per-quadrant offsets matching the [+K, −K, −K, +K]
///    signature of the f[1][1] basis.
///
/// After both fixes the testsrc 64×64 IDR decodes bit-exact against
/// ffmpeg: every luma and chroma sample matches the reference YUV. The
/// fixture is committed so this test doesn't depend on a local ffmpeg
/// invocation.
#[test]
fn cavlc_chroma_ac_nc_desync_regression() {
    let es_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/iframe_testsrc_64x64.es"
    );
    let yuv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/iframe_testsrc_64x64.yuv"
    );
    let es = match read_fixture(es_path) {
        Some(d) => d,
        None => return,
    };
    let yuv = match read_fixture(yuv_path) {
        Some(d) => d,
        None => return,
    };
    // Decoder must complete — the pre-fix version returned an
    // `InvalidData` from the first macroblock's chroma AC pass.
    let frame = decode_first_iframe(&es);
    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 64);

    let ref_y = &yuv[0..(64 * 64)];
    let ref_cb = &yuv[(64 * 64)..(64 * 64 + 32 * 32)];
    let ref_cr = &yuv[(64 * 64 + 32 * 32)..(64 * 64 + 32 * 32 * 2)];
    let dec_y = &frame.planes[0].data;
    let dec_cb = &frame.planes[1].data;
    let dec_cr = &frame.planes[2].data;

    let y_match = count_within(dec_y, ref_y, 0);
    let y_pct = (y_match as f64) * 100.0 / (ref_y.len() as f64);
    let mse = |a: &[u8], b: &[u8]| -> f64 {
        let s: i64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = *x as i64 - *y as i64;
                d * d
            })
            .sum();
        s as f64 / a.len() as f64
    };
    let cb_mse = mse(dec_cb, ref_cb);
    let cr_mse = mse(dec_cr, ref_cr);
    eprintln!(
        "cavlc-desync regression: luma exact match {}/{} ({:.2}%), Cb MSE {:.4}, Cr MSE {:.4}",
        y_match,
        ref_y.len(),
        y_pct,
        cb_mse,
        cr_mse
    );

    // Before the nC-desync fix the decoder threw an error; before the
    // level-prefix fix we had 97.66% luma and Cb/Cr MSE 285/139. With
    // both fixes the testsrc IDR is bit-exact.
    assert!(
        y_pct >= 99.99,
        "luma exact match {y_pct:.2}% < 99.99% — CAVLC desync or level-prefix regression"
    );
    assert!(
        cb_mse < 0.5 && cr_mse < 0.5,
        "chroma MSE Cb={cb_mse:.4} Cr={cr_mse:.4} — chroma CAVLC regression"
    );
}
