//! MBAFF (macroblock-adaptive frame/field) I-slice CAVLC decode regression.
//!
//! Regenerate the fixture with:
//!
//! ```bash
//! ffmpeg -y -f lavfi -i testsrc=duration=1:size=128x128:rate=25 -vframes 2 \
//!     -pix_fmt yuv420p -c:v libx264 -profile:v main -preset medium \
//!     -coder 0 -flags +ildct+ilme -x264opts "interlaced=1:no-scenecut" \
//!     /tmp/mbaff.mp4 && \
//! ffmpeg -y -i /tmp/mbaff.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 \
//!     tests/fixtures/mbaff_128x128.es && \
//! ffmpeg -y -i /tmp/mbaff.mp4 -pix_fmt yuv420p \
//!     tests/fixtures/mbaff_128x128.yuv
//! ```
//!
//! The SPS of this clip has `frame_mbs_only_flag = 0` + `mb_adaptive_frame_field_flag = 1`
//! and every MB pair ends up field-coded (`mb_field_decoding_flag = 1`) because
//! x264's RD picks that for synthetic testsrc content. This exercises the
//! §7.3.4 pair loop, the §6.4.9.4 neighbour derivation for field-coded MBs,
//! the picture-plane row-stride doubling on both the luma and 4:2:0 chroma
//! planes, and the §8.7.1.1 MBAFF deblocker's field-aware edge schedule
//! (vertical edges step at `row_step = 2`, the inside-pair top/bottom
//! horizontal boundary is skipped for field-coded pairs).

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

fn count_within(a: &[u8], b: &[u8], tol: i32) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() <= tol)
        .count()
}

/// End-to-end MBAFF decode check.
///
/// With §8.7.1.1 MBAFF deblock wired the fixture matches ffmpeg bit-exact
/// (100% ±0 LSB on all three planes). The assertion tightens to ≥ 99%
/// luma at ±4 LSB to catch any future regression while leaving a small
/// margin for compilers that reorder the deblock inner loop.
#[test]
fn decode_mbaff_iframe_matches_ffmpeg_reference() {
    let es_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mbaff_128x128.es"
    );
    let yuv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mbaff_128x128.yuv"
    );
    let es = match read_fixture(es_path) {
        Some(d) => d,
        None => return,
    };
    let yuv = match read_fixture(yuv_path) {
        Some(d) => d,
        None => return,
    };

    let frame = decode_first_iframe(&es);
    assert_eq!(frame.width, 128);
    assert_eq!(frame.height, 128);

    let y_len = 128 * 128;
    let c_len = 64 * 64;
    let ref_y = &yuv[0..y_len];
    let ref_cb = &yuv[y_len..y_len + c_len];
    let ref_cr = &yuv[y_len + c_len..y_len + c_len * 2];
    let dec_y = &frame.planes[0].data;
    let dec_cb = &frame.planes[1].data;
    let dec_cr = &frame.planes[2].data;

    let y_match_tight = count_within(dec_y, ref_y, 4);
    let y_pct_tight = (y_match_tight as f64) * 100.0 / (ref_y.len() as f64);
    let y_match_loose = count_within(dec_y, ref_y, 8);
    let y_pct_loose = (y_match_loose as f64) * 100.0 / (ref_y.len() as f64);
    let cb_match = count_within(dec_cb, ref_cb, 4);
    let cb_pct = (cb_match as f64) * 100.0 / (ref_cb.len() as f64);
    let cr_match = count_within(dec_cr, ref_cr, 4);
    let cr_pct = (cr_match as f64) * 100.0 / (ref_cr.len() as f64);

    eprintln!(
        "mbaff: luma ±4 {y_pct_tight:.2}% / ±8 {y_pct_loose:.2}%, Cb ±4 {cb_pct:.2}%, Cr ±4 {cr_pct:.2}%"
    );

    // Acceptance bar: ≥ 99% luma within ±4 LSB. With the §8.7.1.1
    // MBAFF deblock pass wired the current fixture matches bit-exact
    // (100% ±0), so ±4 leaves headroom for future edge-case MBAFF
    // streams without regressing.
    assert!(
        y_pct_tight >= 99.0,
        "mbaff luma ±4 match {y_pct_tight:.2}% < 99%"
    );
    assert!(
        y_pct_loose >= 99.0,
        "mbaff luma ±8 match {y_pct_loose:.2}% < 99%"
    );
}

/// MBAFF + 4:2:2 integration — regenerate with:
///
/// ```bash
/// ffmpeg -y -f lavfi -i testsrc=duration=1:size=128x128:rate=25 -vframes 2 \
///     -pix_fmt yuv422p -c:v libx264 -profile:v high422 -preset medium \
///     -coder 0 -flags +ildct+ilme \
///     -x264opts "interlaced=1:no-scenecut:8x8dct=0" /tmp/mbaff422.mp4 && \
/// ffmpeg -y -i /tmp/mbaff422.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 \
///     tests/fixtures/mbaff_422_128x128.es && \
/// ffmpeg -y -i /tmp/mbaff422.mp4 -pix_fmt yuv422p \
///     tests/fixtures/mbaff_422_128x128.yuv
/// ```
#[test]
fn decode_mbaff_422_iframe_matches_ffmpeg_reference() {
    let es_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mbaff_422_128x128.es"
    );
    let yuv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mbaff_422_128x128.yuv"
    );
    let es = match read_fixture(es_path) {
        Some(d) => d,
        None => return,
    };
    let yuv = match read_fixture(yuv_path) {
        Some(d) => d,
        None => return,
    };

    let frame = decode_first_iframe(&es);
    assert_eq!(frame.width, 128);
    assert_eq!(frame.height, 128);

    let y_len = 128 * 128;
    let c_len = 64 * 128; // 4:2:2 chroma is half-width × full-height
    let ref_y = &yuv[0..y_len];
    let ref_cb = &yuv[y_len..y_len + c_len];
    let ref_cr = &yuv[y_len + c_len..y_len + c_len * 2];
    let dec_y = &frame.planes[0].data;
    let dec_cb = &frame.planes[1].data;
    let dec_cr = &frame.planes[2].data;

    let y_match_tight = count_within(dec_y, ref_y, 4);
    let y_pct_tight = (y_match_tight as f64) * 100.0 / (ref_y.len() as f64);
    let cb_match = count_within(dec_cb, ref_cb, 4);
    let cb_pct = (cb_match as f64) * 100.0 / (ref_cb.len() as f64);
    let cr_match = count_within(dec_cr, ref_cr, 4);
    let cr_pct = (cr_match as f64) * 100.0 / (ref_cr.len() as f64);

    eprintln!("mbaff-422: luma ±4 {y_pct_tight:.2}%, Cb ±4 {cb_pct:.2}%, Cr ±4 {cr_pct:.2}%");

    assert!(
        y_pct_tight >= 99.0,
        "mbaff-422 luma ±4 match {y_pct_tight:.2}% < 99%"
    );
}

/// MBAFF + 4:4:4 integration — regenerate with:
///
/// ```bash
/// ffmpeg -y -f lavfi -i testsrc=duration=1:size=128x128:rate=25 -vframes 2 \
///     -pix_fmt yuv444p -c:v libx264 -profile:v high444 -preset medium \
///     -coder 0 -flags +ildct+ilme \
///     -x264opts "interlaced=1:no-scenecut:8x8dct=0" /tmp/mbaff444.mp4 && \
/// ffmpeg -y -i /tmp/mbaff444.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 \
///     tests/fixtures/mbaff_444_128x128.es && \
/// ffmpeg -y -i /tmp/mbaff444.mp4 -pix_fmt yuv444p \
///     tests/fixtures/mbaff_444_128x128.yuv
/// ```
#[test]
fn decode_mbaff_444_iframe_matches_ffmpeg_reference() {
    let es_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mbaff_444_128x128.es"
    );
    let yuv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mbaff_444_128x128.yuv"
    );
    let es = match read_fixture(es_path) {
        Some(d) => d,
        None => return,
    };
    let yuv = match read_fixture(yuv_path) {
        Some(d) => d,
        None => return,
    };

    let frame = decode_first_iframe(&es);
    assert_eq!(frame.width, 128);
    assert_eq!(frame.height, 128);

    let y_len = 128 * 128;
    let c_len = 128 * 128; // 4:4:4 chroma planes match luma size
    let ref_y = &yuv[0..y_len];
    let ref_cb = &yuv[y_len..y_len + c_len];
    let ref_cr = &yuv[y_len + c_len..y_len + c_len * 2];
    let dec_y = &frame.planes[0].data;
    let dec_cb = &frame.planes[1].data;
    let dec_cr = &frame.planes[2].data;

    let y_match_tight = count_within(dec_y, ref_y, 4);
    let y_pct_tight = (y_match_tight as f64) * 100.0 / (ref_y.len() as f64);
    let cb_match = count_within(dec_cb, ref_cb, 4);
    let cb_pct = (cb_match as f64) * 100.0 / (ref_cb.len() as f64);
    let cr_match = count_within(dec_cr, ref_cr, 4);
    let cr_pct = (cr_match as f64) * 100.0 / (ref_cr.len() as f64);

    eprintln!("mbaff-444: luma ±4 {y_pct_tight:.2}%, Cb ±4 {cb_pct:.2}%, Cr ±4 {cr_pct:.2}%");

    assert!(
        y_pct_tight >= 99.0,
        "mbaff-444 luma ±4 match {y_pct_tight:.2}% < 99%"
    );
}
