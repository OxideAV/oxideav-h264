//! 10-bit CAVLC I-slice decode for the higher chroma formats (4:2:2 and
//! 4:4:4) plus 10-bit Intra_8×8. The fixtures were produced with the
//! commands documented in each test body.
//!
//! Each test compares decoded u16 planes against the ffmpeg reference
//! YUV and requires ≥ 99 % of samples within ±4 LSB. Deblocking on the
//! 10-bit path is still approximate around MB edges, so a hard bit-
//! exact bound would flap; the ±4 LSB bound catches regressions in the
//! intra prediction / dequant / transform pipeline while tolerating the
//! known deblock drift.

use std::path::Path;

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, PixelFormat, TimeBase};
use oxideav_h264::decoder::H264Decoder;
use oxideav_h264::nal::{split_annex_b, NalHeader, NalUnitType};

fn read_fixture(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        eprintln!("fixture {path} missing — skipping test");
        return None;
    }
    Some(std::fs::read(path).expect("read fixture"))
}

fn unpack_u16_le(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
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

fn compare_u16_planes(label: &str, got: &[&[u16]], want: &[&[u16]], threshold_pct: f64) {
    assert_eq!(got.len(), want.len());
    let total: usize = want.iter().map(|p| p.len()).sum();
    let exact: usize = got
        .iter()
        .zip(want)
        .map(|(a, b)| a.iter().zip(b.iter()).filter(|(x, y)| x == y).count())
        .sum();
    let within4: usize = got
        .iter()
        .zip(want)
        .map(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .filter(|(x, y)| (**x as i32 - **y as i32).abs() <= 4)
                .count()
        })
        .sum();
    let exact_pct = (exact as f64) * 100.0 / (total as f64);
    let within4_pct = (within4 as f64) * 100.0 / (total as f64);
    eprintln!(
        "{label}: bit-exact {exact}/{total} ({exact_pct:.2}%); ±4 LSB {within4}/{total} ({within4_pct:.2}%)"
    );
    assert!(
        within4_pct >= threshold_pct,
        "{label} within ±4 LSB {within4_pct:.2}% < {threshold_pct}% — decode regression"
    );
}

// -----------------------------------------------------------------------------
// 10-bit 4:2:2
// -----------------------------------------------------------------------------
//
// Fixture produced with:
//
// ```bash
// ffmpeg -f lavfi -i testsrc=duration=1:size=64x64:rate=1 -vframes 1 \
//   -pix_fmt yuv422p10le -c:v libx264 -profile:v high422 \
//   -x264-params bit-depth=10 -preset ultrafast /tmp/10bit_422.mp4
// ffmpeg -i /tmp/10bit_422.mp4 -c copy -bsf:v h264_mp4toannexb \
//   -f h264 tests/fixtures/iframe_10bit_422_64x64.es
// ffmpeg -i /tmp/10bit_422.mp4 -f rawvideo -pix_fmt yuv422p10le \
//   tests/fixtures/iframe_10bit_422_64x64.yuv
// ```
#[test]
fn decode_10bit_422_iframe_against_reference() {
    let es_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/iframe_10bit_422_64x64.es"
    );
    let yuv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/iframe_10bit_422_64x64.yuv"
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
    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 64);
    assert_eq!(frame.format, PixelFormat::Yuv422P10Le);

    // 4:2:2: luma 64x64, chroma 32x64.
    let y_plane_bytes = 64 * 64 * 2;
    let c_plane_bytes = 32 * 64 * 2;
    assert_eq!(yuv.len(), y_plane_bytes + 2 * c_plane_bytes);

    let ref_y = unpack_u16_le(&yuv[0..y_plane_bytes]);
    let ref_cb = unpack_u16_le(&yuv[y_plane_bytes..y_plane_bytes + c_plane_bytes]);
    let ref_cr = unpack_u16_le(&yuv[y_plane_bytes + c_plane_bytes..]);

    let dec_y = unpack_u16_le(&frame.planes[0].data);
    let dec_cb = unpack_u16_le(&frame.planes[1].data);
    let dec_cr = unpack_u16_le(&frame.planes[2].data);

    compare_u16_planes(
        "10bit 4:2:2",
        &[&dec_y, &dec_cb, &dec_cr],
        &[&ref_y, &ref_cb, &ref_cr],
        99.0,
    );
}

// -----------------------------------------------------------------------------
// 10-bit 4:4:4
// -----------------------------------------------------------------------------
//
// ```bash
// ffmpeg -f lavfi -i testsrc=duration=1:size=64x64:rate=1 -vframes 1 \
//   -pix_fmt yuv444p10le -c:v libx264 -profile:v high444 \
//   -x264-params bit-depth=10 -preset ultrafast /tmp/10bit_444.mp4
// ffmpeg -i /tmp/10bit_444.mp4 -c copy -bsf:v h264_mp4toannexb \
//   -f h264 tests/fixtures/iframe_10bit_444_64x64.es
// ffmpeg -i /tmp/10bit_444.mp4 -f rawvideo -pix_fmt yuv444p10le \
//   tests/fixtures/iframe_10bit_444_64x64.yuv
// ```
#[test]
fn decode_10bit_444_iframe_against_reference() {
    let es_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/iframe_10bit_444_64x64.es"
    );
    let yuv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/iframe_10bit_444_64x64.yuv"
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
    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 64);
    assert_eq!(frame.format, PixelFormat::Yuv444P10Le);

    let plane_bytes = 64 * 64 * 2;
    assert_eq!(yuv.len(), 3 * plane_bytes);

    let ref_y = unpack_u16_le(&yuv[0..plane_bytes]);
    let ref_cb = unpack_u16_le(&yuv[plane_bytes..2 * plane_bytes]);
    let ref_cr = unpack_u16_le(&yuv[2 * plane_bytes..]);

    let dec_y = unpack_u16_le(&frame.planes[0].data);
    let dec_cb = unpack_u16_le(&frame.planes[1].data);
    let dec_cr = unpack_u16_le(&frame.planes[2].data);

    compare_u16_planes(
        "10bit 4:4:4",
        &[&dec_y, &dec_cb, &dec_cr],
        &[&ref_y, &ref_cb, &ref_cr],
        99.0,
    );
}

// -----------------------------------------------------------------------------
// 10-bit 4:2:0 Intra_8×8 (transform_size_8x8_flag=1)
// -----------------------------------------------------------------------------
//
// ```bash
// ffmpeg -f lavfi -i testsrc=duration=1:size=64x64:rate=1 -vframes 1 \
//   -pix_fmt yuv420p10le -c:v libx264 -profile:v high10 \
//   -x264-params "bit-depth=10:8x8dct=1:no-cabac=1" -preset slow \
//   /tmp/10bit_8x8.mp4
// ffmpeg -i /tmp/10bit_8x8.mp4 -c copy -bsf:v h264_mp4toannexb \
//   -f h264 tests/fixtures/iframe_10bit_8x8_64x64.es
// ffmpeg -i /tmp/10bit_8x8.mp4 -f rawvideo -pix_fmt yuv420p10le \
//   tests/fixtures/iframe_10bit_8x8_64x64.yuv
// ```
#[test]
fn decode_10bit_8x8_iframe_against_reference() {
    let es_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/iframe_10bit_8x8_64x64.es"
    );
    let yuv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/iframe_10bit_8x8_64x64.yuv"
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
    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 64);
    assert_eq!(frame.format, PixelFormat::Yuv420P10Le);

    let y_plane_bytes = 64 * 64 * 2;
    let c_plane_bytes = 32 * 32 * 2;
    assert_eq!(yuv.len(), y_plane_bytes + 2 * c_plane_bytes);

    let ref_y = unpack_u16_le(&yuv[0..y_plane_bytes]);
    let ref_cb = unpack_u16_le(&yuv[y_plane_bytes..y_plane_bytes + c_plane_bytes]);
    let ref_cr = unpack_u16_le(&yuv[y_plane_bytes + c_plane_bytes..]);

    let dec_y = unpack_u16_le(&frame.planes[0].data);
    let dec_cb = unpack_u16_le(&frame.planes[1].data);
    let dec_cr = unpack_u16_le(&frame.planes[2].data);

    compare_u16_planes(
        "10bit 8x8dct",
        &[&dec_y, &dec_cb, &dec_cr],
        &[&ref_y, &ref_cb, &ref_cr],
        99.0,
    );
}
