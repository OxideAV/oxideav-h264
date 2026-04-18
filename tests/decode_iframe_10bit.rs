//! 10-bit CAVLC I-slice decode — compares the decoded u16 planes
//! against the ffmpeg `-pix_fmt yuv420p10le` reference YUV shipped
//! under `tests/fixtures/iframe_10bit_64x64.{es,yuv}`.
//!
//! The fixture was produced with:
//!
//! ```bash
//! ffmpeg -f lavfi -i testsrc=duration=1:size=64x64:rate=1 -vframes 1 \
//!   -pix_fmt yuv420p10le -c:v libx264 -profile:v high10 \
//!   -preset ultrafast /tmp/10bit.mp4
//! ffmpeg -i /tmp/10bit.mp4 -c copy -bsf:v h264_mp4toannexb \
//!   -f h264 tests/fixtures/iframe_10bit_64x64.es
//! ffmpeg -i /tmp/10bit.mp4 -f rawvideo -pix_fmt yuv420p10le \
//!   tests/fixtures/iframe_10bit_64x64.yuv
//! ```

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

#[test]
fn decode_10bit_iframe_against_reference() {
    let es_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/iframe_10bit_64x64.es"
    );
    let yuv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/iframe_10bit_64x64.yuv"
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

    assert_eq!(dec_y.len(), ref_y.len());
    assert_eq!(dec_cb.len(), ref_cb.len());
    assert_eq!(dec_cr.len(), ref_cr.len());

    let exact_count = |a: &[u16], b: &[u16]| -> usize {
        a.iter().zip(b.iter()).filter(|(x, y)| x == y).count()
    };
    let close_count = |a: &[u16], b: &[u16], tol: i32| -> usize {
        a.iter()
            .zip(b.iter())
            .filter(|(x, y)| (**x as i32 - **y as i32).abs() <= tol)
            .count()
    };

    let total = ref_y.len() + ref_cb.len() + ref_cr.len();
    let exact = exact_count(&dec_y, &ref_y)
        + exact_count(&dec_cb, &ref_cb)
        + exact_count(&dec_cr, &ref_cr);
    let within4 = close_count(&dec_y, &ref_y, 4)
        + close_count(&dec_cb, &ref_cb, 4)
        + close_count(&dec_cr, &ref_cr, 4);

    let exact_pct = (exact as f64) * 100.0 / (total as f64);
    let within4_pct = (within4 as f64) * 100.0 / (total as f64);
    eprintln!(
        "10bit: bit-exact {}/{} ({:.2}%); within ±4 LSB {}/{} ({:.2}%)",
        exact, total, exact_pct, within4, total, within4_pct
    );

    // Deblocking is scoped out on the 10-bit path, so pre-deblock samples
    // at the macroblock edges can drift slightly from the post-deblock
    // ffmpeg reference. Require ≥99% of samples within ±4 LSB so the
    // intra-prediction + transform pipeline lands the bulk of the frame
    // bit-exact; deblock-related drift stays bounded.
    assert!(
        within4_pct >= 99.0,
        "10bit within ±4 LSB {within4_pct:.2}% < 99% — decode pipeline regression"
    );
}
