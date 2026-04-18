//! 10-bit CABAC I-slice decode — compares the decoded u16 planes against
//! the ffmpeg `-pix_fmt yuv420p10le` reference YUV shipped under
//! `tests/fixtures/10bit_cabac_i_64x64.{es,yuv}`.
//!
//! The fixture was produced with:
//!
//! ```bash
//! ffmpeg -f lavfi -i testsrc=duration=1:size=64x64:rate=1 \
//!   -profile:v high10 -preset ultrafast -coder 1 -vframes 1 \
//!   -pix_fmt yuv420p10le /tmp/10bit_cabac_i.mp4
//! ffmpeg -i /tmp/10bit_cabac_i.mp4 -c copy -bsf:v h264_mp4toannexb \
//!   -f h264 tests/fixtures/10bit_cabac_i_64x64.es
//! ffmpeg -i /tmp/10bit_cabac_i.mp4 -f rawvideo -pix_fmt yuv420p10le \
//!   tests/fixtures/10bit_cabac_i_64x64.yuv
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
fn decode_10bit_cabac_iframe_against_reference() {
    let es_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/10bit_cabac_i_64x64.es"
    );
    let yuv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/10bit_cabac_i_64x64.yuv"
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

    let exact_count = |a: &[u16], b: &[u16]| -> usize {
        a.iter().zip(b.iter()).filter(|(x, y)| x == y).count()
    };

    let total = ref_y.len() + ref_cb.len() + ref_cr.len();
    let exact = exact_count(&dec_y, &ref_y)
        + exact_count(&dec_cb, &ref_cb)
        + exact_count(&dec_cr, &ref_cr);

    let exact_pct = (exact as f64) * 100.0 / (total as f64);
    eprintln!(
        "10bit CABAC I: bit-exact {}/{} ({:.2}%)",
        exact, total, exact_pct
    );

    assert!(
        exact_pct >= 99.0,
        "10bit CABAC I bit-exact {exact_pct:.2}% < 99% — CABAC 10-bit I regression"
    );
}
