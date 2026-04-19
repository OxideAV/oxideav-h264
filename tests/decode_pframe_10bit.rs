//! 10-bit CAVLC P-slice decode — compares the decoded u16 planes
//! against the ffmpeg `-pix_fmt yuv420p10le` reference YUV shipped
//! under `tests/fixtures/pframe_10bit_64x64.{es,yuv}`.
//!
//! The fixture was produced with:
//!
//! ```bash
//! ffmpeg -f lavfi -i testsrc=duration=1:size=64x64:rate=12 -vframes 3 \
//!   -pix_fmt yuv420p10le -c:v libx264 -profile:v high10 -preset ultrafast \
//!   -x264-params "bframes=0:keyint=3:no-scenecut=1" /tmp/p_10bit.mp4
//! ffmpeg -i /tmp/p_10bit.mp4 -c copy -bsf:v h264_mp4toannexb \
//!   -f h264 tests/fixtures/pframe_10bit_64x64.es
//! ffmpeg -i /tmp/p_10bit.mp4 -f rawvideo -pix_fmt yuv420p10le \
//!   tests/fixtures/pframe_10bit_64x64.yuv
//! ```

use std::path::Path;

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, PixelFormat, TimeBase, VideoFrame};
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

fn decode_all_frames(es: &[u8]) -> Vec<VideoFrame> {
    let nalus = split_annex_b(es);
    let mut sps_nal: Option<&[u8]> = None;
    let mut pps_nal: Option<&[u8]> = None;
    let mut slice_nals: Vec<&[u8]> = Vec::new();
    for nalu in &nalus {
        let h = NalHeader::parse(nalu[0]).unwrap();
        match h.nal_unit_type {
            NalUnitType::Sps if sps_nal.is_none() => sps_nal = Some(nalu),
            NalUnitType::Pps if pps_nal.is_none() => pps_nal = Some(nalu),
            NalUnitType::SliceIdr | NalUnitType::SliceNonIdr => slice_nals.push(nalu),
            _ => {}
        }
    }
    let sps = sps_nal.expect("no SPS");
    let pps = pps_nal.expect("no PPS");
    assert!(!slice_nals.is_empty(), "no slice NALs");

    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let mut frames = Vec::new();
    for (idx, slice) in slice_nals.iter().enumerate() {
        let mut packet_data = Vec::new();
        if idx == 0 {
            packet_data.extend_from_slice(&[0, 0, 0, 1]);
            packet_data.extend_from_slice(sps);
            packet_data.extend_from_slice(&[0, 0, 0, 1]);
            packet_data.extend_from_slice(pps);
        }
        packet_data.extend_from_slice(&[0, 0, 0, 1]);
        packet_data.extend_from_slice(slice);
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), packet_data)
            .with_pts(idx as i64)
            .with_keyframe(idx == 0);
        dec.send_packet(&pkt).expect("send_packet");
        while let Ok(Frame::Video(f)) = dec.receive_frame() {
            frames.push(f);
        }
    }
    dec.flush().expect("flush");
    while let Ok(Frame::Video(f)) = dec.receive_frame() {
        frames.push(f);
    }
    frames
}

#[test]
fn decode_10bit_pframes_against_reference() {
    let es_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/pframe_10bit_64x64.es"
    );
    let yuv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/pframe_10bit_64x64.yuv"
    );
    let es = match read_fixture(es_path) {
        Some(d) => d,
        None => return,
    };
    let yuv = match read_fixture(yuv_path) {
        Some(d) => d,
        None => return,
    };

    let frames = decode_all_frames(&es);
    assert_eq!(frames.len(), 3, "expected 3 decoded frames");
    for f in &frames {
        assert_eq!(f.width, 64);
        assert_eq!(f.height, 64);
        assert_eq!(f.format, PixelFormat::Yuv420P10Le);
    }

    let y_plane_bytes = 64 * 64 * 2;
    let c_plane_bytes = 32 * 32 * 2;
    let frame_bytes = y_plane_bytes + 2 * c_plane_bytes;
    assert_eq!(yuv.len(), 3 * frame_bytes);

    let close_count = |a: &[u16], b: &[u16], tol: i32| -> usize {
        a.iter()
            .zip(b.iter())
            .filter(|(x, y)| (**x as i32 - **y as i32).abs() <= tol)
            .count()
    };
    let exact_count =
        |a: &[u16], b: &[u16]| -> usize { a.iter().zip(b.iter()).filter(|(x, y)| x == y).count() };

    let mut total = 0usize;
    let mut total_exact = 0usize;
    let mut total_within4 = 0usize;
    for (i, frame) in frames.iter().enumerate() {
        let ref_off = i * frame_bytes;
        let ref_y = unpack_u16_le(&yuv[ref_off..ref_off + y_plane_bytes]);
        let ref_cb =
            unpack_u16_le(&yuv[ref_off + y_plane_bytes..ref_off + y_plane_bytes + c_plane_bytes]);
        let ref_cr =
            unpack_u16_le(&yuv[ref_off + y_plane_bytes + c_plane_bytes..ref_off + frame_bytes]);

        let dec_y = unpack_u16_le(&frame.planes[0].data);
        let dec_cb = unpack_u16_le(&frame.planes[1].data);
        let dec_cr = unpack_u16_le(&frame.planes[2].data);

        assert_eq!(dec_y.len(), ref_y.len());
        assert_eq!(dec_cb.len(), ref_cb.len());
        assert_eq!(dec_cr.len(), ref_cr.len());

        let frame_total = ref_y.len() + ref_cb.len() + ref_cr.len();
        let frame_exact = exact_count(&dec_y, &ref_y)
            + exact_count(&dec_cb, &ref_cb)
            + exact_count(&dec_cr, &ref_cr);
        let frame_within4 = close_count(&dec_y, &ref_y, 4)
            + close_count(&dec_cb, &ref_cb, 4)
            + close_count(&dec_cr, &ref_cr, 4);

        eprintln!(
            "10bit P-frame #{i}: bit-exact {}/{} ({:.2}%); within ±4 LSB {}/{} ({:.2}%)",
            frame_exact,
            frame_total,
            (frame_exact as f64) * 100.0 / (frame_total as f64),
            frame_within4,
            frame_total,
            (frame_within4 as f64) * 100.0 / (frame_total as f64),
        );

        total += frame_total;
        total_exact += frame_exact;
        total_within4 += frame_within4;
    }

    let exact_pct = (total_exact as f64) * 100.0 / (total as f64);
    let within4_pct = (total_within4 as f64) * 100.0 / (total as f64);
    eprintln!(
        "10bit (all 3 frames): bit-exact {}/{} ({:.2}%); within ±4 LSB {}/{} ({:.2}%)",
        total_exact, total, exact_pct, total_within4, total, within4_pct,
    );

    assert!(
        exact_pct >= 99.0,
        "10bit P-slice bit-exact rate {exact_pct:.2}% < 99% — decode regression"
    );
}
