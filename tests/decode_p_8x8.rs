//! Decode a 128×128 CAVLC P-slice clip produced with `8x8dct=1` and compare
//! the decoded frames against ffmpeg's reference YUV. Exercises the CAVLC
//! inter 8×8 integer-transform path (§8.5.13), the conditional §7.3.5.1
//! `transform_size_8x8_flag` parse, and the FFmpeg-style per-sub-block
//! `non_zero_count_cache` neighbour indexing (`libavcodec/h264_cavlc.c`).
//!
//! Fixture regeneration:
//!
//! ```bash
//! ffmpeg -y -f lavfi -i "testsrc=duration=1:size=128x128:rate=12" -vframes 4 \
//!     -pix_fmt yuv420p -c:v libx264 -profile:v high -preset ultrafast \
//!     -coder 0 -x264opts "8x8dct=1:no-scenecut:bframes=0:keyint=4:min-keyint=4" \
//!     /tmp/p8x8.mp4
//! ffmpeg -y -i /tmp/p8x8.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 \
//!     tests/fixtures/p_8x8_cavlc_128x128.es
//! ffmpeg -y -i /tmp/p8x8.mp4 -f rawvideo -pix_fmt yuv420p \
//!     tests/fixtures/p_8x8_cavlc_128x128.yuv
//! ```

use std::path::PathBuf;

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::decoder::H264Decoder;
use oxideav_h264::nal::{split_annex_b, NalHeader, NalUnitType};

const W: usize = 128;
const H: usize = 128;
const FRAME_BYTES: usize = W * H * 3 / 2;

fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    p
}

fn read_fixture(name: &str) -> Option<Vec<u8>> {
    let p = fixture_path(name);
    if !p.exists() {
        eprintln!("fixture {:?} missing — skipping test", p);
        return None;
    }
    Some(std::fs::read(&p).expect("read fixture"))
}

fn count_within(a: &[u8], b: &[u8], tol: i32) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() <= tol)
        .count()
}

fn split_frames(es: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<Vec<u8>>) {
    let nalus = split_annex_b(es);
    let mut sps = None;
    let mut pps = None;
    let mut frames: Vec<Vec<u8>> = Vec::new();
    for nalu in &nalus {
        let h = NalHeader::parse(nalu[0]).unwrap();
        match h.nal_unit_type {
            NalUnitType::Sps => sps = Some(nalu.to_vec()),
            NalUnitType::Pps => pps = Some(nalu.to_vec()),
            NalUnitType::SliceIdr | NalUnitType::SliceNonIdr => {
                let mut frame = Vec::new();
                frame.extend_from_slice(&[0, 0, 0, 1]);
                frame.extend_from_slice(nalu);
                frames.push(frame);
            }
            _ => {}
        }
    }
    (sps.unwrap(), pps.unwrap(), frames)
}

#[test]
fn decode_p_slice_8x8_transform_against_reference() {
    let es = match read_fixture("p_8x8_cavlc_128x128.es") {
        Some(d) => d,
        None => return,
    };
    let yuv_all = match read_fixture("p_8x8_cavlc_128x128.yuv") {
        Some(d) => d,
        None => return,
    };
    assert!(yuv_all.len() >= FRAME_BYTES, "reference YUV too small");

    let (sps, pps, frame_nalus) = split_frames(&es);
    assert!(
        frame_nalus.len() >= 2,
        "need at least 2 slice NALUs (I + P), got {}",
        frame_nalus.len()
    );

    let mut dec = H264Decoder::new(CodecId::new("h264"));

    let mut primer = Vec::new();
    primer.extend_from_slice(&[0, 0, 0, 1]);
    primer.extend_from_slice(&sps);
    primer.extend_from_slice(&[0, 0, 0, 1]);
    primer.extend_from_slice(&pps);
    let primer_pkt = Packet::new(0, TimeBase::new(1, 90_000), primer);
    dec.send_packet(&primer_pkt).expect("primer send_packet");
    while dec.receive_frame().is_ok() {}

    let mut decoded: Vec<Vec<u8>> = Vec::new();
    for (idx, frame) in frame_nalus.iter().enumerate() {
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), frame.clone())
            .with_pts(idx as i64)
            .with_keyframe(idx == 0);
        dec.send_packet(&pkt).expect("send_packet");
        let f = match dec.receive_frame().expect("receive_frame") {
            Frame::Video(f) => f,
            other => panic!("expected video, got {:?}", std::mem::discriminant(&other)),
        };
        assert_eq!(f.width, W as u32);
        assert_eq!(f.height, H as u32);
        let mut buf = Vec::with_capacity(FRAME_BYTES);
        buf.extend_from_slice(&f.planes[0].data);
        buf.extend_from_slice(&f.planes[1].data);
        buf.extend_from_slice(&f.planes[2].data);
        decoded.push(buf);
    }

    assert!(
        decoded.len() == frame_nalus.len(),
        "decoded {} of {} frames",
        decoded.len(),
        frame_nalus.len()
    );

    // Compare luma planes — the 8×8 transform path is the subject; chroma
    // uses the existing 4×4 path. Aim for bit-exact (±0) on each frame; the
    // ultrafast preset + no-deblock fixture produces an integer match with
    // the ffmpeg reference when dequant + IDCT are correct.
    let luma_bytes = W * H;
    let mut total_match = 0usize;
    let mut total_samples = 0usize;
    for (i, dec_buf) in decoded.iter().enumerate() {
        let ref_off = i * FRAME_BYTES;
        let ref_buf = &yuv_all[ref_off..ref_off + FRAME_BYTES];
        let m_y = count_within(&dec_buf[..luma_bytes], &ref_buf[..luma_bytes], 0);
        let pct = (m_y as f64) * 100.0 / (luma_bytes as f64);
        eprintln!(
            "P 8×8 frame {i}: luma bit-exact {}/{} ({:.2}%)",
            m_y, luma_bytes, pct
        );
        total_match += m_y;
        total_samples += luma_bytes;
    }
    let overall = (total_match as f64) * 100.0 / (total_samples as f64);
    eprintln!(
        "P 8×8 luma bit-exact overall: {:.2}% over {} frames",
        overall,
        decoded.len()
    );
    assert!(
        overall >= 99.0,
        "P 8×8 luma bit-match {:.2}% < 99%",
        overall
    );
}
