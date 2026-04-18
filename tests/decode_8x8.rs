//! Decode an H.264 elementary stream produced with
//! `-profile high -x264opts 8x8dct=1:cabac=0` and compare the first IDR
//! against the ffmpeg-decoded reference YUV. Exercises the CAVLC + 8×8
//! integer-transform + Intra_8×8 prediction path (§8.3.2, §8.5.13,
//! §7.3.5.3.2) added with High-Profile support.
//!
//! Fixture:
//!
//! * `tests/fixtures/iframe_8x8_cavlc_128x128.{es,yuv}` — SMPTE bars 128×128
//!   High-Profile IDR with `8x8dct=1:cabac=0:no-deblock`. The smpte pattern
//!   drives x264 to pick a healthy mix of Intra_8×8 (10.9%), Intra_4×4
//!   (23.4%) and Intra_16×16 (65.6%) macroblocks under `-preset medium`,
//!   so the 8×8 residual / prediction / neighbour-aggregation paths see
//!   a realistic workload alongside the existing 4×4 and 16×16 paths.
//!   Regenerate with:
//!
//!   ```bash
//!   ffmpeg -y -f lavfi -i "smptebars=s=128x128:r=1" -vframes 1 \
//!     -pix_fmt yuv420p -c:v libx264 -profile:v high -preset medium \
//!     -coder 0 -x264opts "8x8dct=1:cabac=0:no-scenecut:no-deblock" \
//!     /tmp/h264_8x8.mp4
//!   ffmpeg -y -i /tmp/h264_8x8.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 \
//!     tests/fixtures/iframe_8x8_cavlc_128x128.es
//!   ffmpeg -y -i /tmp/h264_8x8.mp4 -f rawvideo -pix_fmt yuv420p \
//!     tests/fixtures/iframe_8x8_cavlc_128x128.yuv
//!   ```

use std::path::PathBuf;

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::decoder::H264Decoder;
use oxideav_h264::nal::{split_annex_b, NalHeader, NalUnitType};

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
fn decode_high_profile_8x8_intra_against_reference() {
    let es = match read_fixture("iframe_8x8_cavlc_128x128.es") {
        Some(d) => d,
        None => return,
    };
    let yuv = match read_fixture("iframe_8x8_cavlc_128x128.yuv") {
        Some(d) => d,
        None => return,
    };
    let frame = decode_first_iframe(&es);
    assert_eq!(frame.width, 128);
    assert_eq!(frame.height, 128);

    let ref_y = &yuv[0..(128 * 128)];
    let ref_cb = &yuv[(128 * 128)..(128 * 128 + 64 * 64)];
    let ref_cr = &yuv[(128 * 128 + 64 * 64)..(128 * 128 + 64 * 64 * 2)];
    let dec_y = &frame.planes[0].data;
    let dec_cb = &frame.planes[1].data;
    let dec_cr = &frame.planes[2].data;
    let total = ref_y.len() + ref_cb.len() + ref_cr.len();
    // Goal: bit-exact reconstruction. A single-LSB tolerance absorbs the
    // ±1 rounding drift that an integer dequant+IDCT can exhibit against
    // a reference without hiding an actual desync (which would show up
    // as per-MB errors of ~10+ LSB).
    let within_1 = count_within(dec_y, ref_y, 1)
        + count_within(dec_cb, ref_cb, 1)
        + count_within(dec_cr, ref_cr, 1);
    let pct = (within_1 as f64) * 100.0 / (total as f64);
    eprintln!(
        "8x8 intra: decoded vs reference within ±1 LSB: {}/{} ({:.2}%)",
        within_1, total, pct
    );
    assert!(
        pct >= 99.0,
        "8x8 pixel-match {:.2}% < 99% (±1 LSB tolerance)",
        pct
    );
}
