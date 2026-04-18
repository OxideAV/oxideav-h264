//! Decode a CAVLC B-frame clip encoded with **temporal direct** MV
//! prediction (§8.4.1.2.3) against an ffmpeg YUV reference.
//!
//! The fixture (`tests/fixtures/bframe_temporal_64x64.{es,yuv}`) is a 64×64
//! `smptebars` pattern, 12 frames with keyint=6 and `bf=2`, encoded with
//! `-direct-pred temporal` so every `B_Direct` / `B_Skip` MB exercises the
//! colocated-block MV path (confirmed via the `direct=2` entry in the x264
//! options SEI). The slice headers carry `direct_spatial_mv_pred_flag = 0`,
//! which previously triggered `Error::Unsupported` in this decoder.
//!
//! To regenerate (run from the crate root):
//! ```bash
//! ffmpeg -y -f lavfi -i "smptebars=size=64x64:rate=24:duration=0.5" \
//!   -pix_fmt yuv420p -c:v libx264 -profile:v main -preset medium -g 6 \
//!   -bf 2 -direct-pred temporal -coder 0 \
//!   -x264opts no-scenecut:no-weightb:weightp=0:no-mbtree:no-8x8dct \
//!   /tmp/bframe_temporal.mp4
//! ffmpeg -y -i /tmp/bframe_temporal.mp4 -c:v copy -bsf:v h264_mp4toannexb \
//!   -f h264 tests/fixtures/bframe_temporal_64x64.es
//! ffmpeg -y -i /tmp/bframe_temporal.mp4 -f rawvideo -pix_fmt yuv420p \
//!   tests/fixtures/bframe_temporal_64x64.yuv
//! ```

use std::path::Path;

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::decoder::H264Decoder;
use oxideav_h264::nal::{split_annex_b, NalHeader, NalUnitType};

const W: usize = 64;
const H: usize = 64;
const FRAME_BYTES: usize = W * H * 3 / 2;

fn read_fixture(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        eprintln!("fixture {path} missing — skipping temporal-direct test");
        return None;
    }
    Some(std::fs::read(path).expect("read fixture"))
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

fn count_within(a: &[u8], b: &[u8], tol: i32) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() <= tol)
        .count()
}

/// Peak-signal-to-noise-ratio between two YUV buffers of the same length.
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sq_err: u64 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x as i32 - y as i32;
        sq_err += (d * d) as u64;
    }
    if sq_err == 0 {
        return 99.0;
    }
    let mse = (sq_err as f64) / (a.len() as f64);
    10.0 * (255.0 * 255.0 / mse).log10()
}

#[test]
fn decode_temporal_direct_clip_against_reference() {
    // Fixture lives in-tree so the test works without ffmpeg at run-time.
    let es_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bframe_temporal_64x64.es"
    );
    let yuv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bframe_temporal_64x64.yuv"
    );
    let es = match read_fixture(es_path) {
        Some(d) => d,
        None => return,
    };
    let yuv_all = match read_fixture(yuv_path) {
        Some(d) => d,
        None => return,
    };
    assert!(yuv_all.len() >= FRAME_BYTES, "reference YUV too small");
    let total_ref_frames = yuv_all.len() / FRAME_BYTES;
    assert!(
        total_ref_frames >= 3,
        "need ≥ 3 frames, got {total_ref_frames}"
    );

    let (sps, pps, frame_nalus) = split_frames(&es);
    assert!(
        frame_nalus.len() >= 3,
        "need ≥ 3 coded slices, got {}",
        frame_nalus.len()
    );

    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let mut primer = Vec::new();
    primer.extend_from_slice(&[0, 0, 0, 1]);
    primer.extend_from_slice(&sps);
    primer.extend_from_slice(&[0, 0, 0, 1]);
    primer.extend_from_slice(&pps);
    let primer_pkt = Packet::new(0, TimeBase::new(1, 90_000), primer);
    dec.send_packet(&primer_pkt).expect("primer");
    while dec.receive_frame().is_ok() {}

    let mut decoded_frames: Vec<Vec<u8>> = Vec::new();
    let mut parse_failures = 0;
    for (idx, frame) in frame_nalus.iter().enumerate() {
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), frame.clone())
            .with_pts(idx as i64)
            .with_keyframe(idx == 0);
        if let Err(e) = dec.send_packet(&pkt) {
            eprintln!("frame {idx}: send_packet returned {e:?}");
            parse_failures += 1;
            continue;
        }
        while let Ok(Frame::Video(f)) = dec.receive_frame() {
            let mut buf = Vec::with_capacity(FRAME_BYTES);
            buf.extend_from_slice(&f.planes[0].data);
            buf.extend_from_slice(&f.planes[1].data);
            buf.extend_from_slice(&f.planes[2].data);
            decoded_frames.push(buf);
        }
    }
    dec.flush().expect("flush");
    while let Ok(Frame::Video(f)) = dec.receive_frame() {
        let mut buf = Vec::with_capacity(FRAME_BYTES);
        buf.extend_from_slice(&f.planes[0].data);
        buf.extend_from_slice(&f.planes[1].data);
        buf.extend_from_slice(&f.planes[2].data);
        decoded_frames.push(buf);
    }

    assert!(!decoded_frames.is_empty(), "decoder produced zero frames");
    assert_eq!(
        parse_failures, 0,
        "{} frames rejected during temporal-direct decode",
        parse_failures
    );

    // Compare luma-only bit-match against ffmpeg's reference YUV. The luma
    // plane occupies the first W*H bytes of each YUV frame.
    let luma_bytes = W * H;
    let mut total_luma_match = 0usize;
    let mut total_luma_samples = 0usize;
    let mut total_psnr = 0.0f64;
    let cmp_count = decoded_frames.len().min(total_ref_frames);
    for i in 0..cmp_count {
        let dec_buf = &decoded_frames[i];
        let ref_off = i * FRAME_BYTES;
        let ref_buf = &yuv_all[ref_off..ref_off + FRAME_BYTES];
        let luma_match = count_within(&dec_buf[..luma_bytes], &ref_buf[..luma_bytes], 0);
        total_luma_match += luma_match;
        total_luma_samples += luma_bytes;
        let p = psnr(dec_buf, ref_buf);
        total_psnr += p;
        eprintln!(
            "temporal-direct frame {i}: luma bit-match: {}/{} ({:.2}%)  PSNR={:.2} dB",
            luma_match,
            luma_bytes,
            (luma_match as f64) * 100.0 / (luma_bytes as f64),
            p
        );
    }
    let luma_match_pct = (total_luma_match as f64) * 100.0 / (total_luma_samples as f64);
    let avg_psnr = total_psnr / cmp_count as f64;
    eprintln!(
        "temporal-direct overall: {} frames, luma bit-match: {:.2}%, avg PSNR: {:.2} dB",
        cmp_count, luma_match_pct, avg_psnr
    );
    // Task requirement: ≥ 75% luma bit-match against ffmpeg reference YUV.
    assert!(
        luma_match_pct >= 75.0,
        "temporal-direct luma bit-match {luma_match_pct:.2}% below 75% threshold"
    );
}
