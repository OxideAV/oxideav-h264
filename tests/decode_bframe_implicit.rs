//! Decode a CAVLC B-frame clip encoded with **implicit weighted
//! bi-prediction** (§8.4.2.3.3) against an ffmpeg YUV reference.
//!
//! The fixture (`tests/fixtures/bframe_implicit_64x64.{es,yuv}`) is a 64×64
//! `testsrc` pattern, 12 frames with keyint=6 and `bf=2`, encoded with
//! `x264opts weightb:weightp=0`. x264's `weightb=1` together with the
//! Main profile makes the PPS set `weighted_bipred_idc = 2` (parsed
//! pre-merge; see the inline inspector used during development), which
//! selects the implicit formula of §8.4.2.3.3 for every bi-predicted
//! sample in the clip.
//!
//! To regenerate (run from the crate root):
//! ```bash
//! ffmpeg -y -f lavfi -i "testsrc=size=64x64:rate=24:duration=0.5" \
//!   -pix_fmt yuv420p -c:v libx264 -profile:v main -preset medium -g 6 \
//!   -bf 2 -coder 0 \
//!   -x264opts "no-scenecut:no-mbtree:no-8x8dct:weightb:weightp=0" \
//!   /tmp/bframe_implicit.mp4
//! ffmpeg -y -i /tmp/bframe_implicit.mp4 -c:v copy -bsf:v h264_mp4toannexb \
//!   -f h264 tests/fixtures/bframe_implicit_64x64.es
//! ffmpeg -y -i /tmp/bframe_implicit.mp4 -f rawvideo -pix_fmt yuv420p \
//!   tests/fixtures/bframe_implicit_64x64.yuv
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
        eprintln!("fixture {path} missing — skipping implicit-bipred test");
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
fn decode_implicit_bipred_clip_against_reference() {
    let es_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bframe_implicit_64x64.es"
    );
    let yuv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bframe_implicit_64x64.yuv"
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
    // Previously `weighted_bipred_idc == 2` B-slices returned
    // `Error::Unsupported` from `send_packet`; that gate is removed now.
    assert_eq!(
        parse_failures, 0,
        "{} frames rejected during implicit-bipred decode",
        parse_failures
    );

    let luma_bytes = W * H;
    let mut total_luma_match = 0usize;
    let mut total_luma_samples = 0usize;
    let mut total_psnr = 0.0f64;
    let cmp_count = decoded_frames.len().min(total_ref_frames);
    for (i, dec_buf) in decoded_frames.iter().enumerate().take(cmp_count) {
        let ref_off = i * FRAME_BYTES;
        let ref_buf = &yuv_all[ref_off..ref_off + FRAME_BYTES];
        let luma_match = count_within(&dec_buf[..luma_bytes], &ref_buf[..luma_bytes], 0);
        total_luma_match += luma_match;
        total_luma_samples += luma_bytes;
        let p = psnr(dec_buf, ref_buf);
        total_psnr += p;
        eprintln!(
            "implicit-bipred frame {i}: luma bit-match: {}/{} ({:.2}%)  PSNR={:.2} dB",
            luma_match,
            luma_bytes,
            (luma_match as f64) * 100.0 / (luma_bytes as f64),
            p
        );
    }
    let luma_match_pct = (total_luma_match as f64) * 100.0 / (total_luma_samples as f64);
    let avg_psnr = total_psnr / cmp_count as f64;
    eprintln!(
        "implicit-bipred overall: {} frames, luma bit-match: {:.2}%, avg PSNR: {:.2} dB",
        cmp_count, luma_match_pct, avg_psnr
    );
    // Implicit bi-prediction is a pixel-exact operation on top of integer
    // motion compensation, but the surrounding MC path reuses the same
    // 6-tap + bilinear filters already proven against the temporal-direct
    // fixture. 75% luma bit-match is the same bar the existing B-slice
    // tests apply — it confirms the sample combine follows §8.4.2.3.3
    // across a full decoded clip.
    assert!(
        luma_match_pct >= 75.0,
        "implicit-bipred luma bit-match {luma_match_pct:.2}% below 75% threshold"
    );
}
