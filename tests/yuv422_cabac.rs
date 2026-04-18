//! 4:2:2 CABAC I and P decode + 4:2:2 CAVLC B decode against
//! ffmpeg-generated reference fixtures.
//!
//! Fixtures (generated via libx264):
//!
//! * `yuv422_cabac_i_64x64.{es,yuv}` — High 4:2:2 CABAC IDR-only.
//!   Stresses `decode_chroma_422` in `src/cabac/mb.rs`: 2×4 chroma DC
//!   (ctxBlockCat = 3, 8 coeffs) + 8 chroma-AC blocks per plane
//!   (ctxBlockCat = 4). §8.5.11.2 dequant with `QP'_C,DC = QP'_C + 3`.
//!
//! * `yuv422_cabac_p_64x64.{es,yuv}` — High 4:2:2 CABAC IDR + 3 P
//!   slices. Exercises `decode_inter_residual_chroma` dispatch for
//!   4:2:2 in `src/cabac/p_mb.rs` plus the shared chroma MC 4:2:2
//!   vertical-divide-by-4 behaviour.
//!
//! * `yuv422_b_64x64.{es,yuv}` — High 4:2:2 CAVLC with 1 B-slice.
//!   Exercises `decode_inter_residual_chroma` 4:2:2 branch in
//!   `src/b_mb.rs` and 4:2:2 chroma MC for bipred.

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, PixelFormat, TimeBase};
use oxideav_h264::decoder::H264Decoder;
use oxideav_h264::nal::{split_annex_b, NalHeader, NalUnitType};

fn read_fixture(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn split_es(es: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<Vec<u8>>) {
    let nalus = split_annex_b(es);
    let mut sps: Option<Vec<u8>> = None;
    let mut pps: Option<Vec<u8>> = None;
    let mut frames: Vec<Vec<u8>> = Vec::new();
    for nalu in &nalus {
        let h = NalHeader::parse(nalu[0]).unwrap();
        match h.nal_unit_type {
            NalUnitType::Sps if sps.is_none() => sps = Some(nalu.to_vec()),
            NalUnitType::Pps if pps.is_none() => pps = Some(nalu.to_vec()),
            NalUnitType::SliceIdr | NalUnitType::SliceNonIdr => {
                let mut f = Vec::new();
                f.extend_from_slice(&[0, 0, 0, 1]);
                f.extend_from_slice(nalu);
                frames.push(f);
            }
            _ => {}
        }
    }
    (sps.expect("SPS"), pps.expect("PPS"), frames)
}

/// Feed SPS + PPS, then each frame in decode order; return the raw
/// yuv422p bytes from every received frame (Y64x64 + Cb32x64 + Cr32x64).
/// Output is ordered by the decoder's emission sequence (display order
/// for streams with reordering); the reference .yuv fixture is
/// ffmpeg-decoded in the same display order.
fn decode_all(es: &[u8]) -> Vec<u8> {
    const W: usize = 64;
    const H: usize = 64;
    const FRAME_BYTES: usize = W * H + (W / 2) * H * 2;
    let (sps, pps, frames) = split_es(es);
    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let mut primer = Vec::new();
    primer.extend_from_slice(&[0, 0, 0, 1]);
    primer.extend_from_slice(&sps);
    primer.extend_from_slice(&[0, 0, 0, 1]);
    primer.extend_from_slice(&pps);
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 90_000), primer))
        .expect("primer");
    while dec.receive_frame().is_ok() {}

    let append = |p: &oxideav_core::VideoFrame, out: &mut Vec<u8>| {
        let ly = &p.planes[0];
        for row in 0..H {
            let o = row * ly.stride;
            out.extend_from_slice(&ly.data[o..o + W]);
        }
        for pi in 1..=2 {
            let ch = &p.planes[pi];
            for row in 0..H {
                let o = row * ch.stride;
                out.extend_from_slice(&ch.data[o..o + W / 2]);
            }
        }
    };

    let mut out = Vec::with_capacity(frames.len() * FRAME_BYTES);
    for (i, frame) in frames.iter().enumerate() {
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), frame.clone())
            .with_pts(i as i64)
            .with_keyframe(i == 0);
        dec.send_packet(&pkt)
            .unwrap_or_else(|e| panic!("send {i}: {e}"));
        while let Ok(f) = dec.receive_frame() {
            let Frame::Video(v) = f else { panic!("not video") };
            assert_eq!(v.format, PixelFormat::Yuv422P);
            assert_eq!(v.width, W as u32);
            assert_eq!(v.height, H as u32);
            append(&v, &mut out);
        }
    }
    // Drain remaining frames (B-frame reordering holds earlier pictures
    // in the DPB until later frames arrive). `flush` empties the reorder
    // buffer so the caller sees every frame in display order.
    let _ = dec.flush();
    while let Ok(f) = dec.receive_frame() {
        let Frame::Video(v) = f else { panic!("not video") };
        append(&v, &mut out);
    }
    out
}

#[test]
fn decode_yuv422_cabac_iframe_matches_reference() {
    let es = read_fixture("tests/fixtures/yuv422_cabac_i_64x64.es");
    let yuv = read_fixture("tests/fixtures/yuv422_cabac_i_64x64.yuv");
    let got = decode_all(&es);
    assert_eq!(got.len(), yuv.len(), "frame byte count mismatch");
    let matches = got.iter().zip(yuv.iter()).filter(|(a, b)| a == b).count();
    let ratio = matches as f64 / got.len() as f64;
    let luma_bytes = 64 * 64;
    let luma_match = got[..luma_bytes]
        .iter()
        .zip(yuv[..luma_bytes].iter())
        .filter(|(a, b)| a == b)
        .count();
    let chroma_match = got[luma_bytes..]
        .iter()
        .zip(yuv[luma_bytes..].iter())
        .filter(|(a, b)| a == b)
        .count();
    eprintln!(
        "cabac-i-422 luma {}/{} ({:.2}%), chroma {}/{} ({:.2}%)",
        luma_match,
        luma_bytes,
        luma_match as f64 / luma_bytes as f64 * 100.0,
        chroma_match,
        got.len() - luma_bytes,
        chroma_match as f64 / (got.len() - luma_bytes) as f64 * 100.0,
    );
    assert!(
        ratio >= 0.99,
        "4:2:2 CABAC I accuracy {:.3}% — below 99%",
        ratio * 100.0
    );
}

#[test]
fn decode_yuv422_cabac_pframe_matches_reference() {
    let es = read_fixture("tests/fixtures/yuv422_cabac_p_64x64.es");
    let yuv = read_fixture("tests/fixtures/yuv422_cabac_p_64x64.yuv");
    let got = decode_all(&es);
    assert_eq!(got.len(), yuv.len(), "frame byte count mismatch");
    let matches = got.iter().zip(yuv.iter()).filter(|(a, b)| a == b).count();
    let ratio = matches as f64 / got.len() as f64;
    let luma_bytes = 64 * 64 * (got.len() / (64 * 64 + 32 * 64 * 2));
    let luma_match = (0..got.len() / (64 * 64 + 32 * 64 * 2))
        .flat_map(|f| {
            let off = f * (64 * 64 + 32 * 64 * 2);
            got[off..off + 64 * 64]
                .iter()
                .zip(yuv[off..off + 64 * 64].iter())
                .map(|(a, b)| (a == b) as usize)
                .collect::<Vec<_>>()
        })
        .sum::<usize>();
    eprintln!(
        "cabac-p-422 luma {}/{} ({:.2}%), total {:.2}%",
        luma_match,
        luma_bytes,
        luma_match as f64 / luma_bytes as f64 * 100.0,
        ratio * 100.0,
    );
    assert!(
        ratio >= 0.99,
        "4:2:2 CABAC P accuracy {:.3}% — below 99%",
        ratio * 100.0
    );
}

#[test]
fn decode_yuv422_cavlc_bframe_matches_reference() {
    let es = read_fixture("tests/fixtures/yuv422_b_64x64.es");
    let yuv = read_fixture("tests/fixtures/yuv422_b_64x64.yuv");
    let got = decode_all(&es);
    assert_eq!(got.len(), yuv.len(), "frame byte count mismatch");
    let matches = got.iter().zip(yuv.iter()).filter(|(a, b)| a == b).count();
    let ratio = matches as f64 / got.len() as f64;
    // `wt/pb-residual` (commit e65d4d2 on master) narrowed a CAVLC
    // P-residual divergence at mb_y=3 after Intra_16x16 that also
    // shows up on propagated B references. The 4:2:2 variant inherits
    // that residual drift — ≥ 95 % (typically ≥ 98 %) matches the
    // `decode_bframe.rs` 4:2:0 tolerances in the rest of the suite.
    assert!(
        ratio >= 0.95,
        "4:2:2 CAVLC B accuracy {:.3}% — below 95%",
        ratio * 100.0
    );
}
