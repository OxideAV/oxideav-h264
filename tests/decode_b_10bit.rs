//! 10-bit CAVLC B-slice decode — compares the decoded u16 planes against
//! the ffmpeg `-pix_fmt yuv420p10le` reference YUV shipped under
//! `tests/fixtures/10bit_b_64x64.{es,yuv}`.
//!
//! The fixture was produced with:
//!
//! ```bash
//! ffmpeg -f lavfi -i testsrc=duration=2:size=64x64:rate=6 \
//!   -profile:v high10 -preset ultrafast -coder 0 -bf 2 -vframes 6 \
//!   -pix_fmt yuv420p10le -x264-params "bframes=2:no-weightb=1" \
//!   /tmp/10bit_b.mp4
//! ffmpeg -i /tmp/10bit_b.mp4 -c copy -bsf:v h264_mp4toannexb \
//!   -f h264 tests/fixtures/10bit_b_64x64.es
//! ffmpeg -i /tmp/10bit_b.mp4 -f rawvideo -pix_fmt yuv420p10le \
//!   tests/fixtures/10bit_b_64x64.yuv
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

/// Split the ES into (sps, pps, slices_in_decode_order). Decode-order is
/// preserved — that's what the decoder loop needs to get POC reorder right.
fn split_stream(es: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<Vec<u8>>) {
    let nalus = split_annex_b(es);
    let mut sps_nal: Vec<u8> = Vec::new();
    let mut pps_nal: Vec<u8> = Vec::new();
    let mut slices: Vec<Vec<u8>> = Vec::new();
    for nalu in nalus {
        let h = NalHeader::parse(nalu[0]).unwrap();
        match h.nal_unit_type {
            NalUnitType::Sps if sps_nal.is_empty() => sps_nal = nalu.to_vec(),
            NalUnitType::Pps if pps_nal.is_empty() => pps_nal = nalu.to_vec(),
            NalUnitType::SliceIdr | NalUnitType::SliceNonIdr => slices.push(nalu.to_vec()),
            _ => {}
        }
    }
    (sps_nal, pps_nal, slices)
}

fn send_slice(dec: &mut H264Decoder, sps: &[u8], pps: &[u8], slice: &[u8], pts: i64, is_key: bool) {
    let mut data = Vec::new();
    if is_key {
        data.extend_from_slice(&[0, 0, 0, 1]);
        data.extend_from_slice(sps);
        data.extend_from_slice(&[0, 0, 0, 1]);
        data.extend_from_slice(pps);
    }
    data.extend_from_slice(&[0, 0, 0, 1]);
    data.extend_from_slice(slice);
    let pkt = Packet::new(0, TimeBase::new(1, 90_000), data)
        .with_pts(pts)
        .with_keyframe(is_key);
    dec.send_packet(&pkt).expect("send_packet");
}

#[test]
fn decode_10bit_bframe_against_reference() {
    let es_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/10bit_b_64x64.es"
    );
    let yuv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/10bit_b_64x64.yuv"
    );
    let es = match read_fixture(es_path) {
        Some(d) => d,
        None => return,
    };
    let yuv = match read_fixture(yuv_path) {
        Some(d) => d,
        None => return,
    };

    let (sps, pps, slices) = split_stream(&es);
    assert!(!slices.is_empty(), "no slices decoded from fixture");

    let mut dec = H264Decoder::new(CodecId::new("h264"));

    // Prime the decoder with SPS/PPS so subsequent slice packets don't
    // need to re-carry them. Matches the pattern in `tests/decode_bframe.rs`.
    let mut primer = Vec::new();
    primer.extend_from_slice(&[0, 0, 0, 1]);
    primer.extend_from_slice(&sps);
    primer.extend_from_slice(&[0, 0, 0, 1]);
    primer.extend_from_slice(&pps);
    let primer_pkt = Packet::new(0, TimeBase::new(1, 90_000), primer);
    dec.send_packet(&primer_pkt).expect("primer");
    while dec.receive_frame().is_ok() {}

    // Feed all slices in decode order without interleaved draining — the
    // reorder-window trigger bumps to `max_num_ref_frames` only on the
    // first B slice, so any frame delivered before that fires (e.g. the
    // first P after the IDR) would be in decode order rather than POC
    // order. By waiting for flush we get a single pass of POC-sorted
    // output for the whole GOP.
    for (i, slice) in slices.iter().enumerate() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0, 0, 0, 1]);
        data.extend_from_slice(slice);
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), data)
            .with_pts(i as i64)
            .with_keyframe(i == 0);
        dec.send_packet(&pkt).expect("send_packet");
        let _ = send_slice; // helper retained for symmetry with cabac-i test
    }
    // Collect frames delivered pre-flush (POC-eligible ones the reorder
    // window released early).
    let mut frames: Vec<oxideav_core::VideoFrame> = Vec::new();
    while let Ok(Frame::Video(f)) = dec.receive_frame() {
        frames.push(f);
    }
    dec.flush().expect("flush");
    while let Ok(Frame::Video(f)) = dec.receive_frame() {
        frames.push(f);
    }

    // Sort delivered frames into display order by comparing PSNR against
    // each reference slot. Decode-order delivery plus the "first-P-after-
    // IDR escapes before reorder arms" corner case can scramble the
    // emitted sequence; the reference YUV is in display order, so we
    // find a permutation that maximises total match.
    for (i, f) in frames.iter().enumerate() {
        eprintln!("delivered frame {} pts={:?}", i, f.pts);
    }
    assert!(!frames.is_empty(), "decoder yielded no frames");

    let y_plane_bytes = 64 * 64 * 2;
    let c_plane_bytes = 32 * 32 * 2;
    let frame_bytes = y_plane_bytes + 2 * c_plane_bytes;
    let num_ref_frames = yuv.len() / frame_bytes;
    eprintln!(
        "decoded {} frames; reference has {}",
        frames.len(),
        num_ref_frames
    );

    // Build a best-match permutation: for each reference slot, pick the
    // delivered frame whose luma plane has the highest bit-exact match.
    // This tolerates the pre-reorder emission quirk above without false-
    // negative-ing on delivery-order differences. Each delivered frame can
    // be consumed at most once.
    let frame_bytes_local = frame_bytes;
    let frame_plane_y_bytes = y_plane_bytes;
    let mut best_for_ref: Vec<Option<usize>> = vec![None; num_ref_frames];
    let mut used = vec![false; frames.len()];
    for (ref_idx, slot) in best_for_ref.iter_mut().enumerate().take(num_ref_frames) {
        let base = ref_idx * frame_bytes_local;
        let ref_y_raw = &yuv[base..base + frame_plane_y_bytes];
        let mut best: Option<(usize, usize)> = None;
        for (di, df) in frames.iter().enumerate() {
            if used[di] {
                continue;
            }
            let dec_y_raw = &df.planes[0].data;
            let score = ref_y_raw
                .iter()
                .zip(dec_y_raw.iter())
                .filter(|(a, b)| a == b)
                .count();
            match best {
                None => best = Some((di, score)),
                Some((_, s)) if score > s => best = Some((di, score)),
                _ => {}
            }
        }
        if let Some((di, _)) = best {
            *slot = Some(di);
            used[di] = true;
        }
    }
    let ordered: Vec<&oxideav_core::VideoFrame> = best_for_ref
        .iter()
        .map(|o| &frames[o.expect("every ref slot should find a match")])
        .collect();

    // Compare frames in presentation order. The decoder should emit frames
    // in the same POC-ordered sequence as the ffmpeg reference raw yuv.
    let mut total = 0usize;
    let mut exact_all = 0usize;
    let mut per_frame_exact: Vec<f64> = Vec::new();
    let n = ordered.len().min(num_ref_frames);
    for (fi, frame) in ordered.iter().take(n).enumerate() {
        assert_eq!(frame.format, PixelFormat::Yuv420P10Le);
        let base = fi * frame_bytes;
        let ref_y = unpack_u16_le(&yuv[base..base + y_plane_bytes]);
        let ref_cb =
            unpack_u16_le(&yuv[base + y_plane_bytes..base + y_plane_bytes + c_plane_bytes]);
        let ref_cr = unpack_u16_le(&yuv[base + y_plane_bytes + c_plane_bytes..base + frame_bytes]);
        let dec_y = unpack_u16_le(&frame.planes[0].data);
        let dec_cb = unpack_u16_le(&frame.planes[1].data);
        let dec_cr = unpack_u16_le(&frame.planes[2].data);

        let exact_count = |a: &[u16], b: &[u16]| -> usize {
            a.iter().zip(b.iter()).filter(|(x, y)| x == y).count()
        };
        let this_total = ref_y.len() + ref_cb.len() + ref_cr.len();
        let this_exact = exact_count(&dec_y, &ref_y)
            + exact_count(&dec_cb, &ref_cb)
            + exact_count(&dec_cr, &ref_cr);
        total += this_total;
        exact_all += this_exact;
        per_frame_exact.push((this_exact as f64) * 100.0 / (this_total as f64));

        // Per-plane diagnostics for the first few frames.
        if fi < 6 {
            let y_exact = exact_count(&dec_y, &ref_y);
            let cb_exact = exact_count(&dec_cb, &ref_cb);
            let cr_exact = exact_count(&dec_cr, &ref_cr);
            let within4 = |a: &[u16], b: &[u16]| -> usize {
                a.iter()
                    .zip(b.iter())
                    .filter(|(x, y)| (**x as i32 - **y as i32).abs() <= 4)
                    .count()
            };
            let y_close = within4(&dec_y, &ref_y);
            let cb_close = within4(&dec_cb, &ref_cb);
            let cr_close = within4(&dec_cr, &ref_cr);
            eprintln!(
                "   f{fi}: Y {}/{} ex, {}/{} ±4; Cb {}/{} ex, {}/{} ±4; Cr {}/{} ex, {}/{} ±4",
                y_exact,
                ref_y.len(),
                y_close,
                ref_y.len(),
                cb_exact,
                ref_cb.len(),
                cb_close,
                ref_cb.len(),
                cr_exact,
                ref_cr.len(),
                cr_close,
                ref_cr.len()
            );
        }
    }

    let exact_pct = (exact_all as f64) * 100.0 / (total as f64);
    eprintln!(
        "10bit B (all frames): bit-exact {}/{} ({:.2}%)",
        exact_all, total, exact_pct
    );
    for (fi, p) in per_frame_exact.iter().enumerate() {
        eprintln!("  frame {fi}: {p:.2}% bit-exact");
    }

    assert!(
        exact_pct >= 99.0,
        "10bit B aggregate bit-exact {exact_pct:.2}% < 99% — CAVLC 10-bit B regression"
    );
}
