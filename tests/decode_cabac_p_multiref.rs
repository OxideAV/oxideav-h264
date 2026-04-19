//! Decode a 128×128 CABAC clip with multi-reference P slices, exercising
//! the `ref_idx_l0` / `sub_ref_idx_l0` ctxIdxInc derivation inside inter
//! macroblocks whose partition-1 or sub-MB-1+ neighbour lies inside the
//! *same* MB.
//!
//! Spec coverage: §9.3.3.1.1.6 (ref_idx_lX ctxIdxInc) derives its bin-0
//! increment from the left/above 4×4 neighbour blocks' `ref_idx_lX`. When
//! the neighbour block is inside the same MB (P_16x8[1], P_8x16[1], or
//! sub-MBs 1/2/3 of a P_8x8), the decoder must read a freshly-decoded
//! value; reading the stale `MbInfo` default (`coded = false`) collapses
//! the ctxIdxInc to 0, which then mis-interprets a subsequent `ref_idx`
//! unary suffix and surfaces as the real-world `sub_ref_idx_l0 N out of
//! range` decode failure seen by `oxideplay`. This fixture + test lock the
//! progressive-publish fix into place.
//!
//! Fixture regeneration:
//!
//! ```sh
//! ffmpeg -y -f lavfi -i testsrc=duration=0.17:size=128x128:rate=24 \
//!   -vframes 2 -pix_fmt yuv420p -c:v libx264 -profile:v main \
//!   -preset ultrafast -coder 1 \
//!   -x264-params ref=2:bframes=0:no-mbtree=1:keyint=30:no-scenecut=1:partitions=all \
//!   /tmp/p_multiref.mp4
//! ffmpeg -y -i /tmp/p_multiref.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 \
//!   tests/fixtures/p_cabac_multiref_128x128.es
//! ffmpeg -y -i /tmp/p_multiref.mp4 -f rawvideo -pix_fmt yuv420p \
//!   tests/fixtures/p_cabac_multiref_128x128.yuv
//! ```

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::decoder::H264Decoder;
use oxideav_h264::nal::{split_annex_b, NalHeader, NalUnitType};

const W: usize = 128;
const H: usize = 128;
const FRAME_BYTES: usize = W * H * 3 / 2;

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

#[test]
fn decode_cabac_p_multiref_against_reference() {
    let es = include_bytes!("fixtures/p_cabac_multiref_128x128.es");
    let yuv_all = include_bytes!("fixtures/p_cabac_multiref_128x128.yuv");
    let frame_count = yuv_all.len() / FRAME_BYTES;
    assert!(
        frame_count >= 2,
        "reference YUV too small ({} bytes)",
        yuv_all.len()
    );

    let (sps, pps, frame_nalus) = split_frames(es);
    assert!(
        frame_nalus.len() >= 2,
        "expected at least 2 slice NALUs (IDR + P), got {}",
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

    let mut decoded_frames: Vec<Vec<u8>> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    for (idx, frame) in frame_nalus.iter().enumerate() {
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), frame.clone())
            .with_pts(idx as i64)
            .with_keyframe(idx == 0);
        if let Err(e) = dec.send_packet(&pkt) {
            errors.push(format!("frame {idx}: send_packet: {e:?}"));
            break;
        }
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => {
                let mut buf = Vec::with_capacity(FRAME_BYTES);
                buf.extend_from_slice(&vf.planes[0].data);
                buf.extend_from_slice(&vf.planes[1].data);
                buf.extend_from_slice(&vf.planes[2].data);
                decoded_frames.push(buf);
            }
            Ok(_) => unreachable!(),
            Err(e) => {
                errors.push(format!("frame {idx}: receive_frame: {e:?}"));
                break;
            }
        }
    }

    const LUMA_BYTES: usize = W * H;
    let mut luma_pcts = Vec::with_capacity(decoded_frames.len());
    for (i, dec_buf) in decoded_frames.iter().enumerate() {
        let ref_off = i * FRAME_BYTES;
        let ref_buf = &yuv_all[ref_off..ref_off + FRAME_BYTES];
        let luma_match = count_within(&dec_buf[..LUMA_BYTES], &ref_buf[..LUMA_BYTES], 8);
        let luma_pct = (luma_match as f64) * 100.0 / (LUMA_BYTES as f64);
        let total_match = count_within(dec_buf, ref_buf, 8);
        let total_pct = (total_match as f64) * 100.0 / (FRAME_BYTES as f64);
        eprintln!(
            "CABAC P multiref frame {i}: luma ±8 LSB = {}/{} ({:.2}%), \
             total ±8 LSB = {}/{} ({:.2}%)",
            luma_match, LUMA_BYTES, luma_pct, total_match, FRAME_BYTES, total_pct
        );
        luma_pcts.push(luma_pct);
    }
    if !errors.is_empty() {
        for e in &errors {
            eprintln!("CABAC P multiref decode error: {e}");
        }
    }
    eprintln!(
        "decoded {}/{} frames; errors={}",
        decoded_frames.len(),
        frame_nalus.len(),
        errors.len()
    );

    assert!(
        errors.is_empty(),
        "CABAC P multiref decode errors: {errors:?}"
    );
    assert_eq!(
        decoded_frames.len(),
        frame_nalus.len(),
        "expected all frames to decode"
    );
    for (i, pct) in luma_pcts.iter().enumerate() {
        assert!(
            *pct >= 99.0,
            "CABAC P multiref frame {i} luma match {pct:.2}% < 99%"
        );
    }
}
