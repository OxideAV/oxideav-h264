//! Decode a short Main-profile (CABAC) clip with P-slices and compare the
//! output against the ffmpeg YUV reference.
//!
//! Fixture regeneration (run outside the crate):
//!
//! ```sh
//! ffmpeg -y -f lavfi -i testsrc=duration=0.5:size=64x64:rate=12 \
//!   -vframes 3 -pix_fmt yuv420p -c:v libx264 -profile:v main \
//!   -preset ultrafast -coder 1 -weightp 0 \
//!   -x264opts no-scenecut:no-mbtree:bframes=0:keyint=12:min-keyint=12 \
//!   /tmp/cabac_p.mp4
//! ffmpeg -y -i /tmp/cabac_p.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 \
//!   tests/fixtures/cabac_p_64x64.es
//! ffmpeg -y -i /tmp/cabac_p.mp4 -f rawvideo -pix_fmt yuv420p \
//!   tests/fixtures/cabac_p_64x64.yuv
//! ```
//!
//! The fixture has 3 frames: IDR + 2 P-slices, all Main-profile CABAC with
//! no weighted P, no B-frames, 8×4 transform disabled (libx264 ultrafast).

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::decoder::H264Decoder;
use oxideav_h264::nal::{split_annex_b, NalHeader, NalUnitType};

const W: usize = 64;
const H: usize = 64;
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
fn decode_cabac_p_slices_against_reference() {
    let es = include_bytes!("fixtures/cabac_p_64x64.es");
    let yuv_all = include_bytes!("fixtures/cabac_p_64x64.yuv");
    assert!(
        yuv_all.len() >= 3 * FRAME_BYTES,
        "reference YUV too small ({} bytes, expected >= {})",
        yuv_all.len(),
        3 * FRAME_BYTES
    );

    let (sps, pps, frame_nalus) = split_frames(es);
    assert_eq!(
        frame_nalus.len(),
        3,
        "expected 3 slice NALUs (I + 2 P), got {}",
        frame_nalus.len()
    );

    let mut dec = H264Decoder::new(CodecId::new("h264"));

    // Seed SPS + PPS via a primer packet.
    let mut primer = Vec::new();
    primer.extend_from_slice(&[0, 0, 0, 1]);
    primer.extend_from_slice(&sps);
    primer.extend_from_slice(&[0, 0, 0, 1]);
    primer.extend_from_slice(&pps);
    let primer_pkt = Packet::new(0, TimeBase::new(1, 90_000), primer);
    dec.send_packet(&primer_pkt).expect("primer send_packet");
    while dec.receive_frame().is_ok() {}

    // Drive every frame through the decoder. All three frames (IDR + 2 P)
    // are now hard-asserted bit-exact (≥99% luma ±8 LSB match) against the
    // ffmpeg reference. Frame 0 locked in with the §9.3.3.1.1.9 CBF
    // neighbour fix (cabac-recon2 wt) and the per-8×8-bit CBP luma ctx fix
    // (wt 5b818d8). Frames 1..2 locked in once `decode_mb_type_p` was
    // rewired to the spec-correct bin tree (inter-first, `1` → intra) and
    // the intra-in-P suffix was moved from ctxIdxOffset 3 to the
    // spec-mandated 17 (see FFmpeg `decode_cabac_intra_mb_type(sl, 17, 0)`).
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
            "CABAC P fixture frame {i}: luma ±8 LSB match = {}/{} ({:.2}%), \
             total ±8 LSB = {}/{} ({:.2}%)",
            luma_match, LUMA_BYTES, luma_pct, total_match, FRAME_BYTES, total_pct
        );
        luma_pcts.push(luma_pct);
    }
    if !errors.is_empty() {
        for e in &errors {
            eprintln!("CABAC P decode error: {e}");
        }
    }
    eprintln!(
        "decoded {}/{} frames; errors={}",
        decoded_frames.len(),
        frame_nalus.len(),
        errors.len()
    );
    // All three frames (IDR + 2 P-slices) must decode bit-exact (≥99% luma
    // ±8 LSB) after the CABAC P mb_type / intra-in-P ctxIdxOffset fix.
    assert!(
        errors.is_empty(),
        "CABAC P decode errors: {errors:?}",
    );
    assert_eq!(
        decoded_frames.len(),
        3,
        "expected 3 decoded frames (IDR + 2 P), got {}",
        decoded_frames.len()
    );
    for (i, pct) in luma_pcts.iter().enumerate() {
        assert!(
            *pct >= 99.0,
            "CABAC P fixture frame {i} luma match {pct:.2}% < 99%",
        );
    }
}
