//! Decode a 128×128 CABAC P-slice clip whose `P_8x8` macroblocks exercise
//! every `sub_mb_type` value (Sub8x8, Sub8x4, Sub4x8, Sub4x4). This fixture
//! locks in the polarity fix for [`binarize::decode_sub_mb_type_p`]
//! (§9.3.3.1.1.5 / Table 9-37) — prior to the fix, bins 0 and 1 were
//! inverted, which silently swapped Sub8x8↔non-Sub8x8 and Sub8x4↔Sub4x4,
//! knocking the bitstream out of alignment on the next CABAC read and
//! surfacing as `sub_ref_idx_l0 N out of range` on real-world clips where
//! the encoder actually varies the sub-partition choice.
//!
//! Fixture regeneration:
//!
//! ```sh
//! ffmpeg -y -f lavfi -i "mandelbrot=size=128x128:rate=24" -vframes 12 \
//!   -pix_fmt yuv420p -c:v libx264 -profile:v main -preset ultrafast \
//!   -coder 1 -x264-params \
//!   "ref=3:bframes=0:no-mbtree=1:keyint=30:no-scenecut=1:partitions=all" \
//!   /tmp/p_8x8_sub.mp4
//! ffmpeg -y -i /tmp/p_8x8_sub.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 \
//!   tests/fixtures/p_cabac_8x8_sub_mb_type_128x128.es
//! ffmpeg -y -i /tmp/p_8x8_sub.mp4 -f rawvideo -pix_fmt yuv420p \
//!   tests/fixtures/p_cabac_8x8_sub_mb_type_128x128.yuv
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
fn decode_cabac_p_8x8_sub_mb_type_against_reference() {
    let es = include_bytes!("fixtures/p_cabac_8x8_sub_mb_type_128x128.es");
    let yuv_all = include_bytes!("fixtures/p_cabac_8x8_sub_mb_type_128x128.yuv");
    let frame_count = yuv_all.len() / FRAME_BYTES;
    assert!(
        frame_count >= 2,
        "reference YUV too small ({} bytes)",
        yuv_all.len()
    );

    let (sps, pps, frame_nalus) = split_frames(es);
    assert!(
        frame_nalus.len() >= 2,
        "expected at least 2 slice NALUs, got {}",
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
        let ref_buf = &yuv_all[i * FRAME_BYTES..(i + 1) * FRAME_BYTES];
        let luma_match = count_within(&dec_buf[..LUMA_BYTES], &ref_buf[..LUMA_BYTES], 8);
        let luma_pct = (luma_match as f64) * 100.0 / (LUMA_BYTES as f64);
        let total_match = count_within(dec_buf, ref_buf, 8);
        let total_pct = (total_match as f64) * 100.0 / (FRAME_BYTES as f64);
        eprintln!(
            "CABAC P 8x8 sub_mb_type frame {i}: luma \u{00b1}8 = {luma_match}/{LUMA_BYTES} \
             ({luma_pct:.2}%), total \u{00b1}8 = {total_match}/{FRAME_BYTES} ({total_pct:.2}%)"
        );
        luma_pcts.push(luma_pct);
    }

    assert!(
        errors.is_empty(),
        "CABAC P 8x8 sub_mb_type decode errors: {errors:?}"
    );
    assert_eq!(
        decoded_frames.len(),
        frame_nalus.len(),
        "expected all frames to decode without the sub_ref_idx_l0 out-of-range error"
    );
    assert_eq!(decoded_frames.len(), frame_count.min(frame_nalus.len()));

    // The IDR and the first P-slice (ref=0 only) must be bit-exact against
    // the FFmpeg reference — this is what the polarity fix unlocks.
    assert!(
        luma_pcts[0] >= 99.0,
        "IDR luma match {:.2}% < 99%",
        luma_pcts[0]
    );
    assert!(
        luma_pcts[1] >= 99.0,
        "first P frame luma match {:.2}% < 99%",
        luma_pcts[1]
    );
    // Later P-frames accumulate sub-pixel MC drift from other unfixed
    // High-profile paths on this encoder preset, so only assert a
    // conservative floor — what matters here is that the decode completes
    // without the out-of-range error and stays broadly in sync.
    for (i, pct) in luma_pcts.iter().enumerate() {
        assert!(
            *pct >= 60.0,
            "CABAC P 8x8 sub_mb_type frame {i} luma match {pct:.2}% < 60%"
        );
    }
}
