//! Smoke test: feed a real h264 mp4 through the decoder via the
//! container + codec pipeline and count the frames produced. This
//! exercises the full path including multi-slice assembly — even if
//! individual slice CABAC/CAVLC fails, the decoder must not crash or
//! produce significantly more frames than the stream has coded
//! pictures.
//!
//! This is a smoke test only: we don't assert an exact frame count
//! because the CABAC parser in this crate still has known limitations
//! on some slices. We just assert:
//!   1. the decoder doesn't panic
//!   2. the number of frames emitted is reasonable relative to the
//!      packet count (fewer than the packet count when CABAC fails on
//!      some slices, AND never more than the stream has primary coded
//!      pictures even with multiple slices per picture).

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264::nal::AnnexBSplitter;
use std::path::PathBuf;

/// Feed raw Annex B bytes through H264CodecDecoder and return the
/// number of frames emitted. Never panics on decode errors — just
/// counts successful `receive_frame()` returns.
fn count_frames_from_annex_b(bytes: &[u8]) -> usize {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 1000), bytes.to_vec()).with_pts(0);
    let _ = dec.send_packet(&pkt);
    let _ = dec.flush();
    let mut count = 0;
    while let Ok(Frame::Video(_)) = dec.receive_frame() {
        count += 1;
    }
    count
}

/// §7.4.1.2.4 smoke: walking a whole annex-b stream should never emit
/// more frames than the number of primary coded pictures in it. We
/// count primary coded pictures using the same conditions as
/// `H264CodecDecoder::is_first_vcl_of_new_picture` so our expected
/// count is comparable.
fn count_expected_primary_pictures(bytes: &[u8]) -> usize {
    use oxideav_h264::decoder::{Decoder, Event};
    let mut dec = Decoder::new();
    let mut primaries = 0usize;
    // Previous slice's identity: (frame_num, pps_id, poc_lsb,
    // delta_poc_bottom, delta_poc_0, delta_poc_1, idr_flag, idr_pic_id,
    // field_pic_flag, ref_idc_is_nonzero).
    type Key = (u32, u32, u32, i32, i32, i32, bool, u32, bool, bool);
    let mut prev: Option<Key> = None;
    for ev in dec.process_annex_b(bytes) {
        if let Ok(Event::Slice {
            nal_unit_type,
            nal_ref_idc,
            header,
            ..
        }) = ev
        {
            let idr = nal_unit_type == 5;
            let key = (
                header.frame_num,
                header.pic_parameter_set_id,
                header.pic_order_cnt_lsb,
                header.delta_pic_order_cnt_bottom,
                header.delta_pic_order_cnt[0],
                header.delta_pic_order_cnt[1],
                idr,
                if idr { header.idr_pic_id } else { 0 },
                header.field_pic_flag,
                nal_ref_idc != 0,
            );
            if prev.as_ref() != Some(&key) {
                primaries += 1;
                prev = Some(key);
            }
        }
    }
    primaries
}

/// Run the smoke test on the nature_704x576 multi-slice clip. This
/// stream has 1780 slices grouped into roughly 1780/3 = ~593 pictures
/// (observed via inspect).
#[test]
fn nature_704x576_multi_slice_doesnt_explode() {
    let path = PathBuf::from(
        "/home/magicaltux/projects/oxideav/samples/samples.ffmpeg.org/V-codecs/h264/nature_704x576_25Hz_1500kbits.h264",
    );
    if !path.exists() {
        eprintln!("skip: nature sample missing");
        return;
    }
    let bytes = std::fs::read(&path).expect("read nature");

    // Cheap first-pass: total NAL count.
    let nal_count = AnnexBSplitter::new(&bytes).count();
    eprintln!("nature: {} total NALs", nal_count);

    let expected = count_expected_primary_pictures(&bytes);
    let frames = count_frames_from_annex_b(&bytes);
    eprintln!(
        "nature: expected ~{} primary coded pictures, got {} frames",
        expected, frames
    );

    // Sanity: frames must not exceed primary picture count +1 (flush
    // tail). If our assembly logic were broken and emitted one frame
    // per slice we'd see ~1780 frames, far more than expected.
    assert!(
        frames <= expected + 1,
        "got {} frames but only {} primary coded pictures expected — multi-slice assembly is broken",
        frames,
        expected
    );
}
