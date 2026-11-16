//! A sequence parameter set re-sent mid-sequence with another picture size
//! takes effect only at the next IDR picture (§7.4.1.2.1: the SPS active
//! for a coded video sequence changes only at an IDR access unit), so a
//! non-IDR picture naming it is refused rather than decoded at the new size
//! against references of the old one.
//!
//! `fixtures/sps_size_change_mid_sequence.h264` splices two libx264 encodes
//! of FFmpeg's `testsrc` (128x96 and 160x96, six pictures each, an IDR every
//! three, SPS id 0 in both): the 128x96 stream's SPS, PPS and first IDR, then
//! the 160x96 stream's SPS and PPS and its first P picture (frame_num 1, so
//! the frame_num rules pass it), then the 160x96 stream from its first IDR.

use oxideav_core::{CodecId, Decoder, ExecutionContext, Frame, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const STREAM: &[u8] = include_bytes!("fixtures/sps_size_change_mid_sequence.h264");

/// The luma bytes of each decoded picture, and the slices the decoder
/// skipped.
fn decode(threads: usize) -> (Vec<usize>, u64) {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    dec.set_execution_context(&ExecutionContext { threads });
    let pkt = Packet::new(0, TimeBase::new(1, 25), STREAM.to_vec()).with_pts(0);
    let _ = dec.send_packet(&pkt);
    let _ = dec.flush();
    let mut sizes = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => sizes.push(vf.planes[0].data.len()),
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    (sizes, dec.decode_error_count())
}

#[test]
fn a_new_picture_size_waits_for_the_next_idr_picture() {
    let (small, large) = (128 * 96, 160 * 96);
    for threads in [1, 4] {
        let (sizes, skipped) = decode(threads);
        let mut want = vec![small];
        want.extend([large; 6]);
        assert_eq!(sizes, want, "{threads} thread(s): pictures by luma bytes");
        assert!(
            skipped >= 1,
            "{threads} thread(s): the P picture before the IDR is refused"
        );
    }
}
