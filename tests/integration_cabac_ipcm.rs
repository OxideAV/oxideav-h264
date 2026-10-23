//! §7.3.5 / §9.3.1.2 — a CABAC slice holding several I_PCM macroblocks
//! after its first one. Each I_PCM payload is located from the
//! arithmetic decoder's position, which must stay relative to the CABAC
//! segment after the engine is re-initialised behind an earlier I_PCM.
//!
//! The fixture is one x264 lossless (`--qp 0`, CABAC) IDR picture, 96x64:
//! the right half is noise, which x264 codes as I_PCM macroblocks in every
//! row (macroblocks 3..5 of rows 0..2), with I_NxN / I_16x16 macroblocks
//! between them. The `.yuv` is the picture as a black-box reference decoder
//! outputs it (planar 4:2:0), which for a lossless stream is the source.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("read fixture")
}

#[test]
fn cabac_slice_with_i_pcm_after_the_first_macroblock_decodes_exactly() {
    let stream = fixture("x264_lossless_ipcm_cabac_96x64.h264");
    let reference = fixture("x264_lossless_ipcm_cabac_96x64.yuv");
    let mut dec = oxideav_h264::h264_decoder::H264CodecDecoder::new(CodecId::new("h264"));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), stream).with_pts(0))
        .expect("send_packet");
    dec.flush().expect("flush");
    let Ok(Frame::Video(frame)) = dec.receive_frame() else {
        panic!("no picture decoded");
    };
    assert_eq!(dec.decode_error_count(), 0, "a slice failed to decode");
    let mut picture = Vec::new();
    for plane in frame.image_planes() {
        picture.extend_from_slice(&plane.data);
    }
    assert_eq!(picture.len(), reference.len());
    let differing = picture
        .iter()
        .zip(&reference)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "{differing} samples differ from the reference"
    );
}
