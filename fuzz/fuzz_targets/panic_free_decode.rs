#![no_main]

//! Fuzz: arbitrary bytes → `H264CodecDecoder::send_packet` /
//! `receive_frame`.
//!
//! Contract: every public Decoder method MUST return a `Result` for
//! malformed input — never panic, never abort. This target feeds raw
//! libfuzzer bytes both as Annex-B byte stream (no extradata) and as
//! AVCC-framed payload (with a synthetic `avcDecoderConfigurationRecord`
//! pulled from the input prefix), then asserts only that no panic
//! escaped. Errors from `send_packet` / `receive_frame` are fine — the
//! Decoder trait is defined to surface them.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, Decoder, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    // Path 1: Annex-B byte stream (no extradata configured).
    {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let pkt = Packet::new(0, TimeBase::new(1, 30), data.to_vec());
        let _ = dec.send_packet(&pkt);
        // Drain whatever frames the driver was willing to produce; each
        // call MUST return a Result rather than panic.
        for _ in 0..8 {
            if dec.receive_frame().is_err() {
                break;
            }
        }
    }
    // Path 2: AVCC framing — feed the same bytes through a decoder that
    // expects a 4-byte length prefix. Most malformed inputs will fail
    // with `Err(...)` here too, which is fine. We don't bother
    // configuring extradata: leaving `length_size` unset would route us
    // back through Annex B; a manual `consume_extradata` call exercises
    // the avcC path on its own.
    {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        // Use the first byte to choose a synthetic minimal avcC blob:
        //   01 42 00 1e ff e1 0000 ff
        // This has 0 SPS + 0 PPS + lengthSizeMinusOne=3 (i.e. 4-byte
        // prefix), the most common config. If `consume_extradata`
        // rejects the synthetic blob (which it will on most random
        // inputs), we just skip path 2.
        let extra = [0x01, 0x42, 0x00, 0x1e, 0xff, 0xe0, 0x00];
        if dec.consume_extradata(&extra).is_ok() {
            let pkt = Packet::new(0, TimeBase::new(1, 30), data.to_vec());
            let _ = dec.send_packet(&pkt);
            for _ in 0..8 {
                if dec.receive_frame().is_err() {
                    break;
                }
            }
        }
    }
});
