//! Raw Annex-B byte-stream demuxer (`.h264`) end to end: probe →
//! demux → decode with the crate's own decoder.
//!
//! Fixture: `x264_testsrc_128x96_25f.h264` — 25 frames of a 128×96
//! 4:2:0 test pattern written by x264 (High profile, CABAC, B-frames,
//! weighted prediction), extracted from MP4 to the Annex-B byte
//! stream with an external muxer.

use oxideav_core::{Frame, RuntimeContext};

fn fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/x264_testsrc_128x96_25f.h264")
}

#[test]
fn raw_byte_stream_probes_demuxes_and_decodes_every_frame() {
    let mut ctx = RuntimeContext::new();
    oxideav_h264::register(&mut ctx);

    // No extension hint: the content alone identifies the stream.
    let mut f = std::fs::File::open(fixture()).unwrap();
    let name = ctx.containers.probe_input(&mut f, None).unwrap();
    assert_eq!(name, "h264");
    assert_eq!(ctx.containers.container_for_extension("264"), Some("h264"));

    let mut dm = ctx
        .containers
        .open_demuxer(&name, Box::new(f), &ctx.codecs)
        .unwrap();
    let params = dm.streams()[0].params.clone();
    assert_eq!(params.codec_id.as_str(), "h264");
    assert_eq!((params.width, params.height), (Some(128), Some(96)));
    assert_eq!(
        params.pixel_format,
        Some(oxideav_core::PixelFormat::Yuv420P)
    );
    assert!(params.extradata.is_empty(), "parameter sets stay in band");

    let mut dec = ctx.codecs.first_decoder(&params).unwrap();
    let (mut packets, mut keyframes, mut frames) = (0, 0, 0);
    let check = |frame: Frame| {
        let Frame::Video(v) = frame else {
            panic!("video frame expected")
        };
        assert_eq!(
            v.planes[0].stride * (v.planes[0].data.len() / v.planes[0].stride),
            128 * 96
        );
    };
    while let Ok(pkt) = dm.next_packet() {
        packets += 1;
        keyframes += usize::from(pkt.flags.keyframe);
        dec.send_packet(&pkt).unwrap();
        while let Ok(f) = dec.receive_frame() {
            frames += 1;
            check(f);
        }
    }
    dec.flush().unwrap();
    while let Ok(f) = dec.receive_frame() {
        frames += 1;
        check(f);
    }
    assert_eq!(packets, 25, "one packet per access unit");
    assert_eq!(keyframes, 1);
    assert_eq!(frames, 25);
}
