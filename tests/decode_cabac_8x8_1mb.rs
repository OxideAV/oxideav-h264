// Regression for the CABAC I-slice 8×8-transform residual drift
// (wt/cabac-8x8-drift). The pre-fix decoder over-read `rem_intra4x4_pred_mode`
// in bypass mode, consuming 3 raw bypass bits per non-MPM 4×4 block instead
// of 3 context-coded bins at ctxIdx 69. Under ``-preset medium -coder 1
// -x264-params 8x8dct=1:i8x8=1`` (High Profile) the CABAC state drifted by
// a handful of bits inside the first I-slice MB, which cascaded so far that
// the `end_of_slice_flag` never fired — raising
// ``h264 cabac slice: end_of_slice_flag did not fire before last MB'' and
// downstream `ref_idx_l0 N out of range` on longer fixtures.
//
// See §9.3.3.1.1.7 / Table 9-34 (ctxIdxOffset 276, ctxIdxInc 0 → absolute
// ctxIdx 69 for all three FL bins of rem_intra4x4_pred_mode).
//
// Fixture: a 32×32 High-Profile IDR with transform_8x8 enabled. A 32×32
// frame is just large enough that x264 picks transform_size_8x8_flag = 1 on
// at least one macroblock, so the 8×8 residual path is exercised alongside
// the 4×4 intra-pred-mode flow that previously drifted.
//
// Regenerate with:
// ```bash
// ffmpeg -y -f lavfi -i 'mandelbrot=size=32x32:rate=1' -vframes 1 \
//   -pix_fmt yuv420p -c:v libx264 -profile:v high -preset medium \
//   -coder 1 -x264-params 8x8dct=1:i8x8=1:no-scenecut=1:keyint=1 \
//   /tmp/cabac_8x8_32.mp4
// ffmpeg -y -i /tmp/cabac_8x8_32.mp4 -c:v copy -bsf:v h264_mp4toannexb \
//   tests/fixtures/cabac_8x8_1mb.es
// ffmpeg -y -i /tmp/cabac_8x8_32.mp4 -pix_fmt yuv420p \
//   tests/fixtures/cabac_8x8_1mb.yuv
// ```

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::decoder::H264Decoder;
use oxideav_h264::nal::{split_annex_b, NalHeader, NalUnitType};

#[test]
fn cabac_8x8_drift_does_not_panic_end_of_slice() {
    let es = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/cabac_8x8_1mb.es"
    ))
    .unwrap();
    let nalus = split_annex_b(&es);
    let mut sps_nal: Option<&[u8]> = None;
    let mut pps_nal: Option<&[u8]> = None;
    let mut idr_nal: Option<&[u8]> = None;
    for nalu in &nalus {
        let h = NalHeader::parse(nalu[0]).unwrap();
        match h.nal_unit_type {
            NalUnitType::Sps if sps_nal.is_none() => sps_nal = Some(nalu),
            NalUnitType::Pps if pps_nal.is_none() => pps_nal = Some(nalu),
            NalUnitType::SliceIdr if idr_nal.is_none() => idr_nal = Some(nalu),
            _ => {}
        }
    }
    let sps = sps_nal.unwrap();
    let pps = pps_nal.unwrap();
    let idr = idr_nal.unwrap();
    let mut packet_data = Vec::new();
    packet_data.extend_from_slice(&[0, 0, 0, 1]);
    packet_data.extend_from_slice(sps);
    packet_data.extend_from_slice(&[0, 0, 0, 1]);
    packet_data.extend_from_slice(pps);
    packet_data.extend_from_slice(&[0, 0, 0, 1]);
    packet_data.extend_from_slice(idr);
    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 90_000), packet_data)
        .with_pts(0)
        .with_keyframe(true);
    // Before the fix this produced `InvalidData("h264 cabac slice:
    // end_of_slice_flag did not fire before last MB")` because the CABAC
    // bit-cursor drifted by ~3 bits per non-MPM 4×4 intra-pred-mode block
    // and the last MB never reached its terminating bin.
    dec.send_packet(&pkt).expect(
        "h264 cabac 8×8 High Profile slice failed to terminate — \
         rem_intra4x4_pred_mode should be regular-mode ctx 69 bins, \
         not bypass (§9.3.3.1.1.7)",
    );
    let f = match dec.receive_frame().expect("receive_frame") {
        Frame::Video(f) => f,
        _ => panic!("non-video frame"),
    };
    assert_eq!(f.width, 32);
    assert_eq!(f.height, 32);
    // Pixel-exact matches for this regression live in the 128×128
    // decode_cabac_8x8 soft-check. Here we only guard against the
    // bitstream-drift signature surfacing again.
}
