//! Chroma format coverage probes.
//!
//! `oxideav-h264` currently ships 4:2:0 (`chroma_format_idc = 1`) decode
//! only. The spec (§6.4.1 / §7.4.2.1.1) defines three other chroma modes
//! that the MB-layer pipeline does *not* yet handle — different chroma
//! plane dimensions, different DC-transform sizes, different intra mode
//! set (§8.3.4 / §8.3.3), different sub-pel filter (§8.4.2.2.1) and
//! different deblock internal-edge layout (§8.7.2). The front-end
//! decoder rejects those streams at slice entry so upstream callers can
//! route to a fallback rather than receive silently-corrupt YUV.
//!
//! These tests lock the reject in place and anchor the acceptance
//! fixtures that will be re-enabled when 4:2:2 / 4:4:4 support lands:
//!
//! * `tests/fixtures/iframe_yuv422_64x64.{es,yuv}` — High 4:2:2
//!   64×64 IDR produced by:
//!
//!   ```bash
//!   ffmpeg -y -f lavfi -i testsrc=duration=1:size=64x64:rate=1 -vframes 1 \
//!     -pix_fmt yuv422p -c:v libx264 -profile:v high422 -preset ultrafast \
//!     /tmp/yuv422.mp4
//!   ffmpeg -y -i /tmp/yuv422.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 \
//!     tests/fixtures/iframe_yuv422_64x64.es
//!   ffmpeg -y -i /tmp/yuv422.mp4 -pix_fmt yuv422p \
//!     tests/fixtures/iframe_yuv422_64x64.yuv
//!   ```
//!
//! * `tests/fixtures/iframe_yuv444_64x64.{es,yuv}` — High 4:4:4
//!   Predictive 64×64 IDR produced analogously with `-pix_fmt yuv444p`
//!   and `-profile:v high444`.

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, Packet, TimeBase};
use oxideav_h264::decoder::H264Decoder;
use oxideav_h264::nal::{extract_rbsp, split_annex_b, NalHeader, NalUnitType};
use oxideav_h264::sps::parse_sps;

fn read_fixture(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// Locate the first SPS in an Annex B stream and return its parsed chroma
/// format idc. Panics if no SPS is present (the fixtures always carry one).
fn chroma_format_of(es: &[u8]) -> u32 {
    for nalu in split_annex_b(es) {
        let h = NalHeader::parse(nalu[0]).unwrap();
        if h.nal_unit_type == NalUnitType::Sps {
            let rbsp = extract_rbsp(&nalu[1..]);
            let sps = parse_sps(&h, &rbsp).expect("parse SPS");
            return sps.chroma_format_idc;
        }
    }
    panic!("no SPS in stream");
}

fn first_packet(es: &[u8]) -> Packet {
    // Feed the decoder SPS + PPS + first slice back-to-back; it extracts
    // the IDR start-code and the surrounding parameter sets from the
    // Annex B bytes.
    let nalus = split_annex_b(es);
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
    let sps = sps_nal.expect("no SPS");
    let pps = pps_nal.expect("no PPS");
    let idr = idr_nal.expect("no IDR slice");
    let mut packet_data = Vec::new();
    packet_data.extend_from_slice(&[0, 0, 0, 1]);
    packet_data.extend_from_slice(sps);
    packet_data.extend_from_slice(&[0, 0, 0, 1]);
    packet_data.extend_from_slice(pps);
    packet_data.extend_from_slice(&[0, 0, 0, 1]);
    packet_data.extend_from_slice(idr);
    Packet::new(0, TimeBase::new(1, 90_000), packet_data)
        .with_pts(0)
        .with_keyframe(true)
}

#[test]
fn yuv422_fixture_carries_chroma_format_idc_2() {
    // §6.4.1 / §7.4.2.1.1 — ffmpeg `high422` encoder must emit an SPS
    // with `chroma_format_idc = 2` (4:2:2, SubWidthC=2 / SubHeightC=1).
    // If this ever changes (fixture regeneration), the reject assertion
    // below would trigger on the wrong branch.
    let es = read_fixture("tests/fixtures/iframe_yuv422_64x64.es");
    assert_eq!(chroma_format_of(&es), 2, "expected 4:2:2 fixture");
}

#[test]
fn yuv444_fixture_carries_chroma_format_idc_3() {
    // §6.4.1 — ffmpeg `high444` emits `chroma_format_idc = 3` (4:4:4,
    // SubWidthC=SubHeightC=1, no chroma sub-sampling).
    let es = read_fixture("tests/fixtures/iframe_yuv444_64x64.es");
    assert_eq!(chroma_format_of(&es), 3, "expected 4:4:4 fixture");
}

#[test]
fn decode_yuv422_iframe_matches_reference() {
    // §6.4.1 — `chroma_format_idc = 2` (4:2:2): chroma planes have half
    // the luma width but the same height. Chroma DC uses the 2×4
    // Hadamard (§8.5.11.2), each MB carries 8 chroma AC blocks per
    // plane (§9.2.1.2 ChromaArrayType == 2), and intra chroma
    // prediction runs over an 8×16 tile (§8.3.4). This test drives a
    // 64×64 yuv422p IDR produced by x264 (`high422` profile) and
    // asserts the reconstructed frame matches the ffmpeg-decoded
    // reference at ≥ 99 % of samples (ideally bit-exact).
    use oxideav_core::PixelFormat;
    use oxideav_core::Frame;
    let es = read_fixture("tests/fixtures/iframe_yuv422_64x64.es");
    let ref_yuv = read_fixture("tests/fixtures/iframe_yuv422_64x64.yuv");
    let mut dec = H264Decoder::new(CodecId::new("h264"));
    dec.send_packet(&first_packet(&es))
        .expect("4:2:2 CAVLC I decode should succeed");
    let frame = match dec.receive_frame().expect("frame") {
        Frame::Video(v) => v,
        _ => panic!("not a video frame"),
    };
    assert_eq!(frame.format, PixelFormat::Yuv422P);
    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 64);
    assert_eq!(frame.planes.len(), 3);
    // Reference fixture is raw yuv422p: 64×64 Y followed by 32×64 Cb then
    // 32×64 Cr — chroma stride 32, same height as luma.
    assert_eq!(ref_yuv.len(), 64 * 64 + 32 * 64 * 2);
    let mut got = Vec::with_capacity(ref_yuv.len());
    // Luma plane
    {
        let p = &frame.planes[0];
        for row in 0..64usize {
            let off = row * p.stride;
            got.extend_from_slice(&p.data[off..off + 64]);
        }
    }
    for pi in 1..=2usize {
        let p = &frame.planes[pi];
        for row in 0..64usize {
            let off = row * p.stride;
            got.extend_from_slice(&p.data[off..off + 32]);
        }
    }
    assert_eq!(got.len(), ref_yuv.len());
    let matches = got
        .iter()
        .zip(ref_yuv.iter())
        .filter(|(a, b)| a == b)
        .count();
    let ratio = matches as f64 / got.len() as f64;
    assert!(
        ratio >= 0.99,
        "4:2:2 decode accuracy {:.3}% — below 99%",
        ratio * 100.0
    );
}

#[test]
fn decode_yuv444_iframe_matches_reference() {
    // §6.4.1 — `chroma_format_idc = 3` (4:4:4): chroma planes have the
    // same dimensions as luma and reuse the luma Intra_4×4 / Intra_16×16
    // predictors plus the luma residual schedule (no chroma DC 2×2). The
    // CAVLC I-slice path decodes three plane-aligned luma-style residual
    // streams per macroblock. This test drives a 64×64 yuv444p IDR
    // produced by x264 (`high444` profile) and asserts the reconstructed
    // frame matches the ffmpeg-decoded reference byte-for-byte (or at
    // worst ≥ 99% of the samples).
    use oxideav_core::PixelFormat;
    use oxideav_core::Frame;
    let es = read_fixture("tests/fixtures/iframe_yuv444_64x64.es");
    let ref_yuv = read_fixture("tests/fixtures/iframe_yuv444_64x64.yuv");
    let mut dec = H264Decoder::new(CodecId::new("h264"));
    dec.send_packet(&first_packet(&es))
        .expect("4:4:4 CAVLC I decode should succeed");
    let frame = match dec.receive_frame().expect("frame") {
        Frame::Video(v) => v,
        _ => panic!("not a video frame"),
    };
    assert_eq!(frame.format, PixelFormat::Yuv444P);
    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 64);
    assert_eq!(frame.planes.len(), 3);
    // Reference fixture is raw yuv444p: 64×64 Y followed by 64×64 Cb then
    // 64×64 Cr — all at stride 64.
    assert_eq!(ref_yuv.len(), 64 * 64 * 3);
    let mut got = Vec::with_capacity(64 * 64 * 3);
    for p in &frame.planes {
        for row in 0..64 {
            let off = row * p.stride;
            got.extend_from_slice(&p.data[off..off + 64]);
        }
    }
    assert_eq!(got.len(), ref_yuv.len());
    let matches = got
        .iter()
        .zip(ref_yuv.iter())
        .filter(|(a, b)| a == b)
        .count();
    let ratio = matches as f64 / got.len() as f64;
    assert!(
        ratio >= 0.99,
        "4:4:4 decode accuracy {:.3}% — below 99%",
        ratio * 100.0
    );
    assert_eq!(
        got, ref_yuv,
        "expected bit-exact 4:4:4 decode against x264-encoded reference"
    );
}

#[test]
fn chroma_plane_dims_for_known_formats() {
    use oxideav_h264::picture::{chroma_plane_h, chroma_plane_w, chroma_subsampling};
    // §6.4.1 Table 6-1.
    assert_eq!(chroma_subsampling(1), (2, 2));
    assert_eq!(chroma_subsampling(2), (2, 1));
    assert_eq!(chroma_subsampling(3), (1, 1));
    // 64×64 luma.
    assert_eq!((chroma_plane_w(64, 1), chroma_plane_h(64, 1)), (32, 32));
    assert_eq!((chroma_plane_w(64, 2), chroma_plane_h(64, 2)), (32, 64));
    assert_eq!((chroma_plane_w(64, 3), chroma_plane_h(64, 3)), (64, 64));
}
