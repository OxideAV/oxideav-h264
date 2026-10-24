//! MBAFF (interlaced, `mb_adaptive_frame_field_flag = 1`) P / B
//! decode pinned to a black-box reference decoder's output.
//!
//! Each fixture is a synthetic, clean-licensed stream (lavfi `testsrc2`
//! re-interlaced top-field-first, encoded black-box with libx264) that
//! exercises one MBAFF rule. Expected output is the per-frame FNV-1a-64
//! of the reference decoder's planar 4:2:0 frames (`ffmpeg -f rawvideo
//! -pix_fmt yuv420p`), so the test needs no external tool. Every stream is
//! decoded at worker budgets 1 and 8.
//!
//! Recipe (all three): `ffmpeg -f lavfi -i testsrc2=size=WxH:rate=R
//! -vf tinterlace=mode=interleave_top,setfield=tff -frames:v N -c:v libx264
//! -preset medium -profile:v high -pix_fmt yuv420p -flags +ildct+ilme
//! -field_order tt -x264-params interlaced=1:tff=1:threads=1:<extra>`
//!
//! * `mbaff_implicit_wp_320x192.h264` — R=50, N=3, extra
//!   `bframes=1:ref=1:direct=none:weightb=1:partitions=none:8x8dct=0:cabac=0:b-adapt=0`:
//!   implicit weighted bi-prediction (`weighted_bipred_idc = 2`) in field
//!   macroblocks, where §8.4.2.3.2 takes currPicOrField / pic0 / pic1 as
//!   the FIELDS of the macroblock's parity and of the refIdxLX-selected
//!   parity, not the frames.
//! * `mbaff_spatial_direct_320x192.h264` — R=50, N=9, extra
//!   `bframes=3:ref=1:direct=spatial:weightb=0`: spatial direct
//!   B_Skip / B_Direct_16x16 where the co-located pair's frame / field
//!   coding differs from the current macroblock's — §8.4.1.2.2 colZeroFlag
//!   reads the co-located block through §8.4.1.2.1 Table 8-8 (mbAddrCol6 /
//!   mbAddrCol7 and yM) with mvCol unscaled (NOTE 2).
//! * `mbaff_intra8x8_field_left_640x352.h264` — R=60, N=24, extra
//!   `bframes=3:ref=1:no-deblock=1`, cut to its first two access units
//!   (IDR + first P): an Intra_8x8 frame macroblock whose left neighbour
//!   is an Intra_4x4 field macroblock — §8.3.2.1 eq. 8-72 takes
//!   Intra4x4PredMode[luma8x8BlkIdxA * 4 + 3] (n = 3) for luma8x8BlkIdx 2;
//!   the mode then propagates into the field pair below.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, ExecutionContext, Frame, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn decode_hashes(name: &str, threads: usize) -> Vec<u64> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).expect("read fixture");
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    dec.set_execution_context(&ExecutionContext::with_threads(threads));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), bytes).with_pts(0))
        .expect("send_packet");
    dec.flush().expect("flush");
    let mut hashes = Vec::new();
    while let Ok(Frame::Video(v)) = dec.receive_frame() {
        let mut frame = Vec::new();
        for p in v.image_planes() {
            frame.extend_from_slice(&p.data);
        }
        hashes.push(fnv1a64(&frame));
    }
    assert_eq!(
        dec.decode_error_count(),
        0,
        "{name}: a slice failed to decode"
    );
    hashes
}

fn assert_matches_reference(name: &str, expected: &[u64]) {
    for threads in [1, 8] {
        let got = decode_hashes(name, threads);
        assert_eq!(
            got.len(),
            expected.len(),
            "{name}: frame count at {threads} thread(s)"
        );
        for (i, (g, e)) in got.iter().zip(expected).enumerate() {
            assert_eq!(
                g, e,
                "{name}: frame {i} differs from the reference decoder at {threads} thread(s)"
            );
        }
    }
}

#[test]
fn mbaff_implicit_weighted_bipred_in_field_macroblocks() {
    assert_matches_reference(
        "mbaff_implicit_wp_320x192.h264",
        &[
            0x3b26_bd8b_a197_23e4,
            0x326c_2a07_1e87_cc21,
            0x9e0a_b3e9_7e88_78bb,
        ],
    );
}

#[test]
fn mbaff_spatial_direct_with_mixed_colocated_pairs() {
    assert_matches_reference(
        "mbaff_spatial_direct_320x192.h264",
        &[
            0xdd1f_ce3d_6cfb_1413,
            0xd92c_e11a_2089_b29c,
            0x6430_af32_2005_d794,
            0x4cf2_9e06_7d6a_2a18,
            0x5b66_c0a8_55d2_a6e1,
            0x836b_2253_c463_1aac,
            0x16cc_a659_c851_9748,
            0x311a_3a95_1c04_ca6d,
            0x1770_882b_9aaf_2dac,
        ],
    );
}

#[test]
fn mbaff_intra8x8_frame_mb_with_intra4x4_field_left_neighbour() {
    assert_matches_reference(
        "mbaff_intra8x8_field_left_640x352.h264",
        &[0x846f_cc81_08fa_7de8, 0x6cd4_eb7f_21cc_bec1],
    );
}
