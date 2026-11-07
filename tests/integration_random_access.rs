//! Random access and loss resilience (#11) — decoding that does not
//! start at an IDR picture, as for a broadcast stream joined mid-way
//! (DVB carries no IDR at all: entry points are non-IDR I pictures or
//! intra-refresh P pictures marked by recovery point SEI messages).
//!
//! * D.2.8 — decoding starts at the access unit carrying the recovery
//!   point (or at an intra-coded picture); references to pictures that
//!   precede it are inferred as all-intra mid-grey pictures (NOTE 3),
//!   and pictures preceding the recovery point in output order (open-GOP
//!   leading B pictures, the refresh period of an intra-refresh stream)
//!   are decoded but not output.
//! * §8.2.5.2 NOTE 2 — a frame_num jump the SPS does not allow is an
//!   unintentional loss: the decoder runs the gap process with
//!   concealment references and resynchronises at the next recovery
//!   point instead of rejecting every later picture.
//!
//! Fixtures are synthetic, clean-licensed streams (lavfi `testsrc2`
//! encoded black-box with libx264, access unit delimiters inserted);
//! expected output is the per-frame FNV-1a-64 of a black-box reference
//! decoder's planar 4:2:0 output (`ffmpeg -f rawvideo -pix_fmt yuv420p`)
//! for the same cut / damaged stream, so the test needs no external tool.
//!
//! * `mbaff_open_gop_320x192.h264` — `testsrc2=size=320x192:rate=50`,
//!   `tinterlace=mode=interleave_top,setfield=tff`, 20 frames,
//!   `-profile:v high -flags +ildct+ilme -field_order tt -bsf:v
//!   h264_metadata=aud=insert -x264-params
//!   interlaced=1:tff=1:threads=1:keyint=8:min-keyint=8:open-gop=1:bframes=3:b-adapt=0:b-pyramid=normal:ref=3:weightp=2:slices=2:scenecut=0:repeat-headers=1`
//!   — 1080i-broadcast-shaped MBAFF CABAC, two slices per picture,
//!   open GOPs whose non-IDR I pictures are followed (in decoding order)
//!   by three leading B pictures that reference the previous GOP.
//! * `intra_refresh_192x128.h264` — `testsrc2=size=192x128:rate=25`,
//!   30 frames, `-x264-params
//!   threads=1:keyint=10:intra-refresh=1:bframes=0:ref=1:scenecut=0:repeat-headers=1`
//!   — no I picture after the first: every tenth P picture carries a
//!   recovery point SEI message with recovery_frame_cnt = 9.

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

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).expect("read fixture")
}

/// Split an Annex B stream into access units at its access unit
/// delimiters (§7.4.1.2.3; every fixture carries one per access unit).
fn access_units(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i + 3 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            let s = if i > 0 && data[i - 1] == 0 { i - 1 } else { i };
            if data[i + 3] & 0x1f == 9 {
                starts.push(s);
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut aus = Vec::with_capacity(starts.len());
    for (k, &s) in starts.iter().enumerate() {
        let e = starts.get(k + 1).copied().unwrap_or(data.len());
        aus.push(&data[s..e]);
    }
    aus
}

/// Decode `stream` one access unit per packet at a worker budget;
/// returns the per-frame hashes.
fn decode_hashes(stream: &[u8], threads: usize) -> Vec<u64> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    dec.set_execution_context(&ExecutionContext::with_threads(threads));
    let mut hashes = Vec::new();
    let mut drain = |dec: &mut H264CodecDecoder| {
        while let Ok(f) = dec.receive_frame() {
            if let Frame::Video(v) = f {
                let mut frame = Vec::new();
                for p in v.image_planes() {
                    frame.extend_from_slice(&p.data);
                }
                hashes.push(fnv1a64(&frame));
            }
        }
    };
    for (n, au) in access_units(stream).into_iter().enumerate() {
        let _ =
            dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), au.to_vec()).with_pts(n as i64));
        drain(&mut dec);
    }
    dec.flush().expect("flush");
    drain(&mut dec);
    hashes
}

fn cut_at(stream: &[u8], first_au: usize) -> Vec<u8> {
    access_units(stream)[first_au..].concat()
}

fn without(stream: &[u8], dropped_au: usize) -> Vec<u8> {
    access_units(stream)
        .into_iter()
        .enumerate()
        .filter(|&(k, _)| k != dropped_au)
        .flat_map(|(_, au)| au.iter().copied())
        .collect()
}

fn assert_hashes(what: &str, stream: &[u8], expected: &[u64]) {
    for threads in [1, 8] {
        let got = decode_hashes(stream, threads);
        assert_eq!(
            got.len(),
            expected.len(),
            "{what}: frame count at {threads} thread(s)"
        );
        for (i, (g, e)) in got.iter().zip(expected).enumerate() {
            assert_eq!(
                g, e,
                "{what}: frame {i} differs from the reference decoder at {threads} thread(s)"
            );
        }
    }
}

/// The black-box reference decoder's output for `intra_refresh_192x128`
/// decoded from its IDR: 30 frames.
const INTRA_REFRESH_FULL: [u64; 30] = [
    0x4586_a697_dcb4_d56f,
    0x2ef0_95c0_062c_57e8,
    0x5371_7dc1_3386_8cde,
    0x4300_61b2_3fac_0c88,
    0x6cb9_e796_9c52_be69,
    0xd61a_e47e_40bd_847f,
    0xacc4_e8b2_9d7c_026d,
    0x3392_d26d_f641_5106,
    0x7148_92f2_9907_d65d,
    0xdee6_165c_dfd0_855a,
    0xfa54_8308_c8a3_a897,
    0x67be_0941_defe_13e3,
    0x6426_282b_48d2_094b,
    0x4076_93a8_f50d_724e,
    0xe59e_981d_c247_41bf,
    0x3e23_b7eb_f943_9c1f,
    0xb090_c3be_dc8e_6f35,
    0x1448_5bf2_76c5_cd19,
    0x7a30_5ce2_a0e5_7944,
    0x6ea4_6ead_e404_25da,
    0xa288_bdfc_b53c_6405,
    0x1aa3_d077_932a_cff4,
    0x2ad4_20f9_fa80_5b12,
    0x5db0_bbdc_a87b_5e92,
    0x5502_a0c4_c4ef_a913,
    0xe8d8_2caa_78f7_aacc,
    0xe2ff_1792_d134_f171,
    0xd66b_ae45_b0eb_fbbb,
    0xfa2d_4c66_e118_db00,
    0xa8b5_f43e_b0cf_32ae,
];

#[test]
fn mbaff_open_gop_entered_at_a_non_idr_i_picture() {
    // Access unit 5 is the first non-IDR I picture; the three B
    // pictures after it in decoding order precede it in output order
    // and reference the GOP before the cut. The reference decoder
    // outputs the 12 pictures from the I picture on.
    let stream = cut_at(&fixture("mbaff_open_gop_320x192.h264"), 5);
    assert_hashes(
        "mbaff open GOP from AU 5",
        &stream,
        &[
            0xfa97_63f7_34ee_1321,
            0x1f28_820a_cc58_0f1e,
            0x4cf4_2628_5f4d_451f,
            0x9cc9_fd30_bbd1_0de8,
            0x7932_5adb_1674_0da3,
            0x3343_341c_69d1_c1a2,
            0x0db6_f85d_acdb_0bc7,
            0x8b90_798f_64ab_bfc7,
            0x3d3c_58f3_11e5_371e,
            0xf9c6_8ccc_7c69_f3df,
            0x45b1_1e8b_250f_459a,
            0x83d7_0988_8521_df63,
        ],
    );
}

#[test]
fn intra_refresh_entered_at_a_recovery_point_p_picture() {
    // Access unit 10 is a P picture with recovery_frame_cnt = 9: the
    // nine pictures of the refresh period are decoded against the
    // inferred grey references but not output; output starts at the
    // recovery point (frame 19 of the full stream) and matches the
    // full decode from there on.
    let stream = cut_at(&fixture("intra_refresh_192x128.h264"), 10);
    assert_hashes(
        "intra refresh from AU 10",
        &stream,
        &INTRA_REFRESH_FULL[19..],
    );
}

#[test]
fn lost_reference_picture_is_concealed_and_resynchronised() {
    // Access unit 4 (a reference P picture) is lost. Every later
    // picture is still decoded and output (29 frames, the reference
    // decoder's count); the pictures before the loss and those from
    // the next recovery point (frame 19) on are exact.
    let stream = without(&fixture("intra_refresh_192x128.h264"), 4);
    for threads in [1, 8] {
        let got = decode_hashes(&stream, threads);
        assert_eq!(got.len(), 29, "frame count at {threads} thread(s)");
        assert_eq!(
            got[..4],
            INTRA_REFRESH_FULL[..4],
            "pictures before the loss at {threads} thread(s)"
        );
        assert_eq!(
            got[18..],
            INTRA_REFRESH_FULL[19..],
            "pictures from the recovery point at {threads} thread(s)"
        );
    }
}
