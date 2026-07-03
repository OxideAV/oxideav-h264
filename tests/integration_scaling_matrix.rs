//! Round-388 — **non-flat scaling-list encode** integration tests.
//!
//! Exercises `EncoderConfig::scaling_matrix` on the CAVLC `encode_idr`
//! path at 4:2:0:
//!  * `SeqDefault` — SPS `seq_scaling_matrix_present_flag = 1` with
//!    every list coded as UseDefaultScalingMatrixFlag (§7.3.2.1.1.1,
//!    one `delta_scale = -8` per list) → Table 7-3/7-4 default
//!    matrices at sequence level.
//!  * `PicDefault` — the §7.3.2.2 PPS tail carries
//!    `pic_scaling_matrix_present_flag = 1` with the same UseDefault
//!    coding (6 lists, + 2 8x8 lists under `transform_8x8`).
//!  * The encoder's forward quantiser divides by the §8.5.9
//!    weightScale and its local recon runs the decoder's scaled
//!    inverse, so the emitted stream must reproduce the recon
//!    bit-exactly through BOTH our decoder and the black-box
//!    reference decoder — across I_16x16, I_4x4 and I_8x8 MBs.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::{Encoder, EncoderConfig, ScalingMatrixMode, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264::nal::{parse_nal_unit, AnnexBSplitter, NalUnitType};

const W: usize = 64;
const H: usize = 64;

/// Mixed content: smooth gradient (I_16x16 wins), oriented edges
/// (I_4x4 / I_8x8 win) and textured chroma, so the three-way RDO
/// exercises every intra transform shape under the scaled quantiser.
fn make_source() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let base = 40 + ((i * 2 + j) as u32 % 160);
            let tex = if i >= 32 && j >= 32 {
                (i as u32 * 37 + j as u32 * 101 + (i as u32 ^ j as u32) * 13) % 61
            } else if i < 32 && j >= 32 {
                if (i / 4) % 2 == 0 {
                    30
                } else {
                    0
                }
            } else {
                0
            };
            y[j * W + i] = (base + tex).min(235) as u8;
        }
    }
    let cw = W / 2;
    let ch = H / 2;
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            // Smooth chroma gradients: the scaled chroma DC + sparse AC
            // paths are exercised without tripping the (separately
            // fixed) dense-chroma nC coeff_token table selection.
            u[j * cw + i] = (80 + i as u32 * 2) as u8;
            v[j * cw + i] = (190u32.saturating_sub(j as u32 * 3)).max(30) as u8;
        }
    }
    (y, u, v)
}

fn encode(mode: ScalingMatrixMode, transform_8x8: bool) -> oxideav_h264::encoder::EncodedIdr {
    let (y, u, v) = make_source();
    let frame = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y,
        u: &u,
        v: &v,
    };
    let cfg = EncoderConfig {
        scaling_matrix: mode,
        transform_8x8,
        // The transform_8x8 variants run at a finer QP: under the
        // default matrices' heavier high-frequency weights the 8x8
        // trial only wins on this content when enough texture
        // survives quantisation (QP 26 → zero Intra_8x8 picks).
        qp: if transform_8x8 { 18 } else { 26 },
        ..EncoderConfig::new(W as u32, H as u32)
    };
    Encoder::new(cfg).encode_idr(&frame)
}

fn decode_own(stream: &[u8]) -> VideoFrame {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream.to_vec()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => return vf,
            Ok(_) => continue,
            Err(e) => panic!("no video frame decoded: {e:?}"),
        }
    }
}

fn plane_max_diff(vf: &VideoFrame, plane: usize, recon: &[u8], w: usize, h: usize) -> u32 {
    let p = &vf.planes[plane];
    let mut max = 0u32;
    for r in 0..h {
        for c in 0..w {
            max = max.max((p.data[r * p.stride + c] as i32).abs_diff(recon[r * w + c] as i32));
        }
    }
    max
}

fn psnr_y(orig: &[u8], recon: &[u8]) -> f64 {
    let mse: f64 = orig
        .iter()
        .zip(recon.iter())
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum::<f64>()
        / orig.len() as f64;
    if mse == 0.0 {
        99.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

fn assert_self_roundtrip(idr: &oxideav_h264::encoder::EncodedIdr, tag: &str) {
    let vf = decode_own(&idr.annex_b);
    assert_eq!(plane_max_diff(&vf, 0, &idr.recon_y, W, H), 0, "{tag} luma");
    assert_eq!(
        plane_max_diff(&vf, 1, &idr.recon_u, W / 2, H / 2),
        0,
        "{tag} Cb"
    );
    assert_eq!(
        plane_max_diff(&vf, 2, &idr.recon_v, W / 2, H / 2),
        0,
        "{tag} Cr"
    );
}

fn assert_ffmpeg_interop(idr: &oxideav_h264::encoder::EncodedIdr, tag: &str) {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: reference decoder binary not present");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h264-sm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let h264 = dir.join(format!("{tag}.h264"));
    let yuv = dir.join(format!("{tag}.yuv"));
    std::fs::write(&h264, &idr.annex_b).expect("write stream");
    let status = std::process::Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-y"])
        .arg(&yuv)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success(), "reference decoder rejected the stream");
    let data = std::fs::read(&yuv).expect("read yuv");
    let (cw, ch) = (W / 2, H / 2);
    assert_eq!(data.len(), W * H + 2 * cw * ch);
    assert_eq!(&data[..W * H], &idr.recon_y[..], "{tag}: luma mismatch");
    assert_eq!(
        &data[W * H..W * H + cw * ch],
        &idr.recon_u[..],
        "{tag}: Cb mismatch"
    );
    assert_eq!(
        &data[W * H + cw * ch..],
        &idr.recon_v[..],
        "{tag}: Cr mismatch"
    );
}

#[test]
fn seq_default_sps_signals_matrices_and_promotes_profile() {
    let idr = encode(ScalingMatrixMode::SeqDefault, false);
    let mut saw = false;
    for nal in AnnexBSplitter::new(&idr.annex_b) {
        let unit = parse_nal_unit(nal).expect("parse NAL");
        if unit.header.nal_unit_type == NalUnitType::Sps {
            let sps = oxideav_h264::sps::Sps::parse(&unit.rbsp).expect("parse SPS");
            assert!(sps.profile_idc >= 100, "High-family profile required");
            assert!(sps.seq_scaling_matrix_present_flag);
            saw = true;
        }
    }
    assert!(saw);
}

#[test]
fn pic_default_pps_signals_matrices() {
    let idr = encode(ScalingMatrixMode::PicDefault, false);
    let mut saw = false;
    for nal in AnnexBSplitter::new(&idr.annex_b) {
        let unit = parse_nal_unit(nal).expect("parse NAL");
        if unit.header.nal_unit_type == NalUnitType::Pps {
            let pps = oxideav_h264::pps::Pps::parse(&unit.rbsp).expect("parse PPS");
            let ext = pps.extension.as_ref().expect("PPS tail present");
            assert!(ext.pic_scaling_matrix_present_flag);
            saw = true;
        }
    }
    assert!(saw);
}

#[test]
fn seq_default_self_roundtrip_bit_exact() {
    let idr = encode(ScalingMatrixMode::SeqDefault, false);
    assert_self_roundtrip(&idr, "seq-default");
    // Quality floor: default matrices shape quantisation but the
    // picture must stay comfortably intact at QP 26.
    let (y, _, _) = make_source();
    assert!(psnr_y(&y, &idr.recon_y) > 32.0, "PSNR floor");
}

#[test]
fn seq_default_transform_8x8_self_roundtrip_bit_exact() {
    let idr = encode(ScalingMatrixMode::SeqDefault, true);
    assert!(
        idr.i8x8_mb_count > 0,
        "smooth MBs must still pick Intra_8x8 under the scaled RDO"
    );
    assert_self_roundtrip(&idr, "seq-default-8x8");
}

#[test]
fn pic_default_self_roundtrip_bit_exact() {
    let idr = encode(ScalingMatrixMode::PicDefault, false);
    assert_self_roundtrip(&idr, "pic-default");
}

#[test]
fn pic_default_transform_8x8_self_roundtrip_bit_exact() {
    let idr = encode(ScalingMatrixMode::PicDefault, true);
    assert_self_roundtrip(&idr, "pic-default-8x8");
}

#[test]
fn seq_default_reference_decoder_interop_bit_exact() {
    let idr = encode(ScalingMatrixMode::SeqDefault, false);
    assert_ffmpeg_interop(&idr, "seq-default");
}

#[test]
fn seq_default_transform_8x8_reference_decoder_interop_bit_exact() {
    let idr = encode(ScalingMatrixMode::SeqDefault, true);
    assert_ffmpeg_interop(&idr, "seq-default-8x8");
}

#[test]
fn pic_default_reference_decoder_interop_bit_exact() {
    let idr = encode(ScalingMatrixMode::PicDefault, false);
    assert_ffmpeg_interop(&idr, "pic-default");
}

#[test]
fn pic_default_transform_8x8_reference_decoder_interop_bit_exact() {
    let idr = encode(ScalingMatrixMode::PicDefault, true);
    assert_ffmpeg_interop(&idr, "pic-default-8x8");
}

/// The flat mode must remain bit-identical to the historical output:
/// the `_w` quantisers with all-16 weightScale reduce to the exact
/// legacy arithmetic, so a flat encode still ffmpeg-interops clean.
#[test]
fn flat_mode_unchanged_and_interops() {
    let idr = encode(ScalingMatrixMode::Flat, false);
    assert_self_roundtrip(&idr, "flat");
    assert_ffmpeg_interop(&idr, "flat");
}
