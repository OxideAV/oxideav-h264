//! Round-448 — **monochrome (4:0:0, `chroma_format_idc = 0`) encode +
//! decode**.
//!
//! §7.4.2.1.1 / §6.2: `chroma_format_idc = 0` selects ChromaArrayType
//! 0 — the coded picture has a luma plane only. Syntax differences
//! from 4:2:0 exercised here:
//!   * §7.3.2.1.1 — the SPS codes `chroma_format_idc = 0` inside the
//!     chroma-extended group (High-family profile required; §A.2.4
//!     lists 4:0:0 in the High profile's supported chroma range).
//!   * §7.3.5.1 — no `intra_chroma_pred_mode`.
//!   * §7.3.5.3 — `residual()` invokes `residual_luma()` only; no
//!     chroma DC / AC blocks exist.
//!   * §9.1.2 — `coded_block_pattern` uses the Table 9-4(b) column
//!     (ChromaArrayType ∈ {0, 3}): cbp is luma-only, 0..=15.
//!   * §8.7 — the deblocking filter runs on the luma plane only.
//!
//! Gates: bit-exact self-roundtrip (decoder output == encoder local
//! recon) on an IDR + P + P GOP, single-plane output shape, and a
//! byte-exact black-box cross-check against a stock reference decoder
//! binary when present.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 80;
const H: usize = 64;

/// Textured luma frame `k` of a slow pan + brightness drift, so P
/// pictures find real motion and coded residual.
fn make_luma(k: usize) -> Vec<u8> {
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let x = i + 2 * k; // 2 px/frame horizontal pan
            let base = 40 + ((x * 2 + j) % 160) as u32;
            let tex = ((x * 37 + j * 101 + (x ^ j) * 13) % 61) as u32;
            let drift = (3 * k) as u32;
            y[j * W + i] = (base + tex / 2 + drift).min(235) as u8;
        }
    }
    y
}

struct MonoGop {
    annex_b: Vec<u8>,
    /// Per-frame luma recon (the encoder's decoder-mirror).
    recon: Vec<Vec<u8>>,
}

fn encode_mono_gop_entropy(qp: i32, n_p: usize, cabac: bool) -> MonoGop {
    let cfg = EncoderConfig {
        chroma_format_idc: 0,
        profile_idc: 100,
        cabac,
        qp,
        ..EncoderConfig::new(W as u32, H as u32)
    };
    let enc = Encoder::new(cfg);

    let y0 = make_luma(0);
    let f0 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y0,
        u: &[],
        v: &[],
    };
    let idr = if cabac {
        enc.encode_idr_cabac(&f0)
    } else {
        enc.encode_idr(&f0)
    };
    let mut annex_b = idr.annex_b.clone();
    let mut recon = vec![idr.recon_y.clone()];

    // P chain: each picture references the previous one.
    let mut prev_idr = Some(idr);
    let mut prev_p: Option<oxideav_h264::encoder::EncodedP> = None;
    for k in 1..=n_p {
        let yk = make_luma(k);
        let fk = YuvFrame {
            width: W as u32,
            height: H as u32,
            y: &yk,
            u: &[],
            v: &[],
        };
        let encode = |prev_ref: &EncodedFrameRef<'_>| {
            if cabac {
                enc.encode_p_cabac(&fk, prev_ref, k as u32, 2 * k as u32)
            } else {
                enc.encode_p(&fk, prev_ref, k as u32, 2 * k as u32)
            }
        };
        let p = if let Some(prev) = prev_p.take() {
            encode(&EncodedFrameRef::from(&prev))
        } else {
            let idr = prev_idr.take().expect("idr present");
            encode(&EncodedFrameRef::from(&idr))
        };
        annex_b.extend_from_slice(&p.annex_b);
        recon.push(p.recon_y.clone());
        prev_p = Some(p);
    }
    MonoGop { annex_b, recon }
}

fn encode_mono_gop(qp: i32, n_p: usize) -> MonoGop {
    encode_mono_gop_entropy(qp, n_p, false)
}

fn decode_ours(annex_b: &[u8]) -> Vec<VideoFrame> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), annex_b.to_vec()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    while let Ok(Frame::Video(vf)) = dec.receive_frame() {
        frames.push(vf);
    }
    frames
}

fn assert_mono_roundtrip(gop: &MonoGop, tag: &str) {
    let frames = decode_ours(&gop.annex_b);
    assert_eq!(frames.len(), gop.recon.len(), "{tag}: frame count");
    for (i, (vf, exp)) in frames.iter().zip(gop.recon.iter()).enumerate() {
        // ChromaArrayType 0 → single-plane output.
        assert_eq!(vf.planes.len(), 1, "{tag}: frame {i} plane count");
        let p = &vf.planes[0];
        assert_eq!(p.stride, W, "{tag}: frame {i} stride");
        let mismatches = p
            .data
            .iter()
            .zip(exp.iter())
            .filter(|(a, b)| a != b)
            .count();
        if mismatches != 0 {
            let mut mbs: Vec<(usize, usize)> = Vec::new();
            for (off, (a, b)) in p.data.iter().zip(exp.iter()).enumerate() {
                if a != b {
                    let key = ((off % W) / 16, (off / W) / 16);
                    if !mbs.contains(&key) {
                        mbs.push(key);
                    }
                }
            }
            eprintln!("{tag}: frame {i} diverging MBs {mbs:?}");
        }
        assert_eq!(
            mismatches,
            0,
            "{tag}: frame {i} luma: {mismatches}/{} samples differ",
            exp.len(),
        );
    }
}

/// Black-box cross-check: a stock reference decoder binary must
/// reconstruct the exact same luma plane. Skips when the binary is
/// not present.
///
/// The reference decoder surfaces a 4:0:0 coded stream as `yuv420p`
/// output (synthetic mid-gray chroma planes appended); asking its
/// scaler for `gray` output instead runs a colourspace conversion
/// that perturbs the luma samples, so we take the native rawvideo
/// output and compare the luma portion of each 3/2-sized frame.
fn reference_decoder_check(gop: &MonoGop, tag: &str) {
    let refbin = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !refbin.exists() {
        eprintln!("skip {tag}: reference decoder binary not present");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h264-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bs = dir.join("input.h264");
    let out = dir.join("out.yuv");
    std::fs::write(&bs, &gop.annex_b).unwrap();
    let status = std::process::Command::new(refbin)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&bs)
        .args(["-f", "rawvideo"])
        .arg(&out)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success(), "{tag}: reference decoder failed");
    let raw = std::fs::read(&out).unwrap();
    let frame_bytes = W * H * 3 / 2;
    assert_eq!(
        raw.len(),
        frame_bytes * gop.recon.len(),
        "{tag}: reference decoder output size",
    );
    for (i, exp) in gop.recon.iter().enumerate() {
        let got = &raw[i * frame_bytes..i * frame_bytes + W * H];
        let mismatches = got.iter().zip(exp.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(
            mismatches,
            0,
            "{tag}: frame {i}: {mismatches}/{} samples differ vs reference decoder",
            exp.len(),
        );
    }
    if std::env::var("OXIDEAV_KEEP_STREAM").is_err() {
        let _ = std::fs::remove_dir_all(&dir);
    } else {
        eprintln!("kept stream at {}", bs.display());
    }
}

#[test]
fn mono_idr_self_roundtrip_bit_exact() {
    let gop = encode_mono_gop(26, 0);
    assert_mono_roundtrip(&gop, "mono-idr");
}

#[test]
fn mono_idr_p_p_self_roundtrip_bit_exact() {
    let gop = encode_mono_gop(26, 2);
    assert_mono_roundtrip(&gop, "mono-idr-p-p");
}

#[test]
fn mono_low_qp_self_roundtrip_bit_exact() {
    // Dense residual (QP 12): every AC path in the luma-only writer
    // carries coefficients; the §9.2.1.1 nC chain is exercised hard.
    let gop = encode_mono_gop(12, 2);
    assert_mono_roundtrip(&gop, "mono-low-qp");
}

#[test]
fn mono_high_qp_self_roundtrip_bit_exact() {
    // Sparse residual (QP 44): P pictures produce real P_Skip runs.
    let gop = encode_mono_gop(44, 2);
    assert_mono_roundtrip(&gop, "mono-high-qp");
}

#[test]
fn mono_idr_reference_decoder_byte_exact() {
    let gop = encode_mono_gop(26, 0);
    reference_decoder_check(&gop, "mono-idr-ref");
}

#[test]
fn mono_idr_p_p_reference_decoder_byte_exact() {
    let gop = encode_mono_gop(26, 2);
    reference_decoder_check(&gop, "mono-gop-ref");
}

#[test]
fn mono_low_qp_reference_decoder_byte_exact() {
    let gop = encode_mono_gop(12, 2);
    reference_decoder_check(&gop, "mono-lowqp-ref");
}

// ---- CABAC entropy path (round-448 second leg) ----

#[test]
fn mono_cabac_idr_self_roundtrip_bit_exact() {
    let gop = encode_mono_gop_entropy(26, 0, true);
    assert_mono_roundtrip(&gop, "mono-cabac-idr");
}

#[test]
fn mono_cabac_idr_p_p_self_roundtrip_bit_exact() {
    let gop = encode_mono_gop_entropy(26, 2, true);
    assert_mono_roundtrip(&gop, "mono-cabac-idr-p-p");
}

#[test]
fn mono_cabac_low_qp_self_roundtrip_bit_exact() {
    let gop = encode_mono_gop_entropy(12, 2, true);
    assert_mono_roundtrip(&gop, "mono-cabac-low-qp");
}

#[test]
fn mono_cabac_high_qp_self_roundtrip_bit_exact() {
    // Sparse residual: real mb_skip_flag runs through §9.3.3.1.1.1.
    let gop = encode_mono_gop_entropy(44, 2, true);
    assert_mono_roundtrip(&gop, "mono-cabac-high-qp");
}

#[test]
fn mono_cabac_gop_reference_decoder_byte_exact() {
    let gop = encode_mono_gop_entropy(26, 2, true);
    reference_decoder_check(&gop, "mono-cabac-gop-ref");
}

#[test]
fn mono_cabac_low_qp_reference_decoder_byte_exact() {
    let gop = encode_mono_gop_entropy(12, 2, true);
    reference_decoder_check(&gop, "mono-cabac-lowqp-ref");
}

// ---- B pictures at 4:0:0 (round-448 B leg) ----

fn encode_mono_ibp(qp: i32, cabac: bool, temporal: bool) -> MonoGop {
    let cfg = EncoderConfig {
        chroma_format_idc: 0,
        profile_idc: 100,
        cabac,
        qp,
        direct_temporal_mv_pred: temporal,
        max_num_ref_frames: 2,
        ..EncoderConfig::new(W as u32, H as u32)
    };
    let enc = Encoder::new(cfg);
    let planes: Vec<Vec<u8>> = (0..3).map(make_luma).collect();
    let mk = |k: usize| YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &planes[k],
        u: &[],
        v: &[],
    };
    let idr = if cabac {
        enc.encode_idr_cabac(&mk(0))
    } else {
        enc.encode_idr(&mk(0))
    };
    // Decode order IDR (poc 0), P (display 2, poc 4), B (display 1, poc 2).
    let p = if cabac {
        enc.encode_p_cabac(&mk(2), &EncodedFrameRef::from(&idr), 1, 4)
    } else {
        enc.encode_p(&mk(2), &EncodedFrameRef::from(&idr), 1, 4)
    };
    let b = if cabac {
        enc.encode_b_cabac(
            &mk(1),
            &EncodedFrameRef::from(&idr),
            &EncodedFrameRef::from(&p),
            1,
            2,
        )
    } else {
        enc.encode_b(
            &mk(1),
            &EncodedFrameRef::from(&idr),
            &EncodedFrameRef::from(&p),
            1,
            2,
        )
    };
    let mut annex_b = idr.annex_b.clone();
    annex_b.extend_from_slice(&p.annex_b);
    annex_b.extend_from_slice(&b.annex_b);
    // Display order: IDR, B, P.
    MonoGop {
        annex_b,
        recon: vec![idr.recon_y, b.recon_y, p.recon_y],
    }
}

#[test]
fn mono_b_cavlc_spatial_self_roundtrip_bit_exact() {
    let gop = encode_mono_ibp(26, false, false);
    assert_mono_roundtrip(&gop, "mono-b-cavlc-spatial");
}

#[test]
fn mono_b_cavlc_temporal_self_roundtrip_bit_exact() {
    let gop = encode_mono_ibp(26, false, true);
    assert_mono_roundtrip(&gop, "mono-b-cavlc-temporal");
}

#[test]
fn mono_b_cabac_spatial_self_roundtrip_bit_exact() {
    let gop = encode_mono_ibp(26, true, false);
    assert_mono_roundtrip(&gop, "mono-b-cabac-spatial");
}

#[test]
fn mono_b_cabac_temporal_self_roundtrip_bit_exact() {
    let gop = encode_mono_ibp(26, true, true);
    assert_mono_roundtrip(&gop, "mono-b-cabac-temporal");
}

#[test]
fn mono_b_cavlc_reference_decoder_byte_exact() {
    let gop = encode_mono_ibp(26, false, false);
    reference_decoder_check(&gop, "mono-b-cavlc-ref");
}

#[test]
fn mono_b_cabac_reference_decoder_byte_exact() {
    let gop = encode_mono_ibp(26, true, false);
    reference_decoder_check(&gop, "mono-b-cabac-ref");
}

#[test]
fn mono_b_temporal_reference_decoder_byte_exact() {
    let gop = encode_mono_ibp(26, false, true);
    reference_decoder_check(&gop, "mono-b-temporal-ref");
}

// ---- 8x8 transform at 4:0:0 (round-448) ----

fn encode_mono_gop_t8x8(qp: i32, n_p: usize, cabac: bool) -> MonoGop {
    let cfg = EncoderConfig {
        chroma_format_idc: 0,
        profile_idc: 100,
        cabac,
        qp,
        transform_8x8: true,
        ..EncoderConfig::new(W as u32, H as u32)
    };
    let enc = Encoder::new(cfg);
    let y0 = make_luma(0);
    let f0 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y0,
        u: &[],
        v: &[],
    };
    let idr = if cabac {
        enc.encode_idr_cabac(&f0)
    } else {
        enc.encode_idr(&f0)
    };
    let mut annex_b = idr.annex_b.clone();
    let mut recon = vec![idr.recon_y.clone()];
    let mut prev_idr = Some(idr);
    let mut prev_p: Option<oxideav_h264::encoder::EncodedP> = None;
    for k in 1..=n_p {
        let yk = make_luma(k);
        let fk = YuvFrame {
            width: W as u32,
            height: H as u32,
            y: &yk,
            u: &[],
            v: &[],
        };
        let encode = |prev_ref: &EncodedFrameRef<'_>| {
            if cabac {
                enc.encode_p_cabac(&fk, prev_ref, k as u32, 2 * k as u32)
            } else {
                enc.encode_p(&fk, prev_ref, k as u32, 2 * k as u32)
            }
        };
        let p = if let Some(prev) = prev_p.take() {
            encode(&EncodedFrameRef::from(&prev))
        } else {
            let idr = prev_idr.take().expect("idr present");
            encode(&EncodedFrameRef::from(&idr))
        };
        annex_b.extend_from_slice(&p.annex_b);
        recon.push(p.recon_y.clone());
        prev_p = Some(p);
    }
    MonoGop { annex_b, recon }
}

#[test]
fn mono_transform_8x8_cavlc_self_roundtrip_bit_exact() {
    // Low QP so the inter 8x8 trial genuinely wins on textured MBs.
    let gop = encode_mono_gop_t8x8(20, 2, false);
    assert_mono_roundtrip(&gop, "mono-t8x8-cavlc");
}

#[test]
fn mono_transform_8x8_cabac_self_roundtrip_bit_exact() {
    let gop = encode_mono_gop_t8x8(20, 2, true);
    assert_mono_roundtrip(&gop, "mono-t8x8-cabac");
}

#[test]
fn mono_transform_8x8_cabac_idr_codes_intra_8x8_mbs() {
    // Coverage pin: the CABAC 4:0:0 IDR must genuinely code Intra_8x8
    // macroblocks under transform_8x8 (not silently fall back to
    // I_16x16 everywhere) so the §7.3.5.3.3 blockCat-5 luma-only
    // residual path is exercised on the wire.
    let cfg = EncoderConfig {
        chroma_format_idc: 0,
        profile_idc: 100,
        cabac: true,
        qp: 30,
        transform_8x8: true,
        ..EncoderConfig::new(W as u32, H as u32)
    };
    let enc = Encoder::new(cfg);
    // Smooth low-frequency content: the 8x8 transform's coarser basis
    // wins the Lagrangian trial on gentle ramps, where 4x4 spends
    // extra coefficients re-describing the gradient per sub-block.
    let mut y0 = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let v = 40.0
                + 60.0 * ((i as f64) * 0.07).sin()
                + 50.0 * ((j as f64) * 0.05).cos()
                + (i as f64) * 0.3;
            y0[j * W + i] = v.clamp(16.0, 235.0) as u8;
        }
    }
    let f0 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y0,
        u: &[],
        v: &[],
    };
    let idr = enc.encode_idr_cabac(&f0);
    // The stream (whatever the per-MB picks) must still roundtrip.
    let gop = MonoGop {
        annex_b: idr.annex_b.clone(),
        recon: vec![idr.recon_y.clone()],
    };
    assert_mono_roundtrip(&gop, "mono-t8x8-i8x8-pin");
    reference_decoder_check(&gop, "mono-t8x8-i8x8-pin-ref");
    eprintln!("mono CABAC t8x8 IDR: {} Intra_8x8 MBs", idr.i8x8_mb_count);
    assert!(
        idr.i8x8_mb_count > 0,
        "expected at least one Intra_8x8 MB in the 4:0:0 CABAC IDR"
    );
}

#[test]
fn mono_transform_8x8_cavlc_reference_decoder_byte_exact() {
    let gop = encode_mono_gop_t8x8(20, 2, false);
    reference_decoder_check(&gop, "mono-t8x8-cavlc-ref");
}

#[test]
fn mono_transform_8x8_cabac_reference_decoder_byte_exact() {
    let gop = encode_mono_gop_t8x8(20, 2, true);
    reference_decoder_check(&gop, "mono-t8x8-cabac-ref");
}

// ---- Intra_4x4 (I_NxN) at 4:0:0 (round-448 CAVLC leg) ----

#[test]
fn mono_i4x4_idr_codes_i_nxn_and_roundtrips() {
    // Sharp diagonal structure at low QP: the per-block 9-mode
    // Intra_4x4 trial beats Intra_16x16 on edges, so the two-way RDO
    // genuinely codes I_NxN macroblocks (coverage pin below).
    let cfg = EncoderConfig {
        chroma_format_idc: 0,
        profile_idc: 100,
        qp: 16,
        ..EncoderConfig::new(W as u32, H as u32)
    };
    let enc = Encoder::new(cfg);
    let mut y0 = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            // Diagonal bands + a hard step every 8 px.
            let band = ((i + j) / 4) % 2;
            let step = if (i / 8) % 2 == 0 { 60 } else { 0 };
            y0[j * W + i] = (60 + band * 90 + step) as u8;
        }
    }
    let f0 = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y0,
        u: &[],
        v: &[],
    };
    let idr = enc.encode_idr(&f0);
    eprintln!("mono I_NxN IDR: {} Intra_4x4 MBs", idr.i4x4_mb_count);
    assert!(
        idr.i4x4_mb_count > 0,
        "expected at least one I_NxN (Intra_4x4) MB in the 4:0:0 CAVLC IDR"
    );
    let gop = MonoGop {
        annex_b: idr.annex_b.clone(),
        recon: vec![idr.recon_y.clone()],
    };
    assert_mono_roundtrip(&gop, "mono-i4x4");
    reference_decoder_check(&gop, "mono-i4x4-ref");
}

#[test]
fn mono_i4x4_gop_self_roundtrip_and_reference() {
    // The textured pan content at low QP mixes I_16x16 and I_NxN MBs
    // in one picture and chains P pictures on top.
    let gop = encode_mono_gop(16, 2);
    assert_mono_roundtrip(&gop, "mono-i4x4-gop");
    reference_decoder_check(&gop, "mono-i4x4-gop-ref");
}
