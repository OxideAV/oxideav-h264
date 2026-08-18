//! Round-448 — **separate-colour-plane** (`separate_colour_plane_flag
//! = 1`, High 4:4:4 Predictive) encode + decode.
//!
//! §7.4.2.1.1: when the flag is set, ChromaArrayType is 0 and the
//! primary coded picture consists of three separately-coded monochrome
//! planes, each slice naming its plane via the §7.4.3
//! `colour_plane_id` u(2). §8.1: the decoding process is invoked
//! three times — "as if only a coded video sequence with monochrome
//! colour format with that particular value of colour_plane_id would
//! be present" — with the three outputs assigned to S_L / S_Cb / S_Cr.
//! The decoder mirrors that literally: three monochrome sub-decoders,
//! slice routing by colour_plane_id, plane-triple reassembly.
//!
//! Gates:
//!   * bit-exact self-roundtrip: decoder output == the per-plane
//!     encoder reconstructions, on IDR-only and IDR+P+P sequences,
//!     CAVLC and CABAC, QP 12 / 26 / 44;
//!   * three-plane output shape (yuv444p layout, full-resolution
//!     chroma);
//!   * §8.1 compositional identity: each plane of the SCP decode is
//!     byte-identical to decoding the SAME coded plane data wrapped
//!     as a standalone 4:0:0 monochrome stream (the two coded
//!     representations share every slice-layer bit below the SPS /
//!     slice-header framing);
//!   * black-box cross-check against a stock reference decoder binary
//!     when it accepts the stream (skipped, with a note, when the
//!     binary rejects separate-colour-plane streams).

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::scp::{encode_scp_sequence, ScpConfig, ScpEncoded};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 80;
const H: usize = 64;

/// Three distinct textured planes for display frame `k` — each plane
/// pans / drifts differently so a plane-routing mix-up cannot cancel
/// out.
fn make_planes(k: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    let mut cb = vec![0u8; W * H];
    let mut cr = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            let x = i + 2 * k;
            let base = 40 + ((x * 2 + j) % 160) as u32;
            let tex = ((x * 37 + j * 101 + (x ^ j) * 13) % 61) as u32;
            y[j * W + i] = (base + tex / 2 + (3 * k) as u32).min(235) as u8;
            let xb = i + k; // slower pan
            cb[j * W + i] =
                (60 + ((xb * 3 + j * 2) % 140) as u32 + ((xb * 17 + j * 31) % 23) as u32).min(235)
                    as u8;
            let xr = i + 3 * k; // faster pan
            cr[j * W + i] = (200u32
                .saturating_sub(((xr + 2 * j) % 150) as u32)
                .saturating_add(((xr * 7 + j * 11) % 19) as u32))
            .min(235) as u8;
        }
    }
    (y, cb, cr)
}

fn encode_scp(qp: i32, n_p: usize, cabac: bool) -> ScpEncoded {
    let frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..=n_p).map(make_planes).collect();
    let refs: Vec<(&[u8], &[u8], &[u8])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    encode_scp_sequence(
        &ScpConfig {
            width: W as u32,
            height: H as u32,
            qp,
            cabac,
            b_frame: false,
            direct_temporal: false,
            transform_8x8: false,
        },
        &refs,
    )
}

/// IDR + P + P with `transform_8x8_mode_flag = 1`: each plane's inter
/// MBs run the §8.6.4 8x8-vs-4x4 luma trial.
fn encode_scp_t8x8(qp: i32, cabac: bool) -> ScpEncoded {
    let frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..=2).map(make_planes).collect();
    let refs: Vec<(&[u8], &[u8], &[u8])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    encode_scp_sequence(
        &ScpConfig {
            width: W as u32,
            height: H as u32,
            qp,
            cabac,
            b_frame: false,
            direct_temporal: false,
            transform_8x8: true,
        },
        &refs,
    )
}

/// IDR-B-P mini-GOP: three display frames, the middle one coded as a
/// non-reference B whose two references are the SAME PLANE's IDR and
/// P reconstructions (§8.4.1.2.2 spatial or §8.4.1.2.3 temporal
/// direct inside each plane).
fn encode_scp_b(qp: i32, cabac: bool, temporal: bool) -> ScpEncoded {
    let frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..=2).map(make_planes).collect();
    let refs: Vec<(&[u8], &[u8], &[u8])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    encode_scp_sequence(
        &ScpConfig {
            width: W as u32,
            height: H as u32,
            qp,
            cabac,
            b_frame: true,
            direct_temporal: temporal,
            transform_8x8: false,
        },
        &refs,
    )
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

fn assert_scp_roundtrip(enc: &ScpEncoded, tag: &str) {
    let frames = decode_ours(&enc.annex_b);
    assert_eq!(frames.len(), enc.recon_frames.len(), "{tag}: frame count");
    for (i, (vf, exp)) in frames.iter().zip(enc.recon_frames.iter()).enumerate() {
        assert_eq!(vf.planes.len(), 3, "{tag}: frame {i} plane count");
        for (p, name) in ["Y", "Cb", "Cr"].iter().enumerate() {
            let plane = &vf.planes[p];
            assert_eq!(plane.stride, W, "{tag}: frame {i} {name} stride");
            let mismatches = plane
                .data
                .iter()
                .zip(exp[p].iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                mismatches,
                0,
                "{tag}: frame {i} plane {name}: {mismatches}/{} samples differ",
                exp[p].len(),
            );
        }
    }
}

#[test]
fn scp_cavlc_idr_self_roundtrip_bit_exact() {
    let enc = encode_scp(26, 0, false);
    assert_scp_roundtrip(&enc, "scp-cavlc-idr");
}

#[test]
fn scp_cavlc_idr_p_p_self_roundtrip_bit_exact() {
    let enc = encode_scp(26, 2, false);
    assert_scp_roundtrip(&enc, "scp-cavlc-gop");
}

#[test]
fn scp_cavlc_low_qp_self_roundtrip_bit_exact() {
    let enc = encode_scp(12, 2, false);
    assert_scp_roundtrip(&enc, "scp-cavlc-lowqp");
}

#[test]
fn scp_cavlc_high_qp_self_roundtrip_bit_exact() {
    // Sparse residual: real P_Skip runs inside every plane.
    let enc = encode_scp(44, 2, false);
    assert_scp_roundtrip(&enc, "scp-cavlc-highqp");
}

#[test]
fn scp_cabac_idr_self_roundtrip_bit_exact() {
    let enc = encode_scp(26, 0, true);
    assert_scp_roundtrip(&enc, "scp-cabac-idr");
}

#[test]
fn scp_cabac_idr_p_p_self_roundtrip_bit_exact() {
    let enc = encode_scp(26, 2, true);
    assert_scp_roundtrip(&enc, "scp-cabac-gop");
}

#[test]
fn scp_cabac_low_qp_self_roundtrip_bit_exact() {
    let enc = encode_scp(12, 2, true);
    assert_scp_roundtrip(&enc, "scp-cabac-lowqp");
}

/// §8.1 compositional identity: decoding the separate-colour-plane
/// stream must reproduce, plane for plane, the decode of each plane's
/// coded data as a standalone 4:0:0 monochrome stream. The per-plane
/// slice payloads are bit-identical between the two framings (only
/// the SPS chroma signalling and the `colour_plane_id` header field
/// differ), so this pins the routing layer against the already-gated
/// monochrome decode path.
#[test]
fn scp_planes_match_standalone_monochrome_decodes() {
    use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};

    let enc = encode_scp(26, 2, false);
    let scp_frames = decode_ours(&enc.annex_b);
    assert_eq!(scp_frames.len(), 3);

    for plane in 0..3usize {
        // Re-encode this plane's source as a standalone monochrome
        // stream with the same QP / GOP shape…
        let mono = Encoder::new(EncoderConfig {
            chroma_format_idc: 0,
            profile_idc: 100,
            qp: 26,
            ..EncoderConfig::new(W as u32, H as u32)
        });
        let sources: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..=2).map(make_planes).collect();
        let src = |k: usize| -> &[u8] {
            match plane {
                0 => &sources[k].0,
                1 => &sources[k].1,
                _ => &sources[k].2,
            }
        };
        let f0 = YuvFrame {
            width: W as u32,
            height: H as u32,
            y: src(0),
            u: &[],
            v: &[],
        };
        let idr = mono.encode_idr(&f0);
        let mut stream = idr.annex_b.clone();
        let f1 = YuvFrame {
            width: W as u32,
            height: H as u32,
            y: src(1),
            u: &[],
            v: &[],
        };
        let p1 = mono.encode_p(&f1, &EncodedFrameRef::from(&idr), 1, 2);
        stream.extend_from_slice(&p1.annex_b);
        let f2 = YuvFrame {
            width: W as u32,
            height: H as u32,
            y: src(2),
            u: &[],
            v: &[],
        };
        let p2 = mono.encode_p(&f2, &EncodedFrameRef::from(&p1), 2, 4);
        stream.extend_from_slice(&p2.annex_b);

        // …decode it, and compare frame-by-frame against the matching
        // plane of the separate-colour-plane decode.
        let mono_frames = decode_ours(&stream);
        assert_eq!(mono_frames.len(), 3, "plane {plane}: mono frame count");
        for (i, mf) in mono_frames.iter().enumerate() {
            assert_eq!(mf.planes.len(), 1);
            let scp_plane = &scp_frames[i].planes[plane];
            let mismatches = mf.planes[0]
                .data
                .iter()
                .zip(scp_plane.data.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                mismatches, 0,
                "plane {plane} frame {i}: SCP decode differs from standalone monochrome decode",
            );
        }
    }
}

/// Black-box cross-check. The stock reference decoder binary is probed
/// with the stream; if it decodes (some builds reject
/// separate-colour-plane streams), its yuv444p output must match our
/// three-plane reconstruction byte-exactly.
fn reference_decoder_check(enc: &ScpEncoded, tag: &str) {
    let refbin = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !refbin.exists() {
        eprintln!("skip {tag}: reference decoder binary not present");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h264-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bs = dir.join("input.h264");
    let out = dir.join("out.yuv");
    std::fs::write(&bs, &enc.annex_b).unwrap();
    let status = std::process::Command::new(refbin)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&bs)
        .args(["-f", "rawvideo"])
        .arg(&out)
        .status()
        .expect("spawn reference decoder");
    if !status.success() {
        eprintln!("note {tag}: reference decoder rejected the separate-colour-plane stream; black-box gate skipped");
        if std::env::var("OXIDEAV_KEEP_STREAM").is_err() {
            let _ = std::fs::remove_dir_all(&dir);
        } else {
            eprintln!("kept stream at {}", bs.display());
        }
        return;
    }
    let raw = std::fs::read(&out).unwrap();
    let frame_bytes = W * H * 3;
    assert_eq!(
        raw.len(),
        frame_bytes * enc.recon_frames.len(),
        "{tag}: reference decoder output size (expected yuv444p frames)",
    );
    for (i, exp) in enc.recon_frames.iter().enumerate() {
        for (p, name) in ["Y", "Cb", "Cr"].iter().enumerate() {
            let got = &raw[i * frame_bytes + p * W * H..i * frame_bytes + (p + 1) * W * H];
            let mismatches = got
                .iter()
                .zip(exp[p].iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                mismatches,
                0,
                "{tag}: frame {i} plane {name}: {mismatches}/{} samples differ vs reference decoder",
                exp[p].len(),
            );
        }
    }
    if std::env::var("OXIDEAV_KEEP_STREAM").is_err() {
        let _ = std::fs::remove_dir_all(&dir);
    } else {
        eprintln!("kept stream at {}", bs.display());
    }
}

#[test]
fn scp_cavlc_gop_reference_decoder_byte_exact() {
    let enc = encode_scp(26, 2, false);
    reference_decoder_check(&enc, "scp-cavlc-ref");
}

#[test]
fn scp_cabac_gop_reference_decoder_byte_exact() {
    let enc = encode_scp(26, 2, true);
    reference_decoder_check(&enc, "scp-cabac-ref");
}

// ---- B access units (round-448 B leg) ----

#[test]
fn scp_cavlc_b_spatial_self_roundtrip_bit_exact() {
    let enc = encode_scp_b(26, false, false);
    assert_scp_roundtrip(&enc, "scp-cavlc-b-spatial");
}

#[test]
fn scp_cavlc_b_temporal_self_roundtrip_bit_exact() {
    let enc = encode_scp_b(26, false, true);
    assert_scp_roundtrip(&enc, "scp-cavlc-b-temporal");
}

#[test]
fn scp_cabac_b_spatial_self_roundtrip_bit_exact() {
    let enc = encode_scp_b(26, true, false);
    assert_scp_roundtrip(&enc, "scp-cabac-b-spatial");
}

#[test]
fn scp_cabac_b_temporal_self_roundtrip_bit_exact() {
    let enc = encode_scp_b(26, true, true);
    assert_scp_roundtrip(&enc, "scp-cabac-b-temporal");
}

// ---- 8x8 transform at ChromaArrayType 0 (round-448) ----

#[test]
fn scp_cavlc_transform_8x8_self_roundtrip_bit_exact() {
    let enc = encode_scp_t8x8(20, false);
    assert_scp_roundtrip(&enc, "scp-cavlc-t8x8");
}

#[test]
fn scp_cabac_transform_8x8_self_roundtrip_bit_exact() {
    let enc = encode_scp_t8x8(20, true);
    assert_scp_roundtrip(&enc, "scp-cabac-t8x8");
}

/// Anti-OOM guard: a NON-conforming stream that carries only
/// `colour_plane_id = 0` slices (§7.4.1.2 requires all three planes
/// per access unit) must neither panic nor grow the plane-pairing
/// queue without bound — the decoder drops unpairable plane pictures
/// past a cap and counts them as decode errors.
#[test]
fn scp_plane0_only_stream_is_bounded_and_errors() {
    use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};

    const SW: usize = 32;
    const SH: usize = 32;
    let enc = Encoder::new(EncoderConfig {
        chroma_format_idc: 0,
        profile_idc: 244,
        colour_plane_id: Some(0),
        qp: 40,
        ..EncoderConfig::new(SW as u32, SH as u32)
    });
    let y: Vec<u8> = (0..SW * SH).map(|i| (i % 251) as u8).collect();
    let f = YuvFrame {
        width: SW as u32,
        height: SH as u32,
        y: &y,
        u: &[],
        v: &[],
    };
    let idr = enc.encode_idr(&f);
    let mut stream = idr.annex_b.clone();
    let mut prev: Option<oxideav_h264::encoder::EncodedP> = None;
    for k in 1..90u32 {
        let p = if let Some(pp) = prev.take() {
            enc.encode_p(&f, &EncodedFrameRef::from(&pp), k, 2 * k)
        } else {
            enc.encode_p(&f, &EncodedFrameRef::from(&idr), k, 2 * k)
        };
        stream.extend_from_slice(&p.annex_b);
        prev = Some(p);
    }

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    // No complete (Y, Cb, Cr) triple ever forms → no video frames.
    assert!(!matches!(dec.receive_frame(), Ok(Frame::Video(_))));
    // The overflow past the pairing cap was surfaced as decode errors.
    assert!(
        dec.decode_error_count() > 0,
        "expected dropped unpairable plane pictures to be counted"
    );
}
