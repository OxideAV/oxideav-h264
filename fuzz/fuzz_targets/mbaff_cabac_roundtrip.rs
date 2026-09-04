#![no_main]

//! Fuzz: the round-456 CABAC MBAFF encoder ↔ decoder round trip.
//!
//! The input bytes select the pair-decision policy, the QP, the
//! picture structure (I-only / P, intra fallback) and shape a small
//! interlaced source. The encoder must (1) never panic, (2) produce a
//! stream our decoder reconstructs **bit-exactly** to the encoder's
//! own reconstruction (every §9.3.3.1.1.x MBAFF context increment —
//! `mb_field_decoding_flag`, skip contexts under the §7.4.4 inference,
//! frame/field mvd / ref_idx scaling, cross-pair CBP / CBF probes, the
//! field residual contexts — is exercised by the content-driven
//! decisions), and (3) the decoder must stay panic-free on a
//! byte-flipped copy of that stream.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, Decoder, Frame, Packet, TimeBase};
use oxideav_h264::encoder::mbaff::PairMode;
use oxideav_h264::encoder::mbaff_cabac::{encode_mbaff_cabac_sequence, MbaffCabacConfig};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 48;
const H: usize = 64; // 3x4 MBs → 6 pairs

fn decode(stream: &[u8]) -> Vec<oxideav_core::VideoFrame> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream.to_vec()).with_pts(0);
    if dec.send_packet(&pkt).is_err() {
        return Vec::new();
    }
    let _ = dec.flush();
    let mut out = Vec::new();
    for _ in 0..8 {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => out.push(vf),
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    out
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let pair_mode = match data[0] & 3 {
        0 => PairMode::AllFrame,
        1 => PairMode::AllField,
        2 => PairMode::Checker,
        _ => PairMode::Adaptive,
    };
    let qp = 16 + (data[1] % 30) as i32;
    let p_frames = data[2] & 1 == 1;
    let intra_in_p = data[2] & 2 == 2;
    let n_frames = 1 + (data[3] % 3) as usize;
    let flip = u16::from_le_bytes([data[4], data[5]]) as usize;
    let seed = &data[6..];

    // Shape the source from the input bytes: a byte-driven gradient
    // plus per-field motion so the pair decisions vary.
    let mut frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::with_capacity(n_frames);
    for k in 0..n_frames {
        let mut y = vec![0u8; W * H];
        for row in 0..H {
            let shift = seed[(row / 2) % seed.len()] as usize + 3 * k * (row & 1);
            for col in 0..W {
                y[row * W + col] = seed[(col + shift + row * 7) % seed.len()].wrapping_add((row * 2) as u8);
            }
        }
        let cw = W / 2;
        let ch = H / 2;
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];
        for row in 0..ch {
            for col in 0..cw {
                u[row * cw + col] = seed[(col * 3 + row + k) % seed.len()];
                v[row * cw + col] = seed[(col + row * 5 + 2 * k) % seed.len()].wrapping_mul(3);
            }
        }
        frames.push((y, u, v));
    }
    let refs: Vec<(&[u8], &[u8], &[u8])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    let enc = encode_mbaff_cabac_sequence(
        &MbaffCabacConfig {
            width: W as u32,
            frame_height: H as u32,
            qp,
            pair_mode,
            p_frames,
            intra_in_p,
        },
        &refs,
    );

    // (2) bit-exact round trip through our decoder.
    let decoded = decode(&enc.annex_b);
    assert_eq!(decoded.len(), enc.recon_frames.len(), "frame count");
    for (vf, (ry, ru, rv)) in decoded.iter().zip(enc.recon_frames.iter()) {
        assert_eq!(vf.planes.len(), 3);
        for (plane, exp) in vf.planes.iter().zip([ry, ru, rv]) {
            assert!(
                plane.data.iter().zip(exp.iter()).all(|(a, b)| a == b),
                "CABAC MBAFF round trip diverged"
            );
        }
    }

    // (3) panic-freedom on a corrupted copy of the stream.
    let mut corrupted = enc.annex_b.clone();
    if !corrupted.is_empty() {
        let idx = flip % corrupted.len();
        corrupted[idx] ^= 1 << (data[7] & 7);
        let _ = decode(&corrupted);
    }
});
