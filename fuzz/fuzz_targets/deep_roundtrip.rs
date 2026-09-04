#![no_main]

//! Fuzz: the round-456 high-bit-depth encoder ↔ decoder round trip.
//!
//! Input bytes pick the bit depth (8..=14), chroma format, QP (down to
//! `−QpBdOffsetY`), lossless mode and shape a `u16` source. The
//! encoder must never panic, our decoder must reconstruct the stream
//! **bit-exactly** to the encoder's recon, and in lossless mode the
//! reconstruction must equal the source sample-for-sample. A
//! byte-flipped copy of the stream must decode panic-free.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, Decoder, Frame, Packet, TimeBase};
use oxideav_h264::encoder::deep::{encode_deep_sequence, DeepConfig};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 32;
const H: usize = 32;

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
    let bd = 8 + (data[0] % 7) as u32; // 8..=14
    let cf = 1 + (data[1] % 3) as u32;
    let lossless = data[2] & 1 == 1;
    let off_y = 6 * (bd as i32 - 8);
    let qp = if lossless {
        -off_y
    } else {
        -off_y + (data[3] as i32 % (52 + off_y))
    };
    let p_frames = data[2] & 2 == 2;
    let deblock = data[2] & 4 == 4;
    let n_frames = 1 + (data[4] % 2) as usize;
    let flip = u16::from_le_bytes([data[5], data[6]]) as usize;
    let seed = &data[7..];
    let max = (1u32 << bd) - 1;
    let (cw, ch) = match cf {
        1 => (W / 2, H / 2),
        2 => (W / 2, H),
        _ => (W, H),
    };
    let sample = |i: usize| -> u16 {
        let b = seed[i % seed.len()] as u32;
        ((b * (max + 1)) / 256).min(max) as u16
    };
    let mut frames = Vec::with_capacity(n_frames);
    for k in 0..n_frames {
        let y: Vec<u16> = (0..W * H).map(|i| sample(i + 3 * k + (i / W) * 5)).collect();
        let u: Vec<u16> = (0..cw * ch).map(|i| sample(i * 3 + k)).collect();
        let v: Vec<u16> = (0..cw * ch).map(|i| sample(i * 7 + 2 * k)).collect();
        frames.push((y, u, v));
    }
    let refs: Vec<(&[u16], &[u16], &[u16])> = frames
        .iter()
        .map(|(y, u, v)| (y.as_slice(), u.as_slice(), v.as_slice()))
        .collect();
    let enc = encode_deep_sequence(
        &DeepConfig {
            width: W as u32,
            height: H as u32,
            bit_depth_luma: bd,
            bit_depth_chroma: bd,
            chroma_format_idc: cf,
            qp,
            p_frames,
            intra_in_p: true,
            deblock,
            lossless,
            lossless_interop: data[2] & 8 == 8,
            cabac: data[2] & 16 == 16,
        },
        &refs,
    );
    if lossless {
        for (r, s) in enc.recon_frames.iter().zip(frames.iter()) {
            assert!(r.0 == s.0 && r.1 == s.1 && r.2 == s.2, "lossless recon != source");
        }
    }
    let decoded = decode(&enc.annex_b);
    assert_eq!(decoded.len(), enc.recon_frames.len(), "frame count");
    for (vf, (ry, ru, rv)) in decoded.iter().zip(enc.recon_frames.iter()) {
        assert_eq!(vf.planes.len(), 3);
        for (plane, exp) in vf.planes.iter().zip([ry, ru, rv]) {
            let ok = if bd == 8 {
                plane.data.iter().zip(exp.iter()).all(|(&a, &b)| a as u16 == b)
            } else {
                let bytes: Vec<u8> = exp.iter().flat_map(|s| s.to_le_bytes()).collect();
                plane.data.iter().zip(bytes.iter()).all(|(a, b)| a == b)
            };
            assert!(ok, "deep round trip diverged");
        }
    }
    let mut corrupted = enc.annex_b.clone();
    if !corrupted.is_empty() {
        let idx = flip % corrupted.len();
        corrupted[idx] ^= 1 << (data[7] & 7);
        let _ = decode(&corrupted);
    }
});
