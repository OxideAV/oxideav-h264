//! Round-453 — §D.1.8 / §D.2.8 `recovery_point` SEI emission
//! (`SessionConfig::recovery_point_sei`).
//!
//! Every IDR access unit of the session leads with an SEI NAL carrying
//! payloadType 6 (`recovery_frame_cnt = 0`, `exact_match_flag = 1`,
//! `broken_link_flag = 0`, `changing_slice_group_idc = 0`); non-IDR
//! access units carry no SEI. The message round-trips through the
//! decoder's §7.3.2.3 SEI envelope + §D.1.8 payload parsers, and the
//! stream decodes identically with and without the SEI in both our
//! decoder and the black-box reference decoder (SEI is informative).

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::session::{EncoderSession, SessionConfig};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264::nal::{parse_nal_unit, AnnexBSplitter, NalUnitType};
use oxideav_h264::non_vcl::parse_sei_rbsp;
use oxideav_h264::sei::parse_recovery_point;

const W: usize = 64;
const H: usize = 48;

fn frame(k: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let y: Vec<u8> = (0..W * H)
        .map(|i| (((i % W) * 3 + (i / W) * 5 + k * 9) % 200 + 20) as u8)
        .collect();
    let u = vec![100u8; W * H / 4];
    let v = vec![150u8; W * H / 4];
    (y, u, v)
}

fn encode(recovery: bool, n: usize) -> (Vec<u8>, Vec<bool>) {
    let mut cfg = SessionConfig::constant_qp(W as u32, H as u32, 28);
    cfg.gop_length = 3;
    cfg.recovery_point_sei = recovery;
    let mut s = EncoderSession::new(cfg);
    let mut out = Vec::new();
    let mut idr_flags = Vec::new();
    for k in 0..n {
        let (y, u, v) = frame(k);
        let f = s.encode_frame(&y, &u, &v);
        idr_flags.push(f.is_idr);
        out.extend_from_slice(&f.annex_b);
    }
    (out, idr_flags)
}

fn decode_ours(annex_b: &[u8]) -> Vec<Vec<u8>> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), annex_b.to_vec()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => {
                let mut buf = Vec::new();
                for p in &vf.planes {
                    buf.extend_from_slice(&p.data);
                }
                frames.push(buf);
            }
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    frames
}

/// Per access unit: the SEI RBSPs that precede its first VCL NAL.
fn seis_per_au(stream: &[u8]) -> Vec<Vec<Vec<u8>>> {
    let mut aus: Vec<Vec<Vec<u8>>> = Vec::new();
    let mut pending: Vec<Vec<u8>> = Vec::new();
    for nal in AnnexBSplitter::new(stream) {
        let nu = parse_nal_unit(nal).expect("nal parses");
        match nu.header.nal_unit_type {
            NalUnitType::Sei => pending.push(nu.rbsp.into_owned()),
            NalUnitType::SliceIdr | NalUnitType::SliceNonIdr => {
                aus.push(std::mem::take(&mut pending));
            }
            _ => {}
        }
    }
    aus
}

#[test]
fn recovery_point_sei_leads_every_idr_and_round_trips() {
    let (stream, idr_flags) = encode(true, 7);
    let aus = seis_per_au(&stream);
    assert_eq!(aus.len(), 7);
    assert_eq!(
        idr_flags,
        vec![true, false, false, true, false, false, true]
    );
    for (k, (seis, &is_idr)) in aus.iter().zip(idr_flags.iter()).enumerate() {
        if is_idr {
            assert_eq!(seis.len(), 1, "AU {k}: one SEI NAL before the IDR slice");
            let msgs = parse_sei_rbsp(&seis[0]).expect("sei envelope");
            assert_eq!(msgs.len(), 1);
            assert_eq!(msgs[0].payload_type, 6, "AU {k}: recovery_point");
            let rp = parse_recovery_point(&msgs[0].payload).expect("recovery_point parses");
            assert_eq!(rp.recovery_frame_cnt, 0);
            assert!(rp.exact_match_flag);
            assert!(!rp.broken_link_flag);
            assert_eq!(rp.changing_slice_group_idc, 0);
        } else {
            assert!(seis.is_empty(), "AU {k}: non-IDR carries no SEI");
        }
    }
}

#[test]
fn recovery_point_sei_is_transparent_to_decoding() {
    let (with, _) = encode(true, 6);
    let (without, _) = encode(false, 6);
    assert!(with.len() > without.len());
    let a = decode_ours(&with);
    let b = decode_ours(&without);
    assert_eq!(a.len(), 6);
    assert_eq!(a, b, "SEI must not change the decoded pictures");

    // Black-box reference decoder: same pictures from the SEI stream.
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h264-rp-sei-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bs = dir.join("input.h264");
    let out = dir.join("out.yuv");
    std::fs::write(&bs, &with).unwrap();
    let status = std::process::Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&bs)
        .args(["-f", "rawvideo"])
        .arg(&out)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed");
    let yuv = std::fs::read(&out).unwrap();
    let fb = W * H * 3 / 2;
    assert_eq!(yuv.len(), fb * 6);
    for (i, ours) in a.iter().enumerate() {
        assert_eq!(
            &yuv[i * fb..(i + 1) * fb],
            ours.as_slice(),
            "frame {i} vs reference decoder"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
