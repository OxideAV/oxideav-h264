//! Round-475 — §7.4.3 `idr_pic_id` on the session / registry encoder.
//!
//! "When two consecutive access units in decoding order are both IDR
//! access units, the value of idr_pic_id in the slices of the first
//! such IDR access unit shall differ from the idr_pic_id in the second
//! such IDR access unit. The value of idr_pic_id shall be in the range
//! of 0 to 65535, inclusive." Before this round every IDR carried
//! `idr_pic_id = 0`, so an all-IDR stream (`gop=1` — the HEIF `avcs`
//! image-sequence shape) read, per the §7.4.1.2.4 first-VCL-NAL test,
//! as redundant slices of ONE picture. These tests parse the produced
//! streams with the crate's own slice-header parser and assert the
//! rule over every GOP shape the session produces, then confirm the
//! decoder (ours, and a black-box reference decoder when present)
//! outputs one picture per IDR access unit.

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, Rational, VideoFrame, VideoPlane,
};
use oxideav_h264::nal::{AnnexBSplitter, NalUnitType, parse_nal_unit};
use oxideav_h264::pps::Pps;
use oxideav_h264::slice_header::SliceHeader;
use oxideav_h264::sps::Sps;

const W: u32 = 64;
const H: u32 = 48;

fn frame(n: usize) -> Frame {
    let (w, h) = (W as usize, H as usize);
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; w * h / 4];
    let mut v = vec![0u8; w * h / 4];
    for j in 0..h {
        for i in 0..w {
            let m = if (i / 8 + j / 8 + n).is_multiple_of(3) {
                60
            } else {
                0
            };
            y[j * w + i] = (30 + m + ((i * 3 + j * 2 + n * 7) % 150)) as u8;
        }
    }
    for j in 0..h / 2 {
        for i in 0..w / 2 {
            u[j * (w / 2) + i] = (100 + ((i + n) % 50)) as u8;
            v[j * (w / 2) + i] = (130 + ((j + n * 2) % 40)) as u8;
        }
    }
    Frame::Video(VideoFrame {
        pts: Some(n as i64),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: w / 2,
                data: u,
            },
            VideoPlane {
                stride: w / 2,
                data: v,
            },
        ],
    })
}

fn params(opts: &[(&str, &str)]) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("h264"));
    p.width = Some(W);
    p.height = Some(H);
    p.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
    p.frame_rate = Some(Rational::new(30, 1));
    let mut o = oxideav_core::CodecOptions::new();
    for (k, v) in opts {
        o = o.set(*k, *v);
    }
    p.options = o;
    p
}

fn encode(opts: &[(&str, &str)], frames: usize) -> Vec<Packet> {
    let mut enc = oxideav_h264::h264_encoder::make_encoder(&params(opts)).expect("encoder");
    let mut packets = Vec::new();
    for n in 0..frames {
        enc.send_frame(&frame(n)).expect("send");
        loop {
            match enc.receive_packet() {
                Ok(p) => packets.push(p),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("receive_packet: {e}"),
            }
        }
    }
    enc.flush().expect("flush");
    while let Ok(p) = enc.receive_packet() {
        packets.push(p);
    }
    packets
}

/// Per access unit: `Some(ids)` with one entry per IDR slice NAL, or
/// `None` for a non-IDR access unit. Parsed with the crate's §7.3.3
/// slice-header parser against the in-band SPS / PPS.
fn idr_ids_per_au(packets: &[Packet]) -> Vec<Option<Vec<u32>>> {
    let mut sps: Option<Sps> = None;
    let mut pps: Option<Pps> = None;
    let mut out = Vec::new();
    for p in packets {
        let mut ids: Option<Vec<u32>> = None;
        for nal in AnnexBSplitter::new(&p.data) {
            let parsed = parse_nal_unit(nal).expect("nal");
            match parsed.header.nal_unit_type {
                NalUnitType::Sps => sps = Some(Sps::parse(&parsed.rbsp).expect("sps")),
                NalUnitType::Pps => {
                    let cf = sps.as_ref().map_or(1, |s| s.chroma_format_idc);
                    pps = Some(Pps::parse_with_chroma_format(&parsed.rbsp, cf).expect("pps"));
                }
                NalUnitType::SliceIdr => {
                    let h = SliceHeader::parse(
                        &parsed.rbsp,
                        sps.as_ref().expect("sps before slice"),
                        pps.as_ref().expect("pps before slice"),
                        &parsed.header,
                    )
                    .expect("idr slice header");
                    assert!(h.idr_pic_id <= 65535, "§7.4.3 range");
                    ids.get_or_insert_with(Vec::new).push(h.idr_pic_id);
                }
                _ => {}
            }
        }
        out.push(ids);
    }
    out
}

fn assert_rule(packets: &[Packet], expect_idr_count: usize) -> Vec<u32> {
    let per_au = idr_ids_per_au(packets);
    let mut prev_idr: Option<u32> = None;
    let mut ids = Vec::new();
    for (i, au) in per_au.iter().enumerate() {
        match au {
            Some(slice_ids) => {
                // All slices of one IDR picture carry the same value.
                assert!(
                    slice_ids.windows(2).all(|w| w[0] == w[1]),
                    "AU {i}: idr_pic_id differs between slices {slice_ids:?}"
                );
                let id = slice_ids[0];
                if let Some(p) = prev_idr {
                    assert_ne!(
                        p, id,
                        "AU {i}: consecutive IDR access units share idr_pic_id {id} (§7.4.3)"
                    );
                }
                assert!(
                    packets[i].flags.keyframe,
                    "AU {i}: IDR AU not flagged keyframe"
                );
                prev_idr = Some(id);
                ids.push(id);
            }
            None => {
                assert!(
                    !packets[i].flags.keyframe,
                    "AU {i}: non-IDR AU flagged keyframe"
                );
                prev_idr = None;
            }
        }
    }
    assert_eq!(ids.len(), expect_idr_count, "IDR access-unit count");
    ids
}

fn decode_ours(packets: &[Packet]) -> Vec<VideoFrame> {
    let mut dec = oxideav_h264::h264_decoder::make_decoder(&params(&[])).expect("decoder");
    let mut out = Vec::new();
    for p in packets {
        dec.send_packet(p).expect("send_packet");
        while let Ok(Frame::Video(vf)) = dec.receive_frame() {
            out.push(vf);
        }
    }
    dec.flush().expect("flush");
    while let Ok(Frame::Video(vf)) = dec.receive_frame() {
        out.push(vf);
    }
    out
}

/// `gop=1`: every picture an IDR — the session counter runs 0, 1, 2, …
#[test]
fn all_idr_stream_rotates_idr_pic_id_cavlc() {
    let packets = encode(&[("rc", "cqp"), ("qp", "26"), ("gop", "1")], 6);
    assert_eq!(packets.len(), 6);
    let ids = assert_rule(&packets, 6);
    assert_eq!(ids, vec![0, 1, 2, 3, 4, 5]);
    // The symptom the rule prevents: six IDR pictures decode as six
    // pictures, not as redundant slices of one.
    let frames = decode_ours(&packets);
    assert_eq!(frames.len(), 6, "one output picture per IDR access unit");
}

#[test]
fn all_idr_stream_rotates_idr_pic_id_cabac() {
    let packets = encode(
        &[("rc", "cqp"), ("qp", "30"), ("gop", "1"), ("cabac", "true")],
        5,
    );
    let ids = assert_rule(&packets, 5);
    assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    assert_eq!(decode_ours(&packets).len(), 5);
}

/// Mixed IDR / P (`gop=3`): IDRs at 0, 3, 6 carry 0, 1, 2; P AUs
/// have no idr_pic_id.
#[test]
fn mixed_idr_p_stream_counts_per_idr() {
    let packets = encode(&[("rc", "cqp"), ("qp", "28"), ("gop", "3")], 8);
    let per_au = idr_ids_per_au(&packets);
    let shape: Vec<bool> = per_au.iter().map(|a| a.is_some()).collect();
    assert_eq!(
        shape,
        vec![true, false, false, true, false, false, true, false]
    );
    let ids = assert_rule(&packets, 3);
    assert_eq!(ids, vec![0, 1, 2]);
    assert_eq!(decode_ours(&packets).len(), 8);
}

/// B-picture mini-GOPs truncated before each IDR (`bframes=2`,
/// `gop=4`): decode order `I P B B I …` — the IDRs still count 0, 1.
#[test]
fn b_frame_minigops_keep_the_counter() {
    let packets = encode(
        &[
            ("rc", "cqp"),
            ("qp", "28"),
            ("gop", "4"),
            ("bframes", "2"),
            ("cabac", "true"),
        ],
        8,
    );
    let ids = assert_rule(&packets, 2);
    assert_eq!(ids, vec![0, 1]);
    assert_eq!(decode_ours(&packets).len(), 8);
}

/// Rate-controlled (`cbr`, `gop=2`): the VBV retry loop re-encodes the
/// same IDR picture at a higher QP — the id is fixed per access unit
/// and still advances per IDR.
#[test]
fn rate_controlled_idrs_advance_once_per_access_unit() {
    let packets = encode(&[("rc", "cbr"), ("bitrate", "60000"), ("gop", "2")], 6);
    let ids = assert_rule(&packets, 3);
    assert_eq!(ids, vec![0, 1, 2]);
    assert_eq!(decode_ours(&packets).len(), 6);
}

/// The stateless `Encoder` exposes the value it will write; the
/// session advances it modulo 65536 (§7.4.3 range).
#[test]
fn session_counter_wraps_at_65536() {
    use oxideav_h264::encoder::session::{EncoderSession, SessionConfig};
    use oxideav_h264::encoder::{Encoder, EncoderConfig};
    let e = Encoder::new(EncoderConfig::new(16, 16)).with_idr_pic_id(65535);
    assert_eq!(e.idr_pic_id(), 65535);
    let mut s = EncoderSession::new(SessionConfig::constant_qp(16, 16, 26));
    assert_eq!(s.next_idr_pic_id(), 0);
    let y = vec![128u8; 256];
    let c = vec![128u8; 64];
    s.encode_frame(&y, &c, &c);
    assert_eq!(s.next_idr_pic_id(), 1);
}

#[test]
#[should_panic(expected = "idr_pic_id out of 0..=65535")]
fn encoder_rejects_idr_pic_id_past_the_range() {
    use oxideav_h264::encoder::{Encoder, EncoderConfig};
    let _ = Encoder::new(EncoderConfig::new(16, 16)).with_idr_pic_id(65536);
}

/// Black box: a reference decoder run as an opaque process outputs the
/// same number of pictures from the all-IDR stream, byte-identical to
/// ours.
#[test]
fn all_idr_stream_black_box_matches_ours() {
    let refbin = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !refbin.exists() {
        eprintln!("skip: reference decoder binary not present");
        return;
    }
    let packets = encode(&[("rc", "cqp"), ("qp", "26"), ("gop", "1")], 5);
    let ours = decode_ours(&packets);
    assert_eq!(ours.len(), 5);
    let mut annex_b = Vec::new();
    for p in &packets {
        annex_b.extend_from_slice(&p.data);
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h264-idrid-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bs = dir.join("in.h264");
    let out = dir.join("out.yuv");
    std::fs::write(&bs, &annex_b).unwrap();
    let status = std::process::Command::new(refbin)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&bs)
        .args(["-f", "rawvideo"])
        .arg(&out)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success());
    let raw = std::fs::read(&out).unwrap();
    let fb = (W * H + 2 * (W / 2) * (H / 2)) as usize;
    assert_eq!(raw.len(), fb * 5, "reference decoder picture count");
    for (i, vf) in ours.iter().enumerate() {
        let mut mine = Vec::with_capacity(fb);
        for pl in vf.image_planes() {
            mine.extend_from_slice(&pl.data);
        }
        assert_eq!(mine, &raw[i * fb..(i + 1) * fb], "picture {i} differs");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
