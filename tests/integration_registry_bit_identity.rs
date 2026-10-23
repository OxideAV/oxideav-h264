//! Round-475 — 8-bit 4:2:0 registry streams stay **byte-identical**
//! across the round's encoder changes (the §7.4.3 `idr_pic_id`
//! rotation, the deep-layout widening, the `avcC` option). Digests
//! were pinned from the pre-r475 encoder (0.1.8 + master) over the
//! option matrix below; a change here is a change to what every
//! existing consumer receives. Single-IDR streams (the HEIF `avc1`
//! item / `avcs` sequence shape with `gop >= frames`) match in full;
//! streams with several IDRs match on every access unit up to and
//! including the second IDR's predecessor (the second IDR carries
//! `idr_pic_id = 1` by design — see `integration_idr_pic_id`).

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, Rational, VideoFrame, VideoPlane,
};

const W: u32 = 80;
const H: u32 = 64;

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

/// FNV-1a 64 over the packet bytes.
fn fnv(d: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in d {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn encode(opts: &[(&str, &str)], frames: usize) -> Vec<Packet> {
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
    let mut enc = oxideav_h264::h264_encoder::make_encoder(&p).unwrap();
    let mut packets: Vec<Packet> = Vec::new();
    for n in 0..frames {
        enc.send_frame(&frame(n)).unwrap();
        loop {
            match enc.receive_packet() {
                Ok(p) => packets.push(p),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("{e}"),
            }
        }
    }
    enc.flush().unwrap();
    while let Ok(p) = enc.receive_packet() {
        packets.push(p);
    }
    packets
}

fn check(name: &str, opts: &[(&str, &str)], frames: usize, pins: &[(usize, u64)]) {
    let packets = encode(opts, frames);
    assert_eq!(packets.len(), pins.len(), "{name}: access-unit count");
    for (i, (pk, &(len, digest))) in packets.iter().zip(pins.iter()).enumerate() {
        assert_eq!(pk.data.len(), len, "{name}: AU {i} length");
        assert_eq!(
            fnv(&pk.data),
            digest,
            "{name}: AU {i} bytes differ from the pinned stream"
        );
    }
}

#[test]
fn cqp_cavlc_single_idr() {
    check(
        "cqp26_cavlc",
        &[("rc", "cqp"), ("qp", "26")],
        8,
        &[
            (760, 0x35a5a528bea68dc7),
            (389, 0xad5e3bccc027dc0a),
            (490, 0xeb987d439adb8fc0),
            (501, 0x7b88b5992e47de46),
            (482, 0x916f7c555c4e5fa3),
            (522, 0x27d0a5968ac2dfd6),
            (503, 0x816f159daa42b4f2),
            (506, 0xf8bded85dd1821c1),
        ],
    );
}

#[test]
fn cqp_cabac_single_idr() {
    check(
        "cqp30_cabac",
        &[("rc", "cqp"), ("qp", "30"), ("cabac", "true")],
        8,
        &[
            (498, 0xf307e374cbd227e7),
            (422, 0xe8476f447a0b14ee),
            (425, 0xd123c89590b8b45c),
            (478, 0x57ddd85862c4dea4),
            (455, 0x941e57b04dca83b1),
            (475, 0x49c56453dca3e151),
            (511, 0x44751ee9c582e4d2),
            (460, 0x73bfe7a2be3ef78d),
        ],
    );
}

#[test]
fn cqp_cabac_b_frames_single_idr() {
    check(
        "cqp28_cabac_b2",
        &[
            ("rc", "cqp"),
            ("qp", "28"),
            ("cabac", "true"),
            ("bframes", "2"),
        ],
        9,
        &[
            (589, 0x5de34e61ae9821e9),
            (700, 0x4047b01b6e0c65b2),
            (415, 0x7aef62746f3cf899),
            (458, 0xc00bd6643fe31337),
            (744, 0x014da436360b9aca),
            (419, 0x33ed9e40a3fb2f59),
            (540, 0x960b74c78409bd75),
            (767, 0xa50712939336d81d),
            (389, 0x239377133d786181),
        ],
    );
}

#[test]
fn cbr_cavlc_single_idr() {
    check(
        "cbr_cavlc",
        &[("rc", "cbr"), ("bitrate", "400000")],
        10,
        &[
            (1355, 0x2e7e32c79c4e78b7),
            (653, 0x050a908f58028a6d),
            (1009, 0xb77130bbff51a0df),
            (1226, 0x88defac6e55fc547),
            (1243, 0xcfcde3e2af7ccb28),
            (1298, 0xa7336b6defdb6bd1),
            (1392, 0x51b46195a33eecd7),
            (1326, 0x0e62d707c5037d53),
            (1373, 0xf25ef670b1cb226b),
            (1340, 0x116a415878da3551),
        ],
    );
}

#[test]
fn vbr_cabac_b_frames_single_idr() {
    check(
        "vbr_cabac_b1",
        &[
            ("rc", "vbr"),
            ("bitrate", "300000"),
            ("cabac", "true"),
            ("bframes", "1"),
        ],
        10,
        &[
            (1164, 0x6f1f70d59a12be72),
            (1227, 0x302519b74af62046),
            (527, 0x9395b0b40600316c),
            (1303, 0x7de3057d4072ea77),
            (596, 0x8c47aefc5edef33e),
            (1411, 0xfca9f1de1cab9aab),
            (588, 0xe48c63c7e03d010d),
            (1354, 0x5b8d618094a35f84),
            (607, 0x6fcbc0b4e3ad2de6),
            (1014, 0xc6cdeae303a61ac8),
        ],
    );
}

/// Multi-IDR streams: the first IDR access unit (and everything before
/// the second IDR) is byte-identical to the pre-r475 stream; the
/// second IDR onwards carries the rotated `idr_pic_id` (§7.4.3) and is
/// covered by `integration_idr_pic_id`.
#[test]
fn multi_idr_streams_match_up_to_the_second_idr() {
    let packets = encode(&[("rc", "cqp"), ("qp", "26"), ("gop", "1")], 4);
    assert_eq!(packets[0].data.len(), 760);
    assert_eq!(fnv(&packets[0].data), 0x35a5a528bea68dc7);
    // The pre-r475 second IDR (idr_pic_id = 0) — must now differ.
    assert_ne!(fnv(&packets[1].data), 0x79d375c0d7009506);

    let packets = encode(
        &[("rc", "cqp"), ("qp", "26"), ("cabac", "true"), ("gop", "2")],
        5,
    );
    assert_eq!(fnv(&packets[0].data), 0x0cb5e96a2e29b263);
    assert_eq!(fnv(&packets[1].data), 0xeb8d0bc3a1bde4aa);
    assert_ne!(fnv(&packets[2].data), 0x69922236751d45c7);
}
