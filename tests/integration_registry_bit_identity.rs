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
//!
//! Round 490 re-pinned the P / B access units of the CABAC streams: the
//! CABAC P coder gained P_8x8 (four vectors per macroblock), a mode
//! decision this content takes on many macroblocks — the seven P
//! pictures of `cqp30_cabac` went from 3226 to 2751 bytes. IDR access
//! units and every CAVLC stream are unchanged.

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
    // `H264_PRINT_PINS=1 cargo test --test integration_registry_bit_identity
    // -- --nocapture` lists the current streams in pin form.
    if std::env::var_os("H264_PRINT_PINS").is_some() {
        for pk in &packets {
            eprintln!("{name}: ({}, {:#018x}),", pk.data.len(), fnv(&pk.data));
        }
    }
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
            (337, 0x01f6474721305042),
            (354, 0x23f3b3c2fdaac04c),
            (409, 0x0c35f8ce22b2041a),
            (368, 0x8a3885d0f5a46306),
            (402, 0x29f85f551893eef9),
            (411, 0x8ab1c6fa02dff555),
            (470, 0x7155cce93a3ab05d),
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
            (551, 0x98c3a3d8617df17a),
            (412, 0x0961454709ec6b21),
            (432, 0x945786c2b82defab),
            (604, 0x046740bed1751367),
            (413, 0x2fe9642991d9dcb3),
            (480, 0x9c07f677798e21b1),
            (617, 0xa640d7ce03ae725b),
            (366, 0x0b3877da01698a31),
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
            (933, 0xf7b2e5eed5334458),
            (494, 0x2cee6fe9bd26c8b3),
            (1250, 0x1fa79712bd5e1d25),
            (651, 0x247dab31a4faffda),
            (1206, 0xfe71288158456163),
            (657, 0xeab11d762ca1a1c9),
            (1279, 0x4aabb51f8735b784),
            (739, 0x59f94ebea1549e10),
            (954, 0x1ffecf0a339bbe96),
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
    if std::env::var_os("H264_PRINT_PINS").is_some() {
        eprintln!("multi_idr cabac AU 1: {:#018x}", fnv(&packets[1].data));
    }
    assert_eq!(fnv(&packets[0].data), 0x0cb5e96a2e29b263);
    assert_eq!(fnv(&packets[1].data), 0x06cc464260177b5b);
    assert_ne!(fnv(&packets[2].data), 0x69922236751d45c7);
}
