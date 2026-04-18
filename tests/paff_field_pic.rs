//! PAFF (Picture-Adaptive Frame/Field) I-slice decode — §7.3.3 / §7.4.3.
//!
//! Drives the encoder in its new field-picture mode (each encoded packet
//! represents a single top or bottom field at half the output frame's
//! height) and verifies the decoder accepts the resulting bitstream,
//! emits the decoded field at the expected half-height dimensions, and
//! reconstructs the input samples within the QP-appropriate PSNR bound.
//!
//! Out of scope for this test (the crate rejects these with
//! `Error::Unsupported`): PAFF P / B, MBAFF + PAFF mixed within a coded
//! video sequence, and PAFF at 10-bit or alternate chroma formats.

use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{frame::VideoPlane, CodecId, Frame, Packet, PixelFormat, TimeBase, VideoFrame};
use oxideav_h264::decoder::H264Decoder;
use oxideav_h264::encoder::{H264Encoder, H264EncoderOptions};
use oxideav_h264::nal::{split_annex_b, NalHeader, NalUnitType};
use oxideav_h264::slice::parse_slice_header;

/// Build a diagonal-gradient YUV420P field. Widths / heights are in
/// field-sample units (not full-frame units).
fn make_field(w: u32, h: u32, bias: u8) -> VideoFrame {
    let cw = w / 2;
    let ch = h / 2;
    let mut y = vec![0u8; (w * h) as usize];
    let mut cb = vec![0u8; (cw * ch) as usize];
    let mut cr = vec![0u8; (cw * ch) as usize];
    for r in 0..h {
        for c in 0..w {
            y[(r * w + c) as usize] = ((r + c) as u16 + bias as u16).min(255) as u8;
        }
    }
    for r in 0..ch {
        for c in 0..cw {
            cb[(r * cw + c) as usize] = 96u8.saturating_add(r as u8);
            cr[(r * cw + c) as usize] = 160u8.saturating_sub(c as u8);
        }
    }
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width: w,
        height: h,
        pts: Some(0),
        time_base: TimeBase::new(1, 50),
        planes: vec![
            VideoPlane {
                stride: w as usize,
                data: y,
            },
            VideoPlane {
                stride: cw as usize,
                data: cb,
            },
            VideoPlane {
                stride: cw as usize,
                data: cr,
            },
        ],
    }
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = *x as f64 - *y as f64;
            d * d
        })
        .sum::<f64>()
        / (a.len() as f64);
    if mse < 1e-9 {
        99.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

/// Encode a single field through the PAFF-enabled encoder, then decode it.
fn encode_decode_field(src: &VideoFrame, qp: i32, bottom_field: bool) -> (Vec<u8>, VideoFrame) {
    let mut enc = H264Encoder::new(
        CodecId::new("h264"),
        src.width,
        src.height,
        H264EncoderOptions {
            qp,
            paff_field: Some(bottom_field),
            ..Default::default()
        },
    )
    .expect("encoder::new");
    enc.send_frame(&Frame::Video(src.clone())).expect("send_frame");
    enc.flush().expect("flush");
    let pkt = enc.receive_packet().expect("receive_packet");
    let bytes = pkt.data.clone();

    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let dec_pkt = Packet::new(0, TimeBase::new(1, 50), bytes.clone()).with_pts(0);
    dec.send_packet(&dec_pkt).expect("decoder send_packet");
    let f = match dec.receive_frame().expect("receive_frame") {
        Frame::Video(v) => v,
        _ => panic!("expected video frame"),
    };
    (bytes, f)
}

/// PAFF-mode encoder must write `frame_mbs_only_flag = 0` in the SPS and
/// `field_pic_flag = 1` / `bottom_field_flag` in the slice header. Parse
/// the produced bitstream back through the in-tree SPS / slice-header
/// parsers and assert the flags are exactly what PAFF requires.
#[test]
fn encoder_emits_paff_flags_top_field() {
    let src = make_field(32, 32, 0);
    let (bytes, _) = encode_decode_field(&src, 22, false);

    let nalus = split_annex_b(&bytes);
    let mut sps_nal: Option<&[u8]> = None;
    let mut pps_nal: Option<&[u8]> = None;
    let mut idr_nal: Option<&[u8]> = None;
    for n in &nalus {
        let h = NalHeader::parse(n[0]).unwrap();
        match h.nal_unit_type {
            NalUnitType::Sps => sps_nal = Some(n),
            NalUnitType::Pps => pps_nal = Some(n),
            NalUnitType::SliceIdr => idr_nal = Some(n),
            _ => {}
        }
    }
    let sps_nal = sps_nal.expect("SPS");
    let pps_nal = pps_nal.expect("PPS");
    let idr_nal = idr_nal.expect("IDR slice");

    let sps_header = NalHeader::parse(sps_nal[0]).unwrap();
    let sps_rbsp = oxideav_h264::nal::extract_rbsp(&sps_nal[1..]);
    let sps = oxideav_h264::sps::parse_sps(&sps_header, &sps_rbsp).expect("parse sps");
    assert!(
        !sps.frame_mbs_only_flag,
        "PAFF SPS must carry frame_mbs_only_flag=0"
    );
    assert!(
        !sps.mb_adaptive_frame_field_flag,
        "PAFF-only SPS must carry mb_adaptive_frame_field_flag=0"
    );

    let pps_header = NalHeader::parse(pps_nal[0]).unwrap();
    let pps_rbsp = oxideav_h264::nal::extract_rbsp(&pps_nal[1..]);
    let pps = oxideav_h264::pps::parse_pps(&pps_header, &pps_rbsp, None).expect("parse pps");

    let slice_header = NalHeader::parse(idr_nal[0]).unwrap();
    let slice_rbsp = oxideav_h264::nal::extract_rbsp(&idr_nal[1..]);
    let sh = parse_slice_header(&slice_header, &slice_rbsp, &sps, &pps).expect("parse slice");
    assert!(sh.field_pic_flag, "slice must carry field_pic_flag=1");
    assert!(
        !sh.bottom_field_flag,
        "top-field slice must carry bottom_field_flag=0"
    );
}

#[test]
fn encoder_emits_paff_flags_bottom_field() {
    let src = make_field(32, 32, 16);
    let (bytes, _) = encode_decode_field(&src, 22, true);

    let nalus = split_annex_b(&bytes);
    let idr_nal = nalus
        .iter()
        .copied()
        .find(|n| NalHeader::parse(n[0]).unwrap().nal_unit_type == NalUnitType::SliceIdr)
        .expect("IDR");
    let sps_nal = nalus
        .iter()
        .copied()
        .find(|n| NalHeader::parse(n[0]).unwrap().nal_unit_type == NalUnitType::Sps)
        .expect("SPS");
    let pps_nal = nalus
        .iter()
        .copied()
        .find(|n| NalHeader::parse(n[0]).unwrap().nal_unit_type == NalUnitType::Pps)
        .expect("PPS");

    let sps_header = NalHeader::parse(sps_nal[0]).unwrap();
    let sps_rbsp = oxideav_h264::nal::extract_rbsp(&sps_nal[1..]);
    let sps = oxideav_h264::sps::parse_sps(&sps_header, &sps_rbsp).unwrap();
    let pps_header = NalHeader::parse(pps_nal[0]).unwrap();
    let pps_rbsp = oxideav_h264::nal::extract_rbsp(&pps_nal[1..]);
    let pps = oxideav_h264::pps::parse_pps(&pps_header, &pps_rbsp, None).unwrap();
    let slice_header = NalHeader::parse(idr_nal[0]).unwrap();
    let slice_rbsp = oxideav_h264::nal::extract_rbsp(&idr_nal[1..]);
    let sh = parse_slice_header(&slice_header, &slice_rbsp, &sps, &pps).unwrap();
    assert!(sh.field_pic_flag);
    assert!(sh.bottom_field_flag);
}

/// Round-trip a top-field I-slice. The decoder must emit a half-height
/// VideoFrame whose samples match the input within the QP bound.
#[test]
fn roundtrip_paff_top_field_64x64_qp22() {
    let src = make_field(64, 64, 0);
    let (_bytes, dec) = encode_decode_field(&src, 22, false);
    assert_eq!(dec.width, 64, "field width must match encoder input");
    assert_eq!(dec.height, 64, "field height reflects the field's sample rows");
    let p = psnr(&src.planes[0].data, &dec.planes[0].data);
    eprintln!("paff top-field 64x64 qp22: luma psnr = {:.2} dB", p);
    assert!(p >= 26.0, "paff top-field luma psnr {:.2} < 26 dB", p);
}

/// Round-trip a bottom-field I-slice. Same content, different parity bit.
#[test]
fn roundtrip_paff_bottom_field_64x64_qp22() {
    let src = make_field(64, 64, 16);
    let (_bytes, dec) = encode_decode_field(&src, 22, true);
    assert_eq!(dec.width, 64);
    assert_eq!(dec.height, 64);
    let p = psnr(&src.planes[0].data, &dec.planes[0].data);
    eprintln!("paff bottom-field 64x64 qp22: luma psnr = {:.2} dB", p);
    assert!(p >= 26.0, "paff bottom-field luma psnr {:.2} < 26 dB", p);
}

/// §7.3.2.1.1 — a PAFF-flagged SPS paired with a plain frame-mode slice
/// (field_pic_flag = 0, mb_adaptive_frame_field_flag = 0 per our encoder)
/// does not carry enough MB rows for the frame picture. The decoder
/// currently gates that combination out with `Error::Unsupported` so we
/// don't silently produce corrupt output. This test pins the behaviour.
#[test]
fn decoder_rejects_frame_mode_slice_under_paff_sps() {
    // Build a packet by hand: take the encoder's PAFF SPS/PPS and combine
    // with a frame-mode slice header. Cheapest route is to produce two
    // separate encoder outputs (progressive + PAFF), then splice the
    // PAFF SPS/PPS with the progressive slice. That way the slice's
    // `field_pic_flag` bit is never written, triggering the reject.
    let src = make_field(32, 32, 0);
    // Progressive encoder: SPS has frame_mbs_only_flag = 1, slice has no
    // field_pic_flag bit.
    let mut prog = H264Encoder::new(
        CodecId::new("h264"),
        src.width,
        src.height,
        H264EncoderOptions {
            qp: 22,
            ..Default::default()
        },
    )
    .unwrap();
    prog.send_frame(&Frame::Video(src.clone())).unwrap();
    prog.flush().unwrap();
    let prog_bytes = prog.receive_packet().unwrap().data;

    // PAFF encoder: SPS has frame_mbs_only_flag = 0.
    let mut paff = H264Encoder::new(
        CodecId::new("h264"),
        src.width,
        src.height,
        H264EncoderOptions {
            qp: 22,
            paff_field: Some(false),
            ..Default::default()
        },
    )
    .unwrap();
    paff.send_frame(&Frame::Video(src.clone())).unwrap();
    paff.flush().unwrap();
    let paff_bytes = paff.receive_packet().unwrap().data;

    // Splice: [paff SPS] [paff PPS] [progressive slice].
    let paff_nalus = split_annex_b(&paff_bytes);
    let prog_nalus = split_annex_b(&prog_bytes);
    let sps = paff_nalus
        .iter()
        .copied()
        .find(|n| NalHeader::parse(n[0]).unwrap().nal_unit_type == NalUnitType::Sps)
        .unwrap();
    let pps = paff_nalus
        .iter()
        .copied()
        .find(|n| NalHeader::parse(n[0]).unwrap().nal_unit_type == NalUnitType::Pps)
        .unwrap();
    let slice = prog_nalus
        .iter()
        .copied()
        .find(|n| NalHeader::parse(n[0]).unwrap().nal_unit_type == NalUnitType::SliceIdr)
        .unwrap();
    let mut spliced = Vec::new();
    spliced.extend_from_slice(&[0, 0, 0, 1]);
    spliced.extend_from_slice(sps);
    spliced.extend_from_slice(&[0, 0, 0, 1]);
    spliced.extend_from_slice(pps);
    spliced.extend_from_slice(&[0, 0, 0, 1]);
    spliced.extend_from_slice(slice);

    let mut dec = H264Decoder::new(CodecId::new("h264"));
    let res = dec.send_packet(&Packet::new(0, TimeBase::new(1, 50), spliced).with_pts(0));
    // The decoder sees PAFF SPS + frame-mode slice and returns Unsupported.
    // Any error is fine; we just require the decoder not to panic and not
    // to decode silently.
    assert!(
        res.is_err(),
        "expected Unsupported for frame-mode slice under PAFF SPS"
    );
}

/// Decode a top-field then a bottom-field as two successive IDR packets.
/// The decoder must emit both frames, each carrying `PixelFormat::Yuv420P`
/// at field dimensions, without erroring out.
#[test]
fn decode_paff_top_then_bottom_pair() {
    let top_src = make_field(48, 32, 0);
    let bot_src = make_field(48, 32, 12);

    let mut enc_top = H264Encoder::new(
        CodecId::new("h264"),
        top_src.width,
        top_src.height,
        H264EncoderOptions {
            qp: 24,
            paff_field: Some(false),
            ..Default::default()
        },
    )
    .unwrap();
    enc_top.send_frame(&Frame::Video(top_src.clone())).unwrap();
    enc_top.flush().unwrap();
    let top_pkt = enc_top.receive_packet().unwrap();

    let mut enc_bot = H264Encoder::new(
        CodecId::new("h264"),
        bot_src.width,
        bot_src.height,
        H264EncoderOptions {
            qp: 24,
            paff_field: Some(true),
            ..Default::default()
        },
    )
    .unwrap();
    enc_bot.send_frame(&Frame::Video(bot_src.clone())).unwrap();
    enc_bot.flush().unwrap();
    let bot_pkt = enc_bot.receive_packet().unwrap();

    let mut dec = H264Decoder::new(CodecId::new("h264"));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 50), top_pkt.data).with_pts(0))
        .unwrap();
    let top_out = match dec.receive_frame().unwrap() {
        Frame::Video(v) => v,
        _ => panic!(),
    };

    dec.send_packet(&Packet::new(0, TimeBase::new(1, 50), bot_pkt.data).with_pts(20_000))
        .unwrap();
    let bot_out = match dec.receive_frame().unwrap() {
        Frame::Video(v) => v,
        _ => panic!(),
    };

    assert_eq!((top_out.width, top_out.height), (48, 32));
    assert_eq!((bot_out.width, bot_out.height), (48, 32));
    assert_eq!(top_out.format, PixelFormat::Yuv420P);
    assert_eq!(bot_out.format, PixelFormat::Yuv420P);
}

/// PAFF round-trip with a P-field following the I-field. The encoder's
/// `paff_field = Some(_)` option now emits the opening field as an IDR
/// I-slice and every subsequent field (same parity) as a zero-residual
/// P-slice whose `mb_skip_run` covers every macroblock. The decoder
/// must accept both packets, deliver two VideoFrames at field
/// dimensions, and the P-field decode must equal the I-field decode
/// sample-for-sample since every MB is a zero-MV copy of the IDR.
#[test]
fn roundtrip_paff_p_field_64x64_qp22() {
    let src = make_field(64, 64, 0);
    let mut enc = H264Encoder::new(
        CodecId::new("h264"),
        src.width,
        src.height,
        H264EncoderOptions {
            qp: 22,
            paff_field: Some(false),
            ..Default::default()
        },
    )
    .expect("encoder::new");

    // First frame → IDR I-field.
    enc.send_frame(&Frame::Video(src.clone())).expect("send I");
    let i_pkt = enc.receive_packet().expect("recv I");
    assert!(i_pkt.flags.keyframe, "first PAFF packet must be a keyframe");

    // Second frame → all-skip P-field.
    enc.send_frame(&Frame::Video(src.clone())).expect("send P");
    let p_pkt = enc.receive_packet().expect("recv P");
    assert!(
        !p_pkt.flags.keyframe,
        "PAFF P-field packet must not be flagged as a keyframe"
    );

    // Parse the P-field's slice header back and assert the P + field
    // flags are correct.
    let p_nalus = split_annex_b(&p_pkt.data);
    let sps_nal = p_nalus
        .iter()
        .copied()
        .find(|n| NalHeader::parse(n[0]).unwrap().nal_unit_type == NalUnitType::Sps)
        .expect("SPS");
    let pps_nal = p_nalus
        .iter()
        .copied()
        .find(|n| NalHeader::parse(n[0]).unwrap().nal_unit_type == NalUnitType::Pps)
        .expect("PPS");
    let p_slice_nal = p_nalus
        .iter()
        .copied()
        .find(|n| NalHeader::parse(n[0]).unwrap().nal_unit_type == NalUnitType::SliceNonIdr)
        .expect("P slice");
    let sps_header = NalHeader::parse(sps_nal[0]).unwrap();
    let sps_rbsp = oxideav_h264::nal::extract_rbsp(&sps_nal[1..]);
    let sps = oxideav_h264::sps::parse_sps(&sps_header, &sps_rbsp).unwrap();
    let pps_header = NalHeader::parse(pps_nal[0]).unwrap();
    let pps_rbsp = oxideav_h264::nal::extract_rbsp(&pps_nal[1..]);
    let pps = oxideav_h264::pps::parse_pps(&pps_header, &pps_rbsp, None).unwrap();
    let slice_header = NalHeader::parse(p_slice_nal[0]).unwrap();
    let slice_rbsp = oxideav_h264::nal::extract_rbsp(&p_slice_nal[1..]);
    let sh = parse_slice_header(&slice_header, &slice_rbsp, &sps, &pps).expect("parse P slice");
    assert_eq!(
        sh.slice_type,
        oxideav_h264::slice::SliceType::P,
        "PAFF second field must be slice_type = P"
    );
    assert!(sh.field_pic_flag, "P-field must carry field_pic_flag = 1");
    assert!(
        !sh.bottom_field_flag,
        "top-parity P-field must carry bottom_field_flag = 0"
    );
    assert_eq!(sh.frame_num, 1, "P-field follows IDR → frame_num = 1");

    // Round-trip decode.
    let mut dec = H264Decoder::new(CodecId::new("h264"));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 50), i_pkt.data.clone()).with_pts(0))
        .expect("decode I-field");
    let i_out = match dec.receive_frame().expect("I-field frame") {
        Frame::Video(v) => v,
        _ => panic!("expected video"),
    };
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 50), p_pkt.data.clone()).with_pts(20_000))
        .expect("decode P-field");
    let p_out = match dec.receive_frame().expect("P-field frame") {
        Frame::Video(v) => v,
        _ => panic!("expected video"),
    };

    assert_eq!((p_out.width, p_out.height), (64, 64));
    assert_eq!(p_out.format, PixelFormat::Yuv420P);

    // All MBs skipped with zero MV → the P-field samples exactly equal
    // the I-field samples. Allow zero tolerance across luma and chroma.
    for plane_idx in 0..3 {
        assert_eq!(
            p_out.planes[plane_idx].data, i_out.planes[plane_idx].data,
            "PAFF P-field plane {plane_idx} must be a bit-exact copy of the I-field"
        );
    }

    // Sanity: the P-field reconstruction is also close to the source
    // (bounded by the IDR reconstruction PSNR at QP 22).
    let p = psnr(&src.planes[0].data, &p_out.planes[0].data);
    eprintln!("paff p-field 64x64 qp22: luma psnr vs source = {:.2} dB", p);
    assert!(p >= 26.0, "paff P-field luma psnr {:.2} < 26 dB", p);
}
