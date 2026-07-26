//! Round-430 — HRD/VUI signalling on rate-controlled sessions.
//!
//! Rate-controlled [`EncoderSession`]s annotate their streams with the
//! Annex C buffering contract:
//!
//! * SPS VUI: §E.1.1 timing_info (field-based clock ticks, fixed frame
//!   rate) + a §E.1.2 NAL HRD block whose BitRate / CpbSize mirror the
//!   controller's leaky-bucket model (`cbr_flag` set in CBR mode);
//! * §D.1.2 buffering_period SEI on every IDR access unit
//!   (initial_cpb_removal_delay = the modelled CPB fill in 90 kHz
//!   units);
//! * §D.1.3 pic_timing SEI on every access unit (cpb_removal_delay in
//!   clock ticks since the last buffering-period AU, dpb_output_delay
//!   0 for this no-reorder IPP shape).
//!
//! Validation here: field-exact round-trip through the crate's own
//! SPS/VUI/SEI parsers, clean decode through our decoder, byte-exact
//! agreement with the black-box reference decoder, and black-box
//! probing that demonstrably READS the annotation (the probe derives
//! the tick rate from our VUI instead of its raw-stream default).

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase, VideoFrame};
use oxideav_h264::encoder::rate_control::RateControlConfig;
use oxideav_h264::encoder::session::{EncoderSession, SessionConfig, SessionFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264::nal::{parse_nal_unit, AnnexBSplitter, NalUnitType};
use oxideav_h264::non_vcl::parse_sei_rbsp;
use oxideav_h264::sei::{parse_buffering_period, parse_pic_timing, SeiContext};
use oxideav_h264::vui::VuiParameters;
use std::process::Command;

const FFMPEG: &str = "/opt/homebrew/bin/ffmpeg";
const FFPROBE: &str = "/opt/homebrew/bin/ffprobe";

const W: u32 = 80;
const H: u32 = 64;
const FRAMES: usize = 45;
const GOP: u32 = 30;
const BITRATE: u32 = 120_000;

fn make_frame(n: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (W as usize, H as usize);
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = ((i * 3 + j * 5 + n * 7) % 200) as u8 + 20;
        }
    }
    let u = vec![110u8; (w / 2) * (h / 2)];
    let v = vec![140u8; (w / 2) * (h / 2)];
    (y, u, v)
}

fn run_session(cfg: SessionConfig) -> (Vec<u8>, Vec<SessionFrame>) {
    let mut s = EncoderSession::new(cfg);
    let mut stream = Vec::new();
    let mut infos = Vec::new();
    for n in 0..FRAMES {
        let (y, u, v) = make_frame(n);
        let sf = s.encode_frame(&y, &u, &v);
        stream.extend_from_slice(&sf.annex_b);
        infos.push(sf);
    }
    (stream, infos)
}

fn cbr_stream() -> (Vec<u8>, Vec<SessionFrame>) {
    let mut cfg = SessionConfig::rate_controlled(W, H, RateControlConfig::cbr(BITRATE, 30, 1));
    cfg.gop_length = GOP;
    run_session(cfg)
}

fn decode_own(stream: &[u8]) -> (Vec<VideoFrame>, u64, Option<VuiParameters>) {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let packet = Packet::new(0, TimeBase::new(1, 30), stream.to_vec()).with_pts(0);
    dec.send_packet(&packet).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => frames.push(vf),
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    let vui = dec.stored_sps(0).and_then(|s| s.vui.clone());
    (frames, dec.decode_error_count(), vui)
}

/// Collect the SEI NAL RBSPs of the stream, in order, plus the NAL
/// type sequence of the first access unit.
fn sei_rbsps_and_first_au_types(stream: &[u8]) -> (Vec<Vec<u8>>, Vec<u8>) {
    let mut seis = Vec::new();
    let mut first_au_types = Vec::new();
    let mut seen_vcl = false;
    for nal in AnnexBSplitter::new(stream) {
        let nu = parse_nal_unit(nal).expect("nal parses");
        let ty = nu.header.nal_unit_type.as_u8();
        if !seen_vcl {
            first_au_types.push(ty);
            if (1..=5).contains(&ty) {
                seen_vcl = true;
            }
        }
        if nu.header.nal_unit_type == NalUnitType::Sei {
            seis.push(nu.rbsp.into_owned());
        }
    }
    (seis, first_au_types)
}

fn sei_ctx_from_vui(vui: &VuiParameters) -> SeiContext {
    let hrd = vui.nal_hrd_parameters.as_ref().expect("nal hrd present");
    SeiContext {
        initial_cpb_removal_delay_length_minus1: hrd.initial_cpb_removal_delay_length_minus1,
        cpb_removal_delay_length_minus1: hrd.cpb_removal_delay_length_minus1,
        dpb_output_delay_length_minus1: hrd.dpb_output_delay_length_minus1,
        time_offset_length: hrd.time_offset_length,
        nal_hrd_cpb_cnt_minus1: Some(hrd.cpb_cnt_minus1),
        vcl_hrd_cpb_cnt_minus1: None,
        pic_struct_present_flag: vui.pic_struct_present_flag,
        cpb_dpb_delays_present_flag: true,
        ..SeiContext::default()
    }
}

#[test]
fn cbr_stream_carries_vui_hrd_and_sei_schedule() {
    let (stream, infos) = cbr_stream();

    // Every AU carries annotation bits; constant-QP sessions carry none.
    assert!(infos.iter().all(|sf| sf.sei_bits > 0));

    // -- Own decode: clean, full frame count, VUI exposed. ------------
    let (frames, errors, vui) = decode_own(&stream);
    assert_eq!(frames.len(), FRAMES);
    assert_eq!(errors, 0, "annotated CBR stream must decode cleanly");
    let vui = vui.expect("SPS must carry VUI");

    // §E.1.1 timing_info — 30 fps as field-based ticks.
    let t = vui.timing_info.as_ref().expect("timing info");
    assert_eq!(t.num_units_in_tick, 1);
    assert_eq!(t.time_scale, 60);
    assert!(t.fixed_frame_rate_flag);

    // §E.1.2 NAL HRD — bucket matches the controller's model.
    let hrd = vui.nal_hrd_parameters.as_ref().expect("nal hrd");
    assert_eq!(hrd.cpb_cnt_minus1, 0);
    let bit_rate = (hrd.bit_rate_value_minus1[0] + 1) << (6 + hrd.bit_rate_scale);
    let cpb_size = (hrd.cpb_size_value_minus1[0] + 1) << (4 + hrd.cpb_size_scale);
    assert!(
        (BITRATE..BITRATE + 64).contains(&bit_rate),
        "declared BitRate {bit_rate} must cover the {BITRATE} bps channel (round-up-to-64)"
    );
    // RateControlConfig::cbr uses a one-second CPB.
    assert!(
        (BITRATE..BITRATE + 16).contains(&cpb_size),
        "declared CpbSize {cpb_size} must cover the one-second bucket"
    );
    assert!(hrd.cbr_flag[0], "CBR mode must set cbr_flag");
    assert_eq!(vui.low_delay_hrd_flag, Some(false));
    assert!(vui.vcl_hrd_parameters.is_none());
    assert!(!vui.pic_struct_present_flag);

    // -- SEI schedule. -------------------------------------------------
    let (seis, first_au_types) = sei_rbsps_and_first_au_types(&stream);
    assert_eq!(seis.len(), FRAMES, "one SEI NAL per access unit");
    // §7.4.1.2.3 AU order on the IDR: SPS, PPS, SEI, IDR slice.
    assert_eq!(first_au_types, vec![7, 8, 6, 5]);

    let ctx = sei_ctx_from_vui(&vui);
    let full_90k = (90_000.0 * f64::from(BITRATE) / f64::from(bit_rate)).round() as u32;
    for (n, rbsp) in seis.iter().enumerate() {
        let msgs = parse_sei_rbsp(rbsp).expect("sei envelope");
        let idx_in_gop = n % GOP as usize;
        let is_idr = idx_in_gop == 0;
        if is_idr {
            assert_eq!(msgs.len(), 2, "AU {n}: buffering_period + pic_timing");
            assert_eq!(msgs[0].payload_type, 0);
            let bp = parse_buffering_period(&msgs[0].payload, &ctx).expect("bp parses");
            assert_eq!(bp.seq_parameter_set_id, 0);
            let d = &bp.nal_hrd.as_ref().expect("nal delays")[0];
            assert!(
                d.initial_cpb_removal_delay > 0,
                "§D.2.2 initial_cpb_removal_delay > 0"
            );
            // Delay + offset span the declared bucket (90 kHz units of
            // CpbSize / BitRate — the controller's one-second bucket).
            let sum = d.initial_cpb_removal_delay + d.initial_cpb_removal_delay_offset;
            assert!(
                sum <= full_90k + 1,
                "AU {n}: delay {} + offset {} exceeds the bucket span {full_90k}",
                d.initial_cpb_removal_delay,
                d.initial_cpb_removal_delay_offset
            );
            assert!(bp.vcl_hrd.is_none());
        } else {
            assert_eq!(msgs.len(), 1, "AU {n}: pic_timing only");
        }
        let pt_msg = msgs.last().unwrap();
        assert_eq!(pt_msg.payload_type, 1);
        let pt = parse_pic_timing(&pt_msg.payload, &ctx).expect("pt parses");
        assert_eq!(
            pt.cpb_removal_delay,
            2 * idx_in_gop as u32,
            "AU {n}: cpb_removal_delay counts 2 ticks per frame since the BP AU"
        );
        assert_eq!(pt.dpb_output_delay, 0, "AU {n}: no-reorder IPP output");
        assert!(pt.pic_struct.is_none());
    }
}

#[test]
fn capped_vbr_stream_signals_vbr_bucket() {
    let mut cfg = SessionConfig::rate_controlled(
        W,
        H,
        RateControlConfig::capped_vbr(BITRATE, 2 * BITRATE, 30, 1),
    );
    cfg.gop_length = GOP;
    let (stream, _) = run_session(cfg);
    let (frames, errors, vui) = decode_own(&stream);
    assert_eq!(frames.len(), FRAMES);
    assert_eq!(errors, 0);
    let vui = vui.expect("SPS must carry VUI");
    let hrd = vui.nal_hrd_parameters.as_ref().expect("nal hrd");
    let bit_rate = (hrd.bit_rate_value_minus1[0] + 1) << (6 + hrd.bit_rate_scale);
    assert!(
        (2 * BITRATE..2 * BITRATE + 64).contains(&bit_rate),
        "capped VBR declares the PEAK channel rate, got {bit_rate}"
    );
    assert!(!hrd.cbr_flag[0], "capped VBR must clear cbr_flag");
}

/// Black-box acceptance: the reference tooling reads our annotation
/// (tick rate derived from the VUI instead of its raw-stream default),
/// decodes the annotated stream cleanly under `err_detect explode`
/// (malformed SEI/VUI would fail there), and produces byte-exact YUV
/// versus our decoder.
#[test]
fn black_box_probing_accepts_hrd_annotation() {
    if !std::path::Path::new(FFMPEG).exists() || !std::path::Path::new(FFPROBE).exists() {
        eprintln!("skip black-box probing: binaries not present");
        return;
    }
    let (stream, _) = cbr_stream();
    let dir = std::env::temp_dir().join(format!("oxideav-h264-r430hrd-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let h264 = dir.join("cbr-hrd.h264");
    let yuv = dir.join("cbr-hrd.yuv");
    std::fs::write(&h264, &stream).expect("write stream");

    // Probe: the declared 30 fps rides §E.2.1 field-based ticks, so
    // the probe reports a 60/1 tick rate — raw unannotated streams
    // get its 1200000/1 default instead.
    let probe = Command::new(FFPROBE)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
        ])
        .arg(&h264)
        .output()
        .expect("spawn probe");
    assert!(probe.status.success(), "probe rejected the stream");
    let rate = String::from_utf8_lossy(&probe.stdout).trim().to_string();
    assert_eq!(
        rate, "60/1",
        "probe must derive the tick rate from our VUI timing info"
    );

    // Strict decode: err_detect explode turns any SEI/VUI/HRD parse
    // problem into a hard failure.
    let status = Command::new(FFMPEG)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-err_detect",
            "explode",
            "-i",
        ])
        .arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-y"])
        .arg(&yuv)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success(), "reference decoder rejected the stream");

    // Byte parity with our decoder.
    let (own, errors, _) = decode_own(&stream);
    assert_eq!(errors, 0);
    let raw = std::fs::read(&yuv).expect("read yuv");
    let fsz = (W as usize * H as usize) * 3 / 2;
    assert_eq!(raw.len(), fsz * own.len(), "frame count");
    let ysz = W as usize * H as usize;
    for (n, vf) in own.iter().enumerate() {
        let b = n * fsz;
        assert_eq!(&raw[b..b + ysz], &vf.planes[0].data[..], "f{n} luma");
        assert_eq!(
            &raw[b + ysz..b + ysz + ysz / 4],
            &vf.planes[1].data[..],
            "f{n} cb"
        );
        assert_eq!(
            &raw[b + ysz + ysz / 4..b + fsz],
            &vf.planes[2].data[..],
            "f{n} cr"
        );
    }
}
