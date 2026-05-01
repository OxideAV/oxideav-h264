//! Round-24 — §8.4.1.2.3 B-slice **temporal direct** mode.
//!
//! These tests exercise the new encoder path in which the slice header
//! signals `direct_spatial_mv_pred_flag = 0` and B_Direct / B_Skip MBs
//! carry per-8x8 (mvL0, mvL1) derived by POC-distance scaling of the
//! L1 anchor's per-block colocated motion (eq. 8-191..8-202), instead
//! of round-21/23's spatial neighbour median.
//!
//! Three fixtures, mirroring the round-21/23 style:
//!
//! 1. `round24_temporal_direct_slice_header_signals_temporal` — pin
//!    that `direct_spatial_mv_pred_flag = 0` is emitted by the
//!    encoder when `EncoderConfig::direct_temporal_mv_pred = true`,
//!    and that the decoder accepts the resulting B-slice.
//!
//! 2. `round24_temporal_direct_self_roundtrip_matches_local_recon` —
//!    a non-trivial 5-frame `I-P-B-P-B` fixture (per the round-24
//!    spec). With moving content the temporal-direct derivation
//!    produces non-zero per-8x8 MVs that match the encoder's local
//!    recon to within 1 LSB after round-tripping through our own
//!    decoder.
//!
//! 3. `round24_temporal_direct_ffmpeg_interop` — feed the same 5-frame
//!    stream to libavcodec and verify bit-equivalent decode (max diff
//!    ≤ 1) on the B-frames.
//!
//! Cross-reference: §7.4.3 (`direct_spatial_mv_pred_flag`),
//! §8.4.1.2 (direct entry), §8.4.1.2.1 (colocated picture / block
//! selection — Table 8-6), §8.4.1.2.3 (eq. 8-191 MapColToList0,
//! eq. 8-192 refIdxL1, eq. 8-195/8-196 long-term shortcut, eq.
//! 8-197..8-200 DistScaleFactor scaling, eq. 8-201/8-202 td/tb).

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::{EncodedFrameRef, Encoder, EncoderConfig, YuvFrame};
use oxideav_h264::h264_decoder::H264CodecDecoder;

fn psnr(orig: &[u8], recon: &[u8]) -> f64 {
    let n = orig.len() as f64;
    let sse: f64 = orig
        .iter()
        .zip(recon.iter())
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum();
    if sse <= 0.0 {
        return 99.0;
    }
    let m = sse / n;
    10.0 * (255.0_f64 * 255.0 / m).log10()
}

// ---------------------------------------------------------------------------
// Test 1 — slice-header signalling: `direct_spatial_mv_pred_flag = 0`.
// ---------------------------------------------------------------------------

/// Verify the encoder emits `direct_spatial_mv_pred_flag = 0` in the
/// B-slice header when `EncoderConfig::direct_temporal_mv_pred =
/// true`. The decoder's parsed slice header is the source of truth.
#[test]
fn round24_temporal_direct_slice_header_signals_temporal() {
    const W: usize = 32;
    const H: usize = 16;
    let mut y = vec![0u8; W * H];
    for j in 0..H {
        for i in 0..W {
            y[j * W + i] = (60 + i + j) as u8;
        }
    }
    let u = vec![128u8; (W / 2) * (H / 2)];
    let v = vec![128u8; (W / 2) * (H / 2)];

    let mut cfg = EncoderConfig::new(W as u32, H as u32);
    cfg.profile_idc = 77;
    cfg.max_num_ref_frames = 2;
    cfg.direct_temporal_mv_pred = true; // round-24 switch
    let enc = Encoder::new(cfg);

    let f = YuvFrame {
        width: W as u32,
        height: H as u32,
        y: &y,
        u: &u,
        v: &v,
    };

    let idr = enc.encode_idr(&f);
    let p = enc.encode_p(&f, &EncodedFrameRef::from(&idr), 1, 4);
    let b = enc.encode_b(
        &f,
        &EncodedFrameRef::from(&idr),
        &EncodedFrameRef::from(&p),
        1,
        2,
    );

    // Pluck the B-slice NAL and parse its header.
    use oxideav_h264::nal::{parse_nal_unit, AnnexBSplitter, NalHeader, NalUnitType};
    use oxideav_h264::pps::Pps;
    use oxideav_h264::slice_header::SliceHeader;
    use oxideav_h264::sps::Sps;
    let split = AnnexBSplitter::new(&idr.annex_b);
    let mut sps: Option<Sps> = None;
    let mut pps: Option<Pps> = None;
    for nal_bytes in split {
        let nu = parse_nal_unit(nal_bytes).expect("nal");
        match nu.header.nal_unit_type {
            NalUnitType::Sps => sps = Some(Sps::parse(&nu.rbsp).expect("sps")),
            NalUnitType::Pps => pps = Some(Pps::parse(&nu.rbsp).expect("pps")),
            _ => {}
        }
    }
    let sps = sps.expect("SPS in IDR");
    let pps = pps.expect("PPS in IDR");

    let mut bsplit = AnnexBSplitter::new(&b.annex_b);
    let b_nal_bytes = bsplit.next().expect("B NAL");
    let b_nu = parse_nal_unit(b_nal_bytes).expect("parse B NAL");
    assert_eq!(b_nu.header.nal_unit_type, NalUnitType::SliceNonIdr);
    let header = NalHeader {
        forbidden_zero_bit: 0,
        nal_ref_idc: 0,
        nal_unit_type: NalUnitType::SliceNonIdr,
    };
    let parsed = SliceHeader::parse(&b_nu.rbsp, &sps, &pps, &header).expect("parse B header");
    assert!(
        !parsed.direct_spatial_mv_pred_flag,
        "encoder must signal direct_spatial_mv_pred_flag=0 when direct_temporal_mv_pred=true",
    );
    eprintln!(
        "round-24 slice-header check: direct_spatial_mv_pred_flag = {}",
        parsed.direct_spatial_mv_pred_flag
    );

    // Bonus: round-trip through our own decoder and verify it accepts
    // the temporal-direct stream (otherwise the decoder's
    // `build_temporal_direct_partitions` path is being skipped).
    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p.annex_b);
    combined.extend_from_slice(&b.annex_b);
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), combined.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => frames.push(vf),
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    assert_eq!(frames.len(), 3, "expected 3 decoded frames");
    // The decoded B should match the encoder's local recon within 1 LSB.
    let b_decoded = &frames[1];
    let mut max_diff = 0i32;
    for (k, (&a, &local)) in b_decoded.planes[0]
        .data
        .iter()
        .zip(b.recon_y.iter())
        .enumerate()
    {
        let d = (a as i32 - local as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d <= 1,
            "round-24 B luma mismatch at idx {k}: dec={a}, enc={local} (diff {d})",
        );
    }
    eprintln!(
        "round-24 slice-header test: B={} bytes, max enc/dec diff = {max_diff}",
        b.annex_b.len()
    );
}

// ---------------------------------------------------------------------------
// Test 2 / 3 — 5-frame I-P-B-P-B fixture for non-trivial temporal direct.
// ---------------------------------------------------------------------------
//
// Display order:  I0  B1  P2  B3  P4
//                 |    |   |   |   |
//                 v    v   v   v   v
// Decode order:   I0  P2  B1  P4  B3
//
// We encode in DECODE order: IDR, P (POC 4), B (POC 2), P (POC 8), B (POC 6).
//   B1 references L0 = IDR (POC 0), L1 = P2 (POC 4). pic_order_cnt_lsb = 2.
//   B3 references L0 = P2 (POC 4), L1 = P4 (POC 8). pic_order_cnt_lsb = 6.
// All POCs scaled ×2 to match the encoder's frame-coding convention
// (`TopFieldOrderCnt = pic_order_cnt_lsb`, frame POC = top FOC).

const W: usize = 64;
const H: usize = 32;

/// Frame `n` of a synthetic translating ramp: every frame shifts the
/// same ramp content right by `n` pixels (clamped). With a 4-pixel
/// stride per display step we land on integer-pel ME, which keeps the
/// reconstruction exact and the temporal-direct MVs recognisable.
fn make_translating_frame(disp_idx: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    let shift = disp_idx * 4; // 4-pixel stride per frame
    for j in 0..H {
        for i in 0..W {
            let i_src = i.saturating_sub(shift);
            y[j * W + i] = (60 + i_src + j) as u8;
        }
    }
    let u = vec![128u8; (W / 2) * (H / 2)];
    let v = vec![128u8; (W / 2) * (H / 2)];
    (y, u, v)
}

fn yuv<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> YuvFrame<'a> {
    YuvFrame {
        width: W as u32,
        height: H as u32,
        y,
        u,
        v,
    }
}

#[test]
fn round24_temporal_direct_self_roundtrip_matches_local_recon() {
    let mut cfg = EncoderConfig::new(W as u32, H as u32);
    cfg.profile_idc = 77;
    cfg.max_num_ref_frames = 2;
    cfg.direct_temporal_mv_pred = true;
    let enc = Encoder::new(cfg);

    // Display indices 0..=4 → translating ramp.
    let f0 = make_translating_frame(0);
    let f1 = make_translating_frame(1);
    let f2 = make_translating_frame(2);
    let f3 = make_translating_frame(3);
    let f4 = make_translating_frame(4);

    // Encode in decode order.
    let idr = enc.encode_idr(&yuv(&f0.0, &f0.1, &f0.2)); // POC 0 — display index 0
    let p2 = enc.encode_p(
        &yuv(&f2.0, &f2.1, &f2.2),
        &EncodedFrameRef::from(&idr),
        1,
        4,
    ); // POC 4
    let b1 = enc.encode_b(
        &yuv(&f1.0, &f1.1, &f1.2),
        &EncodedFrameRef::from(&idr),
        &EncodedFrameRef::from(&p2),
        1,
        2,
    ); // POC 2
    let p4 = enc.encode_p(&yuv(&f4.0, &f4.1, &f4.2), &EncodedFrameRef::from(&p2), 2, 8); // POC 8
    let b3 = enc.encode_b(
        &yuv(&f3.0, &f3.1, &f3.2),
        &EncodedFrameRef::from(&p2),
        &EncodedFrameRef::from(&p4),
        2,
        6,
    ); // POC 6

    // Sanity — measure local recon PSNR per encoded frame.
    let psnr_idr = psnr(&f0.0, &idr.recon_y);
    let psnr_p2 = psnr(&f2.0, &p2.recon_y);
    let psnr_b1 = psnr(&f1.0, &b1.recon_y);
    let psnr_p4 = psnr(&f4.0, &p4.recon_y);
    let psnr_b3 = psnr(&f3.0, &b3.recon_y);
    eprintln!(
        "round-24 self-recon PSNR_Y: IDR={psnr_idr:.2} P2={psnr_p2:.2} \
         B1={psnr_b1:.2} P4={psnr_p4:.2} B3={psnr_b3:.2}",
    );
    eprintln!(
        "round-24 stream sizes: IDR={} P2={} B1={} P4={} B3={} bytes",
        idr.annex_b.len(),
        p2.annex_b.len(),
        b1.annex_b.len(),
        p4.annex_b.len(),
        b3.annex_b.len(),
    );
    assert!(psnr_b1 >= 38.0, "B1 local PSNR_Y {psnr_b1:.2} below 38.0");
    assert!(psnr_b3 >= 38.0, "B3 local PSNR_Y {psnr_b3:.2} below 38.0");

    // Self-roundtrip through our own decoder. Frames arrive in POC
    // order via §C.4 bumping (with reorder slack 1): I, B1, P2, B3, P4.
    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p2.annex_b);
    combined.extend_from_slice(&b1.annex_b);
    combined.extend_from_slice(&p4.annex_b);
    combined.extend_from_slice(&b3.annex_b);

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), combined.clone()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");

    let mut frames = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => frames.push(vf),
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    assert_eq!(frames.len(), 5, "expected 5 decoded frames");

    // Display-order indices: 0=I, 1=B1, 2=P2, 3=B3, 4=P4.
    let cmp = |idx: usize, local_recon: &[u8], label: &str| -> i32 {
        let mut max_diff = 0i32;
        for (k, (&a, &local)) in frames[idx].planes[0]
            .data
            .iter()
            .zip(local_recon.iter())
            .enumerate()
        {
            let d = (a as i32 - local as i32).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(
                d <= 1,
                "{label} luma mismatch at idx {k}: dec={a}, enc={local} (diff {d})",
            );
        }
        max_diff
    };
    let md_b1 = cmp(1, &b1.recon_y, "B1");
    let md_b3 = cmp(3, &b3.recon_y, "B3");
    eprintln!("round-24 max enc/dec diff: B1 {md_b1}, B3 {md_b3}");
}

#[test]
fn round24_temporal_direct_ffmpeg_interop() {
    let ffmpeg = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !ffmpeg.exists() {
        eprintln!("skip: ffmpeg not at /opt/homebrew/bin/ffmpeg");
        return;
    }

    let mut cfg = EncoderConfig::new(W as u32, H as u32);
    cfg.profile_idc = 77;
    cfg.max_num_ref_frames = 2;
    cfg.direct_temporal_mv_pred = true;
    let enc = Encoder::new(cfg);

    let f0 = make_translating_frame(0);
    let f1 = make_translating_frame(1);
    let f2 = make_translating_frame(2);
    let f3 = make_translating_frame(3);
    let f4 = make_translating_frame(4);

    let idr = enc.encode_idr(&yuv(&f0.0, &f0.1, &f0.2));
    let p2 = enc.encode_p(
        &yuv(&f2.0, &f2.1, &f2.2),
        &EncodedFrameRef::from(&idr),
        1,
        4,
    );
    let b1 = enc.encode_b(
        &yuv(&f1.0, &f1.1, &f1.2),
        &EncodedFrameRef::from(&idr),
        &EncodedFrameRef::from(&p2),
        1,
        2,
    );
    let p4 = enc.encode_p(&yuv(&f4.0, &f4.1, &f4.2), &EncodedFrameRef::from(&p2), 2, 8);
    let b3 = enc.encode_b(
        &yuv(&f3.0, &f3.1, &f3.2),
        &EncodedFrameRef::from(&p2),
        &EncodedFrameRef::from(&p4),
        2,
        6,
    );

    let mut combined = Vec::new();
    combined.extend_from_slice(&idr.annex_b);
    combined.extend_from_slice(&p2.annex_b);
    combined.extend_from_slice(&b1.annex_b);
    combined.extend_from_slice(&p4.annex_b);
    combined.extend_from_slice(&b3.annex_b);

    let tmpdir = std::env::temp_dir();
    let h264_path = tmpdir.join(format!(
        "oxideav_h264_round24_temporal_{}.h264",
        std::process::id()
    ));
    let yuv_path = tmpdir.join(format!(
        "oxideav_h264_round24_temporal_{}.yuv",
        std::process::id()
    ));
    std::fs::write(&h264_path, &combined).expect("write h264");

    let status = std::process::Command::new(ffmpeg)
        .arg("-loglevel")
        .arg("error")
        .arg("-y")
        .arg("-f")
        .arg("h264")
        .arg("-i")
        .arg(&h264_path)
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(&yuv_path)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg failed");
    let raw = std::fs::read(&yuv_path).expect("read yuv");
    let _ = std::fs::remove_file(&h264_path);
    let _ = std::fs::remove_file(&yuv_path);

    let y_len = W * H;
    let c_len = (W / 2) * (H / 2);
    let frame_len = y_len + 2 * c_len;
    assert_eq!(raw.len(), 5 * frame_len, "expected 5 frames from ffmpeg");

    // ffmpeg writes display-order: 0=I 1=B1 2=P2 3=B3 4=P4.
    let cmp_ff = |disp_idx: usize, local_recon: &[u8], label: &str| -> i32 {
        let off = disp_idx * frame_len;
        let plane = &raw[off..off + y_len];
        let mut max_diff = 0i32;
        for (k, (&a, &local)) in plane.iter().zip(local_recon.iter()).enumerate() {
            let d = (a as i32 - local as i32).abs();
            if d > max_diff {
                max_diff = d;
            }
            assert!(
                d <= 1,
                "ffmpeg {label} luma {k}: ff={a} enc={local} (diff {d})",
            );
        }
        max_diff
    };
    let md_b1 = cmp_ff(1, &b1.recon_y, "B1");
    let md_b3 = cmp_ff(3, &b3.recon_y, "B3");
    eprintln!(
        "round-24 ffmpeg interop: max enc/dec diff B1={md_b1}, B3={md_b3}, total {} bytes",
        combined.len()
    );

    // Per-frame PSNR on B-slices.
    let b1_y = &raw[frame_len..frame_len + y_len];
    let b3_y = &raw[3 * frame_len..3 * frame_len + y_len];
    let psnr_b1 = psnr(&f1.0, b1_y);
    let psnr_b3 = psnr(&f3.0, b3_y);
    eprintln!("round-24 ffmpeg PSNR_Y: B1={psnr_b1:.2} dB, B3={psnr_b3:.2} dB",);
    assert!(psnr_b1 >= 38.0, "ffmpeg B1 PSNR_Y {psnr_b1:.2} below 38.0");
    assert!(psnr_b3 >= 38.0, "ffmpeg B3 PSNR_Y {psnr_b3:.2} below 38.0");
}

// ---------------------------------------------------------------------------
// Unit-style sanity check on the temporal-direct derivation primitive.
// ---------------------------------------------------------------------------
//
// Given the helper inputs we exercise inside the encoder, verify the
// boundary cases cited in §8.4.1.2.3:
//   * mvCol = 0 → mvL0 = mvL1 = 0 regardless of POC distances.
//   * pic1 == pic0 (td == 0) → mvL0 = mvCol, mvL1 = 0 (eq. 8-195/8-196).
//   * symmetric midpoint (curr exactly between pic0 and pic1) →
//     mvL0 = mvCol/2, mvL1 = -mvCol/2 (within rounding).
//
// These mirror the existing tests in `mv_deriv.rs`; we re-pin them
// here at the integration boundary to guard against accidental
// regressions in the encoder's wiring.

#[test]
fn round24_temporal_direct_helper_zero_mv_yields_zero_outputs() {
    use oxideav_h264::mv_deriv::{derive_b_temporal_direct, Mv};
    // mvCol = (0, 0), arbitrary POCs. Result should be (0, 0)/(0, 0).
    let (l0, l1) = derive_b_temporal_direct(Mv::ZERO, 8, 4, 0);
    assert_eq!(l0, Mv::ZERO);
    assert_eq!(l1, Mv::ZERO);
}

#[test]
fn round24_temporal_direct_helper_td_zero_passthrough() {
    use oxideav_h264::mv_deriv::{derive_b_temporal_direct, Mv};
    // td = pic1 - pic0 = 0 → eq. 8-195/8-196: mvL0 = mvCol, mvL1 = 0.
    let mv_col = Mv::new(8, -12);
    let (l0, l1) = derive_b_temporal_direct(mv_col, 4, 6, 4);
    assert_eq!(l0, mv_col);
    assert_eq!(l1, Mv::ZERO);
}

#[test]
fn round24_temporal_direct_helper_symmetric_midpoint() {
    use oxideav_h264::mv_deriv::{derive_b_temporal_direct, Mv};
    // pic0 = 0, pic1 = 8, curr = 4 (exact midpoint).
    //   td = 8, tb = 4, tx = (16384 + 4) / 8 = 2048,
    //   DSF = (4*2048 + 32) >> 6 = 128, clipped to 128.
    //   mvL0 = (128*mvCol + 128) >> 8.
    let mv_col = Mv::new(16, 32);
    let (l0, l1) = derive_b_temporal_direct(mv_col, 8, 4, 0);
    // mvL0 = (128*16 + 128) >> 8 = 8 ; (128*32 + 128) >> 8 = 16
    assert_eq!(l0, Mv::new(8, 16));
    // mvL1 = mvL0 - mvCol = (-8, -16)
    assert_eq!(l1, Mv::new(-8, -16));
}
