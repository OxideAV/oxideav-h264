//! Round-451 — §7.3.2.9 slice data partitioning gates (Extended
//! profile, NAL types 2/3/4).
//!
//! A data-partitioned slice splits its coded data into partition A
//! (slice header + slice_id + every category-2 element), partition B
//! (category-3 residual: intra collective types + I_PCM sample
//! payloads) and partition C (category-4 residual: inter collective
//! types). The gates emit the SAME coded content in single-NAL and
//! partitioned form and require identical reconstruction — any
//! mis-routing of a single read between the three bitstreams
//! desynchronises CAVLC immediately.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::sp::{
    encode_ipcm_idr, encode_si_dual_form_picture, encode_sp_dual_form_picture, encode_sp_picture,
    sp_parameter_sets, SpConfig,
};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const W: usize = 48;
const H: usize = 48;

fn decode_all(stream: &[u8]) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream.to_vec()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    let mut out = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => {
                assert_eq!(vf.planes.len(), 3);
                out.push((
                    vf.planes[0].data.to_vec(),
                    vf.planes[1].data.to_vec(),
                    vf.planes[2].data.to_vec(),
                ));
            }
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    out
}

fn source_frame(t: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; W * H];
    let mut u = vec![0u8; (W / 2) * (H / 2)];
    let mut v = vec![0u8; (W / 2) * (H / 2)];
    for j in 0..H {
        for i in 0..W {
            let base = 40 + ((i + 3 * t) % 32) * 5 + ((j * 7 + 2 * t) % 23);
            let texture = ((i * j + t) % 13) * 3;
            y[j * W + i] = (base + texture).clamp(0, 255) as u8;
        }
    }
    for j in 0..H / 2 {
        for i in 0..W / 2 {
            u[j * (W / 2) + i] = (96 + ((i * 5 + j + 4 * t) % 40) * 2) as u8;
            v[j * (W / 2) + i] = (150 + ((i + j * 3 + 2 * t) % 30) * 2) as u8;
        }
    }
    (y, u, v)
}

/// Split an Annex B stream into NAL units (payload includes the NAL
/// header byte).
fn split_nals(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut start: Option<usize> = None;
    while i + 3 <= stream.len() {
        let sc3 = stream[i..i + 3] == [0, 0, 1];
        let sc4 = i + 4 <= stream.len() && stream[i..i + 4] == [0, 0, 0, 1];
        if sc3 || sc4 {
            if let Some(s) = start {
                out.push(stream[s..i].to_vec());
            }
            i += if sc4 { 4 } else { 3 };
            start = Some(i);
        } else {
            i += 1;
        }
    }
    if let Some(s) = start {
        out.push(stream[s..].to_vec());
    }
    out
}

fn nal_types(stream: &[u8]) -> Vec<u8> {
    split_nals(stream).iter().map(|n| n[0] & 0x1F).collect()
}

/// Gate 1 — a mixed SP slice (inter `P_L0_16x16` + intra I_PCM
/// macroblocks) emitted as A + B + C partitions decodes byte-exactly
/// to the single-NAL form of the same coded content AND to the
/// encoder's §8.6 mirror (deblock off): partition A drives the
/// category-2 parse, I_PCM payloads come from partition B (with
/// alignment relative to that bitstream), and inter residual from
/// partition C.
#[test]
fn dp_mixed_sp_slice_all_three_partitions_decode_identical_to_single_form() {
    let cfg = SpConfig {
        width: W as u32,
        height: H as u32,
        qp: 24,
        qs: 24,
        deblock: false,
    };
    let (y0, u0, v0) = source_frame(0);
    let (y1, u1, v1) = source_frame(1);

    let params = sp_parameter_sets(&cfg);
    let idr = encode_ipcm_idr(&cfg, &y0, &u0, &v0);
    let dual = encode_sp_dual_form_picture(&cfg, (&y1, &u1, &v1), (&y0, &u0, &v0), 1, 2, 2, 5);

    // The partitioned AU genuinely carries all three partition types.
    assert_eq!(nal_types(&dual.partitioned_annex_b), vec![2, 3, 4]);

    let mut single = params.clone();
    single.extend_from_slice(&idr);
    single.extend_from_slice(&dual.single_annex_b);
    let mut split = params;
    split.extend_from_slice(&idr);
    split.extend_from_slice(&dual.partitioned_annex_b);

    let frames_single = decode_all(&single);
    let frames_split = decode_all(&split);
    assert_eq!(frames_single.len(), 2);
    assert_eq!(
        frames_split.len(),
        2,
        "partitioned stream must decode fully"
    );
    for p in 0..3 {
        let (a, b) = match p {
            0 => (&frames_single[1].0, &frames_split[1].0),
            1 => (&frames_single[1].1, &frames_split[1].1),
            _ => (&frames_single[1].2, &frames_split[1].2),
        };
        assert_eq!(a, b, "plane {p}: partitioned vs single-NAL decode");
    }
    // Both equal the encoder mirror (I_PCM MBs = source samples, SP
    // MBs = §8.6.1 reconstruction).
    assert_eq!(frames_split[1].0, dual.recon_y, "luma vs mirror");
    assert_eq!(frames_split[1].1, dual.recon_u, "Cb vs mirror");
    assert_eq!(frames_split[1].2, dual.recon_v, "Cr vs mirror");
}

/// Gate 2 — a partitioned SI picture (A + B: the whole residual is
/// category 3) reproduces the primary SP reconstruction byte-exactly,
/// deblocking on: the §8.6.2 switching identity holds straight
/// through the data-partitioned parse path.
#[test]
fn dp_partitioned_si_picture_keeps_the_switching_identity() {
    let cfg = SpConfig {
        width: W as u32,
        height: H as u32,
        qp: 26,
        qs: 26,
        deblock: true,
    };
    let (ya, ua, va) = source_frame(3);
    let (ya1, ua1, va1) = source_frame(4);
    let (yb, ub, vb) = source_frame(9);

    // Stream A: IDR_A + primary SP_A1 (single-NAL), decoded.
    let mut stream_a = sp_parameter_sets(&cfg);
    stream_a.extend_from_slice(&encode_ipcm_idr(&cfg, &ya, &ua, &va));
    let idr_a_dec = decode_all(&stream_a);
    let sp_a1 = encode_sp_picture(
        &cfg,
        (&ya1, &ua1, &va1),
        (&idr_a_dec[0].0, &idr_a_dec[0].1, &idr_a_dec[0].2),
        1,
        2,
        2,
    );
    stream_a.extend_from_slice(&sp_a1.annex_b);
    let frames_a = decode_all(&stream_a);
    assert_eq!(frames_a.len(), 2);

    // SI switch in dual form after an unrelated anchor.
    let si = encode_si_dual_form_picture(&cfg, &sp_a1.targets, 1, 2);
    assert_eq!(nal_types(&si.partitioned_annex_b), vec![2, 3]);

    let params = sp_parameter_sets(&cfg);
    let idr_b = encode_ipcm_idr(&cfg, &yb, &ub, &vb);
    let mut split = params.clone();
    split.extend_from_slice(&idr_b);
    split.extend_from_slice(&si.partitioned_annex_b);
    let mut single = params;
    single.extend_from_slice(&idr_b);
    single.extend_from_slice(&si.single_annex_b);

    let frames_split = decode_all(&split);
    let frames_single = decode_all(&single);
    assert_eq!(frames_split.len(), 2);
    assert_eq!(frames_single.len(), 2);

    // Identity: both forms == stream A's decoded SP picture.
    assert_eq!(frames_split[1].0, frames_a[1].0, "DP SI luma identity");
    assert_eq!(frames_split[1].1, frames_a[1].1, "DP SI Cb identity");
    assert_eq!(frames_split[1].2, frames_a[1].2, "DP SI Cr identity");
    assert_eq!(frames_single[1].0, frames_a[1].0, "single SI luma identity");
}

/// Gate 3 — losing a required partition is a decode ERROR, never a
/// silent mis-decode: dropping the partition-C NAL (whose residual
/// the partition-A category-2 elements call for) must surface as an
/// error from the decode driver.
#[test]
fn dp_missing_partition_c_is_a_decode_error() {
    let cfg = SpConfig {
        width: W as u32,
        height: H as u32,
        qp: 24,
        qs: 24,
        deblock: false,
    };
    let (y0, u0, v0) = source_frame(5);
    let (y1, u1, v1) = source_frame(6);

    let mut stream = sp_parameter_sets(&cfg);
    stream.extend_from_slice(&encode_ipcm_idr(&cfg, &y0, &u0, &v0));
    let dual = encode_sp_dual_form_picture(&cfg, (&y1, &u1, &v1), (&y0, &u0, &v0), 1, 2, 2, 5);
    // Re-assemble the partitioned AU without the type-4 NAL.
    for nal in split_nals(&dual.partitioned_annex_b) {
        if nal[0] & 0x1F != 4 {
            stream.extend_from_slice(&[0, 0, 0, 1]);
            stream.extend_from_slice(&nal);
        }
    }

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream).with_pts(0);
    let send = dec.send_packet(&pkt);
    let flush = dec.flush();
    assert!(
        send.is_err() || flush.is_err(),
        "dropping partition C must surface a decode error"
    );
}

/// Gate 4 — a partition B/C without a preceding partition A is a
/// stream error.
#[test]
fn dp_partition_b_without_partition_a_is_a_decode_error() {
    let cfg = SpConfig {
        width: W as u32,
        height: H as u32,
        qp: 24,
        qs: 24,
        deblock: false,
    };
    let (y0, u0, v0) = source_frame(7);
    let (y1, u1, v1) = source_frame(8);

    let mut stream = sp_parameter_sets(&cfg);
    stream.extend_from_slice(&encode_ipcm_idr(&cfg, &y0, &u0, &v0));
    let dual = encode_sp_dual_form_picture(&cfg, (&y1, &u1, &v1), (&y0, &u0, &v0), 1, 2, 2, 5);
    // Keep only the B/C partitions (drop A).
    for nal in split_nals(&dual.partitioned_annex_b) {
        if matches!(nal[0] & 0x1F, 3 | 4) {
            stream.extend_from_slice(&[0, 0, 0, 1]);
            stream.extend_from_slice(&nal);
        }
    }
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream).with_pts(0);
    // The codec driver's resilience policy counts-and-skips broken
    // slices rather than killing the stream — the orphan partitions
    // must land in the error counter, never decode silently.
    let send = dec.send_packet(&pkt);
    let flush = dec.flush();
    assert!(
        send.is_err() || flush.is_err() || dec.decode_error_count() > 0,
        "a partition B/C without partition A must surface a decode error"
    );
    assert!(
        dec.decode_error_count() >= 2,
        "both orphan partitions must be counted"
    );
}
