//! Round-451 — 12-bit / 14-bit decode gates (§7.4.2.1.1
//! `bit_depth_*_minus8` up to 6, High 4:4:4 Predictive per §A.2.7 —
//! the only profile whose bit-depth range reaches 14).
//!
//! The 10-bit surface has byte-exact staged fixtures (round 410); the
//! deeper depths had parse/transform support but zero stream
//! coverage. These gates build the streams themselves from the
//! low-level writers:
//!
//! * **I_PCM pictures** — the payload is `u(v)` at BitDepth, so the
//!   decode must return the exact sample values (little-endian u16
//!   planes): pins the SPS parse, the PCM sample reads and the >8-bit
//!   output packing at 12 and 14 bits.
//! * **Intra_16x16 DC pictures with coded residual** — arbitrary
//!   quantised DC levels (luma DC Hadamard §8.5.10, chroma DC
//!   §8.5.11, §8.5.12 at qP′ = QP + QpBdOffset) decoded by our
//!   decoder AND a black-box reference decoder, byte-compared: an
//!   independent check of the deep-bit-depth dequant/transform chain
//!   with no self-mirror in the loop.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::bitstream::BitWriter;
use oxideav_h264::encoder::cavlc::CoeffTokenContext;
use oxideav_h264::encoder::macroblock::{write_intra16x16_mb, I16x16McbConfig};
use oxideav_h264::encoder::nal::build_nal_unit;
use oxideav_h264::encoder::pps::{build_baseline_pps_rbsp, BaselinePpsConfig};
use oxideav_h264::encoder::sps::{build_baseline_sps_rbsp, BaselineSpsConfig};
use oxideav_h264::nal::NalUnitType;

const W: usize = 48;
const H: usize = 48;
const W_MBS: usize = W / 16;
const H_MBS: usize = H / 16;

fn parameter_sets(bit_depth: u32) -> Vec<u8> {
    let sps = build_baseline_sps_rbsp(&BaselineSpsConfig {
        seq_parameter_set_id: 0,
        level_idc: 20,
        width_in_mbs: W_MBS as u32,
        height_in_mbs: H_MBS as u32,
        log2_max_frame_num_minus4: 4,
        log2_max_poc_lsb_minus4: 4,
        max_num_ref_frames: 1,
        // §A.2.7 — High 4:4:4 Predictive admits bit depths 8..=14 at
        // every chroma format (4:2:0 here).
        profile_idc: 244,
        chroma_format_idc: 1,
        separate_colour_plane: false,
        seq_scaling_lists: None,
        bit_depth_luma_minus8: bit_depth - 8,
        bit_depth_chroma_minus8: bit_depth - 8,
        interlaced_fields: false,
        vui: None,
    });
    let pps = build_baseline_pps_rbsp(&BaselinePpsConfig {
        pic_parameter_set_id: 0,
        seq_parameter_set_id: 0,
        pic_init_qp_minus26: 0,
        chroma_qp_index_offset: 0,
        weighted_pred_flag: false,
        weighted_bipred_idc: 0,
        entropy_coding_mode_flag: false,
        transform_8x8_mode_flag: false,
        pic_scaling_lists: None,
        chroma_format_idc: 1,
    });
    let mut out = build_nal_unit(3, NalUnitType::Sps, &sps);
    out.extend_from_slice(&build_nal_unit(3, NalUnitType::Pps, &pps));
    out
}

/// §7.3.3 — IDR I-slice header (deblock off — these are single-frame
/// intra streams and the byte-compare should not depend on §8.7).
fn write_idr_header(bw: &mut BitWriter, qp: i32) {
    bw.ue(0); // first_mb_in_slice
    bw.ue(7); // slice_type: I, all slices in picture
    bw.ue(0); // pic_parameter_set_id
    bw.u(8, 0); // frame_num
    bw.ue(0); // idr_pic_id
    bw.u(8, 0); // pic_order_cnt_lsb
    bw.u(1, 0); // no_output_of_prior_pics_flag
    bw.u(1, 0); // long_term_reference_flag
    bw.se(qp - 26); // slice_qp_delta
    bw.ue(1); // disable_deblocking_filter_idc = 1
}

/// Deterministic deep-bit-depth source planes spanning well beyond the
/// 8-/10-bit ranges.
fn deep_source(bit_depth: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let max = (1u32 << bit_depth) - 1;
    let mut y = vec![0u16; W * H];
    let mut u = vec![0u16; (W / 2) * (H / 2)];
    let mut v = vec![0u16; (W / 2) * (H / 2)];
    for j in 0..H {
        for i in 0..W {
            y[j * W + i] = (((i * 89 + j * 131) as u32 * 97) % (max + 1)) as u16;
        }
    }
    for j in 0..H / 2 {
        for i in 0..W / 2 {
            u[j * (W / 2) + i] = (((i * 53 + j * 71) as u32 * 61 + 1000) % (max + 1)) as u16;
            v[j * (W / 2) + i] = (((i * 41 + j * 97) as u32 * 43 + 2000) % (max + 1)) as u16;
        }
    }
    (y, u, v)
}

/// Build an all-I_PCM IDR access unit at the given bit depth.
fn ipcm_idr(bit_depth: u32, y: &[u16], u: &[u16], v: &[u16]) -> Vec<u8> {
    let mut bw = BitWriter::new();
    write_idr_header(&mut bw, 26);
    let cw = W / 2;
    for mby in 0..H_MBS {
        for mbx in 0..W_MBS {
            bw.ue(25); // mb_type — I_PCM (Table 7-11)
            bw.align_to_byte_zero();
            for row in 0..16 {
                for col in 0..16 {
                    bw.u(
                        bit_depth,
                        u32::from(y[(mby * 16 + row) * W + mbx * 16 + col]),
                    );
                }
            }
            for plane in [u, v] {
                for row in 0..8 {
                    for col in 0..8 {
                        bw.u(
                            bit_depth,
                            u32::from(plane[(mby * 8 + row) * cw + mbx * 8 + col]),
                        );
                    }
                }
            }
        }
    }
    bw.rbsp_trailing_bits();
    build_nal_unit(3, NalUnitType::SliceIdr, &bw.into_bytes())
}

/// Build an IDR of Intra_16x16 DC macroblocks with coded luma +
/// chroma DC residual (`cbp_luma == 0`, `cbp_chroma == 1`): with no
/// AC anywhere every §9.2.1.1 TotalCoeff is 0, so the DC coeff_token
/// context is `Numeric(0)` for every macroblock and no nC grid is
/// needed — the levels are free choices that exercise the §8.5.10 /
/// §8.5.11 / §8.5.12 scaling at qP′ = QP + QpBdOffset.
fn i16x16_dc_idr(qp: i32, seed: u32) -> Vec<u8> {
    let mut bw = BitWriter::new();
    write_idr_header(&mut bw, qp);
    let mut state = seed | 1;
    let mut next = |range: i32| -> i32 {
        // xorshift32 — deterministic, no external deps.
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state % (2 * range as u32 + 1)) as i32 - range
    };
    for _mb in 0..W_MBS * H_MBS {
        let mut dc = [0i32; 16];
        for slot in dc.iter_mut() {
            *slot = next(6);
        }
        let mut dc_cb = [0i32; 4];
        let mut dc_cr = [0i32; 4];
        for slot in dc_cb.iter_mut() {
            *slot = next(4);
        }
        for slot in dc_cr.iter_mut() {
            *slot = next(4);
        }
        write_intra16x16_mb(
            &mut bw,
            &I16x16McbConfig {
                pred_mode: 2, // DC — legal with any neighbour availability
                intra_chroma_pred_mode: 0,
                cbp_luma: 0,
                cbp_chroma: 1,
                mb_qp_delta: 0,
                luma_dc_levels_raster: dc,
                luma_ac_levels: [[0i32; 16]; 16],
                luma_ac_nc: [0i32; 16],
                chroma_dc_cb: dc_cb,
                chroma_dc_cr: dc_cr,
                chroma_ac_cb: [[0i32; 16]; 4],
                chroma_ac_cr: [[0i32; 16]; 4],
                chroma_ac_nc_cb: [0i32; 8],
                chroma_ac_nc_cr: [0i32; 8],
            },
            CoeffTokenContext::Numeric(0),
        )
        .expect("I_16x16 emit");
    }
    bw.rbsp_trailing_bits();
    build_nal_unit(3, NalUnitType::SliceIdr, &bw.into_bytes())
}

/// Decode to per-plane byte vectors (LE u16 packing at >8-bit).
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

use oxideav_h264::h264_decoder::H264CodecDecoder;

fn le_bytes(src: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 2);
    for &s in src {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Black-box cross-check: decode `stream` with the stock reference
/// binary and byte-compare each frame against `ours`. Returns false
/// (skipping the check) when the binary is not present; panics on any
/// sample mismatch.
fn reference_decoder_matches(
    stream: &[u8],
    ours: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
    tag: &str,
) -> bool {
    let refbin = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !refbin.exists() {
        eprintln!("skip {tag}: reference decoder binary not present");
        return false;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h264-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bs = dir.join("input.h264");
    let out = dir.join("out.yuv");
    std::fs::write(&bs, stream).unwrap();
    let status = std::process::Command::new(refbin)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&bs)
        .args(["-f", "rawvideo"])
        .arg(&out)
        .status()
        .expect("spawn reference decoder");
    assert!(status.success(), "{tag}: reference decoder failed");
    let raw = std::fs::read(&out).unwrap();
    let frame_bytes = (W * H + 2 * (W / 2) * (H / 2)) * 2;
    assert_eq!(raw.len(), frame_bytes * ours.len(), "{tag}: frame count");
    for (i, (dy, du, dv)) in ours.iter().enumerate() {
        let f = &raw[i * frame_bytes..(i + 1) * frame_bytes];
        assert_eq!(&f[..W * H * 2], &dy[..], "{tag}: frame {i} luma");
        let c = (W / 2) * (H / 2) * 2;
        assert_eq!(&f[W * H * 2..W * H * 2 + c], &du[..], "{tag}: frame {i} Cb");
        assert_eq!(&f[W * H * 2 + c..], &dv[..], "{tag}: frame {i} Cr");
    }
    let _ = std::fs::remove_dir_all(&dir);
    true
}

/// 12- and 14-bit I_PCM round-trips: the decoded LE-u16 planes equal
/// the source samples exactly, and the full sample range is genuinely
/// exercised (values above the 10-bit ceiling on every plane).
#[test]
fn deep_bit_depth_ipcm_roundtrips_exact() {
    for bit_depth in [12u32, 14] {
        let (y, u, v) = deep_source(bit_depth);
        assert!(y.iter().any(|&s| s > 1023), "{bit_depth}-bit luma range");
        assert!(u.iter().any(|&s| s > 1023), "{bit_depth}-bit Cb range");
        let mut stream = parameter_sets(bit_depth);
        stream.extend_from_slice(&ipcm_idr(bit_depth, &y, &u, &v));
        let frames = decode_all(&stream);
        assert_eq!(frames.len(), 1, "{bit_depth}-bit PCM frame count");
        assert_eq!(frames[0].0, le_bytes(&y), "{bit_depth}-bit PCM luma");
        assert_eq!(frames[0].1, le_bytes(&u), "{bit_depth}-bit PCM Cb");
        assert_eq!(frames[0].2, le_bytes(&v), "{bit_depth}-bit PCM Cr");
        reference_decoder_matches(&stream, &frames, &format!("pcm{bit_depth}"));
    }
}

/// 12- and 14-bit Intra_16x16 DC pictures with coded luma/chroma DC
/// residual: decoded by our decoder and byte-compared against the
/// black-box reference decoder — an independent oracle on the
/// §8.5.10/§8.5.11/§8.5.12 chain at qP′ = QP + QpBdOffset (36 / 50 at
/// QP 26).
#[test]
fn deep_bit_depth_i16x16_dc_residual_matches_reference_decoder() {
    for (bit_depth, seed) in [(12u32, 0xC0FFEE), (14, 0xBEEF01)] {
        let mut stream = parameter_sets(bit_depth);
        stream.extend_from_slice(&i16x16_dc_idr(26, seed));
        let frames = decode_all(&stream);
        assert_eq!(frames.len(), 1, "{bit_depth}-bit I16x16 frame count");
        // The residual must genuinely move samples off the flat DC
        // prediction (1 << (BitDepth − 1)) — pin some variation.
        let mid = (1u32 << (bit_depth - 1)) as u16;
        let distinct = frames[0]
            .0
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .filter(|&s| s != mid)
            .count();
        assert!(
            distinct > W * H / 4,
            "{bit_depth}-bit: residual barely moved the picture ({distinct} samples off mid)"
        );
        reference_decoder_matches(&stream, &frames, &format!("i16x16dc{bit_depth}"));
    }
}
