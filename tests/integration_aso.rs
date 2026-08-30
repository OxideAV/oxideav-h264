//! Round-451 — arbitrary slice order (ASO) gates.
//!
//! The Baseline / Extended profiles (§A.2.1 / §A.2.3) allow the
//! slices of a coded picture to arrive in any order. Decoding must
//! not depend on slice arrival order: the §7.4.1.2.4 same-picture
//! detection has no monotonic-`first_mb_in_slice` assumption, each
//! slice deposits its macroblocks at its own addresses, intra
//! prediction and §9.2.1.1 nC neighbours never cross slice
//! boundaries (slice independence), and the §8.7 deblocking runs
//! once over the assembled picture.
//!
//! Gates: a 3-slice Intra_16x16 DC picture (coded DC residual — real
//! intra prediction and coefficient decode in every slice) and a
//! 3-slice all-I_PCM picture decode byte-identically under every
//! rotation of their slice order, and the in-order form byte-matches
//! a black-box reference decoder.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, Frame, Packet, TimeBase};
use oxideav_h264::encoder::bitstream::BitWriter;
use oxideav_h264::encoder::cavlc::CoeffTokenContext;
use oxideav_h264::encoder::macroblock::{write_intra16x16_mb, I16x16McbConfig};
use oxideav_h264::encoder::nal::build_nal_unit;
use oxideav_h264::encoder::pps::{build_baseline_pps_rbsp, BaselinePpsConfig};
use oxideav_h264::encoder::sps::{build_baseline_sps_rbsp, BaselineSpsConfig};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264::nal::NalUnitType;

const W: usize = 48;
const H: usize = 48;
const W_MBS: usize = W / 16;
const H_MBS: usize = H / 16;

fn parameter_sets() -> Vec<u8> {
    let sps = build_baseline_sps_rbsp(&BaselineSpsConfig {
        bit_depth_luma_minus8: 0,
        bit_depth_chroma_minus8: 0,
        seq_parameter_set_id: 0,
        level_idc: 20,
        width_in_mbs: W_MBS as u32,
        height_in_mbs: H_MBS as u32,
        log2_max_frame_num_minus4: 4,
        log2_max_poc_lsb_minus4: 4,
        max_num_ref_frames: 1,
        // §A.2.1 — ASO is a Baseline-profile tool.
        profile_idc: 66,
        chroma_format_idc: 1,
        separate_colour_plane: false,
        seq_scaling_lists: None,
        interlaced_fields: false,
        mbaff: false,
        vui: None,
    });
    let pps = build_baseline_pps_rbsp(&BaselinePpsConfig {
        redundant_pic_cnt_present_flag: false,
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

/// §7.3.3 — IDR I-slice header starting at MB row `row` (deblock on:
/// the once-per-picture §8.7 pass over the assembled grid is part of
/// what the reorder gates pin).
fn write_idr_slice_header(bw: &mut BitWriter, first_mb: u32) {
    bw.ue(first_mb);
    bw.ue(7); // slice_type: I, all slices in picture
    bw.ue(0); // pic_parameter_set_id
    bw.u(8, 0); // frame_num
    bw.ue(0); // idr_pic_id — identical across the picture's slices
    bw.u(8, 0); // pic_order_cnt_lsb
    bw.u(1, 0); // no_output_of_prior_pics_flag
    bw.u(1, 0); // long_term_reference_flag
    bw.se(0); // slice_qp_delta (QP 26)
    bw.ue(0); // disable_deblocking_filter_idc = 0
    bw.se(0); // slice_alpha_c0_offset_div2
    bw.se(0); // slice_beta_offset_div2
}

/// One-slice-per-MB-row Intra_16x16 DC picture with coded DC residual
/// (`cbp_luma = 0`, `cbp_chroma = 1` — every TotalCoeff is 0 so the
/// coeff_token context is `Numeric(0)` throughout).
fn i16x16_slices(seed: u32) -> Vec<Vec<u8>> {
    let mut state = seed | 1;
    let mut next = |range: i32| -> i32 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state % (2 * range as u32 + 1)) as i32 - range
    };
    let mut slices = Vec::new();
    for row in 0..H_MBS {
        let mut bw = BitWriter::new();
        write_idr_slice_header(&mut bw, (row * W_MBS) as u32);
        for _mb in 0..W_MBS {
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
                    pred_mode: 2, // DC
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
        slices.push(build_nal_unit(3, NalUnitType::SliceIdr, &bw.into_bytes()));
    }
    slices
}

/// One-slice-per-MB-row all-I_PCM picture.
fn ipcm_slices(seed: u8) -> Vec<Vec<u8>> {
    let cw = W / 2;
    let mut slices = Vec::new();
    for row in 0..H_MBS {
        let mut bw = BitWriter::new();
        write_idr_slice_header(&mut bw, (row * W_MBS) as u32);
        for mbx in 0..W_MBS {
            bw.ue(25); // mb_type — I_PCM
            bw.align_to_byte_zero();
            for i in 0..256usize {
                bw.u(
                    8,
                    ((i * 31 + mbx * 7 + row * 13 + seed as usize) % 256) as u32,
                );
            }
            for i in 0..128usize {
                let _ = cw;
                bw.u(
                    8,
                    ((i * 17 + mbx * 3 + row * 5 + seed as usize) % 256) as u32,
                );
            }
        }
        bw.rbsp_trailing_bits();
        slices.push(build_nal_unit(3, NalUnitType::SliceIdr, &bw.into_bytes()));
    }
    slices
}

fn decode_all(stream: &[u8]) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 25), stream.to_vec()).with_pts(0);
    dec.send_packet(&pkt).expect("send_packet");
    dec.flush().expect("flush");
    assert_eq!(dec.decode_error_count(), 0, "no skipped slices");
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

fn assemble(params: &[u8], slices: &[Vec<u8>], order: &[usize]) -> Vec<u8> {
    let mut out = params.to_vec();
    for &i in order {
        out.extend_from_slice(&slices[i]);
    }
    out
}

fn run_aso_case(slices: &[Vec<u8>], tag: &str) {
    let params = parameter_sets();
    let in_order = assemble(&params, slices, &[0, 1, 2]);
    let baseline = decode_all(&in_order);
    assert_eq!(baseline.len(), 1, "{tag}: one picture from 3 slices");
    for order in [[2usize, 0, 1], [1, 2, 0], [2, 1, 0]] {
        let stream = assemble(&params, slices, &order);
        let frames = decode_all(&stream);
        assert_eq!(
            frames.len(),
            1,
            "{tag}: {order:?} must assemble ONE picture"
        );
        assert_eq!(frames[0].0, baseline[0].0, "{tag}: {order:?} luma");
        assert_eq!(frames[0].1, baseline[0].1, "{tag}: {order:?} Cb");
        assert_eq!(frames[0].2, baseline[0].2, "{tag}: {order:?} Cr");
    }
    reference_decoder_check(&in_order, &baseline, tag);
}

/// Black-box check of the in-order stream (the reference binary is an
/// opaque oracle whose output we compare; skipped when absent).
fn reference_decoder_check(stream: &[u8], ours: &[(Vec<u8>, Vec<u8>, Vec<u8>)], tag: &str) {
    let refbin = std::path::Path::new("/opt/homebrew/bin/ffmpeg");
    if !refbin.exists() {
        eprintln!("skip {tag}: reference decoder binary not present");
        return;
    }
    let dir = std::env::temp_dir().join(format!("oxideav-h264-aso-{tag}-{}", std::process::id()));
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
    let frame_bytes = W * H * 3 / 2;
    assert_eq!(raw.len(), frame_bytes * ours.len(), "{tag}: frame count");
    let (dy, du, dv) = &ours[0];
    assert_eq!(&raw[..W * H], &dy[..], "{tag}: luma vs reference decoder");
    let c = (W / 2) * (H / 2);
    assert_eq!(&raw[W * H..W * H + c], &du[..], "{tag}: Cb");
    assert_eq!(&raw[W * H + c..], &dv[..], "{tag}: Cr");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn aso_i16x16_dc_picture_decodes_identically_in_any_slice_order() {
    run_aso_case(&i16x16_slices(0xA50A50), "aso-i16x16");
}

#[test]
fn aso_ipcm_picture_decodes_identically_in_any_slice_order() {
    run_aso_case(&ipcm_slices(3), "aso-ipcm");
}
