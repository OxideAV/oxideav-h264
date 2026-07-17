//! Round-416 — PAFF (picture-adaptive frame/field) encoding driver.
//!
//! Encodes an interlaced sequence as **field pictures**
//! (`field_pic_flag == 1`, §7.4.3): every input frame is split into its
//! top field (even sample rows) and bottom field (odd sample rows), and
//! each field is coded as an independent half-height coded picture
//! (§7.4.2.1.1 eq. 7-26: `PicHeightInMbs = FrameHeightInMbs / 2`).
//! Optionally, selected frames are instead coded as full-height FRAME
//! pictures (`field_pic_flag == 0`) inside the same interlaced stream —
//! the "picture-adaptive" axis.
//!
//! Structure of the emitted stream (Annex B):
//!
//! * SPS: Main profile (77 — §A.2.1 bars interlace from Baseline),
//!   `frame_mbs_only_flag = 0`, `mb_adaptive_frame_field_flag = 0`,
//!   CAVLC, 4:2:0, `pic_order_cnt_type = 0`.
//! * Frame 0: IDR **top field** + non-IDR I **bottom field**. Per
//!   §7.4.3 the two fields of the frame share `frame_num` (= 0; the
//!   second field's `frame_num == PrevRefFrameNum` case is sanctioned
//!   because the preceding reference picture is an IDR field of
//!   opposite parity).
//! * Frame k > 0: either an I/I field pair, a P/P field pair, or an I
//!   FRAME picture. Every picture is a reference (`nal_ref_idc != 0`,
//!   sliding-window marking), `frame_num = k` for both fields.
//! * POC (type 0): `pic_order_cnt_lsb = 2k` for the top field / frame
//!   picture, `2k + 1` for the bottom field, so output order matches
//!   the §8.2.1 eq. 8-1 frame POC = Min(TopFOC, BottomFOC) = 2k.
//!
//! P fields use a single active reference (`num_ref_idx_l0_active = 1`
//! from the PPS default), so every coded `ref_idx` is 0 and no ref_idx
//! bits appear. Per the §8.2.4.2.5 field reference-list initialisation
//! (fields of the ordered reference frames, alternating parity starting
//! with the SAME parity as the current field), `RefPicList0[0]` for the
//! first AND second field of frame k is the **same-parity field of
//! frame k − 1** — which is exactly the reference the per-field motion
//! search here uses. The §8.4 decoding process then runs entirely in
//! field coordinates (half-height planes, field MVs), so the existing
//! frame-based MB encoder applies unchanged to each field.
//!
//! The per-field in-loop deblock runs with `field_pic = 1` so the
//! §8.7.2.1 field rules apply (horizontal intra MB edges take bS = 3;
//! the bS = 4 first bullet requires frame macroblocks or a vertical
//! edge).
//!
//! Scope (fixture-grade, mirrors what the staged PAFF fixtures pin):
//! CAVLC, 4:2:0, top-field-first, same-parity P references only (no
//! cross-parity refs, no B fields), frame pictures only in all-I
//! sequences.

use crate::encoder::deblock::{
    deblock_recon_field, deblock_recon_with_chroma_array_type, MbDeblockInfo,
};
use crate::encoder::nal::build_nal_unit;
use crate::encoder::pps::{build_baseline_pps_rbsp, BaselinePpsConfig};
use crate::encoder::slice::{
    write_idr_i_slice_header, write_p_slice_header, FieldPicSignal, IdrSliceHeaderConfig,
    PSliceHeaderConfig,
};
use crate::encoder::sps::{build_baseline_sps_rbsp, BaselineSpsConfig};
use crate::encoder::{
    min_level_idc_for_picture_size, EncodedFrameRef, Encoder, EncoderConfig, YuvFrame,
};
use crate::encoder::{BitWriter, CavlcNcGrid, IntraGrid, MvGrid};
use crate::nal::NalUnitType;
use crate::transform::{qp_bd_offset, qp_y_to_qp_c_with_bd_offset};

/// Configuration for [`encode_paff_sequence`].
#[derive(Debug, Clone)]
pub struct PaffConfig {
    /// Luma width in samples (multiple of 16).
    pub width: u32,
    /// FRAME luma height in samples (multiple of 32 so each field is
    /// MB-aligned).
    pub frame_height: u32,
    /// Slice QP_Y (coded as `pic_init_qp_minus26`, `slice_qp_delta=0`).
    pub qp: i32,
    /// When `true`, fields after frame 0 are coded as P fields whose
    /// single L0 reference is the same-parity field of the previous
    /// frame; when `false` every picture is intra.
    pub p_fields: bool,
    /// Frame indices to code as full-height I FRAME pictures
    /// (`field_pic_flag = 0`) instead of a field pair. Must be empty
    /// when `p_fields` is set and must not contain 0 (frame 0 is the
    /// IDR field pair).
    pub frame_picture_indices: Vec<usize>,
    /// Round-416 — §8.4.1.4 Table 8-10 axis: when `true` (requires
    /// `p_fields`), the bottom field of frame 0 is coded as a P field
    /// whose single reference is the IDR TOP field — an
    /// OPPOSITE-parity field reference (the only candidate the
    /// §8.2.4.2.5 init can offer a second field whose same-parity list
    /// is empty). The chroma predictor then applies the Table 8-10
    /// `mvCLX[1] = mvLX[1] + 2` adjustment (top reference, bottom
    /// current field).
    pub cross_parity_first_bottom: bool,
    /// Round-416 — frame-reference axis: when `true` (requires
    /// `p_fields`), frame 0 is coded as an **IDR full-height FRAME
    /// picture** (`field_pic_flag = 0`) and frame 1's P fields
    /// reference the parity fields OF THAT FRAME — per §8.2.4.2.5 a
    /// stored frame supplies either parity field as a distinct
    /// reference picture, which a decoder serves as a half-height
    /// field view of the stored frame.
    pub idr_frame_first: bool,
}

/// A reconstructed reference field: (Y, Cb, Cr) half-height planes +
/// the field's own picture order count.
type ReconField = (Vec<u8>, Vec<u8>, Vec<u8>, i32);

/// One encoded PAFF sequence: the Annex B stream plus the full-height
/// per-frame reconstruction (fields re-interleaved, §8.7 post-filter)
/// that a conformant decoder outputs.
pub struct PaffEncoded {
    pub annex_b: Vec<u8>,
    /// Per input frame: (Y, Cb, Cr) full-height 4:2:0 planes.
    pub recon_frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
}

/// Split a full-height 4:2:0 frame into one parity's field planes
/// (§6.4.1: top field = even sample rows, bottom field = odd rows; the
/// 4:2:0 chroma rows split by the same parity).
fn extract_field(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    frame_height: usize,
    bottom: bool,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = width / 2;
    let ch = frame_height / 2;
    let start = usize::from(bottom);
    let mut fy = Vec::with_capacity(width * frame_height / 2);
    for row in (start..frame_height).step_by(2) {
        fy.extend_from_slice(&y[row * width..(row + 1) * width]);
    }
    let mut fu = Vec::with_capacity(cw * ch / 2);
    let mut fv = Vec::with_capacity(cw * ch / 2);
    for row in (start..ch).step_by(2) {
        fu.extend_from_slice(&u[row * cw..(row + 1) * cw]);
        fv.extend_from_slice(&v[row * cw..(row + 1) * cw]);
    }
    (fy, fu, fv)
}

/// Re-interleave two half-height field plane sets into one full-height
/// frame (top → even rows, bottom → odd rows), mirroring the decoder's
/// §C.4.4 output pairing.
fn interleave_planes(top: &[u8], bottom: &[u8], width: usize) -> Vec<u8> {
    debug_assert_eq!(top.len(), bottom.len());
    let field_rows = top.len() / width;
    let mut out = vec![0u8; top.len() * 2];
    for r in 0..field_rows {
        out[(2 * r) * width..(2 * r + 1) * width].copy_from_slice(&top[r * width..(r + 1) * width]);
        out[(2 * r + 1) * width..(2 * r + 2) * width]
            .copy_from_slice(&bottom[r * width..(r + 1) * width]);
    }
    out
}

/// Encode one intra picture's `slice_data()` bits into `sw` using the
/// supplied picture-sized encoder, returning the pre-deblock recon
/// planes + per-MB deblock facts. Mirrors the `encode_idr` MB loop.
fn encode_i_slice_data(
    enc: &Encoder,
    src: &YuvFrame<'_>,
    sw: &mut BitWriter,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<MbDeblockInfo>) {
    let width_mbs = enc.cfg.width / 16;
    let height_mbs = enc.cfg.height / 16;
    let chroma_width = (enc.cfg.width / 2) as usize;
    let chroma_height = (enc.cfg.height / 2) as usize;
    let mut recon_y = vec![0u8; (enc.cfg.width * enc.cfg.height) as usize];
    let mut recon_u = vec![0u8; chroma_width * chroma_height];
    let mut recon_v = vec![0u8; chroma_width * chroma_height];
    let qp_y = enc.cfg.qp;
    let qp_c = qp_y_to_qp_c_with_bd_offset(qp_y, 0, qp_bd_offset(enc.cfg.bit_depth_chroma_minus8));
    let mut nc_grid = CavlcNcGrid::new(width_mbs, height_mbs);
    let mut intra_grid = IntraGrid::new(width_mbs as usize, height_mbs as usize);
    let mut infos = vec![MbDeblockInfo::default(); (width_mbs * height_mbs) as usize];
    for mb_y in 0..height_mbs as usize {
        for mb_x in 0..width_mbs as usize {
            let dbl = enc.encode_mb(
                src,
                mb_x,
                mb_y,
                qp_y,
                qp_c,
                chroma_width,
                chroma_height,
                &mut recon_y,
                &mut recon_u,
                &mut recon_v,
                sw,
                &mut nc_grid,
                &mut intra_grid,
            );
            infos[mb_y * width_mbs as usize + mb_x] = dbl;
        }
    }
    (recon_y, recon_u, recon_v, infos)
}

/// Encode one P picture's `slice_data()` bits into `sw` against a
/// single same-sized reference, returning the pre-deblock recon planes
/// and per-MB deblock facts. Mirrors the `encode_p` MB loop, with the
/// §7.3.4 CAVLC `mb_skip_run` accounting included.
fn encode_p_slice_data(
    enc: &Encoder,
    src: &YuvFrame<'_>,
    prev: &EncodedFrameRef<'_>,
    sw: &mut BitWriter,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<MbDeblockInfo>) {
    let width_mbs = enc.cfg.width / 16;
    let height_mbs = enc.cfg.height / 16;
    let chroma_width = (enc.cfg.width / 2) as usize;
    let chroma_height = (enc.cfg.height / 2) as usize;
    let mut recon_y = vec![0u8; (enc.cfg.width * enc.cfg.height) as usize];
    let mut recon_u = vec![0u8; chroma_width * chroma_height];
    let mut recon_v = vec![0u8; chroma_width * chroma_height];
    let qp_y = enc.cfg.qp;
    let qp_c = qp_y_to_qp_c_with_bd_offset(qp_y, 0, qp_bd_offset(enc.cfg.bit_depth_chroma_minus8));
    let mut nc_grid = CavlcNcGrid::new(width_mbs, height_mbs);
    let mut intra_grid = IntraGrid::new(width_mbs as usize, height_mbs as usize);
    let mut mv_grid = MvGrid::new(width_mbs as usize, height_mbs as usize);
    let mut infos = vec![MbDeblockInfo::default(); (width_mbs * height_mbs) as usize];
    let mut pending_skip: u32 = 0;
    for mb_y in 0..height_mbs as usize {
        for mb_x in 0..width_mbs as usize {
            let dbl = enc.encode_p_mb_with_intra_fallback(
                src,
                prev,
                mb_x,
                mb_y,
                qp_y,
                qp_c,
                chroma_width,
                chroma_height,
                &mut recon_y,
                &mut recon_u,
                &mut recon_v,
                sw,
                &mut nc_grid,
                &mut intra_grid,
                &mut mv_grid,
                &mut pending_skip,
            );
            infos[mb_y * width_mbs as usize + mb_x] = dbl;
        }
    }
    // §7.3.4 — flush a trailing skip run.
    if pending_skip > 0 {
        sw.ue(pending_skip);
    }
    (recon_y, recon_u, recon_v, infos)
}

/// Encode an interlaced 4:2:0 sequence with PAFF field pictures. See
/// the module docs for the stream layout. `frames` are full-height
/// (Y, Cb, Cr) planes, one entry per interlaced frame.
pub fn encode_paff_sequence(cfg: &PaffConfig, frames: &[(&[u8], &[u8], &[u8])]) -> PaffEncoded {
    assert!(cfg.width % 16 == 0, "width must be MB-aligned");
    assert!(
        cfg.frame_height % 32 == 0,
        "frame height must be a multiple of 32 (each field MB-aligned)",
    );
    assert!(!frames.is_empty());
    assert!(
        !cfg.p_fields || cfg.frame_picture_indices.is_empty(),
        "frame pictures inside a P-field sequence are out of scope",
    );
    assert!(
        !cfg.frame_picture_indices.contains(&0),
        "frame 0 is the IDR field pair",
    );
    assert!(
        !cfg.cross_parity_first_bottom || cfg.p_fields,
        "cross_parity_first_bottom is a P-field axis",
    );
    assert!(
        !cfg.idr_frame_first || cfg.p_fields,
        "idr_frame_first is a P-field axis",
    );
    assert!(
        !(cfg.idr_frame_first && cfg.cross_parity_first_bottom),
        "idr_frame_first replaces frame 0's field pair",
    );

    let width = cfg.width as usize;
    let frame_h = cfg.frame_height as usize;
    let field_h = cfg.frame_height / 2;
    let width_mbs = cfg.width / 16;
    let frame_h_mbs = cfg.frame_height / 16;
    let log2_max_frame_num_minus4: u32 = 4;
    let log2_max_poc_lsb_minus4: u32 = 4;
    let frame_num_bits = log2_max_frame_num_minus4 + 4;
    let poc_lsb_bits = log2_max_poc_lsb_minus4 + 4;

    // Field-sized encoder (all field pictures) + frame-sized encoder
    // (mixed FRAME pictures). §A.2.1 bars interlace from Baseline —
    // Main (77) it is; CAVLC only.
    let mk_cfg = |h: u32| {
        let mut c = EncoderConfig::new(cfg.width, h);
        c.qp = cfg.qp;
        c.profile_idc = 77;
        c.max_num_ref_frames = 2;
        c
    };
    let field_enc = Encoder::new(mk_cfg(field_h));
    let frame_enc = Encoder::new(mk_cfg(cfg.frame_height));
    // §8.4.1.4 Table 8-10 — dedicated field encoder for the
    // cross-parity picture (bottom field referencing the top field):
    // chroma predictors add +2 to the vertical MV.
    let xpar_enc = Encoder::new({
        let mut c = mk_cfg(field_h);
        c.table_8_10_cy_offset = 2;
        c
    });

    // SPS (interlaced, FrameHeightInMbs) + PPS.
    let sps_rbsp = build_baseline_sps_rbsp(&BaselineSpsConfig {
        seq_parameter_set_id: 0,
        level_idc: min_level_idc_for_picture_size(width_mbs, frame_h_mbs),
        width_in_mbs: width_mbs,
        height_in_mbs: frame_h_mbs,
        log2_max_frame_num_minus4,
        log2_max_poc_lsb_minus4,
        max_num_ref_frames: 2,
        profile_idc: 77,
        chroma_format_idc: 1,
        seq_scaling_lists: None,
        interlaced_fields: true,
    });
    let pps_rbsp = build_baseline_pps_rbsp(&BaselinePpsConfig {
        pic_scaling_lists: None,
        chroma_format_idc: 1,
        pic_parameter_set_id: 0,
        seq_parameter_set_id: 0,
        pic_init_qp_minus26: cfg.qp - 26,
        chroma_qp_index_offset: 0,
        weighted_pred_flag: false,
        weighted_bipred_idc: 0,
        entropy_coding_mode_flag: false,
        transform_8x8_mode_flag: false,
    });
    let mut stream: Vec<u8> = Vec::new();
    stream.extend_from_slice(&build_nal_unit(3, NalUnitType::Sps, &sps_rbsp));
    stream.extend_from_slice(&build_nal_unit(3, NalUnitType::Pps, &pps_rbsp));

    // Last reconstructed field of each parity — the P references.
    let mut last_top: Option<ReconField> = None;
    let mut last_bottom: Option<ReconField> = None;
    let mut recon_frames = Vec::with_capacity(frames.len());

    for (k, &(fy, fu, fv)) in frames.iter().enumerate() {
        assert_eq!(fy.len(), width * frame_h);
        let frame_num = (k as u32) % (1 << frame_num_bits);

        if k == 0 && cfg.idr_frame_first {
            // ---- IDR full-height FRAME picture (field_pic_flag = 0). ----
            let src = YuvFrame {
                width: cfg.width,
                height: cfg.frame_height,
                y: fy,
                u: fu,
                v: fv,
            };
            let mut sw = BitWriter::new();
            write_idr_i_slice_header(
                &mut sw,
                &IdrSliceHeaderConfig {
                    first_mb_in_slice: 0,
                    slice_type_raw: 7,
                    pic_parameter_set_id: 0,
                    frame_num: 0,
                    frame_num_bits,
                    idr_pic_id: 0,
                    pic_order_cnt_lsb: 0,
                    poc_lsb_bits,
                    slice_qp_delta: 0,
                    disable_deblocking_filter_idc: 0,
                    slice_alpha_c0_offset_div2: 0,
                    slice_beta_offset_div2: 0,
                    field: FieldPicSignal::FramePicture,
                    idr: true,
                    nal_ref_idc: 3,
                },
            );
            let (mut ry, mut ru, mut rv, infos) = encode_i_slice_data(&frame_enc, &src, &mut sw);
            sw.rbsp_trailing_bits();
            stream.extend_from_slice(&build_nal_unit(3, NalUnitType::SliceIdr, &sw.into_bytes()));
            deblock_recon_with_chroma_array_type(
                cfg.width,
                cfg.frame_height,
                cfg.width / 2,
                cfg.frame_height / 2,
                &mut ry,
                &mut ru,
                &mut rv,
                &infos,
                0,
                width_mbs,
                frame_h_mbs,
                1,
            );
            // §8.2.4.2.5 — either parity field of this stored frame is
            // a distinct reference picture: the next frame's P fields
            // reference its parity rows (what a decoder materialises
            // as a field view of the stored frame). Both field POCs
            // equal the frame's (TopFOC == BotFOC == 0 for the IDR).
            let (ty, tu, tv) = extract_field(&ry, &ru, &rv, width, frame_h, false);
            let (by, bu, bv) = extract_field(&ry, &ru, &rv, width, frame_h, true);
            last_top = Some((ty, tu, tv, 0));
            last_bottom = Some((by, bu, bv, 0));
            recon_frames.push((ry, ru, rv));
            continue;
        }

        if cfg.frame_picture_indices.contains(&k) {
            // ---- Full-height I FRAME picture (field_pic_flag = 0). ----
            let src = YuvFrame {
                width: cfg.width,
                height: cfg.frame_height,
                y: fy,
                u: fu,
                v: fv,
            };
            let mut sw = BitWriter::new();
            write_idr_i_slice_header(
                &mut sw,
                &IdrSliceHeaderConfig {
                    first_mb_in_slice: 0,
                    slice_type_raw: 7,
                    pic_parameter_set_id: 0,
                    frame_num,
                    frame_num_bits,
                    idr_pic_id: 0,
                    pic_order_cnt_lsb: (2 * k as u32) % (1 << poc_lsb_bits),
                    poc_lsb_bits,
                    slice_qp_delta: 0,
                    disable_deblocking_filter_idc: 0,
                    slice_alpha_c0_offset_div2: 0,
                    slice_beta_offset_div2: 0,
                    field: FieldPicSignal::FramePicture,
                    idr: false,
                    nal_ref_idc: 2,
                },
            );
            let (mut ry, mut ru, mut rv, infos) = encode_i_slice_data(&frame_enc, &src, &mut sw);
            sw.rbsp_trailing_bits();
            stream.extend_from_slice(&build_nal_unit(
                2,
                NalUnitType::SliceNonIdr,
                &sw.into_bytes(),
            ));
            deblock_recon_with_chroma_array_type(
                cfg.width,
                cfg.frame_height,
                cfg.width / 2,
                cfg.frame_height / 2,
                &mut ry,
                &mut ru,
                &mut rv,
                &infos,
                0,
                width_mbs,
                frame_h_mbs,
                1,
            );
            recon_frames.push((ry, ru, rv));
            // Fields of a stored FRAME as P references are out of
            // scope (asserted above), so last_top/last_bottom stay.
            continue;
        }

        // ---- Field pair: top (even rows) first, then bottom. ----
        let mut pair_recon: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::with_capacity(2);
        for bottom in [false, true] {
            let (sy, su, sv) = extract_field(fy, fu, fv, width, frame_h, bottom);
            let src = YuvFrame {
                width: cfg.width,
                height: field_h,
                y: &sy,
                u: &su,
                v: &sv,
            };
            let field_signal = if bottom {
                FieldPicSignal::BottomField
            } else {
                FieldPicSignal::TopField
            };
            let poc_lsb = (2 * k as u32 + u32::from(bottom)) % (1 << poc_lsb_bits);
            let is_idr = k == 0 && !bottom;
            // Same-parity reference field of the previous frame — what
            // §8.2.4.2.5 puts at RefPicList0[0] for this field.
            let ref_field = if bottom { &last_bottom } else { &last_top };
            // Cross-parity axis: frame 0's bottom field P-references
            // the IDR top field (opposite parity) instead of being an
            // I field. Every other P field references the same-parity
            // field of the previous frame.
            let cross = cfg.cross_parity_first_bottom && k == 0 && bottom;
            let (as_p, p_enc, p_ref) = if cross {
                (last_top.is_some(), &xpar_enc, &last_top)
            } else {
                (cfg.p_fields && ref_field.is_some(), &field_enc, ref_field)
            };

            let mut sw = BitWriter::new();
            // §8.5.6 — every MB of a field picture is a field MB: the
            // CAVLC residual writer must emit the Table 8-13 FIELD
            // scan. (The 8x8 transform would need the Table 8-14 field
            // scan — not wired; the configs above leave it disabled.)
            debug_assert!(!field_enc.cfg.transform_8x8);
            sw.set_field_scan(true);
            if as_p {
                write_p_slice_header(
                    &mut sw,
                    &PSliceHeaderConfig {
                        first_mb_in_slice: 0,
                        slice_type_raw: 5,
                        pic_parameter_set_id: 0,
                        frame_num,
                        frame_num_bits,
                        pic_order_cnt_lsb: poc_lsb,
                        poc_lsb_bits,
                        slice_qp_delta: 0,
                        disable_deblocking_filter_idc: 0,
                        slice_alpha_c0_offset_div2: 0,
                        slice_beta_offset_div2: 0,
                        nal_ref_idc: 2,
                        cabac: None,
                        field: field_signal,
                    },
                );
            } else {
                write_idr_i_slice_header(
                    &mut sw,
                    &IdrSliceHeaderConfig {
                        first_mb_in_slice: 0,
                        slice_type_raw: 7,
                        pic_parameter_set_id: 0,
                        frame_num,
                        frame_num_bits,
                        idr_pic_id: 0,
                        pic_order_cnt_lsb: poc_lsb,
                        poc_lsb_bits,
                        slice_qp_delta: 0,
                        disable_deblocking_filter_idc: 0,
                        slice_alpha_c0_offset_div2: 0,
                        slice_beta_offset_div2: 0,
                        field: field_signal,
                        idr: is_idr,
                        nal_ref_idc: if is_idr { 3 } else { 2 },
                    },
                );
            }

            let (mut ry, mut ru, mut rv, infos) = if as_p {
                let (py, pu, pv, ppoc) = p_ref.as_ref().unwrap();
                let prev = EncodedFrameRef {
                    width: cfg.width,
                    height: field_h,
                    recon_y: py,
                    recon_u: pu,
                    recon_v: pv,
                    partition_mvs: &[],
                    pic_order_cnt: *ppoc,
                };
                encode_p_slice_data(p_enc, &src, &prev, &mut sw)
            } else {
                encode_i_slice_data(&field_enc, &src, &mut sw)
            };
            sw.rbsp_trailing_bits();
            let (ref_idc, nal_type) = if is_idr {
                (3, NalUnitType::SliceIdr)
            } else {
                (2, NalUnitType::SliceNonIdr)
            };
            stream.extend_from_slice(&build_nal_unit(ref_idc, nal_type, &sw.into_bytes()));

            // §8.7 with field_pic = 1 — the reference fields other
            // pictures use MUST be the post-filter samples.
            deblock_recon_field(
                cfg.width,
                field_h,
                cfg.width / 2,
                field_h / 2,
                &mut ry,
                &mut ru,
                &mut rv,
                &infos,
                0,
                width_mbs,
                field_h / 16,
                1,
            );
            let slot = (ry.clone(), ru.clone(), rv.clone(), poc_lsb as i32);
            if bottom {
                last_bottom = Some(slot);
            } else {
                last_top = Some(slot);
            }
            pair_recon.push((ry, ru, rv));
        }
        let (ty, tu, tv) = &pair_recon[0];
        let (by, bu, bv) = &pair_recon[1];
        recon_frames.push((
            interleave_planes(ty, by, width),
            interleave_planes(tu, bu, width / 2),
            interleave_planes(tv, bv, width / 2),
        ));
    }

    PaffEncoded {
        annex_b: stream,
        recon_frames,
    }
}
