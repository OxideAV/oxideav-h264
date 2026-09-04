//! Round-456 — MBAFF frame encoder, **CABAC** entropy path.
//!
//! The CABAC twin of [`super::mbaff`]: interlaced frames are coded as
//! MBAFF frame pictures (§7.4.2.1.1 `frame_mbs_only_flag = 0`,
//! `mb_adaptive_frame_field_flag = 1`, `field_pic_flag = 0`) with the
//! PPS `entropy_coding_mode_flag = 1`. Every context-index increment
//! that the §9.3.3.1.1.x processes derive from neighbouring
//! macroblocks runs through the decoder's own §6.4.10 / §6.4.12.2
//! Table 6-4 machinery ([`crate::macroblock_layer::CabacNeighbourGrid`]
//! and its `cabac_*_cond_terms` / `cabac_mvd_abs_sum` helpers) so the
//! encoder's context selection is, by construction, the one the
//! byte-exact-on-JVT-conformance decoder performs:
//!
//! * §9.3.3.1.1.2 `mb_field_decoding_flag` — ctxIdx 70..=72 from the
//!   left / above *pairs'* field flags (skipped pairs carry their
//!   inferred flag, exactly as the decoder records them).
//! * §9.3.3.1.1.1 `mb_skip_flag` — mbAddrA / mbAddrB resolved through
//!   Table 6-4 under the pair flag the decoder holds at that point:
//!   the coded flag for a bottom MB whose top was coded, else the
//!   §7.4.4 inference (left pair, above pair, frame). A top MB's
//!   skip context is therefore derived under the *inferred* flag even
//!   when the pair ends up coded with a different one (the decoder
//!   only learns the real flag from the bottom MB).
//! * §9.3.3.1.1.6 `ref_idx_l0` — present for every field MB (§7.3.5.1:
//!   `mb_field_decoding_flag != field_pic_flag`) with the eq. 9-12
//!   `refIdxZeroFlagN` frame-reads-field threshold.
//! * §9.3.3.1.1.7 `mvd_l0` — eq. 9-15 / 9-16 vertical |mvd| scaling
//!   across frame/field pair boundaries.
//! * §9.3.3.1.1.4 `coded_block_pattern` — per-bin external luma
//!   probes through Table 6-4 (a mixed pair can route bins 0 and 2 of
//!   the A neighbour to different macroblocks).
//! * §9.3.3.1.1.9 `coded_block_flag` — §6.4.11.4 / §6.4.11.5 block
//!   neighbours across pair boundaries.
//! * Field macroblocks code their residual under the Table 9-34
//!   FIELD ctxIdxOffset families (277 / 338) and the §8.5.6 Table
//!   8-13 field scan; §7.3.4 codes `end_of_slice_flag` only after the
//!   bottom macroblock of a pair.
//!
//! Pixel work reuses the round-453 virtual-neighbourhood scheme
//! (frame MBs run in frame geometry, field MBs in a half-height
//! same-parity field geometry, with Table 6-4-resolved neighbouring
//! samples and motion cells patched in per MB). Mode set: I pictures
//! all-Intra_16x16; P pictures P_Skip / P_L0_16x16 with an optional
//! Intra_16x16 fallback ([`MbaffCabacConfig::intra_in_p`]).

use crate::cabac_ctx::{
    BlockType, CabacContexts, CbpMbaffProbe, MvdComponent, NeighbourCtx, SliceKind,
};
use crate::encoder::cabac_engine::CabacEncoder;
use crate::encoder::cabac_path::{
    build_inter_pred_chroma_local, build_inter_pred_luma_local, chroma_residual_dc_only,
    encode_chroma_inter_420, encode_chroma_intra16x16_420, pick_chroma_mode, pick_intra16x16_mode,
    quantize_intra16x16_luma, reconstruct_intra16x16_luma,
};
use crate::encoder::cabac_syntax::{
    encode_coded_block_pattern, encode_end_of_slice_flag, encode_intra_chroma_pred_mode,
    encode_mb_qp_delta, encode_mb_skip_flag, encode_mb_type_i, encode_mb_type_p, encode_mvd_lx,
    encode_ref_idx_lx, encode_residual_block_cabac_field,
};
use crate::encoder::cavlc::ZZ_TO_FIELD_16;
use crate::encoder::deblock::{
    chroma_nz_mask_from_blocks, deblock_recon_mbaff, luma_nz_mask_from_blocks, MbDeblockInfo,
};
use crate::encoder::mbaff::{
    adaptive_pair_is_field, extract_field_planes, fill_neighbour_grids, patch_neighbour_samples,
    ref_poc_key, virtual_pos, PairMode, PicState, Planes, VirtualEnc,
};
use crate::encoder::me::search_quarter_pel_16x16;
use crate::encoder::nal::build_nal_unit;
use crate::encoder::pps::{build_baseline_pps_rbsp, BaselinePpsConfig};
use crate::encoder::slice::{
    write_idr_i_slice_header, write_p_slice_header, CabacSliceParams, FieldPicSignal,
    IdrSliceHeaderConfig, PSliceHeaderConfig,
};
use crate::encoder::sps::{build_baseline_sps_rbsp, BaselineSpsConfig};
use crate::encoder::transform::{forward_core_4x4, quantize_4x4_w, zigzag_scan_4x4};
use crate::encoder::{
    min_level_idc_for_picture_size, mvp_for_16x16, p_skip_mv, BitWriter, CavlcNcGrid,
    EncoderConfig, IntraGrid, MvGrid, MvGridSlot, ScalingMatrixMode, YuvFrame,
};
use crate::macroblock_layer::{
    cabac_cbf_cond_terms, cabac_mvd_abs_sum, cabac_ref_idx_cond_terms, CabacMbNeighbourInfo,
    CabacNeighbourGrid,
};
use crate::mb_address::{mbaff_mb_to_sample_xy, mbaff_pair_neighbour_addrs};
use crate::mv_deriv::Mv;
use crate::nal::NalUnitType;
use crate::slice_data::infer_pair_field_flag;
use crate::transform::{inverse_transform_4x4, qp_bd_offset, qp_y_to_qp_c_with_bd_offset};

/// Configuration for [`encode_mbaff_cabac_sequence`].
#[derive(Debug, Clone)]
pub struct MbaffCabacConfig {
    /// Luma width in samples (multiple of 16).
    pub width: u32,
    /// Frame luma height in samples (multiple of 32).
    pub frame_height: u32,
    /// Slice QP_Y (`pic_init_qp_minus26`, `slice_qp_delta = 0`).
    pub qp: i32,
    /// Frame/field decision policy per macroblock pair.
    pub pair_mode: PairMode,
    /// Frames after the IDR are P pictures (else every frame is an I
    /// picture).
    pub p_frames: bool,
    /// Allow an Intra_16x16 macroblock inside P pictures when its
    /// prediction beats the inter candidate — exercises the P-slice
    /// intra `mb_type` prefix, `intra_chroma_pred_mode` and the
    /// Intra_16x16 residual contexts against inter MBAFF neighbours.
    pub intra_in_p: bool,
}

/// Output of [`encode_mbaff_cabac_sequence`].
#[derive(Debug, Clone)]
pub struct MbaffCabacEncoded {
    /// Annex B stream: SPS, PPS, one slice NAL per frame.
    pub annex_b: Vec<u8>,
    /// Deblocked reconstruction of every frame (Y, Cb, Cr) in frame
    /// (interleaved-field) geometry — what a conforming decoder outputs.
    pub recon_frames: Vec<Planes>,
    /// Field-coded macroblock pairs across the sequence.
    pub field_pairs: usize,
    /// Frame-coded macroblock pairs across the sequence.
    pub frame_pairs: usize,
    /// P_Skip macroblocks across the sequence.
    pub skipped_mbs: usize,
    /// Intra_16x16 macroblocks coded inside P pictures.
    pub intra_mbs_in_p: usize,
    /// Fully-skipped pairs re-encoded under the §7.4.4 inferred flag.
    pub inference_reencodes: usize,
}

/// One §9.3.3.1.1.2 `mb_field_decoding_flag` bin (ctxIdxOffset 70).
fn encode_mb_field_decoding_flag(
    enc: &mut CabacEncoder,
    ctxs: &mut CabacContexts,
    cab: &CabacNeighbourGrid,
    addr: usize,
    flag: bool,
) {
    let [a_top, b_top, _c, _d] = mbaff_pair_neighbour_addrs(addr as u32, cab.width_in_mbs);
    // condTermFlagN = 1 iff the neighbouring PAIR is available and
    // field-coded (skipped pairs carry their inferred / patched flag).
    let cond = |o: Option<u32>| -> u32 {
        o.and_then(|n| cab.mbs.get(n as usize))
            .map(|i| u32::from(i.available && i.mb_field_decoding_flag))
            .unwrap_or(0)
    };
    let ctx_idx = 70 + cond(a_top) + cond(b_top);
    enc.encode_decision(ctxs.at_mut(ctx_idx as usize), u8::from(flag));
}

/// The minimal `mb_skip_flag` neighbour snapshot the decoder builds at
/// the top of its slice-data loop (§9.3.3.1.1.1 with the §6.4.10 MBAFF
/// mbAddrA / mbAddrB under the grid's current view of the pair flag).
fn skip_neighbour_ctx(cab: &CabacNeighbourGrid, addr: usize) -> NeighbourCtx {
    let mut n = NeighbourCtx::default();
    let (a, b) = cab.neighbour_mb_addrs_mbaff(addr as u32, true);
    if let Some(info) = a.and_then(|x| cab.mbs.get(x as usize)) {
        if info.available {
            n.available_left = true;
            n.mb_skip_flag_left = info.is_skip;
        }
    }
    if let Some(info) = b.and_then(|x| cab.mbs.get(x as usize)) {
        if info.available {
            n.available_above = true;
            n.mb_skip_flag_above = info.is_skip;
        }
    }
    n
}

/// The full per-MB [`NeighbourCtx`] — mirror of the decoder's
/// slice-data construction (MB-level facts from the Table 6-4 A / B
/// macroblocks plus the §9.3.3.1.1.4 per-bin luma CBP probes).
fn full_neighbour_ctx(cab: &CabacNeighbourGrid, addr: usize) -> NeighbourCtx {
    let mut n = NeighbourCtx::default();
    let (a, b) = cab.neighbour_mb_addrs_mbaff(addr as u32, true);
    if let Some(info) = a.and_then(|x| cab.mbs.get(x as usize)) {
        if info.available {
            n.available_left = true;
            n.mb_skip_flag_left = info.is_skip;
            n.left_is_i_pcm = info.is_i_pcm;
            n.left_is_p_or_b_skip = info.is_skip;
            n.left_inter = !info.is_intra;
            n.left_cbp_luma = info.coded_block_pattern_luma;
            n.left_cbp_chroma = info.coded_block_pattern_chroma;
            n.left_is_i_nxn = info.is_i_nxn;
            n.left_is_b_skip_or_direct = info.is_b_skip_or_direct;
            n.left_intra_chroma_pred_mode_nonzero = info.intra_chroma_pred_mode != 0;
            n.left_transform_8x8 = info.transform_size_8x8_flag;
        }
    }
    if let Some(info) = b.and_then(|x| cab.mbs.get(x as usize)) {
        if info.available {
            n.available_above = true;
            n.mb_skip_flag_above = info.is_skip;
            n.above_is_i_pcm = info.is_i_pcm;
            n.above_is_p_or_b_skip = info.is_skip;
            n.above_inter = !info.is_intra;
            n.above_cbp_luma = info.coded_block_pattern_luma;
            n.above_cbp_chroma = info.coded_block_pattern_chroma;
            n.above_is_i_nxn = info.is_i_nxn;
            n.above_is_b_skip_or_direct = info.is_b_skip_or_direct;
            n.above_intra_chroma_pred_mode_nonzero = info.intra_chroma_pred_mode != 0;
            n.above_transform_8x8 = info.transform_size_8x8_flag;
        }
    }
    // §9.3.3.1.1.4 + §6.4.11.2 — external luma CBP probes (A: bins 0 /
    // 2 at (-1, 0) / (-1, 8); B: bins 0 / 1 at (0, -1) / (8, -1)).
    let probe = |xn: i32, yn: i32| -> CbpMbaffProbe {
        match cab.neighbour_block_loc(addr as u32, xn, yn, 16, 16) {
            Some((n_addr, wx, wy)) => match cab.mbs.get(n_addr as usize) {
                Some(info) if info.available => {
                    let blk8n = ((wy / 8) * 2 + (wx / 8)) as u8;
                    CbpMbaffProbe {
                        available: true,
                        is_i_pcm: info.is_i_pcm,
                        is_skip: info.is_skip,
                        cbp_bit_set: ((info.coded_block_pattern_luma >> blk8n) & 1) != 0,
                    }
                }
                _ => CbpMbaffProbe::default(),
            },
            None => CbpMbaffProbe::default(),
        }
    };
    let d = CbpMbaffProbe::default();
    n.cbp_luma_mbaff = Some([
        [probe(-1, 0), d, probe(-1, 8), d],
        [probe(0, -1), probe(8, -1), d, d],
    ]);
    n
}

/// §6.4.3 — upper-left 4x4 unit `(bx, by)` of luma4x4BlkIdx `idx`.
fn blk_xy(idx: usize) -> (usize, usize) {
    let hi = idx / 4;
    let lo = idx % 4;
    ((hi % 2) * 2 + (lo % 2), (hi / 2) * 2 + (lo / 2))
}

/// §8.5.6 — a 16-entry zig-zag list re-permuted into the Table 8-13
/// field scan for field macroblocks (identity for frame MBs).
fn scan16(zz: &[i32; 16], field: bool) -> [i32; 16] {
    if !field {
        return *zz;
    }
    let mut out = [0i32; 16];
    for (k, &src) in ZZ_TO_FIELD_16.iter().enumerate() {
        out[k] = zz[src];
    }
    out
}

/// §8.5.6 — AC-only (15-entry, slot `s` = scan position `s + 1`)
/// zig-zag list re-permuted into the field scan for field MBs.
fn scan15(zz_ac: &[i32; 16], field: bool) -> [i32; 15] {
    let mut out = [0i32; 15];
    if field {
        for k in 1..16 {
            out[k - 1] = zz_ac[ZZ_TO_FIELD_16[k] - 1];
        }
    } else {
        out.copy_from_slice(&zz_ac[..15]);
    }
    out
}

/// Per-slice CABAC coding state.
struct CabacState {
    enc: CabacEncoder,
    ctxs: CabacContexts,
    /// §9.3.3.1.1.5 rolling flag for the next `mb_qp_delta` bin 0.
    prev_qp_delta_nonzero: bool,
    /// §7.3.4 `prevMbSkipped`.
    prev_mb_skipped: bool,
    /// §7.3.4 / §7.4.4 — the pair flag once coded (`None` while the
    /// pair has only produced skipped MBs).
    pending_pair_flag: Option<bool>,
}

struct MbCtx<'a> {
    qp_y: i32,
    qp_c: i32,
    prev_poc: i32,
    /// P references per geometry: `[frame, top field, bottom field]`.
    refs: Option<&'a [Planes; 3]>,
    sources_field: &'a [Planes; 2],
    src_frame: YuvFrame<'a>,
    w4: [i32; 16],
    intra_in_p: bool,
}

struct MbOut {
    skipped: bool,
    intra_in_p: bool,
}

/// Intra_16x16 candidate (prediction + quantised residual + recon
/// facts), computed before any bin is emitted.
struct IntraCand {
    luma_mode: u32,
    chroma_mode: u32,
    luma_pred: [i32; 256],
    luma_dc: [i32; 16],
    luma_ac_quant: [[i32; 16]; 16],
    luma_ac_scan: [[i32; 16]; 16],
    cbp_luma: u8,
    cbp_chroma: u8,
    cb_dc: [i32; 4],
    cr_dc: [i32; 4],
    cb_ac_scan: [[i32; 16]; 4],
    cr_ac_scan: [[i32; 16]; 4],
    cb_res: [i32; 64],
    cr_res: [i32; 64],
    chroma_pred_u: [i32; 64],
    chroma_pred_v: [i32; 64],
    sad: u64,
}

#[allow(clippy::too_many_arguments)]
fn intra16x16_candidate(
    src: &YuvFrame<'_>,
    ve: &VirtualEnc,
    vx: usize,
    vy: usize,
    qp_y: i32,
    qp_c: i32,
    w4: &[i32; 16],
) -> IntraCand {
    let width = src.width as usize;
    let height = src.height as usize;
    let chroma_w = width / 2;
    let chroma_h = height / 2;
    let (luma_mode, luma_pred) = pick_intra16x16_mode(src, &ve.vy, width, height, vx, vy);
    let (chroma_mode, chroma_pred_u, chroma_pred_v) =
        pick_chroma_mode(src.u, src.v, &ve.vu, &ve.vv, chroma_w, chroma_h, vx, vy);
    let mut residual = [0i32; 256];
    let mut sad: u64 = 0;
    for j in 0..16usize {
        for i in 0..16usize {
            let s = src.y[(vy * 16 + j) * width + vx * 16 + i] as i32;
            let d = s - luma_pred[j * 16 + i];
            residual[j * 16 + i] = d;
            sad += d.unsigned_abs() as u64;
        }
    }
    let (luma_dc, luma_ac_quant, luma_ac_scan, any_ac_nz) =
        quantize_intra16x16_luma(&residual, qp_y, false, w4);
    let cbp_luma: u8 = if any_ac_nz { 15 } else { 0 };
    let (cb_dc, cb_ac_scan, _cb_q, cb_dc_nz, cb_ac_nz, cb_res) = encode_chroma_intra16x16_420(
        src.u,
        &ve.vu,
        chroma_w,
        chroma_h,
        vx,
        vy,
        chroma_mode,
        qp_c,
        false,
        w4,
    );
    let (cr_dc, cr_ac_scan, _cr_q, cr_dc_nz, cr_ac_nz, cr_res) = encode_chroma_intra16x16_420(
        src.v,
        &ve.vv,
        chroma_w,
        chroma_h,
        vx,
        vy,
        chroma_mode,
        qp_c,
        false,
        w4,
    );
    let cbp_chroma: u8 = if cb_ac_nz || cr_ac_nz {
        2
    } else if cb_dc_nz || cr_dc_nz {
        1
    } else {
        0
    };
    IntraCand {
        luma_mode: luma_mode as u32,
        chroma_mode: chroma_mode as u32,
        luma_pred,
        luma_dc,
        luma_ac_quant,
        luma_ac_scan,
        cbp_luma,
        cbp_chroma,
        cb_dc,
        cr_dc,
        cb_ac_scan,
        cr_ac_scan,
        cb_res,
        cr_res,
        chroma_pred_u,
        chroma_pred_v,
        sad,
    }
}

/// Emit an Intra_16x16 macroblock's `macroblock_layer()` bins (mb_type
/// already selected by the caller) and reconstruct it into the scratch
/// planes; fills the running CABAC slot `cur`.
#[allow(clippy::too_many_arguments)]
fn emit_intra16x16(
    cs: &mut CabacState,
    cab: &CabacNeighbourGrid,
    cur: &mut CabacMbNeighbourInfo,
    nctx: &NeighbourCtx,
    addr: usize,
    field: bool,
    is_p: bool,
    cand: &IntraCand,
    ve: &mut VirtualEnc,
    vx: usize,
    vy: usize,
    qp_y: i32,
    qp_c: i32,
    w4: &[i32; 16],
) {
    // §7.4.5.1 Table 7-11 — Intra_16x16 row: 1 + 4·group + pred mode.
    let group = match (cand.cbp_luma, cand.cbp_chroma) {
        (0, 0) => 0u32,
        (0, 1) => 1,
        (0, 2) => 2,
        (15, 0) => 3,
        (15, 1) => 4,
        (15, 2) => 5,
        _ => unreachable!("Intra_16x16 cbp_luma is 0 or 15"),
    };
    let row = 1 + group * 4 + cand.luma_mode;
    if is_p {
        encode_mb_type_p(&mut cs.enc, &mut cs.ctxs, nctx, 5 + row);
    } else {
        encode_mb_type_i(&mut cs.enc, &mut cs.ctxs, nctx, row);
    }
    encode_intra_chroma_pred_mode(&mut cs.enc, &mut cs.ctxs, nctx, cand.chroma_mode);
    // §7.3.5 — mb_qp_delta is always present for Intra_16x16.
    encode_mb_qp_delta(&mut cs.enc, &mut cs.ctxs, cs.prev_qp_delta_nonzero, 0);
    cs.prev_qp_delta_nonzero = false;

    cur.is_intra = true;
    cur.is_skip = false;
    cur.coded_block_pattern_luma = cand.cbp_luma;
    cur.coded_block_pattern_chroma = cand.cbp_chroma;
    cur.intra_chroma_pred_mode = cand.chroma_mode as u8;

    // §7.3.5.3 — Intra16x16DCLevel.
    let (ca, cb) = cabac_cbf_cond_terms(
        cab,
        addr as u32,
        cur,
        true,
        BlockType::Luma16x16Dc,
        0,
        false,
        1,
    );
    let dc_list = scan16(&zigzag_scan_4x4(&cand.luma_dc), field);
    cur.cbf_luma_16x16_dc = encode_residual_block_cabac_field(
        &mut cs.enc,
        &mut cs.ctxs,
        BlockType::Luma16x16Dc,
        &dc_list,
        16,
        Some(ca),
        Some(cb),
        false,
        1,
        field,
    );
    if cand.cbp_luma == 15 {
        for blk in 0..16usize {
            let (ca, cb) = cabac_cbf_cond_terms(
                cab,
                addr as u32,
                cur,
                true,
                BlockType::Luma16x16Ac,
                blk as u8,
                false,
                1,
            );
            let list = scan15(&cand.luma_ac_scan[blk], field);
            cur.cbf_luma_16x16_ac[blk] = encode_residual_block_cabac_field(
                &mut cs.enc,
                &mut cs.ctxs,
                BlockType::Luma16x16Ac,
                &list,
                15,
                Some(ca),
                Some(cb),
                false,
                1,
                field,
            );
        }
    }
    emit_chroma_420(
        cs,
        cab,
        cur,
        addr,
        field,
        true,
        cand.cbp_chroma,
        &cand.cb_dc,
        &cand.cr_dc,
        &cand.cb_ac_scan,
        &cand.cr_ac_scan,
    );

    // Reconstruct into the scratch planes.
    let width = ve.enc.config().width as usize;
    let chroma_w = width / 2;
    reconstruct_intra16x16_luma(
        vx,
        vy,
        width,
        &cand.luma_pred,
        &cand.luma_dc,
        &cand.luma_ac_quant,
        cand.cbp_luma,
        qp_y,
        &mut ve.vy,
        w4,
    );
    let (res_u, res_v) = chroma_recon_residual(
        cand.cbp_chroma,
        &cand.cb_dc,
        &cand.cr_dc,
        &cand.cb_res,
        &cand.cr_res,
        qp_c,
        w4,
    );
    for j in 0..8usize {
        for i in 0..8usize {
            let off = (vy * 8 + j) * chroma_w + vx * 8 + i;
            ve.vu[off] = (cand.chroma_pred_u[j * 8 + i] + res_u[j * 8 + i]).clamp(0, 255) as u8;
            ve.vv[off] = (cand.chroma_pred_v[j * 8 + i] + res_v[j * 8 + i]).clamp(0, 255) as u8;
        }
    }
}

/// The decoder-side chroma residual for `cbp_chroma` ∈ {0, 1, 2}: AC +
/// DC, DC only (AC levels not transmitted), or nothing.
#[allow(clippy::too_many_arguments)]
fn chroma_recon_residual(
    cbp_chroma: u8,
    cb_dc: &[i32; 4],
    cr_dc: &[i32; 4],
    cb_res: &[i32; 64],
    cr_res: &[i32; 64],
    qp_c: i32,
    w4: &[i32; 16],
) -> ([i32; 64], [i32; 64]) {
    match cbp_chroma {
        2 => (*cb_res, *cr_res),
        1 => (
            chroma_residual_dc_only(cb_dc, qp_c, w4),
            chroma_residual_dc_only(cr_dc, qp_c, w4),
        ),
        _ => ([0i32; 64], [0i32; 64]),
    }
}

/// §7.3.5.3 — 4:2:0 chroma DC / AC residual bins with the
/// §9.3.3.1.1.9 cat-3 / cat-4 cond terms resolved through the MBAFF
/// grid; records the coded flags into `cur`.
#[allow(clippy::too_many_arguments)]
fn emit_chroma_420(
    cs: &mut CabacState,
    cab: &CabacNeighbourGrid,
    cur: &mut CabacMbNeighbourInfo,
    addr: usize,
    field: bool,
    is_intra: bool,
    cbp_chroma: u8,
    cb_dc: &[i32; 4],
    cr_dc: &[i32; 4],
    cb_ac_scan: &[[i32; 16]; 4],
    cr_ac_scan: &[[i32; 16]; 4],
) {
    if cbp_chroma > 0 {
        for (is_cr, dc) in [(false, cb_dc), (true, cr_dc)] {
            let (ca, cb) = cabac_cbf_cond_terms(
                cab,
                addr as u32,
                cur,
                is_intra,
                BlockType::ChromaDc,
                0,
                is_cr,
                1,
            );
            // Chroma DC keeps its own §8.5.11 2x2 order (no field
            // variant) — the list is emitted as built.
            let coded = encode_residual_block_cabac_field(
                &mut cs.enc,
                &mut cs.ctxs,
                BlockType::ChromaDc,
                dc,
                4,
                Some(ca),
                Some(cb),
                false,
                1,
                field,
            );
            if is_cr {
                cur.cbf_cr_dc = coded;
            } else {
                cur.cbf_cb_dc = coded;
            }
        }
    }
    if cbp_chroma == 2 {
        for (is_cr, ac) in [(false, cb_ac_scan), (true, cr_ac_scan)] {
            for (blk, ac_blk) in ac.iter().enumerate() {
                let (ca, cb) = cabac_cbf_cond_terms(
                    cab,
                    addr as u32,
                    cur,
                    is_intra,
                    BlockType::ChromaAc,
                    blk as u8,
                    is_cr,
                    1,
                );
                let list = scan15(ac_blk, field);
                let coded = encode_residual_block_cabac_field(
                    &mut cs.enc,
                    &mut cs.ctxs,
                    BlockType::ChromaAc,
                    &list,
                    15,
                    Some(ca),
                    Some(cb),
                    false,
                    1,
                    field,
                );
                if is_cr {
                    cur.cbf_cr_ac[blk] = coded;
                } else {
                    cur.cbf_cb_ac[blk] = coded;
                }
            }
        }
    }
}

/// Code one macroblock of the MBAFF picture (CABAC), mirroring the
/// decoder's reconstruction into the frame-geometry planes and its
/// neighbour-grid bookkeeping into `cab`.
#[allow(clippy::too_many_arguments)]
fn code_mb(
    st: &mut PicState,
    cab: &mut CabacNeighbourGrid,
    ve: &mut VirtualEnc,
    ctx: &MbCtx<'_>,
    addr: usize,
    field: bool,
    is_p: bool,
    cs: &mut CabacState,
) -> MbOut {
    let w_mbs = st.w_mbs;
    let (vx, vy) = virtual_pos(addr, w_mbs, field);
    let parity = (addr % 2) as u32;
    let is_bottom = addr % 2 == 1;

    // The MB must know its own geometry before Table 6-4 runs for the
    // sample / motion neighbourhood.
    st.field[addr] = field;
    patch_neighbour_samples(st, ve, addr, vx, vy);
    let mut nc_grid = CavlcNcGrid::new(w_mbs as u32, ve.h_mbs as u32);
    let mut intra_grid = IntraGrid::new(w_mbs, ve.h_mbs);
    let mut mv_grid = MvGrid::new(w_mbs, ve.h_mbs);
    fill_neighbour_grids(
        st,
        addr,
        field,
        vx,
        vy,
        w_mbs,
        ve.h_mbs,
        &mut nc_grid,
        &mut intra_grid,
        &mut mv_grid,
    );

    let width = ve.enc.config().width as usize;
    let height = ve.enc.config().height as usize;
    let chroma_w = width / 2;
    let chroma_h = height / 2;
    let src = if field {
        let (fy, fu, fv) = &ctx.sources_field[parity as usize];
        YuvFrame {
            width: width as u32,
            height: height as u32,
            y: fy,
            u: fu,
            v: fv,
        }
    } else {
        YuvFrame {
            width: ctx.src_frame.width,
            height: ctx.src_frame.height,
            y: ctx.src_frame.y,
            u: ctx.src_frame.u,
            v: ctx.src_frame.v,
        }
    };
    let (qp_y, qp_c) = (ctx.qp_y, ctx.qp_c);
    let w4 = &ctx.w4;

    // §7.4.4 / §9.3.3.1.1.1 — the grid's view of the CURRENT pair flag
    // before any ctxIdxInc derivation: the coded flag when known, else
    // the spatial inference.
    let eff_flag = cs
        .pending_pair_flag
        .unwrap_or_else(|| infer_pair_field_flag(cab, addr as u32, w_mbs as u32));
    cab.mbs[addr].mb_field_decoding_flag = eff_flag;

    let mut cur = CabacMbNeighbourInfo::default();
    let mut dbl = MbDeblockInfo {
        qp_y,
        ..Default::default()
    };

    // ------------------------------------------------------------------
    // P picture: inter candidate (+ optional intra alternative).
    // ------------------------------------------------------------------
    let mut skipped = false;
    let mut intra_in_p = false;
    let mut inter_done = false;
    if is_p {
        let refs = ctx.refs.expect("P picture without references");
        let (ry, ru, rv) = if field {
            &refs[1 + parity as usize]
        } else {
            &refs[0]
        };
        let me = search_quarter_pel_16x16(
            src.y,
            width,
            src.width,
            src.height,
            ry,
            width,
            width as u32,
            height as u32,
            vx,
            vy,
            16,
            16,
        );
        let chosen = Mv::new(me.mv_x, me.mv_y);
        let mvp = mvp_for_16x16(&mv_grid, vx, vy, 0);
        let (_skip_ref, skip_mv) = p_skip_mv(&mv_grid, vx, vy);
        let pred_y = build_inter_pred_luma_local(ry, width as u32, height as u32, vx, vy, chosen);
        let pred_u =
            build_inter_pred_chroma_local(ru, chroma_w as u32, chroma_h as u32, vx, vy, chosen);
        let pred_v =
            build_inter_pred_chroma_local(rv, chroma_w as u32, chroma_h as u32, vx, vy, chosen);

        // Luma residual: 16 4x4 blocks in luma4x4BlkIdx order.
        let mut luma_scan = [[0i32; 16]; 16];
        let mut blk_has_nz = [false; 16];
        let mut recon_res_y = [0i32; 256];
        for blk in 0..16usize {
            let (bx, by) = blk_xy(blk);
            let mut block = [0i32; 16];
            for j in 0..4 {
                for i in 0..4 {
                    let s = src.y[(vy * 16 + by * 4 + j) * width + vx * 16 + bx * 4 + i] as i32;
                    block[j * 4 + i] = s - pred_y[(by * 4 + j) * 16 + bx * 4 + i];
                }
            }
            let coeffs = forward_core_4x4(&block);
            let q = quantize_4x4_w(&coeffs, qp_y, false, w4);
            let scan = zigzag_scan_4x4(&q);
            blk_has_nz[blk] = scan.iter().any(|&v| v != 0);
            luma_scan[blk] = scan;
            let r = inverse_transform_4x4(&q, qp_y, w4, 8).expect("inverse 4x4");
            for j in 0..4 {
                for i in 0..4 {
                    recon_res_y[(by * 4 + j) * 16 + bx * 4 + i] = r[j * 4 + i];
                }
            }
        }
        let mut cbp_luma: u8 = 0;
        for blk8 in 0..4usize {
            if (0..4).any(|sub| blk_has_nz[blk8 * 4 + sub]) {
                cbp_luma |= 1 << blk8;
            }
        }
        let (cb_dc, cb_ac_scan, _cbq, cb_dc_nz, cb_ac_nz, cb_res) =
            encode_chroma_inter_420(src.u, &pred_u, chroma_w, vx, vy, qp_c, w4);
        let (cr_dc, cr_ac_scan, _crq, cr_dc_nz, cr_ac_nz, cr_res) =
            encode_chroma_inter_420(src.v, &pred_v, chroma_w, vx, vy, qp_c, w4);
        let cbp_chroma: u8 = if cb_ac_nz || cr_ac_nz {
            2
        } else if cb_dc_nz || cr_dc_nz {
            1
        } else {
            0
        };
        let is_skip = chosen == skip_mv && cbp_luma == 0 && cbp_chroma == 0;

        // Intra alternative — a plain SAD election (fixture-grade: the
        // point is exercising the P-slice intra syntax under MBAFF
        // neighbours, not RD optimality).
        let intra = if ctx.intra_in_p && !is_skip {
            let cand = intra16x16_candidate(&src, ve, vx, vy, qp_y, qp_c, w4);
            (cand.sad + 64 < me.sad as u64).then_some(cand)
        } else {
            None
        };

        // §9.3.3.1.1.1 — mb_skip_flag under the grid's current pair view.
        let skip_nctx = skip_neighbour_ctx(cab, addr);
        encode_mb_skip_flag(&mut cs.enc, &mut cs.ctxs, SliceKind::P, &skip_nctx, is_skip);
        if is_skip {
            skipped = true;
            cs.prev_qp_delta_nonzero = false;
            // Grid slot exactly as the decoder records a skipped MB.
            let slot = &mut cab.mbs[addr];
            *slot = CabacMbNeighbourInfo {
                available: true,
                is_skip: true,
                mb_field_decoding_flag: eff_flag,
                ref_idx_l0: [0; 4],
                ref_idx_l1: [0; 4],
                ..Default::default()
            };
            // Pure predictor reconstruction.
            for j in 0..16usize {
                for i in 0..16usize {
                    ve.vy[(vy * 16 + j) * width + vx * 16 + i] =
                        pred_y[j * 16 + i].clamp(0, 255) as u8;
                }
            }
            for j in 0..8usize {
                for i in 0..8usize {
                    let off = (vy * 8 + j) * chroma_w + vx * 8 + i;
                    ve.vu[off] = pred_u[j * 8 + i].clamp(0, 255) as u8;
                    ve.vv[off] = pred_v[j * 8 + i].clamp(0, 255) as u8;
                }
            }
            let mv_t = (skip_mv.x as i16, skip_mv.y as i16);
            dbl.mv_l0 = [mv_t; 16];
            dbl.ref_idx_l0 = [0; 4];
            st.mv[addr] = MvGridSlot {
                available: true,
                is_intra: false,
                ref_idx_l0_8x8: [0; 4],
                mv_l0_8x8: [skip_mv; 4],
            };
            inter_done = true;
        } else {
            // §7.3.4 — mb_field_decoding_flag rides the first coded MB
            // of the pair.
            if !is_bottom || cs.prev_mb_skipped {
                encode_mb_field_decoding_flag(&mut cs.enc, &mut cs.ctxs, cab, addr, field);
                cs.pending_pair_flag = Some(field);
                cab.mbs[addr].mb_field_decoding_flag = field;
                if is_bottom {
                    // §7.4.4 NOTE — the skipped top MB takes the
                    // bottom's coded flag.
                    cab.mbs[addr - 1].mb_field_decoding_flag = field;
                }
            }
            debug_assert_eq!(cab.mbs[addr].mb_field_decoding_flag, field);
            let nctx = full_neighbour_ctx(cab, addr);
            if let Some(cand) = intra {
                intra_in_p = true;
                emit_intra16x16(
                    cs, cab, &mut cur, &nctx, addr, field, true, &cand, ve, vx, vy, qp_y, qp_c, w4,
                );
                dbl.is_intra = true;
                dbl.luma_nonzero_4x4 = if cand.cbp_luma == 15 { 0xFFFF } else { 0 };
                dbl.ref_idx_l0 = [-1; 4];
                dbl.ref_poc_l0 = [i32::MIN; 4];
                st.mv[addr] = MvGridSlot {
                    available: true,
                    is_intra: true,
                    ref_idx_l0_8x8: [-1; 4],
                    mv_l0_8x8: [Mv::ZERO; 4],
                };
            } else {
                // mb_type P_L0_16x16 (Table 7-13 raw 0).
                encode_mb_type_p(&mut cs.enc, &mut cs.ctxs, &nctx, 0);
                cur.is_intra = false;
                cur.is_skip = false;
                // §7.3.5.1 — ref_idx_l0 is present for field MBs (the
                // effective list holds 2 · num_ref_idx_l0_active fields).
                if field {
                    let (ca, cb) = cabac_ref_idx_cond_terms(cab, addr as u32, &cur, 0, 0);
                    encode_ref_idx_lx(&mut cs.enc, &mut cs.ctxs, ca, cb, 0);
                }
                cur.ref_idx_l0 = [0; 4];
                let mvd_x = chosen.x - mvp.x;
                let mvd_y = chosen.y - mvp.y;
                let sum_x = cabac_mvd_abs_sum(cab, addr as u32, &cur, 0, MvdComponent::X, 0);
                encode_mvd_lx(&mut cs.enc, &mut cs.ctxs, MvdComponent::X, sum_x, mvd_x);
                let sum_y = cabac_mvd_abs_sum(cab, addr as u32, &cur, 0, MvdComponent::Y, 0);
                encode_mvd_lx(&mut cs.enc, &mut cs.ctxs, MvdComponent::Y, sum_y, mvd_y);
                cur.mvd_l0_x = [mvd_x.clamp(i16::MIN as i32, i16::MAX as i32) as i16; 16];
                cur.mvd_l0_y = [mvd_y.clamp(i16::MIN as i32, i16::MAX as i32) as i16; 16];
                encode_coded_block_pattern(
                    &mut cs.enc,
                    &mut cs.ctxs,
                    &nctx,
                    1,
                    cbp_luma,
                    cbp_chroma,
                );
                if cbp_luma > 0 || cbp_chroma > 0 {
                    encode_mb_qp_delta(&mut cs.enc, &mut cs.ctxs, cs.prev_qp_delta_nonzero, 0);
                }
                cs.prev_qp_delta_nonzero = false;
                cur.coded_block_pattern_luma = cbp_luma;
                cur.coded_block_pattern_chroma = cbp_chroma;
                // §7.3.5.3 — luma 4x4 blocks of the coded quadrants.
                for blk8 in 0..4usize {
                    if (cbp_luma >> blk8) & 1 == 0 {
                        continue;
                    }
                    for sub in 0..4usize {
                        let blk = blk8 * 4 + sub;
                        let (ca, cb) = cabac_cbf_cond_terms(
                            cab,
                            addr as u32,
                            &cur,
                            false,
                            BlockType::Luma4x4,
                            blk as u8,
                            false,
                            1,
                        );
                        let list = scan16(&luma_scan[blk], field);
                        cur.cbf_luma_4x4[blk] = encode_residual_block_cabac_field(
                            &mut cs.enc,
                            &mut cs.ctxs,
                            BlockType::Luma4x4,
                            &list,
                            16,
                            Some(ca),
                            Some(cb),
                            false,
                            1,
                            field,
                        );
                    }
                }
                emit_chroma_420(
                    cs,
                    cab,
                    &mut cur,
                    addr,
                    field,
                    false,
                    cbp_chroma,
                    &cb_dc,
                    &cr_dc,
                    &cb_ac_scan,
                    &cr_ac_scan,
                );
                // Reconstruction.
                for j in 0..16usize {
                    for i in 0..16usize {
                        ve.vy[(vy * 16 + j) * width + vx * 16 + i] =
                            (pred_y[j * 16 + i] + recon_res_y[j * 16 + i]).clamp(0, 255) as u8;
                    }
                }
                let (res_u, res_v) =
                    chroma_recon_residual(cbp_chroma, &cb_dc, &cr_dc, &cb_res, &cr_res, qp_c, w4);
                for j in 0..8usize {
                    for i in 0..8usize {
                        let off = (vy * 8 + j) * chroma_w + vx * 8 + i;
                        ve.vu[off] = (pred_u[j * 8 + i] + res_u[j * 8 + i]).clamp(0, 255) as u8;
                        ve.vv[off] = (pred_v[j * 8 + i] + res_v[j * 8 + i]).clamp(0, 255) as u8;
                    }
                }
                let mv_t = (chosen.x as i16, chosen.y as i16);
                dbl.mv_l0 = [mv_t; 16];
                dbl.ref_idx_l0 = [0; 4];
                dbl.luma_nonzero_4x4 = luma_nz_mask_from_blocks(&blk_has_nz);
                let cb_nz: [bool; 4] = std::array::from_fn(|i| {
                    cbp_chroma == 2 && cb_ac_scan[i].iter().any(|&v| v != 0)
                });
                let cr_nz: [bool; 4] = std::array::from_fn(|i| {
                    cbp_chroma == 2 && cr_ac_scan[i].iter().any(|&v| v != 0)
                });
                dbl.chroma_nonzero_4x4 = chroma_nz_mask_from_blocks(&cb_nz, &cr_nz);
                st.mv[addr] = MvGridSlot {
                    available: true,
                    is_intra: false,
                    ref_idx_l0_8x8: [0; 4],
                    mv_l0_8x8: [chosen; 4],
                };
            }
            inter_done = true;
        }
    }

    // ------------------------------------------------------------------
    // I picture: all-Intra_16x16 (flag coded with every top MB).
    // ------------------------------------------------------------------
    if !inter_done {
        if !is_bottom {
            encode_mb_field_decoding_flag(&mut cs.enc, &mut cs.ctxs, cab, addr, field);
            cs.pending_pair_flag = Some(field);
            cab.mbs[addr].mb_field_decoding_flag = field;
        }
        let nctx = full_neighbour_ctx(cab, addr);
        let cand = intra16x16_candidate(&src, ve, vx, vy, qp_y, qp_c, w4);
        emit_intra16x16(
            cs, cab, &mut cur, &nctx, addr, field, false, &cand, ve, vx, vy, qp_y, qp_c, w4,
        );
        dbl.is_intra = true;
        dbl.luma_nonzero_4x4 = if cand.cbp_luma == 15 { 0xFFFF } else { 0 };
        dbl.ref_idx_l0 = [-1; 4];
        dbl.ref_poc_l0 = [i32::MIN; 4];
        st.mv[addr] = MvGridSlot {
            available: true,
            is_intra: true,
            ref_idx_l0_8x8: [-1; 4],
            mv_l0_8x8: [Mv::ZERO; 4],
        };
    }

    if !skipped {
        // Commit the coded MB's grid slot (decoder mirror).
        cur.available = true;
        cur.mb_field_decoding_flag = field;
        cab.mbs[addr] = cur;
    }
    if is_p {
        cs.prev_mb_skipped = skipped;
    }

    // §8.7.2.1 NOTE 1 — reference-identity keys for the deblock mirror.
    for i in 0..4 {
        if dbl.ref_idx_l0[i] >= 0 {
            dbl.ref_poc_l0[i] = ref_poc_key(ctx.prev_poc, field, parity);
        }
    }
    st.avail[addr] = true;
    st.dbl[addr] = dbl;

    // Scatter the reconstructed MB from the scratch planes into the
    // frame-geometry recon (§6.4.1 y-stride 2 for field MBs).
    let (ox, oy) = mbaff_mb_to_sample_xy(addr as u32, w_mbs as u32, field);
    let (ox, oy) = (ox as usize, oy as usize);
    let stride = if field { 2 } else { 1 };
    for j in 0..16usize {
        let src_row = (vy * 16 + j) * st.w + vx * 16;
        let dst_row = (oy + j * stride) * st.w + ox;
        st.ry[dst_row..dst_row + 16].copy_from_slice(&ve.vy[src_row..src_row + 16]);
    }
    let pair_idx = addr / 2;
    let bot = addr % 2;
    let ccx = (pair_idx % w_mbs) * 8;
    let ccpy = (pair_idx / w_mbs) * 16;
    let coy = if field { ccpy + bot } else { ccpy + bot * 8 };
    for j in 0..8usize {
        let src_row = (vy * 8 + j) * chroma_w + vx * 8;
        let dst_row = (coy + j * stride) * chroma_w + ccx;
        st.ru[dst_row..dst_row + 8].copy_from_slice(&ve.vu[src_row..src_row + 8]);
        st.rv[dst_row..dst_row + 8].copy_from_slice(&ve.vv[src_row..src_row + 8]);
    }

    MbOut {
        skipped,
        intra_in_p,
    }
}

/// Encode an interlaced sequence as CABAC MBAFF frame pictures.
/// `frames` holds frame-geometry 4:2:0 planes (top field = even rows).
pub fn encode_mbaff_cabac_sequence(
    cfg: &MbaffCabacConfig,
    frames: &[(&[u8], &[u8], &[u8])],
) -> MbaffCabacEncoded {
    assert!(cfg.width % 16 == 0, "width must be MB-aligned");
    assert!(
        cfg.frame_height % 32 == 0,
        "frame height must be a multiple of 32 (MB pairs)",
    );
    assert!(!frames.is_empty());

    let w = cfg.width as usize;
    let frame_h = cfg.frame_height as usize;
    let w_mbs = cfg.width / 16;
    let frame_h_mbs = cfg.frame_height / 16;
    let log2_max_frame_num_minus4: u32 = 4;
    let log2_max_poc_lsb_minus4: u32 = 4;
    let frame_num_bits = log2_max_frame_num_minus4 + 4;
    let poc_lsb_bits = log2_max_poc_lsb_minus4 + 4;
    let qp_y = cfg.qp;
    let qp_c = qp_y_to_qp_c_with_bd_offset(qp_y, 0, qp_bd_offset(0));
    let w4 = ScalingMatrixMode::Flat.intra_weights().w4;

    // Main (77): interlace + CABAC.
    let profile_idc: u8 = 77;
    let mk_cfg = |h: u32| {
        let mut c = EncoderConfig::new(cfg.width, h);
        c.qp = cfg.qp;
        c.profile_idc = profile_idc;
        c.cabac = true;
        c
    };
    let mut ve_frame = VirtualEnc::new(mk_cfg(cfg.frame_height));
    let mut ve_field = VirtualEnc::new(mk_cfg(cfg.frame_height / 2));

    let sps_rbsp = build_baseline_sps_rbsp(&BaselineSpsConfig {
        seq_parameter_set_id: 0,
        level_idc: min_level_idc_for_picture_size(w_mbs, frame_h_mbs),
        width_in_mbs: w_mbs,
        height_in_mbs: frame_h_mbs,
        log2_max_frame_num_minus4,
        log2_max_poc_lsb_minus4,
        max_num_ref_frames: 1,
        profile_idc,
        chroma_format_idc: 1,
        separate_colour_plane: false,
        seq_scaling_lists: None,
        bit_depth_luma_minus8: 0,
        bit_depth_chroma_minus8: 0,
        interlaced_fields: true,
        mbaff: true,
        vui: None,
    });
    let pps_rbsp = build_baseline_pps_rbsp(&BaselinePpsConfig {
        pic_parameter_set_id: 0,
        seq_parameter_set_id: 0,
        pic_init_qp_minus26: cfg.qp - 26,
        chroma_qp_index_offset: 0,
        weighted_pred_flag: false,
        weighted_bipred_idc: 0,
        entropy_coding_mode_flag: true,
        transform_8x8_mode_flag: false,
        redundant_pic_cnt_present_flag: false,
        slice_groups: None,
        constrained_intra_pred_flag: false,
        pic_scaling_lists: None,
        chroma_format_idc: 1,
    });
    let mut stream: Vec<u8> = Vec::new();
    stream.extend_from_slice(&build_nal_unit(3, NalUnitType::Sps, &sps_rbsp));
    stream.extend_from_slice(&build_nal_unit(3, NalUnitType::Pps, &pps_rbsp));

    let mut recon_frames: Vec<Planes> = Vec::with_capacity(frames.len());
    let mut prev_refs: Option<[Planes; 3]> = None;
    let mut prev_poc: i32 = 0;

    let mut field_pairs = 0usize;
    let mut frame_pairs = 0usize;
    let mut skipped_mbs = 0usize;
    let mut intra_mbs_in_p = 0usize;
    let mut inference_reencodes = 0usize;

    for (k, &(fy, fu, fv)) in frames.iter().enumerate() {
        assert_eq!(fy.len(), w * frame_h);
        assert_eq!(fu.len(), (w / 2) * (frame_h / 2));
        let is_p = cfg.p_frames && k > 0;
        let frame_num = (k as u32) % (1 << frame_num_bits);
        let poc = 2 * k as i32;

        let mut sw = BitWriter::new();
        if is_p {
            write_p_slice_header(
                &mut sw,
                &PSliceHeaderConfig {
                    first_mb_in_slice: 0,
                    slice_type_raw: 5,
                    pic_parameter_set_id: 0,
                    colour_plane_id: None,
                    frame_num,
                    frame_num_bits,
                    pic_order_cnt_lsb: (poc as u32) % (1 << poc_lsb_bits),
                    poc_lsb_bits,
                    slice_qp_delta: 0,
                    disable_deblocking_filter_idc: 0,
                    slice_alpha_c0_offset_div2: 0,
                    slice_beta_offset_div2: 0,
                    nal_ref_idc: 2,
                    cabac: Some(CabacSliceParams { cabac_init_idc: 0 }),
                    field: FieldPicSignal::FramePicture,
                    rplm_l0: &[],
                    mmco: &[],
                    pred_weight_table: None,
                    num_ref_idx_l0_active_minus1: None,
                    redundant_pic_cnt: None,
                    slice_group_change_cycle: None,
                },
            );
        } else {
            write_idr_i_slice_header(
                &mut sw,
                &IdrSliceHeaderConfig {
                    first_mb_in_slice: 0,
                    slice_type_raw: 7,
                    pic_parameter_set_id: 0,
                    colour_plane_id: None,
                    frame_num,
                    frame_num_bits,
                    idr_pic_id: 0,
                    pic_order_cnt_lsb: (poc as u32) % (1 << poc_lsb_bits),
                    poc_lsb_bits,
                    slice_qp_delta: 0,
                    disable_deblocking_filter_idc: 0,
                    slice_alpha_c0_offset_div2: 0,
                    slice_beta_offset_div2: 0,
                    field: FieldPicSignal::FramePicture,
                    idr: k == 0,
                    nal_ref_idc: if k == 0 { 3 } else { 2 },
                    long_term_reference_flag: false,
                    mmco: &[],
                    redundant_pic_cnt: None,
                    slice_group_change_cycle: None,
                },
            );
        }
        // §7.3.4 cabac_alignment_one_bit.
        while !sw.byte_aligned() {
            sw.u(1, 1);
        }

        let mut st = PicState::new(w, frame_h);
        let mut cab = CabacNeighbourGrid::new_mbaff(w_mbs, frame_h_mbs, true);
        let sources_field: [Planes; 2] = [
            extract_field_planes(fy, fu, fv, w, frame_h, false),
            extract_field_planes(fy, fu, fv, w, frame_h, true),
        ];
        let ctx = MbCtx {
            qp_y,
            qp_c,
            prev_poc,
            refs: prev_refs.as_ref(),
            sources_field: &sources_field,
            src_frame: YuvFrame {
                width: cfg.width,
                height: cfg.frame_height,
                y: fy,
                u: fu,
                v: fv,
            },
            w4,
            intra_in_p: cfg.intra_in_p,
        };

        let kind = if is_p { SliceKind::P } else { SliceKind::I };
        let mut cs = CabacState {
            enc: CabacEncoder::new(),
            ctxs: CabacContexts::init(kind, if is_p { Some(0) } else { None }, qp_y)
                .expect("ctx init"),
            prev_qp_delta_nonzero: false,
            prev_mb_skipped: false,
            pending_pair_flag: None,
        };
        let n_pairs = (w_mbs * frame_h_mbs / 2) as usize;
        for pair in 0..n_pairs {
            let (pc, pr) = (pair % w_mbs as usize, pair / w_mbs as usize);
            let mut field = match cfg.pair_mode {
                PairMode::AllFrame => false,
                PairMode::AllField => true,
                PairMode::Checker => (pc + pr) % 2 == 1,
                PairMode::Adaptive => adaptive_pair_is_field(fy, w, pc, pr),
            };
            let top = 2 * pair;
            loop {
                cs.pending_pair_flag = None;
                let enc_snap = cs.enc.clone();
                let ctxs_snap = cs.ctxs.clone();
                let qpd_snap = cs.prev_qp_delta_nonzero;
                let skipped_snap = cs.prev_mb_skipped;
                let ve = if field { &mut ve_field } else { &mut ve_frame };
                let t = code_mb(&mut st, &mut cab, ve, &ctx, top, field, is_p, &mut cs);
                let b = code_mb(&mut st, &mut cab, ve, &ctx, top + 1, field, is_p, &mut cs);
                if t.skipped && b.skipped {
                    // §7.4.4 — a fully-skipped pair codes no flag; the
                    // decoder infers it. Re-encode under the inferred
                    // flag when it differs from our choice.
                    st.field[top] = field;
                    st.field[top + 1] = field;
                    let inferred = cab.mbs[top].mb_field_decoding_flag;
                    if inferred != field {
                        cs.enc = enc_snap;
                        cs.ctxs = ctxs_snap;
                        cs.prev_qp_delta_nonzero = qpd_snap;
                        cs.prev_mb_skipped = skipped_snap;
                        for a in [top, top + 1] {
                            st.avail[a] = false;
                            st.nc[a] = Default::default();
                            st.mv[a] = MvGridSlot::default();
                            cab.mbs[a] = CabacMbNeighbourInfo::default();
                        }
                        field = inferred;
                        inference_reencodes += 1;
                        continue;
                    }
                }
                skipped_mbs += usize::from(t.skipped) + usize::from(b.skipped);
                intra_mbs_in_p += usize::from(t.intra_in_p) + usize::from(b.intra_in_p);
                break;
            }
            // §7.3.4 — end_of_slice_flag only after the bottom MB.
            encode_end_of_slice_flag(&mut cs.enc, pair + 1 == n_pairs);
            if field {
                field_pairs += 1;
            } else {
                frame_pairs += 1;
            }
        }
        let payload = cs.enc.finish();
        debug_assert!(sw.byte_aligned());
        let mut slice_rbsp = sw.into_bytes();
        slice_rbsp.extend_from_slice(&payload);
        let (nal_type, ref_idc) = if k == 0 {
            (NalUnitType::SliceIdr, 3)
        } else {
            (NalUnitType::SliceNonIdr, 2)
        };
        stream.extend_from_slice(&build_nal_unit(ref_idc, nal_type, &slice_rbsp));

        // §8.7 — MBAFF picture-level deblock on the assembled frame.
        let mut ry = st.ry.clone();
        let mut ru = st.ru.clone();
        let mut rv = st.rv.clone();
        deblock_recon_mbaff(
            cfg.width,
            cfg.frame_height,
            cfg.width / 2,
            cfg.frame_height / 2,
            &mut ry,
            &mut ru,
            &mut rv,
            &st.dbl,
            &st.field,
            0,
            w_mbs,
            frame_h_mbs,
            1,
        );
        let top_view = extract_field_planes(&ry, &ru, &rv, w, frame_h, false);
        let bot_view = extract_field_planes(&ry, &ru, &rv, w, frame_h, true);
        prev_refs = Some([(ry.clone(), ru.clone(), rv.clone()), top_view, bot_view]);
        prev_poc = poc;
        recon_frames.push((ry, ru, rv));
    }

    MbaffCabacEncoded {
        annex_b: stream,
        recon_frames,
        field_pairs,
        frame_pairs,
        skipped_mbs,
        intra_mbs_in_p,
        inference_reencodes,
    }
}
