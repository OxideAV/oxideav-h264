//! Round-451 — SP / SI slice encoder (Extended profile, §8.6).
//!
//! Emits the three picture kinds of the §8.6 switching-picture system:
//!
//! * **Primary SP pictures** (`sp_for_switch_flag == 0`, §8.6.1):
//!   Inter-predicted pictures whose residual is chosen in the
//!   transform domain so the reconstruction is fully determined by the
//!   QSY/QSC-quantised coefficients.
//! * **Switching SP pictures** (`sp_for_switch_flag == 1`, §8.6.2):
//!   Inter-predicted from a *different* reference (e.g. another
//!   stream's decode state) while reproducing a primary SP picture's
//!   reconstruction **bit-exactly** — the defining SP property.
//! * **SI pictures** (§8.6.2 with Intra_4x4 prediction): the same
//!   bit-exact switch with no reference picture at all.
//!
//! The encoder keeps a mirror of the decoder's §8.6 arithmetic (via
//! [`crate::sp_transform`]) so a primary picture's final coefficient
//! arrays can serve as the *targets* a switching picture must hit:
//! for every block the switching encoder computes the §8.6.2 quantised
//! prediction `cs` and transmits `cr = c_target − cs`, which the
//! decoder adds back (eq. 8-433 / 8-437) to land on `c_target`
//! exactly. The chroma DC needs one extra step: eq. 8-431 scales the
//! non-switching DC by an integral factor `k` (for QSC >= 6), so the
//! switching side reproduces it with `dcr_switch = k · dcr_primary`
//! (eq. 8-440's added level makes any integer `dcr` reachable).
//!
//! Simplifications (they do not reduce what the *decoder* is exercised
//! on): single slice per picture, all-zero motion (`P_L0_16x16` with
//! `mvd == (0, 0)` — with every neighbour at zero motion the §8.4.1.3
//! median predictor is zero, so `mv == (0, 0)` everywhere), no skips,
//! an all-I_PCM IDR anchor (bit-exact reference by construction), DC
//! intra modes on SI pictures, `mb_qp_delta == 0`.

use super::bitstream::BitWriter;
use super::cavlc::encode_residual_block_cavlc;
use super::macroblock::{
    inter_cbp_to_codenum_420_422, intra_cbp_to_codenum_420_422, write_i_nxn_mb,
    write_p_l0_16x16_mb, INxNMcbConfig, PL016x16McbConfig,
};
use super::nal::build_nal_unit;
use super::pps::{build_baseline_pps_rbsp, BaselinePpsConfig};
use super::sps::{build_baseline_sps_rbsp, BaselineSpsConfig};
use super::transform::{zigzag_scan_4x4, zigzag_scan_4x4_ac};
use crate::cavlc::CoeffTokenContext;
use crate::intra_pred::{
    predict_4x4, predict_chroma, ChromaArrayType, Intra4x4Mode, IntraChromaMode,
    Neighbour4x4Availability, Samples4x4, SamplesChroma,
};
use crate::macroblock_layer::{derive_nc_luma, CavlcNcGrid, LumaNcKind};
use crate::nal::NalUnitType;
use crate::sp_transform::{
    chroma_dc_switch_scale, chroma_pred_transform, encoder_quant_8_416, encoder_quant_8_428,
    hadamard_2x2, quant_qs_dc, sp_chroma_non_switching, sp_chroma_switching, sp_luma_non_switching,
    sp_luma_switching,
};
use crate::transform::{inverse_transform_4x4, inverse_transform_4x4_dc_preserved, qp_y_to_qp_c};

/// Flat §8.5.9 weightScale (the Extended profile carries no scaling
/// matrices).
const FLAT: [i32; 16] = [16; 16];

/// §6.4.3 — raster-Z 4x4 luma block positions inside a macroblock
/// (blk = quadrant * 4 + sub-block).
const BLK4_XY: [(usize, usize); 16] = [
    (0, 0),
    (4, 0),
    (0, 4),
    (4, 4),
    (8, 0),
    (12, 0),
    (8, 4),
    (12, 4),
    (0, 8),
    (4, 8),
    (0, 12),
    (4, 12),
    (8, 8),
    (12, 8),
    (8, 12),
    (12, 12),
];

/// Configuration for one SP/SI stream.
#[derive(Clone, Copy, Debug)]
pub struct SpConfig {
    /// Picture width in samples (multiple of 16).
    pub width: u32,
    /// Picture height in samples (multiple of 16).
    pub height: u32,
    /// Slice QPY (constant across the stream; `mb_qp_delta == 0`).
    pub qp: i32,
    /// Slice QSY (eq. 7-33; the PPS codes `pic_init_qs_minus26 = 0`
    /// so `slice_qs_delta = qs − 26`). The switching constructions
    /// require QSC >= 6, i.e. `qs >= 6` with a zero chroma offset.
    pub qs: i32,
    /// `false` emits `disable_deblocking_filter_idc = 1` in every
    /// slice header (bit-exact pre-deblock comparisons); `true` leaves
    /// the §8.7 filter on.
    pub deblock: bool,
}

impl SpConfig {
    fn width_mbs(&self) -> usize {
        (self.width / 16) as usize
    }
    fn height_mbs(&self) -> usize {
        (self.height / 16) as usize
    }
    fn disable_deblocking_filter_idc(&self) -> u32 {
        if self.deblock {
            0
        } else {
            1
        }
    }
}

/// Per-macroblock §8.6 coefficient targets retained from a primary SP
/// encode: the exact arrays the decoder feeds §8.5.12 (luma; chroma
/// with the eq. 8-431-scaled DC in slot 0). A switching SP / SI
/// picture reproduces them bit-exactly.
#[derive(Clone)]
pub struct SpMbTargets {
    /// 16 4x4 luma coefficient blocks (row-major, raster-Z order).
    pub luma_c: [[i32; 16]; 16],
    /// Per plane (0 = Cb, 1 = Cr): 4 4x4 chroma coefficient blocks,
    /// DC slot included.
    pub chroma_c: [[[i32; 16]; 4]; 2],
}

/// Everything a switching picture needs to reproduce a primary SP
/// picture, plus the primary's pre-deblock reconstruction (equal to
/// the decoder's output when `deblock == false`; the switching
/// picture's own reconstruction in *all* cases).
pub struct SpTargets {
    pub mbs: Vec<SpMbTargets>,
    pub recon_y: Vec<u8>,
    pub recon_u: Vec<u8>,
    pub recon_v: Vec<u8>,
}

/// One encoded SP-family access unit.
pub struct SpEncodedPicture {
    /// The Annex B access unit (one slice NAL).
    pub annex_b: Vec<u8>,
    /// §8.6 coefficient targets + mirror reconstruction.
    pub targets: SpTargets,
}

// ---------------------------------------------------------------------------
// Parameter sets + IDR anchor
// ---------------------------------------------------------------------------

/// Annex B SPS (Extended profile, 88) + PPS for an SP/SI stream.
pub fn sp_parameter_sets(cfg: &SpConfig) -> Vec<u8> {
    let sps = build_baseline_sps_rbsp(&BaselineSpsConfig {
        seq_parameter_set_id: 0,
        level_idc: 20,
        width_in_mbs: cfg.width / 16,
        height_in_mbs: cfg.height / 16,
        log2_max_frame_num_minus4: 4,
        log2_max_poc_lsb_minus4: 4,
        max_num_ref_frames: 1,
        // §A.2.3 — SP/SI slices belong to the Extended profile.
        profile_idc: 88,
        chroma_format_idc: 1,
        separate_colour_plane: false,
        seq_scaling_lists: None,
        bit_depth_luma_minus8: 0,
        bit_depth_chroma_minus8: 0,
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

/// Encode one all-I_PCM IDR access unit (frame_num 0, POC 0). I_PCM
/// reconstruction is the raw samples, so the pre-deblock decode of
/// this anchor equals `y`/`u`/`v` bit-exactly — a reference whose
/// value needs no prediction mirror at all.
pub fn encode_ipcm_idr(cfg: &SpConfig, y: &[u8], u: &[u8], v: &[u8]) -> Vec<u8> {
    let (w_mbs, h_mbs) = (cfg.width_mbs(), cfg.height_mbs());
    let width = cfg.width as usize;
    let cw = width / 2;
    let mut bw = BitWriter::new();

    // §7.3.3 — IDR I-slice header.
    bw.ue(0); // first_mb_in_slice
    bw.ue(7); // slice_type: I, all slices in picture (Table 7-6)
    bw.ue(0); // pic_parameter_set_id
    bw.u(8, 0); // frame_num (log2_max_frame_num = 8)
    bw.ue(0); // idr_pic_id
    bw.u(8, 0); // pic_order_cnt_lsb (POC type 0)
                // §7.3.3.3 — dec_ref_pic_marking (IDR form).
    bw.u(1, 0); // no_output_of_prior_pics_flag
    bw.u(1, 0); // long_term_reference_flag
    bw.se(cfg.qp - 26); // slice_qp_delta (PPS pic_init_qp_minus26 = 0)
    bw.ue(cfg.disable_deblocking_filter_idc());
    if cfg.deblock {
        bw.se(0); // slice_alpha_c0_offset_div2
        bw.se(0); // slice_beta_offset_div2
    }

    for mby in 0..h_mbs {
        for mbx in 0..w_mbs {
            // §7.3.5 — mb_type = 25 (I_PCM, Table 7-11).
            bw.ue(25);
            // pcm_alignment_zero_bit until byte-aligned.
            bw.align_to_byte_zero();
            for row in 0..16 {
                for col in 0..16 {
                    bw.u(8, y[(mby * 16 + row) * width + mbx * 16 + col] as u32);
                }
            }
            for plane in [u, v] {
                for row in 0..8 {
                    for col in 0..8 {
                        bw.u(8, plane[(mby * 8 + row) * cw + mbx * 8 + col] as u32);
                    }
                }
            }
        }
    }
    bw.rbsp_trailing_bits();
    build_nal_unit(3, NalUnitType::SliceIdr, &bw.into_bytes())
}

// ---------------------------------------------------------------------------
// Slice headers
// ---------------------------------------------------------------------------

/// §7.3.3 — SP slice header (single slice, all-SP picture, zero RPLM,
/// sliding-window marking when `nal_ref_idc != 0`).
#[allow(clippy::too_many_arguments)]
fn write_sp_slice_header(
    bw: &mut BitWriter,
    cfg: &SpConfig,
    frame_num: u32,
    poc_lsb: u32,
    nal_ref_idc: u32,
    sp_for_switch_flag: bool,
) {
    bw.ue(0); // first_mb_in_slice
    bw.ue(8); // slice_type: SP, all slices in picture (Table 7-6)
    bw.ue(0); // pic_parameter_set_id
    bw.u(8, frame_num);
    bw.u(8, poc_lsb);
    bw.u(1, 0); // num_ref_idx_active_override_flag
    bw.u(1, 0); // ref_pic_list_modification_flag_l0
    if nal_ref_idc != 0 {
        bw.u(1, 0); // adaptive_ref_pic_marking_mode_flag (sliding window)
    }
    bw.se(cfg.qp - 26); // slice_qp_delta
    bw.u(1, if sp_for_switch_flag { 1 } else { 0 });
    bw.se(cfg.qs - 26); // slice_qs_delta (PPS pic_init_qs_minus26 = 0)
    bw.ue(cfg.disable_deblocking_filter_idc());
    if cfg.deblock {
        bw.se(0);
        bw.se(0);
    }
}

/// §7.3.3 — SI slice header (intra: no reference machinery).
fn write_si_slice_header(bw: &mut BitWriter, cfg: &SpConfig, frame_num: u32, poc_lsb: u32) {
    bw.ue(0); // first_mb_in_slice
    bw.ue(9); // slice_type: SI, all slices in picture (Table 7-6)
    bw.ue(0); // pic_parameter_set_id
    bw.u(8, frame_num);
    bw.u(8, poc_lsb);
    // Non-reference (nal_ref_idc == 0) — no dec_ref_pic_marking.
    bw.se(cfg.qp - 26); // slice_qp_delta
    bw.se(cfg.qs - 26); // slice_qs_delta (no sp_for_switch_flag on SI)
    bw.ue(cfg.disable_deblocking_filter_idc());
    if cfg.deblock {
        bw.se(0);
        bw.se(0);
    }
}

// ---------------------------------------------------------------------------
// Shared per-MB helpers
// ---------------------------------------------------------------------------

fn gather16(plane: &[u8], stride: usize, x0: usize, y0: usize) -> [i32; 256] {
    let mut out = [0i32; 256];
    for row in 0..16 {
        for col in 0..16 {
            out[row * 16 + col] = plane[(y0 + row) * stride + x0 + col] as i32;
        }
    }
    out
}

fn gather8(plane: &[u8], stride: usize, x0: usize, y0: usize) -> [i32; 64] {
    let mut out = [0i32; 64];
    for row in 0..8 {
        for col in 0..8 {
            out[row * 8 + col] = plane[(y0 + row) * stride + x0 + col] as i32;
        }
    }
    out
}

fn blk4(mb: &[i32; 256], bx: usize, by: usize) -> [i32; 16] {
    let mut out = [0i32; 16];
    for row in 0..4 {
        for col in 0..4 {
            out[row * 4 + col] = mb[(by + row) * 16 + bx + col];
        }
    }
    out
}

fn chroma_blk4(mb: &[i32; 64], blk: usize) -> [i32; 16] {
    let (bx, by) = ((blk % 2) * 4, (blk / 2) * 4);
    let mut out = [0i32; 16];
    for row in 0..4 {
        for col in 0..4 {
            out[row * 4 + col] = mb[(by + row) * 8 + bx + col];
        }
    }
    out
}

/// Decoder-mirror sample reconstruction of one MB from its §8.6 final
/// coefficient arrays: §8.5.12 at qP = QSY/QSC + Clip1, written into
/// the mirror planes.
#[allow(clippy::too_many_arguments)]
fn mirror_recon_mb(
    targets: &SpMbTargets,
    qs_y: i32,
    qs_c: i32,
    mbx: usize,
    mby: usize,
    width: usize,
    recon_y: &mut [u8],
    recon_u: &mut [u8],
    recon_v: &mut [u8],
) {
    for (blk, c) in targets.luma_c.iter().enumerate() {
        let (bx, by) = BLK4_XY[blk];
        let r = inverse_transform_4x4(c, qs_y, &FLAT, 8).expect("QSY in range");
        for row in 0..4 {
            for col in 0..4 {
                let v = r[row * 4 + col].clamp(0, 255) as u8;
                recon_y[(mby * 16 + by + row) * width + mbx * 16 + bx + col] = v;
            }
        }
    }
    let cw = width / 2;
    for (plane_idx, plane_c) in targets.chroma_c.iter().enumerate() {
        let plane = if plane_idx == 0 {
            &mut *recon_u
        } else {
            &mut *recon_v
        };
        for (blk, c) in plane_c.iter().enumerate() {
            let (bx, by) = ((blk % 2) * 4, (blk / 2) * 4);
            let r = inverse_transform_4x4_dc_preserved(c, qs_c, &FLAT, 8).expect("QSC in range");
            for row in 0..4 {
                for col in 0..4 {
                    let v = r[row * 4 + col].clamp(0, 255) as u8;
                    plane[(mby * 8 + by + row) * cw + mbx * 8 + bx + col] = v;
                }
            }
        }
    }
}

/// Derive the writer inputs (scan-order level lists + cbp) from
/// raster-domain residual levels.
struct MbLevels {
    luma_scan: [[i32; 16]; 16],
    cbp_luma: u8,
    dc_cb: [i32; 4],
    dc_cr: [i32; 4],
    ac_cb_scan: [[i32; 16]; 4],
    ac_cr_scan: [[i32; 16]; 4],
    cbp_chroma: u8,
}

fn pack_levels(
    luma_cr: &[[i32; 16]; 16],
    dc_levels: &[[i32; 4]; 2],
    ac_cr: &[[[i32; 16]; 4]; 2],
) -> MbLevels {
    let mut luma_scan = [[0i32; 16]; 16];
    let mut cbp_luma = 0u8;
    for blk in 0..16 {
        luma_scan[blk] = zigzag_scan_4x4(&luma_cr[blk]);
        if luma_scan[blk].iter().any(|&v| v != 0) {
            cbp_luma |= 1 << (blk / 4);
        }
    }
    let mut ac_scan = [[[0i32; 16]; 4]; 2];
    let mut any_ac = false;
    for plane in 0..2 {
        for blk in 0..4 {
            // Raster AC block with DC slot ignored — force it to zero
            // so the AC-only scan never sees it.
            let mut raster = ac_cr[plane][blk];
            raster[0] = 0;
            ac_scan[plane][blk] = zigzag_scan_4x4_ac(&raster);
            if ac_scan[plane][blk].iter().any(|&v| v != 0) {
                any_ac = true;
            }
        }
    }
    let any_dc = dc_levels.iter().flatten().any(|&v| v != 0);
    let cbp_chroma = if any_ac {
        2
    } else if any_dc {
        1
    } else {
        0
    };
    MbLevels {
        luma_scan,
        cbp_luma,
        dc_cb: dc_levels[0],
        dc_cr: dc_levels[1],
        ac_cb_scan: ac_scan[0],
        ac_cr_scan: ac_scan[1],
        cbp_chroma,
    }
}

/// §9.2.1.1 — derive per-block luma nC values and commit this MB's
/// TotalCoeff into the grid.
fn luma_nc_and_commit(
    nc_grid: &mut CavlcNcGrid,
    mb_addr: u32,
    is_intra: bool,
    levels: &MbLevels,
) -> [i32; 16] {
    {
        let cur = &mut nc_grid.mbs[mb_addr as usize];
        cur.is_available = true;
        cur.is_intra = is_intra;
        cur.is_skip = false;
        cur.is_i_pcm = false;
        cur.luma_total_coeff = [0u8; 16];
    }
    let mut nc = [0i32; 16];
    let mut own = [0u8; 16];
    for blk in 0..16usize {
        nc_grid.mbs[mb_addr as usize].luma_total_coeff = own;
        nc[blk] = derive_nc_luma(nc_grid, mb_addr, blk as u8, LumaNcKind::Ac, is_intra, false);
        if (levels.cbp_luma >> (blk / 4)) & 1 == 1 {
            own[blk] = levels.luma_scan[blk].iter().filter(|&&v| v != 0).count() as u8;
        }
    }
    nc_grid.mbs[mb_addr as usize].luma_total_coeff = own;
    nc
}

// ---------------------------------------------------------------------------
// Primary SP pictures (§8.6.1)
// ---------------------------------------------------------------------------

/// Encode one primary SP access unit (`sp_for_switch_flag == 0`).
///
/// * `src` — source planes (Y/Cb/Cr, 4:2:0).
/// * `refp` — the decoded reference planes (the previous access unit's
///   decoder output — deblocked when the stream deblocks).
///
/// All macroblocks are `P_L0_16x16` with zero motion: the prediction
/// of each block is the co-located reference block, and the residual
/// levels are the eq. 8-416-inverting quantisation of
/// `T(src) − T(pred)` so the §8.6.1 chain lands near the source.
#[allow(clippy::too_many_arguments)]
pub fn encode_sp_picture(
    cfg: &SpConfig,
    src: (&[u8], &[u8], &[u8]),
    refp: (&[u8], &[u8], &[u8]),
    frame_num: u32,
    poc_lsb: u32,
    nal_ref_idc: u8,
) -> SpEncodedPicture {
    assert!((0..=51).contains(&cfg.qp) && (0..=51).contains(&cfg.qs));
    let (w_mbs, h_mbs) = (cfg.width_mbs(), cfg.height_mbs());
    let width = cfg.width as usize;
    let cw = width / 2;
    let qp_c = qp_y_to_qp_c(cfg.qp, 0);
    let qs_c = qp_y_to_qp_c(cfg.qs, 0);

    let mut bw = BitWriter::new();
    write_sp_slice_header(
        &mut bw,
        cfg,
        frame_num,
        poc_lsb,
        u32::from(nal_ref_idc),
        false,
    );

    let mut nc_grid = CavlcNcGrid::new(w_mbs as u32, h_mbs as u32);
    let mut mbs = Vec::with_capacity(w_mbs * h_mbs);
    let mut recon_y = vec![0u8; width * cfg.height as usize];
    let mut recon_u = vec![0u8; cw * (cfg.height as usize / 2)];
    let mut recon_v = vec![0u8; cw * (cfg.height as usize / 2)];

    for mby in 0..h_mbs {
        for mbx in 0..w_mbs {
            let mb_addr = (mby * w_mbs + mbx) as u32;
            // ---- Luma: choose levels, mirror-decode, retain targets.
            let pred_mb = gather16(refp.0, width, mbx * 16, mby * 16);
            let src_mb = gather16(src.0, width, mbx * 16, mby * 16);
            let mut luma_cr = [[0i32; 16]; 16];
            let mut luma_c = [[0i32; 16]; 16];
            for blk in 0..16 {
                let (bx, by) = BLK4_XY[blk];
                let p = blk4(&pred_mb, bx, by);
                let s = blk4(&src_mb, bx, by);
                let cp = crate::sp_transform::forward_core_4x4(&p);
                let cwc = crate::sp_transform::forward_core_4x4(&s);
                let mut cr = [0i32; 16];
                for i in 0..4 {
                    for j in 0..4 {
                        let idx = i * 4 + j;
                        cr[idx] =
                            encoder_quant_8_416((cwc[idx] - cp[idx]) as i64, cfg.qp, &FLAT, i, j);
                    }
                }
                luma_cr[blk] = cr;
                // Decoder mirror (§8.6.1.1).
                luma_c[blk] = sp_luma_non_switching(&p, &cr, cfg.qp, cfg.qs, &FLAT);
            }

            // ---- Chroma: DC + AC levels, mirror-decode.
            let mut dc_levels = [[0i32; 4]; 2];
            let mut ac_levels = [[[0i32; 16]; 4]; 2];
            let mut chroma_c = [[[0i32; 16]; 4]; 2];
            for plane in 0..2 {
                let (sp_, rp_) = if plane == 0 {
                    (src.1, refp.1)
                } else {
                    (src.2, refp.2)
                };
                let pred8 = gather8(rp_, cw, mbx * 8, mby * 8);
                let src8 = gather8(sp_, cw, mbx * 8, mby * 8);
                let cp_blocks = chroma_pred_transform(&pred8);
                let cw_blocks = chroma_pred_transform(&src8);
                // AC levels per block (eq. 8-424 inversion).
                for blk in 0..4 {
                    for i in 0..4 {
                        for j in 0..4 {
                            if i == 0 && j == 0 {
                                continue;
                            }
                            let idx = i * 4 + j;
                            ac_levels[plane][blk][idx] = encoder_quant_8_416(
                                (cw_blocks[blk][idx] - cp_blocks[blk][idx]) as i64,
                                qp_c,
                                &FLAT,
                                i,
                                j,
                            );
                        }
                    }
                }
                // DC levels (eq. 8-428 inversion of dcw − dcp).
                let dcp = hadamard_2x2(&[
                    cp_blocks[0][0] as i64,
                    cp_blocks[1][0] as i64,
                    cp_blocks[2][0] as i64,
                    cp_blocks[3][0] as i64,
                ]);
                let dcw = hadamard_2x2(&[
                    cw_blocks[0][0] as i64,
                    cw_blocks[1][0] as i64,
                    cw_blocks[2][0] as i64,
                    cw_blocks[3][0] as i64,
                ]);
                for pos in 0..4 {
                    // Eq. 8-428 level pickup in the §8.5.4 raster
                    // convention (level k ↔ dc position k).
                    dc_levels[plane][pos] = encoder_quant_8_428(dcw[pos] - dcp[pos], qp_c, &FLAT);
                }
                // Decoder mirror (§8.6.1.2).
                chroma_c[plane] = sp_chroma_non_switching(
                    &pred8,
                    &dc_levels[plane],
                    &ac_levels[plane],
                    qp_c,
                    qs_c,
                    &FLAT,
                );
            }

            let targets = SpMbTargets { luma_c, chroma_c };
            mirror_recon_mb(
                &targets,
                cfg.qs,
                qs_c,
                mbx,
                mby,
                width,
                &mut recon_y,
                &mut recon_u,
                &mut recon_v,
            );

            // ---- Emit the macroblock.
            emit_sp_inter_mb(
                &mut bw,
                &mut nc_grid,
                mb_addr,
                &luma_cr,
                &dc_levels,
                &ac_levels,
            );
            mbs.push(targets);
        }
    }
    bw.rbsp_trailing_bits();
    let annex_b = build_nal_unit(nal_ref_idc, NalUnitType::SliceNonIdr, &bw.into_bytes());
    SpEncodedPicture {
        annex_b,
        targets: SpTargets {
            mbs,
            recon_y,
            recon_u,
            recon_v,
        },
    }
}

/// Emit one `P_L0_16x16` SP macroblock (mb_skip_run 0, zero mvd) from
/// raster-domain residual levels.
fn emit_sp_inter_mb(
    bw: &mut BitWriter,
    nc_grid: &mut CavlcNcGrid,
    mb_addr: u32,
    luma_cr: &[[i32; 16]; 16],
    dc_levels: &[[i32; 4]; 2],
    ac_levels: &[[[i32; 16]; 4]; 2],
) {
    let levels = pack_levels(luma_cr, dc_levels, ac_levels);
    let luma_nc = luma_nc_and_commit(nc_grid, mb_addr, false, &levels);
    let (nc_cb, nc_cr) = super::derive_chroma_ac_nc_and_commit_totals(
        nc_grid,
        mb_addr,
        false,
        levels.cbp_chroma,
        &levels.ac_cb_scan,
        &levels.ac_cr_scan,
        1,
    );
    bw.ue(0); // mb_skip_run
    write_p_l0_16x16_mb(
        bw,
        &PL016x16McbConfig {
            transform_size_8x8_flag: None,
            mvd_l0_x: 0,
            mvd_l0_y: 0,
            cbp_luma: levels.cbp_luma,
            cbp_chroma: levels.cbp_chroma,
            mb_qp_delta: 0,
            luma_4x4_levels: levels.luma_scan,
            luma_4x4_nc: luma_nc,
            chroma_dc_cb: levels.dc_cb,
            chroma_dc_cr: levels.dc_cr,
            chroma_ac_cb: levels.ac_cb_scan,
            chroma_ac_cr: levels.ac_cr_scan,
            chroma_ac_nc_cb: nc_cb,
            chroma_ac_nc_cr: nc_cr,
        },
        0,
    )
    .expect("SP P_L0_16x16 emit");
}

// ---------------------------------------------------------------------------
// Switching SP pictures (§8.6.2, sp_for_switch_flag == 1)
// ---------------------------------------------------------------------------

/// Encode a switching SP access unit that reproduces `targets`
/// bit-exactly while predicting from `refp` (a *different* stream's
/// decode state). Non-reference (`nal_ref_idc == 0`).
pub fn encode_sp_switch_picture(
    cfg: &SpConfig,
    targets: &SpTargets,
    refp: (&[u8], &[u8], &[u8]),
    frame_num: u32,
    poc_lsb: u32,
) -> Vec<u8> {
    let (w_mbs, h_mbs) = (cfg.width_mbs(), cfg.height_mbs());
    let width = cfg.width as usize;
    let cw = width / 2;
    let qs_c = qp_y_to_qp_c(cfg.qs, 0);
    // Existence gate for the exact chroma-DC reproduction (see the
    // module docs): eq. 8-431's scale must be integral.
    let _k = chroma_dc_switch_scale(qs_c, &FLAT)
        .expect("switching construction requires QSC >= 6 (choose qs >= 6)");

    let mut bw = BitWriter::new();
    write_sp_slice_header(&mut bw, cfg, frame_num, poc_lsb, 0, true);
    let mut nc_grid = CavlcNcGrid::new(w_mbs as u32, h_mbs as u32);

    for mby in 0..h_mbs {
        for mbx in 0..w_mbs {
            let mb_addr = (mby * w_mbs + mbx) as u32;
            let mb_targets = &targets.mbs[mb_addr as usize];
            let pred_mb = gather16(refp.0, width, mbx * 16, mby * 16);
            let mut luma_cr = [[0i32; 16]; 16];
            for blk in 0..16 {
                let (bx, by) = BLK4_XY[blk];
                let p = blk4(&pred_mb, bx, by);
                // §8.6.2.1 — cs = Q_QSY(T(pred)); transmit
                // cr = c_target − cs so the decoder's eq. 8-433 sum
                // lands on c_target exactly.
                let cs = sp_luma_switching(&p, &[0i32; 16], cfg.qs);
                for idx in 0..16 {
                    luma_cr[blk][idx] = mb_targets.luma_c[blk][idx] - cs[idx];
                }
            }
            let (dc_levels, ac_levels) =
                switch_chroma_levels(mb_targets, refp.1, refp.2, cw, mbx, mby, qs_c);
            emit_sp_inter_mb(
                &mut bw,
                &mut nc_grid,
                mb_addr,
                &luma_cr,
                &dc_levels,
                &ac_levels,
            );
        }
    }
    bw.rbsp_trailing_bits();
    build_nal_unit(0, NalUnitType::SliceNonIdr, &bw.into_bytes())
}

/// Chroma level construction shared by the switching SP and SI
/// encoders: AC levels are `c_target − cs`; DC levels are
/// `k · dcr_primary − dcs` (eq. 8-440's addend), where `dcr_primary`
/// is recovered exactly from the stored eq. 8-431-scaled DC slots.
#[allow(clippy::too_many_arguments)]
fn switch_chroma_levels(
    mb_targets: &SpMbTargets,
    ref_u: &[u8],
    ref_v: &[u8],
    cw: usize,
    mbx: usize,
    mby: usize,
    qs_c: i32,
) -> ([[i32; 4]; 2], [[[i32; 16]; 4]; 2]) {
    let mut dc_levels = [[0i32; 4]; 2];
    let mut ac_levels = [[[0i32; 16]; 4]; 2];
    for plane in 0..2 {
        let rp_ = if plane == 0 { ref_u } else { ref_v };
        let pred8 = gather8(rp_, cw, mbx * 8, mby * 8);
        switch_chroma_levels_from_pred(
            &pred8,
            &mb_targets.chroma_c[plane],
            qs_c,
            &mut dc_levels[plane],
            &mut ac_levels[plane],
        );
    }
    (dc_levels, ac_levels)
}

/// Per-component core of [`switch_chroma_levels`], usable with any
/// prediction source (inter for switching SP, §8.3.4 intra for SI).
fn switch_chroma_levels_from_pred(
    pred8: &[i32; 64],
    target_c: &[[i32; 16]; 4],
    qs_c: i32,
    dc_levels: &mut [i32; 4],
    ac_levels: &mut [[i32; 16]; 4],
) {
    // AC: cs from the §8.6.2.2 quantised prediction (zero levels in →
    // pure cs out), then cr = c_target − cs.
    let cs = sp_chroma_switching(pred8, &[0i32; 4], &[[0i32; 16]; 4], qs_c);
    for blk in 0..4 {
        for idx in 1..16 {
            ac_levels[blk][idx] = target_c[blk][idx] - cs[blk][idx];
        }
    }
    // DC: recover k·f_primary from the stored scaled DC slots
    // (raster: c00(blk) = k · f[blk]), then dcr_primary = H2(k·f)/4k
    // and the transmitted level (eq. 8-440) is k·dcr_primary − dcs.
    let kf = [
        target_c[0][0] as i64,
        target_c[1][0] as i64,
        target_c[2][0] as i64,
        target_c[3][0] as i64,
    ];
    let h_kf = hadamard_2x2(&kf); // = 4k · dcr_primary
    let cp_blocks = chroma_pred_transform(pred8);
    let dcp = hadamard_2x2(&[
        cp_blocks[0][0] as i64,
        cp_blocks[1][0] as i64,
        cp_blocks[2][0] as i64,
        cp_blocks[3][0] as i64,
    ]);
    for pos in 0..4 {
        debug_assert_eq!(h_kf[pos] % 4, 0);
        let k_dcr = h_kf[pos] / 4; // k · dcr_primary
        let dcs = quant_qs_dc(dcp[pos], qs_c);
        let level = k_dcr - dcs;
        dc_levels[pos] = i32::try_from(level).expect("chroma DC level fits i32");
    }
}

// ---------------------------------------------------------------------------
// SI pictures (§8.6.2, Intra_4x4 + Intra_Chroma prediction)
// ---------------------------------------------------------------------------

/// Encode an SI access unit reproducing `targets` bit-exactly with no
/// reference picture at all: every macroblock is the SI type
/// (Table 7-12, Intra_4x4-coded with the DC modes), predicting from
/// the already-switched samples — which, by the bit-exact identity,
/// are the primary picture's own reconstruction.
pub fn encode_si_picture(
    cfg: &SpConfig,
    targets: &SpTargets,
    frame_num: u32,
    poc_lsb: u32,
) -> Vec<u8> {
    let (w_mbs, h_mbs) = (cfg.width_mbs(), cfg.height_mbs());
    let width = cfg.width as usize;
    let cw = width / 2;
    let qs_c = qp_y_to_qp_c(cfg.qs, 0);
    // Existence gate for the exact chroma-DC reproduction (see the
    // module docs): eq. 8-431's scale must be integral.
    let _k = chroma_dc_switch_scale(qs_c, &FLAT)
        .expect("switching construction requires QSC >= 6 (choose qs >= 6)");

    let mut bw = BitWriter::new();
    write_si_slice_header(&mut bw, cfg, frame_num, poc_lsb);
    let mut nc_grid = CavlcNcGrid::new(w_mbs as u32, h_mbs as u32);

    for mby in 0..h_mbs {
        for mbx in 0..w_mbs {
            let mb_addr = (mby * w_mbs + mbx) as u32;
            let (luma_cr, dc_levels, ac_levels) =
                si_mb_levels(cfg, targets, mbx, mby, width, cw, qs_c);

            // ---- Emit the SI macroblock (write_i_nxn_mb emits
            // mb_type ue(0), which in an SI slice IS the Table 7-12 SI
            // macroblock type; the rest of the syntax is the shared
            // Intra_4x4 mb_pred + intra-CBP + residual layout).
            let levels = pack_levels(&luma_cr, &dc_levels, &ac_levels);
            let luma_nc = luma_nc_and_commit(&mut nc_grid, mb_addr, true, &levels);
            let (nc_cb, nc_cr) = super::derive_chroma_ac_nc_and_commit_totals(
                &mut nc_grid,
                mb_addr,
                true,
                levels.cbp_chroma,
                &levels.ac_cb_scan,
                &levels.ac_cr_scan,
                1,
            );
            write_i_nxn_mb(
                &mut bw,
                &INxNMcbConfig {
                    emit_transform_size_8x8_zero: false,
                    // All-DC modes: with every MB in the picture coded
                    // Intra_4x4 and every mode DC, the §8.3.1.1
                    // predicted mode is DC everywhere, so
                    // prev_intra4x4_pred_mode_flag = 1 suffices.
                    prev_intra4x4_pred_mode_flag: [true; 16],
                    rem_intra4x4_pred_mode: [0u8; 16],
                    intra_chroma_pred_mode: 0,
                    cbp_luma: levels.cbp_luma,
                    cbp_chroma: levels.cbp_chroma,
                    mb_qp_delta: 0,
                    luma_4x4_levels: levels.luma_scan,
                    luma_4x4_nc: luma_nc,
                    chroma_dc_cb: levels.dc_cb,
                    chroma_dc_cr: levels.dc_cr,
                    chroma_ac_cb: levels.ac_cb_scan,
                    chroma_ac_cr: levels.ac_cr_scan,
                    chroma_ac_nc_cb: nc_cb,
                    chroma_ac_nc_cr: nc_cr,
                },
            )
            .expect("SI macroblock emit");
        }
    }
    bw.rbsp_trailing_bits();
    build_nal_unit(0, NalUnitType::SliceNonIdr, &bw.into_bytes())
}

// ---------------------------------------------------------------------------
// §7.3.2.9 — data-partitioned emission (round 451).
//
// The Extended profile allows the coded data of a slice to be split
// into three NAL units: partition A (NAL 2 — slice header, slice_id
// and every category-2 element), partition B (NAL 3 — the residual of
// the intra collective types, I_PCM sample payloads included) and
// partition C (NAL 4 — the residual of the inter collective types).
// The emitters below produce BOTH forms of the same coded content —
// one single-NAL slice and one A(+B)(+C) triple — so a gate can
// require the decoder to reconstruct them identically.
// ---------------------------------------------------------------------------

/// Category-2 + residual level data of one prepared SP/SI macroblock.
struct PreppedMbLevels {
    levels: MbLevels,
    luma_nc: [i32; 16],
    nc_cb: [i32; 8],
    nc_cr: [i32; 8],
}

fn prep_mb_levels(
    nc_grid: &mut CavlcNcGrid,
    mb_addr: u32,
    is_intra: bool,
    luma_cr: &[[i32; 16]; 16],
    dc_levels: &[[i32; 4]; 2],
    ac_levels: &[[[i32; 16]; 4]; 2],
) -> PreppedMbLevels {
    let levels = pack_levels(luma_cr, dc_levels, ac_levels);
    let luma_nc = luma_nc_and_commit(nc_grid, mb_addr, is_intra, &levels);
    let (nc_cb, nc_cr) = super::derive_chroma_ac_nc_and_commit_totals(
        nc_grid,
        mb_addr,
        is_intra,
        levels.cbp_chroma,
        &levels.ac_cb_scan,
        &levels.ac_cr_scan,
        1,
    );
    PreppedMbLevels {
        levels,
        luma_nc,
        nc_cb,
        nc_cr,
    }
}

/// §7.3.5.3 — write the residual() of a prepared macroblock into the
/// given sink (the slice's own bitstream, or the partition-B/C
/// stream). Byte-for-byte the layout `write_p_l0_16x16_mb` /
/// `write_i_nxn_mb` emit after the category-2 prefix.
fn write_mb_residual(res: &mut BitWriter, p: &PreppedMbLevels) {
    for blk8 in 0..4u8 {
        if (p.levels.cbp_luma >> blk8) & 1 == 1 {
            for sub in 0..4u8 {
                let blk = (blk8 * 4 + sub) as usize;
                encode_residual_block_cavlc(
                    res,
                    CoeffTokenContext::Numeric(p.luma_nc[blk]),
                    16,
                    &p.levels.luma_scan[blk],
                )
                .expect("luma residual emit");
            }
        }
    }
    if p.levels.cbp_chroma > 0 {
        encode_residual_block_cavlc(res, CoeffTokenContext::ChromaDc420, 4, &p.levels.dc_cb)
            .expect("chroma DC emit");
        encode_residual_block_cavlc(res, CoeffTokenContext::ChromaDc420, 4, &p.levels.dc_cr)
            .expect("chroma DC emit");
    }
    if p.levels.cbp_chroma == 2 {
        for (k, blk) in p.levels.ac_cb_scan.iter().enumerate() {
            encode_residual_block_cavlc(
                res,
                CoeffTokenContext::Numeric(p.nc_cb[k]),
                15,
                &blk[..15],
            )
            .expect("chroma AC emit");
        }
        for (k, blk) in p.levels.ac_cr_scan.iter().enumerate() {
            encode_residual_block_cavlc(
                res,
                CoeffTokenContext::Numeric(p.nc_cr[k]),
                15,
                &blk[..15],
            )
            .expect("chroma AC emit");
        }
    }
}

/// Category-2 prefix of a `P_L0_16x16` SP macroblock (mb_skip_run 0,
/// zero mvd, `mb_qp_delta` 0).
fn write_inter_mb_cat2(a: &mut BitWriter, p: &PreppedMbLevels) {
    a.ue(0); // mb_skip_run
    a.ue(0); // mb_type — P_L0_16x16 (Table 7-13)
    a.se(0); // mvd_l0_x
    a.se(0); // mvd_l0_y
    a.ue(inter_cbp_to_codenum_420_422(
        p.levels.cbp_luma,
        p.levels.cbp_chroma,
    ));
    if p.levels.cbp_luma > 0 || p.levels.cbp_chroma > 0 {
        a.se(0); // mb_qp_delta
    }
}

/// Category-2 prefix of an SI macroblock (Table 7-12 raw 0 — the
/// all-DC `prev_intra4x4_pred_mode_flag = 1` shape the SI encoder
/// emits).
fn write_si_mb_cat2(a: &mut BitWriter, p: &PreppedMbLevels) {
    a.ue(0); // mb_type — SI (Table 7-12)
    for _ in 0..16 {
        a.u(1, 1); // prev_intra4x4_pred_mode_flag
    }
    a.ue(0); // intra_chroma_pred_mode = DC
    a.ue(intra_cbp_to_codenum_420_422(
        p.levels.cbp_luma,
        p.levels.cbp_chroma,
    ));
    if p.levels.cbp_luma > 0 || p.levels.cbp_chroma > 0 {
        a.se(0); // mb_qp_delta
    }
}

/// §7.3.5 — I_PCM sample payload (category 3: partition B in a
/// data-partitioned slice; alignment is relative to THAT bitstream).
fn write_ipcm_payload(b: &mut BitWriter, y_mb: &[u8; 256], cb_mb: &[u8; 64], cr_mb: &[u8; 64]) {
    b.align_to_byte_zero();
    for v in y_mb {
        b.u(8, u32::from(*v));
    }
    for v in cb_mb {
        b.u(8, u32::from(*v));
    }
    for v in cr_mb {
        b.u(8, u32::from(*v));
    }
}

/// Both emitted forms of one coded picture, plus the mirror
/// reconstruction shared by construction.
pub struct SpDualFormPicture {
    /// One-NAL coded slice access unit.
    pub single_annex_b: Vec<u8>,
    /// The same slice as §7.3.2.9 partitions (NAL 2 + optional 3/4).
    pub partitioned_annex_b: Vec<u8>,
    pub recon_y: Vec<u8>,
    pub recon_u: Vec<u8>,
    pub recon_v: Vec<u8>,
}

/// Encode one SP picture MIXING inter and I_PCM macroblocks (every
/// `pcm_period`-th macroblock is I_PCM — an I collective type, whose
/// sample payload is category 3), in both single-NAL and
/// data-partitioned form. With both intra and inter macroblocks in
/// one slice, the partitioned form exercises all three partitions
/// A + B + C.
#[allow(clippy::too_many_arguments)]
pub fn encode_sp_dual_form_picture(
    cfg: &SpConfig,
    src: (&[u8], &[u8], &[u8]),
    refp: (&[u8], &[u8], &[u8]),
    frame_num: u32,
    poc_lsb: u32,
    nal_ref_idc: u8,
    pcm_period: usize,
) -> SpDualFormPicture {
    assert!(pcm_period >= 1);
    let (w_mbs, h_mbs) = (cfg.width_mbs(), cfg.height_mbs());
    let width = cfg.width as usize;
    let cw = width / 2;
    let qp_c = qp_y_to_qp_c(cfg.qp, 0);
    let qs_c = qp_y_to_qp_c(cfg.qs, 0);

    // Single-NAL form.
    let mut w_single = BitWriter::new();
    write_sp_slice_header(
        &mut w_single,
        cfg,
        frame_num,
        poc_lsb,
        u32::from(nal_ref_idc),
        false,
    );
    // Partitioned form: A + B + C sinks.
    let mut w_a = BitWriter::new();
    write_sp_slice_header(
        &mut w_a,
        cfg,
        frame_num,
        poc_lsb,
        u32::from(nal_ref_idc),
        false,
    );
    w_a.ue(0); // §7.3.2.9.1 slice_id
               // §7.3.2.9.2/.3 — each partition's RBSP is one continuous
               // bitstream: slice_id first (our PPS carries no redundant_pic_cnt
               // and the SPS no separate colour planes), then that category's
               // slice-data bits.
    let mut w_b = BitWriter::new();
    w_b.ue(0); // slice_id
    let mut w_c = BitWriter::new();
    w_c.ue(0); // slice_id
    let (mut b_used, mut c_used) = (false, false);

    // The two forms must carry IDENTICAL macroblock data, so levels +
    // nC state are computed once per MB (two independent grids would
    // drift only if the emission differed — it must not).
    let mut nc_grid = CavlcNcGrid::new(w_mbs as u32, h_mbs as u32);
    let mut recon_y = vec![0u8; width * cfg.height as usize];
    let mut recon_u = vec![0u8; cw * (cfg.height as usize / 2)];
    let mut recon_v = vec![0u8; cw * (cfg.height as usize / 2)];

    for mby in 0..h_mbs {
        for mbx in 0..w_mbs {
            let mb_addr = (mby * w_mbs + mbx) as u32;
            if (mb_addr as usize) % pcm_period == pcm_period - 1 {
                // ---- I_PCM macroblock: recon = source samples.
                let mut y_mb = [0u8; 256];
                let mut cb_mb = [0u8; 64];
                let mut cr_mb = [0u8; 64];
                for row in 0..16 {
                    for col in 0..16 {
                        let v = src.0[(mby * 16 + row) * width + mbx * 16 + col];
                        y_mb[row * 16 + col] = v;
                        recon_y[(mby * 16 + row) * width + mbx * 16 + col] = v;
                    }
                }
                for row in 0..8 {
                    for col in 0..8 {
                        let u = src.1[(mby * 8 + row) * cw + mbx * 8 + col];
                        let v = src.2[(mby * 8 + row) * cw + mbx * 8 + col];
                        cb_mb[row * 8 + col] = u;
                        cr_mb[row * 8 + col] = v;
                        recon_u[(mby * 8 + row) * cw + mbx * 8 + col] = u;
                        recon_v[(mby * 8 + row) * cw + mbx * 8 + col] = v;
                    }
                }
                // nC bookkeeping: I_PCM contributes nN = 16 (§9.2.1.1).
                {
                    let cur = &mut nc_grid.mbs[mb_addr as usize];
                    cur.is_available = true;
                    cur.is_intra = true;
                    cur.is_skip = false;
                    cur.is_i_pcm = true;
                }
                // Single form: skip run + mb_type + payload inline.
                w_single.ue(0); // mb_skip_run
                w_single.ue(30); // Table 7-13: I types at 5 + raw; I_PCM = 5 + 25
                write_ipcm_payload(&mut w_single, &y_mb, &cb_mb, &cr_mb);
                // Partitioned: category 2 in A, payload in B.
                w_a.ue(0);
                w_a.ue(30);
                write_ipcm_payload(&mut w_b, &y_mb, &cb_mb, &cr_mb);
                b_used = true;
                continue;
            }

            // ---- P_L0_16x16 SP macroblock (as in encode_sp_picture).
            let pred_mb = gather16(refp.0, width, mbx * 16, mby * 16);
            let src_mb = gather16(src.0, width, mbx * 16, mby * 16);
            let mut luma_cr = [[0i32; 16]; 16];
            let mut luma_c = [[0i32; 16]; 16];
            for blk in 0..16 {
                let (bx, by) = BLK4_XY[blk];
                let p = blk4(&pred_mb, bx, by);
                let s_blk = blk4(&src_mb, bx, by);
                let cp = crate::sp_transform::forward_core_4x4(&p);
                let cwc = crate::sp_transform::forward_core_4x4(&s_blk);
                let mut cr = [0i32; 16];
                for i in 0..4 {
                    for j in 0..4 {
                        let idx = i * 4 + j;
                        cr[idx] =
                            encoder_quant_8_416((cwc[idx] - cp[idx]) as i64, cfg.qp, &FLAT, i, j);
                    }
                }
                luma_cr[blk] = cr;
                luma_c[blk] = sp_luma_non_switching(&p, &cr, cfg.qp, cfg.qs, &FLAT);
            }
            let mut dc_levels = [[0i32; 4]; 2];
            let mut ac_levels = [[[0i32; 16]; 4]; 2];
            let mut chroma_c = [[[0i32; 16]; 4]; 2];
            for plane in 0..2 {
                let (sp_, rp_) = if plane == 0 {
                    (src.1, refp.1)
                } else {
                    (src.2, refp.2)
                };
                let pred8 = gather8(rp_, cw, mbx * 8, mby * 8);
                let src8 = gather8(sp_, cw, mbx * 8, mby * 8);
                let cp_blocks = chroma_pred_transform(&pred8);
                let cw_blocks = chroma_pred_transform(&src8);
                for blk in 0..4 {
                    for i in 0..4 {
                        for j in 0..4 {
                            if i == 0 && j == 0 {
                                continue;
                            }
                            let idx = i * 4 + j;
                            ac_levels[plane][blk][idx] = encoder_quant_8_416(
                                (cw_blocks[blk][idx] - cp_blocks[blk][idx]) as i64,
                                qp_c,
                                &FLAT,
                                i,
                                j,
                            );
                        }
                    }
                }
                let dcp = hadamard_2x2(&[
                    cp_blocks[0][0] as i64,
                    cp_blocks[1][0] as i64,
                    cp_blocks[2][0] as i64,
                    cp_blocks[3][0] as i64,
                ]);
                let dcw = hadamard_2x2(&[
                    cw_blocks[0][0] as i64,
                    cw_blocks[1][0] as i64,
                    cw_blocks[2][0] as i64,
                    cw_blocks[3][0] as i64,
                ]);
                for pos in 0..4 {
                    dc_levels[plane][pos] = encoder_quant_8_428(dcw[pos] - dcp[pos], qp_c, &FLAT);
                }
                chroma_c[plane] = sp_chroma_non_switching(
                    &pred8,
                    &dc_levels[plane],
                    &ac_levels[plane],
                    qp_c,
                    qs_c,
                    &FLAT,
                );
            }
            let targets = SpMbTargets { luma_c, chroma_c };
            mirror_recon_mb(
                &targets,
                cfg.qs,
                qs_c,
                mbx,
                mby,
                width,
                &mut recon_y,
                &mut recon_u,
                &mut recon_v,
            );
            let prepped = prep_mb_levels(
                &mut nc_grid,
                mb_addr,
                false,
                &luma_cr,
                &dc_levels,
                &ac_levels,
            );
            // Single form: category 2 + residual inline.
            write_inter_mb_cat2(&mut w_single, &prepped);
            write_mb_residual(&mut w_single, &prepped);
            // Partitioned: category 2 in A, residual (category 4) in C.
            write_inter_mb_cat2(&mut w_a, &prepped);
            if prepped.levels.cbp_luma > 0 || prepped.levels.cbp_chroma > 0 {
                write_mb_residual(&mut w_c, &prepped);
                c_used = true;
            }
        }
    }

    w_single.rbsp_trailing_bits();
    let single_annex_b = build_nal_unit(
        nal_ref_idc,
        NalUnitType::SliceNonIdr,
        &w_single.into_bytes(),
    );

    w_a.rbsp_trailing_bits();
    let mut partitioned_annex_b = build_nal_unit(
        nal_ref_idc,
        NalUnitType::SliceDataPartitionA,
        &w_a.into_bytes(),
    );
    // §7.4.2.9.2/.3 — a partition RBSP is present exactly when the
    // category-2 elements indicate data of its category.
    if b_used {
        let mut body = w_b;
        body.rbsp_trailing_bits();
        partitioned_annex_b.extend_from_slice(&build_nal_unit(
            nal_ref_idc,
            NalUnitType::SliceDataPartitionB,
            &body.into_bytes(),
        ));
    }
    if c_used {
        let mut body = w_c;
        body.rbsp_trailing_bits();
        partitioned_annex_b.extend_from_slice(&build_nal_unit(
            nal_ref_idc,
            NalUnitType::SliceDataPartitionC,
            &body.into_bytes(),
        ));
    }

    SpDualFormPicture {
        single_annex_b,
        partitioned_annex_b,
        recon_y,
        recon_u,
        recon_v,
    }
}

/// Per-MB SI level computation shared by [`encode_si_picture`] and
/// the dual-form emitter: DC Intra_4x4 / Intra_Chroma prediction
/// against the target reconstruction (bit-exact identity — the
/// decoder's progressively-built samples ARE the primary picture's),
/// then `cr = c_target − cs` per §8.6.2.
#[allow(clippy::type_complexity)]
fn si_mb_levels(
    cfg: &SpConfig,
    targets: &SpTargets,
    mbx: usize,
    mby: usize,
    width: usize,
    cw: usize,
    qs_c: i32,
) -> ([[i32; 16]; 16], [[i32; 4]; 2], [[[i32; 16]; 4]; 2]) {
    let mb_addr = (mby * (width / 16) + mbx) as u32;
    let mb_targets = &targets.mbs[mb_addr as usize];
    let ry = &targets.recon_y;
    let ru = &targets.recon_u;
    let rv = &targets.recon_v;

    let mut luma_cr = [[0i32; 16]; 16];
    for blk in 0..16 {
        let (bx, by) = BLK4_XY[blk];
        let (gx, gy) = (mbx * 16 + bx, mby * 16 + by);
        let av = Neighbour4x4Availability {
            top_left: gx > 0 && gy > 0,
            top: gy > 0,
            // DC never reads top-right; keep the flag exact for the
            // in-picture cases that matter to it anyway.
            top_right: false,
            left: gx > 0,
        };
        let samples = Samples4x4 {
            top_left: if av.top_left {
                ry[(gy - 1) * width + gx - 1] as i32
            } else {
                0
            },
            top: core::array::from_fn(|i| {
                if av.top {
                    ry[(gy - 1) * width + gx + i] as i32
                } else {
                    0
                }
            }),
            top_right: [0i32; 4],
            left: core::array::from_fn(|i| {
                if av.left {
                    ry[(gy + i) * width + gx - 1] as i32
                } else {
                    0
                }
            }),
            availability: av,
        };
        let mut pred = [0i32; 16];
        predict_4x4(Intra4x4Mode::Dc, &samples, 8, &mut pred);
        let cs = sp_luma_switching(&pred, &[0i32; 16], cfg.qs);
        for idx in 0..16 {
            luma_cr[blk][idx] = mb_targets.luma_c[blk][idx] - cs[idx];
        }
    }

    let mut dc_levels = [[0i32; 4]; 2];
    let mut ac_levels = [[[0i32; 16]; 4]; 2];
    for plane in 0..2 {
        let rp_: &[u8] = if plane == 0 { ru } else { rv };
        let (cx, cy) = (mbx * 8, mby * 8);
        let av = Neighbour4x4Availability {
            top_left: cx > 0 && cy > 0,
            top: cy > 0,
            top_right: false,
            left: cx > 0,
        };
        let samples = SamplesChroma {
            top_left: if av.top_left {
                rp_[(cy - 1) * cw + cx - 1] as i32
            } else {
                0
            },
            top: (0..8)
                .map(|i| {
                    if av.top {
                        rp_[(cy - 1) * cw + cx + i] as i32
                    } else {
                        0
                    }
                })
                .collect(),
            left: (0..8)
                .map(|i| {
                    if av.left {
                        rp_[(cy + i) * cw + cx - 1] as i32
                    } else {
                        0
                    }
                })
                .collect(),
            availability: av,
        };
        let mut pred = vec![0i32; 64];
        predict_chroma(
            IntraChromaMode::Dc,
            &samples,
            ChromaArrayType::Yuv420,
            8,
            &mut pred,
        );
        let mut pred8 = [0i32; 64];
        pred8.copy_from_slice(&pred);
        switch_chroma_levels_from_pred(
            &pred8,
            &mb_targets.chroma_c[plane],
            qs_c,
            &mut dc_levels[plane],
            &mut ac_levels[plane],
        );
    }
    (luma_cr, dc_levels, ac_levels)
}

/// Encode an SI picture in both single-NAL and §7.3.2.9 partitioned
/// (A + B) form — every SI macroblock is an intra collective type, so
/// the whole residual is category 3 and lands in partition B.
pub fn encode_si_dual_form_picture(
    cfg: &SpConfig,
    targets: &SpTargets,
    frame_num: u32,
    poc_lsb: u32,
) -> SpDualFormPicture {
    let (w_mbs, h_mbs) = (cfg.width_mbs(), cfg.height_mbs());
    let width = cfg.width as usize;
    let cw = width / 2;
    let qs_c = qp_y_to_qp_c(cfg.qs, 0);
    let _k = chroma_dc_switch_scale(qs_c, &FLAT)
        .expect("switching construction requires QSC >= 6 (choose qs >= 6)");

    let mut w_single = BitWriter::new();
    write_si_slice_header(&mut w_single, cfg, frame_num, poc_lsb);
    let mut w_a = BitWriter::new();
    write_si_slice_header(&mut w_a, cfg, frame_num, poc_lsb);
    w_a.ue(0); // §7.3.2.9.1 slice_id
    let mut w_b = BitWriter::new();
    w_b.ue(0); // slice_id
    let mut b_used = false;

    let mut nc_grid = CavlcNcGrid::new(w_mbs as u32, h_mbs as u32);
    for mby in 0..h_mbs {
        for mbx in 0..w_mbs {
            let mb_addr = (mby * w_mbs + mbx) as u32;
            let (luma_cr, dc_levels, ac_levels) =
                si_mb_levels(cfg, targets, mbx, mby, width, cw, qs_c);
            let prepped = prep_mb_levels(
                &mut nc_grid,
                mb_addr,
                true,
                &luma_cr,
                &dc_levels,
                &ac_levels,
            );
            write_si_mb_cat2(&mut w_single, &prepped);
            write_mb_residual(&mut w_single, &prepped);
            write_si_mb_cat2(&mut w_a, &prepped);
            if prepped.levels.cbp_luma > 0 || prepped.levels.cbp_chroma > 0 {
                write_mb_residual(&mut w_b, &prepped);
                b_used = true;
            }
        }
    }

    w_single.rbsp_trailing_bits();
    let single_annex_b = build_nal_unit(0, NalUnitType::SliceNonIdr, &w_single.into_bytes());
    w_a.rbsp_trailing_bits();
    let mut partitioned_annex_b =
        build_nal_unit(0, NalUnitType::SliceDataPartitionA, &w_a.into_bytes());
    if b_used {
        let mut body = w_b;
        body.rbsp_trailing_bits();
        partitioned_annex_b.extend_from_slice(&build_nal_unit(
            0,
            NalUnitType::SliceDataPartitionB,
            &body.into_bytes(),
        ));
    }
    SpDualFormPicture {
        single_annex_b,
        partitioned_annex_b,
        recon_y: targets.recon_y.clone(),
        recon_u: targets.recon_u.clone(),
        recon_v: targets.recon_v.clone(),
    }
}
