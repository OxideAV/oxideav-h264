//! Round-456 — **high-bit-depth encoder** (10 / 12 / 14-bit samples).
//!
//! The 8-bit encoder pipeline works on `u8` planes end to end; this
//! module is the `>8-bit` twin for the High 10 (110) / High 4:2:2
//! (122) / High 4:4:4 Predictive (244) profiles (§A.2.4 / §A.2.6 /
//! §A.2.7 — only the last admits `bit_depth_*_minus8 > 2`, i.e. 12-
//! and 14-bit). Samples are `u16` in, `u16` out; every arithmetic
//! stage runs the decoder's own bit-depth-aware processes:
//!
//! * §7.4.2.1.1 `BitDepthY / BitDepthC`, `QpBdOffsetY = 6 ·
//!   bit_depth_luma_minus8`, `QpBdOffsetC`; the slice QP_Y may go
//!   down to `−QpBdOffsetY` (§7.4.3, eq. 7-30) and every
//!   transform / scaling stage runs at `QP′Y = QPY + QpBdOffsetY`
//!   (§8.5.12) — the forward quantiser's `qBits = 15 + QP′/6` and
//!   `MF(QP′ % 6)` extend past 51 exactly like the decoder's
//!   LevelScale.
//! * §8.5.8 chroma QP: `qPI = Clip3(−QpBdOffsetC, 51, QPY + offset)`,
//!   Table 8-15 above 29, `QP′C = QPC + QpBdOffsetC`; §8.5.11.2's
//!   `QP′C,DC = QP′C + 3` at 4:2:2 (inside the shared Hadamard helper).
//! * §8.3.3 / §8.3.4 intra prediction with the `1 << (BitDepth − 1)`
//!   DC default and Clip1 at the sample range; §8.4.2.2 quarter-pel
//!   luma / eighth-pel chroma interpolation with `Clip1` at
//!   `(1 << BitDepth) − 1`; §8.7 deblocking with the Table 8-16 /
//!   8-17 α / β / tC0 values scaled by `1 << (BitDepth − 8)`.
//! * §9.2 CAVLC is depth-agnostic apart from the `level_prefix ≥ 16`
//!   escape (§9.2.2.1, High profiles) which the residual writer
//!   already produces for large levels.
//!
//! Chroma formats 4:2:0 / 4:2:2 (§8.5.11.1 2x2 / §8.5.11.2 2x4 chroma
//! DC transforms) and 4:4:4 (chroma coded like luma, §8.3.4.5 /
//! §7.3.5.3). Mode set: I pictures Intra_16x16; P pictures P_Skip /
//! P_L0_16x16 with an optional Intra_16x16 fallback. Single slice,
//! single reference, fixed QP.

use crate::cabac_ctx::{BlockType, MvdComponent, SliceKind};
use crate::cavlc::CoeffTokenContext;
use crate::encoder::cabac_syntax::encode_residual_block_cabac_field;
use crate::encoder::cabac_syntax::encode_transform_size_8x8_flag;
use crate::encoder::cabac_syntax::{
    encode_coded_block_pattern, encode_end_of_slice_flag, encode_intra_chroma_pred_mode,
    encode_mb_qp_delta, encode_mb_skip_flag, encode_mb_type_i, encode_mb_type_p, encode_mvd_lx,
};
use crate::encoder::deblock::{
    chroma_nz_mask_from_blocks, deblock_recon_deep, luma_nz_mask_from_blocks, MbDeblockInfo,
};
use crate::encoder::macroblock::{
    write_intra16x16_mb_chroma, write_intra16x16_mb_in_inter_slice_chroma,
    write_p_l0_16x16_mb_chroma, ChromaWriteKind, PL016x16McbConfig,
};
use crate::encoder::mbaff_cabac::{
    emit_chroma_cabac, emit_i16_plane_cabac, emit_inter_plane_cabac, full_neighbour_ctx,
    skip_neighbour_ctx, CabacState, Plane,
};
use crate::encoder::nal::build_nal_unit;
use crate::encoder::pps::{build_baseline_pps_rbsp, BaselinePpsConfig};
use crate::encoder::slice::CabacSliceParams;
use crate::encoder::slice::{
    write_idr_i_slice_header, write_p_slice_header, FieldPicSignal, IdrSliceHeaderConfig,
    PSliceHeaderConfig,
};
use crate::encoder::sps::{build_sps_rbsp_with_bypass, BaselineSpsConfig};
use crate::encoder::transform::{
    forward_core_4x4, forward_core_8x8, forward_hadamard_2x2, forward_hadamard_4x2,
    forward_hadamard_4x4, quantize_4x4_ac_w, quantize_4x4_w, quantize_8x8_w,
    quantize_chroma_dc_422_w, quantize_chroma_dc_w, quantize_luma_dc_w, scan_chroma_dc_422,
    zigzag_scan_4x4, zigzag_scan_4x4_ac, zigzag_scan_8x8,
};
use crate::encoder::{
    derive_chroma_ac_nc_and_commit_totals, derive_plane_nc_444_i16x16, derive_plane_nc_444_inter,
    min_level_idc_for_picture_size, mvp_for_16x16, p_skip_mv, BitWriter, MvGrid, MvGridSlot,
    ScalingMatrixMode,
};
use crate::inter_pred::{interpolate_chroma, interpolate_luma};
use crate::intra_pred::{
    intra_16x16_mode_permitted, intra_chroma_mode_permitted, predict_16x16, predict_chroma,
    ChromaArrayType, Intra16x16Mode, IntraChromaMode, Neighbour4x4Availability, Samples16x16,
    SamplesChroma,
};
use crate::macroblock_layer::{cabac_mvd_abs_sum, CabacMbNeighbourInfo, CabacNeighbourGrid};
use crate::macroblock_layer::{derive_nc_luma, CavlcNcGrid, LumaNcKind};
use crate::mv_deriv::Mv;
use crate::nal::NalUnitType;
use crate::transform::{
    intra_bypass_dpcm, inverse_hadamard_chroma_dc_420, inverse_hadamard_chroma_dc_422,
    inverse_hadamard_luma_dc_16x16, inverse_transform_4x4, inverse_transform_4x4_dc_preserved,
    inverse_transform_8x8, qp_bd_offset, qp_y_to_qp_c_with_bd_offset,
};

/// §6.4.3 Figure 6-10 — luma4x4BlkIdx → (bx, by) in 4-sample units.
const LUMA_4X4_BLK: [(usize, usize); 16] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (3, 0),
    (2, 1),
    (3, 1),
    (0, 2),
    (1, 2),
    (0, 3),
    (1, 3),
    (2, 2),
    (3, 2),
    (2, 3),
    (3, 3),
];

/// Configuration for [`encode_deep_sequence`].
#[derive(Debug, Clone)]
pub struct DeepConfig {
    /// Luma width / height in samples (multiples of 16).
    pub width: u32,
    pub height: u32,
    /// §7.4.2.1.1 `BitDepthY` (8..=14).
    pub bit_depth_luma: u32,
    /// §7.4.2.1.1 `BitDepthC` (8..=14).
    pub bit_depth_chroma: u32,
    /// `chroma_format_idc` ∈ {1, 2, 3}.
    pub chroma_format_idc: u32,
    /// Slice QP_Y, in `−QpBdOffsetY..=51`.
    pub qp: i32,
    /// Frames after the IDR are P pictures.
    pub p_frames: bool,
    /// Allow Intra_16x16 macroblocks inside P pictures.
    pub intra_in_p: bool,
    /// §8.7 in-loop deblocking (`disable_deblocking_filter_idc = 0`).
    pub deblock: bool,
    /// Lossless coding: SPS `qpprime_y_zero_transform_bypass_flag = 1`
    /// with `qp == −QpBdOffsetY` (QP′Y == 0) — every transform /
    /// scaling stage is bypassed (§8.5.10 eq. 8-319, §8.5.11 eq.
    /// 8-323, §8.5.12 eq. 8-334) and V/H intra residuals are coded as
    /// §8.5.15 DPCM differences; the reconstruction equals the source
    /// exactly.
    pub lossless: bool,
    /// Lossless interoperability mode: never select the intra modes
    /// that invoke the §8.5.15 DPCM (Intra_16x16 vertical /
    /// horizontal, chroma horizontal / vertical — DC and Plane remain)
    /// so the stream reconstructs losslessly even in decoders that skip
    /// §8.5.2 step 3 / §8.5.4 step 3. Round-456 finding: the stock
    /// black-box reference decoder applies the transform-bypass DPCM
    /// to Intra_4x4 / Intra_8x8 only — spec-literal Intra_16x16 or
    /// chroma DPCM streams decode with an accumulated-residual error
    /// there while our decoder reconstructs them exactly. Ignored when
    /// `lossless` is off.
    pub lossless_interop: bool,
    /// `entropy_coding_mode_flag = 1`: code the same macroblock
    /// decisions with CABAC (§9.3) instead of CAVLC.
    pub cabac: bool,
    /// High-profile 8x8 transform (`transform_8x8_mode_flag = 1`): every
    /// other coded P_L0_16x16 macroblock carries its luma residual
    /// through the §8.6.4 / §8.5.13 8x8 transform at QP′ (4:2:0 and
    /// 4:2:2; the 4:4:4 chroma-as-luma planes stay 4x4 — the option is
    /// ignored at ChromaArrayType 3). CAVLC codes the §7.4.5.3.3
    /// four-4x4 split, CABAC the ctxBlockCat-5 residual; under bypass
    /// the 8x8 block is coded as the raw residual (§8.5.13 identity).
    pub transform_8x8: bool,
}

/// One picture's `(Y, Cb, Cr)` `u16` planes.
pub type DeepPlanes = (Vec<u16>, Vec<u16>, Vec<u16>);

/// Output of [`encode_deep_sequence`].
#[derive(Debug, Clone)]
pub struct DeepEncoded {
    /// Annex B stream: SPS, PPS, one slice NAL per frame.
    pub annex_b: Vec<u8>,
    /// Deblocked reconstruction of every frame — what a conforming
    /// decoder outputs (as `u16` samples).
    pub recon_frames: Vec<DeepPlanes>,
    /// `profile_idc` selected for the stream (110 / 122 / 244).
    pub profile_idc: u8,
    /// P_Skip macroblocks across the sequence.
    pub skipped_mbs: usize,
    /// Intra_16x16 macroblocks inside P pictures.
    pub intra_mbs_in_p: usize,
    /// Macroblocks coded with `transform_size_8x8_flag = 1`.
    pub mbs_8x8: usize,
}

/// Picture geometry.
#[derive(Clone, Copy)]
struct Geom {
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    /// Chroma MB tile (MbWidthC, MbHeightC).
    ct_w: usize,
    ct_h: usize,
    w_mbs: usize,
    cf: u32,
    bd_y: u32,
    bd_c: u32,
}

impl Geom {
    fn chroma_array_type(self) -> Option<ChromaArrayType> {
        match self.cf {
            1 => Some(ChromaArrayType::Yuv420),
            2 => Some(ChromaArrayType::Yuv422),
            _ => None,
        }
    }
}

/// Clip1 at the plane's bit depth.
#[inline]
fn clip(v: i32, bd: u32) -> i32 {
    v.clamp(0, (1 << bd) - 1)
}

// ---------------------------------------------------------------------------
// Intra prediction (decoder processes on the encoder's recon planes).
// ---------------------------------------------------------------------------

fn luma_samples16(recon: &[i32], stride: usize, mb_x: usize, mb_y: usize) -> Samples16x16 {
    let x0 = mb_x * 16;
    let y0 = mb_y * 16;
    let top_avail = mb_y > 0;
    let left_avail = mb_x > 0;
    let mut top = [0i32; 16];
    let mut left = [0i32; 16];
    if top_avail {
        top.copy_from_slice(&recon[(y0 - 1) * stride + x0..(y0 - 1) * stride + x0 + 16]);
    }
    if left_avail {
        for (j, l) in left.iter_mut().enumerate() {
            *l = recon[(y0 + j) * stride + x0 - 1];
        }
    }
    let top_left = if top_avail && left_avail {
        recon[(y0 - 1) * stride + x0 - 1]
    } else {
        0
    };
    Samples16x16 {
        top_left,
        top,
        left,
        availability: Neighbour4x4Availability {
            top_left: top_avail && left_avail,
            top: top_avail,
            top_right: false,
            left: left_avail,
        },
    }
}

fn chroma_samples(
    recon: &[i32],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
    ct_w: usize,
    ct_h: usize,
) -> SamplesChroma {
    let x0 = mb_x * ct_w;
    let y0 = mb_y * ct_h;
    let top_avail = mb_y > 0;
    let left_avail = mb_x > 0;
    let top: Vec<i32> = if top_avail {
        recon[(y0 - 1) * stride + x0..(y0 - 1) * stride + x0 + ct_w].to_vec()
    } else {
        vec![0; ct_w]
    };
    let left: Vec<i32> = if left_avail {
        (0..ct_h)
            .map(|j| recon[(y0 + j) * stride + x0 - 1])
            .collect()
    } else {
        vec![0; ct_h]
    };
    let top_left = if top_avail && left_avail {
        recon[(y0 - 1) * stride + x0 - 1]
    } else {
        0
    };
    SamplesChroma {
        top_left,
        top,
        left,
        availability: Neighbour4x4Availability {
            top_left: top_avail && left_avail,
            top: top_avail,
            top_right: false,
            left: left_avail,
        },
    }
}

/// Table 8-4 — pick the Intra_16x16 mode by SAD on one 16x16 tile
/// (`src` row-major 256). Returns `(mode index, prediction, sad)`.
/// `allow_dpcm == false` withholds the V / H modes (the §8.5.15 DPCM
/// triggers under transform bypass — see [`DeepConfig::lossless_interop`]).
fn pick_16x16(
    src: &[i32; 256],
    samples: &Samples16x16,
    bd: u32,
    allow_dpcm: bool,
) -> (u8, [i32; 256], u64) {
    let mut best: Option<(u8, [i32; 256], u64)> = None;
    for idx in 0..4u8 {
        let mode = Intra16x16Mode::from_index(idx).expect("mode index");
        if !intra_16x16_mode_permitted(mode, &samples.availability) {
            continue;
        }
        if !allow_dpcm && idx <= 1 {
            continue;
        }
        let mut pred = [0i32; 256];
        predict_16x16(mode, samples, bd, &mut pred);
        let sad: u64 = src
            .iter()
            .zip(pred.iter())
            .map(|(&s, &p)| (s - p).unsigned_abs() as u64)
            .sum();
        if best.as_ref().map_or(true, |b| sad < b.2) {
            best = Some((idx, pred, sad));
        }
    }
    best.expect("DC is always permitted")
}

/// Table 8-5 — joint Cb+Cr chroma mode election (4:2:0 / 4:2:2).
fn pick_chroma(
    src_u: &[i32],
    src_v: &[i32],
    su: &SamplesChroma,
    sv: &SamplesChroma,
    cat: ChromaArrayType,
    bd: u32,
    allow_dpcm: bool,
) -> (u8, Vec<i32>, Vec<i32>) {
    let n = src_u.len();
    let mut best: Option<(u8, Vec<i32>, Vec<i32>, u64)> = None;
    for idx in 0..4u8 {
        let mode = IntraChromaMode::from_index(idx).expect("mode index");
        if !intra_chroma_mode_permitted(mode, &su.availability) {
            continue;
        }
        if !allow_dpcm && (idx == 1 || idx == 2) {
            continue;
        }
        let mut pu = vec![0i32; n];
        let mut pv = vec![0i32; n];
        predict_chroma(mode, su, cat, bd, &mut pu);
        predict_chroma(mode, sv, cat, bd, &mut pv);
        let sad: u64 = src_u
            .iter()
            .zip(pu.iter())
            .chain(src_v.iter().zip(pv.iter()))
            .map(|(&s, &p)| (s - p).unsigned_abs() as u64)
            .sum();
        if best.as_ref().map_or(true, |b| sad < b.3) {
            best = Some((idx, pu, pv, sad));
        }
    }
    let (idx, pu, pv, _) = best.expect("DC is always permitted");
    (idx, pu, pv)
}

// ---------------------------------------------------------------------------
// Residual chains (forward quantisation mirrored by the decoder inverse).
// ---------------------------------------------------------------------------

/// §8.5.10 Intra_16x16 luma-like chain on one 16x16 residual tile at
/// `qp` = QP′ (bit-depth-offset applied).
struct I16Chain {
    dc_raster: [i32; 16],
    /// AC-only scan lists in luma4x4BlkIdx order.
    ac_scan: [[i32; 16]; 16],
    any_ac: bool,
    /// Reconstructed residual (row-major 256) for the chosen
    /// `cbp_luma` (AC dropped when none is coded).
    recon: [i32; 256],
    blk_has_nz: [bool; 16],
}

fn i16_chain(residual: &[i32; 256], qp: i32, w4: &[i32; 16], bd: u32) -> I16Chain {
    let mut coeffs = [[0i32; 16]; 16]; // raster-indexed 4x4 blocks
    for (raster, c) in coeffs.iter_mut().enumerate() {
        let (bx, by) = (raster % 4, raster / 4);
        let mut block = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                block[j * 4 + i] = residual[(by * 4 + j) * 16 + bx * 4 + i];
            }
        }
        *c = forward_core_4x4(&block);
    }
    let mut dc = [0i32; 16];
    for (raster, c) in coeffs.iter().enumerate() {
        dc[raster] = c[0];
    }
    let dc_raster = quantize_luma_dc_w(&forward_hadamard_4x4(&dc), qp, true, w4[0]);
    let mut ac_quant = [[0i32; 16]; 16]; // raster-indexed
    let mut ac_scan = [[0i32; 16]; 16];
    let mut blk_has_nz = [false; 16];
    let mut any_ac = false;
    for (blkz, &(bx, by)) in LUMA_4X4_BLK.iter().enumerate() {
        let raster = by * 4 + bx;
        let z = quantize_4x4_ac_w(&coeffs[raster], qp, true, w4);
        ac_quant[raster] = z;
        let s = zigzag_scan_4x4_ac(&z);
        blk_has_nz[blkz] = s[..15].iter().any(|&v| v != 0);
        any_ac |= blk_has_nz[blkz];
        ac_scan[blkz] = s;
    }
    let inv_dc = inverse_hadamard_luma_dc_16x16(&dc_raster, qp, w4, bd).expect("luma DC inverse");
    let mut recon = [0i32; 256];
    for raster in 0..16usize {
        let (bx, by) = (raster % 4, raster / 4);
        let mut c = if any_ac { ac_quant[raster] } else { [0i32; 16] };
        c[0] = inv_dc[raster];
        let r = inverse_transform_4x4_dc_preserved(&c, qp, w4, bd).expect("inverse 4x4");
        for j in 0..4 {
            for i in 0..4 {
                recon[(by * 4 + j) * 16 + bx * 4 + i] = r[j * 4 + i];
            }
        }
    }
    if !any_ac {
        blk_has_nz = [false; 16];
    }
    I16Chain {
        dc_raster,
        ac_scan,
        any_ac,
        recon,
        blk_has_nz,
    }
}

/// Inter luma-like chain: 16 full 4x4 blocks (§8.5.12 at QP′).
struct InterChain {
    /// Full 16-entry scan lists in luma4x4BlkIdx order.
    scan: [[i32; 16]; 16],
    blk_has_nz: [bool; 16],
    /// Per-quadrant coded bits.
    cbp: u8,
    recon: [i32; 256],
}

fn inter_chain(residual: &[i32; 256], qp: i32, w4: &[i32; 16], bd: u32) -> InterChain {
    let mut scan = [[0i32; 16]; 16];
    let mut blk_has_nz = [false; 16];
    let mut recon = [0i32; 256];
    for (blkz, &(bx, by)) in LUMA_4X4_BLK.iter().enumerate() {
        let mut block = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                block[j * 4 + i] = residual[(by * 4 + j) * 16 + bx * 4 + i];
            }
        }
        let q = quantize_4x4_w(&forward_core_4x4(&block), qp, false, w4);
        let s = zigzag_scan_4x4(&q);
        blk_has_nz[blkz] = s.iter().any(|&v| v != 0);
        scan[blkz] = s;
        let r = inverse_transform_4x4(&q, qp, w4, bd).expect("inverse 4x4");
        for j in 0..4 {
            for i in 0..4 {
                recon[(by * 4 + j) * 16 + bx * 4 + i] = r[j * 4 + i];
            }
        }
    }
    let mut cbp = 0u8;
    for blk8 in 0..4usize {
        if (0..4).any(|s| blk_has_nz[blk8 * 4 + s]) {
            cbp |= 1 << blk8;
        }
    }
    InterChain {
        scan,
        blk_has_nz,
        cbp,
        recon,
    }
}

/// §8.5.11 chroma chain for one 4:2:0 (8x8) or 4:2:2 (8x16) plane
/// tile at `qp_c` = QP′C.
struct ChromaChain {
    /// DC levels: 4 raster entries (4:2:0) or 8 in ChromaDCLevel scan
    /// order (4:2:2).
    dc: [i32; 8],
    /// AC-only scan lists per 4x4 block (4 or 8 blocks).
    ac: [[i32; 16]; 8],
    dc_nz: bool,
    ac_nz: bool,
    /// Reconstructed residual with AC (row-major `8 × ct_h`).
    recon_full: Vec<i32>,
    /// Reconstructed residual with DC only.
    recon_dc: Vec<i32>,
    ac_blk_nz: [bool; 8],
}

fn chroma_chain(
    residual: &[i32],
    ct_h: usize,
    qp_c: i32,
    intra: bool,
    w4: &[i32; 16],
    bd: u32,
) -> ChromaChain {
    let n_blk = ct_h / 4 * 2;
    let blocks_xy: Vec<(usize, usize)> = (0..n_blk).map(|k| ((k % 2) * 4, (k / 2) * 4)).collect();
    let mut coeffs = vec![[0i32; 16]; n_blk];
    for (k, &(bx, by)) in blocks_xy.iter().enumerate() {
        let mut block = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                block[j * 4 + i] = residual[(by + j) * 8 + bx + i];
            }
        }
        coeffs[k] = forward_core_4x4(&block);
    }
    let mut dc = [0i32; 8];
    let inv_dc: Vec<i32> = if n_blk == 4 {
        let dc_in = [coeffs[0][0], coeffs[1][0], coeffs[2][0], coeffs[3][0]];
        let q = quantize_chroma_dc_w(&forward_hadamard_2x2(&dc_in), qp_c, intra, w4[0]);
        dc[..4].copy_from_slice(&q);
        inverse_hadamard_chroma_dc_420(&q, qp_c, w4, bd)
            .expect("chroma DC inverse")
            .to_vec()
    } else {
        let dc_in: [i32; 8] = std::array::from_fn(|k| coeffs[k][0]);
        let q = quantize_chroma_dc_422_w(&forward_hadamard_4x2(&dc_in), qp_c, intra, w4[0]);
        let levels = scan_chroma_dc_422(&q);
        dc = levels;
        inverse_hadamard_chroma_dc_422(&levels, qp_c, w4, bd)
            .expect("chroma DC 422 inverse")
            .to_vec()
    };
    let mut ac = [[0i32; 16]; 8];
    let mut ac_blk_nz = [false; 8];
    let mut ac_quant = vec![[0i32; 16]; n_blk];
    let mut ac_nz = false;
    for k in 0..n_blk {
        let z = quantize_4x4_ac_w(&coeffs[k], qp_c, intra, w4);
        ac_quant[k] = z;
        let s = zigzag_scan_4x4_ac(&z);
        ac_blk_nz[k] = s[..15].iter().any(|&v| v != 0);
        ac_nz |= ac_blk_nz[k];
        ac[k] = s;
    }
    let dc_nz = dc[..n_blk].iter().any(|&v| v != 0);
    let mut recon_full = vec![0i32; 8 * ct_h];
    let mut recon_dc = vec![0i32; 8 * ct_h];
    for (k, &(bx, by)) in blocks_xy.iter().enumerate() {
        for (out, with_ac) in [(&mut recon_full, true), (&mut recon_dc, false)] {
            let mut c = if with_ac { ac_quant[k] } else { [0i32; 16] };
            c[0] = inv_dc[k];
            let r = inverse_transform_4x4_dc_preserved(&c, qp_c, w4, bd).expect("inverse 4x4");
            for j in 0..4 {
                for i in 0..4 {
                    out[(by + j) * 8 + bx + i] = r[j * 4 + i];
                }
            }
        }
    }
    if !ac_nz {
        ac_blk_nz = [false; 8];
    }
    ChromaChain {
        dc,
        ac,
        dc_nz,
        ac_nz,
        recon_full,
        recon_dc,
        ac_blk_nz,
    }
}

/// §8.6.4 / §8.5.13 8x8-transform inter chain at QP′ (or the §8.5.13
/// bypass identity): four Table 8-14 zig-zag scan lists, per-quadrant
/// CBP and the reconstructed residual.
struct Inter8x8Chain {
    scan: [[i32; 64]; 4],
    cbp: u8,
    recon: [i32; 256],
}

fn inter_chain_8x8(
    residual: &[i32; 256],
    qp: i32,
    w8: &[i32; 64],
    bd: u32,
    bypass: bool,
) -> Inter8x8Chain {
    let mut scan = [[0i32; 64]; 4];
    let mut cbp = 0u8;
    let mut recon = [0i32; 256];
    for (blk8, scan8) in scan.iter_mut().enumerate() {
        let (ox, oy) = ((blk8 % 2) * 8, (blk8 / 2) * 8);
        let mut block = [0i32; 64];
        for j in 0..8 {
            for i in 0..8 {
                block[j * 8 + i] = residual[(oy + j) * 16 + ox + i];
            }
        }
        let (q, r) = if bypass {
            (block, block)
        } else {
            let q = quantize_8x8_w(&forward_core_8x8(&block), qp, false, w8);
            let r = inverse_transform_8x8(&q, qp, w8, bd).expect("inverse 8x8");
            (q, r)
        };
        *scan8 = zigzag_scan_8x8(&q);
        if scan8.iter().any(|&v| v != 0) {
            cbp |= 1 << blk8;
            for j in 0..8 {
                for i in 0..8 {
                    recon[(oy + j) * 16 + ox + i] = r[j * 8 + i];
                }
            }
        }
    }
    Inter8x8Chain { scan, cbp, recon }
}

/// §7.4.5.3.3 — the four-4x4 CAVLC split of an 8x8 scan list:
/// `lumaLevel4x4[i4][k] = lumaLevel8x8[4 * k + i4]`.
fn split_8x8(scan8: &[i32; 64]) -> [[i32; 16]; 4] {
    let mut out = [[0i32; 16]; 4];
    for (i4, list) in out.iter_mut().enumerate() {
        for (k, v) in list.iter_mut().enumerate() {
            *v = scan8[4 * k + i4];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// §8.5.15 lossless transform-bypass chains (round-456).
// ---------------------------------------------------------------------------

/// Inverse of the decoder's §8.5.15 DPCM ([`intra_bypass_dpcm`]):
/// turn the residual `r` (row-major `n_w × n_h`) into the coded
/// differences `f` so that the decoder's running sums (eq. 8-412 /
/// 8-413) rebuild `r` exactly. `hor` = `horPredFlag`.
fn dpcm_forward(r: &[i32], n_w: usize, n_h: usize, hor: bool) -> Vec<i32> {
    let mut f = r.to_vec();
    if hor {
        for i in 0..n_h {
            for j in (1..n_w).rev() {
                f[i * n_w + j] -= r[i * n_w + j - 1];
            }
        }
    } else {
        for i in (1..n_h).rev() {
            for j in 0..n_w {
                f[i * n_w + j] -= r[(i - 1) * n_w + j];
            }
        }
    }
    f
}

/// §8.5.2 / §8.5.4 decoder mirror under bypass: the residual the
/// decoder derives from the coded arrays `f` (row-major), with the
/// DPCM re-applied for V/H prediction.
fn dpcm_recon(f: &[i32], n_w: usize, n_h: usize, hor: Option<bool>) -> Vec<i32> {
    let mut r = f.to_vec();
    if let Some(h) = hor {
        intra_bypass_dpcm(&mut r, n_w, n_h, h);
    }
    r
}

/// Table 8-4 Intra_16x16 mode → `Some(horPredFlag)` for the DPCM
/// modes (0 vertical, 1 horizontal), `None` otherwise.
fn i16_dpcm(mode: u8) -> Option<bool> {
    match mode {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    }
}

/// Table 8-5 intra chroma mode → DPCM flag (1 horizontal, 2 vertical).
fn chroma_dpcm(mode: u8) -> Option<bool> {
    match mode {
        1 => Some(true),
        2 => Some(false),
        _ => None,
    }
}

/// Bypass twin of [`i16_chain`]: coefficients are the (DPCM-coded)
/// residual itself — eq. 8-319 `dcY = c`, eq. 8-334 `r = c`.
/// `mode` is the Intra_16x16 / chroma-as-luma prediction mode (DPCM
/// for V/H).
fn i16_chain_bypass(residual: &[i32; 256], mode: Option<u8>) -> I16Chain {
    let hor = mode.and_then(i16_dpcm);
    let f_dpcm = dpcm_forward(residual, 16, 16, hor.unwrap_or(false));
    let f: &[i32] = if hor.is_some() { &f_dpcm } else { residual };
    let mut dc_raster = [0i32; 16];
    let mut ac_scan = [[0i32; 16]; 16];
    let mut blk_has_nz = [false; 16];
    let mut any_ac = false;
    let mut dc_only = vec![0i32; 256];
    for (blkz, &(bx, by)) in LUMA_4X4_BLK.iter().enumerate() {
        let mut block = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                block[j * 4 + i] = f[(by * 4 + j) * 16 + bx * 4 + i];
            }
        }
        dc_raster[by * 4 + bx] = block[0];
        dc_only[(by * 4) * 16 + bx * 4] = block[0];
        let s = zigzag_scan_4x4_ac(&block);
        blk_has_nz[blkz] = s[..15].iter().any(|&v| v != 0);
        any_ac |= blk_has_nz[blkz];
        ac_scan[blkz] = s;
    }
    let recon_src: &[i32] = if any_ac { f } else { &dc_only };
    let recon_v = dpcm_recon(recon_src, 16, 16, hor);
    let mut recon = [0i32; 256];
    recon.copy_from_slice(&recon_v);
    if !any_ac {
        blk_has_nz = [false; 16];
    }
    I16Chain {
        dc_raster,
        ac_scan,
        any_ac,
        recon,
        blk_has_nz,
    }
}

/// Bypass twin of [`inter_chain`] (no DPCM for inter prediction).
fn inter_chain_bypass(residual: &[i32; 256]) -> InterChain {
    let mut scan = [[0i32; 16]; 16];
    let mut blk_has_nz = [false; 16];
    for (blkz, &(bx, by)) in LUMA_4X4_BLK.iter().enumerate() {
        let mut block = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                block[j * 4 + i] = residual[(by * 4 + j) * 16 + bx * 4 + i];
            }
        }
        let s = zigzag_scan_4x4(&block);
        blk_has_nz[blkz] = s.iter().any(|&v| v != 0);
        scan[blkz] = s;
    }
    let mut cbp = 0u8;
    for blk8 in 0..4usize {
        if (0..4).any(|s| blk_has_nz[blk8 * 4 + s]) {
            cbp |= 1 << blk8;
        }
    }
    InterChain {
        scan,
        blk_has_nz,
        cbp,
        recon: *residual,
    }
}

/// Bypass twin of [`chroma_chain`] (4:2:0 / 4:2:2): eq. 8-323
/// `dcC = c` with the 4:2:2 eq. 8-305 pickup mirrored by
/// [`scan_chroma_dc_422`]; `mode` = intra chroma mode for DPCM.
fn chroma_chain_bypass(residual: &[i32], ct_h: usize, mode: Option<u8>) -> ChromaChain {
    let n_blk = ct_h / 4 * 2;
    let hor = mode.and_then(chroma_dpcm);
    let f_dpcm = dpcm_forward(residual, 8, ct_h, hor.unwrap_or(false));
    let f: &[i32] = if hor.is_some() { &f_dpcm } else { residual };
    let mut dc_blocks = [0i32; 8];
    let mut ac = [[0i32; 16]; 8];
    let mut ac_blk_nz = [false; 8];
    let mut ac_nz = false;
    let mut dc_only = vec![0i32; 8 * ct_h];
    for k in 0..n_blk {
        let (bx, by) = ((k % 2) * 4, (k / 2) * 4);
        let mut block = [0i32; 16];
        for j in 0..4 {
            for i in 0..4 {
                block[j * 4 + i] = f[(by + j) * 8 + bx + i];
            }
        }
        dc_blocks[k] = block[0];
        dc_only[by * 8 + bx] = block[0];
        let s = zigzag_scan_4x4_ac(&block);
        ac_blk_nz[k] = s[..15].iter().any(|&v| v != 0);
        ac_nz |= ac_blk_nz[k];
        ac[k] = s;
    }
    let dc = if n_blk == 4 {
        dc_blocks
    } else {
        scan_chroma_dc_422(&dc_blocks)
    };
    let dc_nz = dc[..n_blk].iter().any(|&v| v != 0);
    let recon_full = dpcm_recon(f, 8, ct_h, hor);
    let recon_dc = dpcm_recon(&dc_only, 8, ct_h, hor);
    if !ac_nz {
        ac_blk_nz = [false; 8];
    }
    ChromaChain {
        dc,
        ac,
        dc_nz,
        ac_nz,
        recon_full,
        recon_dc,
        ac_blk_nz,
    }
}

// ---------------------------------------------------------------------------
// Motion estimation / compensation.
// ---------------------------------------------------------------------------

fn mc_luma(
    reference: &[i32],
    w: usize,
    h: usize,
    mb_x: usize,
    mb_y: usize,
    mv: Mv,
    bd: u32,
) -> [i32; 256] {
    let mut dst = [0i32; 256];
    interpolate_luma(
        reference,
        w,
        w,
        h,
        (mb_x * 16) as i32 + (mv.x >> 2),
        (mb_y * 16) as i32 + (mv.y >> 2),
        (mv.x & 3) as u8,
        (mv.y & 3) as u8,
        16,
        16,
        bd,
        &mut dst,
        16,
    )
    .expect("luma interpolation");
    dst
}

/// §8.4.1.4 / §8.4.2.2.2 chroma predictor per chroma format.
fn mc_chroma(reference: &[i32], g: Geom, mb_x: usize, mb_y: usize, mv: Mv) -> Vec<i32> {
    let mut dst = vec![0i32; g.ct_w * g.ct_h];
    match g.cf {
        3 => {
            interpolate_luma(
                reference,
                g.cw,
                g.cw,
                g.ch,
                (mb_x * 16) as i32 + (mv.x >> 2),
                (mb_y * 16) as i32 + (mv.y >> 2),
                (mv.x & 3) as u8,
                (mv.y & 3) as u8,
                16,
                16,
                g.bd_c,
                &mut dst,
                16,
            )
            .expect("4:4:4 chroma interpolation");
        }
        _ => {
            // 4:2:0: mvC = mvL in 1/8 units; 4:2:2: vertical full
            // height → mvCy = mvL.y · 2 (eq. 8-229/8-230 & Table 8-9).
            let mv_cx = mv.x;
            let mv_cy = if g.cf == 2 { mv.y * 2 } else { mv.y };
            interpolate_chroma(
                reference,
                g.cw,
                g.cw,
                g.ch,
                (mb_x * g.ct_w) as i32 + (mv_cx >> 3),
                (mb_y * g.ct_h) as i32 + (mv_cy >> 3),
                (mv_cx & 7) as u8,
                (mv_cy & 7) as u8,
                g.ct_w as u32,
                g.ct_h as u32,
                g.bd_c,
                &mut dst,
                g.ct_w,
            )
            .expect("chroma interpolation");
        }
    }
    dst
}

fn sad_16x16(src: &[i32; 256], pred: &[i32; 256]) -> u64 {
    src.iter()
        .zip(pred.iter())
        .map(|(&s, &p)| (s - p).unsigned_abs() as u64)
        .sum()
}

/// Integer full search (±8) with §8.4.2.2 edge clamping, then half-
/// and quarter-pel refinement through the decoder's interpolation.
fn motion_search(
    src: &[i32; 256],
    reference: &[i32],
    w: usize,
    h: usize,
    mb_x: usize,
    mb_y: usize,
    bd: u32,
) -> (Mv, u64) {
    let x0 = (mb_x * 16) as i32;
    let y0 = (mb_y * 16) as i32;
    let sample = |x: i32, y: i32| -> i32 {
        let xc = x.clamp(0, w as i32 - 1) as usize;
        let yc = y.clamp(0, h as i32 - 1) as usize;
        reference[yc * w + xc]
    };
    let mut best = (Mv::ZERO, u64::MAX);
    for dy in -8..=8i32 {
        for dx in -8..=8i32 {
            let mut sad = 0u64;
            for j in 0..16 {
                for i in 0..16 {
                    let p = sample(x0 + i + dx, y0 + j + dy);
                    sad += (src[(j * 16 + i) as usize] - p).unsigned_abs() as u64;
                }
            }
            if sad < best.1 {
                best = (Mv::new(dx * 4, dy * 4), sad);
            }
        }
    }
    for step in [2i32, 1] {
        let centre = best.0;
        for dy in [-step, 0, step] {
            for dx in [-step, 0, step] {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let mv = Mv::new(centre.x + dx, centre.y + dy);
                let pred = mc_luma(reference, w, h, mb_x, mb_y, mv, bd);
                let sad = sad_16x16(src, &pred);
                if sad < best.1 {
                    best = (mv, sad);
                }
            }
        }
    }
    best
}

// ---------------------------------------------------------------------------
// nC bookkeeping.
// ---------------------------------------------------------------------------

/// §9.2.1.1 per-block luma nC with the in-MB progressive TotalCoeff
/// commit (the grid slot must already carry the MB's class flags).
/// `lists` are the emitted scan lists; `coded(blk)` gates which blocks
/// carry a residual; `ac_only` counts over the 15-entry AC layout.
fn luma_nc_progressive(
    nc_grid: &mut CavlcNcGrid,
    mb_addr: u32,
    is_intra: bool,
    lists: &[[i32; 16]; 16],
    coded: impl Fn(usize) -> bool,
    ac_only: bool,
) -> [i32; 16] {
    let mut nc = [0i32; 16];
    let mut own = [0u8; 16];
    for blk in 0..16usize {
        nc_grid.mbs[mb_addr as usize].luma_total_coeff = own;
        nc[blk] = derive_nc_luma(nc_grid, mb_addr, blk as u8, LumaNcKind::Ac, is_intra, false);
        if coded(blk) {
            let n = if ac_only { 15 } else { 16 };
            own[blk] = lists[blk][..n].iter().filter(|&&v| v != 0).count() as u8;
        }
    }
    nc_grid.mbs[mb_addr as usize].luma_total_coeff = own;
    nc
}

fn mark_slot(nc_grid: &mut CavlcNcGrid, mb_addr: u32, is_intra: bool, is_skip: bool) {
    let cur = &mut nc_grid.mbs[mb_addr as usize];
    cur.is_available = true;
    cur.is_intra = is_intra;
    cur.is_skip = is_skip;
    cur.is_i_pcm = false;
    cur.luma_total_coeff = [0; 16];
    cur.cb_total_coeff = [0; 8];
    cur.cr_total_coeff = [0; 8];
    cur.cb_luma_total_coeff = [0; 16];
    cur.cr_luma_total_coeff = [0; 16];
}

// ---------------------------------------------------------------------------
// Per-picture state.
// ---------------------------------------------------------------------------

struct Pic {
    y: Vec<i32>,
    u: Vec<i32>,
    v: Vec<i32>,
}

impl Pic {
    fn from_u16(g: Geom, y: &[u16], u: &[u16], v: &[u16]) -> Self {
        assert_eq!(y.len(), g.w * g.h, "luma plane size");
        assert_eq!(u.len(), g.cw * g.ch, "Cb plane size");
        assert_eq!(v.len(), g.cw * g.ch, "Cr plane size");
        Self {
            y: y.iter().map(|&s| s as i32).collect(),
            u: u.iter().map(|&s| s as i32).collect(),
            v: v.iter().map(|&s| s as i32).collect(),
        }
    }
    fn zeroed(g: Geom) -> Self {
        Self {
            y: vec![0; g.w * g.h],
            u: vec![0; g.cw * g.ch],
            v: vec![0; g.cw * g.ch],
        }
    }
    fn to_u16(&self) -> DeepPlanes {
        (
            self.y.iter().map(|&s| s as u16).collect(),
            self.u.iter().map(|&s| s as u16).collect(),
            self.v.iter().map(|&s| s as u16).collect(),
        )
    }
    fn luma_tile(&self, g: Geom, mb_x: usize, mb_y: usize) -> [i32; 256] {
        let mut t = [0i32; 256];
        for j in 0..16 {
            t[j * 16..j * 16 + 16]
                .copy_from_slice(&self.y[(mb_y * 16 + j) * g.w + mb_x * 16..][..16]);
        }
        t
    }
    fn chroma_tile(&self, g: Geom, cr: bool, mb_x: usize, mb_y: usize) -> Vec<i32> {
        let plane = if cr { &self.v } else { &self.u };
        let mut t = vec![0i32; g.ct_w * g.ct_h];
        for j in 0..g.ct_h {
            t[j * g.ct_w..(j + 1) * g.ct_w]
                .copy_from_slice(&plane[(mb_y * g.ct_h + j) * g.cw + mb_x * g.ct_w..][..g.ct_w]);
        }
        t
    }
    fn store_luma(
        &mut self,
        g: Geom,
        mb_x: usize,
        mb_y: usize,
        pred: &[i32; 256],
        res: &[i32; 256],
    ) {
        for j in 0..16 {
            for i in 0..16 {
                self.y[(mb_y * 16 + j) * g.w + mb_x * 16 + i] =
                    clip(pred[j * 16 + i] + res[j * 16 + i], g.bd_y);
            }
        }
    }
    fn store_chroma(
        &mut self,
        g: Geom,
        cr: bool,
        mb_x: usize,
        mb_y: usize,
        pred: &[i32],
        res: &[i32],
    ) {
        let plane = if cr { &mut self.v } else { &mut self.u };
        for j in 0..g.ct_h {
            for i in 0..g.ct_w {
                plane[(mb_y * g.ct_h + j) * g.cw + mb_x * g.ct_w + i] =
                    clip(pred[j * g.ct_w + i] + res[j * g.ct_w + i], g.bd_c);
            }
        }
    }
}

/// Intra_16x16 candidate for one MB (all planes).
struct IntraCand {
    luma_mode: u8,
    chroma_mode: u8,
    luma_pred: [i32; 256],
    luma: I16Chain,
    cbp_luma: u8,
    cbp_chroma: u8,
    pred_u: Vec<i32>,
    pred_v: Vec<i32>,
    /// 4:2:0 / 4:2:2 chroma chains.
    chroma: Option<(ChromaChain, ChromaChain)>,
    /// 4:4:4 luma-like chroma chains.
    planes444: Option<(I16Chain, I16Chain)>,
    sad: u64,
}

#[allow(clippy::too_many_arguments)]
fn intra_candidate(
    g: Geom,
    src: &Pic,
    recon: &Pic,
    mb_x: usize,
    mb_y: usize,
    qp_y: i32,
    qp_c: i32,
    w4: &[i32; 16],
    bypass: bool,
    allow_dpcm: bool,
) -> IntraCand {
    let src_y = src.luma_tile(g, mb_x, mb_y);
    let s16 = luma_samples16(&recon.y, g.w, mb_x, mb_y);
    let (luma_mode, luma_pred, sad) = pick_16x16(&src_y, &s16, g.bd_y, allow_dpcm);
    let mut residual = [0i32; 256];
    for k in 0..256 {
        residual[k] = src_y[k] - luma_pred[k];
    }
    let luma = if bypass {
        i16_chain_bypass(&residual, Some(luma_mode))
    } else {
        i16_chain(&residual, qp_y, w4, g.bd_y)
    };
    let mut cbp_luma: u8 = if luma.any_ac { 15 } else { 0 };
    let src_u = src.chroma_tile(g, false, mb_x, mb_y);
    let src_v = src.chroma_tile(g, true, mb_x, mb_y);
    let (chroma_mode, pred_u, pred_v, chroma, planes444, cbp_chroma) = match g.chroma_array_type() {
        Some(cat) => {
            let su = chroma_samples(&recon.u, g.cw, mb_x, mb_y, g.ct_w, g.ct_h);
            let sv = chroma_samples(&recon.v, g.cw, mb_x, mb_y, g.ct_w, g.ct_h);
            let (mode, pu, pv) = pick_chroma(&src_u, &src_v, &su, &sv, cat, g.bd_c, allow_dpcm);
            let ru: Vec<i32> = src_u.iter().zip(pu.iter()).map(|(s, p)| s - p).collect();
            let rv: Vec<i32> = src_v.iter().zip(pv.iter()).map(|(s, p)| s - p).collect();
            let (cu, cv) = if bypass {
                (
                    chroma_chain_bypass(&ru, g.ct_h, Some(mode)),
                    chroma_chain_bypass(&rv, g.ct_h, Some(mode)),
                )
            } else {
                (
                    chroma_chain(&ru, g.ct_h, qp_c, true, w4, g.bd_c),
                    chroma_chain(&rv, g.ct_h, qp_c, true, w4, g.bd_c),
                )
            };
            let cbp_chroma = if cu.ac_nz || cv.ac_nz {
                2
            } else if cu.dc_nz || cv.dc_nz {
                1
            } else {
                0
            };
            (mode, pu, pv, Some((cu, cv)), None, cbp_chroma)
        }
        None => {
            // §8.3.4.5 — 4:4:4: Cb / Cr predicted with the luma mode on
            // their own neighbours and coded like luma.
            let mode = Intra16x16Mode::from_index(luma_mode).expect("mode");
            let mut pu = [0i32; 256];
            let mut pv = [0i32; 256];
            predict_16x16(
                mode,
                &luma_samples16(&recon.u, g.cw, mb_x, mb_y),
                g.bd_c,
                &mut pu,
            );
            predict_16x16(
                mode,
                &luma_samples16(&recon.v, g.cw, mb_x, mb_y),
                g.bd_c,
                &mut pv,
            );
            let mut ru = [0i32; 256];
            let mut rv = [0i32; 256];
            for k in 0..256 {
                ru[k] = src_u[k] - pu[k];
                rv[k] = src_v[k] - pv[k];
            }
            let (cu, cv) = if bypass {
                (
                    i16_chain_bypass(&ru, Some(luma_mode)),
                    i16_chain_bypass(&rv, Some(luma_mode)),
                )
            } else {
                (
                    i16_chain(&ru, qp_c, w4, g.bd_c),
                    i16_chain(&rv, qp_c, w4, g.bd_c),
                )
            };
            // §7.4.5 Table 7-11 at ChromaArrayType 3 — one cbp_luma
            // value gates the AC of all three planes.
            if cu.any_ac || cv.any_ac {
                cbp_luma = 15;
            }
            (0u8, pu.to_vec(), pv.to_vec(), None, Some((cu, cv)), 0u8)
        }
    };
    IntraCand {
        luma_mode,
        chroma_mode,
        luma_pred,
        luma,
        cbp_luma,
        cbp_chroma,
        pred_u,
        pred_v,
        chroma,
        planes444,
        sad,
    }
}

/// Re-run the luma-like reconstruction of an [`I16Chain`] for a
/// `cbp_luma` decided at MB level (the 4:4:4 three-plane merge can
/// raise it after the chain was built).
fn i16_recon_for_cbp(
    chain: &I16Chain,
    cbp_luma: u8,
    qp: i32,
    w4: &[i32; 16],
    bd: u32,
) -> [i32; 256] {
    if (cbp_luma == 15) == chain.any_ac {
        return chain.recon;
    }
    // cbp 15 with no AC of its own: DC-only recon plus zero AC ==
    // the chain's DC-only recon (chain.any_ac == false → recon is
    // DC-only already). cbp 0 with AC present cannot happen.
    debug_assert!(cbp_luma == 15 && !chain.any_ac);
    let _ = (qp, w4, bd);
    chain.recon
}

fn emit_intra(
    g: Geom,
    w: &mut BitWriter,
    nc_grid: &mut CavlcNcGrid,
    mb_addr: u32,
    cand: &IntraCand,
    in_p_slice: bool,
) {
    mark_slot(nc_grid, mb_addr, true, false);
    let nc_dc = derive_nc_luma(nc_grid, mb_addr, 0, LumaNcKind::Intra16x16Dc, true, false);
    let cbp_luma = cand.cbp_luma;
    let luma_ac_nc = luma_nc_progressive(
        nc_grid,
        mb_addr,
        true,
        &cand.luma.ac_scan,
        |_| cbp_luma == 15,
        true,
    );
    let nc_ctx = CoeffTokenContext::Numeric(nc_dc);
    let emit = |w: &mut BitWriter, chroma: ChromaWriteKind<'_>| {
        if in_p_slice {
            write_intra16x16_mb_in_inter_slice_chroma(
                w,
                5,
                cand.luma_mode,
                cand.chroma_mode,
                cbp_luma,
                cand.cbp_chroma,
                0,
                &cand.luma.dc_raster,
                &cand.luma.ac_scan,
                &luma_ac_nc,
                chroma,
                nc_ctx,
            )
        } else {
            write_intra16x16_mb_chroma(
                w,
                cand.luma_mode,
                cand.chroma_mode,
                cbp_luma,
                cand.cbp_chroma,
                0,
                &cand.luma.dc_raster,
                &cand.luma.ac_scan,
                &luma_ac_nc,
                chroma,
                nc_ctx,
            )
        }
    };
    match (&cand.chroma, &cand.planes444) {
        (Some((cu, cv)), _) => {
            let n_blk = g.ct_h / 4 * 2;
            let (nc_cb, nc_cr) = derive_chroma_ac_nc_and_commit_totals(
                nc_grid,
                mb_addr,
                true,
                cand.cbp_chroma,
                &cu.ac[..n_blk],
                &cv.ac[..n_blk],
                g.cf,
            );
            let kind = if g.cf == 1 {
                ChromaWriteKind::Yuv420 {
                    chroma_dc_cb: cu.dc[..4].try_into().expect("4 DC"),
                    chroma_dc_cr: cv.dc[..4].try_into().expect("4 DC"),
                    chroma_ac_cb: cu.ac[..4].try_into().expect("4 AC"),
                    chroma_ac_cr: cv.ac[..4].try_into().expect("4 AC"),
                    cb_ac_nc: &nc_cb,
                    cr_ac_nc: &nc_cr,
                }
            } else {
                ChromaWriteKind::Yuv422 {
                    chroma_dc_cb: &cu.dc,
                    chroma_dc_cr: &cv.dc,
                    chroma_ac_cb: &cu.ac,
                    chroma_ac_cr: &cv.ac,
                    cb_ac_nc: &nc_cb,
                    cr_ac_nc: &nc_cr,
                }
            };
            emit(w, kind).expect("CAVLC intra MB");
        }
        (None, Some((cu, cv))) => {
            let (cb_dc_nc, cb_ac_nc) =
                derive_plane_nc_444_i16x16(nc_grid, mb_addr, false, cbp_luma, &cu.ac_scan);
            let (cr_dc_nc, cr_ac_nc) =
                derive_plane_nc_444_i16x16(nc_grid, mb_addr, true, cbp_luma, &cv.ac_scan);
            let kind = ChromaWriteKind::Yuv444 {
                cb_dc_raster: &cu.dc_raster,
                cr_dc_raster: &cv.dc_raster,
                cb_ac_levels: &cu.ac_scan,
                cr_ac_levels: &cv.ac_scan,
                cb_ac_nc: &cb_ac_nc,
                cr_ac_nc: &cr_ac_nc,
                cb_dc_nc,
                cr_dc_nc,
                cbp_luma_for_ac_gate: cbp_luma,
            };
            emit(w, kind).expect("CAVLC intra MB");
        }
        _ => unreachable!(),
    }
}

/// Reconstruct an intra candidate into `recon`; returns the deblock
/// facts.
#[allow(clippy::too_many_arguments)]
fn recon_intra(
    g: Geom,
    recon: &mut Pic,
    mb_x: usize,
    mb_y: usize,
    cand: &IntraCand,
    qp_y: i32,
    qp_c: i32,
    w4: &[i32; 16],
) -> MbDeblockInfo {
    let luma_res = i16_recon_for_cbp(&cand.luma, cand.cbp_luma, qp_y, w4, g.bd_y);
    recon.store_luma(g, mb_x, mb_y, &cand.luma_pred, &luma_res);
    match (&cand.chroma, &cand.planes444) {
        (Some((cu, cv)), _) => {
            let pick = |c: &ChromaChain| -> Vec<i32> {
                match cand.cbp_chroma {
                    2 => c.recon_full.clone(),
                    1 => c.recon_dc.clone(),
                    _ => vec![0; c.recon_full.len()],
                }
            };
            recon.store_chroma(g, false, mb_x, mb_y, &cand.pred_u, &pick(cu));
            recon.store_chroma(g, true, mb_x, mb_y, &cand.pred_v, &pick(cv));
        }
        (None, Some((cu, cv))) => {
            let ru = i16_recon_for_cbp(cu, cand.cbp_luma, qp_c, w4, g.bd_c);
            let rv = i16_recon_for_cbp(cv, cand.cbp_luma, qp_c, w4, g.bd_c);
            recon.store_chroma(g, false, mb_x, mb_y, &cand.pred_u, &ru);
            recon.store_chroma(g, true, mb_x, mb_y, &cand.pred_v, &rv);
        }
        _ => unreachable!(),
    }
    MbDeblockInfo {
        is_intra: true,
        qp_y,
        luma_nonzero_4x4: if cand.cbp_luma == 15 {
            luma_nz_mask_from_blocks(&cand.luma.blk_has_nz)
        } else {
            0
        },
        chroma_nonzero_4x4: 0,
        ref_idx_l0: [-1; 4],
        ref_poc_l0: [i32::MIN; 4],
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// CABAC emission (round-456): the same macroblock decisions coded
// under `entropy_coding_mode_flag = 1` through the shared helpers of
// `encoder::mbaff_cabac` (progressive §6.4.9 neighbours).
// ---------------------------------------------------------------------------

/// Sixteen 4x4 scan lists in luma4x4BlkIdx order.
type ScanLists = [[i32; 16]; 16];

/// Intra_16x16 macroblock under CABAC (I or P slice).
fn emit_intra_cabac(
    g: Geom,
    cs: &mut CabacState,
    cab: &mut CabacNeighbourGrid,
    mb_addr: u32,
    cand: &IntraCand,
    in_p_slice: bool,
) {
    let addr = mb_addr as usize;
    let nctx = full_neighbour_ctx(cab, addr, false);
    let group = match (cand.cbp_luma, cand.cbp_chroma) {
        (0, 0) => 0u32,
        (0, 1) => 1,
        (0, 2) => 2,
        (15, 0) => 3,
        (15, 1) => 4,
        (15, 2) => 5,
        _ => unreachable!("Intra_16x16 cbp_luma is 0 or 15"),
    };
    let row = 1 + group * 4 + cand.luma_mode as u32;
    if in_p_slice {
        encode_mb_type_p(&mut cs.enc, &mut cs.ctxs, &nctx, 5 + row);
    } else {
        encode_mb_type_i(&mut cs.enc, &mut cs.ctxs, &nctx, row);
    }
    // §7.3.5.1 — intra_chroma_pred_mode is absent at ChromaArrayType 3.
    if g.cf != 3 {
        encode_intra_chroma_pred_mode(&mut cs.enc, &mut cs.ctxs, &nctx, cand.chroma_mode as u32);
    }
    // §7.3.5 — mb_qp_delta always present for Intra_16x16.
    encode_mb_qp_delta(&mut cs.enc, &mut cs.ctxs, cs.prev_qp_delta_nonzero, 0);
    cs.prev_qp_delta_nonzero = false;

    let mut cur = CabacMbNeighbourInfo {
        is_intra: true,
        coded_block_pattern_luma: cand.cbp_luma,
        coded_block_pattern_chroma: cand.cbp_chroma,
        intra_chroma_pred_mode: cand.chroma_mode,
        ..Default::default()
    };
    emit_i16_plane_cabac(
        cs,
        cab,
        &mut cur,
        addr,
        false,
        Plane::Y,
        &cand.luma.dc_raster,
        &cand.luma.ac_scan,
        cand.cbp_luma,
    );
    match (&cand.chroma, &cand.planes444) {
        (Some((cu, cv)), _) => {
            let n_blk = g.ct_h / 4 * 2;
            emit_chroma_cabac(
                cs,
                cab,
                &mut cur,
                addr,
                false,
                true,
                cand.cbp_chroma,
                &cu.dc,
                &cv.dc,
                &cu.ac[..n_blk],
                &cv.ac[..n_blk],
                g.cf,
            );
        }
        (None, Some((cu, cv))) => {
            emit_i16_plane_cabac(
                cs,
                cab,
                &mut cur,
                addr,
                false,
                Plane::Cb,
                &cu.dc_raster,
                &cu.ac_scan,
                cand.cbp_luma,
            );
            emit_i16_plane_cabac(
                cs,
                cab,
                &mut cur,
                addr,
                false,
                Plane::Cr,
                &cv.dc_raster,
                &cv.ac_scan,
                cand.cbp_luma,
            );
        }
        _ => unreachable!(),
    }
    cur.available = true;
    cab.mbs[addr] = cur;
}

/// P_L0_16x16 macroblock under CABAC (`mb_skip_flag = 0` already coded).
#[allow(clippy::too_many_arguments)]
fn emit_inter_cabac(
    g: Geom,
    cs: &mut CabacState,
    cab: &mut CabacNeighbourGrid,
    mb_addr: u32,
    mvd: Mv,
    cbp_luma: u8,
    cbp_chroma: u8,
    luma_scan: &[[i32; 16]; 16],
    chroma: Option<(&ChromaChain, &ChromaChain)>,
    planes444: Option<(&ScanLists, &ScanLists)>,
    // `Some((flag, scan8))` when the PPS carries `transform_8x8_mode_flag`:
    // the flag is coded after CBP (cbp_luma > 0) and, when set, the
    // luma residual is the four ctxBlockCat-5 lists.
    t8x8: Option<(bool, &[[i32; 64]; 4])>,
) {
    let addr = mb_addr as usize;
    let nctx = full_neighbour_ctx(cab, addr, false);
    encode_mb_type_p(&mut cs.enc, &mut cs.ctxs, &nctx, 0);
    let mut cur = CabacMbNeighbourInfo {
        ref_idx_l0: [0; 4],
        ..Default::default()
    };
    let sum_x = cabac_mvd_abs_sum(cab, mb_addr, &cur, 0, MvdComponent::X, 0);
    encode_mvd_lx(&mut cs.enc, &mut cs.ctxs, MvdComponent::X, sum_x, mvd.x);
    let sum_y = cabac_mvd_abs_sum(cab, mb_addr, &cur, 0, MvdComponent::Y, 0);
    encode_mvd_lx(&mut cs.enc, &mut cs.ctxs, MvdComponent::Y, sum_y, mvd.y);
    cur.mvd_l0_x = [mvd.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16; 16];
    cur.mvd_l0_y = [mvd.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16; 16];
    encode_coded_block_pattern(&mut cs.enc, &mut cs.ctxs, &nctx, g.cf, cbp_luma, cbp_chroma);
    let use_8x8 = match t8x8 {
        Some((flag, _)) if cbp_luma > 0 => {
            // §7.3.5 second gate — coded whenever the tool is on and
            // cbp_luma > 0 (P_L0_16x16 has no sub-8x8 partitions).
            encode_transform_size_8x8_flag(&mut cs.enc, &mut cs.ctxs, &nctx, flag);
            flag
        }
        _ => false,
    };
    if cbp_luma > 0 || cbp_chroma > 0 {
        encode_mb_qp_delta(&mut cs.enc, &mut cs.ctxs, cs.prev_qp_delta_nonzero, 0);
    }
    cs.prev_qp_delta_nonzero = false;
    cur.coded_block_pattern_luma = cbp_luma;
    cur.coded_block_pattern_chroma = cbp_chroma;
    cur.transform_size_8x8_flag = use_8x8;
    if use_8x8 {
        // §7.3.5.3.3 — one ctxBlockCat-5 residual per coded quadrant,
        // coded_block_flag inferred (ChromaArrayType 1 / 2) and folded
        // into the four 4x4 slots.
        let scan8 = t8x8.map(|(_, s)| s).expect("8x8 lists");
        for (blk8, list) in scan8.iter().enumerate() {
            if (cbp_luma >> blk8) & 1 == 0 {
                continue;
            }
            let coded = encode_residual_block_cabac_field(
                &mut cs.enc,
                &mut cs.ctxs,
                BlockType::Luma8x8,
                list,
                64,
                None,
                None,
                true,
                1,
                false,
            );
            for sub in 0..4usize {
                cur.cbf_luma_4x4[blk8 * 4 + sub] = coded;
            }
        }
    } else {
        emit_inter_plane_cabac(
            cs,
            cab,
            &mut cur,
            addr,
            false,
            Plane::Y,
            luma_scan,
            cbp_luma,
        );
    }
    if let Some((cu, cv)) = chroma {
        let n_blk = g.ct_h / 4 * 2;
        emit_chroma_cabac(
            cs,
            cab,
            &mut cur,
            addr,
            false,
            false,
            cbp_chroma,
            &cu.dc,
            &cv.dc,
            &cu.ac[..n_blk],
            &cv.ac[..n_blk],
            g.cf,
        );
    }
    if let Some((cb, cr)) = planes444 {
        emit_inter_plane_cabac(cs, cab, &mut cur, addr, false, Plane::Cb, cb, cbp_luma);
        emit_inter_plane_cabac(cs, cab, &mut cur, addr, false, Plane::Cr, cr, cbp_luma);
    }
    cur.available = true;
    cab.mbs[addr] = cur;
}

/// Encode a >8-bit sequence. `frames` holds `u16` planes at the
/// configured chroma format (top-to-bottom row-major).
pub fn encode_deep_sequence(cfg: &DeepConfig, frames: &[(&[u16], &[u16], &[u16])]) -> DeepEncoded {
    assert!(
        cfg.width % 16 == 0 && cfg.height % 16 == 0,
        "MB-aligned dims"
    );
    assert!(
        (8..=14).contains(&cfg.bit_depth_luma) && (8..=14).contains(&cfg.bit_depth_chroma),
        "bit depths 8..=14"
    );
    assert!(
        matches!(cfg.chroma_format_idc, 1..=3),
        "chroma_format_idc 1..=3"
    );
    assert!(!frames.is_empty());
    let (w, h) = (cfg.width as usize, cfg.height as usize);
    let cf = cfg.chroma_format_idc;
    let (ct_w, ct_h) = match cf {
        1 => (8, 8),
        2 => (8, 16),
        _ => (16, 16),
    };
    let g = Geom {
        w,
        h,
        cw: w * ct_w / 16,
        ch: h * ct_h / 16,
        ct_w,
        ct_h,
        w_mbs: w / 16,
        cf,
        bd_y: cfg.bit_depth_luma,
        bd_c: cfg.bit_depth_chroma,
    };
    let h_mbs = h / 16;
    let off_y = qp_bd_offset(cfg.bit_depth_luma - 8);
    let off_c = qp_bd_offset(cfg.bit_depth_chroma - 8);
    assert!(
        (-off_y..=51).contains(&cfg.qp),
        "QP_Y must lie in −QpBdOffsetY..=51"
    );
    let qp_y = cfg.qp;
    let qp_y_prime = qp_y + off_y;
    assert!(
        !cfg.lossless || qp_y_prime == 0,
        "lossless coding needs QP′Y == 0, i.e. qp == −QpBdOffsetY"
    );
    let bypass = cfg.lossless;
    // Under bypass the V / H intra modes invoke the §8.5.15 DPCM; the
    // interop flag withholds them so the stream never depends on it.
    let allow_dpcm = !(cfg.lossless && cfg.lossless_interop);
    let qp_c_prime = qp_y_to_qp_c_with_bd_offset(qp_y, 0, off_c) + off_c;
    let w4 = ScalingMatrixMode::Flat.intra_weights().w4;

    // §A.2 — profile by depth / chroma format.
    let max_bd = cfg.bit_depth_luma.max(cfg.bit_depth_chroma);
    let profile_idc: u8 = match (cf, max_bd) {
        (1, 8..=10) => 110,
        (2, 8..=10) => 122,
        _ => 244,
    };
    let frame_num_bits = 8u32;
    let poc_lsb_bits = 8u32;
    let w_mbs = g.w_mbs as u32;
    let sps = build_sps_rbsp_with_bypass(
        &BaselineSpsConfig {
            seq_parameter_set_id: 0,
            level_idc: min_level_idc_for_picture_size(w_mbs, h_mbs as u32),
            width_in_mbs: w_mbs,
            height_in_mbs: h_mbs as u32,
            log2_max_frame_num_minus4: 4,
            log2_max_poc_lsb_minus4: 4,
            max_num_ref_frames: 1,
            profile_idc,
            chroma_format_idc: cf,
            separate_colour_plane: false,
            seq_scaling_lists: None,
            bit_depth_luma_minus8: cfg.bit_depth_luma - 8,
            bit_depth_chroma_minus8: cfg.bit_depth_chroma - 8,
            interlaced_fields: false,
            mbaff: false,
            vui: None,
        },
        cfg.lossless,
    );
    let pps = build_baseline_pps_rbsp(&BaselinePpsConfig {
        pic_parameter_set_id: 0,
        seq_parameter_set_id: 0,
        pic_init_qp_minus26: qp_y - 26,
        chroma_qp_index_offset: 0,
        weighted_pred_flag: false,
        weighted_bipred_idc: 0,
        entropy_coding_mode_flag: cfg.cabac,
        transform_8x8_mode_flag: cfg.transform_8x8 && cf != 3,
        redundant_pic_cnt_present_flag: false,
        slice_groups: None,
        constrained_intra_pred_flag: false,
        pic_scaling_lists: None,
        chroma_format_idc: cf,
    });
    let mut stream: Vec<u8> = Vec::new();
    stream.extend_from_slice(&build_nal_unit(3, NalUnitType::Sps, &sps));
    stream.extend_from_slice(&build_nal_unit(3, NalUnitType::Pps, &pps));
    let disable_deblock: u32 = if cfg.deblock { 0 } else { 1 };

    let mut recon_frames: Vec<DeepPlanes> = Vec::with_capacity(frames.len());
    let mut prev: Option<Pic> = None;
    let mut prev_poc = 0i32;
    let mut skipped_mbs = 0usize;
    let mut intra_mbs_in_p = 0usize;
    let mut mbs_8x8 = 0usize;
    let w8 = ScalingMatrixMode::Flat.inter_weights().w8;
    const EMPTY_8X8: [[i32; 64]; 4] = [[0; 64]; 4];

    for (k, &(fy, fu, fv)) in frames.iter().enumerate() {
        let src = Pic::from_u16(g, fy, fu, fv);
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
                    disable_deblocking_filter_idc: disable_deblock,
                    slice_alpha_c0_offset_div2: 0,
                    slice_beta_offset_div2: 0,
                    nal_ref_idc: 2,
                    cabac: cfg.cabac.then_some(CabacSliceParams { cabac_init_idc: 0 }),
                    field: FieldPicSignal::FrameMbsOnly,
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
                    disable_deblocking_filter_idc: disable_deblock,
                    slice_alpha_c0_offset_div2: 0,
                    slice_beta_offset_div2: 0,
                    field: FieldPicSignal::FrameMbsOnly,
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
        if cfg.cabac {
            while !sw.byte_aligned() {
                sw.u(1, 1);
            }
        }
        let kind = if is_p { SliceKind::P } else { SliceKind::I };
        let mut cs = cfg
            .cabac
            .then(|| CabacState::new(kind, if is_p { Some(0) } else { None }, qp_y));
        let mut cab = CabacNeighbourGrid::new_mbaff(w_mbs, h_mbs as u32, false);
        let n_mbs = g.w_mbs * h_mbs;
        let mut recon = Pic::zeroed(g);
        let mut nc_grid = CavlcNcGrid::new(w_mbs, h_mbs as u32);
        let mut mv_grid = MvGrid::new(g.w_mbs, h_mbs);
        let mut dbl = vec![MbDeblockInfo::default(); g.w_mbs * h_mbs];
        let mut pending_skip = 0u32;

        for mb_y in 0..h_mbs {
            for mb_x in 0..g.w_mbs {
                let mb_addr = (mb_y * g.w_mbs + mb_x) as u32;
                if !is_p {
                    let cand = intra_candidate(
                        g, &src, &recon, mb_x, mb_y, qp_y_prime, qp_c_prime, &w4, bypass,
                        allow_dpcm,
                    );
                    if let Some(cs) = cs.as_mut() {
                        emit_intra_cabac(g, cs, &mut cab, mb_addr, &cand, false);
                        encode_end_of_slice_flag(&mut cs.enc, mb_addr as usize + 1 == n_mbs);
                    } else {
                        emit_intra(g, &mut sw, &mut nc_grid, mb_addr, &cand, false);
                    }
                    dbl[mb_addr as usize] = recon_intra(
                        g, &mut recon, mb_x, mb_y, &cand, qp_y_prime, qp_c_prime, &w4,
                    );
                    dbl[mb_addr as usize].qp_y = qp_y;
                    continue;
                }
                let reference = prev.as_ref().expect("P picture without reference");
                let src_y = src.luma_tile(g, mb_x, mb_y);
                let (mv, inter_sad) =
                    motion_search(&src_y, &reference.y, g.w, g.h, mb_x, mb_y, g.bd_y);
                let mvp = mvp_for_16x16(&mv_grid, mb_x, mb_y, 0);
                let (_, skip_mv) = p_skip_mv(&mv_grid, mb_x, mb_y);
                let pred_y = mc_luma(&reference.y, g.w, g.h, mb_x, mb_y, mv, g.bd_y);
                let pred_u = mc_chroma(&reference.u, g, mb_x, mb_y, mv);
                let pred_v = mc_chroma(&reference.v, g, mb_x, mb_y, mv);
                let mut residual = [0i32; 256];
                for i in 0..256 {
                    residual[i] = src_y[i] - pred_y[i];
                }
                let mut luma = if bypass {
                    inter_chain_bypass(&residual)
                } else {
                    inter_chain(&residual, qp_y_prime, &w4, g.bd_y)
                };
                // §8.6.4 8x8 transform on every other MB (coverage
                // policy: both flag values and both neighbour states
                // of the §9.3.3.1.1.10 context in one picture).
                let t8x8_on = cfg.transform_8x8 && cf != 3;
                let t8x8_chain = (t8x8_on && mb_addr % 2 == 0)
                    .then(|| inter_chain_8x8(&residual, qp_y_prime, &w8, g.bd_y, bypass));
                if let Some(c8) = &t8x8_chain {
                    luma.cbp = c8.cbp;
                    luma.recon = c8.recon;
                    for blk8 in 0..4usize {
                        let split = split_8x8(&c8.scan[blk8]);
                        for (sub, list) in split.iter().enumerate() {
                            luma.scan[blk8 * 4 + sub] = *list;
                            luma.blk_has_nz[blk8 * 4 + sub] = (c8.cbp >> blk8) & 1 == 1;
                        }
                    }
                }
                let mut cbp_luma = luma.cbp;
                let src_u = src.chroma_tile(g, false, mb_x, mb_y);
                let src_v = src.chroma_tile(g, true, mb_x, mb_y);
                let ru: Vec<i32> = src_u
                    .iter()
                    .zip(pred_u.iter())
                    .map(|(s, p)| s - p)
                    .collect();
                let rv: Vec<i32> = src_v
                    .iter()
                    .zip(pred_v.iter())
                    .map(|(s, p)| s - p)
                    .collect();
                let mut cbp_chroma = 0u8;
                let mut chroma: Option<(ChromaChain, ChromaChain)> = None;
                let mut planes444: Option<(InterChain, InterChain)> = None;
                if cf == 3 {
                    let ru: [i32; 256] = ru.try_into().expect("256");
                    let rv: [i32; 256] = rv.try_into().expect("256");
                    let (cu, cv) = if bypass {
                        (inter_chain_bypass(&ru), inter_chain_bypass(&rv))
                    } else {
                        (
                            inter_chain(&ru, qp_c_prime, &w4, g.bd_c),
                            inter_chain(&rv, qp_c_prime, &w4, g.bd_c),
                        )
                    };
                    cbp_luma |= cu.cbp | cv.cbp;
                    planes444 = Some((cu, cv));
                } else {
                    let (cu, cv) = if bypass {
                        (
                            chroma_chain_bypass(&ru, g.ct_h, None),
                            chroma_chain_bypass(&rv, g.ct_h, None),
                        )
                    } else {
                        (
                            chroma_chain(&ru, g.ct_h, qp_c_prime, false, &w4, g.bd_c),
                            chroma_chain(&rv, g.ct_h, qp_c_prime, false, &w4, g.bd_c),
                        )
                    };
                    cbp_chroma = if cu.ac_nz || cv.ac_nz {
                        2
                    } else if cu.dc_nz || cv.dc_nz {
                        1
                    } else {
                        0
                    };
                    chroma = Some((cu, cv));
                }
                let is_skip = mv == skip_mv && cbp_luma == 0 && cbp_chroma == 0;
                if is_skip {
                    skipped_mbs += 1;
                    if let Some(cs) = cs.as_mut() {
                        let skip_nctx = skip_neighbour_ctx(&cab, mb_addr as usize, false);
                        encode_mb_skip_flag(
                            &mut cs.enc,
                            &mut cs.ctxs,
                            SliceKind::P,
                            &skip_nctx,
                            true,
                        );
                        cs.prev_qp_delta_nonzero = false;
                        cab.mbs[mb_addr as usize] = CabacMbNeighbourInfo {
                            available: true,
                            is_skip: true,
                            ref_idx_l0: [0; 4],
                            ref_idx_l1: [0; 4],
                            ..Default::default()
                        };
                        encode_end_of_slice_flag(&mut cs.enc, mb_addr as usize + 1 == n_mbs);
                    } else {
                        pending_skip += 1;
                    }
                    mark_slot(&mut nc_grid, mb_addr, false, true);
                    *mv_grid.slot_mut(mb_x, mb_y) = MvGridSlot {
                        available: true,
                        is_intra: false,
                        ref_idx_l0_8x8: [0; 4],
                        mv_l0_8x8: [skip_mv; 4],
                    };
                    let zero = [0i32; 256];
                    recon.store_luma(g, mb_x, mb_y, &pred_y, &zero);
                    let zc = vec![0i32; g.ct_w * g.ct_h];
                    recon.store_chroma(g, false, mb_x, mb_y, &pred_u, &zc);
                    recon.store_chroma(g, true, mb_x, mb_y, &pred_v, &zc);
                    dbl[mb_addr as usize] = MbDeblockInfo {
                        is_intra: false,
                        qp_y,
                        mv_l0: [(skip_mv.x as i16, skip_mv.y as i16); 16],
                        ref_idx_l0: [0; 4],
                        ref_poc_l0: [prev_poc; 4],
                        ..Default::default()
                    };
                    continue;
                }
                // Intra alternative.
                let intra = if cfg.intra_in_p {
                    let cand = intra_candidate(
                        g, &src, &recon, mb_x, mb_y, qp_y_prime, qp_c_prime, &w4, bypass,
                        allow_dpcm,
                    );
                    (cand.sad + 64 < inter_sad).then_some(cand)
                } else {
                    None
                };
                if let Some(cs) = cs.as_mut() {
                    let skip_nctx = skip_neighbour_ctx(&cab, mb_addr as usize, false);
                    encode_mb_skip_flag(&mut cs.enc, &mut cs.ctxs, SliceKind::P, &skip_nctx, false);
                } else {
                    sw.ue(pending_skip);
                    pending_skip = 0;
                }
                if let Some(cand) = intra {
                    intra_mbs_in_p += 1;
                    if let Some(cs) = cs.as_mut() {
                        emit_intra_cabac(g, cs, &mut cab, mb_addr, &cand, true);
                        encode_end_of_slice_flag(&mut cs.enc, mb_addr as usize + 1 == n_mbs);
                    } else {
                        emit_intra(g, &mut sw, &mut nc_grid, mb_addr, &cand, true);
                    }
                    dbl[mb_addr as usize] = recon_intra(
                        g, &mut recon, mb_x, mb_y, &cand, qp_y_prime, qp_c_prime, &w4,
                    );
                    dbl[mb_addr as usize].qp_y = qp_y;
                    *mv_grid.slot_mut(mb_x, mb_y) = MvGridSlot {
                        available: true,
                        is_intra: true,
                        ref_idx_l0_8x8: [-1; 4],
                        mv_l0_8x8: [Mv::ZERO; 4],
                    };
                    continue;
                }
                // P_L0_16x16.
                mark_slot(&mut nc_grid, mb_addr, false, false);
                let luma_nc = luma_nc_progressive(
                    &mut nc_grid,
                    mb_addr,
                    false,
                    &luma.scan,
                    |blk| (cbp_luma >> (blk / 4)) & 1 == 1,
                    false,
                );
                let mvd = Mv::new(mv.x - mvp.x, mv.y - mvp.y);
                // §7.3.5 — the flag is coded (and the 8x8 residual used)
                // only with cbp_luma > 0; otherwise the decoder infers 0.
                let flag_8x8 = t8x8_chain.is_some() && cbp_luma > 0;
                if flag_8x8 {
                    mbs_8x8 += 1;
                }
                let mut mcfg = PL016x16McbConfig {
                    ref_idx_l0: 0,
                    transform_size_8x8_flag: t8x8_on.then_some(flag_8x8),
                    mvd_l0_x: mvd.x,
                    mvd_l0_y: mvd.y,
                    cbp_luma,
                    cbp_chroma,
                    mb_qp_delta: 0,
                    luma_4x4_levels: luma.scan,
                    luma_4x4_nc: luma_nc,
                    chroma_dc_cb: [0; 4],
                    chroma_dc_cr: [0; 4],
                    chroma_ac_cb: [[0; 16]; 4],
                    chroma_ac_cr: [[0; 16]; 4],
                    chroma_ac_nc_cb: [0; 8],
                    chroma_ac_nc_cr: [0; 8],
                };
                // Zero the uncoded quadrants' lists (they are not
                // emitted; keeps the deblock mask honest).
                let mut blk_has_nz = luma.blk_has_nz;
                for (blk, nz) in blk_has_nz.iter_mut().enumerate() {
                    if (cbp_luma >> (blk / 4)) & 1 == 0 {
                        mcfg.luma_4x4_levels[blk] = [0; 16];
                        *nz = false;
                    }
                }
                let mut chroma_nz_mask = 0u16;
                if let Some((cu, cv)) = &chroma {
                    let n_blk = g.ct_h / 4 * 2;
                    let (nc_cb, nc_cr) = derive_chroma_ac_nc_and_commit_totals(
                        &mut nc_grid,
                        mb_addr,
                        false,
                        cbp_chroma,
                        &cu.ac[..n_blk],
                        &cv.ac[..n_blk],
                        cf,
                    );
                    let kind = if cf == 1 {
                        ChromaWriteKind::Yuv420 {
                            chroma_dc_cb: cu.dc[..4].try_into().expect("4 DC"),
                            chroma_dc_cr: cv.dc[..4].try_into().expect("4 DC"),
                            chroma_ac_cb: cu.ac[..4].try_into().expect("4 AC"),
                            chroma_ac_cr: cv.ac[..4].try_into().expect("4 AC"),
                            cb_ac_nc: &nc_cb,
                            cr_ac_nc: &nc_cr,
                        }
                    } else {
                        ChromaWriteKind::Yuv422 {
                            chroma_dc_cb: &cu.dc,
                            chroma_dc_cr: &cv.dc,
                            chroma_ac_cb: &cu.ac,
                            chroma_ac_cr: &cv.ac,
                            cb_ac_nc: &nc_cb,
                            cr_ac_nc: &nc_cr,
                        }
                    };
                    if let Some(cs) = cs.as_mut() {
                        emit_inter_cabac(
                            g,
                            cs,
                            &mut cab,
                            mb_addr,
                            mvd,
                            cbp_luma,
                            cbp_chroma,
                            &mcfg.luma_4x4_levels,
                            Some((cu, cv)),
                            None,
                            t8x8_on.then(|| {
                                (
                                    flag_8x8,
                                    t8x8_chain.as_ref().map(|c| &c.scan).unwrap_or(&EMPTY_8X8),
                                )
                            }),
                        );
                    } else {
                        write_p_l0_16x16_mb_chroma(&mut sw, &mcfg, 0, cf, kind)
                            .expect("CAVLC P MB");
                    }
                    if cbp_chroma == 2 {
                        if cf == 1 {
                            let cb: [bool; 4] = std::array::from_fn(|i| cu.ac_blk_nz[i]);
                            let cr: [bool; 4] = std::array::from_fn(|i| cv.ac_blk_nz[i]);
                            chroma_nz_mask = chroma_nz_mask_from_blocks(&cb, &cr);
                        } else {
                            for blk in 0..8 {
                                if cu.ac_blk_nz[blk] {
                                    chroma_nz_mask |= 1 << blk;
                                }
                                if cv.ac_blk_nz[blk] {
                                    chroma_nz_mask |= 1 << (8 + blk);
                                }
                            }
                        }
                    }
                    let pick = |c: &ChromaChain| -> Vec<i32> {
                        match cbp_chroma {
                            2 => c.recon_full.clone(),
                            1 => c.recon_dc.clone(),
                            _ => vec![0; c.recon_full.len()],
                        }
                    };
                    recon.store_chroma(g, false, mb_x, mb_y, &pred_u, &pick(cu));
                    recon.store_chroma(g, true, mb_x, mb_y, &pred_v, &pick(cv));
                } else if let Some((cu, cv)) = &planes444 {
                    let mut cb_levels = cu.scan;
                    let mut cr_levels = cv.scan;
                    for blk in 0..16 {
                        if (cbp_luma >> (blk / 4)) & 1 == 0 {
                            cb_levels[blk] = [0; 16];
                            cr_levels[blk] = [0; 16];
                        }
                    }
                    let cb_nc = derive_plane_nc_444_inter(
                        &mut nc_grid,
                        mb_addr,
                        false,
                        cbp_luma,
                        &cb_levels,
                    );
                    let cr_nc = derive_plane_nc_444_inter(
                        &mut nc_grid,
                        mb_addr,
                        true,
                        cbp_luma,
                        &cr_levels,
                    );
                    let kind = ChromaWriteKind::Yuv444Inter {
                        cb_levels: &cb_levels,
                        cr_levels: &cr_levels,
                        cb_nc: &cb_nc,
                        cr_nc: &cr_nc,
                        cbp_luma,
                    };
                    if let Some(cs) = cs.as_mut() {
                        emit_inter_cabac(
                            g,
                            cs,
                            &mut cab,
                            mb_addr,
                            mvd,
                            cbp_luma,
                            0,
                            &mcfg.luma_4x4_levels,
                            None,
                            Some((&cb_levels, &cr_levels)),
                            None,
                        );
                    } else {
                        write_p_l0_16x16_mb_chroma(&mut sw, &mcfg, 0, cf, kind)
                            .expect("CAVLC P MB");
                    }
                    recon.store_chroma(g, false, mb_x, mb_y, &pred_u, &cu.recon);
                    recon.store_chroma(g, true, mb_x, mb_y, &pred_v, &cv.recon);
                }
                let mut luma_res = luma.recon;
                for (blk, &(bx, by)) in LUMA_4X4_BLK.iter().enumerate() {
                    if (cbp_luma >> (blk / 4)) & 1 == 0 {
                        for j in 0..4 {
                            for i in 0..4 {
                                luma_res[(by * 4 + j) * 16 + bx * 4 + i] = 0;
                            }
                        }
                    }
                }
                recon.store_luma(g, mb_x, mb_y, &pred_y, &luma_res);
                if let Some(cs) = cs.as_mut() {
                    encode_end_of_slice_flag(&mut cs.enc, mb_addr as usize + 1 == n_mbs);
                }
                *mv_grid.slot_mut(mb_x, mb_y) = MvGridSlot {
                    available: true,
                    is_intra: false,
                    ref_idx_l0_8x8: [0; 4],
                    mv_l0_8x8: [mv; 4],
                };
                dbl[mb_addr as usize] = MbDeblockInfo {
                    is_intra: false,
                    qp_y,
                    luma_nonzero_4x4: luma_nz_mask_from_blocks(&blk_has_nz),
                    chroma_nonzero_4x4: chroma_nz_mask,
                    transform_size_8x8_flag: flag_8x8,
                    mv_l0: [(mv.x as i16, mv.y as i16); 16],
                    ref_idx_l0: [0; 4],
                    ref_poc_l0: [prev_poc; 4],
                    ..Default::default()
                };
            }
        }
        let slice_rbsp = if let Some(cs) = cs {
            let payload = cs.enc.finish();
            debug_assert!(sw.byte_aligned());
            let mut rbsp = sw.into_bytes();
            rbsp.extend_from_slice(&payload);
            rbsp
        } else {
            if pending_skip > 0 {
                sw.ue(pending_skip);
            }
            sw.rbsp_trailing_bits();
            sw.into_bytes()
        };
        let (nal_type, ref_idc) = if k == 0 {
            (NalUnitType::SliceIdr, 3)
        } else {
            (NalUnitType::SliceNonIdr, 2)
        };
        stream.extend_from_slice(&build_nal_unit(ref_idc, nal_type, &slice_rbsp));

        if cfg.deblock {
            deblock_recon_deep(
                cfg.width,
                cfg.height,
                &mut recon.y,
                &mut recon.u,
                &mut recon.v,
                &dbl,
                0,
                cf,
                g.bd_y,
                g.bd_c,
            );
        }
        recon_frames.push(recon.to_u16());
        prev = Some(recon);
        prev_poc = poc;
    }

    DeepEncoded {
        annex_b: stream,
        recon_frames,
        profile_idc,
        skipped_mbs,
        intra_mbs_in_p,
        mbs_8x8,
    }
}
