//! Round-453 — MBAFF (macroblock-adaptive frame/field) frame encoder.
//!
//! Encodes an interlaced sequence as **MBAFF frame pictures**
//! (§7.4.2.1.1 `frame_mbs_only_flag = 0`,
//! `mb_adaptive_frame_field_flag = 1`, `field_pic_flag = 0`): every
//! vertical macroblock pair carries a §7.3.4 `mb_field_decoding_flag`
//! and is coded either as two stacked FRAME macroblocks (§6.4.1
//! eqs. 6-7/6-8) or as two interleaved FIELD macroblocks — the top MB
//! holding the pair's even sample rows and the bottom MB the odd rows
//! (§6.4.1 eqs. 6-9/6-10, y-stride 2).
//!
//! Structure of the emitted stream (Annex B):
//!
//! * SPS: Main profile (77 — §A.2.1 bars interlace from Baseline),
//!   `frame_mbs_only_flag = 0`, `mb_adaptive_frame_field_flag = 1`,
//!   CAVLC, 4:2:0, `pic_order_cnt_type = 0`.
//! * Frame 0: IDR I picture; frames k > 0: P pictures (or non-IDR I
//!   pictures when [`MbaffConfig::p_frames`] is off) referencing the
//!   previous frame, every picture a reference with sliding-window
//!   marking.
//! * One slice per picture; POC lsb = 2k (both fields of a frame
//!   picture share the frame POC — `delta_pic_order_cnt_bottom` is
//!   absent from our PPS).
//!
//! **Neighbour derivation.** The per-MB coding itself reuses the
//! frame-based MB encoders (`Encoder::encode_mb_intra16x16` /
//! `Encoder::encode_p_mb_with_intra_fallback`) on a *virtual
//! neighbourhood*: before each macroblock, the §6.4.12.2 Table 6-4
//! process (the decoder's own [`crate::mb_address::mbaff_neigh_location`])
//! resolves every neighbouring sample location, 4x4 `TotalCoeff`
//! (§9.2.1.1 via §6.4.11.4/.5) and motion-data cell (§8.4.1.3.2 with
//! the eq. 8-217..8-220 frame↔field MV/refIdx scaling) against the
//! per-address state of the real MBAFF picture, and the resolved
//! values are installed as the plain raster A/B/C/D neighbours of a
//! per-MB scratch coordinate system. Frame MBs run in the full-height
//! frame geometry; field MBs run in a half-height field geometry whose
//! MB row is the pair row, so motion compensation reads the §8.4.2.1
//! same-parity field of the reference frame (`refIdxLX = 0`, coded as
//! §7.3.5.1 `ref_idx_l0` te(v) — present for every field MB because
//! the effective list holds `2 * num_ref_idx_l0_active` fields).
//!
//! **Skip pairs.** §7.3.4 codes `mb_field_decoding_flag` with the top
//! MB, or with the bottom MB when the top is skipped; when BOTH MBs of
//! a pair are skipped the flag is inferred per §7.4.4 (left pair, else
//! above pair, else frame). The driver mirrors the inference: when a
//! pair comes out fully skipped under a decision that differs from the
//! §7.4.4 inference, the pair is rolled back and re-encoded with the
//! inferred flag so encoder and decoder agree on the P_Skip motion
//! derivation geometry.
//!
//! Scope (fixture-grade, CAVLC 4:2:0): I pictures are all-Intra_16x16
//! (I_NxN needs the §8.3.1.1 MBAFF pred-mode plumbing); P pictures use
//! the full P_Skip / P_L0_16x16 / P_8x8 / Intra_16x16-fallback mode
//! set with real ME per MB geometry.

use crate::encoder::deblock::{deblock_recon_mbaff, MbDeblockInfo};
use crate::encoder::nal::build_nal_unit;
use crate::encoder::pps::{build_baseline_pps_rbsp, BaselinePpsConfig};
use crate::encoder::slice::{
    write_idr_i_slice_header, write_p_slice_header, FieldPicSignal, IdrSliceHeaderConfig,
    PSliceHeaderConfig,
};
use crate::encoder::sps::{build_baseline_sps_rbsp, BaselineSpsConfig};
use crate::encoder::{
    min_level_idc_for_picture_size, BitWriter, CavlcNcGrid, EncodedFrameRef, Encoder,
    EncoderConfig, IntraGrid, MbQpTracker, MvGrid, MvGridSlot, YuvFrame,
};
use crate::macroblock_layer::{blk4x4_idx, CavlcMbNc};
use crate::mb_address::{mbaff_mb_to_sample_xy, mbaff_neigh_location, mbaff_pair_neighbour_addrs};
use crate::mv_deriv::Mv;
use crate::nal::NalUnitType;
use crate::transform::{qp_bd_offset, qp_y_to_qp_c_with_bd_offset};

/// One picture's (Y, Cb, Cr) planes.
pub(crate) type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// Per-pair frame/field decision policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairMode {
    /// Every pair frame-coded (`mb_field_decoding_flag = 0`).
    AllFrame,
    /// Every pair field-coded (`mb_field_decoding_flag = 1`).
    AllField,
    /// Per-pair vertical-activity decision: field when the summed
    /// same-field row gradient (|y[r] − y[r+2]|) undercuts the frame
    /// row gradient (|y[r] − y[r+1]|) over the pair — the classic
    /// inter-field-motion detector.
    Adaptive,
    /// Checkerboard (pair parity), exercising every §6.4.12.2
    /// frame/field neighbour combination regardless of content.
    Checker,
}

/// Configuration for [`encode_mbaff_sequence`].
#[derive(Debug, Clone)]
pub struct MbaffConfig {
    /// Luma width in samples (multiple of 16).
    pub width: u32,
    /// Frame luma height in samples (multiple of 32 so each field MB
    /// of a pair is 16 rows).
    pub frame_height: u32,
    /// Slice QP_Y (coded as `pic_init_qp_minus26`, `slice_qp_delta=0`).
    pub qp: i32,
    /// Frame/field decision policy per macroblock pair.
    pub pair_mode: PairMode,
    /// When `true`, frames after the IDR are P pictures referencing
    /// the previous frame; when `false` every frame is an I picture.
    pub p_frames: bool,
}

/// Output of [`encode_mbaff_sequence`].
#[derive(Debug, Clone)]
pub struct MbaffEncoded {
    /// Annex B stream: SPS, PPS, one slice NAL per frame.
    pub annex_b: Vec<u8>,
    /// Deblocked reconstruction of every frame (Y, Cb, Cr) — what a
    /// conforming decoder must output, in frame (interleaved-field)
    /// geometry.
    pub recon_frames: Vec<Planes>,
    /// Total field-coded macroblock pairs across the sequence.
    pub field_pairs: usize,
    /// Total frame-coded macroblock pairs across the sequence.
    pub frame_pairs: usize,
    /// Total skipped macroblocks across the sequence (P pictures).
    pub skipped_mbs: usize,
    /// Fully-skipped pairs re-encoded under the §7.4.4 inferred
    /// `mb_field_decoding_flag` because the first-choice flag was not
    /// inferable.
    pub inference_reencodes: usize,
}

/// Per-picture MBAFF state, indexed by the pair-interleaved macroblock
/// address (§6.4.1: `mbAddr = 2 * pairIdx + isBottom`).
pub(crate) struct PicState {
    pub(crate) w: usize,
    pub(crate) w_mbs: usize,
    /// Pre-deblock reconstruction, frame geometry.
    pub(crate) ry: Vec<u8>,
    pub(crate) ru: Vec<u8>,
    pub(crate) rv: Vec<u8>,
    /// §7.4.4 `mb_field_decoding_flag` per MB (pair-shared).
    pub(crate) field: Vec<bool>,
    /// MB has been coded (neighbour availability, §6.4.8).
    pub(crate) avail: Vec<bool>,
    /// §9.2.1.1 CAVLC neighbour state per MB.
    pub(crate) nc: Vec<CavlcMbNc>,
    /// §8.4.1.3.2 motion neighbour state per MB (unscaled, in the
    /// MB's own frame/field geometry).
    pub(crate) mv: Vec<MvGridSlot>,
    /// §8.7 deblock facts per MB.
    pub(crate) dbl: Vec<MbDeblockInfo>,
}

impl PicState {
    pub(crate) fn new(w: usize, h: usize) -> Self {
        let w_mbs = w / 16;
        let n = w_mbs * (h / 16);
        Self {
            w,
            w_mbs,
            ry: vec![0u8; w * h],
            ru: vec![0u8; (w / 2) * (h / 2)],
            rv: vec![0u8; (w / 2) * (h / 2)],
            field: vec![false; n],
            avail: vec![false; n],
            nc: vec![CavlcMbNc::default(); n],
            mv: vec![MvGridSlot::default(); n],
            dbl: vec![MbDeblockInfo::default(); n],
        }
    }

    /// §6.4.12.2 — resolve neighbouring location `(xn, yn)` of MB
    /// `addr` to `(mbAddrN, xW, yW)` via the decoder's Table 6-4
    /// implementation, reading real per-MB field flags.
    pub(crate) fn resolve(
        &self,
        addr: usize,
        xn: i32,
        yn: i32,
        max_w: i32,
        max_h: i32,
    ) -> Option<(u32, i32, i32)> {
        let pair = mbaff_pair_neighbour_addrs(addr as u32, self.w_mbs as u32);
        mbaff_neigh_location(
            xn,
            yn,
            max_w,
            max_h,
            !self.field[addr],
            addr % 2 == 0,
            addr as u32,
            pair,
            |a| !self.field[a as usize],
        )
    }

    /// §6.4.1 — read one luma sample of macroblock `addr` at MB-local
    /// `(xw, yw)` from the frame-geometry recon.
    fn luma_at(&self, addr: u32, xw: i32, yw: i32) -> u8 {
        let a = addr as usize;
        let f = self.field[a];
        let (ox, oy) = mbaff_mb_to_sample_xy(addr, self.w_mbs as u32, f);
        let stride: i32 = if f { 2 } else { 1 };
        let x = ox as i32 + xw;
        let y = oy as i32 + yw * stride;
        self.ry[y as usize * self.w + x as usize]
    }

    /// §6.4.1 — chroma sample of macroblock `addr` at chroma-local
    /// `(xw, yw)` (4:2:0, 8x8 chroma MBs).
    fn chroma_at(&self, cr: bool, addr: u32, xw: i32, yw: i32) -> u8 {
        let a = addr as usize;
        let f = self.field[a];
        let cw = self.w / 2;
        let pair_idx = addr / 2;
        let bot = (addr % 2) as i32;
        let cx = (pair_idx % self.w_mbs as u32) as i32 * 8;
        let cpy = (pair_idx / self.w_mbs as u32) as i32 * 16;
        let (oy, stride) = if f {
            (cpy + bot, 2)
        } else {
            (cpy + bot * 8, 1)
        };
        let plane = if cr { &self.rv } else { &self.ru };
        plane[(oy + yw * stride) as usize * cw + (cx + xw) as usize]
    }
}

/// Scratch coordinate system for one MB-geometry (frame or field).
pub(crate) struct VirtualEnc {
    pub(crate) enc: Encoder,
    pub(crate) h_mbs: usize,
    pub(crate) vy: Vec<u8>,
    pub(crate) vu: Vec<u8>,
    pub(crate) vv: Vec<u8>,
}

impl VirtualEnc {
    pub(crate) fn new(cfg: EncoderConfig) -> Self {
        let (w, h) = (cfg.width as usize, cfg.height as usize);
        Self {
            enc: Encoder::new(cfg),
            h_mbs: h / 16,
            vy: vec![0u8; w * h],
            vu: vec![0u8; (w / 2) * (h / 2)],
            vv: vec![0u8; (w / 2) * (h / 2)],
        }
    }
}

/// Extract one parity field (4:2:0 planes) from frame-geometry planes.
pub(crate) fn extract_field_planes(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    w: usize,
    frame_h: usize,
    bottom: bool,
) -> Planes {
    let off = usize::from(bottom);
    let fy: Vec<u8> = (0..frame_h / 2)
        .flat_map(|r| y[(2 * r + off) * w..(2 * r + off) * w + w].iter().copied())
        .collect();
    let cw = w / 2;
    let ch = frame_h / 2;
    let fu: Vec<u8> = (0..ch / 2)
        .flat_map(|r| {
            u[(2 * r + off) * cw..(2 * r + off) * cw + cw]
                .iter()
                .copied()
        })
        .collect();
    let fv: Vec<u8> = (0..ch / 2)
        .flat_map(|r| {
            v[(2 * r + off) * cw..(2 * r + off) * cw + cw]
                .iter()
                .copied()
        })
        .collect();
    (fy, fu, fv)
}

/// The scratch-system position of MB `addr` under decision `field`:
/// `(vx, vy)` in the MB grid of the corresponding [`VirtualEnc`]
/// (frame geometry for frame MBs, half-height field geometry — MB row
/// = pair row — for field MBs).
pub(crate) fn virtual_pos(addr: usize, w_mbs: usize, field: bool) -> (usize, usize) {
    let pair_idx = addr / 2;
    let bot = addr % 2;
    let vx = pair_idx % w_mbs;
    let pr = pair_idx / w_mbs;
    if field {
        (vx, pr)
    } else {
        (vx, 2 * pr + bot)
    }
}

/// §7.4.4 — the inferred `mb_field_decoding_flag` for a fully-skipped
/// pair: left pair, else above pair, else 0.
fn inferred_pair_flag(st: &PicState, addr: usize) -> bool {
    let [a, b, _c, _d] = mbaff_pair_neighbour_addrs(addr as u32, st.w_mbs as u32);
    for n in [a, b].into_iter().flatten() {
        if st.avail[n as usize] {
            return st.field[n as usize];
        }
    }
    false
}

/// Patch the scratch recon planes around the MB's virtual position
/// with the §6.4.12.2-resolved neighbouring samples (top row incl.
/// corners, left column) so the frame-based intra predictors read
/// exactly what the decoder's Table 6-4 process yields.
pub(crate) fn patch_neighbour_samples(
    st: &PicState,
    ve: &mut VirtualEnc,
    addr: usize,
    vx: usize,
    vy: usize,
) {
    let w = st.w;
    let cw = w / 2;
    // Luma: (xN, yN) over the top row -1..=16 at yN = -1 and the left
    // column at xN = -1 (Intra_16x16 uses -1..=15 / 0..=15; the extra
    // corner sample is harmless).
    for xn in -1..=16i32 {
        if let Some((n, xw, yw)) = st.resolve(addr, xn, -1, 16, 16) {
            debug_assert!(vy > 0 || xn < 0, "resolved top neighbour at picture row 0");
            let (px, py) = (vx as i32 * 16 + xn, vy as i32 * 16 - 1);
            if px >= 0 && (px as usize) < w {
                ve.vy[py as usize * w + px as usize] = st.luma_at(n, xw, yw);
            }
        }
    }
    for yn in 0..16i32 {
        if let Some((n, xw, yw)) = st.resolve(addr, -1, yn, 16, 16) {
            debug_assert!(vx > 0);
            let (px, py) = (vx as i32 * 16 - 1, vy as i32 * 16 + yn);
            ve.vy[py as usize * w + px as usize] = st.luma_at(n, xw, yw);
        }
    }
    // Chroma (4:2:0 — 8x8 chroma MBs).
    for xn in -1..=8i32 {
        if let Some((n, xw, yw)) = st.resolve(addr, xn, -1, 8, 8) {
            let (px, py) = (vx as i32 * 8 + xn, vy as i32 * 8 - 1);
            if px >= 0 && (px as usize) < cw {
                ve.vu[py as usize * cw + px as usize] = st.chroma_at(false, n, xw, yw);
                ve.vv[py as usize * cw + px as usize] = st.chroma_at(true, n, xw, yw);
            }
        }
    }
    for yn in 0..8i32 {
        if let Some((n, xw, yw)) = st.resolve(addr, -1, yn, 8, 8) {
            let (px, py) = (vx as i32 * 8 - 1, vy as i32 * 8 + yn);
            ve.vu[py as usize * cw + px as usize] = st.chroma_at(false, n, xw, yw);
            ve.vv[py as usize * cw + px as usize] = st.chroma_at(true, n, xw, yw);
        }
    }
}

/// Populate the scratch A/B/C/D neighbour slots of the per-MB grids
/// from the real MBAFF state: §9.2.1.1 effective per-4x4 `TotalCoeff`
/// (skip → 0, I_PCM → 16), §8.4.1.3.2 motion cells with the
/// eq. 8-217..8-220 frame↔field scaling, and §6.4.8 availability.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_neighbour_grids(
    st: &PicState,
    addr: usize,
    field: bool,
    vx: usize,
    vy: usize,
    w_mbs: usize,
    h_mbs: usize,
    nc_grid: &mut CavlcNcGrid,
    intra_grid: &mut IntraGrid,
    mv_grid: &mut MvGrid,
) {
    // --- Motion cells: (slot dx, dy, 8x8 cell, xN, yN). ---
    // A feeds partitions 0/2 (queries (-1,0)/(-1,8) → cells #1/#3), B
    // feeds partitions 0/1 ((0,-1)/(8,-1) → cells #2/#3), C the MB
    // corner ((16,-1) → cell #2), D the corner fallback ((-1,-1) →
    // cell #3) — exactly the cells `neighbour_mvs_16x16` /
    // `mvp_for_p_8x8_partition` read.
    const MV_FILLS: [(i32, i32, usize, i32, i32); 6] = [
        (-1, 0, 1, -1, 0),
        (-1, 0, 3, -1, 8),
        (0, -1, 2, 0, -1),
        (0, -1, 3, 8, -1),
        (1, -1, 2, 16, -1),
        (-1, -1, 3, -1, -1),
    ];
    for &(dx, dy, cell, xn, yn) in &MV_FILLS {
        let Some((n, xw, yw)) = st.resolve(addr, xn, yn, 16, 16) else {
            continue;
        };
        let (sx, sy) = (vx as i32 + dx, vy as i32 + dy);
        debug_assert!(
            sx >= 0 && (sx as usize) < w_mbs && sy >= 0 && (sy as usize) < h_mbs,
            "resolved neighbour outside the scratch grid",
        );
        let na = n as usize;
        debug_assert!(st.avail[na]);
        let real = &st.mv[na];
        let nf = st.field[na];
        let slot = mv_grid.slot_mut(sx as usize, sy as usize);
        slot.available = true;
        slot.is_intra = false;
        let cell8 = ((yw / 8) * 2 + (xw / 8)) as usize;
        if real.is_intra || real.ref_idx_l0_8x8[cell8] < 0 {
            // §8.4.1.3.2 — intra neighbour: mvLXN = 0, refIdxLXN = −1
            // while the macroblock itself stays available.
            slot.ref_idx_l0_8x8[cell] = -1;
            slot.mv_l0_8x8[cell] = Mv::ZERO;
        } else {
            let mut mv = real.mv_l0_8x8[cell8];
            let mut ref_idx = real.ref_idx_l0_8x8[cell8];
            if field && !nf {
                // eq. 8-217 / 8-218 — field MB reading a frame MB.
                mv.y /= 2;
                ref_idx *= 2;
            } else if !field && nf {
                // eq. 8-219 / 8-220 — frame MB reading a field MB.
                mv.y *= 2;
                ref_idx /= 2;
            }
            slot.ref_idx_l0_8x8[cell] = ref_idx;
            slot.mv_l0_8x8[cell] = mv;
        }
    }

    // --- §9.2.1.1 luma / chroma TotalCoeff neighbours. ---
    // Effective nN of the real block at the resolved location: 0 for
    // skipped MBs, 16 for I_PCM, else the stored per-4x4 TotalCoeff.
    let effective_luma = |n: u32, xw: i32, yw: i32| -> u8 {
        let s = &st.nc[n as usize];
        if s.is_skip {
            0
        } else if s.is_i_pcm {
            16
        } else {
            s.luma_total_coeff[blk4x4_idx(xw, yw) as usize]
        }
    };
    let effective_chroma = |cr: bool, n: u32, xw: i32, yw: i32| -> u8 {
        let s = &st.nc[n as usize];
        if s.is_skip {
            0
        } else if s.is_i_pcm {
            16
        } else {
            let idx = (2 * (yw / 4) + (xw / 4)) as usize;
            if cr {
                s.cr_total_coeff[idx]
            } else {
                s.cb_total_coeff[idx]
            }
        }
    };
    let mark = |grid: &mut CavlcNcGrid, sx: i32, sy: i32| -> Option<usize> {
        if sx < 0 || sy < 0 || sx as usize >= w_mbs || sy as usize >= h_mbs {
            return None;
        }
        let vaddr = sy as usize * w_mbs + sx as usize;
        let slot = &mut grid.mbs[vaddr];
        slot.is_available = true;
        slot.is_skip = false;
        slot.is_i_pcm = false;
        slot.is_intra = false;
        Some(vaddr)
    };
    // A (left MB in scratch coords): luma blocks (3, k) from
    // queries (-1, 4k); chroma blocks (1, k) from (-1, 4k).
    for k in 0..4i32 {
        if let Some((n, xw, yw)) = st.resolve(addr, -1, 4 * k, 16, 16) {
            if let Some(vaddr) = mark(nc_grid, vx as i32 - 1, vy as i32) {
                nc_grid.mbs[vaddr].luma_total_coeff[blk4x4_idx(12, 4 * k) as usize] =
                    effective_luma(n, xw, yw);
            }
        }
    }
    for k in 0..2i32 {
        if let Some((n, xw, yw)) = st.resolve(addr, -1, 4 * k, 8, 8) {
            if let Some(vaddr) = mark(nc_grid, vx as i32 - 1, vy as i32) {
                let idx = (2 * k + 1) as usize;
                nc_grid.mbs[vaddr].cb_total_coeff[idx] = effective_chroma(false, n, xw, yw);
                nc_grid.mbs[vaddr].cr_total_coeff[idx] = effective_chroma(true, n, xw, yw);
            }
        }
    }
    // B (above MB in scratch coords): luma blocks (k, 3) from
    // (4k, -1); chroma blocks (k, 1) from (4k, -1).
    for k in 0..4i32 {
        if let Some((n, xw, yw)) = st.resolve(addr, 4 * k, -1, 16, 16) {
            if let Some(vaddr) = mark(nc_grid, vx as i32, vy as i32 - 1) {
                nc_grid.mbs[vaddr].luma_total_coeff[blk4x4_idx(4 * k, 12) as usize] =
                    effective_luma(n, xw, yw);
            }
        }
    }
    for k in 0..2i32 {
        if let Some((n, xw, yw)) = st.resolve(addr, 4 * k, -1, 8, 8) {
            if let Some(vaddr) = mark(nc_grid, vx as i32, vy as i32 - 1) {
                let idx = (2 + k) as usize;
                nc_grid.mbs[vaddr].cb_total_coeff[idx] = effective_chroma(false, n, xw, yw);
                nc_grid.mbs[vaddr].cr_total_coeff[idx] = effective_chroma(true, n, xw, yw);
            }
        }
    }

    // --- Intra-mode grid: only availability matters (no I_NxN in the
    // MBAFF mode set — Intra_16x16 neighbours take the §8.3.1.1 DC
    // fallback, which is what a default non-I_NxN slot yields). ---
    for (dx, dy) in [(-1i32, 0i32), (0, -1), (1, -1), (-1, -1)] {
        let (sx, sy) = (vx as i32 + dx, vy as i32 + dy);
        if sx < 0 || sy < 0 || sx as usize >= w_mbs || sy as usize >= h_mbs {
            continue;
        }
        // Availability mirrors the sample-level resolve of the MB
        // corner adjacent to that scratch slot.
        let probe = st.resolve(
            addr,
            if dx < 0 { -1 } else { dx * 16 },
            if dy < 0 { -1 } else { 0 },
            16,
            16,
        );
        if probe.is_some() {
            let slot = intra_grid.slot_mut(sx as usize, sy as usize);
            slot.available = true;
            slot.is_i_nxn = false;
        }
    }
}

/// §8.7.2.1 NOTE 1 — the picture-identity key the decoder stores for a
/// referenced picture: `poc*4` for a frame reference (frame MB),
/// `poc*4 + 1 + parity` for a field reference (field MB; `ref_idx = 0`
/// selects the same-parity field per §8.4.2.1).
pub(crate) fn ref_poc_key(prev_poc: i32, field: bool, mb_parity: u32) -> i32 {
    if field {
        prev_poc * 4 + 1 + mb_parity as i32
    } else {
        prev_poc * 4
    }
}

/// Result of coding one macroblock.
struct MbOut {
    dbl: MbDeblockInfo,
    skipped: bool,
}

struct MbCtx<'a> {
    qp_y: i32,
    qp_c: i32,
    prev_poc: i32,
    /// P references, per geometry: `[frame, top field, bottom field]`.
    refs: Option<&'a [Planes; 3]>,
    sources_field: &'a [Planes; 2],
    src_frame: YuvFrame<'a>,
}

/// Code one macroblock of the MBAFF picture into `sw`, mirroring the
/// decoder's reconstruction into the real frame-geometry planes.
#[allow(clippy::too_many_arguments)]
fn code_mb(
    st: &mut PicState,
    ve: &mut VirtualEnc,
    ctx: &MbCtx<'_>,
    addr: usize,
    field: bool,
    is_p: bool,
    sw: &mut BitWriter,
    tracker: &mut MbQpTracker,
    pending_skip: &mut u32,
) -> MbOut {
    let w_mbs = st.w_mbs;
    let (vx, vy) = virtual_pos(addr, w_mbs, field);
    let parity = (addr % 2) as u32;

    // The MB must know its own geometry before Table 6-4 runs.
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

    tracker.mbaff_field_mb = field;
    sw.set_field_scan(field);

    let src = if field {
        let (fy, fu, fv) = &ctx.sources_field[parity as usize];
        YuvFrame {
            width: ve.enc.config().width,
            height: ve.enc.config().height,
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

    let chroma_w = st.w / 2;
    let chroma_h = (ve.enc.config().height / 2) as usize;

    let skipped;
    let mut dbl;
    if is_p {
        let refs = ctx.refs.expect("P picture without references");
        let (ry, ru, rv) = if field {
            &refs[1 + parity as usize]
        } else {
            &refs[0]
        };
        let prev = EncodedFrameRef {
            width: ve.enc.config().width,
            height: ve.enc.config().height,
            recon_y: ry,
            recon_u: ru,
            recon_v: rv,
            partition_mvs: &[],
            pic_order_cnt: ctx.prev_poc,
        };
        let before = *pending_skip;
        dbl = ve.enc.encode_p_mb_with_intra_fallback(
            &src,
            &prev,
            prev.recon_y,
            vx,
            vy,
            ctx.qp_y,
            ctx.qp_c,
            chroma_w,
            chroma_h,
            &mut ve.vy,
            &mut ve.vu,
            &mut ve.vv,
            sw,
            &mut nc_grid,
            &mut intra_grid,
            &mut mv_grid,
            pending_skip,
            tracker,
        );
        skipped = *pending_skip > before;
    } else {
        // All-Intra_16x16 I picture (see the module scope note).
        let trial = ve.enc.encode_mb_intra16x16(
            &src,
            vx,
            vy,
            ctx.qp_y,
            ctx.qp_c,
            chroma_w,
            chroma_h,
            &mut ve.vy,
            &mut ve.vu,
            &mut ve.vv,
            sw,
            &mut nc_grid,
            &mut intra_grid,
            0,
            None,
        );
        dbl = trial.deblock;
        skipped = false;
    }
    sw.set_field_scan(false);

    // --- Commit the MB into the real MBAFF state. ---
    let vaddr = vy * w_mbs + vx;
    st.nc[addr] = nc_grid.mbs[vaddr];
    st.mv[addr] = if is_p {
        *mv_grid.slot(vx, vy)
    } else {
        MvGridSlot {
            available: true,
            is_intra: true,
            ref_idx_l0_8x8: [-1; 4],
            mv_l0_8x8: [Mv::ZERO; 4],
        }
    };
    // §8.7.2.1 NOTE 1 — rewrite the reference-identity keys with the
    // MBAFF frame/field picture identities the decoder derives.
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

    MbOut { dbl, skipped }
}

/// Per-pair frame/field decision for [`PairMode::Adaptive`]: compare
/// cross-row activity in frame vs same-field arrangement over the
/// pair's luma samples.
pub(crate) fn adaptive_pair_is_field(y: &[u8], w: usize, pair_col: usize, pair_row: usize) -> bool {
    let x0 = pair_col * 16;
    let y0 = pair_row * 32;
    let mut frame_cost: u64 = 0;
    let mut field_cost: u64 = 0;
    for x in x0..x0 + 16 {
        for r in 0..31usize {
            let a = y[(y0 + r) * w + x] as i64;
            let b = y[(y0 + r + 1) * w + x] as i64;
            frame_cost += a.abs_diff(b);
        }
        for r in 0..30usize {
            let a = y[(y0 + r) * w + x] as i64;
            let b = y[(y0 + r + 2) * w + x] as i64;
            field_cost += a.abs_diff(b);
        }
    }
    // Scale to the same comparison count (31 vs 30 diffs per column).
    field_cost * 31 < frame_cost * 30
}

/// Encode an interlaced sequence as MBAFF frame pictures. `frames`
/// holds frame-geometry 4:2:0 planes (top field = even rows).
pub fn encode_mbaff_sequence(cfg: &MbaffConfig, frames: &[(&[u8], &[u8], &[u8])]) -> MbaffEncoded {
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

    // §A.2.1 bars interlace from Baseline — Main (77), CAVLC.
    let profile_idc: u8 = 77;
    let mk_cfg = |h: u32| {
        let mut c = EncoderConfig::new(cfg.width, h);
        c.qp = cfg.qp;
        c.profile_idc = profile_idc;
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
        entropy_coding_mode_flag: false,
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
    // Previous frame's deblocked recon: frame + per-parity field views.
    let mut prev_refs: Option<[Planes; 3]> = None;
    let mut prev_poc: i32 = 0;

    let mut field_pairs = 0usize;
    let mut frame_pairs = 0usize;
    let mut skipped_mbs = 0usize;
    let mut inference_reencodes = 0usize;

    for (k, &(fy, fu, fv)) in frames.iter().enumerate() {
        assert_eq!(fy.len(), w * frame_h);
        assert_eq!(fu.len(), (w / 2) * (frame_h / 2));
        let is_p = cfg.p_frames && k > 0;
        let frame_num = (k as u32) % (1 << frame_num_bits);
        let poc = 2 * k as i32;

        let mut sw = BitWriter::new();
        if k == 0 {
            write_idr_i_slice_header(
                &mut sw,
                &IdrSliceHeaderConfig {
                    first_mb_in_slice: 0,
                    slice_type_raw: 7,
                    pic_parameter_set_id: 0,
                    colour_plane_id: None,
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
                    long_term_reference_flag: false,
                    mmco: &[],
                    redundant_pic_cnt: None,
                    slice_group_change_cycle: None,
                },
            );
        } else if is_p {
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
                    cabac: None,
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
                    idr: false,
                    nal_ref_idc: 2,
                    long_term_reference_flag: false,
                    mmco: &[],
                    redundant_pic_cnt: None,
                    slice_group_change_cycle: None,
                },
            );
        }

        let mut st = PicState::new(w, frame_h);
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
        };

        let mut tracker = MbQpTracker::new(qp_y);
        let mut pending_skip: u32 = 0;
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
                let sw_snap = sw.clone();
                let skip_snap = pending_skip;
                let tracker_snap = tracker;
                let ve = if field { &mut ve_field } else { &mut ve_frame };
                let (top_out, bot_out) = if is_p {
                    // §7.3.4 — flag rides the first non-skipped MB of
                    // the pair (pending until a skip-run flush).
                    tracker.mbaff_flag_pending = Some(field);
                    let t = code_mb(
                        &mut st,
                        ve,
                        &ctx,
                        top,
                        field,
                        true,
                        &mut sw,
                        &mut tracker,
                        &mut pending_skip,
                    );
                    let b = code_mb(
                        &mut st,
                        ve,
                        &ctx,
                        top + 1,
                        field,
                        true,
                        &mut sw,
                        &mut tracker,
                        &mut pending_skip,
                    );
                    (t, b)
                } else {
                    // I slice — no skips: flag coded with the top MB.
                    sw.u(1, u32::from(field));
                    let t = code_mb(
                        &mut st,
                        ve,
                        &ctx,
                        top,
                        field,
                        false,
                        &mut sw,
                        &mut tracker,
                        &mut pending_skip,
                    );
                    let b = code_mb(
                        &mut st,
                        ve,
                        &ctx,
                        top + 1,
                        field,
                        false,
                        &mut sw,
                        &mut tracker,
                        &mut pending_skip,
                    );
                    (t, b)
                };
                if top_out.skipped && bot_out.skipped {
                    // §7.4.4 — a fully-skipped pair codes no
                    // mb_field_decoding_flag; the decoder infers it.
                    // Re-encode under the inferred flag when it
                    // differs from our choice.
                    tracker.mbaff_flag_pending = None;
                    st.field[top] = field;
                    st.field[top + 1] = field;
                    let inferred = inferred_pair_flag(&st, top);
                    if inferred != field {
                        sw = sw_snap;
                        pending_skip = skip_snap;
                        tracker = tracker_snap;
                        st.avail[top] = false;
                        st.avail[top + 1] = false;
                        st.nc[top] = CavlcMbNc::default();
                        st.nc[top + 1] = CavlcMbNc::default();
                        st.mv[top] = MvGridSlot::default();
                        st.mv[top + 1] = MvGridSlot::default();
                        field = inferred;
                        inference_reencodes += 1;
                        continue;
                    }
                }
                if top_out.skipped {
                    skipped_mbs += 1;
                }
                if bot_out.skipped {
                    skipped_mbs += 1;
                }
                break;
            }
            // A fully-skipped pair leaves the pending flag set (the
            // decoder infers it per §7.4.4 — the loop above guarantees
            // our choice equals the inference); clear it before the
            // next pair re-arms it.
            tracker.mbaff_flag_pending = None;
            if field {
                field_pairs += 1;
            } else {
                frame_pairs += 1;
            }
        }
        if pending_skip > 0 {
            // §7.3.4 — trailing skip run.
            sw.ue(pending_skip);
        }
        sw.rbsp_trailing_bits();
        let (nal_type, ref_idc) = if k == 0 {
            (NalUnitType::SliceIdr, 3)
        } else {
            (NalUnitType::SliceNonIdr, 2)
        };
        stream.extend_from_slice(&build_nal_unit(ref_idc, nal_type, &sw.into_bytes()));

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

        // The deblocked frame is the reference for the next picture:
        // whole-frame planes for frame MBs, §8.4.2.1 parity field
        // views for field MBs.
        let top_view = extract_field_planes(&ry, &ru, &rv, w, frame_h, false);
        let bot_view = extract_field_planes(&ry, &ru, &rv, w, frame_h, true);
        prev_refs = Some([(ry.clone(), ru.clone(), rv.clone()), top_view, bot_view]);
        prev_poc = poc;
        recon_frames.push((ry, ru, rv));
    }

    MbaffEncoded {
        annex_b: stream,
        recon_frames,
        field_pairs,
        frame_pairs,
        skipped_mbs,
        inference_reencodes,
    }
}
