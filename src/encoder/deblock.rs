//! §8.7 — In-loop deblocking adapter for the encoder's local recon.
//!
//! The decoder applies §8.7 deblocking as a single picture-level pass
//! after every macroblock has been reconstructed. The encoder's local
//! recon must mirror the decoder's output, so it has to run the exact
//! same filter — otherwise its "I_Have_The_Decoded_Picture" buffer
//! diverges from what downstream decoders see, which would corrupt
//! reference pictures for any future inter-coded frames *and* produce
//! ffmpeg-interop test failures.
//!
//! This module is a thin adapter:
//!
//! 1. Encoder collects per-MB facts into [`MbDeblockInfo`] entries as
//!    each MB is encoded (intra/non-intra, QP_Y, per-4x4 nonzero mask).
//! 2. After all MBs are written, [`deblock_recon`] wraps the encoder's
//!    plain `Vec<u8>` recon planes in a [`Picture`] (i32 samples) and
//!    the per-MB facts in a [`MbGrid`], then calls
//!    [`crate::reconstruct::deblock_picture_full`] verbatim.
//! 3. The filtered samples are copied back into the encoder's recon
//!    buffers so subsequent operations (and the encoder's exported
//!    `EncodedIdr.recon_*`) see the post-deblock picture.
//!
//! Spec references throughout cite ITU-T Rec. H.264 (08/2024). Clean-room.

use crate::mb_grid::{MbGrid, MbInfo};
use crate::picture::Picture;
use crate::pps::Pps;
use crate::reconstruct::deblock_picture_full;

/// Per-MB facts the deblock walker reads from [`MbInfo`]. The encoder
/// fills one of these per MB as it encodes; [`build_mb_grid`] turns
/// the array into a fully-populated [`MbGrid`].
///
/// Round 14 scope (intra-only, single slice):
/// * `is_intra` is always true for our IDR pictures, but kept here so
///   later rounds (P/B slices) can reuse this adapter.
/// * `mv_l0`/`mv_l1`/`ref_idx_*`/`ref_poc_*` are unused for intra-only
///   content — `different_ref_or_mv_luma` short-circuits on
///   `is_intra && q_is_intra`.
///
/// Round 20 adds the L1 mirror (`mv_l1`, `ref_idx_l1`, `ref_poc_l1`)
/// so that B-slice MBs feed the deblock walker's bipred edge-strength
/// derivation correctly. Intra/P MBs leave the L1 fields at their
/// defaults (ref_idx = -1, mv = 0, poc = i32::MIN), which the walker
/// reads as "no L1 contribution" via the `predFlagL1 = 0` branch in
/// §8.7.2.1.
#[derive(Debug, Clone, Copy)]
pub struct MbDeblockInfo {
    pub is_intra: bool,
    pub qp_y: i32,
    /// §8.7.2.1 — per-4x4-luma-block nonzero-coefficient bitmap, indexed
    /// by §6.4.3 Figure 6-10 Z-scan (bit `n` ↔ 4x4 block at Z-scan
    /// position `n`). For Intra_16x16 this is set to 0xFFFF when
    /// `cbp_luma == 15` (all 4x4 blocks transmitted with non-zero AC) and
    /// 0 when `cbp_luma == 0` (DC-only). For I_NxN it's the per-block
    /// non-zero flag of the 8x8-quadrant-transmitted blocks.
    pub luma_nonzero_4x4: u16,
    /// §8.7.2.1 — per-4x4-chroma-block nonzero bitmap. For 4:2:0 only
    /// bits 0..=3 (Cb) and 8..=11 (Cr) are populated.
    pub chroma_nonzero_4x4: u16,
    /// Round-16: §8.4.1 per-4x4-block L0 MV (1/4-pel units). Used by
    /// the deblock walker's `different_ref_or_mv_luma` to derive
    /// boundary strength on inter/inter edges. For intra MBs this is
    /// unused. For our P_L0_16x16 / P_Skip path the same MV is
    /// replicated to all 16 4x4 blocks.
    pub mv_l0: [(i16, i16); 16],
    /// Round-16: §7.4.5.1 per-8x8-partition L0 ref_idx. `-1` for intra
    /// or L0-unused (B_L1_16x16).
    pub ref_idx_l0: [i8; 4],
    /// Round-16: §8.7.2.1 NOTE 1 per-8x8-partition L0 referenced POC.
    /// `i32::MIN` for intra / L0 unused. For our single-IDR-ref encoder
    /// every P MB references POC 0.
    pub ref_poc_l0: [i32; 4],
    /// Round-20: §8.4.1 per-4x4-block L1 MV (1/4-pel units). Same role
    /// as `mv_l0` but for the second reference list (only meaningful
    /// for B-slice MBs whose pred uses L1).
    pub mv_l1: [(i16, i16); 16],
    /// Round-20: §7.4.5.1 per-8x8-partition L1 ref_idx. `-1` for intra
    /// or L1-unused (B_L0_16x16, every P_L0_* MB).
    pub ref_idx_l1: [i8; 4],
    /// Round-20: §8.7.2.1 NOTE 1 per-8x8-partition L1 referenced POC.
    /// `i32::MIN` for intra / L1 unused.
    pub ref_poc_l1: [i32; 4],
}

impl Default for MbDeblockInfo {
    fn default() -> Self {
        Self {
            is_intra: false,
            qp_y: 0,
            luma_nonzero_4x4: 0,
            chroma_nonzero_4x4: 0,
            mv_l0: [(0, 0); 16],
            ref_idx_l0: [-1; 4],
            ref_poc_l0: [i32::MIN; 4],
            mv_l1: [(0, 0); 16],
            ref_idx_l1: [-1; 4],
            ref_poc_l1: [i32::MIN; 4],
        }
    }
}

/// Wrap the encoder's recon planes + per-MB facts into a [`Picture`] +
/// [`MbGrid`], invoke [`deblock_picture_full`], then copy the filtered
/// samples back. After this returns, `recon_y/u/v` reflect the §8.7
/// post-filter picture — what every downstream decoder will reproduce.
///
/// `width`, `height` are luma dimensions in samples (already aligned to
/// 16 by the caller). `chroma_width`/`chroma_height` follow §6.2 (4:2:0
/// — half W and half H).
///
/// `mb_infos` is row-major over (mb_y, mb_x), length `width_mbs *
/// height_mbs`.
///
/// `chroma_qp_index_offset` is the §7.4.2.2 PPS field; the deblock
/// chroma QP derivation needs it.
#[allow(clippy::too_many_arguments)]
pub fn deblock_recon(
    width: u32,
    height: u32,
    chroma_width: u32,
    chroma_height: u32,
    recon_y: &mut [u8],
    recon_u: &mut [u8],
    recon_v: &mut [u8],
    mb_infos: &[MbDeblockInfo],
    chroma_qp_index_offset: i32,
    width_mbs: u32,
    height_mbs: u32,
) {
    deblock_recon_with_chroma_array_type(
        width,
        height,
        chroma_width,
        chroma_height,
        recon_y,
        recon_u,
        recon_v,
        mb_infos,
        chroma_qp_index_offset,
        width_mbs,
        height_mbs,
        /* chroma_array_type */ 1,
    )
}

/// Round-27 — variant of [`deblock_recon`] that takes an explicit
/// `chroma_array_type` (1 = 4:2:0, 2 = 4:2:2). The §8.7 walker reads
/// the chroma layout from `Picture::chroma_array_type`, so picking the
/// right value here is sufficient for the chroma deblock pass to use
/// the 8x16 chroma MB geometry.
#[allow(clippy::too_many_arguments)]
pub fn deblock_recon_with_chroma_array_type(
    width: u32,
    height: u32,
    chroma_width: u32,
    chroma_height: u32,
    recon_y: &mut [u8],
    recon_u: &mut [u8],
    recon_v: &mut [u8],
    mb_infos: &[MbDeblockInfo],
    chroma_qp_index_offset: i32,
    width_mbs: u32,
    height_mbs: u32,
    chroma_array_type: u32,
) {
    debug_assert_eq!(recon_y.len(), (width as usize) * (height as usize));
    debug_assert_eq!(
        recon_u.len(),
        (chroma_width as usize) * (chroma_height as usize)
    );
    debug_assert_eq!(
        recon_v.len(),
        (chroma_width as usize) * (chroma_height as usize)
    );
    debug_assert_eq!(mb_infos.len(), (width_mbs as usize) * (height_mbs as usize));

    // ------- Build a Picture from the encoder's recon planes. -------
    // Picture stores samples as i32; promote from u8. The
    // `chroma_array_type` argument selects 4:2:0 (1) or 4:2:2 (2)
    // chroma layouts for the deblock walker.
    let mut pic = Picture::new(width, height, chroma_array_type, 8, 8);
    for (dst, src) in pic.luma.iter_mut().zip(recon_y.iter()) {
        *dst = *src as i32;
    }
    for (dst, src) in pic.cb.iter_mut().zip(recon_u.iter()) {
        *dst = *src as i32;
    }
    for (dst, src) in pic.cr.iter_mut().zip(recon_v.iter()) {
        *dst = *src as i32;
    }

    // ------- Build the MbGrid from per-MB facts. -------
    let mut grid = MbGrid::new(width_mbs, height_mbs);
    for (i, info) in mb_infos.iter().enumerate() {
        let mb = &mut grid.info[i];
        // §6.4.4 — every MB the encoder produced is "available" for
        // neighbour lookups by the deblock walker.
        mb.available = true;
        mb.is_intra = info.is_intra;
        mb.is_intra_nxn = false; // not used by deblock
        mb.is_i_pcm = false;
        mb.qp_y = info.qp_y;
        mb.luma_nonzero_4x4 = info.luma_nonzero_4x4;
        mb.chroma_nonzero_4x4 = info.chroma_nonzero_4x4;
        mb.transform_size_8x8_flag = false;
        // §6.4.8 third bullet — single slice, slice_id 0 throughout, so
        // no "different slice" neighbour rejections fire.
        mb.slice_id = 0;
        // Round-16: per-4x4 mv / per-8x8 ref_idx + ref_poc for inter MBs.
        // For intra MBs the encoder leaves ref_idx = -1 and the deblock
        // walker's `different_ref_or_mv_luma` is gated on !is_intra.
        mb.mv_l0 = info.mv_l0;
        mb.ref_idx_l0 = info.ref_idx_l0;
        mb.ref_poc_l0 = info.ref_poc_l0;
        // Round-20: same for L1 (B-slice MBs).
        mb.mv_l1 = info.mv_l1;
        mb.ref_idx_l1 = info.ref_idx_l1;
        mb.ref_poc_l1 = info.ref_poc_l1;
    }
    // For unset entries (fully zeroed regions of the grid in pictures
    // that don't fill a power-of-2 — shouldn't happen given the encoder
    // requires 16-pixel-aligned dimensions), MbInfo::default leaves
    // available=false so the deblock walker skips them anyway.
    let _ = MbInfo::default; // marker: see comment above

    // ------- Build a minimal PPS for the deblock pass. -------
    // Only `chroma_qp_index_offset` and (when present)
    // `extension.second_chroma_qp_index_offset` are consulted.
    let pps = Pps {
        pic_parameter_set_id: 0,
        seq_parameter_set_id: 0,
        entropy_coding_mode_flag: false,
        bottom_field_pic_order_in_frame_present_flag: false,
        num_slice_groups_minus1: 0,
        slice_group_map: None,
        num_ref_idx_l0_default_active_minus1: 0,
        num_ref_idx_l1_default_active_minus1: 0,
        weighted_pred_flag: false,
        weighted_bipred_idc: 0,
        pic_init_qp_minus26: 0,
        pic_init_qs_minus26: 0,
        chroma_qp_index_offset,
        deblocking_filter_control_present_flag: true,
        constrained_intra_pred_flag: false,
        redundant_pic_cnt_present_flag: false,
        extension: None,
    };

    // ------- Run the §8.7 picture-level pass. -------
    // alpha/beta offsets default to 0 (slice header writes
    // slice_alpha_c0_offset_div2 = slice_beta_offset_div2 = 0). The
    // walker scales the picture-grid 4x4 edges and bumps bS to 4 on
    // intra MB boundaries automatically — no encoder-side preselection
    // needed.
    deblock_picture_full(
        &mut pic,
        &grid,
        /* alpha_off */ 0,
        /* beta_off */ 0,
        /* bit_depth_y */ 8,
        /* bit_depth_c */ 8,
        &pps,
        /* mbaff_frame_flag */ false,
        &[], // mb_field_flags — empty in non-MBAFF
    );

    // ------- Copy filtered samples back to the encoder's u8 buffers. -------
    for (dst, src) in recon_y.iter_mut().zip(pic.luma.iter()) {
        *dst = (*src).clamp(0, 255) as u8;
    }
    for (dst, src) in recon_u.iter_mut().zip(pic.cb.iter()) {
        *dst = (*src).clamp(0, 255) as u8;
    }
    for (dst, src) in recon_v.iter_mut().zip(pic.cr.iter()) {
        *dst = (*src).clamp(0, 255) as u8;
    }
}

/// Convenience helper: derive a `luma_nonzero_4x4` mask from a per-block
/// "has any non-zero AC coefficient" array (raster-Z order, length 16).
///
/// For Intra_16x16 with `cbp_luma == 15`, callers should pass the per-
/// block flag derived from the actual scan-order coefficients (not
/// blanket 0xFFFF) — even with `cbp_luma == 15` an individual 4x4 may
/// quantise to all-zero AC. The deblock walker checks each block
/// independently per §8.7.2.1.
#[inline]
pub fn luma_nz_mask_from_blocks(blk_has_nz: &[bool; 16]) -> u16 {
    let mut m: u16 = 0;
    for (i, &nz) in blk_has_nz.iter().enumerate() {
        if nz {
            m |= 1u16 << i;
        }
    }
    m
}

/// Convenience helper: chroma nonzero mask from per-4x4-block AC nz
/// arrays (Cb then Cr). Layout matches §8.7.2.1: low 4 bits = Cb,
/// next 4 bits at offset 8 = Cr.
#[inline]
pub fn chroma_nz_mask_from_blocks(cb_has_nz: &[bool; 4], cr_has_nz: &[bool; 4]) -> u16 {
    let mut m: u16 = 0;
    for (i, &nz) in cb_has_nz.iter().enumerate() {
        if nz {
            m |= 1u16 << i;
        }
    }
    for (i, &nz) in cr_has_nz.iter().enumerate() {
        if nz {
            m |= 1u16 << (8 + i);
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two flat-128 macroblocks in a row → no AC anywhere → bS=4 on the
    /// internal MB edge for intra/intra (always-bS=4 for MB-edge intra,
    /// even with no coefficients) → the strong filter activates only
    /// when `|p0-q0| < α` etc. Since both sides are flat 128, the
    /// filter copies samples unchanged.
    #[test]
    fn flat_picture_unchanged() {
        let mut y = vec![128u8; 32 * 16];
        let mut u = vec![128u8; 16 * 8];
        let mut v = vec![128u8; 16 * 8];
        let mb_infos = vec![
            MbDeblockInfo {
                is_intra: true,
                qp_y: 26,
                luma_nonzero_4x4: 0,
                chroma_nonzero_4x4: 0,
                ..Default::default()
            },
            MbDeblockInfo {
                is_intra: true,
                qp_y: 26,
                luma_nonzero_4x4: 0,
                chroma_nonzero_4x4: 0,
                ..Default::default()
            },
        ];
        deblock_recon(32, 16, 16, 8, &mut y, &mut u, &mut v, &mb_infos, 0, 2, 1);
        assert!(y.iter().all(|&v| v == 128));
        assert!(u.iter().all(|&v| v == 128));
        assert!(v.iter().all(|&v| v == 128));
    }

    /// Deliberate block-boundary discontinuity at x=16 (left MB = 124,
    /// right MB = 132, well within the bS=4 strong-filter activation
    /// band at QP=26 → α=38, β=10 per Table 8-16). Confirm the deblock
    /// pass smooths the boundary.
    #[test]
    fn block_boundary_discontinuity_smoothed() {
        let mut y = vec![124u8; 32 * 16];
        let u = vec![128u8; 16 * 8];
        let v = vec![128u8; 16 * 8];
        for j in 0..16 {
            for i in 16..32 {
                y[j * 32 + i] = 132;
            }
        }
        let mut yy = y.clone();
        let mut uu = u.clone();
        let mut vv = v.clone();
        let mb_infos = vec![
            MbDeblockInfo {
                is_intra: true,
                qp_y: 26,
                luma_nonzero_4x4: 0,
                chroma_nonzero_4x4: 0,
                ..Default::default()
            },
            MbDeblockInfo {
                is_intra: true,
                qp_y: 26,
                luma_nonzero_4x4: 0,
                chroma_nonzero_4x4: 0,
                ..Default::default()
            },
        ];
        deblock_recon(32, 16, 16, 8, &mut yy, &mut uu, &mut vv, &mb_infos, 0, 2, 1);
        // Edge samples should now be intermediate (not 124 / 132).
        // Sample at (15, 0) = p0 of vertical edge x=16.
        let p0 = yy[15];
        let q0 = yy[16];
        assert!(
            p0 != 124 || q0 != 132,
            "deblock did not modify the block-boundary discontinuity",
        );
        // Smoothing should pull p0 up toward q0 and q0 down toward p0.
        assert!(p0 > 124, "p0={p0} did not move up");
        assert!(q0 < 132, "q0={q0} did not move down");
    }

    #[test]
    fn luma_nz_mask_packs_correctly() {
        let mut nz = [false; 16];
        nz[0] = true;
        nz[5] = true;
        nz[15] = true;
        assert_eq!(
            luma_nz_mask_from_blocks(&nz),
            (1u16 << 0) | (1u16 << 5) | (1u16 << 15),
        );
    }

    #[test]
    fn chroma_nz_mask_packs_cb_then_cr() {
        let cb = [true, false, false, true];
        let cr = [false, true, false, false];
        assert_eq!(
            chroma_nz_mask_from_blocks(&cb, &cr),
            (1u16 << 0) | (1u16 << 3) | (1u16 << 9),
        );
    }
}
