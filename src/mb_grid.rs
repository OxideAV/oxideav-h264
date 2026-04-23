//! Per-macroblock metadata grid.
//!
//! Stores what's needed for neighbour-based derivations: mb_type, QP_Y
//! (for deblocking bS / chroma QP), intra pred modes (for §8.3.1.1 /
//! §8.3.2.1 mode prediction), coded_block_pattern (for deblock bS and
//! coded_block_flag ctxIdx), and a flag indicating "available for
//! decoding" per §6.4.4.
//!
//! Coordinate conventions follow §6.4.1:
//!   * MB address `n` maps to raster-scan position
//!     `x = n % PicWidthInMbs`, `y = n / PicWidthInMbs` when
//!     MbaffFrameFlag == 0 (the non-MBAFF frame-only path assumed by
//!     this reconstruction layer).
//!
//! Neighbour derivation (§6.4.9, non-MBAFF / non-constrained-intra-pred
//! case): A = left (addr-1, same row), B = above (addr - width),
//! C = above-right (addr - width + 1), D = above-left (addr - width - 1).
//!
//! Clean-room: derived only from ITU-T Rec. H.264 (08/2024).

/// Per-macroblock metadata retained in the grid.
///
/// Inter-prediction MVs / ref_idx are kept so §8.4.1 MVpred in a
/// subsequent MB can consult the 4x4-block MV of the current MB via
/// neighbour lookups (see [`crate::mv_deriv::neighbour_4x4_map`]).
#[derive(Debug, Clone, Copy)]
pub struct MbInfo {
    /// §6.4.4 — true once this MB has been reconstructed and is
    /// available for use by subsequent MBs as a neighbour.
    pub available: bool,
    /// §7.4.5 — true for I_NxN / Intra_16x16 / I_PCM.
    pub is_intra: bool,
    /// §7.4.5 — true for I_PCM.
    pub is_i_pcm: bool,
    /// §7.4.5 — true when this MB's prediction is Intra_4x4 or
    /// Intra_8x8 (i.e. `MbPartPredMode == Intra_4x4/8x8`, aka the
    /// I_NxN mnemonic). Required by §8.3.1.1 step 3 to distinguish
    /// I_NxN from Intra_16x16 / I_PCM neighbours when consulting
    /// `intra_4x4_pred_modes` / `intra_8x8_pred_modes`. The raw
    /// `mb_type_raw` value alone is ambiguous across slice types
    /// (e.g. raw = 5 is Intra_16x16 in I slices but I_NxN after the
    /// P-slice remap).
    pub is_intra_nxn: bool,
    /// Raw mb_type value from the bitstream (for diagnostics).
    pub mb_type_raw: u32,
    /// §7.4.5 — QP_Y post-adjustment for this macroblock.
    pub qp_y: i32,
    /// CodedBlockPatternLuma — low 4 bits one-per-8x8-luma.
    pub cbp_luma: u8,
    /// CodedBlockPatternChroma — 0, 1, or 2 (per §7.4.5).
    pub cbp_chroma: u8,
    /// §8.7.2.1 — per-4x4-luma-block nonzero-coefficient bitmap, indexed
    /// by §6.4.3 Figure 6-10 Z-scan (bit `n` = 4x4 block at Z-scan
    /// position `n`). Used by the deblock boundary-strength derivation:
    /// the third bullet of §8.7.2.1 asks whether either adjacent 4x4
    /// block carries non-zero transform coefficients, which is a
    /// stricter test than the MB-level `cbp_luma != 0`.
    ///
    /// For Intra_16x16 MBs the DC block always has coefficients by
    /// construction; `cbp_luma != 0` implies AC blocks have coefficients
    /// too; for I_PCM the flag is treated as "all blocks have coeffs"
    /// by the deblock walker.
    pub luma_nonzero_4x4: u16,
    /// §8.7.2.1 per-4x4-chroma-block nonzero bitmap. Layout mirrors
    /// the (plane, blk4) pair: low 8 bits = Cb (2 per chroma MB row),
    /// next 8 = Cr. For 4:2:0 only bits 0..=3 and 8..=11 are populated
    /// (four 4x4 chroma blocks per plane); for 4:2:2 bits 0..=7 and
    /// 8..=15.
    pub chroma_nonzero_4x4: u16,
    /// §8.3.1.1 / §8.3.2.1 — Intra_4x4 / Intra_8x8 prediction modes per
    /// 4x4 (or 8x8) block within the MB. For 4x4 we fill 16 entries in
    /// raster order; for 8x8 we fill the first 4 entries of the 8x8
    /// array.
    pub intra_4x4_pred_modes: [u8; 16],
    pub intra_8x8_pred_modes: [u8; 4],
    /// §7.4.5.1 — intra_chroma_pred_mode (0..=3).
    pub intra_chroma_pred_mode: u8,
    /// §7.3.5 — whether this MB uses 8x8 transform.
    pub transform_size_8x8_flag: bool,
    /// §8.4.1 — per-4x4-block list-0 MV, one entry per 4x4 block in
    /// raster order (Figure 6-10). Components are in 1/4-pel units.
    /// `(0, 0)` when this MB did not contribute an L0 MV at that block.
    pub mv_l0: [(i16, i16); 16],
    /// §8.4.1 — per-4x4-block list-1 MV.
    pub mv_l1: [(i16, i16); 16],
    /// §7.4.5.1 / §8.4.1 — per-8x8-partition list-0 reference index.
    /// Sentinel `-1` means "L0 not used" for that partition.
    pub ref_idx_l0: [i8; 4],
    /// Per-8x8-partition list-1 ref_idx. `-1` means "L1 not used".
    pub ref_idx_l1: [i8; 4],
    /// §6.4.8 third bullet — slice identity used to mark a neighbour as
    /// "not available" when it belongs to a different slice than the
    /// current macroblock. Monotonically assigned within a primary coded
    /// picture (0, 1, 2, …) — the raw
    /// `first_mb_in_slice` can't be used directly because it doesn't
    /// increase monotonically across different slice groups.
    ///
    /// `-1` sentinel means "unset" (MB not yet decoded in this picture).
    pub slice_id: i32,
}

impl Default for MbInfo {
    fn default() -> Self {
        Self {
            available: false,
            is_intra: false,
            is_i_pcm: false,
            is_intra_nxn: false,
            mb_type_raw: 0,
            qp_y: 0,
            cbp_luma: 0,
            cbp_chroma: 0,
            luma_nonzero_4x4: 0,
            chroma_nonzero_4x4: 0,
            intra_4x4_pred_modes: [0; 16],
            intra_8x8_pred_modes: [0; 4],
            intra_chroma_pred_mode: 0,
            transform_size_8x8_flag: false,
            mv_l0: [(0, 0); 16],
            mv_l1: [(0, 0); 16],
            // §8.4.1.3.2 — "not used" => refIdx = -1.
            ref_idx_l0: [-1; 4],
            ref_idx_l1: [-1; 4],
            // §6.4.8 — unset until the MB is reconstructed (prevents
            // pre-decode MBs from being treated as in-slice neighbours).
            slice_id: -1,
        }
    }
}

impl MbInfo {
    /// §6.4.3 / Figure 6-10 — given a 4x4 block index (0..=15), return
    /// which 8x8 quadrant (0..=3) it belongs to.
    #[inline]
    pub const fn blk8_of_blk4(blk4: usize) -> usize {
        // Blocks 0..=3 -> quadrant 0; 4..=7 -> 1; 8..=11 -> 2; 12..=15 -> 3.
        blk4 / 4
    }

    /// Set list-0 MV for the 4x4 block at `blk4` (0..=15, raster order).
    #[inline]
    pub fn set_mv_l0(&mut self, blk4: usize, mv: (i16, i16)) {
        self.mv_l0[blk4] = mv;
    }

    /// Set list-1 MV for the 4x4 block at `blk4` (0..=15, raster order).
    #[inline]
    pub fn set_mv_l1(&mut self, blk4: usize, mv: (i16, i16)) {
        self.mv_l1[blk4] = mv;
    }

    /// Get list-0 MV for the 4x4 block at `blk4`.
    #[inline]
    pub fn mv_l0_at(&self, blk4: usize) -> (i16, i16) {
        self.mv_l0[blk4]
    }

    /// Get list-1 MV for the 4x4 block at `blk4`.
    #[inline]
    pub fn mv_l1_at(&self, blk4: usize) -> (i16, i16) {
        self.mv_l1[blk4]
    }
}

/// MB metadata grid. Allocated once per picture, indexed by
/// macroblock address (§7.4.2.1 MbAddr).
#[derive(Debug, Clone)]
pub struct MbGrid {
    pub width_in_mbs: u32,
    pub height_in_mbs: u32,
    pub info: Vec<MbInfo>,
}

impl MbGrid {
    /// Create a fresh all-unavailable grid sized for the given MB
    /// dimensions. MbAddr 0..(w*h) is valid.
    pub fn new(width_in_mbs: u32, height_in_mbs: u32) -> Self {
        let len = (width_in_mbs as usize) * (height_in_mbs as usize);
        Self {
            width_in_mbs,
            height_in_mbs,
            info: vec![MbInfo::default(); len],
        }
    }

    /// Immutable access by address, bounds-checked.
    pub fn get(&self, mb_addr: u32) -> Option<&MbInfo> {
        self.info.get(mb_addr as usize)
    }

    /// Mutable access by address, bounds-checked.
    pub fn get_mut(&mut self, mb_addr: u32) -> Option<&mut MbInfo> {
        self.info.get_mut(mb_addr as usize)
    }

    /// §6.4.1 — macroblock raster coordinates
    /// `(x = addr % width, y = addr / width)`.
    pub fn mb_xy(&self, mb_addr: u32) -> (u32, u32) {
        if self.width_in_mbs == 0 {
            return (0, 0);
        }
        (mb_addr % self.width_in_mbs, mb_addr / self.width_in_mbs)
    }

    /// §6.4.9 — neighbour macroblock addresses
    /// `[A (left), B (above), C (above-right), D (above-left)]`.
    ///
    /// Each entry is `Some(addr)` if that neighbour lies inside the
    /// picture frame, otherwise `None`. This is the frame-only /
    /// non-MBAFF path; MBAFF pair derivation (§6.4.10) is not handled.
    ///
    /// Note: "availability for intra prediction" (§6.4.11) further
    /// requires `mb_grid.info[addr].available == true` and — with
    /// `constrained_intra_pred_flag` — the neighbour being intra. The
    /// caller layers those checks on top.
    pub fn neighbour_mb_addrs(&self, mb_addr: u32) -> [Option<u32>; 4] {
        let (x, y) = self.mb_xy(mb_addr);
        let w = self.width_in_mbs;
        let h = self.height_in_mbs;
        if x >= w || y >= h {
            return [None; 4];
        }
        // A — left neighbour.
        let a = if x > 0 { Some(mb_addr - 1) } else { None };
        // B — above neighbour.
        let b = if y > 0 { Some(mb_addr - w) } else { None };
        // C — above-right neighbour.
        let c = if y > 0 && x + 1 < w {
            Some(mb_addr - w + 1)
        } else {
            None
        };
        // D — above-left neighbour.
        let d = if y > 0 && x > 0 {
            Some(mb_addr - w - 1)
        } else {
            None
        };
        [a, b, c, d]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_allocates_correct_count() {
        let g = MbGrid::new(4, 3);
        assert_eq!(g.info.len(), 12);
        assert!(g.info.iter().all(|e| !e.available));
    }

    #[test]
    fn mb_xy_raster_order() {
        let g = MbGrid::new(4, 4);
        assert_eq!(g.mb_xy(0), (0, 0));
        assert_eq!(g.mb_xy(3), (3, 0));
        assert_eq!(g.mb_xy(4), (0, 1));
        assert_eq!(g.mb_xy(15), (3, 3));
    }

    #[test]
    fn neighbours_top_left_has_none() {
        let g = MbGrid::new(4, 4);
        assert_eq!(g.neighbour_mb_addrs(0), [None, None, None, None]);
    }

    #[test]
    fn neighbours_top_right_only_left() {
        let g = MbGrid::new(4, 4);
        // Top-right corner = addr 3.
        assert_eq!(g.neighbour_mb_addrs(3), [Some(2), None, None, None]);
    }

    #[test]
    fn neighbours_top_row_middle_has_left_only() {
        let g = MbGrid::new(4, 4);
        // Position (2, 0), addr 2. Left=1, above=None, above-right=None, above-left=None.
        assert_eq!(g.neighbour_mb_addrs(2), [Some(1), None, None, None]);
    }

    #[test]
    fn neighbours_bottom_left_has_above() {
        let g = MbGrid::new(4, 4);
        // Position (0, 3), addr 12. Left=None, above=8, above-right=9, above-left=None.
        assert_eq!(g.neighbour_mb_addrs(12), [None, Some(8), Some(9), None]);
    }

    #[test]
    fn neighbours_interior_has_all_four() {
        let g = MbGrid::new(4, 4);
        // Position (2, 2), addr 10.
        // Left=9, above=6, above-right=7, above-left=5.
        assert_eq!(
            g.neighbour_mb_addrs(10),
            [Some(9), Some(6), Some(7), Some(5)]
        );
    }

    #[test]
    fn neighbours_right_edge_no_above_right() {
        let g = MbGrid::new(4, 4);
        // Position (3, 2), addr 11.
        // Left=10, above=7, above-right=None (x+1 out of range), above-left=6.
        assert_eq!(g.neighbour_mb_addrs(11), [Some(10), Some(7), None, Some(6)]);
    }

    #[test]
    fn set_info_via_mut() {
        let mut g = MbGrid::new(2, 2);
        {
            let info = g.get_mut(0).unwrap();
            info.available = true;
            info.qp_y = 26;
            info.is_intra = true;
        }
        let info = g.get(0).unwrap();
        assert!(info.available);
        assert_eq!(info.qp_y, 26);
        assert!(info.is_intra);
    }

    #[test]
    fn get_out_of_range_is_none() {
        let g = MbGrid::new(2, 2);
        assert!(g.get(4).is_none());
        assert!(g.get(100).is_none());
    }

    #[test]
    fn mb_info_default_mv_and_ref_idx_sentinels() {
        // §8.4.1.3.2 — a freshly allocated MbInfo must report "L0/L1
        // not used" (ref_idx = -1) and zero MVs so the neighbour-based
        // MVpred path treats it as unavailable.
        let info = MbInfo::default();
        assert_eq!(info.mv_l0, [(0i16, 0i16); 16]);
        assert_eq!(info.mv_l1, [(0i16, 0i16); 16]);
        assert_eq!(info.ref_idx_l0, [-1i8; 4]);
        assert_eq!(info.ref_idx_l1, [-1i8; 4]);
    }

    #[test]
    fn mb_info_set_and_get_mv_per_4x4_block() {
        let mut info = MbInfo::default();
        info.set_mv_l0(3, (10, -20));
        info.set_mv_l1(5, (-3, 7));
        assert_eq!(info.mv_l0_at(3), (10, -20));
        assert_eq!(info.mv_l1_at(5), (-3, 7));
        // Untouched entries still zero.
        assert_eq!(info.mv_l0_at(0), (0, 0));
        assert_eq!(info.mv_l1_at(15), (0, 0));
    }

    #[test]
    fn mb_info_blk8_of_blk4_mapping() {
        // Figure 6-10 — blocks 0..=3 are in the top-left 8x8 quadrant.
        assert_eq!(MbInfo::blk8_of_blk4(0), 0);
        assert_eq!(MbInfo::blk8_of_blk4(3), 0);
        assert_eq!(MbInfo::blk8_of_blk4(4), 1);
        assert_eq!(MbInfo::blk8_of_blk4(7), 1);
        assert_eq!(MbInfo::blk8_of_blk4(8), 2);
        assert_eq!(MbInfo::blk8_of_blk4(11), 2);
        assert_eq!(MbInfo::blk8_of_blk4(12), 3);
        assert_eq!(MbInfo::blk8_of_blk4(15), 3);
    }
}
