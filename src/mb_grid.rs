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
#[derive(Debug, Clone, Copy, Default)]
pub struct MbInfo {
    /// §6.4.4 — true once this MB has been reconstructed and is
    /// available for use by subsequent MBs as a neighbour.
    pub available: bool,
    /// §7.4.5 — true for I_NxN / Intra_16x16 / I_PCM.
    pub is_intra: bool,
    /// §7.4.5 — true for I_PCM.
    pub is_i_pcm: bool,
    /// Raw mb_type value from the bitstream (for diagnostics).
    pub mb_type_raw: u32,
    /// §7.4.5 — QP_Y post-adjustment for this macroblock.
    pub qp_y: i32,
    /// CodedBlockPatternLuma — low 4 bits one-per-8x8-luma.
    pub cbp_luma: u8,
    /// CodedBlockPatternChroma — 0, 1, or 2 (per §7.4.5).
    pub cbp_chroma: u8,
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
        assert_eq!(
            g.neighbour_mb_addrs(11),
            [Some(10), Some(7), None, Some(6)]
        );
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
}
