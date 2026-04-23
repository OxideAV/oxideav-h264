//! §8.7 — Deblocking filter.
//!
//! Pure per-edge filter functions per ITU-T Rec. H.264 (08/2024).
//! The caller loops over the 4x4 block grid and invokes these
//! functions for each edge; this module does not know about macroblock
//! geometry. Tables 8-16 (α'/β') and 8-17 (t'C0) are transcribed
//! verbatim from the spec.
//!
//! Clean-room: derived **only** from the ITU-T specification PDF,
//! with no reference to other decoder implementations.
//!
//! Note on table numbering: the task brief mentions Tables 8-16, 8-17,
//! and 8-18. In the 2024-08 edition of the spec (V15), α' and β' are
//! both in Table 8-16, and t'C0 is in Table 8-17. The equivalent of
//! "Table 8-18" from earlier editions has been merged into 8-17. We
//! follow the actual spec numbering here.

#![allow(dead_code)]

// --------------------------------------------------------------------
// Public types
// --------------------------------------------------------------------

/// Inputs to filter one edge of 4 pixels across the boundary.
///
/// Layout, 1D along the axis perpendicular to the edge:
///   p3 p2 p1 p0 | q0 q1 q2 q3
///
/// Where `|` is the edge being filtered. p-side is at lower coordinate
/// (above for horizontal edge, left for vertical edge); q-side is at
/// higher coordinate. Per §8.7.2 Figure 8-11.
///
/// Filter may modify p0..p2 and q0..q2 in place. p3/q3 are read-only
/// (used only as inputs to §8.7.2.4 eq. 8-479 / 8-486 strong-filter
/// updates of p2/q2).
#[derive(Debug)]
pub struct EdgeSamples<'a> {
    pub p3: i32,
    pub p2: &'a mut i32,
    pub p1: &'a mut i32,
    pub p0: &'a mut i32,
    pub q0: &'a mut i32,
    pub q1: &'a mut i32,
    pub q2: &'a mut i32,
    pub q3: i32,
}

/// Chroma vs luma selector — they use different filter variants per
/// §8.7.2.2 (the `chromaStyleFilteringFlag` of eq. 8-450 only turns on
/// for chroma when `ChromaArrayType != 3`, so chroma here means
/// 4:2:0 / 4:2:2; for 4:4:4 chroma the luma path is used by the
/// caller).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Plane {
    Luma,
    Chroma,
}

/// Filter strength / context per §8.7.2.1 and §8.7.2.2.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FilterParams {
    /// `bS` ∈ 0..=4 per §8.7.2.1.
    pub bs: u8,
    /// `qP_av = (qP_p + qP_q + 1) >> 1` per §8.7.2.2 eq. 8-453.
    /// For chroma, caller must pass the chroma-mapped QP per §8.5.8
    /// (Table 8-15).
    pub qp_avg: i32,
    /// `filterOffsetA` = `slice_alpha_c0_offset_div2 * 2`, from the
    /// slice header (§7.4.3).
    pub filter_offset_a: i32,
    /// `filterOffsetB` = `slice_beta_offset_div2 * 2`, from the slice
    /// header (§7.4.3).
    pub filter_offset_b: i32,
    /// Bit depth of the sample array being filtered
    /// (`BitDepth_Y` for luma, `BitDepth_C` for chroma). Typically 8.
    pub bit_depth: u32,
}

/// §8.7.2.1 inputs distilled from the macroblock context by the caller.
///
/// The caller (macroblock layer) is responsible for mapping MB-level
/// decisions (intra/inter, slice type, transform block coefficient
/// statuses, motion vectors, reference indices) into these booleans.
#[derive(Debug, Copy, Clone, Default)]
pub struct BsInputs {
    /// True if the macroblock containing p0 is intra-coded.
    pub p_is_intra: bool,
    /// True if the macroblock containing q0 is intra-coded.
    pub q_is_intra: bool,
    /// True if the edge is also a macroblock boundary (required for
    /// bS=4 on intra/SP-SI edges per §8.7.2.1).
    pub is_mb_edge: bool,
    /// True if the slice containing p0 or q0 is SP or SI
    /// (bS=4 / bS=3 path applies for those slices per §8.7.2.1).
    pub is_sp_or_si: bool,
    /// True if either side's transform block (4x4 or 8x8 depending on
    /// `transform_size_8x8_flag`) contains any non-zero transform
    /// coefficient level. Used for bS=2 per §8.7.2.1.
    pub either_has_nonzero_coeffs: bool,
    /// True if the partitions of p and q use different reference
    /// pictures, different numbers of motion vectors, or motion
    /// vectors whose components differ by >=4 in quarter-sample
    /// units. Collapses the full bS=1 test suite from §8.7.2.1.
    pub different_ref_or_mv: bool,
    /// True when §8.7.2.1 `mixedModeEdgeFlag` is set — samples are in
    /// different macroblock pairs, one of which is field and the
    /// other frame. Forces bS=1 when nothing stronger fires.
    pub mixed_mode_edge: bool,
    /// `verticalEdgeFlag` from §8.7.2.1 — bS=4 requires either a
    /// frame-mb MB edge OR `verticalEdgeFlag == 1` (with MbaffFrameFlag
    /// / field_pic_flag set); we fold that into `is_mb_edge` for the
    /// non-MBAFF case but keep this for the MBAFF/field path.
    pub vertical_edge: bool,
    /// True if `MbaffFrameFlag == 1` or `field_pic_flag == 1`, which
    /// is the "allow bS=4 only on vertical edges" gating condition
    /// per §8.7.2.1 (second bS=4 bullet).
    pub mbaff_or_field: bool,
}

// --------------------------------------------------------------------
// Tables transcribed verbatim from the spec
// --------------------------------------------------------------------

/// Table 8-16 — α' as a function of index (0..=51). Spec page 210.
/// Verbatim transcription; reference values are the 8-bit depth case.
const ALPHA_PRIME: [u8; 52] = [
    // indexA 0..=15
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // indexA 16..=25
    4, 4, 5, 6, 7, 8, 9, 10, 12, 13, // indexA 26..=35
    15, 17, 20, 22, 25, 28, 32, 36, 40, 45, // indexA 36..=45
    50, 56, 63, 71, 80, 90, 101, 113, 127, 144, // indexA 46..=51
    162, 182, 203, 226, 255, 255,
];

/// Table 8-16 — β' as a function of index (0..=51). Spec page 210.
const BETA_PRIME: [u8; 52] = [
    // indexB 0..=15
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // indexB 16..=25
    2, 2, 2, 3, 3, 3, 3, 4, 4, 4, // indexB 26..=35
    6, 6, 7, 7, 8, 8, 9, 9, 10, 10, // indexB 36..=45
    11, 11, 12, 12, 13, 13, 14, 14, 15, 15, // indexB 46..=51
    16, 16, 17, 17, 18, 18,
];

/// Table 8-17 — t'C0 as a function of indexA (0..=51) and bS (1..=3).
/// Spec page 210. Verbatim transcription from the 08/2024 edition;
/// 8-bit depth reference. Row layout: [bs=1, bs=2, bs=3] per indexA.
///
/// Table 8-17 main part (indexA 0..=25):
///   indexA:  0 1 2 3 4 5 6 7 8 9 10..22 23 24 25
///   bS=1:    0 0 0 0 0 0 0 0 0 0  0      1  1  1
///   bS=2:    0 0 0 0 0 0 0 0 0 0  0      0  1  1
///   bS=3:    0 0 0 0 0 0 0 0 0 0  0      0  1  1
///
/// Table 8-17 (concluded) (indexA 26..=51):
///   indexA: 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51
///   bS=1:    1  1  1  1  1  1  1  2  2  2  2  3  3  3  4  4  4  5  6  6  7  8  9 10 11 13
///   bS=2:    1  1  1  1  1  2  2  2  2  3  3  3  4  4  5  5  6  7  8  8 10 11 12 13 15 17
///   bS=3:    1  2  2  2  2  3  3  3  4  4  4  5  6  6  7  8  9 10 11 13 14 16 18 20 23 25
const TC0_PRIME: [[u8; 3]; 52] = [
    // indexA 0..=9 — all zeros.
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    // indexA 10..=16 — all zeros.
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    // indexA 17..=20: bS=3 goes to 1; bS=1/bS=2 stay 0.
    [0, 0, 1],   // 17
    [0, 0, 1],   // 18
    [0, 0, 1],   // 19
    [0, 0, 1],   // 20
    // indexA 21..=22: bS=2 also goes to 1.
    [0, 1, 1],   // 21
    [0, 1, 1],   // 22
    // indexA 23..=25: bS=1 also goes to 1 (all three = 1).
    [1, 1, 1],   // 23
    [1, 1, 1],   // 24
    [1, 1, 1],   // 25
    // indexA 26..=29.
    [1, 1, 1],   // 26
    [1, 1, 2],   // 27
    [1, 1, 2],   // 28
    [1, 1, 2],   // 29
    // indexA 30..=32.
    [1, 1, 2],   // 30
    [1, 2, 3],   // 31
    [1, 2, 3],   // 32
    // indexA 33..=35.
    [2, 2, 3],   // 33
    [2, 2, 4],   // 34
    [2, 3, 4],   // 35
    // indexA 36..=38.
    [2, 3, 4],   // 36
    [3, 3, 5],   // 37
    [3, 4, 6],   // 38
    // indexA 39..=41.
    [3, 4, 6],   // 39
    [4, 5, 7],   // 40
    [4, 5, 8],   // 41
    // indexA 42..=44.
    [4, 6, 9],   // 42
    [5, 7, 10],  // 43
    [6, 8, 11],  // 44
    // indexA 45..=47.
    [6, 8, 13],  // 45
    [7, 10, 14], // 46
    [8, 11, 16], // 47
    // indexA 48..=51.
    [9, 12, 18], // 48
    [10, 13, 20],// 49
    [11, 15, 23],// 50
    [13, 17, 25],// 51
];

// --------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------

/// `Clip3(x, y, z) = min(y, max(x, z))` per §5.7.
#[inline]
fn clip3(x: i32, y: i32, z: i32) -> i32 {
    if z < x {
        x
    } else if z > y {
        y
    } else {
        z
    }
}

/// Sample-clip to `[0, (1 << bit_depth) - 1]` — the Clip1_Y / Clip1_C
/// function of §5.7, applied after filtering.
#[inline]
fn clip1(v: i32, bit_depth: u32) -> i32 {
    let hi = (1i32 << bit_depth) - 1;
    clip3(0, hi, v)
}

// --------------------------------------------------------------------
// Public table accessors
// --------------------------------------------------------------------

/// Table 8-16 — α indexed by indexA ∈ 0..=51. Scales with bit depth
/// per §8.7.2.2 eq. 8-456 / 8-458: `α = α' * (1 << (BitDepth - 8))`.
pub fn alpha_from_index(index_a: i32, bit_depth: u32) -> i32 {
    let idx = clip3(0, 51, index_a) as usize;
    let prime = ALPHA_PRIME[idx] as i32;
    // §8.7.2.2 — eq. 8-456 (luma) / 8-458 (chroma).
    prime << bit_depth.saturating_sub(8)
}

/// Table 8-16 — β indexed by indexB ∈ 0..=51. Scales with bit depth
/// per §8.7.2.2 eq. 8-457 / 8-459: `β = β' * (1 << (BitDepth - 8))`.
pub fn beta_from_index(index_b: i32, bit_depth: u32) -> i32 {
    let idx = clip3(0, 51, index_b) as usize;
    let prime = BETA_PRIME[idx] as i32;
    // §8.7.2.2 — eq. 8-457 (luma) / 8-459 (chroma).
    prime << bit_depth.saturating_sub(8)
}

/// Table 8-17 — t'C0(bS, indexA), scaled to t_C0 per §8.7.2.3
/// eq. 8-461 / 8-462: `t_C0 = t'C0 * (1 << (BitDepth - 8))`.
///
/// Defined for bS ∈ 1..=3. Returns 0 for out-of-range bS (bS=0 has no
/// filtering at all, and bS=4 does not use this table — see §8.7.2.4).
pub fn tc0_from(bs: u8, index_a: i32, bit_depth: u32) -> i32 {
    if !(1..=3).contains(&bs) {
        return 0;
    }
    let idx = clip3(0, 51, index_a) as usize;
    let prime = TC0_PRIME[idx][(bs - 1) as usize] as i32;
    prime << bit_depth.saturating_sub(8)
}

// --------------------------------------------------------------------
// §8.7.2.1 — boundary strength
// --------------------------------------------------------------------

/// §8.7.2.1 — derive `bS` in `0..=4` from the abstract MB-level facts
/// the caller gathers. The spec's case ordering (bS=4 strongest down
/// to bS=0) is preserved here.
pub fn derive_boundary_strength(inputs: BsInputs) -> u8 {
    // bS = 4: MB-edge intra or SP/SI (first two bullets of §8.7.2.1),
    // with caveats for MBAFF/field that we fold into a single flag.
    if inputs.is_mb_edge && !inputs.mixed_mode_edge {
        // Frame-MB MB-edge intra or SP/SI.
        if !inputs.mbaff_or_field && (inputs.p_is_intra || inputs.q_is_intra || inputs.is_sp_or_si)
        {
            return 4;
        }
        // MBAFF/field: bS=4 requires verticalEdgeFlag == 1
        // (3rd/4th bullets of §8.7.2.1).
        if inputs.mbaff_or_field
            && inputs.vertical_edge
            && (inputs.p_is_intra || inputs.q_is_intra || inputs.is_sp_or_si)
        {
            return 4;
        }
    }

    // bS = 3: non-mixed-mode intra, or mixed-mode-and-horizontal intra,
    // or SP/SI counterparts (§8.7.2.1 second block).
    let intra_or_spsi = inputs.p_is_intra || inputs.q_is_intra || inputs.is_sp_or_si;
    if intra_or_spsi {
        if !inputs.mixed_mode_edge {
            return 3;
        }
        // mixed-mode + horizontal edge = bS=3 (verticalEdgeFlag==0).
        if inputs.mixed_mode_edge && !inputs.vertical_edge {
            return 3;
        }
    }

    // bS = 2: edge has a transform block with non-zero coefficients on
    // at least one side (§8.7.2.1 third block).
    if inputs.either_has_nonzero_coeffs {
        return 2;
    }

    // bS = 1: mixedModeEdgeFlag forces it, or the "different refs /
    // different MVs / MV delta >= 4 qpel" collapse (§8.7.2.1 fourth
    // block).
    if inputs.mixed_mode_edge || inputs.different_ref_or_mv {
        return 1;
    }

    // bS = 0: no filtering (§8.7.2.1 final bullet).
    0
}

// --------------------------------------------------------------------
// §8.7.2.2 — top-level per-edge filter
// --------------------------------------------------------------------

/// §8.7.2.2 — filter one set of 8 samples (p3..p0, q0..q3) across a
/// single 4x4 block edge. When `bS == 0`, no filtering is performed.
///
/// `plane = Luma` means `chromaEdgeFlag == 0`.
/// `plane = Chroma` means `chromaEdgeFlag == 1` AND the caller has
/// decided `ChromaArrayType != 3` (so `chromaStyleFilteringFlag == 1`
/// per eq. 8-450). For 4:4:4 chroma, callers should pass `Luma`.
pub fn filter_edge(plane: Plane, mut samples: EdgeSamples<'_>, params: FilterParams) {
    // §8.7.2.2 — early-out on bS == 0.
    if params.bs == 0 {
        return;
    }

    // §8.7.2.2 — eq. 8-454 / 8-455.
    let index_a = clip3(0, 51, params.qp_avg + params.filter_offset_a);
    let index_b = clip3(0, 51, params.qp_avg + params.filter_offset_b);

    // §8.7.2.2 — eq. 8-456..8-459.
    let alpha = alpha_from_index(index_a, params.bit_depth);
    let beta = beta_from_index(index_b, params.bit_depth);

    // §8.7.2.2 — eq. 8-460. filterSamplesFlag.
    let p0 = *samples.p0;
    let q0 = *samples.q0;
    let p1 = *samples.p1;
    let q1 = *samples.q1;
    let filter_samples = params.bs != 0
        && (p0 - q0).abs() < alpha
        && (p1 - p0).abs() < beta
        && (q1 - q0).abs() < beta;

    if !filter_samples {
        // eq. 8-451 / 8-452 — identity, samples already unchanged.
        return;
    }

    let chroma_style = matches!(plane, Plane::Chroma);

    if params.bs < 4 {
        // §8.7.2.3 — normal filter.
        filter_normal(&mut samples, params, index_a, beta, chroma_style);
    } else {
        // §8.7.2.4 — strong filter (bS == 4).
        filter_strong(&mut samples, alpha, beta, chroma_style);
    }
}

// --------------------------------------------------------------------
// §8.7.2.3 — filter for bS < 4
// --------------------------------------------------------------------

/// §8.7.2.3 — filter for edges with bS < 4. Updates `p0/p1/q0/q1`
/// (luma); only `p0/q0` for chroma-style (§8.7.2.3 fall-through path,
/// eqs. 8-471 / 8-473).
fn filter_normal(
    samples: &mut EdgeSamples<'_>,
    params: FilterParams,
    index_a: i32,
    beta: i32,
    chroma_style: bool,
) {
    let p2 = *samples.p2;
    let p1 = *samples.p1;
    let p0 = *samples.p0;
    let q0 = *samples.q0;
    let q1 = *samples.q1;
    let q2 = *samples.q2;

    // §8.7.2.3 — eq. 8-461 / 8-462 (bit-depth scaling folded into
    // `tc0_from`).
    let tc0 = tc0_from(params.bs, index_a, params.bit_depth);

    // §8.7.2.3 — eq. 8-463 / 8-464.
    let a_p = (p2 - p0).abs();
    let a_q = (q2 - q0).abs();

    // §8.7.2.3 — eq. 8-465 / 8-466. tC.
    let tc = if !chroma_style {
        // luma: eq. 8-465.
        tc0 + (if a_p < beta { 1 } else { 0 }) + (if a_q < beta { 1 } else { 0 })
    } else {
        // chroma: eq. 8-466.
        tc0 + 1
    };

    // §8.7.2.3 — eq. 8-467 / 8-468 / 8-469. Filtered p0/q0.
    let delta = clip3(-tc, tc, (((q0 - p0) << 2) + (p1 - q1) + 4) >> 3);
    *samples.p0 = clip1(p0 + delta, params.bit_depth);
    *samples.q0 = clip1(q0 - delta, params.bit_depth);

    // §8.7.2.3 — eq. 8-470 / 8-471. Filtered p1 (luma only / !chroma).
    if !chroma_style && a_p < beta {
        // eq. 8-470.
        let step = clip3(-tc0, tc0, (p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1);
        *samples.p1 = p1 + step;
    }
    // else: eq. 8-471 — p1 unchanged (identity).

    // §8.7.2.3 — eq. 8-472 / 8-473. Filtered q1 (luma only / !chroma).
    if !chroma_style && a_q < beta {
        // eq. 8-472.
        let step = clip3(-tc0, tc0, (q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1);
        *samples.q1 = q1 + step;
    }
    // else: eq. 8-473 — q1 unchanged.

    // §8.7.2.3 — eq. 8-474 / 8-475. p2 and q2 always unchanged on the
    // normal-filter path.
    let _ = (p2, q2);
}

// --------------------------------------------------------------------
// §8.7.2.4 — strong filter for bS == 4
// --------------------------------------------------------------------

/// §8.7.2.4 — strong filter used only for `bS == 4`. Luma can update
/// p0/p1/p2 and q0/q1/q2 (with `a_p < β` / `a_q < β` and the
/// `|p0-q0| < (α>>2)+2` refinement condition controlling the strong
/// path). Chroma-style updates only p0/q0.
fn filter_strong(samples: &mut EdgeSamples<'_>, alpha: i32, beta: i32, chroma_style: bool) {
    let p3 = samples.p3;
    let p2 = *samples.p2;
    let p1 = *samples.p1;
    let p0 = *samples.p0;
    let q0 = *samples.q0;
    let q1 = *samples.q1;
    let q2 = *samples.q2;
    let q3 = samples.q3;

    // §8.7.2.3 — eq. 8-463 / 8-464 (reused in 8.7.2.4).
    let a_p = (p2 - p0).abs();
    let a_q = (q2 - q0).abs();

    // §8.7.2.4 — eq. 8-476 (strong-filter p-side condition).
    let strong_p = !chroma_style && a_p < beta && (p0 - q0).abs() < ((alpha >> 2) + 2);

    if strong_p {
        // eqs. 8-477, 8-478, 8-479.
        *samples.p0 = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3;
        *samples.p1 = (p2 + p1 + p0 + q0 + 2) >> 2;
        *samples.p2 = (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3;
    } else {
        // eqs. 8-480, 8-481, 8-482 (also the chroma-style branch).
        *samples.p0 = (2 * p1 + p0 + q1 + 2) >> 2;
        // p1, p2 unchanged.
        let _ = (p1, p2);
    }

    // §8.7.2.4 — eq. 8-483 (strong-filter q-side condition).
    let strong_q = !chroma_style && a_q < beta && (p0 - q0).abs() < ((alpha >> 2) + 2);

    if strong_q {
        // eqs. 8-484, 8-485, 8-486.
        *samples.q0 = (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3;
        *samples.q1 = (p0 + q0 + q1 + q2 + 2) >> 2;
        *samples.q2 = (2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3;
    } else {
        // eqs. 8-487, 8-488, 8-489 (also the chroma-style branch).
        *samples.q0 = (2 * q1 + q0 + p1 + 2) >> 2;
        // q1, q2 unchanged.
        let _ = (q1, q2);
    }
}

// --------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Small helper to build an EdgeSamples from 8 values + produce the
    // filtered output. We keep each value in a stack array so &mut is
    // safe.
    fn run_filter(
        plane: Plane,
        samples: [i32; 8], // p3, p2, p1, p0, q0, q1, q2, q3
        params: FilterParams,
    ) -> [i32; 8] {
        let mut p2 = samples[1];
        let mut p1 = samples[2];
        let mut p0 = samples[3];
        let mut q0 = samples[4];
        let mut q1 = samples[5];
        let mut q2 = samples[6];
        filter_edge(
            plane,
            EdgeSamples {
                p3: samples[0],
                p2: &mut p2,
                p1: &mut p1,
                p0: &mut p0,
                q0: &mut q0,
                q1: &mut q1,
                q2: &mut q2,
                q3: samples[7],
            },
            params,
        );
        [samples[0], p2, p1, p0, q0, q1, q2, samples[7]]
    }

    // ---------------- Table spot-checks (§8.7.2.2 / Table 8-16,
    // §8.7.2.3 / Table 8-17). ----------------

    #[test]
    fn alpha_table_spotcheck() {
        // Values from Table 8-16.
        assert_eq!(alpha_from_index(0, 8), 0);
        assert_eq!(alpha_from_index(15, 8), 0);
        assert_eq!(alpha_from_index(16, 8), 4);
        assert_eq!(alpha_from_index(25, 8), 13);
        assert_eq!(alpha_from_index(36, 8), 50);
        assert_eq!(alpha_from_index(51, 8), 255);
    }

    #[test]
    fn beta_table_spotcheck() {
        // Values from Table 8-16.
        assert_eq!(beta_from_index(0, 8), 0);
        assert_eq!(beta_from_index(15, 8), 0);
        assert_eq!(beta_from_index(16, 8), 2);
        assert_eq!(beta_from_index(25, 8), 4);
        assert_eq!(beta_from_index(40, 8), 13);
        assert_eq!(beta_from_index(51, 8), 18);
    }

    #[test]
    fn tc0_table_spotcheck() {
        // Values from Table 8-17 (main + concluded), 08/2024 edition.
        // bs=1, indexA=0..22 → 0; bs=2, 0..20 → 0; bs=3, 0..16 → 0.
        assert_eq!(tc0_from(1, 0, 8), 0);
        assert_eq!(tc0_from(1, 22, 8), 0);
        // indexA=17: bs=3 becomes 1, bs=1/2 still 0.
        assert_eq!(tc0_from(1, 17, 8), 0);
        assert_eq!(tc0_from(2, 17, 8), 0);
        assert_eq!(tc0_from(3, 17, 8), 1);
        // indexA=21: bs=2 becomes 1.
        assert_eq!(tc0_from(1, 21, 8), 0);
        assert_eq!(tc0_from(2, 21, 8), 1);
        assert_eq!(tc0_from(3, 21, 8), 1);
        // indexA=23: bs=1 also becomes 1 — all three are 1.
        assert_eq!(tc0_from(1, 23, 8), 1);
        assert_eq!(tc0_from(2, 23, 8), 1);
        assert_eq!(tc0_from(3, 23, 8), 1);
        // indexA=24 → (1, 1, 1).
        assert_eq!(tc0_from(1, 24, 8), 1);
        assert_eq!(tc0_from(2, 24, 8), 1);
        assert_eq!(tc0_from(3, 24, 8), 1);
        // indexA=30 row: (1, 1, 2).
        assert_eq!(tc0_from(1, 30, 8), 1);
        assert_eq!(tc0_from(2, 30, 8), 1);
        assert_eq!(tc0_from(3, 30, 8), 2);
        // indexA=36 row: (2, 3, 4).
        assert_eq!(tc0_from(1, 36, 8), 2);
        assert_eq!(tc0_from(2, 36, 8), 3);
        assert_eq!(tc0_from(3, 36, 8), 4);
        // indexA=44 row: (6, 8, 11).
        assert_eq!(tc0_from(1, 44, 8), 6);
        assert_eq!(tc0_from(2, 44, 8), 8);
        assert_eq!(tc0_from(3, 44, 8), 11);
        // indexA=51 row: (13, 17, 25).
        assert_eq!(tc0_from(1, 51, 8), 13);
        assert_eq!(tc0_from(2, 51, 8), 17);
        assert_eq!(tc0_from(3, 51, 8), 25);
        // bs=0 / bs=4 → 0 (not in table).
        assert_eq!(tc0_from(0, 36, 8), 0);
        assert_eq!(tc0_from(4, 36, 8), 0);
    }

    #[test]
    fn alpha_bit_depth_scaling() {
        // eq. 8-456: α = α' * (1 << (BitDepth - 8)).
        assert_eq!(alpha_from_index(36, 10), 50 * 4);
        assert_eq!(beta_from_index(36, 10), 11 * 4);
        // Table 8-17: t'C0 at (bS=3, indexA=36) is 4. At 10-bit, t_C0 = 4 * 4.
        assert_eq!(tc0_from(3, 36, 10), 4 * 4);
    }

    #[test]
    fn tc0_table_every_entry_matches_spec() {
        // Full transcription check: Table 8-17 main (indexA 0..=25) and
        // concluded (indexA 26..=51). Each row-major.
        // bS=3 is zero for idx 0..=16, bS=2 zero for 0..=20, bS=1 zero for
        // 0..=22 (per the main-table spec ordering).
        for idx in 0..=16 {
            assert_eq!(tc0_from(1, idx, 8), 0, "bS=1 idxA={}", idx);
            assert_eq!(tc0_from(2, idx, 8), 0, "bS=2 idxA={}", idx);
            assert_eq!(tc0_from(3, idx, 8), 0, "bS=3 idxA={}", idx);
        }
        // (bS=1, bS=2, bS=3) rows for indexA=17..=51.
        let expected: [(i32, i32, i32, i32); 35] = [
            (17, 0,  0,  1),
            (18, 0,  0,  1),
            (19, 0,  0,  1),
            (20, 0,  0,  1),
            (21, 0,  1,  1),
            (22, 0,  1,  1),
            (23, 1,  1,  1),
            (24, 1,  1,  1),
            (25, 1,  1,  1),
            (26, 1,  1,  1),
            (27, 1,  1,  2),
            (28, 1,  1,  2),
            (29, 1,  1,  2),
            (30, 1,  1,  2),
            (31, 1,  2,  3),
            (32, 1,  2,  3),
            (33, 2,  2,  3),
            (34, 2,  2,  4),
            (35, 2,  3,  4),
            (36, 2,  3,  4),
            (37, 3,  3,  5),
            (38, 3,  4,  6),
            (39, 3,  4,  6),
            (40, 4,  5,  7),
            (41, 4,  5,  8),
            (42, 4,  6,  9),
            (43, 5,  7, 10),
            (44, 6,  8, 11),
            (45, 6,  8, 13),
            (46, 7, 10, 14),
            (47, 8, 11, 16),
            (48, 9, 12, 18),
            (49, 10, 13, 20),
            (50, 11, 15, 23),
            (51, 13, 17, 25),
        ];
        for (idx, e1, e2, e3) in expected {
            assert_eq!(tc0_from(1, idx, 8), e1, "bS=1 idxA={}", idx);
            assert_eq!(tc0_from(2, idx, 8), e2, "bS=2 idxA={}", idx);
            assert_eq!(tc0_from(3, idx, 8), e3, "bS=3 idxA={}", idx);
        }
    }

    // ---------------- §8.7.2.1 boundary-strength derivation ----------

    #[test]
    fn bs_intra_intra_mb_edge_is_4() {
        let bs = derive_boundary_strength(BsInputs {
            p_is_intra: true,
            q_is_intra: true,
            is_mb_edge: true,
            ..Default::default()
        });
        assert_eq!(bs, 4);
    }

    #[test]
    fn bs_intra_not_mb_edge_is_3() {
        let bs = derive_boundary_strength(BsInputs {
            p_is_intra: true,
            q_is_intra: false,
            is_mb_edge: false,
            ..Default::default()
        });
        assert_eq!(bs, 3);
    }

    #[test]
    fn bs_nonintra_nonzero_coeffs_is_2() {
        let bs = derive_boundary_strength(BsInputs {
            either_has_nonzero_coeffs: true,
            ..Default::default()
        });
        assert_eq!(bs, 2);
    }

    #[test]
    fn bs_nonintra_diff_ref_is_1() {
        let bs = derive_boundary_strength(BsInputs {
            different_ref_or_mv: true,
            ..Default::default()
        });
        assert_eq!(bs, 1);
    }

    #[test]
    fn bs_nonintra_same_ref_zero_mv_is_0() {
        let bs = derive_boundary_strength(BsInputs::default());
        assert_eq!(bs, 0);
    }

    #[test]
    fn bs_sp_si_mb_edge_is_4() {
        let bs = derive_boundary_strength(BsInputs {
            is_sp_or_si: true,
            is_mb_edge: true,
            ..Default::default()
        });
        assert_eq!(bs, 4);
    }

    // ---------------- Edge-filter behavioural tests ----------------

    #[test]
    fn filter_bs0_is_noop() {
        let input: [i32; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
        let out = run_filter(
            Plane::Luma,
            input,
            FilterParams {
                bs: 0,
                qp_avg: 30,
                filter_offset_a: 0,
                filter_offset_b: 0,
                bit_depth: 8,
            },
        );
        assert_eq!(out, input, "bS=0 must leave all samples unchanged");
    }

    #[test]
    fn filter_alpha_threshold_skips_filter() {
        // |p0-q0| >= α → filterSamplesFlag = 0 (eq. 8-460). With
        // qp_avg=30, indexA=30, α=28 (Table 8-16: α'[30]=28). Our p0=0
        // q0=50 gives |p0-q0|=50 > 28 → no filter.
        let input: [i32; 8] = [0, 0, 0, 0, 50, 50, 50, 50];
        let out = run_filter(
            Plane::Luma,
            input,
            FilterParams {
                bs: 2,
                qp_avg: 30,
                filter_offset_a: 0,
                filter_offset_b: 0,
                bit_depth: 8,
            },
        );
        assert_eq!(out, input, "|p0-q0| >= α must skip filter");
    }

    #[test]
    fn filter_beta_threshold_skips_filter() {
        // |p1-p0| >= β → filterSamplesFlag = 0 (eq. 8-460). With
        // qp_avg=30, indexB=30, β'=8. p1=10, p0=0 → |10-0| = 10 > 8 →
        // no filter.
        let input: [i32; 8] = [0, 0, 10, 0, 0, 0, 0, 0];
        let out = run_filter(
            Plane::Luma,
            input,
            FilterParams {
                bs: 2,
                qp_avg: 30,
                filter_offset_a: 0,
                filter_offset_b: 0,
                bit_depth: 8,
            },
        );
        assert_eq!(out, input, "|p1-p0| >= β must skip filter");
    }

    #[test]
    fn filter_bs4_luma_strong_smooths_edge() {
        // High-contrast flat edge: all p-side = 100, all q-side = 200.
        // qp_avg = 40 → indexA = 40 → α' = 80, β' = 13.
        // |p0-q0| = 100 < 80? No — 100 >= 80 means filterSamplesFlag=0,
        // so pick a smaller delta. Use p=100, q=108 (|p0-q0|=8 < 80 OK,
        // |p1-p0|=0 < 13 OK).
        //
        // Strong-filter condition |p0-q0| < (α>>2)+2 = 22 → true.
        // a_p = |p2-p0| = 0 < β → strong_p path.
        //   p0' = (p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3
        //       = (100 + 200 + 200 + 216 + 108 + 4) >> 3 = 828 >> 3
        //       = 103.
        //   p1' = (p2 + p1 + p0 + q0 + 2) >> 2
        //       = (100 + 100 + 100 + 108 + 2) >> 2 = 410 >> 2 = 102.
        //   p2' = (2*p3 + 3*p2 + p1 + p0 + q0 + 4) >> 3
        //       = (200 + 300 + 100 + 100 + 108 + 4) >> 3 = 812 >> 3
        //       = 101.
        // a_q = |q2-q0| = 0 < β → strong_q path.
        //   q0' = (p1 + 2*p0 + 2*q0 + 2*q1 + q2 + 4) >> 3
        //       = (100 + 200 + 216 + 216 + 108 + 4) >> 3 = 844 >> 3
        //       = 105.
        //   q1' = (p0 + q0 + q1 + q2 + 2) >> 2
        //       = (100 + 108 + 108 + 108 + 2) >> 2 = 426 >> 2 = 106.
        //   q2' = (2*q3 + 3*q2 + q1 + q0 + p0 + 4) >> 3
        //       = (216 + 324 + 108 + 108 + 100 + 4) >> 3 = 860 >> 3
        //       = 107.
        let input: [i32; 8] = [100, 100, 100, 100, 108, 108, 108, 108];
        let out = run_filter(
            Plane::Luma,
            input,
            FilterParams {
                bs: 4,
                qp_avg: 40,
                filter_offset_a: 0,
                filter_offset_b: 0,
                bit_depth: 8,
            },
        );
        // p3 and q3 are unchanged; p2/p1/p0, q0/q1/q2 filtered.
        assert_eq!(out[0], 100, "p3 is read-only");
        assert_eq!(out[1], 101, "p2' per eq. 8-479");
        assert_eq!(out[2], 102, "p1' per eq. 8-478");
        assert_eq!(out[3], 103, "p0' per eq. 8-477");
        assert_eq!(out[4], 105, "q0' per eq. 8-484");
        assert_eq!(out[5], 106, "q1' per eq. 8-485");
        assert_eq!(out[6], 107, "q2' per eq. 8-486");
        assert_eq!(out[7], 108, "q3 is read-only");
    }

    #[test]
    fn filter_bs1_luma_normal_updates_p0_q0() {
        // qp_avg = 30 → indexA = indexB = 30. Table: α'=28, β'=8,
        // t'C0[bs=1, 30]=1.
        // Choose p=[p3=20, p2=20, p1=20, p0=20], q=[q0=30, q1=30,
        // q2=30, q3=30]. |p0-q0|=10 < 28 ✓. |p1-p0|=0<8 ✓. |q1-q0|=0<8.
        // a_p = |p2-p0| = 0 < β → the eq. 8-465 tC includes +1.
        // a_q = |q2-q0| = 0 < β → the eq. 8-465 tC includes +1.
        // tC0 = 1, so tC = 1+1+1 = 3.
        // delta = Clip3(-3, 3, (((30-20)<<2) + (20-30) + 4) >> 3)
        //       = Clip3(-3, 3, (40 - 10 + 4) >> 3)
        //       = Clip3(-3, 3, 34>>3) = Clip3(-3,3, 4) = 3.
        // p0' = Clip1(20+3) = 23.
        // q0' = Clip1(30-3) = 27.
        // p1 update: a_p < β → eq. 8-470:
        //   step = Clip3(-1,1, (p2 + ((p0+q0+1)>>1) - (p1<<1)) >> 1)
        //        = Clip3(-1,1, (20 + ((20+30+1)>>1) - 40) >> 1)
        //        = Clip3(-1,1, (20 + 25 - 40) >> 1)
        //        = Clip3(-1,1, 5 >> 1) = Clip3(-1,1, 2) = 1.
        //   p1' = 20 + 1 = 21.
        // q1 update: a_q < β → eq. 8-472:
        //   step = Clip3(-1,1, (q2 + ((p0+q0+1)>>1) - (q1<<1)) >> 1)
        //        = Clip3(-1,1, (30 + 25 - 60) >> 1)
        //        = Clip3(-1,1, -5 >> 1) = Clip3(-1,1, -3) = -1.
        //   q1' = 30 + (-1) = 29.
        let input: [i32; 8] = [20, 20, 20, 20, 30, 30, 30, 30];
        let out = run_filter(
            Plane::Luma,
            input,
            FilterParams {
                bs: 1,
                qp_avg: 30,
                filter_offset_a: 0,
                filter_offset_b: 0,
                bit_depth: 8,
            },
        );
        assert_eq!(out[0], 20);
        assert_eq!(out[1], 20, "p2 unchanged in normal filter");
        assert_eq!(out[2], 21, "p1' per eq. 8-470");
        assert_eq!(out[3], 23, "p0' per eq. 8-468");
        assert_eq!(out[4], 27, "q0' per eq. 8-469");
        assert_eq!(out[5], 29, "q1' per eq. 8-472");
        assert_eq!(out[6], 30, "q2 unchanged in normal filter");
        assert_eq!(out[7], 30);
    }

    #[test]
    fn filter_bs1_chroma_only_updates_p0_q0() {
        // Same scenario as luma bS=1 but with chromaStyleFilteringFlag
        // on. Only p0, q0 change. p1, p2, q1, q2 are untouched
        // (eqs. 8-471, 8-473, 8-474, 8-475).
        //
        // qp_avg=30, indexA=indexB=30 → α'=28, β'=8, t'C0[1,30]=1.
        // tC = tC0 + 1 = 2 (eq. 8-466).
        // delta = Clip3(-2,2, (((30-20)<<2)+(20-30)+4)>>3)
        //       = Clip3(-2,2, 4) = 2.
        // p0' = 20+2 = 22; q0' = 30-2 = 28.
        let input: [i32; 8] = [20, 20, 20, 20, 30, 30, 30, 30];
        let out = run_filter(
            Plane::Chroma,
            input,
            FilterParams {
                bs: 1,
                qp_avg: 30,
                filter_offset_a: 0,
                filter_offset_b: 0,
                bit_depth: 8,
            },
        );
        assert_eq!(out[0], 20, "p3 unchanged");
        assert_eq!(out[1], 20, "p2 unchanged for chroma");
        assert_eq!(out[2], 20, "p1 unchanged for chroma");
        assert_eq!(out[3], 22, "p0' per eq. 8-468 with tC=tC0+1");
        assert_eq!(out[4], 28, "q0' per eq. 8-469 with tC=tC0+1");
        assert_eq!(out[5], 30, "q1 unchanged for chroma");
        assert_eq!(out[6], 30, "q2 unchanged for chroma");
        assert_eq!(out[7], 30, "q3 unchanged");
    }

    #[test]
    fn filter_bs4_chroma_only_updates_p0_q0() {
        // Chroma strong filter reduces to eqs. 8-480 / 8-487:
        // p0' = (2*p1 + p0 + q1 + 2) >> 2
        // q0' = (2*q1 + q0 + p1 + 2) >> 2
        // with α=80, β=13 threshold (indexA=40).
        //
        // p=100, q=108: |p0-q0|=8<80 ✓, |p1-p0|=0<13 ✓, |q1-q0|=0<13 ✓.
        // p0' = (200 + 100 + 108 + 2) >> 2 = 410 >> 2 = 102.
        // q0' = (216 + 108 + 100 + 2) >> 2 = 426 >> 2 = 106.
        let input: [i32; 8] = [100, 100, 100, 100, 108, 108, 108, 108];
        let out = run_filter(
            Plane::Chroma,
            input,
            FilterParams {
                bs: 4,
                qp_avg: 40,
                filter_offset_a: 0,
                filter_offset_b: 0,
                bit_depth: 8,
            },
        );
        assert_eq!(out[1], 100, "p2 unchanged for chroma-style strong");
        assert_eq!(out[2], 100, "p1 unchanged for chroma-style strong");
        assert_eq!(out[3], 102, "p0' per eq. 8-480");
        assert_eq!(out[4], 106, "q0' per eq. 8-487");
        assert_eq!(out[5], 108, "q1 unchanged for chroma-style strong");
        assert_eq!(out[6], 108, "q2 unchanged for chroma-style strong");
    }

    #[test]
    fn filter_clamps_to_sample_range() {
        // High-contrast bS=4 edge near 0 should not produce negative
        // samples — Clip1 must clamp (eq. 8-468/8-469 / 8-477/8-484).
        // qp_avg=40 → α=80, β=13. |p0-q0|=5 < (α>>2)+2=22 so strong.
        // Use p=0..0 and q=5..5. p0' = (0 + 0 + 0 + 10 + 5 + 4) >> 3
        //                       = 19 >> 3 = 2. All non-negative.
        let input: [i32; 8] = [0, 0, 0, 0, 5, 5, 5, 5];
        let out = run_filter(
            Plane::Luma,
            input,
            FilterParams {
                bs: 4,
                qp_avg: 40,
                filter_offset_a: 0,
                filter_offset_b: 0,
                bit_depth: 8,
            },
        );
        for v in out.iter() {
            assert!(*v >= 0 && *v <= 255, "sample {} out of 8-bit range", v);
        }
    }
}
