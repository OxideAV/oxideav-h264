//! In-loop deblocking filter — ITU-T H.264 §8.7.
//!
//! Implements the full §8.7 filter on frame-only 4:2:0 8-bit streams:
//!
//! * §8.7.2 boundary strength (bS) derivation per 4-sample edge segment:
//!   - MB edge with either side intra → bS = 4.
//!   - Internal edge with either side intra → bS = 3.
//!   - Either 4×4 block has non-zero coefficients → bS = 2.
//!   - Reference mismatch or quarter-pel MV delta ≥ 4 → bS = 1.
//!   - Otherwise bS = 0 (edge skipped).
//! * §8.7.2.1 `IndexA` / `IndexB` derivation from `qP_avg` + the per-slice
//!   `slice_alpha_c0_offset_div2` / `slice_beta_offset_div2`, against the
//!   Table 8-16 α / Table 8-17 β look-up.
//! * §8.7.2.2 luma filtering — normal filter for bS ∈ {1, 2, 3} and the
//!   intra-specific strong filter for bS = 4.
//! * §8.7.2.3 chroma filtering — same `IndexA` / `IndexB` on the
//!   chroma-QP-mapped QP (Table 8-15 entry via
//!   [`crate::transform::chroma_qp`]), no `aP` / `aQ` side expansion,
//!   two luma-sample-positions per 8-pixel MB (offset 0 and 4). The
//!   spec's `tC = tC0 + 1` bump for chroma is applied.
//! * §8.7.1 vertical-then-horizontal ordering.
//!
//! MBAFF (macroblock-adaptive frame/field) is scoped out — the frame-only
//! `frame_mbs_only_flag = 1` decoder this crate supports never enables it.
//! `disable_deblocking_filter_idc = 2` (filter only within slice) is
//! treated as the default idc = 0 here because this decoder currently
//! emits one slice per picture.

use crate::picture::Picture;
use crate::pps::Pps;
use crate::slice::{SliceHeader, SliceType};
use crate::transform::chroma_qp;

// --- Table 8-16 — α / β indexed by IndexA / IndexB (§8.7.2.1). ---
const ALPHA: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20,
    22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226,
    255, 255,
];
const BETA: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];

// --- Table 8-17 — tC0[bS-1][IndexA] for bS ∈ {1, 2, 3} (§8.7.2.2). ---
const TC0: [[u8; 52]; 3] = [
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 8,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 10,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16,
    ],
];

/// Apply deblocking to all edges in `pic`.
///
/// Per §8.7.1 an MB is processed vertical-edges-first then
/// horizontal-edges; within each direction edges are filtered in raster
/// order left-to-right / top-to-bottom. Both luma and chroma planes are
/// touched on every MB edge; internal luma edges are skipped when the MB
/// uses the 8×8 transform at their sub-block row/column.
pub fn deblock_picture(pic: &mut Picture, pps: &Pps, sh: &SliceHeader) {
    if sh.disable_deblocking_filter_idc == 1 {
        return;
    }
    let alpha_off = sh.slice_alpha_c0_offset_div2 * 2;
    let beta_off = sh.slice_beta_offset_div2 * 2;
    let chroma_off_cb = pps.chroma_qp_index_offset;
    let chroma_off_cr = pps.second_chroma_qp_index_offset;
    let mb_w = pic.mb_width;
    let mb_h = pic.mb_height;

    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            process_mb(
                pic,
                mb_x,
                mb_y,
                alpha_off,
                beta_off,
                chroma_off_cb,
                chroma_off_cr,
                sh.slice_type,
            );
        }
    }
}

fn process_mb(
    pic: &mut Picture,
    mb_x: u32,
    mb_y: u32,
    alpha_off: i32,
    beta_off: i32,
    cqp_off_cb: i32,
    cqp_off_cr: i32,
    slice_type: SliceType,
) {
    let mb_w = pic.mb_width;
    let mb_h = pic.mb_height;
    let _ = (mb_w, mb_h);

    // Vertical edges: MB left edge + 3 internal columns (4, 8, 12 luma /
    // 4 chroma). Internal edges at luma columns 4 and 12 are skipped when
    // the MB uses the 8×8 transform.
    let tr8 = pic.mb_info_at(mb_x, mb_y).transform_8x8;

    // Left MB edge (vertical).
    if mb_x > 0 {
        filter_mb_edge_vertical(
            pic, mb_x, mb_y, 0, alpha_off, beta_off, cqp_off_cb, cqp_off_cr, slice_type,
        );
    }
    // Internal vertical edges.
    for edge_col in [4usize, 8, 12] {
        if tr8 && edge_col != 8 {
            continue;
        }
        filter_mb_edge_vertical(
            pic, mb_x, mb_y, edge_col, alpha_off, beta_off, cqp_off_cb, cqp_off_cr, slice_type,
        );
    }

    // Top MB edge (horizontal).
    if mb_y > 0 {
        filter_mb_edge_horizontal(
            pic, mb_x, mb_y, 0, alpha_off, beta_off, cqp_off_cb, cqp_off_cr, slice_type,
        );
    }
    for edge_row in [4usize, 8, 12] {
        if tr8 && edge_row != 8 {
            continue;
        }
        filter_mb_edge_horizontal(
            pic, mb_x, mb_y, edge_row, alpha_off, beta_off, cqp_off_cb, cqp_off_cr, slice_type,
        );
    }
}

/// Filter one vertical edge of MB `(mb_x, mb_y)` at luma-sample column
/// offset `edge_col` ∈ {0, 4, 8, 12} within the MB. The edge is between
/// sub-block column `(edge_col / 4) - 1 (mod 4)` (p side) and
/// `edge_col / 4` (q side) — when `edge_col == 0` the p side is the
/// last column of the left-neighbour MB.
fn filter_mb_edge_vertical(
    pic: &mut Picture,
    mb_x: u32,
    mb_y: u32,
    edge_col: usize,
    alpha_off: i32,
    beta_off: i32,
    cqp_off_cb: i32,
    cqp_off_cr: i32,
    slice_type: SliceType,
) {
    let is_mb_edge = edge_col == 0;
    let sub_q_col = edge_col / 4; // 0..=3 within the q-side MB
    let (p_mb_x, p_mb_y, p_sub_col) = if is_mb_edge {
        (mb_x - 1, mb_y, 3usize)
    } else {
        (mb_x, mb_y, sub_q_col - 1)
    };
    let q_mb_x = mb_x;
    let q_mb_y = mb_y;

    // Compute bS per 4-luma-sample row (4 entries).
    let mut bs = [0u8; 4];
    for r in 0..4usize {
        bs[r] = derive_bs(
            pic, p_mb_x, p_mb_y, p_sub_col, r, q_mb_x, q_mb_y, sub_q_col, r, is_mb_edge,
            /*vertical=*/ true, slice_type,
        );
    }
    if bs.iter().all(|&b| b == 0) {
        return;
    }

    let qp_p = pic.mb_info_at(p_mb_x, p_mb_y).qp_y;
    let qp_q = pic.mb_info_at(q_mb_x, q_mb_y).qp_y;
    let qp_avg_y = (qp_p + qp_q + 1) >> 1;

    // Luma — filter 16 rows (4 per bS segment).
    let y_base_x = (mb_x * 16) as usize + edge_col;
    let y_base_y = (mb_y * 16) as usize;
    for seg in 0..4usize {
        if bs[seg] == 0 {
            continue;
        }
        filter_luma_edge_4(
            pic,
            y_base_x,
            y_base_y + seg * 4,
            /*vertical=*/ true,
            bs[seg] as i32,
            qp_avg_y,
            alpha_off,
            beta_off,
        );
    }

    // Chroma — only on edges coincident with an 8-luma-sample boundary
    // (edge_col ∈ {0, 8}) because chroma is 4:2:0 sub-sampled and the
    // 8×8 chroma MB has just one mid-edge at luma offset 8.
    if edge_col == 0 || edge_col == 8 {
        for (plane_cb, qp_off) in [(true, cqp_off_cb), (false, cqp_off_cr)] {
            let qp_pc = chroma_qp(qp_p, qp_off);
            let qp_qc = chroma_qp(qp_q, qp_off);
            let qp_avg_c = (qp_pc + qp_qc + 1) >> 1;
            let c_base_x = (mb_x * 8) as usize + edge_col / 2;
            let c_base_y = (mb_y * 8) as usize;
            // Chroma is 4:2:0: 4 rows per segment pair in the MB;
            // chroma's 8-sample MB height splits into two 4-row bands,
            // each driven by the bS of the two luma segments it overlaps
            // (segments 0+1 for top band, 2+3 for bottom band).
            for band in 0..2usize {
                let seg0 = band * 2;
                // Top two chroma rows key on luma bS[seg0]; bottom two
                // on luma bS[seg0+1] (§8.7.2.3 paragraph on chroma 4:2:0
                // per-4-luma-row bS replication).
                for sub in 0..2usize {
                    let bs_here = bs[seg0 + sub] as i32;
                    if bs_here == 0 {
                        continue;
                    }
                    let cy = c_base_y + band * 4 + sub * 2;
                    filter_chroma_edge_2(
                        pic, plane_cb, c_base_x, cy, /*vertical=*/ true, bs_here, qp_avg_c,
                        alpha_off, beta_off,
                    );
                }
            }
        }
    }
}

fn filter_mb_edge_horizontal(
    pic: &mut Picture,
    mb_x: u32,
    mb_y: u32,
    edge_row: usize,
    alpha_off: i32,
    beta_off: i32,
    cqp_off_cb: i32,
    cqp_off_cr: i32,
    slice_type: SliceType,
) {
    let is_mb_edge = edge_row == 0;
    let sub_q_row = edge_row / 4;
    let (p_mb_x, p_mb_y, p_sub_row) = if is_mb_edge {
        (mb_x, mb_y - 1, 3usize)
    } else {
        (mb_x, mb_y, sub_q_row - 1)
    };
    let q_mb_x = mb_x;
    let q_mb_y = mb_y;

    let mut bs = [0u8; 4];
    for c in 0..4usize {
        bs[c] = derive_bs(
            pic, p_mb_x, p_mb_y, c, p_sub_row, q_mb_x, q_mb_y, c, sub_q_row, is_mb_edge,
            /*vertical=*/ false, slice_type,
        );
    }
    if bs.iter().all(|&b| b == 0) {
        return;
    }

    let qp_p = pic.mb_info_at(p_mb_x, p_mb_y).qp_y;
    let qp_q = pic.mb_info_at(q_mb_x, q_mb_y).qp_y;
    let qp_avg_y = (qp_p + qp_q + 1) >> 1;

    let y_base_x = (mb_x * 16) as usize;
    let y_base_y = (mb_y * 16) as usize + edge_row;
    for seg in 0..4usize {
        if bs[seg] == 0 {
            continue;
        }
        filter_luma_edge_4(
            pic,
            y_base_x + seg * 4,
            y_base_y,
            /*vertical=*/ false,
            bs[seg] as i32,
            qp_avg_y,
            alpha_off,
            beta_off,
        );
    }

    if edge_row == 0 || edge_row == 8 {
        for (plane_cb, qp_off) in [(true, cqp_off_cb), (false, cqp_off_cr)] {
            let qp_pc = chroma_qp(qp_p, qp_off);
            let qp_qc = chroma_qp(qp_q, qp_off);
            let qp_avg_c = (qp_pc + qp_qc + 1) >> 1;
            let c_base_x = (mb_x * 8) as usize;
            let c_base_y = (mb_y * 8) as usize + edge_row / 2;
            for band in 0..2usize {
                let seg0 = band * 2;
                for sub in 0..2usize {
                    let bs_here = bs[seg0 + sub] as i32;
                    if bs_here == 0 {
                        continue;
                    }
                    let cx = c_base_x + band * 4 + sub * 2;
                    filter_chroma_edge_2(
                        pic, plane_cb, cx, c_base_y, /*vertical=*/ false, bs_here, qp_avg_c,
                        alpha_off, beta_off,
                    );
                }
            }
        }
    }
}

/// §8.7.2 Derivation process for boundary strength between two adjacent
/// 4×4 sub-blocks `(p_mb, p_sub_col, p_sub_row)` ← → `(q_mb, q_sub_col,
/// q_sub_row)`. `is_mb_edge` is true when the edge lies on an MB
/// boundary (different macroblocks on each side).
#[allow(clippy::too_many_arguments)]
fn derive_bs(
    pic: &Picture,
    p_mb_x: u32,
    p_mb_y: u32,
    p_sub_col: usize,
    p_sub_row: usize,
    q_mb_x: u32,
    q_mb_y: u32,
    q_sub_col: usize,
    q_sub_row: usize,
    is_mb_edge: bool,
    _vertical: bool,
    slice_type: SliceType,
) -> u8 {
    let p_info = pic.mb_info_at(p_mb_x, p_mb_y);
    let q_info = pic.mb_info_at(q_mb_x, q_mb_y);
    let p_intra = p_info.intra;
    let q_intra = q_info.intra;
    if p_intra || q_intra {
        return if is_mb_edge { 4 } else { 3 };
    }
    let p_nc = p_info.luma_nc[p_sub_row * 4 + p_sub_col];
    let q_nc = q_info.luma_nc[q_sub_row * 4 + q_sub_col];
    if p_nc > 0 || q_nc > 0 {
        return 2;
    }
    if matches!(slice_type, SliceType::I | SliceType::SI) {
        return 0;
    }
    // Inter boundary with both sides fully reconstructed but no coded
    // residuals — compare MVs and reference pictures per §8.7.2 inter path.
    let p_idx = p_sub_row * 4 + p_sub_col;
    let q_idx = q_sub_row * 4 + q_sub_col;
    let (p_r0, p_r1) = (p_info.ref_idx_l0[p_idx], p_info.ref_idx_l1[p_idx]);
    let (q_r0, q_r1) = (q_info.ref_idx_l0[q_idx], q_info.ref_idx_l1[q_idx]);
    let (p_v0, p_v1) = (p_info.mv_l0[p_idx], p_info.mv_l1[p_idx]);
    let (q_v0, q_v1) = (q_info.mv_l0[q_idx], q_info.mv_l1[q_idx]);
    // Collapse p and q to a normalised "set of active (list, ref_poc, mv)"
    // pair; when the lists point to the same POC they're considered
    // equivalent. Only the set-compare matters: either both sides point
    // to the same references AND their MVs all lie within 4 quarter-pel
    // units, or not.
    let p_refs = [
        ref_poc(p_info.ref_poc_l0, p_idx, p_r0),
        ref_poc(p_info.ref_poc_l1, p_idx, p_r1),
    ];
    let q_refs = [
        ref_poc(q_info.ref_poc_l0, q_idx, q_r0),
        ref_poc(q_info.ref_poc_l1, q_idx, q_r1),
    ];
    // Spec §8.7.2 "inter" case: bS = 1 when
    // the two reference frames are different OR the MV delta along any
    // component ≥ 4 (quarter-pel units). Treat the two lists independently
    // with a try-both-matchings fallback for the B-slice "swap" case.
    let match_same = same_ref(p_refs[0], q_refs[0])
        && same_ref(p_refs[1], q_refs[1])
        && mv_close(p_v0, q_v0)
        && mv_close(p_v1, q_v1);
    let match_swap = same_ref(p_refs[0], q_refs[1])
        && same_ref(p_refs[1], q_refs[0])
        && mv_close(p_v0, q_v1)
        && mv_close(p_v1, q_v0);
    if match_same || match_swap {
        0
    } else {
        1
    }
}

/// `i32::MIN` sentinel from `Picture::snapshot_ref_pocs` means "list not
/// used at this 4×4 block." For pre-snapshot in-slice lookups we fall back
/// to a synthetic POC keyed on the ref_idx so two blocks with the same
/// list/ref_idx compare equal.
fn ref_poc(pocs: [i32; 16], idx: usize, ref_idx: i8) -> Option<i32> {
    if ref_idx < 0 {
        None
    } else if pocs[idx] != i32::MIN {
        Some(pocs[idx])
    } else {
        Some(-(ref_idx as i32 + 1))
    }
}

fn same_ref(a: Option<i32>, b: Option<i32>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

fn mv_close(a: (i16, i16), b: (i16, i16)) -> bool {
    (a.0 as i32 - b.0 as i32).abs() < 4 && (a.1 as i32 - b.1 as i32).abs() < 4
}

/// Apply the §8.7.2.2 luma filter over a 4-sample edge segment. `vertical`
/// = true filters across the vertical line at column `x0` for rows
/// `y0..y0+4`; false filters across the horizontal line at row `y0` for
/// columns `x0..x0+4`.
#[allow(clippy::too_many_arguments)]
fn filter_luma_edge_4(
    pic: &mut Picture,
    x0: usize,
    y0: usize,
    vertical: bool,
    bs: i32,
    qp_avg: i32,
    alpha_off: i32,
    beta_off: i32,
) {
    let stride = pic.luma_stride();
    let index_a = (qp_avg + alpha_off).clamp(0, 51) as usize;
    let index_b = (qp_avg + beta_off).clamp(0, 51) as usize;
    let alpha = ALPHA[index_a] as i32;
    let beta = BETA[index_b] as i32;
    if alpha == 0 || beta == 0 {
        return;
    }
    let tc0 = if bs < 4 {
        TC0[bs as usize - 1][index_a] as i32
    } else {
        0
    };

    for i in 0..4usize {
        // (p3, p2, p1, p0, q0, q1, q2, q3) offsets relative to the edge.
        let idx = |delta: isize| -> usize {
            if vertical {
                let col = (x0 as isize + delta) as usize;
                (y0 + i) * stride + col
            } else {
                let row = (y0 as isize + delta) as usize;
                row * stride + x0 + i
            }
        };
        filter_luma_line(pic, idx, bs, alpha, beta, tc0);
    }
}

fn filter_luma_line(
    pic: &mut Picture,
    idx: impl Fn(isize) -> usize,
    bs: i32,
    alpha: i32,
    beta: i32,
    tc0: i32,
) {
    let p3_i = idx(-4);
    let p2_i = idx(-3);
    let p1_i = idx(-2);
    let p0_i = idx(-1);
    let q0_i = idx(0);
    let q1_i = idx(1);
    let q2_i = idx(2);
    let q3_i = idx(3);
    let p = [
        pic.y[p0_i] as i32,
        pic.y[p1_i] as i32,
        pic.y[p2_i] as i32,
        pic.y[p3_i] as i32,
    ];
    let q = [
        pic.y[q0_i] as i32,
        pic.y[q1_i] as i32,
        pic.y[q2_i] as i32,
        pic.y[q3_i] as i32,
    ];
    // §8.7.2.2 edge-activity test.
    if (p[0] - q[0]).abs() >= alpha || (p[1] - p[0]).abs() >= beta || (q[1] - q[0]).abs() >= beta {
        return;
    }
    let ap = (p[2] - p[0]).abs() < beta;
    let aq = (q[2] - q[0]).abs() < beta;
    if bs == 4 {
        // §8.7.2.2 strong filter.
        let strong_p = ap && (p[0] - q[0]).abs() < ((alpha >> 2) + 2);
        let strong_q = aq && (p[0] - q[0]).abs() < ((alpha >> 2) + 2);
        let (np0, np1, np2) = if strong_p {
            (
                (p[2] + 2 * p[1] + 2 * p[0] + 2 * q[0] + q[1] + 4) >> 3,
                (p[2] + p[1] + p[0] + q[0] + 2) >> 2,
                (2 * p[3] + 3 * p[2] + p[1] + p[0] + q[0] + 4) >> 3,
            )
        } else {
            (
                (2 * p[1] + p[0] + q[1] + 2) >> 2,
                pic.y[p1_i] as i32,
                pic.y[p2_i] as i32,
            )
        };
        let (nq0, nq1, nq2) = if strong_q {
            (
                (p[1] + 2 * p[0] + 2 * q[0] + 2 * q[1] + q[2] + 4) >> 3,
                (p[0] + q[0] + q[1] + q[2] + 2) >> 2,
                (2 * q[3] + 3 * q[2] + q[1] + q[0] + p[0] + 4) >> 3,
            )
        } else {
            (
                (2 * q[1] + q[0] + p[1] + 2) >> 2,
                pic.y[q1_i] as i32,
                pic.y[q2_i] as i32,
            )
        };
        pic.y[p0_i] = np0.clamp(0, 255) as u8;
        pic.y[p1_i] = np1.clamp(0, 255) as u8;
        pic.y[p2_i] = np2.clamp(0, 255) as u8;
        pic.y[q0_i] = nq0.clamp(0, 255) as u8;
        pic.y[q1_i] = nq1.clamp(0, 255) as u8;
        pic.y[q2_i] = nq2.clamp(0, 255) as u8;
    } else {
        // §8.7.2.2 normal filter.
        let mut tc = tc0;
        if ap {
            tc += 1;
        }
        if aq {
            tc += 1;
        }
        let delta = ((((q[0] - p[0]) << 2) + (p[1] - q[1]) + 4) >> 3).clamp(-tc, tc);
        let np0 = (p[0] + delta).clamp(0, 255);
        let nq0 = (q[0] - delta).clamp(0, 255);
        pic.y[p0_i] = np0 as u8;
        pic.y[q0_i] = nq0 as u8;
        if ap {
            let dp = ((p[2] + ((p[0] + q[0] + 1) >> 1) - 2 * p[1]) >> 1).clamp(-tc0, tc0);
            pic.y[p1_i] = (p[1] + dp).clamp(0, 255) as u8;
        }
        if aq {
            let dq = ((q[2] + ((p[0] + q[0] + 1) >> 1) - 2 * q[1]) >> 1).clamp(-tc0, tc0);
            pic.y[q1_i] = (q[1] + dq).clamp(0, 255) as u8;
        }
    }
}

/// §8.7.2.3 chroma filter on a 2-sample edge segment. Chroma never
/// applies the strong p2/q2 update from the luma path — §8.7.2.3 writes
/// only p0 and q0, using `tC = tC0 + 1` (no `aP`/`aQ` bumps).
#[allow(clippy::too_many_arguments)]
fn filter_chroma_edge_2(
    pic: &mut Picture,
    plane_cb: bool,
    x0: usize,
    y0: usize,
    vertical: bool,
    bs: i32,
    qp_avg: i32,
    alpha_off: i32,
    beta_off: i32,
) {
    let stride = pic.chroma_stride();
    let index_a = (qp_avg + alpha_off).clamp(0, 51) as usize;
    let index_b = (qp_avg + beta_off).clamp(0, 51) as usize;
    let alpha = ALPHA[index_a] as i32;
    let beta = BETA[index_b] as i32;
    if alpha == 0 || beta == 0 {
        return;
    }
    // Chroma tC: for bS < 4, `tC = tC0 + 1` (§8.7.2.3); for bS = 4 the
    // chroma path uses the 2-tap fallback also used by the "weak side" of
    // the luma strong filter.
    let tc = if bs < 4 {
        TC0[bs as usize - 1][index_a] as i32 + 1
    } else {
        0
    };

    for i in 0..2usize {
        let idx = |delta: isize| -> usize {
            if vertical {
                let col = (x0 as isize + delta) as usize;
                (y0 + i) * stride + col
            } else {
                let row = (y0 as isize + delta) as usize;
                row * stride + x0 + i
            }
        };
        let p1_i = idx(-2);
        let p0_i = idx(-1);
        let q0_i = idx(0);
        let q1_i = idx(1);
        let plane = if plane_cb { &mut pic.cb } else { &mut pic.cr };
        let p0 = plane[p0_i] as i32;
        let p1 = plane[p1_i] as i32;
        let q0 = plane[q0_i] as i32;
        let q1 = plane[q1_i] as i32;
        if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }
        if bs == 4 {
            // §8.7.2.3 bS == 4: p0' = (2p1 + p0 + q1 + 2) >> 2,
            // q0' = (2q1 + q0 + p1 + 2) >> 2 (no ap/aq test on chroma).
            plane[p0_i] = ((2 * p1 + p0 + q1 + 2) >> 2).clamp(0, 255) as u8;
            plane[q0_i] = ((2 * q1 + q0 + p1 + 2) >> 2).clamp(0, 255) as u8;
        } else {
            let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
            plane[p0_i] = (p0 + delta).clamp(0, 255) as u8;
            plane[q0_i] = (q0 - delta).clamp(0, 255) as u8;
        }
    }
}
