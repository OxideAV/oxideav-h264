//! High-bit-depth in-loop deblocking filter — ITU-T H.264 §8.7 on u16 samples.
//!
//! Mirrors [`crate::deblock`] but reads / writes the u16 planes
//! [`crate::picture::Picture::y16`] / `cb16` / `cr16`. The bS derivation
//! and edge schedule are identical — only the filter math changes:
//!
//! * α, β, tC0 scale by `1 << (BitDepth - 8)` (§8.7.2.1).
//! * Final clip bound is `(1 << BitDepth) - 1` instead of 255.
//!
//! 4:2:0 only — the high-bit-depth path currently rejects other chroma
//! formats at slice entry.
//!
//! The bS-derivation helper lives in [`crate::deblock`] as a `pub(crate)`
//! function so both paths share one copy; this module only reimplements
//! the §8.7.2.2 / §8.7.2.3 filter math and the edge-schedule wrapper.

use crate::deblock::{derive_bs_for_edge, ALPHA_TABLE, BETA_TABLE, TC0_TABLE};
use crate::picture::Picture;
use crate::pps::Pps;
use crate::slice::{SliceHeader, SliceType};
use crate::transform::chroma_qp;

/// Apply deblocking to all edges in `pic` using the u16 sample planes.
/// Entry point for the high-bit-depth pipeline.
pub fn deblock_picture_hi(pic: &mut Picture, pps: &Pps, sh: &SliceHeader) {
    if sh.disable_deblocking_filter_idc == 1 {
        return;
    }
    let alpha_off = sh.slice_alpha_c0_offset_div2 * 2;
    let beta_off = sh.slice_beta_offset_div2 * 2;
    let chroma_off_cb = pps.chroma_qp_index_offset;
    let chroma_off_cr = pps.second_chroma_qp_index_offset;
    let mb_w = pic.mb_width;
    let mb_h = pic.mb_height;
    let bit_depth_y = pic.bit_depth_y;
    let bit_depth_c = pic.bit_depth_c;

    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            process_mb_hi(
                pic, mb_x, mb_y, alpha_off, beta_off, chroma_off_cb, chroma_off_cr,
                sh.slice_type, bit_depth_y, bit_depth_c,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn process_mb_hi(
    pic: &mut Picture,
    mb_x: u32,
    mb_y: u32,
    alpha_off: i32,
    beta_off: i32,
    cqp_off_cb: i32,
    cqp_off_cr: i32,
    slice_type: SliceType,
    bit_depth_y: u8,
    bit_depth_c: u8,
) {
    let tr8 = pic.mb_info_at(mb_x, mb_y).transform_8x8;

    if mb_x > 0 {
        filter_mb_edge_vertical_hi(
            pic, mb_x, mb_y, 0, alpha_off, beta_off, cqp_off_cb, cqp_off_cr, slice_type,
            bit_depth_y, bit_depth_c,
        );
    }
    for edge_col in [4usize, 8, 12] {
        if tr8 && edge_col != 8 {
            continue;
        }
        filter_mb_edge_vertical_hi(
            pic, mb_x, mb_y, edge_col, alpha_off, beta_off, cqp_off_cb, cqp_off_cr, slice_type,
            bit_depth_y, bit_depth_c,
        );
    }

    if mb_y > 0 {
        filter_mb_edge_horizontal_hi(
            pic, mb_x, mb_y, 0, alpha_off, beta_off, cqp_off_cb, cqp_off_cr, slice_type,
            bit_depth_y, bit_depth_c,
        );
    }
    for edge_row in [4usize, 8, 12] {
        if tr8 && edge_row != 8 {
            continue;
        }
        filter_mb_edge_horizontal_hi(
            pic, mb_x, mb_y, edge_row, alpha_off, beta_off, cqp_off_cb, cqp_off_cr, slice_type,
            bit_depth_y, bit_depth_c,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn filter_mb_edge_vertical_hi(
    pic: &mut Picture,
    mb_x: u32,
    mb_y: u32,
    edge_col: usize,
    alpha_off: i32,
    beta_off: i32,
    cqp_off_cb: i32,
    cqp_off_cr: i32,
    slice_type: SliceType,
    bit_depth_y: u8,
    bit_depth_c: u8,
) {
    let is_mb_edge = edge_col == 0;
    let sub_q_col = edge_col / 4;
    let (p_mb_x, p_mb_y, p_sub_col) = if is_mb_edge {
        (mb_x - 1, mb_y, 3usize)
    } else {
        (mb_x, mb_y, sub_q_col - 1)
    };
    let q_mb_x = mb_x;
    let q_mb_y = mb_y;

    let mut bs = [0u8; 4];
    for r in 0..4usize {
        bs[r] = derive_bs_for_edge(
            pic, p_mb_x, p_mb_y, p_sub_col, r, q_mb_x, q_mb_y, sub_q_col, r, is_mb_edge,
            slice_type,
        );
    }
    if bs.iter().all(|&b| b == 0) {
        return;
    }

    let qp_p = pic.mb_info_at(p_mb_x, p_mb_y).qp_y;
    let qp_q = pic.mb_info_at(q_mb_x, q_mb_y).qp_y;
    let qp_avg_y = (qp_p + qp_q + 1) >> 1;

    let y_base_x = (mb_x * 16) as usize + edge_col;
    let y_base_y = (mb_y * 16) as usize;
    for seg in 0..4usize {
        if bs[seg] == 0 {
            continue;
        }
        filter_luma_edge_4_hi(
            pic, y_base_x, y_base_y + seg * 4, true, bs[seg] as i32, qp_avg_y,
            alpha_off, beta_off, bit_depth_y,
        );
    }

    if edge_col == 0 || edge_col == 8 {
        let mb_height_c = pic.mb_height_chroma();
        for (plane_cb, qp_off) in [(true, cqp_off_cb), (false, cqp_off_cr)] {
            let qp_pc = chroma_qp(qp_p, qp_off);
            let qp_qc = chroma_qp(qp_q, qp_off);
            let qp_avg_c = (qp_pc + qp_qc + 1) >> 1;
            let c_base_x = (mb_x * 8) as usize + edge_col / 2;
            let c_base_y = (mb_y as usize) * mb_height_c;
            for band in 0..2usize {
                for sub in 0..2usize {
                    let bs_idx = band * 2 + sub;
                    let bs_here = bs[bs_idx] as i32;
                    if bs_here == 0 {
                        continue;
                    }
                    let cy = c_base_y + band * 4 + sub * 2;
                    filter_chroma_edge_2_hi(
                        pic, plane_cb, c_base_x, cy, true, bs_here, qp_avg_c,
                        alpha_off, beta_off, bit_depth_c,
                    );
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn filter_mb_edge_horizontal_hi(
    pic: &mut Picture,
    mb_x: u32,
    mb_y: u32,
    edge_row: usize,
    alpha_off: i32,
    beta_off: i32,
    cqp_off_cb: i32,
    cqp_off_cr: i32,
    slice_type: SliceType,
    bit_depth_y: u8,
    bit_depth_c: u8,
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
        bs[c] = derive_bs_for_edge(
            pic, p_mb_x, p_mb_y, c, p_sub_row, q_mb_x, q_mb_y, c, sub_q_row, is_mb_edge,
            slice_type,
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
        filter_luma_edge_4_hi(
            pic, y_base_x + seg * 4, y_base_y, false, bs[seg] as i32, qp_avg_y,
            alpha_off, beta_off, bit_depth_y,
        );
    }

    let chroma_edge_applies = edge_row == 0 || edge_row == 8;
    if chroma_edge_applies {
        let mb_height_c = pic.mb_height_chroma();
        for (plane_cb, qp_off) in [(true, cqp_off_cb), (false, cqp_off_cr)] {
            let qp_pc = chroma_qp(qp_p, qp_off);
            let qp_qc = chroma_qp(qp_q, qp_off);
            let qp_avg_c = (qp_pc + qp_qc + 1) >> 1;
            let c_base_x = (mb_x * 8) as usize;
            let c_base_y = (mb_y as usize) * mb_height_c + edge_row / 2;
            for band in 0..2usize {
                let seg0 = band * 2;
                for sub in 0..2usize {
                    let bs_here = bs[seg0 + sub] as i32;
                    if bs_here == 0 {
                        continue;
                    }
                    let cx = c_base_x + band * 4 + sub * 2;
                    filter_chroma_edge_2_hi(
                        pic, plane_cb, cx, c_base_y, false, bs_here, qp_avg_c,
                        alpha_off, beta_off, bit_depth_c,
                    );
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn filter_luma_edge_4_hi(
    pic: &mut Picture,
    x0: usize,
    y0: usize,
    vertical: bool,
    bs: i32,
    qp_avg: i32,
    alpha_off: i32,
    beta_off: i32,
    bit_depth: u8,
) {
    let stride = pic.luma_stride();
    let index_a = (qp_avg + alpha_off).clamp(0, 51) as usize;
    let index_b = (qp_avg + beta_off).clamp(0, 51) as usize;
    let shift = (bit_depth - 8) as u32;
    let alpha = (ALPHA_TABLE[index_a] as i32) << shift;
    let beta = (BETA_TABLE[index_b] as i32) << shift;
    if alpha == 0 || beta == 0 {
        return;
    }
    let tc0 = if bs < 4 {
        (TC0_TABLE[bs as usize - 1][index_a] as i32) << shift
    } else {
        0
    };
    let max_sample: i32 = (1i32 << bit_depth) - 1;

    for i in 0..4usize {
        let idx = |delta: isize| -> usize {
            if vertical {
                let col = (x0 as isize + delta) as usize;
                (y0 + i) * stride + col
            } else {
                let row = (y0 as isize + delta) as usize;
                row * stride + x0 + i
            }
        };
        filter_luma_line_hi(pic, idx, bs, alpha, beta, tc0, max_sample);
    }
}

fn filter_luma_line_hi(
    pic: &mut Picture,
    idx: impl Fn(isize) -> usize,
    bs: i32,
    alpha: i32,
    beta: i32,
    tc0: i32,
    max_sample: i32,
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
        pic.y16[p0_i] as i32,
        pic.y16[p1_i] as i32,
        pic.y16[p2_i] as i32,
        pic.y16[p3_i] as i32,
    ];
    let q = [
        pic.y16[q0_i] as i32,
        pic.y16[q1_i] as i32,
        pic.y16[q2_i] as i32,
        pic.y16[q3_i] as i32,
    ];
    if (p[0] - q[0]).abs() >= alpha || (p[1] - p[0]).abs() >= beta || (q[1] - q[0]).abs() >= beta {
        return;
    }
    let ap = (p[2] - p[0]).abs() < beta;
    let aq = (q[2] - q[0]).abs() < beta;
    if bs == 4 {
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
                pic.y16[p1_i] as i32,
                pic.y16[p2_i] as i32,
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
                pic.y16[q1_i] as i32,
                pic.y16[q2_i] as i32,
            )
        };
        pic.y16[p0_i] = np0.clamp(0, max_sample) as u16;
        pic.y16[p1_i] = np1.clamp(0, max_sample) as u16;
        pic.y16[p2_i] = np2.clamp(0, max_sample) as u16;
        pic.y16[q0_i] = nq0.clamp(0, max_sample) as u16;
        pic.y16[q1_i] = nq1.clamp(0, max_sample) as u16;
        pic.y16[q2_i] = nq2.clamp(0, max_sample) as u16;
    } else {
        let mut tc = tc0;
        if ap {
            tc += 1;
        }
        if aq {
            tc += 1;
        }
        let delta = ((((q[0] - p[0]) << 2) + (p[1] - q[1]) + 4) >> 3).clamp(-tc, tc);
        let np0 = (p[0] + delta).clamp(0, max_sample);
        let nq0 = (q[0] - delta).clamp(0, max_sample);
        pic.y16[p0_i] = np0 as u16;
        pic.y16[q0_i] = nq0 as u16;
        if ap {
            let dp = ((p[2] + ((p[0] + q[0] + 1) >> 1) - 2 * p[1]) >> 1).clamp(-tc0, tc0);
            pic.y16[p1_i] = (p[1] + dp).clamp(0, max_sample) as u16;
        }
        if aq {
            let dq = ((q[2] + ((p[0] + q[0] + 1) >> 1) - 2 * q[1]) >> 1).clamp(-tc0, tc0);
            pic.y16[q1_i] = (q[1] + dq).clamp(0, max_sample) as u16;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn filter_chroma_edge_2_hi(
    pic: &mut Picture,
    plane_cb: bool,
    x0: usize,
    y0: usize,
    vertical: bool,
    bs: i32,
    qp_avg: i32,
    alpha_off: i32,
    beta_off: i32,
    bit_depth: u8,
) {
    let stride = pic.chroma_stride();
    let index_a = (qp_avg + alpha_off).clamp(0, 51) as usize;
    let index_b = (qp_avg + beta_off).clamp(0, 51) as usize;
    let shift = (bit_depth - 8) as u32;
    let alpha = (ALPHA_TABLE[index_a] as i32) << shift;
    let beta = (BETA_TABLE[index_b] as i32) << shift;
    if alpha == 0 || beta == 0 {
        return;
    }
    let tc = if bs < 4 {
        ((TC0_TABLE[bs as usize - 1][index_a] as i32) << shift) + 1
    } else {
        0
    };
    let max_sample: i32 = (1i32 << bit_depth) - 1;

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
        let plane = if plane_cb { &mut pic.cb16 } else { &mut pic.cr16 };
        let p0 = plane[p0_i] as i32;
        let p1 = plane[p1_i] as i32;
        let q0 = plane[q0_i] as i32;
        let q1 = plane[q1_i] as i32;
        if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }
        if bs == 4 {
            plane[p0_i] = ((2 * p1 + p0 + q1 + 2) >> 2).clamp(0, max_sample) as u16;
            plane[q0_i] = ((2 * q1 + q0 + p1 + 2) >> 2).clamp(0, max_sample) as u16;
        } else {
            let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
            plane[p0_i] = (p0 + delta).clamp(0, max_sample) as u16;
            plane[q0_i] = (q0 - delta).clamp(0, max_sample) as u16;
        }
    }
}
