//! 4:4:4 (ChromaArrayType == 3) B-slice CAVLC macroblock decode.
//!
//! Mirrors [`crate::b_mb`] but with the 4:4:4 pipeline applied on top:
//!
//! * Motion compensation runs the luma 6-tap + bilinear quarter-pel filter
//!   on all three planes (§8.4.2.2.1, ChromaArrayType = 3 Table 8-9).
//! * Residual per MB is three back-to-back luma-style streams — 16
//!   Luma4x4 per plane, gated by the 4-bit inter CBP remapped through
//!   [`crate::p_mb_444::GOLOMB_TO_INTER_CBP_GRAY`].
//!
//! The MV / direct-mode / partition logic is reused from [`crate::b_mb`]
//! via a few newly-exported helpers that write MV state into `pic.mb_info`
//! without issuing MC. This keeps the spatial / temporal direct derivations
//! (§8.4.1.2.2 / §8.4.1.2.3) and partition-shape dispatch DRY.

use oxideav_core::{Error, Result};

use crate::cavlc::{decode_residual_block, BlockKind};
use crate::golomb::BitReaderExt;
use crate::mb::LUMA_BLOCK_RASTER;
use crate::mb_444::{self as mb444};
use crate::mb_type::{
    decode_b_slice_mb_type, decode_b_sub_mb_type, BMbType, BPartition, BSubPartition, PredDir,
};
use crate::motion::{
    apply_luma_weight, average_bipred, implicit_bipred_weights, implicit_weighted_bipred_plane,
    luma_mc_plane, predict_mv_list, weighted_bipred_plane, ImplicitBiWeights,
};
use crate::p_mb_444::GOLOMB_TO_INTER_CBP_GRAY;
use crate::picture::{MbInfo, Picture, INTRA_DC_FAKE};
use crate::pps::Pps;
use crate::slice::{LumaWeight, PredWeightTable, SliceHeader};
use crate::sps::Sps;
use crate::transform::{chroma_qp, dequantize_4x4_scaled, idct_4x4};
use oxideav_core::bits::BitReader;

use crate::b_mb::{
    apply_motion_compensation_b, direct_16x16_compensate_pub, direct_8x8_compensate_pub, BSliceCtx,
    BipredMode,
};

// Re-export the 4:2:0 B sub-MB helpers the 4:4:4 path reuses for syntax
// parsing; `b_mb_444` only differs in MC + residual layout.

/// Decode one coded B-slice macroblock under 4:4:4.
pub fn decode_b_slice_mb_444(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    ctx: &BSliceCtx<'_>,
    prev_qp: &mut i32,
) -> Result<()> {
    debug_assert_eq!(sps.chroma_format_idc, 3);
    let mb_type = br.read_ue()?;
    let bmb = decode_b_slice_mb_type(mb_type)
        .ok_or_else(|| Error::invalid(format!("h264 b-slice 4:4:4: bad B mb_type {mb_type}")))?;
    match bmb {
        BMbType::IntraInB(imb) => {
            mb444::decode_intra_mb_444(br, sps, pps, sh, mb_x, mb_y, pic, prev_qp, imb)
        }
        BMbType::Inter { partition } => decode_b_inter_mb_444(
            br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, ref_list1, ctx, prev_qp, partition,
        ),
    }
}

/// 4:4:4 `B_Skip` — direct-mode MVs + zero residual, applied across all
/// three planes via the luma filter.
pub fn decode_b_skip_mb_444(
    sh: &SliceHeader,
    sps: &Sps,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    ctx: &BSliceCtx<'_>,
    prev_qp: i32,
) -> Result<()> {
    // Initialise MB info for the direct MB — mirrors the 4:2:0 path.
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        *info = MbInfo {
            qp_y: prev_qp,
            coded: true,
            intra: false,
            skipped: true,
            mb_type_p: None,
            p_partition: None,
            ..Default::default()
        };
    }
    // §8.4.1.2 — direct-mode MV derivation is format-agnostic. We run it
    // against a scratch picture clone with 4:2:0 buffers so the 4:2:0
    // `apply_motion_compensation` writes don't corrupt our 4:4:4 chroma
    // planes; then we read back the MB-level MVs and re-run MC with the
    // luma filter on all three planes.
    let (mvs, dirs) = snapshot_mv_state_via_420(
        sh,
        sps,
        ctx,
        mb_x,
        mb_y,
        pic,
        ref_list0,
        ref_list1,
        |sh2, sps2, ctx2, pic2, rl0, rl1| {
            crate::b_mb::direct_16x16_compensate_pub(sh2, sps2, ctx2, mb_x, mb_y, pic2, rl0, rl1)
        },
    )?;
    // Re-do MC on the 4:4:4 picture with the collected MVs.
    apply_mvs_444(pic, ref_list0, ref_list1, sh, ctx, mb_x, mb_y, &mvs, &dirs);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_b_inter_mb_444(
    br: &mut BitReader<'_>,
    sps: &Sps,
    pps: &Pps,
    sh: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    ctx: &BSliceCtx<'_>,
    prev_qp: &mut i32,
    partition: BPartition,
) -> Result<()> {
    // Pre-zero the MB's MV state. Matches the 4:2:0 path.
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        *info = MbInfo {
            qp_y: *prev_qp,
            coded: true,
            intra: false,
            skipped: false,
            ..Default::default()
        };
    }

    // §7.3.5 / §8.4 — syntax and MV derivation are identical between 4:2:0
    // and 4:4:4. The only differences are (a) how chroma is motion
    // compensated (§8.4.2.2 Table 8-9 — luma filter for ChromaArrayType=3)
    // and (b) residual layout (three plane-aligned luma-style streams).
    // We parse + MV-derive by running the 4:2:0 MC path against a scratch
    // clone, snapshot the resulting MB MV state, then re-do MC on the real
    // 4:4:4 picture with `apply_mvs_444`.
    match partition {
        BPartition::Direct16x16 => {
            let (mvs, dirs) = snapshot_mv_state_via_420(
                sh,
                sps,
                ctx,
                mb_x,
                mb_y,
                pic,
                ref_list0,
                ref_list1,
                |sh2, sps2, ctx2, pic2, rl0, rl1| {
                    direct_16x16_compensate_pub(sh2, sps2, ctx2, mb_x, mb_y, pic2, rl0, rl1)
                },
            )?;
            apply_mvs_444(pic, ref_list0, ref_list1, sh, ctx, mb_x, mb_y, &mvs, &dirs);
        }
        BPartition::L0_16x16 | BPartition::L1_16x16 | BPartition::Bi_16x16 => {
            let dir = match partition {
                BPartition::L0_16x16 => PredDir::L0,
                BPartition::L1_16x16 => PredDir::L1,
                BPartition::Bi_16x16 => PredDir::BiPred,
                _ => unreachable!(),
            };
            handle_16x16_single_dir(br, sh, ctx, mb_x, mb_y, pic, ref_list0, ref_list1, dir)?;
        }
        BPartition::TwoPart16x8 { dirs } => {
            let rects = [(0usize, 0usize, 2usize, 4usize), (2, 0, 2, 4)];
            handle_two_partition(
                br, sh, ctx, mb_x, mb_y, pic, ref_list0, ref_list1, &rects, &dirs,
            )?;
        }
        BPartition::TwoPart8x16 { dirs } => {
            let rects = [(0usize, 0usize, 4usize, 2usize), (0, 2, 4, 2)];
            handle_two_partition(
                br, sh, ctx, mb_x, mb_y, pic, ref_list0, ref_list1, &rects, &dirs,
            )?;
        }
        BPartition::B8x8 => {
            handle_b8x8(br, sh, sps, ctx, mb_x, mb_y, pic, ref_list0, ref_list1)?;
        }
    }

    // CBP / qp_delta / residual for 4:4:4 inter.
    let cbp_raw = br.read_ue()?;
    if cbp_raw > 15 {
        return Err(Error::invalid(format!(
            "h264 b-slice 4:4:4: inter CBP {cbp_raw} > 15"
        )));
    }
    let cbp_luma = GOLOMB_TO_INTER_CBP_GRAY[cbp_raw as usize];
    if pps.transform_8x8_mode_flag && cbp_luma != 0 {
        let flag = br.read_flag()?;
        if flag {
            return Err(Error::unsupported(
                "h264: 4:4:4 B inter transform_size_8x8_flag not wired — 4×4 transform only",
            ));
        }
    }
    let needs_qp = cbp_luma != 0;
    if needs_qp {
        let dqp = br.read_se()?;
        *prev_qp = ((*prev_qp + dqp + 52) % 52).clamp(0, 51);
    }
    let qp_y = *prev_qp;
    let qp_cb = chroma_qp(qp_y, pps.chroma_qp_index_offset);
    let qp_cr = chroma_qp(qp_y, pps.second_chroma_qp_index_offset);
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        info.qp_y = qp_y;
        info.intra4x4_pred_mode = [INTRA_DC_FAKE; 16];
        info.cbp_luma = cbp_luma;
        info.cbp_chroma = 0;
    }
    for (plane, qp, scale_idx) in [
        (mb444::Plane::Y, qp_y, 3usize),
        (mb444::Plane::Cb, qp_cb, 4usize),
        (mb444::Plane::Cr, qp_cr, 5usize),
    ] {
        decode_plane_inter_4x4(br, mb_x, mb_y, pic, plane, cbp_luma, qp, scale_idx)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// MV-state snapshot via a 4:2:0 scratch — lets us reuse `b_mb`'s §8.4.1.2
// direct-mode derivations and partition MVD parse without rewriting them.
// ---------------------------------------------------------------------------

/// Clone `pic`'s MB bookkeeping + scaling + poc state into a 4:2:0-shaped
/// scratch picture, invoke the supplied closure (which reads the bitstream
/// and writes MV state + may call 4:2:0 MC), then snapshot the resulting
/// per-4×4 MV/ref state. The MB MV state is copied back into `pic.mb_info`.
/// Returns `(mvs, dirs)` — ready for `apply_mvs_444` to re-run MC on the
/// true 4:4:4 picture.
#[allow(clippy::too_many_arguments)]
fn snapshot_mv_state_via_420<F>(
    sh: &SliceHeader,
    sps: &Sps,
    ctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    f: F,
) -> Result<(PerBlockMv, [Option<PredDir>; 16])>
where
    F: FnOnce(
        &SliceHeader,
        &Sps,
        &BSliceCtx<'_>,
        &mut Picture,
        &[&Picture],
        &[&Picture],
    ) -> Result<()>,
{
    let mut scratch = make_420_scratch(pic, mb_x, mb_y);
    let refs420_storage_0: Vec<Picture> = ref_list0.iter().map(|r| to_420_reference(r)).collect();
    let refs420_storage_1: Vec<Picture> = ref_list1.iter().map(|r| to_420_reference(r)).collect();
    let refs0: Vec<&Picture> = refs420_storage_0.iter().collect();
    let refs1: Vec<&Picture> = refs420_storage_1.iter().collect();
    f(sh, sps, ctx, &mut scratch, &refs0, &refs1)?;
    // Copy MB info at (mb_x, mb_y) back to pic so subsequent neighbours see
    // the right MVs.
    {
        let src = scratch.mb_info_at(mb_x, mb_y).clone();
        let dst = pic.mb_info_mut(mb_x, mb_y);
        *dst = src;
    }
    let info = pic.mb_info_at(mb_x, mb_y);
    let mut mvs = PerBlockMv::default();
    let mut dirs = [None; 16];
    for i in 0..16 {
        let r0 = info.ref_idx_l0[i];
        let r1 = info.ref_idx_l1[i];
        mvs.ref_l0[i] = r0;
        mvs.ref_l1[i] = r1;
        mvs.mv_l0[i] = info.mv_l0[i];
        mvs.mv_l1[i] = info.mv_l1[i];
        dirs[i] = match (r0 >= 0, r1 >= 0) {
            (true, true) => Some(PredDir::BiPred),
            (true, false) => Some(PredDir::L0),
            (false, true) => Some(PredDir::L1),
            (false, false) => None,
        };
    }
    Ok((mvs, dirs))
}

#[derive(Default, Clone, Copy)]
pub(crate) struct PerBlockMv {
    pub mv_l0: [(i16, i16); 16],
    pub mv_l1: [(i16, i16); 16],
    pub ref_l0: [i8; 16],
    pub ref_l1: [i8; 16],
}

/// Clone a 4:4:4 picture into a 4:2:0-shaped picture that copies over the
/// luma plane verbatim and collapses chroma planes to half-size. The MB
/// bookkeeping (including neighbours' MVs / nc cache) is preserved — MV
/// prediction (§8.4.1.3) needs it intact. Used as a scratch so the 4:2:0
/// MC helpers in [`crate::b_mb`] can run without being called on the
/// 4:4:4 buffers directly.
/// Public wrapper for [`make_420_scratch`] so CABAC 4:4:4 can reuse the
/// scratch clone when driving the shared direct-mode MV compensator.
/// Currently staged for the CABAC B-inter follow-up.
#[allow(dead_code)]
pub(crate) fn make_420_scratch_pub(src: &Picture) -> Picture {
    make_420_scratch(src, 0, 0)
}

/// Public wrapper for [`to_420_reference`]. See
/// [`make_420_scratch_pub`].
#[allow(dead_code)]
pub(crate) fn to_420_reference_pub(src: &Picture) -> Picture {
    to_420_reference(src)
}

fn make_420_scratch(src: &Picture, _mb_x: u32, _mb_y: u32) -> Picture {
    let mut dst = Picture::new_with_format(src.mb_width, src.mb_height, 1);
    dst.mb_info = src.mb_info.clone();
    dst.scaling_lists = src.scaling_lists.clone();
    dst.last_mb_qp_delta_was_nonzero = src.last_mb_qp_delta_was_nonzero;
    // Luma plane: same size, copy over so MC reads the correct samples for
    // the current MB's neighbours. Chroma values stay mid-grey — they're
    // overwritten by the 4:4:4 re-MC pass and never read for MV prediction.
    dst.y.copy_from_slice(&src.y);
    dst
}

/// Build a 4:2:0-shaped reference picture with the same luma plane. Used so
/// b_mb.rs's MC helpers index the correct luma samples; the chroma planes
/// on the scratch pic are ignored (re-done through the 4:4:4 path after).
fn to_420_reference(src: &Picture) -> Picture {
    let mut dst = Picture::new_with_format(src.mb_width, src.mb_height, 1);
    dst.mb_info = src.mb_info.clone();
    dst.y.copy_from_slice(&src.y);
    dst
}

// ---------------------------------------------------------------------------
// Partition handlers — all pass through snapshot_mv_state_via_420.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn handle_16x16_single_dir(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    ctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    dir: PredDir,
) -> Result<()> {
    let num_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let num_l1 = sh.num_ref_idx_l1_active_minus1 + 1;
    let ref_l0 = if uses_l0(dir) {
        Some(read_ref_idx(br, num_l0)?)
    } else {
        None
    };
    let ref_l1 = if uses_l1(dir) {
        Some(read_ref_idx(br, num_l1)?)
    } else {
        None
    };
    let mvd_l0 = if uses_l0(dir) {
        Some((br.read_se()?, br.read_se()?))
    } else {
        None
    };
    let mvd_l1 = if uses_l1(dir) {
        Some((br.read_se()?, br.read_se()?))
    } else {
        None
    };
    compensate_444_partition(
        sh, ctx, pic, ref_list0, ref_list1, mb_x, mb_y, 0, 0, 4, 4, dir, ref_l0, ref_l1, mvd_l0,
        mvd_l1,
    )
}

#[allow(clippy::too_many_arguments)]
fn handle_two_partition(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    ctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    rects: &[(usize, usize, usize, usize); 2],
    dirs: &[PredDir; 2],
) -> Result<()> {
    let num_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let num_l1 = sh.num_ref_idx_l1_active_minus1 + 1;
    let mut ref_l0 = [None; 2];
    let mut ref_l1 = [None; 2];
    let mut mvd_l0 = [None; 2];
    let mut mvd_l1 = [None; 2];
    for p in 0..2 {
        if uses_l0(dirs[p]) {
            ref_l0[p] = Some(read_ref_idx(br, num_l0)?);
        }
    }
    for p in 0..2 {
        if uses_l1(dirs[p]) {
            ref_l1[p] = Some(read_ref_idx(br, num_l1)?);
        }
    }
    for p in 0..2 {
        if uses_l0(dirs[p]) {
            mvd_l0[p] = Some((br.read_se()?, br.read_se()?));
        }
    }
    for p in 0..2 {
        if uses_l1(dirs[p]) {
            mvd_l1[p] = Some((br.read_se()?, br.read_se()?));
        }
    }
    for p in 0..2 {
        let (r0, c0, ph, pw) = rects[p];
        compensate_444_partition(
            sh, ctx, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, ph, pw, dirs[p], ref_l0[p],
            ref_l1[p], mvd_l0[p], mvd_l1[p],
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_b8x8(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    sps: &Sps,
    ctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
) -> Result<()> {
    const SUB_OFFSETS: [(usize, usize); 4] = [(0, 0), (0, 2), (2, 0), (2, 2)];
    let num_l0 = sh.num_ref_idx_l0_active_minus1 + 1;
    let num_l1 = sh.num_ref_idx_l1_active_minus1 + 1;

    let mut subs = [BSubPartition::Direct8x8; 4];
    for s in 0..4 {
        let v = br.read_ue()?;
        subs[s] = decode_b_sub_mb_type(v)
            .ok_or_else(|| Error::invalid(format!("h264 b-slice 4:4:4: bad sub_mb_type {v}")))?;
    }
    let mut ref_l0 = [None; 4];
    let mut ref_l1 = [None; 4];
    for s in 0..4 {
        if let Some(dir) = subs[s].pred_dir() {
            if uses_l0(dir) {
                ref_l0[s] = Some(read_ref_idx(br, num_l0)?);
            }
        }
    }
    for s in 0..4 {
        if let Some(dir) = subs[s].pred_dir() {
            if uses_l1(dir) {
                ref_l1[s] = Some(read_ref_idx(br, num_l1)?);
            }
        }
    }
    #[derive(Clone, Copy, Default)]
    struct SubMvd {
        mvd_l0: Option<(i32, i32)>,
        mvd_l1: Option<(i32, i32)>,
    }
    let mut mvds: Vec<Vec<SubMvd>> = Vec::with_capacity(4);
    for s in 0..4 {
        let sp = subs[s];
        let n = sp.num_sub_partitions();
        let mut entry = Vec::with_capacity(n);
        let dir_opt = sp.pred_dir();
        for _ in 0..n {
            if let Some(dir) = dir_opt {
                let mvd_l0 = if uses_l0(dir) {
                    Some((br.read_se()?, br.read_se()?))
                } else {
                    None
                };
                let mvd_l1 = if uses_l1(dir) {
                    Some((br.read_se()?, br.read_se()?))
                } else {
                    None
                };
                entry.push(SubMvd { mvd_l0, mvd_l1 });
            } else {
                entry.push(SubMvd::default());
            }
        }
        mvds.push(entry);
    }
    for s in 0..4 {
        let sp = subs[s];
        let (sr0, sc0) = SUB_OFFSETS[s];
        if matches!(sp, BSubPartition::Direct8x8) {
            // Direct 8×8 — run `direct_8x8_compensate_pub` through a scratch
            // picture and snapshot MVs, same pattern as 16×16 direct.
            let (mvs, dirs) = snapshot_mv_state_via_420(
                sh,
                sps,
                ctx,
                mb_x,
                mb_y,
                pic,
                ref_list0,
                ref_list1,
                |sh2, sps2, ctx2, pic2, rl0, rl1| {
                    direct_8x8_compensate_pub(sh2, sps2, ctx2, mb_x, mb_y, sr0, sc0, pic2, rl0, rl1)
                },
            )?;
            apply_mvs_444_rect(
                pic, ref_list0, ref_list1, sh, ctx, mb_x, mb_y, sr0, sc0, 2, 2, &mvs, &dirs,
            );
            continue;
        }
        let dir = sp.pred_dir().expect("non-direct");
        for sub_idx in 0..sp.num_sub_partitions() {
            let (dr, dc, sh_h, sh_w) = sp.sub_rect(sub_idx);
            let r0 = sr0 + dr;
            let c0 = sc0 + dc;
            let m = mvds[s][sub_idx];
            compensate_444_partition(
                sh, ctx, pic, ref_list0, ref_list1, mb_x, mb_y, r0, c0, sh_h, sh_w, dir, ref_l0[s],
                ref_l1[s], m.mvd_l0, m.mvd_l1,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compensate_444_partition(
    sh: &SliceHeader,
    ctx: &BSliceCtx<'_>,
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    ph: usize,
    pw: usize,
    dir: PredDir,
    ref_idx_l0: Option<i8>,
    ref_idx_l1: Option<i8>,
    mvd_l0: Option<(i32, i32)>,
    mvd_l1: Option<(i32, i32)>,
) -> Result<()> {
    // Derive MVs via the shared §8.4.1.3 predictor.
    let mv_l0 = if let (Some(r), Some(d)) = (ref_idx_l0, mvd_l0) {
        let pmv = predict_mv_list(pic, mb_x, mb_y, r0, c0, ph, pw, r, 0);
        Some((
            pmv.0.wrapping_add(d.0 as i16),
            pmv.1.wrapping_add(d.1 as i16),
        ))
    } else {
        None
    };
    let mv_l1 = if let (Some(r), Some(d)) = (ref_idx_l1, mvd_l1) {
        let pmv = predict_mv_list(pic, mb_x, mb_y, r0, c0, ph, pw, r, 1);
        Some((
            pmv.0.wrapping_add(d.0 as i16),
            pmv.1.wrapping_add(d.1 as i16),
        ))
    } else {
        None
    };
    {
        let info = pic.mb_info_mut(mb_x, mb_y);
        // §9.3.3.1.1.7 — subsequent partitions' MVD ctxIdxInc needs
        // `mvd_l{0,1}_abs` populated for already-decoded blocks of this
        // MB (same layout FFmpeg tracks via `mvd_cache[list]`).
        let abs0 = mvd_l0.map(|(x, y)| {
            (
                (x.unsigned_abs()).min(u16::MAX as u32) as u16,
                (y.unsigned_abs()).min(u16::MAX as u32) as u16,
            )
        });
        let abs1 = mvd_l1.map(|(x, y)| {
            (
                (x.unsigned_abs()).min(u16::MAX as u32) as u16,
                (y.unsigned_abs()).min(u16::MAX as u32) as u16,
            )
        });
        for rr in r0..r0 + ph {
            for cc in c0..c0 + pw {
                let idx = rr * 4 + cc;
                if let (Some(r), Some(mv)) = (ref_idx_l0, mv_l0) {
                    info.ref_idx_l0[idx] = r;
                    info.mv_l0[idx] = mv;
                }
                if let (Some(r), Some(mv)) = (ref_idx_l1, mv_l1) {
                    info.ref_idx_l1[idx] = r;
                    info.mv_l1[idx] = mv;
                }
                if let Some(a) = abs0 {
                    info.mvd_l0_abs[idx] = a;
                }
                if let Some(a) = abs1 {
                    info.mvd_l1_abs[idx] = a;
                }
            }
        }
    }
    // Apply 4:4:4 motion compensation directly on `pic`.
    mc_partition_444(
        pic, ref_list0, ref_list1, sh, ctx, mb_x, mb_y, r0, c0, ph, pw, dir, ref_idx_l0,
        ref_idx_l1, mv_l0, mv_l1,
    );
    Ok(())
}

/// Apply collected per-4×4-block MVs to all three planes of a 4:4:4
/// picture. Used after `snapshot_mv_state_via_420` produced the MV grid
/// from a direct-mode derivation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_mvs_444(
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    sh: &SliceHeader,
    ctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    mvs: &PerBlockMv,
    dirs: &[Option<PredDir>; 16],
) {
    apply_mvs_444_rect(
        pic, ref_list0, ref_list1, sh, ctx, mb_x, mb_y, 0, 0, 4, 4, mvs, dirs,
    )
}

/// Public wrapper for [`apply_mvs_444_rect`] — exposed so the CABAC
/// 4:4:4 B path can run MC on a single 8×8 sub-MB rect (used by the
/// direct-8×8 helper after snapshotting the MV grid).
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_mvs_444_rect_pub(
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    sh: &SliceHeader,
    ctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    h: usize,
    w: usize,
    mvs: &PerBlockMv,
    dirs: &[Option<PredDir>; 16],
) {
    apply_mvs_444_rect(
        pic, ref_list0, ref_list1, sh, ctx, mb_x, mb_y, r0, c0, h, w, mvs, dirs,
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_mvs_444_rect(
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    sh: &SliceHeader,
    ctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    h: usize,
    w: usize,
    mvs: &PerBlockMv,
    dirs: &[Option<PredDir>; 16],
) {
    // Walk the rect in 4×4 blocks and apply MC per block. This is the
    // pessimistic path — direct mode grids the MC per block, so we do the
    // same. Each 4×4 run of identical (dir, ref, mv) could be coalesced,
    // but the overhead is negligible for 64×64 fixtures.
    for rr in r0..r0 + h {
        for cc in c0..c0 + w {
            let i = rr * 4 + cc;
            let dir = dirs[i].unwrap_or(PredDir::L0);
            let r0 = mvs.ref_l0[i];
            let r1 = mvs.ref_l1[i];
            let m0 = mvs.mv_l0[i];
            let m1 = mvs.mv_l1[i];
            mc_partition_444(
                pic,
                ref_list0,
                ref_list1,
                sh,
                ctx,
                mb_x,
                mb_y,
                rr,
                cc,
                1,
                1,
                dir,
                if r0 >= 0 { Some(r0) } else { None },
                if r1 >= 0 { Some(r1) } else { None },
                if r0 >= 0 { Some(m0) } else { None },
                if r1 >= 0 { Some(m1) } else { None },
            );
        }
    }
}

/// 4:4:4 motion compensation for a single partition. Uses the luma 6-tap
/// filter on all three planes. Handles L0 / L1 / BiPred.
#[allow(clippy::too_many_arguments)]
fn mc_partition_444(
    pic: &mut Picture,
    ref_list0: &[&Picture],
    ref_list1: &[&Picture],
    sh: &SliceHeader,
    ctx: &BSliceCtx<'_>,
    mb_x: u32,
    mb_y: u32,
    r0: usize,
    c0: usize,
    h: usize,
    w: usize,
    dir: PredDir,
    ref_idx_l0: Option<i8>,
    ref_idx_l1: Option<i8>,
    mv_l0: Option<(i16, i16)>,
    mv_l1: Option<(i16, i16)>,
) {
    let pw = w * 4;
    let ph = h * 4;
    let base_x = (mb_x as i32) * 16 + (c0 as i32) * 4;
    let base_y = (mb_y as i32) * 16 + (r0 as i32) * 4;
    let stride = pic.luma_stride();
    let off = pic.luma_off(mb_x, mb_y) + r0 * 4 * stride + c0 * 4;

    fn get_plane(r: &Picture, p: mb444::Plane) -> &[u8] {
        match p {
            mb444::Plane::Y => &r.y,
            mb444::Plane::Cb => &r.cb,
            mb444::Plane::Cr => &r.cr,
        }
    }

    let r0p = ref_idx_l0.and_then(|r| {
        if (r as usize) < ref_list0.len() {
            Some(ref_list0[r as usize])
        } else {
            None
        }
    });
    let r1p = ref_idx_l1.and_then(|r| {
        if (r as usize) < ref_list1.len() {
            Some(ref_list1[r as usize])
        } else {
            None
        }
    });

    for plane in [mb444::Plane::Y, mb444::Plane::Cb, mb444::Plane::Cr] {
        let mut tmp0 = vec![0u8; pw * ph];
        let mut tmp1 = vec![0u8; pw * ph];
        match dir {
            PredDir::L0 => {
                if let (Some(rp), Some(mv)) = (r0p, mv_l0) {
                    luma_mc_plane(
                        &mut tmp0,
                        get_plane(rp, plane),
                        rp.luma_stride(),
                        rp.width as i32,
                        rp.height as i32,
                        base_x,
                        base_y,
                        mv.0 as i32,
                        mv.1 as i32,
                        pw,
                        ph,
                    );
                    if let Some(lw) = list_luma_weight(sh, 0, ref_idx_l0.unwrap()) {
                        apply_luma_weight(&mut tmp0, lw);
                    }
                    write_plane(pic, plane, off, stride, &tmp0, pw, ph);
                }
            }
            PredDir::L1 => {
                if let (Some(rp), Some(mv)) = (r1p, mv_l1) {
                    luma_mc_plane(
                        &mut tmp1,
                        get_plane(rp, plane),
                        rp.luma_stride(),
                        rp.width as i32,
                        rp.height as i32,
                        base_x,
                        base_y,
                        mv.0 as i32,
                        mv.1 as i32,
                        pw,
                        ph,
                    );
                    if let Some(lw) = list_luma_weight(sh, 1, ref_idx_l1.unwrap()) {
                        apply_luma_weight(&mut tmp1, lw);
                    }
                    write_plane(pic, plane, off, stride, &tmp1, pw, ph);
                }
            }
            PredDir::BiPred => {
                if let (Some(rp0), Some(mv0), Some(rp1), Some(mv1)) = (r0p, mv_l0, r1p, mv_l1) {
                    luma_mc_plane(
                        &mut tmp0,
                        get_plane(rp0, plane),
                        rp0.luma_stride(),
                        rp0.width as i32,
                        rp0.height as i32,
                        base_x,
                        base_y,
                        mv0.0 as i32,
                        mv0.1 as i32,
                        pw,
                        ph,
                    );
                    luma_mc_plane(
                        &mut tmp1,
                        get_plane(rp1, plane),
                        rp1.luma_stride(),
                        rp1.width as i32,
                        rp1.height as i32,
                        base_x,
                        base_y,
                        mv1.0 as i32,
                        mv1.1 as i32,
                        pw,
                        ph,
                    );
                    let r0_idx = ref_idx_l0.unwrap();
                    let r1_idx = ref_idx_l1.unwrap();
                    let (mode, implicit) = resolve_bipred_mode(ctx, r0_idx, r1_idx);
                    let mut out = vec![0u8; pw * ph];
                    match mode {
                        BipredMode::Explicit => {
                            let pwt = sh.pred_weight_table.as_ref().unwrap();
                            // §8.4.2.3.2 — same weights applied on every
                            // plane under the luma-filter route.
                            let (lw0, lw1) = list_luma_weights_for_bi(pwt, r0_idx, r1_idx);
                            weighted_bipred_plane(
                                &mut out,
                                &tmp0,
                                &tmp1,
                                lw0.weight,
                                lw0.offset,
                                lw1.weight,
                                lw1.offset,
                                pwt.luma_log2_weight_denom,
                            );
                        }
                        BipredMode::Implicit => match implicit {
                            ImplicitBiWeights::Weighted { w0, w1 } => {
                                implicit_weighted_bipred_plane(&mut out, &tmp0, &tmp1, w0, w1);
                            }
                            ImplicitBiWeights::Default => {
                                average_bipred(&mut out, &tmp0, &tmp1);
                            }
                        },
                        BipredMode::Default => {
                            average_bipred(&mut out, &tmp0, &tmp1);
                        }
                    }
                    write_plane(pic, plane, off, stride, &out, pw, ph);
                }
            }
        }
    }
    let _ = (apply_motion_compensation_b, PredWeightTable::default);
}

fn write_plane(
    pic: &mut Picture,
    plane: mb444::Plane,
    off: usize,
    stride: usize,
    src: &[u8],
    pw: usize,
    ph: usize,
) {
    let dst: &mut Vec<u8> = match plane {
        mb444::Plane::Y => &mut pic.y,
        mb444::Plane::Cb => &mut pic.cb,
        mb444::Plane::Cr => &mut pic.cr,
    };
    for rr in 0..ph {
        for cc in 0..pw {
            dst[off + rr * stride + cc] = src[rr * pw + cc];
        }
    }
}

fn resolve_bipred_mode(ctx: &BSliceCtx<'_>, r0: i8, r1: i8) -> (BipredMode, ImplicitBiWeights) {
    match ctx.bipred_mode {
        BipredMode::Implicit => {
            let (weights, ok) = match (
                ctx.list0_pocs.get(r0.max(0) as usize),
                ctx.list1_pocs.get(r1.max(0) as usize),
                ctx.list0_short_term.get(r0.max(0) as usize),
                ctx.list1_short_term.get(r1.max(0) as usize),
            ) {
                (Some(&p0), Some(&p1), Some(&s0), Some(&s1)) => {
                    (implicit_bipred_weights(ctx.curr_poc, p0, p1, s0, s1), true)
                }
                _ => (ImplicitBiWeights::Default, false),
            };
            let _ = ok;
            (BipredMode::Implicit, weights)
        }
        other => (other, ImplicitBiWeights::Default),
    }
}

fn list_luma_weights_for_bi(pwt: &PredWeightTable, r0: i8, r1: i8) -> (LumaWeight, LumaWeight) {
    let idx0 = r0.max(0) as usize;
    let idx1 = r1.max(0) as usize;
    let lw0 = pwt.luma_l0.get(idx0).copied().unwrap_or_default();
    let lw1 = pwt.luma_l1.get(idx1).copied().unwrap_or_default();
    (lw0, lw1)
}

fn list_luma_weight<'a>(sh: &'a SliceHeader, list: u8, ref_idx: i8) -> Option<&'a LumaWeight> {
    let tbl = sh.pred_weight_table.as_ref()?;
    let idx = ref_idx.max(0) as usize;
    if list == 0 {
        tbl.luma_l0.get(idx)
    } else {
        tbl.luma_l1.get(idx)
    }
}

fn read_ref_idx(br: &mut BitReader<'_>, num_ref: u32) -> Result<i8> {
    if num_ref == 0 || num_ref == 1 {
        return Ok(0);
    }
    if num_ref == 2 {
        let bit = br.read_u1()?;
        return Ok(if bit == 1 { 0 } else { 1 });
    }
    Ok(br.read_ue()? as i8)
}

fn uses_l0(d: PredDir) -> bool {
    matches!(d, PredDir::L0 | PredDir::BiPred)
}
fn uses_l1(d: PredDir) -> bool {
    matches!(d, PredDir::L1 | PredDir::BiPred)
}

// ---------------------------------------------------------------------------
// Residual — identical shape to 4:4:4 P (three luma-style streams per MB).
// ---------------------------------------------------------------------------

fn decode_plane_inter_4x4(
    br: &mut BitReader<'_>,
    mb_x: u32,
    mb_y: u32,
    pic: &mut Picture,
    plane: mb444::Plane,
    cbp_luma: u8,
    qp: i32,
    scale_idx: usize,
) -> Result<()> {
    let lo_mb = pic.luma_off(mb_x, mb_y);
    for blk in 0..16usize {
        let (br_row, br_col) = LUMA_BLOCK_RASTER[blk];
        let eight_idx = (br_row / 2) * 2 + (br_col / 2);
        if (cbp_luma >> eight_idx) & 1 == 0 {
            continue;
        }
        let nc = mb444::predict_nc_plane(pic, mb_x, mb_y, plane, br_row, br_col);
        let blk = decode_residual_block(br, nc, BlockKind::Luma4x4)?;
        let mut residual = blk.coeffs;
        let total_coeff = blk.total_coeff;
        let scale = *pic.scaling_lists.matrix_4x4(scale_idx);
        dequantize_4x4_scaled(&mut residual, qp, &scale);
        idct_4x4(&mut residual);
        {
            let (buf, s) = mb444::plane_samples_mut(pic, plane);
            let lo = lo_mb + br_row * 4 * s + br_col * 4;
            for r in 0..4 {
                for c in 0..4 {
                    let base = buf[lo + r * s + c] as i32;
                    buf[lo + r * s + c] = (base + residual[r * 4 + c]).clamp(0, 255) as u8;
                }
            }
        }
        let info = pic.mb_info_mut(mb_x, mb_y);
        mb444::plane_nc_at_mut(info, plane)[br_row * 4 + br_col] = total_coeff as u8;
    }
    Ok(())
}
