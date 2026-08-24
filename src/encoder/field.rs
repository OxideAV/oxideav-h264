//! Round-416 — PAFF (picture-adaptive frame/field) encoding driver.
//!
//! Encodes an interlaced sequence as **field pictures**
//! (`field_pic_flag == 1`, §7.4.3): every input frame is split into its
//! top field (even sample rows) and bottom field (odd sample rows), and
//! each field is coded as an independent half-height coded picture
//! (§7.4.2.1.1 eq. 7-26: `PicHeightInMbs = FrameHeightInMbs / 2`).
//! Optionally, selected frames are instead coded as full-height FRAME
//! pictures (`field_pic_flag == 0`) inside the same interlaced stream —
//! the "picture-adaptive" axis.
//!
//! Structure of the emitted stream (Annex B):
//!
//! * SPS: Main profile (77 — §A.2.1 bars interlace from Baseline),
//!   `frame_mbs_only_flag = 0`, `mb_adaptive_frame_field_flag = 0`,
//!   CAVLC, 4:2:0, `pic_order_cnt_type = 0`.
//! * Frame 0: IDR **top field** + non-IDR I **bottom field**. Per
//!   §7.4.3 the two fields of the frame share `frame_num` (= 0; the
//!   second field's `frame_num == PrevRefFrameNum` case is sanctioned
//!   because the preceding reference picture is an IDR field of
//!   opposite parity).
//! * Frame k > 0: either an I/I field pair, a P/P field pair, or an I
//!   FRAME picture. Every picture is a reference (`nal_ref_idc != 0`,
//!   sliding-window marking), `frame_num = k` for both fields.
//! * POC (type 0): `pic_order_cnt_lsb = 2k` for the top field / frame
//!   picture, `2k + 1` for the bottom field, so output order matches
//!   the §8.2.1 eq. 8-1 frame POC = Min(TopFOC, BottomFOC) = 2k.
//!
//! P fields use a single active reference (`num_ref_idx_l0_active = 1`
//! from the PPS default), so every coded `ref_idx` is 0 and no ref_idx
//! bits appear. Per the §8.2.4.2.5 field reference-list initialisation
//! (fields of the ordered reference frames, alternating parity starting
//! with the SAME parity as the current field), `RefPicList0[0]` for the
//! first AND second field of frame k is the **same-parity field of
//! frame k − 1** — which is exactly the reference the per-field motion
//! search here uses. The §8.4 decoding process then runs entirely in
//! field coordinates (half-height planes, field MVs), so the existing
//! frame-based MB encoder applies unchanged to each field.
//!
//! The per-field in-loop deblock runs with `field_pic = 1` so the
//! §8.7.2.1 field rules apply (horizontal intra MB edges take bS = 3;
//! the bS = 4 first bullet requires frame macroblocks or a vertical
//! edge).
//!
//! Scope (fixture-grade, mirrors what the staged PAFF fixtures pin):
//! CAVLC, 4:2:0, top-field-first, frame pictures only in all-I
//! sequences. P fields reference the same-parity field of the previous
//! frame (plus the round-416 cross-parity + frame-reference axes);
//! round-436 adds non-reference **B field pairs** ([`PaffConfig::
//! b_fields`]) with spatial or temporal direct derivation.

use crate::encoder::deblock::{
    deblock_recon_field, deblock_recon_with_chroma_array_type, MbDeblockInfo,
};
use crate::encoder::nal::build_nal_unit;
use crate::encoder::pps::{build_baseline_pps_rbsp, BaselinePpsConfig};
use crate::encoder::slice::{
    write_b_slice_header, write_idr_i_slice_header, write_p_slice_header, BSliceHeaderConfig,
    EncMmcoOp, EncRplmOp, FieldPicSignal, IdrSliceHeaderConfig, PSliceHeaderConfig,
};
use crate::encoder::sps::{build_baseline_sps_rbsp, BaselineSpsConfig};
use crate::encoder::{
    min_level_idc_for_picture_size, EncodedFrameRef, Encoder, EncoderConfig, FrameRefPartitionMv,
    YuvFrame,
};
use crate::encoder::{BitWriter, CavlcNcGrid, IntraGrid, MvGrid};
use crate::nal::NalUnitType;
use crate::transform::{qp_bd_offset, qp_y_to_qp_c_with_bd_offset};

/// Configuration for [`encode_paff_sequence`].
#[derive(Debug, Clone)]
pub struct PaffConfig {
    /// Luma width in samples (multiple of 16).
    pub width: u32,
    /// FRAME luma height in samples (multiple of 32 so each field is
    /// MB-aligned).
    pub frame_height: u32,
    /// Slice QP_Y (coded as `pic_init_qp_minus26`, `slice_qp_delta=0`).
    pub qp: i32,
    /// When `true`, fields after frame 0 are coded as P fields whose
    /// single L0 reference is the same-parity field of the previous
    /// frame; when `false` every picture is intra.
    pub p_fields: bool,
    /// Frame indices to code as full-height I FRAME pictures
    /// (`field_pic_flag = 0`) instead of a field pair. Must be empty
    /// when `p_fields` is set and must not contain 0 (frame 0 is the
    /// IDR field pair).
    pub frame_picture_indices: Vec<usize>,
    /// Round-416 — §8.4.1.4 Table 8-10 axis: when `true` (requires
    /// `p_fields`), the bottom field of frame 0 is coded as a P field
    /// whose single reference is the IDR TOP field — an
    /// OPPOSITE-parity field reference (the only candidate the
    /// §8.2.4.2.5 init can offer a second field whose same-parity list
    /// is empty). The chroma predictor then applies the Table 8-10
    /// `mvCLX[1] = mvLX[1] + 2` adjustment (top reference, bottom
    /// current field).
    pub cross_parity_first_bottom: bool,
    /// Round-416 — frame-reference axis: when `true` (requires
    /// `p_fields`), frame 0 is coded as an **IDR full-height FRAME
    /// picture** (`field_pic_flag = 0`) and frame 1's P fields
    /// reference the parity fields OF THAT FRAME — per §8.2.4.2.5 a
    /// stored frame supplies either parity field as a distinct
    /// reference picture, which a decoder serves as a half-height
    /// field view of the stored frame.
    pub idr_frame_first: bool,
    /// Round-436 — **B field** axis (requires `p_fields`; incompatible
    /// with the other optional axes). Even display indices code as
    /// anchor field pairs (frame 0 = the IDR/I pair, later anchors =
    /// P/P pairs referencing the previous anchor), odd display indices
    /// code as **non-reference B/B field pairs** emitted AFTER the
    /// following anchor (coding order 0, 2, 1, 4, 3, …). Per the
    /// §8.2.4.2.4 + §8.2.4.2.5 B-field list initialisation (reference
    /// entries ordered by field POC around the current field, parities
    /// alternating starting from the current field's own), each B
    /// field finds the **same-parity field of the previous anchor at
    /// `RefPicList0[0]`** and the **same-parity field of the next
    /// anchor at `RefPicList1[0]`** — exactly the two references the
    /// per-field mode decision uses. A trailing odd display frame
    /// (no following anchor) codes as a P/P pair.
    pub b_fields: bool,
    /// Round-436 — when `true` (requires `b_fields`), the B field
    /// slices signal `direct_spatial_mv_pred_flag = 0` and the
    /// B_Skip / B_Direct_16x16 MVs come from the §8.4.1.2.3
    /// **temporal direct** derivation: colPic is `RefPicList1[0]` (the
    /// same-parity field of the next anchor, §8.4.1.2.1 — a coded
    /// field, so its motion grid is read in field coordinates), and
    /// every DiffPicOrderCnt of eq. 8-201/8-202 runs on the fields'
    /// own §8.2.1 order counts. When `false`, B fields use the
    /// §8.4.1.2.2 spatial direct derivation.
    pub b_temporal_direct: bool,
    /// Round-436 — §8.2.5 **long-term anchor** axis (requires
    /// `p_fields`; incompatible with the other optional axes). Frame
    /// 1's TOP P field promotes BOTH fields of frame 0 to long-term
    /// with a §8.2.5.4.4 MMCO 4 (MaxLongTermFrameIdx = 0) followed by
    /// two §8.2.5.4.3 field MMCO 3 ops — first the same-parity frame-0
    /// top field (eq. 8-30 PicNum 1 ⇒ diff_minus1 1), then the
    /// opposite-parity bottom field (eq. 8-31 PicNum 0 ⇒ diff_minus1
    /// 2; its eviction pre-pass must SPARE the just-promoted top
    /// field, which holds the same LongTermFrameIdx but belongs to the
    /// pair including the new target). From frame 2 on, every P field
    /// references the same-parity long-term frame-0 field, spliced to
    /// `ref_idx` 0 by a §8.2.4.3.2 RPLM `long_term_pic_num` op
    /// (eq. 8-32 same-parity LongTermPicNum = 1), while the short-term
    /// P pairs keep sliding through the §8.2.5.3 window — pinning
    /// per-field long-term promotion, the §8.2.5.4.3 pair exception,
    /// per-field LongTermPicNum arithmetic, the refFrameListLongTerm
    /// §8.2.4.2.5 interleave and long-term RPLM splicing.
    pub long_term_anchor: bool,
    /// Round-436 — §8.2.5.4.1 **field MMCO 1** axis (requires
    /// `p_fields`; incompatible with the other optional axes). Frame
    /// 1's BOTTOM P field carries an MMCO 1 unmarking the frame-1 TOP
    /// field (the first field of its own complementary pair —
    /// eq. 8-39 with CurrPicNum = 3, opposite-parity PicNum = 2 ⇒
    /// difference_of_pic_nums_minus1 = 0). Frame 2's TOP field then
    /// finds no short-term same-parity field in frame 1 and the
    /// §8.2.4.2.5 alternation serves the FRAME-0 top field at
    /// `RefPicList0[0]` (the "missing field is ignored" rule) — the
    /// encoder codes it against frame 0's top field recon, so a
    /// decoder that ignores the per-field unmarking mispredicts every
    /// inter MB of that field.
    pub mmco_unpair_first_top: bool,
    /// Round-436 — enable the **8x8 transform** in the field pictures
    /// (`EncoderConfig::transform_8x8`; SPS auto-promotes to High,
    /// PPS codes `transform_8x8_mode_flag = 1`). Every MB of a field
    /// picture is a field MB, so the CAVLC 8x8 luma coefficients are
    /// emitted in the §8.5.7 Table 8-14 **FIELD scan** (via the
    /// split-pipeline pre-composed scan,
    /// [`crate::encoder::transform::field_scan_8x8_for_cavlc_split`])
    /// — I_8x8 gets a real 3-way intra RDO, P/B MBs run the
    /// 8x8-vs-4x4 inter residual trial.
    pub transform_8x8: bool,
    /// Round-440 — §8.4.2.3.3 **implicit weighted prediction** axis
    /// (requires `b_fields`). The PPS signals
    /// `weighted_bipred_idc = 2` and anchors move to a stride-3
    /// display layout (anchor, B, B, anchor, …) so the two
    /// non-reference B field pairs between anchors sit at UNEQUAL
    /// per-field POC distances: every bipred / direct-Bi macroblock of
    /// a B field combines its two predictions with the
    /// eq. 8-197/8-198 + 8-282/8-283 POC-derived (w0, w1) pair — at
    /// logWD = 5, zero offsets — computed on the FIELDS' own §8.2.1
    /// order counts (first B pair w0/w1 = 43/21, second = 22/42; a
    /// stride-2 layout would collapse every pair to the trivial
    /// 32/32). Applies to luma AND chroma per §8.4.2.3.3.
    pub b_implicit_weight: bool,
    /// Round-440 — **B reference fields** axis (requires `b_fields`).
    /// Anchors move to a stride-4 display layout with a REFERENCE
    /// B field pair midway (`nal_ref_idc = 2`, stored through the
    /// §8.2.5.3 sliding window as a complementary reference field
    /// pair) and two non-reference B pairs on either side:
    /// display/coding order 0, 4, 2ref, 1, 3, 8, 6ref, 5, 7, …. The
    /// non-reference B pairs reference the B fields themselves — the
    /// pair before the reference-B finds it at `RefPicList1[0]`
    /// (making a coded B FIELD the §8.4.1.2.1 colPic of a
    /// temporal-direct derivation), the pair after finds it at
    /// `RefPicList0[0]`. The reference-B pair's mode decision is
    /// restricted to L0/Bi/intra (`EncoderConfig::b_l0_bi_only`) so
    /// its stored L0 motion snapshot matches what a decoder's
    /// co-located read returns.
    pub b_reference_fields: bool,
}

/// A reconstructed reference field: (Y, Cb, Cr) half-height planes +
/// the field's own picture order count.
type ReconField = (Vec<u8>, Vec<u8>, Vec<u8>, i32);

/// Round-436 — a reconstructed anchor field used as a B-field
/// reference: half-height planes, the field's own §8.2.1 order count,
/// and the per-8x8 L0 motion snapshot the §8.4.1.2 direct derivations
/// read from the colocated picture (`RefPicList1[0]`).
struct AnchorField {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    poc: i32,
    partition_mvs: Vec<FrameRefPartitionMv>,
}

impl AnchorField {
    fn as_frame_ref(&self, width: u32, field_h: u32) -> EncodedFrameRef<'_> {
        EncodedFrameRef {
            width,
            height: field_h,
            recon_y: &self.y,
            recon_u: &self.u,
            recon_v: &self.v,
            partition_mvs: &self.partition_mvs,
            pic_order_cnt: self.poc,
        }
    }
}

/// Round-436 — snapshot the per-8x8 L0 motion state of an encoded
/// picture from its `MvGrid`, in the shape `EncodedFrameRef::
/// partition_mvs` expects (mirrors the `encode_p` / `encode_b` export:
/// unencoded or intra slots become `refIdxCol = -1` "intra" entries per
/// the §8.4.1.2.1 colocated probe).
fn partition_mvs_from_grid(mv_grid: &MvGrid) -> Vec<FrameRefPartitionMv> {
    let mut out = Vec::with_capacity(mv_grid.slots.len() * 4);
    for slot in &mv_grid.slots {
        for part in 0..4usize {
            let (mv, ref_idx, is_intra) = if !slot.available || slot.is_intra {
                (crate::mv_deriv::Mv::ZERO, -1, true)
            } else {
                (slot.mv_l0_8x8[part], slot.ref_idx_l0_8x8[part], false)
            };
            out.push(FrameRefPartitionMv {
                mv_l0: (
                    mv.x.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                    mv.y.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                ),
                ref_idx_l0: ref_idx.clamp(-1, 127) as i8,
                is_intra,
            });
        }
    }
    out
}

/// One encoded PAFF sequence: the Annex B stream plus the full-height
/// per-frame reconstruction (fields re-interleaved, §8.7 post-filter)
/// that a conformant decoder outputs.
pub struct PaffEncoded {
    pub annex_b: Vec<u8>,
    /// Per input frame: (Y, Cb, Cr) full-height 4:2:0 planes.
    pub recon_frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
}

/// Split a full-height 4:2:0 frame into one parity's field planes
/// (§6.4.1: top field = even sample rows, bottom field = odd rows; the
/// 4:2:0 chroma rows split by the same parity).
fn extract_field(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    frame_height: usize,
    bottom: bool,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = width / 2;
    let ch = frame_height / 2;
    let start = usize::from(bottom);
    let mut fy = Vec::with_capacity(width * frame_height / 2);
    for row in (start..frame_height).step_by(2) {
        fy.extend_from_slice(&y[row * width..(row + 1) * width]);
    }
    let mut fu = Vec::with_capacity(cw * ch / 2);
    let mut fv = Vec::with_capacity(cw * ch / 2);
    for row in (start..ch).step_by(2) {
        fu.extend_from_slice(&u[row * cw..(row + 1) * cw]);
        fv.extend_from_slice(&v[row * cw..(row + 1) * cw]);
    }
    (fy, fu, fv)
}

/// Re-interleave two half-height field plane sets into one full-height
/// frame (top → even rows, bottom → odd rows), mirroring the decoder's
/// §C.4.4 output pairing.
fn interleave_planes(top: &[u8], bottom: &[u8], width: usize) -> Vec<u8> {
    debug_assert_eq!(top.len(), bottom.len());
    let field_rows = top.len() / width;
    let mut out = vec![0u8; top.len() * 2];
    for r in 0..field_rows {
        out[(2 * r) * width..(2 * r + 1) * width].copy_from_slice(&top[r * width..(r + 1) * width]);
        out[(2 * r + 1) * width..(2 * r + 2) * width]
            .copy_from_slice(&bottom[r * width..(r + 1) * width]);
    }
    out
}

/// Encode one intra picture's `slice_data()` bits into `sw` using the
/// supplied picture-sized encoder, returning the pre-deblock recon
/// planes + per-MB deblock facts. Mirrors the `encode_idr` MB loop.
fn encode_i_slice_data(
    enc: &Encoder,
    src: &YuvFrame<'_>,
    sw: &mut BitWriter,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<MbDeblockInfo>) {
    let width_mbs = enc.cfg.width / 16;
    let height_mbs = enc.cfg.height / 16;
    let chroma_width = (enc.cfg.width / 2) as usize;
    let chroma_height = (enc.cfg.height / 2) as usize;
    let mut recon_y = vec![0u8; (enc.cfg.width * enc.cfg.height) as usize];
    let mut recon_u = vec![0u8; chroma_width * chroma_height];
    let mut recon_v = vec![0u8; chroma_width * chroma_height];
    let qp_y = enc.cfg.qp;
    let qp_c = qp_y_to_qp_c_with_bd_offset(qp_y, 0, qp_bd_offset(enc.cfg.bit_depth_chroma_minus8));
    let mut nc_grid = CavlcNcGrid::new(width_mbs, height_mbs);
    let mut intra_grid = IntraGrid::new(width_mbs as usize, height_mbs as usize);
    let mut infos = vec![MbDeblockInfo::default(); (width_mbs * height_mbs) as usize];
    // Constant-QP field picture: the §7.4.5 chain never moves, but the
    // per-MB writer threads it (every delta stays 0).
    let mut qp_tracker = super::MbQpTracker { cur: qp_y };
    for mb_y in 0..height_mbs as usize {
        for mb_x in 0..width_mbs as usize {
            let dbl = enc.encode_mb(
                src,
                mb_x,
                mb_y,
                qp_y,
                qp_c,
                chroma_width,
                chroma_height,
                &mut recon_y,
                &mut recon_u,
                &mut recon_v,
                sw,
                &mut nc_grid,
                &mut intra_grid,
                &mut qp_tracker,
            );
            infos[mb_y * width_mbs as usize + mb_x] = dbl;
        }
    }
    (recon_y, recon_u, recon_v, infos)
}

/// Round-436 — return shape of [`encode_p_slice_data`]: pre-deblock
/// recon planes, per-MB deblock facts, and the per-8x8 L0 motion
/// snapshot (the §8.4.1.2.1 colocated data a later B field reads when
/// this field sits at `RefPicList1[0]`).
type PSliceDataOut = (
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<MbDeblockInfo>,
    Vec<FrameRefPartitionMv>,
);

/// Encode one P picture's `slice_data()` bits into `sw` against a
/// single same-sized reference, returning the pre-deblock recon planes
/// and per-MB deblock facts. Mirrors the `encode_p` MB loop, with the
/// §7.3.4 CAVLC `mb_skip_run` accounting included.
fn encode_p_slice_data(
    enc: &Encoder,
    src: &YuvFrame<'_>,
    prev: &EncodedFrameRef<'_>,
    sw: &mut BitWriter,
) -> PSliceDataOut {
    let width_mbs = enc.cfg.width / 16;
    let height_mbs = enc.cfg.height / 16;
    let chroma_width = (enc.cfg.width / 2) as usize;
    let chroma_height = (enc.cfg.height / 2) as usize;
    let mut recon_y = vec![0u8; (enc.cfg.width * enc.cfg.height) as usize];
    let mut recon_u = vec![0u8; chroma_width * chroma_height];
    let mut recon_v = vec![0u8; chroma_width * chroma_height];
    let qp_y = enc.cfg.qp;
    let qp_c = qp_y_to_qp_c_with_bd_offset(qp_y, 0, qp_bd_offset(enc.cfg.bit_depth_chroma_minus8));
    let mut nc_grid = CavlcNcGrid::new(width_mbs, height_mbs);
    let mut intra_grid = IntraGrid::new(width_mbs as usize, height_mbs as usize);
    let mut mv_grid = MvGrid::new(width_mbs as usize, height_mbs as usize);
    let mut infos = vec![MbDeblockInfo::default(); (width_mbs * height_mbs) as usize];
    let mut pending_skip: u32 = 0;
    // Constant-QP field picture: the §7.4.5 chain never moves, but the
    // per-MB writers still thread it (delta stays 0).
    let mut qp_tracker = super::MbQpTracker { cur: qp_y };
    for mb_y in 0..height_mbs as usize {
        for mb_x in 0..width_mbs as usize {
            let dbl = enc.encode_p_mb_with_intra_fallback(
                src,
                prev,
                mb_x,
                mb_y,
                qp_y,
                qp_c,
                chroma_width,
                chroma_height,
                &mut recon_y,
                &mut recon_u,
                &mut recon_v,
                sw,
                &mut nc_grid,
                &mut intra_grid,
                &mut mv_grid,
                &mut pending_skip,
                &mut qp_tracker,
            );
            infos[mb_y * width_mbs as usize + mb_x] = dbl;
        }
    }
    // §7.3.4 — flush a trailing skip run.
    if pending_skip > 0 {
        sw.ue(pending_skip);
    }
    let partition_mvs = partition_mvs_from_grid(&mv_grid);
    (recon_y, recon_u, recon_v, infos, partition_mvs)
}

/// Round-436 — encode one non-reference B field picture's
/// `slice_data()` bits into `sw` against one reference per list,
/// returning the pre-deblock recon planes and per-MB deblock facts.
/// Mirrors the `encode_b` MB loop (B_Skip / B_Direct_16x16 / explicit
/// 16x16 / partitions with the §8.4.1.2.2 spatial or §8.4.1.2.3
/// temporal direct derivation per `enc.cfg.direct_temporal_mv_pred`),
/// with the §7.3.4 CAVLC `mb_skip_run` accounting included. All POC
/// arithmetic (temporal direct eq. 8-201/8-202) runs on the fields'
/// own §8.2.1 order counts carried by the `EncodedFrameRef`s.
fn encode_b_slice_data(
    enc: &Encoder,
    src: &YuvFrame<'_>,
    ref_l0: &EncodedFrameRef<'_>,
    ref_l1: &EncodedFrameRef<'_>,
    curr_poc: i32,
    sw: &mut BitWriter,
    // Round-440 — §8.4.2.3.3 implicit weights for the bipred merges
    // (`None` = §8.4.2.3.1 default average).
    weighted: Option<super::WeightedBipredLuma>,
) -> PSliceDataOut {
    let width = enc.cfg.width as usize;
    let width_mbs = enc.cfg.width / 16;
    let height_mbs = enc.cfg.height / 16;
    let chroma_width = (enc.cfg.width / 2) as usize;
    let chroma_height = (enc.cfg.height / 2) as usize;
    let mut recon_y = vec![0u8; (enc.cfg.width * enc.cfg.height) as usize];
    let mut recon_u = vec![0u8; chroma_width * chroma_height];
    let mut recon_v = vec![0u8; chroma_width * chroma_height];
    let qp_y = enc.cfg.qp;
    let qp_c = qp_y_to_qp_c_with_bd_offset(qp_y, 0, qp_bd_offset(enc.cfg.bit_depth_chroma_minus8));
    let mut nc_grid = CavlcNcGrid::new(width_mbs, height_mbs);
    let mut intra_grid = IntraGrid::new(width_mbs as usize, height_mbs as usize);
    // §8.4.1.3 — one MV-prediction grid per reference list.
    let mut mv_grid_l0 = MvGrid::new(width_mbs as usize, height_mbs as usize);
    let mut mv_grid_l1 = MvGrid::new(width_mbs as usize, height_mbs as usize);
    let mut infos = vec![MbDeblockInfo::default(); (width_mbs * height_mbs) as usize];
    let mut pending_skip: u32 = 0;
    // Field B slices run at frame-constant QP: the §7.4.5 chain stays
    // at qp_y and every emitted mb_qp_delta is 0.
    let mut qp_tracker = crate::encoder::MbQpTracker { cur: qp_y };
    for mb_y in 0..height_mbs as usize {
        for mb_x in 0..width_mbs as usize {
            let dbl = enc.encode_b_mb_with_intra_fallback(
                src,
                ref_l0,
                ref_l1,
                mb_x,
                mb_y,
                qp_y,
                qp_c,
                chroma_width,
                chroma_height,
                width,
                &mut recon_y,
                &mut recon_u,
                &mut recon_v,
                sw,
                &mut nc_grid,
                &mut intra_grid,
                &mut mv_grid_l0,
                &mut mv_grid_l1,
                &mut pending_skip,
                curr_poc,
                weighted,
                &mut qp_tracker,
            );
            infos[mb_y * width_mbs as usize + mb_x] = dbl;
        }
    }
    // §7.3.4 — flush the trailing skip run (mirrors `encode_b`: the
    // final `mb_skip_run` is emitted even when zero).
    sw.ue(pending_skip);
    // Round-440 — B fields can themselves serve as references
    // (`PaffConfig::b_reference_fields`): snapshot the per-8x8 L0
    // motion for later direct derivations reading this picture as
    // colPic.
    let partition_mvs = partition_mvs_from_grid(&mv_grid_l0);
    (recon_y, recon_u, recon_v, infos, partition_mvs)
}

/// Round-440 — §8.4.2.3.3 implicit weighted-prediction (w0, w1) pair
/// for one B FIELD, from the fields' own §8.2.1 order counts
/// (eq. 8-201/8-202 distances, eq. 8-197/8-198 DistScaleFactor,
/// eq. 8-280..8-283 weight selection at logWD = 5, zero offsets).
/// Mirrors the decoder's derivation so the encoder's local recon stays
/// bit-exact.
fn implicit_field_weights(curr_poc: i32, poc0: i32, poc1: i32) -> super::WeightedBipredLuma {
    let td = (poc1 - poc0).clamp(-128, 127);
    let (w0, w1) = if td == 0 {
        (32, 32)
    } else {
        let tb = (curr_poc - poc0).clamp(-128, 127);
        let tx = (16384 + (td / 2).abs()) / td;
        let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
        let w1 = dsf >> 2;
        if !(-64..=128).contains(&w1) {
            (32, 32)
        } else {
            (64 - w1, w1)
        }
    };
    super::WeightedBipredLuma {
        log2_wd: 5,
        weight_l0: w0,
        offset_l0: 0,
        weight_l1: w1,
        offset_l1: 0,
        apply_chroma: true,
    }
}

/// Encode an interlaced 4:2:0 sequence with PAFF field pictures. See
/// the module docs for the stream layout. `frames` are full-height
/// (Y, Cb, Cr) planes, one entry per interlaced frame.
pub fn encode_paff_sequence(cfg: &PaffConfig, frames: &[(&[u8], &[u8], &[u8])]) -> PaffEncoded {
    assert!(cfg.width % 16 == 0, "width must be MB-aligned");
    assert!(
        cfg.frame_height % 32 == 0,
        "frame height must be a multiple of 32 (each field MB-aligned)",
    );
    assert!(!frames.is_empty());
    assert!(
        !cfg.p_fields || cfg.frame_picture_indices.is_empty(),
        "frame pictures inside a P-field sequence are out of scope",
    );
    assert!(
        !cfg.frame_picture_indices.contains(&0),
        "frame 0 is the IDR field pair",
    );
    assert!(
        !cfg.cross_parity_first_bottom || cfg.p_fields,
        "cross_parity_first_bottom is a P-field axis",
    );
    assert!(
        !cfg.idr_frame_first || cfg.p_fields,
        "idr_frame_first is a P-field axis",
    );
    assert!(
        !(cfg.idr_frame_first && cfg.cross_parity_first_bottom),
        "idr_frame_first replaces frame 0's field pair",
    );
    assert!(!cfg.b_fields || cfg.p_fields, "b_fields is a P-field axis");
    assert!(
        !cfg.b_fields || (!cfg.cross_parity_first_bottom && !cfg.idr_frame_first),
        "b_fields is incompatible with the other optional axes",
    );
    assert!(
        !cfg.b_temporal_direct || cfg.b_fields,
        "b_temporal_direct requires b_fields",
    );
    assert!(
        !cfg.b_implicit_weight || cfg.b_fields,
        "b_implicit_weight requires b_fields",
    );
    assert!(
        !cfg.b_reference_fields || cfg.b_fields,
        "b_reference_fields requires b_fields",
    );
    assert!(
        !(cfg.b_implicit_weight && cfg.b_reference_fields),
        "b_implicit_weight and b_reference_fields are separate stream axes",
    );
    assert!(
        !cfg.long_term_anchor
            || (cfg.p_fields
                && !cfg.b_fields
                && !cfg.cross_parity_first_bottom
                && !cfg.idr_frame_first
                && !cfg.mmco_unpair_first_top),
        "long_term_anchor is a standalone P-field axis",
    );
    assert!(
        !cfg.mmco_unpair_first_top
            || (cfg.p_fields
                && !cfg.b_fields
                && !cfg.cross_parity_first_bottom
                && !cfg.idr_frame_first),
        "mmco_unpair_first_top is a standalone P-field axis",
    );

    let width = cfg.width as usize;
    let frame_h = cfg.frame_height as usize;
    let field_h = cfg.frame_height / 2;
    let width_mbs = cfg.width / 16;
    let frame_h_mbs = cfg.frame_height / 16;
    let log2_max_frame_num_minus4: u32 = 4;
    let log2_max_poc_lsb_minus4: u32 = 4;
    let frame_num_bits = log2_max_frame_num_minus4 + 4;
    let poc_lsb_bits = log2_max_poc_lsb_minus4 + 4;

    // Field-sized encoder (all field pictures) + frame-sized encoder
    // (mixed FRAME pictures). §A.2.1 bars interlace from Baseline —
    // Main (77) it is; CAVLC only.
    // §A.2.1 bars interlace from Baseline — Main (77); the 8x8
    // transform is a High-profile tool (§A.2.4) so the 8x8 axis
    // promotes to High (100).
    let profile_idc: u8 = if cfg.transform_8x8 { 100 } else { 77 };
    // Round-440 — the B-reference-fields layout holds up to four
    // frame-level reference units live (two anchors + two reference-B
    // pairs across a group boundary).
    let max_num_ref_frames: u32 = if cfg.b_reference_fields { 4 } else { 2 };
    let mk_cfg = |h: u32| {
        let mut c = EncoderConfig::new(cfg.width, h);
        c.qp = cfg.qp;
        c.profile_idc = profile_idc;
        c.max_num_ref_frames = max_num_ref_frames;
        c.transform_8x8 = cfg.transform_8x8;
        c
    };
    let field_enc = Encoder::new(mk_cfg(field_h));
    let frame_enc = Encoder::new(mk_cfg(cfg.frame_height));
    // §8.4.1.4 Table 8-10 — dedicated field encoder for the
    // cross-parity picture (bottom field referencing the top field):
    // chroma predictors add +2 to the vertical MV.
    let xpar_enc = Encoder::new({
        let mut c = mk_cfg(field_h);
        c.table_8_10_cy_offset = 2;
        c
    });

    // SPS (interlaced, FrameHeightInMbs) + PPS.
    let sps_rbsp = build_baseline_sps_rbsp(&BaselineSpsConfig {
        seq_parameter_set_id: 0,
        level_idc: min_level_idc_for_picture_size(width_mbs, frame_h_mbs),
        width_in_mbs: width_mbs,
        height_in_mbs: frame_h_mbs,
        log2_max_frame_num_minus4,
        log2_max_poc_lsb_minus4,
        max_num_ref_frames,
        profile_idc,
        chroma_format_idc: 1,
        separate_colour_plane: false,
        seq_scaling_lists: None,
        bit_depth_luma_minus8: 0,
        bit_depth_chroma_minus8: 0,
        interlaced_fields: true,
        vui: None,
    });
    let pps_rbsp = build_baseline_pps_rbsp(&BaselinePpsConfig {
        pic_scaling_lists: None,
        chroma_format_idc: 1,
        pic_parameter_set_id: 0,
        seq_parameter_set_id: 0,
        pic_init_qp_minus26: cfg.qp - 26,
        chroma_qp_index_offset: 0,
        weighted_pred_flag: false,
        // Round-440 — §7.4.2.2: 2 = §8.4.2.3.3 implicit weighted
        // bipred (weights derived from POC distances, nothing coded
        // in the slice headers).
        weighted_bipred_idc: if cfg.b_implicit_weight { 2 } else { 0 },
        entropy_coding_mode_flag: false,
        transform_8x8_mode_flag: cfg.transform_8x8,
    });
    let mut stream: Vec<u8> = Vec::new();
    stream.extend_from_slice(&build_nal_unit(3, NalUnitType::Sps, &sps_rbsp));
    stream.extend_from_slice(&build_nal_unit(3, NalUnitType::Pps, &pps_rbsp));

    // Round-436 — the B-field coding order (0, 2, 1, 4, 3, …) needs
    // its own driver loop.
    if cfg.b_fields {
        return encode_paff_b_sequence(cfg, frames, stream, frame_num_bits, poc_lsb_bits);
    }

    // Last reconstructed field of each parity — the P references.
    let mut last_top: Option<ReconField> = None;
    let mut last_bottom: Option<ReconField> = None;
    // Round-436 — the frame-0 fields, kept for the long_term_anchor
    // axis (every P field references them) and for the
    // mmco_unpair_first_top axis (frame 2's top field falls back to
    // frame 0's top after frame 1's top is unmarked).
    let mut anchor_top: Option<ReconField> = None;
    let mut anchor_bottom: Option<ReconField> = None;
    let mut recon_frames = Vec::with_capacity(frames.len());

    for (k, &(fy, fu, fv)) in frames.iter().enumerate() {
        assert_eq!(fy.len(), width * frame_h);
        let frame_num = (k as u32) % (1 << frame_num_bits);

        if k == 0 && cfg.idr_frame_first {
            // ---- IDR full-height FRAME picture (field_pic_flag = 0). ----
            let src = YuvFrame {
                width: cfg.width,
                height: cfg.frame_height,
                y: fy,
                u: fu,
                v: fv,
            };
            let mut sw = BitWriter::new();
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
                },
            );
            let (mut ry, mut ru, mut rv, infos) = encode_i_slice_data(&frame_enc, &src, &mut sw);
            sw.rbsp_trailing_bits();
            stream.extend_from_slice(&build_nal_unit(3, NalUnitType::SliceIdr, &sw.into_bytes()));
            deblock_recon_with_chroma_array_type(
                cfg.width,
                cfg.frame_height,
                cfg.width / 2,
                cfg.frame_height / 2,
                &mut ry,
                &mut ru,
                &mut rv,
                &infos,
                0,
                width_mbs,
                frame_h_mbs,
                1,
            );
            // §8.2.4.2.5 — either parity field of this stored frame is
            // a distinct reference picture: the next frame's P fields
            // reference its parity rows (what a decoder materialises
            // as a field view of the stored frame). Both field POCs
            // equal the frame's (TopFOC == BotFOC == 0 for the IDR).
            let (ty, tu, tv) = extract_field(&ry, &ru, &rv, width, frame_h, false);
            let (by, bu, bv) = extract_field(&ry, &ru, &rv, width, frame_h, true);
            last_top = Some((ty, tu, tv, 0));
            last_bottom = Some((by, bu, bv, 0));
            recon_frames.push((ry, ru, rv));
            continue;
        }

        if cfg.frame_picture_indices.contains(&k) {
            // ---- Full-height I FRAME picture (field_pic_flag = 0). ----
            let src = YuvFrame {
                width: cfg.width,
                height: cfg.frame_height,
                y: fy,
                u: fu,
                v: fv,
            };
            let mut sw = BitWriter::new();
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
                    pic_order_cnt_lsb: (2 * k as u32) % (1 << poc_lsb_bits),
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
                },
            );
            let (mut ry, mut ru, mut rv, infos) = encode_i_slice_data(&frame_enc, &src, &mut sw);
            sw.rbsp_trailing_bits();
            stream.extend_from_slice(&build_nal_unit(
                2,
                NalUnitType::SliceNonIdr,
                &sw.into_bytes(),
            ));
            deblock_recon_with_chroma_array_type(
                cfg.width,
                cfg.frame_height,
                cfg.width / 2,
                cfg.frame_height / 2,
                &mut ry,
                &mut ru,
                &mut rv,
                &infos,
                0,
                width_mbs,
                frame_h_mbs,
                1,
            );
            recon_frames.push((ry, ru, rv));
            // Fields of a stored FRAME as P references are out of
            // scope (asserted above), so last_top/last_bottom stay.
            continue;
        }

        // ---- Field pair: top (even rows) first, then bottom. ----
        let mut pair_recon: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::with_capacity(2);
        for bottom in [false, true] {
            let (sy, su, sv) = extract_field(fy, fu, fv, width, frame_h, bottom);
            let src = YuvFrame {
                width: cfg.width,
                height: field_h,
                y: &sy,
                u: &su,
                v: &sv,
            };
            let field_signal = if bottom {
                FieldPicSignal::BottomField
            } else {
                FieldPicSignal::TopField
            };
            let poc_lsb = (2 * k as u32 + u32::from(bottom)) % (1 << poc_lsb_bits);
            let is_idr = k == 0 && !bottom;
            // Same-parity reference field of the previous frame — what
            // §8.2.4.2.5 puts at RefPicList0[0] for this field. The
            // long-term axis instead references the frame-0 anchor
            // field (spliced to ref_idx 0 by the RPLM below); the
            // MMCO-1 axis sends frame 2's top field back to frame 0's
            // top (the §8.2.4.2.5 "missing field is ignored" fallback
            // after frame 1's top was unmarked).
            let ref_field = if cfg.long_term_anchor && k > 0 {
                if bottom {
                    &anchor_bottom
                } else {
                    &anchor_top
                }
            } else if cfg.mmco_unpair_first_top && k == 2 && !bottom {
                &anchor_top
            } else if bottom {
                &last_bottom
            } else {
                &last_top
            };
            // Cross-parity axis: frame 0's bottom field P-references
            // the IDR top field (opposite parity) instead of being an
            // I field. Every other P field references the same-parity
            // field of the previous frame.
            let cross = cfg.cross_parity_first_bottom && k == 0 && bottom;
            let (as_p, p_enc, p_ref) = if cross {
                (last_top.is_some(), &xpar_enc, &last_top)
            } else {
                (cfg.p_fields && ref_field.is_some(), &field_enc, ref_field)
            };

            let mut sw = BitWriter::new();
            // §8.5.6/§8.5.7 — every MB of a field picture is a field
            // MB: the CAVLC residual writer emits the Table 8-13 FIELD
            // scan (4x4) and, under `transform_8x8`, the encoder scans
            // 8x8 coefficients with the Table 8-14 field scan.
            sw.set_field_scan(true);
            if as_p {
                write_p_slice_header(
                    &mut sw,
                    &PSliceHeaderConfig {
                        first_mb_in_slice: 0,
                        slice_type_raw: 5,
                        pic_parameter_set_id: 0,
                        colour_plane_id: None,
                        frame_num,
                        frame_num_bits,
                        pic_order_cnt_lsb: poc_lsb,
                        poc_lsb_bits,
                        slice_qp_delta: 0,
                        disable_deblocking_filter_idc: 0,
                        slice_alpha_c0_offset_div2: 0,
                        slice_beta_offset_div2: 0,
                        nal_ref_idc: 2,
                        cabac: None,
                        field: field_signal,
                        // Long-term axis — §8.2.4.3.2 idc-2: splice
                        // the same-parity long-term anchor field
                        // (eq. 8-32 LongTermPicNum = 2*0 + 1 = 1) to
                        // ref_idx 0. Frame 1's TOP field still sees
                        // the frame-0 fields as ordinary short-term
                        // pictures (the promotion runs AFTER it
                        // decodes) so it needs no RPLM — but every
                        // field from frame 1's BOTTOM on does (the
                        // §8.2.4.2.2 initial list now leads with
                        // short-term fields of the newer frames).
                        rplm_l0: if cfg.long_term_anchor && (k > 1 || (k == 1 && bottom)) {
                            &[EncRplmOp::LongTerm(1)]
                        } else {
                            &[]
                        },
                        mmco: if cfg.long_term_anchor && k == 1 && !bottom {
                            // Long-term axis — promote the frame-0
                            // pair: MMCO 4 unlocks LongTermFrameIdx 0
                            // (§7.4.3.3 requires it before any MMCO
                            // 3/6), then two field MMCO 3 ops
                            // (CurrPicNum = 3: same-parity top PicNum
                            // 1 ⇒ diff 1, opposite-parity bottom
                            // PicNum 0 ⇒ diff 2).
                            &[
                                EncMmcoOp::SetMaxLongTermIdx(1),
                                EncMmcoOp::AssignLongTerm(1, 0),
                                EncMmcoOp::AssignLongTerm(2, 0),
                            ]
                        } else if cfg.mmco_unpair_first_top && k == 1 && bottom {
                            // MMCO-1 axis — frame 1's bottom field
                            // unmarks its own pair's first (top)
                            // field: eq. 8-39 picNumX = CurrPicNum(3)
                            // − 1 = 2 = the opposite-parity eq. 8-31
                            // PicNum of frame 1's top field.
                            &[EncMmcoOp::MarkShortTermUnused(0)]
                        } else {
                            &[]
                        },
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
                        pic_order_cnt_lsb: poc_lsb,
                        poc_lsb_bits,
                        slice_qp_delta: 0,
                        disable_deblocking_filter_idc: 0,
                        slice_alpha_c0_offset_div2: 0,
                        slice_beta_offset_div2: 0,
                        field: field_signal,
                        idr: is_idr,
                        nal_ref_idc: if is_idr { 3 } else { 2 },
                        long_term_reference_flag: false,
                        mmco: &[],
                    },
                );
            }

            let (mut ry, mut ru, mut rv, infos) = if as_p {
                let (py, pu, pv, ppoc) = p_ref.as_ref().unwrap();
                let prev = EncodedFrameRef {
                    width: cfg.width,
                    height: field_h,
                    recon_y: py,
                    recon_u: pu,
                    recon_v: pv,
                    partition_mvs: &[],
                    pic_order_cnt: *ppoc,
                };
                let (ry, ru, rv, infos, _mvs) = encode_p_slice_data(p_enc, &src, &prev, &mut sw);
                (ry, ru, rv, infos)
            } else {
                encode_i_slice_data(&field_enc, &src, &mut sw)
            };
            sw.rbsp_trailing_bits();
            let (ref_idc, nal_type) = if is_idr {
                (3, NalUnitType::SliceIdr)
            } else {
                (2, NalUnitType::SliceNonIdr)
            };
            stream.extend_from_slice(&build_nal_unit(ref_idc, nal_type, &sw.into_bytes()));

            // §8.7 with field_pic = 1 — the reference fields other
            // pictures use MUST be the post-filter samples.
            deblock_recon_field(
                cfg.width,
                field_h,
                cfg.width / 2,
                field_h / 2,
                &mut ry,
                &mut ru,
                &mut rv,
                &infos,
                0,
                width_mbs,
                field_h / 16,
                1,
            );
            let slot = (ry.clone(), ru.clone(), rv.clone(), poc_lsb as i32);
            if k == 0 {
                if bottom {
                    anchor_bottom = Some(slot.clone());
                } else {
                    anchor_top = Some(slot.clone());
                }
            }
            if bottom {
                last_bottom = Some(slot);
            } else {
                last_top = Some(slot);
            }
            pair_recon.push((ry, ru, rv));
        }
        let (ty, tu, tv) = &pair_recon[0];
        let (by, bu, bv) = &pair_recon[1];
        recon_frames.push((
            interleave_planes(ty, by, width),
            interleave_planes(tu, bu, width / 2),
            interleave_planes(tv, bv, width / 2),
        ));
    }

    PaffEncoded {
        annex_b: stream,
        recon_frames,
    }
}

/// Round-436 — PAFF **B-field** sequence driver (see
/// [`PaffConfig::b_fields`]). Coding order interleaves anchors and B
/// pairs: display 0 (IDR top + I bottom), display 2 (P/P), display 1
/// (B/B), display 4 (P/P), display 3 (B/B), … — every B field pair is
/// non-reference (`nal_ref_idc = 0`) and predicts L0 from the
/// same-parity field of the previous anchor and L1 from the
/// same-parity field of the following anchor, which is where the
/// §8.2.4.2.4 + §8.2.4.2.5 field initialisation puts them with the
/// single-entry active lists. `stream` arrives holding the SPS + PPS.
fn encode_paff_b_sequence(
    cfg: &PaffConfig,
    frames: &[(&[u8], &[u8], &[u8])],
    mut stream: Vec<u8>,
    frame_num_bits: u32,
    poc_lsb_bits: u32,
) -> PaffEncoded {
    let width = cfg.width as usize;
    let frame_h = cfg.frame_height as usize;
    let field_h = cfg.frame_height / 2;
    let width_mbs = cfg.width / 16;

    let mk_cfg = || {
        let mut c = EncoderConfig::new(cfg.width, field_h);
        c.qp = cfg.qp;
        c.profile_idc = if cfg.transform_8x8 { 100u8 } else { 77 };
        c.max_num_ref_frames = 2;
        c.transform_8x8 = cfg.transform_8x8;
        c
    };
    let field_enc = Encoder::new(mk_cfg());
    let b_enc = Encoder::new({
        let mut c = mk_cfg();
        c.direct_temporal_mv_pred = cfg.b_temporal_direct;
        c
    });

    let mut recon_frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> =
        vec![(Vec::new(), Vec::new(), Vec::new()); frames.len()];

    // Encode one anchor field pair (display index `d`): IDR top +
    // non-IDR I bottom when `d == 0`, else P/P referencing the
    // same-parity fields of `prev` (§8.2.4.2.5 `RefPicList0[0]`).
    let encode_anchor_pair = |d: usize,
                              frame_num: u32,
                              prev: Option<&[AnchorField; 2]>,
                              // §8.2.4.3 — L0 RPLM ops for the P fields
                              // (round-440: the stride-4 layout's later
                              // anchors must splice the PREVIOUS ANCHOR
                              // pair to index 0 — the §8.2.4.2.2 default
                              // list orders by FrameNumWrap, which puts
                              // the more recently coded REFERENCE B pair
                              // first).
                              rplm_l0: &[EncRplmOp],
                              stream: &mut Vec<u8>,
                              recon_frames: &mut Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>|
     -> [AnchorField; 2] {
        let (fy, fu, fv) = frames[d];
        assert_eq!(fy.len(), width * frame_h);
        let mut out: Vec<AnchorField> = Vec::with_capacity(2);
        for bottom in [false, true] {
            let (sy, su, sv) = extract_field(fy, fu, fv, width, frame_h, bottom);
            let src = YuvFrame {
                width: cfg.width,
                height: field_h,
                y: &sy,
                u: &su,
                v: &sv,
            };
            let field_signal = if bottom {
                FieldPicSignal::BottomField
            } else {
                FieldPicSignal::TopField
            };
            let poc = 2 * d as i32 + i32::from(bottom);
            let poc_lsb = (poc as u32) % (1 << poc_lsb_bits);
            let is_idr = d == 0 && !bottom;
            let mut sw = BitWriter::new();
            // §8.5.6 — every MB of a field picture is a field MB.
            sw.set_field_scan(true);
            let (mut ry, mut ru, mut rv, infos, mvs) = if let Some(prev) = prev {
                let prev_field = &prev[usize::from(bottom)];
                let prev_ref = prev_field.as_frame_ref(cfg.width, field_h);
                write_p_slice_header(
                    &mut sw,
                    &PSliceHeaderConfig {
                        first_mb_in_slice: 0,
                        slice_type_raw: 5,
                        pic_parameter_set_id: 0,
                        colour_plane_id: None,
                        frame_num,
                        frame_num_bits,
                        pic_order_cnt_lsb: poc_lsb,
                        poc_lsb_bits,
                        slice_qp_delta: 0,
                        disable_deblocking_filter_idc: 0,
                        slice_alpha_c0_offset_div2: 0,
                        slice_beta_offset_div2: 0,
                        nal_ref_idc: 2,
                        cabac: None,
                        field: field_signal,
                        rplm_l0,
                        mmco: &[],
                    },
                );
                encode_p_slice_data(&field_enc, &src, &prev_ref, &mut sw)
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
                        pic_order_cnt_lsb: poc_lsb,
                        poc_lsb_bits,
                        slice_qp_delta: 0,
                        disable_deblocking_filter_idc: 0,
                        slice_alpha_c0_offset_div2: 0,
                        slice_beta_offset_div2: 0,
                        field: field_signal,
                        idr: is_idr,
                        nal_ref_idc: if is_idr { 3 } else { 2 },
                        long_term_reference_flag: false,
                        mmco: &[],
                    },
                );
                let (ry, ru, rv, infos) = encode_i_slice_data(&field_enc, &src, &mut sw);
                // I fields carry no inter motion — the colocated probe
                // treats every partition as intra (refIdxCol = -1).
                let n_parts = (width_mbs as usize) * ((field_h / 16) as usize) * 4;
                let mvs = vec![
                    FrameRefPartitionMv {
                        mv_l0: (0, 0),
                        ref_idx_l0: -1,
                        is_intra: true,
                    };
                    n_parts
                ];
                (ry, ru, rv, infos, mvs)
            };
            sw.rbsp_trailing_bits();
            let (ref_idc, nal_type) = if is_idr {
                (3, NalUnitType::SliceIdr)
            } else {
                (2, NalUnitType::SliceNonIdr)
            };
            stream.extend_from_slice(&build_nal_unit(ref_idc, nal_type, &sw.into_bytes()));
            // §8.7 with field_pic = 1 — reference fields must carry the
            // post-filter samples.
            deblock_recon_field(
                cfg.width,
                field_h,
                cfg.width / 2,
                field_h / 2,
                &mut ry,
                &mut ru,
                &mut rv,
                &infos,
                0,
                width_mbs,
                field_h / 16,
                1,
            );
            out.push(AnchorField {
                y: ry,
                u: ru,
                v: rv,
                poc,
                partition_mvs: mvs,
            });
        }
        recon_frames[d] = (
            interleave_planes(&out[0].y, &out[1].y, width),
            interleave_planes(&out[0].u, &out[1].u, width / 2),
            interleave_planes(&out[0].v, &out[1].v, width / 2),
        );
        let mut it = out.into_iter();
        [it.next().unwrap(), it.next().unwrap()]
    };

    // Encode one B field pair for display index `d` between anchors
    // `l0` (earlier) and `l1` (later). Per the round-20 convention
    // (validated against the black-box reference decoder), a
    // non-reference B carries the `frame_num` of the reference picture
    // that precedes it in decoding order; a REFERENCE B pair
    // (round-440 `b_reference_fields`) carries its own incremented
    // frame_num and `nal_ref_idc = 2`, and returns its reconstructed
    // fields + L0 motion snapshots so later B pairs can reference it.
    let encode_b_pair = |d: usize,
                         frame_num: u32,
                         l0: &[AnchorField; 2],
                         l1: &[AnchorField; 2],
                         is_ref: bool,
                         enc: &Encoder,
                         stream: &mut Vec<u8>,
                         recon_frames: &mut Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>|
     -> Option<[AnchorField; 2]> {
        let (fy, fu, fv) = frames[d];
        assert_eq!(fy.len(), width * frame_h);
        let mut pair: Vec<AnchorField> = Vec::with_capacity(2);
        for bottom in [false, true] {
            let (sy, su, sv) = extract_field(fy, fu, fv, width, frame_h, bottom);
            let src = YuvFrame {
                width: cfg.width,
                height: field_h,
                y: &sy,
                u: &su,
                v: &sv,
            };
            let field_signal = if bottom {
                FieldPicSignal::BottomField
            } else {
                FieldPicSignal::TopField
            };
            let poc = 2 * d as i32 + i32::from(bottom);
            let poc_lsb = (poc as u32) % (1 << poc_lsb_bits);
            let ref_l0 = l0[usize::from(bottom)].as_frame_ref(cfg.width, field_h);
            let ref_l1 = l1[usize::from(bottom)].as_frame_ref(cfg.width, field_h);
            // Round-440 — §8.4.2.3.3 implicit weights on the FIELDS'
            // own order counts (`None` keeps the default average).
            let weighted = if cfg.b_implicit_weight {
                Some(implicit_field_weights(
                    poc,
                    l0[usize::from(bottom)].poc,
                    l1[usize::from(bottom)].poc,
                ))
            } else {
                None
            };
            let ref_idc: u32 = if is_ref { 2 } else { 0 };
            let mut sw = BitWriter::new();
            sw.set_field_scan(true);
            write_b_slice_header(
                &mut sw,
                &BSliceHeaderConfig {
                    first_mb_in_slice: 0,
                    slice_type_raw: 6,
                    pic_parameter_set_id: 0,
                    colour_plane_id: None,
                    frame_num,
                    frame_num_bits,
                    pic_order_cnt_lsb: poc_lsb,
                    poc_lsb_bits,
                    direct_spatial_mv_pred_flag: !cfg.b_temporal_direct,
                    slice_qp_delta: 0,
                    disable_deblocking_filter_idc: 0,
                    slice_alpha_c0_offset_div2: 0,
                    slice_beta_offset_div2: 0,
                    nal_ref_idc: ref_idc,
                    pred_weight_table: None,
                    cabac: None,
                    field: field_signal,
                },
            );
            let (mut ry, mut ru, mut rv, infos, mvs) =
                encode_b_slice_data(enc, &src, &ref_l0, &ref_l1, poc, &mut sw, weighted);
            sw.rbsp_trailing_bits();
            stream.extend_from_slice(&build_nal_unit(
                ref_idc as u8,
                NalUnitType::SliceNonIdr,
                &sw.into_bytes(),
            ));
            // §8.7 with field_pic = 1 — the decoder outputs post-filter
            // samples for the B fields too (and a REFERENCE B pair
            // must store the post-filter samples).
            deblock_recon_field(
                cfg.width,
                field_h,
                cfg.width / 2,
                field_h / 2,
                &mut ry,
                &mut ru,
                &mut rv,
                &infos,
                0,
                width_mbs,
                field_h / 16,
                1,
            );
            pair.push(AnchorField {
                y: ry,
                u: ru,
                v: rv,
                poc,
                partition_mvs: mvs,
            });
        }
        recon_frames[d] = (
            interleave_planes(&pair[0].y, &pair[1].y, width),
            interleave_planes(&pair[0].u, &pair[1].u, width / 2),
            interleave_planes(&pair[0].v, &pair[1].v, width / 2),
        );
        if is_ref {
            let mut it = pair.into_iter();
            Some([it.next().unwrap(), it.next().unwrap()])
        } else {
            None
        }
    };

    // Round-440 — reference-B pairs restrict their mode decision so
    // the stored L0 motion snapshot matches a decoder's co-located
    // read (`EncoderConfig::b_l0_bi_only`).
    let ref_b_enc = Encoder::new({
        let mut c = mk_cfg();
        c.direct_temporal_mv_pred = cfg.b_temporal_direct;
        c.b_l0_bi_only = true;
        c
    });

    if cfg.b_reference_fields {
        // Stride-4 layout: 0, 4, 2ref, 1, 3, 8, 6ref, 5, 7, …
        assert!(
            frames.len() >= 5 && frames.len() % 4 == 1,
            "b_reference_fields needs 4k+1 display frames (anchors at multiples of 4)",
        );
        let mut prev_anchor = encode_anchor_pair(0, 0, None, &[], &mut stream, &mut recon_frames);
        let mut ref_fn: u32 = 0;
        let mut d = 4usize;
        while d < frames.len() {
            ref_fn += 1;
            // §8.2.4.3.1 — from the second P/P anchor on, the DPB also
            // holds the previous group's reference-B pair (higher
            // FrameNumWrap than the previous anchor): splice the
            // previous anchor pair back to RefPicList0[0]. Same-parity
            // eq. 8-30 PicNums: CurrPicNum = 2*fn + 1, target =
            // 2*(fn - 2) + 1 ⇒ abs_diff_pic_num_minus1 = 3.
            let anchor_rplm: &[EncRplmOp] = if ref_fn >= 3 {
                &[EncRplmOp::Subtract(3)]
            } else {
                &[]
            };
            let next_anchor = encode_anchor_pair(
                d,
                ref_fn % (1 << frame_num_bits),
                Some(&prev_anchor),
                anchor_rplm,
                &mut stream,
                &mut recon_frames,
            );
            ref_fn += 1;
            let ref_b = encode_b_pair(
                d - 2,
                ref_fn % (1 << frame_num_bits),
                &prev_anchor,
                &next_anchor,
                true,
                &ref_b_enc,
                &mut stream,
                &mut recon_frames,
            )
            .expect("reference B pair returns its fields");
            let _ = encode_b_pair(
                d - 3,
                ref_fn % (1 << frame_num_bits),
                &prev_anchor,
                &ref_b,
                false,
                &b_enc,
                &mut stream,
                &mut recon_frames,
            );
            let _ = encode_b_pair(
                d - 1,
                ref_fn % (1 << frame_num_bits),
                &ref_b,
                &next_anchor,
                false,
                &b_enc,
                &mut stream,
                &mut recon_frames,
            );
            prev_anchor = next_anchor;
            d += 4;
        }
    } else if cfg.b_implicit_weight {
        // Stride-3 layout: 0, 3, 1, 2, 6, 4, 5, … — the two B pairs
        // between anchors sit at unequal POC distances, so the
        // §8.4.2.3.3 weights genuinely leave (32, 32).
        assert!(
            frames.len() >= 4 && frames.len() % 3 == 1,
            "b_implicit_weight needs 3k+1 display frames (anchors at multiples of 3)",
        );
        let mut prev_anchor = encode_anchor_pair(0, 0, None, &[], &mut stream, &mut recon_frames);
        let mut anchor_fn: u32 = 0;
        let mut d = 3usize;
        while d < frames.len() {
            anchor_fn += 1;
            let next_anchor = encode_anchor_pair(
                d,
                anchor_fn % (1 << frame_num_bits),
                Some(&prev_anchor),
                &[],
                &mut stream,
                &mut recon_frames,
            );
            for b_d in [d - 2, d - 1] {
                let _ = encode_b_pair(
                    b_d,
                    anchor_fn % (1 << frame_num_bits),
                    &prev_anchor,
                    &next_anchor,
                    false,
                    &b_enc,
                    &mut stream,
                    &mut recon_frames,
                );
            }
            prev_anchor = next_anchor;
            d += 3;
        }
    } else {
        // Round-436 stride-2 layout: anchor 0, then per following
        // anchor (display 2, 4, …) the anchor pair followed by the B
        // pair it encloses; a trailing odd display frame becomes a
        // P/P anchor pair.
        let mut prev_anchor = encode_anchor_pair(0, 0, None, &[], &mut stream, &mut recon_frames);
        let mut anchor_fn: u32 = 0;
        let mut d = 2usize;
        while d < frames.len() {
            anchor_fn += 1;
            let next_anchor = encode_anchor_pair(
                d,
                anchor_fn % (1 << frame_num_bits),
                Some(&prev_anchor),
                &[],
                &mut stream,
                &mut recon_frames,
            );
            let _ = encode_b_pair(
                d - 1,
                anchor_fn % (1 << frame_num_bits),
                &prev_anchor,
                &next_anchor,
                false,
                &b_enc,
                &mut stream,
                &mut recon_frames,
            );
            prev_anchor = next_anchor;
            d += 2;
        }
        if frames.len() % 2 == 0 {
            // Trailing odd display frame without a following anchor.
            anchor_fn += 1;
            let _ = encode_anchor_pair(
                frames.len() - 1,
                anchor_fn % (1 << frame_num_bits),
                Some(&prev_anchor),
                &[],
                &mut stream,
                &mut recon_frames,
            );
        }
    }

    PaffEncoded {
        annex_b: stream,
        recon_frames,
    }
}
