//! Round-453 — multi-reference P coding with long-term references,
//! adaptive marking (MMCO) and reference list modification (RPLM).
//!
//! Drives the frame-based P macroblock encoder over a decoded-picture
//! buffer the encoder mirrors exactly the way §8.2.5 makes the decoder
//! keep it:
//!
//! * **RefPicList0** (§8.2.4.2.1): short-term references by descending
//!   `PicNum` (= `frame_num` for frames without wrap), then long-term
//!   references by ascending `LongTermPicNum`; truncated to the
//!   slice's `num_ref_idx_l0_active` (§7.3.3 override coded on every
//!   P slice).
//! * **RPLM** (§7.3.3.1 / §8.2.4.3.2): optionally splice a long-term
//!   picture to index 0 (`modification_of_pic_nums_idc = 2`).
//! * **Marking** (§8.2.5): the IDR may be marked long-term
//!   (§7.4.3.3 `long_term_reference_flag`); a P picture may be marked
//!   long-term through MMCO 4 (`max_long_term_frame_idx_plus1`) +
//!   MMCO 6 (`long_term_frame_idx`), a long-term picture may be
//!   unmarked through MMCO 2 (`long_term_pic_num`), and — since
//!   adaptive marking suspends the §8.2.5.3 sliding window — an MMCO 1
//!   (`difference_of_pic_nums_minus1`) evicts the oldest short-term
//!   picture whenever the buffer would exceed `max_num_ref_frames`.
//!   Pictures without MMCO use the sliding window.
//! * **Per-MB reference election**: the §8.4.2.2 quarter-pel search
//!   runs against every active reference; the lowest SAD (with a
//!   small bias toward lower indices — the te(v) / mvp cost) selects
//!   `ref_idx_l0`, coded through §7.3.5.1 / §7.3.5.2 te(v) on
//!   P_L0_16x16 and P_8x8 (P_Skip stays at index 0 per §8.4.1.1).
//!   Motion vector prediction (§8.4.1.3) sees the per-partition
//!   reference indices, and the §8.7.2.1 NOTE 1 deblock identity keys
//!   distinguish the references.
//!
//! Scope: 4:2:0, CAVLC, frame pictures, single slice, every picture a
//! reference.

use crate::encoder::deblock::{deblock_recon_with_chroma_array_type, MbDeblockInfo};
use crate::encoder::me::search_quarter_pel_16x16;
use crate::encoder::nal::build_nal_unit;
use crate::encoder::pps::{build_baseline_pps_rbsp, BaselinePpsConfig};
use crate::encoder::slice::{
    write_idr_i_slice_header, write_p_slice_header, EncMmcoOp, EncRplmOp, FieldPicSignal,
    IdrSliceHeaderConfig, PSliceHeaderConfig,
};
use crate::encoder::sps::{build_baseline_sps_rbsp, BaselineSpsConfig};
use crate::encoder::{
    min_level_idc_for_picture_size, BitWriter, CavlcNcGrid, EncodedFrameRef, Encoder,
    EncoderConfig, IntraGrid, MbQpTracker, MvGrid, YuvFrame,
};
use crate::nal::NalUnitType;
use crate::transform::{qp_bd_offset, qp_y_to_qp_c_with_bd_offset};

/// One picture's (Y, Cb, Cr) planes.
type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// Configuration for [`encode_multiref_sequence`].
#[derive(Debug, Clone)]
pub struct MultiRefConfig {
    /// Luma width in samples (multiple of 16).
    pub width: u32,
    /// Luma height in samples (multiple of 16).
    pub height: u32,
    /// Slice QP_Y.
    pub qp: i32,
    /// SPS `max_num_ref_frames` and the maximum active list size
    /// (1..=16).
    pub num_ref_frames: u32,
    /// §7.4.3.3 — mark the IDR as a long-term reference
    /// (`LongTermFrameIdx = 0`, `MaxLongTermFrameIdx = 0`).
    pub long_term_idr: bool,
    /// Frame index (≥ 1) whose P slice carries MMCO 4 + MMCO 6 marking
    /// the picture long-term (`LongTermFrameIdx` 1 when the IDR holds
    /// index 0, else 0).
    pub mmco_long_term_at: Option<usize>,
    /// Frame index whose P slice carries MMCO 2 unmarking the
    /// long-term picture created by `mmco_long_term_at`.
    pub mmco_unmark_at: Option<usize>,
    /// Frame index whose P slice moves the (first) long-term picture
    /// to `RefPicList0[0]` through an RPLM `long_term_pic_num`
    /// command.
    pub rplm_long_term_first_at: Option<usize>,
}

/// Output of [`encode_multiref_sequence`].
#[derive(Debug, Clone)]
pub struct MultiRefEncoded {
    /// Annex B stream.
    pub annex_b: Vec<u8>,
    /// Deblocked reconstruction of every frame.
    pub recon_frames: Vec<Planes>,
    /// Per `ref_idx_l0` value: number of coded inter macroblocks that
    /// elected it (index = ref_idx).
    pub ref_idx_hist: Vec<usize>,
    /// Coded inter macroblocks whose elected reference is a long-term
    /// picture.
    pub long_term_hits: usize,
    /// Per P picture: the RefPicList0 the slice used, as
    /// `(frame index, is_long_term)` per entry.
    pub lists: Vec<Vec<(usize, bool)>>,
}

/// Encoder-side mirror of one §8.2.5 DPB reference entry.
struct DpbEntry {
    planes: Planes,
    /// Source frame index (diagnostics / `lists`).
    frame_index: usize,
    frame_num: u32,
    poc: i32,
    long_term_idx: Option<u32>,
}

impl DpbEntry {
    fn is_short(&self) -> bool {
        self.long_term_idx.is_none()
    }
}

/// §8.2.4.2.1 — initial RefPicList0 (indices into `dpb`).
fn init_list0(dpb: &[DpbEntry]) -> Vec<usize> {
    let mut short: Vec<usize> = (0..dpb.len()).filter(|&i| dpb[i].is_short()).collect();
    short.sort_by_key(|&i| std::cmp::Reverse(dpb[i].frame_num));
    let mut long: Vec<usize> = (0..dpb.len()).filter(|&i| !dpb[i].is_short()).collect();
    long.sort_by_key(|&i| dpb[i].long_term_idx.unwrap());
    short.into_iter().chain(long).collect()
}

/// Encode an IDR + P chain with the multi-reference / long-term DPB
/// model described in the module docs.
pub fn encode_multiref_sequence(
    cfg: &MultiRefConfig,
    frames: &[(&[u8], &[u8], &[u8])],
) -> MultiRefEncoded {
    assert!(cfg.width % 16 == 0 && cfg.height % 16 == 0);
    assert!((1..=16).contains(&cfg.num_ref_frames));
    assert!(!frames.is_empty());
    assert!(
        cfg.mmco_unmark_at.is_none() || cfg.mmco_long_term_at.is_some(),
        "mmco_unmark_at needs the MMCO long-term picture",
    );
    assert!(
        !matches!(cfg.mmco_long_term_at, Some(0)),
        "MMCO long-term marking applies to P pictures",
    );

    let w = cfg.width as usize;
    let h = cfg.height as usize;
    let w_mbs = cfg.width / 16;
    let h_mbs = cfg.height / 16;
    let cw = w / 2;
    let ch = h / 2;
    let log2_max_frame_num_minus4: u32 = 4;
    let log2_max_poc_lsb_minus4: u32 = 4;
    let frame_num_bits = log2_max_frame_num_minus4 + 4;
    let poc_lsb_bits = log2_max_poc_lsb_minus4 + 4;
    assert!(
        frames.len() < (1 << frame_num_bits),
        "sequence longer than MaxFrameNum — PicNum wrap is out of scope",
    );
    let qp_y = cfg.qp;
    let qp_c = qp_y_to_qp_c_with_bd_offset(qp_y, 0, qp_bd_offset(0));

    let mut ecfg = EncoderConfig::new(cfg.width, cfg.height);
    ecfg.qp = cfg.qp;
    ecfg.max_num_ref_frames = cfg.num_ref_frames;
    let enc = Encoder::new(ecfg);

    let sps_rbsp = build_baseline_sps_rbsp(&BaselineSpsConfig {
        seq_parameter_set_id: 0,
        level_idc: min_level_idc_for_picture_size(w_mbs, h_mbs),
        width_in_mbs: w_mbs,
        height_in_mbs: h_mbs,
        log2_max_frame_num_minus4,
        log2_max_poc_lsb_minus4,
        max_num_ref_frames: cfg.num_ref_frames,
        profile_idc: 66,
        chroma_format_idc: 1,
        separate_colour_plane: false,
        seq_scaling_lists: None,
        bit_depth_luma_minus8: 0,
        bit_depth_chroma_minus8: 0,
        interlaced_fields: false,
        mbaff: false,
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
        pic_scaling_lists: None,
        chroma_format_idc: 1,
    });
    let mut stream: Vec<u8> = Vec::new();
    stream.extend_from_slice(&build_nal_unit(3, NalUnitType::Sps, &sps_rbsp));
    stream.extend_from_slice(&build_nal_unit(3, NalUnitType::Pps, &pps_rbsp));

    let mut dpb: Vec<DpbEntry> = Vec::new();
    let mut recon_frames: Vec<Planes> = Vec::with_capacity(frames.len());
    let mut ref_idx_hist = vec![0usize; cfg.num_ref_frames as usize];
    let mut long_term_hits = 0usize;
    let mut lists: Vec<Vec<(usize, bool)>> = Vec::new();
    // LongTermFrameIdx the MMCO-marked picture received (for MMCO 2).
    let mut mmco_lt_idx: Option<u32> = None;

    for (k, &(fy, fu, fv)) in frames.iter().enumerate() {
        assert_eq!(fy.len(), w * h);
        let frame_num = k as u32;
        let poc = 2 * k as i32;
        let src = YuvFrame {
            width: cfg.width,
            height: cfg.height,
            y: fy,
            u: fu,
            v: fv,
        };

        if k == 0 {
            let idr = if cfg.long_term_idr {
                // §7.4.3.3 long_term_reference_flag = 1 — assemble the
                // IDR slice by hand so the marking bit is coded.
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
                        field: FieldPicSignal::FrameMbsOnly,
                        idr: true,
                        nal_ref_idc: 3,
                        long_term_reference_flag: true,
                        mmco: &[],
                    },
                );
                let (mut ry, mut ru, mut rv, infos) = encode_i_slice_data(&enc, &src, &mut sw);
                sw.rbsp_trailing_bits();
                stream.extend_from_slice(&build_nal_unit(
                    3,
                    NalUnitType::SliceIdr,
                    &sw.into_bytes(),
                ));
                deblock_recon_with_chroma_array_type(
                    cfg.width,
                    cfg.height,
                    cfg.width / 2,
                    cfg.height / 2,
                    &mut ry,
                    &mut ru,
                    &mut rv,
                    &infos,
                    0,
                    w_mbs,
                    h_mbs,
                    1,
                );
                (ry, ru, rv)
            } else {
                let e = enc.encode_idr(&src);
                // Our own SPS/PPS already lead the stream: keep only
                // the IDR slice NAL (the last NAL of `e.annex_b`).
                let idr_nal_start = find_last_nal_start(&e.annex_b);
                stream.extend_from_slice(&e.annex_b[idr_nal_start..]);
                (e.recon_y, e.recon_u, e.recon_v)
            };
            dpb.push(DpbEntry {
                planes: idr.clone(),
                frame_index: 0,
                frame_num: 0,
                poc: 0,
                long_term_idx: if cfg.long_term_idr { Some(0) } else { None },
            });
            recon_frames.push(idr);
            continue;
        }

        // ---- RefPicList0 for this P picture. ----
        let mut list = init_list0(&dpb);
        let mut rplm: Vec<EncRplmOp> = Vec::new();
        if cfg.rplm_long_term_first_at == Some(k) {
            if let Some(pos) = list.iter().position(|&i| !dpb[i].is_short()) {
                let lt = list.remove(pos);
                list.insert(0, lt);
                // §7.3.3.1 idc 2 — long_term_pic_num = LongTermFrameIdx
                // for frame coding (§8.2.4.1 eq. 8-30).
                rplm.push(EncRplmOp::LongTerm(dpb[lt].long_term_idx.unwrap()));
            }
        }
        let n_active = (cfg.num_ref_frames as usize).min(list.len());
        list.truncate(n_active);
        lists.push(
            list.iter()
                .map(|&i| (dpb[i].frame_index, !dpb[i].is_short()))
                .collect(),
        );

        // ---- §7.3.3.3 marking ops carried by this slice. ----
        // The ops are CODED now but APPLIED to the buffer only after
        // the picture is coded (§8.2.5 runs after decoding the current
        // picture; RefPicList0 above reflects the pre-marking buffer).
        let mut mmco: Vec<EncMmcoOp> = Vec::new();
        let mut mark_current_lt: Option<u32> = None;
        let mut evict_short: Option<u32> = None;
        let mut unmark_lt: Option<u32> = None;
        let oldest_short = |dpb: &[DpbEntry]| -> u32 {
            dpb.iter()
                .filter(|e| e.is_short())
                .min_by_key(|e| e.frame_num)
                .map(|e| e.frame_num)
                .expect("a short-term picture to evict")
        };
        if cfg.mmco_long_term_at == Some(k) {
            let lt_idx = if cfg.long_term_idr { 1 } else { 0 };
            // Adaptive marking suspends the sliding window: evict the
            // oldest short-term picture ourselves when the buffer
            // would overflow after adding the current picture.
            if dpb.len() as u32 >= cfg.num_ref_frames {
                let oldest = oldest_short(&dpb);
                // difference_of_pic_nums_minus1 = CurrPicNum − PicNumX − 1.
                mmco.push(EncMmcoOp::MarkShortTermUnused(frame_num - oldest - 1));
                evict_short = Some(oldest);
            }
            mmco.push(EncMmcoOp::SetMaxLongTermIdx(lt_idx + 1));
            mmco.push(EncMmcoOp::AssignCurrentLongTerm(lt_idx));
            mark_current_lt = Some(lt_idx);
            mmco_lt_idx = Some(lt_idx);
        } else if cfg.mmco_unmark_at == Some(k) {
            let lt_idx = mmco_lt_idx.expect("MMCO long-term picture to unmark");
            // long_term_pic_num = LongTermFrameIdx for frames.
            mmco.push(EncMmcoOp::MarkLongTermUnused(lt_idx));
            unmark_lt = Some(lt_idx);
            // Unmarking frees one slot; evict a short-term picture too
            // if the current picture would still overflow the buffer.
            if dpb.len() as u32 > cfg.num_ref_frames {
                let oldest = oldest_short(&dpb);
                mmco.push(EncMmcoOp::MarkShortTermUnused(frame_num - oldest - 1));
                evict_short = Some(oldest);
            }
        }

        // ---- Slice header. ----
        let mut sw = BitWriter::new();
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
                field: FieldPicSignal::FrameMbsOnly,
                rplm_l0: &rplm,
                mmco: &mmco,
                pred_weight_table: None,
                num_ref_idx_l0_active_minus1: Some(n_active as u32 - 1),
            },
        );

        // ---- Slice data with per-MB reference election. ----
        let refs: Vec<EncodedFrameRef<'_>> = list
            .iter()
            .map(|&i| EncodedFrameRef {
                width: cfg.width,
                height: cfg.height,
                recon_y: &dpb[i].planes.0,
                recon_u: &dpb[i].planes.1,
                recon_v: &dpb[i].planes.2,
                partition_mvs: &[],
                pic_order_cnt: dpb[i].poc,
            })
            .collect();
        let mut recon_y = vec![0u8; w * h];
        let mut recon_u = vec![0u8; cw * ch];
        let mut recon_v = vec![0u8; cw * ch];
        let mut nc_grid = CavlcNcGrid::new(w_mbs, h_mbs);
        let mut intra_grid = IntraGrid::new(w_mbs as usize, h_mbs as usize);
        let mut mv_grid = MvGrid::new(w_mbs as usize, h_mbs as usize);
        let mut infos = vec![MbDeblockInfo::default(); (w_mbs * h_mbs) as usize];
        let mut pending_skip: u32 = 0;
        let mut tracker = MbQpTracker::new(qp_y);
        tracker.num_ref_idx_l0_active_minus1_slice = n_active as u32 - 1;
        for mb_y in 0..h_mbs as usize {
            for mb_x in 0..w_mbs as usize {
                // Reference election: lowest quarter-pel SAD, biased
                // by the index cost (te(v) bits + mvp mismatch risk).
                let mut best = 0usize;
                let mut best_cost = u64::MAX;
                for (ri, r) in refs.iter().enumerate() {
                    let me = search_quarter_pel_16x16(
                        fy, w, cfg.width, cfg.height, r.recon_y, w, cfg.width, cfg.height, mb_x,
                        mb_y, 16, 16,
                    );
                    let cost = me.sad as u64 + 4 * ri as u64;
                    if cost < best_cost {
                        best_cost = cost;
                        best = ri;
                    }
                }
                tracker.ref_idx_l0 = best as i32;
                let before_skip = pending_skip;
                let dbl = enc.encode_p_mb_with_intra_fallback(
                    &src,
                    &refs[best],
                    refs[best].recon_y,
                    mb_x,
                    mb_y,
                    qp_y,
                    qp_c,
                    cw,
                    ch,
                    &mut recon_y,
                    &mut recon_u,
                    &mut recon_v,
                    &mut sw,
                    &mut nc_grid,
                    &mut intra_grid,
                    &mut mv_grid,
                    &mut pending_skip,
                    &mut tracker,
                );
                let skipped = pending_skip > before_skip;
                if !dbl.is_intra && !skipped {
                    ref_idx_hist[best] += 1;
                    if !dpb[list[best]].is_short() {
                        long_term_hits += 1;
                    }
                }
                infos[mb_y * w_mbs as usize + mb_x] = dbl;
            }
        }
        if pending_skip > 0 {
            sw.ue(pending_skip);
        }
        sw.rbsp_trailing_bits();
        stream.extend_from_slice(&build_nal_unit(
            2,
            NalUnitType::SliceNonIdr,
            &sw.into_bytes(),
        ));
        deblock_recon_with_chroma_array_type(
            cfg.width,
            cfg.height,
            cfg.width / 2,
            cfg.height / 2,
            &mut recon_y,
            &mut recon_u,
            &mut recon_v,
            &infos,
            0,
            w_mbs,
            h_mbs,
            1,
        );

        // ---- §8.2.5 marking of the current picture. ----
        if let Some(lt) = unmark_lt {
            dpb.retain(|e| e.long_term_idx != Some(lt));
        }
        if let Some(oldest) = evict_short {
            dpb.retain(|e| !(e.is_short() && e.frame_num == oldest));
        }
        if mmco.is_empty() {
            // §8.2.5.3 sliding window.
            if dpb.len() as u32 >= cfg.num_ref_frames {
                let oldest = dpb
                    .iter()
                    .filter(|e| e.is_short())
                    .min_by_key(|e| e.frame_num)
                    .map(|e| e.frame_num)
                    .expect("sliding window needs a short-term picture");
                dpb.retain(|e| !(e.is_short() && e.frame_num == oldest));
            }
        }
        dpb.push(DpbEntry {
            planes: (recon_y.clone(), recon_u.clone(), recon_v.clone()),
            frame_index: k,
            frame_num,
            poc,
            long_term_idx: mark_current_lt,
        });
        debug_assert!(dpb.len() as u32 <= cfg.num_ref_frames);
        recon_frames.push((recon_y, recon_u, recon_v));
    }

    MultiRefEncoded {
        annex_b: stream,
        recon_frames,
        ref_idx_hist,
        long_term_hits,
        lists,
    }
}

/// Byte offset of the last Annex B start code in `stream`.
fn find_last_nal_start(stream: &[u8]) -> usize {
    let mut last = 0;
    let mut i = 0;
    while i + 3 < stream.len() {
        if stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 0 && stream[i + 3] == 1 {
            last = i;
            i += 4;
        } else {
            i += 1;
        }
    }
    last
}

/// Encode one I picture's `slice_data()` (Intra_16x16 / I_NxN RDO per
/// `Encoder::encode_mb`) into `sw`; returns the pre-deblock recon and
/// per-MB deblock facts.
fn encode_i_slice_data(
    enc: &Encoder,
    src: &YuvFrame<'_>,
    sw: &mut BitWriter,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<MbDeblockInfo>) {
    let w_mbs = enc.config().width / 16;
    let h_mbs = enc.config().height / 16;
    let cw = (enc.config().width / 2) as usize;
    let ch = (enc.config().height / 2) as usize;
    let mut ry = vec![0u8; (enc.config().width * enc.config().height) as usize];
    let mut ru = vec![0u8; cw * ch];
    let mut rv = vec![0u8; cw * ch];
    let qp_y = enc.config().qp;
    let qp_c = qp_y_to_qp_c_with_bd_offset(qp_y, 0, qp_bd_offset(0));
    let mut nc_grid = CavlcNcGrid::new(w_mbs, h_mbs);
    let mut intra_grid = IntraGrid::new(w_mbs as usize, h_mbs as usize);
    let mut infos = vec![MbDeblockInfo::default(); (w_mbs * h_mbs) as usize];
    let mut tracker = MbQpTracker::new(qp_y);
    for mb_y in 0..h_mbs as usize {
        for mb_x in 0..w_mbs as usize {
            let dbl = enc.encode_mb(
                src,
                mb_x,
                mb_y,
                qp_y,
                qp_c,
                cw,
                ch,
                &mut ry,
                &mut ru,
                &mut rv,
                sw,
                &mut nc_grid,
                &mut intra_grid,
                &mut tracker,
            );
            infos[mb_y * w_mbs as usize + mb_x] = dbl;
        }
    }
    (ry, ru, rv, infos)
}
