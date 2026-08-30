//! Round-453 — multi-slice pictures, flexible macroblock ordering
//! (FMO, slice group map types 0..=6), arbitrary slice order (ASO),
//! redundant slices and constrained intra prediction on the Baseline
//! encoder.
//!
//! * **Slices.** A picture's macroblocks are split into slices of at
//!   most [`SlicesConfig::mbs_per_slice`] macroblocks (per slice
//!   group). Each slice restarts the §9.2.1.1 / §8.4.1.3 / §8.3.1.1
//!   neighbour state, so macroblocks of other slices are unavailable
//!   (§6.4.8), and intra prediction receives the same-slice
//!   availability mask explicitly (`predict_16x16_avail`).
//! * **FMO.** With [`SlicesConfig::slice_groups`] the PPS codes
//!   §7.3.2.2 `num_slice_groups_minus1` + the slice group map
//!   parameters, and the encoder derives `MbToSliceGroupMap` with the
//!   decoder's own §8.2.2 implementation. Each slice group's
//!   macroblocks are coded in raster order (§8.2.2 NextMbAddress),
//!   each group forming its own slice(s); map types 3..=5 code
//!   `slice_group_change_cycle` in every slice header (eq. 7-37 width).
//! * **ASO.** [`SlicesConfig::aso`] emits a picture's slice NAL units
//!   in reverse order (§7.4.1.2.3 / §A.2.1: Baseline decoders accept
//!   any slice order).
//! * **Redundant slices.** [`SlicesConfig::redundant`] appends, after
//!   the primary slices, a `redundant_pic_cnt = 1` re-encode of every
//!   slice at a coarser QP (§7.4.3; decoders discard them).
//! * **Constrained intra prediction.**
//!   [`SlicesConfig::constrained_intra`] codes
//!   `constrained_intra_pred_flag = 1`: an Intra_16x16 macroblock in a
//!   P slice treats inter-coded neighbours as unavailable (§8.3.1.2 /
//!   §8.3.3 / §8.3.4).
//!
//! Scope: Baseline (66), CAVLC, 4:2:0, frame pictures; I pictures
//! are Intra_16x16, P pictures use the full P mode set with the
//! Intra_16x16 fallback.

use crate::encoder::deblock::{deblock_recon_with_chroma_array_type, MbDeblockInfo};
use crate::encoder::nal::build_nal_unit;
use crate::encoder::pps::{build_baseline_pps_rbsp, BaselinePpsConfig};
use crate::encoder::slice::{
    write_idr_i_slice_header, write_p_slice_header, FieldPicSignal, IdrSliceHeaderConfig,
    PSliceHeaderConfig,
};
use crate::encoder::sps::{build_baseline_sps_rbsp, BaselineSpsConfig};
use crate::encoder::{
    min_level_idc_for_picture_size, BitWriter, CavlcNcGrid, EncodedFrameRef, Encoder,
    EncoderConfig, IntraGrid, MbQpTracker, MvGrid, YuvFrame,
};
use crate::mb_address::{map_unit_to_slice_group_map, mb_to_slice_group_map};
use crate::nal::NalUnitType;
use crate::pps::{Pps, SliceGroupMap};
use crate::transform::{qp_bd_offset, qp_y_to_qp_c_with_bd_offset};

/// One picture's (Y, Cb, Cr) planes.
type Planes = (Vec<u8>, Vec<u8>, Vec<u8>);

/// Configuration for [`encode_slices_sequence`].
#[derive(Debug, Clone)]
pub struct SlicesConfig {
    /// Luma width in samples (multiple of 16).
    pub width: u32,
    /// Luma height in samples (multiple of 16).
    pub height: u32,
    /// Slice QP_Y of the primary slices.
    pub qp: i32,
    /// §7.3.2.2 FMO: `(num_slice_groups, map)` — the decoder's
    /// [`SliceGroupMap`] shape; `None` = one slice group.
    pub slice_groups: Option<(u32, SliceGroupMap)>,
    /// §7.4.3 `slice_group_change_cycle` for map types 3..=5 (ignored
    /// otherwise).
    pub slice_group_change_cycle: u32,
    /// Maximum macroblocks per slice within a slice group; 0 = one
    /// slice per slice group.
    pub mbs_per_slice: usize,
    /// §7.4.2.2 `constrained_intra_pred_flag`.
    pub constrained_intra: bool,
    /// Emit each picture's slices in reverse order (ASO).
    pub aso: bool,
    /// Append `redundant_pic_cnt = 1` copies of every slice.
    pub redundant: bool,
    /// Frames after the IDR are P pictures (else I pictures).
    pub p_frames: bool,
}

/// Output of [`encode_slices_sequence`].
#[derive(Debug, Clone)]
pub struct SlicesEncoded {
    /// Annex B stream.
    pub annex_b: Vec<u8>,
    /// Deblocked reconstruction of every frame.
    pub recon_frames: Vec<Planes>,
    /// Primary slices per picture.
    pub slices_per_picture: Vec<usize>,
    /// Intra-coded macroblocks in P pictures whose availability mask
    /// dropped at least one neighbour that the raster rule would have
    /// offered (other slice, or inter under constrained intra).
    pub masked_intra_mbs: usize,
    /// Intra macroblocks coded in P pictures.
    pub intra_mbs_in_p: usize,
}

/// eq. 7-37 — bit width of `slice_group_change_cycle`.
fn change_cycle_bits(pic_size_in_map_units: u32, change_rate: u32) -> u32 {
    let v = pic_size_in_map_units as f64 / change_rate as f64 + 1.0;
    v.log2().ceil() as u32
}

/// Encode an IDR + I/P chain as multi-slice / FMO pictures.
pub fn encode_slices_sequence(
    cfg: &SlicesConfig,
    frames: &[(&[u8], &[u8], &[u8])],
) -> SlicesEncoded {
    assert!(cfg.width % 16 == 0 && cfg.height % 16 == 0);
    assert!(!frames.is_empty());
    let w = cfg.width as usize;
    let h = cfg.height as usize;
    let w_mbs = cfg.width / 16;
    let h_mbs = cfg.height / 16;
    let n_mbs = (w_mbs * h_mbs) as usize;
    let cw = w / 2;
    let ch = h / 2;
    let log2_max_frame_num_minus4: u32 = 4;
    let log2_max_poc_lsb_minus4: u32 = 4;
    let frame_num_bits = log2_max_frame_num_minus4 + 4;
    let poc_lsb_bits = log2_max_poc_lsb_minus4 + 4;
    let qp_c_of = |qp: i32| qp_y_to_qp_c_with_bd_offset(qp, 0, qp_bd_offset(0));

    let mut ecfg = EncoderConfig::new(cfg.width, cfg.height);
    ecfg.qp = cfg.qp;
    let enc = Encoder::new(ecfg);
    // Redundant slices are coded at a coarser QP with their own
    // encoder instance (the QP is baked into `EncoderConfig`).
    let redundant_qp = (cfg.qp + 6).min(51);
    let enc_redundant = Encoder::new({
        let mut c = EncoderConfig::new(cfg.width, cfg.height);
        c.qp = redundant_qp;
        c
    });

    // ---- SPS / PPS. ----
    let sps_rbsp = build_baseline_sps_rbsp(&BaselineSpsConfig {
        seq_parameter_set_id: 0,
        level_idc: min_level_idc_for_picture_size(w_mbs, h_mbs),
        width_in_mbs: w_mbs,
        height_in_mbs: h_mbs,
        log2_max_frame_num_minus4,
        log2_max_poc_lsb_minus4,
        max_num_ref_frames: 1,
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
        redundant_pic_cnt_present_flag: cfg.redundant,
        pic_scaling_lists: None,
        chroma_format_idc: 1,
        slice_groups: cfg.slice_groups.clone(),
        constrained_intra_pred_flag: cfg.constrained_intra,
    });
    let mut stream: Vec<u8> = Vec::new();
    stream.extend_from_slice(&build_nal_unit(3, NalUnitType::Sps, &sps_rbsp));
    stream.extend_from_slice(&build_nal_unit(3, NalUnitType::Pps, &pps_rbsp));

    // ---- §8.2.2 MbToSliceGroupMap via the decoder's derivation. ----
    let (num_groups, group_of): (u32, Vec<u32>) = match &cfg.slice_groups {
        None => (1, vec![0; n_mbs]),
        Some((n, map)) => {
            let pps = Pps {
                pic_parameter_set_id: 0,
                seq_parameter_set_id: 0,
                entropy_coding_mode_flag: false,
                bottom_field_pic_order_in_frame_present_flag: false,
                num_slice_groups_minus1: n - 1,
                slice_group_map: Some(map.clone()),
                num_ref_idx_l0_default_active_minus1: 0,
                num_ref_idx_l1_default_active_minus1: 0,
                weighted_pred_flag: false,
                weighted_bipred_idc: 0,
                pic_init_qp_minus26: cfg.qp - 26,
                pic_init_qs_minus26: 0,
                chroma_qp_index_offset: 0,
                deblocking_filter_control_present_flag: true,
                constrained_intra_pred_flag: cfg.constrained_intra,
                redundant_pic_cnt_present_flag: cfg.redundant,
                extension: None,
            };
            let map_units = map_unit_to_slice_group_map(
                &pps,
                n_mbs as u32,
                w_mbs,
                cfg.slice_group_change_cycle,
                false,
            )
            .expect("slice group map");
            (
                *n,
                mb_to_slice_group_map(&map_units, true, false, false, w_mbs),
            )
        }
    };
    let change_cycle: Option<(u32, u32)> = match &cfg.slice_groups {
        Some((
            _,
            SliceGroupMap::Changing {
                change_rate_minus1, ..
            },
        )) => Some((
            cfg.slice_group_change_cycle,
            change_cycle_bits(n_mbs as u32, change_rate_minus1 + 1),
        )),
        _ => None,
    };
    // Slices as ordered macroblock address lists.
    let mut slice_mbs: Vec<Vec<usize>> = Vec::new();
    for g in 0..num_groups {
        let mbs: Vec<usize> = (0..n_mbs).filter(|&a| group_of[a] == g).collect();
        if mbs.is_empty() {
            continue;
        }
        let chunk = if cfg.mbs_per_slice == 0 {
            mbs.len()
        } else {
            cfg.mbs_per_slice
        };
        for c in mbs.chunks(chunk) {
            slice_mbs.push(c.to_vec());
        }
    }
    let mut slice_of = vec![usize::MAX; n_mbs];
    for (sid, mbs) in slice_mbs.iter().enumerate() {
        for &a in mbs {
            slice_of[a] = sid;
        }
    }

    let mut recon_frames: Vec<Planes> = Vec::with_capacity(frames.len());
    let mut prev: Option<Planes> = None;
    let mut slices_per_picture = Vec::new();
    let mut masked_intra_mbs = 0usize;
    let mut intra_mbs_in_p = 0usize;

    for (k, &(fy, fu, fv)) in frames.iter().enumerate() {
        assert_eq!(fy.len(), w * h);
        let is_p = cfg.p_frames && k > 0;
        let frame_num = k as u32;
        let poc = 2 * k as u32;
        let src = YuvFrame {
            width: cfg.width,
            height: cfg.height,
            y: fy,
            u: fu,
            v: fv,
        };
        let prev_ref = prev.as_ref().map(|p| EncodedFrameRef {
            width: cfg.width,
            height: cfg.height,
            recon_y: &p.0,
            recon_u: &p.1,
            recon_v: &p.2,
            partition_mvs: &[],
            pic_order_cnt: 2 * (k as i32 - 1),
        });

        let mut recon_y = vec![0u8; w * h];
        let mut recon_u = vec![0u8; cw * ch];
        let mut recon_v = vec![0u8; cw * ch];
        let mut infos = vec![MbDeblockInfo::default(); n_mbs];
        let mut is_intra = vec![false; n_mbs];
        let mut nals: Vec<Vec<u8>> = Vec::new();
        let mut redundant_nals: Vec<Vec<u8>> = Vec::new();

        for (sid, mbs) in slice_mbs.iter().enumerate() {
            let first_mb = mbs[0] as u32;
            // Primary slice, then (optionally) its redundant copy on a
            // throwaway recon.
            for redundant in [false, true] {
                if redundant && !cfg.redundant {
                    break;
                }
                let e = if redundant { &enc_redundant } else { &enc };
                let qp_y = if redundant { redundant_qp } else { cfg.qp };
                let qp_c = qp_c_of(qp_y);
                let mut sw = BitWriter::new();
                let rpc = if cfg.redundant {
                    Some(u32::from(redundant))
                } else {
                    None
                };
                let (nal_type, ref_idc) = if k == 0 {
                    (NalUnitType::SliceIdr, 3)
                } else {
                    (NalUnitType::SliceNonIdr, 2)
                };
                if is_p {
                    write_p_slice_header(
                        &mut sw,
                        &PSliceHeaderConfig {
                            first_mb_in_slice: first_mb,
                            slice_type_raw: 5,
                            pic_parameter_set_id: 0,
                            colour_plane_id: None,
                            frame_num,
                            frame_num_bits,
                            pic_order_cnt_lsb: poc % (1 << poc_lsb_bits),
                            poc_lsb_bits,
                            slice_qp_delta: qp_y - cfg.qp,
                            disable_deblocking_filter_idc: 0,
                            slice_alpha_c0_offset_div2: 0,
                            slice_beta_offset_div2: 0,
                            nal_ref_idc: 2,
                            cabac: None,
                            field: FieldPicSignal::FrameMbsOnly,
                            rplm_l0: &[],
                            mmco: &[],
                            pred_weight_table: None,
                            num_ref_idx_l0_active_minus1: None,
                            redundant_pic_cnt: rpc,
                            slice_group_change_cycle: change_cycle,
                        },
                    );
                } else {
                    write_idr_i_slice_header(
                        &mut sw,
                        &IdrSliceHeaderConfig {
                            first_mb_in_slice: first_mb,
                            slice_type_raw: 7,
                            pic_parameter_set_id: 0,
                            colour_plane_id: None,
                            frame_num,
                            frame_num_bits,
                            idr_pic_id: 0,
                            pic_order_cnt_lsb: poc % (1 << poc_lsb_bits),
                            poc_lsb_bits,
                            slice_qp_delta: qp_y - cfg.qp,
                            disable_deblocking_filter_idc: 0,
                            slice_alpha_c0_offset_div2: 0,
                            slice_beta_offset_div2: 0,
                            field: FieldPicSignal::FrameMbsOnly,
                            idr: k == 0,
                            nal_ref_idc: ref_idc,
                            long_term_reference_flag: false,
                            mmco: &[],
                            redundant_pic_cnt: rpc,
                            slice_group_change_cycle: change_cycle,
                        },
                    );
                }

                // Per-slice neighbour state (§6.4.8: other slices are
                // unavailable).
                let mut nc_grid = CavlcNcGrid::new(w_mbs, h_mbs);
                let mut intra_grid = IntraGrid::new(w_mbs as usize, h_mbs as usize);
                let mut mv_grid = MvGrid::new(w_mbs as usize, h_mbs as usize);
                let mut pending_skip: u32 = 0;
                let mut tracker = MbQpTracker::new(qp_y);
                // Redundant slices reconstruct onto a scratch copy.
                let (mut sy, mut su, mut sv) = if redundant {
                    (recon_y.clone(), recon_u.clone(), recon_v.clone())
                } else {
                    (Vec::new(), Vec::new(), Vec::new())
                };
                let mut is_intra_scratch = is_intra.clone();
                for &addr in mbs {
                    let (mb_x, mb_y) = (addr % w_mbs as usize, addr / w_mbs as usize);
                    let (ry, ru, rv, intra_flags) = if redundant {
                        (&mut sy, &mut su, &mut sv, &mut is_intra_scratch)
                    } else {
                        (&mut recon_y, &mut recon_u, &mut recon_v, &mut is_intra)
                    };
                    // §6.4.8 same-slice availability, plus §8.3.1.2
                    // constrained intra (P slices only).
                    let avail = |n: Option<usize>| -> bool {
                        n.is_some_and(|na| {
                            slice_of[na] == sid
                                && (!cfg.constrained_intra || !is_p || intra_flags[na])
                        })
                    };
                    let up = (mb_y > 0).then(|| addr - w_mbs as usize);
                    let left = (mb_x > 0).then(|| addr - 1);
                    let ul = (mb_y > 0 && mb_x > 0).then(|| addr - w_mbs as usize - 1);
                    let mask = (avail(up), avail(left), avail(ul));
                    let raster = (mb_y > 0, mb_x > 0, mb_y > 0 && mb_x > 0);
                    tracker.intra_nb = Some(mask);
                    let dbl = if is_p {
                        let pr = prev_ref.as_ref().expect("P reference");
                        e.encode_p_mb_with_intra_fallback(
                            &src,
                            pr,
                            pr.recon_y,
                            mb_x,
                            mb_y,
                            qp_y,
                            qp_c,
                            cw,
                            ch,
                            ry,
                            ru,
                            rv,
                            &mut sw,
                            &mut nc_grid,
                            &mut intra_grid,
                            &mut mv_grid,
                            &mut pending_skip,
                            &mut tracker,
                        )
                    } else {
                        e.encode_mb_intra16x16(
                            &src,
                            mb_x,
                            mb_y,
                            qp_y,
                            qp_c,
                            cw,
                            ch,
                            ry,
                            ru,
                            rv,
                            &mut sw,
                            &mut nc_grid,
                            &mut intra_grid,
                            0,
                            Some(mask),
                        )
                        .deblock
                    };
                    intra_flags[addr] = dbl.is_intra;
                    if !redundant {
                        infos[addr] = dbl;
                        if is_p && dbl.is_intra {
                            intra_mbs_in_p += 1;
                            if mask != raster {
                                masked_intra_mbs += 1;
                            }
                        }
                    }
                }
                if pending_skip > 0 {
                    sw.ue(pending_skip);
                }
                sw.rbsp_trailing_bits();
                let nal = build_nal_unit(ref_idc as u8, nal_type, &sw.into_bytes());
                if redundant {
                    redundant_nals.push(nal);
                } else {
                    nals.push(nal);
                }
            }
        }
        if cfg.aso {
            nals.reverse();
        }
        for n in nals.iter().chain(redundant_nals.iter()) {
            stream.extend_from_slice(n);
        }
        slices_per_picture.push(slice_mbs.len());

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
        prev = Some((recon_y.clone(), recon_u.clone(), recon_v.clone()));
        recon_frames.push((recon_y, recon_u, recon_v));
    }

    SlicesEncoded {
        annex_b: stream,
        recon_frames,
        slices_per_picture,
        masked_intra_mbs,
        intra_mbs_in_p,
    }
}
