//! Round-448 — **separate-colour-plane** (§7.4.2.1.1
//! `separate_colour_plane_flag = 1`) encoding driver.
//!
//! A separate-colour-plane stream codes the three colour components of
//! the 4:4:4 format as three independent monochrome-coded planes: the
//! SPS carries the on-wire pair (`chroma_format_idc = 3`,
//! `separate_colour_plane_flag = 1`), so ChromaArrayType is 0 and
//! every coded slice uses the monochrome syntax, with the §7.4.3
//! `colour_plane_id` u(2) field between `pic_parameter_set_id` and
//! `frame_num` naming the plane (0 = Y, 1 = Cb, 2 = Cr). There is no
//! coding dependency between planes (§7.4.2.1.1 NOTE 4), and the §7.4.1.2
//! access-unit constraint only asks that, per `colour_plane_id`, slices
//! appear in increasing `first_mb_in_slice` order — this driver emits
//! each access unit as the plane-0 slice, then plane 1, then plane 2.
//!
//! Structure of the emitted stream (Annex B):
//!
//! * SPS (High 4:4:4 Predictive, 244) + PPS — shared by all three
//!   planes, emitted once by the plane-0 encoder (the three per-plane
//!   encoders are configured identically, so their SPS/PPS bytes are
//!   byte-identical; planes 1 / 2 have theirs stripped).
//! * Access unit 0: IDR slices for planes 0 / 1 / 2.
//! * Access unit k > 0 (when P frames are requested): P slices for
//!   planes 0 / 1 / 2, each predicting from ITS OWN plane's previous
//!   reconstruction — the §8.1 three-invocation decode model in
//!   encoder form.
//!
//! Both entropy coders are supported (`ScpConfig::cabac`); each plane
//! runs the full round-448 monochrome pipeline (Intra_16x16 RDO IDR,
//! P_Skip / P_L0_16x16 quarter-pel ME, luma-only §8.7 deblocking).

use crate::encoder::{EncodedFrameRef, EncodedIdr, EncodedP, Encoder, EncoderConfig, YuvFrame};

/// Configuration for [`encode_scp_sequence`].
#[derive(Debug, Clone)]
pub struct ScpConfig {
    /// Luma width in samples (multiple of 16). All three planes share
    /// the full picture geometry (4:4:4).
    pub width: u32,
    /// Luma height in samples (multiple of 16).
    pub height: u32,
    /// Slice QP_Y — per the §7.4.5 NOTE on ChromaArrayType 0, every
    /// plane (including Cb / Cr) decodes with the luma quantisation
    /// parameter derivation, no chroma offset applied.
    pub qp: i32,
    /// When `true`, use the CABAC entropy coder (`encode_idr_cabac` /
    /// `encode_p_cabac`); otherwise CAVLC.
    pub cabac: bool,
    /// When `true`, code an IDR-B-P mini-GOP (requires exactly THREE
    /// input frames, display order 0 / 1 / 2): access units are
    /// emitted in decode order IDR(0), P(2), B(1), with each plane's B
    /// slice bi-predicting from ITS OWN plane's IDR and P
    /// reconstructions (`RefPicList0[0]` = IDR, `RefPicList1[0]` = P).
    /// `direct_temporal` picks §8.4.1.2.3 temporal (else §8.4.1.2.2
    /// spatial) direct derivation inside each plane.
    pub b_frame: bool,
    /// §7.4.3 `direct_spatial_mv_pred_flag` = 0 (temporal direct) when
    /// `true`. Only consulted with `b_frame`.
    pub direct_temporal: bool,
    /// Round-448 — when `true` the PPS signals
    /// `transform_8x8_mode_flag = 1` and every plane's inter path runs
    /// the §8.6.4 8x8-vs-4x4 luma trial (the §7.3.5 second-gate flag
    /// is coded on qualifying MBs). Per §8.5.9 with
    /// `separate_colour_plane_flag = 1` the 8x8 dequant of plane p
    /// uses scaling list `2 * p + mbIsInterFlag` — flat here.
    pub transform_8x8: bool,
}

/// One encoded separate-colour-plane sequence plus the per-frame
/// three-plane reconstruction the decoder must reproduce byte-exactly.
pub struct ScpEncoded {
    /// The full Annex B byte stream.
    pub annex_b: Vec<u8>,
    /// Per display frame: the [Y, Cb, Cr] plane reconstructions
    /// (each `width * height` bytes — full-resolution 4:4:4 planes).
    pub recon_frames: Vec<[Vec<u8>; 3]>,
}

/// Split an Annex B byte stream into NAL units (payload slices,
/// start codes excluded).
fn split_annex_b(data: &[u8]) -> Vec<&[u8]> {
    let mut nals = Vec::new();
    let mut i = 0usize;
    let mut start: Option<usize> = None;
    while i + 2 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            if let Some(s) = start {
                let mut end = i;
                // A 4-byte start code's leading zero belongs to the
                // delimiter, not the previous NAL.
                if end > s && data[end - 1] == 0 {
                    end -= 1;
                }
                nals.push(&data[s..end]);
            }
            start = Some(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    if let Some(s) = start {
        nals.push(&data[s..]);
    }
    nals
}

/// Re-emit `nals` as an Annex B stream, keeping only NAL units whose
/// `nal_unit_type` passes `keep`.
fn filter_nals(data: &[u8], keep: impl Fn(u8) -> bool) -> Vec<u8> {
    let mut out = Vec::new();
    for nal in split_annex_b(data) {
        let Some(&h) = nal.first() else { continue };
        if keep(h & 0x1F) {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(nal);
        }
    }
    out
}

/// Encode `frames` (full-resolution Y / Cb / Cr planes, 4:4:4) as a
/// separate-colour-plane stream: frame 0 is the IDR access unit,
/// every later frame a P access unit predicting plane-wise from its
/// predecessor.
pub fn encode_scp_sequence(cfg: &ScpConfig, frames: &[(&[u8], &[u8], &[u8])]) -> ScpEncoded {
    assert!(!frames.is_empty(), "need at least one frame");
    let mk_enc = |plane: u8| {
        Encoder::new(EncoderConfig {
            chroma_format_idc: 0,
            profile_idc: 244,
            colour_plane_id: Some(plane),
            cabac: cfg.cabac,
            qp: cfg.qp,
            direct_temporal_mv_pred: cfg.direct_temporal,
            transform_8x8: cfg.transform_8x8,
            max_num_ref_frames: 2,
            ..EncoderConfig::new(cfg.width, cfg.height)
        })
    };
    let encs = [mk_enc(0), mk_enc(1), mk_enc(2)];
    let plane_src = |k: usize, p: usize| -> &[u8] {
        match p {
            0 => frames[k].0,
            1 => frames[k].1,
            _ => frames[k].2,
        }
    };
    // Wrap one plane's samples as a monochrome input frame.
    fn plane_yuv<'a>(w: u32, h: u32, y: &'a [u8]) -> YuvFrame<'a> {
        YuvFrame {
            width: w,
            height: h,
            y,
            u: &[],
            v: &[],
        }
    }

    let mut annex_b: Vec<u8> = Vec::new();
    let mut recon_frames: Vec<[Vec<u8>; 3]> = Vec::new();

    // ---- Access unit 0: IDR, planes 0..3 in order. ----
    let mut idrs: Vec<EncodedIdr> = Vec::with_capacity(3);
    for (p, enc) in encs.iter().enumerate() {
        let f = plane_yuv(cfg.width, cfg.height, plane_src(0, p));
        let idr = if cfg.cabac {
            enc.encode_idr_cabac(&f)
        } else {
            enc.encode_idr(&f)
        };
        if p == 0 {
            // Plane 0 carries the shared SPS + PPS.
            annex_b.extend_from_slice(&idr.annex_b);
        } else {
            // Planes 1 / 2: identical parameter sets — strip them,
            // keep the coded slice NAL(s) only.
            annex_b.extend_from_slice(&filter_nals(&idr.annex_b, |t| t != 7 && t != 8));
        }
        idrs.push(idr);
    }
    recon_frames.push([
        idrs[0].recon_y.clone(),
        idrs[1].recon_y.clone(),
        idrs[2].recon_y.clone(),
    ]);

    // ---- IDR-B-P mini-GOP mode (round-448 B leg). ----
    if cfg.b_frame {
        assert_eq!(
            frames.len(),
            3,
            "b_frame mode codes exactly one IDR-B-P mini-GOP (3 frames)"
        );
        // Decode order: P anchor (display 2, frame_num 1, POC lsb 4)…
        let mut ps: Vec<EncodedP> = Vec::with_capacity(3);
        for (p, enc) in encs.iter().enumerate() {
            let f = plane_yuv(cfg.width, cfg.height, plane_src(2, p));
            let coded = if cfg.cabac {
                enc.encode_p_cabac(&f, &EncodedFrameRef::from(&idrs[p]), 1, 4)
            } else {
                enc.encode_p(&f, &EncodedFrameRef::from(&idrs[p]), 1, 4)
            };
            annex_b.extend_from_slice(&coded.annex_b);
            ps.push(coded);
        }
        // …then the non-reference B (display 1, frame_num 1, POC lsb 2).
        let mut recon_b: Vec<Vec<u8>> = Vec::with_capacity(3);
        for (p, enc) in encs.iter().enumerate() {
            let f = plane_yuv(cfg.width, cfg.height, plane_src(1, p));
            let l0 = EncodedFrameRef::from(&idrs[p]);
            let l1 = EncodedFrameRef::from(&ps[p]);
            let coded = if cfg.cabac {
                enc.encode_b_cabac(&f, &l0, &l1, 1, 2)
            } else {
                enc.encode_b(&f, &l0, &l1, 1, 2)
            };
            annex_b.extend_from_slice(&coded.annex_b);
            recon_b.push(coded.recon_y.clone());
        }
        // Display order: IDR (already pushed), B, P.
        recon_frames.push([recon_b[0].clone(), recon_b[1].clone(), recon_b[2].clone()]);
        recon_frames.push([
            ps[0].recon_y.clone(),
            ps[1].recon_y.clone(),
            ps[2].recon_y.clone(),
        ]);
        return ScpEncoded {
            annex_b,
            recon_frames,
        };
    }

    // ---- Access units 1..: P, each plane predicting from itself. ----
    let mut prev_p: Option<[EncodedP; 3]> = None;
    for k in 1..frames.len() {
        let mut aus: Vec<EncodedP> = Vec::with_capacity(3);
        for (p, enc) in encs.iter().enumerate() {
            let f = plane_yuv(cfg.width, cfg.height, plane_src(k, p));
            let prev_ref = match &prev_p {
                Some(ps) => EncodedFrameRef::from(&ps[p]),
                None => EncodedFrameRef::from(&idrs[p]),
            };
            let coded = if cfg.cabac {
                enc.encode_p_cabac(&f, &prev_ref, k as u32, 2 * k as u32)
            } else {
                enc.encode_p(&f, &prev_ref, k as u32, 2 * k as u32)
            };
            annex_b.extend_from_slice(&coded.annex_b);
            aus.push(coded);
        }
        recon_frames.push([
            aus[0].recon_y.clone(),
            aus[1].recon_y.clone(),
            aus[2].recon_y.clone(),
        ]);
        let arr: [EncodedP; 3] = aus.try_into().map_err(|_| ()).expect("three planes");
        prev_p = Some(arr);
    }

    ScpEncoded {
        annex_b,
        recon_frames,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_annex_b_handles_3_and_4_byte_start_codes() {
        let data = [
            0, 0, 0, 1, 0x67, 0xAA, // SPS-ish
            0, 0, 1, 0x68, 0xBB, // PPS-ish (3-byte start code)
            0, 0, 0, 1, 0x65, 0xCC, 0xDD, // slice-ish
        ];
        let nals = split_annex_b(&data);
        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0], &[0x67, 0xAA][..]);
        assert_eq!(nals[1], &[0x68, 0xBB][..]);
        assert_eq!(nals[2], &[0x65, 0xCC, 0xDD][..]);
    }

    #[test]
    fn filter_nals_drops_parameter_sets() {
        let data = [
            0, 0, 0, 1, 0x67, 0xAA, 0, 0, 0, 1, 0x68, 0xBB, 0, 0, 0, 1, 0x65, 0xCC,
        ];
        let out = filter_nals(&data, |t| t != 7 && t != 8);
        assert_eq!(out, vec![0, 0, 0, 1, 0x65, 0xCC]);
    }
}
