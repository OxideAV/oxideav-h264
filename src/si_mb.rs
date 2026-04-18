//! SI-slice macroblock decode — ITU-T H.264 §7.3.5 / §7.4.5 (Table 7-12) /
//! §8.6.1.
//!
//! SI-slice (Switching-Intra) is a stream-switching variant of an I-slice:
//! every macroblock is intra-coded, with one extra SI-specific macroblock
//! type (`mb_type = 0` → `SI`) that precedes the I-slice set. The encoded
//! macroblock types are:
//!
//! * `0` → `SI` — a 4×4 intra-predicted macroblock whose residual is
//!   dequantized with an SI-specific quantiser parameter `QS`
//!   (`slice_qs_delta`, §7.4.3, §8.6.1). Syntactically identical to
//!   `I_NxN`: the decoder reads 16 `prev_intra4x4_pred_mode_flag` (+
//!   optional `rem_intra4x4_pred_mode`), an `intra_chroma_pred_mode`, a
//!   CBP, an optional `mb_qp_delta`, and a CAVLC residual payload.
//! * `1..=26` → re-indexes against Table 7-11 with a one-slot offset
//!   (same mapping as I-slice `0..=25`).
//!
//! For a *primary* SI slice (the only kind anyone produces in practice),
//! §8.6.1's SP/SI reconstruction step collapses back to the standard
//! intra reconstruction when `QP == QS`: the §8.6.1 dequant+requant pair
//! is the identity map. Every fixture we generate pins `slice_qs_delta`
//! so that equality holds, so this module's SI mb_type decode simply
//! routes to the existing `decode_intra_mb_given_imb(INxN)` path.
//!
//! Secondary SI (used as a resync target for an SP switch) needs the full
//! §8.6.1 remap — that path is not wired and such slices return
//! [`oxideav_core::Error::Unsupported`].

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::mb::decode_intra_mb_given_imb;
use crate::mb_type::{decode_i_slice_mb_type, IMbType};
use crate::picture::Picture;
use crate::pps::Pps;
use crate::slice::SliceHeader;
use crate::sps::Sps;

/// Decode an SI-slice `mb_type` (Table 7-12). `mb_type = 0` selects the
/// SI intra macroblock (wire-identical to `I_NxN`); `mb_type >= 1`
/// re-indexes against Table 7-11 (`I_NxN` / `I_16x16_*` / `I_PCM`).
pub fn decode_si_slice_mb_type(mb_type: u32) -> Option<IMbType> {
    if mb_type == 0 {
        Some(IMbType::INxN)
    } else {
        decode_i_slice_mb_type(mb_type - 1)
    }
}

/// CAVLC SI-slice data loop. Mirrors [`crate::mb::decode_i_slice_data`] but
/// the per-MB `mb_type` routes through Table 7-12.
///
/// For primary SI slices `slice_qs_delta == slice_qp_delta` is enforced;
/// §8.6.1's dequant-then-requant then collapses to the identity, so the
/// existing intra reconstruction path returns bit-exact samples. A
/// secondary SI slice (with a non-trivial QS/QP pair) is rejected at the
/// first macroblock.
pub fn decode_si_slice_data(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
) -> Result<()> {
    if sh.slice_qs_delta != sh.slice_qp_delta {
        return Err(Error::unsupported(
            "h264: secondary SI-slice (QS != QP, §8.6.1 SP/SI remap) not yet wired",
        ));
    }
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = if sh.field_pic_flag {
        sps.pic_height_in_map_units()
    } else {
        sps.pic_height_in_mbs()
    };
    let total_mbs = mb_w * mb_h;
    if sh.first_mb_in_slice >= total_mbs {
        return Err(Error::invalid(
            "h264 si-slice: first_mb_in_slice out of range",
        ));
    }
    let mut prev_qp = (pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta).clamp(0, 51);
    let mut mb_addr = sh.first_mb_in_slice;
    while mb_addr < total_mbs {
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        let mb_type = br.read_ue()?;
        let imb = decode_si_slice_mb_type(mb_type)
            .ok_or_else(|| Error::invalid(format!("h264 si-slice: bad SI mb_type {mb_type}")))?;
        decode_intra_mb_given_imb(br, sps, pps, sh, mb_x, mb_y, pic, &mut prev_qp, imb)?;
        mb_addr += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn si_mb_type_zero_is_inxn() {
        assert!(matches!(decode_si_slice_mb_type(0), Some(IMbType::INxN)));
    }

    #[test]
    fn si_mb_type_one_maps_to_i_inxn() {
        // Table 7-12 mb_type = 1 → I-slice Table 7-11 mb_type = 0 = I_NxN.
        assert!(matches!(decode_si_slice_mb_type(1), Some(IMbType::INxN)));
    }

    #[test]
    fn si_mb_type_two_maps_to_i16x16_0_0_0() {
        // Table 7-12 mb_type = 2 → I-slice Table 7-11 mb_type = 1 = I_16x16_0_0_0.
        match decode_si_slice_mb_type(2).unwrap() {
            IMbType::I16x16 {
                intra16x16_pred_mode,
                cbp_luma,
                cbp_chroma,
            } => {
                assert_eq!(intra16x16_pred_mode, 0);
                assert_eq!(cbp_luma, 0);
                assert_eq!(cbp_chroma, 0);
            }
            other => panic!("expected I_16x16_0_0_0, got {:?}", other),
        }
    }

    #[test]
    fn si_mb_type_twenty_six_is_ipcm() {
        assert!(matches!(decode_si_slice_mb_type(26), Some(IMbType::IPcm)));
    }

    #[test]
    fn si_mb_type_twenty_seven_is_none() {
        assert!(decode_si_slice_mb_type(27).is_none());
    }
}
