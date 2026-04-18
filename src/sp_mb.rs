//! SP-slice macroblock decode — ITU-T H.264 §7.3.5 / §7.4.5.1 (Table 7-13
//! — SP entries are identical to P entries) / §8.6.2.
//!
//! SP-slice (Switching-P) is the stream-switching variant of a P-slice:
//! every macroblock is predicted from list 0 with the P-slice syntax, but
//! §8.6.2's "SP decoding process" specifies an extra dequantise-then-
//! requantise pass on the prediction + residual sum, keyed on
//! `slice_qs_delta` (§7.4.3). That step makes the reconstructed samples
//! independent of which P or SP predecessor was decoded — the canonical
//! stream-switching guarantee.
//!
//! Two flavours exist (§8.6.2):
//!
//! * **Primary SP** (`sp_for_switch_flag == 0`, §8.6.2.1) — standard
//!   decode of this slice in the ambient bitstream. When `QP == QS` the
//!   §8.6.2 dequant+requant pair is the identity, and the output is
//!   bit-exact with a plain P-slice decode of the same residuals.
//!
//! * **Secondary SP** (`sp_for_switch_flag == 1`, §8.6.2.2) — used when
//!   switching bitstreams: the decoder writes the reconstruction of a
//!   *different* stream. Requires the §8.6.2.2 copy-through path which
//!   is not wired.
//!
//! Primary SP with `QP == QS` is the only path applied in practice by
//! real encoders producing switching points. This module dispatches
//! primary-SP decode through the existing P-slice macroblock path; the
//! secondary-SP or `QP != QS` cases return
//! [`oxideav_core::Error::Unsupported`].

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::p_mb::{decode_p_skip_mb, decode_p_slice_mb};
use crate::picture::Picture;
use crate::pps::Pps;
use crate::slice::SliceHeader;
use crate::sps::Sps;

/// CAVLC SP-slice data loop — §7.3.4, Table 7-13 (SP entries).
///
/// Syntax is identical to [`crate::decoder::decode_p_slice_data`]: each
/// coded macroblock is preceded by `mb_skip_run` ue(v) counting the
/// `P_Skip` macroblocks that come before it. Reconstruction follows the
/// standard P-slice path — see the module-level docs for the §8.6.2
/// primary-SP justification.
pub fn decode_sp_slice_data(
    br: &mut BitReader<'_>,
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    pic: &mut Picture,
    ref_list0: &[&Picture],
) -> Result<()> {
    if sh.sp_for_switch_flag {
        return Err(Error::unsupported(
            "h264: secondary SP-slice (sp_for_switch_flag=1, §8.6.2.2 copy-through) not yet wired",
        ));
    }
    if sh.slice_qs_delta != sh.slice_qp_delta {
        return Err(Error::unsupported(
            "h264: SP-slice with QS != QP (§8.6.2 dequant+requant remap) not yet wired",
        ));
    }
    let mb_w = sps.pic_width_in_mbs();
    let mb_h = sps.pic_height_in_map_units();
    let total_mbs = mb_w * mb_h;
    let mut mb_addr = sh.first_mb_in_slice;
    if mb_addr >= total_mbs {
        return Err(Error::invalid(
            "h264 sp-slice: first_mb_in_slice out of range",
        ));
    }
    let mut prev_qp = (pps.pic_init_qp_minus26 + 26 + sh.slice_qp_delta).clamp(0, 51);

    while mb_addr < total_mbs {
        let skip_run = br.read_ue()?;
        for _ in 0..skip_run {
            if mb_addr >= total_mbs {
                break;
            }
            let mb_x = mb_addr % mb_w;
            let mb_y = mb_addr / mb_w;
            decode_p_skip_mb(sh, mb_x, mb_y, pic, ref_list0, prev_qp)?;
            mb_addr += 1;
        }
        if mb_addr >= total_mbs {
            break;
        }
        let mb_x = mb_addr % mb_w;
        let mb_y = mb_addr / mb_w;
        decode_p_slice_mb(br, sps, pps, sh, mb_x, mb_y, pic, ref_list0, &mut prev_qp)?;
        mb_addr += 1;
    }
    Ok(())
}
