//! Pure-Rust H.264 / AVC (ITU-T H.264 | ISO/IEC 14496-10) decoder + Baseline
//! I-frame-only encoder.
//!
//! # What is decoded today
//!
//! * NAL unit framing in **both** Annex B (`00 00 [00] 01` start codes) and
//!   AVCC length-prefixed form (used inside MP4 `mdat`), emulation-prevention
//!   byte stripping (§7.4.1.1), and `AVCDecoderConfigurationRecord` parsing
//!   (ISO/IEC 14496-15 §5.2.4.1).
//! * SPS / PPS / slice-header parsing (§7.3.2.1.1, §7.3.2.2, §7.3.3).
//! * **I-slice** pixel reconstruction for **both CAVLC and CABAC** — intra
//!   prediction, residual decode, 4×4 IDCT + Hadamard, spec-accurate
//!   in-loop deblocking (§8.7). The CABAC I-path covers I_16×16, I_NxN and
//!   I_PCM (with the mid-stream arithmetic-engine re-init mandated by
//!   §9.3.1.2).
//! * **Spec-accurate §8.7 in-loop deblocking filter** — per-4-pixel-edge
//!   boundary strength (bS) derivation distinguishing MB-edge vs internal
//!   edges, intra/inter sides, non-zero residual, and MV/reference
//!   mismatch (quarter-pel delta ≥ 4 or different ref POC → bS = 1).
//!   Luma filtering uses the §8.7.2.2 normal filter for bS ∈ {1, 2, 3}
//!   and the strong filter for bS = 4 (MB-edge intra). Chroma filtering
//!   (§8.7.2.3) runs on both planes with `chroma_qp_index_offset` /
//!   `second_chroma_qp_index_offset` applied through the Table 8-15 QP
//!   map, and uses the `tC = tC0 + 1` chroma bump. Internal luma edges
//!   are transform-size-aware: under `transform_size_8x8_flag = 1` only
//!   the mid-MB edge at offset 8 is filtered (not offsets 4 and 12).
//! * **High Profile 8×8 transform** on the CAVLC path — intra AND inter.
//!   `transform_size_8x8_flag = 1` macroblocks (§7.3.5.3.2) decode as four
//!   interleaved CAVLC sub-blocks via the `zigzag_scan8x8_cavlc` table,
//!   with FFmpeg-style per-sub-block `non_zero_count_cache` neighbour
//!   indexing and post-decode `nnz[0] += nnz[1] + nnz[8] + nnz[9]`
//!   aggregation, then through the §8.5.13 dequant + inverse 8×8 integer
//!   transform. The intra path couples this to §8.3.2 Intra_8×8 prediction
//!   (with the (1,2,1) reference-sample pre-filter); the P-slice path
//!   applies the 8×8 residual on top of motion-compensated samples, gated
//!   on the §7.3.5.1 conditional (`pps.transform_8x8_mode_flag` + `cbp_luma
//!   > 0` + no sub-8×8 motion partitions for P_8x8). Bit-exact against
//!   ffmpeg's libavcodec reference on both a smpte-bars 128×128 IDR and a
//!   128×128 P-slice clip encoded with `-x264opts 8x8dct=1`.
//! * **CAVLC baseline P-slice** pixel reconstruction (§8.4) — P_L0_16×16 /
//!   16×8 / 8×16, P_8×8 (all four sub-partition shapes) and P_8×8ref0,
//!   P_Skip, plus intra-in-P macroblocks. Luma 6-tap half-pel + bilinear
//!   quarter-pel MC; bilinear 1/8-pel chroma MC; edge-replicated
//!   out-of-picture reads.
//! * **CABAC Main-profile P-slice** entropy decode (§9.3) —
//!   `mb_skip_flag`, `mb_type` (Table 9-36), `sub_mb_type` (Table 9-38),
//!   `ref_idx_l0` (unary + neighbour-indexed ctxIdxInc), `mvd_l0_{x,y}`
//!   (UEGk with §9.3.3.1.1.7 neighbour-magnitude ctxIdxInc),
//!   `coded_block_pattern`, `mb_qp_delta`, and 4×4 residual coding.
//!   `weighted_pred_flag = 1` is parsed but weights are not yet wired
//!   on the CABAC path.
//! * **CABAC High-profile 8×8 transform** (§9.3.3.1.1.10) — intra AND
//!   inter. `transform_size_8x8_flag` is binarised as FL(1) with
//!   `ctxIdxInc = condTermFlagA + condTermFlagB` against ctxIdxOffset 399
//!   (slots 399/400/401 per §9.3.3.1.1.10). The 8×8 luma residual is
//!   then read as a single 64-coefficient CABAC block with `ctxBlockCat =
//!   5`: `coded_block_flag` at ctxIdxOffset 1012, `significant_coeff_flag`
//!   at 402, `last_significant_coeff_flag` at 417, `coeff_abs_level_minus1`
//!   at 426. Per-position ctxIdxInc for sig/last uses the §9.3.3.1.3
//!   `ctxIdxMap[63][2]` table wired in [`cabac::residual::LUMA8X8_SIG_LAST_CTX_INC`].
//!   The reconstructed 8×8 block flows through `dequantize_8x8_scaled` +
//!   `idct_8x8`, matching the CAVLC 8×8 path bit-for-bit on the shared
//!   §8.5.13 math.
//! * **Multi-reference DPB** (Annex C / §8.2.5) — [`dpb::Dpb`] holds up
//!   to `sps.max_num_ref_frames` reconstructed reference frames. Each
//!   P-slice builds a fresh `RefPicList0` (§8.2.4.2.1) — short-term
//!   references first (descending `FrameNumWrap`), then long-term
//!   references (ascending `LongTermFrameIdx`) — and `ref_idx_l0 > 0`
//!   indexes into it. Picture order count is derived for
//!   `pic_order_cnt_type == 0`, `== 1` (§8.2.1.2 — with
//!   `offset_for_non_ref_pic`, `offset_for_top_to_bottom_field`, and the
//!   per-reference `offset_for_ref_frame[]` cycle applied to the slice
//!   header's `delta_pic_order_cnt[0..1]`), and `== 2` (§8.2.1.3).
//!   Sliding-window marking (§8.2.5.3) and MMCO operations 1 / 2 / 3 / 4
//!   / 5 / 6 (§8.2.5.4) are implemented.
//! * **Explicit weighted P-slice prediction** (§7.3.3.2 / §8.4.2.3.2) —
//!   the slice's `pred_weight_table` is parsed when the PPS has
//!   `weighted_pred_flag = 1`, and the per-reference (weight, offset,
//!   log2_denom) triples are applied to the MC output before the residual
//!   is added.
//! * **CAVLC B-slice** pixel reconstruction (§7.3.5 / §8.4) —
//!   `B_Direct_16x16` + `B_Skip` via both **spatial direct** MV derivation
//!   (§8.4.1.2.2, `direct_spatial_mv_pred_flag = 1`) and **temporal direct**
//!   derivation (§8.4.1.2.3, `direct_spatial_mv_pred_flag = 0`). The
//!   temporal-direct path walks the colocated 4×4 block in `RefPicList1[0]`,
//!   reads its MV + the POC of the ref it pointed to, and rescales via
//!   `DistScaleFactor` using POC arithmetic — intra-colocated blocks use
//!   zero MVs, long-term references short-circuit the scaler to identity.
//!   Also supported: `B_L0_16x16` / `B_L1_16x16` / `B_Bi_16x16`, the 16×8 /
//!   8×16 pair variants (L0/L0, L1/L1, L0/L1, L1/L0, L0/Bi, L1/Bi, Bi/L0,
//!   Bi/L1, Bi/Bi), and `B_8x8` with every Table 7-17 sub-partition type
//!   (`B_Direct_8x8`, `B_{L0,L1,Bi}_{8x8,8x4,4x8,4x4}`). Intra-in-B MBs
//!   dispatch back into the I-slice decode helper. Default bi-prediction
//!   averages the L0 and L1 samples per `(pred_L0 + pred_L1 + 1) >> 1`
//!   (§8.4.2.3.1); explicit weighted bipred (§8.4.2.3.2) uses the slice's
//!   `pred_weight_table` for both lists when `weighted_bipred_idc == 1`;
//!   **implicit** weighted bipred (§8.4.2.3.3) is derived from POC
//!   differences via `DistScaleFactor` when `weighted_bipred_idc == 2`,
//!   with a default-equal-weights fallback for long-term references or
//!   out-of-range `DistScaleFactor`.
//! * **B-slice RefPicList0 / RefPicList1** construction per §8.2.4.2.3 —
//!   short-term past/future split around `CurrPicOrderCnt` with the
//!   per-list ordering reversal, followed by ascending long-term
//!   references.
//! * **Reference picture list modification** (RPLM, §7.3.3.1 /
//!   §8.2.4.3) — the per-slice `ref_pic_list_modification` commands
//!   (idc = 0 short-term subtract, idc = 1 short-term add, idc = 2
//!   long-term select) are parsed into [`slice::SliceHeader::rplm_l0`] /
//!   [`slice::SliceHeader::rplm_l1`] and applied over the default-built
//!   lists via [`dpb::apply_rplm`]. Both P- and B-slice list 0 / list 1
//!   reordering are supported.
//! * **POC output reordering** — the DPB's pending-output queue is
//!   parameterised by a reorder window that the decoder bumps up on
//!   the first B-slice. Pictures emerge in POC order via
//!   `receive_frame`, letting a `I B B P B B …` decode-order stream
//!   deliver frames in presentation order. The SPS VUI's
//!   `bitstream_restriction.max_num_reorder_frames` (§E.2.1) is now
//!   parsed and used as the window size when present; the old
//!   `max_num_ref_frames` upper bound remains as a conservative
//!   fallback for streams that don't carry the VUI field.
//! * **Custom scaling lists** (§7.4.2.1.1.1 / §7.4.2.2) — the SPS and
//!   PPS scaling-list matrices are parsed out of their zig-zag
//!   coded form into raster order, resolved per Table 7-2 (PPS
//!   overrides SPS; rule A falls back to `Default_4x4_Intra` /
//!   `_Inter` or `Default_8x8_Intra` / `_Inter`; rule B inherits
//!   from the previous slot), and applied by the 4×4 / 8×8 dequant
//!   functions. Luma Intra_16×16 DC and chroma 2×2 DC also consume
//!   the per-category `(0,0)` weight so the full §8.5.10 / §8.5.11.1
//!   dequant chain honours non-flat matrices. Round-trips a
//!   64×64 IDR encoded with x264 `cqmfile=` custom matrices to within
//!   ±2 of the ffmpeg-decoded reference.
//!
//! # What is encoded today
//!
//! See [`encoder::H264Encoder`] — Baseline Profile, I-frames only,
//! Intra_16×16 DC_PRED, CAVLC, single slice per frame, 4:2:0, 8-bit,
//! fixed QP, deblocking disabled. The encoder and decoder round-trip
//! to PSNR ≥ 28 dB for gradient content at QP 22 and ≥ 40 dB for solid
//! content.
//!
//! # Out of scope (returns `Error::Unsupported` or the encoder refuses)
//!
//! * CABAC B-slices (decode); any CABAC encoding; any P/B slice encoding.
//!   CABAC P-slices are wired (including 8×8 transform) but weighted-P
//!   on the CABAC path is out of scope on this first pass.
//! * B-slice MBAFF and B-slice 8×8 transform (consistent with the
//!   existing 8×8 scoping).
//! * Interlaced coding / MBAFF / PAFF.
//! * 4:2:2 / 4:4:4 chroma, bit depths above 8.
//! * Rate control, adaptive QP, mode decision (Intra4×4 / Plane / Vertical /
//!   Horizontal), or any psychovisual tuning. The encoder always emits
//!   Intra_16×16 with DC_PRED and a fixed QP.
//!
//! This crate has no runtime dependencies beyond `oxideav-core` and
//! `oxideav-codec`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::derivable_impls)]

pub mod b_mb;
pub mod bitreader;
pub mod bitwriter;
pub mod cabac;
pub mod cavlc;
pub mod cavlc_enc;
pub mod deblock;
pub mod decoder;
pub mod dpb;
pub mod encoder;
pub mod fwd_transform;
pub mod intra_pred;
pub mod mb;
pub mod mb_type;
pub mod motion;
pub mod nal;
pub mod p_mb;
pub mod picture;
pub mod pps;
pub mod scaling_list;
pub mod slice;
pub mod sps;
pub mod tables;
pub mod transform;

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecCapabilities, CodecId, PixelFormat};

/// The canonical oxideav codec id for H.264 / AVC video.
pub const CODEC_ID_STR: &str = "h264";

/// Register this crate's decoder and baseline I-only encoder with a codec
/// registry. Both are registered under the same `"h264"` id; the registry's
/// preference + priority logic picks one at resolve time.
pub fn register(reg: &mut CodecRegistry) {
    let dec_caps = CodecCapabilities::video("h264_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(8192, 8192);
    reg.register_decoder_impl(CodecId::new(CODEC_ID_STR), dec_caps, decoder::make_decoder);

    // Encoder: Baseline I-only. The registered caps advertise
    // `intra_only = true` and cap the output at level 3.0 maxima.
    let enc_caps = CodecCapabilities::video("h264_baseline_i_sw")
        .with_lossy(true)
        .with_intra_only(true)
        .with_max_size(720, 576)
        .with_pixel_format(PixelFormat::Yuv420P);
    reg.register_encoder_impl(CodecId::new(CODEC_ID_STR), enc_caps, encoder::make_encoder);
}
