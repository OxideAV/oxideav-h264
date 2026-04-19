//! CABAC entropy **encoder** for H.264.
//!
//! This is the mirror of the `crate::cabac` decode tree — every routine
//! here has a 1-to-1 decode counterpart (binarise + arithmetic encode vs.
//! binarise + arithmetic decode). See ITU-T H.264 (07/2019) §9.3 and
//! cross-reference x264's `encoder/cabac.c`.
//!
//! Layout:
//!
//! * [`engine`] — arithmetic encoder (§9.3.4.4) with `codILow` / `codIRange`
//!   state, renormalisation and the §9.3.4.2 "put-bit with follow"
//!   outstanding-bit mechanism. Also provides `encode_bypass` (§9.3.4.2)
//!   and `encode_terminate` (§9.3.4.5).
//! * [`binarize`] — per-syntax encoder wrappers. Each function mirrors a
//!   `cabac::binarize::decode_*` routine: binarise the value, invoke the
//!   right mix of regular-mode / bypass bins on the arithmetic encoder.
//! * [`mb`] — macroblock-level glue for CABAC I-slice emission. Uses the
//!   decoder's shared context-derivation helpers (`mb_type_i_ctx_idx_inc`
//!   et al) unchanged.
//!
//! The encoder only needs to support the subset of syntax the baseline
//! encoder emits: I-slice `mb_type` (I_16×16), `intra_chroma_pred_mode`,
//! `coded_block_pattern` (implicit for I_16×16), `mb_qp_delta`,
//! `coded_block_flag`, `significant_coeff_flag`,
//! `last_significant_coeff_flag`, `coeff_abs_level_minus1`,
//! `coeff_sign_flag`, plus the slice-end `end_of_slice_flag` terminate bin
//! (§9.3.3.2.4).

pub mod binarize;
pub mod engine;
pub mod mb;
