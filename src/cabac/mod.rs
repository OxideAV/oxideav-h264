//! CABAC (Context-based Adaptive Binary Arithmetic Coding) for H.264.
//!
//! See ITU-T H.264 (07/2019) §9.3.

pub mod b_mb;
pub mod binarize;
pub mod context;
pub mod engine;
pub mod mb;
pub mod mb_hi;
pub mod p_mb;
pub mod residual;
pub mod tables;
