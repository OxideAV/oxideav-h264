//! Integration test for explicit weighted P-slice prediction
//! (ITU-T H.264 §8.4.2.3.2).
//!
//! We build a slice header with a `pred_weight_table` that halves luma
//! brightness (weight = 1, log2_denom = 1, offset = 0), drive one P_Skip
//! macroblock through [`oxideav_h264::p_mb::decode_p_skip_mb`] with a
//! known reference picture, and assert the decoded samples are the
//! reference halved (within the §8.4.2.3.2 rounding rule).
//!
//! Explicit-mode only — implicit bi-prediction (§8.4.2.3.3) is out of scope.

use oxideav_h264::p_mb::decode_p_skip_mb;
use oxideav_h264::picture::Picture;
use oxideav_h264::slice::{
    ChromaWeight, LumaWeight, PredWeightTable, SliceHeader, SliceType,
};

fn make_slice_with_luma_weight(lw: LumaWeight, cw: ChromaWeight) -> SliceHeader {
    let tbl = PredWeightTable {
        luma_log2_weight_denom: lw.log2_denom,
        chroma_log2_weight_denom: cw.log2_denom,
        luma_l0: vec![lw],
        chroma_l0: vec![cw],
        luma_l1: Vec::new(),
        chroma_l1: Vec::new(),
    };
    SliceHeader {
        first_mb_in_slice: 0,
        slice_type_raw: 0,
        slice_type: SliceType::P,
        pic_parameter_set_id: 0,
        colour_plane_id: 0,
        frame_num: 0,
        field_pic_flag: false,
        bottom_field_flag: false,
        idr_pic_id: 0,
        pic_order_cnt_lsb: 0,
        delta_pic_order_cnt_bottom: 0,
        delta_pic_order_cnt: [0, 0],
        redundant_pic_cnt: 0,
        direct_spatial_mv_pred_flag: false,
        num_ref_idx_active_override_flag: false,
        num_ref_idx_l0_active_minus1: 0,
        num_ref_idx_l1_active_minus1: 0,
        cabac_init_idc: 0,
        slice_qp_delta: 0,
        sp_for_switch_flag: false,
        slice_qs_delta: 0,
        disable_deblocking_filter_idc: 0,
        slice_alpha_c0_offset_div2: 0,
        slice_beta_offset_div2: 0,
        slice_data_bit_offset: 0,
        is_idr: false,
        pred_weight_table: Some(tbl),
        idr_no_output_of_prior_pics_flag: false,
        idr_long_term_reference_flag: false,
        adaptive_ref_pic_marking_mode_flag: false,
        mmco_commands: Vec::new(),
    }
}

#[test]
fn p_skip_halves_luma_and_chroma_with_weight_one_denom_one() {
    let mb_w = 2u32;
    let mb_h = 2u32;

    let mut reference = Picture::new(mb_w, mb_h);
    // Ramp luma so we can distinguish scaling from copy: value = 8 * x.
    for y in 0..reference.height as usize {
        for x in 0..reference.width as usize {
            let stride = reference.luma_stride();
            reference.y[y * stride + x] = (8 * x).min(248) as u8;
        }
    }
    for p in reference.cb.iter_mut() {
        *p = 200;
    }
    for p in reference.cr.iter_mut() {
        *p = 80;
    }

    let mut pic = Picture::new(mb_w, mb_h);

    // weight=1, log2_denom=1, offset=0 → (x + 1) >> 1 (rounded halving).
    let lw = LumaWeight {
        weight: 1,
        offset: 0,
        log2_denom: 1,
        present: true,
    };
    // For chroma we also halve.
    let cw = ChromaWeight {
        weight: [1, 1],
        offset: [0, 0],
        log2_denom: 1,
        present: true,
    };
    let sh = make_slice_with_luma_weight(lw, cw);

    // Apply P_Skip at every MB — the predictor will be (0,0) because no prior
    // MB has been decoded (§8.4.1.1 degenerate case), which means the decoded
    // samples come from the collocated reference samples, then get weighted.
    for mby in 0..mb_h {
        for mbx in 0..mb_w {
            decode_p_skip_mb(&sh, mbx, mby, &mut pic, &[&reference], 26).expect("p_skip");
        }
    }

    // Verify luma was halved (with rounding).
    let lstride = pic.luma_stride();
    for y in 0..pic.height as usize {
        for x in 0..pic.width as usize {
            let src = reference.y[y * lstride + x] as i32;
            let expected = ((src + 1) >> 1) as u8;
            let got = pic.y[y * lstride + x];
            assert_eq!(
                got, expected,
                "luma mismatch at ({x},{y}) src={src} expected={expected} got={got}"
            );
        }
    }

    // Verify chroma Cb / Cr halved: 200/2 ≈ 100 with rounding, 80/2 = 40.
    let cstride = pic.chroma_stride();
    for y in 0..(pic.height / 2) as usize {
        for x in 0..(pic.width / 2) as usize {
            assert_eq!(pic.cb[y * cstride + x], 100, "cb at ({x},{y})");
            assert_eq!(pic.cr[y * cstride + x], 40, "cr at ({x},{y})");
        }
    }
}

#[test]
fn p_skip_additive_offset_only() {
    let mb_w = 1u32;
    let mb_h = 1u32;
    let mut reference = Picture::new(mb_w, mb_h);
    for p in reference.y.iter_mut() {
        *p = 100;
    }
    for p in reference.cb.iter_mut() {
        *p = 100;
    }
    for p in reference.cr.iter_mut() {
        *p = 100;
    }
    let mut pic = Picture::new(mb_w, mb_h);

    // weight = 1<<denom, offset = 30 → luma becomes x + 30 (clipped).
    let lw = LumaWeight {
        weight: 1 << 4,
        offset: 30,
        log2_denom: 4,
        present: true,
    };
    let cw = ChromaWeight {
        weight: [1 << 3, 1 << 3],
        offset: [-20, 50],
        log2_denom: 3,
        present: true,
    };
    let sh = make_slice_with_luma_weight(lw, cw);

    decode_p_skip_mb(&sh, 0, 0, &mut pic, &[&reference], 26).expect("p_skip");

    let lstride = pic.luma_stride();
    for y in 0..pic.height as usize {
        for x in 0..pic.width as usize {
            assert_eq!(pic.y[y * lstride + x], 130, "luma at ({x},{y})");
        }
    }
    let cstride = pic.chroma_stride();
    for y in 0..(pic.height / 2) as usize {
        for x in 0..(pic.width / 2) as usize {
            // Cb: x + -20 = 80
            assert_eq!(pic.cb[y * cstride + x], 80);
            // Cr: x + 50 = 150
            assert_eq!(pic.cr[y * cstride + x], 150);
        }
    }
}

#[test]
fn p_skip_without_weight_table_is_plain_copy() {
    // If the slice has no pred_weight_table, MC output must equal the
    // reference at MV (0, 0) — no accidental scaling.
    let mb_w = 1u32;
    let mb_h = 1u32;
    let mut reference = Picture::new(mb_w, mb_h);
    for p in reference.y.iter_mut() {
        *p = 77;
    }
    for p in reference.cb.iter_mut() {
        *p = 200;
    }
    for p in reference.cr.iter_mut() {
        *p = 50;
    }
    let mut pic = Picture::new(mb_w, mb_h);

    let sh = SliceHeader {
        first_mb_in_slice: 0,
        slice_type_raw: 0,
        slice_type: SliceType::P,
        pic_parameter_set_id: 0,
        colour_plane_id: 0,
        frame_num: 0,
        field_pic_flag: false,
        bottom_field_flag: false,
        idr_pic_id: 0,
        pic_order_cnt_lsb: 0,
        delta_pic_order_cnt_bottom: 0,
        delta_pic_order_cnt: [0, 0],
        redundant_pic_cnt: 0,
        direct_spatial_mv_pred_flag: false,
        num_ref_idx_active_override_flag: false,
        num_ref_idx_l0_active_minus1: 0,
        num_ref_idx_l1_active_minus1: 0,
        cabac_init_idc: 0,
        slice_qp_delta: 0,
        sp_for_switch_flag: false,
        slice_qs_delta: 0,
        disable_deblocking_filter_idc: 0,
        slice_alpha_c0_offset_div2: 0,
        slice_beta_offset_div2: 0,
        slice_data_bit_offset: 0,
        is_idr: false,
        pred_weight_table: None,
        idr_no_output_of_prior_pics_flag: false,
        idr_long_term_reference_flag: false,
        adaptive_ref_pic_marking_mode_flag: false,
        mmco_commands: Vec::new(),
    };

    decode_p_skip_mb(&sh, 0, 0, &mut pic, &[&reference], 26).expect("p_skip");

    assert!(pic.y.iter().all(|&v| v == 77));
    assert!(pic.cb.iter().all(|&v| v == 200));
    assert!(pic.cr.iter().all(|&v| v == 50));
}
