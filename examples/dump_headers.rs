// Scratch harness: parse an Annex B stream with the crate's NAL/slice
// header parsers and print one line per slice — coded picture
// structure (§7.4.3 field_pic_flag / bottom_field_flag), slice type,
// frame_num, POC LSB, direct flag, MBAFF (§7.4.2.1.1) — plus SPS/PPS
// summaries. Conformance-stream triage aid: shows the (FLD / FRM /
// AFRM) per-picture coding-structure sequence of Table 8-7 without
// running reconstruction.
use oxideav_h264::decoder::{Decoder, Event};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump_headers <annexb.264> [max]");
    let max: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let stream = std::fs::read(&path).expect("read stream");
    let mut dec = Decoder::new();
    let mut n = 0usize;
    for ev in dec.process_annex_b(&stream) {
        match ev {
            Ok(Event::SpsStored(id)) => println!("SPS id={id}"),
            Ok(Event::PpsStored(id)) => println!("PPS id={id}"),
            Ok(Event::Slice {
                nal_unit_type,
                nal_ref_idc,
                header: h,
                sps,
                ..
            }) => {
                if h.first_mb_in_slice == 0 {
                    n += 1;
                    if n > max {
                        break;
                    }
                }
                let structure = if h.field_pic_flag {
                    if h.bottom_field_flag {
                        "FLD-bot"
                    } else {
                        "FLD-top"
                    }
                } else if sps.mb_adaptive_frame_field_flag {
                    "AFRM"
                } else {
                    "FRM"
                };
                println!(
                    "pic#{:3} nal={} ref_idc={} {:7} type={:?} frame_num={} poc_lsb={} first_mb={} dsmvp={} nref=({},{})",
                    n - 1,
                    nal_unit_type,
                    nal_ref_idc,
                    structure,
                    h.slice_type,
                    h.frame_num,
                    h.pic_order_cnt_lsb,
                    h.first_mb_in_slice,
                    u8::from(h.direct_spatial_mv_pred_flag),
                    h.num_ref_idx_l0_active_minus1 + 1,
                    h.num_ref_idx_l1_active_minus1 + 1,
                );
            }
            Ok(_) => {}
            Err(e) => println!("ERR: {e:?}"),
        }
    }
}
