// Scratch harness: parse the FIRST coded picture of an Annex B stream
// down to slice_data level and print one line per macroblock —
// address, §7.4.4 mb_field_decoding_flag, mb_type, CBP. MBAFF/PAFF
// triage aid (shows the per-pair frame/field decisions and the mb_type
// walk without running pixel reconstruction).
use oxideav_h264::decoder::{Decoder, Event};
use oxideav_h264::slice_data::parse_slice_data;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump_mbs <annexb.264> [n_mbs] [pic_index]");
    let max: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let pic_index: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let stream = std::fs::read(&path).expect("read stream");
    let mut dec = Decoder::new();
    let mut n_pic = 0usize;
    for ev in dec.process_annex_b(&stream) {
        if let Ok(Event::Slice {
            header,
            rbsp,
            slice_data_cursor,
            pps,
            sps,
            ..
        }) = ev
        {
            if header.first_mb_in_slice == 0 {
                if n_pic > pic_index {
                    break;
                }
                n_pic += 1;
            }
            if n_pic != pic_index + 1 {
                continue;
            }
            println!(
                "slice: type={:?} field={} bottom={} first_mb={} mbaff_sps={}",
                header.slice_type,
                header.field_pic_flag,
                header.bottom_field_flag,
                header.first_mb_in_slice,
                sps.mb_adaptive_frame_field_flag,
            );
            match parse_slice_data(
                &rbsp,
                slice_data_cursor.0,
                slice_data_cursor.1,
                &header,
                &sps,
                &pps,
            ) {
                Ok(sd) => {
                    for (i, (mb, field)) in sd
                        .macroblocks
                        .iter()
                        .zip(sd.mb_field_decoding_flags.iter())
                        .take(max)
                        .enumerate()
                    {
                        println!(
                            "mb {:4} field={} type={:?} cbp={:#x} qp_delta={}",
                            i, field, mb.mb_type, mb.coded_block_pattern, mb.mb_qp_delta
                        );
                    }
                    println!("total mbs parsed: {}", sd.macroblocks.len());
                }
                Err(e) => println!("parse_slice_data error: {e:?}"),
            }
        }
    }
}
