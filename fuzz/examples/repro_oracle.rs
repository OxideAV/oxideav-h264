//! Temporary repro helper: run one input through the libavcodec oracle
//! and our decoder, printing which arms fire. Not part of CI.

use oxideav_core::{CodecId, Decoder, Frame, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;
use oxideav_h264_fuzz::libavcodec;

fn main() {
    let path = std::env::args().nth(1).expect("usage: repro_oracle <file>");
    let data = std::fs::read(&path).expect("read input");
    println!(
        "input: {} bytes, libavcodec available: {}",
        data.len(),
        libavcodec::available()
    );

    match libavcodec::decode_h264(&data) {
        libavcodec::OracleResult::Unavailable => println!("oracle: UNAVAILABLE"),
        libavcodec::OracleResult::Rejected => println!("oracle: REJECTED"),
        libavcodec::OracleResult::Frame(f) => println!(
            "oracle: FRAME {}x{} chroma_log2=({},{})",
            f.width, f.height, f.chroma_w_log2, f.chroma_h_log2
        ),
    }

    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let pkt = Packet::new(0, TimeBase::new(1, 30), data.to_vec());
    match dec.send_packet(&pkt) {
        Err(e) => println!("ours: send_packet err: {e:?}"),
        Ok(()) => {
            if let Err(e) = dec.flush() {
                println!("ours: flush err: {e:?}");
            }
            let mut n = 0;
            loop {
                match dec.receive_frame() {
                    Ok(Frame::Video(v)) => {
                        n += 1;
                        println!(
                            "ours: FRAME #{n} planes={} stride0={} len0={}",
                            v.planes.len(),
                            v.planes.first().map(|p| p.stride).unwrap_or(0),
                            v.planes.first().map(|p| p.data.len()).unwrap_or(0)
                        );
                    }
                    Ok(_) => println!("ours: non-video frame"),
                    Err(e) => {
                        println!("ours: receive_frame end: {e:?}");
                        break;
                    }
                }
                if n > 8 {
                    break;
                }
            }
        }
    }
    println!("ours: decode_error_count = {}", dec.decode_error_count());
    for id in 0..32 {
        if let Some(sps) = dec.stored_sps(id) {
            println!(
                "ours: sps[{id}] profile={} chroma_idc={} bd_luma={} bd_chroma={} gaps_allowed={} poc_type={} w_mbs={} h_map={}",
                sps.profile_idc,
                sps.chroma_format_idc,
                sps.bit_depth_luma_minus8 + 8,
                sps.bit_depth_chroma_minus8 + 8,
                sps.gaps_in_frame_num_value_allowed_flag,
                sps.pic_order_cnt_type,
                sps.pic_width_in_mbs_minus1 + 1,
                sps.pic_height_in_map_units_minus1 + 1,
            );
        }
    }
    if let Some(sh) = &dec.last_slice {
        println!(
            "ours: last_slice type={:?} frame_num={} num_ref_l0={} first_mb={}",
            sh.slice_type,
            sh.frame_num,
            sh.num_ref_idx_l0_active_minus1 + 1,
            sh.first_mb_in_slice
        );
    }
}
