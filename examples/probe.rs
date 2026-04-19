//! Standalone debug harness for the CABAC I-slice correctness bug.
//!
//! Feeds an Annex-B H.264 elementary stream into `oxideav-h264` one
//! frame at a time and reports:
//!
//! * which frame first returns a `send_packet` error (if any),
//! * how many bytes of the decoded YUV differ from an external
//!   reference (if a reference YUV is provided),
//! * the maximum absolute deviation per decoded frame.
//!
//! Intended as a bisect tool for the CABAC desync tracked in
//! `tests/cabac_high_regression.rs`. Produce the inputs with:
//!
//! ```sh
//! ffmpeg -y -f lavfi -i testsrc=duration=1:size=WIDTHxHEIGHT:rate=24 \
//!   -vframes N -pix_fmt yuv420p -c:v libx264 \
//!   -profile:v high -level 4.0 -preset medium -coder 1 -bf 2 -refs 1 -g N \
//!   /tmp/in.mp4
//! ffmpeg -y -i /tmp/in.mp4 -c:v copy -bsf:v h264_mp4toannexb -f h264 /tmp/in.es
//! ffmpeg -y -i /tmp/in.mp4 -f rawvideo -pix_fmt yuv420p /tmp/in.yuv
//!
//! cargo run --release -p oxideav-h264 --example probe -- \
//!     /tmp/in.es WIDTH HEIGHT /tmp/in.yuv
//! ```
//!
//! The reference YUV argument is optional. When omitted the tool
//! just reports which frame breaks decoding.

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_h264::nal::{split_annex_b, NalHeader, NalUnitType};
use std::env;

fn main() {
    let mut args = env::args().skip(1);
    let es_path = args
        .next()
        .expect("usage: probe <file.es> <width> <height> [reference.yuv]");
    let width: usize = args
        .next()
        .expect("width")
        .parse()
        .expect("width is an integer");
    let height: usize = args
        .next()
        .expect("height")
        .parse()
        .expect("height is an integer");
    let ref_yuv_path = args.next();

    let frame_bytes_yuv = width * height * 3 / 2;
    let es = std::fs::read(&es_path).expect("read es");
    let ref_yuv = ref_yuv_path
        .as_ref()
        .map(|p| std::fs::read(p).expect("read reference yuv"));

    let nalus = split_annex_b(&es);
    let mut sps: Option<Vec<u8>> = None;
    let mut pps: Option<Vec<u8>> = None;
    let mut frames: Vec<Vec<u8>> = Vec::new();
    for nalu in &nalus {
        if nalu.is_empty() {
            continue;
        }
        let h = NalHeader::parse(nalu[0]).unwrap();
        match h.nal_unit_type {
            NalUnitType::Sps => sps = Some(nalu.to_vec()),
            NalUnitType::Pps => pps = Some(nalu.to_vec()),
            NalUnitType::SliceIdr | NalUnitType::SliceNonIdr => {
                let mut frame = Vec::new();
                frame.extend_from_slice(&[0, 0, 0, 1]);
                frame.extend_from_slice(nalu);
                frames.push(frame);
            }
            _ => {}
        }
    }
    println!(
        "probe: {width}×{height}  sps={:?}B  pps={:?}B  frames={}",
        sps.as_ref().map(|v| v.len()),
        pps.as_ref().map(|v| v.len()),
        frames.len()
    );

    let mut params = CodecParameters::video(CodecId::new("h264"));
    params.extradata.clear();
    let mut dec = oxideav_h264::decoder::make_decoder(&params).expect("make_decoder");

    // Prime SPS + PPS as one combined Annex-B packet.
    let mut header = Vec::new();
    header.extend_from_slice(&[0, 0, 0, 1]);
    header.extend_from_slice(sps.as_ref().expect("SPS missing"));
    header.extend_from_slice(&[0, 0, 0, 1]);
    header.extend_from_slice(pps.as_ref().expect("PPS missing"));
    let pkt = Packet::new(0, TimeBase::new(1, 24), header);
    let _ = dec.send_packet(&pkt);

    let mut display_idx = 0usize;
    for (i, frame_bytes) in frames.iter().enumerate() {
        let pkt = Packet::new(0, TimeBase::new(1, 24), frame_bytes.clone());
        match dec.send_packet(&pkt) {
            Ok(()) => {
                while let Ok(Frame::Video(vf)) = dec.receive_frame() {
                    if let Some(ref yuv) = ref_yuv {
                        compare_frame_against_ref(
                            &vf,
                            yuv,
                            width,
                            height,
                            frame_bytes_yuv,
                            display_idx,
                        );
                    }
                    display_idx += 1;
                }
                println!("  decode #{i:>2}: ok ({} B)", frame_bytes.len());
            }
            Err(e) => {
                println!("  decode #{i:>2}: ERROR {e}");
                break;
            }
        }
    }
}

/// Flatten one decoded frame into yuv420p bytes, then report how far
/// it is from the matching frame in the reference YUV. `width`/`height`
/// here are the expected picture dimensions — we trust them over the
/// plane `stride` because some planes pad rows for alignment.
fn compare_frame_against_ref(
    vf: &oxideav_core::VideoFrame,
    ref_yuv: &[u8],
    width: usize,
    height: usize,
    frame_bytes_yuv: usize,
    display_idx: usize,
) {
    let mut ours = Vec::with_capacity(frame_bytes_yuv);
    for (pi, p) in vf.planes.iter().enumerate() {
        let w = if pi == 0 { width } else { width / 2 };
        let h = if pi == 0 { height } else { height / 2 };
        for row in 0..h {
            let off = row * p.stride;
            ours.extend_from_slice(&p.data[off..off + w]);
        }
    }
    if display_idx * frame_bytes_yuv + frame_bytes_yuv > ref_yuv.len() {
        println!(
            "    display #{display_idx}: ref YUV too short for comparison ({} bytes)",
            ref_yuv.len()
        );
        return;
    }
    let their =
        &ref_yuv[display_idx * frame_bytes_yuv..display_idx * frame_bytes_yuv + frame_bytes_yuv];
    let n = ours.len().min(their.len());
    let diff = (0..n).filter(|&b| ours[b] != their[b]).count();
    let max_abs = (0..n)
        .map(|b| (ours[b] as i32 - their[b] as i32).unsigned_abs())
        .max()
        .unwrap_or(0);
    println!("    display #{display_idx:>2}: diff {diff}/{n} bytes  max|Δ|={max_abs}");
    // Per-MB luma diff summary, sorted/limited.
    if std::env::var("OXIDEAV_H264_DIFF_MB").is_ok() {
        let luma_size = width * height;
        let chroma_size = (width / 2) * (height / 2);
        let mb_w = width.div_ceil(16);
        let mb_h = height.div_ceil(16);
        let mut mb_diffs = Vec::new();
        for my in 0..mb_h {
            for mx in 0..mb_w {
                let mut dl = 0usize;
                let mut maxal = 0u32;
                for r in 0..16 {
                    let y = my * 16 + r;
                    if y >= height {
                        continue;
                    }
                    for c in 0..16 {
                        let x = mx * 16 + c;
                        if x >= width {
                            continue;
                        }
                        let i = y * width + x;
                        if i < luma_size && ours[i] != their[i] {
                            dl += 1;
                            let a = (ours[i] as i32 - their[i] as i32).unsigned_abs();
                            if a > maxal {
                                maxal = a;
                            }
                        }
                    }
                }
                let mut dc = 0usize;
                let mut maxac = 0u32;
                for r in 0..8 {
                    let y = my * 8 + r;
                    if y >= height / 2 {
                        continue;
                    }
                    for c in 0..8 {
                        let x = mx * 8 + c;
                        if x >= width / 2 {
                            continue;
                        }
                        let cb_off = luma_size + y * (width / 2) + x;
                        let cr_off = luma_size + chroma_size + y * (width / 2) + x;
                        if ours[cb_off] != their[cb_off] {
                            dc += 1;
                            let a = (ours[cb_off] as i32 - their[cb_off] as i32).unsigned_abs();
                            if a > maxac {
                                maxac = a;
                            }
                        }
                        if ours[cr_off] != their[cr_off] {
                            dc += 1;
                            let a = (ours[cr_off] as i32 - their[cr_off] as i32).unsigned_abs();
                            if a > maxac {
                                maxac = a;
                            }
                        }
                    }
                }
                if dl > 0 || dc > 0 {
                    mb_diffs.push((mx, my, dl, maxal, dc, maxac));
                }
            }
        }
        // First by raster order to find the first divergence point.
        for (mx, my, dl, maxal, dc, maxac) in mb_diffs.iter().take(20) {
            println!("      raster mb ({mx:>2},{my:>2}): luma {dl:>3} max|Δ|={maxal} | chroma {dc:>3} max|Δ|={maxac}");
        }
        println!("      total: {} MBs differ", mb_diffs.len());
    }
}
