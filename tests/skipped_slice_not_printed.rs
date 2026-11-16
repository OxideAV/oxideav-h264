//! A slice the decoder skips is counted in `decode_error_count()` and nothing
//! is printed: a library leaves reporting to its caller.
//!
//! The test runs itself again as a child process (selected by an environment
//! variable) so it can read what the decoder wrote to the real standard error.
//! The stream is hand-built: a Main profile sequence and picture parameter set
//! and an IDR slice whose `first_mb_in_slice` is not 0, which §7.4.3 and Annex A
//! forbid for the first slice of a picture outside Baseline and Extended
//! profiles, so the decoder refuses the slice.

use std::process::Command;

use oxideav_core::{CodecId, Decoder, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;

const CHILD: &str = "OXIDEAV_H264_SKIPPED_SLICE_CHILD";

#[derive(Default)]
struct Bits {
    bytes: Vec<u8>,
    used: u32,
}

impl Bits {
    fn bit(&mut self, b: u32) {
        if self.used == 0 {
            self.bytes.push(0);
        }
        if b != 0 {
            *self.bytes.last_mut().unwrap() |= 0x80 >> self.used;
        }
        self.used = (self.used + 1) % 8;
    }
    fn u(&mut self, n: u32, v: u32) {
        for i in (0..n).rev() {
            self.bit((v >> i) & 1);
        }
    }
    fn ue(&mut self, v: u32) {
        let x = v + 1;
        let len = 32 - x.leading_zeros();
        self.u(len - 1, 0);
        self.u(len, x);
    }
    fn se(&mut self, v: i32) {
        self.ue(if v > 0 {
            2 * v as u32 - 1
        } else {
            (-2 * v) as u32
        });
    }
    fn trailing(&mut self) {
        self.bit(1);
        while self.used != 0 {
            self.bit(0);
        }
    }
}

fn nal(ref_idc: u8, unit_type: u8, rbsp: &[u8]) -> Vec<u8> {
    let mut out = vec![0, 0, 0, 1, (ref_idc << 5) | unit_type];
    let mut zeros = 0;
    for &b in rbsp {
        if zeros >= 2 && b <= 3 {
            out.push(3);
            zeros = 0;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
    }
    out
}

fn stream() -> Vec<u8> {
    let mut sps = Bits::default();
    sps.u(8, 77); // Main
    sps.u(8, 0);
    sps.u(8, 30);
    sps.ue(0); // seq_parameter_set_id
    sps.ue(0); // log2_max_frame_num_minus4
    sps.ue(2); // pic_order_cnt_type
    sps.ue(1); // max_num_ref_frames
    sps.bit(0);
    sps.ue(1); // pic_width_in_mbs_minus1: two macroblocks wide
    sps.ue(0); // pic_height_in_map_units_minus1
    sps.bit(1); // frame_mbs_only_flag
    sps.bit(1); // direct_8x8_inference_flag
    sps.bit(0); // frame_cropping_flag
    sps.bit(0); // vui_parameters_present_flag
    sps.trailing();

    let mut pps = Bits::default();
    pps.ue(0);
    pps.ue(0);
    pps.bit(0); // CAVLC
    pps.bit(0);
    pps.ue(0);
    pps.ue(0);
    pps.ue(0);
    pps.bit(0);
    pps.u(2, 0);
    pps.se(0);
    pps.se(0);
    pps.se(0);
    pps.bit(0);
    pps.bit(0);
    pps.bit(0);
    pps.trailing();

    let mut slice = Bits::default();
    slice.ue(1); // first_mb_in_slice: not 0
    slice.ue(7); // I
    slice.ue(0);
    slice.u(4, 0); // frame_num
    slice.ue(0); // idr_pic_id
    slice.bit(0);
    slice.bit(0);
    slice.se(0);
    slice.ue(25); // I_PCM, never reached
    slice.trailing();

    let mut out = nal(3, 7, &sps.bytes);
    out.extend(nal(3, 8, &pps.bytes));
    out.extend(nal(3, 5, &slice.bytes));
    out
}

#[test]
fn a_skipped_slice_is_counted_and_not_printed() {
    if std::env::var_os(CHILD).is_some() {
        let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
        let pkt = Packet::new(0, TimeBase::new(1, 25), stream()).with_pts(0);
        let _ = dec.send_packet(&pkt);
        let _ = dec.flush();
        println!("SKIPPED={}", dec.decode_error_count());
        return;
    }
    let out = Command::new(std::env::current_exe().unwrap())
        .args([
            "a_skipped_slice_is_counted_and_not_printed",
            "--exact",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .output()
        .expect("run the child process");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "child failed: {stdout}{stderr}");
    let skipped: u64 = stdout
        .lines()
        .find_map(|l| l.strip_prefix("SKIPPED="))
        .expect("the child reports its count")
        .parse()
        .unwrap();
    assert!(skipped >= 1, "the refused slice is counted");
    assert!(
        !stderr.contains("skipped"),
        "nothing is printed for a skipped slice, got: {stderr}"
    );
}
