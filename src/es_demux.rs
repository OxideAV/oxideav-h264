//! Raw H.264 elementary-stream ("Annex B byte stream") container.
//!
//! A `.h264` / `.264` / `.avc` file is just the §B.1 byte stream:
//! NAL units behind `00 00 01` / `00 00 00 01` start codes, parameter
//! sets in band, no timestamps. This demuxer
//!
//! * probes for a byte stream that opens with a start code and carries
//!   a parseable sequence parameter set early on;
//! * declares one `h264` video stream whose dimensions (after the SPS
//!   frame-cropping window), pixel layout and — when the VUI carries
//!   `timing_info` — frame rate come from the first SPS (no extradata:
//!   the decoder reads the in-band parameter sets);
//! * emits one packet per access unit (§7.4.1.2.3), Annex-B framed,
//!   numbered in decode order (`dts` = access-unit index; `pts` is left
//!   unset because display order is only known after decoding).
//!
//! The whole input is read at open: raw elementary streams carry no
//! index to seek through, and access-unit boundaries need a look-ahead
//! of one NAL anyway.

use std::io::Read;

use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, ContainerRegistry, Demuxer, Error, Packet,
    PixelFormat, ProbeData, ProbeScore, Rational, ReadSeek, Result, StreamInfo, TimeBase,
};

use crate::nal::{parse_nal_unit, AnnexBSplitter};
use crate::sps::Sps;

/// Container name the demuxer registers under.
pub const CONTAINER_NAME: &str = "h264";

/// File extensions that name a raw H.264 byte stream.
const EXTENSIONS: &[&str] = &["h264", "264", "avc", "26l", "jsv"];

/// Frame rate assumed when the SPS has no VUI `timing_info`.
const DEFAULT_FPS: i64 = 25;

/// Register the raw byte-stream demuxer, its probe and extensions.
pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer(CONTAINER_NAME, open_demuxer);
    reg.register_probe(CONTAINER_NAME, probe);
    for ext in EXTENSIONS {
        reg.register_extension(ext, CONTAINER_NAME);
    }
}

const NAL_SLICE: u8 = 1;
const NAL_IDR: u8 = 5;
const NAL_SPS: u8 = 7;

fn nal_type(nal: &[u8]) -> u8 {
    nal.first().map_or(0, |b| b & 0x1f)
}

/// The first SPS among `nals` that parses.
fn first_sps<'a>(nals: impl Iterator<Item = &'a [u8]>) -> Option<Sps> {
    nals.filter(|n| nal_type(n) == NAL_SPS)
        .find_map(|n| Sps::parse(&parse_nal_unit(n).ok()?.rbsp).ok())
}

/// §B.1 byte stream: a start code at the very beginning (after any
/// `leading_zero_8bits`), `forbidden_zero_bit` clear on the first NAL
/// and a parseable SPS within the probe window. MPEG-1/2 video and
/// program streams (`00 00 01 B3` / `BA`) fail the forbidden bit; raw
/// HEVC fails the SPS parse.
fn probe(p: &ProbeData) -> ProbeScore {
    let lead = p.buf.iter().take_while(|&&b| b == 0).count();
    if lead < 2 || p.buf.get(lead) != Some(&1) {
        return 0;
    }
    match p.buf.get(lead + 1) {
        Some(h) if h & 0x80 == 0 && matches!(h & 0x1f, 1 | 5..=9) => {}
        _ => return 0,
    }
    if first_sps(AnnexBSplitter::new(p.buf)).is_none() {
        return 0;
    }
    if p.ext.is_some_and(|e| EXTENSIONS.contains(&e)) {
        75
    } else {
        50
    }
}

fn open_demuxer(
    mut input: Box<dyn ReadSeek>,
    _codecs: &dyn CodecResolver,
) -> Result<Box<dyn Demuxer>> {
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;
    Ok(Box::new(H264EsDemuxer::new(&data)?))
}

/// The demuxer: every access unit is pre-split at open.
pub struct H264EsDemuxer {
    streams: Vec<StreamInfo>,
    units: std::vec::IntoIter<Vec<u8>>,
    next_dts: i64,
}

impl H264EsDemuxer {
    /// Split `data` (a whole byte stream) into access units.
    pub fn new(data: &[u8]) -> Result<Self> {
        let sps = first_sps(AnnexBSplitter::new(data))
            .ok_or_else(|| Error::invalid("h264 byte stream: no sequence parameter set"))?;
        let units = split_access_units(data);
        let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        let (w, h) = display_dims(&sps);
        params.width = Some(w);
        params.height = Some(h);
        params.pixel_format = pixel_format(&sps);
        let rate = frame_rate(&sps);
        params.frame_rate = Some(rate);
        let time_base = TimeBase::new(rate.den, rate.num);
        Ok(Self {
            streams: vec![StreamInfo {
                index: 0,
                time_base,
                duration: Some(units.len() as i64),
                start_time: Some(0),
                params,
            }],
            units: units.into_iter(),
            next_dts: 0,
        })
    }
}

impl Demuxer for H264EsDemuxer {
    fn format_name(&self) -> &str {
        CONTAINER_NAME
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        let data = self.units.next().ok_or(Error::Eof)?;
        let mut pkt = Packet::new(0, self.streams[0].time_base, data);
        pkt.dts = Some(self.next_dts);
        pkt.pts = None;
        pkt.duration = Some(1);
        pkt.flags.keyframe = AnnexBSplitter::new(&pkt.data).any(|n| nal_type(n) == NAL_IDR);
        self.next_dts += 1;
        Ok(pkt)
    }
}

/// §7.4.1.2.3 access-unit boundaries: an access unit ends before an
/// access unit delimiter, SPS, PPS, SEI, NAL types 14..=18, or the
/// first VCL NAL of a new primary coded picture (`first_mb_in_slice`
/// = 0, i.e. the slice header's leading `ue(v)` is the single bit `1`)
/// — whichever follows a VCL NAL of the current unit. Each access unit
/// is re-emitted with 4-byte start codes.
fn split_access_units(data: &[u8]) -> Vec<Vec<u8>> {
    let mut units = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    let mut has_vcl = false;
    for nal in AnnexBSplitter::new(data) {
        let ty = nal_type(nal);
        let vcl = matches!(ty, NAL_SLICE..=NAL_IDR);
        let starts_unit = if vcl {
            nal.get(1).is_some_and(|b| b & 0x80 != 0)
        } else {
            matches!(ty, 6..=9 | 14..=18)
        };
        if has_vcl && starts_unit {
            units.push(std::mem::take(&mut cur));
            has_vcl = false;
        }
        cur.extend_from_slice(&[0, 0, 0, 1]);
        cur.extend_from_slice(nal);
        has_vcl |= vcl;
    }
    if has_vcl {
        units.push(cur);
    }
    units
}

/// §7.4.2.1.1: the frame size after the cropping window (eqs. 7-19 ..
/// 7-22).
fn display_dims(sps: &Sps) -> (u32, u32) {
    let w = sps.pic_width_in_mbs() * 16;
    let h = sps.frame_height_in_mbs() * 16;
    let Some(c) = &sps.frame_cropping else {
        return (w, h);
    };
    let cat = sps.chroma_array_type();
    let (sub_w, sub_h) = match cat {
        1 => (2, 2),
        2 => (2, 1),
        _ => (1, 1),
    };
    let crop_x = if cat == 0 { 1 } else { sub_w };
    let frame_mbs = if sps.frame_mbs_only_flag { 1 } else { 2 };
    let crop_y = frame_mbs * if cat == 0 { 1 } else { sub_h };
    (
        w.saturating_sub(crop_x * (c.left + c.right)),
        h.saturating_sub(crop_y * (c.top + c.bottom)),
    )
}

/// The planar layout the decoder emits for this SPS.
fn pixel_format(sps: &Sps) -> Option<PixelFormat> {
    use PixelFormat::*;
    let luma = sps.bit_depth_luma_minus8 + 8;
    if sps.chroma_array_type() != 0 && sps.bit_depth_chroma_minus8 + 8 != luma {
        return None;
    }
    Some(match (sps.chroma_array_type(), luma) {
        (0, 8) => Gray8,
        (0, 10) => Gray10Le,
        (0, 12) => Gray12Le,
        (1, 8) => Yuv420P,
        (2, 8) => Yuv422P,
        (3, 8) => Yuv444P,
        (1, 10) => Yuv420P10Le,
        (2, 10) => Yuv422P10Le,
        (3, 10) => Yuv444P10Le,
        (1, 12) => Yuv420P12Le,
        (2, 12) => Yuv422P12Le,
        (3, 12) => Yuv444P12Le,
        _ => return None,
    })
}

/// §E.2.1: a frame lasts `2 * num_units_in_tick / time_scale` seconds.
fn frame_rate(sps: &Sps) -> Rational {
    sps.vui
        .as_ref()
        .and_then(|v| v.timing_info.as_ref())
        .filter(|t| t.num_units_in_tick > 0 && t.time_scale > 0)
        .map(|t| {
            let (num, den) = reduce(t.time_scale as i64, 2 * t.num_units_in_tick as i64);
            Rational::new(num, den)
        })
        .unwrap_or(Rational::new(DEFAULT_FPS, 1))
}

fn reduce(a: i64, b: i64) -> (i64, i64) {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    (a / x, b / x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn access_units_split_on_parameter_sets_and_new_pictures() {
        let sps = [0x67, 0x42];
        let pps = [0x68, 0xce];
        let idr = [0x65, 0x88]; // first_mb_in_slice = 0
        let idr2 = [0x65, 0x48]; // first_mb_in_slice != 0 (same picture)
        let p = [0x41, 0x9a]; // new picture
        let mut s = Vec::new();
        for n in [&sps[..], &pps, &idr, &idr2, &p] {
            s.extend_from_slice(&[0, 0, 1]);
            s.extend_from_slice(n);
        }
        let units = split_access_units(&s);
        assert_eq!(units.len(), 2);
        assert_eq!(
            units[0],
            [
                0, 0, 0, 1, 0x67, 0x42, 0, 0, 0, 1, 0x68, 0xce, 0, 0, 0, 1, 0x65, 0x88, 0, 0, 0, 1,
                0x65, 0x48
            ]
        );
        assert_eq!(units[1], [0, 0, 0, 1, 0x41, 0x9a]);
    }

    #[test]
    fn probe_rejects_other_start_code_formats() {
        let mpeg2 = [0, 0, 1, 0xb3, 0x10, 0x00];
        assert_eq!(
            probe(&ProbeData {
                buf: &mpeg2,
                ext: None
            }),
            0
        );
        let hevc = [0, 0, 0, 1, 0x40, 0x01, 0x0c];
        assert_eq!(
            probe(&ProbeData {
                buf: &hevc,
                ext: Some("h264")
            }),
            0
        );
        assert_eq!(
            probe(&ProbeData {
                buf: b"RIFF",
                ext: Some("h264")
            }),
            0
        );
    }

    #[test]
    fn rate_is_half_the_tick_rate() {
        assert_eq!(reduce(50, 2), (25, 1));
        assert_eq!(reduce(60000, 2002), (30000, 1001));
    }
}
