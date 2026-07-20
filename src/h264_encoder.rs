//! H.264 encoder exposed through the `oxideav_core::Encoder` trait so
//! pipelines and muxers can drive it from [`CodecParameters`].
//!
//! Wraps [`crate::encoder::session::EncoderSession`]: IDR + linear P
//! GOPs, 4:2:0, CAVLC (Baseline signalling) or CABAC (Main), Annex B
//! output with in-band SPS/PPS on every IDR (`extradata` stays empty).
//! Three QP policies, selected via the option bag:
//!
//! * `rc=cqp` — fixed QP (`qp` option), the historical behaviour;
//! * `rc=cbr` — constant bit rate at `bitrate` bps with §7.3.2.7
//!   filler keeping the Annex C channel busy;
//! * `rc=vbr` — capped VBR: long-term average `bitrate`, peak
//!   `max_bitrate`;
//! * `rc=auto` (default) — `cbr` when a bitrate is known (option or
//!   `CodecParameters::bit_rate`), else `cqp`.
//!
//! One packet out per frame in; no lookahead, so `flush` has nothing
//! to drain and `pts == dts`.

use std::collections::VecDeque;

use oxideav_core::options::{
    parse_options, CodecOptionsStruct, OptionField, OptionKind, OptionValue,
};
use oxideav_core::{
    CodecId, CodecParameters, Encoder as CoreEncoder, Error, Frame, Packet, Result, TimeBase,
};

use crate::encoder::rate_control::{RateControlConfig, RateControlMode};
use crate::encoder::session::{EncoderSession, SessionConfig, SessionRateControl};

/// Typed option struct for the registry encoder. See the module docs
/// for the semantics of each knob.
#[derive(Debug, Clone)]
pub struct H264EncoderOptions {
    /// Fixed QP for `rc=cqp` (0..=51). Default 26.
    pub qp: u32,
    /// Rate-control mode: `auto` / `cqp` / `cbr` / `vbr`.
    pub rc: String,
    /// Target bitrate in bits per second (0 = take
    /// `CodecParameters::bit_rate`, else fall back to `cqp`).
    pub bitrate: u32,
    /// Peak bitrate for `rc=vbr` in bits per second (0 = 2x target).
    pub max_bitrate: u32,
    /// CPB/VBV buffer size in bits (0 = one second at the channel
    /// rate).
    pub buffer_size: u32,
    /// IDR period in frames (>= 1). Default 30.
    pub gop: u32,
    /// CABAC entropy coding (Main profile signalling). Default false
    /// (CAVLC / Baseline).
    pub cabac: bool,
}

impl Default for H264EncoderOptions {
    fn default() -> Self {
        Self {
            qp: 26,
            rc: "auto".to_string(),
            bitrate: 0,
            max_bitrate: 0,
            buffer_size: 0,
            gop: 30,
            cabac: false,
        }
    }
}

impl CodecOptionsStruct for H264EncoderOptions {
    const SCHEMA: &'static [OptionField] = &[
        OptionField {
            name: "qp",
            kind: OptionKind::U32,
            default: OptionValue::U32(26),
            help: "fixed QP for rc=cqp (0..=51)",
        },
        OptionField {
            name: "rc",
            kind: OptionKind::Enum(&["auto", "cqp", "cbr", "vbr"]),
            default: OptionValue::String(String::new()),
            help: "rate-control mode (auto: cbr when a bitrate is known, else cqp)",
        },
        OptionField {
            name: "bitrate",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "target bitrate in bits/s (0: use CodecParameters::bit_rate)",
        },
        OptionField {
            name: "max_bitrate",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "peak bitrate in bits/s for rc=vbr (0: 2x target)",
        },
        OptionField {
            name: "buffer_size",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "CPB/VBV size in bits (0: one second at the channel rate)",
        },
        OptionField {
            name: "gop",
            kind: OptionKind::U32,
            default: OptionValue::U32(30),
            help: "IDR period in frames (>= 1)",
        },
        OptionField {
            name: "cabac",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(false),
            help: "CABAC entropy coding (Main profile); default CAVLC (Baseline)",
        },
    ];

    fn apply(&mut self, key: &str, value: &OptionValue) -> Result<()> {
        match key {
            "qp" => self.qp = value.as_u32()?,
            "rc" => self.rc = value.as_str()?.to_string(),
            "bitrate" => self.bitrate = value.as_u32()?,
            "max_bitrate" => self.max_bitrate = value.as_u32()?,
            "buffer_size" => self.buffer_size = value.as_u32()?,
            "gop" => self.gop = value.as_u32()?,
            "cabac" => self.cabac = value.as_bool()?,
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

/// `oxideav_core::Encoder` implementation. Construct via
/// [`make_encoder`] (the registry factory) or [`H264CodecEncoder::new`].
pub struct H264CodecEncoder {
    codec_id: CodecId,
    output_params: CodecParameters,
    session: EncoderSession,
    time_base: TimeBase,
    width: usize,
    height: usize,
    frame_index: i64,
    queue: VecDeque<Packet>,
}

impl H264CodecEncoder {
    /// Build an encoder from parameters + option bag. Errors on
    /// missing/invalid geometry, unknown options, or inconsistent
    /// rate-control settings.
    pub fn new(params: &CodecParameters) -> Result<Self> {
        let opts: H264EncoderOptions = parse_options(&params.options)?;

        let width = params
            .width
            .ok_or_else(|| Error::invalid("h264 encoder requires width"))?;
        let height = params
            .height
            .ok_or_else(|| Error::invalid("h264 encoder requires height"))?;
        if width == 0 || height == 0 || width % 16 != 0 || height % 16 != 0 {
            return Err(Error::invalid(format!(
                "h264 encoder requires 16-aligned non-zero dimensions, got {width}x{height}"
            )));
        }
        if let Some(pf) = params.pixel_format {
            if pf != oxideav_core::PixelFormat::Yuv420P {
                return Err(Error::unsupported(format!(
                    "h264 encoder input must be Yuv420P, got {pf:?}"
                )));
            }
        }
        if opts.gop == 0 {
            return Err(Error::invalid("gop must be >= 1"));
        }
        if opts.qp > 51 {
            return Err(Error::invalid(format!("qp {} out of 0..=51", opts.qp)));
        }

        // Frame rate: needed to convert bits/s into per-frame budgets.
        let (fps_num, fps_den) = match params.frame_rate {
            Some(r) if r.num > 0 && r.den > 0 => (r.num as u32, r.den as u32),
            _ => (30, 1),
        };

        let target_bitrate = if opts.bitrate > 0 {
            opts.bitrate
        } else {
            params.bit_rate.unwrap_or(0).min(u64::from(u32::MAX)) as u32
        };

        let mode = match opts.rc.as_str() {
            "" | "auto" => {
                if target_bitrate > 0 {
                    Some(RateControlMode::Cbr)
                } else {
                    None
                }
            }
            "cqp" => None,
            "cbr" => Some(RateControlMode::Cbr),
            "vbr" => Some(RateControlMode::CappedVbr),
            other => {
                return Err(Error::invalid(format!("unknown rc mode '{other}'")));
            }
        };

        let rate_control = match mode {
            None => SessionRateControl::ConstantQp(opts.qp as i32),
            Some(m) => {
                if target_bitrate == 0 {
                    return Err(Error::invalid(
                        "rc=cbr/vbr requires a bitrate (option or CodecParameters::bit_rate)",
                    ));
                }
                let max_bitrate = match m {
                    RateControlMode::Cbr => target_bitrate,
                    RateControlMode::CappedVbr => {
                        if opts.max_bitrate > 0 {
                            if opts.max_bitrate < target_bitrate {
                                return Err(Error::invalid(
                                    "max_bitrate must be >= bitrate for rc=vbr",
                                ));
                            }
                            opts.max_bitrate
                        } else {
                            target_bitrate.saturating_mul(2)
                        }
                    }
                };
                let vbv_buffer_bits = if opts.buffer_size > 0 {
                    opts.buffer_size
                } else {
                    max_bitrate
                };
                SessionRateControl::Controlled(RateControlConfig {
                    mode: m,
                    target_bitrate,
                    max_bitrate,
                    vbv_buffer_bits,
                    fps_num,
                    fps_den,
                    min_qp: 10,
                    max_qp: 51,
                })
            }
        };

        let session = EncoderSession::new(SessionConfig {
            width,
            height,
            gop_length: opts.gop,
            cabac: opts.cabac,
            rate_control,
        });

        let mut output_params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        output_params.width = Some(width);
        output_params.height = Some(height);
        output_params.pixel_format = Some(oxideav_core::PixelFormat::Yuv420P);
        output_params.frame_rate = params.frame_rate;
        if target_bitrate > 0 && mode.is_some() {
            output_params.bit_rate = Some(u64::from(target_bitrate));
        }
        // Annex B with in-band SPS/PPS — no extradata.

        Ok(Self {
            codec_id: CodecId::new(crate::CODEC_ID_STR),
            output_params,
            session,
            time_base: TimeBase::new(i64::from(fps_den), i64::from(fps_num)),
            width: width as usize,
            height: height as usize,
            frame_index: 0,
            queue: VecDeque::new(),
        })
    }

    /// Copy a possibly padded plane into a tightly packed buffer.
    fn packed_plane(plane: &oxideav_core::VideoPlane, w: usize, h: usize) -> Result<Vec<u8>> {
        if plane.stride == w && plane.data.len() == w * h {
            return Ok(plane.data.clone());
        }
        if plane.stride < w || plane.data.len() < plane.stride * h {
            return Err(Error::invalid(format!(
                "plane too small: stride {} data {} for {w}x{h}",
                plane.stride,
                plane.data.len()
            )));
        }
        let mut out = Vec::with_capacity(w * h);
        for row in 0..h {
            let s = row * plane.stride;
            out.extend_from_slice(&plane.data[s..s + w]);
        }
        Ok(out)
    }
}

impl CoreEncoder for H264CodecEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let vf = match frame {
            Frame::Video(vf) => vf,
            _ => return Err(Error::invalid("h264 encoder accepts video frames only")),
        };
        let planes = vf.image_planes();
        if planes.len() != 3 {
            return Err(Error::invalid(format!(
                "h264 encoder expects 3 planes (Yuv420P), got {}",
                planes.len()
            )));
        }
        let (w, h) = (self.width, self.height);
        let y = Self::packed_plane(&planes[0], w, h)?;
        let u = Self::packed_plane(&planes[1], w / 2, h / 2)?;
        let v = Self::packed_plane(&planes[2], w / 2, h / 2)?;

        let sf = self.session.encode_frame(&y, &u, &v);

        let pts = vf.pts.unwrap_or(self.frame_index);
        let mut packet = Packet::new(0, self.time_base, sf.annex_b)
            .with_pts(pts)
            .with_dts(pts);
        packet.duration = Some(1);
        packet.flags.keyframe = sf.is_idr;
        self.queue.push_back(packet);
        self.frame_index += 1;
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.queue.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        // No lookahead: every send_frame already produced its packet.
        Ok(())
    }
}

/// Registry factory (`CodecInfo::encoder`).
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn CoreEncoder>> {
    Ok(Box::new(H264CodecEncoder::new(params)?))
}
