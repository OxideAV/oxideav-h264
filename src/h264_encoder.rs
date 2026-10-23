//! H.264 encoder exposed through the `oxideav_core::Encoder` trait so
//! pipelines and muxers can drive it from [`CodecParameters`].
//!
//! Two coding paths sit behind one factory, chosen by the input pixel
//! format and the `profile` option:
//!
//! * **8-bit 4:2:0** (`Yuv420P`, and `Gray8` coded as 4:2:0 with
//!   constant chroma) — [`crate::encoder::session::EncoderSession`]:
//!   IDR + P (+ B) GOPs, CAVLC (Baseline, 66) or CABAC (Main, 77),
//!   three QP policies, MB-row rate control. Unchanged from round 420
//!   (streams are byte-identical to earlier releases for the same
//!   options).
//! * **High-family layouts** (round 475) — 8-bit 4:2:2 / 4:4:4, and
//!   10- / 12-bit 4:2:0 / 4:2:2 / 4:4:4 (`Yuv422P`, `Yuv444P`,
//!   `Yuv420P10Le` … `Yuv444P12Le`, `Gray10Le` / `Gray12Le` as 4:2:0
//!   with constant chroma) — [`crate::encoder::deep::DeepSession`]
//!   under High (100) / High 10 (110) / High 4:2:2 (122) / High 4:4:4
//!   Predictive (244) per §A.2.4 – §A.2.7: IDR + P GOPs (Intra_16x16 /
//!   P_Skip / P_L0_16x16 with Intra_16x16 fallback), CAVLC or CABAC,
//!   fixed QP (`rc=cqp` only), optional 8x8 transform, deblocking
//!   on/off, and **lossless** coding (`lossless=true`: SPS
//!   `qpprime_y_zero_transform_bypass_flag`, QP′Y = 0, profile 244).
//!   No B pictures and no rate control on this path.
//!
//! Every IDR access unit carries in-band SPS + PPS and a §7.4.3
//! `idr_pic_id` that advances per IDR (0, 1, 2, … mod 65536), so
//! consecutive IDR access units — an all-intra `gop=1` sequence —
//! never share the value. With `format=avcc` the packets are instead
//! 4-byte length-prefixed samples without parameter sets, and
//! `output_params().extradata` holds the ISO/IEC 14496-15
//! `AVCDecoderConfigurationRecord` ([`crate::avcc`]) built from the
//! session's SPS / PPS — the `avc1` sample-entry / HEIF `avcC` shape.
//!
//! QP policies (8-bit 4:2:0 path), selected via the option bag:
//!
//! * `rc=cqp` — fixed QP (`qp` option), the historical behaviour;
//! * `rc=cbr` — constant bit rate at `bitrate` bps with §7.3.2.7
//!   filler keeping the Annex C channel busy;
//! * `rc=vbr` — capped VBR: long-term average `bitrate`, peak
//!   `max_bitrate`;
//! * `rc=auto` (default) — `cbr` when a bitrate is known (option or
//!   `CodecParameters::bit_rate`), else `cqp`. On the High-family path
//!   `auto` always means `cqp`.
//!
//! With `bframes=0` (default) there is no lookahead: one packet out
//! per frame in, `flush` has nothing to drain and `pts == dts`. With
//! `bframes=N` the session buffers up to N pictures until the next
//! anchor, then emits the mini-GOP in decode order (anchor first) —
//! packets carry `dts` = decode counter and `pts` = display index + 1
//! (both in frame-duration units; the +1 is the one-frame reorder
//! latency, keeping `pts >= dts` on every packet), and `flush` drains
//! the buffered tail.

use std::collections::VecDeque;

use oxideav_core::options::{
    CodecOptionsStruct, OptionField, OptionKind, OptionValue, parse_options,
};
use oxideav_core::{
    CodecId, CodecParameters, Encoder as CoreEncoder, Error, Frame, Packet, PixelFormat, Result,
    TimeBase, VideoPlane,
};

use crate::avcc::{AvcDecoderConfigurationRecord, annex_b_to_length_prefixed};
use crate::encoder::deep::{
    DeepConfig, DeepPictureKind, DeepSession, min_profile_idc, profile_admits,
};
use crate::encoder::rate_control::{RateControlConfig, RateControlMode};
use crate::encoder::session::{EncoderSession, SessionConfig, SessionRateControl};
use crate::transform::qp_bd_offset;

/// Typed option struct for the registry encoder. See the module docs
/// for the semantics of each knob.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct H264EncoderOptions {
    /// Fixed QP for `rc=cqp` (0..=51). Default 26. On the High-family
    /// path this is the slice QP_Y (the encoder runs at QP′ = QP +
    /// QpBdOffset internally).
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
    /// CABAC entropy coding (Main profile signalling on the 8-bit
    /// 4:2:0 path). Default false (CAVLC / Baseline).
    pub cabac: bool,
    /// Consecutive non-reference B pictures between anchors (0..=16).
    /// Any non-zero value promotes CAVLC signalling to Main profile
    /// and introduces one frame interval of reorder latency
    /// (`pts != dts`). Default 0. 8-bit 4:2:0 path only.
    pub bframes: u32,
    /// Round-475 — `auto` (default: Baseline / Main for 8-bit 4:2:0
    /// by `cabac` / `bframes`, else the §A.2 minimum High-family
    /// profile for the layout), or an explicit `baseline` / `main` /
    /// `high` / `high10` / `high422` / `high444`. An explicit
    /// High-family profile routes the picture through the
    /// High-family encoder even for 8-bit 4:2:0 input; a profile that
    /// does not admit the layout (§A.2.4 – §A.2.7) is refused.
    pub profile: String,
    /// Round-475 — lossless coding (High 4:4:4 Predictive transform
    /// bypass at QP′Y = 0); the reconstruction equals the source
    /// sample-for-sample. High-family path; `qp` is ignored.
    pub lossless: bool,
    /// Round-475 — High-profile 8x8 transform on P macroblocks
    /// (4:2:0 / 4:2:2). High-family path.
    pub transform_8x8: bool,
    /// Round-475 — §8.7 in-loop deblocking (default true). High-family
    /// path (the 8-bit 4:2:0 path always deblocks).
    pub deblock: bool,
    /// Round-475 — Intra_4x4 (I_NxN) macroblocks in the I pictures of
    /// the High-family path, competing with Intra_16x16 per macroblock
    /// (default true). The 8-bit 4:2:0 path runs its own RDO.
    pub intra_4x4: bool,
    /// Round-475 — packet framing: `annexb` (default: start codes,
    /// in-band SPS / PPS on every IDR, empty extradata) or `avcc`
    /// (4-byte length-prefixed NAL units without parameter sets /
    /// filler; `extradata` = `AVCDecoderConfigurationRecord`).
    pub format: String,
}

impl H264EncoderOptions {
    /// The defaults (see each field).
    pub fn new() -> Self {
        Self::default()
    }
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
            bframes: 0,
            profile: "auto".to_string(),
            lossless: false,
            transform_8x8: false,
            deblock: true,
            intra_4x4: true,
            format: "annexb".to_string(),
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
            help: "rate-control mode (auto: cbr when a bitrate is known, else cqp; High-family layouts: cqp only)",
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
        OptionField {
            name: "bframes",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "non-reference B pictures between anchors (0..=16; >0 implies Main profile; 8-bit 4:2:0 only)",
        },
        OptionField {
            name: "profile",
            kind: OptionKind::Enum(&[
                "auto", "baseline", "main", "high", "high10", "high422", "high444",
            ]),
            default: OptionValue::String(String::new()),
            help: "profile_idc: auto (66/77 for 8-bit 4:2:0, else the §A.2 minimum High-family profile) or explicit",
        },
        OptionField {
            name: "lossless",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(false),
            help: "lossless coding (High 4:4:4 Predictive transform bypass, QP'Y = 0)",
        },
        OptionField {
            name: "transform_8x8",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(false),
            help: "High-profile 8x8 transform on P macroblocks (High-family path, 4:2:0 / 4:2:2)",
        },
        OptionField {
            name: "deblock",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(true),
            help: "in-loop deblocking filter (High-family path; the 8-bit 4:2:0 path always deblocks)",
        },
        OptionField {
            name: "intra_4x4",
            kind: OptionKind::Bool,
            default: OptionValue::Bool(true),
            help: "Intra_4x4 (I_NxN) macroblocks in I pictures (High-family path)",
        },
        OptionField {
            name: "format",
            kind: OptionKind::Enum(&["annexb", "avcc"]),
            default: OptionValue::String(String::new()),
            help: "packet framing: annexb (start codes, in-band SPS/PPS) or avcc (4-byte length prefixes, extradata = AVCDecoderConfigurationRecord)",
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
            "bframes" => self.bframes = value.as_u32()?,
            "profile" => self.profile = value.as_str()?.to_string(),
            "lossless" => self.lossless = value.as_bool()?,
            "transform_8x8" => self.transform_8x8 = value.as_bool()?,
            "deblock" => self.deblock = value.as_bool()?,
            "intra_4x4" => self.intra_4x4 = value.as_bool()?,
            "format" => self.format = value.as_str()?.to_string(),
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

/// Pixel formats the encoder accepts as input, in the order the
/// registry descriptor lists them. Grey inputs are coded as 4:2:0 with
/// constant chroma (`1 << (BitDepth − 1)`); the output
/// [`CodecParameters::pixel_format`] names the coded 4:2:0 layout.
pub fn accepted_pixel_formats() -> &'static [PixelFormat] {
    &[
        PixelFormat::Yuv420P,
        PixelFormat::Yuv422P,
        PixelFormat::Yuv444P,
        PixelFormat::Gray8,
        PixelFormat::Yuv420P10Le,
        PixelFormat::Yuv422P10Le,
        PixelFormat::Yuv444P10Le,
        PixelFormat::Gray10Le,
        PixelFormat::Yuv420P12Le,
        PixelFormat::Yuv422P12Le,
        PixelFormat::Yuv444P12Le,
        PixelFormat::Gray12Le,
    ]
}

/// Input sample layout derived from the pixel format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Layout {
    /// Input chroma format (0 = grey input, no chroma planes).
    chroma_format_idc: u32,
    bit_depth: u32,
}

impl Layout {
    fn of(pf: PixelFormat) -> Option<Self> {
        let (cf, bd) = match pf {
            PixelFormat::Yuv420P => (1, 8),
            PixelFormat::Yuv422P => (2, 8),
            PixelFormat::Yuv444P => (3, 8),
            PixelFormat::Gray8 => (0, 8),
            PixelFormat::Yuv420P10Le => (1, 10),
            PixelFormat::Yuv422P10Le => (2, 10),
            PixelFormat::Yuv444P10Le => (3, 10),
            PixelFormat::Gray10Le => (0, 10),
            PixelFormat::Yuv420P12Le => (1, 12),
            PixelFormat::Yuv422P12Le => (2, 12),
            PixelFormat::Yuv444P12Le => (3, 12),
            PixelFormat::Gray12Le => (0, 12),
            _ => return None,
        };
        Some(Self {
            chroma_format_idc: cf,
            bit_depth: bd,
        })
    }

    fn gray(self) -> bool {
        self.chroma_format_idc == 0
    }

    /// Two bytes (little-endian) per sample above 8 bits.
    fn wide(self) -> bool {
        self.bit_depth > 8
    }

    /// `chroma_format_idc` of the coded stream (grey → 4:2:0).
    fn coded_chroma_format_idc(self) -> u32 {
        self.chroma_format_idc.max(1)
    }

    /// Pixel format of the coded stream (what the decoder outputs).
    fn coded_pixel_format(self) -> PixelFormat {
        match (self.coded_chroma_format_idc(), self.bit_depth) {
            (1, 8) => PixelFormat::Yuv420P,
            (2, 8) => PixelFormat::Yuv422P,
            (3, 8) => PixelFormat::Yuv444P,
            (1, 10) => PixelFormat::Yuv420P10Le,
            (2, 10) => PixelFormat::Yuv422P10Le,
            (3, 10) => PixelFormat::Yuv444P10Le,
            (1, _) => PixelFormat::Yuv420P12Le,
            (2, _) => PixelFormat::Yuv422P12Le,
            _ => PixelFormat::Yuv444P12Le,
        }
    }

    /// Chroma plane dimensions of the coded stream.
    fn chroma_dims(self, w: usize, h: usize) -> (usize, usize) {
        match self.coded_chroma_format_idc() {
            1 => (w / 2, h / 2),
            2 => (w / 2, h),
            _ => (w, h),
        }
    }
}

/// High-family streaming driver: GOP cadence, `frame_num` / POC /
/// `idr_pic_id` bookkeeping over a [`DeepSession`].
struct DeepDriver {
    session: DeepSession,
    gop: u64,
    idx: u64,
    gop_start: u64,
    next_idr_pic_id: u32,
}

impl DeepDriver {
    /// Code one picture; returns the Annex B access unit (SPS + PPS +
    /// slice on IDR pictures) and whether it is an IDR.
    fn encode(&mut self, y: &[u16], u: &[u16], v: &[u16]) -> (Vec<u8>, bool) {
        let d = self.idx;
        self.idx += 1;
        let is_idr = d.is_multiple_of(self.gop);
        if is_idr {
            self.gop_start = d;
        }
        let in_gop = d - self.gop_start;
        let kind = if is_idr {
            DeepPictureKind::Idr
        } else {
            DeepPictureKind::P
        };
        // §7.4.3 — every picture on this path is a reference picture,
        // so frame_num counts pictures since the IDR (mod MaxFrameNum
        // inside the session); POC steps by 2 per picture.
        let frame_num = (in_gop & 0xFF) as u32;
        let poc = ((2 * in_gop) & 0x7FFF_FFFF) as i32;
        let idr_pic_id = self.next_idr_pic_id;
        if is_idr {
            self.next_idr_pic_id = (idr_pic_id + 1) & 0xFFFF;
        }
        let pic = self
            .session
            .encode_picture(y, u, v, kind, frame_num, poc, idr_pic_id);
        let mut au = if is_idr {
            self.session.parameter_sets_annex_b()
        } else {
            Vec::new()
        };
        au.extend_from_slice(&pic.nal);
        (au, is_idr)
    }
}

enum Backend {
    /// 8-bit 4:2:0 session (IDR / P / B, rate control).
    Session(Box<EncoderSession>),
    /// High-family layouts (IDR / P, fixed QP).
    Deep(Box<DeepDriver>),
}

/// `oxideav_core::Encoder` implementation. Construct via
/// [`make_encoder`] (the registry factory) or [`H264CodecEncoder::new`].
pub struct H264CodecEncoder {
    codec_id: CodecId,
    output_params: CodecParameters,
    backend: Backend,
    layout: Layout,
    time_base: TimeBase,
    width: usize,
    height: usize,
    frame_index: i64,
    /// Reorder mode: `bframes > 0` (dts = decode counter, pts =
    /// display index + 1).
    reordered: bool,
    /// Decode-order packet counter (dts source in reorder mode).
    decode_index: i64,
    /// `format=avcc`: length-prefixed samples without parameter sets.
    avcc: bool,
    queue: VecDeque<Packet>,
}

/// Where a request is coded.
enum Route {
    Session,
    Deep(u8),
}

impl H264CodecEncoder {
    /// Build an encoder from parameters + option bag. Errors on
    /// missing/invalid geometry, unknown options, a pixel format
    /// outside [`accepted_pixel_formats`], a profile that does not
    /// admit the layout, or inconsistent rate-control settings.
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
        let pf = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
        let layout = Layout::of(pf).ok_or_else(|| {
            Error::unsupported(format!(
                "h264 encoder input must be one of {:?}, got {pf:?}",
                accepted_pixel_formats()
            ))
        })?;
        if opts.gop == 0 {
            return Err(Error::invalid("gop must be >= 1"));
        }
        if opts.qp > 51 {
            return Err(Error::invalid(format!("qp {} out of 0..=51", opts.qp)));
        }
        if opts.bframes > 16 {
            return Err(Error::invalid(format!(
                "bframes {} out of 0..=16",
                opts.bframes
            )));
        }
        let avcc = match opts.format.as_str() {
            "" | "annexb" => false,
            "avcc" => true,
            other => return Err(Error::invalid(format!("unknown format '{other}'"))),
        };

        // Route: the 8-bit 4:2:0 session unless the layout, an explicit
        // High-family profile, or a High-family-only tool asks for the
        // deep encoder.
        let is_8bit_420 = layout.bit_depth == 8 && layout.coded_chroma_format_idc() == 1;
        let coded_cf = layout.coded_chroma_format_idc();
        let bd = layout.bit_depth;
        let high_tools = opts.lossless || opts.transform_8x8 || !opts.deblock;
        let route = match opts.profile.as_str() {
            "" | "auto" => {
                if is_8bit_420 && !high_tools {
                    Route::Session
                } else {
                    Route::Deep(min_profile_idc(coded_cf, bd, bd, opts.lossless))
                }
            }
            p @ ("baseline" | "main") => {
                if !is_8bit_420 {
                    return Err(Error::unsupported(format!(
                        "profile={p} codes 8-bit 4:2:0 only; {pf:?} needs a High-family profile (§A.2)"
                    )));
                }
                if high_tools {
                    return Err(Error::invalid(format!(
                        "profile={p} admits none of lossless / transform_8x8 / deblock=false (High-family tools)"
                    )));
                }
                if p == "baseline" && (opts.cabac || opts.bframes > 0) {
                    return Err(Error::invalid(
                        "profile=baseline forbids CABAC and B pictures (§A.2.1)",
                    ));
                }
                if p == "main" && !(opts.cabac || opts.bframes > 0) {
                    return Err(Error::invalid(
                        "profile=main is signalled only with cabac=true or bframes>0; CAVLC IDR/P streams are Baseline",
                    ));
                }
                Route::Session
            }
            "high" => Route::Deep(100),
            "high10" => Route::Deep(110),
            "high422" => Route::Deep(122),
            "high444" => Route::Deep(244),
            other => return Err(Error::invalid(format!("unknown profile '{other}'"))),
        };

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

        let mut output_params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        output_params.width = Some(width);
        output_params.height = Some(height);
        output_params.pixel_format = Some(layout.coded_pixel_format());
        output_params.frame_rate = params.frame_rate;

        let (backend, parameter_sets) = match route {
            Route::Session => {
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
                    b_frames: opts.bframes,
                    rate_control,
                    recovery_point_sei: false,
                });
                if target_bitrate > 0 && mode.is_some() {
                    output_params.bit_rate = Some(u64::from(target_bitrate));
                }
                let ps = session.parameter_sets_annex_b();
                (Backend::Session(Box::new(session)), ps)
            }
            Route::Deep(profile_idc) => {
                if !profile_admits(profile_idc, coded_cf, bd, bd, opts.lossless) {
                    return Err(Error::unsupported(format!(
                        "profile_idc {profile_idc} does not admit {pf:?}{} (§A.2.4 – §A.2.7)",
                        if opts.lossless {
                            " with lossless coding (transform bypass needs High 4:4:4 Predictive)"
                        } else {
                            ""
                        }
                    )));
                }
                if opts.bframes > 0 {
                    return Err(Error::unsupported(format!(
                        "bframes={} — B pictures are available on the 8-bit 4:2:0 path only (this layout: IDR + P)",
                        opts.bframes
                    )));
                }
                match opts.rc.as_str() {
                    "" | "auto" | "cqp" => {}
                    other => {
                        return Err(Error::unsupported(format!(
                            "rc={other} — rate control is available on the 8-bit 4:2:0 path only (this layout: rc=cqp)"
                        )));
                    }
                }
                let qp = if opts.lossless {
                    -qp_bd_offset(bd - 8)
                } else {
                    opts.qp as i32
                };
                let session = DeepSession::with_profile(
                    DeepConfig {
                        width,
                        height,
                        bit_depth_luma: bd,
                        bit_depth_chroma: bd,
                        chroma_format_idc: coded_cf,
                        qp,
                        p_frames: true,
                        intra_in_p: true,
                        deblock: opts.deblock,
                        lossless: opts.lossless,
                        // Streams that must also reconstruct losslessly in
                        // decoders that skip the §8.5.15 Intra_16x16 /
                        // chroma DPCM (see `DeepConfig::lossless_interop`).
                        lossless_interop: true,
                        cabac: opts.cabac,
                        transform_8x8: opts.transform_8x8,
                    },
                    profile_idc,
                )
                .with_intra_4x4(opts.intra_4x4);
                let ps = session.parameter_sets_annex_b();
                (
                    Backend::Deep(Box::new(DeepDriver {
                        session,
                        gop: u64::from(opts.gop),
                        idx: 0,
                        gop_start: 0,
                        next_idr_pic_id: 0,
                    })),
                    ps,
                )
            }
        };

        if avcc {
            // ISO/IEC 14496-15 §5.3.2.1 — the record is built from the
            // very SPS / PPS the IDR access units would carry in band.
            let record = AvcDecoderConfigurationRecord::from_annex_b(&parameter_sets)
                .map_err(|e| Error::unsupported(format!("h264 encoder: {e}")))?;
            output_params.extradata = record
                .to_bytes()
                .map_err(|e| Error::unsupported(format!("h264 encoder: {e}")))?;
        }

        Ok(Self {
            codec_id: CodecId::new(crate::CODEC_ID_STR),
            output_params,
            backend,
            layout,
            time_base: TimeBase::new(i64::from(fps_den), i64::from(fps_num)),
            width: width as usize,
            height: height as usize,
            frame_index: 0,
            reordered: opts.bframes > 0,
            decode_index: 0,
            avcc,
            queue: VecDeque::new(),
        })
    }

    /// Package one access unit as a packet on the queue.
    fn push_packet(&mut self, annex_b: Vec<u8>, is_idr: bool, pts: i64, dts: i64) -> Result<()> {
        let data = if self.avcc {
            annex_b_to_length_prefixed(&annex_b, 4, true)
                .map_err(|e| Error::invalid(format!("h264 encoder: {e}")))?
        } else {
            annex_b
        };
        let mut packet = Packet::new(0, self.time_base, data)
            .with_pts(pts)
            .with_dts(dts);
        packet.duration = Some(1);
        packet.flags.keyframe = is_idr;
        self.queue.push_back(packet);
        self.decode_index += 1;
        Ok(())
    }

    /// Package the session AUs into packets on the queue.
    fn enqueue(
        &mut self,
        frames: Vec<crate::encoder::session::SessionFrame>,
        input_pts: i64,
    ) -> Result<()> {
        for sf in frames {
            let (pts, dts) = if self.reordered {
                // Frame-duration units: display slot + the one-frame
                // reorder latency vs the decode counter.
                (sf.display_index as i64 + 1, self.decode_index)
            } else {
                (input_pts, input_pts)
            };
            self.push_packet(sf.annex_b, sf.is_idr, pts, dts)?;
        }
        Ok(())
    }

    /// Copy a possibly padded plane into a tightly packed buffer
    /// (`w` in bytes).
    fn packed_plane(plane: &VideoPlane, w: usize, h: usize) -> Result<Vec<u8>> {
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

    /// A packed plane as `u16` samples (little-endian pairs when
    /// `wide`, widened bytes otherwise).
    fn plane_u16(plane: &VideoPlane, w: usize, h: usize, wide: bool) -> Result<Vec<u16>> {
        let bytes = Self::packed_plane(plane, if wide { 2 * w } else { w }, h)?;
        Ok(if wide {
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| u16::from_le_bytes([p[0], p[1]]))
                .collect()
        } else {
            bytes.into_iter().map(u16::from).collect()
        })
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
        let need = if self.layout.gray() { 1 } else { 3 };
        if planes.len() < need {
            return Err(Error::invalid(format!(
                "h264 encoder expects {need} plane(s) for {:?}, got {}",
                self.layout.coded_pixel_format(),
                planes.len()
            )));
        }
        let (w, h) = (self.width, self.height);
        let (cw, ch) = self.layout.chroma_dims(w, h);
        let input_pts = vf.pts.unwrap_or(self.frame_index);

        match &mut self.backend {
            Backend::Session(session) => {
                let y = Self::packed_plane(&planes[0], w, h)?;
                let (u, v) = if self.layout.gray() {
                    (vec![128u8; cw * ch], vec![128u8; cw * ch])
                } else {
                    (
                        Self::packed_plane(&planes[1], cw, ch)?,
                        Self::packed_plane(&planes[2], cw, ch)?,
                    )
                };
                let frames = session.push_frame(&y, &u, &v);
                self.enqueue(frames, input_pts)?;
            }
            Backend::Deep(driver) => {
                let wide = self.layout.wide();
                let y = Self::plane_u16(&planes[0], w, h, wide)?;
                let (u, v) = if self.layout.gray() {
                    let mid = 1u16 << (self.layout.bit_depth - 1);
                    (vec![mid; cw * ch], vec![mid; cw * ch])
                } else {
                    (
                        Self::plane_u16(&planes[1], cw, ch, wide)?,
                        Self::plane_u16(&planes[2], cw, ch, wide)?,
                    )
                };
                let (au, is_idr) = driver.encode(&y, &u, &v);
                self.push_packet(au, is_idr, input_pts, input_pts)?;
            }
        }
        self.frame_index += 1;
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.queue.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        // Drain the reorder buffer (bframes > 0; a no-op otherwise —
        // every send_frame already produced its packet).
        let tail = match &mut self.backend {
            Backend::Session(session) => session.finish(),
            Backend::Deep(_) => Vec::new(),
        };
        let pts = self.frame_index;
        self.enqueue(tail, pts)
    }
}

/// Registry factory (`CodecInfo::encoder`).
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn CoreEncoder>> {
    Ok(Box::new(H264CodecEncoder::new(params)?))
}
