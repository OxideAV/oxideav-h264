//! Multi-frame GOP encode session with optional rate control.
//!
//! [`Encoder`] is deliberately stateless — each `encode_*` call takes
//! its reference picture explicitly and returns the bitstream plus
//! the local reconstruction. This module layers the state a caller
//! otherwise hand-rolls: IDR cadence, `frame_num` / POC counters,
//! reference-picture carry, and (optionally) a [`RateController`]
//! feedback loop that picks a per-frame QP, re-encodes a picture at a
//! higher QP when it would underflow the Annex C CPB (the stateless
//! entry points make retry a pure function call), and appends
//! §7.3.2.7 filler NAL units in CBR mode when a picture underspends
//! its channel slot.
//!
//! Scope: 4:2:0, frame pictures, IDR + linear P chain (the shape the
//! registry encoder drives). CAVLC by default; CABAC via
//! [`SessionConfig::cabac`].

use crate::encoder::nal::build_filler_nal;
use crate::encoder::rate_control::{
    RateControlConfig, RateControlMode, RateController, RcFrameKind,
};
use crate::encoder::sei::{
    build_buffering_period_payload, build_pic_timing_payload, build_sei_nal,
};
use crate::encoder::{EncodedFrameRef, EncodedIdr, EncodedP, Encoder, EncoderConfig, YuvFrame};
use crate::vui::{HrdParameters, TimingInfo, VuiParameters};

/// Rate-control selection for a session.
#[derive(Debug, Clone, Copy)]
pub enum SessionRateControl {
    /// Fixed QP for every picture — the historical behaviour.
    ConstantQp(i32),
    /// Feedback rate control per [`RateControlConfig`].
    Controlled(RateControlConfig),
}

/// Session configuration.
#[derive(Debug, Clone, Copy)]
pub struct SessionConfig {
    /// Picture width in luma samples (multiple of 16).
    pub width: u32,
    /// Picture height in luma samples (multiple of 16).
    pub height: u32,
    /// IDR period in frames: frame `n` is an IDR iff
    /// `n % gop_length == 0`. Must be >= 1 (1 = all-intra).
    pub gop_length: u32,
    /// Entropy coding: `false` = CAVLC (Baseline signalling), `true`
    /// = CABAC (Main signalling).
    pub cabac: bool,
    /// QP policy.
    pub rate_control: SessionRateControl,
}

impl SessionConfig {
    /// Fixed-QP session, CAVLC, one-second GOP at 30 fps.
    pub fn constant_qp(width: u32, height: u32, qp: i32) -> Self {
        Self {
            width,
            height,
            gop_length: 30,
            cabac: false,
            rate_control: SessionRateControl::ConstantQp(qp),
        }
    }

    /// Rate-controlled session, CAVLC.
    pub fn rate_controlled(width: u32, height: u32, rc: RateControlConfig) -> Self {
        Self {
            width,
            height,
            gop_length: 30,
            cabac: false,
            rate_control: SessionRateControl::Controlled(rc),
        }
    }
}

/// One encoded picture as returned by [`EncoderSession::encode_frame`].
#[derive(Debug, Clone)]
pub struct SessionFrame {
    /// Annex B bytes for this access unit: SPS + PPS + IDR slice on
    /// IDR frames, the P slice otherwise, plus any CBR filler NAL.
    pub annex_b: Vec<u8>,
    /// Whether this picture is an IDR (stream random-access point).
    pub is_idr: bool,
    /// QP the picture was finally encoded at (after any VBV retry).
    pub qp: i32,
    /// Access-unit payload size in bits (excluding filler). Includes
    /// the HRD annotation SEI of rate-controlled sessions — those
    /// bits ride the same channel and the CPB model accounts for
    /// them.
    pub payload_bits: u64,
    /// Round-430 — bits of the HRD annotation SEI included in
    /// `payload_bits` (0 for constant-QP sessions). Lets quality-per-
    /// coding-bit comparisons (RD curves vs unannotated fixed-QP
    /// anchors) separate picture coding from metadata overhead.
    pub sei_bits: u64,
    /// Filler bits appended after the picture (CBR underspend), 0
    /// otherwise.
    pub filler_bits: u64,
}

/// Retained state of the most recent reference picture.
enum PrevPic {
    Idr(EncodedIdr),
    P(EncodedP),
}

impl PrevPic {
    fn as_ref(&self) -> EncodedFrameRef<'_> {
        match self {
            PrevPic::Idr(e) => EncodedFrameRef::from(e),
            PrevPic::P(e) => EncodedFrameRef::from(e),
        }
    }
}

/// Stateful GOP driver over the stateless [`Encoder`]. See the module
/// docs.
pub struct EncoderSession {
    enc: Encoder,
    cfg: SessionConfig,
    rc: Option<RateController>,
    /// Round-430 — §E.1.2 NAL HRD block emitted in the SPS VUI of
    /// rate-controlled sessions; also supplies the §D.1.2 / §D.1.3
    /// field widths for the buffering_period / pic_timing SEI. `None`
    /// for constant-QP sessions (no HRD annotation).
    hrd: Option<HrdParameters>,
    /// Absolute frame index (0-based).
    idx: u64,
    /// Most recent reference picture (None before the first frame and
    /// right after construction).
    prev: Option<PrevPic>,
}

/// Round-430 — derive the §E.1.2 NAL HRD block from the rate-control
/// config. One CPB schedule (SchedSelIdx 0): BitRate is the Annex C
/// arrival (channel) rate, CpbSize the VBV size, `cbr_flag` set in CBR
/// mode. Values are coded at scale 0 (64-bit-per-second / 16-bit
/// granularity), rounded UP so the declared bucket never understates
/// the model.
fn session_hrd_parameters(rcfg: &RateControlConfig) -> HrdParameters {
    let arrival = match rcfg.mode {
        RateControlMode::Cbr => rcfg.target_bitrate,
        RateControlMode::CappedVbr => rcfg.max_bitrate.max(rcfg.target_bitrate),
    };
    HrdParameters {
        cpb_cnt_minus1: 0,
        bit_rate_scale: 0,
        cpb_size_scale: 0,
        // §E.2.2 — BitRate[0] = (bit_rate_value_minus1 + 1) << (6 + 0).
        bit_rate_value_minus1: vec![arrival.div_ceil(64) - 1],
        // §E.2.2 — CpbSize[0] = (cpb_size_value_minus1 + 1) << (4 + 0).
        cpb_size_value_minus1: vec![rcfg.vbv_buffer_bits.div_ceil(16) - 1],
        cbr_flag: vec![rcfg.mode == RateControlMode::Cbr],
        // 24-bit delay fields (90 kHz units) + 24-bit time_offset —
        // ample for any buffer this controller models.
        initial_cpb_removal_delay_length_minus1: 23,
        cpb_removal_delay_length_minus1: 23,
        dpb_output_delay_length_minus1: 23,
        time_offset_length: 24,
    }
}

/// Insert `sei` (a complete Annex B SEI NAL) immediately before the
/// first VCL NAL (type 1..=5) of `annex_b`. §7.4.1.2.3 — SEI NALs
/// must precede the slices of the access unit they describe; the
/// encoder entry points return `[SPS, PPS,] slice`, so the SEI slots
/// in right before the slice. Every NAL our encoder emits uses the
/// 4-byte start code.
fn insert_sei_before_first_vcl(annex_b: &[u8], sei: &[u8]) -> Vec<u8> {
    let mut insert_at = annex_b.len();
    let mut i = 0usize;
    while i + 4 < annex_b.len() {
        if annex_b[i..i + 4] == [0, 0, 0, 1] {
            let nal_type = annex_b[i + 4] & 0x1F;
            if (1..=5).contains(&nal_type) {
                insert_at = i;
                break;
            }
            i += 4;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::with_capacity(annex_b.len() + sei.len());
    out.extend_from_slice(&annex_b[..insert_at]);
    out.extend_from_slice(sei);
    out.extend_from_slice(&annex_b[insert_at..]);
    out
}

/// Maximum VBV re-encode attempts per picture. Each retry raises QP
/// by 2; 6 retries span +12 QP (4 quantiser-step doublings), which
/// caps any realistic overshoot.
const MAX_VBV_RETRIES: u32 = 6;

impl EncoderSession {
    /// Create a session. Panics on inconsistent config (non-mod-16
    /// dimensions, `gop_length == 0`, invalid QP / rate parameters).
    pub fn new(cfg: SessionConfig) -> Self {
        assert!(cfg.gop_length >= 1, "gop_length must be >= 1");
        if let SessionRateControl::ConstantQp(qp) = cfg.rate_control {
            assert!((0..=51).contains(&qp), "constant QP out of 0..=51");
        }
        let mut ecfg = EncoderConfig::new(cfg.width, cfg.height);
        if cfg.cabac {
            ecfg.cabac = true;
            ecfg.profile_idc = 77;
        }
        // The PPS anchor QP. Per-frame QPs ride slice_qp_delta, so any
        // mid-range anchor works; the EncoderConfig default (26) keeps
        // fixed-QP streams byte-identical to the historical output
        // when the caller asks for QP 26.
        if let SessionRateControl::ConstantQp(qp) = cfg.rate_control {
            ecfg.qp = qp;
        }
        let (rc, hrd) = match cfg.rate_control {
            SessionRateControl::ConstantQp(_) => (None, None),
            SessionRateControl::Controlled(rcfg) => {
                // Round-430 — HRD/VUI signalling: annotate the stream
                // with §E.1.1 timing info + a §E.1.2 NAL HRD block
                // matching the Annex C leaky-bucket model the
                // controller runs, so the CBR contract is declared
                // in-band (buffering_period / pic_timing SEI carry the
                // per-AU schedule below).
                let hrd = session_hrd_parameters(&rcfg);
                ecfg.vui = Some(VuiParameters {
                    timing_info: Some(TimingInfo {
                        // §E.2.1 — frame duration = 2 clock ticks under
                        // fixed_frame_rate_flag (field-based ticks):
                        // fps = time_scale / (2 * num_units_in_tick).
                        num_units_in_tick: rcfg.fps_den,
                        time_scale: 2 * rcfg.fps_num,
                        fixed_frame_rate_flag: true,
                    }),
                    nal_hrd_parameters: Some(hrd.clone()),
                    low_delay_hrd_flag: Some(false),
                    ..VuiParameters::default()
                });
                (
                    Some(RateController::new(rcfg, cfg.width, cfg.height)),
                    Some(hrd),
                )
            }
        };
        Self {
            enc: Encoder::new(ecfg),
            cfg,
            rc,
            hrd,
            idx: 0,
            prev: None,
        }
    }

    /// Access the rate controller (diagnostics: average bitrate, CPB
    /// fullness). `None` for constant-QP sessions.
    pub fn rate_controller(&self) -> Option<&RateController> {
        self.rc.as_ref()
    }

    /// Number of frames encoded so far.
    pub fn frames_encoded(&self) -> u64 {
        self.idx
    }

    fn encode_at(
        &self,
        frame: &YuvFrame<'_>,
        is_idr: bool,
        frame_num: u32,
        poc_lsb: u32,
        qp: i32,
        row_budget_bits: Option<u64>,
    ) -> PrevPic {
        if is_idr {
            // Round 430 — rate-controlled sessions row-modulate IDR
            // pictures too (they are the largest of a CBR GOP; a
            // mid-frame overshoot used to cost a whole VBV re-encode).
            let e = match (self.cfg.cabac, row_budget_bits) {
                (true, Some(budget)) => self.enc.encode_idr_cabac_rate_adaptive(frame, qp, budget),
                (true, None) => self.enc.encode_idr_cabac_with_qp(frame, qp),
                (false, Some(budget)) => self.enc.encode_idr_rate_adaptive(frame, qp, budget),
                (false, None) => self.enc.encode_idr_with_qp(frame, qp),
            };
            PrevPic::Idr(e)
        } else {
            let prev = self.prev.as_ref().expect("P frame requires a reference");
            let r = prev.as_ref();
            // Rate control: MB-row QP modulation toward the
            // controller's per-frame target (round 420 CAVLC, round
            // 430 CABAC).
            let e = match (self.cfg.cabac, row_budget_bits) {
                (true, Some(budget)) => self
                    .enc
                    .encode_p_cabac_rate_adaptive(frame, &r, frame_num, poc_lsb, qp, budget),
                (true, None) => self
                    .enc
                    .encode_p_cabac_with_qp(frame, &r, frame_num, poc_lsb, qp),
                (false, Some(budget)) => self
                    .enc
                    .encode_p_rate_adaptive(frame, &r, frame_num, poc_lsb, qp, budget),
                (false, None) => self.enc.encode_p_with_qp(frame, &r, frame_num, poc_lsb, qp),
            };
            PrevPic::P(e)
        }
    }

    /// Encode the next frame in display order. Plane layouts are
    /// tightly packed 4:2:0 (`y.len() == w*h`, `u/v.len() == w*h/4`).
    pub fn encode_frame(&mut self, y: &[u8], u: &[u8], v: &[u8]) -> SessionFrame {
        let (w, h) = (self.cfg.width as usize, self.cfg.height as usize);
        assert_eq!(y.len(), w * h, "luma plane size");
        assert_eq!(u.len(), w * h / 4, "cb plane size");
        assert_eq!(v.len(), w * h / 4, "cr plane size");
        let frame = YuvFrame {
            width: self.cfg.width,
            height: self.cfg.height,
            y,
            u,
            v,
        };

        let idx_in_gop = self.idx % u64::from(self.cfg.gop_length);
        let is_idr = idx_in_gop == 0;
        // §7.4.3 — frame_num counts reference frames modulo
        // MaxFrameNum (our SPS: log2_max_frame_num = 8) and resets at
        // IDR; every picture in this session is a reference.
        let frame_num = (idx_in_gop & 0xFF) as u32;
        // §8.2.1 — POC lsb (8 bits in our SPS), display step 2 per
        // frame, resets at IDR.
        let poc_lsb = ((2 * idx_in_gop) & 0xFF) as u32;

        let kind = if is_idr {
            RcFrameKind::Idr
        } else {
            RcFrameKind::P
        };

        // Round-430 — HRD annotation SEI: every rate-controlled AU
        // carries a §D.1.3 pic_timing (cpb_removal_delay in §E.2.1
        // clock ticks since the last buffering-period AU — 2 per frame
        // under the field-based tick — and dpb_output_delay 0: this
        // IPP session outputs each picture at its removal time); IDR
        // AUs additionally lead with a §D.1.2 buffering_period whose
        // initial_cpb_removal_delay is the controller's modelled CPB
        // fill converted to 90 kHz units.
        let sei_nal: Option<Vec<u8>> = match (&self.hrd, &self.rc) {
            (Some(hrd), Some(rc)) => {
                let mut msgs = Vec::new();
                if is_idr {
                    let rcfg = rc.config();
                    let arrival = match rcfg.mode {
                        RateControlMode::Cbr => f64::from(rcfg.target_bitrate),
                        RateControlMode::CappedVbr => {
                            f64::from(rcfg.max_bitrate.max(rcfg.target_bitrate))
                        }
                    };
                    // §D.2.2 — 90 kHz units; delay > 0 mandated.
                    let delay = ((90_000.0 * rc.cpb_fullness() / arrival).round() as u32).max(1);
                    let full =
                        (90_000.0 * f64::from(rcfg.vbv_buffer_bits) / arrival).round() as u32;
                    let offset = full.saturating_sub(delay);
                    msgs.push((
                        0u32,
                        build_buffering_period_payload(0, hrd, &[(delay, offset)]),
                    ));
                }
                let cpb_removal_delay = (2 * idx_in_gop) as u32;
                msgs.push((1u32, build_pic_timing_payload(hrd, cpb_removal_delay, 0)));
                Some(build_sei_nal(&msgs))
            }
            _ => None,
        };
        let sei_bits = sei_nal.as_ref().map_or(0, |s| 8 * s.len() as u64);

        let rc_plan = self
            .rc
            .as_ref()
            .map(|rc| (rc.plan_frame(kind), rc.config().max_qp));
        let (pic, qp_used, filler_bits) = match rc_plan {
            None => {
                let qp = match self.cfg.rate_control {
                    SessionRateControl::ConstantQp(q) => q,
                    SessionRateControl::Controlled(_) => unreachable!(),
                };
                (
                    self.encode_at(&frame, is_idr, frame_num, poc_lsb, qp, None),
                    qp,
                    0,
                )
            }
            Some((plan, max_qp)) => {
                let row_budget = Some(plan.target_bits.max(1.0) as u64);
                let mut qp = plan.qp;
                let mut pic = self.encode_at(&frame, is_idr, frame_num, poc_lsb, qp, row_budget);
                // VBV hard-cap retry: the stateless encoder makes
                // re-encoding at a higher QP a pure call.
                for _ in 0..MAX_VBV_RETRIES {
                    let bits = sei_bits + 8 * pic_annex_b(&pic).len() as u64;
                    if bits <= plan.max_bits || qp >= max_qp {
                        break;
                    }
                    qp = (qp + 2).min(max_qp);
                    pic = self.encode_at(&frame, is_idr, frame_num, poc_lsb, qp, row_budget);
                }
                // The SEI annotation rides the same channel — count it
                // in the committed AU size so the CPB model stays honest.
                let bits = sei_bits + 8 * pic_annex_b(&pic).len() as u64;
                let outcome = self
                    .rc
                    .as_mut()
                    .expect("rc_plan is Some")
                    .commit_frame(kind, qp, bits);
                (pic, qp, outcome.filler_bits)
            }
        };

        let payload_bits = sei_bits + 8 * pic_annex_b(&pic).len() as u64;
        let mut annex_b = match &sei_nal {
            // §7.4.1.2.3 — the SEI precedes the slices of its AU
            // (after SPS/PPS on IDR frames).
            Some(sei) => insert_sei_before_first_vcl(pic_annex_b(&pic), sei),
            None => pic_annex_b(&pic).to_vec(),
        };
        let mut emitted_filler = 0u64;
        if filler_bits > 0 {
            // Whole-NAL rounding: 6 fixed bytes (start code + header +
            // trailing) plus N ff_bytes. Emit only when the request
            // covers at least the fixed cost; the model absorbs the
            // sub-NAL remainder.
            let total_bytes = filler_bits.div_ceil(8);
            if total_bytes >= 6 {
                let ff = (total_bytes - 6) as usize;
                let nal = build_filler_nal(ff);
                emitted_filler = 8 * nal.len() as u64;
                annex_b.extend_from_slice(&nal);
            }
        }

        self.prev = Some(pic);
        self.idx += 1;

        SessionFrame {
            annex_b,
            is_idr,
            qp: qp_used,
            payload_bits,
            sei_bits,
            filler_bits: emitted_filler,
        }
    }
}

fn pic_annex_b(p: &PrevPic) -> &[u8] {
    match p {
        PrevPic::Idr(e) => &e.annex_b,
        PrevPic::P(e) => &e.annex_b,
    }
}
