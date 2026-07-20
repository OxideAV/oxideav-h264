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
use crate::encoder::rate_control::{RateControlConfig, RateController, RcFrameKind};
use crate::encoder::{EncodedFrameRef, EncodedIdr, EncodedP, Encoder, EncoderConfig, YuvFrame};

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
    /// Picture payload size in bits (excluding filler).
    pub payload_bits: u64,
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
    /// Absolute frame index (0-based).
    idx: u64,
    /// Most recent reference picture (None before the first frame and
    /// right after construction).
    prev: Option<PrevPic>,
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
        let rc = match cfg.rate_control {
            SessionRateControl::ConstantQp(_) => None,
            SessionRateControl::Controlled(rcfg) => {
                Some(RateController::new(rcfg, cfg.width, cfg.height))
            }
        };
        Self {
            enc: Encoder::new(ecfg),
            cfg,
            rc,
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
    ) -> PrevPic {
        if is_idr {
            let e = if self.cfg.cabac {
                self.enc.encode_idr_cabac_with_qp(frame, qp)
            } else {
                self.enc.encode_idr_with_qp(frame, qp)
            };
            PrevPic::Idr(e)
        } else {
            let prev = self.prev.as_ref().expect("P frame requires a reference");
            let r = prev.as_ref();
            let e = if self.cfg.cabac {
                self.enc
                    .encode_p_cabac_with_qp(frame, &r, frame_num, poc_lsb, qp)
            } else {
                self.enc.encode_p_with_qp(frame, &r, frame_num, poc_lsb, qp)
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
                    self.encode_at(&frame, is_idr, frame_num, poc_lsb, qp),
                    qp,
                    0,
                )
            }
            Some((plan, max_qp)) => {
                let mut qp = plan.qp;
                let mut pic = self.encode_at(&frame, is_idr, frame_num, poc_lsb, qp);
                // VBV hard-cap retry: the stateless encoder makes
                // re-encoding at a higher QP a pure call.
                for _ in 0..MAX_VBV_RETRIES {
                    let bits = 8 * pic_annex_b(&pic).len() as u64;
                    if bits <= plan.max_bits || qp >= max_qp {
                        break;
                    }
                    qp = (qp + 2).min(max_qp);
                    pic = self.encode_at(&frame, is_idr, frame_num, poc_lsb, qp);
                }
                let bits = 8 * pic_annex_b(&pic).len() as u64;
                let outcome = self
                    .rc
                    .as_mut()
                    .expect("rc_plan is Some")
                    .commit_frame(kind, qp, bits);
                (pic, qp, outcome.filler_bits)
            }
        };

        let payload_bits = 8 * pic_annex_b(&pic).len() as u64;
        let mut annex_b = pic_annex_b(&pic).to_vec();
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
