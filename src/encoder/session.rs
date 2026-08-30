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
//! Scope: 4:2:0, frame pictures; IDR + linear P chain, or — with
//! [`SessionConfig::b_frames`] > 0 — IDR/P anchors with runs of
//! non-reference B pictures between them (display order `I B..B P
//! B..B P …`, coded anchor-first). CAVLC by default; CABAC via
//! [`SessionConfig::cabac`].
//!
//! With B pictures the session buffers up to `b_frames` input
//! pictures (display order) until the next anchor arrives, then
//! emits the whole mini-GOP in decode order: the anchor P first
//! (predicting from the previous anchor), then each B predicting
//! from the two enclosing anchors (`RefPicList0` = previous,
//! `RefPicList1` = next). Push input through
//! [`EncoderSession::push_frame`] (0..=`b_frames`+1 access units
//! out per call) and drain the tail with
//! [`EncoderSession::finish`]; the 0-lookahead
//! [`EncoderSession::encode_frame`] remains for `b_frames == 0`
//! sessions.

use crate::encoder::nal::build_filler_nal;
use crate::encoder::rate_control::{
    RateControlConfig, RateControlMode, RateController, RcFrameKind,
};
use crate::encoder::sei::{
    build_buffering_period_payload, build_pic_timing_payload, build_recovery_point_payload,
    build_sei_nal,
};
use crate::encoder::{
    EncodedB, EncodedFrameRef, EncodedIdr, EncodedP, Encoder, EncoderConfig, YuvFrame,
};
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
    /// Number of consecutive non-reference B pictures coded between
    /// anchor pictures (0 = the historical IDR + linear P chain).
    /// B slices require Main profile, so any `b_frames > 0` promotes
    /// a CAVLC session's signalling from Baseline to Main
    /// (`profile_idc = 77`). The mini-GOP before an IDR (and at
    /// [`EncoderSession::finish`]) is truncated so every B always has
    /// a following anchor in the same coded video sequence: the
    /// session never predicts across an IDR.
    pub b_frames: u32,
    /// QP policy.
    pub rate_control: SessionRateControl,
    /// Round-453 — attach a §D.1.8 `recovery_point` SEI
    /// (`recovery_frame_cnt = 0`, `exact_match_flag = 1`) to every IDR
    /// access unit, marking it as an exact random-access point.
    pub recovery_point_sei: bool,
}

impl SessionConfig {
    /// Fixed-QP session, CAVLC, one-second GOP at 30 fps.
    pub fn constant_qp(width: u32, height: u32, qp: i32) -> Self {
        Self {
            width,
            height,
            gop_length: 30,
            cabac: false,
            b_frames: 0,
            rate_control: SessionRateControl::ConstantQp(qp),
            recovery_point_sei: false,
        }
    }

    /// Rate-controlled session, CAVLC.
    pub fn rate_controlled(width: u32, height: u32, rc: RateControlConfig) -> Self {
        Self {
            width,
            height,
            gop_length: 30,
            cabac: false,
            b_frames: 0,
            rate_control: SessionRateControl::Controlled(rc),
            recovery_point_sei: false,
        }
    }
}

/// Coding kind of one session access unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionFrameKind {
    /// IDR anchor (stream random-access point).
    Idr,
    /// P anchor (reference picture predicting from the previous
    /// anchor).
    P,
    /// Non-reference B picture between two anchors.
    B,
}

/// One encoded picture as returned by [`EncoderSession::encode_frame`]
/// / [`EncoderSession::push_frame`] (in decode order).
#[derive(Debug, Clone)]
pub struct SessionFrame {
    /// Annex B bytes for this access unit: SPS + PPS + IDR slice on
    /// IDR frames, the P or B slice otherwise, plus any CBR filler
    /// NAL.
    pub annex_b: Vec<u8>,
    /// Whether this picture is an IDR (stream random-access point).
    pub is_idr: bool,
    /// Coding kind of this access unit.
    pub kind: SessionFrameKind,
    /// 0-based display index of the source picture this access unit
    /// codes. Equals the decode position for `b_frames == 0`
    /// sessions; with B pictures the anchor of each mini-GOP is
    /// emitted before the Bs that precede it in display order.
    pub display_index: u64,
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

/// One buffered display-order input picture (owned copy of the
/// tightly packed 4:2:0 planes) awaiting its mini-GOP anchor.
struct PendingFrame {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    /// Absolute 0-based display index.
    display_idx: u64,
}

impl PendingFrame {
    fn as_yuv(&self, width: u32, height: u32) -> YuvFrame<'_> {
        YuvFrame {
            width,
            height,
            y: &self.y,
            u: &self.u,
            v: &self.v,
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
    /// Absolute display index of the next input picture (0-based;
    /// counts pictures ACCEPTED, which may still sit in `pending`).
    idx: u64,
    /// Access units emitted so far (decode order).
    encoded: u64,
    /// Display index of the current GOP's IDR picture.
    gop_start: u64,
    /// Decode-order access-unit counter within the current GOP
    /// (drives the §D.1.3 `cpb_removal_delay`, 2 clock ticks per AU).
    dec_in_gop: u64,
    /// §7.4.3 `frame_num` the NEXT reference picture will carry
    /// (modulo MaxFrameNum; resets to 0 at each IDR). A non-reference
    /// B carries the frame_num of the reference picture that precedes
    /// it in decode order, i.e. `next_ref_frame_num - 1`.
    next_ref_frame_num: u32,
    /// Most recent anchor reference picture (None before the first
    /// frame and right after construction).
    prev: Option<PrevPic>,
    /// Display-order pictures buffered since the last anchor
    /// (`b_frames > 0` only; at most `b_frames` entries).
    pending: Vec<PendingFrame>,
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
        // The §8.2.1 POC lsb of our SPS spans 256 values at 2 per
        // picture; keep a whole mini-GOP well inside half that range
        // so the decoder's wrap heuristic never mis-orders it.
        assert!(cfg.b_frames <= 16, "b_frames must be <= 16");
        if let SessionRateControl::ConstantQp(qp) = cfg.rate_control {
            assert!((0..=51).contains(&qp), "constant QP out of 0..=51");
        }
        let mut ecfg = EncoderConfig::new(cfg.width, cfg.height);
        if cfg.cabac {
            ecfg.cabac = true;
            ecfg.profile_idc = 77;
        }
        if cfg.b_frames > 0 {
            // §A.2.2 — Baseline forbids B slices: promote CAVLC
            // sessions to Main signalling. Both mini-GOP anchors must
            // survive the §8.2.5.3 sliding window for the Bs'
            // RefPicList0/RefPicList1.
            ecfg.profile_idc = ecfg.profile_idc.max(77);
            ecfg.max_num_ref_frames = ecfg.max_num_ref_frames.max(2);
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
            encoded: 0,
            gop_start: 0,
            dec_in_gop: 0,
            next_ref_frame_num: 0,
            prev: None,
            pending: Vec::new(),
        }
    }

    /// Access the rate controller (diagnostics: average bitrate, CPB
    /// fullness). `None` for constant-QP sessions.
    pub fn rate_controller(&self) -> Option<&RateController> {
        self.rc.as_ref()
    }

    /// Number of access units emitted so far. With `b_frames > 0`
    /// this can trail the number of pictures pushed by up to
    /// `b_frames` until [`finish`](Self::finish) drains the tail.
    pub fn frames_encoded(&self) -> u64 {
        self.encoded
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

    /// Encode a non-reference B picture between the retained previous
    /// anchor (`RefPicList0[0]`) and `l1` (the mini-GOP's just-coded
    /// anchor, `RefPicList1[0]`). Rate-controlled sessions row-
    /// modulate B pictures like the anchors (round 443).
    fn encode_b_at(
        &self,
        frame: &YuvFrame<'_>,
        l1: &PrevPic,
        frame_num: u32,
        poc_lsb: u32,
        qp: i32,
        row_budget_bits: Option<u64>,
    ) -> EncodedB {
        let l0 = self
            .prev
            .as_ref()
            .expect("B frame requires a previous anchor");
        let r0 = l0.as_ref();
        let r1 = l1.as_ref();
        match (self.cfg.cabac, row_budget_bits) {
            (true, Some(budget)) => self
                .enc
                .encode_b_cabac_rate_adaptive(frame, &r0, &r1, frame_num, poc_lsb, qp, budget),
            (true, None) => self
                .enc
                .encode_b_cabac_with_qp(frame, &r0, &r1, frame_num, poc_lsb, qp),
            (false, Some(budget)) => self
                .enc
                .encode_b_rate_adaptive(frame, &r0, &r1, frame_num, poc_lsb, qp, budget),
            (false, None) => self
                .enc
                .encode_b_with_qp(frame, &r0, &r1, frame_num, poc_lsb, qp),
        }
    }

    /// Encode the next frame in display order — `b_frames == 0`
    /// sessions only (exactly one access unit out per picture in).
    /// Plane layouts are tightly packed 4:2:0 (`y.len() == w*h`,
    /// `u/v.len() == w*h/4`).
    pub fn encode_frame(&mut self, y: &[u8], u: &[u8], v: &[u8]) -> SessionFrame {
        assert_eq!(
            self.cfg.b_frames, 0,
            "encode_frame has no lookahead; use push_frame/finish with b_frames > 0"
        );
        let mut out = self.push_frame(y, u, v);
        debug_assert_eq!(out.len(), 1);
        out.pop().expect("b_frames == 0 emits one AU per picture")
    }

    /// Push the next frame in display order; returns the access units
    /// (decode order) this picture completed — empty while a mini-GOP
    /// is still buffering, the whole mini-GOP (anchor first, then its
    /// Bs) when this picture is an anchor. Call
    /// [`finish`](Self::finish) after the last picture to drain the
    /// buffered tail. Plane layouts are tightly packed 4:2:0.
    pub fn push_frame(&mut self, y: &[u8], u: &[u8], v: &[u8]) -> Vec<SessionFrame> {
        let (w, h) = (self.cfg.width as usize, self.cfg.height as usize);
        assert_eq!(y.len(), w * h, "luma plane size");
        assert_eq!(u.len(), w * h / 4, "cb plane size");
        assert_eq!(v.len(), w * h / 4, "cr plane size");

        let d = self.idx;
        self.idx += 1;
        let mut out = Vec::new();
        if d % u64::from(self.cfg.gop_length) == 0 {
            // IDR slot. Any buffered pictures belong to the CLOSING
            // GOP: emit them first as a truncated mini-GOP (their
            // anchor is the last buffered picture — a B may never
            // predict from across the IDR).
            out.extend(self.flush_pending());
            self.gop_start = d;
            self.dec_in_gop = 0;
            self.next_ref_frame_num = 0;
            let frame = YuvFrame {
                width: self.cfg.width,
                height: self.cfg.height,
                y,
                u,
                v,
            };
            let (sf, pic) = self.encode_au(&frame, SessionFrameKind::Idr, d, None);
            self.prev = pic;
            out.push(sf);
        } else if self.cfg.b_frames == 0 {
            let frame = YuvFrame {
                width: self.cfg.width,
                height: self.cfg.height,
                y,
                u,
                v,
            };
            let (sf, pic) = self.encode_au(&frame, SessionFrameKind::P, d, None);
            self.prev = pic;
            out.push(sf);
        } else {
            self.pending.push(PendingFrame {
                y: y.to_vec(),
                u: u.to_vec(),
                v: v.to_vec(),
                display_idx: d,
            });
            if self.pending.len() as u32 == self.cfg.b_frames + 1 {
                out.extend(self.encode_minigop());
            }
        }
        out
    }

    /// Drain the buffered mini-GOP tail (if any) at end of stream.
    /// The last buffered picture anchors the tail as a P; the rest
    /// code as Bs between the two anchors, exactly like a truncated
    /// pre-IDR mini-GOP.
    pub fn finish(&mut self) -> Vec<SessionFrame> {
        self.flush_pending()
    }

    fn flush_pending(&mut self) -> Vec<SessionFrame> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        self.encode_minigop()
    }

    /// Encode the buffered mini-GOP in decode order: the LAST
    /// buffered picture is the P anchor (predicting from the previous
    /// anchor), the earlier ones are non-reference Bs between the two
    /// anchors. Clears the buffer and promotes the new anchor.
    fn encode_minigop(&mut self) -> Vec<SessionFrame> {
        let mut pend = std::mem::take(&mut self.pending);
        let anchor = pend.pop().expect("encode_minigop needs pending frames");
        let mut out = Vec::with_capacity(pend.len() + 1);

        let frame = anchor.as_yuv(self.cfg.width, self.cfg.height);
        let (sf, pic) = self.encode_au(&frame, SessionFrameKind::P, anchor.display_idx, None);
        out.push(sf);
        let anchor_pic = pic.expect("anchor encode returns a reference picture");

        for b in &pend {
            let frame = b.as_yuv(self.cfg.width, self.cfg.height);
            let (sf, _) = self.encode_au(
                &frame,
                SessionFrameKind::B,
                b.display_idx,
                Some(&anchor_pic),
            );
            out.push(sf);
        }
        self.prev = Some(anchor_pic);
        out
    }

    /// Encode ONE access unit (any kind) with the session's SEI
    /// annotation, rate-control plan / VBV-retry / commit loop and
    /// CBR filler. Advances the decode-order counters; the caller
    /// stores the returned reference picture (anchors only) once the
    /// mini-GOP's Bs — which predict from the PREVIOUS anchor — are
    /// done.
    fn encode_au(
        &mut self,
        frame: &YuvFrame<'_>,
        kind: SessionFrameKind,
        display_idx: u64,
        l1: Option<&PrevPic>,
    ) -> (SessionFrame, Option<PrevPic>) {
        let is_idr = kind == SessionFrameKind::Idr;
        let disp_in_gop = display_idx - self.gop_start;
        // §7.4.3 — frame_num counts reference pictures modulo
        // MaxFrameNum (our SPS: log2_max_frame_num = 8) and resets at
        // IDR. A non-reference B carries the frame_num of the
        // reference picture preceding it in decode order (its
        // mini-GOP's anchor).
        let frame_num = match kind {
            SessionFrameKind::Idr => 0,
            SessionFrameKind::P => self.next_ref_frame_num & 0xFF,
            SessionFrameKind::B => self.next_ref_frame_num.wrapping_sub(1) & 0xFF,
        };
        // §8.2.1 — POC lsb (8 bits in our SPS), display step 2 per
        // frame, resets at IDR.
        let poc_lsb = ((2 * disp_in_gop) & 0xFF) as u32;

        let rc_kind = match kind {
            SessionFrameKind::Idr => RcFrameKind::Idr,
            SessionFrameKind::P => RcFrameKind::P,
            SessionFrameKind::B => RcFrameKind::B,
        };

        // Round-430 — HRD annotation SEI: every rate-controlled AU
        // carries a §D.1.3 pic_timing (cpb_removal_delay in §E.2.1
        // clock ticks since the last buffering-period AU — 2 per
        // DECODE-ORDER access unit under the field-based tick — and
        // dpb_output_delay covering the B-reorder latency: with
        // `b_frames > 0` every picture is output one frame interval
        // after its display slot's removal time, so the delay is
        // `2 * (disp_in_gop + 1 - dec_in_gop)` ticks — 0 for each B
        // (decoded one AU after its display predecessor, output
        // immediately), `2 * (b_count + 1)` for the anchor decoded
        // ahead of its display slot; 0 for the reorder-free b_frames
        // == 0 sessions); IDR AUs additionally lead with a §D.1.2
        // buffering_period whose initial_cpb_removal_delay is the
        // controller's modelled CPB fill converted to 90 kHz units.
        // Round-453 — §D.1.8 recovery_point on IDR access units.
        let mut msgs: Vec<(u32, Vec<u8>)> = Vec::new();
        if self.cfg.recovery_point_sei && is_idr {
            msgs.push((6u32, build_recovery_point_payload(0, true, false, 0)));
        }
        let sei_nal: Option<Vec<u8>> = match (&self.hrd, &self.rc) {
            (Some(hrd), Some(rc)) => {
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
                let cpb_removal_delay = (2 * self.dec_in_gop) as u32;
                let dpb_output_delay = if self.cfg.b_frames > 0 {
                    let d = 2 * (disp_in_gop as i64 + 1 - self.dec_in_gop as i64);
                    debug_assert!(d >= 0, "negative dpb_output_delay");
                    d.max(0) as u32
                } else {
                    0
                };
                msgs.push((
                    1u32,
                    build_pic_timing_payload(hrd, cpb_removal_delay, dpb_output_delay),
                ));
                Some(build_sei_nal(&msgs))
            }
            _ if !msgs.is_empty() => Some(build_sei_nal(&msgs)),
            _ => None,
        };
        let sei_bits = sei_nal.as_ref().map_or(0, |s| 8 * s.len() as u64);

        let rc_plan = self
            .rc
            .as_ref()
            .map(|rc| (rc.plan_frame(rc_kind), rc.config().max_qp));
        let (pic, qp_used, filler_bits) = match rc_plan {
            None => {
                let qp = match self.cfg.rate_control {
                    SessionRateControl::ConstantQp(q) => q,
                    SessionRateControl::Controlled(_) => unreachable!(),
                };
                (
                    self.encode_pic(frame, kind, frame_num, poc_lsb, qp, None, l1),
                    qp,
                    0,
                )
            }
            Some((plan, max_qp)) => {
                // MB-row modulation toward the controller's soft
                // target: anchors since rounds 420/430, B pictures
                // since round 443.
                let row_budget = Some(plan.target_bits.max(1.0) as u64);
                let mut qp = plan.qp;
                let mut pic = self.encode_pic(frame, kind, frame_num, poc_lsb, qp, row_budget, l1);
                // VBV hard-cap retry: the stateless encoder makes
                // re-encoding at a higher QP a pure call.
                for _ in 0..MAX_VBV_RETRIES {
                    let bits = sei_bits + 8 * au_annex_b(&pic).len() as u64;
                    if bits <= plan.max_bits || qp >= max_qp {
                        break;
                    }
                    qp = (qp + 2).min(max_qp);
                    pic = self.encode_pic(frame, kind, frame_num, poc_lsb, qp, row_budget, l1);
                }
                // The SEI annotation rides the same channel — count it
                // in the committed AU size so the CPB model stays honest.
                let bits = sei_bits + 8 * au_annex_b(&pic).len() as u64;
                let outcome = self
                    .rc
                    .as_mut()
                    .expect("rc_plan is Some")
                    .commit_frame(rc_kind, qp, bits);
                (pic, qp, outcome.filler_bits)
            }
        };

        let payload_bits = sei_bits + 8 * au_annex_b(&pic).len() as u64;
        let mut annex_b = match &sei_nal {
            // §7.4.1.2.3 — the SEI precedes the slices of its AU
            // (after SPS/PPS on IDR frames).
            Some(sei) => insert_sei_before_first_vcl(au_annex_b(&pic), sei),
            None => au_annex_b(&pic).to_vec(),
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

        self.dec_in_gop += 1;
        self.encoded += 1;
        let anchor_pic = match pic {
            AuPic::Anchor(p) => {
                self.next_ref_frame_num = (frame_num + 1) & 0xFF;
                Some(p)
            }
            AuPic::B(_) => None,
        };

        (
            SessionFrame {
                annex_b,
                is_idr,
                kind,
                display_index: display_idx,
                qp: qp_used,
                payload_bits,
                sei_bits,
                filler_bits: emitted_filler,
            },
            anchor_pic,
        )
    }

    /// Dispatch one picture encode by kind (pure — no session-state
    /// mutation, so the VBV retry loop can re-invoke it).
    #[allow(clippy::too_many_arguments)]
    fn encode_pic(
        &self,
        frame: &YuvFrame<'_>,
        kind: SessionFrameKind,
        frame_num: u32,
        poc_lsb: u32,
        qp: i32,
        row_budget_bits: Option<u64>,
        l1: Option<&PrevPic>,
    ) -> AuPic {
        match kind {
            SessionFrameKind::Idr | SessionFrameKind::P => AuPic::Anchor(self.encode_at(
                frame,
                kind == SessionFrameKind::Idr,
                frame_num,
                poc_lsb,
                qp,
                row_budget_bits,
            )),
            SessionFrameKind::B => AuPic::B(self.encode_b_at(
                frame,
                l1.expect("B encode requires the mini-GOP anchor"),
                frame_num,
                poc_lsb,
                qp,
                row_budget_bits,
            )),
        }
    }
}

/// One encoded access unit before session packaging.
enum AuPic {
    Anchor(PrevPic),
    B(EncodedB),
}

fn au_annex_b(p: &AuPic) -> &[u8] {
    match p {
        AuPic::Anchor(PrevPic::Idr(e)) => &e.annex_b,
        AuPic::Anchor(PrevPic::P(e)) => &e.annex_b,
        AuPic::B(e) => &e.annex_b,
    }
}
