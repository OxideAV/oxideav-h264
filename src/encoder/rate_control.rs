//! Encoder rate control — target bitrate → per-frame QP adaptation.
//!
//! Rate control is **non-normative**: §7.4.3 only defines how
//! `slice_qp_delta` / `mb_qp_delta` map to QP_Y, and Annex C defines
//! the hypothetical reference decoder (HRD) buffer model a conformant
//! stream must not violate. *How* an encoder picks QP per frame is
//! encoder freedom. This module implements:
//!
//! * a per-frame feedback controller mapping a target bitrate to a
//!   QP per frame ([`RateController`]), driven by a
//!   complexity-times-quantiser-step model that is re-estimated from
//!   the actual bit cost of every committed frame;
//! * a leaky-bucket coded-picture-buffer (CPB) model in the spirit of
//!   Annex C §C.1: bits arrive at the channel rate, each access unit
//!   is removed instantaneously one frame interval apart. The
//!   controller keeps the modelled fullness inside `[0, buffer]` by
//!   biasing frame targets and by reporting a hard per-frame ceiling
//!   ([`FramePlan::max_bits`]) plus, in CBR mode, the number of
//!   filler bits (§7.3.2.7 `filler_data_rbsp`) needed to stop the
//!   bucket overflowing;
//! * two modes: [`RateControlMode::Cbr`] (channel rate == target
//!   rate, filler emitted on underspend) and
//!   [`RateControlMode::CappedVbr`] (average tracks the target rate,
//!   instantaneous arrivals bounded by a separate maximum rate, no
//!   filler).
//!
//! The QP↔quantiser-step relationship follows §8.5.9: QpC/quantiser
//! scaling doubles every 6 QP steps, with step 1.0 anchored at QP 4.
//! Any monotone anchor works — the feedback loop corrects model error
//! — but using the spec curve makes the first-frame guess land close.

/// Rate-control operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControlMode {
    /// Constant bit rate. The Annex C CPB is fed at exactly
    /// `target_bitrate`; when the encoder underspends enough that the
    /// bucket would overflow, [`CommitOutcome::filler_bits`] reports
    /// how many filler bits (§7.3.2.7) the caller must append to the
    /// access unit to keep the channel busy.
    Cbr,
    /// Capped variable bit rate. The long-term average converges on
    /// `target_bitrate`; the CPB is fed at `max_bitrate` (arrivals
    /// pause when the bucket is full, per the Annex C VBR
    /// interpretation), and no filler is ever requested.
    CappedVbr,
}

/// Static configuration for a [`RateController`].
#[derive(Debug, Clone, Copy)]
pub struct RateControlConfig {
    /// Operating mode.
    pub mode: RateControlMode,
    /// Long-term average target, bits per second. Must be > 0.
    pub target_bitrate: u32,
    /// Peak channel rate, bits per second. Ignored (treated as equal
    /// to `target_bitrate`) in CBR mode; must be `>= target_bitrate`
    /// in capped-VBR mode.
    pub max_bitrate: u32,
    /// CPB / VBV size in **bits**. Must be > 0. One second's worth of
    /// target rate is a conventional default (see
    /// [`RateControlConfig::cbr`]).
    pub vbv_buffer_bits: u32,
    /// Frame rate numerator (frames) — e.g. 30000 for 29.97 fps.
    pub fps_num: u32,
    /// Frame rate denominator (seconds) — e.g. 1001 for 29.97 fps.
    pub fps_den: u32,
    /// Smallest QP the controller may pick (inclusive). Range 0..=51.
    pub min_qp: i32,
    /// Largest QP the controller may pick (inclusive). Range 0..=51.
    pub max_qp: i32,
}

impl RateControlConfig {
    /// CBR at `target_bitrate` bps with a one-second CPB.
    pub fn cbr(target_bitrate: u32, fps_num: u32, fps_den: u32) -> Self {
        Self {
            mode: RateControlMode::Cbr,
            target_bitrate,
            max_bitrate: target_bitrate,
            vbv_buffer_bits: target_bitrate,
            fps_num,
            fps_den,
            min_qp: 10,
            max_qp: 51,
        }
    }

    /// Capped VBR: long-term average `target_bitrate`, peak
    /// `max_bitrate`, one-second CPB at the peak rate.
    pub fn capped_vbr(target_bitrate: u32, max_bitrate: u32, fps_num: u32, fps_den: u32) -> Self {
        Self {
            mode: RateControlMode::CappedVbr,
            target_bitrate,
            max_bitrate,
            vbv_buffer_bits: max_bitrate,
            fps_num,
            fps_den,
            min_qp: 10,
            max_qp: 51,
        }
    }

    fn validate(&self) {
        assert!(self.target_bitrate > 0, "target_bitrate must be > 0");
        assert!(self.vbv_buffer_bits > 0, "vbv_buffer_bits must be > 0");
        assert!(self.fps_num > 0 && self.fps_den > 0, "fps must be > 0");
        assert!(
            (0..=51).contains(&self.min_qp) && (0..=51).contains(&self.max_qp),
            "QP bounds must be within 0..=51"
        );
        assert!(self.min_qp <= self.max_qp, "min_qp must be <= max_qp");
        if self.mode == RateControlMode::CappedVbr {
            assert!(
                self.max_bitrate >= self.target_bitrate,
                "max_bitrate must be >= target_bitrate in capped-VBR mode"
            );
        }
    }

    /// Channel (arrival) rate feeding the CPB model, bits/second.
    fn arrival_rate(&self) -> f64 {
        match self.mode {
            RateControlMode::Cbr => f64::from(self.target_bitrate),
            RateControlMode::CappedVbr => f64::from(self.max_bitrate.max(self.target_bitrate)),
        }
    }
}

/// Frame kind as seen by the controller. Each kind carries its own
/// complexity model (EWMA of `bits * qstep`), so the controller learns
/// the relative cost of I, P and B pictures from the stream itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RcFrameKind {
    /// IDR (or any intra) picture.
    Idr,
    /// Predicted picture.
    P,
    /// Non-reference bi-predicted picture. Because a B picture's
    /// distortion does not propagate (nothing references it), the
    /// controller quantises it coarser than the anchors: its bit
    /// target is scaled so the resulting quantiser step lands
    /// [`B_QP_OFFSET`] QP above the P pictures' (λ-scaled hierarchy
    /// — the §8.5.9 step doubles every 6 QP, so a fixed step ratio
    /// IS a fixed QP offset).
    B,
}

/// Nominal QP offset of non-reference B pictures above P pictures.
/// Applied through the target-bits domain: the B bit target is divided
/// by `2^(B_QP_OFFSET / 6)` (the §8.5.9 step ratio for that offset),
/// so the complexity model maps it back to `QP_P + B_QP_OFFSET` when
/// the content's B/P cost ratio is stable.
const B_QP_OFFSET: f64 = 2.0;

/// Number of frame kinds the controller models.
const NUM_KINDS: usize = 3;

/// Per-frame plan returned by [`RateController::plan_frame`].
#[derive(Debug, Clone, Copy)]
pub struct FramePlan {
    /// QP the frame should be encoded at.
    pub qp: i32,
    /// Soft bit target the QP was derived from (diagnostic).
    pub target_bits: f64,
    /// Hard ceiling: encoding more than this many bits underflows the
    /// CPB model. Callers that can re-encode should retry at a higher
    /// QP while the produced frame exceeds this.
    pub max_bits: u64,
}

/// Result of [`RateController::commit_frame`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CommitOutcome {
    /// CBR only: number of filler bits the caller must transmit with
    /// this access unit so the CPB does not overflow. Rounding up to
    /// whole filler-NAL bytes (§7.3.2.7) is the caller's job; the
    /// model absorbs the sub-byte difference on the next commit.
    pub filler_bits: u64,
    /// The committed frame was larger than the modelled CPB fullness —
    /// a decoder driving Annex C timing at exactly the declared rate
    /// would stall. The model clamps at zero and keeps going, but the
    /// caller should have honoured [`FramePlan::max_bits`].
    pub vbv_underflow: bool,
}

/// §8.5.9-anchored quantiser step: doubles every 6 QP, step 1.0 at
/// QP 4.
fn qstep(qp: i32) -> f64 {
    (2.0f64).powf((qp as f64 - 4.0) / 6.0)
}

/// Inverse of [`qstep`], unrounded.
fn qp_from_qstep(step: f64) -> f64 {
    4.0 + 6.0 * step.max(1e-9).log2()
}

/// Feedback rate controller. See the module docs for the model.
#[derive(Debug, Clone)]
pub struct RateController {
    cfg: RateControlConfig,
    /// Bits per frame interval at the target rate.
    bits_per_frame: f64,
    /// Bits per frame interval at the channel/arrival rate.
    arrival_per_frame: f64,
    /// Modelled decoder-side CPB fullness in bits, after the most
    /// recent removal. Starts just above the 55% steering setpoint
    /// (0.6 x buffer, i.e. a 0.6-buffer initial_cpb_removal_delay):
    /// enough banked bits for the opening IDR without a long
    /// spend-down transient that would inflate the early payload rate
    /// above target.
    fullness: f64,
    /// Complexity model per frame kind: EWMA of `bits * qstep(qp)`.
    cplx: [f64; NUM_KINDS],
    /// Whether `cplx[kind]` has been seeded by a real frame yet.
    cplx_seeded: [bool; NUM_KINDS],
    /// Last QP committed per frame kind (for step clamping).
    last_qp: [Option<i32>; NUM_KINDS],
    /// Frames committed per kind (the observed GOP mix).
    kind_count: [u64; NUM_KINDS],
    /// Total frames committed.
    frames: u64,
    /// Total payload bits committed (excluding filler).
    total_bits: u64,
    /// Total filler bits requested (CBR).
    total_filler_bits: u64,
    /// Picture area in luma samples, for the cold-start QP heuristic.
    pixels_per_frame: f64,
}

impl RateController {
    /// Build a controller for pictures of `width` x `height` luma
    /// samples. Panics when the config is inconsistent (zero rates,
    /// inverted QP bounds, capped-VBR max below target).
    pub fn new(cfg: RateControlConfig, width: u32, height: u32) -> Self {
        cfg.validate();
        assert!(width > 0 && height > 0);
        let bits_per_frame =
            f64::from(cfg.target_bitrate) * f64::from(cfg.fps_den) / f64::from(cfg.fps_num);
        let arrival_per_frame =
            cfg.arrival_rate() * f64::from(cfg.fps_den) / f64::from(cfg.fps_num);
        Self {
            cfg,
            bits_per_frame,
            arrival_per_frame,
            fullness: 0.6 * f64::from(cfg.vbv_buffer_bits),
            cplx: [0.0; NUM_KINDS],
            cplx_seeded: [false; NUM_KINDS],
            last_qp: [None; NUM_KINDS],
            kind_count: [0; NUM_KINDS],
            frames: 0,
            total_bits: 0,
            total_filler_bits: 0,
            pixels_per_frame: f64::from(width) * f64::from(height),
        }
    }

    /// Access the configuration this controller runs with.
    pub fn config(&self) -> &RateControlConfig {
        &self.cfg
    }

    /// Modelled CPB fullness in bits (diagnostic).
    pub fn cpb_fullness(&self) -> f64 {
        self.fullness
    }

    /// Average payload bits per second over the committed frames
    /// (excluding CBR filler); `None` before the first commit.
    pub fn average_bitrate(&self) -> Option<f64> {
        if self.frames == 0 {
            return None;
        }
        let seconds =
            self.frames as f64 * f64::from(self.cfg.fps_den) / f64::from(self.cfg.fps_num);
        Some(self.total_bits as f64 / seconds)
    }

    /// Total filler bits requested so far (CBR accounting).
    pub fn total_filler_bits(&self) -> u64 {
        self.total_filler_bits
    }

    fn kind_index(kind: RcFrameKind) -> usize {
        match kind {
            RcFrameKind::Idr => 0,
            RcFrameKind::P => 1,
            RcFrameKind::B => 2,
        }
    }

    /// Nominal QP offset of each kind relative to the P baseline —
    /// used only to derive a first QP for a kind that has no
    /// complexity history yet from a kind that does. IDR opens finer
    /// (its samples seed every later prediction), non-reference B
    /// opens coarser (nothing references it).
    fn kind_qp_bias(kind: RcFrameKind) -> i32 {
        match kind {
            RcFrameKind::Idr => -2,
            RcFrameKind::P => 0,
            RcFrameKind::B => 2,
        }
    }

    /// Relative per-frame cost weights `(u_I, u_P, u_B)`: the
    /// modelled complexity of each kind, with the B entry divided by
    /// the [`B_QP_OFFSET`] step ratio so that allocating bits
    /// proportionally to `u` yields equal quantiser steps on I/P and
    /// a step `B_QP_OFFSET` QP coarser on B. Unseeded kinds fall back
    /// to nominal ratios against the best seeded reference; ratios of
    /// seeded kinds are clamped so a single pathological frame cannot
    /// starve the others.
    fn kind_weights(&self) -> [f64; NUM_KINDS] {
        /// Nominal intra-over-P cost ratio before both are observed.
        const IDR_DEFAULT_RATIO: f64 = 4.0;
        /// Nominal B-over-P cost ratio before both are observed —
        /// bi-prediction between two nearby anchors routinely halves
        /// the residual cost.
        const B_DEFAULT_RATIO: f64 = 0.6;
        let refc = if self.cplx_seeded[1] && self.cplx[1] > 0.0 {
            self.cplx[1]
        } else if self.cplx_seeded[0] && self.cplx[0] > 0.0 {
            self.cplx[0] / IDR_DEFAULT_RATIO
        } else if self.cplx_seeded[2] && self.cplx[2] > 0.0 {
            self.cplx[2] / B_DEFAULT_RATIO
        } else {
            1.0
        };
        let u_i = if self.cplx_seeded[0] {
            (self.cplx[0] / refc).clamp(1.0, 6.0) * refc
        } else {
            IDR_DEFAULT_RATIO * refc
        };
        let u_p = if self.cplx_seeded[1] {
            self.cplx[1].max(refc * 0.1)
        } else {
            refc
        };
        let u_b_raw = if self.cplx_seeded[2] {
            (self.cplx[2] / refc).clamp(0.125, 1.5) * refc
        } else {
            B_DEFAULT_RATIO * refc
        };
        let u_b = u_b_raw / (2.0f64).powf(B_QP_OFFSET / 6.0);
        [u_i, u_p, u_b]
    }

    /// Observed kind mix: cumulative per-kind frequencies blended
    /// with a nominal one-GOP IPP prior so cold-start allocation
    /// stays sane before the real GOP structure has been seen (the
    /// prior washes out after the first GOP's worth of commits).
    fn kind_mix(&self) -> [f64; NUM_KINDS] {
        const PRIOR_FRAMES: f64 = 30.0;
        const PRIOR: [f64; NUM_KINDS] = [0.1, 0.9, 0.0];
        let total = self.frames as f64 + PRIOR_FRAMES;
        [0usize, 1, 2].map(|k| (PRIOR[k] * PRIOR_FRAMES + self.kind_count[k] as f64) / total)
    }

    /// The seeded kind closest (in prediction distance) to `kind`,
    /// with its last committed QP — the donor for a cold-start QP
    /// derivation. Preference order: P is the baseline everyone
    /// derives from; anchors prefer each other over B.
    fn nearest_seeded_kind(&self, kind: RcFrameKind) -> Option<(RcFrameKind, i32)> {
        let order: [RcFrameKind; 2] = match kind {
            RcFrameKind::Idr => [RcFrameKind::P, RcFrameKind::B],
            RcFrameKind::P => [RcFrameKind::Idr, RcFrameKind::B],
            RcFrameKind::B => [RcFrameKind::P, RcFrameKind::Idr],
        };
        order
            .into_iter()
            .find_map(|k| self.last_qp[Self::kind_index(k)].map(|qp| (k, qp)))
    }

    /// Cold-start QP guess from target bits-per-pixel. Coarse on
    /// purpose — the feedback loop takes over after one frame.
    fn bootstrap_qp(&self, kind: RcFrameKind) -> i32 {
        let bpp = self.bits_per_frame / self.pixels_per_frame;
        let base = if bpp > 0.5 {
            18
        } else if bpp > 0.25 {
            22
        } else if bpp > 0.12 {
            26
        } else if bpp > 0.06 {
            30
        } else if bpp > 0.03 {
            34
        } else {
            38
        };
        let biased = match kind {
            RcFrameKind::Idr => base - 3,
            RcFrameKind::P => base,
            RcFrameKind::B => base + 2,
        };
        biased.clamp(self.cfg.min_qp, self.cfg.max_qp)
    }

    /// Soft bit target for the next frame of `kind`, before QP
    /// mapping. Pro-rata per-frame budget, plus mode-specific
    /// steering:
    ///
    /// * CBR — a buffer-position term pulling the CPB toward 55%
    ///   fullness over about one second. Because the CBR bucket is
    ///   fed at exactly the target rate, holding fullness steady IS
    ///   holding the average on target, so no separate integrator is
    ///   needed.
    /// * Capped VBR — the bucket is fed at the peak rate, so its
    ///   position says nothing about the long-term average; instead a
    ///   proportional-on-accumulated-error integrator (gain 2 /
    ///   window, i.e. a ~half-second time constant) drives the
    ///   running payload average onto the target, and a low-water
    ///   penalty term backs off when the bucket drops under 35%.
    fn frame_target(&self, kind: RcFrameKind) -> f64 {
        let buffer = f64::from(self.cfg.vbv_buffer_bits);
        let fps = f64::from(self.cfg.fps_num) / f64::from(self.cfg.fps_den);
        let window = fps.max(1.0);

        let mut t = self.bits_per_frame;
        match self.cfg.mode {
            RateControlMode::Cbr => {
                // Fullness above the setpoint means the channel has
                // banked bits we are entitled to spend.
                t += 0.5 * (self.fullness - 0.55 * buffer) / window;
            }
            RateControlMode::CappedVbr => {
                if self.frames > 0 {
                    let ideal = self.bits_per_frame * self.frames as f64;
                    let err = self.total_bits as f64 - ideal;
                    let correction = (2.0 * err / window).clamp(-0.6 * t, 0.6 * t);
                    t -= correction;
                }
                // Only a penalty: peak-rate arrivals refill fast, so a
                // high bucket is not licence to overspend the target.
                let low_water = 0.25 * (self.fullness - 0.35 * buffer) / window;
                t += low_water.min(0.0);
            }
        }

        // Kind weighting: split the steered per-frame budget across
        // the observed I/P/B mix in proportion to each kind's cost
        // weight (`t_kind = t * u_kind / Σ_j f_j·u_j`). The mix
        // frequencies make the split conserve the budget exactly
        // (`Σ_k f_k·t_k = t`) whatever the GOP structure, and because
        // bits ≈ cplx / qstep the resulting quantiser steps are EQUAL
        // for I and P (complexity-proportional anchor allocation)
        // and `B_QP_OFFSET` QP coarser for non-reference B (whose
        // weight is divided by the corresponding §8.5.9 step ratio) —
        // a λ-scaled I/P/B hierarchy.
        let u = self.kind_weights();
        let f = self.kind_mix();
        let denom: f64 = (0..NUM_KINDS).map(|j| f[j] * u[j]).sum();
        if denom > 0.0 {
            t *= u[Self::kind_index(kind)] / denom;
        }

        // Never plan below a floor (QP clamping handles starvation
        // more gracefully than a zero target) and never plan into CPB
        // underflow: the frame is removed after one more arrival
        // interval, keep 10% margin.
        let max_safe = 0.9 * (self.fullness + self.arrival_per_frame);
        t.clamp(self.bits_per_frame * 0.1, max_safe.max(1.0))
    }

    /// Plan the next frame: pick QP and report the soft target plus
    /// the hard CPB ceiling. Does not mutate controller state — call
    /// [`commit_frame`](Self::commit_frame) with the encode result
    /// (after any caller-side re-encode loop) to advance the model.
    pub fn plan_frame(&self, kind: RcFrameKind) -> FramePlan {
        let k = Self::kind_index(kind);
        let target = self.frame_target(kind);

        let qp = if self.cplx_seeded[k] {
            let want_step = self.cplx[k] / target.max(1.0);
            let raw = qp_from_qstep(want_step).round() as i32;
            // Clamp the per-frame step so the controller cannot
            // oscillate wildly on complexity spikes.
            let clamped = match self.last_qp[k] {
                Some(prev) => raw.clamp(prev - 3, prev + 3),
                None => raw,
            };
            clamped.clamp(self.cfg.min_qp, self.cfg.max_qp)
        } else if let Some((other_kind, other_qp)) = self.nearest_seeded_kind(kind) {
            // No history for this kind yet — derive from the nearest
            // seeded kind, shifted by the nominal inter-kind biases
            // (IDR a touch lower than P, B a touch higher).
            let bias = Self::kind_qp_bias(kind) - Self::kind_qp_bias(other_kind);
            (other_qp + bias).clamp(self.cfg.min_qp, self.cfg.max_qp)
        } else {
            self.bootstrap_qp(kind)
        };

        // Hard ceiling: bits available in the bucket when this frame
        // is removed (fullness now + one arrival interval), minus a 5%
        // safety margin.
        let max_bits = (0.95 * (self.fullness + self.arrival_per_frame)).max(1.0) as u64;

        FramePlan {
            qp,
            target_bits: target,
            max_bits,
        }
    }

    /// Commit the encode result for one frame: update the complexity
    /// model, run the leaky bucket one frame interval, and report CBR
    /// filler / underflow status.
    pub fn commit_frame(&mut self, kind: RcFrameKind, qp_used: i32, bits: u64) -> CommitOutcome {
        let k = Self::kind_index(kind);
        assert!((0..=51).contains(&qp_used));

        // Complexity model update (EWMA, 60/40 old/new).
        let observed = bits as f64 * qstep(qp_used);
        self.cplx[k] = if self.cplx_seeded[k] {
            0.6 * self.cplx[k] + 0.4 * observed
        } else {
            observed
        };
        self.cplx_seeded[k] = true;
        self.last_qp[k] = Some(qp_used);
        self.kind_count[k] += 1;
        self.frames += 1;
        self.total_bits += bits;

        // Leaky bucket: one frame interval of arrivals, then remove
        // the access unit.
        let buffer = f64::from(self.cfg.vbv_buffer_bits);
        self.fullness += self.arrival_per_frame;
        // Arrivals cannot exceed the physical buffer even before
        // removal (the channel pauses / the mux stalls).
        let mut outcome = CommitOutcome::default();
        self.fullness -= bits as f64;
        if self.fullness < 0.0 {
            outcome.vbv_underflow = true;
            self.fullness = 0.0;
        }
        if self.fullness > buffer {
            match self.cfg.mode {
                RateControlMode::Cbr => {
                    // The encoder must keep the CBR channel busy:
                    // request filler for the overflow amount.
                    let filler = self.fullness - buffer;
                    outcome.filler_bits = filler.ceil() as u64;
                    self.total_filler_bits += outcome.filler_bits;
                    self.fullness = buffer;
                }
                RateControlMode::CappedVbr => {
                    // VBR arrivals simply pause when the bucket fills.
                    self.fullness = buffer;
                }
            }
        }
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic "encoder": bits = complexity / qstep, with a floor.
    fn synth_bits(complexity: f64, qp: i32) -> u64 {
        (complexity / qstep(qp)).max(64.0) as u64
    }

    fn run_sequence(
        mut rc: RateController,
        frames: usize,
        gop: usize,
        complexity_i: f64,
        complexity_p: f64,
        wobble: impl Fn(usize) -> f64,
    ) -> (f64, RateController) {
        let mut total_bits = 0u64;
        for n in 0..frames {
            let kind = if n % gop == 0 {
                RcFrameKind::Idr
            } else {
                RcFrameKind::P
            };
            let plan = rc.plan_frame(kind);
            assert!((rc.config().min_qp..=rc.config().max_qp).contains(&plan.qp));
            let base = match kind {
                RcFrameKind::Idr => complexity_i,
                RcFrameKind::P | RcFrameKind::B => complexity_p,
            };
            let mut qp = plan.qp;
            let mut bits = synth_bits(base * wobble(n), qp);
            // Caller-side hard-cap loop, as a session would do.
            while bits > plan.max_bits && qp < rc.config().max_qp {
                qp += 2;
                bits = synth_bits(base * wobble(n), qp);
            }
            let outcome = rc.commit_frame(kind, qp, bits);
            assert!(
                !outcome.vbv_underflow,
                "underflow at frame {n} (bits={bits}, cap={})",
                plan.max_bits
            );
            total_bits += bits + outcome.filler_bits;
        }
        let seconds =
            frames as f64 * f64::from(rc.config().fps_den) / f64::from(rc.config().fps_num);
        (total_bits as f64 / seconds, rc)
    }

    #[test]
    fn cbr_converges_and_fills_channel() {
        let cfg = RateControlConfig::cbr(1_000_000, 30, 1);
        let rc = RateController::new(cfg, 640, 368);
        let f_start = rc.cpb_fullness();
        let (sent_rate, rc) = run_sequence(rc, 300, 30, 3.0e6, 6.0e5, |n| {
            1.0 + 0.3 * ((n % 7) as f64 - 3.0) / 3.0
        });
        // Bucket conservation: sent bits (payload + filler) must equal
        // arrivals plus the net fullness drain, up to per-frame filler
        // rounding (< 1 bit per frame). This is what "the CBR channel
        // is always exactly full" means once the initial-removal-delay
        // fullness offset is accounted for.
        let seconds = 10.0;
        let arrivals = 1_000_000.0 * seconds;
        let expected_sent = arrivals + (f_start - rc.cpb_fullness());
        let sent = sent_rate * seconds;
        assert!(
            (sent - expected_sent).abs() < 400.0,
            "CBR conservation broken: sent {sent}, expected {expected_sent}"
        );
        // Payload alone must also be close: filler should be a small
        // fraction on content that can use the rate.
        let payload = rc.average_bitrate().unwrap();
        assert!(
            (payload - 1_000_000.0).abs() / 1_000_000.0 < 0.10,
            "payload average {payload} too far from target"
        );
    }

    #[test]
    fn capped_vbr_average_tracks_target() {
        let cfg = RateControlConfig::capped_vbr(800_000, 1_600_000, 25, 1);
        let rc = RateController::new(cfg, 640, 368);
        let (_, rc) = run_sequence(rc, 500, 50, 2.5e6, 5.0e5, |n| {
            if (n / 100) % 2 == 0 {
                1.0
            } else {
                2.2 // sustained complexity surge
            }
        });
        let avg = rc.average_bitrate().unwrap();
        let err = (avg - 800_000.0).abs() / 800_000.0;
        assert!(err < 0.05, "capped-VBR average {avg} off by {err:.4}");
        assert_eq!(rc.total_filler_bits(), 0, "VBR must not request filler");
    }

    #[test]
    fn bucket_never_escapes_bounds() {
        let cfg = RateControlConfig {
            mode: RateControlMode::Cbr,
            target_bitrate: 200_000,
            max_bitrate: 200_000,
            vbv_buffer_bits: 100_000, // tight half-second bucket
            fps_num: 30,
            fps_den: 1,
            min_qp: 10,
            max_qp: 51,
        };
        let mut rc = RateController::new(cfg, 320, 240);
        for n in 0..200 {
            let kind = if n % 40 == 0 {
                RcFrameKind::Idr
            } else {
                RcFrameKind::P
            };
            let plan = rc.plan_frame(kind);
            // Adversarial encoder: always emits exactly the hard cap.
            let outcome = rc.commit_frame(kind, plan.qp, plan.max_bits);
            assert!(!outcome.vbv_underflow, "cap-sized frame underflowed at {n}");
            let f = rc.cpb_fullness();
            assert!((0.0..=100_000.0 + 1.0).contains(&f), "fullness {f} escaped");
        }
    }

    /// IBBP…-patterned run: `gop` frames per IDR, `nb` Bs between
    /// anchors, per-kind synthetic complexities. Frames are walked in
    /// coding order (anchor before the Bs that precede it in display
    /// order) — the order the session commits in.
    fn run_b_sequence(
        mut rc: RateController,
        minigops: usize,
        gop_anchors: usize,
        nb: usize,
        cplx: [f64; 3],
    ) -> (f64, RateController, Vec<(RcFrameKind, i32)>) {
        let mut total_bits = 0u64;
        let mut log = Vec::new();
        let mut frames = 0usize;
        for m in 0..minigops {
            let anchor_kind = if m % gop_anchors == 0 {
                RcFrameKind::Idr
            } else {
                RcFrameKind::P
            };
            for kind in std::iter::once(anchor_kind).chain(std::iter::repeat_n(RcFrameKind::B, nb))
            {
                let plan = rc.plan_frame(kind);
                let base = cplx[RateController::kind_index(kind)];
                let mut qp = plan.qp;
                let mut bits = synth_bits(base, qp);
                while bits > plan.max_bits && qp < rc.config().max_qp {
                    qp += 2;
                    bits = synth_bits(base, qp);
                }
                let outcome = rc.commit_frame(kind, qp, bits);
                assert!(!outcome.vbv_underflow, "underflow in minigop {m}");
                total_bits += bits + outcome.filler_bits;
                log.push((kind, qp));
                frames += 1;
            }
        }
        let seconds =
            frames as f64 * f64::from(rc.config().fps_den) / f64::from(rc.config().fps_num);
        (total_bits as f64 / seconds, rc, log)
    }

    #[test]
    fn cbr_b_gop_converges_with_qp_hierarchy() {
        let cfg = RateControlConfig::cbr(1_000_000, 30, 1);
        let rc = RateController::new(cfg, 640, 368);
        // 100 minigops of (anchor + 2 B) = 300 frames; IDR every 10
        // anchors. B complexity well under P (bi-prediction pays).
        let f_start = rc.cpb_fullness();
        let (sent_rate, rc, log) = run_b_sequence(rc, 100, 10, 2, [3.0e6, 6.0e5, 2.5e5]);
        // CBR bucket conservation: sent bits (payload + filler) equal
        // arrivals plus the net fullness drain (see
        // `cbr_converges_and_fills_channel`), up to filler rounding.
        let seconds = 10.0;
        let arrivals = 1_000_000.0 * seconds;
        let expected_sent = arrivals + (f_start - rc.cpb_fullness());
        let sent = sent_rate * seconds;
        assert!(
            (sent - expected_sent).abs() < 400.0,
            "CBR B-GOP conservation broken: sent {sent}, expected {expected_sent}"
        );
        let payload = rc.average_bitrate().unwrap();
        assert!(
            (payload - 1_000_000.0).abs() / 1_000_000.0 < 0.10,
            "CBR B-GOP payload average {payload} too far from target"
        );
        // Steady-state λ-scaled hierarchy: mean QP_B above mean QP_P
        // (non-reference pictures quantise coarser), and neither
        // anchor kind above B.
        let tail = &log[log.len() / 2..];
        let mean = |k: RcFrameKind| {
            let v: Vec<i32> = tail
                .iter()
                .filter(|(kind, _)| *kind == k)
                .map(|&(_, qp)| qp)
                .collect();
            assert!(!v.is_empty());
            v.iter().sum::<i32>() as f64 / v.len() as f64
        };
        let (qi, qp_, qb) = (
            mean(RcFrameKind::Idr),
            mean(RcFrameKind::P),
            mean(RcFrameKind::B),
        );
        assert!(
            qb > qp_ - 0.5 && qb < qp_ + 6.0,
            "B mean QP {qb:.2} not above P mean QP {qp_:.2}"
        );
        assert!(
            qi <= qb + 0.5,
            "IDR mean QP {qi:.2} above B mean QP {qb:.2}"
        );
    }

    #[test]
    fn capped_vbr_b_gop_tracks_average() {
        let cfg = RateControlConfig::capped_vbr(800_000, 1_600_000, 25, 1);
        let rc = RateController::new(cfg, 640, 368);
        let (_, rc, _) = run_b_sequence(rc, 150, 15, 3, [2.5e6, 5.0e5, 2.0e5]);
        let avg = rc.average_bitrate().unwrap();
        let err = (avg - 800_000.0).abs() / 800_000.0;
        assert!(err < 0.05, "capped-VBR B-GOP average {avg} off by {err:.4}");
        assert_eq!(rc.total_filler_bits(), 0, "VBR must not request filler");
    }

    #[test]
    fn b_gets_smaller_target_than_p() {
        let cfg = RateControlConfig::cbr(1_000_000, 30, 1);
        let mut rc = RateController::new(cfg, 640, 368);
        // Seed all kinds with equal complexity so only the kind
        // weighting separates the targets.
        rc.commit_frame(RcFrameKind::Idr, 28, 60_000);
        rc.commit_frame(RcFrameKind::P, 28, 60_000);
        rc.commit_frame(RcFrameKind::B, 28, 60_000);
        let pp = rc.plan_frame(RcFrameKind::P);
        let pb = rc.plan_frame(RcFrameKind::B);
        assert!(
            pb.target_bits < pp.target_bits,
            "B target {} not below P target {}",
            pb.target_bits,
            pp.target_bits
        );
    }

    #[test]
    fn unseeded_b_derives_qp_from_p_plus_offset() {
        let cfg = RateControlConfig::cbr(600_000, 30, 1);
        let mut rc = RateController::new(cfg, 320, 240);
        rc.commit_frame(RcFrameKind::Idr, 24, 90_000);
        rc.commit_frame(RcFrameKind::P, 27, 25_000);
        let plan = rc.plan_frame(RcFrameKind::B);
        assert_eq!(plan.qp, 29, "cold-start B QP should be last P QP + 2");
        // And an unseeded P still derives from IDR as before.
        let mut rc2 = RateController::new(cfg, 320, 240);
        rc2.commit_frame(RcFrameKind::Idr, 24, 90_000);
        let plan_p = rc2.plan_frame(RcFrameKind::P);
        assert_eq!(plan_p.qp, 26, "cold-start P QP should be last IDR QP + 2");
    }

    #[test]
    fn idr_gets_larger_target_than_p() {
        let cfg = RateControlConfig::cbr(1_000_000, 30, 1);
        let mut rc = RateController::new(cfg, 640, 368);
        // Seed both kinds.
        rc.commit_frame(RcFrameKind::Idr, 28, 120_000);
        rc.commit_frame(RcFrameKind::P, 30, 20_000);
        let pi = rc.plan_frame(RcFrameKind::Idr);
        let pp = rc.plan_frame(RcFrameKind::P);
        assert!(pi.target_bits > pp.target_bits);
    }

    #[test]
    fn plan_is_pure_and_commit_advances() {
        let cfg = RateControlConfig::cbr(500_000, 30, 1);
        let mut rc = RateController::new(cfg, 320, 240);
        let a = rc.plan_frame(RcFrameKind::Idr);
        let b = rc.plan_frame(RcFrameKind::Idr);
        assert_eq!(a.qp, b.qp);
        assert_eq!(a.max_bits, b.max_bits);
        rc.commit_frame(RcFrameKind::Idr, a.qp, 40_000);
        assert_eq!(rc.average_bitrate().unwrap(), 40_000.0 * 30.0);
    }

    #[test]
    fn qstep_anchors() {
        assert!((qstep(4) - 1.0).abs() < 1e-12);
        assert!((qstep(10) - 2.0).abs() < 1e-12);
        assert!((qstep(16) - 4.0).abs() < 1e-12);
        // Round-trip.
        for qp in 0..=51 {
            let back = qp_from_qstep(qstep(qp)).round() as i32;
            assert_eq!(back, qp);
        }
    }

    #[test]
    #[should_panic(expected = "max_bitrate must be >= target_bitrate")]
    fn vbr_max_below_target_panics() {
        let cfg = RateControlConfig {
            mode: RateControlMode::CappedVbr,
            target_bitrate: 1_000_000,
            max_bitrate: 500_000,
            vbv_buffer_bits: 1_000_000,
            fps_num: 30,
            fps_den: 1,
            min_qp: 10,
            max_qp: 51,
        };
        let _ = RateController::new(cfg, 320, 240);
    }
}
