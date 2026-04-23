//! Top-level H.264 decoder driver.
//!
//! Holds parameter sets, tracks active SPS/PPS per §7.4.1.2.1, and
//! consumes raw NAL units producing typed events. Does NOT perform
//! reconstruction yet — slice data is passed through as raw bytes for
//! a future layer to interpret.
//!
//! **Parameter-set activation (§7.4.1.2.1)**:
//! * An SPS is "active" once a PPS that references it is itself
//!   activated. At SPS activation, its syntax elements govern decoding
//!   for the coded video sequence.
//! * A PPS is "active" once a coded slice NAL (or slice-data partition
//!   A / end-of-stream context referring to it) references its
//!   `pic_parameter_set_id`.
//!
//! Concretely: when we see a slice NAL, we look up its
//! `pic_parameter_set_id`; that PPS becomes active, and the SPS it
//! references (via `seq_parameter_set_id`) becomes active at the same
//! time. SPS and PPS parse is stateless — the decoder merely indexes
//! them by id.

#![allow(dead_code)]

use crate::nal::{self, AnnexBSplitter, AvccSplitter, NalHeader, NalUnitType};
use crate::non_vcl::{self, AccessUnitDelimiter, PrimaryPicType, SeiMessage};
use crate::pps::{Pps, PpsError};
use crate::slice_header::{SliceHeader, SliceHeaderError};
use crate::sps::{Sps, SpsError};

#[derive(Debug, thiserror::Error)]
pub enum DecoderError {
    #[error("NAL parse: {0}")]
    Nal(#[from] nal::NalError),
    #[error("SPS parse: {0}")]
    Sps(#[from] SpsError),
    #[error("PPS parse: {0}")]
    Pps(#[from] PpsError),
    #[error("slice header parse: {0}")]
    SliceHeader(#[from] SliceHeaderError),
    #[error("non-VCL parse: {0}")]
    NonVcl(#[from] non_vcl::NonVclError),
    /// §7.4.1.2.1 — slice references a PPS id we have not stored.
    #[error("slice references unknown PPS id {0}")]
    UnknownPps(u32),
    /// §7.4.1.2.1 — the PPS being activated references an SPS id we have
    /// not stored.
    #[error("active PPS references unknown SPS id {0}")]
    UnknownSps(u32),
    /// §7.4.1.2.1 — a slice NAL arrived before any PPS activation was
    /// possible (no PPS stored at all).
    #[error("slice NAL received before any SPS/PPS activation")]
    NoActiveParameterSets,
}

pub type DecoderResult<T> = Result<T, DecoderError>;

/// Events emitted as the decoder consumes NAL units.
///
/// The decoder is a pass-through for everything below the slice header;
/// slice_data parsing is deferred to a future layer.
#[derive(Debug, Clone)]
pub enum Event {
    /// SPS parsed and stored. Payload is `seq_parameter_set_id`.
    SpsStored(u32),
    /// PPS parsed and stored. Payload is `pic_parameter_set_id`.
    PpsStored(u32),
    /// §7.3.2.4 — Access Unit Delimiter.
    AccessUnitDelimiter(PrimaryPicType),
    /// §7.3.3 — slice header parsed; slice_data is still raw bytes.
    Slice {
        /// Raw 5-bit `nal_unit_type` value.
        nal_unit_type: u8,
        /// 2-bit `nal_ref_idc` value.
        nal_ref_idc: u8,
        header: SliceHeader,
        /// The full RBSP bytes after the NAL header byte is stripped
        /// and emulation-prevention bytes removed. Together with
        /// `slice_data_cursor` the caller can pick up at the first bit
        /// of slice_data() without re-parsing the slice_header.
        rbsp: Vec<u8>,
        /// `(byte_index, bit_index)` into `rbsp` marking the first bit
        /// of slice_data(). Feeds `crate::slice_data::parse_slice_data`.
        slice_data_cursor: (usize, u8),
        /// §7.4.1.2.1 — the PPS activated by this slice, captured at
        /// slice-header parse time. Snapshotting here avoids a "latest
        /// PPS wins" race when PPSs with the same id are re-transmitted
        /// across an access unit boundary (e.g. JVT CACQP3 repeatedly
        /// re-sends PPS id 0 with a different `chroma_qp_index_offset`).
        /// Consumers should prefer this snapshot over
        /// [`Decoder::active_pps`] when reconstructing the slice.
        pps: Pps,
        /// §7.4.1.2.1 — the SPS referenced by the activated PPS, captured
        /// at slice-header parse time. Mirrors the snapshot rationale for
        /// `pps` above (though SPS re-transmission is rarer in practice).
        sps: Sps,
    },
    /// §7.3.2.3 — parsed SEI messages. May be empty when the SEI RBSP
    /// carries only rbsp_trailing_bits.
    Sei(Vec<SeiMessage>),
    /// §7.3.2.5 — end of sequence.
    EndOfSequence,
    /// §7.3.2.6 — end of stream.
    EndOfStream,
    /// §7.3.2.7 — filler data (the decoder drops the payload).
    FillerData,
    /// NAL types the decoder doesn't process (reserved / extension —
    /// §7.3.1 values 13..=16, 19..=21, 22..=23, and all unspecified
    /// ranges 0 / 24..=31). Caller may inspect the raw NAL bytes
    /// (including the header byte).
    Ignored {
        nal_unit_type: u8,
        nal_bytes: Vec<u8>,
    },
}

/// Top-level H.264 decoder driver. See the module docs for activation
/// rules (§7.4.1.2.1) and scope.
pub struct Decoder {
    /// §7.4.1.2.1 — SPS database indexed by `seq_parameter_set_id`
    /// (0..=31).
    sps_by_id: Vec<Option<Sps>>,
    /// §7.4.1.2.1 — PPS database indexed by `pic_parameter_set_id`
    /// (0..=255).
    pps_by_id: Vec<Option<Pps>>,
    /// §7.4.1.2.1 — currently active SPS id, set whenever a slice NAL
    /// activates a PPS whose `seq_parameter_set_id` points at it.
    active_sps_id: Option<u32>,
    /// §7.4.1.2.1 — currently active PPS id, set whenever a slice NAL
    /// references it.
    active_pps_id: Option<u32>,
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            sps_by_id: (0..32).map(|_| None).collect(),
            pps_by_id: (0..256).map(|_| None).collect(),
            active_sps_id: None,
            active_pps_id: None,
        }
    }

    /// Process one raw NAL unit — the bytes starting with the NAL header
    /// byte, with *no* Annex B start code prefix and no AVCC length
    /// prefix. Emulation-prevention stripping is handled internally via
    /// [`nal::parse_nal_unit`].
    ///
    /// Returns a single [`Event`] describing what happened.
    pub fn process_nal(&mut self, nal_bytes: &[u8]) -> DecoderResult<Event> {
        // §7.3.1 — parse the 1-byte NAL header; this also strips
        // emulation_prevention_three_byte per §7.4.1.1.
        let nu = nal::parse_nal_unit(nal_bytes)?;
        let rbsp: Vec<u8> = nu.rbsp.into_owned();
        let header = nu.header;

        match header.nal_unit_type {
            // §7.3.2.1 / §7.4.1.2.1 — store SPS.
            NalUnitType::Sps => {
                let sps = Sps::parse(&rbsp)?;
                let id = sps.seq_parameter_set_id;
                // `id <= 31` is validated by Sps::parse; the table slot
                // is therefore always in-bounds.
                self.sps_by_id[id as usize] = Some(sps);
                Ok(Event::SpsStored(id))
            }
            // §7.3.2.2 / §7.4.1.2.1 — store PPS. If the referenced SPS
            // is already stored, use its `chroma_format_idc` so the PPS
            // scaling-matrix tail (if any) is parsed with the correct
            // loop length. Otherwise fall back to 4:2:0 (the common
            // case); this matters only for 4:4:4 streams with a
            // scaling matrix in the PPS.
            NalUnitType::Pps => {
                // Peek the seq_parameter_set_id without committing: parse
                // with the default, then if the referenced SPS is
                // present and has a non-default chroma_format_idc,
                // reparse with the right value.
                let pps = Pps::parse(&rbsp)?;
                let sps_id = pps.seq_parameter_set_id;
                let pps = if let Some(Some(sps)) = self.sps_by_id.get(sps_id as usize) {
                    if sps.chroma_format_idc != 1 {
                        Pps::parse_with_chroma_format(&rbsp, sps.chroma_format_idc)?
                    } else {
                        pps
                    }
                } else {
                    pps
                };
                let id = pps.pic_parameter_set_id;
                self.pps_by_id[id as usize] = Some(pps);
                Ok(Event::PpsStored(id))
            }
            // §7.3.2.3 — SEI envelope.
            NalUnitType::Sei => {
                let msgs = non_vcl::parse_sei_rbsp(&rbsp)?;
                Ok(Event::Sei(msgs))
            }
            // §7.3.2.4 — access unit delimiter.
            NalUnitType::AccessUnitDelimiter => {
                let aud = AccessUnitDelimiter::parse(&rbsp)?;
                Ok(Event::AccessUnitDelimiter(aud.primary_pic_type))
            }
            // §7.3.2.5 — end of sequence.
            NalUnitType::EndOfSequence => Ok(Event::EndOfSequence),
            // §7.3.2.6 — end of stream.
            NalUnitType::EndOfStream => Ok(Event::EndOfStream),
            // §7.3.2.7 — filler.
            NalUnitType::FillerData => {
                // Parse just to validate the payload shape; discard the
                // count.
                let _ = non_vcl::parse_filler_data(&rbsp)?;
                Ok(Event::FillerData)
            }
            // §7.3.2.8 — slice of a non-IDR picture.
            // §7.3.2.9 — slice of an IDR picture.
            NalUnitType::SliceNonIdr | NalUnitType::SliceIdr => {
                self.process_slice_nal(&header, &rbsp)
            }
            // Data-partition NAL types are a legacy Baseline/Extended
            // feature we don't model here. Treat them as "ignored" with
            // the raw NAL bytes so callers can route them elsewhere.
            NalUnitType::SliceDataPartitionA
            | NalUnitType::SliceDataPartitionB
            | NalUnitType::SliceDataPartitionC
            | NalUnitType::SliceAuxiliary => Ok(Event::Ignored {
                nal_unit_type: header.nal_unit_type.as_u8(),
                nal_bytes: nal_bytes.to_vec(),
            }),
            // SPS extension (13), prefix NAL (14), subset SPS (15),
            // depth parameter set (16), reserved (17/18/22/23), slice
            // extension (20/21), and all unspecified values. Pass
            // through as Ignored so the caller can inspect/log.
            NalUnitType::Unspecified(_)
            | NalUnitType::Reserved(_)
            | NalUnitType::SpsExtension
            | NalUnitType::PrefixNalUnit
            | NalUnitType::SubsetSps
            | NalUnitType::DepthParameterSet
            | NalUnitType::SliceExtension
            | NalUnitType::SliceExtensionDepth
            | NalUnitType::UnspecifiedRange(_) => Ok(Event::Ignored {
                nal_unit_type: header.nal_unit_type.as_u8(),
                nal_bytes: nal_bytes.to_vec(),
            }),
        }
    }

    /// §7.4.1.2.1 — slice NAL: resolve PPS, activate PPS+SPS, then
    /// parse the slice header.
    fn process_slice_nal(&mut self, header: &NalHeader, rbsp: &[u8]) -> DecoderResult<Event> {
        // The spec requires a PPS to be activated *before* slice data
        // can be decoded. If nothing has been stored yet, report a
        // clear setup error.
        if self.pps_by_id.iter().all(|p| p.is_none()) {
            return Err(DecoderError::NoActiveParameterSets);
        }
        // Peek the first ue(v) pair (first_mb_in_slice, slice_type) and
        // the third (pic_parameter_set_id) to find which PPS this slice
        // activates, per §7.3.3.
        let pps_id = peek_slice_pps_id(rbsp)?;
        let pps = self
            .pps_by_id
            .get(pps_id as usize)
            .and_then(|p| p.as_ref())
            .ok_or(DecoderError::UnknownPps(pps_id))?
            .clone();
        let sps_id = pps.seq_parameter_set_id;
        let sps = self
            .sps_by_id
            .get(sps_id as usize)
            .and_then(|p| p.as_ref())
            .ok_or(DecoderError::UnknownSps(sps_id))?
            .clone();

        // §7.3.3 — parse the slice header using the newly-activated SPS
        // and PPS.
        let (parsed, slice_data_cursor) = SliceHeader::parse_and_tell(rbsp, &sps, &pps, header)?;

        // §7.4.1.2.1 — commit activation only after a successful parse
        // so malformed slices can't leave the decoder in a half-activated
        // state.
        self.active_pps_id = Some(pps_id);
        self.active_sps_id = Some(sps_id);

        Ok(Event::Slice {
            nal_unit_type: header.nal_unit_type.as_u8(),
            nal_ref_idc: header.nal_ref_idc,
            header: parsed,
            rbsp: rbsp.to_vec(),
            slice_data_cursor,
            pps,
            sps,
        })
    }

    /// §B.1 — consume an Annex B byte stream. Each iteration yields one
    /// [`Event`] per NAL unit (or a [`DecoderError`]). The iterator
    /// borrows the decoder mutably, so events must be consumed before
    /// the next call.
    pub fn process_annex_b<'a>(&'a mut self, stream: &'a [u8]) -> AnnexBEventIter<'a> {
        AnnexBEventIter {
            decoder: self,
            splitter: AnnexBSplitter::new(stream),
        }
    }

    /// AVCC-style length-prefixed framing (MP4 `avc1` / MKV
    /// `V_MPEG4/ISO/AVC`). `nalu_length_size` is
    /// `lengthSizeMinusOne + 1` from the configuration record (1, 2, or
    /// 4).
    pub fn process_avcc<'a>(
        &'a mut self,
        stream: &'a [u8],
        nalu_length_size: u8,
    ) -> DecoderResult<AvccEventIter<'a>> {
        let splitter = AvccSplitter::new(stream, nalu_length_size)?;
        Ok(AvccEventIter {
            decoder: self,
            splitter,
        })
    }

    /// §7.4.1.2.1 — the currently active SPS, or `None` before any
    /// slice has been processed.
    pub fn active_sps(&self) -> Option<&Sps> {
        self.active_sps_id
            .and_then(|id| self.sps_by_id.get(id as usize))
            .and_then(|p| p.as_ref())
    }

    /// §7.4.1.2.1 — the currently active PPS, or `None` before any
    /// slice has been processed.
    pub fn active_pps(&self) -> Option<&Pps> {
        self.active_pps_id
            .and_then(|id| self.pps_by_id.get(id as usize))
            .and_then(|p| p.as_ref())
    }

    /// Look up a stored SPS by `seq_parameter_set_id`.
    pub fn sps(&self, id: u32) -> Option<&Sps> {
        self.sps_by_id.get(id as usize).and_then(|p| p.as_ref())
    }

    /// Look up a stored PPS by `pic_parameter_set_id`.
    pub fn pps(&self, id: u32) -> Option<&Pps> {
        self.pps_by_id.get(id as usize).and_then(|p| p.as_ref())
    }

    /// Currently active SPS id, if a slice has been processed.
    pub fn active_sps_id(&self) -> Option<u32> {
        self.active_sps_id
    }

    /// Currently active PPS id, if a slice has been processed.
    pub fn active_pps_id(&self) -> Option<u32> {
        self.active_pps_id
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

/// §7.3.3 — peek the `pic_parameter_set_id` from the front of a slice
/// RBSP without owning a `BitReader` past the call. Slice headers start
/// with `ue(v) first_mb_in_slice`, `ue(v) slice_type`, `ue(v)
/// pic_parameter_set_id`, so we just skip the first two ue(v) codes
/// and decode the third.
fn peek_slice_pps_id(rbsp: &[u8]) -> Result<u32, SliceHeaderError> {
    use crate::bitstream::BitReader;
    let mut r = BitReader::new(rbsp);
    let _first_mb = r.ue()?;
    let _slice_type = r.ue()?;
    let pps_id = r.ue()?;
    if pps_id > 255 {
        return Err(SliceHeaderError::PpsIdOutOfRange(pps_id));
    }
    Ok(pps_id)
}

/// Iterator over Annex B NAL events, yielded one per NAL unit.
pub struct AnnexBEventIter<'a> {
    decoder: &'a mut Decoder,
    splitter: AnnexBSplitter<'a>,
}

impl<'a> Iterator for AnnexBEventIter<'a> {
    type Item = DecoderResult<Event>;

    fn next(&mut self) -> Option<Self::Item> {
        let nal = self.splitter.next()?;
        Some(self.decoder.process_nal(nal))
    }
}

/// Iterator over AVCC-framed NAL events.
pub struct AvccEventIter<'a> {
    decoder: &'a mut Decoder,
    splitter: AvccSplitter<'a>,
}

impl<'a> Iterator for AvccEventIter<'a> {
    type Item = DecoderResult<Event>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.splitter.next()?;
        let nal = match item {
            Ok(b) => b,
            Err(e) => return Some(Err(DecoderError::Nal(e))),
        };
        Some(self.decoder.process_nal(nal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- tiny MSB-first bit writer, copied from sibling modules ----

    struct BitWriter {
        bytes: Vec<u8>,
        bit_pos: u8,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                bit_pos: 0,
            }
        }

        fn u(&mut self, bits: u32, value: u32) {
            for i in (0..bits).rev() {
                let bit = ((value >> i) & 1) as u8;
                if self.bit_pos == 0 {
                    self.bytes.push(0);
                }
                let idx = self.bytes.len() - 1;
                self.bytes[idx] |= bit << (7 - self.bit_pos);
                self.bit_pos = (self.bit_pos + 1) % 8;
            }
        }

        fn ue(&mut self, value: u32) {
            let v = value + 1;
            let leading = 31 - v.leading_zeros();
            for _ in 0..leading {
                self.u(1, 0);
            }
            self.u(leading + 1, v);
        }

        fn se(&mut self, value: i32) {
            let k = if value <= 0 {
                (-2 * value) as u32
            } else {
                (2 * value - 1) as u32
            };
            self.ue(k);
        }

        fn trailing(&mut self) {
            self.u(1, 1);
            while self.bit_pos != 0 {
                self.u(1, 0);
            }
        }

        fn into_bytes(self) -> Vec<u8> {
            self.bytes
        }
    }

    // ---- fixture builders ----

    /// §7.3.2.1 — minimal Baseline SPS (profile_idc=66), 320x240, POC
    /// type 0 (frame_num 4 bits, pic_order_cnt_lsb 4 bits).
    fn build_minimal_sps_rbsp(sps_id: u32) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.u(8, 66); // profile_idc
        w.u(8, 0); // constraint_sets + reserved_zero_2bits
        w.u(8, 30); // level_idc
        w.ue(sps_id); // seq_parameter_set_id
        w.ue(0); // log2_max_frame_num_minus4
        w.ue(0); // pic_order_cnt_type
        w.ue(0); // log2_max_pic_order_cnt_lsb_minus4
        w.ue(1); // max_num_ref_frames
        w.u(1, 0); // gaps_in_frame_num_value_allowed_flag
        w.ue(19); // pic_width_in_mbs_minus1 — 320px / 16
        w.ue(14); // pic_height_in_map_units_minus1 — 240px / 16
        w.u(1, 1); // frame_mbs_only_flag
        w.u(1, 1); // direct_8x8_inference_flag
        w.u(1, 0); // frame_cropping_flag
        w.u(1, 0); // vui_parameters_present_flag
        w.trailing();
        w.into_bytes()
    }

    /// §7.3.2.2 — minimal CAVLC PPS referencing the given sps_id.
    fn build_minimal_pps_rbsp(pps_id: u32, sps_id: u32) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.ue(pps_id); // pic_parameter_set_id
        w.ue(sps_id); // seq_parameter_set_id
        w.u(1, 0); // entropy_coding_mode_flag = CAVLC
        w.u(1, 0); // bottom_field_pic_order_in_frame_present_flag
        w.ue(0); // num_slice_groups_minus1
        w.ue(0); // num_ref_idx_l0_default_active_minus1
        w.ue(0); // num_ref_idx_l1_default_active_minus1
        w.u(1, 0); // weighted_pred_flag
        w.u(2, 0); // weighted_bipred_idc
        w.se(0); // pic_init_qp_minus26
        w.se(0); // pic_init_qs_minus26
        w.se(0); // chroma_qp_index_offset
        w.u(1, 0); // deblocking_filter_control_present_flag
        w.u(1, 0); // constrained_intra_pred_flag
        w.u(1, 0); // redundant_pic_cnt_present_flag
        w.trailing();
        w.into_bytes()
    }

    /// §7.3.3 — minimal I-slice RBSP body matching the SPS/PPS built
    /// above. `pps_id` must match a stored PPS.
    fn build_minimal_i_slice_rbsp(pps_id: u32) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.ue(0); // first_mb_in_slice
        w.ue(7); // slice_type = 7 (I, all slices same type)
        w.ue(pps_id); // pic_parameter_set_id
        w.u(4, 0); // frame_num (4 bits)
        w.u(4, 0); // pic_order_cnt_lsb (4 bits)
                   // nal_ref_idc != 0 → dec_ref_pic_marking(): sliding window
        w.u(1, 0);
        w.se(0); // slice_qp_delta
                 // No rbsp_trailing_bits here — the slice_data follows in a real
                 // stream; for tests we just let the parser read exactly what it
                 // needs. Append a padding byte so the reader never runs out.
        w.u(8, 0x80);
        w.into_bytes()
    }

    /// Build a full NAL unit (header byte + RBSP) for a given type and
    /// ref_idc. The RBSP must already be de-emulated.
    fn build_nal(nal_unit_type: u8, nal_ref_idc: u8, rbsp: &[u8]) -> Vec<u8> {
        let header = (nal_ref_idc & 0x03) << 5 | (nal_unit_type & 0x1F);
        let mut out = Vec::with_capacity(rbsp.len() + 1);
        out.push(header);
        out.extend_from_slice(rbsp);
        out
    }

    // ---- tests ----

    #[test]
    fn sps_then_pps_are_stored_in_slots() {
        let mut dec = Decoder::new();
        let sps_nal = build_nal(7, 3, &build_minimal_sps_rbsp(0));
        let pps_nal = build_nal(8, 3, &build_minimal_pps_rbsp(0, 0));

        let e1 = dec.process_nal(&sps_nal).unwrap();
        assert!(matches!(e1, Event::SpsStored(0)));
        assert!(dec.sps(0).is_some());

        let e2 = dec.process_nal(&pps_nal).unwrap();
        assert!(matches!(e2, Event::PpsStored(0)));
        assert!(dec.pps(0).is_some());

        // Neither of these touches activation state.
        assert!(dec.active_sps().is_none());
        assert!(dec.active_pps().is_none());
    }

    #[test]
    fn active_params_stay_none_until_slice() {
        // §7.4.1.2.1 — SPS and PPS storage alone don't trigger
        // activation; only a slice reference does.
        let mut dec = Decoder::new();
        let sps_nal = build_nal(7, 3, &build_minimal_sps_rbsp(0));
        let pps_nal = build_nal(8, 3, &build_minimal_pps_rbsp(0, 0));
        // AUD payload: primary_pic_type=0 (000) + stop_bit + pad → 0x10.
        let aud_nal = build_nal(9, 0, &[0x10]);

        let mut stream = Vec::new();
        stream.extend_from_slice(&[0, 0, 0, 1]);
        stream.extend_from_slice(&sps_nal);
        stream.extend_from_slice(&[0, 0, 0, 1]);
        stream.extend_from_slice(&pps_nal);
        stream.extend_from_slice(&[0, 0, 0, 1]);
        stream.extend_from_slice(&aud_nal);

        let events: Vec<_> = dec
            .process_annex_b(&stream)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], Event::SpsStored(0)));
        assert!(matches!(events[1], Event::PpsStored(0)));
        assert!(matches!(
            events[2],
            Event::AccessUnitDelimiter(PrimaryPicType(0))
        ));
        assert!(dec.active_sps_id().is_none());
        assert!(dec.active_pps_id().is_none());
    }

    #[test]
    fn slice_activates_sps_and_pps() {
        let mut dec = Decoder::new();
        let _ = dec
            .process_nal(&build_nal(7, 3, &build_minimal_sps_rbsp(0)))
            .unwrap();
        let _ = dec
            .process_nal(&build_nal(8, 3, &build_minimal_pps_rbsp(0, 0)))
            .unwrap();
        // Non-IDR I-slice referencing PPS id 0.
        let slice_nal = build_nal(1, 3, &build_minimal_i_slice_rbsp(0));
        let ev = dec.process_nal(&slice_nal).unwrap();
        match ev {
            Event::Slice {
                header,
                nal_unit_type,
                ..
            } => {
                assert_eq!(nal_unit_type, 1);
                assert_eq!(header.pic_parameter_set_id, 0);
            }
            other => panic!("expected Slice, got {:?}", other),
        }
        assert_eq!(dec.active_sps_id(), Some(0));
        assert_eq!(dec.active_pps_id(), Some(0));
        assert!(dec.active_sps().is_some());
        assert!(dec.active_pps().is_some());
    }

    #[test]
    fn slice_before_any_pps_is_rejected() {
        let mut dec = Decoder::new();
        let slice_nal = build_nal(1, 3, &build_minimal_i_slice_rbsp(0));
        let err = dec.process_nal(&slice_nal).unwrap_err();
        assert!(matches!(err, DecoderError::NoActiveParameterSets));
    }

    #[test]
    fn slice_referencing_unknown_pps_is_rejected() {
        // Store only PPS 0, but the slice references PPS 5.
        let mut dec = Decoder::new();
        let _ = dec
            .process_nal(&build_nal(7, 3, &build_minimal_sps_rbsp(0)))
            .unwrap();
        let _ = dec
            .process_nal(&build_nal(8, 3, &build_minimal_pps_rbsp(0, 0)))
            .unwrap();
        let slice_nal = build_nal(1, 3, &build_minimal_i_slice_rbsp(5));
        let err = dec.process_nal(&slice_nal).unwrap_err();
        assert!(matches!(err, DecoderError::UnknownPps(5)));
    }

    #[test]
    fn slice_referencing_pps_with_unknown_sps_is_rejected() {
        // PPS 0 points to SPS id 4, which we never stored.
        let mut dec = Decoder::new();
        let _ = dec
            .process_nal(&build_nal(8, 3, &build_minimal_pps_rbsp(0, 4)))
            .unwrap();
        let slice_nal = build_nal(1, 3, &build_minimal_i_slice_rbsp(0));
        let err = dec.process_nal(&slice_nal).unwrap_err();
        assert!(matches!(err, DecoderError::UnknownSps(4)));
    }

    #[test]
    fn aud_event_is_emitted() {
        // primary_pic_type=5 (101) → bits 1 0 1 1 0 0 0 0 = 0xB0.
        let aud_nal = build_nal(9, 0, &[0xB0]);
        let mut dec = Decoder::new();
        let ev = dec.process_nal(&aud_nal).unwrap();
        assert!(matches!(ev, Event::AccessUnitDelimiter(PrimaryPicType(5))));
    }

    #[test]
    fn sei_event_is_emitted() {
        // One SEI message: payload_type=1, payload_size=2, payload=[0xAA,0xBB].
        let rbsp = vec![0x01, 0x02, 0xAA, 0xBB, 0x80];
        let sei_nal = build_nal(6, 0, &rbsp);
        let mut dec = Decoder::new();
        let ev = dec.process_nal(&sei_nal).unwrap();
        match ev {
            Event::Sei(msgs) => {
                assert_eq!(msgs.len(), 1);
                assert_eq!(msgs[0].payload_type, 1);
                assert_eq!(msgs[0].payload, vec![0xAA, 0xBB]);
            }
            other => panic!("expected Sei, got {:?}", other),
        }
    }

    #[test]
    fn end_of_sequence_and_end_of_stream_events() {
        let mut dec = Decoder::new();
        let ev = dec.process_nal(&build_nal(10, 0, &[])).unwrap();
        assert!(matches!(ev, Event::EndOfSequence));
        let ev = dec.process_nal(&build_nal(11, 0, &[])).unwrap();
        assert!(matches!(ev, Event::EndOfStream));
    }

    #[test]
    fn filler_event_is_emitted() {
        // FF FF FF | rbsp_trailing_bits (0x80).
        let filler_nal = build_nal(12, 0, &[0xFF, 0xFF, 0xFF, 0x80]);
        let mut dec = Decoder::new();
        let ev = dec.process_nal(&filler_nal).unwrap();
        assert!(matches!(ev, Event::FillerData));
    }

    #[test]
    fn annex_b_iterator_yields_three_events() {
        let sps_nal = build_nal(7, 3, &build_minimal_sps_rbsp(0));
        let pps_nal = build_nal(8, 3, &build_minimal_pps_rbsp(0, 0));
        let aud_nal = build_nal(9, 0, &[0x10]);

        let mut stream = Vec::new();
        stream.extend_from_slice(&[0, 0, 0, 1]);
        stream.extend_from_slice(&sps_nal);
        stream.extend_from_slice(&[0, 0, 0, 1]);
        stream.extend_from_slice(&pps_nal);
        stream.extend_from_slice(&[0, 0, 0, 1]);
        stream.extend_from_slice(&aud_nal);

        let mut dec = Decoder::new();
        let evs: Vec<_> = dec
            .process_annex_b(&stream)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(evs.len(), 3);
        assert!(matches!(evs[0], Event::SpsStored(0)));
        assert!(matches!(evs[1], Event::PpsStored(0)));
        assert!(matches!(evs[2], Event::AccessUnitDelimiter(_)));
    }

    #[test]
    fn prefix_nal_14_is_ignored() {
        // Prefix NAL unit (SVC/MVC) — type 14. We don't parse these.
        let nal = build_nal(14, 0, &[0xAB, 0xCD]);
        let mut dec = Decoder::new();
        let ev = dec.process_nal(&nal).unwrap();
        match ev {
            Event::Ignored {
                nal_unit_type,
                nal_bytes,
            } => {
                assert_eq!(nal_unit_type, 14);
                assert_eq!(nal_bytes, nal);
            }
            other => panic!("expected Ignored, got {:?}", other),
        }
    }

    #[test]
    fn avcc_iterator_yields_events() {
        // AVCC with 4-byte length prefix: SPS then PPS.
        let sps_nal = build_nal(7, 3, &build_minimal_sps_rbsp(0));
        let pps_nal = build_nal(8, 3, &build_minimal_pps_rbsp(0, 0));
        let mut stream = Vec::new();
        stream.extend_from_slice(&(sps_nal.len() as u32).to_be_bytes());
        stream.extend_from_slice(&sps_nal);
        stream.extend_from_slice(&(pps_nal.len() as u32).to_be_bytes());
        stream.extend_from_slice(&pps_nal);

        let mut dec = Decoder::new();
        let it = dec.process_avcc(&stream, 4).unwrap();
        let evs: Vec<_> = it.collect::<Result<_, _>>().unwrap();
        assert_eq!(evs.len(), 2);
        assert!(matches!(evs[0], Event::SpsStored(0)));
        assert!(matches!(evs[1], Event::PpsStored(0)));
    }

    #[test]
    fn peek_slice_pps_id_matches_body() {
        // Build a slice RBSP with pps_id = 7 and verify peek.
        let mut w = BitWriter::new();
        w.ue(0); // first_mb_in_slice
        w.ue(7); // slice_type
        w.ue(7); // pic_parameter_set_id = 7
        w.u(8, 0); // pad
        let rbsp = w.into_bytes();
        assert_eq!(peek_slice_pps_id(&rbsp).unwrap(), 7);
    }
}
