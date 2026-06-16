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
use crate::sps_extension::{SeqParameterSetExtension, SpsExtensionError};
use crate::subset_sps::{SubsetSps, SubsetSpsError};

#[derive(Debug, thiserror::Error)]
pub enum DecoderError {
    #[error("NAL parse: {0}")]
    Nal(#[from] nal::NalError),
    #[error("SPS parse: {0}")]
    Sps(#[from] SpsError),
    #[error("subset SPS parse: {0}")]
    SubsetSps(#[from] SubsetSpsError),
    #[error("SPS extension parse: {0}")]
    SpsExtension(#[from] SpsExtensionError),
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
    /// §A.2 — Flexible Macroblock Ordering (FMO,
    /// `num_slice_groups_minus1 > 0`) is constrained to the Baseline
    /// (§A.2.1, profile_idc=66) and Extended (§A.2.3, profile_idc=88)
    /// profiles. Every other profile (Main, High, High 10, High 4:2:2,
    /// High 4:4:4, etc.) constrains `num_slice_groups_minus1` to 0
    /// (e.g. §A.2.2: "Picture parameter sets shall have
    /// num_slice_groups_minus1 equal to 0 only."). The §8.2.2
    /// MbToSliceGroupMap derivation IS implemented in `mb_address.rs`,
    /// but the §8.4 reconstruction path walks macroblock addresses in
    /// raster order (slice-group 0 only) — so an FMO PPS with multiple
    /// groups would silently mis-decode. Reject at slice activation
    /// time so we match the common-decoder convention of "FMO is not implemented"
    /// rejection rather than emitting a wrong-pixels frame for streams
    /// the reference decoder refuses.
    #[error("FMO (num_slice_groups_minus1={0} > 0) is not supported in this decoder build")]
    FmoNotSupported(u32),
    /// §G.7.4.1.2.1 — a coded slice MVC extension NAL (type 20)
    /// activated a PPS whose referenced `seq_parameter_set_id` has no
    /// subset SPS (NAL 15) stored. Per the activation rule a type-20
    /// slice's PPS must reference a *subset* SPS, not an ordinary one.
    #[error("MVC slice extension PPS references unknown subset SPS id {0}")]
    UnknownSubsetSps(u32),
    /// §G.7.3.2.13 — the coded slice extension NAL (type 20/21) carries
    /// an SVC (`svc_extension_flag == 1`) or 3D-AVC
    /// (`avc_3d_extension_flag == 1`) body. Only the MVC branch
    /// (`slice_header()` + `slice_data()`) is parsed; the Annex F SVC
    /// and Annex J 3D-AVC slice bodies are not modelled yet.
    #[error("coded slice extension body is not MVC (svc_extension_flag/avc_3d_extension_flag set); not supported")]
    SliceExtensionBodyNotMvc,
}

pub type DecoderResult<T> = Result<T, DecoderError>;

/// Events emitted as the decoder consumes NAL units.
///
/// The decoder is a pass-through for everything below the slice header;
/// slice_data parsing is deferred to a future layer.
// Boxing `header`/`sps` would force every consumer to deref through
// `Box<…>` when destructuring `Event::Slice { header, sps, .. }`. The
// enum is public and pattern-matched everywhere, so we accept the size
// asymmetry here rather than break the external API.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum Event {
    /// SPS parsed and stored. Payload is `seq_parameter_set_id`.
    SpsStored(u32),
    /// §7.3.2.1.3 — subset SPS (NAL unit type 15) parsed and stored.
    /// Payload is the embedded `seq_parameter_set_id`. Per §7.4.1.2.1
    /// subset SPSs live in their own id space, separate from ordinary
    /// SPSs; retrieve via [`Decoder::subset_sps`].
    SubsetSpsStored(u32),
    /// §7.3.2.1.2 — sequence parameter set extension (NAL unit type 13)
    /// parsed and stored. Payload is the `seq_parameter_set_id` of the
    /// ordinary SPS this extension supplements. Carries the auxiliary
    /// coded picture / alpha-blending parameters; retrieve the parsed
    /// body via [`Decoder::sps_extension`].
    SpsExtensionStored(u32),
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
    /// §G.7.3.2.13 — coded slice MVC extension (`nal_unit_type == 20`,
    /// `svc_extension_flag == 0`). The `slice_layer_extension_rbsp()`
    /// MVC branch carries the same §7.3.3 `slice_header()` as a
    /// base-view slice, parsed here against the base SPS embedded in the
    /// subset SPS (§7.3.2.1.3) that the activated PPS references (per the
    /// §G.7.4.1.2.1 activation rule). The §G.7.3.1.1 NAL-unit-header MVC
    /// extension fields are surfaced so the caller can route the slice
    /// to the correct view component.
    SliceExtension {
        /// Raw 5-bit `nal_unit_type` value (always 20 here).
        nal_unit_type: u8,
        /// 2-bit `nal_ref_idc` value.
        nal_ref_idc: u8,
        /// §G.7.3.1.1 NAL-unit-header MVC extension (view_id,
        /// temporal_id, anchor_pic_flag, inter_view_flag, …).
        mvc: nal::NalUnitHeaderMvcExtension,
        /// IdrPicFlag derived from `!mvc.non_idr_flag` (§G.7.4.1.1).
        idr_pic_flag: bool,
        header: SliceHeader,
        /// RBSP body after the NAL header byte + 3 MVC extension bytes
        /// are stripped and emulation-prevention bytes removed.
        rbsp: Vec<u8>,
        /// `(byte_index, bit_index)` into `rbsp` marking the first bit
        /// of slice_data().
        slice_data_cursor: (usize, u8),
        /// §7.4.1.2.1 — the PPS activated by this MVC slice.
        pps: Pps,
        /// §G.7.4.1.2.1 — the subset SPS (NAL 15) the activated PPS
        /// references. The view-coding parameters used to parse the
        /// slice header come from its embedded base `sps`.
        subset_sps: SubsetSps,
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
    /// §7.4.1.2.1 — subset SPS database indexed by
    /// `seq_parameter_set_id` (0..=31). A value space separate from
    /// `sps_by_id`: only coded slice extension NAL units (types 20/21)
    /// activate these.
    subset_sps_by_id: Vec<Option<SubsetSps>>,
    /// §7.3.2.1.2 — sequence parameter set extension database indexed by
    /// `seq_parameter_set_id` (0..=31). An SPS extension (NAL 13) shares
    /// the ordinary SPS id space — it supplements the SPS of the same id
    /// with the auxiliary-coded-picture parameters.
    sps_extension_by_id: Vec<Option<SeqParameterSetExtension>>,
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
            subset_sps_by_id: (0..32).map(|_| None).collect(),
            sps_extension_by_id: (0..32).map(|_| None).collect(),
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
            // §7.3.2.1.3 / §7.4.1.2.1 — store subset SPS (separate id
            // space from ordinary SPSs).
            NalUnitType::SubsetSps => {
                let subset = SubsetSps::parse(&rbsp)?;
                let id = subset.sps.seq_parameter_set_id;
                // `id <= 31` is validated inside
                // `Sps::parse_seq_parameter_set_data`; the table slot is
                // therefore always in-bounds.
                self.subset_sps_by_id[id as usize] = Some(subset);
                Ok(Event::SubsetSpsStored(id))
            }
            // §7.3.2.2 / §7.4.1.2.1 — store PPS. The PPS scaling-matrix
            // tail loop length depends on the referenced SPS's
            // `chroma_format_idc` (4:4:4 carries 6 8x8 lists; everything
            // else carries 2 when transform_8x8 is on), so the SPS that
            // the PPS references MUST already be stored before we can
            // parse the PPS correctly. common H.264 decoders enforce this at PPS
            // arrival time and rejects out-of-order
            // PPS-before-SPS deliveries; before this check, our decoder
            // silently accepted such PPSes, slices later activated them,
            // and we emitted frames that common H.264 decoders rejected outright.
            // Caught by the `ffmpeg_oracle_decode` fuzz target on
            // `crash-065436ed…` (round-120 Fuzz CI failure, task #1044):
            // PPS-then-SPS-then-IDR-x5 with sps_id=0 / pps_id=0; ffmpeg
            // refuses the PPS at storage time ("sps_id 0 out of range"
            // referring to the unstored backing SPS), our decoder used
            // to store the PPS using the default 4:2:0 chroma assumption
            // and then decode the IDR slices into 16x16 frames.
            NalUnitType::Pps => {
                // First peek the seq_parameter_set_id so we can verify
                // the referenced SPS exists BEFORE doing the full parse
                // (which would otherwise mis-parse the scaling-matrix
                // tail under a guessed chroma_format_idc).
                let probe = Pps::parse(&rbsp)?;
                let sps_id = probe.seq_parameter_set_id;
                // §G.7.4.2.2 — for an MVC PPS the referenced sequence
                // parameter set is the *MVC* sequence parameter set,
                // i.e. a subset SPS (NAL 15). The two id spaces are
                // separate (§7.4.1.2.1), so resolve the chroma format
                // from the ordinary SPS table first and fall back to the
                // subset SPS table (its embedded base SPS body) when no
                // ordinary SPS carries this id. The scaling-matrix tail
                // length depends only on that chroma_format_idc, which
                // both flavours carry.
                let chroma_format_idc = self
                    .sps_by_id
                    .get(sps_id as usize)
                    .and_then(|s| s.as_ref())
                    .map(|s| s.chroma_format_idc)
                    .or_else(|| {
                        self.subset_sps_by_id
                            .get(sps_id as usize)
                            .and_then(|s| s.as_ref())
                            .map(|s| s.sps.chroma_format_idc)
                    })
                    .ok_or(DecoderError::UnknownSps(sps_id))?;
                let pps = if chroma_format_idc != 1 {
                    Pps::parse_with_chroma_format(&rbsp, chroma_format_idc)?
                } else {
                    probe
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
            // §7.3.2.13 / §G.7.3.2.13 — coded slice MVC extension
            // (nal_unit_type 20). When the §7.3.1 dispatch parsed an MVC
            // NAL-unit-header extension (`svc_extension_flag == 0`), the
            // `slice_layer_extension_rbsp()` else-branch carries a plain
            // §7.3.3 `slice_header()`, which we parse against the subset
            // SPS the activated PPS references. An SVC
            // (`svc_extension_flag == 1`) body or a missing extension
            // header falls through to `Ignored` — those Annex F slice
            // bodies aren't modelled.
            NalUnitType::SliceExtension => match nu.extension {
                Some(nal::NalUnitHeaderExtension::Mvc(mvc)) => {
                    self.process_mvc_slice_extension_nal(&header, &rbsp, mvc)
                }
                _ => Ok(Event::Ignored {
                    nal_unit_type: header.nal_unit_type.as_u8(),
                    nal_bytes: nal_bytes.to_vec(),
                }),
            },
            // §7.3.2.1.2 / §7.4.1.2.1 — sequence parameter set extension
            // (NAL 13). Parses the auxiliary-coded-picture / alpha-blending
            // parameters and stores them keyed by the supplemented SPS id
            // (the ordinary SPS id space). Auxiliary reconstruction is not
            // wired — per §7.4.2.1.2 it is not required for conformance —
            // but the parameters are now surfaced rather than discarded.
            NalUnitType::SpsExtension => {
                let ext = SeqParameterSetExtension::parse(&rbsp)?;
                let id = ext.seq_parameter_set_id;
                // `id <= 31` validated inside `SeqParameterSetExtension::parse`.
                self.sps_extension_by_id[id as usize] = Some(ext);
                Ok(Event::SpsExtensionStored(id))
            }
            // Prefix NAL (14), depth parameter set (16), reserved
            // (17/18/22/23), slice extension depth (21), and all
            // unspecified values. Pass through as Ignored so the caller
            // can inspect/log.
            NalUnitType::Unspecified(_)
            | NalUnitType::Reserved(_)
            | NalUnitType::PrefixNalUnit
            | NalUnitType::DepthParameterSet
            | NalUnitType::SliceExtensionDepth
            | NalUnitType::UnspecifiedRange(_) => Ok(Event::Ignored {
                nal_unit_type: header.nal_unit_type.as_u8(),
                nal_bytes: nal_bytes.to_vec(),
            }),
        }
    }

    /// §G.7.3.2.13 / §G.7.4.1.2.1 — process a coded slice MVC extension
    /// NAL unit (`nal_unit_type == 20`, `svc_extension_flag == 0`).
    ///
    /// Resolves the activated PPS, then resolves that PPS's
    /// `seq_parameter_set_id` against the **subset SPS** id space (per
    /// §G.7.4.1.2.1: "a subset sequence parameter set RBSP … referred
    /// to by activation of a picture parameter set RBSP … activated by
    /// a coded slice MVC extension NAL unit"). The §7.3.3
    /// `slice_header()` is then parsed against the subset SPS's embedded
    /// base SPS, with `IdrPicFlag = !mvc.non_idr_flag` (§G.7.4.1.1).
    fn process_mvc_slice_extension_nal(
        &mut self,
        header: &NalHeader,
        rbsp: &[u8],
        mvc: nal::NalUnitHeaderMvcExtension,
    ) -> DecoderResult<Event> {
        if self.pps_by_id.iter().all(|p| p.is_none()) {
            return Err(DecoderError::NoActiveParameterSets);
        }
        // Peek pic_parameter_set_id (third ue(v) of the slice header).
        let pps_id = peek_slice_pps_id(rbsp)?;
        let pps = self
            .pps_by_id
            .get(pps_id as usize)
            .and_then(|p| p.as_ref())
            .ok_or(DecoderError::UnknownPps(pps_id))?
            .clone();
        // §A.2 — FMO is not modelled; reject as for base-view slices.
        if pps.num_slice_groups_minus1 > 0 {
            return Err(DecoderError::FmoNotSupported(pps.num_slice_groups_minus1));
        }
        // §G.7.4.1.2.1 — a type-20 slice activates a *subset* SPS
        // (NAL 15), looked up in the separate subset-SPS id space.
        let sps_id = pps.seq_parameter_set_id;
        let subset_sps = self
            .subset_sps_by_id
            .get(sps_id as usize)
            .and_then(|p| p.as_ref())
            .ok_or(DecoderError::UnknownSubsetSps(sps_id))?
            .clone();

        // §G.7.4.1.1 — IdrPicFlag = !non_idr_flag.
        let idr_pic_flag = mvc.non_idr_flag == 0;

        // §7.3.3 — parse the slice header against the subset SPS's base
        // SPS body.
        let (parsed, slice_data_cursor) = SliceHeader::parse_mvc_extension_and_tell(
            rbsp,
            &subset_sps.sps,
            &pps,
            header,
            idr_pic_flag,
        )?;

        Ok(Event::SliceExtension {
            nal_unit_type: header.nal_unit_type.as_u8(),
            nal_ref_idc: header.nal_ref_idc,
            mvc,
            idr_pic_flag,
            header: parsed,
            rbsp: rbsp.to_vec(),
            slice_data_cursor,
            pps,
            subset_sps,
        })
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
        // §A.2 — FMO is constrained to Baseline / Extended profiles, and
        // even there our §8.4 reconstruction path doesn't honour the
        // §8.2.2 MbToSliceGroupMap (it walks raster order). Reject
        // FMO-enabled PPS activation so we don't silently mis-decode a
        // stream that the spec restricts (and that common H.264 decoders reject
        // outright with "FMO is not implemented"). See
        // [`DecoderError::FmoNotSupported`] for the full citation.
        if pps.num_slice_groups_minus1 > 0 {
            return Err(DecoderError::FmoNotSupported(pps.num_slice_groups_minus1));
        }
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

    /// Look up a stored subset SPS (§7.3.2.1.3) by its embedded
    /// `seq_parameter_set_id`. Per §7.4.1.2.1 this id space is separate
    /// from [`Decoder::sps`].
    pub fn subset_sps(&self, id: u32) -> Option<&SubsetSps> {
        self.subset_sps_by_id
            .get(id as usize)
            .and_then(|p| p.as_ref())
    }

    /// Look up a stored sequence parameter set extension (§7.3.2.1.2) by
    /// the `seq_parameter_set_id` of the ordinary SPS it supplements.
    pub fn sps_extension(&self, id: u32) -> Option<&SeqParameterSetExtension> {
        self.sps_extension_by_id
            .get(id as usize)
            .and_then(|p| p.as_ref())
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
    fn pps_referencing_unknown_sps_is_rejected_at_storage() {
        // §7.4.1.2.1 strictness — a PPS whose `seq_parameter_set_id`
        // refers to an SPS not yet stored is rejected at PPS arrival
        // time. Previously this delayed the rejection to slice
        // activation time (`UnknownSps` on the slice path), which let
        // the PPS sit in `pps_by_id[]` parsed under a guessed
        // `chroma_format_idc = 1`. common H.264 decoders reject out-of-order
        // PPS-before-SPS deliveries the same way (regression for
        // `crash-065436ed…`, round-120 fuzz-CI failure / task #1044).
        let mut dec = Decoder::new();
        // PPS 0 points to SPS id 4, which we never stored.
        let err = dec
            .process_nal(&build_nal(8, 3, &build_minimal_pps_rbsp(0, 4)))
            .unwrap_err();
        assert!(
            matches!(err, DecoderError::UnknownSps(4)),
            "expected UnknownSps(4) at PPS storage, got {err:?}"
        );
        // Nothing should have been committed — neither the bad PPS nor
        // any active-pps state.
        assert!(dec.pps(0).is_none());
        assert!(dec.active_pps_id().is_none());
    }

    #[test]
    fn slice_activating_fmo_pps_is_rejected() {
        // §A.2 — FMO (num_slice_groups_minus1 > 0) is not honoured by
        // our §8.4 raster reconstruction. A slice that activates such a
        // PPS must be rejected at activation time so we don't emit a
        // mis-decoded picture for a stream the H.264 reference decoder
        // (common H.264 decoders) would reject as "FMO is not implemented".
        // Regression for the 309-byte fuzz-oracle divergence
        // (`crash-181e9dea7dfa8fcb2c9721a3f9f044214af6167b`): the
        // crash input carried a PPS with num_slice_groups_minus1=2,
        // and our decoder used to silently produce a frame.
        let mut dec = Decoder::new();
        let _ = dec
            .process_nal(&build_nal(7, 3, &build_minimal_sps_rbsp(0)))
            .unwrap();
        // FMO PPS — interleaved (slice_group_map_type=0) with 2 groups.
        let pps_rbsp = {
            let mut w = BitWriter::new();
            w.ue(0); // pic_parameter_set_id
            w.ue(0); // seq_parameter_set_id
            w.u(1, 0); // entropy_coding_mode_flag = CAVLC
            w.u(1, 0); // bottom_field_pic_order_in_frame_present_flag
            w.ue(1); // num_slice_groups_minus1 = 1 (2 groups → FMO active)
            w.ue(0); // slice_group_map_type = 0 (interleaved)
            w.ue(3); // run_length_minus1[0]
            w.ue(3); // run_length_minus1[1]
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
        };
        let _ = dec.process_nal(&build_nal(8, 3, &pps_rbsp)).unwrap();
        let slice_nal = build_nal(1, 3, &build_minimal_i_slice_rbsp(0));
        let err = dec.process_nal(&slice_nal).unwrap_err();
        assert!(
            matches!(err, DecoderError::FmoNotSupported(1)),
            "expected FmoNotSupported(1), got {err:?}"
        );
        // Activation must NOT have committed for a rejected slice.
        assert!(dec.active_pps_id().is_none());
        assert!(dec.active_sps_id().is_none());
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
        // Prefix NAL unit (SVC/MVC) — type 14. The §7.3.1 dispatch
        // consumes 3 extension bytes (svc_extension_flag + the §F/G
        // body), so a syntactically valid type-14 NAL needs at least
        // 3 bytes of payload after the NAL header byte. The §7.4.1.2
        // semantics are still "ignore at the top level" — the decoder
        // forwards the raw bytes as Event::Ignored.
        let nal = build_nal(14, 0, &[0x00, 0x00, 0x00]);
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
    fn sps_extension_13_is_parsed_and_stored() {
        // §7.3.2.1.2 seq_parameter_set_extension_rbsp() body, hand-packed:
        //   seq_parameter_set_id = 2  → ue(v) "011"
        //   aux_format_idc       = 1  → ue(v) "010"
        //   bit_depth_aux_minus8 = 0  → ue(v) "1"
        //   alpha_incr_flag      = 1  → u(1)  "1"
        //   alpha_opaque_value   = 0x1FF (9 bits) "111111111"
        //   alpha_transparent_value = 0x000 (9 bits) "000000000"
        //   additional_extension_flag = 0 → u(1) "0"
        //   rbsp_trailing_bits(): "1" then zero-pad.
        // Bit string:
        //   011 010 1 1 111111111 000000000 0 1
        let bits = "011".to_string() + "010" + "1" + "1" + "111111111" + "000000000" + "0" + "1";
        let mut rbsp = Vec::new();
        let mut acc = 0u8;
        let mut n = 0u8;
        for ch in bits.chars() {
            acc = (acc << 1) | (ch == '1') as u8;
            n += 1;
            if n == 8 {
                rbsp.push(acc);
                acc = 0;
                n = 0;
            }
        }
        if n > 0 {
            rbsp.push(acc << (8 - n));
        }

        let nal = build_nal(13, 0, &rbsp);
        let mut dec = Decoder::new();
        let ev = dec.process_nal(&nal).unwrap();
        assert!(matches!(ev, Event::SpsExtensionStored(2)));

        let ext = dec.sps_extension(2).expect("SPS extension stored");
        assert_eq!(ext.seq_parameter_set_id, 2);
        assert_eq!(ext.aux_format_idc, 1);
        let aux = ext.aux_format.expect("aux block present");
        assert_eq!(aux.bit_depth_aux(), 8);
        assert!(aux.alpha_incr_flag);
        assert_eq!(aux.alpha_opaque_value, 0x1FF);
        assert_eq!(aux.alpha_transparent_value, 0x000);
        assert!(!ext.additional_extension_flag);

        // No extension was stored for any other SPS id.
        assert!(dec.sps_extension(0).is_none());
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

    // ---- §G.7.3.2.13 coded slice MVC extension (NAL 20) ----

    /// §7.3.2.1.3 + §G.7.3.2.1.4 — minimal Stereo High (profile_idc=128)
    /// subset SPS RBSP. The embedded base `seq_parameter_set_data()` is
    /// 4:2:0, POC type 0 with 4-bit frame_num / pic_order_cnt_lsb, and
    /// 320x240 dims so the §7.3.3 slice header built below parses
    /// identically to the base-view path. `seq_parameter_set_id` lives
    /// in the subset-SPS id space.
    fn build_minimal_subset_sps_rbsp(sps_id: u32) -> Vec<u8> {
        let mut w = BitWriter::new();
        // seq_parameter_set_data() — profile 128 takes the chroma-
        // extended gate (§7.3.2.1.1).
        w.u(8, 128); // profile_idc = Stereo High
        w.u(8, 0); // constraint_sets + reserved_zero_2bits
        w.u(8, 30); // level_idc
        w.ue(sps_id); // seq_parameter_set_id
        w.ue(1); // chroma_format_idc = 4:2:0
        w.ue(0); // bit_depth_luma_minus8
        w.ue(0); // bit_depth_chroma_minus8
        w.u(1, 0); // qpprime_y_zero_transform_bypass_flag
        w.u(1, 0); // seq_scaling_matrix_present_flag
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
                   // §7.3.2.1.3 subset_seq_parameter_set_rbsp tail (profile 128 →
                   // MVC extension branch).
        w.u(1, 1); // bit_equal_to_one
                   // §G.7.3.2.1.4 seq_parameter_set_mvc_extension() — two views.
        w.ue(1); // num_views_minus1 = 1
        w.ue(0); // view_id[0]
        w.ue(2); // view_id[1]
        w.ue(1); // num_anchor_refs_l0[1]
        w.ue(0); // anchor_ref_l0[1][0]
        w.ue(0); // num_anchor_refs_l1[1]
        w.ue(1); // num_non_anchor_refs_l0[1]
        w.ue(0); // non_anchor_ref_l0[1][0]
        w.ue(0); // num_non_anchor_refs_l1[1]
        w.ue(0); // num_level_values_signalled_minus1
        w.u(8, 40); // level_idc[0]
        w.ue(0); // num_applicable_ops_minus1[0]
        w.u(3, 1); // applicable_op_temporal_id[0][0]
        w.ue(1); // applicable_op_num_target_views_minus1[0][0]
        w.ue(0); // applicable_op_target_view_id[0][0][0]
        w.ue(2); // applicable_op_target_view_id[0][0][1]
        w.ue(1); // applicable_op_num_views_minus1[0][0]
        w.u(1, 0); // mvc_vui_parameters_present_flag
        w.u(1, 0); // additional_extension2_flag
        w.trailing();
        w.into_bytes()
    }

    /// §G.7.3.1.1 — pack the three MVC NAL-unit-header extension bytes
    /// (`svc_extension_flag = 0` in the MSB, left out of the OR).
    fn pack_mvc_ext_bytes(
        non_idr: u8,
        priority: u8,
        view: u16,
        temporal: u8,
        anchor: u8,
        inter_view: u8,
    ) -> [u8; 3] {
        let bits24: u32 = ((non_idr as u32 & 0x1) << 22)
            | ((priority as u32 & 0x3F) << 16)
            | ((view as u32 & 0x3FF) << 6)
            | ((temporal as u32 & 0x7) << 3)
            | ((anchor as u32 & 0x1) << 2)
            | ((inter_view as u32 & 0x1) << 1)
            | 1; // reserved_one_bit = 1 (§G.7.4.1.1)
        [
            ((bits24 >> 16) & 0xFF) as u8,
            ((bits24 >> 8) & 0xFF) as u8,
            (bits24 & 0xFF) as u8,
        ]
    }

    /// Build a NAL-20 coded slice MVC extension: header byte + 3 MVC
    /// extension bytes + slice RBSP.
    fn build_mvc_slice_ext_nal(
        nal_ref_idc: u8,
        non_idr: u8,
        view_id: u16,
        slice_rbsp: &[u8],
    ) -> Vec<u8> {
        let mut nal = build_nal(20, nal_ref_idc, &[]);
        nal.extend_from_slice(&pack_mvc_ext_bytes(non_idr, 0, view_id, 0, 1, 1));
        nal.extend_from_slice(slice_rbsp);
        nal
    }

    /// Store the SPS / subset SPS / PPS a NAL-20 test needs. The PPS
    /// references `sps_id` which is resolved against the subset-SPS id
    /// space per §G.7.4.1.2.1.
    fn setup_mvc_decoder(sps_id: u32, pps_id: u32) -> Decoder {
        let mut dec = Decoder::new();
        // Ordinary base-view SPS (NAL 7) at the same id, as a real MVC
        // stream carries for view 0.
        dec.process_nal(&build_nal(7, 3, &build_minimal_sps_rbsp(sps_id)))
            .unwrap();
        // Subset SPS (NAL 15) — separate id space.
        let ev = dec
            .process_nal(&build_nal(15, 3, &build_minimal_subset_sps_rbsp(sps_id)))
            .unwrap();
        assert!(matches!(ev, Event::SubsetSpsStored(_)));
        // PPS (NAL 8) referencing that sps_id.
        dec.process_nal(&build_nal(8, 3, &build_minimal_pps_rbsp(pps_id, sps_id)))
            .unwrap();
        dec
    }

    #[test]
    fn mvc_slice_extension_parses_against_subset_sps() {
        // §G.7.3.2.13 — NAL 20, svc_extension_flag=0 → slice_header().
        let mut dec = setup_mvc_decoder(0, 0);
        // non_idr_flag = 1 (non-IDR dependent-view slice), view_id = 2.
        let slice = build_minimal_i_slice_rbsp(0);
        let nal = build_mvc_slice_ext_nal(3, 1, 2, &slice);
        let ev = dec.process_nal(&nal).unwrap();
        match ev {
            Event::SliceExtension {
                nal_unit_type,
                nal_ref_idc,
                mvc,
                idr_pic_flag,
                header,
                subset_sps,
                ..
            } => {
                assert_eq!(nal_unit_type, 20);
                assert_eq!(nal_ref_idc, 3);
                assert_eq!(mvc.view_id, 2);
                assert_eq!(mvc.non_idr_flag, 1);
                // §G.7.4.1.1 — IdrPicFlag = !non_idr_flag.
                assert!(!idr_pic_flag);
                assert_eq!(header.first_mb_in_slice, 0);
                assert_eq!(header.pic_parameter_set_id, 0);
                // The slice was parsed against the subset SPS's base SPS.
                assert_eq!(subset_sps.sps.profile_idc, 128);
            }
            other => panic!("expected SliceExtension, got {other:?}"),
        }
    }

    #[test]
    fn mvc_slice_extension_idr_flag_from_non_idr_flag() {
        // non_idr_flag = 0 → IdrPicFlag = 1 (§G.7.4.1.1). An IDR slice
        // header carries idr_pic_id (§7.3.3), absent from the non-IDR
        // body, so build it explicitly here.
        let mut dec = setup_mvc_decoder(0, 0);
        let mut w = BitWriter::new();
        w.ue(0); // first_mb_in_slice
        w.ue(7); // slice_type = 7 (I)
        w.ue(0); // pic_parameter_set_id
        w.u(4, 0); // frame_num (4 bits)
        w.ue(0); // idr_pic_id — present because IdrPicFlag = 1
        w.u(4, 0); // pic_order_cnt_lsb (4 bits)
        w.u(1, 0); // dec_ref_pic_marking: no_output_of_prior_pics_flag
        w.u(1, 0); // dec_ref_pic_marking: long_term_reference_flag
        w.se(0); // slice_qp_delta
        w.u(8, 0x80); // padding
        let slice = w.into_bytes();
        let nal = build_mvc_slice_ext_nal(3, 0, 2, &slice);
        let ev = dec.process_nal(&nal).unwrap();
        match ev {
            Event::SliceExtension { idr_pic_flag, .. } => {
                assert!(idr_pic_flag, "non_idr_flag=0 must yield IdrPicFlag=1");
            }
            other => panic!("expected SliceExtension, got {other:?}"),
        }
    }

    #[test]
    fn mvc_slice_extension_missing_subset_sps_is_error() {
        // PPS references sps_id 0 but only an ordinary SPS is stored —
        // no subset SPS at id 0. §G.7.4.1.2.1 requires a subset SPS for
        // a type-20 activation.
        let mut dec = Decoder::new();
        dec.process_nal(&build_nal(7, 3, &build_minimal_sps_rbsp(0)))
            .unwrap();
        dec.process_nal(&build_nal(8, 3, &build_minimal_pps_rbsp(0, 0)))
            .unwrap();
        let slice = build_minimal_i_slice_rbsp(0);
        let nal = build_mvc_slice_ext_nal(3, 1, 2, &slice);
        let err = dec.process_nal(&nal).unwrap_err();
        assert!(
            matches!(err, DecoderError::UnknownSubsetSps(0)),
            "expected UnknownSubsetSps(0), got {err:?}"
        );
    }

    #[test]
    fn mvc_slice_extension_unknown_pps_is_error() {
        let mut dec = setup_mvc_decoder(0, 0);
        // Slice references pps_id 5 which was never stored.
        let slice = build_minimal_i_slice_rbsp(5);
        let nal = build_mvc_slice_ext_nal(3, 1, 2, &slice);
        let err = dec.process_nal(&nal).unwrap_err();
        assert!(
            matches!(err, DecoderError::UnknownPps(5)),
            "expected UnknownPps(5), got {err:?}"
        );
    }

    #[test]
    fn svc_slice_extension_falls_through_to_ignored() {
        // svc_extension_flag = 1 → Annex F SVC body, which we don't
        // parse. The NAL must surface as Ignored, not SliceExtension.
        let mut dec = setup_mvc_decoder(0, 0);
        // Build a NAL 20 with svc_extension_flag = 1 (MSB set).
        let mut svc_ext = pack_mvc_ext_bytes(1, 0, 0, 0, 0, 0);
        svc_ext[0] |= 0x80; // set svc_extension_flag
        let mut nal = build_nal(20, 3, &[]);
        nal.extend_from_slice(&svc_ext);
        nal.extend_from_slice(&build_minimal_i_slice_rbsp(0));
        let ev = dec.process_nal(&nal).unwrap();
        assert!(
            matches!(
                ev,
                Event::Ignored {
                    nal_unit_type: 20,
                    ..
                }
            ),
            "expected Ignored for SVC body, got {ev:?}"
        );
    }
}
