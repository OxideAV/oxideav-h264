//! H.264 decoder scaffold exposed through the `oxideav_codec::Decoder`
//! trait so containers + players can route packets at us.
//!
//! What this scaffold currently does:
//! 1. Accepts Annex B byte-stream packets or AVCC-framed packets
//!    (length prefix size taken from `extradata` when present — an
//!    `AVCDecoderConfigurationRecord` per ISO/IEC 14496-15).
//! 2. Walks NAL units through [`crate::decoder::Decoder`] which
//!    captures SPS/PPS and emits parsed slice headers.
//! 3. For IDR slices, drives `slice_data::parse_slice_data` +
//!    `reconstruct::reconstruct_slice` to produce a
//!    [`oxideav_core::VideoFrame`] and queues it for
//!    `receive_frame`.
//! 4. Non-IDR (P/B) slices are currently silently skipped — wiring the
//!    DPB / ref_store through the trait surface is the next step.
//!
//! The trait is what oxideplay/oxideav-pipeline consumes; registering
//! this decoder (via [`crate::register`]) stops the "codec not found"
//! error on the first h264 packet. IDR frames now flow through as real
//! pixels; non-IDR packets still produce no output yet.

use std::collections::VecDeque;

use oxideav_codec::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, TimeBase, VideoFrame,
    VideoPlane,
};

use crate::decoder::{Decoder as H264Driver, Event};
use crate::mb_grid::MbGrid;
use crate::picture::Picture;
use crate::ref_store::NoRefs;
use crate::slice_header::SliceHeader;
use crate::{reconstruct, slice_data};

/// Registry factory — called by the codec registry when a container
/// wants a decoder for H.264.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let mut dec = H264CodecDecoder::new(params.codec_id.clone());
    if !params.extradata.is_empty() {
        dec.consume_extradata(&params.extradata)?;
    }
    Ok(Box::new(dec))
}

/// Parse-and-reconstruct-IDR decoder scaffold.
pub struct H264CodecDecoder {
    codec_id: CodecId,
    /// NAL unit length-prefix size from `avcC`, when present. `None`
    /// means the input is treated as Annex B byte-stream.
    length_size: Option<u8>,
    driver: H264Driver,
    /// Last slice header we parsed — useful for probes / asserts.
    pub last_slice: Option<SliceHeader>,
    eof: bool,
    /// Reconstructed frames queued for `receive_frame`.
    frames: VecDeque<VideoFrame>,
    /// Packet-level pts passed on the most recent `send_packet`. We
    /// stamp the first frame produced from that packet with it.
    pending_pts: Option<i64>,
    /// Packet time_base so downstream consumers can rescale.
    pending_time_base: TimeBase,
}

impl H264CodecDecoder {
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            length_size: None,
            driver: H264Driver::new(),
            last_slice: None,
            eof: false,
            frames: VecDeque::new(),
            pending_pts: None,
            pending_time_base: TimeBase::new(1, 1),
        }
    }

    /// §4.1.3.1 ISO/IEC 14496-15 — `AVCDecoderConfigurationRecord`.
    /// Minimal parse: pick up `lengthSizeMinusOne` and feed the stored
    /// SPS + PPS NAL units through the driver.
    ///
    /// Layout:
    /// ```text
    ///   u8  configurationVersion (= 1)
    ///   u8  AVCProfileIndication
    ///   u8  profile_compatibility
    ///   u8  AVCLevelIndication
    ///   u8  reserved (6 bits, 111111) + lengthSizeMinusOne (2 bits)
    ///   u8  reserved (3 bits, 111) + numOfSequenceParameterSets (5 bits)
    ///     repeated numOfSequenceParameterSets times:
    ///       u16 sequenceParameterSetLength
    ///       <that many> sequenceParameterSetNALUnit
    ///   u8  numOfPictureParameterSets
    ///     repeated:
    ///       u16 pictureParameterSetLength
    ///       <that many> pictureParameterSetNALUnit
    /// ```
    pub fn consume_extradata(&mut self, extra: &[u8]) -> Result<()> {
        if extra.len() < 7 {
            return Err(Error::invalid("h264: extradata shorter than avcC header"));
        }
        if extra[0] != 1 {
            return Err(Error::invalid(
                "h264: avcC configurationVersion must be 1",
            ));
        }
        self.length_size = Some((extra[4] & 0x03) + 1);
        let num_sps = (extra[5] & 0x1f) as usize;
        let mut pos = 6;
        for _ in 0..num_sps {
            if pos + 2 > extra.len() {
                return Err(Error::invalid("h264: avcC truncated at SPS length"));
            }
            let len = u16::from_be_bytes([extra[pos], extra[pos + 1]]) as usize;
            pos += 2;
            if pos + len > extra.len() {
                return Err(Error::invalid("h264: avcC truncated at SPS body"));
            }
            let _ = self
                .driver
                .process_nal(&extra[pos..pos + len])
                .map_err(|e| Error::invalid(format!("h264 avcC SPS: {e}")))?;
            pos += len;
        }
        if pos >= extra.len() {
            return Err(Error::invalid("h264: avcC truncated at PPS count"));
        }
        let num_pps = extra[pos] as usize;
        pos += 1;
        for _ in 0..num_pps {
            if pos + 2 > extra.len() {
                return Err(Error::invalid("h264: avcC truncated at PPS length"));
            }
            let len = u16::from_be_bytes([extra[pos], extra[pos + 1]]) as usize;
            pos += 2;
            if pos + len > extra.len() {
                return Err(Error::invalid("h264: avcC truncated at PPS body"));
            }
            let _ = self
                .driver
                .process_nal(&extra[pos..pos + len])
                .map_err(|e| Error::invalid(format!("h264 avcC PPS: {e}")))?;
            pos += len;
        }
        Ok(())
    }

    /// Handle a single emitted driver event, routing IDR slices into the
    /// reconstruct pipeline. Non-IDR slices only update `last_slice`
    /// (they need ref pictures we don't wire up yet).
    fn handle_event(&mut self, ev: Event) -> Result<()> {
        match ev {
            Event::Slice {
                nal_unit_type,
                nal_ref_idc: _,
                header,
                rbsp,
                slice_data_cursor,
            } => {
                self.last_slice = Some(header.clone());
                if nal_unit_type == 5 {
                    // IDR slice — try to reconstruct a picture.
                    self.reconstruct_idr(&header, &rbsp, slice_data_cursor)?;
                }
                // else: non-IDR is dropped for now; see module docs.
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Drive IDR reconstruction for one slice. Pulls the active SPS/PPS
    /// out of the driver, parses slice_data, allocates a `Picture`, runs
    /// `reconstruct_slice`, converts to a [`VideoFrame`], and queues it.
    fn reconstruct_idr(
        &mut self,
        header: &SliceHeader,
        rbsp: &[u8],
        cursor: (usize, u8),
    ) -> Result<()> {
        // Snapshot SPS and PPS to detach from the borrow on `self.driver`
        // and avoid holding a shared borrow while we mutate `self.frames`.
        let sps = self
            .driver
            .active_sps()
            .cloned()
            .ok_or_else(|| Error::invalid("h264: IDR slice with no active SPS"))?;
        let pps = self
            .driver
            .active_pps()
            .cloned()
            .ok_or_else(|| Error::invalid("h264: IDR slice with no active PPS"))?;

        let sd = slice_data::parse_slice_data(rbsp, cursor.0, cursor.1, header, &sps, &pps)
            .map_err(|e| Error::invalid(format!("h264 slice_data: {e}")))?;

        let width_samples = sps.pic_width_in_mbs() * 16;
        let height_samples = sps.frame_height_in_mbs() * 16;
        let chroma_array_type = sps.chroma_array_type();
        let mut pic = Picture::new(
            width_samples,
            height_samples,
            chroma_array_type,
            sps.bit_depth_luma_minus8 + 8,
            sps.bit_depth_chroma_minus8 + 8,
        );
        let mut grid = MbGrid::new(sps.pic_width_in_mbs(), sps.frame_height_in_mbs());
        reconstruct::reconstruct_slice(&sd, header, &sps, &pps, &NoRefs, &mut pic, &mut grid)
            .map_err(|e| Error::invalid(format!("h264 reconstruct: {e}")))?;

        // Stamp the first frame produced from this packet with the
        // packet's pts; subsequent IDR slices inside the same packet
        // (rare/AU-ish) get no pts.
        let pts = self.pending_pts.take();
        let time_base = self.pending_time_base;
        let vf = picture_to_video_frame(&pic, pts, time_base);
        self.frames.push_back(vf);
        Ok(())
    }
}

impl Decoder for H264CodecDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.pending_pts = packet.pts;
        self.pending_time_base = packet.time_base;
        let data = packet.data.clone();
        match self.length_size {
            Some(n) => {
                // AVCC framing — walk length-prefixed NAL units and
                // hand each to the driver.
                let mut i = 0usize;
                let n = n as usize;
                while i < data.len() {
                    if i + n > data.len() {
                        return Err(Error::invalid("h264: AVCC length prefix truncated"));
                    }
                    let mut len = 0usize;
                    for k in 0..n {
                        len = (len << 8) | data[i + k] as usize;
                    }
                    i += n;
                    if i + len > data.len() {
                        return Err(Error::invalid("h264: AVCC NAL payload truncated"));
                    }
                    let ev = self
                        .driver
                        .process_nal(&data[i..i + len])
                        .map_err(|e| Error::invalid(format!("h264 NAL parse: {e}")))?;
                    self.handle_event(ev)?;
                    i += len;
                }
                Ok(())
            }
            None => {
                // Annex B framing. Collect events first so the driver
                // borrow ends before we recurse into handle_event (which
                // re-borrows self.driver to read active_sps/pps).
                let events: Vec<_> = self.driver.process_annex_b(&data).collect();
                for ev in events {
                    match ev {
                        Ok(ev) => self.handle_event(ev)?,
                        Err(e) => {
                            return Err(Error::invalid(format!("h264 NAL parse: {e}")));
                        }
                    }
                }
                Ok(())
            }
        }
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(vf) = self.frames.pop_front() {
            return Ok(Frame::Video(vf));
        }
        if self.eof {
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.driver = H264Driver::new();
        self.last_slice = None;
        self.eof = false;
        self.frames.clear();
        self.pending_pts = None;
        Ok(())
    }
}

/// Convert a reconstructed [`Picture`] to a [`VideoFrame`]. Samples are
/// clamped to 8-bit (0..=255) for now; higher bit depths will get
/// dedicated pixel formats later.
fn picture_to_video_frame(pic: &Picture, pts: Option<i64>, time_base: TimeBase) -> VideoFrame {
    let format = match (pic.chroma_array_type, pic.bit_depth_luma) {
        (0, 8) => PixelFormat::Gray8,
        (1, 8) => PixelFormat::Yuv420P,
        (2, 8) => PixelFormat::Yuv422P,
        (3, 8) => PixelFormat::Yuv444P,
        // Fall back to 4:2:0 for higher bit depths — samples are still
        // clamped to 8-bit below, which is lossy but keeps the pipeline
        // moving. High-bit-depth streams need a wider pixel format later.
        _ => PixelFormat::Yuv420P,
    };

    let w = pic.width_in_samples as usize;
    let cw = pic.chroma_width() as usize;

    let clamp_u8 = |s: i32| s.clamp(0, 255) as u8;

    let luma: Vec<u8> = pic.luma.iter().map(|&s| clamp_u8(s)).collect();
    let mut planes = vec![VideoPlane {
        stride: w,
        data: luma,
    }];
    if pic.chroma_array_type != 0 {
        let cb: Vec<u8> = pic.cb.iter().map(|&s| clamp_u8(s)).collect();
        let cr: Vec<u8> = pic.cr.iter().map(|&s| clamp_u8(s)).collect();
        planes.push(VideoPlane {
            stride: cw,
            data: cb,
        });
        planes.push(VideoPlane {
            stride: cw,
            data: cr,
        });
    }

    VideoFrame {
        format,
        width: pic.width_in_samples,
        height: pic.height_in_samples,
        pts,
        time_base,
        planes,
    }
}
