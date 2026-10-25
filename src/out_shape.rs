//! The stream-level description of the frames the decoder emits — the
//! answer to `Decoder::output_params` (oxideav-core 0.1.42).
//!
//! A [`VideoFrame`](oxideav_core::VideoFrame) carries only `pts` and
//! planes; its size, layout, rate and colour signal are stated once per
//! stream. Every picture queued for output carries its [`OutShape`]
//! (derived from the SPS active for that picture) so the description
//! can change exactly at the first frame of a new shape, independent of
//! how many pictures sit in the §C.4 reorder buffer and of the worker
//! budget.

use oxideav_core::{CodecId, CodecParameters, ColorSignal, PixelFormat, Rational};

use crate::sps::{FrameCropMargins, Sps};

/// The description of one emitted frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OutShape {
    /// §7.4.2.1.1 — width of the cropping rectangle, luma samples.
    pub width: u32,
    /// §7.4.2.1.1 — height of the cropping rectangle, luma samples.
    pub height: u32,
    /// The planar layout of the emitted planes; `None` for bit-depth
    /// combinations no [`PixelFormat`] describes (9 / 11 / 13 / 14-bit,
    /// or luma and chroma at different depths).
    pub pixel_format: Option<PixelFormat>,
    /// §E.2.1 VUI timing: `time_scale / (2 * num_units_in_tick)`.
    pub frame_rate: Option<Rational>,
    /// §E.2.1 VUI `video_signal_type` / `colour_description`.
    pub color: ColorSignal,
}

/// §7.4.2.1.1 — size of the cropping rectangle of a `w` x `h` decoded
/// picture (margins clamped exactly as the frame conversion clamps
/// them, so the two always agree).
pub(crate) fn crop_dims(w: u32, h: u32, crop: FrameCropMargins) -> (u32, u32) {
    let left = crop.left.min(w);
    let top = crop.top.min(h);
    (
        w - left - crop.right.min(w - left),
        h - top - crop.bottom.min(h - top),
    )
}

/// The planar layout the decoder emits for pictures of `sps`.
///
/// `plane_mode` is set on the three sub-decoders of a
/// `separate_colour_plane_flag == 1` stream, each of which emits one
/// colour plane as a monochrome picture; the routing decoder merges
/// the three into a 4:4:4 frame.
pub(crate) fn sps_pixel_format(sps: &Sps, plane_mode: bool) -> Option<PixelFormat> {
    use PixelFormat::*;
    let luma = sps.bit_depth_luma_minus8 + 8;
    // §7.4.2.1.1 — with separate colour planes every plane is coded
    // as a monochrome picture at the luma bit depth.
    let cat = if sps.separate_colour_plane_flag && !plane_mode {
        3
    } else {
        sps.chroma_array_type()
    };
    if cat != 0 && !sps.separate_colour_plane_flag && sps.bit_depth_chroma_minus8 + 8 != luma {
        return None;
    }
    Some(match (cat, luma) {
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

/// §E.2.1 — the frame rate the VUI timing information signals: a
/// frame lasts `2 * num_units_in_tick / time_scale` seconds.
pub(crate) fn sps_frame_rate(sps: &Sps) -> Option<Rational> {
    let t = sps.vui.as_ref()?.timing_info.as_ref()?;
    if t.num_units_in_tick == 0 || t.time_scale == 0 {
        return None;
    }
    let (num, den) = reduce(t.time_scale as i64, 2 * t.num_units_in_tick as i64);
    Some(Rational::new(num, den))
}

fn reduce(a: i64, b: i64) -> (i64, i64) {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    (a / x, b / x)
}

/// §E.2.1 — the colour signal of the VUI `video_signal_type`: the
/// range from `video_full_range_flag`, the H.273 code points from
/// `colour_description` (each `2` = unspecified when absent).
pub(crate) fn sps_color_signal(sps: &Sps) -> ColorSignal {
    let Some(vst) = sps.vui.as_ref().and_then(|v| v.video_signal_type.as_ref()) else {
        return ColorSignal::unspecified();
    };
    let (p, t, m) = vst
        .colour_description
        .as_ref()
        .map(|c| {
            (
                c.colour_primaries,
                c.transfer_characteristics,
                c.matrix_coefficients,
            )
        })
        .unwrap_or((2, 2, 2));
    ColorSignal::from_code_points(p, t, m, vst.video_full_range_flag)
}

/// What a picture's SPS fixes about its output, whatever its size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ShapeTraits {
    pub pixel_format: Option<PixelFormat>,
    pub frame_rate: Option<Rational>,
    pub color: ColorSignal,
}

impl ShapeTraits {
    pub(crate) fn of_sps(sps: &Sps, plane_mode: bool) -> Self {
        Self {
            pixel_format: sps_pixel_format(sps, plane_mode),
            frame_rate: sps_frame_rate(sps),
            color: sps_color_signal(sps),
        }
    }

    /// The shape of a `w` x `h` decoded picture output with the
    /// cropping margins `crop` (the frame margins, or their
    /// [`FrameCropMargins::for_field`] form for a lone field).
    pub(crate) fn shape(self, w: u32, h: u32, crop: FrameCropMargins) -> OutShape {
        let (width, height) = crop_dims(w, h, crop);
        OutShape {
            width,
            height,
            pixel_format: self.pixel_format,
            frame_rate: self.frame_rate,
            color: self.color,
        }
    }
}

impl OutShape {
    /// The shape of a frame picture of `sps`, before any picture of it
    /// has been decoded.
    pub(crate) fn of_sps(sps: &Sps, plane_mode: bool) -> Self {
        ShapeTraits::of_sps(sps, plane_mode).shape(
            sps.pic_width_in_mbs() * 16,
            sps.frame_height_in_mbs() * 16,
            sps.frame_crop_margins(),
        )
    }

    /// The 4:4:4 frame merged from three separate-colour-plane
    /// monochrome pictures of this shape.
    pub(crate) fn merged_planes(mut self) -> Self {
        use PixelFormat::*;
        self.pixel_format = match self.pixel_format {
            Some(Gray8) => Some(Yuv444P),
            Some(Gray10Le) => Some(Yuv444P10Le),
            Some(Gray12Le) => Some(Yuv444P12Le),
            other => other,
        };
        self
    }

    /// As the `CodecParameters` `output_params` answers.
    pub(crate) fn to_params(self, codec_id: &CodecId) -> CodecParameters {
        let mut p = CodecParameters::video(codec_id.clone());
        p.width = Some(self.width);
        p.height = Some(self.height);
        p.pixel_format = self.pixel_format;
        p.frame_rate = self.frame_rate;
        p.color_signal = self.color;
        p
    }
}
