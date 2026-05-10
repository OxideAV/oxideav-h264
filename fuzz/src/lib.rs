//! Runtime libavcodec interop for the H.264 cross-decode fuzz oracle.
//!
//! The shared library is loaded via `dlopen` at first call — there is
//! no `ffmpeg-sys` / `rusty_ffmpeg`-style build-script dep that would
//! pull libavcodec source into the workspace's cargo dep tree. The
//! oracle harness checks `libavcodec::available()` up front and
//! `return`s early when the shared library isn't installed, so fuzz
//! binaries built on a host without ffmpeg simply do nothing instead
//! of panicking.
//!
//! Workspace policy bars consulting libavcodec / x264 / openh264 source;
//! we only inspect the public C header (`<libavcodec/avcodec.h>`) for
//! function signatures + the documented `AV_CODEC_ID_H264` constant.
//!
//! Install on Debian / Ubuntu via `apt-get install -y ffmpeg` (which
//! pulls `libavcodec61` or whichever is current). On macOS use
//! `brew install ffmpeg`.

#![allow(unsafe_code)]

pub mod libavcodec {
    use libloading::{Library, Symbol};
    use std::ffi::c_void;
    use std::sync::OnceLock;

    /// `AV_CODEC_ID_H264` — documented value `28` in the public
    /// `enum AVCodecID` (libavcodec/codec_id.h). Stable across all
    /// libavcodec major versions we target (58 through 62).
    pub const AV_CODEC_ID_H264: i32 = 28;

    /// `AVERROR(EAGAIN)` — `-EAGAIN` per the documented AVERROR macro.
    /// `EAGAIN == 11` on Linux/glibc; on macOS it's `35`. We only test
    /// against this in CI (Linux runner), so the Linux value is fine.
    /// Misidentifying it just means we treat EAGAIN as a hard error,
    /// which the harness already tolerates as "skip this iteration".
    const AVERROR_EAGAIN: i32 = -11;

    /// Conventional libavcodec shared-object names the loader will try
    /// in order. Covers Debian / Ubuntu (versioned `.so.NN`), the
    /// unversioned `-dev` symlink, and macOS (`.dylib`).
    const CANDIDATES: &[&str] = &[
        "libavcodec.so.62",
        "libavcodec.so.61",
        "libavcodec.so.60",
        "libavcodec.so.59",
        "libavcodec.so.58",
        "libavcodec.so",
        "libavcodec.dylib",
    ];

    fn lib() -> Option<&'static Library> {
        static LIB: OnceLock<Option<Library>> = OnceLock::new();
        LIB.get_or_init(|| {
            for name in CANDIDATES {
                // SAFETY: `Library::new` is documented as unsafe because
                // the loaded library may run code at load time. We
                // accept that risk for fuzz tooling — libavcodec is a
                // well-behaved shared library distributed by distros.
                if let Ok(l) = unsafe { Library::new(name) } {
                    return Some(l);
                }
            }
            None
        })
        .as_ref()
    }

    /// True iff a libavcodec shared library was successfully loaded.
    /// Cross-decode fuzz harnesses early-return when this is false so
    /// the binary still runs without an oracle (the assertions just
    /// don't fire).
    pub fn available() -> bool {
        lib().is_some()
    }

    /// An 8-bit YUV frame as decoded by libavcodec.
    ///
    /// `chroma_w_log2` / `chroma_h_log2` describe the chroma subsampling
    /// ratio for comparison with our decoder: 4:2:0 → (1, 1); 4:2:2 →
    /// (1, 0); 4:4:4 → (0, 0). The harness only asserts these match —
    /// it never tries to convert formats.
    pub struct DecodedYuv {
        pub width: u32,
        pub height: u32,
        pub chroma_w_log2: u8,
        pub chroma_h_log2: u8,
        /// Tightly packed Y plane (length `width * height`).
        pub y: Vec<u8>,
        /// Tightly packed Cb plane.
        pub cb: Vec<u8>,
        /// Tightly packed Cr plane.
        pub cr: Vec<u8>,
    }

    /// Outcome of asking libavcodec to decode a buffer.
    pub enum OracleResult {
        /// Library not loadable / decoder factory not registered.
        Unavailable,
        /// libavcodec rejected the input (parser or decoder error,
        /// or accepted-but-no-frame-emitted). Our decoder MUST also
        /// reject in this case (fuzz contract).
        Rejected,
        /// libavcodec produced a frame.
        Frame(DecodedYuv),
    }

    /// Selected libavcodec AVPixelFormat values we recognise. Values
    /// per the public `enum AVPixelFormat` in `libavutil/pixfmt.h` —
    /// stable since libavutil 50.
    const AV_PIX_FMT_YUV420P: i32 = 0;
    const AV_PIX_FMT_YUV422P: i32 = 4;
    const AV_PIX_FMT_YUV444P: i32 = 5;
    const AV_PIX_FMT_YUVJ420P: i32 = 12;
    const AV_PIX_FMT_YUVJ422P: i32 = 13;
    const AV_PIX_FMT_YUVJ444P: i32 = 14;

    fn chroma_log2(pix_fmt: i32) -> Option<(u8, u8)> {
        match pix_fmt {
            AV_PIX_FMT_YUV420P | AV_PIX_FMT_YUVJ420P => Some((1, 1)),
            AV_PIX_FMT_YUV422P | AV_PIX_FMT_YUVJ422P => Some((1, 0)),
            AV_PIX_FMT_YUV444P | AV_PIX_FMT_YUVJ444P => Some((0, 0)),
            _ => None,
        }
    }

    /// Feed `data` (Annex-B byte stream OR raw NAL bytes) to
    /// libavcodec's H.264 decoder, then drain a single frame. Returns:
    /// - `Unavailable` when libavcodec / `avcodec_find_decoder(H264)`
    ///   isn't installed
    /// - `Rejected` when the input fails to decode (any error path)
    /// - `Frame(...)` on the first picture produced
    ///
    /// All FFI is wrapped in `unsafe` blocks; resources (codec context,
    /// packet, frame) are freed before each early return via the
    /// closure-and-cleanup pattern.
    pub fn decode_h264(data: &[u8]) -> OracleResult {
        // ---- Function-pointer types --------------------------------
        // All signatures are taken verbatim from the public
        // <libavcodec/avcodec.h> + <libavutil/frame.h> headers, which
        // are documentation, not implementation. The opaque struct
        // pointers (AVCodec / AVCodecContext / AVPacket / AVFrame /
        // AVDictionary) are kept as `*mut c_void` since we never
        // dereference them from Rust — only pass them back to the C
        // side or read selected fields by documented byte offset.
        type FindDecoderFn = unsafe extern "C" fn(i32) -> *const c_void;
        type AllocContext3Fn = unsafe extern "C" fn(*const c_void) -> *mut c_void;
        type Open2Fn = unsafe extern "C" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32;
        type PacketAllocFn = unsafe extern "C" fn() -> *mut c_void;
        type FrameAllocFn = unsafe extern "C" fn() -> *mut c_void;
        type SendPacketFn = unsafe extern "C" fn(*mut c_void, *const c_void) -> i32;
        type ReceiveFrameFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
        type PacketFreeFn = unsafe extern "C" fn(*mut *mut c_void);
        type FrameFreeFn = unsafe extern "C" fn(*mut *mut c_void);
        type FreeContextFn = unsafe extern "C" fn(*mut *mut c_void);

        let Some(l) = lib() else {
            return OracleResult::Unavailable;
        };
        unsafe {
            let find_decoder: Symbol<FindDecoderFn> = match l.get(b"avcodec_find_decoder") {
                Ok(s) => s,
                Err(_) => return OracleResult::Unavailable,
            };
            let alloc_context3: Symbol<AllocContext3Fn> = match l.get(b"avcodec_alloc_context3") {
                Ok(s) => s,
                Err(_) => return OracleResult::Unavailable,
            };
            let open2: Symbol<Open2Fn> = match l.get(b"avcodec_open2") {
                Ok(s) => s,
                Err(_) => return OracleResult::Unavailable,
            };
            let packet_alloc: Symbol<PacketAllocFn> = match l.get(b"av_packet_alloc") {
                Ok(s) => s,
                Err(_) => return OracleResult::Unavailable,
            };
            let frame_alloc: Symbol<FrameAllocFn> = match l.get(b"av_frame_alloc") {
                Ok(s) => s,
                Err(_) => return OracleResult::Unavailable,
            };
            let send_packet: Symbol<SendPacketFn> = match l.get(b"avcodec_send_packet") {
                Ok(s) => s,
                Err(_) => return OracleResult::Unavailable,
            };
            let receive_frame: Symbol<ReceiveFrameFn> = match l.get(b"avcodec_receive_frame") {
                Ok(s) => s,
                Err(_) => return OracleResult::Unavailable,
            };
            let packet_free: Symbol<PacketFreeFn> = match l.get(b"av_packet_free") {
                Ok(s) => s,
                Err(_) => return OracleResult::Unavailable,
            };
            let frame_free: Symbol<FrameFreeFn> = match l.get(b"av_frame_free") {
                Ok(s) => s,
                Err(_) => return OracleResult::Unavailable,
            };
            let free_context: Symbol<FreeContextFn> = match l.get(b"avcodec_free_context") {
                Ok(s) => s,
                Err(_) => return OracleResult::Unavailable,
            };

            let codec = find_decoder(AV_CODEC_ID_H264);
            if codec.is_null() {
                return OracleResult::Unavailable;
            }
            let mut ctx = alloc_context3(codec);
            if ctx.is_null() {
                return OracleResult::Unavailable;
            }
            let mut pkt = packet_alloc();
            if pkt.is_null() {
                free_context(&mut ctx);
                return OracleResult::Unavailable;
            }
            let mut frame = frame_alloc();
            if frame.is_null() {
                packet_free(&mut pkt);
                free_context(&mut ctx);
                return OracleResult::Unavailable;
            }

            // Result is computed inside a closure so the cleanup
            // epilogue runs unconditionally on every early return.
            let result = (|| -> OracleResult {
                if open2(ctx, codec, std::ptr::null_mut()) < 0 {
                    return OracleResult::Rejected;
                }

                // Populate AVPacket.data + AVPacket.size by documented
                // byte offset. The AVPacket prefix layout is stable
                // since libavcodec 57 (the bump that turned it into a
                // ref-counted wrapper) and matches:
                //   off  0  *AVBufferRef  buf
                //   off  8  i64           pts
                //   off 16  i64           dts
                //   off 24  *u8           data
                //   off 32  i32           size
                // We only set data + size for a single non-ref-counted
                // input packet; the buf pointer stays NULL so libavcodec
                // copies the bytes itself before send_packet returns.
                const OFF_PKT_DATA: usize = 24;
                const OFF_PKT_SIZE: usize = 32;
                let pkt_bytes = pkt as *mut u8;
                (pkt_bytes.add(OFF_PKT_DATA) as *mut *const u8).write_unaligned(data.as_ptr());
                (pkt_bytes.add(OFF_PKT_SIZE) as *mut i32).write_unaligned(data.len() as i32);

                let sp = send_packet(ctx, pkt);
                if sp < 0 && sp != AVERROR_EAGAIN {
                    return OracleResult::Rejected;
                }
                // Send EOF (NULL packet) so the decoder flushes any
                // buffered frames. Spec: a NULL packet enters draining
                // mode (avcodec_send_packet docs).
                let _ = send_packet(ctx, std::ptr::null());

                let rc = receive_frame(ctx, frame);
                if rc < 0 {
                    return OracleResult::Rejected;
                }

                // Read AVFrame fields by documented byte offset. The
                // AVFrame prefix layout is stable since libavutil 55:
                //   off   0  *u8 x AV_NUM_DATA_POINTERS  data[8]
                //   off  64  i32 x AV_NUM_DATA_POINTERS  linesize[8]
                //   off  96  **u8                          extended_data
                //   off 104  i32                           width
                //   off 108  i32                           height
                //   off 112  i32                           nb_samples
                //   off 116  i32                           format
                // (AV_NUM_DATA_POINTERS == 8 since the libavutil 51 bump
                // — pre-dates everything we support.) The format field
                // for video frames is an AVPixelFormat.
                const OFF_FRAME_DATA: usize = 0;
                const OFF_FRAME_LINESIZE: usize = 64;
                const OFF_FRAME_WIDTH: usize = 104;
                const OFF_FRAME_HEIGHT: usize = 108;
                const OFF_FRAME_FORMAT: usize = 116;
                let f_bytes = frame as *const u8;
                let data_arr = f_bytes.add(OFF_FRAME_DATA) as *const *const u8;
                let linesize_arr = f_bytes.add(OFF_FRAME_LINESIZE) as *const i32;
                let width = (f_bytes.add(OFF_FRAME_WIDTH) as *const i32).read_unaligned();
                let height = (f_bytes.add(OFF_FRAME_HEIGHT) as *const i32).read_unaligned();
                let format = (f_bytes.add(OFF_FRAME_FORMAT) as *const i32).read_unaligned();
                if width <= 0 || height <= 0 {
                    return OracleResult::Rejected;
                }
                let Some((cw_log2, ch_log2)) = chroma_log2(format) else {
                    // Unsupported pixel format (10-bit, NV12, …). Treat
                    // as rejected rather than asserting equality on a
                    // format our decoder also doesn't produce.
                    return OracleResult::Rejected;
                };
                let w = width as usize;
                let h = height as usize;
                let cw = w >> cw_log2;
                let ch = h >> ch_log2;
                let y_ptr = data_arr.add(0).read_unaligned();
                let cb_ptr = data_arr.add(1).read_unaligned();
                let cr_ptr = data_arr.add(2).read_unaligned();
                let y_stride = linesize_arr.add(0).read_unaligned();
                let cb_stride = linesize_arr.add(1).read_unaligned();
                let cr_stride = linesize_arr.add(2).read_unaligned();
                if y_ptr.is_null()
                    || cb_ptr.is_null()
                    || cr_ptr.is_null()
                    || y_stride <= 0
                    || cb_stride <= 0
                    || cr_stride <= 0
                {
                    return OracleResult::Rejected;
                }
                let mut y = vec![0u8; w * h];
                for row in 0..h {
                    let src = y_ptr.add(row * y_stride as usize);
                    std::ptr::copy_nonoverlapping(src, y.as_mut_ptr().add(row * w), w);
                }
                let mut cb = vec![0u8; cw * ch];
                let mut cr = vec![0u8; cw * ch];
                for row in 0..ch {
                    let src_cb = cb_ptr.add(row * cb_stride as usize);
                    let src_cr = cr_ptr.add(row * cr_stride as usize);
                    std::ptr::copy_nonoverlapping(src_cb, cb.as_mut_ptr().add(row * cw), cw);
                    std::ptr::copy_nonoverlapping(src_cr, cr.as_mut_ptr().add(row * cw), cw);
                }
                OracleResult::Frame(DecodedYuv {
                    width: width as u32,
                    height: height as u32,
                    chroma_w_log2: cw_log2,
                    chroma_h_log2: ch_log2,
                    y,
                    cb,
                    cr,
                })
            })();

            frame_free(&mut frame);
            packet_free(&mut pkt);
            free_context(&mut ctx);
            result
        }
    }
}
