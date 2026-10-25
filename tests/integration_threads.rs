//! A decoder granted a worker budget through
//! `Decoder::set_execution_context` must emit exactly the frames the
//! single-threaded decoder emits, for every budget.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, ExecutionContext, Frame, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;

fn drain(dec: &mut H264CodecDecoder, frames: &mut Vec<Vec<u8>>) {
    while let Ok(Frame::Video(v)) = dec.receive_frame() {
        let mut out = Vec::new();
        for p in v.image_planes() {
            out.extend_from_slice(&p.data);
        }
        frames.push(out);
    }
}

fn decode_all(bytes: &[u8], threads: usize) -> Vec<Vec<u8>> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    dec.set_execution_context(&ExecutionContext::with_threads(threads));
    let packet = Packet::new(0, TimeBase::new(1, 25), bytes.to_vec()).with_pts(0);
    dec.send_packet(&packet).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    drain(&mut dec, &mut frames);
    frames
}

/// Split an Annex B byte stream into its NAL units (each with its
/// start code).
fn split_nals(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push(if i > 0 && data[i - 1] == 0 { i - 1 } else { i });
            i += 3;
        } else {
            i += 1;
        }
    }
    (0..starts.len())
        .map(|k| &data[starts[k]..starts.get(k + 1).copied().unwrap_or(data.len())])
        .collect()
}

/// Feed one NAL unit per packet and drain after every packet, the way
/// a demuxer-driven pipeline does.
fn decode_streamed(bytes: &[u8], threads: usize) -> Vec<Vec<u8>> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    dec.set_execution_context(&ExecutionContext::with_threads(threads));
    let mut frames = Vec::new();
    for (i, nal) in split_nals(bytes).into_iter().enumerate() {
        let packet = Packet::new(0, TimeBase::new(1, 25), nal.to_vec()).with_pts(i as i64);
        let _ = dec.send_packet(&packet);
        drain(&mut dec, &mut frames);
    }
    dec.flush().expect("flush");
    drain(&mut dec, &mut frames);
    frames
}

fn assert_thread_invariant(name: &str) {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).expect("read fixture");
    assert_thread_invariant_bytes(name, &bytes, &[2, 3, 4, 8]);
}

fn assert_same(name: &str, what: &str, serial: &[Vec<u8>], threaded: &[Vec<u8>]) {
    assert_eq!(serial.len(), threaded.len(), "{name}: frame count {what}");
    for (i, (a, b)) in serial.iter().zip(threaded).enumerate() {
        assert!(a == b, "{name}: frame {i} differs {what}");
    }
}

fn assert_thread_invariant_bytes(name: &str, bytes: &[u8], budgets: &[usize]) {
    let serial = decode_all(bytes, 1);
    assert!(!serial.is_empty(), "{name}: no frames decoded");
    assert_same(
        name,
        "streamed serially",
        &serial,
        &decode_streamed(bytes, 1),
    );
    for &threads in budgets {
        let what = format!("at {threads} threads");
        assert_same(name, &what, &serial, &decode_all(bytes, threads));
        let what = format!("streamed at {threads} threads");
        assert_same(name, &what, &serial, &decode_streamed(bytes, threads));
    }
}

/// A budget change and a `reset` in the middle of a stream leave the
/// decoder producing exactly the frames a fresh decoder produces.
#[test]
fn budget_change_and_reset_mid_stream() {
    let path = format!(
        "{}/tests/fixtures/x264_testsrc_128x96_25f.h264",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).expect("read fixture");
    let serial = decode_all(&bytes, 1);
    let nals = split_nals(&bytes);

    // Budget 8 → 1 → 4 while streaming.
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    let mut frames = Vec::new();
    for (i, nal) in nals.iter().enumerate() {
        let budget = match i * 3 / nals.len() {
            0 => 8,
            1 => 1,
            _ => 4,
        };
        dec.set_execution_context(&ExecutionContext::with_threads(budget));
        let packet = Packet::new(0, TimeBase::new(1, 25), nal.to_vec()).with_pts(i as i64);
        let _ = dec.send_packet(&packet);
        drain(&mut dec, &mut frames);
    }
    dec.flush().expect("flush");
    drain(&mut dec, &mut frames);
    assert_same("budget change", "", &serial, &frames);

    // Half the stream at budget 8, reset, then the whole stream.
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    dec.set_execution_context(&ExecutionContext::with_threads(8));
    for nal in &nals[..nals.len() / 2] {
        let _ = dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), nal.to_vec()));
        let mut sink = Vec::new();
        drain(&mut dec, &mut sink);
    }
    dec.reset().expect("reset");
    let packet = Packet::new(0, TimeBase::new(1, 25), bytes.to_vec()).with_pts(0);
    dec.send_packet(&packet).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    drain(&mut dec, &mut frames);
    assert_same("reset", "", &serial, &frames);
}

#[test]
fn x264_p_b_stream_is_thread_invariant() {
    assert_thread_invariant("x264_testsrc_128x96_25f.h264");
}

#[test]
fn mbaff_stream_is_thread_invariant() {
    assert_thread_invariant("mbaff_iframe_128x96.h264");
}

/// Every stream of the `docs/video/h264/fixtures/` corpus (PAFF field
/// pictures, MBAFF, 4:2:2 / 4:4:4, 10-bit, multi-slice, weighted and
/// temporal-direct B, lossless, scaling matrices, ...) decodes
/// identically at budgets 2 and 8. Skipped when the docs corpus is not
/// checked out next to the crate (CI of the crate alone).
#[test]
fn docs_fixture_corpus_is_thread_invariant() {
    let root = std::path::Path::new("../../docs/video/h264/fixtures");
    let Ok(entries) = std::fs::read_dir(root) else {
        eprintln!(
            "skip docs_fixture_corpus_is_thread_invariant: {} missing",
            root.display()
        );
        return;
    };
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("input.h264").is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert!(
        !names.is_empty(),
        "docs fixture corpus has no input.h264 files"
    );
    for name in &names {
        let bytes = std::fs::read(root.join(name).join("input.h264")).expect("read fixture");
        assert_thread_invariant_bytes(name, &bytes, &[2, 4, 8]);
    }
    eprintln!(
        "docs fixture corpus: {} streams thread-invariant",
        names.len()
    );
}

/// The staged JVT conformance streams: PAFF (CABAC + CAVLC), PAFF + MBAFF,
/// 4:0:0 monochrome and the 14-bit separate-colour-plane 4:4:4 stream.
/// Skipped when the docs corpus is not checked out.
#[test]
fn staged_conformance_streams_are_thread_invariant() {
    let dir = std::env::var("OXIDEAV_H264_CONFORMANCE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("../../docs/video/h264/conformance"));
    if !dir.is_dir() {
        eprintln!(
            "skip staged_conformance_streams_are_thread_invariant: {} missing",
            dir.display()
        );
        return;
    }
    for file in [
        "camp_mot_picaff0_full.26l",
        "CAPAMA3_Sand_F.264",
        "CAPA1_TOSHIBA_B.264",
        "CVPA1_TOSHIBA_B.264",
        "HPCAMOLQ_BRCM_B.264",
        "HPCVMOLQ_BRCM_B.264",
        "PPH444P9_Mitsubishi_A.264",
    ] {
        let path = dir.join(file);
        if !path.is_file() {
            eprintln!("skip {file}: not staged");
            continue;
        }
        let bytes = std::fs::read(&path).expect("read conformance stream");
        assert_thread_invariant_bytes(file, &bytes, &[2, 4, 8]);
    }
}

/// Frames plus the decode-error count of a whole-stream decode that
/// tolerates rejected packets.
fn decode_tolerant(bytes: &[u8], threads: usize, streamed: bool) -> (Vec<Vec<u8>>, u64) {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    dec.set_execution_context(&ExecutionContext::with_threads(threads));
    let mut frames = Vec::new();
    if streamed {
        for (i, nal) in split_nals(bytes).into_iter().enumerate() {
            let packet = Packet::new(0, TimeBase::new(1, 25), nal.to_vec()).with_pts(i as i64);
            let _ = dec.send_packet(&packet);
            drain(&mut dec, &mut frames);
        }
    } else {
        let packet = Packet::new(0, TimeBase::new(1, 25), bytes.to_vec()).with_pts(0);
        let _ = dec.send_packet(&packet);
    }
    let _ = dec.flush();
    drain(&mut dec, &mut frames);
    (frames, dec.decode_error_count())
}

fn assert_tolerant_invariant(name: &str, bytes: &[u8]) {
    for streamed in [false, true] {
        let (serial, errors) = decode_tolerant(bytes, 1, streamed);
        eprintln!("{name}: {} frames, {errors} skipped slices", serial.len());
        for threads in [2, 8] {
            let (frames, e) = decode_tolerant(bytes, threads, streamed);
            let what = format!("at {threads} threads (streamed: {streamed})");
            assert_same(name, &what, &serial, &frames);
            assert_eq!(errors, e, "{name}: decode error count {what}");
        }
    }
}

/// The archived fuzzer inputs (broken slices, unavailable intra
/// neighbours, empty-DPB inter slices, ...) take the same error paths
/// — same frames, same skipped-slice count — at every budget.
#[test]
fn fuzz_regressions_are_thread_invariant() {
    let dir = format!(
        "{}/tests/fixtures/fuzz_regressions",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .expect("fuzz fixtures")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    names.sort();
    for path in names {
        let bytes = std::fs::read(&path).expect("read fixture");
        assert_tolerant_invariant(&path.display().to_string(), &bytes);
    }
}

/// Corrupted copies of a P / B stream (bytes overwritten at
/// pseudo-random positions inside the slice data) decode identically
/// at every budget: slices that fail to parse or reconstruct are
/// skipped at the same macroblock, pictures left incomplete are
/// dropped the same way, and the pictures referencing them follow.
#[test]
fn corrupted_streams_are_thread_invariant() {
    let path = format!(
        "{}/tests/fixtures/x264_testsrc_128x96_25f.h264",
        env!("CARGO_MANIFEST_DIR")
    );
    let clean = std::fs::read(&path).expect("read fixture");
    let mut seed = 0x2545_f491_u32;
    for case in 0..24 {
        let mut bytes = clean.clone();
        for _ in 0..1 + case % 4 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Keep the first parameter sets intact.
            let pos = 64 + (seed as usize >> 8) % (bytes.len() - 64);
            bytes[pos] ^= (seed as u8) | 1;
        }
        assert_tolerant_invariant(&format!("corruption case {case}"), &bytes);
    }
}
