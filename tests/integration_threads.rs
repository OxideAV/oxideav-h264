//! A decoder granted a worker budget through
//! `Decoder::set_execution_context` must emit exactly the frames the
//! single-threaded decoder emits, for every budget.

use oxideav_core::Decoder as _;
use oxideav_core::{CodecId, ExecutionContext, Frame, Packet, TimeBase};
use oxideav_h264::h264_decoder::H264CodecDecoder;

fn decode_all(bytes: &[u8], threads: usize) -> Vec<Vec<u8>> {
    let mut dec = H264CodecDecoder::new(CodecId::new("h264"));
    dec.set_execution_context(&ExecutionContext::with_threads(threads));
    let packet = Packet::new(0, TimeBase::new(1, 25), bytes.to_vec()).with_pts(0);
    dec.send_packet(&packet).expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = Vec::new();
    while let Ok(Frame::Video(v)) = dec.receive_frame() {
        let mut out = Vec::new();
        for p in v.image_planes() {
            out.extend_from_slice(&p.data);
        }
        frames.push(out);
    }
    frames
}

fn assert_thread_invariant(name: &str) {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).expect("read fixture");
    assert_thread_invariant_bytes(name, &bytes, &[2, 3, 4, 8]);
}

fn assert_thread_invariant_bytes(name: &str, bytes: &[u8], budgets: &[usize]) {
    let serial = decode_all(bytes, 1);
    assert!(!serial.is_empty(), "{name}: no frames decoded");
    for &threads in budgets {
        let threaded = decode_all(bytes, threads);
        assert_eq!(
            serial.len(),
            threaded.len(),
            "{name}: frame count at {threads} threads"
        );
        for (i, (a, b)) in serial.iter().zip(&threaded).enumerate() {
            assert!(a == b, "{name}: frame {i} differs at {threads} threads");
        }
    }
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
        assert_thread_invariant_bytes(name, &bytes, &[2, 8]);
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
        assert_thread_invariant_bytes(file, &bytes, &[2, 8]);
    }
}
