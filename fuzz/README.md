# oxideav-h264 fuzz targets

Three `cargo-fuzz` (libFuzzer) targets live in `fuzz_targets/`:

* `panic_free_decode` — feeds raw bytes through `H264CodecDecoder::send_packet` /
  `receive_frame` along two parallel paths (Annex-B byte stream and AVCC
  framing with a synthetic `avcDecoderConfigurationRecord`). Contract: no
  call may panic; every error is surfaced as `Result::Err`.
* `ffmpeg_oracle_decode` — cross-validates `H264CodecDecoder` against a
  `dlopen`-loaded `libavcodec` (when the shared library is present at
  runtime). The harness skips silently when libavcodec is unavailable.
* `sei_payload` — exercises the §D.2 / §G.13.2 SEI parser surface with
  arbitrary bytes under varied `SeiContext` shapes.

## Artifacts directory

`artifacts/<target>/` is libFuzzer's quarantine bucket: each surviving
input that triggered a panic, an OOM, or another abort is dropped there
with a `crash-`, `oom-`, `leak-`, or `timeout-` filename prefix. Inputs
are committed into git on first observation so future runs can replay
them as regression cases (the libFuzzer `cargo fuzz run <target>
<artifact-file>` invocation re-feeds the single file to the harness in
`-runs=1` mode).

### Round-194 sweep (2026-05-31)

Four `oom-*` artifacts under `artifacts/panic_free_decode/`
(`1cf28e07…`, `35149f63…`, `87d75308…`, `e4eded82…`) were preserved
from earlier rounds when the decoder could be driven into >2 GiB
allocations on adversarial Annex-B inputs. After the round-177 §D.2.20
`num_slice_groups_in_set_minus1` bound and the round-187 §8.2.1 POC
`i64` staging, each of the four artifacts was re-fed through the
current `panic_free_decode` binary:

```
$ for f in fuzz/artifacts/panic_free_decode/oom-*; do
    /tmp/oxideav-h264-r194-target/.../panic_free_decode -runs=1 $f
  done
# all four: "Executed … in 0 ms", exit 0
```

All four exit cleanly within libFuzzer's default 2 GiB RSS budget and
were deleted. The directory is now empty.

The next time `cargo fuzz run panic_free_decode` lands a fresh `oom-*`
under it, this section is the place to record the surviving artifact
plus the clause / fixup that's expected to close it.

## Re-running an artifact locally

Requires nightly Rust (cargo-fuzz needs `-Zsanitizer=address`):

```
PATH=$HOME/.rustup/toolchains/nightly-aarch64-apple-darwin/bin:$PATH \
RUSTUP_TOOLCHAIN=nightly \
CARGO_TARGET_DIR=/tmp/oxideav-h264-fuzz-target \
cargo fuzz run panic_free_decode fuzz/artifacts/panic_free_decode/<file>
```

The default libFuzzer settings (no overrides) reproduce the same OOM /
panic budget CI applies.
