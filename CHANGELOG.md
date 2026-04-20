# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- **Entire previous implementation.** The pre-rewrite codebase decodes
  many H.264 streams but accumulates spec-deviations on real-world
  content that bug-by-bug fixing stopped paying off; archived to the
  [`old`](https://github.com/OxideAV/oxideav-h264/tree/old) branch
  for reference.

### Added

- Empty crate skeleton with the standard CI / release-plz / LICENSE
  layout. `register()` is a no-op so the workspace aggregator keeps
  building.
- `README.md` is now a spec coverage matrix that grows with the
  rewrite.

### Notes

The rewrite is driven by ITU-T Rec. H.264 | ISO/IEC 14496-10 (2024-08
edition) as the single authoritative source. No external decoder code
is consulted while writing the implementation; conformance is verified
by behavioural diff against an external reference decoder run
separately.
