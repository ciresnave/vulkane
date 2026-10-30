# Third-party notices

**No third-party code is known to be in this crate.** Written when the crate moved from Unpopped
(`unpopped-slang`, source commit `213df4f896b5e0a4dd0db4ef84bce54ab18f9654`, Unpopped v0.34.11).

## Origin

- The imported tree (`Cargo.toml`, `LICENSE-APACHE`, `LICENSE-MIT`, `src/lib.rs`) is blob-identical to
  `crates/unpopped-slang/` at that commit (checked 2026-10-30 with `git rev-parse <rev>:<path>` on both sides).
  Its git history is preserved: 165 commits on `crates/unpopped-slang` and its pre-split path
  `crates/unpopped/src/slang.rs` (19 by Eric Evans, 140 by "Eric Evans' AI Agent", 6 by "Unpopped Agent").
- Before it existed as a crate the emitter was carved out of Unpopped's core, which itself was carved from Baracuda
  (same owner, same licence); those commits are in the history.
- Every source file declares `SPDX-License-Identifier: MIT OR Apache-2.0`; this repository's `.github/spdx_gate.py`
  passes (309 of 309 tracked source files).

## What was checked

A search of `src/lib.rs` for origin words (copyright, licence, llama, candle, "adapted from", "derived from",
"based on", URLs) found none that refer to third-party code (the substring "ported" matches only "unsupported").
No overlap measurement against third-party Slang/HLSL sources was run; the crate's output is Slang text, and HLSL/Slang intrinsic names (`exp`, `sqrt`, `max`) are language vocabulary, not
authored expression.

Limit: this is a scan and a lineage trace, not an overlap proof against every possible upstream.

## Tool used in CI (not shipped, not linked)

CI downloads the Slang compiler `slangc` to run the crate's compile test: `slang-2026.20-linux-x86_64.tar.gz` from
https://github.com/shader-slang/slang/releases/tag/v2026.20 (tag object `d8729bec562ca9d9c669f7b13263c06067cda40b`),
sha256 `e863e568c204810d6e8bc31852f056aef6763c55ea9925db2db7affa060347c9`. That digest is the one GitHub records for the
asset; it was re-computed on 2026-10-30 from a downloaded copy (83,407,874 bytes) and matched. The workflow re-checks it with
`sha256sum -c` on every run. Slang is Apache-2.0 with LLVM exception (its tarball carries `LICENSE` and `LICENSES/`); nothing from
it is copied into, linked into or published with this crate.
