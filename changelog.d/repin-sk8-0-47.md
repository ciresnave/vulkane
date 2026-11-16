kind: **BREAKING**: repin unpopped to ~0.47.0 (sk8) and fishstringer to ~0.2.0; `unpopped::convert` is now the `unpopped-convert` crate
---
`unpopped`, `unpopped-vocab`, `unpopped-cpu-c` 0.43.2 -> ~0.47.0 and `fishstringer-slang-emit`/`-slang-parse`/`-kernels-source` -> ~0.2.0 across the emitter crates. `unpopped`s `convert` feature moved to the `unpopped-convert` crate, so `vulkane-hlsl-emit` and `vulkane-hlsl-parse` depend on it and import `unpopped_convert::`. Verified: `cargo check --workspace --all-targets --all-features`; tests under gpu-run (see PR).
