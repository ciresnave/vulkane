# Vulkane roadmap

Owner: the Vulkane lane. Priorities come from `C:\Projects\PORTFOLIO-ROADMAP.md`; version numbers are
allocated by the PM at gate time. Status values: `todo`, `doing`, `blocked (<on what>)`, `done`.

Scope ruling (CireSnave, 2026-10-30, verbatim): *"Vulkane should own SPIR-V and create a parser and
emitter. ... Vulkane should own GLSL, HLSL, and WGSL and create the parsers and emitters for those."*
and *"Okay.  Vulkane now owns Slang in our ecosystem.  We should move the Slang kernels from Fuel to
Vulkane, shouldn't we?  Fuel can import them still."* Whichever project owns a language writes its parser
AND its emitter; Unpopped holds neither (contract: Unpopped's `Frontend` / `Backend` / `Lowering`; see
`Unpopped/docs/parser-emitter-ownership-map.md`).

| # | Row | Status | Notes |
|---|-----|--------|-------|
| 0 | This ROADMAP.md | done | in the kernel-move PR |
| 1 | Move Fuel's Slang kernels into `vulkane-kernels-source` | merged (#105, 0.16.3); awaiting first crates.io publish (PM) | history-preserving; provenance traced per file; PM publishes, then the Fuel lane switches to the published crate and deletes its copy |
| 1a | Trace third-party origins in the moved files (webgpu-blas in `matmul.slang`; llama.cpp `get_scale_min_k4` in `dequant_q4_km.slang`; others) | done (see the crate's THIRD-PARTY-NOTICES.md) | notices travel with the files |
| 1b | Make the SPIR-V compile step output-dir configurable (it hard-codes `../../fuel-vulkan-kernels/spv`) | done | PTX/cubin stays with Fuel (`fuel-cuda-kernels`) |
| 1c | `softmax_f64.slang` compiles to SPIR-V that `spirv-val` rejects (`GLSL.std.450 Exp` on `double`) | todo (issue #106) | pre-existing in Fuel's `.spv` at `e3ff3427`; Fuel lane told |
| 1d | Characterisation test for `hoist_extinst_imports.py`, then split `hoist_imports` (Codacy: CC 34, 75 lines; subprocess findings accepted) | todo | disposition on #105 |
| 2 | Slang parser + emitter on Unpopped's contract | blocked (design note with the Unpopped lane first) | `unpopped-slang` (emitter) and the Slang front end in Unpopped's `convert.rs` exist today |
| 2a | Unpopped ROADMAP row E2: 8 unary + copysign/nextafter polyfills in `unpopped-slang` (Slang src + tests) | blocked (row 2 design note agreed with the Unpopped lane and merged) | PM notice 2026-10-30: Slang emitter work, now Vulkane's; do not edit Unpopped before that note merges |
| 3a | SPIR-V parser + emitter | design note in PR (`docs/design/spirv-parser-emitter.md`) | S1-S5 options; reuse `rspirv`; parser first |
| 3e | Bump stale deps: `naga` 29 to 30.0.1, `shaderc` 0.8 to 0.10.1 | todo | found while writing 3a; CireSnave's latest-versions rule |
| 3b | GLSL parser + emitter | todo | one language per PR |
| 3c | HLSL parser + emitter | todo | |
| 3d | WGSL parser + emitter | todo | existing `naga` feature is the starting point |
