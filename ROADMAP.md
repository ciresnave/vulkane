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
| 1c | `softmax_f64.slang` compiles to SPIR-V that `spirv-val` rejects (`GLSL.std.450 Exp` on `double`) | validation fixed (#119); numerical test + second `exp` defect fixed (PR open, 0.19.4) | pre-existing in Fuel's `.spv` at `e3ff3427`; the test found `pow2_int` garbage for results below 2^-1022 (x in [-745,-708.4]); `unary_f64` shares it |
| 1d | Characterisation test for `hoist_extinst_imports.py`, then split `hoist_imports` (Codacy: CC 34, 75 lines; subprocess findings accepted) | step 1 done: 7 characterisation tests + CI job; step 2 done (PR open): `hoist_imports` split into `_function_state`, `_collect_imports`, `_last_match`, `_insertion_point`, `_apply_aliases` (+2 tests, 9 total); behaviour identical to the old function on 178 corpus modules + 7 synthetic cases | disposition on #105 |
| 2 | Slang parser + emitter on Unpopped's contract | emitter imported as `vulkane-slang-emit` (#113 merged, 0.16.10); parser = `vulkane-slang-parse` (#117 merged, 0.16.14; unpopped 0.35.0 / tree-sitter 0.27 in 0.17.0) | handover note merged (#107); Unpopped lane removes `unpopped-slang` after the first publish |
| 2c | `vulkane-slang-emit`: Codacy complexity findings on imported code (`slang_unary_fp` CC 32, `slang_binary_fp` CC 16, `slang_ctype` 12, `body_has_params` 10, `lower` 55 lines, file 908 lines) and the `std::env::temp_dir` use in the slangc test | `temp_dir` part done (PR open: `tempfile::tempdir()`); complexity items todo (deferred; disposition on #113) | exhaustive op-to-spelling `match` tables moved unchanged; refactor only with the emitter work (E1c/E2), not in the import |
| 2b | Run `vulkane-slang-emit`'s `slangc`-gated test (`emitted_slang_compiles_with_slangc`, `#[ignore]`d) on the self-hosted runner | done (PR open): hosted ubuntu job, pinned sha256-verified slangc 2026.20, `VULKANE_REQUIRE_SLANGC`; `#[ignore]` removed | portfolio rule: never leave runnable tests ignored |
| 2d | Conformance cells for `vulkane-slang-emit` (the 3 Unpopped cells, D3 of the handover note) | done (PR open) | `tests/`: cell_token_is_not_a_computation (2), dtype_lowering_coverage (7), lower_kernel_default_is_the_typed_decline (3); Unpopped removes its copies after this merges |
| 2a | Unpopped ROADMAP row E2: 8 unary + copysign/nextafter polyfills in `unpopped-slang` (Slang src + tests) | blocked (row 2 design note agreed with the Unpopped lane and merged) | PM notice 2026-10-30: Slang emitter work, now Vulkane's; do not edit Unpopped before that note merges |
| 3a | SPIR-V parser + emitter | design note merged (#108); `vulkane-spirv-parse` merged (#120, 0.17.2: model, reflection, 178-module round-trip); spirv-val verdict + invalid-module tests merged (#121, 0.17.3) | `vulkane-spirv-emit` (PR open, needs unpopped 0.37.1 `with_words`, published 2026-10-31): scalar elementwise f32/f64/i32/u32, every op `spirv-val`-clean; NEXT: behaviour vs `unpopped-cpu-c` on a device (G4.2), then the Slang-vs-SPIR-V cross-check |
| 3e | Bump stale deps: `naga` 29 to 30.0.1, `shaderc` 0.8 to 0.10.1 | done (#109) | found while writing 3a; CireSnave's latest-versions rule |
| 3b | GLSL parser + emitter | design note merged (#110, `docs/design/glsl-parser-emitter.md`) | G1 measured 2026-10-31 (note section): `tree-sitter-glsl` 0.2.0 clean on 0/29 (2 grammar gaps; 26/29 without the bare `layout(local_size_*) in;` lines), `glsl-lang` 13/29, naga 16/29; decision on grammar fix open |
| 3c | HLSL parser + emitter | design note merged (#111, `docs/design/hlsl-parser-emitter.md`) | no HLSL corpus exists; 131/147 Slang kernels compile as HLSL with dxc (proxy) |
| 3d | WGSL parser + emitter | design note merged (#112, `docs/design/wgsl-parser-emitter.md`) | no compute WGSL exists in the portfolio (5 graphics files); keep last in order |
