# WGSL parser and emitter in Vulkane

**Status: design note, options first, no code.** ROADMAP row 3d. Written by the Vulkane lane 2026-10-30. Same structure and
rules as `slang-parser-emitter-handover.md` (D1 naming, D3, D5), `spirv-parser-emitter.md` and `glsl-parser-emitter.md`.
Only what differs for WGSL is argued here.

## Ruling this serves

CireSnave, 2026-10-30, verbatim: *"Vulkane should own GLSL, HLSL, and WGSL and create the parsers and emitters for those."*
Crates: `vulkane-wgsl-parse`, `vulkane-wgsl-emit`.

## What exists today (measured)

Vulkane `origin/main` `9fa47db`, 2026-10-30; crates.io API the same day.

| piece | state |
|---|---|
| WGSL files | 5, all in `vulkane/examples/shaders/` (326 lines in total). Entry points: **7 `@vertex`, 8 `@fragment`, 0 `@compute`**; none uses `var<storage>` or `workgroup`. These are graphics examples, not kernels. |
| WGSL in the API | `safe::naga::compile_wgsl` (naga `wgsl-in` to `spv-out`). No tree, no emitter. |
| Kernel consumers | None. Fuel's kernels are Slang and GLSL; no portfolio project authors compute WGSL today. |
| Candidate crates | `wgsl-parse` 0.6.0 (MIT OR Apache-2.0, 2026-10-04, 714k downloads, wesl-rs); `naga` 30.0.1 (MIT OR Apache-2.0, 2026-08-22); `tree-sitter-wgsl-bevy` 0.1.4 (MIT, 2025-09-18, 16k); `tree-sitter-wgsl` 0.0.6 (CC0-1.0, **2022-06-13**, stale); `wgsl_parser` 0.5.0 (2024-06-01, 7k). |

The corpus finding matters more than the parser choice: **WGSL has no compute kernels to lift** in this portfolio, so
the lifting half of the contract (`lift_elementwise` / `lift_reduction` / `lift_scan`) has no real input here, and the
emitter would be built without a consumer.

## Decisions needed, with a recommendation each

### W1. Parser technology

| option | for | against |
|---|---|---|
| **A. `tree-sitter-wgsl-bevy`** (recommended if a `Frontend` is wanted) | Maintained (2025), MIT, CST, fits `Frontend` unchanged like Slang. | 0.1.4; coverage unmeasured. The older `tree-sitter-wgsl` is 2022, CC0 and predates current WGSL (for example `requires`/`diagnostic` directives): do not use. |
| B. `wgsl-parse` | Maintained Rust AST parser by the WESL project, tracks the spec, dual-licensed like ours, no C build. | Its AST is not a `tree_sitter::Tree`, so it cannot be a `Frontend` without the second core seam raised in the SPIR-V and GLSL notes; strict (hard errors, no error tolerance). |
| C. `naga` `wgsl-in` | Already a dependency; reference-quality WGSL front end. | Produces naga IR, not a syntax tree; dropping to naga IR loses source structure. Stays the *oracle* and the compile path. |

Recommendation: measure A and B on the 5 files plus a hand-written compute corpus (below) in the first PR; choose A if
its error rate on that set is zero, otherwise B and ask the Unpopped lane about a non-tree-sitter seam (a question that
SPIR-V already forces, so it should be asked once for all languages).

**Measured 2026-11-07 (step 1; `origin/main` `d7ccc45`; tree-sitter 0.27.2, `tree-sitter-wgsl-bevy` 0.1.4, `wgsl-parse` 0.6.0).**
Corpus: the 5 graphics examples plus 3 hand-written compute fixtures in `wgsl-measurement/` (elementwise, workgroup
reduction with `workgroupBarrier()`, scan with `enable f16;` / `requires` / `>>=`). All 8 parse AND validate under `naga`
30.0.1 `wgsl-in` (control: `fn ( {{` is rejected), so every rejection below is the parser's, not the input's.

| parser | files with errors (of 8) | what fails |
|---|---|---|
| A. `tree-sitter-wgsl-bevy` | **2** (both compute fixtures; the 5 graphics files and the elementwise fixture are clean) | (1) `workgroupBarrier();` as a statement: ERROR at every call site; (2) `d >>= 1u` compound assignment; (3) `requires <ext>;` directive. One-construct probes isolate each (control: `*=`, `enable f16;`, `struct` with and without a trailing `;`, `@workgroup_size(256, 1, 1)`, `for`, `var<workgroup>`, `var<uniform>` and `@builtin(local_invocation_index)` all parse clean). |
| B. `wgsl-parse` | **0** | nothing; it also rejects the garbage control. |

**Decision (W1): B, `wgsl-parse`.** The recommendation's own criterion (A only if its error rate is zero) is not met, and the
gap that decides it is `workgroupBarrier();`: every reduction and scan needs it, so A cannot read the kernels this crate
exists for. Patching A means vendoring a grammar (the HLSL precedent was to tolerate gaps, which is not an option when the
gap is the synchronisation primitive). Consequence, as predicted above: `wgsl-parse` is not a `tree_sitter::Tree`, so it
cannot be an Unpopped `Frontend` today. Step 2 therefore either (i) ships `vulkane-wgsl-parse` as a Rust-AST crate with its own
reflection and no `lift_*`, or (ii) waits on the Unpopped lane's answer to the non-tree-sitter seam question (the same one
SPIR-V and GLSL raise); this note recommends (i) first, since nothing consumes WGSL yet. Not measured: `wgsl-parse` on error
tolerance and spans beyond "it rejects garbage"; whether A's three gaps are fixed on `tree-sitter-wgsl-bevy` main (unreleased
commits were not tried); the harness is `wgsl-measurement/probe_harness.rs.txt` (A only; B is one `str::parse` call).

### W2. Corpus

None exists for compute. Recommendation: the parser PR adds a small hand-written WGSL compute corpus (elementwise,
reduction with `workgroup` memory and `workgroupBarrier()`, scan) and the 5 graphics examples as syntax coverage. No
Fuel kernel is ported to WGSL; that is a consumer decision and nobody has asked.

### W3. Emitter

`unpopped::Backend` / `Lowering` via `unpopped::cfamily` where it fits. WGSL is not C-family: declarations are `var<storage,
read_write> a: array<f32>;`, `fn f(...) -> T`, `let`/`var`, and `@compute @workgroup_size(256)` replaces `layout(...) in`. So
the emitter either needs `cfamily` to parameterise declaration order or it gets its own small printer; deciding which is the
first task of the emitter PR (read `cfamily`'s extension points before assuming). Types: WGSL has no `f64`, no 8-bit
integers, and `f16` only with `enable f16;`; the emitter must **decline** (typed decline, as `unpopped-slang` does) any
`OpDef` that needs a dtype WGSL lacks, not approximate it.

### W4. Oracles

Validity: the emitted file compiles through `naga` (`compile_wgsl`) and the SPIR-V passes `spirv-val`
(`--target-env vulkan1.1`); negative control: a type-error file is rejected. A second WGSL implementation (Tint) is not
installed in this environment, so only one compiler is available; say so in the test. Behaviour: run on the GPU and compare
with `unpopped-cpu-c` (GPU tests via `gpu-run.ps1` / self-hosted runner, never `#[ignore]`d). Parser: round-trip over the
corpus compiles iff the original does.

## Proposed sequence (each its own PR; none starts before this note merges)

1. ~~Measurement PR: A vs B over the corpus (W1, W2), decision recorded here.~~ **Done 2026-11-07**: B (`wgsl-parse`); the 3 compute fixtures are committed in `wgsl-measurement/` and step 2 adopts them.
2. `vulkane-wgsl-parse` with the compute corpus.
3. `vulkane-wgsl-emit` for the dtypes WGSL has, typed declines for the rest (W3).
4. Lifting once at least one consumer exists; until then this row stays parser plus emitter only.

## Not decided here

Whether WGSL belongs at the end of the queue given it has no consumer. CireSnave's ruling assigns it; this note only reports
that nothing consumes it, so the order (SPIR-V, GLSL, HLSL, WGSL) should keep WGSL last.

## Provenance

File and stage counts from `git ls-files` and grep over the 5 files at `origin/main` `9fa47db` (positive control: the same
grep finds 15 entry points). Crate versions, dates, licences and repositories from the crates.io API on 2026-10-30. Not
measured, marked above: parser error rates, `cfamily`'s extension points, `f16` handling in naga.
