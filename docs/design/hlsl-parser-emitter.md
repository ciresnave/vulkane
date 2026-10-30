# HLSL parser and emitter in Vulkane

**Status: design note, options first, no code.** ROADMAP row 3c. Written by the Vulkane lane 2026-10-30. Same structure and
rules as `slang-parser-emitter-handover.md` (D1 naming, D3 conformance in the emitter crate, D5 release coupling),
`spirv-parser-emitter.md` (S4 oracles) and `glsl-parser-emitter.md` (G1-G5). Only what differs for HLSL is argued here.

## Ruling this serves

CireSnave, 2026-10-30, verbatim: *"Vulkane should own GLSL, HLSL, and WGSL and create the parsers and emitters for those."*
Crates: `vulkane-hlsl-parse`, `vulkane-hlsl-emit`.

## What exists today (measured)

Vulkane `origin/main` `9fa47db`, 2026-10-30; crates.io API the same day; Vulkan SDK 1.4.350 (`dxc` 1.9.0.5347).

| piece | state |
|---|---|
| HLSL files in the repo | **0** (`git ls-files` for `.hlsl`/`.fx`; the same query lists 5 `.wgsl`, so the filter works). |
| HLSL in the API | `safe::shaderc` documents HLSL input (glslang's HLSL front end); no HLSL tree, no emitter. |
| Proxy corpus | Slang is largely an HLSL superset. Compiling the 147 `.slang` kernels as HLSL with `dxc -T cs_6_2 -E main -spirv -fspv-target-env=vulkan1.1 -enable-16bit-types`: **131 accepted, 16 rejected**. First error of each reject: 12 "modifiers must appear before type", 3 "missing entry point definition" (entry not named `main`), 1 "unknown type name 'var'", all Slang-only syntax. Control: a deliberately broken file is rejected. One accepted output was also passed through `spirv-val --target-env vulkan1.1` (valid); not done for all 131. |
| Candidate crates | `tree-sitter-hlsl` 0.2.0 (MIT, 2025-03-23, 75k downloads, repository theHamsta/tree-sitter-hlsl); `hassle-rs` 0.12.0 (MIT, 2025-06-13, 6.2M downloads, FFI to DXC). `tree-sitter-slang` (the grammar behind Unpopped's Slang parser, 0.3.1) is by the same repository owner; whether the two grammars share lineage is **not measured**. |

## Decisions needed, with a recommendation each

### H1. Parser technology

| option | for | against |
|---|---|---|
| **A. `tree-sitter-hlsl`** (recommended) | Fits `Frontend` unchanged (CST), exactly like the Slang parser; error-tolerant. MIT. | 0.2.0 community grammar; HLSL-specific constructs in kernels (`[numthreads]`, `RWStructuredBuffer<T>`, `groupshared`, `register(...)`, semantics `: SV_...`) unmeasured. |
| B. `hassle-rs` / DXC | Authoritative front end and a SPIR-V (and DXIL) back end. | A compiler FFI that needs `dxcompiler` at runtime; no CST. It is the *oracle*, not the parser, and a runtime DLL dependency in a library crate is the opposite of the dependency-light emitter rule. |
| C. Reuse the Slang parser | Slang accepts most HLSL (not measured here). | Wrong ownership: it would label HLSL as Slang and inherit Slang residue; a lie in diagnostics. |

First step of the implementation PR: parse the proxy corpus (the 131 DXC-accepted files, plus 3-5 hand-written pure-HLSL
files with `groupshared`, semantics and `register`) with `tree-sitter-hlsl` and count ERROR/MISSING nodes, with a control.

### H2. Corpus

HLSL has no corpus in the portfolio. Recommendation: **the 131 DXC-accepted `.slang` files are the working corpus**,
labelled as such (they are Slang files that happen to be HLSL), plus a small hand-written pure-HLSL set added in the parser
PR so shader-model features Slang does not exercise (semantics, `cbuffer`, `register`) are covered. Fuel is not asked to
write HLSL.

### H3. Emitter

`unpopped::Backend` / `Lowering` via `unpopped::cfamily`, scalar elementwise first, same declined set as the Slang
emitter. The Slang and HLSL spellings are close (`RWStructuredBuffer<float>`, `[numthreads]`, `SV_DispatchThreadID`), so the
two emitters should differ by a small table; the design check is that HLSL-specific output (`register(uN)`, `cbuffer`, `: SV_`
semantics, `-enable-16bit-types` when `half`/`int16_t` appear) is a thin layer over shared C-family emission, not a copy.

### H4. Oracles

Same three as GLSL G4 with DXC instead of glslang: emitted file compiles with `dxc -T cs_6_x -spirv -fspv-target-env=vulkan1.1`
and the SPIR-V passes `spirv-val` (negative control as above); run on the GPU and compare to `unpopped-cpu-c` (GPU tests via
`gpu-run.ps1` / self-hosted runner, never `#[ignore]`d); parser round-trip over the corpus compiles iff the original does.
`shaderc`'s HLSL path is a second compiler for the same check; if they disagree, report both.

## Proposed sequence (each its own PR; none starts before this note merges)

1. Measurement PR: `tree-sitter-hlsl` coverage over the corpus (H1); decision recorded in this note.
2. `vulkane-hlsl-parse` with the pure-HLSL additions to the corpus (H2).
3. `vulkane-hlsl-emit` scalar elementwise (H3, H4).
4. Lifting via `lift_*` for files that survive the residue rule.

## Not decided here

WGSL (3d). HLSL's DXIL output (not a Vulkan target). Whether Fuel ever authors HLSL (no current consumer).

## Provenance

File counts from `git ls-files` at `origin/main` `9fa47db`. The 131/16 split and error reasons come from running
`dxc.exe` (1.9.0.5347, Vulkan SDK 1.4.350) over every `.slang` in `vulkane-kernels-source/kernels/` on 2026-10-30 with the
flags quoted; the first line of the first error was counted per rejected file. Crate versions, dates, licences and
repositories from the crates.io API the same day. Not measured, marked above: `tree-sitter-hlsl` error rate, grammar lineage
with `tree-sitter-slang`, `spirv-val` over all 131 outputs.
