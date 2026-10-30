# GLSL parser and emitter in Vulkane

**Status: design note, options first, no code.** ROADMAP row 3b. Written by the Vulkane lane 2026-10-30. Follows the
structure and rules of `slang-parser-emitter-handover.md` (D1 crate naming, D3 conformance inside the emitter crate, D5
release coupling) and `spirv-parser-emitter.md` (S4 oracle approach).

## Ruling this serves

CireSnave, 2026-10-30, verbatim: *"Vulkane should own GLSL, HLSL, and WGSL and create the parsers and emitters for those."*
Language owner writes parser AND emitter. Crates: `vulkane-glsl-parse`, `vulkane-glsl-emit`.

## What exists today (measured)

Vulkane `origin/main` `9fa47db`, 2026-10-30; crates.io API the same day; Vulkan SDK 1.4.350 tools.

| piece | state |
|---|---|
| Corpus | 27 `.glsl` in `vulkane-kernels-source/kernels/`, all `#version 450`, all compute. Extensions used (files): `GL_EXT_shader_explicit_arithmetic_types_int16` 11, `..._float16` 10, `GL_KHR_memory_scope_semantics` 5, `GL_KHR_cooperative_matrix` 5, `GL_KHR_shader_subgroup_arithmetic` 2. Shared memory / barrier / atomic / subgroup keywords appear in 19 of 27. Plus 2 `.comp` under `vulkane/examples/shaders/`. |
| Reference compiler | `glslangValidator -V -S comp`: 25 of 27 compile at its default target; the other 2 (`matvec.glsl`, `matvec_bf16_b.glsl`, subgroup ops, "requires SPIR-V 1.3") compile with `--target-env vulkan1.1`, which is what `compile.sh` passes. **27 of 27 compile at Vulkan 1.1.** |
| GLSL to SPIR-V in Vulkane | `safe::naga::compile_glsl` (naga `glsl-in`) and `safe::shaderc::compile_glsl` (glslang). Both are compilers, neither exposes a syntax tree. |
| Unpopped side | No GLSL parser or emitter (ownership map, 2026-09 inventory, now superseded by section 9). The C-family emitter helper `unpopped::cfamily` is public and used by `unpopped-slang`. |
| Candidate crates | `tree-sitter-glsl` 0.2.0 (MIT, 2025-03-16, 160k downloads); `glsl` 7.0.0 (BSD-3-Clause, 2023-12-22, 1.1M downloads) and `glsl-lang` 0.8.1 (BSD-3-Clause, 2025-09-15): pure-Rust GLSL parsers with their own AST. |

## Decisions needed, with a recommendation each

### G1. Parser technology

| option | for | against |
|---|---|---|
| **A. `tree-sitter-glsl`** (recommended) | The only option that fits Unpopped's `Frontend` seam unchanged: `parse` returns a `tree_sitter::Tree`, the same shape as the Slang parser, so `lift_elementwise` / `lift_reduction` / `lift_scan` apply as is. Error-tolerant CST, so a file with an unsupported extension still parses and the residue list reports it. MIT. | Grammar is community-maintained, 0.2.0; coverage of the GLSL extensions in the corpus (`uint16_t`, `float16_t`, `coopmat`, `layout(...) buffer`) is **unmeasured** and must be before committing. |
| B. `glsl-lang` | Full GLSL 4.60 preprocessor and AST in Rust, no C build. | Different tree type, so it cannot be a `Frontend` without core growing a second seam (same finding as SPIR-V consequence 1); BSD-3-Clause is compatible but adds a notice to ship; hard-fails on syntax it does not model rather than tolerating it. |
| C. Reuse `glslang` through `shaderc` | Authoritative: it is what the corpus is compiled with. | Output is SPIR-V, not a tree; cannot lift. Remains the *oracle*, not the parser. |

First step of the implementation PR, before any code is kept: parse all 27 corpus files plus the 2 `.comp` with
`tree-sitter-glsl` and count ERROR/MISSING nodes per file, with a control (a deliberately broken file must produce
errors). If coverage is poor, fall back to B and raise the seam question with the Unpopped lane.

### G2. What the parser lifts

Same position as SPIR-V S2: ship the language-level model first (entry point, `local_size`, buffer bindings and layouts,
`#extension` set, parameters of the `Params` block), which is useful for reflection and for the Fuel switch, and lift to
`unpopped::ir` (elementwise, reduction, scan) via the existing `lift_*` functions as the second step, with the same residue
rule: shared memory, barriers, atomics, subgroup and cooperative-matrix constructs (19 of 27 files here) are *refused as
residue, not approximated*.

### G3. Emitter

`unpopped::Backend`/`Lowering` implementation through `unpopped::cfamily`, as `unpopped-slang` does. Scope of the first
version: scalar elementwise, same declined set as the Slang emitter, so the two emitters are checked against each other on
the same `OpDef`s. GLSL specifics the emitter owns: `#version 450` header, `layout(local_size_x=...) in`, `layout(std430)
buffer` blocks (Slang's `RWStructuredBuffer` has no GLSL spelling to copy), explicit `uint16_t`/`float16_t` extension lines
when the dtype needs them, and `GL_KHR_*` extensions only when an op requires one.

### G4. Oracles

1. Validity: every emitted file compiles with `glslangValidator -V --target-env vulkan1.1 -S comp` and the result passes
   `spirv-val --target-env vulkan1.1`; negative control: a file with a type error is rejected.
2. Behaviour: run the compiled module through Vulkane's compute path and compare with `unpopped-cpu-c` on the same inputs
   (GPU tests via `gpu-run.ps1` or the self-hosted runner, never `#[ignore]`d).
3. Parser: for the 27-file corpus, `parse` followed by re-emit of the model must compile to SPIR-V that `spirv-val` accepts
   iff the original does (original 27 of 27).

### G5. Relationship to `vulkane::safe::{naga,shaderc}`

Those stay: they compile GLSL; the new crates read and write it. No duplication, no removal. `naga` `glsl-in` is
unsuitable as the parser (it consumes GLSL into naga IR and does not accept the extensions the corpus uses; this is the
expected cause, **not measured here**, and the G1 measurement should include it as a comparison column).

## Proposed sequence (each its own PR; none starts before this note merges)

1. Measurement PR: tree-sitter-glsl coverage over the corpus (G1), written into this note's table; decide A or B.
2. `vulkane-glsl-parse`: model + residue + reflection, corpus round-trip test (G4.3).
3. `vulkane-glsl-emit`: scalar elementwise, G4.1 and G4.2.
4. Lifting to `ir` via `lift_*` for the files that survive the residue rule.

## Not decided here

HLSL (3c) and WGSL (3d) get their own notes. Metal has no owner yet (ownership map). Whether Fuel's 27 GLSL kernels should
eventually be generated rather than hand-written is a Fuel decision.

## Provenance

Counts from `git ls-files` and file contents at `origin/main` `9fa47db`; compile verdicts from `glslangValidator` in the
Vulkan SDK 1.4.350 run over every `.glsl` file (25 pass at default target, 2 fail, 27 pass at Vulkan 1.1; the failure text was
read). Crate versions, dates and licences from the crates.io API on 2026-10-30. Not measured here, marked above: the
`tree-sitter-glsl` error rate over the corpus and `naga glsl-in`'s behaviour on the corpus' extensions.
