# GLSL parser and emitter in Vulkane

**Status: design note, options first, no code; G1 measured 2026-10-31 (decision open).** ROADMAP row 3b. Written by the Vulkane lane 2026-10-30. Follows the
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

## G1 measurement (2026-10-31)

Run at Vulkane `origin/main` `56e5dcb` over the 27 `vulkane-kernels-source/kernels/*.glsl` plus the 2
`vulkane/examples/shaders/*.comp` (29 files). Parsers: `tree-sitter-glsl` 0.2.0 on `tree-sitter` 0.27.1, `glsl-lang`
0.8.1 (`TranslationUnit::parse`), naga 30.0.1 `glsl-in` (`Frontend::parse`, `ShaderStage::Compute`). tree-sitter
errors are ERROR plus MISSING nodes found by a full tree walk. Control: a deliberately broken string
(`void main( { int = ; }}}`) gives 2 ERROR nodes in tree-sitter and a parse error in `glsl-lang`.

| parser | files clean | notes |
|---|---|---|
| `tree-sitter-glsl` 0.2.0, files as they are | **0 of 29** | every one of the 29 files has an ERROR at the `;` of the bare `layout(local_size_x = N, ...) in;` declaration; the 3 `flash_attn_split_*` files have a second ERROR at `(float(kj) - float(qi))`, a parenthesised expression that begins with a keyword type |
| `tree-sitter-glsl`, the bare `layout(local_size_*) in;` lines deleted | 26 of 29 | the 3 failures are the `(float(kj) - ...)` shape only; **zero** other errors in any file, so the explicit-arithmetic types, `coopmat`, `layout(...) buffer` and `shared` all parse |
| `glsl-lang` 0.8.1 | 13 of 29 | the 16 failures are all `uint16_t` / `float16_t` (the `GL_EXT_shader_explicit_arithmetic_types_*` extensions) as unknown type names |
| naga 30.0.1 `glsl-in` | 16 of 29 | failures: `uint16_t` unknown type (9), `coopmat` unknown (1), `uint16_t` unknown variable (1), `subgroupAdd` unknown function (2: `matvec`, `matvec_bf16_b`) |

Reading it: the tree-sitter grammar is missing two small constructs, not broad coverage. One of them (the `local_size`
declaration) is in every compute shader and is the thing a reflection model most wants, so **as published it cannot be
used**. Option B is worse on this corpus (13 of 29, failing on the extension types the corpus is built on), and
naga `glsl-in` is confirmed unsuitable (G5's expected cause, now measured: 16 of 29). The deleted-line column is a
what-if, not a parse of the real files, and says nothing about whether the rest of each tree is correct (only that it
has no ERROR/MISSING node).

**Decision still open, recommendation:** keep A and fix the grammar upstream (`tree-sitter-glsl`: `layout(...) in;`
as a declaration, and a parenthesised expression that begins with a keyword type), then require the fixed release. Fallbacks, in
order: Vulkane publishes a fixed grammar crate itself; failing that, a pre-pass in `vulkane-glsl-parse` that lifts
the bare `layout(local_size_*) in;` out of the source before parsing (it is one regular line, and the values are
read by the model rather than the tree) plus a `(float(` workaround, which is a hack. **Ruled out by the PM 2026-10-31: no pre-pass hack.** The
Unpopped seam question for B is not raised: B is not viable here.

### The two gaps: repros (run) and patch sketches (NOT run)

Repros, each run against `tree-sitter-glsl` 0.2.0 on `tree-sitter` 0.27.1 (2026-10-31); the second has a control.

```glsl
// repro1.comp -> ERROR at `layout(local_size_x = 64) in` (the qualifier-only declaration)
#version 450
layout(local_size_x = 64) in;
void main() {}

// repro2.comp -> 2 ERROR nodes, at `float(` and at `)`
#version 450
void main() {
    float s = (float(1) - 1.0);
}

// control2.comp -> 0 errors: the same expression with the type not directly after `(`
#version 450
void main() {
    float s = (1.0 - float(1));
}
```

Cause, read from the 0.2.0 `grammar.js` (extends `tree-sitter-c` 0.24.2): (1) `declaration` is `qualifiers` followed by
either `identifier field_declaration_list ...` or the C declaration (a type and declarators), so a declaration that is
qualifiers and `;` alone matches nothing. (2) The inherited C `cast_expression` reads `(float` as the start of a C cast
`(type) expr`; GLSL has no C-style casts, so a `(` followed by a keyword type should begin a parenthesised constructor
call. `s += float(kj)`, `float(1) - float(2)` and `(1.0 - float(1))` all parse (checked), which is why this
stayed hidden.

Patch sketches. **Neither was built or run:** the tree-sitter CLI was not available in the session (an `npm install` of it
was not permitted), so these are the shape of the fix for a maintainer to complete and test, not a verified patch.

```js
// (1) in `declaration`, add an alternative before the existing `seq(repeat(...), choice(...))`:
seq(repeat1(choice('in', 'out', 'uniform', 'buffer', 'shared', $.layout_specification)), ";"),

// (2) GLSL has no C cast: stop `(float` being read as one, e.g. drop it from the expression choice
//     (exact idiom is the maintainer's: an override of `cast_expression` that can never match).
```

Upstream: `https://github.com/tree-sitter-grammars/tree-sitter-glsl` (the crate's `repository` field says `theHamsta/tree-sitter-glsl`;
the README links the `tree-sitter-grammars` org). Licence MIT. Submission is an outward post and waits on CireSnave (PM).

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
read). Crate versions, dates and licences from the crates.io API on 2026-10-30. The `tree-sitter-glsl`, `glsl-lang` and
`naga glsl-in` rates are measured in the G1 measurement section (2026-10-31).
