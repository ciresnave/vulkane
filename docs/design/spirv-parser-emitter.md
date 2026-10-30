# SPIR-V parser and emitter in Vulkane

**Status: design note, options first, no code.** ROADMAP row 3a. Written by the Vulkane lane 2026-10-30.

## Ruling this serves

CireSnave, 2026-10-30, verbatim: *"Vulkane should own SPIR-V and create a parser and emitter."* The language owner
writes the parser AND the emitter; Unpopped holds neither (`Unpopped/docs/parser-emitter-ownership-map.md` section 9).
Naming and test rules follow the Slang note (`slang-parser-emitter-handover.md`, D1 and D3): two crates,
`vulkane-spirv-parse` and `vulkane-spirv-emit`, conformance cells inside the emitter crate against a registry `unpopped`.

## What exists today (measured)

Vulkane `origin/main` after #105, 2026-10-30; crates.io API the same day.

| piece | state |
|---|---|
| SPIR-V in Vulkane | `safe/shader.rs` (91 lines) wraps a `Vec<u32>` into a `ShaderModule`; no decoding, no encoding. 9 committed `.spv` files. |
| SPIR-V producers | `safe::naga` (GLSL/WGSL to SPIR-V, feature `naga`) and `safe::shaderc` (GLSL/HLSL to SPIR-V, feature `shaderc`). Neither reads SPIR-V. |
| Corpus | `vulkane-kernels-source/compile.sh` builds 178 `.spv` from 147 Slang and 27 GLSL files (measured in #105; 177 of 178 pass `spirv-val --target-env vulkan1.1`, the one failure is #106). Fuel's `fuel-vulkan-kernels/spv/` holds committed copies. |
| Unpopped side | `Frontend` in `unpopped::convert` returns a `tree_sitter::Tree` (fields `name`, `kernel_marker`, `residue`, `out_name`, `in_prefix`, `parse`). `Backend`/`Lowering` produce text. |
| Candidate crates | `rspirv` 0.13.0+sdk-1.4.341.0 (Apache-2.0, 2026-04-10, about 1.4M downloads): binary parser, builder, disassembler. `spirv` 0.4.0+sdk-1.4.341.0: the grammar enums, 42M downloads. `naga` 30.0.1 (2026-08-22) has `spv-in`/`spv-out`. |

**Finding, outside this note:** `vulkane/Cargo.toml` pins `naga = "29"` (latest 30.0.1) and `shaderc = "0.8"` (latest
0.10.1, 2025-09-06). CireSnave's standing rule is latest versions at all times. Tracked as its own PR, not part of this note.

## The structural problem: SPIR-V does not fit `Frontend`

`Frontend::parse` returns a tree-sitter CST and the `lift_*` functions walk C-family syntax by node kind and text. SPIR-V
is a binary SSA module with no grammar, no source text, no CST. Two consequences:

1. A SPIR-V "parser" cannot be a `Frontend` as defined. It needs either a second neutral entry point in core that takes
   an already-built structure (for example `lift_from(...)` over an SSA function) or its own lowering to `ir`.
2. The lifted subset is also different in kind. Source frontends lift *elementwise / reduction / scan* patterns from
   syntax. A SPIR-V module is post-optimisation: loops are unrolled or restructured, buffers are `OpAccessChain`s on
   storage-buffer variables, workgroup shape is an execution mode. Pattern-lifting back to `ir::OpDef` will accept far
   fewer modules than the same kernel's source did.

This is Unpopped core's contract, not Vulkane's to edit; it is listed so the options below are priced honestly.

## Decisions needed, with a recommendation each

### S1. Binary layer: build or reuse

| option | for | against |
|---|---|---|
| **A. Depend on `rspirv` (+ `spirv`)** (recommended) | Maintained against each SDK grammar (version string carries the SDK), the builder enforces id/type bookkeeping, Apache-2.0 is compatible with our `MIT OR Apache-2.0` crates. Both parse and emit share one data model. | Third-party dependency whose release cadence follows the Khronos SDK, not ours; its data model (`Module`/`Instruction`/`Operand`) leaks into our public types unless wrapped. |
| B. Own decoder/encoder generated from `spirv.core.grammar.json` | No dependency; Vulkane already has a spec-driven generator (`vulkan_gen`) for `vk.xml`, so the technique exists. | Re-implements what `rspirv` and `spirv` already do; the grammar JSON must be tracked per SDK release. Largest cost, least benefit. |
| C. Use `naga` `spv-in` / `spv-out` as the layer | Already an optional dependency; one dependency for GLSL, WGSL and SPIR-V. | `naga` imports into *its* IR and drops what it cannot model; SPIR-V to naga to SPIR-V is not round-trip safe, which an emitter we own must be. |

### S2. What the parser produces

| option | meaning |
|---|---|
| **A. A faithful module model, no lifting** (recommended first) | `vulkane-spirv-parse` decodes a module to typed functions, blocks, ids and decorations (wrapping S1), and exposes entry points, execution modes, descriptor bindings and capabilities. This is useful on its own (reflection of the 178 kernels) and is the input any later lifting needs. |
| B. Lift to `unpopped::ir` immediately | Delivers the cross-language conversion the ownership map wants, but accepts few modules (see above) and needs the core entry point from consequence 1. Premature before A exists. |

Recommendation: A now; B as a separate row once Unpopped decides the SSA entry point.

### S3. What the emitter consumes

The emitter implements `unpopped::Backend`/`Lowering` as Slang does: input is `ir::OpDef` / `KernelFn`, output is a
SPIR-V module. Unlike the text backends it produces binary words, so the `Backend` output type must carry bytes or
`Vec<u32>`. Whether the existing trait does is unverified here; reading `Backend` is the first task of the
implementation PR, and if it is text-only that is a finding for the Unpopped lane, not a workaround in Vulkane.
Scope for the first emitter: scalar elementwise, the same declined set as `unpopped-slang`, so the two emitters can be
checked against each other (see S4).

### S4. Correctness oracle

`spirv-val --target-env vulkan1.1` (SPIR-V Tools) is available (Vulkan SDK 1.4.350 here) and is the validity oracle:
every emitted module must pass it, with a negative control (a corrupted module is rejected). Behaviour oracle:
the emitter's output for an elementwise op is run through the existing Vulkane compute path and compared with the
CPU C emitter (`unpopped-cpu-c`) over the same inputs. GPU tests run under `gpu-run.ps1` or the self-hosted runner,
never `#[ignore]`d (portfolio rule). Parser oracle: round-trip `parse` then `emit-words` is byte-identical for all 178
corpus modules, and `spirv-val` verdicts match before and after. #106 (`softmax_f64`) is the built-in failing case:
the parser must load it and the validator must still reject it, proving the parser does not launder invalid modules.

### S5. Crate layout and release

Same as Slang D1/D5: `vulkane-spirv-parse` (module model) and `vulkane-spirv-emit` (Backend), both in Vulkane's single
version series. `vulkane-spirv-emit` depends on `vulkane-spirv-parse` only if it reuses the model; with S1-A it can
use `rspirv` directly, which is the lower-coupling choice. Crates.io dependencies only.

## Proposed sequence (each its own PR; none starts before this note merges)

1. Bump `naga` and `shaderc` to latest (separate maintenance PR; independent of this note).
2. `vulkane-spirv-parse`: module model over `rspirv`, reflection of entry points/bindings, corpus round-trip test over the 178 modules.
3. Read Unpopped's `Backend` output type; if text-only, [ASK] the Unpopped lane before any emitter code.
4. `vulkane-spirv-emit`: scalar elementwise, validated by `spirv-val`, compared against `unpopped-cpu-c`.
5. Optional: lifting to `unpopped::ir` once core has an SSA entry point (S2-B).

## Not decided here

GLSL, HLSL, WGSL (rows 3b-3d: separate notes). Relationship between SPIR-V emission and Slang's own `slangc` SPIR-V output
(the Slang compiler remains the production path for the kernels; this emitter does not replace it).

## Provenance

Crate versions, dates and licences read from the crates.io API on 2026-10-30 (a nonexistent crate name returns an error,
so the query discriminates). File counts from `git ls-files` on `origin/main`. The `Frontend` field list was read from
Unpopped `origin/main`'s `crates/unpopped/src/convert.rs`. The `Backend` output type and `naga`'s round-trip behaviour were
not read or tested for this note and are marked as such above.
