# Slang parser and emitter: handover from Unpopped to Vulkane

**Status: design note, no code.** ROADMAP rows 2 and 2a. Written by the Vulkane lane 2026-10-30; the Unpopped lane has
agreed (b) in principle and reviews this note before it merges. Nothing in Unpopped is edited until it merges and the
PM rules.

## Ruling this serves

CireSnave, 2026-10-30, verbatim: *"Vulkane should own SPIR-V and create a parser and emitter. ... Vulkane should own
GLSL, HLSL, and WGSL and create the parsers and emitters for those."* and *"Okay.  Vulkane now owns Slang in our
ecosystem.  We should move the Slang kernels from Fuel to Vulkane, shouldn't we?  Fuel can import them still."* The
kernel move is done (`vulkane-kernels-source`, #105). The language owner writes the parser AND the emitter; Unpopped
holds neither.

## What exists today (measured)

Unpopped `origin/main` `d2e3f2b`, 2026-10-30. Published versions read from the crates.io API the same day.

| piece | where | size | notes |
|---|---|---|---|
| Slang **emitter** | `crates/unpopped-slang/src/lib.rs` | 60,360 B | `Backend`/`Lowering` impl. Uses only `unpopped`'s public modules (`backend`, `cfamily`, `ir`, `plan`) and `unpopped_vocab`. Declines every schedule but `Scalar` elementwise (`numthreads` hard-coded 256). 155 commits on the path (139 by "Eric Evans' AI Agent", 10 Eric Evans, 6 "Unpopped Agent"). |
| Slang **parser** | `crates/unpopped/src/convert.rs`, feature `convert` | `parse_slang`, `SLANG_RESIDUE`, `pub const SLANG: Frontend` | Slang-only parts. `Frontend`, `lift_elementwise`, `lift_reduction`, `lift_scan`, `lift` are language-neutral and stay. Grammar crate `tree-sitter-slang` 0.3 (MIT), named in exactly one place (`parse_slang`). |
| Slang corpus report | `crates/unpopped/examples/slang_corpus_report.rs` + `docs/slang-corpus-sizing.md` | | Measured the parser over Fuel's 147 `.slang` files. The same 147 files are now `vulkane-kernels-source/kernels/*.slang`. |
| Consumer of the emitter | `crates/unpopped-conformance` (`publish = false`, test host) | dev-dep on `unpopped-slang` | 3 tests: `cell_token_is_not_a_computation`, `dtype_lowering_coverage`, `lower_kernel_default_is_the_typed_decline`. No other crate in the workspace depends on `unpopped-slang`. |
| Published | crates.io | `unpopped` / `unpopped-slang` / `unpopped-vocab` max **0.34.3**; the workspace is at **0.34.10** | A crate here may depend only on a published version. |

## Decisions needed, with a recommendation each

### D1. Crate layout in Vulkane

| option | for | against |
|---|---|---|
| **A. Two crates, `vulkane-slang-emit` and `vulkane-slang-parse`** (recommended) | Same shape as Baracuda's agreed `baracuda-cuda-emit` / proposed `baracuda-cuda-parse` (Unpopped map section 4): one naming rule for every language owner. The emitter stays free of tree-sitter (its `convert` feature exists only to keep the C build out of the default). | Two manifests, two publishes. |
| B. One crate `vulkane-slang` with features `emit` / `parse` | One publish. | Mixes a dependency-light emitter with a C-compiling parser behind a feature, which is the arrangement `unpopped-slang` already needs a feature to undo. |
| C. Leave the emitter in Unpopped as a reference emitter | No move. | Contradicts the ruling (language owner writes both) and splits Slang across two version series. Unpopped's map section 9 says Unpopped keeps `unpopped-slang`; that text predates the Slang ruling and is the Unpopped lane's to correct. |

### D2. What happens to the name `unpopped-slang` on crates.io

Recommended: **publish the new crates under new names and leave `unpopped-slang` published but marked deprecated by a
final release that says where it went.** Transferring crates.io ownership would keep the name but leaves a Vulkane-owned
crate called `unpopped-*`, which repeats the problem this move fixes. Decision belongs to the PM (publishing) and the
Unpopped lane (the last release).

### D3. The conformance dev-dependency cannot survive the move as written

After the move `vulkane-slang-emit` depends on a **registry** `unpopped`; `unpopped-conformance` depends on the
**workspace** `unpopped`. Cargo treats these as two packages, so the `Backend` trait the emitter implements is not the
trait the conformance tests call. It would only compile with a `[patch.crates-io]` of `unpopped` to the local path, and
then at every breaking `unpopped` 0.n release the emitter's requirement (`"0.n-1"`) cannot be met by the patched
workspace version: Unpopped's CI would be red until Vulkane published, and Vulkane cannot publish until Unpopped has.

Recommended: **the three Slang cells move into `vulkane-slang-emit/tests/`** and run against the registry `unpopped`.
No new published harness is needed: the Unpopped lane read the tests (2026-10-30) and reports they use only public
`unpopped` (`try_generate`, `ir::{OpDef, input}`), `unpopped_vocab`, and the backend under test, nothing from
`unpopped-conformance`'s `src`. (Checked for `dtype_lowering_coverage.rs` at `d2e3f2b`: its imports are `unpopped::{ir, try_generate}`, `unpopped_vocab`, `unpopped_slang::Slang` and `unpopped_cpu_c::CpuC`, so the moved test also needs `unpopped-cpu-c` as a registry dev-dependency; the other two tests were not read by me.) The Unpopped lane
will not publish a conformance kit; shared cells can be asked for later.

### D4. Parser seam

`Frontend` is a plain struct of function pointers whose `parse` returns a `tree_sitter::Tree`, so core already exposes
tree-sitter in a public type. Recommended: **keep `Frontend` and the `lift_*` functions in `unpopped::convert` and treat
them as the public parser seam**, which means they come under the semver promise (they are `pub` in a default-off
feature today). `vulkane-slang-parse` supplies `SLANG`, `parse_slang`, and the residue list, and depends on
`unpopped` with `convert`. Core then drops `tree-sitter-slang` and the `SLANG` const. **That removal is a breaking change
to `unpopped`**, so it takes a 0.n bump allocated by the PM, not a patch release, and it lands with step 3.

Consequence for core, owned by the Unpopped lane: the tests in `convert.rs` that use `SLANG` (about lines 1054-1150) and
the toy frontend at about line 987 that borrows `parse_slang` need a frontend that does not need the Slang grammar.

### D5. Release coupling

`vulkane-slang-emit` and `-parse` take `unpopped = "0.n"`, and Unpopped breaks at 0.n. With CireSnave's standing rule
that dependencies are always the latest, **every breaking Unpopped release forces a same-day Vulkane release** of both
crates. That is the price of the ownership split, not a defect to design away; recorded so it is priced. Versions follow
Vulkane's single series (the PM allocates), not Unpopped's: the crates depend on `unpopped`, they are not
"which version of Unpopped does this work with" markers, so the one-version exception does not apply.

## Proposed sequence (each step its own PR; none starts before this note merges)

1. **Gate (PM / Unpopped):** publish a current `unpopped`. The published 0.34.3 does contain `KernelFn` (76 mentions
   in its `src/`, downloaded and grepped 2026-10-30), so that is not the blocker; the blocker is that the emitter source
   being imported is 0.34.10 and must compile against a *published* `unpopped`. The Unpopped lane reports the PM has ruled
   it will publish the four-crate set (`unpopped`, `unpopped-vocab`, `unpopped-cpu-c`, `unpopped-slang`) as **0.34.11**
   right after Unpopped #195 merges (corrected by the Unpopped lane, relayed 2026-10-30; an earlier relay said 0.34.3,
   which crates.io already lists as the max, so it could not have been a new publish). Not yet published as of this
   writing: confirm on the registry before relying on it. The first thing step 2 does is build the imported crate
   against the published `unpopped` and report.
   Until a published version works, neither lane writes E1c or E2.
2. **Vulkane: `vulkane-slang-emit`** by history-preserving import of `crates/unpopped-slang` (the technique used for #105:
   path-limited `git fast-export`, blob-identity check against the source commit), then adapt the manifest to a
   crates.io `unpopped`. Publish (PM).
3. **Unpopped:** remove `crates/unpopped-slang`, move the three conformance cells (D3), correct map section 9, and cut
   the final `unpopped-slang` release (D2). Unpopped's own PRs.
4. **Vulkane: `vulkane-slang-parse`** from `parse_slang` / `SLANG` / `SLANG_RESIDUE`; its test corpus is
   `vulkane-kernels-source`'s 147 `.slang` files (re-run Unpopped's corpus sizing on them as the baseline, same method,
   same dtype handling, so the numbers are comparable).
5. **Vulkane: E1c** (Slang emission from `KernelFn`, design in Unpopped's `docs/e1-emit-from-the-kernel-layer.md`, with
   `cpu-c` E1a/E1b as the reference) **and E2** (8 unary polyfills: erf, gelu, erfc, asinh, acosh, atanh, cbrt, lgamma;
   plus copysign, nextafter), authored in `vulkane-slang-emit`.

## Not decided here

- SPIR-V, GLSL, HLSL, WGSL (ROADMAP row 3): separate notes, one language per PR. This note's D1 naming rule and D3
  conformance rule are meant to be reused by them.
- The IR-union additions Slang needs (groupshared, barriers, wave ops: Unpopped's `docs/ir-union-surveys/slang.md`).
  IR nodes are Unpopped's; lifting and emitting them is Vulkane's, after step 5.
- Whether `softmax_f64.slang` (#106) is fixed in the kernel or by the emitter's f64 `exp`: a numerics decision with the
  Fuel lane.

## Provenance

Every figure above was read from Unpopped `origin/main` `d2e3f2b` or the crates.io API on 2026-10-30; none is
carried from memory. The file sizes and commit counts are of `crates/unpopped-slang` as it stands there.
