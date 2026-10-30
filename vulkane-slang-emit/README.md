# vulkane-slang-emit

The Slang compute-shader emitter for [Unpopped](https://crates.io/crates/unpopped): it lowers Unpopped's
language-neutral IR (`ExprDag`, `KernelPlan`) to Slang source, using HLSL overloaded intrinsics,
`StructuredBuffer<T>` / `RWStructuredBuffer<T>` operands and a `[numthreads]` entry point.

Moved here from `unpopped-slang` (Unpopped `213df4f`, v0.34.11) with its git history, because Vulkane owns Slang
in the CireSnave ecosystem: the language owner writes the parser and the emitter. The crate is otherwise unchanged;
it depends on a published `unpopped` / `unpopped-vocab`.

Not yet on crates.io (first publish is the PM's); once published, add it with `cargo add vulkane-slang-emit`.

Feature `convert` additionally lifts emitted Slang back to the IR (pulls tree-sitter grammars, which compile C).

Scope today: scalar contiguous elementwise kernels; every other schedule is a typed decline. Float kernels need
`slangc -denorm-mode-fp32 preserve` (see `required_slangc_args` and the crate docs).

Licensed under either of Apache-2.0 or MIT, at your option. Provenance: `THIRD-PARTY-NOTICES.md`.
