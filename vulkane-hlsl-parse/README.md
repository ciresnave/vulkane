# vulkane-hlsl-parse

The HLSL front end for [Unpopped](https://crates.io/crates/unpopped): a tree-sitter HLSL parser (`parse_hlsl`),
the HLSL residue list (`HLSL_RESIDUE`) and the `HLSL` `Frontend`, which feeds Unpopped's `lift_elementwise` /
`lift_reduction` / `lift_scan` walkers to turn HLSL compute kernels into IR.

Vulkane owns HLSL in the CireSnave ecosystem (ruling 2026-10-30): the language owner writes the parser and
(next) the emitter, `vulkane-hlsl-emit`. Unpopped keeps the language-neutral `Frontend` and walkers.

Known grammar residue (`tree-sitter-hlsl` 0.2.0): `cbuffer N : register(bK) { .. }` and
`tbuffer N : register(tK) { .. }` report `MISSING ;`, and the attribute `[shader("compute")]` is an `ERROR`.
They are tolerated, and pinned by `tests/fixtures.rs` so an upstream fix is noticed.

Not yet on crates.io (first publish is the PM's).

`tree-sitter` must be the same minor as the one `unpopped` uses, because `Frontend::parse` returns a
`tree_sitter::Tree`.

Licensed under either of Apache-2.0 or MIT, at your option. Provenance: `THIRD-PARTY-NOTICES.md`.
