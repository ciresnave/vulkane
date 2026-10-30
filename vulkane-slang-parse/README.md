# vulkane-slang-parse

The Slang front end for [Unpopped](https://crates.io/crates/unpopped): a tree-sitter Slang parser
(`parse_slang`), the Slang residue list (`SLANG_RESIDUE`) and the `SLANG` `Frontend`, which feeds Unpopped's
`lift_elementwise` / `lift_reduction` / `lift_scan` walkers to turn Slang compute kernels into IR.

Moved here from `unpopped::convert` (Unpopped `213df4f`, v0.34.11) because Vulkane owns Slang in the CireSnave
ecosystem: the language owner writes the parser and the emitter (`vulkane-slang-emit`). Unpopped keeps the
language-neutral `Frontend` and walkers.

Not yet on crates.io (first publish is the PM's); once published, add it with `cargo add vulkane-slang-parse`.

`tree-sitter` must be the same minor as the one `unpopped` uses, because `Frontend::parse` returns a
`tree_sitter::Tree`.

Licensed under either of Apache-2.0 or MIT, at your option. Provenance: `THIRD-PARTY-NOTICES.md`.
