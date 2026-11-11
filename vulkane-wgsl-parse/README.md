# vulkane-wgsl-parse

A WGSL front end: parse WGSL source into an owned Rust model and reflect its entry points
(`@compute`/`@vertex`/`@fragment`, `@workgroup_size`) and resource bindings (`@group`/`@binding`).

```rust
let m = vulkane_wgsl_parse::parse_wgsl("@compute @workgroup_size(64) fn main() {}").unwrap();
assert_eq!(m.entry_points[0].name, "main");
```

Built on [`wgsl-parse`](https://crates.io/crates/wgsl-parse) (decision W1 in
`docs/design/wgsl-parser-emitter.md`). It is a Rust-AST crate, not an Unpopped `Frontend`. Constants are
not evaluated: a non-literal `@group`/`@binding` is an error, a non-literal `@workgroup_size` is kept as
source text. Tested over a compute corpus with `naga` as the validity oracle.

Licensed under MIT OR Apache-2.0.
