# vulkane_derive

Derive macros for the [`vulkane`](https://crates.io/crates/vulkane) Vulkan bindings crate.

Provides `#[derive(Vertex)]`, which generates the vertex input binding and attribute
descriptions for a `#[repr(C)]` struct.

You normally do not depend on this crate directly: enable the `derive` feature of `vulkane`
instead.

```toml
[dependencies]
vulkane = { version = "0.21", features = ["derive"] }
```

See the `vulkane` documentation and `vulkane/examples/derive_vertex.rs` for usage.

Licensed under either of MIT or Apache-2.0, at your option.
