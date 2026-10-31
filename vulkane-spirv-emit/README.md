# vulkane-spirv-emit

The SPIR-V compute-shader emitter for [Unpopped](https://crates.io/crates/unpopped), part of
[Vulkane](https://github.com/ciresnave/vulkane). `SpirV` implements `unpopped::Backend`: it lowers a scalar
elementwise `OpDef` to a binary SPIR-V module and returns it as the kernel's word stream
(`GeneratedKernel::words()`), with the disassembly as `source`.

```rust,ignore
use unpopped::generate;
let kernel = generate(&op, &key, &vulkane_spirv_emit::SpirV);
let words: &[u32] = kernel.words().expect("the SPIR-V backend always supplies words");
```

Module shape: Vulkan 1.1 / SPIR-V 1.3, `LocalSize 256 1 1`, descriptor set 0 with `input0..` at bindings `0..n-1` and
`output` at binding `n`, no bounds guard (dispatch exactly `n / 256` groups). v1 scope and the declined set are in the
crate docs. Validate with `spirv-val --target-env vulkan1.1` before trusting a module.
