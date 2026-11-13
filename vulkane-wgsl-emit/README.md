# vulkane-wgsl-emit

The WGSL compute-shader emitter for [Unpopped](https://crates.io/crates/unpopped): `Wgsl` lowers a scalar contiguous
elementwise `KernelPlan` to a WGSL module: `@group(0) @binding(K) var<storage, read> input{K}: array<T>;`, one
`var<storage, read_write> output: array<T>;`, and a `@compute @workgroup_size(256, 1, 1)` entry. Operands are bound 0..N in
declaration order (inputs, then the output).

Dtypes: `f32` (and `F32Strict`), `i32`, `u32`. Everything else is a **typed decline**, never an approximation: WGSL has no
`f64`, no 64-bit or 8-bit integers (and no 16-bit integers), and `f16` needs `enable f16;` (not emitted yet). Ops with no exact
WGSL spelling (`erf` family, `copysign`, `nextafter`, a fused `fma`, casts, integer div/rem) decline with the op named.

WGSL has no way to request IEEE subnormal handling, so f32 results near zero are the device's. The emitted header says so.

Not yet on crates.io (first publish is the PM's). Licensed under either of Apache-2.0 or MIT, at your option.
