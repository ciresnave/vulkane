# vulkane-hlsl-emit

The HLSL compute-shader emitter for [Unpopped](https://crates.io/crates/unpopped): `Hlsl` lowers a scalar
elementwise `KernelPlan` to HLSL source that `dxc -T cs_6_2 -spirv` compiles. It is a thin layer over
`vulkane-slang-emit` (the Slang text for this path is already HLSL); it changes the entry prefix, the
header and the compiler-argument line (`required_dxc_args`: `-denorm preserve` for f32). Scope and declined
set are the Slang emitter's.

Bindings carry no `register(..)`: with `-spirv`, `t0`/`t1`/`u0` collide on SPIR-V binding 0. Declaration
order gives bindings 0..N.

Not yet on crates.io (first publish is the PM's).

Licensed under either of Apache-2.0 or MIT, at your option. Provenance: `THIRD-PARTY-NOTICES.md`.
