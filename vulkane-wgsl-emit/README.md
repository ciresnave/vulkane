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

## Declined: possible patent exposure (board 434 C extended)

CireSnave 2026-11-16, verbatim: *"For that patent, we add a note about the potential for infringement if non-Nvidia-specific emitters lower those operations to hardware ops and we ensure our projects' emitters for other manufacturers' hardware know about that and we consider ourselves not infringing."* The patent is NVIDIA US11803380B2 (a synchronization barrier with an arrival counter, a phase and a hardware copy operator).

This emitter targets non-NVIDIA hardware and does **not** lower `Stmt::MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`, or `CopyBulk` on an mbarrier to hardware operations; it refuses them by name with a typed error. `Stmt::WgmmaWait` stays on hold for CireSnave's answer. The earlier board 434 C declines (`AsyncWait`, `CopyAsync`, `AsyncCommit`, `Collective`) stand. Status: `Backend::lower_kernel` is not implemented here, so the default typed refusal (`LowerError::UnsupportedKernel`) already covers every one of these; the by-name refusal and its per-statement tests wait on a published `unpopped` that has these `Stmt` variants (crates.io latest is 0.47.0, which has none; ROADMAP row 8l).
