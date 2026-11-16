# vulkane-hlsl-emit

The HLSL compute-shader emitter for [Unpopped](https://crates.io/crates/unpopped): `Hlsl` lowers a scalar
elementwise `KernelPlan` to HLSL source that `dxc -T cs_6_2 -spirv` compiles. It is a thin layer over
`fishstringer-slang-emit` (the Slang text for this path is already HLSL); it changes the entry prefix, the
header and the compiler-argument line (`required_dxc_args`: `-denorm preserve` for f32). Scope and declined
set are the Slang emitter's.

Bindings carry no `register(..)`: with `-spirv`, `t0`/`t1`/`u0` collide on SPIR-V binding 0. Declaration
order gives bindings 0..N.

Not yet on crates.io (first publish is the PM's).

Licensed under either of Apache-2.0 or MIT, at your option. Provenance: `THIRD-PARTY-NOTICES.md`.

## Declined: possible patent exposure (board 434 C extended)

CireSnave 2026-11-16, verbatim: *"For that patent, we add a note about the potential for infringement if non-Nvidia-specific emitters lower those operations to hardware ops and we ensure our projects' emitters for other manufacturers' hardware know about that and we consider ourselves not infringing."* The patent is NVIDIA US11803380B2 (a synchronization barrier with an arrival counter, a phase and a hardware copy operator).

This emitter targets non-NVIDIA hardware and does **not** lower `Stmt::MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`, `CopyBulk` on an mbarrier, or `Tcgen05Commit` (it arrives on an mbarrier) to hardware operations; it refuses them by name with a typed error. `Stmt::WgmmaWait` stays on hold for CireSnave's answer. The earlier board 434 C declines (`AsyncWait`, `CopyAsync`, `AsyncCommit`, `Collective`) stand. Status: `Backend::lower_kernel` is not implemented here, so the default typed refusal (`LowerError::UnsupportedKernel`) already covers every one of these; the by-name refusal and its per-statement tests wait on a published `unpopped` that has these `Stmt` variants (crates.io latest is 0.47.0, which has none; ROADMAP row 8l).
