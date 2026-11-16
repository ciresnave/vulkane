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

## Declined: possible patent exposure (board 434 C extended)

CireSnave 2026-11-16, verbatim: *"For that patent, we add a note about the potential for infringement if non-Nvidia-specific emitters lower those operations to hardware ops and we ensure our projects' emitters for other manufacturers' hardware know about that and we consider ourselves not infringing."* The patent is NVIDIA US11803380B2 (a synchronization barrier with an arrival counter, a phase and a hardware copy operator).

This emitter targets non-NVIDIA hardware and does **not** lower `Stmt::MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`, or `CopyBulk` on an mbarrier to hardware operations; it refuses them by name with a typed error. `Stmt::WgmmaWait` stays on hold for CireSnave's answer. The earlier board 434 C declines (`AsyncWait`, `CopyAsync`, `AsyncCommit`, `Collective`) stand. Status: `Backend::lower_kernel` is not implemented here, so the default typed refusal (`LowerError::UnsupportedKernel`) already covers every one of these; the by-name refusal and its per-statement tests wait on a published `unpopped` that has these `Stmt` variants (crates.io latest is 0.47.0, which has none; ROADMAP row 8l).
