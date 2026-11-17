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

## Declined: possible patent exposure (board 434 C and its extension)

CireSnave 2026-11-16, verbatim: *"For that patent, we add a note about the potential for infringement if non-Nvidia-specific emitters lower those operations to hardware ops and we ensure our projects' emitters for other manufacturers' hardware know about that and we consider ourselves not infringing."* The patent is NVIDIA US11803380B2 (a synchronization barrier with an arrival counter, a phase and a hardware copy operator).

This emitter targets non-NVIDIA hardware and does **not** lower `Stmt::MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`, `CopyBulk` on an mbarrier, `Tcgen05Commit` (it arrives on an mbarrier), or `WgmmaWait` to hardware operations; it refuses them by name with a typed error. `WgmmaWait` is a separate patent (NVIDIA US12204897B2 and its continuation US12710961B2: a wait on a count of pending matrix-multiply operations); same rule, CireSnave 2026-11-16 (second ruling), verbatim: *"I think we treat that patent like the others that are Nvidia-only. Add a note and specifically notify our non-Nvidia-emitter builders about the potential for infringement and not to."* **Builders of non-NVIDIA emitters: lowering any of these to a hardware operation risks infringement; do not.** Board 434 C (CireSnave 2026-11-13, "434: C."), verbatim in the ruling: emitters may lower `AsyncWait`, `CopyAsync`, `AsyncCommit` and `Collective` for NVIDIA devices and **decline by name for non-NVIDIA devices**, which this emitter is; the async three are NVIDIA US11294713B2 and US12118382B2 (an API that blocks a thread's instructions around an asynchronous shared-memory copy). Eleven names in all: those four and the seven above. Status: `Backend::lower_kernel` refuses each of the eleven by name with `LowerError::UnsupportedKernel` (the text carries `declined: possible exposure, board 434 C` for the four of board 434 C and `... board 434 C extended` for the seven of the extension), at the top level and inside `If`, `For` and `While` at any depth, and refuses every other kernel (including `CollectiveTuple`, the tuple form of `Collective`, which no ruling names) with the same unnamed typed error because this emitter does not lower kernel-layer kernels yet. `tests/patent_decline.rs` pins all eleven names at six nesting shapes and has a control that an innocent kernel does not carry the marker. The walk (`src/patent_decline.rs`) matches every `Stmt` variant explicitly with no wildcard arm, so a new statement in `unpopped` fails to compile until it is classified.
