# vulkane-glsl-emit

The GLSL compute-shader emitter for [Unpopped](https://crates.io/crates/unpopped). `Glsl::lower_kernel` lowers an
integer kernel-layer kernel to GLSL 4.50 through `vulkane-kernel-text`: `std430` storage buffers, `shared` arrays,
`barrier()` with the memory barriers, `Fence`, and structured control flow. Anything it cannot spell is a typed
`UnsupportedKernel` naming it; the NVIDIA-only statements of board 434 C are refused by name.

Licensed under MIT OR Apache-2.0.
