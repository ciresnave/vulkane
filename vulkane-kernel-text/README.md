# vulkane-kernel-text

The shared text walker for Vulkane's kernel emitters. Given an Unpopped kernel-layer `KernelFn` and a `Dialect`
(`vulkane-glsl-emit`, `vulkane-wgsl-emit`), it lowers integer kernels to shader source: buffers, static workgroup
arrays, locals, `Load`/`Store`, barriers and fences, and structured control flow. What a dialect cannot spell is a
typed `UnsupportedKernel` naming it, never an approximation. The NVIDIA-only statements of board 434 C are refused by
name before anything else.

Licensed under MIT OR Apache-2.0.
