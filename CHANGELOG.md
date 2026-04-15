# Changelog

All notable changes to vulkane will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.3] — 2026-04-15

### Added

- **Optional `shaderc` feature** — runtime GLSL/HLSL → SPIR-V compilation via the Khronos reference `glslang` compiler (wrapped by `shaderc-rs`). Complements the existing `naga` feature for cases that need full GLSL extension support, HLSL input, or glslang-only optimization passes.
  - `vulkane::safe::shaderc::compile_glsl(source, kind, file_name, entry_point) -> Result<Vec<u32>, ShadercError>` — common case.
  - `vulkane::safe::shaderc::compile_with_options(..., |opts| { ... })` — HLSL input, optimization level, macro defines, include callbacks, target Vulkan version.
  - Re-exports `ShaderKind`, `SourceLanguage`, `TargetEnv` from `shaderc`.
  - New `Error::ShadercCompile(String)` variant bridged from `ShadercError`.
  - Build requires either the LunarG Vulkan SDK (`VULKAN_SDK` env var), a system libshaderc, or a C++ build toolchain (CMake + Python + C++ compiler) for the source-build fallback. See README for details.

## [0.4.0] — 2026-04-10

### Added

- **45 Format constants** (up from 11) covering 8/16/32-bit, depth, and BC compressed formats. No more reaching into `vulkane::raw::bindings::VkFormat` for vertex attribute formats.
- **`Format::bytes_per_pixel()`** — returns the byte size per pixel for common uncompressed formats.
- **`BufferCopy::full(size)`** — one-liner for the common offset-0 copy case.
- **`#[derive(Vertex)]` proc macro** (new `vulkane_derive` crate, opt-in via `derive` feature) — auto-generates `VertexInputBinding` + `VertexInputAttribute` from `#[repr(C)]` structs. Supports `f32`, `[f32; 2..4]`, `u32`, `[u32; 2..4]`, `i32`, `[i32; 2..3]`, `[u8; 4]`, `u16`, `i16`. Provides both `::binding()` (vertex rate) and `::instance_binding()` (instance rate).
- New example: `derive_vertex` — instanced triangles using the derive macro.

## [0.3.0] — 2026-04-10

### Added

- **Pipeline builder extensions:**
  - `depth_bias(constant, slope, clamp)` — shadow acne prevention.
  - `depth_compare_op(CompareOp)` with `CompareOp` enum (NEVER / LESS / EQUAL / LESS_OR_EQUAL / GREATER / NOT_EQUAL / GREATER_OR_EQUAL / ALWAYS).
  - `InputRate` (VERTEX / INSTANCE) on `VertexInputBinding` — instanced rendering.
  - `color_attachment_count(n)` — multi-attachment / G-buffer pipelines.
  - `dynamic_viewport_scissor()` — resize-friendly pipelines with `set_viewport` / `set_scissor`.
- **Depth image views** — `ImageView::new_2d_depth` for depth-aspect views.
- **Image barrier aspect mask** — `ImageBarrier` gains `aspect_mask` field + `::color()` / `::depth()` convenience constructors.
- **`ClearValue` enum** + `begin_render_pass_ext` for mixed color + depth/stencil clear values.
- **Comparison sampler** — `SamplerCreateInfo::compare_op` for shadow map sampling.
- **Allocation helpers:**
  - `Buffer::new_bound(device, physical, info, flags)` — 5-step boilerplate → 1 call.
  - `Image::new_2d_bound(device, physical, info, flags)` — same for images + auto color view.
  - `Queue::upload_buffer<T>(device, physical, qf, data, usage)` — staging upload in one call.
  - `Queue::upload_image_rgba(device, physical, qf, w, h, pixels)` — image upload with layout transitions.
- New examples: `depth_prepass`, `instanced_mesh`, `shadow_map`, `deferred_shading`.

### Breaking

- `ImageBarrier` now requires `aspect_mask: u32` field. Use `ImageBarrier::color(...)` or `ImageBarrier::depth(...)` constructors.

## [0.2.0] — 2026-04-09

### Added

- **Typed pipeline stage and access mask constants** — `PipelineStage`, `AccessFlags` (32-bit), `PipelineStage2`, `AccessFlags2` (64-bit for Sync2). All barrier, timestamp, and sync APIs now accept these types instead of raw `u32` / `u64`.
- **Convenience constructors:**
  - `QueueCreateInfo::single(family_index)` — one queue, priority 1.0.
  - `WaitSemaphore::binary(sem, stage)` / `::timeline(sem, value, stage)`.
  - `SignalSemaphore::binary(sem)` / `::timeline(sem, value)`.
  - `RenderPass::simple_color(device, format, load, store, final_layout)`.
  - `Queue::one_shot(device, qf, |rec| { ... })` — fire-and-forget command recording.
- **Raw escape hatch** — `Device::dispatch()` and `Instance::dispatch()` expose the full dispatch tables for calling any Vulkan function alongside safe wrapper types.
- New examples: `buffer_upload`, `raw_interop`, `allocator_compute`.

### Breaking

- All barrier/sync API signatures changed from raw `u32`/`u64` to typed `PipelineStage`/`AccessFlags`. Migration: `0x800` → `PipelineStage::COMPUTE_SHADER`.
- `WaitSemaphore::dst_stage_mask` changed from `u32` to `PipelineStage`.

## [0.1.0] — 2026-04-08

### Added

- Initial release: complete Vulkan bindings generated from vk.xml + safe RAII wrapper covering compute and graphics end-to-end.
- **Raw bindings** (`vulkane::raw`) — all types, enums, structs, function pointers, and three-tier dispatch tables generated from the spec.
- **Safe wrapper** (`vulkane::safe`) — RAII handles for Instance, Device, Buffer, Image, ImageView, Sampler, DeviceMemory, ShaderModule, DescriptorSetLayout/Pool/Set, PipelineLayout, ComputePipeline, GraphicsPipeline (with builder), RenderPass, Framebuffer, Surface (Win32/Wayland/Xlib/Xcb/Metal), Swapchain, CommandPool/Buffer, Fence, Semaphore (binary + timeline), QueryPool.
- **VMA-style sub-allocator** — TLSF + linear pools, custom user pools, dedicated allocations, persistent mapping, defragmentation, memory budget queries.
- **Device groups** — unified single/multi-GPU device representation with per-allocation and per-submission device masks.
- **DeviceFeatures builder** — Vulkan 1.0/1.1/1.2/1.3 feature chain construction.
- **Optional `naga` feature** — `compile_glsl` + `compile_wgsl` → SPIR-V at runtime.
- **`fetch-spec` feature** — auto-download vk.xml from Khronos GitHub.
- 7 bundled examples: device_info, fill_buffer, compute_square, compute_image_invert, compile_shader, headless_triangle, textured_quad, windowed_triangle.
- Tree-based XML parser (roxmltree), vk.xml api-attribute filtering, VKSC profile exclusion.
- CI on Linux/Windows/macOS with Mesa Lavapipe for headless GPU tests.
