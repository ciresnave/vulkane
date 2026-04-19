# Changelog

All notable changes to vulkane will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] — 2026-04-19

Allocator-side VRAM observability: the `Allocator` can now surface the driver's per-heap budget numbers in one call, fire budget-pressure callbacks when usage crosses a configurable threshold, and predictively check whether a prospective allocation would exceed the budget — all without requiring the user to opt into `VK_EXT_memory_budget` manually.

### Added

- **`Allocator::vram_budget()` / `Allocator::vram_used()`** — scalar-byte convenience helpers that sum the driver-reported budget and usage across every `DEVICE_LOCAL` memory heap. The single-number answer ML schedulers, profilers, and UI indicators typically want.
- **`Allocator::has_memory_budget_support()`** — returns `true` iff the budget numbers are authoritative (both `vkGetPhysicalDeviceMemoryProperties2` is loaded *and* `VK_EXT_memory_budget` is enabled on the device). Use this to distinguish "heap is empty" from "no query support".
- **Budget-pressure callback registry** — `Allocator::register_pressure_callback(threshold, hysteresis, closure)` fires a `PressureEvent` when a heap's `usage / budget` fraction rises past `threshold` (`PressureKind::Crossed`), falls back below `threshold - hysteresis` (`PressureKind::Relieved`), or — via `would_fit` — is projected to rise past `threshold` on a pending allocation (`PressureKind::Predictive`). Per-heap hysteresis latching prevents flutter near the threshold. Callbacks are invoked after every internal allocator lock has been released, so they may call back into the `Allocator` without deadlocking. `unregister_pressure_callback(id)` removes a registration.
- **`Allocator::would_fit(size, memory_type_index) -> FitStatus`** — proactively computes whether a forthcoming allocation would keep usage under the driver's soft budget, fires `Predictive` events for any threshold it would cross, and returns the projected heap stats (`current_usage`, `budget`, `projected_usage`, `projected_fraction`, `fits`). Lets schedulers free resources *before* attempting an allocation rather than reacting to a `Crossed` event after the fact.
- **`Device::enabled_extensions()` / `Device::is_extension_enabled(name)`** — introspect the final extension list sent to `vkCreateDevice`. Captures both explicit user requests and any extension the safe wrapper auto-enabled.
- **New documentation**: `vulkane/docs/DEFRAG_FOR_ML.md` — a dedicated walkthrough of the existing `build_defragmentation_plan` / `apply_defragmentation_plan` API aimed at ML-framework integrators, including a full worked tensor-pool compaction example and guidance on layering defrag under budget-based eviction.

### Changed

- **Device creation now auto-enables `VK_EXT_memory_budget`** when the physical device advertises it. The extension is passive — enabling it only causes the driver to populate `VkPhysicalDeviceMemoryBudgetPropertiesEXT` on `vkGetPhysicalDeviceMemoryProperties2` calls — so this is observable in `Device::enabled_extensions()` but has no runtime cost when unused. Opt-out is not currently exposed; file an issue if you need it.
- `Allocator::query_budget()` doc comment clarified: budget numbers are meaningful when `has_memory_budget_support()` returns `true`, which the auto-enable path makes the default on supported drivers.

## [0.6.0] — 2026-04-16

Major version: every Vulkan extension and feature bit is now reachable from safe code via generated builders. Layer 1 + Layer 2 + Layer 3 of the extension-handling architecture are all landed, plus **Phase 1 of Layer 4** — RAII wrappers for every previously-unwrapped Vulkan handle type.

### Added

- **`PNextChainable` trait and `PNextChain` builder** (Layer 2) — a generic pNext-chain mechanism replaces every hand-rolled pointer-patching site in the crate.
  - `PNextChainable` is implemented by the generator for every `#[repr(C)]` struct in `vk.xml` whose first two fields are `sType: VkStructureType` and `pNext` — **1225 impls** emitted from the current spec.
  - `PNextChain` owns heap-stable boxed nodes, relinks `pNext` pointers on push, and supports typed read-back (`get::<T>()` / `get_mut::<T>()`) for output-direction queries like `vkGetPhysicalDeviceMemoryProperties2` + `VK_EXT_memory_budget`.
  - Every ad-hoc pNext site in `vulkane` has been rewritten to use the chain (device creation, queue submit, semaphore create, memory allocate, memory-budget query).
- **Generated `DeviceFeatures`** (Layer 1) — `vulkan_gen` now emits one `with_<feature>()` builder method per unique feature bit across every struct that extends `VkPhysicalDeviceFeatures2`. **541 feature-bit methods** generated from the current spec. Name collisions between core-aggregate structs (`VkPhysicalDeviceVulkan12Features`) and promoted/extension structs (`VkPhysicalDeviceTimelineSemaphoreFeaturesKHR`) are resolved by routing the method to the highest-priority struct; the other path remains reachable via `chain_extension_feature()`.
- **Generated `DeviceExtensions` / `InstanceExtensions`** (Layer 3) — one `<vendor>_<ext>()` enable-method per non-disabled extension, with transitive `requires` resolved at generation time. **416 device + 44 instance** methods emitted from the current spec. Fresh extensions not yet in your copy of `vk.xml` are reachable through `enable_raw(name)`.
- **Generated RAII handle wrappers** (Layer 4 — Phase 1) — one safe, Drop-aware wrapper for every Vulkan handle type whose Create / Destroy pair fits the standard four-/three-parameter shape and isn't already covered by a hand-written wrapper. **25 new safe types** in `vulkane::safe::auto`, including `AccelerationStructureKHR`, `AccelerationStructureNV`, `MicromapEXT`, `VideoSessionKHR`, `VideoSessionParametersKHR`, `DeferredOperationKHR`, `DescriptorUpdateTemplate`, `PrivateDataSlot`, `ValidationCacheEXT`, `BufferView`, `SamplerYcbcrConversion`, `IndirectCommandsLayoutEXT/NV`, `IndirectExecutionSetEXT`, and more. Creating or destroying any of these previously required `unsafe { dispatch().vk… }` — now it's one safe call with automatic cleanup on drop.
- **`Allocation` now implements `Drop`** — a forgotten `allocator.free()` no longer leaks the slot in the TLSF pool. `AllocationInner` carries a `Weak<AllocatorInner>` back-reference, so the slot is reclaimed when the last `Arc<AllocationInner>` clone goes out of scope. `Allocator::free(allocation)` is kept for callers who prefer the imperative style — it now just `drop`s. `vulkane::safe::MemoryRequirements` is now re-exported from the crate root so call sites that build it directly don't need to reach into the buffer module.
- **Generated safe-method ext traits for every Vulkan command** (Layer 4 — Phase 2) — **600 safe methods** across 5 ext traits (`DeviceExt` 237, `CommandBufferRecordingExt` 266, `PhysicalDeviceExt` 78, `QueueExt` 15, `InstanceExt` 4). Every Vulkan command with a recognizable dispatch target now has a safe method — no `unsafe { dispatch().vkX.unwrap()(…) }` required anywhere in user code. Methods keep the `vk_` prefix (e.g. `vk_cmd_trace_rays_khr`), take raw Vulkan parameter types, and return `Result<VkResult>` for VkResult-returning commands (with error codes in `Err`, success codes like `VK_INCOMPLETE` / `VK_SUBOPTIMAL_KHR` in `Ok`). Users opt in per trait: `use vulkane::safe::CommandBufferRecordingExt;`. Ergonomic sugar (slice collapsing, typed output params, enumerate helpers) deferred to a future polish pass.
- `camel_to_snake` helper in `vulkan_gen::codegen` for consistent Vulkan identifier → Rust method-name conversion across generators.

### Breaking

- **`DeviceCreateInfo::enabled_extensions` is now `Option<&DeviceExtensions>`** (previously `&[&str]`). Migrate:

  ```rust
  // before
  let exts = [KHR_SWAPCHAIN_EXTENSION];
  DeviceCreateInfo { enabled_extensions: &exts, .. }
  // after
  let exts = DeviceExtensions::new().khr_swapchain();
  DeviceCreateInfo { enabled_extensions: Some(&exts), .. }
  ```

- **`InstanceCreateInfo::enabled_extensions` is now `Option<&InstanceExtensions>`** (previously `&[&str]`). Same migration pattern.
- **`DeviceFeatures` fields and hand-written builder methods are gone**, replaced by the 541 generated `with_<feature>()` methods. Callers who were constructing `DeviceFeatures { features11: …, features12: …, .. }` manually should use the builder instead. The generator picks names identical to pre-existing ones (`with_timeline_semaphore`, `with_buffer_device_address`, …) so most call sites are unaffected.
- **Hand-written extension-name constants removed** (`KHR_SURFACE_EXTENSION`, `KHR_SWAPCHAIN_EXTENSION`, `DEBUG_UTILS_EXTENSION`, `EXT_METAL_SURFACE_EXTENSION`, `KHR_WIN32/WAYLAND/XLIB/XCB_SURFACE_EXTENSION`). Use the generated `crate::raw::bindings::<NAME>_EXTENSION_NAME` constants or (preferred) the `<vendor>_<ext>()` builder methods.
- **`PNextChainable` requires `Clone + Default + 'static`** (previously `Default + 'static`). All `vk.xml`-generated structs derive `Clone`, so this is only a source-level break for code that hand-implemented the trait.

## [0.5.0] — 2026-04-16

### Added

- **`ShaderRegistry` for precompiled SPIR-V shaders** — new `vulkane::safe::shaders` module providing a small, shared abstraction for applications that ship compiled `.spv` artifacts (embedded via `include_bytes!` and/or loaded from disk).
  - `ShaderSource { name: &'static str, spv: &'static [u8] }` — one entry per compiled shader.
  - `ShaderRegistry::new().with_embedded(&[...]).with_env_override("MY_APP_OVERRIDE_DIR")` — builder-style setup.
  - `registry.load(name) -> Cow<'_, [u8]>` — bytes.
  - `registry.load_words(name) -> Vec<u32>` — SPIR-V word layout.
  - `registry.load_module(&device, name) -> ShaderModule` — full device-bound module in one call.
  - Runtime disk override: if the configured env var points at a directory and `<dir>/<name>.spv` exists, it is loaded instead of the embedded default; otherwise the registry falls through to the embedded table. Ideal for shader developers iterating without rebuilding the whole binary.

### Breaking

- **`Error::ShaderLoad` payload changed from `String` to `ShaderLoadError`.** The old variant preserved only a string description; the new one carries a structured enum (`NotFound` / `Io { name, source }` / `MalformedSpirv { name, byte_len }`) so consumers can match on the failure reason. Migration for manual constructions: convert `Error::ShaderLoad(format!("..."))` into the matching `ShaderLoadError` variant. Code that already used `From<ShaderLoadError> for Error` (via `?` on a `ShaderRegistry` call) needs no changes.

## [0.4.5] — 2026-04-15

### Added

- **Optional `slang` feature** — runtime Slang → SPIR-V compilation via the `shader-slang` crate (Khronos Slang compiler). Slang adds modules, generics, interfaces, and — most relevant for ML compute on Vulkan — built-in automatic differentiation: tag a function `[Differentiable]` and request forward and backward entry points from the same compiled module.
  - `vulkane::safe::slang::SlangSession::{new, with_search_paths, load_file}` — session-based API for compiling one module into many entry-point SPIR-V blobs.
  - `vulkane::safe::slang::SlangModule::compile_entry_point(name) -> Result<Vec<u32>, SlangError>`.
  - `vulkane::safe::slang::compile_slang_file(search_dir, module, entry)` — one-shot convenience.
  - Re-exports `CompileTarget`, `OptimizationLevel`, `Stage` from `shader-slang`.
  - New `Error::SlangCompile(String)` variant bridged from `SlangError`.
  - Requires `VULKAN_SDK` (SDK ships `slangc`) or `SLANG_DIR` at build/link time; `slang.dll` / `libslang.so` must be on the runtime library search path.
  - **Current limitation**: `shader-slang` 0.1.0 does not expose inline source compilation; Slang modules must live in `.slang` files resolved through session search paths. Will be lifted when a newer `shader-slang` ships.

## [0.4.4] — 2026-04-15

### Documentation

- Sync the crate-level `vulkane/README.md` (shown on crates.io and docs.rs) with the repo-root README: add the `shaderc` feature entry, the runtime-shader-compilation section, the naga-vs-shaderc selection table, and shaderc build requirements. The 0.4.3 release updated only the repo-root copy.

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
