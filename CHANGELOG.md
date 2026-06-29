# Changelog

All notable changes to vulkane will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.3] — 2026-06-28

### Added — physical-device identity

- `PhysicalDevice::device_identity() -> Option<DeviceIdentity>` exposes the device's stable identity for out-of-band correlation: `device_uuid` / `driver_uuid` (always, from `VkPhysicalDeviceIDProperties`, Vulkan 1.1 core), `device_luid` (`Some` only when the platform marks it valid — Windows) plus its `device_node_mask`, and `pci: Option<PciBusInfo>` (`Some` only when the device advertises `VK_EXT_pci_bus_info`). One `vkGetPhysicalDeviceProperties2` call, gated honestly: `None` when props2 is unavailable, and each sub-field is `Some` only when its source is actually present. This is the *join key* a caller needs to match a `VkPhysicalDevice` against an out-of-band GPU source — NVML by UUID, DXGI/D3DKMT by LUID, Linux sysfs (`gpu_busy_percent`) by PCI address — or against the same device seen through CUDA/D3D/OpenGL. Added because Vulkan exposes **no** cross-process GPU load / utilization / queue-depth query beyond the VRAM `memory_budget`; identity is the most Vulkane can (and should) provide toward that, with the load lookup itself living in a separate, API-agnostic layer. New public types `safe::DeviceIdentity` and `safe::PciBusInfo`.

### Added — Profile v1 conformance lock-in

- Vulkane is confirmed conformant to Fuel's **Kernel-Seam Interop Contract — Profile v1** (ratified 2026-06-20) in its **FDX-only, BDA-subset** role. No API change was required — the contract pins Vulkane to a *named surface*, all of which shipped in 0.8.2: `AllocatorOptions::buffer_device_address` / `Allocator::new_with_options`, `BufferUsage::SHADER_DEVICE_ADDRESS`, `DeviceFeatures::with_buffer_device_address`, and `Buffer::device_address`. Added [`tests/profile_v1_conformance.rs`](vulkane/tests/profile_v1_conformance.rs), a compile-time lock-in (mirroring the `Send + Sync` lock-ins on `Queue` / `CommandBuffer`) so a future rename, removal, or signature change of any named-surface item fails Vulkane's CI rather than `fuel-vulkan-backend`'s build. This operationalizes the contract's §7.2 rule that *a Vulkane major bump triggers a re-check of the named surface* — the surface is pinned by behavior, not by a `>= 0.8.2` version floor.

## [0.8.2] — 2026-06-19

### Added — device-address-capable allocator

- `Allocator::new_with_options(device, physical, AllocatorOptions { buffer_device_address: true })` makes every `VkDeviceMemory` block the allocator allocates carry `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`. Buffers sub-allocated from such an allocator (created with `BufferUsage::SHADER_DEVICE_ADDRESS`) now return a valid GPU virtual address from `Buffer::device_address()`. Previously the flag was only set on the manual `DeviceMemory::allocate_with` path, so addresses read from pooled or `Buffer::new_bound` buffers were invalid on strict drivers. The flag lives on the block (not the buffer) because one block backs many sub-allocations, mirroring VMA's `VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT`. Requires the `bufferDeviceAddress` device feature. `Allocator::new` is unchanged (defaults to `buffer_device_address: false`). Unblocks downstream consumers (Fuel) that address tensors via `buffer_reference` in shaders, where the tensor's data pointer is a `VkDeviceAddress`.

## [0.8.1] — 2026-05-21

### Added — thread-safety markers

- `safe::Queue` is now `Send + Sync` via `unsafe impl`. Vulkan queues have no thread affinity at the API level, but the application owns external synchronization per `VkQueue` for `vkQueueSubmit` / `vkQueueWaitIdle` / `vkQueueBindSparse` / `vkQueuePresentKHR`. Callers can share `&Queue` across threads (e.g. via `Arc<Mutex<Queue>>` or a scheduler) so long as concurrent submissions on the same queue handle are prevented. Unblocks downstream consumers (Fuel) that build work on worker threads and submit through a serializer.
- `safe::CommandBuffer` is now `Send + Sync` via `unsafe impl`, for the same reason. Recording APIs take `&mut self`, so the Rust borrow checker already prevents concurrent recording on the same buffer; sharing `&CommandBuffer` across threads is sound. The per-pool external-sync contract for `vkFreeCommandBuffers` (called from `Drop`) remains the caller's responsibility.
- Compile-time `Send + Sync` lock-in assertions added for both types so future field additions cannot silently regress this guarantee.

## [0.8.0] — 2026-04-21

Coverage release: full ray-tracing surface, external-memory / external-semaphore interop, synchronization-2 barriers, push descriptors, dynamic rendering, descriptor-buffer binding, timeline semaphores, subgroup-size control, memory priority, and generator-emitted ergonomic safe signatures for ~545 Vulkan commands. Two latent codegen correctness bugs fixed along the way.

### Added — curated extension wrappers

- **Ray tracing** — `safe::AccelerationStructure` (BLAS/TLAS/Generic, AABB + Triangles + Instances geometry), `safe::RayTracingPipeline` with `ShaderGroup` enum (General/TrianglesHit/ProceduralHit) and `ShaderBindingRegion`, `PhysicalDevice::ray_tracing_pipeline_properties`, `CommandBufferRecording::build_acceleration_structure` / `bind_ray_tracing_pipeline` / `trace_rays`, `Device::acceleration_structure_build_sizes`. Examples: [`ray_tracing_as_build`](vulkane/examples/ray_tracing_as_build.rs) builds a live BLAS + TLAS on the local GPU end-to-end.
- **External memory / semaphore interop** — `DeviceMemory::get_win32_handle` / `get_fd`, `Semaphore::get_win32_handle` / `get_fd` / `import_win32_handle` / `import_fd`, `Win32Handle` newtype, `SemaphoreImportWin32` / `SemaphoreImportFd`. Unblocks CUDA, HIP, DX12, and DMA-BUF bridging. Example: [`external_memory_export`](vulkane/examples/external_memory_export.rs).
- **Synchronization 2** — `CommandBufferRecording::memory_barrier2` / `image_barrier2` / `buffer_barrier2` with 64-bit `PipelineStage2` / `AccessFlags2`.
- **Dynamic rendering** — `CommandBufferRecording::begin_rendering` / `end_rendering` with `RenderingInfo` + `RenderingAttachment`.
- **Push descriptors** — `CommandBufferRecording::push_descriptor_set` taking a `&[PushDescriptorWrite]` that hides the `VkWriteDescriptorSet` layout.
- **Descriptor buffer** (`VK_EXT_descriptor_buffer`) — `DescriptorSetLayout::descriptor_buffer_size` / `descriptor_buffer_binding_offset` queries, `CommandBufferRecording::bind_descriptor_buffers` / `set_descriptor_buffer_offsets`.
- **Timeline semaphores** — `Semaphore::timeline_with_pnext` composes caller-supplied chains with the mandatory `VkSemaphoreTypeCreateInfo`.
- **Compute pipeline options** — `ComputePipelineOptions` carries `required_subgroup_size` (`VK_EXT_subgroup_size_control`), `specialization`, and `cache` in one bag; `ComputePipeline::with_options` is the general constructor.
- **Memory priority** — `MemoryAllocateInfo::priority: Option<f32>` auto-chains `VkMemoryPriorityAllocateInfoEXT`.
- **Shader integer dot product** — `PhysicalDevice::shader_integer_dot_product_properties() -> ShaderIntegerDotProductProperties` with `has_any_int8_acceleration()` helper.
- **pNext extension points** on every safe create-info builder: `DeviceCreateInfo::pnext`, `InstanceCreateInfo::pnext`, `MemoryAllocateInfo::pnext`, plus new `with_pnext` constructors on `Buffer`, `Image`, `Fence`, and `Semaphore`. Any unwrapped extension struct can now be layered on without dropping to raw.

### Added — generated ergonomic traits (Phase 3)

- `DeviceSafeExt`, `InstanceSafeExt`, `PhysicalDeviceSafeExt`, `QueueSafeExt`, `CommandBufferRecordingSafeExt` — auto-generated per-command methods with idiomatic Rust signatures alongside the raw Phase-2 `DeviceExt` etc. traits. **545 ergonomic methods** emitted from `vk.xml`:
  - **Slice coalescing** — `(count: u32, const T*)` pairs collapse into `&[T]` inputs. `cmd_pipeline_barrier(..., &[MemoryBarrier], &[BufferMemoryBarrier], &[ImageMemoryBarrier])` is one signature.
  - **Enumerate** — `(*mut u32 count, *mut T data)` pairs become `Result<Vec<T>>` / `Vec<T>` return types. `enumerate_physical_devices` issues the classic two-call count-then-fill idiom automatically.
  - **Single-output** — trailing `*mut T` parameters become `Result<T>` returns (`get_memory_win32_handle_khr(info: &…) -> Result<HANDLE>`).
  - **Reference input structs** — `*const T` parameters become `&T`.
  - **Scalar return passthrough** — `VkDeviceAddress` / `VkBool32` / typed handles pass through untouched (`get_buffer_device_address(info: &…) -> VkDeviceAddress`).
  - Commands with unsupported shapes (pointer-to-pointer, parallel slices sharing one count, `len` pointing inside a struct) fall through to the raw Phase-2 traits — no method emitted.

### Fixed — generator correctness

- **Nested C-array layout** — `VkTransformMatrixKHR.matrix[3][4]` was emitted as `[f32; 3]` (12 bytes) instead of `[[f32; 4]; 3]` (48 bytes). Every multi-dimensional `float matrix[a][b]` field in `vk.xml` was silently truncated. Fixed in `struct_gen::map_type_to_rust`; any ray-tracing workload using `VkAccelerationStructureInstanceKHR` was affected.
- **Transitive extension-dep walker** — `transitive_requires` harvested per-`<require>` `depends` attributes and treated them as extension prerequisites. In `vk.xml` those attributes mark *conditional* enum inclusion (e.g. "expose these extra debug-report enums if the user also enables debug_report"), not dependencies. Enabling `VK_KHR_acceleration_structure` therefore silently tried to enable `VK_EXT_debug_report` (an *instance* extension) at device creation, causing `ERROR_EXTENSION_NOT_PRESENT` on every driver. Fixed to use only the canonical top-level `requires` attribute.

### Breaking

- `DeviceCreateInfo` gained a `pnext: Option<&PNextChain>` field (default `None`). Callers using `..Default::default()` are unaffected; anyone constructing the struct with explicit named fields must add it (or switch to update syntax).
- `InstanceCreateInfo` gained the same `pnext` field.
- `MemoryAllocateInfo` gained `pnext` and `priority` fields. Direct struct-literal callers must supply both or use `..Default::default()` — the struct now derives `Default`.
- `CommandBufferRecording::memory_barrier2` / `image_barrier2` / `buffer_barrier2` return `Result<()>` (Sync2 function pointers may be absent on pre-1.3 devices without `VK_KHR_synchronization2`). Previously no sync2 methods existed, so this is only new-code exposure.
- `ComputePipeline::with_specialization_and_cache` is retained as a shim; new callers should prefer `ComputePipeline::with_options`.

### Test + example coverage

- 249 total workspace tests pass. 10 new generator pattern-matcher unit tests. 5 new live-device tests exercising generated ergonomic traits against a real driver. 2 new example programs that **run live** against the local GPU — `external_memory_export` exports a real Win32 HANDLE, `ray_tracing_as_build` builds a real BLAS + TLAS on the RT hardware.

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
