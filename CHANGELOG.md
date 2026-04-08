# Changelog

All notable changes to spock will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial public release of spock — Vulkan API bindings generated entirely from the official `vk.xml` specification.
- **Safe wrapper module** (`spock::safe`) — RAII wrappers covering the **complete compute path**: `Instance`, `PhysicalDevice`, `Device`, `Queue`, `Buffer`, `DeviceMemory` (with `MappedMemory`), `ShaderModule` (takes `&[u32]` SPIR-V), `DescriptorSetLayout`, `DescriptorPool`, `DescriptorSet`, `PipelineLayout`, `ComputePipeline`, `CommandPool`, `CommandBuffer`, and `Fence`. Every handle is destroyed automatically via `Drop`. No manual `vkDestroy*` calls.
- **Optional `naga` feature** — pulls in `naga` 29 with `glsl-in` + `spv-out` only. Exposes `spock::safe::naga::compile_glsl(source, stage)` returning `Vec<u32>` SPIR-V. Disabled by default; users with their own SPIR-V pay nothing.
- **`fill_buffer` example** that exercises the safe wrapper end-to-end on a real GPU using `vkCmdFillBuffer`.
- **`compute_square` example** — complete compute round trip: loads pre-compiled SPIR-V, creates a storage buffer of 256 `u32`s, builds descriptor set + pipeline layout + compute pipeline, dispatches, and verifies the GPU squared every element. Validated on real hardware (NVIDIA RTX 4070) and on Lavapipe in CI.
- **`compile_shader` example** — one-shot helper (under the `naga` feature) that regenerates the pre-compiled `square_buffer.spv` from the GLSL source.
- **Validation layers + debug-utils messenger** — `InstanceCreateInfo::validation()` convenience enables `VK_LAYER_KHRONOS_validation` and `VK_EXT_debug_utils` together with a default `eprintln!`-style callback. For finer control, set `enabled_layers`, `enabled_extensions`, and `debug_callback` directly. The callback is invoked from a Rust closure (`Box<dyn Fn(&DebugMessage) + Send + Sync>`); the FFI plumbing (trampoline + leaked-box user-data) is handled internally and the messenger is destroyed automatically on `Instance` drop.
- **Instance / device extension and layer enable lists** on `InstanceCreateInfo` (`enabled_layers`, `enabled_extensions`) and `DeviceCreateInfo` (`enabled_extensions`).
- **Layer / extension enumeration** — `Instance::enumerate_layer_properties()`, `Instance::enumerate_extension_properties()`, and `PhysicalDevice::enumerate_extension_properties()` for runtime introspection of what the loader exposes before requesting it.
- **Compute essentials**:
  - **Push constants** — `PushConstantRange`, `PipelineLayout::with_push_constants`, and `CommandBufferRecording::push_constants` for the cheapest possible per-dispatch parameter passing.
  - **Specialization constants** — `SpecializationConstants` typed builder (`add_u32` / `add_i32` / `add_f32` / `add_bool`) and `ComputePipeline::with_specialization`. Lets shaders bake in workgroup sizes, unroll factors, dtype switches at pipeline creation time.
  - **Buffer-to-buffer copy** (`copy_buffer`) and `BufferCopy` region struct, enabling the staging-buffer pattern (`HOST_VISIBLE` upload → `DEVICE_LOCAL` copy).
  - **Indirect dispatch** (`dispatch_indirect`) — workgroup count read from a GPU buffer at dispatch time.
  - **Query pools** — `QueryPool::timestamps`, `QueryPool::pipeline_statistics`, `get_results_u64`, plus `reset_query_pool` and `write_timestamp` on the recording API.
  - **`PhysicalDevice::find_dedicated_compute_queue` / `find_dedicated_transfer_queue`** — prefer compute-without-graphics (async compute on NV/AMD) and transfer-without-compute (DMA engines on discrete GPUs), with sensible fallbacks.
  - **`PhysicalDevice::timestamp_period`** for converting GPU ticks to nanoseconds, and **`PhysicalDeviceProperties::max_push_constants_size`**.
- **2D storage images** — `Image`, `ImageView`, `Image2dCreateInfo`, `Format`, `ImageUsage`, `ImageLayout`, `ImageBarrier`, `BufferImageCopy`, plus the matching command-buffer methods `image_barrier`, `copy_buffer_to_image`, `copy_image_to_buffer`, and the `DescriptorSet::write_storage_image` write helper. Sufficient for the entire compute-on-2D-image flow: pre-fill via staging buffer, layout-transition `UNDEFINED → TRANSFER_DST → GENERAL`, dispatch a compute shader using `imageLoad` / `imageStore`, transition `GENERAL → TRANSFER_SRC`, copy back to a staging buffer, and read.
- **`compute_image_invert` example** — runs a tiny GLSL shader (`invert_image.comp`) on a 64×64 RGBA8 storage image and verifies every pixel's RGB was inverted by the GPU. Validated on real hardware.
- **Binary and timeline semaphores** — `Semaphore::binary(device)` and `Semaphore::timeline(device, initial_value)`, with `current_value`, `signal_value`, and `wait_value` for the timeline case. `Queue::submit_with_sync(cmds, wait_semaphores, signal_semaphores, fence)` accepts mixed binary/timeline wait/signal lists; the safe wrapper builds the `VkTimelineSemaphoreSubmitInfo` chain automatically when timeline semaphores are present.
- **Pipeline cache** — `PipelineCache::new` / `PipelineCache::with_data` to create an empty or pre-populated cache, `PipelineCache::data` to serialize for disk persistence, and `ComputePipeline::with_specialization_and_cache` to plug a cache into pipeline creation. Implementation-specific blob is opaque and silently ignored on driver mismatch, so it's safe to always pass any previously saved bytes.
- **Synchronization2** — `CommandBufferRecording::memory_barrier2` and `image_barrier2` use `vkCmdPipelineBarrier2` with 64-bit `VkPipelineStageFlags2` / `VkAccessFlags2`. Returns `MissingFunction` when the device doesn't expose `vkCmdPipelineBarrier2` (Vulkan 1.0/1.1 without `VK_KHR_synchronization2`); the legacy `memory_barrier` and `image_barrier` continue to work everywhere.
- **VMA-style sub-allocator** (`spock::safe::Allocator`) — pools many sub-allocations into a small number of `vkAllocateMemory` blocks so apps don't hit `maxMemoryAllocationCount` (~4096 on most drivers). Highlights:
  - **Two-Level Segregated Fit** (TLSF) free-list per block: O(1) allocation and free, low fragmentation, the same algorithm AMD's VMA uses.
  - **Memory-type selection** by `(required, preferred)` property flag pair, driven by the `AllocationUsage` hint (`Auto`, `DeviceLocal`, `HostVisible`, `HostVisibleDeviceLocal`).
  - **Dedicated allocations** when the user opts in (`AllocationCreateInfo::dedicated = true`) or when the request is larger than half the block size; falls through to a direct `vkAllocateMemory`.
  - **Persistent mapping** — pass `mapped: true` for host-visible allocations and `Allocation::mapped_ptr()` returns a stable pointer for the lifetime of the allocation.
  - **`Allocator::create_buffer` / `Allocator::create_image_2d`** — convenience constructors that build the resource, call `vkBind*Memory` with the right offset, and return the `(Resource, Allocation)` pair.
  - **Statistics**: `AllocationStatistics` exposes `block_bytes`, `allocation_bytes`, `block_count`, `allocation_count`, `free_region_count`, `peak_allocation_bytes`, and `dedicated_allocation_count`.
  - **Default block sizes**: 256 MiB on heaps ≥ 4 GiB, 64 MiB on smaller heaps — same as VMA.
  - 9 unit tests for TLSF + linear strategies (alignment, no-overlap, fragmentation, full-pool failure, reset) and 7 integration tests on real Vulkan (creation, pool sub-allocation reuse, dedicated path, image-via-pool, persistent mapped pointer, peak watermark, many-small-buffers-share-one-block).
- **Buffer device addresses** — `BufferUsage::SHADER_DEVICE_ADDRESS` flag and `Buffer::device_address()` method. Returns the GPU virtual address of a buffer for use with `VK_KHR_buffer_device_address` (Vulkan 1.2 core) shaders. Surfaces a clean `MissingFunction` error when the device wasn't created with the `bufferDeviceAddress` feature enabled.
- **`PhysicalDevice::cooperative_matrix_properties`** — query the supported `(M, N, K, A_type, B_type, C_type, Result_type, scope)` shapes via `VK_KHR_cooperative_matrix`, the building blocks for native GPU matrix-multiply-and-accumulate (the foundation for ML workloads). Returns an empty `Vec` cleanly when the extension isn't enabled. New `CooperativeMatrixProperties` accessor type. Marked `unsafe` because calling the underlying function without the instance extension enabled is undefined behaviour on some implementations.
- **Device features enable list** — new `DeviceFeatures` builder type wrapping `VkPhysicalDeviceFeatures` plus the Vulkan 1.1 / 1.2 / 1.3 aggregate feature structs. `DeviceCreateInfo::enabled_features` accepts an `Option<&DeviceFeatures>` and the safe wrapper builds the `VkPhysicalDeviceFeatures2` `pNext` chain for `vkCreateDevice` automatically. Builder methods include `with_buffer_device_address`, `with_timeline_semaphore`, `with_synchronization2`, `with_dynamic_rendering`, `with_descriptor_indexing`, `with_runtime_descriptor_array`, `with_shader_int8`, `with_shader_float16`, `with_sampler_anisotropy`, and several more. With features enabled the `Buffer::device_address`, `Semaphore::timeline`, and `memory_barrier2` paths can finally be exercised end-to-end. Also adds `PhysicalDevice::supported_features` for runtime feature support introspection.
- **Lavapipe integration in CI** — Linux runners install Mesa's software Vulkan implementation so the real-Vulkan integration tests actually exercise the loader on every CI run.
- **Strict clippy on CI** — `cargo clippy --workspace -- -D warnings` is enforced.
- Tree-based XML parser using `roxmltree` that correctly extracts nested struct members and command parameters.
- Code generator producing `#[repr(C)]` structs with correct pointer/array/const field types.
- Code generator producing `#[repr(i32)]` enums with extension values merged into base enums.
- Code generator producing `pub union` for Vulkan union types (e.g., `VkClearColorValue`).
- Function pointer typedefs with `unsafe extern "system"` calling convention.
- Three-tier dispatch tables (`VkEntryDispatchTable`, `VkInstanceDispatchTable`, `VkDeviceDispatchTable`) generated from vk.xml.
- `VulkanLibrary` runtime loader with `load_entry`, `load_instance`, and `load_device` methods.
- `VkResultExt` trait with `into_result()`, `is_success()`, `is_error()` for ergonomic `?` propagation.
- `VkResult` implements `std::error::Error` for use with `Box<dyn Error>` return types.
- `vk_check!` macro for one-line Vulkan call validation.
- Doc comments harvested from vk.xml `comment` attributes and `<comment>` child elements.
- C-to-Rust transpiler for `VK_MAKE_API_VERSION` and other version-encoding macros — bit positions are derived from vk.xml, not hardcoded.
- `fetch-spec` Cargo feature for automatic vk.xml download from the Khronos GitHub repository.
- `VK_VERSION` environment variable to pin a specific Vulkan version (e.g., `VK_VERSION=1.3.250`).
- `VK_XML_PATH` environment variable to use a local vk.xml at any path.
- `device_info` example demonstrating instance creation, physical device enumeration, memory queries, queue family inspection, and logical device creation/destruction.
- Real-Vulkan integration test that validates the entire pipeline against actual GPU drivers (skips gracefully when no driver is installed).
- GitHub Actions CI matrix on Linux, Windows, and macOS.

### Supported Vulkan Versions

- Minimum: **Vulkan 1.2.175** (the first version with `VK_MAKE_API_VERSION` macros).
- Maximum: latest from the Khronos `Vulkan-Docs` `main` branch.

### Notes

- Spock generates ~52,000 lines of Rust bindings covering ~1,478 structs, ~1,343 type aliases, ~148 Rust enums, ~3,064 constants, and ~657 function pointer typedefs from a typical recent vk.xml.
- Nothing about Vulkan is hardcoded in spock's source code. To target a new Vulkan version, swap the vk.xml file (or set `VK_VERSION`) and rebuild.
- All structs implement `Default`. To construct a `Vk*CreateInfo` ergonomically,
  set just the `sType` and required fields and use `..Default::default()` for
  the rest:

  ```rust
  let create_info = VkInstanceCreateInfo {
      sType: VkStructureType::STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      pApplicationInfo: &app_info,
      ..Default::default()
  };
  ```

### Known limitations

- No builder pattern helpers (`VkInstanceCreateInfo::builder()...`). Use the
  `..Default::default()` shorthand above instead. Builder generation is on the
  roadmap but adds significant generated-code volume.
- The safe wrapper module covers the **complete compute path** but does not
  yet cover graphics-specific functionality: surfaces, swapchains, render
  passes, graphics pipelines, images for color attachments, or samplers.
  Use `spock::raw` for those use cases.
- The real-Vulkan integration tests run on Linux CI runners via Lavapipe
  (Mesa's software rasterizer). Windows and macOS CI runners don't have a
  Vulkan ICD by default, so the integration tests skip gracefully there.
