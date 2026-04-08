# Spock

Raw Vulkan API bindings for Rust, generated entirely from the official `vk.xml` specification.

[![CI](https://github.com/ciresnave/spock/actions/workflows/ci.yml/badge.svg)](https://github.com/ciresnave/spock/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/spock.svg)](https://crates.io/crates/spock)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

## What is Spock?

Spock is a Rust crate that generates **complete** Vulkan API bindings from `vk.xml`, the
official machine-readable Vulkan specification maintained by Khronos. Every type, constant,
struct, enum, function pointer, and dispatch table is derived from the XML at build time.
**Nothing is hardcoded.**

To target a different Vulkan version, swap `vk.xml` (or set the `VK_VERSION` environment
variable) and rebuild — that's the entire upgrade procedure.

Spock exposes Vulkan through two complementary APIs:

- [`spock::raw`](spock/src/raw) — direct FFI bindings, exactly as the spec defines them.
  Maximum control, zero overhead.
- [`spock::safe`](spock/src/safe) — RAII wrappers with automatic cleanup, `Result`-based
  error handling, and type-safe enums. Covers the **complete compute path** *and* the
  **complete graphics path**:

  - **Instance / Device** — `Instance`, `InstanceCreateInfo` (with optional validation
    layer + debug-utils messenger), `PhysicalDevice`, `PhysicalDeviceGroup`, `Device`
    (always backed by a possibly-singleton physical-device group), `Queue`,
    `DeviceFeatures` builder for the Vulkan 1.0/1.1/1.2/1.3 feature chain.
  - **Memory** — `Buffer`, `Image` (2D, color/depth attachment, storage, sampled),
    `ImageView`, `Sampler`, `DeviceMemory`, plus a **VMA-style sub-allocator**
    (`Allocator`) with TLSF + linear pools, custom user pools, dedicated allocations,
    persistent mapping, defragmentation, and memory budget queries.
  - **Compute** — `ComputePipeline` (with specialization constants and pipeline
    cache), `PipelineLayout` (with push constants), `DescriptorSetLayout`,
    `DescriptorPool`, `DescriptorSet` (storage buffer / uniform buffer / storage
    image / combined image sampler), `ShaderModule` (takes `&[u32]` SPIR-V), with an
    optional `naga` feature for GLSL → SPIR-V at runtime.
  - **Graphics** — `RenderPass`, `Framebuffer`, `GraphicsPipeline` (with a focused
    `GraphicsPipelineBuilder`), `Surface` (Win32 / Wayland / Metal), `Swapchain`
    with the standard acquire / submit / present semaphore loop.
  - **Synchronization** — `Fence`, `Semaphore` (binary and timeline), command-buffer
    `memory_barrier` / `image_barrier` plus their `Synchronization2` 64-bit
    counterparts, `QueryPool` (timestamps + pipeline statistics), and the Vulkan 1.2
    `vkGetBufferDeviceAddress` path for bindless / pointer-chasing compute kernels.

## Why Spock?

If you're already using `ash`, here's the trade-off:

| Aspect | spock | ash |
| --- | --- | --- |
| Source of truth | `vk.xml` (parsed at build time) | Hand-curated bindings module |
| New Vulkan version support | Swap vk.xml, rebuild | Wait for crate update |
| New extension support | Automatic on next vk.xml fetch | Wait for crate update |
| Generated lines of Rust | ~52,000 | ~30,000 |
| Hand-written FFI | None | Some |
| Safe-API surface | Compute + graphics RAII layer in `spock::safe` (instance, device, buffer, image, sampler, render pass, framebuffer, graphics + compute pipelines, swapchain, allocator, sync, queries) | Raw FFI; safe wrappers come from third-party crates (`vulkano`, `vulkanalia`) |
| Sub-allocator | TLSF + linear pools, custom user pools, defragmentation, memory budget queries built into `spock::safe::Allocator` | None — you BYO via `gpu-allocator` or `vk-mem` |
| GLSL→SPIR-V at runtime | Optional `naga` feature | None |
| Maturity | New | Battle-tested |

**Spock is the right choice if:**

- you want bindings that always match the Vulkan version your driver supports
- you want to opt into new extensions the moment Khronos publishes them
- you don't want to depend on a third party shipping updates
- you want to read the Vulkan spec and have confidence the bindings reflect it exactly

**Ash is the right choice if:**

- you want maximum maturity and a large body of existing example code
- you prefer ergonomic builders over raw FFI structs
- you're building higher-level wrappers and want a stable foundation

The two crates are not mutually exclusive. Spock targets the same C ABI as the Vulkan spec,
so a downstream crate can build a safe wrapper layer on top of either.

## First impression — safe API

```rust
use spock::safe::{
    ApiVersion, Buffer, BufferCreateInfo, BufferUsage, CommandPool, DeviceCreateInfo,
    DeviceMemory, Fence, Instance, InstanceCreateInfo, MemoryPropertyFlags, QueueCreateInfo,
    QueueFlags,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load Vulkan and create an instance — RAII handles cleanup.
    let instance = Instance::new(InstanceCreateInfo {
        application_name: Some("hello-spock"),
        api_version: ApiVersion::V1_0,
        ..Default::default()
    })?;

    // Pick a physical device with a transfer-capable queue.
    let physical = instance
        .enumerate_physical_devices()?
        .into_iter()
        .find(|pd| pd.find_queue_family(QueueFlags::TRANSFER).is_some())
        .ok_or("no compatible GPU")?;
    let qf = physical.find_queue_family(QueueFlags::TRANSFER).unwrap();

    // Create a logical device with one queue.
    let device = physical.create_device(DeviceCreateInfo {
        queue_create_infos: &[QueueCreateInfo {
            queue_family_index: qf,
            queue_priorities: vec![1.0],
        }],
    })?;
    let queue = device.get_queue(qf, 0);

    // Allocate a host-visible buffer.
    let buffer = Buffer::new(&device, BufferCreateInfo {
        size: 1024,
        usage: BufferUsage::TRANSFER_DST,
    })?;
    let req = buffer.memory_requirements();
    let mt = physical
        .find_memory_type(req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT)
        .unwrap();
    let mut memory = DeviceMemory::allocate(&device, req.size, mt)?;
    buffer.bind_memory(&memory, 0)?;

    // Record vkCmdFillBuffer and submit it to the GPU.
    let pool = CommandPool::new(&device, qf)?;
    let mut cmd = pool.allocate_primary()?;
    cmd.begin()?.fill_buffer(&buffer, 0, 1024, 0xDEADBEEF);
    let fence = Fence::new(&device)?;
    queue.submit(&[&cmd], Some(&fence))?;
    fence.wait(u64::MAX)?;

    // Read it back from the host side.
    let mapped = memory.map()?;
    assert_eq!(&mapped.as_slice()[..4], &0xDEADBEEFu32.to_ne_bytes());
    println!("GPU filled the buffer with 0xDEADBEEF");

    // Everything drops here in the right order — no manual vkDestroy* calls.
    Ok(())
}
```

## Bundled examples

| Example | What it shows |
| --- | --- |
| [`device_info`](spock/examples/device_info.rs) | Raw-API instance creation, physical device enumeration, queue family inspection |
| [`fill_buffer`](spock/examples/fill_buffer.rs) | Safe-API: full host→GPU→host round trip via `vkCmdFillBuffer` |
| [`compute_square`](spock/examples/compute_square.rs) | Safe-API: complete compute path — load SPIR-V, descriptor set, compute pipeline, dispatch, verify the GPU squared every element |
| [`compute_image_invert`](spock/examples/compute_image_invert.rs) | Safe-API: 2D storage image compute — RGBA8 round trip with layout transitions, copy buffer↔image, dispatch invert shader, verify per-pixel |
| [`compile_shader`](spock/examples/compile_shader.rs) (`--features naga`) | Compile every `*.comp` / `*.vert` / `*.frag` under `examples/shaders/` to SPIR-V using the optional `naga` feature |
| [`headless_triangle`](spock/examples/headless_triangle.rs) | Safe-API: full graphics pipeline — render pass, framebuffer, graphics pipeline, vertex/fragment shaders, draw, copy back, verify pixels were rasterized |
| [`windowed_triangle`](spock/examples/windowed_triangle.rs) | Safe-API: opens a real OS window via `winit`, creates a Win32/Wayland/Metal surface, builds a swapchain, and runs the standard acquire/submit/present loop with two frames in flight |

The compute examples and the headless triangle run in CI on every platform via Mesa
Lavapipe; the windowed triangle is built but not run in CI (it requires a display
server). To run any of them locally:

```bash
cargo run -p spock --features fetch-spec --example headless_triangle
cargo run -p spock --features fetch-spec --example compute_square
cargo run -p spock --features fetch-spec --example windowed_triangle  # opens a window
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
spock = "0.1"
```

If you don't have a local copy of `vk.xml` (most people don't), enable auto-download:

```toml
[dependencies]
spock = { version = "0.1", features = ["fetch-spec"] }
```

## Providing vk.xml

The build script resolves the Vulkan specification in this order:

1. **`VK_XML_PATH` environment variable** — point to any local `vk.xml` file:

   ```bash
   VK_XML_PATH=/path/to/vk.xml cargo build
   ```

2. **Local copy** — place `vk.xml` at `spec/registry/Vulkan-Docs/xml/vk.xml` relative to the workspace root.

3. **Auto-download** (requires `fetch-spec` feature) — downloads from the Khronos GitHub repository:

   ```bash
   # Download the latest version
   cargo build --features fetch-spec

   # Pin to a specific version
   VK_VERSION=1.3.250 cargo build --features fetch-spec
   ```

   Pinned versions are cached permanently; the latest is refreshed after 24 hours.

## Supported Vulkan Versions

Spock supports Vulkan specification versions **1.2.175** through the latest release.

Version 1.2.175 is the minimum because it's the first version that introduced the
`VK_MAKE_API_VERSION` / `VK_API_VERSION_*` macro family, which replaced the deprecated
`VK_MAKE_VERSION` / `VK_VERSION_*` macros. Spock transpiles these macros from C to Rust
`const fn` at build time, so they must be present in the specification.

| Vulkan version | Status |
| --- | --- |
| 1.2.175 (minimum) | Tested |
| 1.3.250 | Tested |
| 1.4.348 | Tested |
| latest (main branch) | Tested |

## What Gets Generated

Everything in this table is derived entirely from `vk.xml` at build time:

| Category | Count (recent vk.xml) | Description |
| --- | --- | --- |
| Structs | ~1,478 | `#[repr(C)]` with correct pointer/array/const field types |
| Unions | ~14 | `#[repr(C)] pub union` with `unsafe { mem::zeroed() }` defaults |
| Type aliases | ~1,343 | Dispatchable handles (`*mut c_void`), non-dispatchable handles (`u64`), bitmasks, base types |
| Rust enums | ~148 | `#[repr(i32)]` with extension values merged from all extensions |
| Constants | ~3,064 | Including bitmask flag values emitted as `pub const` |
| Function pointer typedefs | ~657 | `unsafe extern "system" fn(...)` for every Vulkan command |
| Dispatch tables | 3 | Entry, instance, and device tables generated from first-parameter classification |
| Version functions | All | `vk_make_api_version`, `vk_api_version_major`, etc. transpiled from C macros |
| Doc comments | ~960 | Harvested from vk.xml `<comment>` and `comment` attributes |

## Features

| Feature | Description |
| --- | --- |
| `build-support` (default) | Enables XML parsing and code generation during build |
| `fetch-spec` | Enables automatic download of `vk.xml` from the Khronos GitHub repository |
| `naga` | Pulls in `naga` 29 with `glsl-in` + `spv-out` only. Exposes `spock::safe::naga::compile_glsl(source, stage) -> Vec<u32>` for runtime GLSL→SPIR-V compilation. Disabled by default — users with their own SPIR-V pay nothing. |

## Loader API

The runtime loader is a thin wrapper around `libloading`:

```rust
use spock::raw::VulkanLibrary;

let library = VulkanLibrary::new()?;             // dlopen vulkan-1.dll / libvulkan.so.1
let entry = unsafe { library.load_entry() };     // global functions
let inst  = unsafe { library.load_instance(instance) };       // instance functions
let dev   = unsafe { library.load_device(instance, device) }; // device functions
```

Each dispatch table contains every relevant Vulkan command as an `Option<fn_ptr>` field.
Functions that the loaded Vulkan implementation doesn't support are `None`.

## Result helpers

```rust
use spock::raw::VkResultExt;

unsafe { vkCreateInstance(&info, std::ptr::null(), &mut instance) }
    .into_result()?;  // VkResult::SUCCESS -> Ok(()), anything else -> Err(VkResult)
```

`VkResult` implements `std::error::Error` so you can use `?` directly with
`Box<dyn Error>` return types.

## Architecture

```text
       vk.xml
         │
         ▼
  vulkan_gen crate
   ┌──────────────┐
   │ tree_parser  │  (roxmltree DOM parser)
   │      │       │
   │      ▼       │
   │  vk_types    │  (parsed data structures)
   │      │       │
   │      ▼       │
   │   codegen    │  (8 generator modules + assembler)
   └──────┬───────┘
          │
          ▼
    spock crate
   ┌─────────────────────────┐
   │  build.rs               │  resolves vk.xml, calls
   │                         │  vulkan_gen::generate_bindings
   │                         │
   │  raw/                   │
   │   bindings.rs (~52k LoC, generated from vk.xml)
   │   loader.rs   (VulkanLibrary, dispatch tables)
   │   result.rs   (VkResultExt, vk_check!)
   │                         │
   │  safe/                  │
   │   instance.rs / device.rs / physical.rs
   │   buffer.rs / image.rs / memory.rs / sampler
   │   render_pass.rs / graphics_pipeline.rs
   │   pipeline.rs (compute) / descriptor.rs / shader.rs
   │   surface.rs / swapchain.rs
   │   command.rs / sync.rs / query.rs
   │   features.rs / debug.rs
   │   allocator/ (TLSF + linear + dedicated, defrag,
   │              custom user pools, statistics, budget)
   │   naga.rs (optional GLSL → SPIR-V via the `naga` feature)
   └─────────────────────────┘
```

The parser uses DOM-based XML parsing (`roxmltree`) to correctly handle nested elements
like `<member>` and `<param>` that contain mixed text and child element content. The
safe wrapper layer is hand-written on top of the generated raw bindings — every safe
type is a thin RAII shell around a `VkFoo` handle plus an `Arc<DeviceInner>` (or
`Arc<InstanceInner>`) for parent lifetime tracking.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in spock by you, as defined in the Apache-2.0 license, shall be dual-licensed
as above, without any additional terms or conditions.
