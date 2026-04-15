# Vulkane

Vulkan for Rust: complete bindings generated from the official `vk.xml`
specification, plus a safe RAII wrapper that covers compute and graphics
end-to-end — from instance creation through shadow mapping and deferred
shading.

[![CI](https://github.com/ciresnave/vulkane/actions/workflows/ci.yml/badge.svg)](https://github.com/ciresnave/vulkane/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/vulkane.svg)](https://crates.io/crates/vulkane)
[![docs.rs](https://docs.rs/vulkane/badge.svg)](https://docs.rs/vulkane)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

## What is Vulkane?

Vulkane generates **complete** Vulkan API bindings from `vk.xml`, the
official machine-readable specification maintained by Khronos. Every
type, constant, struct, enum, function pointer, and dispatch table is
derived at build time. **Nothing is hardcoded.**

To target a different Vulkan version, swap `vk.xml` and rebuild.

Vulkane exposes Vulkan through two complementary APIs:

- **`vulkane::raw`** — direct FFI bindings, exactly as the spec
  defines them. Maximum control, zero overhead.
- **`vulkane::safe`** — RAII wrappers with automatic cleanup,
  `Result`-based error handling, typed flags, and convenience
  helpers. Covers compute **and** graphics:

  - **Instance / Device** — `Instance`, `PhysicalDevice`,
    `PhysicalDeviceGroup`, `Device`, `Queue`, `DeviceFeatures`
    builder (1.0 / 1.1 / 1.2 / 1.3 features).
  - **Memory** — `Buffer`, `Image`, `ImageView`, `Sampler`,
    `DeviceMemory`, plus a **VMA-style sub-allocator** (`Allocator`)
    with TLSF + linear pools, custom pools, dedicated allocations,
    persistent mapping, defragmentation, and budget queries.
  - **Convenience helpers** — `Buffer::new_bound`,
    `Image::new_2d_bound`, `Queue::upload_buffer<T>`,
    `Queue::upload_image_rgba`, `Queue::one_shot` — collapse the
    5-step allocate-bind pattern into one call.
  - **Compute** — `ComputePipeline`, `PipelineLayout`,
    `DescriptorSet`, `ShaderModule`, specialization constants,
    pipeline cache, push constants.
  - **Graphics** — `RenderPass` (with `simple_color` shortcut),
    `Framebuffer`, `GraphicsPipelineBuilder` (depth bias, `CompareOp`,
    `InputRate`, multi-attachment blend, dynamic viewport/scissor),
    `Surface` (Win32 / Wayland / Xlib / Xcb / Metal), `Swapchain`.
  - **Synchronization** — typed `PipelineStage` / `AccessFlags`
    (plus 64-bit `PipelineStage2` / `AccessFlags2` for Sync2),
    `Fence`, `Semaphore` (binary + timeline), `ImageBarrier::color`
    / `::depth`, `ClearValue`, `QueryPool`.
  - **Derive macros** — `#[derive(Vertex)]` auto-generates vertex
    input layouts from `#[repr(C)]` structs (optional `derive`
    feature).
  - **Raw escape hatch** — `Device::dispatch()` /
    `Instance::dispatch()` expose the full dispatch tables so you
    can drop to raw Vulkan for anything the safe wrapper doesn't
    cover yet.

## Quick Start

```toml
[dependencies]
vulkane = { version = "0.4", features = ["fetch-spec"] }
```

```rust
use vulkane::safe::{
    ApiVersion, Buffer, BufferCreateInfo, BufferUsage, CommandPool,
    DeviceCreateInfo, DeviceMemory, Fence, Instance, InstanceCreateInfo,
    MemoryPropertyFlags, PipelineStage, AccessFlags, QueueCreateInfo,
    QueueFlags,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let instance = Instance::new(InstanceCreateInfo {
        application_name: Some("hello-vulkane"),
        api_version: ApiVersion::V1_0,
        ..Default::default()
    })?;

    let physical = instance
        .enumerate_physical_devices()?
        .into_iter()
        .find(|pd| pd.find_queue_family(QueueFlags::TRANSFER).is_some())
        .ok_or("no compatible GPU")?;
    let qf = physical.find_queue_family(QueueFlags::TRANSFER).unwrap();

    let device = physical.create_device(DeviceCreateInfo {
        queue_create_infos: &[QueueCreateInfo::single(qf)],
        ..Default::default()
    })?;
    let queue = device.get_queue(qf, 0);

    // One-call buffer allocation (no manual memory_requirements + find_type + bind).
    let (buffer, mut memory) = Buffer::new_bound(
        &device, &physical,
        BufferCreateInfo { size: 1024, usage: BufferUsage::TRANSFER_DST },
        MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // One-shot command recording + submit + wait.
    queue.one_shot(&device, qf, |rec| {
        rec.fill_buffer(&buffer, 0, 1024, 0xDEADBEEF);
        rec.memory_barrier(
            PipelineStage::TRANSFER, PipelineStage::HOST,
            AccessFlags::TRANSFER_WRITE, AccessFlags::HOST_READ,
        );
        Ok(())
    })?;

    let mapped = memory.map()?;
    assert_eq!(&mapped.as_slice()[..4], &0xDEADBEEFu32.to_ne_bytes());
    println!("GPU filled the buffer with 0xDEADBEEF — it works!");
    Ok(())
}
```

## Bundled Examples

15 examples ship with the crate, from basic compute through advanced
graphics techniques. All are headless (runnable in CI) except
`windowed_triangle`.

| Example | Technique |
| --- | --- |
| [`device_info`](vulkane/examples/device_info.rs) | Raw API: instance, physical device enumeration, queue families |
| [`fill_buffer`](vulkane/examples/fill_buffer.rs) | Safe API: `vkCmdFillBuffer` round trip |
| [`compute_square`](vulkane/examples/compute_square.rs) | Compute: SPIR-V, descriptor set, pipeline, dispatch, verify |
| [`compute_image_invert`](vulkane/examples/compute_image_invert.rs) | Compute: 2D storage image, layout transitions, per-pixel verify |
| [`compile_shader`](vulkane/examples/compile_shader.rs) | Compile GLSL/WGSL → SPIR-V via naga (`--features naga`) |
| [`headless_triangle`](vulkane/examples/headless_triangle.rs) | Graphics: render pass, pipeline, draw, readback |
| [`textured_quad`](vulkane/examples/textured_quad.rs) | Graphics: texture upload, sampler, WGSL fragment shader |
| [`windowed_triangle`](vulkane/examples/windowed_triangle.rs) | Windowed: winit + surface + swapchain + present loop |
| [`buffer_upload`](vulkane/examples/buffer_upload.rs) | `Queue::one_shot` staging upload pattern |
| [`allocator_compute`](vulkane/examples/allocator_compute.rs) | `Allocator::create_buffer` — 2 lines vs 5 |
| [`raw_interop`](vulkane/examples/raw_interop.rs) | `Device::dispatch()` + `.raw()` escape hatch |
| [`depth_prepass`](vulkane/examples/depth_prepass.rs) | Depth-only pass + color EQUAL — early-Z prepass |
| [`instanced_mesh`](vulkane/examples/instanced_mesh.rs) | 100 triangles via `InputRate::INSTANCE` |
| [`shadow_map`](vulkane/examples/shadow_map.rs) | Two-pass shadow mapping: depth bias, comparison sampler, uniform buffers |
| [`deferred_shading`](vulkane/examples/deferred_shading.rs) | G-buffer (3 MRT) + fullscreen Phong lighting pass |
| [`derive_vertex`](vulkane/examples/derive_vertex.rs) | `#[derive(Vertex)]` auto-generated vertex layouts (`--features derive`) |

```bash
cargo run -p vulkane --features fetch-spec --example headless_triangle
cargo run -p vulkane --features fetch-spec --example shadow_map
cargo run -p vulkane --features fetch-spec,derive --example derive_vertex
```

## `#[derive(Vertex)]`

Enable the `derive` feature to auto-generate vertex input layouts:

```toml
vulkane = { version = "0.4", features = ["fetch-spec", "derive"] }
```

```rust
use vulkane::Vertex;

#[derive(Vertex, Clone, Copy)]
#[repr(C)]
struct MyVertex {
    position: [f32; 3],  // R32G32B32_SFLOAT, location 0
    normal:   [f32; 3],  // R32G32B32_SFLOAT, location 1
    uv:       [f32; 2],  // R32G32_SFLOAT,    location 2
}

// In pipeline setup:
let bindings = [MyVertex::binding(0)];
let attributes = MyVertex::attributes(0);
builder.vertex_input(&bindings, &attributes)
```

Strides, offsets, and Vulkan format enums are computed at compile time.
Supports `f32`, `[f32; 2..4]`, `u32`, `[u32; 2..4]`, `i32`,
`[i32; 2..3]`, `[u8; 4]`, `u16`, `i16`. For per-instance data, use
`MyStruct::instance_binding(n)` instead of `::binding(n)`.

## Runtime shader compilation

Vulkane offers two optional, independent back-ends for compiling shader
source to SPIR-V at runtime. Enable one, the other, or both — they are
completely separate modules.

### `naga` — pure Rust (WGSL + subset of GLSL)

Zero external build dependencies. Best choice when:

- You want a pure-Rust build (no CMake, no C++ toolchain).
- You are targeting modern GLSL or WGSL.
- WGSL's combined image-samplers fit your rendering code.

```toml
vulkane = { version = "0.4", features = ["naga"] }
```

```rust
use vulkane::safe::naga::compile_glsl;
use naga::ShaderStage;

let spirv = compile_glsl(
    r#"#version 450
       layout(local_size_x = 64) in;
       layout(set = 0, binding = 0, std430) buffer Data { uint v[]; };
       void main() { v[gl_GlobalInvocationID.x] *= 2; }"#,
    ShaderStage::Compute,
)?;
// Pass straight into ShaderModule::from_spirv(&device, &spirv).
```

### `shaderc` — Khronos glslang (full GLSL + HLSL)

Wraps the Khronos reference compiler. Best choice when:

- You need **full GLSL** support (`#include`, `GL_*` extensions, older
  core versions, legacy sample shaders).
- You are compiling **HLSL** for a Vulkan target.
- You want optimization passes (`OptimizationLevel::Size` or
  `Performance`) or explicit macro defines / include resolution.

```toml
vulkane = { version = "0.4", features = ["shaderc"] }
```

```rust
use vulkane::safe::shaderc::{compile_glsl, ShaderKind};

let spirv = compile_glsl(
    r#"#version 450
       layout(local_size_x = 64) in;
       void main() {}"#,
    ShaderKind::Compute,
    "doubler.comp",   // virtual file name (used in error messages and #include resolution)
    "main",           // entry-point name
)?;
```

For fine-grained control — HLSL input, optimization level, macro
defines, include callbacks, target Vulkan version — use
`compile_with_options`:

```rust
use vulkane::safe::shaderc::{compile_with_options, ShaderKind, SourceLanguage};
use shaderc::OptimizationLevel;

let spirv = compile_with_options(
    hlsl_source,
    ShaderKind::Fragment,
    "shader.hlsl",
    "main",
    |opts| {
        opts.set_source_language(SourceLanguage::HLSL);
        opts.set_optimization_level(OptimizationLevel::Size);
        opts.add_macro_definition("USE_PBR", Some("1"));
    },
)?;
```

#### Build requirements for `shaderc`

`shaderc-rs` locates libshaderc in this order:

1. **`SHADERC_LIB_DIR`** env var — point to a directory containing
   `libshaderc_combined`.
2. **`VULKAN_SDK`** env var — installing the
   [LunarG Vulkan SDK](https://vulkan.lunarg.com/) sets this
   automatically, and ships a prebuilt `libshaderc_combined`. **This is
   the easiest path.**
3. **`pkg-config`** / system libraries (typical on Linux distros that
   package `shaderc`).
4. **Source build fallback** — compiles glslang from C++ source.
   Requires CMake ≥ 3.17 (newer CMake may need
   `CMAKE_POLICY_VERSION_MINIMUM=3.5` until shaderc-sys updates its
   pinned sources), Python 3, and a working C++ toolchain. First build
   takes 1–3 minutes.

If neither the SDK nor a system package is available and you don't
want to build from source, use the `naga` feature instead.

#### When to pick which

| Need                                   | `naga` | `shaderc` |
| -------------------------------------- | :----: | :-------: |
| Pure Rust build (no C++ toolchain)     |   ✅   |    ❌     |
| Modern GLSL (core, no extensions)      |   ✅   |    ✅     |
| Full GLSL (`#include`, `GL_*` exts)    |   ❌   |    ✅     |
| HLSL input                             |   ❌   |    ✅     |
| WGSL input                             |   ✅   |    ❌     |
| Optimization passes                    |   ❌   |    ✅     |
| Smallest binary / fastest cold build   |   ✅   |    ❌     |

Both features can be enabled simultaneously — pick per shader at the
call site.

## Why Vulkane over ash?

| Aspect | Vulkane | ash |
| --- | --- | --- |
| Source of truth | `vk.xml` at build time | Hand-curated bindings |
| New Vulkan version | Swap vk.xml, rebuild | Wait for crate update |
| Safe wrapper | Built-in: compute + graphics + allocator + sync | Raw FFI only |
| Sub-allocator | TLSF + linear pools + defrag built in | BYO (`gpu-allocator`) |
| Vertex layout | `#[derive(Vertex)]` | Manual |
| Pipeline builder | Depth bias, CompareOp, multi-attach, dynamic viewport | N/A (raw structs) |
| Allocation helpers | `Buffer::new_bound`, `Queue::upload_buffer<T>` | N/A |
| GLSL/WGSL→SPIR-V | Optional `naga` (pure Rust) or `shaderc` (glslang) feature | N/A |
| Raw escape hatch | `device.dispatch()` + `.raw()` | N/A (always raw) |
| Maturity | New (0.4) | Battle-tested |

## Features

| Feature | Description |
| --- | --- |
| `build-support` (default) | XML parsing and code generation during build |
| `fetch-spec` | Auto-download vk.xml from Khronos GitHub |
| `naga` | `compile_glsl` + `compile_wgsl` → SPIR-V at runtime (pure Rust) |
| `shaderc` | `compile_glsl` / `compile_with_options` → SPIR-V via Khronos glslang (full GLSL + HLSL) |
| `derive` | `#[derive(Vertex)]` proc macro for vertex layouts |

## Providing vk.xml

1. **`VK_XML_PATH`** env var — point to any local `vk.xml`
2. **Local copy** at `spec/registry/Vulkan-Docs/xml/vk.xml`
3. **Auto-download** (`--features fetch-spec`), optionally pinned
   with `VK_VERSION=1.3.250`

## Supported Vulkan Versions

**1.2.175** through the latest release. The minimum is set by the
`VK_MAKE_API_VERSION` macros introduced in that version.

## Architecture

```text
vk.xml → vulkan_gen (roxmltree parser + codegen) → vulkane crate
                                                    ├── raw/   (generated FFI bindings)
                                                    ├── safe/  (RAII wrapper)
                                                    └── vulkane_derive (proc macros)
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in vulkane by you, as defined in the Apache-2.0
license, shall be dual-licensed as above, without any additional terms
or conditions.
