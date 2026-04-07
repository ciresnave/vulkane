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

## Why Spock?

If you're already using `ash`, here's the trade-off:

| Aspect                     | spock                            | ash                              |
| -------------------------- | -------------------------------- | -------------------------------- |
| Source of truth            | `vk.xml` (parsed at build time)  | Hand-curated bindings module     |
| New Vulkan version support | Swap vk.xml, rebuild             | Wait for crate update            |
| New extension support      | Automatic on next vk.xml fetch   | Wait for crate update            |
| Generated lines of Rust    | ~52,000                          | ~30,000                          |
| Hand-written FFI           | None                             | Some                             |
| API style                  | Raw FFI + minimal helpers        | Builder patterns + safe wrappers |
| Maturity                   | New                              | Battle-tested                    |

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

## First impression

```rust
use spock::raw::bindings::*;
use spock::raw::{VkResultExt, VulkanLibrary};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load vulkan-1.dll / libvulkan.so.1 / etc.
    let library = VulkanLibrary::new()?;

    // Load global Vulkan functions (vkCreateInstance, etc.)
    let entry = unsafe { library.load_entry() };

    // Create an instance the normal Vulkan way
    let app_info = VkApplicationInfo {
        sType: VkStructureType::STRUCTURE_TYPE_APPLICATION_INFO,
        pNext: std::ptr::null(),
        pApplicationName: c"hello-spock".as_ptr() as *const i8,
        applicationVersion: vk_make_api_version(0, 0, 1, 0),
        pEngineName: std::ptr::null(),
        engineVersion: 0,
        apiVersion: VK_API_VERSION_1_0,
    };
    let create_info = VkInstanceCreateInfo {
        sType: VkStructureType::STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo: &app_info,
        ..Default::default()
    };
    let mut instance: VkInstance = std::ptr::null_mut();
    unsafe { (entry.vkCreateInstance.unwrap())(&create_info, std::ptr::null(), &mut instance) }
        .into_result()?;

    // Load instance-level functions for the new instance
    let inst = unsafe { library.load_instance(instance) };

    // Enumerate physical devices
    let mut count = 0u32;
    unsafe { (inst.vkEnumeratePhysicalDevices.unwrap())(instance, &mut count, std::ptr::null_mut()) }
        .into_result()?;
    println!("Found {} GPU(s)", count);

    // Clean up
    unsafe { (inst.vkDestroyInstance.unwrap())(instance, std::ptr::null()) };
    Ok(())
}
```

For a more complete example that creates a logical device and inspects memory heaps and queue
families, see [`spock/examples/device_info.rs`](spock/examples/device_info.rs).

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

| Vulkan version       | Status |
| -------------------- | ------ |
| 1.2.175 (minimum)    | Tested |
| 1.3.250              | Tested |
| 1.4.348              | Tested |
| latest (main branch) | Tested |

## What Gets Generated

Everything in this table is derived entirely from `vk.xml` at build time:

| Category                  | Count (recent vk.xml) | Description                                                                                  |
| ------------------------- | --------------------- | -------------------------------------------------------------------------------------------- |
| Structs                   | ~1,478                | `#[repr(C)]` with correct pointer/array/const field types                                    |
| Unions                    | ~14                   | `#[repr(C)] pub union` with `unsafe { mem::zeroed() }` defaults                              |
| Type aliases              | ~1,343                | Dispatchable handles (`*mut c_void`), non-dispatchable handles (`u64`), bitmasks, base types |
| Rust enums                | ~148                  | `#[repr(i32)]` with extension values merged from all extensions                              |
| Constants                 | ~3,064                | Including bitmask flag values emitted as `pub const`                                         |
| Function pointer typedefs | ~657                  | `unsafe extern "system" fn(...)` for every Vulkan command                                    |
| Dispatch tables           | 3                     | Entry, instance, and device tables generated from first-parameter classification             |
| Version functions         | All                   | `vk_make_api_version`, `vk_api_version_major`, etc. transpiled from C macros                 |
| Doc comments              | ~960                  | Harvested from vk.xml `<comment>` and `comment` attributes                                   |

## Features

| Feature                   | Description                                                                |
| ------------------------- | -------------------------------------------------------------------------- |
| `build-support` (default) | Enables XML parsing and code generation during build                       |
| `fetch-spec`              | Enables automatic download of `vk.xml` from the Khronos GitHub repository  |

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
   ┌──────────────┐
   │  build.rs    │  (resolves vk.xml, calls vulkan_gen::generate_bindings)
   │      │       │
   │      ▼       │
   │ vulkan_      │
   │ bindings.rs  │  (generated, ~52k lines)
   │      │       │
   │      ▼       │
   │  raw/loader  │  (VulkanLibrary, dispatch tables)
   │  raw/result  │  (VkResultExt, vk_check!)
   └──────────────┘
```

The parser uses DOM-based XML parsing (`roxmltree`) to correctly handle nested elements
like `<member>` and `<param>` that contain mixed text and child element content.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in spock by you, as defined in the Apache-2.0 license, shall be dual-licensed
as above, without any additional terms or conditions.
