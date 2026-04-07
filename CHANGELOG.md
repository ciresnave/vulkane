# Changelog

All notable changes to spock will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial public release of spock — raw Vulkan API bindings generated entirely from the official `vk.xml` specification.
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
- No safe wrapper layer. Spock provides raw FFI bindings; building a safe API
  on top is intentionally left as a separate concern.
- Cross-platform CI runs build and unit tests on Linux/macOS/Windows, but the
  real-Vulkan integration test only runs on developer machines with a Vulkan
  driver installed (it skips gracefully elsewhere).
