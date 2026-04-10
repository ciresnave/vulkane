//! # Vulkane — Vulkan API Bindings + Safe Wrapper for Rust
//!
//! Vulkane generates complete Vulkan API bindings from the official
//! `vk.xml` specification, plus a safe RAII wrapper covering compute
//! and graphics end-to-end — from instance creation through shadow
//! mapping and deferred shading.
//!
//! # Getting Started
//!
//! Add to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! vulkane = { version = "0.4", features = ["fetch-spec"] }
//! ```
//!
//! ## Step 1: Create an instance and find a GPU
//!
//! ```rust,no_run
//! use vulkane::safe::{Instance, InstanceCreateInfo, ApiVersion, QueueFlags};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let instance = Instance::new(InstanceCreateInfo {
//!     application_name: Some("my-app"),
//!     api_version: ApiVersion::V1_0,
//!     ..Default::default()
//! })?;
//!
//! let physical = instance
//!     .enumerate_physical_devices()?
//!     .into_iter()
//!     .find(|pd| pd.find_queue_family(QueueFlags::GRAPHICS).is_some())
//!     .ok_or("no compatible GPU")?;
//!
//! println!("Using GPU: {}", physical.properties().device_name());
//! # Ok(()) }
//! ```
//!
//! ## Step 2: Create a device and queue
//!
//! ```rust,ignore
//! use vulkane::safe::{DeviceCreateInfo, QueueCreateInfo};
//!
//! let qf = physical.find_queue_family(QueueFlags::GRAPHICS).unwrap();
//! let device = physical.create_device(DeviceCreateInfo {
//!     queue_create_infos: &[QueueCreateInfo::single(qf)],
//!     ..Default::default()
//! })?;
//! let queue = device.get_queue(qf, 0);
//! ```
//!
//! ## Step 3: Allocate a buffer (one-call)
//!
//! ```rust,ignore
//! use vulkane::safe::{Buffer, BufferCreateInfo, BufferUsage, MemoryPropertyFlags};
//!
//! let (buffer, memory) = Buffer::new_bound(
//!     &device, &physical,
//!     BufferCreateInfo { size: 4096, usage: BufferUsage::STORAGE_BUFFER },
//!     MemoryPropertyFlags::DEVICE_LOCAL,
//! )?;
//! ```
//!
//! ## Step 4: Record and submit GPU work
//!
//! ```rust,ignore
//! use vulkane::safe::{PipelineStage, AccessFlags};
//!
//! queue.one_shot(&device, qf, |rec| {
//!     rec.fill_buffer(&buffer, 0, 4096, 0xDEADBEEF);
//!     rec.memory_barrier(
//!         PipelineStage::TRANSFER, PipelineStage::HOST,
//!         AccessFlags::TRANSFER_WRITE, AccessFlags::HOST_READ,
//!     );
//!     Ok(())
//! })?;
//! ```
//!
//! # Two API Surfaces
//!
//! - **[`raw`]** — direct FFI bindings, exactly as the spec defines
//!   them. Maximum control, zero overhead. Every Vulkan type, struct,
//!   enum, and function pointer is available.
//!
//! - **[`safe`]** — RAII wrappers with automatic cleanup, typed flags,
//!   and convenience helpers. Covers compute + graphics: instance,
//!   device, buffer, image, sampler, render pass, framebuffer,
//!   graphics + compute pipelines, swapchain, sub-allocator, sync
//!   primitives, query pools, and more.
//!
//! The two layers interoperate seamlessly. Call `.raw()` on any safe
//! type to get the underlying Vulkan handle, and
//! [`Device::dispatch()`](safe::Device::dispatch) /
//! [`Instance::dispatch()`](safe::Instance::dispatch) to access the
//! full dispatch tables for Vulkan functions the safe wrapper doesn't
//! cover yet.
//!
//! # Cargo Features
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `build-support` (default) | XML parsing and code generation during build |
//! | `fetch-spec` | Auto-download vk.xml from the Khronos GitHub repository |
//! | `naga` | `compile_glsl` + `compile_wgsl` for runtime GLSL/WGSL → SPIR-V |
//! | `derive` | `#[derive(Vertex)]` for automatic vertex input layout generation |
//!
//! # Providing vk.xml
//!
//! 1. **`VK_XML_PATH`** env var — point to any local vk.xml file
//! 2. **Local copy** at `spec/registry/Vulkan-Docs/xml/vk.xml`
//! 3. **Auto-download** with `--features fetch-spec` (optionally pin
//!    with `VK_VERSION=1.3.250`)
//!
//! # Supported Vulkan Versions
//!
//! **1.2.175** through the latest release. The minimum is set by the
//! `VK_MAKE_API_VERSION` macros introduced in that version.

// Re-export all raw bindings
pub mod raw;

// Safe RAII wrapper module
pub mod safe;

// Re-export commonly used items at crate root for convenience
pub use raw::bindings::*;

// Re-export the derive macro when the `derive` feature is enabled,
// so users can write `use vulkane::Vertex;` or `#[derive(vulkane::Vertex)]`.
#[cfg(feature = "derive")]
pub use vulkane_derive::Vertex;

/// Version information for these bindings
pub mod version {
    /// The version of these bindings
    pub const BINDINGS_VERSION: &str = env!("CARGO_PKG_VERSION");

    /// Build timestamp (set during build)
    pub const BUILD_TIMESTAMP: &str = env!("BUILD_TIMESTAMP");
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod unit_tests;
