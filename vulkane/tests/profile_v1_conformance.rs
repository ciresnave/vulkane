//! Profile v1 conformance — the Kernel-Seam Interop Contract named surface.
//!
//! Vulkane's role in Fuel's kernel-seam contract (Profile v1, ratified
//! 2026-06-20) is **FDX-only, the BDA subset**: it ships no kernels and puts
//! nothing on the seam wire. Its entire obligation is to expose a stable
//! *named surface* that `fuel-vulkan-backend` builds the BDA tensor handoff
//! on:
//!
//! * `AllocatorOptions { buffer_device_address }` + `Allocator::new_with_options`
//!   — opt every memory block into `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` so
//!   pooled buffers can yield a valid GPU address.
//! * `BufferUsage::SHADER_DEVICE_ADDRESS` — the per-buffer usage flag the
//!   caller sets on each buffer-table entry.
//! * `DeviceFeatures::with_buffer_device_address` — enable the device feature
//!   so `vkGetBufferDeviceAddress` is loaded.
//! * `Buffer::device_address` — return the `VkDeviceAddress` that becomes FDX
//!   `data` on `kDLVulkan`.
//!
//! The contract pins Vulkane to this **named surface, not a crate version**
//! (`>= 0.8.2` is only "the first version that exposes it"). Its §7.2
//! therefore states that *a Vulkane major bump triggers a re-check of the
//! named surface*. This test is that re-check, made mechanical: a compile-time
//! lock-in mirroring the `Send + Sync` lock-ins on `Queue` / `CommandBuffer`.
//! If a future change renames, removes, or alters the signature of any item
//! above, this file fails to compile — surfacing the conformance break in
//! Vulkane's own CI instead of in `fuel-vulkan-backend`'s build.
//!
//! Runtime behaviour of this surface is covered separately by
//! `test_allocator_buffer_device_address` in `safe_wrapper_test.rs`, which
//! exercises the full pooled-buffer BDA path against a real driver.

use vulkane::safe::{
    Allocator, AllocatorOptions, Buffer, BufferUsage, Device, DeviceFeatures, PhysicalDevice,
    Result,
};

/// Compile-time lock-in for the Profile v1 (BDA-subset) named surface.
///
/// The body needs no Vulkan device — every statement is a type-checked
/// reference to a named item. Compilation *is* the assertion; the test
/// passing merely confirms the lock-in linked.
#[test]
fn profile_v1_named_surface_locked_in() {
    // (1) The allocator-wide opt-in field — exact name, read as `bool` from a
    //     default value so an *additive* field elsewhere doesn't trip this.
    let _opt_in: bool = AllocatorOptions::default().buffer_device_address;

    // (2) The constructor that consumes it — exact signature.
    let _new_with_options: fn(&Device, &PhysicalDevice, AllocatorOptions) -> Result<Allocator> =
        Allocator::new_with_options;

    // (3) The per-buffer usage flag the caller stamps on each buffer-table entry.
    let _usage: BufferUsage = BufferUsage::SHADER_DEVICE_ADDRESS;

    // (4) The device-feature enabler (builder form; receiver type left free so
    //     a by-value/by-ref change in the generated builder doesn't matter here).
    let _ = DeviceFeatures::default().with_buffer_device_address();

    // (5) The address accessor — exact signature; its `u64` becomes FDX `data`.
    let _device_address: fn(&Buffer) -> Result<u64> = Buffer::device_address;
}
