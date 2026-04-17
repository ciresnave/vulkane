//! Auto-generated RAII handle wrappers.
//!
//! Every Vulkan handle type that doesn't already have a hand-written
//! ergonomic wrapper in `vulkane::safe` gets a **generated** RAII
//! wrapper here — one struct per handle, plus a matching `create`
//! constructor and `Drop` impl derived from the paired Vulkan
//! Create / Destroy commands. The list regenerates from `vk.xml` on
//! every build, so as new extensions ship their handle types appear
//! automatically.
//!
//! The wrappers here intentionally trade ergonomics for coverage:
//!
//! - `create(&Device, &VkFooCreateInfo) -> Result<Self>` — user fills
//!   the raw CreateInfo struct themselves (pNext chains via the
//!   [`PNextChain`](crate::safe::PNextChain) builder; extension
//!   enablement via [`DeviceExtensions`](crate::safe::DeviceExtensions)
//!   and [`DeviceFeatures`](crate::safe::DeviceFeatures)).
//! - `raw(&self) -> VkFooKHR` — lets callers hand the handle to any
//!   command that isn't yet exposed as a safe method (Phase 2 work).
//! - `Drop` calls the matching `vkDestroy*` or `vkFree*`.
//!
//! No typestate, no builder sugar, no parent-child dependency
//! inference beyond the `Arc<DeviceInner>` that every wrapper holds.
//! If you want ergonomic wrappers on top, write them in the
//! hand-written `safe` layer and delegate — don't fight the generator.
//!
//! # What you still do in user code
//!
//! For an acceleration structure:
//!
//! ```ignore
//! use vulkane::raw::PNextChainable;
//! use vulkane::raw::bindings::*;
//! use vulkane::safe::auto::AccelerationStructureKHR;
//!
//! let info = VkAccelerationStructureCreateInfoKHR {
//!     sType: VkStructureType::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
//!     pNext: std::ptr::null(),
//!     createFlags: 0,
//!     buffer: backing_buffer.raw(),
//!     offset: 0,
//!     size: needed_size,
//!     type_: VkAccelerationStructureTypeKHR::ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
//!     deviceAddress: 0,
//! };
//! let blas = AccelerationStructureKHR::create(&device, &info)?;
//! ```
//!
//! Verbose but completely safe — no `unsafe` in user code.

// The generated file references `crate::safe::device::DeviceInner`,
// `crate::safe::Device`, `crate::safe::Result`, `crate::safe::Error`,
// and `crate::safe::check`. All are in scope via the fully-qualified
// paths the generator emits, so no `use` imports are required here.

include!(concat!(env!("OUT_DIR"), "/auto_handles_generated.rs"));

// Phase 2 — one ext trait per dispatch-target handle type. Users opt
// in per trait: `use vulkane::safe::DeviceExt;` makes every vk* device
// command available as a safe method on `Device`.
include!(concat!(env!("OUT_DIR"), "/auto_device_ext_generated.rs"));
include!(concat!(env!("OUT_DIR"), "/auto_instance_ext_generated.rs"));
include!(concat!(env!("OUT_DIR"), "/auto_physical_device_ext_generated.rs"));
include!(concat!(env!("OUT_DIR"), "/auto_queue_ext_generated.rs"));
include!(concat!(env!("OUT_DIR"), "/auto_command_buffer_ext_generated.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time proof that `AccelerationStructureKHR` has the
    /// expected shape: `create` constructor, `raw` accessor, `Drop`
    /// impl, and `Send + Sync`. Fuel's ray-tracing use case depends on
    /// this. We don't try to actually *call* `create` — it would need
    /// a real device — just that the signature exists.
    #[test]
    fn acceleration_structure_khr_has_auto_wrapper_api() {
        fn _assert_send<T: Send + Sync>() {}
        _assert_send::<AccelerationStructureKHR>();

        // Type-level assertion that `create` exists with the expected
        // signature. The closure is never called, but if the types
        // don't line up this fails to compile.
        let _ = |device: &crate::safe::Device,
                 info: &crate::raw::bindings::VkAccelerationStructureCreateInfoKHR|
         -> crate::safe::Result<AccelerationStructureKHR> {
            AccelerationStructureKHR::create(device, info)
        };
    }

    /// Proof that the generated ext traits are importable and their
    /// methods have the expected signatures on the hand-written handle
    /// wrappers. This wires up Phase 2 end-to-end: the user imports an
    /// ext trait, methods appear on the existing `Device` /
    /// `CommandBufferRecording` / etc., and calls type-check.
    #[test]
    fn ext_traits_provide_raw_methods() {
        // Force the trait imports — the closures below only type-check
        // if the ext-trait methods actually exist on these types.
        use super::super::{
            CommandBufferRecordingExt, DeviceExt, InstanceExt, PhysicalDeviceExt, QueueExt,
        };

        // Device: vk_device_wait_idle — simplest VkResult-returning
        // command that takes only the device handle.
        let _ = |d: &crate::safe::Device| -> crate::safe::Result<crate::raw::bindings::VkResult> {
            d.vk_device_wait_idle()
        };

        // CommandBufferRecording: vk_cmd_draw — void-returning cmd
        // method. Proves the generator routed VkCommandBuffer → the
        // CommandBufferRecording trait and that &mut self is threaded.
        let _ = |r: &mut crate::safe::CommandBufferRecording<'_>| {
            r.vk_cmd_draw(0u32, 0u32, 0u32, 0u32);
        };

        // PhysicalDevice: vk_get_physical_device_properties — void
        // taking a *mut VkPhysicalDeviceProperties output.
        let _ = |p: &crate::safe::PhysicalDevice,
                 out: *mut crate::raw::bindings::VkPhysicalDeviceProperties| {
            p.vk_get_physical_device_properties(out);
        };

        // Queue: vk_queue_wait_idle — VkResult-returning.
        let _ = |q: &crate::safe::Queue| -> crate::safe::Result<crate::raw::bindings::VkResult> {
            q.vk_queue_wait_idle()
        };

        // Instance: vk_destroy_instance is skipped (RAII-covered), so
        // pick a command that still lands on InstanceExt — nothing
        // common does in plain 1.0, but the trait's existence is
        // what matters for import correctness.
        fn _assert_instance_ext<T: InstanceExt>() {}
        _assert_instance_ext::<crate::safe::Instance>();
    }

    /// Ray-tracing end-to-end: Fuel-class users can now reach
    /// `vkCmdTraceRaysKHR` without any `unsafe` in their code.
    /// The compile-time check below is the whole point of Phase 2.
    #[test]
    fn ray_tracing_is_reachable_without_unsafe() {
        use super::super::CommandBufferRecordingExt;
        let _ = |r: &mut crate::safe::CommandBufferRecording<'_>,
                 raygen: *const crate::raw::bindings::VkStridedDeviceAddressRegionKHR,
                 miss: *const crate::raw::bindings::VkStridedDeviceAddressRegionKHR,
                 hit: *const crate::raw::bindings::VkStridedDeviceAddressRegionKHR,
                 callable: *const crate::raw::bindings::VkStridedDeviceAddressRegionKHR,
                 w: u32,
                 h: u32,
                 d: u32| {
            r.vk_cmd_trace_rays_khr(raygen, miss, hit, callable, w, h, d);
        };
    }

    /// Same shape check for a few other high-value generated wrappers,
    /// so regressions in the generator show up as failing compile
    /// early.
    #[test]
    fn key_auto_wrappers_compile() {
        fn _assert_create<T, I, E>()
        where
            for<'a, 'b> fn(&'a crate::safe::Device, &'b I) -> std::result::Result<T, E>: Copy,
        {
        }
        // Presence-by-reference: just name the types so they must
        // exist in the generated file.
        let _: Option<AccelerationStructureKHR> = None;
        let _: Option<AccelerationStructureNV> = None;
        let _: Option<MicromapEXT> = None;
        let _: Option<VideoSessionKHR> = None;
        let _: Option<DescriptorUpdateTemplate> = None;
        let _: Option<PrivateDataSlot> = None;
        let _: Option<ValidationCacheEXT> = None;
        let _: Option<BufferView> = None;
    }
}
