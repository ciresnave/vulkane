// SPDX-License-Identifier: MIT OR Apache-2.0
//! Demonstrates mixing safe wrapper types with raw Vulkan API calls.
//!
//! Creates an instance and device using the safe RAII wrapper, then
//! drops to the raw dispatch table to call `vkGetPhysicalDeviceProperties`
//! directly — proving the escape hatch works seamlessly.
//!
//! This pattern is useful when the safe wrapper doesn't cover a
//! particular Vulkan function yet: create and manage handles via the
//! safe layer (which handles lifetimes, error wrapping, and cleanup),
//! then call the raw function through `device.dispatch()` /
//! `instance.dispatch()` + `.raw()` handles.
//!
//! Run with: `cargo run -p vulkane --features fetch-spec --example raw_interop`

use vulkane::raw::bindings::*;
use vulkane::safe::{
    ApiVersion, DeviceCreateInfo, Instance, InstanceCreateInfo, QueueCreateInfo, QueueFlags,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Safe-wrapper initialization — standard boilerplate.
    let instance = Instance::new(InstanceCreateInfo {
        application_name: Some("vulkane raw_interop"),
        api_version: ApiVersion::V1_0,
        ..Default::default()
    })?;
    let physical = instance
        .enumerate_physical_devices()?
        .into_iter()
        .next()
        .ok_or("no Vulkan device")?;

    let qf = physical
        .find_queue_family(QueueFlags::TRANSFER)
        .ok_or("no transfer queue family")?;
    let device = physical.create_device(DeviceCreateInfo {
        queue_create_infos: &[QueueCreateInfo::single(qf)],
        ..Default::default()
    })?;

    println!("[OK] Created instance + device via the safe wrapper");

    // 2. Escape hatch: call a raw Vulkan function through the
    //    instance dispatch table.
    //
    //    instance.dispatch() returns &VkInstanceDispatchTable — every
    //    field is an Option<fn_ptr>. We grab the physical-device
    //    handle via the safe wrapper's .raw() accessor and call the
    //    function pointer directly.

    let get_props = instance
        .dispatch()
        .vkGetPhysicalDeviceProperties
        .ok_or("vkGetPhysicalDeviceProperties not loaded")?;

    let mut raw_props: VkPhysicalDeviceProperties = unsafe { std::mem::zeroed() };
    // Safety: physical.raw() is a valid VkPhysicalDevice, raw_props
    // is a valid output parameter.
    unsafe { get_props(physical.raw(), &mut raw_props) };

    // Read the device name from the raw struct.
    let name_bytes = &raw_props.deviceName;
    let name_len = name_bytes
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(name_bytes.len());
    let name_vec: Vec<u8> = name_bytes[..name_len].iter().map(|&b| b as u8).collect();
    let name = std::str::from_utf8(&name_vec).unwrap_or("<non-UTF8>");
    println!("[OK] Raw vkGetPhysicalDeviceProperties returned: {name}");

    // Verify it matches what the safe wrapper returns.
    let safe_name = physical.properties().device_name().to_string();
    assert_eq!(name, &safe_name, "raw and safe device names should match");
    println!("[OK] Raw name matches safe wrapper: {safe_name}");

    // 3. Escape hatch on the device side: call vkGetDeviceQueue
    //    directly through the device dispatch table.
    let get_queue = device
        .dispatch()
        .vkGetDeviceQueue
        .ok_or("vkGetDeviceQueue not loaded")?;

    let mut raw_queue: VkQueue = std::ptr::null_mut();
    // Safety: device.raw() is valid; queue family and index 0 were
    // requested at device creation time.
    unsafe { get_queue(device.raw(), qf, 0, &mut raw_queue) };

    // Compare with the safe wrapper's queue handle.
    let safe_queue = device.get_queue(qf, 0);
    assert_eq!(
        raw_queue,
        safe_queue.raw(),
        "raw and safe queue handles should match"
    );
    println!("[OK] Raw vkGetDeviceQueue handle matches safe Queue::raw()");

    device.wait_idle()?;
    println!("\n=== raw_interop example PASSED ===");
    Ok(())
}
