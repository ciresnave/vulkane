//! Live-device exercises of the Phase-3 generated ergonomic safe
//! traits (`DeviceSafeExt`, `InstanceSafeExt`, `PhysicalDeviceSafeExt`,
//! `QueueSafeExt`). The earlier `safe::auto::tests` module proves the
//! trait methods *exist* at compile time; this file proves they
//! *actually work* against a real driver.
//!
//! All tests skip gracefully when Vulkan is unavailable — and say so, through
//! [`common`], so that under `VULKANE_REQUIRE_DEVICE` a skip is a failure.
//! They do not probe for any specific extension — they only touch core-1.0 /
//! core-1.1 functionality that every Vulkan loader exposes, so *every*
//! precondition in this file is device-absence and every skip here is
//! legitimately fatal in an environment that promised a device.

mod common;

use vulkane::safe::{
    ApiVersion, DeviceSafeExt, Instance, InstanceSafeExt, PhysicalDeviceSafeExt, QueueSafeExt,
};

/// The previous version also returned the `Instance`, and every caller bound it
/// as `_instance` — it was kept alive for nothing, since `Device` holds its own
/// `Arc` to the instance internals.
fn bootstrap() -> Result<(vulkane::safe::Device, u32), common::Missing> {
    let (device, _physical, qf) = common::compute_device("vulkane generator-live-device test")?;
    Ok((device, qf))
}

#[test]
fn generated_instance_enumerate_physical_devices_live() {
    // Proof the generator's count-then-fill enumerate pattern produces
    // a working two-call sequence against a real loader.
    // `instance_and_devices` enumerates via the HAND-WRITTEN path, which is
    // precisely the side this test compares the generated one against -- so
    // routing through the guard does not change what is being compared.
    let (instance, safe_physdevs) =
        match common::instance_and_devices("vulkane-generator-live", ApiVersion::V1_0) {
            Ok(v) => v,
            Err(cause) => return common::skip(cause),
        };
    let raw_physdevs = <Instance as InstanceSafeExt>::enumerate_physical_devices(&instance)
        .expect("InstanceSafeExt::enumerate_physical_devices");
    // Both paths should report the same number of physical devices.
    assert_eq!(
        raw_physdevs.len(),
        safe_physdevs.len(),
        "generated InstanceSafeExt::enumerate_physical_devices must match hand-written count"
    );
}

#[test]
fn generated_device_wait_idle_live() {
    // Simplest VkResult-returning Device method — proves the generated
    // body's `Result<()>` translation works against a real driver.
    let (device, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
    };
    <vulkane::safe::Device as DeviceSafeExt>::device_wait_idle(&device)
        .expect("device_wait_idle must succeed on an idle device");
}

#[test]
fn generated_queue_wait_idle_live() {
    // Queue-dispatch via the generated QueueSafeExt.
    let (device, qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
    };
    let queue = device.get_queue(qf, 0);
    <vulkane::safe::Queue as QueueSafeExt>::queue_wait_idle(&queue)
        .expect("queue_wait_idle must succeed on an idle queue");
}

#[test]
fn generated_physical_device_get_queue_family_properties_live() {
    // Generated void-return enumerate: should match the hand-written
    // enumerate that the safe wrapper surfaces.
    let (device, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
    };
    let _ = device;
    // Through the guard. The helper keeps "no ICD" and "enumeration failed"
    // distinct, so neither collapses into an empty Vec that would run the loop
    // below zero times.
    let (_instance, devices) =
        match common::instance_and_devices("vulkane-generator-live", ApiVersion::V1_0) {
            Ok(v) => v,
            Err(cause) => return common::skip(cause),
        };
    if devices.is_empty() {
        return common::skipped("an ICD is present but reports no physical devices");
    }
    for pd in devices {
        let generated = <vulkane::safe::PhysicalDevice as PhysicalDeviceSafeExt>::get_physical_device_queue_family_properties(&pd);
        let handwritten = pd.queue_family_properties();
        assert_eq!(
            generated.len(),
            handwritten.len(),
            "PhysicalDeviceSafeExt::get_physical_device_queue_family_properties must match safe wrapper count"
        );
    }
}

#[test]
fn generated_physical_device_get_properties_live() {
    // Generated single-output pattern: driver fills
    // VkPhysicalDeviceProperties, we return it.
    // Through the guard; both failure causes stay distinct in the helper.
    let (_instance, devices) =
        match common::instance_and_devices("vulkane-generator-live", ApiVersion::V1_0) {
            Ok(v) => v,
            Err(cause) => return common::skip(cause),
        };
    if devices.is_empty() {
        return common::skipped("an ICD is present but reports no physical devices");
    }
    for pd in devices {
        let generated = <vulkane::safe::PhysicalDevice as PhysicalDeviceSafeExt>::get_physical_device_properties(&pd);
        // Generated struct carries the C layout — `deviceName` is
        // [c_char; 256] and non-empty for any real adapter.
        let first_byte = generated.deviceName[0];
        assert_ne!(
            first_byte, 0,
            "deviceName should be populated by the driver"
        );
    }
}
