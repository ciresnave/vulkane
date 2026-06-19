//! Live-device exercises of the Phase-3 generated ergonomic safe
//! traits (`DeviceSafeExt`, `InstanceSafeExt`, `PhysicalDeviceSafeExt`,
//! `QueueSafeExt`). The earlier `safe::auto::tests` module proves the
//! trait methods *exist* at compile time; this file proves they
//! *actually work* against a real driver.
//!
//! All tests skip gracefully when Vulkan is unavailable. They do not
//! probe for any specific extension — they only touch core-1.0 /
//! core-1.1 functionality that every Vulkan loader exposes.

use vulkane::safe::{
    ApiVersion, DeviceCreateInfo, DeviceSafeExt, Instance, InstanceCreateInfo, InstanceSafeExt,
    PhysicalDeviceSafeExt, QueueCreateInfo, QueueFlags, QueueSafeExt,
};

fn bootstrap() -> Option<(Instance, vulkane::safe::Device, u32)> {
    let instance = Instance::new(InstanceCreateInfo {
        application_name: Some("vulkane generator-live-device test"),
        api_version: ApiVersion::V1_0,
        ..Default::default()
    })
    .ok()?;
    // NB: call the generated trait method via UFCS — the hand-written
    // inherent `Instance::enumerate_physical_devices` shadows it by
    // returning the safe `Vec<PhysicalDevice>` wrapper. We want the
    // raw-handle version for this test.
    let raw_phys = <Instance as InstanceSafeExt>::enumerate_physical_devices(&instance).ok()?;
    if raw_phys.is_empty() {
        return None;
    }
    // Still need a safe PhysicalDevice to create a device; get it via
    // the hand-written enumerate.
    let physical = instance
        .enumerate_physical_devices()
        .ok()?
        .into_iter()
        .find(|pd| pd.find_queue_family(QueueFlags::COMPUTE).is_some())?;
    let qf = physical.find_queue_family(QueueFlags::COMPUTE)?;
    let device = physical
        .create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo::single(qf)],
            ..Default::default()
        })
        .ok()?;
    Some((instance, device, qf))
}

#[test]
fn generated_instance_enumerate_physical_devices_live() {
    // Proof the generator's count-then-fill enumerate pattern produces
    // a working two-call sequence against a real loader.
    let Ok(instance) = Instance::new(InstanceCreateInfo::default()) else {
        eprintln!("SKIP: Vulkan not available");
        return;
    };
    let raw_physdevs = <Instance as InstanceSafeExt>::enumerate_physical_devices(&instance)
        .expect("InstanceSafeExt::enumerate_physical_devices");
    let safe_physdevs = instance
        .enumerate_physical_devices()
        .expect("Instance::enumerate_physical_devices (hand-written)");
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
    let Some((_instance, device, _qf)) = bootstrap() else {
        return;
    };
    <vulkane::safe::Device as DeviceSafeExt>::device_wait_idle(&device)
        .expect("device_wait_idle must succeed on an idle device");
}

#[test]
fn generated_queue_wait_idle_live() {
    // Queue-dispatch via the generated QueueSafeExt.
    let Some((_instance, device, qf)) = bootstrap() else {
        return;
    };
    let queue = device.get_queue(qf, 0);
    <vulkane::safe::Queue as QueueSafeExt>::queue_wait_idle(&queue)
        .expect("queue_wait_idle must succeed on an idle queue");
}

#[test]
fn generated_physical_device_get_queue_family_properties_live() {
    // Generated void-return enumerate: should match the hand-written
    // enumerate that the safe wrapper surfaces.
    let Some((_instance, device, _qf)) = bootstrap() else {
        return;
    };
    let _ = device;
    let instance = Instance::new(InstanceCreateInfo::default()).unwrap();
    for pd in instance.enumerate_physical_devices().unwrap_or_default() {
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
    let instance = match Instance::new(InstanceCreateInfo::default()) {
        Ok(i) => i,
        Err(_) => return,
    };
    for pd in instance.enumerate_physical_devices().unwrap_or_default() {
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
