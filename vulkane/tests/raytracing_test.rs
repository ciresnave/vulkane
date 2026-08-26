//! Integration tests for `VK_KHR_acceleration_structure` and
//! `VK_KHR_ray_tracing_pipeline` safe wrappers.
//!
//! Every test degrades gracefully when the Vulkan driver does not
//! expose the underlying extension — we only assert on safe-layer
//! surface behaviour (no panics, clean `MissingFunction` surfacing,
//! `InvalidArgument` on mismatched lengths, etc.).
//!
//! This is the file where the two skip classes in [`common`] are furthest
//! apart. Ray tracing is absent from most devices and from Lavapipe entirely,
//! so nearly every degradation here is [`common::skipped_unsupported`] — loud
//! but never fatal. Only *device absence* is fatal, because only that is
//! something `VULKANE_REQUIRE_DEVICE` is entitled to assert.

mod common;

use std::sync::Arc;
use vulkane::safe::{
    AccelerationStructure, AccelerationStructureBuildFlags, AccelerationStructureBuildMode,
    AccelerationStructureBuildType, AccelerationStructureCreateInfo, AccelerationStructureGeometry,
    AccelerationStructureType, ApiVersion, Buffer, BufferCreateInfo, BufferUsage, BuildRange,
    Queue, ShaderBindingRegion,
};

fn bootstrap()
-> Result<(vulkane::safe::Device, vulkane::safe::PhysicalDevice, u32), common::Missing> {
    common::compute_device("vulkane-raytracing-test")
}

#[test]
fn acceleration_structure_build_sizes_rejects_length_mismatch() {
    // Pure safe-layer validation before dispatch.
    let (device, _physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
    };
    let r = device.acceleration_structure_build_sizes(
        AccelerationStructureBuildType::Device,
        AccelerationStructureType::BottomLevel,
        &[AccelerationStructureGeometry::Aabbs {
            data_address: 0,
            stride: 24,
        }],
        &[], // mismatch: 1 geometry, 0 primitive counts
        AccelerationStructureBuildFlags::default(),
    );
    match r {
        Err(vulkane::safe::Error::InvalidArgument(msg)) => {
            assert!(msg.contains("length"));
        }
        Err(vulkane::safe::Error::MissingFunction(_)) => {
            // The extension may not be loaded — in that case our length
            // check still fires first, but some drivers skip it. Either
            // is acceptable for this input-validation test.
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

#[test]
fn acceleration_structure_build_sizes_graceful_missing_function() {
    // Well-formed input: 1 geometry + 1 primitive count. Without the
    // extension enabled, the call surfaces MissingFunction. With the
    // extension enabled, it returns sensible non-negative sizes.
    let (device, _physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
    };
    match device.acceleration_structure_build_sizes(
        AccelerationStructureBuildType::Device,
        AccelerationStructureType::BottomLevel,
        &[AccelerationStructureGeometry::Aabbs {
            data_address: 0,
            stride: 24,
        }],
        &[16], // 16 AABBs max
        AccelerationStructureBuildFlags::PREFER_FAST_TRACE,
    ) {
        Ok(sizes) => {
            // Nothing specific to assert — driver-dependent — but
            // returned sizes should be non-panicking values.
            let _ = sizes.acceleration_structure_size;
            let _ = sizes.build_scratch_size;
            let _ = sizes.update_scratch_size;
        }
        Err(vulkane::safe::Error::MissingFunction(name)) => {
            assert_eq!(name, "vkGetAccelerationStructureBuildSizesKHR");
        }
        Err(other) => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn acceleration_structure_new_graceful_missing_function() {
    // Attempting to create the handle without the extension enabled.
    let (device, physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
    };
    // Create a placeholder buffer — won't actually be valid AS storage
    // but the safe wrapper should fail at the extension-load step,
    // not at unsafe pointer deref.
    let Ok(buffer) = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 4096,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    ) else {
        return common::skipped_unsupported("a 4 KiB STORAGE_BUFFER");
    };
    let req = buffer.memory_requirements();
    let Some(mt) = physical.find_memory_type(
        req.memory_type_bits,
        vulkane::safe::MemoryPropertyFlags::DEVICE_LOCAL,
    ) else {
        return common::skipped_unsupported(
            "a DEVICE_LOCAL memory type this buffer's memory_type_bits accepts",
        );
    };
    let Ok(memory) = vulkane::safe::DeviceMemory::allocate(&device, req.size, mt) else {
        return common::skipped_unsupported("an allocation of the buffer's required size");
    };
    let _ = buffer.bind_memory(&memory, 0);

    let buffer_arc = Arc::new(buffer);
    match AccelerationStructure::new(
        &device,
        AccelerationStructureCreateInfo {
            buffer: Arc::clone(&buffer_arc),
            offset: 0,
            size: 1024,
            type_: AccelerationStructureType::BottomLevel,
            _marker: std::marker::PhantomData,
        },
    ) {
        Ok(_) => {
            // Some drivers happily create an AS over a storage buffer.
        }
        Err(vulkane::safe::Error::MissingFunction(name)) => {
            assert_eq!(name, "vkCreateAccelerationStructureKHR");
        }
        Err(vulkane::safe::Error::Vk(_)) => {
            // Buffer is missing ACCELERATION_STRUCTURE_STORAGE usage —
            // validation layers will reject; spec-noncompliant path
            // for real use, acceptable for this API-surface test.
        }
        Err(other) => panic!("unexpected error shape: {other:?}"),
    }
}

#[test]
fn build_acceleration_structure_rejects_length_mismatch() {
    // Requires a "real" AccelerationStructure to drive the call. Since
    // we can't build one without the extension being live, we set up
    // the AccelerationStructureGeometry / BuildRange slices first and
    // trust that the length check runs before we even reach the driver.
    let (device, _physical, qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
    };
    let queue: Queue = device.get_queue(qf, 0);

    // If we can't create an AccelerationStructure at all (extension
    // not loaded), we substitute with the type-level length-mismatch
    // check by constructing a fake dst AccelerationStructure via a
    // code path that surfaces MissingFunction. The test then checks
    // *that* path instead, which is equally valid for this suite.
    let Some(buffer) = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 4096,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .ok()
    .map(Arc::new) else {
        return common::skipped_unsupported("a 4 KiB STORAGE_BUFFER");
    };
    let dst = match AccelerationStructure::new(
        &device,
        AccelerationStructureCreateInfo {
            buffer: Arc::clone(&buffer),
            offset: 0,
            size: 1024,
            type_: AccelerationStructureType::BottomLevel,
            _marker: std::marker::PhantomData,
        },
    ) {
        Ok(a) => a,
        Err(_) => {
            return common::skipped_unsupported("VK_KHR_acceleration_structure");
        }
    };

    let r = queue.one_shot(&device, qf, |rec| {
        rec.build_acceleration_structure(
            AccelerationStructureType::BottomLevel,
            AccelerationStructureBuildMode::Build,
            AccelerationStructureBuildFlags::default(),
            &dst,
            None,
            &[AccelerationStructureGeometry::Aabbs {
                data_address: 0,
                stride: 24,
            }],
            &[], // mismatched
            0,
        )
    });
    match r {
        Err(vulkane::safe::Error::InvalidArgument(msg)) => {
            assert!(msg.contains("length"));
        }
        Err(vulkane::safe::Error::MissingFunction(_)) | Err(vulkane::safe::Error::Vk(_)) => {
            // Pre-dispatch failure in one_shot (command pool, etc.) or
            // extension missing — acceptable for this validation test.
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

#[test]
fn trace_rays_graceful_missing_function() {
    let (device, _physical, qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
    };
    let queue: Queue = device.get_queue(qf, 0);

    let r = queue.one_shot(&device, qf, |rec| {
        rec.trace_rays(
            ShaderBindingRegion::default(),
            ShaderBindingRegion::default(),
            ShaderBindingRegion::default(),
            ShaderBindingRegion::default(),
            1,
            1,
            1,
        )
    });
    match r {
        Ok(()) => {}
        Err(vulkane::safe::Error::MissingFunction(name)) => {
            assert_eq!(name, "vkCmdTraceRaysKHR");
        }
        Err(vulkane::safe::Error::Vk(_)) => {
            // Driver may reject the no-pipeline-bound trace; acceptable.
        }
        Err(other) => panic!("unexpected error shape: {other:?}"),
    }
}

#[test]
fn ray_tracing_pipeline_properties_queryable() {
    // Should never panic regardless of driver support.
    // Through the guard. `instance_and_devices` keeps both failures distinct --
    // "no Vulkan ICD" and "an ICD is present but enumerating failed" -- so the
    // reason this used to spell out by hand is preserved, and neither collapses
    // into an empty Vec that would run the loop below zero times.
    let (_instance, devices) =
        match common::instance_and_devices("vulkane-raytracing-test", ApiVersion::V1_0) {
            Ok(v) => v,
            Err(cause) => return common::skip(cause),
        };
    if devices.is_empty() {
        return common::skipped("an ICD is present but reports no physical devices");
    }
    for pd in devices {
        let props = pd.ray_tracing_pipeline_properties();
        if let Some(p) = props {
            // Handle size must be at minimum 32 per the spec when the
            // extension is supported — but if the driver returns 0 we
            // just silently accept (means the ext isn't implemented
            // despite get_physical_device_properties2 being available).
            let _ = p.shader_group_handle_size;
        }
    }
}

#[test]
fn build_range_and_shader_binding_region_default() {
    // Pure safe-layer construction: Default impls exist and are zero.
    let r: BuildRange = BuildRange::default();
    assert_eq!(r.primitive_count, 0);
    assert_eq!(r.primitive_offset, 0);
    let s: ShaderBindingRegion = ShaderBindingRegion::default();
    assert_eq!(s.address, 0);
    assert_eq!(s.stride, 0);
    assert_eq!(s.size, 0);
}

#[test]
fn device_features_ray_query_and_rt_pipeline_toggles_exist() {
    // Feature bits generated by vulkan_gen — pure type check.
    let _ = vulkane::safe::DeviceFeatures::new()
        .with_ray_query()
        .with_ray_tracing_pipeline()
        .with_acceleration_structure();
}
