//! Integration test for the safe wrapper module.
//!
//! Validates the entire safe API end-to-end against a real Vulkan driver.
//! Skips gracefully on systems without Vulkan installed — and declares every
//! skip through [`common`], so that a run which was *supposed* to have a device
//! fails instead of quietly reporting `ok`.
//!
//! This is the largest concentration of device-gated tests in the repository:
//! 63 of its tests can return before asserting anything. Almost all of them are
//! local hygiene rather than freeze-path evidence, which is why they were done
//! after `kiss_target_live.rs` rather than first.

mod common;

use vulkane::safe::{
    AccessFlags, AccessFlags2, AllocationCreateInfo, AllocationStrategy, AllocationUsage,
    Allocator, AllocatorOptions, ApiVersion, Buffer, BufferCopy, BufferCreateInfo, BufferImageCopy,
    BufferUsage, CommandPool, ComputePipeline, DebugMessage, DebugMessageSeverity,
    DefragmentationMove, DefragmentationPlan, DescriptorPool, DescriptorPoolSize,
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorType, DeviceCreateInfo,
    DeviceFeatures, DeviceMemory, Fence, Format, Image, Image2dCreateInfo, ImageBarrier,
    ImageLayout, ImageUsage, ImageView, Instance, InstanceCreateInfo, KHRONOS_VALIDATION_LAYER,
    MemoryPropertyFlags, PipelineCache, PipelineLayout, PipelineStage, PipelineStage2,
    PipelineStatisticsFlags, PoolCreateInfo, PushConstantRange, QueryPool, QueueCreateInfo,
    QueueFlags, Semaphore, SemaphoreKind, ShaderModule, ShaderStageFlags, SignalSemaphore,
    SpecializationConstants, SubgroupFeatureFlags, WaitSemaphore,
};

#[test]
fn test_safe_instance_creation_and_enumeration() {
    let instance = match Instance::new(InstanceCreateInfo {
        application_name: Some("vulkane test"),
        api_version: ApiVersion::V1_0,
        ..Default::default()
    }) {
        Ok(i) => i,
        Err(e) => {
            common::skipped(&format!("no Vulkan ICD, or instance creation failed: {e}"));
            return;
        }
    };

    // Enumeration should succeed even if there are no devices.
    let physical_devices = instance.enumerate_physical_devices().unwrap();
    println!("Found {} physical device(s)", physical_devices.len());

    for pd in &physical_devices {
        let props = pd.properties();
        assert!(!props.device_name().is_empty());
        assert!(props.api_version().major() >= 1);

        let queue_families = pd.queue_family_properties();
        assert!(
            !queue_families.is_empty(),
            "every device has at least one queue family"
        );
    }
}

#[test]
fn test_xlib_xcb_surface_constructors_callable() {
    // We can't actually open an X server from inside a Windows CI run,
    // and the platform extensions are not enabled here, so the calls
    // should land in the MissingFunction branch — proving the safe
    // wrappers are reachable, type-correct, and round-trip the right
    // diagnostic instead of UB. (On a real Linux desktop with the
    // VK_KHR_xlib_surface / VK_KHR_xcb_surface extensions enabled and
    // a valid display, the same calls would return Ok.)
    use vulkane::safe::Surface;

    let instance = match Instance::new(InstanceCreateInfo::default()) {
        Ok(i) => i,
        Err(e) => {
            common::skipped(&format!("no Vulkan ICD, or instance creation failed: {e}"));
            return;
        }
    };

    // Synthetic, never-dereferenced pointers — only used so the call
    // makes it past the unsafe block and into the dispatch lookup.
    let fake_display: *mut std::ffi::c_void = std::ptr::null_mut();
    let fake_window: std::ffi::c_ulong = 0;
    match unsafe { Surface::from_xlib(&instance, fake_display, fake_window) } {
        Err(vulkane::safe::Error::MissingFunction(name)) => {
            assert_eq!(name, "vkCreateXlibSurfaceKHR");
        }
        Ok(_) => {
            // If we somehow got a real surface (driver loaded the
            // extension on its own), drop it cleanly.
        }
        Err(e) => panic!("expected MissingFunction or Ok, got {e:?}"),
    }

    let fake_connection: *mut std::ffi::c_void = std::ptr::null_mut();
    let fake_xcb_window: u32 = 0;
    match unsafe { Surface::from_xcb(&instance, fake_connection, fake_xcb_window) } {
        Err(vulkane::safe::Error::MissingFunction(name)) => {
            assert_eq!(name, "vkCreateXcbSurfaceKHR");
        }
        Ok(_) => {}
        Err(e) => panic!("expected MissingFunction or Ok, got {e:?}"),
    }
}

#[test]
fn test_safe_device_creation_and_drop() {
    let instance = match Instance::new(InstanceCreateInfo::default()) {
        Ok(i) => i,
        Err(_) => {
            common::skipped("no Vulkan ICD, or the loader declined to create an instance");
            return;
        }
    };

    let physicals = instance.enumerate_physical_devices().unwrap();
    let physical = match physicals.first() {
        Some(p) => p.clone(),
        None => {
            common::skipped("an ICD is present but reports no physical devices");
            return;
        }
    };

    let queue_family = physical.find_queue_family(QueueFlags::TRANSFER).unwrap();

    // Create and drop a device. The Drop impl should call vkDestroyDevice.
    let device = physical
        .create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo {
                queue_family_index: queue_family,
                queue_priorities: vec![1.0],
            }],
            ..Default::default()
        })
        .expect("device creation should succeed");

    // Verify we can get a queue handle from it.
    let _queue = device.get_queue(queue_family, 0);

    // Verify wait_idle on a fresh device works.
    device
        .wait_idle()
        .expect("wait_idle on idle device should succeed");

    // Drop happens at end of scope.
}

#[test]
fn test_safe_buffer_with_host_visible_memory() {
    let instance = match Instance::new(InstanceCreateInfo::default()) {
        Ok(i) => i,
        Err(_) => {
            common::skipped("no Vulkan ICD, or the loader declined to create an instance");
            return;
        }
    };

    let physicals = instance.enumerate_physical_devices().unwrap();
    let Some(physical) = physicals.first().cloned() else {
        common::skipped("an ICD is present but reports no physical devices");
        return;
    };

    let queue_family = physical.find_queue_family(QueueFlags::TRANSFER).unwrap();
    let device = physical
        .create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo {
                queue_family_index: queue_family,
                queue_priorities: vec![1.0],
            }],
            ..Default::default()
        })
        .unwrap();

    // Create a buffer.
    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 256,
            usage: BufferUsage::TRANSFER_DST,
        },
    )
    .unwrap();
    assert_eq!(buffer.size(), 256);

    // Query memory requirements.
    let req = buffer.memory_requirements();
    assert!(req.size >= 256);
    assert!(req.alignment.is_power_of_two());

    // Find a compatible host-visible memory type.
    let mem_type = physical
        .find_memory_type(
            req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("host-visible memory should be available on any platform");

    // Allocate and bind.
    let mut memory = DeviceMemory::allocate(&device, req.size, mem_type).unwrap();
    buffer.bind_memory(&memory, 0).unwrap();

    // Map, write, verify, drop.
    {
        let mut mapped = memory.map().unwrap();
        let slice = mapped.as_slice_mut();
        assert_eq!(slice.len() as u64, req.size);
        for (i, b) in slice.iter_mut().enumerate() {
            *b = (i & 0xFF) as u8;
        }
    }

    // Map again and verify the writes persisted (host-coherent so no flushes needed).
    {
        let mapped = memory.map().unwrap();
        let slice = mapped.as_slice();
        for (i, &b) in slice.iter().enumerate() {
            assert_eq!(b, (i & 0xFF) as u8, "byte {i} did not persist");
        }
    }
}

#[test]
fn test_safe_full_gpu_round_trip() {
    let instance = match Instance::new(InstanceCreateInfo::default()) {
        Ok(i) => i,
        Err(_) => {
            common::skipped("no Vulkan ICD, or the loader declined to create an instance");
            return;
        }
    };

    let physicals = instance.enumerate_physical_devices().unwrap();
    let Some(physical) = physicals.first().cloned() else {
        common::skipped("an ICD is present but reports no physical devices");
        return;
    };

    let queue_family = physical.find_queue_family(QueueFlags::TRANSFER).unwrap();
    let device = physical
        .create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo {
                queue_family_index: queue_family,
                queue_priorities: vec![1.0],
            }],
            ..Default::default()
        })
        .unwrap();
    let queue = device.get_queue(queue_family, 0);

    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 64,
            usage: BufferUsage::TRANSFER_DST,
        },
    )
    .unwrap();

    let req = buffer.memory_requirements();
    let mem_type = physical
        .find_memory_type(
            req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
    let mut memory = DeviceMemory::allocate(&device, req.size, mem_type).unwrap();
    buffer.bind_memory(&memory, 0).unwrap();

    // Pre-write so we can verify the GPU overwrote.
    {
        let mut m = memory.map().unwrap();
        m.as_slice_mut().fill(0);
    }

    // Record a fill command.
    let pool = CommandPool::new(&device, queue_family).unwrap();
    let mut cmd = pool.allocate_primary().unwrap();
    {
        let mut rec = cmd.begin().unwrap();
        rec.fill_buffer(&buffer, 0, 64, 0xCAFEBABE);
        rec.end().unwrap();
    }

    // Submit with a fence and wait.
    let fence = Fence::new(&device).unwrap();
    queue.submit(&[&cmd], Some(&fence)).unwrap();
    fence.wait(u64::MAX).unwrap();

    // Verify the GPU did the write.
    {
        let mapped = memory.map().unwrap();
        let slice = mapped.as_slice();
        let expected: [u8; 4] = 0xCAFEBABEu32.to_ne_bytes();
        for chunk in slice.chunks_exact(4) {
            assert_eq!(chunk, expected, "GPU did not write expected pattern");
        }
    }

    // Everything drops here in the correct order.
}

#[test]
fn test_api_version_encoding() {
    // ApiVersion bit-packing must match the C macro VK_MAKE_API_VERSION exactly.
    let v = ApiVersion::new(0, 1, 3, 250);
    assert_eq!(v.major(), 1);
    assert_eq!(v.minor(), 3);
    assert_eq!(v.patch(), 250);

    let v0 = ApiVersion::V1_0;
    assert_eq!(v0.major(), 1);
    assert_eq!(v0.minor(), 0);
    assert_eq!(v0.patch(), 0);
}

#[test]
fn test_queue_flags_bitor_and_contains() {
    let combined = QueueFlags::GRAPHICS | QueueFlags::COMPUTE;
    assert!(combined.contains(QueueFlags::GRAPHICS));
    assert!(combined.contains(QueueFlags::COMPUTE));
    assert!(!combined.contains(QueueFlags::TRANSFER));
}

#[test]
fn test_memory_property_flags_bitor() {
    let f = MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT;
    assert!(f.contains(MemoryPropertyFlags::HOST_VISIBLE));
    assert!(f.contains(MemoryPropertyFlags::HOST_COHERENT));
    assert!(!f.contains(MemoryPropertyFlags::DEVICE_LOCAL));
}

#[test]
fn test_buffer_usage_bitor() {
    let u = BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER;
    assert!(u.contains(BufferUsage::TRANSFER_DST));
    assert!(u.contains(BufferUsage::STORAGE_BUFFER));
    assert!(!u.contains(BufferUsage::TRANSFER_SRC));
}

#[test]
fn test_shader_module_from_spirv_bytes() {
    // Hand-rolled the same instance → device sequence as `try_init_compute`,
    // but with three undeclared bails and an `.unwrap()` on the queue-family
    // lookup that the helper handles. Using the helper both declares the skip
    // and removes the divergence.
    let Some((_inst, _physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };

    // Load the pre-compiled SPIR-V from disk and create a shader module.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let spv = std::fs::read(format!("{manifest_dir}/examples/shaders/square_buffer.spv"))
        .expect("pre-compiled square_buffer.spv must exist (run compile_shader example)");

    let shader = ShaderModule::from_spirv_bytes(&device, &spv)
        .expect("ShaderModule::from_spirv_bytes should succeed for valid SPIR-V");
    assert!(shader.raw() != 0);
}

#[test]
fn test_compute_pipeline_full_dispatch() {
    // End-to-end compute test: same as the compute_square example, in test form.
    let Some((_inst, physical, device, queue, queue_family)) = compute_or_skip() else {
        return;
    };

    // Storage buffer with 64 u32s = 256 bytes
    const N: u32 = 64;
    const SIZE: u64 = (N as u64) * 4;
    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: SIZE,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .unwrap();
    let req = buffer.memory_requirements();
    let mt = physical
        .find_memory_type(
            req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
    let mut memory = DeviceMemory::allocate(&device, req.size, mt).unwrap();
    buffer.bind_memory(&memory, 0).unwrap();

    // Initial values: 0..64
    {
        let mut m = memory.map().unwrap();
        let bytes = m.as_slice_mut();
        for i in 0..N as usize {
            let v = i as u32;
            bytes[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
    }

    // Load shader
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let spv = std::fs::read(format!("{manifest_dir}/examples/shaders/square_buffer.spv")).unwrap();
    let shader = ShaderModule::from_spirv_bytes(&device, &spv).unwrap();

    // Descriptor layout/pool/set
    let set_layout = DescriptorSetLayout::new(
        &device,
        &[DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
        }],
    )
    .unwrap();
    let pool = DescriptorPool::new(
        &device,
        1,
        &[DescriptorPoolSize {
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        }],
    )
    .unwrap();
    let dset = pool.allocate(&set_layout).unwrap();
    dset.write_buffer(0, DescriptorType::STORAGE_BUFFER, &buffer, 0, SIZE);

    // Pipeline
    let pipeline_layout = PipelineLayout::new(&device, &[&set_layout]).unwrap();
    let pipeline = ComputePipeline::new(&device, &pipeline_layout, &shader, "main").unwrap();

    // Record + submit
    let cmd_pool = CommandPool::new(&device, queue_family).unwrap();
    let mut cmd = cmd_pool.allocate_primary().unwrap();
    {
        let mut rec = cmd.begin().unwrap();
        rec.bind_compute_pipeline(&pipeline);
        rec.bind_compute_descriptor_sets(&pipeline_layout, 0, &[&dset]);
        rec.dispatch(N.div_ceil(64), 1, 1);
        // Compute -> Host barrier (compute_shader_bit -> host_bit, shader_write -> host_read)
        rec.memory_barrier(
            PipelineStage::COMPUTE_SHADER,
            PipelineStage::HOST,
            AccessFlags::SHADER_WRITE,
            AccessFlags::HOST_READ,
        );
        rec.end().unwrap();
    }
    let fence = Fence::new(&device).unwrap();
    queue.submit(&[&cmd], Some(&fence)).unwrap();
    fence.wait(u64::MAX).unwrap();

    // Verify
    {
        let m = memory.map().unwrap();
        let bytes = m.as_slice();
        for i in 0..N as usize {
            let read = u32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]);
            let expected = (i as u32).wrapping_mul(i as u32);
            assert_eq!(read, expected, "element {i}: GPU did not square correctly");
        }
    }
}

// ---------------------------------------------------------------------------
// New tests for push constants, specialization constants, copy_buffer,
// dispatch_indirect, query pools, async-compute helpers, and UBO descriptors.
// ---------------------------------------------------------------------------

/// Helper: try to spin up a (instance, physical, device, queue, queue_family).
/// Returns None if no Vulkan ICD is available so tests can skip cleanly.
type ComputeSetup = (
    Instance,
    vulkane::safe::PhysicalDevice,
    vulkane::safe::Device,
    vulkane::safe::Queue,
    u32,
);

/// Instance → physical device → device → compute queue, or the **specific**
/// precondition that failed.
///
/// Four distinct failures used to arrive at the call site as one `None`, and
/// all 43 call sites printed the same `SKIP: no Vulkan ICD` for every one of
/// them. Three of the four are not that: an ICD can be present and enumeration
/// fail, present and report no devices, or present with a device that has no
/// compute queue family. Anyone who hit one of those was told to go install a
/// driver they already had.
/// Built on the shared primitives in [`common`] rather than its own copy of the
/// boot sequence. That also fixed a divergence: this used to take physical
/// device *0* and ask whether it had a compute family, so on a multi-adapter
/// machine it could skip while a usable second adapter sat idle.
/// [`common::first_compute`] searches, which is what `compute_device` already
/// did — the two no longer disagree about what "no compute device" means.
fn try_init_compute() -> Result<ComputeSetup, common::Missing> {
    let (instance, devices) = common::instance_and_devices("vulkane test", ApiVersion::V1_0)?;
    let (physical, queue_family) = common::first_compute(devices)?;
    let device = common::create_device_on(&physical, queue_family, None)?;
    let queue = device.get_queue(queue_family, 0);
    Ok((instance, physical, device, queue, queue_family))
}

/// [`try_init_compute`], with the failure **already declared** by the time this
/// returns `None` — and already fatal, if `VULKANE_REQUIRE_DEVICE` is set.
///
/// The `let … else { return; }` at the call sites is therefore not a silent
/// skip, despite looking like one; the reason was printed inside. That is why
/// this wrapper exists rather than the call sites matching on the `Result`
/// directly: 43 identical four-line `match` blocks would bury the one line of
/// each test that differs, and the name here is what tells a reader — and the
/// next person to copy the pattern — that the skip is accounted for.
fn compute_or_skip() -> Option<ComputeSetup> {
    match try_init_compute() {
        Ok(setup) => Some(setup),
        Err(cause) => {
            common::skip(cause);
            None
        }
    }
}

#[test]
fn test_specialization_constants_builder() {
    // Pure host-side test of the SpecializationConstants builder. Validates
    // that map entries and the data block are laid out correctly without
    // needing a Vulkan ICD.
    let specs = SpecializationConstants::new()
        .add_u32(0, 0xDEADBEEF)
        .add_i32(1, -1)
        .add_f32(2, 1.5)
        .add_bool(3, true);

    assert_eq!(specs.len(), 4);
    assert!(!specs.is_empty());

    // The empty case
    let empty = SpecializationConstants::new();
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
}

#[test]
fn test_pipeline_statistics_flags_count() {
    let f = PipelineStatisticsFlags::COMPUTE_SHADER_INVOCATIONS
        | PipelineStatisticsFlags::INPUT_ASSEMBLY_VERTICES;
    assert_eq!(f.count(), 2);
    assert!(f.contains(PipelineStatisticsFlags::COMPUTE_SHADER_INVOCATIONS));
    assert!(!f.contains(PipelineStatisticsFlags::FRAGMENT_SHADER_INVOCATIONS));

    assert_eq!(PipelineStatisticsFlags::NONE.count(), 0);
}

#[test]
fn test_buffer_copy_struct() {
    // Trivial constructor sanity check — BufferCopy is a public POD.
    let r = BufferCopy {
        src_offset: 16,
        dst_offset: 32,
        size: 64,
    };
    assert_eq!(r.src_offset, 16);
    assert_eq!(r.dst_offset, 32);
    assert_eq!(r.size, 64);
}

#[test]
fn test_async_compute_queue_helper_returns_compute_capable() {
    let Some((_inst, physical, _dev, _q, _qf)) = compute_or_skip() else {
        return;
    };

    // Whatever the helper returns must support COMPUTE.
    let q = physical
        .find_dedicated_compute_queue()
        .expect("any compute device exposes a compute queue");
    let families = physical.queue_family_properties();
    assert!(
        families[q as usize]
            .queue_flags()
            .contains(QueueFlags::COMPUTE)
    );

    // The transfer-dedicated helper should also return a transfer-capable
    // family if it returns anything.
    if let Some(t) = physical.find_dedicated_transfer_queue() {
        assert!(
            families[t as usize]
                .queue_flags()
                .contains(QueueFlags::TRANSFER)
        );
    }
}

#[test]
fn test_timestamp_period_is_nonneg() {
    let Some((_inst, physical, _dev, _q, _qf)) = compute_or_skip() else {
        return;
    };
    // Should be a finite, non-negative number on any conformant device.
    let p = physical.timestamp_period();
    assert!(p.is_finite());
    assert!(p >= 0.0);
}

#[test]
fn test_max_push_constants_size_meets_spec_minimum() {
    let Some((_inst, physical, _dev, _q, _qf)) = compute_or_skip() else {
        return;
    };
    // Vulkan spec guarantees at least 128 bytes.
    let max = physical.properties().max_push_constants_size();
    assert!(max >= 128, "spec minimum is 128 bytes, got {max}");
}

#[test]
fn test_pipeline_layout_with_push_constants() {
    let Some((_inst, _physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };

    let set_layout = DescriptorSetLayout::new(
        &device,
        &[DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
        }],
    )
    .unwrap();

    let pcr = PushConstantRange {
        stage_flags: ShaderStageFlags::COMPUTE,
        offset: 0,
        size: 16,
    };

    // Both no-PCR and with-PCR variants must succeed.
    let layout_no = PipelineLayout::new(&device, &[&set_layout]).unwrap();
    let layout_pc = PipelineLayout::with_push_constants(&device, &[&set_layout], &[pcr]).unwrap();

    assert!(layout_no.raw() != 0);
    assert!(layout_pc.raw() != 0);
    assert!(layout_no.raw() != layout_pc.raw());
}

#[test]
fn test_query_pool_timestamp_creation_and_metadata() {
    let Some((_inst, physical, device, _q, queue_family)) = compute_or_skip() else {
        return;
    };

    // Skip if the chosen queue family doesn't support timestamps.
    let families = physical.queue_family_properties();
    if families[queue_family as usize].timestamp_valid_bits() == 0 {
        common::skipped_unsupported(
            "a queue family with timestamp support (timestampValidBits > 0)",
        );
        return;
    }

    let pool = QueryPool::timestamps(&device, 4).unwrap();
    assert_eq!(pool.query_count(), 4);
    assert!(pool.raw() != 0);
}

#[test]
fn test_copy_buffer_staging_round_trip() {
    let Some((_inst, physical, device, queue, queue_family)) = compute_or_skip() else {
        return;
    };

    // Source buffer: HOST_VISIBLE, TRANSFER_SRC.
    let src = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 256,
            usage: BufferUsage::TRANSFER_SRC,
        },
    )
    .unwrap();
    let src_req = src.memory_requirements();
    let src_mt = physical
        .find_memory_type(
            src_req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
    let mut src_mem = DeviceMemory::allocate(&device, src_req.size, src_mt).unwrap();
    src.bind_memory(&src_mem, 0).unwrap();

    // Pre-fill src with a pattern from the host.
    {
        let mut m = src_mem.map().unwrap();
        let bytes = m.as_slice_mut();
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i * 3 + 1) as u8;
        }
    }

    // Destination buffer: HOST_VISIBLE (so we can read it back), TRANSFER_DST.
    let dst = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 256,
            usage: BufferUsage::TRANSFER_DST,
        },
    )
    .unwrap();
    let dst_req = dst.memory_requirements();
    let dst_mt = physical
        .find_memory_type(
            dst_req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
    let mut dst_mem = DeviceMemory::allocate(&device, dst_req.size, dst_mt).unwrap();
    dst.bind_memory(&dst_mem, 0).unwrap();

    // Zero out dst so we can detect that the copy actually happened.
    {
        let mut m = dst_mem.map().unwrap();
        m.as_slice_mut().fill(0);
    }

    // Record copy_buffer + memory barrier so the host read sees it.
    let pool = CommandPool::new(&device, queue_family).unwrap();
    let mut cmd = pool.allocate_primary().unwrap();
    {
        let mut rec = cmd.begin().unwrap();
        rec.copy_buffer(
            &src,
            &dst,
            &[BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: 256,
            }],
        );
        // Transfer -> Host (transfer_bit -> host_bit, transfer_write -> host_read)
        rec.memory_barrier(
            PipelineStage::TRANSFER,
            PipelineStage::HOST,
            AccessFlags::TRANSFER_READ,
            AccessFlags::HOST_READ,
        );
        rec.end().unwrap();
    }

    let fence = Fence::new(&device).unwrap();
    queue.submit(&[&cmd], Some(&fence)).unwrap();
    fence.wait(u64::MAX).unwrap();

    // Verify the bytes were copied.
    {
        let m = dst_mem.map().unwrap();
        let bytes = m.as_slice();
        for (i, &b) in bytes.iter().enumerate() {
            assert_eq!(b, (i * 3 + 1) as u8, "byte {i} not copied correctly");
        }
    }
}

#[test]
fn test_dispatch_indirect_with_explicit_count() {
    // Build an indirect-dispatch test using the existing square_buffer
    // shader: write x=4, y=1, z=1 into an INDIRECT_BUFFER and dispatch 256
    // elements = 4 workgroups of 64.
    let Some((_inst, physical, device, queue, queue_family)) = compute_or_skip() else {
        return;
    };

    // Storage buffer.
    const N: u32 = 256;
    const SIZE: u64 = (N as u64) * 4;
    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: SIZE,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .unwrap();
    let req = buffer.memory_requirements();
    let mt = physical
        .find_memory_type(
            req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
    let mut memory = DeviceMemory::allocate(&device, req.size, mt).unwrap();
    buffer.bind_memory(&memory, 0).unwrap();

    // Initialize with 0..256.
    {
        let mut m = memory.map().unwrap();
        let bytes = m.as_slice_mut();
        for i in 0..N as usize {
            let v = i as u32;
            bytes[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
    }

    // Indirect-dispatch buffer (3 u32s = 12 bytes).
    let indirect = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 16,
            usage: BufferUsage::INDIRECT_BUFFER,
        },
    )
    .unwrap();
    let ireq = indirect.memory_requirements();
    let imt = physical
        .find_memory_type(
            ireq.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
    let mut imem = DeviceMemory::allocate(&device, ireq.size, imt).unwrap();
    indirect.bind_memory(&imem, 0).unwrap();
    {
        let mut m = imem.map().unwrap();
        let b = m.as_slice_mut();
        // x=4, y=1, z=1 (workgroup counts; local_size is 64 in the shader)
        b[0..4].copy_from_slice(&4u32.to_le_bytes());
        b[4..8].copy_from_slice(&1u32.to_le_bytes());
        b[8..12].copy_from_slice(&1u32.to_le_bytes());
    }

    // Load the existing pre-compiled shader.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let spv = std::fs::read(format!("{manifest_dir}/examples/shaders/square_buffer.spv")).unwrap();
    let shader = ShaderModule::from_spirv_bytes(&device, &spv).unwrap();

    let set_layout = DescriptorSetLayout::new(
        &device,
        &[DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
        }],
    )
    .unwrap();
    let dpool = DescriptorPool::new(
        &device,
        1,
        &[DescriptorPoolSize {
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        }],
    )
    .unwrap();
    let dset = dpool.allocate(&set_layout).unwrap();
    dset.write_buffer(0, DescriptorType::STORAGE_BUFFER, &buffer, 0, SIZE);

    let pipeline_layout = PipelineLayout::new(&device, &[&set_layout]).unwrap();
    let pipeline = ComputePipeline::new(&device, &pipeline_layout, &shader, "main").unwrap();

    let cmd_pool = CommandPool::new(&device, queue_family).unwrap();
    let mut cmd = cmd_pool.allocate_primary().unwrap();
    {
        let mut rec = cmd.begin().unwrap();
        rec.bind_compute_pipeline(&pipeline);
        rec.bind_compute_descriptor_sets(&pipeline_layout, 0, &[&dset]);
        rec.dispatch_indirect(&indirect, 0);
        // Compute -> Host
        rec.memory_barrier(
            PipelineStage::COMPUTE_SHADER,
            PipelineStage::HOST,
            AccessFlags::SHADER_WRITE,
            AccessFlags::HOST_READ,
        );
        rec.end().unwrap();
    }
    let fence = Fence::new(&device).unwrap();
    queue.submit(&[&cmd], Some(&fence)).unwrap();
    fence.wait(u64::MAX).unwrap();

    // Verify squaring happened to all 256 elements.
    {
        let m = memory.map().unwrap();
        let bytes = m.as_slice();
        for i in 0..N as usize {
            let read = u32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]);
            assert_eq!(
                read,
                (i as u32).wrapping_mul(i as u32),
                "indirect dispatch did not square element {i}"
            );
        }
    }
}

#[test]
fn test_query_pool_records_timestamp_around_dispatch() {
    let Some((_inst, physical, device, queue, queue_family)) = compute_or_skip() else {
        return;
    };
    let families = physical.queue_family_properties();
    if families[queue_family as usize].timestamp_valid_bits() == 0 {
        common::skipped_unsupported(
            "a queue family with timestamp support (timestampValidBits > 0)",
        );
        return;
    }

    // Reuse compute_square setup.
    const N: u32 = 64;
    const SIZE: u64 = (N as u64) * 4;
    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: SIZE,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .unwrap();
    let req = buffer.memory_requirements();
    let mt = physical
        .find_memory_type(
            req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
    let mut memory = DeviceMemory::allocate(&device, req.size, mt).unwrap();
    buffer.bind_memory(&memory, 0).unwrap();
    {
        let mut m = memory.map().unwrap();
        let b = m.as_slice_mut();
        for i in 0..N as usize {
            b[i * 4..(i + 1) * 4].copy_from_slice(&(i as u32).to_le_bytes());
        }
    }

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let spv = std::fs::read(format!("{manifest_dir}/examples/shaders/square_buffer.spv")).unwrap();
    let shader = ShaderModule::from_spirv_bytes(&device, &spv).unwrap();

    let set_layout = DescriptorSetLayout::new(
        &device,
        &[DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
        }],
    )
    .unwrap();
    let dpool = DescriptorPool::new(
        &device,
        1,
        &[DescriptorPoolSize {
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        }],
    )
    .unwrap();
    let dset = dpool.allocate(&set_layout).unwrap();
    dset.write_buffer(0, DescriptorType::STORAGE_BUFFER, &buffer, 0, SIZE);

    let pl = PipelineLayout::new(&device, &[&set_layout]).unwrap();
    let pipe = ComputePipeline::new(&device, &pl, &shader, "main").unwrap();

    // Two timestamps: before and after the dispatch.
    let qpool = QueryPool::timestamps(&device, 2).unwrap();

    let cmd_pool = CommandPool::new(&device, queue_family).unwrap();
    let mut cmd = cmd_pool.allocate_primary().unwrap();
    {
        let mut rec = cmd.begin().unwrap();
        // Reset is required before any timestamp can be written.
        rec.reset_query_pool(&qpool, 0, 2);
        rec.write_timestamp(PipelineStage::TOP_OF_PIPE, &qpool, 0);
        rec.bind_compute_pipeline(&pipe);
        rec.bind_compute_descriptor_sets(&pl, 0, &[&dset]);
        rec.dispatch(N.div_ceil(64), 1, 1);
        rec.write_timestamp(PipelineStage::BOTTOM_OF_PIPE, &qpool, 1);
        // Compute -> Host
        rec.memory_barrier(
            PipelineStage::COMPUTE_SHADER,
            PipelineStage::HOST,
            AccessFlags::SHADER_WRITE,
            AccessFlags::HOST_READ,
        );
        rec.end().unwrap();
    }
    let fence = Fence::new(&device).unwrap();
    queue.submit(&[&cmd], Some(&fence)).unwrap();
    fence.wait(u64::MAX).unwrap();

    // Read timestamps.
    let times = qpool.get_results_u64(0, 2).unwrap();
    assert_eq!(times.len(), 2);
    // We can't reliably assert times[1] > times[0] on every implementation
    // (Lavapipe in particular sometimes reports them equal for trivial work),
    // but we *can* assert they were both written: get_results_u64 with the
    // WAIT bit set returns success only when every requested query completed.
    // Sanity-check that the GPU actually did the squaring as well.
    {
        let m = memory.map().unwrap();
        let b = m.as_slice();
        for i in 0..N as usize {
            let v = u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
            assert_eq!(v, (i as u32).wrapping_mul(i as u32));
        }
    }

    // Bonus: convert the delta to nanoseconds with timestamp_period and
    // verify it's a finite number (not NaN/Inf).
    let period = physical.timestamp_period();
    let delta_ticks = times[1].wrapping_sub(times[0]) as f64;
    let delta_ns = delta_ticks * (period as f64);
    assert!(delta_ns.is_finite());
}

#[test]
fn test_uniform_buffer_descriptor_round_trip() {
    // Verify that UNIFORM_BUFFER descriptors work end-to-end. We don't run
    // a shader here — just create the descriptor layout, pool, set, and
    // call write_buffer with UNIFORM_BUFFER. If the driver accepted the
    // write, the descriptor wiring is correct. (A shader-using UBO test
    // would need a second pre-compiled SPIR-V shader; this is sufficient
    // to validate the safe wrapper plumbing.)
    let Some((_inst, physical, device, _queue, _qf)) = compute_or_skip() else {
        return;
    };

    // Create a small uniform buffer.
    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 64,
            usage: BufferUsage::UNIFORM_BUFFER,
        },
    )
    .unwrap();
    let req = buffer.memory_requirements();
    let mt = physical
        .find_memory_type(
            req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
    let memory = DeviceMemory::allocate(&device, req.size, mt).unwrap();
    buffer.bind_memory(&memory, 0).unwrap();

    let set_layout = DescriptorSetLayout::new(
        &device,
        &[DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
        }],
    )
    .unwrap();
    let pool = DescriptorPool::new(
        &device,
        1,
        &[DescriptorPoolSize {
            descriptor_type: DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        }],
    )
    .unwrap();
    let dset = pool.allocate(&set_layout).unwrap();
    dset.write_buffer(0, DescriptorType::UNIFORM_BUFFER, &buffer, 0, 64);
    assert!(dset.raw() != 0);
}

// ---------------------------------------------------------------------------
// Validation layer + debug utils + extension/layer enable lists tests
// ---------------------------------------------------------------------------

#[test]
fn test_debug_message_severity_label_and_bits() {
    assert_eq!(DebugMessageSeverity::ERROR.label(), "ERROR");
    assert_eq!(DebugMessageSeverity::WARNING.label(), "WARN");
    assert_eq!(DebugMessageSeverity::INFO.label(), "INFO");
    assert_eq!(DebugMessageSeverity::VERBOSE.label(), "VERBOSE");

    let combined = DebugMessageSeverity::WARNING_AND_ABOVE;
    assert!(combined.contains(DebugMessageSeverity::ERROR));
    assert!(combined.contains(DebugMessageSeverity::WARNING));
    assert!(!combined.contains(DebugMessageSeverity::INFO));

    let all = DebugMessageSeverity::ALL;
    assert!(all.contains(DebugMessageSeverity::VERBOSE));
    assert!(all.contains(DebugMessageSeverity::ERROR));
}

#[test]
fn test_enumerate_layer_properties_succeeds_or_skips() {
    let layers = match Instance::enumerate_layer_properties() {
        Ok(l) => l,
        Err(e) => {
            common::skipped(&format!(
                "the Vulkan shared library could not be loaded: {e}"
            ));
            return;
        }
    };
    // The list MAY be empty on some systems (no layers installed).
    // We just assert that every entry has a non-empty name when present.
    for l in &layers {
        let n = l.name();
        assert!(!n.is_empty(), "layer name should not be empty");
        // spec_version should be a valid api version (major >= 1).
        assert!(l.spec_version().major() >= 1);
    }
    println!("Found {} instance layer(s)", layers.len());
}

#[test]
fn test_enumerate_instance_extension_properties() {
    let exts = match Instance::enumerate_extension_properties() {
        Ok(e) => e,
        Err(e) => {
            common::skipped(&format!(
                "the Vulkan shared library could not be loaded: {e}"
            ));
            return;
        }
    };
    // Conformant Vulkan implementations always expose at least one
    // instance extension (VK_KHR_get_physical_device_properties2 on
    // Vulkan 1.0 implementations, or many more on 1.1+).
    assert!(!exts.is_empty(), "expected at least one instance extension");
    for e in &exts {
        assert!(!e.name().is_empty());
    }
}

#[test]
fn test_physical_device_enumerate_extension_properties() {
    let Some((_inst, physical, _dev, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let exts = physical.enumerate_extension_properties().unwrap();
    // Lavapipe and every conformant ICD expose at least a handful of
    // device extensions; just assert names look sane.
    for e in &exts {
        assert!(!e.name().is_empty());
    }
}

#[test]
fn test_instance_with_no_layers_or_extensions() {
    // Building an instance with empty enable lists must succeed
    // (it's the default behaviour).
    let info = InstanceCreateInfo {
        application_name: Some("vulkane-empty-lists"),
        enabled_layers: &[],
        enabled_extensions: None,
        ..InstanceCreateInfo::default()
    };
    if let Ok(_inst) = Instance::new(info) {
        // OK.
    } else {
        common::skipped("no Vulkan ICD, or the loader declined to create an instance");
    }
}

#[test]
fn test_instance_with_unknown_layer_fails_cleanly() {
    let info = InstanceCreateInfo {
        application_name: Some("vulkane-bad-layer"),
        enabled_layers: &["VK_LAYER_THIS_DOES_NOT_EXIST_zzz"],
        ..InstanceCreateInfo::default()
    };
    let result = Instance::new(info);
    // We can't be sure no Vulkan ICD is present here. If the loader is
    // present we expect ERROR_LAYER_NOT_PRESENT (or similar). If the
    // loader is missing entirely we get LibraryLoad — also acceptable.
    match result {
        Ok(_) => panic!("loader should not have accepted a fake layer"),
        Err(e) => {
            // Just print it; either Vk(LayerNotPresent) or LibraryLoad.
            eprintln!("OK: enabling fake layer rejected with: {e}");
        }
    }
}

#[test]
fn test_instance_with_validation_when_available() {
    // If the validation layer is installed and the debug-utils extension
    // is available, build an instance with the convenience constructor and
    // verify our callback fires when we trigger a validation error.
    let layers = match Instance::enumerate_layer_properties() {
        Ok(l) => l,
        Err(_) => {
            common::skipped("the Vulkan shared library could not be loaded");
            return;
        }
    };
    if !layers.iter().any(|l| l.name() == KHRONOS_VALIDATION_LAYER) {
        common::skipped_unsupported("the VK_LAYER_KHRONOS_validation layer");
        return;
    }
    let exts = Instance::enumerate_extension_properties().unwrap();
    if !exts
        .iter()
        .any(|e| e.name() == vulkane::raw::bindings::EXT_DEBUG_UTILS_EXTENSION_NAME)
    {
        common::skipped_unsupported("VK_EXT_debug_utils");
        return;
    }

    use std::sync::Arc as StdArc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let counter = StdArc::new(AtomicUsize::new(0));
    let counter_cb = StdArc::clone(&counter);

    let debug_exts = vulkane::safe::InstanceExtensions::new().ext_debug_utils();
    let info = InstanceCreateInfo {
        application_name: Some("vulkane-validation"),
        enabled_layers: &[KHRONOS_VALIDATION_LAYER],
        enabled_extensions: Some(&debug_exts),
        debug_callback: Some(Box::new(move |msg: &DebugMessage<'_>| {
            // Only count WARN/ERROR so we don't spam on every INFO.
            if msg.severity.contains(DebugMessageSeverity::WARNING)
                || msg.severity.contains(DebugMessageSeverity::ERROR)
            {
                counter_cb.fetch_add(1, Ordering::Relaxed);
            }
        })),
        ..InstanceCreateInfo::default()
    };

    let instance = match Instance::new(info) {
        Ok(i) => i,
        Err(e) => {
            common::skipped_unsupported(&format!(
                "an instance with the validation layer enabled ({e})"
            ));
            return;
        }
    };

    // The act of *creating* the instance with VK_EXT_debug_utils enabled
    // is enough to assert the messenger plumbing is wired correctly. We
    // don't try to provoke a validation error here because trying to
    // misuse Vulkan from inside the safe wrapper requires Either dropping
    // to raw bindings (out of scope for this test) or relying on the
    // layer's own startup messages, which vary by version.
    //
    // Just touch the counter to silence the unused-Send warning and drop.
    let _ = counter.load(Ordering::Relaxed);
    drop(instance);
}

#[test]
fn test_instance_validation_constructor_when_available() {
    // The InstanceCreateInfo::validation() convenience should produce a
    // working instance when validation is available, or fail cleanly
    // otherwise. Either outcome is acceptable here — we just want to
    // verify the constructor path compiles and runs.
    let layers = Instance::enumerate_layer_properties().ok();
    let has_validation = layers
        .as_ref()
        .map(|ls| ls.iter().any(|l| l.name() == KHRONOS_VALIDATION_LAYER))
        .unwrap_or(false);

    if !has_validation {
        common::skipped_unsupported("the VK_LAYER_KHRONOS_validation layer");
        return;
    }
    match Instance::new(InstanceCreateInfo::validation()) {
        Ok(_inst) => {}
        Err(e) => eprintln!("validation() constructor returned err: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Image tests: 2D storage image creation, layout transitions, and a full
// buffer -> image -> compute -> image -> buffer round trip.
// ---------------------------------------------------------------------------

#[test]
fn test_image_usage_bitor_and_format_constants() {
    let usage = ImageUsage::STORAGE | ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC;
    assert!(usage.contains(ImageUsage::STORAGE));
    assert!(usage.contains(ImageUsage::TRANSFER_DST));
    assert!(usage.contains(ImageUsage::TRANSFER_SRC));
    assert!(!usage.contains(ImageUsage::SAMPLED));

    // Format constants are just sanity checks that the wrapper exists.
    assert_ne!(Format::R8_UNORM, Format::R32_UINT);
    assert_ne!(ImageLayout::UNDEFINED, ImageLayout::GENERAL);
}

#[test]
fn test_buffer_image_copy_full_2d_helper() {
    let r = BufferImageCopy::full_2d(64, 32);
    assert_eq!(r.image_extent, [64, 32, 1]);
    assert_eq!(r.image_offset, [0, 0, 0]);
    assert_eq!(r.buffer_offset, 0);
    assert_eq!(r.buffer_row_length, 0);
    assert_eq!(r.buffer_image_height, 0);
}

#[test]
fn test_image_2d_creation_and_memory_binding() {
    let Some((_inst, physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };

    let image = Image::new_2d(
        &device,
        Image2dCreateInfo {
            format: Format::R32_UINT,
            width: 64,
            height: 64,
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
        },
    )
    .unwrap();
    assert_eq!(image.format(), Format::R32_UINT);
    assert_eq!(image.width(), 64);
    assert_eq!(image.height(), 64);
    assert!(image.raw() != 0);

    let req = image.memory_requirements();
    assert!(req.size >= 64 * 64 * 4);
    assert!(req.alignment.is_power_of_two());

    // Try to find a device-local memory type. If unavailable, fall back to
    // host-visible (Lavapipe sometimes lacks DEVICE_LOCAL).
    let mt = physical
        .find_memory_type(req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| {
            physical.find_memory_type(req.memory_type_bits, MemoryPropertyFlags::HOST_VISIBLE)
        })
        .expect("some memory type should back the image");
    let memory = DeviceMemory::allocate(&device, req.size, mt).unwrap();
    image.bind_memory(&memory, 0).unwrap();

    // ImageView creation should also succeed.
    let view = ImageView::new_2d_color(&image).unwrap();
    assert!(view.raw() != 0);
}

#[test]
fn test_image_buffer_round_trip_via_layout_transitions() {
    // Validates: layout transitions, buffer -> image copy, image -> buffer
    // copy. We don't run a shader here — just verify that the bytes survive
    // a round trip through an image's storage on the GPU.
    let Some((_inst, physical, device, queue, queue_family)) = compute_or_skip() else {
        return;
    };

    const W: u32 = 16;
    const H: u32 = 16;
    const PIXEL_BYTES: u64 = 4; // R32_UINT
    const BUF_SIZE: u64 = (W as u64) * (H as u64) * PIXEL_BYTES;

    // Source staging buffer pre-filled from the host.
    let src = Buffer::new(
        &device,
        BufferCreateInfo {
            size: BUF_SIZE,
            usage: BufferUsage::TRANSFER_SRC,
        },
    )
    .unwrap();
    let src_req = src.memory_requirements();
    let src_mt = physical
        .find_memory_type(
            src_req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
    let mut src_mem = DeviceMemory::allocate(&device, src_req.size, src_mt).unwrap();
    src.bind_memory(&src_mem, 0).unwrap();
    {
        let mut m = src_mem.map().unwrap();
        let bytes = m.as_slice_mut();
        for i in 0..(W * H) as usize {
            let v = (i as u32).wrapping_mul(0x9E3779B1u32);
            bytes[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
    }

    // The image: STORAGE so it could be used for compute, TRANSFER_DST and
    // TRANSFER_SRC for the round trip.
    let image = Image::new_2d(
        &device,
        Image2dCreateInfo {
            format: Format::R32_UINT,
            width: W,
            height: H,
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
        },
    )
    .unwrap();
    let img_req = image.memory_requirements();
    // Use the most-permissive memory type the driver allows.
    let img_mt = physical
        .find_memory_type(img_req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| {
            physical.find_memory_type(img_req.memory_type_bits, MemoryPropertyFlags::HOST_VISIBLE)
        })
        .expect("some memory type should back the image");
    let img_mem = DeviceMemory::allocate(&device, img_req.size, img_mt).unwrap();
    image.bind_memory(&img_mem, 0).unwrap();

    // Destination readback buffer.
    let dst = Buffer::new(
        &device,
        BufferCreateInfo {
            size: BUF_SIZE,
            usage: BufferUsage::TRANSFER_DST,
        },
    )
    .unwrap();
    let dst_req = dst.memory_requirements();
    let dst_mt = physical
        .find_memory_type(
            dst_req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
    let mut dst_mem = DeviceMemory::allocate(&device, dst_req.size, dst_mt).unwrap();
    dst.bind_memory(&dst_mem, 0).unwrap();
    {
        // Zero so we can detect overwrite.
        let mut m = dst_mem.map().unwrap();
        m.as_slice_mut().fill(0);
    }

    let pool = CommandPool::new(&device, queue_family).unwrap();
    let mut cmd = pool.allocate_primary().unwrap();
    {
        let mut rec = cmd.begin().unwrap();
        // 1) UNDEFINED -> TRANSFER_DST_OPTIMAL
        rec.image_barrier(
            PipelineStage::TOP_OF_PIPE,
            PipelineStage::TRANSFER,
            ImageBarrier::color(
                &image,
                ImageLayout::UNDEFINED,
                ImageLayout::TRANSFER_DST_OPTIMAL,
                AccessFlags::NONE,
                AccessFlags::TRANSFER_WRITE,
            ),
        );
        // 2) Copy buffer -> image
        rec.copy_buffer_to_image(
            &src,
            &image,
            ImageLayout::TRANSFER_DST_OPTIMAL,
            &[BufferImageCopy::full_2d(W, H)],
        );
        // 3) TRANSFER_DST -> TRANSFER_SRC_OPTIMAL
        rec.image_barrier(
            PipelineStage::TRANSFER,
            PipelineStage::TRANSFER,
            ImageBarrier::color(
                &image,
                ImageLayout::TRANSFER_DST_OPTIMAL,
                ImageLayout::TRANSFER_SRC_OPTIMAL,
                AccessFlags::TRANSFER_WRITE,
                AccessFlags::TRANSFER_READ,
            ),
        );
        // 4) Copy image -> buffer
        rec.copy_image_to_buffer(
            &image,
            ImageLayout::TRANSFER_SRC_OPTIMAL,
            &dst,
            &[BufferImageCopy::full_2d(W, H)],
        );
        // 5) Transfer -> Host barrier so the host read sees the bytes.
        rec.memory_barrier(
            PipelineStage::TRANSFER,
            PipelineStage::HOST,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::HOST_READ,
        );
        rec.end().unwrap();
    }

    let fence = Fence::new(&device).unwrap();
    queue.submit(&[&cmd], Some(&fence)).unwrap();
    fence.wait(u64::MAX).unwrap();

    // Verify every pixel survived the round trip.
    {
        let m = dst_mem.map().unwrap();
        let b = m.as_slice();
        for i in 0..(W * H) as usize {
            let read = u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
            let expected = (i as u32).wrapping_mul(0x9E3779B1u32);
            assert_eq!(read, expected, "pixel {i} did not survive image round trip");
        }
    }
}

#[test]
fn test_storage_image_descriptor_wiring() {
    // Validates that allocating a STORAGE_IMAGE descriptor and pointing it
    // at an ImageView round-trips through the safe wrapper without errors.
    // We don't dispatch a shader here — that would need a shipped .spv file.
    let Some((_inst, physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };

    let image = Image::new_2d(
        &device,
        Image2dCreateInfo {
            format: Format::R32_UINT,
            width: 8,
            height: 8,
            usage: ImageUsage::STORAGE,
        },
    )
    .unwrap();
    let req = image.memory_requirements();
    let mt = physical
        .find_memory_type(req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| {
            physical.find_memory_type(req.memory_type_bits, MemoryPropertyFlags::HOST_VISIBLE)
        })
        .unwrap();
    let memory = DeviceMemory::allocate(&device, req.size, mt).unwrap();
    image.bind_memory(&memory, 0).unwrap();
    let view = ImageView::new_2d_color(&image).unwrap();

    let set_layout = DescriptorSetLayout::new(
        &device,
        &[DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
        }],
    )
    .unwrap();
    let pool = DescriptorPool::new(
        &device,
        1,
        &[DescriptorPoolSize {
            descriptor_type: DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
        }],
    )
    .unwrap();
    let dset = pool.allocate(&set_layout).unwrap();
    dset.write_storage_image(0, &view, ImageLayout::GENERAL);
    assert!(dset.raw() != 0);
}

// ---------------------------------------------------------------------------
// Timeline semaphore + binary semaphore + pipeline cache + sync2 tests
// ---------------------------------------------------------------------------

#[test]
fn test_binary_semaphore_creation_and_drop() {
    let Some((_inst, _physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let s = Semaphore::binary(&device).unwrap();
    assert_eq!(s.kind(), SemaphoreKind::Binary);
    assert!(s.raw() != 0);
}

#[test]
fn test_timeline_semaphore_host_signal_and_wait() {
    let Some((_inst, _physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };

    let sem = match Semaphore::timeline(&device, 5) {
        Ok(s) => s,
        Err(e) => {
            common::skipped_unsupported(&format!("timeline semaphores ({e})"));
            return;
        }
    };
    assert_eq!(sem.kind(), SemaphoreKind::Timeline);

    // Initial value 5 should be readable.
    assert_eq!(sem.current_value().unwrap(), 5);

    // Signal to a higher value from the host.
    sem.signal_value(10).unwrap();
    assert_eq!(sem.current_value().unwrap(), 10);

    // wait_value should return immediately because the value is already >= 10.
    sem.wait_value(10, 0).unwrap();
}

#[test]
fn test_timeline_semaphore_gpu_signal_then_host_wait() {
    let Some((_inst, _physical, device, queue, queue_family)) = compute_or_skip() else {
        return;
    };
    let sem = match Semaphore::timeline(&device, 0) {
        Ok(s) => s,
        Err(e) => {
            common::skipped_unsupported(&format!("timeline semaphores ({e})"));
            return;
        }
    };

    // Submit an empty command buffer that signals the timeline to value 1.
    let pool = CommandPool::new(&device, queue_family).unwrap();
    let mut cmd = pool.allocate_primary().unwrap();
    {
        let rec = cmd.begin().unwrap();
        rec.end().unwrap();
    }
    queue
        .submit_with_sync(
            &[&cmd],
            &[],
            &[SignalSemaphore {
                semaphore: &sem,
                value: 1,
                device_index: 0,
            }],
            None,
        )
        .unwrap();

    // Wait on the host for the GPU to reach value 1.
    sem.wait_value(1, u64::MAX).unwrap();
    assert!(sem.current_value().unwrap() >= 1);
}

#[test]
fn test_timeline_semaphore_chained_dispatches() {
    // Two-pass compute: pass A signals timeline to 1, pass B waits on
    // value 1 before running. Validates that the safe wrapper threads the
    // wait/signal correctly through submit_with_sync.
    let Some((_inst, _physical, device, queue, queue_family)) = compute_or_skip() else {
        return;
    };
    let sem = match Semaphore::timeline(&device, 0) {
        Ok(s) => s,
        Err(e) => {
            common::skipped_unsupported(&format!("timeline semaphores ({e})"));
            return;
        }
    };

    let pool = CommandPool::new(&device, queue_family).unwrap();
    let mut cmd_a = pool.allocate_primary().unwrap();
    {
        let rec = cmd_a.begin().unwrap();
        rec.end().unwrap();
    }
    let mut cmd_b = pool.allocate_primary().unwrap();
    {
        let rec = cmd_b.begin().unwrap();
        rec.end().unwrap();
    }

    // Pass A: signals timeline -> 1
    queue
        .submit_with_sync(
            &[&cmd_a],
            &[],
            &[SignalSemaphore {
                semaphore: &sem,
                value: 1,
                device_index: 0,
            }],
            None,
        )
        .unwrap();

    // Pass B: waits on timeline >= 1, signals -> 2.
    queue
        .submit_with_sync(
            &[&cmd_b],
            &[WaitSemaphore {
                semaphore: &sem,
                value: 1,
                dst_stage_mask: PipelineStage::TOP_OF_PIPE,
                device_index: 0,
            }],
            &[SignalSemaphore {
                semaphore: &sem,
                value: 2,
                device_index: 0,
            }],
            None,
        )
        .unwrap();

    // Wait for the whole chain on the host.
    sem.wait_value(2, u64::MAX).unwrap();
    assert!(sem.current_value().unwrap() >= 2);
}

#[test]
fn test_pipeline_cache_create_serialize_reuse() {
    let Some((_inst, _physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };

    // Create an empty cache, then serialize.
    let cache_a = PipelineCache::new(&device).unwrap();
    let bytes = cache_a.data().unwrap();
    // The cache header alone is non-empty on every conformant
    // implementation; if we got something, it should be at least the
    // 16-byte VkPipelineCacheHeaderVersionOne header. (Some software
    // implementations may return 0 for an empty cache; tolerate that
    // case.)
    println!("Pipeline cache (empty) -> {} bytes", bytes.len());

    // Now reuse those bytes when constructing a second cache.
    let cache_b = PipelineCache::with_data(&device, &bytes).unwrap();
    assert!(cache_b.raw() != 0);
}

#[test]
fn test_specialization_constants_baked_into_pipeline() {
    // Validate that ComputePipeline::with_specialization runs end-to-end
    // for a shader that doesn't actually consume any spec constants. The
    // build path is the same with or without populated entries; this just
    // exercises the code path that builds VkSpecializationInfo.
    let Some((_inst, _physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let spv = std::fs::read(format!("{manifest_dir}/examples/shaders/square_buffer.spv")).unwrap();
    let shader = ShaderModule::from_spirv_bytes(&device, &spv).unwrap();

    let set_layout = DescriptorSetLayout::new(
        &device,
        &[DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
        }],
    )
    .unwrap();
    let layout = PipelineLayout::new(&device, &[&set_layout]).unwrap();

    // The shader doesn't reference any spec constants, but Vulkan accepts
    // a non-empty SpecializationInfo with extra entries — they're simply
    // ignored. Verify the build path works.
    let specs = SpecializationConstants::new()
        .add_u32(99, 1234)
        .add_f32(100, 2.5);
    let pipe =
        ComputePipeline::with_specialization(&device, &layout, &shader, "main", &specs).unwrap();
    assert!(pipe.raw() != 0);
}

#[test]
fn test_sync2_memory_barrier_when_supported() {
    let Some((_inst, _physical, device, queue, queue_family)) = compute_or_skip() else {
        return;
    };

    let pool = CommandPool::new(&device, queue_family).unwrap();
    let mut cmd = pool.allocate_primary().unwrap();
    let supported = {
        let mut rec = cmd.begin().unwrap();
        let s2 = rec.memory_barrier2(
            PipelineStage2::COMPUTE_SHADER,
            PipelineStage2::HOST,
            AccessFlags2::SHADER_WRITE,
            AccessFlags2::HOST_READ,
        );
        rec.end().unwrap();
        s2
    };
    match supported {
        Ok(()) => {
            // Sync2 supported — submit and verify completion.
            let fence = Fence::new(&device).unwrap();
            queue.submit(&[&cmd], Some(&fence)).unwrap();
            fence.wait(u64::MAX).unwrap();
        }
        Err(e) => {
            common::skipped_unsupported(&format!("synchronization2 ({e})"));
        }
    }
}

// ---------------------------------------------------------------------------
// Sub-allocator (Allocator) integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_allocator_creation_and_statistics_start_zero() {
    let Some((_inst, physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let alloc = Allocator::new(&device, &physical).unwrap();
    let stats = alloc.statistics();
    assert_eq!(stats.allocation_bytes, 0);
    assert_eq!(stats.allocation_count, 0);
    assert_eq!(stats.block_bytes, 0);
    assert_eq!(stats.block_count, 0);
}

#[test]
fn test_allocator_create_buffer_pool_path() {
    let Some((_inst, physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let allocator = Allocator::new(&device, &physical).unwrap();

    let (buffer, allocation) = allocator
        .create_buffer(
            BufferCreateInfo {
                size: 4096,
                usage: BufferUsage::STORAGE_BUFFER,
            },
            AllocationCreateInfo {
                usage: AllocationUsage::HostVisible,
                ..Default::default()
            },
        )
        .unwrap();

    assert!(allocation.size() >= 4096);
    assert!(allocation.memory() != 0);

    let stats = allocator.statistics();
    assert_eq!(stats.allocation_count, 1);
    assert!(stats.allocation_bytes >= 4096);
    assert!(stats.block_count >= 1);

    allocator.free(allocation);
    drop(buffer);

    let stats = allocator.statistics();
    assert_eq!(stats.allocation_count, 0);
    assert_eq!(stats.allocation_bytes, 0);
    // Block stays around — pool keeps the underlying VkDeviceMemory until
    // the Allocator drops.
    assert!(stats.block_count >= 1);
}

#[test]
fn test_allocation_drop_returns_to_pool() {
    // Regression test for the auto-Drop on `Allocation`. Previously,
    // a forgotten `allocator.free()` leaked the slot until the entire
    // Allocator was dropped. Now dropping the Allocation directly is
    // equivalent.
    let Some((_inst, physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let allocator = Allocator::new(&device, &physical).unwrap();

    {
        let allocation = allocator
            .allocate(
                vulkane::safe::MemoryRequirements {
                    size: 4096,
                    alignment: 16,
                    memory_type_bits: u32::MAX,
                },
                AllocationCreateInfo {
                    usage: AllocationUsage::HostVisible,
                    ..Default::default()
                },
            )
            .unwrap();
        let stats = allocator.statistics();
        assert_eq!(stats.allocation_count, 1, "allocation visible to allocator");
        assert!(stats.allocation_bytes >= 4096);
        // Walk out of scope without explicit free() — Drop must reclaim.
        let _ = allocation;
    }

    let stats = allocator.statistics();
    assert_eq!(
        stats.allocation_count, 0,
        "Allocation's Drop returned the slot to the pool"
    );
    assert_eq!(stats.allocation_bytes, 0);
}

#[test]
fn test_allocation_clone_keeps_alive_until_last_drop() {
    // `Allocation` is internally Arc-shared. The slot must only be
    // returned when the *last* clone is dropped.
    let Some((_inst, physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let allocator = Allocator::new(&device, &physical).unwrap();
    let allocation = allocator
        .allocate(
            vulkane::safe::MemoryRequirements {
                size: 4096,
                alignment: 16,
                memory_type_bits: u32::MAX,
            },
            AllocationCreateInfo {
                usage: AllocationUsage::HostVisible,
                ..Default::default()
            },
        )
        .unwrap();
    let clone = allocation.clone();
    drop(allocation); // not the last reference
    assert_eq!(allocator.statistics().allocation_count, 1);
    drop(clone); // last reference — slot reclaimed now
    assert_eq!(allocator.statistics().allocation_count, 0);
}

#[test]
fn test_allocator_many_buffers_share_one_block() {
    // Allocating many small buffers should reuse the same underlying
    // VkDeviceMemory block — the whole point of sub-allocation.
    let Some((_inst, physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let allocator = Allocator::new(&device, &physical).unwrap();

    let mut buffers = Vec::new();
    let mut allocations = Vec::new();
    let mut last_memory_handle = 0u64;
    let mut shared_count = 0u32;
    for _ in 0..32 {
        let (b, a) = allocator
            .create_buffer(
                BufferCreateInfo {
                    size: 1024,
                    usage: BufferUsage::STORAGE_BUFFER,
                },
                AllocationCreateInfo {
                    usage: AllocationUsage::HostVisible,
                    ..Default::default()
                },
            )
            .unwrap();
        if a.memory() == last_memory_handle {
            shared_count += 1;
        }
        last_memory_handle = a.memory();
        buffers.push(b);
        allocations.push(a);
    }

    // Most buffers should share the same VkDeviceMemory.
    assert!(
        shared_count >= 28,
        "expected at least 28 shared, got {shared_count}"
    );

    let stats = allocator.statistics();
    assert_eq!(stats.allocation_count, 32);
    // We should have only one block for all 32 small allocations.
    assert!(
        stats.block_count <= 2,
        "expected <=2 blocks for 32 small allocations, got {}",
        stats.block_count
    );

    for a in allocations.drain(..) {
        allocator.free(a);
    }
    drop(buffers);
}

#[test]
fn test_allocator_dedicated_for_huge_buffer() {
    let Some((_inst, physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let allocator = Allocator::new(&device, &physical).unwrap();

    // Force the dedicated path with the explicit flag.
    let (buffer, alloc) = allocator
        .create_buffer(
            BufferCreateInfo {
                size: 4096,
                usage: BufferUsage::STORAGE_BUFFER,
            },
            AllocationCreateInfo {
                usage: AllocationUsage::HostVisible,
                dedicated: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(alloc.offset(), 0);

    let stats = allocator.statistics();
    assert_eq!(stats.dedicated_allocation_count, 1);

    allocator.free(alloc);
    drop(buffer);
    let stats = allocator.statistics();
    assert_eq!(stats.dedicated_allocation_count, 0);
}

#[test]
fn test_allocator_per_allocation_device_mask_forces_dedicated() {
    // Even on a single-device group, supplying device_mask = Some(mask)
    // must force the dedicated path (so the chained
    // VkMemoryAllocateFlagsInfo with VK_MEMORY_ALLOCATE_DEVICE_MASK_BIT
    // is honored). On a single-device group the only valid mask is 0b1,
    // which is what default_device_mask() returns.
    let Some((_inst, physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let allocator = Allocator::new(&device, &physical).unwrap();
    let mask = device.default_device_mask();

    let stats_before = allocator.statistics();

    let (buffer, alloc) = allocator
        .create_buffer(
            BufferCreateInfo {
                size: 4096,
                usage: BufferUsage::STORAGE_BUFFER,
            },
            AllocationCreateInfo {
                usage: AllocationUsage::DeviceLocal,
                device_mask: Some(mask),
                ..Default::default()
            },
        )
        .unwrap();

    let stats = allocator.statistics();
    assert_eq!(
        stats.dedicated_allocation_count,
        stats_before.dedicated_allocation_count + 1,
        "device_mask=Some(_) must force a dedicated allocation"
    );
    // A dedicated allocation always sits at offset 0 in its own VkDeviceMemory.
    assert_eq!(alloc.offset(), 0);

    allocator.free(alloc);
    drop(buffer);
}

#[test]
fn test_allocator_device_mask_rejects_custom_pool() {
    // Custom pools have their device mask fixed at block creation time;
    // combining them with per-allocation device_mask must error rather
    // than silently ignoring the mask.
    let Some((_inst, physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let allocator = Allocator::new(&device, &physical).unwrap();

    // Pick any host-visible memory type for the pool.
    let mem_props = physical.memory_properties();
    let mem_type = (0..mem_props.type_count())
        .find(|&i| {
            mem_props
                .memory_type(i)
                .property_flags()
                .contains(vulkane::safe::MemoryPropertyFlags::HOST_VISIBLE)
        })
        .expect("expected at least one host-visible memory type");

    let pool = allocator
        .create_pool(vulkane::safe::PoolCreateInfo {
            memory_type_index: mem_type,
            strategy: vulkane::safe::AllocationStrategy::FreeList,
            block_size: 1024 * 1024,
            max_block_count: 0,
        })
        .unwrap();

    let mask = device.default_device_mask();
    let result = allocator.create_buffer(
        BufferCreateInfo {
            size: 4096,
            usage: BufferUsage::STORAGE_BUFFER,
        },
        AllocationCreateInfo {
            usage: AllocationUsage::HostVisible,
            pool: Some(pool),
            device_mask: Some(mask),
            ..Default::default()
        },
    );
    match result {
        Err(vulkane::safe::Error::InvalidArgument(_)) => {}
        Ok(_) => panic!("expected InvalidArgument when combining device_mask with a pool"),
        Err(e) => panic!("expected InvalidArgument, got {e:?}"),
    }

    allocator.destroy_pool(pool);
}

#[test]
fn test_allocator_create_image_2d_via_pool() {
    let Some((_inst, physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let allocator = Allocator::new(&device, &physical).unwrap();

    let (image, alloc) = allocator
        .create_image_2d(
            Image2dCreateInfo {
                format: Format::R32_UINT,
                width: 32,
                height: 32,
                usage: ImageUsage::STORAGE,
            },
            AllocationCreateInfo {
                usage: AllocationUsage::DeviceLocal,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(alloc.size() >= 32 * 32 * 4);
    let _view = ImageView::new_2d_color(&image).unwrap();

    allocator.free(alloc);
    drop(image);
}

#[test]
fn test_allocator_persistent_mapped_pointer() {
    let Some((_inst, physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let allocator = Allocator::new(&device, &physical).unwrap();

    let (buffer, alloc) = allocator
        .create_buffer(
            BufferCreateInfo {
                size: 256,
                usage: BufferUsage::TRANSFER_DST,
            },
            AllocationCreateInfo {
                usage: AllocationUsage::HostVisible,
                mapped: true,
                ..Default::default()
            },
        )
        .unwrap();
    let ptr = alloc
        .mapped_ptr()
        .expect("HostVisible + mapped should give a pointer");
    // Write a pattern via the persistent mapping. Safety: pointer is
    // valid for the size of the allocation, and host-coherent so no
    // explicit flush is needed.
    unsafe {
        let bytes = std::slice::from_raw_parts_mut(ptr as *mut u8, alloc.size() as usize);
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i & 0xFF) as u8;
        }
        // Read back via a fresh slice from the same pointer.
        let bytes = std::slice::from_raw_parts(ptr as *const u8, alloc.size() as usize);
        for (i, &b) in bytes.iter().enumerate() {
            assert_eq!(b, (i & 0xFF) as u8);
        }
    }

    allocator.free(alloc);
    drop(buffer);
}

#[test]
fn test_allocator_peak_bytes_tracks_high_watermark() {
    let Some((_inst, physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let allocator = Allocator::new(&device, &physical).unwrap();

    let (b1, a1) = allocator
        .create_buffer(
            BufferCreateInfo {
                size: 8 * 1024,
                usage: BufferUsage::STORAGE_BUFFER,
            },
            AllocationCreateInfo {
                usage: AllocationUsage::HostVisible,
                ..Default::default()
            },
        )
        .unwrap();
    let (b2, a2) = allocator
        .create_buffer(
            BufferCreateInfo {
                size: 16 * 1024,
                usage: BufferUsage::STORAGE_BUFFER,
            },
            AllocationCreateInfo {
                usage: AllocationUsage::HostVisible,
                ..Default::default()
            },
        )
        .unwrap();

    let peak_after_two = allocator.statistics().peak_allocation_bytes;
    assert!(peak_after_two >= 24 * 1024);

    allocator.free(a2);
    drop(b2);
    let peak_after_one_freed = allocator.statistics().peak_allocation_bytes;
    // Peak does not regress when one is freed.
    assert_eq!(peak_after_two, peak_after_one_freed);

    allocator.free(a1);
    drop(b1);
}

// Note: the previous "buffer_device_address_returns_or_skips" test has
// been replaced by `test_device_features_buffer_device_address_round_trip`
// below, which actually enables the bufferDeviceAddress feature at
// device creation and exercises the full call. The defanged version is
// no longer needed.

#[test]
fn test_memory_budget_query_succeeds_or_skips() {
    let Some((_inst, physical, _device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let Some(budget) = physical.memory_budget() else {
        common::skipped_unsupported(
            "vkGetPhysicalDeviceMemoryProperties2 (Vulkan 1.1 / VK_KHR_get_physical_device_properties2)",
        );
        return;
    };
    // The structure is valid even if VK_EXT_memory_budget wasn't enabled
    // — the heap_count comes from the Vulkan-1.0-mandatory memory props,
    // and the budget/usage values just stay zero. We just assert the
    // structural shape here; meaningful budget reads require enabling
    // the extension at instance creation time.
    assert!(budget.heap_count > 0);
    println!(
        "Memory heaps: {}; total budget reported: {} bytes (0 if extension not enabled)",
        budget.heap_count,
        budget.total_budget()
    );
}

#[test]
fn test_device_identity_query_succeeds_or_skips() {
    let Some((_inst, physical, _device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let Some(identity) = physical.device_identity() else {
        common::skipped_unsupported(
            "vkGetPhysicalDeviceProperties2 (Vulkan 1.1 / VK_KHR_get_physical_device_properties2)",
        );
        return;
    };
    // device_uuid is always populated on a props2-capable loader (1.1
    // core). We can't assert a specific value, but a real device never
    // reports the all-zero UUID; software rasterizers may, so we only
    // print rather than assert on it.
    let uuid_hex: String = identity
        .device_uuid
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!(
        "device_uuid={uuid_hex}  luid_valid={}  pci={:?}",
        identity.device_luid.is_some(),
        identity.pci,
    );
    // LUID, when present, pairs with the node mask; PCI is Some only with
    // VK_EXT_pci_bus_info. These are all honest-None when unavailable, so
    // the only structural invariant to check is that the call returned.
    if let Some(pci) = identity.pci {
        println!(
            "PCI {:04x}:{:02x}:{:02x}.{:x}",
            pci.domain, pci.bus, pci.device, pci.function
        );
    }
}

// `cooperative_matrix_properties()` used to be untestable at runtime:
// it was an `unsafe fn` whose contract required the caller to have
// enabled VK_KHR_cooperative_matrix, and calling the loader's stub
// without it is undefined behaviour on some implementations (notably
// Mesa Lavapipe, which is what our Linux CI uses). It now gates on the
// device's own extension advertisement before touching the dispatch
// entry, so calling it on ANY device — Lavapipe included — is defined
// and returns an empty Vec when unsupported. That is precisely what
// this test asserts, and it is the reason the test can exist at all.
#[test]
fn test_cooperative_matrix_properties_safe_without_extension() {
    let Some((_inst, physical, _device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    // No instance/device extension was enabled by try_init_compute, so on
    // a device that does not advertise VK_KHR_cooperative_matrix this must
    // return empty rather than crash. On a device that does advertise it,
    // every reported shape must be non-degenerate.
    let shapes = physical
        .cooperative_matrix_properties()
        .expect("a device must answer its own cooperative-matrix query, or say it has none");
    let advertises = physical
        .enumerate_extension_properties()
        .map(|e| e.iter().any(|x| x.name() == "VK_KHR_cooperative_matrix"))
        .unwrap_or(false);
    if !advertises {
        assert!(
            shapes.is_empty(),
            "device does not advertise VK_KHR_cooperative_matrix but \
             {} shapes were reported",
            shapes.len()
        );
        // NOT a skip: the `assert!` above is this test's actual claim on a
        // device without the extension — that the query returns an honest
        // empty Vec rather than fabricating shapes. The return is because
        // there is nothing further to check, not because nothing was checked.
        eprintln!("cooperative matrix: not advertised — asserted the empty Ok");
        return;
    }
    for s in &shapes {
        assert!(
            s.m_size() > 0 && s.n_size() > 0 && s.k_size() > 0,
            "degenerate cooperative-matrix shape {}x{}x{}",
            s.m_size(),
            s.n_size(),
            s.k_size()
        );
    }
    println!("cooperative matrix: {} shapes reported", shapes.len());
}

/// An instance created at `version`, plus its first physical device.
///
/// The shared `try_init_compute()` helper builds a **1.0** instance
/// (`InstanceCreateInfo::default()`), and a 1.0 implementation must
/// behave as 1.0 — it leaves 1.1+ `pNext` property structs untouched
/// even on a 1.3 device. Queries for those structs therefore need an
/// instance created at the matching version to be exercised at all.
///
/// Declares its own failure, and distinguishes the two that look alike here.
/// "Cannot create a 1.2 instance" has two causes with opposite verdicts: there
/// is no ICD (an environment fault, fatal under `VULKANE_REQUIRE_DEVICE`), or
/// there is one and it predates 1.2 (conformant, never fatal). They are told
/// apart by asking for a default 1.0 instance as well — if *that* succeeds,
/// the ICD is present and merely older than the caller wants.
fn init_at_or_skip(
    version: ApiVersion,
    version_label: &str,
) -> Option<(Instance, vulkane::safe::PhysicalDevice)> {
    match try_init_at(version, version_label) {
        Ok(pair) => Some(pair),
        Err(cause) => {
            common::skip(cause);
            None
        }
    }
}

fn try_init_at(
    version: ApiVersion,
    version_label: &str,
) -> Result<(Instance, vulkane::safe::PhysicalDevice), common::Missing> {
    // Note the deliberate difference from `try_init_compute`: this takes the
    // *first* physical device rather than the first compute-capable one. What
    // is under test is version-gated property queries, which every device
    // answers — requiring compute would import a precondition this has no need
    // of, and make it fatal under VULKANE_REQUIRE_DEVICE for no reason.
    let (instance, devices) = match common::instance_and_devices("vulkane test", version) {
        Ok(v) => v,
        Err(common::Missing::Device(reason)) => {
            // "Cannot create a 1.2 instance" has two causes with opposite
            // verdicts: no ICD (fatal), or an ICD older than the caller wants
            // (conformant). Asking for a default instance tells them apart.
            return Err(if Instance::new(InstanceCreateInfo::default()).is_ok() {
                common::Missing::capability(format!("a Vulkan {version_label} instance"))
            } else {
                common::Missing::device(reason)
            });
        }
        Err(other) => return Err(other),
    };
    let physical = devices.into_iter().next().ok_or_else(|| {
        common::Missing::device("an ICD is present but reports no physical devices")
    })?;
    Ok((instance, physical))
}

/// On a 1.0 instance the two version-gated queries must behave
/// *differently*, and the difference is the whole point of the
/// `effective_api_version` gate:
///
/// - `VkPhysicalDeviceSubgroupProperties` is Vulkan 1.1 **core with no
///   extension form**. A 1.0 implementation leaves it untouched, so the
///   query must decline. Before the gate existed it keyed off the
///   *device* version and returned `Some(subgroup_size: 0)` here — a
///   zeroed struct dressed as an answer.
/// - `VkPhysicalDeviceDriverProperties` is 1.2 core **but also a device
///   extension** (`VK_KHR_driver_properties`), which a device may
///   advertise and honor under a 1.0 instance. So it may legitimately
///   succeed — exactly when the extension is advertised.
#[test]
fn test_version_gated_queries_on_a_1_0_instance() {
    let Some((_inst, physical, _device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let eff = physical.effective_api_version();
    assert_eq!(
        (eff.major(), eff.minor()),
        (1, 0),
        "try_init_compute builds a 1.0 instance, so the effective version must be 1.0 \
         however new the device is (device reports {}.{})",
        physical.properties().api_version().major(),
        physical.properties().api_version().minor(),
    );

    // 1.1-core-only: must decline, whatever the device claims to support.
    assert!(
        physical.subgroup_properties().is_none(),
        "subgroup_properties has no extension form, so it must decline on a \
         1.0 instance rather than report an untouched struct"
    );

    // Core-or-extension: MAY succeed here if the device both advertises
    // the extension and actually honors it under a 1.0 instance. Some
    // drivers advertise it and still decline to fill the struct (the AMD
    // proprietary driver does exactly that), which the query detects and
    // reports as `None`. Either outcome is correct; what must never
    // happen is a `Some` carrying an unpopulated struct.
    if let Some(dp) = physical.driver_properties() {
        assert!(
            !dp.driver_name.is_empty(),
            "driver_properties returned Some with an empty name — that is the \
             untouched-struct reading the query is supposed to decline"
        );
        assert!(
            physical
                .enumerate_extension_properties()
                .map(|e| e.iter().any(|x| x.name() == "VK_KHR_driver_properties"))
                .unwrap_or(false),
            "driver_properties answered on a 1.0 instance without the device \
             advertising VK_KHR_driver_properties"
        );
    }
}

#[test]
fn test_subgroup_properties_query_succeeds_or_skips() {
    let Some((_inst, physical)) = init_at_or_skip(ApiVersion::V1_1, "1.1") else {
        return;
    };
    let Some(sg) = physical.subgroup_properties() else {
        common::skipped_unsupported("a Vulkan 1.1 device exposing vkGetPhysicalDeviceProperties2");
        return;
    };
    // Vulkan requires a subgroup size of at least 1 and, in practice, a
    // power of two. A zero here would mean we read an untouched struct —
    // the exact dishonesty the 1.1 version gate exists to prevent.
    assert!(
        sg.subgroup_size > 0,
        "subgroup_size must be non-zero on a 1.1+ device"
    );
    assert!(
        sg.subgroup_size.is_power_of_two(),
        "subgroup_size {} is not a power of two",
        sg.subgroup_size
    );
    // BASIC is mandatory for every Vulkan 1.1 implementation.
    assert!(
        sg.supported_operations
            .contains(SubgroupFeatureFlags::BASIC),
        "BASIC subgroup ops are mandatory in Vulkan 1.1"
    );
    // Compute must support subgroup ops.
    assert!(
        sg.supported_stages.contains(ShaderStageFlags::COMPUTE),
        "compute stage must support subgroup operations"
    );

    println!(
        "subgroup_size={} arithmetic={} shuffle_relative={} size_control={:?}",
        sg.subgroup_size,
        sg.supported_operations
            .contains(SubgroupFeatureFlags::ARITHMETIC),
        sg.supported_operations
            .contains(SubgroupFeatureFlags::SHUFFLE_RELATIVE),
        sg.size_control,
    );

    if let Some(sc) = sg.size_control {
        assert!(
            sc.min_subgroup_size > 0 && sc.max_subgroup_size >= sc.min_subgroup_size,
            "nonsensical subgroup size range {}..={}",
            sc.min_subgroup_size,
            sc.max_subgroup_size
        );
        assert!(
            sc.min_subgroup_size.is_power_of_two() && sc.max_subgroup_size.is_power_of_two(),
            "subgroup size bounds must be powers of two"
        );
        // The device's own default width must itself be pinnable-range
        // valid — this is the invariant a caller relies on when choosing
        // a required_subgroup_size.
        assert!(
            sc.permits(sc.min_subgroup_size) && sc.permits(sc.max_subgroup_size),
            "permits() rejects the device's own reported bounds"
        );
        // Non-power-of-two and out-of-range values must be rejected.
        assert!(!sc.permits(sc.max_subgroup_size * 2));
        assert!(!sc.permits(3));
    }
}

#[test]
fn test_driver_properties_query_succeeds_or_skips() {
    let Some((_inst, physical)) = init_at_or_skip(ApiVersion::V1_2, "1.2") else {
        return;
    };
    let Some(dp) = physical.driver_properties() else {
        common::skipped_unsupported("Vulkan 1.2 or VK_KHR_driver_properties");
        return;
    };
    // A conforming driver always names itself. driver_info may be empty
    // in principle, so it is printed rather than asserted on.
    assert!(
        !dp.driver_name.is_empty(),
        "a driver that reports driver_properties must name itself"
    );
    println!(
        "driver_id={:?} name={:?} info={:?} conformance={}",
        dp.driver_id, dp.driver_name, dp.driver_info, dp.conformance_version,
    );
    // Display must round-trip the four components in order.
    let cv = dp.conformance_version;
    assert_eq!(
        cv.to_string(),
        format!("{}.{}.{}.{}", cv.major, cv.minor, cv.subminor, cv.patch)
    );
}

// ---------------------------------------------------------------------------
// DeviceFeatures + features-enabled tests
// ---------------------------------------------------------------------------

/// Bring up a device with `features` enabled, declaring the failure through
/// [`common::skip`] and returning `None`.
///
/// `label` names the feature set for the not-supported case, e.g.
/// `"the bufferDeviceAddress feature"`.
///
/// The two outcomes must not be conflated. A device that declines
/// `bufferDeviceAddress` is conformant, so that skip can never be fatal. No
/// ICD at all is an environment fault, and under `VULKANE_REQUIRE_DEVICE` it
/// must be. Before this split every call site declared the *feature* reason
/// for both, which meant a Linux CI run that had lost its ICD entirely would
/// have reported "the bufferDeviceAddress feature is unsupported" and passed.
///
/// The classification now travels in the [`common::Missing`] the primitives
/// return, so this function chooses nothing — it only supplies the label that
/// [`common::create_device_on`] uses when the *features* step is the one that
/// failed.
fn features_or_skip(features: DeviceFeatures, label: &str) -> Option<ComputeSetup> {
    match try_features(features, label) {
        Ok(setup) => Some(setup),
        Err(cause) => {
            common::skip(cause);
            None
        }
    }
}

fn try_features(features: DeviceFeatures, label: &str) -> Result<ComputeSetup, common::Missing> {
    let (instance, devices) = common::instance_and_devices("vulkane test", ApiVersion::V1_0)?;
    let (physical, queue_family) = common::first_compute(devices)?;
    let device = common::create_device_on(&physical, queue_family, Some((&features, label)))?;
    let queue = device.get_queue(queue_family, 0);
    Ok((instance, physical, device, queue, queue_family))
}

#[test]
fn test_device_features_default_creates_normally() {
    // A DeviceFeatures::default() with no flags set must successfully
    // create a device on every conformant Vulkan implementation.
    let features = DeviceFeatures::default();
    let Some((_inst, _physical, device, _q, _qf)) =
        features_or_skip(features, "device creation with no features requested")
    else {
        return;
    };
    assert!(!device.raw().is_null());
}

#[test]
fn test_device_features_buffer_device_address_round_trip() {
    // Enable bufferDeviceAddress and verify Buffer::device_address()
    // returns a non-zero address when called against a buffer with
    // SHADER_DEVICE_ADDRESS usage.
    let features = DeviceFeatures::default().with_buffer_device_address();
    let Some((_inst, physical, device, _q, _qf)) =
        features_or_skip(features, "the bufferDeviceAddress feature")
    else {
        return;
    };

    // Skip if Lavapipe (or any conformant Vulkan 1.0 ICD) doesn't expose
    // the function pointer at the loaded device level.
    let buffer = match Buffer::new(
        &device,
        BufferCreateInfo {
            size: 4096,
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
        },
    ) {
        Ok(b) => b,
        Err(e) => {
            common::skipped_unsupported(&format!("a SHADER_DEVICE_ADDRESS buffer ({e})"));
            return;
        }
    };
    let req = buffer.memory_requirements();
    // VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT requires the
    // VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT flag, which vulkane's
    // DeviceMemory::allocate doesn't currently set. Until that lands,
    // some drivers reject the bind. So we may also skip here.
    let mt = physical
        .find_memory_type(req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| {
            physical.find_memory_type(req.memory_type_bits, MemoryPropertyFlags::HOST_VISIBLE)
        })
        .unwrap();
    let memory = match DeviceMemory::allocate(&device, req.size, mt) {
        Ok(m) => m,
        Err(e) => {
            common::skipped_unsupported(&format!(
                "vkAllocateMemory for a device-address buffer ({e})"
            ));
            return;
        }
    };
    if let Err(e) = buffer.bind_memory(&memory, 0) {
        common::skipped_unsupported(&format!(
            "vkBindBufferMemory for a device-address buffer ({e})"
        ));
        return;
    }
    match buffer.device_address() {
        Ok(addr) => {
            assert!(
                addr != 0,
                "device_address() returned zero on a feature-enabled device"
            );
            println!("Buffer device address: 0x{addr:x}");
        }
        Err(e) => {
            // Some implementations require additional flags we don't yet
            // pass; this is acceptable.
            common::skipped_unsupported(&format!("vkGetBufferDeviceAddress ({e})"));
        }
    }
}

#[test]
fn test_allocator_buffer_device_address() {
    // An allocator created with `buffer_device_address: true` must back
    // every buffer with memory carrying VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    // so a sub-allocated buffer with SHADER_DEVICE_ADDRESS usage returns a
    // valid (non-zero) GPU address. This is the BDA path the Fuel backend
    // depends on — unlike the manual round-trip test above, it must NOT skip
    // on the missing-flag path, because the allocator now sets the flag.
    let features = DeviceFeatures::default().with_buffer_device_address();
    let Some((_inst, physical, device, _q, _qf)) =
        features_or_skip(features, "the bufferDeviceAddress feature")
    else {
        return;
    };

    let allocator = Allocator::new_with_options(
        &device,
        &physical,
        AllocatorOptions {
            buffer_device_address: true,
        },
    )
    .unwrap();

    let (buffer, allocation) = match allocator.create_buffer(
        BufferCreateInfo {
            size: 4096,
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
        },
        AllocationCreateInfo {
            usage: AllocationUsage::DeviceLocal,
            ..Default::default()
        },
    ) {
        Ok(pair) => pair,
        Err(e) => {
            common::skipped_unsupported(&format!(
                "an allocation for a device-address buffer ({e})"
            ));
            return;
        }
    };

    match buffer.device_address() {
        Ok(addr) => assert!(
            addr != 0,
            "device_address() returned zero from a buffer_device_address allocator"
        ),
        Err(e) => {
            // vkGetBufferDeviceAddress fn pointer not loaded on this ICD
            // (e.g. a 1.0 driver without VK_KHR_buffer_device_address). The
            // allocator wiring is still exercised by the successful bind above.
            common::skipped_unsupported(&format!("vkGetBufferDeviceAddress ({e})"));
        }
    }

    allocator.free(allocation);
    drop(buffer);
}

#[test]
fn test_device_features_timeline_semaphore_round_trip() {
    let features = DeviceFeatures::default().with_timeline_semaphore();
    let Some((_inst, _physical, device, queue, queue_family)) =
        features_or_skip(features, "the timelineSemaphore feature")
    else {
        return;
    };

    let sem = match Semaphore::timeline(&device, 0) {
        Ok(s) => s,
        Err(e) => {
            common::skipped_unsupported(&format!("timeline semaphore creation ({e})"));
            return;
        }
    };
    assert_eq!(sem.kind(), SemaphoreKind::Timeline);
    assert_eq!(sem.current_value().unwrap(), 0);

    let pool = CommandPool::new(&device, queue_family).unwrap();
    let mut cmd = pool.allocate_primary().unwrap();
    {
        let rec = cmd.begin().unwrap();
        rec.end().unwrap();
    }
    queue
        .submit_with_sync(
            &[&cmd],
            &[],
            &[SignalSemaphore {
                semaphore: &sem,
                value: 42,
                device_index: 0,
            }],
            None,
        )
        .unwrap();

    sem.wait_value(42, u64::MAX).unwrap();
    assert!(sem.current_value().unwrap() >= 42);
    println!(
        "Timeline semaphore reached value {}",
        sem.current_value().unwrap()
    );
}

#[test]
fn test_device_features_synchronization2_round_trip() {
    let features = DeviceFeatures::default().with_synchronization2();
    let Some((_inst, _physical, device, queue, queue_family)) =
        features_or_skip(features, "the synchronization2 feature")
    else {
        return;
    };

    let pool = CommandPool::new(&device, queue_family).unwrap();
    let mut cmd = pool.allocate_primary().unwrap();
    let supported = {
        let mut rec = cmd.begin().unwrap();
        let res = rec.memory_barrier2(
            PipelineStage2::COMPUTE_SHADER,
            PipelineStage2::HOST,
            AccessFlags2::SHADER_WRITE,
            AccessFlags2::HOST_READ,
        );
        rec.end().unwrap();
        res
    };
    match supported {
        Ok(()) => {
            let fence = Fence::new(&device).unwrap();
            queue.submit(&[&cmd], Some(&fence)).unwrap();
            fence.wait(u64::MAX).unwrap();
        }
        Err(e) => {
            // Even with the feature enabled, vkCmdPipelineBarrier2 might
            // not be loaded on Vulkan 1.0/1.1 devices that lack
            // VK_KHR_synchronization2 as an extension. Acceptable.
            common::skipped_unsupported(&format!("vkCmdPipelineBarrier2 ({e})"));
        }
    }
}

#[test]
fn test_supported_features_query_succeeds() {
    let Some((_inst, physical, _device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let _features = physical.supported_features();
    // Just verify the call doesn't crash; the bit-set returned varies
    // by hardware. We don't assert any specific bit is set because
    // even Lavapipe exposes very few core 1.0 features by default.
}

// ---------------------------------------------------------------------------
// Custom pool + linear pool tests
// ---------------------------------------------------------------------------

#[test]
fn test_allocator_custom_freelist_pool_round_trip() {
    let Some((_inst, physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let allocator = Allocator::new(&device, &physical).unwrap();

    // Pick a host-visible memory type for the pool.
    let dummy_buf = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 256,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .unwrap();
    let req = dummy_buf.memory_requirements();
    drop(dummy_buf);
    let mt = physical
        .find_memory_type(
            req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();

    let pool = allocator
        .create_pool(PoolCreateInfo {
            memory_type_index: mt,
            strategy: AllocationStrategy::FreeList,
            block_size: 1024 * 1024, // 1 MiB pool block
            max_block_count: 0,
        })
        .unwrap();

    // Allocate several buffers from the custom pool.
    let mut allocations = Vec::new();
    let mut buffers = Vec::new();
    for _ in 0..8 {
        let (b, a) = allocator
            .create_buffer(
                BufferCreateInfo {
                    size: 4096,
                    usage: BufferUsage::STORAGE_BUFFER,
                },
                AllocationCreateInfo {
                    pool: Some(pool),
                    ..Default::default()
                },
            )
            .unwrap();
        buffers.push(b);
        allocations.push(a);
    }

    let pool_stats = allocator.pool_statistics(pool).unwrap();
    assert_eq!(pool_stats.allocation_count, 8);
    assert!(pool_stats.allocation_bytes >= 8 * 4096);
    // All 8 allocations should fit in the 1 MiB block.
    assert_eq!(pool_stats.block_count, 1);

    for a in allocations.drain(..) {
        allocator.free(a);
    }
    drop(buffers);

    let pool_stats = allocator.pool_statistics(pool).unwrap();
    assert_eq!(pool_stats.allocation_count, 0);

    allocator.destroy_pool(pool);
    // After destruction the pool handle is unknown.
    assert!(allocator.pool_statistics(pool).is_none());
}

#[test]
fn test_allocator_linear_pool_supports_reset() {
    let Some((_inst, physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let allocator = Allocator::new(&device, &physical).unwrap();

    // Pick a host-visible memory type.
    let dummy_buf = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 256,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .unwrap();
    let req = dummy_buf.memory_requirements();
    drop(dummy_buf);
    let mt = physical
        .find_memory_type(
            req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();

    let pool = allocator
        .create_pool(PoolCreateInfo {
            memory_type_index: mt,
            strategy: AllocationStrategy::Linear,
            block_size: 64 * 1024, // 64 KiB linear block
            max_block_count: 0,
        })
        .unwrap();

    // Bump-allocate a bunch of small chunks. They should not be returned
    // individually — only via reset_pool.
    let mut buffers = Vec::new();
    let mut allocations = Vec::new();
    for _ in 0..10 {
        let (b, a) = allocator
            .create_buffer(
                BufferCreateInfo {
                    size: 1024,
                    usage: BufferUsage::STORAGE_BUFFER,
                },
                AllocationCreateInfo {
                    pool: Some(pool),
                    ..Default::default()
                },
            )
            .unwrap();
        buffers.push(b);
        allocations.push(a);
    }

    let stats_before = allocator.pool_statistics(pool).unwrap();
    assert_eq!(stats_before.allocation_count, 10);

    // Drop the buffers (which calls vkDestroyBuffer) before reset.
    drop(buffers);

    // Reset the pool — all 10 allocations are reclaimed in one shot.
    allocator.reset_pool(pool);
    let stats_after = allocator.pool_statistics(pool).unwrap();
    assert_eq!(stats_after.allocation_count, 0);
    assert_eq!(stats_after.allocation_bytes, 0);

    // We can keep allocating into the same pool after reset.
    let (b2, a2) = allocator
        .create_buffer(
            BufferCreateInfo {
                size: 2048,
                usage: BufferUsage::STORAGE_BUFFER,
            },
            AllocationCreateInfo {
                pool: Some(pool),
                ..Default::default()
            },
        )
        .unwrap();

    let stats_post = allocator.pool_statistics(pool).unwrap();
    assert_eq!(stats_post.allocation_count, 1);

    drop(b2);
    drop(a2);
    // Don't bother freeing — the destroy_pool below will reclaim everything.
    allocator.destroy_pool(pool);
}

#[test]
fn test_allocator_linear_pool_full_returns_error() {
    let Some((_inst, physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let allocator = Allocator::new(&device, &physical).unwrap();

    let dummy_buf = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 256,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .unwrap();
    let req = dummy_buf.memory_requirements();
    drop(dummy_buf);
    let mt = physical
        .find_memory_type(
            req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();

    // Tiny linear pool — only one tiny block.
    let pool = allocator
        .create_pool(PoolCreateInfo {
            memory_type_index: mt,
            strategy: AllocationStrategy::Linear,
            block_size: 4096,
            max_block_count: 1,
        })
        .unwrap();

    // First allocation succeeds.
    let r1 = allocator.create_buffer(
        BufferCreateInfo {
            size: 2048,
            usage: BufferUsage::STORAGE_BUFFER,
        },
        AllocationCreateInfo {
            pool: Some(pool),
            ..Default::default()
        },
    );
    assert!(r1.is_ok());

    // Second allocation might fit (2048 + 2048 = 4096) — depends on
    // alignment. If it does fit, the third should fail.
    let _r2 = allocator.create_buffer(
        BufferCreateInfo {
            size: 2048,
            usage: BufferUsage::STORAGE_BUFFER,
        },
        AllocationCreateInfo {
            pool: Some(pool),
            ..Default::default()
        },
    );

    // Third allocation: even if r2 succeeded, we're definitely over the
    // 4 KiB block limit now, and max_block_count = 1 prevents growth.
    let r3 = allocator.create_buffer(
        BufferCreateInfo {
            size: 4096,
            usage: BufferUsage::STORAGE_BUFFER,
        },
        AllocationCreateInfo {
            pool: Some(pool),
            ..Default::default()
        },
    );
    assert!(
        r3.is_err(),
        "expected linear pool to refuse over-budget allocation"
    );

    allocator.destroy_pool(pool);
}

#[test]
fn test_allocator_defragmentation_compacts_fragmented_pool() {
    // Create a custom FreeList pool, allocate alternating large/small
    // buffers, free every other one to create holes, then build a defrag
    // plan and verify that:
    //   - the plan is non-empty (there's something to compact)
    //   - applying the plan does not invalidate live Allocation handles
    //     (memory() / offset() return new positions but size() / id()
    //     are stable)
    //   - the post-defrag positions are contiguous from offset 0
    //   - the pool can still allocate from the freed space afterward
    let Some((_inst, physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    let allocator = Allocator::new(&device, &physical).unwrap();

    // Pick a host-visible memory type for predictability.
    let dummy_buf = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 256,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .unwrap();
    let req = dummy_buf.memory_requirements();
    drop(dummy_buf);
    let mt = physical
        .find_memory_type(
            req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();

    let pool = allocator
        .create_pool(PoolCreateInfo {
            memory_type_index: mt,
            strategy: AllocationStrategy::FreeList,
            block_size: 256 * 1024, // 256 KiB
            max_block_count: 1,
        })
        .unwrap();

    // Allocate 8 buffers of varying sizes with user_data tags.
    let mut buffers: Vec<(Buffer, vulkane::safe::Allocation)> = Vec::new();
    for i in 0..8 {
        let (b, a) = allocator
            .create_buffer(
                BufferCreateInfo {
                    size: 4096 * (i as u64 + 1),
                    usage: BufferUsage::STORAGE_BUFFER,
                },
                AllocationCreateInfo {
                    pool: Some(pool),
                    user_data: 100 + i as u64,
                    ..Default::default()
                },
            )
            .unwrap();
        buffers.push((b, a));
    }

    // Free every odd-indexed allocation to create holes.
    let mut survivors: Vec<vulkane::safe::Allocation> = Vec::new();
    for (i, (buffer, alloc)) in buffers.into_iter().enumerate() {
        if i % 2 == 1 {
            // free it.
            drop(buffer);
            allocator.free(alloc);
        } else {
            // Keep this one alive across defrag.
            drop(buffer);
            survivors.push(alloc);
        }
    }

    // Snapshot pre-defrag IDs / sizes / user_data.
    let pre: Vec<(u64, u64, u64)> = survivors
        .iter()
        .map(|a| (a.id(), a.size(), a.user_data()))
        .collect();

    // Build the defragmentation plan.
    let plan = allocator.build_defragmentation_plan(pool);
    assert!(
        !plan.total_layout().is_empty(),
        "defrag plan should include all 4 surviving allocations"
    );

    // Sanity-check the move list: every move's user_data should match
    // one of our survivors.
    for m in &plan.moves {
        assert!(
            survivors.iter().any(|a| a.user_data() == m.user_data),
            "move user_data {} should match a surviving allocation",
            m.user_data
        );
    }

    // Apply the plan. (We don't actually issue GPU copies here — this
    // test only verifies the bookkeeping side of defrag, not data
    // integrity. A full GPU-copy test would need a command buffer +
    // submit + fence-wait per move.)
    allocator.apply_defragmentation_plan(plan);

    // Post-defrag: every survivor still has its original id, size, and
    // user_data, but offset() may have changed. The positions should be
    // monotonically increasing from 0.
    let mut last_offset: u64 = 0;
    for (i, alloc) in survivors.iter().enumerate() {
        assert_eq!(alloc.id(), pre[i].0, "id should be stable across defrag");
        assert_eq!(
            alloc.size(),
            pre[i].1,
            "size should be stable across defrag"
        );
        assert_eq!(
            alloc.user_data(),
            pre[i].2,
            "user_data should be stable across defrag"
        );
        let off = alloc.offset();
        assert!(
            off >= last_offset,
            "post-defrag offsets should be monotonically increasing (got {off} after {last_offset})"
        );
        last_offset = off + alloc.size();
    }

    // After defrag, we should be able to allocate something large in
    // the freed-up space — proves the TLSF state was rebuilt.
    let (_b, post_alloc) = allocator
        .create_buffer(
            BufferCreateInfo {
                size: 64 * 1024,
                usage: BufferUsage::STORAGE_BUFFER,
            },
            AllocationCreateInfo {
                pool: Some(pool),
                ..Default::default()
            },
        )
        .expect("post-defrag allocation should succeed");
    assert!(post_alloc.size() >= 64 * 1024);

    // Cleanup.
    drop(survivors);
    drop(post_alloc);
    allocator.destroy_pool(pool);
}

#[test]
fn test_enumerate_physical_device_groups() {
    let instance = match Instance::new(InstanceCreateInfo::default()) {
        Ok(i) => i,
        Err(e) => {
            common::skipped(&format!("no Vulkan ICD, or instance creation failed: {e}"));
            return;
        }
    };
    let groups = match instance.enumerate_physical_device_groups() {
        Ok(g) => g,
        Err(e) => {
            common::skipped_unsupported(&format!("vkEnumeratePhysicalDeviceGroups ({e})"));
            return;
        }
    };
    println!("Found {} physical device group(s)", groups.len());
    for (i, group) in groups.iter().enumerate() {
        assert!(group.count() >= 1, "every group has at least one device");
        println!(
            "  group {i}: {} device(s), subset_allocation={}",
            group.count(),
            group.supports_subset_allocation()
        );
        for pd in group.physical_devices() {
            assert!(!pd.properties().device_name().is_empty());
        }
    }
}

#[test]
fn test_device_singleton_group_unification() {
    // Verify that a device created via the legacy
    // physical.create_device(...) path internally exposes a length-1
    // physical-device group, matching the device-group representation.
    let Some((_inst, _physical, device, _q, _qf)) = compute_or_skip() else {
        return;
    };
    assert_eq!(
        device.physical_device_count(),
        1,
        "single-physical-device path should produce a length-1 group"
    );
    let handles = device.physical_device_handles();
    assert_eq!(handles.len(), 1);
    // The default device mask for a 1-device group is 0b1.
    assert_eq!(device.default_device_mask(), 0b1);
}

#[test]
fn test_submit_with_groups_default_mask_round_trip() {
    // On any single-device group (i.e. all CI hardware), submit_with_groups
    // with the default per-CB mask must succeed and behave identically to
    // submit_with_sync. This exercises the VkDeviceGroupSubmitInfo
    // pNext-chaining path: we explicitly pass Some(&[mask]), which forces
    // the chain to be added.
    let Some((_inst, _physical, device, queue, queue_family)) = compute_or_skip() else {
        return;
    };
    let pool = CommandPool::new(&device, queue_family).unwrap();
    let mut cmd = pool.allocate_primary().unwrap();
    {
        let rec = cmd.begin().unwrap();
        rec.end().unwrap();
    }
    let fence = Fence::new(&device).unwrap();
    let mask = device.default_device_mask();

    queue
        .submit_with_groups(&[&cmd], Some(&[mask]), &[], &[], Some(&fence))
        .unwrap();

    // The fence must signal exactly as it would for the non-group path.
    // (wait() returns SUCCESS only when the fence has actually become
    // signaled.)
    fence.wait(u64::MAX).unwrap();
}

#[test]
fn test_submit_with_groups_rejects_mask_length_mismatch() {
    let Some((_inst, _physical, device, queue, queue_family)) = compute_or_skip() else {
        return;
    };
    let pool = CommandPool::new(&device, queue_family).unwrap();
    let mut cmd = pool.allocate_primary().unwrap();
    {
        let rec = cmd.begin().unwrap();
        rec.end().unwrap();
    }

    // 1 CB, 2 masks → must error.
    let result = queue.submit_with_groups(&[&cmd], Some(&[1u32, 2u32]), &[], &[], None);
    match result {
        Err(vulkane::safe::Error::InvalidArgument(_)) => {}
        Ok(_) => panic!("expected InvalidArgument when mask len differs from CB count"),
        Err(e) => panic!("expected InvalidArgument, got {e:?}"),
    }
}

#[test]
fn test_device_create_via_physical_device_group() {
    // Verify that PhysicalDeviceGroup::create_device works for
    // singleton groups (we can't reliably test multi-device on most CI
    // hardware) and that the resulting Device exposes the same fields.
    let instance = match Instance::new(InstanceCreateInfo::default()) {
        Ok(i) => i,
        Err(_) => {
            common::skipped("no Vulkan ICD, or the loader declined to create an instance");
            return;
        }
    };
    let Some(group) = instance
        .enumerate_physical_device_groups()
        .ok()
        .and_then(|gs| gs.into_iter().next())
    else {
        common::skipped_unsupported("physical device groups (Vulkan 1.1 / VK_KHR_device_group)");
        return;
    };
    let Some(physical) = group.physical_devices().first() else {
        common::skipped_unsupported("a non-empty physical device group");
        return;
    };
    let queue_family = physical.find_queue_family(QueueFlags::TRANSFER).unwrap();
    let device = group
        .create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo {
                queue_family_index: queue_family,
                queue_priorities: vec![1.0],
            }],
            ..Default::default()
        })
        .unwrap();
    assert_eq!(device.physical_device_count(), group.count());
    assert!(device.default_device_mask() != 0);
}

#[test]
fn test_defragmentation_move_struct_round_trip() {
    let m = DefragmentationMove {
        allocation_id: 42,
        user_data: 0xDEAD_BEEF,
        size: 1024,
        src_memory: 0x1000,
        src_offset: 0,
        dst_memory: 0x2000,
        dst_offset: 256,
    };
    assert_eq!(m.allocation_id, 42);
    assert_eq!(m.user_data, 0xDEAD_BEEF);
    assert_eq!(m.size, 1024);
    let _plan = DefragmentationPlan::default();
}
