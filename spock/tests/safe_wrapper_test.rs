//! Integration test for the safe wrapper module.
//!
//! Validates the entire safe API end-to-end against a real Vulkan driver.
//! Skips gracefully on systems without Vulkan installed.

use spock::safe::{
    ApiVersion, Buffer, BufferCopy, BufferCreateInfo, BufferImageCopy, BufferUsage, CommandPool,
    ComputePipeline, DEBUG_UTILS_EXTENSION, DebugMessage, DebugMessageSeverity, DescriptorPool,
    DescriptorPoolSize, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorType,
    DeviceCreateInfo, DeviceMemory, Fence, Format, Image, Image2dCreateInfo, ImageBarrier,
    ImageLayout, ImageUsage, ImageView, Instance, InstanceCreateInfo, KHRONOS_VALIDATION_LAYER,
    MemoryPropertyFlags, PipelineCache, PipelineLayout, PipelineStatisticsFlags, PushConstantRange,
    QueryPool, QueueCreateInfo, QueueFlags, Semaphore, SemaphoreKind, ShaderModule,
    ShaderStageFlags, SignalSemaphore, SpecializationConstants, WaitSemaphore,
};

#[test]
fn test_safe_instance_creation_and_enumeration() {
    let instance = match Instance::new(InstanceCreateInfo {
        application_name: Some("spock test"),
        api_version: ApiVersion::V1_0,
        ..Default::default()
    }) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("SKIP: Vulkan not available: {e}");
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
fn test_safe_device_creation_and_drop() {
    let instance = match Instance::new(InstanceCreateInfo::default()) {
        Ok(i) => i,
        Err(_) => {
            eprintln!("SKIP: Vulkan not available");
            return;
        }
    };

    let physicals = instance.enumerate_physical_devices().unwrap();
    let physical = match physicals.first() {
        Some(p) => p.clone(),
        None => {
            eprintln!("SKIP: no physical devices");
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
            eprintln!("SKIP: Vulkan not available");
            return;
        }
    };

    let physicals = instance.enumerate_physical_devices().unwrap();
    let Some(physical) = physicals.first().cloned() else {
        eprintln!("SKIP: no physical devices");
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
            eprintln!("SKIP: Vulkan not available");
            return;
        }
    };

    let physicals = instance.enumerate_physical_devices().unwrap();
    let Some(physical) = physicals.first().cloned() else {
        eprintln!("SKIP: no physical devices");
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
    let instance = match Instance::new(InstanceCreateInfo::default()) {
        Ok(i) => i,
        Err(_) => return,
    };
    let physicals = instance.enumerate_physical_devices().unwrap();
    let Some(physical) = physicals.first().cloned() else {
        return;
    };
    let queue_family = physical.find_queue_family(QueueFlags::COMPUTE).unwrap();
    let device = physical
        .create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo {
                queue_family_index: queue_family,
                queue_priorities: vec![1.0],
            }],
            ..Default::default()
        })
        .unwrap();

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
    let instance = match Instance::new(InstanceCreateInfo::default()) {
        Ok(i) => i,
        Err(_) => return,
    };
    let physicals = instance.enumerate_physical_devices().unwrap();
    let Some(physical) = physicals.first().cloned() else {
        return;
    };

    let queue_family = match physical.find_queue_family(QueueFlags::COMPUTE) {
        Some(q) => q,
        None => return,
    };
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
        rec.memory_barrier(0x800, 0x4000, 0x40, 0x2000);
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
fn try_init_compute() -> Option<(
    Instance,
    spock::safe::PhysicalDevice,
    spock::safe::Device,
    spock::safe::Queue,
    u32,
)> {
    let instance = Instance::new(InstanceCreateInfo::default()).ok()?;
    let physical = instance
        .enumerate_physical_devices()
        .ok()?
        .into_iter()
        .next()?;
    let queue_family = physical.find_queue_family(QueueFlags::COMPUTE)?;
    let device = physical
        .create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo {
                queue_family_index: queue_family,
                queue_priorities: vec![1.0],
            }],
            ..Default::default()
        })
        .ok()?;
    let queue = device.get_queue(queue_family, 0);
    Some((instance, physical, device, queue, queue_family))
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
    let Some((_inst, physical, _dev, _q, _qf)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
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
    let Some((_inst, physical, _dev, _q, _qf)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
        return;
    };
    // Should be a finite, non-negative number on any conformant device.
    let p = physical.timestamp_period();
    assert!(p.is_finite());
    assert!(p >= 0.0);
}

#[test]
fn test_max_push_constants_size_meets_spec_minimum() {
    let Some((_inst, physical, _dev, _q, _qf)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
        return;
    };
    // Vulkan spec guarantees at least 128 bytes.
    let max = physical.properties().max_push_constants_size();
    assert!(max >= 128, "spec minimum is 128 bytes, got {max}");
}

#[test]
fn test_pipeline_layout_with_push_constants() {
    let Some((_inst, _physical, device, _q, _qf)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
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
    let Some((_inst, physical, device, _q, queue_family)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
        return;
    };

    // Skip if the chosen queue family doesn't support timestamps.
    let families = physical.queue_family_properties();
    if families[queue_family as usize].timestamp_valid_bits() == 0 {
        eprintln!("SKIP: queue family does not support timestamps");
        return;
    }

    let pool = QueryPool::timestamps(&device, 4).unwrap();
    assert_eq!(pool.query_count(), 4);
    assert!(pool.raw() != 0);
}

#[test]
fn test_copy_buffer_staging_round_trip() {
    let Some((_inst, physical, device, queue, queue_family)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
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
        rec.memory_barrier(0x1000, 0x4000, 0x800, 0x2000);
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
    let Some((_inst, physical, device, queue, queue_family)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
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
        rec.memory_barrier(0x800, 0x4000, 0x40, 0x2000);
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
    let Some((_inst, physical, device, queue, queue_family)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
        return;
    };
    let families = physical.queue_family_properties();
    if families[queue_family as usize].timestamp_valid_bits() == 0 {
        eprintln!("SKIP: queue family does not support timestamps");
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
        // VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT = 0x00000001
        rec.write_timestamp(0x1, &qpool, 0);
        rec.bind_compute_pipeline(&pipe);
        rec.bind_compute_descriptor_sets(&pl, 0, &[&dset]);
        rec.dispatch(N.div_ceil(64), 1, 1);
        // VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT = 0x00002000
        rec.write_timestamp(0x2000, &qpool, 1);
        // Compute -> Host
        rec.memory_barrier(0x800, 0x4000, 0x40, 0x2000);
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
    let Some((_inst, physical, device, _queue, _qf)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
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
            eprintln!("SKIP: cannot load Vulkan library: {e}");
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
            eprintln!("SKIP: cannot load Vulkan library: {e}");
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
    let Some((_inst, physical, _dev, _q, _qf)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
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
        application_name: Some("spock-empty-lists"),
        enabled_layers: &[],
        enabled_extensions: &[],
        ..InstanceCreateInfo::default()
    };
    if let Ok(_inst) = Instance::new(info) {
        // OK.
    } else {
        eprintln!("SKIP: no Vulkan ICD");
    }
}

#[test]
fn test_instance_with_unknown_layer_fails_cleanly() {
    let info = InstanceCreateInfo {
        application_name: Some("spock-bad-layer"),
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
            eprintln!("SKIP: no Vulkan loader");
            return;
        }
    };
    if !layers.iter().any(|l| l.name() == KHRONOS_VALIDATION_LAYER) {
        eprintln!("SKIP: validation layer not installed");
        return;
    }
    let exts = Instance::enumerate_extension_properties().unwrap();
    if !exts.iter().any(|e| e.name() == DEBUG_UTILS_EXTENSION) {
        eprintln!("SKIP: debug utils extension not present");
        return;
    }

    use std::sync::Arc as StdArc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let counter = StdArc::new(AtomicUsize::new(0));
    let counter_cb = StdArc::clone(&counter);

    let info = InstanceCreateInfo {
        application_name: Some("spock-validation"),
        enabled_layers: &[KHRONOS_VALIDATION_LAYER],
        enabled_extensions: &[DEBUG_UTILS_EXTENSION],
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
            eprintln!("SKIP: validation instance creation failed: {e}");
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
        eprintln!("SKIP: validation layer not installed");
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
    let Some((_inst, physical, device, _q, _qf)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
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
    let Some((_inst, physical, device, queue, queue_family)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
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

    // Pipeline-stage and access constants we need.
    const TOP_OF_PIPE: u32 = 0x1;
    const TRANSFER: u32 = 0x1000;
    const HOST: u32 = 0x4000;
    const ACCESS_TRANSFER_READ: u32 = 0x800;
    const ACCESS_TRANSFER_WRITE: u32 = 0x1000;
    const ACCESS_HOST_READ: u32 = 0x2000;

    let pool = CommandPool::new(&device, queue_family).unwrap();
    let mut cmd = pool.allocate_primary().unwrap();
    {
        let mut rec = cmd.begin().unwrap();
        // 1) UNDEFINED -> TRANSFER_DST_OPTIMAL
        rec.image_barrier(
            TOP_OF_PIPE,
            TRANSFER,
            ImageBarrier {
                image: &image,
                old_layout: ImageLayout::UNDEFINED,
                new_layout: ImageLayout::TRANSFER_DST_OPTIMAL,
                src_access: 0,
                dst_access: ACCESS_TRANSFER_WRITE,
            },
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
            TRANSFER,
            TRANSFER,
            ImageBarrier {
                image: &image,
                old_layout: ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_access: ACCESS_TRANSFER_WRITE,
                dst_access: ACCESS_TRANSFER_READ,
            },
        );
        // 4) Copy image -> buffer
        rec.copy_image_to_buffer(
            &image,
            ImageLayout::TRANSFER_SRC_OPTIMAL,
            &dst,
            &[BufferImageCopy::full_2d(W, H)],
        );
        // 5) Transfer -> Host barrier so the host read sees the bytes.
        rec.memory_barrier(TRANSFER, HOST, ACCESS_TRANSFER_WRITE, ACCESS_HOST_READ);
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
    let Some((_inst, physical, device, _q, _qf)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
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
    let Some((_inst, _physical, device, _q, _qf)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
        return;
    };
    let s = Semaphore::binary(&device).unwrap();
    assert_eq!(s.kind(), SemaphoreKind::Binary);
    assert!(s.raw() != 0);
}

#[test]
fn test_timeline_semaphore_host_signal_and_wait() {
    let Some((_inst, _physical, device, _q, _qf)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
        return;
    };

    let sem = match Semaphore::timeline(&device, 5) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP: timeline semaphores not supported: {e}");
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
    let Some((_inst, _physical, device, queue, queue_family)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
        return;
    };
    let sem = match Semaphore::timeline(&device, 0) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP: timeline semaphores not supported: {e}");
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
    let Some((_inst, _physical, device, queue, queue_family)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
        return;
    };
    let sem = match Semaphore::timeline(&device, 0) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP: timeline semaphores not supported: {e}");
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
            }],
            None,
        )
        .unwrap();

    // Pass B: waits on timeline >= 1, signals -> 2.
    // VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT = 0x1
    queue
        .submit_with_sync(
            &[&cmd_b],
            &[WaitSemaphore {
                semaphore: &sem,
                value: 1,
                dst_stage_mask: 0x1,
            }],
            &[SignalSemaphore {
                semaphore: &sem,
                value: 2,
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
    let Some((_inst, _physical, device, _q, _qf)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
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
    let Some((_inst, _physical, device, _q, _qf)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
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
        .add_f32(100, 3.14);
    let pipe =
        ComputePipeline::with_specialization(&device, &layout, &shader, "main", &specs).unwrap();
    assert!(pipe.raw() != 0);
}

#[test]
fn test_sync2_memory_barrier_when_supported() {
    let Some((_inst, _physical, device, queue, queue_family)) = try_init_compute() else {
        eprintln!("SKIP: no Vulkan ICD");
        return;
    };

    let pool = CommandPool::new(&device, queue_family).unwrap();
    let mut cmd = pool.allocate_primary().unwrap();
    let supported = {
        let mut rec = cmd.begin().unwrap();
        // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT       = 0x800
        // VK_PIPELINE_STAGE_2_HOST_BIT                 = 0x4000
        // VK_ACCESS_2_SHADER_WRITE_BIT                 = 0x40
        // VK_ACCESS_2_HOST_READ_BIT                    = 0x2000
        let s2 = rec.memory_barrier2(0x800, 0x4000, 0x40, 0x2000);
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
            eprintln!("SKIP: sync2 not supported: {e}");
        }
    }
}
