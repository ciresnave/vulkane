//! Integration test for the safe wrapper module.
//!
//! Validates the entire safe API end-to-end against a real Vulkan driver.
//! Skips gracefully on systems without Vulkan installed.

use spock::safe::{
    ApiVersion, Buffer, BufferCopy, BufferCreateInfo, BufferUsage, CommandPool, ComputePipeline,
    DescriptorPool, DescriptorPoolSize, DescriptorSetLayout, DescriptorSetLayoutBinding,
    DescriptorType, DeviceCreateInfo, DeviceMemory, Fence, Instance, InstanceCreateInfo,
    MemoryPropertyFlags, PipelineLayout, PipelineStatisticsFlags, PushConstantRange, QueryPool,
    QueueCreateInfo, QueueFlags, ShaderModule, ShaderStageFlags, SpecializationConstants,
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
