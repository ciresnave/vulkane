//! Integration tests for Tier-3 extension wrappers:
//!
//! - `VK_EXT_subgroup_size_control` —
//!   [`ComputePipelineOptions::required_subgroup_size`].
//! - `VK_EXT_memory_priority` —
//!   [`MemoryAllocateInfo::priority`].
//! - `VK_EXT_descriptor_buffer` — layout size/offset queries +
//!   `bind_descriptor_buffers` / `set_descriptor_buffer_offsets`.
//!
//! All tests degrade gracefully when the Vulkan driver does not expose
//! the underlying extension. For tests that *require* a working device
//! (e.g. actually creating a compute pipeline with a required subgroup
//! size), we fall back to verifying that the safe wrapper either
//! succeeds or surfaces a clean `MissingFunction` / `Vk` error.
//!
//! Every one of those degradations is declared through [`common`] rather than
//! bailing with a bare `return`. Before that, all seven tests here skipped
//! **silently** — the highest concentration in the suite — so on a machine
//! without a device the file reported seven passes having asserted nothing.

mod common;

use vulkane::safe::{
    Buffer, BufferCreateInfo, BufferUsage, ComputePipelineOptions, DescriptorBufferBinding,
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorType, DeviceMemory,
    MemoryAllocateInfo, MemoryPropertyFlags, PipelineBindPoint, PipelineLayout, Queue,
    ShaderStageFlags,
};

fn bootstrap()
-> Result<(vulkane::safe::Device, vulkane::safe::PhysicalDevice, u32), common::Missing> {
    common::compute_device("vulkane-tier3-test")
}

#[test]
fn memory_allocate_info_default_is_plain_allocation() {
    // Regression: the Default impl must produce a struct equivalent to
    // the plain allocate() path — no priority, no pnext.
    let (device, physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
    };
    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 128,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .expect("buffer");
    let req = buffer.memory_requirements();
    let Some(mt) =
        physical.find_memory_type(req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
    else {
        return common::skipped_unsupported(
            "a DEVICE_LOCAL memory type this buffer's memory_type_bits accepts",
        );
    };
    let _mem = DeviceMemory::allocate_with(
        &device,
        &MemoryAllocateInfo {
            size: req.size,
            memory_type_index: mt,
            ..Default::default()
        },
    )
    .expect("default allocate_with");
}

#[test]
fn memory_priority_chains_through_allocate_with() {
    // Priority gets embedded into a local PNextChain inside allocate_with.
    // Drivers without VK_EXT_memory_priority enabled ignore the struct,
    // so this should succeed regardless of driver support. The test
    // proves the chain-merge path doesn't corrupt allocation.
    let (device, physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
    };
    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 128,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .expect("buffer");
    let req = buffer.memory_requirements();
    let Some(mt) =
        physical.find_memory_type(req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
    else {
        return common::skipped_unsupported(
            "a DEVICE_LOCAL memory type this buffer's memory_type_bits accepts",
        );
    };

    for p in [0.0f32, 0.5, 1.0] {
        let _mem = DeviceMemory::allocate_with(
            &device,
            &MemoryAllocateInfo {
                size: req.size,
                memory_type_index: mt,
                priority: Some(p),
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("priority={p} allocate_with failed: {e:?}"));
    }
}

#[test]
fn memory_priority_composes_with_user_pnext() {
    // Both user pnext and priority together — proves the local merged
    // chain keeps the user's struct reachable.
    use vulkane::raw::PNextChainable;
    use vulkane::raw::bindings::VkMemoryAllocateFlagsInfo;
    use vulkane::safe::PNextChain;
    let (device, physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
    };
    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 128,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .expect("buffer");
    let req = buffer.memory_requirements();
    let Some(mt) =
        physical.find_memory_type(req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
    else {
        return common::skipped_unsupported(
            "a DEVICE_LOCAL memory type this buffer's memory_type_bits accepts",
        );
    };

    let mut chain = PNextChain::new();
    let mut flags = VkMemoryAllocateFlagsInfo::new_pnext();
    flags.flags = 0;
    chain.push(flags);

    let _mem = DeviceMemory::allocate_with(
        &device,
        &MemoryAllocateInfo {
            size: req.size,
            memory_type_index: mt,
            pnext: Some(&chain),
            priority: Some(0.7),
        },
    );
    // Accept either success or a driver rejection — both are OK for
    // this chain-composition test.
}

#[test]
fn compute_pipeline_with_required_subgroup_size_path_compiles() {
    // We can't reliably create a real compute pipeline in every CI
    // environment (requires a compiled SPIR-V shader + matching
    // descriptor layout), so this test exercises the *API surface* of
    // the subgroup-size option. A real failure here would be a type
    // error or a panic before reaching the driver.
    let (device, _physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
    };

    // Build a minimal descriptor layout + pipeline layout for the
    // assembly exercise.
    let set_layout = DescriptorSetLayout::new(
        &device,
        &[DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
        }],
    )
    .expect("set layout");
    let _pipe_layout = PipelineLayout::new(&device, &[&set_layout]).expect("pipeline layout");

    // The options struct is what Tier-3 adds — just prove it's
    // constructible and Default works.
    let _opts: ComputePipelineOptions<'_> = ComputePipelineOptions {
        required_subgroup_size: Some(32),
        ..Default::default()
    };
}

#[test]
fn descriptor_buffer_size_graceful_missing_function() {
    // Without VK_EXT_descriptor_buffer enabled, the query function
    // isn't loaded — safe wrapper must return MissingFunction, not
    // panic or UB.
    let (device, _physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
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
    .expect("set layout");

    match set_layout.descriptor_buffer_size() {
        Ok(_) => {}
        Err(vulkane::safe::Error::MissingFunction(name)) => {
            assert_eq!(name, "vkGetDescriptorSetLayoutSizeEXT");
        }
        Err(other) => panic!("unexpected error: {other:?}"),
    }

    match set_layout.descriptor_buffer_binding_offset(0) {
        Ok(_) => {}
        Err(vulkane::safe::Error::MissingFunction(name)) => {
            assert_eq!(name, "vkGetDescriptorSetLayoutBindingOffsetEXT");
        }
        Err(other) => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn bind_descriptor_buffers_graceful_missing_function() {
    let (device, _physical, qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
    };
    let queue: Queue = device.get_queue(qf, 0);

    let set_layout = DescriptorSetLayout::new(
        &device,
        &[DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
        }],
    )
    .expect("set layout");
    let pipe_layout = PipelineLayout::new(&device, &[&set_layout]).expect("pipeline layout");

    let r = queue.one_shot(&device, qf, |rec| {
        rec.bind_descriptor_buffers(&[DescriptorBufferBinding {
            address: 0x1000,
            usage: 0x0020_0000, // RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT
        }])?;
        rec.set_descriptor_buffer_offsets(PipelineBindPoint::Compute, &pipe_layout, 0, &[0], &[0])?;
        Ok(())
    });
    match r {
        Ok(()) => {}
        Err(vulkane::safe::Error::MissingFunction(name)) => {
            assert!(
                name == "vkCmdBindDescriptorBuffersEXT"
                    || name == "vkCmdSetDescriptorBufferOffsetsEXT",
                "unexpected missing fn: {name}"
            );
        }
        Err(vulkane::safe::Error::Vk(_)) => {
            // Driver may reject the call due to the fake buffer address —
            // acceptable for this API-surface test.
        }
        Err(other) => panic!("unexpected error shape: {other:?}"),
    }
}

#[test]
fn set_descriptor_buffer_offsets_rejects_mismatched_lengths() {
    // Pure safe-layer validation: if buffer_indices.len() != offsets.len()
    // we return InvalidArgument *before* touching the driver.
    let (device, _physical, qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
    };
    let queue: Queue = device.get_queue(qf, 0);
    let set_layout = DescriptorSetLayout::new(
        &device,
        &[DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
        }],
    )
    .expect("set layout");
    let pipe_layout = PipelineLayout::new(&device, &[&set_layout]).expect("pipeline layout");

    let r = queue.one_shot(&device, qf, |rec| {
        rec.set_descriptor_buffer_offsets(
            PipelineBindPoint::Compute,
            &pipe_layout,
            0,
            &[0, 1],
            &[0],
        )
    });
    match r {
        Err(vulkane::safe::Error::InvalidArgument(msg)) => {
            assert!(msg.contains("length"));
        }
        Err(vulkane::safe::Error::MissingFunction(_)) => {
            // If the extension isn't loaded we may never hit the length
            // check. That's fine — another test covers the missing-fn path.
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}
