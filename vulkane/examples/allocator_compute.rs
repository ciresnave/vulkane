// SPDX-License-Identifier: MIT OR Apache-2.0
//! Same workflow as `compute_square` but using the [`Allocator`] for
//! buffer creation — 2 lines instead of 5 per buffer.
//!
//! Demonstrates that the recommended allocation path is dramatically
//! simpler than manual `DeviceMemory::allocate` + `bind_memory`:
//!
//! ```ignore
//! let (buffer, alloc) = allocator.create_buffer(
//!     BufferCreateInfo { size: 1024, usage: BufferUsage::STORAGE_BUFFER },
//!     AllocationCreateInfo { usage: AllocationUsage::HostVisible, mapped: true, ..Default::default() },
//! )?;
//! ```
//!
//! Run with: `cargo run -p vulkane --features fetch-spec --example allocator_compute`

use vulkane::safe::{
    AccessFlags, AllocationCreateInfo, AllocationUsage, Allocator, ApiVersion, BufferCreateInfo,
    BufferUsage, CommandPool, ComputePipeline, DescriptorPool, DescriptorPoolSize,
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorType, DeviceCreateInfo, Fence,
    Instance, InstanceCreateInfo, PipelineLayout, PipelineStage, QueueCreateInfo, QueueFlags,
    ShaderModule, ShaderStageFlags,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let spv_path = format!("{manifest_dir}/examples/shaders/square_buffer.spv");
    let spv_bytes = std::fs::read(&spv_path).map_err(|e| {
        format!("could not read {spv_path}: {e} (run compile_shader example first)")
    })?;
    println!("[OK] Loaded {} bytes of SPIR-V", spv_bytes.len());

    let instance = Instance::new(InstanceCreateInfo {
        application_name: Some("vulkane allocator_compute"),
        api_version: ApiVersion::V1_0,
        ..Default::default()
    })?;
    let physical = instance
        .enumerate_physical_devices()?
        .into_iter()
        .find(|pd| pd.find_queue_family(QueueFlags::COMPUTE).is_some())
        .ok_or("no GPU with a compute queue")?;
    println!("[OK] Using GPU: {}", physical.properties().device_name());

    let qf = physical.find_queue_family(QueueFlags::COMPUTE).unwrap();
    let device = physical.create_device(DeviceCreateInfo {
        queue_create_infos: &[QueueCreateInfo::single(qf)],
        ..Default::default()
    })?;
    let queue = device.get_queue(qf, 0);

    // Create the allocator — this is the VMA-style sub-allocator.
    let allocator = Allocator::new(&device, &physical)?;

    // Allocate a host-visible, persistently-mapped storage buffer.
    // This replaces the 5-step: Buffer::new → memory_requirements →
    // find_memory_type → DeviceMemory::allocate → bind_memory.
    const N: usize = 256;
    const SIZE: u64 = (N * 4) as u64;
    let (buffer, alloc) = allocator.create_buffer(
        BufferCreateInfo {
            size: SIZE,
            usage: BufferUsage::STORAGE_BUFFER,
        },
        AllocationCreateInfo {
            usage: AllocationUsage::HostVisible,
            mapped: true,
            ..Default::default()
        },
    )?;
    println!(
        "[OK] Allocated {} bytes via Allocator (offset={}, size={})",
        SIZE,
        alloc.offset(),
        alloc.size()
    );

    // Write initial data through the persistent mapping.
    let ptr = alloc
        .mapped_ptr()
        .ok_or("allocation was not persistently mapped")?;
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u32, N) };
    for (i, slot) in slice.iter_mut().enumerate() {
        *slot = i as u32;
    }
    println!("[OK] Wrote 0..{N} to the buffer via persistent mapping");

    // Pipeline setup (same as compute_square).
    let shader = ShaderModule::from_spirv_bytes(&device, &spv_bytes)?;
    let set_layout = DescriptorSetLayout::new(
        &device,
        &[DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
        }],
    )?;
    let layout = PipelineLayout::new(&device, &[&set_layout])?;
    let pipeline = ComputePipeline::new(&device, &layout, &shader, "main")?;

    let pool = DescriptorPool::new(
        &device,
        1,
        &[DescriptorPoolSize {
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        }],
    )?;
    let desc_set = pool.allocate(&set_layout)?;
    desc_set.write_buffer(0, DescriptorType::STORAGE_BUFFER, &buffer, 0, SIZE);

    // Record + submit.
    let cmd_pool = CommandPool::new(&device, qf)?;
    let mut cmd = cmd_pool.allocate_primary()?;
    {
        let mut rec = cmd.begin()?;
        rec.bind_compute_pipeline(&pipeline);
        rec.bind_compute_descriptor_sets(&layout, 0, &[&desc_set]);
        rec.dispatch((N as u32).div_ceil(64), 1, 1);
        rec.memory_barrier(
            PipelineStage::COMPUTE_SHADER,
            PipelineStage::HOST,
            AccessFlags::SHADER_WRITE,
            AccessFlags::HOST_READ,
        );
        rec.end()?;
    }
    let fence = Fence::new(&device)?;
    queue.submit(&[&cmd], Some(&fence))?;
    fence.wait(u64::MAX)?;
    println!("[OK] GPU finished compute dispatch");

    // Verify results through the same persistent mapping.
    for (i, &got) in slice.iter().enumerate() {
        let expected = (i as u32) * (i as u32);
        assert_eq!(got, expected, "element {i}: got {got}, expected {expected}");
    }
    println!("[OK] Verified all {N} elements were squared");

    // Cleanup: free the allocation (returns the sub-allocation to
    // the TLSF pool), drop the buffer, let the allocator clean up
    // the rest.
    allocator.free(alloc);
    drop(buffer);

    let stats = allocator.statistics();
    println!(
        "[OK] Allocator stats: {} block(s), {} bytes allocated",
        stats.block_count, stats.block_bytes
    );

    device.wait_idle()?;
    println!("\n=== allocator_compute example PASSED ===");
    Ok(())
}
