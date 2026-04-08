//! End-to-end compute shader example using the spock safe wrapper.
//!
//! This example:
//!
//! 1. Loads the pre-compiled `square_buffer.spv` shader from disk.
//! 2. Creates an instance, picks a compute-capable physical device, and
//!    creates a logical device with one compute queue.
//! 3. Allocates a host-visible storage buffer of 256 `u32`s.
//! 4. Writes the integers 0..256 into the buffer from the host.
//! 5. Builds a descriptor set layout, pool, set, and pipeline layout.
//! 6. Creates a compute pipeline from the shader and pipeline layout.
//! 7. Records a command buffer that binds the pipeline + descriptor set
//!    and dispatches `ceil(256/64) = 4` workgroups.
//! 8. Submits, waits on a fence, and verifies that every element was
//!    squared by the GPU.
//! 9. Drops everything via RAII.
//!
//! Run with: `cargo run --example compute_square -p spock --features fetch-spec`
//!
//! The shader source is in `examples/shaders/square_buffer.comp` and the
//! pre-compiled SPIR-V is checked in alongside it. To regenerate the SPIR-V
//! after editing the GLSL, run:
//!
//!   cargo run -p spock --features naga,fetch-spec --example compile_shader

use spock::safe::{
    ApiVersion, Buffer, BufferCreateInfo, BufferUsage, CommandPool, ComputePipeline,
    DescriptorPool, DescriptorPoolSize, DescriptorSetLayout, DescriptorSetLayoutBinding,
    DescriptorType, DeviceCreateInfo, DeviceMemory, Fence, Instance, InstanceCreateInfo,
    MemoryPropertyFlags, PipelineLayout, QueueCreateInfo, QueueFlags, ShaderModule,
    ShaderStageFlags,
};

const ELEMENT_COUNT: u32 = 256;
const BUFFER_SIZE: u64 = (ELEMENT_COUNT as u64) * 4; // 256 * sizeof(u32)
const WORKGROUP_SIZE: u32 = 64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // -------------------------------------------------------------------
    // 1. Load the pre-compiled SPIR-V shader
    // -------------------------------------------------------------------
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let spv_path = format!("{manifest_dir}/examples/shaders/square_buffer.spv");
    let spv_bytes = match std::fs::read(&spv_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ERROR: could not read pre-compiled shader at {spv_path}: {e}");
            eprintln!(
                "Run `cargo run -p spock --features naga,fetch-spec --example compile_shader` to regenerate it."
            );
            return Err(e.into());
        }
    };
    println!("[OK] Loaded {} bytes of SPIR-V", spv_bytes.len());

    // -------------------------------------------------------------------
    // 2. Instance + physical device + logical device + queue
    // -------------------------------------------------------------------
    let instance = match Instance::new(InstanceCreateInfo {
        application_name: Some("spock compute_square example"),
        api_version: ApiVersion::V1_0,
        ..Default::default()
    }) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("SKIP: could not create Vulkan instance: {e}");
            eprintln!("(Install a Vulkan driver such as Lavapipe to run this example.)");
            return Ok(());
        }
    };
    println!("[OK] Created VkInstance");

    let physical = instance
        .enumerate_physical_devices()?
        .into_iter()
        .find(|pd| pd.find_queue_family(QueueFlags::COMPUTE).is_some())
        .ok_or("No physical device with a compute-capable queue family")?;
    println!("[OK] Using GPU: {}", physical.properties().device_name());

    let queue_family_index = physical.find_queue_family(QueueFlags::COMPUTE).unwrap();
    let device = physical.create_device(DeviceCreateInfo {
        queue_create_infos: &[QueueCreateInfo {
            queue_family_index,
            queue_priorities: vec![1.0],
        }],
    })?;
    let queue = device.get_queue(queue_family_index, 0);
    println!("[OK] Created VkDevice with compute queue");

    // -------------------------------------------------------------------
    // 3. Allocate a host-visible storage buffer
    // -------------------------------------------------------------------
    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: BUFFER_SIZE,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )?;

    let req = buffer.memory_requirements();
    let mem_type = physical
        .find_memory_type(
            req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or("No host-visible+coherent memory type available")?;
    let mut memory = DeviceMemory::allocate(&device, req.size, mem_type)?;
    buffer.bind_memory(&memory, 0)?;
    println!(
        "[OK] Allocated and bound {} bytes of storage buffer memory",
        req.size
    );

    // -------------------------------------------------------------------
    // 4. Write the integers 0..256 into the buffer
    // -------------------------------------------------------------------
    {
        let mut mapped = memory.map()?;
        let bytes = mapped.as_slice_mut();
        for i in 0..ELEMENT_COUNT as usize {
            let v = i as u32;
            bytes[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        println!("[OK] Wrote 0..{ELEMENT_COUNT} to the buffer from the host");
    }

    // -------------------------------------------------------------------
    // 5. Shader module + descriptor set layout + pool + set
    // -------------------------------------------------------------------
    let shader = ShaderModule::from_spirv_bytes(&device, &spv_bytes)?;
    println!("[OK] Created VkShaderModule from SPIR-V");

    let set_layout = DescriptorSetLayout::new(
        &device,
        &[DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
        }],
    )?;

    let pool = DescriptorPool::new(
        &device,
        1, // max_sets
        &[DescriptorPoolSize {
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        }],
    )?;
    let descriptor_set = pool.allocate(&set_layout)?;
    descriptor_set.write_buffer(0, DescriptorType::STORAGE_BUFFER, &buffer, 0, BUFFER_SIZE);
    println!("[OK] Allocated and wrote descriptor set");

    // -------------------------------------------------------------------
    // 6. Pipeline layout + compute pipeline
    // -------------------------------------------------------------------
    let pipeline_layout = PipelineLayout::new(&device, &[&set_layout])?;
    let pipeline = ComputePipeline::new(&device, &pipeline_layout, &shader, "main")?;
    println!("[OK] Created compute pipeline");

    // -------------------------------------------------------------------
    // 7. Record + submit a command buffer
    // -------------------------------------------------------------------
    let cmd_pool = CommandPool::new(&device, queue_family_index)?;
    let mut cmd = cmd_pool.allocate_primary()?;
    {
        let mut rec = cmd.begin()?;
        rec.bind_compute_pipeline(&pipeline);
        rec.bind_compute_descriptor_sets(&pipeline_layout, 0, &[&descriptor_set]);

        // Dispatch ceil(N/local_size_x) workgroups along X.
        let group_count_x = ELEMENT_COUNT.div_ceil(WORKGROUP_SIZE);
        rec.dispatch(group_count_x, 1, 1);

        // Memory barrier so the host read after fence wait sees the GPU writes.
        // VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x800
        // VK_PIPELINE_STAGE_HOST_BIT           = 0x4000
        // VK_ACCESS_SHADER_WRITE_BIT           = 0x40
        // VK_ACCESS_HOST_READ_BIT              = 0x2000
        rec.memory_barrier(0x800, 0x4000, 0x40, 0x2000);

        rec.end()?;
    }
    println!("[OK] Recorded compute dispatch");

    let fence = Fence::new(&device)?;
    queue.submit(&[&cmd], Some(&fence))?;
    fence.wait(u64::MAX)?;
    println!("[OK] GPU finished compute work");

    // -------------------------------------------------------------------
    // 8. Read back and verify
    // -------------------------------------------------------------------
    {
        let mapped = memory.map()?;
        let bytes = mapped.as_slice();
        let mut wrong = 0usize;
        for i in 0..ELEMENT_COUNT as usize {
            let read = u32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]);
            let expected = (i as u32).wrapping_mul(i as u32);
            if read != expected {
                if wrong < 5 {
                    eprintln!("  index {i}: got {read}, expected {expected}");
                }
                wrong += 1;
            }
        }
        if wrong > 0 {
            return Err(format!("{wrong} elements had wrong values").into());
        }
    }
    println!("[OK] Verified all {ELEMENT_COUNT} elements were squared by the GPU");

    device.wait_idle()?;
    println!();
    println!("=== compute_square example PASSED ===");
    println!("(All Vulkan resources will now be dropped via RAII.)");
    Ok(())
}
