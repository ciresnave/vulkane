// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end 2D storage-image compute example.
//!
//! This example demonstrates the entire image-on-compute path of the safe
//! wrapper:
//!
//! 1. Loads the pre-compiled `invert_image.spv` shader from disk.
//! 2. Creates a 64×64 RGBA8 storage image and a host-visible staging buffer.
//! 3. Pre-fills the staging buffer with a colorful gradient.
//! 4. Records a command buffer that:
//!    - Transitions the image `UNDEFINED -> TRANSFER_DST_OPTIMAL`
//!    - Copies the staging buffer into the image
//!    - Transitions `TRANSFER_DST_OPTIMAL -> GENERAL` (the storage layout)
//!    - Binds the image to a `STORAGE_IMAGE` descriptor and dispatches the
//!      shader with `ceil(64/8) × ceil(64/8) = 8 × 8` workgroups
//!    - Transitions `GENERAL -> TRANSFER_SRC_OPTIMAL`
//!    - Copies the image back into the staging buffer
//! 5. Reads the staging buffer and verifies every pixel was inverted.
//!
//! Run with: `cargo run --example compute_image_invert -p vulkane --features fetch-spec`
//!
//! To regenerate the SPIR-V after editing the GLSL source:
//!   cargo run -p vulkane --features naga,fetch-spec --example compile_shader

use vulkane::safe::{
    AccessFlags, ApiVersion, Buffer, BufferCreateInfo, BufferImageCopy, BufferUsage, CommandPool,
    ComputePipeline, DescriptorPool, DescriptorPoolSize, DescriptorSetLayout,
    DescriptorSetLayoutBinding, DescriptorType, DeviceCreateInfo, DeviceMemory, Fence, Format,
    Image, Image2dCreateInfo, ImageBarrier, ImageLayout, ImageUsage, ImageView, Instance,
    InstanceCreateInfo, MemoryPropertyFlags, PipelineLayout, PipelineStage, QueueCreateInfo,
    QueueFlags, ShaderModule, ShaderStageFlags,
};

const W: u32 = 64;
const H: u32 = 64;
const PIXEL_BYTES: u64 = 4;
const BUF_SIZE: u64 = (W as u64) * (H as u64) * PIXEL_BYTES;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load the pre-compiled SPIR-V shader.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let spv_path = format!("{manifest_dir}/examples/shaders/invert_image.spv");
    let spv_bytes = match std::fs::read(&spv_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ERROR: could not read pre-compiled shader at {spv_path}: {e}");
            eprintln!(
                "Run `cargo run -p vulkane --features naga,fetch-spec --example compile_shader` to regenerate it."
            );
            return Err(e.into());
        }
    };
    println!("[OK] Loaded {} bytes of SPIR-V", spv_bytes.len());

    // 2. Instance + physical device + logical device + queue.
    let instance = match Instance::new(InstanceCreateInfo {
        application_name: Some("vulkane compute_image_invert example"),
        api_version: ApiVersion::V1_0,
        ..InstanceCreateInfo::default()
    }) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("SKIP: could not create Vulkan instance: {e}");
            return Ok(());
        }
    };

    let physical = instance
        .enumerate_physical_devices()?
        .into_iter()
        .find(|pd| pd.find_queue_family(QueueFlags::COMPUTE).is_some())
        .ok_or("No physical device with a compute-capable queue family")?;
    println!("[OK] Using GPU: {}", physical.properties().device_name());

    let queue_family_index = physical.find_queue_family(QueueFlags::COMPUTE).unwrap();
    let device = physical.create_device(DeviceCreateInfo {
        queue_create_infos: &[QueueCreateInfo::single(queue_family_index)],
        ..Default::default()
    })?;
    let queue = device.get_queue(queue_family_index, 0);

    // 3. Allocate the image and a host-visible staging buffer.
    let image = Image::new_2d(
        &device,
        Image2dCreateInfo {
            format: Format::R8G8B8A8_UNORM,
            width: W,
            height: H,
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
        },
    )?;
    let img_req = image.memory_requirements();
    let img_mt = physical
        .find_memory_type(img_req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| {
            physical.find_memory_type(img_req.memory_type_bits, MemoryPropertyFlags::HOST_VISIBLE)
        })
        .ok_or("no compatible memory type for image")?;
    let img_mem = DeviceMemory::allocate(&device, img_req.size, img_mt)?;
    image.bind_memory(&img_mem, 0)?;
    let view = ImageView::new_2d_color(&image)?;
    println!("[OK] Created 64x64 RGBA8 storage image");

    let staging = Buffer::new(
        &device,
        BufferCreateInfo {
            size: BUF_SIZE,
            usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        },
    )?;
    let st_req = staging.memory_requirements();
    let st_mt = physical
        .find_memory_type(
            st_req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or("no host-visible memory type")?;
    let mut st_mem = DeviceMemory::allocate(&device, st_req.size, st_mt)?;
    staging.bind_memory(&st_mem, 0)?;

    // Pre-fill staging with a gradient: r = x*255/(W-1), g = y*255/(H-1), b = 128, a = 255.
    {
        let mut m = st_mem.map()?;
        let bytes = m.as_slice_mut();
        for y in 0..H {
            for x in 0..W {
                let i = ((y * W + x) * 4) as usize;
                bytes[i] = (x * 255 / (W - 1)) as u8;
                bytes[i + 1] = (y * 255 / (H - 1)) as u8;
                bytes[i + 2] = 128;
                bytes[i + 3] = 255;
            }
        }
    }
    println!("[OK] Wrote gradient into staging buffer");

    // 4. Shader, descriptor layout, pool, set, pipeline.
    let shader = ShaderModule::from_spirv_bytes(&device, &spv_bytes)?;
    let set_layout = DescriptorSetLayout::new(
        &device,
        &[DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
        }],
    )?;
    let dpool = DescriptorPool::new(
        &device,
        1,
        &[DescriptorPoolSize {
            descriptor_type: DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
        }],
    )?;
    let dset = dpool.allocate(&set_layout)?;
    dset.write_storage_image(0, &view, ImageLayout::GENERAL);

    let pipeline_layout = PipelineLayout::new(&device, &[&set_layout])?;
    let pipeline = ComputePipeline::new(&device, &pipeline_layout, &shader, "main")?;
    println!("[OK] Built compute pipeline for storage image");

    // 5. Record + submit a command buffer that does the whole round trip.
    let cmd_pool = CommandPool::new(&device, queue_family_index)?;
    let mut cmd = cmd_pool.allocate_primary()?;
    {
        let mut rec = cmd.begin()?;

        // UNDEFINED -> TRANSFER_DST_OPTIMAL
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
        // Upload pixels.
        rec.copy_buffer_to_image(
            &staging,
            &image,
            ImageLayout::TRANSFER_DST_OPTIMAL,
            &[BufferImageCopy::full_2d(W, H)],
        );
        // TRANSFER_DST -> GENERAL (storage layout)
        rec.image_barrier(
            PipelineStage::TRANSFER,
            PipelineStage::COMPUTE_SHADER,
            ImageBarrier::color(
                &image,
                ImageLayout::TRANSFER_DST_OPTIMAL,
                ImageLayout::GENERAL,
                AccessFlags::TRANSFER_WRITE,
                AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE,
            ),
        );

        // Dispatch the invert shader.
        rec.bind_compute_pipeline(&pipeline);
        rec.bind_compute_descriptor_sets(&pipeline_layout, 0, &[&dset]);
        rec.dispatch(W.div_ceil(8), H.div_ceil(8), 1);

        // GENERAL -> TRANSFER_SRC for readback.
        rec.image_barrier(
            PipelineStage::COMPUTE_SHADER,
            PipelineStage::TRANSFER,
            ImageBarrier::color(
                &image,
                ImageLayout::GENERAL,
                ImageLayout::TRANSFER_SRC_OPTIMAL,
                AccessFlags::SHADER_WRITE,
                AccessFlags::TRANSFER_READ,
            ),
        );
        // Copy back to staging.
        rec.copy_image_to_buffer(
            &image,
            ImageLayout::TRANSFER_SRC_OPTIMAL,
            &staging,
            &[BufferImageCopy::full_2d(W, H)],
        );
        // Transfer -> Host barrier so the host map sees the bytes.
        rec.memory_barrier(
            PipelineStage::TRANSFER,
            PipelineStage::HOST,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::HOST_READ,
        );

        rec.end()?;
    }

    let fence = Fence::new(&device)?;
    queue.submit(&[&cmd], Some(&fence))?;
    fence.wait(u64::MAX)?;
    println!("[OK] GPU finished compute work");

    // Verify every pixel was inverted in R/G/B (alpha unchanged).
    {
        let m = st_mem.map()?;
        let bytes = m.as_slice();
        let mut wrong = 0usize;
        for y in 0..H {
            for x in 0..W {
                let i = ((y * W + x) * 4) as usize;
                let er = 255 - (x * 255 / (W - 1)) as u8;
                let eg = 255 - (y * 255 / (H - 1)) as u8;
                let eb = 255 - 128u8;
                let ea = 255u8;
                if bytes[i] != er || bytes[i + 1] != eg || bytes[i + 2] != eb || bytes[i + 3] != ea
                {
                    if wrong < 5 {
                        eprintln!(
                            "  pixel ({x},{y}): got ({},{},{},{}), expected ({er},{eg},{eb},{ea})",
                            bytes[i],
                            bytes[i + 1],
                            bytes[i + 2],
                            bytes[i + 3]
                        );
                    }
                    wrong += 1;
                }
            }
        }
        if wrong > 0 {
            return Err(format!("{wrong} pixels had wrong values").into());
        }
    }
    println!(
        "[OK] Verified all {}x{} pixels were inverted by the GPU",
        W, H
    );

    device.wait_idle()?;
    println!();
    println!("=== compute_image_invert example PASSED ===");
    Ok(())
}
