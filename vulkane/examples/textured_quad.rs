// SPDX-License-Identifier: MIT OR Apache-2.0
//! Headless textured quad: build a 4x4 RGBA8 checkerboard, upload it
//! into a sampled image, render a 4-vertex triangle-strip quad that
//! samples that texture, then read back the framebuffer and verify
//! both the centre pixel is non-black AND the rendered output picked
//! up the texture's color values.
//!
//! This example exercises the full graphics + sampling path of the
//! safe wrapper:
//!
//! 1. Loads pre-compiled `textured_quad.wgsl.spv` (multi-entry-point
//!    SPIR-V — `vs_main` for the vertex stage and `fs_main` for the
//!    fragment stage).
//! 2. Creates a 4x4 R8G8B8A8 sampled image populated from a CPU-side
//!    checkerboard via a staging buffer + layout transitions.
//! 3. Creates a `Sampler` (linear, clamp-to-edge — the default).
//! 4. Builds a descriptor set with a separated `SAMPLED_IMAGE` at
//!    binding 0 and a `SAMPLER` at binding 1 (matching the WGSL
//!    shader's separated texture/sampler bindings).
//! 5. Creates a 256x256 RGBA8 color attachment + render pass +
//!    framebuffer for headless rendering.
//! 6. Builds a graphics pipeline with no vertex input, triangle-strip
//!    topology, and the textured-quad pipeline layout.
//! 7. Records: upload texture -> transition to SHADER_READ_ONLY ->
//!    begin render pass -> bind pipeline + descriptor set -> draw 4
//!    vertices -> end render pass -> copy attachment to readback
//!    buffer.
//! 8. Submits, waits, and verifies the rendered pixels match the
//!    expected checkerboard colors after texture sampling.
//!
//! Run with: `cargo run -p vulkane --features fetch-spec --example textured_quad`

use vulkane::safe::{
    AccessFlags, ApiVersion, AttachmentLoadOp, AttachmentStoreOp, Buffer, BufferCreateInfo,
    BufferImageCopy, BufferUsage, CommandPool, DescriptorPool, DescriptorPoolSize,
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorType, DeviceCreateInfo,
    DeviceMemory, Fence, Format, Framebuffer, GraphicsPipelineBuilder, GraphicsShaderStage, Image,
    Image2dCreateInfo, ImageBarrier, ImageLayout, ImageUsage, ImageView, Instance,
    InstanceCreateInfo, MemoryPropertyFlags, PipelineLayout, PipelineStage, PrimitiveTopology,
    QueueCreateInfo, QueueFlags, RenderPass, Sampler, SamplerCreateInfo, ShaderModule,
    ShaderStageFlags,
};

const W: u32 = 256;
const H: u32 = 256;
const PIXEL_BYTES: u64 = 4;
const FB_BYTES: u64 = (W as u64) * (H as u64) * PIXEL_BYTES;

const TEX_W: u32 = 4;
const TEX_H: u32 = 4;
const TEX_BYTES: u64 = (TEX_W as u64) * (TEX_H as u64) * PIXEL_BYTES;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load multi-entry-point SPIR-V.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let spv_path = format!("{manifest_dir}/examples/shaders/textured_quad.wgsl.spv");
    let spv_bytes = std::fs::read(&spv_path).map_err(|e| {
        format!(
            "could not read {spv_path}: {e} \
             (run `cargo run -p vulkane --features naga,fetch-spec --example compile_shader`)"
        )
    })?;
    println!(
        "[OK] Loaded textured_quad SPIR-V ({} bytes)",
        spv_bytes.len()
    );

    // 2. Instance + physical + device + queue.
    let instance = match Instance::new(InstanceCreateInfo {
        application_name: Some("vulkane textured_quad"),
        api_version: ApiVersion::V1_0,
        ..InstanceCreateInfo::default()
    }) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("SKIP: could not create Vulkan instance: {e}");
            return Ok(());
        }
    };
    let physical = match instance
        .enumerate_physical_devices()?
        .into_iter()
        .find(|pd| pd.find_queue_family(QueueFlags::GRAPHICS).is_some())
    {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no physical device with a graphics queue family");
            return Ok(());
        }
    };
    println!("[OK] Using GPU: {}", physical.properties().device_name());

    let queue_family_index = physical.find_queue_family(QueueFlags::GRAPHICS).unwrap();
    let device = physical.create_device(DeviceCreateInfo {
        queue_create_infos: &[QueueCreateInfo::single(queue_family_index)],
        ..Default::default()
    })?;
    let queue = device.get_queue(queue_family_index, 0);

    // 3. Build the 4x4 RGBA8 checkerboard on the host.
    //    Even (i+j) cells are red, odd cells are green. We'll later
    //    verify the rendered output contains both colors after
    //    sampling.
    let mut texels = [0u8; (TEX_W * TEX_H * 4) as usize];
    for j in 0..TEX_H {
        for i in 0..TEX_W {
            let idx = ((j * TEX_W + i) * 4) as usize;
            let red = (i + j) % 2 == 0;
            texels[idx] = if red { 255 } else { 0 };
            texels[idx + 1] = if red { 0 } else { 255 };
            texels[idx + 2] = 0;
            texels[idx + 3] = 255;
        }
    }
    println!("[OK] Built {TEX_W}x{TEX_H} checkerboard texture on the host");

    // 4. Staging buffer for the texture upload.
    let stage = Buffer::new(
        &device,
        BufferCreateInfo {
            size: TEX_BYTES,
            usage: BufferUsage::TRANSFER_SRC,
        },
    )?;
    let stage_req = stage.memory_requirements();
    let stage_mt = physical
        .find_memory_type(
            stage_req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or("no host-visible memory type for staging")?;
    let mut stage_mem = DeviceMemory::allocate(&device, stage_req.size, stage_mt)?;
    stage.bind_memory(&stage_mem, 0)?;
    {
        let mut m = stage_mem.map()?;
        m.as_slice_mut()[..texels.len()].copy_from_slice(&texels);
    }

    // 5. The sampled texture image.
    let texture = Image::new_2d(
        &device,
        Image2dCreateInfo {
            format: Format::R8G8B8A8_UNORM,
            width: TEX_W,
            height: TEX_H,
            usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
        },
    )?;
    let tex_req = texture.memory_requirements();
    let tex_mt = physical
        .find_memory_type(tex_req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| {
            physical.find_memory_type(tex_req.memory_type_bits, MemoryPropertyFlags::HOST_VISIBLE)
        })
        .ok_or("no compatible memory type for texture")?;
    let tex_mem = DeviceMemory::allocate(&device, tex_req.size, tex_mt)?;
    texture.bind_memory(&tex_mem, 0)?;
    let tex_view = ImageView::new_2d_color(&texture)?;
    let sampler = Sampler::new(&device, SamplerCreateInfo::default())?;
    println!("[OK] Created sampled texture + view + sampler");

    // 6. Color attachment image for the rendered output.
    let color = Image::new_2d(
        &device,
        Image2dCreateInfo {
            format: Format::R8G8B8A8_UNORM,
            width: W,
            height: H,
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
        },
    )?;
    let color_req = color.memory_requirements();
    let color_mt = physical
        .find_memory_type(
            color_req.memory_type_bits,
            MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .or_else(|| {
            physical.find_memory_type(
                color_req.memory_type_bits,
                MemoryPropertyFlags::HOST_VISIBLE,
            )
        })
        .ok_or("no compatible memory type for color attachment")?;
    let color_mem = DeviceMemory::allocate(&device, color_req.size, color_mt)?;
    color.bind_memory(&color_mem, 0)?;
    let color_view = ImageView::new_2d_color(&color)?;
    println!("[OK] Created {W}x{H} color attachment");

    // 7. Render pass + framebuffer.
    let render_pass = RenderPass::simple_color(
        &device,
        Format::R8G8B8A8_UNORM,
        AttachmentLoadOp::CLEAR,
        AttachmentStoreOp::STORE,
        ImageLayout::TRANSFER_SRC_OPTIMAL,
    )?;
    let framebuffer = Framebuffer::new(&device, &render_pass, &[&color_view], W, H)?;

    // 8. Descriptor set: binding 0 = sampled image, binding 1 = sampler.
    //    This matches the WGSL shader's `texture_2d` + `sampler` pair.
    let set_layout = DescriptorSetLayout::new(
        &device,
        &[
            DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::FRAGMENT,
            },
            DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: DescriptorType::SAMPLER,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::FRAGMENT,
            },
        ],
    )?;
    let descriptor_pool = DescriptorPool::new(
        &device,
        1,
        &[
            DescriptorPoolSize {
                descriptor_type: DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
            },
            DescriptorPoolSize {
                descriptor_type: DescriptorType::SAMPLER,
                descriptor_count: 1,
            },
        ],
    )?;
    let descriptor_set = descriptor_pool.allocate(&set_layout)?;
    descriptor_set.write_sampled_image(0, &tex_view, ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    descriptor_set.write_sampler(1, &sampler);

    // 9. Pipeline layout + graphics pipeline.
    let pipeline_layout = PipelineLayout::new(&device, &[&set_layout])?;
    let shader = ShaderModule::from_spirv_bytes(&device, &spv_bytes)?;
    let pipeline = GraphicsPipelineBuilder::new(&pipeline_layout, &render_pass)
        .stage(GraphicsShaderStage::Vertex, &shader, "vs_main")
        .stage(GraphicsShaderStage::Fragment, &shader, "fs_main")
        .topology(PrimitiveTopology::TRIANGLE_STRIP)
        .viewport_extent(W, H)
        .cull_mode(vulkane::safe::CullMode::NONE)
        .front_face(vulkane::safe::FrontFace::COUNTER_CLOCKWISE)
        .build(&device)?;
    println!("[OK] Built graphics pipeline (textured_quad)");

    // 10. Readback buffer for the rendered framebuffer.
    let readback = Buffer::new(
        &device,
        BufferCreateInfo {
            size: FB_BYTES,
            usage: BufferUsage::TRANSFER_DST,
        },
    )?;
    let rb_req = readback.memory_requirements();
    let rb_mt = physical
        .find_memory_type(
            rb_req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or("no host-visible memory type for readback")?;
    let mut rb_mem = DeviceMemory::allocate(&device, rb_req.size, rb_mt)?;
    readback.bind_memory(&rb_mem, 0)?;

    // 11. Record + submit.
    let cmd_pool = CommandPool::new(&device, queue_family_index)?;
    let mut cmd = cmd_pool.allocate_primary()?;
    {
        let mut rec = cmd.begin()?;

        // Texture: UNDEFINED -> TRANSFER_DST
        rec.image_barrier(
            PipelineStage::TOP_OF_PIPE,
            PipelineStage::TRANSFER,
            ImageBarrier::color(
                &texture,
                ImageLayout::UNDEFINED,
                ImageLayout::TRANSFER_DST_OPTIMAL,
                AccessFlags::NONE,
                AccessFlags::TRANSFER_WRITE,
            ),
        );
        // Upload texture from staging buffer.
        rec.copy_buffer_to_image(
            &stage,
            &texture,
            ImageLayout::TRANSFER_DST_OPTIMAL,
            &[BufferImageCopy::full_2d(TEX_W, TEX_H)],
        );
        // Texture: TRANSFER_DST -> SHADER_READ_ONLY (for the fragment stage)
        rec.image_barrier(
            PipelineStage::TRANSFER,
            PipelineStage::FRAGMENT_SHADER,
            ImageBarrier::color(
                &texture,
                ImageLayout::TRANSFER_DST_OPTIMAL,
                ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                AccessFlags::TRANSFER_WRITE,
                AccessFlags::SHADER_READ,
            ),
        );

        // Begin render pass on the color attachment, draw the textured quad.
        rec.begin_render_pass(&render_pass, &framebuffer, &[[0.0, 0.0, 0.0, 1.0]]);
        rec.bind_graphics_pipeline(&pipeline);
        rec.bind_graphics_descriptor_sets(&pipeline_layout, 0, &[&descriptor_set]);
        rec.draw(4, 1, 0, 0);
        rec.end_render_pass();

        // Color attachment is now in TRANSFER_SRC_OPTIMAL (per the
        // render pass finalLayout). Copy it to the readback buffer.
        rec.copy_image_to_buffer(
            &color,
            ImageLayout::TRANSFER_SRC_OPTIMAL,
            &readback,
            &[BufferImageCopy::full_2d(W, H)],
        );
        // Transfer -> Host so the host read sees the bytes.
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
    println!("[OK] GPU finished rendering");

    // 12. Verify the rendered output picked up the texture.
    let m = rb_mem.map()?;
    let bytes = m.as_slice();

    let cx = W / 2;
    let cy = H / 2;
    let i = ((cy * W + cx) * 4) as usize;
    let cr = bytes[i];
    let cg = bytes[i + 1];
    let cb = bytes[i + 2];
    let ca = bytes[i + 3];
    println!("[OK] Centre pixel: ({cr}, {cg}, {cb}, {ca})");

    // The quad is centered, so the centre pixel must be inside it and
    // therefore non-black (the clear color is black, the texture cells
    // are red or green — never both zero on RGB).
    assert!(
        cr != 0 || cg != 0 || cb != 0,
        "centre pixel is black — quad was not rasterized or texture not sampled"
    );

    // Also verify both checkerboard colors made it through. Walk the
    // whole image and tally how many pixels are red-ish, green-ish, or
    // black (background).
    let mut reds = 0u32;
    let mut greens = 0u32;
    let mut blacks = 0u32;
    for px in 0..(W * H) as usize {
        let r = bytes[px * 4];
        let g = bytes[px * 4 + 1];
        let b = bytes[px * 4 + 2];
        if r > 100 && g < 50 && b < 50 {
            reds += 1;
        } else if g > 100 && r < 50 && b < 50 {
            greens += 1;
        } else if r == 0 && g == 0 && b == 0 {
            blacks += 1;
        }
    }
    let total = W * H;
    println!("[OK] Pixel tally: {reds} red, {greens} green, {blacks} black (of {total})");
    assert!(
        reds > 100,
        "expected at least 100 red pixels from the checkerboard texture, got {reds}"
    );
    assert!(
        greens > 100,
        "expected at least 100 green pixels from the checkerboard texture, got {greens}"
    );
    assert!(
        blacks > 100,
        "expected at least 100 black background pixels around the quad, got {blacks}"
    );

    drop(m);
    device.wait_idle()?;
    println!();
    println!("=== textured_quad example PASSED ===");
    Ok(())
}
