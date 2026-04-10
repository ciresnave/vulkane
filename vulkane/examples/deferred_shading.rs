//! Deferred shading: two-pass rendering with a G-buffer pass writing
//! world position, normal, and albedo to 3 separate color attachments,
//! followed by a fullscreen lighting pass that reads the G-buffer as
//! textures and computes Phong lighting.
//!
//! This exercises:
//! - Multiple color attachments per render pass (G-buffer)
//! - `color_attachment_count(3)` on the pipeline builder
//! - Multiple sampled image descriptors for the lighting pass
//! - Fullscreen triangle rendering (no vertex input)
//! - R32G32B32A32_SFLOAT format for position/normal (high precision)
//! - Two separate render passes with different pipelines
//!
//! Run with: `cargo run -p vulkane --features fetch-spec --example deferred_shading`

use vulkane::safe::{
    AccessFlags, ApiVersion, AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, Buffer,
    BufferCreateInfo, BufferImageCopy, BufferUsage, ClearValue, CommandPool, DescriptorPool,
    DescriptorPoolSize, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorType,
    DeviceCreateInfo, Fence, Format, Framebuffer, GraphicsPipelineBuilder, GraphicsShaderStage,
    Image, Image2dCreateInfo, ImageLayout, ImageUsage, Instance,
    InstanceCreateInfo, MemoryPropertyFlags, PipelineLayout, PipelineStage, QueueCreateInfo,
    QueueFlags, RenderPass, RenderPassCreateInfo, Sampler, SamplerCreateInfo, ShaderModule,
    ShaderStageFlags,
};

const W: u32 = 256;
const H: u32 = 256;
const VERTEX_COUNT: u32 = 9; // 6 ground + 3 triangle

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let spv_path = format!("{manifest_dir}/examples/shaders/deferred_shading.wgsl.spv");
    let spv_bytes = std::fs::read(&spv_path).map_err(|e| {
        format!("could not read {spv_path}: {e} (run compile_shader first)")
    })?;

    let instance = Instance::new(InstanceCreateInfo {
        application_name: Some("vulkane deferred_shading"),
        api_version: ApiVersion::V1_0,
        ..Default::default()
    })?;
    let physical = instance
        .enumerate_physical_devices()?
        .into_iter()
        .find(|pd| pd.find_queue_family(QueueFlags::GRAPHICS).is_some())
        .ok_or("no GPU with a graphics queue")?;
    println!("[OK] Using GPU: {}", physical.properties().device_name());

    let qf = physical.find_queue_family(QueueFlags::GRAPHICS).unwrap();
    let device = physical.create_device(DeviceCreateInfo {
        queue_create_infos: &[QueueCreateInfo::single(qf)],
        ..Default::default()
    })?;
    let queue = device.get_queue(qf, 0);
    let shader = ShaderModule::from_spirv_bytes(&device, &spv_bytes)?;

    // --- G-buffer images ---
    // R32G32B32A32_SFLOAT might not be available on all hardware.
    // Fall back to R16G16B16A16_SFLOAT which is universally supported.
    let gbuf_format = Format(vulkane::raw::bindings::VkFormat::FORMAT_R16G16B16A16_SFLOAT);

    let (_g_pos_img, _g_pos_mem, g_pos_view) = Image::new_2d_bound(
        &device, &physical,
        Image2dCreateInfo {
            format: gbuf_format,
            width: W, height: H,
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
        },
        MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let (_g_norm_img, _g_norm_mem, g_norm_view) = Image::new_2d_bound(
        &device, &physical,
        Image2dCreateInfo {
            format: gbuf_format,
            width: W, height: H,
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
        },
        MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let (_g_albedo_img, _g_albedo_mem, g_albedo_view) = Image::new_2d_bound(
        &device, &physical,
        Image2dCreateInfo {
            format: Format::R8G8B8A8_UNORM,
            width: W, height: H,
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
        },
        MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    println!("[OK] Created 3 G-buffer images ({W}x{H})");

    // Final color output.
    let (final_img, _final_mem, final_view) = Image::new_2d_bound(
        &device, &physical,
        Image2dCreateInfo {
            format: Format::R8G8B8A8_UNORM,
            width: W, height: H,
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
        },
        MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    // --- Render passes ---
    // G-buffer pass: 3 color attachments, no depth.
    let gbuf_rp = RenderPass::new(
        &device,
        RenderPassCreateInfo {
            color_attachments: &[
                AttachmentDescription {
                    format: gbuf_format,
                    load_op: AttachmentLoadOp::CLEAR,
                    store_op: AttachmentStoreOp::STORE,
                    initial_layout: ImageLayout::UNDEFINED,
                    final_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
                AttachmentDescription {
                    format: gbuf_format,
                    load_op: AttachmentLoadOp::CLEAR,
                    store_op: AttachmentStoreOp::STORE,
                    initial_layout: ImageLayout::UNDEFINED,
                    final_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
                AttachmentDescription {
                    format: Format::R8G8B8A8_UNORM,
                    load_op: AttachmentLoadOp::CLEAR,
                    store_op: AttachmentStoreOp::STORE,
                    initial_layout: ImageLayout::UNDEFINED,
                    final_layout: ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
            ],
            depth_attachment: None,
        },
    )?;
    let gbuf_fb = Framebuffer::new(
        &device,
        &gbuf_rp,
        &[&g_pos_view, &g_norm_view, &g_albedo_view],
        W, H,
    )?;

    // Lighting pass: 1 color attachment.
    let light_rp = RenderPass::simple_color(
        &device,
        Format::R8G8B8A8_UNORM,
        AttachmentLoadOp::CLEAR,
        AttachmentStoreOp::STORE,
        ImageLayout::TRANSFER_SRC_OPTIMAL,
    )?;
    let light_fb = Framebuffer::new(&device, &light_rp, &[&final_view], W, H)?;
    println!("[OK] Created G-buffer + lighting render passes");

    // --- Pipelines ---
    let gbuf_layout = PipelineLayout::new(&device, &[])?;
    let gbuf_pipeline = GraphicsPipelineBuilder::new(&gbuf_layout, &gbuf_rp)
        .stage(GraphicsShaderStage::Vertex, &shader, "vs_gbuffer")
        .stage(GraphicsShaderStage::Fragment, &shader, "fs_gbuffer")
        .viewport_extent(W, H)
        .cull_mode(vulkane::safe::CullMode::NONE)
        .color_attachment_count(3)
        .build(&device)?;

    // Lighting descriptor set: 3 sampled images + 1 sampler.
    let light_set_layout = DescriptorSetLayout::new(
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
                descriptor_type: DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::FRAGMENT,
            },
            DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::FRAGMENT,
            },
            DescriptorSetLayoutBinding {
                binding: 3,
                descriptor_type: DescriptorType::SAMPLER,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::FRAGMENT,
            },
        ],
    )?;
    let light_layout = PipelineLayout::new(&device, &[&light_set_layout])?;
    let light_pipeline = GraphicsPipelineBuilder::new(&light_layout, &light_rp)
        .stage(GraphicsShaderStage::Vertex, &shader, "vs_lighting")
        .stage(GraphicsShaderStage::Fragment, &shader, "fs_lighting")
        .viewport_extent(W, H)
        .cull_mode(vulkane::safe::CullMode::NONE)
        .build(&device)?;

    let desc_pool = DescriptorPool::new(
        &device,
        1,
        &[
            DescriptorPoolSize { descriptor_type: DescriptorType::SAMPLED_IMAGE, descriptor_count: 3 },
            DescriptorPoolSize { descriptor_type: DescriptorType::SAMPLER, descriptor_count: 1 },
        ],
    )?;
    let desc_set = desc_pool.allocate(&light_set_layout)?;
    let sampler = Sampler::new(&device, SamplerCreateInfo::default())?;
    desc_set.write_sampled_image(0, &g_pos_view, ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    desc_set.write_sampled_image(1, &g_norm_view, ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    desc_set.write_sampled_image(2, &g_albedo_view, ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    desc_set.write_sampler(3, &sampler);
    println!("[OK] Built G-buffer + lighting pipelines");

    // Readback.
    let (readback, mut rb_mem) = Buffer::new_bound(
        &device, &physical,
        BufferCreateInfo { size: (W * H * 4) as u64, usage: BufferUsage::TRANSFER_DST },
        MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // --- Record ---
    let cmd_pool = CommandPool::new(&device, qf)?;
    let mut cmd = cmd_pool.allocate_primary()?;
    {
        let mut rec = cmd.begin()?;

        // G-buffer pass.
        rec.begin_render_pass_ext(
            &gbuf_rp,
            &gbuf_fb,
            &[
                ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
                ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
                ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
            ],
        );
        rec.bind_graphics_pipeline(&gbuf_pipeline);
        rec.draw(VERTEX_COUNT, 1, 0, 0);
        rec.end_render_pass();

        // G-buffer images are now in SHADER_READ_ONLY_OPTIMAL
        // (set by the render pass finalLayout).

        // Lighting pass.
        rec.begin_render_pass_ext(
            &light_rp,
            &light_fb,
            &[ClearValue::Color([0.0, 0.0, 0.0, 1.0])],
        );
        rec.bind_graphics_pipeline(&light_pipeline);
        rec.bind_graphics_descriptor_sets(&light_layout, 0, &[&desc_set]);
        rec.draw(3, 1, 0, 0); // fullscreen triangle
        rec.end_render_pass();

        // Copy final output to readback.
        rec.copy_image_to_buffer(
            &final_img,
            ImageLayout::TRANSFER_SRC_OPTIMAL,
            &readback,
            &[BufferImageCopy::full_2d(W, H)],
        );
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

    let m = rb_mem.map()?;
    let bytes = m.as_slice();

    let mut lit = 0u32;
    let mut dark_bg = 0u32;
    for px in 0..(W * H) as usize {
        let r = bytes[px * 4];
        let g = bytes[px * 4 + 1];
        let b = bytes[px * 4 + 2];
        let lum = (r as u32 + g as u32 + b as u32) / 3;
        if lum > 30 {
            lit += 1;
        } else {
            dark_bg += 1;
        }
    }
    println!("[OK] Pixel tally: {lit} lit, {dark_bg} dark background");
    assert!(
        lit > 1000,
        "expected visible lit geometry from deferred shading, got only {lit} bright pixels"
    );

    drop(m);
    device.wait_idle()?;
    println!("\n=== deferred_shading example PASSED ===");
    Ok(())
}
