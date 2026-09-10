// SPDX-License-Identifier: MIT OR Apache-2.0
//! Depth prepass: two-pass rendering demonstrating depth-only rendering
//! followed by a color pass that uses the depth buffer for early-Z rejection.
//!
//! Pass 1 (depth-only): renders a triangle into a depth attachment with
//!   depth test + depth write enabled, no color output.
//! Pass 2 (color): renders the same triangle into a color attachment with
//!   depth test (EQUAL) + depth read-only. Only fragments that match the
//!   depth prepass survive — proving the depth prepass wrote correct values.
//!
//! This exercises:
//! - `ImageView::new_2d_depth` (depth aspect views)
//! - `ImageBarrier::depth` (depth aspect barriers)
//! - `ClearValue::DepthStencil` (depth clear in begin_render_pass_ext)
//! - `CompareOp::EQUAL` in the graphics pipeline builder
//! - `RenderPass` with both color and depth attachments
//!
//! Run with: `cargo run -p vulkane --features fetch-spec --example depth_prepass`

use vulkane::safe::{
    AccessFlags, ApiVersion, AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, Buffer,
    BufferCreateInfo, BufferImageCopy, BufferUsage, ClearValue, CommandPool, CompareOp,
    DeviceCreateInfo, DeviceMemory, Fence, Format, Framebuffer, GraphicsPipelineBuilder,
    GraphicsShaderStage, Image, Image2dCreateInfo, ImageLayout, ImageUsage, ImageView, Instance,
    InstanceCreateInfo, MemoryPropertyFlags, PipelineLayout, PipelineStage, QueueCreateInfo,
    QueueFlags, RenderPass, RenderPassCreateInfo, ShaderModule,
};

const W: u32 = 256;
const H: u32 = 256;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let spv_path = format!("{manifest_dir}/examples/shaders/depth_prepass.wgsl.spv");
    let spv_bytes = std::fs::read(&spv_path)
        .map_err(|e| format!("could not read {spv_path}: {e} (run compile_shader first)"))?;

    let instance = Instance::new(InstanceCreateInfo {
        application_name: Some("vulkane depth_prepass"),
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

    // Color attachment.
    let (color_img, _color_mem, color_view) = Image::new_2d_bound(
        &device,
        &physical,
        Image2dCreateInfo {
            format: Format::R8G8B8A8_UNORM,
            width: W,
            height: H,
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
        },
        MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    // Depth attachment.
    let depth_img = Image::new_2d(
        &device,
        Image2dCreateInfo {
            format: Format::D32_SFLOAT,
            width: W,
            height: H,
            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
        },
    )?;
    let d_req = depth_img.memory_requirements();
    let d_mt = physical
        .find_memory_type(d_req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| {
            physical.find_memory_type(d_req.memory_type_bits, MemoryPropertyFlags::HOST_VISIBLE)
        })
        .ok_or("no memory type for depth image")?;
    let depth_mem = DeviceMemory::allocate(&device, d_req.size, d_mt)?;
    depth_img.bind_memory(&depth_mem, 0)?;
    let depth_view = ImageView::new_2d_depth(&depth_img)?;
    println!("[OK] Created {W}x{H} color + depth attachments");

    // Render pass: color (CLEAR→STORE, UNDEFINED→TRANSFER_SRC) + depth (CLEAR→STORE).
    let render_pass = RenderPass::new(
        &device,
        RenderPassCreateInfo {
            color_attachments: &[AttachmentDescription {
                format: Format::R8G8B8A8_UNORM,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::TRANSFER_SRC_OPTIMAL,
            }],
            depth_attachment: Some(AttachmentDescription {
                format: Format::D32_SFLOAT,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            }),
        },
    )?;
    let framebuffer = Framebuffer::new(&device, &render_pass, &[&color_view, &depth_view], W, H)?;
    println!("[OK] Created render pass + framebuffer (color + depth)");

    // Two pipelines from the same shader module (multi-entry-point WGSL).
    let shader = ShaderModule::from_spirv_bytes(&device, &spv_bytes)?;
    let pipeline_layout = PipelineLayout::new(&device, &[])?;

    // Pass 1 pipeline: depth write, no color write (use fs_depth).
    let depth_pipeline = GraphicsPipelineBuilder::new(&pipeline_layout, &render_pass)
        .stage(GraphicsShaderStage::Vertex, &shader, "vs_main")
        .stage(GraphicsShaderStage::Fragment, &shader, "fs_depth")
        .viewport_extent(W, H)
        .depth_test(true, true)
        .cull_mode(vulkane::safe::CullMode::NONE)
        .build(&device)?;

    // Pass 2 pipeline: depth test EQUAL, no depth write, color write (use fs_color).
    let color_pipeline = GraphicsPipelineBuilder::new(&pipeline_layout, &render_pass)
        .stage(GraphicsShaderStage::Vertex, &shader, "vs_main")
        .stage(GraphicsShaderStage::Fragment, &shader, "fs_color")
        .viewport_extent(W, H)
        .depth_test(true, false)
        .depth_compare_op(CompareOp::EQUAL)
        .cull_mode(vulkane::safe::CullMode::NONE)
        .build(&device)?;
    println!("[OK] Built depth + color pipelines");

    // Readback buffer.
    let (readback, mut rb_mem) = Buffer::new_bound(
        &device,
        &physical,
        BufferCreateInfo {
            size: (W * H * 4) as u64,
            usage: BufferUsage::TRANSFER_DST,
        },
        MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // Record: both passes in one render pass instance.
    // Pass 1 draws with depth pipeline, pass 2 draws with color pipeline.
    // Since both are in the same subpass, the depth buffer is available
    // in-flight between the two draw calls.
    let cmd_pool = CommandPool::new(&device, qf)?;
    let mut cmd = cmd_pool.allocate_primary()?;
    {
        let mut rec = cmd.begin()?;
        rec.begin_render_pass_ext(
            &render_pass,
            &framebuffer,
            &[
                ClearValue::Color([0.0, 0.0, 0.0, 1.0]),
                ClearValue::DepthStencil {
                    depth: 1.0,
                    stencil: 0,
                },
            ],
        );

        // Pass 1: depth only.
        rec.bind_graphics_pipeline(&depth_pipeline);
        rec.draw(3, 1, 0, 0);

        // Pass 2: color with depth EQUAL.
        rec.bind_graphics_pipeline(&color_pipeline);
        rec.draw(3, 1, 0, 0);

        rec.end_render_pass();

        // Copy color to readback.
        rec.copy_image_to_buffer(
            &color_img,
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

    // Verify: the colour pass should have written orange pixels exactly
    // where the depth prepass wrote depth. The triangle covers ~24% of
    // the viewport (same as headless_triangle).
    let m = rb_mem.map()?;
    let bytes = m.as_slice();
    let cx = W / 2;
    let cy = H / 2;
    let i = ((cy * W + cx) * 4) as usize;
    let r = bytes[i];
    let g = bytes[i + 1];
    let b = bytes[i + 2];
    println!("[OK] Centre pixel: ({r}, {g}, {b})");

    // Orange = (255, ~153, ~26) from (1.0, 0.6, 0.1).
    assert!(
        r > 200 && g > 100 && b < 80,
        "centre pixel should be orange — depth prepass + EQUAL test failed"
    );

    let mut orange = 0u32;
    for px in 0..(W * H) as usize {
        if bytes[px * 4] > 200 && bytes[px * 4 + 1] > 100 && bytes[px * 4 + 2] < 80 {
            orange += 1;
        }
    }
    println!("[OK] {orange} / {} orange pixels", W * H);
    assert!(
        orange > 5000,
        "expected at least 5000 orange pixels from the depth-prepass + color pass"
    );

    drop(m);
    device.wait_idle()?;
    println!("\n=== depth_prepass example PASSED ===");
    Ok(())
}
