// SPDX-License-Identifier: MIT OR Apache-2.0
//! Instanced mesh rendering: draw 100 triangles at different positions
//! using per-instance vertex attributes and instanced draw calls.
//!
//! This exercises:
//! - `InputRate::INSTANCE` (per-instance vertex buffer binding)
//! - `Queue::upload_buffer` (one-shot data upload)
//! - `draw(vertex_count, instance_count, 0, 0)` with instance_count > 1
//! - `VertexInputBinding` + `VertexInputAttribute` for multi-binding input
//!
//! Run with: `cargo run -p vulkane --features fetch-spec --example instanced_mesh`

use vulkane::safe::{
    AccessFlags, ApiVersion, AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, Buffer,
    BufferCreateInfo, BufferImageCopy, BufferUsage, ClearValue, CommandPool, DeviceCreateInfo,
    Fence, Format, Framebuffer, GraphicsPipelineBuilder, GraphicsShaderStage, Image,
    Image2dCreateInfo, ImageLayout, ImageUsage, InputRate, Instance, InstanceCreateInfo,
    MemoryPropertyFlags, PipelineLayout, PipelineStage, QueueCreateInfo, QueueFlags, RenderPass,
    RenderPassCreateInfo, ShaderModule, VertexInputAttribute, VertexInputBinding,
};

const W: u32 = 256;
const H: u32 = 256;
const INSTANCE_COUNT: u32 = 100;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let spv_path = format!("{manifest_dir}/examples/shaders/instanced_mesh.wgsl.spv");
    let spv_bytes = std::fs::read(&spv_path)
        .map_err(|e| format!("could not read {spv_path}: {e} (run compile_shader first)"))?;

    let instance = Instance::new(InstanceCreateInfo {
        application_name: Some("vulkane instanced_mesh"),
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

    // Generate a simple triangle (3 vertices, position only = vec3<f32>).
    let vertices: Vec<[f32; 3]> = vec![[0.0, -0.3, 0.0], [0.3, 0.3, 0.0], [-0.3, 0.3, 0.0]];

    // Generate 100 instance offsets on a 10x10 grid.
    let mut offsets: Vec<[f32; 3]> = Vec::with_capacity(INSTANCE_COUNT as usize);
    for row in 0..10 {
        for col in 0..10 {
            let x = (col as f32 - 4.5) * 2.0;
            let y = (row as f32 - 4.5) * 2.0;
            offsets.push([x, y, 0.0]);
        }
    }

    // Upload vertex + instance buffers.
    let (vert_buf, _vert_mem) = queue.upload_buffer(
        &device,
        &physical,
        qf,
        &vertices,
        BufferUsage::VERTEX_BUFFER,
    )?;
    let (inst_buf, _inst_mem) =
        queue.upload_buffer(&device, &physical, qf, &offsets, BufferUsage::VERTEX_BUFFER)?;
    println!(
        "[OK] Uploaded {} vertex + {} instance entries",
        vertices.len(),
        offsets.len()
    );

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
            depth_attachment: None,
        },
    )?;
    let framebuffer = Framebuffer::new(&device, &render_pass, &[&color_view], W, H)?;

    let shader = ShaderModule::from_spirv_bytes(&device, &spv_bytes)?;
    let pipeline_layout = PipelineLayout::new(&device, &[])?;

    let bindings = [
        VertexInputBinding {
            binding: 0,
            stride: 12, // vec3<f32> = 12 bytes
            input_rate: InputRate::VERTEX,
        },
        VertexInputBinding {
            binding: 1,
            stride: 12, // vec3<f32> = 12 bytes
            input_rate: InputRate::INSTANCE,
        },
    ];
    let attributes = [
        VertexInputAttribute {
            location: 0,
            binding: 0,
            format: Format::R32G32B32_SFLOAT,
            offset: 0,
        },
        VertexInputAttribute {
            location: 1,
            binding: 1,
            format: Format::R32G32B32_SFLOAT,
            offset: 0,
        },
    ];

    let pipeline = GraphicsPipelineBuilder::new(&pipeline_layout, &render_pass)
        .stage(GraphicsShaderStage::Vertex, &shader, "vs_main")
        .stage(GraphicsShaderStage::Fragment, &shader, "fs_main")
        .vertex_input(&bindings, &attributes)
        .viewport_extent(W, H)
        .cull_mode(vulkane::safe::CullMode::NONE)
        .build(&device)?;
    println!("[OK] Built instanced graphics pipeline");

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

    // Record + submit.
    let cmd_pool = CommandPool::new(&device, qf)?;
    let mut cmd = cmd_pool.allocate_primary()?;
    {
        let mut rec = cmd.begin()?;
        rec.begin_render_pass_ext(
            &render_pass,
            &framebuffer,
            &[ClearValue::Color([0.0, 0.0, 0.0, 1.0])],
        );
        rec.bind_graphics_pipeline(&pipeline);
        rec.bind_vertex_buffers(0, &[(&vert_buf, 0)]);
        rec.bind_vertex_buffers(1, &[(&inst_buf, 0)]);
        rec.draw(3, INSTANCE_COUNT, 0, 0);
        rec.end_render_pass();

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

    let m = rb_mem.map()?;
    let bytes = m.as_slice();
    let mut painted = 0u32;
    for px in 0..(W * H) as usize {
        let r = bytes[px * 4];
        let g = bytes[px * 4 + 1];
        let b = bytes[px * 4 + 2];
        if r != 0 || g != 0 || b != 0 {
            painted += 1;
        }
    }
    println!(
        "[OK] {painted} / {} non-black pixels ({:.1}%)",
        W * H,
        painted as f32 / (W * H) as f32 * 100.0
    );
    assert!(
        painted > 1000,
        "expected significant pixel coverage from 100 instanced triangles"
    );

    drop(m);
    device.wait_idle()?;
    println!("\n=== instanced_mesh example PASSED ===");
    Ok(())
}
