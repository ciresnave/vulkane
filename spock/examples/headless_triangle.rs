//! Headless triangle: render a colored RGB triangle to a 256x256 R8G8B8A8
//! image using a real graphics pipeline, then read the pixels back via a
//! staging buffer and verify the centre pixel is non-black.
//!
//! Demonstrates the safe wrapper's full graphics path without needing a
//! window or swapchain:
//!
//! 1. Loads pre-compiled `triangle.vert.spv` / `triangle.frag.spv`.
//! 2. Creates a 256x256 RGBA8 color attachment image (TRANSFER_SRC so
//!    we can read it back).
//! 3. Builds a single-subpass render pass clearing to black on load
//!    and storing on end.
//! 4. Builds a graphics pipeline with no vertex input (the vertex
//!    shader uses gl_VertexIndex to pick from a hardcoded array).
//! 5. Records a command buffer that begins the render pass, draws 3
//!    vertices, ends the pass, and copies the image to a staging
//!    buffer.
//! 6. Submits, waits, and verifies that the centre pixel of the image
//!    is non-zero (proves the triangle was rasterized).
//!
//! Run with: `cargo run -p spock --features fetch-spec --example headless_triangle`

use spock::safe::{
    ApiVersion, Buffer, BufferCreateInfo, BufferImageCopy, BufferUsage, CommandPool,
    DeviceCreateInfo, DeviceMemory, Fence, Format, Framebuffer, GraphicsPipelineBuilder,
    GraphicsShaderStage, Image, Image2dCreateInfo, ImageBarrier, ImageLayout, ImageUsage,
    ImageView, Instance, InstanceCreateInfo, MemoryPropertyFlags, PipelineLayout, QueueCreateInfo,
    QueueFlags, RenderPass, RenderPassCreateInfo, ShaderModule,
};

const W: u32 = 256;
const H: u32 = 256;
const PIXEL_BYTES: u64 = 4;
const BUF_SIZE: u64 = (W as u64) * (H as u64) * PIXEL_BYTES;

const TOP_OF_PIPE: u32 = 0x1;
const TRANSFER: u32 = 0x1000;
const COLOR_ATTACHMENT_OUTPUT: u32 = 0x400;
const HOST: u32 = 0x4000;
const ACCESS_TRANSFER_READ: u32 = 0x800;
const ACCESS_TRANSFER_WRITE: u32 = 0x1000;
const ACCESS_HOST_READ: u32 = 0x2000;
const ACCESS_COLOR_ATTACHMENT_WRITE: u32 = 0x100;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load shaders.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let vert_path = format!("{manifest_dir}/examples/shaders/triangle.vert.spv");
    let frag_path = format!("{manifest_dir}/examples/shaders/triangle.frag.spv");
    let vert_bytes = std::fs::read(&vert_path).map_err(|e| {
        format!("could not read {vert_path}: {e} (run `cargo run -p spock --features naga,fetch-spec --example compile_shader`)")
    })?;
    let frag_bytes =
        std::fs::read(&frag_path).map_err(|e| format!("could not read {frag_path}: {e}"))?;
    println!(
        "[OK] Loaded vertex shader ({} bytes), fragment shader ({} bytes)",
        vert_bytes.len(),
        frag_bytes.len()
    );

    // 2. Instance + physical + device + queue.
    let instance = match Instance::new(InstanceCreateInfo {
        application_name: Some("spock headless_triangle"),
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
        .find(|pd| pd.find_queue_family(QueueFlags::GRAPHICS).is_some())
        .ok_or("No physical device with a graphics queue family")?;
    println!("[OK] Using GPU: {}", physical.properties().device_name());

    let queue_family_index = physical.find_queue_family(QueueFlags::GRAPHICS).unwrap();
    let device = physical.create_device(DeviceCreateInfo {
        queue_create_infos: &[QueueCreateInfo {
            queue_family_index,
            queue_priorities: vec![1.0],
        }],
        ..Default::default()
    })?;
    let queue = device.get_queue(queue_family_index, 0);

    // 3. Color attachment image.
    let image = Image::new_2d(
        &device,
        Image2dCreateInfo {
            format: Format::R8G8B8A8_UNORM,
            width: W,
            height: H,
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
        },
    )?;
    let img_req = image.memory_requirements();
    let img_mt = physical
        .find_memory_type(img_req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| {
            physical.find_memory_type(img_req.memory_type_bits, MemoryPropertyFlags::HOST_VISIBLE)
        })
        .ok_or("no compatible memory type")?;
    let img_mem = DeviceMemory::allocate(&device, img_req.size, img_mt)?;
    image.bind_memory(&img_mem, 0)?;
    let view = ImageView::new_2d_color(&image)?;
    println!("[OK] Created {W}x{H} R8G8B8A8 color attachment");

    // 4. Render pass: one color attachment, clear on load, store on end,
    //    UNDEFINED -> TRANSFER_SRC_OPTIMAL so we can copy out at the end.
    let render_pass = RenderPass::new(
        &device,
        RenderPassCreateInfo {
            color_attachments: &[spock::safe::AttachmentDescription {
                format: Format::R8G8B8A8_UNORM,
                load_op: spock::safe::AttachmentLoadOp::CLEAR,
                store_op: spock::safe::AttachmentStoreOp::STORE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::TRANSFER_SRC_OPTIMAL,
            }],
            depth_attachment: None,
        },
    )?;
    let framebuffer = Framebuffer::new(&device, &render_pass, &[&view], W, H)?;
    println!("[OK] Created render pass + framebuffer");

    // 5. Pipeline.
    let vert = ShaderModule::from_spirv_bytes(&device, &vert_bytes)?;
    let frag = ShaderModule::from_spirv_bytes(&device, &frag_bytes)?;
    let pipeline_layout = PipelineLayout::new(&device, &[])?;
    let pipeline = GraphicsPipelineBuilder::new(&pipeline_layout, &render_pass)
        .stage(GraphicsShaderStage::Vertex, &vert, "main")
        .stage(GraphicsShaderStage::Fragment, &frag, "main")
        .viewport_extent(W, H)
        .cull_mode(spock::safe::CullMode::NONE)
        .front_face(spock::safe::FrontFace::COUNTER_CLOCKWISE)
        .build(&device)?;
    println!("[OK] Built graphics pipeline");

    // 6. Staging buffer for readback.
    let staging = Buffer::new(
        &device,
        BufferCreateInfo {
            size: BUF_SIZE,
            usage: BufferUsage::TRANSFER_DST,
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

    // 7. Record + submit.
    let cmd_pool = CommandPool::new(&device, queue_family_index)?;
    let mut cmd = cmd_pool.allocate_primary()?;
    {
        let mut rec = cmd.begin()?;

        // The render pass already transitions UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
        // and back to TRANSFER_SRC_OPTIMAL via initial/final layouts, so we
        // don't need an explicit barrier before begin_render_pass.
        rec.begin_render_pass(&render_pass, &framebuffer, &[[0.0, 0.0, 0.0, 1.0]]);
        rec.bind_graphics_pipeline(&pipeline);
        rec.draw(3, 1, 0, 0);
        rec.end_render_pass();

        // Image is now in TRANSFER_SRC_OPTIMAL (per render pass finalLayout).
        // Copy it to the staging buffer.
        rec.copy_image_to_buffer(
            &image,
            ImageLayout::TRANSFER_SRC_OPTIMAL,
            &staging,
            &[BufferImageCopy::full_2d(W, H)],
        );
        // Transfer -> Host
        rec.memory_barrier(TRANSFER, HOST, ACCESS_TRANSFER_WRITE, ACCESS_HOST_READ);

        // Suppress unused-import warnings for the access constants.
        let _ = (
            TOP_OF_PIPE,
            COLOR_ATTACHMENT_OUTPUT,
            ACCESS_TRANSFER_READ,
            ACCESS_COLOR_ATTACHMENT_WRITE,
        );
        let _: ImageBarrier;

        rec.end()?;
    }

    let fence = Fence::new(&device)?;
    queue.submit(&[&cmd], Some(&fence))?;
    fence.wait(u64::MAX)?;
    println!("[OK] GPU finished rendering");

    // 8. Verify the centre pixel of the image is non-black (proves the
    //    triangle was actually rasterized).
    let m = st_mem.map()?;
    let bytes = m.as_slice();
    let cx = W / 2;
    let cy = H / 2;
    let i = ((cy * W + cx) * 4) as usize;
    let r = bytes[i];
    let g = bytes[i + 1];
    let b = bytes[i + 2];
    let a = bytes[i + 3];
    println!("[OK] Centre pixel: ({r}, {g}, {b}, {a})");
    if r == 0 && g == 0 && b == 0 {
        return Err("centre pixel is black — triangle was not rasterized".into());
    }

    // Count how many non-black pixels we have. A clean triangle covers
    // about 24% of a 256x256 viewport (the hardcoded triangle in the
    // shader covers vertices at (0,-0.7), (0.7, 0.7), (-0.7, 0.7),
    // which is roughly 0.49 * 0.5 = 24.5%). Allow a wide tolerance.
    let mut painted = 0u32;
    for i in 0..(W * H) as usize {
        let r = bytes[i * 4];
        let g = bytes[i * 4 + 1];
        let b = bytes[i * 4 + 2];
        if r != 0 || g != 0 || b != 0 {
            painted += 1;
        }
    }
    let total = W * H;
    let pct = painted as f32 / total as f32 * 100.0;
    println!("[OK] {painted} / {total} non-black pixels ({pct:.1}%)");
    assert!(
        painted > 5000,
        "fewer than 5000 painted pixels — render didn't cover much of the viewport"
    );

    drop(m);
    device.wait_idle()?;
    println!();
    println!("=== headless_triangle example PASSED ===");
    Ok(())
}
