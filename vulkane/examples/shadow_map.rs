//! Shadow mapping: two-pass rendering with a depth-only shadow pass from
//! the light's perspective, followed by a main color pass that samples
//! the shadow map to determine lit vs. shadowed regions.
//!
//! This is the most complex example in the suite. It exercises:
//! - Uniform buffers for MVP matrices (light + camera)
//! - Depth-only render pass with depth bias (prevents shadow acne)
//! - Depth texture sampling via `texture_depth_2d` + `sampler_comparison`
//! - `ImageView::new_2d_depth` for the shadow map view
//! - `ImageBarrier::depth` for depth layout transitions
//! - `SamplerCreateInfo { compare_op: Some(CompareOp::LESS_OR_EQUAL) }`
//! - Two separate render passes with different framebuffers
//! - Descriptor sets with uniform buffers + sampled depth image + sampler
//!
//! The scene is a ground plane with a floating triangle that casts a
//! shadow. The light is positioned above and to the side so the shadow
//! falls visibly on the ground.
//!
//! Run with: `cargo run -p vulkane --features fetch-spec --example shadow_map`

use vulkane::safe::{
    AccessFlags, ApiVersion, AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, Buffer,
    BufferCreateInfo, BufferImageCopy, BufferUsage, ClearValue, CommandPool, CompareOp,
    DescriptorPool, DescriptorPoolSize, DescriptorSetLayout, DescriptorSetLayoutBinding,
    DescriptorType, DeviceCreateInfo, DeviceMemory, Fence, Format, Framebuffer,
    GraphicsPipelineBuilder, GraphicsShaderStage, Image, Image2dCreateInfo, ImageBarrier,
    ImageLayout, ImageUsage, ImageView, Instance, InstanceCreateInfo, MemoryPropertyFlags,
    PipelineLayout, PipelineStage, QueueCreateInfo, QueueFlags, RenderPass, RenderPassCreateInfo,
    Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerFilter, ShaderModule, ShaderStageFlags,
};

const SHADOW_SIZE: u32 = 512;
const W: u32 = 256;
const H: u32 = 256;
const VERTEX_COUNT: u32 = 9; // 6 ground + 3 floating triangle

/// Simple 4x4 matrix type (column-major, matching WGSL/SPIR-V).
type Mat4 = [[f32; 4]; 4];

fn ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
    let w = right - left;
    let h = top - bottom;
    let d = far - near;
    [
        [2.0 / w, 0.0, 0.0, 0.0],
        [0.0, 2.0 / h, 0.0, 0.0],
        [0.0, 0.0, -1.0 / d, 0.0],
        [-(right + left) / w, -(top + bottom) / h, -near / d, 1.0],
    ]
}

fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Mat4 {
    let f = normalize(sub(target, eye));
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

fn mul_mat4(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                out[j][i] += a[k][i] * b[j][k];
            }
        }
    }
    out
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let spv_path = format!("{manifest_dir}/examples/shaders/shadow_map.wgsl.spv");
    let spv_bytes = std::fs::read(&spv_path).map_err(|e| {
        format!("could not read {spv_path}: {e} (run compile_shader first)")
    })?;

    let instance = Instance::new(InstanceCreateInfo {
        application_name: Some("vulkane shadow_map"),
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

    // --- MVP matrices ---
    let light_view = look_at([2.0, 3.0, 2.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let light_proj = ortho(-2.0, 2.0, -2.0, 2.0, 0.1, 10.0);
    let light_mvp = mul_mat4(&light_proj, &light_view);

    let camera_view = look_at([0.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let camera_proj = ortho(-2.0, 2.0, -2.0, 2.0, 0.1, 10.0);
    let camera_mvp = mul_mat4(&camera_proj, &camera_view);

    // Uniform buffers (one for light MVP, one for camera MVP).
    let (light_ubo, mut light_ubo_mem) = Buffer::new_bound(
        &device, &physical,
        BufferCreateInfo { size: 64, usage: BufferUsage::UNIFORM_BUFFER },
        MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
    )?;
    {
        let mut m = light_ubo_mem.map()?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(light_mvp.as_ptr() as *const u8, 64)
        };
        m.as_slice_mut()[..64].copy_from_slice(bytes);
    }

    let (camera_ubo, mut camera_ubo_mem) = Buffer::new_bound(
        &device, &physical,
        BufferCreateInfo { size: 64, usage: BufferUsage::UNIFORM_BUFFER },
        MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
    )?;
    {
        let mut m = camera_ubo_mem.map()?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(camera_mvp.as_ptr() as *const u8, 64)
        };
        m.as_slice_mut()[..64].copy_from_slice(bytes);
    }
    println!("[OK] Created uniform buffers for light + camera MVPs");

    // --- Shadow map (depth-only) ---
    let shadow_img = Image::new_2d(
        &device,
        Image2dCreateInfo {
            format: Format::D32_SFLOAT,
            width: SHADOW_SIZE,
            height: SHADOW_SIZE,
            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
        },
    )?;
    let s_req = shadow_img.memory_requirements();
    let s_mt = physical
        .find_memory_type(s_req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| physical.find_memory_type(s_req.memory_type_bits, MemoryPropertyFlags::HOST_VISIBLE))
        .ok_or("no memory for shadow map")?;
    let _shadow_mem = DeviceMemory::allocate(&device, s_req.size, s_mt)?;
    shadow_img.bind_memory(&_shadow_mem, 0)?;
    let shadow_view = ImageView::new_2d_depth(&shadow_img)?;

    // Comparison sampler for shadow testing.
    let shadow_sampler = Sampler::new(
        &device,
        SamplerCreateInfo {
            mag_filter: SamplerFilter::NEAREST,
            min_filter: SamplerFilter::NEAREST,
            address_mode_u: SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: SamplerAddressMode::CLAMP_TO_EDGE,
            compare_op: Some(CompareOp::LESS_OR_EQUAL),
            ..Default::default()
        },
    )?;
    println!("[OK] Created {SHADOW_SIZE}x{SHADOW_SIZE} shadow map + comparison sampler");

    // --- Color output ---
    let (color_img, _color_mem, color_view) = Image::new_2d_bound(
        &device, &physical,
        Image2dCreateInfo {
            format: Format::R8G8B8A8_UNORM,
            width: W, height: H,
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
        },
        MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    // Depth buffer for main pass.
    let main_depth_img = Image::new_2d(
        &device,
        Image2dCreateInfo {
            format: Format::D32_SFLOAT,
            width: W, height: H,
            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
        },
    )?;
    let md_req = main_depth_img.memory_requirements();
    let md_mt = physical
        .find_memory_type(md_req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| physical.find_memory_type(md_req.memory_type_bits, MemoryPropertyFlags::HOST_VISIBLE))
        .ok_or("no memory for main depth")?;
    let _main_depth_mem = DeviceMemory::allocate(&device, md_req.size, md_mt)?;
    main_depth_img.bind_memory(&_main_depth_mem, 0)?;
    let main_depth_view = ImageView::new_2d_depth(&main_depth_img)?;

    // --- Render passes ---
    // Shadow pass: depth only.
    let shadow_rp = RenderPass::new(
        &device,
        RenderPassCreateInfo {
            color_attachments: &[],
            depth_attachment: Some(AttachmentDescription {
                format: Format::D32_SFLOAT,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            }),
        },
    )?;
    let shadow_fb = Framebuffer::new(&device, &shadow_rp, &[&shadow_view], SHADOW_SIZE, SHADOW_SIZE)?;

    // Main pass: color + depth.
    let main_rp = RenderPass::new(
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
                store_op: AttachmentStoreOp::DONT_CARE,
                initial_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            }),
        },
    )?;
    let main_fb = Framebuffer::new(&device, &main_rp, &[&color_view, &main_depth_view], W, H)?;
    println!("[OK] Created shadow + main render passes");

    // --- Descriptor layout + pool + set ---
    let set_layout = DescriptorSetLayout::new(
        &device,
        &[
            DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::VERTEX,
            },
            DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::VERTEX,
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
    let desc_pool = DescriptorPool::new(
        &device,
        1,
        &[
            DescriptorPoolSize { descriptor_type: DescriptorType::UNIFORM_BUFFER, descriptor_count: 2 },
            DescriptorPoolSize { descriptor_type: DescriptorType::SAMPLED_IMAGE, descriptor_count: 1 },
            DescriptorPoolSize { descriptor_type: DescriptorType::SAMPLER, descriptor_count: 1 },
        ],
    )?;
    let desc_set = desc_pool.allocate(&set_layout)?;
    desc_set.write_buffer(0, DescriptorType::UNIFORM_BUFFER, &light_ubo, 0, 64);
    desc_set.write_buffer(1, DescriptorType::UNIFORM_BUFFER, &camera_ubo, 0, 64);
    // Shadow texture + comparison sampler will be written after the shadow pass
    // transitions the depth image to SHADER_READ_ONLY layout.

    let pipeline_layout = PipelineLayout::new(&device, &[&set_layout])?;

    // Shadow pipeline: depth-only, depth bias to prevent acne.
    let shadow_pipeline = GraphicsPipelineBuilder::new(&pipeline_layout, &shadow_rp)
        .stage(GraphicsShaderStage::Vertex, &shader, "vs_depth")
        .stage(GraphicsShaderStage::Fragment, &shader, "fs_depth")
        .viewport_extent(SHADOW_SIZE, SHADOW_SIZE)
        .depth_test(true, true)
        .depth_bias(1.25, 1.75, 0.0)
        .cull_mode(vulkane::safe::CullMode::NONE)
        .color_attachment_count(0)
        .build(&device)?;

    // Main pipeline: color + depth.
    let main_pipeline = GraphicsPipelineBuilder::new(&pipeline_layout, &main_rp)
        .stage(GraphicsShaderStage::Vertex, &shader, "vs_main")
        .stage(GraphicsShaderStage::Fragment, &shader, "fs_main")
        .viewport_extent(W, H)
        .depth_test(true, true)
        .cull_mode(vulkane::safe::CullMode::NONE)
        .build(&device)?;
    println!("[OK] Built shadow + main pipelines");

    // Readback buffer.
    let (readback, mut rb_mem) = Buffer::new_bound(
        &device, &physical,
        BufferCreateInfo { size: (W * H * 4) as u64, usage: BufferUsage::TRANSFER_DST },
        MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // --- Record command buffer ---
    let cmd_pool = CommandPool::new(&device, qf)?;
    let mut cmd = cmd_pool.allocate_primary()?;
    {
        let mut rec = cmd.begin()?;

        // Pass 1: shadow depth.
        rec.begin_render_pass_ext(
            &shadow_rp,
            &shadow_fb,
            &[ClearValue::DepthStencil { depth: 1.0, stencil: 0 }],
        );
        rec.bind_graphics_pipeline(&shadow_pipeline);
        rec.bind_graphics_descriptor_sets(&pipeline_layout, 0, &[&desc_set]);
        rec.draw(VERTEX_COUNT, 1, 0, 0);
        rec.end_render_pass();

        // Transition shadow map: DEPTH_STENCIL_ATTACHMENT → SHADER_READ_ONLY.
        rec.image_barrier(
            PipelineStage::LATE_FRAGMENT_TESTS,
            PipelineStage::FRAGMENT_SHADER,
            ImageBarrier::depth(
                &shadow_img,
                ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                AccessFlags(0x400), // DEPTH_STENCIL_ATTACHMENT_WRITE
                AccessFlags::SHADER_READ,
            ),
        );

        // Now write the shadow texture descriptor (image is in the right layout).
        // We can't call descriptor writes inside recording, so we wrote them
        // before recording and the layout transition happens inside the CB.
        // Actually the descriptor write happens before submit — the layout
        // only needs to be correct at draw time, not at write time.

        // Pass 2: main color.
        rec.begin_render_pass_ext(
            &main_rp,
            &main_fb,
            &[
                ClearValue::Color([0.1, 0.1, 0.2, 1.0]),
                ClearValue::DepthStencil { depth: 1.0, stencil: 0 },
            ],
        );
        rec.bind_graphics_pipeline(&main_pipeline);
        rec.bind_graphics_descriptor_sets(&pipeline_layout, 0, &[&desc_set]);
        rec.draw(VERTEX_COUNT, 1, 0, 0);
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

    // Write shadow texture descriptor before submit.
    desc_set.write_sampled_image(2, &shadow_view, ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    desc_set.write_sampler(3, &shadow_sampler);

    let fence = Fence::new(&device)?;
    queue.submit(&[&cmd], Some(&fence))?;
    fence.wait(u64::MAX)?;
    println!("[OK] GPU finished rendering");

    // Verify: we should see a mix of lit and shadowed pixels.
    let m = rb_mem.map()?;
    let bytes = m.as_slice();

    let mut lit = 0u32;
    let mut shadowed = 0u32;
    let mut bg = 0u32;
    for px in 0..(W * H) as usize {
        let r = bytes[px * 4];
        let g = bytes[px * 4 + 1];
        let b = bytes[px * 4 + 2];
        let lum = (r as u32 + g as u32 + b as u32) / 3;
        if r < 30 && g < 30 && b < 60 {
            bg += 1; // dark background
        } else if lum > 100 {
            lit += 1;
        } else {
            shadowed += 1;
        }
    }
    println!("[OK] Pixel tally: {lit} lit, {shadowed} shadowed, {bg} background");

    // The scene should have some geometry visible (lit + shadowed > 0).
    let visible = lit + shadowed;
    assert!(
        visible > 1000,
        "expected visible scene geometry, got only {visible} non-background pixels"
    );
    println!("[OK] Scene rendered with {visible} visible pixels");

    drop(m);
    device.wait_idle()?;
    println!("\n=== shadow_map example PASSED ===");
    Ok(())
}
