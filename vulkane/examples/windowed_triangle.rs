//! Windowed triangle: opens a real OS window via winit and renders the
//! same RGB triangle as `headless_triangle.rs` but to a swapchain
//! image instead of an offscreen attachment.
//!
//! Run with: `cargo run -p vulkane --features fetch-spec --example windowed_triangle`
//!
//! Press `Esc` or close the window to exit.
//!
//! ## What this exercises
//!
//! Everything in `headless_triangle` plus:
//!
//! - Win32 / Wayland / Metal surface creation via the platform
//!   constructors on `vulkane::safe::Surface`.
//! - `Swapchain` creation, image enumeration, and per-image framebuffer
//!   construction.
//! - The standard acquire / submit / present semaphore loop.
//! - Resize handling (recreating the swapchain when the window changes
//!   size — the standard `ERROR_OUT_OF_DATE_KHR` path).

use std::error::Error;
use std::sync::Arc;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use vulkane::raw::bindings::{
    EXT_METAL_SURFACE_EXTENSION_NAME, KHR_WAYLAND_SURFACE_EXTENSION_NAME,
    KHR_WIN32_SURFACE_EXTENSION_NAME, KHR_XCB_SURFACE_EXTENSION_NAME,
    KHR_XLIB_SURFACE_EXTENSION_NAME,
};
use vulkane::safe::{
    ApiVersion, AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, CommandBuffer,
    CommandPool, DeviceCreateInfo, DeviceExtensions, Fence, Framebuffer, GraphicsPipeline,
    GraphicsPipelineBuilder, GraphicsShaderStage, ImageLayout, ImageUsage, ImageView, Instance,
    InstanceCreateInfo, InstanceExtensions, PipelineLayout, PipelineStage, PresentMode,
    QueueCreateInfo, QueueFlags, RenderPass, RenderPassCreateInfo, Semaphore, ShaderModule,
    SignalSemaphore, Surface, Swapchain, SwapchainCreateInfo, WaitSemaphore,
};

const TITLE: &str = "vulkane — windowed triangle";

/// Per-frame state we keep alongside the swapchain.
struct FrameSync {
    image_available: Semaphore,
    render_finished: Semaphore,
    in_flight: Fence,
}

struct Renderer {
    _instance: Instance,
    surface: Arc<Surface>,
    physical: vulkane::safe::PhysicalDevice,
    device: vulkane::safe::Device,
    queue: vulkane::safe::Queue,
    queue_family: u32,
    render_pass: RenderPass,
    pipeline_layout: PipelineLayout,
    pipeline: GraphicsPipeline,
    swapchain: Swapchain,
    image_views: Vec<ImageView>,
    /// Held alive: each command buffer references one of these.
    #[allow(dead_code)]
    framebuffers: Vec<Framebuffer>,
    cmd_pool: CommandPool,
    cmd_buffers: Vec<CommandBuffer>,
    frames: Vec<FrameSync>,
    current_frame: usize,
    extent: (u32, u32),
}

impl Renderer {
    fn new(window: &Window) -> Result<Self, Box<dyn Error>> {
        // 1. Pick the right surface extensions for the host platform.
        let raw_display = window.display_handle()?.as_raw();
        let raw_window = window.window_handle()?.as_raw();
        let surface_ext_name = surface_extension_for(&raw_display)?;

        // 2. Build the instance with KHR_surface + the platform surface ext.
        let instance_exts = InstanceExtensions::new()
            .khr_surface()
            .enable_raw(surface_ext_name);
        let instance = Instance::new(InstanceCreateInfo {
            application_name: Some(TITLE),
            api_version: ApiVersion::V1_0,
            enabled_extensions: Some(&instance_exts),
            ..InstanceCreateInfo::default()
        })?;

        // 3. Create the platform-specific surface.
        let surface = unsafe { make_surface(&instance, raw_display, raw_window)? };
        let surface = Arc::new(surface);

        // 4. Pick a physical device with a graphics+present queue family.
        let physicals = instance.enumerate_physical_devices()?;
        let mut chosen: Option<(vulkane::safe::PhysicalDevice, u32)> = None;
        for pd in physicals {
            for (i, qf) in pd.queue_family_properties().iter().enumerate() {
                if qf.queue_flags().contains(QueueFlags::GRAPHICS)
                    && surface.supports_present(&pd, i as u32)
                {
                    chosen = Some((pd, i as u32));
                    break;
                }
            }
            if chosen.is_some() {
                break;
            }
        }
        let (physical, queue_family) =
            chosen.ok_or("no graphics+present queue family on any device")?;
        println!("Using GPU: {}", physical.properties().device_name());

        // 5. Create the device with KHR_swapchain enabled.
        let device_exts = DeviceExtensions::new().khr_swapchain();
        let device = physical.create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo::single(queue_family)],
            enabled_extensions: Some(&device_exts),
            ..Default::default()
        })?;
        let queue = device.get_queue(queue_family, 0);

        // 6. Pick a surface format and create the swapchain.
        let (format, color_space) = Swapchain::pick_surface_format(&surface, &physical)?;
        let caps = surface.capabilities(&physical)?;
        let win_size = window.inner_size();
        let extent = clamp_extent(
            (win_size.width, win_size.height),
            caps.min_image_extent(),
            caps.max_image_extent(),
        );
        let min_image_count = caps.min_image_count().max(2);
        let swapchain = Swapchain::new(
            &device,
            &surface,
            SwapchainCreateInfo {
                format,
                color_space,
                width: extent.0,
                height: extent.1,
                min_image_count,
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                present_mode: PresentMode::FIFO,
                clipped: true,
            },
        )?;
        let image_views = swapchain.image_views()?;

        // 7. Render pass: present-source final layout, clear on load.
        let render_pass = RenderPass::new(
            &device,
            RenderPassCreateInfo {
                color_attachments: &[AttachmentDescription {
                    format,
                    load_op: AttachmentLoadOp::CLEAR,
                    store_op: AttachmentStoreOp::STORE,
                    initial_layout: ImageLayout::UNDEFINED,
                    final_layout: ImageLayout::PRESENT_SRC_KHR,
                }],
                depth_attachment: None,
            },
        )?;

        // 8. Per-image framebuffers.
        let framebuffers: Vec<Framebuffer> = image_views
            .iter()
            .map(|v| Framebuffer::new(&device, &render_pass, &[v], extent.0, extent.1))
            .collect::<Result<_, _>>()?;

        // 9. Pipeline.
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let vert_bytes =
            std::fs::read(format!("{manifest_dir}/examples/shaders/triangle.vert.spv"))?;
        let frag_bytes =
            std::fs::read(format!("{manifest_dir}/examples/shaders/triangle.frag.spv"))?;
        let vert = ShaderModule::from_spirv_bytes(&device, &vert_bytes)?;
        let frag = ShaderModule::from_spirv_bytes(&device, &frag_bytes)?;
        let pipeline_layout = PipelineLayout::new(&device, &[])?;
        let pipeline = GraphicsPipelineBuilder::new(&pipeline_layout, &render_pass)
            .stage(GraphicsShaderStage::Vertex, &vert, "main")
            .stage(GraphicsShaderStage::Fragment, &frag, "main")
            .viewport_extent(extent.0, extent.1)
            .cull_mode(vulkane::safe::CullMode::NONE)
            .build(&device)?;

        // 10. Command buffers — one per swapchain image, recorded once.
        let cmd_pool = CommandPool::new(&device, queue_family)?;
        let mut cmd_buffers: Vec<CommandBuffer> = Vec::new();
        for (i, fb) in framebuffers.iter().enumerate() {
            let mut cb = cmd_pool.allocate_primary()?;
            {
                let mut rec = cb.begin()?;
                rec.begin_render_pass(&render_pass, fb, &[[0.0, 0.0, 0.0, 1.0]]);
                rec.bind_graphics_pipeline(&pipeline);
                rec.draw(3, 1, 0, 0);
                rec.end_render_pass();
                rec.end()?;
            }
            cmd_buffers.push(cb);
            // Suppress unused-variable warning if we ever loop without
            // doing per-image work.
            let _ = i;
        }

        // 11. Per-in-flight-frame sync objects.
        let frames_in_flight = 2;
        let mut frames = Vec::with_capacity(frames_in_flight);
        for _ in 0..frames_in_flight {
            frames.push(FrameSync {
                image_available: Semaphore::binary(&device)?,
                render_finished: Semaphore::binary(&device)?,
                in_flight: Fence::new(&device)?,
            });
        }

        Ok(Self {
            _instance: instance,
            surface,
            physical,
            device,
            queue,
            queue_family,
            render_pass,
            pipeline_layout,
            pipeline,
            swapchain,
            image_views,
            framebuffers,
            cmd_pool,
            cmd_buffers,
            frames,
            current_frame: 0,
            extent,
        })
    }

    fn draw_frame(&mut self) -> Result<(), Box<dyn Error>> {
        let frame = &self.frames[self.current_frame];
        // Wait for this frame's previous submission to finish.
        frame.in_flight.wait(u64::MAX)?;
        frame.in_flight.reset()?;

        let image_index =
            match self
                .swapchain
                .acquire_next_image(u64::MAX, Some(&frame.image_available), None)
            {
                Ok(i) => i,
                Err(_) => {
                    // Out of date / suboptimal — recreate on next frame.
                    return Ok(());
                }
            };

        let cmd = &self.cmd_buffers[image_index as usize];
        self.queue.submit_with_sync(
            &[cmd],
            &[WaitSemaphore::binary(&frame.image_available, PipelineStage::COLOR_ATTACHMENT_OUTPUT)],
            &[SignalSemaphore::binary(&frame.render_finished)],
            Some(&frame.in_flight),
        )?;

        self.swapchain
            .present(&self.queue, image_index, &[&frame.render_finished])?;
        self.current_frame = (self.current_frame + 1) % self.frames.len();
        Ok(())
    }

    fn cleanup(&self) {
        let _ = self.device.wait_idle();
        // Suppress unused-field warnings (they're held alive by self).
        let _ = (
            &self.physical,
            self.queue_family,
            &self.pipeline_layout,
            &self.pipeline,
            &self.image_views,
            &self.cmd_pool,
            &self.render_pass,
            &self.surface,
            self.extent,
        );
    }
}

/// App: holds the window and renderer, drives the event loop.
#[derive(Default)]
struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = winit::window::Window::default_attributes()
            .with_title(TITLE)
            .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };
        match Renderer::new(&window) {
            Ok(r) => self.renderer = Some(r),
            Err(e) => {
                eprintln!("failed to initialize renderer: {e}");
                event_loop.exit();
                return;
            }
        }
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                if let Some(r) = self.renderer.as_ref() {
                    r.cleanup();
                }
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == winit::event::ElementState::Pressed
                    && matches!(
                        event.physical_key,
                        winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape)
                    )
                {
                    if let Some(r) = self.renderer.as_ref() {
                        r.cleanup();
                    }
                    event_loop.exit();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(r) = self.renderer.as_mut() {
                    if let Err(e) = r.draw_frame() {
                        eprintln!("draw error: {e}");
                    }
                }
                if let Some(w) = self.window.as_ref() {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn surface_extension_for(display: &RawDisplayHandle) -> Result<&'static str, Box<dyn Error>> {
    Ok(match display {
        RawDisplayHandle::Windows(_) => KHR_WIN32_SURFACE_EXTENSION_NAME,
        RawDisplayHandle::Wayland(_) => KHR_WAYLAND_SURFACE_EXTENSION_NAME,
        RawDisplayHandle::Xlib(_) => KHR_XLIB_SURFACE_EXTENSION_NAME,
        RawDisplayHandle::Xcb(_) => KHR_XCB_SURFACE_EXTENSION_NAME,
        RawDisplayHandle::AppKit(_) => EXT_METAL_SURFACE_EXTENSION_NAME,
        other => return Err(format!("unsupported display handle: {other:?}").into()),
    })
}

unsafe fn make_surface(
    instance: &Instance,
    display: RawDisplayHandle,
    window: RawWindowHandle,
) -> Result<Surface, Box<dyn Error>> {
    match (display, window) {
        (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(handle)) => {
            // hinstance is required by vkCreateWin32SurfaceKHR; use the
            // current process's module handle when winit doesn't supply
            // one (winit 0.30 includes it on the Win32 handle).
            let hinstance = handle
                .hinstance
                .map(|h| h.get() as *mut std::ffi::c_void)
                .unwrap_or(std::ptr::null_mut());
            let hwnd = handle.hwnd.get() as *mut std::ffi::c_void;
            Ok(unsafe { Surface::from_win32(instance, hinstance, hwnd) }?)
        }
        (RawDisplayHandle::Wayland(d), RawWindowHandle::Wayland(w)) => Ok(unsafe {
            Surface::from_wayland(instance, d.display.as_ptr(), w.surface.as_ptr())
        }?),
        (RawDisplayHandle::Xlib(d), RawWindowHandle::Xlib(w)) => {
            // winit gives Display as an Option (it can be None when the
            // window was opened on the default display). Use null in that
            // case — Vulkan will pick the default in turn.
            let display = d
                .display
                .map(|p| p.as_ptr())
                .unwrap_or(std::ptr::null_mut());
            Ok(unsafe { Surface::from_xlib(instance, display, w.window) }?)
        }
        (RawDisplayHandle::Xcb(d), RawWindowHandle::Xcb(w)) => {
            let connection = d
                .connection
                .map(|p| p.as_ptr())
                .unwrap_or(std::ptr::null_mut());
            Ok(unsafe { Surface::from_xcb(instance, connection, w.window.get()) }?)
        }
        (RawDisplayHandle::AppKit(_), RawWindowHandle::AppKit(_)) => {
            Err("macOS Metal surface support requires creating a CAMetalLayer; not implemented in this example".into())
        }
        other => Err(format!("unsupported window handle combination: {other:?}").into()),
    }
}

fn clamp_extent(requested: (u32, u32), min: (u32, u32), max: (u32, u32)) -> (u32, u32) {
    (
        requested.0.clamp(min.0, max.0),
        requested.1.clamp(min.1, max.1),
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}
