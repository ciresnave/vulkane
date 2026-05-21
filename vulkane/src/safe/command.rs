//! Safe wrapper for `VkCommandPool` and `VkCommandBuffer`.
//!
//! All GPU work in Vulkan is recorded into command buffers before
//! being submitted to a queue. The typical pattern is:
//!
//! 1. Create a [`CommandPool`] from a queue family index.
//! 2. Allocate a [`CommandBuffer`] from the pool.
//! 3. Call [`cmd.begin()`](CommandBuffer::begin) to get a
//!    [`CommandBufferRecording`] handle.
//! 4. Record commands: barriers, copies, dispatches, draws.
//! 5. Call [`rec.end()`](CommandBufferRecording::end).
//! 6. Submit with [`Queue::submit`](super::Queue::submit).
//!
//! For fire-and-forget work (uploads, transitions),
//! [`Queue::one_shot`](super::Queue::one_shot) wraps steps 1–6 into a
//! single closure call.
//!
//! ## Key recording methods
//!
//! - **Barriers:** [`memory_barrier`](CommandBufferRecording::memory_barrier),
//!   [`image_barrier`](CommandBufferRecording::image_barrier) (typed
//!   [`PipelineStage`](super::PipelineStage) / [`AccessFlags`](super::AccessFlags))
//! - **Copies:** [`copy_buffer`](CommandBufferRecording::copy_buffer),
//!   [`copy_buffer_to_image`](CommandBufferRecording::copy_buffer_to_image),
//!   [`copy_image_to_buffer`](CommandBufferRecording::copy_image_to_buffer),
//!   [`fill_buffer`](CommandBufferRecording::fill_buffer)
//! - **Compute:** [`bind_compute_pipeline`](CommandBufferRecording::bind_compute_pipeline),
//!   [`dispatch`](CommandBufferRecording::dispatch)
//! - **Graphics:** [`begin_render_pass`](CommandBufferRecording::begin_render_pass),
//!   [`bind_graphics_pipeline`](CommandBufferRecording::bind_graphics_pipeline),
//!   [`draw`](CommandBufferRecording::draw),
//!   [`draw_indexed`](CommandBufferRecording::draw_indexed)

use super::acceleration_structure::{
    AccelerationStructure, AccelerationStructureBuildFlags, AccelerationStructureBuildMode,
    AccelerationStructureGeometry, AccelerationStructureType, BuildRange,
};
use super::descriptor::{DescriptorSet, ShaderStageFlags};
use super::device::DeviceInner;
use super::flags::{AccessFlags, AccessFlags2, PipelineStage, PipelineStage2};
use super::graphics_pipeline::GraphicsPipeline;
use super::image::{BufferImageCopy, Image, ImageBarrier, ImageLayout, ImageView, Sampler};
use super::pipeline::{ComputePipeline, PipelineLayout};
use super::query::QueryPool;
use super::ray_tracing_pipeline::{RayTracingPipeline, ShaderBindingRegion};
use super::render_pass::{Framebuffer, RenderPass};
use super::{Buffer, Device, Error, Result, check};
use crate::raw::bindings::*;
use std::sync::Arc;

/// Which pipeline type a command applies to. Mirrors the subset of
/// `VkPipelineBindPoint` that the safe wrapper actually supports
/// (compute and graphics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineBindPoint {
    Graphics,
    Compute,
}

impl PipelineBindPoint {
    #[inline]
    fn to_raw(self) -> VkPipelineBindPoint {
        match self {
            Self::Graphics => VkPipelineBindPoint::PIPELINE_BIND_POINT_GRAPHICS,
            Self::Compute => VkPipelineBindPoint::PIPELINE_BIND_POINT_COMPUTE,
        }
    }
}

/// A single descriptor write recorded by
/// [`CommandBufferRecording::push_descriptor_set`].
///
/// Each variant corresponds to one of the descriptor types currently
/// supported by the safe wrapper (the same set covered by
/// [`DescriptorSet::write_buffer`](super::DescriptorSet::write_buffer)
/// and friends). The enum is borrow-only — buffers, image views, and
/// samplers are referenced by `&`, so the push-descriptor call must
/// complete before any of the referenced resources is dropped.
#[derive(Clone, Copy)]
pub enum PushDescriptorWrite<'a> {
    /// `STORAGE_BUFFER` — storage-readwrite in compute shaders.
    StorageBuffer {
        binding: u32,
        buffer: &'a Buffer,
        offset: u64,
        range: u64,
    },
    /// `UNIFORM_BUFFER` — read-only UBO.
    UniformBuffer {
        binding: u32,
        buffer: &'a Buffer,
        offset: u64,
        range: u64,
    },
    /// `STORAGE_IMAGE` — shader-storage image (image load/store).
    StorageImage {
        binding: u32,
        view: &'a ImageView,
        layout: ImageLayout,
    },
    /// `SAMPLED_IMAGE` — textures in the separated-sampler style
    /// (paired with a separate `SAMPLER` binding).
    SampledImage {
        binding: u32,
        view: &'a ImageView,
        layout: ImageLayout,
    },
    /// `COMBINED_IMAGE_SAMPLER` — one binding holds both the view and
    /// its sampler (OpenGL-style `texture2D`).
    CombinedImageSampler {
        binding: u32,
        sampler: &'a Sampler,
        view: &'a ImageView,
        layout: ImageLayout,
    },
    /// `SAMPLER` — standalone sampler (paired with
    /// [`SampledImage`](Self::SampledImage) at a different binding).
    Sampler { binding: u32, sampler: &'a Sampler },
}

/// A clear value for [`begin_render_pass_ext`](CommandBufferRecording::begin_render_pass_ext).
///
/// Use [`Color`](Self::Color) for color attachments and
/// [`DepthStencil`](Self::DepthStencil) for depth/stencil attachments.
#[derive(Debug, Clone, Copy)]
pub enum ClearValue {
    /// RGBA float clear color (e.g. `[0.0, 0.0, 0.0, 1.0]` for black).
    Color([f32; 4]),
    /// Depth + stencil clear (e.g. `depth: 1.0, stencil: 0`).
    DepthStencil { depth: f32, stencil: u32 },
}

/// Description of a single attachment for
/// [`CommandBufferRecording::begin_rendering`] — the dynamic-rendering
/// replacement for per-subpass `VkRenderPass` attachment references.
///
/// Unlike the classic render-pass model, dynamic rendering has no
/// "attachment description" object: each draw-time attachment carries
/// its load/store ops and clear value directly. One
/// [`RenderingAttachment`] maps to one `VkRenderingAttachmentInfo`.
#[derive(Clone, Copy)]
pub struct RenderingAttachment<'a> {
    /// The image view to render into.
    pub view: &'a ImageView,
    /// The image's layout during rendering — typically
    /// [`ImageLayout::COLOR_ATTACHMENT_OPTIMAL`](super::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
    /// for color and
    /// [`ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL`](super::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    /// for depth/stencil.
    pub layout: ImageLayout,
    /// What to do with existing contents of the attachment at
    /// render-begin.
    pub load_op: super::AttachmentLoadOp,
    /// What to do with the attachment's contents at render-end.
    pub store_op: super::AttachmentStoreOp,
    /// Value to clear to when `load_op == CLEAR`. Ignored otherwise.
    pub clear_value: Option<ClearValue>,
}

/// Parameters for [`CommandBufferRecording::begin_rendering`] — the
/// `VK_KHR_dynamic_rendering` (core in 1.3) replacement for
/// `VkRenderPassBeginInfo`.
///
/// Dynamic rendering eliminates `VkRenderPass` and `VkFramebuffer`
/// objects entirely: the attachments and their load/store ops are
/// supplied at `begin_rendering` time, which matches how modern
/// graphics APIs (Metal, DX12, WebGPU) express render targets.
#[derive(Clone, Copy)]
pub struct RenderingInfo<'a> {
    /// Render area, in pixels. Typically
    /// `{ offset: (0,0), extent: (width, height) }` of the smallest
    /// attachment.
    pub render_area: VkRect2D,
    /// Number of layers to render to. `1` for non-layered rendering.
    pub layer_count: u32,
    /// Bitmask of view indices to render in a single pass when
    /// multiview is enabled; `0` disables multiview.
    pub view_mask: u32,
    /// Color attachments. Empty slice means depth-only rendering.
    pub color_attachments: &'a [RenderingAttachment<'a>],
    /// Optional depth attachment.
    pub depth_attachment: Option<RenderingAttachment<'a>>,
    /// Optional stencil attachment.
    pub stencil_attachment: Option<RenderingAttachment<'a>>,
}

/// One bound descriptor buffer for
/// [`CommandBufferRecording::bind_descriptor_buffers`] —
/// `VK_EXT_descriptor_buffer`.
///
/// A descriptor buffer is any `VkBuffer` allocated with the
/// `RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT` or
/// `SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT` usage flag, containing descriptor
/// bytes written via `vkGetDescriptorEXT`. At record time the caller
/// passes up to `maxDescriptorBufferBindings` of them in a single
/// command, then addresses individual sets with
/// [`CommandBufferRecording::set_descriptor_buffer_offsets`].
#[derive(Debug, Clone, Copy)]
pub struct DescriptorBufferBinding {
    /// GPU virtual address of the buffer — typically
    /// [`Buffer::device_address`](super::Buffer::device_address). The
    /// buffer must have been created with
    /// `BufferUsage::SHADER_DEVICE_ADDRESS` and one of the descriptor
    /// buffer usage bits.
    pub address: u64,
    /// Usage flags the driver should treat the buffer as carrying.
    /// Pass the raw `VkBufferUsageFlags` value —
    /// `RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT` (0x0020_0000),
    /// `SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT` (0x0040_0000), or their
    /// bitwise-OR.
    pub usage: u32,
}

/// One source -> destination region for [`CommandBufferRecording::copy_buffer`].
#[derive(Debug, Clone, Copy)]
pub struct BufferCopy {
    /// Byte offset in the source buffer.
    pub src_offset: u64,
    /// Byte offset in the destination buffer.
    pub dst_offset: u64,
    /// Number of bytes to copy.
    pub size: u64,
}

impl BufferCopy {
    /// Copy `size` bytes from offset 0 in both source and destination.
    pub const fn full(size: u64) -> Self {
        Self {
            src_offset: 0,
            dst_offset: 0,
            size,
        }
    }
}

/// A safe wrapper around `VkCommandPool`.
///
/// Command pools are destroyed automatically on drop. The handle keeps the
/// parent device alive via an `Arc`.
pub struct CommandPool {
    pub(crate) handle: VkCommandPool,
    pub(crate) device: Arc<DeviceInner>,
}

impl CommandPool {
    /// Create a new command pool that allocates command buffers for the
    /// given queue family.
    pub fn new(device: &Device, queue_family_index: u32) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreateCommandPool
            .ok_or(Error::MissingFunction("vkCreateCommandPool"))?;

        let info = VkCommandPoolCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex: queue_family_index,
            ..Default::default()
        };

        let mut handle: VkCommandPool = 0;
        // Safety: info is valid for the call, device is valid.
        check(unsafe { create(device.inner.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
        })
    }

    /// Returns the raw `VkCommandPool` handle.
    pub fn raw(&self) -> VkCommandPool {
        self.handle
    }

    /// Allocate a single primary command buffer from this pool.
    pub fn allocate_primary(&self) -> Result<CommandBuffer> {
        let allocate = self
            .device
            .dispatch
            .vkAllocateCommandBuffers
            .ok_or(Error::MissingFunction("vkAllocateCommandBuffers"))?;

        let info = VkCommandBufferAllocateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool: self.handle,
            level: VkCommandBufferLevel::COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
            ..Default::default()
        };

        let mut handle: VkCommandBuffer = std::ptr::null_mut();
        // Safety: info is valid for the call, output handle slot is valid.
        check(unsafe { allocate(self.device.handle, &info, &mut handle) })?;

        Ok(CommandBuffer {
            handle,
            device: Arc::clone(&self.device),
            // Keep the pool alive so the command buffer can be freed on drop.
            pool: self.handle,
        })
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroyCommandPool {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}

/// A safe wrapper around `VkCommandBuffer`.
///
/// Command buffers are freed back to their parent pool on drop.
///
/// To record commands, call [`begin`](Self::begin), which returns a
/// [`CommandBufferRecording`] guard. Recording is finished automatically
/// when the guard is dropped (or you can call [`end`](CommandBufferRecording::end)
/// explicitly to detect errors).
///
/// # Thread safety
///
/// `CommandBuffer` is `Send + Sync`. Recording APIs ([`begin`](Self::begin)
/// and the resulting [`CommandBufferRecording`]) take `&mut self`, so the
/// Rust borrow checker already prevents concurrent recording on the same
/// buffer. Sharing `&CommandBuffer` across threads (e.g. recording on one
/// thread and submitting on another) is sound.
///
/// The Vulkan spec also requires external synchronization per parent pool
/// for `vkFreeCommandBuffers`/`vkResetCommandPool`; this wrapper's `Drop`
/// calls `vkFreeCommandBuffers`, so do not drop a `CommandBuffer` while
/// another thread is resetting its pool.
pub struct CommandBuffer {
    pub(crate) handle: VkCommandBuffer,
    pub(crate) device: Arc<DeviceInner>,
    /// We need the pool handle to call `vkFreeCommandBuffers`. The pool itself
    /// is kept alive by the user holding a [`CommandPool`] — but the
    /// `Arc<DeviceInner>` keeps the device alive long enough for our `Drop`
    /// impl to run.
    #[allow(dead_code)]
    pool: VkCommandPool,
}

// SAFETY: `VkCommandBuffer` is a dispatchable handle (an opaque pointer),
// so the auto-derived `Send`/`Sync` would be missing. The handle has no
// thread affinity. Recording entry points on this wrapper take `&mut self`,
// so the Rust type system enforces single-threaded recording without us
// needing to forbid `Sync`. Callers remain responsible for the per-pool
// external-sync contract documented on `CommandBuffer`.
unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

impl CommandBuffer {
    /// Returns the raw `VkCommandBuffer` handle.
    pub fn raw(&self) -> VkCommandBuffer {
        self.handle
    }

    /// Begin recording commands. Returns a guard that finishes recording
    /// when dropped.
    pub fn begin(&mut self) -> Result<CommandBufferRecording<'_>> {
        let begin = self
            .device
            .dispatch
            .vkBeginCommandBuffer
            .ok_or(Error::MissingFunction("vkBeginCommandBuffer"))?;

        let info = VkCommandBufferBeginInfo {
            sType: VkStructureType::STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            ..Default::default()
        };

        // Safety: handle is valid, info is valid for the call.
        check(unsafe { begin(self.handle, &info) })?;

        Ok(CommandBufferRecording { buffer: self })
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        if let Some(free) = self.device.dispatch.vkFreeCommandBuffers {
            // Safety: handles are valid; we are the sole owner of this command buffer.
            unsafe { free(self.device.handle, self.pool, 1, &self.handle) };
        }
    }
}

/// RAII guard returned by [`CommandBuffer::begin`].
///
/// Recording commands while this guard is alive. Drop calls
/// `vkEndCommandBuffer`. Use [`end`](Self::end) explicitly if you need to
/// detect errors from `vkEndCommandBuffer`.
pub struct CommandBufferRecording<'a> {
    buffer: &'a mut CommandBuffer,
}

impl<'a> CommandBufferRecording<'a> {
    /// Raw `VkCommandBuffer` currently being recorded. Used by the
    /// auto-generated `CommandBufferRecordingExt` impl; the `pub(crate)`
    /// visibility keeps it off the public API surface while letting
    /// generated code inside `vulkane::safe::auto` reach it.
    #[inline]
    pub(crate) fn raw_cmd(&self) -> VkCommandBuffer {
        self.buffer.handle
    }

    /// Raw `VkDeviceDispatchTable` for the device this command buffer
    /// belongs to.
    #[inline]
    pub(crate) fn device_dispatch(&self) -> &VkDeviceDispatchTable {
        &self.buffer.device.dispatch
    }

    /// Record `vkCmdFillBuffer`: fill `size` bytes of `buffer` starting at
    /// `dst_offset` with the constant `data`.
    pub fn fill_buffer(&mut self, buffer: &Buffer, dst_offset: u64, size: u64, data: u32) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdFillBuffer
            .expect("vkCmdFillBuffer is required by Vulkan 1.0");
        // Safety: command buffer is in recording state, buffer handle is valid.
        unsafe { cmd(self.buffer.handle, buffer.handle, dst_offset, size, data) };
    }

    /// Record `vkCmdBindPipeline` for a compute pipeline.
    pub fn bind_compute_pipeline(&mut self, pipeline: &ComputePipeline) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBindPipeline
            .expect("vkCmdBindPipeline is required by Vulkan 1.0");
        // Safety: command buffer is in recording state, pipeline handle is valid.
        // VK_PIPELINE_BIND_POINT_COMPUTE = 1
        unsafe {
            cmd(
                self.buffer.handle,
                VkPipelineBindPoint::PIPELINE_BIND_POINT_COMPUTE,
                pipeline.handle,
            )
        };
    }

    /// Record `vkCmdBindDescriptorSets` to bind one or more descriptor sets
    /// to a compute pipeline starting at the given set number.
    pub fn bind_compute_descriptor_sets(
        &mut self,
        layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[&DescriptorSet],
    ) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBindDescriptorSets
            .expect("vkCmdBindDescriptorSets is required by Vulkan 1.0");

        let raw: Vec<VkDescriptorSet> = descriptor_sets.iter().map(|s| s.handle).collect();
        // Safety: command buffer is in recording state, layout and sets are valid,
        // raw lives for the duration of the call.
        unsafe {
            cmd(
                self.buffer.handle,
                VkPipelineBindPoint::PIPELINE_BIND_POINT_COMPUTE,
                layout.handle,
                first_set,
                raw.len() as u32,
                raw.as_ptr(),
                0,
                std::ptr::null(),
            )
        };
    }

    /// Record `vkCmdPushDescriptorSetKHR` — `VK_KHR_push_descriptor`.
    ///
    /// Push descriptors bypass the descriptor-pool + `VkDescriptorSet`
    /// lifecycle entirely: the writes are embedded directly in the
    /// command buffer and consumed by the driver at execution time.
    /// There is nothing to allocate, free, or reset.
    ///
    /// Typical use cases:
    ///
    /// - Per-dispatch descriptor updates that would otherwise need a
    ///   fresh `DescriptorSet` per frame (dynamic per-object resources).
    /// - Small ML graphs where allocating and retiring a
    ///   `DescriptorPool` per dispatch would dominate the workload.
    ///
    /// Requirements:
    ///
    /// - `VK_KHR_push_descriptor` enabled on the device.
    /// - The `PipelineLayout` was created with a descriptor-set layout
    ///   whose `flags` include `PUSH_DESCRIPTOR_BIT_KHR` at the
    ///   corresponding set index.
    ///
    /// Returns `MissingFunction("vkCmdPushDescriptorSetKHR")` if the
    /// extension wasn't enabled.
    pub fn push_descriptor_set(
        &mut self,
        bind_point: PipelineBindPoint,
        layout: &PipelineLayout,
        set: u32,
        writes: &[PushDescriptorWrite<'_>],
    ) -> Result<()> {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdPushDescriptorSet
            .ok_or(Error::MissingFunction("vkCmdPushDescriptorSetKHR"))?;

        // Owning backing storage for per-write `VkDescriptor{Buffer,Image}Info`
        // values — we need stable pointers into these for the duration of
        // the call, which is why they are allocated here rather than inside
        // the match arm.
        let mut buffer_infos: Vec<VkDescriptorBufferInfo> = Vec::with_capacity(writes.len());
        let mut image_infos: Vec<VkDescriptorImageInfo> = Vec::with_capacity(writes.len());
        let mut raw_writes: Vec<VkWriteDescriptorSet> = Vec::with_capacity(writes.len());

        // First pass: populate the owning Vecs. Capacity is pre-reserved
        // above so addresses of pushed elements are stable through the
        // loop.
        for w in writes {
            match w {
                PushDescriptorWrite::StorageBuffer {
                    buffer,
                    offset,
                    range,
                    ..
                }
                | PushDescriptorWrite::UniformBuffer {
                    buffer,
                    offset,
                    range,
                    ..
                } => {
                    buffer_infos.push(VkDescriptorBufferInfo {
                        buffer: buffer.handle,
                        offset: *offset,
                        range: *range,
                    });
                }
                PushDescriptorWrite::StorageImage { view, layout, .. }
                | PushDescriptorWrite::SampledImage { view, layout, .. } => {
                    image_infos.push(VkDescriptorImageInfo {
                        sampler: 0,
                        imageView: view.handle,
                        imageLayout: layout.0,
                    });
                }
                PushDescriptorWrite::CombinedImageSampler {
                    sampler,
                    view,
                    layout,
                    ..
                } => {
                    image_infos.push(VkDescriptorImageInfo {
                        sampler: sampler.handle,
                        imageView: view.handle,
                        imageLayout: layout.0,
                    });
                }
                PushDescriptorWrite::Sampler { sampler, .. } => {
                    image_infos.push(VkDescriptorImageInfo {
                        sampler: sampler.handle,
                        imageView: 0,
                        imageLayout: VkImageLayout::IMAGE_LAYOUT_UNDEFINED,
                    });
                }
            }
        }

        // Second pass: build the `VkWriteDescriptorSet` array, pointing
        // each entry at the matching slot in the owning Vecs by the
        // consumption counters below.
        let mut buf_i = 0usize;
        let mut img_i = 0usize;
        for w in writes {
            let (descriptor_type, p_buf, p_img) = match w {
                PushDescriptorWrite::StorageBuffer { .. } => {
                    let p = &buffer_infos[buf_i] as *const _;
                    buf_i += 1;
                    (
                        VkDescriptorType::DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        p,
                        std::ptr::null(),
                    )
                }
                PushDescriptorWrite::UniformBuffer { .. } => {
                    let p = &buffer_infos[buf_i] as *const _;
                    buf_i += 1;
                    (
                        VkDescriptorType::DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        p,
                        std::ptr::null(),
                    )
                }
                PushDescriptorWrite::StorageImage { .. } => {
                    let p = &image_infos[img_i] as *const _;
                    img_i += 1;
                    (
                        VkDescriptorType::DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        std::ptr::null(),
                        p,
                    )
                }
                PushDescriptorWrite::SampledImage { .. } => {
                    let p = &image_infos[img_i] as *const _;
                    img_i += 1;
                    (
                        VkDescriptorType::DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                        std::ptr::null(),
                        p,
                    )
                }
                PushDescriptorWrite::CombinedImageSampler { .. } => {
                    let p = &image_infos[img_i] as *const _;
                    img_i += 1;
                    (
                        VkDescriptorType::DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        std::ptr::null(),
                        p,
                    )
                }
                PushDescriptorWrite::Sampler { .. } => {
                    let p = &image_infos[img_i] as *const _;
                    img_i += 1;
                    (
                        VkDescriptorType::DESCRIPTOR_TYPE_SAMPLER,
                        std::ptr::null(),
                        p,
                    )
                }
            };
            let binding = match w {
                PushDescriptorWrite::StorageBuffer { binding, .. }
                | PushDescriptorWrite::UniformBuffer { binding, .. }
                | PushDescriptorWrite::StorageImage { binding, .. }
                | PushDescriptorWrite::SampledImage { binding, .. }
                | PushDescriptorWrite::CombinedImageSampler { binding, .. }
                | PushDescriptorWrite::Sampler { binding, .. } => *binding,
            };
            raw_writes.push(VkWriteDescriptorSet {
                sType: VkStructureType::STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet: 0, // Unused for push descriptors — the set is pushed by index.
                dstBinding: binding,
                dstArrayElement: 0,
                descriptorCount: 1,
                descriptorType: descriptor_type,
                pImageInfo: p_img,
                pBufferInfo: p_buf,
                ..Default::default()
            });
        }

        // Safety: command buffer is recording; layout handle is valid;
        // the owning `buffer_infos` / `image_infos` / `raw_writes` Vecs
        // live until end of this function and their addresses were
        // frozen before any raw_writes entry captured them (capacity
        // was reserved so no reallocation occurs during fill).
        unsafe {
            cmd(
                self.buffer.handle,
                bind_point.to_raw(),
                layout.handle,
                set,
                raw_writes.len() as u32,
                raw_writes.as_ptr(),
            )
        };
        Ok(())
    }

    /// Record `vkCmdDispatch` to launch a 3D grid of workgroups.
    pub fn dispatch(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdDispatch
            .expect("vkCmdDispatch is required by Vulkan 1.0");
        // Safety: command buffer is in recording state, a compute pipeline
        // and compatible descriptor sets must be bound (caller's responsibility).
        unsafe {
            cmd(
                self.buffer.handle,
                group_count_x,
                group_count_y,
                group_count_z,
            )
        };
    }

    /// Record `vkCmdDispatchIndirect`: read the workgroup count from a
    /// buffer at runtime.
    ///
    /// `buffer` must be a `INDIRECT_BUFFER` and at `offset` must contain a
    /// `VkDispatchIndirectCommand` (three `u32`s: x, y, z workgroup counts).
    /// The buffer is read on the GPU at dispatch time, so the host can pass
    /// counts that another shader wrote earlier in the same submission.
    pub fn dispatch_indirect(&mut self, buffer: &Buffer, offset: u64) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdDispatchIndirect
            .expect("vkCmdDispatchIndirect is required by Vulkan 1.0");
        // Safety: command buffer is in recording state, buffer handle is
        // valid and contains a VkDispatchIndirectCommand at the given offset
        // (caller's responsibility).
        unsafe { cmd(self.buffer.handle, buffer.handle, offset) };
    }

    /// Record `vkCmdCopyBuffer`: copy one or more byte regions from `src`
    /// to `dst`. Both buffers must have appropriate transfer usage flags.
    ///
    /// At least one region must be supplied.
    pub fn copy_buffer(&mut self, src: &Buffer, dst: &Buffer, regions: &[BufferCopy]) {
        debug_assert!(
            !regions.is_empty(),
            "vkCmdCopyBuffer requires at least one region"
        );
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdCopyBuffer
            .expect("vkCmdCopyBuffer is required by Vulkan 1.0");

        let raw: Vec<VkBufferCopy> = regions
            .iter()
            .map(|r| VkBufferCopy {
                srcOffset: r.src_offset,
                dstOffset: r.dst_offset,
                size: r.size,
            })
            .collect();

        // Safety: command buffer is in recording state, buffer handles are
        // valid, raw outlives the call.
        unsafe {
            cmd(
                self.buffer.handle,
                src.handle,
                dst.handle,
                raw.len() as u32,
                raw.as_ptr(),
            )
        };
    }

    /// Record `vkCmdPushConstants`: copy the `bytes` slice into the push
    /// constant range starting at `offset` in pipeline layout `layout` for
    /// the given shader `stages`.
    ///
    /// `offset` and `bytes.len()` must both be multiples of 4 and must lie
    /// entirely within a push constant range declared in `layout` for the
    /// given stages.
    pub fn push_constants(
        &mut self,
        layout: &PipelineLayout,
        stages: ShaderStageFlags,
        offset: u32,
        bytes: &[u8],
    ) {
        debug_assert!(
            offset % 4 == 0,
            "push constant offset must be a multiple of 4"
        );
        debug_assert!(
            bytes.len() % 4 == 0,
            "push constant size must be a multiple of 4"
        );
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdPushConstants
            .expect("vkCmdPushConstants is required by Vulkan 1.0");
        // Safety: command buffer is in recording state, layout is valid,
        // bytes lives for the duration of the call.
        unsafe {
            cmd(
                self.buffer.handle,
                layout.handle,
                stages.0,
                offset,
                bytes.len() as u32,
                bytes.as_ptr() as *const _,
            )
        };
    }

    /// Record `vkCmdResetQueryPool`: reset all queries in `pool` in the
    /// given range to the unavailable state. Must be called before any
    /// query in the range is used in this submission.
    pub fn reset_query_pool(&mut self, pool: &QueryPool, first_query: u32, query_count: u32) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdResetQueryPool
            .expect("vkCmdResetQueryPool is required by Vulkan 1.0");
        // Safety: command buffer is in recording state, pool handle is valid.
        unsafe { cmd(self.buffer.handle, pool.handle, first_query, query_count) };
    }

    /// Record `vkCmdWriteTimestamp`: write the current pipeline-stage
    /// timestamp into the given query slot.
    ///
    /// `pipeline_stage` should be a single [`PipelineStage`] bit (e.g.
    /// `PipelineStage::TOP_OF_PIPE`, `PipelineStage::BOTTOM_OF_PIPE`,
    /// `PipelineStage::COMPUTE_SHADER`).
    pub fn write_timestamp(&mut self, pipeline_stage: PipelineStage, pool: &QueryPool, query: u32) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdWriteTimestamp
            .expect("vkCmdWriteTimestamp is required by Vulkan 1.0");
        // Safety: command buffer is in recording state, pool is valid,
        // query slot must be in bounds (caller's responsibility).
        unsafe { cmd(self.buffer.handle, pipeline_stage.0, pool.handle, query) };
    }

    /// Record an image memory barrier (one-image, color-aspect, single-mip,
    /// single-layer). This is the simplified path that compute storage
    /// images use for layout transitions like
    /// `UNDEFINED -> GENERAL` (before writing) or
    /// `GENERAL -> TRANSFER_SRC_OPTIMAL` (before reading back).
    pub fn image_barrier(
        &mut self,
        src_stage: PipelineStage,
        dst_stage: PipelineStage,
        barrier: ImageBarrier<'_>,
    ) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdPipelineBarrier
            .expect("vkCmdPipelineBarrier is required by Vulkan 1.0");

        let raw = VkImageMemoryBarrier {
            sType: VkStructureType::STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask: barrier.src_access.0,
            dstAccessMask: barrier.dst_access.0,
            oldLayout: barrier.old_layout.0,
            newLayout: barrier.new_layout.0,
            srcQueueFamilyIndex: !0u32, // VK_QUEUE_FAMILY_IGNORED
            dstQueueFamilyIndex: !0u32,
            image: barrier.image.handle,
            subresourceRange: VkImageSubresourceRange {
                aspectMask: barrier.aspect_mask,
                baseMipLevel: 0,
                levelCount: 1,
                baseArrayLayer: 0,
                layerCount: 1,
            },
            ..Default::default()
        };

        // Safety: command buffer is in recording state, raw outlives the call.
        unsafe {
            cmd(
                self.buffer.handle,
                src_stage.0,
                dst_stage.0,
                0,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                1,
                &raw,
            )
        };
    }

    /// Record `vkCmdCopyBufferToImage`: copy bytes from a buffer into one
    /// or more regions of an image. The image must be in
    /// `TRANSFER_DST_OPTIMAL` (or `GENERAL`) layout.
    pub fn copy_buffer_to_image(
        &mut self,
        src: &Buffer,
        dst: &Image,
        dst_layout: super::ImageLayout,
        regions: &[BufferImageCopy],
    ) {
        debug_assert!(
            !regions.is_empty(),
            "vkCmdCopyBufferToImage requires at least one region"
        );
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdCopyBufferToImage
            .expect("vkCmdCopyBufferToImage is required by Vulkan 1.0");

        let raw: Vec<VkBufferImageCopy> = regions.iter().map(|r| r.to_raw()).collect();
        // Safety: handles are valid, raw outlives the call.
        unsafe {
            cmd(
                self.buffer.handle,
                src.handle,
                dst.handle,
                dst_layout.0,
                raw.len() as u32,
                raw.as_ptr(),
            )
        };
    }

    /// Record a global memory barrier using `vkCmdPipelineBarrier2`
    /// (Synchronization2 — Vulkan 1.3 core, or
    /// `VK_KHR_synchronization2` on 1.1/1.2). Stage and access masks are
    /// 64-bit (`VK_PIPELINE_STAGE_2_*` / `VK_ACCESS_2_*`).
    ///
    /// Sync2 is strictly more expressive than the legacy
    /// [`memory_barrier`](Self::memory_barrier): it has explicit stage 2
    /// bits for compute, transfer, and host that the legacy API leaves
    /// implicit. Returns an error wrapping `MissingFunction` if the device
    /// does not expose `vkCmdPipelineBarrier2`.
    pub fn memory_barrier2(
        &mut self,
        src_stage: PipelineStage2,
        dst_stage: PipelineStage2,
        src_access: AccessFlags2,
        dst_access: AccessFlags2,
    ) -> Result<()> {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdPipelineBarrier2
            .ok_or(Error::MissingFunction("vkCmdPipelineBarrier2"))?;

        let mb = VkMemoryBarrier2 {
            sType: VkStructureType::STRUCTURE_TYPE_MEMORY_BARRIER_2,
            srcStageMask: src_stage.0,
            srcAccessMask: src_access.0,
            dstStageMask: dst_stage.0,
            dstAccessMask: dst_access.0,
            ..Default::default()
        };
        let info = VkDependencyInfo {
            sType: VkStructureType::STRUCTURE_TYPE_DEPENDENCY_INFO,
            memoryBarrierCount: 1,
            pMemoryBarriers: &mb,
            ..Default::default()
        };
        // Safety: command buffer is recording, info and mb live until end of call.
        unsafe { cmd(self.buffer.handle, &info) };
        Ok(())
    }

    /// Record an image memory barrier using `vkCmdPipelineBarrier2`
    /// (Synchronization2). Same one-image, color-aspect, single-mip,
    /// single-layer simplification as [`image_barrier`](Self::image_barrier).
    /// Stage masks are 64-bit. See [`memory_barrier2`](Self::memory_barrier2)
    /// for general notes.
    pub fn image_barrier2(
        &mut self,
        src_stage: PipelineStage2,
        dst_stage: PipelineStage2,
        barrier: ImageBarrier<'_>,
    ) -> Result<()> {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdPipelineBarrier2
            .ok_or(Error::MissingFunction("vkCmdPipelineBarrier2"))?;

        let ib = VkImageMemoryBarrier2 {
            sType: VkStructureType::STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            srcStageMask: src_stage.0,
            srcAccessMask: barrier.src_access.0 as u64,
            dstStageMask: dst_stage.0,
            dstAccessMask: barrier.dst_access.0 as u64,
            oldLayout: barrier.old_layout.0,
            newLayout: barrier.new_layout.0,
            srcQueueFamilyIndex: !0u32,
            dstQueueFamilyIndex: !0u32,
            image: barrier.image.handle,
            subresourceRange: VkImageSubresourceRange {
                aspectMask: barrier.aspect_mask,
                baseMipLevel: 0,
                levelCount: 1,
                baseArrayLayer: 0,
                layerCount: 1,
            },
            ..Default::default()
        };
        let info = VkDependencyInfo {
            sType: VkStructureType::STRUCTURE_TYPE_DEPENDENCY_INFO,
            imageMemoryBarrierCount: 1,
            pImageMemoryBarriers: &ib,
            ..Default::default()
        };
        // Safety: command buffer is recording, info and ib live until end of call.
        unsafe { cmd(self.buffer.handle, &info) };
        Ok(())
    }

    /// Record a buffer memory barrier using `vkCmdPipelineBarrier2`
    /// (Synchronization2). Use this when you need per-buffer hazard
    /// tracking — for example, between a compute dispatch that writes a
    /// result buffer and a later dispatch that reads it. Covers the
    /// entire buffer (offset 0, size `VK_WHOLE_SIZE`).
    ///
    /// Stage and access masks are 64-bit. Returns an error wrapping
    /// `MissingFunction` if the device does not expose
    /// `vkCmdPipelineBarrier2`.
    pub fn buffer_barrier2(
        &mut self,
        src_stage: PipelineStage2,
        dst_stage: PipelineStage2,
        src_access: AccessFlags2,
        dst_access: AccessFlags2,
        buffer: &Buffer,
    ) -> Result<()> {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdPipelineBarrier2
            .ok_or(Error::MissingFunction("vkCmdPipelineBarrier2"))?;

        let bb = VkBufferMemoryBarrier2 {
            sType: VkStructureType::STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            srcStageMask: src_stage.0,
            srcAccessMask: src_access.0,
            dstStageMask: dst_stage.0,
            dstAccessMask: dst_access.0,
            srcQueueFamilyIndex: !0u32,
            dstQueueFamilyIndex: !0u32,
            buffer: buffer.handle,
            offset: 0,
            size: !0u64, // VK_WHOLE_SIZE
            ..Default::default()
        };
        let info = VkDependencyInfo {
            sType: VkStructureType::STRUCTURE_TYPE_DEPENDENCY_INFO,
            bufferMemoryBarrierCount: 1,
            pBufferMemoryBarriers: &bb,
            ..Default::default()
        };
        // Safety: command buffer is recording; info and bb live until end of call.
        unsafe { cmd(self.buffer.handle, &info) };
        Ok(())
    }

    /// Record `vkCmdCopyImageToBuffer`: copy bytes from one or more image
    /// regions into a buffer. The image must be in `TRANSFER_SRC_OPTIMAL`
    /// (or `GENERAL`) layout.
    pub fn copy_image_to_buffer(
        &mut self,
        src: &Image,
        src_layout: super::ImageLayout,
        dst: &Buffer,
        regions: &[BufferImageCopy],
    ) {
        debug_assert!(
            !regions.is_empty(),
            "vkCmdCopyImageToBuffer requires at least one region"
        );
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdCopyImageToBuffer
            .expect("vkCmdCopyImageToBuffer is required by Vulkan 1.0");

        let raw: Vec<VkBufferImageCopy> = regions.iter().map(|r| r.to_raw()).collect();
        // Safety: handles are valid, raw outlives the call.
        unsafe {
            cmd(
                self.buffer.handle,
                src.handle,
                src_layout.0,
                dst.handle,
                raw.len() as u32,
                raw.as_ptr(),
            )
        };
    }

    /// Record a global memory barrier between two pipeline stages.
    ///
    /// This is a simplified `vkCmdPipelineBarrier` that emits a single
    /// `VkMemoryBarrier`. Useful for guaranteeing that compute writes are
    /// visible to subsequent host reads (or to subsequent shader work).
    pub fn memory_barrier(
        &mut self,
        src_stage: PipelineStage,
        dst_stage: PipelineStage,
        src_access: AccessFlags,
        dst_access: AccessFlags,
    ) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdPipelineBarrier
            .expect("vkCmdPipelineBarrier is required by Vulkan 1.0");

        let barrier = VkMemoryBarrier {
            sType: VkStructureType::STRUCTURE_TYPE_MEMORY_BARRIER,
            srcAccessMask: src_access.0,
            dstAccessMask: dst_access.0,
            ..Default::default()
        };

        // Safety: command buffer is in recording state, barrier lives for the call.
        unsafe {
            cmd(
                self.buffer.handle,
                src_stage.0,
                dst_stage.0,
                0,
                1,
                &barrier,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
            )
        };
    }

    /// Record `vkCmdBeginRenderPass`: start a render pass instance
    /// targeting `framebuffer`. The clear values must contain one
    /// entry per render-pass attachment that uses
    /// `LoadOp::CLEAR` (or one entry per attachment, with the others
    /// ignored — the safest option).
    ///
    /// `clear_colors` are interpreted as `f32; 4` per slot.
    pub fn begin_render_pass(
        &mut self,
        render_pass: &RenderPass,
        framebuffer: &Framebuffer,
        clear_colors: &[[f32; 4]],
    ) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBeginRenderPass
            .expect("vkCmdBeginRenderPass is required by Vulkan 1.0");

        let raw_clears: Vec<VkClearValue> = clear_colors
            .iter()
            .map(|c| VkClearValue {
                color: VkClearColorValue { float32: *c },
            })
            .collect();

        let info = VkRenderPassBeginInfo {
            sType: VkStructureType::STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass: render_pass.handle,
            framebuffer: framebuffer.handle,
            renderArea: VkRect2D {
                offset: VkOffset2D { x: 0, y: 0 },
                extent: VkExtent2D {
                    width: framebuffer.width,
                    height: framebuffer.height,
                },
            },
            clearValueCount: raw_clears.len() as u32,
            pClearValues: if raw_clears.is_empty() {
                std::ptr::null()
            } else {
                raw_clears.as_ptr()
            },
            ..Default::default()
        };
        // VK_SUBPASS_CONTENTS_INLINE = 0
        // Safety: command buffer is recording, info+raw_clears live until call end.
        unsafe {
            cmd(
                self.buffer.handle,
                &info,
                VkSubpassContents::SUBPASS_CONTENTS_INLINE,
            )
        };
    }

    /// Like [`begin_render_pass`](Self::begin_render_pass), but accepts
    /// [`ClearValue`] entries so you can mix color and depth/stencil
    /// clears in a single render pass (e.g. one color attachment + one
    /// depth attachment).
    pub fn begin_render_pass_ext(
        &mut self,
        render_pass: &RenderPass,
        framebuffer: &Framebuffer,
        clear_values: &[ClearValue],
    ) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBeginRenderPass
            .expect("vkCmdBeginRenderPass is required by Vulkan 1.0");

        let raw_clears: Vec<VkClearValue> = clear_values
            .iter()
            .map(|cv| match cv {
                ClearValue::Color(c) => VkClearValue {
                    color: VkClearColorValue { float32: *c },
                },
                ClearValue::DepthStencil { depth, stencil } => VkClearValue {
                    depthStencil: VkClearDepthStencilValue {
                        depth: *depth,
                        stencil: *stencil,
                    },
                },
            })
            .collect();

        let info = VkRenderPassBeginInfo {
            sType: VkStructureType::STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass: render_pass.handle,
            framebuffer: framebuffer.handle,
            renderArea: VkRect2D {
                offset: VkOffset2D { x: 0, y: 0 },
                extent: VkExtent2D {
                    width: framebuffer.width,
                    height: framebuffer.height,
                },
            },
            clearValueCount: raw_clears.len() as u32,
            pClearValues: if raw_clears.is_empty() {
                std::ptr::null()
            } else {
                raw_clears.as_ptr()
            },
            ..Default::default()
        };
        unsafe {
            cmd(
                self.buffer.handle,
                &info,
                VkSubpassContents::SUBPASS_CONTENTS_INLINE,
            )
        };
    }

    /// Record `vkCmdEndRenderPass`.
    pub fn end_render_pass(&mut self) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdEndRenderPass
            .expect("vkCmdEndRenderPass is required by Vulkan 1.0");
        // Safety: command buffer is in recording state.
        unsafe { cmd(self.buffer.handle) };
    }

    /// Record `vkCmdBeginRendering` (Vulkan 1.3 core, or
    /// `VK_KHR_dynamic_rendering` on 1.1/1.2).
    ///
    /// Dynamic rendering replaces `VkRenderPass` + `VkFramebuffer` with
    /// a single call that names the attachments and their load/store
    /// ops directly. The matching [`end_rendering`](Self::end_rendering)
    /// closes the rendering scope.
    ///
    /// Returns `MissingFunction("vkCmdBeginRendering")` if the feature /
    /// extension is not enabled on the device.
    pub fn begin_rendering(&mut self, info: RenderingInfo<'_>) -> Result<()> {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBeginRendering
            .ok_or(Error::MissingFunction("vkCmdBeginRendering"))?;

        let to_raw = |a: &RenderingAttachment<'_>| -> VkRenderingAttachmentInfo {
            let clear = match a.clear_value {
                Some(ClearValue::Color([r, g, b, al])) => VkClearValue {
                    color: VkClearColorValue {
                        float32: [r, g, b, al],
                    },
                },
                Some(ClearValue::DepthStencil { depth, stencil }) => VkClearValue {
                    depthStencil: VkClearDepthStencilValue { depth, stencil },
                },
                None => unsafe { std::mem::zeroed() },
            };
            VkRenderingAttachmentInfo {
                sType: VkStructureType::STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                pNext: std::ptr::null(),
                imageView: a.view.handle,
                imageLayout: a.layout.0,
                resolveMode: unsafe { std::mem::zeroed() },
                resolveImageView: 0,
                resolveImageLayout: VkImageLayout::IMAGE_LAYOUT_UNDEFINED,
                loadOp: a.load_op.0,
                storeOp: a.store_op.0,
                clearValue: clear,
            }
        };

        let color_raw: Vec<VkRenderingAttachmentInfo> =
            info.color_attachments.iter().map(to_raw).collect();
        let depth_raw = info.depth_attachment.as_ref().map(to_raw);
        let stencil_raw = info.stencil_attachment.as_ref().map(to_raw);

        let raw = VkRenderingInfo {
            sType: VkStructureType::STRUCTURE_TYPE_RENDERING_INFO,
            pNext: std::ptr::null(),
            flags: 0,
            renderArea: info.render_area,
            layerCount: info.layer_count,
            viewMask: info.view_mask,
            colorAttachmentCount: color_raw.len() as u32,
            pColorAttachments: if color_raw.is_empty() {
                std::ptr::null()
            } else {
                color_raw.as_ptr()
            },
            pDepthAttachment: depth_raw
                .as_ref()
                .map_or(std::ptr::null(), |a| a as *const _),
            pStencilAttachment: stencil_raw
                .as_ref()
                .map_or(std::ptr::null(), |a| a as *const _),
        };

        // Safety: command buffer recording; raw, color_raw, depth_raw,
        // stencil_raw all outlive the synchronous call.
        unsafe { cmd(self.buffer.handle, &raw) };
        Ok(())
    }

    /// Record `vkCmdEndRendering` — closes a scope opened by
    /// [`begin_rendering`](Self::begin_rendering).
    ///
    /// Returns `MissingFunction("vkCmdEndRendering")` if the
    /// corresponding feature / extension is not enabled.
    pub fn end_rendering(&mut self) -> Result<()> {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdEndRendering
            .ok_or(Error::MissingFunction("vkCmdEndRendering"))?;
        // Safety: command buffer is in recording state.
        unsafe { cmd(self.buffer.handle) };
        Ok(())
    }

    /// Record `vkCmdBindDescriptorBuffersEXT` —
    /// `VK_EXT_descriptor_buffer`.
    ///
    /// Establishes which `VkBuffer`(s) hold descriptor data for the
    /// subsequent dispatches / draws. The actual set-to-offset mapping
    /// is chosen per-dispatch with
    /// [`set_descriptor_buffer_offsets`](Self::set_descriptor_buffer_offsets).
    ///
    /// The number of bindings cannot exceed
    /// `VkPhysicalDeviceDescriptorBufferPropertiesEXT.maxDescriptorBufferBindings`.
    pub fn bind_descriptor_buffers(&mut self, bindings: &[DescriptorBufferBinding]) -> Result<()> {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBindDescriptorBuffersEXT
            .ok_or(Error::MissingFunction("vkCmdBindDescriptorBuffersEXT"))?;

        let raw: Vec<VkDescriptorBufferBindingInfoEXT> = bindings
            .iter()
            .map(|b| VkDescriptorBufferBindingInfoEXT {
                sType: VkStructureType::STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
                pNext: std::ptr::null(),
                address: b.address,
                usage: b.usage,
            })
            .collect();

        // Safety: command buffer is recording; raw lives for the
        // duration of the call.
        unsafe { cmd(self.buffer.handle, raw.len() as u32, raw.as_ptr()) };
        Ok(())
    }

    /// Record `vkCmdSetDescriptorBufferOffsetsEXT` —
    /// `VK_EXT_descriptor_buffer`.
    ///
    /// For each set index in `[first_set, first_set + buffer_indices.len())`,
    /// selects which previously-bound descriptor buffer (by index into
    /// the last [`bind_descriptor_buffers`](Self::bind_descriptor_buffers)
    /// call) and what byte offset within it to read the set's
    /// descriptors from.
    ///
    /// `buffer_indices` and `offsets` must be parallel slices of the
    /// same length.
    pub fn set_descriptor_buffer_offsets(
        &mut self,
        bind_point: PipelineBindPoint,
        layout: &PipelineLayout,
        first_set: u32,
        buffer_indices: &[u32],
        offsets: &[u64],
    ) -> Result<()> {
        if buffer_indices.len() != offsets.len() {
            return Err(Error::InvalidArgument(
                "buffer_indices and offsets must have equal length",
            ));
        }
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdSetDescriptorBufferOffsetsEXT
            .ok_or(Error::MissingFunction("vkCmdSetDescriptorBufferOffsetsEXT"))?;
        // Safety: command buffer recording; layout valid; slices live
        // for the call; length parity checked above.
        unsafe {
            cmd(
                self.buffer.handle,
                bind_point.to_raw(),
                layout.handle,
                first_set,
                buffer_indices.len() as u32,
                buffer_indices.as_ptr(),
                offsets.as_ptr(),
            )
        };
        Ok(())
    }

    /// Record `vkCmdBuildAccelerationStructuresKHR` —
    /// `VK_KHR_acceleration_structure`.
    ///
    /// Builds (or updates, if `mode == Update`) a single acceleration
    /// structure from `geometries` + `ranges`. Both slices must have
    /// the same length — each `BuildRange` describes how many
    /// primitives to consume from the matching geometry.
    ///
    /// `scratch_address` is the GPU virtual address of a scratch buffer
    /// of ≥ [`BuildSizes::build_scratch_size`](super::BuildSizes::build_scratch_size)
    /// bytes (or `update_scratch_size` when `mode == Update`). The
    /// scratch buffer must have been created with
    /// `STORAGE_BUFFER | SHADER_DEVICE_ADDRESS` usage and bound to
    /// device memory before this call.
    ///
    /// `src` is only used when `mode == Update` — it's the
    /// previously-built structure whose layout the update reads.
    ///
    /// For multi-AS builds in a single command, drop to the raw
    /// [`CommandBufferRecordingExt::vk_cmd_build_acceleration_structures_khr`](crate::safe::auto::CommandBufferRecordingExt::vk_cmd_build_acceleration_structures_khr)
    /// entry point.
    #[allow(clippy::too_many_arguments)]
    pub fn build_acceleration_structure(
        &mut self,
        type_: AccelerationStructureType,
        mode: AccelerationStructureBuildMode,
        flags: AccelerationStructureBuildFlags,
        dst: &AccelerationStructure,
        src: Option<&AccelerationStructure>,
        geometries: &[AccelerationStructureGeometry],
        ranges: &[BuildRange],
        scratch_address: u64,
    ) -> Result<()> {
        if geometries.len() != ranges.len() {
            return Err(Error::InvalidArgument(
                "geometries and ranges must have equal length",
            ));
        }
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBuildAccelerationStructuresKHR
            .ok_or(Error::MissingFunction(
                "vkCmdBuildAccelerationStructuresKHR",
            ))?;

        let raw_geoms: Vec<VkAccelerationStructureGeometryKHR> =
            geometries.iter().map(|g| g.to_raw()).collect();
        let raw_ranges: Vec<VkAccelerationStructureBuildRangeInfoKHR> =
            ranges.iter().map(|r| r.to_raw()).collect();

        let info = VkAccelerationStructureBuildGeometryInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            pNext: std::ptr::null(),
            r#type: type_.to_raw(),
            flags: flags.0,
            mode: mode.to_raw(),
            srcAccelerationStructure: src.map_or(0, |s| s.raw()),
            dstAccelerationStructure: dst.raw(),
            geometryCount: raw_geoms.len() as u32,
            pGeometries: if raw_geoms.is_empty() {
                std::ptr::null()
            } else {
                raw_geoms.as_ptr()
            },
            ppGeometries: std::ptr::null(),
            scratchData: VkDeviceOrHostAddressKHR {
                deviceAddress: scratch_address,
            },
        };

        // vkCmdBuildAccelerationStructuresKHR takes
        // `ppBuildRangeInfos: *mut *const VkAccelerationStructureBuildRangeInfoKHR`
        // — an array of pointers, one per `pInfos` entry, each pointing
        // at a range array of length equal to that info's geometryCount.
        // We're driving a single info here.
        let ranges_ptr: *const VkAccelerationStructureBuildRangeInfoKHR = raw_ranges.as_ptr();
        let range_ptrs: [*const VkAccelerationStructureBuildRangeInfoKHR; 1] = [ranges_ptr];

        // Safety: command buffer is recording; info, raw_geoms, raw_ranges,
        // and range_ptrs all live until end of function. `ppGeometries`
        // is null so the driver uses pGeometries directly.
        unsafe { cmd(self.buffer.handle, 1, &info, range_ptrs.as_ptr() as *mut _) };
        Ok(())
    }

    /// Bind a [`RayTracingPipeline`] for subsequent
    /// [`trace_rays`](Self::trace_rays) dispatches.
    ///
    /// Requires `VK_KHR_ray_tracing_pipeline` enabled on the device.
    pub fn bind_ray_tracing_pipeline(&mut self, pipeline: &RayTracingPipeline) -> Result<()> {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBindPipeline
            .ok_or(Error::MissingFunction("vkCmdBindPipeline"))?;
        // Safety: command buffer is recording, pipeline handle is valid.
        // VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR = 1000165000
        unsafe {
            cmd(
                self.buffer.handle,
                VkPipelineBindPoint::PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                pipeline.raw(),
            )
        };
        Ok(())
    }

    /// Record `vkCmdTraceRaysKHR` — `VK_KHR_ray_tracing_pipeline`.
    ///
    /// Launches a `width × height × depth` grid of ray-generation
    /// shader invocations, each one driving BVH traversal through the
    /// currently-bound acceleration structure(s).
    ///
    /// The four [`ShaderBindingRegion`](super::ShaderBindingRegion)s
    /// describe the rgen / miss / hit / callable sub-regions of the
    /// SBT buffer. Unused regions pass `Default::default()` (a
    /// zero-address, zero-stride, zero-size region is legal per spec
    /// for miss / hit / callable). The rgen region is mandatory and
    /// must contain exactly one handle.
    ///
    /// Requires `VK_KHR_ray_tracing_pipeline` enabled on the device.
    #[allow(clippy::too_many_arguments)]
    pub fn trace_rays(
        &mut self,
        raygen: ShaderBindingRegion,
        miss: ShaderBindingRegion,
        hit: ShaderBindingRegion,
        callable: ShaderBindingRegion,
        width: u32,
        height: u32,
        depth: u32,
    ) -> Result<()> {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdTraceRaysKHR
            .ok_or(Error::MissingFunction("vkCmdTraceRaysKHR"))?;
        let rgen = raygen.to_raw();
        let rmiss = miss.to_raw();
        let rhit = hit.to_raw();
        let rcall = callable.to_raw();
        // Safety: command buffer recording; the four region structs
        // all live until end of function.
        unsafe {
            cmd(
                self.buffer.handle,
                &rgen,
                &rmiss,
                &rhit,
                &rcall,
                width,
                height,
                depth,
            )
        };
        Ok(())
    }

    /// Record `vkCmdBindPipeline` for a graphics pipeline.
    pub fn bind_graphics_pipeline(&mut self, pipeline: &GraphicsPipeline) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBindPipeline
            .expect("vkCmdBindPipeline is required by Vulkan 1.0");
        // Safety: command buffer is recording, pipeline handle is valid.
        unsafe {
            cmd(
                self.buffer.handle,
                VkPipelineBindPoint::PIPELINE_BIND_POINT_GRAPHICS,
                pipeline.handle,
            )
        };
    }

    /// Record `vkCmdBindDescriptorSets` for a graphics pipeline.
    pub fn bind_graphics_descriptor_sets(
        &mut self,
        layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[&DescriptorSet],
    ) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBindDescriptorSets
            .expect("vkCmdBindDescriptorSets is required by Vulkan 1.0");

        let raw: Vec<VkDescriptorSet> = descriptor_sets.iter().map(|s| s.handle).collect();
        // Safety: command buffer is recording, raw lives until call end.
        unsafe {
            cmd(
                self.buffer.handle,
                VkPipelineBindPoint::PIPELINE_BIND_POINT_GRAPHICS,
                layout.handle,
                first_set,
                raw.len() as u32,
                raw.as_ptr(),
                0,
                std::ptr::null(),
            )
        };
    }

    /// Record `vkCmdBindVertexBuffers`. Each entry is a `(buffer, offset)`
    /// tuple. The first entry is bound to slot `first_binding`, the
    /// next to `first_binding + 1`, etc.
    pub fn bind_vertex_buffers(&mut self, first_binding: u32, buffers_offsets: &[(&Buffer, u64)]) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBindVertexBuffers
            .expect("vkCmdBindVertexBuffers is required by Vulkan 1.0");

        let raw_buffers: Vec<VkBuffer> = buffers_offsets.iter().map(|(b, _)| b.handle).collect();
        let raw_offsets: Vec<u64> = buffers_offsets.iter().map(|(_, o)| *o).collect();
        // Safety: command buffer is recording, raw_* outlive the call.
        unsafe {
            cmd(
                self.buffer.handle,
                first_binding,
                raw_buffers.len() as u32,
                raw_buffers.as_ptr(),
                raw_offsets.as_ptr(),
            )
        };
    }

    /// Record `vkCmdBindIndexBuffer`. `index_type` is `VK_INDEX_TYPE_UINT16` (0)
    /// or `VK_INDEX_TYPE_UINT32` (1).
    pub fn bind_index_buffer(&mut self, buffer: &Buffer, offset: u64, index_type: u32) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdBindIndexBuffer
            .expect("vkCmdBindIndexBuffer is required by Vulkan 1.0");
        // Safety: command buffer is recording, buffer handle is valid.
        let it = match index_type {
            0 => VkIndexType::INDEX_TYPE_UINT16,
            _ => VkIndexType::INDEX_TYPE_UINT32,
        };
        unsafe { cmd(self.buffer.handle, buffer.handle, offset, it) };
    }

    /// Record `vkCmdDraw`: emit `vertex_count` vertices starting at
    /// `first_vertex`, repeated for `instance_count` instances starting
    /// at `first_instance`.
    pub fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdDraw
            .expect("vkCmdDraw is required by Vulkan 1.0");
        // Safety: command buffer is recording, pipeline must be bound.
        unsafe {
            cmd(
                self.buffer.handle,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            )
        };
    }

    /// Record `vkCmdDrawIndexed`.
    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdDrawIndexed
            .expect("vkCmdDrawIndexed is required by Vulkan 1.0");
        // Safety: command buffer is recording, pipeline + index buffer must be bound.
        unsafe {
            cmd(
                self.buffer.handle,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        };
    }

    /// Record `vkCmdSetViewport`. Used when the pipeline was built with
    /// `VK_DYNAMIC_STATE_VIEWPORT`.
    pub fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdSetViewport
            .expect("vkCmdSetViewport is required by Vulkan 1.0");
        let viewport = VkViewport {
            x,
            y,
            width,
            height,
            minDepth: 0.0,
            maxDepth: 1.0,
        };
        // Safety: command buffer is recording, viewport lives until call end.
        unsafe { cmd(self.buffer.handle, 0, 1, &viewport) };
    }

    /// Record `vkCmdSetScissor`.
    pub fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdSetScissor
            .expect("vkCmdSetScissor is required by Vulkan 1.0");
        let scissor = VkRect2D {
            offset: VkOffset2D { x, y },
            extent: VkExtent2D { width, height },
        };
        // Safety: command buffer is recording, scissor lives until call end.
        unsafe { cmd(self.buffer.handle, 0, 1, &scissor) };
    }

    /// Finish recording explicitly. This is what `Drop` does — call this
    /// only if you want to inspect errors from `vkEndCommandBuffer`.
    pub fn end(self) -> Result<()> {
        let end = self
            .buffer
            .device
            .dispatch
            .vkEndCommandBuffer
            .ok_or(Error::MissingFunction("vkEndCommandBuffer"))?;
        // Safety: command buffer is in recording state.
        let result = unsafe { end(self.buffer.handle) };
        // Suppress the Drop impl's call.
        std::mem::forget(self);
        check(result)
    }
}

impl<'a> Drop for CommandBufferRecording<'a> {
    fn drop(&mut self) {
        if let Some(end) = self.buffer.device.dispatch.vkEndCommandBuffer {
            // Safety: command buffer is in recording state.
            // We ignore errors here; users who care should call end() explicitly.
            unsafe { end(self.buffer.handle) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time lock-in: `CommandBuffer` must stay `Send + Sync`.
    /// Downstreams (Fuel) build command buffers on worker threads and
    /// submit them from a scheduler thread. If a future field addition
    /// reintroduces a non-`Send`/`Sync` type, this assertion will fail
    /// to compile.
    #[test]
    fn command_buffer_is_send_sync() {
        fn _assert<T: Send + Sync>() {}
        _assert::<CommandBuffer>();
    }
}
