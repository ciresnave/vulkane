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

use super::descriptor::{DescriptorSet, ShaderStageFlags};
use super::device::DeviceInner;
use super::flags::{AccessFlags, AccessFlags2, PipelineStage, PipelineStage2};
use super::graphics_pipeline::GraphicsPipeline;
use super::image::{BufferImageCopy, Image, ImageBarrier};
use super::pipeline::{ComputePipeline, PipelineLayout};
use super::query::QueryPool;
use super::render_pass::{Framebuffer, RenderPass};
use super::{Buffer, Device, Error, Result, check};
use crate::raw::bindings::*;
use std::sync::Arc;

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
