//! Safe wrapper for `VkCommandPool` and `VkCommandBuffer`.

use super::descriptor::{DescriptorSet, ShaderStageFlags};
use super::device::DeviceInner;
use super::pipeline::{ComputePipeline, PipelineLayout};
use super::query::QueryPool;
use super::{Buffer, Device, Error, Result, check};
use crate::raw::bindings::*;
use std::sync::Arc;

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
    /// `pipeline_stage` should be a single `VK_PIPELINE_STAGE_*` bit (e.g.
    /// `0x00000001` = TOP_OF_PIPE, `0x00002000` = BOTTOM_OF_PIPE,
    /// `0x00000800` = COMPUTE_SHADER).
    pub fn write_timestamp(&mut self, pipeline_stage: u32, pool: &QueryPool, query: u32) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdWriteTimestamp
            .expect("vkCmdWriteTimestamp is required by Vulkan 1.0");
        // Safety: command buffer is in recording state, pool is valid,
        // query slot must be in bounds (caller's responsibility).
        unsafe { cmd(self.buffer.handle, pipeline_stage, pool.handle, query) };
    }

    /// Record a global memory barrier between two pipeline stages.
    ///
    /// This is a simplified `vkCmdPipelineBarrier` that emits a single
    /// `VkMemoryBarrier`. Useful for guaranteeing that compute writes are
    /// visible to subsequent host reads (or to subsequent shader work).
    pub fn memory_barrier(
        &mut self,
        src_stage: u32,
        dst_stage: u32,
        src_access: u32,
        dst_access: u32,
    ) {
        let cmd = self
            .buffer
            .device
            .dispatch
            .vkCmdPipelineBarrier
            .expect("vkCmdPipelineBarrier is required by Vulkan 1.0");

        let barrier = VkMemoryBarrier {
            sType: VkStructureType::STRUCTURE_TYPE_MEMORY_BARRIER,
            srcAccessMask: src_access,
            dstAccessMask: dst_access,
            ..Default::default()
        };

        // Safety: command buffer is in recording state, barrier lives for the call.
        unsafe {
            cmd(
                self.buffer.handle,
                src_stage,
                dst_stage,
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
