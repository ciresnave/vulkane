//! Typed pipeline stage and access mask constants for synchronization.
//!
//! These types wrap the raw `VK_PIPELINE_STAGE_*` and `VK_ACCESS_*` bit
//! masks in thin newtypes so barrier calls are self-documenting:
//!
//! ```ignore
//! rec.memory_barrier(
//!     PipelineStage::COMPUTE_SHADER,
//!     PipelineStage::HOST,
//!     AccessFlags::SHADER_WRITE,
//!     AccessFlags::HOST_READ,
//! );
//! ```
//!
//! All four types expose a `pub` inner field, so users can always pass
//! raw values for extension bits not covered here:
//! `PipelineStage(0x1234)`.

// ---------------------------------------------------------------------------
// Legacy (32-bit) pipeline stages — VK_PIPELINE_STAGE_*
// ---------------------------------------------------------------------------

/// A bitmask of Vulkan pipeline stages (`VkPipelineStageFlagBits`).
///
/// Used by [`CommandBufferRecording::memory_barrier`](super::CommandBufferRecording::memory_barrier),
/// [`CommandBufferRecording::image_barrier`](super::CommandBufferRecording::image_barrier),
/// [`CommandBufferRecording::write_timestamp`](super::CommandBufferRecording::write_timestamp),
/// and [`WaitSemaphore::dst_stage_mask`](super::WaitSemaphore).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PipelineStage(pub u32);

impl PipelineStage {
    pub const TOP_OF_PIPE: Self = Self(0x0000_0001);
    pub const DRAW_INDIRECT: Self = Self(0x0000_0002);
    pub const VERTEX_INPUT: Self = Self(0x0000_0004);
    pub const VERTEX_SHADER: Self = Self(0x0000_0008);
    pub const TESSELLATION_CONTROL_SHADER: Self = Self(0x0000_0010);
    pub const TESSELLATION_EVALUATION_SHADER: Self = Self(0x0000_0020);
    pub const GEOMETRY_SHADER: Self = Self(0x0000_0040);
    pub const FRAGMENT_SHADER: Self = Self(0x0000_0080);
    pub const EARLY_FRAGMENT_TESTS: Self = Self(0x0000_0100);
    pub const LATE_FRAGMENT_TESTS: Self = Self(0x0000_0200);
    pub const COLOR_ATTACHMENT_OUTPUT: Self = Self(0x0000_0400);
    pub const COMPUTE_SHADER: Self = Self(0x0000_0800);
    pub const TRANSFER: Self = Self(0x0000_1000);
    pub const BOTTOM_OF_PIPE: Self = Self(0x0000_2000);
    pub const HOST: Self = Self(0x0000_4000);
    pub const ALL_GRAPHICS: Self = Self(0x0000_8000);
    pub const ALL_COMMANDS: Self = Self(0x0001_0000);

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for PipelineStage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

// ---------------------------------------------------------------------------
// Legacy (32-bit) access masks — VK_ACCESS_*
// ---------------------------------------------------------------------------

/// A bitmask of Vulkan memory access types (`VkAccessFlagBits`).
///
/// Used in [`ImageBarrier`](super::ImageBarrier) and
/// [`CommandBufferRecording::memory_barrier`](super::CommandBufferRecording::memory_barrier).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AccessFlags(pub u32);

impl AccessFlags {
    pub const NONE: Self = Self(0);
    pub const INDIRECT_COMMAND_READ: Self = Self(0x0000_0001);
    pub const INDEX_READ: Self = Self(0x0000_0002);
    pub const VERTEX_ATTRIBUTE_READ: Self = Self(0x0000_0004);
    pub const UNIFORM_READ: Self = Self(0x0000_0008);
    pub const INPUT_ATTACHMENT_READ: Self = Self(0x0000_0010);
    pub const SHADER_READ: Self = Self(0x0000_0020);
    pub const SHADER_WRITE: Self = Self(0x0000_0040);
    pub const COLOR_ATTACHMENT_READ: Self = Self(0x0000_0080);
    pub const COLOR_ATTACHMENT_WRITE: Self = Self(0x0000_0100);
    pub const DEPTH_STENCIL_ATTACHMENT_READ: Self = Self(0x0000_0200);
    pub const DEPTH_STENCIL_ATTACHMENT_WRITE: Self = Self(0x0000_0400);
    pub const TRANSFER_READ: Self = Self(0x0000_0800);
    pub const TRANSFER_WRITE: Self = Self(0x0000_1000);
    pub const HOST_READ: Self = Self(0x0000_2000);
    pub const HOST_WRITE: Self = Self(0x0000_4000);
    pub const MEMORY_READ: Self = Self(0x0000_8000);
    pub const MEMORY_WRITE: Self = Self(0x0001_0000);

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for AccessFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

// ---------------------------------------------------------------------------
// Synchronization2 (64-bit) pipeline stages — VK_PIPELINE_STAGE_2_*
// ---------------------------------------------------------------------------

/// A 64-bit bitmask of Vulkan pipeline stages for the Synchronization2 API
/// (`VkPipelineStageFlagBits2`).
///
/// Used by [`CommandBufferRecording::memory_barrier2`](super::CommandBufferRecording::memory_barrier2) and
/// [`CommandBufferRecording::image_barrier2`](super::CommandBufferRecording::image_barrier2).
/// Vulkan 1.3 core or `VK_KHR_synchronization2` on 1.1/1.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PipelineStage2(pub u64);

impl PipelineStage2 {
    pub const NONE: Self = Self(0);
    pub const TOP_OF_PIPE: Self = Self(0x0000_0001);
    pub const DRAW_INDIRECT: Self = Self(0x0000_0002);
    pub const VERTEX_INPUT: Self = Self(0x0000_0004);
    pub const VERTEX_SHADER: Self = Self(0x0000_0008);
    pub const TESSELLATION_CONTROL_SHADER: Self = Self(0x0000_0010);
    pub const TESSELLATION_EVALUATION_SHADER: Self = Self(0x0000_0020);
    pub const GEOMETRY_SHADER: Self = Self(0x0000_0040);
    pub const FRAGMENT_SHADER: Self = Self(0x0000_0080);
    pub const EARLY_FRAGMENT_TESTS: Self = Self(0x0000_0100);
    pub const LATE_FRAGMENT_TESTS: Self = Self(0x0000_0200);
    pub const COLOR_ATTACHMENT_OUTPUT: Self = Self(0x0000_0400);
    pub const COMPUTE_SHADER: Self = Self(0x0000_0800);
    pub const ALL_TRANSFER: Self = Self(0x0000_1000);
    pub const BOTTOM_OF_PIPE: Self = Self(0x0000_2000);
    pub const HOST: Self = Self(0x0000_4000);
    pub const ALL_GRAPHICS: Self = Self(0x0000_8000);
    pub const ALL_COMMANDS: Self = Self(0x0001_0000);
    pub const COPY: Self = Self(0x1_0000_0000);
    pub const RESOLVE: Self = Self(0x2_0000_0000);
    pub const BLIT: Self = Self(0x4_0000_0000);
    pub const CLEAR: Self = Self(0x8_0000_0000);
    pub const INDEX_INPUT: Self = Self(0x10_0000_0000);
    pub const VERTEX_ATTRIBUTE_INPUT: Self = Self(0x20_0000_0000);
    pub const PRE_RASTERIZATION_SHADERS: Self = Self(0x40_0000_0000);

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for PipelineStage2 {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

// ---------------------------------------------------------------------------
// Synchronization2 (64-bit) access masks — VK_ACCESS_2_*
// ---------------------------------------------------------------------------

/// A 64-bit bitmask of Vulkan memory access types for the Synchronization2
/// API (`VkAccessFlagBits2`).
///
/// Used in [`CommandBufferRecording::memory_barrier2`](super::CommandBufferRecording::memory_barrier2) and
/// [`CommandBufferRecording::image_barrier2`](super::CommandBufferRecording::image_barrier2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AccessFlags2(pub u64);

impl AccessFlags2 {
    pub const NONE: Self = Self(0);
    pub const INDIRECT_COMMAND_READ: Self = Self(0x0000_0001);
    pub const INDEX_READ: Self = Self(0x0000_0002);
    pub const VERTEX_ATTRIBUTE_READ: Self = Self(0x0000_0004);
    pub const UNIFORM_READ: Self = Self(0x0000_0008);
    pub const INPUT_ATTACHMENT_READ: Self = Self(0x0000_0010);
    pub const SHADER_READ: Self = Self(0x0000_0020);
    pub const SHADER_WRITE: Self = Self(0x0000_0040);
    pub const COLOR_ATTACHMENT_READ: Self = Self(0x0000_0080);
    pub const COLOR_ATTACHMENT_WRITE: Self = Self(0x0000_0100);
    pub const DEPTH_STENCIL_ATTACHMENT_READ: Self = Self(0x0000_0200);
    pub const DEPTH_STENCIL_ATTACHMENT_WRITE: Self = Self(0x0000_0400);
    pub const TRANSFER_READ: Self = Self(0x0000_0800);
    pub const TRANSFER_WRITE: Self = Self(0x0000_1000);
    pub const HOST_READ: Self = Self(0x0000_2000);
    pub const HOST_WRITE: Self = Self(0x0000_4000);
    pub const MEMORY_READ: Self = Self(0x0000_8000);
    pub const MEMORY_WRITE: Self = Self(0x0001_0000);
    pub const SHADER_SAMPLED_READ: Self = Self(0x1_0000_0000);
    pub const SHADER_STORAGE_READ: Self = Self(0x2_0000_0000);
    pub const SHADER_STORAGE_WRITE: Self = Self(0x4_0000_0000);

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for AccessFlags2 {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
