//! A safe, RAII-based Rust API over the raw Vulkan bindings.
//!
//! This module wraps every Vulkan handle in a Rust type with automatic
//! `Drop` cleanup, `Result`-based errors, typed flags, and convenience
//! helpers — so you never write a manual `vkDestroy*` call.
//!
//! # Design principles
//!
//! - **RAII** — every handle is destroyed on drop.
//! - **Minimal overhead** — handle wrappers hold the raw handle + an
//!   `Arc` to the parent for dispatch and lifetime tracking. No global
//!   state, no locking in hot paths.
//! - **Typed flags** — [`PipelineStage`], [`AccessFlags`], [`Format`],
//!   [`BufferUsage`], [`ImageUsage`] etc. replace raw hex constants.
//! - **Result-based errors** — every fallible call returns
//!   `Result<T, Error>`.
//!
//! # Coverage
//!
//! The safe wrapper covers **compute and graphics end-to-end**:
//!
//! - **Core:** [`Instance`], [`PhysicalDevice`], [`Device`], [`Queue`],
//!   [`DeviceFeatures`], [`PhysicalDeviceGroup`]
//! - **Memory:** [`Buffer`] (with [`new_bound`](Buffer::new_bound)),
//!   [`Image`] (with [`new_2d_bound`](Image::new_2d_bound)),
//!   [`ImageView`] (color + depth), [`Sampler`] (with comparison),
//!   [`DeviceMemory`], [`Allocator`] (VMA-style sub-allocator)
//! - **Upload helpers:** [`Queue::upload_buffer`],
//!   [`Queue::upload_image_rgba`], [`Queue::one_shot`]
//! - **Compute:** [`ComputePipeline`], [`PipelineLayout`],
//!   [`DescriptorSet`], [`ShaderModule`], specialization constants,
//!   pipeline cache, push constants
//! - **Graphics:** [`RenderPass`] (with [`simple_color`](RenderPass::simple_color)),
//!   [`Framebuffer`], [`GraphicsPipelineBuilder`] (depth bias,
//!   [`CompareOp`], [`InputRate`], multi-attachment blend, dynamic
//!   viewport/scissor), [`Surface`], [`Swapchain`]
//! - **Sync:** [`PipelineStage`] / [`AccessFlags`] (typed, 32+64-bit),
//!   [`Fence`], [`Semaphore`] (binary + timeline), [`ImageBarrier`]
//!   (color + depth aspect), [`ClearValue`], [`QueryPool`]
//! - **Raw escape hatch:** [`Device::dispatch()`] /
//!   [`Instance::dispatch()`] expose the full dispatch tables
//!
//! # Quick example
//!
//! ```no_run
//! use vulkane::safe::{Instance, InstanceCreateInfo, ApiVersion, QueueFlags};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let instance = Instance::new(InstanceCreateInfo {
//!     application_name: Some("my-app"),
//!     api_version: ApiVersion::V1_0,
//!     ..Default::default()
//! })?;
//!
//! for pd in instance.enumerate_physical_devices()? {
//!     println!("Found GPU: {}", pd.properties().device_name());
//! }
//! # Ok(())
//! # }
//! ```

use crate::raw::bindings::VkResult;

mod allocator;
pub mod auto;
mod buffer;
mod command;
mod debug;
mod descriptor;
pub(crate) mod device;
mod extensions;
mod features;
mod graphics_pipeline;
mod image;
mod flags;
mod instance;
mod memory;
#[cfg(feature = "naga")]
pub mod naga;
#[cfg(feature = "shaderc")]
pub mod shaderc;
#[cfg(feature = "slang")]
pub mod slang;
mod physical;
mod pipeline;
pub mod pnext;
mod query;
mod render_pass;
mod shader;
mod shaders;
mod surface;
mod swapchain;
mod sync;

pub use allocator::{
    Allocation, AllocationCreateInfo, AllocationStatistics, AllocationStrategy, AllocationUsage,
    Allocator, DefragmentationMove, DefragmentationPlan, FitStatus, PoolCreateInfo, PoolHandle,
    PressureCallbackId, PressureEvent, PressureKind,
};
pub use buffer::{Buffer, BufferCreateInfo, BufferUsage, MemoryRequirements};
pub use command::{BufferCopy, ClearValue, CommandBuffer, CommandBufferRecording, CommandPool};
pub use debug::{
    DebugCallback, DebugMessage, DebugMessageSeverity, DebugMessageType, default_callback,
};
pub use descriptor::{
    DescriptorPool, DescriptorPoolSize, DescriptorSet, DescriptorSetLayout,
    DescriptorSetLayoutBinding, DescriptorType, ShaderStageFlags,
};
pub use device::{
    Device, DeviceCreateInfo, Queue, QueueCreateInfo, SignalSemaphore, WaitSemaphore,
};
pub use auto::{
    CommandBufferRecordingExt, DeviceExt, InstanceExt, PhysicalDeviceExt, QueueExt,
};
pub use extensions::{DeviceExtensions, InstanceExtensions};
pub use features::DeviceFeatures;
pub use flags::{AccessFlags, AccessFlags2, PipelineStage, PipelineStage2};
pub use graphics_pipeline::{
    CompareOp, CullMode, FrontFace, GraphicsPipeline, GraphicsPipelineBuilder,
    GraphicsShaderStage, InputRate, PolygonMode, PrimitiveTopology, VertexInputAttribute,
    VertexInputBinding,
};
pub use image::{
    BufferImageCopy, Format, Image, Image2dCreateInfo, ImageBarrier, ImageLayout, ImageUsage,
    ImageView, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerFilter, SamplerMipmapMode,
};
pub use instance::{
    ApiVersion, ExtensionProperties, Instance, InstanceCreateInfo, KHRONOS_VALIDATION_LAYER,
    LayerProperties,
};
pub use memory::{DeviceMemory, MappedMemory, MemoryPropertyFlags};
pub use physical::{
    CooperativeMatrixProperties, MemoryBudget, MemoryHeap, MemoryHeapFlags, MemoryType,
    PhysicalDevice, PhysicalDeviceGroup, PhysicalDeviceProperties, PhysicalDeviceType,
    QueueFamilyProperties, QueueFlags,
};
pub use pipeline::{
    ComputePipeline, PipelineCache, PipelineLayout, PushConstantRange, SpecializationConstants,
};
pub use pnext::PNextChain;
pub use query::{PipelineStatisticsFlags, QueryPool, QueryType};
pub use render_pass::{
    AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, Framebuffer, RenderPass,
    RenderPassCreateInfo,
};
pub use shader::ShaderModule;
pub use shaders::{ShaderLoadError, ShaderRegistry, ShaderSource};
pub use surface::{PresentMode, Surface, SurfaceCapabilities, SurfaceFormat};
pub use swapchain::{Swapchain, SwapchainCreateInfo};
pub use sync::{Fence, Semaphore, SemaphoreKind};

/// Error type returned by all fallible operations in [`vulkane::safe`](crate::safe).
#[derive(Debug)]
pub enum Error {
    /// The Vulkan loader could not find or load the runtime library.
    LibraryLoad(libloading::Error),

    /// The Vulkan loader was missing a required function.
    MissingFunction(&'static str),

    /// A Vulkan API call returned a non-`SUCCESS` result.
    Vk(VkResult),

    /// A C string contained an interior NUL byte.
    InvalidString(std::ffi::NulError),

    /// The caller passed an argument combination that the wrapper rejects
    /// before issuing the Vulkan call. The string is a static, human-readable
    /// description of what's wrong.
    InvalidArgument(&'static str),

    /// Looking up, reading, or decoding a precompiled shader via
    /// [`ShaderRegistry`] failed.
    ///
    /// This replaces the earlier `ShaderLoad(String)` variant;
    /// `ShaderLoadError` carries the structured failure reason
    /// (not found / I/O failure / malformed SPIR-V length).
    ShaderLoad(ShaderLoadError),

    /// GLSL-to-SPIR-V compilation via [`naga`] failed.
    /// Only emitted when the `naga` Cargo feature is enabled.
    #[cfg(feature = "naga")]
    NagaCompile(String),

    /// GLSL/HLSL-to-SPIR-V compilation via [`shaderc`] failed.
    /// Only emitted when the `shaderc` Cargo feature is enabled.
    #[cfg(feature = "shaderc")]
    ShadercCompile(String),

    /// Slang-to-SPIR-V compilation via [`shader-slang`] failed.
    /// Only emitted when the `slang` Cargo feature is enabled.
    #[cfg(feature = "slang")]
    SlangCompile(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LibraryLoad(e) => write!(f, "failed to load Vulkan library: {e}"),
            Self::MissingFunction(name) => write!(f, "Vulkan function not loaded: {name}"),
            Self::Vk(result) => write!(f, "Vulkan call failed: {result:?}"),
            Self::InvalidString(e) => write!(f, "invalid C string: {e}"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::ShaderLoad(e) => write!(f, "shader load failed: {e}"),
            #[cfg(feature = "naga")]
            Self::NagaCompile(s) => write!(f, "GLSL compilation failed: {s}"),
            #[cfg(feature = "shaderc")]
            Self::ShadercCompile(s) => write!(f, "shaderc compilation failed: {s}"),
            #[cfg(feature = "slang")]
            Self::SlangCompile(s) => write!(f, "Slang compilation failed: {s}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LibraryLoad(e) => Some(e),
            Self::InvalidString(e) => Some(e),
            Self::ShaderLoad(e) => Some(e),
            _ => None,
        }
    }
}

impl From<libloading::Error> for Error {
    fn from(e: libloading::Error) -> Self {
        Self::LibraryLoad(e)
    }
}

impl From<VkResult> for Error {
    fn from(e: VkResult) -> Self {
        Self::Vk(e)
    }
}

impl From<std::ffi::NulError> for Error {
    fn from(e: std::ffi::NulError) -> Self {
        Self::InvalidString(e)
    }
}

/// Convenience alias for `Result<T, vulkane::safe::Error>`.
pub type Result<T> = std::result::Result<T, Error>;

/// Helper: convert a `VkResult` into a `Result<()>`.
pub(crate) fn check(result: VkResult) -> Result<()> {
    if result == VkResult::SUCCESS {
        Ok(())
    } else {
        Err(Error::Vk(result))
    }
}
