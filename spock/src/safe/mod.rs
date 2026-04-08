//! A safe, RAII-based Rust API over the raw Vulkan bindings.
//!
//! This module provides type-safe wrappers around Vulkan handles with the
//! following design principles:
//!
//! - **RAII**: every handle is destroyed automatically via `Drop`. No manual
//!   `vkDestroy*` calls.
//! - **Minimal overhead**: handle wrappers are zero-cost, holding the raw
//!   handle plus an `Arc` to the parent for dispatch and lifetime tracking.
//!   No global state, no locking, no boxing in hot paths.
//! - **Type-safe**: parent-child relationships (`Device` owns `Buffer`, etc.)
//!   are encoded as `Arc` ownership so children can outlive any local scope
//!   that drops the parent.
//! - **Borrow-checked**: where lifetimes cleanly map to Vulkan parent-child
//!   relationships, we use `&` references instead of `Arc` clones.
//! - **Result-based errors**: every fallible call returns `Result<T, Error>`.
//!
//! # Scope
//!
//! The safe wrapper covers the **complete compute path**:
//!
//! - [`Instance`] / [`InstanceCreateInfo`]
//! - [`PhysicalDevice`]
//! - [`Device`] / [`DeviceCreateInfo`] / [`Queue`]
//! - [`DeviceMemory`] (with mapping)
//! - [`Buffer`] / [`BufferCreateInfo`]
//! - [`ShaderModule`] (takes any `&[u32]` of SPIR-V)
//! - [`DescriptorSetLayout`] / [`DescriptorPool`] / [`DescriptorSet`]
//! - [`PipelineLayout`] / [`ComputePipeline`]
//! - [`CommandPool`] / [`CommandBuffer`] (record `dispatch`, `bind_pipeline`,
//!   `bind_descriptor_sets`, `fill_buffer`, `memory_barrier`)
//! - [`Fence`]
//!
//! Graphics-specific functionality (swapchains, render passes, graphics
//! pipelines, images, samplers) is not yet covered. Use
//! [`spock::raw`](crate::raw) for those use cases.
//!
//! Compiling shaders is left to the user; spock takes SPIR-V as `&[u32]`.
//! With the optional `naga` feature enabled, `safe::naga::compile_glsl`
//! provides a convenience function for compiling GLSL at runtime via the
//! [`naga`](https://docs.rs/naga) crate.
//!
//! # Example
//!
//! ```no_run
//! use spock::safe::{Instance, InstanceCreateInfo, ApiVersion};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let instance = Instance::new(InstanceCreateInfo {
//!     application_name: Some("my-app"),
//!     application_version: ApiVersion::new(0, 0, 1, 0),
//!     engine_name: None,
//!     engine_version: ApiVersion::new(0, 0, 1, 0),
//!     api_version: ApiVersion::V1_0,
//! })?;
//!
//! let physical_devices = instance.enumerate_physical_devices()?;
//! for pd in &physical_devices {
//!     let props = pd.properties();
//!     println!("Found GPU: {}", props.device_name());
//! }
//! # Ok(())
//! # }
//! ```

use crate::raw::bindings::VkResult;

mod buffer;
mod command;
mod descriptor;
mod device;
mod instance;
mod memory;
#[cfg(feature = "naga")]
pub mod naga;
mod physical;
mod pipeline;
mod query;
mod shader;
mod sync;

pub use buffer::{Buffer, BufferCreateInfo, BufferUsage};
pub use command::{BufferCopy, CommandBuffer, CommandBufferRecording, CommandPool};
pub use descriptor::{
    DescriptorPool, DescriptorPoolSize, DescriptorSet, DescriptorSetLayout,
    DescriptorSetLayoutBinding, DescriptorType, ShaderStageFlags,
};
pub use device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo};
pub use instance::{ApiVersion, Instance, InstanceCreateInfo};
pub use memory::{DeviceMemory, MappedMemory, MemoryPropertyFlags};
pub use physical::{
    MemoryHeap, MemoryHeapFlags, MemoryType, PhysicalDevice, PhysicalDeviceProperties,
    PhysicalDeviceType, QueueFamilyProperties, QueueFlags,
};
pub use pipeline::{ComputePipeline, PipelineLayout, PushConstantRange, SpecializationConstants};
pub use query::{PipelineStatisticsFlags, QueryPool, QueryType};
pub use shader::ShaderModule;
pub use sync::Fence;

/// Error type returned by all fallible operations in [`spock::safe`](crate::safe).
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

    /// GLSL-to-SPIR-V compilation via [`naga`] failed.
    /// Only emitted when the `naga` Cargo feature is enabled.
    #[cfg(feature = "naga")]
    NagaCompile(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LibraryLoad(e) => write!(f, "failed to load Vulkan library: {e}"),
            Self::MissingFunction(name) => write!(f, "Vulkan function not loaded: {name}"),
            Self::Vk(result) => write!(f, "Vulkan call failed: {result:?}"),
            Self::InvalidString(e) => write!(f, "invalid C string: {e}"),
            #[cfg(feature = "naga")]
            Self::NagaCompile(s) => write!(f, "GLSL compilation failed: {s}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LibraryLoad(e) => Some(e),
            Self::InvalidString(e) => Some(e),
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

/// Convenience alias for `Result<T, spock::safe::Error>`.
pub type Result<T> = std::result::Result<T, Error>;

/// Helper: convert a `VkResult` into a `Result<()>`.
pub(crate) fn check(result: VkResult) -> Result<()> {
    if result == VkResult::SUCCESS {
        Ok(())
    } else {
        Err(Error::Vk(result))
    }
}
