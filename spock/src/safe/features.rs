//! Device feature enable lists.
//!
//! Vulkan exposes optional functionality via "feature" booleans on the
//! `VkPhysicalDeviceFeatures` struct (Vulkan 1.0 core features) and via
//! per-version aggregate structs `VkPhysicalDeviceVulkan{11,12,13}Features`
//! (added in 1.1, 1.2, and 1.3 respectively). To use a feature, the
//! application must:
//!
//! 1. Verify the device supports it via `vkGetPhysicalDeviceFeatures2`.
//! 2. Re-pass the same feature struct chain when creating the device.
//!
//! [`DeviceFeatures`] wraps the entire chain (1.0 + optional 1.1 + 1.2 +
//! 1.3) and supplies builder-style accessors for the most commonly
//! requested features. Pass it via
//! [`DeviceCreateInfo::enabled_features`](super::DeviceCreateInfo::enabled_features).
//!
//! ```ignore
//! use spock::safe::{DeviceFeatures, DeviceCreateInfo, QueueCreateInfo};
//!
//! let features = DeviceFeatures::default()
//!     .with_buffer_device_address()
//!     .with_timeline_semaphore()
//!     .with_synchronization2();
//!
//! let device = physical.create_device(DeviceCreateInfo {
//!     queue_create_infos: &[QueueCreateInfo { /* ... */ }],
//!     enabled_features: Some(&features),
//!     ..Default::default()
//! })?;
//! ```

use crate::raw::bindings::*;

/// A set of device feature bits to enable at device creation time.
///
/// This wraps the Vulkan 1.0 `VkPhysicalDeviceFeatures` plus the optional
/// 1.1 / 1.2 / 1.3 aggregate feature structs. Use the `with_*` builder
/// methods to turn individual features on; unset fields default to
/// `VK_FALSE` (the Vulkan-mandated "do not enable" value).
///
/// `DeviceFeatures` does not enforce that the requested features are
/// actually supported by the physical device — that check happens inside
/// `vkCreateDevice` and surfaces as `ERROR_FEATURE_NOT_PRESENT`. Query
/// support up-front with [`PhysicalDevice::supported_features`](super::PhysicalDevice::supported_features)
/// if you want to fall back gracefully.
#[derive(Clone)]
pub struct DeviceFeatures {
    /// Vulkan 1.0 core features.
    pub features10: VkPhysicalDeviceFeatures,
    /// Vulkan 1.1 features (subgroup, multiview, 16-bit storage, ...).
    /// Always populated; only chained into device creation if any field
    /// is non-default.
    pub features11: VkPhysicalDeviceVulkan11Features,
    /// Vulkan 1.2 features (timeline semaphore, buffer device address,
    /// descriptor indexing, ...).
    pub features12: VkPhysicalDeviceVulkan12Features,
    /// Vulkan 1.3 features (synchronization2, dynamic rendering, ...).
    pub features13: VkPhysicalDeviceVulkan13Features,
}

impl Default for DeviceFeatures {
    fn default() -> Self {
        Self {
            features10: VkPhysicalDeviceFeatures::default(),
            features11: VkPhysicalDeviceVulkan11Features {
                sType: VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                ..Default::default()
            },
            features12: VkPhysicalDeviceVulkan12Features {
                sType: VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                ..Default::default()
            },
            features13: VkPhysicalDeviceVulkan13Features {
                sType: VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                ..Default::default()
            },
        }
    }
}

impl DeviceFeatures {
    /// New zero-initialised feature set. Same as [`Default::default`].
    pub fn new() -> Self {
        Self::default()
    }

    // ----- Vulkan 1.0 core features -----

    /// Enable [`robustBufferAccess`](VkPhysicalDeviceFeatures::robustBufferAccess).
    pub fn with_robust_buffer_access(mut self) -> Self {
        self.features10.robustBufferAccess = 1;
        self
    }

    /// Enable [`samplerAnisotropy`](VkPhysicalDeviceFeatures::samplerAnisotropy).
    pub fn with_sampler_anisotropy(mut self) -> Self {
        self.features10.samplerAnisotropy = 1;
        self
    }

    /// Enable [`fillModeNonSolid`](VkPhysicalDeviceFeatures::fillModeNonSolid)
    /// (line / point fill modes).
    pub fn with_fill_mode_non_solid(mut self) -> Self {
        self.features10.fillModeNonSolid = 1;
        self
    }

    /// Enable [`pipelineStatisticsQuery`](VkPhysicalDeviceFeatures::pipelineStatisticsQuery)
    /// — required for `QueryPool::pipeline_statistics`.
    pub fn with_pipeline_statistics_query(mut self) -> Self {
        self.features10.pipelineStatisticsQuery = 1;
        self
    }

    /// Enable [`shaderInt64`](VkPhysicalDeviceFeatures::shaderInt64).
    pub fn with_shader_int64(mut self) -> Self {
        self.features10.shaderInt64 = 1;
        self
    }

    /// Enable [`shaderFloat64`](VkPhysicalDeviceFeatures::shaderFloat64).
    pub fn with_shader_float64(mut self) -> Self {
        self.features10.shaderFloat64 = 1;
        self
    }

    // ----- Vulkan 1.2 features -----

    /// Enable
    /// [`bufferDeviceAddress`](VkPhysicalDeviceVulkan12Features::bufferDeviceAddress)
    /// — required to call [`Buffer::device_address`](super::Buffer::device_address).
    pub fn with_buffer_device_address(mut self) -> Self {
        self.features12.bufferDeviceAddress = 1;
        self
    }

    /// Enable
    /// [`timelineSemaphore`](VkPhysicalDeviceVulkan12Features::timelineSemaphore)
    /// — required to construct [`Semaphore::timeline`](super::Semaphore::timeline).
    pub fn with_timeline_semaphore(mut self) -> Self {
        self.features12.timelineSemaphore = 1;
        self
    }

    /// Enable
    /// [`hostQueryReset`](VkPhysicalDeviceVulkan12Features::hostQueryReset).
    pub fn with_host_query_reset(mut self) -> Self {
        self.features12.hostQueryReset = 1;
        self
    }

    /// Enable
    /// [`descriptorIndexing`](VkPhysicalDeviceVulkan12Features::descriptorIndexing)
    /// — required for bindless descriptor patterns.
    pub fn with_descriptor_indexing(mut self) -> Self {
        self.features12.descriptorIndexing = 1;
        self
    }

    /// Enable
    /// [`runtimeDescriptorArray`](VkPhysicalDeviceVulkan12Features::runtimeDescriptorArray).
    pub fn with_runtime_descriptor_array(mut self) -> Self {
        self.features12.runtimeDescriptorArray = 1;
        self
    }

    /// Enable
    /// [`shaderInt8`](VkPhysicalDeviceVulkan12Features::shaderInt8).
    pub fn with_shader_int8(mut self) -> Self {
        self.features12.shaderInt8 = 1;
        self
    }

    /// Enable
    /// [`shaderFloat16`](VkPhysicalDeviceVulkan12Features::shaderFloat16).
    pub fn with_shader_float16(mut self) -> Self {
        self.features12.shaderFloat16 = 1;
        self
    }

    // ----- Vulkan 1.3 features -----

    /// Enable
    /// [`synchronization2`](VkPhysicalDeviceVulkan13Features::synchronization2)
    /// — required for [`memory_barrier2`](super::CommandBufferRecording::memory_barrier2)
    /// and [`image_barrier2`](super::CommandBufferRecording::image_barrier2).
    pub fn with_synchronization2(mut self) -> Self {
        self.features13.synchronization2 = 1;
        self
    }

    /// Enable
    /// [`dynamicRendering`](VkPhysicalDeviceVulkan13Features::dynamicRendering)
    /// — lets you skip render-pass / framebuffer setup for graphics work.
    pub fn with_dynamic_rendering(mut self) -> Self {
        self.features13.dynamicRendering = 1;
        self
    }

    /// Enable
    /// [`maintenance4`](VkPhysicalDeviceVulkan13Features::maintenance4).
    pub fn with_maintenance4(mut self) -> Self {
        self.features13.maintenance4 = 1;
        self
    }

    // ----- Bulk setters -----

    /// Enable every feature flag in `features10` directly. Useful when
    /// you've queried a device's supported features via
    /// [`PhysicalDevice::supported_features`](super::PhysicalDevice::supported_features)
    /// and want to enable all of them (the default for some apps).
    pub fn with_all_features10(mut self, features10: VkPhysicalDeviceFeatures) -> Self {
        self.features10 = features10;
        self
    }
}

impl std::fmt::Debug for DeviceFeatures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceFeatures")
            .field(
                "buffer_device_address",
                &(self.features12.bufferDeviceAddress != 0),
            )
            .field(
                "timeline_semaphore",
                &(self.features12.timelineSemaphore != 0),
            )
            .field("synchronization2", &(self.features13.synchronization2 != 0))
            .field(
                "sampler_anisotropy",
                &(self.features10.samplerAnisotropy != 0),
            )
            .finish_non_exhaustive()
    }
}
