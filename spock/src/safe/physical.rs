//! Safe wrapper for `VkPhysicalDevice`.

use super::instance::{ApiVersion, InstanceInner};
use super::{Device, DeviceCreateInfo, Error, Result};
use crate::raw::bindings::*;
use std::ffi::CStr;
use std::sync::Arc;

/// A handle to a Vulkan physical device (a GPU or other implementation).
///
/// Physical devices are not destroyed; they are owned by the instance.
/// This handle is alive as long as its parent [`Instance`](super::Instance) is alive.
#[derive(Clone)]
pub struct PhysicalDevice {
    pub(crate) instance: Arc<InstanceInner>,
    pub(crate) handle: VkPhysicalDevice,
}

impl PhysicalDevice {
    pub(crate) fn new(instance: Arc<InstanceInner>, handle: VkPhysicalDevice) -> Self {
        Self { instance, handle }
    }

    /// Returns the raw `VkPhysicalDevice` handle.
    pub fn raw(&self) -> VkPhysicalDevice {
        self.handle
    }

    /// Query the physical device's properties (name, vendor, API version, etc.).
    pub fn properties(&self) -> PhysicalDeviceProperties {
        let get = self
            .instance
            .dispatch
            .vkGetPhysicalDeviceProperties
            .expect("vkGetPhysicalDeviceProperties is required by Vulkan 1.0");

        // Safety: handle is valid (came from vkEnumeratePhysicalDevices),
        // raw is freshly-zeroed but Vulkan will overwrite all fields.
        let mut raw: VkPhysicalDeviceProperties = unsafe { std::mem::zeroed() };
        unsafe { get(self.handle, &mut raw) };
        PhysicalDeviceProperties { raw }
    }

    /// Query the physical device's queue family properties.
    pub fn queue_family_properties(&self) -> Vec<QueueFamilyProperties> {
        let get = self
            .instance
            .dispatch
            .vkGetPhysicalDeviceQueueFamilyProperties
            .expect("vkGetPhysicalDeviceQueueFamilyProperties is required by Vulkan 1.0");

        let mut count: u32 = 0;
        // Safety: count query, output ptr is null.
        unsafe { get(self.handle, &mut count, std::ptr::null_mut()) };

        // Safety: each element will be overwritten by the driver.
        let mut raw: Vec<VkQueueFamilyProperties> =
            vec![unsafe { std::mem::zeroed() }; count as usize];
        // Safety: raw has space for `count` elements.
        unsafe { get(self.handle, &mut count, raw.as_mut_ptr()) };

        raw.into_iter()
            .map(|r| QueueFamilyProperties { raw: r })
            .collect()
    }

    /// Query the physical device's memory properties (heaps and types).
    pub fn memory_properties(&self) -> MemoryProperties {
        let get = self
            .instance
            .dispatch
            .vkGetPhysicalDeviceMemoryProperties
            .expect("vkGetPhysicalDeviceMemoryProperties is required by Vulkan 1.0");

        // Safety: driver will overwrite all relevant fields.
        let mut raw: VkPhysicalDeviceMemoryProperties = unsafe { std::mem::zeroed() };
        unsafe { get(self.handle, &mut raw) };
        MemoryProperties { raw }
    }

    /// Create a logical [`Device`] from this physical device.
    pub fn create_device(&self, info: DeviceCreateInfo<'_>) -> Result<Device> {
        Device::new(self, info)
    }

    /// Find the index of the first queue family that supports the given flags.
    pub fn find_queue_family(&self, required: QueueFlags) -> Option<u32> {
        self.queue_family_properties()
            .iter()
            .enumerate()
            .find(|(_, qf)| qf.queue_flags().contains(required))
            .map(|(i, _)| i as u32)
    }

    /// Find a "dedicated" compute queue family — one that supports
    /// `COMPUTE` but **not** `GRAPHICS`. On modern NVIDIA / AMD GPUs this
    /// returns the async-compute queue family, which can run compute work
    /// concurrently with the universal graphics+compute queue.
    ///
    /// If no dedicated compute family exists (most integrated GPUs and
    /// software rasterizers fall in this bucket), this falls back to the
    /// first family that supports `COMPUTE` at all — i.e. the same answer
    /// as `find_queue_family(QueueFlags::COMPUTE)`. Returns `None` only when
    /// the device exposes no compute-capable queues, which should not
    /// happen on any conformant Vulkan implementation.
    pub fn find_dedicated_compute_queue(&self) -> Option<u32> {
        let families = self.queue_family_properties();
        // Prefer compute-without-graphics.
        for (i, qf) in families.iter().enumerate() {
            let flags = qf.queue_flags();
            if flags.contains(QueueFlags::COMPUTE) && !flags.contains(QueueFlags::GRAPHICS) {
                return Some(i as u32);
            }
        }
        // Fallback: any compute queue.
        for (i, qf) in families.iter().enumerate() {
            if qf.queue_flags().contains(QueueFlags::COMPUTE) {
                return Some(i as u32);
            }
        }
        None
    }

    /// Find a "dedicated" transfer queue family — one that supports
    /// `TRANSFER` but **not** `GRAPHICS` or `COMPUTE`. On discrete GPUs
    /// this is typically the DMA / copy engine and is the right place to
    /// run staging-buffer uploads concurrently with compute work.
    ///
    /// Falls back to `find_queue_family(QueueFlags::TRANSFER)` (which the
    /// Vulkan spec guarantees succeeds for any graphics-or-compute family).
    pub fn find_dedicated_transfer_queue(&self) -> Option<u32> {
        let families = self.queue_family_properties();
        for (i, qf) in families.iter().enumerate() {
            let flags = qf.queue_flags();
            if flags.contains(QueueFlags::TRANSFER)
                && !flags.contains(QueueFlags::GRAPHICS)
                && !flags.contains(QueueFlags::COMPUTE)
            {
                return Some(i as u32);
            }
        }
        for (i, qf) in families.iter().enumerate() {
            if qf.queue_flags().contains(QueueFlags::TRANSFER) {
                return Some(i as u32);
            }
        }
        None
    }

    /// The number of nanoseconds per timestamp tick on this device.
    ///
    /// `vkCmdWriteTimestamp` writes a `u64` count of implementation-defined
    /// ticks; multiply by this value to get nanoseconds. Returns `0.0` on
    /// devices that do not support timestamps at all (which is rare — most
    /// modern GPUs do).
    pub fn timestamp_period(&self) -> f32 {
        self.properties().timestamp_period()
    }

    /// Find the index of the first memory type that has all the required
    /// property flags AND is allowed by the memory_type_bits mask.
    ///
    /// `memory_type_bits` typically comes from a `VkMemoryRequirements`
    /// returned by `vkGetBufferMemoryRequirements` etc.
    pub fn find_memory_type(
        &self,
        memory_type_bits: u32,
        required: super::MemoryPropertyFlags,
    ) -> Option<u32> {
        let props = self.memory_properties();
        for i in 0..props.type_count() {
            let allowed = (memory_type_bits & (1 << i)) != 0;
            if allowed && props.memory_type(i).property_flags().contains(required) {
                return Some(i);
            }
        }
        None
    }
}

/// Strongly-typed wrapper around `VkPhysicalDeviceProperties`.
#[derive(Clone)]
pub struct PhysicalDeviceProperties {
    raw: VkPhysicalDeviceProperties,
}

impl PhysicalDeviceProperties {
    /// Vulkan API version supported by the device.
    pub fn api_version(&self) -> ApiVersion {
        ApiVersion(self.raw.apiVersion)
    }

    /// Driver version (vendor-specific encoding).
    pub fn driver_version(&self) -> u32 {
        self.raw.driverVersion
    }

    /// PCI vendor ID.
    pub fn vendor_id(&self) -> u32 {
        self.raw.vendorID
    }

    /// PCI device ID.
    pub fn device_id(&self) -> u32 {
        self.raw.deviceID
    }

    /// The kind of physical device (discrete GPU, integrated, virtual, CPU, ...).
    pub fn device_type(&self) -> PhysicalDeviceType {
        PhysicalDeviceType(self.raw.deviceType)
    }

    /// Number of nanoseconds per timestamp tick. See
    /// [`PhysicalDevice::timestamp_period`].
    pub fn timestamp_period(&self) -> f32 {
        self.raw.limits.timestampPeriod
    }

    /// Maximum push constant size in bytes guaranteed by this device.
    /// Vulkan guarantees at least 128 bytes; most desktop GPUs report 256.
    pub fn max_push_constants_size(&self) -> u32 {
        self.raw.limits.maxPushConstantsSize
    }

    /// Human-readable device name.
    pub fn device_name(&self) -> String {
        // Safety: deviceName is a null-terminated array of c_char per spec.
        unsafe {
            CStr::from_ptr(self.raw.deviceName.as_ptr())
                .to_string_lossy()
                .into_owned()
        }
    }
}

/// Strongly-typed wrapper around `VkPhysicalDeviceType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalDeviceType(pub VkPhysicalDeviceType);

impl PhysicalDeviceType {
    pub const OTHER: Self = Self(VkPhysicalDeviceType::PHYSICAL_DEVICE_TYPE_OTHER);
    pub const INTEGRATED_GPU: Self =
        Self(VkPhysicalDeviceType::PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    pub const DISCRETE_GPU: Self = Self(VkPhysicalDeviceType::PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
    pub const VIRTUAL_GPU: Self = Self(VkPhysicalDeviceType::PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU);
    pub const CPU: Self = Self(VkPhysicalDeviceType::PHYSICAL_DEVICE_TYPE_CPU);
}

impl std::fmt::Debug for PhysicalDeviceProperties {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicalDeviceProperties")
            .field("device_name", &self.device_name())
            .field("device_type", &self.device_type())
            .field("api_version", &self.api_version())
            .field("driver_version", &self.driver_version())
            .field("vendor_id", &self.vendor_id())
            .field("device_id", &self.device_id())
            .finish()
    }
}

/// Strongly-typed wrapper around `VkQueueFlags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueFlags(pub u32);

impl QueueFlags {
    pub const GRAPHICS: Self = Self(0x1);
    pub const COMPUTE: Self = Self(0x2);
    pub const TRANSFER: Self = Self(0x4);
    pub const SPARSE_BINDING: Self = Self(0x8);

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for QueueFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Strongly-typed wrapper around `VkQueueFamilyProperties`.
#[derive(Clone)]
pub struct QueueFamilyProperties {
    raw: VkQueueFamilyProperties,
}

impl QueueFamilyProperties {
    pub fn queue_flags(&self) -> QueueFlags {
        QueueFlags(self.raw.queueFlags)
    }

    pub fn queue_count(&self) -> u32 {
        self.raw.queueCount
    }

    pub fn timestamp_valid_bits(&self) -> u32 {
        self.raw.timestampValidBits
    }
}

impl std::fmt::Debug for QueueFamilyProperties {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueueFamilyProperties")
            .field("queue_flags", &self.queue_flags())
            .field("queue_count", &self.queue_count())
            .finish()
    }
}

/// Strongly-typed wrapper around `VkPhysicalDeviceMemoryProperties`.
#[derive(Clone)]
pub struct MemoryProperties {
    raw: VkPhysicalDeviceMemoryProperties,
}

impl MemoryProperties {
    pub fn type_count(&self) -> u32 {
        self.raw.memoryTypeCount
    }

    pub fn heap_count(&self) -> u32 {
        self.raw.memoryHeapCount
    }

    pub fn memory_type(&self, index: u32) -> MemoryType {
        MemoryType {
            raw: self.raw.memoryTypes[index as usize],
        }
    }

    pub fn memory_heap(&self, index: u32) -> MemoryHeap {
        MemoryHeap {
            raw: self.raw.memoryHeaps[index as usize],
        }
    }
}

/// A memory type description.
#[derive(Clone)]
pub struct MemoryType {
    raw: VkMemoryType,
}

impl MemoryType {
    pub fn property_flags(&self) -> super::MemoryPropertyFlags {
        super::MemoryPropertyFlags(self.raw.propertyFlags)
    }

    pub fn heap_index(&self) -> u32 {
        self.raw.heapIndex
    }
}

/// A memory heap description.
#[derive(Clone)]
pub struct MemoryHeap {
    raw: VkMemoryHeap,
}

impl MemoryHeap {
    pub fn size(&self) -> u64 {
        self.raw.size
    }

    pub fn flags(&self) -> MemoryHeapFlags {
        MemoryHeapFlags(self.raw.flags)
    }
}

/// Strongly-typed wrapper around `VkMemoryHeapFlags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryHeapFlags(pub u32);

impl MemoryHeapFlags {
    pub const DEVICE_LOCAL: Self = Self(0x1);

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

// Re-use Error so callers don't need a separate import.
#[allow(dead_code)]
fn _ensure_error_is_used(_: Error) {}
