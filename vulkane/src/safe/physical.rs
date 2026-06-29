//! Safe wrapper for `VkPhysicalDevice` — a GPU discovered by the
//! Vulkan loader.
//!
//! A [`PhysicalDevice`] represents a piece of hardware (or software
//! rasterizer) that can run Vulkan commands. Use it to:
//!
//! - Query properties: [`properties()`](PhysicalDevice::properties),
//!   [`memory_properties()`](PhysicalDevice::memory_properties)
//! - Find queue families:
//!   [`find_queue_family(QueueFlags::GRAPHICS)`](PhysicalDevice::find_queue_family)
//! - Find memory types:
//!   [`find_memory_type(bits, flags)`](PhysicalDevice::find_memory_type)
//! - Create a logical device:
//!   [`create_device(info)`](PhysicalDevice::create_device)
//!
//! ```ignore
//! let physical = instance
//!     .enumerate_physical_devices()?
//!     .into_iter()
//!     .find(|pd| pd.find_queue_family(QueueFlags::GRAPHICS).is_some())
//!     .ok_or("no compatible GPU")?;
//!
//! println!("Using: {}", physical.properties().device_name());
//! ```

use super::instance::{ApiVersion, ExtensionProperties, InstanceInner};
use super::{Device, DeviceCreateInfo, Error, Result, check};
use crate::raw::PNextChainable;
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

// Safety: VkPhysicalDevice is documented by the Vulkan spec as safe to
// share between threads. The InstanceInner is already Send + Sync.
unsafe impl Send for PhysicalDevice {}
unsafe impl Sync for PhysicalDevice {}

impl PhysicalDevice {
    pub(crate) fn new(instance: Arc<InstanceInner>, handle: VkPhysicalDevice) -> Self {
        Self { instance, handle }
    }

    /// Returns the raw `VkPhysicalDevice` handle.
    pub fn raw(&self) -> VkPhysicalDevice {
        self.handle
    }

    /// Returns a reference to the parent instance's dispatch table.
    /// Used by [`Allocator`](super::Allocator) to look up
    /// `vkGetPhysicalDeviceMemoryProperties`. Hidden from rustdoc — not
    /// part of the stable public API.
    #[doc(hidden)]
    pub fn instance(&self) -> &VkInstanceDispatchTable {
        &self.instance.dispatch
    }

    /// Query the physical device's supported Vulkan 1.0 feature bits.
    /// Combine with the [`DeviceFeatures`](super::DeviceFeatures) builder
    /// when enabling all device-supported features.
    pub fn supported_features(&self) -> VkPhysicalDeviceFeatures {
        let get = self
            .instance
            .dispatch
            .vkGetPhysicalDeviceFeatures
            .expect("vkGetPhysicalDeviceFeatures is required by Vulkan 1.0");
        // Safety: handle is valid; struct will be fully overwritten.
        let mut feats: VkPhysicalDeviceFeatures = unsafe { std::mem::zeroed() };
        unsafe { get(self.handle, &mut feats) };
        feats
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

    /// Enumerate the device-level extensions exposed by this physical device.
    pub fn enumerate_extension_properties(&self) -> Result<Vec<ExtensionProperties>> {
        let enumerate = self
            .instance
            .dispatch
            .vkEnumerateDeviceExtensionProperties
            .ok_or(Error::MissingFunction(
                "vkEnumerateDeviceExtensionProperties",
            ))?;

        let mut count: u32 = 0;
        // Safety: count query, output ptr is null. Layer name null = core extensions.
        check(unsafe {
            enumerate(
                self.handle,
                std::ptr::null(),
                &mut count,
                std::ptr::null_mut(),
            )
        })?;
        let mut raw: Vec<VkExtensionProperties> =
            vec![unsafe { std::mem::zeroed() }; count as usize];
        // Safety: raw has space for `count` elements.
        check(unsafe { enumerate(self.handle, std::ptr::null(), &mut count, raw.as_mut_ptr()) })?;
        Ok(raw.into_iter().map(ExtensionProperties::from_raw).collect())
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

    /// Enumerate the supported cooperative matrix shapes (`VK_KHR_cooperative_matrix`).
    ///
    /// Cooperative matrices are GPU primitives for matrix-multiply-and-
    /// accumulate operations — the building block of modern ML and
    /// signal-processing workloads. Each [`CooperativeMatrixProperties`]
    /// entry describes one supported `(M, N, K, A_type, B_type, C_type,
    /// Result_type)` shape that the device's compute units can execute
    /// natively.
    ///
    /// Returns an empty `Vec` if the device does not expose
    /// `VK_KHR_cooperative_matrix`.
    ///
    /// # Safety
    ///
    /// The caller must have created the parent [`Instance`](super::Instance)
    /// with `VK_KHR_cooperative_matrix` in
    /// [`InstanceCreateInfo::enabled_extensions`](super::InstanceCreateInfo::enabled_extensions).
    /// Calling this function when the extension was not enabled is
    /// undefined behaviour on some implementations (notably software
    /// rasterizers like Lavapipe), even though the function pointer may
    /// have been loaded by the loader. The Vulkan loader will happily
    /// hand back a stub for any KHR function name it knows about; the
    /// stub may then crash when called against a device that doesn't
    /// implement the extension.
    pub unsafe fn cooperative_matrix_properties(&self) -> Vec<CooperativeMatrixProperties> {
        let Some(get) = self
            .instance
            .dispatch
            .vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
        else {
            return Vec::new();
        };
        let mut count: u32 = 0;
        // Safety: count query, output ptr is null.
        if unsafe { get(self.handle, &mut count, std::ptr::null_mut()) } != VkResult::SUCCESS {
            return Vec::new();
        }
        // Note: cannot use `mem::zeroed()` here because `VkScopeKHR` has
        // no zero variant and the generated `Default` produces
        // `SCOPE_DEVICE_KHR`. Initialize via the per-struct Default impl
        // and patch sType in one shot.
        let mut raw: Vec<VkCooperativeMatrixPropertiesKHR> = (0..count as usize)
            .map(|_| VkCooperativeMatrixPropertiesKHR {
                sType: VkStructureType::STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
                ..Default::default()
            })
            .collect();
        // Safety: raw has space for `count` elements.
        if unsafe { get(self.handle, &mut count, raw.as_mut_ptr()) } != VkResult::SUCCESS {
            return Vec::new();
        }
        raw.into_iter()
            .map(|r| CooperativeMatrixProperties { raw: r })
            .collect()
    }

    /// Query per-heap memory budget via `VK_EXT_memory_budget`.
    ///
    /// `VK_EXT_memory_budget` lets the driver report a soft per-heap
    /// "budget" the application should respect — exceeding it isn't an
    /// error, but the driver may start swapping or evicting if it's
    /// repeatedly violated. The reported `usage` is the driver's estimate
    /// of how many bytes are currently allocated from each heap.
    ///
    /// Returns `None` if `vkGetPhysicalDeviceMemoryProperties2` is not
    /// loaded (Vulkan 1.0 without `VK_KHR_get_physical_device_properties2`)
    /// — the call always returns *something* useful when the loader has
    /// `vkGetPhysicalDeviceMemoryProperties2` available, but the budget
    /// numbers will only be meaningful when `VK_EXT_memory_budget` is
    /// enabled at instance creation time.
    pub fn memory_budget(&self) -> Option<MemoryBudget> {
        let get2 = self
            .instance
            .dispatch
            .vkGetPhysicalDeviceMemoryProperties2?;

        // Output-direction chain: driver writes into both structs.
        let mut budget_chain = crate::safe::PNextChain::new();
        budget_chain.push(VkPhysicalDeviceMemoryBudgetPropertiesEXT::new_pnext());
        let mut props2 = VkPhysicalDeviceMemoryProperties2 {
            sType: VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
            pNext: budget_chain.head_mut(),
            ..Default::default()
        };
        // Safety: handle is valid; props2 and the chain both live for
        // the call's duration.
        unsafe { get2(self.handle, &mut props2) };

        let budget = budget_chain.get::<VkPhysicalDeviceMemoryBudgetPropertiesEXT>()?;
        Some(MemoryBudget {
            heap_count: props2.memoryProperties.memoryHeapCount,
            budget: budget.heapBudget,
            usage: budget.heapUsage,
        })
    }

    /// Query this device's stable identity — UUIDs, the LUID (where the
    /// platform marks it valid), and the PCI bus address (where the
    /// device advertises `VK_EXT_pci_bus_info`).
    ///
    /// This is the **join key** for correlating a `VkPhysicalDevice` with
    /// the same GPU as seen by out-of-band sources and other APIs:
    /// `device_uuid` matches NVML / CUDA / OpenGL (`nvmlDeviceGetUUID`);
    /// `device_luid` matches a DXGI adapter or D3DKMT node on Windows;
    /// `pci` matches a Linux `/sys/bus/pci/devices/...` node (and thus
    /// amdgpu `gpu_busy_percent`). Vulkan itself exposes **no GPU
    /// load / utilization / queue-depth query** beyond the VRAM
    /// [`memory_budget`](Self::memory_budget) — identity is the hook that
    /// lets a caller go ask the right vendor/OS source out of band.
    ///
    /// Returns `None` only when `vkGetPhysicalDeviceProperties2` is
    /// unavailable (Vulkan 1.0 with no
    /// `VK_KHR_get_physical_device_properties2`). Otherwise the UUID
    /// fields are always populated; [`device_luid`](DeviceIdentity::device_luid)
    /// is `Some` only when the platform reports it valid (Windows), and
    /// [`pci`](DeviceIdentity::pci) is `Some` only when the device
    /// advertises `VK_EXT_pci_bus_info`.
    pub fn device_identity(&self) -> Option<DeviceIdentity> {
        let get2 = self.instance.dispatch.vkGetPhysicalDeviceProperties2?;

        // Only chain the PCI-bus-info struct when the device actually
        // advertises the extension. A driver that doesn't implement it
        // leaves the struct untouched, so chaining it unconditionally
        // would report a bogus `0000:00:00.0` instead of an honest
        // `None`.
        let has_pci = self
            .enumerate_extension_properties()
            .map(|exts| exts.iter().any(|e| e.name() == "VK_EXT_pci_bus_info"))
            .unwrap_or(false);

        let mut chain = crate::safe::PNextChain::new();
        chain.push(VkPhysicalDeviceIDProperties::new_pnext());
        if has_pci {
            chain.push(VkPhysicalDevicePCIBusInfoPropertiesEXT::new_pnext());
        }
        let mut props2 = VkPhysicalDeviceProperties2 {
            sType: VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            pNext: chain.head_mut(),
            ..Default::default()
        };
        // Safety: handle valid; props2 + chain live for the call.
        unsafe { get2(self.handle, &mut props2) };

        let id = chain.get::<VkPhysicalDeviceIDProperties>()?;
        let pci = if has_pci {
            chain
                .get::<VkPhysicalDevicePCIBusInfoPropertiesEXT>()
                .map(|p| PciBusInfo {
                    domain: p.pciDomain,
                    bus: p.pciBus,
                    device: p.pciDevice,
                    function: p.pciFunction,
                })
        } else {
            None
        };

        Some(DeviceIdentity {
            device_uuid: id.deviceUUID,
            driver_uuid: id.driverUUID,
            device_luid: (id.deviceLUIDValid != 0).then_some(id.deviceLUID),
            device_node_mask: id.deviceNodeMask,
            pci,
        })
    }

    /// Query shader integer-dot-product acceleration properties
    /// (`VK_KHR_shader_integer_dot_product`, core in Vulkan 1.3).
    ///
    /// Describes which integer-dot-product SPIR-V ops the device
    /// accelerates natively. For ML workloads the 8-bit and 4×8-bit
    /// packed variants are what you typically care about: they map
    /// directly onto int8-quantized matmul and convolution kernels.
    ///
    /// Returns `None` if `vkGetPhysicalDeviceProperties2` is not
    /// available (Vulkan 1.0 without
    /// `VK_KHR_get_physical_device_properties2`). The boolean fields
    /// will be `false` across the board on devices that do not
    /// implement the extension — a safe all-zeros reading.
    pub fn shader_integer_dot_product_properties(
        &self,
    ) -> Option<ShaderIntegerDotProductProperties> {
        let get2 = self.instance.dispatch.vkGetPhysicalDeviceProperties2?;

        let mut chain = crate::safe::PNextChain::new();
        chain.push(VkPhysicalDeviceShaderIntegerDotProductProperties::new_pnext());
        let mut props2 = VkPhysicalDeviceProperties2 {
            sType: VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            pNext: chain.head_mut(),
            ..Default::default()
        };
        // Safety: handle valid; props2 + chain live for the call.
        unsafe { get2(self.handle, &mut props2) };

        let raw = chain.get::<VkPhysicalDeviceShaderIntegerDotProductProperties>()?;
        Some(ShaderIntegerDotProductProperties {
            dot_product_8bit_unsigned: raw.integerDotProduct8BitUnsignedAccelerated != 0,
            dot_product_8bit_signed: raw.integerDotProduct8BitSignedAccelerated != 0,
            dot_product_8bit_mixed: raw.integerDotProduct8BitMixedSignednessAccelerated != 0,
            dot_product_4x8bit_packed_unsigned: raw
                .integerDotProduct4x8BitPackedUnsignedAccelerated
                != 0,
            dot_product_4x8bit_packed_signed: raw.integerDotProduct4x8BitPackedSignedAccelerated
                != 0,
            dot_product_4x8bit_packed_mixed: raw
                .integerDotProduct4x8BitPackedMixedSignednessAccelerated
                != 0,
            dot_product_16bit_unsigned: raw.integerDotProduct16BitUnsignedAccelerated != 0,
            dot_product_16bit_signed: raw.integerDotProduct16BitSignedAccelerated != 0,
            dot_product_32bit_unsigned: raw.integerDotProduct32BitUnsignedAccelerated != 0,
            dot_product_32bit_signed: raw.integerDotProduct32BitSignedAccelerated != 0,
            dot_product_64bit_unsigned: raw.integerDotProduct64BitUnsignedAccelerated != 0,
            dot_product_64bit_signed: raw.integerDotProduct64BitSignedAccelerated != 0,
            dot_product_accumulating_sat_8bit_signed: raw
                .integerDotProductAccumulatingSaturating8BitSignedAccelerated
                != 0,
            dot_product_accumulating_sat_8bit_unsigned: raw
                .integerDotProductAccumulatingSaturating8BitUnsignedAccelerated
                != 0,
            dot_product_accumulating_sat_4x8bit_packed_signed: raw
                .integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated
                != 0,
            dot_product_accumulating_sat_4x8bit_packed_unsigned: raw
                .integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated
                != 0,
        })
    }

    /// Query `VK_KHR_ray_tracing_pipeline` runtime properties — SBT
    /// handle size, alignment, recursion limits.
    ///
    /// All are required to lay out a shader binding table correctly.
    /// Returns `None` if `vkGetPhysicalDeviceProperties2` is not
    /// available; returns a struct with all-zero values on a driver
    /// that doesn't implement the extension.
    pub fn ray_tracing_pipeline_properties(
        &self,
    ) -> Option<super::ray_tracing_pipeline::RayTracingPipelineProperties> {
        let get2 = self.instance.dispatch.vkGetPhysicalDeviceProperties2?;
        let mut chain = crate::safe::PNextChain::new();
        chain.push(VkPhysicalDeviceRayTracingPipelinePropertiesKHR::new_pnext());
        let mut props2 = VkPhysicalDeviceProperties2 {
            sType: VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            pNext: chain.head_mut(),
            ..Default::default()
        };
        // Safety: handle valid; props2 + chain live for the call.
        unsafe { get2(self.handle, &mut props2) };
        let raw = chain.get::<VkPhysicalDeviceRayTracingPipelinePropertiesKHR>()?;
        Some(super::ray_tracing_pipeline::RayTracingPipelineProperties {
            shader_group_handle_size: raw.shaderGroupHandleSize,
            max_ray_recursion_depth: raw.maxRayRecursionDepth,
            max_shader_group_stride: raw.maxShaderGroupStride,
            shader_group_base_alignment: raw.shaderGroupBaseAlignment,
            shader_group_handle_alignment: raw.shaderGroupHandleAlignment,
            max_ray_dispatch_invocation_count: raw.maxRayDispatchInvocationCount,
            max_ray_hit_attribute_size: raw.maxRayHitAttributeSize,
        })
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

/// Per-heap memory budget snapshot from `VK_EXT_memory_budget`.
///
/// `budget[i]` is the soft cap the driver suggests respecting for heap
/// `i`; `usage[i]` is the driver's estimate of currently-allocated bytes.
/// Both arrays are length `heap_count`. Heap indices in this struct match
/// the indices returned by [`PhysicalDevice::memory_properties`].
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    pub heap_count: u32,
    pub budget: [u64; 16],
    pub usage: [u64; 16],
}

/// Safe view of
/// [`VkPhysicalDeviceShaderIntegerDotProductProperties`](crate::raw::bindings::VkPhysicalDeviceShaderIntegerDotProductProperties).
///
/// Each field is `true` when the device accelerates that SPIR-V dot-product
/// variant natively. For ML workloads the 8-bit and 4×8-bit-packed signals
/// are the high-value ones — they gate whether int8 quantized matmul /
/// convolution compiles down to hardware SIMD-dot (e.g. DP4a on AMD,
/// __dp4a on NVIDIA) or a slower fallback.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShaderIntegerDotProductProperties {
    pub dot_product_8bit_unsigned: bool,
    pub dot_product_8bit_signed: bool,
    pub dot_product_8bit_mixed: bool,
    pub dot_product_4x8bit_packed_unsigned: bool,
    pub dot_product_4x8bit_packed_signed: bool,
    pub dot_product_4x8bit_packed_mixed: bool,
    pub dot_product_16bit_unsigned: bool,
    pub dot_product_16bit_signed: bool,
    pub dot_product_32bit_unsigned: bool,
    pub dot_product_32bit_signed: bool,
    pub dot_product_64bit_unsigned: bool,
    pub dot_product_64bit_signed: bool,
    pub dot_product_accumulating_sat_8bit_signed: bool,
    pub dot_product_accumulating_sat_8bit_unsigned: bool,
    pub dot_product_accumulating_sat_4x8bit_packed_signed: bool,
    pub dot_product_accumulating_sat_4x8bit_packed_unsigned: bool,
}

impl ShaderIntegerDotProductProperties {
    /// `true` if the device accelerates *any* int8 or 4×8-bit-packed
    /// dot-product variant — the minimum bar for hardware-accelerated
    /// int8-quantized inference.
    pub fn has_any_int8_acceleration(&self) -> bool {
        self.dot_product_8bit_signed
            || self.dot_product_8bit_unsigned
            || self.dot_product_8bit_mixed
            || self.dot_product_4x8bit_packed_signed
            || self.dot_product_4x8bit_packed_unsigned
            || self.dot_product_4x8bit_packed_mixed
    }
}

impl MemoryBudget {
    /// Total budget summed across all heaps.
    pub fn total_budget(&self) -> u64 {
        self.budget[..self.heap_count as usize].iter().sum()
    }
    /// Total usage summed across all heaps.
    pub fn total_usage(&self) -> u64 {
        self.usage[..self.heap_count as usize].iter().sum()
    }
}

/// Stable identity of a physical device, from
/// [`PhysicalDevice::device_identity`].
///
/// Sourced from `VkPhysicalDeviceIDProperties` (Vulkan 1.1 core) plus,
/// when the device advertises `VK_EXT_pci_bus_info`, its PCI bus address.
/// The point of this struct is **out-of-band correlation**: Vulkan can
/// tell you *which* GPU you hold but not how busy it is, so identity is
/// what lets a caller match this device against a vendor/OS telemetry
/// source (NVML by UUID, DXGI/D3DKMT by LUID, Linux sysfs by PCI address)
/// — or against the same device exposed through CUDA, D3D, or OpenGL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceIdentity {
    /// Universally-unique device identifier, stable across processes,
    /// reboots, and driver reinstalls. The same value the device reports
    /// to CUDA / OpenGL and to `nvmlDeviceGetUUID`. Always populated.
    pub device_uuid: [u8; 16],
    /// UUID of the driver build. Devices driven by the same driver share
    /// this; useful for telling two ICDs apart.
    pub driver_uuid: [u8; 16],
    /// Locally-unique device identifier — `Some` only on platforms that
    /// mark it valid (Windows, via `deviceLUIDValid`). Pair with
    /// [`device_node_mask`](Self::device_node_mask) to match a DXGI
    /// adapter (`IDXGIAdapter::GetDesc`) or a D3DKMT node. `None` on
    /// Linux and other LUID-less platforms — match by
    /// [`device_uuid`](Self::device_uuid) or [`pci`](Self::pci) there.
    pub device_luid: Option<[u8; 8]>,
    /// Node mask scoping [`device_luid`](Self::device_luid) within a
    /// linked-adapter set. Meaningful only when `device_luid` is `Some`.
    pub device_node_mask: u32,
    /// PCI bus address, present only when the device advertises
    /// `VK_EXT_pci_bus_info`. `None` on software rasterizers and any
    /// platform that doesn't expose PCI topology.
    pub pci: Option<PciBusInfo>,
}

/// PCI bus address of a physical device — `domain:bus:device.function`,
/// from `VK_EXT_pci_bus_info`.
///
/// On Linux this maps directly to the
/// `/sys/bus/pci/devices/<domain>:<bus>:<device>.<function>` sysfs node
/// (and so to the amdgpu `gpu_busy_percent` file); on any platform it
/// pins the device to a stable hardware slot for correlation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PciBusInfo {
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
    pub function: u32,
}

/// A Vulkan physical device group: a set of one or more physical devices
/// that share `VkDeviceMemory` allocations and can run in tandem with
/// per-allocation / per-submission `device_mask` parameters.
///
/// Single-device "groups" of length 1 are the overwhelmingly common case
/// and behave identically to a non-grouped [`PhysicalDevice`]. Multi-GPU
/// systems (e.g. dual SLI / CrossFire / explicit-multi-GPU) expose
/// genuine groups via
/// [`Instance::enumerate_physical_device_groups`](super::Instance::enumerate_physical_device_groups).
///
/// Use [`PhysicalDeviceGroup::create_device`] to create a [`Device`]
/// that internally tracks every physical device in the group. Single
/// physical devices created via [`PhysicalDevice::create_device`]
/// produce a [`Device`] that internally wraps a singleton group, so
/// every code path in the safe wrapper sees the same shape.
#[derive(Clone)]
#[allow(dead_code)] // `instance` keeps the parent alive even if unread.
pub struct PhysicalDeviceGroup {
    pub(crate) instance: Arc<InstanceInner>,
    pub(crate) physical_devices: Vec<PhysicalDevice>,
    pub(crate) subset_allocation: bool,
}

impl PhysicalDeviceGroup {
    /// Returns the physical devices in this group, in the order
    /// `vkEnumeratePhysicalDeviceGroups` reported them. Always at
    /// least one element; usually exactly one on consumer hardware.
    pub fn physical_devices(&self) -> &[PhysicalDevice] {
        &self.physical_devices
    }

    /// Number of physical devices in this group.
    pub fn count(&self) -> u32 {
        self.physical_devices.len() as u32
    }

    /// `true` if the implementation supports subset memory allocations
    /// across this group (allowing per-device-mask allocation flags).
    /// Always `false` on single-device groups.
    pub fn supports_subset_allocation(&self) -> bool {
        self.subset_allocation
    }

    /// Create a logical [`Device`] from this group.
    pub fn create_device(&self, info: DeviceCreateInfo<'_>) -> Result<Device> {
        Device::new_group(self, info)
    }
}

impl std::fmt::Debug for PhysicalDeviceGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicalDeviceGroup")
            .field("count", &self.count())
            .field("subset_allocation", &self.subset_allocation)
            .finish()
    }
}

/// One supported cooperative-matrix shape, as returned by
/// [`PhysicalDevice::cooperative_matrix_properties`].
#[derive(Clone)]
pub struct CooperativeMatrixProperties {
    raw: VkCooperativeMatrixPropertiesKHR,
}

impl CooperativeMatrixProperties {
    pub fn m_size(&self) -> u32 {
        self.raw.MSize
    }
    pub fn n_size(&self) -> u32 {
        self.raw.NSize
    }
    pub fn k_size(&self) -> u32 {
        self.raw.KSize
    }
    /// Component type of operand A. The value is the raw
    /// `VkComponentTypeKHR` enum.
    pub fn a_type(&self) -> VkComponentTypeKHR {
        self.raw.AType
    }
    pub fn b_type(&self) -> VkComponentTypeKHR {
        self.raw.BType
    }
    pub fn c_type(&self) -> VkComponentTypeKHR {
        self.raw.CType
    }
    pub fn result_type(&self) -> VkComponentTypeKHR {
        self.raw.ResultType
    }
    /// Whether the implementation saturates accumulator overflow.
    pub fn saturating_accumulation(&self) -> bool {
        self.raw.saturatingAccumulation != 0
    }
    pub fn scope(&self) -> VkScopeKHR {
        self.raw.scope
    }
}

impl std::fmt::Debug for CooperativeMatrixProperties {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CooperativeMatrixProperties")
            .field("M", &self.m_size())
            .field("N", &self.n_size())
            .field("K", &self.k_size())
            .field("AType", &self.a_type())
            .field("BType", &self.b_type())
            .field("CType", &self.c_type())
            .field("ResultType", &self.result_type())
            .finish()
    }
}

// Re-use Error so callers don't need a separate import.
#[allow(dead_code)]
fn _ensure_error_is_used(_: Error) {}
