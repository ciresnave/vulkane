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
    /// Returns an empty `Vec` if the device does not advertise
    /// `VK_KHR_cooperative_matrix`.
    ///
    /// # Why this is safe to call unconditionally
    ///
    /// The Vulkan loader will hand back a non-null function pointer for
    /// any KHR entry point it knows the *name* of, whether or not the
    /// device implements it; calling such a stub against a device that
    /// doesn't implement the extension can crash (notably on software
    /// rasterizers like Lavapipe). A non-null dispatch entry is therefore
    /// not evidence the call is legal.
    ///
    /// This method does not rely on one. It first asks the device itself
    /// — [`enumerate_extension_properties`](Self::enumerate_extension_properties)
    /// — whether it advertises `VK_KHR_cooperative_matrix`, and returns
    /// an empty `Vec` when it does not, so the loader stub is never
    /// reached. That is the same honest-gating discipline
    /// [`device_identity`](Self::device_identity) applies to
    /// `VK_EXT_pci_bus_info`, and it makes the extension check an
    /// invariant of the call rather than an obligation on the caller.
    ///
    /// `Ok(empty)` is thus unambiguous: this device was asked and reports no
    /// cooperative-matrix support. Callers that previously wrapped this in
    /// `unsafe { .. }` after checking the extension list themselves can delete
    /// both the check and the block.
    ///
    /// # Why this returns `Result` rather than a `Vec`
    ///
    /// It used to return `Vec` and say an empty one was unambiguous. It was
    /// not: an absent entry point and a non-`SUCCESS` result also produced an
    /// empty `Vec`, so "the device has none" and "we could not tell" shared a
    /// spelling — a doc claiming a guarantee the code did not make.
    ///
    /// That is not only a documentation defect. `vulkane::kiss` derives the
    /// `cm-` field of a `vulkan:` token from this list, and an empty list
    /// spells `cm-none`. A transient query failure on a device with eleven
    /// shapes would therefore have derived a token asserting it has no
    /// cooperative-matrix support at all — and under KISS-CLASSIFY §6.8-0002
    /// byte-exact matching, a wrong token is not a degraded answer, it is a
    /// different cell. `Err` makes that unrepresentable.
    pub fn cooperative_matrix_properties(
        &self,
    ) -> crate::safe::Result<Vec<CooperativeMatrixProperties>> {
        // Gate on the device's own advertisement, not on the loader
        // handing us a function pointer — see the doc comment above.
        let advertised = self
            .enumerate_extension_properties()
            .map(|exts| exts.iter().any(|e| e.name() == "VK_KHR_cooperative_matrix"))
            .unwrap_or(false);
        if !advertised {
            // The one case that is genuinely "no support": the device was asked
            // and said no. Everything below is "we could not tell", which is a
            // different answer and must not share this one's spelling.
            return Ok(Vec::new());
        }

        let Some(get) = self
            .instance
            .dispatch
            .vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
        else {
            return Err(crate::safe::Error::MissingFunction(
                "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR",
            ));
        };
        let mut count: u32 = 0;
        // Safety: count query, output ptr is null.
        let r = unsafe { get(self.handle, &mut count, std::ptr::null_mut()) };
        if r != VkResult::SUCCESS {
            return Err(crate::safe::Error::Vk(r));
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
        let r = unsafe { get(self.handle, &mut count, raw.as_mut_ptr()) };
        if r != VkResult::SUCCESS {
            return Err(crate::safe::Error::Vk(r));
        }
        Ok(raw
            .into_iter()
            .map(|r| CooperativeMatrixProperties { raw: r })
            .collect())
    }

    /// Supported cooperative-*vector* combinations via
    /// `VK_NV_cooperative_vector`.
    ///
    /// Gated on the device's own advertisement rather than on a non-null
    /// dispatch entry, for the reason spelled out on
    /// [`cooperative_matrix_properties`](Self::cooperative_matrix_properties):
    /// a loader may hand back a stub for a call the device does not implement,
    /// and invoking it can crash.
    ///
    /// `Ok(empty)` means "this device was asked and reports no
    /// cooperative-vector support". A missing entry point or a non-`SUCCESS`
    /// result is `Err` — "we could not tell" is a different answer from "there
    /// is none" and does not share its spelling.
    ///
    /// This is a **separate capability** from cooperative matrix and is queried
    /// separately because the two report different things. It is also the only
    /// place the packed component types can appear: they are defined by
    /// `VK_NV_cooperative_vector` with no dependency on
    /// `VK_KHR_cooperative_matrix`, so no amount of reading cooperative-matrix
    /// properties will ever observe one.
    pub fn cooperative_vector_properties(
        &self,
    ) -> crate::safe::Result<Vec<CooperativeVectorProperties>> {
        let advertised = self
            .enumerate_extension_properties()
            .map(|exts| exts.iter().any(|e| e.name() == "VK_NV_cooperative_vector"))
            .unwrap_or(false);
        if !advertised {
            return Ok(Vec::new());
        }

        let Some(get) = self
            .instance
            .dispatch
            .vkGetPhysicalDeviceCooperativeVectorPropertiesNV
        else {
            return Err(crate::safe::Error::MissingFunction(
                "vkGetPhysicalDeviceCooperativeVectorPropertiesNV",
            ));
        };
        let mut count: u32 = 0;
        // Safety: count query, output ptr is null.
        let r = unsafe { get(self.handle, &mut count, std::ptr::null_mut()) };
        if r != VkResult::SUCCESS {
            return Err(crate::safe::Error::Vk(r));
        }
        let mut raw: Vec<VkCooperativeVectorPropertiesNV> = (0..count as usize)
            .map(|_| VkCooperativeVectorPropertiesNV {
                sType: VkStructureType::STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV,
                ..Default::default()
            })
            .collect();
        // Safety: raw has space for `count` elements.
        let r = unsafe { get(self.handle, &mut count, raw.as_mut_ptr()) };
        if r != VkResult::SUCCESS {
            return Err(crate::safe::Error::Vk(r));
        }
        Ok(raw
            .into_iter()
            .map(|r| CooperativeVectorProperties { raw: r })
            .collect())
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
    /// Returns `None` when `vkGetPhysicalDeviceProperties2` is unavailable
    /// (Vulkan 1.0 with no `VK_KHR_get_physical_device_properties2`), **or
    /// when the [`effective_api_version`](Self::effective_api_version) is
    /// below 1.1** — `VkPhysicalDeviceIDProperties` is 1.1 core, and a 1.0
    /// implementation leaves the chained struct untouched. That is the same
    /// reasoning applied to `VK_EXT_pci_bus_info` a few lines below, and it
    /// matters more here: a zeroed *optional* field reads as "absent", but a
    /// zeroed **join key** reads as a valid UUID that every device shares.
    ///
    /// Measured on one machine with two GPUs (2026-09-03): before this gate,
    /// a `V1_0` instance — the [`InstanceCreateInfo::api_version`](super::InstanceCreateInfo::api_version)
    /// **default** — reported `device_uuid` as all-zero for *both* an AMD
    /// Radeon 610M and an NVIDIA RTX 4070, so the two compared **equal**. At
    /// `V1_1` they are distinct. A caller correlating with NVML/CUDA would
    /// have matched the wrong GPU, or all of them.
    ///
    /// So raise [`InstanceCreateInfo::api_version`](super::InstanceCreateInfo::api_version)
    /// to 1.1 or higher if this returns `None` unexpectedly, as with
    /// [`subgroup_properties`](Self::subgroup_properties).
    ///
    /// Otherwise the UUID fields are populated;
    /// [`device_luid`](DeviceIdentity::device_luid) is `Some` only when the
    /// platform reports it valid (Windows), and
    /// [`pci`](DeviceIdentity::pci) is `Some` only when the device
    /// advertises `VK_EXT_pci_bus_info`.
    pub fn device_identity(&self) -> Option<DeviceIdentity> {
        let get2 = self.instance.dispatch.vkGetPhysicalDeviceProperties2?;

        // `VkPhysicalDeviceIDProperties` is Vulkan 1.1 core. On a 1.0
        // implementation the driver skips the unrecognized chained struct
        // and leaves it as we allocated it -- all zeros -- and `chain.get()`
        // hands that back indistinguishably from a real answer. Decline
        // instead: an all-zero UUID is not a conservative reading of a join
        // key, it is a value that makes every device compare equal.
        if (
            self.effective_api_version().major(),
            self.effective_api_version().minor(),
        ) < (1, 1)
        {
            return None;
        }

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

    /// The Vulkan version whose features are actually reachable through
    /// this handle: `min(instance apiVersion, device apiVersion)`.
    ///
    /// This is the version that governs property queries, and it is
    /// **not** the same as
    /// [`properties().api_version()`](PhysicalDeviceProperties::api_version).
    /// A Vulkan implementation must behave as the version the *instance*
    /// requested in `VkApplicationInfo::apiVersion`, so an instance
    /// created at 1.0 will leave 1.1+ `pNext` property structs untouched
    /// even on a device that reports 1.3 — the caller reads back a
    /// zeroed struct that looks like a real answer.
    ///
    /// Query methods that chain version-gated structs
    /// ([`subgroup_properties`](Self::subgroup_properties),
    /// [`driver_properties`](Self::driver_properties)) gate on this, so
    /// they return an honest `None` rather than a zeroed reading. If one
    /// of them declines on hardware you expect to support the feature,
    /// raise the `api_version` in
    /// [`InstanceCreateInfo`](super::InstanceCreateInfo) — which defaults
    /// to [`ApiVersion::V1_0`](super::ApiVersion::V1_0).
    pub fn effective_api_version(&self) -> ApiVersion {
        let device = self.properties().api_version();
        let instance = self.instance.api_version;
        if (instance.major(), instance.minor()) <= (device.major(), device.minor()) {
            instance
        } else {
            device
        }
    }

    /// Query the device's subgroup ("wave" / "warp") properties —
    /// `VkPhysicalDeviceSubgroupProperties` (Vulkan 1.1 core), plus the
    /// `VkPhysicalDeviceSubgroupSizeControlProperties` size range
    /// (Vulkan 1.3 core, or `VK_EXT_subgroup_size_control`) when the
    /// device exposes it.
    ///
    /// Subgroup width is the single most important specialization axis
    /// for a compute kernel: it is 32 on NVIDIA, 64 on AMD GCN/CDNA
    /// (32 or 64 on RDNA), and 8/16/32 on Intel. A kernel that reduces
    /// across a subgroup either reads the width at runtime
    /// (`gl_SubgroupSize` / `WaveGetLaneCount()`) or is compiled for a
    /// fixed width and pinned with
    /// [`ComputePipelineOptions::required_subgroup_size`](super::ComputePipelineOptions::required_subgroup_size).
    /// This method is what makes the second option usable: pinning a
    /// size is only legal within the
    /// [`size_control`](SubgroupProperties::size_control) range, and
    /// until now Vulkane let you set that field without offering any way
    /// to read the bounds it must satisfy.
    ///
    /// Returns `None` when the
    /// [`effective_api_version`](Self::effective_api_version) is below
    /// 1.1 (`VkPhysicalDeviceSubgroupProperties` is 1.1 core and has no
    /// extension form, so a 1.0 implementation leaves the struct
    /// untouched and reporting `subgroup_size: 0` would be a lie), or
    /// when `vkGetPhysicalDeviceProperties2` is unavailable.
    /// [`size_control`](SubgroupProperties::size_control) is `Some` only
    /// when the device actually advertises it.
    ///
    /// Note the gate is on the **effective** version, so this declines on
    /// a 1.0-created [`Instance`](super::Instance) even against a 1.3
    /// device. [`InstanceCreateInfo::api_version`](super::InstanceCreateInfo::api_version)
    /// defaults to 1.0, so raise it if this returns `None` unexpectedly.
    pub fn subgroup_properties(&self) -> Option<SubgroupProperties> {
        let get2 = self.instance.dispatch.vkGetPhysicalDeviceProperties2?;

        // VkPhysicalDeviceSubgroupProperties is Vulkan 1.1 core with no
        // extension form. A 1.0 implementation ignores the unrecognized
        // pNext struct and leaves it zeroed, which would read back as a
        // subgroup size of 0 — decline honestly instead.
        let api = self.effective_api_version();
        if api.major() < 1 || (api.major() == 1 && api.minor() < 1) {
            return None;
        }

        // Size control is 1.3 core, or the EXT before that. Chain it only
        // when one of those holds, so an untouched struct can't be
        // mistaken for a real min/max of 0.
        let has_size_control = (api.major() > 1 || api.minor() >= 3)
            || self
                .enumerate_extension_properties()
                .map(|exts| {
                    exts.iter()
                        .any(|e| e.name() == "VK_EXT_subgroup_size_control")
                })
                .unwrap_or(false);

        let mut chain = crate::safe::PNextChain::new();
        chain.push(VkPhysicalDeviceSubgroupProperties::new_pnext());
        if has_size_control {
            chain.push(VkPhysicalDeviceSubgroupSizeControlProperties::new_pnext());
        }
        let mut props2 = VkPhysicalDeviceProperties2 {
            sType: VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            pNext: chain.head_mut(),
            ..Default::default()
        };
        // Safety: handle valid; props2 + chain live for the call.
        unsafe { get2(self.handle, &mut props2) };

        let sg = chain.get::<VkPhysicalDeviceSubgroupProperties>()?;
        let size_control = if has_size_control {
            chain
                .get::<VkPhysicalDeviceSubgroupSizeControlProperties>()
                .map(|sc| SubgroupSizeControl {
                    min_subgroup_size: sc.minSubgroupSize,
                    max_subgroup_size: sc.maxSubgroupSize,
                    max_compute_workgroup_subgroups: sc.maxComputeWorkgroupSubgroups,
                    required_subgroup_size_stages: super::ShaderStageFlags(
                        sc.requiredSubgroupSizeStages,
                    ),
                })
        } else {
            None
        };

        Some(SubgroupProperties {
            subgroup_size: sg.subgroupSize,
            supported_stages: super::ShaderStageFlags(sg.supportedStages),
            supported_operations: SubgroupFeatureFlags(sg.supportedOperations),
            quad_operations_in_all_stages: sg.quadOperationsInAllStages != 0,
            size_control,
        })
    }

    /// Query the driver's identity — `VkPhysicalDeviceDriverProperties`
    /// (Vulkan 1.2 core, or `VK_KHR_driver_properties`).
    ///
    /// [`PhysicalDeviceProperties::driver_version`](PhysicalDeviceProperties::driver_version)
    /// is a bare `u32` whose bit-packing is **vendor-defined** — NVIDIA
    /// packs it (22,14,6,10), AMD (22,10,10,10), Intel-on-Windows
    /// (18,14) — so it cannot be decoded portably and is only good for
    /// equality comparison. This method returns what a caller actually
    /// wants instead: a [`driver_id`](DriverProperties::driver_id)
    /// enum naming the ICD (`MESA_RADV` vs `AMD_PROPRIETARY` vs
    /// `NVIDIA_PROPRIETARY` vs `MESA_LLVMPIPE` …), a human-readable
    /// driver name and version string, and the Vulkan CTS
    /// [`conformance_version`](DriverProperties::conformance_version)
    /// the driver claims to pass.
    ///
    /// Two things this is for: **stable cache keys** — a
    /// `(driver_id, driver_info)` pair is a portable, human-legible
    /// identity for "the compiler that produced this SPIR-V binary,"
    /// which a raw vendor-packed `u32` is not — and **quirk gating**,
    /// since ICDs for the same hardware genuinely differ (RADV and
    /// AMDVLK do not make the same codegen choices).
    ///
    /// Returns `None` when `vkGetPhysicalDeviceProperties2` is
    /// unavailable, or when the
    /// [`effective_api_version`](Self::effective_api_version) is below
    /// 1.2 and the device does not advertise `VK_KHR_driver_properties`
    /// — rather than reporting a zeroed struct as though the driver had
    /// answered. As with
    /// [`subgroup_properties`](Self::subgroup_properties), a 1.0-created
    /// [`Instance`](super::Instance) declines regardless of the device.
    pub fn driver_properties(&self) -> Option<DriverProperties> {
        let get2 = self.instance.dispatch.vkGetPhysicalDeviceProperties2?;

        let api = self.effective_api_version();
        let supported = (api.major() > 1 || api.minor() >= 2)
            || self
                .enumerate_extension_properties()
                .map(|exts| exts.iter().any(|e| e.name() == "VK_KHR_driver_properties"))
                .unwrap_or(false);
        if !supported {
            return None;
        }

        let mut chain = crate::safe::PNextChain::new();
        chain.push(VkPhysicalDeviceDriverProperties::new_pnext());
        let mut props2 = VkPhysicalDeviceProperties2 {
            sType: VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            pNext: chain.head_mut(),
            ..Default::default()
        };
        // Safety: handle valid; props2 + chain live for the call.
        unsafe { get2(self.handle, &mut props2) };

        let d = chain.get::<VkPhysicalDeviceDriverProperties>()?;
        // Safety: both are null-terminated char arrays per the spec.
        let driver_name = unsafe { CStr::from_ptr(d.driverName.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let driver_info = unsafe { CStr::from_ptr(d.driverInfo.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        // Advertising the extension is necessary but NOT sufficient: an
        // implementation running as 1.0 may advertise
        // `VK_KHR_driver_properties` as a *device* extension and still
        // leave the struct untouched, because the instance was created
        // below the version that processes it. (Observed on the AMD
        // proprietary driver: advertised at instance 1.0, populated only
        // at instance 1.2+.) A conforming driver that genuinely answers
        // always names itself, so an empty name means "not answered" —
        // decline rather than hand back a hollow struct whose `driver_id`
        // is really just the zero-initializer.
        if driver_name.is_empty() {
            return None;
        }

        Some(DriverProperties {
            driver_id: VkDriverId::from_raw(d.driverID),
            driver_id_raw: d.driverID,
            driver_name,
            driver_info,
            conformance_version: ConformanceVersion {
                major: d.conformanceVersion.major,
                minor: d.conformanceVersion.minor,
                subminor: d.conformanceVersion.subminor,
                patch: d.conformanceVersion.patch,
            },
        })
    }

    /// Query the shader arithmetic capabilities that gate reduced-precision
    /// kernels — `shaderFloat16` / `shaderInt8`
    /// (`VkPhysicalDeviceShaderFloat16Int8Features`, Vulkan 1.2 core) and the
    /// 16-/8-bit storage-buffer access features (Vulkan 1.1 and 1.2 core
    /// respectively).
    ///
    /// These decide whether a half-precision or quantized kernel can exist on
    /// a device at all, which makes them a specialization axis rather than a
    /// tuning knob: a kernel built against `shaderFloat16` is not merely
    /// slower without it, it is invalid. Compute precision is a distinct
    /// question from *storage* precision — a device can accept 16-bit data in
    /// a storage buffer while doing the arithmetic in f32 — so the two are
    /// reported separately rather than collapsed.
    ///
    /// Returns `None` when `vkGetPhysicalDeviceFeatures2` is unavailable, or
    /// when the [`effective_api_version`](Self::effective_api_version) is
    /// below 1.2 and the device advertises neither
    /// `VK_KHR_shader_float16_int8` nor `VK_KHR_8bit_storage` — rather than
    /// reporting an untouched all-`false` struct as though the driver had
    /// answered. As elsewhere, a 1.0-created [`Instance`](super::Instance)
    /// declines regardless of the device.
    pub fn shader_arithmetic_features(&self) -> Option<ShaderArithmeticFeatures> {
        let get2 = self.instance.dispatch.vkGetPhysicalDeviceFeatures2?;

        let api = self.effective_api_version();
        let core_1_2 = api.major() > 1 || api.minor() >= 2;
        // `unwrap_or_default()` here is deliberate and the reason is narrow: a
        // failed enumeration yields an empty list, `has(..)` is false for
        // everything, and the guard below returns `None` — a decline, which is
        // the honest answer when we could not establish whether the extension
        // is present. The OUTCOME is right; note that the REASONING is
        // accidental. Nothing distinguishes "queried, none present" from "the
        // query failed", so anything added below that proceeds on the basis of
        // `exts` would silently treat a failure as "no extensions".
        let exts = self.enumerate_extension_properties().unwrap_or_default();
        let has = |name: &str| exts.iter().any(|e| e.name() == name);
        if !core_1_2 && !has("VK_KHR_shader_float16_int8") && !has("VK_KHR_8bit_storage") {
            return None;
        }

        let mut chain = crate::safe::PNextChain::new();
        chain.push(VkPhysicalDeviceShaderFloat16Int8Features::new_pnext());
        chain.push(VkPhysicalDevice16BitStorageFeatures::new_pnext());
        chain.push(VkPhysicalDevice8BitStorageFeatures::new_pnext());
        let mut features2 = VkPhysicalDeviceFeatures2 {
            sType: VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            pNext: chain.head_mut(),
            ..Default::default()
        };
        // Safety: handle valid; features2 + chain live for the call.
        unsafe { get2(self.handle, &mut features2) };

        let f16i8 = chain.get::<VkPhysicalDeviceShaderFloat16Int8Features>()?;
        let s16 = chain.get::<VkPhysicalDevice16BitStorageFeatures>();
        let s8 = chain.get::<VkPhysicalDevice8BitStorageFeatures>();

        Some(ShaderArithmeticFeatures {
            shader_float16: f16i8.shaderFloat16 != 0,
            shader_int8: f16i8.shaderInt8 != 0,
            storage_buffer_16bit: s16.is_some_and(|s| s.storageBuffer16BitAccess != 0),
            storage_buffer_8bit: s8.is_some_and(|s| s.storageBuffer8BitAccess != 0),
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
    ///
    /// ⚠️ **An all-zero reading here is NOT the safe default it is for a
    /// capability struct, and the difference is the field's KIND.**
    ///
    /// [`Self::shader_integer_dot_product_properties`] returns booleans, so
    /// all-false reads as *"this device accelerates none of these"* — true, and
    /// useful. **These are LIMITS.** All-zero reads as
    /// `shader_group_handle_size: 0` and `shader_group_base_alignment: 0`,
    /// which is not "no ray tracing" — it is a shader binding table with
    /// zero-sized handles and zero alignment. A caller who lays out an SBT from
    /// it gets a zero-stride table rather than an error.
    ///
    /// **So check the extension is present before believing these numbers.**
    /// `enumerate_extension_properties` is the check; this method deliberately
    /// does not perform it, because doing so would make an absent extension and
    /// a driver that reports zeros indistinguishable — and only the caller
    /// knows which of those matters to them.
    ///
    /// Observed rather than reasoned, on one machine with two GPUs
    /// (2026-09-03): `VK_EXT_shader_long_vector` is present on the RTX 4070 and
    /// absent on the integrated Radeon 610M. Its limit,
    /// `maxVectorComponents`, reads 1024 on the first and — through an ungated
    /// `pNext` query — zero on the second. **Zero components is a plausible
    /// number and a false one.** The same shape applies to every limit-valued
    /// property in this file.
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

    /// The kind of physical device (discrete GPU, integrated, virtual, CPU, ...),
    /// or `None` if the implementation reported a kind this spec revision does
    /// not define. See [`device_type_raw`](Self::device_type_raw).
    pub fn device_type(&self) -> Option<PhysicalDeviceType> {
        VkPhysicalDeviceType::from_raw(self.raw.deviceType).map(PhysicalDeviceType)
    }

    /// Raw `VkPhysicalDeviceType` value, exactly as the implementation
    /// reported it — including values this build cannot name.
    pub fn device_type_raw(&self) -> i32 {
        self.raw.deviceType
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

/// Which subgroup operation classes a device supports, from
/// `VkPhysicalDeviceSubgroupProperties::supportedOperations`.
///
/// `BASIC` is the floor guaranteed by Vulkan 1.1; everything above it is
/// optional. A reduction kernel wants at least `ARITHMETIC`; a scan or a
/// prefix-sum additionally wants `SHUFFLE_RELATIVE` or `CLUSTERED`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubgroupFeatureFlags(pub u32);

impl SubgroupFeatureFlags {
    /// `subgroupBarrier`, `subgroupElect`, `subgroupMemoryBarrier` —
    /// mandatory on every Vulkan 1.1 implementation.
    pub const BASIC: Self = Self(SUBGROUP_FEATURE_BASIC_BIT);
    /// `subgroupAll` / `subgroupAny` / `subgroupAllEqual`.
    pub const VOTE: Self = Self(SUBGROUP_FEATURE_VOTE_BIT);
    /// `subgroupAdd` / `subgroupMul` / `subgroupMin` / `subgroupMax` and
    /// their inclusive/exclusive scans — the class a cross-lane reduction
    /// (`WaveActiveSum`) needs.
    pub const ARITHMETIC: Self = Self(SUBGROUP_FEATURE_ARITHMETIC_BIT);
    /// `subgroupBallot` and friends.
    pub const BALLOT: Self = Self(SUBGROUP_FEATURE_BALLOT_BIT);
    /// `subgroupShuffle` / `subgroupShuffleXor`.
    pub const SHUFFLE: Self = Self(SUBGROUP_FEATURE_SHUFFLE_BIT);
    /// `subgroupShuffleUp` / `subgroupShuffleDown`.
    pub const SHUFFLE_RELATIVE: Self = Self(SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT);
    /// Clustered reductions over power-of-two lane groups.
    pub const CLUSTERED: Self = Self(SUBGROUP_FEATURE_CLUSTERED_BIT);
    /// Quad (2×2) shuffles and broadcasts.
    pub const QUAD: Self = Self(SUBGROUP_FEATURE_QUAD_BIT);
    /// `VK_KHR_shader_subgroup_rotate` (Vulkan 1.4 core).
    pub const ROTATE: Self = Self(SUBGROUP_FEATURE_ROTATE_BIT);
    /// Clustered rotate, from the same extension.
    pub const ROTATE_CLUSTERED: Self = Self(SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT);
    /// `VK_NV_shader_subgroup_partitioned`.
    pub const PARTITIONED_NV: Self = Self(SUBGROUP_FEATURE_PARTITIONED_BIT_NV);

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for SubgroupFeatureFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// A device's subgroup ("wave" on AMD/HLSL, "warp" on NVIDIA) properties.
///
/// Returned by [`PhysicalDevice::subgroup_properties`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgroupProperties {
    /// Lanes per subgroup — 32 on NVIDIA, 64 on AMD GCN/CDNA, 32 or 64
    /// on RDNA, 8/16/32 on Intel. This is the default width a shader
    /// sees; where [`size_control`](Self::size_control) is present the
    /// pipeline may pin a different one within its range.
    pub subgroup_size: u32,
    /// Which shader stages may use subgroup operations at all. Compute is
    /// guaranteed; the rest are optional.
    pub supported_stages: super::ShaderStageFlags,
    /// Which classes of subgroup operation the device implements.
    pub supported_operations: SubgroupFeatureFlags,
    /// Whether quad operations work in every stage in
    /// [`supported_stages`](Self::supported_stages), not just fragment
    /// and compute.
    pub quad_operations_in_all_stages: bool,
    /// The pinnable subgroup-size range, when the device exposes
    /// `VK_EXT_subgroup_size_control` (Vulkan 1.3 core). `None` means
    /// the size is fixed at [`subgroup_size`](Self::subgroup_size) and
    /// [`ComputePipelineOptions::required_subgroup_size`](super::ComputePipelineOptions::required_subgroup_size)
    /// must be left unset.
    pub size_control: Option<SubgroupSizeControl>,
}

/// The subgroup-size range a pipeline may pin, from
/// `VkPhysicalDeviceSubgroupSizeControlProperties`.
///
/// This is the range
/// [`ComputePipelineOptions::required_subgroup_size`](super::ComputePipelineOptions::required_subgroup_size)
/// must fall within — use [`permits`](Self::permits) to check a
/// candidate before building the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubgroupSizeControl {
    /// Smallest pinnable subgroup size. A power of two, per the spec.
    pub min_subgroup_size: u32,
    /// Largest pinnable subgroup size. A power of two, per the spec.
    pub max_subgroup_size: u32,
    /// Upper bound on subgroups per compute workgroup.
    pub max_compute_workgroup_subgroups: u32,
    /// Which stages accept a pinned size. Check for
    /// [`ShaderStageFlags::COMPUTE`](super::ShaderStageFlags::COMPUTE)
    /// before pinning one on a compute pipeline.
    pub required_subgroup_size_stages: super::ShaderStageFlags,
}

impl SubgroupSizeControl {
    /// Whether `size` is a legal
    /// [`required_subgroup_size`](super::ComputePipelineOptions::required_subgroup_size)
    /// on this device: a power of two within
    /// `[min_subgroup_size, max_subgroup_size]`.
    ///
    /// This checks the size range only. Also confirm the stage accepts a
    /// pinned size — see
    /// [`permits_in_compute`](Self::permits_in_compute), which checks
    /// both.
    pub const fn permits(&self, size: u32) -> bool {
        size.is_power_of_two() && size >= self.min_subgroup_size && size <= self.max_subgroup_size
    }

    /// Whether `size` may be pinned on a **compute** pipeline: both
    /// [`permits`](Self::permits) and the compute stage appearing in
    /// [`required_subgroup_size_stages`](Self::required_subgroup_size_stages).
    pub const fn permits_in_compute(&self, size: u32) -> bool {
        self.permits(size)
            && self
                .required_subgroup_size_stages
                .contains(super::ShaderStageFlags::COMPUTE)
    }
}

/// Shader arithmetic capabilities that gate reduced-precision kernels.
///
/// Returned by [`PhysicalDevice::shader_arithmetic_features`]. Compute
/// precision and storage precision are separate questions and are reported
/// separately: a device may accept 16-bit data in a storage buffer while
/// performing the arithmetic itself in f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ShaderArithmeticFeatures {
    /// `shaderFloat16` — half-precision *arithmetic* in shaders.
    pub shader_float16: bool,
    /// `shaderInt8` — 8-bit integer *arithmetic* in shaders.
    pub shader_int8: bool,
    /// `storageBuffer16BitAccess` — 16-bit types readable/writable in storage
    /// buffers, independent of whether arithmetic on them is supported.
    pub storage_buffer_16bit: bool,
    /// `storageBuffer8BitAccess` — likewise for 8-bit types.
    pub storage_buffer_8bit: bool,
}

/// Driver identity — which ICD is behind this physical device, and what
/// Vulkan conformance level it claims.
///
/// Returned by [`PhysicalDevice::driver_properties`]. Prefer this over
/// [`PhysicalDeviceProperties::driver_version`](PhysicalDeviceProperties::driver_version),
/// whose bit-packing is vendor-defined and therefore not portably
/// decodable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverProperties {
    /// Which ICD this is — `DRIVER_ID_MESA_RADV`,
    /// `DRIVER_ID_AMD_PROPRIETARY`, `DRIVER_ID_NVIDIA_PROPRIETARY`,
    /// `DRIVER_ID_MESA_LLVMPIPE`, … Two ICDs driving the *same* hardware
    /// are distinct here, which is what makes this the right axis for
    /// gating a driver-specific workaround.
    ///
    /// `None` means the ICD reported an ID this spec revision does not
    /// define. That is routine rather than exceptional — new driver IDs are
    /// registered regularly, so any driver newer than the pinned `vk.xml` can
    /// land here. Gate workarounds on the named value; key caches on
    /// [`driver_id_raw`](Self::driver_id_raw), which stays distinct even for
    /// an ICD this build cannot name.
    pub driver_id: Option<VkDriverId>,
    /// Raw `VkDriverId` value, exactly as the ICD reported it.
    ///
    /// Always meaningful, including when [`driver_id`](Self::driver_id) is
    /// `None`. Two unrecognized ICDs remain distinguishable here, which is why
    /// this — not the named form — is the right shader-cache key.
    pub driver_id_raw: i32,
    /// Vendor's name for the driver, e.g. `"radv"`, `"NVIDIA"`.
    pub driver_name: String,
    /// Free-form version detail, e.g. `"Mesa 24.1.2"`. Together with
    /// [`driver_id`](Self::driver_id) this forms a stable, legible
    /// shader-cache key.
    pub driver_info: String,
    /// The Vulkan CTS version the driver claims conformance to. All-zero
    /// on a driver that makes no claim.
    pub conformance_version: ConformanceVersion,
}

/// A Vulkan CTS conformance version, `major.minor.subminor.patch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct ConformanceVersion {
    pub major: u8,
    pub minor: u8,
    pub subminor: u8,
    pub patch: u8,
}

impl std::fmt::Display for ConformanceVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{}.{}.{}",
            self.major, self.minor, self.subminor, self.patch
        )
    }
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

/// One supported cooperative-*vector* combination, as returned by
/// [`PhysicalDevice::cooperative_vector_properties`].
///
/// Cooperative vector is a different capability from cooperative matrix, not a
/// variation of it: there are no `M`/`N`/`K` dimensions, and the five component
/// types describe an input, three *interpretations*, and a result. The
/// interpretation fields are where the packed types
/// (`VK_COMPONENT_TYPE_SINT8_PACKED_NV`, `..._UINT8_PACKED_NV`) actually appear
/// — they are defined by `VK_NV_cooperative_vector` and are not gated on
/// `VK_KHR_cooperative_matrix`, which is why the cooperative-matrix query
/// cannot observe them.
///
/// Every component accessor returns `Option`, for the same reason the
/// cooperative-matrix ones do: the driver writes these fields and may report a
/// value this build's `vk.xml` does not define. Reading such a value as a Rust
/// enum would be undefined behaviour, so the generated struct types them as
/// `i32` and conversion happens here where `None` is representable.
#[derive(Clone)]
pub struct CooperativeVectorProperties {
    raw: VkCooperativeVectorPropertiesNV,
}

impl CooperativeVectorProperties {
    /// Component type of the input vector, or `None` if the implementation
    /// reported a value this spec revision does not define.
    pub fn input_type(&self) -> Option<VkComponentTypeKHR> {
        VkComponentTypeKHR::from_raw(self.raw.inputType)
    }
    /// Raw `VkComponentTypeKHR` for the input, exactly as reported.
    pub fn input_type_raw(&self) -> i32 {
        self.raw.inputType
    }
    /// How the input is interpreted — one of the positions a *packed* component
    /// type can legitimately appear in.
    pub fn input_interpretation(&self) -> Option<VkComponentTypeKHR> {
        VkComponentTypeKHR::from_raw(self.raw.inputInterpretation)
    }
    /// See [`input_type_raw`](Self::input_type_raw).
    pub fn input_interpretation_raw(&self) -> i32 {
        self.raw.inputInterpretation
    }
    /// How the matrix operand is interpreted.
    pub fn matrix_interpretation(&self) -> Option<VkComponentTypeKHR> {
        VkComponentTypeKHR::from_raw(self.raw.matrixInterpretation)
    }
    /// See [`input_type_raw`](Self::input_type_raw).
    pub fn matrix_interpretation_raw(&self) -> i32 {
        self.raw.matrixInterpretation
    }
    /// How the bias operand is interpreted.
    pub fn bias_interpretation(&self) -> Option<VkComponentTypeKHR> {
        VkComponentTypeKHR::from_raw(self.raw.biasInterpretation)
    }
    /// See [`input_type_raw`](Self::input_type_raw).
    pub fn bias_interpretation_raw(&self) -> i32 {
        self.raw.biasInterpretation
    }
    /// Component type of the result.
    pub fn result_type(&self) -> Option<VkComponentTypeKHR> {
        VkComponentTypeKHR::from_raw(self.raw.resultType)
    }
    /// See [`input_type_raw`](Self::input_type_raw).
    pub fn result_type_raw(&self) -> i32 {
        self.raw.resultType
    }
    /// Whether this combination transposes the matrix operand.
    pub fn transpose(&self) -> bool {
        self.raw.transpose != 0
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
    /// Component type of operand A, or `None` if the implementation reported a
    /// value this spec revision does not define.
    ///
    /// `None` is not an error — it means the driver supports a component type
    /// newer than the `vk.xml` this build was generated from. Use
    /// [`a_type_raw`](Self::a_type_raw) when you need to preserve or report
    /// that value rather than discard it.
    pub fn a_type(&self) -> Option<VkComponentTypeKHR> {
        VkComponentTypeKHR::from_raw(self.raw.AType)
    }
    /// Raw `VkComponentTypeKHR` value for operand A, exactly as the
    /// implementation reported it — including values this build cannot name.
    pub fn a_type_raw(&self) -> i32 {
        self.raw.AType
    }
    /// See [`a_type`](Self::a_type).
    pub fn b_type(&self) -> Option<VkComponentTypeKHR> {
        VkComponentTypeKHR::from_raw(self.raw.BType)
    }
    /// See [`a_type_raw`](Self::a_type_raw).
    pub fn b_type_raw(&self) -> i32 {
        self.raw.BType
    }
    /// See [`a_type`](Self::a_type).
    pub fn c_type(&self) -> Option<VkComponentTypeKHR> {
        VkComponentTypeKHR::from_raw(self.raw.CType)
    }
    /// See [`a_type_raw`](Self::a_type_raw).
    pub fn c_type_raw(&self) -> i32 {
        self.raw.CType
    }
    /// See [`a_type`](Self::a_type).
    pub fn result_type(&self) -> Option<VkComponentTypeKHR> {
        VkComponentTypeKHR::from_raw(self.raw.ResultType)
    }
    /// See [`a_type_raw`](Self::a_type_raw).
    pub fn result_type_raw(&self) -> i32 {
        self.raw.ResultType
    }
    /// Whether the implementation saturates accumulator overflow.
    pub fn saturating_accumulation(&self) -> bool {
        self.raw.saturatingAccumulation != 0
    }
    /// Scope the cooperative matrix operates at, or `None` if the
    /// implementation reported a scope this spec revision does not define.
    pub fn scope(&self) -> Option<VkScopeKHR> {
        VkScopeKHR::from_raw(self.raw.scope)
    }
    /// Raw `VkScopeKHR` value, exactly as the implementation reported it.
    pub fn scope_raw(&self) -> i32 {
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

#[cfg(test)]
mod driver_written_enum_tests {
    use super::*;

    /// A driver reporting a component type newer than the pinned `vk.xml` — the
    /// exact case that used to be undefined behaviour.
    ///
    /// **This test could not have been written before the field became `i32`.**
    /// `VkCooperativeMatrixPropertiesKHR::AType` was typed as the Rust enum
    /// `VkComponentTypeKHR`, so placing an undeclared discriminant in it was UB
    /// at the moment of construction — the test would have been demonstrating
    /// the bug by committing it. Now the field is a plain integer, the
    /// out-of-range value is representable, and the *decision* about what it
    /// means moves to a checked conversion the caller must handle.
    ///
    /// `999_999` stands in for a value from a future extension. Nothing about
    /// the number matters except that this build cannot name it.
    #[test]
    fn an_undefined_component_type_is_representable_and_reported_honestly() {
        let props = CooperativeMatrixProperties {
            raw: VkCooperativeMatrixPropertiesKHR {
                AType: 999_999,
                ..Default::default()
            },
        };

        assert_eq!(
            props.a_type(),
            None,
            "a value this build cannot name must convert to None, not to some \
             arbitrary variant that happens to share a discriminant"
        );
        assert_eq!(
            props.a_type_raw(),
            999_999,
            "the raw accessor must preserve exactly what the driver reported — \
             discarding it would make two different unknown types indistinguishable"
        );
    }

    /// The checked path must still resolve values this build *does* know, or
    /// the test above would pass for the trivial reason that conversion always
    /// fails.
    #[test]
    fn a_defined_component_type_still_converts() {
        let props = CooperativeMatrixProperties {
            raw: VkCooperativeMatrixPropertiesKHR {
                AType: VkComponentTypeKHR::COMPONENT_TYPE_FLOAT32_KHR as i32,
                ..Default::default()
            },
        };

        assert_eq!(
            props.a_type(),
            Some(VkComponentTypeKHR::COMPONENT_TYPE_FLOAT32_KHR)
        );
        assert_eq!(
            props.a_type_raw(),
            VkComponentTypeKHR::COMPONENT_TYPE_FLOAT32_KHR as i32
        );
    }

    /// Same hazard, different field: driver IDs are registered continuously, so
    /// an ICD newer than the pinned spec is the *expected* case rather than an
    /// exotic one.
    ///
    /// This pins **why `DriverProperties` carries both forms.** The checked
    /// conversion is lossy by construction — every unrecognized ICD becomes the
    /// same `None`, so two different unknown drivers are indistinguishable
    /// through it. A shader cache keyed on that would happily reuse one
    /// driver's binaries for another. The raw value keeps them apart, which is
    /// what makes it, not the named form, the correct cache key.
    #[test]
    fn unknown_driver_ids_collapse_when_named_but_stay_distinct_raw() {
        assert_eq!(VkDriverId::from_raw(999_999), None);
        assert_eq!(VkDriverId::from_raw(999_998), None);

        // Lossy on purpose: the named form cannot tell these apart...
        assert_eq!(VkDriverId::from_raw(999_999), VkDriverId::from_raw(999_998));
        // ...which is precisely why the raw value is retained alongside it.
        assert_ne!(999_999, 999_998);
    }
}
