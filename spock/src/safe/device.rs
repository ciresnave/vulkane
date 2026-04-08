//! Safe wrapper for `VkDevice` and `VkQueue`.

use super::features::DeviceFeatures;
use super::physical::PhysicalDevice;
use super::sync::Semaphore;
use super::{Error, Result, check};
use crate::raw::VulkanLibrary;
use crate::raw::bindings::*;
use std::ffi::{CString, c_char, c_void};
use std::sync::Arc;

/// One semaphore wait in [`Queue::submit_with_sync`].
///
/// `value` is ignored for binary semaphores; for timeline semaphores it
/// is the counter value the wait must reach before the submission may
/// proceed. `dst_stage_mask` is the `VK_PIPELINE_STAGE_*` mask
/// of stages that need to wait.
#[derive(Clone, Copy)]
pub struct WaitSemaphore<'a> {
    pub semaphore: &'a Semaphore,
    pub value: u64,
    pub dst_stage_mask: u32,
}

/// One semaphore signal in [`Queue::submit_with_sync`].
///
/// `value` is ignored for binary semaphores; for timeline semaphores it
/// is the counter value the submission writes when it completes.
#[derive(Clone, Copy)]
pub struct SignalSemaphore<'a> {
    pub semaphore: &'a Semaphore,
    pub value: u64,
}

/// Parameters for creating a single device queue.
#[derive(Debug, Clone)]
pub struct QueueCreateInfo {
    /// The index of the queue family to create the queue from.
    pub queue_family_index: u32,
    /// Priority for each queue, in the range `[0.0, 1.0]`.
    /// The number of queues to create equals `queue_priorities.len()`.
    pub queue_priorities: Vec<f32>,
}

/// Parameters for [`PhysicalDevice::create_device`].
#[derive(Debug, Clone)]
pub struct DeviceCreateInfo<'a> {
    /// One or more queues to create from queue families.
    pub queue_create_infos: &'a [QueueCreateInfo],
    /// Names of device extensions to enable. Each must be one that
    /// [`PhysicalDevice::enumerate_extension_properties`] reports as available.
    pub enabled_extensions: &'a [&'a str],
    /// Optional Vulkan feature bits to enable. When `Some`, the safe
    /// wrapper builds a `VkPhysicalDeviceFeatures2` chain (with the 1.1
    /// / 1.2 / 1.3 sub-structs as needed) and passes it via `pNext` to
    /// `vkCreateDevice`.
    ///
    /// When `None`, no features are enabled — equivalent to passing
    /// `DeviceFeatures::default()`.
    pub enabled_features: Option<&'a DeviceFeatures>,
}

// `&[T]` does implement `Default` (returns an empty slice), so technically
// this could be derived — but clippy's suggestion fights with the lifetime
// parameter on `DeviceCreateInfo<'a>`. The manual impl is clearer.
#[allow(clippy::derivable_impls)]
impl<'a> Default for DeviceCreateInfo<'a> {
    fn default() -> Self {
        Self {
            queue_create_infos: &[],
            enabled_extensions: &[],
            enabled_features: None,
        }
    }
}

/// Internal state shared between [`Device`] and its child handles.
pub(crate) struct DeviceInner {
    pub(crate) handle: VkDevice,
    pub(crate) dispatch: VkDeviceDispatchTable,
    /// Keep the parent instance alive as long as any device child is alive.
    /// Field is unread but its `Drop` semantics are essential.
    #[allow(dead_code)]
    pub(crate) instance: Arc<super::instance::InstanceInner>,
    /// All physical devices in the device group this `Device` was
    /// created from. Always at least one element. Length 1 for the
    /// overwhelmingly common case of `physical.create_device(...)`.
    pub(crate) physical_devices: Vec<VkPhysicalDevice>,
}

// Safety: VkDevice is documented by the Vulkan spec as safe to share between
// threads. Individual function calls have their own external synchronization
// requirements (queue submission, command buffer recording, etc. — these are
// the caller's responsibility), but the handle itself is thread-safe.
unsafe impl Send for DeviceInner {}
unsafe impl Sync for DeviceInner {}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        if let Some(destroy) = self.dispatch.vkDestroyDevice {
            // Safety: handle is valid (constructed by Device::new), and
            // by the Arc invariant we are the last owner.
            unsafe { destroy(self.handle, std::ptr::null()) };
        }
    }
}

/// A safe wrapper around `VkDevice`.
///
/// The device is destroyed automatically when the last `Device` clone (and
/// the last child handle that holds an `Arc<DeviceInner>`) is dropped.
#[derive(Clone)]
pub struct Device {
    pub(crate) inner: Arc<DeviceInner>,
}

impl Device {
    /// Create a logical [`Device`] from a [`PhysicalDeviceGroup`].
    /// Internally calls `vkCreateDevice` with a
    /// `VkDeviceGroupDeviceCreateInfo` chain listing every physical
    /// device in the group. The resulting `Device` exposes
    /// [`physical_device_count`](Self::physical_device_count) > 1 and
    /// every operation that takes a `device_mask` defaults to "all
    /// devices in the group".
    pub(crate) fn new_group(
        group: &super::physical::PhysicalDeviceGroup,
        info: DeviceCreateInfo<'_>,
    ) -> Result<Self> {
        // Use the first physical device as the entry point for
        // vkCreateDevice (the spec says any device in the group works).
        let physical = group
            .physical_devices
            .first()
            .ok_or(Error::Vk(VkResult::ERROR_INITIALIZATION_FAILED))?;
        let raw_handles: Vec<VkPhysicalDevice> =
            group.physical_devices.iter().map(|p| p.handle).collect();

        Self::new_inner(physical, info, raw_handles)
    }

    pub(crate) fn new(physical: &PhysicalDevice, info: DeviceCreateInfo<'_>) -> Result<Self> {
        // Single-physical-device case: store a length-1 group.
        Self::new_inner(physical, info, vec![physical.handle])
    }

    fn new_inner(
        physical: &PhysicalDevice,
        info: DeviceCreateInfo<'_>,
        raw_physical_handles: Vec<VkPhysicalDevice>,
    ) -> Result<Self> {
        let create = physical
            .instance
            .dispatch
            .vkCreateDevice
            .ok_or(Error::MissingFunction("vkCreateDevice"))?;

        // Build the raw VkDeviceQueueCreateInfo array. We need to keep the
        // priority slices alive across the call.
        let mut raw_infos: Vec<VkDeviceQueueCreateInfo> =
            Vec::with_capacity(info.queue_create_infos.len());
        for qci in info.queue_create_infos {
            raw_infos.push(VkDeviceQueueCreateInfo {
                sType: VkStructureType::STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                pNext: std::ptr::null(),
                flags: 0,
                queueFamilyIndex: qci.queue_family_index,
                queueCount: qci.queue_priorities.len() as u32,
                pQueuePriorities: qci.queue_priorities.as_ptr(),
            });
        }

        // Owned CString vectors for extension names — kept alive across the
        // create call.
        let ext_cstrings: Vec<CString> = info
            .enabled_extensions
            .iter()
            .map(|s| CString::new(*s))
            .collect::<std::result::Result<_, _>>()?;
        let ext_ptrs: Vec<*mut c_char> = ext_cstrings
            .iter()
            .map(|s| s.as_ptr() as *mut c_char)
            .collect();

        // Build the optional VkPhysicalDeviceFeatures2 chain. Each
        // sub-struct's pNext points at the next one. Both `chain_owned`
        // and `features2_owned` must outlive the create call, so we
        // bind them with explicit `let`.
        let chain_owned: Option<(
            VkPhysicalDeviceFeatures2,
            VkPhysicalDeviceVulkan11Features,
            VkPhysicalDeviceVulkan12Features,
            VkPhysicalDeviceVulkan13Features,
        )>;
        let p_next: *const c_void;
        let p_enabled_features: *const VkPhysicalDeviceFeatures;

        if let Some(f) = info.enabled_features {
            // Lay out the chain in reverse (innermost first) so each
            // pNext can point at a stable address.
            //
            //   features2 -> v11 -> v12 -> v13 -> null
            //
            // The pNext is a `*mut c_void` per Vulkan, so we have to
            // const-cast our way out of immutable references.
            let mut v13 = f.features13;
            v13.pNext = std::ptr::null_mut();

            let mut v12 = f.features12;
            v12.pNext = (&v13 as *const _ as *mut c_void).cast();
            // We need v12.pNext to point at v13. But v13 is a local — it
            // moves into the tuple below; the address is only stable
            // *after* the tuple is on the stack. So we patch pNext after
            // construction. (See chain_owned binding below.)

            let mut v11 = f.features11;
            v11.pNext = std::ptr::null_mut();

            let mut features2 = VkPhysicalDeviceFeatures2 {
                sType: VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                pNext: std::ptr::null_mut(),
                features: f.features10,
            };
            features2.pNext = std::ptr::null_mut();

            chain_owned = Some((features2, v11, v12, v13));

            // Now patch pNext pointers to refer to the stable tuple.
            // Safety: borrow_chain has Some so unwrap is fine. The tuple
            // is bound for the rest of this function so the references
            // are valid until vkCreateDevice returns.
            let chain_ref = chain_owned.as_ref().unwrap();
            // chain_ref is (&features2, &v11, &v12, &v13)
            // We need to mutate the pNext fields, so cast through.
            unsafe {
                let f2_ptr = &chain_ref.0 as *const _ as *mut VkPhysicalDeviceFeatures2;
                let v11_ptr = &chain_ref.1 as *const _ as *mut VkPhysicalDeviceVulkan11Features;
                let v12_ptr = &chain_ref.2 as *const _ as *mut VkPhysicalDeviceVulkan12Features;
                let v13_ptr = &chain_ref.3 as *const _ as *mut VkPhysicalDeviceVulkan13Features;
                (*f2_ptr).pNext = v11_ptr.cast();
                (*v11_ptr).pNext = v12_ptr.cast();
                (*v12_ptr).pNext = v13_ptr.cast();
                (*v13_ptr).pNext = std::ptr::null_mut();
            }
            p_next = &chain_ref.0 as *const _ as *const c_void;
            // When using features2 in pNext, pEnabledFeatures must be null.
            p_enabled_features = std::ptr::null();
        } else {
            chain_owned = None;
            p_next = std::ptr::null();
            p_enabled_features = std::ptr::null();
        }

        // If this is a multi-device group, prepend a
        // VkDeviceGroupDeviceCreateInfo to the pNext chain.
        let group_info_owned: Option<VkDeviceGroupDeviceCreateInfo>;
        let final_p_next: *const c_void = if raw_physical_handles.len() > 1 {
            let mut g = VkDeviceGroupDeviceCreateInfo {
                sType: VkStructureType::STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO,
                pNext: p_next, // chain in the existing pNext (features, etc.)
                physicalDeviceCount: raw_physical_handles.len() as u32,
                pPhysicalDevices: raw_physical_handles.as_ptr(),
            };
            // Bind to a stable address.
            group_info_owned = Some(g);
            // Re-borrow once it's settled.
            let g_ref = group_info_owned.as_ref().unwrap();
            // Safety: g_ref lives until end of scope.
            g = *g_ref;
            let _ = g;
            group_info_owned.as_ref().unwrap() as *const _ as *const c_void
        } else {
            group_info_owned = None;
            p_next
        };

        let create_info = VkDeviceCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext: final_p_next,
            queueCreateInfoCount: raw_infos.len() as u32,
            pQueueCreateInfos: raw_infos.as_ptr(),
            enabledExtensionCount: ext_ptrs.len() as u32,
            ppEnabledExtensionNames: if ext_ptrs.is_empty() {
                std::ptr::null()
            } else {
                ext_ptrs.as_ptr()
            },
            pEnabledFeatures: p_enabled_features,
            ..Default::default()
        };

        let mut handle: VkDevice = std::ptr::null_mut();
        // Safety: create_info is valid for the call. raw_infos, name
        // CStrings, the chain_owned tuple, raw_physical_handles, and
        // group_info_owned all live until end of scope.
        check(unsafe { create(physical.handle, &create_info, std::ptr::null(), &mut handle) })?;
        // Suppress dead-code warning for the chain bindings — their
        // *purpose* is to live until after the call.
        let _ = chain_owned;
        let _ = group_info_owned;

        // Load device-level dispatch table.
        // Safety: handle and instance are both valid.
        let library: &VulkanLibrary = &physical.instance.library;
        let dispatch = unsafe { library.load_device(physical.instance.handle, handle) };

        Ok(Self {
            inner: Arc::new(DeviceInner {
                handle,
                dispatch,
                instance: Arc::clone(&physical.instance),
                physical_devices: raw_physical_handles,
            }),
        })
    }

    /// Number of physical devices in this device's group. Always at
    /// least 1; usually exactly 1 on consumer hardware.
    pub fn physical_device_count(&self) -> u32 {
        self.inner.physical_devices.len() as u32
    }

    /// Returns the raw physical device handles in this device's group.
    /// Length matches [`physical_device_count`](Self::physical_device_count).
    pub fn physical_device_handles(&self) -> &[VkPhysicalDevice] {
        &self.inner.physical_devices
    }

    /// Returns the default device mask: a bitmask with one bit set per
    /// physical device in the group. For a single-device group this is
    /// `0b1`; for a 2-device group it's `0b11`; etc. Used as the
    /// "submit / allocate to all devices in the group" sentinel.
    pub fn default_device_mask(&self) -> u32 {
        let n = self.inner.physical_devices.len() as u32;
        if n >= 32 {
            0xFFFF_FFFF
        } else {
            (1u32 << n) - 1
        }
    }

    /// Returns the raw `VkDevice` handle.
    pub fn raw(&self) -> VkDevice {
        self.inner.handle
    }

    /// Get the i-th queue of the given family.
    ///
    /// Note: per the Vulkan spec, the queue must have been requested at
    /// device creation time via [`QueueCreateInfo`]. Calling this with
    /// indices that weren't requested is undefined behavior.
    pub fn get_queue(&self, queue_family_index: u32, queue_index: u32) -> Queue {
        let get = self
            .inner
            .dispatch
            .vkGetDeviceQueue
            .expect("vkGetDeviceQueue is required by Vulkan 1.0");

        let mut queue: VkQueue = std::ptr::null_mut();
        // Safety: device handle is valid, queue indices were validated by
        // the spec at device creation. The output is just a handle; nothing
        // is allocated.
        unsafe {
            get(
                self.inner.handle,
                queue_family_index,
                queue_index,
                &mut queue,
            )
        };

        Queue {
            handle: queue,
            device: Arc::clone(&self.inner),
        }
    }

    /// Wait for the device to become idle.
    pub fn wait_idle(&self) -> Result<()> {
        let wait = self
            .inner
            .dispatch
            .vkDeviceWaitIdle
            .ok_or(Error::MissingFunction("vkDeviceWaitIdle"))?;
        // Safety: device handle is valid.
        check(unsafe { wait(self.inner.handle) })
    }
}

/// A handle to a queue belonging to a [`Device`].
///
/// Queues are owned by the device and cannot be destroyed independently.
/// This handle keeps the device alive via an `Arc`.
#[derive(Clone)]
pub struct Queue {
    pub(crate) handle: VkQueue,
    pub(crate) device: Arc<DeviceInner>,
}

impl Queue {
    /// Returns the raw `VkQueue` handle.
    pub fn raw(&self) -> VkQueue {
        self.handle
    }

    /// Submit one or more command buffers, optionally signaling a fence on completion.
    ///
    /// This is a convenience for the common case where no semaphores are
    /// needed. Use [`submit_with_sync`](Self::submit_with_sync) when you
    /// need to wait on or signal binary or timeline semaphores.
    pub fn submit(
        &self,
        command_buffers: &[&super::CommandBuffer],
        signal_fence: Option<&super::Fence>,
    ) -> Result<()> {
        self.submit_with_sync(command_buffers, &[], &[], signal_fence)
    }

    /// Submit command buffers with explicit semaphore wait/signal lists.
    ///
    /// `wait_semaphores` is the set of semaphores the submission must wait
    /// on before any commands begin (paired with their per-stage masks).
    /// `signal_semaphores` is the set of semaphores the submission will
    /// signal once all commands have completed.
    ///
    /// Both lists may freely mix binary and timeline semaphores. For
    /// binary semaphores the `value` field is ignored; for timeline
    /// semaphores it is the counter value waited-for or signaled.
    pub fn submit_with_sync(
        &self,
        command_buffers: &[&super::CommandBuffer],
        wait_semaphores: &[WaitSemaphore<'_>],
        signal_semaphores: &[SignalSemaphore<'_>],
        signal_fence: Option<&super::Fence>,
    ) -> Result<()> {
        let submit = self
            .device
            .dispatch
            .vkQueueSubmit
            .ok_or(Error::MissingFunction("vkQueueSubmit"))?;

        let raw_cmds: Vec<VkCommandBuffer> = command_buffers.iter().map(|c| c.raw()).collect();

        let raw_wait: Vec<VkSemaphore> =
            wait_semaphores.iter().map(|w| w.semaphore.raw()).collect();
        let raw_wait_stages: Vec<u32> = wait_semaphores.iter().map(|w| w.dst_stage_mask).collect();
        let raw_wait_values: Vec<u64> = wait_semaphores.iter().map(|w| w.value).collect();

        let raw_signal: Vec<VkSemaphore> = signal_semaphores
            .iter()
            .map(|s| s.semaphore.raw())
            .collect();
        let raw_signal_values: Vec<u64> = signal_semaphores.iter().map(|s| s.value).collect();

        // Build the timeline submit info chain. We always include it; for
        // binary-only submits the per-semaphore values are ignored by the
        // implementation. The pointers must outlive the submit call, so we
        // bind them with explicit `let`.
        let timeline_info = VkTimelineSemaphoreSubmitInfo {
            sType: VkStructureType::STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            pNext: std::ptr::null(),
            waitSemaphoreValueCount: raw_wait_values.len() as u32,
            pWaitSemaphoreValues: if raw_wait_values.is_empty() {
                std::ptr::null()
            } else {
                raw_wait_values.as_ptr()
            },
            signalSemaphoreValueCount: raw_signal_values.len() as u32,
            pSignalSemaphoreValues: if raw_signal_values.is_empty() {
                std::ptr::null()
            } else {
                raw_signal_values.as_ptr()
            },
        };

        // If timeline semaphores aren't supported by the loaded device, we
        // still pass the chain — the spec says implementations that don't
        // recognise an unrecognised pNext chain entry must ignore it.
        // (We could elide it for purely binary submits, but it's harmless.)
        let p_next: *const std::ffi::c_void =
            if self.device.dispatch.vkGetSemaphoreCounterValue.is_some() {
                &timeline_info as *const _ as *const _
            } else {
                std::ptr::null()
            };

        let submit_info = VkSubmitInfo {
            sType: VkStructureType::STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: p_next,
            waitSemaphoreCount: raw_wait.len() as u32,
            pWaitSemaphores: if raw_wait.is_empty() {
                std::ptr::null()
            } else {
                raw_wait.as_ptr()
            },
            pWaitDstStageMask: if raw_wait_stages.is_empty() {
                std::ptr::null()
            } else {
                raw_wait_stages.as_ptr()
            },
            commandBufferCount: raw_cmds.len() as u32,
            pCommandBuffers: raw_cmds.as_ptr(),
            signalSemaphoreCount: raw_signal.len() as u32,
            pSignalSemaphores: if raw_signal.is_empty() {
                std::ptr::null()
            } else {
                raw_signal.as_ptr()
            },
        };

        let fence_handle = signal_fence.map_or(0u64, super::Fence::raw);

        // Safety: submit_info is valid for the call, all the Vec backing
        // pointers outlive the call.
        check(unsafe { submit(self.handle, 1, &submit_info, fence_handle) })
    }

    /// Wait for the queue to become idle.
    pub fn wait_idle(&self) -> Result<()> {
        let wait = self
            .device
            .dispatch
            .vkQueueWaitIdle
            .ok_or(Error::MissingFunction("vkQueueWaitIdle"))?;
        // Safety: queue handle is valid.
        check(unsafe { wait(self.handle) })
    }
}
