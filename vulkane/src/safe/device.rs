//! Safe wrapper for `VkDevice` and `VkQueue`.
//!
//! A [`Device`] is the logical connection to a GPU — all resources
//! (buffers, images, pipelines, command pools) are created from it.
//! A [`Queue`] is a submission endpoint belonging to a device; work
//! is recorded into command buffers and submitted to a queue.
//!
//! ## Typical workflow
//!
//! ```ignore
//! use vulkane::safe::{DeviceCreateInfo, QueueCreateInfo, QueueFlags};
//!
//! let qf = physical.find_queue_family(QueueFlags::GRAPHICS).unwrap();
//! let device = physical.create_device(DeviceCreateInfo {
//!     queue_create_infos: &[QueueCreateInfo::single(qf)],
//!     ..Default::default()
//! })?;
//! let queue = device.get_queue(qf, 0);
//!
//! // One-shot command recording + submit + wait:
//! queue.one_shot(&device, qf, |rec| {
//!     rec.fill_buffer(&buffer, 0, 1024, 0xDEADBEEF);
//!     Ok(())
//! })?;
//! ```
//!
//! ## Raw escape hatch
//!
//! [`Device::dispatch()`] returns a reference to the full
//! `VkDeviceDispatchTable` so you can call any Vulkan function the
//! safe wrapper doesn't cover yet. Combine with [`.raw()`](Device::raw)
//! for the first argument.

use super::features::DeviceFeatures;
use super::flags::PipelineStage;
use super::physical::PhysicalDevice;
use super::sync::{Fence, Semaphore};
use super::{Error, Result, check};
use crate::raw::VulkanLibrary;
use crate::raw::bindings::*;
use std::ffi::{CString, c_char, c_void};
use std::sync::Arc;

/// One semaphore wait in [`Queue::submit_with_sync`].
///
/// `value` is ignored for binary semaphores; for timeline semaphores it
/// is the counter value the wait must reach before the submission may
/// proceed. `dst_stage_mask` selects the pipeline stages that must
/// wait for the semaphore (e.g.
/// [`PipelineStage::COLOR_ATTACHMENT_OUTPUT`]).
///
/// `device_index` selects which physical device in a [`Device`] group
/// the wait happens on. `0` is the right choice for non-group devices
/// and is the default of any wait that does not opt in. The field is
/// only consulted when [`Queue::submit_with_groups`] is called *and*
/// at least one wait, signal, or command-buffer mask is non-default.
#[derive(Clone, Copy)]
pub struct WaitSemaphore<'a> {
    pub semaphore: &'a Semaphore,
    pub value: u64,
    pub dst_stage_mask: PipelineStage,
    /// Physical-device index inside the [`Device`] group. Defaults to `0`.
    pub device_index: u32,
}

/// One semaphore signal in [`Queue::submit_with_sync`].
///
/// `value` is ignored for binary semaphores; for timeline semaphores it
/// is the counter value the submission writes when it completes.
///
/// See [`WaitSemaphore::device_index`] for `device_index` semantics.
#[derive(Clone, Copy)]
pub struct SignalSemaphore<'a> {
    pub semaphore: &'a Semaphore,
    pub value: u64,
    /// Physical-device index inside the [`Device`] group. Defaults to `0`.
    pub device_index: u32,
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

impl QueueCreateInfo {
    /// Create a single queue with priority 1.0 from the given family.
    ///
    /// This is the right choice 90% of the time — most applications
    /// need exactly one queue from one family.
    pub fn single(queue_family_index: u32) -> Self {
        Self {
            queue_family_index,
            queue_priorities: vec![1.0],
        }
    }
}

impl WaitSemaphore<'_> {
    /// Wait on a binary semaphore at the given pipeline stage.
    pub fn binary(semaphore: &Semaphore, dst_stage: PipelineStage) -> WaitSemaphore<'_> {
        WaitSemaphore {
            semaphore,
            value: 0,
            dst_stage_mask: dst_stage,
            device_index: 0,
        }
    }

    /// Wait on a timeline semaphore to reach `value` at the given
    /// pipeline stage.
    pub fn timeline(
        semaphore: &Semaphore,
        value: u64,
        dst_stage: PipelineStage,
    ) -> WaitSemaphore<'_> {
        WaitSemaphore {
            semaphore,
            value,
            dst_stage_mask: dst_stage,
            device_index: 0,
        }
    }
}

impl SignalSemaphore<'_> {
    /// Signal a binary semaphore on completion.
    pub fn binary(semaphore: &Semaphore) -> SignalSemaphore<'_> {
        SignalSemaphore {
            semaphore,
            value: 0,
            device_index: 0,
        }
    }

    /// Signal a timeline semaphore with the given value on completion.
    pub fn timeline(semaphore: &Semaphore, value: u64) -> SignalSemaphore<'_> {
        SignalSemaphore {
            semaphore,
            value,
            device_index: 0,
        }
    }
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

    /// Returns a reference to the device-level Vulkan dispatch table.
    ///
    /// Use this to call raw Vulkan functions that the safe wrapper
    /// doesn't cover yet. Each field is an `Option<fn_ptr>` — `None`
    /// if the driver doesn't expose it, `Some` if callable.
    ///
    /// Combine with [`.raw()`](Self::raw) to get the raw `VkDevice`
    /// handle for the first argument.
    pub fn dispatch(&self) -> &VkDeviceDispatchTable {
        &self.inner.dispatch
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
    ///
    /// If any [`WaitSemaphore::device_index`] or
    /// [`SignalSemaphore::device_index`] is non-zero, a
    /// `VkDeviceGroupSubmitInfo` is automatically chained on `pNext` so
    /// the wait/signal happens on the requested physical device. To
    /// also override the per-command-buffer device mask (which physical
    /// devices in the group execute the work), use
    /// [`submit_with_groups`](Self::submit_with_groups) instead.
    pub fn submit_with_sync(
        &self,
        command_buffers: &[&super::CommandBuffer],
        wait_semaphores: &[WaitSemaphore<'_>],
        signal_semaphores: &[SignalSemaphore<'_>],
        signal_fence: Option<&super::Fence>,
    ) -> Result<()> {
        self.submit_with_groups(
            command_buffers,
            None,
            wait_semaphores,
            signal_semaphores,
            signal_fence,
        )
    }

    /// Like [`submit_with_sync`](Self::submit_with_sync), but with an
    /// explicit per-command-buffer device-mask list. When
    /// `command_buffer_device_masks` is `Some`, its length must equal
    /// `command_buffers.len()`; each entry is a bitmask of physical
    /// devices in the [`Device`] group that should execute the
    /// corresponding command buffer.
    ///
    /// When `command_buffer_device_masks` is `None`, the per-CB masks
    /// fall back to "all devices in the group" (the [`Device`]'s
    /// [`default_device_mask`](Device::default_device_mask)).
    ///
    /// `VkDeviceGroupSubmitInfo` is chained whenever any of the
    /// following is true:
    /// - `command_buffer_device_masks` is `Some(_)`
    /// - any wait `device_index` is non-zero
    /// - any signal `device_index` is non-zero
    ///
    /// On a single-device group all of these reduce to no-ops, so it is
    /// safe to call this method unconditionally.
    pub fn submit_with_groups(
        &self,
        command_buffers: &[&super::CommandBuffer],
        command_buffer_device_masks: Option<&[u32]>,
        wait_semaphores: &[WaitSemaphore<'_>],
        signal_semaphores: &[SignalSemaphore<'_>],
        signal_fence: Option<&super::Fence>,
    ) -> Result<()> {
        let submit = self
            .device
            .dispatch
            .vkQueueSubmit
            .ok_or(Error::MissingFunction("vkQueueSubmit"))?;

        if let Some(masks) = command_buffer_device_masks
            && masks.len() != command_buffers.len()
        {
            return Err(Error::InvalidArgument(
                "submit_with_groups: command_buffer_device_masks length must \
                 match command_buffers length",
            ));
        }

        let raw_cmds: Vec<VkCommandBuffer> = command_buffers.iter().map(|c| c.raw()).collect();

        let raw_wait: Vec<VkSemaphore> =
            wait_semaphores.iter().map(|w| w.semaphore.raw()).collect();
        let raw_wait_stages: Vec<u32> = wait_semaphores.iter().map(|w| w.dst_stage_mask.0).collect();
        let raw_wait_values: Vec<u64> = wait_semaphores.iter().map(|w| w.value).collect();
        let raw_wait_device_indices: Vec<u32> =
            wait_semaphores.iter().map(|w| w.device_index).collect();

        let raw_signal: Vec<VkSemaphore> = signal_semaphores
            .iter()
            .map(|s| s.semaphore.raw())
            .collect();
        let raw_signal_values: Vec<u64> = signal_semaphores.iter().map(|s| s.value).collect();
        let raw_signal_device_indices: Vec<u32> = signal_semaphores
            .iter()
            .map(|s| s.device_index)
            .collect();

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
        let mut p_next: *const std::ffi::c_void =
            if self.device.dispatch.vkGetSemaphoreCounterValue.is_some() {
                &timeline_info as *const _ as *const _
            } else {
                std::ptr::null()
            };

        // Decide whether we need a VkDeviceGroupSubmitInfo. Either the
        // caller passed an explicit per-CB mask list, or any wait/signal
        // device_index is non-zero. We materialize the per-CB mask vector
        // (synthesizing the default mask when omitted) so the chain
        // pointers are valid through the call.
        let needs_group_chain = command_buffer_device_masks.is_some()
            || raw_wait_device_indices.iter().any(|&i| i != 0)
            || raw_signal_device_indices.iter().any(|&i| i != 0);

        let cb_masks_owned: Vec<u32> = if needs_group_chain {
            match command_buffer_device_masks {
                Some(masks) => masks.to_vec(),
                None => {
                    // Default = "all devices in the group", which on a
                    // single-device group is just 0b1.
                    let default_mask = (1u32 << self.device.physical_devices.len() as u32)
                        .wrapping_sub(1);
                    vec![default_mask; raw_cmds.len()]
                }
            }
        } else {
            Vec::new()
        };

        let group_info = VkDeviceGroupSubmitInfo {
            sType: VkStructureType::STRUCTURE_TYPE_DEVICE_GROUP_SUBMIT_INFO,
            pNext: p_next,
            waitSemaphoreCount: raw_wait_device_indices.len() as u32,
            pWaitSemaphoreDeviceIndices: if raw_wait_device_indices.is_empty() {
                std::ptr::null()
            } else {
                raw_wait_device_indices.as_ptr()
            },
            commandBufferCount: cb_masks_owned.len() as u32,
            pCommandBufferDeviceMasks: if cb_masks_owned.is_empty() {
                std::ptr::null()
            } else {
                cb_masks_owned.as_ptr()
            },
            signalSemaphoreCount: raw_signal_device_indices.len() as u32,
            pSignalSemaphoreDeviceIndices: if raw_signal_device_indices.is_empty() {
                std::ptr::null()
            } else {
                raw_signal_device_indices.as_ptr()
            },
        };

        if needs_group_chain {
            p_next = &group_info as *const _ as *const _;
        }

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
        // pointers (raw_*, cb_masks_owned, timeline_info, group_info)
        // outlive the call.
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

    /// Record commands in a one-shot command buffer, submit, and wait
    /// for completion before returning.
    ///
    /// This is the simplest way to do fire-and-forget GPU work like
    /// uploading data, issuing layout transitions, or running a single
    /// compute dispatch. The command pool, command buffer, and fence
    /// are created and destroyed internally.
    ///
    /// ```ignore
    /// queue.one_shot(&device, queue_family, |rec| {
    ///     rec.copy_buffer(&staging, &device_buf, &[BufferCopy::full(size)]);
    ///     Ok(())
    /// })?;
    /// ```
    pub fn one_shot<F>(&self, device: &Device, queue_family_index: u32, record: F) -> Result<()>
    where
        F: FnOnce(&mut super::CommandBufferRecording<'_>) -> Result<()>,
    {
        let pool = super::CommandPool::new(device, queue_family_index)?;
        let mut cmd = pool.allocate_primary()?;
        {
            let mut rec = cmd.begin()?;
            record(&mut rec)?;
            rec.end()?;
        }
        let fence = Fence::new(device)?;
        self.submit(&[&cmd], Some(&fence))?;
        fence.wait(u64::MAX)?;
        Ok(())
    }

    /// Upload a slice of `Copy` data into a new device-local buffer.
    ///
    /// Internally creates a staging buffer, maps and copies the data,
    /// issues a one-shot `vkCmdCopyBuffer`, and waits. Returns a
    /// device-local buffer ready for shader use.
    ///
    /// `usage` is ORed with `TRANSFER_DST` automatically so you only
    /// need to specify the final usage (e.g. `BufferUsage::STORAGE_BUFFER`
    /// or `BufferUsage::VERTEX_BUFFER`).
    pub fn upload_buffer<T: Copy>(
        &self,
        device: &Device,
        physical: &PhysicalDevice,
        queue_family_index: u32,
        data: &[T],
        usage: super::BufferUsage,
    ) -> Result<(super::Buffer, super::DeviceMemory)> {
        use super::buffer::{Buffer, BufferCreateInfo};
        use super::command::BufferCopy;

        let byte_size = std::mem::size_of_val(data) as u64;

        // Staging (host-visible, TRANSFER_SRC).
        let (staging, mut staging_mem) = Buffer::new_bound(
            device,
            physical,
            BufferCreateInfo {
                size: byte_size,
                usage: super::BufferUsage::TRANSFER_SRC,
            },
            super::MemoryPropertyFlags::HOST_VISIBLE | super::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        {
            let mut m = staging_mem.map()?;
            let src = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_size as usize)
            };
            m.as_slice_mut()[..byte_size as usize].copy_from_slice(src);
        }

        // Device-local (TRANSFER_DST | user usage).
        let (gpu_buf, gpu_mem) = Buffer::new_bound(
            device,
            physical,
            BufferCreateInfo {
                size: byte_size,
                usage: super::BufferUsage::TRANSFER_DST | usage,
            },
            super::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .or_else(|_| {
            // Fallback for integrated GPUs without a separate device-local heap.
            Buffer::new_bound(
                device,
                physical,
                BufferCreateInfo {
                    size: byte_size,
                    usage: super::BufferUsage::TRANSFER_DST | usage,
                },
                super::MemoryPropertyFlags::HOST_VISIBLE,
            )
        })?;

        // One-shot copy.
        self.one_shot(device, queue_family_index, |rec| {
            rec.copy_buffer(
                &staging,
                &gpu_buf,
                &[BufferCopy::full(byte_size)],
            );
            Ok(())
        })?;

        Ok((gpu_buf, gpu_mem))
    }

    /// Upload RGBA8 pixel data into a new device-local sampled image.
    ///
    /// Creates a staging buffer, copies the pixels, issues a one-shot
    /// command buffer with layout transitions
    /// (`UNDEFINED → TRANSFER_DST → SHADER_READ_ONLY`), and waits.
    /// Returns a device-local image + color view ready for sampling.
    ///
    /// `pixels` must be exactly `width * height * 4` bytes of
    /// tightly-packed RGBA8 data.
    pub fn upload_image_rgba(
        &self,
        device: &Device,
        physical: &PhysicalDevice,
        queue_family_index: u32,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Result<(super::Image, super::DeviceMemory, super::ImageView)> {
        use super::buffer::{Buffer, BufferCreateInfo};
        use super::flags::AccessFlags;
        use super::image::{
            BufferImageCopy, Image, Image2dCreateInfo, ImageBarrier, ImageView,
        };

        let byte_size = (width as u64) * (height as u64) * 4;
        assert_eq!(
            pixels.len() as u64, byte_size,
            "upload_image_rgba: pixels.len() must be width * height * 4"
        );

        // Staging buffer.
        let (staging, mut staging_mem) = Buffer::new_bound(
            device,
            physical,
            BufferCreateInfo {
                size: byte_size,
                usage: super::BufferUsage::TRANSFER_SRC,
            },
            super::MemoryPropertyFlags::HOST_VISIBLE | super::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        {
            let mut m = staging_mem.map()?;
            m.as_slice_mut()[..pixels.len()].copy_from_slice(pixels);
        }

        // Device-local image (TRANSFER_DST | SAMPLED).
        let image = Image::new_2d(
            device,
            Image2dCreateInfo {
                format: super::Format::R8G8B8A8_UNORM,
                width,
                height,
                usage: super::ImageUsage::TRANSFER_DST | super::ImageUsage::SAMPLED,
            },
        )?;
        let req = image.memory_requirements();
        let type_index = physical
            .find_memory_type(req.memory_type_bits, super::MemoryPropertyFlags::DEVICE_LOCAL)
            .or_else(|| {
                physical.find_memory_type(
                    req.memory_type_bits,
                    super::MemoryPropertyFlags::HOST_VISIBLE,
                )
            })
            .ok_or(Error::InvalidArgument(
                "no compatible memory type for sampled image",
            ))?;
        let img_mem = super::DeviceMemory::allocate(device, req.size, type_index)?;
        image.bind_memory(&img_mem, 0)?;
        let view = ImageView::new_2d_color(&image)?;

        // One-shot: transition → copy → transition.
        self.one_shot(device, queue_family_index, |rec| {
            rec.image_barrier(
                PipelineStage::TOP_OF_PIPE,
                PipelineStage::TRANSFER,
                ImageBarrier::color(
                    &image,
                    super::ImageLayout::UNDEFINED,
                    super::ImageLayout::TRANSFER_DST_OPTIMAL,
                    AccessFlags::NONE,
                    AccessFlags::TRANSFER_WRITE,
                ),
            );
            rec.copy_buffer_to_image(
                &staging,
                &image,
                super::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[BufferImageCopy::full_2d(width, height)],
            );
            rec.image_barrier(
                PipelineStage::TRANSFER,
                PipelineStage::FRAGMENT_SHADER,
                ImageBarrier::color(
                    &image,
                    super::ImageLayout::TRANSFER_DST_OPTIMAL,
                    super::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    AccessFlags::TRANSFER_WRITE,
                    AccessFlags::SHADER_READ,
                ),
            );
            Ok(())
        })?;

        Ok((image, img_mem, view))
    }
}
