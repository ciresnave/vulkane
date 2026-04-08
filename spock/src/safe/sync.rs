//! Safe wrappers for Vulkan synchronization primitives.
//!
//! - [`Fence`] — CPU↔GPU sync. Use to wait on the host for a queue
//!   submission to finish.
//! - [`Semaphore`] — GPU↔GPU sync. Two flavors:
//!   - **Binary** (Vulkan 1.0): a one-shot toggle, signaled once and
//!     consumed by exactly one wait.
//!   - **Timeline** (Vulkan 1.2 core): a monotonically-increasing 64-bit
//!     counter that the host *and* the GPU can read, signal, and wait on.
//!     Strictly more powerful than binary semaphores — represents an
//!     entire DAG of dependencies in one object. Requires Vulkan 1.2 (or
//!     `VK_KHR_timeline_semaphore` on 1.0/1.1) and the
//!     `timelineSemaphore` device feature, both of which are universal on
//!     modern desktop GPUs.

use super::device::DeviceInner;
use super::{Device, Error, Result, check};
use crate::raw::bindings::*;
use std::sync::Arc;

/// A safe wrapper around `VkFence`.
///
/// Fences are CPU-GPU synchronization primitives: a queue submission can
/// signal a fence on completion, and the host can wait on the fence.
///
/// The fence is destroyed automatically on drop.
pub struct Fence {
    pub(crate) handle: VkFence,
    pub(crate) device: Arc<DeviceInner>,
}

impl Fence {
    /// Create a new fence in the unsignaled state.
    pub fn new(device: &Device) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreateFence
            .ok_or(Error::MissingFunction("vkCreateFence"))?;

        let info = VkFenceCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_FENCE_CREATE_INFO,
            ..Default::default()
        };

        let mut handle: VkFence = 0;
        // Safety: info is valid for the call, device is valid.
        check(unsafe { create(device.inner.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
        })
    }

    /// Returns the raw `VkFence` handle.
    pub fn raw(&self) -> VkFence {
        self.handle
    }

    /// Block the calling thread until the fence is signaled, or until the
    /// timeout (in nanoseconds) elapses. Pass `u64::MAX` to wait forever.
    pub fn wait(&self, timeout_nanos: u64) -> Result<()> {
        let wait = self
            .device
            .dispatch
            .vkWaitForFences
            .ok_or(Error::MissingFunction("vkWaitForFences"))?;

        // Safety: handle is valid; we wait on a single fence (count = 1).
        check(unsafe {
            wait(
                self.device.handle,
                1,
                &self.handle,
                1, // wait_all (one fence, doesn't matter)
                timeout_nanos,
            )
        })
    }

    /// Reset the fence back to the unsignaled state.
    pub fn reset(&self) -> Result<()> {
        let reset = self
            .device
            .dispatch
            .vkResetFences
            .ok_or(Error::MissingFunction("vkResetFences"))?;
        // Safety: handle is valid.
        check(unsafe { reset(self.device.handle, 1, &self.handle) })
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroyFence {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}

/// Whether a [`Semaphore`] is a one-shot binary semaphore or a counted
/// timeline semaphore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemaphoreKind {
    /// Vulkan 1.0 binary semaphore: signaled or unsignaled, consumed by
    /// exactly one wait.
    Binary,
    /// Vulkan 1.2 timeline semaphore: holds a monotonically increasing
    /// `u64` counter.
    Timeline,
}

/// A safe wrapper around `VkSemaphore`.
///
/// Semaphores synchronize work between queues (GPU↔GPU) and between the
/// host and a timeline semaphore (CPU↔GPU). Use [`Semaphore::binary`] for
/// the classic Vulkan 1.0 binary semaphore and [`Semaphore::timeline`] for
/// the much more flexible Vulkan 1.2 timeline semaphore.
///
/// The semaphore is destroyed automatically on drop.
pub struct Semaphore {
    pub(crate) handle: VkSemaphore,
    pub(crate) device: Arc<DeviceInner>,
    pub(crate) kind: SemaphoreKind,
}

impl Semaphore {
    /// Create a new binary semaphore in the unsignaled state.
    pub fn binary(device: &Device) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreateSemaphore
            .ok_or(Error::MissingFunction("vkCreateSemaphore"))?;

        let info = VkSemaphoreCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            ..Default::default()
        };

        let mut handle: VkSemaphore = 0;
        // Safety: info is valid for the call, device is valid.
        check(unsafe { create(device.inner.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
            kind: SemaphoreKind::Binary,
        })
    }

    /// Create a new timeline semaphore with the given initial counter value.
    ///
    /// Returns an error wrapping `MissingFunction` if the device does not
    /// expose `vkGetSemaphoreCounterValue` (the canonical way to detect
    /// that timeline semaphores are not available — Vulkan 1.0/1.1 without
    /// `VK_KHR_timeline_semaphore`).
    pub fn timeline(device: &Device, initial_value: u64) -> Result<Self> {
        // Sanity-check that timeline semaphores are even loadable.
        if device.inner.dispatch.vkGetSemaphoreCounterValue.is_none() {
            return Err(Error::MissingFunction("vkGetSemaphoreCounterValue"));
        }
        let create = device
            .inner
            .dispatch
            .vkCreateSemaphore
            .ok_or(Error::MissingFunction("vkCreateSemaphore"))?;

        let type_info = VkSemaphoreTypeCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            semaphoreType: VkSemaphoreType::SEMAPHORE_TYPE_TIMELINE,
            initialValue: initial_value,
            ..Default::default()
        };

        let info = VkSemaphoreCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            pNext: &type_info as *const _ as *const _,
            ..Default::default()
        };

        let mut handle: VkSemaphore = 0;
        // Safety: info is valid for the call (and pNext points at type_info,
        // which lives until end of scope).
        check(unsafe { create(device.inner.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
            kind: SemaphoreKind::Timeline,
        })
    }

    /// Returns the raw `VkSemaphore` handle.
    pub fn raw(&self) -> VkSemaphore {
        self.handle
    }

    /// Whether this is a binary or timeline semaphore.
    pub fn kind(&self) -> SemaphoreKind {
        self.kind
    }

    /// Read the current counter value of a timeline semaphore.
    /// Returns an error if called on a binary semaphore.
    pub fn current_value(&self) -> Result<u64> {
        if self.kind != SemaphoreKind::Timeline {
            return Err(Error::MissingFunction("current_value on binary semaphore"));
        }
        let get = self
            .device
            .dispatch
            .vkGetSemaphoreCounterValue
            .ok_or(Error::MissingFunction("vkGetSemaphoreCounterValue"))?;
        let mut v: u64 = 0;
        // Safety: device and handle are valid.
        check(unsafe { get(self.device.handle, self.handle, &mut v) })?;
        Ok(v)
    }

    /// Signal a timeline semaphore from the host with the given value.
    /// `value` must be strictly greater than the current value (otherwise
    /// Vulkan returns an error).
    pub fn signal_value(&self, value: u64) -> Result<()> {
        if self.kind != SemaphoreKind::Timeline {
            return Err(Error::MissingFunction("signal_value on binary semaphore"));
        }
        let signal = self
            .device
            .dispatch
            .vkSignalSemaphore
            .ok_or(Error::MissingFunction("vkSignalSemaphore"))?;

        let info = VkSemaphoreSignalInfo {
            sType: VkStructureType::STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
            semaphore: self.handle,
            value,
            ..Default::default()
        };
        // Safety: handle and info are valid for the call.
        check(unsafe { signal(self.device.handle, &info) })
    }

    /// Block the calling thread until the timeline semaphore reaches at
    /// least `value`, or until `timeout_nanos` elapses (`u64::MAX` for no
    /// timeout).
    pub fn wait_value(&self, value: u64, timeout_nanos: u64) -> Result<()> {
        if self.kind != SemaphoreKind::Timeline {
            return Err(Error::MissingFunction("wait_value on binary semaphore"));
        }
        let wait = self
            .device
            .dispatch
            .vkWaitSemaphores
            .ok_or(Error::MissingFunction("vkWaitSemaphores"))?;

        let info = VkSemaphoreWaitInfo {
            sType: VkStructureType::STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            semaphoreCount: 1,
            pSemaphores: &self.handle,
            pValues: &value,
            ..Default::default()
        };
        // Safety: handle and info are valid for the call.
        check(unsafe { wait(self.device.handle, &info, timeout_nanos) })
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroySemaphore {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}
