//! Ergonomic wrappers for `VK_KHR_external_semaphore_{win32,fd}`.
//!
//! These extensions let you share semaphore payloads across API
//! boundaries — for example, waiting on a CUDA stream from a Vulkan
//! queue submit, or signalling a D3D12 fence from a Vulkan queue.
//!
//! The export/import model mirrors external memory:
//!
//! 1. Enable `VK_KHR_external_semaphore` and the platform-specific
//!    sub-extension on the [`Device`](super::Device).
//! 2. At creation, attach `VkExportSemaphoreCreateInfo` via
//!    [`Semaphore::binary_with_pnext`](super::Semaphore::binary_with_pnext)
//!    or [`Semaphore::timeline_with_pnext`](super::Semaphore::timeline_with_pnext).
//! 3. Call [`Semaphore::get_win32_handle`] / [`Semaphore::get_fd`] to
//!    extract the handle to hand to CUDA / HIP / D3D12.
//! 4. For importing, create a placeholder semaphore then call
//!    [`Semaphore::import_win32_handle`] / [`Semaphore::import_fd`]
//!    passing the handle obtained from the other API.

use super::{Error, Result, Semaphore, check};
use crate::raw::bindings::*;

/// Parameters for [`Semaphore::import_win32_handle`].
///
/// Named cross-process handles (the `name` field of
/// `VkImportSemaphoreWin32HandleInfoKHR`) are not exposed here — most
/// interop callers use unnamed handles sourced from another API
/// (CUDA / D3D12). Use the raw `DeviceExt::vk_import_semaphore_win32_handle_khr`
/// entry point if you need named-handle import.
#[cfg(windows)]
#[derive(Debug, Clone, Copy)]
pub struct SemaphoreImportWin32 {
    /// The handle type the payload represents.
    pub handle_type: VkExternalSemaphoreHandleTypeFlagBits,
    /// Flags (e.g. `VK_SEMAPHORE_IMPORT_TEMPORARY_BIT`).
    pub flags: VkSemaphoreImportFlags,
    /// The Win32 handle to import.
    pub handle: HANDLE,
}

/// Parameters for [`Semaphore::import_fd`].
#[cfg(unix)]
#[derive(Debug)]
pub struct SemaphoreImportFd {
    /// The handle type the payload represents.
    pub handle_type: VkExternalSemaphoreHandleTypeFlagBits,
    /// Flags (e.g. `VK_SEMAPHORE_IMPORT_TEMPORARY_BIT`).
    pub flags: VkSemaphoreImportFlags,
    /// The file descriptor to import. Ownership transfers to Vulkan on
    /// success.
    pub fd: std::os::fd::OwnedFd,
}

impl Semaphore {
    /// Export this semaphore as a Windows `HANDLE`.
    ///
    /// Requires:
    ///
    /// - `VK_KHR_external_semaphore` and `VK_KHR_external_semaphore_win32`
    ///   enabled on the device.
    /// - The semaphore was created with `VkExportSemaphoreCreateInfo`
    ///   listing `handle_type` in its `handleTypes` mask — use
    ///   [`Semaphore::binary_with_pnext`] or
    ///   [`Semaphore::timeline_with_pnext`].
    ///
    /// The returned handle is a fresh NT handle for NT handle types and
    /// a shared KMT handle for KMT types; disposal rules are the same as
    /// for [`super::Win32Handle`].
    #[cfg(windows)]
    pub fn get_win32_handle(
        &self,
        handle_type: VkExternalSemaphoreHandleTypeFlagBits,
    ) -> Result<super::Win32Handle> {
        let f = self
            .device
            .dispatch
            .vkGetSemaphoreWin32HandleKHR
            .ok_or(Error::MissingFunction("vkGetSemaphoreWin32HandleKHR"))?;

        let info = VkSemaphoreGetWin32HandleInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
            pNext: std::ptr::null(),
            semaphore: self.handle,
            handleType: handle_type,
        };

        let mut raw: HANDLE = std::ptr::null_mut();
        // Safety: info valid for the call; output handle written on success.
        check(unsafe { f(self.device.handle, &info, &mut raw) })?;

        Ok(super::Win32Handle { raw, handle_type })
    }

    /// Import a Win32 handle payload into this semaphore.
    ///
    /// The semaphore must already exist (create one with
    /// [`Semaphore::binary`]). The import replaces or temporarily
    /// overrides the semaphore's payload depending on `flags`.
    #[cfg(windows)]
    pub fn import_win32_handle(&self, params: SemaphoreImportWin32) -> Result<()> {
        let f = self
            .device
            .dispatch
            .vkImportSemaphoreWin32HandleKHR
            .ok_or(Error::MissingFunction("vkImportSemaphoreWin32HandleKHR"))?;

        let info = VkImportSemaphoreWin32HandleInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR,
            pNext: std::ptr::null(),
            semaphore: self.handle,
            flags: params.flags,
            handleType: params.handle_type,
            handle: params.handle,
            name: std::ptr::null_mut(),
        };

        // Safety: info valid for the call. Driver consumes the handle
        // reference (duplicates or takes ownership per spec).
        check(unsafe { f(self.device.handle, &info) })
    }

    /// Export this semaphore as a POSIX file descriptor.
    ///
    /// Returned fd is owned and closes on drop — hand it off before the
    /// `OwnedFd` drops if the consumer must inherit it.
    #[cfg(unix)]
    pub fn get_fd(
        &self,
        handle_type: VkExternalSemaphoreHandleTypeFlagBits,
    ) -> Result<std::os::fd::OwnedFd> {
        use std::os::fd::{FromRawFd, OwnedFd};

        let f = self
            .device
            .dispatch
            .vkGetSemaphoreFdKHR
            .ok_or(Error::MissingFunction("vkGetSemaphoreFdKHR"))?;

        let info = VkSemaphoreGetFdInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
            pNext: std::ptr::null(),
            semaphore: self.handle,
            handleType: handle_type,
        };

        let mut fd: i32 = -1;
        // Safety: info valid for the call; fd written on success.
        check(unsafe { f(self.device.handle, &info, &mut fd) })?;

        if fd < 0 {
            return Err(Error::InvalidArgument(
                "driver returned a negative file descriptor on success",
            ));
        }
        // Safety: fd was just allocated by the driver; spec transfers
        // ownership to the caller.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// Import a file descriptor payload into this semaphore.
    ///
    /// Ownership of the fd transfers to Vulkan on success; on failure
    /// the fd remains owned by the caller (via the `OwnedFd` that was
    /// passed in, which is dropped normally).
    #[cfg(unix)]
    pub fn import_fd(&self, params: SemaphoreImportFd) -> Result<()> {
        use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};

        let f = self
            .device
            .dispatch
            .vkImportSemaphoreFdKHR
            .ok_or(Error::MissingFunction("vkImportSemaphoreFdKHR"))?;

        let raw_fd = params.fd.into_raw_fd();

        let info = VkImportSemaphoreFdInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
            pNext: std::ptr::null(),
            semaphore: self.handle,
            flags: params.flags,
            handleType: params.handle_type,
            fd: raw_fd,
        };

        // Safety: info valid; Vulkan takes ownership of the fd on success.
        let r = unsafe { f(self.device.handle, &info) };
        if (r as i32) < 0 {
            // On failure Vulkan did not consume the fd — reconstitute an
            // OwnedFd and let Drop close it, keeping the no-leak guarantee.
            // Safety: raw_fd came from IntoRawFd above and has not been
            // closed or duplicated since.
            drop(unsafe { OwnedFd::from_raw_fd(raw_fd) });
            return Err(Error::Vk(r));
        }
        Ok(())
    }
}
