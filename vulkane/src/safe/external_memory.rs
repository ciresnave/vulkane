// SPDX-License-Identifier: MIT OR Apache-2.0
//! Ergonomic wrappers for `VK_KHR_external_memory_{win32,fd}`.
//!
//! These extensions let you export an allocation's platform-native handle
//! (a Windows `HANDLE` or a POSIX file descriptor) so it can be imported
//! by a *different* API on the same device — the typical use cases are:
//!
//! - CUDA / HIP interop (`cudaExternalMemory`, `hipExternalMemory`).
//! - DirectX 12 interop (`ID3D12Resource` sharing on Windows).
//! - cross-process or cross-engine buffer sharing (DMA-BUF on Linux).
//!
//! ## Creating an exportable allocation
//!
//! The export capability is configured at *creation time*. You must:
//!
//! 1. Enable `VK_KHR_external_memory` plus the platform-specific
//!    sub-extension (`_win32` or `_fd`) on the [`Device`](super::Device).
//! 2. Attach `VkExternalMemoryBufferCreateInfo` /
//!    `VkExternalMemoryImageCreateInfo` to the buffer / image create
//!    info via [`Buffer::new_with_pnext`](super::Buffer::new_with_pnext)
//!    or [`Image::new_2d_with_pnext`](super::Image::new_2d_with_pnext).
//! 3. Attach `VkExportMemoryAllocateInfo` to the memory allocate info
//!    via [`DeviceMemory::allocate_with`](super::DeviceMemory::allocate_with).
//! 4. After binding, call [`DeviceMemory::get_win32_handle`] or
//!    [`DeviceMemory::get_fd`] to extract the native handle.
//!
//! ```ignore
//! use vulkane::raw::bindings::*;
//! use vulkane::raw::PNextChainable;
//! use vulkane::safe::{
//!     Buffer, BufferCreateInfo, BufferUsage, DeviceMemory, MemoryAllocateInfo, PNextChain,
//! };
//!
//! # let (device, physical) = todo!();
//! // 1. Buffer marked as exportable.
//! let mut buf_chain = PNextChain::new();
//! let mut buf_ext = VkExternalMemoryBufferCreateInfo::new_pnext();
//! buf_ext.handleTypes = EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
//! buf_chain.push(buf_ext);
//! let buffer = Buffer::new_with_pnext(
//!     &device,
//!     BufferCreateInfo { size: 4096, usage: BufferUsage::STORAGE_BUFFER },
//!     Some(&buf_chain),
//! )?;
//!
//! // 2. Memory marked as exportable.
//! let req = buffer.memory_requirements();
//! let mt = physical.find_memory_type(req.memory_type_bits, /*…*/).unwrap();
//! let mut mem_chain = PNextChain::new();
//! let mut mem_ext = VkExportMemoryAllocateInfo::new_pnext();
//! mem_ext.handleTypes = EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
//! mem_chain.push(mem_ext);
//! let memory = DeviceMemory::allocate_with(&device, &MemoryAllocateInfo {
//!     size: req.size,
//!     memory_type_index: mt,
//!     pnext: Some(&mem_chain),
//! })?;
//! buffer.bind_memory(&memory, 0)?;
//!
//! // 3. Extract the Win32 HANDLE to hand to CUDA / D3D12.
//! let handle = memory.get_win32_handle(EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT)?;
//! # Ok::<(), vulkane::safe::Error>(())
//! ```

use super::{DeviceMemory, Error, Result, check};
use crate::raw::bindings::*;

/// A Windows `HANDLE` returned from a Vulkan external-memory or
/// external-semaphore export.
///
/// The handle is *not* closed on drop. The Vulkan spec makes the caller
/// responsible for handle ownership: for NT handle types
/// (`OPAQUE_WIN32`, `D3D11_TEXTURE`, `D3D12_HEAP`, `D3D12_RESOURCE`) the
/// returned handle is a fresh NT handle that must be released with
/// `CloseHandle` when no longer needed. For KMT handle types
/// (`_KMT_BIT`) the handle is a shared KMT handle whose lifetime is
/// controlled separately; closing it would be incorrect.
///
/// Because the correct disposal depends on the handle type — and
/// because most callers forward the handle straight to another API
/// (CUDA, D3D12) which assumes ownership — `Win32Handle` does *not*
/// auto-close. Callers must track ownership themselves.
#[cfg(windows)]
#[derive(Debug)]
pub struct Win32Handle {
    /// The raw `HANDLE` value. `*mut c_void` on Windows; may be null if
    /// the driver returns `NULL` for a non-exportable allocation.
    pub raw: HANDLE,
    /// The handle type that was requested when exporting — useful for
    /// deciding whether `CloseHandle` is the right disposal.
    pub handle_type: VkExternalMemoryHandleTypeFlagBits,
}

// HANDLE is *mut c_void which is !Send + !Sync by default. In practice
// Win32 HANDLEs are Send (they are process-global integer-like values).
// Mark explicitly so the caller can thread it to another runtime.
#[cfg(windows)]
unsafe impl Send for Win32Handle {}
#[cfg(windows)]
unsafe impl Sync for Win32Handle {}

impl DeviceMemory {
    /// Export this allocation as a Windows `HANDLE`.
    ///
    /// Requires:
    ///
    /// - `VK_KHR_external_memory` and `VK_KHR_external_memory_win32` enabled
    ///   on the device.
    /// - The allocation was created with `VkExportMemoryAllocateInfo`
    ///   listing `handle_type` in its `handleTypes` mask.
    ///
    /// For KMT handle types (`EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT`
    /// and friends) the returned `HANDLE` is a shared KMT handle; it is
    /// *not* closeable with `CloseHandle`. For NT handle types the caller
    /// must eventually call `CloseHandle` on the returned value (or pass
    /// it to another API that assumes ownership, like cuImportExternalMemory).
    #[cfg(windows)]
    pub fn get_win32_handle(
        &self,
        handle_type: VkExternalMemoryHandleTypeFlagBits,
    ) -> Result<Win32Handle> {
        let f = self
            .device
            .dispatch
            .vkGetMemoryWin32HandleKHR
            .ok_or(Error::MissingFunction("vkGetMemoryWin32HandleKHR"))?;

        let info = VkMemoryGetWin32HandleInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
            pNext: std::ptr::null(),
            memory: self.handle,
            handleType: handle_type,
        };

        let mut raw: HANDLE = std::ptr::null_mut();
        // Safety: info is valid for the call; output HANDLE is written
        // by the driver on success.
        check(unsafe { f(self.device.handle, &info, &mut raw) })?;

        Ok(Win32Handle { raw, handle_type })
    }

    /// Export this allocation as a POSIX file descriptor.
    ///
    /// Requires:
    ///
    /// - `VK_KHR_external_memory` and `VK_KHR_external_memory_fd` enabled
    ///   on the device.
    /// - The allocation was created with `VkExportMemoryAllocateInfo`
    ///   listing `handle_type` in its `handleTypes` mask
    ///   (typically `EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT` or
    ///   `EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`).
    ///
    /// The returned fd is owned — it will be closed on drop. Hand it to
    /// CUDA / HIP / a child process before the [`std::os::fd::OwnedFd`]
    /// is dropped if the consumer is meant to inherit ownership.
    #[cfg(unix)]
    pub fn get_fd(
        &self,
        handle_type: VkExternalMemoryHandleTypeFlagBits,
    ) -> Result<std::os::fd::OwnedFd> {
        use std::os::fd::{FromRawFd, OwnedFd};

        let f = self
            .device
            .dispatch
            .vkGetMemoryFdKHR
            .ok_or(Error::MissingFunction("vkGetMemoryFdKHR"))?;

        let info = VkMemoryGetFdInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
            pNext: std::ptr::null(),
            memory: self.handle,
            handleType: handle_type,
        };

        let mut fd: i32 = -1;
        // Safety: info is valid; output fd is written on success.
        check(unsafe { f(self.device.handle, &info, &mut fd) })?;

        if fd < 0 {
            return Err(Error::InvalidArgument(
                "driver returned a negative file descriptor on success",
            ));
        }
        // Safety: fd was just produced by the driver as a newly-opened
        // file descriptor that the spec transfers ownership of to us.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}
