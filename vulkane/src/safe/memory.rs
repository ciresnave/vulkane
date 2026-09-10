// SPDX-License-Identifier: MIT OR Apache-2.0
//! Safe wrapper for `VkDeviceMemory` and host memory mapping.
//!
//! [`DeviceMemory`] represents an allocation of GPU memory. It must be
//! bound to a [`Buffer`](super::Buffer) or [`Image`](super::Image)
//! before either can be used.
//!
//! For most use cases, prefer the convenience helpers that handle
//! allocation and binding together:
//!
//! - [`Buffer::new_bound`](super::Buffer::new_bound) /
//!   [`Image::new_2d_bound`](super::Image::new_2d_bound) — one-call
//!   create + allocate + bind.
//! - [`Queue::upload_buffer`](super::Queue::upload_buffer) — staging
//!   upload in one call.
//! - [`Allocator::create_buffer`](super::Allocator::create_buffer) —
//!   sub-allocated from a pool.
//!
//! Use `DeviceMemory` directly only when you need precise control over
//! memory type selection, sub-resource binding offsets, or manual
//! mapping.

use super::device::DeviceInner;
use super::pnext::PNextChain;
use super::{Device, Error, Result, check};
use crate::raw::bindings::*;
use std::sync::Arc;

/// Parameters for [`DeviceMemory::allocate_with`].
///
/// Use this when a plain `(size, memory_type_index)` allocation isn't
/// enough — typically because you need to chain an extension struct
/// onto the allocation's `pNext`:
///
/// - `VkMemoryAllocateFlagsInfo` for `bufferDeviceAddress` or device
///   masks on a multi-GPU group.
/// - `VkExportMemoryAllocateInfo` / `VkExportMemoryAllocateInfoKHR` for
///   `VK_KHR_external_memory` interop with CUDA, HIP, or DX12.
/// - `VkMemoryDedicatedAllocateInfo` for resources the driver asks you
///   to bind to a dedicated allocation.
///
/// Every extension struct auto-implements
/// [`PNextChainable`](crate::raw::PNextChainable) and has a
/// `::new_pnext()` constructor, so building the chain is:
///
/// ```ignore
/// let mut chain = PNextChain::new();
/// let mut export = VkExportMemoryAllocateInfo::new_pnext();
/// export.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT as u32;
/// chain.push(export);
/// let mem = DeviceMemory::allocate_with(&device, &MemoryAllocateInfo {
///     size: req.size,
///     memory_type_index: mt,
///     pnext: Some(&chain),
/// })?;
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryAllocateInfo<'a> {
    /// Size in bytes. Must be ≥ the `size` reported by
    /// `vkGet*MemoryRequirements` for the resource you intend to bind.
    pub size: u64,
    /// Index into `VkPhysicalDeviceMemoryProperties.memoryTypes` selecting
    /// which heap + property flags the allocation comes from.
    pub memory_type_index: u32,
    /// Optional `pNext` chain to attach to `VkMemoryAllocateInfo`. The
    /// chain is borrowed for the duration of the call and does not need
    /// to outlive the returned [`DeviceMemory`].
    pub pnext: Option<&'a PNextChain>,
    /// Optional `VK_EXT_memory_priority` hint in `[0.0, 1.0]`. `0.5` is
    /// the implicit default the driver uses when no priority is
    /// specified. Higher values make the allocation less likely to be
    /// evicted or demoted to system memory under pressure.
    ///
    /// When `Some`, a `VkMemoryPriorityAllocateInfoEXT` is automatically
    /// prepended to whatever the caller passed in [`pnext`](Self::pnext)
    /// — no manual chain manipulation is needed. Requires
    /// `VK_EXT_memory_priority` to be enabled on the device; otherwise
    /// the driver silently ignores the chained struct.
    pub priority: Option<f32>,
}

/// Strongly-typed wrapper around `VkMemoryPropertyFlags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryPropertyFlags(pub u32);

impl MemoryPropertyFlags {
    pub const DEVICE_LOCAL: Self = Self(0x1);
    pub const HOST_VISIBLE: Self = Self(0x2);
    pub const HOST_COHERENT: Self = Self(0x4);
    pub const HOST_CACHED: Self = Self(0x8);
    pub const LAZILY_ALLOCATED: Self = Self(0x10);

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for MemoryPropertyFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// A safe wrapper around `VkDeviceMemory`.
///
/// Memory is freed automatically on drop. The handle keeps the parent device
/// alive via an `Arc`.
pub struct DeviceMemory {
    pub(crate) handle: VkDeviceMemory,
    pub(crate) size: u64,
    pub(crate) device: Arc<DeviceInner>,
}

impl DeviceMemory {
    /// Allocate device memory of the given size from the given memory type index.
    ///
    /// For allocations that need extension-struct `pNext` chaining (external
    /// memory export, buffer-device-address flags, dedicated allocations, …)
    /// use [`allocate_with`](Self::allocate_with) with a [`MemoryAllocateInfo`]
    /// instead.
    pub fn allocate(device: &Device, size: u64, memory_type_index: u32) -> Result<Self> {
        Self::allocate_with(
            device,
            &MemoryAllocateInfo {
                size,
                memory_type_index,
                pnext: None,
                priority: None,
            },
        )
    }

    /// Allocate device memory with a fully-specified [`MemoryAllocateInfo`],
    /// including an optional `pNext` chain for extension structs.
    pub fn allocate_with(device: &Device, info: &MemoryAllocateInfo<'_>) -> Result<Self> {
        let allocate = device
            .inner
            .dispatch
            .vkAllocateMemory
            .ok_or(Error::MissingFunction("vkAllocateMemory"))?;

        // If `priority` is set, build a local chain that prepends a
        // VkMemoryPriorityAllocateInfoEXT to whatever the caller passed.
        // If only `pnext` is set we can use the caller's chain pointer
        // directly; if neither, the head is null.
        let local_chain: Option<PNextChain> = info.priority.map(|p| {
            let mut chain = PNextChain::new();
            chain.push(VkMemoryPriorityAllocateInfoEXT {
                sType: VkStructureType::STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT,
                pNext: std::ptr::null(),
                priority: p,
            });
            if let Some(user) = info.pnext {
                chain.append(user.clone());
            }
            chain
        });

        let p_next = if let Some(chain) = local_chain.as_ref() {
            chain.head()
        } else {
            info.pnext.map_or(std::ptr::null(), |c| c.head())
        };

        let raw = VkMemoryAllocateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: p_next,
            allocationSize: info.size,
            memoryTypeIndex: info.memory_type_index,
        };

        let mut handle: VkDeviceMemory = 0;
        // Safety: raw is valid for the call, device handle is valid. The
        // optional pNext chain — either `local_chain` or the caller's
        // borrowed one — lives for the duration of this synchronous
        // call.
        check(unsafe { allocate(device.inner.handle, &raw, std::ptr::null(), &mut handle) })?;
        drop(local_chain);

        Ok(Self {
            handle,
            size: info.size,
            device: Arc::clone(&device.inner),
        })
    }

    /// Returns the raw `VkDeviceMemory` handle.
    pub fn raw(&self) -> VkDeviceMemory {
        self.handle
    }

    /// Returns the size of the allocated memory in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Map the entire allocation into host address space and return a
    /// guard that unmaps on drop.
    ///
    /// The memory must have been allocated from a memory type with the
    /// `HOST_VISIBLE` property flag.
    pub fn map(&mut self) -> Result<MappedMemory<'_>> {
        let map = self
            .device
            .dispatch
            .vkMapMemory
            .ok_or(Error::MissingFunction("vkMapMemory"))?;

        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // Safety: handle is valid; we ask for the entire allocation.
        // The pointer is owned by Vulkan; we wrap it in a guard.
        check(unsafe { map(self.device.handle, self.handle, 0, self.size, 0, &mut ptr) })?;

        Ok(MappedMemory {
            ptr,
            size: self.size,
            memory: self,
        })
    }
}

impl Drop for DeviceMemory {
    fn drop(&mut self) {
        if let Some(free) = self.device.dispatch.vkFreeMemory {
            // Safety: handle is valid; we are the sole owner.
            unsafe { free(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}

/// RAII guard for a host-mapped device memory region.
///
/// On drop, the memory is unmapped via `vkUnmapMemory`.
///
/// Use [`as_slice_mut`](Self::as_slice_mut) to get a `&mut [u8]` view of the
/// mapped region, or [`as_ptr`](Self::as_ptr) for a raw pointer.
pub struct MappedMemory<'a> {
    ptr: *mut std::ffi::c_void,
    size: u64,
    memory: &'a mut DeviceMemory,
}

impl<'a> MappedMemory<'a> {
    /// Returns the raw mapped pointer.
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    /// Returns the mapped region as a mutable byte slice.
    ///
    /// The slice is only valid for the lifetime of this `MappedMemory` guard.
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        // Safety: ptr is valid for `size` bytes (Vulkan guarantees this for
        // a successful vkMapMemory of the entire allocation), and the
        // exclusive borrow of `self` ensures no aliasing.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast::<u8>(), self.size as usize) }
    }

    /// Returns the mapped region as an immutable byte slice.
    pub fn as_slice(&self) -> &[u8] {
        // Safety: ptr is valid for `size` bytes; the borrow of `self` ensures
        // no concurrent mutation through this guard.
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.size as usize) }
    }
}

impl<'a> Drop for MappedMemory<'a> {
    fn drop(&mut self) {
        if let Some(unmap) = self.memory.device.dispatch.vkUnmapMemory {
            // Safety: handle is still valid (memory has not been freed),
            // and we hold an exclusive borrow on memory which prevents
            // concurrent map/unmap calls.
            unsafe { unmap(self.memory.device.handle, self.memory.handle) };
        }
    }
}
