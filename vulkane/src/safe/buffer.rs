//! Safe wrapper for `VkBuffer`.
//!
//! Buffers are linear arrays of bytes on the GPU used for vertex data,
//! index data, uniform data, storage, staging transfers, and more.
//!
//! ## Creating a buffer
//!
//! **One-call (recommended for simple cases):**
//! ```ignore
//! let (buffer, memory) = Buffer::new_bound(
//!     &device, &physical,
//!     BufferCreateInfo { size: 4096, usage: BufferUsage::STORAGE_BUFFER },
//!     MemoryPropertyFlags::DEVICE_LOCAL,
//! )?;
//! ```
//!
//! **Staging upload (host data → device-local buffer):**
//! ```ignore
//! let (buffer, memory) = queue.upload_buffer(
//!     &device, &physical, qf, &my_data, BufferUsage::VERTEX_BUFFER,
//! )?;
//! ```
//!
//! **Sub-allocated (for many buffers sharing large blocks):**
//! ```ignore
//! let (buffer, alloc) = allocator.create_buffer(info, alloc_info)?;
//! ```
//!
//! **Manual (full control):**
//! ```ignore
//! let buffer = Buffer::new(&device, info)?;
//! let req = buffer.memory_requirements();
//! let mt = physical.find_memory_type(req.memory_type_bits, flags)?;
//! let memory = DeviceMemory::allocate(&device, req.size, mt)?;
//! buffer.bind_memory(&memory, 0)?;
//! ```

use super::device::DeviceInner;
use super::{Device, DeviceMemory, Error, Result, check};
use crate::raw::bindings::*;
use std::sync::Arc;

/// Strongly-typed wrapper around `VkBufferUsageFlags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferUsage(pub u32);

impl BufferUsage {
    pub const TRANSFER_SRC: Self = Self(0x1);
    pub const TRANSFER_DST: Self = Self(0x2);
    pub const UNIFORM_TEXEL_BUFFER: Self = Self(0x4);
    pub const STORAGE_TEXEL_BUFFER: Self = Self(0x8);
    pub const UNIFORM_BUFFER: Self = Self(0x10);
    pub const STORAGE_BUFFER: Self = Self(0x20);
    pub const INDEX_BUFFER: Self = Self(0x40);
    pub const VERTEX_BUFFER: Self = Self(0x80);
    pub const INDIRECT_BUFFER: Self = Self(0x100);
    /// `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` — required to call
    /// [`Buffer::device_address`]. Requires Vulkan 1.2 (or
    /// `VK_KHR_buffer_device_address` on 1.0/1.1) and the
    /// `bufferDeviceAddress` device feature.
    pub const SHADER_DEVICE_ADDRESS: Self = Self(0x20000);

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for BufferUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Parameters for [`Buffer::new`].
#[derive(Debug, Clone)]
pub struct BufferCreateInfo {
    /// Size of the buffer in bytes.
    pub size: u64,
    /// How the buffer will be used (transfer, storage, uniform, etc.).
    pub usage: BufferUsage,
}

/// Memory requirements for a buffer.
#[derive(Debug, Clone, Copy)]
pub struct MemoryRequirements {
    pub size: u64,
    pub alignment: u64,
    /// Bitmask of memory type indices that can be used to back this buffer.
    pub memory_type_bits: u32,
}

/// A safe wrapper around `VkBuffer`.
///
/// The buffer is destroyed automatically on drop. The handle keeps the parent
/// device alive via an `Arc`.
///
/// To use a buffer, you must:
/// 1. Create it with [`Buffer::new`].
/// 2. Query its memory requirements via [`memory_requirements`](Self::memory_requirements).
/// 3. Allocate compatible memory via [`DeviceMemory::allocate`].
/// 4. Bind the memory to the buffer via [`bind_memory`](Self::bind_memory).
pub struct Buffer {
    pub(crate) handle: VkBuffer,
    pub(crate) device: Arc<DeviceInner>,
    pub(crate) size: u64,
}

impl Buffer {
    /// Create a new buffer with the given size and usage.
    pub fn new(device: &Device, info: BufferCreateInfo) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreateBuffer
            .ok_or(Error::MissingFunction("vkCreateBuffer"))?;

        let create_info = VkBufferCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size: info.size,
            usage: info.usage.0,
            sharingMode: VkSharingMode::SHARING_MODE_EXCLUSIVE,
            ..Default::default()
        };

        let mut handle: VkBuffer = 0;
        // Safety: create_info is valid for the call, device is valid.
        check(unsafe {
            create(
                device.inner.handle,
                &create_info,
                std::ptr::null(),
                &mut handle,
            )
        })?;

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
            size: info.size,
        })
    }

    /// Create a buffer, allocate memory with the given property flags,
    /// and bind them together in one call. Returns the buffer and its
    /// backing [`DeviceMemory`].
    ///
    /// This is the middle-ground convenience between the manual 5-step
    /// pattern and the full [`Allocator`](super::Allocator). Use it
    /// for one-off allocations where you don't need sub-allocation
    /// pooling.
    pub fn new_bound(
        device: &Device,
        physical: &super::PhysicalDevice,
        info: BufferCreateInfo,
        memory_flags: super::MemoryPropertyFlags,
    ) -> Result<(Buffer, DeviceMemory)> {
        let buffer = Buffer::new(device, info)?;
        let req = buffer.memory_requirements();
        let type_index = physical
            .find_memory_type(req.memory_type_bits, memory_flags)
            .ok_or(Error::InvalidArgument(
                "no compatible memory type for the requested property flags",
            ))?;
        let memory = DeviceMemory::allocate(device, req.size, type_index)?;
        buffer.bind_memory(&memory, 0)?;
        Ok((buffer, memory))
    }

    /// Returns the raw `VkBuffer` handle.
    pub fn raw(&self) -> VkBuffer {
        self.handle
    }

    /// Returns the size of the buffer in bytes (as requested at creation time).
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Query the memory requirements for this buffer.
    pub fn memory_requirements(&self) -> MemoryRequirements {
        let get = self
            .device
            .dispatch
            .vkGetBufferMemoryRequirements
            .expect("vkGetBufferMemoryRequirements is required by Vulkan 1.0");

        // Safety: device and buffer handles are valid; output struct will
        // be fully overwritten by the driver.
        let mut req: VkMemoryRequirements = unsafe { std::mem::zeroed() };
        unsafe { get(self.device.handle, self.handle, &mut req) };
        MemoryRequirements {
            size: req.size,
            alignment: req.alignment,
            memory_type_bits: req.memoryTypeBits,
        }
    }

    /// Bind a previously allocated [`DeviceMemory`] to this buffer at the
    /// given offset.
    pub fn bind_memory(&self, memory: &DeviceMemory, offset: u64) -> Result<()> {
        let bind = self
            .device
            .dispatch
            .vkBindBufferMemory
            .ok_or(Error::MissingFunction("vkBindBufferMemory"))?;
        // Safety: handles are valid, offset is in bounds (caller is responsible).
        check(unsafe { bind(self.device.handle, self.handle, memory.handle, offset) })
    }

    /// Return the GPU virtual address of this buffer (its base byte
    /// address as the GPU would see it via `OpLoadDeviceAddress` /
    /// `buffer_reference` in shaders).
    ///
    /// Requires:
    /// - The buffer to have been created with
    ///   [`BufferUsage::SHADER_DEVICE_ADDRESS`].
    /// - The device to have been created with the `bufferDeviceAddress`
    ///   feature enabled.
    /// - Vulkan 1.2 (the function is core in 1.2) or
    ///   `VK_KHR_buffer_device_address` enabled on 1.0/1.1.
    ///
    /// Returns an error wrapping `MissingFunction` when the function
    /// pointer isn't loaded — most commonly because the feature wasn't
    /// enabled at device creation time.
    pub fn device_address(&self) -> Result<u64> {
        let get = self
            .device
            .dispatch
            .vkGetBufferDeviceAddress
            .ok_or(Error::MissingFunction("vkGetBufferDeviceAddress"))?;
        let info = VkBufferDeviceAddressInfo {
            sType: VkStructureType::STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            buffer: self.handle,
            ..Default::default()
        };
        // Safety: info is valid for the call.
        let addr = unsafe { get(self.device.handle, &info) };
        Ok(addr)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroyBuffer {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}
