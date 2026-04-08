//! Safe wrappers for descriptor sets, layouts, and pools.
//!
//! In Vulkan, descriptor sets are how shaders access resources (buffers,
//! images, samplers). To bind a buffer to a compute shader you need:
//!
//! 1. A [`DescriptorSetLayout`] describing the bindings the shader expects.
//! 2. A [`DescriptorPool`] from which to allocate descriptor sets.
//! 3. A [`DescriptorSet`] allocated from the pool, with bindings written to
//!    point at actual buffers/images.
//!
//! The set layout becomes part of the [`PipelineLayout`](super::PipelineLayout),
//! which is in turn part of the [`ComputePipeline`](super::ComputePipeline).

use super::device::DeviceInner;
use super::{Buffer, Device, Error, Result, check};
use crate::raw::bindings::*;
use std::sync::Arc;

/// What kind of resource a descriptor binding represents.
///
/// Currently only `STORAGE_BUFFER` and `UNIFORM_BUFFER` are exposed since
/// they're sufficient for compute work. Sampler/image variants will be added
/// when the safe wrapper grows graphics support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorType(pub VkDescriptorType);

impl DescriptorType {
    pub const STORAGE_BUFFER: Self = Self(VkDescriptorType::DESCRIPTOR_TYPE_STORAGE_BUFFER);
    pub const UNIFORM_BUFFER: Self = Self(VkDescriptorType::DESCRIPTOR_TYPE_UNIFORM_BUFFER);
}

/// Which pipeline stages may access a descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShaderStageFlags(pub u32);

impl ShaderStageFlags {
    pub const VERTEX: Self = Self(0x1);
    pub const FRAGMENT: Self = Self(0x10);
    pub const COMPUTE: Self = Self(0x20);
    pub const ALL_GRAPHICS: Self = Self(0x1F);
    pub const ALL: Self = Self(0x7FFFFFFF);

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for ShaderStageFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// One binding in a [`DescriptorSetLayout`] — a `(binding, type, count, stages)` tuple.
#[derive(Debug, Clone, Copy)]
pub struct DescriptorSetLayoutBinding {
    /// The binding number that the shader uses (e.g., `layout(binding = 0)` in GLSL).
    pub binding: u32,
    /// What kind of resource lives at this binding.
    pub descriptor_type: DescriptorType,
    /// Number of descriptors at this binding (e.g., for arrays of buffers).
    pub descriptor_count: u32,
    /// Which shader stages may access this binding.
    pub stage_flags: ShaderStageFlags,
}

/// A safe wrapper around `VkDescriptorSetLayout`.
///
/// Set layouts describe the *shape* of descriptor sets. They are referenced by
/// pipeline layouts and by descriptor set allocations. They are destroyed
/// automatically on drop.
pub struct DescriptorSetLayout {
    pub(crate) handle: VkDescriptorSetLayout,
    pub(crate) device: Arc<DeviceInner>,
}

impl DescriptorSetLayout {
    /// Create a new descriptor set layout from a list of bindings.
    pub fn new(device: &Device, bindings: &[DescriptorSetLayoutBinding]) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreateDescriptorSetLayout
            .ok_or(Error::MissingFunction("vkCreateDescriptorSetLayout"))?;

        let raw_bindings: Vec<VkDescriptorSetLayoutBinding> = bindings
            .iter()
            .map(|b| VkDescriptorSetLayoutBinding {
                binding: b.binding,
                descriptorType: b.descriptor_type.0,
                descriptorCount: b.descriptor_count,
                stageFlags: b.stage_flags.0,
                pImmutableSamplers: std::ptr::null(),
            })
            .collect();

        let info = VkDescriptorSetLayoutCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount: raw_bindings.len() as u32,
            pBindings: raw_bindings.as_ptr(),
            ..Default::default()
        };

        let mut handle: VkDescriptorSetLayout = 0;
        // Safety: info is valid for the call, raw_bindings outlives it.
        check(unsafe { create(device.inner.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
        })
    }

    /// Returns the raw `VkDescriptorSetLayout` handle.
    pub fn raw(&self) -> VkDescriptorSetLayout {
        self.handle
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroyDescriptorSetLayout {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}

/// Pool size for one descriptor type — `(type, count)` pair.
#[derive(Debug, Clone, Copy)]
pub struct DescriptorPoolSize {
    pub descriptor_type: DescriptorType,
    pub descriptor_count: u32,
}

/// A safe wrapper around `VkDescriptorPool`.
///
/// Descriptor pools are arenas from which descriptor sets are allocated.
/// They are destroyed automatically on drop, which also frees all sets
/// allocated from them.
pub struct DescriptorPool {
    pub(crate) handle: VkDescriptorPool,
    pub(crate) device: Arc<DeviceInner>,
}

impl DescriptorPool {
    /// Create a new descriptor pool that can hold `max_sets` descriptor sets,
    /// with the given per-type budget.
    pub fn new(device: &Device, max_sets: u32, sizes: &[DescriptorPoolSize]) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreateDescriptorPool
            .ok_or(Error::MissingFunction("vkCreateDescriptorPool"))?;

        let raw_sizes: Vec<VkDescriptorPoolSize> = sizes
            .iter()
            .map(|s| VkDescriptorPoolSize {
                r#type: s.descriptor_type.0,
                descriptorCount: s.descriptor_count,
            })
            .collect();

        let info = VkDescriptorPoolCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets: max_sets,
            poolSizeCount: raw_sizes.len() as u32,
            pPoolSizes: raw_sizes.as_ptr(),
            ..Default::default()
        };

        let mut handle: VkDescriptorPool = 0;
        // Safety: info and raw_sizes are valid for the call.
        check(unsafe { create(device.inner.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
        })
    }

    /// Returns the raw `VkDescriptorPool` handle.
    pub fn raw(&self) -> VkDescriptorPool {
        self.handle
    }

    /// Allocate one descriptor set from this pool, using the given layout.
    pub fn allocate(&self, layout: &DescriptorSetLayout) -> Result<DescriptorSet> {
        let allocate = self
            .device
            .dispatch
            .vkAllocateDescriptorSets
            .ok_or(Error::MissingFunction("vkAllocateDescriptorSets"))?;

        let info = VkDescriptorSetAllocateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool: self.handle,
            descriptorSetCount: 1,
            pSetLayouts: &layout.handle,
            ..Default::default()
        };

        let mut handle: VkDescriptorSet = 0;
        // Safety: info is valid for the call.
        check(unsafe { allocate(self.device.handle, &info, &mut handle) })?;

        Ok(DescriptorSet {
            handle,
            device: Arc::clone(&self.device),
        })
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroyDescriptorPool {
            // Safety: handle is valid; this also frees all sets allocated
            // from the pool, so we don't need a separate vkFreeDescriptorSets.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}

/// A safe wrapper around `VkDescriptorSet`.
///
/// Descriptor sets are allocated from a [`DescriptorPool`]; their lifetime is
/// tied to the pool. We don't implement `Drop` for `DescriptorSet` because
/// the pool's `Drop` frees all its sets in one operation.
pub struct DescriptorSet {
    pub(crate) handle: VkDescriptorSet,
    pub(crate) device: Arc<DeviceInner>,
}

impl DescriptorSet {
    /// Returns the raw `VkDescriptorSet` handle.
    pub fn raw(&self) -> VkDescriptorSet {
        self.handle
    }

    /// Update one binding in this set to point at a buffer.
    ///
    /// Equivalent to a single `vkUpdateDescriptorSets` call with one
    /// `VkWriteDescriptorSet` of type `STORAGE_BUFFER` or `UNIFORM_BUFFER`.
    pub fn write_buffer(
        &self,
        binding: u32,
        descriptor_type: DescriptorType,
        buffer: &Buffer,
        offset: u64,
        range: u64,
    ) {
        let update = self
            .device
            .dispatch
            .vkUpdateDescriptorSets
            .expect("vkUpdateDescriptorSets is required by Vulkan 1.0");

        let info = VkDescriptorBufferInfo {
            buffer: buffer.handle,
            offset,
            range,
        };

        let write = VkWriteDescriptorSet {
            sType: VkStructureType::STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet: self.handle,
            dstBinding: binding,
            descriptorCount: 1,
            descriptorType: descriptor_type.0,
            pBufferInfo: &info,
            ..Default::default()
        };

        // Safety: handle is valid, write/info live for the duration of the call.
        unsafe { update(self.device.handle, 1, &write, 0, std::ptr::null()) };
    }
}
