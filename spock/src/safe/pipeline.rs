//! Safe wrappers for `VkPipelineLayout` and `VkPipeline` (compute only for now).

use super::descriptor::{DescriptorSetLayout, ShaderStageFlags};
use super::device::DeviceInner;
use super::shader::ShaderModule;
use super::{Device, Error, Result, check};
use crate::raw::bindings::*;
use std::ffi::CString;
use std::sync::Arc;

/// One push constant range — a `(stage_flags, offset, size)` tuple.
///
/// Push constants are small chunks of data (typically 16-128 bytes — the
/// guaranteed minimum is 128 bytes per the Vulkan spec) that the host pushes
/// directly into the command buffer for a single draw or dispatch. They are
/// the cheapest way to pass per-invocation parameters to a shader: there's
/// no descriptor set, no buffer, no memory binding — just an inline copy in
/// the command buffer.
///
/// In GLSL, push constants are declared with:
///
/// ```glsl
/// layout(push_constant) uniform PushConsts {
///     uint count;
///     float scale;
/// } pc;
/// ```
///
/// `offset` and `size` are in bytes and must be multiples of 4. The Vulkan
/// spec requires that ranges declared in a single pipeline layout do not
/// overlap.
#[derive(Debug, Clone, Copy)]
pub struct PushConstantRange {
    /// Which shader stages will read this range.
    pub stage_flags: ShaderStageFlags,
    /// Byte offset of this range from the start of push constant memory.
    pub offset: u32,
    /// Size of this range in bytes.
    pub size: u32,
}

/// A safe wrapper around `VkPipelineLayout`.
///
/// A pipeline layout describes the descriptor set layouts and push constant
/// ranges that a pipeline expects. It's destroyed automatically on drop.
pub struct PipelineLayout {
    pub(crate) handle: VkPipelineLayout,
    pub(crate) device: Arc<DeviceInner>,
}

impl PipelineLayout {
    /// Create a pipeline layout from descriptor set layouts only (no push
    /// constants).
    ///
    /// This is a convenience for the common case. Use
    /// [`with_push_constants`](Self::with_push_constants) when you also need
    /// to declare push constant ranges.
    pub fn new(device: &Device, set_layouts: &[&DescriptorSetLayout]) -> Result<Self> {
        Self::with_push_constants(device, set_layouts, &[])
    }

    /// Create a pipeline layout from descriptor set layouts and push
    /// constant ranges.
    ///
    /// Either slice may be empty. Push constant ranges must be non-overlapping
    /// per the Vulkan spec.
    pub fn with_push_constants(
        device: &Device,
        set_layouts: &[&DescriptorSetLayout],
        push_constant_ranges: &[PushConstantRange],
    ) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreatePipelineLayout
            .ok_or(Error::MissingFunction("vkCreatePipelineLayout"))?;

        let raw_layouts: Vec<VkDescriptorSetLayout> =
            set_layouts.iter().map(|l| l.handle).collect();

        let raw_pcrs: Vec<VkPushConstantRange> = push_constant_ranges
            .iter()
            .map(|r| VkPushConstantRange {
                stageFlags: r.stage_flags.0,
                offset: r.offset,
                size: r.size,
            })
            .collect();

        let info = VkPipelineLayoutCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount: raw_layouts.len() as u32,
            pSetLayouts: raw_layouts.as_ptr(),
            pushConstantRangeCount: raw_pcrs.len() as u32,
            pPushConstantRanges: raw_pcrs.as_ptr(),
            ..Default::default()
        };

        let mut handle: VkPipelineLayout = 0;
        // Safety: info, raw_layouts, and raw_pcrs are all valid for the call.
        check(unsafe { create(device.inner.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
        })
    }

    /// Returns the raw `VkPipelineLayout` handle.
    pub fn raw(&self) -> VkPipelineLayout {
        self.handle
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroyPipelineLayout {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}

/// A typed builder for SPIR-V specialization constants.
///
/// Specialization constants are values baked into the SPIR-V at pipeline
/// creation time. The shader declares them with `layout(constant_id = N)
/// const TYPE NAME = DEFAULT;` and the host can override the default with a
/// value when building the pipeline. This is the canonical way to set
/// workgroup sizes, unroll factors, and dtype switches without recompiling
/// the shader.
///
/// # Example
///
/// ```ignore
/// let specs = SpecializationConstants::new()
///     .add_u32(0, 64)   // local_size_x
///     .add_u32(1, 4)    // unroll factor
///     .add_f32(2, 1.5); // gain
/// let pipeline = ComputePipeline::with_specialization(
///     &device, &layout, &shader, "main", &specs,
/// )?;
/// ```
#[derive(Default, Clone)]
pub struct SpecializationConstants {
    entries: Vec<VkSpecializationMapEntry>,
    data: Vec<u8>,
}

impl SpecializationConstants {
    /// Create an empty specialization constants set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a `u32` specialization constant with the given `constant_id`.
    pub fn add_u32(mut self, constant_id: u32, value: u32) -> Self {
        self.push_bytes(constant_id, &value.to_ne_bytes());
        self
    }

    /// Add an `i32` specialization constant with the given `constant_id`.
    pub fn add_i32(mut self, constant_id: u32, value: i32) -> Self {
        self.push_bytes(constant_id, &value.to_ne_bytes());
        self
    }

    /// Add an `f32` specialization constant with the given `constant_id`.
    pub fn add_f32(mut self, constant_id: u32, value: f32) -> Self {
        self.push_bytes(constant_id, &value.to_ne_bytes());
        self
    }

    /// Add a `bool` specialization constant. SPIR-V represents these as
    /// 32-bit values: `0` for false, non-zero for true.
    pub fn add_bool(mut self, constant_id: u32, value: bool) -> Self {
        let v: u32 = if value { 1 } else { 0 };
        self.push_bytes(constant_id, &v.to_ne_bytes());
        self
    }

    /// Returns true if no constants have been added.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the number of constants currently in this set.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    fn push_bytes(&mut self, constant_id: u32, bytes: &[u8]) {
        let offset = self.data.len() as u32;
        self.entries.push(VkSpecializationMapEntry {
            constantID: constant_id,
            offset,
            size: bytes.len(),
        });
        self.data.extend_from_slice(bytes);
    }

    /// Build a raw `VkSpecializationInfo` that borrows from this struct.
    /// The returned struct is only valid for the lifetime of `self`.
    pub(crate) fn as_raw(&self) -> VkSpecializationInfo {
        VkSpecializationInfo {
            mapEntryCount: self.entries.len() as u32,
            pMapEntries: self.entries.as_ptr(),
            dataSize: self.data.len(),
            pData: self.data.as_ptr() as *const _,
        }
    }
}

/// A safe wrapper around a compute `VkPipeline`.
///
/// Compute pipelines bundle a single compute shader stage with a pipeline
/// layout. To dispatch work, bind the pipeline and a compatible descriptor
/// set on a command buffer and call `dispatch`.
///
/// Pipelines are destroyed automatically on drop.
pub struct ComputePipeline {
    pub(crate) handle: VkPipeline,
    pub(crate) device: Arc<DeviceInner>,
}

impl ComputePipeline {
    /// Create a compute pipeline from a shader module and pipeline layout
    /// with no specialization constants.
    ///
    /// `entry_point` is the GLSL/SPIR-V `main` function name (typically `"main"`).
    pub fn new(
        device: &Device,
        layout: &PipelineLayout,
        shader: &ShaderModule,
        entry_point: &str,
    ) -> Result<Self> {
        Self::with_specialization(
            device,
            layout,
            shader,
            entry_point,
            &SpecializationConstants::new(),
        )
    }

    /// Create a compute pipeline with specialization constants baked in.
    ///
    /// The constants must match the IDs declared in the shader's SPIR-V via
    /// `OpSpecConstant*`. Mismatched IDs are silently ignored by the driver.
    pub fn with_specialization(
        device: &Device,
        layout: &PipelineLayout,
        shader: &ShaderModule,
        entry_point: &str,
        specialization: &SpecializationConstants,
    ) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreateComputePipelines
            .ok_or(Error::MissingFunction("vkCreateComputePipelines"))?;

        // Keep the C string and the spec info alive across the call.
        let entry_c = CString::new(entry_point)?;

        // Build the spec info on the stack so its pointers stay valid.
        let spec_raw = specialization.as_raw();
        let p_spec = if specialization.is_empty() {
            std::ptr::null()
        } else {
            &spec_raw as *const _
        };

        let stage = VkPipelineShaderStageCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: 0x20, // VK_SHADER_STAGE_COMPUTE_BIT
            module: shader.handle,
            pName: entry_c.as_ptr(),
            pSpecializationInfo: p_spec,
            ..Default::default()
        };

        let info = VkComputePipelineCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage,
            layout: layout.handle,
            ..Default::default()
        };

        let mut handle: VkPipeline = 0;
        // Safety: info, stage, entry_c, spec_raw, and the data inside
        // `specialization` all live for the duration of this call.
        check(unsafe {
            create(
                device.inner.handle,
                0, // pipelineCache: VK_NULL_HANDLE
                1,
                &info,
                std::ptr::null(),
                &mut handle,
            )
        })?;

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
        })
    }

    /// Returns the raw `VkPipeline` handle.
    pub fn raw(&self) -> VkPipeline {
        self.handle
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroyPipeline {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}
