//! Safe wrappers for `VkPipelineLayout` and `VkPipeline` (compute only for now).

use super::descriptor::DescriptorSetLayout;
use super::device::DeviceInner;
use super::shader::ShaderModule;
use super::{Device, Error, Result, check};
use crate::raw::bindings::*;
use std::ffi::CString;
use std::sync::Arc;

/// A safe wrapper around `VkPipelineLayout`.
///
/// A pipeline layout describes the descriptor set layouts and push constants
/// that a pipeline expects. It's destroyed automatically on drop.
pub struct PipelineLayout {
    pub(crate) handle: VkPipelineLayout,
    pub(crate) device: Arc<DeviceInner>,
}

impl PipelineLayout {
    /// Create a pipeline layout from one or more descriptor set layouts.
    /// Push constants are not yet exposed.
    pub fn new(device: &Device, set_layouts: &[&DescriptorSetLayout]) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreatePipelineLayout
            .ok_or(Error::MissingFunction("vkCreatePipelineLayout"))?;

        let raw_layouts: Vec<VkDescriptorSetLayout> =
            set_layouts.iter().map(|l| l.handle).collect();

        let info = VkPipelineLayoutCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount: raw_layouts.len() as u32,
            pSetLayouts: raw_layouts.as_ptr(),
            ..Default::default()
        };

        let mut handle: VkPipelineLayout = 0;
        // Safety: info and raw_layouts are valid for the call.
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
    /// Create a compute pipeline from a shader module and pipeline layout.
    ///
    /// `entry_point` is the GLSL/SPIR-V `main` function name (typically `"main"`).
    pub fn new(
        device: &Device,
        layout: &PipelineLayout,
        shader: &ShaderModule,
        entry_point: &str,
    ) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreateComputePipelines
            .ok_or(Error::MissingFunction("vkCreateComputePipelines"))?;

        // Keep the C string alive across the call.
        let entry_c = CString::new(entry_point)?;

        let stage = VkPipelineShaderStageCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: 0x20, // VK_SHADER_STAGE_COMPUTE_BIT
            module: shader.handle,
            pName: entry_c.as_ptr(),
            ..Default::default()
        };

        let info = VkComputePipelineCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage,
            layout: layout.handle,
            ..Default::default()
        };

        let mut handle: VkPipeline = 0;
        // Safety: info, stage, entry_c all live for the duration of the call.
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
