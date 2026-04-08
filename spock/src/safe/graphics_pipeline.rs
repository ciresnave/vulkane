//! Safe wrapper around `VkPipeline` for graphics pipelines.
//!
//! Graphics pipelines bundle every piece of fixed-function state plus
//! the vertex / fragment / etc. shader stages into a single immutable
//! object. The Vulkan API for creating one involves passing nine
//! sub-structs to `vkCreateGraphicsPipelines`; this module wraps that
//! with a focused builder.
//!
//! ## Scope
//!
//! For 0.1 the safe wrapper supports the most common shape:
//! - Vertex + fragment shader stages
//! - Optional vertex input bindings + attributes
//! - Triangle list topology (the default; configurable)
//! - One viewport / scissor matching a render pass attachment size
//! - Standard fill / cull / front-face raster state
//! - 1× MSAA
//! - One color attachment with optional alpha blending
//! - Optional depth test/write
//!
//! Geometry / tessellation stages, multiple subpasses, dynamic state,
//! and color-write masks are reachable via [`spock::raw`](crate::raw)
//! when needed.
//!
//! ## Example
//!
//! ```ignore
//! use spock::safe::*;
//!
//! let pipeline = GraphicsPipelineBuilder::new(&pipeline_layout, &render_pass)
//!     .stage(ShaderStage::Vertex, &vert_shader, "main")
//!     .stage(ShaderStage::Fragment, &frag_shader, "main")
//!     .vertex_input(
//!         &[VertexInputBinding { binding: 0, stride: 12 }],
//!         &[VertexInputAttribute {
//!             location: 0,
//!             binding: 0,
//!             format: Format::R32G32B32_SFLOAT,
//!             offset: 0,
//!         }],
//!     )
//!     .viewport_extent(800, 600)
//!     .build(&device)?;
//! ```

use super::descriptor::ShaderStageFlags;
use super::device::DeviceInner;
use super::image::Format;
use super::pipeline::PipelineLayout;
use super::render_pass::RenderPass;
use super::shader::ShaderModule;
use super::{Device, Error, Result, check};
use crate::raw::bindings::*;
use std::ffi::CString;
use std::sync::Arc;

/// Which shader stage to attach. The graphics builder accepts vertex
/// and fragment stages explicitly; tessellation/geometry stages are not
/// in scope for 0.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicsShaderStage {
    Vertex,
    Fragment,
}

impl GraphicsShaderStage {
    fn bit(self) -> u32 {
        match self {
            Self::Vertex => 0x1,    // VK_SHADER_STAGE_VERTEX_BIT
            Self::Fragment => 0x10, // VK_SHADER_STAGE_FRAGMENT_BIT
        }
    }
}

/// One vertex buffer binding declaration.
#[derive(Debug, Clone, Copy)]
pub struct VertexInputBinding {
    pub binding: u32,
    pub stride: u32,
}

/// One vertex attribute (per-vertex shader input) declaration.
#[derive(Debug, Clone, Copy)]
pub struct VertexInputAttribute {
    pub location: u32,
    pub binding: u32,
    pub format: Format,
    pub offset: u32,
}

/// Primitive topology — what kind of geometric shapes the input forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimitiveTopology(pub VkPrimitiveTopology);

impl PrimitiveTopology {
    pub const POINT_LIST: Self = Self(VkPrimitiveTopology::PRIMITIVE_TOPOLOGY_POINT_LIST);
    pub const LINE_LIST: Self = Self(VkPrimitiveTopology::PRIMITIVE_TOPOLOGY_LINE_LIST);
    pub const LINE_STRIP: Self = Self(VkPrimitiveTopology::PRIMITIVE_TOPOLOGY_LINE_STRIP);
    pub const TRIANGLE_LIST: Self = Self(VkPrimitiveTopology::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pub const TRIANGLE_STRIP: Self = Self(VkPrimitiveTopology::PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP);
    pub const TRIANGLE_FAN: Self = Self(VkPrimitiveTopology::PRIMITIVE_TOPOLOGY_TRIANGLE_FAN);
}

/// Polygon fill mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolygonMode(pub VkPolygonMode);

impl PolygonMode {
    pub const FILL: Self = Self(VkPolygonMode::POLYGON_MODE_FILL);
    pub const LINE: Self = Self(VkPolygonMode::POLYGON_MODE_LINE);
    pub const POINT: Self = Self(VkPolygonMode::POLYGON_MODE_POINT);
}

/// Cull mode bit mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CullMode(pub u32);

impl CullMode {
    pub const NONE: Self = Self(0);
    pub const FRONT: Self = Self(0x1);
    pub const BACK: Self = Self(0x2);
    pub const FRONT_AND_BACK: Self = Self(0x3);
}

/// Front-face winding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrontFace(pub VkFrontFace);

impl FrontFace {
    pub const CLOCKWISE: Self = Self(VkFrontFace::FRONT_FACE_CLOCKWISE);
    pub const COUNTER_CLOCKWISE: Self = Self(VkFrontFace::FRONT_FACE_COUNTER_CLOCKWISE);
}

/// Builder for a graphics [`GraphicsPipeline`].
pub struct GraphicsPipelineBuilder<'a> {
    layout: &'a PipelineLayout,
    render_pass: &'a RenderPass,
    subpass: u32,
    vertex_shader: Option<(&'a ShaderModule, &'a str)>,
    fragment_shader: Option<(&'a ShaderModule, &'a str)>,
    vertex_bindings: &'a [VertexInputBinding],
    vertex_attributes: &'a [VertexInputAttribute],
    topology: PrimitiveTopology,
    polygon_mode: PolygonMode,
    cull_mode: CullMode,
    front_face: FrontFace,
    viewport_width: u32,
    viewport_height: u32,
    blend_enable: bool,
    depth_test: bool,
    depth_write: bool,
}

impl<'a> GraphicsPipelineBuilder<'a> {
    /// Start a new builder targeting the given pipeline layout and
    /// render pass. Defaults are: triangle-list topology, fill polygon,
    /// back-face culling, counter-clockwise front face, 1×1 viewport
    /// (override with [`viewport_extent`](Self::viewport_extent)),
    /// no blending, no depth test.
    pub fn new(layout: &'a PipelineLayout, render_pass: &'a RenderPass) -> Self {
        Self {
            layout,
            render_pass,
            subpass: 0,
            vertex_shader: None,
            fragment_shader: None,
            vertex_bindings: &[],
            vertex_attributes: &[],
            topology: PrimitiveTopology::TRIANGLE_LIST,
            polygon_mode: PolygonMode::FILL,
            cull_mode: CullMode::BACK,
            front_face: FrontFace::COUNTER_CLOCKWISE,
            viewport_width: 1,
            viewport_height: 1,
            blend_enable: false,
            depth_test: false,
            depth_write: false,
        }
    }

    /// Attach a shader stage. Vertex and fragment may both be set;
    /// other stages aren't supported in this 0.1 builder.
    pub fn stage(
        mut self,
        stage: GraphicsShaderStage,
        shader: &'a ShaderModule,
        entry_point: &'a str,
    ) -> Self {
        match stage {
            GraphicsShaderStage::Vertex => self.vertex_shader = Some((shader, entry_point)),
            GraphicsShaderStage::Fragment => self.fragment_shader = Some((shader, entry_point)),
        }
        self
    }

    /// Declare vertex input bindings + attributes. By default the
    /// builder produces a no-vertex-input pipeline (suitable for
    /// fullscreen-triangle techniques).
    pub fn vertex_input(
        mut self,
        bindings: &'a [VertexInputBinding],
        attributes: &'a [VertexInputAttribute],
    ) -> Self {
        self.vertex_bindings = bindings;
        self.vertex_attributes = attributes;
        self
    }

    /// Set the viewport / scissor extent. Required: defaults to 1×1.
    pub fn viewport_extent(mut self, width: u32, height: u32) -> Self {
        self.viewport_width = width;
        self.viewport_height = height;
        self
    }

    pub fn topology(mut self, topology: PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    pub fn polygon_mode(mut self, mode: PolygonMode) -> Self {
        self.polygon_mode = mode;
        self
    }

    pub fn cull_mode(mut self, mode: CullMode) -> Self {
        self.cull_mode = mode;
        self
    }

    pub fn front_face(mut self, face: FrontFace) -> Self {
        self.front_face = face;
        self
    }

    pub fn alpha_blending(mut self, enable: bool) -> Self {
        self.blend_enable = enable;
        self
    }

    pub fn depth_test(mut self, test: bool, write: bool) -> Self {
        self.depth_test = test;
        self.depth_write = write;
        self
    }

    /// Compile the pipeline. Consumes the builder.
    pub fn build(self, device: &Device) -> Result<GraphicsPipeline> {
        let create = device
            .inner
            .dispatch
            .vkCreateGraphicsPipelines
            .ok_or(Error::MissingFunction("vkCreateGraphicsPipelines"))?;

        let vert = self
            .vertex_shader
            .ok_or(Error::Vk(VkResult::ERROR_INITIALIZATION_FAILED))?;
        let frag = self
            .fragment_shader
            .ok_or(Error::Vk(VkResult::ERROR_INITIALIZATION_FAILED))?;

        // Keep entry-point CStrings alive for the duration of the call.
        let vert_entry = CString::new(vert.1)?;
        let frag_entry = CString::new(frag.1)?;

        let stages = [
            VkPipelineShaderStageCreateInfo {
                sType: VkStructureType::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: GraphicsShaderStage::Vertex.bit(),
                module: vert.0.handle,
                pName: vert_entry.as_ptr(),
                ..Default::default()
            },
            VkPipelineShaderStageCreateInfo {
                sType: VkStructureType::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: GraphicsShaderStage::Fragment.bit(),
                module: frag.0.handle,
                pName: frag_entry.as_ptr(),
                ..Default::default()
            },
        ];

        let raw_bindings: Vec<VkVertexInputBindingDescription> = self
            .vertex_bindings
            .iter()
            .map(|b| VkVertexInputBindingDescription {
                binding: b.binding,
                stride: b.stride,
                inputRate: VkVertexInputRate::VERTEX_INPUT_RATE_VERTEX,
            })
            .collect();
        let raw_attributes: Vec<VkVertexInputAttributeDescription> = self
            .vertex_attributes
            .iter()
            .map(|a| VkVertexInputAttributeDescription {
                location: a.location,
                binding: a.binding,
                format: a.format.0,
                offset: a.offset,
            })
            .collect();
        let vertex_input = VkPipelineVertexInputStateCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount: raw_bindings.len() as u32,
            pVertexBindingDescriptions: if raw_bindings.is_empty() {
                std::ptr::null()
            } else {
                raw_bindings.as_ptr()
            },
            vertexAttributeDescriptionCount: raw_attributes.len() as u32,
            pVertexAttributeDescriptions: if raw_attributes.is_empty() {
                std::ptr::null()
            } else {
                raw_attributes.as_ptr()
            },
            ..Default::default()
        };

        let input_assembly = VkPipelineInputAssemblyStateCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology: self.topology.0,
            primitiveRestartEnable: 0,
            ..Default::default()
        };

        let viewport = VkViewport {
            x: 0.0,
            y: 0.0,
            width: self.viewport_width as f32,
            height: self.viewport_height as f32,
            minDepth: 0.0,
            maxDepth: 1.0,
        };
        let scissor = VkRect2D {
            offset: VkOffset2D { x: 0, y: 0 },
            extent: VkExtent2D {
                width: self.viewport_width,
                height: self.viewport_height,
            },
        };
        let viewport_state = VkPipelineViewportStateCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount: 1,
            pViewports: &viewport,
            scissorCount: 1,
            pScissors: &scissor,
            ..Default::default()
        };

        let raster_state = VkPipelineRasterizationStateCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depthClampEnable: 0,
            rasterizerDiscardEnable: 0,
            polygonMode: self.polygon_mode.0,
            cullMode: self.cull_mode.0,
            frontFace: self.front_face.0,
            depthBiasEnable: 0,
            depthBiasConstantFactor: 0.0,
            depthBiasClamp: 0.0,
            depthBiasSlopeFactor: 0.0,
            lineWidth: 1.0,
            ..Default::default()
        };

        let multisample_state = VkPipelineMultisampleStateCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            rasterizationSamples: SAMPLE_COUNT_1_BIT,
            sampleShadingEnable: 0,
            minSampleShading: 1.0,
            pSampleMask: std::ptr::null(),
            alphaToCoverageEnable: 0,
            alphaToOneEnable: 0,
            ..Default::default()
        };

        let depth_stencil_state = VkPipelineDepthStencilStateCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depthTestEnable: if self.depth_test { 1 } else { 0 },
            depthWriteEnable: if self.depth_write { 1 } else { 0 },
            depthCompareOp: VkCompareOp::COMPARE_OP_LESS_OR_EQUAL,
            depthBoundsTestEnable: 0,
            stencilTestEnable: 0,
            front: VkStencilOpState::default(),
            back: VkStencilOpState::default(),
            minDepthBounds: 0.0,
            maxDepthBounds: 1.0,
            ..Default::default()
        };

        let color_blend_attachment = VkPipelineColorBlendAttachmentState {
            blendEnable: if self.blend_enable { 1 } else { 0 },
            srcColorBlendFactor: VkBlendFactor::BLEND_FACTOR_SRC_ALPHA,
            dstColorBlendFactor: VkBlendFactor::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            colorBlendOp: VkBlendOp::BLEND_OP_ADD,
            srcAlphaBlendFactor: VkBlendFactor::BLEND_FACTOR_ONE,
            dstAlphaBlendFactor: VkBlendFactor::BLEND_FACTOR_ZERO,
            alphaBlendOp: VkBlendOp::BLEND_OP_ADD,
            // VK_COLOR_COMPONENT_R_BIT | G | B | A = 0xF
            colorWriteMask: 0xF,
        };

        let color_blend_state = VkPipelineColorBlendStateCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logicOpEnable: 0,
            logicOp: VkLogicOp::LOGIC_OP_COPY,
            attachmentCount: 1,
            pAttachments: &color_blend_attachment,
            blendConstants: [0.0; 4],
            ..Default::default()
        };

        let info = VkGraphicsPipelineCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            stageCount: stages.len() as u32,
            pStages: stages.as_ptr(),
            pVertexInputState: &vertex_input,
            pInputAssemblyState: &input_assembly,
            pTessellationState: std::ptr::null(),
            pViewportState: &viewport_state,
            pRasterizationState: &raster_state,
            pMultisampleState: &multisample_state,
            pDepthStencilState: &depth_stencil_state,
            pColorBlendState: &color_blend_state,
            pDynamicState: std::ptr::null(),
            layout: self.layout.handle,
            renderPass: self.render_pass.handle,
            subpass: self.subpass,
            basePipelineHandle: 0,
            basePipelineIndex: -1,
            ..Default::default()
        };

        let mut handle: VkPipeline = 0;
        // Safety: every pointer in `info` is into a local that lives
        // until end of scope. The vert_entry / frag_entry CStrings
        // similarly outlive the call.
        check(unsafe {
            create(
                device.inner.handle,
                0, // pipelineCache
                1,
                &info,
                std::ptr::null(),
                &mut handle,
            )
        })?;

        // Suppress dead-code lint warnings for the locals whose only
        // *purpose* is to keep their address stable across the call.
        let _ = (
            stages,
            raw_bindings,
            raw_attributes,
            vertex_input,
            input_assembly,
            viewport,
            scissor,
            viewport_state,
            raster_state,
            multisample_state,
            depth_stencil_state,
            color_blend_attachment,
            color_blend_state,
            vert_entry,
            frag_entry,
        );

        Ok(GraphicsPipeline {
            handle,
            device: Arc::clone(&device.inner),
        })
    }
}

/// A safe wrapper around a graphics `VkPipeline`.
///
/// Pipelines are destroyed automatically on drop.
pub struct GraphicsPipeline {
    pub(crate) handle: VkPipeline,
    pub(crate) device: Arc<DeviceInner>,
}

impl GraphicsPipeline {
    /// Returns the raw `VkPipeline` handle.
    pub fn raw(&self) -> VkPipeline {
        self.handle
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroyPipeline {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}

// `ShaderStageFlags` is currently used only by the descriptor module,
// but the graphics pipeline references shader stage bits via the
// `GraphicsShaderStage::bit()` enum. Re-export the type so users who
// want to define descriptor sets that span vertex + fragment stages
// don't need a second import.
#[doc(hidden)]
pub fn _shader_stage_flags() -> ShaderStageFlags {
    ShaderStageFlags::ALL
}
