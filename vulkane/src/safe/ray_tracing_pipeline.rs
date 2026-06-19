//! Safe wrappers for `VK_KHR_ray_tracing_pipeline`.
//!
//! A ray-tracing pipeline pairs five shader stages — ray-generation
//! (rgen), any-hit (rahit), closest-hit (rchit), miss (rmiss), and
//! intersection (rint) — organised into *shader groups*. Each group is
//! a callable unit the BVH traversal engine invokes when it needs to
//! generate a ray, resolve a hit, miss the scene, or intersect a
//! procedural primitive.
//!
//! Unlike compute pipelines, ray-tracing pipelines are not dispatched
//! by binding + calling a fixed shader. The traversal hardware calls
//! shader groups by *index* into a **shader binding table** (SBT) — a
//! GPU-visible buffer laid out with one entry per group. The tracer is
//! launched with [`CommandBufferRecording::trace_rays`], which takes
//! four "strided regions" of the SBT: rgen, miss, hit, and callable.
//!
//! ## Minimal build flow
//!
//! 1. Create shader modules for each stage (rgen / rmiss / rchit /
//!    …). Use the generic [`ShaderModule`](super::ShaderModule) —
//!    ray-tracing SPIR-V is loaded the same way as any other SPIR-V.
//! 2. Collect them into a `pStages` array (shared with all shader
//!    groups by index).
//! 3. Describe the shader *groups* — [`ShaderGroup::General`] for
//!    rgen/rmiss/callable, [`ShaderGroup::TrianglesHit`] pairing a
//!    rchit (and optional rahit), [`ShaderGroup::ProceduralHit`]
//!    adding a rint for AABB primitives.
//! 4. [`RayTracingPipeline::new`] — create the pipeline object.
//! 5. Query `VkPhysicalDeviceRayTracingPipelinePropertiesKHR` for
//!    `shaderGroupHandleSize` + `shaderGroupHandleAlignment` +
//!    `shaderGroupBaseAlignment` — see
//!    [`PhysicalDevice::ray_tracing_pipeline_properties`](super::PhysicalDevice::ray_tracing_pipeline_properties).
//! 6. [`RayTracingPipeline::get_shader_group_handles`] — fetch the
//!    opaque per-group handles.
//! 7. Allocate a GPU-visible buffer, lay out one handle per group at
//!    the required alignment, upload.
//! 8. At dispatch time, build four
//!    [`ShaderBindingRegion`](ShaderBindingRegion)s pointing at the
//!    rgen / miss / hit / callable spans of that buffer and call
//!    [`CommandBufferRecording::trace_rays`].
//!
//! For inline ray queries from compute shaders (no rtpipeline, just
//! `rayQueryEXT`), you skip this entire module — enable the `rayQuery`
//! feature on the device and use acceleration structures directly from
//! your compute pipeline.

use super::device::DeviceInner;
use super::pipeline::PipelineLayout;
use super::shader::ShaderModule;
use super::{Device, Error, Result, check};
use crate::raw::bindings::*;
use std::ffi::CString;
use std::sync::Arc;

/// A shader-group in a ray-tracing pipeline.
///
/// Indices refer to entries in the `stages` slice passed to
/// [`RayTracingPipeline::new`]. Use `u32::MAX` (`VK_SHADER_UNUSED_KHR`)
/// where a slot is unused — the helpers take `Option<u32>` and
/// translate.
#[derive(Debug, Clone, Copy)]
pub enum ShaderGroup {
    /// A single general-purpose shader: rgen, rmiss, or callable.
    General {
        /// Index into the stages array.
        shader: u32,
    },
    /// A triangle-hit group: closest-hit plus optional any-hit.
    /// Invoked when a ray hits the triangle geometry in a BLAS.
    TrianglesHit {
        closest_hit: Option<u32>,
        any_hit: Option<u32>,
    },
    /// A procedural-hit group: closest-hit plus an intersection
    /// shader (required) plus optional any-hit. Invoked when a ray
    /// bounding-box-hits an AABB primitive — the intersection shader
    /// decides whether the hit actually lies inside the primitive.
    ProceduralHit {
        closest_hit: Option<u32>,
        any_hit: Option<u32>,
        intersection: u32,
    },
}

impl ShaderGroup {
    fn to_raw(self) -> VkRayTracingShaderGroupCreateInfoKHR {
        const UNUSED: u32 = !0u32; // VK_SHADER_UNUSED_KHR
        let (r#type, general, closest, any, intersection) = match self {
            Self::General { shader } => (
                VkRayTracingShaderGroupTypeKHR::RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                shader,
                UNUSED,
                UNUSED,
                UNUSED,
            ),
            Self::TrianglesHit {
                closest_hit,
                any_hit,
            } => (
                VkRayTracingShaderGroupTypeKHR::RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                UNUSED,
                closest_hit.unwrap_or(UNUSED),
                any_hit.unwrap_or(UNUSED),
                UNUSED,
            ),
            Self::ProceduralHit {
                closest_hit,
                any_hit,
                intersection,
            } => (
                VkRayTracingShaderGroupTypeKHR::RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR,
                UNUSED,
                closest_hit.unwrap_or(UNUSED),
                any_hit.unwrap_or(UNUSED),
                intersection,
            ),
        };
        VkRayTracingShaderGroupCreateInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            pNext: std::ptr::null(),
            r#type,
            generalShader: general,
            closestHitShader: closest,
            anyHitShader: any,
            intersectionShader: intersection,
            pShaderGroupCaptureReplayHandle: std::ptr::null(),
        }
    }
}

/// Which Vulkan shader stage a module plays in a ray-tracing pipeline.
/// Passed per-stage in [`RayTracingStage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RayTracingShaderStage {
    Raygen,
    Miss,
    ClosestHit,
    AnyHit,
    Intersection,
    Callable,
}

impl RayTracingShaderStage {
    #[inline]
    fn to_bit(self) -> u32 {
        match self {
            Self::Raygen => 0x0000_0100,
            Self::AnyHit => 0x0000_0200,
            Self::ClosestHit => 0x0000_0400,
            Self::Miss => 0x0000_0800,
            Self::Intersection => 0x0000_1000,
            Self::Callable => 0x0000_2000,
        }
    }
}

/// One shader stage in a ray-tracing pipeline's `stages` array.
///
/// Shader groups reference stages by their index in this array.
pub struct RayTracingStage<'a> {
    pub stage: RayTracingShaderStage,
    pub module: &'a ShaderModule,
    pub entry_point: &'a str,
}

/// A single strided region of the shader-binding table.
///
/// Four of these are passed to [`CommandBufferRecording::trace_rays`](crate::safe::CommandBufferRecording::trace_rays) —
/// one each for rgen / miss / hit / callable. Each region points at a
/// contiguous run of group handles inside the SBT buffer with a
/// per-entry `stride` (which must be ≥ `shaderGroupHandleSize` and a
/// multiple of `shaderGroupHandleAlignment`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ShaderBindingRegion {
    /// GPU virtual address of the first entry, or `0` for an empty /
    /// unused region.
    pub address: u64,
    /// Bytes between successive entries.
    pub stride: u64,
    /// Total size in bytes.
    pub size: u64,
}

impl ShaderBindingRegion {
    pub(crate) fn to_raw(self) -> VkStridedDeviceAddressRegionKHR {
        VkStridedDeviceAddressRegionKHR {
            deviceAddress: self.address,
            stride: self.stride,
            size: self.size,
        }
    }
}

/// Safe wrapper for a `VK_KHR_ray_tracing_pipeline` pipeline.
///
/// Destroyed automatically on drop.
pub struct RayTracingPipeline {
    pub(crate) handle: VkPipeline,
    pub(crate) device: Arc<DeviceInner>,
    pub(crate) group_count: u32,
}

impl RayTracingPipeline {
    /// Create a ray-tracing pipeline.
    ///
    /// `stages` defines the shader modules available to the pipeline;
    /// `groups` references those stages by index to form rgen / miss /
    /// hit / callable groups. `max_recursion_depth` bounds the recursive
    /// `traceRayEXT` depth; must be ≤
    /// `VkPhysicalDeviceRayTracingPipelinePropertiesKHR.maxRayRecursionDepth`.
    pub fn new(
        device: &Device,
        layout: &PipelineLayout,
        stages: &[RayTracingStage<'_>],
        groups: &[ShaderGroup],
        max_recursion_depth: u32,
    ) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreateRayTracingPipelinesKHR
            .ok_or(Error::MissingFunction("vkCreateRayTracingPipelinesKHR"))?;

        // Owning storage for entry-point CStrings and raw stage structs.
        let entry_cstrs: Vec<CString> = stages
            .iter()
            .map(|s| CString::new(s.entry_point))
            .collect::<std::result::Result<_, _>>()?;
        let raw_stages: Vec<VkPipelineShaderStageCreateInfo> = stages
            .iter()
            .zip(entry_cstrs.iter())
            .map(|(s, name)| VkPipelineShaderStageCreateInfo {
                sType: VkStructureType::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext: std::ptr::null(),
                flags: 0,
                stage: s.stage.to_bit(),
                module: s.module.raw(),
                pName: name.as_ptr(),
                pSpecializationInfo: std::ptr::null(),
            })
            .collect();
        let raw_groups: Vec<VkRayTracingShaderGroupCreateInfoKHR> =
            groups.iter().map(|g| g.to_raw()).collect();

        let info = VkRayTracingPipelineCreateInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
            pNext: std::ptr::null(),
            flags: 0,
            stageCount: raw_stages.len() as u32,
            pStages: if raw_stages.is_empty() {
                std::ptr::null()
            } else {
                raw_stages.as_ptr()
            },
            groupCount: raw_groups.len() as u32,
            pGroups: if raw_groups.is_empty() {
                std::ptr::null()
            } else {
                raw_groups.as_ptr()
            },
            maxPipelineRayRecursionDepth: max_recursion_depth,
            pLibraryInfo: std::ptr::null(),
            pLibraryInterface: std::ptr::null(),
            pDynamicState: std::ptr::null(),
            layout: layout.raw(),
            basePipelineHandle: 0,
            basePipelineIndex: -1,
        };

        let mut handle: VkPipeline = 0;
        // Safety: info, raw_stages, raw_groups, entry_cstrs all live
        // for the synchronous call.
        check(unsafe {
            create(
                device.inner.handle,
                0, // deferredOperation — no deferred build
                0, // pipelineCache — none
                1,
                &info,
                std::ptr::null(),
                &mut handle,
            )
        })?;

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
            group_count: raw_groups.len() as u32,
        })
    }

    /// Raw `VkPipeline` handle.
    pub fn raw(&self) -> VkPipeline {
        self.handle
    }

    /// Number of shader groups this pipeline was built with.
    pub fn group_count(&self) -> u32 {
        self.group_count
    }

    /// Fetch the opaque per-group handles into `dst`.
    ///
    /// `dst.len()` must equal
    /// `group_count * shaderGroupHandleSize`. Use
    /// [`PhysicalDevice::ray_tracing_pipeline_properties`](super::PhysicalDevice::ray_tracing_pipeline_properties)
    /// to query the handle size at runtime.
    ///
    /// The copied handles are the bytes you upload into the SBT
    /// buffer — one per group, each at `shaderGroupHandleAlignment`.
    pub fn get_shader_group_handles(
        &self,
        first_group: u32,
        group_count: u32,
        dst: &mut [u8],
    ) -> Result<()> {
        let f = self
            .device
            .dispatch
            .vkGetRayTracingShaderGroupHandlesKHR
            .ok_or(Error::MissingFunction(
                "vkGetRayTracingShaderGroupHandlesKHR",
            ))?;
        // Safety: dst buffer is valid for `len` bytes; handle is valid.
        check(unsafe {
            f(
                self.device.handle,
                self.handle,
                first_group,
                group_count,
                dst.len(),
                dst.as_mut_ptr() as *mut std::ffi::c_void,
            )
        })
    }
}

impl Drop for RayTracingPipeline {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroyPipeline {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}

/// Safe view of
/// [`VkPhysicalDeviceRayTracingPipelinePropertiesKHR`](crate::raw::bindings::VkPhysicalDeviceRayTracingPipelinePropertiesKHR).
#[derive(Debug, Clone, Copy, Default)]
pub struct RayTracingPipelineProperties {
    /// Size of one shader-group handle in bytes. Every SBT entry
    /// must start with a handle of exactly this size.
    pub shader_group_handle_size: u32,
    /// Maximum recursion depth allowed by the implementation.
    pub max_ray_recursion_depth: u32,
    /// Maximum stride between SBT entries.
    pub max_shader_group_stride: u32,
    /// Required alignment of the *start* of each SBT region's data
    /// (rgen / miss / hit / callable), typically 64 bytes.
    pub shader_group_base_alignment: u32,
    /// Required alignment of each individual group handle within a
    /// region, typically 32 bytes.
    pub shader_group_handle_alignment: u32,
    /// Maximum total invocations per `vkCmdTraceRaysKHR` (`width *
    /// height * depth`).
    pub max_ray_dispatch_invocation_count: u32,
    /// Maximum size of per-hit payload (`hitAttributeEXT`).
    pub max_ray_hit_attribute_size: u32,
}
