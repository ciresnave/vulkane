//! Safe wrapper for `VK_KHR_acceleration_structure`.
//!
//! An *acceleration structure* (AS) is a BVH-shaped spatial index built
//! by the driver over geometry data in a `VkBuffer`. Once built, GPU
//! shaders can traverse the AS with hardware-accelerated ray queries —
//! either through the classic ray-tracing pipeline (`rgen` / `rchit` /
//! `rmiss` shaders + shader binding tables) or through inline
//! `rayQueryEXT` calls in ordinary compute / graphics shaders.
//!
//! ## Two-level layout
//!
//! Applications typically build a **two-level** structure:
//!
//! - **BLAS** (bottom-level) — references a single resource's geometry,
//!   which is one of:
//!   - triangle soup (vertex + optional index buffers)
//!   - axis-aligned bounding boxes (for arbitrary implicit primitives)
//!   - *not* instances — only a TLAS can reference instances.
//! - **TLAS** (top-level) — instances pointing at BLASes, each with its
//!   own transform matrix. TLAS input geometry is always "instances".
//!
//! For ray tracing to accelerate the work it's intended to accelerate,
//! geometry you update frequently belongs in the TLAS (instance
//! transforms are cheap to re-emit) while geometry you rarely touch
//! belongs in the BLAS (rebuilding triangle soup is expensive).
//!
//! ## Build flow
//!
//! 1. Pick a [`AccelerationStructureType`] (Bottom / Top / Generic).
//! 2. Describe your geometry via [`AccelerationStructureGeometry`].
//! 3. Ask the driver how big the AS and scratch buffers need to be:
//!    [`Device::acceleration_structure_build_sizes`].
//! 4. Allocate a `VkBuffer` with `ACCELERATION_STRUCTURE_STORAGE_KHR`
//!    usage at ≥ `acceleration_structure_size` bytes.
//! 5. Allocate a separate scratch `VkBuffer` with `STORAGE_BUFFER +
//!    SHADER_DEVICE_ADDRESS` usage at ≥ `build_scratch_size` bytes.
//! 6. Create the AS handle: [`AccelerationStructure::new`].
//! 7. Record the build:
//!    [`CommandBufferRecording::build_acceleration_structure`].
//! 8. Barrier between the build command and any shader that reads the
//!    AS (`VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR` in the
//!    destination mask).
//!
//! The device address returned by
//! [`AccelerationStructure::device_address`] is what shaders bind to
//! via their `accelerationStructureEXT` descriptor slot or
//! `rayQueryEXT` initializer.

use super::device::DeviceInner;
use super::{Buffer, Device, Error, Result};
use crate::raw::bindings::*;
use crate::safe::auto::AccelerationStructureKHR;
use std::sync::Arc;

/// Whether the AS is top- or bottom-level. See module-level docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelerationStructureType {
    /// BLAS — holds geometry (triangles or AABBs).
    BottomLevel,
    /// TLAS — holds instance pointers into BLASes.
    TopLevel,
    /// Type deferred until build time. Rarely the right choice — pick a
    /// concrete type when you know it.
    Generic,
}

impl AccelerationStructureType {
    #[inline]
    pub(crate) fn to_raw(self) -> VkAccelerationStructureTypeKHR {
        match self {
            Self::BottomLevel => {
                VkAccelerationStructureTypeKHR::ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
            }
            Self::TopLevel => {
                VkAccelerationStructureTypeKHR::ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
            }
            Self::Generic => {
                VkAccelerationStructureTypeKHR::ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR
            }
        }
    }
}

/// Where the build will run.
///
/// - [`Device`](Self::Device) — build recorded into a command buffer
///   and executed by the GPU. The common choice.
/// - [`Host`](Self::Host) — build runs on the CPU via
///   `vkBuildAccelerationStructuresKHR`. Useful for offline asset
///   preparation and for devices where GPU builds are slow.
/// - [`HostOrDevice`](Self::HostOrDevice) — size query returns a bound
///   that works for either; pick one at build time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelerationStructureBuildType {
    Host,
    Device,
    HostOrDevice,
}

impl AccelerationStructureBuildType {
    #[inline]
    pub(crate) fn to_raw(self) -> VkAccelerationStructureBuildTypeKHR {
        match self {
            Self::Host => VkAccelerationStructureBuildTypeKHR::ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_KHR,
            Self::Device => VkAccelerationStructureBuildTypeKHR::ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            Self::HostOrDevice => VkAccelerationStructureBuildTypeKHR::ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_OR_DEVICE_KHR,
        }
    }
}

/// Whether this build creates a fresh AS or updates an existing one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelerationStructureBuildMode {
    Build,
    Update,
}

impl AccelerationStructureBuildMode {
    #[inline]
    pub(crate) fn to_raw(self) -> VkBuildAccelerationStructureModeKHR {
        match self {
            Self::Build => {
                VkBuildAccelerationStructureModeKHR::BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
            }
            Self::Update => {
                VkBuildAccelerationStructureModeKHR::BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
            }
        }
    }
}

/// Hints to the driver about the intended build trade-offs. Mirrors
/// `VkBuildAccelerationStructureFlagsKHR`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AccelerationStructureBuildFlags(pub u32);

impl AccelerationStructureBuildFlags {
    /// `ALLOW_UPDATE_BIT` — AS can be used as `src` in a subsequent
    /// `Update` build. Required when you plan to call update rebuilds.
    pub const ALLOW_UPDATE: Self = Self(0x1);
    /// `ALLOW_COMPACTION_BIT` — AS can be compacted via `vkCmdCopyAccelerationStructureKHR`.
    pub const ALLOW_COMPACTION: Self = Self(0x2);
    /// `PREFER_FAST_TRACE_BIT` — optimise the build for trace
    /// throughput; good default for static scenes.
    pub const PREFER_FAST_TRACE: Self = Self(0x4);
    /// `PREFER_FAST_BUILD_BIT` — optimise for build throughput; good
    /// default for dynamic scenes that rebuild every frame.
    pub const PREFER_FAST_BUILD: Self = Self(0x8);
    /// `LOW_MEMORY_BIT` — optimise for smaller AS + scratch buffers.
    pub const LOW_MEMORY: Self = Self(0x10);
}

impl std::ops::BitOr for AccelerationStructureBuildFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// One geometry input for a BLAS or TLAS build.
///
/// Hides `VkAccelerationStructureGeometryKHR`'s tag/union structure
/// behind a Rust-idiomatic enum. The "address" fields are raw GPU
/// virtual addresses (typically from
/// [`Buffer::device_address`](super::Buffer::device_address)) for
/// device builds, or host pointers for host builds.
#[derive(Debug, Clone, Copy)]
pub enum AccelerationStructureGeometry {
    /// Triangle soup — for BLAS. `vertex_data_address` points at an
    /// array of vertices laid out per `vertex_format` every
    /// `vertex_stride` bytes. `max_vertex` is the largest vertex index
    /// referenced (used for validation / size estimation).
    Triangles {
        vertex_format: VkFormat,
        vertex_data_address: u64,
        vertex_stride: u64,
        max_vertex: u32,
        index_type: VkIndexType,
        /// Device address of the index array; `0` for non-indexed
        /// (sequential) triangles.
        index_data_address: u64,
        /// Device address of a `VkTransformMatrixKHR` applied to all
        /// vertices; `0` for identity.
        transform_data_address: u64,
    },
    /// Axis-aligned bounding box geometry — for BLAS. Each primitive is
    /// a `VkAabbPositionsKHR` (6 × f32 = min.xyz, max.xyz). Used for
    /// procedural primitives (spheres, voxels, data points).
    Aabbs {
        /// Device address of the AABB array.
        data_address: u64,
        /// Byte stride between successive AABBs (must be ≥ 24).
        stride: u64,
    },
    /// Instance array — for TLAS only. Each element is a
    /// `VkAccelerationStructureInstanceKHR` describing one BLAS
    /// reference + transform + mask.
    Instances {
        /// Device address of the instance array (or array of pointers
        /// to instances, if `array_of_pointers` is `true`).
        data_address: u64,
        /// `true` → `data_address` points at an array of *pointers* to
        /// instance structs; `false` → array of structs directly.
        array_of_pointers: bool,
    },
}

impl AccelerationStructureGeometry {
    pub(crate) fn to_raw(self) -> VkAccelerationStructureGeometryKHR {
        match self {
            Self::Triangles {
                vertex_format,
                vertex_data_address,
                vertex_stride,
                max_vertex,
                index_type,
                index_data_address,
                transform_data_address,
            } => VkAccelerationStructureGeometryKHR {
                sType: VkStructureType::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                pNext: std::ptr::null(),
                geometryType: VkGeometryTypeKHR::GEOMETRY_TYPE_TRIANGLES_KHR,
                geometry: VkAccelerationStructureGeometryDataKHR {
                    triangles: VkAccelerationStructureGeometryTrianglesDataKHR {
                        sType: VkStructureType::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                        pNext: std::ptr::null(),
                        vertexFormat: vertex_format,
                        vertexData: VkDeviceOrHostAddressConstKHR {
                            deviceAddress: vertex_data_address,
                        },
                        vertexStride: vertex_stride,
                        maxVertex: max_vertex,
                        indexType: index_type,
                        indexData: VkDeviceOrHostAddressConstKHR {
                            deviceAddress: index_data_address,
                        },
                        transformData: VkDeviceOrHostAddressConstKHR {
                            deviceAddress: transform_data_address,
                        },
                    },
                },
                flags: 0,
            },
            Self::Aabbs {
                data_address,
                stride,
            } => VkAccelerationStructureGeometryKHR {
                sType: VkStructureType::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                pNext: std::ptr::null(),
                geometryType: VkGeometryTypeKHR::GEOMETRY_TYPE_AABBS_KHR,
                geometry: VkAccelerationStructureGeometryDataKHR {
                    aabbs: VkAccelerationStructureGeometryAabbsDataKHR {
                        sType: VkStructureType::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
                        pNext: std::ptr::null(),
                        data: VkDeviceOrHostAddressConstKHR {
                            deviceAddress: data_address,
                        },
                        stride,
                    },
                },
                flags: 0,
            },
            Self::Instances {
                data_address,
                array_of_pointers,
            } => VkAccelerationStructureGeometryKHR {
                sType: VkStructureType::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                pNext: std::ptr::null(),
                geometryType: VkGeometryTypeKHR::GEOMETRY_TYPE_INSTANCES_KHR,
                geometry: VkAccelerationStructureGeometryDataKHR {
                    instances: VkAccelerationStructureGeometryInstancesDataKHR {
                        sType: VkStructureType::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                        pNext: std::ptr::null(),
                        arrayOfPointers: array_of_pointers as VkBool32,
                        data: VkDeviceOrHostAddressConstKHR {
                            deviceAddress: data_address,
                        },
                    },
                },
                flags: 0,
            },
        }
    }
}

/// One build-range entry paired with each geometry in a build.
///
/// `primitive_count` is the count of triangles / AABBs / instances to
/// consume from the geometry's data address. The `*_offset` fields let
/// you carve out a sub-range of a larger buffer without building
/// separate geometries per region.
#[derive(Debug, Clone, Copy, Default)]
pub struct BuildRange {
    pub primitive_count: u32,
    pub primitive_offset: u32,
    pub first_vertex: u32,
    pub transform_offset: u32,
}

impl BuildRange {
    pub(crate) fn to_raw(self) -> VkAccelerationStructureBuildRangeInfoKHR {
        VkAccelerationStructureBuildRangeInfoKHR {
            primitiveCount: self.primitive_count,
            primitiveOffset: self.primitive_offset,
            firstVertex: self.first_vertex,
            transformOffset: self.transform_offset,
        }
    }
}

/// Result of [`Device::acceleration_structure_build_sizes`].
///
/// The sizes are the driver's conservative *upper* bounds for
/// `max_primitive_counts`. Your AS backing buffer must be ≥
/// `acceleration_structure_size` and your scratch buffer must be ≥
/// `build_scratch_size` (or `update_scratch_size` for update builds).
#[derive(Debug, Clone, Copy, Default)]
pub struct BuildSizes {
    pub acceleration_structure_size: u64,
    pub update_scratch_size: u64,
    pub build_scratch_size: u64,
}

/// Safe wrapper around an acceleration-structure handle plus its
/// backing buffer.
///
/// Holds an `Arc<Buffer>` so the backing buffer stays alive as long as
/// the AS does — the Vulkan spec requires the buffer to outlive the
/// `VkAccelerationStructureKHR` handle.
///
/// Destroyed automatically on drop via the auto-generated
/// [`AccelerationStructureKHR`] RAII wrapper.
pub struct AccelerationStructure {
    raw: AccelerationStructureKHR,
    backing: Arc<Buffer>,
    type_: AccelerationStructureType,
    device: Arc<DeviceInner>,
}

/// Parameters for [`AccelerationStructure::new`].
pub struct AccelerationStructureCreateInfo<'a> {
    /// Backing buffer the AS bytes live in. Must have been created with
    /// `ACCELERATION_STRUCTURE_STORAGE_BIT_KHR` usage.
    pub buffer: Arc<Buffer>,
    /// Byte offset into `buffer` at which the AS starts.
    pub offset: u64,
    /// Size in bytes. Must be ≥ the
    /// [`BuildSizes::acceleration_structure_size`] reported for the
    /// matching geometry.
    pub size: u64,
    /// Whether this is a BLAS, TLAS, or generic AS.
    pub type_: AccelerationStructureType,
    /// Phantom to allow future `'a`-borrowed fields without API churn.
    pub _marker: std::marker::PhantomData<&'a ()>,
}

impl AccelerationStructure {
    /// Create a fresh acceleration-structure handle over a backing
    /// buffer range. Does *not* build the structure — the returned
    /// handle contains no geometry yet; call
    /// [`CommandBufferRecording::build_acceleration_structure`](crate::safe::CommandBufferRecording::build_acceleration_structure)
    /// next.
    pub fn new(device: &Device, info: AccelerationStructureCreateInfo<'_>) -> Result<Self> {
        let raw_info = VkAccelerationStructureCreateInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            pNext: std::ptr::null(),
            createFlags: 0,
            buffer: info.buffer.raw(),
            offset: info.offset,
            size: info.size,
            r#type: info.type_.to_raw(),
            deviceAddress: 0,
        };
        let raw = AccelerationStructureKHR::create(device, &raw_info)?;
        Ok(Self {
            raw,
            backing: info.buffer,
            type_: info.type_,
            device: Arc::clone(&device.inner),
        })
    }

    /// Returns the raw `VkAccelerationStructureKHR` handle.
    pub fn raw(&self) -> VkAccelerationStructureKHR {
        self.raw.raw()
    }

    /// Type this AS was created with.
    pub fn type_(&self) -> AccelerationStructureType {
        self.type_
    }

    /// Backing buffer the AS bytes live in.
    pub fn backing_buffer(&self) -> &Arc<Buffer> {
        &self.backing
    }

    /// Device address of this AS — the value shaders bind to.
    ///
    /// For ray queries inside a compute shader, pass this into the
    /// `accelerationStructureEXT` descriptor or the `rayQueryEXT`
    /// initializer. For ray-tracing pipelines, the TLAS address is
    /// typically placed in a descriptor set.
    ///
    /// Returns `Err(MissingFunction(..))` if
    /// `vkGetAccelerationStructureDeviceAddressKHR` wasn't loaded.
    pub fn device_address(&self) -> Result<u64> {
        let f = self
            .device
            .dispatch
            .vkGetAccelerationStructureDeviceAddressKHR
            .ok_or(Error::MissingFunction(
                "vkGetAccelerationStructureDeviceAddressKHR",
            ))?;
        let info = VkAccelerationStructureDeviceAddressInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            pNext: std::ptr::null(),
            accelerationStructure: self.raw.raw(),
        };
        // Safety: handle is valid; info lives for the call.
        Ok(unsafe { f(self.device.handle, &info) })
    }
}

impl Device {
    /// Ask the driver how big an AS and its scratch buffer need to be
    /// to hold the given geometry.
    ///
    /// `max_primitive_counts` must be the same length as `geometries` —
    /// each entry is the maximum primitive count for the matching
    /// geometry (i.e. the upper bound on `BuildRange.primitive_count`
    /// for any subsequent build against the returned size).
    ///
    /// Returns `Err(MissingFunction(..))` if
    /// `vkGetAccelerationStructureBuildSizesKHR` wasn't loaded.
    pub fn acceleration_structure_build_sizes(
        &self,
        build_type: AccelerationStructureBuildType,
        type_: AccelerationStructureType,
        geometries: &[AccelerationStructureGeometry],
        max_primitive_counts: &[u32],
        flags: AccelerationStructureBuildFlags,
    ) -> Result<BuildSizes> {
        if geometries.len() != max_primitive_counts.len() {
            return Err(Error::InvalidArgument(
                "geometries and max_primitive_counts must have equal length",
            ));
        }
        let f = self
            .inner
            .dispatch
            .vkGetAccelerationStructureBuildSizesKHR
            .ok_or(Error::MissingFunction(
                "vkGetAccelerationStructureBuildSizesKHR",
            ))?;

        let raw_geoms: Vec<VkAccelerationStructureGeometryKHR> =
            geometries.iter().map(|g| g.to_raw()).collect();

        let info = VkAccelerationStructureBuildGeometryInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            pNext: std::ptr::null(),
            r#type: type_.to_raw(),
            flags: flags.0,
            mode: VkBuildAccelerationStructureModeKHR::BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            srcAccelerationStructure: 0,
            dstAccelerationStructure: 0,
            geometryCount: raw_geoms.len() as u32,
            pGeometries: if raw_geoms.is_empty() {
                std::ptr::null()
            } else {
                raw_geoms.as_ptr()
            },
            ppGeometries: std::ptr::null(),
            scratchData: VkDeviceOrHostAddressKHR { deviceAddress: 0 },
        };

        let mut sizes = VkAccelerationStructureBuildSizesInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
            pNext: std::ptr::null_mut(),
            accelerationStructureSize: 0,
            updateScratchSize: 0,
            buildScratchSize: 0,
        };

        // Safety: all inputs outlive the synchronous call; output is
        // written by the driver.
        unsafe {
            f(
                self.inner.handle,
                build_type.to_raw(),
                &info,
                max_primitive_counts.as_ptr(),
                &mut sizes,
            )
        };

        Ok(BuildSizes {
            acceleration_structure_size: sizes.accelerationStructureSize,
            update_scratch_size: sizes.updateScratchSize,
            build_scratch_size: sizes.buildScratchSize,
        })
    }
}
