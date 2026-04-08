//! Safe wrapper for `VkQueryPool` — timestamp and pipeline-statistics queries.
//!
//! Query pools are how you measure GPU-side timing and statistics. The
//! typical workflow:
//!
//! 1. Create a [`QueryPool`] of the desired type and size.
//! 2. In a command buffer, call [`reset_query_pool`](super::CommandBufferRecording::reset_query_pool)
//!    *before* writing any queries (this is **required** every submission).
//! 3. Record [`write_timestamp`](super::CommandBufferRecording::write_timestamp)
//!    or `vkCmdBeginQuery`/`vkCmdEndQuery` to populate the queries.
//! 4. Submit the command buffer and wait for it to finish.
//! 5. Read the results back with [`QueryPool::get_results_u64`].
//!
//! For timestamps, the resulting `u64` values are in implementation-defined
//! ticks. To convert to nanoseconds, multiply by
//! [`PhysicalDevice::timestamp_period`](super::PhysicalDevice::timestamp_period).

use super::device::DeviceInner;
use super::{Device, Error, Result, check};
use crate::raw::bindings::*;
use std::sync::Arc;

/// What kind of values a [`QueryPool`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryType(pub VkQueryType);

impl QueryType {
    /// Pixel-occlusion queries (graphics — counts samples passing the depth/
    /// stencil tests). Useful only for graphics pipelines.
    pub const OCCLUSION: Self = Self(VkQueryType::QUERY_TYPE_OCCLUSION);
    /// Pipeline-statistics queries (vertex/fragment/compute invocations,
    /// primitives generated, etc.). Requires the
    /// `pipelineStatisticsQuery` device feature.
    pub const PIPELINE_STATISTICS: Self = Self(VkQueryType::QUERY_TYPE_PIPELINE_STATISTICS);
    /// GPU timestamp queries. Requires `queue_family.timestampValidBits > 0`
    /// on the queue family the commands run on.
    pub const TIMESTAMP: Self = Self(VkQueryType::QUERY_TYPE_TIMESTAMP);
}

/// Pipeline statistics flag bits — used as `pipeline_statistics` when
/// creating a `PIPELINE_STATISTICS` query pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineStatisticsFlags(pub u32);

impl PipelineStatisticsFlags {
    pub const NONE: Self = Self(0);
    pub const INPUT_ASSEMBLY_VERTICES: Self = Self(0x1);
    pub const INPUT_ASSEMBLY_PRIMITIVES: Self = Self(0x2);
    pub const VERTEX_SHADER_INVOCATIONS: Self = Self(0x4);
    pub const GEOMETRY_SHADER_INVOCATIONS: Self = Self(0x8);
    pub const GEOMETRY_SHADER_PRIMITIVES: Self = Self(0x10);
    pub const CLIPPING_INVOCATIONS: Self = Self(0x20);
    pub const CLIPPING_PRIMITIVES: Self = Self(0x40);
    pub const FRAGMENT_SHADER_INVOCATIONS: Self = Self(0x80);
    pub const TESSELLATION_CONTROL_SHADER_PATCHES: Self = Self(0x100);
    pub const TESSELLATION_EVALUATION_SHADER_INVOCATIONS: Self = Self(0x200);
    pub const COMPUTE_SHADER_INVOCATIONS: Self = Self(0x400);

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Number of statistics enabled in this mask. The result vector returned
    /// by `get_results_u64` for a `PIPELINE_STATISTICS` pool stores this
    /// many `u64`s per query.
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }
}

impl std::ops::BitOr for PipelineStatisticsFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// A safe wrapper around `VkQueryPool`.
///
/// Query pools are arrays of fixed-size query slots. They are destroyed
/// automatically on drop.
pub struct QueryPool {
    pub(crate) handle: VkQueryPool,
    pub(crate) device: Arc<DeviceInner>,
    pub(crate) query_count: u32,
    /// Number of `u64` result values produced per query — `1` for timestamp
    /// and occlusion, `popcount(pipeline_statistics)` for pipeline stats.
    pub(crate) values_per_query: u32,
}

impl QueryPool {
    /// Create a timestamp query pool with `query_count` slots.
    pub fn timestamps(device: &Device, query_count: u32) -> Result<Self> {
        Self::new_inner(
            device,
            QueryType::TIMESTAMP,
            query_count,
            PipelineStatisticsFlags::NONE,
            1,
        )
    }

    /// Create a pipeline-statistics query pool with `query_count` slots,
    /// collecting the statistics enabled in `stats`.
    ///
    /// Each query in the pool will produce `stats.count()` `u64` values when
    /// read back. The Vulkan spec requires that at least one statistic is
    /// enabled.
    ///
    /// Requires the `pipelineStatisticsQuery` device feature.
    pub fn pipeline_statistics(
        device: &Device,
        query_count: u32,
        stats: PipelineStatisticsFlags,
    ) -> Result<Self> {
        debug_assert!(
            stats.0 != 0,
            "pipeline_statistics requires at least one bit set"
        );
        Self::new_inner(
            device,
            QueryType::PIPELINE_STATISTICS,
            query_count,
            stats,
            stats.count(),
        )
    }

    fn new_inner(
        device: &Device,
        query_type: QueryType,
        query_count: u32,
        pipeline_stats: PipelineStatisticsFlags,
        values_per_query: u32,
    ) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreateQueryPool
            .ok_or(Error::MissingFunction("vkCreateQueryPool"))?;

        let info = VkQueryPoolCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            queryType: query_type.0,
            queryCount: query_count,
            pipelineStatistics: pipeline_stats.0,
            ..Default::default()
        };

        let mut handle: VkQueryPool = 0;
        // Safety: info is valid for the call, device is valid.
        check(unsafe { create(device.inner.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
            query_count,
            values_per_query,
        })
    }

    /// Returns the raw `VkQueryPool` handle.
    pub fn raw(&self) -> VkQueryPool {
        self.handle
    }

    /// Returns the number of query slots in this pool.
    pub fn query_count(&self) -> u32 {
        self.query_count
    }

    /// Read `query_count` queries starting at `first_query` as `u64` values
    /// (with the `64_BIT` and `WAIT` flags set, so the call blocks until the
    /// values are ready).
    ///
    /// The returned vector is `query_count * values_per_query` long. For a
    /// timestamp pool that's `query_count` `u64`s. For a pipeline-statistics
    /// pool with `N` enabled bits that's `query_count * N` `u64`s, with the
    /// statistics for each query laid out in the order of their bit indices
    /// (low to high).
    pub fn get_results_u64(&self, first_query: u32, query_count: u32) -> Result<Vec<u64>> {
        let get = self
            .device
            .dispatch
            .vkGetQueryPoolResults
            .ok_or(Error::MissingFunction("vkGetQueryPoolResults"))?;

        let total = (query_count as usize) * (self.values_per_query as usize);
        let mut data: Vec<u64> = vec![0; total];

        // Stride is the number of bytes between consecutive *queries* (not
        // values), so we use values_per_query * 8.
        let stride = (self.values_per_query as u64) * 8;

        // VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
        let flags: u32 = 0x1 | 0x2;

        // Safety: data has space for `total` u64s = `total * 8` bytes,
        // matching the requested layout.
        check(unsafe {
            get(
                self.device.handle,
                self.handle,
                first_query,
                query_count,
                total * 8,
                data.as_mut_ptr() as *mut _,
                stride,
                flags,
            )
        })?;

        Ok(data)
    }
}

impl Drop for QueryPool {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroyQueryPool {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}
