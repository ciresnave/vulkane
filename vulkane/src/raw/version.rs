// SPDX-License-Identifier: MIT OR Apache-2.0
//! Vulkan version utilities
//!
//! Version constants (VK_API_VERSION_1_0, etc.) and version manipulation
//! functions (vk_make_api_version, vk_api_version_major, etc.) are generated
//! from vk.xml and available via `crate::raw::bindings::*`.

use crate::raw::bindings::*;

/// Represents a Vulkan API version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Version {
    /// The major version (`VK_API_VERSION_MAJOR`), 1 for every released Vulkan.
    pub major: u32,
    /// The minor version (`VK_API_VERSION_MINOR`), e.g. 3 for Vulkan 1.3.
    pub minor: u32,
    /// The patch version (`VK_API_VERSION_PATCH`).
    pub patch: u32,
}

impl Version {
    /// A version from its three parts. No range check: the packed encoding
    /// ([`to_raw`](Self::to_raw)) keeps `minor` in 10 bits and `patch` in 12.
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Unpack a `uint32_t` Vulkan API version. The 3-bit variant field is
    /// dropped, so `from_raw(v).to_raw() == v` only when the variant was 0.
    pub fn from_raw(version: u32) -> Self {
        Self {
            major: vk_api_version_major(version),
            minor: vk_api_version_minor(version),
            patch: vk_api_version_patch(version),
        }
    }

    /// Pack into a `uint32_t` Vulkan API version, with variant 0
    /// (`VK_MAKE_API_VERSION(0, major, minor, patch)`).
    pub fn to_raw(&self) -> u32 {
        vk_make_api_version(0, self.major, self.minor, self.patch)
    }
}
