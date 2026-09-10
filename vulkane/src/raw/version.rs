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
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    pub fn from_raw(version: u32) -> Self {
        Self {
            major: vk_api_version_major(version),
            minor: vk_api_version_minor(version),
            patch: vk_api_version_patch(version),
        }
    }

    pub fn to_raw(&self) -> u32 {
        vk_make_api_version(0, self.major, self.minor, self.patch)
    }
}
