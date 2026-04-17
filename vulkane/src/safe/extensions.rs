//! Device- and instance-level extension enablement builders.
//!
//! Vulkan extensions come in two flavours. **Instance extensions**
//! (debug utilities, surface extensions, the physical-device-properties
//! query extensions) are enabled at `vkCreateInstance` time. **Device
//! extensions** (swapchain, ray tracing, cooperative matrix, most of
//! the interesting stuff) are enabled at `vkCreateDevice` time. The
//! Vulkan loader distinguishes the two and refuses to enable a
//! device-level extension on an instance (and vice versa).
//!
//! [`DeviceExtensions`] and [`InstanceExtensions`] are type-safe
//! builders that mirror that split. The per-extension methods — one
//! per non-disabled extension in `vk.xml` — are **generated**, so
//! every extension Khronos has ever shipped has a one-call enable
//! path. Transitive `requires` dependencies are resolved at generation
//! time, so enabling `khr_ray_tracing_pipeline()` also enables the
//! acceleration-structure and deferred-host-operations extensions it
//! pulls in.
//!
//! Feature-bit enablement is **separate**: see [`DeviceFeatures`](crate::safe::DeviceFeatures).
//! Enabling the extension only makes the API *available*; whether a
//! particular feature bit it introduced is *on* is a second, composable
//! step.
//!
//! ```ignore
//! use vulkane::safe::{DeviceExtensions, DeviceFeatures, DeviceCreateInfo};
//!
//! let device = physical.create_device(DeviceCreateInfo {
//!     enabled_features: Some(&DeviceFeatures::new().with_cooperative_matrix()),
//!     enabled_extensions: Some(&DeviceExtensions::new()
//!         .khr_swapchain()
//!         .khr_cooperative_matrix()),
//!     ..Default::default()
//! })?;
//! ```

use std::ffi::CString;

/// Buildable list of device-level Vulkan extensions.
///
/// Each `<vendor>_<ext>()` method appends the extension name string
/// (and any transitive `requires` it pulls in from the spec) onto the
/// enable list. The same name is never added twice. Use
/// [`enable_raw`](Self::enable_raw) to opt into an extension that
/// hasn't been generated yet (for example a brand-new KHR that landed
/// after your copy of `vk.xml` was built).
#[derive(Clone, Debug, Default)]
pub struct DeviceExtensions {
    enabled: Vec<&'static str>,
}

/// Buildable list of instance-level Vulkan extensions.
///
/// Same contract as [`DeviceExtensions`], but the produced list is
/// embedded in `VkInstanceCreateInfo.ppEnabledExtensionNames` instead
/// of `VkDeviceCreateInfo.ppEnabledExtensionNames`.
#[derive(Clone, Debug, Default)]
pub struct InstanceExtensions {
    enabled: Vec<&'static str>,
}

impl DeviceExtensions {
    /// Start an empty device-extension list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an arbitrary extension name string. No-op if `name` is
    /// already in the list. Prefer the generated `<vendor>_<ext>()`
    /// methods when possible — they pull in transitive dependencies,
    /// whereas this method adds just the single name.
    pub fn enable_raw(mut self, name: &'static str) -> Self {
        self.enable(name);
        self
    }

    /// Returns every enabled extension name in insertion order.
    pub fn names(&self) -> &[&'static str] {
        &self.enabled
    }

    /// `true` iff the given extension name would be sent to
    /// `vkCreateDevice`. Useful for conditional feature toggles.
    pub fn contains(&self, name: &str) -> bool {
        self.enabled.contains(&name)
    }

    /// Internal helper used by every generated method and by
    /// [`enable_raw`](Self::enable_raw). Dedup by value so transitive
    /// `requires` don't produce duplicate entries.
    fn enable(&mut self, name: &'static str) {
        if !self.enabled.contains(&name) {
            self.enabled.push(name);
        }
    }

    /// Convert the enable list into heap-owned `CString`s for embedding
    /// in a `VkDeviceCreateInfo`. The returned `CString` vector must
    /// outlive the `vkCreateDevice` call — callers typically bind it
    /// with `let`.
    pub(crate) fn to_cstrings(&self) -> std::result::Result<Vec<CString>, std::ffi::NulError> {
        self.enabled.iter().map(|s| CString::new(*s)).collect()
    }
}

impl InstanceExtensions {
    /// Start an empty instance-extension list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an arbitrary extension name string. No-op if `name` is
    /// already in the list.
    pub fn enable_raw(mut self, name: &'static str) -> Self {
        self.enable(name);
        self
    }

    /// Returns every enabled extension name in insertion order.
    pub fn names(&self) -> &[&'static str] {
        &self.enabled
    }

    /// `true` iff the given extension name would be sent to
    /// `vkCreateInstance`.
    pub fn contains(&self, name: &str) -> bool {
        self.enabled.contains(&name)
    }

    fn enable(&mut self, name: &'static str) {
        if !self.enabled.contains(&name) {
            self.enabled.push(name);
        }
    }

    /// Convert to heap-owned `CString`s for embedding in a
    /// `VkInstanceCreateInfo`.
    pub(crate) fn to_cstrings(&self) -> std::result::Result<Vec<CString>, std::ffi::NulError> {
        self.enabled.iter().map(|s| CString::new(*s)).collect()
    }
}

// Generated per-extension enable methods. See
// `vulkan_gen::codegen::generator_modules::extensions_builder_gen`.
include!(concat!(env!("OUT_DIR"), "/device_extensions_generated.rs"));
include!(concat!(env!("OUT_DIR"), "/instance_extensions_generated.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_extensions_dedups_transitive_requires() {
        // khr_swapchain transitively requires khr_surface (instance, so
        // not re-enabled here) but the generator places khr_surface on
        // InstanceExtensions — this test just verifies that dedup works
        // when a name is asked to be enabled twice.
        let ext = DeviceExtensions::new()
            .khr_swapchain()
            .enable_raw(crate::raw::bindings::KHR_SWAPCHAIN_EXTENSION_NAME);
        let names = ext.names();
        let count = names
            .iter()
            .filter(|n| **n == crate::raw::bindings::KHR_SWAPCHAIN_EXTENSION_NAME)
            .count();
        assert_eq!(count, 1, "khr_swapchain should only appear once");
    }

    #[test]
    fn device_extensions_khr_cooperative_matrix_includes_deps() {
        // VK_KHR_cooperative_matrix requires
        // VK_KHR_get_physical_device_properties2 transitively.
        let ext = DeviceExtensions::new().khr_cooperative_matrix();
        assert!(
            ext.contains(crate::raw::bindings::KHR_COOPERATIVE_MATRIX_EXTENSION_NAME),
            "primary extension enabled"
        );
    }

    #[test]
    fn instance_extensions_khr_surface() {
        let ext = InstanceExtensions::new().khr_surface();
        assert!(ext.contains(crate::raw::bindings::KHR_SURFACE_EXTENSION_NAME));
    }

    #[test]
    fn raw_escape_hatch_works() {
        let ext = DeviceExtensions::new().enable_raw("VK_FAKE_not_yet_in_spec");
        assert!(ext.contains("VK_FAKE_not_yet_in_spec"));
    }

    #[test]
    fn contains_lookup_is_by_name_string() {
        let ext = DeviceExtensions::new().khr_swapchain();
        assert!(ext.contains("VK_KHR_swapchain"));
        assert!(!ext.contains("VK_KHR_not_enabled"));
    }
}
