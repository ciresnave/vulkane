//! Device feature enable lists.
//!
//! Vulkan exposes optional functionality via "feature" booleans —
//! originally on the flat [`VkPhysicalDeviceFeatures`] struct in Vulkan
//! 1.0, and now on ~100 additional `*Features` structs chained via
//! `pNext` when a device is created. To enable a feature the
//! application must:
//!
//! 1. Verify the device supports it via `vkGetPhysicalDeviceFeatures2`.
//! 2. Re-pass the same feature struct chain when creating the device.
//!
//! [`DeviceFeatures`] hides all that plumbing. The struct owns a
//! [`PNextChain`] pre-seeded with [`VkPhysicalDeviceFeatures2`] at the
//! head; each `with_<feature>()` call either twiddles a bit in the
//! core 1.0 struct (embedded in `features2.features`) or lazily
//! appends the matching extension struct to the chain and sets the
//! appropriate bit.
//!
//! The full list of `with_*` methods is **generated from `vk.xml`** —
//! one per unique feature bit across every struct that extends
//! `VkPhysicalDeviceFeatures2`. See
//! `OUT_DIR/device_features_generated.rs` in a built workspace for the
//! concrete list.
//!
//! ```ignore
//! use vulkane::safe::{DeviceFeatures, DeviceCreateInfo, QueueCreateInfo};
//!
//! let features = DeviceFeatures::new()
//!     .with_buffer_device_address()      // Vulkan 1.2 core
//!     .with_timeline_semaphore()         // Vulkan 1.2 core
//!     .with_synchronization2()           // Vulkan 1.3 core
//!     .with_cooperative_matrix();        // VK_KHR_cooperative_matrix
//!
//! let device = physical.create_device(DeviceCreateInfo {
//!     queue_create_infos: &[QueueCreateInfo { /* ... */ }],
//!     enabled_features: Some(&features),
//!     ..Default::default()
//! })?;
//! ```
//!
//! # Escape hatch
//!
//! Features whose bits end up routed through a priority collision
//! (e.g. you want to set `timelineSemaphore` on the KHR extension
//! struct instead of the Vulkan 1.2 aggregate — the generated method
//! only targets the highest-priority struct) can be reached with
//! [`DeviceFeatures::chain_extension_feature`]. The user supplies a
//! fully-initialised feature struct and the wrapper chains it verbatim.

use crate::raw::PNextChainable;
use crate::raw::bindings::{VkPhysicalDeviceFeatures, VkPhysicalDeviceFeatures2};
use crate::safe::PNextChain;

/// Buildable list of Vulkan device features to enable at
/// `vkCreateDevice` time.
///
/// Use [`DeviceFeatures::new`] as the starting point and chain
/// `with_<feature>()` calls for each bit you want on. The call-site
/// API is stable across Vulkan versions because it is generated from
/// `vk.xml`; any new feature bit added to a future spec shows up as
/// a new `with_*` method automatically.
///
/// Supplying a feature that the device does not actually support
/// produces `VK_ERROR_FEATURE_NOT_PRESENT` from `vkCreateDevice`.
/// `DeviceFeatures` does no up-front validation — query support with
/// [`PhysicalDevice::supported_features`](super::PhysicalDevice::supported_features)
/// if you need to degrade gracefully.
pub struct DeviceFeatures {
    chain: PNextChain,
}

impl Default for DeviceFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceFeatures {
    /// Build an empty feature set. No bits are enabled yet.
    pub fn new() -> Self {
        let mut chain = PNextChain::new();
        chain.push(VkPhysicalDeviceFeatures2::new_pnext());
        Self { chain }
    }

    /// Chain an arbitrary Vulkan feature struct onto the `pNext` list.
    ///
    /// Use this when the spec has a feature bit that the generated
    /// `with_*` methods route through a higher-priority struct than you
    /// want (for example to enable `timelineSemaphore` on
    /// `VkPhysicalDeviceTimelineSemaphoreFeaturesKHR` for a driver that
    /// doesn't expose the Vulkan 1.2 aggregate), or when a brand-new
    /// extension struct isn't yet covered by `vk.xml` that your crate
    /// was built against.
    ///
    /// The struct must fill in every field you care about before this
    /// call; `PNextChain` only patches `pNext`, never body fields.
    pub fn chain_extension_feature<T: PNextChainable>(mut self, item: T) -> Self {
        self.chain.push(item);
        self
    }

    /// Access the chain built so far (read-only). Mainly useful for
    /// tests and diagnostics.
    #[cfg(test)]
    pub(crate) fn chain(&self) -> &PNextChain {
        &self.chain
    }

    /// Return a cloned pNext chain suitable for attaching to a
    /// `VkDeviceCreateInfo`. Used by
    /// [`PhysicalDevice::create_device`](super::PhysicalDevice::create_device)
    /// so the caller's `DeviceFeatures` remains usable for additional
    /// device creations.
    pub(crate) fn clone_chain_for_device_create(&self) -> PNextChain {
        self.chain.clone()
    }

    /// Mutable access to the Vulkan 1.0 core `VkPhysicalDeviceFeatures`
    /// struct embedded in the chain's head
    /// `VkPhysicalDeviceFeatures2`. Called by generated core-1.0 bit
    /// setters.
    ///
    /// Panics only if the chain header struct was somehow removed —
    /// which cannot happen through the public API.
    fn features10_mut(&mut self) -> &mut VkPhysicalDeviceFeatures {
        &mut self
            .chain
            .get_mut::<VkPhysicalDeviceFeatures2>()
            .expect("chain head is always VkPhysicalDeviceFeatures2")
            .features
    }

    /// Find or insert an extension feature struct of type `T` and
    /// return a mutable reference so a generated setter can flip the
    /// right bit. Called by every non-core `with_*` method in the
    /// generated impl block.
    fn ensure_ext<T: PNextChainable>(&mut self) -> &mut T {
        if self.chain.get::<T>().is_none() {
            self.chain.push(T::new_pnext());
        }
        self.chain
            .get_mut::<T>()
            .expect("struct was just pushed onto the chain")
    }
}

impl std::fmt::Debug for DeviceFeatures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceFeatures")
            .field("chain", &self.chain)
            .finish()
    }
}

// Generated `with_<feature>()` methods — one per unique feature bit
// across every struct that extends `VkPhysicalDeviceFeatures2`. See
// `vulkan_gen::codegen::generator_modules::device_features_gen`.
include!(concat!(env!("OUT_DIR"), "/device_features_generated.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::bindings::{
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR, VkPhysicalDeviceVulkan12Features,
        VkPhysicalDeviceVulkan13Features, VkStructureType,
    };

    #[test]
    fn new_has_only_features2_head() {
        let f = DeviceFeatures::new();
        assert_eq!(f.chain().len(), 1);
        let types: Vec<_> = f.chain().structure_types().collect();
        assert_eq!(
            types,
            vec![VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2]
        );
    }

    #[test]
    fn core_1_0_bit_flips_inside_features2() {
        let f = DeviceFeatures::new().with_sampler_anisotropy();
        // Chain length is unchanged — the bit lives inside features2.
        assert_eq!(f.chain().len(), 1);
        let features2 = f
            .chain()
            .get::<VkPhysicalDeviceFeatures2>()
            .expect("head present");
        assert_eq!(features2.features.samplerAnisotropy, 1);
    }

    #[test]
    fn v12_bit_lazily_pushes_struct() {
        let f = DeviceFeatures::new().with_timeline_semaphore();
        assert_eq!(f.chain().len(), 2);
        let v12 = f
            .chain()
            .get::<VkPhysicalDeviceVulkan12Features>()
            .expect("v12 pushed");
        assert_eq!(v12.timelineSemaphore, 1);
    }

    #[test]
    fn multiple_bits_on_same_struct_share_one_push() {
        let f = DeviceFeatures::new()
            .with_timeline_semaphore()
            .with_buffer_device_address();
        // features2 head + one VkPhysicalDeviceVulkan12Features.
        assert_eq!(f.chain().len(), 2);
        let v12 = f
            .chain()
            .get::<VkPhysicalDeviceVulkan12Features>()
            .unwrap();
        assert_eq!(v12.timelineSemaphore, 1);
        assert_eq!(v12.bufferDeviceAddress, 1);
    }

    #[test]
    fn mixing_core_versions_pushes_one_struct_per_version() {
        let f = DeviceFeatures::new()
            .with_sampler_anisotropy() // 1.0 -> no push
            .with_timeline_semaphore() // 1.2 -> push v12
            .with_synchronization2(); // 1.3 -> push v13
        assert_eq!(f.chain().len(), 3);
        assert!(f.chain().get::<VkPhysicalDeviceVulkan12Features>().is_some());
        assert!(f.chain().get::<VkPhysicalDeviceVulkan13Features>().is_some());
    }

    #[test]
    fn escape_hatch_chains_any_feature_struct() {
        let mut coop = VkPhysicalDeviceCooperativeMatrixFeaturesKHR::new_pnext();
        coop.cooperativeMatrix = 1;
        let f = DeviceFeatures::new().chain_extension_feature(coop);
        let back = f
            .chain()
            .get::<VkPhysicalDeviceCooperativeMatrixFeaturesKHR>()
            .expect("coop chained");
        assert_eq!(back.cooperativeMatrix, 1);
    }

    #[test]
    fn cooperative_matrix_generated_method_works() {
        // Fuel's request: a one-call enable path for
        // VK_KHR_cooperative_matrix's `cooperativeMatrix` bit.
        let f = DeviceFeatures::new().with_cooperative_matrix();
        let coop = f
            .chain()
            .get::<VkPhysicalDeviceCooperativeMatrixFeaturesKHR>()
            .expect("coop struct auto-pushed");
        assert_eq!(coop.cooperativeMatrix, 1);
    }
}
