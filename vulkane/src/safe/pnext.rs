//! [`PNextChain`] — a safe, ordered builder for Vulkan `pNext` chains.
//!
//! Vulkan extends many create-info and query structs by letting callers
//! link extra structs onto a `pNext` singly-linked list. Every link
//! starts with `sType` + `pNext` so drivers can walk the chain without
//! knowing the individual struct types.
//!
//! `PNextChain` owns a heap-allocated copy of each link, stitches the
//! `pNext` pointers together in insertion order, and exposes a raw
//! head pointer that can be embedded in a parent struct. All of the
//! unsafe pointer plumbing — address stability, link patching,
//! downcasting on read-back — lives here so that call sites can just
//! `.push(…)` typed feature structs and hand the chain to the Vulkan
//! entry point.
//!
//! ```ignore
//! use vulkane::raw::bindings::{VkPhysicalDeviceVulkan12Features, VkPhysicalDeviceVulkan13Features};
//! use vulkane::raw::PNextChainable;
//! use vulkane::safe::PNextChain;
//!
//! let mut chain = PNextChain::new();
//! chain.push(VkPhysicalDeviceVulkan12Features {
//!     timelineSemaphore: 1,
//!     ..VkPhysicalDeviceVulkan12Features::new_pnext()
//! });
//! chain.push(VkPhysicalDeviceVulkan13Features {
//!     synchronization2: 1,
//!     ..VkPhysicalDeviceVulkan13Features::new_pnext()
//! });
//!
//! // chain.head() is now a *const c_void pointing at the first struct,
//! // with pNext on that struct pointing at the second, and pNext on
//! // the second being null. Embed it in VkDeviceCreateInfo.pNext.
//! ```

use core::ffi::c_void;
use std::any::{Any, TypeId};

use crate::raw::PNextChainable;
use crate::raw::bindings::{VkBaseOutStructure, VkStructureType};

/// Internal trait implemented for every concrete `T: PNextChainable` so
/// that the chain can store mixed-type nodes in a single `Vec`.
///
/// The `*mut VkBaseOutStructure` projection is the only operation the
/// chain actually needs at runtime; the `Any` accessors power the
/// typed [`PNextChain::get`]/[`get_mut`](PNextChain::get_mut) readback
/// used after output-direction Vulkan calls.
trait ErasedChain: Any {
    fn as_base_mut_ptr(&mut self) -> *mut VkBaseOutStructure;
    fn as_any_ref(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn stored_type_id(&self) -> TypeId;
    /// Deep-clone the concrete struct into a fresh box. `pNext` on
    /// the cloned copy is re-zeroed so the caller's [`PNextChain`]
    /// can re-patch the link pointers.
    fn clone_erased(&self) -> Box<dyn ErasedChain>;
}

impl<T: PNextChainable> ErasedChain for T {
    #[inline]
    fn as_base_mut_ptr(&mut self) -> *mut VkBaseOutStructure {
        <T as PNextChainable>::as_base_mut_ptr(self)
    }
    #[inline]
    fn as_any_ref(&self) -> &dyn Any {
        self
    }
    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    #[inline]
    fn stored_type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }
    fn clone_erased(&self) -> Box<dyn ErasedChain> {
        let mut cloned = self.clone();
        // Safety: trait contract guarantees `pNext` sits immediately
        // after `sType` at the start of the struct. Zero it so the
        // destination chain can re-link on its own.
        unsafe {
            let base = &mut cloned as *mut T as *mut VkBaseOutStructure;
            (*base).pNext = std::ptr::null_mut();
        }
        Box::new(cloned)
    }
}

/// Ordered, owning builder for a Vulkan `pNext` chain.
///
/// Each value pushed is boxed onto the heap so its address is stable
/// across subsequent pushes. [`head`](Self::head) / [`head_mut`](Self::head_mut)
/// expose the first node as a raw pointer suitable for embedding in a
/// parent struct's `pNext` field.
///
/// The chain is **ordered**: the first call to [`push`](Self::push) becomes
/// the first link (pointed at by `head`), and subsequent pushes are
/// linked onto the tail. Vulkan itself is order-insensitive within a
/// chain, but keeping insertion order stable makes debugging easier and
/// makes tests deterministic.
#[derive(Default)]
pub struct PNextChain {
    nodes: Vec<Box<dyn ErasedChain>>,
}

impl Clone for PNextChain {
    fn clone(&self) -> Self {
        let mut new = Self {
            nodes: self.nodes.iter().map(|n| n.clone_erased()).collect(),
        };
        new.relink();
        new
    }
}

impl PNextChain {
    /// Create an empty chain. `head()` will return null until something
    /// is pushed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a pNext-chainable struct onto the tail of the chain.
    ///
    /// The struct is heap-allocated via `Box` so its address remains
    /// stable for the lifetime of the chain. `pNext` pointers on every
    /// previously-chained node are re-patched so the chain forms a
    /// valid singly-linked list.
    pub fn push<T: PNextChainable>(&mut self, item: T) -> &mut Self {
        self.nodes.push(Box::new(item));
        self.relink();
        self
    }

    /// Move every node from `other` onto the tail of this chain. The
    /// moved boxes keep their heap addresses, so any raw pointer
    /// previously captured into a node's body remains valid.
    ///
    /// Useful when a caller hands us a preassembled chain (e.g. a
    /// [`DeviceFeatures`](crate::safe::DeviceFeatures) builder) and we
    /// need to prefix or suffix it with additional nodes we built
    /// locally.
    pub fn append(&mut self, mut other: PNextChain) -> &mut Self {
        self.nodes.append(&mut other.nodes);
        self.relink();
        self
    }

    /// `true` if no nodes have been pushed.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Number of nodes in the chain.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Raw head pointer suitable for an input-direction CreateInfo's
    /// `pNext` field (typed `*const c_void`). Returns null for an empty
    /// chain.
    pub fn head(&self) -> *const c_void {
        self.head_raw() as *const c_void
    }

    /// Raw head pointer suitable for an output-direction query's
    /// `pNext` field (typed `*mut c_void`). Returns null for an empty
    /// chain.
    pub fn head_mut(&mut self) -> *mut c_void {
        self.head_raw_mut() as *mut c_void
    }

    /// Return the first link as a typed base-out pointer.
    fn head_raw(&self) -> *const VkBaseOutStructure {
        match self.nodes.first() {
            // Cast through *const () to avoid needing a &mut reference;
            // the VkBaseOutStructure layout is compatible with any
            // PNextChainable struct's first two fields by the trait
            // safety contract.
            Some(b) => {
                let erased: &dyn ErasedChain = b.as_ref();
                erased as *const dyn ErasedChain as *const VkBaseOutStructure
            }
            None => std::ptr::null(),
        }
    }

    fn head_raw_mut(&mut self) -> *mut VkBaseOutStructure {
        match self.nodes.first_mut() {
            Some(b) => b.as_base_mut_ptr(),
            None => std::ptr::null_mut(),
        }
    }

    /// Look up a chained struct by concrete type. Returns the first
    /// node whose `TypeId` matches `T`, or `None` if no such node was
    /// pushed.
    ///
    /// Useful after output-direction calls (e.g. driving
    /// `vkGetPhysicalDeviceFeatures2` with the chain as the query sink)
    /// to read back the values the driver wrote.
    pub fn get<T: PNextChainable>(&self) -> Option<&T> {
        let target = TypeId::of::<T>();
        for n in &self.nodes {
            if n.stored_type_id() == target {
                return n.as_any_ref().downcast_ref::<T>();
            }
        }
        None
    }

    /// Mutable variant of [`get`](Self::get).
    pub fn get_mut<T: PNextChainable>(&mut self) -> Option<&mut T> {
        let target = TypeId::of::<T>();
        for n in &mut self.nodes {
            if n.stored_type_id() == target {
                return n.as_any_mut().downcast_mut::<T>();
            }
        }
        None
    }

    /// Iterate the `sType` of each node in chain order.
    ///
    /// Intended for assertions in tests and for diagnostic logging;
    /// the hot path does not call this.
    pub fn structure_types(&self) -> impl Iterator<Item = VkStructureType> + '_ {
        self.nodes.iter().map(|n| {
            // Safety: each node's Box owns heap memory whose first
            // field is VkStructureType per the trait contract.
            unsafe {
                let base = n.as_ref() as *const dyn ErasedChain as *const VkBaseOutStructure;
                (*base).sType
            }
        })
    }

    /// Rebuild the pNext links to reflect current node addresses.
    ///
    /// Called after every `push`; cheap (O(n) pointer writes) for any
    /// realistic chain length.
    fn relink(&mut self) {
        // Snapshot raw pointers first so we can write without needing
        // simultaneous mutable borrows into the Vec.
        let ptrs: Vec<*mut VkBaseOutStructure> = self
            .nodes
            .iter_mut()
            .map(|b| b.as_base_mut_ptr())
            .collect();
        for (i, &p) in ptrs.iter().enumerate() {
            let next = if i + 1 < ptrs.len() {
                ptrs[i + 1]
            } else {
                std::ptr::null_mut()
            };
            // Safety: each pointer was just obtained from a unique
            // mutable borrow via `iter_mut`, and no &mut reference into
            // the Vec is alive during this loop.
            unsafe {
                (*p).pNext = next;
            }
        }
    }
}

impl std::fmt::Debug for PNextChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PNextChain")
            .field("len", &self.nodes.len())
            .field(
                "structure_types",
                &self.structure_types().collect::<Vec<_>>(),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::bindings::{
        VkPhysicalDeviceVulkan11Features, VkPhysicalDeviceVulkan12Features,
        VkPhysicalDeviceVulkan13Features,
    };

    #[test]
    fn empty_chain_has_null_head() {
        let chain = PNextChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert!(chain.head().is_null());
    }

    #[test]
    fn single_push_links_to_null() {
        let mut chain = PNextChain::new();
        chain.push(VkPhysicalDeviceVulkan12Features::new_pnext());
        assert_eq!(chain.len(), 1);
        assert!(!chain.head().is_null());
        unsafe {
            let head = chain.head() as *const VkBaseOutStructure;
            assert_eq!(
                (*head).sType,
                VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
            );
            assert!((*head).pNext.is_null());
        }
    }

    #[test]
    fn multi_push_forms_ordered_chain() {
        let mut chain = PNextChain::new();
        chain.push(VkPhysicalDeviceVulkan11Features::new_pnext());
        chain.push(VkPhysicalDeviceVulkan12Features::new_pnext());
        chain.push(VkPhysicalDeviceVulkan13Features::new_pnext());

        let types: Vec<_> = chain.structure_types().collect();
        assert_eq!(
            types,
            vec![
                VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            ]
        );

        // Walk the raw pNext chain and confirm it matches.
        unsafe {
            let mut cur = chain.head() as *const VkBaseOutStructure;
            let mut seen = Vec::new();
            while !cur.is_null() {
                seen.push((*cur).sType);
                cur = (*cur).pNext as *const VkBaseOutStructure;
            }
            assert_eq!(seen, types);
        }
    }

    #[test]
    fn new_pnext_sets_stype_correctly() {
        let f = VkPhysicalDeviceVulkan12Features::new_pnext();
        assert_eq!(
            f.sType,
            VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
        );
        // Everything else must be zeroed.
        assert!(f.pNext.is_null());
        assert_eq!(f.timelineSemaphore, 0);
        assert_eq!(f.bufferDeviceAddress, 0);
    }

    #[test]
    fn get_returns_pushed_struct() {
        let mut chain = PNextChain::new();
        let mut v12 = VkPhysicalDeviceVulkan12Features::new_pnext();
        v12.timelineSemaphore = 1;
        v12.bufferDeviceAddress = 1;
        chain.push(v12);
        chain.push(VkPhysicalDeviceVulkan13Features::new_pnext());

        let got = chain
            .get::<VkPhysicalDeviceVulkan12Features>()
            .expect("v12 present");
        assert_eq!(got.timelineSemaphore, 1);
        assert_eq!(got.bufferDeviceAddress, 1);

        // Lookup for an absent type returns None.
        assert!(
            chain
                .get::<crate::raw::bindings::VkPhysicalDeviceFeatures2>()
                .is_none()
        );
    }

    #[test]
    fn extension_struct_chains_like_core_features() {
        // Regression proof for Fuel's use case: an arbitrary KHR feature
        // struct can be chained alongside core feature structs without
        // any hand-written per-extension support in vulkane.
        use crate::raw::bindings::VkPhysicalDeviceCooperativeMatrixFeaturesKHR;

        let mut chain = PNextChain::new();
        chain.push(VkPhysicalDeviceVulkan12Features::new_pnext());
        let mut coop = VkPhysicalDeviceCooperativeMatrixFeaturesKHR::new_pnext();
        coop.cooperativeMatrix = 1;
        chain.push(coop);

        let types: Vec<_> = chain.structure_types().collect();
        assert_eq!(
            types,
            vec![
                VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
            ]
        );

        // The pushed value survives the round-trip through the chain.
        let back = chain
            .get::<VkPhysicalDeviceCooperativeMatrixFeaturesKHR>()
            .expect("coop present");
        assert_eq!(back.cooperativeMatrix, 1);
        assert_eq!(back.cooperativeMatrixRobustBufferAccess, 0);
    }

    #[test]
    fn push_after_head_still_relinks_previous_tail() {
        let mut chain = PNextChain::new();
        chain.push(VkPhysicalDeviceVulkan11Features::new_pnext());
        chain.push(VkPhysicalDeviceVulkan12Features::new_pnext());
        // First push set v11.pNext = null. Second push must re-patch
        // v11.pNext to point at v12 — this is the regression we most
        // care about.
        unsafe {
            let head = chain.head() as *const VkBaseOutStructure;
            let second = (*head).pNext as *const VkBaseOutStructure;
            assert!(!second.is_null());
            assert_eq!(
                (*second).sType,
                VkStructureType::STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
            );
        }
    }
}
