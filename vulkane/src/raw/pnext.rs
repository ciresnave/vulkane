//! The [`PNextChainable`] trait — generated implementations of this trait
//! identify every Vulkan struct that can be linked into a `pNext` chain.
//!
//! Every Vulkan struct whose first two fields are
//! `sType: VkStructureType` followed by `pNext: *mut c_void` gets a
//! generated `unsafe impl PNextChainable for VkFoo` in the raw bindings.
//! The trait carries the `VkStructureType` value the Vulkan spec assigns
//! to that struct and exposes raw-pointer casts to the common
//! [`VkBaseOutStructure`] head so that generic chain-walking code in
//! [`crate::safe::pnext`] can stitch them together.
//!
//! Users should never need to implement this trait by hand — the
//! code generator emits one impl per qualifying struct in `vk.xml`.

use super::bindings::{VkBaseOutStructure, VkStructureType};

/// Marker + helper trait for Vulkan structs that can participate in a
/// `pNext` chain.
///
/// # Safety
///
/// Implementors must be `#[repr(C)]` with:
///
/// 1. `sType: VkStructureType` as the first field (offset 0).
/// 2. `pNext: *mut c_void` as the second field (immediately after
///    `sType`, so that a pointer to the struct can be cast to
///    [`*mut VkBaseOutStructure`] without violating layout rules).
///
/// [`STRUCTURE_TYPE`](Self::STRUCTURE_TYPE) must be the exact
/// `VkStructureType` enumerant the Vulkan spec requires for this struct
/// — driver behaviour when sType disagrees is undefined.
pub unsafe trait PNextChainable: Clone + Default + 'static {
    /// The `VkStructureType` value the Vulkan spec requires for this
    /// struct type.
    const STRUCTURE_TYPE: VkStructureType;

    /// View this struct as a generic [`VkBaseOutStructure`] pointer so
    /// that chain-walking code can read/patch `sType` and `pNext`
    /// without knowing the concrete type.
    #[inline]
    fn as_base_ptr(&self) -> *const VkBaseOutStructure {
        self as *const Self as *const VkBaseOutStructure
    }

    /// Mutable variant of [`as_base_ptr`](Self::as_base_ptr).
    #[inline]
    fn as_base_mut_ptr(&mut self) -> *mut VkBaseOutStructure {
        self as *mut Self as *mut VkBaseOutStructure
    }

    /// Construct an instance with `sType` correctly initialised and
    /// every other field zero-initialised.
    ///
    /// Equivalent to `{ sType: Self::STRUCTURE_TYPE, ..Default::default() }`
    /// but without naming the field, so callers can use it in generic
    /// code.
    #[inline]
    fn new_pnext() -> Self
    where
        Self: Sized,
    {
        let mut s = Self::default();
        // Safety: the trait contract guarantees `sType` is the first
        // field of `Self`, so a `*mut Self` can be cast to
        // `*mut VkStructureType` and written without violating layout.
        unsafe {
            *(&raw mut s as *mut VkStructureType) = Self::STRUCTURE_TYPE;
        }
        s
    }
}
