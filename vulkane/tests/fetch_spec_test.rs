// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration test verifying that the generated bindings work correctly
//! regardless of vk.xml source (local file or auto-downloaded).
//!
//! Run with: cargo test -p vulkane --test fetch_spec_test

use vulkane::raw::bindings::*;

#[test]
fn test_generated_bindings_have_core_types() {
    // These types must exist regardless of which vk.xml version was used
    let _: VkInstance = std::ptr::null_mut();
    let _: VkDevice = std::ptr::null_mut();
    let _: VkBuffer = 0u64;
    let _: VkBool32 = 0u32;
    let _: VkFlags = 0u32;
}

#[test]
fn test_generated_bindings_have_version_functions() {
    // Version functions must be transpiled from whatever vk.xml was used
    let v = vk_make_api_version(0, 1, 0, 0);
    assert_eq!(vk_api_version_major(v), 1);
    assert_eq!(vk_api_version_minor(v), 0);
    assert_eq!(vk_api_version_patch(v), 0);
    assert_eq!(vk_api_version_variant(v), 0);
}

#[test]
fn test_generated_bindings_have_structs_with_members() {
    // VkApplicationInfo must have all its fields
    let info = VkApplicationInfo::default();
    assert!(info.pNext.is_null());
}

#[test]
fn test_generated_bindings_have_dispatch_tables() {
    let _: Option<Box<VkEntryDispatchTable>> = None;
    let _: Option<Box<VkInstanceDispatchTable>> = None;
    let _: Option<Box<VkDeviceDispatchTable>> = None;
}
