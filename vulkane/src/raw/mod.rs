//! Raw Vulkan bindings and loader functionality
//!
//! This module provides direct access to Vulkan functions and types.

// Include generated bindings from build script
pub mod bindings {
    //! Generated Vulkan bindings from XML specification
    // Bring common C types into the module scope so generated bindings that
    // reference `c_void`, `c_char`, and similarly prefixed const-type names
    // can compile without requiring edits to the generated file. Prefer
    // `core::ffi` so this module works in `no_std` contexts as well as std.
    #![allow(
        non_camel_case_types,
        non_snake_case,
        non_upper_case_globals,
        dead_code,
        clippy::all
    )]

    use core::ffi::{c_char, c_ulong, c_void};

    include!(concat!(env!("OUT_DIR"), "/vulkan_bindings.rs"));
}

// Version definitions
pub mod version;

// VkResult -> Rust Result helpers
pub mod result;
pub use result::VkResultExt;

// Loader functionality
pub mod loader;
pub use loader::VulkanLibrary;

// pNext chain trait (implemented for every generated struct with
// sType/pNext head fields; impls are emitted by vulkan_gen).
pub mod pnext;
pub use pnext::PNextChainable;
