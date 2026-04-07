//! Runtime loader for the Vulkan shared library and per-scope dispatch tables.
//!
//! Vulkan function pointers are not statically linked. Instead, an application
//! loads them at runtime in three stages:
//!
//! 1. **Entry-level functions** (no instance needed): `vkCreateInstance`,
//!    `vkEnumerateInstanceVersion`, etc. These are loaded from
//!    `vkGetInstanceProcAddr(NULL, ...)`.
//! 2. **Instance-level functions**: everything that takes a `VkInstance` or
//!    `VkPhysicalDevice` as its first argument. Loaded from
//!    `vkGetInstanceProcAddr(instance, ...)` after `vkCreateInstance`.
//! 3. **Device-level functions**: everything that takes a `VkDevice`,
//!    `VkCommandBuffer`, or `VkQueue` as its first argument. Loaded from
//!    `vkGetDeviceProcAddr(device, ...)` after `vkCreateDevice`. Device-level
//!    dispatch is faster than instance-level dispatch because it skips a
//!    layer of internal indirection in the loader.
//!
//! Spock provides three generated dispatch table structs (one per stage),
//! each populated by [`VulkanLibrary`] from the running Vulkan implementation.
//! Every Vulkan command appears as an `Option<fn_ptr>` field — `None` if the
//! current driver doesn't expose it, `Some` if it can be called.
//!
//! # Typical usage
//!
//! ```no_run
//! use spock::raw::bindings::*;
//! use spock::raw::{VkResultExt, VulkanLibrary};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Stage 1: open vulkan-1.dll / libvulkan.so.1 / etc.
//! let library = VulkanLibrary::new()?;
//!
//! // Stage 2: load global functions
//! let entry = unsafe { library.load_entry() };
//!
//! // Create a VkInstance using the entry table
//! let info = VkInstanceCreateInfo::default();
//! let mut instance: VkInstance = std::ptr::null_mut();
//! unsafe { (entry.vkCreateInstance.unwrap())(&info, std::ptr::null(), &mut instance) }
//!     .into_result()?;
//!
//! // Stage 3: load instance functions
//! let inst = unsafe { library.load_instance(instance) };
//!
//! // Optionally, after creating a VkDevice:
//! //   let dev = unsafe { library.load_device(instance, device) };
//! # Ok(())
//! # }
//! ```
//!
//! All dispatch table contents are generated from `vk.xml`; no function names
//! are hardcoded in spock's source.

use crate::raw::bindings::*;
use std::ffi::c_void;
use std::sync::Arc;

/// Loads the Vulkan shared library and provides `vkGetInstanceProcAddr`.
///
/// This is the entry point for runtime function loading. After construction,
/// use [`load_entry`](Self::load_entry), [`load_instance`](Self::load_instance),
/// and [`load_device`](Self::load_device) to obtain the per-stage dispatch tables.
///
/// `VulkanLibrary` keeps the underlying shared library loaded as long as it
/// (or any clone of it) is alive. Cloning is cheap — the library handle is
/// shared via `Arc`.
pub struct VulkanLibrary {
    _library: Arc<libloading::Library>,
    get_instance_proc_addr: unsafe extern "system" fn(*mut c_void, *const i8) -> *mut c_void,
}

impl VulkanLibrary {
    /// Load the Vulkan runtime library.
    pub fn new() -> Result<Self, libloading::Error> {
        let library = unsafe {
            #[cfg(windows)]
            let lib = libloading::Library::new("vulkan-1.dll")?;
            #[cfg(unix)]
            let lib = libloading::Library::new("libvulkan.so.1")?;
            Arc::new(lib)
        };

        let get_instance_proc_addr = unsafe {
            *library.get::<unsafe extern "system" fn(*mut c_void, *const i8) -> *mut c_void>(
                b"vkGetInstanceProcAddr\0",
            )?
        };

        Ok(Self {
            _library: library,
            get_instance_proc_addr,
        })
    }

    /// Load global (entry-level) functions that don't require a VkInstance.
    pub unsafe fn load_entry(&self) -> VkEntryDispatchTable {
        let gipa = self.get_instance_proc_addr;
        unsafe {
            VkEntryDispatchTable::load(|name| {
                (gipa)(std::ptr::null_mut(), name.as_ptr() as *const i8)
            })
        }
    }

    /// Load instance-level functions for the given VkInstance.
    pub unsafe fn load_instance(&self, instance: VkInstance) -> VkInstanceDispatchTable {
        let gipa = self.get_instance_proc_addr;
        unsafe {
            VkInstanceDispatchTable::load(|name| {
                (gipa)(instance as *mut c_void, name.as_ptr() as *const i8)
            })
        }
    }

    /// Load device-level functions for the given VkDevice.
    ///
    /// `instance` is the VkInstance that owns the device — it is required because
    /// `vkGetDeviceProcAddr` is loaded via `vkGetInstanceProcAddr(instance, ...)`.
    pub unsafe fn load_device(
        &self,
        instance: VkInstance,
        device: VkDevice,
    ) -> VkDeviceDispatchTable {
        let gipa = self.get_instance_proc_addr;

        // First, get vkGetDeviceProcAddr via the instance loader.
        // Per the Vulkan spec, vkGetDeviceProcAddr must be loaded with a
        // valid VkInstance handle (not NULL).
        let gdpa_name = c"vkGetDeviceProcAddr";
        let gdpa_ptr = unsafe { (gipa)(instance as *mut c_void, gdpa_name.as_ptr() as *const i8) };

        if !gdpa_ptr.is_null() {
            // Use vkGetDeviceProcAddr for fastest device-level dispatch.
            let gdpa: unsafe extern "system" fn(*mut c_void, *const i8) -> *mut c_void =
                unsafe { std::mem::transmute(gdpa_ptr) };
            unsafe {
                VkDeviceDispatchTable::load(|name| {
                    (gdpa)(device as *mut c_void, name.as_ptr() as *const i8)
                })
            }
        } else {
            // Fallback: load via instance proc addr (slower, instance-level dispatch).
            unsafe {
                VkDeviceDispatchTable::load(|name| {
                    (gipa)(instance as *mut c_void, name.as_ptr() as *const i8)
                })
            }
        }
    }
}
