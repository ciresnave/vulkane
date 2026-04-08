//! Safe wrapper for `VkInstance`.

use super::debug::{
    DebugCallback, DebugMessageSeverity, DebugMessageType, RealDebugCallbackFn, default_callback,
    trampoline,
};
use super::{Error, PhysicalDevice, Result, check};
use crate::raw::VulkanLibrary;
use crate::raw::bindings::*;
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::Arc;

/// A Vulkan API version, encoded as a `u32` per the Vulkan spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApiVersion(pub u32);

impl ApiVersion {
    /// `VK_API_VERSION_1_0`
    pub const V1_0: Self = Self(VK_API_VERSION_1_0);
    /// `VK_API_VERSION_1_1`
    pub const V1_1: Self = Self(VK_API_VERSION_1_1);
    /// `VK_API_VERSION_1_2`
    pub const V1_2: Self = Self(VK_API_VERSION_1_2);
    /// `VK_API_VERSION_1_3`
    pub const V1_3: Self = Self(VK_API_VERSION_1_3);
    /// `VK_API_VERSION_1_4`
    pub const V1_4: Self = Self(VK_API_VERSION_1_4);

    /// Construct a custom version using `vk_make_api_version`.
    pub const fn new(variant: u32, major: u32, minor: u32, patch: u32) -> Self {
        // Reproduce the bit-packing here so this is a const fn.
        // Same formula as the generated `vk_make_api_version`.
        Self((variant << 29) | (major << 22) | (minor << 12) | patch)
    }

    /// Extract the major version component.
    pub const fn major(self) -> u32 {
        (self.0 >> 22) & 0x7F
    }

    /// Extract the minor version component.
    pub const fn minor(self) -> u32 {
        (self.0 >> 12) & 0x3FF
    }

    /// Extract the patch version component.
    pub const fn patch(self) -> u32 {
        self.0 & 0xFFF
    }
}

impl std::fmt::Display for ApiVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major(), self.minor(), self.patch())
    }
}

impl Default for ApiVersion {
    fn default() -> Self {
        Self::V1_0
    }
}

/// The canonical name of the Khronos validation layer.
pub const KHRONOS_VALIDATION_LAYER: &str = "VK_LAYER_KHRONOS_validation";
/// The canonical name of the debug-utils instance extension.
pub const DEBUG_UTILS_EXTENSION: &str = "VK_EXT_debug_utils";

/// Parameters for [`Instance::new`].
///
/// Most fields can be left at their defaults. Use the
/// [`validation`](Self::validation) helper to enable the Khronos validation
/// layer and the debug-utils extension in one go.
pub struct InstanceCreateInfo<'a> {
    /// Application name (will be NUL-terminated for the C API).
    pub application_name: Option<&'a str>,
    /// Application version.
    pub application_version: ApiVersion,
    /// Engine name.
    pub engine_name: Option<&'a str>,
    /// Engine version.
    pub engine_version: ApiVersion,
    /// Vulkan API version the application targets.
    pub api_version: ApiVersion,
    /// Names of layers to enable. Each must be a layer that
    /// [`Instance::enumerate_layer_properties`] reports as available.
    pub enabled_layers: &'a [&'a str],
    /// Names of instance extensions to enable. Each must be an extension that
    /// [`Instance::enumerate_extension_properties`] reports as available.
    pub enabled_extensions: &'a [&'a str],
    /// Optional debug-utils callback. If `Some`, the resulting [`Instance`]
    /// will hold a `VkDebugUtilsMessengerEXT` that delivers messages to the
    /// callback. The `VK_EXT_debug_utils` extension *must* be present in
    /// [`enabled_extensions`](Self::enabled_extensions) for this to take
    /// effect — if the extension isn't enabled the callback is silently
    /// ignored at instance creation time.
    pub debug_callback: Option<Box<DebugCallback>>,
}

impl<'a> Default for InstanceCreateInfo<'a> {
    fn default() -> Self {
        Self {
            application_name: None,
            application_version: ApiVersion::V1_0,
            engine_name: None,
            engine_version: ApiVersion::V1_0,
            api_version: ApiVersion::V1_0,
            enabled_layers: &[],
            enabled_extensions: &[],
            debug_callback: None,
        }
    }
}

impl<'a> InstanceCreateInfo<'a> {
    /// Convenience that returns a [`InstanceCreateInfo`] preconfigured for
    /// validation: enables [`KHRONOS_VALIDATION_LAYER`] and
    /// [`DEBUG_UTILS_EXTENSION`], and installs a default `eprintln!`-based
    /// callback that prints WARNING and ERROR messages to stderr.
    ///
    /// Pair with `..InstanceCreateInfo::default()` if you also want to set
    /// other fields:
    ///
    /// ```ignore
    /// let info = InstanceCreateInfo {
    ///     application_name: Some("my-app"),
    ///     ..InstanceCreateInfo::validation()
    /// };
    /// ```
    pub fn validation() -> Self {
        Self {
            enabled_layers: &[KHRONOS_VALIDATION_LAYER],
            enabled_extensions: &[DEBUG_UTILS_EXTENSION],
            debug_callback: Some(default_callback()),
            ..Self::default()
        }
    }
}

impl<'a> std::fmt::Debug for InstanceCreateInfo<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstanceCreateInfo")
            .field("application_name", &self.application_name)
            .field("application_version", &self.application_version)
            .field("engine_name", &self.engine_name)
            .field("engine_version", &self.engine_version)
            .field("api_version", &self.api_version)
            .field("enabled_layers", &self.enabled_layers)
            .field("enabled_extensions", &self.enabled_extensions)
            .field("debug_callback", &self.debug_callback.is_some())
            .finish()
    }
}

/// Internal state shared between [`Instance`] and its child handles.
///
/// Lives inside an `Arc` so child handles can keep the instance alive even
/// after the user drops their `Instance` clone.
pub(crate) struct InstanceInner {
    pub(crate) library: VulkanLibrary,
    pub(crate) handle: VkInstance,
    pub(crate) dispatch: VkInstanceDispatchTable,
    /// `VkDebugUtilsMessengerEXT` handle, if a debug callback was registered.
    /// Stored as `u64` because `VkDebugUtilsMessengerEXT` is a non-dispatchable
    /// handle (`u64`).
    debug_messenger: u64,
    /// The leaked `Box<Box<DebugCallback>>` whose pointer we passed as
    /// `pUserData`. We hold it here so the trampoline can keep dereferencing
    /// it and so `Drop` can free it after destroying the messenger.
    debug_callback_box: *mut Box<DebugCallback>,
}

// Safety: VkInstance is documented by the Vulkan spec as safe to share
// between threads. Individual function calls have their own external
// synchronization requirements (which are the user's responsibility), but
// the handle itself is thread-safe to access. The dispatch table contains
// only function pointers which are also Send + Sync. The
// `*mut Box<DebugCallback>` is sound to share because the box itself owns
// a `Send + Sync` callback (enforced by the `DebugCallback` type alias) and
// the pointer is only ever dereferenced from the trampoline, which Vulkan
// already serializes around messenger destruction.
unsafe impl Send for InstanceInner {}
unsafe impl Sync for InstanceInner {}

impl Drop for InstanceInner {
    fn drop(&mut self) {
        // Destroy the debug messenger before the instance.
        if self.debug_messenger != 0
            && let Some(destroy) = self.dispatch.vkDestroyDebugUtilsMessengerEXT
        {
            // Safety: handle is valid (we created it), instance is still alive.
            unsafe { destroy(self.handle, self.debug_messenger, std::ptr::null()) };
        }
        if let Some(destroy) = self.dispatch.vkDestroyInstance {
            // Safety: handle is valid (constructed by Instance::new), and
            // by the Arc invariant we are the last owner.
            unsafe { destroy(self.handle, std::ptr::null()) };
        }
        // Free the leaked callback box now that no one will read it.
        if !self.debug_callback_box.is_null() {
            // Safety: we leaked this Box ourselves at instance creation.
            drop(unsafe { Box::from_raw(self.debug_callback_box) });
        }
    }
}

/// Properties of one available instance layer.
#[derive(Clone)]
pub struct LayerProperties {
    raw: VkLayerProperties,
}

impl LayerProperties {
    pub fn name(&self) -> String {
        // Safety: layerName is a NUL-terminated array per spec.
        unsafe { CStr::from_ptr(self.raw.layerName.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }
    pub fn description(&self) -> String {
        unsafe { CStr::from_ptr(self.raw.description.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }
    pub fn spec_version(&self) -> ApiVersion {
        ApiVersion(self.raw.specVersion)
    }
    pub fn implementation_version(&self) -> u32 {
        self.raw.implementationVersion
    }
}

impl std::fmt::Debug for LayerProperties {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerProperties")
            .field("name", &self.name())
            .field("spec_version", &self.spec_version())
            .field("implementation_version", &self.implementation_version())
            .finish()
    }
}

/// Properties of one available instance or device extension.
#[derive(Clone)]
pub struct ExtensionProperties {
    raw: VkExtensionProperties,
}

impl ExtensionProperties {
    pub(crate) fn from_raw(raw: VkExtensionProperties) -> Self {
        Self { raw }
    }
    pub fn name(&self) -> String {
        unsafe { CStr::from_ptr(self.raw.extensionName.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }
    pub fn spec_version(&self) -> u32 {
        self.raw.specVersion
    }
}

impl std::fmt::Debug for ExtensionProperties {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionProperties")
            .field("name", &self.name())
            .field("spec_version", &self.spec_version())
            .finish()
    }
}

/// A safe wrapper around `VkInstance`.
///
/// The instance is destroyed automatically when the last `Instance` clone
/// (and the last child handle that holds an `Arc<InstanceInner>`) is dropped.
#[derive(Clone)]
pub struct Instance {
    pub(crate) inner: Arc<InstanceInner>,
}

impl Instance {
    /// Enumerate the layers that the running Vulkan implementation makes
    /// available. Useful for checking that the validation layer is installed
    /// before requesting it.
    pub fn enumerate_layer_properties() -> Result<Vec<LayerProperties>> {
        let library = VulkanLibrary::new()?;
        // Safety: just-loaded library has a valid entry table.
        let entry = unsafe { library.load_entry() };
        let enumerate = entry
            .vkEnumerateInstanceLayerProperties
            .ok_or(Error::MissingFunction("vkEnumerateInstanceLayerProperties"))?;

        let mut count: u32 = 0;
        // Safety: count query, output ptr is null.
        check(unsafe { enumerate(&mut count, std::ptr::null_mut()) })?;
        let mut raw: Vec<VkLayerProperties> = vec![unsafe { std::mem::zeroed() }; count as usize];
        // Safety: raw has space for `count` elements.
        check(unsafe { enumerate(&mut count, raw.as_mut_ptr()) })?;
        Ok(raw
            .into_iter()
            .map(|r| LayerProperties { raw: r })
            .collect())
    }

    /// Enumerate the instance-level extensions exposed by the running Vulkan
    /// implementation. Useful for checking that an extension (e.g.
    /// `VK_EXT_debug_utils`) is available before enabling it.
    pub fn enumerate_extension_properties() -> Result<Vec<ExtensionProperties>> {
        let library = VulkanLibrary::new()?;
        // Safety: just-loaded library has a valid entry table.
        let entry = unsafe { library.load_entry() };
        let enumerate =
            entry
                .vkEnumerateInstanceExtensionProperties
                .ok_or(Error::MissingFunction(
                    "vkEnumerateInstanceExtensionProperties",
                ))?;

        let mut count: u32 = 0;
        // Safety: count query, output ptr is null. Layer name null = core extensions.
        check(unsafe { enumerate(std::ptr::null(), &mut count, std::ptr::null_mut()) })?;
        let mut raw: Vec<VkExtensionProperties> =
            vec![unsafe { std::mem::zeroed() }; count as usize];
        // Safety: raw has space for `count` elements.
        check(unsafe { enumerate(std::ptr::null(), &mut count, raw.as_mut_ptr()) })?;
        Ok(raw
            .into_iter()
            .map(|r| ExtensionProperties { raw: r })
            .collect())
    }

    /// Load the Vulkan library and create a new `VkInstance`.
    pub fn new(info: InstanceCreateInfo<'_>) -> Result<Self> {
        let library = VulkanLibrary::new()?;

        // Convert the optional application/engine name strings to CStrings
        // so we can keep them alive across the call.
        let app_name_c = info.application_name.map(CString::new).transpose()?;
        let engine_name_c = info.engine_name.map(CString::new).transpose()?;

        // Build owned CString vectors for layer / extension names so the
        // pointers we pass to Vulkan stay valid for the duration of the call.
        let layer_cstrings: Vec<CString> = info
            .enabled_layers
            .iter()
            .map(|s| CString::new(*s))
            .collect::<std::result::Result<_, _>>()?;
        let ext_cstrings: Vec<CString> = info
            .enabled_extensions
            .iter()
            .map(|s| CString::new(*s))
            .collect::<std::result::Result<_, _>>()?;

        // Vulkan's bindings declare these as `*const *mut c_char` (rather
        // than the const-correct `*const *const c_char`); cast through.
        let layer_ptrs: Vec<*mut c_char> = layer_cstrings
            .iter()
            .map(|s| s.as_ptr() as *mut c_char)
            .collect();
        let ext_ptrs: Vec<*mut c_char> = ext_cstrings
            .iter()
            .map(|s| s.as_ptr() as *mut c_char)
            .collect();

        let app_info = VkApplicationInfo {
            sType: VkStructureType::STRUCTURE_TYPE_APPLICATION_INFO,
            pNext: std::ptr::null(),
            pApplicationName: app_name_c.as_deref().map_or(std::ptr::null(), CStr::as_ptr),
            applicationVersion: info.application_version.0,
            pEngineName: engine_name_c
                .as_deref()
                .map_or(std::ptr::null(), CStr::as_ptr),
            engineVersion: info.engine_version.0,
            apiVersion: info.api_version.0,
        };

        let create_info = VkInstanceCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo: &app_info,
            enabledLayerCount: layer_ptrs.len() as u32,
            ppEnabledLayerNames: if layer_ptrs.is_empty() {
                std::ptr::null()
            } else {
                layer_ptrs.as_ptr()
            },
            enabledExtensionCount: ext_ptrs.len() as u32,
            ppEnabledExtensionNames: if ext_ptrs.is_empty() {
                std::ptr::null()
            } else {
                ext_ptrs.as_ptr()
            },
            ..Default::default()
        };

        // Load entry-level functions to get vkCreateInstance.
        // Safety: VulkanLibrary just loaded successfully, the entry table
        // points at valid driver functions.
        let entry = unsafe { library.load_entry() };
        let create = entry
            .vkCreateInstance
            .ok_or(Error::MissingFunction("vkCreateInstance"))?;

        let mut handle: VkInstance = std::ptr::null_mut();
        // Safety: create_info is valid for the duration of the call. All
        // pointee data (app_info, name CStrings, layer/ext name vectors) live
        // until end of scope.
        let result = unsafe { create(&create_info, std::ptr::null(), &mut handle) };
        check(result)?;

        // Now that we have a real VkInstance, load the instance dispatch table.
        // Safety: handle is the freshly-created valid instance.
        let dispatch = unsafe { library.load_instance(handle) };

        // If the user supplied a debug callback AND the extension was
        // enabled (so vkCreateDebugUtilsMessengerEXT is loadable), register
        // it. Otherwise leave debug_messenger = 0.
        let mut debug_messenger: u64 = 0;
        let mut debug_callback_box: *mut Box<DebugCallback> = std::ptr::null_mut();

        if let Some(cb) = info.debug_callback
            && let Some(create_msgr) = dispatch.vkCreateDebugUtilsMessengerEXT
        {
            // Leak the callback box so the trampoline has a stable pointer.
            // Inner Box gives us a thin pointer to the dyn-trait fat box.
            let leaked: *mut Box<DebugCallback> = Box::into_raw(Box::new(cb));
            debug_callback_box = leaked;

            let real: RealDebugCallbackFn = trampoline;
            // Safety: the layouts of `RealDebugCallbackFn` and the
            // generated `PFN_vkDebugUtilsMessengerCallbackEXT` are both
            // `Option<unsafe extern "system" fn(...)>` and the C ABI is the
            // same — the generated typedef just has an empty argument list
            // because the codegen doesn't parse function pointer arguments.
            let pfn = unsafe {
                std::mem::transmute::<
                    Option<RealDebugCallbackFn>,
                    PFN_vkDebugUtilsMessengerCallbackEXT,
                >(Some(real))
            };

            let create_msgr_info = VkDebugUtilsMessengerCreateInfoEXT {
                sType: VkStructureType::STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                pNext: std::ptr::null(),
                flags: 0,
                messageSeverity: DebugMessageSeverity::ALL.0,
                messageType: DebugMessageType::ALL.0,
                pfnUserCallback: pfn,
                pUserData: leaked as *mut c_void,
            };

            // Safety: create_msgr_info and the leaked box outlive this call;
            // handle is the just-created valid instance.
            let res = unsafe {
                create_msgr(
                    handle,
                    &create_msgr_info,
                    std::ptr::null(),
                    &mut debug_messenger,
                )
            };
            // If creating the messenger failed for some reason (e.g. the
            // user enabled the extension but the layer rejected the call),
            // free the leaked box and surface the error.
            if let Err(e) = check(res) {
                // Drop the leaked box first.
                let _ = unsafe { Box::from_raw(leaked) };
                // Then destroy the instance and bail.
                if let Some(destroy) = dispatch.vkDestroyInstance {
                    unsafe { destroy(handle, std::ptr::null()) };
                }
                return Err(e);
            }
        }

        Ok(Self {
            inner: Arc::new(InstanceInner {
                library,
                handle,
                dispatch,
                debug_messenger,
                debug_callback_box,
            }),
        })
    }

    /// Returns the raw `VkInstance` handle.
    ///
    /// # Safety
    ///
    /// The caller must not call `vkDestroyInstance` on the returned handle —
    /// the safe wrapper owns its lifetime and will destroy it on drop.
    pub fn raw(&self) -> VkInstance {
        self.inner.handle
    }

    /// Enumerate the physical devices visible to this instance.
    pub fn enumerate_physical_devices(&self) -> Result<Vec<PhysicalDevice>> {
        let enumerate = self
            .inner
            .dispatch
            .vkEnumeratePhysicalDevices
            .ok_or(Error::MissingFunction("vkEnumeratePhysicalDevices"))?;

        let mut count: u32 = 0;
        // Safety: count query — the device handles array pointer is null.
        check(unsafe { enumerate(self.inner.handle, &mut count, std::ptr::null_mut()) })?;
        let mut handles: Vec<VkPhysicalDevice> = vec![std::ptr::null_mut(); count as usize];
        // Safety: handles has space for `count` elements.
        check(unsafe { enumerate(self.inner.handle, &mut count, handles.as_mut_ptr()) })?;

        Ok(handles
            .into_iter()
            .map(|h| PhysicalDevice::new(Arc::clone(&self.inner), h))
            .collect())
    }
}
