//! Safe wrapper for `VkSurfaceKHR` — the link between Vulkan and a
//! platform window system.
//!
//! Surfaces are platform-specific. The user creates one via the
//! constructor matching their windowing system:
//!
//! - [`Surface::from_win32`] — Win32 (Windows)
//! - [`Surface::from_wayland`] — Wayland (Linux)
//! - [`Surface::from_xlib`] — Xlib (Linux / *BSD)
//! - [`Surface::from_xcb`] — Xcb (Linux / *BSD)
//! - [`Surface::from_metal`] — Metal (macOS / iOS via MoltenVK)
//!
//! The surface itself does very little — its purpose is to be the
//! target of a [`Swapchain`](super::Swapchain), which is what actually
//! produces presentable images.
//!
//! ## Required instance extensions
//!
//! Creating any surface requires `VK_KHR_surface` to be enabled at
//! [`Instance`](super::Instance) creation time, plus the
//! platform-specific extension:
//!
//! - Win32: `VK_KHR_win32_surface`
//! - Wayland: `VK_KHR_wayland_surface`
//! - Xlib: `VK_KHR_xlib_surface`
//! - Xcb: `VK_KHR_xcb_surface`
//! - Metal: `VK_EXT_metal_surface`
//!
//! Use [`InstanceCreateInfo::enabled_extensions`](super::InstanceCreateInfo::enabled_extensions)
//! to enable them.

use super::instance::InstanceInner;
use super::physical::PhysicalDevice;
use super::{Error, Instance, Result, check};
use crate::raw::bindings::*;
use std::sync::Arc;

// Extension name string constants live in [`crate::raw::bindings`] —
// the code generator emits `KHR_SURFACE_EXTENSION_NAME` etc. from
// vk.xml, so we don't hand-maintain a duplicate set here. Prefer the
// generated `<vendor>_<ext>()` builder methods on
// [`InstanceExtensions`](super::InstanceExtensions) and
// [`DeviceExtensions`](super::DeviceExtensions) over reaching for the
// raw strings directly.

/// A safe wrapper around `VkSurfaceKHR`.
///
/// Surfaces are destroyed automatically on drop. They keep the parent
/// instance alive via an `Arc`.
pub struct Surface {
    pub(crate) handle: VkSurfaceKHR,
    pub(crate) instance: Arc<InstanceInner>,
}

// Safety: VkSurfaceKHR is a non-dispatchable handle, safe to share between
// threads. Creation/destruction must be externally synchronized; we
// only do those at construction and Drop.
unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

impl Surface {
    /// Create a `VkSurfaceKHR` from a Win32 `(HINSTANCE, HWND)` pair.
    ///
    /// # Safety
    ///
    /// Both `hinstance` and `hwnd` must be valid Win32 handles for the
    /// lifetime of the resulting `Surface`. The handles can be obtained
    /// from any Win32 window library (`winit`, raw Win32 API, etc.).
    pub unsafe fn from_win32(
        instance: &Instance,
        hinstance: *mut std::ffi::c_void,
        hwnd: *mut std::ffi::c_void,
    ) -> Result<Self> {
        let create = instance
            .inner
            .dispatch
            .vkCreateWin32SurfaceKHR
            .ok_or(Error::MissingFunction("vkCreateWin32SurfaceKHR"))?;

        let info = VkWin32SurfaceCreateInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
            hinstance,
            hwnd,
            ..Default::default()
        };

        let mut handle: VkSurfaceKHR = 0;
        // Safety: info is valid for the call; instance handle is valid;
        // the caller has guaranteed the Win32 handles are live.
        check(unsafe { create(instance.inner.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            instance: Arc::clone(&instance.inner),
        })
    }

    /// Create a `VkSurfaceKHR` from a Wayland `(wl_display, wl_surface)`
    /// pair.
    ///
    /// # Safety
    ///
    /// Both pointers must be valid for the lifetime of the resulting
    /// `Surface`. The pointers are reinterpreted via `as` casts to
    /// satisfy the bindings' opaque-pointer wrapping (the codegen
    /// treats Wayland's `wl_display` and `wl_surface` C types as
    /// already-pointer aliases, so the field type is `*mut *mut c_void`
    /// even though the value is just an opaque handle).
    pub unsafe fn from_wayland(
        instance: &Instance,
        display: *mut std::ffi::c_void,
        surface: *mut std::ffi::c_void,
    ) -> Result<Self> {
        let create = instance
            .inner
            .dispatch
            .vkCreateWaylandSurfaceKHR
            .ok_or(Error::MissingFunction("vkCreateWaylandSurfaceKHR"))?;

        let info = VkWaylandSurfaceCreateInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
            display: display as *mut _,
            surface: surface as *mut _,
            ..Default::default()
        };

        let mut handle: VkSurfaceKHR = 0;
        // Safety: info is valid; caller has guaranteed pointer validity.
        check(unsafe { create(instance.inner.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            instance: Arc::clone(&instance.inner),
        })
    }

    /// Create a `VkSurfaceKHR` from an Xlib `(Display*, Window)` pair.
    ///
    /// `display` is a pointer to an `Xlib` `Display` connection
    /// (typically obtained from `XOpenDisplay`). `window` is the X11
    /// window XID. The bindings generator emits `Window` as
    /// `c_ulong`, which matches the C ABI on every platform that
    /// implements Xlib.
    ///
    /// # Safety
    ///
    /// `display` must be a valid Xlib `Display*` for the lifetime of
    /// the resulting `Surface`. `window` must be a valid X11 window
    /// XID belonging to that display.
    pub unsafe fn from_xlib(
        instance: &Instance,
        display: *mut std::ffi::c_void,
        window: std::ffi::c_ulong,
    ) -> Result<Self> {
        let create = instance
            .inner
            .dispatch
            .vkCreateXlibSurfaceKHR
            .ok_or(Error::MissingFunction("vkCreateXlibSurfaceKHR"))?;

        let info = VkXlibSurfaceCreateInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
            // The bindings type the dpy field as `*mut Display` where
            // Display is itself an opaque pointer alias, so the field
            // type is `*mut *mut c_void`. Cast through the alias.
            dpy: display as *mut _,
            window,
            ..Default::default()
        };

        let mut handle: VkSurfaceKHR = 0;
        // Safety: info is valid for the call; instance handle is valid;
        // the caller has guaranteed display + window are live.
        check(unsafe { create(instance.inner.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            instance: Arc::clone(&instance.inner),
        })
    }

    /// Create a `VkSurfaceKHR` from an Xcb `(xcb_connection_t*, xcb_window_t)`
    /// pair.
    ///
    /// `connection` is a pointer to an `xcb_connection_t` (typically
    /// from `xcb_connect`). `window` is an `xcb_window_t` XID, which is
    /// a `uint32_t` per the XCB ABI.
    ///
    /// # Safety
    ///
    /// `connection` must be a valid `xcb_connection_t*` for the lifetime
    /// of the resulting `Surface`. `window` must be a valid window XID on
    /// that connection.
    pub unsafe fn from_xcb(
        instance: &Instance,
        connection: *mut std::ffi::c_void,
        window: u32,
    ) -> Result<Self> {
        let create = instance
            .inner
            .dispatch
            .vkCreateXcbSurfaceKHR
            .ok_or(Error::MissingFunction("vkCreateXcbSurfaceKHR"))?;

        let info = VkXcbSurfaceCreateInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
            // Same alias-cast trick as Xlib.
            connection: connection as *mut _,
            window,
            ..Default::default()
        };

        let mut handle: VkSurfaceKHR = 0;
        // Safety: info is valid for the call; instance handle is valid;
        // the caller has guaranteed connection + window are live.
        check(unsafe { create(instance.inner.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            instance: Arc::clone(&instance.inner),
        })
    }

    /// Create a `VkSurfaceKHR` from a `CAMetalLayer*` (macOS / iOS via
    /// MoltenVK).
    ///
    /// # Safety
    ///
    /// `metal_layer` must be a valid pointer to a `CAMetalLayer` for
    /// the lifetime of the resulting `Surface`.
    pub unsafe fn from_metal(
        instance: &Instance,
        metal_layer: *const std::ffi::c_void,
    ) -> Result<Self> {
        let create = instance
            .inner
            .dispatch
            .vkCreateMetalSurfaceEXT
            .ok_or(Error::MissingFunction("vkCreateMetalSurfaceEXT"))?;

        let info = VkMetalSurfaceCreateInfoEXT {
            sType: VkStructureType::STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
            pLayer: metal_layer as *const _,
            ..Default::default()
        };

        let mut handle: VkSurfaceKHR = 0;
        // Safety: info is valid; caller has guaranteed layer validity.
        check(unsafe { create(instance.inner.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            instance: Arc::clone(&instance.inner),
        })
    }

    /// Returns the raw `VkSurfaceKHR` handle.
    pub fn raw(&self) -> VkSurfaceKHR {
        self.handle
    }

    /// Query whether the given physical device + queue family combination
    /// can present to this surface.
    pub fn supports_present(&self, physical: &PhysicalDevice, queue_family: u32) -> bool {
        let Some(get) = self.instance.dispatch.vkGetPhysicalDeviceSurfaceSupportKHR else {
            return false;
        };
        let mut supported: VkBool32 = 0;
        // Safety: physical handle and self.handle are both valid.
        let res = unsafe { get(physical.raw(), queue_family, self.handle, &mut supported) };
        res == VkResult::SUCCESS && supported != 0
    }

    /// Query the surface capabilities for the given physical device.
    pub fn capabilities(&self, physical: &PhysicalDevice) -> Result<SurfaceCapabilities> {
        let get = self
            .instance
            .dispatch
            .vkGetPhysicalDeviceSurfaceCapabilitiesKHR
            .ok_or(Error::MissingFunction(
                "vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
            ))?;
        let mut raw: VkSurfaceCapabilitiesKHR = unsafe { std::mem::zeroed() };
        // Safety: physical handle and self.handle are both valid.
        check(unsafe { get(physical.raw(), self.handle, &mut raw) })?;
        Ok(SurfaceCapabilities { raw })
    }

    /// Enumerate the (format, color space) pairs the given physical
    /// device supports for this surface.
    pub fn formats(&self, physical: &PhysicalDevice) -> Result<Vec<SurfaceFormat>> {
        let get = self
            .instance
            .dispatch
            .vkGetPhysicalDeviceSurfaceFormatsKHR
            .ok_or(Error::MissingFunction(
                "vkGetPhysicalDeviceSurfaceFormatsKHR",
            ))?;
        let mut count: u32 = 0;
        // Safety: physical handle and self.handle are both valid.
        check(unsafe {
            get(
                physical.raw(),
                self.handle,
                &mut count,
                std::ptr::null_mut(),
            )
        })?;
        let mut raw: Vec<VkSurfaceFormatKHR> = vec![unsafe { std::mem::zeroed() }; count as usize];
        // Safety: raw has space for `count` elements.
        check(unsafe { get(physical.raw(), self.handle, &mut count, raw.as_mut_ptr()) })?;
        Ok(raw.into_iter().map(|r| SurfaceFormat { raw: r }).collect())
    }

    /// Enumerate the present modes the given physical device supports
    /// for this surface.
    pub fn present_modes(&self, physical: &PhysicalDevice) -> Result<Vec<PresentMode>> {
        let get = self
            .instance
            .dispatch
            .vkGetPhysicalDeviceSurfacePresentModesKHR
            .ok_or(Error::MissingFunction(
                "vkGetPhysicalDeviceSurfacePresentModesKHR",
            ))?;
        let mut count: u32 = 0;
        // Safety: physical handle and self.handle are both valid.
        check(unsafe {
            get(
                physical.raw(),
                self.handle,
                &mut count,
                std::ptr::null_mut(),
            )
        })?;
        let mut raw: Vec<VkPresentModeKHR> =
            vec![VkPresentModeKHR::PRESENT_MODE_FIFO_KHR; count as usize];
        // Safety: raw has space for `count` elements.
        check(unsafe { get(physical.raw(), self.handle, &mut count, raw.as_mut_ptr()) })?;
        Ok(raw.into_iter().map(PresentMode).collect())
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        if let Some(destroy) = self.instance.dispatch.vkDestroySurfaceKHR {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.instance.handle, self.handle, std::ptr::null()) };
        }
    }
}

/// Surface capability snapshot returned by [`Surface::capabilities`].
#[derive(Clone)]
pub struct SurfaceCapabilities {
    pub(crate) raw: VkSurfaceCapabilitiesKHR,
}

impl std::fmt::Debug for SurfaceCapabilities {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurfaceCapabilities")
            .field("min_image_count", &self.min_image_count())
            .field("max_image_count", &self.max_image_count())
            .field("current_extent", &self.current_extent())
            .field("min_image_extent", &self.min_image_extent())
            .field("max_image_extent", &self.max_image_extent())
            .field("supported_usage_flags", &self.supported_usage_flags())
            .field("supported_transforms", &self.supported_transforms())
            .field("current_transform", &self.current_transform())
            .field(
                "supported_composite_alpha",
                &self.supported_composite_alpha(),
            )
            .finish()
    }
}

impl SurfaceCapabilities {
    pub fn min_image_count(&self) -> u32 {
        self.raw.minImageCount
    }
    pub fn max_image_count(&self) -> u32 {
        self.raw.maxImageCount
    }
    pub fn current_extent(&self) -> (u32, u32) {
        (self.raw.currentExtent.width, self.raw.currentExtent.height)
    }
    pub fn min_image_extent(&self) -> (u32, u32) {
        (
            self.raw.minImageExtent.width,
            self.raw.minImageExtent.height,
        )
    }
    pub fn max_image_extent(&self) -> (u32, u32) {
        (
            self.raw.maxImageExtent.width,
            self.raw.maxImageExtent.height,
        )
    }
    pub fn supported_usage_flags(&self) -> u32 {
        self.raw.supportedUsageFlags
    }
    pub fn supported_transforms(&self) -> u32 {
        self.raw.supportedTransforms
    }
    pub fn current_transform(&self) -> u32 {
        self.raw.currentTransform
    }
    pub fn supported_composite_alpha(&self) -> u32 {
        self.raw.supportedCompositeAlpha
    }
}

/// One supported `(format, color_space)` pair returned by
/// [`Surface::formats`].
#[derive(Clone, Copy)]
pub struct SurfaceFormat {
    pub(crate) raw: VkSurfaceFormatKHR,
}

impl std::fmt::Debug for SurfaceFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurfaceFormat")
            .field("format", &self.format())
            .field("color_space", &self.color_space())
            .finish()
    }
}

impl SurfaceFormat {
    pub fn format(&self) -> super::Format {
        super::Format(self.raw.format)
    }
    pub fn color_space(&self) -> VkColorSpaceKHR {
        self.raw.colorSpace
    }
}

/// One supported present mode returned by [`Surface::present_modes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentMode(pub VkPresentModeKHR);

impl PresentMode {
    /// Vsync, double-buffered. Always supported.
    pub const FIFO: Self = Self(VkPresentModeKHR::PRESENT_MODE_FIFO_KHR);
    /// Vsync but allows tearing if the application falls behind.
    pub const FIFO_RELAXED: Self = Self(VkPresentModeKHR::PRESENT_MODE_FIFO_RELAXED_KHR);
    /// No vsync — render as fast as possible, may tear.
    pub const IMMEDIATE: Self = Self(VkPresentModeKHR::PRESENT_MODE_IMMEDIATE_KHR);
    /// Triple-buffered: latest frame replaces the queued one.
    pub const MAILBOX: Self = Self(VkPresentModeKHR::PRESENT_MODE_MAILBOX_KHR);
}
