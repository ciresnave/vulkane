//! Safe wrapper for `VkSwapchainKHR` and the present loop.
//!
//! A swapchain is a queue of presentable images managed by the
//! windowing system. The application:
//!
//! 1. Calls [`Swapchain::acquire_next_image`] to get the index of the
//!    next image it should render into. The call signals the supplied
//!    semaphore (and/or fence) when the image is actually free.
//! 2. Records a command buffer that renders into that image.
//! 3. Submits the command buffer with a wait on the acquire semaphore
//!    and a signal on a "render finished" semaphore.
//! 4. Calls [`Swapchain::present`] to push the rendered image to the
//!    surface, waiting on the render-finished semaphore.
//!
//! Each acquire/present cycle uses a fresh pair of semaphores so the
//! pipeline can stay full while the user is rendering frame N+1.

use super::device::DeviceInner;
use super::image::{ImageUsage, ImageView};
use super::physical::PhysicalDevice;
use super::surface::{PresentMode, Surface};
use super::sync::Semaphore;
use super::{Device, Error, Format, Queue, Result, check};
use crate::raw::bindings::*;
use std::sync::Arc;

/// Parameters for [`Swapchain::new`]. The defaults pick reasonable
/// values for a "draw to a window with vsync" use case.
#[derive(Debug, Clone, Copy)]
pub struct SwapchainCreateInfo {
    pub format: Format,
    pub color_space: VkColorSpaceKHR,
    pub width: u32,
    pub height: u32,
    pub min_image_count: u32,
    pub image_usage: ImageUsage,
    pub present_mode: PresentMode,
    /// `true` to allow the windowing system to clip pixels not visible
    /// on screen.
    pub clipped: bool,
}

impl Default for SwapchainCreateInfo {
    fn default() -> Self {
        Self {
            format: Format::B8G8R8A8_UNORM,
            color_space: VkColorSpaceKHR::COLOR_SPACE_SRGB_NONLINEAR_KHR,
            width: 800,
            height: 600,
            min_image_count: 2,
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            present_mode: PresentMode::FIFO,
            clipped: true,
        }
    }
}

/// A safe wrapper around `VkSwapchainKHR`.
///
/// Swapchains are destroyed automatically on drop. Their underlying
/// images are owned by the implementation; the safe wrapper exposes
/// them via [`Swapchain::image_views`] which creates one
/// [`ImageView`] per swapchain image.
pub struct Swapchain {
    pub(crate) handle: VkSwapchainKHR,
    pub(crate) device: Arc<DeviceInner>,
    /// Keep the surface alive for the swapchain's lifetime.
    #[allow(dead_code)]
    pub(crate) surface: Arc<super::instance::InstanceInner>,
    pub(crate) format: Format,
    pub(crate) extent: (u32, u32),
    /// Raw image handles owned by the implementation. Not destroyed by
    /// the safe wrapper — they belong to the swapchain.
    pub(crate) images: Vec<VkImage>,
}

impl Swapchain {
    /// Create a swapchain on the given surface.
    pub fn new(device: &Device, surface: &Surface, info: SwapchainCreateInfo) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreateSwapchainKHR
            .ok_or(Error::MissingFunction("vkCreateSwapchainKHR"))?;

        let create_info = VkSwapchainCreateInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            surface: surface.handle,
            minImageCount: info.min_image_count,
            imageFormat: info.format.0,
            imageColorSpace: info.color_space,
            imageExtent: VkExtent2D {
                width: info.width,
                height: info.height,
            },
            imageArrayLayers: 1,
            imageUsage: info.image_usage.0,
            imageSharingMode: VkSharingMode::SHARING_MODE_EXCLUSIVE,
            preTransform: SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
            compositeAlpha: COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode: info.present_mode.0,
            clipped: if info.clipped { 1 } else { 0 },
            oldSwapchain: 0,
            ..Default::default()
        };

        let mut handle: VkSwapchainKHR = 0;
        // Safety: create_info is valid for the call.
        check(unsafe {
            create(
                device.inner.handle,
                &create_info,
                std::ptr::null(),
                &mut handle,
            )
        })?;

        // Fetch the swapchain images.
        let get_images = device
            .inner
            .dispatch
            .vkGetSwapchainImagesKHR
            .ok_or(Error::MissingFunction("vkGetSwapchainImagesKHR"))?;
        let mut count: u32 = 0;
        check(unsafe {
            get_images(
                device.inner.handle,
                handle,
                &mut count,
                std::ptr::null_mut(),
            )
        })?;
        let mut images: Vec<VkImage> = vec![0; count as usize];
        check(unsafe { get_images(device.inner.handle, handle, &mut count, images.as_mut_ptr()) })?;

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
            surface: Arc::clone(&surface.instance),
            format: info.format,
            extent: (info.width, info.height),
            images,
        })
    }

    /// Returns the raw `VkSwapchainKHR` handle.
    pub fn raw(&self) -> VkSwapchainKHR {
        self.handle
    }

    /// Returns the format the swapchain images use.
    pub fn format(&self) -> Format {
        self.format
    }

    /// Returns the swapchain extent (width, height).
    pub fn extent(&self) -> (u32, u32) {
        self.extent
    }

    /// Returns the number of swapchain images.
    pub fn image_count(&self) -> u32 {
        self.images.len() as u32
    }

    /// Create one [`ImageView`] for each swapchain image. Use these as
    /// the color attachments of your per-swapchain-image framebuffers.
    pub fn image_views(&self) -> Result<Vec<ImageView>> {
        let create = self
            .device
            .dispatch
            .vkCreateImageView
            .ok_or(Error::MissingFunction("vkCreateImageView"))?;

        let mut views = Vec::with_capacity(self.images.len());
        for &image in &self.images {
            let info = VkImageViewCreateInfo {
                sType: VkStructureType::STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                image,
                viewType: VkImageViewType::IMAGE_VIEW_TYPE_2D,
                format: self.format.0,
                components: VkComponentMapping {
                    r: VkComponentSwizzle::COMPONENT_SWIZZLE_IDENTITY,
                    g: VkComponentSwizzle::COMPONENT_SWIZZLE_IDENTITY,
                    b: VkComponentSwizzle::COMPONENT_SWIZZLE_IDENTITY,
                    a: VkComponentSwizzle::COMPONENT_SWIZZLE_IDENTITY,
                },
                subresourceRange: VkImageSubresourceRange {
                    aspectMask: IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel: 0,
                    levelCount: 1,
                    baseArrayLayer: 0,
                    layerCount: 1,
                },
                ..Default::default()
            };
            let mut handle: VkImageView = 0;
            // Safety: info is valid for the call; image is owned by the
            // swapchain and remains valid for our lifetime.
            check(unsafe { create(self.device.handle, &info, std::ptr::null(), &mut handle) })?;
            views.push(ImageView {
                handle,
                device: Arc::clone(&self.device),
            });
        }
        Ok(views)
    }

    /// Acquire the index of the next swapchain image, signaling
    /// `signal_semaphore` (and/or `signal_fence`) when the image is
    /// actually free.
    ///
    /// Returns the image index. Pass this to your framebuffer-per-image
    /// table and to [`present`](Self::present).
    pub fn acquire_next_image(
        &self,
        timeout_nanos: u64,
        signal_semaphore: Option<&Semaphore>,
        signal_fence: Option<&super::Fence>,
    ) -> Result<u32> {
        let acquire = self
            .device
            .dispatch
            .vkAcquireNextImageKHR
            .ok_or(Error::MissingFunction("vkAcquireNextImageKHR"))?;
        let mut index: u32 = 0;
        let sem = signal_semaphore.map_or(0u64, Semaphore::raw);
        let fence = signal_fence.map_or(0u64, super::Fence::raw);
        // Safety: handles are valid; index is a stack variable.
        check(unsafe {
            acquire(
                self.device.handle,
                self.handle,
                timeout_nanos,
                sem,
                fence,
                &mut index,
            )
        })?;
        Ok(index)
    }

    /// Present the swapchain image at the given index, waiting on the
    /// `wait_semaphores` first (these should be signaled by the
    /// queue submission that rendered into the image).
    pub fn present(
        &self,
        queue: &Queue,
        image_index: u32,
        wait_semaphores: &[&Semaphore],
    ) -> Result<()> {
        let present = self
            .device
            .dispatch
            .vkQueuePresentKHR
            .ok_or(Error::MissingFunction("vkQueuePresentKHR"))?;
        let raw_waits: Vec<VkSemaphore> = wait_semaphores.iter().map(|s| s.raw()).collect();
        let info = VkPresentInfoKHR {
            sType: VkStructureType::STRUCTURE_TYPE_PRESENT_INFO_KHR,
            waitSemaphoreCount: raw_waits.len() as u32,
            pWaitSemaphores: if raw_waits.is_empty() {
                std::ptr::null()
            } else {
                raw_waits.as_ptr()
            },
            swapchainCount: 1,
            pSwapchains: &self.handle,
            pImageIndices: &image_index,
            ..Default::default()
        };
        // Safety: info, raw_waits, and the index ref live until end of call.
        check(unsafe { present(queue.raw(), &info) })
    }

    /// Pick the best `(format, color_space)` pair from the surface's
    /// supported list. Prefers `B8G8R8A8_SRGB` if available, otherwise
    /// the first supported format.
    pub fn pick_surface_format(
        surface: &Surface,
        physical: &PhysicalDevice,
    ) -> Result<(Format, VkColorSpaceKHR)> {
        let formats = surface.formats(physical)?;
        if formats.is_empty() {
            return Err(Error::Vk(VkResult::ERROR_FORMAT_NOT_SUPPORTED));
        }
        for f in &formats {
            if f.format() == Some(Format::B8G8R8A8_SRGB)
                && f.color_space() == Some(VkColorSpaceKHR::COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                // Both are `Some` by the guard above.
                return Ok((f.format().unwrap(), f.color_space().unwrap()));
            }
        }
        // Fall back to the first pair this build can actually name. A driver
        // may report a format or colour space from a newer spec revision; this
        // helper returns a typed pair, so it skips those rather than inventing
        // a name for one. Reach for `format_raw`/`color_space_raw` and build
        // the swapchain directly if you need one of them.
        formats
            .iter()
            .find_map(|f| Some((f.format()?, f.color_space()?)))
            .ok_or(Error::Vk(VkResult::ERROR_FORMAT_NOT_SUPPORTED))
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroySwapchainKHR {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}
