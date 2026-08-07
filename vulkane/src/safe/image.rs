//! Safe wrappers for `VkImage` and `VkImageView` — currently focused on
//! 2D storage images for compute. Sampler and graphics-only image flows
//! will land alongside the graphics path.
//!
//! The typical compute usage is:
//!
//! 1. Create an [`Image`] with `STORAGE | TRANSFER_SRC | TRANSFER_DST` usage.
//! 2. Allocate device-local [`DeviceMemory`] sized by the image's
//!    memory requirements and bind it.
//! 3. Create an [`ImageView`] over the image (typically the whole image,
//!    one mip level, one array layer).
//! 4. Transition the image from `UNDEFINED` to `GENERAL` (the layout that
//!    storage images use) with `cmd.image_barrier(...)`.
//! 5. Bind the view to a `STORAGE_IMAGE` descriptor and dispatch.
//! 6. Optionally `cmd.copy_image_to_buffer` to read the result back to
//!    HOST_VISIBLE memory.

use super::buffer::MemoryRequirements;
use super::device::DeviceInner;
use super::{Device, DeviceMemory, Error, Result, check};
use crate::raw::bindings::*;
use std::sync::Arc;

/// Pixel format. Only the variants that are typical for compute work are
/// pre-defined as constants — wrap any `VkFormat` directly via the tuple
/// constructor for less common cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format(pub VkFormat);

impl Format {
    // 8-bit per-channel
    pub const R8_UNORM: Self = Self(VkFormat::FORMAT_R8_UNORM);
    pub const R8_SNORM: Self = Self(VkFormat::FORMAT_R8_SNORM);
    pub const R8_UINT: Self = Self(VkFormat::FORMAT_R8_UINT);
    pub const R8_SINT: Self = Self(VkFormat::FORMAT_R8_SINT);
    pub const R8G8_UNORM: Self = Self(VkFormat::FORMAT_R8G8_UNORM);
    pub const R8G8_UINT: Self = Self(VkFormat::FORMAT_R8G8_UINT);
    pub const R8G8B8A8_UNORM: Self = Self(VkFormat::FORMAT_R8G8B8A8_UNORM);
    pub const R8G8B8A8_SNORM: Self = Self(VkFormat::FORMAT_R8G8B8A8_SNORM);
    pub const R8G8B8A8_UINT: Self = Self(VkFormat::FORMAT_R8G8B8A8_UINT);
    pub const R8G8B8A8_SINT: Self = Self(VkFormat::FORMAT_R8G8B8A8_SINT);
    pub const R8G8B8A8_SRGB: Self = Self(VkFormat::FORMAT_R8G8B8A8_SRGB);
    pub const B8G8R8A8_UNORM: Self = Self(VkFormat::FORMAT_B8G8R8A8_UNORM);
    pub const B8G8R8A8_SRGB: Self = Self(VkFormat::FORMAT_B8G8R8A8_SRGB);

    // 16-bit per-channel
    pub const R16_UINT: Self = Self(VkFormat::FORMAT_R16_UINT);
    pub const R16_SINT: Self = Self(VkFormat::FORMAT_R16_SINT);
    pub const R16_SFLOAT: Self = Self(VkFormat::FORMAT_R16_SFLOAT);
    pub const R16G16_SFLOAT: Self = Self(VkFormat::FORMAT_R16G16_SFLOAT);
    pub const R16G16B16A16_SFLOAT: Self = Self(VkFormat::FORMAT_R16G16B16A16_SFLOAT);
    pub const R16G16B16A16_UINT: Self = Self(VkFormat::FORMAT_R16G16B16A16_UINT);
    pub const R16G16B16A16_UNORM: Self = Self(VkFormat::FORMAT_R16G16B16A16_UNORM);

    // 32-bit per-channel
    pub const R32_UINT: Self = Self(VkFormat::FORMAT_R32_UINT);
    pub const R32_SINT: Self = Self(VkFormat::FORMAT_R32_SINT);
    pub const R32_SFLOAT: Self = Self(VkFormat::FORMAT_R32_SFLOAT);
    pub const R32G32_UINT: Self = Self(VkFormat::FORMAT_R32G32_UINT);
    pub const R32G32_SINT: Self = Self(VkFormat::FORMAT_R32G32_SINT);
    pub const R32G32_SFLOAT: Self = Self(VkFormat::FORMAT_R32G32_SFLOAT);
    pub const R32G32B32_UINT: Self = Self(VkFormat::FORMAT_R32G32B32_UINT);
    pub const R32G32B32_SINT: Self = Self(VkFormat::FORMAT_R32G32B32_SINT);
    pub const R32G32B32_SFLOAT: Self = Self(VkFormat::FORMAT_R32G32B32_SFLOAT);
    pub const R32G32B32A32_UINT: Self = Self(VkFormat::FORMAT_R32G32B32A32_UINT);
    pub const R32G32B32A32_SINT: Self = Self(VkFormat::FORMAT_R32G32B32A32_SINT);
    pub const R32G32B32A32_SFLOAT: Self = Self(VkFormat::FORMAT_R32G32B32A32_SFLOAT);

    // Depth / stencil
    pub const D16_UNORM: Self = Self(VkFormat::FORMAT_D16_UNORM);
    pub const D32_SFLOAT: Self = Self(VkFormat::FORMAT_D32_SFLOAT);
    pub const D24_UNORM_S8_UINT: Self = Self(VkFormat::FORMAT_D24_UNORM_S8_UINT);
    pub const D32_SFLOAT_S8_UINT: Self = Self(VkFormat::FORMAT_D32_SFLOAT_S8_UINT);

    // Compressed (BC / DXT)
    pub const BC1_RGB_UNORM: Self = Self(VkFormat::FORMAT_BC1_RGB_UNORM_BLOCK);
    pub const BC1_RGB_SRGB: Self = Self(VkFormat::FORMAT_BC1_RGB_SRGB_BLOCK);
    pub const BC3_UNORM: Self = Self(VkFormat::FORMAT_BC3_UNORM_BLOCK);
    pub const BC3_SRGB: Self = Self(VkFormat::FORMAT_BC3_SRGB_BLOCK);
    pub const BC5_UNORM: Self = Self(VkFormat::FORMAT_BC5_UNORM_BLOCK);
    pub const BC7_UNORM: Self = Self(VkFormat::FORMAT_BC7_UNORM_BLOCK);
    pub const BC7_SRGB: Self = Self(VkFormat::FORMAT_BC7_SRGB_BLOCK);

    /// Returns the byte size per pixel for common uncompressed formats.
    /// Returns `None` for compressed or unknown formats.
    pub const fn bytes_per_pixel(&self) -> Option<u32> {
        match self.0 {
            VkFormat::FORMAT_R8_UNORM
            | VkFormat::FORMAT_R8_SNORM
            | VkFormat::FORMAT_R8_UINT
            | VkFormat::FORMAT_R8_SINT => Some(1),
            VkFormat::FORMAT_R8G8_UNORM
            | VkFormat::FORMAT_R8G8_UINT
            | VkFormat::FORMAT_R16_UINT
            | VkFormat::FORMAT_R16_SINT
            | VkFormat::FORMAT_R16_SFLOAT
            | VkFormat::FORMAT_D16_UNORM => Some(2),
            VkFormat::FORMAT_R8G8B8A8_UNORM
            | VkFormat::FORMAT_R8G8B8A8_SNORM
            | VkFormat::FORMAT_R8G8B8A8_UINT
            | VkFormat::FORMAT_R8G8B8A8_SINT
            | VkFormat::FORMAT_R8G8B8A8_SRGB
            | VkFormat::FORMAT_B8G8R8A8_UNORM
            | VkFormat::FORMAT_B8G8R8A8_SRGB
            | VkFormat::FORMAT_R16G16_SFLOAT
            | VkFormat::FORMAT_R32_UINT
            | VkFormat::FORMAT_R32_SINT
            | VkFormat::FORMAT_R32_SFLOAT
            | VkFormat::FORMAT_D32_SFLOAT
            | VkFormat::FORMAT_D24_UNORM_S8_UINT => Some(4),
            VkFormat::FORMAT_R16G16B16A16_SFLOAT
            | VkFormat::FORMAT_R16G16B16A16_UINT
            | VkFormat::FORMAT_R16G16B16A16_UNORM
            | VkFormat::FORMAT_R32G32_UINT
            | VkFormat::FORMAT_R32G32_SINT
            | VkFormat::FORMAT_R32G32_SFLOAT
            | VkFormat::FORMAT_D32_SFLOAT_S8_UINT => Some(8),
            VkFormat::FORMAT_R32G32B32_UINT
            | VkFormat::FORMAT_R32G32B32_SINT
            | VkFormat::FORMAT_R32G32B32_SFLOAT => Some(12),
            VkFormat::FORMAT_R32G32B32A32_UINT
            | VkFormat::FORMAT_R32G32B32A32_SINT
            | VkFormat::FORMAT_R32G32B32A32_SFLOAT => Some(16),
            _ => None,
        }
    }
}

/// Image layout — Vulkan tracks images through several access-pattern
/// "layouts" the implementation uses to choose tiling and compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageLayout(pub VkImageLayout);

impl ImageLayout {
    pub const UNDEFINED: Self = Self(VkImageLayout::IMAGE_LAYOUT_UNDEFINED);
    pub const GENERAL: Self = Self(VkImageLayout::IMAGE_LAYOUT_GENERAL);
    pub const COLOR_ATTACHMENT_OPTIMAL: Self =
        Self(VkImageLayout::IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    pub const DEPTH_STENCIL_ATTACHMENT_OPTIMAL: Self =
        Self(VkImageLayout::IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    pub const SHADER_READ_ONLY_OPTIMAL: Self =
        Self(VkImageLayout::IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    pub const TRANSFER_SRC_OPTIMAL: Self = Self(VkImageLayout::IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    pub const TRANSFER_DST_OPTIMAL: Self = Self(VkImageLayout::IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    pub const PRESENT_SRC_KHR: Self = Self(VkImageLayout::IMAGE_LAYOUT_PRESENT_SRC_KHR);
}

/// Image usage flag bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageUsage(pub u32);

impl ImageUsage {
    pub const TRANSFER_SRC: Self = Self(0x1);
    pub const TRANSFER_DST: Self = Self(0x2);
    pub const SAMPLED: Self = Self(0x4);
    pub const STORAGE: Self = Self(0x8);
    pub const COLOR_ATTACHMENT: Self = Self(0x10);
    pub const DEPTH_STENCIL_ATTACHMENT: Self = Self(0x20);
    pub const TRANSIENT_ATTACHMENT: Self = Self(0x40);
    pub const INPUT_ATTACHMENT: Self = Self(0x80);

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for ImageUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Parameters for [`Image::new_2d`].
#[derive(Debug, Clone, Copy)]
pub struct Image2dCreateInfo {
    pub format: Format,
    pub width: u32,
    pub height: u32,
    pub usage: ImageUsage,
}

/// A safe wrapper around `VkImage`.
///
/// The image is destroyed automatically on drop. The handle keeps the parent
/// device alive via an `Arc`.
///
/// To use an image, you must:
/// 1. Create it with [`Image::new_2d`].
/// 2. Query its memory requirements via [`memory_requirements`](Self::memory_requirements).
/// 3. Allocate compatible device memory and bind it via [`bind_memory`](Self::bind_memory).
/// 4. Wrap it in an [`ImageView`] for descriptor writes.
pub struct Image {
    pub(crate) handle: VkImage,
    pub(crate) device: Arc<DeviceInner>,
    pub(crate) format: Format,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl Image {
    /// Create a new 2D image. Tiling is always `OPTIMAL`, mip levels = 1,
    /// array layers = 1, sample count = 1, initial layout = UNDEFINED.
    /// These defaults are sufficient for compute storage images.
    pub fn new_2d(device: &Device, info: Image2dCreateInfo) -> Result<Self> {
        Self::new_2d_with_pnext(device, info, None)
    }

    /// Create a new 2D image, optionally attaching an extension `pNext`
    /// chain to `VkImageCreateInfo`. Same defaults as [`new_2d`](Self::new_2d).
    ///
    /// Typical uses:
    ///
    /// - `VkExternalMemoryImageCreateInfo` — declare an image as compatible
    ///   with external-memory handle types (DX12/CUDA/HIP interop).
    /// - `VkImageFormatListCreateInfo` — supply the mutable-format view-format
    ///   list up-front.
    /// - `VkImageCompressionControlEXT` — fine-grained image compression
    ///   control.
    pub fn new_2d_with_pnext(
        device: &Device,
        info: Image2dCreateInfo,
        pnext: Option<&super::pnext::PNextChain>,
    ) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreateImage
            .ok_or(Error::MissingFunction("vkCreateImage"))?;

        let create_info = VkImageCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            pNext: pnext.map_or(std::ptr::null(), |c| c.head()),
            imageType: VkImageType::IMAGE_TYPE_2D,
            format: info.format.0,
            extent: VkExtent3D {
                width: info.width,
                height: info.height,
                depth: 1,
            },
            mipLevels: 1,
            arrayLayers: 1,
            samples: SAMPLE_COUNT_1_BIT,
            tiling: VkImageTiling::IMAGE_TILING_OPTIMAL,
            usage: info.usage.0,
            sharingMode: VkSharingMode::SHARING_MODE_EXCLUSIVE,
            initialLayout: VkImageLayout::IMAGE_LAYOUT_UNDEFINED,
            ..Default::default()
        };

        let mut handle: VkImage = 0;
        // Safety: create_info is valid for the call, device is valid. The
        // optional pNext chain is borrowed through `pnext` and lives for the
        // duration of this synchronous call.
        check(unsafe {
            create(
                device.inner.handle,
                &create_info,
                std::ptr::null(),
                &mut handle,
            )
        })?;

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
            format: info.format,
            width: info.width,
            height: info.height,
        })
    }

    /// Create a 2D image, allocate memory, bind, and create a color
    /// [`ImageView`] — all in one call. Returns the image, its backing
    /// [`DeviceMemory`], and the view.
    ///
    /// Always creates a color-aspect view; for depth images use
    /// [`new_2d`](Self::new_2d) + [`ImageView::new_2d_depth`] manually.
    pub fn new_2d_bound(
        device: &Device,
        physical: &super::PhysicalDevice,
        info: Image2dCreateInfo,
        memory_flags: super::MemoryPropertyFlags,
    ) -> Result<(Image, super::DeviceMemory, ImageView)> {
        let image = Image::new_2d(device, info)?;
        let req = image.memory_requirements();
        let type_index = physical
            .find_memory_type(req.memory_type_bits, memory_flags)
            .ok_or(Error::InvalidArgument(
                "no compatible memory type for the requested property flags",
            ))?;
        let memory = super::DeviceMemory::allocate(device, req.size, type_index)?;
        image.bind_memory(&memory, 0)?;
        let view = ImageView::new_2d_color(&image)?;
        Ok((image, memory, view))
    }

    /// Returns the raw `VkImage` handle.
    pub fn raw(&self) -> VkImage {
        self.handle
    }

    pub fn format(&self) -> Format {
        self.format
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Query the memory requirements for this image.
    pub fn memory_requirements(&self) -> MemoryRequirements {
        let get = self
            .device
            .dispatch
            .vkGetImageMemoryRequirements
            .expect("vkGetImageMemoryRequirements is required by Vulkan 1.0");

        // Safety: handles are valid; output struct will be fully overwritten.
        let mut req: VkMemoryRequirements = unsafe { std::mem::zeroed() };
        unsafe { get(self.device.handle, self.handle, &mut req) };
        MemoryRequirements {
            size: req.size,
            alignment: req.alignment,
            memory_type_bits: req.memoryTypeBits,
        }
    }

    /// Bind a previously allocated [`DeviceMemory`] to this image at the
    /// given offset.
    pub fn bind_memory(&self, memory: &DeviceMemory, offset: u64) -> Result<()> {
        let bind = self
            .device
            .dispatch
            .vkBindImageMemory
            .ok_or(Error::MissingFunction("vkBindImageMemory"))?;
        // Safety: handles are valid, offset is the caller's responsibility.
        check(unsafe { bind(self.device.handle, self.handle, memory.handle, offset) })
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroyImage {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}

/// A safe wrapper around `VkImageView`.
///
/// Views describe how a (sub)region of an image is interpreted by shaders.
/// They are destroyed automatically on drop.
pub struct ImageView {
    pub(crate) handle: VkImageView,
    pub(crate) device: Arc<DeviceInner>,
}

impl ImageView {
    /// Create a 2D color view over the entire image — single mip, single
    /// layer, identity component swizzle. Sufficient for storage-image use.
    pub fn new_2d_color(image: &Image) -> Result<Self> {
        let create = image
            .device
            .dispatch
            .vkCreateImageView
            .ok_or(Error::MissingFunction("vkCreateImageView"))?;

        let info = VkImageViewCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            image: image.handle,
            viewType: VkImageViewType::IMAGE_VIEW_TYPE_2D,
            format: image.format.0,
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
        // Safety: info is valid for the call, image handle is valid.
        check(unsafe { create(image.device.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            device: Arc::clone(&image.device),
        })
    }

    /// Create a 2D depth view over the entire image — single mip, single
    /// layer. Use this for depth attachments (e.g. shadow maps,
    /// depth prepasses). The image must have a depth format like
    /// [`Format::D32_SFLOAT`] or [`Format::D24_UNORM_S8_UINT`].
    pub fn new_2d_depth(image: &Image) -> Result<Self> {
        let create = image
            .device
            .dispatch
            .vkCreateImageView
            .ok_or(Error::MissingFunction("vkCreateImageView"))?;

        let info = VkImageViewCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            image: image.handle,
            viewType: VkImageViewType::IMAGE_VIEW_TYPE_2D,
            format: image.format.0,
            components: VkComponentMapping {
                r: VkComponentSwizzle::COMPONENT_SWIZZLE_IDENTITY,
                g: VkComponentSwizzle::COMPONENT_SWIZZLE_IDENTITY,
                b: VkComponentSwizzle::COMPONENT_SWIZZLE_IDENTITY,
                a: VkComponentSwizzle::COMPONENT_SWIZZLE_IDENTITY,
            },
            subresourceRange: VkImageSubresourceRange {
                aspectMask: IMAGE_ASPECT_DEPTH_BIT,
                baseMipLevel: 0,
                levelCount: 1,
                baseArrayLayer: 0,
                layerCount: 1,
            },
            ..Default::default()
        };

        let mut handle: VkImageView = 0;
        check(unsafe { create(image.device.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            device: Arc::clone(&image.device),
        })
    }

    /// Returns the raw `VkImageView` handle.
    pub fn raw(&self) -> VkImageView {
        self.handle
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroyImageView {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}

/// One image-region copy descriptor for [`copy_buffer_to_image`](super::CommandBufferRecording::copy_buffer_to_image).
#[derive(Debug, Clone, Copy)]
pub struct BufferImageCopy {
    /// Byte offset in the source/destination buffer.
    pub buffer_offset: u64,
    /// Row length in pixels (0 = tightly packed).
    pub buffer_row_length: u32,
    /// Image height in pixels (0 = tightly packed).
    pub buffer_image_height: u32,
    /// Top-left of the region in the image (X, Y, Z).
    pub image_offset: [i32; 3],
    /// Size of the region in the image (W, H, D).
    pub image_extent: [u32; 3],
}

impl BufferImageCopy {
    /// Convenience: a tightly-packed full-image copy at offset 0 of a 2D
    /// image with the given size.
    pub fn full_2d(width: u32, height: u32) -> Self {
        Self {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_offset: [0, 0, 0],
            image_extent: [width, height, 1],
        }
    }

    pub(crate) fn to_raw(self) -> VkBufferImageCopy {
        VkBufferImageCopy {
            bufferOffset: self.buffer_offset,
            bufferRowLength: self.buffer_row_length,
            bufferImageHeight: self.buffer_image_height,
            imageSubresource: VkImageSubresourceLayers {
                aspectMask: IMAGE_ASPECT_COLOR_BIT,
                mipLevel: 0,
                baseArrayLayer: 0,
                layerCount: 1,
            },
            imageOffset: VkOffset3D {
                x: self.image_offset[0],
                y: self.image_offset[1],
                z: self.image_offset[2],
            },
            imageExtent: VkExtent3D {
                width: self.image_extent[0],
                height: self.image_extent[1],
                depth: self.image_extent[2],
            },
        }
    }
}

/// Texel filter mode for a [`Sampler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerFilter(pub VkFilter);

impl SamplerFilter {
    pub const NEAREST: Self = Self(VkFilter::FILTER_NEAREST);
    pub const LINEAR: Self = Self(VkFilter::FILTER_LINEAR);
}

/// Mipmap mode for a [`Sampler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerMipmapMode(pub VkSamplerMipmapMode);

impl SamplerMipmapMode {
    pub const NEAREST: Self = Self(VkSamplerMipmapMode::SAMPLER_MIPMAP_MODE_NEAREST);
    pub const LINEAR: Self = Self(VkSamplerMipmapMode::SAMPLER_MIPMAP_MODE_LINEAR);
}

/// UV(W) addressing mode for a [`Sampler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerAddressMode(pub VkSamplerAddressMode);

impl SamplerAddressMode {
    pub const REPEAT: Self = Self(VkSamplerAddressMode::SAMPLER_ADDRESS_MODE_REPEAT);
    pub const MIRRORED_REPEAT: Self =
        Self(VkSamplerAddressMode::SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT);
    pub const CLAMP_TO_EDGE: Self = Self(VkSamplerAddressMode::SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    pub const CLAMP_TO_BORDER: Self =
        Self(VkSamplerAddressMode::SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
}

/// Parameters for [`Sampler::new`]. Defaults to a sensible "nearest
/// linear-magnification clamp-to-edge" sampler suitable for textured
/// quads. Override fields as needed.
#[derive(Debug, Clone, Copy)]
pub struct SamplerCreateInfo {
    pub mag_filter: SamplerFilter,
    pub min_filter: SamplerFilter,
    pub mipmap_mode: SamplerMipmapMode,
    pub address_mode_u: SamplerAddressMode,
    pub address_mode_v: SamplerAddressMode,
    pub address_mode_w: SamplerAddressMode,
    pub anisotropy: Option<f32>,
    /// Enable depth comparison sampling. When `Some(op)`, the sampler
    /// performs a comparison against a reference depth value (as used by
    /// `textureSampleCompare` in WGSL / `shadow2D` in GLSL). Essential
    /// for shadow mapping.
    pub compare_op: Option<super::CompareOp>,
}

impl Default for SamplerCreateInfo {
    fn default() -> Self {
        Self {
            mag_filter: SamplerFilter::LINEAR,
            min_filter: SamplerFilter::LINEAR,
            mipmap_mode: SamplerMipmapMode::LINEAR,
            address_mode_u: SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: SamplerAddressMode::CLAMP_TO_EDGE,
            anisotropy: None,
            compare_op: None,
        }
    }
}

/// A safe wrapper around `VkSampler`.
///
/// Samplers describe how an image is sampled inside a shader: filter
/// mode, addressing mode, anisotropic filtering, etc. They are
/// destroyed automatically on drop.
pub struct Sampler {
    pub(crate) handle: VkSampler,
    pub(crate) device: Arc<DeviceInner>,
}

impl Sampler {
    /// Create a new sampler.
    pub fn new(device: &Device, info: SamplerCreateInfo) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreateSampler
            .ok_or(Error::MissingFunction("vkCreateSampler"))?;

        let raw_info = VkSamplerCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            magFilter: info.mag_filter.0,
            minFilter: info.min_filter.0,
            mipmapMode: info.mipmap_mode.0,
            addressModeU: info.address_mode_u.0,
            addressModeV: info.address_mode_v.0,
            addressModeW: info.address_mode_w.0,
            mipLodBias: 0.0,
            anisotropyEnable: if info.anisotropy.is_some() { 1 } else { 0 },
            maxAnisotropy: info.anisotropy.unwrap_or(1.0),
            compareEnable: if info.compare_op.is_some() { 1 } else { 0 },
            compareOp: info
                .compare_op
                .map_or(VkCompareOp::COMPARE_OP_NEVER, |c| c.0),
            minLod: 0.0,
            maxLod: 0.0,
            borderColor: VkBorderColor::BORDER_COLOR_FLOAT_OPAQUE_BLACK,
            unnormalizedCoordinates: 0,
            ..Default::default()
        };

        let mut handle: VkSampler = 0;
        // Safety: raw_info is valid for the call.
        check(unsafe {
            create(
                device.inner.handle,
                &raw_info,
                std::ptr::null(),
                &mut handle,
            )
        })?;

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
        })
    }

    /// Returns the raw `VkSampler` handle.
    pub fn raw(&self) -> VkSampler {
        self.handle
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroySampler {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}

/// A simplified image memory barrier for layout transitions.
///
/// Operates on a single mip level, single array layer. Use
/// `aspect_mask` to select color or depth aspect (default: color).
#[derive(Clone, Copy)]
pub struct ImageBarrier<'a> {
    pub image: &'a Image,
    pub old_layout: ImageLayout,
    pub new_layout: ImageLayout,
    pub src_access: super::AccessFlags,
    pub dst_access: super::AccessFlags,
    /// Aspect mask: `IMAGE_ASPECT_COLOR_BIT` (default) or
    /// `IMAGE_ASPECT_DEPTH_BIT`. Use the [`color`](Self::color) /
    /// [`depth`](Self::depth) constructors for convenience.
    pub aspect_mask: u32,
}

impl<'a> ImageBarrier<'a> {
    /// Create a color-aspect image barrier.
    pub fn color(
        image: &'a Image,
        old_layout: ImageLayout,
        new_layout: ImageLayout,
        src_access: super::AccessFlags,
        dst_access: super::AccessFlags,
    ) -> Self {
        Self {
            image,
            old_layout,
            new_layout,
            src_access,
            dst_access,
            aspect_mask: IMAGE_ASPECT_COLOR_BIT,
        }
    }

    /// Create a depth-aspect image barrier.
    pub fn depth(
        image: &'a Image,
        old_layout: ImageLayout,
        new_layout: ImageLayout,
        src_access: super::AccessFlags,
        dst_access: super::AccessFlags,
    ) -> Self {
        Self {
            image,
            old_layout,
            new_layout,
            src_access,
            dst_access,
            aspect_mask: IMAGE_ASPECT_DEPTH_BIT,
        }
    }
}
