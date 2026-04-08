//! Safe wrappers for `VkRenderPass` and `VkFramebuffer`.
//!
//! A render pass describes the *structure* of a graphics-pipeline
//! rendering operation: which attachments exist, how they're loaded /
//! stored, what subpasses use them, and the layout transitions that
//! happen at the boundaries. A framebuffer binds a render pass's
//! attachment slots to specific [`ImageView`]s.
//!
//! For 0.1, the safe wrapper exposes the most common shape — a single
//! subpass with one or two color attachments and an optional depth
//! attachment. More elaborate multi-subpass setups can drop to
//! [`spock::raw`](crate::raw) or use a future expanded API.
//!
//! ## Example
//!
//! ```ignore
//! use spock::safe::{
//!     RenderPass, RenderPassCreateInfo, AttachmentDescription, AttachmentLoadOp,
//!     AttachmentStoreOp, Format, ImageLayout,
//! };
//!
//! let render_pass = RenderPass::new(
//!     &device,
//!     RenderPassCreateInfo {
//!         color_attachments: &[AttachmentDescription {
//!             format: Format::R8G8B8A8_UNORM,
//!             load_op: AttachmentLoadOp::Clear,
//!             store_op: AttachmentStoreOp::Store,
//!             initial_layout: ImageLayout::UNDEFINED,
//!             final_layout: ImageLayout::TRANSFER_SRC_OPTIMAL,
//!         }],
//!         depth_attachment: None,
//!     },
//! )?;
//! ```

use super::device::DeviceInner;
use super::image::{Format, ImageLayout, ImageView};
use super::{Device, Error, Result, check};
use crate::raw::bindings::*;
use std::sync::Arc;

/// Action to perform on an attachment when a render pass instance begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttachmentLoadOp(pub VkAttachmentLoadOp);

impl AttachmentLoadOp {
    /// Initial contents are preserved.
    pub const LOAD: Self = Self(VkAttachmentLoadOp::ATTACHMENT_LOAD_OP_LOAD);
    /// The attachment is cleared to a fixed value at render pass start.
    pub const CLEAR: Self = Self(VkAttachmentLoadOp::ATTACHMENT_LOAD_OP_CLEAR);
    /// Initial contents are not preserved.
    pub const DONT_CARE: Self = Self(VkAttachmentLoadOp::ATTACHMENT_LOAD_OP_DONT_CARE);
}

/// Action to perform on an attachment when a render pass instance ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttachmentStoreOp(pub VkAttachmentStoreOp);

impl AttachmentStoreOp {
    pub const STORE: Self = Self(VkAttachmentStoreOp::ATTACHMENT_STORE_OP_STORE);
    pub const DONT_CARE: Self = Self(VkAttachmentStoreOp::ATTACHMENT_STORE_OP_DONT_CARE);
}

/// Description of one attachment in a render pass.
#[derive(Debug, Clone, Copy)]
pub struct AttachmentDescription {
    pub format: Format,
    pub load_op: AttachmentLoadOp,
    pub store_op: AttachmentStoreOp,
    pub initial_layout: ImageLayout,
    pub final_layout: ImageLayout,
}

/// Parameters for [`RenderPass::new`]. Currently models a single
/// subpass with N color attachments and an optional depth attachment.
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderPassCreateInfo<'a> {
    pub color_attachments: &'a [AttachmentDescription],
    pub depth_attachment: Option<AttachmentDescription>,
}

/// A safe wrapper around `VkRenderPass`.
///
/// Render passes are destroyed automatically on drop. They keep the
/// parent device alive via an `Arc`.
pub struct RenderPass {
    pub(crate) handle: VkRenderPass,
    pub(crate) device: Arc<DeviceInner>,
    /// Cache the attachment count so the user doesn't have to track it
    /// when constructing framebuffers.
    pub(crate) attachment_count: u32,
}

impl RenderPass {
    /// Create a new single-subpass render pass.
    pub fn new(device: &Device, info: RenderPassCreateInfo<'_>) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreateRenderPass
            .ok_or(Error::MissingFunction("vkCreateRenderPass"))?;

        // Build attachment descriptions for both color and (optional) depth.
        let mut raw_attachments: Vec<VkAttachmentDescription> = Vec::new();
        let mut color_refs: Vec<VkAttachmentReference> = Vec::new();
        for (i, a) in info.color_attachments.iter().enumerate() {
            raw_attachments.push(VkAttachmentDescription {
                format: a.format.0,
                samples: SAMPLE_COUNT_1_BIT,
                loadOp: a.load_op.0,
                storeOp: a.store_op.0,
                stencilLoadOp: VkAttachmentLoadOp::ATTACHMENT_LOAD_OP_DONT_CARE,
                stencilStoreOp: VkAttachmentStoreOp::ATTACHMENT_STORE_OP_DONT_CARE,
                initialLayout: a.initial_layout.0,
                finalLayout: a.final_layout.0,
                ..Default::default()
            });
            color_refs.push(VkAttachmentReference {
                attachment: i as u32,
                layout: VkImageLayout::IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            });
        }

        let depth_ref_storage;
        let p_depth_ref: *const VkAttachmentReference;
        if let Some(d) = info.depth_attachment {
            let attach_idx = raw_attachments.len() as u32;
            raw_attachments.push(VkAttachmentDescription {
                format: d.format.0,
                samples: SAMPLE_COUNT_1_BIT,
                loadOp: d.load_op.0,
                storeOp: d.store_op.0,
                stencilLoadOp: VkAttachmentLoadOp::ATTACHMENT_LOAD_OP_DONT_CARE,
                stencilStoreOp: VkAttachmentStoreOp::ATTACHMENT_STORE_OP_DONT_CARE,
                initialLayout: d.initial_layout.0,
                finalLayout: d.final_layout.0,
                ..Default::default()
            });
            depth_ref_storage = Some(VkAttachmentReference {
                attachment: attach_idx,
                layout: VkImageLayout::IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            });
            p_depth_ref = depth_ref_storage.as_ref().unwrap() as *const _;
        } else {
            depth_ref_storage = None;
            p_depth_ref = std::ptr::null();
        }

        let subpass = VkSubpassDescription {
            pipelineBindPoint: VkPipelineBindPoint::PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount: color_refs.len() as u32,
            pColorAttachments: color_refs.as_ptr(),
            pDepthStencilAttachment: p_depth_ref,
            ..Default::default()
        };

        let raw_info = VkRenderPassCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount: raw_attachments.len() as u32,
            pAttachments: raw_attachments.as_ptr(),
            subpassCount: 1,
            pSubpasses: &subpass,
            ..Default::default()
        };

        let mut handle: VkRenderPass = 0;
        // Safety: raw_info, raw_attachments, color_refs, depth_ref_storage,
        // and subpass all live until end of scope.
        check(unsafe {
            create(
                device.inner.handle,
                &raw_info,
                std::ptr::null(),
                &mut handle,
            )
        })?;
        let _ = depth_ref_storage; // pin

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
            attachment_count: raw_attachments.len() as u32,
        })
    }

    /// Returns the raw `VkRenderPass` handle.
    pub fn raw(&self) -> VkRenderPass {
        self.handle
    }

    /// Number of attachments declared in this render pass (color + depth).
    pub fn attachment_count(&self) -> u32 {
        self.attachment_count
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroyRenderPass {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}

/// A safe wrapper around `VkFramebuffer`.
///
/// Framebuffers bind a render pass's attachment slots to concrete
/// [`ImageView`]s. They are destroyed automatically on drop.
pub struct Framebuffer {
    pub(crate) handle: VkFramebuffer,
    pub(crate) device: Arc<DeviceInner>,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl Framebuffer {
    /// Create a framebuffer that binds `attachments` to the slots in
    /// `render_pass` and renders to a `width`x`height` region.
    ///
    /// The number and order of `attachments` must match the
    /// declaration order in the [`RenderPassCreateInfo`] (color
    /// attachments first, then optional depth).
    pub fn new(
        device: &Device,
        render_pass: &RenderPass,
        attachments: &[&ImageView],
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreateFramebuffer
            .ok_or(Error::MissingFunction("vkCreateFramebuffer"))?;

        let raw_views: Vec<VkImageView> = attachments.iter().map(|v| v.handle).collect();

        let info = VkFramebufferCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            renderPass: render_pass.handle,
            attachmentCount: raw_views.len() as u32,
            pAttachments: raw_views.as_ptr(),
            width,
            height,
            layers: 1,
            ..Default::default()
        };

        let mut handle: VkFramebuffer = 0;
        // Safety: info and raw_views are valid for the call.
        check(unsafe { create(device.inner.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
            width,
            height,
        })
    }

    /// Returns the raw `VkFramebuffer` handle.
    pub fn raw(&self) -> VkFramebuffer {
        self.handle
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroyFramebuffer {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}
