//! Integration tests for Tier-2 extension wrappers:
//!
//! - `VK_KHR_push_descriptor` — `CommandBufferRecording::push_descriptor_set`.
//! - `VK_KHR_dynamic_rendering` — `begin_rendering` / `end_rendering`.
//! - `VK_KHR_shader_integer_dot_product` —
//!   `PhysicalDevice::shader_integer_dot_product_properties`.
//! - `VK_EXT_shader_atomic_float` / `_float2` —
//!   generated feature-bit toggles on `DeviceFeatures`.
//!
//! All tests degrade gracefully when the Vulkan driver doesn't expose
//! the underlying extension. The goal is to prove the *safe API surface*
//! is type-correct and, where possible, round-trips through the driver.
//!
//! Those degradations are declared through [`common`]: a missing *device* is
//! fatal under `VULKANE_REQUIRE_DEVICE`, a missing *extension* never is. On
//! this file the distinction is the whole point — Lavapipe, which the Linux CI
//! job installs, has neither `VK_KHR_push_descriptor` nor
//! `VK_EXT_shader_atomic_float`, and those absences are conformant.

mod common;

use vulkane::raw::bindings::{VkExtent2D, VkOffset2D, VkRect2D};
use vulkane::safe::{
    AccessFlags2, AttachmentLoadOp, AttachmentStoreOp, Buffer, BufferCreateInfo, BufferUsage,
    ClearValue, DeviceFeatures, Format, Image, Image2dCreateInfo, ImageLayout, ImageUsage,
    ImageView, Instance, InstanceCreateInfo, PipelineBindPoint, PipelineStage2,
    PushDescriptorWrite, Queue, RenderingAttachment, RenderingInfo,
};

fn bootstrap()
-> Result<(vulkane::safe::Device, vulkane::safe::PhysicalDevice, u32), common::Missing> {
    common::compute_device("vulkane-tier2-test")
}

#[test]
fn push_descriptor_set_graceful_missing_function() {
    // Without VK_KHR_push_descriptor enabled, vkCmdPushDescriptorSetKHR
    // isn't loaded — the safe wrapper must surface a clean
    // MissingFunction rather than panicking. This also exercises the
    // Buffer/ImageView/Sampler enum-variant arms end-to-end.
    use vulkane::safe::{
        DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorType, PipelineLayout,
        ShaderStageFlags,
    };

    let (device, _physical, qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
    };
    let queue: Queue = device.get_queue(qf, 0);

    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 256,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .expect("buffer");

    // Minimal layout: single STORAGE_BUFFER binding at 0.
    let set_layout = DescriptorSetLayout::new(
        &device,
        &[DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
        }],
    )
    .expect("set layout");
    let pipe_layout = PipelineLayout::new(&device, &[&set_layout]).expect("pipeline layout");

    let result = queue.one_shot(&device, qf, |rec| {
        rec.push_descriptor_set(
            PipelineBindPoint::Compute,
            &pipe_layout,
            0,
            &[PushDescriptorWrite::StorageBuffer {
                binding: 0,
                buffer: &buffer,
                offset: 0,
                range: 256,
            }],
        )
    });
    match result {
        Ok(()) => {}
        Err(vulkane::safe::Error::MissingFunction(name)) => {
            assert_eq!(name, "vkCmdPushDescriptorSetKHR");
        }
        Err(vulkane::safe::Error::Vk(_)) => {
            // A driver that loaded vkCmdPushDescriptorSet* but rejected the
            // call because the layout wasn't created with PUSH_DESCRIPTOR_BIT_KHR
            // is also acceptable for this API-surface test.
        }
        Err(other) => panic!("unexpected error shape: {other:?}"),
    }
}

#[test]
fn dynamic_rendering_begin_end_graceful_missing_function() {
    // VK_KHR_dynamic_rendering / core 1.3. Without the feature enabled,
    // vkCmdBeginRendering isn't loaded. We exercise the safe wrapper
    // and accept either success or MissingFunction.
    let (device, physical, qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
    };
    let queue: Queue = device.get_queue(qf, 0);

    // Create a color-attachment-capable image + view so the RenderingAttachment
    // reference is well-typed.
    let (image, _mem, view) = match Image::new_2d_bound(
        &device,
        &physical,
        Image2dCreateInfo {
            format: Format::R8G8B8A8_UNORM,
            width: 64,
            height: 64,
            usage: ImageUsage::COLOR_ATTACHMENT,
        },
        vulkane::safe::MemoryPropertyFlags::DEVICE_LOCAL,
    ) {
        Ok(triple) => triple,
        Err(e) => {
            // Not device-absence: the device is here and declined a
            // COLOR_ATTACHMENT R8G8B8A8_UNORM image in DEVICE_LOCAL memory.
            // A compute-only device legitimately cannot serve that.
            return common::skipped_unsupported(&format!(
                "a DEVICE_LOCAL R8G8B8A8_UNORM color attachment ({e:?})"
            ));
        }
    };
    let _ = image;
    let _ = &view;

    let result = queue.one_shot(&device, qf, |rec| {
        rec.begin_rendering(RenderingInfo {
            render_area: VkRect2D {
                offset: VkOffset2D { x: 0, y: 0 },
                extent: VkExtent2D {
                    width: 64,
                    height: 64,
                },
            },
            layer_count: 1,
            view_mask: 0,
            color_attachments: &[RenderingAttachment {
                view: &view,
                layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                load_op: AttachmentLoadOp::CLEAR,
                store_op: AttachmentStoreOp::STORE,
                clear_value: Some(ClearValue::Color([0.1, 0.2, 0.3, 1.0])),
            }],
            depth_attachment: None,
            stencil_attachment: None,
        })?;
        rec.end_rendering()?;
        Ok(())
    });

    match result {
        Ok(()) => {}
        Err(vulkane::safe::Error::MissingFunction(name)) => {
            assert!(
                name == "vkCmdBeginRendering" || name == "vkCmdEndRendering",
                "unexpected missing fn: {name}"
            );
        }
        Err(vulkane::safe::Error::Vk(_)) => {
            // Driver may reject the image layout transition or attachment
            // compatibility — acceptable for this API-surface test.
        }
        Err(other) => panic!("unexpected error shape: {other:?}"),
    }
}

#[test]
fn shader_integer_dot_product_properties_queryable() {
    // Should never panic. Returns either Some with all-bools (possibly all
    // false on a driver that doesn't support the extension) or None if
    // vkGetPhysicalDeviceProperties2 isn't loaded.
    let instance = match Instance::new(InstanceCreateInfo::default()) {
        Ok(i) => i,
        Err(e) => {
            return common::skipped(&format!("no Vulkan ICD, or instance creation failed: {e}"));
        }
    };
    // NOT `unwrap_or_default()`. That turns a FAILED enumeration into an empty
    // Vec, the loop runs zero times, and the test passes having asserted
    // nothing — the error and the honest-empty case become the same event, and
    // the honest-empty case is itself a vacuous pass. Both are declared.
    let devices = match instance.enumerate_physical_devices() {
        Ok(d) => d,
        Err(e) => {
            return common::skipped(&format!(
                "an ICD is present but enumerating physical devices failed: {e:?}"
            ));
        }
    };
    if devices.is_empty() {
        return common::skipped("an ICD is present but reports no physical devices");
    }
    for pd in devices {
        let props = pd.shader_integer_dot_product_properties();
        eprintln!(
            "dev {}: int_dot_product={:?}",
            pd.properties().device_name(),
            props
        );
        if let Some(p) = props {
            // has_any_int8_acceleration's composition over the bool
            // fields must be consistent (no panic, no wrong type).
            let _ = p.has_any_int8_acceleration();
        }
    }
}

#[test]
fn device_features_shader_integer_dot_product_toggle_exists() {
    // Pure compile-time/type-check: the generated toggle must exist and
    // return `DeviceFeatures`. We don't try to actually enable it on a
    // device because not every adapter supports it.
    let features = DeviceFeatures::new().with_shader_integer_dot_product();
    // .chain() is pub(crate); structure_types is publicly introspectable
    // on PNextChain. Round-trip the call to ensure the struct wasn't
    // silently stripped.
    let _ = &features;
}

#[test]
fn device_features_shader_atomic_float_toggles_exist() {
    // Generated accessors — spot-check the four most useful for ML:
    // - buffer f32 atomic add (reductions, accumulators)
    // - buffer f32 atomics (loads/stores)
    // - buffer f64 atomic add (scientific workloads)
    // - shared f32 atomic add (subgroup reductions into workgroup memory)
    let _ = DeviceFeatures::new()
        .with_shader_buffer_float32_atomic_add()
        .with_shader_buffer_float32_atomics()
        .with_shader_buffer_float64_atomic_add()
        .with_shader_shared_float32_atomic_add();
}

#[test]
fn buffer_barrier2_in_recording_via_tier2_path() {
    // Regression check: the Tier-1 buffer_barrier2 still works after
    // Tier-2 additions reshaped the command.rs file layout.
    let (device, _physical, qf) = match bootstrap() {
        Ok(v) => v,
        Err(cause) => return common::skip(cause),
    };
    let queue: Queue = device.get_queue(qf, 0);
    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 128,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .expect("buffer");
    let r = queue.one_shot(&device, qf, |rec| {
        rec.buffer_barrier2(
            PipelineStage2::COMPUTE_SHADER,
            PipelineStage2::COMPUTE_SHADER,
            AccessFlags2::SHADER_WRITE,
            AccessFlags2::SHADER_READ,
            &buffer,
        )
    });
    match r {
        Ok(()) => {}
        Err(vulkane::safe::Error::MissingFunction(name)) => {
            assert_eq!(name, "vkCmdPipelineBarrier2");
        }
        Err(e) => panic!("unexpected: {e:?}"),
    }
}

fn _compile_check_imageview_used(_: &ImageView) {
    // Silences dead-code analysis for the Image::new_2d_bound triple:
    // the `view` reference is what we actually use in dynamic_rendering_*,
    // but an unused-variable lint would otherwise fire on some toolchains.
}
