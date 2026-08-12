//! Integration tests for the Tier-1 extension-support additions:
//!
//! - `pNext`-chain extension points on safe create-info builders
//!   (Memory/Buffer/Image/Fence/Semaphore/Device/Instance).
//! - Ergonomic wrappers for `VK_KHR_external_memory_{win32,fd}` and
//!   `VK_KHR_external_semaphore_{win32,fd}`.
//! - `buffer_barrier2` addition to sync2 command recording.
//!
//! Every test degrades gracefully when Vulkan is unavailable or the
//! relevant extension isn't supported by the host driver — we only
//! assert on surface behaviour that does not require the extension to
//! be live. The goal is to prove the **API surface** works, not to
//! exercise real cross-API interop (which would need a companion
//! CUDA/HIP/D3D12 process).

mod common;

use vulkane::raw::PNextChainable;
#[cfg(unix)]
use vulkane::raw::bindings::EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#[cfg(windows)]
use vulkane::raw::bindings::EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
use vulkane::raw::bindings::{
    MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, VkExportMemoryAllocateInfo,
    VkExternalMemoryBufferCreateInfo, VkExternalMemoryImageCreateInfo, VkMemoryAllocateFlagsInfo,
    VkStructureType,
};
use vulkane::safe::{
    Buffer, BufferCreateInfo, BufferUsage, DeviceCreateInfo, DeviceMemory, Fence, Format, Image,
    Image2dCreateInfo, ImageUsage, Instance, InstanceCreateInfo, MemoryAllocateInfo,
    MemoryPropertyFlags, PNextChain, QueueCreateInfo, Semaphore,
};

/// Boot an instance → physical device → device, or name the precondition that
/// failed. Picks any queue family with COMPUTE.
///
/// This used to declare "Vulkan not available" for the *first* of its five
/// failures and return a bare `None` for the other four, so a run that got as
/// far as `vkCreateDevice` and failed there reported nothing at all.
fn bootstrap() -> Result<(vulkane::safe::Device, vulkane::safe::PhysicalDevice, u32), &'static str>
{
    common::compute_device("vulkane-ext-pnext-test")
}

#[test]
fn memory_allocate_with_empty_pnext_matches_plain_allocate() {
    // Regression: allocate_with(info{ pnext: None }) must behave identically
    // to allocate(). Both call the same underlying path now.
    let (device, physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(why) => return common::skipped(why),
    };

    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 256,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .expect("buffer creation");
    let req = buffer.memory_requirements();
    let mt = physical
        .find_memory_type(req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
        .expect("some device-local memory type");

    // Plain allocate.
    let _a = DeviceMemory::allocate(&device, req.size, mt).expect("plain allocate");

    // allocate_with + None pnext.
    let _b = DeviceMemory::allocate_with(
        &device,
        &MemoryAllocateInfo {
            size: req.size,
            memory_type_index: mt,
            ..Default::default()
        },
    )
    .expect("allocate_with None");
}

#[test]
fn memory_allocate_with_pnext_accepts_allocate_flags_info() {
    // Using VkMemoryAllocateFlagsInfo via the safe pnext field proves the
    // chain is being plumbed through. This struct is core 1.1 so it's
    // broadly supported — we just need the memory type to exist.
    let (device, physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(why) => return common::skipped(why),
    };

    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 256,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .expect("buffer creation");
    let req = buffer.memory_requirements();
    let mt =
        match physical.find_memory_type(req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL) {
            Some(m) => m,
            None => {
                return common::skipped_unsupported(
                    "a DEVICE_LOCAL memory type this buffer's memory_type_bits accepts",
                );
            }
        };

    let mut chain = PNextChain::new();
    let mut flags = VkMemoryAllocateFlagsInfo::new_pnext();
    // DEVICE_ADDRESS_BIT requires the feature to actually be enabled for the
    // allocation to succeed, so we pass the *least*-invasive flag here: zero.
    // The pNext attachment itself is what we're regression-testing; value 0 is
    // spec-legal and drivers must accept it.
    flags.flags = 0;
    chain.push(flags);

    let mem = DeviceMemory::allocate_with(
        &device,
        &MemoryAllocateInfo {
            size: req.size,
            memory_type_index: mt,
            pnext: Some(&chain),
            ..Default::default()
        },
    );
    // We don't require success — some driver/mem-type combinations reject
    // the chain extension entirely. We only require the *call* not to
    // panic, not to UB, and to return a clean Result.
    drop(mem);
}

#[test]
fn buffer_new_with_pnext_accepts_external_memory_info() {
    // Attaching VkExternalMemoryBufferCreateInfo proves the Buffer::new_with_pnext
    // path delivers the chain to the driver. This is the buffer-side piece
    // of VK_KHR_external_memory_{win32,fd} export.
    let (device, _physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(why) => return common::skipped(why),
    };

    let mut chain = PNextChain::new();
    let mut ext = VkExternalMemoryBufferCreateInfo::new_pnext();
    // Pick platform-appropriate handle type.
    #[cfg(windows)]
    {
        ext.handleTypes = EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    }
    #[cfg(not(windows))]
    {
        ext.handleTypes = EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    }
    chain.push(ext);

    // Not every driver enables the corresponding extension, so Err here is
    // acceptable; we only require the safe wrapper to deliver the chain
    // cleanly. The generated struct.sType is set by new_pnext(); we only
    // need to check the method doesn't panic.
    let result = Buffer::new_with_pnext(
        &device,
        BufferCreateInfo {
            size: 256,
            usage: BufferUsage::STORAGE_BUFFER,
        },
        Some(&chain),
    );
    drop(result);
}

#[test]
fn image_new_2d_with_pnext_accepts_external_memory_info() {
    let (device, _physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(why) => return common::skipped(why),
    };

    let mut chain = PNextChain::new();
    let mut ext = VkExternalMemoryImageCreateInfo::new_pnext();
    #[cfg(windows)]
    {
        ext.handleTypes = EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    }
    #[cfg(not(windows))]
    {
        ext.handleTypes = EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    }
    chain.push(ext);

    let result = Image::new_2d_with_pnext(
        &device,
        Image2dCreateInfo {
            format: Format::R8G8B8A8_UNORM,
            width: 16,
            height: 16,
            usage: ImageUsage::STORAGE,
        },
        Some(&chain),
    );
    drop(result);
}

#[test]
fn fence_new_with_pnext_empty_chain_works() {
    let (device, _physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(why) => return common::skipped(why),
    };
    let empty = PNextChain::new();
    let _f = Fence::new_with_pnext(&device, Some(&empty)).expect("fence with empty pnext chain");
}

#[test]
fn semaphore_binary_with_pnext_empty_chain_works() {
    let (device, _physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(why) => return common::skipped(why),
    };
    let empty = PNextChain::new();
    let _s = Semaphore::binary_with_pnext(&device, Some(&empty))
        .expect("binary semaphore with empty pnext chain");
}

#[test]
fn semaphore_timeline_with_pnext_combines_chains() {
    // Timeline semaphores need an internal VkSemaphoreTypeCreateInfo — the
    // new timeline_with_pnext path prepends it then appends whatever the
    // caller supplied. Passing an empty extra chain must still produce a
    // valid timeline semaphore.
    let (device, _physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(why) => return common::skipped(why),
    };
    if device.dispatch().vkGetSemaphoreCounterValue.is_none() {
        return common::skipped_unsupported(
            "timeline semaphores (Vulkan 1.2 / VK_KHR_timeline_semaphore)",
        );
    }
    let empty = PNextChain::new();
    let sem = match Semaphore::timeline_with_pnext(&device, 7, Some(&empty)) {
        Ok(s) => s,
        Err(e) => {
            return common::skipped_unsupported(&format!("timeline semaphore creation ({e:?})"));
        }
    };
    assert_eq!(sem.kind(), vulkane::safe::SemaphoreKind::Timeline);
    let v = sem.current_value().expect("read timeline value");
    assert_eq!(v, 7, "initial value round-trips through pnext chain");
}

#[test]
fn device_create_info_pnext_is_plumbed_without_error() {
    // Pass an empty extra pNext chain to DeviceCreateInfo and confirm the
    // device still creates. This proves the internal chain + user chain
    // composition path in new_inner doesn't accidentally truncate the
    // chain head when user pnext is None-empty.
    //
    // Deliberately does *not* go through `bootstrap()`. What is under test is
    // chain composition, which needs a device and any queue family at all —
    // not a compute one. Requiring compute would narrow the environments this
    // runs in for no gain, and since a failed precondition here is fatal under
    // VULKANE_REQUIRE_DEVICE, over-constraining it would fail runs on hardware
    // perfectly able to answer the question the test asks.
    let instance = match Instance::new(InstanceCreateInfo::default()) {
        Ok(i) => i,
        Err(e) => {
            return common::skipped(&format!("no Vulkan ICD, or instance creation failed: {e}"));
        }
    };
    let physical = match instance.enumerate_physical_devices() {
        Ok(devices) => match devices.into_iter().next() {
            Some(p) => p,
            None => return common::skipped("an ICD is present but reports no physical devices"),
        },
        Err(e) => {
            return common::skipped(&format!("enumerating physical devices failed: {e}"));
        }
    };
    // Family 0. Every physical device exposes at least one queue family, and
    // this test does not care which kind it is.
    if physical.queue_family_properties().is_empty() {
        return common::skipped("the physical device reports no queue families at all");
    }
    let qf = 0u32;

    let empty = PNextChain::new();
    let _device = physical
        .create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo::single(qf)],
            pnext: Some(&empty),
            ..Default::default()
        })
        .expect("device with empty extra pnext chain");
}

#[test]
fn instance_create_info_pnext_is_plumbed_without_error() {
    let empty = PNextChain::new();
    let result = Instance::new(InstanceCreateInfo {
        pnext: Some(&empty),
        ..Default::default()
    });
    match result {
        Ok(_) => {}
        Err(e) => common::skipped(&format!("no Vulkan ICD, or instance creation failed: {e}")),
    }
}

#[cfg(windows)]
#[test]
fn device_memory_get_win32_handle_graceful_missing_function() {
    // Without VK_KHR_external_memory_win32 enabled, vkGetMemoryWin32HandleKHR
    // isn't loaded — the safe wrapper must surface a clean
    // MissingFunction error rather than panicking.
    let (device, physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(why) => return common::skipped(why),
    };
    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 128,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .expect("buffer");
    let req = buffer.memory_requirements();
    let Some(mt) =
        physical.find_memory_type(req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
    else {
        return common::skipped_unsupported(
            "a DEVICE_LOCAL memory type this buffer's memory_type_bits accepts",
        );
    };
    let mem = DeviceMemory::allocate(&device, req.size, mt).expect("allocate");

    match mem.get_win32_handle(EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
        Err(vulkane::safe::Error::MissingFunction(name)) => {
            assert_eq!(name, "vkGetMemoryWin32HandleKHR");
        }
        Ok(_) => {
            // A driver that auto-loads the KHR extension is fine too —
            // we only require that the safe call completes without UB.
        }
        Err(other) => panic!("unexpected error shape: {other:?}"),
    }
}

#[cfg(windows)]
#[test]
fn semaphore_get_win32_handle_graceful_missing_function() {
    use vulkane::raw::bindings::EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    let (device, _physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(why) => return common::skipped(why),
    };
    let sem = Semaphore::binary(&device).expect("binary semaphore");
    match sem.get_win32_handle(EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
        Err(vulkane::safe::Error::MissingFunction(name)) => {
            assert_eq!(name, "vkGetSemaphoreWin32HandleKHR");
        }
        Ok(_) => {}
        Err(other) => panic!("unexpected error shape: {other:?}"),
    }
}

#[cfg(unix)]
#[test]
fn device_memory_get_fd_graceful_missing_function() {
    let (device, physical, _qf) = match bootstrap() {
        Ok(v) => v,
        Err(why) => return common::skipped(why),
    };
    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 128,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .expect("buffer");
    let req = buffer.memory_requirements();
    let Some(mt) =
        physical.find_memory_type(req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
    else {
        return common::skipped_unsupported(
            "a DEVICE_LOCAL memory type this buffer's memory_type_bits accepts",
        );
    };
    let mem = DeviceMemory::allocate(&device, req.size, mt).expect("allocate");

    match mem.get_fd(EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) {
        Err(vulkane::safe::Error::MissingFunction(name)) => {
            assert_eq!(name, "vkGetMemoryFdKHR");
        }
        Ok(_) => {}
        Err(other) => panic!("unexpected error shape: {other:?}"),
    }
}

#[test]
fn buffer_barrier2_graceful_missing_function() {
    // Exercise the new buffer_barrier2 ergonomic method — we don't actually
    // need to run a real dispatch to prove the path compiles and hits
    // either Ok() or MissingFunction(vkCmdPipelineBarrier2).
    use vulkane::safe::{AccessFlags2, PipelineStage2, Queue};

    let (device, _physical, qf) = match bootstrap() {
        Ok(v) => v,
        Err(why) => return common::skipped(why),
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

    let result = queue.one_shot(&device, qf, |rec| {
        rec.buffer_barrier2(
            PipelineStage2::COMPUTE_SHADER,
            PipelineStage2::COMPUTE_SHADER,
            AccessFlags2::SHADER_WRITE,
            AccessFlags2::SHADER_READ,
            &buffer,
        )
    });
    match result {
        Ok(()) => {}
        Err(vulkane::safe::Error::MissingFunction(name)) => {
            assert_eq!(name, "vkCmdPipelineBarrier2");
        }
        Err(other) => panic!("unexpected error shape: {other:?}"),
    }
    // Make sure device_address / leak paths don't crash on drop.
    let _ = MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT; // silences an unused-import warning across cfgs
    let _ = VkStructureType::STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    let _ = VkExportMemoryAllocateInfo::new_pnext();
}
