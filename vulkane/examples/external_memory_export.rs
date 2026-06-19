//! External-memory export end-to-end using the vulkane safe wrappers.
//!
//! Demonstrates the full flow for `VK_KHR_external_memory_{win32,fd}` —
//! the extension pair used to hand Vulkan-allocated GPU memory off to a
//! different API on the same device (CUDA, HIP, ROCm, DX12, DMA-BUF).
//!
//! The example:
//!
//! 1. Creates an instance + picks a compute-capable physical device.
//! 2. Creates a logical device with the core extensions
//!    (`VK_KHR_external_memory_{win32,fd}` + their instance-level
//!    prerequisites) enabled. Skips cleanly if the device doesn't
//!    expose them.
//! 3. Creates a storage buffer marked exportable via
//!    `VkExternalMemoryBufferCreateInfo` attached through
//!    [`Buffer::new_with_pnext`].
//! 4. Allocates backing memory marked exportable via
//!    `VkExportMemoryAllocateInfo` attached through
//!    [`MemoryAllocateInfo::pnext`].
//! 5. Binds buffer → memory, then extracts the platform-native handle
//!    via [`DeviceMemory::get_win32_handle`] (Windows) or
//!    [`DeviceMemory::get_fd`] (Unix).
//! 6. Prints the handle value — in a real application you'd pass this
//!    to `cuImportExternalMemory` / `hipImportExternalMemory` /
//!    `ID3D12Device::OpenSharedHandle` to import into the consumer
//!    API.
//!
//! Run with: `cargo run --example external_memory_export -p vulkane`

use std::sync::Arc;
use vulkane::raw::PNextChainable;
use vulkane::raw::bindings::{VkExportMemoryAllocateInfo, VkExternalMemoryBufferCreateInfo};
use vulkane::safe::{
    ApiVersion, Buffer, BufferCreateInfo, BufferUsage, DeviceCreateInfo, DeviceExtensions,
    DeviceMemory, Instance, InstanceCreateInfo, InstanceExtensions, MemoryAllocateInfo,
    MemoryPropertyFlags, PNextChain, QueueCreateInfo, QueueFlags,
};

#[cfg(windows)]
const HANDLE_TYPE: u32 = vulkane::raw::bindings::EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#[cfg(not(windows))]
const HANDLE_TYPE: u32 = vulkane::raw::bindings::EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Instance with the platform prerequisites.
    let instance_exts = InstanceExtensions::new()
        .khr_get_physical_device_properties2()
        .khr_external_memory_capabilities();
    let instance = match Instance::new(InstanceCreateInfo {
        application_name: Some("vulkane external_memory_export"),
        api_version: ApiVersion::V1_1,
        enabled_extensions: Some(&instance_exts),
        ..Default::default()
    }) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("SKIP: cannot create Vulkan instance: {e:?}");
            return Ok(());
        }
    };

    // Pick any compute-capable GPU.
    let Some(physical) = instance
        .enumerate_physical_devices()?
        .into_iter()
        .find(|pd| pd.find_queue_family(QueueFlags::COMPUTE).is_some())
    else {
        eprintln!("SKIP: no compute-capable GPU found");
        return Ok(());
    };
    let qf = physical.find_queue_family(QueueFlags::COMPUTE).unwrap();
    println!("Using GPU: {}", physical.properties().device_name());

    // Device with the platform-specific external-memory extension.
    #[cfg(windows)]
    let dev_exts = DeviceExtensions::new()
        .khr_external_memory()
        .khr_external_memory_win32();
    #[cfg(not(windows))]
    let dev_exts = DeviceExtensions::new()
        .khr_external_memory()
        .khr_external_memory_fd();

    let device = match physical.create_device(DeviceCreateInfo {
        queue_create_infos: &[QueueCreateInfo::single(qf)],
        enabled_extensions: Some(&dev_exts),
        ..Default::default()
    }) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: external-memory extension unsupported on this device: {e:?}");
            return Ok(());
        }
    };

    // ── Step 1: buffer marked exportable ─────────────────────────────
    //
    // VkExternalMemoryBufferCreateInfo tells the driver "this buffer is
    // a candidate to back external memory" — without it the allocation
    // will succeed but the export call fails.
    let mut buf_chain = PNextChain::new();
    let mut buf_ext = VkExternalMemoryBufferCreateInfo::new_pnext();
    buf_ext.handleTypes = HANDLE_TYPE;
    buf_chain.push(buf_ext);

    let buffer = Buffer::new_with_pnext(
        &device,
        BufferCreateInfo {
            size: 4096,
            usage: BufferUsage::STORAGE_BUFFER,
        },
        Some(&buf_chain),
    )?;

    // ── Step 2: memory marked exportable ─────────────────────────────
    //
    // VkExportMemoryAllocateInfo makes the allocation exportable for the
    // listed handle types. Chained onto MemoryAllocateInfo.pnext.
    let req = buffer.memory_requirements();
    let memory_type_index = physical
        .find_memory_type(req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
        .ok_or("no device-local memory type")?;

    let mut mem_chain = PNextChain::new();
    let mut mem_ext = VkExportMemoryAllocateInfo::new_pnext();
    mem_ext.handleTypes = HANDLE_TYPE;
    mem_chain.push(mem_ext);

    let memory = DeviceMemory::allocate_with(
        &device,
        &MemoryAllocateInfo {
            size: req.size,
            memory_type_index,
            pnext: Some(&mem_chain),
            priority: None,
        },
    )?;

    // Bind the buffer → memory.
    buffer.bind_memory(&memory, 0)?;

    // ── Step 3: extract the native handle ────────────────────────────
    #[cfg(windows)]
    {
        match memory.get_win32_handle(HANDLE_TYPE) {
            Ok(h) => {
                println!(
                    "Exported Win32 HANDLE: {:?} (handle_type = 0x{:08x})",
                    h.raw, h.handle_type
                );
                println!(
                    "Hand this HANDLE to cuImportExternalMemory / \
                     ID3D12Device::OpenSharedHandle to import into the consumer API."
                );
                println!(
                    "Caller is responsible for closing it (for NT handle types) \
                     via CloseHandle once the consumer is done with it."
                );
            }
            Err(e) => {
                eprintln!("get_win32_handle failed: {e:?}");
            }
        }
    }

    #[cfg(not(windows))]
    {
        use std::os::fd::AsRawFd;
        match memory.get_fd(HANDLE_TYPE) {
            Ok(fd) => {
                println!(
                    "Exported POSIX file descriptor: {} (handle_type = 0x{:08x})",
                    fd.as_raw_fd(),
                    HANDLE_TYPE
                );
                println!(
                    "Hand this fd to cuImportExternalMemory / hipImportExternalMemory / \
                     a child process via SCM_RIGHTS to import into the consumer API. \
                     The OwnedFd closes on drop if ownership is not transferred."
                );
                // Keep it alive briefly so the print is meaningful; in real
                // code you'd hand it off before this scope ends.
                let _ = fd;
            }
            Err(e) => {
                eprintln!("get_fd failed: {e:?}");
            }
        }
    }

    // Silence unused-Arc warning on the backing buffer — the buffer
    // must outlive the export call, which this scope enforces.
    let _arc = Arc::new(());

    Ok(())
}
