//! Spock example: end-to-end GPU work using the safe wrapper.
//!
//! This example uses [`spock::safe`] to:
//!
//! 1. Create a Vulkan instance.
//! 2. Pick the first physical device.
//! 3. Create a logical device with a transfer-capable queue.
//! 4. Allocate a host-visible buffer.
//! 5. Map it and write a known pattern to verify the round-trip.
//! 6. Record a `vkCmdFillBuffer` command into a command buffer.
//! 7. Submit the command buffer to the queue with a fence.
//! 8. Wait on the fence.
//! 9. Map the buffer again and verify the GPU overwrote our data.
//! 10. Drop everything — RAII handles all the `vkDestroy*` calls.
//!
//! Run with: `cargo run --example fill_buffer -p spock --features fetch-spec`
//!
//! On Linux CI machines, install Lavapipe (`mesa-vulkan-drivers`) to provide
//! a software Vulkan implementation. The example skips with a notice if no
//! Vulkan driver is available.

use spock::safe::{
    ApiVersion, Buffer, BufferCreateInfo, BufferUsage, CommandPool, DeviceCreateInfo, DeviceMemory,
    Fence, Instance, InstanceCreateInfo, MemoryPropertyFlags, QueueCreateInfo, QueueFlags,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Try to create the instance. If Vulkan isn't installed, exit cleanly.
    let instance = match Instance::new(InstanceCreateInfo {
        application_name: Some("spock fill_buffer example"),
        api_version: ApiVersion::V1_0,
        ..Default::default()
    }) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("SKIP: could not create Vulkan instance: {e}");
            eprintln!("(Install a Vulkan driver such as Lavapipe to run this example.)");
            return Ok(());
        }
    };
    println!("[OK] Created VkInstance");

    // Pick the first physical device that has a transfer-capable queue family.
    let physical_devices = instance.enumerate_physical_devices()?;
    let physical = physical_devices
        .into_iter()
        .find(|pd| pd.find_queue_family(QueueFlags::TRANSFER).is_some())
        .ok_or("No physical device with a transfer-capable queue family")?;

    let props = physical.properties();
    println!(
        "[OK] Using GPU: {} (Vulkan {})",
        props.device_name(),
        props.api_version()
    );

    // Find a transfer-capable queue family.
    let queue_family_index = physical
        .find_queue_family(QueueFlags::TRANSFER)
        .expect("transfer-capable queue family was found above");

    // Create a logical device with one queue from that family.
    let device = physical.create_device(DeviceCreateInfo {
        queue_create_infos: &[QueueCreateInfo {
            queue_family_index,
            queue_priorities: vec![1.0],
        }],
        ..Default::default()
    })?;
    let queue = device.get_queue(queue_family_index, 0);
    println!("[OK] Created VkDevice and got transfer queue");

    // Create a 1024-byte buffer with TRANSFER_DST usage.
    const BUFFER_SIZE: u64 = 1024;
    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: BUFFER_SIZE,
            usage: BufferUsage::TRANSFER_DST,
        },
    )?;
    println!("[OK] Created VkBuffer ({} bytes)", BUFFER_SIZE);

    // Allocate host-visible memory backing the buffer.
    let req = buffer.memory_requirements();
    let memory_type_index = physical
        .find_memory_type(
            req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or("No host-visible+coherent memory type available")?;

    let mut memory = DeviceMemory::allocate(&device, req.size, memory_type_index)?;
    buffer.bind_memory(&memory, 0)?;
    println!(
        "[OK] Allocated and bound {} bytes of host-visible memory",
        req.size
    );

    // Pre-write a known pattern from the host so we can verify the GPU
    // overwrote it.
    {
        let mut mapped = memory.map()?;
        let slice = mapped.as_slice_mut();
        slice.fill(0x11);
        // Verify the host write
        assert!(slice.iter().all(|&b| b == 0x11));
        println!("[OK] Host-wrote 0x11 to all bytes of the buffer");
    } // unmap on drop

    // Build a command buffer that fills the buffer with 0xDEADBEEF.
    let pool = CommandPool::new(&device, queue_family_index)?;
    let mut cmd = pool.allocate_primary()?;
    {
        let mut recording = cmd.begin()?;
        // vkCmdFillBuffer fills `size` bytes with the 32-bit constant `data`.
        recording.fill_buffer(&buffer, 0, BUFFER_SIZE, 0xDEADBEEF);
        recording.end()?;
    }
    println!("[OK] Recorded vkCmdFillBuffer with pattern 0xDEADBEEF");

    // Submit with a fence and wait for completion.
    let fence = Fence::new(&device)?;
    queue.submit(&[&cmd], Some(&fence))?;
    fence.wait(u64::MAX)?;
    println!("[OK] GPU finished the fill operation");

    // Map again and verify the GPU did its work.
    {
        let mut mapped = memory.map()?;
        let slice = mapped.as_slice_mut();
        // Each 4 bytes should be 0xDEADBEEF in little-endian (Vulkan spec).
        let expected: [u8; 4] = 0xDEADBEEFu32.to_ne_bytes();
        for chunk in slice.chunks_exact(4) {
            assert_eq!(chunk, expected, "GPU did not write the expected pattern");
        }
        println!("[OK] Verified all bytes match 0xDEADBEEF");
    }

    // Wait for the device to be idle before dropping everything (defensive;
    // Drop order would handle it but explicit wait makes the example clearer).
    device.wait_idle()?;

    println!();
    println!("=== fill_buffer example PASSED ===");
    println!("(All resources will now be dropped via RAII — no manual vkDestroy* calls.)");
    Ok(())
}
