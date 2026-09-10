// SPDX-License-Identifier: MIT OR Apache-2.0
//! Simplest possible "put data on the GPU" example.
//!
//! Shows the [`Queue::one_shot`] helper for staging-buffer uploads:
//! allocate a host-visible staging buffer, write data, then
//! `queue.one_shot()` a single `vkCmdCopyBuffer` to move it into a
//! device-local buffer. No manual command pool, command buffer, or
//! fence management.
//!
//! Run with: `cargo run -p vulkane --features fetch-spec --example buffer_upload`

use vulkane::safe::{
    AccessFlags, ApiVersion, Buffer, BufferCopy, BufferCreateInfo, BufferUsage, DeviceCreateInfo,
    DeviceMemory, Instance, InstanceCreateInfo, MemoryPropertyFlags, PipelineStage,
    QueueCreateInfo, QueueFlags,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let instance = Instance::new(InstanceCreateInfo {
        application_name: Some("vulkane buffer_upload"),
        api_version: ApiVersion::V1_0,
        ..Default::default()
    })?;
    let physical = instance
        .enumerate_physical_devices()?
        .into_iter()
        .find(|pd| pd.find_queue_family(QueueFlags::TRANSFER).is_some())
        .ok_or("no GPU with a transfer queue")?;
    println!("[OK] Using GPU: {}", physical.properties().device_name());

    let qf = physical.find_queue_family(QueueFlags::TRANSFER).unwrap();
    let device = physical.create_device(DeviceCreateInfo {
        queue_create_infos: &[QueueCreateInfo::single(qf)],
        ..Default::default()
    })?;
    let queue = device.get_queue(qf, 0);

    const DATA: [u32; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
    const SIZE: u64 = (DATA.len() * 4) as u64;

    // 1. Staging buffer (host-visible, TRANSFER_SRC).
    let staging = Buffer::new(
        &device,
        BufferCreateInfo {
            size: SIZE,
            usage: BufferUsage::TRANSFER_SRC,
        },
    )?;
    let s_req = staging.memory_requirements();
    let s_mt = physical
        .find_memory_type(
            s_req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or("no host-visible memory type")?;
    let mut s_mem = DeviceMemory::allocate(&device, s_req.size, s_mt)?;
    staging.bind_memory(&s_mem, 0)?;
    {
        let mut m = s_mem.map()?;
        let dst = m.as_slice_mut();
        for (i, v) in DATA.iter().enumerate() {
            dst[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    println!("[OK] Wrote {} bytes to staging buffer", SIZE);

    // 2. Device-local buffer (TRANSFER_DST | STORAGE).
    let gpu_buf = Buffer::new(
        &device,
        BufferCreateInfo {
            size: SIZE,
            usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        },
    )?;
    let g_req = gpu_buf.memory_requirements();
    let g_mt = physical
        .find_memory_type(g_req.memory_type_bits, MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| {
            physical.find_memory_type(g_req.memory_type_bits, MemoryPropertyFlags::HOST_VISIBLE)
        })
        .ok_or("no device-local memory type")?;
    let g_mem = DeviceMemory::allocate(&device, g_req.size, g_mt)?;
    gpu_buf.bind_memory(&g_mem, 0)?;

    // 3. One-shot copy. The entire record-submit-wait dance in one call.
    queue.one_shot(&device, qf, |rec| {
        rec.copy_buffer(&staging, &gpu_buf, &[BufferCopy::full(SIZE)]);
        Ok(())
    })?;
    println!("[OK] Copied staging → device-local via queue.one_shot()");

    // 4. Read it back (copy device → staging, then map).
    queue.one_shot(&device, qf, |rec| {
        rec.copy_buffer(&gpu_buf, &staging, &[BufferCopy::full(SIZE)]);
        rec.memory_barrier(
            PipelineStage::TRANSFER,
            PipelineStage::HOST,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::HOST_READ,
        );
        Ok(())
    })?;
    {
        let m = s_mem.map()?;
        let bytes = m.as_slice();
        for (i, &expected) in DATA.iter().enumerate() {
            let got = u32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]);
            assert_eq!(
                got, expected,
                "element {i} mismatch: got {got}, expected {expected}"
            );
        }
    }
    println!("[OK] Read-back verified: all {} elements match", DATA.len());

    device.wait_idle()?;
    println!("\n=== buffer_upload example PASSED ===");
    Ok(())
}
