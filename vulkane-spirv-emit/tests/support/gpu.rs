// SPDX-License-Identifier: MIT OR Apache-2.0
//! Run one emitted elementwise SPIR-V kernel on a live device: `n` inputs bound at `0..n`, the
//! output at binding `n`, every buffer sized to whole workgroups (the kernel has no bounds
//! check). Opens the device through the safe wrapper and the shared `tests/common`, so a missing
//! device follows `VULKANE_REQUIRE_DEVICE` like every other live test.
//!
//! The same shape as the helpers in `int_shift_live.rs`, generalised to any input count.

use vulkane::safe::{
    AccessFlags, ApiVersion, Buffer, BufferCreateInfo, BufferUsage, CommandPool, ComputePipeline,
    DescriptorPool, DescriptorPoolSize, DescriptorSetLayout, DescriptorSetLayoutBinding,
    DescriptorType, DeviceMemory, Fence, MemoryPropertyFlags, PipelineLayout, PipelineStage,
    ShaderModule, ShaderStageFlags,
};
use vulkane_spirv_emit::WORKGROUP_SIZE;

use crate::common;

/// The value every output cell holds before the dispatch; a survivor is a never-wrote bug.
pub(crate) const SENTINEL: u32 = 0xAAAA_AAAA;

fn host_buffer(
    device: &vulkane::safe::Device,
    physical: &vulkane::safe::PhysicalDevice,
    size: u64,
) -> (Buffer, DeviceMemory) {
    let buffer = Buffer::new(
        device,
        BufferCreateInfo {
            size,
            usage: BufferUsage::STORAGE_BUFFER,
        },
    )
    .unwrap();
    let req = buffer.memory_requirements();
    let mt = physical
        .find_memory_type(
            req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
    let memory = DeviceMemory::allocate(device, req.size, mt).unwrap();
    buffer.bind_memory(&memory, 0).unwrap();
    (buffer, memory)
}

fn write_words(mem: &mut DeviceMemory, words: &[u32]) {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut m = mem.map().unwrap();
    m.as_slice_mut()[..bytes.len()].copy_from_slice(&bytes);
}

/// A device ready to run kernels.
pub(crate) struct Gpu {
    _instance: vulkane::safe::Instance,
    physical: vulkane::safe::PhysicalDevice,
    device: vulkane::safe::Device,
    queue: vulkane::safe::Queue,
    qf: u32,
}

impl Gpu {
    /// Instance, first compute device and queue; `None` once the skip is declared.
    pub(crate) fn open(tag: &str) -> Option<Gpu> {
        let (inst, devices) = match common::instance_and_devices(tag, ApiVersion::V1_1) {
            Ok(x) => x,
            Err(cause) => {
                common::skip(cause);
                return None;
            }
        };
        let (physical, qf) = match common::first_compute(devices) {
            Ok(x) => x,
            Err(cause) => {
                common::skip(cause);
                return None;
            }
        };
        let device = match common::create_device_on(&physical, qf, None) {
            Ok(d) => d,
            Err(cause) => {
                common::skip(cause);
                return None;
            }
        };
        let queue = device.get_queue(qf, 0);
        Some(Gpu {
            _instance: inst,
            physical,
            device,
            queue,
            qf,
        })
    }

    /// Run `words` (entry point `name`) over `inputs` (equal length, 32-bit cells) and return the
    /// output cells, one per input element.
    pub(crate) fn run(&self, name: &str, words: &[u32], inputs: &[&[u32]]) -> Vec<u32> {
        let n = inputs[0].len();
        assert!(inputs.iter().all(|i| i.len() == n), "ragged inputs");
        let groups = n.div_ceil(WORKGROUP_SIZE as usize);
        let padded = groups * WORKGROUP_SIZE as usize;
        let bytes = (padded * 4) as u64;
        let device = &self.device;

        let mut held = Vec::new();
        for data in inputs {
            let (buf, mut mem) = host_buffer(device, &self.physical, bytes);
            write_words(&mut mem, data);
            held.push((buf, mem));
        }
        let (o_buf, mut o_mem) = host_buffer(device, &self.physical, bytes);
        write_words(&mut o_mem, &vec![SENTINEL; padded]);

        let count = inputs.len() as u32 + 1;
        let binding = |binding| DescriptorSetLayoutBinding {
            binding,
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
        };
        let bindings: Vec<_> = (0..count).map(binding).collect();
        let set_layout = DescriptorSetLayout::new(device, &bindings).unwrap();
        let pool = DescriptorPool::new(
            device,
            1,
            &[DescriptorPoolSize {
                descriptor_type: DescriptorType::STORAGE_BUFFER,
                descriptor_count: count,
            }],
        )
        .unwrap();
        let dset = pool.allocate(&set_layout).unwrap();
        for (i, (buf, _)) in held.iter().enumerate() {
            dset.write_buffer(i as u32, DescriptorType::STORAGE_BUFFER, buf, 0, bytes);
        }
        dset.write_buffer(count - 1, DescriptorType::STORAGE_BUFFER, &o_buf, 0, bytes);

        let layout = PipelineLayout::new(device, &[&set_layout]).unwrap();
        let shader = ShaderModule::from_spirv(device, words).unwrap();
        let pipeline = ComputePipeline::new(device, &layout, &shader, name).unwrap();

        let cmd_pool = CommandPool::new(device, self.qf).unwrap();
        let mut cmd = cmd_pool.allocate_primary().unwrap();
        {
            let mut rec = cmd.begin().unwrap();
            rec.bind_compute_pipeline(&pipeline);
            rec.bind_compute_descriptor_sets(&layout, 0, &[&dset]);
            rec.dispatch(groups as u32, 1, 1);
            rec.memory_barrier(
                PipelineStage::COMPUTE_SHADER,
                PipelineStage::HOST,
                AccessFlags::SHADER_WRITE,
                AccessFlags::HOST_READ,
            );
            rec.end().unwrap();
        }
        let fence = Fence::new(device).unwrap();
        self.queue.submit(&[&cmd], Some(&fence)).unwrap();
        fence.wait(u64::MAX).unwrap();

        let m = o_mem.map().unwrap();
        let raw = m.as_slice();
        (0..n)
            .map(|i| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect()
    }
}
