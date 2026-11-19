// SPDX-License-Identifier: MIT OR Apache-2.0
//! Run one emitted elementwise SPIR-V kernel on a live device: `n` inputs bound at `0..n`, the
//! output at binding `n`, every buffer sized to whole workgroups (the kernel has no bounds
//! check). Opens the device through the safe wrapper and the shared `tests/common`, so a missing
//! device follows `VULKANE_REQUIRE_DEVICE` like every other live test.
//!
//! The same shape as the helpers in `int_shift_live.rs`, generalised to any input count.

use vulkane::safe::{
    AccessFlags, ApiVersion, Buffer, BufferCreateInfo, BufferUsage, CommandPool, ComputePipeline,
    DescriptorPool, DescriptorPoolSize, DescriptorSet, DescriptorSetLayout,
    DescriptorSetLayoutBinding, DescriptorType, DeviceFeatures, DeviceMemory, Fence,
    MemoryPropertyFlags, PipelineLayout, PipelineStage, ShaderModule, ShaderStageFlags,
    SpecializationConstants,
};
use vulkane_spirv_emit::WORKGROUP_SIZE;

use crate::common;

/// The value every output cell holds before the dispatch; a survivor is a never-wrote bug.
pub(crate) const SENTINEL: u32 = 0xAAAA_AAAA;

fn host_buffer(
    device: &vulkane::safe::Device,
    physical: &vulkane::safe::PhysicalDevice,
    size: u64,
    usage: BufferUsage,
) -> (Buffer, DeviceMemory) {
    let buffer = Buffer::new(device, BufferCreateInfo { size, usage }).unwrap();
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
    #[allow(dead_code)] // test binaries that need features use `open_with`
    pub(crate) fn open(tag: &str) -> Option<Gpu> {
        Self::open_with(tag, None)
    }

    /// [`Gpu::open`] on a device created with `features` (and the label a refusal reports).
    #[allow(dead_code)]
    pub(crate) fn open_with(tag: &str, features: Option<(&DeviceFeatures, &str)>) -> Option<Gpu> {
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
        let device = match common::create_device_on(&physical, qf, features) {
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

    /// One storage-buffer binding per buffer, in order, each covering `bytes`.
    /// With `uniform` (a buffer and its size), one more binding after them, a uniform buffer.
    fn bind(
        &self,
        bufs: &[&Buffer],
        bytes: u64,
        uniform: Option<(&Buffer, u64)>,
    ) -> (DescriptorSetLayout, DescriptorPool, DescriptorSet) {
        let device = &self.device;
        let count = bufs.len() as u32;
        let binding = |binding, descriptor_type| DescriptorSetLayoutBinding {
            binding,
            descriptor_type,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
        };
        let mut bindings: Vec<_> = (0..count)
            .map(|b| binding(b, DescriptorType::STORAGE_BUFFER))
            .collect();
        let mut sizes = vec![DescriptorPoolSize {
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: count,
        }];
        if uniform.is_some() {
            bindings.push(binding(count, DescriptorType::UNIFORM_BUFFER));
            sizes.push(DescriptorPoolSize {
                descriptor_type: DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            });
        }
        let set_layout = DescriptorSetLayout::new(device, &bindings).unwrap();
        let pool = DescriptorPool::new(device, 1, &sizes).unwrap();
        let dset = pool.allocate(&set_layout).unwrap();
        for (i, buf) in bufs.iter().enumerate() {
            dset.write_buffer(i as u32, DescriptorType::STORAGE_BUFFER, buf, 0, bytes);
        }
        if let Some((buf, size)) = uniform {
            dset.write_buffer(count, DescriptorType::UNIFORM_BUFFER, buf, 0, size);
        }
        (set_layout, pool, dset)
    }

    /// Record one dispatch of `groups` workgroups, submit it and wait.
    fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        layout: &PipelineLayout,
        dset: &DescriptorSet,
        groups: u32,
    ) {
        let cmd_pool = CommandPool::new(&self.device, self.qf).unwrap();
        let mut cmd = cmd_pool.allocate_primary().unwrap();
        {
            let mut rec = cmd.begin().unwrap();
            rec.bind_compute_pipeline(pipeline);
            rec.bind_compute_descriptor_sets(layout, 0, &[dset]);
            rec.dispatch(groups, 1, 1);
            rec.memory_barrier(
                PipelineStage::COMPUTE_SHADER,
                PipelineStage::HOST,
                AccessFlags::SHADER_WRITE,
                AccessFlags::HOST_READ,
            );
            rec.end().unwrap();
        }
        let fence = Fence::new(&self.device).unwrap();
        self.queue.submit(&[&cmd], Some(&fence)).unwrap();
        fence.wait(u64::MAX).unwrap();
    }

    /// Run `words` (entry point `name`) over `inputs` (equal length, 32-bit cells) and return the
    /// output cells, one per input element.
    #[allow(dead_code)] // each test binary includes this file and uses one of `run`, `run_spec`
    pub(crate) fn run(&self, name: &str, words: &[u32], inputs: &[&[u32]]) -> Vec<u32> {
        self.run_spec(name, words, inputs, None)
    }

    /// [`Gpu::run`] with specialization constants baked into the pipeline.
    #[allow(dead_code)]
    pub(crate) fn run_spec(
        &self,
        name: &str,
        words: &[u32],
        inputs: &[&[u32]],
        spec: Option<&SpecializationConstants>,
    ) -> Vec<u32> {
        self.run_full(name, words, inputs, spec, None)
    }

    /// [`Gpu::run`] with a uniform buffer holding `params` bound after the output (the Slang and
    /// HLSL kernel path's params block: strides and bases of the buffer views).
    #[allow(dead_code)]
    pub(crate) fn run_params(
        &self,
        name: &str,
        words: &[u32],
        inputs: &[&[u32]],
        params: &[u8],
    ) -> Vec<u32> {
        self.run_full(name, words, inputs, None, Some(params))
    }

    fn run_full(
        &self,
        name: &str,
        words: &[u32],
        inputs: &[&[u32]],
        spec: Option<&SpecializationConstants>,
        params: Option<&[u8]>,
    ) -> Vec<u32> {
        let n = inputs[0].len();
        assert!(inputs.iter().all(|i| i.len() == n), "ragged inputs");
        let groups = n.div_ceil(WORKGROUP_SIZE as usize);
        let padded = groups * WORKGROUP_SIZE as usize;
        let bytes = (padded * 4) as u64;
        let device = &self.device;

        let mut held = Vec::new();
        for data in inputs {
            let (buf, mut mem) =
                host_buffer(device, &self.physical, bytes, BufferUsage::STORAGE_BUFFER);
            write_words(&mut mem, data);
            held.push((buf, mem));
        }
        let (o_buf, mut o_mem) =
            host_buffer(device, &self.physical, bytes, BufferUsage::STORAGE_BUFFER);
        write_words(&mut o_mem, &vec![SENTINEL; padded]);

        let mut bufs: Vec<&Buffer> = held.iter().map(|(b, _)| b).collect();
        bufs.push(&o_buf);
        let pbuf = params.map(|p| {
            let (buf, mut mem) = host_buffer(
                device,
                &self.physical,
                p.len() as u64,
                BufferUsage::UNIFORM_BUFFER,
            );
            let mut m = mem.map().unwrap();
            m.as_slice_mut()[..p.len()].copy_from_slice(p);
            drop(m);
            (buf, mem, p.len() as u64)
        });
        let uniform = pbuf.as_ref().map(|(b, _, n)| (b, *n));
        let (set_layout, _pool, dset) = self.bind(&bufs, bytes, uniform);

        let layout = PipelineLayout::new(device, &[&set_layout]).unwrap();
        let shader = ShaderModule::from_spirv(device, words).unwrap();
        let pipeline = match spec {
            Some(s) => ComputePipeline::with_specialization(device, &layout, &shader, name, s),
            None => ComputePipeline::new(device, &layout, &shader, name),
        }
        .unwrap();
        self.dispatch(&pipeline, &layout, &dset, groups as u32);

        let m = o_mem.map().unwrap();
        let raw = m.as_slice();
        (0..n)
            .map(|i| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect()
    }
}
