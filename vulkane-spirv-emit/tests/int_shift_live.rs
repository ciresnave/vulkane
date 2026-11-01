// SPDX-License-Identifier: MIT OR Apache-2.0
//! The emitted `Shl`/`Shr` on `i32` and `u32` run on a live device and agree with Unpopped's
//! rule (`docs/int-div-shift-semantics.md`, rule 3): the count's bits are read as unsigned and
//! taken modulo 32; `<<` discards shifted-out bits for any `a`; `>>` is arithmetic on `i32` and
//! logical on `u32`. SPIR-V alone makes a count `>= 32` poison (3.3.14), so without the emitted
//! mask a device is free to return anything for those rows.
//!
//! Integer `Div` is not covered here: Unpopped's `Backend::lower` still panics on it for
//! integer dtypes (the plan gate is closed until the emitters adopt the guards, ROADMAP E16),
//! so the emitter cannot be handed one.
//!
//! Without a device the test follows `VULKANE_REQUIRE_DEVICE` like every other live test.

#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(dead_code, unreachable_pub, missing_docs)]
mod common;

use unpopped::ir::{BinaryOp, OpDef, input};
use unpopped::try_generate;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};
use vulkane::safe::{
    AccessFlags, ApiVersion, Buffer, BufferCreateInfo, BufferUsage, CommandPool, ComputePipeline,
    DescriptorPool, DescriptorPoolSize, DescriptorSet, DescriptorSetLayout,
    DescriptorSetLayoutBinding, DescriptorType, DeviceMemory, Fence, MemoryPropertyFlags,
    PipelineLayout, PipelineStage, ShaderModule, ShaderStageFlags,
};
use vulkane_spirv_emit::{SpirV, WORKGROUP_SIZE};

/// Dividends / values shifted: zero, one, minus one, the extremes, and two mixed bit patterns.
const VALUES: [u32; 9] = [
    0,
    1,
    0xFFFF_FFFF,
    0x8000_0000,
    0x7FFF_FFFF,
    0x1234_5678,
    0xDEAD_BEEF,
    0x0000_FFFF,
    0xFFFF_0000,
];
/// Counts: in range, the width and just past it, a byte's worth, minus one, and `MIN`.
const COUNTS: [u32; 12] = [
    0,
    1,
    7,
    31,
    32,
    33,
    63,
    64,
    255,
    0xFFFF_FFFF,
    0x8000_0000,
    0x8000_0001,
];

fn expected(op: BinaryOp, signed: bool, a: u32, count: u32) -> u32 {
    match (op, signed) {
        (BinaryOp::Shl, _) => a.wrapping_shl(count),
        (BinaryOp::Shr, true) => (a as i32).wrapping_shr(count) as u32,
        (BinaryOp::Shr, false) => a.wrapping_shr(count),
        _ => unreachable!("only Shl and Shr are exercised"),
    }
}

fn kernel_words(op: BinaryOp, dt: ElementKind) -> (String, Vec<u32>) {
    // A small `align` defeats vectorization, so the plan is `Schedule::Scalar`.
    let operand = OperandDesc::new(1, &[1 << 20], &[1], dt, 4);
    let key = structure_key(
        OpCategory::BinaryElementwise,
        &[operand, operand, operand],
        ArchSku::Sm89,
    );
    let def = OpDef::elementwise("b", 2, &[dt], input(0).binary(op, input(1)));
    let k = try_generate(&def, &key, &SpirV).expect("the SPIR-V backend spells integer shifts");
    let words = k
        .words()
        .expect("the SPIR-V backend supplies words")
        .to_vec();
    (k.name.clone(), words)
}

type Dev = vulkane::safe::Device;

fn host_buffer(
    device: &Dev,
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

type Opened = (
    vulkane::safe::Instance,
    vulkane::safe::PhysicalDevice,
    Dev,
    vulkane::safe::Queue,
    u32,
);

/// Instance, first compute device, and queue; `None` once the skip is declared.
fn open_device() -> Option<Opened> {
    let (inst, devices) = match common::instance_and_devices("int_shift", ApiVersion::V1_1) {
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
    Some((inst, physical, device, queue, qf))
}

/// Three storage-buffer bindings (inputs 0 and 1, output 2) bound to `bufs`.
fn bind_three(
    device: &Dev,
    bufs: [&Buffer; 3],
    bytes: u64,
) -> (DescriptorSetLayout, DescriptorPool, DescriptorSet) {
    let binding = |binding| DescriptorSetLayoutBinding {
        binding,
        descriptor_type: DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
        stage_flags: ShaderStageFlags::COMPUTE,
    };
    let set_layout =
        DescriptorSetLayout::new(device, &[binding(0), binding(1), binding(2)]).unwrap();
    let pool = DescriptorPool::new(
        device,
        1,
        &[DescriptorPoolSize {
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 3,
        }],
    )
    .unwrap();
    let dset = pool.allocate(&set_layout).unwrap();
    for (i, buf) in bufs.iter().enumerate() {
        dset.write_buffer(i as u32, DescriptorType::STORAGE_BUFFER, buf, 0, bytes);
    }
    (set_layout, pool, dset)
}

/// Record one dispatch of `groups` workgroups, submit it and wait.
fn dispatch_and_wait(
    dev: &Opened,
    pipeline: &ComputePipeline,
    layout: &PipelineLayout,
    dset: &DescriptorSet,
    groups: u32,
) {
    let (_, _, device, queue, qf) = dev;
    let cmd_pool = CommandPool::new(device, *qf).unwrap();
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
    let fence = Fence::new(device).unwrap();
    queue.submit(&[&cmd], Some(&fence)).unwrap();
    fence.wait(u64::MAX).unwrap();
}

/// Run one kernel over `(a, count)` pairs on the device; `None` after a declared skip.
fn run(name: &str, words: &[u32], a: &[u32], b: &[u32]) -> Option<Vec<u32>> {
    let dev = open_device()?;
    let (_, physical, device, _, _) = &dev;
    // The kernel has no bounds check: size every buffer to whole workgroups.
    let groups = a.len().div_ceil(WORKGROUP_SIZE as usize);
    let padded = groups * WORKGROUP_SIZE as usize;
    let bytes = (padded * 4) as u64;
    let (a_buf, mut a_mem) = host_buffer(device, physical, bytes);
    let (b_buf, mut b_mem) = host_buffer(device, physical, bytes);
    let (o_buf, mut o_mem) = host_buffer(device, physical, bytes);
    write_words(&mut a_mem, a);
    write_words(&mut b_mem, b);
    write_words(&mut o_mem, &vec![0xAAAA_AAAA; padded]);

    let (set_layout, _pool, dset) = bind_three(device, [&a_buf, &b_buf, &o_buf], bytes);
    let layout = PipelineLayout::new(device, &[&set_layout]).unwrap();
    let shader = ShaderModule::from_spirv(device, words).unwrap();
    let pipeline = ComputePipeline::new(device, &layout, &shader, name).unwrap();
    dispatch_and_wait(&dev, &pipeline, &layout, &dset, groups as u32);

    let m = o_mem.map().unwrap();
    let raw = m.as_slice();
    Some(
        (0..a.len())
            .map(|i| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect(),
    )
}

#[test]
fn emitted_shifts_match_the_unpopped_rule_on_a_device() {
    common::require_serialization_lock();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for &v in &VALUES {
        for &c in &COUNTS {
            a.push(v);
            b.push(c);
        }
    }
    let mut checked = 0usize;
    for (dt, signed) in [(ElementKind::I32, true), (ElementKind::U32, false)] {
        for op in [BinaryOp::Shl, BinaryOp::Shr] {
            let (name, words) = kernel_words(op, dt);
            let Some(got) = run(&name, &words, &a, &b) else {
                return; // skip already declared
            };
            for (i, g) in got.iter().enumerate() {
                let want = expected(op, signed, a[i], b[i]);
                assert_eq!(
                    *g, want,
                    "{op:?} {dt:?}: {:#010x} shifted by {:#010x} gave {g:#010x}, want {want:#010x}",
                    a[i], b[i]
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 4 * VALUES.len() * COUNTS.len());
    eprintln!("int_shift: {checked} device results agree with the Unpopped rule");
}
