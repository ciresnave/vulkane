// SPDX-License-Identifier: MIT OR Apache-2.0
//! Numerical check of `vulkane-kernels-source/kernels/softmax_f64.slang` on a
//! live device (issue #106, ROADMAP row 1c).
//!
//! The kernel used to fail `spirv-val` (`GLSL.std.450 Exp` on `double`); #119 made it
//! validate by computing `exp` with the pure-arithmetic `poly_exp_d`. Validating is not
//! the same as being right, so this runs the kernel and compares each output with a CPU
//! f64 softmax.
//!
//! Needs `slangc` (the same one `compile.sh` uses) and a device with `shaderFloat64` and
//! subgroup arithmetic. Without `slangc` it declares a skip, fatal under
//! `VULKANE_REQUIRE_SLANGC`; without a device it follows `VULKANE_REQUIRE_DEVICE` like
//! every other live test.

mod common;

use std::path::PathBuf;
use std::process::Command;

use vulkane::safe::{
    AccessFlags, ApiVersion, Buffer, BufferCreateInfo, BufferUsage, CommandPool, ComputePipeline,
    DescriptorPool, DescriptorPoolSize, DescriptorSet, DescriptorSetLayout,
    DescriptorSetLayoutBinding, DescriptorType, DeviceFeatures, DeviceMemory, Fence,
    MemoryPropertyFlags, PipelineLayout, PipelineStage, ShaderModule, ShaderStageFlags,
};

const REQUIRE_SLANGC: &str = "VULKANE_REQUIRE_SLANGC";

fn slangc() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("SLANGC") {
        return Some(PathBuf::from(p));
    }
    let name = if cfg!(windows) {
        "slangc.exe"
    } else {
        "slangc"
    };
    if let Some(sdk) = std::env::var_os("VULKAN_SDK") {
        for sub in ["Bin", "bin"] {
            let p = PathBuf::from(&sdk).join(sub).join(name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join(name))
        .find(|p| p.is_file())
}

/// `slangc`, or `None` after declaring the skip; the skip is a panic under
/// `VULKANE_REQUIRE_SLANGC`.
fn slangc_or_skip() -> Option<PathBuf> {
    let found = slangc();
    if found.is_none() {
        let msg = "SKIP: no slangc (set SLANGC, VULKAN_SDK, or put it on PATH)";
        if std::env::var(REQUIRE_SLANGC).is_ok_and(|v| !v.is_empty() && v != "0") {
            panic!("{msg}, and {REQUIRE_SLANGC} is set");
        }
        eprintln!("{msg}");
    }
    found
}

/// Compile the kernel exactly as `compile.sh` does (`-target spirv -profile glsl_450
/// -entry main`). `softmax_f64.slang` has no in-function `OpExtInstImport`, so
/// `hoist_extinst_imports.py` is a no-op for it; the assertion below keeps that true.
fn compile_kernel(slangc: &PathBuf) -> Vec<u8> {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let src = format!("{manifest}/../vulkane-kernels-source/kernels/softmax_f64.slang");
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("softmax_f64.spv");
    let st = Command::new(slangc)
        .args([
            &src, "-target", "spirv", "-profile", "glsl_450", "-entry", "main", "-o",
        ])
        .arg(&out)
        .output()
        .expect("spawn slangc");
    assert!(
        st.status.success(),
        "slangc failed: {}",
        String::from_utf8_lossy(&st.stderr)
    );
    std::fs::read(&out).expect("read spv")
}

fn cpu_softmax(row: &[f64]) -> Vec<f64> {
    let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = row.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// Deterministic rows (no rand dependency): a mix of ordinary values, a wide spread that
/// underflows `exp(x - max)` to 0, and a constant row.
fn rows(n_cols: usize) -> Vec<Vec<f64>> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let ordinary: Vec<f64> = (0..n_cols).map(|_| next() * 40.0 - 20.0).collect();
    let wide: Vec<f64> = (0..n_cols).map(|_| next() * 1600.0 - 800.0).collect();
    let tiny_gaps: Vec<f64> = (0..n_cols).map(|_| next() * 1e-3).collect();
    let constant = vec![3.25; n_cols];
    vec![ordinary, wide, tiny_gaps, constant]
}

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

type Opened = (
    vulkane::safe::Instance,
    vulkane::safe::PhysicalDevice,
    vulkane::safe::Device,
    vulkane::safe::Queue,
    u32,
);

/// Instance, device with `shaderFloat64`, and queue; `None` once the skip is declared.
fn open_device() -> Option<Opened> {
    let (inst, devices) = match common::instance_and_devices("softmax_f64", ApiVersion::V1_1) {
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
    let features = DeviceFeatures::new().with_shader_float64();
    let device = match common::create_device_on(&physical, qf, Some((&features, "shaderFloat64"))) {
        Ok(d) => d,
        Err(cause) => {
            common::skip(cause);
            return None;
        }
    };
    let queue = device.get_queue(qf, 0);
    Some((inst, physical, device, queue, qf))
}

type Dev = vulkane::safe::Device;

fn bindings_layout(device: &Dev) -> DescriptorSetLayout {
    let binding = |binding, descriptor_type| DescriptorSetLayoutBinding {
        binding,
        descriptor_type,
        descriptor_count: 1,
        stage_flags: ShaderStageFlags::COMPUTE,
    };
    DescriptorSetLayout::new(
        device,
        &[
            binding(0, DescriptorType::STORAGE_BUFFER),
            binding(1, DescriptorType::STORAGE_BUFFER),
            binding(2, DescriptorType::UNIFORM_BUFFER),
        ],
    )
    .unwrap()
}

fn pool_for_bindings(device: &Dev) -> DescriptorPool {
    let size = |descriptor_type, descriptor_count| DescriptorPoolSize {
        descriptor_type,
        descriptor_count,
    };
    DescriptorPool::new(
        device,
        1,
        &[
            size(DescriptorType::STORAGE_BUFFER, 2),
            size(DescriptorType::UNIFORM_BUFFER, 1),
        ],
    )
    .unwrap()
}

/// Record one dispatch of `n_rows` workgroups, submit it and wait.
fn dispatch_and_wait(
    device: &Dev,
    queue: &vulkane::safe::Queue,
    qf: u32,
    (pipeline, layout, dset): (&ComputePipeline, &PipelineLayout, &DescriptorSet),
    n_rows: u32,
) {
    let cmd_pool = CommandPool::new(device, qf).unwrap();
    let mut cmd = cmd_pool.allocate_primary().unwrap();
    {
        let mut rec = cmd.begin().unwrap();
        rec.bind_compute_pipeline(pipeline);
        rec.bind_compute_descriptor_sets(layout, 0, &[dset]);
        rec.dispatch(n_rows, 1, 1);
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

/// Write `bytes` at the start of a host-visible allocation.
fn write_bytes(mem: &mut DeviceMemory, bytes: &[u8]) {
    let mut m = mem.map().unwrap();
    m.as_slice_mut()[..bytes.len()].copy_from_slice(bytes);
}

/// Run the kernel over `data` (`n_rows` x `n_cols`) and return its output.
fn run(spv: &[u8], n_rows: u32, n_cols: u32, data: &[f64]) -> Option<Vec<f64>> {
    let (_inst, physical, device, queue, qf) = open_device()?;
    let bytes = (data.len() * 8) as u64;
    let storage = BufferUsage::STORAGE_BUFFER;
    let (in_buf, mut in_mem) = host_buffer(&device, &physical, bytes, storage);
    let (out_buf, mut out_mem) = host_buffer(&device, &physical, bytes, storage);
    let (par_buf, mut par_mem) = host_buffer(&device, &physical, 8, BufferUsage::UNIFORM_BUFFER);
    let input: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_bytes(&mut in_mem, &input);
    write_bytes(&mut out_mem, &vec![0xAA; data.len() * 8]);
    let params: Vec<u8> = [n_rows, n_cols]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    write_bytes(&mut par_mem, &params);

    let set_layout = bindings_layout(&device);
    let pool = pool_for_bindings(&device);
    let dset = pool.allocate(&set_layout).unwrap();
    dset.write_buffer(0, DescriptorType::STORAGE_BUFFER, &in_buf, 0, bytes);
    dset.write_buffer(1, DescriptorType::STORAGE_BUFFER, &out_buf, 0, bytes);
    dset.write_buffer(2, DescriptorType::UNIFORM_BUFFER, &par_buf, 0, 8);
    let layout = PipelineLayout::new(&device, &[&set_layout]).unwrap();
    let shader = ShaderModule::from_spirv_bytes(&device, spv).unwrap();
    let pipeline = ComputePipeline::new(&device, &layout, &shader, "main").unwrap();
    dispatch_and_wait(&device, &queue, qf, (&pipeline, &layout, &dset), n_rows);

    let m = out_mem.map().unwrap();
    let raw = m.as_slice();
    Some(
        (0..data.len())
            .map(|i| f64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().unwrap()))
            .collect(),
    )
}

/// Compare every output of `rows` against the CPU reference; returns how many were checked
/// and raises `worst` to the largest relative error seen on non-denormal outputs.
fn check_rows(n_cols: usize, rows: &[Vec<f64>], got: &[f64], worst: &mut f64) -> usize {
    let mut checked = 0;
    for (r, row) in rows.iter().enumerate() {
        let want = cpu_softmax(row);
        let out = &got[r * n_cols..(r + 1) * n_cols];
        let sum: f64 = out.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "n_cols={n_cols} row={r}: outputs sum to {sum}, not 1"
        );
        for (c, (g, w)) in out.iter().zip(&want).enumerate() {
            assert!(g.is_finite(), "n_cols={n_cols} row={r} col={c}: {g}");
            // Relative error, with an absolute floor for denormal-small outputs.
            let err = (g - w).abs() / w.abs().max(1e-300);
            if *w > 1e-290 {
                *worst = worst.max(err);
            }
            assert!(
                err < 1e-13 || (g - w).abs() < 1e-300,
                "n_cols={n_cols} row={r} col={c}: got {g:e}, want {w:e} (rel {err:e})"
            );
            checked += 1;
        }
    }
    checked
}

#[test]
fn softmax_f64_matches_cpu_reference_on_device() {
    common::require_serialization_lock();
    let Some(slangc) = slangc_or_skip() else {
        return;
    };
    let spv = compile_kernel(&slangc);

    let mut worst = 0.0f64;
    let mut checked = 0usize;
    for &n_cols in &[1usize, 7, 33, 256, 1000, 4097] {
        let rows = rows(n_cols);
        let flat: Vec<f64> = rows.iter().flatten().cloned().collect();
        let Some(got) = run(&spv, rows.len() as u32, n_cols as u32, &flat) else {
            return; // skip already declared
        };
        assert_eq!(got.len(), flat.len());
        checked += check_rows(n_cols, &rows, &got, &mut worst);
    }
    eprintln!("softmax_f64: {checked} outputs checked, worst relative error {worst:e}");
}

/// Sweep `x - max` over the kernel's whole `exp` domain. One row whose max is 0, so the
/// kernel computes `exp(x)` for each x directly; reports the first x where the result is
/// not within 1e-13 of `exp(x)/sum`.
#[test]
fn softmax_f64_exp_domain_sweep() {
    common::require_serialization_lock();
    let Some(slangc) = slangc_or_skip() else {
        return;
    };
    let spv = compile_kernel(&slangc);
    let mut row = vec![0.0f64];
    row.extend((1..=3000).map(|i| -(i as f64) * 0.25)); // 0 .. -750, past the x < -745 cutoff
    let n = row.len();
    let Some(got) = run(&spv, 1, n as u32, &row) else {
        return;
    };
    let want = cpu_softmax(&row);
    let mut bad = Vec::new();
    for i in 0..n {
        let (g, w) = (got[i], want[i]);
        if w > 1e-300 && ((g - w).abs() / w > 1e-13 || g.is_sign_negative()) {
            bad.push((row[i], g, w));
        }
    }
    eprintln!(
        "sweep: {} of {n} points off; first 5: {:?}",
        bad.len(),
        &bad[..bad.len().min(5)]
    );
    if let Some(last) = bad.last() {
        eprintln!("sweep: last bad x = {}", last.0);
    }
    assert!(bad.is_empty());
}
