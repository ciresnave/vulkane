// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP rows 8a, 8b and 8g for GLSL: the block-reverse kernel and the nested break/continue/early
//! return kernel of `vulkane-spirv-emit`'s tests, lowered by `Glsl::lower_kernel`, compiled by
//! `glslangValidator` to SPIR-V, validated by `spirv-val` and run on a device against the same host
//! references the SPIR-V emitter's tests use.
//!
//! Without a device, `glslangValidator` or `spirv-val` the tests follow `VULKANE_REQUIRE_DEVICE`,
//! `VULKANE_REQUIRE_GLSLANG` and `VULKANE_REQUIRE_SPIRV_VAL` rather than passing silently.

#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(dead_code, unreachable_pub, missing_docs)]
mod common;
#[path = "../../vulkane-spirv-emit/tests/support/flow_kernel.rs"]
mod flow_kernel;
#[path = "../../vulkane-spirv-emit/tests/support/gpu.rs"]
mod gpu;
#[path = "../../vulkane-spirv-emit/tests/support/reverse_kernel.rs"]
mod reverse_kernel;

use std::io::Write;
use std::process::{Command, Stdio};

use unpopped::backend::{Backend, GeneratedKernel, LowerError};
use unpopped::kernel::KernelFn;
use vulkane_glsl_emit::Glsl;
use vulkane_spirv_emit::WORKGROUP_SIZE;

fn lower(k: &KernelFn) -> GeneratedKernel {
    Glsl.lower_kernel(k)
        .expect("the GLSL backend lowers this kernel")
}

fn reverse(with_barrier: bool) -> GeneratedKernel {
    lower(&reverse_kernel::reverse(
        WORKGROUP_SIZE,
        with_barrier,
        false,
    ))
}

/// `glslangValidator` to SPIR-V words; `None` when it is absent (a failure under `VULKANE_REQUIRE_GLSLANG`).
fn glslang_words(k: &GeneratedKernel) -> Option<Vec<u32>> {
    let dir = tempfile::tempdir().unwrap();
    let (src, spv) = (dir.path().join("k.comp"), dir.path().join("k.spv"));
    std::fs::write(&src, &k.source).unwrap();
    let out = Command::new("glslangValidator")
        .args(["-V", "--target-env", "vulkan1.1", "-S", "comp", "-o"])
        .arg(&spv)
        .arg(&src)
        .output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            assert!(
                std::env::var_os("VULKANE_REQUIRE_GLSLANG").is_none(),
                "VULKANE_REQUIRE_GLSLANG is set but `glslangValidator` could not be run: {e}"
            );
            return None;
        }
    };
    assert!(
        out.status.success(),
        "glslangValidator failed:\n{}{}\n--- source ---\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
        k.source
    );
    let bytes = std::fs::read(&spv).unwrap();
    Some(
        bytes
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect(),
    )
}

/// `spirv-val --target-env vulkan1.1` over `words`; `None` when it is absent and not required.
fn validates(words: &[u32]) -> Option<(bool, String)> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let child = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.1", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut c| {
            let _ = c.stdin.take().expect("piped stdin").write_all(&bytes);
            c.wait_with_output()
        });
    match child {
        Ok(o) => Some((
            o.status.success(),
            String::from_utf8_lossy(&o.stderr).into_owned(),
        )),
        Err(e) => {
            assert!(
                std::env::var_os("VULKANE_REQUIRE_SPIRV_VAL").is_none(),
                "VULKANE_REQUIRE_SPIRV_VAL is set but `spirv-val` could not be run: {e}"
            );
            None
        }
    }
}

#[test]
fn the_reverse_kernel_has_the_barrier_and_its_control_does_not() {
    let (with, without) = (reverse(true), reverse(false));
    let barrier = "memoryBarrierShared(); memoryBarrierBuffer(); barrier();";
    assert_eq!(with.source.matches(barrier).count(), 1, "{}", with.source);
    assert_eq!(
        without.source.matches("barrier()").count(),
        0,
        "{}",
        without.source
    );
    assert!(
        with.source.contains("shared uint a0[256];"),
        "{}",
        with.source
    );
    assert!(with.source.contains("readonly buffer"), "{}", with.source);
    assert!(with.source.contains("writeonly buffer"), "{}", with.source);
    assert!(with.source.contains("#version 450"), "{}", with.source);
}

#[test]
fn glslang_compiles_both_kernels_and_spirv_val_accepts_them() {
    for k in [reverse(true), lower(&flow_kernel::flow_kernel())] {
        let Some(words) = glslang_words(&k) else {
            return;
        };
        if let Some((ok, err)) = validates(&words) {
            assert!(ok, "spirv-val rejected `{}`:\n{err}\n{}", k.name, k.source);
        }
    }
}

#[test]
fn a_block_reversed_through_shared_memory_is_exact_with_the_barrier() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("glsl_kernel_barrier") else {
        return; // skip already declared
    };
    let input: Vec<u32> = (0..WORKGROUP_SIZE).map(|i| i * 7 + 1).collect();
    let want: Vec<u32> = input.iter().rev().copied().collect();
    let k = reverse(true);
    let Some(words) = glslang_words(&k) else {
        return;
    };
    assert_eq!(gpu.run("main", &words, &[&input]), want);
    // Control, reported not asserted (a missing barrier is a race a device may not show).
    let racy = reverse(false);
    if let Some(words) = glslang_words(&racy) {
        let got = gpu.run("main", &words, &[&input]);
        let wrong = got.iter().zip(&want).filter(|(g, w)| g != w).count();
        eprintln!(
            "glsl kernel_barrier control: without the barrier {wrong} of {} cells differ",
            want.len()
        );
    }
}

#[test]
fn nested_loops_break_continue_and_return_match_the_host_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("glsl_kernel_flow") else {
        return;
    };
    let k = lower(&flow_kernel::flow_kernel());
    let Some(words) = glslang_words(&k) else {
        return;
    };
    let got = gpu.run("main", &words, &[&[0u32; 256]]);
    for (t, &g) in got.iter().enumerate() {
        match flow_kernel::expected(t as u32) {
            Some(w) => assert_eq!(g, w, "thread {t}"),
            None => assert_eq!(g, gpu::SENTINEL, "thread {t} returned before its store"),
        }
    }
}

fn declined(k: &KernelFn, needle: &str) {
    match Glsl.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains(needle), "wanted `{needle}` in `{detail}`");
        }
        other => panic!("wanted an UnsupportedKernel naming `{needle}`, got {other:?}"),
    }
}

#[test]
fn what_the_walker_does_not_lower_is_a_typed_decline_naming_it() {
    use unpopped::exec::{Extent, Scope};
    use unpopped::kernel::{BarrierKind, Stmt};
    let mut f = reverse_kernel::reverse(WORKGROUP_SIZE, true, false);
    f.compute = unpopped_vocab::ElementKind::F32;
    declined(&f, "integer kernels only");

    let mut dynamic = reverse_kernel::reverse(WORKGROUP_SIZE, true, true);
    dynamic.allocs[0].extent = Extent::Param(4);
    declined(&dynamic, "not a flat buffer");

    let mut device = reverse_kernel::reverse(WORKGROUP_SIZE, true, false);
    device.body.0.insert(
        3,
        Stmt::Barrier {
            scope: Scope::Device,
            kind: BarrierKind::Full,
        },
    );
    declined(&device, "Barrier");

    let mut grid = reverse_kernel::reverse(WORKGROUP_SIZE, true, false);
    grid.body.0.push(Stmt::GridDepWait);
    declined(&grid, "GridDepWait");
}
