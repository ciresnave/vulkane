// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP rows 8a and 8b on a live device: a block reversed through workgroup memory. Every
//! thread writes `in[t]` to `tile[t]`, then reads `tile[N-1-t]`, so the result is right only if all
//! the writes are visible before the reads, which is what `OpControlBarrier` gives.
//!
//! The barriered kernel must reverse exactly. The control (the same kernel with the barrier
//! deleted) is run too and its mismatch count is **reported, not asserted**: a missing barrier is a
//! data race, and whether a given device shows it is not something SPIR-V promises (a device that
//! runs the whole workgroup in lockstep may still come out right). The deterministic half of the
//! control is `src/kernel_tests.rs`: the module without the barrier has no `OpControlBarrier`.
//!
//! Without a device the test follows `VULKANE_REQUIRE_DEVICE` like every other live test.

#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(dead_code, unreachable_pub, missing_docs)]
mod common;
#[path = "support/gpu.rs"]
mod gpu;
#[path = "support/reverse_kernel.rs"]
mod reverse_kernel;
use reverse_kernel::reverse;

use unpopped::backend::Backend;
use vulkane::safe::SpecializationConstants;
use vulkane_spirv_emit::{SpirV, WORKGROUP_SIZE};

fn run(gpu: &gpu::Gpu, with_barrier: bool, dynamic: bool, input: &[u32]) -> Vec<u32> {
    let g = SpirV
        .lower_kernel(&reverse(WORKGROUP_SIZE, with_barrier, dynamic))
        .expect("the SPIR-V backend lowers the reverse kernel");
    let words = g.words().expect("words").to_vec();
    // SpecId 2 is the scalar parameter's index; the extent is fixed per pipeline.
    let spec = SpecializationConstants::new().add_u32(2, WORKGROUP_SIZE);
    gpu.run_spec(&g.name, &words, &[input], dynamic.then_some(&spec))
}

#[test]
fn a_block_reversed_through_workgroup_memory_is_exact_with_the_barrier() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("kernel_barrier") else {
        return; // skip already declared
    };
    let n = WORKGROUP_SIZE as usize;
    let input: Vec<u32> = (0..n as u32).map(|i| i * 7 + 1).collect();
    let want: Vec<u32> = input.iter().rev().copied().collect();

    let got = run(&gpu, true, false, &input);
    assert_eq!(got, want, "the barriered kernel must reverse exactly");

    // Control, reported not asserted (see the module docs).
    let racy = run(&gpu, false, false, &input);
    let wrong = racy.iter().zip(&want).filter(|(g, w)| g != w).count();
    eprintln!(
        "kernel_barrier control: without the barrier {wrong} of {n} cells differ on this device"
    );
}

/// Row 8a: the same reversal with the shared array sized by a specialization constant set to the
/// block size when the pipeline is created.
#[test]
fn a_specialization_sized_shared_array_reverses_exactly() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("kernel_dyn_extent") else {
        return; // skip already declared
    };
    let input: Vec<u32> = (0..WORKGROUP_SIZE).map(|i| i * 3 + 5).collect();
    let want: Vec<u32> = input.iter().rev().copied().collect();
    assert_eq!(run(&gpu, true, true, &input), want);
}
