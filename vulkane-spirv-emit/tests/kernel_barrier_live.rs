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

use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr as E;
use unpopped::kernel::{BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind::U32;
use vulkane::safe::SpecializationConstants;
use vulkane_spirv_emit::{SpirV, WORKGROUP_SIZE};

/// `dynamic`: the shared array is sized by scalar parameter 2 (local 4), a specialization constant.
fn reverse(with_barrier: bool, dynamic: bool) -> KernelFn {
    let n = i64::from(WORKGROUP_SIZE);
    let tid = || E::Local(0);
    let mut body = vec![
        Stmt::Let {
            local: 0,
            init: E::Exec(ExecId::ThreadIdx(0)),
        },
        Stmt::Load {
            local: 1,
            place: Place::Logical {
                buffer: 0,
                coords: vec![tid()],
            },
        },
        Stmt::Store {
            place: Place::Alloc {
                alloc: 0,
                index: tid(),
            },
            value: E::Local(1),
        },
    ];
    if with_barrier {
        body.push(Stmt::Barrier {
            scope: Scope::Block,
            kind: BarrierKind::Full,
        });
    }
    body.extend([
        Stmt::Let {
            local: 2,
            init: E::Sub(Box::new(E::ConstInt(n - 1, U32)), Box::new(tid())),
        },
        Stmt::Load {
            local: 3,
            place: Place::Alloc {
                alloc: 0,
                index: E::Local(2),
            },
        },
        Stmt::Store {
            place: Place::Logical {
                buffer: 1,
                coords: vec![tid()],
            },
            value: E::Local(3),
        },
    ]);
    let mut k = KernelFn::new(
        "reverse_live",
        U32,
        vec![
            Param::buffer("input", U32, BufferRole::Read, None),
            Param::buffer("output", U32, BufferRole::Write, None),
        ],
        vec![LocalDecl::scalar(U32); 4],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([WORKGROUP_SIZE, 1, 1], [1, 1, 1]));
    let extent = if dynamic {
        k.params.push(Param::scalar("n", U32, 4));
        k.locals.push(LocalDecl::scalar(U32));
        Extent::Param(4)
    } else {
        Extent::Static(WORKGROUP_SIZE)
    };
    k.allocs = vec![AllocDecl::new("tile", MemorySpace::Shared, U32, extent)];
    k
}

fn run(gpu: &gpu::Gpu, with_barrier: bool, dynamic: bool, input: &[u32]) -> Vec<u32> {
    let g = SpirV
        .lower_kernel(&reverse(with_barrier, dynamic))
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
