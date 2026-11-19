// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP rows 8a, 8b and 8g for WGSL: the block-reverse kernel and the nested break/continue/early
//! return kernel of `vulkane-spirv-emit`'s tests, lowered by `Wgsl::lower_kernel`, parsed and
//! validated by `naga`, compiled to SPIR-V by naga and run on a device against the same host
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

use unpopped::backend::{Backend, GeneratedKernel, LowerError};
use unpopped::kernel::KernelFn;
use vulkane_spirv_emit::WORKGROUP_SIZE;
use vulkane_wgsl_emit::Wgsl;

fn lower(k: &KernelFn) -> GeneratedKernel {
    Wgsl.lower_kernel(k)
        .expect("the WGSL backend lowers this kernel")
}

fn reverse(with_barrier: bool) -> GeneratedKernel {
    lower(&reverse_kernel::reverse(
        WORKGROUP_SIZE,
        with_barrier,
        false,
    ))
}

/// Parse and validate WGSL `source` with naga; the SPIR-V words naga writes for it.
fn naga_words(name: &str, source: &str) -> Result<Vec<u32>, String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("naga validation of {name}: {e:?}"))?;
    naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
        .map_err(|e| format!("naga SPIR-V output for {name}: {e:?}"))
}

fn words(k: &GeneratedKernel) -> Vec<u32> {
    naga_words(&k.name, &k.source).unwrap_or_else(|e| {
        panic!(
            "naga rejects `{}`:
{e}
{}",
            k.name, k.source
        )
    })
}

#[test]
fn the_reverse_kernel_has_the_barrier_and_its_control_does_not() {
    let (with, without) = (reverse(true), reverse(false));
    assert_eq!(
        with.source.matches("workgroupBarrier();").count(),
        1,
        "{}",
        with.source
    );
    assert_eq!(
        without.source.matches("workgroupBarrier").count(),
        0,
        "{}",
        without.source
    );
    assert!(
        with.source.contains("var<workgroup> a0: array<u32, 256>;"),
        "{}",
        with.source
    );
    assert!(
        with.source.contains("var<storage, read> b0"),
        "{}",
        with.source
    );
    assert!(
        with.source.contains("@compute @workgroup_size(256, 1, 1)"),
        "{}",
        with.source
    );
}

#[test]
fn naga_accepts_both_kernels() {
    for k in [
        reverse(true),
        reverse(false),
        lower(&flow_kernel::flow_kernel()),
    ] {
        words(&k);
    }
}

/// ROADMAP 8b's "naga negative" did not hold when measured: naga 30.0 (all validation flags, all
/// capabilities) ACCEPTS a `workgroupBarrier()` under thread-dependent control flow, although the
/// WGSL spec requires uniform control flow there (Tint rejects it). So the guard is the walker's
/// own: `check_kernel`'s `BarrierDivergent` stops the kernel before any text is written.
#[test]
fn a_barrier_under_divergent_control_is_declined_before_any_text_is_written() {
    use unpopped::ir::{BinaryOp::CmpEq, ScalarExpr as E};
    use unpopped::kernel::{BarrierKind, Block, Stmt};
    let thread0 = E::Binary(
        CmpEq,
        Box::new(E::Exec(unpopped::exec::ExecId::ThreadIdx(0))),
        Box::new(E::ConstInt(0, unpopped_vocab::ElementKind::U32)),
    );
    let mut k = reverse_kernel::reverse(WORKGROUP_SIZE, true, false);
    k.body.0.insert(
        3,
        Stmt::If {
            cond: thread0,
            then: Block(vec![Stmt::Barrier {
                scope: unpopped::exec::Scope::Block,
                kind: BarrierKind::Full,
            }]),
            els: Block(vec![]),
        },
    );
    declined(&k, "BarrierDivergent");
    // The measurement above, kept honest: if naga starts rejecting it this line says so.
    let src = "@group(0) @binding(0) var<storage, read_write> b: array<u32>;
               @compute @workgroup_size(64)
               fn k(@builtin(local_invocation_id) lid: vec3<u32>) {
                   if (b[lid.x] != 0u) { workgroupBarrier(); }
}
";
    eprintln!(
        "naga 30.0 on a divergent barrier: accepted = {}",
        naga_words("k", src).is_ok()
    );
}

#[test]
fn a_block_reversed_through_workgroup_memory_is_exact_with_the_barrier() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("wgsl_kernel_barrier") else {
        return; // skip already declared
    };
    let input: Vec<u32> = (0..WORKGROUP_SIZE).map(|i| i * 7 + 1).collect();
    let want: Vec<u32> = input.iter().rev().copied().collect();
    let k = reverse(true);
    assert_eq!(gpu.run(&k.name, &words(&k), &[&input]), want);
    // Control, reported not asserted (a missing barrier is a race a device may not show).
    let racy = reverse(false);
    let got = gpu.run(&racy.name, &words(&racy), &[&input]);
    let wrong = got.iter().zip(&want).filter(|(g, w)| g != w).count();
    eprintln!(
        "wgsl kernel_barrier control: without the barrier {wrong} of {} cells differ",
        want.len()
    );
}

#[test]
fn nested_loops_break_continue_and_return_match_the_host_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("wgsl_kernel_flow") else {
        return;
    };
    let k = lower(&flow_kernel::flow_kernel());
    let got = gpu.run(&k.name, &words(&k), &[&[0u32; 256]]);
    for (t, &g) in got.iter().enumerate() {
        match flow_kernel::expected(t as u32) {
            Some(w) => assert_eq!(g, w, "thread {t}"),
            None => assert_eq!(g, gpu::SENTINEL, "thread {t} returned before its store"),
        }
    }
}

fn declined(k: &KernelFn, needle: &str) {
    match Wgsl.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains(needle), "wanted `{needle}` in `{detail}`");
        }
        other => panic!("wanted an UnsupportedKernel naming `{needle}`, got {other:?}"),
    }
}

#[test]
fn what_wgsl_cannot_say_is_a_typed_decline_naming_it() {
    use unpopped::exec::Scope;
    use unpopped::kernel::{BarrierKind, MemoryOrder, Stmt};
    let mut fence = reverse_kernel::reverse(WORKGROUP_SIZE, true, false);
    fence.body.0.insert(
        3,
        Stmt::Fence {
            scope: Scope::Block,
            order: MemoryOrder::AcqRel,
        },
    );
    declined(&fence, "Fence");

    let mut subgroup = reverse_kernel::reverse(WORKGROUP_SIZE, true, false);
    subgroup.body.0.insert(
        3,
        Stmt::Barrier {
            scope: Scope::Subgroup,
            kind: BarrierKind::Full,
        },
    );
    declined(&subgroup, "Barrier");

    let mut float = reverse_kernel::reverse(WORKGROUP_SIZE, true, false);
    float.compute = unpopped_vocab::ElementKind::F32;
    declined(&float, "integer kernels only");
}
