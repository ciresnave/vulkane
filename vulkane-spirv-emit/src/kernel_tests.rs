// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tests for `Backend::lower_kernel` (ROADMAP rows 8a and 8b, first slice): a producer/consumer
//! through workgroup memory, with and without the barrier, and the typed declines.

use unpopped::backend::{Backend, GeneratedKernel, LowerError};
use unpopped::exec::{ExecId, Extent, Scope};
use unpopped::ir::ScalarExpr as E;
use unpopped::kernel::{BarrierKind, Block, KernelFn, LocalDecl, MemoryOrder, Param, Stmt};
use unpopped_vocab::ElementKind::{F32, U32};

use crate::SpirV;
use crate::tests::assert_valid;

const N: u32 = 64;

#[path = "../tests/support/reverse_kernel.rs"]
mod reverse_kernel;

/// The block-reverse kernel over `N` threads with a static shared array.
fn reverse(with_barrier: bool) -> KernelFn {
    reverse_kernel::reverse(N, with_barrier, false)
}

fn tid() -> E {
    E::Exec(ExecId::ThreadIdx(0))
}

fn c(v: i64) -> E {
    E::ConstInt(v, U32)
}

fn lower(k: &KernelFn) -> Result<GeneratedKernel, LowerError> {
    SpirV.lower_kernel(k)
}

fn count(src: &str, needle: &str) -> usize {
    src.lines().filter(|l| l.contains(needle)).count()
}

#[test]
fn the_reverse_kernel_validates_and_has_the_workgroup_shape() {
    let g = lower(&reverse(true)).expect("lowers");
    assert_valid(&g);
    assert_eq!(count(&g.source, "OpControlBarrier"), 1, "{}", g.source);
    assert!(g.source.contains("Workgroup"), "{}", g.source);
    assert!(
        g.source.contains(&format!("LocalSize {N} 1 1")),
        "{}",
        g.source
    );
    // A `Read` buffer is `NonWritable`, the `Write` one is not.
    assert_eq!(count(&g.source, "NonWritable"), 1, "{}", g.source);
}

/// Control: the instruction check sees the barrier, so its absence in the other module is a
/// real absence (the same query finds the one in the barriered module).
#[test]
fn dropping_the_barrier_drops_exactly_the_control_barrier() {
    let with = lower(&reverse(true)).expect("lowers");
    let without = lower(&reverse(false)).expect("lowers");
    assert_valid(&without);
    assert_eq!(count(&with.source, "OpControlBarrier"), 1);
    assert_eq!(count(&without.source, "OpControlBarrier"), 0);
    assert_ne!(with.words(), without.words());
}

#[test]
fn the_barrier_orders_workgroup_and_buffer_memory_at_workgroup_scope() {
    let g = lower(&reverse(true)).expect("lowers");
    let line = g
        .source
        .lines()
        .find(|l| l.contains("OpControlBarrier"))
        .unwrap();
    let ids: Vec<&str> = line
        .split_whitespace()
        .filter(|w| w.starts_with('%'))
        .collect();
    let konst = |id: &str| -> u32 {
        let def = g
            .source
            .lines()
            .find(|l| l.trim_start().starts_with(&format!("{id} = ")))
            .unwrap();
        def.split_whitespace().last().unwrap().parse().unwrap()
    };
    assert_eq!(konst(ids[0]), 2, "execution scope Workgroup");
    assert_eq!(konst(ids[1]), 2, "memory scope Workgroup");
    // AcquireRelease (0x8) | UniformMemory (0x40) | WorkgroupMemory (0x100)
    assert_eq!(konst(ids[2]), 0x148, "semantics");
}

#[test]
fn a_fence_validates_and_seqcst_collapses_to_acquire_release() {
    for (order, bits) in [
        (MemoryOrder::Acquire, 0x2 | 0x140),
        (MemoryOrder::Release, 0x4 | 0x140),
        (MemoryOrder::AcqRel, 0x8 | 0x140),
        (MemoryOrder::SeqCst, 0x8 | 0x140),
    ] {
        let mut k = reverse(true);
        k.body.0.insert(
            3,
            Stmt::Fence {
                scope: Scope::Block,
                order,
            },
        );
        let g = lower(&k).expect("lowers");
        assert_valid(&g);
        let line = g
            .source
            .lines()
            .find(|l| l.contains("OpMemoryBarrier"))
            .unwrap();
        let sem = line.split_whitespace().last().unwrap();
        let def = g
            .source
            .lines()
            .find(|l| l.trim_start().starts_with(&format!("{sem} = ")))
            .unwrap();
        assert_eq!(
            def.split_whitespace()
                .last()
                .unwrap()
                .parse::<u32>()
                .unwrap(),
            bits,
            "{order:?}"
        );
    }
}

fn declined(k: &KernelFn, needle: &str) {
    match lower(k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains(needle), "wanted `{needle}` in `{detail}`");
        }
        other => panic!("wanted an UnsupportedKernel naming `{needle}`, got {other:?}"),
    }
}

#[test]
fn what_this_slice_does_not_lower_is_a_typed_decline_naming_it() {
    let mut f = reverse(true);
    f.compute = F32;
    declined(&f, "integer kernels only");

    let mut d = reverse(true);
    d.locals.push(LocalDecl::scalar(U32));
    d.allocs[0].extent = Extent::Param(4);
    declined(&d, "check_kernel refused");

    let mut s = reverse(true);
    s.params.push(Param::scalar("n", U32, 4));
    s.locals.push(LocalDecl::scalar(U32));
    declined(&s, "is not lowered");

    let mut dev = reverse(true);
    dev.body.0.insert(
        3,
        Stmt::Barrier {
            scope: Scope::Device,
            kind: BarrierKind::Full,
        },
    );
    declined(&dev, "Barrier");

    let mut named = reverse(true);
    named.body.0.insert(
        3,
        Stmt::Barrier {
            scope: Scope::Block,
            kind: BarrierKind::Named { id: 1, count: N },
        },
    );
    declined(&named, "Barrier");

    let mut grid_dep = reverse(true);
    grid_dep.body.0.push(Stmt::GridDepWait);
    declined(&grid_dep, "GridDepWait");
}

#[path = "../tests/support/flow_kernel.rs"]
mod flow_kernel;
use flow_kernel::{bin, flow_kernel};

#[test]
fn nested_loops_with_break_continue_and_early_return_validate() {
    let g = lower(&flow_kernel()).expect("lowers");
    assert_valid(&g);
    assert_eq!(count(&g.source, "OpLoopMerge"), 2, "{}", g.source);
    assert_eq!(count(&g.source, "OpSelectionMerge"), 3, "{}", g.source);
    assert_eq!(count(&g.source, "OpReturn"), 2, "{}", g.source);
}

/// Control: the validator sees a structural fault. Swapping the merge and continue operands of the
/// first `OpLoopMerge` makes a module `spirv-val` must reject, while the intact one is accepted.
#[test]
fn a_seeded_invalid_merge_is_rejected_by_the_validator() {
    let g = lower(&flow_kernel()).expect("lowers");
    assert_valid(&g);
    let mut words = g.words().expect("words").to_vec();
    let at = (5..words.len())
        .find(|&n| words[n] & 0xFFFF == 246 && words[n] >> 16 == 4)
        .expect("an OpLoopMerge");
    words.swap(at + 1, at + 2);
    if let Some((ok, err)) = crate::tests::validate(&words) {
        assert!(
            !ok,
            "spirv-val accepted a loop whose merge and continue are swapped"
        );
        assert!(!err.is_empty());
    }
}

#[test]
fn a_barrier_under_thread_divergent_control_is_a_decline_with_the_checkers_answer() {
    use unpopped::ir::BinaryOp::CmpEq;
    let mut k = reverse(true);
    k.body.0.insert(
        3,
        Stmt::If {
            cond: bin(CmpEq, tid(), c(0)),
            then: Block(vec![Stmt::Barrier {
                scope: Scope::Block,
                kind: BarrierKind::Full,
            }]),
            els: Block(vec![]),
        },
    );
    declined(&k, "BarrierDivergent");
}

/// `reverse` with the shared array sized by a scalar parameter (local 4).
fn reverse_dynamic() -> KernelFn {
    let mut k = reverse(true);
    k.params.push(Param::scalar("n", U32, 4));
    k.locals.push(LocalDecl::scalar(U32));
    k.allocs[0].extent = Extent::Param(4);
    k
}

#[test]
fn a_dynamic_extent_is_a_specialization_constant_sized_array() {
    let g = lower(&reverse_dynamic()).expect("lowers");
    assert_valid(&g);
    assert_eq!(count(&g.source, "OpSpecConstant"), 1, "{}", g.source);
    // The scalar parameter is parameter 2: its SpecId is 2.
    assert!(g.source.contains("SpecId 2"), "{}", g.source);
    let spec = g
        .source
        .lines()
        .find(|l| l.contains("OpSpecConstant"))
        .and_then(|l| l.split_whitespace().next())
        .expect("a spec constant id");
    let array = g
        .source
        .lines()
        .find(|l| l.contains("OpTypeArray"))
        .expect("an array type");
    assert!(
        array.trim_end().ends_with(spec),
        "the array length is the spec constant: {array}"
    );
    // The static kernel has no spec constant at all (the control for the query above).
    let fixed = lower(&reverse(true)).expect("lowers");
    assert_eq!(count(&fixed.source, "OpSpecConstant"), 0);
}
