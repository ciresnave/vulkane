// SPDX-License-Identifier: MIT OR Apache-2.0
//! The statements declined for possible patent exposure (board 434 C extended; NVIDIA US11803380B2, US12204897B2)
//! are refused **by name** with a typed error, at the top level and at any `If`/`For`/`While` depth.
//!
//! Controls: a kernel without a declined statement is still refused (this emitter does not lower kernels yet) but the
//! refusal does NOT carry the declined marker, so the assertion below can tell the two refusals apart; and the
//! walk visits all seven names, so a name dropped from the table fails here.

use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{ReduceOp, ScalarExpr};
use unpopped::kernel::{Block, BufferRole, CollectiveKind, KernelFn, Param, Place, Stmt};
use unpopped_vocab::ElementKind;
use vulkane_spirv_emit::SpirV;

const MARKER_434C: &str = "declined: possible exposure, board 434 C";
const MARKER_EXT: &str = "declined: possible exposure, board 434 C extended";

/// A nesting shape: its label and how it wraps a statement.
type Shape = (&'static str, Box<dyn Fn(Stmt) -> Vec<Stmt>>);

fn u32c(v: u64) -> ScalarExpr {
    ScalarExpr::ConstInt(v as _, ElementKind::U32)
}

fn bar() -> Place {
    Place::Alloc {
        alloc: 0,
        index: u32c(0),
    }
}

fn kernel(body: Vec<Stmt>) -> KernelFn {
    let mut k = KernelFn::new(
        "patent",
        ElementKind::U32,
        vec![Param::buffer(
            "out",
            ElementKind::U32,
            BufferRole::ReadWrite,
            Some(1),
        )],
        vec![],
        Block(body),
    )
    .with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]));
    k.allocs = vec![AllocDecl::new(
        "bar",
        MemorySpace::Shared,
        ElementKind::U64,
        Extent::Static(1),
    )];
    k
}

/// Every declined statement, with the name the refusal must carry and the ruling text it must carry.
fn declined() -> Vec<(&'static str, &'static str, Stmt)> {
    let ext = MARKER_EXT;
    vec![
        (
            "CopyAsync",
            MARKER_434C,
            Stmt::CopyAsync {
                dst: bar(),
                src: bar(),
                elems: 1,
                valid: None,
                cache: Default::default(),
                l2_prefetch: None,
            },
        ),
        ("AsyncCommit", MARKER_434C, Stmt::AsyncCommit),
        ("AsyncWait", MARKER_434C, Stmt::AsyncWait { pending: 0 }),
        (
            "CollectiveTuple",
            MARKER_434C,
            Stmt::CollectiveTuple {
                kind: CollectiveKind::ReduceWith { combine: 0 },
                values: vec![u32c(1), u32c(2)],
                results: vec![0, 1],
                scope: Scope::Subgroup,
            },
        ),
        (
            "Collective",
            MARKER_434C,
            Stmt::Collective {
                kind: CollectiveKind::Reduce(ReduceOp::Sum),
                value: u32c(1),
                result: 0,
                scope: Scope::Subgroup,
            },
        ),
        (
            "MbarInit",
            ext,
            Stmt::MbarInit {
                bar: bar(),
                count: u32c(1),
            },
        ),
        (
            "MbarArrive",
            ext,
            Stmt::MbarArrive {
                bar: bar(),
                tx: None,
            },
        ),
        (
            "MbarCompleteTx",
            ext,
            Stmt::MbarCompleteTx {
                bar: bar(),
                bytes: u32c(4),
            },
        ),
        (
            "MbarWait",
            ext,
            Stmt::MbarWait {
                bar: bar(),
                parity: u32c(0),
            },
        ),
        (
            "CopyBulk",
            ext,
            Stmt::CopyBulk {
                dst: bar(),
                src: bar(),
                elems: 1,
                bar: bar(),
            },
        ),
        ("Tcgen05Commit", ext, Stmt::Tcgen05Commit { bar: bar() }),
        ("WgmmaWait", ext, Stmt::WgmmaWait { pending: 0 }),
    ]
}

fn refusal(body: Vec<Stmt>) -> String {
    match SpirV.lower_kernel(&kernel(body)) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!("expected the typed UnsupportedKernel refusal, got {other:?}"),
    }
}

fn nest_if_then(s: Stmt) -> Stmt {
    Stmt::If {
        cond: u32c(1),
        then: Block(vec![s]),
        els: Block(vec![]),
    }
}

fn nest_if_else(s: Stmt) -> Stmt {
    Stmt::If {
        cond: u32c(1),
        then: Block(vec![]),
        els: Block(vec![s]),
    }
}

fn nest_for(s: Stmt) -> Stmt {
    Stmt::For {
        local: 0,
        start: u32c(0),
        end: u32c(2),
        step: u32c(1),
        unroll: Default::default(),
        body: Block(vec![s]),
    }
}

fn nest_while(s: Stmt) -> Stmt {
    Stmt::While {
        cond: u32c(1),
        body: Block(vec![s]),
    }
}

#[test]
fn each_declined_statement_is_refused_by_name_at_every_depth() {
    assert_eq!(
        declined().len(),
        12,
        "the declined list is twelve statements"
    );
    for (name, marker, _) in declined() {
        let shapes: Vec<Shape> = vec![
            ("top level", Box::new(|s| vec![s])),
            ("if then", Box::new(|s| vec![nest_if_then(s)])),
            ("if else", Box::new(|s| vec![nest_if_else(s)])),
            ("for", Box::new(|s| vec![nest_for(s)])),
            ("while", Box::new(|s| vec![nest_while(s)])),
            (
                "for>if>while",
                Box::new(|s| vec![nest_for(nest_if_else(nest_while(s)))]),
            ),
        ];
        for (where_, build) in shapes {
            let stmt = declined()
                .into_iter()
                .find(|(n, _, _)| *n == name)
                .map(|(_, _, s)| s)
                .expect("the statement exists");
            let detail = refusal(build(stmt));
            assert!(
                marker == MARKER_EXT || !detail.contains("extended"),
                "{name}: a board 434 C refusal must not claim the extension: {detail}"
            );
            assert!(
                detail.contains(&format!("Stmt::{name}")) && detail.contains(marker),
                "{name} at {where_}: the refusal must name it and carry the marker, got: {detail}"
            );
        }
    }
}

#[test]
fn a_kernel_without_a_declined_statement_is_refused_without_the_marker() {
    // The control: the same walk, no declined statement. It must not be mistaken for one.
    let detail = refusal(vec![nest_for(nest_if_then(Stmt::Return))]);
    assert!(
        !detail.contains(MARKER_434C) && !detail.contains("Stmt::"),
        "an innocent kernel must not carry the declined marker: {detail}"
    );
    let empty = refusal(vec![]);
    assert!(!empty.contains(MARKER_434C), "{empty}");
}
