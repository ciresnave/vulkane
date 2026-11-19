// SPDX-License-Identifier: MIT OR Apache-2.0
//! The row 8a/8b test kernel, shared by the unit tests (`src/kernel_tests.rs`) and the live tests
//! (`tests/kernel_barrier_live.rs`): each thread writes `in[t]` to `tile[t]`, then reads
//! `tile[n-1-t]` into `out[t]`, the block reversed through shared memory, which is right only when
//! every write is visible before the reads.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr as E;
use unpopped::kernel::{BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind::U32;

/// `in[t]` into `tile[t]`.
fn produce() -> Vec<Stmt> {
    vec![
        Stmt::Let {
            local: 0,
            init: E::Exec(ExecId::ThreadIdx(0)),
        },
        Stmt::Load {
            local: 1,
            place: Place::Logical {
                buffer: 0,
                coords: vec![E::Local(0)],
            },
        },
        Stmt::Store {
            place: Place::Alloc {
                alloc: 0,
                index: E::Local(0),
            },
            value: E::Local(1),
        },
    ]
}

/// `out[t] = tile[n-1-t]`.
fn consume(n: u32) -> Vec<Stmt> {
    let last = E::ConstInt(i64::from(n) - 1, U32);
    vec![
        Stmt::Let {
            local: 2,
            init: E::Sub(Box::new(last), Box::new(E::Local(0))),
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
                coords: vec![E::Local(0)],
            },
            value: E::Local(3),
        },
    ]
}

/// The block-reverse kernel over `n` threads. `dynamic`: the shared array is sized by scalar
/// parameter 2 (local 4), a specialization constant, instead of the static `n`.
pub(crate) fn reverse(n: u32, with_barrier: bool, dynamic: bool) -> KernelFn {
    let mut body = produce();
    if with_barrier {
        body.push(Stmt::Barrier {
            scope: Scope::Block,
            kind: BarrierKind::Full,
        });
    }
    body.extend(consume(n));
    let params = vec![
        Param::buffer("input", U32, BufferRole::Read, None),
        Param::buffer("output", U32, BufferRole::Write, None),
    ];
    let mut k = KernelFn::new(
        "reverse",
        U32,
        params,
        vec![LocalDecl::scalar(U32); 4],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([n, 1, 1], [1, 1, 1]));
    let extent = if dynamic {
        k.params.push(Param::scalar("n", U32, 4));
        k.locals.push(LocalDecl::scalar(U32));
        Extent::Param(4)
    } else {
        Extent::Static(n)
    };
    k.allocs = vec![AllocDecl::new("tile", MemorySpace::Shared, U32, extent)];
    k
}
