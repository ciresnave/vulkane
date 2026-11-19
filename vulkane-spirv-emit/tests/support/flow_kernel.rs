// SPDX-License-Identifier: MIT OR Apache-2.0
//! The row 8g test kernel, shared by the unit tests (`src/kernel_tests.rs`) and the live test
//! (`tests/kernel_flow_live.rs`) so both lower the same kernel, plus the host reference it is
//! checked against.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr as E;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind::U32;

fn tid() -> E {
    E::Exec(ExecId::ThreadIdx(0))
}

fn c(v: i64) -> E {
    E::ConstInt(v, U32)
}

fn sub(a: E, b: E) -> E {
    E::Sub(Box::new(a), Box::new(b))
}

pub(crate) fn bin(op: unpopped::ir::BinaryOp, a: E, b: E) -> E {
    E::Binary(op, Box::new(a), Box::new(b))
}

fn add(a: E, b: E) -> E {
    E::Add(Box::new(a), Box::new(b))
}

/// Nested control flow: a `For` with a `Break` and a `Continue` under `If`s, a `While` and an early
/// `Return`. Thread `t` leaves the `For` at `i == t`, sums the even `i` it passed, reduces the sum
/// modulo 40 by repeated subtraction, and stores it; thread 7 returns before the store.
pub(crate) fn flow_kernel() -> KernelFn {
    use unpopped::ir::BinaryOp::{BitAnd, CmpEq, CmpGe, CmpNe};
    let (t, acc, i, j) = (E::Local(0), E::Local(1), E::Local(2), E::Local(3));
    let body = vec![
        Stmt::Let {
            local: 0,
            init: tid(),
        },
        Stmt::Let {
            local: 1,
            init: c(0),
        },
        Stmt::For {
            local: 2,
            start: c(0),
            end: c(32),
            step: c(1),
            unroll: unpopped::kernel::Unroll::None,
            body: Block(vec![
                Stmt::If {
                    cond: bin(CmpEq, i.clone(), t.clone()),
                    then: Block(vec![Stmt::Break]),
                    els: Block(vec![]),
                },
                Stmt::If {
                    cond: bin(CmpNe, bin(BitAnd, i.clone(), c(1)), c(0)),
                    then: Block(vec![Stmt::Continue]),
                    els: Block(vec![]),
                },
                Stmt::Assign {
                    local: 1,
                    value: add(acc.clone(), i),
                },
            ]),
        },
        Stmt::Let {
            local: 3,
            init: acc,
        },
        Stmt::While {
            cond: bin(CmpGe, j.clone(), c(40)),
            body: Block(vec![Stmt::Assign {
                local: 3,
                value: sub(j.clone(), c(40)),
            }]),
        },
        Stmt::If {
            cond: bin(CmpEq, t.clone(), c(7)),
            then: Block(vec![Stmt::Return]),
            els: Block(vec![]),
        },
        Stmt::Store {
            place: Place::Logical {
                buffer: 1,
                coords: vec![t],
            },
            value: j,
        },
    ];
    let mut k = KernelFn::new(
        "flow",
        U32,
        vec![
            Param::buffer("input", U32, BufferRole::Read, None),
            Param::buffer("output", U32, BufferRole::Write, None),
        ],
        vec![LocalDecl::scalar(U32); 4],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([256, 1, 1], [1, 1, 1]));
    k
}

/// What thread `t` stores, or `None` when it returns first: the even `i` below `min(t, 32)` summed,
/// then reduced modulo 40.
#[allow(dead_code)]
pub(crate) fn expected(t: u32) -> Option<u32> {
    if t == 7 {
        return None;
    }
    let acc: u32 = (0..32).take_while(|&i| i != t).filter(|i| i % 2 == 0).sum();
    Some(acc % 40)
}
