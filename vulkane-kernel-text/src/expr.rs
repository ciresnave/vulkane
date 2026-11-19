// SPDX-License-Identifier: MIT OR Apache-2.0
//! Expressions: lowered through the language-neutral `lower_dag` with the dialect's spellers.

use unpopped::backend::{Decline, DeclinedOp, LowerError, Lowering, Spelling, lower_dag};
use unpopped::ir::{ExprDag, ScalarExpr};
use unpopped::kernel::KernelFn;
use unpopped::types::Ty;
use unpopped_vocab::ElementKind;

use crate::dialect::Dialect;
use crate::scope::Scope;

fn declined(why: impl Into<String>) -> Result<Spelling, LowerError> {
    Ok(Spelling::Declined(Decline::UnsupportedOp {
        op: DeclinedOp::Kernel,
        why: why.into(),
    }))
}

/// The hoisted declarations (in order) and the root spelling of `e`.
pub(crate) fn lower(
    d: &dyn Dialect,
    kernel: &KernelFn,
    sc: &Scope,
    e: &ScalarExpr,
) -> Result<(Vec<String>, String), LowerError> {
    let signed = sc.signed;
    let tys: Vec<Ty> = kernel.locals.iter().map(|l| l.ty()).collect();
    let leaf = |k: u8| {
        declined(format!(
            "an operand leaf Input({k}) is not a kernel expression"
        ))
    };
    let local = |i: u32| Ok(Spelling::Spelled(format!("l{i}")));
    let exec = |id| match launch_const(sc, id) {
        Some(v) => Ok(Spelling::Spelled(d.lit(i64::from(v), signed))),
        None => d.exec(id, signed).map_or_else(
            || declined(format!("execution id {id:?} is not lowered")),
            |s| Ok(Spelling::Spelled(s)),
        ),
    };
    let unary =
        |op, _: String| declined(format!("unary op {op:?} is not lowered (integer kernels)"));
    let binary = |op, a: String, b: String| match d.binary(op, &a, &b, signed) {
        Some(s) => Ok(Spelling::Spelled(s)),
        None => declined(format!(
            "binary op {op:?} has no integer spelling in {}",
            d.name()
        )),
    };
    let arith = |op, a: String, b: String| match d.arith(op, &a, &b) {
        Some(s) => Ok(Spelling::Spelled(s)),
        None => declined(format!(
            "arithmetic {op:?} is not lowered (integer division is not total here)"
        )),
    };
    let select =
        |c: String, a: String, b: String| Ok(Spelling::Spelled(d.select(&c, &a, &b, signed)));
    let constant = |_: f64| declined("a float constant in an integer kernel; use ConstInt");
    let const_int = |v: i64, t: ElementKind| {
        let fits = if signed {
            i32::try_from(v).is_ok()
        } else {
            u32::try_from(v).is_ok()
        };
        if t == kernel.compute && fits {
            Ok(Spelling::Spelled(d.lit(v, signed)))
        } else {
            declined(format!(
                "ConstInt({v}, {t:?}) is not exact at {:?}",
                kernel.compute
            ))
        }
    };
    let temp = |_: &str, name: &str, rhs: &str| d.temp(name, rhs, signed);
    let lowering = Lowering::builder(&leaf, &unary, &binary)
        .arith(&arith)
        .select(&select)
        .constant(&constant)
        .const_int(&const_int)
        .temp(&temp)
        .local(&local)
        .exec(&exec)
        .local_types(&tys)
        .compute_dtype(kernel.compute)
        .build();
    lower_dag(&ExprDag::from_expr(e), d.name(), &lowering)
}

/// `BlockDim` and `GridDim` are constants of the launch.
fn launch_const(sc: &Scope, id: unpopped::exec::ExecId) -> Option<u32> {
    use unpopped::exec::ExecId::{BlockDim, GridDim};
    match id {
        BlockDim(a) if a < 3 => Some(sc.block[usize::from(a)]),
        GridDim(a) if a < 3 => Some(sc.grid[usize::from(a)]),
        _ => None,
    }
}
