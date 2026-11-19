// SPDX-License-Identifier: MIT OR Apache-2.0
//! Expressions: lowered through the language-neutral `lower_dag` with the dialect's spellers.

use unpopped::backend::{Decline, DeclinedOp, LowerError, Lowering, Spelling, lower_dag};
use unpopped::exec::ExecId;
use unpopped::ir::{ArithOp, BinaryOp, ExprDag, ScalarExpr};
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

/// The seams `lower_dag` calls, over one dialect and one kernel.
struct Spell<'a> {
    d: &'a dyn Dialect,
    kernel: &'a KernelFn,
    sc: &'a Scope,
}

impl Spell<'_> {
    fn exec(&self, id: ExecId) -> Result<Spelling, LowerError> {
        let signed = self.sc.signed;
        match launch_const(self.sc, id) {
            Some(v) => Ok(Spelling::Spelled(self.d.lit(i64::from(v), signed))),
            None => match self.d.exec(id, signed) {
                Some(s) => Ok(Spelling::Spelled(s)),
                None => declined(format!("execution id {id:?} is not lowered")),
            },
        }
    }

    fn binary(&self, op: BinaryOp, a: &str, b: &str) -> Result<Spelling, LowerError> {
        match self.d.binary(op, a, b, self.sc.signed) {
            Some(s) => Ok(Spelling::Spelled(s)),
            None => declined(format!(
                "binary op {op:?} has no integer spelling in {}",
                self.d.name()
            )),
        }
    }

    fn arith(&self, op: ArithOp, a: &str, b: &str) -> Result<Spelling, LowerError> {
        match self.d.arith(op, a, b) {
            Some(s) => Ok(Spelling::Spelled(s)),
            None => declined(format!(
                "arithmetic {op:?} is not lowered (integer division is not total here)"
            )),
        }
    }

    fn const_int(&self, v: i64, t: ElementKind) -> Result<Spelling, LowerError> {
        let signed = self.sc.signed;
        let fits = if signed {
            i32::try_from(v).is_ok()
        } else {
            u32::try_from(v).is_ok()
        };
        if t == self.kernel.compute && fits {
            Ok(Spelling::Spelled(self.d.lit(v, signed)))
        } else {
            declined(format!(
                "ConstInt({v}, {t:?}) is not exact at {:?}",
                self.kernel.compute
            ))
        }
    }
}

/// The hoisted declarations (in order) and the root spelling of `e`.
pub(crate) fn lower(
    d: &dyn Dialect,
    kernel: &KernelFn,
    sc: &Scope,
    e: &ScalarExpr,
) -> Result<(Vec<String>, String), LowerError> {
    let (sp, signed) = (Spell { d, kernel, sc }, sc.signed);
    let tys: Vec<Ty> = kernel.locals.iter().map(|l| l.ty()).collect();
    let leaf = |k: u8| {
        declined(format!(
            "an operand leaf Input({k}) is not a kernel expression"
        ))
    };
    let local = |i: u32| Ok(Spelling::Spelled(format!("l{i}")));
    let unary =
        |op, _: String| declined(format!("unary op {op:?} is not lowered (integer kernels)"));
    let binary = |op, a: String, b: String| sp.binary(op, &a, &b);
    let arith = |op, a: String, b: String| sp.arith(op, &a, &b);
    let select =
        |c: String, a: String, b: String| Ok(Spelling::Spelled(d.select(&c, &a, &b, signed)));
    let constant = |_: f64| declined("a float constant in an integer kernel; use ConstInt");
    let const_int = |v: i64, t: ElementKind| sp.const_int(v, t);
    let exec = |id| sp.exec(id);
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
fn launch_const(sc: &Scope, id: ExecId) -> Option<u32> {
    use unpopped::exec::ExecId::{BlockDim, GridDim};
    match id {
        BlockDim(a) if a < 3 => Some(sc.block[usize::from(a)]),
        GridDim(a) if a < 3 => Some(sc.grid[usize::from(a)]),
        _ => None,
    }
}
