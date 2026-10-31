// SPDX-License-Identifier: MIT OR Apache-2.0
//! IR to SPIR-V compute-shader lowering: the [`SpirV`] backend.
//!
//! Unlike the text backends this one emits a binary module. It builds the module with
//! [`rspirv`]'s builder and hands the words to Unpopped through
//! [`GeneratedKernel::with_words`] (Unpopped 0.37.1); the kernel's `source` carries the
//! module's disassembly, the human text the words were built from.
//!
//! ## How the body is lowered
//!
//! The body math goes through the same language-neutral [`lower_dag`] every other backend
//! uses, so sharing, hoisting and the plan gate are not reimplemented. Only the spellers
//! differ: an operand "spelling" here is a SPIR-V result id (decimal text), and each speller
//! emits its instruction into the builder and returns the new id. `lower_dag` visits a node
//! after its operands and memoizes it, so every id is defined before it is used.
//!
//! ## Module shape (Vulkan 1.1, SPIR-V 1.3)
//!
//! - `Shader`, `Logical GLSL450`, one `GLCompute` entry point named like the kernel, `LocalSize 256 1 1`.
//! - Descriptor set 0: `input0..input{n-1}` at bindings `0..n-1`, `output` at binding `n`. Each is a
//!   `StorageBuffer` block holding one run-time array of the element type.
//! - The body runs for `GlobalInvocationId.x` with **no bounds guard**: dispatch exactly `n / 256`
//!   groups, as the Slang emitter does.
//!
//! ## Subnormals
//!
//! A float kernel declares `DenormPreserve` for its width (capability `DenormPreserve`, extension
//! `SPV_KHR_float_controls`), the module-level equivalent of the `-denorm-mode-fp32 preserve`
//! argument `vulkane-slang-emit` asks its consumer to pass. A device that cannot honour it
//! must refuse the pipeline; whether a device does is a measurement (see `vulkane-slang-emit`'s
//! module docs), not something to read off feature bits.
//!
//! ## v1 scope (honest boundary)
//!
//! - Dtypes: `F32`/`F32Strict`, `F64` (`Float64`), `I32`, `U32`. Everything else is declined through
//!   [`Backend::supports_dtype`].
//! - `F64` spells only the ops GLSL.std.450 defines at 64 bits (`Sqrt`, `FAbs`, rounding, `NMin`/`NMax`,
//!   comparisons, `FRem`); the transcendentals are declined at `F64`, not approximated.
//! - Schedule: `Scalar` only, single output, no runtime `Param`s.
//! - The same declined set as `unpopped-slang`: `Erf`, `Erfc`, `Gelu`, `Lgamma`, `Cbrt`, `Asinh`,
//!   `Acosh`, `Atanh`, `Copysign`, `Nextafter`, `Fma`, `Cast`, `Bitcast`, so the emitters can be
//!   checked against each other on the same ops.

mod ops;
mod skeleton;

use std::cell::RefCell;
use std::collections::HashMap;

use rspirv::binary::{Assemble, Disassemble};
use rspirv::dr::Builder;
use rspirv::spirv::{ExecutionMode, ExecutionModel, FunctionControl, Word};
use unpopped::backend::{Backend, DtypeSupport, GeneratedKernel, LowerError, Lowering, lower_dag};
use unpopped::cfamily::{assert_no_int_div_or_const, dtype_tag};
use unpopped::ir::{DagNode, ExprDag, NodeId, ScalarExpr};
use unpopped::plan::{KernelPlan, Schedule};
use unpopped_vocab::{ElementKind, TargetId};

use ops::{Em, Family, lo};
use skeleton::declare;

/// The SPIR-V compute-shader backend. Lowers a [`KernelPlan`] to a SPIR-V module.
#[derive(Copy, Clone, Debug, Default)]
pub struct SpirV;

/// The workgroup size every emitted kernel declares (`LocalSize 256 1 1`).
pub const WORKGROUP_SIZE: u32 = 256;

/// Whether `dt` is spellable. **The one statement**: [`Backend::supports_dtype`] and
/// [`Backend::lower`] both call it. The target is not consulted: every dtype here is part of
/// Vulkan 1.1's mandatory shader model except `F64`, whose `shaderFloat64` feature the consumer
/// enables (named in the module docs), so there is no target field to read.
fn dtype_ok(dt: ElementKind, _target: TargetId) -> DtypeSupport {
    if Family::of(dt).is_some() {
        DtypeSupport::Supported
    } else {
        DtypeSupport::Unsupported
    }
}

impl Backend for SpirV {
    fn name(&self) -> &str {
        "spirv"
    }
    fn provider(&self) -> &str {
        "vulkane"
    }

    fn supports_dtype(&self, dtype: ElementKind, target: TargetId) -> DtypeSupport {
        dtype_ok(dtype, target)
    }

    fn lower(&self, plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        if !dtype_ok(plan.dtype, plan.key.target).is_supported() {
            return Err(LowerError::UnsupportedDtype {
                dtype: plan.dtype,
                detail: format!(
                    "spirv backend v1 spells f32, f64, i32 and u32 only; {:?} is declined",
                    plan.dtype
                ),
            });
        }
        check_shape(plan)?;
        // Same plan-gate backstop as the C-family emitters: infix int `/` and f64 `Const`.
        if unpopped::plan::is_int_dtype(plan.dtype) {
            assert_no_int_div_or_const(plan.body, plan.dtype, false, false);
        }
        match plan.schedule {
            Schedule::Scalar => emit_scalar(plan),
            other => Err(LowerError::UnsupportedSchedule {
                detail: format!(
                    "spirv backend v1: the scalar contiguous Elementwise path only; got schedule {other:?}"
                ),
            }),
        }
    }
}

/// Does the body read a runtime scalar `Param`? (v1 is parameterless.)
fn reads_param(body: &ScalarExpr) -> bool {
    let dag = ExprDag::from_expr(body);
    (0..dag.len()).any(|i| {
        let id = NodeId::try_from(i).expect("node count fits NodeId");
        matches!(dag.node(id), DagNode::Param(_))
    })
}

/// The plan shapes v1 emits: one output of the compute dtype, no runtime `Param`.
fn check_shape(plan: &KernelPlan<'_>) -> Result<(), LowerError> {
    if plan.n_outputs != 1 {
        return Err(LowerError::UnsupportedPlanShape {
            detail: format!(
                "spirv backend v1: single-output only; op '{}' has {} outputs",
                plan.op_name, plan.n_outputs
            ),
        });
    }
    if plan.out_dtype_of(0) != plan.dtype {
        return Err(LowerError::UnsupportedPlanShape {
            detail: format!(
                "spirv backend v1: uniform output dtype only; op '{}' stores {:?} from a {:?} compute",
                plan.op_name,
                plan.out_dtype_of(0),
                plan.dtype
            ),
        });
    }
    if reads_param(plan.body) {
        return Err(LowerError::UnsupportedPlanShape {
            detail: format!(
                "spirv backend v1: parameterless elementwise only; op '{}' reads a runtime scalar Param",
                plan.op_name
            ),
        });
    }
    Ok(())
}

/// Lower `body` through `lower_dag` with the SPIR-V spellers; returns the result id.
fn lower_body(em: &RefCell<Em>, body: &ScalarExpr) -> Result<Word, LowerError> {
    let acc = |k: u8| em.borrow_mut().load_input(k);
    let unary = |op, x: String| {
        let mut e = em.borrow_mut();
        let x = e.resolve(&x);
        e.unary(op, x)
    };
    let binary = |op, a: String, b: String| {
        let mut e = em.borrow_mut();
        let (a, b) = (e.resolve(&a), e.resolve(&b));
        e.binary(op, a, b)
    };
    let arith = |op, a: String, b: String| {
        let mut e = em.borrow_mut();
        let (a, b) = (e.resolve(&a), e.resolve(&b));
        e.arith(op, a, b)
    };
    let select = |c: String, a: String, b: String| {
        let mut e = em.borrow_mut();
        let (c, a, b) = (e.resolve(&c), e.resolve(&a), e.resolve(&b));
        e.select(c, a, b)
    };
    let constant = |v: f64| Ok(em.borrow_mut().constant(v));
    // The declaration half of hoisting: record the name, emit no text (the value is an SSA id).
    let temp = |_ctype: &str, name: &str, rhs: &str| {
        let mut e = em.borrow_mut();
        let w = e.resolve(rhs);
        e.temps.insert(name.to_string(), w);
        String::new()
    };
    let (prelude, root) = lower_dag(
        &ExprDag::from_expr(body),
        "spirv",
        &Lowering::builder(&acc, &unary, &binary)
            .arith(&arith)
            .select(&select)
            .constant(&constant)
            .temp(&temp)
            .build(),
    )?;
    // The `temp` seam above returns no text, so a prelude of empty lines is expected.
    debug_assert!(
        prelude.iter().all(String::is_empty),
        "spirv emits no prelude text"
    );
    Ok(em.borrow().resolve(&root))
}

/// Emit the module for the scalar contiguous Elementwise path.
fn emit_scalar(plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
    let fam = Family::of(plan.dtype).expect("dtype_ok implies a family");
    let name = format!("unpopped_spirv_{}_{}", plan.op_name, dtype_tag(plan.dtype));
    let mut sk = declare(fam, usize::from(plan.n_inputs));

    let main = lo(sk
        .b
        .begin_function(sk.void, None, FunctionControl::NONE, sk.fn_ty))?;
    lo(sk.b.begin_block(None))?;
    let gvec = lo(sk.b.load(sk.vec3, None, sk.gid, None, []))?;
    let index = lo(sk.b.composite_extract(sk.u32_t, None, gvec, [0]))?;

    let (output, gid) = (sk.output, sk.gid);
    let em = RefCell::new(Em {
        b: sk.b,
        fam,
        dtype: plan.dtype,
        glsl: sk.glsl,
        t: sk.t,
        boolean: sk.boolean,
        zero_u32: sk.zero_u32,
        inputs: sk.inputs,
        index,
        temps: HashMap::new(),
    });
    let root = lower_body(&em, plan.body)?;
    let mut e = em.into_inner();
    e.store_and_return(output, root)?;

    declare_entry(&mut e.b, fam, main, gid, &name);
    let module = e.b.module();
    let source = module.disassemble();
    let words = module.assemble();
    Ok(GeneratedKernel::new(name, source).with_words(words))
}

/// The entry point, its workgroup size and (for floats) the denorm mode.
fn declare_entry(b: &mut Builder, fam: Family, main: Word, gid: Word, name: &str) {
    b.entry_point(ExecutionModel::GLCompute, main, name, [gid]);
    b.execution_mode(main, ExecutionMode::LocalSize, [WORKGROUP_SIZE, 1, 1]);
    if fam.is_float() {
        let width = if fam == Family::F64 { 64 } else { 32 };
        b.execution_mode(main, ExecutionMode::DenormPreserve, [width]);
    }
}

#[cfg(test)]
mod tests;
