// SPDX-License-Identifier: MIT OR Apache-2.0
//! IR to WGSL compute-shader lowering for [Unpopped](https://crates.io/crates/unpopped).
//!
//! Vulkane owns WGSL in the CireSnave ecosystem (ruling 2026-10-30). [`Wgsl`] emits a WGSL compute kernel for the scalar
//! contiguous Elementwise path: `var<storage, read> input{K}: array<T>` operands, one `var<storage, read_write> output:
//! array<T>`, and a `@compute @workgroup_size(256, 1, 1)` entry taking `@builtin(global_invocation_id)`.
//!
//! ## Its own spellers, over the shared walker (design note W3)
//!
//! WGSL is not C-family (no ternary, `select(f, t, cond)`, `var<storage>` declarations, `@compute` entries), so the C
//! spellers in `unpopped::cfamily` do not fit. The body math still lowers through the same language-neutral
//! [`unpopped::backend::lower_dag`]; only the per-op spellers (the `spell` module) and the harness are WGSL's. That is the shape
//! `fishstringer-slang-emit` has.
//!
//! ## Dtypes and declines
//!
//! `f32` (and `F32Strict`), `i32` and `u32` are spelled. Every other dtype is a typed decline, **never an approximation**:
//! WGSL has no `f64`, no 64-bit or 8-bit integers (and no 16-bit integers), and `f16` needs `enable f16;`, which this
//! version does not emit (ROADMAP 8e). Ops with no exact spelling decline with the op named: the `erf` family, `copysign`,
//! `nextafter`, a fused `fma` (WGSL's `fma` is not guaranteed to round once), casts, integer division and remainder.
//!
//! ## Subnormals
//!
//! WGSL has no way to request IEEE subnormal handling, so f32 results near zero are the device's, unlike the Slang and HLSL
//! emitters, whose compiler arguments (`-denorm preserve`) fix it. Every float kernel's header says so.
//!
//! ## Bindings
//!
//! Operands are bound `@group(0) @binding(0..N)` in declaration order: inputs, then the output.

mod spell;

use unpopped::backend::{
    Backend, DtypeSupport, GeneratedKernel, LowerError, Lowering, Spelling, lower_dag,
};
use unpopped::cfamily::{assert_no_int_div_or_const, dtype_tag, params_used};
use unpopped::ir::{ExprDag, ScalarExpr};
use unpopped::plan::{KernelPlan, Schedule};
use unpopped_vocab::{ElementKind, TargetId};

/// The WGSL compute-shader backend. Lowers a [`KernelPlan`] to `.wgsl` source.
#[derive(Copy, Clone, Debug, Default)]
pub struct Wgsl;

/// The WGSL scalar type for a dtype, or `None` if WGSL has no such type (or this version does not spell it).
fn wgsl_type(dt: ElementKind) -> Option<&'static str> {
    match dt {
        ElementKind::F32 | ElementKind::F32Strict => Some("f32"),
        ElementKind::I32 => Some("i32"),
        ElementKind::U32 => Some("u32"),
        _ => None,
    }
}

/// Why a dtype has no spelling, in WGSL's terms. The one statement: [`Backend::supports_dtype`] and [`Backend::lower`] both
/// read it, so they cannot disagree.
fn why_not(dt: ElementKind) -> &'static str {
    match dt {
        ElementKind::F64 => "WGSL has no f64",
        ElementKind::I64 | ElementKind::U64 => "WGSL has no 64-bit integers",
        ElementKind::I8 | ElementKind::U8 | ElementKind::Bool => "WGSL has no 8-bit integers",
        ElementKind::I16 | ElementKind::U16 => "WGSL has no 16-bit integers",
        ElementKind::F16 | ElementKind::Bf16 => {
            "f16 needs `enable f16;`, which this version does not emit (ROADMAP 8e); WGSL has no bf16"
        }
        _ => "no WGSL spelling for this dtype",
    }
}

/// Subnormal handling is the device's: WGSL cannot ask for IEEE behaviour. Said in the header of every float kernel.
const DENORM_NOTE: &str = "// note: WGSL cannot request IEEE subnormal handling; f32 results near zero are the device's\n";

/// Does `e` read a runtime scalar parameter? (This version is parameterless.)
fn body_has_params(e: &ScalarExpr) -> bool {
    !params_used(e).is_empty()
}

impl Backend for Wgsl {
    fn name(&self) -> &str {
        "wgsl"
    }
    fn provider(&self) -> &str {
        "vulkane"
    }

    fn supports_dtype(&self, dtype: ElementKind, _target: TargetId) -> DtypeSupport {
        if wgsl_type(dtype).is_some() {
            DtypeSupport::Supported
        } else {
            DtypeSupport::Unsupported
        }
    }

    fn lower(&self, plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        let Some(ty) = wgsl_type(plan.dtype) else {
            return Err(LowerError::UnsupportedDtype {
                dtype: plan.dtype,
                detail: format!("wgsl backend: {:?}: {}", plan.dtype, why_not(plan.dtype)),
            });
        };
        check_shape(plan)?;
        if unpopped::plan::is_int_dtype(plan.dtype) {
            // A plan-gate invariant (`build_plan` rejects these upstream), so reaching it is a caller bug.
            assert_no_int_div_or_const(plan.body, plan.dtype, false, false);
        }
        match plan.schedule {
            Schedule::Scalar => emit_scalar(plan, ty),
            other => Err(LowerError::UnsupportedSchedule {
                detail: format!(
                    "wgsl backend: the scalar contiguous Elementwise path only, got schedule {other:?}"
                ),
            }),
        }
    }
}

fn check_shape(plan: &KernelPlan<'_>) -> Result<(), LowerError> {
    let refuse = |detail: String| Err(LowerError::UnsupportedPlanShape { detail });
    if plan.n_outputs != 1 {
        return refuse(format!(
            "wgsl backend: single-output only; op '{}' has {} outputs",
            plan.op_name, plan.n_outputs
        ));
    }
    if plan.out_dtype_of(0) != plan.dtype {
        return refuse(format!(
            "wgsl backend: uniform output dtype only; op '{}' stores {:?} from a {:?} compute",
            plan.op_name,
            plan.out_dtype_of(0),
            plan.dtype
        ));
    }
    if body_has_params(plan.body) {
        return refuse(format!(
            "wgsl backend: parameterless elementwise only; op '{}' reads a runtime scalar Param",
            plan.op_name
        ));
    }
    Ok(())
}

fn emit_scalar(plan: &KernelPlan<'_>, ty: &str) -> Result<GeneratedKernel, LowerError> {
    let name = format!("unpopped_wgsl_{}_{}", plan.op_name, dtype_tag(plan.dtype));
    let dtype = plan.dtype;
    let acc = |idx: u8| Ok(Spelling::Spelled(format!("input{idx}[i]")));
    let (prelude, root) = lower_dag(
        &ExprDag::from_expr(plan.body),
        ty,
        &Lowering::builder(&acc, &|op, x| spell::unary(op, x, dtype), &|op, a, b| {
            spell::binary(op, a, b, dtype)
        })
        .select(&|c, a, b| spell::select(c, a, b, dtype))
        .constant(&spell::constant)
        .temp(&|_ty, tmp, rhs| format!("let {tmp} = {rhs};"))
        .build(),
    )?;
    Ok(GeneratedKernel::new(
        name.clone(),
        harness(plan, &name, ty, &prelude, &root),
    ))
}

fn harness(plan: &KernelPlan<'_>, name: &str, ty: &str, prelude: &[String], root: &str) -> String {
    let mut s = String::new();
    s.push_str("// Generated by unpopped (wgsl backend) - do not edit.\n");
    s.push_str(&format!(
        "// op: {} | cell: {}\n",
        plan.op_name,
        plan.key.to_token()
    ));
    if matches!(ty, "f32") {
        s.push_str(DENORM_NOTE);
    }
    for i in 0..plan.n_inputs {
        s.push_str(&format!(
            "@group(0) @binding({i}) var<storage, read> input{i}: array<{ty}>;\n"
        ));
    }
    s.push_str(&format!(
        "@group(0) @binding({}) var<storage, read_write> output: array<{ty}>;\n",
        plan.n_inputs
    ));
    s.push_str("@compute @workgroup_size(256, 1, 1)\n");
    s.push_str(&format!(
        "fn {name}(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let i = gid.x;\n"
    ));
    for decl in prelude {
        s.push_str(&format!("    {decl}\n"));
    }
    s.push_str(&format!("    output[i] = {root};\n}}\n"));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        assert_eq!(Wgsl.name(), "wgsl");
        assert_eq!(Wgsl.provider(), "vulkane");
    }

    #[test]
    fn dtype_table_and_reasons_agree() {
        for dt in [ElementKind::F32, ElementKind::I32, ElementKind::U32] {
            assert!(wgsl_type(dt).is_some(), "{dt:?}");
        }
        for dt in [
            ElementKind::F64,
            ElementKind::I64,
            ElementKind::U8,
            ElementKind::I16,
            ElementKind::F16,
        ] {
            assert!(wgsl_type(dt).is_none(), "{dt:?}");
            assert!(!why_not(dt).starts_with("no WGSL spelling"), "{dt:?}");
        }
    }
}
