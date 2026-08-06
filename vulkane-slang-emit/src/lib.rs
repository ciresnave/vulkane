//! IR→Slang compute-shader lowering — the third backend (v1), and the **first
//! idiomatically non-C-family emitter**.
//!
//! [`Slang`] emits a Slang/HLSL compute kernel for the scalar contiguous
//! Elementwise path: `StructuredBuffer<T>` inputs + one `RWStructuredBuffer<T>`
//! output, a `[numthreads(256,1,1)]` entry taking `uint3 tid :
//! SV_DispatchThreadID`, and `output[i] = <body>;`. The body math lowers through
//! the SAME language-neutral [`lower_dag`] the CUDA and CpuC scalar paths use —
//! only the per-op *spellers* and the type/harness surface are Slang-specific.
//!
//! ## What is reused vs. new
//!
//! Unlike CpuC (which reuses CUDA's C spellers verbatim), Slang **cannot**: HLSL
//! uses overloaded intrinsics (`exp`/`sqrt`/`max`, not `expf`/`sqrtf`/`fmaxf`)
//! and lacks the `f`-literal-suffix and several C math functions. So this backend
//! supplies a fresh Slang unary/binary/select speller set (`slang_unary_fp` etc.)
//! over the SAME [`UnaryOp`]/[`BinaryOp`] enums, plus a `[numthreads]` harness.
//! Everything upstream of the spellers — the IR, the plan gate, `lower_dag`,
//! `const_lit` — is reused unchanged.
//!
//! ## Naming — round-trips through the Slang lifter
//!
//! Buffers are `output` / `input{K}` (not `out`/`in`, which are HLSL keywords),
//! matching [`crate::convert::lift_elementwise_slang`]'s convention — so emitted
//! Slang re-lifts to the same IR (the residue-round-trip contract the "one IR, N
//! languages" hub rests on; proven by `slang_emit_round_trips_through_the_lifter`).
//!
//! ## v1 scope (honest boundary)
//!
//! - Dtypes: `F32`/`F32Strict`→`float`, `F64`→`double`, `I32`→`int`,
//!   `I64`→`int64_t`. `F16`/`Bf16`/`S8`/`U8`/`U32` are declined (no clean scalar
//!   Slang type in the base profile) — a typed decline via [`Backend::supports_dtype`].
//! - Schedule: [`Schedule::Scalar`] only (the contiguous single-output Elementwise
//!   path). Every other schedule panics — the honest v1 boundary, same convention
//!   as the CUDA / CpuC emitters.
//! - Ops needing a polyfill (`Erf`/`Erfc`/`Gelu`/`Lgamma`/`Cbrt`/`Asinh`/`Acosh`/
//!   `Atanh`; `Copysign`/`Nextafter`) are declined — a documented follow-up.
//! - Harness is exact-dispatch (`uint i = tid.x;`, no `n` bounds guard yet) so it
//!   stays byte-parseable by the lifter; a `cbuffer`-carried `n` + guard is a
//!   follow-up. Params (scalar `p{i}` via a `cbuffer`) are likewise a follow-up —
//!   v1 is parameterless.
//!
//! ## Known C-shaped seam assumption this backend surfaces
//!
//! [`crate::backend::const_lit`] spells non-finite constants as C's
//! `NAN`/`INFINITY` macros, which are not valid Slang — so a body carrying a
//! NaN/Inf *constant* emits invalid Slang today. Finite-constant bodies (the vast
//! majority: add/mul/relu/affine/…) are unaffected. This is exactly the kind of
//! accidentally-C-shaped assumption a non-C backend is meant to shake out before
//! the emitter seam is frozen into a versioned ABI; a Slang-aware const spelling
//! is the fix (tracked as a seam follow-up).

use crate::backend::{Backend, GeneratedKernel, Lowering, lower_dag};
use crate::cfamily::{assert_no_int_div_or_const, dtype_tag};
use crate::ir::{BinaryOp, ExprDag, ScalarExpr, UnaryOp};
use crate::plan::{KernelPlan, Schedule};
use baracuda_kernel_vocab::ElementKind;

/// The Slang compute-shader backend. Lowers a [`KernelPlan`] to `.slang` source.
#[derive(Copy, Clone, Debug, Default)]
pub struct Slang;

/// Slang scalar type for a dtype, or `None` if v1 declines it.
fn slang_ctype(dt: ElementKind) -> Option<&'static str> {
    match dt {
        ElementKind::F32 | ElementKind::F32Strict => Some("float"),
        ElementKind::F64 => Some("double"),
        ElementKind::I32 => Some("int"),
        ElementKind::I64 => Some("int64_t"),
        // F16/Bf16/S8/U8/U32 declined in v1 (no clean base-profile scalar type).
        _ => None,
    }
}

impl Backend for Slang {
    fn name(&self) -> &str {
        "slang"
    }
    fn provider(&self) -> &str {
        // The in-tree Slang reference emitter is provided by the generator itself.
        "unpopped"
    }

    fn supports_dtype(&self, dtype: ElementKind) -> bool {
        slang_ctype(dtype).is_some()
    }

    fn lower(&self, plan: &KernelPlan<'_>) -> GeneratedKernel {
        let ctype = slang_ctype(plan.dtype).unwrap_or_else(|| {
            panic!(
                "slang backend v1: unsupported dtype {:?} — f32/f32s/f64/i32/i64 only \
                 (f16/bf16/s8/u8/u32 are declined; no clean base-profile scalar type)",
                plan.dtype
            )
        });
        assert!(
            plan.n_outputs == 1,
            "slang backend v1: single-output only (the N-store emitter is a follow-up); \
             op '{}' has {} outputs",
            plan.op_name,
            plan.n_outputs
        );
        assert!(
            plan.out_dtype_of(0) == plan.dtype,
            "slang backend v1: uniform output dtype only — op '{}' stores {:?} from a {:?} \
             compute (hetero out-dtype, e.g. a u8 keep-mask, is a follow-up)",
            plan.op_name,
            plan.out_dtype_of(0),
            plan.dtype
        );
        assert!(
            !body_has_params(plan.body),
            "slang backend v1: parameterless elementwise only — op '{}' reads a runtime scalar \
             Param (a cbuffer-carried param is a follow-up)",
            plan.op_name
        );
        // Int Div/Const backstop, REUSED from the CUDA emitter (same dtype-blind
        // hazard: infix `/` is device-UB at an int dtype, and a `Const` is an f64
        // literal). Mirrors CpuC::lower / Cuda::lower.
        if crate::plan::is_int_dtype(plan.dtype) {
            // v1 is Elementwise/Scalar-only (panics below on any other
            // schedule), so the reduction-predicate exemption never applies
            // here — always `false` for both flags (inert; `in_reduction`
            // false already excludes the exemption regardless of
            // `at_reduction_root`), same coverage as before Task 3b.
            assert_no_int_div_or_const(plan.body, plan.dtype, false, false);
        }
        match plan.schedule {
            Schedule::Scalar => emit_scalar_slang(plan, ctype),
            other => panic!(
                "slang backend v1: the scalar contiguous Elementwise path ONLY — got schedule \
                 {other:?}. Vectorized / Strided / Reduction / RowReduce / Contraction / Scan / \
                 Window / RowSort / Im2Col are follow-ups."
            ),
        }
    }
}

/// The scalar contiguous Elementwise emitter — the Slang twin of `emit_scalar`
/// (CUDA) / `emit_scalar_cpu` (CpuC). Same body math ([`lower_dag`] over the
/// shared seam), but a `StructuredBuffer`/`[numthreads]`/`SV_DispatchThreadID`
/// compute-shader harness instead of the `extern "C" __global__` grid-stride one.
fn emit_scalar_slang(plan: &KernelPlan<'_>, ctype: &str) -> GeneratedKernel {
    let name = format!("unpopped_slang_{}_{}", plan.op_name, dtype_tag(plan.dtype));
    let n = plan.n_inputs;
    let mut s = String::new();
    s.push_str("// Generated by baracuda-kernelgen (slang backend) — do not edit.\n");
    s.push_str(&format!(
        "// op: {} | cell: {}\n",
        plan.op_name,
        plan.key.to_token()
    ));
    for i in 0..n {
        s.push_str(&format!("StructuredBuffer<{ctype}> input{i};\n"));
    }
    s.push_str(&format!("RWStructuredBuffer<{ctype}> output;\n"));
    s.push_str("[numthreads(256, 1, 1)]\n");
    s.push_str(&format!(
        "void {name}(uint3 tid : SV_DispatchThreadID) {{\n"
    ));
    s.push_str("    uint i = tid.x;\n");
    let acc = |idx: u8| format!("input{idx}[i]");
    let (prelude, root) = lower_dag(
        &ExprDag::from_expr(plan.body),
        ctype,
        &Lowering {
            leaf: &acc,
            reduced: &|i| unreachable!("no Reduced leaf outside RowReduce: red{i}"),
            coord: &|d| {
                panic!(
                    "slang backend: Coord({d}) reached the scalar emitter — Coord bodies lower \
                     via Strided only (the linear-index kernel has no per-axis coordinates)"
                )
            },
            unary: &|op, x| slang_unary(op, x, plan.dtype),
            binary: &|op, a, b| slang_binary(op, a, b, plan.dtype),
            select: &|c, a, b| slang_select(c, a, b, plan.dtype),
        },
    );
    for decl in &prelude {
        s.push_str(&format!("    {decl}\n"));
    }
    s.push_str(&format!("    output[i] = {root};\n"));
    s.push_str("}\n");
    GeneratedKernel { name, source: s }
}

/// Does `e` read a runtime scalar [`ScalarExpr::Param`]? (v1 is parameterless.)
fn body_has_params(e: &ScalarExpr) -> bool {
    match e {
        ScalarExpr::Param(_) => true,
        ScalarExpr::Unary(_, x) => body_has_params(x),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => body_has_params(a) || body_has_params(b),
        ScalarExpr::Select(c, a, b) => {
            body_has_params(c) || body_has_params(a) || body_has_params(b)
        }
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_) => false,
    }
}

/// Lower a unary op for `dtype` in Slang. f32/f64 share ONE speller (HLSL
/// intrinsics are overloaded — no `expf` vs `exp` split); integers have no unary
/// math (the ir admissibility table rejects it, so the panic is the backstop).
fn slang_unary(op: UnaryOp, x: String, dtype: ElementKind) -> String {
    match dtype {
        ElementKind::F32 | ElementKind::F32Strict | ElementKind::F64 => slang_unary_fp(op, x),
        other => panic!(
            "slang backend: no unary math for dtype {other:?} — v1 unary is float/double \
             (integer dtypes have no unary math)"
        ),
    }
}

/// Slang unary spellers (overloaded over float/double). Intrinsics that HLSL/Slang
/// provides directly are used as-is; the compare/select-shaped ops mirror the
/// CUDA ternaries verbatim (Slang ternary syntax is identical) for matching NaN
/// semantics. Ops with no base-profile Slang intrinsic are declined.
fn slang_unary_fp(op: UnaryOp, x: String) -> String {
    match op {
        UnaryOp::Neg => format!("(-{x})"),
        UnaryOp::Abs => format!("abs({x})"),
        UnaryOp::Sqr => format!("({x}*{x})"),
        UnaryOp::Sqrt => format!("sqrt({x})"),
        UnaryOp::Rsqrt => format!("rsqrt({x})"),
        UnaryOp::Recip => format!("(1.0/{x})"),
        UnaryOp::Exp => format!("exp({x})"),
        UnaryOp::Log => format!("log({x})"),
        UnaryOp::Tanh => format!("tanh({x})"),
        UnaryOp::Sigmoid => format!("(1.0/(1.0+exp(-{x})))"),
        // NaN-propagating (matches PyTorch / the CUDA speller): `x < 0` is false
        // for NaN, so NaN passes through.
        UnaryOp::Relu => format!("({x} < 0.0 ? 0.0 : {x})"),
        UnaryOp::Silu => format!("({x}*(1.0/(1.0+exp(-{x}))))"),
        UnaryOp::Sin => format!("sin({x})"),
        UnaryOp::Cos => format!("cos({x})"),
        UnaryOp::Floor => format!("floor({x})"),
        UnaryOp::Ceil => format!("ceil({x})"),
        UnaryOp::Round => format!("round({x})"), // HLSL round = round-half-to-even (matches rintf)
        UnaryOp::Sign => format!("({x} > 0.0 ? 1.0 : ({x} < 0.0 ? -1.0 : 0.0))"),
        UnaryOp::Step => format!("({x} > 0.0 ? 1.0 : 0.0)"), // heaviside(x, 0): step(0)=0
        UnaryOp::Trunc => format!("trunc({x})"),
        UnaryOp::Exp2 => format!("exp2({x})"),
        UnaryOp::Log2 => format!("log2({x})"),
        UnaryOp::Expm1 => format!("(exp({x})-1.0)"),
        UnaryOp::Log10 => format!("(log({x})*0.4342944819032518)"), // 1/ln(10)
        UnaryOp::Log1p => format!("log(1.0+{x})"),
        UnaryOp::Sinh => format!("sinh({x})"),
        UnaryOp::Cosh => format!("cosh({x})"),
        UnaryOp::Tan => format!("tan({x})"),
        UnaryOp::Asin => format!("asin({x})"),
        UnaryOp::Acos => format!("acos({x})"),
        UnaryOp::Atan => format!("atan({x})"),
        // Declined in v1 — no base-profile Slang intrinsic; each needs a polyfill
        // (a documented follow-up). The honest boundary a non-C backend surfaces.
        UnaryOp::Erf
        | UnaryOp::Erfc
        | UnaryOp::Gelu
        | UnaryOp::Lgamma
        | UnaryOp::Cbrt
        | UnaryOp::Asinh
        | UnaryOp::Acosh
        | UnaryOp::Atanh => panic!(
            "slang backend v1: {op:?} has no base-profile Slang intrinsic — declined \
             (erf/erfc/gelu/lgamma/cbrt/asinh/acosh/atanh need polyfills; a follow-up)"
        ),
    }
}

/// Lower a non-infix binary op for `dtype` in Slang. f32/f64 share the overloaded
/// fp speller; I32/I64 route to the raw-operator int speller.
fn slang_binary(op: BinaryOp, a: String, b: String, dtype: ElementKind) -> String {
    match dtype {
        ElementKind::F32 | ElementKind::F32Strict => slang_binary_fp(op, a, b, "float"),
        ElementKind::F64 => slang_binary_fp(op, a, b, "double"),
        ElementKind::I32 | ElementKind::I64 => slang_binary_int(op, a, b, dtype),
        other => panic!("slang backend: no binary math for dtype {other:?}"),
    }
}

/// Slang fp binary spellers. `ct` (`"float"`/`"double"`) casts the compare
/// operands so the compare is decided IN THE COMPUTE DTYPE (the CUDA `binary_f32`
/// double-promotion lesson). Non-infix intrinsics are overloaded so `ct` is unused
/// by them. `Max`/`Min` stay the NaN-propagating compare-selects (torch semantics);
/// `FmaxIeee`/`FminIeee` are the NaN-suppressing intrinsics.
fn slang_binary_fp(op: BinaryOp, a: String, b: String, ct: &str) -> String {
    match op {
        // A ON TIES (`>=`/`<=`): the KISS-Ops `max_prop`/`min_prop` normative
        // decomposition (`cmp_ge`/`cmp_le` select a) — signed-zero-tie-visible
        // (see the CUDA `binary_f32` note).
        BinaryOp::Max => {
            format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} >= {b} ? {a} : {b})))")
        }
        BinaryOp::Min => {
            format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} <= {b} ? {a} : {b})))")
        }
        BinaryOp::Pow => format!("pow({a}, {b})"),
        // Floored remainder (torch.remainder, sign-of-divisor) — not fmod.
        BinaryOp::Rem => format!("({a} - floor({a} / {b}) * {b})"),
        BinaryOp::RemTrunc => format!("fmod({a}, {b})"),
        BinaryOp::Atan2 => format!("atan2({a}, {b})"),
        BinaryOp::FmaxIeee => format!("max({a}, {b})"),
        BinaryOp::FminIeee => format!("min({a}, {b})"),
        BinaryOp::CmpEq => format!("(({ct}){a} == ({ct}){b} ? 1.0 : 0.0)"),
        BinaryOp::CmpNe => format!("(({ct}){a} != ({ct}){b} ? 1.0 : 0.0)"),
        BinaryOp::CmpLt => format!("(({ct}){a} < ({ct}){b} ? 1.0 : 0.0)"),
        BinaryOp::CmpLe => format!("(({ct}){a} <= ({ct}){b} ? 1.0 : 0.0)"),
        BinaryOp::CmpGt => format!("(({ct}){a} > ({ct}){b} ? 1.0 : 0.0)"),
        BinaryOp::CmpGe => format!("(({ct}){a} >= ({ct}){b} ? 1.0 : 0.0)"),
        BinaryOp::Copysign | BinaryOp::Nextafter => panic!(
            "slang backend v1: {op:?} has no base-profile Slang intrinsic — declined (a follow-up)"
        ),
        BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor
        | BinaryOp::Shl
        | BinaryOp::Shr
        | BinaryOp::LogicalAnd
        | BinaryOp::LogicalOr
        | BinaryOp::LogicalXor => {
            panic!("slang backend: {op:?} is int-only — it has no float lowering")
        }
    }
}

/// Slang integer binary speller (I32/I64) — the raw operators, matching the CUDA
/// `binary_int` speller. Logical ops are U8(Bool)-only (declined in v1), so they
/// backstop-panic here.
fn slang_binary_int(op: BinaryOp, a: String, b: String, dtype: ElementKind) -> String {
    match op {
        BinaryOp::BitAnd => format!("({a} & {b})"),
        BinaryOp::BitOr => format!("({a} | {b})"),
        BinaryOp::BitXor => format!("({a} ^ {b})"),
        BinaryOp::Shl => format!("({a} << {b})"),
        BinaryOp::Shr => format!("({a} >> {b})"),
        other => panic!(
            "slang backend: {other:?} has no integer lowering at dtype {dtype:?} — v1 int binary \
             is bitwise/shift only (logical ops are U8-only, which v1 declines)"
        ),
    }
}

/// Ternary select for `dtype` in Slang — the identity-cast-pinned ternary (the
/// CUDA `select_f32` contract: `(ct)` casts pin the compare + arms in the compute
/// dtype against a suffix-less `Const` literal, and no arithmetic touches an arm).
fn slang_select(c: String, a: String, b: String, dtype: ElementKind) -> String {
    match dtype {
        ElementKind::F32 | ElementKind::F32Strict => {
            format!("(((float)({c})) != 0.0 ? (float)({a}) : (float)({b}))")
        }
        ElementKind::F64 => {
            format!("(((double)({c})) != 0.0 ? (double)({a}) : (double)({b}))")
        }
        other => panic!(
            "slang backend: Select has no {other:?} lowering — v1 select is float-only (f32/f64)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate;
    use crate::ir::{OpDef, input};
    use baracuda_kernel_vocab::{ArchSku, OpCategory, OperandDesc, structure_key};

    /// A binary Elementwise cell whose small `align` defeats vectorization, so
    /// `build_plan` derives [`Schedule::Scalar`] (the path Slang v1 serves).
    fn binary_scalar_key(dt: ElementKind, align: u32) -> baracuda_kernel_vocab::StructureKey {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, align);
        structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89)
    }

    fn unary_scalar_key(dt: ElementKind, align: u32) -> baracuda_kernel_vocab::StructureKey {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, align);
        structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89)
    }

    #[test]
    fn f32_elementwise_emits_a_slang_compute_shader() {
        let op = OpDef::elementwise(
            "affine",
            2,
            &[ElementKind::F32],
            input(0) * input(1) + input(0),
        );
        let k = generate(&op, &binary_scalar_key(ElementKind::F32, 4), &Slang);
        let src = &k.source;
        assert_eq!(k.name, "unpopped_slang_affine_f32");
        // The Slang compute-shader harness.
        assert!(
            src.contains("StructuredBuffer<float> input0;"),
            "in0 buffer missing:\n{src}"
        );
        assert!(
            src.contains("StructuredBuffer<float> input1;"),
            "in1 buffer missing:\n{src}"
        );
        assert!(
            src.contains("RWStructuredBuffer<float> output;"),
            "output buffer missing:\n{src}"
        );
        assert!(
            src.contains("[numthreads(256, 1, 1)]"),
            "numthreads missing:\n{src}"
        );
        assert!(
            src.contains("void unpopped_slang_affine_f32(uint3 tid : SV_DispatchThreadID)"),
            "entry signature missing:\n{src}"
        );
        assert!(
            src.contains("uint i = tid.x;"),
            "dispatch index missing:\n{src}"
        );
        // The infix body — byte-identical to the neutral driver's output.
        assert!(
            src.contains("output[i] = ((input0[i] * input1[i]) + input0[i]);"),
            "infix body missing:\n{src}"
        );
        // NONE of the CUDA launch harness tokens.
        for tok in [
            "blockIdx",
            "threadIdx",
            "__global__",
            "__restrict__",
            "gridDim",
            "expf",
        ] {
            assert!(!src.contains(tok), "CUDA token `{tok}` leaked into:\n{src}");
        }
    }

    #[test]
    fn unary_uses_overloaded_hlsl_intrinsics_not_c_suffixed() {
        use crate::ir::UnaryOp;
        let op = OpDef::elementwise("e", 1, &[ElementKind::F32], input(0).unary(UnaryOp::Exp));
        let k = generate(&op, &unary_scalar_key(ElementKind::F32, 4), &Slang);
        assert!(
            k.source.contains("output[i] = exp(input0[i]);"),
            "overloaded exp missing:\n{}",
            k.source
        );
        assert!(
            !k.source.contains("expf"),
            "C-suffixed expf leaked:\n{}",
            k.source
        );
    }

    #[test]
    fn f64_uses_double_and_same_overloaded_intrinsics() {
        use crate::ir::UnaryOp;
        let op = OpDef::elementwise("s", 1, &[ElementKind::F64], input(0).unary(UnaryOp::Sqrt));
        let k = generate(&op, &unary_scalar_key(ElementKind::F64, 8), &Slang);
        assert!(
            k.source.contains("StructuredBuffer<double> input0;"),
            "double buffer missing:\n{}",
            k.source
        );
        assert!(
            k.source.contains("output[i] = sqrt(input0[i]);"),
            "overloaded sqrt missing:\n{}",
            k.source
        );
    }

    #[test]
    fn declines_f16_and_non_scalar_dtypes_via_supports_dtype() {
        assert!(!Slang.supports_dtype(ElementKind::F16));
        assert!(!Slang.supports_dtype(ElementKind::Bf16));
        assert!(!Slang.supports_dtype(ElementKind::U32));
        assert!(!Slang.supports_dtype(ElementKind::S8));
        for dt in [
            ElementKind::F32,
            ElementKind::F64,
            ElementKind::I32,
            ElementKind::I64,
        ] {
            assert!(Slang.supports_dtype(dt), "{dt:?} should be supported");
        }
    }

    #[test]
    #[should_panic(expected = "scalar contiguous Elementwise path ONLY")]
    fn declines_non_scalar_schedule() {
        // A V4-aligned contiguous binary f32 op keys Schedule::Vectorized.
        let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
        let _ = generate(&op, &binary_scalar_key(ElementKind::F32, 256), &Slang);
    }

    /// The residue-round-trip contract: emitted Slang re-lifts to the SAME IR —
    /// proving emit↔lift symmetry (a language must be re-emittable to dump its
    /// un-liftable spans back out). Needs the `convert` feature for the lifter.
    #[cfg(feature = "convert")]
    #[test]
    fn slang_emit_round_trips_through_the_lifter() {
        use crate::convert::lift_elementwise_slang;
        use crate::ir::ScalarExpr;
        let op = OpDef::elementwise("mul", 2, &[ElementKind::F32], input(0) * input(1));
        let k = generate(&op, &binary_scalar_key(ElementKind::F32, 4), &Slang);
        let lifted = lift_elementwise_slang(&k.source, "mul", &[ElementKind::F32])
            .expect("emitted Slang must re-lift");
        assert_eq!(
            lifted.op.body,
            ScalarExpr::Mul(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(1))
            ),
            "round-trip IR mismatch; emitted source:\n{}",
            k.source
        );
    }

    /// Device/toolchain validation: every emitted kernel COMPILES with the real
    /// Slang compiler (`slangc`) to SPIR-V — the Slang analog of the ondevice
    /// nvcc harnesses, and a stronger check than the tree-sitter round-trip (it
    /// type-checks + lowers, not just parses). Covers the infix / unary-intrinsic
    /// / double / ternary / compare-cast / NaN-propagating / int-operator speller
    /// paths. Ignored — needs `slangc` on PATH.
    #[test]
    #[ignore = "requires the slang compiler (slangc) on PATH"]
    fn emitted_slang_compiles_with_slangc() {
        use crate::ir::{BinaryOp, UnaryOp};
        use std::process::Command;

        let kernels = vec![
            generate(
                &OpDef::elementwise(
                    "affine",
                    2,
                    &[ElementKind::F32],
                    input(0) * input(1) + input(0),
                ),
                &binary_scalar_key(ElementKind::F32, 4),
                &Slang,
            ),
            generate(
                &OpDef::elementwise("expo", 1, &[ElementKind::F32], input(0).unary(UnaryOp::Exp)),
                &unary_scalar_key(ElementKind::F32, 4),
                &Slang,
            ),
            generate(
                &OpDef::elementwise(
                    "sqrtd",
                    1,
                    &[ElementKind::F64],
                    input(0).unary(UnaryOp::Sqrt),
                ),
                &unary_scalar_key(ElementKind::F64, 8),
                &Slang,
            ),
            generate(
                &OpDef::elementwise(
                    "relu",
                    1,
                    &[ElementKind::F32],
                    input(0).unary(UnaryOp::Relu),
                ),
                &unary_scalar_key(ElementKind::F32, 4),
                &Slang,
            ),
            generate(
                &OpDef::elementwise(
                    "gt",
                    2,
                    &[ElementKind::F32],
                    input(0).binary(BinaryOp::CmpGt, input(1)),
                ),
                &binary_scalar_key(ElementKind::F32, 4),
                &Slang,
            ),
            generate(
                &OpDef::elementwise(
                    "maxb",
                    2,
                    &[ElementKind::F32],
                    input(0).binary(BinaryOp::Max, input(1)),
                ),
                &binary_scalar_key(ElementKind::F32, 4),
                &Slang,
            ),
            generate(
                &OpDef::elementwise(
                    "band",
                    2,
                    &[ElementKind::I32],
                    input(0).binary(BinaryOp::BitAnd, input(1)),
                ),
                &binary_scalar_key(ElementKind::I32, 4),
                &Slang,
            ),
        ];

        let dir = std::env::temp_dir();
        for k in &kernels {
            let src_path = dir.join(format!("{}.slang", k.name));
            std::fs::write(&src_path, &k.source).expect("write .slang");
            let spv_path = dir.join(format!("{}.spv", k.name));
            let out = Command::new("slangc")
                .arg(&src_path)
                .arg("-entry")
                .arg(&k.name)
                .arg("-stage")
                .arg("compute")
                .arg("-target")
                .arg("spirv")
                .arg("-o")
                .arg(&spv_path)
                .output()
                .expect("run slangc");
            assert!(
                out.status.success(),
                "slangc failed for '{}':\n--- stderr ---\n{}\n--- source ---\n{}",
                k.name,
                String::from_utf8_lossy(&out.stderr),
                k.source
            );
        }
    }
}
