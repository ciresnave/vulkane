// SPDX-License-Identifier: MIT OR Apache-2.0
//! The validity oracle (ROADMAP 3d, W4): every kernel the emitter writes must parse and validate in `naga`, and lower to
//! SPIR-V. **One WGSL compiler only**: Tint is not installed here, so a module naga accepts and Tint rejects would pass.
//! Behaviour against a device is a separate test (the GPU oracle), not this one.

use unpopped::backend::{Backend, LowerError};
use unpopped::ir::{BinaryOp, Expr, OpDef, UnaryOp, input, konst};
use unpopped::try_generate;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key};
use vulkane_wgsl_emit::Wgsl;

fn key(n_operands: usize, cat: OpCategory, dt: ElementKind) -> StructureKey {
    // A small `align` defeats vectorization, so the plan is `Schedule::Scalar`.
    let a = OperandDesc::new(1, &[1 << 20], &[1], dt, 4);
    structure_key(cat, &vec![a; n_operands], ArchSku::Sm89)
}

fn emit(name: &str, n: u8, dt: ElementKind, body: Expr) -> Result<String, LowerError> {
    let cat = if n == 1 {
        OpCategory::UnaryElementwise
    } else {
        OpCategory::BinaryElementwise
    };
    let op = OpDef::elementwise(name, n, &[dt], body);
    try_generate(&op, &key(usize::from(n) + 1, cat, dt), &Wgsl).map(|k| k.source)
}

/// Parse, validate, and lower to SPIR-V; return the module's entry-point count.
fn naga_accepts(src: &str) -> Result<usize, String> {
    let module = naga::front::wgsl::parse_str(src).map_err(|e| e.emit_to_string(src))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("{e:?}"))?;
    naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
        .map_err(|e| format!("spv: {e:?}"))?;
    Ok(module.entry_points.len())
}

const F32_UNARY: [UnaryOp; 31] = [
    UnaryOp::Neg,
    UnaryOp::Abs,
    UnaryOp::Sqr,
    UnaryOp::Sqrt,
    UnaryOp::Rsqrt,
    UnaryOp::Recip,
    UnaryOp::Exp,
    UnaryOp::Log,
    UnaryOp::Tanh,
    UnaryOp::Sigmoid,
    UnaryOp::Relu,
    UnaryOp::Silu,
    UnaryOp::Sin,
    UnaryOp::Cos,
    UnaryOp::Floor,
    UnaryOp::Ceil,
    UnaryOp::Round,
    UnaryOp::Sign,
    UnaryOp::Step,
    UnaryOp::Trunc,
    UnaryOp::Exp2,
    UnaryOp::Log2,
    UnaryOp::Expm1,
    UnaryOp::Log10,
    UnaryOp::Log1p,
    UnaryOp::Sinh,
    UnaryOp::Cosh,
    UnaryOp::Tan,
    UnaryOp::Asin,
    UnaryOp::Acos,
    UnaryOp::Atan,
];

const F32_BINARY: [BinaryOp; 14] = [
    BinaryOp::Max,
    BinaryOp::Min,
    BinaryOp::Pow,
    BinaryOp::Rem,
    BinaryOp::RemTrunc,
    BinaryOp::Atan2,
    BinaryOp::FmaxIeee,
    BinaryOp::FminIeee,
    BinaryOp::CmpEq,
    BinaryOp::CmpNe,
    BinaryOp::CmpLt,
    BinaryOp::CmpLe,
    BinaryOp::CmpGt,
    BinaryOp::CmpGe,
];

const INT_BINARY: [BinaryOp; 5] = [
    BinaryOp::BitAnd,
    BinaryOp::BitOr,
    BinaryOp::BitXor,
    BinaryOp::Shl,
    BinaryOp::Shr,
];

#[test]
fn every_f32_unary_op_validates() {
    let f = ElementKind::F32;
    for op in F32_UNARY {
        let name = format!("u_{op:?}").to_lowercase();
        let src = emit(&name, 1, f, input(0).unary(op)).unwrap_or_else(|e| panic!("{op:?}: {e}"));
        assert_eq!(naga_accepts(&src), Ok(1), "{op:?}:\n{src}");
    }
}

#[test]
fn every_f32_binary_op_validates() {
    let f = ElementKind::F32;
    for op in F32_BINARY {
        let name = format!("b_{op:?}").to_lowercase();
        let src = emit(&name, 2, f, input(0).binary(op, input(1)))
            .unwrap_or_else(|e| panic!("{op:?}: {e}"));
        assert_eq!(naga_accepts(&src), Ok(1), "{op:?}:\n{src}");
    }
}

#[test]
fn arithmetic_and_integer_ops_validate_at_every_spelled_dtype() {
    for dt in [ElementKind::F32, ElementKind::I32, ElementKind::U32] {
        let bodies: [(&str, Expr); 5] = [
            ("add", input(0) + input(1)),
            ("sub", input(0) - input(1)),
            ("mul", input(0) * input(1)),
            ("mixed", input(0) * input(1) + input(0) - input(1)),
            ("fconst", input(0) + konst(2.5)),
        ];
        for (tag, body) in bodies {
            if tag == "fconst" && dt != ElementKind::F32 {
                continue; // an f64 constant has no integer meaning (plan gate)
            }
            let src = emit(&format!("a_{tag}"), 2, dt, body)
                .unwrap_or_else(|e| panic!("{tag} {dt:?}: {e}"));
            assert_eq!(naga_accepts(&src), Ok(1), "{tag} {dt:?}:\n{src}");
        }
    }
    for dt in [ElementKind::I32, ElementKind::U32] {
        for op in INT_BINARY {
            let src = emit("ib", 2, dt, input(0).binary(op, input(1)))
                .unwrap_or_else(|e| panic!("{op:?} {dt:?}: {e}"));
            assert_eq!(naga_accepts(&src), Ok(1), "{op:?} {dt:?}:\n{src}");
        }
    }
}

#[test]
fn harness_shape_is_pinned() {
    let src = emit("add", 2, ElementKind::F32, input(0) + input(1)).unwrap();
    for line in [
        "@group(0) @binding(0) var<storage, read> input0: array<f32>;",
        "@group(0) @binding(1) var<storage, read> input1: array<f32>;",
        "@group(0) @binding(2) var<storage, read_write> output: array<f32>;",
        "@compute @workgroup_size(256, 1, 1)",
        "fn unpopped_wgsl_add_f32(@builtin(global_invocation_id) gid: vec3<u32>) {",
        "output[i] = ",
    ] {
        assert!(src.contains(line), "missing `{line}`:\n{src}");
    }
    assert!(src.contains("cannot request IEEE subnormal"), "{src}");
}

/// The shift count is a `u32` in WGSL: an `i32` count is converted, a `u32` count is not. Control: the mask is present on both.
#[test]
fn shift_counts_are_u32_and_masked() {
    let shl = |dt| emit("sh", 2, dt, input(0).binary(BinaryOp::Shl, input(1))).unwrap();
    assert!(shl(ElementKind::I32).contains("<< (u32(input1[i]) & 31u)"));
    assert!(shl(ElementKind::U32).contains("<< (input1[i] & 31u)"));
}

fn declined(r: Result<String, LowerError>, needle: &str) {
    let e = r.expect_err("must decline");
    assert!(e.to_string().contains(needle), "{needle}: {e}");
}

#[test]
fn dtypes_wgsl_lacks_are_typed_declines_naming_why() {
    let add = || input(0) + input(1);
    declined(emit("a", 2, ElementKind::F64, add()), "no f64");
    declined(emit("a", 2, ElementKind::I64, add()), "no 64-bit");
    declined(emit("a", 2, ElementKind::U8, add()), "no 8-bit");
    declined(emit("a", 2, ElementKind::I16, add()), "no 16-bit");
    declined(emit("a", 2, ElementKind::F16, add()), "enable f16");
    // Control: the same op at a spelled dtype lowers.
    assert!(emit("a", 2, ElementKind::F32, add()).is_ok());
}

#[test]
fn ops_without_an_exact_spelling_decline_with_the_op_named() {
    let f = ElementKind::F32;
    declined(emit("u", 1, f, input(0).unary(UnaryOp::Erf)), "Erf");
    declined(
        emit("b", 2, f, input(0).binary(BinaryOp::Copysign, input(1))),
        "Copysign",
    );
    // A non-finite constant is a shader-creation error in WGSL, so it declines rather than emit a module naga rejects.
    declined(emit("c", 1, f, input(0) + konst(f64::INFINITY)), "Constant");
}

/// The oracle is not vacuous: a type error is rejected by the same function that accepts the emitted modules.
#[test]
fn the_oracle_rejects_a_type_error() {
    let bad = "@group(0) @binding(0) var<storage, read_write> output: array<f32>;\n\
               @compute @workgroup_size(1) fn k() { output[0] = 1u; }\n";
    assert!(naga_accepts(bad).is_err());
    assert!(naga_accepts("not wgsl").is_err());
}

#[test]
fn supports_dtype_agrees_with_lower() {
    let target = unpopped_vocab::TargetId::parse("vulkan:sg32.arith-f16-i8").expect("token");
    for dt in [
        ElementKind::F32,
        ElementKind::I32,
        ElementKind::U32,
        ElementKind::F64,
        ElementKind::I64,
        ElementKind::U8,
        ElementKind::F16,
    ] {
        let lowers = emit("a", 2, dt, input(0) + input(1)).is_ok();
        assert_eq!(
            Wgsl.supports_dtype(dt, target).is_supported(),
            lowers,
            "{dt:?}"
        );
    }
}
