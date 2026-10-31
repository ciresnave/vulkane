// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tests for the SPIR-V backend. Validity is judged by `spirv-val --target-env vulkan1.1`
//! (module over stdin); without it a test skips with a note unless `VULKANE_REQUIRE_SPIRV_VAL`
//! is set, in which case a missing validator fails (CI sets it).

use std::io::Write;
use std::process::{Command, Stdio};

use super::*;
use unpopped::backend::GeneratedKernel;
use unpopped::ir::{BinaryOp, OpDef, UnaryOp, input, konst};
use unpopped::{generate, try_generate};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, TargetId, structure_key};

const TARGET_ENV: &str = "vulkan1.1";

fn key(cat: OpCategory, n_operands: usize, dt: ElementKind) -> unpopped_vocab::StructureKey {
    // A small `align` defeats vectorization, so `build_plan` derives `Schedule::Scalar`.
    let a = OperandDesc::new(1, &[1 << 20], &[1], dt, 4);
    structure_key(cat, &vec![a; n_operands], ArchSku::Sm89)
}

fn unary_key(dt: ElementKind) -> unpopped_vocab::StructureKey {
    key(OpCategory::UnaryElementwise, 2, dt)
}

fn binary_key(dt: ElementKind) -> unpopped_vocab::StructureKey {
    key(OpCategory::BinaryElementwise, 3, dt)
}

/// `Some((accepted, stderr))`, or `None` when `spirv-val` is unavailable and not required.
fn validate(words: &[u32]) -> Option<(bool, String)> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let out = Command::new("spirv-val")
        .args(["--target-env", TARGET_ENV, "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            // spirv-val may reject and exit before reading everything: a broken pipe is not an error.
            let _ = child.stdin.take().expect("piped stdin").write_all(&bytes);
            child.wait_with_output()
        });
    match out {
        Ok(o) => Some((
            o.status.success(),
            String::from_utf8_lossy(&o.stderr).into_owned(),
        )),
        Err(e) => {
            assert!(
                std::env::var_os("VULKANE_REQUIRE_SPIRV_VAL").is_none(),
                "VULKANE_REQUIRE_SPIRV_VAL is set but `spirv-val` could not be run: {e}"
            );
            eprintln!("spirv-val unavailable ({e}); skipping");
            None
        }
    }
}

fn assert_valid(k: &GeneratedKernel) {
    let words = k.words().expect("the SPIR-V backend supplies words");
    if let Some((ok, err)) = validate(words) {
        assert!(
            ok,
            "spirv-val rejected `{}`:\n{err}\n--- disassembly ---\n{}",
            k.name, k.source
        );
    }
}

fn gen_unary(op: UnaryOp, dt: ElementKind) -> Result<GeneratedKernel, LowerError> {
    let def = OpDef::elementwise("u", 1, &[dt], input(0).unary(op));
    try_generate(&def, &unary_key(dt), &SpirV)
}

fn gen_binary(op: BinaryOp, dt: ElementKind) -> Result<GeneratedKernel, LowerError> {
    let def = OpDef::elementwise("b", 2, &[dt], input(0).binary(op, input(1)));
    try_generate(&def, &binary_key(dt), &SpirV)
}

const F32_UNARY: &[UnaryOp] = &[
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

/// The unary ops GLSL.std.450 defines at 64 bits (plus plain arithmetic).
const F64_UNARY: &[UnaryOp] = &[
    UnaryOp::Neg,
    UnaryOp::Abs,
    UnaryOp::Sqr,
    UnaryOp::Sqrt,
    UnaryOp::Recip,
    UnaryOp::Relu,
    UnaryOp::Floor,
    UnaryOp::Ceil,
    UnaryOp::Round,
    UnaryOp::Sign,
    UnaryOp::Step,
    UnaryOp::Trunc,
];

const F32_BINARY: &[BinaryOp] = &[
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

const F64_BINARY: &[BinaryOp] = &[
    BinaryOp::Max,
    BinaryOp::Min,
    BinaryOp::Rem,
    BinaryOp::RemTrunc,
    BinaryOp::FmaxIeee,
    BinaryOp::FminIeee,
    BinaryOp::CmpEq,
    BinaryOp::CmpNe,
    BinaryOp::CmpLt,
    BinaryOp::CmpLe,
    BinaryOp::CmpGt,
    BinaryOp::CmpGe,
];

#[test]
fn every_f32_unary_op_validates() {
    for &op in F32_UNARY {
        let k =
            gen_unary(op, ElementKind::F32).unwrap_or_else(|e| panic!("{op:?} declined: {e:?}"));
        assert_valid(&k);
    }
}

#[test]
fn every_f64_unary_op_in_the_64_bit_set_validates() {
    for &op in F64_UNARY {
        let k =
            gen_unary(op, ElementKind::F64).unwrap_or_else(|e| panic!("{op:?} declined: {e:?}"));
        assert_valid(&k);
    }
}

#[test]
fn the_transcendentals_are_declined_at_f64_and_spelled_at_f32() {
    for op in [UnaryOp::Exp, UnaryOp::Sin, UnaryOp::Tanh, UnaryOp::Log2] {
        let err = gen_unary(op, ElementKind::F64).expect_err("not defined at 64 bits");
        assert!(
            format!("{err:?}").contains("Unary"),
            "decline must name the op: {err:?}"
        );
        gen_unary(op, ElementKind::F32).expect("positive control: the same op at f32 generates");
    }
}

#[test]
fn every_f32_and_f64_binary_op_validates() {
    for &op in F32_BINARY {
        let k =
            gen_binary(op, ElementKind::F32).unwrap_or_else(|e| panic!("{op:?} declined: {e:?}"));
        assert_valid(&k);
    }
    for &op in F64_BINARY {
        let k =
            gen_binary(op, ElementKind::F64).unwrap_or_else(|e| panic!("{op:?} declined: {e:?}"));
        assert_valid(&k);
    }
    for op in [BinaryOp::Pow, BinaryOp::Atan2] {
        gen_binary(op, ElementKind::F64).expect_err("pow/atan2 are not defined at 64 bits");
    }
}

#[test]
fn integer_bit_ops_validate_and_shift_right_follows_signedness() {
    for dt in [ElementKind::I32, ElementKind::U32] {
        for op in [
            BinaryOp::BitAnd,
            BinaryOp::BitOr,
            BinaryOp::BitXor,
            BinaryOp::Shl,
            BinaryOp::Shr,
        ] {
            let k = gen_binary(op, dt).unwrap_or_else(|e| panic!("{op:?}/{dt:?} declined: {e:?}"));
            assert_valid(&k);
            if op == BinaryOp::Shr {
                let want = if dt == ElementKind::I32 {
                    "OpShiftRightArithmetic"
                } else {
                    "OpShiftRightLogical"
                };
                assert!(
                    k.source.contains(want),
                    "{dt:?} Shr must be {want}:\n{}",
                    k.source
                );
            }
        }
    }
    // Integer add/sub/mul go through the arith seam.
    let def = OpDef::elementwise(
        "iarith",
        2,
        &[ElementKind::I32],
        input(0) * input(1) + input(0) - input(1),
    );
    assert_valid(&generate(&def, &binary_key(ElementKind::I32), &SpirV));
}

#[test]
fn arithmetic_select_and_non_finite_constants_validate() {
    let def = OpDef::elementwise(
        "affine",
        2,
        &[ElementKind::F32],
        input(0) * input(1) + input(0) / input(1) - input(1),
    );
    assert_valid(&generate(&def, &binary_key(ElementKind::F32), &SpirV));

    let nonfinite = OpDef::elementwise(
        "nonfinite",
        1,
        &[ElementKind::F32],
        (input(0) + konst(f64::INFINITY)) + (konst(f64::NEG_INFINITY) + konst(f64::NAN)),
    );
    let k = generate(&nonfinite, &unary_key(ElementKind::F32), &SpirV);
    assert_valid(&k);
    // The three non-finite values keep their IEEE bit patterns as constant words.
    let words = k.words().unwrap();
    for (what, bits) in [
        ("NaN", 0x7fc0_0000u32),
        ("+inf", 0x7f80_0000),
        ("-inf", 0xff80_0000),
    ] {
        assert!(
            words.contains(&bits),
            "{what} constant {bits:#x} missing:\n{}",
            k.source
        );
    }
    // Control for the lookup: a kernel with no NaN constant does not contain the NaN word.
    let plain = gen_unary(UnaryOp::Neg, ElementKind::F32).unwrap();
    assert!(!plain.words().unwrap().contains(&0x7fc0_0000));

    let sel = OpDef::elementwise(
        "sel",
        3,
        &[ElementKind::F32],
        input(0).select(input(1), input(2)),
    );
    assert_valid(&generate(
        &sel,
        &key(OpCategory::BinaryElementwise, 4, ElementKind::F32),
        &SpirV,
    ));
}

#[test]
fn a_shared_subexpression_is_emitted_once() {
    // (x*x) is used twice; lower_dag hoists it and the speller runs once for it.
    let def = OpDef::elementwise(
        "shared",
        1,
        &[ElementKind::F32],
        (input(0) * input(0)) + (input(0) * input(0)).unary(UnaryOp::Sqrt),
    );
    let k = generate(&def, &unary_key(ElementKind::F32), &SpirV);
    assert_valid(&k);
    assert_eq!(
        k.source.matches("OpFMul").count(),
        1,
        "one multiply for the shared node:\n{}",
        k.source
    );
}

/// The module is what the docs say: entry point named like the kernel, `LocalSize 256 1 1`,
/// `input0..` at bindings `0..n-1` and `output` at `n`, all `StorageBuffer` in set 0.
#[test]
fn module_shape_reflects_as_documented() {
    let def = OpDef::elementwise(
        "affine",
        3,
        &[ElementKind::F32],
        input(0) * input(1) + input(2),
    );
    let k = generate(
        &def,
        &key(OpCategory::BinaryElementwise, 4, ElementKind::F32),
        &SpirV,
    );
    assert_eq!(k.name, "unpopped_spirv_affine_f32");
    let m = vulkane_spirv_parse::Module::parse(k.words().unwrap()).expect("parses");
    let eps = m.entry_points();
    assert_eq!(eps.len(), 1);
    assert_eq!(eps[0].name, k.name);
    assert_eq!(eps[0].local_size, Some([WORKGROUP_SIZE, 1, 1]));
    let mut bs: Vec<_> = m
        .bindings()
        .into_iter()
        .filter(|b| b.storage_class == rspirv::spirv::StorageClass::StorageBuffer)
        .map(|b| (b.set, b.binding))
        .collect();
    bs.sort();
    assert_eq!(
        bs,
        vec![(Some(0), 0), (Some(0), 1), (Some(0), 2), (Some(0), 3)]
    );
    // Re-encoding the parsed module reproduces the emitted words.
    assert_eq!(m.to_words(), k.words().unwrap());
}

#[test]
fn float_kernels_declare_denorm_preserve_and_integer_kernels_do_not() {
    let f = gen_unary(UnaryOp::Neg, ElementKind::F32).unwrap();
    assert!(f.source.contains("DenormPreserve 32"), "{}", f.source);
    let d = gen_unary(UnaryOp::Neg, ElementKind::F64).unwrap();
    assert!(d.source.contains("DenormPreserve 64"), "{}", d.source);
    let i = gen_binary(BinaryOp::BitAnd, ElementKind::I32).unwrap();
    assert!(!i.source.contains("DenormPreserve"), "{}", i.source);
}

#[test]
fn words_are_part_of_the_kernel_identity() {
    let a = gen_unary(UnaryOp::Neg, ElementKind::F32).unwrap();
    let b = gen_unary(UnaryOp::Neg, ElementKind::F32).unwrap();
    assert_eq!(a.words(), b.words(), "emission is deterministic");
    assert_eq!(a.revision_hash(), b.revision_hash());
    let c = gen_unary(UnaryOp::Abs, ElementKind::F32).unwrap();
    assert_ne!(a.revision_hash(), c.revision_hash());
    assert_eq!(a.words().unwrap()[0], 0x0723_0203, "SPIR-V magic number");
}

/// **Negative control.** A validator that accepts everything would pass every test above; this
/// shows `spirv-val` rejects a damaged copy of a module it accepts.
#[test]
fn the_validator_rejects_a_corrupted_module() {
    let k = gen_unary(UnaryOp::Neg, ElementKind::F32).unwrap();
    let good = k.words().unwrap().to_vec();
    let Some((ok, _)) = validate(&good) else {
        return;
    };
    assert!(ok, "control module must validate");
    // Truncate: the module ends inside an instruction.
    let truncated = &good[..good.len() - 2];
    let (ok, _) = validate(truncated).expect("validator ran once already");
    assert!(!ok, "a truncated module must be rejected");
    // Wrong magic number.
    let mut bad = good.clone();
    bad[0] = 0xdead_beef;
    let (ok, _) = validate(&bad).expect("validator ran once already");
    assert!(!ok, "a module with a bad magic number must be rejected");
}

#[test]
fn declines_match_the_slang_emitters_set() {
    for op in [
        UnaryOp::Erf,
        UnaryOp::Erfc,
        UnaryOp::Gelu,
        UnaryOp::Lgamma,
        UnaryOp::Cbrt,
        UnaryOp::Asinh,
        UnaryOp::Acosh,
        UnaryOp::Atanh,
    ] {
        gen_unary(op, ElementKind::F32).expect_err("polyfill ops are declined");
    }
    for op in [BinaryOp::Copysign, BinaryOp::Nextafter] {
        gen_binary(op, ElementKind::F32).expect_err("declined by design");
    }
    let key3 = key(OpCategory::BinaryElementwise, 4, ElementKind::F32);
    let fused = OpDef::elementwise(
        "fma",
        3,
        &[ElementKind::F32],
        input(0).fma(input(1), input(2)),
    );
    let err = try_generate(&fused, &key3, &SpirV).expect_err("Fma is declined");
    assert!(format!("{err:?}").contains("Fma"), "{err:?}");
    let unfused = OpDef::elementwise(
        "mulAdd",
        3,
        &[ElementKind::F32],
        input(0) * input(1) + input(2),
    );
    try_generate(&unfused, &key3, &SpirV).expect("positive control: the unfused form generates");
}

#[test]
fn unsupported_dtypes_and_schedules_are_typed_declines() {
    let sku: TargetId = ArchSku::Sm89.into();
    for dt in [
        ElementKind::F16,
        ElementKind::Bf16,
        ElementKind::I8,
        ElementKind::U8,
        ElementKind::I64,
    ] {
        assert!(!SpirV.supports_dtype(dt, sku).is_supported(), "{dt:?}");
    }
    for dt in [
        ElementKind::F32,
        ElementKind::F32Strict,
        ElementKind::F64,
        ElementKind::I32,
        ElementKind::U32,
    ] {
        assert!(SpirV.supports_dtype(dt, sku).is_supported(), "{dt:?}");
    }
    // A V4-aligned contiguous binary f32 op keys `Schedule::Vectorized`.
    let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
    let k = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
    let err = try_generate(&op, &k, &SpirV).expect_err("only the scalar schedule is served");
    assert!(
        matches!(err, LowerError::UnsupportedSchedule { .. }),
        "{err:?}"
    );
}
