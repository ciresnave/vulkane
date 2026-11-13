// SPDX-License-Identifier: MIT OR Apache-2.0
//! WGSL spellers: the per-op text over the neutral [`UnaryOp`] / [`BinaryOp`] enums.
//!
//! WGSL is not C-family, so none of the C spellers apply: there is no ternary (`select(f, t, cond)` takes the *false* arm
//! first), the float remainder `%` truncates, shift counts are `u32`, and a constant NaN or infinity is a shader-creation
//! error. Each speller is a table row, not a `match` arm, so adding an op is data.

use unpopped::backend::{Decline, DeclinedOp, LowerError, Spelling};
use unpopped::ir::{BinaryOp, UnaryOp};
use unpopped_vocab::ElementKind;

type Spelled = Result<Spelling, LowerError>;

/// `{x}` is the operand.
const UNARY_F32: &[(UnaryOp, &str)] = &[
    (UnaryOp::Neg, "(-{x})"),
    (UnaryOp::Abs, "abs({x})"),
    (UnaryOp::Sqr, "({x}*{x})"),
    (UnaryOp::Sqrt, "sqrt({x})"),
    (UnaryOp::Rsqrt, "inverseSqrt({x})"),
    (UnaryOp::Recip, "(1.0/{x})"),
    (UnaryOp::Exp, "exp({x})"),
    (UnaryOp::Log, "log({x})"),
    (UnaryOp::Tanh, "tanh({x})"),
    (UnaryOp::Sigmoid, "(1.0/(1.0+exp(-{x})))"),
    // NaN-propagating: `x < 0` is false for NaN, so NaN passes through (the CUDA and Slang spelling, as a select).
    (UnaryOp::Relu, "select({x}, 0.0, {x} < 0.0)"),
    (UnaryOp::Silu, "({x}*(1.0/(1.0+exp(-{x}))))"),
    (UnaryOp::Sin, "sin({x})"),
    (UnaryOp::Cos, "cos({x})"),
    (UnaryOp::Floor, "floor({x})"),
    (UnaryOp::Ceil, "ceil({x})"),
    // WGSL `round` is round-half-to-even, like `rintf`.
    (UnaryOp::Round, "round({x})"),
    (
        UnaryOp::Sign,
        "select(select(0.0, -1.0, {x} < 0.0), 1.0, {x} > 0.0)",
    ),
    (UnaryOp::Step, "select(0.0, 1.0, {x} > 0.0)"),
    (UnaryOp::Trunc, "trunc({x})"),
    (UnaryOp::Exp2, "exp2({x})"),
    (UnaryOp::Log2, "log2({x})"),
    (UnaryOp::Expm1, "(exp({x})-1.0)"),
    (UnaryOp::Log10, "(log({x})*0.4342944819032518)"),
    (UnaryOp::Log1p, "log(1.0+{x})"),
    (UnaryOp::Sinh, "sinh({x})"),
    (UnaryOp::Cosh, "cosh({x})"),
    (UnaryOp::Tan, "tan({x})"),
    (UnaryOp::Asin, "asin({x})"),
    (UnaryOp::Acos, "acos({x})"),
    (UnaryOp::Atan, "atan({x})"),
];

/// `{a}` and `{b}` are the operands. Max/Min are the NaN-propagating compare-selects, A on ties (`>=` / `<=`).
const BINARY_F32: &[(BinaryOp, &str)] = &[
    (
        BinaryOp::Max,
        "select(select(select({b}, {a}, {a} >= {b}), {b}, {b} != {b}), {a}, {a} != {a})",
    ),
    (
        BinaryOp::Min,
        "select(select(select({b}, {a}, {a} <= {b}), {b}, {b} != {b}), {a}, {a} != {a})",
    ),
    (BinaryOp::Pow, "pow({a}, {b})"),
    // Floored remainder (torch.remainder, sign of the divisor), not WGSL's truncating `%`.
    (BinaryOp::Rem, "({a} - floor({a} / {b}) * {b})"),
    (BinaryOp::RemTrunc, "({a} % {b})"),
    (BinaryOp::Atan2, "atan2({a}, {b})"),
    (BinaryOp::FmaxIeee, "max({a}, {b})"),
    (BinaryOp::FminIeee, "min({a}, {b})"),
    (BinaryOp::CmpEq, "select(0.0, 1.0, {a} == {b})"),
    (BinaryOp::CmpNe, "select(0.0, 1.0, {a} != {b})"),
    (BinaryOp::CmpLt, "select(0.0, 1.0, {a} < {b})"),
    (BinaryOp::CmpLe, "select(0.0, 1.0, {a} <= {b})"),
    (BinaryOp::CmpGt, "select(0.0, 1.0, {a} > {b})"),
    (BinaryOp::CmpGe, "select(0.0, 1.0, {a} >= {b})"),
];

/// The integer rows. The shift count is taken modulo the width (Unpopped integer rule 3); WGSL already masks it, and the
/// mask is written anyway so the text states the rule. A signed operand's count is converted to `u32` first, because WGSL
/// requires a `u32` count.
const BINARY_INT: &[(BinaryOp, &str)] = &[
    (BinaryOp::BitAnd, "({a} & {b})"),
    (BinaryOp::BitOr, "({a} | {b})"),
    (BinaryOp::BitXor, "({a} ^ {b})"),
];

fn lookup<K: PartialEq + Copy>(table: &[(K, &'static str)], key: K) -> Option<&'static str> {
    table.iter().find(|(k, _)| *k == key).map(|(_, t)| *t)
}

fn declined_unary(op: UnaryOp, why: &str) -> Spelled {
    Ok(Spelling::Declined(Decline::UnsupportedOp {
        op: DeclinedOp::Unary(op),
        why: why.to_string(),
    }))
}

fn declined_binary(op: BinaryOp, dtype: ElementKind, why: &str) -> Spelled {
    Ok(Spelling::Declined(Decline::UnsupportedDtypeForOp {
        op: DeclinedOp::Binary(op),
        dtype,
        why: why.to_string(),
    }))
}

/// Spell a unary op. Only the float dtypes WGSL has (`f32`) carry unary math; the rest of the elementwise surface is
/// declined with the op named.
pub(crate) fn unary(op: UnaryOp, x: String, dtype: ElementKind) -> Spelled {
    if !is_f32(dtype) {
        return Ok(Spelling::Declined(Decline::UnsupportedDtypeForOp {
            op: DeclinedOp::Unary(op),
            dtype,
            why: "unary math is f32 in the WGSL emitter; integer dtypes have none".to_string(),
        }));
    }
    match lookup(UNARY_F32, op) {
        Some(t) => Ok(Spelling::Spelled(t.replace("{x}", &x))),
        None => declined_unary(
            op,
            "no exact WGSL spelling; erf/erfc/gelu/lgamma/cbrt/asinh/acosh/atanh need polyfills",
        ),
    }
}

fn is_f32(dtype: ElementKind) -> bool {
    matches!(dtype, ElementKind::F32 | ElementKind::F32Strict)
}

fn shift(op: BinaryOp, a: &str, b: &str, dtype: ElementKind) -> String {
    let count = if dtype == ElementKind::I32 {
        format!("(u32({b}) & 31u)")
    } else {
        format!("({b} & 31u)")
    };
    let sym = if op == BinaryOp::Shl { "<<" } else { ">>" };
    format!("({a} {sym} {count})")
}

fn binary_int(op: BinaryOp, a: String, b: String, dtype: ElementKind) -> Spelled {
    if let Some(t) = lookup(BINARY_INT, op) {
        return Ok(Spelling::Spelled(t.replace("{a}", &a).replace("{b}", &b)));
    }
    if matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
        return Ok(Spelling::Spelled(shift(op, &a, &b, dtype)));
    }
    declined_binary(
        op,
        dtype,
        "no integer lowering for this op; the logical ops are the bespoke Bool surface",
    )
}

fn binary_f32(op: BinaryOp, a: String, b: String, dtype: ElementKind) -> Spelled {
    match lookup(BINARY_F32, op) {
        Some(t) => Ok(Spelling::Spelled(t.replace("{a}", &a).replace("{b}", &b))),
        None => declined_binary(
            op,
            dtype,
            "declined by design (copysign, nextafter), or integer-only with no float lowering",
        ),
    }
}

/// Spell a non-infix binary op for `dtype`.
pub(crate) fn binary(op: BinaryOp, a: String, b: String, dtype: ElementKind) -> Spelled {
    match dtype {
        ElementKind::F32 | ElementKind::F32Strict => binary_f32(op, a, b, dtype),
        ElementKind::I32 | ElementKind::U32 => binary_int(op, a, b, dtype),
        other => declined_binary(op, other, "no binary math at this dtype"),
    }
}

/// The select: the arms and the condition are pinned to f32, and `select`'s false arm comes first.
pub(crate) fn select(c: String, a: String, b: String, dtype: ElementKind) -> Spelled {
    if is_f32(dtype) {
        Ok(Spelling::Spelled(format!(
            "select(f32({b}), f32({a}), f32({c}) != 0.0)"
        )))
    } else {
        Ok(Spelling::Declined(Decline::UnsupportedDtypeForOp {
            op: DeclinedOp::Select,
            dtype,
            why: "select is f32-only in the WGSL emitter".to_string(),
        }))
    }
}

/// A float constant. A finite one is Rust's shortest round-trip text, which is a valid abstract-float literal; one outside
/// f32's range is a shader-creation error in WGSL, so it declines. NaN and the infinities are declined too: WGSL rejects a
/// constant expression that evaluates to one.
pub(crate) fn constant(v: f64) -> Spelled {
    if v.is_finite() && v.abs() <= f64::from(f32::MAX) {
        return Ok(Spelling::Spelled(format!("{v:?}")));
    }
    Ok(Spelling::Declined(Decline::UnsupportedOp {
        op: DeclinedOp::Constant,
        why: format!(
            "{v:?} is not a finite f32 constant; WGSL rejects constant NaN, infinity and out-of-range values"
        ),
    }))
}
