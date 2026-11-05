// SPDX-License-Identifier: MIT OR Apache-2.0
//! The cases the behaviour oracle runs, how closely a device must match the host for each, and
//! which input rows an op is specified on. Shared by both tests in `oracle_live.rs`.

use unpopped::ir::{BinaryOp, Expr, OpDef, UnaryOp, input};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key};

/// How closely the device must match the host for one op.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum Tol {
    /// Bit-identical (NaN equals NaN).
    Exact,
    /// Bit-identical except that `+0` and `-0` are the same: SPIR-V does not specify the sign of a
    /// zero result (`OpFRem`, `sign`, IEEE `maxNum`/`minNum` ties), and a host keeps its own.
    ZeroSign,
    /// `|got - want| <= rel * |want| + abs`, for functions whose accuracy Vulkan only bounds.
    Close { rel: f32, abs: f32 },
}

const CLOSE: Tol = Tol::Close {
    rel: 1e-5,
    abs: 1e-5,
};
/// Vulkan bounds `sin`/`cos` by an absolute error of 2^-11 on [-pi, pi] and says nothing tighter.
const TRIG: Tol = Tol::Close {
    rel: 1e-4,
    abs: 5e-4,
};

fn within(g: f32, w: f32, rel: f32, abs: f32) -> bool {
    if g.is_infinite() || w.is_infinite() {
        return g == w;
    }
    (g - w).abs() <= rel * w.abs() + abs
}

fn agree_f32(tol: Tol, g: f32, w: f32) -> bool {
    if g.is_nan() && w.is_nan() {
        return true;
    }
    match tol {
        Tol::Exact => false,
        Tol::ZeroSign => g == 0.0 && w == 0.0,
        Tol::Close { rel, abs } => within(g, w, rel, abs),
    }
}

/// Do the two result cells agree under `tol`? Non-float cells must be bit-identical.
pub(crate) fn agree(tol: Tol, dt: ElementKind, got: u32, want: u32) -> bool {
    got == want
        || (dt == ElementKind::F32 && agree_f32(tol, f32::from_bits(got), f32::from_bits(want)))
}

/// Which input rows the op is specified on. Outside it SPIR-V leaves the result undefined (or, for
/// a signed-zero input, device-dependent), so a disagreement there is reported, not failed.
#[derive(Copy, Clone, Debug)]
pub(crate) enum Domain {
    /// Every row.
    All,
    /// GLSL.std.450 `Pow`: undefined for `x < 0` and for `x == 0 && y <= 0`.
    Pow,
    /// GLSL.std.450 `Atan2`: undefined for `x == 0 && y == 0`; a `-0` `y` is also excluded, because
    /// whether a device keeps the sign of a zero input is not something SPIR-V promises.
    Atan2,
}

impl Domain {
    pub(crate) fn holds(self, row: &[u32]) -> bool {
        let f = |i: usize| f32::from_bits(row[i]);
        match self {
            Domain::All => true,
            Domain::Pow => f(0) > 0.0,
            Domain::Atan2 => !(f(0) == 0.0 && f(1) == 0.0) && row[0] != (-0.0f32).to_bits(),
        }
    }
}

/// One op at one dtype, with the input columns it is run on.
pub(crate) struct Case {
    pub(crate) domain: Domain,
    pub(crate) name: String,
    pub(crate) dt: ElementKind,
    pub(crate) tol: Tol,
    pub(crate) def: OpDef,
    pub(crate) key: StructureKey,
    pub(crate) inputs: Vec<Vec<u32>>,
}

fn key(n_operands: usize, cat: OpCategory, dt: ElementKind) -> StructureKey {
    // A small `align` defeats vectorization, so the plan is `Schedule::Scalar` for every backend.
    let a = OperandDesc::new(1, &[1 << 20], &[1], dt, 4);
    structure_key(cat, &vec![a; n_operands], ArchSku::Sm89)
}

const F32_VALUES: [f32; 13] = [
    -3.5, -2.5, -1.0, -0.5, -0.0, 0.0, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 10.0,
];
const INT_VALUES: [u32; 9] = [
    0,
    1,
    0xFFFF_FFFF,
    0x8000_0000,
    0x7FFF_FFFF,
    0x1234_5678,
    0xDEAD_BEEF,
    0x0000_FFFF,
    7,
];

fn f32_bits() -> Vec<u32> {
    F32_VALUES.iter().map(|v| v.to_bits()).collect()
}

/// All ordered pairs of `values` as two parallel columns.
fn pairs(values: &[u32]) -> Vec<Vec<u32>> {
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for &x in values {
        for &y in values {
            a.push(x);
            b.push(y);
        }
    }
    vec![a, b]
}

fn unary_tol(op: UnaryOp) -> Tol {
    use UnaryOp::*;
    match op {
        Neg | Abs | Floor | Ceil | Trunc | Round | Relu | Step => Tol::Exact,
        // `sign(+-0)` is 0, and which zero a device returns is not specified (Lavapipe: `-0`).
        Sign => Tol::ZeroSign,
        // Vulkan gives these no tight accuracy bound; Lavapipe's `asin`/`acos` are a few 1e-5 off.
        Sin | Cos | Tan | Asin | Acos => TRIG,
        _ => CLOSE,
    }
}

fn binary_tol(op: BinaryOp) -> Tol {
    use BinaryOp::*;
    match op {
        Max | Min | CmpEq | CmpNe | CmpLt | CmpLe | CmpGt | CmpGe | Rem => Tol::Exact,
        // IEEE `maxNum`/`minNum` of `+0` and `-0` may return either zero.
        RemTrunc | FmaxIeee | FminIeee => Tol::ZeroSign,
        _ => CLOSE,
    }
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

fn unary_cases() -> impl Iterator<Item = Case> {
    let f = ElementKind::F32;
    F32_UNARY.into_iter().map(move |op| {
        let name = format!("u_{op:?}_f32").to_lowercase();
        Case {
            domain: Domain::All,
            def: OpDef::elementwise(&name, 1, &[f], input(0).unary(op)),
            name,
            dt: f,
            tol: unary_tol(op),
            key: key(2, OpCategory::UnaryElementwise, f),
            inputs: vec![f32_bits()],
        }
    })
}

fn binary_cases() -> impl Iterator<Item = Case> {
    let f = ElementKind::F32;
    F32_BINARY.into_iter().map(move |op| {
        let name = format!("b_{op:?}_f32").to_lowercase();
        let domain = match op {
            BinaryOp::Pow => Domain::Pow,
            BinaryOp::Atan2 => Domain::Atan2,
            _ => Domain::All,
        };
        Case {
            domain,
            def: OpDef::elementwise(&name, 2, &[f], input(0).binary(op, input(1))),
            name,
            dt: f,
            tol: binary_tol(op),
            key: key(3, OpCategory::BinaryElementwise, f),
            inputs: pairs(&f32_bits()),
        }
    })
}

type Build = fn() -> Expr;

fn arith_cases() -> impl Iterator<Item = Case> {
    let f = ElementKind::F32;
    let arith: [(&str, Build); 4] = [
        ("add", || input(0) + input(1)),
        ("sub", || input(0) - input(1)),
        ("mul", || input(0) * input(1)),
        ("div", || input(0) / input(1)),
    ];
    arith.into_iter().map(move |(tag, build)| {
        let name = format!("a_{tag}_f32");
        Case {
            domain: Domain::All,
            def: OpDef::elementwise(&name, 2, &[f], build()),
            name,
            dt: f,
            tol: if tag == "div" { CLOSE } else { Tol::Exact },
            key: key(3, OpCategory::BinaryElementwise, f),
            inputs: pairs(&f32_bits()),
        }
    })
}

fn int_cases() -> impl Iterator<Item = Case> {
    [ElementKind::I32, ElementKind::U32]
        .into_iter()
        .flat_map(|dt| {
            let tag = if dt == ElementKind::I32 { "i32" } else { "u32" };
            let bit_ops = [
                BinaryOp::BitAnd,
                BinaryOp::BitOr,
                BinaryOp::BitXor,
                BinaryOp::Shl,
                BinaryOp::Shr,
            ];
            let mut v: Vec<Case> = bit_ops
                .into_iter()
                .map(|op| {
                    let name = format!("b_{op:?}_{tag}").to_lowercase();
                    int_case(
                        name.clone(),
                        dt,
                        OpDef::elementwise(&name, 2, &[dt], input(0).binary(op, input(1))),
                    )
                })
                .collect();
            let name = format!("a_mixed_{tag}");
            let body = input(0) * input(1) + input(0) - input(1);
            v.push(int_case(
                name.clone(),
                dt,
                OpDef::elementwise(&name, 2, &[dt], body),
            ));
            v
        })
}

fn int_case(name: String, dt: ElementKind, def: OpDef) -> Case {
    Case {
        domain: Domain::All,
        name,
        dt,
        tol: Tol::Exact,
        def,
        key: key(3, OpCategory::BinaryElementwise, dt),
        inputs: pairs(&INT_VALUES),
    }
}

/// Every case the oracle runs: f32 unary and binary ops, f32 arithmetic, i32/u32 integer ops.
pub(crate) fn cases() -> Vec<Case> {
    unary_cases()
        .chain(binary_cases())
        .chain(arith_cases())
        .chain(int_cases())
        .collect()
}
