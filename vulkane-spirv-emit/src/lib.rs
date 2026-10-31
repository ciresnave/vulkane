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

use std::cell::RefCell;
use std::collections::HashMap;

use rspirv::binary::{Assemble, Disassemble};
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel,
    FunctionControl, MemoryModel, StorageClass, Word,
};
use unpopped::backend::{
    Backend, Decline, DeclinedOp, DtypeSupport, GeneratedKernel, LowerError, Lowering, Spelling,
    lower_dag,
};
use unpopped::cfamily::{assert_no_int_div_or_const, dtype_tag};
use unpopped::ir::{ArithOp, BinaryOp, ExprDag, ScalarExpr, UnaryOp};
use unpopped::plan::{KernelPlan, Schedule};
use unpopped_vocab::{ElementKind, TargetId};

/// GLSL.std.450 instruction numbers this backend uses, from `extinst.glsl.std.450.grammar.json`
/// (SPIR-V Headers, Vulkan SDK 1.4.350). `rspirv` 0.13 ships no enum for them.
#[derive(Copy, Clone, Debug)]
#[repr(u32)]
enum GLOp {
    RoundEven = 2,
    Trunc = 3,
    FAbs = 4,
    Floor = 8,
    Ceil = 9,
    Sin = 13,
    Cos = 14,
    Tan = 15,
    Asin = 16,
    Acos = 17,
    Atan = 18,
    Sinh = 19,
    Cosh = 20,
    Tanh = 21,
    Atan2 = 25,
    Pow = 26,
    Exp = 27,
    Log = 28,
    Exp2 = 29,
    Log2 = 30,
    Sqrt = 31,
    InverseSqrt = 32,
    NMin = 79,
    NMax = 80,
}

/// Unary ops that are one GLSL.std.450 call **and** defined at 64 bits.
const GL_UNARY_WIDE: &[(UnaryOp, GLOp)] = &[
    (UnaryOp::Abs, GLOp::FAbs),
    (UnaryOp::Sqrt, GLOp::Sqrt),
    (UnaryOp::Floor, GLOp::Floor),
    (UnaryOp::Ceil, GLOp::Ceil),
    // Round half to even, matching `rintf`.
    (UnaryOp::Round, GLOp::RoundEven),
    (UnaryOp::Trunc, GLOp::Trunc),
];

/// Unary ops that are one GLSL.std.450 call but are **not** defined at 64 bits.
const GL_UNARY_NARROW: &[(UnaryOp, GLOp)] = &[
    (UnaryOp::Rsqrt, GLOp::InverseSqrt),
    (UnaryOp::Exp, GLOp::Exp),
    (UnaryOp::Log, GLOp::Log),
    (UnaryOp::Tanh, GLOp::Tanh),
    (UnaryOp::Sin, GLOp::Sin),
    (UnaryOp::Cos, GLOp::Cos),
    (UnaryOp::Exp2, GLOp::Exp2),
    (UnaryOp::Log2, GLOp::Log2),
    (UnaryOp::Sinh, GLOp::Sinh),
    (UnaryOp::Cosh, GLOp::Cosh),
    (UnaryOp::Tan, GLOp::Tan),
    (UnaryOp::Asin, GLOp::Asin),
    (UnaryOp::Acos, GLOp::Acos),
    (UnaryOp::Atan, GLOp::Atan),
];

/// Binary ops that are one GLSL.std.450 call and defined at 64 bits: the NaN-suppressing min/max.
const GL_BINARY_WIDE: &[(BinaryOp, GLOp)] = &[
    (BinaryOp::FmaxIeee, GLOp::NMax),
    (BinaryOp::FminIeee, GLOp::NMin),
];

/// Binary ops that are one GLSL.std.450 call but not defined at 64 bits.
const GL_BINARY_NARROW: &[(BinaryOp, GLOp)] =
    &[(BinaryOp::Pow, GLOp::Pow), (BinaryOp::Atan2, GLOp::Atan2)];

/// Ops declined with the Slang emitter's set: no base-profile intrinsic, each needs a polyfill.
const POLYFILL_UNARY: &[UnaryOp] = &[
    UnaryOp::Erf,
    UnaryOp::Erfc,
    UnaryOp::Gelu,
    UnaryOp::Lgamma,
    UnaryOp::Cbrt,
    UnaryOp::Asinh,
    UnaryOp::Acosh,
    UnaryOp::Atanh,
];

fn lookup<K: PartialEq>(table: &[(K, GLOp)], key: &K) -> Option<GLOp> {
    table.iter().find(|(k, _)| k == key).map(|&(_, g)| g)
}

/// The SPIR-V compute-shader backend. Lowers a [`KernelPlan`] to a SPIR-V module.
#[derive(Copy, Clone, Debug, Default)]
pub struct SpirV;

/// The workgroup size every emitted kernel declares (`LocalSize 256 1 1`).
pub const WORKGROUP_SIZE: u32 = 256;

/// Which scalar family a dtype lowers to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Family {
    F32,
    F64,
    I32,
    U32,
}

impl Family {
    fn of(dt: ElementKind) -> Option<Family> {
        match dt {
            ElementKind::F32 | ElementKind::F32Strict => Some(Family::F32),
            ElementKind::F64 => Some(Family::F64),
            ElementKind::I32 => Some(Family::I32),
            ElementKind::U32 => Some(Family::U32),
            _ => None,
        }
    }
    fn is_float(self) -> bool {
        matches!(self, Family::F32 | Family::F64)
    }
    fn bytes(self) -> u32 {
        if self == Family::F64 { 8 } else { 4 }
    }
}

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

fn lo<T, E: std::fmt::Display>(r: Result<T, E>) -> Result<T, LowerError> {
    r.map_err(|e| LowerError::UnsupportedOp {
        detail: format!("spirv builder: {e}"),
    })
}

fn sp(w: Word) -> Spelling {
    Spelling::Spelled(w.to_string())
}

fn decline_unary(op: UnaryOp, dtype: ElementKind, why: &str) -> Spelling {
    Spelling::Declined(Decline::UnsupportedDtypeForOp {
        op: DeclinedOp::Unary(op),
        dtype,
        why: why.to_string(),
    })
}

fn decline_binary(op: BinaryOp, dtype: ElementKind, why: &str) -> Spelling {
    Spelling::Declined(Decline::UnsupportedDtypeForOp {
        op: DeclinedOp::Binary(op),
        dtype,
        why: why.to_string(),
    })
}

/// The module under construction plus the ids every speller needs.
struct Em {
    b: Builder,
    fam: Family,
    dtype: ElementKind,
    glsl: Word,
    t: Word,
    boolean: Word,
    zero_u32: Word,
    inputs: Vec<Word>,
    index: Word,
    /// `lower_dag` names a hoisted shared node `tmpN` and hands that name to its consumers; the
    /// `temp` seam records which result id each name stands for.
    temps: HashMap<String, Word>,
}

impl Em {
    /// The result id an operand spelling stands for: a decimal id, or a hoisted `tmpN`.
    fn resolve(&self, s: &str) -> Word {
        match self.temps.get(s) {
            Some(&w) => w,
            None => s
                .parse()
                .expect("operand spellings are SPIR-V result ids or recorded temporaries"),
        }
    }

    fn fconst(&mut self, v: f64) -> Word {
        match self.fam {
            Family::F64 => self.b.constant_bit64(self.t, v.to_bits()),
            _ => self.b.constant_bit32(self.t, (v as f32).to_bits()),
        }
    }

    fn gl(&mut self, op: GLOp, args: &[Word]) -> Result<Word, LowerError> {
        let ops: Vec<Operand> = args.iter().map(|&a| Operand::IdRef(a)).collect();
        lo(self.b.ext_inst(self.t, None, self.glsl, op as u32, ops))
    }

    /// `cond ? 1.0 : 0.0` in the compute dtype.
    fn flag(&mut self, cond: Word) -> Result<Word, LowerError> {
        let (one, zero) = (self.fconst(1.0), self.fconst(0.0));
        lo(self.b.select(self.t, None, cond, one, zero))
    }

    fn sigmoid(&mut self, x: Word) -> Result<Word, LowerError> {
        let nx = lo(self.b.f_negate(self.t, None, x))?;
        let e = self.gl(GLOp::Exp, &[nx])?;
        let one = self.fconst(1.0);
        let d = lo(self.b.f_add(self.t, None, one, e))?;
        lo(self.b.f_div(self.t, None, one, d))
    }

    /// `x > 0 ? 1 : 0`.
    fn step(&mut self, x: Word) -> Result<Word, LowerError> {
        let (zero, one) = (self.fconst(0.0), self.fconst(1.0));
        let gt = lo(self.b.f_ord_greater_than(self.boolean, None, x, zero))?;
        lo(self.b.select(self.t, None, gt, one, zero))
    }

    /// `x > 0 ? 1 : (x < 0 ? -1 : 0)`; NaN gives 0, as the Slang speller's compare chain does.
    fn sign(&mut self, x: Word) -> Result<Word, LowerError> {
        let (zero, one, mone) = (self.fconst(0.0), self.fconst(1.0), self.fconst(-1.0));
        let gt = lo(self.b.f_ord_greater_than(self.boolean, None, x, zero))?;
        let lt = lo(self.b.f_ord_less_than(self.boolean, None, x, zero))?;
        let neg = lo(self.b.select(self.t, None, lt, mone, zero))?;
        lo(self.b.select(self.t, None, gt, one, neg))
    }

    /// `x < 0 ? 0 : x`: NaN-propagating, since `x < 0` is false for NaN.
    fn relu(&mut self, x: Word) -> Result<Word, LowerError> {
        let zero = self.fconst(0.0);
        let lt = lo(self.b.f_ord_less_than(self.boolean, None, x, zero))?;
        lo(self.b.select(self.t, None, lt, zero, x))
    }

    /// The ops built from plain arithmetic and selects; valid at every float width.
    fn unary_arith(&mut self, op: UnaryOp, x: Word) -> Result<Option<Word>, LowerError> {
        let t = self.t;
        let r = match op {
            UnaryOp::Neg => lo(self.b.f_negate(t, None, x))?,
            UnaryOp::Sqr => lo(self.b.f_mul(t, None, x, x))?,
            UnaryOp::Recip => {
                let one = self.fconst(1.0);
                lo(self.b.f_div(t, None, one, x))?
            }
            UnaryOp::Relu => self.relu(x)?,
            UnaryOp::Sign => self.sign(x)?,
            UnaryOp::Step => self.step(x)?,
            _ => return Ok(None),
        };
        Ok(Some(r))
    }

    /// Everything defined at 64 bits: the one-call table plus the arithmetic ops.
    fn unary_wide(&mut self, op: UnaryOp, x: Word) -> Result<Option<Word>, LowerError> {
        match lookup(GL_UNARY_WIDE, &op) {
            Some(g) => self.gl(g, &[x]).map(Some),
            None => self.unary_arith(op, x),
        }
    }

    /// Transcendentals composed of other ops (32-bit only).
    fn unary_composite(&mut self, op: UnaryOp, x: Word) -> Result<Option<Word>, LowerError> {
        let (t, one) = (self.t, self.fconst(1.0));
        let r = match op {
            UnaryOp::Sigmoid => self.sigmoid(x)?,
            UnaryOp::Silu => {
                let s = self.sigmoid(x)?;
                lo(self.b.f_mul(t, None, x, s))?
            }
            UnaryOp::Expm1 => {
                let e = self.gl(GLOp::Exp, &[x])?;
                lo(self.b.f_sub(t, None, e, one))?
            }
            // 1/ln(10)
            UnaryOp::Log10 => {
                let l = self.gl(GLOp::Log, &[x])?;
                let k = self.fconst(std::f64::consts::LOG10_E);
                lo(self.b.f_mul(t, None, l, k))?
            }
            UnaryOp::Log1p => {
                let s = lo(self.b.f_add(t, None, one, x))?;
                self.gl(GLOp::Log, &[s])?
            }
            _ => return Ok(None),
        };
        Ok(Some(r))
    }

    /// The transcendental ops: 32-bit only (GLSL.std.450 does not define them at 64 bits).
    fn unary_narrow(&mut self, op: UnaryOp, x: Word) -> Result<Option<Word>, LowerError> {
        match lookup(GL_UNARY_NARROW, &op) {
            Some(g) => self.gl(g, &[x]).map(Some),
            None => self.unary_composite(op, x),
        }
    }

    fn unary(&mut self, op: UnaryOp, x: Word) -> Result<Spelling, LowerError> {
        if !self.fam.is_float() {
            return Ok(decline_unary(
                op,
                self.dtype,
                "v1 unary is float/double; integer dtypes have no unary math",
            ));
        }
        if POLYFILL_UNARY.contains(&op) {
            return Ok(Spelling::Declined(Decline::UnsupportedOp {
                op: DeclinedOp::Unary(op),
                why: "declined with the Slang emitter's set: erf/erfc/gelu/lgamma/cbrt/asinh/acosh/atanh need polyfills".to_string(),
            }));
        }
        if let Some(r) = self.unary_wide(op, x)? {
            return Ok(sp(r));
        }
        if self.fam == Family::F64 {
            return Ok(decline_unary(
                op,
                self.dtype,
                "GLSL.std.450 does not define this function at 64 bits",
            ));
        }
        match self.unary_narrow(op, x)? {
            Some(r) => Ok(sp(r)),
            None => Ok(decline_unary(op, self.dtype, "no SPIR-V spelling")),
        }
    }

    /// NaN-propagating max/min, A on ties (`>=`/`<=`): the KISS-Ops `max_prop`/`min_prop`
    /// decomposition, in the order the Slang speller uses.
    fn nan_minmax(&mut self, is_max: bool, a: Word, b: Word) -> Result<Word, LowerError> {
        let (t, bl) = (self.t, self.boolean);
        let na = lo(self.b.is_nan(bl, None, a))?;
        let nb = lo(self.b.is_nan(bl, None, b))?;
        let cmp = if is_max {
            lo(self.b.f_ord_greater_than_equal(bl, None, a, b))?
        } else {
            lo(self.b.f_ord_less_than_equal(bl, None, a, b))?
        };
        let pick = lo(self.b.select(t, None, cmp, a, b))?;
        let with_b = lo(self.b.select(t, None, nb, b, pick))?;
        lo(self.b.select(t, None, na, a, with_b))
    }

    /// Floored remainder (torch.remainder, sign-of-divisor), not fmod.
    fn rem_floored(&mut self, a: Word, b: Word) -> Result<Word, LowerError> {
        let t = self.t;
        let q = lo(self.b.f_div(t, None, a, b))?;
        let f = self.gl(GLOp::Floor, &[q])?;
        let m = lo(self.b.f_mul(t, None, f, b))?;
        lo(self.b.f_sub(t, None, a, m))
    }

    /// The six comparisons, as `1.0` / `0.0` in the compute dtype. `!=` is unordered (NaN is
    /// unequal to everything), the rest ordered, matching C's operators.
    fn compare(&mut self, op: BinaryOp, a: Word, b: Word) -> Result<Option<Word>, LowerError> {
        let bl = self.boolean;
        let cond = match op {
            BinaryOp::CmpEq => self.b.f_ord_equal(bl, None, a, b),
            BinaryOp::CmpNe => self.b.f_unord_not_equal(bl, None, a, b),
            BinaryOp::CmpLt => self.b.f_ord_less_than(bl, None, a, b),
            BinaryOp::CmpLe => self.b.f_ord_less_than_equal(bl, None, a, b),
            BinaryOp::CmpGt => self.b.f_ord_greater_than(bl, None, a, b),
            BinaryOp::CmpGe => self.b.f_ord_greater_than_equal(bl, None, a, b),
            _ => return Ok(None),
        };
        self.flag(lo(cond)?).map(Some)
    }

    /// The value for a float binary op, or `None` when the op has no spelling at this width.
    fn binary_fp_value(
        &mut self,
        op: BinaryOp,
        a: Word,
        b: Word,
    ) -> Result<Option<Word>, LowerError> {
        if let Some(g) = lookup(GL_BINARY_WIDE, &op) {
            return self.gl(g, &[a, b]).map(Some);
        }
        if self.fam == Family::F32 {
            if let Some(g) = lookup(GL_BINARY_NARROW, &op) {
                return self.gl(g, &[a, b]).map(Some);
            }
        }
        match op {
            BinaryOp::Max | BinaryOp::Min => self.nan_minmax(op == BinaryOp::Max, a, b).map(Some),
            BinaryOp::Rem => self.rem_floored(a, b).map(Some),
            BinaryOp::RemTrunc => lo(self.b.f_rem(self.t, None, a, b)).map(Some),
            _ => self.compare(op, a, b),
        }
    }

    fn binary_fp(&mut self, op: BinaryOp, a: Word, b: Word) -> Result<Spelling, LowerError> {
        if let Some(r) = self.binary_fp_value(op, a, b)? {
            return Ok(sp(r));
        }
        Ok(match op {
            BinaryOp::Pow | BinaryOp::Atan2 => decline_binary(
                op,
                self.dtype,
                "GLSL.std.450 does not define this function at 64 bits",
            ),
            BinaryOp::Copysign | BinaryOp::Nextafter => {
                Spelling::Declined(Decline::UnsupportedOp {
                    op: DeclinedOp::Binary(op),
                    why: "declined by design with the Slang emitter's set".to_string(),
                })
            }
            _ => decline_binary(op, self.dtype, "no float lowering for this op"),
        })
    }

    /// `>>` follows the signedness of the dtype.
    fn shift_right(&mut self, a: Word, b: Word) -> Result<Word, LowerError> {
        if self.fam == Family::I32 {
            lo(self.b.shift_right_arithmetic(self.t, None, a, b))
        } else {
            lo(self.b.shift_right_logical(self.t, None, a, b))
        }
    }

    fn binary_int(&mut self, op: BinaryOp, a: Word, b: Word) -> Result<Spelling, LowerError> {
        let t = self.t;
        let r = match op {
            BinaryOp::BitAnd => lo(self.b.bitwise_and(t, None, a, b))?,
            BinaryOp::BitOr => lo(self.b.bitwise_or(t, None, a, b))?,
            BinaryOp::BitXor => lo(self.b.bitwise_xor(t, None, a, b))?,
            BinaryOp::Shl => lo(self.b.shift_left_logical(t, None, a, b))?,
            BinaryOp::Shr => self.shift_right(a, b)?,
            _ => {
                return Ok(decline_binary(
                    op,
                    self.dtype,
                    "no integer lowering for this op; the logical ops are the Bool surface",
                ));
            }
        };
        Ok(sp(r))
    }

    fn binary(&mut self, op: BinaryOp, a: Word, b: Word) -> Result<Spelling, LowerError> {
        if self.fam.is_float() {
            self.binary_fp(op, a, b)
        } else {
            self.binary_int(op, a, b)
        }
    }

    fn arith_float(&mut self, op: ArithOp, a: Word, b: Word) -> Result<Option<Word>, LowerError> {
        let t = self.t;
        let r = match op {
            ArithOp::Add => lo(self.b.f_add(t, None, a, b))?,
            ArithOp::Sub => lo(self.b.f_sub(t, None, a, b))?,
            ArithOp::Mul => lo(self.b.f_mul(t, None, a, b))?,
            ArithOp::Div => lo(self.b.f_div(t, None, a, b))?,
            _ => return Ok(None),
        };
        Ok(Some(r))
    }

    /// Integer division is device-UB at zero; the plan gate rejects it upstream, and it is not
    /// spelled here.
    fn arith_int(&mut self, op: ArithOp, a: Word, b: Word) -> Result<Option<Word>, LowerError> {
        let t = self.t;
        let r = match op {
            ArithOp::Add => lo(self.b.i_add(t, None, a, b))?,
            ArithOp::Sub => lo(self.b.i_sub(t, None, a, b))?,
            ArithOp::Mul => lo(self.b.i_mul(t, None, a, b))?,
            _ => return Ok(None),
        };
        Ok(Some(r))
    }

    fn arith(&mut self, op: ArithOp, a: Word, b: Word) -> Result<Spelling, LowerError> {
        let r = if self.fam.is_float() {
            self.arith_float(op, a, b)?
        } else {
            self.arith_int(op, a, b)?
        };
        Ok(match r {
            Some(w) => sp(w),
            None => Spelling::Declined(Decline::UnsupportedDtypeForOp {
                op: DeclinedOp::Arith(op),
                dtype: self.dtype,
                why: "no SPIR-V spelling for this arithmetic op".to_string(),
            }),
        })
    }

    fn select(&mut self, c: Word, a: Word, b: Word) -> Result<Spelling, LowerError> {
        if !self.fam.is_float() {
            return Ok(Spelling::Declined(Decline::UnsupportedDtypeForOp {
                op: DeclinedOp::Select,
                dtype: self.dtype,
                why: "v1 select is float-only (f32/f64)".to_string(),
            }));
        }
        // `c != 0.0`, unordered: a NaN condition selects `a`, as the Slang speller's `!=` does.
        let zero = self.fconst(0.0);
        let cond = lo(self.b.f_unord_not_equal(self.boolean, None, c, zero))?;
        Ok(sp(lo(self.b.select(self.t, None, cond, a, b))?))
    }

    /// Load `input{k}[i]`.
    fn load_input(&mut self, k: u8) -> Result<Spelling, LowerError> {
        let var = self.inputs[usize::from(k)];
        let ptr_t = self
            .b
            .type_pointer(None, StorageClass::StorageBuffer, self.t);
        let ptr = lo(self
            .b
            .access_chain(ptr_t, None, var, [self.zero_u32, self.index]))?;
        Ok(sp(lo(self.b.load(self.t, None, ptr, None, []))?))
    }

    /// A constant as a SPIR-V constant id. NaN and the infinities keep their IEEE bit patterns,
    /// so no C macro spelling is involved.
    fn constant(&mut self, v: f64) -> Spelling {
        sp(self.fconst(v))
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
    if body_has_params(plan.body) {
        return Err(LowerError::UnsupportedPlanShape {
            detail: format!(
                "spirv backend v1: parameterless elementwise only; op '{}' reads a runtime scalar Param",
                plan.op_name
            ),
        });
    }
    Ok(())
}

/// Ids of the module-level declarations the function body refers to.
struct Skeleton {
    b: Builder,
    glsl: Word,
    void: Word,
    fn_ty: Word,
    t: Word,
    boolean: Word,
    u32_t: Word,
    vec3: Word,
    gid: Word,
    zero_u32: Word,
    inputs: Vec<Word>,
    output: Word,
}

/// Capabilities, memory model, types, the invocation-id input and the `n_in + 1` storage-buffer
/// blocks (`struct { T data[]; }`, set 0, bindings `0..=n_in`, the last being the output).
fn declare(fam: Family, n_in: usize) -> Skeleton {
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    if fam == Family::F64 {
        b.capability(Capability::Float64);
    }
    if fam.is_float() {
        b.capability(Capability::DenormPreserve);
        b.extension("SPV_KHR_float_controls");
    }
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
    let glsl = b.ext_inst_import("GLSL.std.450");

    let void = b.type_void();
    let fn_ty = b.type_function(void, []);
    let boolean = b.type_bool();
    let u32_t = b.type_int(32, 0);
    let t = match fam {
        Family::F32 => b.type_float(32, None),
        Family::F64 => b.type_float(64, None),
        Family::I32 => b.type_int(32, 1),
        Family::U32 => u32_t,
    };
    let vec3 = b.type_vector(u32_t, 3);
    let gid_ptr = b.type_pointer(None, StorageClass::Input, vec3);
    let gid = b.variable(gid_ptr, None, StorageClass::Input, None);
    b.decorate(
        gid,
        Decoration::BuiltIn,
        [Operand::BuiltIn(BuiltIn::GlobalInvocationId)],
    );

    let rt_array = b.type_runtime_array(t);
    b.decorate(
        rt_array,
        Decoration::ArrayStride,
        [Operand::LiteralBit32(fam.bytes())],
    );
    let block = b.type_struct([rt_array]);
    b.decorate(block, Decoration::Block, []);
    b.member_decorate(block, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
    let block_ptr = b.type_pointer(None, StorageClass::StorageBuffer, block);
    let mut buffers = Vec::with_capacity(n_in + 1);
    for binding in 0..=n_in {
        let v = b.variable(block_ptr, None, StorageClass::StorageBuffer, None);
        b.decorate(v, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        let slot = u32::try_from(binding).expect("binding fits u32");
        b.decorate(v, Decoration::Binding, [Operand::LiteralBit32(slot)]);
        buffers.push(v);
    }
    let output = buffers.pop().expect("output buffer");
    let zero_u32 = b.constant_bit32(u32_t, 0);
    Skeleton {
        b,
        glsl,
        void,
        fn_ty,
        t,
        boolean,
        u32_t,
        vec3,
        gid,
        zero_u32,
        inputs: buffers,
        output,
    }
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
    let ptr_t = e.b.type_pointer(None, StorageClass::StorageBuffer, e.t);
    let dst = lo(e.b.access_chain(ptr_t, None, output, [e.zero_u32, e.index]))?;
    lo(e.b.store(dst, root, None, []))?;
    lo(e.b.ret())?;
    lo(e.b.end_function())?;

    e.b.entry_point(ExecutionModel::GLCompute, main, name.clone(), [gid]);
    e.b.execution_mode(main, ExecutionMode::LocalSize, [WORKGROUP_SIZE, 1, 1]);
    if fam.is_float() {
        let width = if fam == Family::F64 { 64 } else { 32 };
        e.b.execution_mode(main, ExecutionMode::DenormPreserve, [width]);
    }
    let module = e.b.module();
    let source = module.disassemble();
    let words = module.assemble();
    Ok(GeneratedKernel::new(name, source).with_words(words))
}

/// Does `e` read a runtime scalar [`ScalarExpr::Param`]? (v1 is parameterless.)
fn body_has_params(e: &ScalarExpr) -> bool {
    match e {
        ScalarExpr::Param(_) => true,
        ScalarExpr::Unary(_, x) => body_has_params(x),
        ScalarExpr::Cast(x, _)
        | ScalarExpr::Bitcast(x, _)
        | ScalarExpr::Qualified(x, _)
        | ScalarExpr::Extract(x, _)
        | ScalarExpr::BitcastPacked(x, ..) => body_has_params(x),
        ScalarExpr::Pack(xs) => xs.iter().any(body_has_params),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => body_has_params(a) || body_has_params(b),
        ScalarExpr::Select(c, a, b) | ScalarExpr::Fma(c, a, b) => {
            body_has_params(c) || body_has_params(a) || body_has_params(b)
        }
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_)
        | ScalarExpr::CoordInt(..)
        | ScalarExpr::ConstInt(..)
        | ScalarExpr::Local(_)
        | ScalarExpr::Exec(_) => false,
    }
}

#[cfg(test)]
mod tests;
