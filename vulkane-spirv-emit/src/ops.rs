// SPDX-License-Identifier: MIT OR Apache-2.0
//! The op spellers: each emits SPIR-V instructions into the builder and returns the result id.

use std::collections::HashMap;

use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{StorageClass, Word};
use unpopped::backend::{Decline, DeclinedOp, LowerError, Spelling};
use unpopped::ir::{ArithOp, BinaryOp, UnaryOp};
use unpopped_vocab::ElementKind;

/// GLSL.std.450 instruction numbers this backend uses, from `extinst.glsl.std.450.grammar.json`
/// (SPIR-V Headers, Vulkan SDK 1.4.350). `rspirv` 0.13 ships no enum for them.
#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub(crate) enum GLOp {
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

/// Which scalar family a dtype lowers to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Family {
    F32,
    F64,
    I32,
    U32,
}

impl Family {
    pub(crate) fn of(dt: ElementKind) -> Option<Family> {
        match dt {
            ElementKind::F32 | ElementKind::F32Strict => Some(Family::F32),
            ElementKind::F64 => Some(Family::F64),
            ElementKind::I32 => Some(Family::I32),
            ElementKind::U32 => Some(Family::U32),
            _ => None,
        }
    }
    pub(crate) fn is_float(self) -> bool {
        matches!(self, Family::F32 | Family::F64)
    }
    pub(crate) fn bytes(self) -> u32 {
        if self == Family::F64 { 8 } else { 4 }
    }
}

pub(crate) fn lo<T, E: std::fmt::Display>(r: Result<T, E>) -> Result<T, LowerError> {
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
pub(crate) struct Em {
    pub(crate) b: Builder,
    pub(crate) fam: Family,
    pub(crate) dtype: ElementKind,
    pub(crate) glsl: Word,
    pub(crate) t: Word,
    pub(crate) boolean: Word,
    pub(crate) zero_u32: Word,
    pub(crate) inputs: Vec<Word>,
    pub(crate) index: Word,
    /// `lower_dag` names a hoisted shared node `tmpN` and hands that name to its consumers; the
    /// `temp` seam records which result id each name stands for.
    pub(crate) temps: HashMap<String, Word>,
}

impl Em {
    /// The result id an operand spelling stands for: a decimal id, or a hoisted `tmpN`.
    pub(crate) fn resolve(&self, s: &str) -> Word {
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

    fn neg(&mut self, x: Word) -> Result<Word, LowerError> {
        lo(self.b.f_negate(self.t, None, x))
    }

    fn sqr(&mut self, x: Word) -> Result<Word, LowerError> {
        lo(self.b.f_mul(self.t, None, x, x))
    }

    fn recip(&mut self, x: Word) -> Result<Word, LowerError> {
        let one = self.fconst(1.0);
        lo(self.b.f_div(self.t, None, one, x))
    }

    fn sigmoid(&mut self, x: Word) -> Result<Word, LowerError> {
        let nx = self.neg(x)?;
        let e = self.gl(GLOp::Exp, &[nx])?;
        let one = self.fconst(1.0);
        let d = lo(self.b.f_add(self.t, None, one, e))?;
        lo(self.b.f_div(self.t, None, one, d))
    }

    fn silu(&mut self, x: Word) -> Result<Word, LowerError> {
        let s = self.sigmoid(x)?;
        lo(self.b.f_mul(self.t, None, x, s))
    }

    fn expm1(&mut self, x: Word) -> Result<Word, LowerError> {
        let e = self.gl(GLOp::Exp, &[x])?;
        let one = self.fconst(1.0);
        lo(self.b.f_sub(self.t, None, e, one))
    }

    /// `ln(x) * (1/ln(10))`.
    fn log10(&mut self, x: Word) -> Result<Word, LowerError> {
        let l = self.gl(GLOp::Log, &[x])?;
        let k = self.fconst(std::f64::consts::LOG10_E);
        lo(self.b.f_mul(self.t, None, l, k))
    }

    fn log1p(&mut self, x: Word) -> Result<Word, LowerError> {
        let one = self.fconst(1.0);
        let s = lo(self.b.f_add(self.t, None, one, x))?;
        self.gl(GLOp::Log, &[s])
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
        let r = match op {
            UnaryOp::Neg => self.neg(x),
            UnaryOp::Sqr => self.sqr(x),
            UnaryOp::Recip => self.recip(x),
            UnaryOp::Relu => self.relu(x),
            UnaryOp::Sign => self.sign(x),
            UnaryOp::Step => self.step(x),
            _ => return Ok(None),
        };
        r.map(Some)
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
        let r = match op {
            UnaryOp::Sigmoid => self.sigmoid(x),
            UnaryOp::Silu => self.silu(x),
            UnaryOp::Expm1 => self.expm1(x),
            UnaryOp::Log10 => self.log10(x),
            UnaryOp::Log1p => self.log1p(x),
            _ => return Ok(None),
        };
        r.map(Some)
    }

    /// The transcendental ops: 32-bit only (GLSL.std.450 does not define them at 64 bits).
    fn unary_narrow(&mut self, op: UnaryOp, x: Word) -> Result<Option<Word>, LowerError> {
        match lookup(GL_UNARY_NARROW, &op) {
            Some(g) => self.gl(g, &[x]).map(Some),
            None => self.unary_composite(op, x),
        }
    }

    pub(crate) fn unary(&mut self, op: UnaryOp, x: Word) -> Result<Spelling, LowerError> {
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

    /// The shift count as Unpopped defines it (`docs/int-div-shift-semantics.md`, rule 3): its
    /// bits read as unsigned, modulo the 32-bit width, i.e. `count & 31`. SPIR-V makes a count
    /// `>= 32` poison (3.3.14), so the mask is not optional; it also defines a negative count.
    fn shift_count(&mut self, count: Word) -> Result<Word, LowerError> {
        let mask = self.b.constant_bit32(self.t, 31);
        lo(self.b.bitwise_and(self.t, None, count, mask))
    }

    /// `<<` discards the bits shifted out and is defined for any `a`, signed or not.
    fn shift_left(&mut self, a: Word, b: Word) -> Result<Word, LowerError> {
        let count = self.shift_count(b)?;
        lo(self.b.shift_left_logical(self.t, None, a, count))
    }

    /// `>>` follows the signedness of the dtype: arithmetic on `i32`, logical on `u32`.
    fn shift_right(&mut self, a: Word, b: Word) -> Result<Word, LowerError> {
        let count = self.shift_count(b)?;
        let r = if self.fam == Family::I32 {
            self.b.shift_right_arithmetic(self.t, None, a, count)
        } else {
            self.b.shift_right_logical(self.t, None, a, count)
        };
        lo(r)
    }

    fn int_value(&mut self, op: BinaryOp, a: Word, b: Word) -> Result<Option<Word>, LowerError> {
        let t = self.t;
        let r = match op {
            BinaryOp::BitAnd => lo(self.b.bitwise_and(t, None, a, b)),
            BinaryOp::BitOr => lo(self.b.bitwise_or(t, None, a, b)),
            BinaryOp::BitXor => lo(self.b.bitwise_xor(t, None, a, b)),
            BinaryOp::Shl => self.shift_left(a, b),
            BinaryOp::Shr => self.shift_right(a, b),
            _ => return Ok(None),
        };
        r.map(Some)
    }

    fn binary_int(&mut self, op: BinaryOp, a: Word, b: Word) -> Result<Spelling, LowerError> {
        Ok(match self.int_value(op, a, b)? {
            Some(r) => sp(r),
            None => decline_binary(
                op,
                self.dtype,
                "no integer lowering for this op; the logical ops are the Bool surface",
            ),
        })
    }

    pub(crate) fn binary(
        &mut self,
        op: BinaryOp,
        a: Word,
        b: Word,
    ) -> Result<Spelling, LowerError> {
        if self.fam.is_float() {
            self.binary_fp(op, a, b)
        } else {
            self.binary_int(op, a, b)
        }
    }

    fn arith_float(&mut self, op: ArithOp, a: Word, b: Word) -> Result<Option<Word>, LowerError> {
        let t = self.t;
        let r = match op {
            ArithOp::Add => self.b.f_add(t, None, a, b),
            ArithOp::Sub => self.b.f_sub(t, None, a, b),
            ArithOp::Mul => self.b.f_mul(t, None, a, b),
            ArithOp::Div => self.b.f_div(t, None, a, b),
            _ => return Ok(None),
        };
        lo(r).map(Some)
    }

    /// Integer division is device-UB at zero; the plan gate rejects it upstream, and it is not
    /// spelled here.
    fn arith_int(&mut self, op: ArithOp, a: Word, b: Word) -> Result<Option<Word>, LowerError> {
        let t = self.t;
        let r = match op {
            ArithOp::Add => self.b.i_add(t, None, a, b),
            ArithOp::Sub => self.b.i_sub(t, None, a, b),
            ArithOp::Mul => self.b.i_mul(t, None, a, b),
            _ => return Ok(None),
        };
        lo(r).map(Some)
    }

    pub(crate) fn arith(&mut self, op: ArithOp, a: Word, b: Word) -> Result<Spelling, LowerError> {
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

    pub(crate) fn select(&mut self, c: Word, a: Word, b: Word) -> Result<Spelling, LowerError> {
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
    pub(crate) fn load_input(&mut self, k: u8) -> Result<Spelling, LowerError> {
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
    pub(crate) fn constant(&mut self, v: f64) -> Spelling {
        sp(self.fconst(v))
    }

    /// Store `value` to `output[i]` and close the function.
    pub(crate) fn store_and_return(&mut self, output: Word, value: Word) -> Result<(), LowerError> {
        let ptr_t = self
            .b
            .type_pointer(None, StorageClass::StorageBuffer, self.t);
        let dst = lo(self
            .b
            .access_chain(ptr_t, None, output, [self.zero_u32, self.index]))?;
        lo(self.b.store(dst, value, None, []))?;
        lo(self.b.ret())?;
        lo(self.b.end_function())
    }
}
