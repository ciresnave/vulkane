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
    temps: std::collections::HashMap<String, Word>,
}

fn sp(w: Word) -> Spelling {
    Spelling::Spelled(w.to_string())
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

    /// The unary ops GLSL.std.450 defines at 64 bits, plus the arithmetic ones.
    fn unary_wide(&mut self, op: UnaryOp, x: Word) -> Result<Option<Word>, LowerError> {
        let t = self.t;
        let r = match op {
            UnaryOp::Neg => lo(self.b.f_negate(t, None, x))?,
            UnaryOp::Abs => self.gl(GLOp::FAbs, &[x])?,
            UnaryOp::Sqr => lo(self.b.f_mul(t, None, x, x))?,
            UnaryOp::Sqrt => self.gl(GLOp::Sqrt, &[x])?,
            UnaryOp::Recip => {
                let one = self.fconst(1.0);
                lo(self.b.f_div(t, None, one, x))?
            }
            // NaN-propagating: `x < 0` is false for NaN, so NaN passes through.
            UnaryOp::Relu => {
                let zero = self.fconst(0.0);
                let lt = lo(self.b.f_ord_less_than(self.boolean, None, x, zero))?;
                lo(self.b.select(t, None, lt, zero, x))?
            }
            UnaryOp::Floor => self.gl(GLOp::Floor, &[x])?,
            UnaryOp::Ceil => self.gl(GLOp::Ceil, &[x])?,
            // Round half to even, matching `rintf`.
            UnaryOp::Round => self.gl(GLOp::RoundEven, &[x])?,
            UnaryOp::Trunc => self.gl(GLOp::Trunc, &[x])?,
            UnaryOp::Sign => {
                let (zero, one, mone) = (self.fconst(0.0), self.fconst(1.0), self.fconst(-1.0));
                let gt = lo(self.b.f_ord_greater_than(self.boolean, None, x, zero))?;
                let lt = lo(self.b.f_ord_less_than(self.boolean, None, x, zero))?;
                let neg = lo(self.b.select(t, None, lt, mone, zero))?;
                lo(self.b.select(t, None, gt, one, neg))?
            }
            UnaryOp::Step => {
                let (zero, one) = (self.fconst(0.0), self.fconst(1.0));
                let gt = lo(self.b.f_ord_greater_than(self.boolean, None, x, zero))?;
                lo(self.b.select(t, None, gt, one, zero))?
            }
            _ => return Ok(None),
        };
        Ok(Some(r))
    }

    /// The transcendental ops: 32-bit only (GLSL.std.450 does not define them at 64 bits).
    fn unary_narrow(&mut self, op: UnaryOp, x: Word) -> Result<Option<Word>, LowerError> {
        let t = self.t;
        let r = match op {
            UnaryOp::Rsqrt => self.gl(GLOp::InverseSqrt, &[x])?,
            UnaryOp::Exp => self.gl(GLOp::Exp, &[x])?,
            UnaryOp::Log => self.gl(GLOp::Log, &[x])?,
            UnaryOp::Tanh => self.gl(GLOp::Tanh, &[x])?,
            UnaryOp::Sigmoid => self.sigmoid(x)?,
            UnaryOp::Silu => {
                let s = self.sigmoid(x)?;
                lo(self.b.f_mul(t, None, x, s))?
            }
            UnaryOp::Sin => self.gl(GLOp::Sin, &[x])?,
            UnaryOp::Cos => self.gl(GLOp::Cos, &[x])?,
            UnaryOp::Exp2 => self.gl(GLOp::Exp2, &[x])?,
            UnaryOp::Log2 => self.gl(GLOp::Log2, &[x])?,
            UnaryOp::Expm1 => {
                let e = self.gl(GLOp::Exp, &[x])?;
                let one = self.fconst(1.0);
                lo(self.b.f_sub(t, None, e, one))?
            }
            // 1/ln(10)
            UnaryOp::Log10 => {
                let l = self.gl(GLOp::Log, &[x])?;
                let k = self.fconst(std::f64::consts::LOG10_E);
                lo(self.b.f_mul(t, None, l, k))?
            }
            UnaryOp::Log1p => {
                let one = self.fconst(1.0);
                let s = lo(self.b.f_add(t, None, one, x))?;
                self.gl(GLOp::Log, &[s])?
            }
            UnaryOp::Sinh => self.gl(GLOp::Sinh, &[x])?,
            UnaryOp::Cosh => self.gl(GLOp::Cosh, &[x])?,
            UnaryOp::Tan => self.gl(GLOp::Tan, &[x])?,
            UnaryOp::Asin => self.gl(GLOp::Asin, &[x])?,
            UnaryOp::Acos => self.gl(GLOp::Acos, &[x])?,
            UnaryOp::Atan => self.gl(GLOp::Atan, &[x])?,
            _ => return Ok(None),
        };
        Ok(Some(r))
    }

    fn unary(&mut self, op: UnaryOp, x: Word) -> Result<Spelling, LowerError> {
        if !self.fam.is_float() {
            return Ok(decline_unary(
                op,
                self.dtype,
                "v1 unary is float/double; integer dtypes have no unary math",
            ));
        }
        if matches!(
            op,
            UnaryOp::Erf
                | UnaryOp::Erfc
                | UnaryOp::Gelu
                | UnaryOp::Lgamma
                | UnaryOp::Cbrt
                | UnaryOp::Asinh
                | UnaryOp::Acosh
                | UnaryOp::Atanh
        ) {
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

    fn binary_fp(&mut self, op: BinaryOp, a: Word, b: Word) -> Result<Spelling, LowerError> {
        let (t, bl) = (self.t, self.boolean);
        let r = match op {
            // NaN-propagating, A on ties (`>=`/`<=`): the KISS-Ops `max_prop`/`min_prop`
            // decomposition, in the order the Slang speller uses.
            BinaryOp::Max | BinaryOp::Min => {
                let na = lo(self.b.is_nan(bl, None, a))?;
                let nb = lo(self.b.is_nan(bl, None, b))?;
                let cmp = if op == BinaryOp::Max {
                    lo(self.b.f_ord_greater_than_equal(bl, None, a, b))?
                } else {
                    lo(self.b.f_ord_less_than_equal(bl, None, a, b))?
                };
                let pick = lo(self.b.select(t, None, cmp, a, b))?;
                let with_b = lo(self.b.select(t, None, nb, b, pick))?;
                lo(self.b.select(t, None, na, a, with_b))?
            }
            BinaryOp::FmaxIeee => self.gl(GLOp::NMax, &[a, b])?,
            BinaryOp::FminIeee => self.gl(GLOp::NMin, &[a, b])?,
            // Floored remainder (torch.remainder, sign-of-divisor), not fmod.
            BinaryOp::Rem => {
                let q = lo(self.b.f_div(t, None, a, b))?;
                let f = self.gl(GLOp::Floor, &[q])?;
                let m = lo(self.b.f_mul(t, None, f, b))?;
                lo(self.b.f_sub(t, None, a, m))?
            }
            BinaryOp::RemTrunc => lo(self.b.f_rem(t, None, a, b))?,
            BinaryOp::CmpEq => {
                let c = lo(self.b.f_ord_equal(bl, None, a, b))?;
                self.flag(c)?
            }
            BinaryOp::CmpNe => {
                let c = lo(self.b.f_unord_not_equal(bl, None, a, b))?;
                self.flag(c)?
            }
            BinaryOp::CmpLt => {
                let c = lo(self.b.f_ord_less_than(bl, None, a, b))?;
                self.flag(c)?
            }
            BinaryOp::CmpLe => {
                let c = lo(self.b.f_ord_less_than_equal(bl, None, a, b))?;
                self.flag(c)?
            }
            BinaryOp::CmpGt => {
                let c = lo(self.b.f_ord_greater_than(bl, None, a, b))?;
                self.flag(c)?
            }
            BinaryOp::CmpGe => {
                let c = lo(self.b.f_ord_greater_than_equal(bl, None, a, b))?;
                self.flag(c)?
            }
            BinaryOp::Pow | BinaryOp::Atan2 if self.fam == Family::F32 => {
                let g = if op == BinaryOp::Pow {
                    GLOp::Pow
                } else {
                    GLOp::Atan2
                };
                self.gl(g, &[a, b])?
            }
            BinaryOp::Pow | BinaryOp::Atan2 => {
                return Ok(decline_binary(
                    op,
                    self.dtype,
                    "GLSL.std.450 does not define this function at 64 bits",
                ));
            }
            BinaryOp::Copysign | BinaryOp::Nextafter => {
                return Ok(Spelling::Declined(Decline::UnsupportedOp {
                    op: DeclinedOp::Binary(op),
                    why: "declined by design with the Slang emitter's set".to_string(),
                }));
            }
            _ => {
                return Ok(decline_binary(
                    op,
                    self.dtype,
                    "no float lowering for this op",
                ));
            }
        };
        Ok(sp(r))
    }

    fn binary_int(&mut self, op: BinaryOp, a: Word, b: Word) -> Result<Spelling, LowerError> {
        let t = self.t;
        let r = match op {
            BinaryOp::BitAnd => lo(self.b.bitwise_and(t, None, a, b))?,
            BinaryOp::BitOr => lo(self.b.bitwise_or(t, None, a, b))?,
            BinaryOp::BitXor => lo(self.b.bitwise_xor(t, None, a, b))?,
            BinaryOp::Shl => lo(self.b.shift_left_logical(t, None, a, b))?,
            BinaryOp::Shr if self.fam == Family::I32 => {
                lo(self.b.shift_right_arithmetic(t, None, a, b))?
            }
            BinaryOp::Shr => lo(self.b.shift_right_logical(t, None, a, b))?,
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

    fn arith(&mut self, op: ArithOp, a: Word, b: Word) -> Result<Spelling, LowerError> {
        let t = self.t;
        let r = if self.fam.is_float() {
            match op {
                ArithOp::Add => lo(self.b.f_add(t, None, a, b))?,
                ArithOp::Sub => lo(self.b.f_sub(t, None, a, b))?,
                ArithOp::Mul => lo(self.b.f_mul(t, None, a, b))?,
                ArithOp::Div => lo(self.b.f_div(t, None, a, b))?,
                _ => return Ok(self.arith_decline(op)),
            }
        } else {
            match op {
                ArithOp::Add => lo(self.b.i_add(t, None, a, b))?,
                ArithOp::Sub => lo(self.b.i_sub(t, None, a, b))?,
                ArithOp::Mul => lo(self.b.i_mul(t, None, a, b))?,
                // Integer division is device-UB at zero; the plan gate rejects it upstream.
                _ => return Ok(self.arith_decline(op)),
            }
        };
        Ok(sp(r))
    }

    fn arith_decline(&self, op: ArithOp) -> Spelling {
        Spelling::Declined(Decline::UnsupportedDtypeForOp {
            op: DeclinedOp::Arith(op),
            dtype: self.dtype,
            why: "no SPIR-V spelling for this arithmetic op".to_string(),
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
}

/// A float or integer constant as a SPIR-V constant id. NaN and the infinities keep their
/// IEEE bit patterns, so no C macro spelling is involved.
fn const_spelling(em: &mut Em, v: f64) -> Spelling {
    sp(em.fconst(v))
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

/// Emit the module for the scalar contiguous Elementwise path.
fn emit_scalar(plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
    let fam = Family::of(plan.dtype).expect("dtype_ok implies a family");
    let name = format!("unpopped_spirv_{}_{}", plan.op_name, dtype_tag(plan.dtype));
    let n_in = usize::from(plan.n_inputs);

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
    let v3u = b.type_vector(u32_t, 3);
    let gid_ptr = b.type_pointer(None, StorageClass::Input, v3u);
    let gid = b.variable(gid_ptr, None, StorageClass::Input, None);
    b.decorate(
        gid,
        Decoration::BuiltIn,
        [Operand::BuiltIn(BuiltIn::GlobalInvocationId)],
    );

    // One storage-buffer block per operand: `struct { T data[]; }`.
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

    let main = lo(b.begin_function(void, None, FunctionControl::NONE, fn_ty))?;
    lo(b.begin_block(None))?;
    let gvec = lo(b.load(v3u, None, gid, None, []))?;
    let index = lo(b.composite_extract(u32_t, None, gvec, [0]))?;

    let em = RefCell::new(Em {
        b,
        fam,
        dtype: plan.dtype,
        glsl,
        t,
        boolean,
        zero_u32,
        inputs: buffers,
        index,
        temps: std::collections::HashMap::new(),
    });

    let acc = |k: u8| em.borrow_mut().load_input(k);
    let unary = |op, x: String| {
        let mut e = em.borrow_mut();
        let x = e.resolve(&x);
        e.unary(op, x)
    };
    let binary = |op, a: String, b: String| {
        let mut e = em.borrow_mut();
        let (a, b) = (e.resolve(&a), e.resolve(&b));
        if e.fam.is_float() {
            e.binary_fp(op, a, b)
        } else {
            e.binary_int(op, a, b)
        }
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
    let constant = |v: f64| Ok(const_spelling(&mut em.borrow_mut(), v));
    // The declaration half of hoisting: record the name, emit no text (the value is an SSA id).
    let temp = |_ctype: &str, name: &str, rhs: &str| {
        let mut e = em.borrow_mut();
        let w = e.resolve(rhs);
        e.temps.insert(name.to_string(), w);
        String::new()
    };

    let (prelude, root) = lower_dag(
        &ExprDag::from_expr(plan.body),
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

    let mut e = em.into_inner();
    let root = e.resolve(&root);
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
