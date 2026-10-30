// SPDX-License-Identifier: MIT OR Apache-2.0
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
//! matching `unpopped::convert::SLANG`'s convention — the `Frontend` that
//! `lift_elementwise` is given — so emitted
//! Slang re-lifts to the same IR (the residue-round-trip contract the "one IR, N
//! languages" hub rests on; proven by `slang_emit_round_trips_through_the_lifter`).
//!
//! ## Subnormals: a compile flag, not source
//!
//! Without a declared mode, SPIR-V leaves f32 subnormal behaviour to the device, and the devices
//! tried flush: the emitted `relu` (`x < 0.0 ? 0.0 : x`) returned a negative subnormal unchanged
//! where cpu-c and CUDA return 0 (8,388,607 of 2^32 inputs; AMD Radeon 610M and RTX 4070 Laptop,
//! exhaustive, `unpopped-harness`). The mode is a `slangc` argument, not something the `.slang`
//! text can carry, so it is the consumer's to pass: [`required_slangc_args`] names it, and every
//! float kernel's header comment repeats it. With `-denorm-mode-fp32 preserve` (SPIR-V
//! `DenormPreserve 32`) the same run had zero mismatches on both devices.
//!
//! **A device must not be gated on `VkPhysicalDeviceFloatControlsProperties` alone.** The RTX 4070
//! reports `shaderDenormPreserveFloat32 = false` and `shaderDenormFlushToZeroFloat32 = false`, yet
//! flushes by default and preserves when asked; AMD reports both true and honours both. What a
//! device does is a measurement, to be recorded per device (the harness report), not read off the
//! property bits. How a plan declines on it waits on a KISS capability field; until then an `Exact`
//! float kernel is only as exact as the flag the consumer passed.
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
//! ## A C-shaped seam assumption this backend surfaced, now closed
//!
//! [`unpopped::backend::const_lit`] spells non-finite constants as C's
//! `NAN`/`INFINITY` macros, which are not valid Slang: a body carrying a NaN/Inf
//! *constant* used to emit Slang that `slangc` rejected (`undefined identifier
//! 'INFINITY'`, found by the harness's C14 probe, 2026-10-24). The `constant` seam
//! now spells them with `slang_const` (`asfloat(0x7f800000u)` and friends, the
//! IEEE bit patterns), and finite constants still go through `const_lit` unchanged.

use unpopped::backend::{
    Backend, Decline, DeclinedOp, DtypeSupport, GeneratedKernel, LowerError, Lowering, Spelling,
    const_lit, lower_dag,
};
use unpopped::cfamily::{assert_no_int_div_or_const, dtype_tag};
use unpopped::ir::{BinaryOp, ExprDag, ScalarExpr, UnaryOp};
use unpopped::plan::{KernelPlan, Schedule};
use unpopped_vocab::{ElementKind, TargetId};

/// The Slang compute-shader backend. Lowers a [`KernelPlan`] to `.slang` source.
#[derive(Copy, Clone, Debug, Default)]
pub struct Slang;

/// A constant as Slang source. `const_lit` is valid Slang for every finite value but spells NaN
/// and the infinities as C macros (`NAN`, `INFINITY`) that Slang does not define, so those three
/// are spelled as their IEEE-754 binary32 bit patterns through `asfloat`. A kernel that widens the
/// result to f64 widens the float, which keeps NaN and the infinities.
fn slang_const(v: f64) -> String {
    if v.is_nan() {
        "asfloat(0x7fc00000u)".to_string()
    } else if v == f64::INFINITY {
        "asfloat(0x7f800000u)".to_string()
    } else if v == f64::NEG_INFINITY {
        "asfloat(0xff800000u)".to_string()
    } else {
        const_lit(v)
    }
}

/// Slang scalar type for a dtype, or `None` if v1 declines it.
fn slang_ctype(dt: ElementKind) -> Option<&'static str> {
    match dt {
        ElementKind::F32 | ElementKind::F32Strict => Some("float"),
        ElementKind::F64 => Some("double"),
        ElementKind::I32 => Some("int"),
        ElementKind::I64 => Some("int64_t"),
        // `uint`/`uint64_t`. The v1 comment here said U32 had "no clean
        // base-profile scalar type", which was simply wrong: Slang documents
        // `int`/`int32_t` and `uint`/`uint32_t` as the two integer types that are
        // *universally* supported, so this was never a capability question — it
        // was unwritten. The coverage table said so in as many words while this
        // line claimed the opposite, and the table was right.
        ElementKind::U32 => Some("uint"),
        ElementKind::U64 => Some("uint64_t"),
        // Narrow integers. Spellable ONLY where the target advertises 8-bit
        // arithmetic — `slang_ctype` answers "is there a type name", and
        // `supports_dtype` is what consults the target. Keeping the split means
        // this function stays a pure name table.
        ElementKind::I8 => Some("int8_t"),
        ElementKind::U8 => Some("uint8_t"),
        // `Bool` is stored as U8 (FKC §5 pins Bool->U8), so it spells the same
        // 8-bit type and gates the same way. `dtype_tag` still says "bool" — the
        // NAME and the STORAGE are different facts, exactly as CpuC has it
        // (tag "bool", ctype "unsigned char").
        ElementKind::Bool => Some("uint8_t"),
        // 16-bit integers. Unblocked 2026-09-02 by kiss-vulkan-vocab 0.4.0 /
        // vulkane 0.14.0 shipping capability vocabulary v5, which names `i16`
        // (`shaderInt16`). Before that the vocabulary could not express the
        // capability at all, so declining was the only honest answer — it was an
        // UNSHIPPED vocabulary, not a missing one.
        ElementKind::I16 => Some("int16_t"),
        ElementKind::U16 => Some("uint16_t"),
        // F16/Bf16 remain declined (the spelling seam). See `supports_dtype`.
        _ => None,
    }
}

/// Whether `dt` is spellable for `target`. **The one statement.**
///
/// [`Backend::supports_dtype`] and [`Backend::lower`] both call this, so they
/// cannot disagree. They did, for the length of one commit: adding `int8_t` to
/// `slang_ctype` made `lower` succeed on a `cuda:` target that advertises no
/// 8-bit arithmetic, because `lower` consulted only the name table and
/// `supports_dtype` was never on that path. **A capability gate that the
/// lowering path does not call is not a gate.**
fn dtype_ok(dt: ElementKind, target: TargetId) -> DtypeSupport {
    // `target` is now CONSULTED for the narrow integers — the gap this
    // comment used to describe is closed for i8/u8. What follows records why
    // the answer differs per width, and what is still unanswerable.
    //
    // Slang's own conformance docs: only `int`/`int32_t` and `uint`/`uint32_t`
    // are universally supported, and *"the others depend on target +
    // capabilities"*. So `i8`/`i16`/`u8`/`u16` are spellable on a capable
    // target, and declining them everywhere is over-refusal.
    //
    // # The reason is NOT "no capability data", and it differs per width
    //
    // This comment used to say the blocker was a missing machine-readable
    // manifest (KISS #171). Vulkane — who **owns** the `vulkan:` capability
    // vocabulary under §6.8-0004 — answered directly, and the truth is
    // asymmetric:
    //
    // * **`i8`/`u8`: a token already answers this.** The `<arith>` field
    //   carries `i8`, which names `shaderInt8` — 8-bit integer *arithmetic*.
    //   A target whose arith set contains `i8` does 8-bit integer math.
    //   Declining these is genuine over-refusal, answerable today with no
    //   manifest and no new API.
    // * **`i16`/`u16`: the vocabulary cannot express it.** The published
    //   `<arith>` names are exactly `dot8`, `f16`, `i8`, `st16`, `st8` —
    //   **`shaderInt16` is not among them**. No `vulkan:` token asserts
    //   16-bit integer arithmetic, so no consumer can derive it. Vulkane's
    //   phrasing, which is the right line for this code: *the vocabulary
    //   names `shaderInt8` and does not name `shaderInt16`; absence here is
    //   silence, not denial.* They record it as their gap, and naming it
    //   later bumps the vocabulary version rather than being additive.
    //
    // # NEVER infer 16-bit arithmetic from `st16`
    //
    // `st16` is `storageBuffer16BitAccess` — a **storage** capability. The
    // vulkan vocabulary §2.3 keeps compute precision and storage precision
    // as separate members precisely because they are separate: a device may
    // accept 16-bit data in a buffer and perform the arithmetic in `f32`.
    // Reading `st16` as permission to emit 16-bit integer math is a
    // **silently wrong lowering on conformant hardware**, and the token
    // would not be at fault. Verified absent from this crate today; written
    // down because it is exactly the inference a future reader would think
    // was an obvious win.
    //
    // # Three answers (issue #6)
    //
    // "The device lacks it" and "the vocabulary cannot say" call for different
    // responses (change hardware vs wait for a spec), which is Vulkane's point.
    // `i8`/`u8`/`bool`: the vocabulary HAS the token, so a target that does not
    // carry it is `Unsupported`. `i16`/`u16`: the vocabulary has no token, so a
    // target that does not carry one is `NotExpressible` — silence, not denial.
    if slang_ctype(dt).is_none() {
        return DtypeSupport::Unsupported;
    }
    // The narrow integers are the only dtypes whose answer depends on the
    // target. Everything else `slang_ctype` names is universally spellable.
    if !matches!(
        dt,
        ElementKind::I8 | ElementKind::U8 | ElementKind::Bool | ElementKind::I16 | ElementKind::U16
    ) {
        return DtypeSupport::Supported;
    }
    // 8-bit ARITHMETIC is `shaderInt8`, spelled `i8` in the `<arith>` field.
    //
    // ⚠️ `i8` is SIGNEDNESS-AGNOSTIC — it names the capability, not a
    // component type — so `u8` gates on it too. There is no `u8` token and
    // its absence is not an omission; signedness lives in the component-type
    // vocabulary (`cm-`/`cv-`), a different alphabet. Reading the absence of
    // `u8` as "unsigned 8-bit is unsupported" is the collision Vulkane's own
    // doc flags because it trips people. It tripped me.
    //
    // ⚠️ NOT `st8`. That is `storageBuffer8BitAccess` — 8-bit data in a
    // buffer — and a conformant device may accept 8-bit storage while doing
    // the arithmetic in `f32`. Gating compute on a storage token is a
    // silently wrong lowering on hardware that is behaving correctly.
    //
    // **This gates on the stronger requirement because it cannot see the weaker
    // one, and that IS AN OVER-REFUSAL — written down as one so the next reader
    // does not mistake it for the capability boundary.** A kernel that only
    // loads and stores these bytes is legal under `st8` alone; one that computes
    // in 8-bit needs `i8`. The distinction turns on **what the kernel does, not
    // on the type**, and `supports_dtype` is handed no plan, so it cannot tell
    // movement from compute and must assume compute.
    //
    // **`Bool` is the sharpest case and the least obvious.** FKC §5 pins
    // Bool->U8 storage, and `binary_int` spells its logical ops as
    // `((a != 0 && b != 0) ? 1 : 0)`. Whether that is 8-bit arithmetic or 32-bit
    // arithmetic on promoted operands is a property of the target's compiler
    // rather than of the IR — Slang, like C, promotes small integers. So a
    // `bool` kernel may well be legal on an `st8`-only device and this gate
    // refuses it anyway.
    //
    // **The fix is a plan-aware capability query, not a weaker constant here.**
    // Weakening this to `st8` would admit genuine 8-bit arithmetic on a device
    // that cannot do it — the failure that does not announce itself, traded for
    // one that merely costs coverage.
    // 8-bit dtypes gate on `i8` (`shaderInt8`), 16-bit on `i16` (`shaderInt16`).
    // **Both tokens are SIGNEDNESS-AGNOSTIC** — each names a capability, not a
    // component type — so `u8` gates on `i8` and `u16` gates on `i16`. There is
    // no `u8` or `u16` token and neither absence is an omission.
    let needed = if matches!(dt, ElementKind::I16 | ElementKind::U16) {
        "i16"
    } else {
        "i8"
    };
    let advertised = target
        .capability_field("arith")
        .is_some_and(|arith| arith.iter().any(|t| t == needed));
    if advertised {
        DtypeSupport::Supported
    } else if needed == "i16" {
        // The vocabulary names no 16-bit-arithmetic token yet, so a target
        // without one is silent rather than lacking. **Absence is silence, not
        // permission** — and not denial either.
        DtypeSupport::NotExpressible
    } else {
        // The token exists and the target does not carry it (or does not speak
        // about arithmetic at all — a `cuda:` or `metal:` target). **Absence is
        // silence, not permission.**
        DtypeSupport::Unsupported
    }
}

/// The `slangc` arguments a kernel of `dtype` needs for its subnormals to follow IEEE, as SPIR-V
/// `DenormPreserve`. Empty for integer dtypes. See the module docs ("Subnormals") for why this
/// is a compile argument and why it is not gated on device properties.
pub fn required_slangc_args(dtype: ElementKind) -> &'static [&'static str] {
    match dtype {
        ElementKind::F32 | ElementKind::F32Strict => &["-denorm-mode-fp32", "preserve"],
        ElementKind::F64 => &["-denorm-mode-fp64", "preserve"],
        _ => &[],
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

    fn supports_dtype(&self, dtype: ElementKind, target: TargetId) -> DtypeSupport {
        dtype_ok(dtype, target)
    }

    fn lower(&self, plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        // The ONE statement of slang's legality surface. The JIT used to keep a
        // parallel copy derived from CUDA's rules and apply it to every backend.
        if !dtype_ok(plan.dtype, plan.key.target).is_supported() {
            return Err(LowerError::UnsupportedDtype {
                dtype: plan.dtype,
                detail: format!(
                    concat!(
                        "slang backend: {:?} is not spellable for target `{}`. f16/bf16 are declined outright; ",
                        "i8/u8 require the target to advertise 8-bit arithmetic (`i8` in the vulkan ",
                        "`<arith>` field — NOT `st8`, which is storage only), and i16/u16 require ",
                        "`i16` in the same field"
                    ),
                    plan.dtype,
                    plan.key.target.as_str()
                ),
            });
        }
        let ctype = slang_ctype(plan.dtype).expect("dtype_ok implies a ctype");
        if plan.n_outputs != 1 {
            return Err(LowerError::UnsupportedPlanShape {
                detail: format!(
                    "slang backend v1: single-output only (the N-store emitter is a                      follow-up); op '{}' has {} outputs",
                    plan.op_name, plan.n_outputs
                ),
            });
        }
        if plan.out_dtype_of(0) != plan.dtype {
            return Err(LowerError::UnsupportedPlanShape {
                detail: format!(
                    "slang backend v1: uniform output dtype only — op '{}' stores {:?} from                      a {:?} compute (hetero out-dtype, e.g. a u8 keep-mask, is a follow-up)",
                    plan.op_name,
                    plan.out_dtype_of(0),
                    plan.dtype
                ),
            });
        }
        if body_has_params(plan.body) {
            return Err(LowerError::UnsupportedPlanShape {
                detail: format!(
                    "slang backend v1: parameterless elementwise only — op '{}' reads a                      runtime scalar Param (a cbuffer-carried param is a follow-up)",
                    plan.op_name
                ),
            });
        }
        // Int Div/Const backstop, REUSED from the CUDA emitter (same dtype-blind
        // hazard: infix `/` is device-UB at an int dtype, and a `Const` is an f64
        // literal). Mirrors CpuC::lower / Cuda::lower.
        //
        // Still an assert rather than an Err: this is a PLAN-gate invariant
        // (`check_int_op_admissibility` rejects these upstream in `build_plan`),
        // so reaching it means the plan gate was bypassed — a caller bug, not an
        // unsupported request. `try_build_plan` is where that becomes a typed
        // refusal; here it stays a backstop that should be unreachable.
        if unpopped::plan::is_int_dtype(plan.dtype) {
            assert_no_int_div_or_const(plan.body, plan.dtype, false, false);
        }
        match plan.schedule {
            Schedule::Scalar => emit_scalar_slang(plan, ctype),
            other => Err(LowerError::UnsupportedSchedule {
                detail: format!(
                    "slang backend v1: the scalar contiguous Elementwise path ONLY — got                      schedule {other:?}. Vectorized / Strided / Reduction / RowReduce /                      Contraction / Scan / Window / RowSort / Im2Col are follow-ups."
                ),
            }),
        }
    }
}

/// The scalar contiguous Elementwise emitter — the Slang twin of `emit_scalar`
/// (CUDA) / `emit_scalar_cpu` (CpuC). Same body math ([`lower_dag`] over the
/// shared seam), but a `StructuredBuffer`/`[numthreads]`/`SV_DispatchThreadID`
/// compute-shader harness instead of the `extern "C" __global__` grid-stride one.
fn emit_scalar_slang(plan: &KernelPlan<'_>, ctype: &str) -> Result<GeneratedKernel, LowerError> {
    let name = format!("unpopped_slang_{}_{}", plan.op_name, dtype_tag(plan.dtype));
    let n = plan.n_inputs;
    let mut s = String::new();
    s.push_str("// Generated by unpopped (slang backend) — do not edit.\n");
    s.push_str(&format!(
        "// op: {} | cell: {}\n",
        plan.op_name,
        plan.key.to_token()
    ));
    let args = required_slangc_args(plan.dtype);
    if !args.is_empty() {
        s.push_str(&format!(
            "// slangc: {}
",
            args.join(" ")
        ));
    }
    for i in 0..n {
        s.push_str(&format!("StructuredBuffer<{ctype}> input{i};\n"));
    }
    s.push_str(&format!("RWStructuredBuffer<{ctype}> output;\n"));
    s.push_str("[numthreads(256, 1, 1)]\n");
    s.push_str(&format!(
        "void {name}(uint3 tid : SV_DispatchThreadID) {{\n"
    ));
    s.push_str("    uint i = tid.x;\n");
    let acc = |idx: u8| Ok(Spelling::Spelled(format!("input{idx}[i]")));
    let (prelude, root) = lower_dag(
        &ExprDag::from_expr(plan.body),
        ctype,
        // Through the builder — the only route open to a backend outside the
        // `unpopped` crate, which this one now is. See the same note in
        // `unpopped-cpu-c`.
        //
        // No `.arith()`: Slang lowers no struct-typed dtype, so the C infix
        // default is right for everything it spells. That is a real answer, not
        // an omission — and it is why the default is a working spelling rather
        // than a panic like `reduced`/`coord`.
        &Lowering::builder(
            &acc,
            &|op, x| slang_unary(op, x, plan.dtype),
            &|op, a, b| slang_binary(op, a, b, plan.dtype),
        )
        .select(&|c, a, b| slang_select(c, a, b, plan.dtype))
        .constant(&|v| Ok(Spelling::Spelled(slang_const(v))))
        .build(),
    )?;
    for decl in &prelude {
        s.push_str(&format!("    {decl}\n"));
    }
    s.push_str(&format!("    output[i] = {root};\n"));
    s.push_str("}\n");
    Ok(GeneratedKernel::new(name, s))
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

/// Lower a unary op for `dtype` in Slang. f32/f64 share ONE speller (HLSL
/// intrinsics are overloaded — no `expf` vs `exp` split); integers have no unary
/// math (the ir admissibility table rejects it, so the panic is the backstop).
fn slang_unary(op: UnaryOp, x: String, dtype: ElementKind) -> Result<Spelling, LowerError> {
    match dtype {
        ElementKind::F32 | ElementKind::F32Strict | ElementKind::F64 => slang_unary_fp(op, x),
        other => Ok(Spelling::Declined(Decline::UnsupportedDtypeForOp {
            op: DeclinedOp::Unary(op),
            dtype: other,
            why: "v1 unary is float/double; integer dtypes have no unary math".to_string(),
        })),
    }
}

/// Slang unary spellers (overloaded over float/double). Intrinsics that HLSL/Slang
/// provides directly are used as-is; the compare/select-shaped ops mirror the
/// CUDA ternaries verbatim (Slang ternary syntax is identical) for matching NaN
/// semantics. Ops with no base-profile Slang intrinsic are declined.
fn slang_unary_fp(op: UnaryOp, x: String) -> Result<Spelling, LowerError> {
    let spelled = match op {
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
        | UnaryOp::Atanh => return Ok(Spelling::Declined(Decline::UnsupportedOp {
            op: DeclinedOp::Unary(op),
            why: "no base-profile Slang intrinsic; erf/erfc/gelu/lgamma/cbrt/asinh/acosh/atanh need polyfills".to_string(),
        })),
    };
    Ok(Spelling::Spelled(spelled))
}

/// Lower a non-infix binary op for `dtype` in Slang. f32/f64 share the overloaded
/// fp speller; I32/I64 route to the raw-operator int speller.
fn slang_binary(
    op: BinaryOp,
    a: String,
    b: String,
    dtype: ElementKind,
) -> Result<Spelling, LowerError> {
    match dtype {
        ElementKind::F32 | ElementKind::F32Strict => slang_binary_fp(op, a, b, "float"),
        ElementKind::F64 => slang_binary_fp(op, a, b, "double"),
        ElementKind::I32 | ElementKind::I64 => slang_binary_int(op, a, b, dtype),
        other => Ok(Spelling::Declined(Decline::UnsupportedDtypeForOp {
            op: DeclinedOp::Binary(op),
            dtype: other,
            why: "no binary math at this dtype".to_string(),
        })),
    }
}

/// Slang fp binary spellers. `ct` (`"float"`/`"double"`) casts the compare
/// operands so the compare is decided IN THE COMPUTE DTYPE (the CUDA `binary_f32`
/// double-promotion lesson). Non-infix intrinsics are overloaded so `ct` is unused
/// by them. `Max`/`Min` stay the NaN-propagating compare-selects (torch semantics);
/// `FmaxIeee`/`FminIeee` are the NaN-suppressing intrinsics.
fn slang_binary_fp(op: BinaryOp, a: String, b: String, ct: &str) -> Result<Spelling, LowerError> {
    let spelled = match op {
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
        BinaryOp::Copysign | BinaryOp::Nextafter => {
            return Ok(Spelling::Declined(Decline::UnsupportedOp {
                op: DeclinedOp::Binary(op),
                why: "declined by design, or int-only with no float lowering".to_string(),
            }));
        }
        BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor
        | BinaryOp::Shl
        | BinaryOp::Shr
        | BinaryOp::LogicalAnd
        | BinaryOp::LogicalOr
        | BinaryOp::LogicalXor => {
            return Ok(Spelling::Declined(Decline::UnsupportedOp {
                op: DeclinedOp::Binary(op),
                why: "declined by design, or int-only with no float lowering".to_string(),
            }));
        }
    };
    Ok(Spelling::Spelled(spelled))
}

/// Slang integer binary speller (I32/I64) — the raw operators, matching the CUDA
/// `binary_int` speller. Logical ops are U8(Bool)-only (declined in v1), so they
/// backstop-panic here.
fn slang_binary_int(
    op: BinaryOp,
    a: String,
    b: String,
    dtype: ElementKind,
) -> Result<Spelling, LowerError> {
    let spelled = match op {
        BinaryOp::BitAnd => format!("({a} & {b})"),
        BinaryOp::BitOr => format!("({a} | {b})"),
        BinaryOp::BitXor => format!("({a} ^ {b})"),
        BinaryOp::Shl => format!("({a} << {b})"),
        BinaryOp::Shr => format!("({a} >> {b})"),
        // Carries the dtype, because "no integer lowering for this op" is a claim
        // about the PAIR: the logical ops are the Bool surface and decline here at
        // every integer dtype, which is a different fact from an op nothing spells.
        _ => {
            return Ok(Spelling::Declined(Decline::UnsupportedDtypeForOp {
                op: DeclinedOp::Binary(op),
                dtype,
                why: "no integer lowering for this op; the logical ops are the bespoke                       Bool surface".to_string(),
            }));
        }
    };
    Ok(Spelling::Spelled(spelled))
}

/// Ternary select for `dtype` in Slang — the identity-cast-pinned ternary (the
/// CUDA `select_f32` contract: `(ct)` casts pin the compare + arms in the compute
/// dtype against a suffix-less `Const` literal, and no arithmetic touches an arm).
fn slang_select(
    c: String,
    a: String,
    b: String,
    dtype: ElementKind,
) -> Result<Spelling, LowerError> {
    Ok(match dtype {
        ElementKind::F32 | ElementKind::F32Strict => Spelling::Spelled(format!(
            "(((float)({c})) != 0.0 ? (float)({a}) : (float)({b}))"
        )),
        ElementKind::F64 => Spelling::Spelled(format!(
            "(((double)({c})) != 0.0 ? (double)({a}) : (double)({b}))"
        )),
        other => Spelling::Declined(Decline::UnsupportedDtypeForOp {
            op: DeclinedOp::Select,
            dtype: other,
            why: "v1 select is float-only (f32/f64)".to_string(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use unpopped::generate;
    use unpopped::ir::{OpDef, input};
    use unpopped_vocab::{ArchSku, OpCategory, OperandDesc, structure_key};

    /// Float kernels name the `slangc` argument their subnormals need; integer kernels name none.
    /// The header must agree with [`required_slangc_args`] (one source), per dtype.
    #[test]
    fn float_kernels_name_the_denorm_flag_and_integer_kernels_do_not() {
        let cases = [
            (
                ElementKind::F32,
                Some("// slangc: -denorm-mode-fp32 preserve"),
            ),
            (
                ElementKind::F64,
                Some("// slangc: -denorm-mode-fp64 preserve"),
            ),
            (ElementKind::I32, None),
        ];
        for (dt, want) in cases {
            let op = OpDef::elementwise("dn", 2, &[dt], input(0) + input(1));
            let k = generate(&op, &binary_scalar_key(dt, 4), &Slang);
            let line = k.source.lines().find(|l| l.starts_with("// slangc:"));
            assert_eq!(
                line, want,
                "{dt:?}:
{}",
                k.source
            );
            let args = required_slangc_args(dt);
            assert_eq!(args.is_empty(), want.is_none(), "{dt:?}");
            if let Some(w) = want {
                assert_eq!(w, format!("// slangc: {}", args.join(" ")), "{dt:?}");
            }
        }
    }

    /// The decline text for a 16-bit integer must say what is true today: vocabulary v5 shipped,
    /// so i16/u16 need `i16` in the target's `<arith>` field. It used to say they "wait on
    /// vocabulary v5". Adopters match this text (ROADMAP X5).
    #[test]
    fn sixteen_bit_decline_names_the_arith_token_not_a_vocabulary_wait() {
        let op = OpDef::elementwise("w16", 2, &[ElementKind::I16], input(0) + input(1));
        let key = binary_scalar_key(ElementKind::I16, 4);
        let msg = match std::panic::catch_unwind(|| generate(&op, &key, &Slang)) {
            Ok(k) => panic!(
                "expected a decline on a target without i16:
{}",
                k.source
            ),
            Err(e) => e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .expect("string panic payload"),
        };
        assert!(
            msg.contains("i16/u16 require `i16` in the same field"),
            "{msg}"
        );
        assert!(!msg.contains("wait on vocabulary v5"), "{msg}");
    }

    /// A NaN or infinity constant must not emit a C macro Slang does not define. Control: the
    /// finite constant beside it is spelled exactly as before.
    #[test]
    fn non_finite_constants_are_not_spelled_as_c_macros() {
        use unpopped::ir::konst;
        for (v, want) in [
            (f64::INFINITY, "asfloat(0x7f800000u)"),
            (f64::NEG_INFINITY, "asfloat(0xff800000u)"),
            (f64::NAN, "asfloat(0x7fc00000u)"),
        ] {
            let op = OpDef::elementwise("nf", 2, &[ElementKind::F32], input(0) + konst(v));
            let k = generate(&op, &binary_scalar_key(ElementKind::F32, 4), &Slang);
            assert!(k.source.contains(want), "{v}:\n{}", k.source);
            assert!(
                !k.source.contains("INFINITY") && !k.source.contains("NAN"),
                "{v}:\n{}",
                k.source
            );
        }
        let op = OpDef::elementwise("fin", 2, &[ElementKind::F32], input(0) + konst(2.5));
        let k = generate(&op, &binary_scalar_key(ElementKind::F32, 4), &Slang);
        assert!(
            k.source.contains("2.5") && !k.source.contains("asfloat"),
            "{}",
            k.source
        );
    }

    /// A binary Elementwise cell whose small `align` defeats vectorization, so
    /// `build_plan` derives [`Schedule::Scalar`] (the path Slang v1 serves).
    fn binary_scalar_key(dt: ElementKind, align: u32) -> unpopped_vocab::StructureKey {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, align);
        structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89)
    }

    fn unary_scalar_key(dt: ElementKind, align: u32) -> unpopped_vocab::StructureKey {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, align);
        structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89)
    }

    /// The 8-bit gate reads the target, and refuses when the target is silent.
    ///
    /// # Both arms, deliberately
    ///
    /// A gate that always refuses is **indistinguishable from a correctly strict
    /// one** until something that should pass, doesn't — and if no corpus
    /// exercises the dtype, nothing ever notices. A one-armed test proves only
    /// that the matcher I wrote agrees with itself.
    ///
    /// # Order does NOT matter here, and that is worth stating
    ///
    /// §6.8-0002's byte-exact rule governs *matching whole sets* and *spelling*
    /// one — a builder must sort before joining, because `arith-i8-f16` matches
    /// nothing. This gate does neither: it asks whether the parsed tuple list
    /// **contains** `i8`, and membership is legitimately order-insensitive. The
    /// permuted arm below is here to pin that reading rather than to pass by
    /// accident, because a future reader who knows the order rule will otherwise
    /// assume this code is broken.
    #[test]
    fn eight_bit_gates_on_the_arith_field_and_refuses_when_silent() {
        let t = |tok: &str| TargetId::parse(tok).expect("valid token");
        let can = |tok: &str, dt| Slang.supports_dtype(dt, t(tok)).is_supported();

        // ADVERTISED -> supported, and `u8` gates on `i8` because `shaderInt8`
        // is signedness-agnostic. There is no `u8` token to look for.
        assert!(can("vulkan:sg32.arith-f16-i8", ElementKind::I8));
        assert!(can("vulkan:sg32.arith-f16-i8", ElementKind::U8));
        // Permuted: malformed to SPELL, but membership still holds. Pinned so
        // nobody "fixes" this into an order-sensitive check.
        assert!(can("vulkan:sg32.arith-i8-f16", ElementKind::U8));

        // NOT ADVERTISED -> refused. This is the arm that makes the test mean
        // something.
        assert!(!can("vulkan:sg32.arith-f16", ElementKind::I8));
        assert!(!can("vulkan:sg32.arith-f16", ElementKind::U8));
        assert!(!can("vulkan:sg32.arith-none", ElementKind::U8));

        // SILENT -> refused. A `cuda:` token has no `arith` field at all, and
        // absence is silence rather than permission.
        assert!(!can("cuda:sm89", ElementKind::U8));
        assert!(!can("vulkan:sg32.ops-abr", ElementKind::U8));

        // ⚠️ `st8` is NOT `i8`. `storageBuffer8BitAccess` means 8-bit data may
        // live in a buffer; a conformant device may accept that and still do the
        // arithmetic in f32. Gating compute on a storage token is a silently
        // wrong lowering on hardware behaving correctly, so a target offering
        // storage-only must still refuse.
        assert!(!can("vulkan:sg32.arith-f16.st8-yes", ElementKind::U8));

        // And an `i8` sitting in some OTHER field must not answer for `arith` —
        // the reader takes the named field, never a substring of the token.
        assert!(!can("vulkan:sg32.arith-none.cm-i8", ElementKind::U8));

        // `bool` gates the same way, because FKC §5 stores it as U8. This arm
        // is the one most likely to be "corrected" later by someone who reasons
        // that a boolean is not really 8-bit arithmetic — which may even be true
        // on a given compiler, and is exactly why the refusal is recorded as an
        // over-refusal in `dtype_ok` rather than defended as a boundary.
        assert!(can("vulkan:sg32.arith-f16-i8", ElementKind::Bool));
        assert!(!can("vulkan:sg32.arith-f16", ElementKind::Bool));
        assert!(!can("vulkan:sg32.st8-yes", ElementKind::Bool));
        assert!(!can("cuda:sm89", ElementKind::Bool));

        // 16-BIT GATES ON ITS OWN TOKEN, and this is the arm that would pass
        // by accident under a gate that checked only `i8`. `arith-f16-i8`
        // advertises 8-bit arithmetic and says nothing about 16-bit, so a target
        // offering i8 alone must still refuse i16/u16.
        assert!(can("vulkan:sg32.arith-i16", ElementKind::I16));
        assert!(can("vulkan:sg32.arith-i16", ElementKind::U16));
        assert!(can("vulkan:sg32.arith-f16-i16-i8", ElementKind::U16));
        assert!(
            !can("vulkan:sg32.arith-f16-i8", ElementKind::I16),
            "i8 must not answer for i16 — separate capabilities"
        );
        assert!(
            !can("vulkan:sg32.arith-i16", ElementKind::U8),
            "and i16 must not answer for u8, in the other direction"
        );
        assert!(!can("cuda:sm89", ElementKind::U16));

        // Universally-spellable dtypes are unaffected by the target.
        assert!(can("cuda:sm89", ElementKind::F32));
        assert!(can("vulkan:sg32.arith-none", ElementKind::U32));
    }

    /// `u32`/`u64` lower to `uint`/`uint64_t`.
    ///
    /// The v1 `slang_ctype` comment claimed U32 had "no clean base-profile
    /// scalar type". That was never true — Slang documents `int`/`int32_t` and
    /// `uint`/`uint32_t` as the two integer types that are *universally*
    /// supported. The coverage table said so (`this one is entirely ours to
    /// add`) while the code said the opposite, and the table was right.
    ///
    /// # The warrant covers u32, and u64 rides on i64's existing assumption
    ///
    /// Slang's universality claim is about the **32-bit** pair. `uint64_t` is a
    /// capability-dependent type, exactly as `int64_t` is — and this backend has
    /// claimed `int64_t` with no capability check since v1. So u64 is admitted
    /// for **symmetry with an assumption already being made**, not because the
    /// universality argument reaches it. If i64 is over-claiming here then so is
    /// u64, and both are answered by the same `supports_dtype` three-state
    /// [`DtypeSupport`] — not by this test.
    #[test]
    fn unsigned_elementwise_emits_uint_buffers() {
        for (dt, ctype, tag) in [
            (ElementKind::U32, "uint", "u32"),
            (ElementKind::U64, "uint64_t", "u64"),
        ] {
            let op = OpDef::elementwise("bits", 2, &[dt], input(0) * input(1) + input(0));
            let k = generate(&op, &binary_scalar_key(dt, 4), &Slang);
            let src = &k.source;
            assert_eq!(k.name, format!("unpopped_slang_bits_{tag}"));
            assert!(
                src.contains(&format!("StructuredBuffer<{ctype}> input0;")),
                "{tag}: operand buffer is not {ctype}:
{src}"
            );
            assert!(
                src.contains(&format!("RWStructuredBuffer<{ctype}> output;")),
                "{tag}: output buffer is not {ctype}:
{src}"
            );
            assert!(
                src.contains("output[i] = ((input0[i] * input1[i]) + input0[i]);"),
                "{tag}: body did not lower:
{src}"
            );
            // The index is `uint` and so is the data at u32. A ctype that
            // collided with the loop variable would still emit and still read
            // fine here, so this pins the DECLARATION rather than a usage.
            assert!(
                src.contains("uint i = tid.x;"),
                "{tag}: index decl:
{src}"
            );
        }
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
        use unpopped::ir::UnaryOp;
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
        use unpopped::ir::UnaryOp;
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

    /// Issue #6: the two refusals are DIFFERENT facts. 8-bit has a vocabulary
    /// token, so a target without it lacks the capability (`Unsupported`); 16-bit
    /// has none, so a target without it is silent (`NotExpressible`). Both are
    /// not-supported, and a `bool` could not tell them apart.
    /// The OCP MX element formats (KISS-CLASSIFY 6.1-0014) are storage and key identity
    /// only: Slang has no type to spell them with, on any target.
    #[test]
    fn the_mx_element_formats_are_not_computed_on_any_target() {
        let t = |tok: &str| TargetId::parse(tok).expect("valid token");
        for dt in [
            ElementKind::F4E2M1,
            ElementKind::F6E2M3,
            ElementKind::F6E3M2,
        ] {
            for tok in ["vulkan:sg32.arith-f16-i8", "cuda:sm89"] {
                assert!(
                    !Slang.supports_dtype(dt, t(tok)).is_supported(),
                    "{dt:?} on {tok}"
                );
            }
        }
        // Control: a computed dtype is supported on the same target.
        assert!(
            Slang
                .supports_dtype(ElementKind::F32, t("vulkan:sg32.arith-f16-i8"))
                .is_supported()
        );
    }

    #[test]
    fn supports_dtype_separates_a_lacking_device_from_a_silent_vocabulary() {
        let t = |tok: &str| TargetId::parse(tok).expect("valid token");
        let at = |tok: &str, dt| Slang.supports_dtype(dt, t(tok));

        assert_eq!(
            at("vulkan:sg32.arith-f16-i8", ElementKind::I8),
            DtypeSupport::Supported
        );
        // Token exists, target lacks it: the device lacks it.
        assert_eq!(
            at("vulkan:sg32.arith-f16", ElementKind::I8),
            DtypeSupport::Unsupported
        );
        assert_eq!(at("cuda:sm89", ElementKind::U8), DtypeSupport::Unsupported);
        // No token exists for 16-bit arithmetic: the vocabulary cannot say.
        assert_eq!(
            at("vulkan:sg32.arith-f16-i8", ElementKind::I16),
            DtypeSupport::NotExpressible
        );
        assert_eq!(
            at("vulkan:sg32.arith-f16-i8", ElementKind::U16),
            DtypeSupport::NotExpressible
        );
        // A dtype Slang never spells is its own refusal, not a vocabulary gap.
        assert_eq!(
            at("vulkan:sg32.arith-f16-i8", ElementKind::F16),
            DtypeSupport::Unsupported
        );
    }

    #[test]
    fn declines_f16_and_non_scalar_dtypes_via_supports_dtype() {
        assert!(
            !Slang
                .supports_dtype(ElementKind::F16, ArchSku::Sm89.into())
                .is_supported()
        );
        assert!(
            !Slang
                .supports_dtype(ElementKind::Bf16, ArchSku::Sm89.into())
                .is_supported()
        );
        // U32/U64 now lower — see `slang_ctype` and
        // `unsigned_elementwise_emits_uint_buffers`. i8/u8/i16/u16 stay declined.
        assert!(
            !Slang
                .supports_dtype(ElementKind::I8, ArchSku::Sm89.into())
                .is_supported()
        );
        assert!(
            !Slang
                .supports_dtype(ElementKind::U8, ArchSku::Sm89.into())
                .is_supported()
        );
        assert!(
            !Slang
                .supports_dtype(ElementKind::I16, ArchSku::Sm89.into())
                .is_supported()
        );
        assert!(
            !Slang
                .supports_dtype(ElementKind::U16, ArchSku::Sm89.into())
                .is_supported()
        );
        assert!(
            !Slang
                .supports_dtype(ElementKind::I8, ArchSku::Sm89.into())
                .is_supported()
        );
        for dt in [
            ElementKind::F32,
            ElementKind::F64,
            ElementKind::I32,
            ElementKind::I64,
        ] {
            assert!(
                Slang
                    .supports_dtype(dt, ArchSku::Sm89.into())
                    .is_supported(),
                "{dt:?} should be supported"
            );
        }
    }

    #[test]
    #[should_panic(expected = "scalar contiguous Elementwise path ONLY")]
    fn declines_non_scalar_schedule() {
        // A V4-aligned contiguous binary f32 op keys Schedule::Vectorized.
        let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
        let _ = generate(&op, &binary_scalar_key(ElementKind::F32, 256), &Slang);
    }

    /// **Slang declines `Fma` rather than spelling a two-rounding substitute.**
    ///
    /// SPIR-V's `Fma` (and HLSL `mad`) carry no single-rounding guarantee on every
    /// Vulkan device, so this emitter has no spelling it can promise is fused. The
    /// contract of `ScalarExpr::Fma` is one rounding, so the honest answer is a
    /// typed decline. Control: the same shape built from `Mul`+`Add` generates, so
    /// the decline is about the fused node and not about the key or the dtype.
    #[test]
    fn declines_fma_but_generates_the_unfused_form() {
        let key = binary_scalar_key(ElementKind::F32, 4);
        let fused = OpDef::elementwise(
            "fma",
            3,
            &[ElementKind::F32],
            input(0).fma(input(1), input(2)),
        );
        let err = unpopped::try_generate(&fused, &key, &Slang)
            .expect_err("Slang must decline a fused multiply-add");
        assert!(
            format!("{err:?}").contains("Fma"),
            "the decline must name Fma, got {err:?}"
        );
        let unfused = OpDef::elementwise(
            "mulAdd",
            3,
            &[ElementKind::F32],
            input(0) * input(1) + input(2),
        );
        unpopped::try_generate(&unfused, &key, &Slang)
            .expect("the unfused form must still generate (positive control)");
    }

    /// **Slang declines `Cast`/`Bitcast` until it can spell the exact rounding and
    /// saturation.** `typed_expressions` in the cpu_c end-to-end suite is the model for
    /// the spelling; until Slang has one, a typed op is a typed decline, never a plain
    /// HLSL cast. Control: the same op without the cast generates.
    #[test]
    fn declines_a_cast_but_generates_the_untyped_form() {
        // A round trip through i32 keeps the store type at f32, so the decline under
        // test is the Cast seam and not Slang's separate hetero-output decline.
        let op = OpDef::elementwise(
            "roundTrip",
            1,
            &[ElementKind::F32],
            input(0)
                .cast(
                    ElementKind::I32,
                    unpopped::ir::RoundMode::TowardZero,
                    unpopped::ir::Saturation::Clamp,
                )
                .cast(
                    ElementKind::F32,
                    unpopped::ir::RoundMode::NearestEven,
                    unpopped::ir::Saturation::Wrap,
                ),
        );
        let key = binary_scalar_key(ElementKind::F32, 4);
        let err = unpopped::try_generate(&op, &key, &Slang)
            .expect_err("Slang must decline a Cast it cannot spell exactly");
        assert!(format!("{err:?}").contains("Cast"), "{err:?}");
        let plain = OpDef::elementwise("neg", 1, &[ElementKind::F32], input(0) + input(0));
        let key = binary_scalar_key(ElementKind::F32, 4);
        let _ = unpopped::try_generate(&plain, &key, &Slang)
            .expect("the untyped form must still generate (positive control)");
    }

    /// A `Qualified` operation is a typed decline in Slang, never the bare operation
    /// (which would silently drop the rounding mode / ftz / approx). Control: the same
    /// op unqualified generates, so the decline is the qualifier and nothing else.
    #[test]
    fn declines_a_qualified_op_and_names_the_qualifier() {
        let key = binary_scalar_key(ElementKind::F32, 4);
        for q in [
            (input(0) + input(0)).ftz(),
            (input(0) / input(0)).approx(),
            (input(0) * input(0)).rounded(unpopped::ir::RoundMode::TowardZero),
        ] {
            let op = OpDef::elementwise("q", 1, &[ElementKind::F32], q);
            let err = unpopped::try_generate(&op, &key, &Slang)
                .expect_err("Slang must decline a Qualified op it cannot honour");
            assert!(format!("{err:?}").contains("Qualified"), "{err:?}");
        }
        let plain = OpDef::elementwise("p", 1, &[ElementKind::F32], input(0) + input(0));
        unpopped::try_generate(&plain, &key, &Slang)
            .expect("the unqualified form must still generate (positive control)");
    }

    /// The residue-round-trip contract: emitted Slang re-lifts to the SAME IR —
    /// proving emit↔lift symmetry (a language must be re-emittable to dump its
    /// un-liftable spans back out). Needs the `convert` feature for the lifter.
    #[cfg(feature = "convert")]
    #[test]
    fn slang_emit_round_trips_through_the_lifter() {
        use unpopped::convert::{SLANG, lift_elementwise};
        use unpopped::ir::ScalarExpr;
        let op = OpDef::elementwise("mul", 2, &[ElementKind::F32], input(0) * input(1));
        let k = generate(&op, &binary_scalar_key(ElementKind::F32, 4), &Slang);
        let lifted = lift_elementwise(&SLANG, &k.source, "mul", &[ElementKind::F32])
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
        use std::process::Command;
        use unpopped::ir::{BinaryOp, UnaryOp};

        let kernels = vec![
            generate(
                &OpDef::elementwise(
                    "nonfinite",
                    1,
                    &[ElementKind::F32],
                    (input(0) + unpopped::ir::konst(f64::INFINITY))
                        + (unpopped::ir::konst(f64::NEG_INFINITY) + unpopped::ir::konst(f64::NAN)),
                ),
                &unary_scalar_key(ElementKind::F32, 4),
                &Slang,
            ),
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
