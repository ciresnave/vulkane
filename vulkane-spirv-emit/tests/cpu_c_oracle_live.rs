// SPDX-License-Identifier: MIT OR Apache-2.0
//! G4.2, the behaviour oracle (`docs/design/spirv-parser-emitter.md` section S4): the same
//! `OpDef` is lowered by `unpopped-cpu-c` (portable C, compiled and run on the host) and by this
//! emitter (SPIR-V, run on a live device) over the same input bits, and the two result
//! bit-streams are compared cell by cell.
//!
//! Integer ops, comparisons, `min`/`max` and the rounding family must agree bit for bit.
//! Transcendentals are not required to: Vulkan grants them a precision budget and a host libm
//! another, so they are held to a tolerance that is small next to any wrong-function error.
//! Both NaNs count as equal (payloads are not specified across the two).
//!
//! `f64` is not run: a device needs `shaderFloat64` enabled, which the shared test device does not
//! request. The validity of the `f64` modules is covered by `spirv-val` in the unit tests.
//!
//! Without a device, or without a host C compiler, the test follows `VULKANE_REQUIRE_DEVICE`
//! (device) and `CI` (compiler, see `support/cc.rs`) rather than passing silently.

#[path = "support/cc.rs"]
mod cc;
#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(dead_code, unreachable_pub, missing_docs)]
mod common;
#[path = "support/gpu.rs"]
mod gpu;

use std::fmt::Write as _;
use std::process::Command;

use unpopped::ir::{BinaryOp, OpDef, UnaryOp, input};
use unpopped::try_generate;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key};
use vulkane_spirv_emit::SpirV;

/// How closely the device must match the host for one op.
#[derive(Copy, Clone, Debug, PartialEq)]
enum Tol {
    /// Bit-identical (NaN equals NaN).
    Exact,
    /// Bit-identical except that `+0` and `-0` are the same: SPIR-V does not specify the sign of a
    /// zero result (`OpFRem`), and a host `fmod` keeps the dividend's.
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

fn agree(tol: Tol, dt: ElementKind, got: u32, want: u32) -> bool {
    if got == want {
        return true;
    }
    if dt != ElementKind::F32 {
        return false;
    }
    let (g, w) = (f32::from_bits(got), f32::from_bits(want));
    if g.is_nan() && w.is_nan() {
        return true;
    }
    match tol {
        Tol::Exact => false,
        Tol::ZeroSign => g == 0.0 && w == 0.0,
        Tol::Close { rel, abs } => {
            if g.is_infinite() || w.is_infinite() {
                return g == w;
            }
            (g - w).abs() <= rel * w.abs() + abs
        }
    }
}

/// Which input rows the op is specified on. Outside it SPIR-V leaves the result undefined (or, for
/// a signed-zero input, device-dependent), so a disagreement there is reported, not failed.
#[derive(Copy, Clone, Debug)]
enum Domain {
    All,
    /// GLSL.std.450 `Pow`: undefined for `x < 0` and for `x == 0 && y <= 0`.
    Pow,
    /// GLSL.std.450 `Atan2`: undefined for `x == 0 && y == 0`; a `-0` `y` is also excluded, because
    /// whether a device keeps the sign of a zero input is not something SPIR-V promises.
    Atan2,
}

impl Domain {
    fn holds(self, row: &[u32]) -> bool {
        let f = |i: usize| f32::from_bits(row[i]);
        match self {
            Domain::All => true,
            Domain::Pow => f(0) > 0.0,
            Domain::Atan2 => !(f(0) == 0.0 && f(1) == 0.0) && row[0] != (-0.0f32).to_bits(),
        }
    }
}

struct Case {
    domain: Domain,
    name: String,
    dt: ElementKind,
    tol: Tol,
    def: OpDef,
    key: StructureKey,
    inputs: Vec<Vec<u32>>,
}

fn key(n_operands: usize, cat: OpCategory, dt: ElementKind) -> StructureKey {
    // A small `align` defeats vectorization, so the plan is `Schedule::Scalar` for both backends.
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
        Neg | Abs | Floor | Ceil | Trunc | Round | Relu | Sign | Step => Tol::Exact,
        Sin | Cos | Tan => TRIG,
        _ => CLOSE,
    }
}

fn binary_tol(op: BinaryOp) -> Tol {
    use BinaryOp::*;
    match op {
        Max | Min | FmaxIeee | FminIeee | CmpEq | CmpNe | CmpLt | CmpLe | CmpGt | CmpGe | Rem => {
            Tol::Exact
        }
        RemTrunc => Tol::ZeroSign,
        _ => CLOSE,
    }
}

fn cases() -> Vec<Case> {
    use BinaryOp as B;
    use UnaryOp as U;
    let f = ElementKind::F32;
    let mut v = Vec::new();

    for op in [
        U::Neg,
        U::Abs,
        U::Sqr,
        U::Sqrt,
        U::Rsqrt,
        U::Recip,
        U::Exp,
        U::Log,
        U::Tanh,
        U::Sigmoid,
        U::Relu,
        U::Silu,
        U::Sin,
        U::Cos,
        U::Floor,
        U::Ceil,
        U::Round,
        U::Sign,
        U::Step,
        U::Trunc,
        U::Exp2,
        U::Log2,
        U::Expm1,
        U::Log10,
        U::Log1p,
        U::Sinh,
        U::Cosh,
        U::Tan,
        U::Asin,
        U::Acos,
        U::Atan,
    ] {
        let name = format!("u_{op:?}_f32").to_lowercase();
        v.push(Case {
            domain: Domain::All,
            def: OpDef::elementwise(&name, 1, &[f], input(0).unary(op)),
            name,
            dt: f,
            tol: unary_tol(op),
            key: key(2, OpCategory::UnaryElementwise, f),
            inputs: vec![f32_bits()],
        });
    }

    for op in [
        B::Max,
        B::Min,
        B::Pow,
        B::Rem,
        B::RemTrunc,
        B::Atan2,
        B::FmaxIeee,
        B::FminIeee,
        B::CmpEq,
        B::CmpNe,
        B::CmpLt,
        B::CmpLe,
        B::CmpGt,
        B::CmpGe,
    ] {
        let name = format!("b_{op:?}_f32").to_lowercase();
        v.push(Case {
            domain: match op {
                B::Pow => Domain::Pow,
                B::Atan2 => Domain::Atan2,
                _ => Domain::All,
            },
            def: OpDef::elementwise(&name, 2, &[f], input(0).binary(op, input(1))),
            name,
            dt: f,
            tol: binary_tol(op),
            key: key(3, OpCategory::BinaryElementwise, f),
            inputs: pairs(&f32_bits()),
        });
    }

    type Build = fn() -> unpopped::ir::Expr;
    let arith: [(&str, Build); 4] = [
        ("add", || input(0) + input(1)),
        ("sub", || input(0) - input(1)),
        ("mul", || input(0) * input(1)),
        ("div", || input(0) / input(1)),
    ];
    for (tag, build) in arith {
        let name = format!("a_{tag}_f32");
        v.push(Case {
            domain: Domain::All,
            def: OpDef::elementwise(&name, 2, &[f], build()),
            name,
            dt: f,
            tol: if tag == "div" { CLOSE } else { Tol::Exact },
            key: key(3, OpCategory::BinaryElementwise, f),
            inputs: pairs(&f32_bits()),
        });
    }

    for dt in [ElementKind::I32, ElementKind::U32] {
        let tag = if dt == ElementKind::I32 { "i32" } else { "u32" };
        for op in [B::BitAnd, B::BitOr, B::BitXor, B::Shl, B::Shr] {
            let name = format!("b_{op:?}_{tag}").to_lowercase();
            v.push(Case {
                domain: Domain::All,
                def: OpDef::elementwise(&name, 2, &[dt], input(0).binary(op, input(1))),
                name,
                dt,
                tol: Tol::Exact,
                key: key(3, OpCategory::BinaryElementwise, dt),
                inputs: pairs(&INT_VALUES),
            });
        }
        let name = format!("a_mixed_{tag}");
        v.push(Case {
            domain: Domain::All,
            def: OpDef::elementwise(&name, 2, &[dt], input(0) * input(1) + input(0) - input(1)),
            name,
            dt,
            tol: Tol::Exact,
            key: key(3, OpCategory::BinaryElementwise, dt),
            inputs: pairs(&INT_VALUES),
        });
    }
    v
}

fn ctype(dt: ElementKind) -> &'static str {
    match dt {
        ElementKind::F32 => "float",
        ElementKind::I32 => "int",
        ElementKind::U32 => "unsigned int",
        other => panic!("{other:?} is not run by this oracle"),
    }
}

/// One C program holding every case's kernel; it prints `case <index>` then each output cell as
/// eight hex digits of its bit pattern.
fn c_program(cases: &[Case], sources: &[(String, String)]) -> String {
    let mut s = String::from("#include <stdio.h>\n#include <string.h>\n#include <math.h>\n");
    for (_, src) in sources {
        s.push_str(src);
        s.push('\n');
    }
    s.push_str("int main(void) {\n");
    for (ci, (case, (kname, _))) in cases.iter().zip(sources).enumerate() {
        let n = case.inputs[0].len();
        let ct = ctype(case.dt);
        for (k, col) in case.inputs.iter().enumerate() {
            let lits: Vec<String> = col.iter().map(|w| format!("{w}u")).collect();
            let _ = writeln!(
                s,
                "    static const unsigned in{ci}_{k}[{n}] = {{{}}};",
                lits.join(", ")
            );
        }
        let _ = writeln!(s, "    static unsigned out{ci}[{n}];");
        let args: Vec<String> = (0..case.inputs.len())
            .map(|k| format!("(const {ct}*)in{ci}_{k}"))
            .collect();
        let _ = writeln!(s, "    {kname}({}, ({ct}*)out{ci}, {n});", args.join(", "));
        let _ = writeln!(
            s,
            "    printf(\"case {ci}\\n\"); for (int i = 0; i < {n}; ++i) printf(\"%08x\\n\", out{ci}[i]);"
        );
    }
    s.push_str("    return 0;\n}\n");
    s
}

/// Compile and run the C program; the per-case output columns.
fn host_results(cc: &cc::CCompiler, cases: &[Case], sources: &[(String, String)]) -> Vec<Vec<u32>> {
    let dir = std::env::temp_dir().join(format!("vulkane_cpu_c_oracle_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let c_file = dir.join("oracle.c");
    let exe = dir.join("oracle.exe");
    std::fs::write(&c_file, c_program(cases, sources)).unwrap();
    cc.compile(&c_file, &exe, false)
        .unwrap_or_else(|e| panic!("the CPU-C oracle program did not compile:\n{e}"));
    let out = Command::new(&exe).output().expect("run the oracle program");
    assert!(
        out.status.success(),
        "oracle program exited {:?}",
        out.status
    );
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let mut cols: Vec<Vec<u32>> = Vec::new();
    for line in text.lines() {
        if line.starts_with("case ") {
            cols.push(Vec::new());
        } else {
            cols.last_mut()
                .expect("a result before any case header")
                .push(u32::from_str_radix(line.trim(), 16).expect("hex cell"));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    cols
}

#[test]
fn agree_has_a_control_that_can_fail() {
    let one = 1.0f32.to_bits();
    let next = f32::from_bits(one + 1).to_bits();
    assert!(!agree(Tol::Exact, ElementKind::F32, one, next));
    assert!(agree(CLOSE, ElementKind::F32, one, next));
    assert!(!agree(CLOSE, ElementKind::F32, one, 2.0f32.to_bits()));
    assert!(!agree(Tol::Exact, ElementKind::I32, 1, 2));
    assert!(agree(
        Tol::Exact,
        ElementKind::F32,
        f32::NAN.to_bits(),
        (-f32::NAN).to_bits()
    ));
}

#[test]
fn emitted_spirv_matches_the_cpu_c_oracle_on_a_device() {
    common::require_serialization_lock();
    let Some(cc) = cc::find_compiler() else {
        eprintln!("SKIP: no host C compiler (tried cc, gcc, clang, cl and the vcvars fallback)");
        return;
    };
    let Some(gpu) = gpu::Gpu::open("cpu_c_oracle") else {
        return; // skip already declared
    };

    let cases = cases();
    let mut spirv = Vec::new();
    let mut sources = Vec::new();
    for c in &cases {
        let words = try_generate(&c.def, &c.key, &SpirV)
            .unwrap_or_else(|e| panic!("{}: the SPIR-V backend declined: {e:?}", c.name));
        let c_kernel = try_generate(&c.def, &c.key, &CpuC)
            .unwrap_or_else(|e| panic!("{}: the CPU-C backend declined: {e:?}", c.name));
        spirv.push((
            words.name.clone(),
            words.words().expect("SPIR-V words").to_vec(),
        ));
        sources.push((c_kernel.name.clone(), c_kernel.source.clone()));
    }

    let host = host_results(&cc, &cases, &sources);
    assert_eq!(host.len(), cases.len(), "the oracle printed every case");

    let mut compared = 0usize;
    let (mut bad, mut undefined) = (Vec::new(), Vec::new());
    for (i, c) in cases.iter().enumerate() {
        let (name, words) = &spirv[i];
        let inputs: Vec<&[u32]> = c.inputs.iter().map(Vec::as_slice).collect();
        let got = gpu.run(name, words, &inputs);
        assert_eq!(got.len(), host[i].len(), "{}: result length", c.name);
        for (j, (&g, &w)) in got.iter().zip(&host[i]).enumerate() {
            let row: Vec<u32> = c.inputs.iter().map(|col| col[j]).collect();
            let in_domain = c.domain.holds(&row);
            compared += usize::from(in_domain);
            if !agree(c.tol, c.dt, g, w) {
                let args: Vec<String> = c
                    .inputs
                    .iter()
                    .map(|col| format!("{:#010x}", col[j]))
                    .collect();
                let line = format!(
                    "{} ({:?}) inputs [{}]: device {g:#010x}, host {w:#010x}",
                    c.name,
                    c.tol,
                    args.join(", ")
                );
                if in_domain {
                    bad.push(line);
                } else {
                    undefined.push(line);
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {compared} cells disagree with unpopped-cpu-c:\n{}",
        bad.len(),
        bad.join("\n")
    );
    eprintln!(
        "cpu_c_oracle: {} cells outside the ops' specified domains disagree (reported, not failed):\n{}",
        undefined.len(),
        undefined.join("\n")
    );
    eprintln!(
        "cpu_c_oracle: {compared} in-domain device cells over {} kernels agree with unpopped-cpu-c",
        cases.len()
    );
}
