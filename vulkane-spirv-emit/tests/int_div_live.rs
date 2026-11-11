// SPDX-License-Identifier: MIT OR Apache-2.0
//! The emitted integer `Div` and `Rem` on `i32` and `u32` run on a live device and agree with
//! Unpopped's total rule (`docs/int-div-shift-semantics.md`, rules 1 and 2, Unpopped ROADMAP E16):
//! a zero divisor gives the dividend for `/` and `0` for `%`, and signed `MIN / -1` gives `MIN`
//! for `/` and `0` for `%`. SPIR-V leaves those inputs undefined (3.3.13), so without the emitted
//! guard a device may return anything for them.
//!
//! `Rem` is not an operation: it is `Select(b == 0, 0, a - (a / b) * b)` over the total `Div`,
//! built here from the same IR an Unpopped caller writes. The vectors are the document's:
//! divisors `0, 1, -1, MIN` and dividends `0, 1, -1, MIN, MAX`, plus a few mixed values.
//!
//! A device that happens not to trap or misbehave on the unguarded forms cannot show a missing
//! guard here (the same limit as `int_shift_live.rs`); the unit tests in `src/tests.rs` pin the
//! guard's structure, and this test guards the device path.
//!
//! Without a device the test follows `VULKANE_REQUIRE_DEVICE` like every other live test.

#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(dead_code, unreachable_pub, missing_docs)]
mod common;
#[path = "support/gpu.rs"]
#[allow(dead_code)]
mod gpu;

use unpopped::ir::{BinaryOp, Expr, OpDef, input, konst_int};
use unpopped::try_generate;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};
use vulkane_spirv_emit::SpirV;

/// Dividends: zero, one, minus one, `MIN`, `MAX`, and mixed patterns.
const DIVIDENDS: [u32; 9] = [
    0,
    1,
    0xFFFF_FFFF,
    0x8000_0000,
    0x7FFF_FFFF,
    7,
    0xFFFF_FFF9, // -7
    0xDEAD_BEEF,
    0x1234_5678,
];
/// Divisors: zero, one, minus one, `MIN`, `MAX`, and small mixed values.
const DIVISORS: [u32; 8] = [
    0,
    1,
    0xFFFF_FFFF,
    0x8000_0000,
    0x7FFF_FFFF,
    2,
    3,
    0xFFFF_FFFD, // -3
];

/// The total rule's quotient.
fn want_div(signed: bool, a: u32, b: u32) -> u32 {
    if b == 0 {
        return a;
    }
    if signed {
        (a as i32).wrapping_div(b as i32) as u32 // MIN / -1 wraps to MIN
    } else {
        a / b
    }
}

/// The total rule's remainder: `0` for a zero divisor and for `MIN / -1`.
fn want_rem(signed: bool, a: u32, b: u32) -> u32 {
    if b == 0 {
        return 0;
    }
    if signed {
        (a as i32).wrapping_rem(b as i32) as u32
    } else {
        a % b
    }
}

/// `a - (a / b) * b`, guarded to `0` for a zero divisor: Unpopped's integer `Rem` composition.
fn rem_expr(dt: ElementKind) -> Expr {
    let (a, b) = (input(0), input(1));
    let composed = a.clone() - (a / b.clone()) * b.clone();
    let is_zero = b.binary(BinaryOp::CmpEq, konst_int(0, dt));
    is_zero.select(konst_int(0, dt), composed)
}

fn kernel(name: &str, dt: ElementKind, body: Expr) -> (String, Vec<u32>) {
    // A small `align` defeats vectorization, so the plan is `Schedule::Scalar`.
    let operand = OperandDesc::new(1, &[1 << 20], &[1], dt, 4);
    let key = structure_key(
        OpCategory::BinaryElementwise,
        &[operand, operand, operand],
        ArchSku::Sm89,
    );
    let def = OpDef::elementwise(name, 2, &[dt], body);
    let k = try_generate(&def, &key, &SpirV).expect("the SPIR-V backend spells integer Div/Rem");
    let words = k.words().expect("the backend supplies words").to_vec();
    (k.name.clone(), words)
}

/// The host's answer: `(signed, a, b) -> cell`.
type Want = fn(bool, u32, u32) -> u32;

#[test]
fn emitted_integer_div_and_rem_match_the_unpopped_rule_on_a_device() {
    common::require_serialization_lock();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for &x in &DIVIDENDS {
        for &y in &DIVISORS {
            a.push(x);
            b.push(y);
        }
    }
    let Some(gpu) = gpu::Gpu::open("int_div") else {
        return; // skip already declared
    };
    let mut checked = 0usize;
    for (dt, signed) in [(ElementKind::I32, true), (ElementKind::U32, false)] {
        let cases: [(&str, Expr, Want); 2] = [
            ("div", input(0) / input(1), want_div),
            ("rem", rem_expr(dt), want_rem),
        ];
        for (label, body, want) in cases {
            let (name, words) = kernel(label, dt, body);
            let got = gpu.run(&name, &words, &[&a, &b]);
            for (i, g) in got.iter().enumerate() {
                let w = want(signed, a[i], b[i]);
                assert_eq!(
                    *g, w,
                    "{label} {dt:?}: {:#010x} over {:#010x} gave {g:#010x}, want {w:#010x}",
                    a[i], b[i]
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 4 * DIVIDENDS.len() * DIVISORS.len());
    eprintln!("int_div: {checked} device results agree with the Unpopped rule");
}
