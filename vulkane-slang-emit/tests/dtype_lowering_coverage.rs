// SPDX-License-Identifier: MIT OR Apache-2.0
// Moved from Unpopped (crates/unpopped-conformance/tests/dtype_lowering_coverage.rs at 5500fd1726dce3ff9eba76c407c75ed656575357,
// same owner, MIT OR Apache-2.0): the Slang emitter is now this crate, so its conformance cells live beside it
// and run against the published `unpopped`. Only the `Slang` import path changed.
//! What this generator can actually **emit**, per dtype, per backend.
//!
//! # Why this is a test and not a line in a doc
//!
//! The deferred register carried these numbers in prose — "22/22 named, CpuC
//! lowers 9, Slang 5" — and by the time anyone read them they were wrong: the
//! measured figures are **8 and 4**. Not a large error, and that is the point.
//! A coverage claim decays silently, because nothing about adding or removing a
//! dtype arm forces the sentence describing it to change.
//!
//! So the claim lives here, as a table that has to be edited when the behaviour
//! moves. Adding a dtype to a backend fails this test until the table is
//! updated, which makes the update part of the change rather than a follow-up
//! nobody schedules.
//!
//! # Recognition is not lowering, and the gap is the useful number
//!
//! `unpopped-vocab` **recognizes** all 24 KISS §6.1 tokens and that is verified
//! against KISS's generated manifest. This file measures something different and
//! much smaller: for how many of those can this generator actually produce a
//! kernel. Conflating the two would let "we support 24 dtypes" mean two
//! incompatible things depending on who is asking — which is exactly the
//! distinction KISS draws between its recognition set and its usable set.
//!
//! A decline here is **conformant**, not a bug: a party that cannot lower a
//! dtype must still *name* it, so it can decline as a known dtype rather than as
//! an unknown token. This file pins which is which.

use unpopped::ir::{OpDef, input};
use unpopped::try_generate;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};
use vulkane_slang_emit::Slang;

/// What a backend does with a dtype, and **why**.
///
/// The distinctions among the non-lowering states are the point of this enum. A
/// single `false` column puts "nobody has written this yet", "this needs
/// something else first", and "this is decided and correct" in one cell — and
/// they call for completely different responses. Collapsed, the worklist
/// silently grows entries that will never be done, settled decisions get
/// re-litigated by whoever reads the table as a TODO list, and work that is
/// merely *ordered* looks impossible.
///
/// The `Blocked` variant exists because of a mistake this file made: Slang's
/// narrow integers were labelled `ByDesign` — permanently declined — on a
/// misreading of "depends on target + capabilities" as "cannot". They are
/// ordinary work behind one missing mechanism.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Status {
    /// Lowers today.
    Lowers,
    /// Not implemented yet — a genuine worklist entry, nothing in the way.
    NotYet,
    /// Worklist, but a **prerequisite** has to land first. Still work, and the
    /// named blocker is what makes it schedulable in the right order.
    Blocked(&'static str),
    /// **Deliberately** declined, permanently, for the recorded reason. A
    /// typed decline here is conformant behaviour, not a gap.
    ///
    /// **Its only members are the two MX scales, and it took three wrong
    /// attempts to get one right.** Slang's narrow integers were labelled
    /// `ByDesign` on a misreading of "depends on target + capabilities" as
    /// "cannot"; `u32` and `u64` on a circular reason ("never a compute
    /// operand") that described the implementation rather than a decision. All
    /// three were really `Blocked`, and all three now lower or are scheduled.
    ///
    /// `f8e8m0` is different in kind: KISS-CLASSIFY §6.1-0013 makes it a
    /// sibling-operand **scale**, never an element value dtype (`f8e6m2` is the
    /// same by name but is RESERVED, so it is not in this table at all). That is a
    /// statement about what the dtype IS, not about what we have built.
    ///
    /// A `ByDesign` entry should stay rare enough to be suspicious. The test
    /// below refuses one whose reason mentions deferral, which is the shape the
    /// three wrong ones had.
    ByDesign(&'static str),
}
use Status::{Blocked, ByDesign, Lowers, NotYet};

/// Slang's own conformance docs: *"Only `int`/`int32_t` and `uint`/`uint32_t`
/// are universally supported; the others depend on target + capabilities."*
///
/// **Read carefully, that says Slang CAN spell these — on a target whose
/// capabilities allow it.** So this is not a limitation of Slang, and labelling
/// it one (as an earlier version of this file did) blames the target for a gap
/// that is ours.
///
/// There were **two** blockers here and only one of them was ours. That one is
/// now fixed, so this reason is rewritten rather than left implying the old
/// state.
///
/// **Gone — the API blocker.** `supports_dtype` used to take only a dtype, so a
/// backend could answer "always" or "never" and nothing between. Faced with a
/// conditionally-available type the only *sound* answer was "never", since
/// claiming support unconditionally emits `int8_t` for a target that cannot
/// compile it — the "decline, do not fall through" rule the whole backend
/// contract rests on. It now takes a `TargetId`. (The same closed-enum root
/// cause also excluded a `vulkan:` vector from the cross-project byte-match;
/// that leg now reports 20/20 with zero capability exclusions.)
///
/// **Remaining — and it is NOT "no capability data". Asked the owner.**
///
/// This used to say the blocker was a missing machine-readable manifest (KISS
/// #171). Vulkane, who owns the `vulkan:` vocabulary under §6.8-0004, answered
/// directly and the real state is **asymmetric**:
///
/// * `i8`/`u8` — **answerable today.** The `<arith>` field carries `i8`, naming
///   `shaderInt8`. Declining these is over-refusal, and no manifest is needed.
/// * `i16`/`u16` — **the vocabulary cannot express it.** The published `<arith>`
///   names are exactly `dot8`, `f16`, `i8`, `st16`, `st8`; `shaderInt16` is not
///   among them. Vulkane records it as their gap; naming it later bumps the
///   vocabulary version rather than being additive.
///
/// And the trap they flagged hardest, recorded because it is the inference a
/// future reader would think obvious: **never derive 16-bit arithmetic from
/// `st16`.** That is `storageBuffer16BitAccess` — *storage*, not compute. A
/// device may take 16-bit data in a buffer and do the math in `f32`, so reading
/// it as permission to emit 16-bit integer arithmetic is a silently wrong
/// lowering on conformant hardware.
///
/// So the four stay `Blocked`, but for two different reasons, and the 8-bit
/// pair is unblocked work rather than a wait. Splitting them needs a third
/// `supports_dtype` state — supported / unsupported / *not expressible in this
/// vocabulary version* — which is a peer-implemented-trait change and is with
/// Eric.
const SLANG_NARROW: &str = "i8/u8/i16/u16 are GATED, not refused: Slang spells them      (int8_t/uint8_t/int16_t/uint16_t) when the target advertises the matching arithmetic      capability — `i8` (shaderInt8) for the 8-bit pair, `i16` (shaderInt16) for the 16-bit      pair. Both tokens are signedness-agnostic, so u8 gates on i8 and u16 on i16. i16 became      expressible on 2026-09-02 with kiss-vulkan-vocab 0.4.0 / vulkane 0.14.0 (capability      vocabulary v5); it was an UNSHIPPED vocabulary, never a missing one. This row probes      cuda:sm89, which carries no <arith> field at all, so the refusal here is correct";

/// `f8e8m0` is an OCP Microscaling **scale** dtype (and `f8e6m2` its reserved
/// sibling, which `the_reserved_dtypes_never_lower` covers), and
/// KISS-CLASSIFY §6.1-0013 is explicit about what that means: it is "the
/// per-block shared scale of an MX-encoded value operand, carried as a **sibling
/// operand**, **never an element value dtype**." §6.2-0002 confirms it from the
/// other side — its float special-value pinning lists `f8e4m3fn`/`f8e5m2` and
/// their `fnuz` siblings, and omits these two.
///
/// So lowering them as a compute dtype would be a category error rather than
/// progress: a kernel does not compute *in* a scale, it uses one to dequantize
/// the block that scale belongs to. That work is the quant/scale-sibling model,
/// on a different axis entirely.
///
/// These are the first genuine `ByDesign` entries. The variant's doc says such
/// an entry should be rare enough to be suspicious — so: the claim is not "we
/// haven't got to it", it is that the §6.1 row exists to be *named and carried*,
/// not computed with, and a future version should not quietly change that.
/// `Bool` is stored as U8 (FKC §5), so Slang spells it `uint8_t` and gates it on
/// the same 8-bit-arithmetic capability as `i8`/`u8`. This row probes
/// `cuda:sm89`, which carries no `<arith>` field, so the refusal is correct here
/// rather than unwritten — which is why it moved from `NotYet` to `Blocked`.
const SLANG_BOOL: &str = "bool is stored as U8 (FKC §5) and gates on 8-bit arithmetic      (`i8` in the vulkan <arith> field). Refused for cuda:sm89, which advertises no      <arith> field at all. NOTE: this is a deliberate OVER-refusal — a bool kernel may      only need `st8` storage if its compiler promotes to 32-bit, and supports_dtype is      handed no plan so it cannot tell movement from compute";

const MX_SCALE: &str = "MX shared-exponent scale (KISS-CLASSIFY 6.1-0013): a sibling operand \
     that scales a block, never an element value dtype a kernel computes in";

/// Every §6.1 dtype, with what each backend does with a plain elementwise `Add`.
///
/// The two RESERVED dtypes are absent entirely rather than listed as declines:
/// `f8e4m3fnuz` / `f8e5m2fnuz` have no computation semantics at this schema
/// version and lowering them would be a conformance violation. That is a third
/// kind of fact again, and it is asserted separately below.
const COVERAGE: &[(&str, ElementKind, Status, Status)] = &[
    // name          dtype                      CpuC   Slang
    // CpuC since 0.15.0: `unsigned short` carrier + `cfamily::half_helpers`.
    ("f16", ElementKind::F16, Lowers, NotYet),
    ("bf16", ElementKind::Bf16, Lowers, NotYet),
    ("f32", ElementKind::F32, Lowers, Lowers),
    ("f64", ElementKind::F64, Lowers, Lowers),
    ("i8", ElementKind::I8, Lowers, Blocked(SLANG_NARROW)),
    ("i16", ElementKind::I16, Lowers, Blocked(SLANG_NARROW)),
    ("u8", ElementKind::U8, Lowers, Blocked(SLANG_NARROW)),
    ("u16", ElementKind::U16, Lowers, Blocked(SLANG_NARROW)),
    ("i32", ElementKind::I32, Lowers, Lowers),
    ("i64", ElementKind::I64, Lowers, Lowers),
    ("u32", ElementKind::U32, Lowers, Lowers),
    ("u64", ElementKind::U64, Lowers, Lowers),
    ("bool", ElementKind::Bool, Lowers, Blocked(SLANG_BOOL)),
    ("f8e4m3fn", ElementKind::Fp8E4M3FN, Lowers, NotYet),
    ("f8e5m2", ElementKind::Fp8E5M2, Lowers, NotYet),
    (
        "f8e8m0",
        ElementKind::F8E8M0,
        ByDesign(MX_SCALE),
        ByDesign(MX_SCALE),
    ),
    ("i4", ElementKind::I4, Lowers, NotYet),
    ("u4", ElementKind::U4, Lowers, NotYet),
    ("b1", ElementKind::B1, Lowers, NotYet),
    ("c64", ElementKind::Complex64, Lowers, NotYet),
    ("c128", ElementKind::Complex128, Lowers, NotYet),
];

/// An extent and alignment that select the **scalar** schedule.
///
/// Deliberately 7 elements at 4-byte alignment: both backends lower only
/// `Schedule::Scalar`, and a divisible extent at a wide alignment elects a
/// vectorized schedule instead. Getting this wrong does not fail loudly — it
/// reports `UnsupportedSchedule` for *every* dtype and reads as "nothing is
/// supported". It cost me a 0/22 measurement before I noticed `f32` was
/// declining for a reason that had nothing to do with dtypes, which is why the
/// schedule is pinned by its own control below.
fn scalar_shape(dt: ElementKind) -> OperandDesc {
    OperandDesc::new(1, &[7], &[1], dt, 4)
}

fn lowers(dt: ElementKind, slang: bool) -> bool {
    // The probe op has to be one the dtype ADMITS, or the table measures the
    // admissibility gate instead of the backend. `bool` is the case that forced
    // this: its ops normalize to 0/1 (§6.1), so `Add` is refused — `true + true`
    // is 2, not a value of the dtype — and the meaningful surface is the logical
    // ops. Probing every dtype with `Add` reported `bool` as unlowerable when
    // what it actually cannot do is arithmetic.
    let op = if dt == ElementKind::Bool {
        OpDef::elementwise(
            "and",
            2,
            &[dt],
            input(0).binary(unpopped::ir::BinaryOp::LogicalAnd, input(1)),
        )
    } else {
        OpDef::elementwise("add", 2, &[dt], input(0) + input(1))
    };
    let d = scalar_shape(dt);
    let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);
    // A plan-gate rejection is a panic on this path; a backend decline is an
    // `Err`. Both mean "does not lower", and the distinction between them is a
    // separate concern from coverage.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if slang {
            try_generate(&op, &key, &Slang).is_ok()
        } else {
            try_generate(&op, &key, &CpuC).is_ok()
        }
    }))
    .unwrap_or(false)
}

#[test]
fn the_coverage_table_matches_what_the_backends_actually_do() {
    let mut wrong = Vec::new();
    let (mut c_n, mut s_n) = (0, 0);
    for &(name, dt, want_c, want_s) in COVERAGE {
        for (backend, slang, want) in [("CpuC", false, want_c), ("Slang", true, want_s)] {
            let got = lowers(dt, slang);
            if got {
                if slang { s_n += 1 } else { c_n += 1 }
            }
            if got != (want == Lowers) {
                wrong.push(format!(
                    "{name}: {backend} table says {want:?}, actually lowers={got}"
                ));
            }
        }
    }
    println!("\nmeasured lowering coverage — CpuC {c_n}/21, Slang {s_n}/21");
    for w in &wrong {
        println!("   {w}");
    }
    assert!(
        wrong.is_empty(),
        "{} entr(ies) stale. If you ADDED a dtype this is the reminder to \
         update the table; if you removed one, say why here.",
        wrong.len()
    );
}

/// Every deliberate decline carries a reason, and the reasons are real.
///
/// `ByDesign` is the entry that can rot worst: it looks settled, so nobody
/// re-checks it, and an empty or vague justification is indistinguishable from
/// a `NotYet` somebody wanted off the worklist. Each one is a claim about a
/// target or about this generator's design, and it should read as one.
#[test]
fn every_deliberate_decline_states_why() {
    let mut n = 0;
    for &(name, _, c, s) in COVERAGE {
        for st in [c, s] {
            if let ByDesign(why) | Blocked(why) = st {
                // Substance, not format. An earlier version of this also
                // demanded a colon, which failed a perfectly good reason for
                // being punctuated differently — the assertion was encoding my
                // habits rather than the property.
                assert!(
                    why.len() > 20,
                    "{name}: a ByDesign decline needs a real reason, got {why:?}"
                );
                assert!(
                    !why.to_lowercase().contains("todo") && !why.to_lowercase().contains("later"),
                    "{name}: {why:?} describes deferral, not a decision — that is `NotYet`"
                );
                n += 1;
            }
        }
    }
    assert!(
        n > 0,
        "no deliberate declines — has the enum stopped being used?"
    );
    println!(
        "
{n} deliberate decline(s), each with a stated reason"
    );
}

/// The scalar-schedule control.
///
/// Every `false` above is only meaningful if the probe reaches the dtype gate at
/// all. A shape that elects a vectorized schedule declines for *every* dtype and
/// would make this whole file read as "nothing lowers" while asserting nothing —
/// the vacuous-green shape. `f32` lowering proves the probe gets through.
#[test]
fn the_probe_reaches_the_dtype_gate() {
    assert!(
        lowers(ElementKind::F32, false),
        "control: f32 must lower on CpuC, or the shape is selecting a schedule \
         no backend supports and every `false` above is measuring the wrong thing"
    );
    assert!(
        lowers(ElementKind::F32, true),
        "control: f32 must lower on Slang"
    );
}

/// The two RESERVED dtypes must never lower, at any schema version.
///
/// Separate from the table because it is a different kind of fact: the table
/// records what is *not done yet* and is expected to fill in over time, whereas
/// these are forbidden by KISS-CLASSIFY-6.1-0001 and filling them in would be a
/// conformance violation. A `false` that must stay `false` does not belong in a
/// column of `false`s that are all invitations.
#[test]
fn the_reserved_dtypes_never_lower() {
    for (name, dt) in [
        ("f8e4m3fnuz", ElementKind::Fp8E4M3FNUZ),
        ("f8e5m2fnuz", ElementKind::Fp8E5M2FNUZ),
        ("f8e6m2", ElementKind::F8E6M2),
    ] {
        assert!(dt.is_reserved(), "{name} must be marked reserved");
        assert!(
            !lowers(dt, false) && !lowers(dt, true),
            "{name} is RESERVED — no computation semantics at this schema version. \
             Lowering it is a conformance violation, not progress."
        );
    }
}

/// Coverage is a strict subset of recognition, and naming the gap is the point.
///
/// If these ever converge, this crate lowers everything it can name and the
/// assertion should be inverted rather than deleted.
#[test]
fn recognition_exceeds_lowering_and_the_gap_is_named() {
    let recognized = 24; // the closed §6.1 set, verified in unpopped-vocab
    let usable = 21; // recognized minus the three reserved (KISS-CLASSIFY-6.1-0001)
    assert_eq!(
        COVERAGE.len(),
        usable,
        "the table covers every usable dtype"
    );

    let c = COVERAGE.iter().filter(|e| e.2 == Lowers).count();
    let s = COVERAGE.iter().filter(|e| e.3 == Lowers).count();
    assert!(
        c < usable && s < usable,
        "lowering now covers every usable dtype — good news; invert this test"
    );

    // The worklist is `NotYet` only. A `ByDesign` decline is a conclusion, and
    // counting it as outstanding work would keep it on the list forever.
    // `Blocked` is still work — it is scheduled behind a prerequisite, not
    // excluded. Only `ByDesign` leaves the worklist.
    let todo: Vec<&str> = COVERAGE
        .iter()
        .filter(|e| matches!(e.2, NotYet | Blocked(_)))
        .map(|e| e.0)
        .collect();
    let settled: Vec<&str> = COVERAGE
        .iter()
        .filter(|e| matches!(e.2, ByDesign(_)))
        .map(|e| e.0)
        .collect();

    println!(
        "\nrecognized {recognized} · usable {usable} · CpuC lowers {c} · Slang lowers {s}\n\
         CpuC worklist ({}): {todo:?}\n\
         CpuC settled declines ({}): {settled:?}",
        todo.len(),
        settled.len()
    );
}

/// **Every backend either emits a real store or returns a typed decline — never a
/// panic, never an empty body.**
///
/// # The axis restriction is GONE, and the sequence is worth keeping
///
/// This was `..._and_never_panics`, and **Baracuda caught that the name claimed
/// more than the test checked** while dev-depping these emitters. The loop
/// iterates `ElementKind::ALL` with a **fixed op**, so "never panics" was measured
/// on the **dtype axis only** — on the **op** axis both emitters panicked
/// deliberately, a known `KISS-EMIT-6.8-0004` divergence.
///
/// Rather than rename around that, the gap was **pinned as an inequality**: a test
/// asserting the op-axis panic *existed*, so that the day it became a typed
/// decline the test would fail and force a deliberate deletion.
///
/// **It fired.** `Lowering` now returns `Result<Spelling, LowerError>`, the
/// emitters have **zero** `panic!` between them, and the pin failed with its own
/// instruction to delete itself and widen this note. So the axis qualifier is
/// removed here because the restriction no longer exists — not because anyone
/// remembered to check.
///
/// That is the whole argument for pinning a gap rather than documenting it: a gap
/// nothing asserts is a gap nobody notices closing, and this one would otherwise
/// have left a stale axis-note describing a restriction that had been lifted.
///
/// # What this recovers, and what it deliberately does not
///
/// `c584818` moved the emitters out of the core, which stranded three of
/// Baracuda's cross-backend tests: they re-emit through `CpuC`/`Slang`, and with
/// those crates unpublished a downstream consumer cannot dev-dep them. Baracuda
/// carries that as a **documented, tracked deferral** and restores the tests when
/// the emitters publish.
///
/// **The deferred gap is "three-way agreement", not "three tests"** — and the
/// two-way share has a home here, because this crate already dev-deps both
/// emitters for exactly this reason. So this closes the `CpuC` x `Slang` part
/// with no publish and no irreversible act.
///
/// **It does NOT close the three-way part.** Agreement *including* Cuda is
/// unreachable from here: the dependency points one way and this crate cannot see
/// `baracuda-cuda-emit`. That share stays open until the emitters publish, and is
/// stated so nobody later reads this test as having closed the whole thing --
/// which would be the overclaiming failure this repo keeps finding, in a poor
/// place to introduce a fresh one.
///
/// # Why it is not folded into the coverage table above
///
/// The table **cannot** assert this property: [`lowers`] wraps its probe in
/// `catch_unwind(..).unwrap_or(false)`, so a **panic and a decline are the same
/// answer there** -- deliberately, since for a coverage question both mean "does
/// not lower". That conflation is what this test refuses. A backend that panicked
/// on every dtype would show as uniformly unsupported above, and green.
///
/// Baracuda's version of this shape is what surfaced Slang's intentional
/// `Copysign`/`Nextafter` decline, so it has a track record of catching something
/// real rather than being hypothetical.
///
/// # The store spelling is per-language AND per-dtype, which this test learned
///
/// There is no neutral spelling to assert -- a store's target-language surface is
/// emitter-supplied -- so the accepted spellings travel with the backend.
///
/// The first version looked only for `out[i] =` and **failed on `i4`**, correctly,
/// and not because the emitter was wrong. A **sub-byte store is a statement, not
/// an assignment**: several elements share a byte, so there is no per-element
/// lvalue and `CpuC` emits `unpopped_i4_store(out, i, ..)`. That is the exact
/// distinction `cpu_c`'s own comment warns about -- writing `out[i] =` for a
/// sub-byte dtype "compiles, it runs, and it scrambles every lane past the first."
///
/// So a backend has a *set* of store spellings, not one. Asserting a single form
/// would have made this test wrong for four dtypes while looking rigorous.
#[test]
fn every_backend_emits_a_real_store_or_declines_typed_and_never_panics() {
    // Same probe-op choice as `lowers`: `bool` admits the logical ops, not `Add`
    // (its ops normalize to 0/1, so `true + true` is not a value of the dtype).
    fn probe(dt: ElementKind) -> OpDef {
        if dt == ElementKind::Bool {
            OpDef::elementwise(
                "and",
                2,
                &[dt],
                input(0).binary(unpopped::ir::BinaryOp::LogicalAnd, input(1)),
            )
        } else {
            OpDef::elementwise("add", 2, &[dt], input(0) + input(1))
        }
    }

    let mut emitted = 0usize;
    let mut declined = 0usize;

    for dt in ElementKind::ALL {
        let op = probe(dt);
        let d = scalar_shape(dt);
        let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);

        // `CpuC` has two accepted spellings: the plain assignment, and the packed
        // statement used where a sub-byte dtype has no per-element lvalue.
        for (name, slang, stores) in [
            ("CpuC", false, &["out[i] =", "_store(out, i,"][..]),
            ("Slang", true, &["output[i] ="][..]),
        ] {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if slang {
                    try_generate(&op, &key, &Slang)
                } else {
                    try_generate(&op, &key, &CpuC)
                }
            }));

            match outcome {
                Err(_) => panic!(
                    "{name} PANICKED on {dt:?}. A backend that cannot serve a cell must \
                     return a typed decline, not unwind -- and the coverage table above \
                     cannot catch this, because it maps a panic and a decline to the same \
                     answer."
                ),
                Ok(Ok(kernel)) => {
                    assert!(
                        !kernel.source.trim().is_empty(),
                        "{name} returned Ok on {dt:?} with an EMPTY body -- a kernel that \
                         emits nothing is worse than a decline, because it looks like success"
                    );
                    assert!(
                        stores.iter().any(|m| kernel.source.contains(m)),
                        "{name} returned Ok on {dt:?} but the body has no store (none of \
                         {stores:?}). An emitted kernel that never writes its output is the \
                         all-zero / never-wrote failure class, and it passes every \
                         byte-identity check:\n{}",
                        kernel.source
                    );
                    emitted += 1;
                }
                // A typed decline is a correct answer; WHICH dtypes decline is the
                // coverage table's business rather than this test's.
                Ok(Err(_)) => declined += 1,
            }
        }
    }

    // Positive controls: without these the test passes vacuously if one arm never
    // executes.
    assert!(
        emitted > 0,
        "no backend emitted for any dtype -- the store assertion never ran, so this test \
         proved nothing"
    );
    assert!(
        declined > 0,
        "every backend emitted for every dtype -- the decline path never ran, so this test \
         is not exercising the decline-vs-panic distinction it exists for"
    );
    println!("emitted with a real store: {emitted}   typed declines: {declined}");
}

/// **An op-level refusal is now a MATCHABLE value naming the op it refused.**
///
/// This is what the `Lowering`-returns-`Result` change bought, and it is worth a
/// test of its own because the previous shape — a `panic!` — was satisfiable by
/// any unrelated crash.
///
/// # The bar this had to clear
///
/// Baracuda's cross-backend suite asserts *"Slang declines `Copysign`"* by
/// inspecting a **panic payload**, precisely so an unrelated panic could not
/// satisfy it. The transition to a typed value had to make that discrimination
/// **stronger, not weaker** — otherwise their test would have degraded at exactly
/// the moment this API improved.
///
/// `matches!(d, Decline::UnsupportedOp { op: DeclinedOp::Binary(Copysign), .. })` is
/// structurally unsatisfiable by an unrelated failure, where a payload substring
/// is not. That is the bar cleared.
///
/// # Scope limit, stated because it changes what this test can prove
///
/// **That matchable form lives at the SEAM, and the seam is not reachable from
/// here.** An emitter builds its `Lowering` internally, so from outside the crate
/// the observable property is the weaker one: an op-level refusal is a **typed
/// `Err`** rather than a panic. That is what this test asserts, and it deliberately
/// checks the *variant* rather than only the message.
///
/// The stronger property is real but is exercised by a **backend author**, who
/// constructs the seam and matches on `Decline` directly — which is precisely
/// Baracuda's position. Asserting more than this from here would be claiming a
/// reach the test does not have.
///
/// # Why the decline is a success value
///
/// Note the shape: `Ok(Spelling::Declined(..))`, not `Err(..)`. Owed to Vulkane —
/// with a single error type, `let s = spell(..)?;` compiles, reads naturally, and
/// **silently converts a decline into a propagated failure**. Making a decline a
/// success value means `?` propagates only real failures and a decline *forces* a
/// match. The lazy path is the correct one, which matters because the lazy path is
/// the one that ships.
#[test]
fn an_op_level_refusal_is_a_typed_decline_not_a_panic() {
    use unpopped::backend::LowerError;
    use unpopped::ir::BinaryOp;

    let dt = ElementKind::F32;
    let op = OpDef::elementwise(
        "copysign",
        2,
        &[dt],
        input(0).binary(BinaryOp::Copysign, input(1)),
    );
    let d = scalar_shape(dt);
    let key = structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89);

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        try_generate(&op, &key, &Slang)
    }));

    let result = outcome.expect(
        "an op-level refusal must NOT unwind. This is KISS-EMIT-6.8-0004: an emitter          must not panic on any input, and an op it does not spell IS input.",
    );

    match result {
        Err(LowerError::UnsupportedOp { detail }) => {
            assert!(
                detail.contains("Copysign"),
                "the decline must name what it refused; got: {detail}"
            );
        }
        Err(other) => panic!(
            "expected UnsupportedOp naming Copysign, got {other:?}. The variant is the              matchable part -- a consumer branching on WHY cannot use the message."
        ),
        Ok(k) => panic!(
            "Slang emitted for Copysign, which it declines by design:
{}",
            k.source
        ),
    }
}
