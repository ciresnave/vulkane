// SPDX-License-Identifier: MIT OR Apache-2.0
// Moved from Unpopped (crates/unpopped-conformance/tests/cell_token_is_not_a_computation.rs at 5500fd1726dce3ff9eba76c407c75ed656575357,
// same owner, MIT OR Apache-2.0): the Slang emitter is now this crate, so its conformance cells live beside it
// and run against the published `unpopped`. Only the `Slang` import path changed.
//! A cell token names **what was asked for**. It cannot name **what runs**.
//!
//! `structure_key(op_category, operands, target)` takes no op body. Two
//! different computations of the same category, over the same operand shapes,
//! on the same target therefore derive the **same token** — by construction,
//! not by accident. That is the design: the token is a *schedule cell*, and a
//! schedule is chosen from shapes and dtypes, not from arithmetic.
//!
//! # Why this is measured here rather than argued from the signature
//!
//! "The body is not a parameter, therefore the token collides" is a correct
//! reading of a function signature and a *wrong* way to establish a property
//! that four other things depend on. The signature could grow a body term; a
//! backend could fold the body into the plan; the token codec could change. So
//! the collision is pinned as behaviour, with the separation that makes it
//! survivable pinned beside it — on **every** reference emitter, because this
//! is a claim about the standard and not about one backend.
//!
//! # What depends on it
//!
//! - **Caching** is safe: KISS-Synth §6.7-0004 requires a hit to match
//!   `(structure_key, revision_hash)`, and the revision hash is over the emitted
//!   source. Same cell, different arithmetic ⇒ different source ⇒ miss. Pinned
//!   below as the source-differs half.
//! - **Dispatch is not.** [`DispatchTable`] is keyed by the token *alone*:
//!   `from_entries`/`merge`/`normalize` keep one row per token, and `merge`
//!   treats a differing `winner_entry` as a competing schedule **variant** to
//!   be evicted on margin — not as a co-resident route. So a table that ever
//!   held two *computations* in one cell would benchmark them against each
//!   other and route both to whichever measured faster.
//!
//! That last one is a real precondition on the table's producer — *all
//! candidates in a cell compute the same thing* — which the type cannot check
//! and, until this file, nothing stated. The collapse is pinned below so the
//! precondition is executable rather than a sentence someone has to find.

use unpopped::backend::Backend;
use unpopped::ir::{OpDef, input};
use unpopped::try_generate;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::{
    ArchSku, DispatchEntry, DispatchTable, ElementKind, Implementor, OpCategory, OperandDesc,
    Provenance, StructureKey, structure_key,
};
use vulkane_slang_emit::Slang;

fn cell() -> StructureKey {
    let d = OperandDesc::new(1, &[7], &[1], ElementKind::F32, 4);
    structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89)
}

/// Two ops that differ **only** in arithmetic.
///
/// They share a kernel name deliberately. Name them apart and the emitted
/// sources differ because the symbols differ, and the source-differs assertion
/// passes without the arithmetic ever reaching it — green, and blind to the one
/// thing it exists to measure. A first draft of this test did exactly that
/// against a stub backend that emits `/* stub: f */` for any body, and the
/// same-name discipline is what caught it.
fn sum_and_product() -> (OpDef, OpDef) {
    (
        OpDef::elementwise("f", 2, &[ElementKind::F32], input(0) + input(1)),
        OpDef::elementwise("f", 2, &[ElementKind::F32], input(0) * input(1)),
    )
}

#[test]
fn every_emitter_gives_two_computations_one_token_and_two_sources() {
    let backends: [&dyn Backend; 2] = [&CpuC, &Slang];
    let key = cell();
    let (sum, prod) = sum_and_product();
    let mut checked = 0;

    for b in backends {
        let a = try_generate(&sum, &key, b).expect("f32 add must lower on a reference emitter");
        let m = try_generate(&prod, &key, b).expect("f32 mul must lower on a reference emitter");

        let pa = a.provenance().expect("generate stamps provenance");
        let pm = m.provenance().expect("generate stamps provenance");

        assert_eq!(
            pa.structure_key,
            key.to_token(),
            "{}: the stamp must be the cell that was asked for",
            b.name()
        );
        assert_eq!(
            pa,
            pm,
            "{}: the whole stamp is request-shaped — if a body term ever reaches \
             it, every consumer keying on the token must be re-read",
            b.name()
        );

        assert_ne!(
            a.source,
            m.source,
            "{}: same name, same cell, different arithmetic must still emit \
             different source — if these matched, the source is not carrying the \
             body and the token-equality above would prove nothing",
            b.name()
        );
        checked += 1;
    }

    // Vacuity control for the token-equality above: the stamp is not a
    // constant. A *different* cell — same ops, one element instead of seven —
    // must stamp differently, or `pa == pm` would hold for any two kernels and
    // certify nothing.
    let other = {
        let d = OperandDesc::new(1, &[8], &[1], ElementKind::F32, 4);
        structure_key(OpCategory::BinaryElementwise, &[d, d, d], ArchSku::Sm89)
    };
    let elsewhere = try_generate(&sum, &other, &CpuC).expect("lowers");
    assert_ne!(
        elsewhere.provenance().expect("stamped").structure_key,
        key.to_token(),
        "a different cell must stamp a different token"
    );
    assert_eq!(checked, 2, "both reference emitters must be measured");
}

/// The consequence: a cell holds **one** route, so it must hold one computation.
///
/// Not a defect in [`DispatchTable`] — one winner per cell is the whole job.
/// It is a precondition on whoever fills the table, and it is invisible from
/// the type: nothing in a `DispatchEntry` names the computation, so two
/// computations in one cell are indistinguishable from two schedule variants of
/// one, which is exactly what `merge` is built to arbitrate.
#[test]
fn a_cell_holds_one_route_so_a_second_computation_evicts_the_first() {
    let token = cell().to_token();
    let row = |entry: &str| DispatchEntry {
        structure_key: token.clone(),
        winner: Implementor::Generated,
        winner_entry: Some(entry.to_string()),
        margin: 1.0,
        ranked: Vec::new(),
        provenance: Provenance::Seeded,
        measured_on: None,
    };

    let t = DispatchTable::from_entries(vec![row("k_sum"), row("k_prod")]);

    assert_eq!(
        t.entries.len(),
        1,
        "one token, one row — two computations cannot both be routed"
    );
    assert_eq!(
        t.entries[0].winner_entry.as_deref(),
        Some("k_prod"),
        "last-writer-wins, matching `merge`'s override semantics — so which \
         computation survives is an artifact of insertion order, which is why \
         the caller must never put two in one cell"
    );
}
