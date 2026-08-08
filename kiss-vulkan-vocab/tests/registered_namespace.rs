//! The emitted vocabulary must match the **registered** `vulkan:` namespace.
//!
//! Vulkane owns the content of KISS-Classify §6.8's `vulkan:` namespace, and
//! that content is published as a document in the KISS repository —
//! `spec/namespaces/vulkan.md`. This crate is the implementation of it. Until
//! now, nothing checked the two against each other: the document listed a set
//! of component-type spellings, this crate emitted a set, and the claim that
//! they were the same set was maintained entirely by hand.
//!
//! That gap is not hypothetical, and the `s` → `i` rename below is the case
//! that proved it. During the KISS `sk4` schema event, vulkane was asked to
//! rename these spellings as a crate-side change — but they are governed by the
//! namespace document, not by §6.1's `structure_key` dtype set, which is a
//! *different vocabulary on a different axis*. Renaming the crate alone would
//! have emitted `i32` while the published namespace still said `s32`.
//!
//! I simulated that rename to see what the existing suite would say. **20 of
//! its 21 tests passed.** Round-tripping and canonical spelling both stay true
//! under a consistent rename — a round-trip test cannot detect that the whole
//! vocabulary moved, because it only checks that the crate agrees with itself.
//! The single failure was `rejects_unsorted_or_duplicated_coop_shapes`, which
//! broke *incidentally*, because `i32` and `s32` sort differently. That is the
//! worst kind of coverage: it fails, but it names the wrong cause. A maintainer
//! reading "unsorted coop shapes" would correct the sort expectation and move
//! on, never learning the vocabulary had drifted from its own published spec.
//!
//! The rename has since happened *properly*: the namespace document was
//! amended first (vocabulary version 2, `i`-prefixed signed integers), and the
//! crate followed. The constant below moved in the same commit as the
//! `ComponentType` arms, which is the workflow this file exists to enforce.
//!
//! **Known limit.** This pins the crate against a hand-transcribed copy of the
//! document, so it catches the *crate* drifting. It cannot notice the document
//! moving — that direction still depends on a human reading both. It is a
//! ratchet, not a proof.
//!
//! So this file pins the vocabulary itself. If the registered namespace is
//! amended, update `REGISTERED_COMPONENT_TYPES` **in the same change** as the
//! `ComponentType` arms — and if these tests fail, the question to answer is
//! "was the namespace document amended?", not "how do I make this pass?".

use kiss_vulkan_vocab::*;

/// Verbatim from `spec/namespaces/vulkan.md` (KISS, `origin/main`) at
/// **vocabulary version 2**, the paragraph reading:
///
/// > and each component type is one of `f16`, `f32`, `f64`, `bf16`, `i8`,
/// > `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, or `x<n>` for a
/// > `VkComponentTypeKHR` [this vocabulary does not name]
///
/// `x<n>` is exercised separately, since it is a pattern rather than a literal.
const REGISTERED_COMPONENT_TYPES: &[&str] = &[
    "f16", "f32", "f64", "bf16", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
];

/// Every named `ComponentType`, paired with the spelling the registered
/// namespace requires. Written out rather than derived, so that renaming a
/// variant's spelling cannot silently rewrite the expectation too.
const NAMED_COMPONENT_TYPES: &[(ComponentType, &str)] = &[
    (ComponentType::F16, "f16"),
    (ComponentType::F32, "f32"),
    (ComponentType::F64, "f64"),
    (ComponentType::BF16, "bf16"),
    (ComponentType::S8, "i8"),
    (ComponentType::S16, "i16"),
    (ComponentType::S32, "i32"),
    (ComponentType::S64, "i64"),
    (ComponentType::U8, "u8"),
    (ComponentType::U16, "u16"),
    (ComponentType::U32, "u32"),
    (ComponentType::U64, "u64"),
];

/// What to do when one of these fails. Shared, because the answer is the same
/// for every assertion here and repeating it per-message buries the part that
/// differs.
const ON_FAILURE: &str = "\n\nIf a spelling changed deliberately, the registered namespace document \
     (KISS `spec/namespaces/vulkan.md`) must be amended in the same change, and \
     REGISTERED_COMPONENT_TYPES updated with it. If it did not change \
     deliberately, revert it. What must not happen is editing this list alone: \
     that silences the guard instead of recording the decision.";

/// The `cm-` section of a token: everything after the last `.cm-`.
///
/// Deliberately parsed out of the emitted string rather than read from a
/// structured accessor. The property under test is what this crate *emits*,
/// compared against a document maintained elsewhere — routing that through the
/// crate's own API would only prove the crate agrees with itself, which is the
/// exact blind spot described at the top of this file.
fn cm_section(token: &str) -> &str {
    token
        .rsplit_once(".cm-")
        .map(|(_, section)| section)
        .expect("every emitted token carries a `.cm-` section")
}

/// Build a target whose cooperative-matrix section is a single shape using
/// `c` in all four operand positions, so the emitted token carries that
/// component type's spelling four times and nothing else varies.
fn token_for(c: ComponentType) -> String {
    VulkanTarget {
        subgroup: Subgroup::Fixed(32),
        ops: OpClasses::NONE,
        arith: Arith::NONE,
        coop: CoopMatrix::from_shapes(vec![CoopShape {
            m: 16,
            n: 16,
            k: 16,
            a: c,
            b: c,
            c,
            result: c,
            saturating: false,
        }]),
    }
    .to_token()
}

#[test]
fn every_component_type_spells_a_registered_token() {
    for &(component, expected) in NAMED_COMPONENT_TYPES {
        let token = token_for(component);
        let expected_section = format!("cm-16-16-16-{expected}-{expected}-{expected}-{expected}");

        assert!(
            token.contains(&expected_section),
            "{component:?} must spell as {expected:?}, which the registered \
             `vulkan:` namespace lists. Got token: {token}{ON_FAILURE}",
        );
    }
}

/// The complement of the test above, and the one that actually catches a
/// wholesale vocabulary swap: no emitted spelling may be *outside* the
/// registered set. A consistent rename passes every round-trip test in this
/// crate while failing this one.
#[test]
fn no_component_type_spells_an_unregistered_token() {
    for &(component, _) in NAMED_COMPONENT_TYPES {
        let token = token_for(component);
        // `16-16-16-<a>-<b>-<c>-<result>` — skip the three dimensions.
        for spelling in cm_section(&token).split('-').skip(3) {
            assert!(
                REGISTERED_COMPONENT_TYPES.contains(&spelling),
                "emitted component spelling {spelling:?} is not in the registered \
                 `vulkan:` namespace vocabulary {REGISTERED_COMPONENT_TYPES:?}.\n\n\
                 This is what a vocabulary-wide rename looks like from the outside: \
                 round-tripping still works, canonical spelling still holds, and the \
                 crate has quietly stopped implementing its own published \
                 namespace.{ON_FAILURE}",
            );
        }
    }
}

/// Both directions, so a spelling that is emitted but not accepted (or the
/// reverse) is caught. §6.8-0002 needs one target to have exactly one
/// spelling; a parse/spell asymmetry breaks byte-exact matching.
#[test]
fn every_registered_token_is_accepted_on_parse() {
    for spelling in REGISTERED_COMPONENT_TYPES {
        let token = format!(
            "vulkan:sg32.ops-none.arith-none.cm-16-16-16-{spelling}-{spelling}-{spelling}-{spelling}"
        );
        let parsed = VulkanTarget::parse(&token)
            .unwrap_or_else(|e| panic!("registered spelling {spelling:?} must parse, got {e:?}"));
        assert_eq!(
            parsed.to_token(),
            token,
            "registered spelling {spelling:?} must round-trip byte-exactly"
        );
    }
}

/// The `x<n>` escape is part of the registered vocabulary and is what keeps an
/// unknown component type honest rather than mis-spelled. It is a pattern, so
/// it is checked separately from the literal list.
#[test]
fn the_unknown_escape_is_spelled_as_the_namespace_requires() {
    let token = token_for(ComponentType::Other(1_000_141_000));
    assert!(
        token.contains("cm-16-16-16-x1000141000-x1000141000-x1000141000-x1000141000"),
        "an unnamed component type must spell as `x<n>` carrying the raw value: {token}"
    );
    assert_eq!(
        VulkanTarget::parse(&token)
            .expect("x<n> must parse")
            .to_token(),
        token
    );
}
