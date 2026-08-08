//! The emitted vocabulary must match the **registered** `vulkan:` namespace.
//!
//! Vulkane owns the content of KISS-Classify §6.8's `vulkan:` namespace, and
//! that content is published as a document in the KISS repository —
//! `spec/namespaces/vulkan.md`. This crate is the implementation of it. Until
//! now, nothing checked the two against each other: the document listed a set
//! of component-type spellings, this crate emitted a set, and the claim that
//! they were the same set was maintained entirely by hand.
//!
//! That gap is not hypothetical. During the KISS `sk4` schema event, vulkane
//! was asked to apply an `s` → `i` rename to these very spellings — a change
//! that belongs to §6.1's `structure_key` dtype set, which is a *different
//! vocabulary on a different axis*. Applying it here emits `i32` where the
//! registered namespace says `s32`, breaking conformance with the namespace
//! this crate defines.
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
//! So this file pins the vocabulary itself. If the registered namespace is
//! amended, update `REGISTERED_COMPONENT_TYPES` **in the same change** as the
//! `ComponentType` arms — and if these tests fail, the question to answer is
//! "was the namespace document amended?", not "how do I make this pass?".

use kiss_vulkan_vocab::*;

/// Verbatim from `spec/namespaces/vulkan.md` (KISS, `origin/main`), the
/// paragraph reading:
///
/// > and each component type is one of `f16`, `f32`, `f64`, `bf16`, `s8`,
/// > `s16`, `s32`, `s64`, `u8`, `u16`, `u32`, `u64`, or `x<n>` for a
/// > `VkComponentTypeKHR` [this vocabulary does not name]
///
/// `x<n>` is exercised separately, since it is a pattern rather than a literal.
const REGISTERED_COMPONENT_TYPES: &[&str] = &[
    "f16", "f32", "f64", "bf16", "s8", "s16", "s32", "s64", "u8", "u16", "u32", "u64",
];

/// Every named `ComponentType`, paired with the spelling the registered
/// namespace requires. Written out rather than derived, so that renaming a
/// variant's spelling cannot silently rewrite the expectation too.
const NAMED_COMPONENT_TYPES: &[(ComponentType, &str)] = &[
    (ComponentType::F16, "f16"),
    (ComponentType::F32, "f32"),
    (ComponentType::F64, "f64"),
    (ComponentType::BF16, "bf16"),
    (ComponentType::S8, "s8"),
    (ComponentType::S16, "s16"),
    (ComponentType::S32, "s32"),
    (ComponentType::S64, "s64"),
    (ComponentType::U8, "u8"),
    (ComponentType::U16, "u16"),
    (ComponentType::U32, "u32"),
    (ComponentType::U64, "u64"),
];

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
             `vulkan:` namespace lists. Got token: {token}\n\n\
             If this failed because the spelling was deliberately changed, the \
             registered namespace document (KISS `spec/namespaces/vulkan.md`) \
             has to be amended in the same change — otherwise this crate and \
             the namespace it defines disagree, and nothing else will notice.",
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
        let cm = token
            .split(".cm-")
            .nth(1)
            .expect("token carries a cm- section");
        // `16-16-16-<a>-<b>-<c>-<result>`
        for spelling in cm.split('-').skip(3) {
            assert!(
                REGISTERED_COMPONENT_TYPES.contains(&spelling),
                "emitted component spelling {spelling:?} is not in the registered \
                 `vulkan:` namespace vocabulary {REGISTERED_COMPONENT_TYPES:?}.\n\n\
                 This is what a vocabulary-wide rename looks like from the outside: \
                 round-tripping still works, canonical spelling still holds, and the \
                 crate has quietly stopped implementing its own published namespace. \
                 Amend `spec/namespaces/vulkan.md` in the same change, or revert the \
                 rename.",
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
