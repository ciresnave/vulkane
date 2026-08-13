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
/// **vocabulary version 3**, the paragraph reading:
///
/// > and each component type is one of `f16`, `f32`, `f64`, `bf16`, `i8`,
/// > `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f8e4m3fn`, `f8e5m2`, or
/// > `x<n>` for a `VkComponentTypeKHR` [this vocabulary does not name]
///
/// `x<n>` is exercised separately, since it is a pattern rather than a literal.
const REGISTERED_COMPONENT_TYPES: &[&str] = &[
    "f16", "f32", "f64", "bf16", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f8e4m3fn",
    "f8e5m2",
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
    (ComponentType::F8E4M3FN, "f8e4m3fn"),
    (ComponentType::F8E5M2, "f8e5m2"),
];

/// Spellings KISS **reserves** and this vocabulary must therefore never emit.
///
/// `f8e4m3fnuz` / `f8e5m2fnuz` are members of KISS's closed dtype set but carry
/// no computation semantics — a `structure_key` using one must be met with a
/// typed decline. Vulkan exposes no enumerant for either, so nothing should
/// ever derive them.
///
/// This is here because the *reason* the FP8 mapping is safe is that these two
/// are excluded. If a future change wired `FLOAT8_E4M3_EXT` to the `fnuz`
/// spelling — a one-character slip in a string literal — every round-trip test
/// in this crate would still pass, because the crate would agree with itself.
/// This is the assertion that would not.
const RESERVED_NEVER_EMITTED: &[&str] = &["f8e4m3fnuz", "f8e5m2fnuz"];

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

/// No component type may spell a **reserved** KISS dtype.
///
/// The FP8 mapping rests entirely on this exclusion: `FLOAT8_E4M3_EXT` maps to
/// `f8e4m3fn` rather than `f8e4m3fnuz` *because* the `fnuz` spellings are
/// reserved with no computation semantics, so a type a device computes with
/// cannot coherently be one. That argument is only worth as much as its
/// enforcement, and a `fnuz` slip is four characters in a string literal that
/// every round-trip and canonical-spelling test in this crate would sail past —
/// the crate would simply agree with itself about the wrong spelling.
#[test]
fn no_component_type_spells_a_reserved_dtype() {
    for &(component, _) in NAMED_COMPONENT_TYPES {
        let token = token_for(component);
        for spelling in cm_section(&token).split('-').skip(3) {
            assert!(
                !RESERVED_NEVER_EMITTED.contains(&spelling),
                "{component:?} spells {spelling:?}, which KISS reserves \
                 (KISS-CLASSIFY-6.1-0001: recognized on parse, no computation \
                 semantics, must be answered with a typed decline). A derived \
                 token must never contain one — it would claim a device computes \
                 in a format the spec says carries no semantics at this schema \
                 version.{ON_FAILURE}"
            );
        }
    }
}

/// The reserved spellings must also not be *parseable* into a named variant.
///
/// Complements the test above: that one proves we never emit `fnuz`, this one
/// proves we do not quietly accept it either. A reserved dtype arriving in a
/// token is not this vocabulary's to resolve — it is an unnamed type, so it
/// must land in `Other` via the `x<n>` route or fail to parse, never silently
/// become `F8E4M3FN`.
#[test]
fn reserved_spellings_do_not_parse_as_named_component_types() {
    for reserved in RESERVED_NEVER_EMITTED {
        let token = format!(
            "vulkan:sg32.ops-none.arith-none.cm-16-16-16-{reserved}-{reserved}-{reserved}-{reserved}"
        );
        assert!(
            VulkanTarget::parse(&token).is_err(),
            "the reserved spelling {reserved:?} parsed as a valid target. It is \
             not in this vocabulary and must not be accepted as though it were: \
             accepting it maps a no-semantics dtype onto a real component type."
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

/// The direction the tests above leave implicit: a registered spelling must
/// parse back to **the variant it names**, not merely to something that
/// re-spells identically.
///
/// `every_registered_token_is_accepted_on_parse` asserts a token survives
/// `parse` → `to_token` byte-exactly, which is a property of the *string*. It
/// would still hold if two variants shared a spelling, or if a spelling
/// resolved to a different variant that happened to spell the same way. Those
/// are unlikely, and unlikely is not the standard this file is held to — the
/// whole point of pinning a vocabulary against a document is that "surely that
/// couldn't happen" stops being load-bearing.
///
/// Checked through `VulkanTarget::parse` rather than a `ComponentType` parser
/// because the crate exposes no public per-component parse; the parsed
/// `CoopShape` fields are the reachable evidence, and going through the real
/// token is the more faithful test anyway.
#[test]
fn every_registered_spelling_parses_back_to_its_own_variant() {
    for &(component, spelling) in NAMED_COMPONENT_TYPES {
        let token = token_for(component);
        let parsed = VulkanTarget::parse(&token)
            .unwrap_or_else(|e| panic!("token for {spelling:?} must parse, got {e:?}"));

        let CoopMatrix::Shapes(shapes) = &parsed.coop else {
            panic!(
                "expected an enumerated shape list for {spelling:?}, got {:?}",
                parsed.coop
            );
        };
        let shape = shapes.first().expect("exactly one shape was spelled");

        // Every operand position carries the same component here, so this
        // establishes spelling↔variant identity — *not* that the positions are
        // wired to the right fields. A parser that swapped `a` and `result`
        // would pass this and every round-trip test in the crate, because the
        // re-spelled token is byte-identical when all four are equal. That
        // property is checked separately below.
        for (position, got) in [
            ("a", shape.a),
            ("b", shape.b),
            ("c", shape.c),
            ("result", shape.result),
        ] {
            assert_eq!(
                got, component,
                "{spelling:?} in operand position {position} parsed back as {got:?}, \
                 not {component:?}{ON_FAILURE}"
            );
        }
    }
}

/// Operand positions must survive the round trip in the right order.
///
/// Found while writing the test above: with one component in all four
/// positions, a parser that transposed `a` and `result` re-spells to a
/// byte-identical token and passes everything. So the fixture here uses four
/// *distinct* components, which is the only arrangement in which position is
/// observable at all.
#[test]
fn operand_positions_survive_the_round_trip_in_order() {
    let shape = CoopShape {
        m: 16,
        n: 8,
        k: 32,
        a: ComponentType::S8,
        b: ComponentType::U8,
        c: ComponentType::S32,
        result: ComponentType::F32,
        saturating: false,
    };
    let token = VulkanTarget {
        subgroup: Subgroup::Fixed(32),
        ops: OpClasses::NONE,
        arith: Arith::NONE,
        coop: CoopMatrix::from_shapes(vec![shape]),
    }
    .to_token();

    // Spelled in declaration order: M-N-K-A-B-C-R.
    assert!(
        token.contains("cm-16-8-32-i8-u8-i32-f32"),
        "operands must spell in M-N-K-A-B-C-R order: {token}{ON_FAILURE}"
    );

    let parsed = VulkanTarget::parse(&token).expect("must parse");
    let CoopMatrix::Shapes(shapes) = &parsed.coop else {
        panic!("expected an enumerated shape list, got {:?}", parsed.coop);
    };
    assert_eq!(
        shapes.first().copied(),
        Some(shape),
        "operands came back transposed"
    );
}
