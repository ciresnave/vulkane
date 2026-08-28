//! Emit the KISS-CLASSIFY §6.8-0008 vocabulary manifest for the `vulkan:`
//! namespace, as JSON on stdout.
//!
//! Run with:
//! `cargo run --example emit_vocabulary_manifest -p kiss-vulkan-vocab`
//!
//! # What this is
//!
//! §6.8-0008 lets a namespace publish its capability-set vocabulary as a
//! machine-readable manifest, so a consumer binds against an artifact instead
//! of hand-transcribing an annex or hand-parsing its prose. KISS pins the
//! **envelope**; the content stays the maintainer's (§6.8-0004).
//!
//! `vulkan` is `kind: "generated"` — a grammar over an open product space —
//! where `cuda` is `enumerated`, a closed list with every token in `members`.
//! That difference in *kind* is why the envelope exists at all.
//!
//! # The vectors are the normative part, not the grammar
//!
//! §6.8-0013 is explicit that for `kind: generated` the `field_spec` is
//! *documentation* — "canonicalization cannot be validated from a grammar" —
//! and the contract is the `vectors` array, which a conformant producer's
//! output MUST reproduce byte-exact.
//!
//! That is not ceremony here. Two of the five fields choose their encoding by a
//! **length-conditional switch**: above [`COOP_DIGEST_THRESHOLD`] bytes the
//! canonical enumeration is replaced by the FNV-1a-64 digest *of that same
//! string*. No alphabet and no regex expresses that rule, so a grammar-only
//! manifest lets a consumer **recognize** every token and still **produce** the
//! wrong one — and under §6.8-0002 byte-exact matching a wrong token is a
//! *different cell* rather than a degraded answer, so it surfaces as a silent
//! cache miss instead of an error.
//!
//! Every vector carries its **input as raw field values**, never as this
//! crate's types, so a second implementation can derive the expected token from
//! the manifest alone. A vector expressed in our own vocabulary would be
//! reproducible only by us, which is the opposite of what a contract is for.
//!
//! # The threshold vectors are found, not asserted
//!
//! §6.8-0013 wants each length-conditional field "presented *at* and
//! *immediately across* its boundary, so both forms are pinned at the exact
//! byte count that flips them". This program **searches** for inputs whose
//! canonical enumeration measures exactly 512 and exactly 513 bytes rather than
//! hand-computing them, and it measures by asking the crate for the emitted
//! field rather than by re-deriving a length formula. A formula that drifted
//! from the emitter would produce a manifest that is wrong in precisely the
//! place the manifest exists to pin.
//!
//! # Provenance
//!
//! `generated_from` names **this crate**, which is what emitted the bytes.
//! §6.8-0011 requires exactly that — *"Provenance names the producer; agreement
//! is a relation between two artifacts, and neither settles which is the
//! source."* The agreement obligation against `spec/namespaces/vulkan.md` is a
//! separate one and is owed regardless.

use kiss_vulkan_vocab::*;
use std::fmt::Write as _;

fn main() {
    print!("{}", manifest());
}

/// The manifest, as a string.
///
/// Separated from `main` so the freshness gate in
/// `tests/vocabulary_manifest.rs` can call it directly. The test `#[path]`-
/// includes this file rather than shelling out to `cargo run`, so the gate
/// compares the committed artifact against **this** generator rather than
/// against a second copy of the logic — which is the whole point of an
/// emit-and-compare gate and would be defeated by re-implementing it.
pub fn manifest() -> String {
    let mut o = String::new();

    writeln!(o, "{{").unwrap();
    writeln!(o, "  \"schema\": \"kiss-namespace-vocabulary-v1\",").unwrap();
    writeln!(o, "  \"namespace\": \"{NAMESPACE}\",").unwrap();
    // An integer: no quotes, no decimal point. §6.8-0008 — "a gate that
    // truncates a fractional value is not a gate."
    writeln!(o, "  \"vocabulary_version\": {VOCABULARY_VERSION},").unwrap();
    writeln!(
        o,
        "  \"generated_from\": \"{} {} (examples/emit_vocabulary_manifest.rs)\",",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION")
    )
    .unwrap();
    writeln!(o, "  \"kind\": \"generated\",").unwrap();
    writeln!(
        o,
        "  \"grammar\": \"vulkan:<subgroup>.<ops>.<arith>.<coop>.<coopvec>\","
    )
    .unwrap();
    writeln!(o, "  \"coverage_note\": \"{}\",", esc(COVERAGE_NOTE)).unwrap();

    emit_declarative(&mut o);
    emit_field_spec(&mut o);
    emit_vectors(&mut o);

    writeln!(o, "}}").unwrap();
    o
}

// ---------------------------------------------------------------------------
// Declarative half — §6.8-0012 requires this to suffice for a PARSE-only
// consumer. Everything a reader needs to recognise a well-formed token and
// reject a malformed one, and nothing that requires running a canonicalisation.
// ---------------------------------------------------------------------------

fn emit_declarative(o: &mut String) {
    writeln!(o, "  \"declarative\": {{").unwrap();
    writeln!(o, "    \"field_count\": 5,").unwrap();
    writeln!(o, "    \"field_separator\": \".\",").unwrap();
    writeln!(o, "    \"tuple_separator\": \"-\",").unwrap();
    writeln!(o, "    \"tuple_list_separator\": \",\",").unwrap();
    writeln!(o, "    \"ops_alphabet\": \"{}\",", OpClasses::alphabet()).unwrap();
    writeln!(o, "    \"arith_names\": {},", str_array(&Arith::alphabet())).unwrap();
    let spellings: Vec<String> = named_component_types()
        .iter()
        .map(|c| c.spelling())
        .collect();
    let refs: Vec<&str> = spellings.iter().map(String::as_str).collect();
    writeln!(o, "    \"component_types\": {},", str_array(&refs)).unwrap();
    writeln!(o, "    \"unnamed_component_escape\": \"x<n>\",").unwrap();
    writeln!(o, "    \"empty_set_spelling\": \"<prefix>-none\",").unwrap();
    writeln!(o, "    \"digest_marker\": \"fnv1a64-<hex16>\",").unwrap();
    writeln!(o, "    \"digest_threshold_bytes\": {COOP_DIGEST_THRESHOLD}").unwrap();
    writeln!(o, "  }},").unwrap();
}

// ---------------------------------------------------------------------------
// Production half — documentation only, per §6.8-0013. The binding contract is
// `vectors`.
// ---------------------------------------------------------------------------

fn emit_field_spec(o: &mut String) {
    let specs: [(&str, &str, &str); 5] = [
        (
            "subgroup",
            "sg",
            "The CHOSEN subgroup specialization, not the device envelope. \
             `sg<width>` for a pinned power-of-two width; `sgdyn` for a \
             width-agnostic kernel that reads the width at runtime. One device \
             commonly admits several, and they are different binaries, so a \
             device yields a SET of valid tokens rather than one.",
        ),
        (
            "ops",
            "ops-",
            "Subgroup operation classes, spelled as JUXTAPOSED single ASCII \
             letters in the canonical order given by `ops_alphabet`. \
             Juxtaposition is safe only because that alphabet is fixed-width \
             (§6.8-0006).",
        ),
        (
            "arith",
            "arith-",
            "Arithmetic capabilities, spelled as NAMED parts joined by `-` in \
             the canonical order given by `arith_names` — the names are \
             variable-length, so juxtaposition would not stay uniquely \
             decodable as the set grows. Note `st8`/`st16` are STORAGE \
             capabilities and are not compute precision: a conformant device \
             may accept 16-bit data in a buffer and perform the arithmetic in \
             f32. Reading one as the other is a silently wrong lowering.",
        ),
        (
            "coop",
            "cm-",
            "Cooperative-MATRIX shapes. Each tuple is M-N-K plus four component \
             types, joined by `-`; tuples are joined by `,`. Sorted and \
             deduplicated canonically, because driver report order is not \
             guaranteed stable and the token must be byte-identical either way. \
             LENGTH-CONDITIONAL — see the `threshold` and `digest_input` \
             vectors.",
        ),
        (
            "coopvec",
            "cv-",
            "Cooperative-VECTOR combinations. Each tuple is five component \
             types plus a transpose flag. Same sort/dedup rule as <coop>, and \
             length-conditional on the same 512-byte threshold — but measured \
             and digested INDEPENDENTLY. The two fields switch on their own \
             bytes and never together, which is why both carry their own \
             threshold vectors.",
        ),
    ];

    writeln!(o, "  \"field_spec\": [").unwrap();
    for (i, (field, prefix, note)) in specs.iter().enumerate() {
        let comma = if i + 1 == specs.len() { "" } else { "," };
        writeln!(
            o,
            "    {{ \"field\": \"{}\", \"prefix\": \"{}\", \"note\": \"{}\" }}{}",
            field,
            prefix,
            esc(note),
            comma
        )
        .unwrap();
    }
    writeln!(o, "  ],").unwrap();
}

// ---------------------------------------------------------------------------
// Vectors — the normative contract (§6.8-0013).
// ---------------------------------------------------------------------------

fn emit_vectors(o: &mut String) {
    let mut v: Vec<String> = Vec::new();

    // -- order: a non-canonically-ordered input and its canonical output.
    let unsorted = vec![big_shape(3), big_shape(1), big_shape(2)];
    v.push(coop_vector(
        "order",
        "Shapes presented in non-canonical order. A producer that emitted \
         driver order would differ from an honest peer on the same device, and \
         under byte-exact matching that is a different cell rather than a \
         degraded answer.",
        &unsorted,
    ));

    // -- dedup: a duplicate-bearing input and its deduped output.
    let dupes = vec![big_shape(1), big_shape(2), big_shape(1), big_shape(2)];
    v.push(coop_vector(
        "dedup",
        "A duplicate-bearing input. Deduplication happens before spelling, so \
         a device reporting the same shape twice yields the same token as one \
         reporting it once.",
        &dupes,
    ));

    // -- the same two for <coopvec>, because the fields are independent.
    let cv_unsorted = vec![combo(3), combo(1), combo(2)];
    v.push(coopvec_vector(
        "order",
        "Cooperative-VECTOR combinations in non-canonical order. Pinned \
         separately from <coop> because the two fields canonicalize \
         independently — an implementation that sorted one and not the other \
         would pass a <coop>-only vector set.",
        &cv_unsorted,
    ));
    let cv_dupes = vec![combo(1), combo(2), combo(1)];
    v.push(coopvec_vector(
        "dedup",
        "Duplicate cooperative-vector combinations.",
        &cv_dupes,
    ));

    // -- the two SET-VALUED scalar fields. Nothing pinned these before: every
    //    vector above carries `ops-none` and `arith-none`, so a consumer could
    //    read the alphabet and still not know how two members are joined.
    v.push(set_field_vector(
        "arith",
        "Two arithmetic capabilities, given in NON-CANONICAL order. Pins that \
         `<arith>` joins its names with `-` and never juxtaposes them, and that \
         the canonical order is the alphabet's own order rather than the order \
         a device reported them in. Both matter because matching is byte-exact: \
         `arith-i8-f16` is a different cell, not a differently-written same cell.",
        OpClasses::NONE,
        Arith::FLOAT16 | Arith::INT8,
    ));
    v.push(set_field_vector(
        "ops",
        "Three operation classes, given in NON-CANONICAL order. Pinned \
         SEPARATELY from `<arith>` because the two set-valued fields do NOT \
         spell alike: `<ops>` JUXTAPOSES single letters while `<arith>` joins \
         variable-length names with `-`. A vector for one says nothing about \
         the other, and an implementer who generalised from `<arith>` alone \
         would emit `ops-a-b-r`.",
        OpClasses::BASIC | OpClasses::BALLOT | OpClasses::ROTATE,
        Arith::NONE,
    ));

    // -- threshold + digest_input, per field, at and immediately across.
    match find_coop_at_and_across() {
        Some((at, across)) => {
            v.push(coop_threshold_vector("threshold", AT_NOTE, &at));
            v.push(coop_threshold_vector("threshold", ACROSS_NOTE, &across));
            v.push(coop_digest_input_vector(&across));
        }
        None => panic!(
            "could not construct a <coop> input measuring exactly {} and {} \
             bytes. The search family no longer spans the boundary with \
             1-byte granularity; widen it rather than weakening the vector.",
            COOP_DIGEST_THRESHOLD,
            COOP_DIGEST_THRESHOLD + 1
        ),
    }

    match find_coopvec_at_and_across() {
        Some((at, across)) => {
            v.push(coopvec_threshold_vector("threshold", AT_NOTE, &at));
            v.push(coopvec_threshold_vector("threshold", ACROSS_NOTE, &across));
            v.push(coopvec_digest_input_vector(&across));
        }
        None => panic!(
            "could not construct a <coopvec> input measuring exactly {} and {} \
             bytes; widen the search family rather than weakening the vector.",
            COOP_DIGEST_THRESHOLD,
            COOP_DIGEST_THRESHOLD + 1
        ),
    }

    writeln!(o, "  \"vectors\": [").unwrap();
    for (i, entry) in v.iter().enumerate() {
        let comma = if i + 1 == v.len() { "" } else { "," };
        writeln!(o, "    {entry}{comma}").unwrap();
    }
    writeln!(o, "  ]").unwrap();
}

const AT_NOTE: &str = "Canonical enumeration measuring EXACTLY the threshold. \
                       Still spelled in full — the switch is strictly above \
                       the threshold, not at it. An implementation using `>=` \
                       fails here and passes every straddling test that never \
                       lands on the boundary.";

const ACROSS_NOTE: &str = "One byte across the threshold. Spelled as the \
                           digest. Paired with the vector above, these pin the \
                           exact byte count that flips the form.";

// ---------------------------------------------------------------------------
// Shape and combination families.
//
// Two tuple widths per field, so the search below can hit an exact byte count:
// the wide family alone moves in ~25-byte steps and cannot land on 512.
// Distinctness matters as much as width — a family that collided would be
// deduplicated to something shorter than the search believed it had built.
// ---------------------------------------------------------------------------

/// Wide tuple: two-digit dimensions, four-char-ish component spellings.
fn big_shape(i: u32) -> CoopShape {
    CoopShape {
        m: 10 + i % 90,
        n: 10 + (i / 90) % 90,
        k: 16,
        a: ComponentType::F16,
        b: ComponentType::F16,
        c: ComponentType::F32,
        result: ComponentType::F32,
        saturating: false,
    }
}

/// Narrow tuple: single-digit dimensions and the shortest component spellings.
fn small_shape(i: u32) -> CoopShape {
    CoopShape {
        m: 1 + i % 9,
        n: 1 + (i / 9) % 9,
        k: 1,
        a: ComponentType::S8,
        b: ComponentType::S8,
        c: ComponentType::S8,
        result: ComponentType::S8,
        saturating: false,
    }
}

fn combo(i: u32) -> CoopVecCombo {
    CoopVecCombo {
        input: ComponentType::U32,
        input_interpretation: ComponentType::S8Packed,
        matrix_interpretation: ComponentType::S8,
        bias_interpretation: ComponentType::S32,
        result: ComponentType::Other(1000 + i),
        transpose: false,
    }
}

/// Narrow combination — `Other(n)` with a short `n` keeps the tuple shorter.
fn small_combo(i: u32) -> CoopVecCombo {
    CoopVecCombo {
        input: ComponentType::U8,
        input_interpretation: ComponentType::U8,
        matrix_interpretation: ComponentType::U8,
        bias_interpretation: ComponentType::U8,
        result: ComponentType::Other(i),
        transpose: false,
    }
}

// ---------------------------------------------------------------------------
// Measurement + search.
//
// Length is measured by asking the crate for the field it actually emitted,
// never by re-deriving a formula. A formula that drifted from the emitter would
// make this manifest wrong in exactly the place it exists to pin.
// ---------------------------------------------------------------------------

fn token_of(coop: CoopMatrix, coopvec: CoopVector) -> String {
    VulkanTarget {
        subgroup: Subgroup::Fixed(32),
        ops: OpClasses::NONE,
        arith: Arith::NONE,
        coop,
        coopvec,
    }
    .to_token()
}

/// Cross-check: below the threshold the crate's enumeration must be exactly
/// what it spells into the field. Above it there is nothing to compare against,
/// which is why the check runs where it can rather than not at all.
fn assert_enumeration_matches_spelled_field(shapes: &[CoopShape]) {
    let e = measured_coop_enumeration(shapes);
    if e.len() > COOP_DIGEST_THRESHOLD {
        return;
    }
    let tok = token_of(CoopMatrix::from_shapes(shapes.to_vec()), CoopVector::None);
    let spelled = tok
        .split(".cm-")
        .nth(1)
        .and_then(|t| t.split(".cv-").next())
        .expect("token always carries a cm- field");
    assert_eq!(
        e, spelled,
        concat!(
            "the canonical enumeration and the spelled field disagree below the ",
            "threshold; the manifest would pin a digest_input the emitter never ",
            "uses"
        )
    );
}

/// Search the two-width family for enumerations of exactly `T` and `T+1` bytes.
fn find_coop_at_and_across() -> Option<(Vec<CoopShape>, Vec<CoopShape>)> {
    let mut at = None;
    let mut across = None;
    for wide in 0..40u32 {
        for narrow in 0..60u32 {
            let mut s: Vec<CoopShape> = (0..wide).map(big_shape).collect();
            s.extend((0..narrow).map(small_shape));
            if s.is_empty() {
                continue;
            }
            let n = measured_coop_enumeration(&s).len();
            if n == COOP_DIGEST_THRESHOLD && at.is_none() {
                assert_enumeration_matches_spelled_field(&s);
                at = Some(s.clone());
            }
            if n == COOP_DIGEST_THRESHOLD + 1 && across.is_none() {
                across = Some(s);
            }
            if let (Some(a), Some(b)) = (&at, &across) {
                return Some((a.clone(), b.clone()));
            }
        }
    }
    None
}

fn find_coopvec_at_and_across() -> Option<(Vec<CoopVecCombo>, Vec<CoopVecCombo>)> {
    let mut at = None;
    let mut across = None;
    for wide in 0..40u32 {
        for narrow in 0..80u32 {
            let mut c: Vec<CoopVecCombo> = (0..wide).map(combo).collect();
            c.extend((0..narrow).map(small_combo));
            if c.is_empty() {
                continue;
            }
            let n = measured_coopvec_enumeration(&c).len();
            if n == COOP_DIGEST_THRESHOLD && at.is_none() {
                at = Some(c.clone());
            }
            if n == COOP_DIGEST_THRESHOLD + 1 && across.is_none() {
                across = Some(c);
            }
            if let (Some(a), Some(b)) = (&at, &across) {
                return Some((a.clone(), b.clone()));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Vector serialisation. Inputs go in as raw field values so the manifest is
// reproducible without this crate.
// ---------------------------------------------------------------------------

fn shape_json(s: &CoopShape) -> String {
    format!(
        "{{\"m\":{},\"n\":{},\"k\":{},\"a\":\"{}\",\"b\":\"{}\",\"c\":\"{}\",\"result\":\"{}\",\"saturating\":{}}}",
        s.m,
        s.n,
        s.k,
        s.a.spelling(),
        s.b.spelling(),
        s.c.spelling(),
        s.result.spelling(),
        s.saturating
    )
}

fn combo_json(c: &CoopVecCombo) -> String {
    format!(
        "{{\"input\":\"{}\",\"input_interpretation\":\"{}\",\"matrix_interpretation\":\"{}\",\"bias_interpretation\":\"{}\",\"result\":\"{}\",\"transpose\":{}}}",
        c.input.spelling(),
        c.input_interpretation.spelling(),
        c.matrix_interpretation.spelling(),
        c.bias_interpretation.spelling(),
        c.result.spelling(),
        c.transpose
    )
}

fn coop_vector(pins: &str, note: &str, shapes: &[CoopShape]) -> String {
    let token = token_of(CoopMatrix::from_shapes(shapes.to_vec()), CoopVector::None);
    format!(
        "{{ \"pins\": \"{}\", \"field\": \"coop\", \"note\": \"{}\", \"input\": {{ \"subgroup\": 32, \"ops\": [], \"arith\": [], \"coop\": [{}], \"coopvec\": [] }}, \"token\": \"{}\" }}",
        pins,
        esc(note),
        shapes.iter().map(shape_json).collect::<Vec<_>>().join(","),
        token
    )
}

/// A vector for one of the two SET-VALUED scalar fields, `<ops>` and `<arith>`.
///
/// These existed in the declarative half and in no vector, which is a gap a
/// consumer found rather than a gap anyone here noticed: every vector shipped
/// before this one carried `ops-none` and `arith-none`, so the manifest pinned
/// the two fields' ALPHABETS while pinning nothing about how a multi-member set
/// is written down. A downstream implementer asked whether `arith` with two
/// members is `arith-f16i8`, `arith-f16-i8`, or a repeated field, and the
/// machine-readable artifact could not answer.
///
/// `input` lists the members in a DELIBERATELY non-canonical order, exactly as
/// the `<coop>` order vector does, so one vector pins two things a parser cannot
/// infer from the alphabet: the join, and the canonical order.
/// The text an empty set spells after its `<field>-` prefix is stripped.
///
/// Named because it must be REJECTED before a field is decomposed into members,
/// and a bare `"none"` at the comparison site reads like a member.
const EMPTY_SET_MEMBER: &str = "none";

fn set_field_vector(field: &str, note: &str, ops: OpClasses, arith: Arith) -> String {
    let token = VulkanTarget {
        subgroup: Subgroup::Fixed(32),
        ops,
        arith,
        coop: CoopMatrix::None,
        coopvec: CoopVector::None,
    }
    .to_token();

    // The input members are DERIVED FROM THE TOKEN, not passed alongside it.
    // An earlier draft took them as a separate argument and I promptly wrote
    // members that did not correspond to the flags -- the vector would have
    // taught a reader that {r,a,b} spells `blw`. Deriving them makes the two
    // halves of the vector incapable of disagreeing, which is the same reason
    // the coop vectors compute their token from the shapes they display.
    let spelled = token
        .split(':')
        .nth(1)
        .expect("a token has a namespace prefix")
        .split('.')
        .find(|f| f.starts_with(field))
        .unwrap_or_else(|| panic!("token has no `{field}` field: {token}"))
        .strip_prefix(field)
        .and_then(|r| r.strip_prefix('-'))
        .unwrap_or_else(|| panic!("`{field}` field is not `{field}-...`: {token}"));

    // The EMPTY-SET SENTINEL has to be rejected before the field is decomposed,
    // not after. `ops-none` strips to `"none"`, and `<ops>` decomposes by
    // character, so it becomes `["n","o","n","e"]` -- four members, which sails
    // through a `len() > 1` check and emits nonsense. `<arith>` splits on `-` and
    // yields `["none"]`, so the same check catches it BY ACCIDENT of arity. A
    // guard that holds for one field and not the other is not a guard.
    assert!(
        spelled != EMPTY_SET_MEMBER,
        "a set-spelling vector was built from an EMPTY set: `{field}-{spelled}` \
         is the empty-set sentinel, not a member list. Pass flags with at least \
         two members."
    );

    // `<ops>` juxtaposes single letters; `<arith>` joins names with `-`. That
    // difference is the whole point of having a vector for each.
    let canonical: Vec<String> = if field == "ops" {
        spelled.chars().map(|c| c.to_string()).collect()
    } else {
        spelled.split('-').map(str::to_string).collect()
    };
    assert!(
        canonical.len() > 1,
        "a set-spelling vector must carry MORE THAN ONE member, or it pins \
         nothing about how members are joined -- got {canonical:?}"
    );

    // Reversed, so the input is non-canonical by construction rather than by
    // an author remembering to scramble it.
    let members = canonical
        .iter()
        .rev()
        .map(|m| format!("\"{m}\""))
        .collect::<Vec<_>>()
        .join(",");
    let (ops_in, arith_in) = if field == "ops" {
        (members, String::new())
    } else {
        (String::new(), members)
    };
    format!(
        "{{ \"pins\": \"set-spelling\", \"field\": \"{}\", \"note\": \"{}\", \"input\": {{ \"subgroup\": 32, \"ops\": [{}], \"arith\": [{}], \"coop\": [], \"coopvec\": [] }}, \"token\": \"{}\" }}",
        field,
        esc(note),
        ops_in,
        arith_in,
        token
    )
}

fn coopvec_vector(pins: &str, note: &str, combos: &[CoopVecCombo]) -> String {
    let token = token_of(CoopMatrix::None, CoopVector::from_combos(combos.to_vec()));
    format!(
        "{{ \"pins\": \"{}\", \"field\": \"coopvec\", \"note\": \"{}\", \"input\": {{ \"subgroup\": 32, \"ops\": [], \"arith\": [], \"coop\": [], \"coopvec\": [{}] }}, \"token\": \"{}\" }}",
        pins,
        esc(note),
        combos.iter().map(combo_json).collect::<Vec<_>>().join(","),
        token
    )
}

fn coop_threshold_vector(pins: &str, note: &str, shapes: &[CoopShape]) -> String {
    let token = token_of(CoopMatrix::from_shapes(shapes.to_vec()), CoopVector::None);
    let measured = measured_coop_enumeration(shapes);
    format!(
        "{{ \"pins\": \"{}\", \"field\": \"coop\", \"note\": \"{}\", \"enumeration_bytes\": {}, \"threshold_bytes\": {}, \"input\": {{ \"subgroup\": 32, \"ops\": [], \"arith\": [], \"coop\": [{}], \"coopvec\": [] }}, \"token\": \"{}\" }}",
        pins,
        esc(note),
        measured.len(),
        COOP_DIGEST_THRESHOLD,
        shapes.iter().map(shape_json).collect::<Vec<_>>().join(","),
        token
    )
}

fn coopvec_threshold_vector(pins: &str, note: &str, combos: &[CoopVecCombo]) -> String {
    let token = token_of(CoopMatrix::None, CoopVector::from_combos(combos.to_vec()));
    let measured = measured_coopvec_enumeration(combos);
    format!(
        "{{ \"pins\": \"{}\", \"field\": \"coopvec\", \"note\": \"{}\", \"enumeration_bytes\": {}, \"threshold_bytes\": {}, \"input\": {{ \"subgroup\": 32, \"ops\": [], \"arith\": [], \"coop\": [], \"coopvec\": [{}] }}, \"token\": \"{}\" }}",
        pins,
        esc(note),
        measured.len(),
        COOP_DIGEST_THRESHOLD,
        combos.iter().map(combo_json).collect::<Vec<_>>().join(","),
        token
    )
}

/// The exact byte string fed to the digest.
///
/// §6.8-0013 wants this pinned separately from the threshold "so a producer may
/// disagree about *whether* to digest but never about *what* is digested". The
/// two are different failure modes and only one of them is visible in the token.
fn coop_digest_input_vector(shapes: &[CoopShape]) -> String {
    let s = measured_coop_enumeration(shapes);
    format!(
        "{{ \"pins\": \"digest_input\", \"field\": \"coop\", \"note\": \"{}\", \"digest_input\": \"{}\", \"digest_input_bytes\": {}, \"digest\": \"fnv1a64-{:016x}\" }}",
        esc(DIGEST_INPUT_NOTE),
        esc(&s),
        s.len(),
        fnv1a64(s.as_bytes())
    )
}

fn coopvec_digest_input_vector(combos: &[CoopVecCombo]) -> String {
    let s = measured_coopvec_enumeration(combos);
    format!(
        "{{ \"pins\": \"digest_input\", \"field\": \"coopvec\", \"note\": \"{}\", \"digest_input\": \"{}\", \"digest_input_bytes\": {}, \"digest\": \"fnv1a64-{:016x}\" }}",
        esc(DIGEST_INPUT_NOTE),
        esc(&s),
        s.len(),
        fnv1a64(s.as_bytes())
    )
}

const DIGEST_INPUT_NOTE: &str = "The exact byte string measured against the threshold AND fed to the \
     FNV-1a-64 digest — the same string, which is the property that makes the \
     switch reproducible. Pinned separately from `threshold` because a \
     producer can agree about whether to digest and still digest something \
     else; that disagreement is invisible in the token, which carries only the \
     hash.";

/// The canonical enumeration, taken from the crate rather than rebuilt.
///
/// The first draft of this file re-implemented the tuple spelling here so it
/// could measure inputs above the threshold, where the emitted field shows only
/// a hash. It got the cooperative-VECTOR tuple wrong — appending a sixth field
/// for `transpose` where the crate appends `-t` only when true — which would
/// have published a `digest_input` no conformant producer could reproduce, in
/// the one vector whose entire purpose is pinning what gets digested. Two
/// implementations of one rule, in the artifact written to stop exactly that.
fn measured_coop_enumeration(shapes: &[CoopShape]) -> String {
    CoopMatrix::from_shapes(shapes.to_vec())
        .canonical_enumeration()
        .expect("a non-empty shape list always has an enumeration")
}

fn measured_coopvec_enumeration(combos: &[CoopVecCombo]) -> String {
    CoopVector::from_combos(combos.to_vec())
        .canonical_enumeration()
        .expect("a non-empty combination list always has an enumeration")
}

// ---------------------------------------------------------------------------

const COVERAGE_NOTE: &str = "What this manifest does and does not pin. The \
    DECLARATIVE half suffices to PARSE a `vulkan:` token — field count, \
    separators, alphabets, component spellings, the unnamed escape, and the \
    digest marker. It does NOT suffice to PRODUCE one. Two of the five fields, \
    <coop> and <coopvec>, choose their encoding by a LENGTH-CONDITIONAL \
    switch: above 512 bytes the canonical enumeration is replaced by the \
    FNV-1a-64 digest of that same string. No alphabet or regex expresses that, \
    so a consumer binding only against `grammar` can recognise every token and \
    still emit the wrong one — and under KISS-CLASSIFY-6.8-0002 byte-exact \
    matching a wrong token is a DIFFERENT CELL, not a degraded answer, so it \
    surfaces as a silent cache miss rather than an error. The `vectors` array \
    is therefore the normative contract for producers (6.8-0013). It pins \
    canonical ORDER, DEDUP, both length-conditional THRESHOLDS at and \
    immediately across 512 bytes, and the exact DIGEST_INPUT byte string \
    measured against each — so a producer may disagree about WHETHER to digest \
    but never about WHAT is digested. This namespace has two length-conditional \
    fields, so `threshold` and `digest_input` are present per-field rather than \
    omitted, and they are pinned SEPARATELY because the fields switch on their \
    own bytes and never together. NOT pinned here: which tokens a given device \
    admits. That is a deriver's job and needs a driver, which 6.9-0003 forbids \
    requiring of a token producer — the token names a CHOSEN specialization, so \
    one device yields a set of valid tokens rather than one. Also not claimed: \
    currency. This records the vocabulary version its bytes were generated \
    against, and a stamp proves BINDING, not CURRENCY.";

/// Every `ComponentType` this vocabulary version names, in canonical order.
///
/// Hand-listed rather than derived, and kept complete by
/// `every_variant_is_accounted_for` in the crate's own tests — an exhaustive
/// `match` that stops compiling when a variant is added. It lives inside the
/// defining crate because `ComponentType` is `#[non_exhaustive]`, which forces
/// any `match` written elsewhere to carry a `_` arm and therefore never break.
fn named_component_types() -> Vec<ComponentType> {
    vec![
        ComponentType::F16,
        ComponentType::F32,
        ComponentType::F64,
        ComponentType::BF16,
        ComponentType::S8,
        ComponentType::S16,
        ComponentType::S32,
        ComponentType::S64,
        ComponentType::U8,
        ComponentType::U16,
        ComponentType::U32,
        ComponentType::U64,
        ComponentType::F8E4M3FN,
        ComponentType::F8E5M2,
        ComponentType::S8Packed,
        ComponentType::U8Packed,
    ]
}

fn str_array(items: &[&str]) -> String {
    format!(
        "[{}]",
        items
            .iter()
            .map(|s| format!("\"{s}\""))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o
}
