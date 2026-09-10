// SPDX-License-Identifier: MIT OR Apache-2.0
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
    // ⚠️ The manifest's OWN vectors digest, the counterpart to
    // `vocabulary_version` directly above it: §6.8-0017 has a reader compare a
    // demonstration's `sufficiency.vectors_digest` against "the manifest's own",
    // and `vocabulary_version` is the top-level field that phrase means for the
    // other half of the pair.
    //
    // The clause requires only that a reader be able to REBUILD this from
    // `vectors` by the pinned construction; carrying it is the MAY half. Carried
    // anyway, and only because it costs 24 bytes: with no value in the artifact a
    // local gate can only compare a number this file computed against a number
    // this file computed. That is the vacuously-green shape -- green for its
    // author, and silent about every other party.
    writeln!(
        o,
        "  \"vectors_digest\": \"fnv1a64-{:016x}\",",
        kiss_vulkan_vocab::fnv1a64(vectors_digest_input(&build_vectors()).as_bytes())
    )
    .unwrap();
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

    // KISS-CLASSIFY-6.8-0017: a manifest MUST carry a `sufficiency` object with
    // a `status` of exactly `demonstrated` or `unexercised`. `unexercised` is
    // DECLARED, never inferred from absence -- an absent field cannot be told
    // apart from a forgotten one.
    //
    // ⚠️ THIS SAYS `unexercised` DESPITE A REAL DEMONSTRATION HAVING HAPPENED,
    // and the reason is the whole point of the field.
    //
    // Baracuda reproduced all TWELVE vectors of the 0.4.1 manifest
    // byte-identically from the manifest alone -- they extracted it, deleted the
    // archive so this source was physically unreachable, used no docs.rs and
    // asked nothing -- and listed what they had to supply from outside. That was
    // the first §5.3-condition-2-shaped demonstration anywhere in the suite.
    //
    // But this manifest has THIRTEEN vectors. The 13th pins `transpose`, added
    // BECAUSE of that reproduction. So the artifact published here is not the
    // artifact that was reproduced, and `demonstrated` would assert a run
    // against a vector set that did not exist when the run happened.
    //
    // §6.8-0013 makes the `vectors` array the normative contract, and a vector
    // set can change without `vocabulary_version` changing: adding a vector
    // documents a token that was ALREADY legal -- `CoopVecCombo::spell` has
    // emitted `-t` since before any of this. The LANGUAGE did not move; the
    // CONTRACT did. §6.8-0017's currency check keys on `vocabulary_version`, so
    // a literal reading would have let `demonstrated` carry forward.
    //
    // Declining what the clause would have allowed, on the clause's own
    // principle: claiming a demonstration on the strength of a run against a
    // different contract is the flattering inference the `derived` array exists
    // to stop, one level up. The remedy is a re-run against what actually
    // ships, which tests whether the fixes closed the guesses -- a re-assertion
    // cannot.
    writeln!(o, "  \"sufficiency\": {{").unwrap();
    writeln!(o, "    \"status\": \"unexercised\",").unwrap();
    // ⚠️ The count is DERIVED from the vector list, not spelled out.
    //
    // The first version of this note said "thirteen" and was stale one commit
    // later, when three vectors closing baracuda's residue landed and
    // `vocabulary_version` did not move. The field whose whole purpose is to
    // say "this artifact is not the one that was reproduced" had itself gone
    // stale about which artifact this is.
    //
    // That is the same currency gap reported against §6.8-0017 upstream: a
    // reproduction record keyed on `vocabulary_version` is blind to the vector
    // set moving underneath it. Deriving the number is the one fix available
    // inside this file — it cannot drift, because there is nothing to update.
    writeln!(
        o,
        "    \"note\": \"{}\"",
        esc(&format!(
            "A byte-identical reproduction of the TWELVE-vector 0.4.1 manifest was performed \
             by baracuda from the manifest alone. This manifest has {n_vectors}, the extra \
             ones added because of that reproduction, so what ships here is not what was \
             reproduced. Recorded as unexercised rather than carried forward: §6.8-0013 makes \
             the vectors array the normative contract, and it moved. A re-run against this \
             artifact is the remedy, and its value is the `guessed` list, not the pass.",
            n_vectors = build_vectors().len()
        ))
    )
    .unwrap();
    writeln!(o, "  }},").unwrap();

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
    // ⚠️ The angle-bracket convention, stated because NOTHING said it.
    //
    // Five values in this manifest carry `<...>` placeholders -- four here
    // (`unnamed_component_escape`, `empty_set_spelling`, `digest_marker`,
    // `unnamed_component_escape_order`) and the top-level `grammar`. A reader
    // had to infer that the brackets were not literal text.
    //
    // ⚠️ Found by baracuda rebuilding the digest from this manifest alone.
    // They read `digest_marker` as a PREFIX and emitted
    // `fnv1a64-<hex16>-5fb4518c42b73202`. That was the only error in an
    // otherwise byte-exact reproduction, and the field name argues for their
    // reading: a thing called a MARKER sounds like literal text, while its
    // value is a format string. The `<hex16>` inside the value is the only
    // thing that says otherwise.
    //
    // Renaming `digest_marker` would fix the instance. Stating the notation
    // fixes the class -- which is the lesson `sgdyn` taught by sitting beside
    // `transpose` unnoticed while `transpose` was being closed.
    writeln!(
        o,
        "    \"notation\": \"{}\",",
        esc(
            "An angle-bracketed name is a PLACEHOLDER for a value, never literal \
             text: `fnv1a64-<hex16>` means the marker `fnv1a64-` followed by \
             sixteen hex digits, not a string containing `<hex16>`. This applies \
             to every value in this manifest, including `grammar`. Text outside \
             the brackets is literal."
        )
    )
    .unwrap();
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
    // ⚠️ A TEMPLATE, and it must stay one.
    //
    // This read `<prefix>-none`, which is WRONG against this manifest's own
    // `prefix` values rather than merely underspecified: `ops` declares
    // prefix `ops-` and `coop` declares `cm-`, so `<prefix>` + `-none`
    // composes to `ops--none` and `cm--none` -- a double dash, for four of
    // the five fields, while the tokens spell `ops-none` and `cm-none`.
    //
    // ⚠️ The first repair replaced it with a correct SENTENCE, and the drill
    // caught that: restoring the defect left the gate GREEN, because prose
    // cannot be composed and the test had quietly hardcoded the rule instead
    // of following the stated one. A machine-checkable value that is wrong
    // is a bug; an unfalsifiable value that is right is a worse bug, because
    // the next wrong one arrives with nothing able to say so.
    //
    // Every prefix that can spell the empty set already carries its own
    // trailing `-`, so the separator belongs in the PREFIX, not in this rule.
    // `declarative.notation` says angle brackets are placeholders and text
    // outside them is literal, which is what makes `<prefix>none` readable.
    writeln!(o, "    \"empty_set_spelling\": \"<prefix>none\",").unwrap();
    writeln!(o, "    \"digest_marker\": \"fnv1a64-<hex16>\",").unwrap();
    // ⚠️ The ALGORITHM and its constants, not merely the marker. Baracuda
    // reproduced all twelve vectors from this manifest and named this the
    // single strongest gap: "a vector pins the OUTPUT, not the algorithm."
    // FNV-1 and FNV-1a differ by one letter -- the operands are swapped --
    // emit different bytes, and BOTH satisfy the `fnv1a64-<hex16>` marker
    // shape, so a wrong-constant producer fails the digest vectors without
    // ever learning why. No number of additional vectors closes that.
    //
    // Written from the crate constants rather than transcribed, so the
    // manifest cannot disagree with the digest it describes.
    writeln!(o, "    \"digest_algorithm\": \"FNV-1a 64-bit\",").unwrap();
    writeln!(
        o,
        "    \"digest_offset_basis\": \"{:#018x}\",",
        kiss_vulkan_vocab::FNV_OFFSET_BASIS
    )
    .unwrap();
    writeln!(
        o,
        "    \"digest_prime\": \"{:#x}\",",
        kiss_vulkan_vocab::FNV_PRIME
    )
    .unwrap();
    // ⚠️ The BYTES the digest runs over, which is not the same fact as the
    // algorithm above. FNV-1a is defined on a byte sequence, and every spelling
    // this vocabulary can produce is ASCII, so no digest_input vector can ever
    // discriminate UTF-8 from Latin-1 from UTF-16 — they agree on the whole
    // corpus. The vectors are not weak here; the question is OUTSIDE what a
    // vector can ask. Stated because a manifest-only reader must otherwise guess.
    writeln!(
        o,
        "    \"digest_encoding\": \"UTF-8 bytes of the enumeration string\","
    )
    .unwrap();
    // Both pinned digests happen to have no leading zero nibble, so the vectors
    // cannot distinguish zero-padded from trimmed hex. Stated rather than sampled.
    writeln!(
        o,
        "    \"digest_hex\": \"lowercase, zero-padded to exactly 16 digits\","
    )
    .unwrap();
    // Every sampled run is equal-width, so `x9` vs `x10` never discriminates
    // numeric from lexicographic ordering. Stated for the same reason.
    writeln!(
        o,
        "    \"unnamed_component_escape_order\": \"numeric on <n>, not lexicographic\","
    )
    .unwrap();
    // ⚠️ The construction of `vectors_digest`'s input, which §6.8-0017 requires
    // the NAMESPACE to pin. KISS pins the algorithm and constants and explicitly
    // does not define this input, because §6.8-0007's input is "the canonical
    // enumeration string it REPLACES" and the vectors array replaces nothing.
    //
    // ⚠️ Defined over THIS MANIFEST'S OWN RENDERING rather than a re-serialization.
    // A reader already holds the bytes; asking them to re-serialize reintroduces
    // every question canonical JSON leaves open -- key order, whitespace, number
    // formatting, unicode escaping -- none of which any clause settles. Taking the
    // text as rendered has exactly one answer.
    //
    // ⚠️ Note-excluded, which the clause requires and which also happens to make
    // the input pure ASCII here: 7 of 16 vectors carry non-ASCII in a note, 0 of 16
    // without. That is luck rather than design, so the encoding is STATED.
    writeln!(
        o,
        "    \"vectors_digest_input\": \"{}\",",
        esc(
            "Take each element of `vectors` AS RAW TEXT, exactly as this manifest \
             renders it -- do NOT parse and re-serialize it, because a producer that \
             re-serializes emits its own key order and whitespace and gets different \
             bytes. Trim surrounding whitespace and any trailing comma. Delete the \
             element's `note` member and the `, ` separating it, locating the member's \
             end by JSON string-scanning rather than by searching for `, `, since a \
             note's own prose contains that sequence. Concatenate the results in array \
             order, each followed by U+000A, and digest the UTF-8 bytes using the \
             algorithm and constants above. Defined over whatever members an element \
             actually carries, never a fixed template: a `digest_input` vector carries \
             no `token` at all."
        )
    )
    .unwrap();
    // ⚠️ The SORT KEY, which "sorted and deduplicated canonically" never gave.
    //
    // Found by baracuda writing a producer from the 0.4.2 manifest alone.
    // Their sort ranked the component types and OMITTED the flag, so two
    // tuples differing only in `-sat` or `-t` compared EQUAL and a stable
    // sort emitted them in input order. Their output was not a decision at
    // all; it was an artifact of the sort being stable.
    //
    // ⚠️ The gap could not have existed at 0.4.1: with no saturating and no
    // transposing vector there was no pair differing in nothing but a flag.
    // Closing GUESS-5 and GUESS-7 CREATED this question, and the vectors
    // answer it while the prose did not.
    writeln!(
        o,
        "    \"tuple_sort_key\": \"{}\",",
        esc(
            "A `<coop>` tuple's leading M, N and K compare as INTEGERS, before \
             any component: as text `10` precedes `9`, so a producer sorting the \
             spelled tuple emits the reverse order. Components then compare by \
             INDEX IN `component_types`, field by field in the order they are \
             spelled -- NOT by comparing the spelled strings, which disagrees: \
             `u32` precedes `u8` as text and follows it in the array. Unnamed \
             `x<n>` escapes sort after every named type, numerically on n. Any \
             trailing flag (`-sat`, `-t`) is the LAST key rather than outside the \
             comparison, so the unflagged form orders before the flagged one. \
             Deduplication happens after sorting and before the length is measured."
        )
    )
    .unwrap();
    // ⚠️ What the threshold measures. `enumeration_bytes` names the quantity
    // but nothing said the field PREFIX is excluded, and three or four bytes
    // decide a borderline field -- the exact case where enumerate-versus-
    // digest is hardest to test and a disagreement is a different cell.
    writeln!(
        o,
        "    \"measured_length_excludes_prefix\": \"{}\",",
        esc(
            "The byte count compared against `digest_threshold_bytes` is the canonical \
             enumeration alone. The field prefix (`cm-`, `cv-`) is NOT counted, and \
             neither is the digest marker that replaces the enumeration above the \
             threshold."
        )
    )
    .unwrap();
    writeln!(o, "    \"digest_threshold_bytes\": {COOP_DIGEST_THRESHOLD}").unwrap();
    writeln!(o, "  }},").unwrap();
}

// ---------------------------------------------------------------------------
// Production half — documentation only, per §6.8-0013. The binding contract is
// `vectors`.
// ---------------------------------------------------------------------------

fn emit_field_spec(o: &mut String) {
    // ⚠️ `input_shape` names the KEYS a caller hands a producer. The prose
    // above names CONCEPTS -- "M-N-K plus four component types" -- and the
    // mapping from those to `m`,`n`,`k`,`a`,`b`,`c`,`result` existed nowhere:
    // measured, ZERO of the input key names appeared delimited anywhere in
    // the manifest. A producer reading different keys CRASHES rather than
    // mismatching, so no vector could ever catch it -- the corpus pins what a
    // token SPELLS and said nothing about what a caller must HAND a producer.
    //
    // ⚠️ ALL REQUIRED also dissolves a second gap rather than choosing a side
    // in it: `tuple_sort_key` says dedup happens after sorting and never says
    // what makes two tuples EQUAL. Spelled-dedup and structural-dedup diverge
    // exactly when a key is absent-versus-explicitly-default. With no absent
    // case the readings coincide, which is what this implementation does --
    // `CoopShape::saturating` is a `bool`, not an `Option<bool>`.
    //
    // Both found by baracuda producing from the published 0.4.6 manifest.
    let specs: [(&str, &str, &str, &str); 5] = [
        (
            "subgroup",
            "sg",
            "The CHOSEN subgroup specialization, not the device envelope. \
             `sg<width>` for a pinned power-of-two width; `sgdyn` for a \
             width-agnostic kernel that reads the width at runtime. One device \
             commonly admits several, and they are different binaries, so a \
             device yields a SET of valid tokens rather than one.",
            "a scalar: the integer subgroup width, or the string \"dynamic\" for the width-agnostic case",
        ),
        (
            "ops",
            "ops-",
            "Subgroup operation classes, spelled as JUXTAPOSED single ASCII \
             letters in the canonical order given by `ops_alphabet`. \
             Juxtaposition is safe only because that alphabet is fixed-width \
             (§6.8-0006). A repeated member is ABSORBED: this field is a SET, \
             so the inputs `[\"b\", \"b\", \"w\"]` and `[\"b\", \"w\"]` spell ONE \
             token, `ops-bw`.",
            "an array of the selected operation-class letters",
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
             f32. Reading one as the other is a silently wrong lowering. A \
             repeated member is ABSORBED here too: `<arith>` is a SET and its \
             input is deduplicated before spelling.",
            "an array of the selected arithmetic capability names",
        ),
        (
            "coop",
            "cm-",
            "Cooperative-MATRIX shapes. Each tuple is M-N-K plus four component \
             types, joined by `-`, followed by a literal `-sat` when and only \
             when the shape saturates; tuples are joined by `,`. Sorted and \
             deduplicated canonically, because driver report order is not \
             guaranteed stable and the token must be byte-identical either way. \
             LENGTH-CONDITIONAL — see the `threshold` and `digest_input` \
             vectors.",
            "an array of objects with keys `m`, `n`, `k`, `a`, `b`, `c`, `result`, `saturating` -- ALL REQUIRED, none omissible",
        ),
        (
            "coopvec",
            "cv-",
            "Cooperative-VECTOR combinations. Each tuple is five component \
             types, followed by a literal `-t` when and only when the \
             combination transposes. Same sort/dedup rule as <coop>, and \
             length-conditional on the same 512-byte threshold — but measured \
             and digested INDEPENDENTLY. The two fields switch on their own \
             bytes and never together, which is why both carry their own \
             threshold vectors.",
            "an array of objects with keys `input`, `input_interpretation`, `matrix_interpretation`, `bias_interpretation`, `result`, `transpose` -- ALL REQUIRED, none omissible",
        ),
    ];

    writeln!(o, "  \"field_spec\": [").unwrap();
    for (i, (field, prefix, note, input_shape)) in specs.iter().enumerate() {
        let comma = if i + 1 == specs.len() { "" } else { "," };
        writeln!(
            o,
            "    {{ \"field\": \"{}\", \"prefix\": \"{}\", \"input_shape\": \"{}\", \"note\": \"{}\" }}{}",
            field,
            prefix,
            esc(input_shape),
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

/// One vector object with its `note` member removed.
///
/// ⚠️ Hand-written because this crate has no dependencies, dev-dependencies
/// included (KISS-CLASSIFY-6.9-0003), so there is no JSON parser to reach for.
///
/// ⚠️ The escape awareness is not decoration. `esc` writes a quote as backslash-
/// quote, and several notes contain quotes -- the `subgroup` note names the input
/// spelling "dynamic" in them. A naive scan to the next quote stops INSIDE those
/// notes and truncates the object mid-string, yielding a shorter digest input
/// that is still a perfectly plausible string.
fn without_note(json: &str) -> String {
    const KEY: &str = "\"note\": \"";
    let Some(at) = json.find(KEY) else {
        return json.to_string();
    };
    let body = &json[at + KEY.len()..];
    let mut end = None;
    let mut escaped = false;
    for (i, c) in body.char_indices() {
        if escaped {
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == '\"' {
            end = Some(i);
            break;
        }
    }
    let end = end.expect("a note string that never closes");
    let rest = &body[end + 1..];
    format!("{}{}", &json[..at], rest.strip_prefix(", ").unwrap_or(rest))
}

/// The bytes `vectors_digest` runs over, per the construction pinned in
/// `declarative` (KISS-CLASSIFY-6.8-0017).
fn vectors_digest_input(vectors: &[String]) -> String {
    let mut s = String::new();
    for v in vectors {
        s.push_str(&without_note(v));
        s.push('\n');
    }
    s
}

/// That COMPONENTS compare by ordinal and not as text, in `<coopvec>`.
///
/// ⚠️ The sibling of `integer_order_vector`, and found the same way: by
/// measuring rather than assuming. After adding a deliberate discriminator for
/// `<coop>`, the same count over `<coopvec>` came back at ONE -- and that one
/// is a threshold vector, pinning component ordering incidentally to its own
/// purpose. The exact defect a foreign reader had just reported for `<coop>`,
/// sitting in the field the fix did not sweep.
///
/// ⚠️ Fixing the instance that TAUGHT the class, and not the class, is the
/// error this file has now made six times. The count is the only thing that
/// catches it, which is why the gate asserts it PER FIELD rather than overall.
fn component_order_vector() -> String {
    coopvec_vector(
        "component_order",
        "Two combinations differing only in their input type, `u8` and `u32`, \
         given u32-first. Pins that components compare by INDEX IN \
         `component_types` -- u8 is index 8 and u32 is index 10 -- and not as \
         text, where `u32` precedes `u8`. A producer sorting spellings emits \
         the reverse of this token.",
        &[
            CoopVecCombo {
                input: ComponentType::U32,
                ..combo(1)
            },
            CoopVecCombo {
                input: ComponentType::U8,
                ..combo(1)
            },
        ],
    )
}

/// That M, N and K compare as INTEGERS and not as text.
///
/// ⚠️ Added because the corpus pinned this in exactly ONE vector, and that
/// vector's `pins` field says `threshold`. Measured across every multi-tuple
/// `<coop>` vector: four agree under numeric and lexicographic order and so
/// discriminate nothing; only the 25-tuple threshold vector separates them,
/// and it does so incidentally to the reason it exists.
///
/// ⚠️ So an editor changing that vector's shape set FOR THRESHOLD REASONS --
/// the whole reason to touch it -- could silently delete the only evidence of
/// integer ordering in the manifest, with nothing to say so. A population of
/// one, load-bearing by accident.
///
/// Found by a foreign party after the rule became precise enough for the
/// omission to stand out: it named components, escapes, flags and dedup, and
/// never the three leading dimensions. Precision made the hole visible.
fn integer_order_vector() -> String {
    coop_vector(
        "integer_order",
        "Two shapes whose M values are 9 and 10, given larger-first. Pins that \
         the leading dimensions compare NUMERICALLY: as text `10` precedes `9`, \
         so a producer sorting the spelled tuple emits the reverse. This is the \
         only vector whose PURPOSE is that distinction -- the threshold vector \
         happens to make it too, which is why this one exists.",
        &[
            CoopShape {
                m: 10,
                ..big_shape(1)
            },
            CoopShape {
                m: 9,
                ..big_shape(1)
            },
        ],
    )
}

/// Where an unnamed `x<n>` escape sorts against a NAMED type.
///
/// ⚠️ Added because nothing exercised it. `tuple_sort_key` claims escapes
/// sort after every named type, and that is true in the code -- `Other(u32)` is
/// the last variant, so the derived ordering puts it last -- but ZERO vectors
/// put a named type and an escape at the SAME position, so no reader could
/// check the claim and no producer could be caught getting it wrong.
///
/// ⚠️ Stating an unexercised rule is how the previous sort-key defect shipped:
/// the prose said `lexicographically` and the corpus sorted by ordinal, and
/// every vector was correct so nothing could disagree with the sentence.
fn escape_order_vector() -> String {
    coopvec_vector(
        "escape_order",
        "A named component and an unnamed `x<n>` escape at the SAME tuple \
         position, given escape-first. Pins that escapes sort AFTER every named \
         type rather than by their spelling -- `f16` and `x5` compare as `f` \
         before `x` by accident here, so a producer sorting spellings passes \
         this vector; what it cannot pass is the `u8`/`u32` pair in the \
         threshold vectors, and the two together pin ordinal order.",
        &[
            CoopVecCombo {
                result: ComponentType::Other(5),
                ..combo(1)
            },
            CoopVecCombo {
                result: ComponentType::F16,
                ..combo(1)
            },
        ],
    )
}

/// The `transpose` flag, which nothing pinned before.
///
// ⚠️ The TRANSPOSE flag, which nothing pinned before.
//
// `field_spec` describes it -- "five component types plus a transpose flag"
// -- and `field_spec` is the half §6.8-0013 calls DOCUMENTATION. All 56
// combos across the other vectors carry `transpose: false`, so the half
// §6.8-0013 calls the NORMATIVE CONTRACT never showed that it existed.
//
// Found by baracuda reproducing this manifest from scratch: they emitted all
// twelve vectors byte-identically WITHOUT ever producing a `-t` suffix,
// because no vector demanded one. Their pass and their blindness had the
// same cause, and neither was visible from the score.
//
// A flag that is usually absent is exactly what a sample omits.
fn transpose_vector() -> String {
    coopvec_vector(
        "transpose",
        "A transposing cooperative-vector combination beside a non-transposing one. Pins that `transpose` is spelled as a trailing `-t` rather than as a sixth component type, and that a clear flag costs nothing: the two combos differ ONLY in the flag, so the suffix is the only difference between their spellings.",
        &[
            CoopVecCombo {
                transpose: true,
                ..combo(1)
            },
            combo(1),
        ],
    )
}

/// The one `<subgroup>` spelling that is not a number.
///
// ⚠️ SUBGROUP, whose DYNAMIC spelling no vector reached.
//
// Same shape as `transpose` immediately above, and sitting right beside
// it: `field_spec` describes width-agnostic compilation, and all twelve
// original vectors passed an INTEGER width, so `sgdyn` existed only in the
// half §6.8-0013 calls documentation. Closing `transpose` without sweeping
// for its shape left the sibling in place — the fix was per-instance and
// the defect was per-class.
//
// The gap is specifically the INPUT representation: a reader who has only
// ever seen `"subgroup": 32` cannot know what a producer passes to ask for
// the dynamic case, so this vector pins the input spelling and the token
// together.
fn dynamic_subgroup_vector() -> String {
    subgroup_vector(
        "subgroup",
        "Width-agnostic compilation. Pins BOTH that the dynamic case spells \
        `sgdyn` rather than a width, AND that its input is the string `dynamic` \
        rather than a number, a null, or an absent field — which no vector \
        passing an integer width could ever show.",
    )
}

/// The `-sat` suffix, which was in neither half of the manifest.
///
// ⚠️ SATURATING, absent from BOTH halves of the manifest — worse than
// `transpose`, which at least `field_spec` described.
//
// Reported by baracuda as "`saturating` is not spelled into the coop
// tuple". That conclusion is wrong — it spells a trailing `-sat`,
// measured — but the REASONING was sound and the action it called for was
// right: they observed that all twelve vectors carry `saturating: false`,
// so a producer that appended the field would still match every one. That
// holds whether the suffix exists or not, which is exactly why the vectors
// could not answer it.
//
// ⚠️ And nothing in the prose could have corrected them: `field_spec`
// said "M-N-K plus four component types" and stopped, so the suffix
// appeared in NO vector and NO description. A reader building a producer
// from this manifest could not have emitted `-sat` by any route. The prose
// is fixed above; this vector fixes the normative half.
fn saturating_vector() -> String {
    coop_vector(
        "saturating",
        "Two shapes differing ONLY in `saturating`, one true and one false. The \
         saturating one takes a trailing `-sat` and the two remain SEPARATE \
         tuples, so a producer that ignored the field would emit one tuple where \
         the vocabulary emits two. Pins the suffix, its position after the \
         component types, and that the non-saturating form carries no marker at \
         all.",
        &[
            CoopShape {
                saturating: true,
                ..big_shape(1)
            },
            big_shape(1),
        ],
    )
}

/// The tie-break for shapes agreeing on m, n and k.
///
// ⚠️ The TIE-BREAK FIELD ORDER, for shapes agreeing on m, n and k.
//
// The only duplicate-bearing vector is `dedup`, whose repeated shapes are
// IDENTICAL — they exercise deduplication and say nothing about ordering,
// because collapsing them needs no comparison beyond equality. Nothing
// pinned what happens when two DISTINCT shapes tie on the dimensions.
//
// These two swap `a` and `b`, so the field order is what decides: under
// (a, b, c, result) the f16-first shape sorts first; under (b, a, ...) the
// other does. A vector where only one field varies cannot separate the two.
fn tiebreak_vector() -> String {
    coop_vector(
        "tiebreak",
        "Two distinct shapes agreeing on m, n and k, given in non-canonical \
         order, differing by a SWAP of `a` and `b`. Pins that the tie-break \
         descends (a, b, c, result) in that order rather than any other \
         permutation — a swap is the only input shape that can, since every field \
         order agrees when just one field differs.",
        &[
            CoopShape {
                a: ComponentType::F32,
                b: ComponentType::F16,
                ..big_shape(1)
            },
            CoopShape {
                a: ComponentType::F16,
                b: ComponentType::F32,
                ..big_shape(1)
            },
        ],
    )
}

/// The vectors that pin a FLAG or a NON-NUMERIC spelling.
///
/// ⚠️ Grouped because they share a failure mode, not because they share a
/// field. Each one pins something that is USUALLY ABSENT from a sample:
/// `transpose` and `saturating` were false in every shape the vocabulary had
/// ever enumerated, `sgdyn` is the one `<subgroup>` spelling that is not a
/// number, and the tie-break only exists when two shapes collide on m, n, k.
///
/// A flag that is usually absent is exactly what a sample omits, and all four
/// were found by a foreign party reproducing this manifest — not here.
fn flag_vectors() -> Vec<String> {
    vec![
        transpose_vector(),
        dynamic_subgroup_vector(),
        saturating_vector(),
        tiebreak_vector(),
        escape_order_vector(),
        integer_order_vector(),
        component_order_vector(),
    ]
}

/// The `order` and `dedup` vectors, for both length-conditional fields.
///
/// Pinned separately per field because the two canonicalize independently:
/// an implementation that sorted one and not the other would pass a
/// `<coop>`-only vector set.
fn order_and_dedup_vectors() -> Vec<String> {
    // -- order: a non-canonically-ordered input and its canonical output.
    let unsorted = vec![big_shape(3), big_shape(1), big_shape(2)];
    let mut v = vec![coop_vector(
        "order",
        "Shapes presented in non-canonical order. A producer that emitted \
         driver order would differ from an honest peer on the same device, and \
         under byte-exact matching that is a different cell rather than a \
         degraded answer.",
        &unsorted,
    )];

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

    v
}

/// The two SET-VALUED scalar fields, `<ops>` and `<arith>`.
fn set_field_vectors() -> Vec<String> {
    // -- the two SET-VALUED scalar fields. Nothing pinned these before: every
    //    vector above carries `ops-none` and `arith-none`, so a consumer could
    //    read the alphabet and still not know how two members are joined.
    let mut v = vec![set_field_vector(
        "arith",
        "Two arithmetic capabilities, given in NON-CANONICAL order. Pins that \
         `<arith>` joins its names with `-` and never juxtaposes them, and that \
         the canonical order is the alphabet's own order rather than the order \
         a device reported them in. Both matter because matching is byte-exact: \
         `arith-i8-f16` is a different cell, not a differently-written same cell.",
        OpClasses::NONE,
        Arith::FLOAT16 | Arith::INT8,
    )];
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

    // ⚠️ That a REPEATED member is absorbed. `field_spec` said "sorted and
    // deduplicated" for `<coop>` and `<coopvec>` and said NOTHING about these
    // two, and NO vector carried a duplicate -- so a producer that never
    // deduplicated passed every one of them.
    //
    // ⚠️ Found by baracuda running a producer that does not deduplicate: 17 of
    // 17 pass, with a control mutation firing at 14/3 so the zero is a
    // measurement rather than a non-run. The gap PREDATED the release they
    // found it in -- they checked, rather than reporting it as created by the
    // previous fix, which would have confirmed their own published prediction.
    //
    // Their own producer wrote `set(vals)` and never recorded the choice: a
    // DECLARED ledger cannot record a decision you did not notice making.
    v.push(dedup_set_vector(
        "ops",
        "An input REPEATING a member. `<ops>` is a SET, so `b` twice is \
         absorbed and the token spells it once -- a producer that concatenated \
         its input would emit `ops-bbw`, which under §6.8-0002 is a different \
         cell rather than a differently-written same one. NEITHER set-dedup vector derives \
         its input from its token, because a token spells each member once \
         and a duplicate cannot be recovered from it.",
        OpClasses::BASIC | OpClasses::ROTATE,
        Arith::NONE,
        &["b", "b", "w"],
    ));

    // ⚠️ The SAME rule, on a field with a DIFFERENT SPELLING. One vector
    // could not stand for both: `<ops>` juxtaposes single letters, so a
    // concatenating producer emits a DOUBLED LETTER (`ops-bbw`), while
    // `<arith>` joins named parts with `-` and emits a REPEATED PART
    // (`arith-f16-f16-i8`). The defect is one rule and two surfaces, and a
    // reader who generalized from the `<ops>` vector would be generalizing
    // from the spelling rather than from the rule.
    //
    // Found by review on #84, not by a gate: the `<arith>` field_spec note
    // said ABSORBED while only `<ops>` carried a vector. The test now reads
    // WHICH fields claim absorption and requires a vector for each, so a
    // third such claim cannot arrive uncovered.
    v.push(dedup_set_vector(
        "arith",
        "An input REPEATING a member, in the field whose members are NAMED \
         rather than single letters. `<arith>` is a SET, so `f16` twice is \
         absorbed and the token spells it once -- a producer that \
         concatenated its input would emit `arith-f16-f16-i8`, which under \
         §6.8-0002 is a different cell rather than a differently-written \
         same one.",
        OpClasses::NONE,
        Arith::FLOAT16 | Arith::INT8,
        &["f16", "f16", "i8"],
    ));

    v
}

/// `threshold` and `digest_input`, per field, at and immediately across.
fn threshold_vectors() -> Vec<String> {
    let mut v: Vec<String> = Vec::new();

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

    v
}

/// Build the vector list.
///
/// Split from the printing so the COUNT is available to `sufficiency`,
/// which is emitted earlier in the file and previously spelled the number
/// out by hand.
fn build_vectors() -> Vec<String> {
    let mut v = order_and_dedup_vectors();
    v.extend(flag_vectors());
    v.extend(set_field_vectors());
    v.extend(threshold_vectors());
    v
}

fn emit_vectors(o: &mut String) {
    let v = build_vectors();
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

/// A vector for `<subgroup>` when the width is not a number.
///
/// Every other vector in this file hardcodes `"subgroup": 32`, which is why
/// the dynamic case needs its own constructor rather than another
/// `coop_vector` call: the gap being closed is in the INPUT half, and a helper
/// that cannot vary the input cannot express it.
fn subgroup_vector(pins: &str, note: &str) -> String {
    let token = VulkanTarget {
        subgroup: Subgroup::Dynamic,
        ops: OpClasses::NONE,
        arith: Arith::NONE,
        coop: CoopMatrix::None,
        coopvec: CoopVector::None,
    }
    .to_token();
    format!(
        "{{ \"pins\": \"{}\", \"field\": \"subgroup\", \"note\": \"{}\", \"input\": {{ \"subgroup\": \"dynamic\", \"ops\": [], \"arith\": [], \"coop\": [], \"coopvec\": [] }}, \"token\": \"{}\" }}",
        pins,
        esc(note),
        token
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

/// A set field whose INPUT repeats a member, pinning that the repeat is
/// absorbed.
///
/// ⚠️ The one set vector whose input is NOT derived from its token, and that
/// is precisely the point. Every other one derives input FROM the token so the
/// two halves cannot disagree -- deliberately -- and that safety property is
/// exactly what makes a duplicate unrepresentable, since a token spells each
/// member once. The property that prevents one class of error prevented this
/// vector from existing.
fn dedup_set_vector(
    field: &str,
    note: &str,
    ops: OpClasses,
    arith: Arith,
    input: &[&str],
) -> String {
    let token = VulkanTarget {
        subgroup: Subgroup::Fixed(32),
        ops,
        arith,
        coop: CoopMatrix::None,
        coopvec: CoopVector::None,
    }
    .to_token();
    let unique: std::collections::BTreeSet<&&str> = input.iter().collect();
    assert!(
        unique.len() < input.len(),
        "a dedup vector's input must REPEAT a member or it pins nothing about \
         absorption -- got {input:?}"
    );
    let members = input
        .iter()
        .map(|m| format!("\"{m}\""))
        .collect::<Vec<_>>()
        .join(",");
    let (ops_in, arith_in) = if field == "ops" {
        (members, String::new())
    } else {
        (String::new(), members)
    };
    format!(
        "{{ \"pins\": \"set-dedup\", \"field\": \"{}\", \"note\": \"{}\", \"input\": {{ \"subgroup\": 32, \"ops\": [{}], \"arith\": [{}], \"coop\": [], \"coopvec\": [] }}, \"token\": \"{}\" }}",
        field,
        esc(note),
        ops_in,
        arith_in,
        token
    )
}

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
        "{{ \"pins\": \"{}\", \"threshold_of\": \"coop\", \"note\": \"{}\", \"enumeration_bytes\": {}, \"threshold_bytes\": {}, \"input\": {{ \"subgroup\": 32, \"ops\": [], \"arith\": [], \"coop\": [{}], \"coopvec\": [] }}, \"token\": \"{}\" }}",
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
        "{{ \"pins\": \"{}\", \"threshold_of\": \"coopvec\", \"note\": \"{}\", \"enumeration_bytes\": {}, \"threshold_bytes\": {}, \"input\": {{ \"subgroup\": 32, \"ops\": [], \"arith\": [], \"coop\": [], \"coopvec\": [{}] }}, \"token\": \"{}\" }}",
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
    against, and a stamp proves BINDING, not CURRENCY. And one limit the \
    vectors cannot express at all: `ops_alphabet` and `arith_names` are \
    THEMSELVES in lexicographic order, so no vector can distinguish sorting by \
    the declared array index from sorting the spellings as text for those two \
    fields, and a producer implementing the wrong one passes every vector here. \
    `component_types` is NOT alphabetical -- `f64` precedes `bf16` -- and is \
    therefore discriminable, which is why it carries deliberate ordering \
    vectors and the other two cannot. The rule is the array's own order in all \
    three cases; this corpus can only PROVE it for the third. Finally, what \
    a reproduction's residue is and is not. A residue counts what ONE READER \
    NOTICED CHOOSING -- never what this prose leaves open. It is a lower bound \
    of unknown tightness, so a falling series across releases is evidence about \
    READERS rather than about this document. An earlier version of this note \
    reported such a series as a trend; the party who produced the numbers \
    refuted it by finding a gap their own count had missed, in a release they \
    had already scored. Treat a passing reproduction as evidence about the \
    VECTORS, and never as evidence that this prose is complete.";

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
