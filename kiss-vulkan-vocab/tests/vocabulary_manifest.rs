//! The committed §6.8-0008 manifest must be what this crate emits today, and
//! must satisfy the envelope KISS pins.
//!
//! # The freshness gate
//!
//! §6.8-0011 requires a manifest to "**agree** with its prose annex under an
//! emit-and-`git diff --exit-code` freshness gate". This file is the emit half,
//! run as a test so it fails in CI rather than only when someone remembers to
//! regenerate.
//!
//! It `#[path]`-includes the generator rather than re-implementing it or
//! shelling out to `cargo run`. Re-implementing would compare the artifact
//! against a second copy of the logic, which is what an emit-and-compare gate
//! exists to rule out; shelling out would make the test depend on a nested
//! cargo invocation holding the build lock.
//!
//! # What this file does NOT close
//!
//! **Agreement with `spec/namespaces/vulkan.md` is a separate obligation and is
//! not tested here.** §6.8-0011 splits provenance from agreement — *"Provenance
//! names the producer; agreement is a relation between two artifacts, and
//! neither settles which is the source."* This gate proves the manifest is
//! fresh against the crate. It does not prove the crate agrees with the annex,
//! which is the gap `registered_namespace.rs` calls "a ratchet, not a proof".
//! Saying so here rather than letting a green run imply otherwise.

#[path = "../examples/emit_vocabulary_manifest.rs"]
#[allow(dead_code)]
mod emitter;

use std::path::PathBuf;

fn committed_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("manifest")
        .join("vulkan-vocabulary.json")
}

fn committed() -> String {
    let p = committed_path();
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("cannot read the committed manifest at {}: {e}", p.display()))
        // The file is committed with LF; a Windows checkout without the
        // .gitattributes rule would otherwise fail this on line endings alone
        // and send the reader hunting for a content change that isn't there.
        .replace("\r\n", "\n")
}

/// The emit-and-compare freshness gate.
#[test]
fn committed_manifest_is_byte_identical_to_a_fresh_emission() {
    let fresh = emitter::manifest();
    let on_disk = committed();

    if fresh != on_disk {
        // Report the first divergence rather than dumping 23KB of JSON at
        // someone — a diff nobody reads is a failure message that only says
        // "something changed".
        // Byte offset of the first difference, then windows clamped OUTWARD to
        // char boundaries. This manifest is full of em-dashes, so a raw
        // `at ± 60` lands mid-character often, and the old version fell back to
        // the literal string "<boundary>" when it did — a diagnostic that
        // silently degrades exactly when it is needed.
        let at = fresh
            .as_bytes()
            .iter()
            .zip(on_disk.as_bytes())
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| fresh.len().min(on_disk.len()));
        let window = |s: &str| {
            let mut start = at.saturating_sub(60).min(s.len());
            while start > 0 && !s.is_char_boundary(start) {
                start -= 1;
            }
            let mut end = (at + 60).min(s.len());
            while end < s.len() && !s.is_char_boundary(end) {
                end += 1;
            }
            s[start..end].replace('\n', "⏎")
        };
        panic!(
            "the committed vocabulary manifest is stale.\n\n\
             First divergence at byte {at}.\n  emitted:   …{}…\n  committed: …{}…\n\n\
             Regenerate it in the same change that altered the vocabulary:\n  \
             cargo run --example emit_vocabulary_manifest -p kiss-vulkan-vocab \\\n    \
             > kiss-vulkan-vocab/manifest/vulkan-vocabulary.json\n\n\
             Do not edit the manifest by hand. It is the machine-readable form \
             of a vocabulary another project binds against, and a hand-edit \
             makes it disagree with the crate that is supposed to produce it — \
             which is the drift this gate exists to catch.",
            window(&fresh),
            window(&on_disk)
        );
    }
}

/// §6.8-0008's envelope: every required field present, `schema` recognised.
#[test]
fn manifest_carries_every_field_the_envelope_requires() {
    let m = committed();
    for key in [
        "schema",
        "namespace",
        "vocabulary_version",
        "generated_from",
        "kind",
        "grammar",
        "coverage_note",
    ] {
        assert!(
            m.contains(&format!("\"{key}\":")),
            "the manifest is missing the required envelope field {key:?}. \
             §6.8-0008 lists these explicitly and says a reader MUST reject \
             with a typed decline a manifest missing any of them — so omitting \
             one does not degrade the artifact, it invalidates it."
        );
    }
    assert!(
        m.contains("\"schema\": \"kiss-namespace-vocabulary-v1\""),
        "the manifest's schema id is not `kiss-namespace-vocabulary-v1`; a \
         reader MUST reject an unrecognized schema."
    );
    assert!(
        m.contains("\"kind\": \"generated\""),
        "`vulkan` is a grammar over an open product space, so its kind is \
         `generated`. §6.8-0010 makes `kind` an OPEN set and requires a reader \
         encountering an unknown one to decline rather than guess the nearer \
         of the two it knows."
    );
}

/// `vocabulary_version` must be an **integer**, and the check must be able to
/// fail on a float.
///
/// §6.8-0008 states the reason inline: *"an integer — a gate that truncates a
/// fractional value is not a gate."* A clause that anticipates its own defeat
/// deserves a test that does too, so this asserts the emitted form is an
/// integer literal **and** demonstrates on a fabricated float that the check
/// rejects it. Asserting only the happy path would leave a check that cannot
/// tell the two apart.
#[test]
fn vocabulary_version_is_an_integer_and_a_float_would_be_rejected() {
    let m = committed();

    let value = m
        .split("\"vocabulary_version\":")
        .nth(1)
        .and_then(|t| t.split(',').next())
        .map(str::trim)
        .expect("manifest carries a vocabulary_version");

    assert!(
        is_integer_literal(value),
        "vocabulary_version is {value:?}, which is not an integer literal. \
         §6.8-0008: \"a gate that truncates a fractional value is not a gate.\" \
         A quoted value fails for the same reason — a consumer comparing it \
         numerically would parse it first, and a parse that truncates is the \
         defeat the clause names."
    );
    assert_eq!(
        value,
        kiss_vulkan_vocab::VOCABULARY_VERSION.to_string(),
        "the manifest's vocabulary_version disagrees with the crate's"
    );

    // Negative controls: the predicate must reject what the clause warns about.
    for bad in ["4.0", "4.5", "\"4\"", "4e0", " 4 .0", "+4", "0x4"] {
        assert!(
            !is_integer_literal(bad),
            "is_integer_literal accepted {bad:?}; the gate would truncate it \
             and report success, which is exactly the failure §6.8-0008 names"
        );
    }
    for good in ["0", "4", "17", "4294967295"] {
        assert!(
            is_integer_literal(good),
            "is_integer_literal rejected {good:?}, which is a valid version"
        );
    }
}

/// A bare decimal integer: no sign, no point, no exponent, no quotes, no radix
/// prefix, and no leading zero on a multi-digit value.
fn is_integer_literal(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()) && (s.len() == 1 || !s.starts_with('0'))
}

/// §6.8-0013: for `kind: generated` the vectors are the normative contract, and
/// the required coverage is enumerated. A namespace with no length-conditional
/// field may omit `threshold`/`digest_input` **and must say so** — `vulkan` has
/// two, so both must be present for both.
#[test]
fn vectors_cover_every_canonicalization_the_clause_requires() {
    let m = committed();

    for pins in ["order", "dedup", "threshold", "digest_input"] {
        assert!(
            m.contains(&format!("\"pins\": \"{pins}\"")),
            "no vector pins {pins:?}. §6.8-0013 enumerates the required \
             coverage for `kind: generated`, and a missing tag is a coverage \
             hole rather than a smaller vector set: the grammar cannot validate \
             canonicalization, so whatever the vectors omit is unpinned."
        );
    }

    // Both length-conditional fields, both sides of the boundary, both digests.
    for field in ["coop", "coopvec"] {
        let count = m.match_indices(&format!("\"field\": \"{field}\"")).count();
        assert!(
            count >= 5,
            "field {field:?} has only {count} vectors; expected at least five \
             (order, dedup, threshold-at, threshold-across, digest_input). The \
             two length-conditional fields measure and digest INDEPENDENTLY, so \
             covering one does not cover the other — an implementation that \
             switched `coop` correctly and `coopvec` early would pass a \
             single-field vector set."
        );
    }

    assert!(
        m.contains("\"enumeration_bytes\": 512") && m.contains("\"enumeration_bytes\": 513"),
        "the threshold vectors do not sit at 512 and 513 bytes. §6.8-0013 wants \
         each length-conditional field presented AT and IMMEDIATELY ACROSS its \
         boundary, \"so both forms are pinned at the exact byte count that flips \
         them\". A straddling pair that never lands on the boundary cannot tell \
         `>` from `>=`."
    );
}

/// The digest is over the pinned `digest_input`, and the pinned input is the
/// same string the threshold measured.
///
/// §6.8-0013 wants this separable from the threshold "so a producer may
/// disagree about *whether* to digest but never about *what* is digested".
/// Those are different failures and only one of them is visible in the token —
/// the token carries the hash, so a producer digesting the wrong string emits a
/// well-formed token that matches nothing.
#[test]
fn each_digest_is_the_hash_of_the_digest_input_it_pins() {
    let m = committed();
    let mut checked = 0;

    for chunk in m.split("\"pins\": \"digest_input\"").skip(1) {
        let entry = chunk.split('}').next().unwrap_or_default();
        let field = between(entry, "\"digest_input\": \"", "\", \"digest_input_bytes\"")
            .expect("digest_input vector carries its input string");
        let declared_len: usize = between(entry, "\"digest_input_bytes\": ", ",")
            .and_then(|s| s.trim().parse().ok())
            .expect("digest_input vector carries its byte count");
        let digest =
            between(entry, "\"digest\": \"", "\"").expect("digest_input vector carries a digest");

        let unescaped = field.replace("\\\"", "\"").replace("\\\\", "\\");
        assert_eq!(
            unescaped.len(),
            declared_len,
            "a digest_input vector declares {declared_len} bytes but carries \
             {}; the length a consumer measures against the threshold and the \
             string it hashes must be the same string",
            unescaped.len()
        );
        assert_eq!(
            digest,
            format!(
                "fnv1a64-{:016x}",
                kiss_vulkan_vocab::fnv1a64(unescaped.as_bytes())
            ),
            "a digest_input vector's digest is not the FNV-1a-64 of the input \
             it pins. This is the one disagreement invisible in a token — the \
             token carries only the hash, so a producer that digests the wrong \
             string emits a well-formed token matching nothing."
        );
        checked += 1;
    }

    assert_eq!(
        checked, 2,
        "expected one digest_input vector per length-conditional field, found \
         {checked}. `vulkan` has two such fields and they digest independently."
    );
}

fn between<'a>(hay: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let s = hay.find(start)? + start.len();
    let rest = &hay[s..];
    let e = rest.find(end)?;
    Some(&rest[..e])
}
