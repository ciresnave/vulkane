// SPDX-License-Identifier: MIT OR Apache-2.0
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
// The example is compiled INTO this test so the emitter is exercised rather
// than a copy of it. The test calls a subset of what the example defines, and
// the rest is live in the example binary -- so this says "unused by this
// test", not "unused".
#[allow(dead_code)]
mod emitter;

use std::path::PathBuf;

fn committed_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("manifest")
        .join("vulkan-vocabulary.json")
}

/// The committed manifest, **exactly as it sits on disk**.
///
/// This used to normalize `\r\n` before comparing, and that normalization was
/// hiding a real defect rather than tolerating a cosmetic one. `.gitattributes`
/// did not pin this path, so a Windows checkout stored LF and checked out CRLF
/// — the committed artifact was **not** byte-identical to a fresh emission, and
/// the emit-and-`git diff --exit-code` gate KISS-CLASSIFY-6.8-0011 asks for
/// could not have been armed here at all. The test passed the whole time,
/// because it was comparing something neither party actually had.
///
/// The path is pinned `text eol=lf` now, so the comparison can be exact.
fn committed() -> String {
    let p = committed_path();
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("cannot read the committed manifest at {}: {e}", p.display()))
}

/// The emit-and-compare freshness gate.
#[test]
fn committed_manifest_is_byte_identical_to_a_fresh_emission() {
    let fresh = emitter::manifest();
    let on_disk = committed();

    // Line endings get their own message. Reporting "stale" for a CRLF checkout
    // would send the reader hunting for a content change that does not exist,
    // and the fix is completely different from the fix for real staleness.
    if fresh != on_disk && fresh == on_disk.replace("\r\n", "\n") {
        // The claim is exactly what the condition above tested: the two agree
        // once CRLF is folded to LF. That admits a mixed-ending file as well as
        // a uniformly-CRLF one, so the message says "contains CRLF" rather than
        // "is CRLF" — the wider claim would be true in the common case and
        // wrong in the one that is harder to diagnose.
        let crlf = on_disk.matches("\r\n").count();
        panic!(
            "the committed manifest differs from a fresh emission ONLY in line \
             endings — it contains {crlf} CRLF line ending(s) and the emitter \
             produces LF.\n\n\
             The content is fine. `.gitattributes` should be pinning\n  \
             kiss-vulkan-vocab/manifest/*.json text eol=lf\n\
             and this checkout is not honouring it. Re-materialize the file:\n  \
             rm {} && git checkout -- {}\n\n\
             This is not cosmetic: the artifact is byte-compared, and a copy \
             carrying CRLF cannot satisfy an emit-and-`git diff --exit-code` \
             gate.",
            committed_path().display(),
            committed_path().display()
        );
    }

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

    // ⚠️ NAME THE SUBJECT BEFORE READING IT.
    //
    // The read below takes the text after the FIRST `vocabulary_version`,
    // which is the top-level field only because the emitter happens to write
    // it before `sufficiency`. Nothing asserted that, and §6.8-0017 REQUIRES
    // a second `vocabulary_version` inside `sufficiency` the day `status`
    // becomes `demonstrated` -- which is the point of that whole apparatus.
    //
    // So this is a latent subject-selection defect with a KNOWN trigger: the
    // instrument would keep returning a real value, from a real field, and
    // report on whichever one the file happened to order first.
    //
    // Prompted by baracuda retracting a finding measured against the wrong
    // kernel: their selector took `variants[1]` and every downstream control
    // honestly confirmed a real, different subject. A CONTROL VALIDATES THE
    // INSTRUMENT, NEVER THE SUBJECT -- only naming what it points at reaches
    // this class.
    let occurrences = m.matches("\"vocabulary_version\":").count();
    assert_eq!(
        occurrences, 1,
        "the manifest carries {occurrences} `vocabulary_version` keys, and the \
         read below takes the FIRST. That was unambiguous while there was one. \
         §6.8-0017 adds `sufficiency.vocabulary_version` when `status` becomes \
         `demonstrated`, so disambiguate by scope here rather than relying on \
         emission order -- the value returned stays well-formed either way, \
         which is why nothing else would notice."
    );

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
        // ⚠️ Counts BOTH spellings. A threshold vector names its field with
        // `threshold_of` (KISS-CLASSIFY-6.8-0016, merged 2026-09-06); every other
        // vector kind still uses `field`, because `threshold_of` on a vector that
        // pins no boundary would be a category error.
        //
        // This test broke the moment that rename landed, which is the point worth
        // recording: it is a READER KEYED ON THE OLD NAME, in the same repository
        // as the emitter, and nothing connected the two but a string. That is the
        // failure mode the clause rename exists to prevent between projects,
        // reproduced inside one crate within minutes.
        let count = m.match_indices(&format!("\"field\": \"{field}\"")).count()
            + m.match_indices(&format!("\"threshold_of\": \"{field}\""))
                .count();
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

/// The committed manifest must satisfy KISS-CLASSIFY-6.8-0016's own rejection
/// conditions, checked here rather than trusted.
///
/// §6.8-0016 (merged 2026-09-06) requires that a `threshold`-tagged vector carry
/// `threshold_of` and `enumeration_bytes`, and that **a reader MUST reject** a
/// manifest in which, for any value of `threshold_of`, the threshold vectors do
/// not include a pair whose `enumeration_bytes` are N and N+1, **or** in which
/// that pair's two emitted `token` values are equal.
///
/// ⚠️ This asserts the conditions a KISS reader will apply to us, from our side,
/// so a divergence fails here rather than in somebody else's decline. The clause
/// exists because **adjacency does not establish straddling** — enumerations of
/// 3 and 4 bytes are adjacent and both far below a 512-byte boundary — and the
/// differing-token condition is what makes the byte pair mean anything: a
/// declared boundary that flips no behaviour is a wrong boundary.
///
/// Hand-parsed: this crate has no dependencies, dev-dependencies included, which
/// §6.9-0003 requires and `zero_dependency.rs` enforces.
#[test]
fn threshold_vectors_straddle_their_boundary_per_6_8_0016() {
    let text = std::fs::read_to_string(committed_path()).expect("committed manifest");

    fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
        let at = line.find(&format!("\"{key}\": "))? + key.len() + 4;
        let rest = &line[at..];
        Some(if let Some(r) = rest.strip_prefix('"') {
            &r[..r.find('"')?]
        } else {
            let end = rest.find([',', '}']).unwrap_or(rest.len());
            rest[..end].trim()
        })
    }

    let rows: Vec<(&str, u64, &str)> = text
        .lines()
        .filter(|l| l.contains("\"pins\": \"threshold\""))
        .map(|l| {
            let of = field(l, "threshold_of").unwrap_or_else(|| {
                panic!(
                    "a threshold vector without `threshold_of`; §6.8-0016 makes it MUST: {l:.120}"
                )
            });
            let bytes: u64 = field(l, "enumeration_bytes")
                .unwrap_or_else(|| panic!("threshold vector without `enumeration_bytes`: {l:.120}"))
                .parse()
                .expect("enumeration_bytes is a number");
            let token = field(l, "token")
                .unwrap_or_else(|| panic!("threshold vector without `token`: {l:.120}"));
            (of, bytes, token)
        })
        .collect();

    // Positive control: a parser that silently matched nothing would satisfy
    // every assertion below by having nothing to check.
    assert!(
        rows.len() >= 2,
        "found {} threshold vectors; the manifest has length-conditional fields, so \
         too few means the parser broke rather than the manifest shrank",
        rows.len()
    );

    let mut fields: Vec<&str> = rows.iter().map(|(f, _, _)| *f).collect();
    fields.sort_unstable();
    fields.dedup();
    for f in fields {
        let mut group: Vec<&(&str, u64, &str)> = rows.iter().filter(|(o, _, _)| *o == f).collect();
        group.sort_by_key(|(_, b, _)| *b);
        let pair = group
            .windows(2)
            .find(|w| w[1].1 == w[0].1 + 1)
            .unwrap_or_else(|| {
                panic!(
                    "threshold_of={f:?} has no N/N+1 pair; byte counts are {:?}. \
                     Adjacency in the LIST is not adjacency in the BYTES -- a reader \
                     MUST reject this under §6.8-0016.",
                    group.iter().map(|(_, b, _)| *b).collect::<Vec<_>>()
                )
            });
        assert_ne!(
            pair[0].2, pair[1].2,
            "threshold_of={f:?}: the N/N+1 pair at {} and {} emits the SAME token, so the \
             declared boundary flips nothing and is a wrong boundary",
            pair[0].1, pair[1].1
        );
    }
}

/// The `sufficiency` block, hand-sliced.
///
/// Shared by the two tests that read it. This crate has no dependencies,
/// dev-dependencies included, so there is no JSON parser to reach for -- and
/// the slicer needs its own positive control for exactly that reason.
fn sufficiency_block() -> String {
    let text = std::fs::read_to_string(committed_path()).expect("committed manifest");
    let start = text
        .find("\"sufficiency\"")
        .expect("§6.8-0017: `sufficiency` is absent. A reader MUST reject this.");
    let block: String = text[start..]
        .lines()
        .take_while(|l| !l.trim_start().starts_with("},"))
        .collect::<Vec<_>>()
        .join("\n");
    // Positive control: a slice that captured nothing would satisfy an
    // absence-based check by having nothing to contradict it.
    assert!(
        block.len() > 30 && block.contains('{'),
        "the sufficiency block did not parse out ({block:?}); the extractor broke \
         rather than the manifest shrinking"
    );
    block
}

/// One scalar value out of a hand-sliced JSON block, quoted or bare.
fn block_value(block: &str, key: &str) -> Option<String> {
    let at = block.find(&format!("\"{key}\": "))? + key.len() + 4;
    let rest = &block[at..];
    Some(if let Some(r) = rest.strip_prefix('"') {
        r[..r.find('"')?].to_owned()
    } else {
        rest[..rest.find([',', '\n']).unwrap_or(rest.len())]
            .trim()
            .to_owned()
    })
}

/// The manifest must satisfy KISS-CLASSIFY-6.8-0017's rejection conditions,
/// checked here rather than trusted.
///
/// A reader MUST reject a manifest whose `sufficiency` is absent, whose
/// `status` is **absent** or is any other token, or which claims `demonstrated`
/// without all five of `reproduced_by`, `artifact`, `vocabulary_version`,
/// `guessed` and `derived`.
///
/// ⚠️ The absent-`status` arm is checked SEPARATELY from the wrong-token arm,
/// mirroring the clause's own reason for stating them separately: an
/// enumeration of wrong values does not reach a value that is not there. That
/// distinction is not pedantry — it is how §6.8-0017 came to mandate a field it
/// never named, and the clause says so about itself.
///
/// Hand-parsed: this crate has no dependencies, dev-dependencies included.
#[test]
fn sufficiency_is_declared_per_6_8_0017() {
    let block = sufficiency_block();
    let value = |key: &str| block_value(&block, key);

    // Arm 1: absent. Stated apart from arm 2 on purpose.
    let status = value("status").expect(
        "§6.8-0017: `sufficiency` carries no `status`. This is the ABSENT arm, and it is \
         the one an enumeration of wrong tokens does not reach.",
    );
    // Arm 2: any other token.
    assert!(
        status == "demonstrated" || status == "unexercised",
        "§6.8-0017: `status` is {status:?}; exactly `demonstrated` or `unexercised`"
    );

    // Arm 3: `demonstrated` without all five.
    if status == "demonstrated" {
        for k in [
            "reproduced_by",
            "artifact",
            "vocabulary_version",
            "guessed",
            "derived",
        ] {
            assert!(
                value(k).is_some() || block.contains(&format!("\"{k}\"")),
                "§6.8-0017: `status` is `demonstrated` without `{k}`. All five are \
                 required, and `guessed`/`derived` MAY be empty but MUST be present \
                 — an empty array is the strong claim, and a reader is entitled to \
                 see it made."
            );
        }
    }
}

/// The `sufficiency` note must name the vector count this manifest carries.
///
/// Lifted out of `sufficiency_is_declared_per_6_8_0017` rather than left as a
/// fourth arm: that test already carried three arms about the SHAPE of the
/// block, and this one is about its CONTENT agreeing with the rest of the
/// file. A failure in either should name which of the two went wrong.
#[test]
fn the_sufficiency_note_names_the_manifests_own_vector_count() {
    let block = sufficiency_block();

    // Arm 4: the note must NAME the vector count this manifest actually carries.
    //
    // ⚠️ Not hypothetical. The first `sufficiency` note said "This manifest
    // has thirteen" and was wrong one commit later, when three vectors landed
    // and `vocabulary_version` did not move. The block whose entire purpose is
    // to say "what ships here is not what was reproduced" had gone stale about
    // what ships here.
    //
    // Stated as a POSITIVE requirement rather than a bound. The first draft of
    // this arm asserted `n <= vectors` over every digit run in the note, which
    // a stale `13` satisfies as comfortably as a correct `16` — a gate that
    // cannot fail the defect it is named after. Requiring the true count to
    // APPEAR has no such hole: there is exactly one number that satisfies it.
    let vectors = committed().matches("\"pins\": ").count();
    assert!(
        vectors > 0,
        "positive control: the vector extractor found none, so the check below \
         would be comparing against zero and any note at all would pass it"
    );
    let named: Vec<usize> = block
        .split(|c: char| !c.is_ascii_digit())
        .filter(|w| !w.is_empty())
        .filter_map(|w| w.parse().ok())
        .collect();
    assert!(
        named.contains(&vectors),
        "the sufficiency note names {named:?} but this manifest carries \
         {vectors} vectors, and the note must say so. A number here is a claim \
         about WHICH ARTIFACT is shipping; a stale one asserts a reproduction \
         of something that is not being shipped — the exact failure §6.8-0017 \
         exists to prevent, arriving from inside the field meant to prevent it."
    );
}

/// Shared by the three vectors that pin a spelling nothing had produced.
///
/// ⚠️ These exist because the first draft of the `saturating` note asserted
/// the OPPOSITE of what this vocabulary does. It claimed `saturating` was not
/// spelled into the tuple and that two shapes differing only in that field
/// would collapse into one; the emitted token spells a trailing `-sat` and
/// keeps both. The note was wrong for the same reason the gap existed — no
/// vector had ever produced the token, so nothing could contradict a
/// plausible sentence about it.
///
/// A vector is self-consistent BY CONSTRUCTION here: the emitter derives the
/// token from the same code that spells it, so input and token can never
/// disagree. That is deliberate, and it is also why a vector cannot catch a
/// wrong NOTE. These tests are the only thing standing between the halves.
fn vector_line(pins: &str) -> String {
    committed()
        .lines()
        .find(|l| l.contains(&format!("\"pins\": \"{pins}\"")))
        .unwrap_or_else(|| {
            panic!(
                "no vector pins {pins:?}. This is the ABSENT arm: the test \
                 cannot check a spelling that no vector produces, and a \
                 silently-skipped check is what let the wrong note ship."
            )
        })
        .to_string()
}

fn vector_token(pins: &str) -> String {
    let line = vector_line(pins);
    between(&line, "\"token\": \"", "\"")
        .unwrap_or_else(|| panic!("vector {pins:?} carries no token"))
        .to_string()
}

/// The `<coop>` tuples of a vector's token, split on `,`.
fn coop_tuples(pins: &str) -> Vec<String> {
    let token = vector_token(pins);
    token
        .split('.')
        .find(|p| p.starts_with("cm-"))
        .unwrap_or_else(|| panic!("token {token:?} has no <coop> field"))
        .trim_start_matches("cm-")
        .split(',')
        .map(str::to_string)
        .collect()
}

#[test]
fn the_subgroup_vector_pins_the_dynamic_spelling() {
    let sg = vector_token("subgroup");
    assert!(
        sg.starts_with("vulkan:sgdyn."),
        "the subgroup vector's token is {sg:?}, which does not spell `sgdyn`. \
         The width-agnostic case is the one spelling `<subgroup>` has that is \
         not a number, so a token carrying a width here pins nothing new."
    );
    assert!(
        vector_line("subgroup").contains("\"subgroup\": \"dynamic\""),
        "the subgroup vector's INPUT does not spell the dynamic case as the \
         string \"dynamic\". The gap this vector closes is in the input half — \
         a reader who has only seen `\"subgroup\": 32` cannot know what to pass \
         — so an input spelled any other way leaves the gap open."
    );
}

#[test]
fn the_saturating_vector_pins_the_sat_suffix() {
    let tuples = coop_tuples("saturating");
    assert_eq!(
        tuples.len(),
        2,
        "the saturating vector spells {} <coop> tuple(s), expected 2. Two \
         shapes differing only in `saturating` are DISTINCT, and a producer \
         that dropped the field would emit one tuple where this emits two — \
         under byte-exact matching, a different cell rather than a near miss.",
        tuples.len()
    );
    assert_eq!(
        tuples.iter().filter(|t| t.ends_with("-sat")).count(),
        1,
        "expected exactly one of {tuples:?} to end in `-sat`. The suffix marks \
         the saturating form and the non-saturating form carries no marker at \
         all; two markers or none would both mean the flag is not what \
         distinguishes them."
    );
}

#[test]
fn the_tiebreak_vector_pins_the_field_order() {
    let tuples = coop_tuples("tiebreak");
    assert_eq!(
        tuples.len(),
        2,
        "the tiebreak vector spells {} <coop> tuple(s), expected 2 — two \
         DISTINCT shapes agreeing on m, n and k. If they collapsed, they were \
         not distinct and the vector pins nothing about ordering.",
        tuples.len()
    );
    let field = |t: &str, i: usize| t.split('-').nth(i).unwrap_or_default().to_string();
    // m-n-k-a-b-c-result: `a` is index 3, `b` is index 4.
    assert_eq!(
        (field(&tuples[0], 3), field(&tuples[0], 4)),
        ("f16".to_string(), "f32".to_string()),
        "the first tiebreak tuple is {:?}; expected `a`=f16 and `b`=f32. The \
         two shapes SWAP `a` and `b`, so this ordering is what proves the \
         tie-break descends (a, b, c, result) rather than some other \
         permutation — under (b, a, ...) the other shape would sort first.",
        tuples[0]
    );
    assert_eq!(
        (field(&tuples[1], 3), field(&tuples[1], 4)),
        ("f32".to_string(), "f16".to_string()),
        "the second tiebreak tuple is {:?}; expected the swapped pair. Both \
         tuples must be present and in this order — checking only the first \
         would pass on a vector that dropped the second entirely.",
        tuples[1]
    );
}

/// The `vectors` array's element texts, exactly as the manifest renders them.
///
/// ⚠️ Sliced from the raw text, not from a parse, because the pinned
/// construction is defined over the rendering. A reader already holds these
/// bytes; re-serializing them would reintroduce key order, whitespace, number
/// formatting and unicode escaping, none of which any clause settles.
fn vector_elements(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut inside = false;
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with("\"vectors\": [") {
            inside = true;
            continue;
        }
        if inside {
            if t == "]" || t == "]," {
                break;
            }
            out.push(t.strip_suffix(',').unwrap_or(t).to_string());
        }
    }
    out
}

/// One element with its `note` member — and the `, ` separating it — removed.
///
/// ⚠️ A second implementation ON PURPOSE. The emitter's `without_note` is in
/// scope here (this file `#[path]`-includes the generator), and calling it
/// would compare the emitter against itself and pass for any construction
/// whatsoever, including one no second party could follow. The digest exists so
/// that two parties agree, so the check has to be a second party or it is not a
/// check.
fn element_without_note(e: &str) -> String {
    const KEY: &str = "\"note\": \"";
    let Some(at) = e.find(KEY) else {
        return e.to_string();
    };
    let body = &e[at + KEY.len()..];
    let mut escaped = false;
    let mut end = None;
    for (i, c) in body.char_indices() {
        if escaped {
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == '"' {
            end = Some(i);
            break;
        }
    }
    let end = end.expect("a note string that never closes");
    let rest = &body[end + 1..];
    format!("{}{}", &e[..at], rest.strip_prefix(", ").unwrap_or(rest))
}

/// The construction MUST be pinned, and MUST say enough to be followed.
///
/// The pin is the load-bearing half: KISS pins the algorithm and constants and
/// deliberately does NOT define this input, because §6.8-0007's input is "the
/// canonical enumeration string it REPLACES" and the `vectors` array replaces
/// nothing. An unpinned digest is a number every party computes differently
/// while each compares only against its own.
#[test]
fn the_vectors_digest_construction_is_pinned_in_declarative() {
    let text = committed();
    let pin = between(&text, "\"vectors_digest_input\": \"", "\"").expect(
        "§6.8-0017: `declarative` carries no `vectors_digest_input`. Without it \
         a reader cannot rebuild the digest, and the value beside it can only \
         ever agree with itself.",
    );
    assert!(
        pin.contains("UTF-8"),
        "the pinned construction {pin:?} does not state its character encoding, \
         which §6.8-0017 requires by name. ⚠️ It is unobservable in this corpus \
         precisely BECAUSE excluding notes leaves the input pure ASCII — which \
         is the same accident that made a foreign reader guess §6.8-0007's \
         encoding one level down. A corpus that cannot expose a question is the \
         reason to state its answer, not a reason to omit it."
    );
    assert!(
        pin.contains("note"),
        "the pinned construction {pin:?} does not say what happens to `note`. \
         §6.8-0017 requires every note excluded: a prose correction changes \
         nothing a producer must produce, and a currency check that goes stale \
         on a typo trains its reader to re-run for nothing."
    );
}

/// A reader MUST be able to rebuild `vectors_digest` from `vectors` by the
/// pinned construction and compare byte-exact (KISS-CLASSIFY-6.8-0017).
///
/// The same rebuild was performed once outside this repository, in another
/// language, written from the pinned sentence alone; it produced this value.
#[test]
fn vectors_digest_is_rebuildable_from_the_pinned_construction() {
    let text = committed();
    let elements = vector_elements(&text);
    // Positive control: an empty slice digests the empty string to the FNV
    // basis, which is a perfectly well-formed sixteen-hex-digit answer.
    assert!(
        elements.len() > 10,
        "sliced {} vector elements; the extractor broke rather than the \
         manifest shrinking, and an empty slice still produces a plausible digest",
        elements.len()
    );

    let input: String = elements
        .iter()
        .map(|e| element_without_note(e) + "\n")
        .collect();
    assert!(
        !input.contains("\"note\""),
        "the rebuilt input still contains a `note` member, so the stripper \
         matched nothing and the digest below is over the wrong bytes"
    );

    let rebuilt = format!(
        "fnv1a64-{:016x}",
        kiss_vulkan_vocab::fnv1a64(input.as_bytes())
    );
    let carried = between(&text, "\"vectors_digest\": \"", "\"")
        .expect("the manifest carries no top-level `vectors_digest`");
    assert_eq!(
        rebuilt, carried,
        "rebuilding `vectors_digest` by the pinned construction gives \
         {rebuilt}, but the manifest carries {carried}. §6.8-0017 requires the \
         rebuild to compare BYTE-EXACT, so a producer may disagree about \
         whether the digest is current but never about what was digested. A \
         mismatch means the pinned sentence does not describe what this emitter \
         does — and that sentence is the only thing a second party has."
    );

    // ⚠️ Negative control. The input is pure ASCII only BECAUSE notes are
    // excluded, and exclusion is a clause requirement. If keeping them produced
    // the same digest, the exclusion would be inert and the agreement above
    // would say nothing about it.
    let with_notes: String = elements.iter().map(|e| e.clone() + "\n").collect();
    assert_ne!(
        format!(
            "fnv1a64-{:016x}",
            kiss_vulkan_vocab::fnv1a64(with_notes.as_bytes())
        ),
        carried,
        "digesting WITH the notes yields the same value as without them, so \
         either the notes are empty or the stripper did nothing. Excluding them \
         would then be an untested requirement rather than a behaviour, and a \
         prose fix would silently move the digest."
    );
}

/// The angle-bracket convention must be STATED, not left to be inferred.
///
/// ⚠️ Found by baracuda rebuilding `vectors_digest` from this manifest alone.
/// They read `digest_marker` — value `fnv1a64-<hex16>` — as a PREFIX rather
/// than a template and emitted `fnv1a64-<hex16>-5fb4518c42b73202`. It was the
/// only error in an otherwise byte-exact reproduction, and the field name
/// argues for their reading: a thing called a MARKER sounds like literal text
/// while its value is a format string.
///
/// Renaming that one field would fix the instance. This gate is the class:
/// several values carry `<...>` and a reader had nothing telling them the
/// brackets were not literal. Same shape as `sgdyn` sitting beside `transpose`
/// unnoticed while `transpose` was being closed.
#[test]
fn the_placeholder_notation_is_stated_rather_than_inferred() {
    let text = committed();

    let notation = between(&text, "\"notation\": \"", "\"").expect(
        "§6.8-0017: `declarative` carries no `notation`. Several values in this \
         manifest use `<...>` placeholders, and without a statement of the \
         convention a reader must infer it — one who infers wrong emits \
         well-formed bytes that match nothing, which is the failure byte-exact \
         matching gives no warning about.",
    );
    assert!(
        notation.contains("PLACEHOLDER") && notation.contains("outside the brackets"),
        "the notation entry {notation:?} does not say both what an \
         angle-bracketed name IS and what the text around it is. Saying only \
         the first leaves a reader guessing whether the surrounding characters \
         are part of the pattern."
    );

    // ⚠️ Positive control, and it is the whole point of the gate. A convention
    // statement describing a convention nobody uses is decoration; this asserts
    // there are real placeholders for it to govern, so the test fails if the
    // notation outlives the thing it explains.
    let bracketed = text
        .lines()
        .filter(|l| l.contains("\": \"") && l.contains('<') && !l.contains("\"notation\""))
        .count();
    assert!(
        bracketed >= 3,
        "only {bracketed} manifest values contain `<`, so the notation entry \
         explains almost nothing. Either the placeholders were removed — in \
         which case delete the notation with them — or this extractor stopped \
         finding them, which would let a future manifest drop every placeholder \
         and still pass."
    );

    // The field that caused the misreading must be covered by name, because it
    // is the one whose NAME argues against the convention.
    assert!(
        notation.contains("fnv1a64-<hex16>"),
        "the notation entry {notation:?} does not name `digest_marker`'s value. \
         That is the field a foreign reader actually got wrong, and a general \
         rule stated without the instance it exists for is the easiest kind to \
         read past."
    );
}

/// Composing `empty_set_spelling` against the declared prefixes must reproduce
/// the spellings the tokens actually use.
///
/// ⚠️ This entry read `<prefix>-none` and was WRONG, not merely underspecified.
/// `ops` declares prefix `ops-` and `coop` declares `cm-`, so the rule composed
/// to `ops--none` and `cm--none` — a double dash, for four of the five fields —
/// while the tokens spell `ops-none` and `cm-none`. A reader following the
/// declarative half against the declared prefixes gets bytes that match
/// nothing, and under §6.8-0002 that is a different cell rather than an error.
///
/// Raised by a foreign reader as *unpinned*; measured here it was
/// *contradictory*, which is the stronger fault and the one no amount of
/// additional vectors would have surfaced — every vector spells the RIGHT
/// bytes, so only composing the stated rule reveals that it disagrees.
#[test]
fn the_empty_set_rule_reproduces_the_spellings_the_tokens_use() {
    let text = committed();

    // What the tokens actually spell, e.g. `ops-none`, `cm-none`.
    let mut observed: Vec<String> = Vec::new();
    for (at, _) in text.match_indices("-none") {
        let head = &text[..at];
        let start = head
            .rfind(|c: char| !c.is_ascii_alphanumeric() && c != '-')
            .map_or(0, |i| i + 1);
        let word = format!("{}-none", &head[start..]);
        if word.len() > 5 && !observed.contains(&word) {
            observed.push(word);
        }
    }
    // Positive control: with nothing observed, every assertion below is vacuous.
    assert!(
        observed.len() >= 3,
        "found only {} empty-set spellings in the manifest ({observed:?}); the \
         extractor broke rather than the vocabulary shrinking, and an empty \
         list satisfies the check below by having nothing to contradict it",
        observed.len()
    );

    // Every prefix the manifest declares, composed per the stated rule.
    let mut composed: Vec<String> = Vec::new();
    let rule = between(&text, "\"empty_set_spelling\": \"", "\"")
        .expect("`declarative` carries no `empty_set_spelling`");
    assert!(
        rule.contains("<prefix>"),
        "`empty_set_spelling` {rule:?} is not a composable template. A test \
         cannot fail when the rule it is meant to check is the wrong one."
    );

    const PK: &str = "\"prefix\": \"";
    for (at, _) in text.match_indices(PK) {
        let p = &text[at + PK.len()..];
        if let Some(end) = p.find('"') {
            composed.push(rule.replace("<prefix>", &p[..end]));
        }
    }
    assert!(
        !composed.is_empty(),
        "no `prefix` values found, so the rule was composed against nothing"
    );

    for word in &observed {
        assert!(
            composed.contains(word),
            "the tokens spell {word:?}, which composing `empty_set_spelling` \
             against the declared prefixes does not produce — it yields \
             {composed:?}. The declarative half and the vectors disagree, and a \
             reader who has only the declarative half emits the losing spelling."
        );
    }
}

/// The sort key must name the flag, and the tokens must show the direction.
///
/// ⚠️ `sorted and deduplicated canonically` never said whether the trailing
/// flag participates in the KEY. A foreign producer ranked the components and
/// omitted it, so two tuples differing only in `-sat` compared EQUAL and a
/// stable sort emitted them in input order — an output that was not a decision
/// at all. The gap could not have existed before the saturating and transposing
/// vectors were added: closing two gaps created a third.
#[test]
fn the_sort_key_names_the_flag_and_the_tokens_agree() {
    let text = committed();
    let key = between(&text, "\"tuple_sort_key\": \"", "\"")
        .expect("§6.8-0017: `declarative` carries no `tuple_sort_key`");
    assert!(
        key.contains("-sat") && key.contains("-t") && key.contains("LAST"),
        "the sort key {key:?} does not name the trailing flags and their \
         position. Naming the components alone is exactly the reading that \
         makes two flag-differing tuples compare equal."
    );

    // And the behaviour it describes, measured rather than trusted: in both
    // flag-bearing vectors the UNFLAGGED tuple is spelled first.
    for (pins, flag, prefix) in [("saturating", "-sat", "cm-"), ("transpose", "-t", "cv-")] {
        let line = vector_line(pins);
        let token = between(&line, "\"token\": \"", "\"").expect("token");
        let field = token
            .split('.')
            .find(|p| p.starts_with(prefix) && !p.ends_with("-none"))
            .expect("a tuple field");
        let tuples: Vec<&str> = field.split_once('-').unwrap().1.split(',').collect();
        assert_eq!(
            tuples.len(),
            2,
            "the {pins} vector no longer carries the flag-differing pair that \
             pins this ordering"
        );
        assert!(
            !tuples[0].ends_with(flag) && tuples[1].ends_with(flag),
            "in the {pins} vector the flagged tuple is spelled first ({tuples:?}), \
             contradicting `tuple_sort_key`. The prose and the vectors must not \
             disagree: a reader trusting the prose emits the other order."
        );
    }
}

/// Every rule-shaped value must be on the checked list.
///
/// ⚠️ baracuda's class, after `empty_set_spelling` turned out to CONTRADICT the
/// vectors rather than merely under-specify them: *a declarative rule that
/// disagrees with the corpus is invisible to every vector-based check, because
/// the vectors are all correct.* A vector suite validates the vectors against a
/// producer; it never validates a RULE against the vectors. Only composing the
/// stated rule and comparing its output to the corpus finds it.
///
/// ⚠️ The list is DECLARED, not accumulated. An accumulating check cannot report
/// a rule it never reached — the same construction that hid a residue item from
/// its own ledger — so a new placeholder-bearing value that nobody wrote a check
/// for fails here instead of passing silently.
#[test]
fn every_rule_shaped_value_is_on_the_checked_list() {
    const CHECKED: &[&str] = &[
        "grammar",
        "unnamed_component_escape",
        "empty_set_spelling",
        "digest_marker",
        "unnamed_component_escape_order",
        // EXECUTED rather than composed: it is prose describing an
        // algorithm, so the instrument is
        // `executing_the_stated_sort_rule_reproduces_the_token_order`,
        // which runs the rule and compares. The composition gate cannot
        // reach a rule that is not a template.
        "tuple_sort_key",
        "notation",
        "vectors_digest_input",
        // Prose that MENTIONS placeholders rather than being one. These
        // three cannot be composed against the corpus: `notation` defines the
        // convention, `vectors_digest_input` is an instruction, and
        // `coverage_note` quotes examples. Listed so the gate stays a
        // DECLARATION rather than a filter that silently drops what it
        // cannot handle.
        "coverage_note",
    ];

    let text = committed();
    let mut found: Vec<String> = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if !t.contains('<') || !t.starts_with('"') {
            continue;
        }
        if let Some(end) = t[1..].find('"') {
            found.push(t[1..=end].to_string());
        }
    }
    // Positive control: nothing found means the scan broke, and every
    // assertion below would be satisfied by an empty list.
    assert!(
        found.len() >= 4,
        "found only {found:?} rule-shaped values; the scan broke rather than \
         the manifest losing its placeholders"
    );
    for key in &found {
        assert!(
            CHECKED.contains(&key.as_str()),
            "`{key}` carries a `<...>` placeholder but is not on the checked \
             list. Either compose it against the corpus in \
             `the_composable_rules_produce_what_the_tokens_spell`, or add it \
             here with a comment saying why it cannot be composed. A rule that \
             nothing composes is one nobody can discover is wrong."
        );
    }
}

/// The composable rules must produce what the tokens actually spell.
#[test]
fn the_composable_rules_produce_what_the_tokens_spell() {
    let text = committed();
    let rule = |k: &str| {
        between(&text, &format!("\"{k}\": \""), "\"")
            .unwrap_or_else(|| panic!("no `{k}` in the manifest"))
            .to_string()
    };

    // `grammar` — every token must have the shape it states.
    let grammar = rule("grammar");
    let fields = grammar.matches('.').count() + 1;
    let mut tokens = 0;
    for (at, _) in text.match_indices("\"token\": \"vulkan:") {
        let tok = &text[at + "\"token\": \"".len()..];
        let tok = &tok[..tok.find('"').expect("token end")];
        assert_eq!(
            tok.matches('.').count() + 1,
            fields,
            "token {tok:?} has {} fields; `grammar` states {fields}. \
             §6.8-0002 matches byte-exact, so a field-count disagreement \
             between the rule and the corpus is a cell nobody can reach.",
            tok.matches('.').count() + 1
        );
        tokens += 1;
    }
    assert!(
        tokens >= 10,
        "only {tokens} tokens checked against `grammar`"
    );

    // `digest_marker` -- `fnv1a64-<hex16>` means sixteen lowercase hex digits.
    //
    // ⚠️ Scoped to where a digest actually LIVES: token values and the
    // top-level `vectors_digest`. A whole-file scan matched the rule stating
    // ITSELF -- `fnv1a64-<hex16>` in `digest_marker`, and the same string
    // quoted inside `notation` as "`fnv1a64-` followed by sixteen hex digits".
    // Skipping those by their next character would have been a filter tuned to
    // the prose that happens to exist today; naming the corpus is the fix.
    let marker = rule("digest_marker");
    let lit = marker.split('<').next().expect("marker literal");
    let mut corpus: Vec<String> = vec![rule("vectors_digest")];
    for (at, _) in text.match_indices("\"token\": \"") {
        let t = &text[at + "\"token\": \"".len()..];
        corpus.push(t[..t.find('"').expect("token end")].to_string());
    }
    let mut digests = 0;
    for entry in &corpus {
        for (at, _) in entry.match_indices(lit) {
            let hex: String = entry[at + lit.len()..].chars().take(16).collect();
            assert!(
                hex.len() == 16
                    && hex
                        .chars()
                        .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
                "in {entry:?} the text after {lit:?} is {hex:?}, which is not \
                 sixteen lowercase hex digits as `digest_marker` states"
            );
            digests += 1;
        }
    }
    assert!(
        digests >= 2,
        "only {digests} digests checked against the marker"
    );

    // `unnamed_component_escape` — `x<n>` means a literal `x` then digits.
    let escape = rule("unnamed_component_escape");
    let head = escape.split('<').next().expect("escape literal");
    assert_eq!(
        head, "x",
        "`unnamed_component_escape` is {escape:?}; its literal head is {head:?} \
         and the tokens spell unnamed components with `x`"
    );
}

/// Rank one tuple field the way `tuple_sort_key` says.
///
/// M, N and K compare as INTEGERS; everything else compares as a component,
/// by its index in `component_types`, with unnamed `x<n>` escapes after every
/// named type. A component spelling never parses as a bare integer, so the two
/// ranks never meet at the same position.
fn field_rank(types: &[&str], f: &str) -> (u8, usize) {
    if let Ok(n) = f.parse::<usize>() {
        return (0, n);
    }
    if let Some(i) = types.iter().position(|t| *t == f) {
        return (1, i);
    }
    let n: usize = f
        .strip_prefix('x')
        .and_then(|d| d.parse().ok())
        .unwrap_or_else(|| {
            panic!("field {f:?} is not an integer, a named type, or an `x<n>` escape")
        });
    (1, types.len() + n)
}

/// Executing the stated sort rule must reproduce the order the tokens carry.
///
/// ⚠️ `tuple_sort_key` has been wrong twice. It read "lexicographically by
/// their spelled components" while the vocabulary sorts by ordinal — `u32`
/// precedes `u8` as text and follows it in `component_types`. And it described
/// components, escapes, flags and dedup while saying nothing at all about the
/// three leading DIMENSIONS of a `<coop>` tuple, which compare as integers.
///
/// ⚠️ This covers BOTH fields. An earlier version checked `<coopvec>` only, on
/// the judgement that `<coop>`'s leading integers "would say nothing new". That
/// judgement was wrong, and a foreign reader found what it would have said.
#[test]
fn executing_the_stated_sort_rule_reproduces_the_token_order() {
    let text = committed();

    // Executing a hardcoded rule proves what the CORPUS does and nothing about
    // what the manifest CLAIMS — the blindness that let `empty_set_spelling`
    // ship wrong. These tie the stated rule to the executed one.
    let stated = between(&text, "\"tuple_sort_key\": \"", "\"").expect("tuple_sort_key");
    for needle in [
        "component_types",
        "NOT by comparing the spelled",
        "INTEGERS",
    ] {
        assert!(
            stated.contains(needle),
            "`tuple_sort_key` is {stated:?} and does not mention {needle:?}. Each \
             clause was added because a producer got that case wrong; dropping one \
             restores a defect that shipped."
        );
    }

    let types_raw = between(&text, "\"component_types\": [", "]").expect("component_types");
    let types: Vec<&str> = types_raw
        .split(',')
        .map(|s| s.trim().trim_matches('"'))
        .filter(|s| !s.is_empty())
        .collect();
    assert!(types.len() > 10, "component_types parsed as {types:?}");

    let key = |t: &str| {
        let (body, flag) = match t.strip_suffix("-t").or_else(|| t.strip_suffix("-sat")) {
            Some(b) => (b, 1u8),
            None => (t, 0),
        };
        let mut k: Vec<(u8, usize)> = body.split('-').map(|c| field_rank(&types, c)).collect();
        k.push((flag, 0));
        k
    };

    let mut checked = 0;
    let mut discriminating = std::collections::BTreeMap::<&str, usize>::new();
    for prefix in [".cm-", ".cv-"] {
        for (at, _) in text.match_indices(prefix) {
            let f = &text[at + 1..];
            let f = &f[..f.find(['.', '"']).expect("field end")];
            let body = &f[3..];
            if body == "none" || body.starts_with("fnv1a64-") {
                continue;
            }
            let emitted: Vec<&str> = body.split(',').collect();
            if emitted.len() < 2 {
                continue;
            }
            let mut sorted = emitted.clone();
            sorted.sort_by_key(|t| key(t));
            assert_eq!(
                sorted, emitted,
                "executing `tuple_sort_key` on {emitted:?} yields {sorted:?}. The \
                 stated rule and the corpus disagree, and a producer following the \
                 prose emits a different enumeration — and, above the threshold, a \
                 different digest."
            );
            let mut lexicographic = emitted.clone();
            lexicographic.sort_unstable();
            if lexicographic != emitted {
                *discriminating.entry(prefix).or_default() += 1;
            }
            checked += 1;
        }
    }
    assert!(
        checked >= 5,
        "only {checked} multi-tuple fields checked across <coop> and <coopvec>"
    );
    // ⚠️ Without this the assertion above passes under a plain string sort too,
    // and would say nothing about which rule the manifest states. Integer
    // ordering was pinned by exactly ONE vector before this gate existed, and
    // that vector's `pins` says `threshold` — load-bearing by accident.
    // ⚠️ PER FIELD, not overall. An overall count of two is satisfied by two
    // discriminators in one field and none in the other -- which is exactly the
    // state this repository was in when a foreign reader found that `<coop>`
    // integer ordering rested on a single vector whose `pins` says `threshold`.
    // Measuring the same count over `<coopvec>` afterwards returned ONE.
    for prefix in [".cm-", ".cv-"] {
        let n = discriminating.get(prefix).copied().unwrap_or(0);
        assert!(
            n >= 2,
            "only {n} {prefix} vector(s) distinguish the stated order from a plain \
             string sort. One is a POPULATION OF ONE: the vector carrying it can be \
             edited for its own stated purpose and silently take the evidence with \
             it, and nothing connects the two."
        );
    }
}

/// The two alphabets that cannot discriminate their own ordering must still
/// coincide with it — and this fails the day one stops, which is the day a
/// discriminating vector becomes both possible and necessary.
///
/// ⚠️ `ops_alphabet` and `arith_names` are themselves in lexicographic order, so
/// NO input can distinguish "sort by the declared array's index" from "sort the
/// spellings as text" for those fields. A producer implementing the wrong rule
/// passes every vector that could ever be written. `component_types` is not
/// alphabetical, which is precisely why it is the field that produced two
/// ordering defects and now carries two deliberate vectors.
///
/// ⚠️ That is a third category. An unstated rule is undiscriminated; an
/// incidentally-witnessed one is fragile; this one is STATED, RIGHT, and
/// UNFALSIFIABLE — the corpus cannot test it even in principle. It has a
/// trigger rather than a fragility, and this is the trigger.
///
/// Found by a foreign party reproducing the published manifest.
#[test]
fn the_alphabets_that_cannot_discriminate_their_own_order_still_coincide() {
    let text = committed();
    let array = |k: &str| -> Vec<String> {
        between(&text, &format!("\"{k}\": ["), "]")
            .unwrap_or_else(|| panic!("no `{k}` array"))
            .split(',')
            .map(|s| s.trim().trim_matches('"').to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };

    let k = "arith_names";
    let v = array(k);
    assert!(v.len() > 3, "`{k}` parsed as {v:?}");
    let mut sorted = v.clone();
    sorted.sort();
    assert_eq!(
        v, sorted,
        "`{k}` is no longer in lexicographic order. Until now the declared \
         order and the text order COINCIDED, so no vector could distinguish \
         them and a producer sorting spellings passed everything. That is no \
         longer true: add a vector whose canonical output differs under the \
         two rules, or a producer implementing the wrong one is now silently \
         wrong and nothing here will say so."
    );

    // `ops_alphabet` is a STRING of juxtaposed letters, not an array — the same
    // property, a different shape, and a helper written for arrays would have
    // skipped it silently.
    let ops = between(&text, "\"ops_alphabet\": \"", "\"").expect("ops_alphabet");
    let mut ops_sorted: Vec<char> = ops.chars().collect();
    ops_sorted.sort_unstable();
    assert_eq!(
        ops.chars().collect::<Vec<_>>(),
        ops_sorted,
        "`ops_alphabet` is no longer in letter order. The same warning applies: \
         until now no vector could tell index order from text order for this \
         field, and now one must."
    );

    // ⚠️ Positive control, and it is the whole reason this test can be trusted:
    // `component_types` must NOT coincide. If every array were sorted the
    // assertions above would hold for a comparison that never discriminates
    // anything, and this test would be measuring nothing.
    let ct = array("component_types");
    let mut ct_sorted = ct.clone();
    ct_sorted.sort();
    assert_ne!(
        ct, ct_sorted,
        "`component_types` is now in lexicographic order too, so the comparison \
         above no longer distinguishes anything and its passing says nothing. \
         This is the control: one array must disagree with its own text order, \
         or the check is vacuous."
    );
}

/// Every vector's input must carry exactly the keys `input_shape` declares.
///
/// ⚠️ The key names existed nowhere before this: `field_spec` names CONCEPTS
/// ("M-N-K plus four component types") and the input JSON uses `m`,`n`,`k`,
/// `a`,`b`,`c`,`result`,`saturating`. Measured, ZERO of those names appeared
/// delimited anywhere in the manifest. A producer reading different keys
/// CRASHES rather than mismatching, so no vector could ever have caught it —
/// the corpus pins what a token SPELLS and said nothing about what a caller
/// must HAND a producer.
///
/// ⚠️ ALL REQUIRED also dissolves a second gap instead of picking a side in it.
/// `tuple_sort_key` says dedup happens after sorting and never says what makes
/// two tuples EQUAL; spelled-dedup and structural-dedup diverge exactly when a
/// key is absent-versus-explicitly-default. With no absent case they coincide.
///
/// This reads the DECLARED key list out of `input_shape` rather than hardcoding
/// it, because a gate that restates the rule cannot fail when the rule is wrong
/// — the defect that let `empty_set_spelling` ship contradicting its own corpus.
#[test]
fn every_vector_input_carries_the_keys_input_shape_declares() {
    let text = committed();

    let declared = |field: &str| -> Vec<String> {
        let at = text
            .find(&format!("\"field\": \"{field}\""))
            .unwrap_or_else(|| panic!("no field_spec entry for {field}"));
        let shape = between(&text[at..], "\"input_shape\": \"", "\"")
            .unwrap_or_else(|| panic!("{field} declares no `input_shape`"));
        shape
            .split('`')
            .skip(1)
            .step_by(2)
            .map(str::to_string)
            .collect()
    };

    for (field, prefix) in [("coop", "\"coop\": ["), ("coopvec", "\"coopvec\": [")] {
        let keys = declared(field);
        assert!(
            keys.len() >= 5,
            "`input_shape` for {field} names {keys:?}; expected the full key list. \
             If it stopped naming them, a producer is back to guessing what a \
             caller hands it — and no vector can catch that, because the wrong \
             keys crash rather than mismatch."
        );

        // Every object in every vector's input for this field.
        let mut checked = 0;
        for (at, _) in text.match_indices(prefix) {
            let rest = &text[at + prefix.len()..];
            let arr = &rest[..rest.find(']').expect("array end")];
            for obj in arr.split('{').skip(1) {
                let body = &obj[..obj.find('}').unwrap_or(obj.len())];
                for k in &keys {
                    assert!(
                        body.contains(&format!("\"{k}\"")),
                        "a {field} input object omits `{k}`: {body:?}. \
                         `input_shape` declares every key REQUIRED, and an \
                         omitted one reopens the dedup-equality question — \
                         spelled-dedup and structural-dedup differ exactly when \
                         a key is absent versus explicitly default."
                    );
                }
                checked += 1;
            }
        }
        // Positive control: no objects found means the extractor broke, and
        // every assertion above was satisfied by having nothing to check.
        assert!(
            checked >= 2,
            "found only {checked} {field} input object(s); the extractor broke \
             rather than the corpus shrinking"
        );
    }
}

/// A repeated set member must be absorbed, and a vector must prove it.
///
/// ⚠️ `field_spec` said "sorted and deduplicated" for `<coop>` and `<coopvec>`
/// and said NOTHING about `<ops>`/`<arith>`, and NO vector carried a duplicate —
/// so a producer that never deduplicated passed all seventeen. Found by a
/// foreign party running exactly that producer, with a control mutation firing
/// so the zero was a measurement rather than a non-run.
///
/// ⚠️ Their own producer wrote `set(vals)` and never recorded the choice. A
/// DECLARED ledger fixes "a path nothing reaches" and does nothing for "a
/// decision I made without seeing it as one" — `set(vals)` felt like typing.
/// Every field whose `field_spec` note CLAIMS a repeated member is absorbed.
///
/// ⚠️ Read from the manifest rather than listed in the test. The previous
/// version checked `<ops>` because `<ops>` was what it named, so an `<arith>`
/// claim shipped with no vector and stayed green -- caught in review on #84.
/// A hardcoded list closes that instance and leaves the third claim free.
fn fields_claiming_absorption(text: &str) -> Vec<&str> {
    text.lines()
        .filter(|l| l.contains("\"input_shape\"") && l.contains("ABSORBED"))
        .filter_map(|l| between(l, "\"field\": \"", "\""))
        .collect()
}

/// The DISTINCT members of a set-dedup vector's input, having checked that the
/// input actually repeats one -- without a repeat the vector discriminates
/// nothing, since a concatenating and an absorbing producer agree on it.
fn dedup_vector_members(field: &str, line: &str) -> Vec<String> {
    let raw = between(line, &format!("\"{field}\": ["), "]")
        .unwrap_or_else(|| panic!("the `{field}` dedup vector carries no `{field}` input"));
    let members: Vec<&str> = raw
        .split(',')
        .map(|m| m.trim().trim_matches('"'))
        .filter(|m| !m.is_empty())
        .collect();
    let unique: std::collections::BTreeSet<&str> = members.iter().copied().collect();
    assert!(
        unique.len() < members.len(),
        "the `{field}` set-dedup vector's input {members:?} carries no repeat, \
         so it discriminates nothing: a concatenating producer and an absorbing \
         one emit the same token from it."
    );
    unique.into_iter().map(str::to_string).collect()
}

/// ⚠️ The instrument states its own precondition. Counting a member's
/// occurrences in the token tells absorption from concatenation only while no
/// member is a substring of another. That holds for `arith_names` today and
/// trivially for single letters, but a future name could make the count
/// silently wrong -- so it is asserted rather than assumed.
fn assert_members_are_mutually_distinguishable(field: &str, unique: &[String]) {
    for a in unique {
        for b in unique {
            assert!(
                a == b || !b.contains(a.as_str()),
                "`{a}` is a substring of `{b}` in `{field}`, so counting \
                 occurrences cannot tell absorption from concatenation. Split \
                 on the field's separator instead."
            );
        }
    }
}

/// The field's segment of a vector's token, with its `<field>-` prefix removed.
fn spelled_field(field: &str, line: &str) -> String {
    let token = between(line, "\"token\": \"", "\"")
        .unwrap_or_else(|| panic!("the `{field}` dedup vector carries no token"));
    let prefix = format!("{field}-");
    token
        .split('.')
        .find(|f| f.starts_with(&prefix))
        .unwrap_or_else(|| panic!("the token spells no `{field}` field"))
        .trim_start_matches(&prefix)
        .to_string()
}

#[test]
fn a_repeated_set_member_is_absorbed_and_a_vector_proves_it() {
    let text = committed();

    let claimed = fields_claiming_absorption(&text);
    assert!(
        !claimed.is_empty(),
        "no field_spec note claims a repeated member is ABSORBED. Either the \
         rule left the prose -- in which case the vectors below pin behaviour \
         no producer is told about -- or this check can no longer find the \
         claim, and a check that cannot find its subject cannot fail."
    );

    for field in &claimed {
        let needle = format!("\"pins\": \"set-dedup\", \"field\": \"{field}\"");
        let line = text
            .lines()
            .find(|l| l.contains(&needle))
            .unwrap_or_else(|| {
                panic!(
                    "the `{field}` field_spec note says a repeated member is \
                 ABSORBED, and no vector pins it. A producer that concatenates \
                 its input passes every vector while violating a documented \
                 rule -- under §6.8-0002 `{field}` spelled twice is a different \
                 cell, not a differently-written same one."
                )
            });

        let unique = dedup_vector_members(field, line);
        assert_members_are_mutually_distinguishable(field, &unique);

        // The whole property, in a form that need not know whether the field
        // juxtaposes letters or joins named parts with `-`: each member appears
        // EXACTLY ONCE. A concatenating producer spells the repeat twice --
        // `ops-bbw`, `arith-f16-f16-i8` -- and fails here.
        let spelled = spelled_field(field, line);
        for m in &unique {
            let n = spelled.matches(m.as_str()).count();
            assert_eq!(
                n, 1,
                "`{field}` distinct members {unique:?} but the token spells \
                 `{spelled}`, which contains `{m}` {n} times. A repeat must be \
                 ABSORBED, not concatenated."
            );
        }
    }
}
