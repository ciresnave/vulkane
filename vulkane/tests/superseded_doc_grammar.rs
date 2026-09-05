//! The grammar quoted in the superseded namespace proposal must be byte-identical
//! to the manifest's `grammar` field.
//!
//! `docs/kiss-vulkan-namespace-proposal.md` carries a header warning readers not to
//! take a grammar from its body, and — because a warning that says nothing concrete
//! is a weak warning — it reproduces the ratified grammar. **That reproduction is a
//! second copy of a machine-owned fact**, and this file is what stops it drifting.
//!
//! It exists because the first version of that header did drift, immediately, in a
//! way review caught and I did not: the fenced block held
//!
//! ```text
//! vulkan:<subgroup>.<ops>.<arith>.<coop>.<coopvec>        vocabulary_version 5
//! ```
//!
//! which is not the manifest's `grammar` string, and which yields an **invalid
//! token** if a reader copies the line — in a document whose entire subject is
//! byte-exact matching. Both halves were real. The fix was not only to correct the
//! text but to make the copy unable to go wrong again.
//!
//! The alternative was to delete the quotation and point at the manifest, which is
//! what the fix to KISS's registry row did for a field *count*. The difference is
//! that a count restated in a registry has no reader who needs it there, whereas a
//! reader of this header is being told "not that grammar — this one" and needs to
//! see which. So the copy stays and is pinned, rather than being removed.

use std::path::PathBuf;

fn repo_file(rel: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {} — {e}", p.display()))
}

/// The `grammar` value from the emitted manifest.
///
/// A narrow parse rather than a JSON dependency: the value is emitted by
/// `kiss-vulkan-vocab/examples/emit_vocabulary_manifest.rs` as a single line and
/// contains no quote or escape. `None` when the field is absent or malformed, so
/// the caller can fail rather than silently comparing against nothing.
fn manifest_grammar(src: &str) -> Option<&str> {
    let after = src.split_once("\"grammar\": \"")?.1;
    let (value, _) = after.split_once('"')?;
    if value.is_empty() { None } else { Some(value) }
}

/// The single line inside the header's fenced block.
///
/// The block sits in a blockquote, so every line is prefixed `> `. Returns `None`
/// when there is no fenced block or it does not hold exactly one line — either is a
/// structural change this test must not silently tolerate.
fn quoted_grammar(doc: &str) -> Option<String> {
    let after = doc.split_once("> ```")?.1;
    let (body, _) = after.split_once("> ```")?;
    let lines: Vec<String> = body
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.trim_start()
                .trim_start_matches('>')
                .trim_start()
                .to_string()
        })
        .collect();
    match lines.len() {
        1 => Some(lines.into_iter().next().unwrap()),
        _ => None,
    }
}

#[test]
fn the_superseded_doc_quotes_the_manifest_grammar_exactly() {
    let manifest = repo_file("../kiss-vulkan-vocab/manifest/vulkan-vocabulary.json");
    let doc = repo_file("docs/kiss-vulkan-namespace-proposal.md");

    // Positive control on BOTH extractions. A scan that matches nothing must fail,
    // not pass — otherwise renaming the field or restructuring the header turns this
    // test into a green that compares two absences.
    let expected = manifest_grammar(&manifest)
        .expect("no `grammar` field in the manifest — this test compared nothing");
    let quoted = quoted_grammar(&doc)
        .expect("no single-line fenced block in the doc header — this test compared nothing");

    assert_eq!(
        quoted, expected,
        concat!(
            "the grammar quoted in docs/kiss-vulkan-namespace-proposal.md is not the ",
            "manifest's `grammar` field.\n\n  doc:      {}\n  manifest: {}\n\n",
            "The fenced block must hold the grammar and NOTHING else — no version ",
            "suffix, no trailing commentary. A reader of a document about byte-exact ",
            "matching may copy that line, and it has to be a usable value."
        ),
        quoted, expected
    );
}

#[test]
fn the_quoted_grammar_is_a_usable_token_shape() {
    let doc = repo_file("docs/kiss-vulkan-namespace-proposal.md");
    let quoted = quoted_grammar(&doc).expect("no single-line fenced block in the doc header");

    assert!(
        quoted.starts_with("vulkan:"),
        "the quoted grammar does not start with the namespace prefix: {quoted}"
    );
    // The specific thing review caught: a version suffix appended inside the block.
    assert!(
        !quoted.contains("vocabulary_version"),
        "the fenced block carries a version suffix again — a copied line would be an \
         invalid token: {quoted}"
    );
    assert!(
        !quoted.contains("  "),
        "the fenced block carries padded trailing commentary again: {quoted}"
    );
}

/// The extractors must be able to FAIL. Both return `Option` precisely so the tests
/// above can distinguish "compared and matched" from "found nothing and said ok",
/// and that distinction is worthless unless the `None` path is reachable.
#[test]
fn the_extractors_return_none_rather_than_a_false_match() {
    assert_eq!(manifest_grammar("{ \"namespace\": \"vulkan\" }"), None);
    assert_eq!(manifest_grammar("{ \"grammar\": \"\" }"), None);
    assert_eq!(quoted_grammar("no fenced block here"), None);
    assert_eq!(
        quoted_grammar("> ```\n> one\n> two\n> ```"),
        None,
        "two lines in the block must not be silently reduced to one"
    );
    assert_eq!(
        quoted_grammar("> ```\n> vulkan:x\n> ```").as_deref(),
        Some("vulkan:x"),
        "and the happy path must still work, or the None cases prove nothing"
    );
}

/// The document's **live body** — every line that is not inside a blockquote.
///
/// The header warning and the discharged-status note are blockquotes, and both
/// deliberately *quote* the false sentences they retire — a retraction that cannot
/// name what it retracts is a weak retraction. So a whole-document scan for those
/// sentences would fire on the retraction itself, and could be satisfied only by
/// deleting the history. The body is what a reader takes as current; the body is
/// what these tests pin.
fn live_body(doc: &str) -> String {
    doc.lines()
        .filter(|l| !l.trim_start().starts_with('>'))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The `**Status: …**` line from the live body — the field a reader trusts first.
///
/// `None` when there is no such line, so a caller fails rather than passing on an
/// absence: a renamed or deleted status line must not read as a clean result.
fn status_line(doc: &str) -> Option<String> {
    live_body(doc)
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("**Status:"))
        .map(str::to_string)
}

/// The status line must not still announce an open proposal.
///
/// This document was granted, filed, and revised four times. Until 2026-09-05 its
/// status line read *"Status: PROPOSAL — opening a design thread"* anyway — and the
/// header immediately above it already said otherwise, **and even listed the
/// neighbouring sentence among the statements now false**. The correction was
/// present, accurate, and positioned where it could not defend the field a reader
/// reads first.
///
/// A status line is the highest-authority text in a document. A reader who notices
/// the contradiction resolves it in favour of whatever is labelled authoritative,
/// so a stale value there does not merely survive next to the correction — it
/// outranks it.
#[test]
fn the_status_line_does_not_still_claim_an_open_proposal() {
    let doc = repo_file("docs/kiss-vulkan-namespace-proposal.md");
    let status = status_line(&doc)
        .expect("no `**Status:` line in the live body — this test checked nothing");

    for banned in ["PROPOSAL", "opening a design thread"] {
        assert!(
            !status.contains(banned),
            "the status line claims an open proposal again: {status}\n\n\
             This document is SUPERSEDED — the namespace was granted, filed and \
             revised four times. Quoting the old wording inside the blockquote note \
             is intended; asserting it as the document's current status is not."
        );
    }
}

/// The same, for the sentence the header itself flags as false.
///
/// The header lists *"asks four questions before anything is filed"* among the
/// statements that were true on 2026-07-31 and are false now — and the sentence
/// nevertheless stood in the body, in the present tense, for the entire time that
/// list existed. Naming a falsehood is not replacing it.
#[test]
fn the_body_does_not_still_ask_the_four_questions_in_the_present_tense() {
    let doc = repo_file("docs/kiss-vulkan-namespace-proposal.md");
    let body = live_body(&doc);

    // Positive control: the body must still look like this document, or the
    // assertion below is satisfied by having read nothing.
    assert!(
        body.contains("target_capability"),
        "live_body() returned text that does not look like this document — the \
         blockquote filter has eaten the body and these checks are vacuous"
    );

    assert!(
        !body.contains("asks four questions"),
        "the body asks the four questions in the present tense again. All four were \
         answered and the namespace was filed; the header has said so since \
         supersession."
    );
}

/// The extractors must be able to return nothing, and must not mistake quoted
/// history for a live claim. Without this, both tests above could be green while
/// checking an empty string.
#[test]
fn the_body_filter_distinguishes_quoted_history_from_live_text() {
    // A blockquote quoting the retired wording is history, not the status.
    assert_eq!(
        status_line("> **Status: PROPOSAL — opening a design thread.**\n\nbody"),
        None,
        "a status line inside a blockquote must not be read as the document's status"
    );

    // ... and a live one must be found, or the None case above proves nothing.
    assert_eq!(
        status_line("**Status: SUPERSEDED — sent 2026-07-31.** rest").as_deref(),
        Some("**Status: SUPERSEDED — sent 2026-07-31.** rest"),
    );

    assert_eq!(status_line("no status line at all"), None);
    assert_eq!(live_body("> quoted\n> more").trim(), "");
    assert_eq!(live_body("> quoted\nlive").trim(), "live");
}
