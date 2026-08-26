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
