//! The documented `shader-slang` limitation must stay tied to the dependency it
//! describes.
//!
//! Three places say the same thing: `shader-slang` 0.1.0 cannot load Slang from a
//! source string, so modules must live in `.slang` files. Two of them are live
//! documentation — `README.md` and `src/safe/slang.rs`. The third is the `[0.4.5]`
//! CHANGELOG entry, which is history and correctly describes what was true at that
//! release; it is deliberately not checked here.
//!
//! The limitation is real and it is owned. What it did not have was a **detector**.
//! The blocker lifts when a newer `shader-slang` ships and this crate picks it up,
//! and nothing connected that event to the prose, so on the day the dependency
//! moved, two live documents would have gone on telling users to keep their Slang
//! in files and nothing anywhere would have gone red.
//!
//! **Checked against the lock, not the manifest, and that distinction is the whole
//! point of the file.** `vulkane/Cargo.toml` requires `"0.1"` — a caret range. An
//! upstream `0.1.1` that re-exposes `loadModuleFromSourceString` would be taken by
//! a routine `cargo update` with **no manifest edit at all**, so a check against
//! the declared requirement would sit green through exactly the event it was
//! written for. `Cargo.lock` is tracked here and records the resolved version, so
//! it moves on every version change, patch included.
//!
//! A roadmap line would not have caught this, and neither would a date. This is the
//! cheapest thing that observably changes state while the item is undone.

use std::path::PathBuf;

fn repo_file(rel: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {} — {e}", p.display()))
}

/// The exact `shader-slang` version this workspace resolves to, from `Cargo.lock`.
///
/// A narrow parse rather than a TOML dependency: lock entries are `[[package]]`
/// blocks with `name` and `version` on adjacent lines. `None` when the package is
/// absent or the block is reshaped, so the caller fails rather than comparing
/// against nothing.
fn locked_version(lock: &str, package: &str) -> Option<String> {
    let needle = format!("name = \"{package}\"");
    let after = lock.split_once(needle.as_str())?.1;
    let line = after
        .lines()
        .find(|l| l.trim_start().starts_with("version = "))?;
    let after_eq = line.split_once('"')?.1;
    let (v, _) = after_eq.split_once('"')?;
    if v.is_empty() {
        None
    } else {
        Some(v.to_string())
    }
}

/// Every version-shaped token that a document attaches to the name `shader-slang`.
///
/// Scans a short window after each mention, so the bare mentions — the build
/// instructions, the GitHub releases URL — contribute nothing rather than being
/// miscounted. Returns them all: a second copy of the claim appearing later is
/// exactly the drift this file exists to catch.
fn claimed_versions(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (i, _) in src.match_indices("shader-slang") {
        let tail: String = src[i + "shader-slang".len()..].chars().take(24).collect();
        if let Some(v) = first_version(&tail) {
            out.push(v);
        }
    }
    out
}

fn first_version(s: &str) -> Option<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            let tok: String = chars[start..i].iter().collect();
            let tok = tok.trim_end_matches('.').to_string();
            if tok.contains('.') {
                return Some(tok);
            }
        } else {
            i += 1;
        }
    }
    None
}

/// The version the docs blame must be the version actually being built against.
///
/// This is the assertion that goes red the day the limitation lifts.
#[test]
fn the_documented_slang_version_is_the_one_actually_resolved() {
    let lock = repo_file("../Cargo.lock");
    let resolved = locked_version(&lock, "shader-slang")
        .expect("no `shader-slang` package block in Cargo.lock — this test compared nothing");

    for (label, rel) in [
        ("README.md", "../README.md"),
        ("slang.rs", "src/safe/slang.rs"),
    ] {
        let doc = repo_file(rel);
        let claims = claimed_versions(&doc);

        // Positive control: the scan must find the claim it is about to check. An
        // empty list would satisfy the loop below by never entering it.
        assert!(
            !claims.is_empty(),
            "{label} no longer attaches any version to `shader-slang`. Either the \
             limitation note was removed — in which case delete this test — or it \
             was reworded past the scanner and is now unchecked."
        );

        for claimed in &claims {
            assert_eq!(
                claimed, &resolved,
                "{label} documents the inline-source limitation against \
                 `shader-slang` {claimed}, but Cargo.lock resolves {resolved}.\n\n\
                 If the dependency moved, the limitation may no longer hold: check \
                 whether the new release exposes a source-string loader. If it does, \
                 remove the note from README.md (\"Current limitation\") and from \
                 src/safe/slang.rs (\"Note on inline source\") and add \
                 `SlangSession::load_source`. If it does not, update the version in \
                 both notes. The CHANGELOG entry under [0.4.5] is history and stays \
                 as it is."
            );
        }
    }
}

/// The other direction: shipped, but the documentation never noticed.
///
/// `slang.rs` names the method the limitation blocks — `SlangSession::load_source`.
/// If that method ever exists while the note still says it cannot, the note is
/// false in the more embarrassing direction: a feature users are being told to
/// work around.
#[test]
fn the_limitation_note_is_gone_once_the_method_it_blocks_exists() {
    let slang = repo_file("src/safe/slang.rs");

    let note = slang.contains("does not expose a source-string");
    let method = slang.contains("fn load_source");

    assert!(
        !(note && method),
        "src/safe/slang.rs defines `load_source` while still documenting that \
         `shader-slang` cannot load from a source string. One of the two is wrong, \
         and the note is the likelier one — remove it here and in README.md."
    );

    // Control on both probes, so this cannot pass by matching neither. Exactly one
    // of the two states must hold, and today it is the note.
    assert!(
        note || method,
        "neither the limitation note nor `fn load_source` is present in \
         src/safe/slang.rs — this test has lost its subject and is vacuous"
    );
}

/// The parsers must be able to fail, or every assertion above can be green while
/// comparing against nothing.
#[test]
fn the_parsers_return_none_rather_than_a_false_match() {
    let lock = "[[package]]\nname = \"shader-slang\"\nversion = \"0.1.0\"\n";
    assert_eq!(
        locked_version(lock, "shader-slang").as_deref(),
        Some("0.1.0")
    );
    assert_eq!(
        locked_version(lock, "not-a-package"),
        None,
        "an absent package must not read as a match"
    );
    assert_eq!(
        locked_version("[[package]]\nname = \"shader-slang\"\n", "shader-slang"),
        None,
        "a block with no version line must fail rather than yield nothing quietly"
    );
    assert_eq!(
        locked_version(
            "[[package]]\nname = \"shader-slang\"\nversion = \"\"\n",
            "shader-slang"
        ),
        None
    );

    // The lock lists packages alphabetically, so the version taken must be the one
    // in THIS package's block and not a later one.
    let two = "[[package]]\nname = \"shader-slang\"\nversion = \"0.1.0\"\n\n\
               [[package]]\nname = \"zzz\"\nversion = \"9.9.9\"\n";
    assert_eq!(
        locked_version(two, "shader-slang").as_deref(),
        Some("0.1.0")
    );

    // A bare mention contributes no version; a versioned one does.
    assert!(claimed_versions("see shader-slang/slang releases for builds").is_empty());
    assert_eq!(
        claimed_versions("`shader-slang` 0.1.0 on crates.io"),
        vec!["0.1.0"]
    );
}
