//! Every test file and every example must be compiled by some CI leg.
//!
//! # The gap this closes
//!
//! A test file gated behind `#![cfg(feature = "x")]` compiles to **nothing**
//! when `x` is off. `cargo test` then reports `running 0 tests ... ok` for it,
//! which is indistinguishable in the summary from a file whose tests all
//! passed. Nothing is skipped, nothing is declared, nothing is red — the file
//! simply is not there.
//!
//! That is the *silent absence* failure, and it is worse than the silent skip
//! this suite's [`common`] helpers were written for. A skip at least executes
//! the test binary and can be made to announce itself. Absence has nothing to
//! announce from, because no code was built.
//!
//! It is not hypothetical here. Until the CI change that accompanied this file,
//! `vertex_derive_test.rs` and `kiss_target_live.rs` had **never once run in
//! CI** — the workflow enabled neither `derive` nor `kiss-target`. The
//! regression test for signed-integer vertex attributes being handed a float
//! format was, for its entire existence, dead weight in every CI run that
//! reported success.
//!
//! An unbuilt **example** is the same defect pointed at users instead of at us:
//! it is code shipped in the crate that no one compiled. Seven of eighteen were
//! built when this file was written.
//!
//! # Why a test rather than a checklist
//!
//! Both gaps were found by an audit, and an audit only holds until the next
//! person adds a file. This asserts the property continuously: add a
//! feature-gated test file or a new example without touching `ci.yml`, and this
//! fails with the name of what you added.
//!
//! # What it cannot see
//!
//! It reads the workflow as text and checks that the feature name appears in
//! some `--features` list. It cannot know whether that leg's step actually ran,
//! whether the runner had the native toolchain, or whether the tests inside
//! asserted anything. Those are different guarantees, held by different
//! mechanisms: `VULKANE_REQUIRE_DEVICE` for the second, [`common::skipped`] for
//! the third. This one answers exactly one question — *was it built at all?*

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Repository root, derived from this crate's manifest directory.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the vulkane crate always has a parent directory")
        .to_path_buf()
}

/// True when the parent directory is this workspace rather than a registry
/// cache — i.e. we are looking at a repository checkout.
///
/// `tests/` ships inside the published crate, so this file runs for anyone who
/// vendors vulkane and runs `cargo test`. There, `repo_root()` is whatever
/// directory the crate was unpacked into and `.github/` does not exist; the
/// guard is a property of the *repository*, not of the crate, and has nothing
/// to check.
///
/// The workspace manifest is the signal, deliberately rather than
/// `GITHUB_ACTIONS`. Keying on the CI environment variable would make the guard
/// silently vacuous everywhere else, including on the machine of whoever is
/// about to add an uncovered test file — and a coverage guard that only runs
/// after you have pushed is worth much less than one that fails locally first.
fn in_repository_checkout() -> bool {
    std::fs::read_to_string(repo_root().join("Cargo.toml")).is_ok_and(|s| s.contains("[workspace]"))
}

/// The workflow text, or `None` when there is no repository to check.
///
/// Absent **and** not a checkout is a legitimate non-applicability, declared
/// rather than passed over in silence. Absent **while** in a checkout is a
/// broken guard and panics: that is the case where someone moved or deleted the
/// workflow and every one of these tests would otherwise start reporting `ok`
/// while verifying nothing.
fn workflow() -> Option<String> {
    let path = repo_root().join(".github/workflows/ci.yml");
    match std::fs::read_to_string(&path) {
        Ok(text) => Some(text),
        Err(e) => {
            assert!(
                !in_repository_checkout(),
                "cannot read {} ({e}), but this *is* a workspace checkout. The \
                 CI-coverage guard has nothing to read, so it would pass while \
                 checking nothing. Restore the workflow or fix this path — do \
                 not let the guard go quiet.",
                path.display()
            );
            eprintln!(
                "SKIP: {} not present and this is not a workspace checkout \
                 (packaged or vendored crate) — the CI-coverage guard is a \
                 property of the repository and does not apply here",
                path.display()
            );
            None
        }
    }
}

/// One `cargo test` invocation in the workflow.
struct TestLeg {
    /// Features this invocation enables.
    features: BTreeSet<String>,
    /// The `--test <name>` targets it names. **Empty means all of them**, which
    /// is what a bare `cargo test` compiles.
    only: BTreeSet<String>,
}

impl TestLeg {
    /// Whether this leg compiles a test file gated on `feature` with file stem
    /// `stem`.
    fn builds(&self, feature: &str, stem: &str) -> bool {
        self.features.contains(feature) && (self.only.is_empty() || self.only.contains(stem))
    }
}

/// Every `cargo test` invocation in the workflow, with its features and target
/// restriction.
///
/// **Only `cargo test` counts, and that is the whole point of this function.**
/// The first version scanned the file for any `--features` list, which was
/// wrong in a way that made the guard report coverage it did not have:
/// `cargo build --features fetch-spec,derive --example derive_vertex` enables
/// `derive`, but `cargo build` does not compile integration tests, so
/// `vertex_derive_test.rs` could have been absent from every test leg while
/// this guard called it covered. A coverage guard that reports false coverage
/// is worse than no guard, because it converts an open question into a wrong
/// answer.
///
/// `--test <name>` is tracked for the same reason one level down: a leg reading
/// `cargo test --features shaderc --test shaderc_test` compiles exactly one
/// test target, so it cannot vouch for a second file gated on `shaderc`.
fn test_legs() -> Option<Vec<TestLeg>> {
    let text = workflow()?;
    let mut legs = Vec::new();

    for line in text.lines() {
        if !line.contains("cargo test") {
            continue;
        }
        let mut features = BTreeSet::new();
        let mut only = BTreeSet::new();
        let words: Vec<&str> = line.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            match *word {
                "--features" => {
                    if let Some(list) = words.get(i + 1) {
                        features.extend(
                            list.split(',')
                                .map(str::trim)
                                .filter(|f| !f.is_empty())
                                .map(str::to_string),
                        );
                    }
                }
                "--test" => {
                    if let Some(name) = words.get(i + 1) {
                        only.insert((*name).to_string());
                    }
                }
                _ => {}
            }
        }
        legs.push(TestLeg { features, only });
    }

    assert!(
        !legs.is_empty(),
        "parsed no `cargo test` invocations out of ci.yml — the workflow format \
         changed and this guard is now vacuous, which is the exact failure it \
         exists to prevent. Fix the parser, do not delete the test."
    );
    Some(legs)
}

/// Every feature enabled by some `cargo test` leg. For error messages.
fn features_enabled_in_ci() -> BTreeSet<String> {
    test_legs()
        .into_iter()
        .flatten()
        .flat_map(|leg| leg.features)
        .collect()
}

fn rust_files_in(dir: &Path) -> Vec<PathBuf> {
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot list {}: {e}", dir.display()));
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .collect();
    files.sort();
    files
}

/// The feature a file gates its *whole contents* on, if any.
///
/// Only the crate-level `#![cfg(feature = "…")]` counts. An inner
/// `#[cfg(feature = "…")]` on one item leaves the rest of the file compiled, so
/// the file is not absent and this guard has nothing to say about it.
///
/// Matched per line, and only on a line that *starts* with the attribute.
/// Searching the whole file for the marker finds it inside doc comments too —
/// including this file's own prose, which is how the first version reported
/// that `ci_coverage.rs` was gated on a feature named `x`. The self-check below
/// caught it on the first run, which is the argument for having a self-check.
fn file_level_feature_gate(path: &Path) -> Option<String> {
    let src = std::fs::read_to_string(path).ok()?;
    let marker = "#![cfg(feature = \"";
    for line in src.lines() {
        let line = line.trim_start();
        if line.starts_with("//") {
            continue;
        }
        if let Some(rest) = line.strip_prefix(marker) {
            let end = rest.find('"')?;
            return Some(rest[..end].to_string());
        }
    }
    None
}

#[test]
fn every_feature_gated_test_file_is_built_by_some_ci_leg() {
    let Some(legs) = test_legs() else { return };
    let enabled = features_enabled_in_ci();
    let mut missing = Vec::new();

    for dir in ["vulkane/tests", "kiss-vulkan-vocab/tests"] {
        let dir = repo_root().join(dir);
        if !dir.is_dir() {
            continue;
        }
        for file in rust_files_in(&dir) {
            let Some(feature) = file_level_feature_gate(&file) else {
                continue; // Ungated: built by every leg.
            };
            let stem = file
                .file_stem()
                .expect("a .rs file has a stem")
                .to_string_lossy();
            if !legs.iter().any(|leg| leg.builds(&feature, &stem)) {
                missing.push(format!(
                    "  {} is gated on `{feature}`, which no `cargo test` leg \
                     enables for this target",
                    file.strip_prefix(repo_root()).unwrap_or(&file).display()
                ));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "these test files compile to nothing in CI and report `running 0 tests ... ok`:\n\
         {}\n\n\
         Features CI does enable: {enabled:?}\n\n\
         Add the feature to a `--features` list in .github/workflows/ci.yml. If it \
         genuinely cannot run there — a native toolchain CI does not have, say — \
         that is a real gap and it belongs in the workflow as a comment stating \
         why, not silently absent from it. A file no leg builds is not covered by \
         anything, and the summary will not tell you.",
        missing.join("\n")
    );
}

#[test]
fn every_example_is_built_by_ci() {
    let Some(text) = workflow() else { return };
    let dir = repo_root().join("vulkane/examples");
    let mut missing = Vec::new();

    for file in rust_files_in(&dir) {
        let name = file
            .file_stem()
            .expect("a .rs file has a stem")
            .to_string_lossy()
            .to_string();
        if !text.contains(&format!("--example {name}")) {
            missing.push(name);
        }
    }

    assert!(
        missing.is_empty(),
        "these examples are never compiled by CI: {missing:?}\n\n\
         An example that does not build is a broken artifact shipped to users, \
         and nothing in the repository would notice. Add each to the \"Build \
         examples\" step in .github/workflows/ci.yml with whatever features it \
         needs."
    );
}

/// The guard above is only worth having if it can fail. A parser that silently
/// matches nothing would pass both tests forever while checking nothing — the
/// same "exists but enforces nothing" shape the rest of this suite keeps
/// running into.
#[test]
fn the_parser_actually_finds_the_features_it_claims_to() {
    let Some(legs) = test_legs() else { return };
    let enabled = features_enabled_in_ci();

    // `fetch-spec` is on every leg; if the parser works at all it finds this.
    assert!(
        enabled.contains("fetch-spec"),
        "parsed features {enabled:?} without `fetch-spec`, which every CI leg \
         enables — the parser is not reading what it thinks it is"
    );

    // And it must not be matching everything indiscriminately.
    assert!(
        !enabled.contains("definitely-not-a-real-feature"),
        "the parser reports a feature that does not exist"
    );

    // The restriction to `cargo test` must actually bite. `derive` appears in
    // a `cargo build --example derive_vertex` line *and* in a `cargo test`
    // line; if the parser ever goes back to scanning every `--features` list,
    // a leg with no unrestricted test invocation would still look like
    // coverage. Assert the shape the guard depends on: at least one leg
    // compiles all test targets.
    assert!(
        legs.iter().any(|leg| leg.only.is_empty()),
        "no `cargo test` leg compiles all test targets — every leg names \
         `--test <target>`, so a newly added test file would be built by none \
         of them while this guard still reported its feature as covered"
    );

    // A `--test`-restricted leg must be parsed as restricted, not as blanket
    // coverage. Without this the `only` field could silently stop populating
    // and every restricted leg would start vouching for files it never builds.
    let restricted: Vec<&TestLeg> = legs.iter().filter(|leg| !leg.only.is_empty()).collect();
    assert!(
        !restricted.is_empty(),
        "expected at least one `--test <target>` leg (the shaderc job is \
         restricted); the `--test` parser is not populating"
    );
    for leg in restricted {
        assert!(
            !leg.builds("fetch-spec", "definitely_not_a_test_file"),
            "a `--test`-restricted leg claimed to build a target it does not name"
        );
    }

    // A file-level gate must be detectable, using a file known to carry one.
    let gated = repo_root().join("vulkane/tests/kiss_target_live.rs");
    assert_eq!(
        file_level_feature_gate(&gated).as_deref(),
        Some("kiss-target"),
        "the file-level gate detector stopped recognising a known gate"
    );

    // An ungated file must read as ungated.
    let ungated = repo_root().join("vulkane/tests/ci_coverage.rs");
    assert_eq!(
        file_level_feature_gate(&ungated),
        None,
        "this file is not feature-gated; if the detector says otherwise it is \
         matching something it should not"
    );
}
