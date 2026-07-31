//! Enforces KISS-CLASSIFY-6.9-0003 structurally rather than by intention.
//!
//! The clause requires that producing, serializing, or parsing a
//! `target_capability` token need no compute driver, kernel runtime, GPU
//! library, or backend dynamic library — an implementation must manage with
//! its language's standard library alone, and the reference implementation
//! holds no exemption.
//!
//! A comment asserting "no dependencies" is worth nothing: it stays true only
//! until someone adds one, and nothing fails when they do. This test is the
//! enforcement. It reads this crate's own manifest and fails the build if the
//! dependency table is ever non-empty.
//!
//! (Deliberately hand-parsed rather than using a TOML crate: pulling one in as
//! a dev-dependency to check that there are no dependencies would be its own
//! small joke, and dev-dependencies are where a real one would most plausibly
//! sneak in.)

/// Sections whose contents would breach the clause if populated.
const FORBIDDEN_TABLES: &[&str] = &[
    "[dependencies]",
    "[dev-dependencies]",
    "[build-dependencies]",
];

#[test]
fn manifest_declares_no_dependencies() {
    let manifest = include_str!("../Cargo.toml");

    let mut current: Option<&str> = None;
    let mut offenders: Vec<(String, String)> = Vec::new();

    for raw in manifest.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            current = FORBIDDEN_TABLES.iter().copied().find(|t| *t == line);
            continue;
        }
        // A target-specific table such as
        // [target.'cfg(windows)'.dependencies] would also breach the clause.
        if let Some(table) = current {
            offenders.push((table.to_string(), line.to_string()));
        }
    }

    assert!(
        offenders.is_empty(),
        "kiss-vulkan-vocab must have zero dependencies (KISS-CLASSIFY-6.9-0003): \
         a conformance implementation has to be able to produce and parse tokens \
         with only its standard library. Found:\n{}",
        offenders
            .iter()
            .map(|(t, l)| format!("  {t}  {l}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn no_target_specific_dependency_tables() {
    let manifest = include_str!("../Cargo.toml");
    for raw in manifest.lines() {
        let line = raw.trim();
        if line.starts_with('[') && line.contains("dependencies") {
            assert!(
                FORBIDDEN_TABLES.contains(&line),
                "unexpected dependency table `{line}` — if this is a real table it \
                 must be added to FORBIDDEN_TABLES so its contents are checked, \
                 not left unexamined"
            );
        }
    }
}

#[test]
fn crate_links_no_vulkan() {
    // The vocabulary must be usable by a conformance implementation on a
    // machine with no Vulkan loader at all. This is a compile-time proof by
    // construction — if this test binary links, the crate pulled in no
    // driver — but the assertion documents the intent for a reader.
    let t = kiss_vulkan_vocab::VulkanTarget::parse(
        "vulkan:sg64.ops-abr.arith-f16-i8.cm-16-16-16-f16-f16-f32-f32",
    );
    assert!(t.is_ok(), "parsing must not require a Vulkan loader");
}
