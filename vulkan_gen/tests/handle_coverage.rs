//! Every Vulkan handle type must have a *decided* fate.
//!
//! `safe_handles_gen` wraps the handles whose create/destroy commands fit a
//! fixed shape. Handles that don't fit are skipped — and a skip is invisible:
//! the build succeeds, the bindings compile, and the handle simply has no
//! safe wrapper. Nobody finds out until someone reaches for it.
//!
//! So the generator sorts every handle into one of three buckets — wrapped
//! by hand, wrapped automatically, or excluded on the record with a reason —
//! and reports anything that lands outside all three as `unclassified`.
//! This test asserts that set is empty for the `vk.xml` the crate ships with,
//! which is what turns "we think we covered everything" into a check.
//!
//! When it fails, a new spec revision introduced a handle nobody has looked
//! at. The fix is a decision, not a suppression: add it to `HAND_WRITTEN` if
//! it deserves a bespoke wrapper, or to `KNOWN_UNWRAPPABLE` with the reason
//! its lifecycle can't be derived.

use std::path::{Path, PathBuf};

use vulkan_gen::codegen::generator_modules::safe_handles_gen::{
    KNOWN_UNWRAPPABLE, generate_safe_handles,
};

/// The `vk.xml` the workspace pins, i.e. what a default build generates from.
fn pinned_vk_xml() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("vulkane")
        .join("vk.xml")
}

fn run_generator() -> vulkan_gen::codegen::generator_modules::safe_handles_gen::SafeHandlesStats {
    let xml = pinned_vk_xml();
    assert!(
        xml.exists(),
        "pinned vk.xml not found at {} — this test compares the generator against the \
         spec the crate actually ships with, so there is nothing to check without it",
        xml.display()
    );

    let temp = tempfile::TempDir::new().expect("create temp dir");
    let intermediate = temp.path().to_path_buf();
    vulkan_gen::parse_vulkan_spec(&xml, &intermediate).expect("parse vk.xml to intermediate JSON");

    generate_safe_handles(
        &intermediate,
        &intermediate.join("auto_handles_generated.rs"),
    )
    .expect("generate safe handle wrappers")
}

#[test]
fn every_handle_is_wrapped_hand_written_or_explicitly_excluded() {
    let stats = run_generator();

    assert!(
        stats.unclassified.is_empty(),
        "{} handle type(s) have no safe wrapper and no recorded reason: {:?}\n\n\
         Each is reachable only through raw dispatch. Decide which bucket it belongs in and \
         say so in vulkan_gen::codegen::generator_modules::safe_handles_gen:\n  \
         - HAND_WRITTEN — it has (or should have) a bespoke wrapper in vulkane::safe\n  \
         - KNOWN_UNWRAPPABLE — its lifecycle isn't create/destroy; record why\n\
         Do not widen the shape matcher just to make this pass: a generated Drop that \
         doesn't match the handle's real lifecycle compiles and then violates valid usage.",
        stats.unclassified.len(),
        stats.unclassified,
    );

    // A generator that wrapped nothing would also report nothing unclassified.
    assert!(
        stats.generated > 0,
        "generator produced no wrappers at all — the assertion above would pass vacuously"
    );
}

/// The exclusion list is only meaningful if each entry is still *needed*.
/// An entry for a handle the generator has since learned to wrap would sit
/// there suppressing a wrapper nobody wants suppressed.
#[test]
fn every_known_unwrappable_entry_carries_a_reason() {
    for (handle, reason) in KNOWN_UNWRAPPABLE {
        assert!(
            handle.starts_with("Vk"),
            "KNOWN_UNWRAPPABLE entry {handle:?} is not a Vulkan handle type name"
        );
        assert!(
            reason.len() > 40,
            "KNOWN_UNWRAPPABLE entry for {handle} has no substantive reason: {reason:?}. \
             The list exists so the next reader knows why the handle is excluded — \
             an entry without that is just a silent skip with extra steps."
        );
    }
}
