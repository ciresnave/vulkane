//! Shared test helpers. Currently: making a skipped test say so.
//!
//! # Why this exists
//!
//! An early `return` from a `#[test]` is a **pass**. It asserts nothing and
//! reports `ok`, and nothing in the summary distinguishes it from a test that
//! ran every assertion. Device-gated tests do this constantly and legitimately
//! — a machine with no Vulkan ICD cannot exercise a `VkPhysicalDevice` — so
//! the pattern is not a bug in itself. What is a bug is that the *result* is
//! indistinguishable either way.
//!
//! An audit of this repository found **101 of 333 tests** able to return early
//! and still report `ok`. That is not a number to fix by making every one of
//! them fatal: a suite that is permanently red on a laptop without a GPU
//! becomes a suite people stop running, and the next person who needs a green
//! local run reintroduces a silent skip. That loop is how 101 accumulated.
//!
//! So the rule here is **declare everywhere, fail loud only where the evidence
//! is actually required**:
//!
//! - Always print a greppable `SKIP:` line naming the reason.
//! - Under [`REQUIRE_DEVICE`], panic instead — for the environments that are
//!   *supposed* to have a device, where a skip means the evidence silently
//!   evaporated rather than "you're on a laptop".
//!
//! # Known limit
//!
//! `cargo test` captures stdout and stderr and only echoes them for a *failing*
//! test, so the `SKIP:` line is invisible in a green run unless `--nocapture`
//! is passed. Declaring the skip does not by itself make it visible in CI. The
//! thing that makes a skip impossible to miss is [`REQUIRE_DEVICE`] turning it
//! into a failure; the printed line is for a human reading a local run.

/// Set this to any non-empty value to turn a device-gated skip into a failure.
///
/// Intended for environments where a device is guaranteed — a machine with a
/// real GPU, or a CI job with a software Vulkan implementation installed. In
/// those places a skip is not "no hardware here", it is "the evidence this run
/// was supposed to produce did not get produced", and that should be loud.
pub const REQUIRE_DEVICE: &str = "VULKANE_REQUIRE_DEVICE";

/// Whether skips are currently fatal.
#[allow(dead_code)]
pub fn skips_are_fatal() -> bool {
    std::env::var_os(REQUIRE_DEVICE).is_some_and(|v| !v.is_empty())
}

/// Declare that a test could not run, and why.
///
/// Panics instead when [`REQUIRE_DEVICE`] is set. Returns `()`, so a caller can
/// both declare and bail in one statement:
///
/// ```ignore
/// let Some(caps) = caps() else {
///     return common::skipped("no Vulkan ICD");
/// };
/// ```
#[allow(dead_code)]
pub fn skipped(reason: &str) {
    if skips_are_fatal() {
        panic!(
            "test skipped, but {REQUIRE_DEVICE} is set: {reason}\n\n\
             This environment is declared to have a Vulkan device, so a skip \
             here means the evidence this run exists to produce was not \
             produced — while the run would otherwise have reported `ok`. \
             Either the device really is absent (fix the environment) or the \
             precondition is wrong (fix the test)."
        );
    }
    eprintln!("SKIP: {reason}");
}
