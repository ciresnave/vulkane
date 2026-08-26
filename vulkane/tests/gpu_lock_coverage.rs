//! A live-device test must acquire its device through a guarded helper.
//!
//! # Why a source scan rather than only a runtime check
//!
//! `common::require_serialization_lock` refuses to touch a device outside the
//! machine-wide `Global\gpu-run` mutex — but only from the helpers that call it.
//! When it was added it sat in `instance_and_devices` alone, and the suite was
//! green **because most live tests build their own instance and never reach
//! it**. Running `kiss_target_live` without the wrapper passed.
//!
//! That is the defect the runtime guard exists to prevent, occurring inside it:
//! **a guard whose absence is indistinguishable from its success.** A guard
//! covering nothing is green in exactly the same way as one that works, so the
//! only thing that could catch it was testing whether the guard *fired* rather
//! than whether the suite was *green*.
//!
//! Adding the call to today's sites closes today and says nothing about
//! tomorrow. This scan is the half that holds: a new live test that acquires a
//! device by an unguarded route fails here, at review time, rather than
//! silently joining the population.
//!
//! # This population is a LOWER BOUND, and that matters
//!
//! A source scan's reach is **naming and syntax**. It counts the spellings it
//! knows — currently the single spelling `Instance::new(` — and a test that
//! acquires a device by any other route is invisible to it. **A scanner that knows it is
//! a lower bound is honest; the same scanner without this paragraph reads as
//! coverage.** If a device-touching route appears that this does not recognise,
//! the fix is to teach the scanner, not to trust the green.
//!
//! # The baseline shrinks and never grows
//!
//! [`UNGUARDED_BUDGET`] is **debt, not an allowlist**. An allowlist of the
//! existing sites would be permanent by default and quietly become the spec —
//! the ratchet would hold the line and nothing would ever move it. A count that
//! may only go **down** makes the remaining sites visible as work, and makes
//! finishing them the only way to change the number.
//!
//! Raised with Fuel's architect, whose repo has the same shape at ~30x the size
//! (60 direct `CudaDevice::new(0)` sites in `*_live.rs` alone). Both constraints
//! in this file are theirs.

use std::cmp::Ordering;
use std::path::PathBuf;

/// Direct device-acquisition sites still outside a guarded helper.
///
/// **May only decrease.** Each one is a live test that can touch the GPU
/// without the machine-wide lock; the guarded helpers in `common::` are the
/// destination.
///
/// 20 at 2026-08-20, then 8 the same day, then **0 at 2026-08-26**.
///
/// `safe_wrapper_test.rs` held twelve of the original twenty: seven routed
/// through `instance_and_devices`, and five kept a direct call because
/// **instance creation is what they test** — an unknown layer failing cleanly,
/// empty option lists being accepted, the `validation()` constructor. A sixth
/// of that kind is `instance_create_info_pnext_is_plumbed_without_error`. All
/// six call the guard themselves and carry [`DELIBERATE_MARKER`] with a reason.
///
/// The last eight — in `extension_pnext_test`, `generator_live_device_test`,
/// `kiss_target_live`, `raytracing_test` and `tier2_extensions_test` — were
/// routed through the guarded helpers on 2026-08-26.
///
/// **Zero is a floor, not a finish.** The scan's reach is naming and syntax, so
/// this says "no site spelled `Instance::new(` sits outside a helper", which is
/// a smaller claim than "no test can touch the GPU unguarded". What zero does
/// buy is that the next such site is now a REGRESSION rather than one more
/// entry in a backlog — the number can no longer absorb it silently.
const UNGUARDED_BUDGET: usize = 0;

/// Spellings that acquire a device. Extend when a new route appears — see the
/// lower-bound note in the module docs.
const ACQUIRES_A_DEVICE: &[&str] = &["Instance::new("];

/// Marks a direct acquisition that is deliberate and guarded in place.
///
/// A few tests have instance creation as their SUBJECT — an unknown layer
/// failing cleanly, empty option lists being accepted, the `validation()`
/// constructor — and routing those through a helper would test the helper's
/// arguments instead of the thing under test. They call
/// `require_serialization_lock()` themselves and carry this marker with a
/// reason.
///
/// This is an exemption WITH A STATED REASON, the same shape as
/// `GPU_RUN_UNGUARDED`. Without it the budget could never reach a true floor:
/// legitimate direct sites would sit in the count forever, the number would
/// stop falling, and a ratchet that cannot reach zero stops being read.
const DELIBERATE_MARKER: &str = "GPU-LOCK-DIRECT:";

/// The helpers that call `require_serialization_lock` before acquiring.
///
/// These are the DESTINATION for the sites counted below, not an exemption from
/// counting. An earlier version used them to exempt whole files and thereby
/// waved through twelve sites in one file; see the note in `unguarded_sites`.
const GUARDED_HELPERS: &[&str] = &[
    "instance_and_devices(",
    "compute_device(",
    "first_compute(",
    "create_device_on(",
];

fn tests_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn rust_test_files() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let dir = tests_dir();
    let entries =
        std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
    out.sort();
    out
}

/// How many lines above a site count as "attached to it".
///
/// Narrow on purpose: a marker far from the call it excuses would drift away
/// during an edit and start exempting something it was never written for.
///
/// It was 4 and had to become 8, which is itself worth recording. Adding
/// `require_serialization_lock()` to a marked site — the fix for a review
/// finding — pushed the marker two lines further from the call and **the site
/// silently became debt again**, count 8 → 9. The window and the thing it
/// measures are coupled, so widening the comment moved the boundary. The
/// ratchet caught it immediately, which is the argument for having had it.
const MARKER_WINDOW_LINES: usize = 8;

/// Do the lines just above `at` carry [`DELIBERATE_MARKER`]?
fn marked_above(src: &str, at: usize) -> bool {
    window_above(src, at).any(|l| l.contains(DELIBERATE_MARKER))
}

/// Does the same window carry the guard call itself?
///
/// **A marker alone must not suppress the scan.** An exemption that switches a
/// check off and asserts nothing in its place is the defect this file exists to
/// prevent, reintroduced at the granularity of one comment: the absence of a
/// signal made indistinguishable from success.
///
/// Found in review, on the one marked site whose justification was a claim about
/// *control flow* — "the lock is already held by the call that just failed" —
/// rather than a check. The inference was true, and it was true about **another
/// function's body**, so it would have stopped being true silently.
///
/// A marked site must therefore also call the guard within the same window. The
/// marker says *why* the site is direct; the call is what makes it safe.
/// Requiring both means the comment can never be the only thing standing there.
fn guarded_above(src: &str, at: usize) -> bool {
    window_above(src, at).any(|l| l.contains("require_serialization_lock()"))
}

/// The lines above `at`, nearest first.
fn window_above(src: &str, at: usize) -> impl Iterator<Item = &str> {
    let start = src[..at].rfind('\n').unwrap_or(0);
    src[..start].rsplit('\n').take(MARKER_WINDOW_LINES)
}

/// Count of unguarded acquisition sites, with the files they live in.
fn unguarded_sites() -> (usize, Vec<(String, usize)>) {
    let mut total = 0;
    let mut per_file = Vec::new();

    for path in rust_test_files() {
        // This file scans sources and never touches a device; excluding it by
        // name rather than by cleverness, so the exclusion is visible.
        if path
            .file_name()
            .is_some_and(|n| n == "gpu_lock_coverage.rs")
        {
            continue;
        }
        // An unreadable file must FAIL, not be skipped. A scan that cannot
        // read a file reports no finding for it, and "no finding" is exactly
        // what a clean file looks like — the null-without-a-positive-control
        // shape, in a scanner written after the previous one exempted whole
        // files and passed at 1-of-20.
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                concat!(
                    "cannot read {} — the scan cannot see this file, and an ",
                    "unreadable file is indistinguishable from a clean one: {}"
                ),
                path.display(),
                e
            )
        });
        // Counted per SITE, not per file. The first version of this scan
        // exempted any file that mentioned a guarded helper anywhere — which
        // waved through all twelve direct sites in `safe_wrapper_test.rs`
        // because the file also uses `instance_and_devices` elsewhere. It
        // reported 1 unguarded site out of 20 and passed. The scanner had
        // exactly the defect it was written to catch, one level up: covered in
        // name, covering almost nothing, and green either way.
        // A site is debt unless the lines just above it carry the marker.
        // Counted by position rather than by totals, so a file cannot offset a
        // genuinely unguarded site with a marked one elsewhere — the whole-file
        // arithmetic that made the first version of this scan report 1-of-20.
        let n = ACQUIRES_A_DEVICE
            .iter()
            .flat_map(|needle| src.match_indices(needle))
            .filter(|(at, _)| !marked_above(&src, *at))
            .count();

        // A marker not backed by the guard call is worse than no marker: it
        // suppresses the scan AND records a reason, so it reads as considered
        // rather than missing. Checked here rather than trusted.
        for (at, _) in ACQUIRES_A_DEVICE
            .iter()
            .flat_map(|needle| src.match_indices(needle))
        {
            assert!(
                !marked_above(&src, at) || guarded_above(&src, at),
                "{}:{} carries {} but does not call \
                 `require_serialization_lock()` within {} lines.\n\n\
                 The marker exempts this site from the scan. Without the guard \
                 call it exempts it from the RUNTIME check too — while \
                 recording a reason, which reads as considered. An exemption \
                 must say why AND still assert the property.",
                path.display(),
                src[..at].matches('\n').count() + 1,
                DELIBERATE_MARKER,
                MARKER_WINDOW_LINES
            );
        }

        if n > 0 {
            total += n;
            per_file.push((
                path.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("?")
                    .to_string(),
                n,
            ));
        }
    }
    (total, per_file)
}

/// The debt may shrink, never grow.
#[test]
fn unguarded_device_acquisitions_do_not_increase() {
    let (total, per_file) = unguarded_sites();
    let detail = per_file
        .iter()
        .map(|(f, n)| format!("  {n:>3}  {f}"))
        .collect::<Vec<_>>()
        .join("\n");

    // A `match` on the ordering rather than two asserts. At a budget of 0 the
    // old `assert!(total <= UNGUARDED_BUDGET, ..)` became an absurd comparison
    // -- `usize <= 0` is `== 0`, and clippy is right to reject it. Branching on
    // `cmp` keeps BOTH directions and both messages, and makes the two-sidedness
    // of the ratchet explicit instead of implied by a pair of assertions.
    match total.cmp(&UNGUARDED_BUDGET) {
        Ordering::Equal => {}
        Ordering::Greater => panic!(
            concat!(
                "unguarded device acquisitions rose to {} (budget {}).

{}

",
                "A live test acquiring a device outside a guarded helper can touch ",
                "the GPU without the machine-wide `Global\\gpu-run` mutex, and ",
                "nothing at runtime will say so -- the run completes, the tests ",
                "pass, and the only difference is a mutex nobody observes.

",
                "Route it through `common::instance_and_devices` (or another helper ",
                "that calls `require_serialization_lock`) rather than raising this ",
                "number. The budget is DEBT, not an allowlist: it may only go down."
            ),
            total, UNGUARDED_BUDGET, detail
        ),
        // Currently UNREACHABLE: `total` is a `usize` and the budget is 0. Kept
        // because it becomes live again the moment the budget is non-zero, and
        // there is one real way for that to happen -- teaching
        // `ACQUIRES_A_DEVICE` a new spelling can raise the count legitimately,
        // and the budget then has to come back down. This is not the same as a
        // check that could never fail; it is one that cannot fail TODAY.
        Ordering::Less => panic!(
            concat!(
                "unguarded device acquisitions fell to {} -- below the recorded ",
                "budget of {}. That is good; lower UNGUARDED_BUDGET to {} in the ",
                "same change so the ratchet keeps holding at the new level.

{}

",
                "Left as a failure rather than a pass on purpose: a budget that ",
                "silently tolerates being beaten stops being a ratchet and becomes ",
                "a ceiling nobody lowers."
            ),
            total, UNGUARDED_BUDGET, total, detail
        ),
    }
}

/// The guarded helpers must actually be guarded.
///
/// The budget above counts direct acquisition sites and treats the
/// [`GUARDED_HELPERS`] as the safe destination for them. That is only true while
/// those helpers call `require_serialization_lock` — **if the call were removed,
/// every test routing through them would become unguarded and the site count
/// would not move by one.** The scanner would report the same number and mean
/// something entirely different, which is why this is a separate assertion
/// rather than a comment on the constant.
#[test]
fn the_guarded_helpers_still_call_the_guard() {
    let common = tests_dir().join("common").join("mod.rs");
    let src = std::fs::read_to_string(&common)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", common.display()));

    assert!(
        src.contains("fn require_serialization_lock"),
        "common/mod.rs no longer defines `require_serialization_lock`; every \
         file this scan counts as guarded would silently become unguarded, and \
         the site count would not move."
    );
    assert!(
        src.contains("require_serialization_lock();"),
        "common/mod.rs defines `require_serialization_lock` but never calls it. \
         The helpers this scan treats as guarded would be guarded in name only \
         — and the budget above would keep reporting the same number while \
         meaning something entirely different."
    );
}

/// A device-acquiring helper that lives in `common/` must be listed here.
///
/// Not a completeness proof — see the lower-bound note in the module docs — but
/// it catches the specific case of a *new* helper being added to `common/` and
/// this scan not learning about it, which would silently widen what counts as
/// guarded.
#[test]
fn every_common_helper_that_acquires_a_device_is_listed() {
    let common = tests_dir().join("common").join("mod.rs");
    let src = std::fs::read_to_string(&common)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", common.display()));

    let mut examined = 0usize;
    for (name, signature) in public_signatures(&src) {
        // Only helpers that hand back a device or an instance matter here.
        let returns_device = signature.contains("PhysicalDevice")
            || signature.contains("Instance")
            || signature.contains("Device");
        if !returns_device {
            continue;
        }
        examined += 1;
        let hint = format!("{name}(");
        assert!(
            GUARDED_HELPERS.iter().any(|h| *h == hint),
            "common/mod.rs exposes `{name}`, whose signature mentions a device \
             or instance, but GUARDED_HELPERS does not list it.\n\n\
             A caller routing through it would be counted as UNGUARDED by this \
             scan (a false alarm), or — if it does not call \
             `require_serialization_lock` — would be genuinely unguarded while \
             looking fine. Add it to GUARDED_HELPERS and make sure it calls the \
             guard.\n\n  signature: {signature}"
        );
    }

    // POSITIVE CONTROL. Without this the test passes when it finds NOTHING, and
    // finding nothing is exactly what it did: every device helper in
    // `common/mod.rs` has a multi-line signature, and the first version matched
    // only single lines. It examined zero helpers and reported ok — the same
    // null-without-a-control shape as the scanner it guards.
    assert!(
        examined >= GUARDED_HELPERS.len(),
        "the signature scan examined only {examined} device-returning helpers, \
         but GUARDED_HELPERS names {}. It is not seeing the file — a scan that \
         matches nothing passes, which is how the single-line version of this \
         check reported ok while examining zero helpers.",
        GUARDED_HELPERS.len()
    );
}

/// `(name, whole signature)` for each `pub fn` in a source file.
///
/// Joins from `pub fn` to the opening `{`, because **every device helper in
/// `common/mod.rs` has a multi-line signature** — `compute_device`,
/// `instance_and_devices`, `first_compute` and `create_device_on` all wrap their
/// parameters, so a line-at-a-time match sees none of them and the caller
/// silently examines an empty set.
fn public_signatures(src: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = src;
    while let Some(i) = rest.find("pub fn ") {
        let after = &rest[i + "pub fn ".len()..];
        let Some(name_end) = after.find(['(', '<', ' ']) else {
            break;
        };
        let name = after[..name_end].trim().to_string();
        // Up to the body, or the whole remainder if this is the last item.
        let sig_end = after.find(" {").unwrap_or(after.len().min(400));
        out.push((name, after[..sig_end].replace('\n', " ")));
        rest = after;
    }
    out
}

/// Keeps the scanner's own path assumptions honest.
#[test]
fn the_scan_actually_reads_files() {
    let files = rust_test_files();
    assert!(
        files.len() > 5,
        "the scan found only {} test files under {} — it is looking in the \
         wrong place, and a scan that reads nothing reports zero unguarded \
         sites and passes.",
        files.len(),
        tests_dir().display()
    );
    assert!(
        files.iter().any(|p| p.ends_with("safe_wrapper_test.rs")),
        "the scan did not find safe_wrapper_test.rs, which is the largest \
         population of direct acquisition sites in the suite"
    );
}

/// Focused tests for the marker window.
///
/// `marked_above` decides whether a site is debt, so an off-by-one in it either
/// exempts something unguarded or counts something guarded — and both are
/// silent, because the only visible effect is a number that looks plausible
/// either way. Asked for in review, and the window had already moved once
/// underneath the sites it measures.
#[cfg(test)]
mod marker_window_tests {
    use super::{DELIBERATE_MARKER, MARKER_WINDOW_LINES, guarded_above, marked_above};

    fn src_with_gap(gap: usize) -> (String, usize) {
        let mut s = format!("// {DELIBERATE_MARKER} reason\n");
        for i in 0..gap {
            s.push_str(&format!("// filler {i}\n"));
        }
        s.push_str("    Instance::new(x);\n");
        let at = s.find("Instance::new(").expect("target present");
        (s, at)
    }

    #[test]
    fn a_marker_directly_above_is_honoured() {
        let (s, at) = src_with_gap(0);
        assert!(marked_above(&s, at));
    }

    /// The exact boundary, both sides. A window tested only in the middle
    /// cannot tell `take(n)` from `take(n + 1)`.
    #[test]
    fn the_window_ends_where_it_says_it_does() {
        let (s, at) = src_with_gap(MARKER_WINDOW_LINES - 1);
        assert!(
            marked_above(&s, at),
            "a marker exactly at the window edge must still count"
        );
        let (s, at) = src_with_gap(MARKER_WINDOW_LINES);
        assert!(
            !marked_above(&s, at),
            "a marker one line past the window must not count — otherwise the \
             window is wider than it claims and a distant comment can exempt a \
             site it was never written for"
        );
    }

    /// A marker on the same line, or below, is not "above".
    #[test]
    fn a_marker_at_or_below_the_site_does_not_count() {
        let same = format!("    Instance::new(x); // {DELIBERATE_MARKER} nope\n");
        let at = same.find("Instance::new(").unwrap();
        assert!(!marked_above(&same, at));

        let below = format!("    Instance::new(x);\n// {DELIBERATE_MARKER} nope\n");
        let at = below.find("Instance::new(").unwrap();
        assert!(!marked_above(&below, at));
    }

    /// Fewer than a window's worth of lines must not panic or over-read.
    #[test]
    fn the_start_of_a_file_is_handled() {
        let s = "Instance::new(x);\n";
        assert!(!marked_above(s, 0));

        let s = format!("// {DELIBERATE_MARKER} r\nInstance::new(x);\n");
        let at = s.find("Instance::new(").unwrap();
        assert!(marked_above(&s, at));
    }

    /// The guard call is looked for in the same window, on the same terms.
    #[test]
    fn the_guard_call_uses_the_same_window() {
        let s = "    require_serialization_lock();\n    Instance::new(x);\n";
        let at = s.find("Instance::new(").unwrap();
        assert!(guarded_above(s, at));

        let mut far = String::from("    require_serialization_lock();\n");
        for i in 0..MARKER_WINDOW_LINES {
            far.push_str(&format!("// filler {i}\n"));
        }
        far.push_str("    Instance::new(x);\n");
        let at = far.find("Instance::new(").unwrap();
        assert!(
            !guarded_above(&far, at),
            "a guard call beyond the window must not vouch for this site — the \
             two checks have to agree about what 'nearby' means, or a marker \
             could be honoured while the call it depends on is not found"
        );
    }
}
