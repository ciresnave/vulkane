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
//! - Print a greppable `SKIP:` line naming the reason — in both modes, so the
//!   line is present in the runs where it is fatal too.
//! - Under [`REQUIRE_DEVICE`], panic instead — for the environments that are
//!   *supposed* to have a device, where a skip means the evidence silently
//!   evaporated rather than "you're on a laptop".
//!
//!
//! # Why the `#[allow(dead_code)]` on almost everything here
//!
//! Cargo compiles each file in `tests/` into its OWN binary, and `mod common;`
//! includes this module separately into every one of them. A helper used by
//! four test binaries and not the fifth is dead code *in the fifth*, so without
//! the attribute the suite emits a warning per helper per binary that does not
//! happen to call it -- and `-D warnings` turns those into failures.
//!
//! So the attributes here say "unused in THIS binary", not "unused". They are
//! not covering anything: removing one and watching the warning name a specific
//! binary is the check, and the answer will name a test that simply does not
//! need that helper.
//!
//! This is stated once, at the module, rather than repeated at each item --
//! there is one reason and it is the same reason every time.
//! # Two kinds of skip, and why conflating them would undo the mechanism
//!
//! "No device" and "device without this extension" are not the same event, and
//! only the first is an environment fault.
//!
//! [`REQUIRE_DEVICE`] asserts *a device exists*. It cannot assert the device
//! supports everything, because no device does: Lavapipe — the software
//! implementation the Linux CI job installs — has no ray tracing, no
//! `VK_EXT_descriptor_buffer`, no external-memory handles. Those absences are
//! conformant. Routing them through [`skipped`] would make the Linux job red
//! for reasons nobody can fix, and the first person to hit it would unset the
//! variable — which would take the device requirement down with it and restore
//! a green that means nothing.
//!
//! So capability gates use [`skipped_unsupported`], which declares just as
//! loudly and is never fatal. The split is the thing that lets the fatal mode
//! stay switched on.
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

/// What a test needed and did not get — carrying **which class of skip it is**,
/// not just a message.
///
/// The fatal/non-fatal split is the load-bearing decision in this module, and
/// until this type existed it was enforced only by whoever wrote the call site
/// picking the right function. A helper that produces a cause far from where it
/// is reported — [`compute_device`], `features_or_skip` — handed back a bare
/// string, and the classification was re-decided by hand at every site. Both
/// ways of getting it wrong are bad, and they are bad asymmetrically:
///
/// - A capability gate misrouted to [`Missing::Device`] turns CI red for
///   something conformant. Loud, annoying, and someone will "fix" it by
///   unsetting [`REQUIRE_DEVICE`] — taking the whole mechanism with it.
/// - Device absence misrouted to [`Missing::Capability`] is **silent**. The run
///   stays green and the evidence quietly stops being produced, which is the
///   exact failure this module was written to end.
///
/// Carrying the class with the cause means the decision is made once, where the
/// cause is known, and [`skip`] routes it mechanically. There is no per-site
/// judgement left to get wrong.
#[allow(dead_code)]
pub enum Missing {
    /// No usable device. An environment that promised one is misconfigured, so
    /// this is fatal under [`REQUIRE_DEVICE`].
    Device(String),
    /// A device is present but does not support what the test needs. Conformant,
    /// therefore never fatal.
    Capability(String),
}

#[allow(dead_code)]
impl Missing {
    /// No instance, no physical device, no queue family the suite requires.
    pub fn device(reason: impl Into<String>) -> Self {
        Missing::Device(reason.into())
    }

    /// An optional extension, feature, or format the device does not offer.
    pub fn capability(what: impl Into<String>) -> Self {
        Missing::Capability(what.into())
    }
}

/// Declare a [`Missing`], routing to the right behaviour by its class.
///
/// Returns `()`, so a caller can declare and bail in one statement:
///
/// ```ignore
/// let (device, physical, qf) = match common::compute_device("my-test") {
///     Ok(v) => v,
///     Err(cause) => return common::skip(cause),
/// };
/// ```
#[allow(dead_code)]
pub fn skip(cause: Missing) {
    match cause {
        Missing::Device(reason) => skipped(&reason),
        Missing::Capability(what) => skipped_unsupported(&what),
    }
}

/// Declare that a test could not run because **the device itself was absent**,
/// and why.
///
/// Panics instead when [`REQUIRE_DEVICE`] is set. Returns `()`, so a caller can
/// both declare and bail in one statement:
///
/// ```ignore
/// let Some(caps) = caps() else {
///     return common::skipped("no Vulkan ICD");
/// };
/// ```
///
/// Use this only for preconditions an environment that promised a device is
/// obliged to meet — no instance, no physical device, no queue family of a kind
/// every device has. For "the device is here but lacks this extension", use
/// [`skipped_unsupported`]; see the module docs for why the distinction is what
/// keeps the fatal mode usable.
#[allow(dead_code)]
pub fn skipped(reason: &str) {
    // Printed before the fatal check, so the `SKIP:` line exists in both modes.
    // Panicking first would mean the one line you would grep for is missing
    // from exactly the runs where it matters most.
    eprintln!("SKIP: {reason}");

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
}

/// Declare that a test could not run because the device is **present but does
/// not support** what the test needs.
///
/// Never fatal, including under [`REQUIRE_DEVICE`] — an optional extension
/// being absent is a conformant device, not a broken environment, and there is
/// nothing for the person reading a red CI job to fix.
///
/// `what` should name the capability, not the symptom: `"VK_KHR_ray_tracing_
/// pipeline"` tells a reader which device would run this test, where
/// `"vkCmdTraceRaysKHR not loaded"` only restates the branch they are already
/// looking at.
///
/// The line is still printed, so `--nocapture` on a machine that *should* have
/// the capability shows it going unexercised — which is the failure mode a
/// silent `return` hides.
#[allow(dead_code)]
pub fn skipped_unsupported(what: &str) {
    eprintln!("SKIP (unsupported): {what}");
}

/// A logical device with a compute-capable queue family, or the **specific**
/// precondition that failed.
///
/// Four test files had a byte-identical `bootstrap() -> Option<...>` differing
/// only in the application name, and each of the five ways it can fail arrived
/// at the call site as the same `None`. That is the shape the review of #8
/// flagged on `kiss_target_live`'s `caps()`: under [`REQUIRE_DEVICE`] the
/// returned string *is* the failure message someone has to act on, and
/// "no Vulkan" when the real cause was "no compute queue family" sends them to
/// fix the wrong thing.
///
/// Every cause here is treated as device-absence rather than
/// capability-absence, so all of them are fatal under [`REQUIRE_DEVICE`]. Pair
/// with [`skipped`], not [`skipped_unsupported`].
///
/// That is a **declared precondition of this test suite**, not something the
/// Vulkan spec entails. The guarantee the spec gives about compute queues is
/// implementation-wide and conditional on graphics being exposed at all; it
/// does not promise that any *given* physical device has a compute-capable
/// family, and a conformant device may have none. The callers of this helper
/// need compute, and the environments that set [`REQUIRE_DEVICE`] are declared
/// to provide a device that can run them — so "no compute queue family here"
/// means that declaration is wrong, which is worth failing over.
///
/// A test that does **not** need compute should not reach for this helper. It
/// would inherit a precondition it does not have, and turn hardware capable of
/// answering its question into a failed run. See
/// `extension_pnext_test::device_create_info_pnext_is_plumbed_without_error`
/// for one that bootstraps itself for exactly this reason.
#[allow(dead_code)]
pub fn compute_device(
    app_name: &str,
) -> Result<(vulkane::safe::Device, vulkane::safe::PhysicalDevice, u32), Missing> {
    let (_instance, devices) = instance_and_devices(app_name, vulkane::safe::ApiVersion::V1_0)?;
    let (physical, qf) = first_compute(devices)?;
    let device = create_device_on(&physical, qf, None)?;
    // `PhysicalDevice` holds an `Arc<InstanceInner>`, so dropping `_instance`
    // here does not tear the instance down under the returned handles.
    Ok((device, physical, qf))
}

/// Set by `gpu-run.ps1` in the child environment while it holds the
/// machine-wide `Global\gpu-run` mutex.
///
/// The token already existed for nested-invocation passthrough. Nothing read it
/// from Rust until this, so a test could touch the GPU outside the lock and
/// nothing anywhere would say so.
const GPU_RUN_HELD: &str = "GPU_RUN_HELD";

/// The pid the wrapper exported. Checked against the lockfile, never alone.
const GPU_RUN_HELD_PID: &str = "GPU_RUN_HELD_PID";

/// A deliberate, *stated* bypass. Must carry a reason.
///
/// Not the same variable as [`GPU_RUN_HELD`], and that separation is the whole
/// design. If the escape hatch were "set `GPU_RUN_HELD` yourself", a hand-set
/// value would be indistinguishable from the wrapper's — rebuilding the defect
/// inside its own fix. A reason-bearing variable makes a bypass possible,
/// costly to normalize, and **greppable in logs**, so one that quietly became
/// habitual is discoverable afterwards.
const GPU_RUN_UNGUARDED: &str = "GPU_RUN_UNGUARDED";

/// Refuse to touch a device outside the machine-wide GPU serialization lock.
///
/// # Why this is a hard failure and not a skip
///
/// Everything else absent in this file degrades into [`Missing`], and that
/// taxonomy is deliberate: `Device` is fatal under [`REQUIRE_DEVICE`],
/// `Capability` never is. **An unguarded run must not join it.** Skipping would
/// make "you forgot the wrapper" indistinguishable from "this machine has no
/// GPU" — the exact conflation the two-class split exists to prevent. This is a
/// misconfiguration of the *runner*, not a property of the *machine*, so it
/// fails at the point of use, where the person who forgot is standing.
///
/// # Why it exists at all
///
/// Written after I bypassed the wrapper myself: `cargo test --workspace
/// --features …,kiss-target` chained behind a `cargo fmt`, with
/// *"`cargo test --workspace` IS a GPU run"* already in my durable notes. It
/// enumerated physical devices outside the lock, nothing went wrong, and
/// **nothing could have told me** — the guard's absence is indistinguishable
/// from its success. The run completes, the tests pass, and the only difference
/// is a mutex nobody observes.
///
/// The lock is the only thing standing between a concurrent run and the
/// 2026-07-31 host-aperture bugcheck, so the failure being silent is the part
/// that matters rather than the failure being rare.
///
/// # Known gap, stated rather than hidden
///
/// This checks the variable, not the lock. A `GPU_RUN_HELD=1` exported by hand
/// into a long-lived shell satisfies it forever after — and that shell is
/// exactly where someone would set it while debugging. Verifying
/// `GPU_RUN_HELD_PID` is still alive would close it, but costs a subprocess
/// spawn per run from Rust without adding a dependency. Raised with Fuel's
/// architect, who owns the wrapper; a lockfile the wrapper creates and removes
/// would make the check a cheap `Path::exists` instead.
#[allow(dead_code)]
pub fn require_serialization_lock() {
    if lock_is_held() {
        return;
    }

    // A bypass is allowed, but never silently — silence is the defect.
    if let Some(why) = std::env::var_os(GPU_RUN_UNGUARDED).filter(|v| !v.is_empty()) {
        eprintln!(
            concat!(
                "PROCEEDING UNGUARDED: {} — this run touches the GPU outside ",
                "the machine-wide Global\\gpu-run mutex, so a concurrent run ",
                "on this machine is not prevented."
            ),
            why.to_string_lossy()
        );
        return;
    }

    panic!(
        concat!(
            "this test touches a physical device, but the machine-wide GPU ",
            "serialization lock is not held.\n\n",
            "The contract, so this stays true wherever the wrapper lives:\n",
            "  {}=1 and {}=<pid>, plus a `gpu-run.lock` in TEMP whose `pid`\n",
            "  field is that same pid. All three are set by `gpu-run.ps1` while\n",
            "  it holds the lock, and the lockfile is removed when it releases.\n\n",
            "Run through `gpu-run.ps1 -Project vulkane -- <cmd>` from wherever ",
            "you have it. Invoke it with `pwsh`, not `powershell`: some copies ",
            "do not parse under Windows PowerShell 5.1 (Fuel GAP-223), and a ",
            "stale checkout is the likely one to hand.\n\n",
            "If this run genuinely must bypass the lock, say why:\n",
            "  {}=\"<reason>\"\n",
            "which proceeds and prints the reason, so a bypass that became ",
            "habitual is greppable rather than invisible.\n\n",
            "This is a hard failure rather than a skip on purpose. A skip would ",
            "be indistinguishable from \"this machine has no GPU\", and that ",
            "conflation is what the Device/Capability split in this file exists ",
            "to prevent."
        ),
        GPU_RUN_HELD, GPU_RUN_HELD_PID, GPU_RUN_UNGUARDED
    );
}

/// Is the lock held *right now*, by the process that exported our environment?
///
/// Two witnesses, and both are needed:
///
/// - `GPU_RUN_HELD_PID` — who claims to hold it. Exported into our environment
///   by the wrapper, and **survives the wrapper's death**, which is the problem.
/// - `gpu-run.lock` in `TEMP` — written when the lock is taken and **deleted in
///   the wrapper's `finally`**. Its `pid` field is the live answer.
///
/// Checking only the variable is defeated by a shell that outlives one run: a
/// `GPU_RUN_HELD=1` exported by hand while debugging satisfies it forever after,
/// and that shell is exactly where someone would set it. Requiring the lockfile
/// to exist *and* name the same pid answers **"is the lock held now, by the
/// process that exported this"** rather than "did something once say so".
///
/// Process-liveness would have been the obvious alternative and is strictly
/// worse: it costs a subprocess spawn, and a recycled pid satisfies it.
///
/// Two residuals, stated rather than smoothed — Fuel's architect named both, and
/// liveness would not have escaped either:
///
/// - This must resolve the **same** `TEMP` the wrapper used. A process with a
///   different `TEMP` sees no lockfile and is refused, which is the safe
///   direction but can confuse.
/// - A hard-killed holder can leave the file behind. Combined with a matching
///   exported pid in a dead shell that is a real, narrow window.
fn lock_is_held() -> bool {
    if !std::env::var_os(GPU_RUN_HELD).is_some_and(|v| v == "1") {
        return false;
    }
    let Some(pid) = std::env::var_os(GPU_RUN_HELD_PID) else {
        return false;
    };
    let Some(temp) = std::env::var_os("TEMP").or_else(|| std::env::var_os("TMPDIR")) else {
        return false;
    };
    let Ok(meta) = std::fs::read_to_string(std::path::Path::new(&temp).join("gpu-run.lock")) else {
        // Deleted in the wrapper's `finally`, so absence means released.
        return false;
    };
    lockfile_names_pid(&meta, &pid.to_string_lossy())
}

/// Does this lockfile's `pid` field equal `pid` **exactly**?
///
/// The first version was `meta.contains(&format!("\"pid\":{pid}"))`, and that is
/// a false positive whenever the exported pid is a *prefix* of the real one —
/// an environment carrying `GPU_RUN_HELD_PID=12` matches a lockfile written by
/// pid `123`, and the guard reports the lock held when it is not.
///
/// **A guard that says "held" when it is not held, inside the guard whose whole
/// subject is that absence is indistinguishable from success.** Caught in review
/// (Copilot), not by any test here, because every real pid on this machine is
/// four or five digits and a prefix collision needs two specific values.
///
/// So the digit run after `"pid":` is compared as a whole token: it must end at
/// a non-digit, which in the compact JSON the wrapper writes is `,` or `}`.
/// Parsed by hand rather than by taking a JSON dependency into a test helper.
fn lockfile_names_pid(meta: &str, pid: &str) -> bool {
    if pid.is_empty() || !pid.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    let mut rest = meta;
    while let Some(i) = rest.find("\"pid\":") {
        let after = &rest[i + "\"pid\":".len()..];
        let end = after
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(after.len());
        if &after[..end] == pid {
            return true;
        }
        rest = after;
    }
    false
}

/// An instance at `api_version`, plus its physical devices.
///
/// The first two steps of every bootstrap in the suite, extracted so the boot
/// sequence and its failure wording live in one place. Four helpers had grown
/// their own copy of this prefix, which is how `try_init_compute` and
/// `compute_device` came to disagree about which device to pick.
///
/// Both failures here are device-absence: no ICD, or an ICD that cannot answer
/// enumeration. "Reports zero devices" is checked by the callers that need a
/// device, since [`init_at_or_skip`]-style callers want the first device and
/// compute callers want the first *compute-capable* one — different messages
/// for different searches.
///
/// [`init_at_or_skip`]: # "safe_wrapper_test's version-gated bootstrap"
#[allow(dead_code)]
pub fn instance_and_devices(
    app_name: &str,
    api_version: vulkane::safe::ApiVersion,
) -> Result<(vulkane::safe::Instance, Vec<vulkane::safe::PhysicalDevice>), Missing> {
    use vulkane::safe::{Instance, InstanceCreateInfo};

    require_serialization_lock();

    let instance = Instance::new(InstanceCreateInfo {
        application_name: Some(app_name),
        api_version,
        ..Default::default()
    })
    .map_err(|_| Missing::device("no Vulkan ICD, or the loader declined to create an instance"))?;

    let devices = instance.enumerate_physical_devices().map_err(|_| {
        Missing::device("an ICD is present but enumerating physical devices failed")
    })?;

    Ok((instance, devices))
}

/// The first device exposing a compute-capable queue family, and that family's
/// index **on that device**.
///
/// `find` then `position` on the same predicate, so the index is into the
/// chosen device's families rather than whichever device happened to be first.
/// Searching past device 0 matters on a multi-adapter machine: taking the first
/// device and asking whether *it* has compute produces a skip that says "no
/// compute queue family" while a perfectly usable second adapter sits unused.
#[allow(dead_code)]
pub fn first_compute(
    devices: Vec<vulkane::safe::PhysicalDevice>,
) -> Result<(vulkane::safe::PhysicalDevice, u32), Missing> {
    use vulkane::safe::{QueueFamilyProperties, QueueFlags};

    if devices.is_empty() {
        return Err(Missing::device(
            "an ICD is present but reports no physical devices",
        ));
    }
    let compute = |q: &QueueFamilyProperties| q.queue_flags().contains(QueueFlags::COMPUTE);
    let physical = devices
        .into_iter()
        .find(|pd| pd.queue_family_properties().iter().any(compute))
        .ok_or_else(|| {
            Missing::device("no physical device exposes a compute-capable queue family")
        })?;
    let qf = physical
        .queue_family_properties()
        .iter()
        .position(compute)
        .ok_or_else(|| {
            Missing::device("no physical device exposes a compute-capable queue family")
        })? as u32;
    Ok((physical, qf))
}

/// `vkCreateDevice` on `physical` with one queue from family `qf`.
///
/// `features` is what decides the **class** of a failure, and it is the reason
/// this is one function rather than inlined at each site. With no features
/// requested, a device that will not create is a broken environment. With
/// features requested, the overwhelmingly likely cause is that this device does
/// not support them — conformant, and not something to fail CI over. Passing
/// `Some((features, label))` therefore classifies the failure as a capability
/// gap and reports `label`.
#[allow(dead_code)]
pub fn create_device_on(
    physical: &vulkane::safe::PhysicalDevice,
    qf: u32,
    features: Option<(&vulkane::safe::DeviceFeatures, &str)>,
) -> Result<vulkane::safe::Device, Missing> {
    // Creating a logical device touches the GPU, so this is an acquisition even
    // though it takes an already-obtained physical device. Every current caller
    // reaches it through a guarded helper, which is exactly why it needs its own
    // call: "all current callers are guarded" is a fact about today.
    require_serialization_lock();

    use vulkane::safe::{DeviceCreateInfo, QueueCreateInfo};

    physical
        .create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo::single(qf)],
            enabled_features: features.map(|(f, _)| f),
            ..Default::default()
        })
        .map_err(|_| match features {
            Some((_, label)) => Missing::capability(label),
            None => Missing::device("a compute-capable device was found but vkCreateDevice failed"),
        })
}

#[cfg(test)]
mod lock_witness_tests {
    use super::lockfile_names_pid;

    /// The defect Copilot found: a prefix must not match.
    #[test]
    fn a_prefix_pid_does_not_match() {
        let meta = r#"{"v":1,"pid":123,"project":"vulkane"}"#;
        assert!(lockfile_names_pid(meta, "123"));
        assert!(
            !lockfile_names_pid(meta, "12"),
            "pid 12 matched a lockfile written by pid 123 — the substring form \
             of this check reported the lock held when it was not"
        );
        assert!(!lockfile_names_pid(meta, "1"));
        assert!(!lockfile_names_pid(meta, "1234"));
    }

    /// The digit run must end at a delimiter, whichever one the writer used.
    #[test]
    fn the_pid_field_may_end_at_a_comma_or_a_brace() {
        assert!(lockfile_names_pid(r#"{"pid":77,"x":1}"#, "77"));
        assert!(lockfile_names_pid(r#"{"x":1,"pid":77}"#, "77"));
    }

    /// A pid that is not a number cannot match anything, rather than being
    /// pasted into a search and matching by accident.
    #[test]
    fn a_non_numeric_pid_never_matches() {
        let meta = r#"{"pid":123}"#;
        assert!(!lockfile_names_pid(meta, ""));
        assert!(!lockfile_names_pid(meta, "12a"));
        assert!(!lockfile_names_pid(meta, "\"123\""));
    }

    /// A lockfile with no pid field at all is not a match.
    #[test]
    fn absent_or_malformed_metadata_does_not_match() {
        assert!(!lockfile_names_pid("", "123"));
        assert!(!lockfile_names_pid(r#"{"project":"vulkane"}"#, "123"));
        assert!(!lockfile_names_pid(r#"{"pid":}"#, "123"));
    }
}
