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
