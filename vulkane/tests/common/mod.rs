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
) -> Result<(vulkane::safe::Device, vulkane::safe::PhysicalDevice, u32), &'static str> {
    use vulkane::safe::{
        ApiVersion, DeviceCreateInfo, Instance, InstanceCreateInfo, QueueCreateInfo, QueueFlags,
    };

    let instance = Instance::new(InstanceCreateInfo {
        application_name: Some(app_name),
        api_version: ApiVersion::V1_0,
        ..Default::default()
    })
    .map_err(|_| "no Vulkan ICD, or the loader declined to create an instance")?;

    let devices = instance
        .enumerate_physical_devices()
        .map_err(|_| "an ICD is present but enumerating physical devices failed")?;
    if devices.is_empty() {
        return Err("an ICD is present but reports no physical devices");
    }

    // `find` then `position` on the same predicate, so the index returned is an
    // index into *this* device's families rather than whichever device happened
    // to be first.
    let compute =
        |q: &vulkane::safe::QueueFamilyProperties| q.queue_flags().contains(QueueFlags::COMPUTE);
    let physical = devices
        .into_iter()
        .find(|pd| pd.queue_family_properties().iter().any(compute))
        .ok_or("no physical device exposes a compute-capable queue family")?;
    let qf = physical
        .queue_family_properties()
        .iter()
        .position(compute)
        .ok_or("no physical device exposes a compute-capable queue family")? as u32;

    let device = physical
        .create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo::single(qf)],
            ..Default::default()
        })
        .map_err(|_| "a compute-capable device was found but vkCreateDevice failed")?;

    // `PhysicalDevice` holds an `Arc<InstanceInner>`, so dropping `instance`
    // here does not tear the instance down under the returned handles.
    Ok((device, physical, qf))
}
