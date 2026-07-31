//! Live-device tests for `vulkane::kiss` — deriving KISS-Classify §6.8
//! `vulkan:` `target_capability` tokens from real hardware.
//!
//! These exist to answer a question spec text cannot: **is the vocabulary
//! actually derivable?** A canonical spelling can read cleanly in a document
//! and still turn out ambiguous or underdetermined the moment something tries
//! to produce it from a real `VkPhysicalDevice`. Finding that out before the
//! vocabulary is ratified means a wording change; finding it out afterwards
//! means a spec revision.
//!
//! Skips gracefully with no Vulkan ICD present.

#![cfg(feature = "kiss-target")]

use kiss_vulkan_vocab::{CoopMatrix, Subgroup, VulkanTarget};
use vulkane::kiss::DeviceCapabilities;
use vulkane::safe::*;

/// A 1.3 instance — the arithmetic, driver, and subgroup-size-control queries
/// all gate on the *instance* version, so a lower one silently starves the
/// deriver of the very axes it exists to report.
fn caps() -> Option<(Instance, PhysicalDevice, DeviceCapabilities)> {
    let instance = Instance::new(InstanceCreateInfo {
        api_version: ApiVersion::V1_3,
        ..Default::default()
    })
    .ok()?;
    let physical = instance
        .enumerate_physical_devices()
        .ok()?
        .into_iter()
        .next()?;
    let c = DeviceCapabilities::of(&physical)?;
    Some((instance, physical, c))
}

#[test]
fn derives_a_parseable_canonical_token_for_every_admissible_choice() {
    let Some((_i, physical, caps)) = caps() else {
        eprintln!("SKIP: no Vulkan ICD, or the device declined subgroup properties");
        return;
    };
    println!("device: {}", physical.properties().device_name());
    println!("caps:   {caps:?}");

    let choices = caps.admissible_subgroups();
    assert!(
        !choices.is_empty(),
        "a device that reports subgroup properties must admit at least one specialization"
    );
    // Width-agnostic is always admissible: such a kernel runs at any width.
    assert!(choices.contains(&Subgroup::Dynamic));

    for sg in &choices {
        let target = caps.target_for(*sg);
        let token = target.to_token();
        println!("  {token}");

        // The round-trip is the property §6.8-0002 depends on: the token is
        // the identity, so it must survive serialization exactly.
        let reparsed = VulkanTarget::parse(&token)
            .unwrap_or_else(|e| panic!("derived token failed to parse: {token}\n  {e}"));
        assert_eq!(reparsed, target, "round-trip changed the target: {token}");
        assert_eq!(reparsed.to_token(), token, "re-spelling was not idempotent");

        // A device must admit the target derived from its own capabilities.
        assert!(
            caps.admits(&target),
            "device does not admit its own derived target: {token}"
        );
    }
}

#[test]
fn derivation_is_deterministic_across_calls() {
    // Two reads of the same device must produce byte-identical tokens.
    // Driver-reported cooperative-matrix order is not guaranteed stable, and
    // an unsorted list would surface here as a token that differs run to run —
    // which under byte-exact matching means a cache that never hits.
    let Some((_i, physical, first)) = caps() else {
        eprintln!("SKIP: no Vulkan ICD");
        return;
    };
    let second = DeviceCapabilities::of(&physical).expect("second read");
    assert_eq!(first, second, "capability read is not deterministic");

    for sg in first.admissible_subgroups() {
        assert_eq!(
            first.target_for(sg).to_token(),
            second.target_for(sg).to_token(),
            "token derivation is not deterministic at {sg:?}"
        );
    }
}

#[test]
fn subgroup_choices_match_the_reported_range() {
    let Some((_i, physical, caps)) = caps() else {
        eprintln!("SKIP: no Vulkan ICD");
        return;
    };
    let sg = physical
        .subgroup_properties()
        .expect("already succeeded above");

    let fixed: Vec<u32> = caps
        .admissible_subgroups()
        .into_iter()
        .filter_map(|s| match s {
            Subgroup::Fixed(w) => Some(w),
            Subgroup::Dynamic => None,
        })
        .collect();

    match sg.size_control {
        Some(sc) => {
            // Every admissible width must be pinnable per the device's own
            // validity rule — this is exactly what `permits` guards, and a
            // deriver that offered an unpinnable width would produce a token
            // for a pipeline that cannot be created.
            for w in &fixed {
                assert!(
                    sc.permits(*w),
                    "derived width {w} is outside the device's pinnable range {}..={}",
                    sc.min_subgroup_size,
                    sc.max_subgroup_size
                );
            }
            assert!(
                fixed.contains(&sc.min_subgroup_size) && fixed.contains(&sc.max_subgroup_size),
                "the range endpoints {}..={} must both be admissible, got {fixed:?}",
                sc.min_subgroup_size,
                sc.max_subgroup_size
            );
        }
        None => {
            // No size control: exactly the one fixed width the device reports.
            assert_eq!(fixed, vec![sg.subgroup_size]);
        }
    }
}

#[test]
fn a_narrower_kernel_target_is_admitted_but_spells_differently() {
    // The cell-sharing property that motivates §3a: a kernel requiring less
    // than the device offers must spell its own narrower token, and that token
    // must be distinct from the device's maximal one — otherwise every kernel
    // on a given device would collapse into one cell.
    let Some((_i, _p, caps)) = caps() else {
        eprintln!("SKIP: no Vulkan ICD");
        return;
    };
    let full = caps.target_for(Subgroup::Dynamic);
    let minimal = VulkanTarget {
        subgroup: Subgroup::Dynamic,
        ops: kiss_vulkan_vocab::OpClasses::NONE,
        arith: kiss_vulkan_vocab::Arith::NONE,
        coop: CoopMatrix::None,
    };
    assert!(
        caps.admits(&minimal),
        "every device admits a kernel that requires nothing"
    );
    if full != minimal {
        assert_ne!(
            full.to_token(),
            minimal.to_token(),
            "a maximal and a minimal kernel must not share a cell"
        );
    }
}

#[test]
fn rejects_a_target_the_device_cannot_run() {
    let Some((_i, _p, caps)) = caps() else {
        eprintln!("SKIP: no Vulkan ICD");
        return;
    };
    // A width no real device pins.
    let impossible = VulkanTarget {
        subgroup: Subgroup::Fixed(4096),
        ops: kiss_vulkan_vocab::OpClasses::NONE,
        arith: kiss_vulkan_vocab::Arith::NONE,
        coop: CoopMatrix::None,
    };
    assert!(!caps.admits(&impossible));
}
