//! Live-device tests for `PhysicalDevice::cooperative_vector_properties`.
//!
//! # Why this query exists separately from cooperative matrix
//!
//! The packed component types — `VK_COMPONENT_TYPE_SINT8_PACKED_NV` and
//! `..._UINT8_PACKED_NV` — are defined by `VK_NV_cooperative_vector` and are
//! **not** gated on `VK_KHR_cooperative_matrix`. Reading cooperative-matrix
//! properties can therefore never observe one, no matter the device.
//!
//! That was checked against hardware rather than argued: an RTX 4070 reports 16
//! cooperative-vector combinations whose component values include `1000491000`
//! and `1000491001` (the packed pair) alongside `1000491002` / `1000491003`
//! (FP8), while its cooperative-matrix properties contain none of them.
//!
//! # These skip without a device — and say so
//!
//! Absence of the *extension* is a capability gap, declared and never fatal:
//! most devices have no cooperative-vector support, and an AMD Radeon 610M in
//! the same machine reports zero combinations. Absence of a *device* is fatal
//! under `VULKANE_REQUIRE_DEVICE`.

mod common;

use vulkane::safe::ApiVersion;

#[test]
fn cooperative_vector_properties_are_well_formed_and_nameable() {
    // Scans *every* physical device, not just the first compute-capable one.
    // On a machine with an AMD iGPU listed before an NVIDIA discrete part, the
    // first-device version declared "VK_NV_cooperative_vector unsupported"
    // while the capability sat on device 1 — a capability skip that was simply
    // wrong about the machine it was running on.
    let (_instance, devices) =
        match common::instance_and_devices("vulkane-coopvec-test", ApiVersion::V1_0) {
            Ok(v) => v,
            Err(cause) => return common::skip(cause),
        };

    let Some((physical, combos)) = devices
        .into_iter()
        .map(|pd| {
            let combos = pd.cooperative_vector_properties();
            (pd, combos)
        })
        .find(|(_, combos)| !combos.is_empty())
    else {
        return common::skipped_unsupported("VK_NV_cooperative_vector on any physical device");
    };
    let _ = &physical;

    for c in &combos {
        // Every component position must round-trip through the raw value. This
        // is the property the generated `i32` typing exists to make checkable:
        // a driver may report something this build cannot name, and that must
        // surface as `None` rather than as an invalid enum discriminant.
        for (label, named, raw) in [
            ("input", c.input_type(), c.input_type_raw()),
            (
                "input_interpretation",
                c.input_interpretation(),
                c.input_interpretation_raw(),
            ),
            (
                "matrix_interpretation",
                c.matrix_interpretation(),
                c.matrix_interpretation_raw(),
            ),
            (
                "bias_interpretation",
                c.bias_interpretation(),
                c.bias_interpretation_raw(),
            ),
            ("result", c.result_type(), c.result_type_raw()),
        ] {
            if let Some(named) = named {
                assert_eq!(
                    named as i32, raw,
                    "{label}: the named component must equal the raw value it came from"
                );
            }
            // `None` is legitimate — a newer driver than this vk.xml — so it is
            // reported rather than asserted against.
            if named.is_none() {
                eprintln!("note: {label} reported unnameable component {raw}");
            }
        }
    }
}

/// The packed types are reachable *here* and nowhere else, which is the entire
/// reason this query was added.
///
/// Not an assertion that a device must support them — most do not. It asserts
/// the direction: if a packed value shows up at all, it shows up in a
/// cooperative-vector interpretation field and never in cooperative-matrix
/// properties. A future change that tried to name the packed types off the
/// matrix query would still pass every other test in this repository.
#[test]
fn packed_component_types_never_appear_in_cooperative_matrix_properties() {
    const SINT8_PACKED: i32 = 1_000_491_000;
    const UINT8_PACKED: i32 = 1_000_491_001;

    let (_instance, devices) =
        match common::instance_and_devices("vulkane-coopvec-test", ApiVersion::V1_0) {
            Ok(v) => v,
            Err(cause) => return common::skip(cause),
        };

    for m in devices
        .iter()
        .flat_map(|pd| pd.cooperative_matrix_properties())
    {
        for raw in [
            m.a_type_raw(),
            m.b_type_raw(),
            m.c_type_raw(),
            m.result_type_raw(),
        ] {
            assert!(
                raw != SINT8_PACKED && raw != UINT8_PACKED,
                "a packed component type appeared in cooperative-matrix properties \
                 ({raw}). That contradicts the registry, where the packed values are \
                 defined by VK_NV_cooperative_vector with no cooperative-matrix \
                 dependency — and it would mean the matrix query can observe them \
                 after all, which changes where they should be named."
            );
        }
    }
}
