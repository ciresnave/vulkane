//! Emit a **raw device-property fixture** for the KISS `vulkan:` namespace, as
//! JSON on stdout.
//!
//! Run with:
//! `cargo run --example kiss_device_fixture -p vulkane > fixture.json`
//!
//! # What this is for
//!
//! KISS umbrella §5.3 condition 1 (as rewritten by KISS #221) counts derivers
//! **per field** of a golden vector: each field needs *two* parties that derive
//! it independently, and a party that receives a value and reproduces it
//! byte-exactly demonstrates faithful *passthrough*, not derivation. Under that
//! clause the `target` field has exactly one deriver in the whole chain —
//! vulkane. KISS carries a hand-written literal, Fuel interpolates it, Unpopped
//! passes it through opaquely.
//!
//! A second deriver therefore cannot be found by asking another party to agree
//! with our token. It has to *derive* one. Every candidate (kiss-ref in
//! particular) satisfies the §8-0004 freeze gate precisely by **not** loading
//! Vulkan, so it cannot read a live device. This file is the remaining shape:
//! vulkane supplies the *input* — what the driver said — and some other party
//! derives the token from it. Supplying the input is not deriving the answer.
//!
//! # The one rule this file exists to obey
//!
//! **Nothing token-shaped, nothing pre-derived, nothing canonicalized.** If the
//! fixture carried any part of the answer, a second party consuming it would be
//! performing passthrough on that part and condition 1 would fail for exactly
//! the reason it was written. Concretely:
//!
//! - **Raw integers and bools only.** Component types are the `i32` the driver
//!   reported. No [`ComponentType`], no `f16`/`i8packed` spellings, no `ops-`
//!   letters, no assembled suffix — this example deliberately does **not**
//!   enable the `kiss-target` feature and never links `kiss-vulkan-vocab`, so
//!   emitting a token from here is not merely avoided but unavailable.
//! - **No derived predicates.** `ShaderIntegerDotProductProperties::has_any_int8_acceleration`
//!   is an OR across six fields — that OR *is* a derivation step, and
//!   `arith-dot8` is downstream of it. All sixteen bools are emitted separately
//!   and the consumer does its own reducing.
//! - **Driver report order, unsorted and un-deduplicated.** `DeviceCapabilities`
//!   sorts and dedups the cooperative arrays, because driver order is not
//!   guaranteed stable and the token must be byte-identical either way. That
//!   canonical ordering is itself part of the derivation, and handing it over
//!   pre-sorted would donate the second party a step they are supposed to take.
//!   What the driver said, in the order it said it.
//!
//! # Two things recorded that are easy to omit and change the answer
//!
//! **The instance API version.** A device reports different things depending on
//! the version the *instance* requested — an implementation must behave as the
//! version asked for, however new the hardware is, so 1.1+ `pNext` property
//! structs read back zeroed under a 1.0 instance and a zero is indistinguishable
//! from a real answer. The capture conditions are part of the data.
//!
//! **Query failure, kept distinct from an empty result.** Both cooperative
//! queries return `Result<Vec<_>>` specifically so that "this device supports
//! none" and "the query failed" stay different answers; collapsing them is the
//! defect that API shape was changed to prevent. A fixture rendering an error as
//! `[]` would re-introduce it at the file format, and a consumer would derive
//! `cm-none` for a device whose support is simply unknown — under §6.8-0002 a
//! different cell, not a degraded one. `status` carries that distinction.
//!
//! # What is deliberately absent
//!
//! No `deviceUUID`, `driverUUID`, `deviceLUID`, or `pipelineCacheUUID`.
//! [`PhysicalDevice::device_identity`] is never called. Those identify a
//! *machine*, not a model of hardware, and this file is meant to be shared
//! across projects; `vendorID`/`deviceID`/`deviceName` already identify the part
//! well enough for provenance. None of them is a deriver input in any case.

use std::fmt::Write as _;
use vulkane::safe::{ApiVersion, PhysicalDevice};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ask for the highest version that will actually create, and record which
    // one won. Requesting 1.0 here would silently zero every 1.1+ property
    // struct below; requesting 1.4 unconditionally would fail on a loader that
    // does not implement it.
    let (instance, api) = [
        (ApiVersion::V1_4, "1.4"),
        (ApiVersion::V1_3, "1.3"),
        (ApiVersion::V1_2, "1.2"),
        (ApiVersion::V1_1, "1.1"),
        (ApiVersion::V1_0, "1.0"),
    ]
    .into_iter()
    .find_map(|(v, label)| {
        vulkane::safe::Instance::new(vulkane::safe::InstanceCreateInfo {
            application_name: Some("vulkane-kiss-device-fixture"),
            api_version: v,
            ..Default::default()
        })
        .ok()
        .map(|i| (i, label))
    })
    .ok_or("no Vulkan ICD, or the loader declined to create an instance at any version")?;

    let devices = instance.enumerate_physical_devices()?;

    let mut out = String::new();
    writeln!(out, "{{")?;
    writeln!(out, "  \"schema\": \"vulkane-device-fixture-v1\",")?;
    writeln!(
        out,
        "  \"note\": \"{}\",",
        esc(concat!(
            "Raw device properties as reported by the driver. Contains no token, ",
            "no vocabulary spelling, no canonical ordering, and no derived ",
            "predicate — deriving a KISS vulkan-namespace capability set from ",
            "this file is the consumer's work, which is the entire point. ",
            "Arrays are in driver report order, deliberately unsorted and ",
            "un-deduplicated. Note this sentence avoids writing the namespace ",
            "prefix in its token form on purpose: tests/kiss_fixture_is_underived.rs ",
            "scans the whole file, prose included, and an exemption for prose ",
            "would be a hole a real leak could hide in."
        ))
    )?;
    writeln!(out, "  \"instance_api_version\": \"{api}\",")?;
    writeln!(
        out,
        "  \"vulkane_version\": \"{}\",",
        env!("CARGO_PKG_VERSION")
    )?;
    writeln!(out, "  \"devices\": [")?;

    let last = devices.len().saturating_sub(1);
    for (i, pd) in devices.iter().enumerate() {
        emit_device(&mut out, i, pd)?;
        writeln!(out, "{}", if i == last { "" } else { "," })?;
    }

    writeln!(out, "  ]")?;
    writeln!(out, "}}")?;
    print!("{out}");
    Ok(())
}

fn emit_device(
    out: &mut String,
    index: usize,
    pd: &PhysicalDevice,
) -> Result<(), Box<dyn std::error::Error>> {
    let props = pd.properties();
    let api = props.api_version().0;

    writeln!(out, "    {{")?;
    writeln!(out, "      \"index\": {index},")?;
    writeln!(
        out,
        "      \"device_name\": \"{}\",",
        esc(&props.device_name())
    )?;
    writeln!(out, "      \"vendor_id\": {},", props.vendor_id())?;
    writeln!(out, "      \"device_id\": {},", props.device_id())?;
    writeln!(out, "      \"driver_version\": {},", props.driver_version())?;
    writeln!(
        out,
        "      \"device_type_raw\": {},",
        props.device_type_raw()
    )?;
    writeln!(
        out,
        "      \"device_api_version\": \"{}.{}.{}\",",
        api >> 22,
        (api >> 12) & 0x3ff,
        api & 0xfff
    )?;

    // Subgroup properties. `None` is a real outcome, not a bug: it is what a
    // device reports under an instance below 1.1, and the consumer must be able
    // to see that it happened rather than read a fabricated zero.
    match pd.subgroup_properties() {
        Some(sg) => {
            writeln!(out, "      \"subgroup_properties\": {{")?;
            writeln!(out, "        \"present\": true,")?;
            writeln!(out, "        \"subgroup_size\": {},", sg.subgroup_size)?;
            // Raw flag bits, not the decoded op classes: decoding is the
            // consumer's derivation and `ops-abr…` is downstream of it.
            writeln!(
                out,
                "        \"supported_operations_bits\": {},",
                sg.supported_operations.0
            )?;
            writeln!(
                out,
                "        \"supported_stages_bits\": {},",
                sg.supported_stages.0
            )?;
            writeln!(
                out,
                "        \"quad_operations_in_all_stages\": {},",
                sg.quad_operations_in_all_stages
            )?;
            match sg.size_control {
                Some(sc) => {
                    writeln!(out, "        \"size_control\": {{")?;
                    writeln!(
                        out,
                        "          \"min_subgroup_size\": {},",
                        sc.min_subgroup_size
                    )?;
                    writeln!(
                        out,
                        "          \"max_subgroup_size\": {},",
                        sc.max_subgroup_size
                    )?;
                    writeln!(
                        out,
                        "          \"max_compute_workgroup_subgroups\": {},",
                        sc.max_compute_workgroup_subgroups
                    )?;
                    writeln!(
                        out,
                        "          \"required_subgroup_size_stages_bits\": {}",
                        sc.required_subgroup_size_stages.0
                    )?;
                    writeln!(out, "        }}")?;
                }
                None => writeln!(out, "        \"size_control\": null")?,
            }
            writeln!(out, "      }},")?;
        }
        None => writeln!(
            out,
            "      \"subgroup_properties\": {{ \"present\": false }},"
        )?,
    }

    // Core `VkPhysicalDeviceFeatures` bits. `shaderInt16`/`shaderInt64`/
    // `shaderFloat64` are here because vocabulary v5 names them; they are
    // emitted whether or not any published vocabulary version spells them,
    // since the fixture records what the device said, not what we can spell.
    let f = pd.supported_features();
    writeln!(out, "      \"core_features\": {{")?;
    writeln!(out, "        \"shader_int16\": {},", f.shaderInt16 != 0)?;
    writeln!(out, "        \"shader_int64\": {},", f.shaderInt64 != 0)?;
    writeln!(out, "        \"shader_float64\": {}", f.shaderFloat64 != 0)?;
    writeln!(out, "      }},")?;

    match pd.shader_arithmetic_features() {
        Some(a) => {
            writeln!(out, "      \"shader_arithmetic_features\": {{")?;
            writeln!(out, "        \"present\": true,")?;
            writeln!(out, "        \"shader_float16\": {},", a.shader_float16)?;
            writeln!(out, "        \"shader_int8\": {},", a.shader_int8)?;
            writeln!(
                out,
                "        \"storage_buffer_16bit\": {},",
                a.storage_buffer_16bit
            )?;
            writeln!(
                out,
                "        \"storage_buffer_8bit\": {}",
                a.storage_buffer_8bit
            )?;
            writeln!(out, "      }},")?;
        }
        None => writeln!(
            out,
            "      \"shader_arithmetic_features\": {{ \"present\": false }},"
        )?,
    }

    // All sixteen bools. `has_any_int8_acceleration()` is deliberately not
    // called — that OR is the consumer's step.
    match pd.shader_integer_dot_product_properties() {
        Some(d) => {
            writeln!(out, "      \"shader_integer_dot_product\": {{")?;
            writeln!(out, "        \"present\": true,")?;
            for (name, v) in [
                ("dot_product_8bit_unsigned", d.dot_product_8bit_unsigned),
                ("dot_product_8bit_signed", d.dot_product_8bit_signed),
                ("dot_product_8bit_mixed", d.dot_product_8bit_mixed),
                (
                    "dot_product_4x8bit_packed_unsigned",
                    d.dot_product_4x8bit_packed_unsigned,
                ),
                (
                    "dot_product_4x8bit_packed_signed",
                    d.dot_product_4x8bit_packed_signed,
                ),
                (
                    "dot_product_4x8bit_packed_mixed",
                    d.dot_product_4x8bit_packed_mixed,
                ),
                ("dot_product_16bit_unsigned", d.dot_product_16bit_unsigned),
                ("dot_product_16bit_signed", d.dot_product_16bit_signed),
                ("dot_product_32bit_unsigned", d.dot_product_32bit_unsigned),
                ("dot_product_32bit_signed", d.dot_product_32bit_signed),
                ("dot_product_64bit_unsigned", d.dot_product_64bit_unsigned),
                ("dot_product_64bit_signed", d.dot_product_64bit_signed),
                (
                    "dot_product_accumulating_sat_8bit_signed",
                    d.dot_product_accumulating_sat_8bit_signed,
                ),
                (
                    "dot_product_accumulating_sat_8bit_unsigned",
                    d.dot_product_accumulating_sat_8bit_unsigned,
                ),
                (
                    "dot_product_accumulating_sat_4x8bit_packed_signed",
                    d.dot_product_accumulating_sat_4x8bit_packed_signed,
                ),
                (
                    "dot_product_accumulating_sat_4x8bit_packed_unsigned",
                    d.dot_product_accumulating_sat_4x8bit_packed_unsigned,
                ),
            ] {
                writeln!(out, "        \"{name}\": {v},")?;
            }
            // Trailing key so the loop above can end every line with a comma.
            writeln!(out, "        \"_all_sixteen_emitted\": true")?;
            writeln!(out, "      }},")?;
        }
        None => writeln!(
            out,
            "      \"shader_integer_dot_product\": {{ \"present\": false }},"
        )?,
    }

    match pd.cooperative_matrix_properties() {
        Ok(shapes) => {
            writeln!(out, "      \"cooperative_matrix\": {{")?;
            writeln!(out, "        \"status\": \"ok\",")?;
            writeln!(out, "        \"driver_order\": [")?;
            let last = shapes.len().saturating_sub(1);
            for (i, s) in shapes.iter().enumerate() {
                write!(
                    out,
                    concat!(
                        "          {{ \"m\": {}, \"n\": {}, \"k\": {}, ",
                        "\"a_type_raw\": {}, \"b_type_raw\": {}, \"c_type_raw\": {}, ",
                        "\"result_type_raw\": {}, \"saturating_accumulation\": {}, ",
                        "\"scope_raw\": {} }}"
                    ),
                    s.m_size(),
                    s.n_size(),
                    s.k_size(),
                    s.a_type_raw(),
                    s.b_type_raw(),
                    s.c_type_raw(),
                    s.result_type_raw(),
                    s.saturating_accumulation(),
                    s.scope_raw()
                )?;
                writeln!(out, "{}", if i == last { "" } else { "," })?;
            }
            writeln!(out, "        ]")?;
            writeln!(out, "      }},")?;
        }
        Err(e) => {
            writeln!(out, "      \"cooperative_matrix\": {{")?;
            writeln!(out, "        \"status\": \"error\",")?;
            writeln!(out, "        \"error\": \"{}\"", esc(&format!("{e:?}")))?;
            writeln!(out, "      }},")?;
        }
    }

    match pd.cooperative_vector_properties() {
        Ok(combos) => {
            writeln!(out, "      \"cooperative_vector\": {{")?;
            writeln!(out, "        \"status\": \"ok\",")?;
            writeln!(out, "        \"driver_order\": [")?;
            let last = combos.len().saturating_sub(1);
            for (i, c) in combos.iter().enumerate() {
                write!(
                    out,
                    concat!(
                        "          {{ \"input_type_raw\": {}, ",
                        "\"input_interpretation_raw\": {}, ",
                        "\"matrix_interpretation_raw\": {}, ",
                        "\"bias_interpretation_raw\": {}, ",
                        "\"result_type_raw\": {}, \"transpose\": {} }}"
                    ),
                    c.input_type_raw(),
                    c.input_interpretation_raw(),
                    c.matrix_interpretation_raw(),
                    c.bias_interpretation_raw(),
                    c.result_type_raw(),
                    c.transpose()
                )?;
                writeln!(out, "{}", if i == last { "" } else { "," })?;
            }
            writeln!(out, "        ]")?;
            writeln!(out, "      }}")?;
        }
        Err(e) => {
            writeln!(out, "      \"cooperative_vector\": {{")?;
            writeln!(out, "        \"status\": \"error\",")?;
            writeln!(out, "        \"error\": \"{}\"", esc(&format!("{e:?}")))?;
            writeln!(out, "      }}")?;
        }
    }

    write!(out, "    }}")?;
    Ok(())
}

/// Minimal JSON string escaping. Device names are vendor-controlled text, so
/// this handles the control range rather than assuming they are tame.
fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o
}
