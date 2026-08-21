//! `#[derive(Vertex)]` layout tests — CPU only, no Vulkan device required.
//!
//! These assert the *format mapping* the derive picks, not just that it
//! compiles. That distinction matters: a wrong format here is a silent
//! defect. `R32G32B32_SFLOAT` and `R32G32B32_SINT` are both 12 bytes with
//! identical vertex-buffer layout, so substituting one for the other
//! builds cleanly, passes the validation layers, and only shows up as
//! garbage geometry at draw time — the GPU reinterprets the integer bit
//! pattern as IEEE-754. Assert the value, not that the call returned.

#![cfg(feature = "derive")]

use vulkane::Vertex;
use vulkane::safe::{Format, InputRate};

/// Every integer field width the derive claims to support.
#[derive(Vertex, Clone, Copy)]
#[repr(C)]
struct IntVertex {
    scalar: i32,
    pair: [i32; 2],
    triple: [i32; 3],
    quad: [i32; 4],
}

/// The float and unsigned paths, to pin the rest of the table.
#[derive(Vertex, Clone, Copy)]
#[repr(C)]
struct MixedVertex {
    position: [f32; 3],
    uv: [f32; 2],
    indices: [u32; 4],
    color: [u8; 4],
    small_u: u16,
    small_i: i16,
}

#[test]
fn signed_int_fields_map_to_sint_formats_not_sfloat() {
    let attrs = IntVertex::attributes(0);

    assert_eq!(attrs[0].format, Format::R32_SINT);
    assert_eq!(attrs[1].format, Format::R32G32_SINT);
    // The regression: `[i32; 3]` previously fell through to
    // `R32G32B32_SFLOAT` because no SINT constant existed.
    assert_eq!(attrs[2].format, Format::R32G32B32_SINT);
    assert_eq!(attrs[3].format, Format::R32G32B32A32_SINT);

    // Nothing in an all-signed-integer vertex should be a float format.
    for attr in attrs {
        assert_ne!(attr.format, Format::R32G32B32_SFLOAT);
        assert_ne!(attr.format, Format::R32G32B32A32_SFLOAT);
    }
}

#[test]
fn mixed_fields_map_to_their_documented_formats() {
    let attrs = MixedVertex::attributes(0);

    assert_eq!(attrs[0].format, Format::R32G32B32_SFLOAT);
    assert_eq!(attrs[1].format, Format::R32G32_SFLOAT);
    assert_eq!(attrs[2].format, Format::R32G32B32A32_UINT);
    assert_eq!(attrs[3].format, Format::R8G8B8A8_UINT);
    assert_eq!(attrs[4].format, Format::R16_UINT);
    assert_eq!(attrs[5].format, Format::R16_SINT);
}

/// Each attribute's format must know its own size, and that size must
/// match the Rust field it was derived from. This is what catches a
/// `Format` constant added without a matching `bytes_per_pixel` arm.
#[test]
fn every_derived_format_reports_a_size_matching_its_rust_field() {
    let int_sizes = [4u32, 8, 12, 16];
    let int_attrs = IntVertex::attributes(0);
    // `zip` STOPS AT THE SHORTER SIDE. Without this, a derive emitting fewer
    // attributes than expected would silently drop the trailing expectations —
    // and emitting NONE would run the loop zero times and pass having asserted
    // nothing. The loop cannot tell "all correct" from "none checked".
    assert_eq!(
        int_attrs.len(),
        int_sizes.len(),
        concat!(
            "IntVertex derived {} attributes but {} sizes are expected; the ",
            "zip below would silently compare only the shorter prefix"
        ),
        int_attrs.len(),
        int_sizes.len()
    );
    for (attr, expected) in int_attrs.iter().zip(int_sizes) {
        assert_eq!(
            attr.format.bytes_per_pixel(),
            Some(expected),
            "format {:?} at location {} has no size or the wrong one",
            attr.format,
            attr.location
        );
    }

    let mixed_sizes = [12u32, 8, 16, 4, 2, 2];
    let mixed_attrs = MixedVertex::attributes(0);
    assert_eq!(
        mixed_attrs.len(),
        mixed_sizes.len(),
        concat!(
            "MixedVertex derived {} attributes but {} sizes are expected; ",
            "see the note on the zip above"
        ),
        mixed_attrs.len(),
        mixed_sizes.len()
    );
    for (attr, expected) in mixed_attrs.iter().zip(mixed_sizes) {
        assert_eq!(
            attr.format.bytes_per_pixel(),
            Some(expected),
            "format {:?} at location {} has no size or the wrong one",
            attr.format,
            attr.location
        );
    }
}

#[test]
fn locations_are_sequential_and_offsets_follow_repr_c() {
    let attrs = IntVertex::attributes(7);

    for (i, attr) in attrs.iter().enumerate() {
        assert_eq!(attr.location, i as u32);
        assert_eq!(attr.binding, 7, "binding argument must be threaded through");
    }

    assert_eq!(attrs[0].offset, 0);
    assert_eq!(attrs[1].offset, 4);
    assert_eq!(attrs[2].offset, 12);
    assert_eq!(attrs[3].offset, 24);
}

#[test]
fn binding_stride_is_the_struct_size() {
    let binding = IntVertex::binding(3);
    assert_eq!(binding.binding, 3);
    assert_eq!(binding.stride, size_of::<IntVertex>() as u32);
    assert_eq!(binding.stride, 40);
    assert_eq!(binding.input_rate, InputRate::VERTEX);

    let instance = IntVertex::instance_binding(3);
    assert_eq!(instance.stride, binding.stride);
    assert_eq!(instance.input_rate, InputRate::INSTANCE);
}
