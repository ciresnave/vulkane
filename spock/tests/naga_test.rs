//! Tests for the optional `naga` GLSL -> SPIR-V compilation feature.
//!
//! Only compiled when the `naga` feature is enabled. With the feature off,
//! this file produces a single no-op test that documents the situation.

#![cfg(feature = "naga")]

use naga::ShaderStage;
use spock::safe::naga::compile_glsl;

#[test]
fn test_compile_trivial_compute_shader() {
    let glsl = r#"
        #version 450
        layout(local_size_x = 1) in;
        void main() {}
    "#;

    let words = compile_glsl(glsl, ShaderStage::Compute).expect("trivial compute shader compiles");

    // Validate basic SPIR-V structure: starts with the SPIR-V magic number.
    assert!(!words.is_empty(), "compiled SPIR-V should not be empty");
    assert_eq!(
        words[0], 0x07230203,
        "SPIR-V binary should start with the magic number 0x07230203"
    );
}

#[test]
fn test_compile_storage_buffer_compute() {
    let glsl = r#"
        #version 450
        layout(local_size_x = 64) in;
        layout(set = 0, binding = 0, std430) buffer Data {
            uint values[];
        };
        void main() {
            uint i = gl_GlobalInvocationID.x;
            values[i] = values[i] * 2u;
        }
    "#;

    let words =
        compile_glsl(glsl, ShaderStage::Compute).expect("storage buffer compute shader compiles");

    assert!(
        words.len() > 10,
        "non-trivial shader should produce multiple SPIR-V words"
    );
    assert_eq!(words[0], 0x07230203);
}

#[test]
fn test_compile_invalid_glsl_returns_error() {
    let bad_glsl = r#"
        #version 450
        this is not valid GLSL at all
    "#;

    let result = compile_glsl(bad_glsl, ShaderStage::Compute);
    assert!(
        result.is_err(),
        "invalid GLSL should produce a compile error"
    );
}
