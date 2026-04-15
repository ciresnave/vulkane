//! Tests for the optional `shaderc` GLSL -> SPIR-V compilation feature.
//!
//! Only compiled when the `shaderc` feature is enabled.

#![cfg(feature = "shaderc")]

use vulkane::safe::shaderc::{ShaderKind, compile_glsl};

#[test]
fn test_compile_trivial_compute_shader() {
    let glsl = r#"
        #version 450
        layout(local_size_x = 1) in;
        void main() {}
    "#;

    let words = compile_glsl(glsl, ShaderKind::Compute, "trivial.comp", "main")
        .expect("trivial compute shader compiles");

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

    let words = compile_glsl(glsl, ShaderKind::Compute, "doubler.comp", "main")
        .expect("storage buffer compute shader compiles");

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

    let result = compile_glsl(bad_glsl, ShaderKind::Compute, "bad.comp", "main");
    assert!(
        result.is_err(),
        "invalid GLSL should produce a compile error"
    );
}

#[test]
fn test_compile_vertex_and_fragment_pair() {
    let vs = r#"
        #version 450
        layout(location = 0) in vec2 position;
        void main() { gl_Position = vec4(position, 0.0, 1.0); }
    "#;
    let fs = r#"
        #version 450
        layout(location = 0) out vec4 color;
        void main() { color = vec4(1.0); }
    "#;

    let vs_words = compile_glsl(vs, ShaderKind::Vertex, "triangle.vert", "main")
        .expect("vertex shader compiles");
    let fs_words = compile_glsl(fs, ShaderKind::Fragment, "triangle.frag", "main")
        .expect("fragment shader compiles");

    assert_eq!(vs_words[0], 0x07230203);
    assert_eq!(fs_words[0], 0x07230203);
}
