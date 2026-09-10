// SPDX-License-Identifier: MIT OR Apache-2.0
// Depth prepass + color pass shaders.
//
// vs_main: hardcoded triangle vertices, pass-through position.
// fs_depth: minimal fragment shader for the depth-only pass (writes nothing).
// fs_color: outputs a solid orange color for the color pass.

struct VertexOut {
    @builtin(position) position: vec4<f32>,
};

// Hardcoded triangle covering roughly the center of the viewport.
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>( 0.0, -0.7),
        vec2<f32>( 0.7,  0.7),
        vec2<f32>(-0.7,  0.7),
    );
    var out: VertexOut;
    // Place at z=0.5 so depth testing is meaningful.
    out.position = vec4<f32>(positions[vid], 0.5, 1.0);
    return out;
}

// Depth-only pass: no color output, just let the rasterizer write depth.
@fragment
fn fs_depth() {}

// Color pass: output a solid orange.
@fragment
fn fs_color() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.6, 0.1, 1.0);
}
