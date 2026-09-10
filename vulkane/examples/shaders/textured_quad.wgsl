// SPDX-License-Identifier: MIT OR Apache-2.0
// Textured quad: a 4-vertex triangle strip with hardcoded positions
// and UVs. Samples a texture+sampler pair bound at group 0.
//
// Compiled with naga's WGSL frontend (`safe::naga::compile_wgsl`)
// which handles the separated `texture_2d<f32>` + `sampler` shape
// cleanly — unlike the GLSL frontend, which struggles with
// `uniform sampler2D` syntactic combinations.

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOut {
    // 4 vertices forming a triangle strip covering most of the screen.
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-0.8, -0.8),
        vec2<f32>( 0.8, -0.8),
        vec2<f32>(-0.8,  0.8),
        vec2<f32>( 0.8,  0.8),
    );
    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    var out: VertexOut;
    out.position = vec4<f32>(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv);
}
