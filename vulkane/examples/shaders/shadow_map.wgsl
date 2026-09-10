// SPDX-License-Identifier: MIT OR Apache-2.0
// Shadow map shaders.
//
// vs_depth + fs_depth: depth-only pass from the light's perspective.
// vs_main + fs_main: color pass from the camera, sampling the shadow map.

// --- Depth pass (light POV) ---

struct DepthVertexOut {
    @builtin(position) position: vec4<f32>,
};

// Hardcoded ground plane + raised triangle.
// 6 vertices for a ground quad + 3 for a floating triangle = 9 total.
fn get_scene_vertex(vid: u32) -> vec3<f32> {
    var verts = array<vec3<f32>, 9>(
        // Ground quad (two triangles).
        vec3<f32>(-1.0, -0.5, -1.0),
        vec3<f32>( 1.0, -0.5, -1.0),
        vec3<f32>( 1.0, -0.5,  1.0),
        vec3<f32>(-1.0, -0.5, -1.0),
        vec3<f32>( 1.0, -0.5,  1.0),
        vec3<f32>(-1.0, -0.5,  1.0),
        // Floating triangle (casts shadow on the ground).
        vec3<f32>( 0.0,  0.3,  0.0),
        vec3<f32>( 0.4, -0.1,  0.3),
        vec3<f32>(-0.4, -0.1, -0.3),
    );
    return verts[vid];
}

// Push constants: 4x4 MVP matrix.
struct MVP {
    mvp: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> light_mvp: MVP;

@vertex
fn vs_depth(@builtin(vertex_index) vid: u32) -> DepthVertexOut {
    var out: DepthVertexOut;
    let pos = get_scene_vertex(vid);
    out.position = light_mvp.mvp * vec4<f32>(pos, 1.0);
    return out;
}

@fragment
fn fs_depth() {}

// --- Main pass (camera POV) ---

struct MainVertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) light_space_pos: vec4<f32>,
    @location(1) world_y: f32,
};

@group(0) @binding(1) var<uniform> camera_mvp: MVP;
@group(0) @binding(2) var shadow_tex: texture_depth_2d;
@group(0) @binding(3) var shadow_samp: sampler_comparison;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> MainVertexOut {
    var out: MainVertexOut;
    let pos = get_scene_vertex(vid);
    out.position = camera_mvp.mvp * vec4<f32>(pos, 1.0);
    out.light_space_pos = light_mvp.mvp * vec4<f32>(pos, 1.0);
    out.world_y = pos.y;
    return out;
}

@fragment
fn fs_main(in: MainVertexOut) -> @location(0) vec4<f32> {
    // Project light-space position to UV + depth.
    let ndc = in.light_space_pos.xyz / in.light_space_pos.w;
    let uv = ndc.xy * 0.5 + 0.5;
    let shadow_depth = ndc.z;

    // Shadow test using comparison sampler.
    let lit = textureSampleCompare(shadow_tex, shadow_samp, uv, shadow_depth);

    // Ground = grey, triangle = green, darkened in shadow.
    let is_ground = select(0.0, 1.0, in.world_y < -0.3);
    let base_color = mix(vec3<f32>(0.2, 0.8, 0.2), vec3<f32>(0.6, 0.6, 0.6), is_ground);
    let final_color = base_color * (0.3 + 0.7 * lit);
    return vec4<f32>(final_color, 1.0);
}
