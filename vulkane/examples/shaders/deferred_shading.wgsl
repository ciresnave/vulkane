// Deferred shading: G-buffer pass + lighting pass.
//
// G-buffer pass writes to 3 color attachments:
//   location(0) = world position (RGBA32F)
//   location(1) = world normal   (RGBA32F)
//   location(2) = albedo color   (RGBA8)
//
// Lighting pass reads the G-buffer textures and computes Phong lighting.

// --- G-buffer pass ---

struct GbufferVertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct GbufferFragOut {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) albedo: vec4<f32>,
};

// Hardcoded scene: ground plane + colored triangle.
// 6 ground vertices + 3 triangle vertices = 9 total.
@vertex
fn vs_gbuffer(@builtin(vertex_index) vid: u32) -> GbufferVertexOut {
    // Positions.
    var positions = array<vec3<f32>, 9>(
        vec3<f32>(-1.0, -0.3, -1.0),
        vec3<f32>( 1.0, -0.3, -1.0),
        vec3<f32>( 1.0, -0.3,  1.0),
        vec3<f32>(-1.0, -0.3, -1.0),
        vec3<f32>( 1.0, -0.3,  1.0),
        vec3<f32>(-1.0, -0.3,  1.0),
        vec3<f32>( 0.0,  0.5,  0.0),
        vec3<f32>( 0.4, -0.1,  0.3),
        vec3<f32>(-0.4, -0.1, -0.3),
    );
    // Normals.
    var normals = array<vec3<f32>, 9>(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    // Colors: grey ground, green triangle.
    var colors = array<vec3<f32>, 9>(
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.2, 0.8, 0.2),
        vec3<f32>(0.2, 0.8, 0.2),
        vec3<f32>(0.2, 0.8, 0.2),
    );

    var out: GbufferVertexOut;
    let pos = positions[vid];
    // Simple orthographic projection.
    out.position = vec4<f32>(pos.x * 0.6, pos.y * 0.6 + 0.1, pos.z * 0.3 + 0.5, 1.0);
    out.world_pos = pos;
    out.normal = normals[vid];
    out.color = colors[vid];
    return out;
}

@fragment
fn fs_gbuffer(in: GbufferVertexOut) -> GbufferFragOut {
    var out: GbufferFragOut;
    out.position = vec4<f32>(in.world_pos, 1.0);
    out.normal = vec4<f32>(normalize(in.normal), 0.0);
    out.albedo = vec4<f32>(in.color, 1.0);
    return out;
}

// --- Lighting pass ---

struct LightVertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle (3 vertices cover the entire screen).
@vertex
fn vs_lighting(@builtin(vertex_index) vid: u32) -> LightVertexOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: LightVertexOut;
    out.position = vec4<f32>(positions[vid], 0.0, 1.0);
    out.uv = positions[vid] * 0.5 + 0.5;
    return out;
}

@group(0) @binding(0) var g_position: texture_2d<f32>;
@group(0) @binding(1) var g_normal: texture_2d<f32>;
@group(0) @binding(2) var g_albedo: texture_2d<f32>;
@group(0) @binding(3) var g_sampler: sampler;

@fragment
fn fs_lighting(in: LightVertexOut) -> @location(0) vec4<f32> {
    let pos = textureSample(g_position, g_sampler, in.uv).xyz;
    let normal = textureSample(g_normal, g_sampler, in.uv).xyz;
    let albedo = textureSample(g_albedo, g_sampler, in.uv).rgb;

    // Simple directional light.
    let light_dir = normalize(vec3<f32>(1.0, 2.0, 1.5));
    let ambient = 0.15;
    let diffuse = max(dot(normal, light_dir), 0.0);
    let lit = albedo * (ambient + diffuse);

    // If normal is zero-length, this pixel wasn't written by the G-buffer
    // pass — it's background. Output dark blue.
    let has_geometry = step(0.01, length(normal));
    let final_color = mix(vec3<f32>(0.05, 0.05, 0.15), lit, has_geometry);

    return vec4<f32>(final_color, 1.0);
}
