// SPDX-License-Identifier: MIT OR Apache-2.0
// Instanced mesh shaders.
//
// vs_main: per-vertex position (binding 0) + per-instance offset (binding 1).
// fs_main: simple color based on instance ID for visual variety.

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) instance_offset: vec3<f32>,
    @builtin(instance_index) iid: u32,
) -> VertexOut {
    var out: VertexOut;
    let world_pos = pos + instance_offset;
    // Simple orthographic-ish projection: scale down and center.
    out.position = vec4<f32>(world_pos.x * 0.1, world_pos.y * 0.1, world_pos.z * 0.5 + 0.5, 1.0);
    // Color varies by instance ID for visual variety.
    let r = f32((iid * 37u) % 256u) / 255.0;
    let g = f32((iid * 73u) % 256u) / 255.0;
    let b = f32((iid * 127u) % 256u) / 255.0;
    out.color = vec3<f32>(r, g, b);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
