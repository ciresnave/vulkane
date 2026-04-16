// In-place scaled accumulate: dst[i] += src[i] * scale.
// One thread per element. Reads dst, reads src, writes dst.

struct Params {
    n: u32,
    scale: f32,
};

@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    dst[i] = dst[i] + src[i] * params.scale;
}
