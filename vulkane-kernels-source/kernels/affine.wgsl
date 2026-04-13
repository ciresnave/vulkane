// Affine: y = x * mul + add. Handles AddScalar and MulScalar.

struct Params {
    n: u32,
    _pad: u32,
    mul: f32,
    add: f32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }
    output[idx] = input[idx] * params.mul + params.add;
}
