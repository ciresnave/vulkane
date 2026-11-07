// SPDX-License-Identifier: MIT OR Apache-2.0
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> tile: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn sum_f32(@builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(workgroup_id) wid: vec3<u32>) {
    let i = wid.x * 256u + lid.x;
    var v: f32 = 0.0;
    if (i < params.n) { v = src[i]; }
    tile[lid.x] = v;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) { tile[lid.x] = tile[lid.x] + tile[lid.x + s]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) { dst[wid.x] = tile[0]; }
}
