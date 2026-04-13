// Element-wise binary operations. One thread per element.

struct Params {
    n: u32,
    op_id: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const OP_ADD: u32     = 0u;
const OP_SUB: u32     = 1u;
const OP_MUL: u32     = 2u;
const OP_DIV: u32     = 3u;
const OP_MAX: u32     = 4u;
const OP_MIN: u32     = 5u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }

    let x = a[idx];
    let y = b[idx];
    var z: f32;

    switch params.op_id {
        case OP_ADD { z = x + y; }
        case OP_SUB { z = x - y; }
        case OP_MUL { z = x * y; }
        case OP_DIV { z = x / y; }
        case OP_MAX { z = max(x, y); }
        case OP_MIN { z = min(x, y); }
        default     { z = x; }
    }

    output[idx] = z;
}
