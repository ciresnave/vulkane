// Parallel reduction: sum, max, or min of all elements → single value.
// One workgroup, tree reduction.

struct Params {
    n: u32,
    op_id: u32, // 0=sum, 1=max, 2=min
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const OP_SUM: u32 = 0u;
const OP_MAX: u32 = 1u;
const OP_MIN: u32 = 2u;

var<workgroup> wg_data: array<f32, 256>;

fn identity(op: u32) -> f32 {
    switch op {
        case OP_SUM { return 0.0; }
        case OP_MAX { return -3.402823e+38; }
        case OP_MIN { return 3.402823e+38; }
        default     { return 0.0; }
    }
}

fn combine(op: u32, a: f32, b: f32) -> f32 {
    switch op {
        case OP_SUM { return a + b; }
        case OP_MAX { return max(a, b); }
        case OP_MIN { return min(a, b); }
        default     { return a; }
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let op = params.op_id;

    // Each thread accumulates a stride of the input.
    var acc = identity(op);
    var i = tid;
    while i < params.n {
        acc = combine(op, acc, input[i]);
        i += 256u;
    }
    wg_data[tid] = acc;
    workgroupBarrier();

    // Tree reduction.
    var stride = 128u;
    while stride > 0u {
        if tid < stride {
            wg_data[tid] = combine(op, wg_data[tid], wg_data[tid + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if tid == 0u {
        output[0] = wg_data[0];
    }
}
