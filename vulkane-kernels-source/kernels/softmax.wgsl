// Fused softmax along the last dimension.
// One workgroup per row. Each thread handles a subset of columns.

struct Params {
    n_rows: u32,
    n_cols: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    if row >= params.n_rows { return; }

    let tid = lid.x;
    let row_offset = row * params.n_cols;

    // Step 1: find row max (parallel reduction).
    var local_max: f32 = -3.402823e+38; // -FLT_MAX
    var col = tid;
    while col < params.n_cols {
        local_max = max(local_max, input[row_offset + col]);
        col += 256u;
    }
    shared_max[tid] = local_max;
    workgroupBarrier();

    // Tree reduction for max.
    var stride = 128u;
    while stride > 0u {
        if tid < stride {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let row_max = shared_max[0];
    workgroupBarrier();

    // Step 2: compute exp(x - max) and sum.
    var local_sum: f32 = 0.0;
    col = tid;
    while col < params.n_cols {
        let val = exp(input[row_offset + col] - row_max);
        output[row_offset + col] = val; // temporarily store exp values
        local_sum += val;
        col += 256u;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction for sum.
    stride = 128u;
    while stride > 0u {
        if tid < stride {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let row_sum = shared_sum[0];
    workgroupBarrier();

    // Step 3: normalize.
    let inv_sum = 1.0 / row_sum;
    col = tid;
    while col < params.n_cols {
        output[row_offset + col] *= inv_sum;
        col += 256u;
    }
}
