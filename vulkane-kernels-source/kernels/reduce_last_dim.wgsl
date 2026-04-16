// Per-row reduction along the last dimension.
//
// Input: [n_rows, n_cols]  contiguous, row-major.
// Output: [n_rows]         one element per row.
//
// op_id: 0 = sum, 1 = max, 2 = min.
//
// Style matches softmax.wgsl: one workgroup per row, `while` loops,
// right-shift for halving, and op_id selection inlined per-pass with
// explicit identity values (rather than a helper function that returns
// from multiple branches). Naga's SPIR-V codegen is less surefooted
// around runtime-selected helper-fn returns inside compute loops with
// workgroup barriers; inlining avoids that footgun.

struct Params {
    n_rows: u32,
    n_cols: u32,
    op_id: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> wg_data: array<f32, 256>;

const F32_NEG_INF: f32 = -3.402823e+38;
const F32_POS_INF: f32 =  3.402823e+38;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    if row >= params.n_rows { return; }

    let tid = lid.x;
    let row_offset = row * params.n_cols;
    let op = params.op_id;

    // Identity value for this op.
    var acc: f32;
    if op == 0u {
        acc = 0.0;
    } else if op == 1u {
        acc = F32_NEG_INF;
    } else {
        acc = F32_POS_INF;
    }

    // Per-thread partial across a stride of 256.
    var col = tid;
    while col < params.n_cols {
        let v = input[row_offset + col];
        if op == 0u {
            acc = acc + v;
        } else if op == 1u {
            acc = max(acc, v);
        } else {
            acc = min(acc, v);
        }
        col += 256u;
    }
    wg_data[tid] = acc;
    workgroupBarrier();

    // Tree reduction within the workgroup.
    var stride = 128u;
    while stride > 0u {
        if tid < stride {
            let a = wg_data[tid];
            let b = wg_data[tid + stride];
            if op == 0u {
                wg_data[tid] = a + b;
            } else if op == 1u {
                wg_data[tid] = max(a, b);
            } else {
                wg_data[tid] = min(a, b);
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if tid == 0u {
        output[row] = wg_data[0];
    }
}
