// Fused root-mean-square normalization along the last dimension.
//
// Formula:  y = x / sqrt(mean(x², last) + eps)
//
// Input:  [n_rows, n_cols]  contiguous, row-major.
// Output: [n_rows, n_cols]  same shape.
//
// One workgroup per row. Threads cooperate via workgroup-shared memory
// to compute `sum(x²)` in a tree reduction, then each thread divides
// its own elements by the resulting rRMS. Two GPU-side passes over
// the row, no intermediate buffers — replaces the 8-kernel-launch
// decomposition (sqr → sum_dim → mul_scalar → add_scalar → sqrt →
// broadcast alloc+strided → div) with a single dispatch.
//
// Mirrors the style of softmax.wgsl and reduce_last_dim.wgsl so naga
// produces the same shape of SPIR-V we know works on NVIDIA.

struct Params {
    n_rows: u32,
    n_cols: u32,
    eps: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> wg_sumsq: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    if row >= params.n_rows { return; }

    let tid = lid.x;
    let row_offset = row * params.n_cols;

    // Step 1: sum(x²) across the row, striding by 256.
    var local_sumsq: f32 = 0.0;
    var col = tid;
    while col < params.n_cols {
        let v = input[row_offset + col];
        local_sumsq = local_sumsq + v * v;
        col += 256u;
    }
    wg_sumsq[tid] = local_sumsq;
    workgroupBarrier();

    // Tree reduction.
    var stride = 128u;
    while stride > 0u {
        if tid < stride {
            wg_sumsq[tid] = wg_sumsq[tid] + wg_sumsq[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Compute rRMS on every thread (broadcast of wg_sumsq[0] via
    // shared mem + barrier; cheap since all threads read the same
    // bank-0 word).
    let mean_sq = wg_sumsq[0] / f32(params.n_cols);
    let r_rms = 1.0 / sqrt(mean_sq + params.eps);

    // Step 2: scale output.
    col = tid;
    while col < params.n_cols {
        output[row_offset + col] = input[row_offset + col] * r_rms;
        col += 256u;
    }
}
