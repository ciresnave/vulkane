// Fused backward for `rms_norm_last_dim`.
//
// Forward:  y_j = x_j / sqrt(mean(x², last) + eps)
//             = x_j * r_rms        where r_rms = 1 / sqrt(mean_sq + eps)
//
// Backward (derived by closed-form, not autodiff — RMSNorm has a
// clean analytic gradient that's cheaper to code directly than to
// coax Slang's __diff into handling the reduction correctly):
//
//   let s       = sum_i(g_y_i * x_i)
//   let mean_sq = mean(x²)
//   grad_x_j    = r_rms * (g_y_j - x_j * s / (n * (mean_sq + eps)))
//
// Needs two reductions per row: sum(x²) and sum(g_y * x). Both run
// through the same subgroup-reduce pattern as the forward kernel,
// back-to-back, in a single dispatch. Replaces the 9-node primitive
// decomposition in `Tensor::backward` (sqr → mean_dim → add_scalar →
// sqrt → mul → sum_dim → mul_scalar → div → reshape → broadcast →
// mul → sub → div) that currently runs at every RMSNorm site on the
// training path.
//
// Bindings:
//   set 0, binding 0: x        (storage, readonly)
//   set 0, binding 1: upstream (storage, readonly)  // g_y
//   set 0, binding 2: grad_x   (storage, readwrite)
//   set 0, binding 3: params   (uniform buffer)

struct Params {
    uint n_rows;
    uint n_cols;
    float eps;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   x_buf;
[[vk::binding(1, 0)]] StructuredBuffer<float>   g_y_buf;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> grad_x_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>    params;

// Two shared slots: one per reduction. Max 32 subgroups per
// workgroup — covers every desktop GPU at workgroup size 256.
groupshared float wg_sumsq[32];
groupshared float wg_gx[32];

// Reduce a per-thread `value` across a 256-thread workgroup using
// subgroup-level arithmetic. Returns the total on thread 0 only; the
// caller broadcasts via `wg_slot` if other threads need it.
float reduce_workgroup(float value, uint tid, inout float wg_slot[32]) {
    float sg_sum = WaveActiveSum(value);
    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        wg_slot[sg_id] = sg_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    uint n_sg = 256u / WaveGetLaneCount();
    if (sg_id == 0u) {
        float v = (tid < n_sg) ? wg_slot[tid] : 0.0;
        float s = WaveActiveSum(v);
        if (WaveIsFirstLane()) {
            wg_slot[0] = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    return wg_slot[0];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint row_off = row * params.n_cols;

    // Phase 1: compute both row reductions side-by-side.
    float local_sumsq = 0.0;
    float local_gx = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        float xi = x_buf[row_off + col];
        float gi = g_y_buf[row_off + col];
        local_sumsq += xi * xi;
        local_gx += gi * xi;
    }

    float total_sumsq = reduce_workgroup(local_sumsq, tid, wg_sumsq);
    float total_gx    = reduce_workgroup(local_gx,    tid, wg_gx);

    // Phase 2: per-element grad_x.
    float n = float(params.n_cols);
    float mean_sq = total_sumsq / n;
    float denom_sq = mean_sq + params.eps;      // = 1/r_rms²
    float r_rms = 1.0 / sqrt(denom_sq);
    float coeff = total_gx / (n * denom_sq);    // x_j * coeff = second term inside parens

    for (uint col = tid; col < params.n_cols; col += 256u) {
        float xi = x_buf[row_off + col];
        float gi = g_y_buf[row_off + col];
        grad_x_buf[row_off + col] = r_rms * (gi - xi * coeff);
    }
}
