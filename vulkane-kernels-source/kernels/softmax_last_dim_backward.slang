// Fused backward for softmax along the last dimension.
//
// Forward:  y_j = exp(x_j - max) / sum_i(exp(x_i - max))
//
// Backward (closed-form, one reduction):
//   dot   = sum_i(y_i * g_i)
//   dx_j  = y_j * (g_j - dot)
//
// Inputs:
//   y   — forward softmax output  [n_rows, n_cols]
//   g   — upstream gradient        [n_rows, n_cols]
// Output:
//   dx  — gradient of softmax input [n_rows, n_cols]
//
// Bindings match softmax.slang layout + one extra buffer:
//   set 0, binding 0: y     (storage, readonly)
//   set 0, binding 1: g     (storage, readonly)
//   set 0, binding 2: dx    (storage, readwrite)
//   set 0, binding 3: params (uniform)
//
// One workgroup per row, 256 threads, subgroup reduction.

struct Params {
    uint n_rows;
    uint n_cols;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   y_buf;
[[vk::binding(1, 0)]] StructuredBuffer<float>   g_buf;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> dx_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>    params;

groupshared float wg_dot[32];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint row_off = row * params.n_cols;

    // Phase 1: partial dot(y, g) over strided slice.
    float local_dot = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        local_dot += y_buf[row_off + col] * g_buf[row_off + col];
    }

    // Reduce within subgroup, then across subgroups.
    float sg_sum = WaveActiveSum(local_dot);
    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        wg_dot[sg_id] = sg_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    uint n_sg = 256u / WaveGetLaneCount();
    if (sg_id == 0u) {
        float v = (tid < n_sg) ? wg_dot[tid] : 0.0;
        float s = WaveActiveSum(v);
        if (WaveIsFirstLane()) {
            wg_dot[0] = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float dot = wg_dot[0];

    // Phase 2: dx_j = y_j * (g_j - dot).
    for (uint col = tid; col < params.n_cols; col += 256u) {
        float yv = y_buf[row_off + col];
        float gv = g_buf[row_off + col];
        dx_buf[row_off + col] = yv * (gv - dot);
    }
}
