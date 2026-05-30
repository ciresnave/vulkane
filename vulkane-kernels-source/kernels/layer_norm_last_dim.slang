// Fused layer normalization along the last dimension (f32).
//
// Formula:  y[i] = (x[i] - mean) / sqrt(var + eps)
//   mean = sum(x) / n
//   var  = sum((x - mean)²) / n
//
// Two-pass: first reduction gets the mean (subgroup-tree), second
// reduction gets the variance from (x - mean)² (subgroup-tree).
// Both reductions use the same cross-subgroup clamp shape as the
// RmsNorm kernel.
//
// Input/output: [n_rows, n_cols] contiguous, row-major, f32.

struct Params {
    uint n_rows;
    uint n_cols;
    float eps;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>    params;

groupshared float wg_data[32];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint row_offset = row * params.n_cols;

    // ---- Phase 1: mean ----
    float local_sum = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        local_sum += input[row_offset + col];
    }
    float sg_sum = WaveActiveSum(local_sum);
    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        wg_data[sg_id] = sg_sum;
    }
    GroupMemoryBarrierWithGroupSync();
    uint n_subgroups = 256u / WaveGetLaneCount();
    if (sg_id == 0u) {
        float v = (tid < n_subgroups) ? wg_data[tid] : 0.0;
        float s = WaveActiveSum(v);
        if (WaveIsFirstLane()) {
            wg_data[0] = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float total_sum = wg_data[0];
    float mean = total_sum / float(params.n_cols);

    // ---- Phase 2: variance ----
    GroupMemoryBarrierWithGroupSync();   // barrier between reductions
    float local_var = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        float d = input[row_offset + col] - mean;
        local_var += d * d;
    }
    float sg_var = WaveActiveSum(local_var);
    if (WaveIsFirstLane()) {
        wg_data[sg_id] = sg_var;
    }
    GroupMemoryBarrierWithGroupSync();
    if (sg_id == 0u) {
        float v = (tid < n_subgroups) ? wg_data[tid] : 0.0;
        float s = WaveActiveSum(v);
        if (WaveIsFirstLane()) {
            wg_data[0] = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float total_var = wg_data[0];
    float var = total_var / float(params.n_cols);
    float inv_std = 1.0 / sqrt(var + params.eps);

    // ---- Phase 3: normalize ----
    for (uint col = tid; col < params.n_cols; col += 256u) {
        output[row_offset + col] = (input[row_offset + col] - mean) * inv_std;
    }
}
