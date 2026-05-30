// Fused softmax along the last dimension — f64.
//
// Formula:  y_j = exp(x_j - max) / sum_i(exp(x_i - max))
//
// Native f64 end-to-end (matches baracuda's CUDA pattern). Requires
// shaderFloat64 + GroupNonUniformArithmetic. Uses GLSL.std.450 Exp /
// Max for double operands — NOT OpenCL.std (broken on NVIDIA Windows
// Vulkan).
//
// Same Params shape (8 bytes) and 3-phase structure as the f32 kernel.

struct Params {
    uint n_rows;
    uint n_cols;
};

[[vk::binding(0, 0)]] StructuredBuffer<double>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>     params;

groupshared double wg_max[32];
groupshared double wg_sum[32];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint row_off = row * params.n_cols;

    // Phase 1 — per-row max in f64. -DBL_MAX as the per-lane sentinel.
    double local_max = -1.7976931348623157e+308;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        double v = input[row_off + col];
        local_max = max(local_max, v);
    }

    double sg_max = WaveActiveMax(local_max);
    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        wg_max[sg_id] = sg_max;
    }
    GroupMemoryBarrierWithGroupSync();

    uint n_sg = 256u / WaveGetLaneCount();
    if (sg_id == 0u) {
        double v = (tid < n_sg) ? wg_max[tid] : -1.7976931348623157e+308;
        double m = WaveActiveMax(v);
        if (WaveIsFirstLane()) {
            wg_max[0] = m;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    double row_max = wg_max[0];

    // Phase 2 — exp + write + accumulate sum.
    double local_sum = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        double v = exp(input[row_off + col] - row_max);
        output[row_off + col] = v;
        local_sum += v;
    }

    double sg_sum = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) {
        wg_sum[sg_id] = sg_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (sg_id == 0u) {
        double v = (tid < n_sg) ? wg_sum[tid] : 0.0;
        double s = WaveActiveSum(v);
        if (WaveIsFirstLane()) {
            wg_sum[0] = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    double row_sum = wg_sum[0];

    // Phase 3 — normalize.
    double inv_sum = 1.0 / row_sum;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        output[row_off + col] *= inv_sum;
    }
}
