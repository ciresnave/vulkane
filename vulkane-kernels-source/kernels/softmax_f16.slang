// Fused softmax along the last dimension — f16.
//
// Formula:  y_j = exp(x_j - max) / sum_i(exp(x_i - max))
//
// Mixed precision: storage is f16, intermediate exp + sum + rsqrt in
// f32 (matches baracuda's CUDA pattern). Phase 2 writes `exp(x - max)`
// to the output as f16 (one rounding); Phase 3 reads it back, scales
// by 1/sum in f32, casts back to f16 (second rounding). The
// double-rounding is bounded at ~2 ULP for outputs in [0, 1].
//
// Requires VK_KHR_16bit_storage + shaderFloat16.
//
// Binding layout matches the f32 version exactly (Params is 8 bytes:
// `{ u32 n_rows, u32 n_cols }`).

struct Params {
    uint n_rows;
    uint n_cols;
};

[[vk::binding(0, 0)]] StructuredBuffer<float16_t>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float16_t> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>        params;

groupshared float wg_max[32];
groupshared float wg_sum[32];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint row_off = row * params.n_cols;

    // Phase 1 — per-thread max in f32.
    float local_max = -3.402823e+38;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        float v = float(input[row_off + col]);
        local_max = max(local_max, v);
    }

    float sg_max = WaveActiveMax(local_max);
    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        wg_max[sg_id] = sg_max;
    }
    GroupMemoryBarrierWithGroupSync();

    uint n_sg = 256u / WaveGetLaneCount();
    if (sg_id == 0u) {
        float v = (tid < n_sg) ? wg_max[tid] : -3.402823e+38;
        float m = WaveActiveMax(v);
        if (WaveIsFirstLane()) {
            wg_max[0] = m;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float row_max = wg_max[0];

    // Phase 2 — exp + write intermediate + accumulate sum in f32.
    float local_sum = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        float v = exp(float(input[row_off + col]) - row_max);
        output[row_off + col] = float16_t(v);
        local_sum += v;
    }

    float sg_sum = WaveActiveSum(local_sum);
    if (WaveIsFirstLane()) {
        wg_sum[sg_id] = sg_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (sg_id == 0u) {
        float v = (tid < n_sg) ? wg_sum[tid] : 0.0;
        float s = WaveActiveSum(v);
        if (WaveIsFirstLane()) {
            wg_sum[0] = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float row_sum = wg_sum[0];

    // Phase 3 — normalize. Read back the f16 intermediate, scale in
    // f32, cast to f16 at store.
    float inv_sum = 1.0 / row_sum;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        float v = float(output[row_off + col]) * inv_sum;
        output[row_off + col] = float16_t(v);
    }
}
