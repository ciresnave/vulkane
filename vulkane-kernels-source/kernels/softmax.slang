// Fused softmax along the last dimension.
//
// Formula:  y_j = exp(x_j - max) / sum_i(exp(x_i - max))
//
// One workgroup per row. Replaces the WGSL softmax's 256-element
// tree reductions with subgroup reductions (`WaveActiveMax`,
// `WaveActiveSum`) plus a single cross-subgroup fixup.
//
// Binding layout matches the WGSL version bit-for-bit so this is a
// drop-in swap:
//   set 0, binding 0: input   (storage, readonly)
//   set 0, binding 1: output  (storage, readwrite)
//   set 0, binding 2: params  (uniform buffer)
//
// Workgroup size 256; subgroup arithmetic requires SPIR-V 1.3
// (auto-upgraded by slangc when these ops are used).

struct Params {
    uint n_rows;
    uint n_cols;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>    params;

// Same cap as rms_norm — 32 slots covers every desktop subgroup
// size at workgroup size 256.
groupshared float wg_max[32];
groupshared float wg_sum[32];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint row_off = row * params.n_cols;

    // Phase 1 — per-thread max over a strided slice.
    float local_max = -3.402823e+38; // -FLT_MAX
    for (uint col = tid; col < params.n_cols; col += 256u) {
        float v = input[row_off + col];
        local_max = max(local_max, v);
    }

    // Reduce max within subgroup, then across subgroups.
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

    // Phase 2 — shift + exp, parallel sum.
    float local_sum = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        float v = exp(input[row_off + col] - row_max);
        output[row_off + col] = v;
        local_sum += v;
    }

    // Reduce sum — same pattern.
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

    // Phase 3 — normalize.
    float inv_sum = 1.0 / row_sum;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        output[row_off + col] *= inv_sum;
    }
}
