// Fused root-mean-square normalization along the last dimension — f16.
//
// Formula:  y = x / sqrt(mean(x², last) + eps)
//
// Input:  [n_rows, n_cols]  contiguous, row-major, float16_t.
// Output: [n_rows, n_cols]  same shape, float16_t.
//
// Mixed precision: storage is f16, accumulation + rsqrt are f32. f16's
// 10-bit mantissa cannot resolve sum-of-squares across long rows (e.g.
// hidden_dim=4096) without catastrophic underflow once the partial sum
// grows. Matches baracuda's CUDA pattern ("f16/bf16 accumulate in f32").
//
// Requires VK_KHR_16bit_storage + shaderFloat16 (same as unary_f16).
//
// Binding layout (matches f32):
//   set 0, binding 0: input   (storage, readonly, float16_t)
//   set 0, binding 1: output  (storage, readwrite, float16_t)
//   set 0, binding 2: params  (uniform buffer)
//
// One workgroup per row; subgroup-tree reduction; cross-subgroup clamp
// `(tid < n_subgroups)` preserved from f32 to avoid the LLaMA-NaN
// composed-divergence bug fixed in the CUDA path.

struct Params {
    uint n_rows;
    uint n_cols;
    float eps;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float16_t>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float16_t> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>        params;

groupshared float wg_subgroup_sums[32];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
    uint3 gid   : SV_GroupID,
    uint3 gtid  : SV_GroupThreadID)
{
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint row_offset = row * params.n_cols;

    // Phase 1 — partial sum(x²) in f32. Cast f16 → f32 at load.
    float local_sumsq = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        float v = float(input[row_offset + col]);
        local_sumsq += v * v;
    }

    // Phase 2 — subgroup-tree reduction in f32.
    float sg_sum = WaveActiveSum(local_sumsq);
    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        wg_subgroup_sums[sg_id] = sg_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    float total_sumsq = 0.0;
    uint n_subgroups = 256u / WaveGetLaneCount();
    if (sg_id == 0u) {
        float v = (tid < n_subgroups) ? wg_subgroup_sums[tid] : 0.0;
        float s = WaveActiveSum(v);
        if (WaveIsFirstLane()) {
            wg_subgroup_sums[0] = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    total_sumsq = wg_subgroup_sums[0];

    float mean_sq = total_sumsq / float(params.n_cols);
    float r_rms = 1.0 / sqrt(mean_sq + params.eps);

    // Phase 3 — scale output. Multiply in f32, cast to f16 at store.
    for (uint col = tid; col < params.n_cols; col += 256u) {
        float v = float(input[row_offset + col]);
        output[row_offset + col] = float16_t(v * r_rms);
    }
}
