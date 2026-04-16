// Fused root-mean-square normalization along the last dimension.
//
// Formula:  y = x / sqrt(mean(x², last) + eps)
//
// Input:  [n_rows, n_cols]  contiguous, row-major.
// Output: [n_rows, n_cols]  same shape.
//
// Replaces the WGSL `rms_norm_last_dim.wgsl`. Differences:
//   * Subgroup reduction (`WaveActiveSum`) instead of a 256-element
//     tree reduction through shared memory — lower shared-memory
//     traffic and fewer barrier roundtrips.
//   * Written in Slang so future fused kernels can share the
//     `rms_norm_rrms` helper and, once we adopt `[Differentiable]`
//     patterns more broadly, get a free backward kernel.
//
// Binding layout matches the WGSL version so it's a drop-in swap:
//   set 0, binding 0: input   (storage, readonly)
//   set 0, binding 1: output  (storage, readwrite)
//   set 0, binding 2: params  (uniform buffer)
//
// Workgroup size 256; one workgroup per row. On an NVIDIA GPU that's
// 8 subgroups of 32 lanes — the final cross-subgroup reduction uses
// one `WaveActiveSum` over at most 32 partial sums.

struct Params {
    uint n_rows;
    uint n_cols;
    float eps;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>    params;

// One slot per possible subgroup (NVIDIA=32 lanes, AMD=32 or 64,
// Intel=8/16/32). With workgroup size 256, max subgroups is 256.
// In practice <=32 covers every desktop GPU.
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

    // Phase 1 — each thread computes a partial sum(x²) over a strided
    // slice of the row.
    float local_sumsq = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        float v = input[row_offset + col];
        local_sumsq += v * v;
    }

    // Phase 2 — reduce within each subgroup, then across subgroups.
    float sg_sum = WaveActiveSum(local_sumsq);
    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        wg_subgroup_sums[sg_id] = sg_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    // The first subgroup pulls in all per-subgroup partials and
    // reduces them. n_subgroups = 256 / lane_count, always <= 32, so
    // a single WaveActiveSum over the first subgroup suffices.
    float total_sumsq = 0.0;
    uint n_subgroups = 256u / WaveGetLaneCount();
    if (sg_id == 0u) {
        float v = (tid < n_subgroups) ? wg_subgroup_sums[tid] : 0.0;
        float s = WaveActiveSum(v);
        if (WaveIsFirstLane()) {
            // Stash in slot 0 so the other subgroups can read it.
            wg_subgroup_sums[0] = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    total_sumsq = wg_subgroup_sums[0];

    float mean_sq = total_sumsq / float(params.n_cols);
    float r_rms = 1.0 / sqrt(mean_sq + params.eps);

    // Phase 3 — scale output.
    for (uint col = tid; col < params.n_cols; col += 256u) {
        output[row_offset + col] = input[row_offset + col] * r_rms;
    }
}
