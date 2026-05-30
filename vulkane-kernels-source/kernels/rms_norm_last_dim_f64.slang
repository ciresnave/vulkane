// Fused root-mean-square normalization along the last dimension — f64.
//
// Formula:  y = x / sqrt(mean(x², last) + eps)
//
// Input:  [n_rows, n_cols]  contiguous, row-major, double.
// Output: [n_rows, n_cols]  same shape, double.
//
// Native f64 end-to-end (matches baracuda's CUDA pattern: "f32/f64 use
// native precision throughout"). Requires shaderFloat64 +
// GroupNonUniformArithmetic capabilities; RTX 4070 + recent NVIDIA
// drivers expose both. Uses GLSL.std.450 Sqrt on doubles for the rsqrt,
// NOT OpenCL.std (which is unusable on NVIDIA Windows Vulkan — see
// feedback_opencl_std_not_usable_vulkan_nvidia_windows).
//
// Binding layout (matches f32, but Params.eps is double so the struct
// is still 16 bytes — { u32, u32, f64 } with 8-byte alignment):
//   set 0, binding 0: input   (storage, readonly, double)
//   set 0, binding 1: output  (storage, readwrite, double)
//   set 0, binding 2: params  (uniform buffer)
//
// One workgroup per row; subgroup-tree reduction in f64; cross-subgroup
// clamp preserved.

struct Params {
    uint   n_rows;
    uint   n_cols;
    double eps;
};

[[vk::binding(0, 0)]] StructuredBuffer<double>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>     params;

groupshared double wg_subgroup_sums[32];

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

    double local_sumsq = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        double v = input[row_offset + col];
        local_sumsq += v * v;
    }

    double sg_sum = WaveActiveSum(local_sumsq);
    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        wg_subgroup_sums[sg_id] = sg_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    double total_sumsq = 0.0;
    uint n_subgroups = 256u / WaveGetLaneCount();
    if (sg_id == 0u) {
        double v = (tid < n_subgroups) ? wg_subgroup_sums[tid] : 0.0;
        double s = WaveActiveSum(v);
        if (WaveIsFirstLane()) {
            wg_subgroup_sums[0] = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    total_sumsq = wg_subgroup_sums[0];

    double mean_sq = total_sumsq / double(params.n_cols);
    double r_rms = 1.0 / sqrt(mean_sq + params.eps);

    for (uint col = tid; col < params.n_cols; col += 256u) {
        output[row_offset + col] = input[row_offset + col] * r_rms;
    }
}
