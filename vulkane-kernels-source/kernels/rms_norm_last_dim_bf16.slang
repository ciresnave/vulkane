// Fused root-mean-square normalization along the last dimension — bf16.
//
// Formula:  y = x / sqrt(mean(x², last) + eps)
//
// Input:  [n_rows, n_cols]  contiguous, row-major, bf16 packed
//                           two-per-u32 (lane 0 = low 16 bits, lane 1
//                           = high 16 bits — matches binary_bf16.slang).
// Output: [n_rows, n_cols]  same shape, same packing.
//
// PRECONDITION: n_cols MUST be even. The caller (VulkanBackend's
// rms_norm_last_dim_bf16_bytes) is responsible; LLM hidden_dim is
// always even so this is a free assumption in practice.
//
// Mixed precision: storage is bf16 (u32-packed), accumulation + rsqrt
// are f32. Same precedent as f16/CUDA-baracuda.
//
// Lane-pair scheme (option A): each lane processes one packed u32 per
// iteration — that's TWO bf16 lanes per thread per iter, 512 bf16 per
// workgroup iter. Avoids the write-race on bf16-pair u32 words.
//
// No special Slang extension needed — uses raw `uint` storage and
// manual bit unpack/pack (NOT VK_KHR_shader_bfloat16, which has poor
// cross-vendor support).
//
// Binding layout:
//   set 0, binding 0: input   (storage, readonly, uint = packed bf16)
//   set 0, binding 1: output  (storage, readwrite, uint = packed bf16)
//   set 0, binding 2: params  (uniform buffer)

struct Params {
    uint  n_rows;
    uint  n_cols;     // bf16 element count (NOT u32 word count)
    float eps;
    uint  _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;     // packed bf16
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   params;

groupshared float wg_subgroup_sums[32];

float bf16_bits_to_f32(uint b) { return asfloat(b << 16u); }

uint f32_to_bf16_bits(float x) {
    uint bits = asuint(x);
    uint exp_bits = (bits >> 23u) & 0xFFu;
    uint mant_low = bits & 0x7FFFFu;
    if (exp_bits == 0xFFu && mant_low != 0u) {
        return (bits & 0x80000000u) >> 16u | 0x7FC0u;
    }
    uint lsb = (bits >> 16u) & 1u;
    uint rounded = bits + 0x7FFFu + lsb;
    return (rounded >> 16u) & 0xFFFFu;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
    uint3 gid   : SV_GroupID,
    uint3 gtid  : SV_GroupThreadID)
{
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint n_pairs = params.n_cols >> 1u;                 // n_cols / 2
    uint row_offset_u32 = row * n_pairs;

    // Phase 1 — sum(x²) in f32 across all bf16 lanes in this row. Each
    // lane processes one u32 word = two bf16 lanes per iter.
    float local_sumsq = 0.0;
    for (uint pair = tid; pair < n_pairs; pair += 256u) {
        uint packed = input[row_offset_u32 + pair];
        float v0 = bf16_bits_to_f32(packed & 0xFFFFu);
        float v1 = bf16_bits_to_f32(packed >> 16u);
        local_sumsq += v0 * v0 + v1 * v1;
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

    // Phase 3 — scale, repack, store. One u32 per lane per iter — no
    // write race because adjacent bf16 pairs live in distinct u32 words
    // assigned to distinct lanes.
    for (uint pair = tid; pair < n_pairs; pair += 256u) {
        uint packed_in = input[row_offset_u32 + pair];
        float v0 = bf16_bits_to_f32(packed_in & 0xFFFFu);
        float v1 = bf16_bits_to_f32(packed_in >> 16u);
        uint out_lo = f32_to_bf16_bits(v0 * r_rms);
        uint out_hi = f32_to_bf16_bits(v1 * r_rms);
        output[row_offset_u32 + pair] = out_lo | (out_hi << 16u);
    }
}
