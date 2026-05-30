// LayerNorm last-dim — bf16. Packed-u32 storage with lane-pair reads
// and pair-thread Phase-3 writes (no race; each lane handles one u32
// pair per iter). Mixed precision: math in f32.
//
// PRECONDITION: n_cols must be even.

struct Params {
    uint n_rows;
    uint n_cols;
    float eps;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   params;

groupshared float wg_data[32];

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
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint n_pairs = params.n_cols >> 1u;
    uint row_off_u32 = row * n_pairs;

    // ---- Phase 1: mean ----
    float local_sum = 0.0;
    for (uint pair = tid; pair < n_pairs; pair += 256u) {
        uint packed = input[row_off_u32 + pair];
        local_sum += bf16_bits_to_f32(packed & 0xFFFFu)
                   + bf16_bits_to_f32(packed >> 16u);
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
    float mean = wg_data[0] / float(params.n_cols);

    // ---- Phase 2: variance ----
    GroupMemoryBarrierWithGroupSync();
    float local_var = 0.0;
    for (uint pair = tid; pair < n_pairs; pair += 256u) {
        uint packed = input[row_off_u32 + pair];
        float v0 = bf16_bits_to_f32(packed & 0xFFFFu) - mean;
        float v1 = bf16_bits_to_f32(packed >> 16u) - mean;
        local_var += v0 * v0 + v1 * v1;
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
    float var = wg_data[0] / float(params.n_cols);
    float inv_std = 1.0 / sqrt(var + params.eps);

    // ---- Phase 3: normalize ----
    for (uint pair = tid; pair < n_pairs; pair += 256u) {
        uint packed = input[row_off_u32 + pair];
        float v0 = (bf16_bits_to_f32(packed & 0xFFFFu) - mean) * inv_std;
        float v1 = (bf16_bits_to_f32(packed >> 16u)    - mean) * inv_std;
        uint out_lo = f32_to_bf16_bits(v0);
        uint out_hi = f32_to_bf16_bits(v1);
        output[row_off_u32 + pair] = out_lo | (out_hi << 16u);
    }
}
