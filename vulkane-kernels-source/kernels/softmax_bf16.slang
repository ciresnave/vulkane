// Fused softmax along the last dimension — bf16.
//
// Formula:  y_j = exp(x_j - max) / sum_i(exp(x_i - max))
//
// Mixed precision: storage is bf16 (packed two-per-u32, lane 0 = low
// 16), all arithmetic in f32. Lane-pair scheme (each lane processes
// one u32 = two bf16) carries through all three phases — Phase 2
// writes an intermediate packed u32 to output, Phase 3 reads back the
// same u32 (per-lane index sets are identical across phases, no
// races) and rescales.
//
// PRECONDITION: n_cols MUST be even. LLM hidden_dim always is.
//
// No special Slang extension needed — uses raw `uint` storage and the
// same bf16<->f32 helpers as binary_bf16.slang.

struct Params {
    uint n_rows;
    uint n_cols;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;     // packed bf16
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   params;

groupshared float wg_max[32];
groupshared float wg_sum[32];

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
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint n_pairs = params.n_cols >> 1u;
    uint row_off_u32 = row * n_pairs;

    // Phase 1 — per-row max over both bf16 lanes in each u32.
    float local_max = -3.402823e+38;
    for (uint pair = tid; pair < n_pairs; pair += 256u) {
        uint packed = input[row_off_u32 + pair];
        float v0 = bf16_bits_to_f32(packed & 0xFFFFu);
        float v1 = bf16_bits_to_f32(packed >> 16u);
        local_max = max(local_max, max(v0, v1));
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

    // Phase 2 — exp + write packed intermediate + accumulate sum in f32.
    float local_sum = 0.0;
    for (uint pair = tid; pair < n_pairs; pair += 256u) {
        uint packed = input[row_off_u32 + pair];
        float v0 = exp(bf16_bits_to_f32(packed & 0xFFFFu) - row_max);
        float v1 = exp(bf16_bits_to_f32(packed >> 16u) - row_max);
        local_sum += v0 + v1;
        uint out_lo = f32_to_bf16_bits(v0);
        uint out_hi = f32_to_bf16_bits(v1);
        output[row_off_u32 + pair] = out_lo | (out_hi << 16u);
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

    // Phase 3 — normalize. Read back the bf16 intermediate, scale in
    // f32, repack, store. Each lane's pair-index set is the same as in
    // Phase 2, so no inter-lane write race.
    float inv_sum = 1.0 / row_sum;
    for (uint pair = tid; pair < n_pairs; pair += 256u) {
        uint packed_in = output[row_off_u32 + pair];
        float v0 = bf16_bits_to_f32(packed_in & 0xFFFFu) * inv_sum;
        float v1 = bf16_bits_to_f32(packed_in >> 16u) * inv_sum;
        uint out_lo = f32_to_bf16_bits(v0);
        uint out_hi = f32_to_bf16_bits(v1);
        output[row_off_u32 + pair] = out_lo | (out_hi << 16u);
    }
}
