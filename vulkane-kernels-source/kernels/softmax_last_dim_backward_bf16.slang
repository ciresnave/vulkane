// Softmax-last-dim backward — bf16.
//
// Storage: bf16 packed two-per-u32. Pair-thread within Phase 2 (each
// lane processes one u32 pair = 2 bf16 lanes per iter, writes one u32
// per iter). Phase 1 is also pair-thread for the dot reduction.
//
// PRECONDITION: `n_cols % 2 == 0`.

struct Params {
    uint n_rows;
    uint n_cols;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   y_buf;
[[vk::binding(1, 0)]] StructuredBuffer<uint>   g_buf;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> dx_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>   params;

groupshared float wg_dot[32];

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

    float local_dot = 0.0;
    for (uint pair = tid; pair < n_pairs; pair += 256u) {
        uint y_packed = y_buf[row_off_u32 + pair];
        uint g_packed = g_buf[row_off_u32 + pair];
        float y0 = bf16_bits_to_f32(y_packed & 0xFFFFu);
        float y1 = bf16_bits_to_f32(y_packed >> 16u);
        float g0 = bf16_bits_to_f32(g_packed & 0xFFFFu);
        float g1 = bf16_bits_to_f32(g_packed >> 16u);
        local_dot += y0 * g0 + y1 * g1;
    }

    float sg_sum = WaveActiveSum(local_dot);
    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        wg_dot[sg_id] = sg_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    uint n_sg = 256u / WaveGetLaneCount();
    if (sg_id == 0u) {
        float v = (tid < n_sg) ? wg_dot[tid] : 0.0;
        float s = WaveActiveSum(v);
        if (WaveIsFirstLane()) {
            wg_dot[0] = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float dot = wg_dot[0];

    for (uint pair = tid; pair < n_pairs; pair += 256u) {
        uint y_packed = y_buf[row_off_u32 + pair];
        uint g_packed = g_buf[row_off_u32 + pair];
        float y0 = bf16_bits_to_f32(y_packed & 0xFFFFu);
        float y1 = bf16_bits_to_f32(y_packed >> 16u);
        float g0 = bf16_bits_to_f32(g_packed & 0xFFFFu);
        float g1 = bf16_bits_to_f32(g_packed >> 16u);
        float dx0 = y0 * (g0 - dot);
        float dx1 = y1 * (g1 - dot);
        uint out_lo = f32_to_bf16_bits(dx0);
        uint out_hi = f32_to_bf16_bits(dx1);
        dx_buf[row_off_u32 + pair] = out_lo | (out_hi << 16u);
    }
}
