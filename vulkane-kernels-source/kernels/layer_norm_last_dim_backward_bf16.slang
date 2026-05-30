// LayerNorm-last-dim backward — bf16. Packed-u32 storage with lane-
// pair reads in Phase 1 and pair-thread output writes in Phase 3.
// Math in f32; same 4 reductions as f32.
// PRECONDITION: n_cols must be even.

struct Params {
    uint n_rows;
    uint n_cols;
    float eps;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   x_buf;
[[vk::binding(1, 0)]] StructuredBuffer<uint>   g_buf;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> dx_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>   params;

groupshared float wg_x[32];
groupshared float wg_x2[32];
groupshared float wg_g[32];
groupshared float wg_gx[32];

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

    // ---- Phase 1: 4 partial sums (lane-pair reads) ----
    float local_x = 0.0, local_x2 = 0.0;
    float local_g = 0.0, local_gx = 0.0;
    for (uint pair = tid; pair < n_pairs; pair += 256u) {
        uint xw = x_buf[row_off_u32 + pair];
        uint gw = g_buf[row_off_u32 + pair];
        float x0 = bf16_bits_to_f32(xw & 0xFFFFu);
        float x1 = bf16_bits_to_f32(xw >> 16u);
        float g0 = bf16_bits_to_f32(gw & 0xFFFFu);
        float g1 = bf16_bits_to_f32(gw >> 16u);
        local_x  += x0 + x1;
        local_x2 += x0 * x0 + x1 * x1;
        local_g  += g0 + g1;
        local_gx += g0 * x0 + g1 * x1;
    }

    // ---- Phase 2: subgroup-tree + cross-subgroup reduction ----
    float sg_x  = WaveActiveSum(local_x);
    float sg_x2 = WaveActiveSum(local_x2);
    float sg_g  = WaveActiveSum(local_g);
    float sg_gx = WaveActiveSum(local_gx);

    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        wg_x[sg_id]  = sg_x;
        wg_x2[sg_id] = sg_x2;
        wg_g[sg_id]  = sg_g;
        wg_gx[sg_id] = sg_gx;
    }
    GroupMemoryBarrierWithGroupSync();

    uint n_sg = 256u / WaveGetLaneCount();
    if (sg_id == 0u) {
        float vx  = (tid < n_sg) ? wg_x[tid]  : 0.0;
        float vx2 = (tid < n_sg) ? wg_x2[tid] : 0.0;
        float vg  = (tid < n_sg) ? wg_g[tid]  : 0.0;
        float vgx = (tid < n_sg) ? wg_gx[tid] : 0.0;

        float tx  = WaveActiveSum(vx);
        float tx2 = WaveActiveSum(vx2);
        float tg  = WaveActiveSum(vg);
        float tgx = WaveActiveSum(vgx);

        if (WaveIsFirstLane()) {
            wg_x[0]  = tx;
            wg_x2[0] = tx2;
            wg_g[0]  = tg;
            wg_gx[0] = tgx;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    float sum_x  = wg_x[0];
    float sum_x2 = wg_x2[0];
    float sum_g  = wg_g[0];
    float sum_gx = wg_gx[0];

    float n = float(params.n_cols);
    float mu = sum_x / n;
    float var = sum_x2 / n - mu * mu;
    float rstd = 1.0 / sqrt(var + params.eps);
    float mean_g = sum_g / n;
    float mean_gxc = (sum_gx - sum_g * mu) / n;
    float rstd2 = rstd * rstd;

    // ---- Phase 3: per-pair dx writes ----
    for (uint pair = tid; pair < n_pairs; pair += 256u) {
        uint xw = x_buf[row_off_u32 + pair];
        uint gw = g_buf[row_off_u32 + pair];
        float x0 = bf16_bits_to_f32(xw & 0xFFFFu);
        float x1 = bf16_bits_to_f32(xw >> 16u);
        float g0 = bf16_bits_to_f32(gw & 0xFFFFu);
        float g1 = bf16_bits_to_f32(gw >> 16u);
        float xc0 = x0 - mu, xc1 = x1 - mu;
        float dx0 = rstd * (g0 - mean_g - xc0 * rstd2 * mean_gxc);
        float dx1 = rstd * (g1 - mean_g - xc1 * rstd2 * mean_gxc);
        uint out_lo = f32_to_bf16_bits(dx0);
        uint out_hi = f32_to_bf16_bits(dx1);
        dx_buf[row_off_u32 + pair] = out_lo | (out_hi << 16u);
    }
}
