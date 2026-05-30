// LayerNorm-last-dim backward — f16. Mixed precision: f16 storage,
// f32 accumulation in the 4 reductions + per-element compute.

struct Params {
    uint n_rows;
    uint n_cols;
    float eps;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float16_t>   x_buf;
[[vk::binding(1, 0)]] StructuredBuffer<float16_t>   g_buf;
[[vk::binding(2, 0)]] RWStructuredBuffer<float16_t> dx_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>        params;

groupshared float wg_x[32];
groupshared float wg_x2[32];
groupshared float wg_g[32];
groupshared float wg_gx[32];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint row_off = row * params.n_cols;

    float local_x = 0.0, local_x2 = 0.0;
    float local_g = 0.0, local_gx = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        float xi = float(x_buf[row_off + col]);
        float gi = float(g_buf[row_off + col]);
        local_x  += xi;
        local_x2 += xi * xi;
        local_g  += gi;
        local_gx += gi * xi;
    }

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
    for (uint col = tid; col < params.n_cols; col += 256u) {
        float xi = float(x_buf[row_off + col]);
        float gi = float(g_buf[row_off + col]);
        float xc = xi - mu;
        float dx = rstd * (gi - mean_g - xc * rstd2 * mean_gxc);
        dx_buf[row_off + col] = float16_t(dx);
    }
}
