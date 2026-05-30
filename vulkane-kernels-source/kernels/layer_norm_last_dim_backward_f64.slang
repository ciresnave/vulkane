// LayerNorm-last-dim backward — f64. Native f64 end-to-end.

struct Params {
    uint   n_rows;
    uint   n_cols;
    double eps;
};

[[vk::binding(0, 0)]] StructuredBuffer<double>   x_buf;
[[vk::binding(1, 0)]] StructuredBuffer<double>   g_buf;
[[vk::binding(2, 0)]] RWStructuredBuffer<double> dx_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>     params;

groupshared double wg_x[32];
groupshared double wg_x2[32];
groupshared double wg_g[32];
groupshared double wg_gx[32];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint row_off = row * params.n_cols;

    double local_x = 0.0, local_x2 = 0.0;
    double local_g = 0.0, local_gx = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        double xi = x_buf[row_off + col];
        double gi = g_buf[row_off + col];
        local_x  += xi;
        local_x2 += xi * xi;
        local_g  += gi;
        local_gx += gi * xi;
    }

    double sg_x  = WaveActiveSum(local_x);
    double sg_x2 = WaveActiveSum(local_x2);
    double sg_g  = WaveActiveSum(local_g);
    double sg_gx = WaveActiveSum(local_gx);

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
        double vx  = (tid < n_sg) ? wg_x[tid]  : 0.0;
        double vx2 = (tid < n_sg) ? wg_x2[tid] : 0.0;
        double vg  = (tid < n_sg) ? wg_g[tid]  : 0.0;
        double vgx = (tid < n_sg) ? wg_gx[tid] : 0.0;

        double tx  = WaveActiveSum(vx);
        double tx2 = WaveActiveSum(vx2);
        double tg  = WaveActiveSum(vg);
        double tgx = WaveActiveSum(vgx);

        if (WaveIsFirstLane()) {
            wg_x[0]  = tx;
            wg_x2[0] = tx2;
            wg_g[0]  = tg;
            wg_gx[0] = tgx;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    double sum_x  = wg_x[0];
    double sum_x2 = wg_x2[0];
    double sum_g  = wg_g[0];
    double sum_gx = wg_gx[0];

    double n = double(params.n_cols);
    double mu = sum_x / n;
    double var = sum_x2 / n - mu * mu;
    double rstd = 1.0 / sqrt(var + params.eps);
    double mean_g = sum_g / n;
    double mean_gxc = (sum_gx - sum_g * mu) / n;

    double rstd2 = rstd * rstd;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        double xi = x_buf[row_off + col];
        double gi = g_buf[row_off + col];
        double xc = xi - mu;
        dx_buf[row_off + col] = rstd * (gi - mean_g - xc * rstd2 * mean_gxc);
    }
}
