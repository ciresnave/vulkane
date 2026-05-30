// Softmax-last-dim backward — f64. Native f64 end-to-end. Requires
// shaderFloat64 + GroupNonUniformArithmetic.

struct Params {
    uint n_rows;
    uint n_cols;
};

[[vk::binding(0, 0)]] StructuredBuffer<double>   y_buf;
[[vk::binding(1, 0)]] StructuredBuffer<double>   g_buf;
[[vk::binding(2, 0)]] RWStructuredBuffer<double> dx_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>     params;

groupshared double wg_dot[32];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint row_off = row * params.n_cols;

    double local_dot = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        local_dot += y_buf[row_off + col] * g_buf[row_off + col];
    }

    double sg_sum = WaveActiveSum(local_dot);
    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        wg_dot[sg_id] = sg_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    uint n_sg = 256u / WaveGetLaneCount();
    if (sg_id == 0u) {
        double v = (tid < n_sg) ? wg_dot[tid] : 0.0;
        double s = WaveActiveSum(v);
        if (WaveIsFirstLane()) {
            wg_dot[0] = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    double dot = wg_dot[0];

    for (uint col = tid; col < params.n_cols; col += 256u) {
        double yv = y_buf[row_off + col];
        double gv = g_buf[row_off + col];
        dx_buf[row_off + col] = yv * (gv - dot);
    }
}
