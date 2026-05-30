// Softmax-last-dim backward — f16.
//
// Formula:
//   dot   = sum_i(y_i * g_i)
//   dx_j  = y_j * (g_j - dot)
//
// Mixed precision: y/g/dx are float16_t; the dot-product reduction
// and the per-element compute run in f32 (mantissa rounding compounds
// quickly under reductions). Bindings + 8-byte Params match the f32
// version exactly.

struct Params {
    uint n_rows;
    uint n_cols;
};

[[vk::binding(0, 0)]] StructuredBuffer<float16_t>   y_buf;
[[vk::binding(1, 0)]] StructuredBuffer<float16_t>   g_buf;
[[vk::binding(2, 0)]] RWStructuredBuffer<float16_t> dx_buf;
[[vk::binding(3, 0)]] ConstantBuffer<Params>        params;

groupshared float wg_dot[32];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint row_off = row * params.n_cols;

    float local_dot = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        float yv = float(y_buf[row_off + col]);
        float gv = float(g_buf[row_off + col]);
        local_dot += yv * gv;
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

    for (uint col = tid; col < params.n_cols; col += 256u) {
        float yv = float(y_buf[row_off + col]);
        float gv = float(g_buf[row_off + col]);
        dx_buf[row_off + col] = float16_t(yv * (gv - dot));
    }
}
