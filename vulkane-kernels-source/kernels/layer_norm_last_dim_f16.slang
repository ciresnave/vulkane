// LayerNorm last-dim — f16. Mixed precision: f16 storage, f32 mean +
// variance accumulation.

struct Params {
    uint n_rows;
    uint n_cols;
    float eps;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float16_t>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float16_t> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>        params;

groupshared float wg_data[32];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint row_offset = row * params.n_cols;

    float local_sum = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        local_sum += float(input[row_offset + col]);
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

    GroupMemoryBarrierWithGroupSync();
    float local_var = 0.0;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        float d = float(input[row_offset + col]) - mean;
        local_var += d * d;
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

    for (uint col = tid; col < params.n_cols; col += 256u) {
        float v = (float(input[row_offset + col]) - mean) * inv_std;
        output[row_offset + col] = float16_t(v);
    }
}
