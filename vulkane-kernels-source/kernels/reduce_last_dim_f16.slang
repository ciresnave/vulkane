// Per-row reduction along the last dimension — f16.
//
// Input:  [n_rows, n_cols]  contiguous, row-major, float16_t.
// Output: [n_rows]          one float16_t per row.
//
// op_id: 0 = sum, 1 = max, 2 = min, 3 = mean (sum / n_cols).
//
// Mixed precision: storage is f16; the per-thread accumulator and the
// subgroup/cross-subgroup reductions are all f32. f16's 10-bit
// mantissa loses precision under long-row sums (and matches the same
// rule baracuda follows on CUDA).
//
// Requires VK_KHR_16bit_storage + shaderFloat16.
//
// Binding layout matches the f32 version (Params is 16 bytes:
// `{ u32 n_rows, u32 n_cols, u32 op_id, u32 _pad }`).

struct Params {
    uint n_rows;
    uint n_cols;
    uint op_id;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float16_t>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float16_t> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>        params;

groupshared float wg_data[32];

static const float F32_NEG_INF = -3.402823e+38;
static const float F32_POS_INF =  3.402823e+38;

float identity_of(uint op) {
    if (op == 0u) return 0.0;
    if (op == 1u) return F32_NEG_INF;
    if (op == 2u) return F32_POS_INF;
    return 0.0;                         // mean
}

float combine(uint op, float a, float b) {
    if (op == 0u) return a + b;
    if (op == 1u) return max(a, b);
    if (op == 2u) return min(a, b);
    return a + b;                       // mean accumulates as sum
}

float subgroup_reduce(uint op, float x) {
    if (op == 0u) return WaveActiveSum(x);
    if (op == 1u) return WaveActiveMax(x);
    if (op == 2u) return WaveActiveMin(x);
    return WaveActiveSum(x);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint row_off = row * params.n_cols;
    uint op = params.op_id;

    float acc = identity_of(op);
    for (uint col = tid; col < params.n_cols; col += 256u) {
        acc = combine(op, acc, float(input[row_off + col]));
    }

    float sg = subgroup_reduce(op, acc);
    uint sg_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        wg_data[sg_id] = sg;
    }
    GroupMemoryBarrierWithGroupSync();

    uint n_sg = 256u / WaveGetLaneCount();
    if (sg_id == 0u) {
        float v = (tid < n_sg) ? wg_data[tid] : identity_of(op);
        float r = subgroup_reduce(op, v);
        if (WaveIsFirstLane()) {
            if (op == 3u) {
                r = r / float(params.n_cols);
            }
            output[row] = float16_t(r);
        }
    }
}
