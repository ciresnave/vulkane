// Per-row reduction along the last dimension.
//
// Input: [n_rows, n_cols]  contiguous, row-major.
// Output: [n_rows]         one element per row.
//
// op_id: 0 = sum, 1 = max, 2 = min.
//
// Slang port of the WGSL reducer. Uses subgroup arithmetic
// (`WaveActiveSum`, `WaveActiveMax`, `WaveActiveMin`) instead of a
// 256-element tree reduction through shared memory. Binding layout
// matches the WGSL version so it's a drop-in swap.

struct Params {
    uint n_rows;
    uint n_cols;
    uint op_id;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>    params;

// One slot per subgroup for the cross-subgroup fixup. 32 covers
// every desktop GPU at workgroup size 256.
groupshared float wg_data[32];

static const float F32_NEG_INF = -3.402823e+38;
static const float F32_POS_INF =  3.402823e+38;

// Identity value per op.
float identity_of(uint op) {
    if (op == 0u) return 0.0;
    if (op == 1u) return F32_NEG_INF;
    return F32_POS_INF;
}

float combine(uint op, float a, float b) {
    if (op == 0u) return a + b;
    if (op == 1u) return max(a, b);
    return min(a, b);
}

// Subgroup reduction dispatched by op_id. Slang's subgroup intrinsics
// don't take a runtime op selector so we branch once per call.
float subgroup_reduce(uint op, float x) {
    if (op == 0u) return WaveActiveSum(x);
    if (op == 1u) return WaveActiveMax(x);
    return WaveActiveMin(x);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint row_off = row * params.n_cols;
    uint op = params.op_id;

    // Phase 1 — per-thread local reduction over strided slice.
    float acc = identity_of(op);
    for (uint col = tid; col < params.n_cols; col += 256u) {
        acc = combine(op, acc, input[row_off + col]);
    }

    // Phase 2 — reduce within subgroup, then across subgroups.
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
            output[row] = r;
        }
    }
}
