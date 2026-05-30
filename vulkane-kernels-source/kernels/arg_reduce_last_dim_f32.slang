// Argmax / Argmin along the last dimension — f32. Per-row workgroup,
// tree reduction in shared memory tracking (val, idx) pairs.
//
// op_id: 0 = argmax, 1 = argmin.
// Output: u32 indices (one per row).
//
// Tie-breaking: the kernel keeps the LOWER index on equality
// (matches numpy / PyTorch default).

struct Params {
    uint n_rows;
    uint n_cols;
    uint op_id;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>  output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>    params;

groupshared float wg_val[256];
groupshared uint  wg_idx[256];

static const float F32_NEG_INF = -3.402823e+38;
static const float F32_POS_INF =  3.402823e+38;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint row_off = row * params.n_cols;
    bool is_max = (params.op_id == 0u);
    float identity_v = is_max ? F32_NEG_INF : F32_POS_INF;

    // Phase 1: per-thread local best over a strided slice.
    float best_v = identity_v;
    uint  best_i = 0u;
    for (uint col = tid; col < params.n_cols; col += 256u) {
        float v = input[row_off + col];
        bool better = is_max ? (v > best_v) : (v < best_v);
        // On ties, keep the lower index (already so: we initialize
        // with col=tid and only replace on strict better).
        if (better) {
            best_v = v;
            best_i = col;
        }
    }
    wg_val[tid] = best_v;
    wg_idx[tid] = best_i;
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: tree reduction in shared memory. On equality keep the
    // LOWER index by checking `<` rather than `<=` / `>`.
    uint stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            uint other = tid + stride;
            float ov = wg_val[other];
            uint  oi = wg_idx[other];
            bool take_other = is_max
                ? (ov > wg_val[tid] || (ov == wg_val[tid] && oi < wg_idx[tid]))
                : (ov < wg_val[tid] || (ov == wg_val[tid] && oi < wg_idx[tid]));
            if (take_other) {
                wg_val[tid] = ov;
                wg_idx[tid] = oi;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        stride = stride >> 1u;
    }

    if (tid == 0u) {
        output[row] = wg_idx[0];
    }
}
