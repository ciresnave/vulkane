// Argmax / Argmin along the last dimension — bf16. Lane-pair reads
// from packed-u32 input; f32 reductions in shared memory. Tracks
// the GLOBAL bf16-element index (not the pair index) so the output
// matches the unpacked convention.
//
// PRECONDITION: n_cols must be even (lane-pair).

struct Params {
    uint n_rows;
    uint n_cols;
    uint op_id;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>    input;   // packed bf16
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>  output;  // u32 indices
[[vk::binding(2, 0)]] ConstantBuffer<Params>    params;

groupshared float wg_val[256];
groupshared uint  wg_idx[256];

static const float F32_NEG_INF = -3.402823e+38;
static const float F32_POS_INF =  3.402823e+38;

float bf16_bits_to_f32(uint b) { return asfloat(b << 16u); }

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint row = gid.x;
    if (row >= params.n_rows) return;

    uint tid = gtid.x;
    uint n_pairs = params.n_cols >> 1u;
    uint row_off_u32 = row * n_pairs;
    bool is_max = (params.op_id == 0u);
    float identity_v = is_max ? F32_NEG_INF : F32_POS_INF;

    float best_v = identity_v;
    uint  best_i = 0u;
    // Each lane reads its strided u32 pair, considers both halves.
    for (uint pair = tid; pair < n_pairs; pair += 256u) {
        uint packed = input[row_off_u32 + pair];
        float v0 = bf16_bits_to_f32(packed & 0xFFFFu);
        float v1 = bf16_bits_to_f32(packed >> 16u);
        uint i0 = pair * 2u;
        uint i1 = i0 + 1u;
        if (is_max) {
            if (v0 > best_v) { best_v = v0; best_i = i0; }
            if (v1 > best_v) { best_v = v1; best_i = i1; }
        } else {
            if (v0 < best_v) { best_v = v0; best_i = i0; }
            if (v1 < best_v) { best_v = v1; best_i = i1; }
        }
    }
    wg_val[tid] = best_v;
    wg_idx[tid] = best_i;
    GroupMemoryBarrierWithGroupSync();

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
