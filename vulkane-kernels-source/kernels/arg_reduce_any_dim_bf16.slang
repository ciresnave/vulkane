// Argmax / Argmin along arbitrary dim — bf16. Counterpart to
// arg_reduce_any_dim_f32; reads packed-u32 input with manual lane
// select since the reduction direction is strided (stride > 1
// means we cannot use a lane-pair scan).

struct Params {
    uint n_outer;
    uint n_inner;
    uint d_dim;
    uint op_id;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;   // packed bf16
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   p;

static const float F32_NEG_INF = -3.402823e+38;
static const float F32_POS_INF =  3.402823e+38;

float bf16_bits_to_f32(uint b) { return asfloat(b << 16u); }

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    uint total = p.n_outer * p.n_inner;
    if (tid >= total) return;

    uint outer = tid / p.n_inner;
    uint inner = tid % p.n_inner;
    uint base = outer * (p.d_dim * p.n_inner) + inner;

    bool is_max = (p.op_id == 0u);
    float best_v = is_max ? F32_NEG_INF : F32_POS_INF;
    uint best_i = 0u;

    for (uint i = 0u; i < p.d_dim; i++) {
        uint idx = base + i * p.n_inner;
        uint word_idx = idx >> 1u;
        uint lane = idx & 1u;
        uint packed = input[word_idx];
        uint bits = (lane == 0u) ? (packed & 0xFFFFu) : (packed >> 16u);
        float v = bf16_bits_to_f32(bits);
        bool better = is_max ? (v > best_v) : (v < best_v);
        if (better) {
            best_v = v;
            best_i = i;
        }
    }

    output[tid] = best_i;
}
