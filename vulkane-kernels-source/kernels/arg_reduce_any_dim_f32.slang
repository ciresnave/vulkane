// Argmax / Argmin along ARBITRARY dim — f32. Slow path counterpart
// to arg_reduce_last_dim_f32 (which dedicates a workgroup per row
// and uses a tree reduction). For non-last dims a workgroup-per-row
// dispatch is awkward because the row's elements are strided, so
// instead we use one thread per OUTPUT element and let each thread
// scan the `d_dim` values serially. Suitable for small/medium d_dim
// (the common LM-head case still routes through the last-dim kernel).
//
// op_id: 0 = argmax, 1 = argmin.
// Output: u32 indices.
// Tie-breaking: lower index wins on equality (matches numpy / PyTorch).
//
// Logical layout: input shape = [..., d_dim, ...], with n_outer
// elements before `dim` and n_inner elements after `dim`. The output
// drops `dim`, so it has n_outer * n_inner total elements.

struct Params {
    uint n_outer;
    uint n_inner;
    uint d_dim;
    uint op_id;
};

[[vk::binding(0, 0)]] StructuredBuffer<float>  input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   p;

static const float F32_NEG_INF = -3.402823e+38;
static const float F32_POS_INF =  3.402823e+38;

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
        float v = input[base + i * p.n_inner];
        bool better = is_max ? (v > best_v) : (v < best_v);
        if (better) {
            best_v = v;
            best_i = i;
        }
    }

    output[tid] = best_i;
}
