// Argmax / Argmin along arbitrary dim — f64. See arg_reduce_any_dim_f32
// for the dispatch shape; differs only in the buffer element type
// and the f64 identity sentinels.

struct Params {
    uint n_outer;
    uint n_inner;
    uint d_dim;
    uint op_id;
};

[[vk::binding(0, 0)]] StructuredBuffer<double> input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   p;

static const double F64_NEG_INF = -1.7976931348623157e+308;
static const double F64_POS_INF =  1.7976931348623157e+308;

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
    double best_v = is_max ? F64_NEG_INF : F64_POS_INF;
    uint   best_i = 0u;

    for (uint i = 0u; i < p.d_dim; i++) {
        double v = input[base + i * p.n_inner];
        bool better = is_max ? (v > best_v) : (v < best_v);
        if (better) {
            best_v = v;
            best_i = i;
        }
    }

    output[tid] = best_i;
}
