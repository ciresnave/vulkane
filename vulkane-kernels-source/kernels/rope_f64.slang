// Fused rotary position embedding — f64. Native f64 end-to-end.
// Requires shaderFloat64. Mirrors rope.slang exactly except for the
// storage type.

struct Params {
    uint outer;
    uint seq;
    uint head_dim;
    uint total;
    uint x_s0;
    uint x_s1;
    uint x_s_seq;
    uint x_s_hd;
    uint x_outer1;
    uint x_contiguous;
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]] StructuredBuffer<double>   x;
[[vk::binding(1, 0)]] StructuredBuffer<double>   cos_tab;
[[vk::binding(2, 0)]] StructuredBuffer<double>   sin_tab;
[[vk::binding(3, 0)]] RWStructuredBuffer<double> out;
[[vk::binding(4, 0)]] ConstantBuffer<Params>     p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.total) return;

    uint h = p.head_dim / 2;
    uint i  = tid % h;
    uint os = tid / h;
    uint s  = os % p.seq;
    uint o  = os / p.seq;

    uint row_off   = (o * p.seq + s) * p.head_dim;
    uint table_off = s * p.head_dim;

    double x0, x1;
    if (p.x_contiguous == 1) {
        x0 = x[row_off + i];
        x1 = x[row_off + i + h];
    } else {
        uint o1 = o % p.x_outer1;
        uint o0 = o / p.x_outer1;
        uint x_base = o0 * p.x_s0 + o1 * p.x_s1 + s * p.x_s_seq;
        x0 = x[x_base + i * p.x_s_hd];
        x1 = x[x_base + (i + h) * p.x_s_hd];
    }

    double c0 = cos_tab[table_off + i];
    double s0 = sin_tab[table_off + i];
    double c1 = cos_tab[table_off + i + h];
    double s1 = sin_tab[table_off + i + h];

    out[row_off + i]     = x0 * c0 - x1 * s0;
    out[row_off + i + h] = x1 * c1 + x0 * s1;
}
