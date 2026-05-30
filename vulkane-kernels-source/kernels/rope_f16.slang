// Fused rotary position embedding — f16.
//
// Mirrors rope.slang exactly; only the storage type changes.
// Mixed precision: x/cos/sin/out are float16_t; the rotation arithmetic
// is in f32 (matches baracuda's CUDA pattern — multiply-add chains
// with f16 inputs would compound rounding error past 1 ULP of f16).
//
// Requires VK_KHR_16bit_storage + shaderFloat16.

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

[[vk::binding(0, 0)]] StructuredBuffer<float16_t>   x;
[[vk::binding(1, 0)]] StructuredBuffer<float16_t>   cos_tab;
[[vk::binding(2, 0)]] StructuredBuffer<float16_t>   sin_tab;
[[vk::binding(3, 0)]] RWStructuredBuffer<float16_t> out;
[[vk::binding(4, 0)]] ConstantBuffer<Params>        p;

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

    float x0, x1;
    if (p.x_contiguous == 1) {
        x0 = float(x[row_off + i]);
        x1 = float(x[row_off + i + h]);
    } else {
        uint o1 = o % p.x_outer1;
        uint o0 = o / p.x_outer1;
        uint x_base = o0 * p.x_s0 + o1 * p.x_s1 + s * p.x_s_seq;
        x0 = float(x[x_base + i * p.x_s_hd]);
        x1 = float(x[x_base + (i + h) * p.x_s_hd]);
    }

    float c0 = float(cos_tab[table_off + i]);
    float s0 = float(sin_tab[table_off + i]);
    float c1 = float(cos_tab[table_off + i + h]);
    float s1 = float(sin_tab[table_off + i + h]);

    out[row_off + i]     = float16_t(x0 * c0 - x1 * s0);
    out[row_off + i + h] = float16_t(x1 * c1 + x0 * s1);
}
