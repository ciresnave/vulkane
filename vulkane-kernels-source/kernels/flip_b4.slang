// Flip along a single axis. View input/output as a flat 3-tuple
// (outer, dim_size, inner) — the byte kernel takes care of any
// higher-dimensional reshape upstream.
//
// Rule: out[o, j, n] = in[o, dim_size - 1 - j, n]
//
// 4-byte element variant (f32 / i32 / u32). Dtype-agnostic — we
// memcpy entire u32 words.

struct Params {
    uint outer_count;
    uint dim_size;
    uint inner_count;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>   p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    uint plane = p.dim_size * p.inner_count;
    uint total = p.outer_count * plane;
    if (tid >= total) return;

    uint o = tid / plane;
    uint rest = tid % plane;
    uint j = rest / p.inner_count;
    uint n = rest % p.inner_count;

    uint src_j = (p.dim_size - 1u) - j;
    uint src_idx = (o * p.dim_size + src_j) * p.inner_count + n;
    output[tid] = input[src_idx];
}
