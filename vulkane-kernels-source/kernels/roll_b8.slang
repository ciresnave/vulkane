// Cyclic shift along a single axis, 8-byte elements (f64 / i64).

struct Params {
    uint outer_count;
    uint dim_size;
    uint inner_count;
    int  shift;
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

    int  d = int(p.dim_size);
    int  src_j_signed = (int(j) - p.shift) % d;
    if (src_j_signed < 0) src_j_signed += d;
    uint src_j = uint(src_j_signed);

    uint src_elem = (o * p.dim_size + src_j) * p.inner_count + n;
    uint src_w = src_elem * 2u;
    uint dst_w = tid * 2u;
    output[dst_w]      = input[src_w];
    output[dst_w + 1u] = input[src_w + 1u];
}
