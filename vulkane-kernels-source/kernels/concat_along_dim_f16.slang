// Concat along arbitrary dim — f16.
// Pure data movement; mirrors concat_along_dim.slang with float16_t
// storage. Requires VK_KHR_16bit_storage + shaderFloat16.

struct Params {
    uint out_d0, out_d1, out_d2, out_d3;
    uint concat_dim;
    uint a_dim;
    uint b_dim;
    uint total;
    uint a_s0, a_s1, a_s2, a_s3;
    uint b_s0, b_s1, b_s2, b_s3;
};

[[vk::binding(0, 0)]] StructuredBuffer<float16_t>   a_buf;
[[vk::binding(1, 0)]] StructuredBuffer<float16_t>   b_buf;
[[vk::binding(2, 0)]] RWStructuredBuffer<float16_t> dst;
[[vk::binding(3, 0)]] ConstantBuffer<Params>        p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint tid = gid.x;
    if (tid >= p.total) return;

    uint c3 = tid % p.out_d3;
    uint r2 = tid / p.out_d3;
    uint c2 = r2 % p.out_d2;
    uint r1 = r2 / p.out_d2;
    uint c1 = r1 % p.out_d1;
    uint c0 = r1 / p.out_d1;

    uint coord_at_concat;
    if      (p.concat_dim == 0) coord_at_concat = c0;
    else if (p.concat_dim == 1) coord_at_concat = c1;
    else if (p.concat_dim == 2) coord_at_concat = c2;
    else                        coord_at_concat = c3;

    bool from_a = coord_at_concat < p.a_dim;
    uint c_adjusted = from_a ? coord_at_concat : (coord_at_concat - p.a_dim);

    uint k0 = (p.concat_dim == 0) ? c_adjusted : c0;
    uint k1 = (p.concat_dim == 1) ? c_adjusted : c1;
    uint k2 = (p.concat_dim == 2) ? c_adjusted : c2;
    uint k3 = (p.concat_dim == 3) ? c_adjusted : c3;

    float16_t value;
    if (from_a) {
        uint off = k0 * p.a_s0 + k1 * p.a_s1 + k2 * p.a_s2 + k3 * p.a_s3;
        value = a_buf[off];
    } else {
        uint off = k0 * p.b_s0 + k1 * p.b_s1 + k2 * p.b_s2 + k3 * p.b_s3;
        value = b_buf[off];
    }

    dst[tid] = value;
}
