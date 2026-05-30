// Affine: y = x * mul + add. f16 variant of affine.slang. Stride-aware.
// Math at f32 (matches the half-precision convention; the precision
// guarantee on the dispatch side already reflects this).

struct Params {
    uint  out_size;
    uint  flags;           // bit 0: input_contiguous
    float mul;
    float add;
    uint  shape0, shape1, shape2, shape3;
    uint  in_s0, in_s1, in_s2, in_s3;
};

[[vk::binding(0, 0)]] StructuredBuffer<float16_t>   input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float16_t> output;
[[vk::binding(2, 0)]] ConstantBuffer<Params>        p;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint idx = gid.x;
    if (idx >= p.out_size) return;

    float16_t x;
    bool in_contig = (p.flags & 1u) != 0u;

    if (in_contig) {
        x = input[idx];
    } else {
        uint c3 = idx % p.shape3;
        uint r2 = idx / p.shape3;
        uint c2 = r2 % p.shape2;
        uint r1 = r2 / p.shape2;
        uint c1 = r1 % p.shape1;
        uint c0 = r1 / p.shape1;
        uint in_off = c0 * p.in_s0 + c1 * p.in_s1 + c2 * p.in_s2 + c3 * p.in_s3;
        x = input[in_off];
    }

    float y = float(x) * p.mul + p.add;
    output[idx] = float16_t(y);
}
